"""Ringing cancellation: measure, model, and correct luma artifacts using
the horizontal sync interval.

This module implements the pipeline specified in RINGING_INSTRUCTIONS.md,
plus the luma transient improvement (LTI) stage the correction feeds
(the section at the end of the file; field.py runs it under --lti_gain).

The horizontal sync interval is the one stretch of every video line whose
ideal shape is known from the video standard, so it serves as the reference
for measuring what the signal chain did to the signal.  It is read as seven
measurement points in time order, with state carried from each point into
the next:

    1. active area           picture content; supplies the entry amplitude
    2. active falling edge   active area -> front porch transition
    3. front porch           flat at blanking level
    4. sync falling edge     blanking -> sync tip, timing reference
    5. sync tip              flat at sync tip level
    6. sync rising edge      sync tip -> blanking
    7. back porch            flat at blanking level, the ENTIRE stretch
                             from the sync rising edge to the start of
                             active video (the color burst is removed from
                             luma by the color-under process, so the whole
                             span is measurable)

Three artifact types are modeled, in the order the signal chain applies
them (ghost first, before the recorder; smear and ringing in the recorder
and capture chain):

    GHOST     a delayed, scaled, non-oscillating copy of the signal -
              fitted so it never pollutes the smear and ring
              measurements, reported, and left in the output (it may be
              an attribute of the source recording)
    RINGING   decaying resonant oscillations after each transient,
              amplitude proportional to the transient, fixed
              frequencies; zero or more components (complex tracked
              poles)
    SMEAR     non-oscillating decays after each transient - under- or
              overshoot with a single decay; real tracked poles,
              realized as one-pole relaxation shelves whose exact
              inverse puts the smeared energy back into the transition

Smear and ringing are corrected by the fixed-point exact inverse of the
fitted channel, at a strength governed by measured no-harm evidence.
The VHSHQ record-side detail enhancer (a symmetric factor) is measured
on every tape and inverted only where the format flag says the recorder
carried it.

Standing constraints (see RINGING_RULES.md): calibration uses the sync
interval only; the correction is strictly causal; every quantity is derived
from supplied system parameters or measured from the signal, never
hard-coded; the correction must be temporally stable and reset itself when
the recording changes.

Everything is measured in IRE (blanking = 0, sync tip = vsync_ire).  The
supplied reference levels are used only for that unit conversion; every
decision inside the module uses levels measured from the signal itself.
"""

import dataclasses
import json
import math
import os

import numba as nb
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.special

# ---- luma phase-profile sidecar (hsEE7) --------------------------------
# The user's ruling: "incorporate the luma phase profile into the ringing
# correction."  The sibling measurement arc's transient-aligned
# instantaneous-frequency step (demod_raw, pre-deemphasis), pushed through
# the decoder's own deemphasis and video LPF, predicts this witness's
# rise aftermath first-principles (r=+0.88 at x0.90 amplitude on the
# countdown tape, per head).  The sidecar carries that predicted
# aftermath per head and edge, WITH the certification each kernel earned
# against the sync witness (lambda = clip(r,0,1)^2) and its amplitude
# ratio rho; the sync witness stays the arbiter - a kernel enters only
# at its certified weight, and the applied scale min(lambda/rho, 1)
# honors the measured amount-asymmetry law (over-correction destroys;
# err low).  Configured by a readable JSON file next to this module.
_PHASE_PROFILE_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ringing_phase_profile.json")
_phase_profile_state = {"mtime": None, "data": None}


def _load_phase_profile():
    """The configured phase-profile sidecar, or None (config absent,
    disabled, or unreadable).  Reloaded when the config file changes."""
    try:
        mtime = os.path.getmtime(_PHASE_PROFILE_CONFIG)
    except OSError:
        return None
    if _phase_profile_state["mtime"] != mtime:
        _phase_profile_state["mtime"] = mtime
        _phase_profile_state["data"] = None
        try:
            with open(_PHASE_PROFILE_CONFIG) as handle:
                config = json.load(handle)
            if config.get("enabled"):
                with np.load(config["profile"]) as archive:
                    _phase_profile_state["data"] = {
                        key: np.array(archive[key])
                        for key in archive.files}
                _phase_profile_state["data"][
                    "landing_bank_enabled"] = bool(
                    config.get("landing_bank", True))
                # default: the synthesized blend is muted while channel_eq
                # carries the RF physics
                _phase_profile_state["data"][
                    "synth_blend_under_channel_eq"] = bool(
                    config.get("synth_blend_under_channel_eq", False))
        except (OSError, ValueError, KeyError):
            _phase_profile_state["data"] = None
    return _phase_profile_state["data"]


# Name of this module's sub-dictionary inside the shared per-decode state
# (rf.field_averages.group_delay), so its lifetime and reset semantics stay
# owned by the existing state store.
STATE_KEY = "hsync_model"

# The number of reciprocal luma-bandwidth periods a band-limited transition
# needs before the flat region after it is trustworthy.  A first-order view
# of the decoder's luma low-pass settles to a few percent within three time
# constants; the decoder's actual filter is higher order (steeper skirt,
# similar settling), so three reciprocal bandwidths is the working guard.
TRANSITION_SETTLE_BANDWIDTHS = 3.0

# The vertical sync interval is three sections (pre-equalizing pulses,
# serrated vertical sync, post-equalizing pulses), each numPulses pulses
# long, at half-line rate: 3 * numPulses / 2 lines.  Lines before that
# carry equalizing or serration pulses instead of normal horizontal sync
# and are excluded from measurement.
VSYNC_SECTIONS = 3

# Field-level health thresholds, as fractions of the sync depth.  The two
# sync edges must each measure close to one sync depth, and the two
# blanking readings (front porch, back porch) must agree.  The margins are
# deliberately generous: the decode input is not bit-reproducible under
# threading, and a health verdict that flips on a marginal field re-seeds
# the averages one field late and shifts the early output (measured on the
# home tape as a ~0.2 IRE transient over twelve fields).
HEALTH_STEP_TOLERANCE = 0.25
HEALTH_LEVEL_AGREEMENT = 0.25

# A field must contribute at least this many measurable lines before its
# statistics are trusted; below it a single noisy stretch could steer the
# average.  One tenth of a field's normal line count.
MINIMUM_USABLE_LINES = 20

# The averages re-bootstrap after this many consecutive unhealthy fields
# (in units of the averaging horizon).  A recording change, a gap, or an
# unrecorded stretch must unlock the state - a stale lock lasts forever -
# but a single bad field must not throw the model away.
RESET_HORIZONS = 2

# Entry-amplitude strata for the active falling edge witness, in fractions
# of the sync depth per stratum.  A quarter of the calibrated reference
# depth balances amplitude resolution against per-stratum population.
# The strata exist for MEASUREMENT AND REPORTING (the linearity verdict);
# no correction is derived from them in this stage.
STRATUM_DEPTH_FRACTION = 0.25

# A line only enters a fall stratum when its active area sits measurably
# above blanking - half a stratum, so classification noise cannot flip a
# genuinely flat line into the first stratum.
STRATUM_MINIMUM_FRACTION = 0.125

# The artifact fit consumes a SLOWER average than the field-tracking one:
# fitting resonances on a four-field average was measured (in the previous
# pipeline) to flip pole classifications field to field; a thirty-two
# field horizon cured it.  Eight averaging horizons reproduces that at the
# default setting.
SLOW_HORIZON_FACTOR = 8

# Damped-mode estimation capacity FLOOR.  The working capacity is derived
# from the witnessed region itself (_pencil_capacity): the estimator must
# be able to hold every distinct component the full sync interval can
# witness - a fixed cap of eight was measured to fuse near-lying ring
# components and starve a still-visible family - and this floor only
# protects degenerate (very short) window layouts.
MODE_CAPACITY = 8

# Relaxation (smear) slots in the tracked model.  The requirements
# document allows one smearing artifact comprising multiple components
# with their own characteristics.  Two slots is the resolvable count:
# a relaxation is witnessable between the edge settle time and half the
# longest flat window (the witnessable-settle bound below), a range of
# about one decade of time constants, and two relaxations closer than a
# factor of ~3 in settle time are numerically collinear over such a
# window (their design columns merge), so a third slot could only ever
# split one physical decay in two.
SMEAR_COMPONENT_SLOTS = 2

# A mode is RINGING when it completes at least one FULL oscillation
# cycle within its own decay time - rotation angle x decay time >= 2 pi,
# equivalently quality factor Q >= pi (cycles to 1/e = Q / pi >= 1).
# Anything slower dies before completing a cycle: the eye sees a smear,
# not an echo, and cancelling it as a ring is broadband softening
# (measured: a 0.62 MHz Q 1.1 hump certified as a ring blurred the
# picture; as a smear component it is corrected without detail loss).
RINGING_MINIMUM_ROTATION = 2.0 * math.pi

# Tracked poles are matched field-to-field within one DFT resolution cell
# of the longest residual window; a pole that moves further is a
# different mode, not a drifted one.
TRACK_MATCH_RESOLUTION_CELLS = 1.0


def _resolvability_radius(pole_a, pole_b, match_radius):
    """One tracked slot per RESOLVABLE mode (hsEE2).

    An oscillatory mode decaying in tau samples has a Lorentzian line
    of full half-width 2/tau in pole angle (1/(pi tau) in cycles - the
    same linewidth the persistent floor's guard uses); when two ring
    slots sit inside either one's line, no residual window can tell
    them apart, and letting both live was measured (the countdown
    tape) to make the per-update solves trade one STATIONARY 1.8 MHz
    ring between in-linewidth siblings - the shipped numerator rotated
    through more than 180 degrees across the decode, the corrections
    time-averaged against themselves, and 78% of the ring survived.
    The DFT resolution cell stays as the floor.  The criterion applies
    to the ring families only: a relaxation's line is centered at DC
    for every decay, so line-center separation cannot resolve real
    poles at all (the criterion would degenerately fold every smear),
    and their identity axis - the decay itself - showed no churn.
    """
    radius = match_radius
    for pole in (pole_a, pole_b):
        if abs(pole.imag) > 0.0:
            magnitude = min(abs(pole), 1.0 - 1e-9)
            radius = max(radius, -2.0 * math.log(magnitude))
    return radius

# Ridge conditioning for the model's least-squares solves, as a fraction
# of the normal matrix's mean diagonal - pure conditioning, small enough
# never to bias a well-posed fit.
PENCIL_CONDITION_RIDGE = 1e-3

# The amplitude dependence: each resonator's numerator changes linearly
# with the transition amplitude, fitted from the fall-aligned strata and
# anchored at the sync calibration (the change is zero at the sync
# depth).  Applied through a NO-HARM scale: the largest fraction of the
# fitted dependence at which no stratum's closure worsens beyond its own
# noise.  Measured: the countdown tape's strata all close better at full
# scale (its porch tails halve), while the home tape's picture-fall
# aftermath differs from its sync response in SHAPE - no amplitude
# scaling of the sync basis can represent it, every parameterization
# tried (linear slope, per-section power) harmed its light strata, and
# the no-harm scale is what keeps that harm out while the countdown
# tape keeps its correction.  The amplitude input is clamped to the
# range the strata witnessed - outside it nothing extrapolates.

# The no-harm bound's tolerance: coherent structure below this level was
# repeatedly invisible in the earlier pipeline's eye tests against these
# tapes' 0.75+ IRE line noise, and structure above it was user-visible
# (the bound the earlier residual-stage magnitude guard used).  A
# stratum may not be worsened beyond this, in quadrature with its own
# statistical noise.
VISIBILITY_FLOOR_IRE = 0.3

# The corrector's artifact tail must stay below unity magnitude on the
# whole frequency axis for the fixed-point inversion 1/(1 + R) to be causal
# and stable with a margin; above this bound the fit is scaled down
# uniformly.  0.5 bounds the corrector's gain at 2x (6 dB), which also
# bounds its noise amplification.
ARTIFACT_MAGNITUDE_LIMIT = 0.5


@dataclasses.dataclass(frozen=True)
class HsyncGeometry:
    """Where each measurement point of the horizontal sync interval lies.

    Every length is derived from the video standard's timing (SysParams)
    and the decoder's own configuration; positions are in output samples,
    kept as decimals (the video standard's durations are not integer
    numbers of samples), measured from the start of the line.  The line
    starts at the midpoint of the sync falling edge, which is the timing
    reference the time base corrector aligns to.
    """

    # sampling
    sample_rate_mhz: float          # output sample rate (4 x fsc), exact
    samples_per_line: int           # output samples per video line

    # measurement point boundaries, in decimal samples from line start
    front_porch_samples: float      # duration of the front porch
    sync_pulse_samples: float       # sync fall midpoint -> rise midpoint
    sync_transition_samples: float  # 10-90% duration of one sync edge
    active_video_start_sample: float

    # levels, from the video standard
    sync_depth_ire: float           # blanking minus sync tip, in IRE

    # the decoder's own luminance low-pass corner: nothing above it can
    # be a transient-driven luma artifact (structure there is chroma /
    # burst remnant - uncorrelated with the known input, excluded by
    # the instructions' gathering rule)
    luma_lowpass_mhz: float

    # measurement guards and line selection
    transition_settle_samples: int  # samples a band-limited transition
                                    # needs to finish settling
    first_measurable_line: int      # first buffer row carrying normal
                                    # horizontal sync (after vertical sync)
    last_measurable_line: int       # one past the last usable buffer row

    # format capabilities
    detail_emphasis: bool           # this format's recorders carry the
                                    # record-side delay-line detail
                                    # enhancer (the VHSHQ format flag):
                                    # the measured symmetric factor is
                                    # then inverted, not only modeled


@dataclasses.dataclass(frozen=True)
class MeasurementWindow:
    """The per-line measurement window and the landmarks inside it.

    The window is anchored at the sync falling edge midpoint (index
    `sync_fall_index`), reaches back across the front porch into the end
    of the previous line's active area, and forward across the sync pulse
    and back porch to just before active video begins.  All slices are
    guarded so a band-limited transition has settled before a flat region
    is read.
    """

    samples_before_sync_fall: int
    samples_after_sync_fall: int
    sync_fall_index: int            # == samples_before_sync_fall
    total_samples: int

    # flat-region slices, as (start, stop) pairs relative to window start
    entry_context: tuple            # previous line's active area, for the
                                    # entry amplitude
    front_porch: tuple
    sync_tip: tuple
    back_porch: tuple               # sync rise to the start of active
                                    # video; the color burst is filtered
                                    # out of luma, so the whole span is
                                    # one flat region
    upcoming_context: tuple         # the start of THIS line's active
                                    # area, read only to know whether
                                    # picture content follows the back
                                    # porch (see the contamination trim)

    level_guard_samples: int        # isolation for LEVEL reads: only the
                                    # transition core must be excluded (a
                                    # median tolerates a decaying tail)
    crossing_search_radius: int     # how far from nominal the sync fall
                                    # midpoint may land


def build_geometry(sys_params, decoder_params, samples_per_line,
                   line_offset, line_count, include_vertical_interval=False):
    """Derive the sync-interval geometry from the supplied system parameters.

    `sys_params` is the video standard's timing table (SysParams), and
    `decoder_params` the decoder's configuration (DecoderParams); together
    they are the only sources of every number here.

    `include_vertical_interval` admits the equalizing and serration lines
    that are normally excluded. It exists for ONE purpose: a LOW-FREQUENCY
    response measurement. A horizontal sync pulse is 4.7 microseconds and
    reaches no lower than a few hundred kilohertz; the broad vertical sync
    pulses are tens of times longer and are the only long pulses the
    signal contains, so a low-frequency measurement that excludes them has
    nothing to measure with. It must stay FALSE everywhere else - those
    lines carry pulses of a different width and at twice line rate, and
    folding them into an average of horizontal sync pulses would corrupt
    it. Default False, so every existing path is unchanged.
    """
    sample_rate_mhz = float(sys_params["outfreq"])

    microseconds_to_samples = sample_rate_mhz  # 1 us at rate megahertz

    luma_lowpass_hz = float(decoder_params["video_lpf_freq"])
    transition_settle_samples = int(math.ceil(
        TRANSITION_SETTLE_BANDWIDTHS * sample_rate_mhz * 1e6 / luma_lowpass_hz))

    lines_of_vertical_sync = int(math.ceil(
        VSYNC_SECTIONS * sys_params["numPulses"] / 2.0))

    active_start_us = sys_params["activeVideoUS"][0]

    return HsyncGeometry(
        sample_rate_mhz=sample_rate_mhz,
        samples_per_line=int(samples_per_line),
        front_porch_samples=(
            sys_params["frontPorchUS"] * microseconds_to_samples),
        sync_pulse_samples=(
            sys_params["hsyncPulseUS"] * microseconds_to_samples),
        sync_transition_samples=(
            sys_params["syncTransitionUS"] * microseconds_to_samples),
        active_video_start_sample=active_start_us * microseconds_to_samples,
        sync_depth_ire=abs(float(sys_params["vsync_ire"])),
        luma_lowpass_mhz=luma_lowpass_hz / 1e6,
        transition_settle_samples=transition_settle_samples,
        first_measurable_line=(line_offset if include_vertical_interval
                               else line_offset + lines_of_vertical_sync),
        # the population is defined by the specification, not derived
        # from waveforms (the user's rule): every measured line must
        # carry a genuine horizontal sync pulse.  Interlaced fields are
        # an odd multiple of half a line long, so the buffer's FINAL
        # row spans the field boundary and carries the next field's
        # half-line-rate (equalizing) pulse - its tip is half the
        # horizontal pulse width by specification, and accumulating it
        # mixed blanking level into every tip lag past the half-width
        # point (measured: a 4-5x cross-line variance plateau from
        # 2.3 us onward that down-weighted the tip's back half and
        # starved the low-frequency ring components).
        last_measurable_line=line_count + line_offset - 1,
        detail_emphasis=bool(sys_params.get("detail_emphasis", False)),
    )


def plan_measurement_window(geometry):
    """Lay out the per-line window and its flat-region slices.

    Each flat region is read only where every neighbouring transition has
    finished settling: one settle guard past the end of the transition
    before it, one settle guard short of the transition after it.  The
    window ends one settle guard before active video begins - measured on
    the countdown tape, content-locked structure appears well before the
    nominal active start, so the trailing lags additionally carry a
    variance-based trim in the accumulated statistics.
    """
    settle = geometry.transition_settle_samples
    edge = geometry.sync_transition_samples
    porch = geometry.front_porch_samples
    sync_width = geometry.sync_pulse_samples

    # Two different isolation needs, two different guards.  The artifact
    # FIT must wait the full settle time so a transition's own
    # band-limited shape is never read as an artifact.  A LEVEL read only
    # needs the transition core out of the slice - a median is robust to
    # the decaying tail riding on the flat (that tail IS the artifact
    # being measured) - so it is guarded by the edge duration plus one
    # reciprocal bandwidth.  With the full settle guard the front porch
    # slice would be empty on PAL (29.3-sample porch, 16-sample guard on
    # both sides).
    level_guard = int(math.ceil(edge + settle / TRANSITION_SETTLE_BANDWIDTHS))

    # Reach back over the porch and the active falling edge far enough to
    # read the entry amplitude: the fall itself (one edge plus settle),
    # then two settle guards of active-area context for a median level.
    samples_before = int(math.ceil(porch + edge + 3 * settle))
    # the window reaches a little INTO active video: those lags are never
    # fitted, but they say whether picture content follows this line's
    # back porch - content-locked structure ahead of the active start
    # was measured to reach ~20 samples back into the porch, and only
    # comparing content-followed lines against dark-followed ones can
    # find its true extent
    samples_after = int(math.ceil(
        geometry.active_video_start_sample + level_guard + 2 * settle))
    anchor = samples_before

    def bounded(start, stop):
        start = int(max(0, math.ceil(start)))
        stop = int(min(samples_before + samples_after, math.floor(stop)))
        return (start, stop) if stop > start else (start, start)

    # REGION STARTS SIT AT THE PRECEDING EDGE'S 10-90 END, not a settle
    # guard past it: the artifact's largest part - overshoot,
    # undershoot, the first ring lobe - lives in that first
    # reciprocal-bandwidth, and guarding it out delayed every
    # measurement and correction until after the first oscillation
    # (the user's bug report, consistent on all samples).  Region ENDS
    # keep their guards - the NEXT transition's band-limited shape
    # begins before its crossing and is nobody's artifact.  The level
    # medians tolerate the included onset samples (a median is robust
    # to the decaying tail - that tail IS the artifact being measured).
    entry_context = bounded(0, anchor - (porch + edge / 2.0 + level_guard))
    front_porch = bounded(anchor - porch + edge / 2.0,
                          anchor - level_guard)
    sync_tip = bounded(anchor + edge / 2.0,
                       anchor + sync_width - (edge / 2.0 + level_guard))
    back_porch = bounded(
        anchor + sync_width + edge / 2.0,
        anchor + geometry.active_video_start_sample - settle)
    upcoming_context = bounded(
        anchor + geometry.active_video_start_sample + level_guard,
        samples_before + samples_after)

    return MeasurementWindow(
        samples_before_sync_fall=samples_before,
        samples_after_sync_fall=samples_after,
        sync_fall_index=anchor,
        total_samples=samples_before + samples_after,
        entry_context=entry_context,
        front_porch=front_porch,
        sync_tip=sync_tip,
        back_porch=back_porch,
        upcoming_context=upcoming_context,
        level_guard_samples=level_guard,
        crossing_search_radius=settle,
    )


def fractional_shift(values, offset):
    """Shift a window by a fractional number of samples, causally exact.

    The shift is realized as a phase ramp in the frequency domain.  The
    window's two ends sit at different levels (active area on one side,
    back porch on the other), and a plain circular shift would wrap that
    step and ring it across the window - so the straight line connecting
    the endpoints is removed first, shifted analytically, and restored.
    """
    sample_count = len(values)
    positions = np.arange(sample_count, dtype=np.float64)
    endpoint_slope = (values[-1] - values[0]) / (sample_count - 1)
    straight_line = values[0] + endpoint_slope * positions
    residual = values - straight_line
    spectrum = np.fft.rfft(residual)
    turn = np.exp(-2j * np.pi * np.fft.rfftfreq(sample_count) * offset)
    shifted_residual = np.fft.irfft(spectrum * turn, sample_count)
    shifted_line = values[0] + endpoint_slope * (positions - offset)
    return shifted_residual + shifted_line


def _median_of(window, region):
    """Median level over one of the window's flat-region slices."""
    start, stop = region
    if stop <= start:
        return float("nan")
    return float(np.median(window[start:stop]))


def _region_medians(windows, region):
    """Median level of one flat-region slice, for every window at once."""
    start, stop = region
    if stop <= start:
        return np.full(len(windows), np.nan)
    return np.median(windows[:, start:stop], axis=1)


def _find_falling_crossings(windows, nominal_index, search_radius,
                            upper_levels, lower_levels):
    """The sub-sample falling-midpoint position, for every window at once.

    For each window: scan around the nominal position for sample pairs
    straddling the midpoint of that window's own two levels, take the
    pair nearest the nominal position, and interpolate linearly inside
    it.  Windows with no crossing in the span return NaN.

    Everything here is batch numpy: this runs inside the decoder's main
    loop, and per-line Python was measured to slow the field loop enough
    to disturb the decoder's thread scheduling.
    """
    midpoints = 0.5 * (upper_levels + lower_levels)
    first = max(int(nominal_index - search_radius), 0)
    last = min(int(nominal_index + search_radius), windows.shape[1] - 2)
    span = windows[:, first:last + 2]
    here, after = span[:, :-1], span[:, 1:]
    straddles = ((here >= midpoints[:, None])
                 & (after <= midpoints[:, None]) & (here > after))
    offsets = np.arange(first, last + 1, dtype=np.float64)
    distance_from_nominal = np.abs(offsets - nominal_index)
    # prefer the crossing nearest the nominal position; non-straddling
    # pairs are pushed beyond every real candidate
    candidate_cost = np.where(straddles, distance_from_nominal[None, :],
                              np.inf)
    best = np.argmin(candidate_cost, axis=1)
    rows = np.arange(len(windows))
    found = np.isfinite(candidate_cost[rows, best])
    best_here = here[rows, best]
    best_after = after[rows, best]
    fraction = (best_here - midpoints) / np.where(
        best_here == best_after, 1.0, best_here - best_after)
    crossings = offsets[best] + fraction
    return np.where(found, crossings, np.nan)


def _fractional_shift_rows(windows, offsets):
    """fractional_shift applied to every row at once (same math)."""
    row_count, sample_count = windows.shape
    positions = np.arange(sample_count, dtype=np.float64)
    endpoint_slopes = (windows[:, -1] - windows[:, 0]) / (sample_count - 1)
    straight_lines = (windows[:, :1]
                      + endpoint_slopes[:, None] * positions[None, :])
    residuals = windows - straight_lines
    spectra = np.fft.rfft(residuals, axis=1)
    turns = np.exp(-2j * np.pi
                   * np.fft.rfftfreq(sample_count)[None, :]
                   * offsets[:, None])
    shifted_residuals = np.fft.irfft(spectra * turns, sample_count, axis=1)
    shifted_lines = (windows[:, :1] + endpoint_slopes[:, None]
                     * (positions[None, :] - offsets[:, None]))
    return shifted_residuals + shifted_lines


def measure_field_lines(field_lines_ire, geometry, window_plan, state,
                        average_fields):
    """Measure one field's sync intervals and fold them into the state.

    `field_lines_ire` is the field as a 2-D array (rows x samples per
    line) in IRE, row 0 being the first line of the field; the same code
    path serves the decoder at runtime and the offline report instrument.

    Returns a diagnostics dictionary describing what this field
    contributed; the same dictionary is stored at state['last_field'].
    """
    anchor = window_plan.sync_fall_index
    total = window_plan.total_samples
    row_count, samples_per_line = field_lines_ire.shape
    flat = field_lines_ire.reshape(-1)
    # remembered so the debug figure can realize the shipping correction
    # (strength needs the averaging horizon) without re-plumbing it
    state["average_fields"] = average_fields

    first_line = max(geometry.first_measurable_line, 1)
    last_line = min(geometry.last_measurable_line, row_count)
    line_numbers = np.arange(first_line, last_line)
    window_starts = line_numbers * samples_per_line - anchor
    inside = (window_starts >= 0) & (window_starts + total <= flat.size)
    window_starts = window_starts[inside]

    windows = flat[window_starts[:, None]
                   + np.arange(total)[None, :]].astype(np.float64)

    porch_levels = _region_medians(windows, window_plan.front_porch)
    tip_levels = _region_medians(windows, window_plan.sync_tip)
    # a real horizontal sync fall spans about one sync depth; anything
    # much shallower is a dropout or noise.  (Vertical-interval and
    # field-boundary half-line pulses never reach this test: the
    # measurable line range excludes them BY LINE NUMBER from the
    # specification - see build_geometry.)
    deep_enough = (np.isfinite(porch_levels) & np.isfinite(tip_levels)
                   & (porch_levels - tip_levels
                      > 0.5 * geometry.sync_depth_ire))
    windows = windows[deep_enough]
    porch_levels = porch_levels[deep_enough]
    tip_levels = tip_levels[deep_enough]

    crossings = _find_falling_crossings(
        windows, anchor, window_plan.crossing_search_radius,
        porch_levels, tip_levels)
    located = np.isfinite(crossings)
    windows = windows[located]
    porch_levels = porch_levels[located]
    tip_levels = tip_levels[located]
    crossings = crossings[located]

    # PER-FIELD LEVEL ANCHOR: the buffer measured here has not been
    # through the decoder's per-field black-level adjustment yet, so
    # consecutive fields carry a DC wander that the accumulated
    # variance would book as noise at every lag - measured to raise
    # the certification floor enough to degrade every pencil Q and
    # break parity with the offline instrument (which reads the
    # adjusted output).  The front porch is the reference level
    # (RINGING_RULES 19), so each field is anchored to its own median
    # porch level before accumulating; all downstream levels are
    # relative, so nothing else changes.
    field_anchor = float(np.median(porch_levels))
    windows = windows - field_anchor
    porch_levels = porch_levels - field_anchor
    tip_levels = tip_levels - field_anchor

    # move each measured crossing ONTO the anchor: the shift delays its
    # input by the given offset, so the offset is the distance the
    # crossing must travel, anchor minus crossing
    aligned_windows = _fractional_shift_rows(windows, anchor - crossings)

    # re-measure on the aligned windows so the residual reports what the
    # accumulation actually receives
    aligned_crossings = _find_falling_crossings(
        aligned_windows, anchor, window_plan.crossing_search_radius,
        porch_levels, tip_levels)
    residuals = np.abs(aligned_crossings - anchor)
    alignment_residual = (float(np.nanmean(residuals))
                          if np.any(np.isfinite(residuals)) else float("nan"))

    entry_levels = _region_medians(aligned_windows,
                                   window_plan.entry_context)
    entry_amplitudes = entry_levels - porch_levels
    upcoming_levels = _region_medians(aligned_windows,
                                      window_plan.upcoming_context)
    upcoming_amplitudes = upcoming_levels - porch_levels

    diagnostics = {
        "usable_lines": len(aligned_windows),
        "healthy": False,
        "reset": False,
    }

    if len(aligned_windows) >= MINIMUM_USABLE_LINES:
        stacked = aligned_windows
        field_mean = stacked.mean(axis=0)
        field_variance = stacked.var(axis=0)

        porch_level = _median_of(field_mean, window_plan.front_porch)
        tip_level = _median_of(field_mean, window_plan.sync_tip)
        back_porch_level = _median_of(field_mean, window_plan.back_porch)
        fall_step = porch_level - tip_level
        rise_step = back_porch_level - tip_level

        depth = geometry.sync_depth_ire
        diagnostics.update({
            "front_porch_ire": porch_level,
            "sync_tip_ire": tip_level,
            "back_porch_ire": back_porch_level,
            "alignment_residual": alignment_residual,
        })
        diagnostics["healthy"] = bool(
            abs(fall_step - depth) < HEALTH_STEP_TOLERANCE * depth
            and abs(rise_step - depth) < HEALTH_STEP_TOLERANCE * depth
            and abs(back_porch_level - porch_level)
            < HEALTH_LEVEL_AGREEMENT * depth
        )

        if diagnostics["healthy"]:
            state["consecutive_unhealthy_fields"] = 0
            _blend_field_into_averages(
                state, field_mean, field_variance, average_fields)
            _blend_field_into_averages(
                state, field_mean, field_variance,
                SLOW_HORIZON_FACTOR * max(average_fields, 1),
                key_prefix="slow_")
            # a second average over only the lines whose UPCOMING active
            # area is dark: structure that appears in the all-lines mean
            # but not here is caused by the picture that follows, not by
            # the sync interval, and must never enter the artifact fit
            # (it was measured, in the previous pipeline, to turn into a
            # ghost when fitted).  A handful of lines per field is
            # enough - the average runs at the slow horizon.
            dark_followed = (np.isfinite(upcoming_amplitudes)
                             & (upcoming_amplitudes
                                < STRATUM_MINIMUM_FRACTION
                                * geometry.sync_depth_ire))
            if int(np.count_nonzero(dark_followed)) >= 4:
                dark_stack = aligned_windows[dark_followed]
                _blend_field_into_averages(
                    state, dark_stack.mean(axis=0), dark_stack.var(axis=0),
                    SLOW_HORIZON_FACTOR * max(average_fields, 1),
                    key_prefix="dark_")
                state["dark_lines_average"] = _blend_scalar(
                    state.get("dark_lines_average"),
                    float(np.count_nonzero(dark_followed)),
                    state.get("dark_fields_accumulated", 0) - 1,
                    SLOW_HORIZON_FACTOR * max(average_fields, 1))
            # RISE-ALIGNED accumulation: under fall alignment the back
            # porch is smeared by the line-to-line sync-width jitter
            # (the phase-averaging attenuates exactly the back-porch
            # ring being modeled), so a second slow accumulation
            # aligns each line at its own sync RISE crossing.  Same
            # health gate, same reset lifecycle, and a dark-followed
            # companion for the rise frame's contamination trim.
            back_levels = _region_medians(windows, window_plan.back_porch)
            rise_nominal = anchor + geometry.sync_pulse_samples
            rise_crossings = _find_falling_crossings(
                -windows, rise_nominal,
                window_plan.crossing_search_radius,
                -tip_levels, -back_levels)
            rise_located = np.isfinite(rise_crossings)
            if int(np.count_nonzero(rise_located)) \
                    >= MINIMUM_USABLE_LINES:
                rise_aligned = _fractional_shift_rows(
                    windows[rise_located],
                    rise_nominal - rise_crossings[rise_located])
                _blend_field_into_averages(
                    state, rise_aligned.mean(axis=0),
                    rise_aligned.var(axis=0),
                    SLOW_HORIZON_FACTOR * max(average_fields, 1),
                    key_prefix="rise_")
                rise_dark = dark_followed[rise_located]
                if int(np.count_nonzero(rise_dark)) >= 4:
                    rise_dark_stack = rise_aligned[rise_dark]
                    _blend_field_into_averages(
                        state, rise_dark_stack.mean(axis=0),
                        rise_dark_stack.var(axis=0),
                        SLOW_HORIZON_FACTOR * max(average_fields, 1),
                        key_prefix="rise_dark_")
            _accumulate_fall_strata(
                state, stacked, entry_amplitudes, entry_levels,
                porch_levels, geometry, window_plan)
            state["usable_lines_average"] = _blend_scalar(
                state.get("usable_lines_average"),
                float(len(aligned_windows)),
                state.get("fields_accumulated", 0), average_fields)

    if not diagnostics["healthy"]:
        misses = state.get("consecutive_unhealthy_fields", 0) + 1
        state["consecutive_unhealthy_fields"] = misses
        if (misses >= RESET_HORIZONS * max(average_fields, 1)
                and state.get("fields_accumulated", 0) > 0):
            # a recording change, gap, or unrecorded stretch: everything
            # accumulated describes a signal that is no longer there
            resets = state.get("resets", 0) + 1
            geometry_kept = state.get("geometry")
            state.clear()
            state["resets"] = resets
            if geometry_kept is not None:
                state["geometry"] = geometry_kept
            diagnostics["reset"] = True

    state["last_field"] = diagnostics
    if diagnostics["reset"]:
        history = state.setdefault("health_history", [])
        history.append("reset")
    else:
        history = state.setdefault("health_history", [])
        history.append("healthy" if diagnostics["healthy"] else "rejected")
    return diagnostics


def _blend_scalar(held, value, fields_seen, average_fields):
    """One scalar under the same averaging semantics as the windows."""
    if held is None or fields_seen == 0:
        return value
    weight = 1.0 / min(fields_seen + 1, max(average_fields, 1))
    return held * (1.0 - weight) + value * weight


def _blend_field_into_averages(state, field_mean, field_variance,
                               average_fields, key_prefix=""):
    """Fold one healthy field's statistics into the running averages.

    Plain mean while fewer fields than the averaging horizon have been
    seen (so early fields carry equal weight), exponential thereafter (so
    the model tracks slow drift and, with the health gate, recording
    changes).  Non-finite input never enters the average: one poisoned
    value would corrupt every following field.

    With `key_prefix` the same semantics maintain a second, slower set of
    averages (the artifact fit's input) alongside the field-tracking one.
    """
    if not (np.all(np.isfinite(field_mean))
            and np.all(np.isfinite(field_variance))):
        return
    mean_key = key_prefix + "interval_mean"
    variance_key = key_prefix + "interval_variance"
    count_key = key_prefix + "fields_accumulated"
    fields_seen = state.get(count_key, 0)
    held_mean = state.get(mean_key)
    if held_mean is None or held_mean.shape != field_mean.shape:
        state[mean_key] = field_mean.copy()
        state[variance_key] = field_variance.copy()
        state[count_key] = 1
        return
    weight = 1.0 / min(fields_seen + 1, max(average_fields, 1))
    state[mean_key] = held_mean * (1.0 - weight) + field_mean * weight
    state[variance_key] = (
        state[variance_key] * (1.0 - weight) + field_variance * weight)
    state[count_key] = fields_seen + 1


def _fall_window_layout(geometry, window_plan):
    """The sub-window each stratum accumulates, around the ACTIVE fall.

    Lags before the crossing hold the entry level; lags after it hold
    the fall's aftermath across the front porch.  The aftermath is only
    valid until the sync falling edge's own response begins, and that
    distance varies per line (the porch start is variable), so validity
    is tracked per lag.
    """
    settle = geometry.transition_settle_samples
    before_fall = 2 * settle
    after_fall = int(math.ceil(geometry.front_porch_samples)) + settle
    return before_fall, after_fall


def _accumulate_fall_strata(state, aligned_windows, entry_amplitudes,
                            entry_levels, porch_levels, geometry,
                            window_plan):
    """Accumulate the active falling edge's aftermath by amplitude,
    aligned at the ACTIVE FALL itself.

    The sync edges witness exactly one transition amplitude - the sync
    depth - so the active falling edge, whose amplitude varies with
    picture content, is the only measurement of how the artifacts scale
    with amplitude.  The windows arriving here are aligned at the SYNC
    fall; the active fall's own position varies line to line (the porch
    start is variable), and averaging without re-aligning smears the
    aftermath's shape - so each line is re-aligned at its own active
    fall crossing before it joins its stratum.
    """
    stratum_depth = STRATUM_DEPTH_FRACTION * geometry.sync_depth_ire
    minimum = STRATUM_MINIMUM_FRACTION * geometry.sync_depth_ire
    anchor = window_plan.sync_fall_index
    before_fall, after_fall = _fall_window_layout(geometry, window_plan)
    length = before_fall + after_fall

    qualifying = (np.isfinite(entry_amplitudes)
                  & (entry_amplitudes >= minimum))
    if not np.any(qualifying):
        return
    windows = aligned_windows[qualifying]
    amplitudes = entry_amplitudes[qualifying]
    crossings = _find_falling_crossings(
        windows,
        anchor - geometry.front_porch_samples,
        int(geometry.front_porch_samples / 2.0),
        entry_levels[qualifying], porch_levels[qualifying])
    located = np.isfinite(crossings)
    windows = windows[located]
    amplitudes = amplitudes[located]
    crossings = crossings[located]
    if not len(windows):
        return

    whole = np.floor(crossings).astype(int)
    starts = whole - before_fall
    inside = (starts >= 0) & (starts + length <= windows.shape[1])
    windows = windows[inside]
    amplitudes = amplitudes[inside]
    crossings = crossings[inside]
    starts = starts[inside]
    whole = whole[inside]
    if not len(windows):
        return

    sub_windows = windows[np.arange(len(windows))[:, None],
                          starts[:, None] + np.arange(length)[None, :]]
    aligned = _fractional_shift_rows(sub_windows, whole - crossings)

    # the aftermath is only trustworthy until the sync fall's own
    # response begins: one edge width plus a settle guard short of the
    # sync fall crossing, per line
    settle = geometry.transition_settle_samples
    limit = (anchor - geometry.sync_transition_samples - settle
             - crossings + before_fall)
    lag_positions = np.arange(length, dtype=np.float64)[None, :]
    valid = lag_positions < np.maximum(limit[:, None], before_fall)

    strata = state.setdefault("fall_strata", {})
    stratum_indices = (amplitudes // stratum_depth).astype(int)
    for stratum_index in np.unique(stratum_indices):
        member = stratum_indices == stratum_index
        stratum = strata.get(int(stratum_index))
        if stratum is None:
            stratum = {"window_sum": np.zeros(length),
                       "lag_count": np.zeros(length),
                       "amplitude_sum": 0.0, "line_count": 0}
            strata[int(stratum_index)] = stratum
        stratum["window_sum"] += np.sum(
            np.where(valid[member], aligned[member], 0.0), axis=0)
        stratum["lag_count"] += np.sum(valid[member], axis=0)
        stratum["amplitude_sum"] += float(np.sum(amplitudes[member]))
        stratum["line_count"] += int(np.count_nonzero(member))


def _strata_swing_factor(state, geometry, window_plan,
                         sync_swing_per_step):
    """How much larger, per unit step, the artifact swing runs at the
    worst picture-amplitude stratum than at the sync edge.

    The transient gate's envelope predicts each event's ring swing
    from the SYNC-calibrated model, but the strata witness shows swing
    per unit step is not amplitude-linear - an envelope sized at the
    sync ratio would leak the worst stratum's ring past the gate.  The
    factor is the worst measured stratum-to-sync ratio, floored at 1
    (never used to shrink the envelope), and 1.0 when unmeasured.
    """
    if sync_swing_per_step <= 0.0:
        return 1.0
    strata = state.get("fall_strata") or {}
    before_fall, _ = _fall_window_layout(geometry, window_plan)
    settle = geometry.transition_settle_samples
    worst = 1.0
    for stratum in strata.values():
        if stratum["line_count"] < MINIMUM_USABLE_LINES:
            continue
        counts = stratum["lag_count"]
        valid = counts >= 0.5 * stratum["line_count"]
        if int(np.count_nonzero(valid[before_fall:])) < 6:
            continue
        mean = stratum["window_sum"] / np.maximum(counts, 1.0)
        amplitude = stratum["amplitude_sum"] / stratum["line_count"]
        if amplitude <= 0.0:
            continue
        entry = float(np.median(mean[:max(before_fall - settle, 2)]))
        trailing = np.flatnonzero(valid)
        porch = float(np.median(mean[trailing[-6:]]))
        crossing, width = _measure_edge(
            mean, before_fall, settle, entry, porch)
        if not (np.isfinite(crossing) and np.isfinite(width)):
            continue
        positions = np.arange(len(mean), dtype=np.float64)
        reference = entry + (porch - entry) * _smooth_step(
            positions, crossing, width)
        start = int(math.ceil(crossing + width))
        deviation = np.where(valid, mean - reference, 0.0)[start:]
        if not len(deviation):
            continue
        swing = float(np.max(np.abs(deviation)))
        worst = max(worst, (swing / amplitude) / sync_swing_per_step)
    return worst


def _measure_landing_law(state, geometry, window_plan):
    """THE SIDEBAND AMPLITUDE LAW's measurement (hsEE5).

    The artifact is level-dependent (the FM carrier rides the level),
    and the correction's fitted response is calibrated at the sync
    fall - landing at the TIP.  This measures how much of that same
    response the picture-fall strata want: each stratum (line-end
    falls landing at BLANKING, observed in the front porch - the
    rule-19 calibration class) is projected onto the model's own
    realized unit-fall correction, line-weighted across solid strata.
    Measured on both content tapes: ~0 - the picture artifact rings
    at level-shifted frequencies the sync-calibrated shape does not
    match, so transplanting it is pure harm there.  The law scales
    each gated FALL event by its landing level, interpolated between
    the two witnessed anchors (tip -> 1 by construction, blanking ->
    the measured want) and held beyond.  No strata witnessed (the
    pulse-and-bar capture) -> no law -> scale 1 everywhere.
    """
    strata = state.get("fall_strata") or {}
    model = state.get("model")
    if model is None or not strata:
        return None
    margin = 4 * geometry.transition_settle_samples
    length = margin + window_plan.total_samples
    step_signal = np.zeros(length)
    step_signal[margin:] = -1.0
    # the projection target is the RING-CLASS response alone: the
    # smear/even class legitimately serves the porch (the home tape's
    # front porch lives on it, and this frame's own per-stratum
    # reference absorbs slow content anyway), so only the oscillatory
    # class answers to the law
    ring_sections = [section for section in model["sections"]
                     if abs(section["pole"].imag) > 0.0]
    if not ring_sections:
        return None
    edge_kernel = _edge_drive_kernel(model["edges"]["sync_fall"][1])
    rise_kernel = (_edge_drive_kernel(model["rise_edge_width"])
                   if model.get("rise_edge_width") else None)
    unit_response = _modeled_artifact(
        step_signal, ring_sections, edge_kernel, 1.0, 0.0, None, None,
        rise_kernel)
    settle = geometry.transition_settle_samples
    before_fall, _ = _fall_window_layout(geometry, window_plan)
    points = []
    for stratum in strata.values():
        if stratum["line_count"] < MINIMUM_USABLE_LINES:
            continue
        counts = stratum["lag_count"]
        valid = counts >= 0.5 * stratum["line_count"]
        if int(np.count_nonzero(valid[before_fall:])) < 6:
            continue
        mean = stratum["window_sum"] / np.maximum(counts, 1.0)
        amplitude = stratum["amplitude_sum"] / stratum["line_count"]
        if amplitude <= 4.0:
            continue
        entry = float(np.median(mean[:max(before_fall - settle, 2)]))
        trailing = np.flatnonzero(valid)
        porch = float(np.median(mean[trailing[-6:]]))
        crossing, width = _measure_edge(
            mean, before_fall, settle, entry, porch)
        if not (np.isfinite(crossing) and np.isfinite(width)):
            continue
        positions = np.arange(len(mean), dtype=np.float64)
        reference = entry + (porch - entry) * _smooth_step(
            positions, crossing, width)
        start = int(math.ceil(crossing + width))
        deviation = np.where(valid, mean - reference, 0.0)[start:]
        if len(deviation) < 8:
            continue
        offset = int(round(margin + start - crossing))
        if offset < 0 or offset + len(deviation) > len(unit_response):
            continue
        applied_shape = unit_response[
            offset:offset + len(deviation)] * amplitude
        energy = float(applied_shape @ applied_shape)
        if energy < 1e-9:
            continue
        want = float(deviation @ applied_shape) / energy
        points.append((float(amplitude), float(want),
                       int(stratum["line_count"])))
    if not points:
        return None
    total_lines = float(sum(lines for _, _, lines in points))
    blank_want = sum(want * lines for _, want, lines in points) \
        / max(total_lines, 1.0)
    return {
        "levels": (float(model["levels"]["sync_tip"]),
                   float(model["levels"]["front_porch"])),
        "scales": (1.0, float(np.clip(blank_want, 0.0, 1.0))),
        "points": points,
    }


# =============================================================================
# The artifact model - damped modes fitted to the whole interval
# =============================================================================

def _denominator_of(pole):
    """The real polynomial whose root(s) realize one tracked pole.

    A complex pole represents a conjugate pair (an oscillation); a real
    pole is a plain decay.  Poles are stored with non-negative angle.
    """
    if abs(pole.imag) > 0.0:
        return np.array([1.0, -2.0 * pole.real, abs(pole) ** 2])
    return np.array([1.0, -pole.real])


def _shelf_deviation(values, pole_a):
    """(H_a - 1) applied to a sequence, at unit mix: the one-pole
    relaxation SHELF that models a smear component.

    Physical smear is energy dispersion after a low-pass: a fraction c
    of the signal passes through the relaxation (1-a)/(1 - a z^-1) and
    the rest passes clean, H = (1-c) + c (1-a)/(1 - a z^-1).  H is
    exactly unity at DC for every c (level neutral by construction),
    minimum-phase for 0 <= c <= 1, and its deviation from identity is
    LINEAR in c - so each smear component is one least-squares column
    per view, this function evaluated on the view's reference.  Unlike
    the earlier drive-coupled additive pair (whose response vanished at
    high frequency through the edge kernel, so its inversion only
    DELETED the tail), the shelf keeps its high-frequency deficit: the
    exact inverse boosts the edge band and the smeared energy lands
    back in the transition slope - the requirements document's
    energy-relocation rule.  Its inverse settles monotonically to zero:
    no oscillation, no repetition.
    """
    filtered = scipy.signal.lfilter([1.0 - pole_a], [1.0, -pole_a],
                                    values)
    return filtered - values


def _mode_rotation(pole):
    """How far a mode rotates within its own decay time (radians)."""
    magnitude = min(abs(pole), 1.0 - 1e-6)
    decay_samples = -1.0 / math.log(magnitude)
    return abs(np.angle(pole)) * decay_samples


def estimate_decay_modes(residual_windows, capacity, noise_floor):
    """Shared damped modes across several residual windows (matrix pencil).

    Each residual is modeled as a sum of decaying (possibly oscillating)
    exponentials; the pencil recovers their poles.  Stacking every
    window's data rows makes the fit joint - all windows share one pole
    set, which is what physical resonances do - while each window keeps
    its own amplitudes.  Rank is truncated where the singular values
    reach the accumulated noise floor, so noise is never promoted to a
    mode; `capacity` bounds the count above the model's own limits.
    """
    # one shared pencil width across every window, so the stacked rows
    # describe one shared pole set; just past the capacity, which keeps
    # even the short front porch window contributing rows when it can
    columns = capacity + 2
    row_blocks = []
    shifted_blocks = []
    for residual in residual_windows:
        rows = len(residual) - columns
        if rows < 1:
            continue
        block = np.lib.stride_tricks.sliding_window_view(
            residual, columns + 1)[:rows]
        row_blocks.append(block[:, :-1])
        shifted_blocks.append(block[:, 1:])
    if not row_blocks:
        return np.array([], dtype=complex)
    current = np.vstack(row_blocks)
    shifted = np.vstack(shifted_blocks)
    left, singular_values, right_h = np.linalg.svd(
        current, full_matrices=False)
    keep = singular_values > max(
        noise_floor * math.sqrt(current.shape[0]),
        max(current.shape) * np.finfo(float).eps * singular_values[0])
    rank = min(int(np.count_nonzero(keep)), capacity)
    if rank == 0:
        return np.array([], dtype=complex)
    reduced = (np.diag(1.0 / singular_values[:rank])
               @ left[:, :rank].conj().T @ shifted
               @ right_h[:rank, :].conj().T)
    poles = np.linalg.eigvals(reduced)
    # canonical form: non-negative angle (a conjugate pair is one entry),
    # magnitude inside the unit circle (a growing "mode" is a noise
    # artifact - nothing physical grows)
    cleaned = []
    for pole in poles:
        if not np.isfinite(pole):
            continue
        if pole.imag < 0.0:
            continue
        magnitude = abs(pole)
        if magnitude >= 1.0 or magnitude < 1e-3:
            continue
        cleaned.append(complex(pole))
    return np.array(cleaned, dtype=complex)


def _line_locked_floor(windows, sample_rate_mhz, maximum_ring_mhz,
                       guard_mhz):
    """The accumulated mean's PERSISTENT floor inside the certification
    windows, median across windows.

    Line-locked content (color-under and burst remnants, comb
    structure) survives accumulation - unlike the statistical noise
    floor it does not shrink as fields accumulate - and by the
    standing rule it is "not correlated with the known input data".
    As the statistical floor sinks beneath it, the pencil reads
    ring-plus-persistent-tone as one longer-decaying ring: measured
    on the home tape as every product's decay stretching monotonically
    through the decode (0.48 -> 1.25 us over 190 updates) with the
    shipped strength climbing in step, and the corrected audits
    reading Q ~ 30 at the length bound.  A mode's testimony ends
    where this floor begins, so the certification floor is the larger
    of the two.

    Measured ONLY where line-locked content can live apart from real
    products: the known carrier bin (the burst subcarrier sits at
    exactly rate/4 - the output rate is four times the subcarrier by
    construction) and the band beyond the ring bound plus `guard_mhz`
    (one spectral half-width of the FASTEST witnessable ring, whose
    decay is the edge width; half-width 1/(pi tau)) - real products
    just under the bound spread that far, and counting their skirt as
    contamination was measured to de-certify the home tape's genuine
    3.0-3.3 MHz family (fp 0.37 -> 0.90 with a visible 2.7 residual).
    """
    subcarrier_mhz = sample_rate_mhz / 4.0
    levels = []
    for window in windows:
        if len(window) < 8:
            continue
        spectrum = np.fft.rfft(window)
        frequencies = np.fft.rfftfreq(len(window),
                                      d=1.0 / sample_rate_mhz)
        bin_width = sample_rate_mhz / len(window)
        mask = (frequencies > maximum_ring_mhz + guard_mhz) \
            | (np.abs(frequencies - subcarrier_mhz) <= bin_width)
        if not np.any(mask):
            continue
        # Parseval for rfft: interior bins carry double weight
        band_energy = 2.0 * float(np.sum(np.abs(spectrum[mask]) ** 2))
        levels.append(math.sqrt(band_energy) / len(window))
    if not levels:
        return 0.0
    return float(np.median(levels))


def _smooth_step(positions, crossing, width_10_90):
    """A clean band-limited step: 0 before, 1 after, measured rise width.

    An error-function step whose 10-90% width matches the measured edge;
    the excitation reference must contain the decoder's own band-limit,
    or the fit would read the edge's finite rise as an artifact.
    """
    # the 10-90% width of erf(x / (sigma sqrt 2)) is 2.563 sigma
    sigma = max(width_10_90, 0.5) / 2.563
    return 0.5 * (1.0 + scipy.special.erf(
        (positions - crossing) / (sigma * math.sqrt(2.0))))


def _measure_edge(values, nominal_index, search_radius, before_level,
                  after_level):
    """One transition's crossing position and 10-90% width, measured."""
    window = values if after_level < before_level else -values
    upper = before_level if after_level < before_level else -before_level
    lower = after_level if after_level < before_level else -after_level
    crossings = _find_falling_crossings(
        window[None, :], nominal_index, search_radius,
        np.array([upper]), np.array([lower]))
    crossing = float(crossings[0])
    if not np.isfinite(crossing):
        return float("nan"), float("nan")
    span = abs(before_level - after_level)
    low_mark = min(before_level, after_level) + 0.1 * span
    high_mark = min(before_level, after_level) + 0.9 * span
    first = max(int(crossing) - search_radius * 2, 0)
    last = min(int(crossing) + search_radius * 2, len(values) - 1)
    segment = values[first:last + 1]
    above = np.flatnonzero(segment >= high_mark)
    below = np.flatnonzero(segment <= low_mark)
    if len(above) == 0 or len(below) == 0:
        return crossing, float("nan")
    if after_level < before_level:
        width = float(below[0] - above[-1]) if below[0] > above[-1] else 1.0
    else:
        width = float(above[0] - below[-1]) if above[0] > below[-1] else 1.0
    return crossing, abs(width)


def _build_excitation(slow_mean, geometry, window_plan):
    """The clean reference interval: what the signal SHOULD look like.

    Piecewise flat at the measured levels, stepping at each measured
    transition with each edge's own measured rise width.  Every artifact
    is then the difference between the accumulated mean and this
    reference.  The edge cores carry no fit weight, so the precise edge
    shape is not load-bearing - it only has to carry the band-limit.
    """
    anchor = window_plan.sync_fall_index
    positions = np.arange(window_plan.total_samples, dtype=np.float64)
    entry_level = _median_of(slow_mean, window_plan.entry_context)
    porch_level = _median_of(slow_mean, window_plan.front_porch)
    tip_level = _median_of(slow_mean, window_plan.sync_tip)
    back_level = _median_of(slow_mean, window_plan.back_porch)
    search = window_plan.crossing_search_radius

    # the active falling edge sits about one front porch before the sync
    # fall; its position varies with content, so it is searched wider
    active_fall_nominal = anchor - geometry.front_porch_samples
    active_fall, active_width = _measure_edge(
        slow_mean, active_fall_nominal, 2 * search, entry_level, porch_level)
    sync_fall, fall_width = _measure_edge(
        slow_mean, anchor, search, porch_level, tip_level)
    sync_rise_nominal = anchor + geometry.sync_pulse_samples
    sync_rise, rise_width = _measure_edge(
        slow_mean, sync_rise_nominal, search, tip_level, back_level)

    # EACH SYNC EDGE KEEPS ITS OWN MEASURED WIDTH.  The old shared
    # width ("the same channel property seen twice") baked the
    # assumption that rise and fall are symmetric - on real decks they
    # are not (FM limiter and emphasis asymmetry), and with the onset
    # now fully weighted the edge shape IS load-bearing: a rise
    # reference built with the fall's width manufactures a phantom
    # first-lobe deviation at every rise that the correction then
    # fights (the user's diagnosis: the rise side is modeled out of
    # phase / slightly misaligned - measured: both widths came out
    # identical to three decimals on every tape because this line made
    # them so).  The per-polarity drive kernels downstream only become
    # real with independent widths.

    edges = {
        "active_fall": (active_fall, active_width, entry_level, porch_level),
        "sync_fall": (sync_fall, fall_width, porch_level, tip_level),
        "sync_rise": (sync_rise, rise_width, tip_level, back_level),
    }
    excitation = np.full_like(positions, entry_level)
    for crossing, width, before_level, after_level in edges.values():
        if not (np.isfinite(crossing) and np.isfinite(width)):
            return None, None, None
        excitation += (after_level - before_level) * _smooth_step(
            positions, crossing, width)
    levels = {"entry": entry_level, "front_porch": porch_level,
              "sync_tip": tip_level, "back_porch": back_level}
    return excitation, edges, levels


def _reference_uncertainty(length, edges, deviation):
    """Per-lag uncertainty OF THE REFERENCE around each transition.

    The reference's error-function step matches the real band-limited
    edge only to a residual whose size is measured (the deviation inside
    the core) and whose reach follows the edge's own slope profile.  A
    hard core exclusion left the first-lobe zone completely
    unconstrained - the model was free to paint a 2 IRE lobe right
    after the edge, visible in the corrected output - so the exclusion
    is CONTINUOUS instead: the reference uncertainty is added to the
    mean's variance, the weight falls smoothly inside the core and
    recovers immediately past it, and every lag the eye sees stays
    constrained.  The model is still never asked to reproduce the edge
    itself (the uncertainty peaks exactly there).
    """
    positions = np.arange(length, dtype=np.float64)
    uncertainty = np.zeros(length)
    for crossing, width, _, _ in edges.values():
        if not (np.isfinite(crossing) and np.isfinite(width)):
            continue
        # the 10-90% width of an error-function step is 2.563 sigma;
        # the erf mismatch profile follows the edge's slope (gaussian).
        # The mismatch is measured over the PRE-CROSSING half of the
        # core ONLY: the causal model cannot reach before the edge, so
        # pre-half deviation is pure reference error - while the
        # post-half deviation IS the artifact onset (overshoot,
        # undershoot, the first ring lobe).  Measuring the mismatch
        # over the whole core let the onset's own size inflate the
        # uncertainty that silences it (the user's bug report: the
        # correction and the measurements wait until after the first
        # oscillation on every sample; measured before as the home
        # tape's onset under-weighted 3-4x).  Artifacts are measured
        # immediately so the transition comes out to spec.
        sigma = max(width, 0.5) / 2.563
        core_start = max(int(math.floor(crossing - width)), 0)
        core_stop = min(int(math.ceil(crossing)), length)
        if core_stop <= core_start or deviation is None:
            continue
        mismatch = float(np.max(np.abs(deviation[core_start:core_stop])))
        profile = np.exp(-0.5 * ((positions - crossing) / sigma) ** 2)
        uncertainty = np.maximum(uncertainty, mismatch * profile)
    return uncertainty


def _fit_weights(state, window_plan, edges, effective_observations,
                 deviation=None):
    """Per-lag least-squares weights for the artifact fit.

    Inverse variance OF THE ACCUMULATED MEAN - the fit target - so noisy
    lags count less; with zero weight where the data is not the
    channel's response to a known transition (the picture content before
    the active falling edge), and the reference's own measured
    uncertainty added around every transition core (see
    _reference_uncertainty - the artifact model must never be asked to
    reproduce an edge, but every lag past the core stays constrained).

    The mean's variance is the cross-line variance divided by
    `effective_observations` (usable lines per field x fields inside the
    averaging horizon - the same accounting as the model's noise floor).
    The distinction is load-bearing: the section significances are the
    fit's energies in these whitened units, and whitening with the
    per-line variance instead understated them by that same factor
    (~8000 at full accumulation) - measured to crush every ringing
    section to a few percent applied strength while the smear's long
    tail alone survived.
    """
    variance = np.maximum(state["slow_interval_variance"], 1e-6)
    # SELF-SILENCING GUARD: the fit target is the accumulated MEAN, and
    # a lag whose cross-line variance is inflated by the phase jitter
    # of the very component being corrected must not silence its own
    # mean evidence (measured on the countdown tape: the back porch's
    # first lags - the incomplete rising transient and its ring, the
    # exact evidence the eye sees - carried 5-20x the porch variance
    # and the whitening starved them, leaving the tape uncorrected).
    # The cap is the established content boundary, four times the front
    # porch's variance: above it is picture content (the trims own
    # that), below it is sync-interval measurement and keeps its say.
    porch_variance = float(np.median(
        variance[slice(*window_plan.front_porch)]))
    variance = np.minimum(variance, 4.0 * porch_variance)
    variance = variance / max(effective_observations, 1.0)
    variance = variance + _reference_uncertainty(
        len(variance), edges, deviation) ** 2
    weights = 1.0 / variance
    weights[: window_plan.front_porch[0]] = 0.0
    weights[window_plan.back_porch[1]:] = 0.0

    # CONTAMINATION TRIM.  Structure locked to the start of the upcoming
    # active area reaches back into the porch (measured ~20 samples,
    # further than any settling argument predicts), and fitting it turns
    # it into a ghost painted across the whole picture.  It exists only
    # on lines followed by content, so the difference between the
    # all-lines mean and the dark-followed mean measures it per lag; the
    # weight falls continuously as that difference clears the dark
    # mean's own noise.
    dark_mean = state.get("dark_interval_mean")
    if dark_mean is not None and dark_mean.shape == weights.shape:
        dark_lines = max(state.get("dark_lines_average", 1.0), 1.0)
        dark_fields = max(state.get("dark_fields_accumulated", 1), 1)
        dark_noise = np.sqrt(
            np.maximum(state["dark_interval_variance"], 1e-6)
            / (dark_lines * dark_fields))
        difference = state["slow_interval_mean"] - dark_mean
        rise_settled = window_plan.back_porch[0]
        # a CONSTANT offset between the two populations is a level
        # property (lines followed by darkness sit at slightly different
        # blanking than the average line - measured ~0.45 IRE on the
        # countdown tape); only the SHAPE of the difference marks
        # upcoming-content contamination, so the offset is removed
        # before the comparison
        offset = float(np.median(difference[rise_settled:]))
        contamination = np.abs(difference - offset)
        confidence = dark_noise ** 2 / (dark_noise ** 2
                                        + contamination ** 2)
        weights[rise_settled:] *= confidence[rise_settled:]
    return weights


def _shelf_filter_coefficients(pole_a, mix):
    """The one-pole relaxation shelf H at a given mix, as filter taps.

    H = ((1 - c a) - a (1 - c) z^-1) / (1 - a z^-1), exactly unity at
    DC for every mix c.  Its EXACT inverse (numerator and denominator
    swapped) is a single one-pole filter with a REAL pole and a REAL
    zero, whose impulse response extends out and settles without ever
    oscillating - the requirements document's smear correction shape.
    Inverting the smear through the joint ring fixed point instead was
    measured to add an oscillatory effect on the pulse-and-bar tape:
    the inverse of a SUM of shelves can carry complex poles even though
    every component is non-oscillatory.
    """
    numerator = np.array([1.0 - mix * pole_a, -pole_a * (1.0 - mix)])
    denominator = np.array([1.0, -pole_a])
    return numerator, denominator


def _even_relaxation_response(values, pole_a, edge_kernel, polarity,
                              noise_bias=0.0):
    """The EVEN half of a smear component: the relaxation driven by the
    MAGNITUDE of the change.

    A linear shelf answers opposite transitions with opposite-signed
    aftermaths - but the countdown tape's channel sags and recovers
    upward after BOTH sync edges (the falling edge over-completes and
    the rising edge under-completes, one polarity-asymmetric response).
    That is the nonlinearity the standing rules reserve the parametric
    machinery for: the same certified relaxation pole, excited by
    |change| through the same edge kernel, fits a same-direction sag
    after every transient; its per-view mix is the smear pair's second
    least-squares coefficient (the slot the linear shelf leaves free).
    Each edge's aftermath amplitude belongs to the polarity that made
    it (the transient starts from a different level per direction), so
    the drive is HALF-WAVE: polarity -1 responds to falling changes,
    +1 to rising changes - the fall view's evidence calibrates the
    falling mix and the rise view's the rising mix, and the application
    uses both directly with no cross-polarity blend (a blended single
    mix was measured to over-lift the countdown tape's tip and to shut
    the pulse-and-bar tape down entirely through the no-harm governor).
    Each edge's response decays to zero, so no permanent level shift
    can result.
    """
    change = np.diff(values, prepend=values[:1])
    # NOISE DEBIAS: a half-wave drive RECTIFIES noise - max(n, 0) has
    # positive mean, so on a noisy input the even response would ride
    # on every sample as a standing pedestal that a clean-reference
    # calibration never saw (measured: the realized correction
    # diverged from the fit and the no-harm governor shut the
    # pulse-and-bar tape down).  The earlier remedy - a 3-sigma
    # threshold gate on the change - sat at the same scale as the
    # countdown tape's slow edge slopes and starved the clean-
    # reference fit columns eight-fold against the realized response
    # (the phantom sag that held that tape's correction at strength
    # 0.06).  The scale-stable form is PLAIN rectification plus the
    # analytic mean of the rectified noise, subtracted: for zero-mean
    # Gaussian sample noise sigma the first difference carries
    # sigma_d = sqrt(2) sigma, and E[max(n, 0)] = sigma_d / sqrt(2 pi)
    # (the half-normal mean) - so the caller passes noise_bias =
    # sigma_d / sqrt(2 pi) for ITS input's own noise (zero for the
    # clean references, the accumulated floor for accumulated means,
    # the line noise for raw fields), and the drive's expectation over
    # noise is zero at EVERY scale: fit columns and realized response
    # finally live at the same scale.  The fit's drive IS the
    # application's drive.
    if polarity < 0:
        rectified = np.maximum(-change, 0.0) - noise_bias
    else:
        rectified = np.maximum(change, 0.0) - noise_bias
    center = (len(edge_kernel) - 1) // 2
    spread = scipy.signal.fftconvolve(rectified, edge_kernel,
                                      mode="full")
    half_wave_drive = spread[center:center + len(values)]
    return scipy.signal.lfilter([1.0 - pole_a], [1.0, -pole_a],
                                half_wave_drive)


def _ghost_deviation(reference, ghost_track):
    """The pre-VCR ghost's contribution to a reference.

    Ghosting is an attenuated DELAYED COPY of the signal (multipath /
    impedance reflection) applied BEFORE the recorder's smear and
    ringing - the requirements document's sequencing - so the smear and
    ring stages received the GHOSTED signal, and fitting them against a
    ghost-free reference lets the ghost's delayed edges pollute both
    measurements.  Realized level-neutrally as
    mix (shift(reference) - reference): exactly zero over flat regions
    (the measured levels already carry the ghost's DC share), the
    delayed-edge signature at each transition.  The ghost itself is NOT
    corrected here - it may be an attribute of the source recording.
    """
    if not ghost_track:
        return np.zeros_like(reference)
    mix = float(ghost_track.get("mix", 0.0))
    lag = float(ghost_track.get("lag", 0.0))
    if mix == 0.0 or not np.isfinite(lag) or lag <= 0.0:
        return np.zeros_like(reference)
    return mix * (fractional_shift(reference, lag) - reference)


def _wiener_residual_spectrum(deviation, reference, weights,
                              mean_variance, sample_rate_mhz):
    """The residual channel spectrum after the known transient: Wiener
    deconvolution of the accumulated deviation by the reference's own
    change.

    Identifies WHICH frequencies remain locked to the transient (ring
    components the time-domain certification may have missed, and the
    validation view of what the applied correction leaves).  Wiener
    loading with the accumulated mean's own noise power keeps bins the
    drive cannot witness at zero instead of exploding.  IDENTIFICATION
    AND VALIDATION ONLY - never amplitude fitting: a truncated window's
    spectral division manufactures phantom features that survive
    averaging (the analog-arc lesson), so the fit stays entirely in the
    time domain and only SEEDS candidates from the peaks found here.

    Returns (frequencies_mhz, |residual response|, noise reference) -
    a peak standing above the reference is a component the transient
    still excites.
    """
    span = weights > 0.05 * float(np.max(weights)) if np.any(weights) \
        else weights > 0.0
    windowed = np.where(span, deviation, 0.0)
    drive = np.diff(reference, prepend=reference[:1])
    deviation_spectrum = np.fft.rfft(windowed)
    drive_spectrum = np.fft.rfft(drive)
    drive_power = np.abs(drive_spectrum) ** 2
    # per-bin noise power of the windowed accumulated deviation: the
    # mean's per-lag variance summed over the span (flat across bins
    # for uncorrelated lag noise)
    noise_power = float(np.sum(np.where(span, mean_variance, 0.0)))
    load = max(noise_power, 1e-12)
    response = (deviation_spectrum * np.conj(drive_spectrum)
                / (drive_power + load))
    reference_level = (math.sqrt(load) * np.sqrt(drive_power)
                       / (drive_power + load))
    frequencies = np.fft.rfftfreq(len(deviation)) * sample_rate_mhz
    return frequencies, np.abs(response), reference_level


def _pencil_capacity(window_plan):
    """Damped-mode estimation capacity, derived from the witnessed region.

    The pencil's column count is capacity + 2 and each window contributes
    (length - columns) data rows, so one third of the shortest aftermath
    window keeps roughly two rows per column from that window alone - the
    classical pencil operating point - while letting the mode count grow
    with the region actually witnessed instead of a fixed cap (a fixed
    cap was measured to fuse near-lying ring components and starve a
    family the eye still saw).  MODE_CAPACITY stays as the floor so a
    degenerate window layout never starves the estimator entirely.
    """
    aftermath_lengths = [stop - start for start, stop in
                         (window_plan.sync_tip, window_plan.back_porch)
                         if stop - start >= 8]
    if not aftermath_lengths:
        return MODE_CAPACITY
    return max(MODE_CAPACITY, min(aftermath_lengths) // 3)


def _blend_view_numerators(pairs, alphas):
    """The APPLIED numerator from the per-view pairs.

    pairs is (view, tap) with view 0 = fall, view 1 = rise; alphas the
    per-view significances.  The combined significance is the union
    a_r + (1 - a_r) a_f.
    """
    fall_alpha, rise_alpha = float(alphas[0]), float(alphas[1])
    combined = rise_alpha + (1.0 - rise_alpha) * fall_alpha
    if combined <= 0.0:
        return np.zeros(2), 0.0
    # RISE GOVERNS where it is significant, fading continuously to the
    # fall pair as its significance falls.  The rise-aligned view is
    # the phase-honest instrument for the sync aftermath; the
    # fall-aligned view's late lags are phase-smeared by sync-width
    # jitter, and an energy-weighted average of the two views' pairs
    # was measured to CANCEL (they fit the same component in antiphase
    # on the jitter-rotated view), leaving the ring uncorrected.  The
    # early failure of rise-priority (a painted dip right after the
    # edge) was a separate defect - the unconstrained hard-exclusion
    # gap, since replaced by the continuous reference-uncertainty
    # weighting - and the no-harm application scale now guards the
    # applied realization as a whole.
    blended = (rise_alpha * pairs[1]
               + (1.0 - rise_alpha) * fall_alpha * pairs[0]) / combined
    return blended, combined


def _combined_alpha(alphas):
    """Union significance of a per-view alpha pair (see the blend)."""
    return float(alphas[1] + (1.0 - alphas[1]) * alphas[0])


def _tracked_slots(state, ring_slots, smear_slots):
    total = ring_slots + smear_slots
    return state.setdefault("pole_track", {
        "poles": np.zeros(total, dtype=complex),
        # per-view numerators (view 0 = fall, view 1 = rise) and
        # per-view significances; the applied numerator is blended by
        # _blend_view_numerators
        "numerators": np.zeros((total, 2, 2)),
        "alpha": np.zeros((total, 2)),
        "energies": np.zeros((total, 2)),
        # per-slot occupancy age (hsEE3): the ESTIMATES of a stationary
        # mode converge as 1/age; see _update_pole_track
        "ages": np.zeros(total),
        # POLARITY-OWNED RING SLOTS (the user's model: each polarity's
        # transient carries its own ring product, completely isolated
        # from the other's - measured on the countdown tape as tip
        # 1.54 MHz vs back porch 2.05 served by ONE joint pole at
        # their 1.815 average, leaving the 1.46 MHz residual).  The
        # ring slots split into a FALL-owned range and a RISE-owned
        # range; matching, take-over and folding never cross the
        # boundary.  The fall family gets the odd slot: two regions
        # witness falling aftermaths (front porch and tip) against the
        # back porch alone for rises.
        "fall_ring_slots": (ring_slots + 1) // 2,
        "rise_ring_slots": ring_slots // 2,
        # slot layout: the first ring_slots hold oscillations, the rest
        # relaxations - recorded here so matching never depends on
        # module constants (the capacities are derived per geometry)
        "ring_slots": ring_slots,
        "smear_slots": smear_slots,
        "updates": 0,
    })


def _mode_family(pole):
    return ("ringing" if _mode_rotation(pole) > RINGING_MINIMUM_ROTATION
            else "smear")


def _update_pole_track(track, fitted_poles, fitted_numerators,
                       fitted_alpha, fitted_energies, window_length,
                       slow_horizon, fitted_polarities=None):
    """Blend this field's fit into the tracked model, continuously.

    Each fitted pole is matched to its nearest tracked slot within one
    DFT resolution cell of the residual window; matched slots blend
    position, numerator and significance at the slow horizon; unmatched
    tracked slots fade their significance toward zero (a mode is never
    discarded discretely - discrete membership flips were measured, in
    the previous pipeline, to invert the applied correction field to
    field).  New poles take over the weakest faded slot.
    """
    match_radius = (TRACK_MATCH_RESOLUTION_CELLS
                    * 2.0 * math.pi / max(window_length, 1))
    blend = 1.0 / min(track["updates"] + 1, max(slow_horizon, 1))
    # STATIONARY-CONVERGING ESTIMATES (hsEE3): a slot's pole and
    # numerator estimate a STATIONARY mode (measured: the countdown
    # ring identical in all thirds of a decode), so they blend at
    # 1/age - the running mean, converging as 1/N - while the fixed-
    # rate EMA (1/slow_horizon forever) moved an established pole up
    # to ~12% toward EVERY update's noisy fit: the pair's energy split
    # kept breathing, the shipped phasor turned ~85 degrees across a
    # decode, and the stationary ring shipped mis-phased.
    # SIGNIFICANCE (alpha, energies) keeps the responsive fixed-rate
    # blend: it is the evidence gate, and gating must follow the
    # decode; true recording changes are the reset machinery's job.
    ages = track.setdefault("ages", np.zeros(len(track["poles"])))
    # slots are allocated per family - the first ring_slots hold
    # oscillations (split into a FALL-owned and a RISE-owned range:
    # each polarity's ring products are isolated, so matching never
    # crosses the boundary), the rest relaxations - so the model's own
    # family limits hold in the track and not just within a single
    # field's fit
    fall_ring = track.get("fall_ring_slots", track["ring_slots"])

    def family_slots(pole, polarity):
        if _mode_family(pole) == "ringing":
            if polarity == 1:
                return range(fall_ring, track["ring_slots"])
            if polarity == 0:
                return range(fall_ring)
            return range(track["ring_slots"])
        return range(track["ring_slots"],
                     track["ring_slots"] + track["smear_slots"])
    if fitted_polarities is None:
        fitted_polarities = [None] * len(fitted_poles)
    matched_slots = set()
    for pole, numerator, alpha, energy, polarity in zip(
            fitted_poles, fitted_numerators, fitted_alpha,
            fitted_energies, fitted_polarities):
        allowed = [slot for slot in family_slots(pole, polarity)
                   if slot not in matched_slots]
        if not allowed:
            continue
        distances = np.full(len(track["poles"]), np.inf)
        distances[allowed] = np.abs(track["poles"][allowed] - pole)
        nearest = int(np.argmin(distances))
        # the matching radius is the RESOLVABILITY radius (hsEE2): a
        # fitted pole inside an existing slot's own spectral line is
        # that slot's mode and must blend into it - founding a sibling
        # inside the line is what bred the countdown tape's churn
        if distances[nearest] > _resolvability_radius(
                pole, track["poles"][nearest], match_radius):
            # take over the least significant unmatched slot (combined
            # significance - the union of both views' evidence)
            candidates = allowed
            if not candidates:
                continue
            nearest = min(candidates,
                          key=lambda slot: _combined_alpha(
                              track["alpha"][slot]))
            if _combined_alpha(track["alpha"][nearest]) \
                    > _combined_alpha(alpha):
                continue
            track["poles"][nearest] = pole
            track["numerators"][nearest] = numerator
            track["alpha"][nearest] = alpha * blend
            track["energies"][nearest] = energy * blend
            ages[nearest] = 1.0
            matched_slots.add(nearest)
            continue
        matched_slots.add(nearest)
        estimate_blend = 1.0 / (ages[nearest] + 1.0)
        track["poles"][nearest] = (
            track["poles"][nearest] * (1.0 - estimate_blend)
            + pole * estimate_blend)
        track["numerators"][nearest] = (
            track["numerators"][nearest] * (1.0 - estimate_blend)
            + numerator * estimate_blend)
        track["alpha"][nearest] = (
            track["alpha"][nearest] * (1.0 - blend) + alpha * blend)
        track["energies"][nearest] = (
            track["energies"][nearest] * (1.0 - blend) + energy * blend)
        ages[nearest] += 1.0
    for slot in range(len(track["alpha"])):
        if slot not in matched_slots:
            track["alpha"][slot] *= 1.0 - blend
            track["energies"][slot] *= 1.0 - blend

    # UNRESOLVABLE-PAIR FOLD: two same-family slots inside one
    # resolvability radius (the DFT cell floored by either mode's own
    # spectral linewidth, hsEE2 - see _resolvability_radius) are ONE
    # mode - the matcher above feeds each field's fitted pole to
    # whichever twin is nearer, so they split the evidence and the
    # farther twin starves while still shipping its stale numerators
    # (measured: the home tape's 2.26/2.34 MHz ring pair - the
    # long-tail member fell from significance 1.0 to 0.04 over 150
    # fields while its stale coefficients kept applying; the countdown
    # tape's stationary 1.8 MHz ring traded between three in-linewidth
    # siblings, the shipped phasor rotating >180 degrees while 78% of
    # the ring survived).  The fold preserves the shipped per-view sum
    # alpha x numerator exactly, so the applied correction does not
    # move at the fold (the temporal-stability rule); pairs farther
    # apart than either line still track separately (the recorded
    # beat-of-near-lying-rings lesson stands - below its own linewidth
    # a beat could never have been followed anyway).
    for family_slots in (range(fall_ring),
                         range(fall_ring, track["ring_slots"]),
                         range(track["ring_slots"],
                               track["ring_slots"]
                               + track["smear_slots"])):
        live = sorted(
            (slot for slot in family_slots
             if _combined_alpha(track["alpha"][slot]) > 0.0),
            key=lambda slot: _combined_alpha(track["alpha"][slot]),
            reverse=True)
        for index, keep in enumerate(live):
            if _combined_alpha(track["alpha"][keep]) <= 0.0:
                continue
            for fold in live[index + 1:]:
                if _combined_alpha(track["alpha"][fold]) <= 0.0:
                    continue
                if abs(track["poles"][keep]
                       - track["poles"][fold]) >= _resolvability_radius(
                        track["poles"][keep], track["poles"][fold],
                        match_radius):
                    continue
                keep_weight = float(np.sum(track["energies"][keep]))
                fold_weight = float(np.sum(track["energies"][fold]))
                total_weight = keep_weight + fold_weight
                if total_weight > 0.0:
                    track["poles"][keep] = (
                        track["poles"][keep] * keep_weight
                        + track["poles"][fold] * fold_weight
                    ) / total_weight
                for view in (0, 1):
                    summed = (
                        track["alpha"][keep][view]
                        * track["numerators"][keep][view]
                        + track["alpha"][fold][view]
                        * track["numerators"][fold][view])
                    merged_alpha = min(
                        1.0, track["alpha"][keep][view]
                        + track["alpha"][fold][view])
                    if merged_alpha > 0.0:
                        track["numerators"][keep][view] = \
                            summed / merged_alpha
                    track["alpha"][keep][view] = merged_alpha
                track["energies"][keep] += track["energies"][fold]
                # the surviving slot has observed its mode for as long
                # as its longest-lived constituent
                ages[keep] = max(ages[keep], ages[fold])
                track["poles"][fold] = 0.0
                track["numerators"][fold] = 0.0
                track["alpha"][fold] = 0.0
                track["energies"][fold] = 0.0
                ages[fold] = 0.0
    track["updates"] += 1
    return track


def _artifact_sections(track):
    """The tracked model as a list of section records worth using.

    Each section is one resonator: its pole, the numerator calibrated at
    the sync depth, the numerator's slope with transition amplitude
    (fitted from the picture-amplitude strata; zero until fitted), its
    significance, and its slot in the track.
    """
    sections = []
    for slot in range(len(track["poles"])):
        pole = track["poles"][slot]
        view_alphas = track["alpha"][slot]
        applied_pair, applied_alpha = _blend_view_numerators(
            track["numerators"][slot], view_alphas)
        if applied_alpha <= 0.0 or abs(pole) <= 0.0:
            continue
        sections.append({
            "slot": slot,
            "pole": pole,
            "numerator": applied_pair,
            "alpha": applied_alpha,
            # per-view data for the honest per-view predictions (each
            # measurement view is predicted by its own pair, exactly
            # what its rows fitted)
            "view_numerators": np.array(track["numerators"][slot]),
            "view_alphas": np.array(view_alphas),
        })
    return sections


def _view_sections(sections, view_index):
    """The sections as ONE VIEW witnesses them: each section's numerator
    and significance from that view's own fit rows.  Used for the
    per-view predictions (figures, tail reports, the detail witness) -
    the APPLICATION always uses the blended sections."""
    view = []
    for section in sections:
        pairs = section.get("view_numerators")
        alphas = section.get("view_alphas")
        if pairs is None or alphas is None:
            view.append(section)
            continue
        if alphas[view_index] <= 0.0:
            continue
        adapted = dict(section)
        adapted["numerator"] = pairs[view_index]
        adapted["alpha"] = float(alphas[view_index])
        view.append(adapted)
    return view


def _edge_drive_kernel(edge_width):
    """The shape of a real transition's drive: a band-limited impulse.

    Artifacts are excited by CHANGE, and a physical change is never a
    raw sample difference - it is spread over the measured edge width by
    the decoder's own band-limit.  The drive kernel is the analytic
    derivative of the reference edge (a Gaussian of the measured 10-90%
    width), normalized to unit sum, odd-length so its center falls on an
    integer sample.  It belongs to the model's realization: without it
    the fitted response grows without bound above the signal band and
    the stability margin needlessly crushes the in-band correction.
    """
    # the 10-90% width of an error-function step is 2.563 sigma
    sigma = max(edge_width, 0.5) / 2.563
    reach = int(math.ceil(4.0 * sigma))
    positions = np.arange(-reach, reach + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / float(np.sum(kernel))


def _artifact_frequency_response(sections, edge_kernel, grid_size=512):
    """R on the unit circle, for the stability margin and the report.

    Includes the band-limited drive kernel, exactly as the response is
    realized: R = (1 - z^-1) K(z) T(z).
    """
    angles = np.linspace(0.0, math.pi, grid_size)
    z_inverse = np.exp(-1j * angles)
    # the kernel is CENTERED on the transition - the drive fires during
    # the edge, spread symmetrically by the band-limit - so its spectrum
    # carries no bulk delay (the realization absorbs the half-kernel of
    # lookahead as the bulk timing advance the causality rule allows)
    center = (len(edge_kernel) - 1) / 2.0
    kernel_spectrum = np.zeros(grid_size, dtype=complex)
    for delay, coefficient in enumerate(edge_kernel):
        kernel_spectrum += coefficient * z_inverse ** (delay - center)
    ring_response = np.zeros(grid_size, dtype=complex)
    shelf_product = np.ones(grid_size, dtype=complex)
    for section in sections:
        pole = section["pole"]
        if abs(pole.imag) > 0.0:
            denominator = _denominator_of(pole)
            numerator = section["numerator"]
            numerator_values = numerator[0] + numerator[1] * z_inverse
            denominator_values = np.zeros(grid_size, dtype=complex)
            for power, coefficient in enumerate(denominator):
                denominator_values += coefficient * z_inverse ** power
            ring_response += (section["alpha"] * numerator_values
                              / denominator_values)
        else:
            # the smear shelves act on the SIGNAL as a product cascade
            # (the realization the application inverts stage by stage):
            # no drive kernel, no differentiator - keeping their
            # high-frequency deficit is exactly what lets the inversion
            # put the smeared energy back into the transition
            mix = section["alpha"] * section["numerator"][0]
            numerator, denominator = _shelf_filter_coefficients(
                pole.real, mix)
            shelf_product *= ((numerator[0] + numerator[1] * z_inverse)
                              / (denominator[0]
                                 + denominator[1] * z_inverse))

    # the EVEN mixes are nonlinear (magnitude-driven); for the
    # stability bound each is priced as the same relaxation driven
    # linearly through the drive operator, times 4/pi - the worst
    # fundamental gain of a full-wave rectifier - a conservative
    # ceiling, since rectification redistributes rather than
    # amplifies the change spectrum
    even_response = np.zeros(grid_size, dtype=complex)
    for section in sections:
        pole = section["pole"]
        if abs(pole.imag) > 0.0:
            continue
        pairs = section.get("view_numerators")
        alphas = section.get("view_alphas")
        if pairs is None or alphas is None:
            continue
        even_mix = max(abs(float(alphas[0]) * float(pairs[0][1])),
                       abs(float(alphas[1]) * float(pairs[1][1])))
        if even_mix != 0.0:
            even_response += ((4.0 / math.pi) * even_mix
                              * (1.0 - pole.real)
                              / (1.0 - pole.real * z_inverse))
    combined = (shelf_product * (
        1.0 + (1.0 - z_inverse) * kernel_spectrum * ring_response)
        + (1.0 - z_inverse) * kernel_spectrum * even_response)
    return angles, combined - 1.0


def _artifact_impulse(sections, length, edge_kernel):
    """The modeled artifact tail behind a unit step (its signature),
    driven through the same operator the application uses."""
    step = np.ones(length)
    step[: length // 8] = 0.0
    artifact = _modeled_artifact(step, sections, edge_kernel, 1.0)
    return artifact[length // 8:]


def _rise_fit_weights(state, window_plan, rise_edges, geometry,
                      effective_observations, deviation=None):
    """Per-lag weights for the RISE-aligned view, mirroring
    _fit_weights: inverse variance of the rise-aligned accumulated
    mean, zero outside the view's trustworthy span.

    Under rise alignment only the neighbourhood of the rise is sharp -
    the fall side is smeared by the line-to-line sync-width jitter - so
    the span reaches one front porch back from the rise and forward
    across the back porch.  The contamination trim compares against the
    rise-aligned dark-followed accumulation, exactly as the fall view
    trims against its own (the instructions' exclusion rule: structure
    uncorrelated with the known sync transitions never enters the fit).
    """
    variance = np.maximum(state["rise_interval_variance"], 1e-6)
    # same self-silencing guard as _fit_weights: the component's own
    # line-to-line jitter must not starve its mean evidence
    porch_variance = float(np.median(
        variance[slice(*window_plan.front_porch)]))
    variance = np.minimum(variance, 4.0 * porch_variance)
    variance = variance / max(effective_observations, 1.0)
    variance = variance + _reference_uncertainty(
        len(variance), {"sync_rise": rise_edges["sync_rise"]},
        deviation) ** 2
    weights = 1.0 / variance
    crossing = rise_edges["sync_rise"][0]
    frame_start = int(crossing - geometry.front_porch_samples)
    weights[:max(frame_start, 0)] = 0.0
    weights[window_plan.back_porch[1]:] = 0.0

    dark_mean = state.get("rise_dark_interval_mean")
    if dark_mean is not None and dark_mean.shape == weights.shape:
        dark_lines = max(state.get("dark_lines_average", 1.0), 1.0)
        dark_fields = max(state.get("rise_dark_fields_accumulated", 1), 1)
        dark_noise = np.sqrt(
            np.maximum(state["rise_dark_interval_variance"], 1e-6)
            / (dark_lines * dark_fields))
        difference = state["rise_interval_mean"] - dark_mean
        settled = window_plan.back_porch[0]
        offset = float(np.median(difference[settled:]))
        contamination = np.abs(difference - offset)
        confidence = dark_noise ** 2 / (dark_noise ** 2
                                        + contamination ** 2)
        weights[settled:] *= confidence[settled:]
    return weights


def _measure_detail_extent(rise_view, geometry):
    """How many symmetric detail taps the pre-rise witness certifies.

    The record-side delay-line detail enhancer of HQ-era decks is a
    symmetric (linear-phase) factor, and its pre-edge half is the
    acausal fingerprint no causal stage can produce - so the deviation
    BEFORE the sync rise, after the causal model's own prediction is
    removed, witnesses it alone.  The extent is the farthest pre-edge
    lag whose deviation clears three standard deviations of the
    accumulated mean; zero when nothing pre-edge is significant.
    """
    crossing, width = rise_view["edges"]["sync_rise"][:2]
    deviation = rise_view["mean"] - rise_view["modeled"]
    weights = rise_view["weights"]
    edge_start = int(math.floor(crossing - width))
    extent = 0
    for lag in range(1, edge_start):
        position = edge_start - lag
        if position < 0 or weights[position] <= 0.0:
            continue
        sigma = 1.0 / math.sqrt(weights[position])
        if abs(deviation[position]) > 3.0 * sigma:
            extent = lag
    return extent


def _solve_detail_taps(views, tap_count):
    """The symmetric detail factor's taps, directly from the pre-edge
    deviation profiles.

    Before its own edge a causal model contributes nothing, so the
    pre-edge residual (measured mean minus the causal model's
    prediction) is a pure witness of the symmetric factor - and for a
    step of height h through symmetric taps a_m, the deviation k
    samples before the edge is h * S_k with S_k = sum of a_m for
    m >= k.  The taps are therefore the first difference of the
    pre-edge profile divided by the step height: a closed form with
    one estimate per tap and no deconvolution (a least-squares tap
    solve was measured to produce large mutually-cancelling taps from
    the near-collinear columns of neighbouring delays).  The views'
    profiles combine with inverse-variance weights, and each tap is
    Wiener-shrunk by its own propagated variance, so an unwitnessed
    tap falls to zero instead of being thresholded.
    """
    if tap_count < 1:
        return np.zeros(0)
    sums = np.zeros(tap_count + 2)
    sum_weights = np.zeros(tap_count + 2)
    for view in views:
        edge_key = "sync_fall" if view["name"] == "fall" else "sync_rise"
        crossing, width, before_level, after_level = view["edges"][edge_key]
        step = after_level - before_level
        if abs(step) < 1e-6:
            continue
        deviation = view["mean"] - view["modeled"]
        edge_start = int(math.floor(crossing - width))
        for lag in range(1, tap_count + 1):
            position = edge_start - lag
            if position < 0 or view["weights"][position] <= 0.0:
                continue
            # S_lag in step units; the variance propagates from this
            # lag's own accumulated-mean variance (1 / weight)
            weight = view["weights"][position] * step * step
            sums[lag] += (deviation[position] / step) * weight
            sum_weights[lag] += weight
    witnessed = sum_weights > 0.0
    cumulative = np.where(witnessed,
                          sums / np.maximum(sum_weights, 1e-12), 0.0)
    variances = np.where(witnessed,
                         1.0 / np.maximum(sum_weights, 1e-12), 0.0)
    solved = np.zeros(tap_count)
    spread = np.zeros(tap_count)
    for delay in range(1, tap_count + 1):
        if not witnessed[delay]:
            continue
        solved[delay - 1] = cumulative[delay] - cumulative[delay + 1]
        spread[delay - 1] = variances[delay] + variances[delay + 1]
    return np.where(
        spread > 0.0,
        solved * solved ** 2 / (solved ** 2 + spread + 1e-18),
        0.0)


def _symmetric_detail_kernel(detail_taps):
    """The symmetric factor A as a centered kernel: taps a_m at +-m
    around a direct tap of 1 - 2 sum(a), so the kernel sums to exactly
    one (unity at DC by construction)."""
    taps = np.asarray(detail_taps, dtype=np.float64)
    center = np.array([1.0 - 2.0 * float(np.sum(taps))])
    return np.concatenate([taps[::-1], center, taps])


def _symmetric_factor_deviation(values, detail_taps):
    """(A - identity) applied to a sequence: the symmetric factor's own
    deviation.  Subtracted from the fit target so the causal sections
    are never asked to absorb the enhancer's post-edge half (its
    pre-edge half they cannot reach - causality - and an unmodeled
    post-edge half would bias every section numerator)."""
    taps = np.asarray(detail_taps, dtype=np.float64)
    if not len(taps) or not np.any(taps != 0.0):
        return np.zeros_like(values)
    kernel = _symmetric_detail_kernel(taps)
    half = len(taps)
    full = np.convolve(values, kernel)
    return full[half:half + len(values)] - values


def _detail_inverse_kernel(detail_taps, tolerance):
    """The symmetric factor's exact inverse as a finite even kernel.

    The inverse is the Neumann series sum over k of (delta - A)^k,
    summed until the next term's peak falls below the level tolerance.
    The series converges whenever the deviation kernel's absolute sum
    is below one (the rigorous sufficient condition, checked here; the
    Wiener-shrunk measured taps sit far inside it), and EVERY partial
    sum has exactly unit DC gain - (delta - A) sums to zero, so each
    power beyond k=0 sums to zero: level neutrality holds at any
    truncation, by construction.  Returns (kernel, center_index).
    """
    taps = np.asarray(detail_taps, dtype=np.float64)
    if not len(taps) or not np.any(taps != 0.0):
        return np.array([1.0]), 0
    a_kernel = _symmetric_detail_kernel(taps)
    half = len(taps)
    residual = -a_kernel
    residual[half] += 1.0
    if float(np.sum(np.abs(residual))) >= 1.0:
        # outside the guaranteed-convergence region: refuse rather than
        # risk a diverging inverse (an unmeasured condition, never seen
        # with shrunk taps, but the bound costs nothing to honor)
        return np.array([1.0]), 0
    inverse = np.array([1.0])
    term = np.array([1.0])
    while True:
        term = np.convolve(term, residual)
        grown = np.zeros(len(term))
        offset = (len(term) - len(inverse)) // 2
        grown[offset:offset + len(inverse)] = inverse
        inverse = grown + term
        if float(np.max(np.abs(term))) < tolerance:
            break
    return inverse, (len(inverse) - 1) // 2


def fit_artifact_model(state, geometry, window_plan, average_fields):
    """Fit the ghost, smear and ring model to the accumulated views.

    The whole interval is explained at once: a clean reference is built
    from the measured levels and transitions, the difference between the
    accumulated mean and that reference is the artifact record, and one
    set of components - the ghost column, ring resonators driven by the
    reference's own changes, and smear shelves, with state carried
    straight through all seven measurement points by the filtering
    itself - is fitted by weighted least squares over BOTH aligned
    views (the fall-anchored train and the rise-anchored back porch),
    each view whitened by its own accumulated noise.
    """
    slow_mean = state.get("slow_interval_mean")
    if slow_mean is None:
        return None
    anchor = window_plan.sync_fall_index

    excitation, edges, levels = _build_excitation(
        slow_mean, geometry, window_plan)
    if excitation is None:
        return None
    slow_horizon = SLOW_HORIZON_FACTOR * max(average_fields, 1)
    lines = max(state.get("usable_lines_average", 1.0), 1.0)
    accumulated_fields = min(state.get("slow_fields_accumulated", 1),
                             slow_horizon)
    # ORDER OF OPERATIONS (the requirements document's sequencing):
    # ghosting was applied BEFORE the recorder's smear and ringing, so
    # the ghost is handled first - its delayed-edge signature is a
    # shared least-squares column below, and the pencil certifies smear
    # and ring from the ghost-subtracted residual.  (Certifying smear
    # before rings as a second ordering stage was tried and reverted:
    # the home tape came out visibly worse.)
    ghost_track = state.get("ghost_track")
    reference = excitation
    # unit-mix ghost signature per view (zero until a lag is locked);
    # the mix itself is a least-squares coefficient below - integrating
    # scan increments instead was measured to run away on the countdown
    # tape (the reference absorbed a phantom ghost and every later
    # stage fitted garbage)
    ghost_signature_fall = _ghost_deviation(
        excitation, {"mix": 1.0,
                     "lag": (ghost_track or {}).get("lag", 0.0)})
    tracked_mix = float((ghost_track or {}).get("mix", 0.0))
    # the detail taps are solved from the pre-edge witness AFTER the
    # fit (causality keeps that witness pure); subtracting their
    # predicted deviation from the fit targets was measured to DIVERGE
    # (tap error -> section bias -> larger tap error, integrating
    # without bound), so the targets keep the symmetric wiggle and the
    # causal sections are simply never asked to explain the pre-edge
    # half they cannot reach
    detail_taps = state.get("detail_taps", np.zeros(0))
    fall_deviation = slow_mean - reference
    weights = _fit_weights(state, window_plan, edges,
                           lines * accumulated_fields, fall_deviation)
    fitted_lags = weights > 0.0

    # --- the rise-aligned view (measurement points 6 and 7) -------------
    # Under fall alignment the back porch is phase-smeared by the
    # line-to-line sync-width jitter - the averaging attenuates exactly
    # the back-porch ring being modeled - so when the rise-aligned
    # accumulation is available it OWNS the back-porch measurement, and
    # the fall view keeps the front porch, falling edge and tip.  Each
    # region is read by the view aligned at ITS reference transition;
    # the model itself still spans the whole interval in both views (the
    # instructions' carried-over state lives in the sections' filtering,
    # not in which aligned view reads which region), and the same lines
    # are never fitted twice through two alignments.
    rise_view = None
    rise_mean = state.get("rise_interval_mean")
    if rise_mean is not None:
        rise_excitation, rise_edges, rise_levels = _build_excitation(
            rise_mean, geometry, window_plan)
        if rise_excitation is not None:
            rise_accumulated = min(
                state.get("rise_fields_accumulated", 1), slow_horizon)
            rise_reference = rise_excitation
            rise_ghost_signature = _ghost_deviation(
                rise_excitation, {"mix": 1.0,
                                  "lag": (ghost_track or {}).get(
                                      "lag", 0.0)})
            rise_deviation = rise_mean - rise_reference
            rise_weights = _rise_fit_weights(
                state, window_plan, rise_edges, geometry,
                lines * rise_accumulated, rise_deviation)
            if np.any(rise_weights > 0.0):
                rise_view = {
                    "name": "rise",
                    "mean": rise_mean,
                    "excitation": rise_reference,
                    "edges": rise_edges,
                    "levels": rise_levels,
                    "weights": rise_weights,
                    "deviation": rise_deviation,
                }
    # SYNC-GENERATOR LEVEL-STEP EXCLUSION (the user's rule: structure not
    # correlated with the transients must not be matched).  The echo
    # consensus classifies persistent single-edge structure carrying a
    # level step as the pulse generator's own level step (measured: a
    # +0.5 IRE shift 2.3 us after the countdown tape's sync rise, a
    # +0.6 IRE shift 2.0 us after the pulse-and-bar tape's fall; the
    # picture-domain gauge shows NO echo at those lags after content
    # edges - matched as echoes they re-fired after every picture
    # transition as injected ghosting, and left in the fit targets
    # they force phantom rings).  Generator level-step spans carry ZERO fit
    # weight in every view that sees them - the whitened solve, the
    # closure projections and the ghost scan all honor weights - and
    # the pencil windows are split around them.  Nothing corrects a
    # generator level step: it is source content and ships untouched.
    level_steps = state.get("source_level_steps") or []
    if level_steps:
        # only the step's own TRANSITION span is unusable evidence -
        # the samples after it are level-shifted but otherwise intact,
        # and their ring testimony is kept through a free level-step
        # LEVEL COLUMN in the solve (zeroing a full kernel length per
        # generator level step was measured to wipe the countdown tape's back-
        # porch witness and drop its 1.7 MHz ring from the model
        # entirely, leaving a 1 IRE residual ring uncorrected)
        step_settle_reach = geometry.transition_settle_samples
        weights = weights.copy()
        for step_edge, step_lag in level_steps:
            if step_edge == 0:
                crossing = edges["sync_fall"][0]
            elif "sync_rise" in edges:
                crossing = edges["sync_rise"][0]
            else:
                continue
            span_start = int(round(crossing + step_lag))
            weights[max(span_start, 0):
                    span_start + step_settle_reach] = 0.0
            if step_edge == 1 and rise_view is not None:
                rise_crossing = rise_view["edges"]["sync_rise"][0]
                rise_start = int(round(rise_crossing + step_lag))
                rise_view["weights"] = rise_view["weights"].copy()
                rise_view["weights"][max(rise_start, 0):
                                     rise_start + step_settle_reach] = 0.0
        fitted_lags = weights > 0.0

    # PRE-RISE SOURCE SETTLE (the user's rulings: the energy before
    # the rising pulse is the deck's HQ detail enhancer's pre-shoot -
    # source content, to be fixed at the source - and a corrected
    # fall's aftermath overshoots and then SETTLES TO ZERO, so by the
    # rise nothing fall-attributed may still be firing).  Each edge's
    # own causal sections cannot reach their own pre-edge span, but
    # the FALL-attributed stages see the pre-rise span as post-edge
    # and were measured absorbing it: cancelling the home tape's
    # pre-rise HQ dip ADDED a +0.5-0.75 IRE rise at the tip's end,
    # and cancelling the countdown tape's pre-rise generator step
    # subtracted ~2 IRE at the rise base.  Removing the span's fit
    # WEIGHT instead was measured worse (home's added rise grew to
    # +2.0 IRE): unwitnessed, the smear/tail extrapolation through
    # the span is unconstrained and the closure chases the unmatched
    # source dip.  So the span keeps its weight and the fit TARGET is
    # zero there - the solve actively settles every fall-attributed
    # stage before the rise, and the source structure ships
    # untouched.  The span is the tracked acausal witness's own reach
    # (the symmetric detail factor's tap count - that witness reads
    # the raw means and modeled predictions directly and is
    # unaffected) plus the settle guard for the fall frame's
    # sync-width jitter smear.
    pre_rise_span = 0
    if len(detail_taps):
        pre_rise_span = (len(detail_taps)
                         + geometry.transition_settle_samples)
    settle_fall_span = settle_rise_span = None
    if pre_rise_span > 0 and "sync_rise" in edges:
        rise_crossing, rise_width = edges["sync_rise"][:2]
        settle_stop = max(
            int(math.ceil(rise_crossing - rise_width / 2.0)), 0)
        settle_start = max(
            int(math.floor(settle_stop - pre_rise_span)), 0)
        settle_fall_span = (settle_start, settle_stop)
        fall_deviation = fall_deviation.copy()
        fall_deviation[settle_start:settle_stop] = 0.0
        # SOFT settle: the span is the tip's quietest stretch, so its
        # inverse-variance weight is the heaviest in the interval, and
        # at full weight the zero target overpowers the aftermath
        # testimony it serves (measured on the home tape's decode: the
        # fall smear bent from +1.6 to +0.9 to satisfy it, the early-
        # tip misfit that created went to the echo scan as two giant
        # taps at -0.50/-0.24, and every picture stratum worsened 2x -
        # the same quietest-region dominance the closure block
        # documents for the front porch).  The settle witness is
        # capped at the fall aftermath's own median weight: stages
        # settle as far as the aftermath evidence allows, never at its
        # expense.
        aftermath_weights = weights[window_plan.sync_tip[0]:settle_start]
        witnessed_aftermath = aftermath_weights[aftermath_weights > 0.0]
        if witnessed_aftermath.size:
            settle_weight = float(np.median(witnessed_aftermath))
            weights = weights.copy()
            weights[settle_start:settle_stop] = np.minimum(
                weights[settle_start:settle_stop], settle_weight)
            fitted_lags = weights > 0.0
        if rise_view is not None:
            view_crossing, view_width = \
                rise_view["edges"]["sync_rise"][:2]
            view_stop = max(
                int(math.ceil(view_crossing - view_width / 2.0)), 0)
            view_start = max(
                int(math.floor(view_stop - pre_rise_span)), 0)
            settle_rise_span = (view_start, view_stop)
            rise_view["deviation"] = rise_view["deviation"].copy()
            rise_view["deviation"][view_start:view_stop] = 0.0
            # same soft cap in the rise frame, against the back
            # porch's own weights (that view's aftermath testimony)
            porch_weights = rise_view["weights"][
                window_plan.back_porch[0]:window_plan.back_porch[1]]
            witnessed_porch = porch_weights[porch_weights > 0.0]
            if witnessed_porch.size:
                rise_settle_weight = float(np.median(witnessed_porch))
                rise_view["weights"] = rise_view["weights"].copy()
                rise_view["weights"][view_start:view_stop] = np.minimum(
                    rise_view["weights"][view_start:view_stop],
                    rise_settle_weight)
    ghost_scan_weights = weights.copy()
    if settle_fall_span is not None:
        # the ghost/echo scan and the dense tails witness through
        # these weights, and their residual is built against the raw
        # mean - the source span is not their evidence (the runtime's
        # echo taps were measured matching it), and a tail column
        # supported only by the span falls to zero, so the tail
        # window ends on quiet ground
        ghost_scan_weights[settle_fall_span[0]:settle_fall_span[1]] = 0.0
    if rise_view is not None:
        weights = weights.copy()
        weights[window_plan.back_porch[0]:] = 0.0
        fitted_lags = weights > 0.0

    # The pencil runs on the ghost- and echo-subtracted residual (the
    # channel's delayed copies are sequenced FIRST per the requirements
    # document; an unmodeled echo pulse inside the certification
    # windows biased the pencil's modes - measured on the countdown
    # tape, whose 2.4 MHz Q16 back-porch ring certified as 2.1 MHz Q5
    # until the lag-34 echo was accounted, leaving the narrower high-
    # frequency half of the ring uncorrected in the picture); smear and
    # ring then certify from the same residual - subtracting the
    # tracked smear shelves before the pencil was tried and REVERTED at
    # Ethan's direction (the home tape came out visibly worse with it).
    # THE FIT'S DRIVE IS THE APPLICATION'S DRIVE, computed by the same
    # operator: the signal's change spread by the measured edge kernel.
    # Calibrating against a plain difference while applying against the
    # kernel-spread difference was measured to soften picture edges 9%
    # and fail the sync-flatness gauge.  (Hoisted above the pencil: the
    # echo subtraction below re-fires this same drive.)
    edge_kernel = _edge_drive_kernel(edges["sync_fall"][1])
    kernel_center = (len(edge_kernel) - 1) // 2
    # PER-POLARITY KERNELS (the user's diagnosis: the rise side is
    # modeled out of phase / slightly misaligned).  Every drive used to
    # be spread with the FALL edge's kernel; the rise's own measured
    # width was never used, so on a tape whose rise and fall widths
    # differ every rise-driven correction carried the wrong edge shape
    # - slightly wrong width, slightly wrong phase center - and the
    # rise side misfired on decode while the identical fall machinery
    # closed.  The rise kernel comes from the rise-aligned view's own
    # measured rise (sharp), falling back to the fall-aligned
    # measurement.
    rise_edge_width = float(
        rise_view["edges"]["sync_rise"][1] if rise_view is not None
        else edges["sync_rise"][1])
    rise_kernel = _edge_drive_kernel(rise_edge_width)

    def spread_change_with(reference_values, kernel):
        change = np.diff(reference_values,
                         prepend=reference_values[:1])
        center = (len(kernel) - 1) // 2
        full = scipy.signal.fftconvolve(change, kernel, mode="full")
        return full[center:center + len(reference_values)]

    def spread_change(reference_values):
        return spread_change_with(reference_values, edge_kernel)

    drive = spread_change(reference)
    rise_drive = (spread_change_with(rise_view["excitation"],
                                     rise_kernel)
                  if rise_view is not None else None)

    def spread_half_change(reference_values, kernel, polarity_sign):
        """One polarity's half-wave drive of a clean reference: the
        change rectified to the given sign, spread with that
        polarity's own edge kernel - the application's exact drive
        for that polarity (see _modeled_artifact's half drives)."""
        change = np.diff(reference_values,
                         prepend=reference_values[:1])
        half = (np.minimum(change, 0.0) if polarity_sign < 0
                else np.maximum(change, 0.0))
        center = (len(kernel) - 1) // 2
        full = scipy.signal.fftconvolve(half, kernel, mode="full")
        return full[center:center + len(reference_values)]

    # POLARITY-OWNED RING DRIVES: a fall-owned ring product is excited
    # by falling transients alone and fitted on the fall view's rows;
    # a rise-owned product by rising transients on the rise view's
    # rows.  Fitting each against its polarity's HALF drive is exactly
    # how the application drives it (the joint columns used each
    # view's FULL drive, so every fitted pair also answered the other
    # polarity's transient inside its view - part of the averaging
    # that merged the two isolated products).
    polarity_half_drives = [
        spread_half_change(reference, edge_kernel, -1),
        (spread_half_change(rise_view["excitation"], rise_kernel, +1)
         if rise_view is not None else None),
    ]

    # the PREVIOUS update's tracked echo taps (the same one-update
    # latency every tracked quantity has)
    echo_taps = [{"lag": lag,
                  "mix": state.get("echo_track", {}).get(lag, 0.0)}
                 for lag in state.get("echo_lags", [])]

    def echo_part(view_drive):
        if not echo_taps:
            return 0.0
        return sum(tap["mix"] * fractional_shift(view_drive, tap["lag"])
                   for tap in echo_taps if tap["mix"] != 0.0)

    pencil_fall = (fall_deviation
                   - tracked_mix * ghost_signature_fall
                   - echo_part(drive))
    pencil_rise = None
    if rise_view is not None:
        pencil_rise = (rise_view["deviation"]
                       - tracked_mix * rise_ghost_signature
                       - echo_part(rise_drive))

    # --- pole estimation on the flat-region residuals -------------------
    # each region is referenced to the same clean excitation the
    # numerator fit uses - NOT to its own "settled" level: a smear can
    # outlast the whole region (measured on the home tape, where the tip
    # never settles), and subtracting a level from an unsettled region
    # manufactures a spurious near-DC mode out of the leftover constant
    def level_step_cuts(view_name):
        """Absolute positions of locked sync-generator level steps in a view -
        the pencil must not read across them (a level shift inside a
        certification window manufactures phantom modes)."""
        cuts = []
        for step_edge, step_lag in (
                state.get("source_level_steps") or []):
            if view_name == "fall":
                if step_edge == 0:
                    crossing = edges["sync_fall"][0]
                elif "sync_rise" in edges:
                    crossing = edges["sync_rise"][0]
                else:
                    continue
                cuts.append(int(round(crossing + step_lag)))
            elif (view_name == "rise" and step_edge == 1
                    and rise_view is not None):
                crossing = rise_view["edges"]["sync_rise"][0]
                cuts.append(int(round(crossing + step_lag)))
        return sorted(cuts)

    # only the step's own transition span is excluded from the pencil
    # windows (matching the weight exclusion): a full-kernel reach left
    # the back-porch pieces too short to certify the countdown tape's
    # 1.7 MHz ring (one period is ~15 samples)
    step_settle_reach_samples = geometry.transition_settle_samples

    def region_residuals(deviation, region, view_name):
        """The region's residual, split around any generator level step inside
        it; pieces shorter than the pencil's minimum are dropped."""
        start, stop = region
        pieces = []
        cursor = start
        for cut in level_step_cuts(view_name):
            if start < cut < stop:
                pieces.append((cursor, cut))
                cursor = min(cut + step_settle_reach_samples, stop)
        pieces.append((cursor, stop))
        return [deviation[lo:hi] for lo, hi in pieces if hi - lo >= 8]

    noise_floor = math.sqrt(
        float(np.median(state["slow_interval_variance"]))
        / (lines * accumulated_fields))
    # each window carries its view of origin and its own noise floor -
    # the certification below decides which VIEW witnesses which mode
    window_catalog = []
    for region in (window_plan.front_porch, window_plan.sync_tip):
        for residual in region_residuals(pencil_fall, region, "fall"):
            window_catalog.append(("fall", residual, noise_floor))
    if rise_view is not None:
        rise_deviation = rise_view["deviation"]
        rise_floor = math.sqrt(
            float(np.median(state["rise_interval_variance"]))
            / (lines * min(state.get("rise_fields_accumulated", 1),
                           slow_horizon)))
        for rise_bp in region_residuals(pencil_rise,
                                        window_plan.back_porch,
                                        "rise"):
            window_catalog.append(("rise", rise_bp, rise_floor))
    else:
        for fall_bp in region_residuals(pencil_fall,
                                        window_plan.back_porch,
                                        "fall"):
            window_catalog.append(("fall", fall_bp, noise_floor))
    residual_windows = [residual for _, residual, _ in window_catalog]
    capacity = _pencil_capacity(window_plan)
    longest_window = max(
        (stop - start) for start, stop in (
            window_plan.front_porch, window_plan.sync_tip,
            window_plan.back_porch))
    maximum_ring_mhz = min(geometry.luma_lowpass_mhz,
                           geometry.sample_rate_mhz / 4.0
                           - geometry.sample_rate_mhz
                           / max(longest_window, 1))
    # the certification floor is the LARGER of the statistical noise
    # floor and the measured line-locked persistent floor (see
    # _line_locked_floor - the decay-drift mechanism); the guard is
    # the fastest witnessable ring's spectral half-width, so real
    # products' skirts never count as contamination
    floor_guard_mhz = 1.0 / (math.pi * max(
        geometry.sync_transition_samples / geometry.sample_rate_mhz,
        1e-3))
    persistent_floor = _line_locked_floor(
        residual_windows, geometry.sample_rate_mhz, maximum_ring_mhz,
        floor_guard_mhz)
    certification_floor = math.sqrt(noise_floor ** 2
                                    + persistent_floor ** 2)
    poles = estimate_decay_modes(residual_windows, capacity,
                                 certification_floor)

    # --- classification and capacity ------------------------------------
    # zero or more ringing components and up to SMEAR_COMPONENT_SLOTS
    # relaxation components, strongest first; the ring family may use
    # everything the pencil can certify (the instructions allow multiple
    # ring components, each with its own characteristics)
    ring_slots = max(capacity - SMEAR_COMPONENT_SLOTS, 1)
    smear_slots = SMEAR_COMPONENT_SLOTS

    # --- Wiener-deconvolution identification ---------------------------
    # each view's deviation deconvolved by its own reference change:
    # the frequencies still locked to the transient.  Peaks the pencil
    # did not deliver become ADDITIONAL ring candidates (the joint
    # least squares fits or rejects them); the spectra are stored for
    # the figure and the offline tool, and the corrected accumulation
    # is run through the same instrument as validation.
    rate = geometry.sample_rate_mhz
    fall_mean_variance = (np.maximum(state["slow_interval_variance"],
                                     1e-6)
                          / max(lines * accumulated_fields, 1.0))
    wiener = {"fall": _wiener_residual_spectrum(
        pencil_fall, reference, weights, fall_mean_variance, rate)}
    if rise_view is not None:
        rise_mean_variance = (
            np.maximum(state["rise_interval_variance"], 1e-6)
            / max(lines * min(state.get("rise_fields_accumulated", 1),
                              slow_horizon), 1.0))
        wiener["rise"] = _wiener_residual_spectrum(
            pencil_rise, rise_view["excitation"],
            rise_view["weights"], rise_mean_variance, rate)
    poles = list(poles)
    # The Wiener spectra are IDENTIFICATION AND VALIDATION ONLY.
    # Seeding the candidate list from spectral peaks was tried and
    # measured to certify burst-skirt and FM-remnant structure (line-
    # locked, so it survives averaging, but not transient-driven) with
    # runaway numerator pairs that drove the split-half cross-validation
    # negative - the model memorized accumulated noise.  Certification
    # stays with the time-domain pencil; the spectra tell the human
    # (figure, offline tool) what remains, before and after correction.

    # LENGTH bounds (RINGING_INSTRUCTIONS "Length"): both families are
    # only ever as long as the SYNC TIP - a ring or smear is almost
    # always completely observable inside the measurement area, so a
    # component whose time constant outlasts the tip's flat duration is
    # extrapolation, not measurement (measured: a tau ~ 4 us relaxation
    # certified from these windows once moved the sync tip a full IRE,
    # and tau ~ 60 us "rings" with runaway numerators throttled the
    # margin).  Extending the smear witness to the full fitted span was
    # also tried and REVERTED: the countdown tape's picture took
    # visible injection from the long shelf and the user's A/B chose
    # this bound.
    tip_flat_samples = (geometry.sync_pulse_samples
                        - geometry.sync_transition_samples)

    def witnessable(pole):
        magnitude = min(abs(pole), 1.0 - 1e-6)
        decay_samples = -1.0 / math.log(magnitude)
        # a component that settles inside the reference edge's own 10-90
        # width lives entirely in the excluded transition core - nothing
        # witnesses it, and anything attributed to it is aliased
        # edge-shape error (measured as fake fast modes fitted with huge
        # mutually-cancelling numerators)
        if decay_samples <= edges["sync_fall"][1]:
            return False
        return decay_samples <= tip_flat_samples

    poles = [pole for pole in poles if witnessable(pole)]

    # FAMILY CLASSIFICATION (requirements document):
    #  - a RING oscillates: at least one full cycle within its decay
    #    time (Q >= pi) AND at least one full cycle inside the longest
    #    witnessed window (f >= rate / window) - anything slower can
    #    never show a complete cycle in the data and is
    #    indistinguishable from a relaxation hump there (measured: a
    #    0.25 MHz "ring" certified from a 54-sample window - one cycle
    #    is 57 samples - was a smear product mis-corrected as ringing).
    #  - a SMEAR does not oscillate at all: REAL poles only.  A slow
    #    complex pole is handed to the smear family as its ENVELOPE
    #    (|pole|), never fitted with its rotation.
    minimum_ring_angle = 2.0 * math.pi / max(longest_window, 1)
    # upper ring bound: the smaller of the decoder's luminance low-pass
    # and the chroma subcarrier (the output rate is four times the
    # subcarrier by construction, so fsc = rate/4).  At and above the
    # subcarrier the accumulated mean carries burst, chroma and FM
    # carrier remnants - phase-locked to the line, so they survive
    # averaging, but NOT driven by the luma transient: fitting them as
    # rings is the instructions' excluded "not correlated with the
    # known input data" (measured: 4-6.6 MHz "rings" with runaway
    # numerators that throttled the real correction's margin).
    # (maximum_ring_mhz is hoisted above the raw pencil - the same
    # bound also scopes the persistent-floor measurement)
    maximum_ring_angle = (2.0 * math.pi * maximum_ring_mhz
                          / geometry.sample_rate_mhz)
    ringing = [pole for pole in poles
               if _mode_rotation(pole) > RINGING_MINIMUM_ROTATION
               and abs(np.angle(pole)) >= minimum_ring_angle
               and abs(np.angle(pole)) <= maximum_ring_angle]
    smear = [complex(min(abs(pole), 1.0 - 1e-6))
             for pole in poles if pole not in ringing]
    # the envelope handoff must still satisfy the relaxation bounds
    smear = [pole for pole in smear if witnessable(pole)]

    # SMEAR-FAMILY MERGE: one physical relaxation certified twice within
    # the windows' resolving power is ONE component - relaxations are
    # identified by settle time, and two closer than a factor of 3 are
    # numerically collinear over these windows (the same bound that
    # sizes SMEAR_COMPONENT_SLOTS).  The longer-decaying member
    # represents a cluster: it is the better-resolved measurement.
    # RINGS ARE NOT MERGED: the measured back-porch aftermath is a BEAT
    # of near-lying ring components (the phase of the envelope slides
    # tens of degrees over the window - a single ring cannot track it,
    # and correcting with the fused frequency was measured to ADD
    # antiphase ripple where the beat drifted out of phase).  The
    # near-collinearity risk a merge would have guarded is carried by
    # the unit ridge and the invertibility escalation in the solve.
    def merged_smears(family):
        kept = []
        for pole in sorted(family, key=abs, reverse=True):
            def settle(candidate):
                return -1.0 / math.log(min(abs(candidate), 1.0 - 1e-6))
            if all(abs(math.log(settle(pole) / settle(other)))
                   > math.log(3.0) for other in kept):
                kept.append(pole)
        return kept

    smear = merged_smears(smear)
    # after classification a smear pole is REAL and a ring pole is
    # complex, by construction - every downstream branch keys on that.
    # SEQUENTIAL MEASUREMENT - SMEAR FIRST, THEN RINGING (the user's
    # ordering): the raw pencil's job here is the SMEAR candidates
    # only; its ring-classified poles are DISCARDED.  Ring candidates
    # are re-estimated after the fitted envelope (smears, echo taps,
    # ghost, region levels) is removed, because the joint solve let
    # the collinear smear and low-frequency ring columns trade energy
    # arbitrarily - it booked oscillatory energy into the countdown
    # tape's smear mixes (the corrected transient moved the WRONG
    # way) - and it calibrated the rings against the un-smear-
    # corrected views while the application runs them on the smear-
    # corrected signal, so the better the smear correction got, the
    # more the ring correction under-shot (the user's eye on the home
    # tape: correcting the smear increased the visible ringing).
    smear_candidates = sorted(smear, key=abs,
                              reverse=True)[:smear_slots]


    # --- numerator fit: whitened least squares over both views ----------
    # (edge_kernel, spread_change and the view drives are hoisted above
    # the pencil - the echo subtraction and the columns share them)
    target = fall_deviation.copy()
    # (the amplitude law is SHIPPING-ONLY: it never touches the fit
    # target - the sync interval sits at the calibration anchor where
    # every gain is exactly 1, and feeding amplitude machinery back
    # into fit targets destabilized the fit, recorded three times)

    # every view contributes its own whitened residual rows to ONE
    # solve, so each section keeps a single numerator supported by all
    # the evidence while every region is read by its honestly-aligned
    # view
    view_drives = [drive]
    view_targets = [target]
    view_weights = [weights]
    # the flat-region levels are refined BY the fit: a median over a
    # region whose artifact has not finished settling is biased (the
    # home tape's tip never settles), and a biased reference level shows
    # up as a constant the resonator sections cannot represent - the
    # least squares then blows their numerators up chasing it.  One free
    # constant per flat region PER VIEW absorbs it.
    level_regions = (window_plan.front_porch, window_plan.sync_tip,
                     window_plan.back_porch)
    view_level_regions = [level_regions]
    if rise_view is not None:
        view_drives.append(rise_drive)
        view_targets.append(rise_deviation)
        view_weights.append(rise_view["weights"])
        # the rise view's trustworthy span covers the tip end and the
        # back porch
        view_level_regions.append((window_plan.sync_tip,
                                   window_plan.back_porch))

    # PER-VIEW NUMERATORS on shared poles: the two sync edges start at
    # different FM carrier levels (different transient-start regimes),
    # and one numerator forced across both views compromises both -
    # measured here as the back-porch ring predicted at half amplitude
    # and three samples late.  Each candidate pole therefore carries an
    # independent (b0, b1) pair per view, fitted only against that
    # view's rows; the pole (the physical resonance) stays shared.  The
    # APPLIED numerator blends the pairs by their own significances
    # with the rise pair governing (see _artifact_sections) - the
    # rise-aligned accumulation is the phase-honest instrument, and the
    # jitter-smeared fall view must not veto it.
    row_counts = [len(view_target) for view_target in view_targets]
    offsets = np.concatenate([[0], np.cumsum(row_counts)])
    stacked_roots = np.sqrt(np.concatenate(view_weights))
    stacked_target = np.concatenate(view_targets)
    total_rows = len(stacked_target)
    view_references = [reference]
    if rise_view is not None:
        view_references.append(rise_view["excitation"])
    echo_lags = state.get("echo_lags", [])

    # LEVEL-STEP SPANS: one free level constant per locked generator step,
    # from the step's position to its region's end, per witnessing
    # view - the level shift is absorbed exactly, and every sample
    # after the step keeps its ring testimony in the solve
    step_level_spans = []
    for step_edge, step_lag in (
            state.get("source_level_steps") or []):
        if step_edge == 0:
            crossing = edges["sync_fall"][0]
            region_end = window_plan.sync_tip[1]
        elif "sync_rise" in edges:
            crossing = edges["sync_rise"][0]
            region_end = window_plan.back_porch[1]
        else:
            continue
        position = int(round(crossing + step_lag))
        if position < region_end:
            step_level_spans.append((0, position, region_end))
        if step_edge == 1 and rise_view is not None:
            rise_crossing = rise_view["edges"]["sync_rise"][0]
            rise_position = int(round(rise_crossing + step_lag))
            if rise_position < window_plan.back_porch[1]:
                step_level_spans.append(
                    (1, rise_position, window_plan.back_porch[1]))

    def solve_pass(pass_candidates, pass_target, include_channel,
                   pass_roots, pass_polarities=None):
        """One whitened least-squares pass over the stacked views.

        `include_channel` adds the channel-copy columns (echo taps and
        the ghost): they belong to the SMEAR pass - delayed copies are
        sequenced first per the requirements document - while the ring
        pass fits resonators alone on the envelope-subtracted residual.
        `pass_polarities` (parallel to the candidates; 0 = fall-owned,
        1 = rise-owned, None = both views) restricts a ring candidate
        to its own polarity: only the owning view's columns are built,
        on that polarity's HALF drive, and the other view's pair rests
        at zero under the ridge - the isolated-product rule.
        Returns the per-candidate fitted quantities, the channel mixes,
        the per-region level corrections, and the pass's own unwhitened
        prediction over the stacked rows (what the next pass must not
        see again).
        """
        if pass_polarities is None:
            pass_polarities = [None] * len(pass_candidates)
        columns = []
        # ONE aftermath carrier: the longest-decay smear candidate (the
        # better-resolved cluster representative, the same ranking
        # merged_smears uses).  The polarity-asymmetric aftermath is one
        # phenomenon; giving every smear pole its own aftermath columns
        # was measured to split into large mutually-cancelling pairs
        # (the home tape's +-5 and -7/+5 even splits, the pulse-and-bar
        # tape's governor collapse).
        aftermath_pole = max(
            (pole for pole in pass_candidates if abs(pole.imag) == 0.0),
            key=abs, default=None)
        for pole, pole_polarity in zip(pass_candidates,
                                       pass_polarities):
            # four columns per candidate: fall then rise, each zero
            # outside its own view's rows; with no rise view the rise
            # columns are all zero and their coefficients rest at zero
            # under the ridge.  A RING contributes a drive-coupled
            # (b0, b1) resonator pair; a SMEAR (real pole) contributes
            # its per-polarity aftermath column (the half-wave
            # relaxation) and a zero placeholder so the layout stays
            # uniform.  A polarity-OWNED ring keeps the uniform layout
            # with only its owning view's columns live, built on that
            # polarity's half drive.
            if abs(pole.imag) > 0.0:
                denominator = _denominator_of(pole)
                for view_index in (0, 1):
                    for numerator in ([1.0], [0.0, 1.0]):
                        column = np.zeros(total_rows)
                        owned = (pole_polarity is None
                                 or view_index == pole_polarity)
                        if view_index < len(view_drives) and owned:
                            column_drive = view_drives[view_index]
                            if pole_polarity is not None:
                                column_drive = \
                                    polarity_half_drives[view_index]
                            start = int(offsets[view_index])
                            stop = int(offsets[view_index + 1])
                            column[start:stop] = scipy.signal.lfilter(
                                numerator, denominator,
                                column_drive)
                        columns.append(column)
            else:
                # per-view witnessability: the rise view's trustworthy
                # span is short (one front porch before the rise
                # through the back porch), and a relaxation slower than
                # that span is indistinguishable there from the level
                # constant and the ghost rectangle - a three-way
                # degeneracy the least squares was measured to split
                # into large mutually-cancelling coefficients.  A long
                # shelf therefore keeps only its fall-view column (the
                # fall view spans the whole interval), and the blend
                # follows the witnessing view automatically through the
                # dead pair's zero significance.
                decay_samples = -1.0 / math.log(min(abs(pole.real),
                                                    1.0 - 1e-6))
                # the rise view's trustworthy reach: one front porch
                # before the rise crossing through the end of the back
                # porch (the same span _rise_fit_weights trusts)
                rise_span = (window_plan.back_porch[1]
                             - (window_plan.sync_fall_index
                                + geometry.sync_pulse_samples
                                - geometry.front_porch_samples))
                # each smear pair is [retired odd slot (always zero),
                # the per-polarity aftermath mix].  The odd shelf
                # column was RETIRED: on a single-polarity span the
                # signed change and the rectified change are the same
                # column, so odd-plus-even fitted a two-dimensional
                # aftermath with four coefficients - an exact
                # degeneracy the least squares split into large
                # cancelling pairs the moment the rectification
                # threshold stopped hiding it.  The per-polarity
                # aftermath subsumes the shelf (a linear shelf is the
                # antisymmetric special case), and shipping it through
                # the drive-coupled half-wave path keeps the whole
                # correction inside the measured edge kernel's band
                # (the odd cascade's exact inverse carried a flat
                # unwitnessed boost to Nyquist - the injected
                # 2.8-3.2 MHz energy in the home tape's residual
                # spectrum).
                for view_index in (0, 1):
                    odd_column = np.zeros(total_rows)
                    even_column = np.zeros(total_rows)
                    witnessed = (view_index == 0
                                 or decay_samples <= rise_span)
                    if (view_index < len(view_references) and witnessed
                            and pole == aftermath_pole):
                        start = int(offsets[view_index])
                        stop = int(offsets[view_index + 1])
                        # the reference is the clean excitation - no
                        # noise, no debias (see
                        # _even_relaxation_response).  Each polarity's
                        # column spreads with ITS OWN edge kernel -
                        # the application drives the rising mix with
                        # the rise kernel, and fitting it with the
                        # fall kernel left a fit/apply shape mismatch
                        # that showed as residual smear on the rising
                        # pulse (the user's eye, home tape)
                        even_column[start:stop] = \
                            _even_relaxation_response(
                                view_references[view_index], pole.real,
                                edge_kernel if view_index == 0
                                else rise_kernel,
                                -1 if view_index == 0 else +1)
                        if view_index == 0:
                            # the falling mix calibrates on the SYNC
                            # fall's aftermath only: the reference's
                            # ACTIVE fall also drives this column, and
                            # letting the front porch (the picture-
                            # regime witness, a different transient-
                            # start level) set the sync mix was
                            # measured to fit junk on the home tape
                            even_column[start:
                                        start
                                        + window_plan.sync_fall_index] \
                                = 0.0
                    columns.append(odd_column)
                    columns.append(even_column)
        if include_channel:
            # ECHO TAP columns, one per consensus-locked lag (a channel
            # property, identical in both views like the ghost): the
            # unit-mix DELAYED DRIVE - each transition's kernel-spread
            # change re-fired at the locked lag.  Drive-coupled, so the
            # echo correction stays inside the measured kernel band;
            # strictly causal (lags clear the settle span); zero-sum
            # drive keeps it exactly level-neutral.
            for echo_lag in echo_lags:
                echo_column = np.zeros(total_rows)
                for view_index, view_drive in enumerate(view_drives):
                    start = int(offsets[view_index])
                    stop = int(offsets[view_index + 1])
                    echo_column[start:stop] = fractional_shift(
                        view_drive, echo_lag)
                columns.append(echo_column)
            # ONE shared ghost column (a channel property, identical in
            # both views): the unit-mix delayed-edge signature, its
            # least-squares coefficient IS the ghost mix.  All zero
            # until the scan locks a lag - the unit ridge then rests
            # the coefficient at zero.
            ghost_column = np.zeros(total_rows)
            ghost_column[offsets[0]:int(offsets[1])] = \
                ghost_signature_fall
            if rise_view is not None:
                ghost_column[int(offsets[1]):int(offsets[2])] = \
                    rise_ghost_signature
            columns.append(ghost_column)
        for view_index, regions in enumerate(view_level_regions):
            for start, stop in regions:
                indicator = np.zeros(total_rows)
                indicator[offsets[view_index] + start:
                          offsets[view_index] + stop] = 1.0
                columns.append(indicator)
        # the level-step columns ride AFTER the standard region
        # levels (the extraction's zip over level regions ignores the
        # tail, so the refined excitation never absorbs them - the
        # step stays part of the measured signal, absorbed only inside
        # the solve)
        for view_index, start, stop in step_level_spans:
            if view_index >= len(view_targets):
                continue
            indicator = np.zeros(total_rows)
            indicator[offsets[view_index] + start:
                      offsets[view_index] + stop] = 1.0
            columns.append(indicator)
        design = np.stack(columns, axis=1) * pass_roots[:, None]
        observed = pass_target * pass_roots
        moment = design.T @ observed
        gram = design.T @ design
        # unit-information ridge IN WHITENED UNITS: a parameter the
        # data cannot distinguish above one noise unit shrinks toward
        # zero instead of ballooning against a near-collinear partner.
        # For any witnessed section the base ridge is negligible -
        # relative size 1/energy, and real sections' energies run in
        # the thousands of whitened units.
        ridge = np.ones(design.shape[1])

        def unpack(solution):
            """Per-candidate (view pairs, alphas, energies)."""
            unpacked = []
            for index, pole in enumerate(pass_candidates):
                pairs = np.zeros((2, 2))
                alphas = np.zeros(2)
                energies = np.zeros(2)
                # in whitened units the expected noise energy of a fit
                # equals its free parameter count - two per view for
                # both families (a ring's b0/b1, a smear's odd shelf
                # mix and even magnitude-driven mix)
                parameter_count = 2.0
                for view_index in (0, 1):
                    first = 4 * index + 2 * view_index
                    pair = solution[first:first + 2]
                    contribution = (design[:, first:first + 2]
                                    @ solution[first:first + 2])
                    energy = float(contribution @ contribution)
                    alphas[view_index] = energy / (energy
                                                   + parameter_count)
                    energies[view_index] = energy
                    pairs[view_index] = pair
                unpacked.append((pairs, alphas, energies))
            return unpacked

        # INVERTIBILITY IN THE FIT: the sections are additive parts of
        # a channel whose exact inverse 1/(1 + R) ships, so no
        # admissible fit may push |R| past the stage-D stability limit
        # - otherwise the margin scale crushes the WHOLE correction to
        # protect the inverse from one runaway section.  When the
        # realized response exceeds the limit, the single worst
        # section's ridge escalates and the whole system re-solves,
        # letting the remaining sections re-absorb whatever real
        # structure that section was carrying; iteration is bounded
        # because each escalation shrinks the offender geometrically
        # toward zero.
        for _ in range(20):
            normal = gram + np.diag(ridge)
            solution = np.linalg.solve(normal, moment)
            trial_sections = []
            for index, (pairs, alphas, energies) in enumerate(
                    unpack(solution)):
                applied_pair, applied_alpha = _blend_view_numerators(
                    pairs, alphas)
                trial_sections.append({
                    "pole": pass_candidates[index],
                    "numerator": applied_pair,
                    "alpha": applied_alpha, "slot": index,
                    # the application is per-polarity (each view's pair
                    # answers its own polarity's transients), so the
                    # invertibility check must bound the WORST view's
                    # realized response, not the blend's
                    "view_numerators": pairs, "view_alphas": alphas,
                })
            worst_index = None
            worst_peak = 0.0
            total_peak = 0.0
            if trial_sections:
                for view_index in (0, 1):
                    per_view = _view_sections(trial_sections,
                                              view_index)
                    if not per_view:
                        continue
                    _, total = _artifact_frequency_response(
                        per_view, edge_kernel)
                    total_peak = max(total_peak,
                                     float(np.max(np.abs(total))))
            if total_peak > ARTIFACT_MAGNITUDE_LIMIT:
                # the biggest standalone response carries the excess -
                # even when no single section exceeds the limit alone,
                # shrinking the largest one moves the total down
                # fastest
                for index, section in enumerate(trial_sections):
                    for view_index in (0, 1):
                        alone_view = _view_sections([section],
                                                    view_index)
                        if not alone_view:
                            continue
                        _, alone = _artifact_frequency_response(
                            alone_view, edge_kernel)
                        peak = float(np.max(np.abs(alone)))
                        if peak > worst_peak:
                            worst_peak = peak
                            worst_index = index
            if worst_index is None:
                break
            ridge[4 * worst_index: 4 * worst_index + 4] *= 4.0
        pass_poles = []
        pass_pairs = []
        pass_alphas = []
        pass_energies = []
        for index, (pairs, alphas, energies) in enumerate(
                unpack(solution)):
            pass_poles.append(pass_candidates[index])
            pass_pairs.append(pairs)
            pass_alphas.append(alphas)
            pass_energies.append(energies)
        echo_base = 4 * len(pass_candidates)
        if include_channel:
            pass_echo = [float(solution[echo_base + index])
                         for index in range(len(echo_lags))]
            pass_ghost = float(solution[echo_base + len(echo_lags)])
            level_start = echo_base + len(echo_lags) + 1
        else:
            pass_echo = [0.0] * len(echo_lags)
            pass_ghost = 0.0
            level_start = echo_base
        pass_levels = np.array(solution[level_start:], dtype=np.float64)
        prediction = np.stack(columns, axis=1) @ solution
        return (pass_poles, pass_pairs, pass_alphas, pass_energies,
                pass_echo, pass_ghost, pass_levels, prediction)

    # PASS 1 - THE ENVELOPE: smears, echo taps, ghost and region
    # levels, fitted on the raw views.
    (smear_poles, smear_pairs, smear_alphas, smear_energies,
     fitted_echo_mixes, fitted_ghost_mix, smear_levels,
     envelope_prediction) = solve_pass(smear_candidates,
                                       stacked_target, True,
                                       stacked_roots)

    # PASS 2 - THE RINGS, measured on what the fitted envelope leaves.
    # The ring pencil re-runs on the envelope-subtracted windows, so
    # the ring candidates certify against the residual the rings will
    # actually correct: the application already runs the ring fixed
    # point on the smear-corrected signal, and the measurement now
    # matches it.
    ring_stacked = stacked_target - envelope_prediction
    # Kept undeflated for the joint re-solve below: the peel deflates its
    # working copy pass by pass, and the final amplitudes have to be fitted
    # against what the rings actually have to explain, not against what the
    # previous peel left of it.
    ring_target_whole = ring_stacked.copy()

    def ring_windows_of(residual_stacked, polarity):
        """The certification windows over the CURRENT ring residual,
        for ONE polarity: the fall's aftermath lives in the front
        porch and tip (fall view), the rise's in the back porch (rise
        view) - each polarity's product certifies from its own
        aftermath alone (the isolated-product rule)."""
        fall_residual = residual_stacked[int(offsets[0]):
                                         int(offsets[1])]
        windows = []
        if polarity == 0:
            for region in (window_plan.front_porch,
                           window_plan.sync_tip):
                windows.extend(region_residuals(fall_residual, region,
                                                "fall"))
        elif rise_view is not None:
            rise_residual = residual_stacked[int(offsets[1]):
                                             int(offsets[2])]
            windows.extend(region_residuals(rise_residual,
                                            window_plan.back_porch,
                                            "rise"))
        else:
            windows.extend(region_residuals(fall_residual,
                                            window_plan.back_porch,
                                            "fall"))
        return windows

    # RING PEELING (the user's doctrine: the corrected sync area must
    # come out FLAT - every artifact profile here is measured and
    # therefore correctable, and each artifact is MULTIPLE ADDITIVE
    # components, each with its own characteristics, tuned
    # independently).  One joint pencil pass merges near-lying
    # families and under-fits the weaker ones; instead the ring
    # estimation REPEATS on its own residual: each pass certifies and
    # fits the components the previous passes left, with its own
    # least-squares amplitudes, until nothing more certifies or the
    # slot budget is spent.  A peeled candidate must be a NEW mode -
    # one track resolution cell away from every component already
    # fitted this update.
    ring_poles = []
    ring_pairs = []
    ring_alphas = []
    ring_energies = []
    ring_polarities = []
    ring_levels = None
    dedup_radius = (TRACK_MATCH_RESOLUTION_CELLS * 2.0 * math.pi
                    / max(longest_window, 1))
    # RESIDUAL-AUDIT RESEEDING (the user-approved closed loop): a ring
    # still VISIBLE in the corrected accumulated views is, by the
    # reference rule, not yet subtracted by the correct amount - so
    # the previous update's shipped correction is realized on the
    # accumulated means, each polarity's corrected aftermath is
    # scanned with the same single-ring detector the offline audit
    # trusts, and any ring clearing the visibility floor seeds THIS
    # update's peel ahead of the pencil (which then deduplicates
    # against it).  The least squares accepts or rejects the seed
    # honestly and the track's EMA smooths the loop.  This recovers
    # products the pencil seeds off-frequency (the countdown tape's
    # 1.40 MHz product seeded at 1.29-1.34 rode at zero significance
    # for rounds while its ring stayed visible - frequency refinement
    # inside the additive fit was tried twice and REFUTED; the honest
    # detector reads the decoded truth instead) and products the
    # certification floor under-admits while their residual is
    # visible.
    reseeds = []
    if state.get("model") is not None \
            and _correction_strength(state, average_fields) > 0.0:
        settle_reach = geometry.transition_settle_samples
        decay_bounds = (edges["sync_fall"][1] + 1.0,
                        tip_flat_samples - 1.0)
        decay_grid = np.geomspace(decay_bounds[0],
                                  max(decay_bounds[1],
                                      decay_bounds[0] + 1.0), 4)
        angle_step = math.pi / max(longest_window, 1)
        scan_views = [(0, slow_mean,
                       int(math.ceil(edges["sync_fall"][0]
                                     + settle_reach)),
                       window_plan.sync_tip[1])]
        if rise_view is not None:
            scan_views.append(
                (1, rise_view["mean"],
                 int(math.ceil(rise_view["edges"]["sync_rise"][0]
                               + settle_reach)),
                 window_plan.back_porch[1]))
        for seed_polarity, view_mean, scan_start, scan_stop \
                in scan_views:
            corrected_view, _ = apply_shipping_correction(
                view_mean, state, geometry)
            segment = corrected_view[scan_start:scan_stop]
            if len(segment) < 8:
                continue
            segment = segment - float(np.median(segment))
            seed_lags = np.arange(len(segment), dtype=np.float64)
            # BAND-EDGE GUARD 1: a seed's frequency must complete one
            # full cycle within ITS OWN scanned segment - the global
            # minimum comes from the LONGEST region, and seeding from
            # a shorter corrected segment below its own cycle bound
            # let the loop pump a marginal near-minimum product (the
            # tracked slot walked to the band minimum at the decay
            # cap, alpha 1.00, consuming two slots)
            seed_minimum_angle = max(minimum_ring_angle,
                                     2.0 * math.pi / len(segment))
            best_seed = None
            for angle in np.arange(seed_minimum_angle,
                                   maximum_ring_angle, angle_step):
                for decay_samples in decay_grid:
                    magnitude = math.exp(-1.0 / decay_samples)
                    envelope = magnitude ** seed_lags
                    basis = np.stack(
                        [envelope * np.cos(angle * seed_lags),
                         envelope * np.sin(angle * seed_lags)],
                        axis=1)
                    gram = basis.T @ basis
                    moment = basis.T @ segment
                    try:
                        pair = np.linalg.solve(
                            gram + np.eye(2) * 1e-9, moment)
                    except np.linalg.LinAlgError:
                        continue
                    onset = float(np.hypot(*pair))
                    explained = float(moment @ pair)
                    if best_seed is None or explained > best_seed[0]:
                        best_seed = (explained, angle, magnitude,
                                     onset)
            # BAND-EDGE GUARD 2 (the user's rule: fix only the
            # oscillating components): the seed must explain the
            # corrected leftover BETTER than a non-oscillatory
            # exponential approach over the same decays - the
            # de-emphasis shelf residue fits a near-DC "ring"
            # otherwise, and the model is currently the layer
            # compensating that shelf
            best_approach = 0.0
            for decay_samples in decay_grid:
                envelope = math.exp(-1.0 / decay_samples) ** seed_lags
                envelope_energy = float(envelope @ envelope)
                if envelope_energy <= 0.0:
                    continue
                overlap = float(envelope @ segment)
                best_approach = max(best_approach,
                                    overlap * overlap
                                    / envelope_energy)
            if best_seed is not None \
                    and best_seed[3] >= VISIBILITY_FLOOR_IRE \
                    and best_seed[0] > best_approach:
                seed_pole = best_seed[2] * np.exp(1j * best_seed[1])
                # RESOLVABILITY GUARD 3 (hsEE2): a seed inside a LIVE
                # tracked slot's own spectral line is that slot's mode
                # already - its apparent frequency offset is the
                # corrected residual's bias, i.e. the corrector's own
                # output steering the model, and re-entering it was
                # measured to WALK the countdown tape's 2 MHz slot
                # from 1.83 to 2.09 MHz across one decode while the
                # stationary ring shipped nearly uncorrected.
                # Genuinely new modes, outside every live line, still
                # seed.
                track = state.get("pole_track")
                already_owned = False
                if track is not None:
                    seed_radius = (TRACK_MATCH_RESOLUTION_CELLS
                                   * 2.0 * math.pi
                                   / max(longest_window, 1))
                    fall_ring = track.get("fall_ring_slots",
                                          track["ring_slots"])
                    family = (range(fall_ring) if seed_polarity == 0
                              else range(fall_ring,
                                         track["ring_slots"]))
                    for slot in family:
                        if _combined_alpha(
                                track["alpha"][slot]) <= 0.0:
                            continue
                        if abs(track["poles"][slot] - seed_pole) \
                                < _resolvability_radius(
                                    seed_pole, track["poles"][slot],
                                    seed_radius):
                            already_owned = True
                            break
                if not already_owned:
                    reseeds.append((seed_polarity, seed_pole))

    # POLARITY-OWNED PEELING: each polarity certifies from its own
    # aftermath windows against its own slot budget, and deduplication
    # never crosses polarities - two products at the same frequency on
    # opposite polarities are two isolated components (the user's
    # model), not one
    fall_ring_budget = (ring_slots + 1) // 2
    polarity_budgets = ((0, fall_ring_budget),
                        (1, ring_slots - fall_ring_budget))
    for ring_polarity, polarity_budget in polarity_budgets:
        accepted_here = [pole for pole, owner
                         in zip(ring_poles, ring_polarities)
                         if owner == ring_polarity]
        for _ in range(max(int(capacity), 1)):
            remaining_slots = polarity_budget - len(accepted_here)
            if remaining_slots <= 0:
                break
            peel_candidates = []
            # the audit reseeds lead their polarity's FIRST pass; the
            # pencil deduplicates against them below
            if not accepted_here:
                for seed_polarity, seed_pole in reseeds:
                    if seed_polarity != ring_polarity:
                        continue
                    if not (witnessable(seed_pole)
                            and _mode_rotation(seed_pole)
                            > RINGING_MINIMUM_ROTATION
                            and abs(np.angle(seed_pole))
                            >= minimum_ring_angle
                            and abs(np.angle(seed_pole))
                            <= maximum_ring_angle):
                        continue
                    if any(abs(seed_pole - accepted) < dedup_radius
                           for accepted in peel_candidates):
                        continue
                    peel_candidates.append(seed_pole)
            peel_windows = ring_windows_of(ring_stacked, ring_polarity)
            peel_floor = math.sqrt(
                noise_floor ** 2 + _line_locked_floor(
                    peel_windows, geometry.sample_rate_mhz,
                    maximum_ring_mhz, floor_guard_mhz) ** 2)

            # STOP AT THE LIMIT, not when the budget runs out.
            #
            # The peel used to end only when its slot budget was spent or
            # no further candidate certified. Neither is a statement about
            # the residual, and the derivation's own rule is that the
            # process ends when what is left is at the noise floor - "we
            # cannot estimate random noise, that is physically not
            # possible, this process ends there".
            #
            # So: if what this polarity's windows still hold is already at
            # or under the floor those same windows imply, there is nothing
            # left to certify and the remaining slots are not spent looking.
            # The floor is the one computed for this peel, so the test
            # tightens as the peel proceeds rather than being fixed at the
            # first pass's estimate.
            if peel_windows:
                remaining = math.sqrt(float(np.mean(
                    [float(np.mean(np.asarray(window, dtype=np.float64) ** 2))
                     for window in peel_windows])))
                if remaining <= peel_floor:
                    state.setdefault("notes", []).append(
                        "peel stopped at the floor: %.4f IRE left against a "
                        "floor of %.4f" % (remaining, peel_floor))
                    break
            for pole in sorted(
                    estimate_decay_modes(peel_windows, capacity,
                                         peel_floor),
                    key=abs, reverse=True):
                if not (witnessable(pole)
                        and _mode_rotation(pole)
                        > RINGING_MINIMUM_ROTATION
                        and abs(np.angle(pole)) >= minimum_ring_angle
                        and abs(np.angle(pole)) <= maximum_ring_angle):
                    continue
                if any(abs(pole - accepted) < dedup_radius
                       for accepted in accepted_here
                       + peel_candidates):
                    continue
                peel_candidates.append(pole)
                if len(peel_candidates) >= remaining_slots:
                    break
            if not peel_candidates:
                break
            (peel_poles, peel_pairs, peel_alphas, peel_energies,
             _peel_echo, _peel_ghost, peel_levels,
             peel_prediction) = solve_pass(
                peel_candidates, ring_stacked, False, stacked_roots,
                [ring_polarity] * len(peel_candidates))
            ring_poles.extend(peel_poles)
            ring_pairs.extend(peel_pairs)
            ring_alphas.extend(peel_alphas)
            ring_energies.extend(peel_energies)
            ring_polarities.extend([ring_polarity] * len(peel_poles))
            accepted_here.extend(peel_poles)
            ring_levels = (peel_levels if ring_levels is None
                           else ring_levels + peel_levels)
            # the next peel certifies against what THIS peel left
            ring_stacked = ring_stacked - peel_prediction

    # THE AMPLITUDES ARE RE-SOLVED AS A WHOLE.
    #
    # The peel above IDENTIFIES the components, and identifying them one at
    # a time is right - a component only becomes visible once the larger
    # ones are out of the way. But each peel also FITTED its amplitudes
    # against a residual the previous peels had already deflated, so every
    # amplitude carried the earlier ones' errors, and two components close
    # enough to overlap within the window's resolution were mis-split: the
    # first fitted took energy belonging to the second, and the second was
    # then fitted to a residual distorted by that error.
    #
    # Ethan's rule, and the reason this is here: "Since each correction will
    # affect all the other frequency responses, it is critical that this
    # residual is measured and corrected AS A WHOLE." So once the support is
    # known, every amplitude is solved together, in one system, against the
    # undeflated target. The support comes from the peel; the amplitudes do
    # not.
    #
    # A component the joint solve cannot sustain is dropped by it, which is
    # the correct outcome: it was an artifact of the order it was found in.
    if len(ring_poles) > 1:
        joint = solve_pass(list(ring_poles), ring_target_whole, False,
                           stacked_roots, list(ring_polarities))
        (joint_poles, joint_pairs, joint_alphas, joint_energies,
         _joint_echo, _joint_ghost, joint_levels, _joint_prediction) = joint
        if len(joint_poles) == len(ring_poles):
            ring_pairs = list(joint_pairs)
            ring_alphas = list(joint_alphas)
            ring_energies = list(joint_energies)
            ring_levels = joint_levels
        else:
            # The solve returned a different support than it was given.
            # Keeping the peel's own answer is the conservative reading -
            # a mismatch here means the two disagree about what is being
            # fitted, and a silently misaligned amplitude set is worse
            # than a sequentially fitted one.
            diagnostics_note = (
                "joint re-solve returned %d of %d components; kept the "
                "peel's amplitudes" % (len(joint_poles), len(ring_poles)))
            state.setdefault("notes", []).append(diagnostics_note)

    if ring_levels is None:
        ring_levels = np.zeros_like(smear_levels)

    # RESIDUAL TAIL - the flatness guarantee (the user's rule: any
    # measured artifact profile here can be corrected; the sync area
    # is the reference and it must come out FLAT).  Whatever profile
    # the certified components leave in the sync aftermath is still a
    # repeatable response to the edges, so it is fitted DIRECTLY as a
    # dense per-view FIR tail on the same clean reference drives -
    # strictly causal (taps begin after the settle span), drive-
    # coupled (applied through the same kernel-spread drive as every
    # other stage), Length-bounded by the tip's flat span, shrunk per
    # lag by its own whitened evidence (the standing alpha form), and
    # EMA-tracked at the slow horizon.  The parametric sections carry
    # the resolvable families; the tail carries the remainder the
    # pencil cannot certify, so the corrected sync area closes to the
    # noise floor instead of to the model's span.
    # tail taps begin at the edge's 10-90 end - the tail corrects the
    # overshoot and first lobe immediately (the settle-span start left
    # the first oscillation uncorrected on every tape)
    tail_first = int(math.ceil(geometry.sync_transition_samples / 2.0))
    tail_last = int(tip_flat_samples)
    tail_track = state.setdefault("residual_tail_track", {})
    tail_updates = (state.get("pole_track") or {}).get("updates", 0) + 1
    tail_blend = 1.0 / min(max(tail_updates, 1),
                           max(slow_horizon, 1))
    # BOTH tails calibrate on the FALL-ALIGNED view.  A dense tail is
    # PHASE-BRITTLE: one sample of relative jitter at 2.2 MHz is ~55
    # degrees, and a tail calibrated on the rise-aligned accumulation
    # does not survive the sync-width jitter between that alignment
    # and the per-field application (measured on the decodes: both
    # tapes' rise-view tails INJECTED sustained tones on the back
    # porch - home 2.29 MHz, countdown 2.12 MHz at 1.5 IRE - while
    # their fall-view tails corrected).  The fall-aligned back porch
    # is phase-smeared by exactly the jitter the application sees, so
    # a tail fitted there is jitter-matched by construction.  The
    # rise-aligned view remains the phase-honest instrument for the
    # parametric sections (smooth low-Q shapes tolerate the jitter);
    # the brittle dense layer does not use it.
    reference_change = np.diff(reference, prepend=reference[:1])
    kernel_center = (len(edge_kernel) - 1) // 2
    falling_spread = scipy.signal.fftconvolve(
        np.minimum(reference_change, 0.0), edge_kernel, mode="full")
    falling_reference_drive = falling_spread[
        kernel_center:kernel_center + len(reference)]
    rise_kernel_center = (len(rise_kernel) - 1) // 2
    rising_reference_spread = scipy.signal.fftconvolve(
        np.maximum(reference_change, 0.0), rise_kernel, mode="full")
    rising_reference_drive = rising_reference_spread[
        rise_kernel_center:rise_kernel_center + len(reference)]
    fall_row_residual = ring_stacked[int(offsets[0]):int(offsets[1])]
    tail_regions = (
        # falling tail: witnessed where the sync fall's aftermath
        # lives (up to the back porch)
        (falling_reference_drive, 0, window_plan.back_porch[0]),
        # rising tail: witnessed on the fall-aligned back porch
        (rising_reference_drive, window_plan.back_porch[0],
         window_plan.back_porch[1]),
    )
    residual_tails = []
    for polarity_drive, support_start, support_stop in tail_regions:
        support = np.zeros(len(fall_row_residual))
        stop = min(support_stop, len(support))
        support[support_start:stop] = 1.0
        root = np.sqrt(np.maximum(
            ghost_scan_weights[:len(support)] * support, 0.0))
        length = len(polarity_drive)
        tap_count = max(tail_last - tail_first, 1)
        columns = []
        for lag in range(tail_first, tail_first + tap_count):
            column = np.zeros(length)
            column[lag:] = polarity_drive[:length - lag]
            columns.append(root * column)
        design = np.stack(columns, axis=1)
        observed = root * fall_row_residual
        # the tail lives ONLY in the certified ring band, BY
        # CONSTRUCTION: below the band the smear and level stages own
        # the evidence, above it the accumulated mean carries burst
        # and FM remnants - line-locked (they survive averaging) but
        # NOT transient-driven, the recorded Wiener-seeding lesson.
        # The solve runs in an in-band basis: the tail is a sum of the
        # window's own Fourier modes inside the ring band, fitted
        # against the raw drive columns - optimal within the band,
        # nothing to truncate afterward.
        positions = np.arange(tap_count, dtype=np.float64)
        # every basis mode carries a raised-cosine fade to ZERO at the
        # window end: the Fourier modes are circularly periodic, and
        # fitting a decaying profile with a periodic basis leaked
        # amplitude into the END taps (measured: tail maxima at the
        # last taps, 4.3 us, on two tapes - taps that re-fire after
        # every picture edge as a delayed artifact as long as the sync
        # tip, the ghost the user saw on the home tape).  A physical
        # response decays, and the Length rule already bounds every
        # component by the tip - the tail now ends at zero by
        # construction.
        fade = 0.5 * (1.0 + np.cos(
            math.pi * positions / max(tap_count - 1, 1)))
        modes = []
        for bin_index in range(tap_count // 2 + 1):
            angle = 2.0 * math.pi * bin_index / tap_count
            if not (minimum_ring_angle <= angle <= maximum_ring_angle):
                continue
            modes.append(fade * np.cos(angle * positions))
            if 0 < bin_index < (tap_count + 1) // 2:
                modes.append(fade * np.sin(angle * positions))
        if not modes or not np.any(root > 0.0):
            residual_tails.append(np.zeros(0))
            continue
        basis = np.stack(modes, axis=1)
        design_modes = design @ basis
        normal = design_modes.T @ design_modes
        ridge_tail = np.trace(normal) / max(normal.shape[0], 1) \
            * PENCIL_CONDITION_RIDGE + 1.0
        try:
            mode_weights = np.linalg.solve(
                normal + ridge_tail * np.eye(normal.shape[0]),
                design_modes.T @ observed)
        except np.linalg.LinAlgError:
            mode_weights = np.zeros(design_modes.shape[1])
        if not np.all(np.isfinite(mode_weights)):
            mode_weights = np.zeros(design_modes.shape[1])
        # per-mode evidence shrinkage: a mode the data cannot support
        # above one whitened noise unit fades toward zero
        for position in range(len(mode_weights)):
            contribution = design_modes[:, position] \
                * mode_weights[position]
            energy = float(contribution @ contribution)
            mode_weights[position] *= energy / (energy + 1.0)
        coefficients = basis @ mode_weights
        tail = np.zeros(tail_last)
        tail[tail_first:tail_first + tap_count] = coefficients
        # the tail honors the same fit-magnitude doctrine as the
        # sections (their escalation bounds every realized response at
        # the artifact magnitude limit): a long resonant leftover
        # fitted as an FIR integrates a large in-band peak (measured
        # |R| 1.08 on the countdown tape) and would throttle the whole
        # correction through the convergence ceiling
        tail_response = np.abs(np.fft.rfft(
            np.convolve(edge_kernel, tail), n=512))
        tail_peak = float(np.max(tail_response)) if len(tail) else 0.0
        if tail_peak > ARTIFACT_MAGNITUDE_LIMIT:
            tail = tail * (ARTIFACT_MAGNITUDE_LIMIT / tail_peak)
        tail_key = len(residual_tails)
        held_tail = tail_track.get(tail_key)
        if held_tail is None or len(held_tail) != len(tail):
            held_tail = tail
        else:
            held_tail = (held_tail * (1.0 - tail_blend)
                         + tail * tail_blend)
        tail_track[tail_key] = held_tail
        residual_tails.append(held_tail)

    fitted_poles = list(ring_poles) + list(smear_poles)
    fitted_numerators = list(ring_pairs) + list(smear_pairs)
    fitted_alpha = list(ring_alphas) + list(smear_alphas)
    fitted_energies = list(ring_energies) + list(smear_energies)
    fitted_polarities = (list(ring_polarities)
                         + [None] * len(smear_poles))
    # both passes' region levels refine the same excitation
    level_corrections = smear_levels + ring_levels
    state["fit_diagnostics"] = {
        "candidates": [
            {"pole": pole, "pairs": pairs, "alphas": alphas,
             "energies": energies}
            for pole, pairs, alphas, energies in zip(fitted_poles,
                                                     fitted_numerators,
                                                     fitted_alpha,
                                                     fitted_energies)],
        "level_corrections": np.array(level_corrections),
    }

    track = _tracked_slots(state, ring_slots, smear_slots)
    track = _update_pole_track(
        track, fitted_poles, fitted_numerators, fitted_alpha,
        fitted_energies, longest_window, slow_horizon,
        fitted_polarities)
    sections = _artifact_sections(track)

    # ghost mix: EMA of the fitted coefficient at the slow horizon,
    # like every other tracked quantity; bounded at half the direct
    # signal (runaway protection, not a tuning knob)
    tracked_ghost = dict(state.get("ghost_track")
                         or {"mix": 0.0, "lag": float("nan")})
    ghost_blend = 1.0 / min(max(track["updates"], 1),
                            max(slow_horizon, 1))
    tracked_ghost["mix"] = float(np.clip(
        tracked_ghost["mix"] * (1.0 - ghost_blend)
        + fitted_ghost_mix * ghost_blend, -0.5, 0.5))
    state["ghost_track"] = tracked_ghost
    tracked_mix = tracked_ghost["mix"]

    # echo tap mixes: EMA at the same slow horizon and the same runaway
    # bound as the ghost mix (each tap is a fraction of the direct
    # drive; half is far beyond any observed echo)
    echo_track = state.setdefault("echo_track", {})
    echo_taps = []
    for echo_lag, fitted_echo in zip(state.get("echo_lags", []),
                                     fitted_echo_mixes):
        held = echo_track.get(echo_lag, 0.0)
        held = float(np.clip(held * (1.0 - ghost_blend)
                             + fitted_echo * ghost_blend, -0.5, 0.5))
        echo_track[echo_lag] = held
        echo_taps.append({"lag": float(echo_lag), "mix": held})

    # --- the amplitude dependence, from the fall-aligned strata ---------

    # --- stability margin and the corrector this model implies ----------
    # the application is per-polarity, so the honest |R| is the WORST
    # view's realized response (the blended response is kept for the
    # figure's channel pane)
    angles, response = _artifact_frequency_response(sections, edge_kernel)
    largest_response = float(np.max(np.abs(response))) if len(sections) \
        else 0.0
    for view_index in (0, 1):
        per_view = _view_sections(sections, view_index)
        if per_view:
            _, view_response = _artifact_frequency_response(
                per_view, edge_kernel)
            largest_response = max(
                largest_response,
                float(np.max(np.abs(view_response))))
    # the residual tails are part of the applied channel response, so
    # the convergence ceiling must see them: R_tail = FFT(kernel * tail)
    for tail in residual_tails:
        if len(tail) and np.any(tail != 0.0):
            tail_response = np.abs(np.fft.rfft(
                np.convolve(edge_kernel, tail), n=512))
            largest_response = max(largest_response,
                                   float(np.max(tail_response)))
    margin_scale = 1.0
    if largest_response > ARTIFACT_MAGNITUDE_LIMIT:
        margin_scale = ARTIFACT_MAGNITUDE_LIMIT / largest_response

    # temporal stability: the applied model's step signature against the
    # previous field's - the metric that read 0.347 when the previous
    # pipeline's correction was visibly unstable
    signature = _artifact_impulse(sections, 128, edge_kernel)
    previous = state.get("model_signature")
    stability = float("nan")
    if previous is not None and np.linalg.norm(previous) > 0.0 \
            and np.linalg.norm(signature) > 0.0:
        stability = float(
            np.dot(signature, previous)
            / (np.linalg.norm(signature) * np.linalg.norm(previous)))
    state["model_signature"] = signature

    # --- ghost scan (detection and report only) -------------------------
    # a ghost is a delayed copy of the signal, so it appears in the fit
    # residual as an echo of the sync falling edge; the residual is
    # matched-filtered with the edge's own shape beyond the ringing
    # support
    refined_excitation = reference.copy()
    for (start, stop), correction in zip(
            level_regions, level_corrections[:len(level_regions)]):
        refined_excitation[start:stop] += correction
    # each view is predicted by ITS OWN fitted pairs (the debug-plot
    # contract: what is shown as the model for a view is exactly what
    # that view's rows fitted); the application uses the blended
    # sections
    # the modeled traces run on the ACCUMULATED means, so the even
    # drive's rectification debias uses the accumulated floor (the
    # clean-reference fit columns carry none; both then agree with the
    # noisy input in expectation - see _even_relaxation_response)
    accumulated_bias = noise_floor * math.sqrt(2.0 / (2.0 * math.pi))
    causal_modeled = (refined_excitation
                      + tracked_mix * ghost_signature_fall
                      + _modeled_artifact(
                          slow_mean, _view_sections(sections, 0),
                          edge_kernel, 1.0,
                          accumulated_bias, echo_taps,
                          rise_kernel=rise_kernel))
    modeled = (causal_modeled
               + _symmetric_factor_deviation(reference,
                                             state.get("detail_taps",
                                                       np.zeros(0))))
    if rise_view is not None:
        rise_refined = rise_view["excitation"].copy()
        for (start, stop), correction in zip(
                view_level_regions[1],
                level_corrections[len(level_regions):]):
            rise_refined[start:stop] += correction
        rise_view["ghost_part"] = tracked_mix * rise_ghost_signature
        rise_view["causal_modeled"] = (
            rise_refined + tracked_mix * rise_ghost_signature
            + _modeled_artifact(
                rise_view["mean"], _view_sections(sections, 1),
                edge_kernel, 1.0, accumulated_bias, echo_taps,
                rise_kernel=rise_kernel))
        rise_view["modeled"] = (
            rise_view["causal_modeled"]
            + _symmetric_factor_deviation(rise_view["excitation"],
                                          detail_taps))
    weight_profile = (ghost_scan_weights
                      / max(float(np.max(ghost_scan_weights)), 1e-12))
    residual = (slow_mean - modeled) * weight_profile
    # each edge's echo can only be seen in that edge's own following
    # flat region: elsewhere the residual is dominated by the next
    # transition's own misfit.  A ghost is a delayed COPY, so the
    # residual's DERIVATIVE is matched against the edge's drive kernel.
    kernel = _edge_drive_kernel(edges["sync_fall"][1])
    # the matched filter only needs the kernel's core - the far tails
    # are below a percent of its peak and only shorten the scan range
    core = np.abs(kernel) > 0.01 * float(np.max(np.abs(kernel)))
    kernel = kernel[core]
    kernel_norm = float(np.linalg.norm(kernel))
    ghost = {"lag": float("nan"), "amplitude": float("nan"),
             "significance": 0.0, "scanned_us": 0.0}
    if kernel_norm > 0.0 and len(sections):
        # the scan starts where the APPLIED model's own step signature
        # has decayed into the noise - closer to the edge the residual
        # is model misfit, not a possible echo; a dead tracked slot with
        # a long decay must not push the scan out of the window, so the
        # support is read off the realized signature itself
        # the exclusion reach comes from the RING sections only: the
        # smear shelves are smooth and nearly orthogonal to the edge-
        # derivative matched filter, and a slow shelf's support would
        # otherwise cover the whole window and blind the scan
        ring_sections = [section for section in sections
                         if abs(section["pole"].imag) > 0.0]
        step_signature = _artifact_impulse(ring_sections, 192,
                                           edge_kernel)
        above_noise = np.flatnonzero(
            np.abs(step_signature) > noise_floor)
        signature_support = (int(above_noise[-1]) + 1 if len(above_noise)
                            else 0)
        residual_change = np.diff(residual, prepend=residual[0])
        scan_regions = (
            (edges["sync_fall"], window_plan.sync_tip),
            (edges["sync_rise"], window_plan.back_porch),
        )
        best = (0.0, float("nan"), float("nan"), float("nan"))
        scanned = 0
        candidates_by_lag = []
        confirmations_by_lag = []
        scan_coverage = []
        for edge_index, ((edge_crossing, _, before_level, after_level),
                         (region_start, region_stop)) \
                in enumerate(scan_regions):
            first = max(int(edge_crossing + signature_support),
                        region_start)
            # CONFIRMATION reach: from the settle span onward.  The
            # discovery scan must exclude the model's own ring support
            # (closer in, the residual is misfit, not a candidate
            # echo), but that exclusion also BLINDS the edge - on the
            # countdown tape the fall side's support reaches past its
            # real lag-26 echo, so a cross-edge correlation test using
            # discovery coverage alone vetoed the real echo and the
            # unclaimed pulse flipped the ring calibration (the
            # recorded lesson).  Confirmation therefore answers a
            # narrower question - "does THIS edge also show a matched
            # pulse at a lag the OTHER edge proposed?" - and may look
            # inside the support, where a ring-shaped misfit is
            # near-orthogonal to the edge-derivative filter.
            confirm_first = max(
                int(edge_crossing + geometry.transition_settle_samples),
                region_start)
            last = region_stop - len(kernel)
            if last > confirm_first:
                scan_coverage.append((edge_index,
                                      confirm_first - edge_crossing,
                                      first - edge_crossing,
                                      last - edge_crossing))
            for start in range(min(confirm_first, first), last):
                overlap = residual_change[start:start + len(kernel)]
                correlation = float(np.dot(overlap, kernel))
                amplitude = correlation / (kernel_norm ** 2)
                # the differenced residual carries sqrt(2) of the mean's
                # noise per sample
                significance = abs(correlation) / (
                    kernel_norm * noise_floor * math.sqrt(2.0))
                if start >= confirm_first:
                    confirmations_by_lag.append(
                        (significance, float(start - edge_crossing),
                         amplitude, edge_index))
                if start < first:
                    continue
                scanned += 1
                candidates_by_lag.append(
                    (significance, float(start - edge_crossing),
                     amplitude, edge_index))
                if significance > best[0]:
                    best = (significance,
                            float(start - edge_crossing), amplitude,
                            after_level - before_level)
        # the strongest few LOCAL peaks (a peak's grid neighbours carry
        # the same echo and are not separate candidates) - diagnostic
        # and lock-consensus input.  Peaks keep their WITNESSING EDGE:
        # a channel echo re-fires after every transition, so its lag is
        # seen after BOTH sync edges, while a generator-borne level
        # shift inside one flat region follows exactly one edge - the
        # consensus below refuses single-edge structure (the user's
        # rule: do not match what is not correlated with the
        # transients).  Dedup is per edge so one edge's peak cannot
        # swallow the other edge's testimony at the same lag.
        candidates_by_lag.sort(reverse=True)
        peaks = []
        per_edge_counts = {}
        for significance, lag, amplitude, edge_index in candidates_by_lag:
            if per_edge_counts.get(edge_index, 0) >= 3:
                continue
            if any(abs(lag - other[1]) <= 2.0
                   and edge_index == other[3] for other in peaks):
                continue
            peaks.append((significance, lag, amplitude, edge_index))
            per_edge_counts[edge_index] = \
                per_edge_counts.get(edge_index, 0) + 1
            if len(peaks) >= 6:
                break
        # confirmation peaks: the same extraction over the wider
        # settle-bounded reach, at the standing 3-sigma bar - the
        # corroboration series the consensus consults about lags the
        # discovery series proposed
        confirmations_by_lag.sort(reverse=True)
        confirmed_peaks = []
        confirm_counts = {}
        for significance, lag, amplitude, edge_index \
                in confirmations_by_lag:
            if significance < 3.0:
                break
            if confirm_counts.get(edge_index, 0) >= 6:
                continue
            if any(abs(lag - other[1]) <= 2.0
                   and edge_index == other[3]
                   for other in confirmed_peaks):
                continue
            confirmed_peaks.append((significance, lag, amplitude,
                                    edge_index))
            confirm_counts[edge_index] = \
                confirm_counts.get(edge_index, 0) + 1
        ghost = {"significance": best[0], "lag": best[1],
                 "amplitude": best[2], "edge_step": best[3],
                 "peaks": peaks, "confirmations": confirmed_peaks,
                 "coverage": scan_coverage,
                 "scanned_us": scanned / geometry.sample_rate_mhz}

    # --- ghost LAG tracking ---------------------------------------------
    # The mix is a least-squares coefficient (above); the scan's job is
    # the LAG.  While no lag is locked the scan sees the full ghost and
    # adopts its delay when significant; once locked, the lag follows
    # the scan only in proportion to its confidence (at convergence the
    # residual copy vanishes, the confidence falls, and the lag holds).
    if not np.isfinite(tracked_ghost.get("lag", float("nan"))):
        # CONSENSUS LOCK.  Locking at the first confident detection was
        # measured to be path-dependent: on the countdown tape the
        # offline replay locked a lag-17 echo that never again appeared
        # in the scan's strongest peaks, while the live decoder locked
        # the persistent lag-26 echo - and the two runs then calibrated
        # the smear against different ghost-subtracted targets and
        # settled into DIFFERENT correction levels (strength 0.9 vs
        # 0.4 on the same tape).  The lag is a physical property of the
        # recording, so the lock is decided from evidence ACCUMULATED
        # over the same slow horizon every other tracked quantity uses:
        # each update folds the scan's local peaks into a per-lag
        # evidence average (absent lags decay), and from one full
        # horizon onward the strongest accumulated lag locks under the
        # existing conventions (3-sigma confidence; the lag must clear
        # the transition settle span - inside it a delayed-copy
        # detection is indistinguishable from the ring/smear onset;
        # measured: the home tape once locked a lag-7 "ghost" that was
        # its own fall undershoot).  Once locked the lag never drifts
        # (a wandering lag re-shapes the ghost column every pass and
        # was part of a measured multi-estimator divergence); a
        # recording change clears the state and the consensus restarts.
        history = state.setdefault("ghost_peak_history", [])
        coverage_log = state.setdefault("ghost_scan_coverage", [])
        update_now = track["updates"]
        confirm_history = state.setdefault("echo_confirm_history", [])
        for significance, lag, amplitude, edge_index \
                in ghost.get("peaks", ()):
            if np.isfinite(lag):
                history.append((update_now, float(lag),
                                float(significance), int(edge_index),
                                float(amplitude)))
        for significance, lag, amplitude, edge_index \
                in ghost.get("confirmations", ()):
            if np.isfinite(lag):
                confirm_history.append((update_now, float(lag),
                                        int(edge_index),
                                        float(amplitude)))
        coverage_log.append((update_now,
                             tuple(ghost.get("coverage", ()))))
        # retain two horizons of evidence (the same two-horizon
        # convention the strength ramp uses: one to accumulate, one to
        # trust the statistics)
        retention = 2 * max(slow_horizon, 1)
        while history and history[0][0] < update_now - retention:
            history.pop(0)
        while confirm_history \
                and confirm_history[0][0] < update_now - retention:
            confirm_history.pop(0)
        while coverage_log and coverage_log[0][0] < update_now - retention:
            coverage_log.pop(0)
        if update_now >= max(slow_horizon, 1) and history:
            # MODAL CLUSTERS PER EDGE: strongest-first, fixed +-2
            # radius (the peak dedup span) around each seed.  Chained
            # single-linkage clustering was measured to fuse the whole
            # evidence field (ring-residue comb plus generator step,
            # both edges, lags 20..43, 150 entries) into ONE cluster
            # that trivially passed every cross-edge test at its mean
            # lag.
            def modal_clusters(entries):
                remaining = sorted(entries, key=lambda e: e[2],
                                   reverse=True)
                grouped = []
                while remaining:
                    seed_lag = remaining[0][1]
                    members = [e for e in remaining
                               if abs(e[1] - seed_lag) <= 2.0]
                    grouped.append(members)
                    remaining = [e for e in remaining
                                 if abs(e[1] - seed_lag) > 2.0]
                return grouped

            def coverage_counts(edge, lag):
                """(confirmation, discovery) covering-update counts.

                PRESENCE is judged only over the updates whose scan
                range actually covered a lag: the ring signature's
                support grows as the model converges and pushes the
                discovery floor outward, and counting those blind
                updates against a low-lag echo starved real candidates
                (measured: the countdown tape's lag-26 echo "faded"
                exactly when the scan stopped covering lag 26)."""
                confirm_covered = 0
                discovery_covered = 0
                for _, intervals in coverage_log:
                    for edge_index, confirm_lo, discovery_lo, hi \
                            in intervals:
                        if edge_index != edge:
                            continue
                        if confirm_lo <= lag < hi:
                            confirm_covered += 1
                        if discovery_lo <= lag < hi:
                            discovery_covered += 1
                return confirm_covered, discovery_covered

            settle_span = geometry.transition_settle_samples
            candidates_all = []
            for edge in (0, 1):
                for members in modal_clusters(
                        [e for e in history if e[3] == edge]):
                    lag = float(np.mean([e[1] for e in members]))
                    significance = float(np.mean(
                        [e[2] for e in members]))
                    amplitude = float(np.mean(
                        [abs(e[4]) for e in members]))
                    confirm_covered, discovery_covered = \
                        coverage_counts(edge, lag)
                    presence = (len({e[0] for e in members})
                                / max(discovery_covered, 1))
                    candidates_all.append({
                        "edge": edge, "lag": lag,
                        "significance": significance,
                        "amplitude": amplitude,
                        "presence": presence,
                        "discovery_covered": discovery_covered,
                    })
            candidates_all.sort(key=lambda c: c["significance"],
                                reverse=True)

            def confirmed_on(edge, lag, discovered_amplitude):
                """Does this edge's CONFIRMATION series persistently
                show a physically consistent response at the lag?
                Consistent means WITHIN A FACTOR OF TWO of the
                discovered amplitude, both ways: an echo is a channel
                property and looks alike from either edge.  On these
                accumulated views everything clears 3 sigma (noise
                runs thousandths of an IRE) - a one-sided bar let the
                countdown tape's big back-porch generator level step "confirm" a
                small same-lag tip response into a locked echo."""
                low = 0.5 * abs(discovered_amplitude)
                high = 2.0 * abs(discovered_amplitude)
                hits = {update for update, lag2, edge2, amplitude
                        in confirm_history
                        if edge2 == edge and abs(lag2 - lag) <= 2.0
                        and low <= abs(amplitude) <= high}
                confirm_covered, _ = coverage_counts(edge, lag)
                return (confirm_covered > 0
                        and len(hits) / confirm_covered >= 0.5)

            passing = []
            steps_found = []
            used = set()
            for index, candidate in enumerate(candidates_all):
                if index in used:
                    continue
                used.add(index)
                if (candidate["presence"] < 0.5
                        or candidate["discovery_covered"] == 0
                        or candidate["significance"] < 3.0
                        or candidate["lag"] <= settle_span):
                    continue
                # TRANSIENT CORRELATION (the user's rule): a channel
                # echo re-fires after EVERY transition, so its lag
                # must be witnessed after BOTH sync edges with
                # comparable relative amplitude.  Structure following
                # exactly one edge at a fixed lag is the pulse
                # generator's own level step (measured: a +0.5 IRE
                # level shift 2.3 us after the countdown tape's sync
                # rise sat exactly at that tape's locked lag-33
                # "echo"; the picture-domain gauge shows NO echo at
                # that lag after content edges - matched as an echo it
                # re-fired after every picture transition as injected
                # ghosting).
                other_edge = 1 - candidate["edge"]
                partner = None
                for other_index, other in enumerate(candidates_all):
                    if other_index in used \
                            or other["edge"] != other_edge:
                        continue
                    if abs(other["lag"] - candidate["lag"]) <= 2.0:
                        partner = (other_index, other)
                        break
                dual = False
                if partner is not None:
                    other_index, other = partner
                    amplitudes = sorted([candidate["amplitude"],
                                         other["amplitude"]])
                    if (other["presence"] >= 0.5
                            and other["significance"] >= 3.0
                            and amplitudes[1] > 0.0
                            and amplitudes[0] / amplitudes[1] >= 0.5):
                        dual = True
                        used.add(other_index)
                if not dual and confirmed_on(
                        other_edge, candidate["lag"],
                        candidate["amplitude"]):
                    # the other edge's DISCOVERY may be blind here
                    # (its own ring support), but its confirmation
                    # series can still testify
                    dual = True
                if dual:
                    passing.append((candidate["significance"],
                                    candidate["lag"]))
                    continue
                # SINGLE-EDGE persistent structure: generator level step only
                # when a SHARP LEVEL STEP crosses it - a visible shift
                # between the flats on either side, matched at edge
                # sharpness (the matched amplitude carries at least
                # half the step: a generator shift went through the
                # same channel and has its edge shape, while a smear
                # KNEE shows a large level difference but matches
                # weakly - the home tape's under-modeled tip knee was
                # misclassified without this test, which would have
                # zeroed the very evidence its correction fits on).  A
                # zero-mean ring-residue comb shows no step at all and
                # stays in the fit evidence, where it belongs.
                step_crossing = (
                    edges["sync_fall"][0] if candidate["edge"] == 0
                    else edges["sync_rise"][0])
                position = int(round(step_crossing
                                     + candidate["lag"]))
                step_reach = len(
                    _edge_drive_kernel(edges["sync_fall"][1]))
                flat_span = max(int(0.7 * geometry.sample_rate_mhz), 4)
                flats_before = slow_mean[
                    max(position - flat_span, 0):position]
                flats_after = slow_mean[
                    position + step_reach:
                    position + step_reach + flat_span]
                if len(flats_before) >= 4 and len(flats_after) >= 4:
                    step_size = float(np.mean(flats_after)
                                      - np.mean(flats_before))
                    if (abs(step_size) >= VISIBILITY_FLOOR_IRE
                            and candidate["amplitude"]
                            >= 0.5 * abs(step_size)):
                        steps_found.append(
                            (int(candidate["edge"]),
                             float(candidate["lag"])))
            if os.environ.get("HSYNC_LEVEL_STEP_DEBUG"):
                for candidate in candidates_all:
                    print("LEVEL-STEP", track["updates"],
                          {key: (round(value, 3)
                                 if isinstance(value, float) else value)
                           for key, value in candidate.items()},
                          flush=True)
                print("LEVEL-STEP verdicts:", steps_found,
                      "passing", [(round(s, 1), round(l, 1))
                                  for s, l in passing], flush=True)
            # the level-step set is DECIDED ONCE, at the first evaluation with a
            # full horizon of evidence (the same decided-once doctrine
            # as the echo lock; a re-decided exclusion re-shapes every
            # fit weight and window each pass)
            state.setdefault("source_level_steps", steps_found)
            if passing:
                passing.sort(reverse=True)
                # the strongest persistent cluster is the reported
                # step-ghost lag hypothesis; EVERY passing cluster is a
                # locked echo tap (see the echo columns in the solve)
                tracked_ghost["lag"] = passing[0][1]
                state["echo_lags"] = [lag for _, lag in passing]
                del history[:]
                del confirm_history[:]
                del coverage_log[:]
    state["ghost_track"] = tracked_ghost

    # the sync pulse width is DEFINED between the half-amplitude points
    # of the falling and rising edges (SMPTE 170M / BT.470); both
    # crossings are measured, so the definition is checked on the data
    # itself and reported against the standard's value
    measured_sync_width = edges["sync_rise"][0] - edges["sync_fall"][0]

    # --- the symmetric detail factor (record-side HQ enhancer) ----------
    # solved AFTER the causal fit: its pre-edge witness is the deviation
    # with ONLY the causal model's prediction removed (a causal section
    # contributes nothing before its edge, so what remains there is the
    # symmetric factor alone - in full, not incrementally).  The taps
    # blend at the slow horizon like every other tracked quantity
    # (temporal stability), and the blended taps feed the NEXT fit's
    # target subtraction - the two estimates alternate and settle
    # together.
    detail_extent = 0
    solved_taps = np.zeros(0)
    if rise_view is not None and rise_view.get("causal_modeled") is not None:
        rise_detail_view = {
            "name": "rise", "edges": rise_view["edges"],
            "mean": rise_view["mean"],
            "modeled": rise_view["causal_modeled"],
            "weights": rise_view["weights"],
        }
        detail_extent = _measure_detail_extent(rise_detail_view, geometry)
        if detail_extent > 0:
            fall_detail_view = {
                "name": "fall", "edges": edges, "mean": slow_mean,
                "modeled": causal_modeled, "weights": ghost_scan_weights,
            }
            solved_taps = _solve_detail_taps(
                [fall_detail_view, rise_detail_view], detail_extent)
    held_taps = np.asarray(state.get("detail_taps", np.zeros(0)))
    tap_length = max(len(held_taps), len(solved_taps))
    padded_held = np.zeros(tap_length)
    padded_held[:len(held_taps)] = held_taps
    padded_solved = np.zeros(tap_length)
    padded_solved[:len(solved_taps)] = solved_taps
    tap_blend = 1.0 / min(max(track["updates"], 1), max(slow_horizon, 1))
    blended_taps = (padded_held * (1.0 - tap_blend)
                    + padded_solved * tap_blend)
    # hard invertibility clamp: the Neumann inverse needs the deviation
    # kernel's absolute sum below one; half that margin bounds the taps
    # so no estimation error can ever run the factor outside the
    # invertible region
    tap_mass = float(np.sum(np.abs(blended_taps)))
    if tap_mass > 0.25:
        blended_taps = blended_taps * (0.25 / tap_mass)
    state["detail_taps"] = blended_taps

    model = {
        "sections": sections,
        "levels": levels,
        "edges": edges,
        "measured_sync_width": measured_sync_width,
        "excitation": reference,
        "ghost_part": tracked_mix * ghost_signature_fall,
        "modeled": modeled,
        "fitted_lags": fitted_lags,
        "fall_weights": weights,
        "wiener": wiener,
        "rise_view": rise_view,
        "detail_taps": state["detail_taps"],
        "detail_extent": detail_extent,
        "response_angles": angles,
        "response": response,
        "largest_response": largest_response,
        "margin_scale": margin_scale,
        "residual_tails": residual_tails,
        "rise_edge_width": float(rise_edge_width),
        "stability": stability,
        "ghost": ghost,
        "echo_taps": echo_taps,
        "noise_floor": noise_floor,
    }
    # the transient gate's witness arrays (hsEE1): the gate's envelope
    # and deflation derive from the MEASURED accumulated artifact -
    # what the input actually carries after its edges - never from the
    # corrector's realized response (measured to overshoot the witness
    # ~5x on the home tape through the tracked-blend gap the no-harm
    # governor exists for, and to mass-block legitimate picture drive)
    detail_part = _symmetric_factor_deviation(
        reference, state.get("detail_taps", np.zeros(0)))
    # trimmed at the modeled span's end (the back porch): beyond it the
    # window holds the next ACTIVE region, which the excitation does
    # not model - measured to masquerade as a 59 IRE "ring swing"
    modeled_end = window_plan.back_porch[1]
    witnessed = {"fall": (slow_mean - refined_excitation
                          - tracked_mix * ghost_signature_fall
                          - detail_part)[:modeled_end],
                 "rise": None}
    if rise_view is not None:
        witnessed["rise"] = (
            rise_view["mean"] - rise_refined - rise_view["ghost_part"]
            - _symmetric_factor_deviation(
                rise_view["excitation"],
                state.get("detail_taps", np.zeros(0))))[:modeled_end]
    model["witnessed_artifact"] = witnessed
    # the amplitude-nonlinearity witness: worst measured
    # stratum-to-sync swing ratio per unit step.  REPORTED ONLY - the
    # crude strata measurement (reference-fit error and smear tails
    # included) read 2.4x on the countdown tape and, multiplied into
    # the envelope, choked legitimate drive; the gate-fidelity
    # instruments watch for ring leaks instead.  (Fresh model dict, so
    # the gate realization cache rebuilds per fit.)
    tip_low, tip_high = window_plan.sync_tip
    sync_swing = float(np.max(np.abs(fall_deviation[tip_low:tip_high]))) \
        if tip_high > tip_low else 0.0
    model["gate_swing_factor"] = _strata_swing_factor(
        state, geometry, window_plan,
        sync_swing / max(geometry.sync_depth_ire, 1e-9))
    state["model"] = model
    # the sideband amplitude law measures against the model's own
    # realized correction, so it is fitted immediately after assembly
    # (the realization above carries no law yet - the law scales
    # exactly that response) and before the no-harm governor, whose
    # projection then scores the law-scaled realization
    model["event_landing_law"] = _measure_landing_law(
        state, geometry, window_plan)

    # --- NO-HARM APPLICATION SCALE --------------------------------------
    # The least squares optimizes each VIEW's forward prediction, but
    # what ships is the blended, fixed-point realization - and the gap
    # between the two was measured to overshoot the witnessed
    # accumulations (the correction painted more structure than it
    # removed on a nearly-clean tape).  The realized correction is
    # therefore applied to the accumulated views themselves, and the
    # projection scale that minimizes the witnessed weighted residual
    # bounds the applied strength: s* = -<deviation, delta> / |delta|^2
    # over both views' weighted lags, clipped to [0, 1].  Exactly the
    # strata no-harm doctrine, applied to the correction as a whole -
    # the witnessed accumulation may never come out worse than measured.
    model["application_scale"] = 1.0
    corrected_full, _ = apply_shipping_correction(
        slow_mean, state, geometry, strength_override=1.0)
    # the symmetric detail factor's deviation is only correctable where
    # its inversion actually ships (the VHSHQ flag); elsewhere it is
    # excluded from the harm accounting - charging the causal
    # correction for a deviation it rightly never touches was measured
    # to deflate the scale on every tape
    current_taps = state.get("detail_taps", np.zeros(0))

    def correctable_deviation(mean, reference, base, ghost_part):
        deviation = mean - reference - ghost_part
        if not geometry.detail_emphasis:
            deviation = deviation - _symmetric_factor_deviation(
                base, current_taps)
        return deviation

    deltas = [corrected_full - slow_mean]
    deviations = [correctable_deviation(
        slow_mean, refined_excitation, reference,
        tracked_mix * ghost_signature_fall)]
    # the pre-rise source span is not the scale's evidence either
    # (chasing the HQ pre-shoot / generator step would dial the whole
    # correction after source content); the correction's own delta
    # stays counted there, so still firing over the span costs scale
    if settle_fall_span is not None:
        deviations[0][settle_fall_span[0]:settle_fall_span[1]] = 0.0
    # THE SCALE'S EVIDENCE IS THE SYNC PULSE'S OWN AFTERMATH.  The
    # applied level is correct when the ringing the sync transient
    # excites is completely cancelled, so the projection is scored
    # where that ringing lives: the fall view from the tip onward, the
    # rise view over the back porch (its weights already begin there).
    # The fall view's fit weights also cover the front porch, but the
    # porch's deviation is the PREVIOUS active fall's landing at
    # picture amplitude - the amplitude-dependence layer's witness,
    # not the sync scale's - and as the quietest region it carries the
    # heaviest inverse-variance weight: leaving it in was measured to
    # dilute the projection to ~1.0 while decode-side closure showed
    # the tip wanting 2.32x (home) - the "not enough correction over
    # the sync tip" the user sees.
    fall_aftermath_weights = weights.copy()
    fall_aftermath_weights[: window_plan.sync_tip[0]] = 0.0
    scale_weights = [fall_aftermath_weights]
    if rise_view is not None:
        corrected_rise_full, _ = apply_shipping_correction(
            rise_view["mean"], state, geometry, strength_override=1.0)
        deltas.append(corrected_rise_full - rise_view["mean"])
        deviations.append(correctable_deviation(
            rise_view["mean"], rise_refined, rise_view["excitation"],
            tracked_mix * rise_ghost_signature))
        if settle_rise_span is not None:
            deviations[-1][settle_rise_span[0]:settle_rise_span[1]] = 0.0
        scale_weights.append(rise_view["weights"])
    overlap = 0.0
    delta_energy = 0.0
    for delta, deviation, view_weights in zip(deltas, deviations,
                                              scale_weights):
        overlap += float(np.sum(view_weights * deviation * delta))
        delta_energy += float(np.sum(view_weights * delta ** 2))
    if delta_energy > 0.0:
        # SYNC CLOSURE OWNS THE CEILING: the projection's minimum can
        # sit ABOVE unit application when the fitted model under-
        # measures the ring's amplitude (the onset's aftermath
        # contaminates its own reference), and the old 1.0 clip held
        # the shipped correction at roughly HALF the level the sync
        # residual asked for on the home tape (level closure on the
        # decoded output: tip wanted 2.32x, back porch 1.89x).  The
        # applied level is correct exactly when the measured sync-area
        # ringing closes (the user's reference), so the projection is
        # trusted upward too.  The feedforward subtraction (hsEE1) is
        # LINEAR in the applied scale - there is no fixed point whose
        # convergence could bound it - so the ceiling derives from the
        # witnessed-residual parabola itself: the weighted residual
        # r(s) = |deviation - s x delta|^2 improves for s below twice
        # its minimizer and does NET HARM in the witnessed views
        # beyond, so twice the current projection is the largest scale
        # any witnessed evidence supports.  It exists to bind the
        # TRACKED value at shipping time when a later fit shrinks the
        # supportable range under an EMA that dialed in higher.
        projected_raw = -overlap / delta_energy
        closure_ceiling = 2.0 * max(float(projected_raw), 0.0)
        projected_scale = float(
            np.clip(projected_raw, 0.0, closure_ceiling))
        # the applied strength is a TRACKED quantity (temporal
        # stability rule): the raw per-pass projection was measured to
        # wobble between 0.5 and 1.0 and the user saw the correction
        # jumping around; the same slow-horizon blending every other
        # tracked quantity uses smooths it while the early plain-mean
        # phase keeps the dial-in fast
        held_scale = state.get("application_scale_track")
        scale_blend = 1.0 / min(max(track["updates"], 1),
                                max(slow_horizon, 1))
        if held_scale is None:
            held_scale = projected_scale
        else:
            held_scale = (held_scale * (1.0 - scale_blend)
                          + projected_scale * scale_blend)
        state["application_scale_track"] = float(held_scale)
        # the ceiling binds at shipping time too: the tracked value
        # follows capped projections, but a later fit can shrink the
        # stability headroom under an EMA that dialed in while the
        # ceiling was higher
        model["application_scale"] = float(min(held_scale,
                                               closure_ceiling))

    # runtime/offline parity witness (standing lesson: verify the
    # RUNTIME'S OWN fitted parameters, never the offline replay alone):
    # HSYNC_MODEL_DUMP=1 prints the live fit summary every 20 updates
    if os.environ.get("HSYNC_MODEL_DUMP") and track["updates"] % 10 == 0:
        summary = "  ".join(
            f"{abs(np.angle(s['pole'])) * geometry.sample_rate_mhz / (2.0 * math.pi):.3f}MHz"
            f"/a{s['alpha']:.2f}"
            f"/b{s['numerator'][0]:+.3f},{s['numerator'][1]:+.3f}"
            for s in sections)
        raw_smears = "  ".join(
            f"raw[{-1.0 / math.log(min(abs(pole), 1 - 1e-9)) / geometry.sample_rate_mhz:.2f}us"
            f" fall {numerator[0][1]:+.3f} rise {numerator[1][1]:+.3f}]"
            for pole, numerator in zip(fitted_poles, fitted_numerators)
            if abs(pole.imag) == 0.0)
        summary += "  " + raw_smears
        if echo_taps:
            summary += "  echo " + " ".join(
                f"{tap['lag']:.1f}@{tap['mix']:+.4f}"
                for tap in echo_taps)
        warm_up_now = min(1.0, track["updates"]
                          / (2.0 * max(average_fields, 1)))
        print(f"HSYNC_MODEL_DUMP update {track['updates']}: {summary}"
              f"  |R| {largest_response:.3f}"
              f" margin {margin_scale:.3f}"
              f" warmup {warm_up_now:.2f}"
              f" ghost {tracked_ghost['mix']:+.3f}@"
              f"{tracked_ghost.get('lag', float('nan')):.1f}"
              f" scale {model['application_scale']:.2f}"
              f" strength {warm_up_now * model['application_scale']:.2f}",
              flush=True)
        # parity witness: the runtime's OWN accumulated views, so the
        # offline replay can be fitted against the exact substrate the
        # decoder calibrates on (the half-wave drive is nonlinear, so
        # substrate differences do not cancel the way linear operators'
        # did)
        views_path = os.environ.get("HSYNC_MODEL_DUMP_VIEWS")
        if views_path:
            # every array in the state, so the offline replay can be
            # refitted against the runtime's EXACT inputs and any
            # remaining divergence bisected to a state group
            arrays = {key: value for key, value in state.items()
                      if isinstance(value, np.ndarray)}
            arrays.update({
                f"scalar_{key}": np.float64(state[key])
                for key in ("usable_lines_average",
                            "slow_fields_accumulated",
                            "rise_fields_accumulated",
                            "dark_fields_accumulated",
                            "rise_dark_fields_accumulated",
                            "fields_accumulated")
                if key in state})
            arrays["poles"] = np.array([s["pole"] for s in sections])
            arrays["view_numerators"] = np.array(
                [s["view_numerators"] for s in sections])
            arrays["view_alphas"] = np.array(
                [s["view_alphas"] for s in sections])
            arrays["application_scale"] = np.float64(
                model["application_scale"])
            for field_name, value in dataclasses.asdict(geometry).items():
                if isinstance(value, (int, float, bool)):
                    arrays[f"geometry_{field_name}"] = np.float64(value)
            for key, value in track.items():
                if isinstance(value, np.ndarray):
                    arrays[f"track_{key}"] = value
            arrays["track_updates"] = np.float64(track["updates"])
            np.savez(f"{views_path}_update{track['updates']}.npz",
                     **arrays)
    return model


def describe_geometry(geometry):
    """A human-readable derivation table, echoed for verification."""
    lines = [
        "hsync interval geometry (all positions in output samples from the "
        "sync fall midpoint):",
        f"  sample rate            {geometry.sample_rate_mhz:.5f} MHz "
        f"({geometry.samples_per_line} samples per line)",
        f"  front porch            {geometry.front_porch_samples:.3f} samples",
        f"  sync pulse width       {geometry.sync_pulse_samples:.3f}",
        f"  sync edge (10-90%)     {geometry.sync_transition_samples:.3f}",
        f"  active video starts    {geometry.active_video_start_sample:.3f}",
        f"  sync depth             {geometry.sync_depth_ire:.1f} IRE",
        f"  transition settle      {geometry.transition_settle_samples} samples",
        f"  measurable lines       {geometry.first_measurable_line}"
        f" .. {geometry.last_measurable_line - 1}",
    ]
    return "\n".join(lines)


def describe_window(window_plan):
    """The measurement window layout, relative to the sync fall midpoint."""
    anchor = window_plan.sync_fall_index

    def span(region):
        start, stop = region
        return f"{start - anchor:+d} .. {stop - anchor:+d}" if stop > start \
            else "(empty)"

    lines = [
        "measurement window (positions relative to the sync fall midpoint):",
        f"  window                 {-window_plan.samples_before_sync_fall:+d}"
        f" .. {window_plan.samples_after_sync_fall:+d}"
        f"  ({window_plan.total_samples} samples)",
        f"  entry context          {span(window_plan.entry_context)}",
        f"  front porch            {span(window_plan.front_porch)}",
        f"  sync tip               {span(window_plan.sync_tip)}",
        f"  back porch             {span(window_plan.back_porch)}",
        f"  upcoming active area   {span(window_plan.upcoming_context)}",
        f"  crossing search        +-{window_plan.crossing_search_radius}",
    ]
    return "\n".join(lines)


# =============================================================================
# The correction - transient-gated feedforward subtraction of the
# modeled artifacts
# =============================================================================

# THE DRIVE DISCIPLINE (the user's directive, hsEE1): the corrector is
# stimulated ONLY by the input's transients - never by the artifact
# oscillation it exists to remove, and never by its own output.  The
# previous realization violated both: its drive was the raw signal's
# per-sample change (so the artifact ring, sitting at a kernel's own
# resonant frequency, was that kernel's most effective stimulus), and
# its fixed-point inversion y = x - R{y} fed each pass's corrected
# output back in as the next pass's drive.  The fixed point could only
# remove the MODELED artifact from the drive, so wherever the model was
# imperfect (the countdown tape's pole-frequency mismatch) the leftover
# ring re-excited the kernels and the corrector painted a phase-shifted
# second copy of the ring - self-excitation.  The fit, meanwhile, has
# always calibrated its coefficients against the CLEAN fitted
# excitation (transient-only stimulus); the transient gate makes the
# application's drive honor the same contract on real signal.
#
# The gate works on CHANGE EVENTS - contiguous same-sign runs of the
# per-sample change - because a real transition is one such run (its
# net integral is the step size) while a decaying ring's half-cycles
# alternate sign, each its own small event.  An event passes when its
# (debiased, deflated) step exceeds a causal envelope that holds, for
# every recent accepted event, the largest ring swing the fitted model
# predicts that event can have caused; the envelope holds FLAT for the
# ring's certified visible span and then releases (a resonator's ring
# envelope builds before it decays, so an envelope decaying from the
# edge instant provably leaks - measured on the countdown tape's
# accumulated means).  Measured on 3036 real lines per tape: every sync
# edge passes with >= 4.5x margin, 96.6-98.6% of legitimate
# active-area event energy passes, and no blocked event exceeds 5 IRE.

# Soft acceptance (temporal stability, standing rule "the correction
# must not change field to field"): a hard threshold makes marginal
# events flicker line-to-line under noise.  The pass weight instead
# ramps 0 -> 1 between the envelope plus two and plus four sigma of
# the event-integral noise (an event's integral telescopes to an
# endpoint difference, so its noise is sqrt(2) x the sample noise).
# The ramp STARTS at two sigma so a noise-only event - sign
# segmentation admits every sign run as a candidate - passes with
# probability under 2.5% (the two-sided tail): a one-sigma start was
# measured to pump the even components with rectified noise drive on
# the countdown tape's smooth low-amplitude strata.  Full admission at
# four sigma keeps the same two-sigma ramp span, and a ring sitting at
# its predicted swing plus typical noise stays fully blocked.
GATE_RAMP_START_SIGMA = 2.0
GATE_RAMP_FULL_SIGMA = 4.0


def _segment_events(change, maximum_length):
    """Contiguous same-sign runs of the change.

    Returns (starts, ends, integrals) - `ends` inclusive, integrals
    signed net sums over each run.  Segmentation is by SIGN alone: a
    noise dead-zone gate was measured to trim every edge's sub-noise
    skirts out of the admitted integral, delivering only 85-95% of
    the step the fit calibrated on (the decoded back porches on all
    three tapes paid for it); noise rejection belongs to the
    acceptance floor, not the segmentation.  Noise-only runs arrive
    as small events and fail the acceptance band.  Runs longer than
    `maximum_length` are split - the acceptance decision needs the
    event's end, and the causality rule allows only a bounded
    lookahead (the transition settle reach; a content ramp longer than
    any physical transition is not one event).
    """
    sign = np.sign(change)
    boundaries = np.flatnonzero(np.diff(sign)) + 1
    bounds = np.concatenate(([0], boundaries, [len(change)]))
    starts = bounds[:-1]
    ends = bounds[1:] - 1
    keep = sign[starts] != 0
    starts, ends = starts[keep], ends[keep]
    lengths = ends - starts + 1
    if np.any(lengths > maximum_length):
        pieces_starts = []
        pieces_ends = []
        for start, end, length in zip(starts, ends, lengths):
            if length <= maximum_length:
                pieces_starts.append(start)
                pieces_ends.append(end)
                continue
            cut = start
            while cut <= end:
                pieces_starts.append(cut)
                pieces_ends.append(min(cut + maximum_length - 1, end))
                cut += maximum_length
        starts = np.asarray(pieces_starts, dtype=np.int64)
        ends = np.asarray(pieces_ends, dtype=np.int64)
    running = np.concatenate(([0.0], np.cumsum(change)))
    integrals = running[ends + 1] - running[starts]
    return starts, ends, integrals


def _decayed_running_max(values, decay_samples):
    """max over k <= n of values[k] * exp(-(n - k) / decay_samples),
    causal, O(n): running max in an exponentially grown frame,
    renormalized per block so the grown weights stay finite."""
    if decay_samples <= 0.0:
        return np.maximum.accumulate(values)
    total = len(values)
    # exp(block / decay) stays far below the float64 overflow ceiling
    # (exp(300) ~ 2e130 against 1.8e308)
    block = max(16, int(300.0 * decay_samples))
    out = np.empty_like(values)
    carry = 0.0
    for begin in range(0, total, block):
        segment = values[begin:begin + block]
        grown = segment * np.exp(np.arange(len(segment)) / decay_samples)
        # the carry enters the block already decayed by one step
        grown[0] = max(grown[0],
                       carry * math.exp(-1.0 / decay_samples))
        running = np.maximum.accumulate(grown)
        out[begin:begin + block] = running * np.exp(
            -np.arange(len(segment)) / decay_samples)
        carry = out[begin + len(segment) - 1]
    return out


def _flat_hold_release_max(contributions, hold_samples, release_samples):
    """The gate's envelope from per-sample contributions: each value is
    held FLAT for `hold_samples` (the fitted ring's certified visible
    span - a resonator's ring envelope builds before it decays, so a
    decay-from-the-edge envelope provably leaks; measured on the
    countdown tape) and then released exponentially with
    `release_samples`.  Causal: the envelope at n sees contributions at
    or before n only."""
    hold = max(int(hold_samples), 0)
    # a trailing (causal) window [n - hold, n]: scipy's positive origin
    # slides the window toward earlier samples (verified against the
    # naive recursion in the validation battery)
    held = scipy.ndimage.maximum_filter1d(
        contributions, size=hold + 1, mode="constant", cval=0.0,
        origin=hold - (hold + 1) // 2)
    delayed = np.zeros_like(contributions)
    if hold + 1 < len(contributions):
        delayed[hold + 1:] = contributions[:len(contributions) - hold - 1]
    released = _decayed_running_max(delayed, release_samples)
    return np.maximum(held, released)


def _event_landing_scales(signs, landings, gate):
    """The sideband amplitude law (hsEE5): per-event correction scale
    as a function of the transition's LANDING LEVEL.

    The artifact the correction models is level-dependent (the FM
    carrier rides the level: sync tip 3.40 MHz, blanking 3.69, white
    4.40 from the supplied deviation map), and the two witnessed
    landing levels disagree about the correction: the sync fall
    (landing at the TIP) wants the fitted response at full scale by
    construction, while the picture-fall strata (line-end falls
    landing at BLANKING, observed in the front porch - the
    calibration class rule 19 sanctions) want essentially NONE of it
    (measured: the projection of every solid stratum's aftermath onto
    the shipped unit-fall response reads ~0 on both content tapes -
    the picture artifact rings at different, level-shifted
    frequencies).  The law interpolates the correction scale linearly
    in landing level between the two witnessed anchors and holds
    beyond them; FALL events only (no rise-side landing witness
    exists yet), and with no witnessed strata the law is flat 1.
    """
    law = gate.get("landing_law")
    scales = np.ones(len(signs))
    if law is None or landings is None:
        return scales
    tip_level, blank_level = law["levels"]
    tip_scale, blank_scale = law["scales"]
    span = blank_level - tip_level
    if span <= 0.0:
        return scales
    position = np.clip((landings - tip_level) / span, 0.0, 1.0)
    fall_scales = tip_scale + (blank_scale - tip_scale) * position
    scales[signs] = fall_scales[signs]
    return scales


def _transient_gate(change, gate, signal=None, extended=False):
    """The drive gate (hsEE1): only detected transients stimulate.

    Segments the change into same-sign events, weighs each against the
    ring-swing envelope of the accepted events before it, and returns
    the gated change: passing events at their soft-acceptance weight,
    deflated to their clean-step estimate; everything else (ring
    half-cycles, noise, sub-envelope texture) contributes nothing.

    Two envelope passes, both over the INPUT's own events (this is the
    gate's decision logic, not output feedback): the first scores
    every event optimistically as if all its predecessors passed; the
    second rebuilds the envelope from the first pass's weighted
    acceptances so a blocked event casts no shadow.  The dominant
    envelope contributors are large edges, which pass in both, so the
    ring-blocking decisions are identical between passes.
    """
    starts, ends, integrals = _segment_events(
        change, gate["maximum_event_samples"])
    if len(starts) == 0:
        return np.zeros_like(change)
    signs = integrals < 0.0
    magnitudes = np.abs(integrals)
    deflate = np.where(signs, gate["deflate_fall"], gate["deflate_rise"])
    clean_steps = magnitudes / deflate
    sigma_integral = gate["sigma_integral"]

    def weights_against(envelope):
        prior = envelope[np.maximum(starts - 1, 0)]
        prior[starts == 0] = 0.0
        if sigma_integral > 0.0:
            span = ((GATE_RAMP_FULL_SIGMA - GATE_RAMP_START_SIGMA)
                    * sigma_integral)
            return np.clip(
                (clean_steps - prior
                 - GATE_RAMP_START_SIGMA * sigma_integral) / span,
                0.0, 1.0)
        return (clean_steps > prior).astype(np.float64)

    contributions = np.zeros_like(change)
    contributions[ends] = gate["fraction"] * clean_steps
    envelope = _flat_hold_release_max(
        contributions, gate["hold_samples"], gate["release_samples"])
    weights = weights_against(envelope)
    contributions = np.zeros_like(change)
    contributions[ends] = gate["fraction"] * clean_steps * weights
    envelope = _flat_hold_release_max(
        contributions, gate["hold_samples"], gate["release_samples"])
    weights = weights_against(envelope)

    # the sideband amplitude law (hsEE5): the RING-CLASS correction of
    # each event is scaled by the law of its LANDING level (the signal
    # value where the event ends); the smear/even class keeps the full
    # drive (the front porch lives on it).  Without the signal
    # (drive-fidelity instruments) or without a fitted law, the two
    # drives are identical.
    law_scales = None
    if signal is not None and gate.get("landing_law") is not None:
        landings = np.asarray(signal, np.float64)[ends]
        law_scales = _event_landing_scales(signs, landings, gate)

    def scatter(event_scales):
        gated = np.zeros_like(change)
        active = event_scales > 0.0
        if np.any(active):
            scale = event_scales[active] / deflate[active]
            lengths = ends[active] - starts[active] + 1
            offsets = np.arange(int(lengths.sum())) - np.repeat(
                np.cumsum(lengths) - lengths, lengths)
            positions = np.repeat(starts[active], lengths) + offsets
            gated[positions] = change[positions] * np.repeat(
                scale, lengths)
        return gated

    gated = scatter(weights)
    ring = (scatter(weights * law_scales) if law_scales is not None
            else gated)
    if not extended:
        return gated, ring
    # LANDING CONFINEMENT (hsEE8): the ring-class artifact's frequency
    # tracks the landing carrier (|f_R - f_c(L)| - measured on the
    # countdown actives within 0.1 MHz), so a kernel witnessed at the
    # sync landings mis-tunes as the landing departs and, applied at
    # full amplitude, ADDS ring (the measured active-area injection).
    # The confinement is the derived cancellation overlap of an
    # exponentially decaying ring against a frequency mismatch:
    # 1 / (1 + (2 pi df tau)^2), df = (MHz per IRE) x landing
    # mismatch, tau = the fitted ring pole's own decay - every
    # quantity from the model, the FM map, or the event itself.  The
    # complementary weight (1 - conf) drives the SOURCE kernels: the
    # landing-INVARIANT (content) ring measured on the actives, which
    # the sync-witnessed kernels already carry at the sync landings.
    confinement = gate.get("landing_confinement")
    if confinement is None or signal is None:
        return (gated, ring, gated, ring, np.zeros_like(change),
                None)
    signal_array = np.asarray(signal, np.float64)
    # the SETTLED landing: the event's last change sample sits on the
    # transition's own structure (measured: sync rises read -5..-7 IRE
    # there), so the level is read past the settle span - the same
    # definition the landing map and the offline bins were built with
    read_offset = int(confinement.get("settle", 0)) + 2
    columns = np.arange(6)
    indices = np.clip(ends[:, None] + read_offset + columns[None, :],
                      0, len(signal_array) - 1)
    event_landings = np.median(signal_array[indices], axis=1)
    witnessed = np.where(signs, confinement["landing_fall"],
                         confinement["landing_rise"])
    # dead zone: the landing is read at the event's end, which sits on
    # the transition's own overshoot - a landing within the witnessed
    # aftermath's swing of the witnessed landing IS the witnessed
    # landing (without this, sync rises read a few IRE high and the
    # confinement wrongly faded the kernels on the back porch)
    dead = np.where(signs, confinement["dead_fall"],
                    confinement["dead_rise"])
    mismatch = np.maximum(np.abs(event_landings - witnessed) - dead,
                          0.0)
    conf = 1.0 / (1.0 + confinement["k"] * mismatch * mismatch)
    law = law_scales if law_scales is not None else 1.0
    bank = gate.get("landing_bank")
    bank_drives = None
    if bank is not None:
        half = 0.5 * bank["width"]
        bank_drives = []
        complement = weights * law * (1.0 - conf)
        for center in bank["centers"]:
            members = np.abs(event_landings - center) <= half
            if np.any(members & (complement > 0.0)):
                bank_drives.append(
                    scatter(np.where(members, complement, 0.0)))
            else:
                bank_drives.append(None)
    return (gated, ring, scatter(weights * conf),
            scatter(weights * law * conf),
            scatter(weights * (1.0 - conf)), bank_drives)


def _modeled_artifact(signal, sections, edge_kernel, strength,
                      even_noise_bias=0.0,
                      echo_taps=None, residual_tails=None,
                      rise_kernel=None, gate=None):
    """The artifact tail the model predicts for a given signal.

    The drive is the signal's change, spread by the measured edge
    kernel exactly as a physical transition spreads it; the kernel is
    centered on the change (its half-kernel of lookahead is the bulk
    timing advance the causality rule allows).  Zero-sum drive makes
    the prediction exactly level-neutral.

    `gate` (hsEE1, the transient-gate parameters from
    _gate_parameters): when given, the drive is the TRANSIENT-GATED
    change - only detected transitions stimulate the kernels, so the
    artifact oscillation in the input can never re-excite the model
    that is subtracting it.  The clean references the fit drives with
    need no gate (their change IS transitions only); the shipping
    application always gates.

    `even_noise_bias` is the analytic mean of the half-wave-rectified
    differenced noise OF THIS INPUT (see _even_relaxation_response):
    zero for clean synthetic references, the accumulated floor's for
    accumulated means, the line noise's for raw fields.  Under a gated
    drive it is forced to zero: outside accepted events the half-drives
    are exactly zero, so the debias would inject the very pedestal it
    exists to cancel.
    """
    change = np.diff(signal, prepend=signal[:1])
    if gate is not None:
        change, _ = _transient_gate(change, gate, signal=signal)
        even_noise_bias = 0.0
    center = (len(edge_kernel) - 1) // 2

    def spread_drive(values):
        full = scipy.signal.fftconvolve(values, edge_kernel, mode="full")
        return full[center:center + len(signal)]

    drive = spread_drive(change)
    # PER-POLARITY half drives for the ring sections: each view's
    # fitted pair is calibrated on ITS polarity's aftermath (the fall
    # view's weights end at the back porch, the rise view's begin
    # there), and applying one blended pair to the full drive was
    # measured, by eye, to PHASE-ROTATE the picture's falling-edge
    # rings instead of cancelling them (the countdown tape's stage-
    # light ringing grew "rounder" as the correction dialed in - the
    # fall and rise views fit near-orthogonal phases there, and the
    # rise-governed blend corrected every fall with the rise's phase).
    # The halves sum exactly to the full change; each half is debiased
    # by the analytic rectified-noise mean so the split adds no
    # pedestal on a noisy input (see _even_relaxation_response).
    # one convolution: min + max halves sum exactly to the change, so
    # each polarity's half-wave drive is spread with ITS OWN edge's
    # kernel (rises used to borrow the fall kernel and every rise
    # correction carried the wrong edge shape - the user's rise-phase
    # diagnosis); the rectified-noise debias keeps each half's
    # expectation zero over noise exactly as before
    falling_drive = spread_drive(np.minimum(change, 0.0)
                                 + even_noise_bias)
    if rise_kernel is None or len(rise_kernel) == len(edge_kernel) \
            and np.array_equal(rise_kernel, edge_kernel):
        rising_drive = drive - falling_drive
    else:
        rise_center = (len(rise_kernel) - 1) // 2
        rising_full = scipy.signal.fftconvolve(
            np.maximum(change, 0.0) - even_noise_bias, rise_kernel,
            mode="full")
        rising_drive = rising_full[rise_center:
                                   rise_center + len(signal)]
    total_echo = 0.0
    if echo_taps:
        # each echo tap re-fires the drive at its locked lag (see the
        # echo columns in the solve: a delayed pulse hugging every
        # transition, drive-coupled and level-neutral by construction)
        total_echo = sum(
            tap["mix"] * fractional_shift(drive, tap["lag"])
            for tap in echo_taps if tap["mix"] != 0.0)
    # smear shelves act on the signal itself, as a forward CASCADE -
    # the same product realization whose exact per-stage inverse the
    # application runs (the fit's per-shelf columns are the first-order
    # linearization; mixes are small, and the no-harm projection
    # measures the realized correction, absorbing the difference)
    smeared = signal
    for section in sections:
        pole = section["pole"]
        if abs(pole.imag) > 0.0:
            continue
        mix = section["alpha"] * section["numerator"][0]
        numerator, denominator = _shelf_filter_coefficients(pole.real,
                                                            mix)
        smeared = scipy.signal.lfilter(numerator, denominator, smeared)
    total = smeared - signal + total_echo
    for section in sections:
        pole = section["pole"]
        if abs(pole.imag) == 0.0:
            # the odd shelf mix is handled by the cascade above; the
            # EVEN half-wave mixes add here, each polarity at ITS OWN
            # view-fitted amplitude (see _even_relaxation_response; the
            # amplitude-slope machinery is a ring-family feature and
            # does not apply)
            pairs = section.get("view_numerators")
            alphas = section.get("view_alphas")
            if pairs is not None and alphas is not None:
                falling_mix = float(alphas[0]) * float(pairs[0][1])
                rising_mix = float(alphas[1]) * float(pairs[1][1])
                # the half drives above are the even machinery's own
                # rectified-and-debiased drives (the falling half
                # negated: the even response answers the MAGNITUDE of
                # the change) - reusing them saves one rectify-and-
                # convolve per smear section per fixed-point iteration
                shelf = ([1.0 - pole.real], [1.0, -pole.real])
                if falling_mix != 0.0:
                    total += falling_mix * scipy.signal.lfilter(
                        shelf[0], shelf[1], -falling_drive)
                if rising_mix != 0.0:
                    total += rising_mix * scipy.signal.lfilter(
                        shelf[0], shelf[1], rising_drive)
            continue
        denominator = _denominator_of(pole)
        pairs = section.get("view_numerators")
        alphas = section.get("view_alphas")
        if pairs is not None and alphas is not None:
            # per-polarity: the fall view's pair answers falling
            # transients, the rise view's pair rising ones - each
            # polarity corrected at the phase its own aftermath
            # calibrated (see the half-drive comment above)
            if alphas[0] > 0.0:
                total += float(alphas[0]) * scipy.signal.lfilter(
                    pairs[0], denominator, falling_drive)
            if alphas[1] > 0.0:
                total += float(alphas[1]) * scipy.signal.lfilter(
                    pairs[1], denominator, rising_drive)
        else:
            total += section["alpha"] * scipy.signal.lfilter(
                section["numerator"], denominator, drive)
    if residual_tails is not None:
        # the per-view residual tails, applied per polarity exactly as
        # they were witnessed: the fall view's tail answers falling
        # transients, the rise view's rising ones
        if len(residual_tails) > 0 and len(residual_tails[0]) \
                and np.any(residual_tails[0] != 0.0):
            total += scipy.signal.lfilter(residual_tails[0], [1.0],
                                          falling_drive)
        if len(residual_tails) > 1 and len(residual_tails[1]) \
                and np.any(residual_tails[1] != 0.0):
            total += scipy.signal.lfilter(residual_tails[1], [1.0],
                                          rising_drive)
    return strength * total


def _gate_parameters(model, geometry, input_noise_ire, strength):
    """The transient gate's parameters, every one derived from the
    fitted model, the geometry, or the input's own noise (hsEE1).

    The model-derived parts (predicted ring swing, hold span, release,
    in-event deflation, and the per-polarity clean-step correction
    kernels at the applied strength) are realized once per fitted
    model (kernels once per strength) and cached on the model dict;
    only the noise term varies per input.
    """
    realized = model.get("gate_realization")
    if realized is None:
        sections = model["sections"]
        edge_kernel = _edge_drive_kernel(model["edges"]["sync_fall"][1])
        # every parameter is measured from the WITNESSED accumulated
        # artifact - what the input actually carries after its sync
        # edges - never from the corrector's realized response (the
        # realized response was measured to overshoot the witness ~5x
        # on the home tape, through the tracked-blend gap the no-harm
        # governor exists for, and its envelope mass-blocked
        # legitimate picture drive)
        ring_decays = [
            -1.0 / math.log(min(abs(section["pole"]), 1.0 - 1e-9))
            for section in sections if abs(section["pole"].imag) > 0.0]
        settle = geometry.transition_settle_samples
        event_cap = int(max(settle, 2 * len(edge_kernel)))
        step = max(float(model["levels"]["front_porch"]
                         - model["levels"]["sync_tip"]), 1e-9)
        witnessed = model.get("witnessed_artifact", {})
        views = []
        fall_artifact = witnessed.get("fall")
        if fall_artifact is not None:
            # the fall view's aftermath spans the rise edge too; the
            # excitation-fit residual AT an edge is settle-span
            # unreliable everywhere else in the fit, so it is blanked
            # here as well (an edge-fit spike is not a ring swing)
            fall_artifact = np.asarray(fall_artifact,
                                       dtype=np.float64).copy()
            if "sync_rise" in model["edges"]:
                other = int(round(model["edges"]["sync_rise"][0]))
                lo = max(other - settle, 0)
                fall_artifact[lo:other + settle] = 0.0
            views.append((fall_artifact,
                          int(round(model["edges"]["sync_fall"][0]))))
        rise_view = model.get("rise_view")
        rise_artifact = witnessed.get("rise")
        if rise_artifact is not None and rise_view is not None:
            views.append((rise_artifact, int(round(
                rise_view["edges"]["sync_rise"][0]))))

        def extrema_swings(aftermath):
            """Per-lag consecutive-extrema swings: a ring half-cycle's
            change-event integral is the value swing between
            consecutive extrema of the artifact, so the envelope must
            cover SWINGS, not single-sided peaks."""
            slope_signs = np.sign(np.diff(aftermath))
            turning = np.flatnonzero(np.diff(slope_signs) != 0) + 1
            points = np.concatenate(
                ([0], turning, [len(aftermath) - 1]))
            values = aftermath[points]
            return points[1:], np.abs(np.diff(values))

        fraction = 0.0
        hold = 0
        deflations = [1.0, 1.0]
        directions = (-1.0, +1.0)
        for view_index, (artifact, crossing) in enumerate(views):
            aftermath = np.asarray(
                artifact[crossing + settle:], dtype=np.float64)
            if len(aftermath) < 4:
                continue
            positions, swings = extrema_swings(aftermath)
            if len(swings):
                largest = int(np.argmax(swings))
                fraction = max(fraction,
                               float(swings[largest]) / step)
                # hold: flat while the witnessed ring can still be
                # BUILDING (a resonator's envelope rises before it
                # decays - the measured flat-hold requirement); after
                # the largest swing the ring only decays and the
                # release covers it.  Scale-free build time.
                hold = max(hold, int(positions[largest]) + settle)
            # deflation: what the edge's own change run carries beyond
            # the clean step - ONLY the artifact's early excursion in
            # the edge's own direction extends the run (an opposite-
            # signed deviation, like the pulse-and-bar tape's back-
            # porch deficit, terminates it instead; counting absolute
            # amplitude was measured to under-drive every rise ~7%
            # and cost the decoded back porches on all three tapes)
            early = aftermath[:max(event_cap - settle, 1)]
            excursion = np.maximum(directions[view_index] * early, 0.0)
            deflations[view_index] = 1.0 + float(
                np.max(excursion)) / step if len(excursion) else 1.0
        # release: the slowest certified ring decay (the envelope must
        # never decay faster than the ring it masks); with no rings,
        # the fastest witnessable decay the edge width supports (the
        # same bound the persistent floor uses)
        release = max(ring_decays) if ring_decays else (
            model["edges"]["sync_fall"][1] * math.pi)
        realized = {
            "fraction": float(fraction),
            "hold_samples": int(hold),
            "release_samples": float(release),
            "deflate_fall": float(deflations[0]),
            "deflate_rise": float(deflations[1]),
            "maximum_event_samples": event_cap,
        }
        model["gate_realization"] = realized
    # the clean-step correction kernels at the applied strength (the
    # inverse is nonlinear in strength, so the cache is per rounded
    # strength; the warm-up ramp visits a handful of values and then
    # settles)
    cache = model.setdefault("step_correction_cache", {})
    strength_key = round(float(strength), 3)
    kernels = cache.get(strength_key)
    if kernels is None:
        sections = model["sections"]
        edge_kernel = _edge_drive_kernel(model["edges"]["sync_fall"][1])
        rise_kernel = (_edge_drive_kernel(model["rise_edge_width"])
                       if model.get("rise_edge_width") else None)
        applied_sections = []
        for section in sections:
            if abs(section["pole"].imag) > 0.0:
                applied_sections.append(section)
            else:
                pairs = section.get("view_numerators")
                if pairs is None or not (pairs[0][1] != 0.0
                                         or pairs[1][1] != 0.0):
                    continue
                even_only = dict(section)
                even_only["numerator"] = np.array([0.0, 0.0])
                applied_sections.append(even_only)
        slowest = max(
            [-1.0 / math.log(min(abs(section["pole"]), 1.0 - 1e-6))
             for section in sections] + [1.0])
        tap_tolerance = (VISIBILITY_FLOOR_IRE
                         / (geometry.sync_depth_ire + 100.0))
        echo_reach = max(
            [tap["lag"] + len(edge_kernel)
             for tap in (model.get("echo_taps") or [])] + [0.0])
        length = int(min(max(
            128, math.ceil(slowest * math.log(1.0 / tap_tolerance)),
            math.ceil(2.0 * echo_reach)),
            geometry.samples_per_line))
        # TWO kernel sets (hsEE5b): the REST kernels (evens, echo
        # taps, residual tails - no rings) take the full gated drive,
        # and the DIFFERENCE kernels (joint realization minus rest -
        # the ring class's contribution INCLUDING the joint inverse's
        # cross-terms) take the law-scaled drive.  Limits: with no
        # fitted law the two sum to exactly the JOINT realization
        # (the pulse-and-bar cross-term delivery, v3's fix); where the
        # law reads ~0 an event gets the rest delivery alone (the v2
        # behavior that won the countdown tip); sync events (law 1)
        # get exactly the joint realization - better than v2, whose
        # cross-term-less sync delivery slipped the home front porch.
        rest_sections = [section for section in applied_sections
                         if abs(section["pole"].imag) == 0.0]
        joint_kernels = _unit_step_corrections(
            applied_sections, edge_kernel, strength, length,
            echo_taps=model.get("echo_taps"),
            residual_tails=model.get("residual_tails"),
            rise_kernel=rise_kernel)
        rest_kernels = _unit_step_corrections(
            rest_sections, edge_kernel, strength, length,
            echo_taps=model.get("echo_taps"),
            residual_tails=model.get("residual_tails"),
            rise_kernel=rise_kernel)
        difference_kernels = (joint_kernels[0] - rest_kernels[0],
                              joint_kernels[1] - rest_kernels[1])
        # PHASE-PROFILE BLEND (hsEE7, the user's ruling - see the
        # sidecar loader's comment).  Per polarity, over the certified
        # settle span only (a mask that stays zero through the edge's
        # own settle, where the profile's ideal-edge fit is unreliable
        # and the witnessed kernels keep full weight): the witnessed
        # kernels deflate by (1 - lambda) and the profile's predicted
        # aftermath enters at min(lambda/rho, 1) - a convex blend, so
        # the two estimates of the same artifact are never both applied
        # in full (the measured amount-asymmetry law: over-correction
        # amplifies the correction's own noise and destroys the
        # benefit, so any blend must approach from below).
        profile_pair = None
        profile = _load_phase_profile()
        head_name = model.get("head_name")
        if profile is not None and head_name:
            settle = max(int(geometry.transition_settle_samples), 1)
            offset = length // 4
            blend_mask = np.zeros(length)
            ramp_start = min(offset + settle, length)
            ramp_stop = min(ramp_start + settle, length)
            if ramp_stop > ramp_start:
                blend_mask[ramp_start:ramp_stop] = 0.5 * (1.0 - np.cos(
                    np.pi * (np.arange(ramp_stop - ramp_start) + 1.0)
                    / (ramp_stop - ramp_start)))
            blend_mask[ramp_stop:] = 1.0
            t0_index = int(profile["t0_index"])
            pair = []
            deflate = [np.ones(length), np.ones(length)]
            for polarity, edge in enumerate(("fall", "rise")):
                deviation = profile.get(f"{head_name}_{edge}")
                if deviation is None:
                    pair.append(np.zeros(length))
                    continue
                certified = float(
                    profile[f"{head_name}_{edge}_lambda"])
                rho = float(profile[f"{head_name}_{edge}_rho"])
                applied = min(certified / max(rho, 1e-6), 1.0)
                kernel = np.zeros(length)
                span = min(len(deviation) - t0_index, length - offset)
                kernel[offset:offset + span] = deviation[
                    t0_index:t0_index + span]
                taper = min(16, span)
                if taper > 0:
                    kernel[offset + span - taper:offset + span] *= \
                        np.linspace(1.0, 0.0, taper)
                sign = -1.0 if edge == "fall" else 1.0
                pair.append(sign * kernel * blend_mask * applied)
                deflate[polarity] = 1.0 - certified * blend_mask
            if any(np.any(kernel != 0.0) for kernel in pair):
                rest_kernels = (rest_kernels[0] * deflate[0],
                                rest_kernels[1] * deflate[1])
                difference_kernels = (
                    difference_kernels[0] * deflate[0],
                    difference_kernels[1] * deflate[1])
                profile_pair = (pair[0], pair[1])
        # SOURCE-CLASS kernels (hsEE8, the user's assignment: the
        # ringing that does NOT correlate to the RF path).  The
        # landing-INVARIANT aftermath measured across the active
        # landing bins - content ringing recorded before tape (the
        # off-air broadcast chain on countdown, record-side
        # enhancement on home, the emphasis remainder on the bar
        # patterns).  Level-independent by measurement, so one kernel
        # serves every landing; driven by the complement of the
        # landing confinement, so at the sync landings (where the
        # witnessed kernels already contain this component) it
        # contributes nothing and the closure floors stand.  Applied
        # at the half-optimum scale certified by split-half (the
        # amount-asymmetry law: err low).
        source_pair = None
        if profile is not None and head_name:
            pair_source = []
            for edge in ("fall", "rise"):
                deviation = profile.get(f"{head_name}_source_{edge}")
                if deviation is None:
                    pair_source.append(np.zeros(length))
                    continue
                applied = float(
                    profile[f"{head_name}_source_{edge}_scale"])
                t0_index = int(profile["t0_index"])
                kernel = np.zeros(length)
                span = min(len(deviation) - t0_index, length - offset)
                kernel[offset:offset + span] = deviation[
                    t0_index:t0_index + span]
                taper = min(16, span)
                if taper > 0:
                    kernel[offset + span - taper:offset + span] *= \
                        np.linspace(1.0, 0.0, taper)
                sign = -1.0 if edge == "fall" else 1.0
                pair_source.append(
                    sign * kernel * blend_mask * applied)
            if any(np.any(kernel != 0.0) for kernel in pair_source):
                source_pair = (pair_source[0], pair_source[1])
        # LANDING-RETUNED PATH BANK (hsEE9, the user's directive:
        # correct - not merely confine - the landing-tracking ring at
        # active landings).  The FM sideband map is linear in landing
        # (f_path(L) = f_R - f_c(L)), and each ring pole SELF-DERIVES
        # its retune law from its own witnessed anchor: the feature
        # frequency is f_R = f_c(L_witnessed) + f_pole, so at landing
        # L the pole's frequency is f_pole - per_ire*(L - L_wit) -
        # no new constants.  Each pole's anchor is the view that
        # witnessed it (fall -> the sync tip landing, rise -> the
        # blanking landing, by dominant view alpha).  Per landing
        # bin (width = the confinement tolerance, one linewidth of
        # the slowest ring), the bank realizes FIRST-ORDER kernels
        # from the retuned ring sections on a clean unit step, at
        # HALF the applied strength (the half-optimum law: the bank
        # is new and errs low); it rides the confinement COMPLEMENT,
        # so at the witnessed landings it is exactly inert and the
        # sync closure is untouched by construction.  The held-out
        # active audit is its governor.
        landing_bank = None
        if profile is not None and head_name \
                and "fm_mhz_per_ire" in profile \
                and profile.get("landing_bank_enabled", True):
            ring_sections = [section for section in model["sections"]
                             if abs(section["pole"].imag) > 0.0]
            ring_decays = [
                -1.0 / math.log(min(abs(section["pole"]), 1.0 - 1e-9))
                for section in ring_sections]
            if ring_sections and ring_decays:
                per_ire = float(profile["fm_mhz_per_ire"])
                rate_mhz = geometry.sample_rate_mhz
                tau_us = max(ring_decays) / rate_mhz
                bin_width = 1.0 / (2.0 * math.pi * per_ire * tau_us)
                landing_fall = float(
                    profile["witnessed_landing_fall"])
                landing_rise = float(
                    profile["witnessed_landing_rise"])
                low = landing_fall - bin_width
                high = 100.0 + bin_width
                count = max(int(math.ceil((high - low) / bin_width)),
                            1)
                centers = low + (np.arange(count) + 0.5) * bin_width
                edge_kernel_bank = _edge_drive_kernel(
                    model["edges"]["sync_fall"][1])
                rise_kernel_bank = (
                    _edge_drive_kernel(model["rise_edge_width"])
                    if model.get("rise_edge_width") else None)
                bank_kernels = []
                offset = length // 4
                for center in centers:
                    retuned = []
                    for section in ring_sections:
                        pole = section["pole"]
                        alphas = section.get("view_alphas")
                        anchor = landing_fall
                        if alphas is not None and \
                                float(alphas[1]) > float(alphas[0]):
                            anchor = landing_rise
                        frequency = (abs(pole) and
                                     abs(np.angle(pole))
                                     / (2.0 * math.pi) * rate_mhz)
                        floor = 1.0 / (2.0 * math.pi * tau_us)
                        target = max(
                            frequency - per_ire * (center - anchor),
                            floor)
                        angle = 2.0 * math.pi * target / rate_mhz
                        moved = dict(section)
                        moved["pole"] = abs(pole) * complex(
                            math.cos(angle), math.sin(angle)
                            * (1.0 if pole.imag >= 0.0 else -1.0))
                        retuned.append(moved)
                    pair = []
                    for direction in (-1.0, +1.0):
                        step = np.zeros(length)
                        step[offset:] = direction
                        artifact = _modeled_artifact(
                            step, retuned, edge_kernel_bank,
                            strength * 0.5, 0.0, None, None,
                            rise_kernel_bank)
                        pair.append(artifact)
                    bank_kernels.append((pair[0], pair[1]))
                # R(f)-SYNTHESIZED BLEND (step175/176, the user's
                # combination directive): when the sidecar carries
                # certified synthesized kernels (from the validated
                # FM simulator over the measured per-head response),
                # each bank bin blends CONVEXLY toward the synthesized
                # kernel at its certified half-optimum weight - the
                # sync witness remains the arbiter, and an uncertified
                # channel (weight 0) leaves the retuned kernel whole.
                synth_centers = profile.get(
                    f"{head_name}_synth_centers")
                if synth_centers is not None:
                    for index, center in enumerate(centers):
                        nearest = int(np.argmin(np.abs(
                            synth_centers - center)))
                        if abs(float(synth_centers[nearest])
                               - center) > bin_width:
                            continue
                        blended = list(bank_kernels[index])
                        for polarity, edge in enumerate(
                                ("fall", "rise")):
                            stack = profile.get(
                                f"{head_name}_synth_{edge}")
                            weight_key = (f"{head_name}_synth_{edge}"
                                          f"_scale")
                            if stack is None or \
                                    weight_key not in profile:
                                continue
                            w = float(profile[weight_key])
                            if w <= 0.0:
                                continue
                            if model.get("channel_eq_active") and not \
                                    profile.get(
                                        "synth_blend_under_channel_eq",
                                        False):
                                continue
                            deviation = np.asarray(
                                stack[nearest], dtype=np.float64)
                            t0_index = int(profile["t0_index"])
                            kernel = np.zeros(length)
                            span = min(len(deviation) - t0_index,
                                       length - offset)
                            kernel[offset:offset + span] = deviation[
                                t0_index:t0_index + span]
                            sign = -1.0 if edge == "fall" else 1.0
                            kernel *= sign * strength * 0.5
                            blended[polarity] = (
                                (1.0 - w) * blended[polarity]
                                + w * kernel)
                        bank_kernels[index] = (blended[0],
                                               blended[1])
                landing_bank = {"centers": centers,
                                "width": bin_width,
                                "kernels": bank_kernels}
        kernels = (rest_kernels, difference_kernels, profile_pair,
                   source_pair, landing_bank)
        cache[strength_key] = kernels
    # noise term of THIS input: an event's integral telescopes to an
    # endpoint difference, so it carries sqrt(2) of the sample noise
    sigma = max(float(input_noise_ire), 0.0)
    gate = dict(realized)
    gate["step_corrections_rest"] = kernels[0]
    gate["step_corrections_difference"] = kernels[1]
    gate["step_corrections_profile"] = kernels[2]
    gate["step_corrections_source"] = kernels[3]
    gate["landing_bank"] = kernels[4]
    profile_data = _load_phase_profile()
    if profile_data is not None and "fm_mhz_per_ire" in \
            getattr(profile_data, "keys", lambda: profile_data)():
        ring_decays = [
            -1.0 / math.log(min(abs(section["pole"]), 1.0 - 1e-9))
            for section in model["sections"]
            if abs(section["pole"].imag) > 0.0]
        if ring_decays:
            tau_us = max(ring_decays) / geometry.sample_rate_mhz
            per_ire = float(profile_data["fm_mhz_per_ire"])
            witnessed_views = model.get("witnessed_artifact", {})
            swings = {}
            for edge in ("fall", "rise"):
                artifact = witnessed_views.get(edge)
                swings[edge] = (float(np.max(np.abs(
                    np.asarray(artifact, dtype=np.float64))))
                    if artifact is not None else 0.0)
            gate["landing_confinement"] = {
                "k": (2.0 * math.pi * per_ire * tau_us) ** 2,
                "landing_fall": float(
                    profile_data["witnessed_landing_fall"]),
                "landing_rise": float(
                    profile_data["witnessed_landing_rise"]),
                "dead_fall": swings["fall"],
                "dead_rise": swings["rise"],
                "settle": int(geometry.transition_settle_samples),
            }
    gate["sigma_integral"] = sigma * math.sqrt(2.0)
    gate["landing_law"] = model.get("event_landing_law")
    return gate


def _unit_step_corrections(sections, edge_kernel, strength, length,
                           echo_taps=None, residual_tails=None,
                           rise_kernel=None):
    """The exact inverse's correction for one clean unit step, per
    polarity: the applied kernels of the gated feedforward path.

    The old application solved y = x - R{y} on the PICTURE - output
    fed back as input, and every oscillation in the picture (the
    artifact itself included) re-drove the model.  The inverse itself
    was not the defect: on a clean isolated step it consolidates the
    energy the channel spread into ring and smear back into the
    transition (the energy-relocation rule), and retiring it wholesale
    was measured to under-deliver the low-frequency (smear-band)
    correction on every back porch - 1/(1 + sR) exceeds the
    first-order 1 - sR by tens of percent exactly where the smear
    response is largest.  So the fixed point survives ONLY here, as a
    model realization on a synthetic clean step - it never sees
    picture content, so nothing the corrector outputs can ever reach
    its input.  The returned arrays are the correction responses to a
    unit falling and unit rising change impulse at index
    `length // 4`; the application convolves the transient-gated
    change halves with them.
    """
    kernels = []
    offset = length // 4
    for direction in (-1.0, +1.0):
        step = np.zeros(length)
        step[offset:] = direction
        corrected = step
        for _ in range(24):
            refined = step - _modeled_artifact(
                corrected, sections, edge_kernel, strength, 0.0,
                echo_taps, residual_tails, rise_kernel)
            movement = float(np.max(np.abs(refined - corrected)))
            corrected = refined
            if movement < 1e-9:
                break
        kernels.append(step - corrected)
    return kernels[0], kernels[1]


def invert_artifacts(signal, sections, edge_kernel, strength, noise_floor,
                     diagnostics=None,
                     even_noise_bias=0.0, echo_taps=None,
                     residual_tails=None, rise_kernel=None, gate=None):
    """Remove the modeled artifacts from a signal: transient-gated
    feedforward subtraction (hsEE1).

    Stimulated ONLY by the input's detected transients, applied once,
    output never fed back.  With a gate, each gated change half is
    convolved with the exact inverse's clean-step correction kernel
    for its polarity (see _unit_step_corrections) - the full
    energy-relocating delivery of the old fixed point, without its
    defect (the old loop ran ON the picture, so the artifact
    oscillation and the corrector's own output re-drove the model; the
    kernels are realized on a synthetic clean step and the picture
    only ever selects and scales them through the gate).  Without a
    gate (clean fit-side references), the forward prediction is
    subtracted directly.  `noise_floor` is kept for the diagnostics
    contract.
    """
    has_tails = residual_tails is not None and any(
        len(tail) and np.any(tail != 0.0) for tail in residual_tails)
    if (not sections and not echo_taps and not has_tails) \
            or strength <= 0.0:
        return signal
    if gate is None:
        corrected = signal - _modeled_artifact(
            signal, sections, edge_kernel, strength,
            even_noise_bias, echo_taps, residual_tails, rise_kernel)
    else:
        change = np.diff(signal, prepend=signal[:1])
        (gated, gated_ring, gated_conf, gated_ring_conf,
         gated_far, bank_drives) = _transient_gate(
            change, gate, signal=signal, extended=True)
        correction = np.zeros_like(signal)
        # each kernel is the response to a UNIT change impulse of its
        # polarity (kernel_fall to a -1 change), so a falling half of
        # magnitude a contributes a x kernel_fall; linearity in the
        # accepted drive is exact.  The REST kernels take the full
        # gated drive; the DIFFERENCE kernels (joint minus rest - the
        # ring class with its cross-terms) take the law-scaled drive,
        # so with no fitted law the sum is exactly the joint
        # realization (hsEE5b - see the kernel-cache comment).
        kernel_pairs = [
            (gate["step_corrections_rest"], gated),
            (gate["step_corrections_difference"], gated_ring_conf)]
        profile_pair = gate.get("step_corrections_profile")
        if profile_pair is not None:
            # the profile kernels carry the TOTAL predicted aftermath
            # at the SYNC landings (every class), so they ride the
            # landing-CONFINED drive like the witnessed ring class;
            # the witnessed kernels were deflated by the certified
            # fraction where these enter (the convex blend)
            kernel_pairs.append((profile_pair, gated_conf))
        bank = gate.get("landing_bank")
        if bank is not None and bank_drives is not None:
            # the landing-retuned path kernels: each bin's events,
            # law-scaled, on the confinement complement - correcting
            # the tracking ring AT ITS OWN LANDING's frequency
            for pair_bank, drive_bank in zip(bank["kernels"],
                                             bank_drives):
                if drive_bank is not None:
                    kernel_pairs.append((pair_bank, drive_bank))
        source_pair = gate.get("step_corrections_source")
        if source_pair is not None:
            # the landing-invariant content ring, driven by the
            # confinement's COMPLEMENT: zero at the sync landings
            # (already inside the witnessed kernels), full where the
            # landing departs and the confined kernels fade
            kernel_pairs.append((source_pair, gated_far))
        for kernels, drive_source in kernel_pairs:
            kernel_fall, kernel_rise = kernels
            offset = len(kernel_fall) // 4
            falling = np.minimum(drive_source, 0.0)
            rising = drive_source - falling
            full = scipy.signal.fftconvolve(-falling, kernel_fall,
                                            mode="full")
            correction += full[offset:offset + len(signal)]
            full = scipy.signal.fftconvolve(rising, kernel_rise,
                                            mode="full")
            correction += full[offset:offset + len(signal)]
        corrected = signal - correction
    if diagnostics is not None:
        diagnostics["iterations"] = 1
        diagnostics["movement"] = 0.0
        diagnostics["noise_floor"] = noise_floor
        diagnostics["converged"] = True
    return corrected


def _correction_strength(state, average_fields):
    """How strongly the tracked model is applied.

    Ramps in over the slow fitting horizon so the correction fades up as
    the model earns confidence; after a state reset it fades up again
    from zero.  The level itself belongs to the sync-closure scale (the
    user's reference: the correction does whatever it takes to undo the
    measured ringing, and the sync residual says how much is enough).
    The stability margin no longer multiplies here - its caution already
    lives in the closure ceiling, and multiplying both under-applied by
    the margin twice.
    """
    track = state.get("pole_track")
    model = state.get("model")
    if track is None or model is None:
        return 0.0
    # the ramp covers two averaging horizons: the accumulation is a
    # plain (equal-weight) mean until one horizon, and the variance
    # estimate that whitens the fit needs the second - beyond that the
    # measured no-harm application scale is the confidence authority
    # (the old eight-horizon ramp predates that governor, double-
    # counted caution, and was seen by the user as the correction
    # taking a very long time to dial in; it also re-imposed the whole
    # wait after every recording-change reset)
    ramp_fields = 2.0 * max(average_fields, 1)
    warm_up = min(1.0, track["updates"] / ramp_fields)
    return warm_up * model.get("application_scale", 1.0)


def _corrector_impulse_response(sections, edge_kernel, strength, length,
                                echo_taps=None, gate=None):
    """The realized corrector's ISOLATED-EVENT response (for the luma
    transient improvement stage, and the debug figure).

    The gated corrector is not LTI: this is its exact response to one
    isolated above-threshold unit event on a clean background (the
    impulse passes a zero-noise gate over an empty envelope, deflated
    like any event), and superposition holds only for events the gate
    accepts.  Callers that assume linearity (the LTI stage) receive
    the response valid for exactly that regime.
    """
    impulse = np.zeros(length)
    impulse[length // 4] = 1.0
    corrected = invert_artifacts(impulse, sections, edge_kernel, strength,
                                 0.0, echo_taps=echo_taps, gate=gate)
    return corrected


def apply_shipping_correction(values, state, geometry, diagnostics=None,
                              strength_override=None,
                              input_noise_ire=None):
    """The exact correction the decoder ships, applied to any sequence.

    ONE code path serves the per-field application AND every debug
    figure's "after correction" trace (the debug-plot contract: a
    corrected trace in a figure must be produced by the correction that
    actually runs, or the figure cannot be trusted for testing).
    Returns (corrected, strength).

    `input_noise_ire` is the per-sample noise of THE INPUT, feeding the
    transient gate's segmentation gate, acceptance band and boundary
    debias (see _gate_parameters): None means the accumulated mean's
    floor (the fit, figure and tool callers all correct accumulated
    views); the per-field runtime caller passes its own line noise.
    """
    values = np.asarray(values, dtype=np.float64)
    model = state.get("model")
    average_fields = state.get("average_fields", 1)
    strength = _correction_strength(state, average_fields)
    if strength_override is not None:
        strength = strength_override
    if model is None or strength <= 0.0 or not (
            model["sections"] or model.get("echo_taps")
            or model.get("residual_tails")):
        return values.copy(), strength
    if input_noise_ire is None:
        input_noise_ire = model["noise_floor"]
    if "head_name" in state:
        model["head_name"] = state["head_name"]
    # the transient gate (hsEE1): only detected transitions drive the
    # subtraction, and the gated drive carries no rectified-noise
    # pedestal, so the even debias is retired with the raw drive it
    # existed for (see _modeled_artifact)
    gate = _gate_parameters(model, geometry, input_noise_ire, strength)
    sections = model["sections"]
    edge_kernel = _edge_drive_kernel(model["edges"]["sync_fall"][1])
    # SMEAR shelves are inverted EXACTLY, one causal one-pole stage per
    # component (see _shelf_filter_coefficients: real pole and zero, so
    # the smear correction can never oscillate); the gated feedforward
    # ring subtraction then runs on the shelf-corrected signal.  Each
    # stage's mix is scaled by the applied strength - exact identity at
    # 0, exact inverse at 1, continuous in between.
    corrected = values
    for section in sections:
        if abs(section["pole"].imag) > 0.0:
            continue
        mix = strength * section["alpha"] * section["numerator"][0]
        numerator, denominator = _shelf_filter_coefficients(
            section["pole"].real, mix)
        # the inverse is causal and stable only while the shelf stays
        # minimum-phase (its zero inside the unit circle); a mix pushed
        # outside that region is never inverted (never seen with the
        # clamped fits, but the bound costs nothing to honor)
        if abs(numerator[1] / numerator[0]) < 1.0:
            corrected = scipy.signal.lfilter(denominator, numerator,
                                             corrected)
    # rings and the smears' EVEN halves subtract through the gated
    # feedforward path (the odd shelf mixes were exactly inverted
    # above, so they enter with a zero odd tap and the cascade inside
    # the prediction is the identity for them)
    gated_sections = []
    for section in sections:
        if abs(section["pole"].imag) > 0.0:
            gated_sections.append(section)
        else:
            pairs = section.get("view_numerators")
            if pairs is None or not (pairs[0][1] != 0.0
                                     or pairs[1][1] != 0.0):
                continue
            even_only = dict(section)
            even_only["numerator"] = np.array([0.0, 0.0])
            gated_sections.append(even_only)
    rise_kernel = (_edge_drive_kernel(model["rise_edge_width"])
                   if model.get("rise_edge_width") else None)
    corrected = invert_artifacts(
        corrected, gated_sections, edge_kernel, strength,
        model["noise_floor"],
        diagnostics=diagnostics,
        echo_taps=model.get("echo_taps"),
        residual_tails=model.get("residual_tails"),
        rise_kernel=rise_kernel, gate=gate)
    # RECORD-SIDE DETAIL EMPHASIS (HQ-era decks): the measured symmetric
    # factor is inverted only on formats whose recorders carry the
    # delay-line enhancer (the VHSHQ flag) - a symmetric kernel is the
    # one sanctioned exception to strict causality, because the artifact
    # itself is acausal by construction (the enhancer works on a delayed
    # tap).  On every other format the factor is measured and reported
    # only.
    detail_taps = model.get("detail_taps", np.zeros(0))
    if geometry.detail_emphasis and len(detail_taps) \
            and np.any(detail_taps != 0.0):
        # a kernel term is worth keeping when its effect on the largest
        # step the format carries (sync depth to peak white,
        # ~sync_depth+100 IRE) clears the visibility floor
        tap_tolerance = (VISIBILITY_FLOOR_IRE
                         / (geometry.sync_depth_ire + 100.0))
        inverse_kernel, inverse_center = _detail_inverse_kernel(
            detail_taps, tap_tolerance)
        if len(inverse_kernel) > 1:
            # applied at the same ramped strength as the causal
            # correction: kernel = delta + strength (inverse - delta)
            applied_kernel = inverse_kernel * strength
            applied_kernel[inverse_center] += 1.0 - strength
            full = scipy.signal.fftconvolve(corrected, applied_kernel,
                                            mode="full")
            corrected = full[inverse_center:
                             inverse_center + len(corrected)]
    # LEVEL NEUTRALITY over whole fields (hsEE1): every modeled
    # component decays to zero within a line, so a nonzero whole-field
    # correction mean is only the accumulated polarity mismatch of the
    # gated drive estimates (per-polarity deflation measured to leave a
    # +0.57 IRE field pedestal on the countdown tape) - unwitnessable
    # from sync, and owned by the level-anchoring chain, so it is
    # removed.  Sub-line inputs (the views, figures and gauges) keep
    # their exact shape: a correction can legitimately hold a nonzero
    # mean over less than a line.
    if len(values) >= 2 * geometry.samples_per_line:
        corrected = corrected + float(np.mean(values)
                                      - np.mean(corrected))
    return corrected, strength


# =============================================================================
# The debug figure - shared by the per-field runtime hook and the offline
# report instrument
# =============================================================================

REGION_SHADING = (
    # attribute on MeasurementWindow, shade color, label
    ("entry_context", "#c8b4e6", "active area"),
    ("front_porch", "#a8d8a8", "front porch"),
    ("sync_tip", "#a8c4e0", "sync tip"),
    ("back_porch", "#a8d8a8", "back porch"),
)

HEALTH_COLORS = {"healthy": "#3a9a3a", "rejected": "#c8c8c8",
                 "reset": "#c03030"}


def render_debug_figure(state, geometry, window_plan, title):
    """The measurement-and-model debug figure, one per decode or field.

    Panels: the accumulated mean interval (full scale and zoomed to
    blanking with the fitted model overlaid), the per-lag variance, the
    fitted artifact model (pole table and the corrector response it
    implies), the entry-amplitude strata, and the per-field health
    timeline.  Returns a matplotlib figure; the caller shows or saves it.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    anchor = window_plan.sync_fall_index
    lags = np.arange(window_plan.total_samples) - anchor
    interval_mean = state.get("interval_mean")
    model = state.get("model")

    # the "after correction" traces come from the SAME code path the
    # decoder ships (apply_shipping_correction) - the debug-plot
    # contract
    corrected_mean = None
    corrected_rise = None
    applied_strength = 0.0
    inversion_info = {}
    rise_mean = state.get("rise_interval_mean")
    if interval_mean is not None and model is not None:
        corrected_mean, applied_strength = apply_shipping_correction(
            interval_mean, state, geometry, diagnostics=inversion_info)
        if rise_mean is not None:
            corrected_rise, _ = apply_shipping_correction(
                rise_mean, state, geometry)

    figure = plt.figure(figsize=(14, 21))
    grid = figure.add_gridspec(
        9, 2,
        height_ratios=[1.8, 1.8, 1.8, 1.6, 1.2, 1.8, 2.0, 1.5, 0.35],
        hspace=0.50, wspace=0.18)
    figure.suptitle(title, fontsize=12)

    def shade(axis, label=False):
        for name, color, text in REGION_SHADING:
            start, stop = getattr(window_plan, name)
            if stop <= start:
                continue
            axis.axvspan(start - anchor, stop - anchor, color=color,
                         alpha=0.35, linewidth=0)
            if label:
                axis.text((start + stop) / 2.0 - anchor, 0.99, text,
                          transform=axis.get_xaxis_transform(),
                          ha="center", va="top", fontsize=7,
                          color="#444444")

    # ---- 1: mean interval, full scale ---------------------------------
    axis_full = figure.add_subplot(grid[0, :])
    shade(axis_full, label=True)
    porch = tip = float("nan")
    if interval_mean is not None:
        axis_full.plot(lags, interval_mean, color="#202020", linewidth=1.2)
        porch = _median_of(interval_mean, window_plan.front_porch)
        tip = _median_of(interval_mean, window_plan.sync_tip)
        back = _median_of(interval_mean, window_plan.back_porch)
        # the two porch levels sit within an IRE of each other on an
        # 80 IRE axis, so their labels are staggered vertically
        for level, name, label_offset in (
                (porch, "front porch", +4.0), (tip, "sync tip", 0.0),
                (back, "back porch", -4.0)):
            axis_full.axhline(level, color="#b04040", linewidth=0.6,
                              linestyle="--", alpha=0.7)
            axis_full.text(lags[-1] + 1, level + label_offset,
                           f"{name} {level:+.2f}",
                           fontsize=7, va="center", color="#b04040")
    def mark_transitions(axis):
        """The start and end of each measured transition (its 10% and
        90% points, half the measured width either side of the
        crossing).  The sync edges are solid; the active falling edge -
        whose width differs from the sync edges' - is dotted."""
        if model is None:
            return
        for name, style in (("sync_fall", "-"), ("sync_rise", "-"),
                            ("active_fall", ":")):
            crossing, width = model["edges"][name][0], model["edges"][name][1]
            for boundary in (crossing - width / 2.0,
                             crossing + width / 2.0):
                axis.axvline(boundary - anchor, color="#7030a0",
                             linewidth=0.7, linestyle=style, alpha=0.8)

    mark_transitions(axis_full)
    axis_full.set_ylabel("IRE")
    axis_full.set_title("accumulated mean interval (levels are measured, "
                        "not nominal; purple marks = measured transition "
                        "start/end)", fontsize=9)
    axis_full.grid(True, alpha=0.25)

    # ---- 1b: the pulse area AFTER the shipping correction --------------
    axis_corrected = figure.add_subplot(grid[1, :], sharex=axis_full)
    shade(axis_corrected)
    if interval_mean is not None:
        axis_corrected.plot(lags, interval_mean, color="#a0a0a0",
                            linewidth=0.9, label="measured")
    if corrected_mean is not None:
        axis_corrected.plot(lags, corrected_mean, color="#207020",
                            linewidth=1.2,
                            label=f"after applied correction "
                                  f"(strength {applied_strength:.2f})")
    mark_transitions(axis_corrected)
    axis_corrected.set_ylabel("IRE")
    axis_corrected.legend(loc="lower right", fontsize=7)
    axis_corrected.set_title(
        "pulse area after the applied correction (same code path as the "
        "decoder)", fontsize=9)
    axis_corrected.grid(True, alpha=0.25)

    # ---- 2: artifact view with the fitted model overlaid --------------
    axis_zoom = figure.add_subplot(grid[2, :], sharex=axis_full)
    shade(axis_zoom)
    if interval_mean is not None:
        axis_zoom.plot(lags, interval_mean - porch, color="#202020",
                       linewidth=1.2, label="measured - front porch")
        axis_zoom.plot(lags, interval_mean - tip, color="#3060a0",
                       linewidth=1.0, alpha=0.8, label="measured - tip")
    if model is not None:
        # the model is drawn over its WHOLE domain - it starts at each
        # transition, exactly where the artifact is excited; the fitted
        # lags (where it was actually constrained) are drawn solid on top
        modeled = model["modeled"]
        axis_zoom.plot(lags, modeled - porch, color="#c03030",
                       linewidth=0.9, linestyle="--", alpha=0.55,
                       label="fitted model")
        constrained = np.where(model["fitted_lags"], modeled, np.nan)
        axis_zoom.plot(lags, constrained - porch, color="#c03030",
                       linewidth=1.4)
        axis_zoom.plot(lags, np.where(model["fitted_lags"], modeled, np.nan)
                       - tip, color="#e08030", linewidth=1.4, alpha=0.8,
                       label="fitted model (tip frame)")
    axis_zoom.axhline(0.0, color="#b04040", linewidth=0.6,
                      linestyle="--", alpha=0.7)
    mark_transitions(axis_zoom)
    axis_zoom.set_ylim(-3.0, 3.0)
    axis_zoom.set_ylabel("IRE from porch / tip")
    axis_zoom.legend(loc="lower right", fontsize=7)
    axis_zoom.set_title("artifact view: measured vs fitted model "
                        "(dashed; gaps = edge cores and content, which "
                        "carry no fit weight)", fontsize=9)
    axis_zoom.grid(True, alpha=0.25)

    # ---- 2b: frame zooms - sync tip (fall) and back porch (rise) -------
    # minimal panes, two traces each: the measured accumulation and the
    # same accumulation through the applied correction.  The fall frame
    # centers on the SYNC TIP (referenced to the tip level so the tip
    # itself is visible); the rise frame shows the BACK PORCH from the
    # rise-aligned accumulation (the phase-honest view of the rise
    # aftermath).
    axis_tip_zoom = figure.add_subplot(grid[3, 0])
    axis_bp_zoom = figure.add_subplot(grid[3, 1])
    guard = window_plan.level_guard_samples
    if interval_mean is not None:
        tip_level = _median_of(interval_mean, window_plan.sync_tip)
        start = max(window_plan.sync_tip[0] - 2 * guard, 0)
        stop = min(window_plan.sync_tip[1] + 2 * guard,
                   len(interval_mean))
        span = np.arange(start, stop)
        axis_tip_zoom.plot(span - anchor, interval_mean[span] - tip_level,
                           color="#202020", linewidth=1.1,
                           label="measured")
        if corrected_mean is not None:
            corrected_tip = _median_of(corrected_mean,
                                       window_plan.sync_tip)
            axis_tip_zoom.plot(span - anchor,
                               corrected_mean[span] - corrected_tip,
                               color="#207020", linewidth=1.1,
                               label="after correction")
        axis_tip_zoom.set_ylim(-3.0, 3.0)
    if rise_mean is not None:
        bp_level = _median_of(rise_mean, window_plan.back_porch)
        start = max(window_plan.back_porch[0] - 2 * guard, 0)
        stop = min(window_plan.back_porch[1] + guard, len(rise_mean))
        span = np.arange(start, stop)
        axis_bp_zoom.plot(span - anchor, rise_mean[span] - bp_level,
                          color="#202020", linewidth=1.1,
                          label="measured (rise-aligned)")
        if corrected_rise is not None:
            corrected_bp = _median_of(corrected_rise,
                                      window_plan.back_porch)
            axis_bp_zoom.plot(span - anchor,
                              corrected_rise[span] - corrected_bp,
                              color="#207020", linewidth=1.1,
                              label="after correction")
        axis_bp_zoom.set_ylim(-3.0, 3.0)
    axis_tip_zoom.set_title("fall frame zoom (sync tip, IRE from tip)",
                            fontsize=9)
    axis_bp_zoom.set_title("rise frame zoom (back porch, IRE from back "
                           "porch; rise-aligned)", fontsize=9)
    for axis in (axis_tip_zoom, axis_bp_zoom):
        axis.axhline(0.0, color="#888888", linewidth=0.5)
        axis.legend(loc="lower right", fontsize=7)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("samples from sync fall midpoint", fontsize=8)

    # ---- 3: per-lag variance ------------------------------------------
    axis_variance = figure.add_subplot(grid[4, :], sharex=axis_full)
    shade(axis_variance)
    variance = state.get("interval_variance")
    if variance is not None:
        axis_variance.semilogy(lags, np.maximum(variance, 1e-4),
                               color="#202020", linewidth=1.0)
        porch_variance = float(np.mean(
            variance[slice(*window_plan.front_porch)]))
        axis_variance.axhline(4.0 * porch_variance, color="#b04040",
                              linewidth=0.6, linestyle="--",
                              label="4x front porch (content threshold)")
        axis_variance.legend(loc="upper right", fontsize=7)
    axis_variance.set_ylabel("IRE$^2$")
    axis_variance.set_title("cross-line variance per lag", fontsize=9)
    axis_variance.grid(True, alpha=0.25, which="both")

    # ---- 4: the fitted artifact model ---------------------------------
    axis_poles = figure.add_subplot(grid[5, 0])
    axis_response = figure.add_subplot(grid[5, 1])
    axis_poles.axis("off")
    if model is not None and model["sections"]:
        rate = geometry.sample_rate_mhz
        rows = ["family     freq MHz   decay us   significance"]
        for section in model["sections"]:
            pole = section["pole"]
            frequency = abs(np.angle(pole)) / (2.0 * math.pi) * rate
            decay_us = (-1.0 / math.log(min(abs(pole), 1.0 - 1e-6))) / rate
            family = ("ringing" if _mode_rotation(pole)
                      > RINGING_MINIMUM_ROTATION else "smear")
            rows.append(f"{family:8s}   {frequency:7.3f}   {decay_us:8.3f}"
                        f"   {section['alpha']:6.3f}")
        rate_mhz = geometry.sample_rate_mhz
        rows.append("")
        rows.append(f"sync pulse width (50%-50%): "
                    f"{model['measured_sync_width'] / rate_mhz:.3f} us "
                    f"measured, {geometry.sync_pulse_samples / rate_mhz:.3f}"
                    f" us spec")
        rows.append("transitions (10-90%): sync edges "
                    f"{model['edges']['sync_fall'][1] / rate_mhz:.3f} us"
                    " (shared width, both edges)   active fall "
                    f"{model['edges']['active_fall'][1] / rate_mhz:.3f} us")
        ghost = model["ghost"]
        rows.append("")
        rows.append(f"ghost scan: peak {ghost['significance']:.1f} sigma"
                    f" at delay {ghost['lag']:+.0f}"
                    f" (amplitude {ghost['amplitude']:+.3f};"
                    f" {ghost.get('scanned_us', 0.0):.1f} us covered)")
        tracked_ghost = state.get("ghost_track") or {}
        # a tracked ghost is only REPORTED as a ghost when its effect
        # on a full sync-depth edge clears the visibility floor - the
        # fitted mixes on ghost-free recordings measure ~+-0.002
        # (invisible), and reporting those as ghosts misled review
        ghost_effect = (abs(tracked_ghost.get("mix", 0.0))
                        * geometry.sync_depth_ire)
        if ghost_effect >= VISIBILITY_FLOOR_IRE:
            rows.append(
                f"tracked ghost: mix {tracked_ghost['mix']:+.4f} at "
                f"delay {tracked_ghost.get('lag', float('nan')):+.1f} "
                f"samples")
        else:
            rows.append("no ghost certified (tracked mix below the "
                        "visibility floor)")
        echo_taps_report = model.get("echo_taps") or []
        if echo_taps_report:
            cells = [f"{tap['lag'] / geometry.sample_rate_mhz:.2f}us "
                     f"mix {tap['mix']:+.4f}"
                     for tap in echo_taps_report]
            rows.append("echo taps (delayed drive): "
                        + "  ".join(cells))
        rows.append(f"stability (vs previous field): "
                    f"{model['stability']:+.4f}")
        rows.append(f"largest |R|: {model['largest_response']:.3f}"
                    f"  (limit {ARTIFACT_MAGNITUDE_LIMIT}, "
                    f"scale {model['margin_scale']:.3f})")
        rows.append(f"mean noise floor: {model['noise_floor']:.4f} IRE")
        if inversion_info:
            rows.append(
                f"fixed-point inverse: {inversion_info['iterations']} "
                f"iterations, movement {inversion_info['movement']:.4f}"
                f" IRE vs floor {inversion_info['noise_floor']:.4f} - "
                + ("CONVERGED (exact inverse: artifact energy "
                   "consolidated into the transition)"
                   if inversion_info["converged"] else "CAPPED"))
        if corrected_mean is not None:
            # the requirements document's indicators, measured on the
            # CORRECTED accumulation: fp and bp normalize to the same
            # level, and the transitions stay as sharp as measured (a
            # widened corrected edge means the correction smoothed the
            # transition's own energy - a failure even when the
            # aftermath reads flatter)
            corrected_fp = _median_of(corrected_mean,
                                      window_plan.front_porch)
            corrected_bp = _median_of(corrected_mean,
                                      window_plan.back_porch)
            rows.append(
                f"porch equality after correction: fp {corrected_fp:+.2f}"
                f"  bp {corrected_bp:+.2f}  difference "
                f"{corrected_bp - corrected_fp:+.2f} IRE")
            widths = []
            for name in ("sync_fall", "sync_rise"):
                crossing, width, before_level, after_level = \
                    model["edges"][name]
                measured_crossing, measured_width = _measure_edge(
                    corrected_mean, int(round(crossing)),
                    window_plan.crossing_search_radius,
                    before_level, after_level)
                if np.isfinite(measured_width) and np.isfinite(width):
                    widths.append(f"{name} {width / rate:.3f} -> "
                                  f"{measured_width / rate:.3f} us")
            if widths:
                rows.append("edge 10-90 after correction: "
                            + "   ".join(widths)
                            + "   (wider = smoothed transition, FAIL)")
        axis_poles.text(0.0, 1.0, "\n".join(rows), family="monospace",
                        fontsize=8, va="top", transform=axis_poles.transAxes)
        frequencies = (model["response_angles"]
                       / (2.0 * math.pi) * rate)
        # the shipped corrector is the gated feedforward subtraction
        # (hsEE1): for one isolated accepted event its response is
        # 1 - sR, not the former exact inverse 1/(1 + sR); the panel
        # shows exactly what ships for that regime
        corrector = 1.0 - model["margin_scale"] * model["response"]
        axis_response.plot(frequencies, np.abs(1.0 + model["response"]),
                           color="#202020", linewidth=1.0,
                           label="channel |1 + R|")
        axis_response.plot(frequencies, np.abs(corrector),
                           color="#c03030", linewidth=1.0,
                           label="corrector |1 - sR| (isolated event)")
        axis_response.axhline(1.0, color="#888888", linewidth=0.5)
        axis_response.set_xlabel("MHz")
        axis_response.legend(fontsize=7)
        axis_response.grid(True, alpha=0.25)
    axis_poles.set_title("fitted artifact model", fontsize=9, loc="left")
    axis_response.set_title("frequency response", fontsize=9)

    # ---- 5: fall-aligned amplitude strata ------------------------------
    axis_strata = figure.add_subplot(grid[6, 0])
    axis_residual = figure.add_subplot(grid[6, 1])
    strata = state.get("fall_strata", {})
    if strata:
        import matplotlib.pyplot as _plt
        colormap = _plt.get_cmap("viridis")
        before_fall, after_fall = _fall_window_layout(geometry, window_plan)
        fall_lags = np.arange(-before_fall, after_fall)
        indices = [index for index in sorted(strata)
                   if strata[index]["line_count"] >= MINIMUM_USABLE_LINES]
        highest = max(indices) if indices else 1
        sections = model["sections"] if model else []
        kernel = (_edge_drive_kernel(model["edges"]["sync_fall"][1])
                  if model else None)
        for index in indices:
            stratum = strata[index]
            counts = stratum["lag_count"]
            valid = counts >= 0.5 * stratum["line_count"]
            mean = stratum["window_sum"] / np.maximum(counts, 1.0)
            amplitude = stratum["amplitude_sum"] / stratum["line_count"]
            trailing = np.flatnonzero(valid)
            porch_level = float(np.median(mean[trailing[-6:]]))
            color = colormap(index / max(highest, 1))
            shown = np.where(valid, mean - porch_level, np.nan)
            axis_strata.plot(fall_lags, shown, color=color, linewidth=1.0,
                             label=f"{amplitude:.0f} IRE"
                                   f" ({stratum['line_count']})")
            if sections and kernel is not None:
                entry_level = float(np.median(
                    mean[:max(before_fall
                              - geometry.transition_settle_samples, 2)]))
                crossing, width = _measure_edge(
                    mean, before_fall,
                    geometry.transition_settle_samples,
                    entry_level, porch_level)
                if np.isfinite(crossing) and np.isfinite(width):
                    positions = np.arange(len(mean), dtype=np.float64)
                    reference = entry_level + (porch_level - entry_level) \
                        * _smooth_step(positions, crossing, width)
                    predicted = reference + _modeled_artifact(
                        reference, sections, kernel, 1.0)
                    # only the aftermath was fitted: the fall core and
                    # the content before it carry no fit weight, and
                    # drawing them would bury the part that matters
                    aftermath = valid & (positions
                                         >= math.ceil(crossing + width))
                    axis_strata.plot(
                        fall_lags,
                        np.where(aftermath, predicted - porch_level,
                                 np.nan),
                        color=color, linewidth=0.9, linestyle="--",
                        alpha=0.7)
                    axis_residual.plot(
                        fall_lags,
                        np.where(aftermath, mean - predicted, np.nan),
                        color=color, linewidth=1.0)
        axis_strata.set_ylim(-3.0, 3.0)
        axis_strata.axhline(0.0, color="#888888", linewidth=0.5)
        axis_strata.legend(loc="lower left", fontsize=6.5,
                           title="entry amplitude (lines)",
                           title_fontsize=7)
        axis_residual.set_ylim(-1.5, 1.5)
        axis_residual.axhline(0.0, color="#888888", linewidth=0.5)
    axis_strata.set_title("active falling edge by amplitude, aligned at "
                          "the fall (solid = measured, dashed = model "
                          "with amplitude slopes)", fontsize=9)
    # the sideband amplitude law (hsEE5): the per-stratum wanted scale
    # of the shipped unit-fall correction, and the applied law
    landing_law = (model or {}).get("event_landing_law")
    if landing_law is not None:
        law_axis = axis_residual.inset_axes([0.62, 0.62, 0.36, 0.34])
        steps = [point[0] for point in landing_law["points"]]
        wants = [point[1] for point in landing_law["points"]]
        law_axis.plot(steps, wants, "o", markersize=3,
                      color="#c03030")
        law_axis.axhline(landing_law["scales"][1], color="#c03030",
                         linewidth=1.0, linestyle="--")
        law_axis.axhline(1.0, color="#3a9a3a", linewidth=1.0)
        law_axis.set_ylim(-0.5, 1.3)
        law_axis.tick_params(labelsize=5.5)
        law_axis.set_title(
            f"sideband law: picture-fall want"
            f" {landing_law['scales'][1]:.2f} (sync = 1)",
            fontsize=6.5)
        law_axis.grid(True, alpha=0.25)
    axis_residual.set_title("measured minus model, per stratum "
                            "(what the amplitude extension leaves)",
                            fontsize=9)
    for axis in (axis_strata, axis_residual):
        axis.set_xlabel("samples from the active falling edge midpoint")
        axis.grid(True, alpha=0.25)

    # ---- 5b: Wiener residual spectra (identification + validation) ----
    # the frequencies still locked to the transient, before and after
    # the applied correction - Ethan's deconvolution instrument.  A
    # peak above the noise reference is a component the transient still
    # excites in the corrected output.
    axis_wiener = figure.add_subplot(grid[7, :])
    if model is not None and model.get("wiener"):
        rate = geometry.sample_rate_mhz
        slow_horizon = SLOW_HORIZON_FACTOR * max(
            state.get("average_fields", 1), 1)
        lines_average = max(state.get("usable_lines_average", 1.0), 1.0)
        for view_name, color_before, color_after in (
                ("fall", "#909090", "#207020"),
                ("rise", "#404040", "#66aa30")):
            spectrum = model["wiener"].get(view_name)
            if spectrum is None:
                continue
            frequencies, magnitude, reference_level = spectrum
            axis_wiener.plot(frequencies, magnitude,
                             color=color_before, linewidth=0.9,
                             label=f"{view_name} measured")
            # the same instrument on the CORRECTED accumulation (the
            # shipping path - validation of what remains)
            if view_name == "fall":
                accumulation = state.get("slow_interval_mean")
                reference = model["excitation"]
                view_weights = model.get("fall_weights")
                variance = state.get("slow_interval_variance")
                accumulated = min(
                    state.get("slow_fields_accumulated", 1),
                    slow_horizon)
            else:
                view = model.get("rise_view") or {}
                accumulation = state.get("rise_interval_mean")
                reference = view.get("excitation")
                view_weights = view.get("weights")
                variance = state.get("rise_interval_variance")
                accumulated = min(
                    state.get("rise_fields_accumulated", 1),
                    slow_horizon)
            if (accumulation is None or reference is None
                    or view_weights is None or variance is None):
                continue
            corrected_view, _ = apply_shipping_correction(
                accumulation, state, geometry)
            _, after_magnitude, _ = _wiener_residual_spectrum(
                corrected_view - reference, reference, view_weights,
                np.maximum(variance, 1e-6)
                / max(lines_average * accumulated, 1.0), rate)
            axis_wiener.plot(frequencies, after_magnitude,
                             color=color_after, linewidth=1.1,
                             label=f"{view_name} after correction")
        frequencies, _, reference_level = model["wiener"]["fall"]
        axis_wiener.plot(frequencies, 3.0 * reference_level,
                         color="#b04040", linewidth=0.7,
                         linestyle=":", label="3x noise reference")
        # limited to the ring family's band: above the chroma
        # subcarrier the spectrum is burst / FM remnant (line-locked
        # but not transient-driven) and would swamp the axis
        band_limit = min(geometry.luma_lowpass_mhz,
                         geometry.sample_rate_mhz / 4.0)
        axis_wiener.set_xlim(0.0, band_limit)
        in_band = frequencies <= band_limit
        tallest = max(
            float(np.max(np.abs(line.get_ydata())[in_band]))
            for line in axis_wiener.get_lines())
        axis_wiener.set_ylim(0.0, 1.2 * max(tallest, 1e-6))
        axis_wiener.legend(fontsize=7, ncol=3)
    axis_wiener.set_title(
        "Wiener residual spectrum: frequencies still locked to the "
        "transient (identification/validation only - the fit stays in "
        "the time domain)", fontsize=9)
    axis_wiener.set_xlabel("MHz", fontsize=8)
    axis_wiener.grid(True, alpha=0.25)

    # ---- 6: health timeline -------------------------------------------
    axis_health = figure.add_subplot(grid[8, :])
    history = state.get("health_history", [])
    if history:
        strip = np.array([
            matplotlib.colors.to_rgb(HEALTH_COLORS[entry])
            for entry in history])
        axis_health.imshow(strip[None, :, :], aspect="auto",
                           extent=(0, len(history), 0, 1))
    axis_health.set_yticks([])
    axis_health.set_xlabel(
        "field number   (green = accumulated, grey = rejected by the "
        "health gate, red = state reset)", fontsize=8)
    return figure


def _show_debug_figure(state, geometry, window_plan):
    """Per-field interactive debug plot (--debug_plot hsync_model)."""
    import matplotlib.pyplot as plt

    field_number = len(state.get("health_history", []))
    figure = render_debug_figure(
        state, geometry, window_plan,
        f"hsync artifact model - after field {field_number}")
    plt.show()
    plt.close(figure)


def process_field(video_buffer, geometry, sync_tip_level, blanking_level,
                  shared_state, average_fields, head_parity=None,
                  debug=False, apply_correction=True):
    """Measure this field's sync intervals, fit the model, correct the field.

    The buffer (in demodulated signal units) is converted to IRE using
    the supplied reference levels, measured (always before this field's
    correction), the artifact model is fitted and tracked, and the
    modeled artifacts are inverted across the whole field.  Returns the
    corrected buffer and the parameters for the luma transient
    improvement stage.  With `debug` true the debug figure is shown
    after each field (--debug_plot hsync_model).
    """
    state = shared_state.setdefault(STATE_KEY, {})
    # PER-HEAD MODELS (hsEE6, the user's directive): each video head
    # is its own channel - measured on the countdown tape, one pooled
    # model ANTI-corrected head A's sync tip (0.43 -> 0.58) while
    # improving head B's, and the sibling arc independently measured
    # 2-3x per-head transfer differences.  The field's parity routes
    # accumulation, track, model, governor, gate, law and strata into
    # that head's own substate; each head is corrected by the model
    # calibrated on its own channel.
    if head_parity is not None:
        state = state.setdefault(
            "head1" if head_parity else "head0", {})
        # parity1 = head A under the standing capture recipes (the
        # sibling arc's measured mapping); names key the profile sidecar
        state["head_name"] = "head_a" if head_parity else "head_b"
    if "geometry" not in state:
        state["geometry"] = geometry

    level_units_per_ire = (
        (blanking_level - sync_tip_level) / geometry.sync_depth_ire)
    if not np.isfinite(level_units_per_ire) or level_units_per_ire == 0.0:
        # the caller unpacks (buffer, lti_parameters) - degrade with the
        # same neutral parameters the no-model path returns
        return video_buffer, {"gain": 0.0, "threshold": 0.1,
                              "blur_radius": 0.0}

    samples_per_line = geometry.samples_per_line
    whole_lines = len(video_buffer) // samples_per_line
    field_lines_ire = (
        np.asarray(video_buffer[:whole_lines * samples_per_line],
                   dtype=np.float64).reshape(whole_lines, samples_per_line)
        - blanking_level) / level_units_per_ire

    window_plan = plan_measurement_window(geometry)
    measure_field_lines(
        field_lines_ire, geometry, window_plan, state, average_fields)

    # fit and track the artifact model once enough fields have
    # accumulated - always on the measurement taken BEFORE this field's
    # correction, so the model never sees its own output
    if state.get("slow_fields_accumulated", 0) >= max(average_fields, 1):
        fit_artifact_model(state, geometry, window_plan, average_fields)

    # Stage D: invert the modeled artifacts across the whole field -
    # one linear operator, applied identically to sync, blanking and
    # picture (a correction confined to blanking hides its picture
    # behavior from every sync gauge; measured lesson from the previous
    # pipeline)
    lti_parameters = {"gain": 0.0, "threshold": 0.1, "blur_radius": 0.0}
    model = state.get("model")
    if model is not None:
        # while the identified RF channel response is applied (channel_eq)
        # the RF-derived synthesized kernels describe physics that stage
        # now carries; the gate mutes their blend unless the sidecar says
        # otherwise (synth_blend_under_channel_eq)
        model["channel_eq_active"] = bool(
            shared_state.get("channel_eq_active", False))
    strength = _correction_strength(state, average_fields)
    # measure-only mode (--debug_plot hsync_model without --inverse_eq):
    # the model is measured, fitted and SHOWN, but the buffer is
    # returned untouched - the plot is available on any decode without
    # changing its output
    if model is not None and strength > 0.0 and apply_correction:
        sections = model["sections"]
        edge_kernel = _edge_drive_kernel(model["edges"]["sync_fall"][1])
        # The amplitude dependence is FITTED AND REPORTED but not yet
        # applied.  Measured on the picture-amplitude strata: applying
        # it closes the residuals
        # at the strata it was fitted from - isolated content-to-porch
        # falls at the line ends - and HARMS the visible in-picture
        # sites (the mic stand's narrow bar 2.29 -> 2.93 IRE, the stage
        # light 5.32 -> 5.64), because a bar's superposed edges are a
        # different excitation regime than an isolated fall.  The third
        # independent falsification of amplitude-scaled application on
        # this project; the witness for in-picture features does not
        # exist yet, and until it does the correction stays linear.
        inversion_diagnostics = {}
        # the runtime input is a RAW field: its even-drive debias uses
        # the per-line noise, not the accumulated floor the fit and the
        # figures correct at (see apply_shipping_correction)
        line_noise_ire = math.sqrt(
            float(np.median(state["slow_interval_variance"])))
        corrected_ire, strength = apply_shipping_correction(
            field_lines_ire.reshape(-1), state, geometry,
            diagnostics=inversion_diagnostics,
            input_noise_ire=line_noise_ire)
        state["inversion_diagnostics"] = inversion_diagnostics
        corrected = (corrected_ire * level_units_per_ire + blanking_level)
        if np.all(np.isfinite(corrected)):
            # THE CORRECTION GOES WHERE THE MEASUREMENT WENT, and nowhere
            # else. The model is calibrated on
            # [first_measurable_line, last_measurable_line) and was being
            # applied to every sample of the buffer - the vertical
            # interval, everything above the first measurable line, and the
            # field's FINAL row, which `build_geometry` excludes precisely
            # because it spans the field boundary and carries the next
            # field's half-line equalizing pulse. Those rows never entered
            # an accumulator, a template, a stratum, a variance or a fit
            # weight, and correcting them applies a model nothing validated
            # against them.
            #
            # The filter is still RUN over the whole buffer, because it is
            # causal with memory across line boundaries and restricting its
            # input would change its state at every row it did reach. Only
            # the write-back is restricted, so the rows the model owns get
            # exactly the correction they would have got, and the rest are
            # left as they arrived.
            #
            # Applying to sync, blanking AND picture alike within those
            # rows is deliberate and unchanged: a correction confined to
            # blanking hides its picture behaviour from every sync gauge.
            corrected = corrected.reshape(whole_lines, samples_per_line)
            first_row = max(int(geometry.first_measurable_line), 0)
            last_row = min(int(geometry.last_measurable_line), whole_lines)
            if last_row > first_row:
                buffer_rows = video_buffer[:whole_lines * samples_per_line]
                buffer_rows = buffer_rows.reshape(whole_lines, samples_per_line)
                buffer_rows[first_row:last_row] = (
                    corrected[first_row:last_row].astype(video_buffer.dtype))
        # the corrector's impulse must cover the slowest component's
        # settle to the level tolerance, or a long smear correction is
        # truncated in the luma-transient stage's view of it: length =
        # tau_max ln(1 / tap tolerance), floored at the historical 128
        # and capped at one line
        slowest = max(
            [-1.0 / math.log(min(abs(section["pole"]), 1.0 - 1e-6))
             for section in sections] + [1.0])
        tap_tolerance = (VISIBILITY_FLOOR_IRE
                         / (geometry.sync_depth_ire + 100.0))
        # an echo tap's corrector needs the lag itself plus the edge
        # kernel's own reach beyond it
        echo_reach = max(
            [tap["lag"] + len(edge_kernel)
             for tap in model.get("echo_taps", [])] + [0.0])
        impulse_length = int(min(max(
            128, math.ceil(slowest * math.log(1.0 / tap_tolerance)),
            math.ceil(2.0 * echo_reach)),
            geometry.samples_per_line))
        impulse = _corrector_impulse_response(
            sections, edge_kernel, strength, impulse_length,
            echo_taps=model.get("echo_taps"),
            gate=_gate_parameters(model, geometry, 0.0, strength))
        state["corrector_impulse"] = impulse
        # the luma transient improvement stage reads the corrector's
        # impulse response from its direct tap onward, and a noise level
        # measured from the signal itself (per-line noise as a fraction
        # of the sync depth) rather than an assumed one
        direct_tap = int(np.argmax(np.abs(impulse)))
        line_noise_ire = math.sqrt(
            float(np.median(state["slow_interval_variance"])))
        lti_parameters = derive_lti_parameters(
            impulse[direct_tap:].astype(np.float32),
            noise_threshold=line_noise_ire / geometry.sync_depth_ire)

    if debug:
        _show_debug_figure(state, geometry, window_plan)
    return video_buffer, lti_parameters


# =============================================================================
# The luma transient improvement (LTI) stage - fed by the applied
# corrector's own impulse response (process_field) and run by field.py
# under --lti_gain
# =============================================================================

def derive_lti_parameters(fir_kernel, noise_threshold):
    """
    Derives optimal Luminance Transient Improvement (LTI) parameters
    analytically from the effective correction kernel.
    """
    total_energy = np.sum(fir_kernel**2)
    if total_energy <= 1e-12:
        return {'gain': 0.0, 'threshold': 0.1, 'blur_radius': 0.0}

    center_energy = fir_kernel[0]**2
    energy_dispersion = 1.0 - (center_energy / total_energy)

    indices = np.arange(len(fir_kernel))
    weighted_spread = np.sum(indices * np.abs(fir_kernel)) / np.sum(np.abs(fir_kernel))

    noise_suppression_factor = max(0.2, 1.0 - 2.0 * noise_threshold)
    lti_gain = np.clip(energy_dispersion * 1.5 * noise_suppression_factor, 0.0, 1.0)
    lti_threshold = np.clip(noise_threshold * 0.75, 0.02, 0.15)

    return {
        'gain': float(lti_gain),
        'threshold': float(lti_threshold),
        'blur_radius': float(weighted_spread),
        'dispersion': float(energy_dispersion),
    }


@nb.njit(cache=True, nogil=True, fastmath=True)
def apply_adaptive_luma_transient_improvement(video_buf, gain, threshold):
    """
    Applies non-linear LTI using parameters derived from the group delay kernel.
    Modifies video_buf in place.
    """
    n = len(video_buf)
    prev = video_buf[0]
    for i in range(1, n - 1):
        curr = video_buf[i]
        nxt = video_buf[i + 1]
        diff = nxt - prev
        abs_diff = abs(diff)
        if abs_diff > threshold:
            grad = curr - prev
            edge_weight = min(1.0, abs_diff / (2.0 * threshold))
            video_buf[i] = curr + (gain * edge_weight) * grad
        prev = curr


if __name__ == "__main__":
    # Verification: derive and print the geometry and window layout for
    # both video standards using the decoder's own parameter tables.
    from vhsdecode.formats import get_format_params, parse_tape_speed
    import logging

    for name, line_offset, line_count in (("NTSC", 0, 263), ("PAL", 2, 312)):
        sys_params, decoder_params = get_format_params(
            name, "VHS", parse_tape_speed("sp"), logging.getLogger(__name__))
        geometry = build_geometry(
            sys_params, decoder_params, sys_params["outlinelen"],
            line_offset, line_count)
        print(f"\n=== {name}")
        print(describe_geometry(geometry))
        print(describe_window(plan_measurement_window(geometry)))
