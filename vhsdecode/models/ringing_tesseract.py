"""RINGING CANCELLATION ON THE TESSERACT GRAPH: one field, one collapse.

Ethan, on what replaces the old ringing module:

  *"I want you to delete the ringing cancellation module and replace it
  with the model based components we have identified. Ringing cancellation
  module is not using this method and needs to be removed and its concepts
  need to be implemented as our multi dimensional component stages."*

  *"We just need to do this whole operation on one field at a time, we can
  remove any averaging that was happening in the ringing removal. Our new
  transform based stage supersedes that work."*

  *"Eventually it collapses down to one real signal which we can subtract
  to remove the residual."*

WHAT THIS MODULE DOES, IN ONE SENTENCE. Every horizontal sync pulse of one
field is read as two known transients - the fall, which lands on the sync
tip, and the rise, which lands on blanking - each event's departure from
the specification's own edge is measured, those departures are the vertices
of a tesseract whose axes are the polarity of the event and the binary
scales of the line index, the cube is folded on all of its dimensions at
once, the contrasts that stand above their own noise and are not chroma
contamination are summed back to ONE REAL SIGNAL per vertex, and that
signal is subtracted from every transient in the field, scaled by the
transient's own signed step.

WHY A FOLD AND NOT A FILTER: THE FIRST DESIGN LAW. It is proved on this
arc that no video-domain linear time-invariant filter can invert the ring:
the chain is linear in the radio-frequency domain and the demodulation is
not, so one magnitude applies to both sweep directions while the artifact
it induces follows the sign of the sweep. `tesseract.one_real_signal` and
`hypercomplex.deconvolve` therefore CANNOT be used as they stand - they
collapse the cube in the LOG-TRANSFER domain and return `exp(-departure)`,
which is a filter. The same fold is used here in the TIME domain, where
the collapse is additive and what comes out is subtracted. The fold is the
identical operator (`tesseract.walsh`, `tesseract.identified`,
`tesseract.Cube`); only the domain, and therefore the realisation, differs.

THE HYPERCOMPLEX CORRESPONDENCE IS WHY THE VERTICES ARE COMPLEX. On a
two-state axis the Fourier transform is the fold, so the fold over the
binary measurement axes IS the hypercomplex analytic signal over those
axes (`hypercomplex`, first paragraph). The remaining axis is the lag, and
its analytic half is taken explicitly: a vertex holds the departure plus
i times its Hilbert transform along the lag axis. That is not decoration.
A contrast between two vertices that differ by a small DELAY has little
in-phase difference and a large quadrature one; read in magnitude alone it
sits at the noise and is discarded. Carrying the analytic form makes the
identification test read both components, which is the difference between
rank one and rank two that this arc has been caught by before. What is
subtracted is the real part, because the capture is a real sample stream.

THE COLOUR BURST IS IN THE LUMA BACK PORCH, AND THE FOLD FINDS IT. The
module this replaces states that "the color burst is removed from luma by
the color-under process, so the whole span is measurable". That is
measurably false, and it matters because the burst sits inside the rise
event's aftermath window. The burst alternates phase from line to line, so
it lands entirely on the line-parity contrast of the fold, where it is
visible and can be refused. The control is decisive - two decodes of the
same deck, the same test pattern and the same flags, differing only in
whether the capture carried chroma at all (`zaroff-ntc7composite-NTSC-SP`
against its `y-only` variant, ten frames each, `-n -f 50 -t 0 -l 10
--inverse_eq -1`):

    line-parity contrast, fields 0-3   inside burst   outside   ratio
    with chroma                        0.42-0.58 IRE  0.20-0.23   2.1-2.7
    y-only, no chroma                  0.04-0.07 IRE  0.04-0.05   0.8-1.6

Removing the chroma from the capture drops the in-burst parity contrast by
six to fifteen times and flattens its concentration to one, while the mean
aftermath the model actually wants is unchanged (2.13-2.28 IRE with chroma
against 2.30-2.36 without, on the fall; 1.50-1.52 against 1.42-1.71 on the
rise). So a contrast concentrated in the burst lags is the chroma path and
not the luma channel, and `chroma_contaminated` refuses it by that
measurement rather than by name. That is why the line-parity axis is always
carried: not to be applied, but so that the burst has somewhere to go other
than the kernel.

WHAT IT ACHIEVES, MEASURED ON SIX DECODES (the first field of each; the
four decoded time-base files in /tmp/claude-1000 plus the two decoded for
the control above). Held out inside the field - the model fitted on half
its lines and scored on the half it never saw - it predicts 97.8 to 99.4
per cent of the aftermath's power. Applied at the standing half gain it
removes 12.0 to 33.6 per cent of each event's own aftermath amplitude,
against a coherent share - the part common to the field's events, which is
all any one kernel can reach - of 34 to 72 per cent of the power. The full
tables are under `choose_depth`, `chroma_contaminated` and `gated_events`.
The module this replaces reached its accuracy by averaging four fields
together; this reaches its own on one field, which is what the directive
at the top asks for.

AND IN THE DECODER, not only in the instrument. Four frames of
`zaroff-ntc7composite-NTSC-SP` decoded at `-n -f 50 -t 0 -l 4
--inverse_eq 4` against the same capture uncorrected: the picture moves by
0.82 IRE rms, 4.1 IRE at the 99th percentile and 7.7 IRE at most, with no
sample moving by ten. That last figure is the one that mattered: the first
run of this decode moved 1.53 per cent of its samples by more than ten IRE
and one by thirty-one, and NONE of it was the subtraction - it was the
luma transient stage downstream, running at a gain of 0.77 to 0.90 because
this module handed it the artifact kernel where it wanted the corrector's
impulse response (`corrector_impulse`). With that corrected the gain it
asks for is 0.010 to 0.017 and the two arms agree to three thousandths of
an IRE.

AND THE PHYSICAL ARGUMENT AGREES WITH THE MEASUREMENT. A linear channel's
response to one edge cannot depend on which line the edge is on. A
line-locked contrast is therefore either contamination or a mechanism that
is not the edge's response, and in both cases it has no business being
convolved with a picture transient.

THE THREE STANDING LAWS THIS OBEYS.

  1. NO VIDEO-DOMAIN INVERSE. The realisation is subtraction of a
     landing/polarity kernel on a gated drive, truncated at what the sync
     interval witnesses (`RingingGeometry.aftermath_samples`, 4.19 us on
     NTSC VHS at four times the subcarrier, which is the sync tip guarded
     by one settling time at each end - the sibling RF-domain synthesis in
     `vhsdecode/luma_transient.py` spans 5.1 us, the same quantity taken
     to the whole sync pulse and unguarded). Nothing here filters.

  2. SYNC ONLY. Every number in the model comes from the horizontal sync
     pulses and the flat blanking regions the specification defines around
     them. The active picture is READ only to place the gated drive - it
     never enters a level, a kernel, a variance or a verdict.

  3. ONE FIELD AT A TIME. `average_fields` is accepted and NOT used to
     blend anything across fields; `shared_state` carries the field's own
     model out for reporting and carries nothing in. This deliberately
     supersedes standing rule 4 of `addons/RINGING_RULES.md` ("never
     remove the temporal averaging"), on the later directive quoted above.

WHAT IT IS NOT. It does not estimate a ghost, it does not re-tune the
de-emphasis, and it does not attempt the level-dependent amplitude law -
the sync interval witnesses exactly two (direction, landing) pairs, so
direction and landing are collinear in it and cannot be separated without
a second landing at the same direction. That is an identifiability
statement, not a deferral, and the arc that CAN see more landings is
`vhsdecode/luma_transient.py`, which synthesises the family from the head's
own measured response.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import hypercomplex
from vhsdecode.models import output_limit
from vhsdecode.models import sync_shape
from vhsdecode.models import tesseract

# The half-amplitude width of the specification's own transition. The
# decoder's format table states the sync transition as a 10-90 time
# (`syncTransitionUS`); an erf 0.5*(1 + erf(t / (sqrt(2) s))) has a 10-90
# width of 2 * 0.9061938 * s, so this factor converts the stated time into
# the erf's scale. It is the same ideal `tools/ringing_measure/
# sync_step_response.py` measures against.
ERF_TEN_NINETY_FACTOR = 2.0 * 0.9061938

# A flat region is read only where the transition before it has settled.
# Three reciprocal bandwidths of the format's own luma low-pass is the
# settling time of a system with that bandwidth to within a few per cent.
# `field.py` reads this so that there is ONE answer in the tree to "how
# long does a transition take to settle".
TRANSITION_SETTLE_BANDWIDTHS = 3.0

# The vertical sync interval is three sections - pre-equalising pulses,
# serrated vertical sync, post-equalising pulses - each `numPulses` pulses
# long at half line rate, so 3 * numPulses / 2 lines. Lines before that
# carry pulses of another width and are not horizontal sync.
VSYNC_SECTIONS = 3

STATE_KEY = "ringing_tesseract"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RingingGeometry:
    """Everything the measurement needs, derived from the format tables.

    Sample counts are held as floats where the specification gives a time,
    so that half a sample of a porch is not rounded away before it is used.
    """

    sample_rate_mhz: float
    samples_per_line: int
    front_porch_samples: float
    sync_pulse_samples: float
    sync_transition_samples: float
    active_video_start_sample: float
    burst_start_sample: float
    burst_end_sample: float
    sync_depth_ire: float
    luma_lowpass_mhz: float
    transition_settle_samples: int
    aftermath_samples: int
    first_measurable_line: int
    last_measurable_line: int
    detail_emphasis: bool

    @property
    def aftermath_us(self) -> float:
        return self.aftermath_samples / self.sample_rate_mhz

    @property
    def burst_lags(self) -> Tuple[int, int]:
        """The colour burst, as lags after the sync RISE crossing.

        The rise's aftermath window starts at the rise and runs forward,
        so the burst - which the format places by time from the line start
        - occupies a known stretch of it. Clipped to the window.
        """
        if not np.isfinite(self.burst_start_sample):
            return (0, 0)
        low = int(math.floor(self.burst_start_sample - self.sync_pulse_samples))
        high = int(math.ceil(self.burst_end_sample - self.sync_pulse_samples))
        return (max(low, 0), min(max(high, 0), self.aftermath_samples))


def build_geometry(sys_params, decoder_params, samples_per_line,
                   line_offset=0, line_count=None,
                   include_vertical_interval=False) -> RingingGeometry:
    """Derive the sync-interval geometry from the supplied format tables.

    The signature matches the module this replaces, so the runtime's call
    site changes by its import alone. `sys_params` is the video standard's
    timing table and `decoder_params` the decoder's configuration;
    together they are the only source of every number below.

    `include_vertical_interval` admits the equalising and serration lines.
    It is accepted for signature compatibility and honoured, but nothing
    in this module asks for it: the correction is calibrated on horizontal
    sync pulses, which is standing rule 1.

    THE TRUNCATION IS DERIVED, NOT CHOSEN. The aftermath of an edge can be
    read only until the next specified transition arrives. After the sync
    fall that is the sync rise, leaving the sync tip; after the sync rise
    it is the start of active video, leaving the back porch. Each is
    shortened by one settling time, because the decoder's zero-phase luma
    low-pass puts a precursor that far ahead of every transition. Both
    polarities must be read over the same lags to be folded, so the window
    is the shorter of the two. From the decoder's own tables: NTSC VHS 60
    samples, 4.191 us (tip 60.30 samples, back porch 61.04); PAL VHS 67
    samples, 3.778 (tip 67.35, back porch 86.86).
    """
    sample_rate_mhz = float(sys_params["outfreq"])
    microseconds_to_samples = sample_rate_mhz

    luma_lowpass_hz = float(decoder_params["video_lpf_freq"])
    transition_settle_samples = int(math.ceil(
        TRANSITION_SETTLE_BANDWIDTHS * sample_rate_mhz * 1e6 / luma_lowpass_hz))

    lines_of_vertical_sync = int(math.ceil(
        VSYNC_SECTIONS * sys_params["numPulses"] / 2.0))

    front_porch = sys_params["frontPorchUS"] * microseconds_to_samples
    pulse = sys_params["hsyncPulseUS"] * microseconds_to_samples
    edge = sys_params["syncTransitionUS"] * microseconds_to_samples
    active_start = sys_params["activeVideoUS"][0] * microseconds_to_samples

    # THE GUARD IS THE SETTLING TIME, not the transition's own width. The
    # aftermath runs from one crossing forward and must stop before the
    # NEXT transition has any effect on it, and a transition's effect
    # begins before its crossing: the decoder's luma low-pass is a
    # zero-phase supergauss, so its impulse response is symmetric and puts
    # a precursor one settling time ahead of every edge. Guarding by the
    # stated 10-90 width instead left the fall's last lag inside the sync
    # rise - measured on a field whose true departure is exactly zero, the
    # last lag read 0.019 per unit step, 0.76 IRE at a 40 IRE step, while
    # every other lag read zero.
    tip_flat = pulse - transition_settle_samples
    back_porch_flat = active_start - pulse - transition_settle_samples
    aftermath = int(math.floor(max(min(tip_flat, back_porch_flat), 1.0)))

    # The colour burst's own window, from the format's colour burst time
    # where it states one. It is used ONLY to recognise chroma
    # contamination in the fold, never to correct anything.
    burst = sys_params.get("colorBurstUS")
    if burst is not None:
        burst_start = float(burst[0]) * microseconds_to_samples
        burst_end = float(burst[1]) * microseconds_to_samples
    else:
        burst_start = burst_end = float("nan")

    if line_count is None:
        line_count = int(sys_params["field_lines"][0])

    return RingingGeometry(
        sample_rate_mhz=sample_rate_mhz,
        samples_per_line=int(samples_per_line),
        front_porch_samples=front_porch,
        sync_pulse_samples=pulse,
        sync_transition_samples=edge,
        active_video_start_sample=active_start,
        burst_start_sample=burst_start,
        burst_end_sample=burst_end,
        sync_depth_ire=abs(float(sys_params["vsync_ire"])),
        luma_lowpass_mhz=luma_lowpass_hz / 1e6,
        transition_settle_samples=transition_settle_samples,
        aftermath_samples=aftermath,
        first_measurable_line=(line_offset if include_vertical_interval
                               else line_offset + lines_of_vertical_sync),
        # The buffer's final row spans the field boundary and carries the
        # next field's half-line-rate pulse, whose tip is half the
        # horizontal width; it is excluded by the specification's own line
        # numbering, never by a waveform test.
        last_measurable_line=int(line_count) + int(line_offset) - 1,
        detail_emphasis=bool(sys_params.get("detail_emphasis", False)),
    )


@dataclasses.dataclass
class MeasurementWindow:
    """The per-line window and the flat regions inside it.

    Kept because the offline instruments read the sync interval as ONE
    window with named regions rather than as two events - the same
    samples, presented the other way round. Offsets are indices into a
    window that starts `samples_before_sync_fall` before the fall.
    """

    samples_before_sync_fall: int
    samples_after_sync_fall: int
    sync_fall_index: int
    total_samples: int
    front_porch: Tuple[int, int]
    sync_tip: Tuple[int, int]
    back_porch: Tuple[int, int]
    level_guard_samples: int
    crossing_search_radius: int


def plan_measurement_window(geometry: RingingGeometry) -> MeasurementWindow:
    """Lay out the window and its flat regions from the specification.

    Each flat region starts at the end of the preceding transition and
    stops one settle guard short of the next one: the artifact's largest
    part - the overshoot and the first lobe - lives in the first
    reciprocal bandwidth after an edge, and guarding it out would delay
    every measurement past the first oscillation. Region ENDS keep their
    guards, because the next transition's band-limited shape begins before
    its crossing and is nobody's artifact.
    """
    settle = geometry.transition_settle_samples
    edge = geometry.sync_transition_samples
    porch = geometry.front_porch_samples
    pulse = geometry.sync_pulse_samples

    # A LEVEL read needs only the transition core out of the slice, because
    # a median tolerates the decaying tail riding on the flat - that tail
    # IS the artifact being measured. One edge duration plus one reciprocal
    # bandwidth; with the full settle guard the front porch slice would be
    # empty on PAL.
    level_guard = int(math.ceil(edge + settle / TRANSITION_SETTLE_BANDWIDTHS))
    samples_before = int(math.ceil(porch + edge + 3 * settle))
    samples_after = int(math.ceil(
        geometry.active_video_start_sample + level_guard + 2 * settle))
    anchor = samples_before
    total = samples_before + samples_after

    def bounded(start, stop):
        start = int(max(0, math.ceil(start)))
        stop = int(min(total, math.floor(stop)))
        return (start, stop) if stop > start else (start, start)

    return MeasurementWindow(
        samples_before_sync_fall=samples_before,
        samples_after_sync_fall=samples_after,
        sync_fall_index=anchor,
        total_samples=total,
        front_porch=bounded(anchor - porch + edge / 2.0, anchor - level_guard),
        sync_tip=bounded(anchor + edge / 2.0,
                         anchor + pulse - (edge / 2.0 + level_guard)),
        back_porch=bounded(anchor + pulse + edge / 2.0,
                           anchor + geometry.active_video_start_sample - settle),
        level_guard_samples=level_guard,
        crossing_search_radius=settle,
    )


def describe_geometry(geometry: RingingGeometry) -> str:
    """One readable block, for the module's own report and for tools."""
    low, high = geometry.burst_lags
    return "\n".join([
        f"sample rate            {geometry.sample_rate_mhz:.6f} MHz",
        f"samples per line       {geometry.samples_per_line}",
        f"front porch            {geometry.front_porch_samples:.2f} samples",
        f"sync pulse             {geometry.sync_pulse_samples:.2f} samples",
        f"sync transition        {geometry.sync_transition_samples:.2f} samples",
        f"active video starts    {geometry.active_video_start_sample:.2f} samples",
        f"colour burst           {geometry.burst_start_sample:.2f}"
        f" .. {geometry.burst_end_sample:.2f} samples"
        f"  (rise lags {low}..{high})",
        f"settle guard           {geometry.transition_settle_samples} samples",
        f"aftermath window       {geometry.aftermath_samples} samples"
        f" = {geometry.aftermath_us:.3f} us",
        f"measurable lines       {geometry.first_measurable_line}"
        f" .. {geometry.last_measurable_line}",
        f"sync depth             {geometry.sync_depth_ire:.4f} IRE",
    ])


def describe_window(plan: MeasurementWindow) -> str:
    return "\n".join([
        f"window                 {plan.total_samples} samples,"
        f" fall at {plan.sync_fall_index}",
        f"front porch            {plan.front_porch}",
        f"sync tip               {plan.sync_tip}",
        f"back porch             {plan.back_porch}",
        f"level guard            {plan.level_guard_samples} samples",
    ])


# ---------------------------------------------------------------------------
# The measurement: every sync pulse of one field, against the specification
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FieldEvents:
    """The field's sync events, per polarity, on a common lag grid.

    `departure` is (polarity, event, lag), in IRE per unit signed step -
    the measured luma less the specification's own edge, divided by the
    event's own step, so that what remains is the channel and not the
    excitation. `step_ire` is that signed step and `landing_ire` the level
    the event settles on, both (polarity, event); `line_index` is the row
    each event came from and `noise_ire` the per-line noise its own flats
    measure.
    """

    departure: np.ndarray
    step_ire: np.ndarray
    landing_ire: np.ndarray
    crossing_sample: np.ndarray
    crossing_fraction: np.ndarray
    run_samples: np.ndarray
    line_index: np.ndarray
    noise_ire: np.ndarray
    polarities: Tuple[str, str] = ("fall", "rise")

    @property
    def count(self) -> int:
        return int(self.departure.shape[1])

    @property
    def lags(self) -> int:
        return int(self.departure.shape[2])

    @property
    def event_span(self) -> int:
        """The longest change run the CALIBRATION events themselves make.

        A transient is corrected with a kernel measured on the sync edge,
        so a change that takes longer to happen than a sync edge does is
        not the kind of event the kernel describes. The field's own sync
        edges therefore set the gate, and nothing is chosen: their change
        runs are measured and the gate is their upper edge.

        The upper edge is read as the 99th percentile rather than the
        maximum because a maximum is not a stable statistic - it grows
        with the sample - while a high quantile of a few hundred events is.
        Measured on six decodes, the sync edges' own change runs have a
        median of 9 to 10 samples, a 90th percentile of 13 to 14 and a
        99th of 16 to 19, against a settle guard of 7 - so a gate set at
        the settle guard excludes the calibration events themselves, and
        the correction then removes 0.4 to 5.9 per cent of the aftermath
        where the gate derived here removes 12.0 to 33.6.
        """
        runs = np.asarray(self.run_samples, dtype=np.float64).reshape(-1)
        runs = runs[np.isfinite(runs)]
        if not runs.size:
            return 0
        return int(math.ceil(float(np.percentile(runs, 99.0))))


def _erf(values: np.ndarray) -> np.ndarray:
    """The error function. scipy is a decoder dependency; the standard
    library's scalar version is the fallback so that this module's
    geometry can be derived without it."""
    try:
        from scipy.special import erf as scipy_erf
        return scipy_erf(values)
    except ImportError:                                   # pragma: no cover
        return np.vectorize(math.erf)(values)


def robust_sigma(values: np.ndarray) -> float:
    """The noise of a flat region, from its own first difference.

    A median absolute deviation of the DIFFERENCE is blind to the slow
    decaying tail that IS the artifact being measured, which a plain
    standard deviation would book as noise. The 0.6745 is the normal
    distribution's own quartile and the root two undoes the differencing.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 4:
        return float("nan")
    difference = np.diff(values)
    deviation = np.median(np.abs(difference - np.median(difference)))
    return float(deviation / 0.6745 / math.sqrt(2.0))


def _run_length(values: np.ndarray, position: int) -> int:
    """How long the same-signed change run containing `position` lasts."""
    change = np.sign(np.diff(values[max(position - 64, 0):position + 64]))
    local = position - max(position - 64, 0)
    direction = 0.0
    for offset in range(local, min(local + 8, change.size)):
        if change[offset] != 0.0:
            direction = change[offset]
            break
    if direction == 0.0:
        return 0
    start = local
    while start > 0 and change[start - 1] in (direction, 0.0):
        start -= 1
    stop = local
    while stop < change.size and change[stop] in (direction, 0.0):
        stop += 1
    return int(stop - start)


def measure_events(field_ire: np.ndarray, geometry: RingingGeometry,
                   align: bool = True) -> Optional[FieldEvents]:
    """Read every measurable line's two sync events, on one lag grid.

    THE LEVELS ARE MEDIANS OF THE SPECIFIED FLATS, per line: the front
    porch (which is the tail of the row before, because the time base
    corrector starts each row at the sync fall), the sync tip and the back
    porch, each guarded off its neighbouring transitions by the settle
    time the format's own luma bandwidth sets.

    THE IDEAL IS THE SPECIFICATION'S EDGE placed at the measured 50 per
    cent crossing: an erf of the stated `syncTransitionUS` between the two
    measured settled levels. A Heaviside ideal would charge the recorder's
    own edge shaping to the playback channel, and an ideal at a nominal
    position would charge the time base to it.

    THE SUB-SAMPLE ALIGNMENT IS LOAD-BEARING (standing rule 11). Each
    event's crossing lands at its own fraction of a sample, so a window
    taken at integer offsets samples every event's aftermath at a
    different phase. `hypercomplex.fractional_delay` puts every window on
    one grid by a single multiplication by a phase ramp, which is exact
    for a band-limited signal where a truncated sinc is not.

    Returns None when the field carries too few usable events - a dropout
    field, or a buffer too short to hold the window.
    """
    rows, columns = int(field_ire.shape[0]), int(field_ire.shape[1])
    flat = np.ascontiguousarray(field_ire, dtype=np.float64).reshape(-1)
    lags = geometry.aftermath_samples
    settle = geometry.transition_settle_samples
    edge = geometry.sync_transition_samples
    erf_scale = edge / ERF_TEN_NINETY_FACTOR
    guard = max(lags // 2, settle + 2)

    first = max(int(geometry.first_measurable_line), 1)
    last = min(int(geometry.last_measurable_line), rows)
    if last - first < 8:
        return None

    floor_ire = float(output_limit.output_profile(
        sample_rate_hz=geometry.sample_rate_mhz * 1e6
    )["quantisation_noise_ire"])

    porch_low = -int(geometry.front_porch_samples) + int(math.ceil(edge))
    porch_high = -int(math.ceil(edge))
    tip_low = int(math.ceil(edge)) + settle
    tip_high = int(geometry.sync_pulse_samples - edge)
    back_low = int(geometry.sync_pulse_samples + edge) + settle
    back_high = int(geometry.active_video_start_sample - edge)
    if tip_high - tip_low < 4 or back_high - back_low < 4:
        return None

    windows: List[List[np.ndarray]] = [[], []]
    fractions: List[List[float]] = [[], []]
    steps: List[List[float]] = [[], []]
    landings: List[List[float]] = [[], []]
    crossings: List[List[float]] = [[], []]
    pre_levels: List[List[float]] = [[], []]
    runs: List[List[int]] = [[], []]
    noise: List[float] = []
    used: List[int] = []

    for line in range(first, last):
        origin = line * columns
        if origin + porch_low < 0 or origin + back_high + lags + guard >= flat.size:
            continue
        porch = float(np.median(flat[origin + porch_low:origin + porch_high]))
        tip = float(np.median(flat[origin + tip_low:origin + tip_high]))
        back = float(np.median(flat[origin + back_low:origin + back_high]))
        sigma = robust_sigma(np.concatenate([
            flat[origin + tip_low:origin + tip_high],
            flat[origin + back_low:origin + back_high]]))
        if not np.isfinite(sigma):
            continue
        # A line can read ZERO noise - on synthetic material, and on a
        # heavily quantised capture whose flats sit on one code. Nothing
        # can be measured below the container's own quantisation, so that
        # is the floor: `output_limit.output_profile` at the decoder's own
        # sample rate, 0.0461 IRE rms for ten bits at four times the
        # subcarrier. Without it a clean field divides by zero in the
        # vertex variance and the whole field is refused.
        sigma = max(sigma, floor_ire)

        found = []
        for pre, post, nominal in ((porch, tip, 0.0),
                                   (tip, back, geometry.sync_pulse_samples)):
            step = post - pre
            # A real horizontal sync edge spans about one sync depth;
            # half of it is the loosest reading under which the pulse is
            # still a pulse rather than a dropout.
            if abs(step) < 0.5 * geometry.sync_depth_ire:
                break
            crossing_level = pre + 0.5 * step
            search_low = origin + int(nominal) - int(math.ceil(edge)) - settle
            search_high = origin + int(nominal) + int(math.ceil(edge)) + settle
            segment = flat[search_low:search_high]
            hits = np.nonzero((segment[:-1] - crossing_level)
                              * (segment[1:] - crossing_level) <= 0.0)[0]
            if not hits.size:
                break
            index = int(hits[0])
            low, high = float(segment[index]), float(segment[index + 1])
            within = 0.0 if high == low else (crossing_level - low) / (high - low)
            crossing = search_low + index + within
            base = int(math.floor(crossing))
            # THE SUB-SAMPLE OFFSET IS THE CROSSING'S OWN, not the
            # interpolation weight inside the pair the sign change was
            # found in. They agree only when that pair starts at
            # floor(crossing); where it does not - and it does not
            # whenever a sample sits exactly on the crossing level - the
            # weight reaches one and the alignment shifts the window by a
            # whole sample, putting the entire lag grid off by one. On a
            # noiseless field that alone left 0.148 per unit step of
            # departure where the true answer is zero.
            fraction = crossing - base
            if base - guard < 0 or base + 1 + lags + guard > flat.size:
                break
            found.append((fraction, step, post, crossing,
                          flat[base - guard:base + 1 + lags + guard],
                          _run_length(flat, base)))
        if len(found) != 2:
            continue

        for polarity, entry in enumerate(found):
            fraction, step, post, crossing, window, run = entry
            runs[polarity].append(run)
            windows[polarity].append(window)
            fractions[polarity].append(fraction)
            steps[polarity].append(step)
            landings[polarity].append(post)
            crossings[polarity].append(crossing)
            pre_levels[polarity].append(post - step)
        noise.append(sigma)
        used.append(line)

    if len(used) < 8:
        return None

    departure = np.zeros((2, len(used), lags))
    for polarity in range(2):
        stack = np.asarray(windows[polarity], dtype=np.float64)
        fraction = np.asarray(fractions[polarity], dtype=np.float64)
        width = stack.shape[1]
        pre = np.asarray(pre_levels[polarity])[:, None]
        step = np.asarray(steps[polarity])[:, None]
        # THE SPECIFICATION'S EDGE IS SUBTRACTED BEFORE THE WINDOW IS
        # SHIFTED, not after. A window taken across a sync edge runs from
        # one settled level to the other, and a phase ramp assumes the
        # record repeats - so shifting the raw window wraps that step
        # around and rings. Measured on a noiseless field whose true
        # departure is exactly zero, shifting the raw window left 0.0057
        # per unit step of alternating ripple on the rise, which is 0.23
        # IRE at a 40 IRE step and six times the field-mean noise floor.
        # The departure itself settles to zero at both ends, so shifting
        # THAT is well conditioned and the same field then reads zero.
        offsets = (np.arange(width, dtype=np.float64)[None, :]
                   - guard - fraction[:, None])
        ideal = pre + step * 0.5 * (
            1.0 + _erf(offsets / (math.sqrt(2.0) * erf_scale)))
        residual = (stack - ideal) / step
        if align:
            # One batched phase ramp: advance every window by its own
            # crossing fraction, so lag j sits exactly j + 1 samples after
            # that event's crossing on every line.
            spectrum = np.fft.rfft(residual, axis=1)
            frequency = np.fft.rfftfreq(width)
            residual = np.fft.irfft(
                spectrum * np.exp(2j * np.pi * np.outer(fraction, frequency)),
                n=width, axis=1)
        departure[polarity] = residual[:, guard + 1:guard + 1 + lags]

    return FieldEvents(
        departure=departure,
        step_ire=np.asarray(steps, dtype=np.float64),
        landing_ire=np.asarray(landings, dtype=np.float64),
        crossing_sample=np.asarray(crossings, dtype=np.float64),
        crossing_fraction=np.asarray(fractions, dtype=np.float64),
        run_samples=np.asarray(runs, dtype=np.float64),
        line_index=np.asarray(used, dtype=int),
        noise_ire=np.asarray(noise, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# The axes: polarity, the chroma nuisance, and the field's own slow scales
# ---------------------------------------------------------------------------

SLOW_AXIS_NAMES = ("field half", "field quarter", "field eighth",
                   "field sixteenth")


def line_axes(depth: int, population: int
              ) -> Tuple[Tuple[str, ...], Tuple[int, ...], Optional[int]]:
    """Which bits of an event's position become axes, at a given depth.

    THE ORDER IS PHYSICAL, NOT ARITHMETICAL. The first line axis is always
    the position's bit 0 - the alternation between adjacent lines - because
    that is where the colour burst goes and it must have somewhere to go
    (see the module docstring's control). Every deeper axis is the NEXT
    SLOWEST scale, taken from the top of the index down: the field's
    halves, then its quarters, then its eighths. So depth 1 is "the burst
    is separated and nothing else varies across the field", depth 2 adds
    "the channel differs between the top and the bottom of the field", and
    so on. `tesseract.from_field_series` builds the same hierarchy over
    field time; this builds it over line time so the whole cube fits
    inside ONE field.

    Returns the axis names, the position bits they read, and the bit that
    is LEFT OVER to split the lines into two arms for the held-out test -
    None when the population has no bit to spare.
    """
    total_bits = max(int(math.floor(math.log2(max(population, 1)))), 1)
    names: List[str] = []
    bits: List[int] = []
    if depth >= 1:
        names.append("line parity")
        bits.append(0)
    for step in range(depth - 1):
        bit = total_bits - 1 - step
        if bit <= 0:
            break
        names.append(SLOW_AXIS_NAMES[step] if step < len(SLOW_AXIS_NAMES)
                     else f"field 1/{2 ** (step + 1)}")
        bits.append(bit)
    spare = total_bits - 1 - (len(bits) - 1) if depth >= 1 else total_bits - 1
    holdout = spare if spare > 0 and spare not in bits else None
    return tuple(names), tuple(bits), holdout


def cube_from_events(events: FieldEvents, geometry: RingingGeometry,
                     depth: int, members: Optional[Sequence[int]] = None
                     ) -> Optional[Tuple[tesseract.Cube, np.ndarray, Tuple[int, ...]]]:
    """One field's events as a tesseract, complex on the lag axis.

    THE AXES. `polarity` separates the sync fall, which lands on the tip,
    from the sync rise, which lands on blanking: it is the landing axis and
    the polarity axis at once, because the sync interval offers only those
    two combinations. The line axes are `line_axes` above.

    THE VERTEX is the mean of the events whose position carries that bit
    pattern, so its noise falls as the count and the identification test
    means something. THE VALUE is the analytic form of that mean along the
    lag axis - real part the departure, imaginary part its Hilbert
    transform - which is the frequency axis's half of the hypercomplex
    expansion. THE VARIANCE is twice the pooled per-line noise over the
    count and over the square of the step, the two being both components
    together, which is the convention `tesseract.from_sync_exports` uses.

    Returns the cube, the count behind each vertex, and the position bits
    the axes read - or None when a vertex would be empty.
    """
    lags = events.lags
    positions = list(range(events.count) if members is None else members)
    names, bits, _ = line_axes(depth, events.count)
    axes = ("polarity",) + names
    shape = (2,) * len(axes)
    total = np.zeros(shape + (lags,), dtype=np.complex128)
    variance = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)

    analytic = events.departure + 1j * hypercomplex.hilbert(
        events.departure, axis=-1)
    for polarity in range(2):
        for position in positions:
            index = (polarity,) + tuple(int((position >> bit) & 1)
                                        for bit in bits)
            total[index] += analytic[polarity, position]
            variance[index] += 2.0 * (
                events.noise_ire[position]
                / abs(events.step_ire[polarity, position])) ** 2
            counts[index] += 1.0
    if counts.min() < 1.0:
        return None
    values = total / counts[..., None]
    vertex_variance = np.broadcast_to(
        (variance / counts ** 2)[..., None], values.shape).copy()
    return tesseract.Cube(axes, values, vertex_variance), counts, bits


def unfold(cube: tesseract.Cube,
           contrasts: Dict[tesseract.Subset, Dict[str, np.ndarray]],
           keep: Optional[Sequence[tesseract.Subset]] = None) -> np.ndarray:
    """THE FOLD RUN BACKWARDS, as a butterfly rather than as a sum.

    `tesseract.reconstruct` adds every contrast into every vertex, which
    costs 4^n vector operations; the fold is its own inverse up to the
    signs - a vertex is parent plus differential on one side of an axis
    and parent minus differential on the other - so the butterfly costs
    n 2^n and gives the identical answer. `test_ringing_tesseract` checks
    that identity against `tesseract.reconstruct` to machine precision.
    """
    order = cube.order
    lags = cube.values.shape[-1]
    stack = np.zeros((2,) * order + (lags,), dtype=np.complex128)
    for subset, entry in contrasts.items():
        if keep is not None and subset not in keep:
            continue
        index = tuple(1 if axis in subset else 0 for axis in cube.axes)
        stack[index] = np.asarray(entry["value"]).reshape(lags)
    for position in range(order):
        parent = np.take(stack, 0, axis=position)
        differential = np.take(stack, 1, axis=position)
        stack = np.stack([parent + differential, parent - differential],
                         axis=position)
    return stack


# ---------------------------------------------------------------------------
# The chroma gate: a contrast that lives in the burst is not the channel
# ---------------------------------------------------------------------------

def chroma_contaminated(cube: tesseract.Cube, geometry: RingingGeometry,
                        contrasts: Optional[Dict] = None) -> Dict[str, object]:
    """Which contrasts are the colour burst rather than the luma channel.

    The rise event's aftermath window contains the colour burst, which the
    format places by time and which alternates phase from one line to the
    next; the fold puts everything that alternates onto the line-parity
    contrasts. A contrast is refused when its energy is CONCENTRATED in
    the burst's own lags - more energy per bin inside them than outside -
    which is a measurement rather than a name, so an axis carrying channel
    structure keeps it and an axis carrying the burst does not.

    MEASURED (first field, depth 3, per unit signed step, this module's own
    instrument). The two contrasts the correction uses sit well below one;
    the two the line parity carries sit well above it, and nothing else
    does:

        contrast                 in burst  outside  ratio    verdict
        75 bars SP
          level (the mean)         0.0202   0.0691   0.29    channel
          polarity                 0.0066   0.0200   0.33    channel
          line parity              0.0108   0.0037   2.92    refused
          polarity:line parity     0.0094   0.0048   1.96    refused
          field half               0.0011   0.0011   0.98    channel
          polarity:field half      0.0018   0.0036   0.51    channel
        home
          level (the mean)         0.0269   0.0504   0.53    channel
          polarity                 0.0062   0.0381   0.16    channel
          line parity              0.0064   0.0018   3.63    refused
          polarity:line parity     0.0066   0.0011   5.86    refused
          field half               0.0021   0.0032   0.67    channel
          polarity:field half      0.0021   0.0022   0.96    channel

    and the y-only control in the module docstring shows those same
    contrasts collapsing by six to fifteen times when the capture carries
    no chroma at all.
    """
    if contrasts is None:
        contrasts = tesseract.walsh(cube)
    lags = cube.values.shape[-1]
    low, high = geometry.burst_lags
    inside = np.zeros(lags, dtype=bool)
    if high > low:
        inside[low:high] = True
    report: Dict[str, object] = {
        "burst_lags": (low, high),
        "gate_available": bool(inside.any()),
        "per_contrast": {},
        "refused": [],
    }
    if not inside.any():
        return report
    refused: List[tesseract.Subset] = []
    for subset, entry in contrasts.items():
        value = np.asarray(entry["value"]).reshape(lags)
        in_rms = float(np.sqrt(np.mean(np.abs(value[inside]) ** 2)))
        out_rms = float(np.sqrt(np.mean(np.abs(value[~inside]) ** 2)))
        ratio = in_rms / max(out_rms, 1e-30)
        contaminated = bool(subset and ratio > 1.0)
        report["per_contrast"][":".join(subset) or "level"] = {
            "in_burst_rms": in_rms, "outside_rms": out_rms,
            "concentration": ratio, "refused": contaminated}
        if contaminated:
            refused.append(subset)
    report["refused"] = refused
    return report


def component_enabled(shared_state: Optional[Dict[str, object]],
                      name: str, default: bool = True) -> bool:
    """Whether a declared component of this stage runs.

    `vhsdecode/pipeline/stages.toml` declares `chroma_gate` and `depth`,
    and `--stages -ringing.chroma_gate` turns one off. The decoder puts
    the parsed selection in the shared state; nothing else here reads it.
    """
    selection = (shared_state or {}).get("stage_selection")
    if not selection:
        return default
    return bool(selection.get("components", {}).get(("ringing", name),
                                                    default))


def _kept_contrasts(cube: tesseract.Cube, geometry: RingingGeometry,
                    sigma: float, chroma_gate: bool = True
                    ) -> Tuple[Dict, List[tesseract.Subset], Dict]:
    """The contrasts that stand above their noise AND are not the burst."""
    contrasts = tesseract.walsh(cube)
    verdicts = tesseract.identified(contrasts, sigma)
    gate = chroma_contaminated(cube, geometry, contrasts)
    refused = set(gate["refused"]) if chroma_gate else set()
    keep = [subset for subset, verdict in verdicts.items()
            if (not subset) or (verdict["identified"] and subset not in refused)]
    return contrasts, keep, {"verdicts": verdicts, "chroma": gate}


# ---------------------------------------------------------------------------
# The depth: chosen by holding out half the field's lines
# ---------------------------------------------------------------------------

def choose_depth(events: FieldEvents, geometry: RingingGeometry,
                 sigma: float = 3.0, max_depth: int = 5,
                 chroma_gate: bool = True) -> Dict[str, object]:
    """HOW DEEP THE CUBE GOES, decided by lines the model never saw.

    The tesseract's own test says which contrasts to keep, by whether they
    stand above their noise. It cannot say HOW MANY AXES to have, because
    every extra axis halves each contrast's noise while doubling their
    count, and this arc has measured what happens when that is not checked
    - a 252-parameter template lowered the in-sample residual and RAISED
    the held-out one on real data. So the depth is chosen the only way it
    can honestly be chosen, by prediction.

    At depth d the axes are `line_axes(d, ...)` and the next slowest bit
    splits the field's lines into two arms. The model is fitted on one arm
    and scored on the other, then the arms are swapped. Both the model and
    the target have the burst-carrying contrasts removed, so what is
    scored is the luma channel and not the chroma the fold has already
    separated out.

    MEASURED (2026-09-06, the first field of six decodes; the held-out
    residual as a fraction of the target's own power):

        decode              d=1     d=2     d=3     d=4     d=5
        75 bars SP        0.0148  0.0177  0.0352  0.0707  0.1455
        75 bars EP        0.0211  0.0336  0.0758  0.1792  0.3785
        chroma noise SP   0.0063  0.0246  0.0268  0.0705  0.1146
        home              0.0223  0.0289  0.0393  0.0674  0.1054
        ntc7 composite SP 0.0105  0.0257  0.0352  0.0573  0.1033
        ntc7 y-only SP    0.0103  0.0118  0.0381  0.0500  0.0878

    EVERY DECODE CHOOSES ONE, and every decode is monotonically worse at
    every axis after it. Depth one is the polarity and the chroma nuisance
    axis and nothing else, so this is a measurement in its own right:
    WITHIN ONE FIELD THE CHANNEL DOES NOT MEASURABLY DRIFT. A field-half
    axis, a field-quarter axis and everything below them buy nothing on
    lines they did not fit and cost between 15 per cent and eighteen times
    in held-out residual. The tesseract's own reading of the same fact is
    that the cube is exactly as large as one field's line count supports,
    and lowering the residual further needs more vertices - which, one
    field at a time, there are not.

    Returns the chosen depth, the whole scan, and the held-out optimal
    gain - the scalar that minimises the residual on lines the model did
    not fit, which `correction_strength` then halves. Measured, that gain
    is 0.989 to 0.997 on the six decodes, so the applied strength sits
    just under one half throughout.
    """
    scan: List[Dict[str, float]] = []
    for depth in range(1, max_depth + 1):
        names, bits, holdout = line_axes(depth, events.count)
        if holdout is None:
            break
        arms = ([position for position in range(events.count)
                 if not ((position >> holdout) & 1)],
                [position for position in range(events.count)
                 if ((position >> holdout) & 1)])
        residual = target = floor = 0.0
        numerator = denominator = 0.0
        kept = 0
        usable = True
        for fit_arm, score_arm in (arms, arms[::-1]):
            built_fit = cube_from_events(events, geometry, depth, fit_arm)
            built_score = cube_from_events(events, geometry, depth, score_arm)
            if built_fit is None or built_score is None:
                usable = False
                break
            fit_cube = built_fit[0]
            score_cube = built_score[0]
            contrasts, keep, _ = _kept_contrasts(fit_cube, geometry, sigma,
                                                 chroma_gate)
            model = unfold(fit_cube, contrasts, keep)
            kept = len(keep) - 1
            score_contrasts = tesseract.walsh(score_cube)
            gate = chroma_contaminated(score_cube, geometry, score_contrasts)
            clean = [subset for subset in score_contrasts
                     if subset not in set(gate["refused"])]
            truth = unfold(score_cube, score_contrasts, clean)
            residual += float(np.mean(np.abs(truth - model) ** 2))
            target += float(np.mean(np.abs(truth) ** 2))
            floor += float(np.mean(score_cube.variance))
            numerator += float(np.real(np.sum(np.conj(model) * truth)))
            denominator += float(np.sum(np.abs(model) ** 2))
        if not usable:
            break
        scan.append({
            "depth": depth,
            "axes": ("polarity",) + names,
            "vertices": 2 ** (1 + len(names)),
            "kept_contrasts": kept,
            "held_out_residual": residual / max(target, 1e-30),
            "held_out_over_floor": residual / max(floor, 1e-30),
            "held_out_gain": numerator / max(denominator, 1e-30),
        })
    if not scan:
        return {"depth": 1, "scan": [], "held_out_gain": 0.0,
                "held_out_residual": 1.0, "held_out_over_floor": float("inf"),
                "kept_contrasts": 0}
    best = min(scan, key=lambda row: row["held_out_residual"])
    return {"depth": int(best["depth"]),
            "axes": best["axes"],
            "held_out_residual": float(best["held_out_residual"]),
            "held_out_over_floor": float(best["held_out_over_floor"]),
            "held_out_gain": float(best["held_out_gain"]),
            "kept_contrasts": int(best["kept_contrasts"]),
            "scan": scan}


def correction_strength(held_out_gain: float) -> float:
    """HALF THE BELIEVED OPTIMUM, which is this arc's standing gain law.

    The optimal scalar on a correction whose model carries a relative
    error rho is 1 / (1 + rho), and applying twice that leaves the
    residual exactly where it started. What makes the halving a rule
    rather than timidity is that rho is systematically UNDER-estimated by
    any split of the same measurement: a split-half sees the noise but not
    the part of the model that is wrong in the same way on both halves.
    Half the believed optimum keeps three quarters of the achievable
    reduction - the gain curve is a parabola, so a half applies
    2(0.5) - (0.5)^2 = 0.75 of it - and cannot cross zero however wrong
    rho turns out to be.

    `held_out_gain` is measured on lines the model did not fit, so it
    already carries the noise part of rho; the halving covers the rest.
    """
    return float(np.clip(0.5 * float(held_out_gain), 0.0, 1.0))


# ---------------------------------------------------------------------------
# The collapse: one real signal per vertex
# ---------------------------------------------------------------------------

def collapse_to_one_signal(cube: tesseract.Cube, geometry: RingingGeometry,
                           sigma: float = 3.0, chroma_gate: bool = True
                           ) -> Dict[str, object]:
    """THE COLLAPSE Ethan describes, done additively so it can be subtracted.

    *"Eventually it collapses down to one real signal which we can subtract
    to remove the residual."*

    The identified contrasts, less the ones the burst owns, are folded back
    up the tree to one departure per vertex; the real part of that is the
    kernel, in IRE per unit signed step, and the imaginary part is its own
    Hilbert transform and is discarded on the way out exactly as
    `hypercomplex.collapse_back` discards it. What is left over - the
    contrasts the cube could not identify, and the ones it refused - is the
    residual, reported against the floor the field's own line count sets.

    The kernel's last tenth is tapered to zero by a raised cosine, because
    the window ends where the next specified transition begins and a step
    at that boundary would be a discontinuity nothing measured. It is the
    same tenth `tools/ringing_measure/sync_step_response.py` tapers.

    LEVEL NEUTRALITY IS BY CONSTRUCTION, not by a constraint. The ideal the
    departure is measured against uses the MEASURED settled levels on both
    sides of the edge, so the departure is already at zero where the signal
    has settled; the kernel therefore adds nothing to any flat region away
    from a transient, which is standing rule 6.
    """
    contrasts, keep, detail = _kept_contrasts(cube, geometry, sigma,
                                              chroma_gate)
    departure = unfold(cube, contrasts, keep)
    residual = cube.values - departure
    lags = cube.values.shape[-1]
    taper_length = max(2, lags // 10)
    taper = np.ones(lags)
    taper[-taper_length:] = 0.5 * (
        1.0 + np.cos(np.linspace(0.0, math.pi, taper_length)))
    kernels = np.real(departure) * taper
    floor = float(np.mean(cube.variance))
    residual_power = float(np.mean(np.abs(residual) ** 2))
    return {
        "kernels": kernels,
        "departure": departure,
        "residual": residual,
        "residual_power": residual_power,
        "floor_power": floor,
        "residual_over_floor": residual_power / max(floor, 1e-300),
        "identified": [":".join(subset) for subset in keep if subset],
        "refused": ([":".join(subset)
                     for subset in detail["chroma"]["refused"]]
                    if chroma_gate else []),
        "verdicts": detail["verdicts"],
        "chroma": detail["chroma"],
        "kernel_is_real": True,
    }


@dataclasses.dataclass
class KernelBank:
    """The collapsed kernels, and which one a given line's transient gets.

    `kernels` is indexed (polarity,) + the line axes + (lag,). `bits` says
    which bit of an event's position each line axis reads, and `position`
    maps a row of the field to that event's position. A row with no
    measured event - one the population excludes, or one whose pulse was a
    dropout - falls back to the mean over the line axes, which is the part
    of the model that does not depend on where in the field it landed.

    THE LIMIT OF WHAT SYNC CAN SAY, stated rather than papered over. The
    two poles of the polarity axis differ in the landing AND in the
    direction of the sweep at once, so this measurement cannot separate
    them: its polarity contrast is their sum. Extrapolating it in landing
    level - giving a picture transition that lands at white the fall kernel
    scaled by its distance from the tip - would assert a split the evidence
    does not contain. The kernel is therefore selected by the SIGN of the
    transient and applied at its own amplitude, and the landing dependence
    is left to the arc that can see more than two landings.
    """

    kernels: np.ndarray
    bits: Tuple[int, ...]
    position: Dict[int, int]

    @property
    def lags(self) -> int:
        return int(self.kernels.shape[-1])

    @property
    def mean(self) -> np.ndarray:
        axes = tuple(range(1, self.kernels.ndim - 1))
        return self.kernels.mean(axis=axes) if axes else self.kernels

    def for_row(self, row: int) -> np.ndarray:
        """The (2, lags) pair of kernels this row's transients get."""
        position = self.position.get(int(row))
        if position is None:
            return self.mean
        index = tuple(int((position >> bit) & 1) for bit in self.bits)
        return self.kernels[(slice(None),) + index]


def kernel_bank(collapse: Dict[str, object], events: FieldEvents,
                bits: Tuple[int, ...]) -> KernelBank:
    """Bind the collapsed kernels to the rows they were measured on."""
    return KernelBank(
        kernels=np.asarray(collapse["kernels"], dtype=np.float64),
        bits=tuple(bits),
        position={int(line): index
                  for index, line in enumerate(events.line_index)})


# ---------------------------------------------------------------------------
# The gated drive, and the subtraction
# ---------------------------------------------------------------------------

def gated_events(values: np.ndarray, noise_ire: float, sigma: float,
                 max_duration: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Where the transients are, and where each one's kernel goes.

    A transient is a maximal run of same-signed change. It enters the gate
    on two conditions, and the returned origin is NOT its start.

    THE TWO CONDITIONS. Its total change must clear the noise of a
    difference of two samples - root two times the per-line noise - by
    `sigma`, the SAME confidence the fold's identification test uses, so
    the module carries one significance and not two. And its change must
    complete within `max_duration` samples, which the caller takes from
    `FieldEvents.event_span`: the field's own sync edges say how long a
    transition of the kind the kernel was measured on takes, and a change
    slower than that is not one. The physical statement behind the second
    condition is the one `vhsdecode/luma_transient.py` opens with - the
    demodulator is a limiter, so a change on a carrier that is not
    sweeping is discarded before it reaches the picture, and everything
    the response does it does at transitions.

    THE ORIGIN IS THE EVENT'S OWN 50 PER CENT CROSSING, because that is
    where the kernel was measured from. Laying the kernel at the run's
    START instead puts it four to five samples early - half the edge's own
    duration - and the measured cost of that mistake is about half the
    correction: with the same model and the same gate, the sync aftermath
    fell by 6.2 to 16.9 per cent with the kernel at the run's start and by
    12.0 to 34.4 per cent with it at the crossing.

    Placing the origin at the NEAREST SAMPLE rather than shifting the
    kernel to the crossing's own sub-sample phase costs nothing
    measurable: correcting only the calibration events, a sixteenth-sample
    phase bank removed 10.6, 19.6, 20.6, 21.9, 20.9 and 23.7 per cent
    against 10.6, 19.4, 20.5, 23.8, 21.0 and 23.5 for the nearest sample -
    within a few tenths on five of six and better on the sixth.

    Returns each event's origin - one sample past its crossing, which is
    kernel lag zero - and its signed step in IRE.
    """
    values = np.asarray(values, dtype=np.float64)
    change = np.diff(values)
    sign = np.sign(change)
    if not np.flatnonzero(sign).size:
        return np.zeros(0, dtype=int), np.zeros(0)
    # A SAMPLE THAT DOES NOT CHANGE ENDS THE RUN. Letting it continue one
    # instead swallows every flat stretch into the neighbouring
    # transition: on a quantised or synthetic line the whole line becomes
    # one run, its duration exceeds any gate, and the transition inside it
    # is never corrected. On noisy material the two rules agree, because
    # an exactly-zero difference is then vanishingly rare.
    boundaries = np.flatnonzero(np.diff(sign) != 0.0) + 1
    starts = np.concatenate([[0], boundaries]).astype(int)
    stops = np.concatenate([boundaries, [sign.size]]).astype(int)
    moving = sign[starts] != 0.0
    starts, stops = starts[moving], stops[moving]
    if not starts.size:
        return np.zeros(0, dtype=int), np.zeros(0)
    step = values[stops] - values[starts]
    keep = np.abs(step) > float(sigma) * math.sqrt(2.0) * float(noise_ire)
    if max_duration is not None and int(max_duration) > 0:
        keep &= (stops - starts) <= int(max_duration)
    starts, stops, step = starts[keep], stops[keep], step[keep]
    if not starts.size:
        return np.zeros(0, dtype=int), np.zeros(0)
    # the 50 per cent crossing inside each run, to the nearest sample. A
    # vectorised segmented version of this was written and thrown away: it
    # was measured at the same speed (0.215 s a field against 0.207) and
    # disagreed with this one by a sample wherever a sample sits exactly
    # on the midpoint, so the loop is kept for being right and readable.
    origins = np.empty(starts.size, dtype=int)
    for index in range(starts.size):
        low, high = starts[index], stops[index]
        segment = values[low:high + 1]
        middle = values[low] + 0.5 * step[index]
        hits = np.nonzero((segment[:-1] - middle) * (segment[1:] - middle)
                          <= 0.0)[0]
        offset = int(hits[0]) if hits.size else (high - low) // 2
        origins[index] = low + offset + 1
    return origins, step


def apply_kernels(field_ire: np.ndarray, bank: KernelBank,
                  geometry: RingingGeometry, noise_ire: float,
                  strength: float, sigma: float = 3.0,
                  max_duration: Optional[int] = None) -> Dict[str, object]:
    """SUBTRACT the landing/polarity kernel at every gated transient.

    One line at a time, so a run never crosses a line boundary and the
    kernel of a line's last transient is not laid across the next line's
    front porch. Within the line the correction is a superposition:
    overlapping transients each contribute their own scaled kernel.

    WHY THIS IS NOT THE FILTER DESIGN LAW 1 FORBIDS. Scaling one kernel by
    every run's step and adding the results would, term by term, be the
    convolution of the signal's derivative with that kernel - a
    video-domain linear time-invariant filter. What breaks that identity
    is that the kernel is CHOSEN BY THE SIGN of each event: a falling
    transient gets the kernel measured at the sync fall, a rising one the
    kernel measured at the sync rise, and the two are far apart. Measured
    on 75 bars SP and home, the half-difference between them has an rms of
    0.0200 and 0.0381 per unit step outside the burst, against a mean of
    0.0691 and 0.0504 - so on one tape the two kernels differ by three
    quarters of what either of them is. No single filter applies two different responses
    according to the sign of the sweep, which is precisely why the arc's
    own proof says a filter cannot invert this and a per-landing
    subtraction can.

    THE SIGN OF THE SUBTRACTION IS THE STEP'S OWN. The kernel is the
    departure per unit SIGNED step, so an event of step s carries s times
    the kernel and s times the kernel is what is taken away. Using the
    magnitude instead inverts the correction at every falling edge -
    measured, that ADDED 3 to 8 per cent to the sync aftermath.

    Applied to sync, blanking AND picture alike within the rows the model
    owns. A correction confined to the blanking interval would hide its
    behaviour on picture from every sync gauge, which is a lesson this arc
    paid for once already.
    """
    rows, columns = field_ire.shape
    lags = bank.lags
    first = max(int(geometry.first_measurable_line), 0)
    last = min(int(geometry.last_measurable_line), rows)
    corrected = np.array(field_ire, dtype=np.float64, copy=True)
    applied = 0
    removed = 0.0
    for row in range(first, last):
        line = field_ire[row]
        origins, steps = gated_events(line, noise_ire, sigma, max_duration)
        if not origins.size:
            continue
        pair = bank.for_row(row)
        correction = np.zeros(columns + lags)
        # THE DRIVE IS AN IMPULSE TRAIN, ONE PER POLARITY, and the sum of
        # the scaled kernels is that train convolved with the kernel. It
        # is written that way because it is the same arithmetic done once
        # per line instead of once per event, and a field carries tens of
        # thousands of events. It stays a gated subtraction and not a
        # filter for the reason above: the train is the gated transients,
        # not the signal, and which of the two kernels an event enters is
        # decided by the sign of its own step.
        for polarity, chosen in ((0, steps < 0.0), (1, steps >= 0.0)):
            if not chosen.any():
                continue
            drive = np.zeros(columns + lags)
            np.add.at(drive, origins[chosen], strength * steps[chosen])
            correction += np.convolve(drive, pair[polarity])[:columns + lags]
        applied += int(origins.size)
        removed += float(np.sum(correction[:columns] ** 2))
        corrected[row] = line - correction[:columns]
    return {"field": corrected, "events": applied,
            "corrected_rows": max(last - first, 0),
            "removed_power": removed}


# ---------------------------------------------------------------------------
# The shape reading of the same kernel, in Hilbert space (offline)
# ---------------------------------------------------------------------------

def kernel_shape(kernel, geometry: RingingGeometry,
                 band_limit_hz: float = sync_shape.VHS_LUMA_BAND_HZ
                 ) -> Dict[str, object]:
    """THE AFTERMATH READ AS A SHAPE, on all three axes, with no order.

    Ethan, 2026-09-06: *"I think the matrix pencil is the wrong approach.
    Use the existing sync shape modeling in hilbert space not the matrix
    pencil."*

    The module replaced here described this same aftermath as a set of
    damped modes recovered by a matrix pencil, and corrected by inverting
    them. A pencil needs an ORDER, and the order was never fixed by the
    data: on planted data it validated and beat the alternative, but on
    real data there is no singular-value knee, so the answer follows the
    pencil's own parameter rather than the tape
    (`vhsdecode/luma_amplitude.py`, on the ripple, records exactly that).
    A parametric fit whose order the evidence does not determine is a knob
    wearing a measurement's clothes.

    What replaces it needs no order at all. `sync_shape.shape_components`
    fits the kernel in the Laplace eigenbasis truncated at the FORMAT's
    own luma band, which makes the split between shape and noise a
    constant rather than a threshold, and returns the frequency, amplitude
    and time axes as three readings of one fit. `hypercomplex.minimum_phase`
    then says how much of the kernel's phase its own magnitude already
    implies, and what is left is the excess - a delay, or an all-pass that
    a magnitude can never invert. Where the pencil asked "how many modes
    and where are their poles", this asks "what is the response, what part
    of it does its magnitude explain, and what remains", and the second
    question has an answer the data determines.

    Returns the shape components, the excess-phase reading, and the
    ordinary spectral summary of the kernel.
    """
    values = np.asarray(kernel, dtype=np.float64).reshape(-1)
    rate_hz = geometry.sample_rate_mhz * 1e6
    shape = sync_shape.shape_components(values, rate_hz, float(band_limit_hz))
    spectrum = np.fft.rfft(values)
    frequency_hz = np.fft.rfftfreq(values.size, d=1.0 / rate_hz)
    log_response = (np.log(np.maximum(np.abs(spectrum), 1e-30))
                    + 1j * np.unwrap(np.angle(spectrum)))
    excess = hypercomplex.excess_phase(log_response)
    band = frequency_hz <= float(band_limit_hz)
    energy = np.abs(spectrum) ** 2
    return {
        "shape": shape,
        "frequency_hz": frequency_hz,
        "spectrum": spectrum,
        "minimum_phase": excess["minimum"],
        "excess_phase": excess["excess"],
        "excess_phase_rms": excess["excess_rms"],
        "in_band_share": float(energy[band].sum() / max(energy.sum(), 1e-30)),
        "peak_hz": float(frequency_hz[int(np.argmax(energy[1:]) + 1)])
        if energy.size > 1 else 0.0,
        "why": ("the shape's own magnitude fixes its minimum-phase part "
                "through Bode's relation and what is left is excess - a "
                "reading with no order to choose, where a damped-mode fit "
                "has one the data does not determine"),
    }


# ---------------------------------------------------------------------------
# The window view of the same measurement, for the offline instruments
# ---------------------------------------------------------------------------

def accumulate_field_lines(field_ire: np.ndarray, geometry: RingingGeometry,
                           plan: MeasurementWindow,
                           state: Dict[str, object]) -> Dict[str, object]:
    """One field's sync intervals as a mean and a variance PER LAG.

    OFFLINE ONLY, AND THE ONE PLACE THAT POOLS. The runtime never calls
    this: `process_field` reads the same events through `measure_events`,
    one field at a time, as design law 3 requires. What this serves is the
    complex-response EXPORT (`tools/ringing_measure/sync_step_response.py`,
    whose npz `tesseract.from_sync_exports` and several model files
    consume), and that export's contract is a pooled measurement - its
    standard errors come from the cross-line scatter of an accumulated
    mean and its first-half/second-half split estimates come from pooling
    two halves of a decode. Removing the pooling HERE would not simplify
    the correction; it would break an interface other lanes read. So the
    pooling lives in the instrument, is called by nothing in the decode
    path, and says so.

    The same events `measure_events` reads, presented as one window with
    named regions rather than as two aftermaths - which is what the
    complex-response instruments want, because they need the flat levels
    on both sides of an edge in one array. Each line's window is placed on
    the common grid by its own sub-sample crossing, anchored to its own
    front porch level, and folded into a running mean and an unbiased
    cross-line variance. A second view is accumulated for the RISE,
    aligned on the sync rise instead, so a consumer measuring the rise's
    response is not reading it through the fall's alignment.

    Call it once per field with the same `state` to pool; the keys carry
    the names the export tools already read.
    """
    rows, columns = field_ire.shape
    flat = np.ascontiguousarray(field_ire, dtype=np.float64).reshape(-1)
    anchor = plan.sync_fall_index
    total = plan.total_samples
    first = max(int(geometry.first_measurable_line), 1)
    last = min(int(geometry.last_measurable_line), rows)

    starts = np.arange(first, last) * columns - anchor
    inside = (starts >= 0) & (starts + total <= flat.size)
    starts = starts[inside]
    if starts.size < 4:
        return {"usable_lines": 0, "healthy": False, "reset": False}

    windows = flat[starts[:, None] + np.arange(total)[None, :]]
    porch = np.median(windows[:, slice(*plan.front_porch)], axis=1)
    tip = np.median(windows[:, slice(*plan.sync_tip)], axis=1)
    back = np.median(windows[:, slice(*plan.back_porch)], axis=1)
    deep = (np.isfinite(porch) & np.isfinite(tip)
            & (porch - tip > 0.5 * geometry.sync_depth_ire))
    windows, porch, tip, back = (windows[deep], porch[deep], tip[deep],
                                 back[deep])
    if windows.shape[0] < 4:
        return {"usable_lines": 0, "healthy": False, "reset": False}

    # the front porch is the reference level (standing rule 19), so each
    # field is anchored to its own median porch before anything is pooled
    field_anchor = float(np.median(porch))
    windows = windows - field_anchor
    porch = porch - field_anchor
    tip = tip - field_anchor
    back = back - field_anchor

    results: Dict[str, object] = {"usable_lines": int(windows.shape[0])}
    for view, level_a, level_b, nominal in (
            ("slow", porch, tip, 0.0),
            ("rise", tip, back, geometry.sync_pulse_samples)):
        crossing = _window_crossings(windows, anchor + nominal,
                                     plan.crossing_search_radius,
                                     level_a, level_b)
        located = np.isfinite(crossing)
        if located.sum() < 4:
            continue
        aligned = _shift_rows(windows[located],
                              (anchor + nominal) - crossing[located])
        count = float(state.get(f"{view}_line_count", 0.0)) + aligned.shape[0]
        total_sum = np.asarray(state.get(f"{view}_sum", 0.0)) + aligned.sum(axis=0)
        total_square = (np.asarray(state.get(f"{view}_square", 0.0))
                        + (aligned ** 2).sum(axis=0))
        state[f"{view}_line_count"] = count
        state[f"{view}_sum"] = total_sum
        state[f"{view}_square"] = total_square
        mean = total_sum / count
        state[f"{view}_interval_mean"] = mean
        state[f"{view}_interval_variance"] = np.maximum(
            (total_square - count * mean ** 2) / max(count - 1.0, 1.0), 0.0)
        state[f"{view}_fields_accumulated"] = int(
            state.get(f"{view}_fields_accumulated", 0)) + 1
        results[f"{view}_lines"] = int(located.sum())

    state["slow_fields_accumulated"] = int(
        state.get("slow_fields_accumulated", 0))
    state["usable_lines_average"] = (
        float(state.get("slow_line_count", windows.shape[0]))
        / max(int(state.get("slow_fields_accumulated", 1)), 1))
    state["geometry"] = geometry
    state["window_plan"] = plan
    state["fall_edge_width_samples"] = _edge_width(
        state.get("slow_interval_mean"), plan.front_porch, plan.sync_tip)
    results.update({"healthy": bool(state.get("slow_fields_accumulated")),
                    "reset": False})
    state["last_field"] = results
    return results


def _edge_width(profile, before, after) -> float:
    """The accumulated fall's own 10-90 transition width, in samples.

    The specification states one and the tape does not meet it, so a
    consumer bounding a group delay by "an edge cannot be delayed more
    than its own width" needs the MEASURED width. NaN when there is no
    profile yet, and the caller falls back to the specified width.
    """
    if profile is None:
        return float("nan")
    values = np.asarray(profile, dtype=np.float64)
    high = float(np.median(values[slice(*before)]))
    low = float(np.median(values[slice(*after)]))
    span = low - high
    if not np.isfinite(span) or span == 0.0:
        return float("nan")
    marks = []
    for fraction in (0.1, 0.9):
        level = high + fraction * span
        # the search spans from the end of the first flat to the end of
        # the second: the 90 per cent crossing can land AFTER the second
        # region's nominal start, and stopping there returned nothing
        segment = values[before[1]:after[1]] - level
        if segment.size < 2:
            return float("nan")
        hits = np.nonzero(segment[:-1] * segment[1:] <= 0.0)[0]
        if not hits.size:
            return float("nan")
        index = int(hits[0])
        first, second = segment[index], segment[index + 1]
        step = 0.0 if second == first else -first / (second - first)
        marks.append(before[1] + index + step)
    return float(abs(marks[1] - marks[0]))


def _window_crossings(windows: np.ndarray, nominal: float, radius: int,
                      level_a: np.ndarray, level_b: np.ndarray) -> np.ndarray:
    """The 50 per cent crossing of each row, to sub-sample precision."""
    low = int(max(0, math.floor(nominal - radius)))
    high = int(min(windows.shape[1] - 1, math.ceil(nominal + radius)))
    target = 0.5 * (level_a + level_b)
    out = np.full(windows.shape[0], np.nan)
    segment = windows[:, low:high]
    for row in range(windows.shape[0]):
        values = segment[row] - target[row]
        hits = np.nonzero(values[:-1] * values[1:] <= 0.0)[0]
        if not hits.size:
            continue
        index = int(hits[0])
        first, second = values[index], values[index + 1]
        fraction = 0.0 if second == first else -first / (second - first)
        out[row] = low + index + fraction
    return out


def _shift_rows(rows: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Delay each row by its own fractional offset, by one phase ramp."""
    width = rows.shape[1]
    spectrum = np.fft.rfft(rows, axis=1)
    frequency = np.fft.rfftfreq(width)
    ramp = np.exp(-2j * np.pi * np.outer(np.asarray(offsets, np.float64),
                                         frequency))
    return np.fft.irfft(spectrum * ramp, n=width, axis=1)


# ---------------------------------------------------------------------------
# The luma transient improvement stage
# ---------------------------------------------------------------------------

def corrector_impulse(kernel: np.ndarray, strength: float) -> np.ndarray:
    """What the whole stage does to one unit step, as an impulse response.

    The subtraction takes `strength * step * kernel` away one sample past
    the transient's crossing, so what a unit step sees is one at the
    crossing and minus the scaled kernel after it. That composite - not
    the artifact kernel on its own - is what a downstream consumer must
    read, and getting the two confused is not academic: the artifact
    kernel has no direct tap at all, so a dispersion measured on it comes
    out near ONE, and the luma transient stage below then runs at almost
    full gain. Measured on four decodes, the artifact kernel reads a
    dispersion of 0.768 to 0.900 where the corrector's impulse reads 0.010
    to 0.017.
    """
    kernel = np.asarray(kernel, dtype=np.float64).reshape(-1)
    return np.concatenate([[1.0], -float(strength) * kernel])


def derive_lti_parameters(kernel: np.ndarray, noise_fraction: float
                          ) -> Dict[str, float]:
    """What the correction's own impulse response says the transient stage
    should do.

    The decoder runs a luma transient improvement stage after this one and
    asks this module what to run it at (`field.py`, under `--lti_gain`;
    with the flag omitted the gain comes from here). `kernel` is the
    CORRECTOR's impulse response (`corrector_impulse`), and every quantity
    below is read off it with no free coefficient:

      gain          the fraction of the impulse's energy that is NOT in
                    its direct tap, which is the share of a transient's
                    energy the chain still leaves off the edge after this
                    stage has done what it can
      blur_radius   the impulse's centre of mass in absolute amplitude, in
                    samples: how far that energy sits from the edge
      threshold     the measured noise as a fraction of the sync depth, so
                    the stage never sharpens noise

    This differs from the formula the old module used, which multiplied
    the dispersion by 1.5 and by a noise suppression factor built from
    three further constants; none of them had a derivation recorded, and
    standing rule 20 does not allow them to be carried forward without one.
    The quantity read is the same one it read - the dispersion of the
    corrector's impulse response from its direct tap onward.
    """
    kernel = np.asarray(kernel, dtype=np.float64).reshape(-1)
    energy = float(np.sum(kernel ** 2))
    if kernel.size == 0 or energy <= 0.0 or not np.isfinite(energy):
        return {"gain": 0.0, "threshold": 0.1, "blur_radius": 0.0,
                "dispersion": 0.0}
    dispersion = 1.0 - float(kernel[0] ** 2) / energy
    weight = np.abs(kernel)
    spread = float(np.sum(np.arange(kernel.size) * weight)
                   / max(np.sum(weight), 1e-30))
    return {
        "gain": float(np.clip(dispersion, 0.0, 1.0)),
        "threshold": float(np.clip(noise_fraction, 0.0, 1.0)),
        "blur_radius": spread,
        "dispersion": float(dispersion),
    }


try:                                                     # pragma: no cover
    import numba as _numba

    _jit = _numba.njit(cache=True, nogil=True, fastmath=True)
except ImportError:                                      # pragma: no cover
    def _jit(function):
        return function


@_jit
def apply_adaptive_luma_transient_improvement(video_buf, gain, threshold):
    """Non-linear luma transient improvement, in place.

    Carried across unchanged from the module this replaces, because it is
    the transient stage itself and not part of the ringing model: every
    sample whose neighbourhood changes by more than `threshold` is pushed
    further along its own gradient, in proportion to how far past the
    threshold that change is. `field.py` runs it under `--lti_gain`.
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


# ---------------------------------------------------------------------------
# The runtime entry point
# ---------------------------------------------------------------------------

def process_field(video_buffer, geometry, sync_tip_level, blanking_level,
                  shared_state, average_fields, head_parity=None,
                  debug=False, apply_correction=True):
    """Measure this field's sync pulses, fold them, subtract the collapse.

    The signature matches the module this replaces, so the runtime's call
    site changes by its import alone. The buffer arrives in demodulated
    signal units and is converted to IRE by the two supplied reference
    levels - blanking at zero and the sync tip at minus the specified
    depth, which is the luma's only absolute scale. Returns the buffer,
    corrected in place within the rows the model owns, and the parameters
    for the luma transient improvement stage.

    ARGUMENTS ACCEPTED AND NOT USED, and why:

      `average_fields`  ACCEPTED AND IGNORED. It set the old module's
                        cross-field accumulation horizon. Ethan's
                        directive is that the operation is done on one
                        field at a time and any averaging is removed, and
                        this module has no cross-field state to average
                        into. Nothing in the returned correction depends
                        on it.

      `head_parity`     RECORDED, NOT USED TO ROUTE. The old module kept a
                        separate accumulated model per video head because
                        it pooled fields and the two heads are different
                        channels. A field belongs to exactly one head by
                        construction, so a per-field model IS a per-head
                        model; the parity is stored so a report can say
                        which head a field's numbers belong to.

      `shared_state`    WRITTEN, NOT READ. The field's model, its scan and
                        its diagnostics are placed there for the debug
                        figure and for offline tools. Nothing is carried
                        in from a previous field, which is what makes the
                        stage reproducible field by field.

    `apply_correction` false gives measure-only mode: the model is
    measured, folded and stored, and the buffer is returned untouched.
    That is what `--debug_plot hsync_model` uses without `--inverse_eq`.
    """
    state = shared_state.setdefault(STATE_KEY, {})
    state.clear()
    if head_parity is not None:
        state["head_name"] = "head_a" if head_parity else "head_b"
        state["head_parity"] = bool(head_parity)
    state["geometry"] = geometry
    state["average_fields_ignored"] = average_fields

    neutral = {"gain": 0.0, "threshold": 0.1, "blur_radius": 0.0}

    units_per_ire = (blanking_level - sync_tip_level) / geometry.sync_depth_ire
    if not np.isfinite(units_per_ire) or units_per_ire == 0.0:
        state["status"] = "no level scale"
        return video_buffer, neutral

    samples_per_line = geometry.samples_per_line
    whole_lines = len(video_buffer) // samples_per_line
    if whole_lines < 8:
        state["status"] = "buffer shorter than the measurable population"
        return video_buffer, neutral

    field_ire = (
        np.asarray(video_buffer[:whole_lines * samples_per_line],
                   dtype=np.float64).reshape(whole_lines, samples_per_line)
        - blanking_level) / units_per_ire

    events = measure_events(field_ire, geometry)
    if events is None:
        state["status"] = "no usable sync events in this field"
        return video_buffer, neutral

    chroma_gate = component_enabled(shared_state, "chroma_gate")
    if component_enabled(shared_state, "depth"):
        depth = choose_depth(events, geometry, chroma_gate=chroma_gate)
    else:
        depth = {"depth": 1, "scan": [], "held_out_gain": 1.0,
                 "held_out_residual": float("nan"),
                 "held_out_over_floor": float("nan"), "kept_contrasts": 0,
                 "axes": ("polarity",) + line_axes(1, events.count)[0]}
    built = cube_from_events(events, geometry, depth["depth"])
    if built is None:
        state["status"] = "cube could not be filled"
        return video_buffer, neutral
    cube, counts, bits = built
    collapse = collapse_to_one_signal(cube, geometry, chroma_gate=chroma_gate)
    bank = kernel_bank(collapse, events, bits)
    strength = correction_strength(depth["held_out_gain"])
    noise_ire = float(np.median(events.noise_ire))

    state.update({
        "events": events, "cube": cube, "counts": counts, "depth": depth,
        "collapse": collapse, "bank": bank, "strength": strength,
        "noise_ire": noise_ire, "status": "measured",
    })

    lti = derive_lti_parameters(
        corrector_impulse(bank.mean.mean(axis=0), strength),
        noise_ire / geometry.sync_depth_ire)

    if apply_correction and strength > 0.0:
        applied = apply_kernels(field_ire, bank, geometry, noise_ire, strength,
                                max_duration=events.event_span)
        corrected = applied["field"] * units_per_ire + blanking_level
        if np.all(np.isfinite(corrected)):
            first = max(int(geometry.first_measurable_line), 0)
            last = min(int(geometry.last_measurable_line), whole_lines)
            if last > first:
                rows = np.asarray(
                    video_buffer[:whole_lines * samples_per_line]).reshape(
                    whole_lines, samples_per_line)
                rows[first:last] = corrected[first:last].astype(
                    np.asarray(video_buffer).dtype)
        state["applied"] = {key: value for key, value in applied.items()
                            if key != "field"}
        state["status"] = "applied"

    if debug:
        _show_debug_figure(state)

    return video_buffer, lti


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def describe_field(state: Dict[str, object]) -> str:
    """The field's model as one readable block."""
    if "depth" not in state:
        return f"ringing_tesseract: {state.get('status', 'not run')}"
    depth = state["depth"]
    collapse = state["collapse"]
    mean = state["bank"].mean
    lines = [
        f"head                   {state.get('head_name', 'unknown')}",
        f"axes                   {', '.join(depth['axes'])}",
        f"held-out residual      {depth['held_out_residual']:.4f}"
        f" of the aftermath, {depth['held_out_over_floor']:.2f} x the floor",
        f"held-out gain          {depth['held_out_gain']:.3f}"
        f"   applied at {state['strength']:.3f}",
        f"contrasts kept         {len(collapse['identified'])}"
        f"   refused as chroma {len(collapse['refused'])}",
        f"in-cube residual       {collapse['residual_over_floor']:.3f}"
        f" x the floor",
        f"per-line noise         {state['noise_ire']:.4f} IRE",
        f"kernel rms             fall {np.sqrt(np.mean(mean[0] ** 2)):.4f}"
        f"   rise {np.sqrt(np.mean(mean[1] ** 2)):.4f}   (per unit step)",
    ]
    if "applied" in state:
        lines.append(f"applied                {state['applied']['events']}"
                     f" events over {state['applied']['corrected_rows']} rows")
    return "\n".join(lines)


def render_debug_figure(state: Dict[str, object], title: str = ""):
    """The per-field figure `--debug_plot hsync_model` shows.

    Four panels: the collapsed kernels, the contrast ladder with its
    chroma verdicts, the held-out depth scan, and the field's own summary.
    matplotlib is imported lazily so it is never a decode dependency.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    geometry = state["geometry"]
    mean = state["bank"].mean
    time_us = np.arange(mean.shape[-1]) / geometry.sample_rate_mhz
    depth = state["depth"]
    collapse = state["collapse"]

    panel = axes[0][0]
    panel.plot(time_us, mean[0] * geometry.sync_depth_ire,
               label="fall, landing on the tip")
    panel.plot(time_us, mean[1] * geometry.sync_depth_ire,
               label="rise, landing on blanking")
    low, high = geometry.burst_lags
    if high > low:
        panel.axvspan(low / geometry.sample_rate_mhz,
                      high / geometry.sample_rate_mhz, alpha=0.12,
                      color="tab:orange", label="colour burst")
    panel.set_xlabel("microseconds after the crossing")
    panel.set_ylabel(f"IRE per {geometry.sync_depth_ire:.0f} IRE step")
    panel.set_title("the collapsed kernel")
    panel.legend(fontsize=7)
    panel.grid(alpha=0.3)

    panel = axes[0][1]
    per_contrast = collapse["chroma"]["per_contrast"]
    names = [name for name in per_contrast if name != "level"]
    concentration = [per_contrast[name]["concentration"] for name in names]
    colours = ["tab:red" if per_contrast[name]["refused"] else "tab:blue"
               for name in names]
    panel.barh(range(len(names)), concentration, color=colours)
    panel.axvline(1.0, color="black", lw=0.8)
    panel.set_yticks(range(len(names)))
    panel.set_yticklabels(names, fontsize=6)
    panel.set_xlabel("burst lags over the rest (red: refused)")
    panel.set_title("what the fold separated")
    panel.grid(alpha=0.3)

    panel = axes[1][0]
    scan = depth["scan"]
    if scan:
        panel.plot([row["depth"] for row in scan],
                   [row["held_out_residual"] for row in scan], "o-")
    else:
        panel.text(0.5, 0.5, "depth pinned\n(--stages -ringing.depth)",
                   ha="center", va="center", fontsize=8)
    panel.axvline(depth["depth"], color="tab:green", ls="--", label="chosen")
    panel.set_xlabel("depth (line axes + 1)")
    panel.set_ylabel("held-out residual, fraction")
    panel.set_title("depth chosen on lines never fitted")
    panel.legend(fontsize=7)
    panel.grid(alpha=0.3)

    panel = axes[1][1]
    panel.axis("off")
    panel.text(0.0, 1.0, describe_field(state), fontsize=8,
               family="monospace", va="top")

    figure.suptitle(title or "ringing on the tesseract graph")
    figure.tight_layout()
    return figure


def _show_debug_figure(state: Dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    figure = render_debug_figure(state)
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    # The geometry both standards derive, printed from the decoder's own
    # format tables - the same self-check the module this replaces ran.
    import logging

    from vhsdecode.formats import get_format_params, parse_tape_speed

    for name, line_offset, line_count in (("NTSC", 0, 263), ("PAL", 2, 312)):
        sys_params, decoder_params = get_format_params(
            name, "VHS", parse_tape_speed("sp"), logging.getLogger(__name__))
        geometry = build_geometry(
            sys_params, decoder_params, sys_params["outlinelen"],
            line_offset, line_count)
        print(f"\n=== {name}")
        print(describe_geometry(geometry))
        print(describe_window(plan_measurement_window(geometry)))
