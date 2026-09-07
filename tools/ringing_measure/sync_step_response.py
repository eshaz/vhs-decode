"""Sync-step complex response export (the LF fold-in contract).

Measures, per decode, the channel's complex frequency response as
witnessed by the sync pulse - the known deterministic step every line -
and writes it in the shape pinned with the head_switch lane
(vhs-decode-b5, 2026-09-02), for folding into the luma_eq response
model and the head_switch correction's low-frequency band.

THE CONTRACT (stated here and in the npz metadata):

  MEASUREMENT POINT: the decoded luma output - post-demodulation,
  post-de-emphasis, post-TBC, output rate (4 x fsc), units IRE
  referenced to the tbc.json black/white points.  Decodes are assumed
  y_comb-off (all standing recipes).  Re-referencing to any other
  point in the chain is the CONSUMER's division; to make it exact the
  npz ships both time-domain references and their spectra.

  IDEAL STEP: amplitude = the measured signed step between settled
  medians of the flanking flat regions; shape = an erf at the SPEC
  sync transition time (SysParams syncTransitionUS); timing = centered
  at the fitted 50% crossing (subsample), which is the PHASE ZERO of
  the ratio.  A Heaviside ideal would attribute record-side edge
  shaping to the playback channel.

  H(f) per head and POLARITY separately (H = rfft(w*d measured) /
  rfft(w*d ideal), derivative route, Tukey-tapered window).  The
  chain is nonlinear (LTI-in-RF + nonlinear demodulation): each H is
  the effective linearization at that edge's operating point.  Only
  the EVEN part (H_fall + H_rise)/2 may enter an LTI consumer; the
  ODD part is a validity bound, never corrected.

  UNCERTAINTY: per-band SE propagated from the cross-line variance of
  the accumulated mean (Gaussian, per component, declared
  approximate), plus first-half/second-half split estimates for an
  empirical check.  The stated resolution_mhz per polarity is the
  reciprocal window length - bins are finer than the information.

  VALIDITY: ~0.1-3 MHz determined; 0.05-0.1 MHz one marginal band;
  below 0.05 MHz nothing exists to measure (window length, per-field
  porch anchoring).  For the ring band a per-polarity SHAPE reading
  rides along (`ringing_tesseract.kernel_shape`, the Laplace eigenbasis
  truncated at the format's luma band, plus Bode's excess phase); it
  replaced the matrix-pencil pole table on 2026-09-06, because a
  damped-mode fit needs an order the evidence does not fix.  Below
  ~0.3 MHz use the nonparametric H (witnessability bound).

Usage: RC_WORK=<dir> python3 sync_step_response.py <prefix>...
Writes <RC_WORK>/<prefix>_sync_step_response.npz and a figure under
<RC_WORK>/hsync_plots/.
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import hsync_model_report as report
from vhsdecode.models import ringing_tesseract as model
from vhsdecode.formats import get_format_params, parse_tape_speed

import logging

WORK_DIRECTORY = os.environ.get("RC_WORK", "/output/claude_validation")
PLOT_DIRECTORY = os.path.join(WORK_DIRECTORY, "hsync_plots")

# The luma band that separates the aftermath's SHAPE from its noise, a
# format constant rather than a fitted knee (`sync_shape`).
SHAPE_BAND_HZ = 3.0e6
FFT_LENGTH = 4096
BAND_MHZ = (0.04, 3.5)
TAPER_ALPHA = 0.15

# the VHS NTSC luma FM deviation map the module itself uses
# (the VHS NTSC deviation map): sync tip 3.40 MHz,
# blanking 3.69, white 4.40.  The fall lands at the tip, the rise at
# blanking - two DIFFERENT carriers, so an LTI channel already rings
# the two aftermaths at |f_pole - f_c(landing)| with no nonlinearity.
# 46's single-H fit uses these anchors to separate the LTI landing
# term from the true emphasis nonlinearity.
CARRIER_TIP_HZ = 3.40e6
CARRIER_PORCH_HZ = 3.69e6
# the erf form 0.5*(1+erf(t/s)) has 10-90 width 1.8124*s
ERF_WIDTH_FACTOR = 2.0 * 0.9061938

sys_params, decoder_params = get_format_params(
    "NTSC", "VHS", parse_tape_speed("sp"), logging.getLogger(__name__))
geometry = model.build_geometry(sys_params, decoder_params,
                                sys_params["outlinelen"], 0, 263)
plan = model.plan_measurement_window(geometry)
rate = geometry.sample_rate_mhz
settle = geometry.transition_settle_samples
frequency_mhz = np.fft.rfftfreq(FFT_LENGTH, d=1.0 / rate)


def accumulate(fields, parity, subset=None):
    """One parity's accumulated sync interval over its fields.

    THE POOLING IS THE EXPORT'S OWN, not the decoder's. The correction
    stage is one field at a time and averages nothing across fields; this
    instrument pools because its published contract is a standard error
    from the cross-line scatter of an accumulated mean and a
    first-half/second-half pair of independent estimates. That is why the
    pooling lives in `ringing_tesseract.accumulate_field_lines`, which
    nothing in the decode path calls.
    """
    state = {}
    indices = range(parity, len(fields), 2)
    if subset == "first":
        indices = list(indices)[: len(list(range(parity, len(fields),
                                                 2))) // 2]
    elif subset == "second":
        full = list(range(parity, len(fields), 2))
        indices = full[len(full) // 2:]
    for index in indices:
        model.accumulate_field_lines(fields[index], geometry, plan, state)
    return state


def effective_count(state, view):
    """How many independent line measurements a view's mean rests on."""
    key = "rise_line_count" if view == "rise" else "slow_line_count"
    return max(float(state.get(key, state.get("slow_line_count", 1.0))), 1.0)


def step_response(state, view):
    """(H, SE, window info) for one polarity of one state, or None."""
    if view == "fall":
        mean = state.get("slow_interval_mean")
        variance = state.get("slow_interval_variance")
        pre_region, post_region = plan.front_porch, plan.sync_tip
    else:
        mean = state.get("rise_interval_mean")
        variance = state.get("rise_interval_variance")
        pre_region, post_region = plan.sync_tip, plan.back_porch
    if mean is None or variance is None:
        return None
    lo = pre_region[0] + 1
    hi = post_region[1] - 1
    pre_level = float(np.median(mean[slice(*pre_region)]))
    # settled read: the far half of the post region, past the settle
    post_interior = mean[post_region[0] + settle:post_region[1] - 1]
    if len(post_interior) < 4:
        return None
    post_level = float(np.median(post_interior))
    step = post_level - pre_level
    if abs(step) < 1.0:
        return None
    # 50% crossing with subsample interpolation
    crossing = pre_level + 0.5 * step
    segment = mean[pre_region[1] - 2:post_region[0] + settle]
    offsets = (np.nonzero((segment[:-1] - crossing)
                          * (segment[1:] - crossing) <= 0)[0])
    if not len(offsets):
        return None
    k = int(offsets[0])
    y0, y1 = segment[k], segment[k + 1]
    fraction = 0.0 if y1 == y0 else float((crossing - y0) / (y1 - y0))
    t0 = (pre_region[1] - 2) + k + fraction        # samples, absolute
    # ideal erf at the SPEC transition width, centered at t0
    s_samples = geometry.sync_transition_samples / ERF_WIDTH_FACTOR
    lags = np.arange(lo, hi, dtype=np.float64)
    ideal = pre_level + step * 0.5 * (
        1.0 + erf((lags - t0) / (math.sqrt(2.0) * s_samples)))
    measured = np.asarray(mean[lo:hi], dtype=np.float64)
    # derivative route with a Tukey taper
    d_measured = np.diff(measured)
    d_ideal = np.diff(ideal)
    window = tukey(len(d_measured), TAPER_ALPHA)
    spectrum_measured = np.fft.rfft(d_measured * window, n=FFT_LENGTH)
    spectrum_ideal = np.fft.rfft(d_ideal * window, n=FFT_LENGTH)
    floor = 0.02 * np.abs(spectrum_ideal).max()
    valid = ((np.abs(spectrum_ideal) > floor)
             & (frequency_mhz >= BAND_MHZ[0])
             & (frequency_mhz <= BAND_MHZ[1]))
    response = np.full(len(frequency_mhz), np.nan + 0j,
                       dtype=np.complex128)
    response[valid] = (spectrum_measured[valid]
                       / spectrum_ideal[valid])
    # per-band SE: variance of the mean per lag -> derivative ->
    # windowed rfft (Gaussian, per component, approximate)
    mean_variance = (np.asarray(variance[lo:hi], np.float64)
                     / effective_count(state, view))
    derivative_variance = mean_variance[:-1] + mean_variance[1:]
    noise_power = 0.5 * float(np.sum(window ** 2
                                     * derivative_variance))
    se = np.full(len(frequency_mhz), np.nan)
    se[valid] = math.sqrt(noise_power) / np.abs(spectrum_ideal[valid])
    # WIENER-DERIVED KERNEL FROM THE SYNC EDGE (Ethan's "missing
    # part"): the edge's DERIVATIVE, deconvolved against the ideal's
    # derivative with the measured noise as the regularizer, brought
    # back to the TIME dimension = the channel impulse response per
    # unit signed step at this view's landing carrier - one measured
    # row of the (frequency x decay) table.  noise_power is the
    # per-bin noise variance of the windowed derivative (white-noise
    # expectation), so the denominator is |I|^2 + sigma^2 per bin.
    # Realization is SUBTRACTION on a gated drive per landing and
    # polarity - NEVER a video-domain filter (design law; the Wiener
    # FIR null).  The linear-even part alone is foldable.
    wiener_response = (np.conj(spectrum_ideal) * spectrum_measured
                       / (np.abs(spectrum_ideal) ** 2 + noise_power))
    kernel = np.fft.irfft(wiener_response, n=FFT_LENGTH)
    # witnessability bound = the tip's flat span (the module's own
    # length rule): a response outlasting it is extrapolation
    tip_flat = int(geometry.sync_pulse_samples
                   - geometry.sync_transition_samples)
    span = int(max(8, min(tip_flat, len(kernel))))
    kernel = kernel[:span].copy()
    tail = max(2, span // 10)
    kernel[-tail:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi,
                                                      tail)))
    return {
        "H": response, "se": se, "valid": valid,
        "mean": measured, "ideal": ideal,
        "window_lags": np.array([lo, hi]),
        "t0_sample": t0, "step_ire": step,
        "resolution_mhz": rate / max(hi - lo, 1),
        "wiener_H": wiener_response,
        "wiener_kernel": kernel,
        "wiener_kernel_us": np.arange(span) / rate,
    }


def minimum_phase_from_magnitude(magnitude):
    """The phase a causal MINIMUM-PHASE system with this magnitude must
    have (Bode gain-phase, real-cepstrum reconstruction), on the rfft
    grid.  Missing or non-positive bins are filled by interpolation and
    held flat beyond the last measured bin - a truncated magnitude
    reconstructs spurious phase, so feed this the WIDEST honest
    magnitude available."""
    values = np.asarray(magnitude, np.float64).copy()
    ok = np.isfinite(values) & (values > 0)
    if ok.sum() < 3:
        return np.full(len(values), np.nan)
    positions = np.arange(len(values))
    values[~ok] = np.interp(positions[~ok], positions[ok], values[ok])
    log_full = np.log(np.maximum(values, 1e-6))
    two_sided = np.concatenate([log_full, log_full[-2:0:-1]])
    cepstrum = np.fft.ifft(two_sided).real
    count = len(cepstrum)
    folded = np.zeros(count)
    folded[0] = cepstrum[0]
    folded[1:count // 2] = 2.0 * cepstrum[1:count // 2]
    folded[count // 2] = cepstrum[count // 2]
    return np.fft.fft(folded).imag[:len(values)]


def pulse_response(state):
    """Differentiate ACROSS THE WHOLE SYNC PULSE (Ethan): the fall and
    the rise together, so the derivative is a two-impulse input whose
    known separation pins the phase at every frequency - a single
    edge's crossing cannot.  H_pulse = rfft(w*d measured) /
    rfft(w*d ideal), ideal = two spec-erf edges at the two MEASURED
    crossings on the measured plateau levels.  Its unwrapped phase and
    group delay are the roll-off's exact phase response.

    Referencing both edges to their MEASURED crossings keeps the
    channel's edge-SHAPE phase and treats the recorded pulse width
    (off spec on this deck) as timing, not channel - reported
    separately as pulse_width_us.  Uses the fall-aligned accumulated
    mean, in which the rise is slightly smeared by line-to-line width
    jitter: negligible for the LF/mid roll-off phase this targets,
    stated here.
    """
    mean = state.get("slow_interval_mean")
    variance = state.get("slow_interval_variance")
    if mean is None or variance is None:
        return None
    lo = plan.front_porch[0] + 1
    hi = plan.back_porch[1] - 1
    fp_level = float(np.median(mean[slice(*plan.front_porch)]))
    tip_level = float(np.median(
        mean[plan.sync_tip[0] + settle:plan.sync_tip[1] - 1]))
    bp_level = float(np.median(
        mean[plan.back_porch[0] + settle:plan.back_porch[1] - 1]))

    def crossing(seg_lo, seg_hi, level_a, level_b, rising):
        segment = mean[seg_lo:seg_hi]
        target = level_a + 0.5 * (level_b - level_a)
        hit = np.nonzero((segment > target) if rising
                         else (segment < target))[0]
        if not len(hit):
            return None
        k = int(hit[0])
        if k == 0:
            return float(seg_lo)
        y0, y1 = segment[k - 1], segment[k]
        frac = 0.0 if y1 == y0 else float((target - y0) / (y1 - y0))
        return seg_lo + (k - 1) + frac

    t0_fall = crossing(plan.front_porch[1] - 2,
                       plan.sync_tip[0] + settle, fp_level, tip_level,
                       rising=False)
    t0_rise = crossing(plan.sync_tip[1] - 2,
                       plan.back_porch[0] + settle, tip_level, bp_level,
                       rising=True)
    if t0_fall is None or t0_rise is None:
        return None
    s_samples = geometry.sync_transition_samples / ERF_WIDTH_FACTOR
    lags = np.arange(lo, hi, dtype=np.float64)
    ideal = (fp_level
             + (tip_level - fp_level) * 0.5 * (1.0 + erf(
                 (lags - t0_fall) / (math.sqrt(2.0) * s_samples)))
             + (bp_level - tip_level) * 0.5 * (1.0 + erf(
                 (lags - t0_rise) / (math.sqrt(2.0) * s_samples))))
    measured = np.asarray(mean[lo:hi], dtype=np.float64)
    d_measured = np.diff(measured)
    d_ideal = np.diff(ideal)
    window = tukey(len(d_measured), TAPER_ALPHA)
    spectrum_measured = np.fft.rfft(d_measured * window, n=FFT_LENGTH)
    spectrum_ideal = np.fft.rfft(d_ideal * window, n=FFT_LENGTH)
    # the two-impulse ideal has comb nulls at f = n / width; mask
    # them (no information there) along with the band limits
    floor = 0.02 * np.abs(spectrum_ideal).max()
    valid = ((np.abs(spectrum_ideal) > floor)
             & (frequency_mhz >= BAND_MHZ[0])
             & (frequency_mhz <= BAND_MHZ[1]))
    response = np.full(len(frequency_mhz), np.nan + 0j,
                       dtype=np.complex128)
    response[valid] = spectrum_measured[valid] / spectrum_ideal[valid]
    mean_variance = (np.asarray(variance[lo:hi], np.float64)
                     / effective_count(state, "fall"))
    derivative_variance = mean_variance[:-1] + mean_variance[1:]
    noise_power = 0.5 * float(np.sum(window ** 2
                                     * derivative_variance))
    se = np.full(len(frequency_mhz), np.nan)
    se[valid] = math.sqrt(noise_power) / np.abs(spectrum_ideal[valid])
    # exact phase: the two-impulse ideal has NARROW comb nulls at
    # n/width (masked as invalid).  Unwrapping across those gaps adds
    # spurious 2*pi jumps and the gradient spikes at the gap edges
    # (measured: phase ran to -2150 deg at 2 MHz, group-delay std of
    # microseconds).  So interpolate H across the nulls within the
    # band and unwrap over a CONTIGUOUS band; the nulls are a few
    # bins wide against a phase that is smooth there.
    phase = np.full(len(frequency_mhz), np.nan)
    group_delay = np.full(len(frequency_mhz), np.nan)
    resolution_bins = max(3, int(round(
        (rate / max(hi - lo, 1)) / (frequency_mhz[1]
                                    - frequency_mhz[0]))) | 1)
    band = ((frequency_mhz >= BAND_MHZ[0])
            & (frequency_mhz <= BAND_MHZ[1]))
    idx = np.nonzero(band)[0]
    vi = np.nonzero(valid & band)[0]
    ceiling_mhz = float(BAND_MHZ[1])
    if len(idx) > 2 and len(vi) > 2:
        real = np.interp(frequency_mhz[idx], frequency_mhz[vi],
                         response[vi].real)
        imag = np.interp(frequency_mhz[idx], frequency_mhz[vi],
                         response[vi].imag)
        unwrapped = np.unwrap(np.angle(real + 1j * imag))
        omega = 2.0 * math.pi * frequency_mhz[idx]   # rad per us
        raw_delay = -np.gradient(unwrapped, omega)     # us
        # smooth over the instrument's own resolution (reciprocal
        # window length) - bins are finer than the information
        kernel_avg = np.ones(resolution_bins) / resolution_bins
        smooth_delay = np.convolve(raw_delay, kernel_avg, mode="same")
        # DATA-DERIVED PHASE CEILING.  Above the roll-off band the
        # measured derivative is dominated by the RING (the aftermath
        # oscillating across the tip), whose delay of microseconds
        # gives a group delay near the pulse width - real signal, but
        # the ring's delay, not the roll-off's phase (measured: -4539
        # deg at 3.5 MHz on home, group delay ~5 us; pnb, with no
        # certified sync ring, stays sane).  A roll-off cannot delay
        # an edge by more than that edge's own transition width, so
        # the phase product ends where |group delay| first exceeds
        # the MEASURED sync edge width (spec width as fallback).  The
        # whole-pulse phase is a roll-off product; the ring band belongs
        # to the per-edge shape reading.  The measured width now comes
        # from the accumulated fall's own 10-90 crossings
        # (`ringing_tesseract._edge_width`) rather than from a fitted
        # edge model, which is the same quantity read without a fit.
        measured_edge = state.get("fall_edge_width_samples")
        edge_us = ((float(measured_edge) / rate)
                   if measured_edge is not None and np.isfinite(measured_edge)
                   else geometry.sync_transition_samples / rate)
        over = ((frequency_mhz[idx] > 0.3)
                & (np.abs(smooth_delay) > edge_us))
        if over.any():
            ceiling_mhz = float(frequency_mhz[idx][np.nonzero(over)[0][0]])
        keep = frequency_mhz[idx] <= ceiling_mhz
        phase[idx[keep]] = unwrapped[keep]
        group_delay[idx[keep]] = smooth_delay[keep]
        # THE LOOP THE OTHER WAY (Ethan): the magnitude alone PREDICTS
        # the phase of a causal minimum-phase roll-off (Bode gain-
        # phase; cepstral reconstruction).  The measured phase minus
        # that prediction is the EXCESS phase - what magnitude can
        # never give: the non-minimum-phase / nonlinear / delay
        # content.  Where the excess is ~0 the roll-off is fully
        # described by its slope; where it is not, that is the Q
        # information.  Not circular: the ideal stays the flat spec
        # pulse; the causal fit is a test on the departure.
        magnitude = np.abs(real + 1j * imag)
        magnitude = np.convolve(magnitude, kernel_avg, mode="same")
        log_full = np.full(len(frequency_mhz),
                           math.log(max(float(magnitude[0]), 1e-6)))
        log_full[idx] = np.log(np.maximum(magnitude, 1e-6))
        log_full[idx[-1]:] = log_full[idx[-1]]
        two_sided = np.concatenate([log_full, log_full[-2:0:-1]])
        cepstrum = np.fft.ifft(two_sided).real
        count = len(cepstrum)
        folded = np.zeros(count)
        folded[0] = cepstrum[0]
        folded[1:count // 2] = 2.0 * cepstrum[1:count // 2]
        folded[count // 2] = cepstrum[count // 2]
        minimum_phase = np.fft.fft(folded).imag[:len(frequency_mhz)]
        min_phase = np.full(len(frequency_mhz), np.nan)
        excess = np.full(len(frequency_mhz), np.nan)
        min_phase[idx[keep]] = minimum_phase[idx[keep]]
        excess[idx[keep]] = unwrapped[keep] - minimum_phase[idx[keep]]
    else:
        min_phase = np.full(len(frequency_mhz), np.nan)
        excess = np.full(len(frequency_mhz), np.nan)
    return {
        "H": response, "se": se, "valid": valid,
        "phase_rad": phase, "group_delay_us": group_delay,
        "phase_ceiling_mhz": ceiling_mhz,
        "min_phase_rad": min_phase, "excess_phase_rad": excess,
        "mean": measured, "ideal": ideal,
        "window_lags": np.array([lo, hi]),
        "t0_fall": t0_fall, "t0_rise": t0_rise,
        "pulse_width_us": (t0_rise - t0_fall) / rate,
        "spec_width_us": geometry.sync_pulse_samples / rate,
    }


# --------------------------------------------------------------------------
# the LOW-FREQUENCY response, from the actual vertical sync pulses
# --------------------------------------------------------------------------

# A horizontal sync pulse is 4.7 microseconds long, so a response measured
# from it carries nothing below a few hundred kilohertz. The vertical
# interval holds the only long pulses in the signal: the broad serrated
# vertical sync, tens of times longer, and the equalizing pulses at twice
# line rate. They are excluded from every other measurement here for good
# reason - their widths differ from a horizontal pulse by specification,
# and averaging them together would corrupt the average. For a LOW-
# FREQUENCY response they are exactly what is needed, so this measurement
# admits them, and admits them the same way everything else here is
# measured: each KIND of line averaged within itself first, then the
# residual derived against that kind's own ideal.

VERTICAL_KINDS = ("equalizing_before", "vertical_sync", "equalizing_after",
                  "horizontal")


def vertical_populations(sys_params, line_offset=0):
    """The spec-defined line populations of the vertical interval.

    The interval is three sections of `numPulses` pulses each at half line
    rate: pre-equalizing, serrated vertical sync, post-equalizing. The
    populations come from those numbers, never from looking at waveforms."""
    pulses = int(sys_params["numPulses"])
    section = int(math.ceil(pulses / 2.0))
    start = int(line_offset)
    first = (start, start + section)
    broad = (first[1], first[1] + section)
    second = (broad[1], broad[1] + section)
    return {
        "equalizing_before": first,
        "vertical_sync": broad,
        "equalizing_after": second,
        "horizontal": (second[1], None),
    }


def _settled(trace, window):
    values = trace[window[0]:window[1]]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def flat_intervals(length, edges, guard_samples):
    """The FLAT parts only: every sample far enough from an edge that the
    transient has settled.

    The transitions are deliberately excluded here. What the flat part of a
    pulse carries is the settled LEVEL, and the level over a long pulse is
    the low-frequency information the horizontal sync is too short to hold.
    The transitions carry something else entirely, and mixing the two would
    put the transient's own shape into a level measurement."""
    keep = np.ones(int(length), dtype=bool)
    for edge in edges:
        low = max(int(np.floor(edge - guard_samples)), 0)
        high = min(int(np.ceil(edge + guard_samples)) + 1, int(length))
        if high > low:
            keep[low:high] = False
    return keep


def vertical_step_response(field_lines_ire, sys_params, geometry,
                           line_offset=0, hsync_transient=None):
    """Each kind of vertical-interval line averaged WITHIN its kind, split
    into the two things a pulse actually carries.

    THE FLAT PART gives the settled level, which over a long pulse is
    low-frequency information no horizontal sync pulse can hold. Only the
    flat part is used for it - every sample within a settling guard of an
    edge is excluded, so the transient's shape never enters a level.

    THE TRANSIENT is not new information. An equalizing pulse's edge is the
    same edge, through the same channel, as the horizontal sync pulse's,
    which has already been measured. So the two are DIFFERENTIATED
    TOGETHER: pass the horizontal transient in as `hsync_transient` and
    what comes back is the DIFFERENCE between them. That difference is the
    measurement - it is zero if the channel treats both edges alike, and
    non-zero only where the response depends on the pulse's width or its
    level, which is precisely the thing worth knowing."""
    populations = vertical_populations(sys_params, line_offset)
    rate_mhz = float(sys_params["outfreq"])
    rows = np.asarray(field_lines_ire, dtype=np.float64)
    depth = abs(float(sys_params["vsync_ire"]))
    transition = float(sys_params["syncTransitionUS"]) * rate_mhz
    guard = transition + float(getattr(geometry, "transition_settle_samples",
                                       transition))
    widths_us = {
        "horizontal": float(sys_params["hsyncPulseUS"]),
        "equalizing_before": float(sys_params["hsyncPulseUS"]) / 2.0,
        "equalizing_after": float(sys_params["hsyncPulseUS"]) / 2.0,
        "vertical_sync": float(sys_params["line_period"]) / 2.0
        - float(sys_params["hsyncPulseUS"]) / 2.0,
    }
    anchor = float(getattr(geometry, "sync_fall_index", 0.0))
    out = {}
    for kind, (start, stop) in populations.items():
        stop = rows.shape[0] if stop is None else min(stop, rows.shape[0])
        if start >= stop:
            continue
        block = rows[start:stop]
        averaged = block.mean(axis=0)
        scatter = (block.std(axis=0) / np.sqrt(block.shape[0])
                   if block.shape[0] > 1 else np.zeros_like(averaged))
        width_samples = widths_us[kind] * rate_mhz
        edges = (anchor, anchor + width_samples)
        flat = flat_intervals(len(averaged), edges, guard)
        # the flat part INSIDE the pulse and the flat part outside it
        index = np.arange(len(averaged), dtype=np.float64)
        inside = flat & (index > edges[0]) & (index < edges[1])
        outside = flat & ((index < edges[0]) | (index > edges[1]))
        tip = float(np.median(averaged[inside])) if inside.sum() > 3 else np.nan
        blank = float(np.median(averaged[outside])) if outside.sum() > 3 else np.nan
        # the TRANSIENT, taken over the falling edge only
        window = slice(max(int(anchor - guard), 0),
                       min(int(anchor + guard) + 1, len(averaged)))
        transient = averaged[window]
        normalised = ((transient - tip) / max(blank - tip, 1e-9)
                      if np.isfinite(tip) and np.isfinite(blank) else transient)
        entry = {
            "lines": int(block.shape[0]),
            "averaged": averaged,
            "error": scatter,
            "flat_mask": flat,
            "tip_ire": tip,
            "blanking_ire": blank,
            "depth_ire": (blank - tip) if np.isfinite(tip) and np.isfinite(blank)
                         else np.nan,
            "depth_error_ire": float(np.sqrt(
                np.nanmean(scatter[inside] ** 2) + np.nanmean(scatter[outside] ** 2)))
                if inside.sum() > 3 and outside.sum() > 3 else np.nan,
            "flat_samples_inside": int(inside.sum()),
            "width_us": widths_us[kind],
            "lowest_frequency_hz": 1.0 / (2.0 * widths_us[kind] * 1e-6),
            "transient": normalised,
            "specified_depth_ire": depth,
        }
        # differentiated against the horizontal transient, where given
        if hsync_transient is not None and len(normalised):
            reference = np.asarray(hsync_transient, dtype=np.float64)
            width = min(len(reference), len(normalised))
            difference = normalised[:width] - reference[:width]
            entry["transient_difference"] = difference
            entry["transient_difference_rms"] = float(
                np.sqrt(np.nanmean(difference ** 2)))
        out[kind] = entry
    # every kind's transient differentiated against the horizontal one, so
    # the shared edge response is removed and only what differs remains
    if hsync_transient is None and "horizontal" in out:
        reference = out["horizontal"]["transient"]
        for kind, entry in out.items():
            if kind == "horizontal":
                continue
            width = min(len(reference), len(entry["transient"]))
            difference = entry["transient"][:width] - reference[:width]
            entry["transient_difference"] = difference
            entry["transient_difference_rms"] = float(
                np.sqrt(np.nanmean(difference ** 2)))
    return out


def vertical_low_frequency_reach(sys_params):
    """How far down each kind of pulse reaches, so the gain from admitting
    the vertical interval is stated rather than assumed."""
    rate = {}
    horizontal = float(sys_params["hsyncPulseUS"])
    broad = (float(sys_params["line_period"]) / 2.0 - horizontal / 2.0)
    rate["horizontal_hz"] = 1.0 / (2.0 * horizontal * 1e-6)
    rate["equalizing_hz"] = 1.0 / (2.0 * (horizontal / 2.0) * 1e-6)
    rate["vertical_sync_hz"] = 1.0 / (2.0 * broad * 1e-6)
    rate["improvement"] = rate["horizontal_hz"] / rate["vertical_sync_hz"]
    return rate


def polarity_shape(product, view):
    """Per-polarity SHAPE of one view's settled aftermath, in Hilbert
    space, with no order to choose.

    WHAT THIS REPLACES, AND WHY (Ethan, 2026-09-06: *"I think the matrix
    pencil is the wrong approach. Use the existing sync shape modeling in
    hilbert space not the matrix pencil."*). This used to fit damped
    exponentials by a matrix pencil and export their poles. A pencil needs
    an ORDER, and the order was never fixed by the evidence: on planted
    data it validated, but on real data there is no singular-value knee,
    so the answer follows the pencil's own capacity parameter rather than
    the tape - the same conclusion `vhsdecode/luma_amplitude.py` records
    for the response ripple.

    `ringing_tesseract.kernel_shape` asks the question the data can
    answer instead. The aftermath is fitted in the Laplace eigenbasis
    truncated at the format's own luma band, which makes the split between
    shape and noise a constant rather than a threshold, and its three axes
    - frequency, amplitude and time - are three readings of ONE fit. Bode's
    relation then says how much of the phase the magnitude already
    implies, and what is left is the excess: a delay, or an all-pass that
    no magnitude can invert.

    Still the polarity split the combined export cannot give: the fall
    aftermath settles against the tip carrier, the rise against the
    blanking carrier.
    """
    mean = np.asarray(product["mean"], np.float64)
    ideal = np.asarray(product["ideal"], np.float64)
    residual = mean - ideal
    t0_local = product["t0_sample"] - product["window_lags"][0]
    start = int(t0_local + settle)
    segment = residual[start:]
    if len(segment) < 16:
        return None
    reading = model.kernel_shape(segment, geometry, SHAPE_BAND_HZ)
    shape = reading["shape"]
    return {
        "amplitude": np.asarray(shape["amplitude"], dtype=float),
        "frequency_hz": np.asarray(shape["frequency_hz"], dtype=float),
        "group_delay_s": np.asarray(shape["group_delay_s"], dtype=float),
        "centroid_hz": float(shape["centroid_hz"]),
        "effective_rank": float(shape["effective_rank"]),
        "noise_rms": float(shape["noise_rms"]),
        "amplitude_rms": float(shape["amplitude_rms"]),
        "excess_phase_rms": float(reading["excess_phase_rms"]),
        "in_band_share": float(reading["in_band_share"]),
    }


METADATA = {
    # the exact interface keys the head_switch consumer (vhs-decode-b5)
    # named, so the contract is machine-readable, not prose-only
    "site": "decoded_luma_output_post_demod_post_deemph_post_tbc_4fsc",
    "ideal_placement": "output_site_same_as_measured",
    "ideal_preemphasized": False,   # ideal is a plain erf at OUTPUT
    #                                 level; consumer re-references to
    #                                 its pre-de-emphasis site by
    #                                 dividing the decoder's own FVideo
    "valid_hz": [50000.0, 3500000.0],
    "valid_hz_note": ("100k-3M determined; 50k-100k one marginal "
                      "band; <50k structurally absent"),
    "units": "IRE_per_tbc_json_black_white",
    "y_comb": "off",
    "measurement_point": ("decoded luma output: post-demod, "
                          "post-de-emphasis, post-TBC, output rate "
                          "(4*fsc), IRE per tbc.json; y_comb assumed "
                          "off; runtime-offline parity established"),
    "ideal_step": ("erf at SPEC syncTransitionUS (10-90), centered at "
                   "the fitted 50% crossing (subsample) = phase zero; "
                   "amplitude = signed step between settled medians"),
    "H_definition": ("rfft(tukey*diff(measured), n=%d) / "
                     "rfft(tukey*diff(ideal), n=%d); per head and "
                     "polarity, each normalized by its OWN signed "
                     "step via the signed ideal" % (FFT_LENGTH,
                                                    FFT_LENGTH)),
    "even_odd": ("H_even=(H_fall+H_rise)/2 is the only part an LTI "
                 "consumer may fold; H_odd=(H_fall-H_rise)/2 is a "
                 "validity bound, never corrected"),
    "uncertainty": ("se = Gaussian per-component SE from cross-line "
                    "variance of the accumulated mean (approximate); "
                    "H_first/H_second are independent-half estimates; "
                    "resolution_mhz per polarity is the true "
                    "information spacing - bins are finer"),
    "validity": ("~0.1-3 MHz determined; 0.05-0.1 one marginal band; "
                 "below 0.05 MHz structurally absent (window length, "
                 "per-field porch anchoring). Ring band belongs to "
                 "the per-polarity shape reading; below ~0.3 MHz use H"),
    "operating_point": ("fall spans ~0 to -40 IRE, rise ~-40 to 0; "
                        "~40 IRE linearizations - the span law b(S) "
                        "and the >4.9 MHz blind zone bound "
                        "extrapolation to large edges"),
    "per_polarity_shape": ("head_<h>_<view>_shape_{amplitude,"
                           "frequency_hz,group_delay_s,centroid_hz,"
                           "effective_rank,noise_rms,amplitude_rms,"
                           "excess_phase_rms,in_band_share} + "
                           "_carrier_hz: the settled aftermath fitted in "
                           "the Laplace eigenbasis truncated at the "
                           "format's 3.0 MHz VHS luma band "
                           "(models/sync_shape), which splits shape from "
                           "noise by a format constant instead of a "
                           "threshold, plus what Bode's relation leaves "
                           "as excess phase. THIS REPLACES THE PENCIL "
                           "POLE TABLE (2026-09-06, Ethan's ruling): a "
                           "damped-mode fit needs an order the evidence "
                           "does not fix - on real data there is no "
                           "singular-value knee - while a shape reading "
                           "needs none. carrier_hz is the VHS NTSC "
                           "deviation-map carrier the view landed on "
                           "(tip 3.40 MHz fall, blanking 3.69 rise)"),
    "wiener_kernel": ("head_<h>_<view>_wiener_kernel (+_us time axis, "
                      "+_wiener_H): the sync edge's DERIVATIVE "
                      "deconvolved against the ideal's derivative "
                      "with the measured noise as regularizer "
                      "(|I|^2 + sigma^2 per bin), brought back to the "
                      "TIME dimension = channel impulse response per "
                      "unit signed step at this view's landing "
                      "carrier = one measured row of the (frequency x "
                      "decay) table indexed in RF. Truncated at the "
                      "tip's flat span (witnessability bound), far "
                      "end tapered. REALIZATION: subtraction on a "
                      "gated drive per landing and polarity - never a "
                      "video-domain filter (design law; the Wiener FIR "
                      "null). Only the linear-even part is foldable."),
    "pulse_response": ("head_<h>_pulse_{H,se,valid,phase_rad,"
                       "group_delay_us,mean,ideal,t0_fall,t0_rise,"
                       "width_us,spec_width_us}: the derivative taken "
                       "ACROSS THE WHOLE SYNC PULSE (fall+rise as one "
                       "two-impulse input) divided by the same for the "
                       "ideal two-edge pulse (spec erf shape, measured "
                       "plateau levels, BOTH edges at their MEASURED "
                       "crossings). The known edge separation pins the "
                       "phase at every frequency, so phase_rad "
                       "(unwrapped over valid bins only - the two-"
                       "impulse ideal has comb nulls at n/width, "
                       "masked) and group_delay_us = -dphi/domega are "
                       "the roll-off's EXACT phase response. Measured-"
                       "crossing referencing keeps the channel's edge-"
                       "SHAPE phase and treats the recorded pulse width "
                       "(off spec on this deck) as timing, reported as "
                       "width_us vs spec_width_us. Fall-aligned mean: "
                       "the rise is slightly smeared by width jitter - "
                       "negligible for the LF/mid roll-off phase. "
                       "phase_ceiling_mhz: the phase product ends where "
                       "|group delay| first exceeds the measured sync "
                       "edge width - above it the derivative is the "
                       "RING's delay (microseconds), not the roll-off; "
                       "the ring band belongs to the per-edge poles. "
                       "min_phase_rad = the phase a causal minimum-"
                       "phase roll-off with this magnitude MUST have "
                       "(cepstral Bode reconstruction); excess_phase_"
                       "rad = measured - min_phase = what magnitude "
                       "cannot give (the loop run the other way as a "
                       "consistency check, not a circularity: the "
                       "ideal stays the flat spec pulse)."),
}


if __name__ == "__main__":
    os.makedirs(PLOT_DIRECTORY, exist_ok=True)
    for prefix in sys.argv[1:]:
        fields, _video_parameters = report.load_decode(prefix)
        export = {"frequency_mhz": frequency_mhz,
                  "rate_mhz": np.float64(rate),
                  "metadata": np.array(json.dumps(METADATA))}
        figure, axes = plt.subplots(2, 2, figsize=(14, 8))
        for row, (parity, head) in enumerate(((1, "a"), (0, "b"))):
            state = accumulate(fields, parity)
            halves = {half: accumulate(fields, parity, half)
                      for half in ("first", "second")}
            # THE HEAD-LEVEL CERTIFIED POLE TABLE IS GONE, deliberately.
            # It came from the parametric channel fit the correction no
            # longer makes, and the estimator behind it needed an order
            # the evidence does not fix. What stands in its place is the
            # per-polarity shape reading below, which needs none.
            per_polarity = {}
            for view in ("fall", "rise"):
                product = step_response(state, view)
                if product is None:
                    continue
                per_polarity[view] = product
                base = f"head_{head}_{view}"
                export[base + "_H"] = product["H"]
                export[base + "_se"] = product["se"]
                export[base + "_valid"] = product["valid"]
                export[base + "_mean"] = product["mean"]
                export[base + "_ideal"] = product["ideal"]
                export[base + "_window_lags"] = product["window_lags"]
                export[base + "_t0_sample"] = np.float64(
                    product["t0_sample"])
                export[base + "_step_ire"] = np.float64(
                    product["step_ire"])
                export[base + "_resolution_mhz"] = np.float64(
                    product["resolution_mhz"])
                for label, half_state in halves.items():
                    half_product = step_response(half_state, view)
                    if half_product is not None:
                        export[f"{base}_H_{label}"] = half_product["H"]
                # the per-polarity SHAPE of the settled aftermath, and the
                # carrier this view landed on (the LTI-landing against
                # nonlinearity split). Replaces the pencil pole table.
                shape = polarity_shape(product, view)
                if shape is not None:
                    for key, value in shape.items():
                        export[f"{base}_shape_{key}"] = (
                            value if isinstance(value, np.ndarray)
                            else np.float64(value))
                export[base + "_carrier_hz"] = np.float64(
                    CARRIER_TIP_HZ if view == "fall"
                    else CARRIER_PORCH_HZ)
                # the Wiener-derived time-domain kernel: one measured
                # row of the (frequency x decay) table at this carrier
                export[base + "_wiener_H"] = product["wiener_H"]
                export[base + "_wiener_kernel"] = product["wiener_kernel"]
                export[base + "_wiener_kernel_us"] = product[
                    "wiener_kernel_us"]
            if "fall" in per_polarity and "rise" in per_polarity:
                both = (per_polarity["fall"]["valid"]
                        & per_polarity["rise"]["valid"])
                h_even = 0.5 * (per_polarity["fall"]["H"]
                                + per_polarity["rise"]["H"])
                h_odd = 0.5 * (per_polarity["fall"]["H"]
                               - per_polarity["rise"]["H"])
                h_even[~both] = np.nan
                h_odd[~both] = np.nan
                export[f"head_{head}_H_even"] = h_even
                export[f"head_{head}_H_odd"] = h_odd
                se_even = 0.5 * np.hypot(per_polarity["fall"]["se"],
                                         per_polarity["rise"]["se"])
                export[f"head_{head}_H_even_se"] = se_even
            # whole-pulse derivative: the roll-off's EXACT phase
            pulse = pulse_response(state)
            if pulse is not None:
                pbase = f"head_{head}_pulse"
                export[pbase + "_H"] = pulse["H"]
                export[pbase + "_se"] = pulse["se"]
                export[pbase + "_valid"] = pulse["valid"]
                export[pbase + "_phase_rad"] = pulse["phase_rad"]
                export[pbase + "_group_delay_us"] = pulse[
                    "group_delay_us"]
                export[pbase + "_mean"] = pulse["mean"]
                export[pbase + "_ideal"] = pulse["ideal"]
                export[pbase + "_window_lags"] = pulse["window_lags"]
                export[pbase + "_t0_fall"] = np.float64(pulse["t0_fall"])
                export[pbase + "_t0_rise"] = np.float64(pulse["t0_rise"])
                export[pbase + "_width_us"] = np.float64(
                    pulse["pulse_width_us"])
                export[pbase + "_spec_width_us"] = np.float64(
                    pulse["spec_width_us"])
                export[pbase + "_phase_ceiling_mhz"] = np.float64(
                    pulse["phase_ceiling_mhz"])
                export[pbase + "_min_phase_rad"] = pulse["min_phase_rad"]
                export[pbase + "_excess_phase_rad"] = pulse[
                    "excess_phase_rad"]
                # CYCLE 2 of the loop (Ethan: cycle the measurements
                # until the residual approaches zero or the noise
                # floor).  Cycle 1 predicted the minimum phase from the
                # pulse's OWN magnitude truncated at the phase ceiling,
                # and a truncated magnitude reconstructs a spurious LF
                # lead (measured +12..+17 deg at 0.1 MHz - unphysical
                # for a roll-off).  Use the neighbouring measurement:
                # the per-edge EVEN magnitude, honest to ~3 MHz.
                h_even = export.get(f"head_{head}_H_even")
                fall_product = per_polarity.get("fall")
                if h_even is not None and fall_product is not None:
                    bin_mhz = frequency_mhz[1] - frequency_mhz[0]
                    smooth_bins = max(3, int(round(
                        fall_product["resolution_mhz"] / bin_mhz)) | 1)
                    magnitude = np.abs(h_even)
                    finite = np.isfinite(magnitude)
                    filled = magnitude.copy()
                    filled[~finite] = np.nan
                    kernel_avg = np.ones(smooth_bins) / smooth_bins
                    held = np.where(finite, magnitude, 0.0)
                    weight = np.convolve(finite.astype(np.float64),
                                         kernel_avg, mode="same")
                    smoothed = np.convolve(held, kernel_avg,
                                           mode="same")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        smoothed = np.where(weight > 0,
                                            smoothed / weight, np.nan)
                    wide_min_phase = minimum_phase_from_magnitude(
                        smoothed)
                    measured_phase = pulse["phase_rad"]
                    have = np.isfinite(measured_phase)
                    excess_wide = np.full(len(frequency_mhz), np.nan)
                    excess_wide[have] = (measured_phase[have]
                                         - wide_min_phase[have])
                    export[pbase + "_min_phase_wideband_rad"] = np.where(
                        have, wide_min_phase, np.nan)
                    export[pbase + "_excess_phase_wideband_rad"] = (
                        excess_wide)
            # figure row
            colors = {"fall": "#2060c0", "rise": "#c05020"}
            for column, view_name in enumerate(("magnitude", "phase")):
                axis = axes[row][column]
                for view, product in per_polarity.items():
                    keep = product["valid"]
                    values = (np.abs(product["H"][keep])
                              if column == 0 else
                              np.degrees(np.angle(product["H"][keep])))
                    axis.plot(frequency_mhz[keep], values,
                              color=colors[view], linewidth=1.0,
                              label=view, alpha=0.8)
                if f"head_{head}_H_even" in export:
                    h_even = export[f"head_{head}_H_even"]
                    keep = np.isfinite(h_even)
                    values = (np.abs(h_even[keep]) if column == 0
                              else np.degrees(np.angle(h_even[keep])))
                    axis.plot(frequency_mhz[keep], values, "k-",
                              linewidth=1.6, label="even (foldable)")
                    if column == 0:
                        se_even = export[f"head_{head}_H_even_se"]
                        axis.fill_between(
                            frequency_mhz[keep],
                            np.abs(h_even[keep]) - se_even[keep],
                            np.abs(h_even[keep]) + se_even[keep],
                            color="black", alpha=0.15,
                            label="even +-1 SE")
                axis.axvspan(0.04, 0.1, color="#c0c0c0", alpha=0.25)
                axis.set_xlim(0.0, 3.5)
                axis.set_xlabel("MHz")
                axis.set_title(
                    f"head {head.upper()} "
                    + ("|H|" if column == 0 else "arg H (deg)")
                    + "   (grey = marginal band)", fontsize=10)
                axis.grid(alpha=0.3)
                axis.legend(fontsize=7)
                if column == 0:
                    axis.axhline(1.0, color="gray", linewidth=0.6)
                else:
                    axis.axhline(0.0, color="gray", linewidth=0.6)
        figure.suptitle(
            f"{prefix}: sync-step complex response "
            f"(per polarity; even = the LTI-foldable part)",
            fontsize=11)
        figure.tight_layout()
        figure_path = os.path.join(
            PLOT_DIRECTORY, f"{prefix}_sync_step_response.png")
        figure.savefig(figure_path, dpi=110, facecolor="white",
                       bbox_inches="tight")
        plt.close(figure)
        # top-level aliases matching the head_switch consumer's
        # pre-written loader (baseband_eq.py::_load_sync_step): freqs
        # in Hz, H_even_<head>/H_odd_<head>/se_even_<head>, scalar
        # site/ideal_preemphasized/valid_hz. The head_<head>_* names
        # remain the canonical form; these are additive so the file reads
        # with zero adaptation on their end. `ring_poles_hz_<head>` is no
        # longer written: it came from the certified pole table, and the
        # pole table is gone with the estimator whose order the data did
        # not fix (2026-09-06). A consumer that wants the ring band should
        # read head_<h>_<view>_shape_* instead.
        export["freqs"] = frequency_mhz * 1e6
        export["site"] = np.array(METADATA["site"])
        export["ideal_preemphasized"] = np.array(
            METADATA["ideal_preemphasized"])
        export["valid_hz"] = np.array(METADATA["valid_hz"],
                                      dtype=float)
        for head in ("a", "b"):
            for consumer_key, canonical in (
                    (f"H_even_{head}", f"head_{head}_H_even"),
                    (f"H_odd_{head}", f"head_{head}_H_odd"),
                    (f"se_even_{head}", f"head_{head}_H_even_se")):
                if canonical in export:
                    export[consumer_key] = export[canonical]
        out = os.path.join(WORK_DIRECTORY,
                           f"{prefix}_sync_step_response.npz")
        np.savez(out, **export)
        print(f"{prefix}: {out}")
        print(f"   figure: {figure_path}")
