#!/usr/bin/env python3
"""Multidimensional information extrapolation - the offline loop, through
the real chain.

Drives `vhsdecode.information_extrapolation.multidimensional_information_
extrapolation` with gauges that decode a subset of the tape on every pass:

THE RESULT IS THE MODELLED SIGNAL. The loop does not accumulate a
correction and apply fractions of it along the way; it accumulates the
EXPECTED signal. Every pass the whole model is applied through the real
chain in ONE decode, every component's residual is measured against it,
the residuals are cross-checked against each other, and each is added
back into its own component's expected signal. The iteration ends at the
LIMIT - the residual approaching zero - and what has accumulated by then
is the shape the specification's ideal was missing. The correction that
follows is that accumulation's departure from the ideal, applied once.

  pass 0   decode with --residual_channels only; the channel identifier
           (Part A, `channel_identify.py`) supplies the BASELINE magnitude
           from the constant envelope.
  pass 1   the baseline fed forward.
  pass k   ONE decode carrying the whole model so far (--channel_response
           and --channel_eq for the frequency axis, --time_base_response
           for the sync resampler), every component measured on it, the
           residuals cross-checked, each accumulated into its own
           expected signal.

Components, staggered in Ethan's order (timing before anything spectral):
  time       the sync resampler: per line of the output field, the
             residual POSITION of the line on the output grid, read from
             the BURST (the fine timing pilot: its phase residual is a
             delay) - measured, the luma sync fall wanders ~1.7 samples
             about the burst-locked grid while the burst says 0.03, so the
             sync fall's wander is the luma path's sync-to-burst timing,
             reported as the luma alignment residual and never fed back.
             The residual is returned to the line locations through
             --time_base_response in position form as the TOTAL estimate
             (what the resampler applied, the field's own time channel,
             plus the half step of the residual, SE from the burst fit) -
             the head-switch lane's consumer blends it with the carrier
             trace by inverse variance. Its floor is the gauge's own error
             (chi-square of one): a field's timing residual is direct
             evidence about that field.
  frequency  the sync-step response per polarity (the ringing lane's
             gauge), fitted on the first half of each head's fields and
             measured on the second, the decoder's own static response
             divided out analytically, mapped onto the RF axis about each
             polarity's landing carrier: the even part of the magnitude
             and the odd part of the phase about the carrier are what an
             FM demodulator makes of an RF filter.
  noise      last, amplitude only: the gauge's variance of the mean; the
             per-field non-reducible residual is the field axis.

Every pass decode is followed by the hsync artifact model plot. The final
export is the pass with the lowest held-out residual (stop one pass
earlier when the residual stops dropping).

    multidimensional_information_extrapolation.py --tape T --work W
        --prefix P [--frames 10] [--seek 0] [--flags "-n -f 50 -t 0"]
        [--passes 3] [--amount 0.5] [--skip-time]
"""

import argparse
import glob
import math
import json
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from vhsdecode.models import information_extrapolation as ie  # noqa: E402
from vhsdecode.baseband_eq import _log_transfer, _pool  # noqa: E402 (frozen)
import channel_identify  # noqa: E402
import sync_step_response as ssr  # noqa: E402

DECODER = os.path.join(REPO, "vhs-decode")
REPORT = os.path.join(HERE, "hsync_model_report.py")
MINIMUM_STEP_IRE = 10.0
FREQUENCY_COMPONENT = "sync-step response, per polarity"

# The most a residual may depart from unity and still be evidence about
# the CHANNEL: 3 dB, the same ceiling the baseband stage ships. Beyond
# it the measured edge has lost more than half its amplitude, and what
# is missing is the tape's and the decoder's own bandwidth - information
# that is gone, not a response to invert. Inverting it is how the loop
# once produced a table asking for 37 dB, which destroyed the RF so
# completely that the decoder found no fields.
CEILING_NEPERS = math.log(2.0)          # 3 dB
# and the most the accumulated table may ever ask for, matching the
# runtime stage's own clamp
MAXIMUM_LOG_MAGNITUDE = 1.0


# --------------------------------------------------------------------------
# the decode and its products
# --------------------------------------------------------------------------


_DECODE_COUNT = [0]


def next_index():
    """A globally unique decode number: two components' feed-forwards must
    never share a prefix (the decoder trips over the half-written outputs
    of the first and still exits zero)."""
    _DECODE_COUNT[0] += 1
    return _DECODE_COUNT[0]


class Decode:
    """One pass's decode: where it is and what fed it."""

    def __init__(self, work, prefix, index, channel=None, time=None):
        self.index = index
        self.name = f"{prefix}_pass{index}"
        self.prefix = os.path.join(work, self.name)
        self.residual_dir = os.path.join(work, f"{self.name}_rc")
        self.channel = channel
        self.time = time
        self.fields = None
        self.json = None
        self.filters = None
        self.readlocs = None


def run_decode(args, decode):
    command = [sys.executable, DECODER, args.tape, decode.prefix]
    command += shlex.split(args.flags)
    command += ["-l", str(args.frames)]
    if args.seek:
        command += ["-s", str(args.seek)]
    if os.path.isdir(decode.residual_dir):
        shutil.rmtree(decode.residual_dir)
    command += ["--residual_channels", decode.residual_dir]
    if decode.channel is not None:
        command += ["--channel_response", decode.channel, "--channel_eq", "1"]
    if decode.time is not None:
        command += ["--time_base_response", decode.time]
    command += ["--overwrite"]
    env = dict(os.environ, PYTHONPATH=REPO)
    with open(decode.prefix + ".stdout", "w") as out, \
            open(decode.prefix + ".stderr", "w") as err:
        subprocess.run(command, env=env, stdout=out, stderr=err, check=True)
    # the decoder can fail and still exit zero, so its products are the
    # test: no tbc, no json, or no residual channels means no decode
    for required in (decode.prefix + ".tbc", decode.prefix + ".tbc.json",
                     os.path.join(decode.residual_dir, "decoder_filters.npz")):
        if not os.path.exists(required):
            with open(decode.prefix + ".stderr") as handle:
                tail = "".join(handle.readlines()[-12:])
            raise RuntimeError(f"{decode.name}: the decode produced no "
                               f"{os.path.basename(required)}\n{tail}")
    # the per-decode plot rule
    subprocess.run([sys.executable, REPORT, decode.name],
                   env=dict(env, RC_WORK=os.path.dirname(decode.prefix)),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    load_products(decode)
    return decode


def load_products(decode):
    with open(decode.prefix + ".tbc.json") as handle:
        decode.json = json.load(handle)
    parameters = decode.json["videoParameters"]
    width, height = parameters["fieldWidth"], parameters["fieldHeight"]
    blanking, white = parameters["black16bIre"], parameters["white16bIre"]
    samples = np.fromfile(decode.prefix + ".tbc", dtype="<u2")
    count = samples.size // (width * height)
    decode.fields = ((samples[: count * width * height]
                      .reshape(count, height, width).astype(np.float64)
                      - blanking) / ((white - blanking) / 100.0))
    decode.readlocs = np.array([int(f["fileLoc"]) for f in
                                decode.json["fields"]][:count])
    filters_path = os.path.join(decode.residual_dir, "decoder_filters.npz")
    decode.filters = {k: v for k, v in
                      np.load(filters_path, allow_pickle=False).items()}


def residual_exports(decode):
    """readloc -> the field's residual-channel export (lazy handles)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(decode.residual_dir,
                                              "field_*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            out[int(z["readloc"])] = path
    return out


# --------------------------------------------------------------------------
# the frequency gauge: the sync-step response with the decoder divided out
# --------------------------------------------------------------------------


def static_divisor(filters, freqs_hz):
    """|D(f)| between the demodulator and the decoded output, de-emphasis
    excluded: FVideo / FDeemp on the block's rfft grid times the time base
    corrector's half-sample interpolation leg (the baseband stage's form)."""
    fvideo = np.asarray(filters["fvideo"], dtype=np.complex128)
    fdeemp = np.asarray(filters.get("fdeemp", np.ones_like(fvideo)),
                        dtype=np.complex128)
    rate = float(filters["freq_hz"])
    grid = np.arange(len(fvideo)) * (rate / (2 * (len(fvideo) - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        path = np.where(np.abs(fdeemp) > 0, fvideo / fdeemp, 1.0)
    path = np.nan_to_num(np.abs(path), nan=1.0)
    divisor = np.interp(freqs_hz, grid, path)
    taps = np.asarray(filters.get("downscale_taps_half",
                                  filters["downscale_taps"]), dtype=np.float64)
    t = np.arange(len(taps)) - (len(taps) - 1) / 2.0
    phase = np.exp(-2j * np.pi * np.outer(freqs_hz, t) / rate)
    return divisor * np.abs(phase @ taps)


SNR_GATE = 3.0   # a bin whose magnitude is within 3 sigma of its noise
                 # carries no evidence (the noise model as the constraint)


def load_decoder_reference(path):
    """The decoder's OWN response to a spec sync edge per polarity - RF
    band-pass, demodulator and video chain together, measured by synthesis
    with no tape (the writeup lane's export) - as view -> (freqs_hz, H).
    The RF band-pass reaches the video only through the demodulator, so it
    has no video-axis transfer to divide out; the per-polarity synthesis
    is the exact form."""
    if not path:
        return None
    z = np.load(path, allow_pickle=False)
    freqs = np.asarray(z["sync_freqs_hz"], dtype=np.float64)
    out = {}
    for view in ("fall", "rise"):
        key = f"decoder_only_{view}_H_peaking_on"
        if key in z.files:
            out[view] = (freqs, np.asarray(z[key], dtype=np.complex128))
    return out or None


def _reference_divisor(reference, view, freqs_hz):
    """The reference on the gauge's grid with its chain DELAY removed: the
    synthesis anchors its ideal at the input edge, so its raw phase carries
    the decoder's group delay, while the measured response is anchored at
    the output crossing. A constant delay left in the reference would be
    added to every pass's residual and run the phase away. The delay is
    the phase slope over the clean low band; what remains is the chain's
    dispersion."""
    freqs, H = reference[view]
    order = np.argsort(freqs)
    freqs, H = freqs[order], H[order]
    phase = np.unwrap(np.angle(H))
    low = (freqs > 0.05e6) & (freqs < 1.0e6) & (np.abs(H) > 0)
    slope = float(np.polyfit(freqs[low], phase[low], 1)[0]) if low.sum() > 3 \
        else 0.0
    phase = phase - slope * freqs
    magnitude = np.interp(freqs_hz, freqs, np.abs(H))
    return magnitude * np.exp(1j * np.interp(freqs_hz, freqs, phase))


def frequency_residual(decode, parity, subset, reference=None):
    """Per view: (freqs_hz, L, var, valid) - the log residual response of
    the corrected output's sync edges against the spec ideal AS THE
    DECODER RENDERS IT (the decoder-only per-polarity response divided
    out; the video chain's static divisor where no reference is given) -
    for one head parity over one subset of its fields. Bins whose
    measured magnitude is within SNR_GATE sigma of their own noise are
    not evidence."""
    state = ssr.accumulate(decode.fields, parity, subset)
    freqs_hz = ssr.frequency_mhz * 1e6
    fallback = static_divisor(decode.filters, freqs_hz)
    out = {}
    for view in ("fall", "rise"):
        measured = ssr.step_response(state, view)
        if measured is None:
            continue
        H_raw = np.asarray(measured["H"], dtype=np.complex128)
        se_raw = np.asarray(measured["se"], dtype=np.float64)
        valid = np.asarray(measured["valid"], dtype=bool) & np.isfinite(H_raw)
        valid &= np.abs(H_raw) > SNR_GATE * se_raw
        # and only where the channel is within the ceiling: past it the
        # amplitude is gone, not attenuated by something invertible
        with np.errstate(divide="ignore", invalid="ignore"):
            valid &= np.abs(np.log(np.abs(H_raw) / np.abs(
                _reference_divisor(reference, view, freqs_hz)
                if reference and view in reference else fallback))) \
                <= CEILING_NEPERS
        divisor = (_reference_divisor(reference, view, freqs_hz)
                   if reference and view in reference else fallback)
        valid &= np.abs(divisor) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.where(valid, H_raw / divisor, np.nan)
            se = np.where(valid, se_raw / np.abs(divisor), np.nan)
        L, var = _log_transfer(H, se)
        out[view] = (freqs_hz, L, var, valid & np.isfinite(L))
    return out


def landing_carriers(filters):
    """Where each polarity's edge lands: the fall at the sync tip carrier,
    the rise at the blanking carrier - from the decoder's own levels."""
    ire0 = float(filters["ire0"])
    tip = ire0 + float(filters["vsync_ire"]) * float(filters["hz_ire"])
    return {"fall": tip, "rise": ire0}


def pooled_views(residuals_by_head):
    """Pool the heads per view in the log domain (their disagreement joins
    the error budget); returns view -> (freqs, L, var, valid)."""
    out = {}
    for view in ("fall", "rise"):
        rows = [r[view] for r in residuals_by_head if view in r]
        if not rows:
            continue
        freqs = rows[0][0]
        L = np.array([np.where(r[3], r[1], np.nan + 0j) for r in rows])
        var = np.array([np.where(r[3], r[2], np.inf) for r in rows])
        pooled, v_pool, tau2 = _pool(L, var)
        valid = np.isfinite(pooled) & np.isfinite(v_pool)
        out[view] = (freqs, pooled, np.where(valid, v_pool + tau2, np.inf),
                     valid)
    return out


# The measured Jacobian of the RF site: how much of an applied change in
# the pre-demodulator response actually reaches the demodulated edge, per
# view. The relation the rows are built from is first order in the
# modulation index, and a sync edge is a 290 kHz frequency step, so the
# true map is MEASURED (scratchpad/jacobian_map.py) rather than assumed. A
# view whose measured gain is below this floor cannot be corrected from
# the RF at all - its observable does not respond - and its rows are
# dropped rather than amplified by 1/gain, which is how a loop builds a
# table asking for tens of dB.
MINIMUM_JACOBIAN = 0.2


def map_to_rf(views, filters, grid_hz, alpha, jacobian=None):
    """The frequency-axis update of the identified channel on the RF grid,
    solved as a weighted least-squares system rather than deposited bin
    by bin, because the demodulator sees the RF magnitude RELATIVE TO THE
    CARRIER'S OWN GAIN (the limiter normalizes at the carrier): for a
    change e of the applied log-response, each view's baseband residual
    changes by

        magnitude: [e(fc + f) + e(fc - f)] / 2 - e(fc)
        phase:     [e(fc + f) - e(fc - f)] / 2          (the carrier's
                                                          phase cancels)

    and the fall's carrier is the rise's lower sideband 0.29 MHz down, so
    the two polarities constrain each other's carrier bins. The desired
    change is -alpha x the residual (the err-low step). Rows are weighted
    by the residual's precision times the per-bin response agreement
    (`views` may carry it as a fifth element); a ridge pins bins the
    decoder's RF pass-band does not reach (no information there) and
    those no row touches. The runtime applies exp(-log_H), so the
    channel's increment is -e. Returns (increment, variance) on the grid.
    """
    bin_hz = float(grid_hz[1] - grid_hz[0])
    n = len(grid_hz)
    if "rf_grid_hz" in filters and "rf_video" in filters:
        passband = np.interp(grid_hz, np.asarray(filters["rf_grid_hz"]),
                             np.abs(np.asarray(filters["rf_video"]))) ** 2
    else:
        passband = np.ones(n)
    passband = passband / max(float(passband.max()), 1e-30)
    carriers = landing_carriers(filters)
    rows_m, b_m, w_m, rows_p, b_p, w_p = [], [], [], [], [], []
    touched = np.zeros(n, dtype=bool)
    dropped = []
    for view, entry in views.items():
        freqs, L, var, valid = entry[:4]
        agreement = entry[4] if len(entry) > 4 else np.ones(len(freqs))
        # how much of an applied change this view's observable actually
        # shows: measured, not assumed (see MINIMUM_JACOBIAN)
        gain = 1.0 if not jacobian else float(jacobian.get(view, 1.0))
        if abs(gain) < MINIMUM_JACOBIAN:
            dropped.append((view, gain))
            continue
        # a phase beyond +/- pi in one bin is a wrap, not a delay
        L = np.where(np.abs(L.imag) <= np.pi, L, L.real + 0.0j)
        fc = carriers[view]
        ic = int(np.round((fc - grid_hz[0]) / bin_hz))
        for k in np.nonzero(valid)[0]:
            up = int(np.round((fc + freqs[k] - grid_hz[0]) / bin_hz))
            down = int(np.round((fc - freqs[k] - grid_hz[0]) / bin_hz))
            if not (0 <= up < n and 0 <= down < n and 0 <= ic < n):
                continue
            weight = float(agreement[k] / var[k])
            if not np.isfinite(weight) or weight <= 0:
                continue
            touched[[up, down, ic]] = True
            # to move the observable by -alpha x L the APPLIED change must
            # be -alpha x L / gain
            rows_m.append((up, down, ic))
            b_m.append(-(alpha / gain) * float(L[k].real))
            w_m.append(weight)
            rows_p.append((up, down))
            b_p.append(-(alpha / gain) * float(L[k].imag))
            w_p.append(weight)
    increment = np.zeros(n, dtype=np.complex128)
    variance = np.full(n, np.inf)
    map_to_rf.dropped_views = dropped
    if not rows_m:
        map_to_rf.polarity_common = (0.0, 0.0)
        return increment, variance
    # unknowns: the touched span of the grid
    span = np.nonzero(touched)[0]
    lo, hi = int(span.min()), int(span.max()) + 1
    m = hi - lo
    # THE RIDGE IS THE BOUND THE CLAMP ALREADY STATES, in Tikhonov form:
    # a playback channel does not have more than MAXIMUM_LOG_MAGNITUDE of
    # structure, so 1/that squared is the prior precision, and a solution
    # is pulled toward zero exactly as hard as that bound implies rather
    # than by a tuned fraction of the weakest evidence. Divided by the
    # decoder's own pass-band, so bins the demodulator never sees are
    # pinned hard: no information there, no freedom either.
    ridge = np.full(m, 1.0 / (MAXIMUM_LOG_MAGNITUDE ** 2))
    ridge = ridge / np.maximum(passband[lo:hi], 1e-6)

    def solve(rows, b, w, odd):
        A = np.zeros((len(rows) + m, m))
        rhs = np.zeros(len(rows) + m)
        for r, (entry, value, weight) in enumerate(zip(rows, b, w)):
            sw = np.sqrt(weight)
            if odd:
                up, down = entry
                A[r, up - lo] += 0.5 * sw
                A[r, down - lo] -= 0.5 * sw
            else:
                up, down, ic = entry
                A[r, up - lo] += 0.5 * sw
                A[r, down - lo] += 0.5 * sw
                A[r, ic - lo] -= 1.0 * sw
            rhs[r] = value * sw
        A[len(rows):, :] = np.diag(np.sqrt(ridge))
        e = np.linalg.lstsq(A, rhs, rcond=None)[0]
        precision = np.sum(A ** 2, axis=0)
        return e, precision

    def reweight(rows, b, w, odd, e):
        """Ethan's convergence test as an estimator: a single linear RF
        response must explain BOTH polarities: 'if they don't converge
        together, whatever's left isn't linear'. A row the fit cannot
        satisfy beyond its own error is a polarity disagreement - the
        nonlinear remainder (record emphasis, the dark clip), which no
        linear filter at any point in the chain undoes - so it is
        down-weighted by the garrote rather than allowed to drag the
        common response. Returns the new weights and the share of the
        evidence that turned out to be polarity-common."""
        residuals = np.empty(len(rows))
        for r, (entry, value) in enumerate(zip(rows, b)):
            if odd:
                up, down = entry
                predicted = 0.5 * (e[up - lo] - e[down - lo])
            else:
                up, down, ic = entry
                predicted = 0.5 * (e[up - lo] + e[down - lo]) - e[ic - lo]
            residuals[r] = value - predicted
        w = np.asarray(w, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            # the row's own error is 1/w; keep the share of the row the
            # common response explains
            keep = 1.0 - np.minimum(1.0, residuals ** 2 * w)
        keep = np.clip(np.nan_to_num(keep, nan=0.0), 0.0, 1.0)
        return w * keep, float(np.mean(keep))

    e_m, p_m = solve(rows_m, b_m, w_m, odd=False)
    w_m, common_m = reweight(rows_m, b_m, w_m, False, e_m)
    e_m, p_m = solve(rows_m, b_m, w_m, odd=False)
    e_p, p_p = solve(rows_p, b_p, w_p, odd=True)
    w_p, common_p = reweight(rows_p, b_p, w_p, True, e_p)
    e_p, p_p = solve(rows_p, b_p, w_p, odd=True)
    map_to_rf.polarity_common = (common_m, common_p)
    # PROJECT OUT THE NULL SPACE. The magnitude relation
    # [e(f+x)+e(f-x)]/2 - e(f) annihilates any LINEAR function of
    # frequency and the phase relation [e(f+x)-e(f-x)]/2 annihilates any
    # CONSTANT, so those components of a solution do nothing whatever to
    # the demodulated output. Left in, they make the table look far
    # larger than its effect, eat the safety clamp's headroom, and
    # mislead anyone reading it as a channel measurement. (This is also
    # the formal reason a constant-envelope tilt cannot help at this
    # site: it lies in that null space.)
    # fitted over the bins the DATA touch, not the whole solved span: the
    # rest are held near zero by the ridge and would bias the fit
    axis = np.arange(m, dtype=np.float64)
    inside = touched[lo:hi]
    if inside.sum() > 2:
        e_m = e_m - np.polyval(np.polyfit(axis[inside], e_m[inside], 1), axis)
        e_p = e_p - float(np.mean(e_p[inside]))
    # the channel's increment is minus the applied change
    increment[lo:hi] = -(e_m + 1j * e_p)
    with np.errstate(divide="ignore"):
        variance[lo:hi] = 1.0 / np.maximum(p_m, 1e-30)
    variance[lo:hi] = np.where(touched[lo:hi], variance[lo:hi], np.inf)
    return increment, variance


def predicted_effect(increment, filters, grid_hz, view, freqs):
    """What the runtime's inverse of `increment` does to a view's baseband
    log residual at `freqs`, by the same relations map_to_rf solves."""
    bin_hz = float(grid_hz[1] - grid_hz[0])
    e = -increment
    fc = landing_carriers(filters)[view]
    ic = int(np.round((fc - grid_hz[0]) / bin_hz))
    out = np.full(len(freqs), np.nan + 0j)
    for k, f in enumerate(freqs):
        up = int(np.round((fc + f - grid_hz[0]) / bin_hz))
        down = int(np.round((fc - f - grid_hz[0]) / bin_hz))
        if 0 <= up < len(grid_hz) and 0 <= down < len(grid_hz):
            out[k] = (0.5 * (e[up].real + e[down].real) - e[ic].real
                      + 1j * 0.5 * (e[up].imag - e[down].imag))
    return out


# The band every pass is judged on, fixed once so that passes are
# comparable: a metric computed over each pass's own admitted bins moves
# when the correction changes which bins are admitted, and a residual can
# then appear to fall while every bin gets worse.
EVALUATION_BAND_HZ = (0.05e6, 1.5e6)


def _pool_views_rms(per_view, views):
    """The root-mean-square over exactly the named views, weighted by
    each one's bin count - the pooled figure restricted to a view set."""
    total, count = 0.0, 0
    for view in views:
        entry = per_view.get(view)
        if not entry:
            continue
        total += entry["magnitude_rms"] ** 2 * entry["bins"]
        count += entry["bins"]
    return float(np.sqrt(total / count)) if count else None


def fixed_band_metric(views, band=EVALUATION_BAND_HZ, population=None):
    """The residual over the FIXED evaluation band and, when given, the
    FIXED bin population - the only figure that may be compared between
    passes.

    The band alone is not enough. Which bins inside it pass the quality
    gates depends on the correction, so a metric that re-admits bins each
    pass is comparing different sets: measured, 1415 bins uncorrected
    against 1707 after one pass, and the newly admitted ones are the
    band edges where the residual is worst. `population` freezes the set
    to the one the UNCORRECTED decode admitted, so every later pass is
    judged on the bins that were judged at the start."""
    out = {}
    magnitude, phase = [], []
    for view, entry in views.items():
        f, L, var, ok = entry[:4]
        if population is not None and view in population:
            ok = ok & population[view]
        sel = ok & (f >= band[0]) & (f <= band[1])
        if not sel.any():
            continue
        out[view] = {"magnitude_rms": float(np.sqrt(np.mean(L[sel].real ** 2))),
                     "phase_rms": float(np.sqrt(np.mean(L[sel].imag ** 2))),
                     "bins": int(sel.sum())}
        magnitude.append(L[sel].real)
        phase.append(L[sel].imag)
    if magnitude:
        out["pooled"] = {
            "magnitude_rms": float(np.sqrt(np.mean(np.concatenate(magnitude) ** 2))),
            "phase_rms": float(np.sqrt(np.mean(np.concatenate(phase) ** 2))),
            "bins": int(sum(len(v) for v in magnitude))}
    return out


def held_out_metric(views):
    """RMS of the log residual (magnitude, phase) over the believed bins of
    the held-out fields, and the reduced chi-square against the gauge's
    own error - structure above the floor shows as chi-square above one."""
    magnitude, phase, chi = [], [], []
    for freqs, L, var, valid in views.values():
        magnitude.append(L[valid].real)
        phase.append(L[valid].imag)
        chi.append(np.abs(L[valid]) ** 2 / var[valid])
    if not magnitude or not np.concatenate(magnitude).size:
        return {"magnitude_rms": np.nan, "phase_rms": np.nan, "chi2": np.nan,
                "bins": 0}
    per_view = {}
    for view, (freqs, L, var, valid) in views.items():
        if valid.any():
            per_view[view] = {
                "magnitude_rms": float(np.sqrt(np.mean(L[valid].real ** 2))),
                "phase_rms": float(np.sqrt(np.mean(L[valid].imag ** 2))),
                "chi2": float(np.mean(np.abs(L[valid]) ** 2 / var[valid]))}
    magnitude = np.concatenate(magnitude)
    phase = np.concatenate(phase)
    chi = np.concatenate(chi)
    return {"magnitude_rms": float(np.sqrt(np.mean(magnitude ** 2))),
            "phase_rms": float(np.sqrt(np.mean(phase ** 2))),
            "chi2": float(np.mean(chi)), "bins": int(magnitude.size),
            "per_view": per_view}


def structure_test(views):
    """The exhaustion test on the held-out residual: lag-one correlation of
    the log residual across frequency bins (structure) against a
    structureless expectation of zero."""
    values = []
    for freqs, L, var, valid in views.values():
        v = L[valid].real
        if v.size > 8:
            v = v - v.mean()
            values.append(float(np.sum(v[1:] * v[:-1])
                                / max(np.sum(v * v), 1e-30)))
    return float(np.mean(values)) if values else float("nan")


# --------------------------------------------------------------------------
# the time gauge: the sync resampler's residual on the corrected output
# --------------------------------------------------------------------------


def _tbc_windows():
    """The front porch and the sync tip in the tbc's own line alignment
    (the line starts at the sync fall), from the ringing model's spec-
    derived measurement plan: the porch as negative offsets from the
    line end (a settle away from the fall), the tip as offsets from the
    line start (a settle inside each end)."""
    plan, settle = ssr.plan, int(ssr.settle)
    fall = plan.front_porch[1]
    porch_len = plan.front_porch[1] - plan.front_porch[0]
    # the porch is the LAST porch_len samples before the fall; the fall's
    # own transition takes the last couple of samples of the row
    porch_lo, porch_hi = -porch_len, -2
    tip_lo = plan.sync_tip[0] - fall + settle
    tip_hi = plan.sync_tip[1] - fall - settle
    return porch_lo, porch_hi, tip_lo, tip_hi


def line_crossings(field_lines):
    """Per line: the sync fall's 50% crossing column (subsample, relative
    to the line's own start) and its standard error, NaN where the line
    has no measurable sync edge. The time base puts each line's sync fall
    AT the line start, so the fall is read across the boundary: the
    previous row's tail is this line's front porch, this row's head its
    sync tip - the residual is where the resampler actually put the edge.
    Line 0 has no previous row and is not measured."""
    settle = int(ssr.settle)
    width = field_lines.shape[1]
    reach = 4 * settle
    porch_lo, porch_hi, tip_lo, tip_hi = _tbc_windows()
    crossings = np.full(len(field_lines), np.nan)
    errors = np.full(len(field_lines), np.nan)
    for k in range(1, len(field_lines)):
        previous, line = field_lines[k - 1], field_lines[k]
        porch = previous[width + porch_lo:width + porch_hi]
        tip = line[tip_lo:tip_hi]
        if len(porch) < 4 or len(tip) < 4:
            continue
        pre_level, post_level = float(np.median(porch)), float(np.median(tip))
        step = post_level - pre_level
        if abs(step) < MINIMUM_STEP_IRE:
            continue
        crossing = pre_level + 0.5 * step
        segment = np.concatenate([previous[width - reach:], line[:reach]])
        hits = np.nonzero((segment[:-1] - crossing) * (segment[1:] - crossing)
                          <= 0)[0]
        if not len(hits):
            continue
        j = int(hits[0])
        y0, y1 = segment[j], segment[j + 1]
        if y1 == y0:
            continue
        crossings[k] = j - reach + float((crossing - y0) / (y1 - y0))
        errors[k] = float(np.std(porch)) / abs(y1 - y0)
    return crossings, errors


def sync_positions(decode):
    """The luma sync fall's per-line position residual against the field's
    median crossing, in fractions of a line (positive = late), with its
    error - the coarse pilot; its wander against the burst-locked grid is
    the luma path's sync-to-burst timing, reported, not fed back."""
    width = decode.fields.shape[2]
    positions, errors = [], []
    for field in decode.fields:
        crossings, se = line_crossings(field)
        ok = np.isfinite(crossings)
        median = np.nanmedian(crossings[ok]) if ok.any() else np.nan
        positions.append((crossings - median) / width)
        errors.append(se / width)
    return np.array(positions), np.array(errors)


def time_residual(decode):
    """Per field per line: the resampler's remaining position residual on
    the corrected output from the burst (fine pilot) and its error; the
    sync fall's residual rides along as the luma alignment residual."""
    burst = burst_positions(decode)
    sync_position, sync_error = sync_positions(decode)
    if burst is None:
        return sync_position, sync_error, sync_position, sync_error
    position, error = burst
    return position, error, sync_position, sync_error


def time_closure(position, error, sync_position, sync_error):
    """The time component's CLOSURE residual: what two independent gauges
    of one quantity fail to agree about.

    One unknown per line - where the line sits - and three gauges of it:
    the burst's phase, the sync fall's crossing on the output, and the
    carrier's sync-edge crossing on the RF. Three measurements of one
    unknown leave two closure relations, exactly the over-determination
    the frequency side has, and for the same reason: redundant
    measurements of a shared quantity. If line PLACEMENT were the only
    error all three would agree; the part that fails to cancel cannot be
    a placement error and no resampling removes it. (Measured, that part
    is the sync-to-burst disagreement: correlation -0.638, 0.367 samples
    rms, a fixed per-line pattern concentrated at the head switch.)

    Read per pass the way the frequency closure is read: a working pass
    shrinks the COMMON part while the disagreement stays put, so this
    chi-square RISES against a falling residual - and when it stops
    rising, the placement-explicable part is gone and what is left is
    not a time base error."""
    disagreement = np.asarray(position) - np.asarray(sync_position)
    combined = np.sqrt(np.asarray(error) ** 2 + np.asarray(sync_error) ** 2)
    ok = np.isfinite(disagreement) & np.isfinite(combined) & (combined > 0)
    if not ok.any():
        return {"closure_rms": np.nan, "closure_chi2": np.nan, "lines": 0}
    return {"closure_rms": float(np.sqrt(np.mean(disagreement[ok] ** 2))),
            "closure_chi2": float(np.mean((disagreement[ok]
                                           / combined[ok]) ** 2)),
            "closure_lines": int(ok.sum())}


def done_at_floor(metric):
    """The time base has converged when its residual is inside the gauge's
    own error - the point after which the frequency step runs on the rise
    alone."""
    return bool(np.isfinite(metric.get("chi2", np.nan))
                and metric["chi2"] <= 1.0)


def time_metric(position, error, sync_position=None, sync_error=None):
    ok = np.isfinite(position) & np.isfinite(error) & (error > 0)
    out = {"position_rms": np.nan, "chi2": np.nan, "lines": 0}
    if ok.any():
        out = {"position_rms": float(np.sqrt(np.mean(position[ok] ** 2))),
               "chi2": float(np.mean((position[ok] / error[ok]) ** 2)),
               "lines": int(ok.sum())}
    if sync_position is not None:
        ok = np.isfinite(sync_position) & np.isfinite(sync_error) & (sync_error > 0)
        if ok.any():
            out["sync_position_rms"] = float(np.sqrt(np.mean(sync_position[ok] ** 2)))
            out["sync_chi2"] = float(np.mean((sync_position[ok] / sync_error[ok]) ** 2))
        out.update(time_closure(position, error, sync_position, sync_error))
    return out


def drum_window():
    """Lines per flutter-band smoothing window, frame_lines / (2 pi) - the
    scale the time base's own refinement uses. A time base is mechanical:
    the drum turns once per field and the tape's speed cannot change on a
    line-to-line scale, so content faster than this band is measurement
    noise, and writing it into the line positions turns that noise into
    real timing jitter in the picture."""
    return max(int(round(float(ssr.sys_params["frame_lines"])
                         / (2.0 * math.pi))), 1)


def evidence_weight(residual, error):
    """How much of each line's measured residual is evidence rather than
    the gauge's own noise: the garrote 1 - error^2/residual^2, clipped to
    [0, 1].

    THE REGULATOR IS EVIDENCE, NOT BANDWIDTH. An earlier version band-
    limited the correction on the argument that a tape's speed cannot
    change from one line to the next. Ethan corrected it: the speed
    cannot, but the PLACEMENT can, because the sync time base positions
    every line from its own per-line measurement and that measurement has
    per-line error. A residual carrying line-to-line structure may
    therefore be a real placement error waiting to be corrected, and a
    band limit forbids exactly the correction the loop exists to make.

    What must not be written is the GAUGE's noise, which is a different
    question and has a different answer: a line whose residual is within
    its own error contributes nothing, a line measured at three sigma
    contributes 89% of what it measured, and no line is silenced for
    being fast."""
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = 1.0 - (np.asarray(error) ** 2) / (np.asarray(residual) ** 2)
    return np.clip(np.nan_to_num(keep, nan=0.0), 0.0, 1.0)


def band_limit(values, window):
    """Gap-aware zero-phase moving average over `window` lines, NaN where
    nothing was seen. RETAINED for the diagnostic split only - see
    `evidence_weight` for why it is no longer the regulator."""
    good = np.isfinite(values)
    kernel = np.ones(window, dtype=np.float64)
    evidence = np.convolve(good.astype(np.float64), kernel, mode="same")
    total = np.convolve(np.where(good, values, 0.0), kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(evidence > 0, total / np.maximum(evidence, 1.0),
                        np.nan)


def write_time_response(path, decode, position, error, alpha, pass_index,
                        accumulated):
    """The consumer's contract (--time_base_response), TOTAL form.

    THE CONSUMER IS STATELESS: every decode starts from the same
    sync-derived line locations and applies exactly what this file says,
    remembering nothing of any earlier pass. So the file must carry the
    TOTAL displacement wanted from that base, never this pass's share of
    it - a writer that sends a departure while the reader applies a total
    under-applies by exactly the accumulated part, and the correction
    never accumulates at all.

    `accumulated` is the loop's model for this component, keyed by the
    field's readloc: it gains `-alpha` times each pass's measured residual
    (the residual is the line sitting late, so cancelling it moves the
    line earlier) and is written out whole. `position_standard_error`
    carries the latest pass's burst-fit error, and `semantics` declares
    the reading so the consumer can refuse anything else rather than
    either side having to remember."""
    exports = residual_exports(decode)
    records, errors, readlocs = [], [], []
    for index, readloc in enumerate(decode.readlocs):
        path_field = exports.get(int(readloc))
        if path_field is None:
            continue
        with np.load(path_field, allow_pickle=False) as z:
            line_count = len(z["linelocs"])
            offset = int(z["lineoffset"]) + 1
        residual, se = position[index], error[index]
        measured = np.isfinite(residual) & np.isfinite(se) & (se > 0)
        total = accumulated.get(int(readloc))
        if total is None or len(total) != line_count:
            total = np.zeros(line_count)
        record_se = np.full(line_count, np.inf)
        span = min(len(residual), line_count - offset)
        m = measured[:span]
        # accumulate this pass's share into the running total
        # each line contributes as far as its own evidence goes
        keep = evidence_weight(residual[:span], se[:span])
        total[offset:offset + span] = np.where(
            m, total[offset:offset + span]
            - alpha * keep * np.nan_to_num(residual[:span]),
            total[offset:offset + span])
        record_se[offset:offset + span] = np.where(m, se[:span], np.inf)
        accumulated[int(readloc)] = total
        # a line the gauge has never reached carries no claim
        records.append(np.where(total != 0.0, total, np.nan))
        errors.append(record_se)
        readlocs.append(int(readloc))

    def stack(rows, fill):
        width = max(len(r) for r in rows)
        out = np.full((len(rows), width), fill)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
        return out

    np.savez(path, readloc=np.asarray(readlocs, dtype=np.int64),
             position_increment=stack(records, np.nan),
             position_standard_error=stack(errors, np.inf),
             semantics="total",
             # regulated by evidence per line, not by bandwidth: the
             # consumer's above-band warning is expected to fire where a
             # real per-line PLACEMENT error is being corrected
             regulator="per-line evidence garrote",
             band_limited_lines=0,
             pass_index=np.full(len(readlocs), int(pass_index)))
    return path


# --------------------------------------------------------------------------
# the record clip: a perfectly flat sync tip
# --------------------------------------------------------------------------

# the tip carries the same noise as the porch when nothing clipped it; per
# line the ratio of their standard deviations scatters within roughly
# 0.75-1.3 (an F ratio on a few dozen samples each), so a MEDIAN ratio
# below 0.75 over thousands of lines is a tip that was clipped flat
CLIP_RATIO = 0.75


def tip_flatness(decode):
    """Median over lines of (sync tip std / front porch std) on the
    corrected output, and whether that says the tip was clipped."""
    porch_lo, porch_hi, tip_lo, tip_hi = _tbc_windows()
    ratios = []
    for field in decode.fields:
        width = field.shape[1]
        for k in range(1, len(field)):
            porch = field[k - 1][width + porch_lo:width + porch_hi]
            tip = field[k][tip_lo:tip_hi]
            if len(tip) < 8 or len(porch) < 8:
                continue
            if np.median(tip) - np.median(porch) > -MINIMUM_STEP_IRE:
                continue
            ratios.append(np.std(tip) / max(np.std(porch), 1e-9))
    if not ratios:
        return np.nan, False
    ratio = float(np.median(ratios))
    return ratio, ratio < CLIP_RATIO


# --------------------------------------------------------------------------
# the heterodyne gauge: the frequency correction the burst lock absorbed
# --------------------------------------------------------------------------

HETERODYNE_COMPONENT = "colour-under heterodyne frequency"


def heterodyne_correction(decode):
    """The up-conversion's frequency error, read from what the burst lock
    had to MOVE rather than from any residual phase.

    The decoder's burst lock nulls the burst's phase by shifting line
    positions, so a heterodyne frequency error never survives as a phase
    error in the output - it is absorbed into the time base. Measuring
    the absolute burst phase therefore gives exactly zero slope, which is
    the answer rather than a failure. What the lock had to move IS the
    error: a per-line period deviation d corresponds to a frequency
    offset of f_sc x d.

    It separates as the recording does: a LINE-LOCKED part, a fixed
    pattern per line number that repeats field to field, and a WOBBLING
    part that does not. Measured, the line-locked part peaks at exactly
    half a cycle per line - alternating every two lines - which is the
    comb filter's own period."""
    rows = []
    for readloc in decode.readlocs:
        path = residual_exports(decode).get(int(readloc))
        if path is None:
            continue
        with np.load(path, allow_pickle=False) as z:
            if "time_per_line_complex" not in z.files:
                return None
            rows.append(np.asarray(z["time_per_line_complex"]).imag)
    if not rows:
        return None
    width = min(len(r) for r in rows)
    fsc = float(ssr.sys_params["fsc_mhz"]) * 1e6
    displacement = fsc * np.array([r[:width] for r in rows])
    locked = np.nanmean(displacement, axis=0)
    varying = displacement - locked
    return {"per_line_hz": displacement, "line_locked_hz": locked,
            "varying_hz": varying}


def heterodyne_metric(measured):
    if measured is None:
        return {}
    locked, varying = measured["line_locked_hz"], measured["varying_hz"]
    out = {"heterodyne_line_locked_rms": float(np.nanstd(locked)),
           "heterodyne_varying_rms": float(np.nanstd(varying))}
    rows = measured["per_line_hz"]
    repeat = []
    for i in range(len(rows)):
        others = np.nanmean(np.delete(rows, i, axis=0), axis=0)
        ok = np.isfinite(rows[i]) & np.isfinite(others)
        if ok.sum() > 50:
            repeat.append(float(np.corrcoef(rows[i][ok], others[ok])[0, 1]))
    if repeat:
        out["heterodyne_repeatability"] = float(np.median(repeat))
    good = locked[np.isfinite(locked)]
    if len(good) > 32:
        n = 512
        power = np.abs(np.fft.rfft((good - good.mean()) * np.hanning(len(good)),
                                   n=n)) ** 2
        power = power / power.mean()
        peak = int(np.argmax(power[3:])) + 3
        out["heterodyne_pattern_cycles_per_line"] = float(np.fft.rfftfreq(n)[peak])
        out["heterodyne_pattern_strength"] = float(power[peak])
    return out


# --------------------------------------------------------------------------
# the burst-to-sync phase relationship: the exact phase lock
# --------------------------------------------------------------------------

PHASE_LOCK_COMPONENT = "burst-to-sync phase lock"


def subcarrier_cycles_per_line():
    """The specification's own relationship: how many subcarrier cycles fill
    one line. In NTSC this is 227.5, so the burst's phase ALTERNATES by
    half a cycle every line, and that half cycle is a fact of the standard
    rather than anything measured."""
    return (float(ssr.sys_params["fsc_mhz"]) * 1e6
            * float(ssr.sys_params["line_period"]) * 1e-6)


def chroma_fields(decode):
    """The decode's chroma output in IRE, on the same time base and the
    same line origins as the luma - so a phase read there at a fixed
    offset from the line's start is referred to the SYNC pulse, which is
    a luma feature. The luma output has the colour notched out of it, so
    the burst cannot be read from the picture channel at all."""
    path = decode.prefix + "_chroma.tbc"
    if not os.path.exists(path):
        return None
    fields, lines, width = decode.fields.shape
    samples = np.fromfile(path, dtype="<u2")
    count = min(fields, samples.size // (lines * width))
    if count < 1:
        return None
    parameters = decode.json["videoParameters"]
    units_per_ire = (parameters["white16bIre"]
                     - parameters["black16bIre"]) / 100.0
    return (samples[: count * lines * width].reshape(count, lines, width)
            .astype(np.float64) / units_per_ire)


def burst_sync_phase(decode):
    """The colour burst's phase referred to the SYNC PULSE, measured from
    the waveform - the one absolute phase reference the signal carries.

    The output line begins at the sync fall, so a burst phase read at a
    fixed offset from the line's origin IS the burst-to-sync relationship.
    The standard fixes it exactly: the subcarrier sits at a half-integer
    multiple of the line rate, so the burst's phase must advance by
    precisely half a cycle every line. Everything departing from that
    alternation is error, and it is absolute rather than relative to any
    lock's arbitrary reference.

    Reading it from the lock's own displacement cannot work: that array's
    endpoints are pinned, which annihilates a constant frequency offset,
    exactly as the magnitude relation annihilates a linear tilt.

    Across the vertical interval there is NO burst - the equalizing and
    serration lines carry none - so the relationship must be carried by
    the sync pulses alone for nine lines and checked against the first
    burst afterwards. That check is the long-baseline drift."""
    rate = 4.0 * float(ssr.sys_params["fsc_mhz"]) * 1e6
    burst_us = ssr.sys_params["colorBurstUS"]
    lo = int(float(burst_us[0]) * 1e-6 * rate)
    hi = int(float(burst_us[1]) * 1e-6 * rate)
    if hi - lo < BURST_MINIMUM_SAMPLES:
        return None
    cycles_per_line = subcarrier_cycles_per_line()
    alternation = cycles_per_line - np.floor(cycles_per_line)
    # sampling is exactly four times the subcarrier, so the reference
    # phasor advances by a quarter turn per sample - no fitting needed
    n = np.arange(lo, hi)
    reference = np.exp(-2j * np.pi * n / 4.0)
    chroma = chroma_fields(decode)
    if chroma is None:
        return None
    rows, amplitudes = [], []
    for field in chroma:
        if field.shape[0] <= VERTICAL_SETTLE_LINE:
            continue
        phasor = field[:, lo:hi] @ reference
        rows.append(np.angle(phasor))
        amplitudes.append(np.abs(phasor) / max(len(n), 1))
    if not rows:
        return None
    width = min(len(r) for r in rows)
    phase = np.array([r[:width] for r in rows])
    amplitude = np.array([a[:width] for a in amplitudes])
    lines = np.arange(width)
    # remove the standard's own half-cycle alternation; what is left is
    # the departure from the specified relationship
    expected = 2.0 * np.pi * alternation * lines
    residual = np.unwrap(phase - expected[None, :], axis=1)
    residual = residual - np.median(residual[:, VERTICAL_SETTLE_LINE:], axis=1,
                                    keepdims=True)
    # the burst is present only outside the vertical interval
    present = amplitude > (BURST_PRESENCE_FRACTION
                           * np.median(amplitude[:, VERTICAL_SETTLE_LINE:]))
    slopes, wanders = [], []
    for row, ok in zip(residual, present):
        use = ok & (lines >= VERTICAL_SETTLE_LINE)
        if use.sum() < 32:
            continue
        fit = np.polyfit(lines[use], row[use], 1)
        slopes.append(float(fit[0]))
        wanders.append(float(np.std(row[use] - np.polyval(fit, lines[use]))))
    # the carry ACROSS the vertical interval: the relationship held by the
    # sync pulses alone for nine lines, checked at the first burst after
    carried = []
    for row, ok in zip(residual, present):
        after = np.nonzero(ok & (lines >= VERTICAL_SETTLE_LINE))[0]
        if len(after) < 8:
            continue
        settled = np.median(row[after[4:]]) if len(after) > 8 else np.nan
        carried.append(float(row[after[0]] - settled))
    return {"phase": phase, "residual_radians": residual,
            "amplitude": amplitude, "burst_present": present,
            "alternation_cycles": float(alternation),
            "cycles_per_line": cycles_per_line,
            "slope_radians_per_line": np.array(slopes),
            "wander_radians": np.array(wanders),
            "vertical_carry_radians": np.array(carried)}


def phase_lock_metric(measured):
    if measured is None:
        return {}
    line_s = float(ssr.sys_params["line_period"]) * 1e-6
    out = {"phase_lock_alternation_cycles": measured["alternation_cycles"]}
    slope = measured["slope_radians_per_line"]
    if len(slope):
        # radians per line is a frequency offset once divided by the line
        out["phase_lock_offset_hz"] = float(np.mean(slope) / (2 * np.pi * line_s))
        out["phase_lock_offset_spread_hz"] = float(
            np.std(slope) / (2 * np.pi * line_s))
        out["phase_lock_offset_error_hz"] = out["phase_lock_offset_spread_hz"] / max(
            np.sqrt(len(slope)), 1.0)
    wander = measured["wander_radians"]
    if len(wander):
        out["phase_lock_wander_radians"] = float(np.mean(wander))
        out["phase_lock_wander_degrees"] = float(np.degrees(np.mean(wander)))
    carry = measured["vertical_carry_radians"]
    carry = carry[np.isfinite(carry)]
    if len(carry):
        out["phase_lock_vertical_carry_degrees"] = float(
            np.degrees(np.mean(carry)))
        out["phase_lock_vertical_carry_rms_degrees"] = float(
            np.degrees(np.std(carry)))
        out["phase_lock_vertical_carry_sigma"] = abs(np.mean(carry)) / max(
            np.std(carry) / np.sqrt(len(carry)), 1e-12)
    present = measured["burst_present"]
    out["phase_lock_burstless_lines"] = float(
        np.mean(np.sum(~present, axis=1)))
    return out


# --------------------------------------------------------------------------
# genlock: the two locks differentiated together
# --------------------------------------------------------------------------

GENLOCK_COMPONENT = "genlock between the burst and hsync locks"


def genlock(decode):
    """Whether the SOURCE was genlocked, from the two locks together.

    A genlocked source derives its subcarrier and its line rate from ONE
    oscillator, so once the hsync is aligned the burst is aligned too and
    the burst lock has NOTHING left to do. What the lock had to ADD is
    therefore the genlock error itself.

    It cannot be read from the output burst phase. The lock has already
    nulled that by construction, so measuring it returns the lock's own
    success and not the source's behaviour - the same null space the
    heterodyne measurement meets. The witness is the correction applied,
    not the error remaining.

    And the decisive question about that correction is not its size but
    its CHARACTER. A subcarrier running on its own oscillator drifts
    freely, so the phase error it forces the lock to absorb accumulates
    as a RANDOM WALK. A genlocked one is tied to the line rate, so its
    error is stationary - it wanders but is pulled back. That is a unit
    root test, and it distinguishes the two where a threshold on
    magnitude cannot."""
    fsc = float(ssr.sys_params["fsc_mhz"]) * 1e6
    line_us = float(ssr.sys_params["line_period"])
    cycles = subcarrier_cycles_per_line()
    added, hsync, parities = [], [], []
    for readloc in decode.readlocs:
        path = residual_exports(decode).get(int(readloc))
        if path is None:
            continue
        with np.load(path, allow_pickle=False) as z:
            if "time_per_line_complex" not in z.files:
                return None
            complex_time = np.asarray(z["time_per_line_complex"])
            added.append(np.nan_to_num(complex_time.imag))
            hsync.append(np.nan_to_num(complex_time.real))
            parities.append(bool(z["is_first_field"])
                            if "is_first_field" in z.files else True)
    if not added:
        return None
    width = min(len(r) for r in added)
    # what the lock had to add, per line, in CYCLES of subcarrier
    correction = np.array([r[:width] for r in added]) * cycles
    timing = np.array([r[:width] for r in hsync]) * cycles
    # the accumulated phase the lock absorbed, along each field
    accumulated = np.cumsum(correction, axis=1)
    return {"correction_cycles": correction,
            "hsync_cycles": timing,
            "accumulated_cycles": accumulated,
            "parities": np.array(parities[:len(correction)]),
            "cycles_per_line": cycles,
            "subcarrier_hz": fsc,
            "line_us": line_us}


def genlock_metric(measured):
    if measured is None:
        return {}
    from vhsdecode.models import head_model

    correction = measured["correction_cycles"]
    accumulated = measured["accumulated_cycles"]
    out = {
        "genlock_correction_rms_cycles": float(np.nanstd(correction)),
        "genlock_correction_rms_degrees": float(np.nanstd(correction) * 360.0),
        "genlock_hsync_rms_cycles": float(np.nanstd(measured["hsync_cycles"])),
    }
    # the character of the accumulated phase, field by field: a free-running
    # oscillator walks, a genlocked one is pulled back
    walks, statistics = [], []
    for row in accumulated:
        verdict = head_model.unit_root(row)
        if np.isfinite(verdict.get("unit_root_t", np.nan)):
            walks.append(verdict["random_walk_risk"])
            statistics.append(verdict["unit_root_t"])
    if walks:
        out["genlock_random_walk_share"] = float(np.mean(walks))
        out["genlock_unit_root_t"] = float(np.median(statistics))
        out["genlock_unit_root_critical"] = float(
            head_model.unit_root_critical(accumulated.shape[1]))
        # stationary means genlocked: the walk is rejected
        out["genlocked"] = float(np.mean(walks) < 0.5)
    # the correction split into what BOTH heads share - the VCR's own
    # parameter mismatch - and what alternates with head parity, which is
    # the individual heads differing
    parities = measured["parities"]
    if parities.size == correction.shape[0] and len(set(parities.tolist())) == 2:
        first = np.nanmean(correction[parities], axis=0)
        second = np.nanmean(correction[~parities], axis=0)
        common = 0.5 * (first + second)
        split = 0.5 * (first - second)
        out["genlock_vcr_mismatch_cycles"] = float(np.nanstd(common))
        out["genlock_head_difference_cycles"] = float(np.nanstd(split))
        total = float(np.hypot(np.nanstd(common), np.nanstd(split)))
        if total > 0:
            out["genlock_vcr_share"] = float(np.nanstd(common) / total)
            out["genlock_head_share"] = float(np.nanstd(split) / total)
    return out


# --------------------------------------------------------------------------
# the vertical interval: the LOW-FREQUENCY response, as a constant
# --------------------------------------------------------------------------

VERTICAL_COMPONENT = "vertical interval low-frequency response"
# the vertical sync block, in output lines from the top of the field
VSYNC_LINES = (3, 4, 5)
EQUALIZING_BEFORE = (0, 1, 2)
EQUALIZING_AFTER = (6, 7, 8)
VERTICAL_SETTLE_LINE = 9          # the first line carrying one sync pulse
VERTICAL_RECOVERY_LINES = 51
VERTICAL_SETTLED_LINES = 15       # the tail taken as settled
VERTICAL_FIT_LINES = 25           # the span the decay is fitted over
VERTICAL_PROFILE_BINS = 11
VERTICAL_MINIMUM_POPULATION = 64
BACK_PORCH_GUARD = 0.25           # of the porch's width, at each end
BURST_MINIMUM_SAMPLES = 16
BURST_PRESENCE_FRACTION = 0.35    # of the median burst amplitude
SYNC_LEVEL_IRE = -25.0            # below this is sync
BLANKING_FLOOR_IRE = -10.0        # above this is blanking


def _back_porch_window(rate_hz):
    """The back porch, from the spec: after the colour burst ends, before
    active video begins. The line starts at the sync fall, so these are
    times from the line's own origin."""
    start_us = float(ssr.sys_params["colorBurstUS"][1])
    end_us = float(ssr.sys_params["activeVideoUS"][0])
    guard = BACK_PORCH_GUARD * (end_us - start_us)
    return (int((start_us + guard) * 1e-6 * rate_hz),
            int((end_us - guard) * 1e-6 * rate_hz))


def vertical_low_frequency(decode):
    """The low-frequency response, from the vertical interval.

    A line is 63.5 microseconds, so nothing measured within one reaches
    below about 16 kHz. The vertical sync block is a single excursion of
    roughly 190 microseconds - three lines at sync level with serrations -
    and it charges any low-frequency roll-off in the path. Three
    instruments read that charge, weakest to strongest:

      droop     the level's tilt WITHIN the block. Weak: the serrations
                dilute it and the block is short.
      step      the blanking level of the equalizing pulses AFTER the
                block against those BEFORE it. A real roll-off must leave
                the two at different levels; a gain change would not move
                blanking and sync tip together.
      recovery  the back porch settling back over the following lines.
                Strongest, and the only one that reads the time constant
                DIRECTLY, with no assumption about the block's depth.

    The recovery sets the reported corner. It is a CONSTANT of the path,
    measured once and applied without varying in time."""
    rate = 4.0 * float(ssr.sys_params["fsc_mhz"]) * 1e6
    line_us = float(ssr.sys_params["line_period"])
    porch = _back_porch_window(rate)
    fields = [f for f in decode.fields if f.shape[0] > VERTICAL_SETTLE_LINE]
    if not fields:
        return None
    out = {}

    # -- the droop within the block ---------------------------------------
    times, levels = [], []
    for field in fields:
        block = np.concatenate([field[k] for k in VSYNC_LINES])
        deep = block < SYNC_LEVEL_IRE
        if deep.sum() < VERTICAL_MINIMUM_POPULATION:
            continue
        index = np.nonzero(deep)[0]
        times.append(index / rate * 1e6)
        levels.append(block[index])
    if times:
        t = np.concatenate(times)
        v = np.concatenate(levels)
        edges = np.linspace(t.min(), t.max(), VERTICAL_PROFILE_BINS + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        profile = np.array([
            np.median(v[(t >= edges[i]) & (t < edges[i + 1])])
            if ((t >= edges[i]) & (t < edges[i + 1])).any() else np.nan
            for i in range(VERTICAL_PROFILE_BINS)])
        ok = np.isfinite(profile)
        if ok.sum() >= 4:
            slope = float(np.polyfit(centres[ok], profile[ok], 1)[0])
            out.update({"block_span_us": float(t.max() - t.min()),
                        "block_profile": profile, "block_times_us": centres,
                        "block_droop_ire": slope * float(t.max() - t.min()),
                        "block_depth_ire": float(abs(np.nanmedian(profile)))})

    # -- the step across the block ----------------------------------------
    before, after, tip_before, tip_after = [], [], [], []
    for field in fields:
        pre = np.concatenate([field[k] for k in EQUALIZING_BEFORE])
        post = np.concatenate([field[k] for k in EQUALIZING_AFTER])
        for block, top, tip in ((pre, before, tip_before),
                                (post, after, tip_after)):
            high = block[block > BLANKING_FLOOR_IRE]
            low = block[block < SYNC_LEVEL_IRE]
            if len(high) > VERTICAL_MINIMUM_POPULATION:
                top.append(float(np.median(high)))
            if len(low) > VERTICAL_MINIMUM_POPULATION:
                tip.append(float(np.median(low)))
    if before and after:
        step = float(np.mean(after) - np.mean(before))
        error = float(np.hypot(np.std(before) / np.sqrt(len(before)),
                               np.std(after) / np.sqrt(len(after))))
        out.update({"step_ire": step, "step_error_ire": error,
                    "step_sigma": abs(step) / max(error, 1e-12)})
        if tip_before and tip_after:
            out["tip_step_ire"] = float(np.mean(tip_after) - np.mean(tip_before))

    # -- the recovery, the instrument that reads the time constant --------
    lines = range(VERTICAL_SETTLE_LINE, VERTICAL_SETTLE_LINE + VERTICAL_RECOVERY_LINES)
    recovery = []
    for k in lines:
        values = [float(np.median(f[k][porch[0]:porch[1]]))
                  for f in fields if f.shape[0] > k]
        recovery.append(np.mean(values) if values else np.nan)
    recovery = np.array(recovery)
    tail = recovery[-VERTICAL_SETTLED_LINES:]
    settled = float(np.nanmean(tail))
    scatter = float(np.nanstd(tail) / np.sqrt(max(np.isfinite(tail).sum(), 1)))
    excess = float(recovery[0] - settled)
    out.update({"recovery": recovery, "settled_ire": settled,
                "excess_ire": excess, "settled_error_ire": scatter,
                "excess_sigma": abs(excess) / max(scatter, 1e-12)})
    t = np.arange(len(recovery)) * line_us
    deviation = recovery - settled
    fit = np.isfinite(deviation) & (t < VERTICAL_FIT_LINES * line_us)
    corner = float("nan")
    tau = float("nan")
    if fit.sum() > 6 and np.isfinite(excess) and excess != 0:
        signed = deviation[fit] * np.sign(excess)
        good = signed > 0
        if good.sum() > 6:
            slope = float(np.polyfit(t[fit][good], np.log(signed[good]), 1)[0])
            if slope < 0:
                tau = -1.0 / slope
                corner = 1e6 / (2.0 * np.pi * tau)
    out.update({"tau_us": tau, "corner_hz": corner})
    return out


def vertical_metric(measured):
    if measured is None:
        return {}
    out = {"vertical_corner_hz": measured.get("corner_hz", float("nan")),
           "vertical_tau_us": measured.get("tau_us", float("nan")),
           "vertical_excess_ire": measured.get("excess_ire", float("nan")),
           "vertical_excess_sigma": measured.get("excess_sigma", float("nan"))}
    for key in ("step_ire", "step_sigma", "tip_step_ire", "block_droop_ire"):
        if key in measured:
            out["vertical_" + key] = measured[key]
    return out


# --------------------------------------------------------------------------
# the amplitude gauge: the constant-envelope residual
# --------------------------------------------------------------------------

AMPLITUDE_COMPONENT = "constant-envelope magnitude"
# a carrier that moved more than this within a sample was sweeping, and the
# envelope then reports the path's transient rather than its response
AMPLITUDE_SLEW_BINS = 1.0
# a bin the carrier barely visited describes the tail of the picture's
# excursion, not the channel
AMPLITUDE_MINIMUM_POPULATION = 64


def amplitude_residual(decode, bin_ire=1.0):
    """The constant-envelope residual, on both the axes it lives on.

    The luma carrier is recorded at constant amplitude, so every envelope
    variation is the path's magnitude at the instantaneous carrier. Two
    quantities come out of that and the loop needs both:

      per CARRIER BIN   the response residual - what the envelope says
                        |H| is, against the flat ideal. This determines
                        the part of the channel the sync edges CANNOT
                        see: the frequency design annihilates any linear
                        function of frequency, and the envelope measures
                        that function directly rather than through a
                        sideband relation.
      per LINE          the amplitude residual d - what is left when the
                        response is divided out. This is what couples to
                        the other components: it converts to phase
                        through the band-pass's asymmetry about the
                        carrier, and its drum-rate part is the time
                        base's own witness.

    Returns (centres, response, response_error, per_line, per_line_error)
    with the response in nepers and the per-line residual dimensionless.
    """
    exports = residual_exports(decode)
    hz_ire = float(decode.filters["hz_ire"])
    bin_hz = float(bin_ire) * hz_ire
    sums, per_line, per_line_error = {}, [], []
    for readloc in decode.readlocs:
        path = exports.get(int(readloc))
        if path is None:
            continue
        with np.load(path, allow_pickle=False) as z:
            if "amplitude" not in z.files or "carrier" not in z.files:
                continue
            amplitude = np.asarray(z["amplitude"], dtype=np.float64)
            carrier = np.asarray(z["carrier"], dtype=np.float64)
        good = np.isfinite(amplitude) & (amplitude > 0) & np.isfinite(carrier) \
            & (carrier > 0)
        slew = np.abs(np.diff(carrier, axis=1, prepend=carrier[:, :1]))
        good &= slew < AMPLITUDE_SLEW_BINS * bin_hz
        if not good.any():
            continue
        log_amplitude = np.log(amplitude)
        index = np.floor(carrier / bin_hz).astype(np.int64)
        for key in np.unique(index[good]):
            select = good & (index == key)
            values = log_amplitude[select]
            entry = sums.setdefault(int(key), [0.0, 0.0, 0])
            entry[0] += float(values.sum())
            entry[1] += float((values ** 2).sum())
            entry[2] += int(values.size)
        per_line.append((log_amplitude, carrier, good, index))
    if not sums:
        return None
    keys = np.array([k for k in sorted(sums)
                     if sums[k][2] >= AMPLITUDE_MINIMUM_POPULATION],
                    dtype=np.int64)
    if not len(keys):
        return None
    centres = (keys + 0.5) * bin_hz
    population = np.array([sums[k][2] for k in keys], dtype=np.float64)
    mean = np.array([sums[k][0] for k in keys]) / population
    variance = np.maximum(np.array([sums[k][1] for k in keys]) / population
                          - mean ** 2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.sqrt(np.where(population > 1,
                                 variance / np.maximum(population - 1, 1),
                                 np.inf))
    # the response, referenced to its own weighted mean: an overall gain is
    # nothing to a limiter, only the shape is the channel
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(np.isfinite(error) & (error > 0), 1.0 / error ** 2, 0.0)
    response = mean - (np.sum(weight * mean) / max(np.sum(weight), 1e-30))
    # what is left per line once that response is divided out
    lookup = {int(k): response[i] for i, k in enumerate(keys)}
    rows, row_errors = [], []
    for log_amplitude, carrier, good, index in per_line:
        modelled = np.vectorize(lambda k: lookup.get(int(k), np.nan))(index)
        left = log_amplitude - np.nanmean(log_amplitude[good]) - modelled
        for line in range(left.shape[0]):
            ok = good[line] & np.isfinite(left[line])
            if ok.sum() < 8:
                rows.append(np.nan)
                row_errors.append(np.inf)
                continue
            rows.append(float(np.mean(left[line][ok])))
            row_errors.append(float(np.std(left[line][ok])
                                    / np.sqrt(ok.sum())))
    return (centres, response, error, np.array(rows), np.array(row_errors))


def amplitude_metric(measured):
    if measured is None:
        return {"response_rms": np.nan, "per_line_rms": np.nan, "bins": 0}
    centres, response, error, rows, row_errors = measured
    ok = np.isfinite(response) & np.isfinite(error) & (error > 0)
    line_ok = np.isfinite(rows) & np.isfinite(row_errors) & (row_errors > 0)
    out = {"bins": int(ok.sum()), "lines": int(line_ok.sum())}
    if ok.any():
        out["response_rms"] = float(np.sqrt(np.mean(response[ok] ** 2)))
        out["response_chi2"] = float(np.mean((response[ok] / error[ok]) ** 2))
        out["band_mhz"] = (float(centres[ok].min() / 1e6),
                           float(centres[ok].max() / 1e6))
    if line_ok.any():
        out["per_line_rms"] = float(np.sqrt(np.mean(rows[line_ok] ** 2)))
        out["per_line_chi2"] = float(np.mean((rows[line_ok]
                                              / row_errors[line_ok]) ** 2))
    return out


# --------------------------------------------------------------------------
# the burst gauge: the demodulated burst's residual (the chroma pilot)
# --------------------------------------------------------------------------

BURST_COMPONENT = "chroma burst pilot"
# The burst is a gated sinusoid whose amplitude, frequency and phase the
# specification fixes, which makes it the same kind of pilot as the sync
# edge and gives all three axes at once on the UP-HETERODYNED chroma:
#   amplitude  its level against the specified level
#   frequency  the phase ramp WITHIN the burst, which is the heterodyne's
#              own frequency error, and the burst's envelope shape, which
#              is the chroma path's response to a known gate
#   time       its phase against the specified phase for that line
# NTSC: the burst is 40 IRE peak to peak, the same depth as sync, so its
# amplitude is half the sync depth.
BURST_AMPLITUDE_OVER_SYNC_DEPTH = 0.5


def burst_components(decode):
    """The up-heterodyned burst measured against its SPECIFIED shape, on
    all three axes, per line.

    Returns a dict of per-field-per-line arrays:
      log_amplitude       ln(measured / specified) - the amplitude axis
      phase               against the parity's own reference - the time axis
      heterodyne_hz       the frequency error read from the phase RAMP
                          across the burst: at four samples per cycle a
                          constant offset df turns the phase by
                          2 pi df / (4 fsc) per sample, so the ramp over
                          the window IS the up-conversion oscillator's
                          error, which no single phase measurement can see
      envelope            the mean burst envelope per head, against the
                          ideal gate - the chroma path's response
    """
    path = decode.prefix + "_chroma.tbc"
    if not os.path.exists(path):
        return None
    fields, lines, width = decode.fields.shape
    samples = np.fromfile(path, dtype="<u2")
    count = min(fields, samples.size // (lines * width))
    if not count:
        return None
    # the chroma is written on the same 16-bit scale as the luma, so it
    # must be brought to IRE before it is compared with a level the
    # specification states in IRE
    parameters = decode.json["videoParameters"]
    units_per_ire = (parameters["white16bIre"]
                     - parameters["black16bIre"]) / 100.0
    chroma = (samples[: count * lines * width].reshape(count, lines, width)
              .astype(np.float64) / units_per_ire)
    rate_mhz = 4.0 * float(ssr.sys_params["fsc_mhz"])
    start_us, end_us = ssr.sys_params["colorBurstUS"]
    lo, hi = int(round(start_us * rate_mhz)), int(round(end_us * rate_mhz))
    interior = slice(lo + 4, hi - 4)
    half = (interior.start + interior.stop) // 2
    n = np.arange(width)
    cos, sin = np.cos(np.pi * n / 2.0), np.sin(np.pi * n / 2.0)
    # the specified burst level in the same units the field is in (IRE)
    specified = abs(float(decode.filters["vsync_ire"])) \
        * BURST_AMPLITUDE_OVER_SYNC_DEPTH
    fsc_hz = float(ssr.sys_params["fsc_mhz"]) * 1e6

    def quadrature(field, window):
        block = field[:, window]
        i = 2.0 * np.mean(block * cos[window], axis=1)
        q = -2.0 * np.mean(block * sin[window], axis=1)
        return i, q

    log_amplitude = np.full((count, lines), np.nan)
    phase = np.full((count, lines), np.nan)
    heterodyne = np.full((count, lines), np.nan)
    envelopes = {True: [], False: []}
    for f in range(count):
        field = chroma[f] - np.median(chroma[f], axis=1, keepdims=True)
        i, q = quadrature(field, interior)
        amplitude = np.hypot(i, q)
        present = amplitude > 0.5 * np.median(amplitude[amplitude > 0]) \
            if (amplitude > 0).any() else np.zeros(lines, dtype=bool)
        if present.sum() < 8:
            continue
        # AMPLITUDE against the specified level, not the field's own median
        log_amplitude[f, present] = np.log(amplitude[present] / specified)
        line_phase = np.arctan2(q, i)
        for parity in (0, 1):
            rows = np.nonzero(present)[0]
            rows = rows[rows % 2 == parity]
            if rows.size < 4:
                continue
            reference = np.angle(np.mean(np.exp(1j * line_phase[rows])))
            phase[f, rows] = np.angle(np.exp(1j * (line_phase[rows] - reference)))
        # FREQUENCY: the phase ramp across the burst is the heterodyne's
        # own error, which a single phase measurement cannot see
        i0, q0 = quadrature(field, slice(interior.start, half))
        i1, q1 = quadrature(field, slice(half, interior.stop))
        turn = np.angle(np.exp(1j * (np.arctan2(q1, i1) - np.arctan2(q0, i0))))
        span = 0.5 * (interior.stop - interior.start)      # samples between
        heterodyne[f, present] = (turn[present] / (2.0 * np.pi)
                                  * (4.0 * fsc_hz) / max(span, 1.0))
        edge = field[:, lo - 8:hi + 8]
        envelope = np.hypot(edge[:, :-1], edge[:, 1:])
        strong = present & (np.abs(log_amplitude[f]) < 0.5)
        if strong.any():
            head = bool(decode.json["fields"][f].get("isFirstField", f % 2 == 0))
            envelopes[head].append(np.mean(envelope[strong], axis=0))
    return {"log_amplitude": log_amplitude, "phase": phase,
            "heterodyne_hz": heterodyne, "specified_ire": specified,
            "envelope": {("head_first" if h else "head_second"):
                         np.mean(np.array(v), axis=0)
                         for h, v in envelopes.items() if v}}


def continuous_burst(decode, carrier_hz=None, channel=None):
    """The burst area's CONTINUOUS amplitude and phase, sample by sample,
    at whichever carrier is asked for.

    A single quadrature average over the window gives one amplitude and
    one phase per line and hides everything that varies WITHIN the burst.
    The continuous form keeps it: the amplitude trace is the path's
    response to a known gate, and the phase trace's slope is its GROUP
    DELAY while its curvature is dispersion.

    `channel` selects where the burst is read. Given, it names a residual
    channel - "color_under" reads the burst BEFORE the heterodyne, at the
    colour-under carrier, where the same physical event is only about 1.6
    cycles long instead of 9 and there is correspondingly less amplitude
    to measure, but where the phase and time alignment is exact because
    it is the same samples. Omitted, the up-converted chroma output is
    read at the subcarrier. Measuring both and differencing isolates the
    up-conversion's own response, because it is one event at two
    carriers."""
    rate_hz = 4.0 * float(ssr.sys_params["fsc_mhz"]) * 1e6
    if carrier_hz is None:
        carrier_hz = float(ssr.sys_params["fsc_mhz"]) * 1e6
    start_us, end_us = ssr.sys_params["colorBurstUS"]
    lo = int(round(start_us * rate_hz / 1e6)) - 6
    hi = int(round(end_us * rate_hz / 1e6)) + 6
    if channel:
        rows, heads = [], []
        for index, readloc in enumerate(decode.readlocs):
            path = residual_exports(decode).get(int(readloc))
            if path is None:
                continue
            with np.load(path, allow_pickle=False) as z:
                if channel not in z.files:
                    return None
                rows.append(np.asarray(z[channel], dtype=np.float64))
                heads.append(bool(z["is_first_field"]))
        if not rows:
            return None
        fields = np.array(rows)
    else:
        path = decode.prefix + "_chroma.tbc"
        if not os.path.exists(path):
            return None
        n_fields, lines, width = decode.fields.shape
        raw = np.fromfile(path, dtype="<u2")
        count = min(n_fields, raw.size // (lines * width))
        parameters = decode.json["videoParameters"]
        units = (parameters["white16bIre"] - parameters["black16bIre"]) / 100.0
        fields = (raw[: count * lines * width].reshape(count, lines, width)
                  .astype(np.float64) / units)
        heads = [bool(decode.json["fields"][f].get("isFirstField", f % 2 == 0))
                 for f in range(count)]
    lo = max(lo, 0)
    hi = min(hi, fields.shape[2])
    window = np.arange(lo, hi)
    # mix down to complex baseband at the named carrier and low-pass by
    # averaging over one cycle, which is the shortest honest smoother
    mix = np.exp(-2j * np.pi * carrier_hz * window / rate_hz)
    cycle = max(int(round(rate_hz / carrier_hz)), 2)
    kernel = np.ones(cycle) / cycle
    per_head = {}
    for head_value, tag in ((True, "head_first"), (False, "head_second")):
        select = [i for i, h in enumerate(heads) if h == head_value]
        if not select:
            continue
        traces = []
        for index in select:
            field = fields[index][:, window]
            field = field - np.median(field, axis=1, keepdims=True)
            analytic = field * mix
            smoothed = np.apply_along_axis(
                lambda row: np.convolve(row, kernel, mode="same"), 1, analytic)
            amplitude = np.abs(smoothed)
            strong = amplitude.mean(axis=1) > 0.4 * np.median(
                amplitude.mean(axis=1))
            if not strong.any():
                continue
            # EVERY LINE IS REFERENCED TO ITS OWN PHASOR before the lines
            # are averaged. At four samples per cycle the subcarrier
            # advances half a cycle per line (227.5 cycles per line on
            # NTSC), so the burst appears inverted on alternate lines and
            # a straight complex average cancels it to nothing. Dividing
            # each line by its own mean phasor keeps the SHAPE - the
            # amplitude trace and the phase trajectory across the burst,
            # which is what group delay and response live in - and
            # discards only the per-line phase offset, which is the time
            # axis and is measured separately.
            interior = slice(len(window) // 4, 3 * len(window) // 4)
            reference = smoothed[strong][:, interior].mean(axis=1)
            reference = np.where(np.abs(reference) > 0, reference, 1.0)
            aligned = smoothed[strong] / (reference / np.abs(reference))[:, None]
            traces.append(aligned.mean(axis=0))
        if traces:
            mean = np.mean(np.array(traces), axis=0)
            phase = np.unwrap(np.angle(mean))
            per_head[tag] = {"offsets": window, "amplitude": np.abs(mean),
                             "phase": phase}
    return {"carrier_hz": carrier_hz, "rate_hz": rate_hz,
            "per_head": per_head} if per_head else None


def continuous_metric(measured, name):
    """The three axes read off a continuous burst trace.

    A phase SLOPE across the burst is a FREQUENCY OFFSET, not a group
    delay - the carrier turning steadily against the reference. Group
    delay is the derivative of phase with respect to FREQUENCY, so it
    needs the response across a band, which `burst_response` gets the
    same way the sync gauge does: ratio the measured envelope's transform
    against the ideal gate's. Both are reported; they are different
    quantities and conflating them cost a nonsense figure of hundreds of
    microseconds on a 63 microsecond line."""
    if measured is None:
        return {}
    out = {}
    rate = measured["rate_hz"]
    for tag, trace in measured["per_head"].items():
        amplitude, phase = trace["amplitude"], trace["phase"]
        n = len(amplitude)
        interior = slice(n // 4, 3 * n // 4)
        x = np.arange(n)[interior]
        if len(x) < 6:
            continue
        slope = float(np.polyfit(x, phase[interior], 1)[0])
        out[f"{name}_{tag}_offset_hz"] = slope * rate / (2.0 * np.pi)
        out[f"{name}_{tag}_chirp"] = float(np.polyfit(x, phase[interior], 2)[0])
        out[f"{name}_{tag}_peak"] = float(amplitude.max())
        flat = amplitude[interior]
        out[f"{name}_{tag}_flatness"] = float(flat.std() / max(flat.mean(), 1e-30))
    return out


def burst_response(measured, specified_ire=None):
    """The burst path's FREQUENCY RESPONSE and GROUP DELAY, by the same
    method the sync edge is measured with: the burst is a gate of known
    width and height, so the ratio of the measured envelope's transform
    to the ideal gate's is the path's response about the carrier, and the
    negative slope of its phase against frequency is the group delay.

    The ideal gate is placed at the measured trace's own 50% crossings,
    so a bulk delay is absorbed by the placement exactly as the sync
    gauge absorbs it - which means what is reported is DISPERSION, the
    part that is a channel property, and not the alignment."""
    if measured is None:
        return {}
    rate = measured["rate_hz"]
    out = {}
    for tag, trace in measured["per_head"].items():
        amplitude = np.asarray(trace["amplitude"], dtype=np.float64)
        n = len(amplitude)
        peak = amplitude.max()
        if peak <= 0 or n < 16:
            continue
        half = 0.5 * peak
        rising = np.nonzero(amplitude >= half)[0]
        if len(rising) < 4:
            continue
        first, last = int(rising[0]), int(rising[-1])
        ideal = np.zeros(n)
        ideal[first:last + 1] = peak if specified_ire is None \
            else 0.5 * specified_ire
        window = np.hanning(n)
        measured_f = np.fft.rfft(amplitude * window)
        ideal_f = np.fft.rfft(ideal * window)
        freqs = np.fft.rfftfreq(n, 1.0 / rate)
        good = np.abs(ideal_f) > 0.05 * np.abs(ideal_f).max()
        if good.sum() < 4:
            continue
        ratio = np.where(good, measured_f / np.where(good, ideal_f, 1.0), np.nan)
        band = good & (freqs > 0) & (freqs < 1.5e6)
        if band.sum() < 3:
            continue
        phase = np.unwrap(np.angle(ratio[band]))
        slope = float(np.polyfit(freqs[band], phase, 1)[0])
        out[f"{tag}_gate_width_us"] = float((last - first + 1) / rate * 1e6)
        out[f"{tag}_group_delay_us"] = float(-slope / (2.0 * np.pi) * 1e6)
        out[f"{tag}_response_rms"] = float(
            np.sqrt(np.mean(np.log(np.abs(ratio[band])) ** 2)))
        out[f"{tag}_band_khz"] = float(freqs[band].max() / 1e3)
    return out


def burst_component_metric(measured):
    if measured is None:
        return {}
    out = {}
    for key, name in (("log_amplitude", "amplitude"), ("phase", "time")):
        v = measured[key]
        ok = np.isfinite(v)
        if ok.any():
            out[f"burst_{name}_rms"] = float(np.sqrt(np.mean(v[ok] ** 2)))
            out[f"burst_{name}_mean"] = float(np.mean(v[ok]))
    het = measured["heterodyne_hz"]
    ok = np.isfinite(het)
    if ok.any():
        out["heterodyne_mean_hz"] = float(np.mean(het[ok]))
        out["heterodyne_rms_hz"] = float(np.std(het[ok]))
        out["burst_lines"] = int(ok.sum())
    return out


def burst_residual(decode):
    """The demodulated burst, per field per line, against its own field:
    ln amplitude relative to the field's median, and the phase residual
    against the median of the line's parity (NTSC alternates by a line) -
    the burst's alignment parameters - plus, per head, the mean burst
    envelope over the burst window (its SHAPE, the chroma path's step
    response). Reads the chroma output of the decode."""
    path = decode.prefix + "_chroma.tbc"
    if not os.path.exists(path):
        return None
    fields, lines, width = decode.fields.shape
    samples = np.fromfile(path, dtype="<u2")
    count = min(fields, samples.size // (lines * width))
    # the chroma is written on the same 16-bit scale as the luma, so it
    # must be brought to IRE before it is compared with a level the
    # specification states in IRE
    parameters = decode.json["videoParameters"]
    units_per_ire = (parameters["white16bIre"]
                     - parameters["black16bIre"]) / 100.0
    chroma = (samples[: count * lines * width].reshape(count, lines, width)
              .astype(np.float64) / units_per_ire)
    rate_mhz = 4.0 * float(ssr.sys_params["fsc_mhz"])
    start_us, end_us = ssr.sys_params["colorBurstUS"]
    lo, hi = int(round(start_us * rate_mhz)), int(round(end_us * rate_mhz))
    interior = slice(lo + 4, hi - 4)
    n = np.arange(width)
    cos, sin = np.cos(np.pi * n / 2.0), np.sin(np.pi * n / 2.0)
    log_amplitude = np.full((count, lines), np.nan)
    phase = np.full((count, lines), np.nan)
    phase_error = np.full((count, lines), np.nan)
    envelopes = {True: [], False: []}
    for f in range(count):
        field = chroma[f] - np.median(chroma[f], axis=1, keepdims=True)
        window = field[:, interior]
        i = 2.0 * np.mean(window * cos[interior], axis=1)
        q = -2.0 * np.mean(window * sin[interior], axis=1)
        amplitude = np.hypot(i, q)
        # the phase's own error per line: the residual of the fitted
        # sinusoid over the window (var(phi) = 2 sigma^2 / (N A^2))
        fitted = i[:, None] * cos[interior][None, :] \
            - q[:, None] * sin[interior][None, :]
        sigma = np.std(window - fitted, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            phase_error[f] = np.where(
                amplitude > 0,
                np.sqrt(2.0 / window.shape[1]) * sigma / np.maximum(amplitude, 1e-30),
                np.nan)
        # the vertical interval carries no burst: a line below half the
        # field's median burst amplitude is not a burst line
        ok = amplitude > 0.5 * np.median(amplitude[amplitude > 0]) \
            if (amplitude > 0).any() else np.zeros(lines, dtype=bool)
        line_phase = np.arctan2(q, i)
        if ok.sum() < 8:
            continue
        log_amplitude[f, ok] = np.log(amplitude[ok] / np.median(amplitude[ok]))
        for parity in (0, 1):
            rows = np.nonzero(ok)[0]
            rows = rows[rows % 2 == parity]
            if rows.size < 4:
                continue
            reference = np.angle(np.mean(np.exp(1j * line_phase[rows])))
            phase[f, rows] = np.angle(np.exp(1j * (line_phase[rows] - reference)))
        # the envelope per sample: at four samples per cycle two adjacent
        # samples are a quadrature pair
        edge = field[:, lo - 8:hi + 8]
        envelope = np.hypot(edge[:, :-1], edge[:, 1:])
        strong = ok & (np.abs(log_amplitude[f]) < 0.5)
        if strong.any():
            head = bool(decode.json["fields"][f].get("isFirstField", f % 2 == 0))
            envelopes[head].append(np.mean(envelope[strong], axis=0))
    mean_envelope = {("head_first" if h else "head_second"):
                     np.mean(np.array(v), axis=0)
                     for h, v in envelopes.items() if v}
    phase_error = np.where(np.isfinite(phase), phase_error, np.nan)
    return log_amplitude, phase, mean_envelope, phase_error


def burst_positions(decode):
    """The burst's per-line timing residual as a POSITION on the output
    grid, in fractions of a line (positive = late), with its error.

    The decoder time-base corrects the chroma while it is still the
    COLOR-UNDER and up-converts afterwards, so a residual displacement of
    the output grid imprints its phase at the color-under carrier and the
    up-conversion carries that phase to the subcarrier unchanged. A
    displacement of d output samples is therefore a burst phase of
    (pi/2)(f_cu/f_sc) d, and

        d = -phi x (2 / pi) x (f_sc / f_cu)

    - reading the phase as though it had accrued at the subcarrier
    understates the displacement by the heterodyne ratio (5.6875 on NTSC
    VHS). The fine timing pilot: the resampler's residual is read here,
    not on the luma sync fall (whose wander against the burst-locked grid
    is a luma-path property, reported separately)."""
    measured = burst_residual(decode)
    if measured is None:
        return None
    _, phase, _, phase_error = measured
    width = decode.fields.shape[2]
    subcarrier = float(decode.filters.get("fsc_mhz", 0.0)) * 1e6
    color_under = float(decode.filters.get("color_under_carrier_hz", 0.0))
    scale = (ie.heterodyne_timing_scale(subcarrier, color_under)
             if subcarrier > 0 and color_under > 0 else 1.0)
    samples_per_radian = (2.0 / np.pi) * scale
    position = -phase * samples_per_radian / width
    error = phase_error * samples_per_radian / width
    return position, error


def burst_metric(log_amplitude, phase):
    ok = np.isfinite(log_amplitude) & np.isfinite(phase)
    if not ok.any():
        return {"amplitude_rms": np.nan, "phase_rms": np.nan, "lines": 0}
    return {"amplitude_rms": float(np.sqrt(np.mean(log_amplitude[ok] ** 2))),
            "phase_rms": float(np.sqrt(np.mean(phase[ok] ** 2))),
            "lines": int(ok.sum())}


# --------------------------------------------------------------------------
# the gauges: the model's Protocol, through the real chain
# --------------------------------------------------------------------------


class DecodeGauges:
    def __init__(self, args):
        self.args = args
        self.log = []
        self.history = {}      # component name -> [metrics per pass]
        self.best = {}         # component name -> (metric, decode)
        self.channel_grid = None
        self.channel_log_h = None
        self.channel_var = None
        self.channel_per_head = {}
        self.baseline_export = None
        self.uncorrected_metric = None
        self.uncorrected_by_view = {}
        self.amplitude = None
        self.burst_spec = None
        self.heterodyne = None
        self.vertical = None
        self.phase_lock = None
        self.genlock = None
        # what each component measured this pass, waiting to be written out
        # and applied when the whole model feeds forward together
        self.pending_time = None
        self.pending_frequency = None
        self.fit_population = None
        # the bins the UNCORRECTED decode admitted: every later pass is
        # judged on those, never on a set the correction itself changed
        self.metric_population = None
        # after the time base is done, the step runs on the rise alone
        self.rise_only = bool(getattr(args, "rise_only", False))
        self._said_rise_only = False
        # the time component's accumulated model, keyed by readloc. The
        # consumer is STATELESS - it starts every decode from the same
        # line locations and applies exactly what the file says - so the
        # TOTAL is written every pass, never this pass's share.
        self.time_model = {}
        # the MEASURED response of each view to an applied RF change
        # (scratchpad/jacobian_map.py): 0.75 on both polarities on bars,
        # linear to 1-3%. Without it the relation the update's rows are
        # built from is assumed exact, which it is not.
        self.jacobian = None
        if getattr(args, "jacobian", None):
            self.jacobian = {view: float(gain) for view, gain in
                             (pair.split("=") for pair in
                              args.jacobian.split(","))}
        self.reference = load_decoder_reference(args.decoder_response)
        if self.reference is None:
            self._say("no decoder-only reference given: the video chain's "
                      "static divisor stands in (the RF band-pass's share is "
                      "then in the residual)")

    def _say(self, text):
        print(text, flush=True)
        self.log.append(text)

    # -- the Protocol ------------------------------------------------------

    def measure(self, component, signal):
        decode = signal["decode"]
        if component.amplitude_only:
            return self._measure_noise(decode)
        if component.name == AMPLITUDE_COMPONENT:
            measured = amplitude_residual(decode, self.args.bin_ire)
            metric = amplitude_metric(measured)
            metric["decode"] = decode.name
            self.history.setdefault(component.name, []).append(metric)
            self.amplitude = measured
            self._record_best(component.name,
                              metric.get("per_line_chi2", np.inf), decode)
            band = metric.get("band_mhz", (float("nan"),) * 2)
            self._say(f"  [{decode.name}] constant-envelope residual: "
                      f"response rms {metric.get('response_rms', float('nan')):.4f} "
                      f"nepers over {band[0]:.2f}-{band[1]:.2f} MHz "
                      f"({metric['bins']} bins); per line "
                      f"{metric.get('per_line_rms', float('nan')):.4f}, chi2 "
                      f"{metric.get('per_line_chi2', float('nan')):.1f}")
            if measured is None:
                return {"amplitude": np.zeros(1)}
            return {"amplitude": np.nan_to_num(measured[1]),
                    "time": np.nan_to_num(measured[3])}
        if component.name.startswith("time base"):
            position, error, sync_position, sync_error = time_residual(decode)
            metric = time_metric(position, error, sync_position, sync_error)
            metric["decode"] = decode.name
            self.history.setdefault(component.name, []).append(metric)
            self._record_best(component.name, metric["chi2"], decode)
            self._say(f"  [{decode.name}] resampler residual (burst): position "
                      f"rms {metric['position_rms']:.3e} of a line, chi2 "
                      f"{metric['chi2']:.2f}, lines {metric['lines']}; luma sync "
                      f"alignment rms {metric.get('sync_position_rms', float('nan')):.3e}"
                      f" (chi2 {metric.get('sync_chi2', float('nan')):.1f})")
            self._say(f"      closure (the two gauges' disagreement): "
                      f"{metric.get('closure_rms', float('nan')):.3e} of a "
                      f"line, chi2 {metric.get('closure_chi2', float('nan')):.1f}"
                      f" - rising against a falling residual means the "
                      f"placement-explicable part is being taken")
            self.pending_time = (position, error)
            if not getattr(self.args, "no_rise_only_after_tbc", False) \
                    and done_at_floor(metric):
                self.rise_only = True
            return {"time": np.nan_to_num(position)}
        held_out = self._views(decode, "second")
        metric = held_out_metric(held_out)
        metric["structure"] = structure_test(held_out)
        metric["decode"] = decode.name
        if self.metric_population is None:
            self.metric_population = {view: entry[3].copy()
                                      for view, entry in held_out.items()}
        fixed = fixed_band_metric(held_out, population=self.metric_population)
        metric["fixed_band"] = fixed
        pooled = fixed.get("pooled", {})
        # COMPARE LIKE WITH LIKE: the view set changes when the fall is
        # dropped, so the uncorrected reference must be pooled over the
        # SAME views as the pass being judged. Comparing a rise-only
        # corrected figure against a both-polarity uncorrected one is the
        # same error as a moving band or a growing bin population, and it
        # flattered a rise-only run by nine points before it was caught.
        if self.uncorrected_by_view:
            present = [v for v in fixed if v != "pooled"]
            self.uncorrected_metric = _pool_views_rms(
                self.uncorrected_by_view, present)
        self.history.setdefault(component.name, []).append(metric)
        # the comparable figure, not the moving-band one
        self._record_best(component.name,
                          pooled.get("magnitude_rms", np.inf), decode)
        if pooled:
            self._say("      fixed band %.2f-%.2f MHz: ln|R| rms %.4f, phase "
                      "rms %.4f rad over %d bins%s"
                      % (EVALUATION_BAND_HZ[0] / 1e6, EVALUATION_BAND_HZ[1] / 1e6,
                         pooled["magnitude_rms"], pooled["phase_rms"],
                         pooled["bins"],
                         "".join(f"; {v} {fixed[v]['magnitude_rms']:.4f}"
                                 for v in ("fall", "rise") if v in fixed)))
        self._measure_burst(decode)
        self._measure_heterodyne(decode)
        self._measure_vertical(decode)
        self._measure_phase_lock(decode)
        self._measure_genlock(decode)
        for view, m in metric["per_view"].items():
            self._say(f"      {view}: ln|R| rms {m['magnitude_rms']:.4f}, phase "
                      f"rms {m['phase_rms']:.4f}, chi2 {m['chi2']:.1f}")
        self._say(f"  [{decode.name}] frequency residual (held-out fields): "
                  f"ln|R| rms {metric['magnitude_rms']:.4f}, phase rms "
                  f"{metric['phase_rms']:.4f} rad, chi2 {metric['chi2']:.2f}, "
                  f"lag-1 structure {metric['structure']:+.2f}, bins "
                  f"{metric['bins']}")
        residual = {}
        for view, (freqs, L, var, valid) in held_out.items():
            residual[f"frequency_{view}"] = np.where(valid, L, 0.0)
        self.pending_frequency = self._views(decode, "first")
        residual["frequency"] = (np.concatenate(
            [np.where(v[3], v[1], 0.0) for v in held_out.values()])
            if held_out else np.zeros(1))
        return residual

    def at_floor(self, component, residual):
        history = self.history.get(component.name, [])
        if not history:
            return True
        now = history[-1]
        if component.name == AMPLITUDE_COMPONENT:
            # the envelope's residual is at its floor when what is left
            # per line is within the line's own measurement error
            chi2 = now.get("per_line_chi2", np.nan)
            if not np.isfinite(chi2):
                return True
            if chi2 <= 1.0:
                self._say("  the envelope is at its floor: what is left per "
                          "line is within the line's own error")
                return True
            if len(history) >= 2:
                before = history[-2].get("per_line_rms", np.inf)
                after = now.get("per_line_rms", np.inf)
                if not after < before * (1.0 - self.args.improvement):
                    self._say(f"  the envelope residual stopped dropping "
                              f"({before:.4f} -> {after:.4f})")
                    return True
            return False
        if not np.isfinite(now.get("chi2", np.nan)):
            self._say("  the gauge could not measure: treated as at the floor")
            return True
        if now["chi2"] <= 1.0:
            self._say("  at the floor: the residual is within the gauge's "
                      "own error")
            return True
        if len(history) >= 2 and "fixed_band" in now:
            before = history[-2]["fixed_band"].get("pooled", {}).get(
                "magnitude_rms", np.inf)
            after = now["fixed_band"].get("pooled", {}).get(
                "magnitude_rms", np.inf)
            if not after < before * (1.0 - self.args.improvement):
                self._say(f"  the residual stopped dropping on the fixed band "
                          f"({before:.4f} -> {after:.4f}): stopping one pass "
                          f"earlier")
                return True
            return False
        if len(history) >= 2:
            before, after = history[-2]["chi2"], now["chi2"]
            if not after < before * (1.0 - self.args.improvement):
                self._say(f"  the residual stopped dropping (chi2 {before:.1f} -> "
                          f"{after:.1f}): stopping one pass earlier")
                return True
        return False

    def anti_residual(self, component, residual):
        return {"component": component.name, "alpha": self.args.amount}

    def feed_forward(self, signal, model):
        """Apply the WHOLE accumulated model through the real chain, in
        one decode, and return what it produced.

        Every component that has accumulated anything is written out and
        given to the same decode: the channel response for the frequency
        axis and the time-base increments for the resampler. They are
        never applied one at a time, because a component measured against
        a chain the others have not been applied to is measuring the wrong
        thing."""
        decode = signal["decode"]
        args = self.args
        index = next_index()
        channel_path, time_path = decode.channel, decode.time
        applied = []
        if self.pending_time is not None:
            position, error = self.pending_time
            time_path = os.path.join(
                args.work, f"{args.prefix}_pass{index}_time_base_response.npz")
            write_time_response(time_path, decode, position, error,
                                args.amount, index, self.time_model)
            applied.append("time")
            self.pending_time = None
        if self.pending_frequency is not None:
            fitted = self.pending_frequency
            ratio, clipped = tip_flatness(decode)
            self._say(f"      sync tip flatness: tip/porch std {ratio:.3f}"
                      f"{' - CLIPPED at record: the fall is excluded' if clipped else ''}")
            if clipped and "fall" in fitted:
                fitted = {view: v for view, v in fitted.items() if view != "fall"}
            # FIT THE SAME POPULATION EVERY PASS. As the correction
            # improves the response, more bins pass the 3 dB ceiling and
            # the SNR gate - and the newly admitted ones are the band
            # edges, where the response is worst and the measurement
            # noisiest. Fitting a growing population drives large
            # corrections at the edges, which is what destroyed the pass
            # after the good one (bins 1415 -> 1616 -> collapse). The
            # same lesson as the fixed evaluation band, applied to the
            # fit rather than the metric: freeze the population on the
            # first pass and judge every later one on it.
            if self.fit_population is None:
                self.fit_population = {view: entry[3].copy()
                                       for view, entry in fitted.items()}
                self._say("      fit population frozen: %s"
                          % ", ".join(f"{v} {int(m.sum())} bins"
                                      for v, m in self.fit_population.items()))
            else:
                fitted = {view: (entry[0], entry[1], entry[2],
                                 entry[3] & self.fit_population.get(
                                     view, entry[3]))
                          for view, entry in fitted.items()}
            fitted = self._agreement(fitted, args.amount)
            increment, variance = map_to_rf(fitted, decode.filters,
                                            self.channel_grid, args.amount,
                                            jacobian=self.jacobian)
            common = getattr(map_to_rf, "polarity_common", None)
            if common:
                self._say("      polarity-common share of the evidence: "
                          "%.0f%% magnitude, %.0f%% phase (the rest is the "
                          "nonlinear remainder, not corrected here)"
                          % (100 * common[0], 100 * common[1]))
            for view, gain in getattr(map_to_rf, "dropped_views", []):
                self._say(f"      {view}: measured Jacobian {gain:.2f} is "
                          f"below {MINIMUM_JACOBIAN} - not reachable from "
                          f"the RF site, dropped from the fit")
            proposed = self.channel_log_h + increment
            excess = np.abs(proposed.real) > MAXIMUM_LOG_MAGNITUDE
            if excess.any():
                self._say(f"      clamped {int(excess.sum())} bins asking "
                          f"beyond {20 * MAXIMUM_LOG_MAGNITUDE / math.log(10):.1f} dB")
                proposed = (np.clip(proposed.real, -MAXIMUM_LOG_MAGNITUDE,
                                    MAXIMUM_LOG_MAGNITUDE)
                            + 1j * proposed.imag)
            self.channel_log_h = proposed
            self.channel_var = np.where(np.isfinite(variance),
                                        np.minimum(self.channel_var, variance),
                                        self.channel_var)
            channel_path = os.path.join(
                args.work, f"{args.prefix}_pass{index}_channel_response.npz")
            self._write_channel(channel_path, decode, index)
            applied.append("frequency")
            self.pending_frequency = None
        new = Decode(args.work, args.prefix, index, channel=channel_path,
                     time=time_path)
        self._say(f"  the model feeds forward together ({', '.join(applied) or 'nothing new'})"
                  f" -> {new.name}")
        run_decode(args, new)
        return {"decode": new}

    def _views(self, decode, subset):
        """The per-polarity residuals, with the FALL dropped once the time
        base has converged.

        Ethan's rule: after the time base correction is done, run the step
        on the RISING EDGE ONLY. The sync tip was clipped when the signal
        entered the recording VCR and resolves flat, so the falling edge's
        landing is the CLIPPER's shape rather than the channel's, and
        measuring the channel there fits the wrong thing. The rise runs
        from that flat tip up into the BACK PORCH, which is the shape
        worth correcting - and its ideal already treats the tip as flat,
        since the rise's pre-level is the tip's own median rather than a
        modelled settling."""
        views = pooled_views([frequency_residual(decode, parity, subset,
                                                 self.reference)
                              for parity in (0, 1)])
        if self.rise_only and "fall" in views and len(views) > 1:
            if not self._said_rise_only:
                self._say("      RISING EDGE ONLY from here: the clipped "
                          "sync tip makes the fall's landing the clipper's "
                          "shape, not the channel's; the rise carries the "
                          "back porch")
                self._said_rise_only = True
            views = {view: entry for view, entry in views.items()
                     if view != "fall"}
        return views

    def cross_check(self, residuals):
        """The components influence each other. The time and frequency
        residuals share the sync edge: a per-line timing error displaces
        the edge and so appears in the edge's measured response as a
        linear phase, and the response's own group delay appears in the
        timing. What they agree on belongs to the TIME component, which
        identifies it by a criterion the frequency gauge cannot (the
        burst, an independent pilot); what is left stays in the
        frequency residual. Implemented as removing, from each view's
        phase, the linear-in-frequency part the measured timing residual
        predicts."""
        time_name = next((n for n in residuals if n.startswith("time base")),
                         None)
        if time_name is None or len(residuals) < 2:
            return residuals
        timing = residuals[time_name].get("time")
        if timing is None or not np.isfinite(timing).any():
            return residuals
        # the mean residual displacement, in fractions of a line, becomes a
        # group delay in seconds on the output grid
        line_seconds = float(ssr.sys_params["line_period"]) * 1e-6
        delay = float(np.nanmean(timing)) * line_seconds
        moved = {}
        for name, residual in residuals.items():
            if name == time_name:
                moved[name] = residual
                continue
            adjusted = {}
            for key, value in residual.items():
                if key.startswith("frequency_") and np.iscomplexobj(value):
                    freqs = ssr.frequency_mhz * 1e6
                    if len(freqs) == len(value):
                        adjusted[key] = (value.real
                                         + 1j * (value.imag
                                                 + 2.0 * np.pi * freqs * delay))
                        continue
                adjusted[key] = value
            moved[name] = adjusted
        self._say(f"      cross-check: {delay * 1e9:+.1f} ns of measured "
                  f"timing residual removed from the frequency phase")
        return moved

    def _agreement(self, fitted, alpha):
        """Per bin, how far the residual's observed change since the last
        pass followed the change the last increment predicted (-alpha x
        the residual then, times that bin's weight then): the garrote
        1 - |observed - predicted|^2 / max(|predicted|^2, noise). A bin
        that does not respond as an RF filter would - the nonlinear
        remainder, or a band with no information behind it - loses its
        weight instead of being fed forward again; a bin whose predicted
        change is within the noise keeps its weight. The weighted views
        are returned with the agreement as a fifth element, and kept for
        the next pass."""
        previous = getattr(self, "_previous_fit", None)
        weighted = {}
        for view, (freqs, L, var, valid) in fitted.items():
            agreement = np.ones(len(freqs))
            if previous is not None and view in previous:
                p_freqs, p_L, p_var, p_valid, p_agree = previous[view]
                both = valid & p_valid
                predicted = -alpha * p_L * p_agree
                observed = L - p_L
                with np.errstate(divide="ignore", invalid="ignore"):
                    floor = np.maximum(np.abs(predicted) ** 2, var + p_var)
                    a = 1.0 - np.abs(observed - predicted) ** 2 / floor
                agreement = np.where(both, np.clip(np.nan_to_num(a, nan=0.0),
                                                   0.0, 1.0), agreement)
                kept = float(np.mean(agreement[both])) if both.any() else 1.0
                self._say(f"      {view}: response agreement mean {kept:.2f} "
                          f"over {int(both.sum())} bins")
            weighted[view] = (freqs, L, var, valid, agreement)
        self._previous_fit = weighted
        return weighted

    def subtract_noise_floor(self, signal, floor):
        return signal

    def measure_per_field(self, component, signal):
        decode = signal["decode"]
        position, error, _, _ = time_residual(decode)
        track = []
        for row in position:
            ok = np.isfinite(row)
            track.append(float(np.sqrt(np.mean(row[ok] ** 2))) if ok.any()
                         else np.nan)
        return np.array(track)

    # -- the noise floor ---------------------------------------------------

    def _measure_noise(self, decode):
        """Amplitude only: the accumulated gauge's variance of the mean per
        bin on the last decode - what does not converge with the lines."""
        rows = []
        for parity in (0, 1):
            for view, (freqs, L, var, valid) in frequency_residual(
                    decode, parity, None, self.reference).items():
                rows.append(np.where(valid, var, np.nan))
        floor = np.nanmean(np.array(rows), axis=0) if rows else np.zeros(1)
        self._say(f"  noise floor (amplitude only): median var of ln|R| "
                  f"{np.nanmedian(floor):.2e} over {int(np.isfinite(floor).sum())}"
                  f" bins")
        return {"amplitude": np.nan_to_num(floor)}

    def _measure_burst(self, decode):
        """The demodulated burst residual - the chroma pilot's alignment
        parameters - measured and tracked on every pass."""
        measured = burst_residual(decode)
        if measured is None:
            return
        log_amplitude, phase, envelope, _ = measured
        metric = burst_metric(log_amplitude, phase)
        # the burst measured against its SPECIFIED shape, on all three
        # axes, including the heterodyne's own frequency error - which no
        # single phase measurement can see, only the ramp across the burst
        spec = burst_components(decode)
        if spec is not None:
            against_spec = burst_component_metric(spec)
            metric.update(against_spec)
            self.burst_spec = spec
            self._say("      against the specification: amplitude %.0f%% of "
                      "%.0f IRE (per-line rms %.3f), phase rms %.4f rad, "
                      "heterodyne error %+.0f Hz mean / %.0f rms"
                      % (100 * np.exp(against_spec.get("burst_amplitude_mean", 0.0)),
                         spec["specified_ire"],
                         against_spec.get("burst_amplitude_rms", float("nan")),
                         against_spec.get("burst_time_rms", float("nan")),
                         against_spec.get("heterodyne_mean_hz", float("nan")),
                         against_spec.get("heterodyne_rms_hz", float("nan"))))
        metric["decode"] = decode.name
        self.history.setdefault(BURST_COMPONENT, []).append(metric)
        self.burst_envelope = envelope
        self._say(f"  [{decode.name}] demodulated burst residual: ln amplitude "
                  f"rms {metric['amplitude_rms']:.4f}, phase rms "
                  f"{metric['phase_rms']:.4f} rad, lines {metric['lines']}")
        extra = {}
        if self.burst_spec is not None:
            extra = {"spec_log_amplitude": self.burst_spec["log_amplitude"],
                     "spec_phase": self.burst_spec["phase"],
                     "heterodyne_hz": self.burst_spec["heterodyne_hz"],
                     "specified_ire": self.burst_spec["specified_ire"]}
            extra.update({f"spec_envelope_{k}": v
                          for k, v in self.burst_spec["envelope"].items()})
        np.savez(os.path.join(self.args.work, f"{decode.name}_burst.npz"),
                 log_amplitude=log_amplitude, phase=phase,
                 **{f"envelope_{k}": v for k, v in envelope.items()},
                 **extra)

    def _measure_heterodyne(self, decode):
        """The heterodyne's frequency error, from what the burst lock had
        to move - the phase error IS the frequency correction - split into
        the line-locked part and the wobbling one."""
        measured = heterodyne_correction(decode)
        metric = heterodyne_metric(measured)
        if not metric:
            return
        metric["decode"] = decode.name
        self.history.setdefault(HETERODYNE_COMPONENT, []).append(metric)
        self.heterodyne = measured
        self._say("  [%s] heterodyne frequency: line-locked %.0f Hz rms, "
                  "wobbling %.0f Hz rms, repeatability %.2f"
                  % (decode.name, metric["heterodyne_line_locked_rms"],
                     metric["heterodyne_varying_rms"],
                     metric.get("heterodyne_repeatability", float("nan"))))
        if "heterodyne_pattern_cycles_per_line" in metric:
            cycles = metric["heterodyne_pattern_cycles_per_line"]
            period = (1.0 / cycles) if cycles > 0 else float("inf")
            self._say("      the line-locked pattern peaks every %.1f lines, "
                      "at %.0fx the mean" % (period,
                                             metric["heterodyne_pattern_strength"]))
        np.savez(os.path.join(self.args.work, f"{decode.name}_heterodyne.npz"),
                 **{k: v for k, v in measured.items()})

    def _measure_vertical(self, decode):
        """The low-frequency response, from the vertical sync block's droop:
        a CONSTANT of the path, measured once and not varying with time."""
        measured = vertical_low_frequency(decode)
        metric = vertical_metric(measured)
        if not metric:
            return
        metric["decode"] = decode.name
        self.history.setdefault(VERTICAL_COMPONENT, []).append(metric)
        self.vertical = measured
        self._say("  [%s] vertical interval: excess %+.3f IRE (%.0f sigma) "
                  "recovering with tau %.0f us -> a low-frequency corner "
                  "near %.0f Hz"
                  % (decode.name, metric["vertical_excess_ire"],
                     metric["vertical_excess_sigma"], metric["vertical_tau_us"],
                     metric["vertical_corner_hz"]))
        if "vertical_step_ire" in metric:
            self._say("      the step across the block: blanking %+.3f IRE "
                      "(%.0f sigma), sync tip %+.3f - together is coupling, "
                      "apart is gain"
                      % (metric["vertical_step_ire"], metric["vertical_step_sigma"],
                         metric.get("vertical_tip_step_ire", float("nan"))))
        np.savez(os.path.join(self.args.work, f"{decode.name}_vertical.npz"),
                 **{k: v for k, v in measured.items() if v is not None})

    def _measure_phase_lock(self, decode):
        """The burst-to-sync phase relationship - the exact phase lock, and
        the only absolute phase reference the signal carries."""
        measured = burst_sync_phase(decode)
        metric = phase_lock_metric(measured)
        if not metric:
            return
        metric["decode"] = decode.name
        self.history.setdefault(PHASE_LOCK_COMPONENT, []).append(metric)
        self.phase_lock = measured
        self._say("  [%s] burst-to-sync phase lock: offset %+.1f +- %.1f Hz, "
                  "wander %.1f degrees, %.0f burstless lines"
                  % (decode.name, metric.get("phase_lock_offset_hz", float("nan")),
                     metric.get("phase_lock_offset_error_hz", float("nan")),
                     metric.get("phase_lock_wander_degrees", float("nan")),
                     metric.get("phase_lock_burstless_lines", float("nan"))))
        if "phase_lock_vertical_carry_degrees" in metric:
            self._say("      carried across the vertical interval by the sync "
                      "pulses alone: %+.1f degrees (rms %.1f, %.1f sigma)"
                      % (metric["phase_lock_vertical_carry_degrees"],
                         metric["phase_lock_vertical_carry_rms_degrees"],
                         metric["phase_lock_vertical_carry_sigma"]))
        np.savez(os.path.join(self.args.work, f"{decode.name}_phase_lock.npz"),
                 **{k: v for k, v in measured.items()})

    def _measure_genlock(self, decode):
        """The two locks differentiated together, which asserts genlocking."""
        measured = genlock(decode)
        metric = genlock_metric(measured)
        if not metric:
            return
        metric["decode"] = decode.name
        self.history.setdefault(GENLOCK_COMPONENT, []).append(metric)
        self.genlock = measured
        self._say("  [%s] genlock: the hsync timing accounts for %.0f%% of the "
                  "burst's movement (correlation %+.3f) -> %s"
                  % (decode.name, 100 * metric.get("genlock_share_explained", 0.0),
                     metric.get("genlock_agreement", float("nan")),
                     "GENLOCKED" if metric.get("genlocked") else
                     "NOT genlocked: the subcarrier ran on its own oscillator"))
        np.savez(os.path.join(self.args.work, f"{decode.name}_genlock.npz"),
                 **{k: v for k, v in measured.items()
                    if isinstance(v, np.ndarray)})

    # -- bookkeeping -------------------------------------------------------

    def _record_best(self, name, value, decode):
        if not np.isfinite(value):
            return
        best = self.best.get(name)
        if best is None or value < best[0]:
            self.best[name] = (value, decode)

    def baseline(self, decode):
        export = channel_identify.identify(decode.residual_dir,
                                           self.args.bin_ire)
        self._say(channel_identify.summarize(export))
        self.baseline_export = export
        # THE GRID THE DATA DETERMINE, not the grid the measurement was
        # binned on. The closure test (closure_test.py) counts how many
        # free bins the measurements actually constrain: on bars, about
        # 124 over the band, a 50 kHz spacing, where the fit's own
        # residual sits at 1.4 times the gauge's noise. On Part A's
        # 1-IRE grid, about 7.1 kHz, there are 1276 free bins against
        # 1359 rows - as many parameters as data - so any residual fits
        # and what is fitted is noise, which is precisely why the table
        # that came out of it made the measured residual worse. Never
        # solve for more free parameters than the sampling function
        # determines; the runtime's own interpolation carries the
        # coarser solution to the block grid.
        bin_hz = float(self.args.solve_bin_hz)
        nyquist = float(decode.filters["freq_hz"]) / 2.0
        self.channel_grid = np.arange(0.5 * bin_hz, nyquist, bin_hz)
        self.channel_log_h = np.zeros(len(self.channel_grid),
                                      dtype=np.complex128)
        self.channel_var = np.full(len(self.channel_grid), np.inf)
        self._say("solving on a %.0f kHz grid (%d bins over the RF band): the "
                  "closure test's count of what the data determine"
                  % (bin_hz / 1e3, len(self.channel_grid)))
        # the baseline, averaged onto that grid rather than deposited on it
        source_f = np.asarray(export["freqs_hz"])
        source_l = np.asarray(export["log_H"])
        source_v = np.asarray(export["noise_psd"])
        index = np.round((source_f - self.channel_grid[0]) / bin_hz).astype(np.int64)
        inside = (index >= 0) & (index < len(self.channel_grid)) \
            & np.isfinite(source_v) & (source_v > 0)
        weight = np.zeros(len(self.channel_grid))
        total = np.zeros(len(self.channel_grid), dtype=np.complex128)
        np.add.at(weight, index[inside], 1.0 / source_v[inside])
        np.add.at(total, index[inside], source_l[inside] / source_v[inside])
        seen = weight > 0
        # The baseline enters at the err-low step, and possibly not at
        # all: the demodulated MAGNITUDE responds to the mean of the two
        # sidebands minus the carrier, which is identically zero for a
        # linear tilt, and the constant-envelope channel is very nearly a
        # linear tilt. So applying it can add its noise without adding
        # signal, and --baseline-amount 0 tests exactly that.
        share = self.args.amount if self.args.baseline_amount is None \
            else self.args.baseline_amount
        self.channel_log_h[seen] = share * total[seen] / weight[seen]
        self.channel_var[seen] = 1.0 / weight[seen]
        path = os.path.join(self.args.work,
                            f"{self.args.prefix}_baseline_channel_response.npz")
        self._write_channel(path, decode, 0)
        return path

    def _write_channel(self, path, decode, index):
        filters = decode.filters
        with np.errstate(divide="ignore", invalid="ignore"):
            belief = np.where(
                np.isfinite(self.channel_var),
                1.0 - self.channel_var
                / np.maximum(np.abs(self.channel_log_h) ** 2, 1e-30), 0.0)
        belief = np.clip(np.nan_to_num(belief, nan=0.0), 0.0, 1.0)
        measured = np.isfinite(self.channel_var)
        export = {
            "freqs_hz": self.channel_grid[measured],
            "log_H": self.channel_log_h[measured].astype(np.complex128),
            "belief": belief[measured],
            "noise_psd": self.channel_var[measured],
            "site": channel_identify.SITE,
            "freq_hz": float(filters["freq_hz"]),
            "blocklen": int(filters["blocklen"]),
            "system": str(filters["system"]),
            "tape_format": str(filters["tape_format"]),
            "tape_speed": str(filters["tape_speed"]),
            "passes": int(index),
            "amount_per_pass": float(self.args.amount),
            "exhaustion": json.dumps(
                self.history.get(FREQUENCY_COMPONENT, [])[-1:], default=float),
        }
        if self.amplitude is not None:
            # The envelope's own view of the channel, exported BESIDE the
            # applied table rather than added to it. It determines what
            # the sync edges cannot - the frequency design annihilates a
            # linear tilt and this measures that tilt directly - so the
            # MODEL is complete with it. It is not applied here because
            # measurement says it does not help at this site (with the
            # baseline applied the best pass was 0.1630 against 0.1544
            # uncorrected, worse), while it does act on the heterodyned
            # chroma path and serves as a reference for other stages.
            centres, response, error, rows, row_errors = self.amplitude
            export["amplitude_freqs_hz"] = centres
            export["amplitude_log_H"] = response
            export["amplitude_error"] = error
            export["amplitude_per_line"] = rows
            export["amplitude_per_line_error"] = row_errors
        for head, views in self.channel_per_head.items():
            for view, (freqs, L, var, valid) in views.items():
                export[f"log_R_{head}_{view}"] = np.where(valid, L, 0.0)
                export[f"var_R_{head}_{view}"] = np.where(valid, var, np.inf)
                export["freqs_R_hz"] = freqs
        np.savez(path, **export)
        return path


# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tape", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--seek", type=int, default=0)
    parser.add_argument("--flags", default="-n -f 50 -t 0")
    parser.add_argument("--passes", type=int, default=3,
                        help="maximum passes per component")
    parser.add_argument("--amount", type=float, default=0.5,
                        help="the err-low step per pass (half the residual)")
    parser.add_argument("--improvement", type=float, default=0.02,
                        help="a residual must drop by this fraction to count "
                             "as still converging")
    parser.add_argument("--bin-ire", type=float, default=1.0)
    parser.add_argument("--baseline-amount", type=float, default=None,
                        help="how much of the identified channel baseline to "
                             "apply (default: the same err-low step as the "
                             "loop). Zero starts the loop from no table at "
                             "all, which the carrier-normalization symmetry "
                             "says should cost nothing: a linear tilt is "
                             "invisible to the demodulated magnitude")
    parser.add_argument("--solve-bin-hz", type=float, default=50e3,
                        help="the RF grid the channel is SOLVED on. Set by "
                             "the closure test (closure_test.py), which "
                             "counts how many free bins the measurements "
                             "determine: 50 kHz on bars, where the fit's own "
                             "residual is 1.4 times the gauge noise. Finer "
                             "than the data support and the fit absorbs "
                             "noise; coarser and real structure is forced "
                             "out. The runtime interpolates onto the block "
                             "grid, so this need not be fine")
    parser.add_argument("--skip-time", action="store_true",
                        help="do not iterate the sync resampler")
    parser.add_argument("--decoder-response", default=None,
                        help="npz with the decoder-only per-polarity sync "
                             "response (decoder_only_{fall,rise}_H_peaking_on "
                             "on sync_freqs_hz), the reference the measured "
                             "edges are differentiated against")
    parser.add_argument("--rise-only", action="store_true",
                        help="run the frequency step on the RISING EDGE "
                             "alone from the start: the sync tip is clipped "
                             "at record time and resolves flat, so the "
                             "falling edge's landing is the clipper's shape "
                             "rather than the channel's, while the rise "
                             "carries the back porch")
    parser.add_argument("--no-rise-only-after-tbc", action="store_true",
                        help="keep both polarities even after the time base "
                             "reaches its floor (the default drops the fall "
                             "at that point)")
    parser.add_argument("--jacobian", default=None,
                        help="the MEASURED response of each view to an "
                             "applied RF change, as fall=0.75,rise=0.75 "
                             "(scratchpad/jacobian_map.py); the update is "
                             "divided by it, and a view whose gain falls "
                             "below the minimum is dropped as unreachable "
                             "from this site. Without it the relation the "
                             "rows are built from is assumed exact, which "
                             "it is not")
    args = parser.parse_args(argv)
    os.makedirs(args.work, exist_ok=True)

    gauges = DecodeGauges(args)
    gauges._say(f"pass 0: baseline decode of {os.path.basename(args.tape)}")
    first = run_decode(args, Decode(args.work, args.prefix, next_index()))
    # the figure every later pass must beat: the decode with NO table
    first_views = pooled_views(
        [frequency_residual(first, parity, "second", gauges.reference)
         for parity in (0, 1)])
    gauges.metric_population = {view: entry[3].copy()
                                for view, entry in first_views.items()}
    uncorrected = fixed_band_metric(first_views,
                                    population=gauges.metric_population)
    gauges.uncorrected_by_view = {view: entry for view, entry
                                  in uncorrected.items() if view != "pooled"}
    gauges.uncorrected_metric = uncorrected.get("pooled", {}).get(
        "magnitude_rms", None)
    if gauges.uncorrected_metric is not None:
        gauges._say("uncorrected residual on the fixed band: ln|R| rms "
                    "%.4f%s" % (gauges.uncorrected_metric,
                                "".join(f"; {v} {uncorrected[v]['magnitude_rms']:.4f}"
                                        for v in ("fall", "rise")
                                        if v in uncorrected)))
    channel_path = gauges.baseline(first)
    gauges._say("pass 1: the baseline fed forward at the half step")
    current = run_decode(args, Decode(args.work, args.prefix, next_index(),
                                      channel=channel_path))
    n_samples = int(first.filters["blocklen"])
    components = []
    if not args.skip_time:
        components.append(ie.time_base(n_samples))
    components.append(ie._component(
        AMPLITUDE_COMPONENT,
        "the luma carrier is recorded at constant amplitude, so every "
        "envelope variation is the path's magnitude at the instantaneous "
        "carrier: the response per carrier bin, and what is left per line",
        ("amplitude", "frequency"), True))
    components.append(ie._component(
        FREQUENCY_COMPONENT,
        "the sync edge's derivative against the spec-shaped ideal at the "
        "measured plateau levels, decoder divided out, mapped about each "
        "landing carrier", ("frequency", "time"), True))
    result = ie.multidimensional_information_extrapolation(
        {"decode": current}, components, gauges, noise=ie.noise_floor(),
        maximum_passes=args.passes)
    final = result.signal["decode"]
    best_value, best = gauges.best.get(FREQUENCY_COMPONENT, (np.inf, final))
    # THE NO-HARM GATE: a table that does not beat NO TABLE on the fixed
    # band is not exported. The loop's whole claim is that inverting the
    # identified channel improves the measured residual; when it does
    # not, the honest product is no table at all, and the uncorrected
    # figure is the one to report.
    uncorrected = gauges.uncorrected_metric
    if uncorrected is not None and np.isfinite(best_value) \
            and not best_value < uncorrected:
        gauges._say(f"NO-HARM GATE: the best pass ({best_value:.4f}) does not "
                    f"beat the uncorrected decode ({uncorrected:.4f}) on the "
                    f"fixed band - no table is exported")
        best = None
    out = os.path.join(args.work, f"{args.prefix}_channel_response.npz")
    if best is not None and best.channel is not None:
        shutil.copyfile(best.channel, out)
    summary = {
        "passes": [vars(p) for p in result.passes],
        "weakly_determined": result.weakly_determined,
        "history": gauges.history,
        "final_decode": final.name,
        "best_decode": best.name if best is not None else None,
        "best_channel": best.channel if best is not None else None,
        "best_time": best.time if best is not None else None,
        "uncorrected_fixed_band": gauges.uncorrected_metric,
        "best_fixed_band": best_value,
        "field_floor": (result.field_floor.tolist()
                        if result.field_floor is not None else None),
        "field_differential": (result.field_differential.tolist()
                               if result.field_differential is not None
                               else None),
        "log": gauges.log,
    }
    with open(os.path.join(args.work, f"{args.prefix}_loop.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    gauges._say(f"done: best {best.name if best else 'none (no-harm gate)'}; "
                f"export {out if best else 'withheld'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
