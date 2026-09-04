"""Carrier-refined time base correction.

A playback speed error does two things to a line at once, because the
recording is frequency modulated: it time-warps the line (the samples of a
slow line span more of the capture than they should) and it shifts every
demodulated level (the carrier's instantaneous frequency scales with head
to tape speed, and frequency IS level after FM demodulation).  Both errors
are the same physical quantity - the speed ratio - so one measurement must
correct both, and the downscale already works that way: it splines
expected-vs-actual line locations, resamples the picture along the
spline's VALUE, and multiplies the demodulated signal by the spline's
DERIVATIVE (`computewow_scaled` / `scale_field`).  Whatever improves the
line locations improves the timing and the level correction together, in
one place, with no way for the two to disagree.

What limits that spline today is the quality of the line locations
feeding it.  The sync-pulse TBC measures each line's 50% sync crossing on
the demodulated video, where the crossing sits on a noisy edge; the
carrier trace (`luma_amplitude.sync_edge_trace`) refines the SAME
crossing, with the same 50%-falling definition, from the demodulated
carrier frequency before de-emphasis - a channel on which amplitude
events cannot move the edge, with estimator noise about a tenth of the
real line-to-line jitter.  This module AUGMENTS the sync-derived time
base with the carrier
trace - Ethan's ruling: the flutter measurement becomes "a fully formed
time base correction used in addition to the sync-pulse-derived TBC...
for timebase correction as well as level correction... put into the
linelocs spline in addition to the existing hsync-derived measurements."

The refinement is deliberately a DEVIATION correction, not a position
replacement.  The final `linelocs` are not raw hsync measurements: the
color path shifts them per line to lock the burst phase and subtracts a
constant subcarrier-phase calibration (`FieldNTSC.process`), so the
trace-minus-linelocs delta carries a systematic offset that belongs to
those conventions, not to the tape.  The MEDIAN delta holds that offset;
moving it would shift the whole picture horizontally and re-fight the
burst lock's absolute alignment.  Only the per-line deviation ABOUT the
median is applied.  Lines the trace could not measure, and lines whose
delta fails the statistical gate, keep their sync-derived location: the
correction falls back to exactly the time base that exists today.

The deviation is applied in the band where the trace is measured to be
the better instrument: the wow-and-flutter band, at and below the drum
rotation rate.  This is a measured decision, from A/B decodes of the
reference tape.  The output's drum-rate timing wobble fell by tens of
percent when the deviations fed the spline, proving the trace's
low-frequency content real - but the LINE-scale deviations, applied
raw, correlated at +0.04 with the uncorrected output's actual per-line
edge deviations (they are two estimators disagreeing about a time base
the burst-locked sync TBC already tracks to better than the trace's own
noise), and their first difference tripled the per-line PERIOD noise
the spline's derivative turns into the level adjust, measurably
modulating picture brightness at the field and drum rates.  A zero-phase
moving average over the gated deviations - gap-aware, so unmeasured
lines contribute no evidence but still receive the local flutter
estimate - keeps the proven band and discards the disagreement; its
width derives from the drum period exactly as the level smoothing's
corner does, one physical scale for both.
"""

import logging
from collections import namedtuple

import numpy as np

import lddecode.core as ldd

from vhsdecode import luma_amplitude


TimeBasePrior = namedtuple(
    "TimeBasePrior",
    ["line_deviation", "standard_error", "band_hz", "witness", "line_position", "position_standard_error"],
    defaults=(None, None),
)
# `line_position` (optional): the prior as a per-line POSITION deviation in
# fractions of a line (a positive value = this line's sync fall came late),
# with its own per-line error. A position witness enters the blend
# directly; a period witness has to be integrated to positions first, and
# integration accumulates its per-line error as a random walk (root-window
# growth) where a direct position witness's error shrinks by root-window
# when smoothed - so the same gauge is worth far more exported as positions.


def _log():
    """The decoder's logger while a decode runs, the module's own otherwise."""
    return ldd.logger if getattr(ldd, "logger", None) is not None else logging.getLogger(__name__)


def _declared_band_limit(table):
    """The writer's own statement of the band it limited to, in lines, or None."""
    return None if table is None else table.get("band_limited_lines")


def _declared_regulator(table):
    """The writer's own statement of how it regulated what it sent, or None."""
    return None if table is None else table.get("regulator")


def _declared_string(z, key):
    """A scalar string the file declares about itself, or None."""
    if key not in z.files:
        return None
    value = z[key]
    value = value.item() if np.ndim(value) == 0 else value[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def _declared_semantics(z):
    """The file's own statement of which quantity it carries, or None."""
    return _declared_string(z, "semantics")


def _drum_window(field):
    """Lines per drum-band smoothing window - frame_lines/(2 pi), the scale
    refine_linelocs already uses (see its comment)."""
    return max(int(round(float(field.rf.SysParams["frame_lines"]) / (2.0 * np.pi))), 1)


def _band_smooth(values, window):
    """Gap-aware moving average over `window` lines; NaN where nothing was seen."""
    good = np.isfinite(values)
    kernel = np.ones(window, dtype=np.float64)
    evidence = np.convolve(good.astype(np.float64), kernel, mode="same")
    total = np.convolve(np.where(good, values, 0.0), kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(evidence > 0, total / np.maximum(evidence, 1.0), np.nan)


def time_base_prior(field):
    """A prior for the line periods from the carrier amplitude's drum-rate wobble.

    Ethan's time-base node: the carrier-amplitude residual (what the head
    switch cancellation measures - the log-amplitude the fitted response
    does not explain) wobbles at the drum's rotation rate, and part of that
    wobble is the same playback timing error the sync-edge trace measures -
    at the drum band it is the DRUM's rotational jitter rather than a tape
    speed error, since the helical head reads at 5.80 m/s against a tape
    moving at 33.35 mm/s, so tape speed is diluted about 174 to 1 into the
    line rate and the capstan's whole tolerance is worth only some tens of
    parts per million of it. Measured
    (2026-09-02, 75-bars and home, 86 fields): the wobble's correlation with
    the period deviation is +0.56 on one head and -0.18 on the other on the
    bars tape, +0.19 and -0.05 on the home recording - its SIGN depends on
    the head, so it is head-to-tape contact modulated by the drum rotation
    with a per-head phase, of which only a per-head share is the speed
    error. The prior is therefore a per-head regression learned over the
    decode (previous fields only, so this field's own trace never
    explains itself), and its standard error is that regression's residual
    scale: an honest weight, which on those tapes lets the prior centre
    very little.

    Returns None until a head has two fields of history or when the
    field carries no amplitude residual; else a TimeBasePrior with
    `line_deviation` = the predicted fractional period deviation per line
    interval (actual/nominal - 1, positive = a long line), `standard_error`
    per line in the same unit, `band_hz` = (0, line rate / window) - the
    drum band and its harmonics up to the flutter-band edge, and the
    witness name.
    """
    rf = field.rf
    model = luma_amplitude.block_model(rf)
    video = field.data["video"]
    if model is None or "demod_raw" not in video or "envelope" not in video:
        return None
    trace = luma_amplitude.sync_edge_trace(field)
    if trace is None:
        return None
    edges, period = trace
    if not period > 0:
        return None
    linelocs = np.asarray(field.linelocs, dtype=np.float64)
    n = min(len(edges), len(linelocs)) - 1
    if n < 3:
        return None
    residual = np.asarray(
        luma_amplitude.block_log_residual(model, video["envelope"], video["demod_raw"]),
        dtype=np.float64,
    )
    # The decoder's own dropout bound limits belief in the residual, as in
    # head_switch; beyond it there is no evidence.
    limit = float(-np.log(rf.dod_options.dod_threshold_p))
    residual = np.where(np.abs(residual) <= limit, residual, np.nan)
    amplitude = np.full(n, np.nan)
    deviation = np.full(n, np.nan)
    edge_view = np.asarray(edges, dtype=np.float64)
    for i in range(n):
        a, b = int(linelocs[i]), int(linelocs[i + 1])
        if 0 < a < b <= len(residual):
            segment = residual[a:b]
            segment = segment[np.isfinite(segment)]
            if len(segment):
                amplitude[i] = np.median(segment)
        if edge_view[i] > 0 and edge_view[i + 1] > 0:
            deviation[i] = (edge_view[i + 1] - edge_view[i]) / period - 1.0
    window = _drum_window(field)
    good = np.isfinite(amplitude) & np.isfinite(deviation)
    if np.count_nonzero(good) < window:
        return None
    # Drum band: both series smoothed over the window and centred on their
    # medians, so the regression sees the wobble, not the level.
    a_band = _band_smooth(amplitude - np.nanmedian(amplitude), window)
    d_band = _band_smooth(deviation - np.nanmedian(deviation), window)
    both = np.isfinite(a_band) & np.isfinite(d_band)

    # Per-head regression state, accumulated over the decode.
    store = rf.__dict__.setdefault("_time_base_prior_state", {})
    head = bool(field.isFirstField)
    state = store.setdefault(head, {"sxx": 0.0, "sxy": 0.0, "syy": 0.0, "n": 0, "fields": 0})
    prior = None
    if state["fields"] >= 2 and state["sxx"] > 0 and state["n"] > 2:
        beta = state["sxy"] / state["sxx"]
        unexplained = max(state["syy"] - beta * state["sxy"], 0.0) / (state["n"] - 1)
        line_deviation = np.where(np.isfinite(a_band), beta * a_band, 0.0)
        standard_error = np.full(n, float(np.sqrt(unexplained)))
        line_rate = 1e6 / float(rf.SysParams["line_period"])
        prior = TimeBasePrior(
            line_deviation, standard_error, (0.0, line_rate / window),
            "carrier amplitude drum-rate wobble, per-head regression",
        )
    # This field joins the history after the prediction was made from it.
    state["sxx"] += float(np.sum(a_band[both] ** 2))
    state["sxy"] += float(np.sum(a_band[both] * d_band[both]))
    state["syy"] += float(np.sum(d_band[both] ** 2))
    state["n"] += int(np.count_nonzero(both))
    state["fields"] += 1
    return prior


def time_base_state(rf):
    """The per-head regression behind `time_base_prior`, for the field axis.

    Per head (True = first field): the learned slope `beta` (fractional
    period deviation per neper of amplitude wobble), `r2` (the share of the
    drum-band period deviation the wobble explains), `standard_error` (the
    prior's per-line error, the regression's residual scale) and `fields`
    seen. The sign of beta flipping between heads is the head-contact
    structure of the wobble - a per-head component to differentiate on.
    """
    out = {}
    for head, state in rf.__dict__.get("_time_base_prior_state", {}).items():
        if state["sxx"] > 0 and state["syy"] > 0 and state["n"] > 2:
            beta = state["sxy"] / state["sxx"]
            unexplained = max(state["syy"] - beta * state["sxy"], 0.0) / (state["n"] - 1)
            out[head] = {
                "beta": float(beta),
                "r2": float(state["sxy"] ** 2 / (state["sxx"] * state["syy"])),
                "standard_error": float(np.sqrt(unexplained)),
                "fields": int(state["fields"]),
                "weight": float(rf.__dict__.get("_time_base_prior_weight", 0.0)),
            }
    return out


def _time_base_response(rf):
    """The offline loop's per-field time anti-residual, loaded once.

    An npz with `readloc` (one per record, the field's own read location as
    every residual-channel export carries it), `line_deviation` (records x
    lines, fractional period deviation, NaN-padded), `standard_error` (the
    same shape and unit) and `pass_index`. Records are looked up by exact
    readloc; a field without a record gets no prior from this source.
    """
    cache = rf.__dict__.get("_time_base_response_table")
    if cache is not None:
        return cache
    path = getattr(rf, "_time_base_response", None)
    table = None
    if path:
        z = np.load(path, allow_pickle=False)
        # The file must say which quantity it carries. This consumer is
        # stateless - every decode applies exactly what the file names to
        # the same sync-derived base - so only a TOTAL can be honoured, and
        # a file that declares anything else is refused rather than
        # misapplied. (Measured: a writer emitting one pass's share to a
        # reader applying it as a total discards every previous pass, so the
        # correction can never accumulate past a single step, which reads as
        # a physical limit and is arithmetic.) A file that declares nothing
        # is used with the total reading and a warning.
        semantics = _declared_semantics(z)
        if semantics is None:
            _log().warning(
                "carrier_tbc: %s declares no `semantics` - read as a TOTAL displacement "
                "from the sync-derived line locations, which is the only reading this "
                "stateless consumer can apply", path)
        elif semantics != "total":
            _log().error(
                "carrier_tbc: %s declares semantics=%r; this consumer is stateless and "
                "can only apply a total, so the file is refused", path, semantics)
            rf.__dict__["_time_base_response_table"] = None
            return None
        table = {
            "readloc": np.asarray(z["readloc"]).astype(np.int64),
            "line_deviation": np.asarray(z["line_deviation"], dtype=np.float64) if "line_deviation" in z.files else None,
            "standard_error": np.asarray(z["standard_error"], dtype=np.float64) if "standard_error" in z.files else None,
            "line_position": np.asarray(z["line_position"], dtype=np.float64) if "line_position" in z.files else None,
            "position_increment": np.asarray(z["position_increment"], dtype=np.float64) if "position_increment" in z.files else None,
            "position_standard_error": np.asarray(z["position_standard_error"], dtype=np.float64) if "position_standard_error" in z.files else None,
            "pass_index": np.asarray(z["pass_index"]).astype(np.int64) if "pass_index" in z.files else None,
            "band_limited_lines": (float(z["band_limited_lines"]) if "band_limited_lines" in z.files else None),
            "regulator": _declared_string(z, "regulator"),
        }
    rf.__dict__["_time_base_response_table"] = table
    return table


def time_base_response_prior(field):
    """The loop's converged time anti-residual for this field, as a prior.

    The sync resampler is one of the loop's derivable components: each
    pass measures its residual (the time channel exported per field) and
    the anti-residual comes back here on the next pass, on the same footing
    as the amplitude witness - one more prior source for refine_linelocs'
    inverse-variance centring. None without a file or a record.

    WHAT THE PRIOR FORM IS COMPARED AGAINST: the flutter-band deviation of
    the true line position from the SYNC-DERIVED line location (the
    field's linelocs before this refinement) - the quantity the carrier
    trace measures and refine_linelocs adds. It is NOT the line's absolute
    position about a nominal grid (the cumulative wow): the sync-derived
    locations already contain that, and a tight-error record of it wins
    the blend and moves every line by the whole wow (measured: burst phase
    wander 0.04 -> 0.93 rad). A loop's anti-residual is an increment, not
    an estimate of this deviation - use `position_increment` for it.
    """
    table = _time_base_response(field.rf)
    if table is None:
        return None
    readloc = int(getattr(field, "readloc", -1))
    hits = np.where(table["readloc"] == readloc)[0]
    if len(hits) == 0:
        return None
    row = hits[-1]
    line_rate = 1e6 / float(field.rf.SysParams["line_period"])
    pass_index = int(table["pass_index"][row]) if table["pass_index"] is not None else 0
    band = (0.0, line_rate / _drum_window(field))
    witness = f"loop time anti-residual, pass {pass_index}"
    position = position_error = None
    if table["line_position"] is not None and table["position_standard_error"] is not None:
        position = table["line_position"][row]
        position_error = table["position_standard_error"][row]
        good = np.isfinite(position) & np.isfinite(position_error) & (position_error > 0)
        position = np.where(good, position, 0.0)
        position_error = np.where(good, position_error, np.inf)
        if not good.any():
            position = position_error = None
    deviation = error = None
    if table["line_deviation"] is not None and table["standard_error"] is not None:
        deviation = table["line_deviation"][row]
        error = table["standard_error"][row]
        good = np.isfinite(deviation) & np.isfinite(error) & (error > 0)
        deviation = np.where(good, deviation, 0.0)
        error = np.where(good, error, np.inf)
        if not good.any():
            deviation = error = None
    if position is None and deviation is None:
        return None
    if deviation is None:
        deviation = np.zeros(len(position))
        error = np.full(len(position), np.inf)
    return TimeBasePrior(deviation, error, band, witness, position, position_error)


def time_base_increment(field):
    """The loop's per-line position INCREMENT for this field, in RF samples.

    The value carries no reference of its own to reconcile against the
    sync-derived line locations: it is simply added to the locations the
    decode would otherwise use, after every other refinement. Positive =
    move the line later. NaN = nothing to add. Nothing here scales it.
    None without a file, a record, or an increment array.

    THIS CONSUMER IS STATELESS, WHICH FIXES WHAT THE VALUE MUST BE. Every
    decode begins from the same sync-derived locations and applies exactly
    what the file says, remembering nothing of any previous pass. So the
    file must always carry the TOTAL displacement wanted from the base,
    never a delta on top of what a previous pass applied - the two coincide
    only for a loop that accumulates a correction and re-derives it each
    pass, and diverge for one that accumulates a model and reports its
    departure. A writer that halves its step per pass expresses that in the
    total it writes, not in an expectation that this side remembers.
    """
    table = _time_base_response(field.rf)
    if table is None or table.get("position_increment") is None:
        return None
    readloc = int(getattr(field, "readloc", -1))
    hits = np.where(table["readloc"] == readloc)[0]
    if len(hits) == 0:
        return None
    row = hits[-1]
    increment = table["position_increment"][row]
    if not np.isfinite(increment).any():
        return None
    scale = _period_samples(field) or 0.0
    error = table.get("position_standard_error")
    error = None if error is None else np.asarray(error[row], dtype=np.float64) * scale
    return np.where(np.isfinite(increment), increment, 0.0) * scale, error


def apply_time_base_increment(field, linelocs):
    """`linelocs` plus the loop's correction where one exists, else unchanged.

    Applied unscaled and unfiltered by contract: the writing loop owns the
    regulation. What this side owes in return is a gauge of what it was
    handed, because the risk of fitting a per-line correction to a per-line
    measurement is that the measurement's own noise is absorbed into the
    geometry - the correction then drives its gauge's residual below that
    gauge's noise while writing line-to-line jitter into the picture. The
    signature is the correction's content ABOVE the flutter band, where no
    mechanical process puts anything: a time base error is slow, so a
    correction carrying line-to-line structure is carrying noise. Both are
    reported for the writer to read back.
    """
    fetched = time_base_increment(field)
    if fetched is None:
        return linelocs
    increment, error = fetched
    refined = np.array(linelocs, dtype=np.float64)
    m = min(len(increment), len(refined))
    refined[:m] += increment[:m]
    applied = increment[:m]
    window = _drum_window(field)
    slow = _band_smooth(applied, window)
    fast = applied - np.where(np.isfinite(slow), slow, 0.0)
    slow_rms = float(np.sqrt(np.nanmean(np.where(np.isfinite(slow), slow, np.nan) ** 2)))
    fast_rms = float(np.sqrt(np.mean(fast ** 2)))
    field.rf.__dict__["_time_base_increment_rms"] = float(np.sqrt(np.mean(applied ** 2)))
    field.rf.__dict__["_time_base_increment_split"] = {
        "flutter_band_rms": slow_rms,
        "above_band_rms": fast_rms,
        "window_lines": int(window),
        "declared_band_limit_lines": _declared_band_limit(_time_base_response(field.rf)),
        "declared_regulator": _declared_regulator(_time_base_response(field.rf)),
    }
    # What must not be written into the geometry is the measuring gauge's
    # own noise. The test for that is EVIDENCE, not bandwidth: a line's
    # placement genuinely can change from its neighbour's, because the sync
    # time base positions every line from its own measurement and that
    # measurement carries per-line error, so fast structure may be a real
    # placement error rather than noise. What distinguishes them is whether
    # the correction exceeds the error of the measurement that produced it.
    # Reported per decode, and said aloud only when the correction as a
    # whole sits at or below its own error - which is the case where
    # applying it writes noise whatever its band.
    chi = None
    if error is not None and np.isfinite(error).any():
        good = np.isfinite(error) & (error > 0) & np.isfinite(applied)
        if good.any():
            chi = float(np.mean((applied[good] / error[good]) ** 2))
    field.rf.__dict__["_time_base_increment_split"]["evidence_chi_square"] = chi
    if (
        chi is not None
        and chi <= 1.0
        and not field.rf.__dict__.get("_time_base_increment_warned")
    ):
        field.rf.__dict__["_time_base_increment_warned"] = True
        _log().warning(
            "carrier_tbc: the time base correction is at or below the error of the "
            "measurement that produced it (mean square %.2f of its own variance) - "
            "applying it writes that measurement's noise into the line geometry; "
            "applied as sent",
            chi)
    return refined


def _period_samples(field):
    """The nominal line period in RF samples."""
    return float(field.rf.linelen) if getattr(field.rf, "linelen", 0) else None


def refine_linelocs(field):
    """Refine this field's final linelocs with the carrier's sync-edge trace.

    Returns a float64 array of the same length as ``field.linelocs`` with
    the flutter-band deviation correction applied per line (scaled by the
    ``--carrier_tbc`` amount), or None when the trace is absent or too
    sparse to trust - in which case the caller keeps the sync-derived time
    base untouched.
    """
    trace = luma_amplitude.sync_edge_trace(field)
    if trace is None:
        return None
    edges, _period = trace

    linelocs = np.asarray(field.linelocs, dtype=np.float64)
    n = min(len(edges), len(linelocs))
    if n < 3:
        return None
    edge_view = np.asarray(edges[:n], dtype=np.float64)
    base = linelocs[:n]

    # -1 marks lines where the trace found no clean crossing (the
    # vsync/equalizing rows chief among them).  Those lines keep their
    # sync-derived location.
    valid = edge_view > 0
    count = int(np.count_nonzero(valid))

    # Usable-trace policy: the trace must describe the field, not a corner
    # of it.  A majority of the field's counted lines is the weakest claim
    # under which the median delta below is still guaranteed to be set by
    # genuine measurements rather than by whatever the invalid rows read -
    # a median tolerates anything short of half its population being wrong.
    linecount = int(getattr(field, "linecount", 0) or 0) or n
    if count < linecount / 2.0:
        return None

    deltas = edge_view[valid] - base[valid]
    # The two conventions' systematic offset (burst lock, subcarrier phase
    # calibration - see the module docstring).  Preserved, never applied:
    # only deviations about it correct anything.
    median = float(np.median(deltas))

    # Quality gate, from the trace's own jitter statistics.  The deltas'
    # robust scale is 1.4826*MAD (the MAD-to-sigma factor for a normal
    # core); over N honest draws from that core the largest |deviation|
    # expected is about sigma*sqrt(2*ln(2N)) (the two-sided extreme-value
    # bound), so a tolerance at that bound keeps essentially every honest
    # line while anything beyond it - a serration edge latched in the
    # vsync region, a dropout crossing the sync tip - is an outlier with
    # high confidence and keeps its sync-derived location instead.
    sigma = 1.4826 * float(np.median(np.abs(deltas - median)))
    if sigma <= 0.0:
        # A degenerate trace (no spread at all over hundreds of float64
        # lines) has nothing to correct and nothing to gate against.
        return None
    tolerance = sigma * np.sqrt(2.0 * np.log(2.0 * count))

    deviation = edge_view - base - median
    gate = valid & (np.abs(deviation) < tolerance)
    if not gate.any():
        return None

    # Restrict the correction to the wow-and-flutter band, where the trace
    # is the proven better instrument (module docstring).  A two-head
    # helical drum lays one field per head pass, so it revolves once per
    # FRAME and the flutter fundamental's period is frame_lines lines; a
    # moving average of frame_lines/(2*pi) lines - the same drum-derived
    # scale as the level adjust's smoothing corner - passes that
    # fundamental nearly whole (a boxcar this short attenuates it under
    # half a dB) while dividing the line-scale estimator disagreement by
    # the square root of its length.  Zero-phase, because a timing
    # correction that lags its flutter would smear what it corrects.
    window = max(int(round(float(field.rf.SysParams["frame_lines"]) / (2.0 * np.pi))), 1)
    kernel = np.ones(window, dtype=np.float64)
    evidence = np.convolve(gate.astype(np.float64), kernel, mode="same")
    smoothed = np.where(
        evidence > 0,
        np.convolve(np.where(gate, deviation, 0.0), kernel, mode="same")
        / np.maximum(evidence, 1.0),
        0.0,
    )
    # The gap-aware average is unbiased but its ends see fewer lines; a
    # residual mean here would shift the field sideways, so the field's
    # alignment is pinned by re-centring over the lines that receive a
    # correction.
    applied = evidence > 0
    smoothed[applied] -= smoothed[applied].mean()

    # Ethan's time-base node: centre the flutter-band deviation on the
    # amplitude channel's prior by inverse variance, where that prior's
    # error beats the trace's own. The trace's error in the band is its
    # ESTIMATOR noise - sigma above is the scale of the disagreement
    # between the two sync instruments on the same lines, which is exactly
    # the estimators' noise (the lines' real jitter is common to both and
    # cancels) - divided by the root of the window; the prior's is its
    # regression residual per line, integrated to a position over the
    # window (root-window errors add). Nothing changes where the prior is
    # absent.
    # Every prior source enters through one inverse-variance combination -
    # the amplitude witness and the offline loop's converged anti-residual
    # compete on the same footing - and a decode with neither is unchanged.
    period_samples = _period_samples(field) or 0.0
    trace_var = (sigma / np.sqrt(window)) ** 2
    precision = np.full(n, 1.0 / trace_var)
    weighted = smoothed / trace_var
    weights = {}
    for source, prior in (
        ("amplitude", time_base_prior(field)),
        ("loop", time_base_response_prior(field)),
    ):
        if prior is None:
            continue
        if prior.line_position is not None:
            # A position witness: smoothed over the window like the trace,
            # its per-line error shrinks by root-window.
            m = min(len(prior.line_position), n)
            position = np.zeros(n)
            position[:m] = prior.line_position[:m] * period_samples
            position = np.where(np.isfinite(position), position, 0.0)
            error = np.asarray(prior.position_standard_error[:m], dtype=np.float64)
            finite = np.isfinite(error) & (error > 0)
            if not finite.any():
                continue
            position = np.convolve(position, kernel, mode="same") / np.maximum(evidence, 1.0)
            position[applied] -= position[applied].mean()
            prior_var = (np.mean(error[finite]) * period_samples) ** 2 / window
        else:
            # A period witness: integrated to positions, its per-line error
            # accumulates over the window as a random walk.
            m = min(len(prior.line_deviation), n - 1)
            position = np.zeros(n)
            position[1 : m + 1] = np.cumsum(prior.line_deviation[:m] * period_samples)
            position[applied] -= position[applied].mean()
            error = np.asarray(prior.standard_error[:m], dtype=np.float64)
            finite = np.isfinite(error) & (error > 0)
            if not finite.any():
                continue
            prior_var = (np.mean(error[finite]) * period_samples) ** 2 * window
        if not (prior_var > 0 and np.isfinite(prior_var)):
            continue
        precision = precision + 1.0 / prior_var
        weighted = weighted + position / prior_var
        weights[source] = float((1.0 / prior_var) / (1.0 / trace_var + 1.0 / prior_var))
    if weights:
        smoothed = np.where(applied, weighted / precision, smoothed)
        field.rf.__dict__["_time_base_prior_weight"] = float(weights.get("amplitude", 0.0))
        field.rf.__dict__["_time_base_prior_weights"] = weights

    # Fractional flag values scale the applied correction; 1 is the whole
    # measured deviation.
    amount = float(field.rf.options.carrier_tbc)

    refined = np.array(linelocs, dtype=np.float64)
    refined[:n] += amount * smoothed
    return refined
