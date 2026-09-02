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

import numpy as np

from vhsdecode import luma_amplitude


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

    # Fractional flag values scale the applied correction; 1 is the whole
    # measured deviation.
    amount = float(field.rf.options.carrier_tbc)

    refined = np.array(linelocs, dtype=np.float64)
    refined[:n] += amount * smoothed
    return refined
