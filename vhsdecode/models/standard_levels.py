"""Normalising the corrected signal so it sits within the video standard.

Ethan: *"after RF correction, I do want to normalize the signal so it sits
within the video standard."*

The standard fixes three levels and the correction must land on them. SMPTE
170M-2004 clause 12.3 and table 1, with ITU-R BT.1700 carrying the same
figures for System M:

| level | value | tolerance |
|---|---|---|
| synchronising tip | **−40 IRE** | ±1 |
| blanking | **0 IRE** | the reference |
| black, System M with setup | +7.5 IRE | ±1 |
| reference white | **+100 IRE** | |

and 140 IRE peak to peak = 1.0 V, so one IRE is 7.143 mV.

TWO ANCHORS, NOT THREE. The map from a measured signal to the standard is
affine - a gain and an offset - so it is determined by two levels and
over-determined by three. Sync tip and blanking are the two to use, for the
reason the whole arc rests on: they are in the reserved intervals, they are
present on every line, and they are content-free, where reference white
appears only when the picture happens to contain it.

MEASURE THE PULSE AT ITS OWN HALF AMPLITUDE, NOT AT A FIXED LEVEL. A
threshold in IRE stops being a width measurement the moment the amplitude
has changed - read at a fixed −20 IRE a decoded sync measured 111 ns narrow,
outside the generator's tolerance, and read at its own half amplitude the
same pulse measured 44 ns, inside it. The same applies to the level anchors:
the tip is where the pulse's own plateau is, not where a nominal number says
it should be.

AND THE ORDER MATTERS, which is why this is a separate step. `--ire0_adjust`
moves the level the decoder treats as zero IRE - by about 36 IRE on one
recording - so a normalisation measured before it and applied after it fights
it. This function takes anchors that the caller has measured on the signal it
is about to normalise, at the point in the chain where the normalisation will
land. It does not go looking for them itself, precisely so that it cannot be
applied at the wrong point.
"""

from typing import Dict, Optional, Sequence

import numpy as np

# SMPTE 170M-2004 table 1 and clause 12.3.
SYNC_TIP_IRE = -40.0
BLANKING_IRE = 0.0
SETUP_IRE = 7.5              # System M; zero for 625-line systems and Japan
REFERENCE_WHITE_IRE = 100.0
IRE_TOLERANCE = 1.0          # on the sync tip and the setup level
VOLTS_PER_IRE = 1.0 / 140.0  # 140 IRE peak to peak is 1.0 V


def affine_to_standard(measured_sync_tip: float,
                       measured_blanking: float) -> Dict[str, float]:
    """The gain and offset that put the two anchors on the standard.

    Returns `y = gain * x + offset` such that the measured sync tip lands on
    −40 IRE and the measured blanking on 0.
    """
    span = float(measured_blanking) - float(measured_sync_tip)
    if not abs(span) > 0:
        raise ValueError("the two anchors are equal, so no gain is defined")
    gain = (BLANKING_IRE - SYNC_TIP_IRE) / span
    offset = BLANKING_IRE - gain * float(measured_blanking)
    return {"gain": float(gain), "offset": float(offset),
            "measured_span": float(span),
            "specified_span": float(BLANKING_IRE - SYNC_TIP_IRE)}


def normalise(signal: np.ndarray, measured_sync_tip: float,
              measured_blanking: float) -> np.ndarray:
    """Put a signal on the standard's levels, given its own two anchors."""
    map_ = affine_to_standard(measured_sync_tip, measured_blanking)
    return (np.asarray(signal, dtype=np.float64) * map_["gain"]
            + map_["offset"])


def compliance(signal: np.ndarray, measured_sync_tip: float,
               measured_blanking: float,
               measured_white: Optional[float] = None,
               setup_ire: float = SETUP_IRE) -> Dict[str, object]:
    """Does the normalised signal sit within the standard?

    The two anchors land on their values by construction, so they prove
    nothing. What the check is FOR is the third level: reference white is
    over-determined once the gain and offset are fixed, so where it lands is
    a measurement of whether the chain's transfer is linear between the sync
    tip and white. A signal whose sync is right and whose white is not has a
    gain error that the two anchors could not see.

    Also reports the excursion, because a correction that lands the levels
    correctly and drives the peaks past the standard's limits has not put
    the signal within the standard: SMPTE 170M clause 12.1 gives 140 IRE
    peak to peak without chroma and clause 12.4 gives 171 IRE with it.
    """
    normalised = normalise(signal, measured_sync_tip, measured_blanking)
    out: Dict[str, object] = {
        "sync_tip_ire": SYNC_TIP_IRE,
        "blanking_ire": BLANKING_IRE,
        "minimum_ire": float(np.min(normalised)),
        "maximum_ire": float(np.max(normalised)),
        "excursion_ire": float(np.ptp(normalised)),
        "within_composite_limit": bool(np.ptp(normalised) <= 171.0),
        "within_luma_limit": bool(np.ptp(normalised) <= 140.0),
        "setup_ire": float(setup_ire),
    }
    out.update(affine_to_standard(measured_sync_tip, measured_blanking))
    if measured_white is not None:
        landed = float(measured_white) * out["gain"] + out["offset"]
        out["white_lands_at_ire"] = landed
        out["white_error_ire"] = landed - REFERENCE_WHITE_IRE
        # a gain error the two anchors cannot see, as a fraction
        out["gain_error"] = ((landed - BLANKING_IRE)
                             / (REFERENCE_WHITE_IRE - BLANKING_IRE) - 1.0)
        out["white_within_tolerance"] = bool(
            abs(landed - REFERENCE_WHITE_IRE) <= 2.0)
    out["why"] = ("the two anchors land by construction and prove nothing; "
                  "reference white is over-determined and is what the check "
                  "is for")
    return out


def half_amplitude_crossings(pulse: np.ndarray,
                             sample_rate_hz: float) -> Dict[str, float]:
    """The pulse's width and rise, read at its OWN half amplitude and with
    sub-sample interpolation.

    Both corrections matter and both were measured. A fixed-level threshold
    read a decoded sync 111 ns narrow, outside the generator's tolerance,
    where its own half amplitude read it 44 ns, inside. And the sample grid
    alone reads a specified 140 ns rise as 150 at 40 MHz, because a sample
    is 25 ns; interpolating between samples on the same edge brings that to
    142.6 ns, and at 50 MHz to 141.8.
    """
    values = np.asarray(pulse, dtype=np.float64).ravel()
    if values.size < 4:
        raise ValueError("too few samples to find an edge")
    # Percentiles directly, not medians of a subset: on a sync pulse the
    # signal sits at blanking for most of its length, so `values >=
    # percentile(75)` can select nothing at all and the median of an empty
    # slice is not a number. The plateaus are where the mass is, and a
    # percentile finds them without a subset that may be empty.
    top = float(np.percentile(values, 90.0))
    bottom = float(np.percentile(values, 5.0))
    depth = bottom - top
    if not abs(depth) > 0:
        raise ValueError("the pulse has no depth to measure")

    def crossing(level, forward=True):
        order = range(1, values.size) if forward else range(values.size - 1,
                                                            0, -1)
        for index in order:
            first, second = values[index - 1], values[index]
            if (first - level) * (second - level) <= 0 and first != second:
                return (index - 1) + (first - level) / (first - second)
        return float("nan")

    half = top + 0.5 * depth
    lead = crossing(half, True)
    trail = crossing(half, False)
    ten = crossing(top + 0.1 * depth, True)
    ninety = crossing(top + 0.9 * depth, True)
    return {
        "top": top,
        "bottom": bottom,
        "depth": depth,
        "width_s": abs(trail - lead) / float(sample_rate_hz),
        "rise_s": abs(ninety - ten) / float(sample_rate_hz),
        "half_amplitude": half,
        "why": ("read at the pulse's own half amplitude, with sub-sample "
                "interpolation; a fixed level is not a width measurement "
                "once the amplitude has changed"),
    }


# --------------------------------------------------------------------------
# The back porch as the ONLY anchor, decomposed against the burst
# --------------------------------------------------------------------------
#
# Ethan: *"the important thing is that the back porch is centered at 0 ire ...
# The scaling of the video is done in our scaling step in the picture stage.
# The sync pulse when it came into the VCR can be assumed to be relatively
# correct with the picture."*
#
# And: *"When performing the back porch measurement, account for color under
# and color carrier leakage on the back porch. Use the reference burst levels
# as the baseline and all the color information color under and color carrier
# as multiple dimensions to this baseline."*
#
# THIS REPLACES A TWO-LEVEL FIT WITH A ONE-LEVEL ANCHOR, and that is the
# simplification, not an approximation of one. The decoder's `hsync` mode
# derives a SCALE from `(porch - tip)/40`, which needs the tip to be a
# trustworthy level - and the tip is exactly the level a clip destroys, an
# AGC referenced to it normalises the damage away, and a dropout collapses.
# Anchoring only the offset needs one level and the one that no dark clip can
# reach.
#
# The colour-under carrier is 40 f_H and the subcarrier 455/2 f_H, both exact
# multiples of the line rate, so both leak onto the porch at frequencies KNOWN
# IN ADVANCE. They are therefore not noise to be averaged down but dimensions
# to be projected out, and projecting them out is what makes the remaining
# baseline the level rather than a level plus whatever chroma happened to be
# present.
BURST_AMPLITUDE_IRE = 40.0          # peak to peak, SMPTE 170M
COLOUR_UNDER_MULTIPLE = 40.0        # f_cu = 40 f_H exactly (SMPTE 32M)
SUBCARRIER_MULTIPLE = 455.0 / 2.0   # f_sc = 455/2 f_H exactly


def porch_dimensions(samples, sample_rate_hz: float,
                     line_rate_hz: float = 30000.0 / 1001.0 * 525.0,
                     multiples: Optional[Sequence[float]] = None
                     ) -> Dict[str, object]:
    """THE BACK PORCH DECOMPOSED: a baseline plus the chroma that leaks onto it.

    The porch is modelled as

        porch(t) = baseline + sum_k a_k cos(2 pi m_k f_H t + phi_k) + residual

    with the multiples `m_k` KNOWN - 40 for the colour-under carrier and
    455/2 for the subcarrier - so each leakage term is one COMPLEX amplitude
    and not a shape to be fitted. That is what makes this a projection rather
    than a fit: the frequencies are specified, so the design matrix is
    determined before any sample is read and nothing can be absorbed into a
    free frequency.

    THE BASELINE IS THE QUANTITY THAT MATTERS. Averaging the porch gives the
    baseline PLUS whatever the leakage happens to average to over the window,
    which is not zero unless the window is a whole number of cycles of every
    leaking carrier. Projecting the carriers out first removes exactly that
    error, and the returned `baseline` is what should be driven to 0 IRE.

    Each leakage term is returned COMPLEX, because an amplitude without its
    phase cannot say whether two lines' leakage adds or cancels.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    if values.size < 8:
        return {"baseline": float(values.mean()) if values.size else 0.0,
                "why": "too few samples to separate a baseline from a carrier"}
    order = list(multiples if multiples is not None
                 else (COLOUR_UNDER_MULTIPLE, SUBCARRIER_MULTIPLE))
    t = np.arange(values.size, dtype=np.float64) / float(sample_rate_hz)

    columns = [np.ones_like(t)]
    names = ["baseline"]
    for multiple in order:
        frequency = float(multiple) * float(line_rate_hz)
        columns.append(np.cos(2.0 * np.pi * frequency * t))
        columns.append(np.sin(2.0 * np.pi * frequency * t))
        names.append(f"{multiple:g} f_H")
    design = np.column_stack(columns)
    solution, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ solution

    leakage: Dict[str, complex] = {}
    for index, multiple in enumerate(order):
        cosine = solution[1 + 2 * index]
        sine = solution[2 + 2 * index]
        # cos and sin components as one complex amplitude: a e^{j phi}
        leakage[f"{multiple:g} f_H"] = complex(cosine, -sine)
    return {
        "baseline": float(solution[0]),
        "leakage": leakage,
        "leakage_ire": {k: float(abs(v)) for k, v in leakage.items()},
        "residual_rms": float(residual.std()),
        "raw_mean": float(values.mean()),
        "baseline_minus_mean": float(solution[0] - values.mean()),
        "samples": int(values.size),
        "conditioning": float(np.linalg.cond(design)),
        "why": ("the leaking frequencies are exact multiples of the line "
                "rate and so are known before any sample is read; that makes "
                "this a projection and not a fit"),
    }


def burst_referenced_baseline(porch_samples, burst_samples,
                              sample_rate_hz: float,
                              line_rate_hz: float = 30000.0 / 1001.0 * 525.0
                              ) -> Dict[str, object]:
    """THE BASELINE WITH THE BURST AS ITS SCALE REFERENCE.

    Ethan: *"Use the reference burst levels as the baseline."* The burst is
    the one element on the porch whose amplitude the standard fixes - 40 IRE
    peak to peak - so measuring it gives the IRE scale from the same lines the
    baseline is read on, with no dependence on the sync tip at all.

    That is the whole point of using it. The tip is the level a dark clip
    takes, an AGC referenced to the tip hides its own error there, and a
    dropout collapses it; the burst is in the middle of the range, is
    specified, and is present on every line. So the offset comes from the
    porch and the scale from the burst, and neither comes from the tip.
    """
    porch = porch_dimensions(porch_samples, sample_rate_hz, line_rate_hz)
    burst = np.asarray(burst_samples, dtype=np.float64).ravel()
    if burst.size < 8:
        return {**porch, "scale_ire_per_unit": None,
                "why": "too few burst samples to set a scale"}
    # peak to peak of the burst, robustly: the specification states a
    # peak-to-peak amplitude, so that is what is measured
    high = float(np.percentile(burst, 97.5))
    low = float(np.percentile(burst, 2.5))
    span = high - low
    return {
        **porch,
        "burst_span_measured": span,
        "burst_span_specified_ire": BURST_AMPLITUDE_IRE,
        "scale_ire_per_unit": (BURST_AMPLITUDE_IRE / span
                               if span > 0 else None),
        "baseline_ire": (porch["baseline"] * BURST_AMPLITUDE_IRE / span
                         if span > 0 else None),
        "why": ("the burst is specified at 40 IRE peak to peak and sits on "
                "the same lines as the baseline, so it sets the scale "
                "without the sync tip - which is the level a clip takes and "
                "an AGC hides"),
    }


def dominant_trend(values, window: int = 0, reject: float = 3.5
                   ) -> Dict[str, object]:
    """THE LEVEL AS A TREND, NOT AS A PER-FIELD FIT.

    Ethan: *"Don't rely on agc being fully consistent, there will be
    occasional large chunks of noise or invalid sync pulse information there
    that will muddy the measurement. Regardless of this, I think we can base
    the levels on the dominant trend that should be constant given our other
    corrections."*

    So the estimator has to survive CHUNKS and not merely outliers. A median
    rejects isolated points and is defeated by a contiguous run of bad
    fields, which is exactly what a dropout or a stretch of unlocked tape
    produces. This uses the median as the centre and rejects on the median
    absolute deviation - a scale that a contiguous bad run inflates but
    cannot capture, provided the run is a minority - and then reports the
    trend as a slow local median rather than a single number, because "should
    be constant given our other corrections" is a claim to be MEASURED and
    not assumed.

    The returned `flat` says whether it actually is constant: if the trend
    moves by more than the scatter of what survives rejection, the levels are
    still drifting and something upstream is not yet corrected.
    """
    series = np.asarray(values, dtype=np.float64).ravel()
    if series.size < 4:
        return {"level": float(series.mean()) if series.size else 0.0,
                "flat": True, "why": "too few points to see a trend"}
    centre = float(np.median(series))
    deviation = np.abs(series - centre)
    scale = float(np.median(deviation)) * 1.4826
    keep = deviation <= max(float(reject) * scale, 1e-12)
    if keep.sum() < 4:
        keep = np.ones_like(series, dtype=bool)
    width = int(window) if window else max(9, series.size // 16 | 1)
    width = min(width | 1, series.size if series.size % 2 else series.size - 1)
    # THE TREND IS TAKEN ON THE SURVIVORS, NOT ON THE RAW SERIES, and this is
    # the difference between an estimator that survives a chunk and one that
    # merely survives outliers. A local median over the raw series FOLLOWS a
    # bad run that is longer than its own window - measured, a 40-field
    # excursion of +45 through a 25-field window reported a trend swing of
    # 45.07 and declared a perfectly flat level "not flat". Replacing the
    # rejected points by interpolation from their neighbours first leaves the
    # trend reading the level either side of the chunk, which is what
    # "dominant trend" means.
    cleaned = series.copy()
    if (~keep).any() and keep.any():
        index = np.arange(series.size, dtype=np.float64)
        cleaned[~keep] = np.interp(index[~keep], index[keep], series[keep])
    padded = np.pad(cleaned, width // 2, mode="edge")
    trend = np.array([np.median(padded[i:i + width])
                      for i in range(series.size)])
    survivors = series[keep]
    spread = float(survivors.std())
    swing = float(trend.max() - trend.min())
    return {
        "level": float(np.median(survivors)),
        "trend": trend,
        "rejected_fraction": float(1.0 - keep.mean()),
        "scatter": spread,
        "trend_swing": swing,
        "flat": bool(swing <= max(spread, 1e-12)),
        "window": width,
        "why": ("a median rejects isolated points and a contiguous bad run "
                "defeats it, so the scale comes from the MAD and the trend "
                "is reported rather than assumed constant"),
    }
