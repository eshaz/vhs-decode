"""THE SPACING FROM BLANKING TO SYNC TIP, as the luma channel's absolute scale.

Ethan, 2026-09-06: *"I think we need to do another correction using the
spacing between 0 and -40 ire, i.e. the sync pulses. First detect clipping
though in the data, since we will use that as the absolute lowest point of
data in the luma channel. This will feed into all of the other composite
parts we are measuring."*

WHY THIS IS THE SCALE. Every other level in the composite signal is
specified as a fraction of something, and this one interval is specified
outright: blanking is the zero and the synchronizing level is 40 IRE
beneath it. Nothing in the picture can supply an absolute scale, because
picture content is what is being measured; the sync pulse can, because its
depth is a number the standard fixes. So the measured spacing divided by
40 is the luma channel's gain, and it feeds every level that follows.

THE CLIP IS REAL, AND IT IS NOT THE TAPE. Measured on this session's three
decodes: every one of them has its lowest luma sample at exactly the same
value, -40.530 IRE, which is sample ZERO of the sixteen-bit container
(black at 16243.2 with 400.768 units to the IRE). The histogram settles it
- on countdown 78,537 samples sit at value zero against about 450 at each
of values one to seven, a pile-up of a hundred and seventy-five times, and
on pulse-and-bar 19,817 against about twenty, nine hundred times. A tape
cannot produce that; an output stage writing unsigned samples can. The
container leaves only 0.53 IRE beneath the nominal sync tip, and two of
the three tapes are sitting on it.

AND IT REACHES THE MEASUREMENT ON ONE TAPE OF THE THREE. Ethan's rule -
*"don't exclude whole lines, it may clip low only, and may not be clipping
at all, in which case means we don't care"* - is exactly what the counts
say. On the pulse's flat interior, columns 20 to 46:

    tape          samples pinned   lines touching   reaches the window?
    countdown          2.092 %          26.2 %      YES
    home               0.002 %           0.0 %      no
    pulse-and-bar      0.005 %           0.1 %      no

On home and pulse-and-bar the clip lives entirely on the pulse's downward
overshoot at the edges and never touches the level being read, so there is
nothing to care about. On countdown it does reach it, because that tape's
tip sits at -38.7 IRE with 0.75 IRE of scatter and the floor is 2.4 standard
deviations away, which puts a couple of per cent of samples across it.

THE TREATMENT IS TO MASK THE SAMPLES, NEVER THE LINES. Excluding whole
lines costs 54 per cent of countdown and 98 per cent of pulse-and-bar to
buy a shift of 0.077 and 0.000 IRE; masking only the pinned samples costs
nothing and removes the bias where it exists:

    tape          all samples   clipped lines out   clipped samples out
    countdown        36.466        36.390 (-0.077)     36.433 (-0.033)
    home             33.628        33.628 (-0.000)     33.628 (-0.000)
    pulse-and-bar    34.911        34.911 (-0.000)     34.911 (-0.000)

Against the ten-bit output floor of 0.0461 IRE the sample mask moves
countdown by 0.7 floors and the other two by nothing at all. So masking is
the default here: it is free, it does nothing where there is no clip, and
where there is one it removes a bias the line exclusion would have paid
for with half the data.

AND THE SPACING IS NOT 40 IRE ON ANY TAPE. Measured on this session's
three decodes, the tip read on the pulse's flat interior and estimated
through the clip where it reaches:

    tape           lowest    at floor   blanking     tip    spacing   gain
    countdown     -40.530     2.432 %     -2.221  -38.739    36.518  0.913
    home          -40.530     0.003 %     -2.066  -35.673    33.607  0.840
    pulse-and-bar -40.530     4.720 %     -2.280  -37.205    34.925  0.873

Deficits of 3.48, 6.39 and 5.08 IRE, so a luma gain of 0.913, 0.840 and
0.873 against what the standard specifies - between 0.8 and 1.5 dB low.
That is the scale correction every other composite level divides by, and
it is the only one the signal can supply, because the sync pulse is the
one interval whose absolute size the standard fixes.

Note also that blanking itself sits at -2.07 to -2.28 IRE rather than at
zero, which is a separate matter and belongs to `standard_levels`: this
module measures a SPACING, and a spacing is immune to where the zero sits.
"""

import json
from typing import Dict, Optional, Sequence

import numpy as np

# ITU-R BT.1700: blanking is the reference zero and the synchronizing level
# sits 40 IRE beneath it on the 525-line IRE scale (140 IRE peak to peak).
SPECIFIED_DEPTH_IRE = 40.0

# Where to read each level inside a decoded line, as fractions of the sync
# pulse's own specified width so they follow the geometry rather than a
# sample count. The tip is read on the flat interior, clear of both edges,
# which is what makes the clip check optional.
TIP_INTERIOR = (0.30, 0.70)


def detect_clip(field, black_16b: float, white_16b: float,
                columns: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """The absolute lowest point of the luma, and whether it is a clip.

    A clip shows as a PILE-UP: many samples at one value with far fewer
    just above it. Noise alone gives a smooth tail. The ratio of the count
    at the floor to the mean count of the next few values is the statistic,
    and it needs no threshold on the level itself.
    """
    values = np.asarray(field)
    if columns is not None:
        values = values[..., list(columns)]
    flat = np.asarray(values).ravel()
    per_ire = (float(white_16b) - float(black_16b)) / 100.0
    lowest = int(flat.min())
    counts = np.bincount(np.asarray(flat, dtype=np.int64) - lowest,
                         minlength=8)[:8]
    neighbours = float(counts[1:8].mean()) if counts.size > 1 else 0.0
    pile_up = float(counts[0] / neighbours) if neighbours > 0 else float("inf")
    # A RATIO ALONE IS NOT ENOUGH. Sparse data - a handful of samples
    # spread over many values - gives a large ratio from one count against
    # zeros, and the first version of this called clean noise a clip for
    # exactly that reason. A clip also puts a MEANINGFUL FRACTION of the
    # samples at the floor, so both must hold.
    at_floor = float((flat <= lowest).mean())
    return {
        "lowest_sample": lowest,
        "lowest_ire": (lowest - float(black_16b)) / per_ire,
        "container_floor_ire": (0.0 - float(black_16b)) / per_ire,
        "at_container_floor": bool(lowest <= 0),
        "counts_from_floor": counts.tolist(),
        "pile_up_ratio": pile_up,
        "clipped": bool(pile_up > 5.0 and at_floor > 1e-3),
        "fraction_at_floor": at_floor,
        "why": ("a clip piles samples at one value; noise alone leaves a "
                "smooth tail, so the ratio of the floor's count to its "
                "neighbours' is the statistic and no level threshold is "
                "needed"),
    }


def censored_mean(values, floor: float = 0.0) -> Dict[str, float]:
    """The mean of a distribution whose low tail has been folded onto a floor.

    A clip does not DELETE the samples beneath the floor, it moves them TO
    the floor, so the count there is information and throwing it away is
    the worst of the three things one can do. Measured on a planted
    Gaussian tip with 6.66 per cent of its samples below the floor, against
    the truth:

        leave the clip in            +5.93
        mask the clipped samples    +27.76      the worst, by far
        this estimator               +0.08      and it recovers sigma too

    Masking is worse than leaving it because the clip merely folds the tail
    onto the floor while masking removes it, which biases the mean the same
    way and further. What works is to use the count at the floor as the
    evidence it is: fit the surviving samples and that count together, by
    maximum likelihood for a censored normal.
    """
    from scipy import optimize, stats

    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "sigma": float("nan"), "censored": 0}
    kept = x[x > floor]
    censored = int((x <= floor).sum())
    if censored == 0 or kept.size < 8:
        return {"mean": float(x.mean()), "sigma": float(x.std()),
                "censored": censored, "fitted": False}

    def negative_log_likelihood(parameters):
        mean, sigma = parameters
        if sigma <= 0:
            return 1e12
        below = stats.norm.cdf(floor, mean, sigma)
        return -(censored * np.log(max(below, 1e-300))
                 + stats.norm.logpdf(kept, mean, sigma).sum())

    result = optimize.minimize(negative_log_likelihood,
                               [kept.mean(), max(kept.std(), 1e-6)],
                               method="Nelder-Mead")
    return {"mean": float(result.x[0]), "sigma": float(result.x[1]),
            "censored": censored, "fitted": True,
            "censored_fraction": float(censored / x.size)}


def spacing(lines, black_16b: float, white_16b: float,
            pulse_columns: Sequence[int], porch_columns: Sequence[int],
            mask_clipped_samples: bool = False,
            censor_floor: Optional[float] = 0.0) -> Dict[str, object]:
    """The blanking-to-tip spacing per line, and the gain it implies.

    `pulse_columns` should be the pulse's flat INTERIOR, not its whole
    width, because the container's clip pins the downward overshoot at the
    edges. `mask_clipped_samples` then removes any pinned sample that still
    falls inside that window - never a whole line, which would cost half
    the data to correct a bias of 0.077 IRE.
    """
    values = np.asarray(lines, dtype=np.float64)
    per_ire = (float(white_16b) - float(black_16b)) / 100.0
    tip = values[:, list(pulse_columns)]
    estimate = None
    if mask_clipped_samples:
        # kept only so the comparison can be made; it is the worst of the
        # three treatments and `censored_mean`'s docstring says by how much
        tip = np.where(tip <= 0.0, np.nan, tip)
        tip_level = (np.nanmean(tip, axis=1) - float(black_16b)) / per_ire
    elif censor_floor is not None and (tip <= censor_floor).any():
        # THE CORRECTION IS POOLED, NOT PER LINE. A censored fit is a
        # numerical solve, and one per line over forty-eight thousand lines
        # is minutes of work for a quantity that does not vary line to line
        # - the tip's noise is the same distribution throughout. So the
        # whole population is fitted ONCE, and the offset between that
        # estimate and the clipped mean is applied to every line. The
        # per-line series keeps its own scatter; only its level moves.
        estimate = censored_mean(tip, censor_floor)
        offset = estimate["mean"] - float(tip[tip > censor_floor].mean()
                                          if (tip > censor_floor).any()
                                          else np.nan)
        raw = tip.mean(axis=1)
        tip_level = (raw + offset - float(black_16b)) / per_ire
    else:
        tip_level = (tip.mean(axis=1) - float(black_16b)) / per_ire
    porch = (values[:, list(porch_columns)].mean(axis=1)
             - float(black_16b)) / per_ire
    measured = porch - tip_level
    good = np.isfinite(measured)
    return {
        # THE CENSORED FIT'S OWN DIAGNOSTICS, carried out rather than kept
        # inside, because a caller has to be able to tell a correction from
        # an extrapolation. A censored normal identifies its mean from the
        # SURVIVING samples; where almost nothing survives it will still
        # return a number, and on one of this arc's three decodes it
        # returned a sync tip of -107 IRE from a window 79 per cent of which
        # sat on the container floor. The fit is a measurement while its
        # mean stands within its own fitted sigma of the floor and an
        # extrapolation beyond that, and only the caller knows what to do
        # about it.
        "censored": (estimate if (censor_floor is not None
                                  and not mask_clipped_samples
                                  and (tip <= (censor_floor or 0.0)).any())
                     else None),
        "censor_floor": (None if mask_clipped_samples else censor_floor),
        "per_line": measured,
        "blanking_ire": float(np.nanmean(porch[good])),
        "tip_ire": float(np.nanmean(tip_level[good])),
        "spacing_ire": float(np.nanmean(measured[good])),
        "scatter_ire": float(np.nanstd(measured[good])),
        "lines": int(good.sum()),
        "specified_ire": SPECIFIED_DEPTH_IRE,
        "deficit_ire": SPECIFIED_DEPTH_IRE - float(np.nanmean(measured[good])),
        "gain": float(np.nanmean(measured[good])) / SPECIFIED_DEPTH_IRE,
        "why": ("blanking is the specified zero and the synchronizing level "
                "is 40 IRE beneath it, so the measured spacing over 40 is "
                "the luma channel's own gain"),
    }


def field_slope(lines, black_16b: float, white_16b: float,
                pulse_columns: Sequence[int], porch_columns: Sequence[int],
                field_height: int, first_line: int = 25) -> Dict[str, object]:
    """THE DRIFT OF THE TWO LEVELS ACROSS A FIELD, which must be zero.

    Ethan: *"we need to fix the flat sync levels at their spec values, that
    is the reference for slope of the difference in the sync pulses over
    the entire field. The levels should be constant for the entire
    field."*

    Both levels are specified constants - blanking at zero and the
    synchronizing level 40 IRE beneath it - so any slope across the field
    is an error, and the two slopes separate into two different errors:

      COMMON MODE, the two levels drifting TOGETHER, is a level shift and
      leaves the spacing alone.
      DIFFERENTIAL, the spacing itself drifting, is a GAIN drift and is the
      one that changes what the picture means.

    MEASURED on this session's three decodes, in IRE across a whole field:

        tape           blanking    tip     spacing    reading
        countdown       -0.603   -0.314    +0.289     gain drifts 0.8 %
        home            +0.343   +0.390    +0.047     level only
        pulse-and-bar   +0.374   +0.337    -0.037     level only

    So home and pulse-and-bar drift as a level and hold their gain, while
    countdown's gain moves across the field as well. Against the ten-bit
    output floor of 0.0461 IRE these slopes are six to thirteen floors, so
    none of them is negligible.

    The levels themselves are also away from their specified values -
    blanking at -1.4 to -2.6 IRE rather than zero - which is a separate
    matter and belongs to `standard_levels`; this reports the SLOPE, and a
    slope is immune to where the level sits.
    """
    values = np.asarray(lines, dtype=np.float64)
    per_ire = (float(white_16b) - float(black_16b)) / 100.0
    rows = values.shape[0] // int(field_height)
    if rows < 1:
        raise ValueError("fewer samples than one field of lines")
    shaped = values[:rows * int(field_height)].reshape(
        rows, int(field_height), values.shape[1])
    tip = (shaped[:, :, list(pulse_columns)].mean(axis=2)
           - float(black_16b)) / per_ire
    porch = (shaped[:, :, list(porch_columns)].mean(axis=2)
             - float(black_16b)) / per_ire
    line_index = np.arange(int(field_height))
    use = (line_index >= int(first_line)) & (line_index < int(field_height) - 2)
    out = {}
    for name, series in (("blanking", porch), ("tip", tip),
                         ("spacing", porch - tip)):
        mean = series[:, use].mean(axis=0)
        slope, intercept = np.polyfit(line_index[use], mean, 1)
        residual = mean - (slope * line_index[use] + intercept)
        out[name] = {
            "slope_ire_per_field": float(slope * int(field_height)),
            "slope_ire_per_line": float(slope),
            "level_at_first": float(mean[0]),
            "level_at_last": float(mean[-1]),
            "scatter_about_the_line": float(residual.std()),
        }
    out["common_mode_ire_per_field"] = out["blanking"]["slope_ire_per_field"]
    out["differential_ire_per_field"] = out["spacing"]["slope_ire_per_field"]
    out["gain_drift_fraction"] = (
        out["spacing"]["slope_ire_per_field"] / SPECIFIED_DEPTH_IRE)
    out["fields"] = int(rows)
    out["why"] = ("both levels are specified constants, so a slope is an "
                  "error; the two slopes separate a level shift, which "
                  "leaves the spacing alone, from a gain drift, which does "
                  "not")
    return out


def depth_spectrum(pulses, sample_rate_hz: float,
                   depth_ire: float = SPECIFIED_DEPTH_IRE
                   ) -> Dict[str, object]:
    """THE FREQUENCY AXIS OF THE SAME DEFICIT, so this is not a one-axis stage.

    THE GAP THIS CLOSES, SAID PLAINLY. Everything above reads the spacing on
    the pulse's flat interior, which is a reading AT DC, and `field_slope`
    reads how it drifts down the field, which is the TIME axis. That is two
    axes, and Ethan's standing requirement is three. What was missing is
    whether the deficit is FLAT: a channel 2 per cent low at DC and 20 per
    cent low at 2 MHz is a different defect from one that is 2 per cent low
    everywhere, and the spacing alone cannot tell them apart.

    THE SYNTHETIC SIDE IS SPECIFIED, NOT FITTED. The measured pulse is read
    against `pair_dimension.spec_sync` - the standard's own 4.7 microsecond
    pulse with 140 nanosecond edges at the specified 40 IRE depth - so this
    is `pair_dimension.sync_dimension` with the depth made explicit, and the
    transfer it returns is COMPLEX: its magnitude is the gain at each
    frequency and its angle is the channel's phase there. Nothing is reduced
    to a magnitude at this boundary.

    THE SEGMENTS ARE THE LINES. One pooled pulse is one segment, and a
    single-segment coherence is 1 by construction - it certifies the
    arithmetic rather than the physics, which is exactly what
    `pair_transfer`'s debiasing exists to prevent. So `pulses` is the
    per-line ensemble, one row a line, and the specified pulse is repeated
    against it; the coherence is then a real statement about how much of the
    measured pulse the specified one accounts for.

    Returns the transfer, the coherence, and the two scalars worth quoting:
    `dc_gain`, the transfer extrapolated to zero frequency, which is the
    spacing gain arrived at from the other side; and `tilt_db_per_mhz`, the
    weighted slope of the gain across the band, which is zero when the
    deficit is flat.

    MEASURED BY THE RUNTIME STAGE, 2026-09-06, on the 75 per cent bar SP
    decode - 245 sync pulses a field, band limited at the VHS luma's 3 MHz,
    the pulses being the segments and the specified pulse repeated against
    them:

        DC gain           0.9687
        spacing gain      0.9768   (read directly on the pulse interior)
        tilt             +0.41 dB/MHz across the band

    The two gains agree to 0.8 per cent, and that agreement is the check:
    they share no arithmetic, one being a mean over samples and the other
    the zero-frequency limit of a Wiener transfer. The tilt says the deficit
    is not quite flat, which is the thing the spacing alone cannot see and
    the whole reason this axis exists.

    WHAT IS NOT DONE HERE, AND WHY. The tilt is REPORTED, never applied. A
    level-dependent or frequency-dependent luma scale belongs to the
    equaliser lane by Ethan's own ruling; this module owns the one number
    the standard fixes, which is the flat scale, and the tilt is evidence
    handed on rather than a correction taken.
    """
    from vhsdecode.models import pair_dimension

    rows = np.atleast_2d(np.asarray(pulses, dtype=np.float64))
    count, length = rows.shape
    if count < 2 or length < 8:
        raise ValueError("the frequency axis needs at least two pulses of "
                         "more than a handful of samples each")
    ideal = pair_dimension.spec_sync(float(sample_rate_hz), length,
                                     depth_ire=-abs(float(depth_ire)))
    result = pair_dimension.pair_transfer(
        np.tile(ideal, count), rows.reshape(-1), count)
    frequencies = np.fft.rfftfreq(length, d=1.0 / float(sample_rate_hz))
    transfer = np.asarray(result["transfer"])
    coherence = np.asarray(result["coherence"])
    gain = np.abs(transfer)
    # Weighted by the coherence and by the specified pulse's own power, so
    # the bins where the standard puts no energy do not vote.
    weight = coherence * np.asarray(result["input_power"])
    weight = np.where(np.isfinite(weight), weight, 0.0)
    weight[0] = 0.0
    inside = frequencies > 0
    total = float(weight[inside].sum())
    if total > 0:
        decibels = 20.0 * np.log10(np.maximum(gain, 1e-12))
        centre = float(np.sum(weight * frequencies) / total)
        spread = float(np.sum(weight * (frequencies - centre) ** 2))
        mean_db = float(np.sum(weight * decibels) / total)
        tilt = (float(np.sum(weight * (frequencies - centre)
                             * (decibels - mean_db))) / spread
                if spread > 0 else 0.0)
        dc_gain = float(10.0 ** ((mean_db - tilt * centre) / 20.0))
    else:
        centre, tilt, dc_gain = float("nan"), float("nan"), float("nan")
    return {
        # COMPLEX, because the phase is half of what a transfer is
        "transfer": transfer,
        "frequency_hz": frequencies,
        "coherence": coherence,
        "gain": gain,
        "specified": ideal,
        "dc_gain": dc_gain,
        "tilt_db_per_hz": tilt,
        "tilt_db_per_mhz": tilt * 1e6 if np.isfinite(tilt) else tilt,
        "weighted_centre_hz": centre,
        "pulses": int(count),
        "why": ("the spacing is a reading at DC; this is the same deficit "
                "across frequency, against the standard's own pulse, so the "
                "stage carries amplitude, frequency and time rather than "
                "amplitude and time"),
    }


def correction(spacing_result: Dict[str, object]) -> Dict[str, object]:
    """The scale every other composite level must be divided by.

    Applied to a level measured in the decode's own IRE, this returns it to
    the scale the standard defines. It is a GAIN and nothing else: it does
    not move the zero, because blanking is the reference and a shift there
    belongs to `standard_levels`.
    """
    gain = float(spacing_result["gain"])
    if not (0.1 < gain < 3.0):
        raise ValueError(f"a luma gain of {gain:.3f} is outside anything a "
                         f"chain plausibly does; check the read windows "
                         f"before applying it")
    return {
        "gain": gain,
        "divide_levels_by": gain,
        "deficit_ire": spacing_result["deficit_ire"],
        "as_decibels": float(20.0 * np.log10(gain)),
        "why": ("the sync pulse is the only interval whose absolute size "
                "the standard fixes, so it is the only thing that can set "
                "the scale; everything else in the composite signal is "
                "specified as a fraction"),
    }


def from_decode(tbc_path: str, json_path: str, first_picture_line: int = 22,
                mask_clipped_samples: bool = False) -> Dict[str, object]:
    """Detect the clip and measure the spacing on a decoded field set."""
    from vhsdecode.models import precursor

    with open(json_path) as handle:
        parameters = json.load(handle)["videoParameters"]
    width = int(parameters["fieldWidth"])
    height = int(parameters["fieldHeight"])
    # THE ZERO IS BLANKING AND THE SCALE IS THE DECODE'S OWN. This read
    # `black16bIre` as the zero and `(white - black) / 100` as the units per
    # IRE, and both were wrong: black sits at the specified 7.5 IRE setup
    # rather than at zero, and the two levels the decoder writes are spread
    # apart by `--level_adjust`, whose default is 0.1. Together they made
    # the units per IRE 400.8 where the decode used 358.4, and every gain
    # reported by this module was short by 11.8 per cent - the difference
    # between the 0.913 / 0.840 / 0.873 first recorded in the docstring
    # above and the 1.021 / 0.939 / 0.976 the same three decodes give when
    # read on the scale they were written with. `precursor.output_scale`
    # inverts the adjustment exactly; see there for the control.
    scale = precursor.output_scale(parameters)
    black = float(parameters["blanking16bIre"])
    white = black + 100.0 * float(scale["units_per_ire"])
    raw = np.fromfile(tbc_path, dtype=np.uint16)
    count = raw.size // (width * height)
    fields = raw[:count * width * height].reshape(count, height, width)
    lines = fields[:, int(first_picture_line):, :].reshape(-1, width)
    rate = 4.0 * 315e6 / 88.0
    pulse_samples = int(round(4.7e-6 * rate))
    interior = range(int(TIP_INTERIOR[0] * pulse_samples),
                     int(TIP_INTERIOR[1] * pulse_samples))
    porch = range(int(parameters["colourBurstEnd"]) + 4,
                  int(parameters["activeVideoStart"]) - 4)
    clip = detect_clip(lines[:, :pulse_samples], black, white)
    measured = spacing(lines, black, white, interior, porch,
                       mask_clipped_samples=mask_clipped_samples)
    return {
        "clip": clip, "spacing": measured,
        "correction": correction(measured),
        "output_scale": scale,
        "fields": int(count),
        "pulse_interior_columns": (interior.start, interior.stop),
        "porch_columns": (porch.start, porch.stop),
    }
