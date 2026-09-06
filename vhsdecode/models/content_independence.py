"""The content-independence check: does a sync-derived series carry the
picture's statistics?

Ethan: *"Content-independence check: compute variance on blanking lines and
test-pattern segments, compare against full-field. Mismatch quantifies
picture-statistics bleed-through."*

THE RULE THIS CHECKS. Every correction in this arc derives from the sync
pulse and the reserved intervals, never from the active picture. That rule
is easy to state and easy to break without noticing: the front porch was
used as a level anchor for weeks before a regression on the preceding
line's content showed it moving at 41 to 66 sigma (-0.21 IRE per IRE of
preceding content, 2.3 IRE of swing over the picture's range - the porch
finding of 2026-09-03, `porch_anchor`). Nothing about that failure was
specific to the front porch. Any quantity measured near the picture can
inherit the picture, and the only way to know is to measure the SAME
quantity where there is no picture and compare.

THE INSTRUMENT. The field itself supplies the control population. ITU-R
BT.1700 Table 3 row j blanks the first 20 lines of every 525-line field
(20H + 1.5 us, docs/BT1700_COLLAPSE.md section 1.2), and the format's own
geometry fills the first nine of them with the equalizing and field-sync
sequences (rows l, m, n: 3H + 3H + 3H, carried in
`sync_geometry.SEQUENCE_LINES`). Lines 10 to 20 therefore carry a normal
line sync and a normal back porch with NO PICTURE on them and NO PICTURE ON
THE LINE BEFORE THEM. Whatever a sync-derived series does on those lines is
what it does in the absence of picture; whatever MORE it does on the picture
lines is either picture or something else that differs between the two
populations, and the check is built to tell those apart. The line numbers
come from the specification and never from the waveform: a population
chosen by looking at the signal would be selected on the very quantity
under test. Lines that a broadcast source fills with inserted test signals
are MEASURED out of the reference population (`occupied_lines`), because
they carry a picture of their own - on the countdown capture lines 15 to 20
carry 6 to 35 IRE RMS of inserted signal and on the home capture lines 19
and 20 carry 35 and 23 IRE RMS (measured 2026-09-05).

THREE POPULATIONS, AS ETHAN NAMED THEM, and what each is for:

    blanking      lines 10-20 less any measured-occupied line: the floor.
    test pattern  picture lines during fields whose content is static (the
                  bars everywhere; the bounce on its plateaus, found by
                  `plateau_fields` from the per-field level and the
                  blank lines' own level noise). On a static pattern the
                  content is identical on every line and every field, so
                  it cannot ADD line-to-line variance; whatever excess this
                  population still shows is NOT picture statistics, it is
                  the instrument's own floor for that series - drift along
                  the field, or a noise level that depends on position in
                  the head's sweep. That makes the static pattern the
                  CONTROL, and `control_attribution` reads every other
                  capture's excess against it.
    full field    every normal-sync line the check can measure.

TWO CHANNELS carry the answer, because static and varying content show up
in different places:

  1. THE VARIANCE CHANNEL (Ethan's statistic). After the per-field term and
     the per-line-index term are removed, the residual variance on a test
     population against the residual variance on the blanking lines, in
     decibels, against what the sample counts alone predict. Content that
     varies from line to line or field to field can only ADD variance.

  2. THE LEVEL CHANNEL. The per-line-index means on the test lines minus
     those on the blanking lines. A static pattern's bleed-through is a
     constant offset between the two populations, which the variance
     channel cannot see and this one can. It is confounded with everything
     else that is fixed to the line's position in the field - the
     low-frequency recovery after the vertical interval (tau about 1.26 ms
     on the home tape, `vertical_low_frequency`), the response tilt along
     the head's sweep - so it is reported with that named and is a weaker
     witness than channel 1.

AND THREE DIAGNOSTICS that say what an excess IS:

  3. THE CONTENT SLOPE, the statistic that found the porch: the residual
     regressed on the PRECEDING line's active level, on picture rows only
     (mixing in blanking rows makes the covariate bimodal and manufactures
     a slope in every window - the trap recorded with the porch finding).
     A slope with a large t is content, by construction.

  4. THE FIELD-RATE COUPLING: the per-field reference term correlated with
     the per-field picture level. On the bounce the whole field alternates
     between black and white, so this reads whether the DECK (its clamp or
     level control) moves the sync with the picture as a whole. That is a
     physical coupling and not a measurement fault, which is why it is
     kept apart from channel 1.

  5. THE ALONG-FIELD PROFILE. The excess is re-measured on the near, middle
     and far thirds of the picture lines. The blanking lines all sit
     within eleven lines of the field's start, so anything that DRIFTS
     within a field - the time base, the colour-under phase, the envelope
     under the head's sweep - is removed at the start by the per-field term
     and grows toward the end: an excess that climbs across the thirds is
     drift, one that is flat is content or noise level. Measured on the
     home capture's burst phase the excess runs from the near to the far
     third and the static control shows the same shape, which is how a
     4.5 dB excess was read as the time base and not the picture.

THE PER-FIELD TERM IS TAKEN FROM THE BLANKING LINES ALONE. Taking it from
every line would push a field-common content effect into the reference
lines with the opposite sign and inflate the floor; taking it from the
content-free lines keeps the floor honest, and costs a known amount which
the count prediction carries: with `n` reference lines per field the
reference residual has variance sigma^2 (1 - 1/n) and every other line
sigma^2 (1 + 1/n), so the ratio predicted by the counts alone is
(1 + 1/n) / (1 - 1/n) and not 1 (+0.79 dB for n = 11, +1.76 dB for the
five lines the countdown capture leaves).

THE DECOMPOSITION RUNS WITHIN EACH HEAD. A two-head drum alternates heads
field by field, and the two heads do not share a profile along the field.
Pooled across heads the per-line-index term is the average of two
profiles, so every field's residual carries half their difference, which
grows with distance along the field; the blanking lines see almost none of
it and the picture lines see all of it, and that reads as content. It is
the per-head pooling trap the vertical-interval work recorded (a pooled
r of -0.69 that vanished within heads), met again.

THE ERROR BAR IS NOT THE GAUSSIAN ONE. The variance of a sample variance
is (kurtosis - 1) sigma^4 / n, which is 2 sigma^4 / n only for a Gaussian.
Sync-derived series have heavier tails than that (dropouts, the head
switch), so the measured kurtosis is used, and `n` is the EFFECTIVE count
after serial correlation along the line sequence
(`head_model.effective_sample_size`), taken on the residual and never on
the raw series.

WHAT WAS MEASURED (tools/ringing_measure/content_independence.py,
2026-09-05, the whole 0.48 s of each zaroff capture and 3 s windows of the
two-hour captures, the per-line series taken from the decoder's own
demodulator; the tables are in that tool's docstring). The front porch
reads as bleed-through on every capture that has content in front of it,
with the largest content slope of any series (-0.035 IRE per IRE on the
bounce at t = -165), which is the porch finding reproduced without the
check having been told about it - the calibration of the instrument. The
sync tip, the back porch after the burst and the sync width also react to
the preceding line's level, by an order of magnitude less (-0.0045,
+0.0035 IRE per IRE and -0.13 ns per IRE on the bounce, at |t| of 21 to
38), and the noise on the tip is 20 per cent higher after a white line
than after a black one.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from vhsdecode.models import sync_geometry
from vhsdecode.models.head_model import effective_sample_size, lag_one


# --------------------------------------------------------------------------
# the populations, from the specification
# --------------------------------------------------------------------------

# ITU-R BT.1700 Table 3 row j, as extracted at docs/BT1700_COLLAPSE.md
# section 1.2: the field-blanking interval is 20H + 1.5 us for 525/60 and
# 25H + a for 625/50. It begins 1.5 us (row K) before the first equalizing
# pulse, so it ends at the leading edge of the sync of line 21 (or 26): the
# first twenty (twenty-five) lines of the field are blanked.
FIELD_BLANKING_LINES = {"525": 20, "625": 25}

# The first blanked line a source MAY fill with an inserted signal. ITU-T
# J.63 Annex II puts the NTC-7 composite line on line 17 of field 1 and the
# combination line on line 280 (field 2's line 17), as `vertical_interval`
# cites; ITU-R BT.473 puts the 625-line insertion test signals on lines 17
# and 18. Which lines a given recording actually carries is MEASURED by
# `occupied_lines` and reported, never assumed - a test-signal generator
# inserts nothing unless told to. Lines below this one set the floor that
# the measurement compares against.
INSERTION_ELIGIBLE_FROM = {"525": 17, "625": 17}

# The verdict threshold: three standard errors, the usual convention for a
# detection claim, recorded here so it is one number and not several.
DETECTION_SIGMA = 3.0

# How many parts the picture lines are cut into for the along-field
# profile: three is the fewest that can show a monotonic climb.
ALONG_FIELD_PARTS = 3


def line_populations(system: str = "NTSC", field_lines: int = 263
                     ) -> Dict[str, object]:
    """The field-relative line sets, from the specification alone.

    Lines are counted from 1 at the first equalizing pulse of the field,
    which is BT.1700's own numbering for field 1 and, for field 2, the
    field-relative form of it (line 264 is 1). `field_lines` is the
    format's count for the field (sysparams `field_lines`).

      geometry        1 to 9: the equalizing and field-sync sequences. No
                      normal line sync, so no sync-derived series exists
                      here; excluded from both populations.
      reference       10 to 20: normal sync, blanked, and the line before
                      each is blanked too. The content-free control.
      first_unblanked 21: unblanked, but the line before it is not, so it
                      belongs to neither population and is kept apart.
      picture         22 to the last line clear of the head switch.
      head_switch     the lines the switch window reaches, from SMPTE 32M's
                      5 to 8 H ahead of the V-sync edge as
                      `sync_geometry.interval_mask` states it, plus its
                      assumed transient. Measured onsets on this material
                      fall at line 259-260 of 263 (the head-switch lane,
                      2026-09-01), inside this window.
    """
    key = sync_geometry._system(system)
    first, middle, last = sync_geometry.SEQUENCE_LINES[key]
    # 625/50 has 2.5 + 2.5 + 2.5 = 7.5 lines: line 8 still carries an
    # equalizing pulse at its start, so the first normal line is 9
    geometry_lines = int(np.ceil(first + middle + last))
    blanked = FIELD_BLANKING_LINES[key]
    line_us = sync_geometry.LINE_PERIOD_US[key]
    window = sync_geometry.interval_mask(key)["head_switch"]
    # the window is stated relative to the NEXT interval's start; a negative
    # start means it begins that many microseconds before the field ends
    reach_lines = int(np.ceil(-float(window["start_us"]) / line_us))
    reach_lines = max(reach_lines, 0)
    first_switch_line = int(field_lines) - reach_lines + 1
    return {
        "system": key,
        "field_lines": int(field_lines),
        "geometry": list(range(1, geometry_lines + 1)),
        "reference": list(range(geometry_lines + 1, blanked + 1)),
        "first_unblanked": [blanked + 1],
        "picture": list(range(blanked + 2, first_switch_line)),
        "head_switch": list(range(first_switch_line, int(field_lines) + 1)),
        "insertion_eligible_from": INSERTION_ELIGIBLE_FROM[key],
        "source": ("ITU-R BT.1700 Table 3 rows j, K, l, m, n; SMPTE 32M "
                   "head-switch placement via sync_geometry.interval_mask"),
    }


def robust_spread(values: np.ndarray) -> float:
    """The standard deviation a Gaussian with this median absolute
    deviation would have (1.4826 MAD), which a dropout cannot inflate."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def occupied_lines(active_rms: np.ndarray, line_index: np.ndarray,
                   populations: Dict[str, object],
                   factor: float = DETECTION_SIGMA) -> Dict[str, object]:
    """Which reference lines carry an inserted signal, MEASURED.

    `active_rms` is each line's active-interval RMS about its own mean; a
    blanked line has only noise there. A line is occupied when its median
    active RMS across fields exceeds the floor set by the reference lines
    below the insertion-eligible line by `factor` times that floor's
    spread AND by at least the floor itself: an inserted signal carries
    tens of IRE of RMS against a floor under one, while the spread among
    the blank lines can be a hundredth of an IRE, and three of those alone
    would flag line 10's vertical-interval recovery slope (measured 0.78
    against 0.70 IRE on the bars capture) as a signal.

    This is a measurement on the RESERVED interval, not on picture, and it
    exists so that a broadcast recording's VITS, VIR and caption lines are
    not counted as content-free. Measured 2026-09-05: countdown lines
    15-20 at 13, 16, 26, 35, 25 and 6 IRE RMS against a 0.77 floor; home
    lines 19-20 at 35 and 23 against 0.62; the bars and bounce captures
    (a generator with nothing inserted) none.
    """
    reference = list(populations["reference"])
    eligible_from = int(populations["insertion_eligible_from"])
    line_index = np.asarray(line_index)
    active_rms = np.asarray(active_rms, dtype=np.float64)
    per_line = {}
    for n in reference:
        picked = active_rms[(line_index == n) & np.isfinite(active_rms)]
        per_line[n] = float(np.median(picked)) if len(picked) else np.nan
    floor_lines = [n for n in reference if n < eligible_from
                   and np.isfinite(per_line[n])]
    if len(floor_lines) < 2:
        return {"occupied": [], "floor": np.nan, "spread": np.nan,
                "margin": np.nan, "per_line": per_line,
                "why": "too few lines below the insertion-eligible line"}
    floor_values = np.array([per_line[n] for n in floor_lines])
    floor = float(np.median(floor_values))
    spread = robust_spread(floor_values)
    margin = max(factor * spread, floor)
    occupied = [n for n in reference
                if np.isfinite(per_line[n]) and per_line[n] > floor + margin]
    return {"occupied": occupied, "floor": floor, "spread": spread,
            "margin": margin, "per_line": per_line}


def plateau_fields(field_levels: Dict[object, float], level_noise: float,
                   factor: float = DETECTION_SIGMA) -> Dict[str, object]:
    """The fields on which a pattern that changes over time is static.

    The per-field picture levels are split at their largest gap. The split
    is real when the gap exceeds `factor` times the spread within the two
    halves; otherwise the level never changes and every field is static.
    A field belongs to a plateau when it lies within `factor` times the
    larger of the plateau's own spread and `level_noise` of the plateau's
    median, and fields between plateaus are in transition. `level_noise`
    is the field-to-field scatter of the SAME level statistic taken on the
    blank lines, so the gate is the measurement's own noise and not a
    number chosen for one pattern.

    Measured 2026-09-05 on the bounce capture: fields 0-12 at +4.4 to +5
    IRE and 13-26 at +103 to +104, the split 99 IRE against a within-
    plateau spread under 0.5, no field in transition; on the bars capture
    +46.9 to +47.0 IRE, no split.
    """
    fields = sorted(field_levels)
    levels = np.array([field_levels[f] for f in fields], dtype=np.float64)
    if len(levels) == 0:
        return {"fields": [], "levels": {}, "static_everywhere": True,
                "plateaus": []}
    order = np.argsort(levels)
    sorted_levels = levels[order]
    gaps = np.diff(sorted_levels)
    noise = float(level_noise) if np.isfinite(level_noise) else 0.0
    if len(gaps) == 0 or gaps.max() <= 0:
        return {"fields": fields, "levels": {f: float(np.median(levels))
                                             for f in fields},
                "static_everywhere": True,
                "plateaus": [{"level": float(np.median(levels)),
                              "fields": fields}]}
    cut = int(np.argmax(gaps))
    low, high = sorted_levels[: cut + 1], sorted_levels[cut + 1:]
    within = np.nanmax([robust_spread(low) if len(low) > 1 else 0.0,
                        robust_spread(high) if len(high) > 1 else 0.0,
                        noise])
    if gaps[cut] <= factor * max(within, 1e-12):
        return {"fields": fields, "levels": {f: float(np.median(levels))
                                             for f in fields},
                "static_everywhere": True,
                "plateaus": [{"level": float(np.median(levels)),
                              "fields": fields}]}
    plateaus = []
    chosen: Dict[object, float] = {}
    for part in (low, high):
        centre = float(np.median(part))
        spread = robust_spread(part) if len(part) > 1 else 0.0
        gate = factor * max(spread if np.isfinite(spread) else 0.0, noise, 1e-12)
        members = [f for f, v in zip(fields, levels) if abs(v - centre) <= gate]
        plateaus.append({"level": centre, "fields": members})
        for f in members:
            chosen[f] = centre
    return {"fields": sorted(chosen), "levels": chosen,
            "static_everywhere": False, "plateaus": plateaus,
            "in_transition": [f for f in fields if f not in chosen]}


# --------------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------------

def two_way_residual(values: np.ndarray, field_index: np.ndarray,
                     line_index: np.ndarray, reference_mask: np.ndarray,
                     minimum_reference: int = 3) -> Dict[str, object]:
    """Remove the per-field term (from the reference lines only) and the
    per-line-index term (across fields), and return what is left.

    The per-field term absorbs whatever moves every line of a field alike
    - gain drift, the deck's level control - and is estimated from the
    content-free lines so a field-common content effect cannot enter the
    floor. The per-line-index term absorbs whatever is fixed to the line's
    position in the field: the recovery after the vertical interval, the
    head-switch disturbance, and on a static pattern the content itself.
    Fields with fewer than `minimum_reference` usable reference lines are
    dropped rather than estimated from too little.
    """
    values = np.asarray(values, dtype=np.float64)
    field_index = np.asarray(field_index)
    line_index = np.asarray(line_index)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    good = np.isfinite(values)

    per_field: Dict[object, float] = {}
    reference_counts: Dict[object, int] = {}
    for field in np.unique(field_index):
        picked = good & reference_mask & (field_index == field)
        count = int(picked.sum())
        if count >= minimum_reference:
            per_field[field] = float(np.mean(values[picked]))
            reference_counts[field] = count
    kept = good & np.isin(field_index, list(per_field.keys()))
    residual = np.full_like(values, np.nan)
    field_term = np.array([per_field.get(f, np.nan) for f in field_index])
    residual[kept] = values[kept] - field_term[kept]

    line_term = np.full_like(values, np.nan)
    per_line: Dict[object, float] = {}
    for line in np.unique(line_index[kept]):
        picked = kept & (line_index == line)
        per_line[line] = float(np.mean(residual[picked]))
    for line, mean in per_line.items():
        line_term[line_index == line] = mean
    residual[kept] = residual[kept] - line_term[kept]

    return {
        "residual": residual,
        "field_term": field_term,
        "line_term": line_term,
        "per_field": per_field,
        "per_line": per_line,
        "reference_per_field": reference_counts,
        "fields_kept": len(per_field),
        "kept": kept,
    }


def grouped_residual(values: np.ndarray, field_index: np.ndarray,
                     line_index: np.ndarray, reference_mask: np.ndarray,
                     group_index: Optional[np.ndarray]) -> Dict[str, object]:
    """The two-way decomposition run WITHIN EACH GROUP (the head) and
    pooled - see the module docstring for why pooling across heads reads
    the heads' profile difference as content."""
    if group_index is None:
        return two_way_residual(values, field_index, line_index,
                                reference_mask)
    group_index = np.asarray(group_index)
    values = np.asarray(values, dtype=np.float64)
    field_index = np.asarray(field_index)
    line_index = np.asarray(line_index)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    residual = np.full_like(values, np.nan)
    field_term = np.full_like(values, np.nan)
    line_term = np.full_like(values, np.nan)
    per_field: Dict[object, float] = {}
    per_line_by_group: Dict[object, Dict[object, float]] = {}
    reference_counts: Dict[object, int] = {}
    kept = np.zeros(len(values), dtype=bool)
    for group in np.unique(group_index):
        inside = group_index == group
        part = two_way_residual(values[inside], field_index[inside],
                                line_index[inside], reference_mask[inside])
        residual[inside] = part["residual"]
        field_term[inside] = part["field_term"]
        line_term[inside] = part["line_term"]
        kept[inside] = part["kept"]
        per_field.update(part["per_field"])
        reference_counts.update(part["reference_per_field"])
        per_line_by_group[group] = part["per_line"]
    lines = set()
    for table in per_line_by_group.values():
        lines.update(table.keys())
    per_line = {line: float(np.mean([table[line] for table in
                                     per_line_by_group.values()
                                     if line in table]))
                for line in lines}
    return {"residual": residual, "field_term": field_term,
            "line_term": line_term, "per_field": per_field,
            "per_line": per_line, "per_line_by_group": per_line_by_group,
            "reference_per_field": reference_counts,
            "fields_kept": len(per_field), "kept": kept}


def predicted_ratio(reference_per_field: Sequence[int]) -> float:
    """The variance ratio (other lines / reference lines) the counts alone
    predict, when nothing else differs between the populations.

    Subtracting a mean estimated from `n` reference lines leaves those
    lines with variance sigma^2 (1 - 1/n) and every other line with
    sigma^2 (1 + 1/n). With `n` varying by field the two are averaged.
    """
    counts = np.asarray([c for c in reference_per_field if c > 1],
                        dtype=np.float64)
    if counts.size == 0:
        return 1.0
    other = float(np.mean(1.0 + 1.0 / counts))
    reference = float(np.mean(1.0 - 1.0 / counts))
    return other / max(reference, 1e-12)


def variance_with_error(residual: np.ndarray) -> Dict[str, float]:
    """A population's residual variance with the error of its logarithm.

    var(s^2) = (kurtosis - 1) sigma^4 / n_effective, so the standard error
    of ln s^2 is sqrt((kurtosis - 1) / n_effective). Gaussian data has
    kurtosis 3 and recovers the textbook 2/n.
    """
    values = np.asarray(residual, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = len(values)
    if count < 4:
        return {"variance": np.nan, "count": float(count),
                "effective": float(count), "log_error": np.nan,
                "kurtosis": np.nan, "lag1": np.nan}
    centred = values - values.mean()
    variance = float(np.mean(centred ** 2))
    kurtosis = float(np.mean(centred ** 4) / max(variance ** 2, 1e-300))
    effective = effective_sample_size(values)
    log_error = float(np.sqrt(max(kurtosis - 1.0, 1e-6)
                              / max(effective - 1.0, 1.0)))
    return {"variance": variance, "count": float(count),
            "effective": float(effective), "log_error": log_error,
            "kurtosis": kurtosis, "lag1": lag_one(values)}


def variance_channel(residual: np.ndarray, test_mask: np.ndarray,
                     reference_mask: np.ndarray,
                     reference_per_field: Sequence[int]) -> Dict[str, float]:
    """Channel 1: the residual variance ratio, test over reference, against
    what the counts predict, in decibels with its error.

    A NEGATIVE excess beyond its error means the reference lines vary
    MORE than the test lines. That happens on series the deck re-acquires
    after the vertical interval - the burst amplitude and phase, whose
    colour-under loops recover over the first lines and recover
    differently field to field (bars burst amplitude -1.6 dB at z -4.5,
    countdown burst phase -3.3 dB at z -15, measured 2026-09-05). The
    reference lines are then not a floor for that series and the verdict
    says so instead of calling the picture clean.
    """
    test = variance_with_error(residual[np.asarray(test_mask, dtype=bool)])
    reference = variance_with_error(
        residual[np.asarray(reference_mask, dtype=bool)])
    predicted = predicted_ratio(reference_per_field)
    out = {
        "test_variance": test["variance"],
        "reference_variance": reference["variance"],
        "test_count": test["count"],
        "reference_count": reference["count"],
        "test_effective": test["effective"],
        "reference_effective": reference["effective"],
        "predicted_ratio": predicted,
        "predicted_db": float(10.0 * np.log10(predicted)),
    }
    if not (np.isfinite(test["variance"]) and np.isfinite(reference["variance"])
            and reference["variance"] > 0 and test["variance"] > 0):
        out.update({"ratio_db": np.nan, "excess_db": np.nan,
                    "error_db": np.nan, "z": np.nan, "verdict": "unmeasured",
                    "added_variance": np.nan})
        return out
    ratio_db = float(10.0 * np.log10(test["variance"] / reference["variance"]))
    error_db = float(10.0 / np.log(10.0)
                     * np.hypot(test["log_error"], reference["log_error"]))
    excess_db = ratio_db - out["predicted_db"]
    z = excess_db / max(error_db, 1e-12)
    if z > DETECTION_SIGMA:
        verdict = "bleed-through"
    elif z < -DETECTION_SIGMA:
        verdict = "reference noisier than picture"
    else:
        verdict = "content-independent"
    out.update({"ratio_db": ratio_db, "excess_db": excess_db,
                "error_db": error_db, "z": float(z), "verdict": verdict,
                # the size, as the content's own variance in the series'
                # units squared: what the picture ADDED beyond the counts
                "added_variance": float(max(
                    test["variance"] - reference["variance"] * predicted, 0.0)),
                "test_kurtosis": test["kurtosis"],
                "reference_kurtosis": reference["kurtosis"]})
    return out


def level_channel(per_line: Dict[object, float], test_lines: Sequence[int],
                  reference_lines: Sequence[int]) -> Dict[str, float]:
    """Channel 2: the per-line-index means on the test lines minus those on
    the reference lines. The static-content witness, confounded with the
    field-time distortion and said so."""
    test = np.array([per_line[n] for n in test_lines if n in per_line])
    reference = np.array([per_line[n] for n in reference_lines
                          if n in per_line])
    if len(test) < 2 or len(reference) < 2:
        return {"shift": np.nan, "error": np.nan, "z": np.nan,
                "verdict": "unmeasured"}
    shift = float(test.mean() - reference.mean())
    error = float(np.hypot(
        test.std(ddof=1) / np.sqrt(max(effective_sample_size(test), 1.0)),
        reference.std(ddof=1)
        / np.sqrt(max(effective_sample_size(reference), 1.0))))
    z = shift / max(error, 1e-12)
    return {"shift": shift, "error": error, "z": float(z),
            "verdict": ("level shift" if abs(z) > DETECTION_SIGMA
                        else "no level shift"),
            "confound": "the field-time recovery after the vertical interval "
                        "and the sweep profile are fixed to the line index too"}


def content_slope(residual: np.ndarray, covariate: np.ndarray,
                  picture_mask: np.ndarray) -> Dict[str, float]:
    """Statistic 3: the residual regressed on a content covariate, on
    picture rows only. Slope in series units per covariate unit, with the
    error inflated for the remainder's serial correlation."""
    picked = (np.asarray(picture_mask, dtype=bool) & np.isfinite(residual)
              & np.isfinite(covariate))
    y = np.asarray(residual, dtype=np.float64)[picked]
    x = np.asarray(covariate, dtype=np.float64)[picked]
    if len(y) < 8 or np.std(x) <= 0:
        return {"slope": np.nan, "error": np.nan, "t": np.nan,
                "count": float(len(y)),
                "covariate_spread": float(np.std(x)) if len(x) else np.nan,
                "swing": np.nan, "verdict": "unmeasured"}
    x_centred = x - x.mean()
    slope = float(x_centred @ (y - y.mean()) / (x_centred @ x_centred))
    fitted = y.mean() + slope * x_centred
    remainder = y - fitted
    effective = effective_sample_size(remainder)
    error = float(np.std(remainder, ddof=2)
                  / np.sqrt(x_centred @ x_centred)
                  * np.sqrt(len(y) / max(effective, 1.0)))
    t = slope / max(error, 1e-15)
    return {"slope": slope, "error": error, "t": float(t),
            "count": float(len(y)), "effective": float(effective),
            "covariate_spread": float(np.std(x)),
            "swing": float(slope * (np.percentile(x, 97.5)
                                    - np.percentile(x, 2.5))),
            "verdict": ("content slope" if abs(t) > DETECTION_SIGMA
                        else "no content slope")}


def field_rate_coupling(per_field_term: Dict[object, float],
                        per_field_content: Dict[object, float]
                        ) -> Dict[str, float]:
    """Statistic 4: the reference lines' per-field term against the field's
    picture level. Reads the deck's own coupling of picture into sync."""
    fields = [f for f in per_field_term if f in per_field_content
              and np.isfinite(per_field_term[f])
              and np.isfinite(per_field_content[f])]
    if len(fields) < 6:
        return {"r": np.nan, "slope": np.nan, "t": np.nan,
                "count": float(len(fields)), "verdict": "unmeasured"}
    a = np.array([per_field_term[f] for f in fields])
    b = np.array([per_field_content[f] for f in fields])
    if np.std(a) <= 0 or np.std(b) <= 0:
        return {"r": np.nan, "slope": np.nan, "t": np.nan,
                "count": float(len(fields)),
                "verdict": "unmeasured (no variation)"}
    r = float(np.corrcoef(a, b)[0, 1])
    slope = float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    remainder = a - (a.mean() + slope * (b - b.mean()))
    effective = effective_sample_size(remainder)
    t = r * np.sqrt(max(effective - 2.0, 1.0) / max(1.0 - r * r, 1e-12))
    return {"r": r, "slope": slope, "t": float(t), "count": float(len(fields)),
            "effective": float(effective),
            "content_spread": float(np.std(b)),
            "verdict": ("field-rate coupling" if abs(t) > DETECTION_SIGMA
                        else "no field-rate coupling")}


def along_field(residual: np.ndarray, line_index: np.ndarray,
                picture_lines: Sequence[int], reference_mask: np.ndarray,
                reference_per_field: Sequence[int],
                parts: int = ALONG_FIELD_PARTS) -> List[Dict[str, float]]:
    """Diagnostic 5: the variance channel on successive parts of the
    picture lines, nearest the reference lines first. An excess that
    climbs part by part is within-field drift, not content."""
    lines = np.asarray(sorted(picture_lines))
    out = []
    for chunk in np.array_split(lines, parts):
        if len(chunk) == 0:
            continue
        mask = np.isin(line_index, chunk)
        result = variance_channel(residual, mask, reference_mask,
                                  reference_per_field)
        result["lines"] = (int(chunk[0]), int(chunk[-1]))
        out.append(result)
    return out


def drift_verdict(profile: Sequence[Dict[str, float]]) -> Dict[str, object]:
    """Whether the along-field profile climbs: the far part's excess minus
    the near part's, against their combined error."""
    usable = [p for p in profile if np.isfinite(p.get("excess_db", np.nan))]
    if len(usable) < 2:
        return {"climb_db": np.nan, "error_db": np.nan, "z": np.nan,
                "verdict": "unmeasured"}
    near, far = usable[0], usable[-1]
    climb = far["excess_db"] - near["excess_db"]
    error = float(np.hypot(near["error_db"], far["error_db"]))
    z = climb / max(error, 1e-12)
    return {"climb_db": float(climb), "error_db": error, "z": float(z),
            "verdict": ("climbs along the field: drift"
                        if z > DETECTION_SIGMA else "flat along the field")}


# --------------------------------------------------------------------------
# the whole check on one series
# --------------------------------------------------------------------------

def check_series(values: np.ndarray, field_index: np.ndarray,
                 line_index: np.ndarray, populations: Dict[str, object],
                 exclude_lines: Sequence[int] = (),
                 preceding_content: Optional[np.ndarray] = None,
                 field_content: Optional[Dict[object, float]] = None,
                 pattern_mask: Optional[np.ndarray] = None,
                 group_index: Optional[np.ndarray] = None
                 ) -> Dict[str, object]:
    """Run every channel on one per-line series.

    `values`, `field_index`, `line_index` are parallel arrays, one entry per
    measured line, with `line_index` the field-relative line number in the
    specification's numbering. `exclude_lines` removes measured-occupied
    reference lines. `preceding_content` is the covariate for the content
    slope (the previous line's active level); `field_content` the
    per-field picture level for the field-rate coupling; `pattern_mask`
    marks the lines of the static test-pattern population, reported
    against the same reference; `group_index` is the head (the field
    parity), and the decomposition is run within each group.

    Populations reported: `blanking` (the reference), `pattern`,
    `picture`, `full_field` (every measured line that is not reference),
    `head_switch`, and the along-field `profile` of the picture lines.
    """
    line_index = np.asarray(line_index)
    field_index = np.asarray(field_index)
    reference_lines = [n for n in populations["reference"]
                       if n not in set(exclude_lines)]
    reference_mask = np.isin(line_index, reference_lines)
    picture_mask = np.isin(line_index, populations["picture"])
    switch_mask = np.isin(line_index, populations["head_switch"])
    full_mask = ~np.isin(line_index, populations["geometry"]) & ~reference_mask
    decomposed = grouped_residual(values, field_index, line_index,
                                  reference_mask, group_index)
    residual = decomposed["residual"]
    counts = list(decomposed["reference_per_field"].values())
    out: Dict[str, object] = {
        "reference_lines": reference_lines,
        "fields": decomposed["fields_kept"],
        "reference_per_field": float(np.mean(counts)) if counts else 0.0,
        "blanking": variance_with_error(residual[reference_mask]),
        "full_field": variance_channel(residual, full_mask, reference_mask,
                                       counts),
        "picture": variance_channel(residual, picture_mask, reference_mask,
                                    counts),
        "head_switch": variance_channel(residual, switch_mask, reference_mask,
                                        counts),
        "level": level_channel(decomposed["per_line"],
                               populations["picture"], reference_lines),
        "profile": along_field(residual, line_index, populations["picture"],
                               reference_mask, counts),
    }
    out["drift"] = drift_verdict(out["profile"])
    if pattern_mask is not None:
        out["pattern"] = variance_channel(
            residual, np.asarray(pattern_mask, dtype=bool) & picture_mask,
            reference_mask, counts)
    if preceding_content is not None:
        out["content_slope"] = content_slope(residual, preceding_content,
                                             picture_mask)
    if field_content is not None:
        out["field_rate"] = field_rate_coupling(decomposed["per_field"],
                                                field_content)
    out["residual"] = residual
    out["decomposition"] = decomposed
    return out


def control_attribution(result: Dict[str, object],
                        control: Dict[str, object],
                        population: str = "full_field") -> Dict[str, float]:
    """The excess of one capture's population above the static-pattern
    control's, with the combined error. What the control shows on a static
    pattern is not picture; what this capture shows beyond it is."""
    mine = result.get(population, {})
    theirs = control.get(population, {})
    a, b = mine.get("excess_db", np.nan), theirs.get("excess_db", np.nan)
    if not (np.isfinite(a) and np.isfinite(b)):
        return {"above_control_db": np.nan, "error_db": np.nan, "z": np.nan,
                "verdict": "unmeasured"}
    error = float(np.hypot(mine.get("error_db", np.nan),
                           theirs.get("error_db", np.nan)))
    above = float(a - b)
    z = above / max(error, 1e-12)
    return {"above_control_db": above, "error_db": error, "z": float(z),
            "control_excess_db": float(b),
            "verdict": ("picture statistics" if z > DETECTION_SIGMA
                        else "within the static control")}


def plant(rng: np.random.Generator, fields: int, populations: Dict[str, object],
          noise_sigma: float, content_sigma: float = 0.0,
          static_shift: float = 0.0, field_drift: float = 0.0,
          line_profile: Optional[np.ndarray] = None,
          within_field_drift: float = 0.0,
          head_profile: float = 0.0
          ) -> Dict[str, np.ndarray]:
    """A synthetic per-line series with a KNOWN bleed-through, for the tests
    and for calibrating the check. Returns parallel arrays plus the planted
    content covariate and the field parity.

    `content_sigma` adds independent content to every picture line;
    `static_shift` offsets the picture lines by a constant (a static
    pattern's bleed-through); `field_drift` moves whole fields;
    `line_profile` is a fixed profile along the field; `within_field_drift`
    is a ramp along each field whose slope is drawn per field (what the
    time base does); `head_profile` is a tilt along the field that
    alternates sign with the field parity (the two heads' differing
    profiles, the pooling trap).
    """
    lines = (list(populations["reference"]) + list(populations["first_unblanked"])
             + list(populations["picture"]) + list(populations["head_switch"]))
    field_index = np.repeat(np.arange(fields), len(lines))
    line_index = np.tile(np.array(lines), fields)
    parity = field_index % 2
    picture = np.isin(line_index, populations["picture"])
    values = rng.normal(0.0, noise_sigma, size=len(line_index))
    if field_drift:
        values += field_drift * rng.normal(size=fields)[field_index]
    if line_profile is not None:
        profile = np.asarray(line_profile, dtype=np.float64)
        values += profile[np.clip(line_index - 1, 0, len(profile) - 1)]
    position = (line_index - lines[0]) / float(lines[-1] - lines[0])
    if within_field_drift:
        slopes = within_field_drift * rng.normal(size=fields)
        values += slopes[field_index] * position
    if head_profile:
        values += head_profile * position * np.where(parity == 0, 1.0, -1.0)
    content = rng.normal(0.0, 1.0, size=len(line_index))
    content[~picture] = 0.0
    values += content_sigma * content
    values[picture] += static_shift
    return {"values": values, "field_index": field_index,
            "line_index": line_index, "content": content, "parity": parity}
