"""The VCR's mechanical transport, as predicted frequencies.

The head model describes the magnetics and the filter model the
electronics. This is the third part, and with them it is the VCR's total
model: every rotating part of the transport turns at a rate its own
diameter fixes, and any eccentricity, bearing wear or stiff bushing in it
disturbs the tape once per revolution. Those disturbances are not diffuse
noise - each lands at a KNOWN frequency, so the specification predicts
where to look before anything is measured, and a line found at a predicted
rate names the part that made it.

Two quantities carry them:

  the TIME BASE, where a speed disturbance shows directly as wow and
  flutter - the tape momentarily running fast or slow;
  the RF ENVELOPE, where a tension or contact disturbance shows as
  amplitude modulation, because head-to-tape pressure sets the spacing
  and spacing sets the level by Wallace's law.

The two respond to different faults, which is what lets them be told
apart: an eccentric capstan moves the time base and barely touches the
envelope, while a sticky guide modulates tension and therefore the
envelope while the capstan holds the speed.

THE SEQUENCE MATTERS. The parts act on the tape in the order it passes
them, and a disturbance introduced upstream of the capstan is partly
regulated out by it, while one downstream is not. So each part carries
where it sits in the path, and whether the capstan's servo stands between
it and the head.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


# The transport, in the order the tape passes it. Diameters are in metres.
# Only the drum's figures are specification certainties for this format -
# a machine of this format must have a 62 mm drum turning at the field
# rate, because that is what lays down the track geometry. The rest vary
# by manufacturer and model, so they are declared as TYPICAL values to
# predict where to look, and the search reports what it actually finds
# rather than assuming the prediction.
TRANSPORT = [
    {"name": "supply reel", "order": 0, "kind": "reel",
     "radius_range_m": (0.012, 0.045), "after_capstan": False,
     "acts_on": "tension", "certain": False},
    {"name": "impedance roller", "order": 1, "kind": "roller",
     "diameter_m": 0.0130, "after_capstan": False,
     "acts_on": "tension", "certain": False},
    {"name": "entry guide", "order": 2, "kind": "guide",
     "diameter_m": 0.0060, "after_capstan": False,
     "acts_on": "tension", "certain": False},
    {"name": "head drum", "order": 3, "kind": "drum",
     "diameter_m": 0.0620, "after_capstan": False,
     "acts_on": "both", "certain": True},
    {"name": "exit guide", "order": 4, "kind": "guide",
     "diameter_m": 0.0060, "after_capstan": False,
     "acts_on": "tension", "certain": False},
    {"name": "capstan", "order": 5, "kind": "capstan",
     "diameter_m": 0.0022, "after_capstan": False,
     "acts_on": "speed", "certain": False},
    {"name": "pinch roller", "order": 6, "kind": "roller",
     "diameter_m": 0.0100, "after_capstan": True,
     "acts_on": "speed", "certain": False},
    {"name": "take-up reel", "order": 7, "kind": "reel",
     "radius_range_m": (0.012, 0.045), "after_capstan": True,
     "acts_on": "tension", "certain": False},
]

# The drum's rotation is fixed by the format, not by the transport: one
# revolution lays down two fields, so it turns at half the field rate.
FIELDS_PER_DRUM_REVOLUTION = 2


def drum_rate_hz(field_rate_hz: float = 59.94) -> float:
    return float(field_rate_hz) / FIELDS_PER_DRUM_REVOLUTION


def rotation_rates(linear_speed_m_s: float,
                   field_rate_hz: float = 59.94) -> List[Dict[str, object]]:
    """Every part's rotation rate, and the harmonics it can produce.

    A part of diameter d passing tape at speed v turns at v/(pi*d). An
    eccentricity gives a disturbance once per revolution; an out-of-round
    or a two-lobe bearing gives twice; a roller with a flat spot gives a
    train of harmonics. The reels are the exception - their radius grows
    as tape winds across, so they sweep a BAND rather than sitting at a
    line, which is itself the way to recognise them."""
    out = []
    for part in TRANSPORT:
        entry = dict(part)
        if part["kind"] == "drum":
            entry["rate_hz"] = drum_rate_hz(field_rate_hz)
            entry["rate_note"] = ("fixed by the format: one revolution "
                                  "lays down two fields")
        elif "diameter_m" in part:
            entry["rate_hz"] = float(linear_speed_m_s) / (
                np.pi * float(part["diameter_m"]))
            entry["rate_note"] = "from its diameter and the tape speed"
        elif "radius_range_m" in part:
            low, high = part["radius_range_m"]
            entry["rate_band_hz"] = (
                float(linear_speed_m_s) / (2.0 * np.pi * high),
                float(linear_speed_m_s) / (2.0 * np.pi * low))
            entry["rate_hz"] = float(np.mean(entry["rate_band_hz"]))
            entry["rate_note"] = ("a BAND, not a line: the radius grows as "
                                  "tape winds across")
        out.append(entry)
    return out


def observable(rates: Sequence[Dict[str, object]], duration_s: float,
               sample_rate_hz: float) -> List[Dict[str, object]]:
    """Which predicted rates a measurement of this length can actually
    resolve, stated before any search is run.

    Nothing slower than one cycle across the record can be told from a
    constant, and nothing faster than half the sampling rate exists. A
    part whose rate falls outside that band is not absent from the tape -
    it is absent from the MEASUREMENT, and saying so first stops a null
    result being read as a clean transport."""
    slowest = 1.0 / max(float(duration_s), 1e-12)
    fastest = float(sample_rate_hz) / 2.0
    out = []
    for entry in rates:
        rate = float(entry.get("rate_hz", np.nan))
        item = dict(entry)
        item["resolvable"] = bool(slowest <= rate <= fastest)
        item["why_not"] = ("" if item["resolvable"] else
                           "slower than one cycle in the record"
                           if rate < slowest else "above half the sampling rate")
        out.append(item)
    return out


def find_lines(series, sample_rate_hz: float, rates: Sequence[Dict[str, object]],
               tolerance: float = 0.08) -> List[Dict[str, object]]:
    """Search a per-field or per-line series for the PREDICTED rates.

    The prediction comes first and the search is told where to look, which
    is what separates this from trawling a spectrum for whatever peak
    happens to be tallest. Each candidate is reported with its height
    against the LOCAL background rather than the global mean, so a line
    sitting on a sloping noise floor is not credited for the slope."""
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 16:
        return []
    values = values - values.mean()
    window = np.hanning(len(values))
    power = np.abs(np.fft.rfft(values * window)) ** 2
    frequency = np.fft.rfftfreq(len(values), 1.0 / float(sample_rate_hz))
    found = []
    for entry in rates:
        rate = float(entry.get("rate_hz", np.nan))
        if not np.isfinite(rate):
            continue
        near = np.abs(frequency - rate) <= tolerance * rate
        if near.sum() < 1:
            continue
        # the local background: a neighbourhood excluding the candidate
        around = (np.abs(frequency - rate) <= 4.0 * tolerance * rate) & ~near
        if around.sum() < 4:
            continue
        peak = float(power[near].max())
        background = float(np.median(power[around]))
        found.append({
            "name": entry["name"],
            "rate_hz": rate,
            "order": entry.get("order"),
            "acts_on": entry.get("acts_on"),
            "after_capstan": entry.get("after_capstan"),
            "certain": entry.get("certain", False),
            "height_over_background": peak / max(background, 1e-30),
            "at_hz": float(frequency[near][int(np.argmax(power[near]))]),
        })
    return sorted(found, key=lambda item: -item["height_over_background"])


# --------------------------------------------------------------------------
# the tape path's geometry, and what each error does to a field
# --------------------------------------------------------------------------

# A video head sweeps one whole field in one pass across the tape. That is
# what makes these geometric errors measurable at all: anything that varies
# ALONG the sweep appears as a variation from the top of a field to the
# bottom, and its SHAPE along the field names the error.
#
#   a PARABOLIC BOW - the tape curving across its width - gives a quadratic
#   variation, worst at the middle or the ends of the sweep and symmetric
#   about its centre;
#   a HEIGHT or X-VALUE offset gives a constant, the head sitting off-track
#   by the same amount all the way across;
#   a GUIDE fault gives a linear tilt, the path drifting steadily as the
#   tape runs from one guide to the other;
#   a REEL offset moves slowly across many fields rather than within one.
#
# So the shape along a field separates them, and the search below fits
# exactly those three shapes and reports which one carries the variance.
PATH_ERRORS = {
    "parabolic_bow_m": {"shape": "quadratic", "unit": "m",
                        "cause": "the tape curving across its width"},
    "height_offset_m": {"shape": "constant", "unit": "m",
                        "cause": "head height or the X-value adjustment"},
    "guide_tilt_m": {"shape": "linear", "unit": "m",
                     "cause": "a guide standing at the wrong height"},
}


def sweep_shapes(line_count: int) -> Dict[str, np.ndarray]:
    """The three shapes a field can carry, on the head's sweep from the top
    of the field to the bottom. Orthogonalised so the variance each explains
    is its own and not shared with the others."""
    position = np.linspace(-1.0, 1.0, int(line_count))
    constant = np.ones_like(position)
    linear = position
    quadratic = position ** 2 - np.mean(position ** 2)   # made orthogonal
    return {"height_offset_m": constant, "guide_tilt_m": linear,
            "parabolic_bow_m": quadratic}


def fit_sweep(per_line) -> Dict[str, object]:
    """Decompose a per-line variation across one field into the path errors.

    Returns each shape's coefficient and the share of the variance it
    carries, so a bow is distinguished from a height offset by its SHAPE
    along the sweep rather than by its size."""
    values = np.asarray(per_line, dtype=np.float64)
    good = np.isfinite(values)
    if good.sum() < 16:
        return {"usable": False, "why": "too few lines to decompose a sweep"}
    shapes = sweep_shapes(len(values))
    names = list(shapes)
    design = np.column_stack([shapes[name] for name in names])[good]
    target = values[good] - values[good].mean()
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    out: Dict[str, object] = {"usable": True}
    total = float(np.var(target))
    for name, coefficient in zip(names, coefficients):
        contribution = coefficient * shapes[name][good]
        out[name] = float(coefficient)
        out["share_" + name] = float(np.var(contribution) / max(total, 1e-30))
        out["cause_" + name] = PATH_ERRORS[name]["cause"]
    residual = target - design @ coefficients
    out["explained"] = float(1.0 - np.var(residual) / max(total, 1e-30))
    out["dominant"] = max(names, key=lambda n: out["share_" + n])
    return out

