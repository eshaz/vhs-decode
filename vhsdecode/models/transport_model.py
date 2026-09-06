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

from typing import Dict, List, Optional, Sequence, Tuple

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


# WHAT A FAULT PUTS ON A ROTATION RATE, AS ORDERS OF THAT RATE.
#
# `rotation_rates` has always named these three and returned only the
# fundamental, so a rate arrived as a bare number with no shape. The orders
# below are that docstring's own three sentences and nothing more: an
# eccentricity is once per revolution, an out-of-round or two-lobe bearing
# is twice, and a flat spot is a train. They are COUNTS fixed by the kind of
# fault, not rates, so nothing here is retyped from a specification.
#
# The train has no end of its own. A flat spot is a localised defect and its
# harmonics run on until the MEASUREMENT stops carrying them, so `harmonics`
# refuses to invent a length: either the caller states the highest order it
# wants or it passes the sampling rate and the series' own Nyquist ends the
# train. That keeps the shape a property of the instrument rather than a
# constant typed here.
FAULT_HARMONICS: Dict[str, Optional[Tuple[int, ...]]] = {
    "eccentricity": (1,),
    "out of round": (2,),
    "two-lobe bearing": (2,),
    "flat spot": None,
}


# The fractional half-width of the band `find_lines` searches around a
# predicted rate, named once so that every statement about what a
# measurement can and cannot separate uses the SAME band the search does.
# `hifi_carriers` reads it off `find_lines`' signature for the same reason.
SEARCH_TOLERANCE = 0.08


def alias_hz(frequency_hz, sample_rate_hz: float):
    """Where a sampled series actually puts a frequency: folded into
    `0 .. sample_rate/2`.

    A rate above half the sampling rate does not vanish from a measurement,
    it MOVES, and it lands on top of whatever already sits where it folds
    to. Saying only that it is "above half the sampling rate" leaves the
    reader to imagine it absent from the measurement when it is in fact
    mixed in with whatever sits at the folded frequency, so the fold is
    reported rather than the refusal alone.
    """
    rate = float(sample_rate_hz)
    f = np.asarray(frequency_hz, dtype=np.float64)
    folded = np.abs(((f + rate / 2.0) % rate) - rate / 2.0)
    return float(folded) if np.ndim(frequency_hz) == 0 else folded


def harmonics(rate_hz: float, fault: str = "eccentricity",
              sample_rate_hz: Optional[float] = None,
              max_order: Optional[int] = None) -> List[Dict[str, object]]:
    """A PART'S SHAPE IN FREQUENCY: the orders its fault puts on its rate,
    and where the sampling puts each one.

    This is the dimension `rotation_rates` named and did not carry. A rate
    alone cannot tell an eccentric roller from a flat-spotted one; the
    orders can, because the two differ in KIND and not in rate, and the
    separability law makes a difference of kind separable where a difference
    of rate is not.

    MEASURED, on the eight parts at the format's 33.35 mm/s and 59.94 Hz
    field rate, why the fold has to be reported with the order. In a series
    sampled once per FIELD the drum's own rate is exactly the Nyquist
    frequency, and its harmonics do not merely alias, they land on the two
    least recoverable places there are:

        order 1   29.9700 Hz  ->  29.9700 Hz   the Nyquist, and bit-identical
                                               to the head A/B alternation
        order 2   59.9401 Hz  ->   0.0000 Hz   DC: the series' own mean
        order 3   89.9101 Hz  ->  29.9700 Hz   back onto the Nyquist

    So a drum with a two-lobe bearing puts its whole second order into the
    mean of a per-field series, where it is not a line at all and where any
    estimator that removes a mean removes it. Per LINE at 15.734 kHz the
    same orders are ordinary low-frequency components - the drum is sampled
    262 times a revolution - and the correlation with the head alternation
    falls from 1.000000 to 0.0296.

    `fault` selects the orders from `FAULT_HARMONICS`. A flat spot's train
    has no end of its own, so give either `max_order` or `sample_rate_hz`
    and let the series' Nyquist end it.
    """
    key = str(fault).lower()
    if key not in FAULT_HARMONICS:
        raise ValueError(f"{fault!r} is not one of {sorted(FAULT_HARMONICS)}")
    rate = float(rate_hz)
    orders = FAULT_HARMONICS[key]
    if orders is None:
        if max_order is None:
            if sample_rate_hz is None:
                raise ValueError(
                    "a flat spot's harmonic train has no end of its own: give "
                    "max_order, or sample_rate_hz so the series' own Nyquist "
                    "ends it")
            max_order = int(np.floor(float(sample_rate_hz) / 2.0
                                     / max(rate, 1e-12)))
        orders = tuple(range(1, max(int(max_order), 1) + 1))
    out: List[Dict[str, object]] = []
    for order in orders:
        frequency = float(order) * rate
        item: Dict[str, object] = {
            "order": int(order), "rate_hz": rate, "at_hz": frequency,
            "fault": key,
        }
        if sample_rate_hz is not None:
            nyquist = float(sample_rate_hz) / 2.0
            item["alias_hz"] = alias_hz(frequency, sample_rate_hz)
            item["folded"] = bool(frequency >= nyquist)
            # exactly DC, which happens when the harmonic is an exact
            # multiple of the sampling rate - the drum's second order
            # in a per-field series is the case that matters
            item["at_dc"] = bool(item["alias_hz"] <= 1e-9 * max(rate, 1e-12))
        out.append(item)
    return out



def rotation_rates(linear_speed_m_s: float,
                   field_rate_hz: float = 59.94) -> List[Dict[str, object]]:
    """Every part's rotation rate, and the harmonics it can produce.

    A part of diameter d passing tape at speed v turns at v/(pi*d). An
    eccentricity gives a disturbance once per revolution; an out-of-round
    or a two-lobe bearing gives twice; a roller with a flat spot gives a
    train of harmonics. The reels are the exception - their radius grows
    as tape winds across, so they sweep a BAND rather than sitting at a
    line, which is itself the way to recognise them.

    WHAT IS RETURNED HERE IS THE FUNDAMENTAL ONLY. The three fault kinds
    named above are the rate's SHAPE and they are built by `harmonics`,
    which also says where the sampling puts each order - the distinction
    that matters, since the drum's second order folds onto DC in any
    per-field series. `harmonic_collisions` reports where one part's orders
    land on another's, which is what the shape costs."""
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
               sample_rate_hz: float,
               tolerance: Optional[float] = None) -> List[Dict[str, object]]:
    """Which predicted rates a measurement of this length can actually
    resolve, stated before any search is run.

    Nothing slower than one cycle across the record can be told from a
    constant, and nothing at or above half the sampling rate survives it. A
    part whose rate falls outside that band is not absent from the tape -
    it is absent from the MEASUREMENT, and saying so first stops a null
    result being read as a clean transport.

    THE COMPARISON AT THE TOP END IS STRICT, AND IT HAS TO BE. This function
    formerly accepted `rate <= sample_rate/2`, which credited a rate sitting
    exactly ON the Nyquist frequency as resolvable. Exactly one modelled rate
    does that and it is the one the format fixes: the drum turns once per two
    fields, so in a series sampled once per FIELD its rate IS the Nyquist
    frequency. Measured over 1024 fields, a drum modulation sampled that way
    is not merely correlated with the head A/B alternation, it is
    bit-identical to it - correlation 1.000000, max absolute difference 0.0 -
    so no search can attribute a line there to one rather than the other.
    Sampled once per LINE at 15.734 kHz the drum is 262 samples a revolution
    and the same correlation falls to 0.0296. The verdict now follows the
    arithmetic instead of contradicting it.

    A FOLDED RATE IS REPORTED WITH WHERE IT LANDS, not merely refused. Above
    the Nyquist a rate moves rather than disappears, so `alias_hz` carries
    the frequency the series actually shows it at and `aliases_onto` names
    any OTHER modelled part whose own alias falls within `tolerance` of it -
    which is the difference between "this part is invisible" and "a line at
    this frequency could be either of these two parts".

    `tolerance` defaults to the fractional band `find_lines` searches, so the
    two cannot drift apart: a pair this function does not call aliased is a
    pair that function would not confuse.
    """
    tol = float(SEARCH_TOLERANCE if tolerance is None else tolerance)
    slowest = 1.0 / max(float(duration_s), 1e-12)
    fastest = float(sample_rate_hz) / 2.0
    out = []
    for entry in rates:
        rate = float(entry.get("rate_hz", np.nan))
        item = dict(entry)
        item["nyquist_hz"] = fastest
        item["slowest_resolvable_hz"] = slowest
        item["alias_hz"] = (alias_hz(rate, sample_rate_hz)
                            if np.isfinite(rate) else float("nan"))
        item["at_nyquist"] = bool(np.isfinite(rate)
                                  and abs(rate - fastest) <= 1e-9 * fastest)
        item["resolvable"] = bool(slowest <= rate < fastest)
        if item["resolvable"]:
            item["why_not"] = ""
        elif rate < slowest:
            item["why_not"] = "slower than one cycle in the record"
        elif item["at_nyquist"]:
            item["why_not"] = ("exactly at half the sampling rate, where it "
                               "cannot be told from the alternating sequence")
        else:
            item["why_not"] = ("at or above half the sampling rate: the "
                               "series shows it at %.6f Hz instead"
                               % item["alias_hz"])
        out.append(item)
    # which pairs the sampling has put on top of one another
    for item in out:
        here = float(item["alias_hz"])
        onto = []
        for other in out:
            if other is item or other.get("name") == item.get("name"):
                continue
            there = float(other["alias_hz"])
            if not (np.isfinite(here) and np.isfinite(there)):
                continue
            if abs(here - there) <= tol * max(here, there, 1e-30):
                onto.append(str(other.get("name")))
        item["aliases_onto"] = tuple(onto)
    return out


def find_lines(series, sample_rate_hz: float, rates: Sequence[Dict[str, object]],
               tolerance: float = SEARCH_TOLERANCE) -> List[Dict[str, object]]:
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


def harmonic_collisions(rates: Sequence[Dict[str, object]],
                        max_order: int = 5,
                        tolerance: Optional[float] = None,
                        sample_rate_hz: Optional[float] = None
                        ) -> List[Dict[str, object]]:
    """WHERE ONE PART'S HARMONIC SITS ON ANOTHER PART'S, which is what the
    shape costs and why it has to be stated with the shape.

    A search told to look at the fundamentals alone reports every part as
    its own line. Once the orders are admitted the picture changes, because
    the diameters this format's decks use put several of them in exact small
    integer ratios, and an exact ratio is a coincidence no amount of tape
    resolves.

    MEASURED on `TRANSPORT` at 33.35 mm/s and a 59.94 Hz field rate, orders
    one to five, the collisions that are EXACT rather than merely close:

        entry guide  x3  =  5.30782 Hz  =  pinch roller  x5
        exit guide   x3  =  5.30782 Hz  =  pinch roller  x5

    and they are exact because the declared diameters are 6.0 mm and
    10.0 mm, a ratio of 3 to 5, so the rates stand in that ratio whatever
    the tape speed. A flat-spotted pinch roller and a flat-spotted guide put
    energy at the same frequency by construction. The guide pair is
    separated from the pinch roller by the axis it acts on - the guides act
    on tension and the pinch roller on speed - so this pair is recoverable
    from the complex residual even though no rate difference reaches it;
    `acts_on` is carried in the result so a caller can see when that escape
    is available and when it is not.

    THE REELS COLLIDE BY BAND, NOT BY LINE, and the case that matters has no
    axis to escape by. A reel sweeps `rate_band_hz` rather than a rate, so
    its order-m harmonic sweeps m times that band. Either reel's SECOND
    order spans 0.2359 to 0.8846 Hz, which contains the impedance roller's
    0.8166 Hz fundamental. Both parts act on TENSION, so a two-lobe reel and
    an eccentric impedance roller are the same line on the same axis, and
    only the reel's DRIFT - which moves as the pack unwinds, where the
    roller's does not - tells them apart.

    THE DRUM IS NOT A DIAMETER CASE. Its rate is fixed by the format rather
    than computed from its 62 mm, so any small-integer ratio between the
    drum's diameter and another part's is arithmetic with no cause behind
    it; `by_diameter` is therefore reported only for two parts that both
    take their rate from a diameter.

    Give `sample_rate_hz` to compare the harmonics where the MEASUREMENT
    puts them rather than where the transport does, which is the only
    comparison that means anything once a harmonic folds.
    """
    tol = float(SEARCH_TOLERANCE if tolerance is None else tolerance)
    top = max(int(max_order), 1)
    entries = [dict(entry) for entry in rates]

    def span(entry, order):
        """The band an order occupies: a point for a line, m times the band
        for a reel."""
        band = entry.get("rate_band_hz")
        if band is not None:
            low, high = (float(band[0]), float(band[1]))
        else:
            low = high = float(entry.get("rate_hz", np.nan))
        return order * low, order * high

    out: List[Dict[str, object]] = []
    for i, first in enumerate(entries):
        for second in entries[i + 1:]:
            for m in range(1, top + 1):
                for n in range(1, top + 1):
                    a_low, a_high = span(first, m)
                    b_low, b_high = span(second, n)
                    if not all(np.isfinite([a_low, a_high, b_low, b_high])):
                        continue
                    scale = max(a_high, b_high, 1e-30)
                    if a_low - b_high > tol * scale or b_low - a_high > tol * scale:
                        continue
                    centre_a = 0.5 * (a_low + a_high)
                    centre_b = 0.5 * (b_low + b_high)
                    separation = abs(centre_a - centre_b) / scale
                    exact = bool(a_low == a_high and b_low == b_high
                                 and separation <= 1e-12)
                    both_from_diameter = ("diameter_m" in first
                                          and "diameter_m" in second
                                          and first.get("kind") != "drum"
                                          and second.get("kind") != "drum")
                    item: Dict[str, object] = {
                        "first": first.get("name"), "first_order": m,
                        "second": second.get("name"), "second_order": n,
                        "at_hz": centre_a, "against_hz": centre_b,
                        "separation": float(separation),
                        "exact": exact,
                        "by_diameter": bool(exact and both_from_diameter),
                        "by_band": bool(a_low != a_high or b_low != b_high),
                        "same_axis": bool(first.get("acts_on")
                                          == second.get("acts_on")),
                        "first_acts_on": first.get("acts_on"),
                        "second_acts_on": second.get("acts_on"),
                    }
                    if sample_rate_hz is not None:
                        item["at_alias_hz"] = alias_hz(centre_a, sample_rate_hz)
                        item["against_alias_hz"] = alias_hz(centre_b,
                                                            sample_rate_hz)
                    out.append(item)
    return sorted(out, key=lambda entry: (not entry["exact"],
                                          entry["separation"],
                                          entry["first_order"] + entry["second_order"]))



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

