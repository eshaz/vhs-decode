"""The deck log: head swaps, maintenance, captures and disciplined time.

Ethan: *"Log head swaps and maintenance dates. Gap geometry resets at a swap
while preamp and servo don't - that's a natural experiment for separating
head terms from deck terms."* And: *"GPS/NTP-disciplined timestamps on
captures. Over a long acquisition, capture-chain drift becomes a hidden
variable correlated with everything; timestamps let you regress it out."*

WHY A LOG IS A MEASUREMENT INSTRUMENT. The identifiability work found that
the magnetic mechanisms collapse to 1.58 distinguishable of six on a
frequency response alone - gap and azimuth are collinear at 1.000, spacing
and thickness at 0.994 - and `tape_model.confounded` lists the pairs a
single measurement cannot split. None of them can be split by more careful
fitting, because the confound is in the physics of one head reading one
tape. What splits them is a CHANGE that touches one side of a pair and
not the other, and the only such changes available are the ones a deck's
life provides: a head swap resets every head-owned parameter and leaves
the preamp, the servo and the tape untouched; a cleaning resets the debris
share of the spacing and not the wear; a belt or pinch-roller change
resets the transport's rates and nothing magnetic. Each is a natural
experiment, and it is only usable if it was WRITTEN DOWN with a date.

The same goes for time. A capture chain drifts - its crystal ages and
warms, the deck's servo reference too - slowly enough that within one
short decode it is a constant, and over a long acquisition it is a slow
term correlated with everything else that is slow. A timestamp per capture
disciplined to GPS or NTP makes that term a REGRESSOR with a known axis;
a file's modification time is a clock nobody checked, and the log records
it as such rather than pretending.

THE FILE IS TOML, ONE PER BENCH, edited by hand: Ethan's standing rule for
tooling is one readable configuration file per tool and nothing in the
shell environment. `tools/ringing_measure/deck_log.toml` is the template,
populated with what is KNOWN about the existing captures and nothing that
is not: the deck and the speed are in the file names, the capture device
and its settings come from the two readme files, the deck's reference
crystals from its schematic, and the only timestamps that exist are file
modification times, recorded as undisciplined. The crystal figures that
bound the drift regressor live in the log's `[clock]` table with their
status spelled out, NOT in this module: they are assumptions about a
bench, and a bench's assumptions belong in its file. `docs/DECK_LOG.md`
describes the schema.

WHAT EACH EVENT RESETS is a physical statement and is kept in one table,
`EVENT_RESETS`, with the reasoning beside each entry. The head-owned
parameters are `head_model.CHANNEL_PARAMETERS` (azimuth, gain, height,
protrusion - the four per-channel figures a drum's two heads each carry)
plus the gap and the head's own spacing shares; the tape-owned ones are
`tape_model.variations`; the deck terms are what neither of those lists:
the preamp, the servo, the automatic level control and the transport's
rotating parts.

Run `PYTHONPATH=/workspaces/vhs-decode python3 -m vhsdecode.models.deck_log
tools/ringing_measure/deck_log.toml` to validate a log and list what it
can separate.
"""

import datetime as _datetime
import sys
import tomllib
from typing import Dict, List, Optional, Sequence

import numpy as np

from vhsdecode.models import head_model, tape_model
from vhsdecode.models.head_model import effective_sample_size

SCHEMA_VERSION = 1

# The ways a capture's clock can be disciplined. A file's modification time
# is a computer clock nobody checked against anything, and it marks the END
# of the write, not the start of the capture.
TIMESTAMP_DISCIPLINE = ("gps", "ntp", "none")

# The keys the `[clock]` table must carry for the drift regressor's bound,
# each an assumption the log labels with its `status`.
CLOCK_KEYS = ("aging_ppm_per_year", "thermal_ppm_per_kelvin",
              "temperature_excursion_kelvin", "status")

SECONDS_PER_YEAR = 365.25 * 86400.0

# --------------------------------------------------------------------------
# what is owned by what
# --------------------------------------------------------------------------

HEAD_TERMS = tuple(head_model.CHANNEL_PARAMETERS) + (
    "gap_m",              # the read gap: fixed by the head's construction
    "head_efficiency",    # the head's flux-to-voltage conversion
    "head_spacing_m",     # the head's own share of the Wallace separation
    "debris_spacing_m",   # the deposited share, which cleaning removes
)

TAPE_TERMS = tuple(str(entry["parameter"]) for entry in tape_model.variations())

DECK_TERMS = (
    "preamp_gain",
    "preamp_equalization",
    "record_current",     # the record amplifier's drive, frozen into a tape
    "agc",                # the automatic level control's set point
    "deemphasis",         # the playback de-emphasis network
    "servo_reference",    # the drum and capstan servos' reference and loops
    "capstan_rate",
    "pinch_roller_rate",
    "idler_rate",
    "guide_alignment",    # tape-path geometry: guides, tracking
)

EVENT_RESETS: Dict[str, Dict[str, object]] = {
    "head swap": {
        "resets": tuple(HEAD_TERMS),
        "why": "a new head has its own gap, azimuth, protrusion and height, "
               "and arrives clean; the preamp, the servo and the tape do not "
               "change because a drum's heads were replaced",
    },
    "head cleaning": {
        "resets": ("debris_spacing_m",),
        "why": "cleaning removes deposited oxide and nothing else: the wear "
               "(protrusion) stays, the tape's own surface roughness stays, "
               "which is what separates the debris share of the spacing "
               "from both",
    },
    "transport service": {
        "resets": ("capstan_rate", "pinch_roller_rate", "idler_rate"),
        "why": "belts, idlers and pinch rollers set the mechanical rates the "
               "transport model predicts; replacing them moves those rates "
               "and touches nothing magnetic or electronic",
    },
    "electronics repair": {
        "resets": ("preamp_gain", "preamp_equalization", "agc", "deemphasis",
                   "servo_reference"),
        "why": "a recapped or repaired board changes the electrical terms "
               "and leaves the heads and the transport alone",
    },
    "alignment": {
        "resets": ("guide_alignment",),
        "why": "tape-path alignment moves the tracking offset - the one "
               "variation that moves the two heads in opposite directions",
    },
}

# Pairs a frequency-domain fit apportions arbitrarily that `tape_model`
# does not list because one member is not a tape term: the head's and the
# tape's shares of one Wallace spacing, the head's efficiency against the
# preamp's gain, and the gap-against-azimuth collinearity the
# identifiability work measured at coherence 1.000.
CHAIN_CONFOUNDS = (
    ("head_spacing_m", "spacing_m", "one Wallace spacing, two shares"),
    ("debris_spacing_m", "head_spacing_m", "spacing shares"),
    ("debris_spacing_m", "spacing_m", "spacing shares"),
    ("head_efficiency", "preamp_gain", "flat gain"),
    ("gain_db", "preamp_gain", "flat gain"),
    ("gap_m", "azimuth_error_degrees", "identifiability: coherence 1.000"),
    ("protrusion_m", "head_spacing_m", "protrusion sets spacing"),
)


# --------------------------------------------------------------------------
# loading and validating
# --------------------------------------------------------------------------

def load(path: str, strict: bool = True) -> Dict[str, object]:
    """Read a deck log. With `strict` the log must validate."""
    with open(path, "rb") as handle:
        log = tomllib.load(handle)
    if strict:
        report = validate(log)
        if report["errors"]:
            raise ValueError("deck log %s: %s" % (
                path, "; ".join(report["errors"])))
    return log


def _as_datetime(value) -> Optional[_datetime.datetime]:
    if isinstance(value, _datetime.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_datetime.timezone.utc)
        return value
    if isinstance(value, _datetime.date):
        return _datetime.datetime(value.year, value.month, value.day,
                                  tzinfo=_datetime.timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = _datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _as_datetime(parsed)
    return None


def validate(log: Dict[str, object]) -> Dict[str, object]:
    """Errors refuse the log; warnings name what is unrecorded.

    A capture without a deck cannot enter any natural experiment, and one
    without a timestamp source cannot enter the drift regressor, so both
    are errors rather than gaps; so is a `[clock]` table without its
    labelled figures, because a bound nobody stated is not a bound. A deck
    whose make is unknown is allowed, with a warning, because the captures
    still exist and the log is where the gap is meant to be visible.
    """
    errors: List[str] = []
    warnings: List[str] = []
    if int(log.get("schema", 0)) != SCHEMA_VERSION:
        errors.append(f"schema must be {SCHEMA_VERSION}")
    clock = log.get("clock")
    if not isinstance(clock, dict):
        errors.append("no [clock] table: the drift regressor's bound is "
                      "unstated")
    else:
        for key in CLOCK_KEYS:
            if key not in clock or clock[key] in ("", None):
                errors.append(f"[clock] lacks {key}")
    devices = {d.get("id"): d for d in log.get("capture_devices", [])
               if d.get("id")}
    tapes = {t.get("id"): t for t in log.get("tapes", []) if t.get("id")}
    decks = {}
    for index, deck in enumerate(log.get("decks", [])):
        identifier = deck.get("id")
        if not identifier:
            errors.append(f"deck #{index} has no id")
            continue
        if identifier in decks:
            errors.append(f"deck id {identifier!r} appears twice")
        decks[identifier] = deck
        if not deck.get("make") or str(deck.get("make")).lower() == "unknown":
            warnings.append(f"deck {identifier!r}: make/model unrecorded")
        if not deck.get("serial"):
            warnings.append(f"deck {identifier!r}: serial number unrecorded")
        for swap in deck.get("head_swaps", []):
            if _as_datetime(swap.get("date")) is None:
                errors.append(f"deck {identifier!r}: a head swap has no date")
        for job in deck.get("maintenance", []):
            if _as_datetime(job.get("date")) is None:
                errors.append(f"deck {identifier!r}: a maintenance entry has "
                              f"no date")
            kind = job.get("kind")
            if kind not in EVENT_RESETS or kind == "head swap":
                errors.append(f"deck {identifier!r}: maintenance kind "
                              f"{kind!r} is not one of "
                              f"{sorted(k for k in EVENT_RESETS if k != 'head swap')}")
    for index, capture in enumerate(log.get("captures", [])):
        name = capture.get("file", f"capture #{index}")
        if not capture.get("deck"):
            errors.append(f"{name}: no deck")
        elif capture["deck"] not in decks:
            errors.append(f"{name}: deck {capture['deck']!r} is not in the log")
        discipline = capture.get("timestamp_discipline")
        if discipline not in TIMESTAMP_DISCIPLINE:
            errors.append(f"{name}: timestamp_discipline must be one of "
                          f"{TIMESTAMP_DISCIPLINE}, got {discipline!r}")
        if not capture.get("timestamp_source"):
            errors.append(f"{name}: no timestamp_source (what clock gave "
                          f"the timestamp)")
        if _as_datetime(capture.get("timestamp")) is None:
            errors.append(f"{name}: no usable timestamp")
        elif discipline == "none":
            warnings.append(f"{name}: timestamp is undisciplined "
                            f"({capture.get('timestamp_source')})")
        for field in ("sample_rate_hz", "bit_depth"):
            if field not in capture:
                errors.append(f"{name}: no {field}")
        device = capture.get("device")
        if device and device not in devices:
            errors.append(f"{name}: capture device {device!r} is not in "
                          f"[[capture_devices]]")
        tape = capture.get("tape")
        if tape and tape not in tapes:
            warnings.append(f"{name}: tape {tape!r} is not in [[tapes]]")
        for field in ("tape", "speed", "device"):
            if not capture.get(field):
                warnings.append(f"{name}: {field} unrecorded")
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "decks": len(decks), "captures": len(log.get("captures", []))}


def clock_assumptions(log: Dict[str, object]) -> Dict[str, object]:
    """The labelled crystal figures from the log's `[clock]` table."""
    clock = log.get("clock")
    if not isinstance(clock, dict) or any(k not in clock for k in CLOCK_KEYS):
        raise ValueError("the deck log has no complete [clock] table; the "
                         "drift bound needs " + ", ".join(CLOCK_KEYS))
    return dict(clock)


# --------------------------------------------------------------------------
# natural experiments
# --------------------------------------------------------------------------

def events_of(deck: Dict[str, object]) -> List[Dict[str, object]]:
    """A deck's events in date order, each with its kind and what it resets."""
    out = []
    for swap in deck.get("head_swaps", []):
        entry = dict(swap)
        entry["kind"] = "head swap"
        entry["when"] = _as_datetime(swap.get("date"))
        out.append(entry)
    for job in deck.get("maintenance", []):
        entry = dict(job)
        entry["when"] = _as_datetime(job.get("date"))
        out.append(entry)
    out = [e for e in out if e["when"] is not None and e.get("kind") in EVENT_RESETS]
    out.sort(key=lambda e: e["when"])
    for entry in out:
        entry["resets"] = tuple(EVENT_RESETS[entry["kind"]]["resets"])
    return out


def _variation_names() -> Dict[str, str]:
    return {str(v["parameter"]): str(v["name"]) for v in tape_model.variations()}


def separated_by(resets: Sequence[str]) -> Dict[str, List[Dict[str, object]]]:
    """Which confounded pairs an event with these resets can split.

    A pair is split when the event moves exactly one of its members. The
    pairs come from `tape_model.confounded` (the physics-level list, each
    checked against `tape_model.separable` to say whether the project's
    instruments could already tell the two apart) and from
    `CHAIN_CONFOUNDS`, the head-against-deck and head-against-tape pairs
    that any frequency-domain fit apportions arbitrarily.
    """
    resets = set(resets)
    names = _variation_names()
    split: List[Dict[str, object]] = []
    still: List[Dict[str, object]] = []
    seen = set()

    def consider(a: str, b: str, source: str):
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        entry: Dict[str, object] = {"parameter": a, "with": b, "source": source}
        if a in names and b in names:
            entry["instruments_separate"] = tape_model.separable(names[a], names[b])
        else:
            entry["instruments_separate"] = False
        if (a in resets) != (b in resets):
            entry["moved"] = a if a in resets else b
            split.append(entry)
        else:
            still.append(entry)

    for pair in tape_model.confounded():
        consider(str(pair["parameter"]), str(pair["with"]), "tape_model")
    for a, b, source in CHAIN_CONFOUNDS:
        consider(a, b, source)
    return {"split": split, "still_confounded": still}


def natural_experiments(log: Dict[str, object]) -> List[Dict[str, object]]:
    """Every pair of captures on one deck that straddles an event, with what
    the pair can separate.

    The strongest pair replays the SAME TAPE before and after, because then
    every tape term is held exactly; the next best plays the same source
    signal on different tapes, which holds the picture and not the tape.
    Both are returned, ranked, and each names what it holds and what the
    event moved. Pairs that straddle several events are returned once per
    event, because each event's resets are a different experiment.

    Dating across a swap is refused here too: `head_model.aging_rate` is a
    wear clock, and a swap resets it, so the calibration must not span one
    (`aging_spans` gives the valid spans, `dating_valid` checks a pair).
    """
    decks = {d["id"]: d for d in log.get("decks", []) if d.get("id")}
    captures = [c for c in log.get("captures", [])
                if c.get("deck") in decks
                and _as_datetime(c.get("timestamp")) is not None]
    out: List[Dict[str, object]] = []
    for identifier, deck in decks.items():
        mine = [c for c in captures if c["deck"] == identifier]
        for event in events_of(deck):
            before = [c for c in mine
                      if _as_datetime(c["timestamp"]) < event["when"]]
            after = [c for c in mine
                     if _as_datetime(c["timestamp"]) >= event["when"]]
            separation = separated_by(event["resets"])
            for earlier in before:
                for later in after:
                    same_tape = bool(earlier.get("tape")
                                     and earlier.get("tape") == later.get("tape"))
                    same_source = bool(earlier.get("source")
                                       and earlier.get("source") == later.get("source"))
                    interval = (_as_datetime(later["timestamp"])
                                - _as_datetime(earlier["timestamp"]))
                    holds = ["tape terms (same tape replayed)"] if same_tape \
                        else (["the picture (same source signal)"] if same_source
                              else [])
                    holds += [f"deck terms: {t}" for t in DECK_TERMS
                              if t not in event["resets"]]
                    out.append({
                        "deck": identifier,
                        "event": event["kind"],
                        "event_date": event["when"].date().isoformat(),
                        "event_detail": {k: v for k, v in event.items()
                                         if k not in ("when", "resets", "kind")},
                        "before": earlier.get("file"),
                        "after": later.get("file"),
                        "interval_days": interval.total_seconds() / 86400.0,
                        "same_tape": same_tape,
                        "same_source": same_source,
                        "rank": 0 if same_tape else (1 if same_source else 2),
                        "moved": list(event["resets"]),
                        "holds": holds,
                        "separates": separation["split"],
                        "still_confounded": separation["still_confounded"],
                        "dating_valid": event["kind"] != "head swap",
                    })
    out.sort(key=lambda e: (e["rank"], e["deck"], e["event_date"]))
    return out


def aging_spans(log: Dict[str, object], deck_id: str) -> List[Dict[str, object]]:
    """The date spans within which `head_model.aging_rate` may be
    calibrated on this deck: between head swaps, since a swap resets the
    wear the rate is measured from."""
    deck = next((d for d in log.get("decks", []) if d.get("id") == deck_id),
                None)
    if deck is None:
        return []
    swaps = [e["when"] for e in events_of(deck) if e["kind"] == "head swap"]
    edges = [None] + swaps + [None]
    spans = []
    for start, stop in zip(edges[:-1], edges[1:]):
        spans.append({"from": start.date().isoformat() if start else "the deck's first use",
                      "to": stop.date().isoformat() if stop else "now",
                      "why": "a head swap resets the wear clock"})
    return spans


def dating_valid(log: Dict[str, object], capture_a: Dict[str, object],
                 capture_b: Dict[str, object]) -> Dict[str, object]:
    """Whether `head_model.interval_years` may be applied to two captures:
    the same deck, and no head swap between them."""
    if capture_a.get("deck") != capture_b.get("deck"):
        return {"valid": False, "why": "different decks: the wear rate is "
                                       "per machine"}
    deck = next((d for d in log.get("decks", [])
                 if d.get("id") == capture_a.get("deck")), None)
    if deck is None:
        return {"valid": False, "why": "deck not in the log"}
    a, b = (_as_datetime(capture_a.get("timestamp")),
            _as_datetime(capture_b.get("timestamp")))
    if a is None or b is None:
        return {"valid": False, "why": "a capture has no timestamp"}
    low, high = min(a, b), max(a, b)
    for event in events_of(deck):
        if event["kind"] == "head swap" and low < event["when"] <= high:
            return {"valid": False,
                    "why": f"a head swap on {event['when'].date()} lies "
                           f"between them and resets the wear clock"}
    return {"valid": True, "why": "same deck, no swap between"}


# --------------------------------------------------------------------------
# the drift regressor
# --------------------------------------------------------------------------

def drift_bound_ppm(span_seconds: float, assumptions: Dict[str, object]
                    ) -> Dict[str, float]:
    """The most a crystal-referenced chain can drift over `span_seconds`,
    in ppm: aging over the span plus the thermal excursion, from the log's
    labelled `[clock]` figures."""
    aging = (float(assumptions["aging_ppm_per_year"]) * float(span_seconds)
             / SECONDS_PER_YEAR)
    thermal = (float(assumptions["thermal_ppm_per_kelvin"])
               * float(assumptions["temperature_excursion_kelvin"]))
    return {"aging_ppm": aging, "thermal_ppm": thermal,
            "total_ppm": aging + thermal, "span_seconds": float(span_seconds),
            "status": str(assumptions.get("status", ""))}


def time_axis(log: Dict[str, object],
              series_by_capture: Dict[str, Dict[str, np.ndarray]]
              ) -> Dict[str, object]:
    """One absolute time axis across captures, from their timestamps, with
    the discipline of every clock that entered it.

    Within a capture the sample clock is the axis and is always usable.
    Across captures the axis is only as good as the timestamps: a single
    undisciplined one makes the cross-capture offsets unknown to within
    the capture's own length and the computer clock's error, and the
    regressor says so instead of fitting through it.
    """
    captures = {c.get("file"): c for c in log.get("captures", [])}
    times, values, labels, disciplines = [], [], [], []
    for name, series in series_by_capture.items():
        capture = captures.get(name)
        if capture is None:
            raise KeyError(f"{name} is not in the log")
        stamp = _as_datetime(capture.get("timestamp"))
        if stamp is None:
            raise ValueError(f"{name} has no timestamp")
        within = np.asarray(series["time_s"], dtype=np.float64)
        times.append(stamp.timestamp() + within)
        values.append(np.asarray(series["values"], dtype=np.float64))
        labels.append(np.full(len(within), name, dtype=object))
        disciplines.append(capture.get("timestamp_discipline", "none"))
    if not times:
        return {"time_s": np.array([]), "values": np.array([]),
                "capture": np.array([]), "disciplines": [],
                "cross_capture_usable": False, "why": "no series"}
    usable = all(d in ("gps", "ntp") for d in disciplines)
    return {"time_s": np.concatenate(times),
            "values": np.concatenate(values),
            "capture": np.concatenate(labels),
            "disciplines": disciplines,
            "cross_capture_usable": bool(usable and len(times) > 1),
            "why": ("every timestamp is GPS or NTP disciplined" if usable
                    else "an undisciplined timestamp leaves the offset "
                         "between captures unknown; only within-capture "
                         "drift is fitted")}


def drift_regressor(log: Dict[str, object],
                    series_by_capture: Dict[str, Dict[str, np.ndarray]],
                    nominal: Optional[float] = None,
                    order: int = 1) -> Dict[str, object]:
    """The capture-chain drift as a regressor, fitted and bounded.

    `series_by_capture` maps a capture file (as named in the log) to
    {"time_s": seconds from the capture's start, "values": the series}.
    The slow term is a polynomial of `order` in time (a straight line by
    default: aging is linear over any span a bench sees, and a thermal
    excursion adds curvature only if the temperature actually moved).
    `nominal` converts the series to parts per million so the fitted
    drift can be checked against the crystal bound from the log's
    `[clock]` table; a level series has no nominal and gets no bound.

    Returns the fitted slow term, the residual with it removed, the slope
    with an error that respects the residual's serial correlation
    (`head_model.effective_sample_size`, on the residual), and the verdict
    against the bound. With an undisciplined timestamp among several
    captures the joint axis is refused and each capture is fitted alone.
    """
    assumptions = clock_assumptions(log)
    axis = time_axis(log, series_by_capture)
    t, y = axis["time_s"], axis["values"]
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    out: Dict[str, object] = {"axis": axis, "count": int(len(y)),
                              "assumptions": assumptions}
    if len(y) < 6:
        out.update({"usable": False, "why": "too few samples"})
        return out
    if not axis["cross_capture_usable"] and len(set(axis["capture"][good])) > 1:
        out.update({"usable": False, "why": axis["why"], "per_capture": {}})
        for name in sorted(set(axis["capture"][good])):
            out["per_capture"][name] = drift_regressor(
                log, {name: series_by_capture[name]}, nominal, order)
        return out
    centre = float(np.mean(t))
    span = float(t.max() - t.min()) or 1.0
    x = (t - centre) / span                      # conditioned to [-0.5, 0.5]
    design = np.column_stack([x ** k for k in range(order + 1)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    slow = design @ coefficients
    residual = y - slow
    degrees = max(len(y) - design.shape[1], 1)
    variance = float(residual @ residual) / degrees
    covariance = variance * np.linalg.pinv(design.T @ design)
    effective = effective_sample_size(residual)
    inflation = np.sqrt(len(y) / max(effective, 1.0))
    slope_per_second = float(coefficients[1] / span)
    slope_error = float(np.sqrt(max(covariance[1, 1], 1e-300)) / span * inflation)
    total = slope_per_second * span
    out.update({
        "usable": True,
        "slow_term": slow, "residual": residual, "time_s": t,
        "slope_per_second": slope_per_second,
        "slope_error": slope_error,
        "t": float(abs(slope_per_second) / max(slope_error, 1e-300)),
        "effective_samples": float(effective),
        "total_drift": float(total),
        "total_drift_error": float(slope_error * span),
        "span_seconds": span,
        "coefficients": coefficients,
        "residual_std": float(np.std(residual)),
        "raw_std": float(np.std(y)),
    })
    if nominal:
        bound = drift_bound_ppm(span, assumptions)
        total_ppm = total / float(nominal) * 1e6
        out.update({
            "total_drift_ppm": float(total_ppm),
            "total_drift_ppm_error": float(slope_error * span / float(nominal) * 1e6),
            "bound": bound,
            "within_crystal_bound": bool(abs(total_ppm) <= bound["total_ppm"]),
            "verdict": ("consistent with capture-chain drift"
                        if abs(total_ppm) <= bound["total_ppm"]
                        else "exceeds what a crystal can do: not the "
                             "capture chain"),
        })
    else:
        out["verdict"] = "no nominal given: the crystal bound does not apply"
    return out


def regress_out(target_values: np.ndarray, regressor: Dict[str, object]
                ) -> Dict[str, object]:
    """Remove the fitted slow term's SHAPE from another series measured on
    the same axis: the share of `target_values` that follows the drift
    regressor, by least squares, and what remains."""
    if not regressor.get("usable"):
        return {"usable": False, "why": regressor.get("why")}
    shape = np.asarray(regressor["slow_term"], dtype=np.float64)
    shape = shape - shape.mean()
    target = np.asarray(target_values, dtype=np.float64)
    if len(target) != len(shape):
        raise ValueError("the target must be sampled on the regressor's axis")
    good = np.isfinite(target)
    power = float(shape[good] @ shape[good])
    if power <= 0:
        return {"usable": False, "why": "the drift term is flat"}
    centred = target[good] - target[good].mean()
    coefficient = float(shape[good] @ centred) / power
    remaining = target - coefficient * shape
    r = coefficient * np.sqrt(power) / max(np.sqrt(float(centred @ centred)), 1e-300)
    return {"usable": True, "coefficient": coefficient,
            "correlation": float(r), "residual": remaining,
            "explained": float(np.clip(r * r, 0.0, 1.0))}


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def describe(log: Dict[str, object]) -> str:
    """A readable account of a log: validation, decks, events, experiments
    and aging spans."""
    lines = []
    report = validate(log)
    lines.append(f"decks {report['decks']}, captures {report['captures']}, "
                 f"{'valid' if report['ok'] else 'INVALID'}")
    for error in report["errors"]:
        lines.append(f"  error: {error}")
    for warning in report["warnings"]:
        lines.append(f"  unrecorded: {warning}")
    clock = log.get("clock", {})
    if isinstance(clock, dict) and all(k in clock for k in CLOCK_KEYS):
        bound = drift_bound_ppm(SECONDS_PER_YEAR, clock)
        lines.append(f"clock bound: {bound['aging_ppm']:.1f} ppm/year aging + "
                     f"{bound['thermal_ppm']:.1f} ppm thermal "
                     f"({clock['status']})")
    for deck in log.get("decks", []):
        events = events_of(deck)
        lines.append(f"deck {deck.get('id')}: {deck.get('make', '?')} "
                     f"{deck.get('model', '')}, {len(events)} dated events")
        for event in events:
            lines.append(f"  {event['when'].date()} {event['kind']}: resets "
                         f"{', '.join(event['resets'])}")
        for span in aging_spans(log, deck.get("id")):
            lines.append(f"  aging_rate may be calibrated from {span['from']} "
                         f"to {span['to']}")
    experiments = natural_experiments(log)
    lines.append(f"natural experiments: {len(experiments)}")
    for experiment in experiments[:20]:
        splits = ", ".join(f"{s['parameter']} from {s['with']}"
                           for s in experiment["separates"])
        lines.append(f"  {experiment['deck']} {experiment['event']} on "
                     f"{experiment['event_date']}: {experiment['before']} -> "
                     f"{experiment['after']} ({experiment['interval_days']:.0f} "
                     f"days, {'same tape' if experiment['same_tape'] else 'same source' if experiment['same_source'] else 'different material'}); "
                     f"separates {splits or 'nothing new'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python3 -m vhsdecode.models.deck_log <deck_log.toml>")
        return 2
    log = load(argv[0], strict=False)
    print(describe(log))
    return 0 if validate(log)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
