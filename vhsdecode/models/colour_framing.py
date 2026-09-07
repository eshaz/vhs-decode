"""Colour framing derived from the carrier's own relationship to hsync.

Ethan: *"The color framing can change if the recording changes. Let's look at
the specs, and just determine this from the actual color carrier relationship
to the hsync. If that measurement fails for some reason (it shouldn't) fall
back to the previous of alternating."*

WHAT IS WRONG TODAY, MEASURED. The decoder decides NTSC colour framing from a
free-running counter - `(field.isFirstField, (field.field_number // 2) % 2)`
at `vhsdecode/chroma.py`. Counted over the shipped decodes in `/output`: 638
carry the ascending 1,2,3,4 and 45 do not, and the same capture decoded from
two different seek points comes out with OPPOSITE framing, because a counter
starting on a first field and one starting on a second differ by exactly one.

A COUNTER CANNOT BE RIGHT IN PRINCIPLE. It carries the framing forward from
wherever it started; a tape carries whatever was recorded, and a tape can
hold more than one recording. A splice, a re-record, a stop and restart -
each is a new source with its own framing, and no counter survives one.

AND THE FRAMING STILL CANNOT BE MEASURED FROM THE CHROMA, which is the part
that took a withdrawn change to establish. Both statements are true at once:
the counter is not right, and the obvious replacement is not available. The
section below has the arithmetic. What follows from holding both is that the
counter STAYS - as the least wrong thing available - and that its failures
are made visible rather than silently corrected, which is why the contract
check in `buildmetadata` is now performed rather than merely parameterised.

THE ARITHMETIC, DERIVED AND NOT QUOTED. In 525/60 the subcarrier is
`455/2` times the line rate, so along one line the subcarrier turns

    455/2 cycles = 227.5 cycles = half a cycle = 180 degrees

and along one field of 262.5 lines it turns

    262.5 * 227.5 = 59718.75 cycles = 0.75 of a cycle = 270 degrees

so the subcarrier-to-horizontal phase advances 270 degrees a field and
returns after FOUR fields, because 4 * 270 = 1080 = three whole turns. That
is the four-field sequence, and it is not a convention - it is the smallest
`n` for which `n * 262.5 * 455/2` is a whole number of cycles.

In 625/50 the subcarrier is `1135/4 f_H + 25 Hz`, the quarter-cycle term
forces eight fields, and the 25 Hz offset contributes exactly one extra cycle
per frame, which is an integer and so does not change the sequence length.

THE TAPE DOES NOT CARRY IT, AND THIS MODULE ONCE ARGUED THAT IT DID.

The objection is that the colour-under is EXACTLY 40 f_H, so it advances a
whole number of cycles per line and per field, and a carrier that advances
nothing cannot carry a four-field pattern. An earlier version of this
docstring answered that the recording APC locks the colour-under against the
INCOMING BURST, so the source's SC/H sequence survives folded into the
recorded phase. THAT ANSWER WAS WRONG, and the arithmetic that settles it is
short enough to state here.

In units of the line rate the subcarrier is 455/2 and the record heterodyne
oscillator sits at `fsc + f_cu = 227.5 + 40 = 267.5`. Over one field of
262.5 lines,

    subcarrier    262.5 x 227.5 = 59718.75 cycles, fractional 3/4 -> 270 deg
    record LO     262.5 x 267.5 = 70218.75 cycles, fractional 3/4 -> 270 deg
    difference                            10500 cycles, fractional 0

The two fractional parts are IDENTICAL. Whatever the APC locks to, the 270
degrees per field cancels exactly in the down-conversion, and 40 f_H was
chosen so that it would. No measurement of the recorded burst against the
sync datum can recover the colour frame, because the quantity is not there.

A decoder change built on the earlier argument was shipped and withdrawn. It
read `burst_phase_avg`, which on a synthetic tape carrying no colour frame at
all still alternates - because it carries the DECODER'S own rotation index,
not the tape's. Measured against the counter it inverted 11 of 20 fields and
destroyed the run-of-two structure. `chroma.colour_frame_parity` records
that in full.

SO THIS MODULE HOLDS THE ARITHMETIC AND NOT AN INSTRUMENT. `sequence`,
`expected_phase` and `decide` are correct and are what a framing decision
must satisfy; what is missing is a measurable input, and it has to come from
somewhere the record heterodyne does not cancel - the luma side's own
sync-to-subcarrier relationship before the chroma is split off, or an
external reference. `settle_the_question` below names what would be needed
and no longer claims the recorded burst can supply it.
"""

from fractions import Fraction
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# 525/60 and 625/50, as exact rationals so the advance is exact.
LINE_RATE = {"NTSC": Fraction(30000, 1001) * 525, "PAL": Fraction(15625)}
SUBCARRIER_RATIO = {"NTSC": Fraction(455, 2), "PAL": Fraction(1135, 4)}
# The 625 subcarrier carries a further +25 Hz, which is exactly one cycle per
# frame and so changes no sequence length.
PAL_OFFSET_HZ = Fraction(25)
LINES_PER_FIELD = {"NTSC": Fraction(525, 2), "PAL": Fraction(625, 2)}


def _system(system: str) -> str:
    name = str(system).upper()
    if name in ("NTSC", "525", "M", "525/60"):
        return "NTSC"
    if name in ("PAL", "625", "625/50", "B", "G", "I"):
        return "PAL"
    raise ValueError(f"no colour framing defined for {system!r}")


def sequence(system: str = "NTSC") -> Dict[str, object]:
    """THE FRAMING ARITHMETIC, derived from the line rate and nothing else.

    Returns the subcarrier's advance per line and per field, and the number
    of fields the sequence takes to close - computed as the smallest `n`
    whose total advance is a whole number of cycles, rather than quoted.
    """
    key = _system(system)
    per_line = SUBCARRIER_RATIO[key]
    per_field = per_line * LINES_PER_FIELD[key]
    length = None
    for n in range(1, 33):
        if (per_field * n).denominator == 1:
            length = n
            break
    return {
        "system": key,
        "cycles_per_line": per_line,
        "degrees_per_line": float((per_line % 1) * 360),
        "cycles_per_field": per_field,
        "degrees_per_field": float((per_field % 1) * 360),
        "fields_in_sequence": length,
        "states": length,
        "why": ("the sequence closes at the smallest n whose total advance "
                "is a whole number of cycles; it is derived, not a "
                "convention"),
    }


def reference_line_offset(field_index: int, system: str = "NTSC") -> int:
    """How many WHOLE LINES the field's reference sync edge is from field
    one's, which is where the burst is actually measured against sync.

    BT.1700 (525-line, item "Field identification"): *"Field I is the field
    in which the first zero crossing of burst on line 10 is positive going;
    field III is negative going."* The reference for fields I and III is
    line 10 and for fields II and IV it is line 273, so consecutive
    reference edges are 263 lines apart (10 to 273) then 262 (273 to the
    next frame's 10) - the ceiling and the floor of the 262.5-line field.
    A sync edge is always a whole number of lines from another sync edge,
    which is the whole point: the half-line only exists mid-line.
    """
    key = _system(system)
    per_field = LINES_PER_FIELD[key]
    frames, odd = divmod(int(field_index), 2)
    frame_lines = int(per_field * 2)
    first_step = int(-(-per_field // 1))          # ceiling: 263 for 525
    return frames * frame_lines + (first_step if odd else 0)


def expected_phase(field_index: int, system: str = "NTSC",
                   at: str = "sync_edge") -> float:
    """The SC/H phase this field of the sequence should show, in degrees.

    `at="sync_edge"` (the default, and the only one an instrument can read)
    advances the subcarrier by the WHOLE LINES between reference sync
    edges: 227.5 cycles a line makes every whole-line advance a multiple of
    180 degrees, so the 525-line sequence at the sync edge is
    [0, 180, 180, 0] and its four states are (phase, field parity).

    `at="half_line"` advances by the specification's 262.5 lines a field -
    [0, 270, 180, 90] - which is the phase at the START of each field, and
    on the odd fields that instant is mid-line where no sync edge and no
    SC/H definition exists.

    THE FIRST VERSION OF THIS RETURNED THE HALF-LINE SEQUENCE AS THE THING
    TO VALIDATE A DECODER AGAINST, and `validate_mapping` then reported the
    shipped map ninety degrees out on fields II and IV. The framing agent
    measured the raw CVBS of three generator captures - about 31,700 lines
    - and found the burst-to-sync phase taking exactly two values, 0 and
    180, at every sync edge, ascending I, II, III, IV at all 123
    consecutive field pairs; validated against the sync-edge sequence, the
    shipped map has zero error on all four entries. The ninety degrees was
    the half-line, 113.75 cycles, and it was this derivation's, not the
    decoder's.
    """
    spec = sequence(system)
    if at == "half_line":
        advance = float(spec["degrees_per_field"])
        return float((advance * int(field_index)) % 360.0)
    if at != "sync_edge":
        raise ValueError("at must be 'sync_edge' or 'half_line'")
    key = _system(system)
    cycles = SUBCARRIER_RATIO[key] * reference_line_offset(field_index, key)
    return float((cycles % 1) * 360)


def decide(sch_phase_deg: float, system: str = "NTSC",
           tolerance_deg: float = 30.0) -> Dict[str, object]:
    """WHICH FIELD OF THE SEQUENCE A MEASURED SC/H PHASE PUTS US ON.

    The states are `360 / n` apart - 90 degrees for NTSC's four, 45 for
    PAL's eight - so a measurement good to better than half that spacing
    identifies the state outright. The margin to the next state is returned
    so a caller can refuse a reading that sits between two.
    """
    spec = sequence(system)
    states = int(spec["states"])
    spacing = 360.0 / states
    measured = float(sch_phase_deg) % 360.0
    errors = np.array([abs(((measured - expected_phase(i, system) + 180.0)
                            % 360.0) - 180.0) for i in range(states)])
    best = int(np.argmin(errors))
    ordered = np.sort(errors)
    margin = float(ordered[1] - ordered[0]) if states > 1 else 360.0
    return {
        "field_in_sequence": best,
        "field_phase_id": best + 1,
        "error_deg": float(errors[best]),
        "margin_deg": margin,
        "state_spacing_deg": spacing,
        "decided": bool(errors[best] <= float(tolerance_deg)
                        and margin >= 0.5 * spacing),
        "why": ("the states are 360/n apart, so a phase good to better than "
                "half that spacing identifies the state; the margin says "
                "whether this one is"),
    }


def confidence(sch_phase_deg: Sequence[float], system: str = "NTSC"
               ) -> Dict[str, object]:
    """IS THE MEASUREMENT WORTH BELIEVING? Ethan's fallback condition.

    A run of consecutive fields must show the sequence ADVANCING at the
    specified rate. Framing read independently per field can be right on
    average and wrong on any given one; a run that advances correctly is
    evidence the measurement is tracking rather than guessing.

    Reports the share of consecutive pairs whose advance matches the
    specification. A measurement that cannot clear this should fall back to
    alternating rather than assert a framing it has not established.
    """
    phases = np.asarray(sch_phase_deg, dtype=np.float64).ravel()
    spec = sequence(system)
    if phases.size < 3:
        return {"trustworthy": False, "why": "too few fields to see a run"}
    advance = float(spec["degrees_per_field"])
    steps = np.diff(phases) % 360.0
    error = np.abs(((steps - advance + 180.0) % 360.0) - 180.0)
    spacing = 360.0 / int(spec["states"])
    agreeing = float(np.mean(error < 0.5 * spacing))
    return {
        "trustworthy": bool(agreeing > 0.9),
        "agreeing_fraction": agreeing,
        "expected_advance_deg": advance,
        "median_advance_deg": float(np.median(steps)),
        "fields": int(phases.size),
        "why": ("the sequence must ADVANCE at the specified rate across "
                "consecutive fields; per-field agreement alone can be right "
                "on average and wrong on every field"),
    }


def alternating_fallback(field_number: int, is_first_field: bool,
                         system: str = "NTSC") -> Dict[str, object]:
    """THE FALLBACK Ethan names: the previous behaviour, kept as a fallback.

    This is what the decoder does today for every field - a counter and the
    field parity. It is kept because a measurement that fails should degrade
    to the old behaviour rather than to nothing, and it is named
    `alternating_fallback` rather than `framing` so that no caller reaches
    for it by accident believing it to be a measurement.
    """
    spec = sequence(system)
    states = int(spec["states"])
    index = (int(field_number) // 2) % (states // 2) if states > 2 else 0
    return {
        "field_in_sequence": int((2 * index + (0 if is_first_field else 1))
                                 % states),
        "measured": False,
        "why": ("a counter carries the framing forward from wherever it "
                "started and cannot survive a splice or a re-record; this is "
                "a fallback and never an answer"),
    }


def framing(sch_phase_deg: Optional[float], field_number: int,
            is_first_field: bool, system: str = "NTSC",
            tolerance_deg: float = 30.0) -> Dict[str, object]:
    """The framing: measured where the measurement holds, counted where not.

    Ethan's rule exactly - derive it from the carrier's relationship to
    hsync, and fall back to alternating only when that fails.
    """
    if sch_phase_deg is not None and np.isfinite(sch_phase_deg):
        decided = decide(sch_phase_deg, system, tolerance_deg)
        if decided["decided"]:
            decided["measured"] = True
            return decided
    fallback = alternating_fallback(field_number, is_first_field, system)
    fallback["field_phase_id"] = fallback["field_in_sequence"] + 1
    return fallback


def settle_the_question() -> Dict[str, str]:
    """WHERE A MEASURABLE COLOUR FRAME COULD STILL COME FROM.

    The question of whether the RECORDED BURST carries it is settled and the
    answer is no - the arithmetic is in this module's docstring and the
    cancellation is exact. What is open is whether anything else does.
    """
    return {
        "settled": ("the colour-under cannot carry it: subcarrier and record "
                    "oscillator both advance a fractional 3/4 cycle per "
                    "field, so the difference advances a whole number and "
                    "the four-field sequence cancels exactly"),
        "not": ("anything derived from burst_phase_avg, which alternates "
                "even on a synthetic tape carrying no colour frame, because "
                "it carries the decoder's own per-line rotation index"),
        "candidate": ("the LUMA side's sync-to-subcarrier relationship "
                      "measured before the chroma is split off, which the "
                      "record heterodyne never touches"),
        "closed": ("the colour-under replica the demodulated luma carries, "
                   "which `vhsdecode/models/colour_lock.py` now measures at "
                   "a resultant of 0.98-0.99 on SP: a NEW channel, but not "
                   "this quantity. `colour_under_field_advance` shows why - "
                   "on 525/60 it advances 10500 cycles a field, fractional "
                   "zero, so it carries no per-field phase at all"),
        "external": ("a reference outside the tape - the standard's line-10 "
                     "burst zero-crossing rule, or a known colour in the "
                     "picture, neither of which is available from the sync "
                     "area alone"),
        "caution": ("a period-4 two-state pattern is cyclically identical to "
                    "its own rotation, so any such test establishes that a "
                    "SEQUENCE is intact and never which field is first"),
    }


def colour_under_field_advance(system: str = "NTSC") -> Dict[str, object]:
    """WHY THE LUMA'S COLOUR-UNDER REPLICA CANNOT CARRY THE COLOUR FRAME.

    A new channel arrived with `vhsdecode/models/colour_lock.py`: the
    colour-under replica the demodulated luma carries, which the decoder's
    up-conversion never touches and which locks at a resultant of 0.98 to 0.99
    on the SP decodes. It is the first genuinely independent reading of the
    tape's colour-under phase this arc has had, so the framing question has to
    be put to it - and the answer is no, for a reason that is one division.

    The colour-under is a MULTIPLE OF THE LINE RATE by specification. On 525/60
    it is exactly forty times it, so over a field of 262.5 lines it advances

        262.5 x 40 = 10500 cycles, fractional part ZERO

    and a carrier that advances a whole number of cycles a field has no
    per-field phase to carry. That is the same reason the module docstring
    gives for the recorded burst, arriving through a different door: the format
    chose 40 f_H so the field-to-field phase would cancel, and it cancels in
    every channel the colour-under reaches, including this one.

    On 625/50 the printed 626.953 kHz carrier gives 40.124992 times the line
    rate and a fractional advance of 0.060 per field, which is NOT a claim that
    PAL's colour-under carries an eight-field sequence. The printed figure is
    rounded to the hertz, and the fraction is a property of that rounding
    rather than of the format; the value is returned so the difference between
    a derived and a printed constant stays visible, exactly as
    `colour_under.carrier_provenance` insists.
    """
    from vhsdecode.models import colour_under

    key = _system(system)
    line_rate = colour_under.line_rate_hz(key)
    carrier = colour_under.carrier_hz(key)
    lines = LINES_PER_FIELD[key]
    ratio = carrier / line_rate
    per_field = ratio * float(lines)
    fractional = per_field % 1.0
    return {
        "system": key,
        "carrier_in_line_rates": float(ratio),
        "cycles_per_field": float(per_field),
        "fractional_advance": float(fractional),
        "can_carry_a_field_sequence": bool(fractional > 1e-9),
        "printed_carrier": colour_under.carrier_provenance(key)["printed"],
        "why": ("a carrier that advances a whole number of cycles per field "
                "holds no per-field phase; on 525/60 the colour-under advances "
                "exactly 10500"),
    }


def dimensions() -> Dict[str, object]:
    """WHICH AXES THE COLOUR FRAME HAS, and the one it does not.

    Ethan, 2026-09-06: *"All of them need to be used and all of them need to
    model all dimensions."* Two of the three are here and derived; the third
    is absent for a reason, and stating it is better than manufacturing a
    number to fill the column.

      FREQUENCY   the sequence LENGTH is a frequency statement and nothing
                  else: it is the smallest `n` for which `n` fields of
                  `525/2` lines at `455/2` subcarrier cycles a line is a
                  whole number of cycles. Four for 525/60, eight for 625/50,
                  and the ratio is what decides it.
      TIME        the advance, 270 degrees a field on 525/60, which is a
                  phase and therefore a time - `expected_phase` gives it in
                  degrees and the subcarrier converts it to seconds.
      AMPLITUDE   ABSENT, and it cannot be otherwise. The quantity is a
                  state index out of `n`; an index has no magnitude, and a
                  magnitude attached to one would be a property of the
                  estimator rather than of the framing.

    WHAT WOULD CARRY AN AMPLITUDE is the CONFIDENCE of a measured framing -
    the resultant of the phase the decision was taken on - and `confidence`
    is written and ready for it. It has no input on this medium, for the
    reason this module's docstring proves at length: the colour-under
    advances a whole number of cycles a field, so the sequence is not on the
    tape to be measured. The axis is missing because the measurement is,
    not because the axis was overlooked.
    """
    ntsc = sequence("NTSC")
    return {
        "frequency": {
            "present": True,
            "quantity": "the sequence length, from the subcarrier's ratio to "
                        "the line rate",
            "value": int(ntsc["states"]),
        },
        "time": {
            "present": True,
            "quantity": "the advance per field, as a phase and so as a time",
            "degrees_per_field": float(ntsc["degrees_per_field"]),
        },
        "amplitude": {
            "present": False,
            "why": "the framing is a state index out of n and an index has "
                   "no magnitude; the quantity that would carry one is the "
                   "confidence of a MEASURED framing, and the colour-under "
                   "advances a whole number of cycles a field so there is "
                   "nothing on the tape to measure it from",
            "would_be": "colour_framing.confidence, which has no input here",
        },
        "why": ("an axis that is absent is reported as absent with the "
                "reason, because filling the column with a number that is "
                "not a measurement is worse than leaving it empty"),
    }


def validate_mapping(mapping: Dict[Tuple[int, int], Tuple[int, float]],
                     system: str = "NTSC") -> Dict[str, object]:
    """VALIDATE A DECODER'S FRAMING MAP AGAINST THE SPECIFIED SEQUENCE.

    Ethan: *"I am seeing a miss-match in the color framing. Look at the spec
    and use our color measurements, specifically genlocking to validate the
    mapping."*

    The mapping is `(isFirstField, parity) -> (fieldPhaseID, base phase)`.
    `fieldPhaseID` names which field of the colour sequence this is, and the
    base phase is what the decoder then IMPOSES on the chroma relative to
    the line grid. The specification fixes the second given the first, AT
    THE SYNC EDGE, because that is the only place a burst-to-sync phase is
    defined (`expected_phase`).

    THE SHIPPED NTSC MAP PASSES. Measured against the sync-edge sequence:

        fieldPhaseID 1   spec   0 deg   map   0 deg    agrees
        fieldPhaseID 2   spec 180 deg   map 180 deg    agrees
        fieldPhaseID 3   spec 180 deg   map 180 deg    agrees
        fieldPhaseID 4   spec   0 deg   map   0 deg    agrees

    A previous version of this docstring reported fields II and IV ninety
    degrees out and called the map's two base phases insufficient for four
    states. That was this module's own error - it validated against the
    half-line sequence [0, 270, 180, 90], whose odd entries fall mid-line
    where there is no sync edge. The four states are (phase, field parity),
    two phases each, and the map carries both. The genlock measurement that
    settled it: the source's SC/H holds to 0 +- 2 degrees with a drift under
    0.02 degrees a field, and the decoder's burst lock to 0.44 degrees a
    line, so nothing in the measurement chain could have hidden a ninety
    degree error had there been one.

    What this does not decide is which SMPTE field a given identifier
    names: cvbsdecode's identifiers were measured one colour frame off the
    standard (its ID 1 lands on field III, deterministically, 48 of 48
    aligned fields), and the tape cannot supply the sequence at all, so
    vhsdecode's counter is seek-relative by necessity. Both are recorded
    in `colour_frame_parity` and neither is a phase error.
    """
    spec = sequence(system)
    states = int(spec["states"])
    rows = []
    for key, value in sorted(mapping.items()):
        identifier, base = value[0], float(value[1])
        wanted = expected_phase(identifier - 1, system)
        error = ((base - wanted + 180.0) % 360.0) - 180.0
        rows.append({
            "key": key,
            "field_phase_id": identifier,
            "specified_deg": wanted,
            "mapped_deg": base,
            "error_deg": error,
            "agrees": bool(abs(error) < 1.0),
        })
    distinct_mapped = sorted({row["mapped_deg"] for row in rows})
    distinct_spec = sorted({expected_phase(i, system) for i in range(states)})
    disagreeing = [row for row in rows if not row["agrees"]]
    return {
        "rows": rows,
        "system": spec["system"],
        "states_specified": len(distinct_spec),
        "states_mapped": len(distinct_mapped),
        "distinct_specified_deg": distinct_spec,
        "distinct_mapped_deg": distinct_mapped,
        "disagreeing": len(disagreeing),
        "worst_error_deg": (max(abs(row["error_deg"]) for row in rows)
                            if rows else 0.0),
        "consistent": not disagreeing,
        "why": ("fieldPhaseID names which field of the sequence this is and "
                "the base phase is what gets imposed; the specification "
                "fixes the second given the first, so a map with fewer "
                "distinct phases than the sequence has states cannot keep "
                "both promises"),
    }
