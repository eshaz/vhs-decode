"""Colour framing derived from the carrier's own relationship to hsync.

Ethan: *"The color framing can change if the recording changes. Let's look at
the specs, and just determine this from the actual color carrier relationship
to the hsync. If that measurement fails for some reason (it shouldn't) fall
back to the previous of alternating."*

WHAT IS WRONG TODAY, MEASURED. The decoder decides NTSC colour framing from a
free-running counter - `(field.isFirstField, (field.field_number // 2) % 2)`
at `vhsdecode/chroma.py` - with no measurement in it, and the contract check
that would catch a wrong sequence is switched off (`check_phase=False`).
Counted over the shipped decodes in `/output`: 638 carry the correct
ascending 1,2,3,4 and 45 do not, and the largest decode in the archive
(`countdown_on`, 59021 fields) is wrong throughout - 59015 descending steps
against 5 ascending. The same capture decoded from two different seek points
comes out with OPPOSITE framing, because a counter starting on a first field
and one starting on a second field differ by exactly one.

AND A COUNTER CANNOT BE RIGHT IN PRINCIPLE, which is Ethan's point. A counter
carries the framing forward from wherever it started; a tape carries whatever
was recorded, and a tape can hold more than one recording. A splice, a
re-record, a stop and restart - each is a new source with its own framing,
and no counter survives one. The framing has to be MEASURED, per field, from
the signal.

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

WHY THE TAPE STILL CARRIES IT, DESPITE THE COLOUR-UNDER BEING LINE-LOCKED.
This is the objection that has to be answered, because on its face the
colour-under carrier destroys the sequence: it is EXACTLY 40 f_H, so it
advances exactly 40 whole cycles a line, which is zero degrees a line, and a
carrier that advances nothing cannot carry a four-field pattern.

The answer is that the recording machine's APC does not synthesise the
colour-under from the line rate alone - it locks it against the INCOMING
BURST. So the phase written to tape is the difference between the source's
subcarrier and the deck's line-locked local oscillator, and the source's
subcarrier is exactly the thing that carries the SC/H sequence. What survives
on tape is therefore the source's framing, folded into the colour-under
phase measured AGAINST HSYNC - which is precisely the relationship Ethan
names, and is why it must be read against hsync rather than in isolation.

THAT ARGUMENT IS NOT A MEASUREMENT, and this module does not pretend it is.
`decide` takes a measured SC/H phase and returns a framing; `confidence`
says whether the measurement is worth believing; and when it is not, the
caller falls back. The measurement that would settle the argument itself is
named in `settle_the_question` below - it needs the burst phase BEFORE the
decoder imposes its own target, over at least sixteen consecutive fields.
"""

from fractions import Fraction
from typing import Dict, Optional, Sequence

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


def expected_phase(field_index: int, system: str = "NTSC") -> float:
    """The SC/H phase this field of the sequence should show, in degrees."""
    spec = sequence(system)
    advance = float(spec["degrees_per_field"])
    return float((advance * int(field_index)) % 360.0)


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
    """The measurement that decides whether the tape carries framing at all.

    Stated as a function so it cannot be lost in prose. The argument above -
    that the recording APC locks the colour-under against the incoming burst,
    so the source's SC/H sequence survives in the colour-under phase measured
    against hsync - is a mechanism, not evidence.
    """
    return {
        "export": ("the pre-imposition burst phase against the sync datum, "
                   "per field, for at least 16 consecutive fields"),
        "not": ("anything measured after upconvert_chroma_phase_comp, which "
                "adds the burst phase to the local oscillator and so returns "
                "the decoder's own imposed target rather than the tape's"),
        "test": ("a period-4 pattern of two states 180 degrees apart, keyed "
                 "to field index; present means the tape carries the "
                 "framing, absent means it does not"),
        "caution": ("a period-4 two-state pattern is cyclically identical to "
                    "its own rotation, so the test establishes that the "
                    "SEQUENCE is intact and not which field is first"),
    }
