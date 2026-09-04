"""The colour framing, measured rather than counted.

The parity was `(field.field_number // 2) % 2`, a free-running counter with no
measurement in it. Counted over this repository's own shipped decodes, 638
carry the correct ascending sequence and 45 do not.
"""

import numpy as np
import pytest

import vhsdecode.chroma as chroma
from vhsdecode.models import colour_framing as cf


class _RF:
    pass


class _Field:
    def __init__(self, phase, number, is_first, rf):
        self.burst_phase_avg = phase
        self.field_number = number
        self.isFirstField = is_first
        self.rf = rf


def _run(count=8, phase_a=10.0, phase_b=190.0, rf=None):
    """a field sequence whose burst phase alternates every two fields, which
    is what the two colour frames do"""
    rf = rf if rf is not None else _RF()
    out = []
    for n in range(count):
        is_first = (n % 2 == 0)
        phase = phase_a if ((n + 1) // 2) % 2 == 0 else phase_b
        parity, measured = chroma.colour_frame_parity(
            _Field(phase, n, is_first, rf))
        identifier, base = chroma.ntsc_color_framing_map[(is_first, parity)]
        out.append((identifier, base, measured))
    return out


def test_the_sequence_ascends_which_is_the_contract():
    """`lddecode/core.py:4557` requires fieldPhaseID to step by one with wrap.
    The old counter produced 1, 4, 3, 2 - the right pattern off by one field."""
    identifiers = [row[0] for row in _run()]
    assert identifiers == [1, 2, 3, 4, 1, 2, 3, 4]
    steps = [(b - a) % 4 for a, b in zip(identifiers, identifiers[1:])]
    assert all(step == 1 for step in steps)


def test_it_is_measured_and_not_counted():
    """Every field after the anchor reads the burst, so a stalled counter, a
    dropped field or a splice cannot slip the framing."""
    rows = _run()
    assert all(row[2] for row in rows), "every field must be measured"

    # a counter that stalls - the failure seen in the wild, where readloc did
    # not advance and the framing inverted for the rest of the decode
    rf = _RF()
    stalled = []
    for n, number in enumerate([0, 1, 2, 2, 3, 4, 5, 6]):
        is_first = (n % 2 == 0)
        phase = 10.0 if ((n + 1) // 2) % 2 == 0 else 190.0
        parity, _ = chroma.colour_frame_parity(
            _Field(phase, number, is_first, rf))
        stalled.append(chroma.ntsc_color_framing_map[(is_first, parity)][0])
    assert stalled == [1, 2, 3, 4, 1, 2, 3, 4], "a stalled counter must not slip it"


def test_it_falls_back_to_the_counter_when_the_burst_is_unusable():
    """Ethan: *"If that measurement fails for some reason (it shouldn't) fall
    back to the previous of alternating."*"""
    rf = _RF()
    for phase in (None, float("nan")):
        parity, measured = chroma.colour_frame_parity(
            _Field(phase, 6, True, rf))
        assert not measured
        assert parity == (6 // 2) % 2


def test_the_anchor_waits_for_a_first_field():
    """ID 1 is keyed (isFirstField, parity 0), so anchoring anything else
    would start the sequence in the wrong place."""
    rf = _RF()
    parity, measured = chroma.colour_frame_parity(_Field(10.0, 0, False, rf))
    assert not measured, "a second field must not set the anchor"
    assert not hasattr(rf, "_colour_frame_anchor")
    parity, measured = chroma.colour_frame_parity(_Field(10.0, 1, True, rf))
    assert measured and parity == 0


def test_the_model_and_the_decoder_agree_on_the_sequence_length():
    assert cf.sequence("NTSC")["fields_in_sequence"] == 4
    assert cf.sequence("PAL")["fields_in_sequence"] == 8
    assert len(set(i for i, _ in chroma.ntsc_color_framing_map.values())) == 4
