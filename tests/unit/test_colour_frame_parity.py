"""The colour frame cannot be measured from the colour-under, and why.

Ethan asked for the framing to be derived from the colour carrier's
relationship to hsync. An attempt was made, shipped, and WITHDRAWN: the
quantity it read cannot carry the colour frame, and the arithmetic that shows
so is short enough to be a test.
"""

from fractions import Fraction

import numpy as np
import pytest

import vhsdecode.chroma as chroma
from vhsdecode.models import colour_framing as cf


class _RF:
    pass


class _Field:
    def __init__(self, phase, number, is_first):
        self.burst_phase_avg = phase
        self.field_number = number
        self.isFirstField = is_first
        self.rf = _RF()


def test_the_colour_under_cannot_carry_the_colour_frame():
    """THE PROOF THE WITHDRAWN CHANGE NEEDED AND DID NOT HAVE.

    In units of the line rate the subcarrier is 455/2 and the record
    heterodyne oscillator is `fsc + f_cu = 267.5`. Over 262.5 lines both
    advance a fractional 3/4 of a cycle, so the 270 degrees per field that
    carries the four-field sequence cancels EXACTLY in the down-conversion
    and the colour-under advances a whole number of cycles. 40 f_H was chosen
    so that it would.
    """
    lines = Fraction(525, 2)
    subcarrier = Fraction(455, 2)
    colour_under = Fraction(40)
    oscillator = subcarrier + colour_under

    for carrier in (subcarrier, oscillator):
        advance = carrier * lines
        assert advance - int(advance) == Fraction(3, 4), carrier

    down_converted = (subcarrier - oscillator) * lines
    assert down_converted.denominator == 1, "must be a whole number of cycles"
    assert down_converted % 1 == 0
    # which is to say: no phase advance per field, so no four-field sequence
    assert float((down_converted - int(down_converted))) == 0.0


def test_the_parity_is_the_counter_and_ignores_the_burst_entirely():
    """The withdrawn version read `burst_phase_avg`. It must not: on a
    synthetic tape carrying no colour frame at all that quantity still
    alternates, because it carries the DECODER'S own rotation index."""
    for number in range(8):
        expected = (number // 2) % 2
        for phase in (0.0, 90.0, 180.0, 270.0, float("nan"), None):
            parity, measured = chroma.colour_frame_parity(
                _Field(phase, number, number % 2 == 0))
            assert parity == expected, (number, phase)
            assert not measured, "nothing here is measured"


def test_the_specification_model_still_knows_the_real_arithmetic():
    """`colour_framing` is correct and is not what was withdrawn - it holds
    the specification's own 270 degrees per field and four-field sequence.
    What it lacks is a measurable input, which is a different problem."""
    spec = cf.sequence("NTSC")
    assert spec["degrees_per_field"] == pytest.approx(270.0)
    assert spec["fields_in_sequence"] == 4
    assert cf.sequence("PAL")["fields_in_sequence"] == 8


def test_the_map_covers_four_identifiers_from_two_base_phases():
    """A note from the survey worth keeping visible: the map carries only two
    distinct target phases where the sequence advances 270 degrees a field,
    so the emitted absolute SC/H takes two states rather than four. The four
    IDENTIFIERS are still distinct, which is what `fieldPhaseID` promises."""
    identifiers = {i for i, _ in chroma.ntsc_color_framing_map.values()}
    bases = {b for _, b in chroma.ntsc_color_framing_map.values()}
    assert identifiers == {1, 2, 3, 4}
    assert bases == {0, 180}


def test_the_sequence_at_the_sync_edge_is_two_phases_and_the_parity():
    """Whole-line advances at 227.5 cycles a line are multiples of 180
    degrees, so at the sync edge the four-field sequence is [0, 180, 180,
    0]; the half-line sequence [0, 270, 180, 90] exists only mid-line."""
    assert [cf.expected_phase(i) for i in range(4)] == [0.0, 180.0, 180.0, 0.0]
    assert [cf.expected_phase(i, at="half_line") for i in range(4)] == [
        0.0, 270.0, 180.0, 90.0]
    assert [cf.reference_line_offset(i) for i in range(4)] == [0, 263, 525, 788]


def test_the_shipped_map_keeps_both_promises_at_the_sync_edge():
    """Measured on about 31,700 generator lines by the framing agent: the
    burst-to-sync phase takes exactly two values at every sync edge and the
    shipped map has zero error on all four identifiers."""
    report = cf.validate_mapping(chroma.ntsc_color_framing_map)
    assert report["consistent"]
    assert report["worst_error_deg"] == pytest.approx(0.0, abs=1.0)
    assert report["states_mapped"] == 2 and report["states_specified"] == 2


def test_a_map_carrying_the_half_line_sequence_would_fail():
    """The sequence a decoder must NOT impose: the mid-line phases."""
    half_line = {(1, 0): (1, 0.0), (0, 1): (2, 270.0),
                 (1, 1): (3, 180.0), (0, 0): (4, 90.0)}
    report = cf.validate_mapping(half_line)
    assert not report["consistent"]
    assert report["worst_error_deg"] == pytest.approx(90.0, abs=1.0)
