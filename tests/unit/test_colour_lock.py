"""The colour lock, its constants, and the head-switch event it detects.

Ethan, this session: *"There is a fixed offset between the colour-under carrier
and the colour lock derived from the luma; use it as another measurement
component."* and *"Colour-under carrier is constant (locked to chroma), colour
carrier is constant (locked to luma), time is constant; any deviation is
residual."*

The tests below are built on synthesised fields whose truth is known, so what
they check is that the instrument reads what was put into it - the measurements
on real tape live in the modules' own docstrings. Every field here is
constructed from the format's own arithmetic and nothing is fitted.
"""

import numpy as np
import pytest

from vhsdecode.models import chroma_head_switch as hs
from vhsdecode.models import colour_free_luma as cfl
from vhsdecode.models import colour_lock as cl
from vhsdecode.models import colour_under

# Four times the NTSC subcarrier and 910 samples a line: the decoder's own
# output grid, so a line is 227.5 subcarrier cycles and exactly 40 colour-under
# cycles, which is the arithmetic the four-line period comes from.
SAMPLE_RATE = 4.0 * colour_under.subcarrier_hz("NTSC")
WIDTH = 910
HEIGHT = 263
ACTIVE = (134, 894)
QUIET = (40, 240)
CENTRE = 32768.0


def _field(hue_deg, rotation_per_line_deg, offset_deg,
           chroma_amplitude=4000.0, luma_amplitude=200.0, seed=0,
           disturb=None):
    """A luma and chroma field carrying one known colour-under phase.

    `disturb` maps a line number to a pair of extra phases, one for each
    channel, so a test can put a head-switch event in by hand and ask the
    instrument to find it.
    """
    rng = np.random.default_rng(seed)
    index = np.arange(WIDTH, dtype=np.float64)
    carrier = colour_under.carrier_hz("NTSC")
    luma = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    chroma = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    for line in range(HEIGHT):
        phase = np.radians(hue_deg + rotation_per_line_deg * line)
        extra_chroma, extra_luma = (disturb or {}).get(line, (0.0, 0.0))
        # The luma carries the colour-under at its own carrier, untouched.
        luma[line] = luma_amplitude * np.cos(
            2 * np.pi * carrier * index / SAMPLE_RATE
            + phase + np.radians(extra_luma))
        # The chroma carries the same vector at the subcarrier, with the
        # decoder's up-conversion offset folded in.
        chroma[line] = chroma_amplitude * np.cos(
            2 * np.pi * (SAMPLE_RATE / 4.0) * index / SAMPLE_RATE
            + phase + np.radians(offset_deg + extra_chroma))
    luma += rng.normal(0.0, 1.0, luma.shape)
    chroma += rng.normal(0.0, 1.0, chroma.shape)
    return luma, chroma + CENTRE


def test_the_three_constants_come_from_the_line_rate_and_nothing_else():
    """DIRECTIVE 7's three constants, each reproduced from the format."""
    for system, multiple in (("NTSC", 40.0), ("PAL", 40.0)):
        constants = cl.constants(system)
        rate = constants["line_rate_hz"]
        expected = multiple * rate + (1953.0 if system == "PAL" else 0.0)
        assert constants["colour_under_hz"] == pytest.approx(expected)
        assert constants["line_period_s"] == pytest.approx(1.0 / rate)
        assert constants["deviation_is"] == "residual"
    ntsc = cl.constants("NTSC")
    # 455/2 times the line rate, which is the subcarrier's definition.
    assert ntsc["subcarrier_hz"] == pytest.approx(455.0 / 2.0
                                                  * ntsc["line_rate_hz"])
    # Locked to the other channel, each way round, as the directive states.
    assert ntsc["colour_under_locked_to"] == "chroma"
    assert ntsc["subcarrier_locked_to"] == "luma"


def test_the_offset_is_recovered_and_is_fixed():
    """The instrument reads back the offset it was given, and calls it fixed."""
    luma, chroma = _field(hue_deg=20.0, rotation_per_line_deg=90.0,
                          offset_deg=37.0)
    result = cl.measure(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET)
    # A field built with a constant offset must read as a perfect lock.
    assert result["resultant"] > 0.99
    # And every residue's entry must be the offset that was put in. The
    # ninety-degree rotation and the four-times-subcarrier frame's own
    # hundred and eighty close after four lines, so all four agree here.
    for degrees in result["offset"]["degrees"]:
        # Two degrees, not one. The instrument reads 38.67 / 36.15 / 38.67 /
        # 36.15 on this field with the noise switched off as well as on, so
        # the departure is the measurement's own and not the noise's: a common
        # +0.4 degrees and a +-1.3 degree two-line alternation, from filtering
        # each 910-sample line on its own, where the filter's edge transient
        # meets a line-start phase that alternates because a line is 227.5
        # subcarrier cycles. It is a bias of the synthesis grid, it is smaller
        # than anything the module claims on tape, and it is stated rather
        # than tuned away.
        assert abs(((degrees - 37.0) + 180.0) % 360.0 - 180.0) < 2.0
    assert result["scale"]["rms_deg"] < 2.0


def test_the_period_is_four_lines_and_a_two_line_table_cannot_hold_it():
    """WHY the table has four entries.

    With the colour-under rotating ninety degrees a line, a table of period two
    is asked to hold two different phases in one slot and its resultant length
    collapses. The four-line table is not a choice of window; it is the period
    the two frames close on.
    """
    luma, chroma = _field(hue_deg=0.0, rotation_per_line_deg=90.0,
                          offset_deg=15.0)
    product = cl.line_product(
        cl.chroma_envelope(chroma, SAMPLE_RATE),
        cl.luma_colour_under(luma, SAMPLE_RATE), ACTIVE)
    four = cl.lock_offset(product, QUIET, period=4)
    two = cl.lock_offset(product, QUIET, period=2)
    assert four["resultant"] > 0.99
    assert two["resultant"] < four["resultant"]


def test_a_field_with_no_colour_under_in_the_luma_does_not_lock():
    """THE CONTROL, in synthesis: no colour-under in the luma, no lock.

    This is the y-only capture's behaviour reproduced where the truth is
    known. The luma still carries energy at the colour-under carrier - noise
    does - so an amplitude test would pass. The lock is what fails.
    """
    rng = np.random.default_rng(7)
    _, chroma = _field(hue_deg=0.0, rotation_per_line_deg=90.0,
                       offset_deg=15.0)
    luma = rng.normal(0.0, 200.0, (HEIGHT, WIDTH))
    result = cl.measure(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET)
    assert result["resultant"] < 0.5


def test_the_residual_is_zero_where_nothing_happened():
    """DIRECTIVE 7's second half: the deviation, and only the deviation."""
    luma, chroma = _field(hue_deg=-40.0, rotation_per_line_deg=-90.0,
                          offset_deg=5.0)
    result = cl.measure(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET)
    quiet = result["residual_deg"][QUIET[0]:QUIET[1]]
    assert np.abs(quiet).max() < 2.0


def test_the_detector_finds_a_planted_head_switch_and_the_quiet_lines_do_not():
    """DIRECTIVE 1's detection half, against a planted event.

    The event is put in the way the tape puts it in: BOTH channels depart,
    in opposite directions, which is what the real measurement in
    `chroma_head_switch` found. The detector reads their difference and so
    sees twice either one.
    """
    disturb = {259: (-35.0, +39.0), 260: (+4.0, -21.0)}
    luma, chroma = _field(hue_deg=10.0, rotation_per_line_deg=90.0,
                          offset_deg=-12.0, disturb=disturb)
    result = cl.measure(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET)
    found = hs.detect(result["residual_deg"], QUIET)
    assert 259 in found["lines"]
    assert found["peak_line"] == 259
    # Nothing above the event is claimed.
    assert all(line >= QUIET[1] for line in found["lines"])


def test_the_compensation_comes_from_the_chroma_and_not_from_the_difference():
    """THE FINDING THAT SHAPED THE MODULE, as an assertion.

    On the same planted field, the chroma's own departure is -35 degrees and
    the luma-chroma residual is about -74. A correction built on the residual
    would rotate by twice what is needed; the table built by `accumulate`
    rotates by what is needed.
    """
    disturb = {259: (-35.0, +39.0)}
    luma, chroma = _field(hue_deg=10.0, rotation_per_line_deg=90.0,
                          offset_deg=-12.0, disturb=disturb)
    result = cl.measure(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET)
    residual = result["residual_deg"][259]
    table = hs.accumulate(None, result["chroma_envelope"], ACTIVE, [259],
                          QUIET)
    own = np.degrees(hs.phases(table)[259])
    assert own == pytest.approx(-35.0, abs=2.0)
    assert residual == pytest.approx(-74.0, abs=4.0)
    # The two differ by about a factor of two, which is the whole point.
    assert abs(residual) > 1.7 * abs(own)


def test_the_correction_removes_the_planted_phase_and_touches_nothing_else():
    """DIRECTIVE 1's compensation half."""
    disturb = {259: (-35.0, +39.0), 260: (+18.0, -6.0)}
    luma, chroma = _field(hue_deg=10.0, rotation_per_line_deg=90.0,
                          offset_deg=-12.0, disturb=disturb)
    result = cl.measure(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET)
    table = hs.phases(hs.accumulate(None, result["chroma_envelope"], ACTIVE,
                                    [259, 260], QUIET))
    corrected = hs.correct_field(chroma, table, centre=CENTRE)
    after = cl.measure(luma, corrected, SAMPLE_RATE, ACTIVE, QUIET)
    envelope = after["chroma_envelope"]
    for line in (259, 260):
        left = hs.chroma_departure(envelope, ACTIVE, line, QUIET)
        assert abs(np.degrees(np.angle(left))) < 3.0
    # Untouched lines are bit-identical, so the correction's reach is exactly
    # the lines the detector named.
    assert np.array_equal(corrected[QUIET[0]:QUIET[1]],
                          chroma[QUIET[0]:QUIET[1]].astype(np.float64))


def test_rotating_a_line_is_a_pure_phase_and_keeps_its_amplitude():
    """The rotation is the analytic signal's, so it moves phase and nothing
    else - which is what makes it safe to apply inside the picture."""
    index = np.arange(WIDTH)
    line = 5000.0 * np.cos(2 * np.pi * index / 4.0 + 0.3)
    turned = hs.rotate_line(line, np.radians(37.0))
    trim = slice(64, WIDTH - 64)  # the Hilbert transform's own edge effect
    assert np.std(turned[trim]) == pytest.approx(np.std(line[trim]), rel=0.02)
    from scipy import signal as sps
    before = np.angle(sps.hilbert(line)[trim])
    after = np.angle(sps.hilbert(turned)[trim])
    shift = np.degrees(np.angle(np.mean(np.exp(1j * (before - after)))))
    assert shift == pytest.approx(37.0, abs=1.0)


def test_the_quiet_span_is_derived_from_the_detector_not_declared():
    """No line number is chosen by hand: the reference is where the detector
    is silent."""
    span = hs.quiet_span(HEIGHT, (20, 262), [258, 259, 260])
    assert span == (20, 256)
    with pytest.raises(ValueError):
        hs.quiet_span(HEIGHT, (20, 262), [21])


def test_the_colour_free_luma_removes_the_planted_coupling():
    """DIRECTIVE 4, against a field where the coupling is known exactly."""
    luma, chroma = _field(hue_deg=25.0, rotation_per_line_deg=90.0,
                          offset_deg=-60.0, luma_amplitude=200.0)
    result = cfl.process(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET,
                         held_out=True)
    assert result["share"]["removed"] > 0.9
    feedback = result["feedback"]
    assert feedback["dimensions"]["phase"]["usable"] is True
    assert feedback["dimensions"]["amplitude"]["usable"] is True
    # The rate axis is refused, and the refusal is carried in the result so a
    # consumer cannot use it by accident.
    assert feedback["dimensions"]["rate"]["usable"] is False


def test_the_colour_free_luma_declines_when_there_is_no_colour_under():
    """The y-only control's behaviour, with no gate and no burst test."""
    rng = np.random.default_rng(11)
    _, chroma = _field(hue_deg=25.0, rotation_per_line_deg=90.0,
                       offset_deg=-60.0)
    luma = rng.normal(0.0, 200.0, (HEIGHT, WIDTH))
    result = cfl.process(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET,
                         held_out=True)
    assert result["share"]["removed"] < 0.05


def test_the_subtraction_is_confined_to_the_active_area():
    """MASKED IN, as directed: sync, blanking and the burst are untouched."""
    luma, chroma = _field(hue_deg=25.0, rotation_per_line_deg=90.0,
                          offset_deg=-60.0)
    result = cfl.process(luma, chroma, SAMPLE_RATE, ACTIVE, QUIET)
    free = result["luma"]
    assert np.array_equal(free[:, :ACTIVE[0]], luma[:, :ACTIVE[0]])
    assert np.array_equal(free[:, ACTIVE[1]:], luma[:, ACTIVE[1]:])
    assert not np.array_equal(free[:, ACTIVE[0]:ACTIVE[1]],
                              luma[:, ACTIVE[0]:ACTIVE[1]])


def test_the_change_point_finds_a_planted_mid_line_break():
    """`split_point` locates where a line breaks, which is the part of the
    mid-line question the evidence does support."""
    values = np.ones(400, dtype=complex)
    values[250:] = np.exp(1j * np.pi)
    found = hs.split_point(values)
    assert found["sample"] == pytest.approx(250, abs=2)
    assert found["gain"] > 2.0
    difference = (found["after_deg"] - found["before_deg"] + 180.0) % 360.0
    assert abs(difference - 180.0) < 5.0 or abs(difference) < 5.0


def test_the_luma_replica_cannot_carry_the_colour_frame_either():
    """The new channel is put to the framing question and refused.

    `colour_framing` says what is missing is a measurable input. The colour
    lock is a new measurable input, so the question has to be asked - and the
    colour-under advances 10500 whole cycles a field on 525/60, so it has no
    per-field phase to carry.
    """
    from vhsdecode.models import colour_framing

    advance = colour_framing.colour_under_field_advance("NTSC")
    assert advance["cycles_per_field"] == pytest.approx(10500.0, abs=1e-6)
    assert advance["fractional_advance"] == pytest.approx(0.0, abs=1e-9)
    assert advance["can_carry_a_field_sequence"] is False
    # And the module now records the candidate as closed rather than open.
    assert "colour_lock" in colour_framing.settle_the_question()["closed"]
