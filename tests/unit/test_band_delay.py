"""The luma and chroma bands as one clock, read twice.

Ethan: 'The difference in time between the luma and chroma bands are the
measurement we can use to observe the delay on each video head. This
applies to playback and recording.'

The two bands were written by one head in one pass, so everything that
moves the signal in time as a whole cancels in their difference and only a
delay that varies with frequency survives.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import band_delay as bd
from vhsdecode.models import colour_under, depth_split, vhs_specification

RATE = 50e6
LINE_S = 1.001 / (525.0 * 30.0)


def _synthetic(lines=40, burst_offset_us=0.0, rate=RATE, with_chroma=True,
               drop_every=None):
    """One radio-frequency capture: an FM luma carrier plus a colour-under
    burst, both at their specified frequencies."""
    per_line = int(round(LINE_S * rate))
    total = per_line * lines
    time = np.arange(total) / rate
    ire = np.zeros(total)
    sync = int(round(4.7e-6 * rate))
    for line in range(lines):
        start = line * per_line
        ire[start:start + sync] = -40.0
        # a little picture, well above the sync tip
        ire[start + int(9e-6 * rate):start + per_line] = 50.0
    frequency = np.array([vhs_specification.carrier_hz_for_ire(v) for v in ire])
    phase = 2.0 * np.pi * np.cumsum(frequency) / rate
    signal = np.cos(phase)
    if with_chroma:
        carrier = colour_under.carrier_hz("NTSC")
        window = bd.specified_interval("NTSC")
        begin = (4.7e-6 + (window["start_us"] + burst_offset_us) * 1e-6)
        length = window["length_us"] * 1e-6
        gate = np.zeros(total)
        for line in range(lines):
            start = line * per_line
            a = start + int(round(begin * rate))
            b = a + int(round(length * rate))
            if b < total:
                gate[a:b] = 1.0
        # A DELAY MOVES THE CARRIER WITH THE ENVELOPE. Shifting the gate
        # while leaving the tone's absolute phase alone is not a delay: it
        # cuts the tone at a different point of its cycle, which distorts
        # the filtered envelope and made a planted 0.20 microsecond shift
        # read as 0.34. The colour-under is 40 times the line rate exactly,
        # so a real burst carries its phase with it.
        delay = burst_offset_us * 1e-6
        signal = signal + 0.5 * gate * np.cos(
            2.0 * np.pi * carrier * (time - delay))
    return signal


def test_the_two_bands_and_the_threshold_come_from_the_standard():
    out = bd.bands("NTSC")
    assert math.isclose(out["chroma_hz"], colour_under.carrier_hz("NTSC"))
    assert math.isclose(out["luma_sync_tip_hz"], 3.4e6, rel_tol=1e-12)
    assert math.isclose(out["luma_peak_white_hz"], 4.4e6, rel_tol=1e-12)
    # the threshold is the midpoint IN IRE, which is not the midpoint in hertz
    assert out["threshold_ire"] == -20.0
    midpoint_in_hertz = 0.5 * (out["luma_sync_tip_hz"] + out["luma_blanking_hz"])
    assert math.isclose(out["threshold_hz"], midpoint_in_hertz, rel_tol=1e-12)
    assert out["ratio"] > 6.0


def test_the_burst_window_is_measured_from_the_sync_pulses_trailing_edge():
    out = bd.specified_interval("NTSC")
    assert math.isclose(out["start_us"], 5.3 - 4.7, abs_tol=1e-12)
    assert 2.5 < out["length_us"] < 2.6
    assert math.isclose(out["centre_us"],
                        out["start_us"] + 0.5 * out["length_us"], rel_tol=1e-12)


def test_the_demodulator_returns_hertz_on_a_planted_carrier():
    """The units are a trap: dividing by two pi reads 3.9 MHz as 0.62."""
    n = 8192
    tone = np.cos(2.0 * np.pi * 3.9e6 * np.arange(n) / RATE)
    got = bd.luma_frequency(tone, RATE)
    middle = got[n // 4:3 * n // 4]
    assert math.isclose(float(np.median(middle)), 3.9e6, rel_tol=2e-3)


def test_the_line_locator_finds_one_rise_a_line_and_ignores_the_picture():
    """The threshold sits below blanking, where no picture content reaches."""
    signal = _synthetic(lines=30)
    rises = bd.sync_rises(bd.luma_frequency(signal, RATE), RATE)
    assert 24 <= len(rises) <= 30
    spacing = np.diff(rises) / RATE
    assert np.allclose(spacing, LINE_S, rtol=2e-3)


def test_the_chroma_envelope_is_complex():
    signal = _synthetic(lines=12)
    envelope = bd.chroma_envelope(signal, RATE)
    assert np.iscomplexobj(envelope)
    assert np.abs(envelope.imag).max() > 0.0


def test_a_planted_burst_shift_is_recovered_on_the_interval():
    """The measurement's own control: move the burst, read the move."""
    readings = {}
    for offset in (0.0, 0.2):
        signal = _synthetic(lines=48, burst_offset_us=offset)
        frequency = bd.luma_frequency(signal, RATE)
        rises = bd.sync_rises(frequency, RATE)
        found = bd.burst_readings(bd.chroma_envelope(signal, RATE), rises, RATE)
        readings[offset] = bd.interval(found, RATE)
    moved = readings[0.2]["median_us"] - readings[0.0]["median_us"]
    assert math.isclose(moved, 0.2, abs_tol=0.03)
    assert abs(readings[0.0]["departure_ns"]) < 120.0


def test_without_a_burst_the_reading_does_not_lock():
    """The y-only captures are the same control on real tape."""
    signal = _synthetic(lines=48, with_chroma=False)
    rises = bd.sync_rises(bd.luma_frequency(signal, RATE), RATE)
    found = bd.burst_readings(bd.chroma_envelope(signal, RATE), rises, RATE)
    quiet = bd.interval(found, RATE)
    loud = bd.interval(
        bd.burst_readings(bd.chroma_envelope(_synthetic(lines=48), RATE),
                          bd.sync_rises(
                              bd.luma_frequency(_synthetic(lines=48), RATE),
                              RATE), RATE), RATE)
    assert quiet["lines"] < loud["lines"] or quiet["sd_us"] > 3.0 * loud["sd_us"]


def test_the_head_labels_come_from_the_vertical_intervals_own_gap():
    """Rises spaced by a gap once a revolution label the two fields."""
    per_field = 262
    rises = []
    position = 0.0
    for revolution in range(6):
        for line in range(2 * per_field):
            rises.append(position)
            position += LINE_S * RATE
        position += 0.6 * LINE_S * RATE          # the interval's own gap
    out = bd.head_labels(np.asarray(rises), RATE)
    labels = out["labels"]
    assert labels is not None
    assert out["rises_per_revolution"] == 2 * per_field
    assert out["rises_per_field"] == per_field
    assert set(np.unique(labels)) <= {0, 1}
    # the label must alternate once a field, not once a line
    changes = np.flatnonzero(np.diff(labels) != 0)
    assert np.allclose(np.diff(changes), per_field)


def test_head_labelling_refuses_a_span_with_no_marker():
    out = bd.head_labels(np.arange(40) * LINE_S * RATE, RATE)
    assert out["labels"] is None
    with pytest.raises(ValueError):
        bd.head_labels(np.arange(4), RATE)


def test_the_tap_difference_cancels_the_record_side_and_carries_its_error():
    record = {"median_us": 2.1386, "standard_error_us": 0.0017}
    playback = {"median_us": 2.0580, "standard_error_us": 0.0022}
    out = bd.tap_difference(record, playback)
    assert math.isclose(out["delay_ns"], -80.6, abs_tol=0.2)
    assert math.isclose(out["error_ns"],
                        math.hypot(1.7, 2.2), abs_tol=0.05)
    assert out["sigma"] > 20.0


def test_the_wallace_law_agrees_with_the_module_that_already_had_it():
    """Two independently written derivations of one minimum-phase relation."""
    here = bd.wallace_comparison(1.0)["ns_per_micron"] * 0.05
    there = depth_split.wallace_delay_difference_s(0.05e-6) * 1e9
    assert math.isclose(here, there, rel_tol=1e-6)


def test_a_shortened_interval_cannot_be_a_separation_loss():
    """The sign is the finding, and the argument is the measured quantity.

    A positive separation delays the chroma more than the luma and so
    LENGTHENS the sync-to-burst interval. An interval that shortens
    requires a negative separation, which does not exist.
    """
    shorter = bd.wallace_comparison(-80.6)
    longer = bd.wallace_comparison(+80.6)
    assert not shorter["separation_is_physical"]
    assert longer["separation_is_physical"]
    assert math.isclose(shorter["equivalent_separation_um"],
                        -longer["equivalent_separation_um"], rel_tol=1e-12)
    assert math.isclose(shorter["luma_minus_chroma_ns"], 80.6, rel_tol=1e-12)


def _measurement(first_us, first_error, second_us, second_error):
    pooled = {"median_us": first_us, "standard_error_us": first_error}
    pooled["per_head"] = {
        0: {"median_us": first_us, "standard_error_us": first_error,
            "lines": 3600},
        1: {"median_us": second_us, "standard_error_us": second_error,
            "lines": 3600},
    }
    return pooled


def test_the_head_difference_carries_no_deck_common_term():
    out = bd.head_difference(_measurement(2.06090, 0.00097, 2.05840, 0.00064))
    assert out["usable"]
    assert math.isclose(out["difference_ns"], 2.50, abs_tol=0.01)
    assert math.isclose(out["error_ns"], math.hypot(0.97, 0.64), abs_tol=0.01)


def test_the_head_difference_declines_when_the_heads_were_not_labelled():
    out = bd.head_difference({"per_head": {}})
    assert not out["usable"]
    assert math.isnan(out["difference_ns"])


def test_the_two_taps_place_the_per_head_delay_outside_the_record_drive():
    """The question the sync-edge instrument states and cannot answer.

    A per-head difference present at the record tap would live in the
    drive electronics; one absent there does not. The record tap's own
    null is what makes that argument, so the test is on the null.
    """
    record = _measurement(2.14049, 0.00044, 2.13968, 0.00045)
    playback = _measurement(2.06090, 0.00097, 2.05840, 0.00064)
    out = bd.relate_to_head_delays(record, playback)
    assert out["record_side_excluded"]
    assert out["record_tap_head_sigma"] < 3.0
    assert math.isclose(out["tap_difference_ns"], -79.59, abs_tol=0.1)
    # three of the sync edge's four delays sit inside this bound, not four
    assert len(out["sync_edge_inside_this_bound"]) == 3


def test_a_record_tap_that_is_not_null_refuses_the_conclusion():
    """If the drive itself differed per head, nothing could be concluded."""
    record = _measurement(2.14049, 0.00010, 2.13000, 0.00010)
    playback = _measurement(2.06090, 0.00097, 2.05840, 0.00064)
    out = bd.relate_to_head_delays(record, playback)
    assert not out["record_side_excluded"]
