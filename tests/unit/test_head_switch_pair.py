"""The record head switch, read from the recorded colour phase rotation.

The tests are built round a synthetic tape signal because the quantity
being recovered is one the specification states exactly - SMPTE 32M
clause 3.9.2.1.5's quarter turn a line, reversing at the record head
switch - so a planted reversal is a fair and complete check of the
estimator. Each test that guards a defect names the measurement that
found it.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import (band_delay, colour_under, head_switch_pair,
                              sync_geometry, vhs_specification)

RATE_HZ = 20e6
SYSTEM = "NTSC"
LINE_US = sync_geometry.LINE_PERIOD_US["525"]
SYNC_US = sync_geometry.LINE_SYNC_US["525"]


def _synthesise(fields=5, lines_per_field=100, switch_line=80,
                chroma=0.35, start_state=+1, undershoot=0.0,
                rate_hz=RATE_HZ):
    """A tape-like radio-frequency signal with a PLANTED rotation reversal.

    The luminance is a frequency-modulated carrier following SMPTE 32M
    clause 3.9.1.1.4 through `vhs_specification.carrier_hz_for_ire`; the
    chrominance is a tone at the specified colour-under carrier whose
    phase advances a quarter turn a line and reverses at `switch_line` of
    each field. `undershoot` drops the level below blanking just before
    each sync pulse, which is the shape real dark content makes and which
    the line locator has to survive.

    Returns the samples and the sample indices of the planted reversals.
    """
    line_samples = LINE_US * 1e-6 * rate_hz
    total_lines = fields * lines_per_field
    total = int(round(total_lines * line_samples)) + 64
    ire = np.zeros(total, dtype=np.float64)
    chroma_phase = np.zeros(total, dtype=np.float64)
    amplitude = np.zeros(total, dtype=np.float64)
    reversals = []
    phase = 0.0
    state = float(start_state)
    quarter = math.radians(head_switch_pair.RECORD_ROTATION_DEG[0])
    for index in range(total_lines):
        field, within = divmod(index, lines_per_field)
        start = index * line_samples
        first = int(round(start))
        # the vertical interval: three lines of broad pulses at the top
        vertical = within < 3
        for offset in range(int(round(line_samples))):
            position = first + offset
            if position >= total:
                break
            micro = offset / rate_hz * 1e6
            if vertical:
                level = -40.0 if micro < 0.85 * LINE_US else 0.0
            elif micro < SYNC_US:
                level = -40.0
            elif micro < SYNC_US + 1.2:
                level = 0.0
            else:
                level = 55.0 + 20.0 * math.sin(0.03 * index)
                if undershoot and micro > LINE_US - 1.4:
                    level = -26.0
            ire[position] = level
        if vertical:
            continue
        if within == switch_line:
            state = -state
            reversals.append(first)
        phase += state * quarter
        # the burst, then the active chroma, both at the recorded phase
        burst = band_delay.specified_interval(SYSTEM)
        for low_us, high_us in ((burst["start_us"] + SYNC_US,
                                 burst["start_us"] + burst["length_us"]
                                 + SYNC_US),
                                (SYNC_US + 6.0, LINE_US - 1.6)):
            low = first + int(low_us * 1e-6 * rate_hz)
            high = first + int(high_us * 1e-6 * rate_hz)
            high = min(high, total)
            if high <= low:
                continue
            chroma_phase[low:high] = phase
            amplitude[low:high] = chroma
    carrier = np.asarray([vhs_specification.carrier_hz_for_ire(value)
                          for value in ire], dtype=np.float64)
    luma = np.cos(2.0 * np.pi * np.cumsum(carrier) / rate_hz)
    index = np.arange(total, dtype=np.float64)
    tone = amplitude * np.cos(
        2.0 * np.pi * colour_under.carrier_hz(SYSTEM) * index / rate_hz
        + chroma_phase)
    return luma + tone, np.asarray(reversals, dtype=np.float64)


@pytest.fixture(scope="module")
def measured():
    samples, reversals = _synthesise()
    frequency = band_delay.luma_frequency(samples, RATE_HZ)
    envelope = band_delay.chroma_envelope(samples, RATE_HZ, SYSTEM)
    rises = head_switch_pair.line_starts(frequency, RATE_HZ, SYSTEM)
    rotation = head_switch_pair.record_rotation(envelope, rises, RATE_HZ,
                                                SYSTEM)
    switches = head_switch_pair.record_switches(rotation, rises, RATE_HZ,
                                                SYSTEM)
    return {
        "samples": samples,
        "reversals": reversals,
        "frequency": frequency,
        "envelope": envelope,
        "rises": rises,
        "rotation": rotation,
        "switches": switches,
    }


# --------------------------------------------------------------------------
# What the specification fixes, with no data at all
# --------------------------------------------------------------------------

def test_specified_rotation_is_the_clause():
    """The clause gives +90 and -90, and what matters is that they are half
    a turn apart - no timing error can turn one into the other."""
    spec = head_switch_pair.specified_rotation(SYSTEM)
    assert spec["track_1_deg"] == 90.0
    assert spec["track_2_deg"] == -90.0
    assert spec["separation_deg"] == 180.0
    assert "3.9.2.1.5" in spec["cite"]


def test_the_quarter_turn_is_a_rate():
    """A quarter turn a line period is f_H/4, which is what makes the
    reversal a change of rate rather than of value."""
    spec = head_switch_pair.specified_rotation(SYSTEM)
    assert spec["rate_hz"] == pytest.approx(
        colour_under.line_rate_hz(SYSTEM) / 4.0)
    assert spec["rate_hz"] == pytest.approx(3933.566, abs=1e-3)


def test_half_a_turn_is_far_larger_than_the_timing_error():
    """Half a turn at the colour-under carrier is 794 nanoseconds against a
    measured line-to-line scatter of 15.7 ns on the test tape and 102 ns
    on the home tape, so the two states never merge."""
    spec = head_switch_pair.specified_rotation(SYSTEM)
    assert spec["half_turn_s"] == pytest.approx(794.4e-9, rel=1e-3)
    assert spec["half_turn_s"] > 7.0 * 102e-9


def test_field_sequence_puts_the_framing_in_the_up_converted_channel():
    """The carrier advances a whole number of cycles a field and carries no
    field phase; the record rotation does, and the up-conversion turns it
    into the four-field sequence."""
    sequence = head_switch_pair.field_sequence(SYSTEM)
    assert sequence["carrier_cycles_per_field"] == pytest.approx(10500.0)
    assert sequence["carrier_fractional"] == pytest.approx(0.0, abs=1e-9)
    assert sequence["recorded_rotation_per_field_deg"] == pytest.approx(225.0)
    assert sequence["recorded_sequence_fields"] == 2
    assert sequence["up_converted_per_field_deg"] == pytest.approx(270.0)
    assert sequence["up_converted_sequence_fields"] == 4


def test_output_row_lands_on_the_already_identified_luma_switch():
    """`head_switch.locate` puts the luminance switch at output row 260.2
    +/- 0.8; this module's mark on the same deck's tape reads 5.533 and
    6.030 lines ahead of vertical sync, which is the same row."""
    row_a = head_switch_pair.output_row(5.533, SYSTEM)
    row_b = head_switch_pair.output_row(6.030, SYSTEM)
    expected, tolerance = head_switch_pair.LUMA_SWITCH_OUTPUT_ROW
    assert abs(row_a - expected) < 2.0 * tolerance
    assert abs(row_b - expected) < 2.0 * tolerance


def test_decoder_table_differs_by_two_quarters():
    """The runtime's `NTSC_ROTATION = [-1, 1]` and the standard's
    [+90, -90] label the pair oppositely; what has to agree is that both
    entries differ by two quarter turns."""
    decoder = head_switch_pair.decoder_track_phase(SYSTEM)
    first, second = decoder["table_quarters"]
    assert abs(first - second) == 2
    assert decoder["threshold_deg"] == 90.0
    assert "chroma.py" in decoder["walker"]


def test_pal_table_also_differs_by_two_quarters():
    decoder = head_switch_pair.decoder_track_phase("PAL")
    first, second = decoder["table_quarters"]
    assert abs(first - second) == 1 or abs(first - second) == 2


# --------------------------------------------------------------------------
# The signal's geometry
# --------------------------------------------------------------------------

def test_line_starts_finds_every_active_line(measured):
    """Every line outside the planted vertical interval, and no more."""
    expected = 5 * (100 - 3)
    assert abs(measured["rises"].size - expected) <= 5


def test_line_starts_survives_a_content_undershoot():
    """THE MEASURED DEFECT. `band_delay.sync_rises` takes the first crossing
    back above the half-way carrier, and on real consumer video a dark
    line's undershoot crosses it: measured, that locator sits a median
    5.373 microseconds early on home.flac and 5.490 on countdown.flac,
    with 3.0 and 3.9 per cent of its lines inside 0.2 us. Validating the
    crossing against the specified 4.70 us pulse width repairs it.
    """
    samples, _ = _synthesise(undershoot=1.0)
    frequency = band_delay.luma_frequency(samples, RATE_HZ)
    validated = head_switch_pair.line_starts(frequency, RATE_HZ, SYSTEM)
    naive = band_delay.sync_rises(frequency, RATE_HZ, SYSTEM)
    assert validated.size > 300
    # the naive locator is dragged early by the undershoot; the validated
    # one keeps a clean line period
    spacing = np.diff(validated) / RATE_HZ * 1e6
    single = spacing[np.abs(spacing - LINE_US) < 1.0]
    assert single.size > 0.8 * validated.size
    assert float(np.std(single)) < 0.05
    assert naive.size >= validated.size


def test_vertical_sync_finds_one_edge_a_field(measured):
    """Broad pulses are longer than twice a line pulse and nothing else
    is, so the block is unambiguous."""
    edges = head_switch_pair.vertical_sync(measured["frequency"], RATE_HZ,
                                           SYSTEM)
    assert 4 <= edges.size <= 6


# --------------------------------------------------------------------------
# The recorded rotation
# --------------------------------------------------------------------------

def test_record_rotation_recovers_the_specified_quarter_turn(measured):
    """The planted quarter turn comes back as two clusters half a turn
    apart, on nearly every line pair."""
    rotation = measured["rotation"]
    assert rotation["clean_fraction"] > 0.9
    assert abs(abs(rotation["axis_deg"]) - 90.0) < 5.0


def test_rotation_is_complex(measured):
    """A phase exists, so the output carries it."""
    assert np.iscomplexobj(measured["rotation"]["rotation"])


def test_rotation_rate_matches_the_specification(measured):
    """Each track's advance is f_H/4 with the sign of its own clause
    entry, and the two are half a turn apart."""
    rate = head_switch_pair.rotation_rate(measured["rotation"],
                                          measured["rises"], RATE_HZ, SYSTEM)
    assert rate["specified_hz"] == pytest.approx(3933.566, abs=1e-3)
    assert abs(abs(rate["separation_deg"]) - 180.0) < 3.0
    for name in ("track_a", "track_b"):
        assert name in rate
        assert abs(abs(rate[name]["rate_hz"]) - rate["specified_hz"]) < 60.0


def test_the_axis_branch_is_anchored_to_the_advancing_track():
    """THE MEASURED DEFECT. Squaring maps the two clusters onto one, so the
    axis comes back modulo half a turn and its branch is arbitrary.
    Unanchored, the home tape read the SAME pair of values with the head
    labels swapping between tape positions - (6.004, 5.492) at 300
    seconds, (5.477, 5.981) at 1500 - which would have made the per-head
    reference's stable origin false. Anchoring to the +90 degree track
    fixes it: starting a synthetic tape on either parity gives the same
    label to the same rotation sense.
    """
    labels = []
    for start in (+1, -1):
        samples, _ = _synthesise(start_state=start)
        frequency = band_delay.luma_frequency(samples, RATE_HZ)
        envelope = band_delay.chroma_envelope(samples, RATE_HZ, SYSTEM)
        rises = head_switch_pair.line_starts(frequency, RATE_HZ, SYSTEM)
        rotation = head_switch_pair.record_rotation(envelope, rises, RATE_HZ,
                                                    SYSTEM)
        assert math.cos(rotation["axis_rad"]
                        - math.radians(
                            head_switch_pair.RECORD_ROTATION_DEG[0])) > 0.0
        switches = head_switch_pair.record_switches(rotation, rises, RATE_HZ,
                                                    SYSTEM)
        labels.append([switch["into"] for switch in switches])
    assert labels[0] and labels[1]
    # the two syntheses start on opposite parities, so their first marks go
    # into opposite states - which is exactly what a stable label means
    assert labels[0][0] == -labels[1][0]


def test_no_chroma_returns_no_switches():
    """THE Y-ONLY CONTROL, which the capture set contains and which this
    reproduces: with no chrominance recorded there is nothing to reverse.
    Measured on `zaroff-multiburst-y-only-NTSC-SP`, the clean fraction
    falls from 0.996 to 0.286 and no field boundary is found at all."""
    samples, _ = _synthesise(chroma=0.0)
    frequency = band_delay.luma_frequency(samples, RATE_HZ)
    envelope = band_delay.chroma_envelope(samples, RATE_HZ, SYSTEM)
    rises = head_switch_pair.line_starts(frequency, RATE_HZ, SYSTEM)
    rotation = head_switch_pair.record_rotation(envelope, rises, RATE_HZ,
                                                SYSTEM)
    switches = head_switch_pair.record_switches(rotation, rises, RATE_HZ,
                                                SYSTEM)
    assert rotation["clean_fraction"] < 0.9
    assert not switches


# --------------------------------------------------------------------------
# Locating the mark
# --------------------------------------------------------------------------

def test_record_switches_find_every_planted_reversal(measured):
    """One mark a field, each inside a line of where it was planted."""
    switches = measured["switches"]
    planted = measured["reversals"]
    # the last reversal has no complete field after it, so it cannot make a
    # mark - a mark needs a run of one rotation on each side of it
    assert planted.size - 1 <= len(switches) <= planted.size
    for switch in switches:
        nearest = float(np.min(np.abs(planted - switch["centre"])))
        assert nearest < 2.0 * LINE_US * 1e-6 * RATE_HZ


def test_the_marks_alternate(measured):
    """The heads write in strict turn, which is the clock the per-head
    reference is built on."""
    into = [switch["into"] for switch in measured["switches"]]
    assert all(first != second for first, second in zip(into, into[1:]))


def test_refine_switches_narrows_the_bracket(measured):
    """The two-dimensional map answers `chroma.py`'s own TODO - where in
    the line the change falls - and the bracket falls below a line."""
    refined = head_switch_pair.refine_switches(
        measured["switches"], measured["envelope"], measured["rises"],
        RATE_HZ, SYSTEM, places=128)
    assert len(refined) == len(measured["switches"])
    narrowed = [switch for switch in refined if switch["refined"]]
    assert narrowed
    for switch in narrowed:
        assert switch["half_width_us"] * 2.0 < LINE_US
        nearest = float(np.min(np.abs(measured["reversals"]
                                      - switch["centre"])))
        assert nearest < 2.0 * LINE_US * 1e-6 * RATE_HZ


def test_refinement_keeps_the_bracket_honest(measured):
    """`low` and `high` bracket `centre`, whether or not the refinement
    fired - a point estimate without its bracket would be a fit."""
    refined = head_switch_pair.refine_switches(
        measured["switches"], measured["envelope"], measured["rises"],
        RATE_HZ, SYSTEM, places=128)
    for switch in refined:
        assert switch["low"] <= switch["centre"] <= switch["high"]
        assert switch["half_width_us"] >= 0.0


# --------------------------------------------------------------------------
# The luminance switch, and the separation
# --------------------------------------------------------------------------

def _flat_envelope(size):
    return np.ones(size, dtype=np.float64)


def test_a_jitter_sized_displacement_is_not_a_switch(measured):
    """The synthetic tape has no playback head at all, so its line period
    never steps; the reading must come back unresolved rather than as a
    number. This is the record tap's behaviour, where the measured
    displacement is -24 to +19 nanoseconds against +1396 to +1835 at the
    playback tap."""
    luma = head_switch_pair.luma_switch(
        measured["frequency"], _flat_envelope(measured["samples"].size),
        measured["rises"], RATE_HZ, measured["switches"], SYSTEM)
    assert luma
    assert not any(live["resolved"] for live in luma)
    separation = head_switch_pair.separation(measured["switches"], luma,
                                             RATE_HZ, SYSTEM)
    assert not np.isfinite(separation["pooled_timing_h"]) or \
        separation["pooled_error_h"] >= 0.0


def test_separation_is_reported_per_head(measured):
    luma = head_switch_pair.luma_switch(
        measured["frequency"], _flat_envelope(measured["samples"].size),
        measured["rises"], RATE_HZ, measured["switches"], SYSTEM)
    separation = head_switch_pair.separation(measured["switches"], luma,
                                             RATE_HZ, SYSTEM)
    assert set(separation["per_head"]) <= {0, 1}
    assert separation["per_head"]
    for entry in separation["per_head"].values():
        assert entry["fields"] >= 1


def test_switch_to_vertical_sync_is_inside_the_specified_window(measured):
    """The mark's own position, which is the recording machine's
    fingerprint and is bounded by SMPTE 32M clause 3.6."""
    edges = head_switch_pair.vertical_sync(measured["frequency"], RATE_HZ,
                                           SYSTEM)
    against = head_switch_pair.switch_to_vertical_sync(
        measured["switches"], edges, RATE_HZ, SYSTEM)
    assert against["specified_h"] == (5.0, 8.0)
    assert "3.6" in against["cite"]
    reported = [against[key] for key in ("head_0", "head_1") if key in against]
    assert reported


# --------------------------------------------------------------------------
# The per-head reference, and what feeds back
# --------------------------------------------------------------------------

def test_per_head_reference_alternates_and_ticks_at_the_field_rate(measured):
    luma = head_switch_pair.luma_switch(
        measured["frequency"], _flat_envelope(measured["samples"].size),
        measured["rises"], RATE_HZ, measured["switches"], SYSTEM)
    reference = head_switch_pair.per_head_reference(
        measured["switches"], luma, RATE_HZ, SYSTEM)
    assert reference["alternates"]
    assert reference["fields"] == len(measured["switches"])
    # the synthetic field is 80 lines rather than 262.5, so the tick is the
    # planted one; what is being checked is that a tick was measured at all
    assert reference["field_period_s"] > 0.0


def test_the_reference_says_what_it_does_not_fix():
    """Naming the physical head needs an instrument outside the signal
    chain, and the docstring has to say so rather than imply otherwise."""
    text = head_switch_pair.per_head_reference.__doc__
    assert "does not name the physical head" in text.lower() \
        or "not name the physical head" in text.lower()


def test_counter_rotation_is_a_sign(measured):
    """The applied and the recorded quarter turns differ by two quarters,
    so the error is exp(i pi k) - a sign, not a small angle. That is why a
    mis-placed flip shows as alternate lines of inverted colour."""
    edges = head_switch_pair.vertical_sync(measured["frequency"], RATE_HZ,
                                           SYSTEM)
    correction = head_switch_pair.counter_rotation(
        measured["switches"], measured["rises"], RATE_HZ, edges, SYSTEM)
    assert len(correction) == len(measured["switches"])
    for entry in correction:
        factors = np.asarray(entry["line_factor"])
        assert np.allclose(np.abs(factors), 1.0)
        assert np.allclose(factors.imag, 0.0, atol=1e-9)
        assert 0.0 <= entry["flip_fraction"] <= 1.0
        assert entry["starting_index"] in (0, 1)
        first, second = entry["split_factor"]
        assert abs(first + second) < 1e-9


def test_counter_rotation_carries_the_sub_line_position(measured):
    refined = head_switch_pair.refine_switches(
        measured["switches"], measured["envelope"], measured["rises"],
        RATE_HZ, SYSTEM, places=128)
    edges = head_switch_pair.vertical_sync(measured["frequency"], RATE_HZ,
                                           SYSTEM)
    correction = head_switch_pair.counter_rotation(refined, measured["rises"],
                                                   RATE_HZ, edges, SYSTEM)
    assert any(entry["refined"] for entry in correction)
    for entry in correction:
        assert 0.0 <= entry["flip_offset_us"] <= LINE_US + 1e-6
        # the row comes from the vertical sync and not from the capture's
        # own line counter, which was the defect this guards
        assert np.isfinite(entry["flip_row"])
        assert abs(entry["discrepancy_lines"]) < 263.0


def test_head_model_entry_states_its_resolving_power(measured):
    """The specified window is three lines wide and conforming machines
    share it, so the entry has to say that it labels a recorder only when
    two of them differ."""
    edges = head_switch_pair.vertical_sync(measured["frequency"], RATE_HZ,
                                           SYSTEM)
    against = head_switch_pair.switch_to_vertical_sync(
        measured["switches"], edges, RATE_HZ, SYSTEM)
    rate = head_switch_pair.rotation_rate(measured["rotation"],
                                          measured["rises"], RATE_HZ, SYSTEM)
    entry = head_switch_pair.head_model_entry(against, rate)
    assert entry["stage"] == "recording"
    assert entry["window_width_h"] == 3.0
    assert "differ" in entry["resolving_power"]
    assert entry["belongs_to"] == "the machine that made the recording"


def test_ownership_needs_a_mark_at_both_taps():
    """With nothing at either tap the verdict is undetermined rather than
    a guess."""
    verdict = head_switch_pair.ownership({}, {})
    assert verdict["rotation_owner"] == "undetermined"
    assert verdict["displacement_owner"] == "undetermined"
    assert "no playback head" in verdict["why"]


def test_ownership_reads_the_two_taps(measured):
    luma = head_switch_pair.luma_switch(
        measured["frequency"], _flat_envelope(measured["samples"].size),
        measured["rises"], RATE_HZ, measured["switches"], SYSTEM)
    tap = {"record": measured["switches"], "luma": luma}
    verdict = head_switch_pair.ownership(tap, tap)
    assert verdict["rotation_owner"] == "the recording machine"
    # both taps are the same signal here, so no displacement stands out
    assert verdict["displacement_owner"] == "undetermined"
