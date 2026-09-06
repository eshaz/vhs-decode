"""The vertical interval's own geometry as the low-frequency probe.

Ethan: *"Also use the equalization pulses in the same way but for longer
frequencies and possibly ghosting."*
"""

import numpy as np
import pytest

from vhsdecode.models import sync_geometry as sg
from vhsdecode.models import interference as inf

BAND = np.linspace(1e3, 7e6, 1024)


def test_a_probe_resolves_no_finer_than_its_own_reciprocal_duration():
    """The arithmetic the whole argument rests on, checked directly."""
    for duration in (4.7, 63.5556, 190.67, 572.0):
        assert sg.resolution_hz(duration) == pytest.approx(
            1e6 / duration, rel=1e-12)


def test_the_interval_reaches_two_decades_below_the_sync_pulse():
    reach = sg.probes("NTSC")
    pulse = reach["the line-sync pulse"]["resolution_hz"]
    interval = reach["the whole vertical interval"]["resolution_hz"]
    assert pulse == pytest.approx(212.8e3, rel=1e-3)
    assert interval == pytest.approx(1.748e3, rel=1e-3)
    assert pulse / interval == pytest.approx(121.7, rel=1e-3)


def test_the_defect_that_was_measured_lies_below_the_pulses_floor():
    """THE CONTROL. A 1.22 us relaxation corners at 1/(2 pi tau) = 130 kHz;
    the sync pulse cannot resolve below 213 kHz. The probe and the defect
    were mismatched, which is why measuring it on the pulse found a tail and
    could not correct it."""
    control = sg.probe_control("NTSC", time_constant_us=1.22)
    assert control["defect_corner_hz"] == pytest.approx(130.5e3, rel=1e-2)
    assert not control["sync_pulse_resolves_the_corner"]
    assert control["interval_resolves_the_corner"]
    assert control["passes"]


def test_a_single_sync_pulse_cannot_see_a_typical_broadcast_ghost():
    """The stronger result: an echo at tau is a ripple of period 1/tau, so
    the reachable delay is 1/(2 df). The pulse's whole window ends at
    2.35 us and a broadcast ghost runs to tens of microseconds."""
    pulse = sg.ghost_reach(4.7, "NTSC")
    interval = sg.ghost_reach(572.0, "NTSC")
    assert pulse["longest_us"] == pytest.approx(2.35, rel=1e-2)
    assert interval["longest_us"] == pytest.approx(286.0, rel=1e-2)
    # both are bounded below by the SPECIFIED pulse edge, not by the window
    assert pulse["shortest_us"] == interval["shortest_us"] == 0.14


def test_the_train_is_built_from_the_timing_table_and_nothing_else():
    train = sg.pulse_train("NTSC", 40e6)
    geometry = train["geometry"]
    assert geometry["total_lines"] == 9.0
    assert train["duration_us"] == pytest.approx(572.0, rel=1e-3)
    # three sequences of six pulses each, at half-line spacing
    assert len(train["marks"]) == 18
    wave = train["waveform"]
    assert wave.min() == pytest.approx(-40.0, abs=1e-6)
    assert wave.max() == pytest.approx(0.0, abs=1e-6)


def test_the_head_switch_clears_the_interval_on_spec_and_not_as_measured():
    """Ethan: *"I do have outliers caused by head switching distortion that
    may be in the vertical sync area."* The specification says they should
    not, and BOTH measured readings say they do - while disagreeing with
    each other, which is why the module holds both."""
    on_spec = sg.head_switch_window("NTSC")
    assert not on_spec["overlaps_the_interval"]
    low, high = on_spec["specified_range_h"]
    for reading in (3.5, -1.0):
        measured = sg.head_switch_window("NTSC", onset_h_before_vsync=reading)
        assert measured["overlaps_the_interval"], reading
        assert measured["from_measurement"]
        assert reading < low, "both readings sit outside the specified range"
    # the onset is SIGNED, so a negative value places the switch after the
    # V-sync leading edge, which is where the second reading puts it
    after = sg.head_switch_window("NTSC", onset_h_before_vsync=-1.0)
    assert after["start_us"] > 0.0


def test_masking_the_switch_costs_little_on_either_reading():
    """THE CONCLUSION SURVIVES THE DISAGREEMENT, which is what makes it safe
    to act on before the mapping question is settled."""
    for reading, removed, resolution in ((3.5, 0.056, 1851.0),
                                         (-1.0, 0.111, 1967.0)):
        masked = sg.interval_mask("NTSC", 40e6, onset_h_before_vsync=reading)
        assert masked["removed_fraction"] == pytest.approx(removed, abs=0.005)
        # the surviving span's resolution is reported, not the full window's
        assert masked["resolution_hz"] > masked["full_resolution_hz"]
        assert masked["resolution_hz"] == pytest.approx(resolution, rel=1e-2)
        # and against a sync pulse's 213 kHz either is nothing
        assert masked["resolution_hz"] < 0.01 * sg.resolution_hz(4.7)


def test_the_differential_recovers_a_planted_response_and_bounds_its_band():
    train = sg.pulse_train("NTSC", 4e6)
    ideal = train["waveform"]
    # a planted single-pole relaxation, applied in the frequency domain
    n = ideal.size
    freqs = np.fft.rfftfreq(n, 1.0 / 4e6)
    planted = 1.0 / (1.0 + 2j * np.pi * freqs * 1.22e-6)
    measured = np.fft.irfft(np.fft.rfft(ideal - ideal.mean()) * planted, n)
    got = sg.differential(measured, ideal - ideal.mean(), 4e6)
    live = got["valid"] & (got["frequency_hz"] > got["lowest_honest_hz"]) \
        & (got["frequency_hz"] < 1e6)
    assert live.sum() > 10
    error = np.abs(got["H"][live] - planted[live])
    assert float(error.max()) < 1e-6
    # and nothing is claimed below the window's own resolution
    assert got["lowest_honest_hz"] == pytest.approx(1748.0, rel=1e-2)


def test_a_complex_input_takes_the_whole_spectrum():
    """Ethan: *"The transfer of all of these are in the complex domain."*
    `rfft` folds the negative frequencies onto the positive ones, which
    destroys a carrier's sideband asymmetry."""
    rate, n = 4e6, 512
    t = np.arange(n) / rate
    x = np.exp(2j * np.pi * 3e5 * t)
    got = sg.differential(x * 0.5, x, rate)
    assert np.iscomplexobj(got["H"])
    assert (got["frequency_hz"] < 0).any(), "negative frequencies must exist"


def test_the_entries_are_ordered_and_subtractable():
    chain = inf.full_chain()
    for name, value in sg.signatures(BAND).items():
        assert inf.subtractable(value), name
        assert inf.position_of(name, chain) == sg.CHAIN_POSITION, name
    # below the recorder's clip, which is where Ethan localises it
    from vhsdecode.models import clipping
    assert sg.CHAIN_POSITION < clipping.CHAIN_POSITION


def test_a_real_pole_cannot_carry_a_ring_and_the_fit_says_so():
    """Ethan: *"a complex pair is not interconnected ... showing as a ringing
    and noise profile on the sync pulse's response."*

    The sync edge and the back-porch tail are ONE channel - the porch is
    where the edge's step settles - so a ring visible on the edge must be
    present in the tail. A single real exponential has no frequency and
    cannot represent one."""
    t = np.linspace(0.0, 3.77, 55)
    rng = np.random.default_rng(3)

    pure = 6.5 - 2.4 * np.exp(-t / 1.3) + rng.normal(0.0, 0.21, t.size)
    got = sg.relaxation_or_ringing(pure, t)
    assert got["decided"] and not got["prefers_ringing"]

    for frequency, decay in ((0.6e6, 1.3), (1.5e6, 0.8)):
        ringing = (6.5 - 2.4 * np.exp(-t / decay)
                   * np.cos(2 * np.pi * frequency * t * 1e-6)
                   + rng.normal(0.0, 0.21, t.size))
        got = sg.relaxation_or_ringing(ringing, t)
        assert got["prefers_ringing"], frequency
        assert got["ringing"]["frequency_hz"] == pytest.approx(
            frequency, rel=0.10)


def test_the_ring_test_is_not_fooled_by_noise():
    """A ring has two more free parameters, so it fits better by
    construction. Measured over sixty draws of pure noise it buys a median
    +7.2 per cent and at most +15.9 - so a measurement must clear that, not
    merely be positive."""
    t = np.linspace(0.0, 3.56, 52)
    rng = np.random.default_rng(11)
    improvements = []
    for _ in range(24):
        got = sg.relaxation_or_ringing(rng.normal(0.0, 0.175, t.size), t)
        improvements.append(got["improvement"])
    assert float(np.median(improvements)) < 0.20
    assert max(improvements) < 0.35, (
        "chance must stay below the +37 per cent the real remainder gives")


def test_the_crossover_between_the_two_probes_is_derived_not_chosen():
    """Ethan: 'Use the eq pulses to get the longer value, and the hsync
    pulses to refine the higher frequency details.' A probe of duration T
    carries nothing below 1/T, so the sync pulse's own 4.7 microseconds is
    the frequency beneath which only the vertical interval can speak."""
    import numpy as np
    from vhsdecode.models import sync_geometry as sg

    edges = sg.crossover_hz()
    assert edges["short_resolution_hz"] == pytest.approx(1.0 / 4.7e-6)
    assert edges["long_resolution_hz"] == pytest.approx(1.0 / 572e-6)
    assert edges["crossover_hz"] == edges["short_resolution_hz"]
    assert edges["decades_below_crossover"] == pytest.approx(2.086, abs=0.01)
    # a narrower pulse would move the join, which is the point of deriving it
    assert sg.crossover_hz(short_probe_us=2.3)["crossover_hz"] > edges["crossover_hz"]


def test_two_probes_join_into_one_response_and_the_level_step_is_measured():
    import numpy as np
    from vhsdecode.models import sync_geometry as sg

    frequency = np.logspace(np.log10(500), np.log10(4e6), 2000)
    # a corner at 50 kHz, below anything the sync pulse can resolve
    truth = 1.0 / (1.0 + 1j * frequency / 50e3)
    low = frequency < 3e5
    high = frequency > 1.5e5
    planted_offset = 0.3
    out = sg.combined_response(frequency[low], truth[low],
                               frequency[high], truth[high] * np.exp(planted_offset))
    assert out["crossover_hz"] == pytest.approx(1.0 / 4.7e-6)
    assert out["from_long_probe"] > 0 and out["from_short_probe"] > 0
    # the gain difference between the two instruments is measured, not fitted
    assert -out["level_offset_nepers"] == pytest.approx(planted_offset, abs=1e-6)
    assert out["overlap_disagreement_nepers"] == pytest.approx(0.0, abs=1e-9)
    joined = np.interp(frequency, out["frequency_hz"], np.abs(out["H"]))
    assert np.abs(np.log(joined / np.abs(truth))).max() < 1e-6


def test_the_whole_sync_region_is_masked_from_the_decode_and_not_assumed():
    """Ethan: 'I need to measure the entire sync region, and mask out the
    active area from this measurement.' The bounds come from the decode's
    own JSON, and where they disagree with BT.1700 the disagreement is
    reported rather than resolved."""
    import numpy as np
    from vhsdecode.models import sync_geometry as sg

    out = sg.blanking_mask(field_width=910, active_start=134, active_end=894)
    assert out["sync_region_samples"] == 150
    assert out["active_samples"] == 760
    assert out["mask"].sum() + out["active"].sum() == 910
    # the region is the two ends of the line, not a middle span
    assert out["mask"][0] and out["mask"][-1] and not out["mask"][500]
    # BT.1700 wants 10.9 us and this decode gives 10.476: six samples short
    assert out["sync_region_us"] == pytest.approx(10.476, abs=0.01)
    assert out["shortfall_samples"] == 6
    assert out["specified_mask"].sum() == out["sync_region_samples"] + 6
    assert not out["within_specification"]
    # and it refuses to guess when it is given nothing to read
    with pytest.raises(ValueError, match="not assumed"):
        sg.blanking_mask()


def test_the_field_mask_keeps_only_timing_and_never_content():
    """Ethan: 'Mask out the entire VBI and the active area, only the vsync
    and sync pulse area should remain. Stop the mask just after the tail
    end of the active area in the front porch.'"""
    import numpy as np
    from vhsdecode.models import sync_geometry as sg

    out = sg.frame_mask(field_width=910, field_height=263,
                        active_start=134, active_end=894)
    assert out["vsync_rows"] == 9
    assert out["vbi_rows_dropped"] == 13
    assert out["picture_rows"] == 263 - 9 - 13
    # the vertical sync block is kept whole; the VBI's test lines are gone
    assert out["mask"][0].all() and out["mask"][8].all()
    assert not out["mask"][9].all() and not out["mask"][21].any()
    # a picture line keeps the sync pulse and the front porch's tail only
    line = out["line_keep"]
    assert line[:out["sync_pulse_end_column"]].all()
    assert not line[134:894].any()                     # no active picture
    assert not line[74:110].any()                      # no colour burst
    assert line[out["keep_from_column"]:].all()        # the front porch tail
    assert out["front_porch_samples_kept"] == 910 - out["keep_from_column"]
    assert out["kept_fraction"] < 0.2
    with pytest.raises(ValueError, match="not assumed"):
        sg.frame_mask()
