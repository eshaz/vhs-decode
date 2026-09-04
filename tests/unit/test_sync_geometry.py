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
