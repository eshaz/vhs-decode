"""The expected specification signal over the whole reserved interval.

Ethan, 2026-09-07: "I think it's just 3d analytical component of the entire
RF with the expected 3d anaytical spec signal for each stage ... Sync and eq
pulses are one model, color (with all it's mappings) is a model that is
connected to the time part of the sync model. Both are then run to generate
the y and the c files." And: "Excellent, the picture stage is what I am
describing again, keep it there."
"""

import numpy as np
import pytest

from vhsdecode.models import spec_signal as ss
from vhsdecode.models import sync_geometry

RATE = 4.0 * 315e6 / 88.0            # the 4 fsc output grid
LINES, WIDTH = 263, 910              # this arc's NTSC decodes
ACTIVE = (134, 894)


def _field(first_field=True):
    return ss.expected_field(LINES, WIDTH, RATE, "NTSC",
                             first_field=first_field, active=ACTIVE)


def test_the_expected_field_carries_the_specified_pulses_at_their_widths():
    """Every number comes from the timing table, so the waveform must read
    back the widths that table states."""
    out = _field()
    wave = out["waveform"]
    per_sample_us = 1e6 / RATE
    half = (ss.SYNC_TIP_IRE + ss.BLANKING_IRE) / 2.0

    # THE WIDTHS ARE READ AT HALF AMPLITUDE, which is where the format
    # states them; every one comes back within a sample of the table.
    picture = wave[30]
    width_us = float((picture < half).sum()) * per_sample_us
    assert width_us == pytest.approx(sync_geometry.LINE_SYNC_US["525"],
                                     abs=1.5 * per_sample_us)

    # the first vertical-sync row carries two equalising pulses
    geometry = out["geometry"]
    equalising = float(geometry["sequences"][0]["pulse_us"])
    assert float((wave[0] < half).sum()) * per_sample_us == pytest.approx(
        2 * equalising, abs=1.5 * per_sample_us)

    # and a field-sync row carries the broad pulses, which are far wider
    broad = float(geometry["sequences"][1]["pulse_us"])
    assert float((wave[4] < half).sum()) * per_sample_us == pytest.approx(
        2 * broad, abs=1.5 * per_sample_us)

    # three of each sequence, in the order the table states
    lows = [float((wave[r] < half).sum()) * per_sample_us for r in range(9)]
    assert all(v == pytest.approx(2 * equalising, abs=1.5 * per_sample_us)
               for v in lows[:3] + lows[6:])
    assert all(v == pytest.approx(2 * broad, abs=1.5 * per_sample_us)
               for v in lows[3:6])


def test_the_levels_are_the_specified_pair_and_nothing_else():
    wave = _field()["waveform"]
    assert wave.max() == pytest.approx(ss.BLANKING_IRE)
    assert wave.min() == pytest.approx(ss.SYNC_TIP_IRE)


def test_the_mask_keeps_the_vertical_sync_whole_and_drops_the_test_lines():
    """The rule is `sync_geometry.frame_mask`'s, and the geometry it comes
    to on this arc's decodes is recorded: nine rows kept whole, thirteen
    dropped, and about a ninth of the field kept."""
    out = _field()
    assert out["rows"] == {"vsync": 9, "vbi": 13, "picture": 241}
    mask = out["mask"]
    assert mask[:9].all()                     # the vertical sync, whole
    assert not mask[9:22].any()               # the test lines, dropped
    assert 0.10 < out["kept_fraction"] < 0.13
    detail = out["mask_detail"]
    assert detail["sync_pulse_end_column"] == 67
    assert detail["front_porch_samples_kept"] == 13


def test_the_mask_reaches_far_below_the_pulse_that_the_edge_alone_can():
    """THE REASON THE INTERVAL IS IN THE MODEL. The 1/T law: a 4.7 us pulse
    resolves 213 kHz; the interval, kept whole, resolves 1.75 kHz."""
    out = _field()
    reach = out["resolution_hz"]
    pulse_only = sync_geometry.resolution_hz(sync_geometry.LINE_SYNC_US["525"])
    assert reach["longest_run_us"] == pytest.approx(572.0, abs=1.0)
    assert reach["lowest_hz"] == pytest.approx(1748.0, rel=0.01)
    assert pulse_only / reach["lowest_hz"] > 100.0


def test_the_half_line_does_not_reach_the_intervals_shape():
    """MEASURED, not assumed. Every sequence of the interval puts its
    pulses at HALF-line spacing, so a row of it is periodic in half a line
    and the two fields' intervals are the same shape on a per-line grid.
    What distinguishes the fields is where the interval begins against the
    line, which the decoder's own time base settles before we see it."""
    first, second = _field(True)["waveform"], _field(False)["waveform"]
    assert np.allclose(first, second)
    # WITHIN a sequence the train repeats every half line, which is what
    # makes the two fields coincide; ACROSS the boundary between the
    # equalising and the broad sequences it does not, because the two
    # pulses differ in width, so the claim is made where it holds
    block = first[:9].reshape(-1)
    half = WIDTH // 2
    inside = slice(WIDTH, 2 * WIDTH)          # the second equalising row
    assert np.allclose(block[inside], np.roll(block, half)[inside], atol=1e-9)
    inside = slice(4 * WIDTH, 5 * WIDTH)      # a field-sync row
    assert np.allclose(block[inside], np.roll(block, half)[inside], atol=1e-9)


def test_a_field_equal_to_its_specification_reads_zero_on_every_dimension():
    out = _field()
    reading = ss.analytic_component(out["waveform"], out["waveform"],
                                    out["mask"], RATE)
    assert reading["amplitude"]["departure_ire"] == pytest.approx(0.0, abs=1e-9)
    assert reading["amplitude"]["gain"] == pytest.approx(1.0, abs=1e-9)
    assert abs(reading["time"]["delay_s"]) < 1e-12
    assert float(np.sqrt(np.mean(np.abs(reading["response"]) ** 2))) < 1e-9


def test_a_planted_depth_deficit_reads_on_the_amplitude_and_not_the_rest():
    out = _field()
    expected = out["waveform"]
    # A GAIN, which is what a depth deficit is: the specified shape scaled
    # about blanking, so the spacing shortens by two IRE and nothing else
    # moves. (Adding a step on the tip alone would change the shape as
    # well, and the reading would rightly show it on the other axes.)
    measured = expected * (38.0 / 40.0)
    reading = ss.analytic_component(measured, expected, out["mask"], RATE)
    assert reading["amplitude"]["departure_ire"] == pytest.approx(-2.0, abs=0.05)
    assert reading["amplitude"]["gain"] == pytest.approx(0.95, abs=0.002)
    assert abs(reading["time"]["delay_s"]) < 1e-12
    # a gain is a real constant on the frequency axis and no phase at all
    assert float(np.std(reading["response"].real)) < 1e-9
    assert float(np.sqrt(np.mean(reading["response"].imag ** 2))) < 1e-9


def test_a_planted_delay_reads_on_the_time_and_not_the_amplitude():
    out = _field()
    expected = out["waveform"]
    shift = 3
    # THE CHANNEL ACTS ON THE SIGNAL, NOT ON A ROW: a delay displaces the
    # whole raster, so it is planted on the field read out in order.
    measured = np.roll(expected.reshape(-1), shift).reshape(expected.shape)
    reading = ss.analytic_component(measured, expected, out["mask"], RATE,
                                    band_hz=(2e3, 2e6))
    assert reading["run_us"] == pytest.approx(572.0, abs=1.0)
    assert reading["time"]["delay_s"] == pytest.approx(shift / RATE, rel=0.1)
    assert reading["amplitude"]["departure_ire"] == pytest.approx(0.0, abs=0.05)


def test_a_planted_response_reads_on_the_frequency_axis():
    """A one-pole applied to the whole field is read back as its own log
    magnitude on the band the mask reaches."""
    from scipy import signal as sps

    out = _field()
    expected = out["waveform"]
    corner = 300e3
    b, a = sps.butter(1, corner / (RATE / 2.0))
    # planted on the field as the channel sees it: one signal, read out in
    # order, filtered, and only then looked at through the mask
    measured = sps.lfilter(b, a, expected.reshape(-1)).reshape(expected.shape)
    reading = ss.analytic_component(measured, expected, out["mask"], RATE,
                                    band_hz=(50e3, 1.5e6))
    f = reading["frequency_hz"]
    magnitude = reading["response"].real
    # the pole's own log magnitude, referred to the lowest carried bin
    truth = -0.5 * np.log1p((f / corner) ** 2)
    read = magnitude - magnitude[0] + truth[0]
    # READ ON THE LONG RUN this is 0.0148 nepers rms; read through the
    # whole mask, whose two hundred and forty-one picture rows put an edge
    # on either side of every pulse, it was 0.835 - fifty-six times worse
    assert float(np.sqrt(np.mean((read - truth) ** 2))) < 0.05


def test_the_edge_agreement_reports_a_shared_band_or_says_it_cannot():
    f = np.linspace(2e3, 4e6, 400)
    shape = -0.3 * (f / 4e6) + 1j * 0.0
    same = ss.edge_agreement(shape, shape[100:], f, f[100:])
    assert same["coherence"] == pytest.approx(1.0, abs=1e-6)
    apart = ss.edge_agreement(shape, np.conj(shape[100:]) * -1.0,
                              f, f[100:])
    assert apart["coherence"] <= 1.0
    too_little = ss.edge_agreement(shape[:2], shape[:2], f[:2], f[:2])
    assert np.isnan(too_little["coherence"])


def test_the_long_run_is_the_interval_and_the_short_ones_are_refused():
    """The measurement that settles where a ratio of spectra may be taken."""
    out = _field()
    runs = ss.long_runs(out["mask"], RATE)
    assert runs["runs"] == 1
    assert runs["all_runs"] > 200
    assert float(runs["lengths"][0]) / RATE * 1e6 == pytest.approx(572.0, abs=1.0)


def test_the_two_probes_join_at_the_crossover_the_format_fixes():
    """Ethan: 'Use the eq pulses to get the longer value, and the hsync
    pulses to refine the higher frequency details.'"""
    from scipy import signal as sps

    out = _field()
    expected = out["waveform"]
    corner = 300e3
    b, a = sps.butter(1, corner / (RATE / 2.0))
    measured = sps.lfilter(b, a, expected.reshape(-1)).reshape(expected.shape)
    high_f = np.linspace(1e5, 4e6, 500)
    high_H = 1.0 / (1.0 + 1j * high_f / corner)
    reading = ss.analytic_component(measured, expected, out["mask"], RATE,
                                    band_hz=(2e3, 3e6),
                                    edge={"frequency_hz": high_f, "H": high_H})
    joined = reading["joined"]
    assert joined["crossover_hz"] == pytest.approx(212765.0, rel=0.01)
    assert joined["from_long_probe"] > 0 and joined["from_short_probe"] > 0
    assert abs(joined["level_offset_nepers"]) < 0.2
