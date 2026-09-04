"""A dimension is the Wiener transfer between two co-registered channels.

Ethan: "The 'dimensions' are the Weiner transform of each complex pair
(i.e. luma frequency vs. luma amplitude)."
"""

import numpy as np
import pytest

from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import pair_dimension as pd


def _one_pole(x, pole=0.7, gain=0.3):
    """A known transfer, applied without scipy so the test has no new
    dependency: y[n] = gain x[n] + pole y[n-1]."""
    y = np.zeros_like(x)
    previous = 0.0
    for index, value in enumerate(x):
        previous = gain * value + pole * previous
        y[index] = previous
    return y


def test_a_known_transfer_is_recovered_in_gain_and_phase():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(8192)
    y = _one_pole(x) + 0.05 * rng.standard_normal(8192)
    result = pd.pair_transfer(x, y, segments=16)
    frequencies = np.fft.rfftfreq(8192 // 16)
    truth = 0.3 / (1.0 - 0.7 * np.exp(-2j * np.pi * frequencies))
    band = result["coherence"] > 0.5
    assert band.sum() > 10
    gain = np.abs(result["transfer"][band]) / np.abs(truth[band])
    assert np.mean(np.abs(gain - 1.0)) < 0.10
    phase = np.angle(result["transfer"][band] / truth[band])
    assert np.degrees(np.mean(np.abs(phase))) < 5.0


def test_the_coherence_is_debiased():
    """With n averaged segments an entirely uncorrelated pair still returns
    1/n, and exactly 1 at n = 1. An undebiased coherence certifies the
    arithmetic rather than the physics."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal(8192)
    z = rng.standard_normal(8192)
    previous = None
    for segments in (4, 16, 64):
        result = pd.pair_transfer(x, z, segments=segments)
        assert result["raw_coherence"].mean() == pytest.approx(
            1.0 / segments, rel=0.3)
        assert result["coherence"].mean() < result["raw_coherence"].mean()
        if previous is not None:
            # more segments, less residual bias
            assert result["coherence"].mean() < previous
        previous = result["coherence"].mean()


def test_the_effective_length_falls_when_places_are_correlated():
    """sphere_floor rests on E|G_ij|^2 = 1/L, which assumes independent
    places. Weighted places are not independent and the count that belongs
    in the floor is the participation ratio of the weights."""
    flat = np.ones(256)
    assert pd.effective_length(flat) == pytest.approx(256.0)
    concentrated = np.zeros(256)
    concentrated[:8] = 1.0
    assert pd.effective_length(concentrated) == pytest.approx(8.0)


def test_the_dimensions_feed_the_ellipse_directly():
    """The ellipse is fitted over the TRANSFERS, not over their
    differences. Differencing two dimensions cancels exactly the shared
    mechanism the ellipse exists to find, and three channels give only
    three differences which sum to zero, so the ensemble would arrive
    rank-deficient by one."""
    rng = np.random.default_rng(13)
    n = 8192
    driver = rng.standard_normal(n)
    channels = {
        "amplitude": driver + 0.3 * rng.standard_normal(n),
        "frequency": _one_pole(driver) + 0.05 * rng.standard_normal(n),
        "time": _one_pole(driver, pole=0.2) + 0.05 * rng.standard_normal(n),
    }
    built = pd.dimensions(channels, segments=16)
    assert len(built) == 3
    for value in built.values():
        assert np.iscomplexobj(value["frequency"])
        assert 0.0 <= value["coherence"].min() <= value["coherence"].max() <= 1.0
    fit = ie.ellipsoid(built, "frequency", real_parameters=True)
    assert fit["rank"] >= 2
    assert fit["information"] == pytest.approx(3.0)

    # the three differences sum to zero, which is why they are not used
    names = list(built)
    a, b, c = (built[name]["frequency"] for name in names)
    assert np.allclose((a - b) + (b - c) + (c - a), 0.0)


def test_the_coherent_pair_is_weighted_not_filtered():
    """gamma(f) Z(f), never W(f) Z(f): W carries units of y per x, so
    multiplying the whole complex pair by it is dimensionally incoherent."""
    rng = np.random.default_rng(14)
    x = rng.standard_normal(8192)
    y = _one_pole(x) + 0.05 * rng.standard_normal(8192)
    kept = pd.coherent_pair(x, y, segments=16)
    assert np.iscomplexobj(kept)
    assert kept.size == 8192 // 16 // 2 + 1
    # an incoherent partner is suppressed relative to a coherent one
    z = rng.standard_normal(8192)
    incoherent = pd.coherent_pair(x, z, segments=16)
    assert np.linalg.norm(incoherent) < np.linalg.norm(kept)


def test_the_specified_sync_pulse_matches_the_standard():
    """SMPTE 170M table 2: 4.7 us wide at half amplitude, 140 ns between the
    10 and 90 per cent points, 40 IRE below blanking. This is the SYNTHETIC
    side of the final dimension and it is specified, not fitted."""
    rate, length = 50e6, 4096
    ideal = pd.spec_sync(rate, length)
    assert ideal.min() == pytest.approx(pd.SYNC_DEPTH_IRE)
    below = np.flatnonzero(ideal <= ideal.min() / 2)
    width = (below[-1] - below[0] + 1) / rate
    assert width == pytest.approx(pd.SYNC_WIDTH_S, abs=2.0 / rate)
    leading = ideal[:length // 2]
    ten = np.flatnonzero(leading <= 0.1 * ideal.min())[0]
    ninety = np.flatnonzero(leading <= 0.9 * ideal.min())[0]
    # the standard states a 10-90 time; a raised cosine's 10-90 is 0.5904 of
    # its transition, so the transition is widened by that reciprocal
    assert (ninety - ten) / rate == pytest.approx(pd.SYNC_RISE_S,
                                                  abs=2.0 / rate)


def test_the_sync_dimension_recovers_a_channel_applied_to_the_pulse():
    """The final dimension needs no new machinery: the specified pulse is
    the input, the measured pulse is the output, and their Wiener transfer
    is a dimension like any other."""
    rate, length = 40e6, 4096
    ideal = pd.spec_sync(rate, length)
    measured = np.zeros_like(ideal)
    previous = 0.0
    for index, value in enumerate(ideal):
        previous = 0.25 * value + 0.75 * previous
        measured[index] = previous
    rng = np.random.default_rng(77)
    measured = measured + 0.02 * np.abs(ideal).max() * rng.standard_normal(
        length)
    result = pd.sync_dimension(measured, rate, segments=16)
    frequencies = np.fft.rfftfreq(length // 16, 1.0 / rate)
    truth = 0.25 / (1.0 - 0.75 * np.exp(-2j * np.pi * frequencies / rate))
    band = result["coherence"] > 0.5
    assert band.sum() > 10
    gain = np.abs(result["transfer"][band]) / np.abs(truth[band])
    assert np.mean(np.abs(gain - 1.0)) < 0.15
    phase = np.angle(result["transfer"][band] / truth[band])
    assert np.degrees(np.mean(np.abs(phase))) < 8.0
    assert result["ideal"].shape == measured.shape


def test_the_burst_is_specified_in_all_three_dimensions():
    """Ethan: *"the color burst can be modeled using the spec. That satisfies
    all three dimensions."* Frequency is an exact ratio, amplitude and phase
    are stated, and the position and duration are whole subcarrier cycles."""
    assert pd.subcarrier_hz() == pytest.approx(3579545.4545, abs=1e-3)
    assert pd.subcarrier_hz() == pytest.approx(
        pd.SUBCARRIER_RATIO * pd.NTSC_LINE_RATE_HZ)
    burst = pd.spec_burst(40e6, 4096)
    assert np.iscomplexobj(burst), "the phase is the point; it must be kept"
    assert np.abs(burst).max() == pytest.approx(
        pd.BURST_AMPLITUDE_IRE / 2.0, rel=1e-3)


def test_the_burst_dimension_recovers_an_absolute_gain_and_phase():
    """What the sync dimension cannot do: the sync pulse carries no carrier,
    so its transfer's phase is relative to an arbitrary datum."""
    burst = pd.spec_burst(40e6, 4096)
    got = pd.burst_dimension(burst * 0.7 * np.exp(1j * 0.3), 40e6)
    live = got["coherence"] > 0.9
    assert live.sum() > 100
    assert np.abs(got["transfer"][live]).mean() == pytest.approx(0.7, rel=1e-6)
    assert np.angle(got["transfer"][live].mean()) == pytest.approx(0.3,
                                                                   abs=1e-6)
    assert got["absolute_phase"]


def test_the_recorded_burst_is_at_the_colour_under_carrier_not_the_subcarrier():
    """What the medium holds is the burst heterodyned to 629 kHz, so this
    dimension measures the colour-under channel first."""
    assert pd.NTSC_COLOUR_UNDER_HZ < pd.subcarrier_hz() / 5.0
    under = pd.spec_burst(40e6, 4096, carrier_hz=pd.NTSC_COLOUR_UNDER_HZ)
    got = pd.burst_dimension(under * 0.5, 40e6,
                             carrier_hz=pd.NTSC_COLOUR_UNDER_HZ)
    live = got["coherence"] > 0.9
    assert np.abs(got["transfer"][live]).mean() == pytest.approx(0.5, rel=1e-6)


def test_a_complex_pair_keeps_its_negative_frequencies():
    """Ethan: *"The transfer of all of these are in the complex domain."*
    `rfft` folds -f onto +f, which is exact for a real signal and destroys a
    carrier's sideband asymmetry."""
    rate, n = 40e6, 2048
    t = np.arange(n) / rate
    x = np.exp(2j * np.pi * 1e6 * t)
    got = pd.pair_transfer(x, x * 0.5, segments=4)
    assert got["transfer"].size == n // 4
    # the real path is unchanged
    real = pd.pair_transfer(np.cos(2 * np.pi * 1e6 * t),
                            np.cos(2 * np.pi * 1e6 * t) * 0.5, segments=4)
    assert real["transfer"].size == n // 4 // 2 + 1
