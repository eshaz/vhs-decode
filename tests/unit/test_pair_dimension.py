"""A dimension is the Wiener transfer between two co-registered channels.

Ethan: "The 'dimensions' are the Weiner transform of each complex pair
(i.e. luma frequency vs. luma amplitude)."
"""

import numpy as np
import pytest

from vhsdecode.models import burst_instrument as bi
from vhsdecode.models import composite_channel as cc
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


# --------------------------------------------------------------------------
# The AMPLITUDE dimension of the burst pair
#
# The burst is specified in all three dimensions, and until `burst_amplitude`
# existed this module measured two of them. The transfer is the frequency
# dimension - it says how the channel is shaped and carries whatever common
# gain the two sides differ by without ever naming it against the standard -
# and on the colour-under path the whole of that gain is the record-side
# doubler of SMPTE 32M clause 3.9.2.1.3.
# --------------------------------------------------------------------------

RATE = 40e6
LENGTH = 4096


def _specified_burst(**kwargs):
    return pd.spec_burst(RATE, LENGTH, **kwargs)


def test_the_record_side_doubler_is_the_whole_of_the_transfer_gain():
    """A UNITY channel carrying nothing but the specified doubler reads as a
    factor of two in the transfer, because the synthetic side is the
    composite specification and the composite has not been doubled."""
    doubler = 10.0 ** (cc.BURST_DOUBLER_DB / 20.0)
    got = pd.burst_dimension(_specified_burst() * doubler, RATE)
    live = got["coherence"] > 0.9
    assert np.abs(got["transfer"][live]).mean() == pytest.approx(doubler,
                                                                 rel=1e-9)
    # and the amplitude dimension names it and takes it out
    assert got["amplitude"]["ratio"] == pytest.approx(1.0, rel=1e-12)
    assert got["amplitude"]["agrees"]
    assert got["amplitude"]["chroma_referred_ire"] == pytest.approx(
        pd.BURST_AMPLITUDE_IRE, rel=1e-12)


def test_leaving_the_doubler_in_is_a_clean_factor_of_two():
    doubler = 10.0 ** (cc.BURST_DOUBLER_DB / 20.0)
    measured = _specified_burst() * doubler
    left_in = pd.burst_amplitude(measured, RATE, doubler_applied=False)
    assert left_in["ratio"] == pytest.approx(doubler, rel=1e-12)
    # the standard allows the doubler +/- 0.5 dB and nothing more, so a
    # whole factor of two is outside its own tolerance and must be refused
    assert not left_in["agrees"]
    assert left_in["tolerance_linear"] == pytest.approx(
        10.0 ** (cc.BURST_DOUBLER_TOLERANCE_DB / 20.0), rel=1e-12)


def test_the_doubler_tolerance_is_the_error_bar_and_not_a_chosen_threshold():
    """The standard's own +/- 0.5 dB is what decides `agrees`, so a burst
    inside the tolerance is accepted and one just outside it is not."""
    tolerance_db = cc.BURST_DOUBLER_TOLERANCE_DB
    for offset_db, expected in ((0.0, True), (tolerance_db * 0.9, True),
                                (-tolerance_db * 0.9, True),
                                (tolerance_db * 1.1, False),
                                (-tolerance_db * 1.1, False)):
        measured = _specified_burst() * 10.0 ** (
            (cc.BURST_DOUBLER_DB + offset_db) / 20.0)
        assert pd.burst_amplitude(measured, RATE)["agrees"] is expected


def test_the_amplitude_is_one_complex_gain_so_the_phase_is_not_discarded():
    """`|g|` is the amplitude and `arg g` is the time, out of one operation,
    because they are one number. A magnitude-only fit would see nothing at
    all here - which is the reduction that made an earlier burst instrument
    rank two of four."""
    reference = _specified_burst()
    for degrees in (0.0, 30.0, 90.0, -45.0, 179.0):
        got = pd.burst_amplitude(reference * np.exp(1j * np.deg2rad(degrees)),
                                 RATE, doubler_applied=False)
        assert abs(got["gain"]) == pytest.approx(1.0, rel=1e-12)
        assert got["phase_deg"] == pytest.approx(degrees, abs=1e-9)
        assert got["seconds"] == pytest.approx(
            degrees * got["per_degree_s"], rel=1e-12)
    # a degree of subcarrier is 0.776 ns, the same clock burst_instrument uses
    assert pd.burst_amplitude(reference, RATE)["per_degree_s"] * 1e9 == \
        pytest.approx(0.776, abs=0.002)


def test_a_real_measurement_is_made_analytic_before_it_is_projected():
    """THE SECOND FACTOR OF TWO. A real cosine projected onto a complex
    exponential returns half its amplitude, which is the same size as the
    doubler and the opposite sign, so the two would cancel and leave a
    level that is right for the wrong reason."""
    reference = _specified_burst()
    real_only = np.real(reference)
    made_analytic = pd.burst_amplitude(real_only, RATE, doubler_applied=False)
    assert abs(made_analytic["gain"]) == pytest.approx(1.0, rel=2e-3)
    # what `astype` alone would have given, which is the error being avoided
    naive = abs(complex(np.vdot(reference, real_only.astype(np.complex128))
                        / np.vdot(reference, reference).real))
    assert naive == pytest.approx(0.5, rel=2e-3)
    assert pd.analytic(reference) is not reference
    assert np.allclose(pd.analytic(reference), reference)


def test_the_level_does_not_depend_on_how_much_blanking_the_window_holds():
    """A mean of magnitudes is a level per sample of window; a projection
    onto the specified burst is a level."""
    reference = _specified_burst()
    tight = pd.burst_amplitude(reference, RATE, doubler_applied=False)
    padded = pd.burst_amplitude(
        np.concatenate([reference, np.zeros(3 * LENGTH, dtype=complex)]),
        RATE, doubler_applied=False)
    assert abs(padded["gain"]) == pytest.approx(abs(tight["gain"]), rel=1e-12)
    # the mean-of-magnitudes reading of the same two windows differs by the
    # window ratio exactly, which is why the projection is used
    ratio = (np.abs(reference).mean()
             / np.abs(np.concatenate([reference,
                                      np.zeros(3 * LENGTH)])).mean())
    assert ratio == pytest.approx(4.0, rel=1e-12)


def test_the_two_instruments_in_the_tree_disagree_and_the_pair_is_the_point():
    """`burst_instrument.amplitude` reads `2 mean|envelope|` and this reads a
    projection. On a perfect burst they differ by the mean-to-peak of the
    specified raised-cosine gate, and off it by the window ratio - so they
    are two independent measurements that can disagree, which is what a pair
    is for."""
    doubler = 10.0 ** (cc.BURST_DOUBLER_DB / 20.0)
    measured = _specified_burst() * doubler
    projection = pd.burst_amplitude(measured, RATE)["chroma_referred_ire"]
    assert projection == pytest.approx(pd.BURST_AMPLITUDE_IRE, rel=1e-12)
    # over the whole window the mean falls as the window grows
    wide = bi.amplitude(measured)["chroma_referred"]
    half = bi.amplitude(measured[:LENGTH // 2])["chroma_referred"]
    assert half == pytest.approx(2.0 * wide, rel=1e-9)
    # on the burst's own support the disagreement is the gate's mean-to-peak
    support = np.flatnonzero(np.abs(_specified_burst()) > 0.0)
    on_support = bi.amplitude(measured[support[0]:support[-1] + 1])
    gate = np.abs(_specified_burst()[support[0]:support[-1] + 1])
    assert on_support["chroma_referred"] / projection == pytest.approx(
        gate.mean() / gate.max(), rel=1e-9)


def test_the_residual_says_when_a_believable_ratio_is_still_wrong():
    """One complex gain describes a burst that has only been scaled and
    rotated. A channel that has changed its SHAPE returns a ratio inside
    the tolerance and a residual that is no longer small, so the pair can
    disagree in a way a bare ratio cannot show."""
    reference = _specified_burst()
    clean = pd.burst_amplitude(reference, RATE, doubler_applied=False)
    assert clean["residual_fraction"] < 1e-20
    # the same level, delivered with the burst's shape changed
    shifted = np.roll(reference, 8)
    moved = pd.burst_amplitude(shifted, RATE, doubler_applied=False)
    assert moved["residual_fraction"] > 0.1


def test_the_amplitude_dimension_is_added_beside_the_transfer_not_folded_in():
    """Nothing above it moves: the transfer still measures the channel's
    shape against the composite specification, doubler and all."""
    got = pd.burst_dimension(_specified_burst() * 0.7 * np.exp(1j * 0.3), RATE)
    live = got["coherence"] > 0.9
    assert np.abs(got["transfer"][live]).mean() == pytest.approx(0.7, rel=1e-6)
    assert np.angle(got["transfer"][live].mean()) == pytest.approx(0.3,
                                                                   abs=1e-6)
    assert set(got) >= {"transfer", "coherence", "standard_error",
                        "reference", "absolute_phase", "amplitude"}
    # a 0.7 channel whose burst still carries the doubler is a chroma level
    # of 0.35 of the specification, which is the number the picture wants
    assert got["amplitude"]["ratio"] == pytest.approx(0.7 / (
        10.0 ** (cc.BURST_DOUBLER_DB / 20.0)), rel=1e-9)
    assert got["amplitude"]["phase_deg"] == pytest.approx(np.degrees(0.3),
                                                          abs=1e-6)


def test_the_colour_under_carrier_sets_the_clock_the_phase_is_read_against():
    """A degree is 0.776 ns at the subcarrier and 4.41 us at 40 f_H, so the
    carrier the burst was measured at has to travel with the phase."""
    under = pd.burst_amplitude(
        pd.spec_burst(RATE, LENGTH, carrier_hz=pd.NTSC_COLOUR_UNDER_HZ),
        RATE, doubler_applied=False, carrier_hz=pd.NTSC_COLOUR_UNDER_HZ)
    assert under["carrier_hz"] == pytest.approx(pd.NTSC_COLOUR_UNDER_HZ)
    assert under["per_degree_s"] == pytest.approx(
        1.0 / (360.0 * pd.NTSC_COLOUR_UNDER_HZ), rel=1e-12)
    assert abs(under["gain"]) == pytest.approx(1.0, rel=1e-9)
    # 40 f_H exactly, so one line is 40 cycles and a degree is that
    # much coarser a clock
    assert under["per_degree_s"] / (1.0 / (360.0 * pd.subcarrier_hz())) == \
        pytest.approx(pd.SUBCARRIER_RATIO / pd.COLOUR_UNDER_MULTIPLE,
                      rel=1e-12)
