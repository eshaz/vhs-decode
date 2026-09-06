"""The noise budget: three predicted floors, summed, against a measured one.

Every term is arithmetic with a planted input, so each test plants the
input and checks the number; the particulate law is checked on a simulated
track, and the instrument that reads the floor is checked on a synthetic
capture whose noise density is known before it is measured.
"""

import numpy as np
import pytest

from vhsdecode.models import capture_profile as cp
from vhsdecode.models import noise_budget as nb
from vhsdecode.models import tape_model


BAND_HZ = 8.0e6                       # Carson's band for the format


@pytest.fixture(scope="module")
def figures():
    return nb.format_figures()


class TestThermal:
    def test_kt_at_room_temperature_is_the_docs_figure(self):
        """docs/ELLIPTICAL_COLLAPSE.md section 7.5: kTB at 293 K over 8 MHz
        is -104.9 dBm; per hertz that is -173.9 dBm."""
        term = nb.thermal_term(band_hz=BAND_HZ)
        assert term["kt_dbm_per_hz"] == pytest.approx(-173.9, abs=0.05)
        assert term["ktb_dbm"] == pytest.approx(-104.9, abs=0.05)

    def test_a_planted_source_gives_v_squared_over_4rktf(self):
        """1 mV rms into 100 ohm makes 2.5 nW available; over kT at 293 K
        with a unity noise factor that is 117.9 dB-Hz."""
        term = nb.thermal_term(carrier_rms_v=1e-3, source_resistance_ohm=100.0,
                               noise_figure_db=0.0, temperature_k=293.0,
                               band_hz=BAND_HZ)
        expected = (1e-3 ** 2 / 400.0) / (nb.BOLTZMANN_J_PER_K * 293.0)
        assert term["carrier_available_w"] == pytest.approx(2.5e-9)
        assert term["c_over_n0_db_hz"] == pytest.approx(10 * np.log10(expected), abs=1e-6)
        assert term["c_over_n0_db_hz"] == pytest.approx(117.9, abs=0.05)
        assert term["c_over_n_band_db"] == pytest.approx(117.9 - 69.03, abs=0.05)

    def test_the_noise_figure_costs_exactly_its_decibels(self):
        quiet = nb.thermal_term(noise_figure_db=0.0)["c_over_n0_db_hz"]
        noisy = nb.thermal_term(noise_figure_db=6.0)["c_over_n0_db_hz"]
        assert quiet - noisy == pytest.approx(6.0, abs=1e-9)

    def test_the_ranges_bracket_the_central_value(self):
        term = nb.thermal_term()
        low, high = term["c_over_n0_range_db_hz"]
        full_low, full_high = term["c_over_n0_full_range_db_hz"]
        assert low < term["c_over_n0_db_hz"] < high
        assert full_low <= low and high <= full_high
        # halving and doubling the head output is +/- 6 dB about the centre
        assert high - low == pytest.approx(12.04, abs=0.01)


class TestQuantisation:
    def test_the_arithmetic_density_is_the_word_lengths_own(self):
        term = nb.quantisation_term(carrier_power=1.0, bits=8, step=1.0,
                                    sample_rate_hz=50e6, band_hz=BAND_HZ)
        assert term["arithmetic_density_per_hz"] == pytest.approx((1.0 / 12.0) / 25e6)
        assert term["density_per_hz"] == term["arithmetic_density_per_hz"]

    def test_the_carrier_the_record_carries(self):
        """8 bits at 50 MSps against a 35.2-code carrier: 112.7 dB-Hz, which
        is 43.7 dB in Carson's band - the prediction in the docstring."""
        term = nb.quantisation_term(35.2 ** 2 / 2.0, 8, 1.0, 50e6, band_hz=BAND_HZ)
        assert term["c_over_n0_db_hz"] == pytest.approx(112.7, abs=0.05)
        assert term["c_over_n_band_db"] == pytest.approx(43.7, abs=0.05)

    def test_each_lost_effective_bit_costs_six_decibels(self):
        full = nb.quantisation_term(600.0, 8, 1.0, 50e6)["c_over_n0_db_hz"]
        seven = nb.quantisation_term(600.0, 8, 1.0, 50e6, effective=7.0)["c_over_n0_db_hz"]
        assert full - seven == pytest.approx(6.02, abs=0.01)

    def test_effective_bits_inverts_the_term(self):
        term = nb.quantisation_term(600.0, 8, 1.0, 50e6, effective=6.5)
        assert nb.effective_bits(term["density_per_hz"], 8, 50e6, 1.0) == pytest.approx(6.5)

    def test_the_container_does_not_change_the_ratio(self):
        """An 8-bit capture carried as int16 has a step of 256; the ratio
        against a carrier measured in the same units is the same."""
        codes = nb.quantisation_term(35.2 ** 2 / 2.0, 8, 1.0, 50e6)["c_over_n0_db_hz"]
        container = nb.quantisation_term((35.2 * 256) ** 2 / 2.0, 8, 256.0, 50e6)["c_over_n0_db_hz"]
        assert codes == pytest.approx(container, abs=1e-9)


class TestParticulate:
    def test_ethans_arithmetic_is_reproduced(self, figures):
        """58 um by lambda / 2 pi by one wavelength at 4.0 MHz and 5.80 m/s
        is 19.4 um^3; with the repository's particle constants 7763
        particles, 38.9 dB; Ethan's ~7000 -> 38.45 dB differs only in the
        assumed particle density."""
        term = nb.particulate_term(4.0e6, figures["writing_speed_m_s"],
                                   figures["track_width_m"], band_hz=BAND_HZ)
        assert term["wavelength_m"] == pytest.approx(1.45e-6)
        assert term["read_depth_m"] == pytest.approx(0.2308e-6, rel=1e-3)
        assert term["volume_m3"] == pytest.approx(19.41e-18, rel=1e-3)
        assert term["particles"] == pytest.approx(7763, rel=1e-3)
        assert term["ten_log_n_db"] == pytest.approx(38.9, abs=0.05)
        assert abs(term["ten_log_n_db"] - 10 * np.log10(7000)) < 0.5

    def test_the_count_band_is_half_the_carrier(self, figures):
        term = nb.particulate_term(4.0e6, figures["writing_speed_m_s"],
                                   figures["track_width_m"])
        assert term["count_band_hz"] == pytest.approx(2.0e6)
        assert term["c_over_n0_db_hz"] == pytest.approx(term["ten_log_n_db"] + 63.01, abs=0.01)

    def test_the_density_is_invariant_to_the_along_track_choice(self, figures):
        """Counting over a tenth of a wavelength gives a tenth of the
        particles, 10 dB less as a count, over ten times the band: the
        same density."""
        whole = nb.particulate_term(4.0e6, 5.80, 58e-6)
        tenth = nb.particulate_term(4.0e6, 5.80, 58e-6, along_track_m=1.45e-7)
        assert whole["c_over_n0_db_hz"] == pytest.approx(tenth["c_over_n0_db_hz"], abs=1e-9)
        assert whole["ten_log_n_db"] - tenth["ten_log_n_db"] == pytest.approx(10.0, abs=1e-9)
        assert tenth["count_band_hz"] == pytest.approx(10 * whole["count_band_hz"])

    def test_saturation_marks_cost_0_9_db(self):
        sine = nb.particulate_term(4.0e6, 5.80, 58e-6)["c_over_n0_db_hz"]
        square = nb.particulate_term(4.0e6, 5.80, 58e-6, saturation=True)["c_over_n0_db_hz"]
        assert sine - square == pytest.approx(10 * np.log10(np.pi ** 2 / 8), abs=1e-9)
        assert sine - square == pytest.approx(0.91, abs=0.01)

    def test_the_guides_recorded_depth_is_the_alternative(self):
        """JVC VTG82063 section 7.2.1: about 0.3 um. Against lambda / 2 pi
        at 4 MHz (0.231 um) that is 1.14 dB more particles."""
        bound = nb.particulate_term(4.0e6, 5.80, 58e-6)
        guide = nb.particulate_term(4.0e6, 5.80, 58e-6, read_depth_m=0.3e-6)
        assert guide["c_over_n0_db_hz"] - bound["c_over_n0_db_hz"] == pytest.approx(
            10 * np.log10(0.3e-6 / bound["read_depth_m"]), abs=1e-9)

    def test_the_repositorys_constants_are_the_ones_used(self):
        term = nb.particulate_term(4.0e6, 5.80, 58e-6)
        assert term["particle_volume_m3"] == tape_model.PARTICLE_VOLUME_M3
        assert term["packing"] == tape_model.PACKING_FRACTION

    @pytest.mark.parametrize("marks", ["sinusoidal", "square"])
    def test_the_law_on_a_simulated_track(self, marks):
        """Particles at Poisson positions along a track, each carrying a
        moment that follows the recorded pattern; the head sweeps past at
        v. The carrier over the noise density must be factor * n * v with
        factor 1/2 for sinusoidal marks and 4 / pi^2 for square ones."""
        rng = np.random.default_rng(11)
        wavelength = 1.45e-6
        per_wavelength = 200.0
        wavelengths = 4000
        bins_per_wavelength = 32
        speed = 5.80
        n_per_m = per_wavelength / wavelength
        length = wavelengths * wavelength
        bins = wavelengths * bins_per_wavelength
        delta = wavelength / bins_per_wavelength
        positions = rng.uniform(0.0, length, rng.poisson(n_per_m * length))
        phase = 2.0 * np.pi * positions / wavelength
        mark = np.sin(phase) if marks == "sinusoidal" else np.sign(np.sin(phase))
        track = np.bincount((positions / delta).astype(np.int64) % bins,
                            weights=mark, minlength=bins)
        rate = speed / delta
        power = np.abs(np.fft.rfft(track)) ** 2 * 2.0 / (rate * bins)
        carrier_bin = wavelengths
        carrier = power[carrier_bin] * rate / bins
        away = np.ones(power.size, dtype=bool)
        away[carrier_bin - 3:carrier_bin + 4] = False
        away[0] = False
        floor = np.median(power[away]) / nb.chi_square_median_factor(1)
        factor = (nb.PARTICLE_LAW_SINUSOIDAL if marks == "sinusoidal"
                  else nb.PARTICLE_LAW_SATURATION)
        predicted = factor * n_per_m * speed
        assert 10 * np.log10(carrier / floor) == pytest.approx(
            10 * np.log10(predicted), abs=0.2)


class TestTheSum:
    def test_three_equal_terms_sum_to_4_77_db_less(self):
        total = nb.sum_of_terms({"a": 100.0, "b": 100.0, "c": 100.0})
        assert total["c_over_n0_db_hz"] == pytest.approx(100.0 - 10 * np.log10(3), abs=1e-9)
        assert all(share == pytest.approx(1 / 3) for share in total["shares"].values())

    def test_a_dominant_term_is_the_sum(self):
        total = nb.sum_of_terms({"a": 100.0, "b": 130.0, "c": 140.0})
        assert total["c_over_n0_db_hz"] == pytest.approx(100.0, abs=0.01)
        assert total["shares"]["a"] > 0.99

    def test_the_shares_sum_to_one(self):
        total = nb.sum_of_terms({"a": 106.9, "b": 112.7, "c": 101.9})
        assert sum(total["shares"].values()) == pytest.approx(1.0)
        assert total["c_over_n0_db_hz"] == pytest.approx(100.4, abs=0.05)


class TestDefinitions:
    def test_carsons_band_is_69_db_below_the_density(self, figures):
        defined = nb.definitions(96.0, figures, 50e6, bin_hz=50e3)
        assert defined["carson_band_db"] == pytest.approx(96.0 - 69.03, abs=0.01)
        assert defined["deviation_band_db"] == pytest.approx(36.0, abs=0.01)
        assert defined["nyquist_band_db"] == pytest.approx(96.0 - 73.98, abs=0.01)
        assert defined["per_bin_db"] == pytest.approx(96.0 - 46.99, abs=0.01)

    def test_the_weighting_network_reproduces_bt1439(self):
        """ITU-R BT.1439-1 as docs/SPECIFICATION_INVENTORY.md section 3.10
        transcribes it: 7.4 dB for flat noise and 12.2 dB for triangular
        noise over 5 MHz, and 14.8 dB asymptotic loss."""
        factors = nb.weighting_noise_factors()
        assert factors["flat_db"] == pytest.approx(7.4, abs=0.1)
        assert factors["triangular_db"] == pytest.approx(12.2, abs=0.1)
        assert factors["asymptotic_loss_db"] == pytest.approx(14.8, abs=0.05)
        assert nb.weighting_power_response(0.0) == pytest.approx(1.0)

    def test_the_deemphasis_is_a_14_db_shelf(self, figures):
        response = nb.deemphasis_power_response(
            np.array([0.0, 1e8]), figures["deemphasis_tau_s"], figures["deemphasis_gain_db"])
        assert response[0] == pytest.approx(1.0)
        assert 10 * np.log10(response[1]) == pytest.approx(-figures["deemphasis_gain_db"], abs=0.01)

    def test_deemphasis_and_weighting_each_raise_the_video_figure(self, figures):
        raw = nb.video_signal_to_noise(96.0, figures, deemphasis=False)
        deemphasised = nb.video_signal_to_noise(96.0, figures)
        weighted = nb.video_signal_to_noise(96.0, figures, weighting=True)
        assert raw < deemphasised < weighted
        # triangular noise integrated to 3 MHz without shaping, by hand
        n0_over_c = 10 ** (-9.6)
        noise = np.sqrt(n0_over_c * figures["baseband_hz"] ** 3 / 3.0)
        assert raw == pytest.approx(20 * np.log10(figures["luma_span_hz"] / noise), abs=0.01)

    def test_labels_near_a_quoted_figure(self, figures):
        defined = nb.definitions(96.0, figures, 50e6)
        near = nb.labels_near(defined, 27.0, 1.0)
        assert near == ["carson_band_db"]


class TestTheGap:
    def _budget(self, figures, measured, uncertainty=0.1):
        return nb.budget(figures, 50e6, 35.2 ** 2 / 2.0, 8, 1.0,
                         measured_c_over_n0_db_hz=measured,
                         measured_uncertainty_db=uncertainty)

    def test_the_prediction_is_the_docstrings(self, figures):
        result = nb.budget(figures, 50e6, 35.2 ** 2 / 2.0, 8, 1.0)
        assert result["terms_db_hz"]["thermal"] == pytest.approx(106.9, abs=0.05)
        assert result["terms_db_hz"]["quantisation"] == pytest.approx(112.7, abs=0.05)
        assert result["terms_db_hz"]["particulate"] == pytest.approx(101.9, abs=0.05)
        assert result["predicted_c_over_n0_db_hz"] == pytest.approx(100.4, abs=0.05)
        assert result["predicted_carson_db"] == pytest.approx(31.4, abs=0.05)

    def test_closes_when_measured_equals_the_sum(self, figures):
        predicted = nb.budget(figures, 50e6, 35.2 ** 2 / 2.0, 8, 1.0)["predicted_c_over_n0_db_hz"]
        result = self._budget(figures, predicted)
        assert result["gap_db"] == pytest.approx(0.0)
        assert result["closes"]
        assert "closes" in result["verdict"]

    def test_a_gap_the_head_output_range_reaches(self, figures):
        """Halving the assumed head output lowers the SUM by 2.3 dB, not
        6, because the thermal term is under a quarter of the noise; a
        1.5 dB gap is inside that."""
        predicted = nb.budget(figures, 50e6, 35.2 ** 2 / 2.0, 8, 1.0)["predicted_c_over_n0_db_hz"]
        result = self._budget(figures, predicted - 1.5)
        assert not result["closes"]
        assert result["closes_within_thermal_range"]
        assert "head amplifier's input" in result["verdict"]

    def test_a_three_db_gap_is_half_unexplained(self, figures):
        predicted = nb.budget(figures, 50e6, 35.2 ** 2 / 2.0, 8, 1.0)["predicted_c_over_n0_db_hz"]
        result = self._budget(figures, predicted - 3.0103)
        assert result["gap_db"] == pytest.approx(3.0103, abs=1e-6)
        assert result["unexplained_share"] == pytest.approx(0.5, abs=1e-4)
        assert not result["closes"]
        # the head-output range alone falls short; every assumption at its
        # extreme reaches it
        assert not result["closes_within_thermal_range"]
        assert result["closes_within_full_range"]
        assert "source resistance" in result["verdict"]

    def test_a_gap_beyond_every_assumption_says_so(self, figures):
        predicted = nb.budget(figures, 50e6, 35.2 ** 2 / 2.0, 8, 1.0)["predicted_c_over_n0_db_hz"]
        result = self._budget(figures, predicted - 15.0)
        assert not result["closes_within_thermal_range"]
        assert not result["closes_within_full_range"]
        assert "no labelled assumption reaches it" in result["verdict"]

    def test_better_than_the_sum_points_at_the_dominant_term(self, figures):
        predicted = nb.budget(figures, 50e6, 35.2 ** 2 / 2.0, 8, 1.0)["predicted_c_over_n0_db_hz"]
        result = self._budget(figures, predicted + 3.0)
        assert result["gap_db"] == pytest.approx(-3.0)
        assert "BETTER" in result["verdict"]
        assert "particulate" in result["verdict"]


# --------------------------------------------------------------------------
# the instrument, on a synthetic capture whose floor is planted
# --------------------------------------------------------------------------

RATE = 50e6
AMPLITUDE_CODES = 35.0
SIGMA_CODES = 2.0
CONTAINER_STEP = 256.0          # 8-bit codes carried as int16, as soundfile reads them


def synthetic_capture(figures, fields=2, seed=1):
    """An FM capture with the format's vertical interval: six equalising,
    six broad and six equalising pulses per field, then picture lines with
    a sync tip, porches and a stepped grey level, white Gaussian noise of
    known density, quantised to 8 bits."""
    tip, blank, hz_ire = figures["tip_hz"], figures["blanking_hz"], figures["hz_ire"]
    line, sync, broad = figures["line_period_s"], figures["line_sync_s"], figures["field_sync_pulse_s"]
    equalising = sync / 2.0
    segments = []
    for _ in range(fields):
        for _ in range(6):
            segments += [(equalising, tip), (line / 2 - equalising, blank)]
        for _ in range(6):
            segments += [(broad, tip), (line / 2 - broad, blank)]
        for _ in range(6):
            segments += [(equalising, tip), (line / 2 - equalising, blank)]
        for k in range(253):
            level = blank + hz_ire * (20.0 + 50.0 * (k % 5) / 4.0)
            segments += [(sync, tip), (sync, blank), (line - 3 * sync, level), (sync, blank)]
    durations = np.array([d for d, _ in segments])
    frequencies = np.array([f for _, f in segments])
    counts = np.rint(durations * RATE).astype(np.int64)
    frequency = np.repeat(frequencies, counts)
    phase = 2.0 * np.pi * np.cumsum(frequency) / RATE
    rng = np.random.default_rng(seed)
    rf = AMPLITUDE_CODES * np.cos(phase) + rng.normal(0.0, SIGMA_CODES, frequency.size)
    return (np.clip(np.rint(rf), -128, 127) * CONTAINER_STEP).astype(np.int16)


def planted_c_over_n0_db_hz():
    """White noise of variance sigma^2 has a one-sided density of
    2 sigma^2 / f_s; the quantiser adds its own d^2 / 12 over Nyquist."""
    noise = 2.0 * SIGMA_CODES ** 2 / RATE + (1.0 / 12.0) / (RATE / 2.0)
    return 10 * np.log10((AMPLITUDE_CODES ** 2 / 2.0) / noise)


@pytest.fixture(scope="module")
def synthetic(figures):
    return synthetic_capture(figures, fields=3)


@pytest.fixture(scope="module")
def measured(figures, synthetic):
    return nb.measure_floor(synthetic, RATE, figures)


class TestTheInstrument:
    def test_the_rate_is_measured_from_the_line_period(self, figures, synthetic):
        """Given a wrong nominal rate (40 MSps for a 50 MSps capture) the
        line period recovers the true one, and the tip lands on spec."""
        rate = nb.sample_rate_from_line_period(synthetic, 40e6, figures)
        assert rate["sample_rate_hz"] == pytest.approx(RATE, rel=2e-3)
        assert rate["tip_within_tolerance"]

    def test_the_word_length_is_read_from_the_lattice(self, measured):
        assert measured["bits"] == 8.0
        assert measured["step"] == CONTAINER_STEP

    def test_the_broad_pulses_are_found(self, measured):
        assert measured["broad_pulses"] >= 10
        assert measured["line_syncs"] >= 400
        assert measured["tip_within_tolerance"]
        assert abs(measured["tip_frequency_measured_hz"] - 3.4e6) < 20e3

    def test_the_carrier_amplitude_is_read_in_codes(self, measured):
        assert measured["carrier_amplitude_tip_codes"] == pytest.approx(AMPLITUDE_CODES, rel=0.03)

    def test_the_floor_is_read_under_the_band(self, measured):
        """The planted density, back to within the measurement's own
        stated uncertainty, from the broad pulses alone; eighteen pulses
        of twenty bins leave a few tenths of a decibel."""
        uncertainty = measured["c_over_n0_uncertainty_db"]
        assert uncertainty < 0.5
        allowed = max(0.5, 3.0 * uncertainty)
        assert measured["c_over_n0_db_hz"] == pytest.approx(planted_c_over_n0_db_hz(), abs=allowed)
        assert measured["c_over_n0_from_mean_spectrum_db_hz"] == pytest.approx(
            planted_c_over_n0_db_hz(), abs=allowed)
        # white noise holds no lines, so the mean and the median agree
        assert abs(measured["in_band_mean_over_median_db"]) < 0.5

    def test_the_effective_bits_read_the_planted_white_noise(self, measured):
        """White noise everywhere means the spectrum's minimum is the
        planted noise, not the quantiser: the effective bits fall short of
        eight by half the log2 of the ratio. The minimum over many noisy
        blocks is corrected for its order-statistic bias, without which it
        read three decibels low."""
        arithmetic = cp.quantization_floor(8, RATE, CONTAINER_STEP * 256)["density_per_hz"]
        planted = 2.0 * (SIGMA_CODES * CONTAINER_STEP) ** 2 / RATE + arithmetic
        expected = 8.0 - 0.5 * np.log2(planted / arithmetic)
        assert measured["effective_bits"] == pytest.approx(expected, abs=0.25)
        assert measured["converter_over_arithmetic_db"] == pytest.approx(
            10 * np.log10(planted / arithmetic), abs=0.75)
        # and the whole spectrum is flat, so the flat region reaches down
        assert measured["converter_flat_from_hz"] < 2e6

    def test_the_envelope_reading_agrees_with_the_spectrum(self, measured):
        """Additive noise alone: the envelope's spread over the analytic
        band reads the same density as the spectrum."""
        assert measured["envelope_c_over_n0_db_hz"] == pytest.approx(
            measured["c_over_n0_db_hz"], abs=1.0)

    def test_the_continuum_reading_agrees_on_white_noise(self, measured):
        """White noise holds no lines and no skirt, so the median-based
        continuum and the mean-based total read the same floor."""
        assert measured["c_over_n0_continuum_db_hz"] == pytest.approx(
            measured["c_over_n0_db_hz"], abs=0.5)
        assert np.isfinite(measured["in_band_independent_bins"])
        assert measured["in_band_independent_bins"] < measured["in_band_bins"]

    def test_effective_bins_under_the_window(self):
        """A rectangular window leaves the bins independent; Blackman-Harris
        correlates them, so six adjacent bins hold about two and a half
        independent estimates - and a Monte Carlo of the block mean's
        relative variance agrees with the derived count."""
        from scipy.signal import windows

        length = 1000
        assert nb.effective_bins(np.ones(length), 6) == pytest.approx(6.0, abs=1e-6)
        window = windows.blackmanharris(length)
        effective = nb.effective_bins(window, 6)
        assert 2.2 < effective < 2.8
        rng = np.random.default_rng(3)
        means = np.array([
            (np.abs(np.fft.rfft(rng.normal(size=length) * window)) ** 2)[100:106].mean()
            for _ in range(3000)])
        relative_variance = means.var() / means.mean() ** 2
        assert relative_variance == pytest.approx(1.0 / effective, rel=0.15)

    def test_the_bounds_are_rf_noises(self):
        """The chance bound, the median factor and Boltzmann's constant are
        the ones tools/ringing_measure/rf_noise.py carries; that module is
        a script beside its tool rather than a package, so it is loaded
        from its path."""
        import importlib.util
        import os

        path = os.path.abspath(os.path.join(os.path.dirname(nb.__file__), "..", "..",
                                            "tools", "ringing_measure", "rf_noise.py"))
        spec = importlib.util.spec_from_file_location("rf_noise", path)
        rf_noise = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rf_noise)
        for averages in (1, 7, 90):
            assert nb._chance_bound(averages, 0.95) == pytest.approx(
                rf_noise.spectrum_confidence(averages, 0.95), rel=1e-12)
        assert nb.chi_square_median_factor(1) == pytest.approx(rf_noise.MEDIAN_OVER_MEAN, abs=1e-9)
        assert nb.BOLTZMANN_J_PER_K == rf_noise.BOLTZMANN

    def test_the_chi_square_median_factor(self):
        assert nb.chi_square_median_factor(1) == pytest.approx(np.log(2), abs=1e-3)
        assert nb.chi_square_median_factor(1000) == pytest.approx(1.0, abs=2e-3)

    def test_gap_closing_only_fills_short_holes(self):
        mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
        closed = nb.close_gaps(mask, 2)
        assert closed.tolist() == [True] * 5 + [False] * 3 + [True] * 2
