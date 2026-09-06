"""The random RF noise model, and the theorem it rests on.

The floor a derivation stops at should be PREDICTED, not fitted. These
tests check the prediction against realisations of the noise it describes -
if the law is right the ratio is one without anything being tuned.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "tools", "ringing_measure"))
import rf_noise as rn                                        # noqa: E402


RATE = 40e6
SAMPLES = 1 << 15
CARRIER = 3.9e6


def _demodulated(sigma, amplitude=1.0, seed=7):
    """A carrier with circular complex Gaussian noise, demodulated the way
    the decoder does it: the derivative of the unwrapped phase."""
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLES) / RATE
    noise = rng.normal(0, sigma, SAMPLES) + 1j * rng.normal(0, sigma, SAMPLES)
    signal = amplitude * np.exp(2j * np.pi * CARRIER * t) + noise
    demod = np.diff(np.unwrap(np.angle(signal))) * RATE / (2 * np.pi)
    return demod - demod.mean(), noise


def _spectrum(signal):
    power = np.abs(np.fft.rfft(signal)) ** 2 * 2.0 / (RATE * len(signal))
    return np.fft.rfftfreq(len(signal), d=1.0 / RATE), power


class TestTheDistribution:
    def test_the_envelope_is_rayleigh(self):
        _, noise = _demodulated(0.02)
        assert np.abs(noise).mean() == pytest.approx(0.02 * np.sqrt(np.pi / 2),
                                                     rel=0.02)

    def test_the_squared_envelope_is_exponential(self):
        _, noise = _demodulated(0.02)
        assert (np.abs(noise) ** 2).mean() == pytest.approx(2 * 0.02 ** 2,
                                                            rel=0.02)

    def test_the_phase_is_uniform(self):
        """Which is exactly why no realisation of the noise is knowable,
        and why 'removing the model' can only mean removing power."""
        _, noise = _demodulated(0.02)
        assert np.angle(noise).std() == pytest.approx(np.pi / np.sqrt(3),
                                                      rel=0.02)


class TestTheImageAfterTheDemodulator:
    @pytest.mark.parametrize("sigma", [0.01, 0.02, 0.05])
    def test_the_law_predicts_the_level_without_being_fitted(self, sigma):
        demod, _ = _demodulated(sigma)
        frequency, power = _spectrum(demod)
        predicted = rn.demodulated_density(frequency, 1.0, sigma, RATE)
        band = (frequency > 0.2e6) & (frequency < 3.0e6)
        assert np.mean(power[band] / predicted[band]) == pytest.approx(1.0,
                                                                       rel=0.05)

    def test_the_spectrum_rises_as_f_squared(self):
        demod, _ = _demodulated(0.02)
        frequency, power = _spectrum(demod)
        band = (frequency > 0.2e6) & (frequency < 3.0e6)
        slope = np.polyfit(np.log10(frequency[band]),
                           np.log10(power[band]), 1)[0]
        assert slope == pytest.approx(2.0, abs=0.15)

    def test_a_median_is_ln2_of_a_mean_and_that_is_not_the_law_failing(self):
        """THE TRAP. Comparing a measured median against a predicted mean
        reads as the law being wrong by 0.693 - and the tell is that the
        error is the same at every noise level, which is a constant rather
        than physics."""
        demod, _ = _demodulated(0.02)
        frequency, power = _spectrum(demod)
        predicted = rn.demodulated_density(frequency, 1.0, 0.02, RATE)
        band = (frequency > 0.2e6) & (frequency < 3.0e6)
        ratio = power[band] / predicted[band]
        assert np.median(ratio) / np.mean(ratio) == pytest.approx(
            rn.MEDIAN_OVER_MEAN, abs=0.02)


class TestRemoval:
    def test_pure_noise_is_reported_at_its_limit(self):
        demod, _ = _demodulated(0.02)
        frequency, power = _spectrum(demod)
        predicted = rn.demodulated_density(frequency, 1.0, 0.02, RATE)
        band = (frequency > 0.2e6) & (frequency < 3.0e6)
        result = rn.verdict(power[band], predicted[band], averages=1)
        assert result["share_at_limit"] > 0.9

    def test_structure_stands_above_the_floor(self):
        """A component the noise model does not account for must survive
        the removal, or the removal is deleting signal."""
        demod, _ = _demodulated(0.02)
        tone = 20000.0 * np.sin(2 * np.pi * 1.0e6
                                * np.arange(len(demod)) / RATE)
        frequency, power = _spectrum(demod + tone)
        predicted = rn.demodulated_density(frequency, 1.0, 0.02, RATE)
        near = np.argmin(np.abs(frequency - 1.0e6))
        result = rn.remove(power[near:near + 1], predicted[near:near + 1],
                           averages=1)
        assert result["excess_power"][0] > 0
        assert not result["at_limit"][0]

    def test_averaging_tightens_the_chance_bound(self):
        """A single periodogram bin is exponential and its tail is long;
        averaging N of them is chi-square with 2N and the bound falls."""
        assert rn.spectrum_confidence(1) > rn.spectrum_confidence(10)
        assert rn.spectrum_confidence(10) > rn.spectrum_confidence(100)
        assert rn.spectrum_confidence(1000) == pytest.approx(1.0, abs=0.1)


class TestTheLimits:
    def test_below_threshold_the_model_does_not_claim_to_reach(self):
        assert rn.above_threshold(1.0, 0.02)
        assert not rn.above_threshold(1.0, 0.5)

    def test_the_quantizer_floor_is_exact_not_measured(self):
        step = 1.0 / 256
        density = rn.quantization_density(step, RATE, np.array([1e6]))
        assert density[0] == pytest.approx((step ** 2 / 12.0) / (RATE / 2))
