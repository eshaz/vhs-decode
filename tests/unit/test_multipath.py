"""A tapped-delay channel fitted in all dimensions, judged out of sample."""

import numpy as np
import pytest

from vhsdecode.models import multipath as mp


def test_planted_taps_come_back_at_their_delays_and_amplitudes():
    frequency = np.linspace(0.2e6, 4.0e6, 900)
    planted = [(0.35e-6, 0.25 + 0.10j), (1.10e-6, -0.12 + 0.05j)]
    departure = np.log(mp.channel(frequency, planted))
    out = mp.fit(frequency, departure, taps=3)
    assert out["removed_fraction"] > 0.95
    found = sorted(out["taps"][:2], key=lambda t: t["delay_s"])
    for tap, (delay, amplitude) in zip(found, planted):
        assert tap["delay_s"] == pytest.approx(delay, abs=out["delay_step_s"] * 2)
        assert tap["magnitude"] == pytest.approx(abs(amplitude), rel=0.15)


def test_a_weak_echo_is_minimum_phase_and_a_dominant_one_is_not():
    """The physics that ruled the general propagation model out: an echo
    weaker than the direct path adds no all-pass at all."""
    frequency = np.linspace(0.2e6, 4.0e6, 900)
    for amplitude in (0.3, 0.7, 0.95):
        response = mp.channel(frequency, [(0.5e-6, amplitude)])
        # a minimum-phase response has no zeros outside the unit circle, so
        # its log magnitude and phase are a Hilbert pair; the test here is
        # simply that the response never approaches a zero
        assert np.abs(response).min() > 1e-3
    dominant = mp.channel(frequency, [(0.5e-6, 1.5)])
    assert np.abs(dominant).min() < 0.6      # it passes close to a zero


def test_the_fit_uses_magnitude_and_phase_together():
    """Fitting the phase alone lets a tap match a phase it has no
    magnitude for, so the departure is fitted complex."""
    frequency = np.linspace(0.2e6, 4.0e6, 600)
    departure = np.log(mp.channel(frequency, [(0.4e-6, 0.2 + 0.2j)]))
    out = mp.fit(frequency, departure, taps=1)
    tap = out["taps"][0]
    # the recovered amplitude carries both parts, not just a modulus
    assert tap["phase_deg"] == pytest.approx(45.0, abs=15.0)
    assert tap["magnitude"] == pytest.approx(abs(0.2 + 0.2j), rel=0.2)


def test_a_fit_that_does_not_hold_out_is_visible_as_such():
    rng = np.random.default_rng(0)
    frequency = np.linspace(0.2e6, 4.0e6, 700)
    real = np.log(mp.channel(frequency, [(0.5e-6, 0.3 + 0.1j)]))
    # the same channel on both halves: it must hold out
    out = mp.held_out(frequency, real, real, taps=2)
    assert out["held_out_removed"] > 0.9
    # SMOOTH noise, independent on the two halves. It must be smooth, or
    # four taps cannot fit white noise in the first place and the test
    # would pass for the wrong reason: against 700 independent bins the
    # in-sample removal is only 1.4 per cent.
    def smooth(seed):
        raw = (rng.standard_normal(700) + 1j * rng.standard_normal(700))
        kernel = np.exp(-0.5 * (np.arange(-40, 41) / 12.0) ** 2)
        kernel /= kernel.sum()
        return np.convolve(raw, kernel, mode="same")

    first, second = smooth(0), smooth(1)
    out = mp.held_out(frequency, first, second, taps=4)
    assert out["in_sample_removed"] > 0.2          # it fits its own half
    assert out["held_out_removed"] < 0.15          # and not the other
