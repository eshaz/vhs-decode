"""The multiply the chroma instrument was missing.

`burst_envelope` reads the burst as `hypot(c_n, c_{n-1})`, which throws away
the angle between the two samples - and that angle is the anti-Hermitian half
of the chroma response, which carries the tape's own roll-off across the
colour-under band. `_complex_envelope` keeps it.
"""

import numpy as np
import pytest

from vhsdecode.chroma import _complex_envelope

SUBCARRIER_HZ = 315e6 / 88.0


def _planted(count=256, seed=0):
    """a known complex envelope riding a real carrier at four times fsc"""
    n = np.arange(count)
    envelope = ((1.0 + 0.4 * np.sin(2 * np.pi * n / 60.0))
                * np.exp(1j * (0.7 + 1.3 * n / count)))
    return envelope, np.real(envelope * (1j) ** n)


def test_the_complex_envelope_recovers_a_planted_phase():
    truth, signal = _planted()
    got = _complex_envelope(signal[None, :], 0)[0]
    interior = slice(4, -4)
    assert np.abs(np.abs(got[interior]) - np.abs(truth[interior])).max() < 5e-3
    residual = np.angle(got[interior] * np.conj(truth[interior]))
    assert np.abs(residual).max() < 5e-3, "the phase is the whole point"


def test_it_beats_hypot_on_the_magnitude_as_well():
    """Not a trade of accuracy for phase - better at both, because hypot
    reads two different samples and is centred half a sample early."""
    truth, signal = _planted()
    interior = slice(4, -4)
    complex_error = np.abs(
        np.abs(_complex_envelope(signal[None, :], 0)[0][interior])
        - np.abs(truth[interior])).max()
    hypot_error = np.abs(
        np.hypot(signal[1:], signal[:-1])[interior]
        - np.abs(truth[1:][interior])).max()
    assert complex_error < hypot_error / 5.0


def test_the_three_tap_kills_the_conjugate_image_at_nyquist():
    """`y_n = b_n + (-1)^n conj(b_n)`; the alias is at Nyquist and the
    symmetric 3-tap is exactly zero there."""
    _truth, signal = _planted()
    n = np.arange(signal.size)
    raw = 2.0 * signal * ((-1j) ** n)
    filtered = _complex_envelope(signal[None, :], 0)[0]

    def nyquist_bin(values):
        return float(np.abs(np.fft.fft(values))[values.size // 2])

    survives = nyquist_bin(filtered) / nyquist_bin(raw)
    assert survives < 0.02, "the conjugate image must be suppressed"
    # and it is the untrimmed ends that leave anything at all
    assert survives > 0, "exactly zero would mean the ends were trimmed"


def test_the_rotation_is_referenced_to_the_line_and_not_the_window():
    """An offset in the column origin is a constant phase error on every
    measurement taken from the result, so the caller's first column must
    enter the rotation."""
    _truth, signal = _planted()
    at_zero = _complex_envelope(signal[None, 8:], 0)[0]
    at_eight = _complex_envelope(signal[None, 8:], 8)[0]
    turn = np.angle(at_eight[4:-4] * np.conj(at_zero[4:-4]))
    # eight samples of (-j)^n is exactly two full turns, i.e. no net rotation
    assert np.abs(turn).max() < 1e-9
    at_one = _complex_envelope(signal[None, 8:], 1)[0]
    quarter = np.angle(at_one[4:-4] * np.conj(at_zero[4:-4]))
    assert np.abs(np.abs(quarter).mean() - np.pi / 2) < 1e-9
