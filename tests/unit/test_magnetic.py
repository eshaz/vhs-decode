"""The magnetic residual: record level, and where its shape comes from.

Ethan: "There should be a shape that defines that against the curve the tape
will exhibit." There is, and it is the self-demagnetisation cap's crossover
moving along the band as the level changes.
"""

import numpy as np
import pytest

from vhsdecode.models import head_model
from vhsdecode.models import magnetic as mag

SPEED = head_model.mechanics_for("VHS", "NTSC", "SP")["writing_speed_m_s"]
BAND = np.linspace(0.5e6, 8e6, 512)


def test_the_control_reads_one_direction():
    """THE CONTROL, and it must be able to fail. Without the cap a linear
    depth law has one derivative shape at every level, so anything but one
    means the construction has gone wrong - which is exactly what happened
    when the depth was written as a fraction of the cap and this read 4.97."""
    result = mag.linear_control(BAND, SPEED)
    assert result["passes"]
    assert result["effective"] == pytest.approx(1.0, abs=0.15)


def test_the_control_catches_the_error_it_exists_for():
    """Reproduce the original mistake - depth as a fraction of the cap - and
    confirm it produces the degenerate response that fooled the first
    estimate. A control that cannot fail is not a control."""
    wavelength = SPEED / np.maximum(BAND, 1.0)
    cap = wavelength / mag.DEMAGNETISATION_DIVISOR
    # the WRONG construction: depth as a fraction of the cap
    for level in (0.5, 1.0, 2.0):
        depth = cap * level
        x = 2.0 * np.pi * depth / wavelength
        # x is now independent of frequency, so the response is a constant
        assert np.allclose(x, x[0])
        response = (1.0 - np.exp(-x)) / x
        logged = np.log(response)
        assert np.linalg.norm(logged - logged.mean()) < 1e-9


def test_the_cap_is_the_entire_source_of_the_level_axis():
    """Removing the self-demagnetisation cap collapses the level axis to one
    direction, so the cap binding at DIFFERENT FREQUENCIES FOR DIFFERENT
    LEVELS is where its independence comes from."""
    with_cap = mag.level_axis(BAND, SPEED)
    assert with_cap["effective"] > 2.0

    wavelength = SPEED / np.maximum(BAND, 1.0)
    shapes = []
    for level in with_cap["levels"]:
        values = []
        for offset in (+0.05, -0.05):
            depth = mag.NOMINAL_DEPTH_M * (level + offset)
            x = 2.0 * np.pi * depth / wavelength
            values.append((1.0 - np.exp(-x)) / np.maximum(x, 1e-12))
        logged = np.log(np.maximum(values[0], 1e-12)
                        / np.maximum(values[1], 1e-12))
        logged = logged - logged.mean()
        shapes.append(logged / np.linalg.norm(logged))
    singular = np.linalg.svd(np.array(shapes), compute_uv=False)
    share = singular ** 2 / (singular ** 2).sum()
    assert 1.0 / np.sum(share ** 2) == pytest.approx(1.0, abs=0.2)


def test_saturation_reduces_the_axis_rather_than_creating_it():
    """The opposite of the intuition: saturation compresses the range of
    depths the levels explore, so it takes information away."""
    plain = mag.level_axis(BAND, SPEED)["effective"]
    for law in (lambda L: 3.3 * (1.0 - np.exp(-L)), lambda L: 3.3 * np.tanh(L)):
        assert mag.level_axis(BAND, SPEED, saturation=law)["effective"] < plain


def test_the_crossover_moves_down_the_band_as_the_level_rises():
    """The shape, in one number: f = speed / (2 pi * nominal depth * level),
    so the cap takes over lower in the band at higher levels."""
    crossovers = [mag.crossover_hz(level, SPEED)
                  for level in (0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(crossovers, crossovers[1:]))
    # and it halves as the level doubles
    assert crossovers[1] / crossovers[2] == pytest.approx(2.0, rel=1e-6)


def test_the_cap_is_shallower_than_the_optimum_everywhere():
    """The guide's lambda/4 optimum is never reachable, because
    1/(2 pi) = 0.159 is smaller than 1/4."""
    assert mag.DEMAGNETISATION_DIVISOR > mag.OPTIMUM_DIVISOR
    wavelength = SPEED / BAND
    cap = mag.demagnetisation_cap(BAND, SPEED)
    assert np.all(cap < wavelength / mag.OPTIMUM_DIVISOR)


def test_the_depth_is_a_length_before_the_cap_is_applied():
    """The error the module exists to not repeat: written as a fraction of
    the cap, x = 2 pi d / lambda is frequency-independent and the response
    is constant. Written as a length, it is not."""
    depth = mag.recording_depth(BAND, 1.0, SPEED)
    wavelength = SPEED / np.maximum(BAND, 1.0)
    x = 2.0 * np.pi * depth / wavelength
    assert not np.allclose(x, x[0])
    # and the cap binds at the short-wavelength end
    assert x[-1] < x[0] or np.any(depth == mag.demagnetisation_cap(BAND, SPEED))
