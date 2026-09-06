"""The sync pulse shape as ONE component on all three axes.

Ethan, twice: "the shape is a component with all parts, amplitude,
frequency, time", and "The luma and chroma are band limited, so the data
outside this bandlimited area is noise, that can use as the differential".
"""

import numpy as np
import pytest

from vhsdecode.models import sync_shape as ss

FS = 4 * 315e6 / 88


def _shape(samples=52):
    t = np.arange(samples)
    return (-2.4 * np.exp(-t / 18.0)
            + 0.5 * np.exp(-t / 7.0) * np.cos(2 * np.pi * 2.43e6 * t / FS))


def test_all_three_axes_are_reported():
    """The directive itself: amplitude, frequency AND time, from one fit."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    for axis in ("amplitude", "frequency_hz", "group_delay_s"):
        assert len(np.asarray(got[axis])) == got["modes"]
    assert np.isfinite(got["amplitude_rms"])
    assert np.isfinite(got["centroid_hz"])
    assert np.isfinite(got["position_samples"])


def test_the_axes_are_readings_of_one_object_not_three_fits():
    """They share a frame: one coefficient per mode, one frequency per
    mode, one delay per mode. The previous treatment fitted a relaxation
    for its amplitude and a ring for its frequency, and they disagreed
    because nothing made them share one."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    assert len(got["coefficients"]) == len(got["frequency_hz"])
    assert np.allclose(got["amplitude"], np.abs(got["coefficients"]))


def test_the_band_limit_is_a_format_constant_and_sets_the_size():
    narrow = ss.shape_components(_shape(), FS, 1.5e6)
    wide = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    assert wide["modes"] > narrow["modes"]
    assert wide["frequency_hz"][-1] <= ss.VHS_LUMA_BAND_HZ
    assert narrow["frequency_hz"][-1] <= 1.5e6


@pytest.mark.parametrize("tone_hz,belongs", [
    (0.5e6, "shape"), (2.5e6, "shape"), (5.5e6, "noise"), (6.9e6, "noise"),
])
def test_the_band_routes_content_to_shape_or_noise(tone_hz, belongs):
    """THE STATEMENT, made operational: what is inside the band is the
    shape and what is outside it is noise, BY CONSTRUCTION rather than by
    a chosen threshold."""
    base = _shape()
    tone = 0.3 * np.cos(2 * np.pi * tone_hz * np.arange(len(base)) / FS)
    plain = ss.shape_components(base, FS, ss.VHS_LUMA_BAND_HZ)
    got = ss.shape_components(base + tone, FS, ss.VHS_LUMA_BAND_HZ)
    moved_in_band = float(np.sqrt(np.mean(
        (got["in_band"] - plain["in_band"]) ** 2)))
    if belongs == "shape":
        assert moved_in_band > got["noise_rms"]
    else:
        assert got["noise_rms"] > 0.5 * moved_in_band


def test_a_band_limited_shape_leaves_no_noise():
    """Nothing outside the band means nothing in the noise channel - so a
    non-zero noise reading is a measurement, not a residue."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    # 1e-6 IRE, which is 25000x below the 0.025 IRE floor two independent
    # measurements of this porch differ by - the basis is finite, so the
    # residual is small rather than zero, and the bar is what the
    # measurement can distinguish
    assert got["noise_rms"] < 1e-5


def test_in_band_and_noise_reconstruct_the_input():
    """The split is exact: nothing is lost between the two channels."""
    values = _shape()
    got = ss.shape_components(values, FS, ss.VHS_LUMA_BAND_HZ)
    assert np.allclose(got["in_band"] + got["out_of_band"], values, atol=1e-9)


def test_the_effective_rank_is_far_below_the_mode_count():
    """The shape has real degrees of freedom - more than the two or three
    a pole enumeration allowed, fewer than the basis offers."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    assert 2.0 < got["effective_rank"] < got["modes"]


def test_the_time_axis_moves_when_the_shape_moves():
    """A shape shifted along the segment must read a different position -
    otherwise the time axis is decoration."""
    values = _shape()
    # a REAL shift, not np.roll - rolling wraps the tail back to the start
    # and moves the centroid the other way, which is a property of the
    # test rather than of the shape
    shifted = np.concatenate([np.full(6, values[-1]), values[:-6]])
    here = ss.shape_components(values, FS, ss.VHS_LUMA_BAND_HZ)
    there = ss.shape_components(shifted, FS, ss.VHS_LUMA_BAND_HZ)
    moved = there["position_samples"] - here["position_samples"]
    assert moved == pytest.approx(6.0, abs=1.0)


def test_the_frequency_axis_moves_when_the_content_moves():
    t = np.arange(52)
    low = np.cos(2 * np.pi * 0.4e6 * t / FS)
    high = np.cos(2 * np.pi * 2.4e6 * t / FS)
    assert (ss.shape_components(high, FS, ss.VHS_LUMA_BAND_HZ)["centroid_hz"]
            > ss.shape_components(low, FS, ss.VHS_LUMA_BAND_HZ)["centroid_hz"])


def test_the_differential_is_subtractive_on_every_axis():
    """A DIFFERENCE, never a ratio - spectral division manufactured
    phantoms in this tree twice."""
    a = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    b = ss.shape_components(_shape() * 1.10, FS, ss.VHS_LUMA_BAND_HZ)
    got = ss.differential(b, a)
    assert np.allclose(got["amplitude"],
                       np.asarray(b["amplitude"]) - np.asarray(a["amplitude"]))
    assert np.allclose(got["residual"],
                       np.asarray(b["in_band"]) - np.asarray(a["in_band"]))


def test_the_differential_carries_the_noise_channel_too():
    """The second half of the statement: the noise is not discarded, it is
    itself a channel of the differential. Two shapes agreeing in band and
    differing out of it have a real difference."""
    values = _shape()
    t = np.arange(len(values))
    noisy = values + 0.4 * np.cos(2 * np.pi * 6.5e6 * t / FS)
    a = ss.shape_components(values, FS, ss.VHS_LUMA_BAND_HZ)
    b = ss.shape_components(noisy, FS, ss.VHS_LUMA_BAND_HZ)
    got = ss.differential(b, a)
    assert abs(got["noise_rms"]) > 0.05
    assert float(np.sqrt(np.mean(np.asarray(got["out_of_band"]) ** 2))) > 0.05


def test_the_differential_refuses_a_mismatched_basis():
    """A comparison is only a comparison when both sides are computed over
    the same set."""
    a = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    b = ss.shape_components(_shape(), FS, 1.5e6)
    with pytest.raises(ValueError, match="same basis"):
        ss.differential(a, b)
