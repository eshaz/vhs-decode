"""The blanking-to-sync-tip spacing as the luma channel's absolute scale.

Ethan: 'do another correction using the spacing between 0 and -40 ire ...
First detect clipping though in the data, since we will use that as the
absolute lowest point of data in the luma channel.' And on the treatment:
'don't exclude whole lines, it may clip low only, and may not be clipping
at all, in which case means we don't care.'
"""

import numpy as np
import pytest

from vhsdecode.models import sync_depth as sd

BLACK, WHITE = 16243.2, 56320.0
PER_IRE = (WHITE - BLACK) / 100.0


def test_a_pile_up_is_a_clip_and_a_smooth_tail_is_not():
    rng = np.random.default_rng(0)
    clean = BLACK - 20 * PER_IRE + 40 * rng.standard_normal((200, 27))
    out = sd.detect_clip(clean, BLACK, WHITE)
    # a ratio alone is not enough: sparse data gives a large one from a
    # single count against zeros, so a meaningful fraction must sit at the
    # floor as well
    assert not out["clipped"]
    assert out["fraction_at_floor"] < 1e-3
    # the same shape, but sitting low enough that its noise crosses the
    # container's floor and is folded onto it
    low = 60.0 + 40.0 * rng.standard_normal((200, 27))
    pinned = np.clip(low, 0, None)
    out = sd.detect_clip(pinned, BLACK, WHITE)
    assert out["clipped"] and out["at_container_floor"]
    assert out["pile_up_ratio"] > 5.0
    assert out["fraction_at_floor"] > 1e-3
    assert out["container_floor_ire"] == pytest.approx(-40.530, abs=0.001)


def test_the_spacing_gives_the_gain_and_a_planted_deficit_comes_back():
    for gain in (1.0, 0.9, 0.84):
        lines = np.zeros((500, 200))
        lines[:, 20:46] = BLACK - gain * 40.0 * PER_IRE     # the tip
        lines[:, 114:130] = BLACK                            # blanking
        out = sd.spacing(lines, BLACK, WHITE, range(20, 46), range(114, 130))
        assert out["spacing_ire"] == pytest.approx(gain * 40.0, abs=1e-6)
        assert out["gain"] == pytest.approx(gain, abs=1e-9)
        assert out["deficit_ire"] == pytest.approx(40.0 * (1 - gain), abs=1e-6)
        assert sd.correction(out)["divide_levels_by"] == pytest.approx(gain)


def test_estimating_through_the_clip_beats_both_leaving_it_and_masking_it():
    """A clip FOLDS the low tail onto the floor; masking DELETES it, which
    biases the same way and further. Measured on a planted Gaussian with
    6.66 per cent censored: leaving it +5.93, masking +27.76, estimating
    through it +0.08."""
    rng = np.random.default_rng(1)
    truth_mean, sigma = 300.0, 200.0
    samples = truth_mean + sigma * rng.standard_normal(200000)
    clipped = np.clip(samples, 0, None)
    left = clipped.mean() - truth_mean
    masked = clipped[clipped > 0].mean() - truth_mean
    estimated = sd.censored_mean(clipped)
    assert masked > left > 0                       # masking is the worst
    assert abs(estimated["mean"] - truth_mean) < 0.1 * abs(left)
    assert estimated["sigma"] == pytest.approx(sigma, rel=0.01)
    assert estimated["fitted"] and estimated["censored"] > 0
    # and with no clip at all it simply returns the mean, unfitted
    clean = sd.censored_mean(samples + 10000.0)
    assert not clean["fitted"] and clean["censored"] == 0


def test_an_implausible_gain_is_refused_rather_than_applied():
    lines = np.zeros((10, 200))
    lines[:, 20:46] = BLACK - 4.0 * PER_IRE          # a 4 IRE "sync pulse"
    lines[:, 114:130] = BLACK
    out = sd.spacing(lines, BLACK, WHITE, range(20, 46), range(114, 130))
    with pytest.raises(ValueError, match="outside anything a chain"):
        sd.correction(out)


def test_the_field_slope_separates_a_level_drift_from_a_gain_drift():
    """Ethan: 'the levels should be constant for the entire field'. Both
    are specified constants, so a slope is an error - and the two slopes
    separate a level shift, which leaves the spacing alone, from a gain
    drift, which does not."""
    height = 263
    line_index = np.arange(height)
    lines = np.zeros((height * 4, 200))
    # a pure LEVEL drift: both levels move together, the spacing does not
    drift = np.tile(0.01 * line_index, 4)[:, None]
    lines[:, 20:46] = BLACK - 40.0 * PER_IRE + drift * PER_IRE
    lines[:, 114:130] = BLACK + drift * PER_IRE
    out = sd.field_slope(lines, BLACK, WHITE, range(20, 46), range(114, 130),
                         field_height=height)
    assert out["blanking"]["slope_ire_per_field"] == pytest.approx(2.63, abs=0.02)
    assert out["differential_ire_per_field"] == pytest.approx(0.0, abs=1e-6)

    # a pure GAIN drift: the tip moves and blanking does not
    lines[:, 114:130] = BLACK
    lines[:, 20:46] = BLACK - 40.0 * PER_IRE + drift * PER_IRE
    out = sd.field_slope(lines, BLACK, WHITE, range(20, 46), range(114, 130),
                         field_height=height)
    assert out["blanking"]["slope_ire_per_field"] == pytest.approx(0.0, abs=1e-6)
    assert out["differential_ire_per_field"] == pytest.approx(-2.63, abs=0.02)
    assert out["gain_drift_fraction"] == pytest.approx(-2.63 / 40.0, abs=1e-3)
