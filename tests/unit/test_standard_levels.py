"""Normalising the corrected signal onto the video standard's levels.

Ethan: "after RF correction, I do want to normalize the signal so it sits
within the video standard." And, on what remains: "That layer is explained by
missing video equipment and camera levels measurements, which we don't know
since they are constant over the signal."

Both statements are here. The constant IS unknowable to any differential -
that is the first test - and the standard is what supplies it.
"""

import numpy as np
import pytest

from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import pair_dimension as pd
from vhsdecode.models import standard_levels as sl


def test_a_constant_is_invisible_to_the_differential():
    """A gain rescales every component alike and an offset shifts them
    alike, so neither moves a single DIRECTION. Source equipment levels are
    in the null space of any differential method - there is nothing for them
    to be differentiated against."""
    rng = np.random.default_rng(55)
    length, count = 380, 13
    shape = rng.standard_normal(length)

    def ensemble(gain, offset):
        return {f"c{i}": {"frequency":
                          gain * (shape + 0.4 * rng.standard_normal(length))
                          + offset} for i in range(count)}

    plain = ie.ellipsoid(ensemble(1.0, 0.0), "frequency")
    scaled = ie.ellipsoid(ensemble(1.7, 0.0), "frequency")
    shifted = ie.ellipsoid(ensemble(1.0, 12.0), "frequency")
    for other in (scaled, shifted):
        assert other["rank"] == plain["rank"]
        assert other["significant"] == plain["significant"]


def test_the_standard_supplies_what_the_differential_cannot_see():
    """Three different unknown source constants, one set of standard levels
    out. That is what a standard is for."""
    rate = 50e6
    line = pd.spec_sync(rate, 4096)
    for gain, offset in ((1.0, 0.0), (1.7, 12.0), (0.6, -5.0)):
        seen = gain * line + offset
        edges = sl.half_amplitude_crossings(seen, rate)
        result = sl.compliance(seen, edges["bottom"], edges["top"],
                               measured_white=edges["top"] + 100.0 * gain)
        assert result["white_lands_at_ire"] == pytest.approx(100.0, abs=0.5)
        assert result["white_within_tolerance"]
        normalised = sl.normalise(seen, edges["bottom"], edges["top"])
        assert normalised.min() == pytest.approx(sl.SYNC_TIP_IRE, abs=0.5)
        assert normalised.max() == pytest.approx(sl.BLANKING_IRE, abs=0.5)


def test_the_pulse_is_read_at_its_own_half_amplitude():
    """A fixed level stops being a width measurement once the amplitude has
    changed. Read at its own half amplitude the pulse measures the same
    whatever gain the source applied."""
    rate = 50e6
    line = pd.spec_sync(rate, 4096)
    widths, rises = [], []
    for gain in (1.0, 1.7, 0.6):
        edges = sl.half_amplitude_crossings(gain * line, rate)
        widths.append(edges["width_s"])
        rises.append(edges["rise_s"])
    assert max(widths) - min(widths) < 2.0 / rate
    assert max(rises) - min(rises) < 2.0 / rate
    assert widths[0] == pytest.approx(pd.SYNC_WIDTH_S, abs=2.0 / rate)


def test_reference_white_is_what_the_compliance_check_is_for():
    """The two anchors land by construction and prove nothing. White is
    over-determined once the gain and offset are fixed, so where it lands
    measures whether the chain is linear between the tip and white."""
    rate = 50e6
    line = pd.spec_sync(rate, 4096)
    edges = sl.half_amplitude_crossings(line, rate)
    # a chain with a 12% gain error above blanking that the anchors cannot see
    result = sl.compliance(line, edges["bottom"], edges["top"],
                           measured_white=edges["top"] + 88.0)
    assert result["white_error_ire"] == pytest.approx(-12.0, abs=0.5)
    assert result["gain_error"] == pytest.approx(-0.12, abs=0.01)
    assert not result["white_within_tolerance"]


def test_a_signal_driven_past_the_standard_is_reported():
    """Landing the levels correctly while driving the peaks past the
    standard's limits has not put the signal within the standard."""
    rate = 50e6
    line = pd.spec_sync(rate, 4096)
    edges = sl.half_amplitude_crossings(line, rate)
    tame = sl.compliance(line, edges["bottom"], edges["top"])
    assert tame["within_luma_limit"]
    wild = np.concatenate([line, np.full(256, 150.0)])
    wild_edges = sl.half_amplitude_crossings(wild, rate)
    hot = sl.compliance(wild, wild_edges["bottom"], wild_edges["top"])
    assert not hot["within_composite_limit"]
