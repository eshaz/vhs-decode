"""The capture chain's own noise, and which of it reaches the picture.

Ethan: 'I think some of the remaining component is the noise profile of
the RF capture chain. The adc that I have does have some noise and energy
leaking from the computer and the clock crystal. The clock crystal should
be derivable from the cxadc spec. I have replaced the crystal with a 40MHz
crystal.'
"""

import math

import numpy as np
import pytest

from vhsdecode.models import capture_chain_noise as cn
from vhsdecode.models import vhs_specification

SUBCARRIER = 315e6 / 88.0


def test_the_sample_rate_follows_the_crystal_and_the_bit_depth():
    """Eight-bit takes the clock, ten-bit takes half, per the reference."""
    assert cn.sample_rate(40e6, 8)["sample_rate_hz"] == 40e6
    assert cn.sample_rate(40e6, 10)["sample_rate_hz"] == 20e6
    stock = cn.sample_rate(cn.STOCK_CRYSTAL_HZ, 8)
    assert math.isclose(stock["sample_rate_hz"], 28.63636e6, rel_tol=1e-12)
    assert stock["nyquist_hz"] == 0.5 * stock["sample_rate_hz"]


def test_the_crystal_family_scales_and_says_how_to_falsify_it():
    at_stock = cn.crystal_spurs(cn.STOCK_CRYSTAL_HZ)["lines_hz"]
    at_forty = cn.crystal_spurs(40e6)["lines_hz"]
    ratio = 40e6 / cn.STOCK_CRYSTAL_HZ
    for name, value in at_stock.items():
        assert math.isclose(at_forty[name] / value, ratio, rel_tol=1e-12)
    assert cn.crystal_spurs(40e6)["scales_with_the_crystal"]
    assert "crystal change" in cn.crystal_spurs(40e6)["falsified_by"]


def test_the_reference_sixth_is_where_the_reference_measured_it():
    """The cxadc reference names 4.773 MHz as one sixth of the stock part."""
    lines = cn.crystal_spurs(cn.STOCK_CRYSTAL_HZ)["lines_hz"]
    assert math.isclose(lines["crystal / 6"], 4.7727e6, rel_tol=1e-4)
    # and with Ethan's part it moves to two thirds of ten
    assert math.isclose(cn.crystal_spurs(40e6)["lines_hz"]["crystal / 6"],
                        40e6 / 6.0, rel_tol=1e-12)


def test_the_stock_crystal_puts_a_spur_exactly_on_the_colour():
    """Eight times the subcarrier, by the card's own design.

    This is why the crystal change matters beyond its rate: the stock
    part's eighth sub-harmonic is the colour, and a spur there cannot be
    filtered out by anything that keeps the colour.
    """
    stock = cn.lands_on_the_subcarrier(cn.STOCK_CRYSTAL_HZ, SUBCARRIER)
    assert stock["any"]
    assert "crystal / 8" in stock["hits"]
    assert math.isclose(stock["ratio_to_subcarrier"], 8.0, rel_tol=1e-6)
    assert abs(stock["hits"]["crystal / 8"]["offset_hz"]) < 10.0


def test_the_replaced_crystal_takes_that_spur_off_the_colour():
    replaced = cn.lands_on_the_subcarrier(40e6, SUBCARRIER)
    assert not replaced["any"]
    assert not math.isclose(replaced["ratio_to_subcarrier"], 8.0, rel_tol=1e-3)
    # it moves to five megahertz, which is outside the colour by a wide margin
    assert math.isclose(cn.crystal_spurs(40e6)["lines_hz"]["crystal / 8"],
                        5.0e6, rel_tol=1e-12)


def test_a_spur_reaches_the_video_by_beating_with_the_carrier():
    """The demodulator is not linear, which is the whole route."""
    tip = vhs_specification.carrier_hz_for_ire(-40.0)
    white = vhs_specification.carrier_hz_for_ire(100.0)
    carriers = np.linspace(tip, white, 5)
    out = cn.beats_into_the_band([6.0e6, 12.0e6], carriers, 3.0e6)
    close, far = out["spurs"]
    assert close["reaches_the_video"]          # 6.0 beats to 1.6 to 2.6 MHz
    assert not far["reaches_the_video"]        # 12.0 beats to 7.6 and beyond
    assert close["closest_beat_hz"] < 3.0e6
    assert far["closest_beat_hz"] > 3.0e6


def test_a_spur_inside_the_fm_band_needs_no_demodulator_at_all():
    tip = vhs_specification.carrier_hz_for_ire(-40.0)
    white = vhs_specification.carrier_hz_for_ire(100.0)
    inside = cn.beats_into_the_band([0.5 * (tip + white)],
                                    np.linspace(tip, white, 5), 3.0e6)
    assert inside["spurs"][0]["inside_the_fm_band"]
    assert inside["spurs"][0]["reaches_the_video"]


def test_the_fixed_spurs_are_labelled_as_an_identification_not_a_measurement():
    """What is measured is that they do not move, not what makes them."""
    assert 6.0e6 in cn.FIXED_SPURS_HZ
    assert 12.0e6 in cn.FIXED_SPURS_HZ
    assert "label" in cn.FIXED_SPUR_STATUS
    assert "measured fixed" in cn.FIXED_SPUR_STATUS


def test_the_beat_frequencies_on_the_sync_tip_are_where_the_test_looked():
    """The prediction that failed its control, kept so it is not remade.

    The response is measured on the sync pulse, so the carrier is at the
    tip and each spur beats to a fixed frequency. A capture taken on an
    oscilloscope, with no card and no crystal, showed the same elevation
    at those frequencies, so the elevation is not the capture chain.
    """
    tip = vhs_specification.carrier_hz_for_ire(-40.0)
    lines = cn.crystal_spurs(40e6)["lines_hz"]
    assert math.isclose(abs(lines["crystal / 6"] - tip), 3.2667e6, rel_tol=1e-3)
    assert math.isclose(abs(lines["crystal / 8"] - tip), 1.6e6, rel_tol=1e-3)
    assert math.isclose(abs(6.0e6 - tip), 2.6e6, rel_tol=1e-3)
