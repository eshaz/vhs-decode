"""The composite channel from the specifications that define it.

The first thing to pin is a negative: SMPTE 170M imposes no bandwidth
restriction on the luminance, so there is no specified luma response to
scale by, and the familiar 4.2 MHz belongs to the transmission channel
rather than to the signal. Everything else here is a cited number.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import composite_channel as cc


def test_the_standard_specifies_no_luma_bandwidth_and_the_limit_is_the_channels():
    frequency = np.array([1e6, 4.0e6, 5.0e6])
    composite = cc.luma_response(frequency, source="composite")
    assert composite["specified_flat"]
    assert not composite["limit_applies"]
    assert np.all(composite["response"] == 1.0)      # flat everywhere
    television = cc.luma_response(frequency, source="television")
    assert television["limit_applies"]
    assert television["limit_hz"] == pytest.approx(4.2e6)
    assert television["response"][0] == 1.0
    assert np.isnan(television["response"][2])       # past the channel's edge
    assert "7.1" in composite["cite"]


def test_the_multiburst_is_how_the_luma_response_is_measured():
    reference = cc.multiburst_reference()
    assert tuple(reference["frequency_hz"]) == (0.5e6, 1e6, 2e6, 3e6, 3.58e6, 4.2e6)
    assert reference["amplitude_ire"] == 50.0
    # equal amplitudes are the specification, so the tolerance is the bar
    assert reference["tolerance_db"] == pytest.approx(0.086, abs=0.002)


def test_the_colour_difference_limit_hits_the_specified_points():
    frequency = np.array([1.3e6, 3.6e6])
    out = cc.colour_difference_limit(frequency)
    assert out["decibels"][0] == pytest.approx(-2.0)
    assert out["decibels"][1] == pytest.approx(-20.0)
    assert "7.2" in out["cite"]
    # the 1953 Q channel is a different, narrower set
    q = cc.colour_difference_limit(np.array([0.4e6, 0.6e6]), channel="Q1953")
    assert q["decibels"][0] == pytest.approx(-2.0)
    assert q["decibels"][1] == pytest.approx(-6.0)
    # nothing is extrapolated past the last named point
    assert np.isnan(cc.colour_difference_limit(np.array([8e6]))["decibels"][0])


def test_the_separator_passes_through_its_two_specified_half_power_points():
    low, high = cc.SEPARATOR_MINUS_3DB_HZ
    out = cc.separator_response(np.array([low, high, math.sqrt(low * high)]))
    decibels = 20 * np.log10(out["response"])
    assert decibels[0] == pytest.approx(-3.0, abs=0.05)
    assert decibels[1] == pytest.approx(-3.0, abs=0.05)
    assert decibels[2] == pytest.approx(0.0, abs=0.05)
    # the order is an assumption and says so, and the implied centre is not
    # quite the stated one
    assert "no order" in out["assumption"]
    assert out["centre_hz"] == pytest.approx(3.546e6, rel=1e-3)
    assert out["centre_hz"] != out["specified_centre_hz"]


def test_the_burst_doubler_is_a_factor_of_two_and_is_flagged():
    chain = cc.chroma_chain()
    doubler = next(s for s in chain["steps"] if "doubler" in s["step"])
    assert doubler["gain_db"] == 6.0
    assert doubler["gain_linear"] == pytest.approx(2.0, rel=0.005)
    assert "factor of two" in chain["burst_doubler_warning"]
    # and the down-conversion is an exact ratio, not a microsecond figure
    down = next(s for s in chain["steps"] if "down-conversion" in s["step"])
    assert down["carrier_hz"] == pytest.approx(40.0 * cc.NTSC_LINE_RATE_HZ)
    assert down["carrier_hz"] == pytest.approx(629370.6, abs=0.1)
    # every step carries a clause
    assert all("SMPTE" in s["cite"] for s in chain["steps"])


def test_the_burst_sits_off_the_separators_centre_so_the_response_has_a_slope():
    """The specification's own three numbers put the carrier on a slope.

    Centred 3.58 MHz with half power at 3.08 and 4.08 gives a geometric
    centre of 3.5449 MHz, which is 34.6 kHz BELOW the subcarrier, so the
    burst sits on the falling side and the response there is not flat.
    """
    shape = cc.burst_amplitude_response()
    assert shape["centre_offset_hz"] < 0.0
    assert shape["slope_db_per_mhz"] < 0.0
    assert math.isclose(shape["slope_db_per_mhz"], -1.1803, abs_tol=1e-3)
    # the burst's band is set by its own duration, not by a chosen number
    assert math.isclose(shape["half_band_hz"],
                        cc.NTSC_SUBCARRIER_HZ / cc.BURST_CYCLES, rel_tol=1e-12)
    # and the tilt across the main lobe follows the slope's sign
    assert shape["tilt_db"] < 0.0


def test_the_specified_tilt_moves_no_bulk_phase_and_no_centroid():
    """It is odd about the burst's centre, so both sums are unmoved.

    This is the correction that matters: a tilt reads as a ramp ACROSS the
    burst and not as a rotation OF it, so anything that sums the burst
    before taking an angle sees none of it.
    """
    out = cc.burst_quadrature_prediction()
    assert abs(out["bulk_phase_deg"]) < 1e-9
    assert abs(out["centroid_shift_samples"]) < 1e-9
    assert out["phase_span_deg"] > 4.0
    assert math.isclose(out["peak_excursion_deg"],
                        out["phase_span_deg"] / 2.0, rel_tol=1e-6)


def test_symmetrising_the_response_about_the_carrier_removes_the_whole_tilt():
    """The control: an even response gives no ramp, to machine precision."""
    out = cc.burst_quadrature_prediction()
    assert abs(out["control_span_deg"]) < 1e-9
    assert out["control_quadrature_share"] < 1e-12
    assert out["quadrature_share"] > 1e-3


def test_the_burst_band_narrows_as_the_burst_lengthens():
    """The 1/T law, on the burst's own duration."""
    short = cc.burst_amplitude_response(cycles=4.0)
    long = cc.burst_amplitude_response(cycles=16.0)
    assert short["half_band_hz"] > long["half_band_hz"]
    assert abs(short["half_band_db"]) > abs(long["half_band_db"])


def test_the_burst_specification_has_one_home():
    """The 9 cycles and the 40 IRE are cited here and bound elsewhere."""
    assert cc.BURST_CYCLES == 9.0
    assert cc.BURST_IRE_PEAK_TO_PEAK == 40.0
    # 227.5 cycles a line is exactly half a turn a line
    assert math.isclose(cc.SUBCARRIER_CYCLES_PER_LINE % 1.0, 0.5, abs_tol=1e-12)
