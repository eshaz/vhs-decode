"""Reading the levels from the carrier's frequency instead of the level.

SMPTE 32M clause 3.9.1.1.4 maps the sync tip to 3.4 MHz and peak white to
4.4, so the 140 IRE excursion IS the 1.0 MHz deviation and a level is a
frequency in other units. The frequency is read from the RF before any
stage; the level is read from the picture after all of them.
"""

import numpy as np
import pytest

from vhsdecode.models import level_from_frequency as lf


def _planted(rate=40e6, seconds=0.004, gain=1.0, chroma=0.0):
    """An FM carrier carrying sync pulses at a chosen level gain."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    line = 1.0 / 15734.264
    phase_in_line = (t % line) / line
    tip = phase_in_line < (4.7e-6 / line)
    frequency = np.where(tip, lf.SYNC_TIP_HZ,
                         lf.SYNC_TIP_HZ + gain * (lf.BLANKING_HZ - lf.SYNC_TIP_HZ))
    signal = np.cos(2 * np.pi * np.cumsum(frequency) / rate)
    if chroma:
        signal = signal + chroma * np.cos(2 * np.pi * 629371.0 * t)
    return signal, rate


def test_the_exchange_rate_between_the_two_units_is_the_specification():
    assert lf.hz_per_ire() == pytest.approx(1.0e6 / 140.0)
    assert lf.BLANKING_HZ == pytest.approx(3.4e6 + 1.0e6 * 40.0 / 140.0)
    assert lf.DEVIATION_HZ == 1.0e6


def test_the_sync_finder_uses_the_specified_duty_cycle():
    signal, rate = _planted()
    got = lf.instantaneous(signal, rate)
    found = lf.find_sync(got["frequency_hz"], rate)
    assert found["duty_percent"] == pytest.approx(100 * 4.7e-6 * 15734.264, abs=0.01)
    # about one pulse per line over the guarded interior
    lines = got["frequency_hz"].size / rate * 15734.264
    assert 0.7 * lines < found["found"] <= lines + 1


def test_a_planted_gain_comes_back_from_the_carrier():
    for gain in (1.0, 0.9):
        signal, rate = _planted(gain=gain, seconds=0.006)
        out = lf.levels(signal, rate)
        assert out["gain"] == pytest.approx(gain, rel=0.05)
        assert out["spacing_ire"] == pytest.approx(40.0 * gain, rel=0.05)


def test_the_band_limit_is_what_keeps_the_burst_out_of_the_luma():
    """A VHS track carries both carriers added, and the instantaneous
    frequency of a sum is neither of theirs."""
    signal, rate = _planted(seconds=0.006, chroma=0.5)
    out = lf.levels(signal, rate)
    assert out["gain"] == pytest.approx(1.0, rel=0.08)
    # without the band limit the chroma corrupts it
    got = lf.instantaneous(signal, rate, band_hz=(0.1e6, 20e6))
    assert np.std(got["frequency_hz"]) > 3.0 * np.std(
        lf.instantaneous(signal, rate)["frequency_hz"])


def test_the_pair_reports_its_own_disagreement():
    out = lf.agreement(1.02, 0.873)
    assert out["difference"] == pytest.approx(0.147)
    assert out["relative"] == pytest.approx(0.147 / 0.873)
    assert "share nothing" in out["why"]
