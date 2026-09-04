"""Tests for the --ire0_adjust hsync submode's per-field hz_ire.

On a degenerate field (dropout / severe sync collapse) the backporch and hsync
measurement windows read the same level, so

    hz_ire = (ire0 - hsync_level) / -vsync_ire

comes out exactly 0. hz_to_output_array then divides out_scale by it; numba's
default error model raises rather than returning inf, so an unguarded field
aborts the entire decode with ZeroDivisionError.
"""

import logging
import types

import numpy as np
import pytest

import vhsdecode.field as vf
from vhsdecode.field import FieldShared

OUTLINECOUNT = 263
OUTLINELEN = 910
GLOBAL_HZ_IRE = 8571.43


@pytest.fixture(autouse=True)
def _logger():
    if getattr(vf.ldd, "logger", None) is None:
        vf.ldd.logger = logging.getLogger("test")


def _field(ire0_adjust="backporch,hsync"):
    """A field stub carrying only what hz_to_output touches."""
    rf = types.SimpleNamespace(
        DecoderParams={
            "ire0": 4_542_857.0,
            "hz_ire": GLOBAL_HZ_IRE,
            "vsync_ire": -40.0,
            "track_ire0_offset": [0.0, 0.0],
        },
        SysParams={"outputZero": 1024.0},
        options=types.SimpleNamespace(ire0_adjust=ire0_adjust, export_raw_tbc=False),
        track_phase=None,
    )
    return types.SimpleNamespace(
        rf=rf,
        # The real field derives these from the format's timing
        # (`FieldShared.level_windows`); the stub supplies what that
        # derivation produces for NTSC VHS at 4fsc, so the geometry the
        # tests below write into the input still lands where it should:
        # the sync tip inside [0, 74) and blanking after it.
        level_windows=lambda: ((7, 60), (112, 128)),
        # the real vectorised measurement, so the stub exercises it
        _window_levels=FieldShared._window_levels,
        outlinecount=OUTLINECOUNT,
        outlinelen=OUTLINELEN,
        out_scale=358.4,
        field_number=0,
    )


def _flat_input(level=5_000_000.0):
    """A flat field: every measurement window reads the identical level."""
    return np.full(OUTLINECOUNT * OUTLINELEN, level, dtype=np.float32)


def test_degenerate_field_does_not_abort_the_decode():
    """ire0 == hsync_level -> hz_ire == 0. Previously ZeroDivisionError."""
    out = FieldShared.hz_to_output(_field(), _flat_input())

    assert out.dtype == np.uint16
    assert len(out) == OUTLINECOUNT * OUTLINELEN


def test_degenerate_field_falls_back_to_the_global_hz_ire():
    """The guarded field decodes as if the hsync adjustment were unavailable.

    Same flat input through the backporch-only submode never computes a per-field
    hz_ire at all, so it already uses the global one. If the fallback works, the
    two must agree.
    """
    guarded = FieldShared.hz_to_output(_field("backporch,hsync"), _flat_input())
    backporch_only = FieldShared.hz_to_output(_field("backporch"), _flat_input())

    np.testing.assert_array_equal(guarded, backporch_only)


def test_hsync_alone_derives_gain_from_two_measured_levels():
    """The hsync submode's gain must come from measured-minus-measured.

    Historically, requesting hsync without backporch mixed the assumed
    DecoderParams ire0 with the measured hsync level - a scale that is not a
    measurement. Now the porch is measured for the gain whenever hsync is
    requested, so hsync-alone and backporch,hsync agree on hz_ire; they
    differ only in whether ire0 is re-anchored.
    """
    inp = _flat_input()
    for line in range(OUTLINECOUNT):
        base = line * OUTLINELEN
        inp[base : base + 74] = 4_000_000.0

    from lddecode.utils import hz_to_output_array

    f = _field("hsync")
    out = FieldShared.hz_to_output(f, inp)
    # measured porch 5e6, measured hsync 4e6 -> hz_ire 25000; ire0 stays the
    # decoder parameter because re-anchoring was not requested.
    expected = hz_to_output_array(
        inp,
        f.rf.DecoderParams["ire0"],
        (5_000_000.0 - 4_000_000.0) / 40.0,
        f.rf.SysParams["outputZero"],
        f.rf.DecoderParams["vsync_ire"],
        f.out_scale,
    )
    np.testing.assert_array_equal(out, expected)


def test_hsync_alone_degenerate_field_still_guarded():
    """The zero/non-finite gain guard covers the measured-measured path too."""
    out = FieldShared.hz_to_output(_field("hsync"), _flat_input())
    reference = FieldShared.hz_to_output(
        _field("doesnotmatch"), _flat_input()
    )
    np.testing.assert_array_equal(out, reference)


def test_healthy_field_still_uses_its_own_hz_ire():
    """The guard must not disturb a field whose windows read different levels.

    hz_to_output measures hsync over the derived tip window and black over
    the derived back porch window, so the sync level has to
    cover the whole first window to move the median.
    """
    inp = _flat_input()
    for line in range(OUTLINECOUNT):
        base = line * OUTLINELEN
        inp[base : base + 74] = 4_000_000.0

    # hz_ire = (5e6 - 4e6) / 40 = 25000, which is not the global 8571.43 --
    # so a field that never hits the guard must not decode like one that does.
    out = FieldShared.hz_to_output(_field(), inp)
    backporch_only = FieldShared.hz_to_output(_field("backporch"), inp)

    assert out.dtype == np.uint16
    assert not np.array_equal(out, backporch_only)


class TestLevelWindows:
    """`FieldShared.level_windows` derives the measurement spans from the
    format's timing. Both spans were hard-coded per system and both were
    misplaced; these are the regression guards for that."""

    @staticmethod
    def _field(system):
        import math
        from vhsdecode.field import FieldShared

        if system == "NTSC":
            outfreq, lpf = 4 * 315e6 / 88 / 1e6, 6.6e6
            sys_params = {"hsyncPulseUS": 4.7, "colorBurstUS": (5.3, 7.8),
                          "activeVideoUS": (9.45, 62.5), "outfreq": outfreq}
        else:
            outfreq, lpf = 4 * 4433618.75 / 1e6, 3.4e6
            sys_params = {"hsyncPulseUS": 4.7, "colorBurstUS": (5.6, 7.85),
                          "activeVideoUS": (10.5, 62.5), "outfreq": outfreq}
        rf = types.SimpleNamespace(SysParams=sys_params,
                                   DecoderParams={"video_lpf_freq": lpf})
        stub = types.SimpleNamespace(
            rf=rf, usectooutpx=lambda x: x * outfreq)
        return FieldShared.level_windows(stub), sys_params, outfreq

    def test_the_porch_starts_at_the_burst_end_not_the_burst_start(self):
        """The old NTSC window began at 74, which is the burst START, so it
        sat in the sync rise's settling tail rather than on the porch."""
        (_, porch), sys_params, outfreq = self._field("NTSC")
        assert porch[0] >= sys_params["colorBurstUS"][1] * outfreq - 1
        assert porch[0] > 74

    def test_the_porch_stops_short_of_active_video(self):
        """A window that reaches the active transition picks up the approach
        to it and moves with picture content; every window stopping a
        settling guard short measured 5-50x cleaner."""
        (_, porch), sys_params, outfreq = self._field("NTSC")
        active_start = sys_params["activeVideoUS"][0] * outfreq
        assert porch[1] < active_start
        assert active_start - porch[1] >= 4

    def test_the_tip_stops_before_the_sync_rise(self):
        """The old tip window ran to `ire0_backporch[0] - 4` = 70, and the
        rise is at output sample 68 - so it measured through the edge."""
        (tip, _), sys_params, outfreq = self._field("NTSC")
        rise = sys_params["hsyncPulseUS"] * outfreq
        assert tip[1] < rise
        assert tip[0] > 0

    def test_both_windows_are_non_degenerate_on_both_systems(self):
        for system in ("NTSC", "PAL"):
            (tip, porch), _, _ = self._field(system)
            assert tip[1] - tip[0] >= 4, system
            assert porch[1] - porch[0] >= 4, system

    def test_the_windows_do_not_overlap_each_other(self):
        for system in ("NTSC", "PAL"):
            (tip, porch), _, _ = self._field(system)
            assert tip[1] <= porch[0], system
