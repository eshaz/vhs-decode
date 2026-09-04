"""Tests for the runtime channel equalizer's table.

A unity export must build a table that is exactly unity (so a decode under
it is byte-identical to one without), a shaped export must build the
conjugate-symmetric inverse that is unity at DC and outside the declared
band, and a file identified for another system or another site is refused.
"""

import logging
import types

import numpy as np
import pytest

from vhsdecode import channel_eq

N = 4096
RATE = 40e6


def _rf(luma_eq=0):
    rf = types.SimpleNamespace(
        Filters={"RFVideo": np.ones(N, dtype=np.complex128)},
        blocklen=N,
        freq_hz=RATE,
        blockcut=128,
        blockcut_end=64,
        system="NTSC",
        tape_format="VHS",
        tape_speed="SP",
        SysParams={"ire0": 3.69e6},
        options=types.SimpleNamespace(luma_eq=luma_eq),
    )
    rf.iretohz = lambda ire, spec=False: 3.69e6 + ire * 7140.0
    return rf


def _export(path, log_h, **extra):
    freqs = np.linspace(0.1e6, 6e6, 300)
    declared = {
        "freqs_hz": freqs,
        "log_H": np.broadcast_to(log_h, freqs.shape).astype(np.complex128),
        "belief": np.ones_like(freqs),
        "site": "rf_pre_demod_post_envelope",
        "freq_hz": RATE,
    }
    declared.update(extra)
    np.savez(path, **declared)
    return str(path)


def test_unity_export_builds_an_exactly_unity_table(tmp_path):
    rf = _rf()
    table = channel_eq.load(rf, _export(tmp_path / "unity.npz", 0.0), 1.0)
    assert np.array_equal(table, np.ones(N, dtype=np.complex128))
    assert channel_eq.is_active(rf)
    assert rf.Filters["ChannelEQ"] is table
    assert rf._ringing_state["channel_eq_active"]
    assert np.array_equal(channel_eq.effective_rf_response(rf), rf.Filters["RFVideo"])


def test_shaped_export_builds_the_inverse_within_the_band(tmp_path):
    rf = _rf()
    table = channel_eq.load(rf, _export(tmp_path / "shaped.npz", 0.1 + 0.2j), 1.0)
    grid = np.arange(N // 2 + 1) * RATE / N
    half = table[: N // 2 + 1]
    assert table[0] == 1.0
    assert table[N // 2].imag == 0.0
    assert np.allclose(table[1 : N // 2], np.conj(table[N - 1 : N // 2 : -1]))
    inside = (grid > 0.2e6) & (grid < 5.9e6)
    assert np.allclose(np.abs(half[inside]), np.exp(-0.1))
    assert np.allclose(np.abs(half[grid > 6.1e6]), 1.0)


def test_amount_scales_the_inverse(tmp_path):
    rf = _rf()
    table = channel_eq.load(rf, _export(tmp_path / "half.npz", 0.1), 0.5)
    grid = np.arange(N // 2 + 1) * RATE / N
    inside = (grid > 0.2e6) & (grid < 5.9e6)
    assert np.allclose(np.abs(table[: N // 2 + 1][inside]), np.exp(-0.05))


def test_effective_response_is_the_product(tmp_path):
    rf = _rf()
    rng = np.random.default_rng(1)
    rf.Filters["RFVideo"] = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    table = channel_eq.load(rf, _export(tmp_path / "shaped.npz", 0.3 - 0.1j), 1.0)
    assert np.allclose(channel_eq.effective_rf_response(rf), rf.Filters["RFVideo"] * table)


def test_inactive_without_a_table():
    rf = _rf()
    assert not channel_eq.is_active(rf)
    assert channel_eq.effective_rf_response(rf) is rf.Filters["RFVideo"]


def test_mismatched_system_is_refused(tmp_path):
    with pytest.raises(ValueError):
        channel_eq.load(_rf(), _export(tmp_path / "pal.npz", 0.0, system="PAL"), 1.0)


def test_wrong_site_is_refused(tmp_path):
    with pytest.raises(ValueError):
        channel_eq.load(_rf(), _export(tmp_path / "bb.npz", 0.0, site="baseband"), 1.0)


def test_missing_key_is_refused(tmp_path):
    path = tmp_path / "short.npz"
    np.savez(path, freqs_hz=np.linspace(1e5, 6e6, 10), log_H=np.zeros(10, complex))
    with pytest.raises(KeyError):
        channel_eq.load(_rf(), str(path), 1.0)


def test_luma_eq_supersession_is_announced(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    channel_eq.load(_rf(luma_eq=0.5), _export(tmp_path / "unity.npz", 0.0), 1.0)
    assert any("superseded" in record.getMessage() for record in caplog.records)


def test_a_wild_table_is_clamped_not_obeyed(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    rf = _rf()
    table = channel_eq.load(rf, _export(tmp_path / "wild.npz", -4.4 + 0.0j), 1.0)
    grid = np.arange(N // 2 + 1) * RATE / N
    inside = (grid > 0.2e6) & (grid < 5.9e6)
    # asked for a gain of exp(4.4) = 81; clamped to exp(1.0) = 2.72
    assert np.allclose(np.abs(table[: N // 2 + 1][inside]),
                       np.exp(channel_eq.MAXIMUM_LOG_MAGNITUDE))
    assert any("clamped" in record.getMessage() for record in caplog.records)


def test_a_non_finite_bin_is_left_uncorrected(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    freqs = np.linspace(0.1e6, 6e6, 300)
    log_h = np.zeros(len(freqs), dtype=np.complex128)
    log_h[100] = np.nan
    path = tmp_path / "nan.npz"
    np.savez(path, freqs_hz=freqs, log_H=log_h, belief=np.ones(len(freqs)),
             site="rf_pre_demod_post_envelope", freq_hz=RATE)
    table = channel_eq.load(_rf(), str(path), 1.0)
    assert np.all(np.isfinite(table))
    assert any("non-finite" in record.getMessage() for record in caplog.records)
    # the surviving bins still carry the identified response
    assert np.allclose(np.abs(table), 1.0)
