"""Draft of tests/unit/test_depth_split.py - paste-ready once the module is
placed at vhsdecode/models/depth_split.py."""
import numpy as np
import pytest

try:
    from vhsdecode.models import depth_split as ds
except ImportError:                      # scratchpad run
    import depth_split as ds


def test_ethan_reference_reproduces_from_specification():
    rep = ds.reproduction()
    assert abs(rep["tape_per_sample_um"][1] - 0.405) < 0.001
    assert abs(rep["lambda_sync_tip_um"][1] - 1.71) < 0.005
    assert abs(rep["lambda_peak_white_um"][1] - 1.32) < 0.005
    assert abs(rep["lambda_chroma_um"][1] - 9.22) < 0.005
    assert abs(rep["read_depth_chroma_um"][1] - 1.467) < 0.001
    # his 0.2 um is the peak-white end of the deviation; the centre is 0.237
    assert 0.21 <= rep["read_depth_luma_um"][1] <= 0.27
    assert abs(rep["thickness_loss_luma_db"][1] - (-25.6)) < 0.1
    assert abs(rep["thickness_loss_chroma_db"][1] - (-10.2)) < 0.1


def test_one_depth_reproduces_both_figures():
    fit = ds.one_depth_for_both()
    assert abs(fit["depth_m"] - 4.37e-6) < 0.05e-6
    assert abs(fit["luma_residual_db"]) < 1.0 and abs(fit["chroma_residual_db"]) < 0.2


def test_ep_moves_the_wavelength_half_a_percent_not_three_times():
    sp = ds.wavelength_m(ds.FM_PEAK_HZ, ds.WRITING_SPEED_M_S["SP"])[0]
    ep = ds.wavelength_m(ds.FM_PEAK_HZ, ds.WRITING_SPEED_M_S["EP"])[0]
    assert abs(ep / sp - 1.0052) < 1e-3


def test_cap_makes_the_luma_depth_term_a_flat_four_decibels():
    grid = np.linspace(3.4e6, 4.4e6, 32)
    db = ds.DB_PER_NEPER * ds.band_specific_depth_nepers(grid, 4.5e-6, capped=True)
    assert np.allclose(db, -3.98, atol=0.01)


def test_shared_term_scales_exactly_with_frequency():
    assert ds.sharing_control()["passes"]


def test_small_depth_collapses_onto_wallace():
    assert ds.small_x_control()["passes"]
    assert ds.small_x_control()["coherence"] > 0.9999


def test_split_is_determined_only_in_the_transition_regime():
    deep = ds.identifiability_control(4.5e-6)
    shallow = ds.identifiability_control(ds.RECORD_DEPTH_M)
    assert deep["condition"] < 3.0 and deep["effective"] > 1.5
    assert shallow["condition"] > 10.0 and shallow["effective"] < 1.05


def test_wallace_delay_difference_matches_hilbert():
    from scipy.signal import hilbert
    d, v = 0.05e-6, 5.80
    f = np.arange(1.0, 200e6, 2e3); w = 2 * np.pi * f
    logmag = -2 * np.pi * d * f / v
    full = np.concatenate([logmag[::-1], logmag])
    phase = -np.imag(hilbert(full))[len(logmag):]
    tau = -np.gradient(phase, w)
    numeric = np.interp(ds.FM_PEAK_HZ, f, tau) - np.interp(ds.COLOUR_UNDER_HZ, f, tau)
    assert abs(numeric - ds.wallace_delay_difference_s(d)) < 0.05e-9
    assert abs(ds.wallace_delay_difference_s(d) * 1e9 - (-10.0)) < 0.05


def test_whole_coating_overshoots_the_measured_mismatch():
    verdict = ds.mismatch_verdict()
    assert abs(verdict["laws"]["whole coating 4.5 um"]["error_db"]) > 8.0
    assert abs(verdict["laws"]["JVC record depth 0.30 um, capped"]["error_db"]) < 1.5


def test_common_mode_timing_on_synthetic_pilots():
    rng = np.random.default_rng(0)
    lines = np.arange(263); fields = 10
    burst = rng.normal(0, 0.16, (fields, 263)); sync = rng.normal(0, 0.32, (fields, 263))
    perfield = burst - sync
    fixed_a = np.zeros(263); fixed_b = np.zeros(263) + 0.03
    out = ds.common_mode_timing(fixed_a, fixed_b, perfield, perfield, burst, burst, sync, sync, lines)
    assert out["passes"]
    assert abs(out["per_head_offset_ns"] + 0.03 * 69.84) < 0.01
