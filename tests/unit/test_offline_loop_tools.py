"""Tests for the offline identification tools' pure functions.

The RF mapping places each polarity's baseband residual at its landing
carrier plus and minus the frequency with the even magnitude and the odd
phase; the channel identifier recovers a planted magnitude tilt from a
synthetic residual-channel directory with the decoder's filter divided out.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "tools", "ringing_measure")
sys.path.insert(0, TOOLS)

import channel_identify  # noqa: E402
import multidimensional_information_extrapolation as loop  # noqa: E402


def _filters():
    return {"ire0": 3.69e6, "vsync_ire": -40.0, "hz_ire": 7142.857}


def test_map_to_rf_satisfies_both_polarities_with_the_carrier_term():
    grid = np.arange(0.5e6, 8e6, 10e3)
    freqs = np.array([0.29e6, 0.5e6, 1.0e6])
    L_fall = np.array([0.2 + 0.1j, 0.1 - 0.05j, -0.3 + 0.05j])
    L_rise = np.array([-0.1 + 0.0j, 0.15 + 0.1j, 0.05 - 0.1j])
    var = np.full(3, 1e-4)
    valid = np.ones(3, dtype=bool)
    views = {"fall": (freqs, L_fall, var, valid), "rise": (freqs, L_rise, var, valid)}
    increment, variance = loop.map_to_rf(views, _filters(), grid, alpha=0.5)
    for view, L in (("fall", L_fall), ("rise", L_rise)):
        effect = loop.predicted_effect(increment, _filters(), grid, view, freqs)
        assert np.allclose(effect.real, -0.5 * L.real, atol=2e-3)
        assert np.allclose(effect.imag, -0.5 * L.imag, atol=2e-3)
    assert np.isfinite(variance[int(np.round((3.69e6 + 0.5e6 - grid[0]) / 10e3))])
    assert np.isinf(variance[5])


def test_map_to_rf_drops_a_wrapped_phase_and_honours_agreement():
    grid = np.arange(0.5e6, 8e6, 10e3)
    freqs = np.array([0.5e6, 1.0e6])
    L = np.array([0.2 + 4.0j, 0.1 + 0.1j])      # a phase beyond pi in bin 0
    var = np.array([1e-4, 1e-4])
    valid = np.array([True, True])
    agreement = np.array([1.0, 0.0])            # bin 1 did not respond
    views = {"rise": (freqs, L, var, valid, agreement)}
    increment, variance = loop.map_to_rf(views, _filters(), grid, alpha=0.5)
    effect = loop.predicted_effect(increment, _filters(), grid, "rise", freqs)
    assert np.isclose(effect[0].real, -0.5 * 0.2, atol=2e-3)
    assert abs(effect[0].imag) < 1e-3                 # the wrapped phase was dropped
    # bin 1 carried no response weight: no phase update reaches it, and its
    # magnitude moves only by the carrier bin's share (the carrier term)
    assert abs(effect[1].imag) < 1e-3
    assert np.isclose(effect[1].real, -(-increment.real)[
        int(np.round((3.69e6 - grid[0]) / 10e3))], atol=1e-6)


def test_landing_carriers_from_the_decoder_levels():
    carriers = loop.landing_carriers(_filters())
    assert carriers["rise"] == 3.69e6
    assert np.isclose(carriers["fall"], 3.69e6 - 40.0 * 7142.857)


def _synthetic_directory(tmp_path, tilt_per_hz):
    rate, n = 40e6, 4096
    rf_grid = np.arange(n // 2 + 1) * rate / n
    rf_video = np.exp(-((rf_grid - 4e6) / 3e6) ** 2).astype(np.complex128)
    np.savez(tmp_path / "decoder_filters.npz", freq_hz=rate, blocklen=n,
             rf_grid_hz=rf_grid, rf_video=rf_video, system="NTSC",
             tape_format="VHS", tape_speed="sp", ire0=3.69e6, hz_ire=7142.857,
             vsync_ire=-40.0, line_period_us=63.5556,
             downscale_taps=np.ones(16) / 16.0, fvideo=np.ones(n // 2 + 1),
             fdeemp=np.ones(n // 2 + 1))
    rng = np.random.default_rng(3)
    for index in range(6):
        carrier = 3.4e6 + 1.0e6 * rng.random((32, 400))
        carrier = np.sort(carrier, axis=1)          # slow sweeps
        path_gain = np.exp(tilt_per_hz * (carrier - 3.9e6))
        rf_gain = np.interp(carrier, rf_grid, np.abs(rf_video))
        amplitude = 2000.0 * path_gain * rf_gain * (1 + 0.002 * rng.standard_normal(carrier.shape))
        np.savez(tmp_path / f"field_{index:06d}.npz", amplitude=amplitude,
                 carrier=carrier, is_first_field=(index % 2 == 0), readloc=index,
                 linelocs=np.arange(34) * 2542.0, lineoffset=1)


def test_identify_recovers_a_planted_tilt(tmp_path):
    tilt = -0.2e-6   # nepers per Hz: -0.2 over 1 MHz
    _synthetic_directory(tmp_path, tilt)
    export = channel_identify.identify(str(tmp_path), bin_ire=4.0,
                                       minimum_population=16)
    believed = export["belief"] > 0.5
    assert believed.sum() > 10
    f = export["freqs_hz"][believed]
    g = export["log_H"][believed].real
    slope = np.polyfit(f - f.mean(), g, 1)[0]
    assert np.isclose(slope, tilt, rtol=0.15)
    assert export["site"] == channel_identify.SITE
    assert set(export["heads"]) == {"head_first", "head_second"}


def _fields_with_tip(clipped, seed=0):
    """Synthetic tbc fields in the decoder's own alignment: the line starts
    at the sync fall, the tip at -40 IRE, blanking at 0, with the same
    noise everywhere - or a tip clipped flat."""
    rng = np.random.default_rng(seed)
    lines, width = 40, 910
    field = np.zeros((lines, width))
    field[:, :67] = -40.0
    field += rng.standard_normal(field.shape) * 1.0
    if clipped:
        field[:, 2:66] = -40.0
    decode = type("D", (), {})()
    decode.fields = np.array([field])
    return decode


def test_tip_flatness_tells_a_clipped_tip_from_a_noisy_one():
    ratio, clipped = loop.tip_flatness(_fields_with_tip(False))
    assert 0.8 < ratio < 1.3 and not clipped
    ratio, clipped = loop.tip_flatness(_fields_with_tip(True))
    assert ratio < 0.1 and clipped


def test_fixed_band_metric_is_computed_on_the_declared_band_only():
    freqs = np.linspace(0.0, 3e6, 301)
    L = np.where(freqs <= 1.5e6, 0.1 + 0.2j, 9.0 + 9.0j)   # rubbish above
    var = np.full(len(freqs), 1e-4)
    ok = np.ones(len(freqs), dtype=bool)
    out = loop.fixed_band_metric({"fall": (freqs, L, var, ok)})
    assert out["fall"]["magnitude_rms"] == pytest.approx(0.1)
    assert out["fall"]["phase_rms"] == pytest.approx(0.2)
    assert out["pooled"]["bins"] == out["fall"]["bins"] < len(freqs)


def test_fixed_band_metric_ignores_bins_the_gauge_rejected():
    freqs = np.linspace(0.0, 1.4e6, 141)
    L = np.full(len(freqs), 0.1 + 0.0j)
    L[:50] = 9.0                     # rejected bins carry rubbish
    var = np.full(len(freqs), 1e-4)
    ok = np.ones(len(freqs), dtype=bool)
    ok[:50] = False
    out = loop.fixed_band_metric({"rise": (freqs, L, var, ok)})
    assert out["rise"]["magnitude_rms"] == pytest.approx(0.1)


def test_measured_jacobian_scales_the_update_and_drops_the_unreachable():
    grid = np.arange(0.5e6, 8e6, 10e3)
    freqs = np.array([0.5e6])
    L = np.array([0.2 + 0.0j])
    var = np.array([1e-4])
    valid = np.array([True])
    views = {"fall": (freqs, L, var, valid), "rise": (freqs, L, var, valid)}
    # a half-responding view needs twice the applied change
    full = loop.map_to_rf(views, _filters(), grid, 0.5,
                          jacobian={"fall": 1.0, "rise": 1.0})[0]
    half = loop.map_to_rf(views, _filters(), grid, 0.5,
                          jacobian={"fall": 0.5, "rise": 0.5})[0]
    assert np.allclose(half, 2.0 * full, atol=1e-9)
    # a view below the floor is dropped, not amplified
    loop.map_to_rf(views, _filters(), grid, 0.5,
                   jacobian={"fall": 0.05, "rise": 1.0})
    assert [v for v, _ in loop.map_to_rf.dropped_views] == ["fall"]


class _FakeDecode:
    """Enough of a decode for the time-file writer: two fields whose
    residual-channel exports name their line geometry."""

    def __init__(self, directory, lines=8, offset=1):
        self.residual_dir = str(directory)
        self.readlocs = np.array([100, 200])
        for index, readloc in enumerate(self.readlocs):
            np.savez(os.path.join(self.residual_dir, f"field_{index:06d}.npz"),
                     readloc=readloc, linelocs=np.arange(lines + 1) * 910.0,
                     lineoffset=offset - 1, time_per_line=np.zeros(lines))


def test_time_file_carries_the_accumulated_total_not_the_pass_share(tmp_path):
    """The consumer is stateless: it starts from the same line locations
    every decode and applies exactly what the file says, so the file must
    carry the TOTAL wanted from that base. A per-pass share would discard
    everything the previous pass applied."""
    decode = _FakeDecode(tmp_path)
    residual = np.full((2, 6), 0.004)
    error = np.full((2, 6), 1e-5)
    model, path = {}, str(tmp_path / "time.npz")
    totals = []
    for index in range(3):
        loop.write_time_response(path, decode, residual, error, 0.5, index,
                                 model)
        with np.load(path, allow_pickle=False) as z:
            assert str(z["semantics"]) == "total"
            totals.append(np.nanmax(np.abs(z["position_increment"])))
    # three passes at half the residual accumulate to three halves of it,
    # each shrunk by its own evidence garrote - here a residual measured
    # at 400 sigma, so the shrinkage is six parts in a million
    assert totals[0] == pytest.approx(0.002, rel=1e-4)
    assert totals[1] == pytest.approx(0.004, rel=1e-4)
    assert totals[2] == pytest.approx(0.006, rel=1e-4)


def test_time_file_cancels_rather_than_adds_to_the_residual(tmp_path):
    """A residual of +r is the line sitting late, so the total written
    must move it EARLIER: the sign is negative."""
    decode = _FakeDecode(tmp_path)
    loop.write_time_response(str(tmp_path / "t.npz"), decode,
                             np.full((2, 6), 0.004), np.full((2, 6), 1e-5),
                             0.5, 0, {})
    with np.load(str(tmp_path / "t.npz"), allow_pickle=False) as z:
        values = z["position_increment"]
        assert np.nanmin(values) < 0 and np.nanmax(values) <= 0


def test_the_regulator_is_evidence_not_bandwidth(tmp_path):
    """A tape's SPEED cannot change line to line, but its line PLACEMENT
    can, because the sync time base positions every line from its own
    per-line measurement. So a fast residual may be a real placement
    error and must not be band-limited away; what must not be written is
    the gauge's noise, which is a question about evidence per line."""
    decode = _FakeDecode(tmp_path, lines=200)
    lines = 200
    # a genuine line-to-line placement error, measured far above its own
    # error - exactly what a band limit would have destroyed
    fast = 0.004 * (-1.0) ** np.arange(lines)
    residual = np.stack([fast, fast])
    error = np.full((2, lines), 1e-5)
    loop.write_time_response(str(tmp_path / "fast.npz"), decode, residual,
                             error, 1.0, 0, {})
    with np.load(str(tmp_path / "fast.npz"), allow_pickle=False) as z:
        written = z["position_increment"][0]
        assert str(z["regulator"]) == "per-line evidence garrote"
    inside = np.isfinite(written)
    # it survives essentially whole: well-measured is well-corrected,
    # however fast it varies
    assert np.std(written[inside]) == pytest.approx(0.004, rel=0.05)

    # and the same residual measured at its own noise level contributes
    # nothing at all
    loop.write_time_response(str(tmp_path / "noise.npz"), decode, residual,
                             np.full((2, lines), 0.004), 1.0, 0, {})
    with np.load(str(tmp_path / "noise.npz"), allow_pickle=False) as z:
        assert not np.isfinite(z["position_increment"]).any()


def test_evidence_weight_is_the_garrote():
    # a residual at its own error carries nothing; at three sigma it
    # carries eight ninths of itself
    assert loop.evidence_weight(np.array([1.0]), np.array([1.0]))[0] == 0.0
    assert loop.evidence_weight(np.array([3.0]), np.array([1.0]))[0] == \
        pytest.approx(1.0 - 1.0 / 9.0)
    assert loop.evidence_weight(np.array([0.5]), np.array([1.0]))[0] == 0.0


def test_the_null_space_is_projected_out_of_the_update():
    """The magnitude relation annihilates any linear function of
    frequency and the phase relation any constant, so neither can change
    the demodulated output. Leaving them in a solution makes the table
    look larger than its effect and eats the safety clamp's headroom."""
    grid = np.arange(0.5e6, 8e6, 20e3)
    freqs = np.linspace(0.1e6, 1.2e6, 24)
    L = 0.05 * np.cos(2 * np.pi * freqs / 0.7e6) + 0.02j * np.ones(len(freqs))
    var = np.full(len(freqs), 1e-4)
    valid = np.ones(len(freqs), dtype=bool)
    views = {"fall": (freqs, L, var, valid), "rise": (freqs, L, var, valid)}
    increment, variance = loop.map_to_rf(views, _filters(), grid, 0.5)
    touched = np.isfinite(variance)
    applied = -increment[touched]
    # against the true bin positions, not a compressed index: the
    # projection is in frequency, and the touched bins need not be
    # contiguous
    axis = np.nonzero(touched)[0].astype(np.float64)
    # no linear term left in the magnitude, no constant in the phase
    assert abs(np.polyfit(axis, applied.real, 1)[0]) < 1e-12
    assert abs(np.mean(applied.imag)) < 1e-12
    # and the projection changes nothing the demodulator can see
    effect = loop.predicted_effect(increment, _filters(), grid, "fall", freqs)
    assert np.allclose(effect.real, -0.5 * L.real, atol=5e-3)


def test_time_closure_is_what_two_gauges_fail_to_agree_about():
    """One unknown per line and three gauges of it leaves closure
    relations. What fails to cancel cannot be a placement error, because
    a placement error moves every gauge together."""
    lines = 100
    common = 0.01 * np.sin(np.arange(lines) / 7.0)     # a real placement error
    disagreement = 0.004 * (-1.0) ** np.arange(lines)  # what cannot be one
    error = np.full(lines, 1e-4)
    out = loop.time_closure(common + disagreement, error, common, error)
    # the common part cancels exactly; only the disagreement survives
    assert out["closure_rms"] == pytest.approx(0.004, rel=1e-6)
    assert out["closure_chi2"] > 100
    # two gauges that agree leave nothing
    quiet = loop.time_closure(common, error, common, error)
    assert quiet["closure_rms"] == pytest.approx(0.0, abs=1e-12)
