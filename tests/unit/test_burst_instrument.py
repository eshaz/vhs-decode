"""The burst as the chroma's instrument, in the same three dimensions the
sync pulse gives the luma.

Ethan: 'It's going to be based on the up-heterodyned burst shape, the
entire three dimensions, how the color stretches, the active area is
contents we want to keep.'
"""

import math

import numpy as np
import pytest

from vhsdecode.models import burst_instrument as bi
from vhsdecode.models import composite_channel as cc

RATE = 4.0 * 315e6 / 88.0


def _ideal(n=128):
    want = int(round(bi.BURST_DURATION_S * RATE))
    envelope = np.zeros(n, dtype=complex)
    start = (n - want) // 2
    envelope[start:start + want] = 1.0
    return envelope, want


def test_the_burst_is_the_coarser_probe_in_frequency_and_the_finer_in_time():
    out = bi.probe_resolution_hz()
    assert out["cycles"] == 9.0
    assert out["duration_us"] == pytest.approx(2.514, abs=0.001)
    assert out["resolution_hz"] == pytest.approx(397.7e3, rel=1e-3)
    # coarser in frequency than the sync pulse's 212.8 kHz
    assert out["resolution_hz"] > 1.0 / 4.7e-6
    # and far finer in time: a degree of subcarrier against a sync edge
    per_degree = 1.0 / (360.0 * cc.NTSC_SUBCARRIER_HZ)
    assert per_degree * 1e9 == pytest.approx(0.776, abs=0.002)
    assert per_degree * 1e9 < 14.0 / 10.0


def test_the_stretch_follows_the_lost_bandwidth_and_matches_the_theory():
    """A rectangle of width W convolved with a Gaussian of sigma has an
    amplitude second moment of W^2/12 + sigma^2, so the stretch grows with
    the loss. Weighting by power instead makes it FALL, which is what the
    first version of this did."""
    ideal, want = _ideal()
    assert bi.envelope_response(ideal, RATE)["stretch"] == pytest.approx(1.0)
    for sigma in (2.0, 4.0, 8.0):
        kernel = np.exp(-0.5 * (np.arange(-31, 32) / sigma) ** 2)
        kernel /= kernel.sum()
        smeared = np.convolve(ideal, kernel, mode="same")
        out = bi.envelope_response(smeared, RATE)
        theory = math.sqrt((want ** 2 / 12.0 + sigma ** 2) / (want ** 2 / 12.0))
        assert out["stretch"] == pytest.approx(theory, rel=1e-3)
        assert out["stretch"] > 1.0


def test_the_amplitude_undoes_the_record_side_doubler():
    ideal, _ = _ideal()
    out = bi.amplitude(ideal, doubler_applied=True)
    assert out["doubler_db"] == 6.0
    assert out["doubler_linear"] == pytest.approx(2.0, rel=0.005)
    # the chroma-referred level is half the burst's, because the record
    # side raised the burst by six decibels and the chroma by nothing
    assert out["chroma_referred"] == pytest.approx(out["burst_peak_to_peak"] / 2.0, rel=0.005)
    raw = bi.amplitude(ideal, doubler_applied=False)
    assert raw["chroma_referred"] == pytest.approx(raw["burst_peak_to_peak"])


def test_the_timing_reads_a_planted_phase_at_subcarrier_resolution():
    ideal, _ = _ideal()
    for degrees in (-30.0, 0.0, 12.0, 90.0):
        out = bi.timing(ideal * np.exp(1j * math.radians(degrees)))
        assert out["phase_deg"] == pytest.approx(degrees, abs=1e-6)
        assert out["seconds"] == pytest.approx(degrees * out["per_degree_s"], rel=1e-9)
    assert bi.timing(ideal)["per_degree_s"] * 1e9 == pytest.approx(0.776, abs=0.002)


def test_all_three_dimensions_come_back_together():
    ideal, _ = _ideal()
    out = bi.three_dimensions(ideal * 0.75 * np.exp(1j * 0.4), RATE)
    assert set(out) >= {"frequency", "amplitude", "time"}
    assert out["frequency"]["stretch"] == pytest.approx(1.0)
    assert out["time"]["phase_rad"] == pytest.approx(0.4, abs=1e-6)
    assert out["amplitude"]["chroma_referred"] > 0


def test_the_differential_is_exact_inside_the_burst_band():
    """Dividing the specified shape back out is a division, so it is exact.

    Inside the band the correction inverts the response bin by bin; the
    residual on the round trip is entirely the out-of-band energy the
    correction deliberately leaves alone.
    """
    envelope, _ = _ideal(512)
    offset = np.fft.fftfreq(envelope.size, d=1.0 / RATE)
    response = cc.separator_response(cc.NTSC_SUBCARRIER_HZ + offset)["response"]
    through = np.fft.ifft(np.fft.fft(envelope) * response)
    out = bi.amplitude_differential(through, RATE, doubler_applied=False)
    inside = np.abs(offset) <= out["half_band_hz"]
    recovered = np.fft.fft(np.asarray(out["corrected"]))
    wanted = np.fft.fft(envelope)
    assert np.allclose(recovered[inside], wanted[inside], atol=1e-9)
    assert out["bins_corrected"] == int(inside.sum())
    assert out["is_complex"]
    assert np.iscomplexobj(out["corrected"])


def test_the_differential_leaves_the_band_the_standard_never_described():
    """Outside the burst's own lobe the division would be by nearly nothing."""
    envelope, _ = _ideal(512)
    out = bi.amplitude_differential(envelope, RATE, doubler_applied=False)
    assert out["bins_corrected"] < envelope.size
    assert math.isclose(out["half_band_hz"],
                        cc.NTSC_SUBCARRIER_HZ / bi.BURST_CYCLES, rel_tol=1e-12)


def test_the_differential_and_the_level_undo_the_doubler_the_same_way():
    """A gain is not a shape, so the two readings must not disagree."""
    envelope, _ = _ideal(256)
    with_doubler = bi.amplitude_differential(envelope, RATE, doubler_applied=True)
    without = bi.amplitude_differential(envelope, RATE, doubler_applied=False)
    ratio = abs(with_doubler["corrected_amplitude"]) / abs(without["corrected_amplitude"])
    level = (bi.amplitude(envelope, True)["chroma_referred"]
             / bi.amplitude(envelope, False)["chroma_referred"])
    assert math.isclose(ratio, level, rel_tol=1e-12)
    assert math.isclose(ratio, 10.0 ** (-cc.BURST_DOUBLER_DB / 20.0), rel_tol=1e-12)


def test_the_per_line_model_comes_from_the_specification_not_from_a_fit():
    """227.5 cycles a line is half a turn; the tape's rotation is 90 degrees."""
    assert math.isclose(bi.LINE_ROTATION_RAD["composite"], math.pi, abs_tol=1e-12)
    assert math.isclose(math.degrees(bi.LINE_ROTATION_RAD["colour_under"]),
                        cc.PHASE_ROTATION_DEG_PER_LINE, abs_tol=1e-12)
    with pytest.raises(ValueError):
        bi.per_line(np.ones((2, 8)), domain="invented")


def test_the_per_line_model_recovers_a_planted_series_exactly():
    lines = 64
    amplitude = np.full((4, lines), cc.BURST_IRE_PEAK_TO_PEAK)
    phase = np.tile((np.arange(lines) * np.pi + 0.3) % (2.0 * np.pi), (4, 1))
    out = bi.per_line(amplitude, phase, domain="composite")
    assert out["amplitude_residual_rms"] < 1e-12
    assert out["phase_residual_sd_deg"] < 1e-9
    assert abs(out["rotation_departure_deg_per_line"]) < 1e-9
    assert out["rotation_confirmed"]
    assert out["amplitude_complex"]["is_complex"]
    assert "relation" in out


def test_a_planted_rotation_error_is_found_and_the_spec_rate_is_refused():
    """The residual's SLOPE is what puts the specified rotation on trial."""
    lines, planted = 120, math.radians(0.01)
    index = np.arange(lines)
    phase = np.tile((index * (np.pi + planted) + 0.3) % (2.0 * np.pi), (6, 1))
    out = bi.per_line(np.full((6, lines), 40.0), phase, domain="composite")
    assert math.isclose(out["rotation_departure_deg_per_line"], 0.01, rel_tol=1e-6)
    assert not out["rotation_confirmed"]


def test_the_phase_residual_does_not_hang_on_one_lines_noise():
    """Anchoring on the first line made every residual carry that line.

    Adding noise to line zero alone must not move the whole field's
    residual, which is what the circular-mean offset buys.
    """
    lines = 120
    index = np.arange(lines)
    clean = np.tile((index * np.pi + 0.3) % (2.0 * np.pi), (4, 1))
    nudged = clean.copy()
    nudged[:, 0] += math.radians(20.0)
    quiet = bi.per_line(np.full((4, lines), 40.0), clean)
    loud = bi.per_line(np.full((4, lines), 40.0), nudged)
    assert abs(loud["phase_residual_sd_deg"] - quiet["phase_residual_sd_deg"]) < 2.0
    assert abs(loud["rotation_departure_deg_per_line"]) < 0.01


def test_a_planted_per_line_departure_survives_the_model():
    lines = 64
    drift = np.linspace(-1.0, 1.0, lines)
    amplitude = cc.BURST_IRE_PEAK_TO_PEAK + np.tile(drift, (4, 1))
    out = bi.per_line(amplitude)
    assert out["amplitude_residual_rms"] > 0.5
    assert np.allclose(np.squeeze(out["level"]), cc.BURST_IRE_PEAK_TO_PEAK)


def test_the_field_constant_calls_white_line_noise_consistent():
    rng = np.random.default_rng(0)
    values = cc.BURST_IRE_PEAK_TO_PEAK + rng.normal(0.0, 0.5, (26, 241))
    out = bi.field_constant(values)
    assert out["consistent"]
    assert 0.5 < out["excess"] < 2.0
    assert abs(out["lag_one"]) < 3.0 * out["white_bar"]
    # white lines predict themselves, so the inflation must be about one
    assert math.isclose(out["correlation_inflation"], 1.0, abs_tol=0.05)
    assert math.isclose(out["correlated_prediction_sd"],
                        out["white_prediction_sd"], rel_tol=0.05)


def test_correlated_lines_raise_the_bar_the_field_scatter_is_judged_against():
    """A correlated series averages more slowly, and the verdict must know it.

    The same field-to-field scatter is inconsistent against white lines and
    consistent against correlated ones, so the prediction has to carry the
    measured correlation or it convicts the specification of the lines'
    behaviour.
    """
    rng = np.random.default_rng(1)
    lines, fields = 241, 26
    white = rng.normal(0.0, 0.5, (fields, lines))
    smooth = np.cumsum(rng.normal(0.0, 0.5, (fields, lines)), axis=1)
    smooth = 0.5 * smooth / smooth.std()
    plain = bi.field_constant(cc.BURST_IRE_PEAK_TO_PEAK + white)
    rough = bi.field_constant(cc.BURST_IRE_PEAK_TO_PEAK + smooth)
    assert rough["lag_one"] > plain["lag_one"] + 0.3
    assert rough["correlation_inflation"] > plain["correlation_inflation"]
    assert rough["correlated_prediction_sd"] > rough["white_prediction_sd"]
    assert rough["excess"] > rough["excess_over_correlated"]


def test_the_verdict_bar_comes_from_the_field_count_not_from_a_constant():
    rng = np.random.default_rng(2)
    few = bi.field_constant(40.0 + rng.normal(0.0, 0.5, (5, 241)))
    many = bi.field_constant(40.0 + rng.normal(0.0, 0.5, (60, 241)))
    assert few["verdict_bar"] > many["verdict_bar"]
    assert many["verdict_bar"] > 1.0


def test_a_planted_per_field_term_is_refused_as_inconsistent():
    """The two readings of one specification then disagree, which is the point."""
    rng = np.random.default_rng(0)
    values = (cc.BURST_IRE_PEAK_TO_PEAK + rng.normal(0.0, 0.5, (26, 241))
              + rng.normal(0.0, 0.10, (26, 1)))
    out = bi.field_constant(values)
    assert not out["consistent"]
    assert out["excess"] > 2.0
    assert out["field_complex"]["is_complex"]


def test_the_field_constant_refuses_a_single_field():
    with pytest.raises(ValueError):
        bi.field_constant(np.ones(241))
    with pytest.raises(ValueError):
        bi.field_constant(np.ones((1, 241)))
