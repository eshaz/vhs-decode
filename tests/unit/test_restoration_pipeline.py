"""The order of the restoration, and the one reversal that destroys it.

Ethan's ordering: the reference is generated at the RF sample rate, the
model is applied while the signal is still a carrier, the demodulation IS
the downsample, and only then is it decimated to 4 fsc. Decimating the
carrier first was measured at thousands of times the output floor.
"""

import numpy as np
import pytest

from vhsdecode.models import output_limit as ol
from vhsdecode.models import restoration_pipeline as rp


def _planted(rate=40e6, n=1 << 16, deviation=0.5e6):
    """An FM carrier holding a whole number of cycles, so the transform-
    domain operators have no wrap-around of their own."""
    t = np.arange(n) / rate
    carrier = round(3.9e6 * n / rate) * rate / n
    modulation = round(60e3 * n / rate) * rate / n
    signal = np.cos(2 * np.pi * carrier * t
                    + (deviation / modulation) * np.sin(2 * np.pi * modulation * t))
    truth = deviation * np.cos(2 * np.pi * modulation * t)
    return signal, truth, rate


def test_the_pipeline_runs_from_rf_to_the_output_grid():
    signal, _truth, rate = _planted()
    out = rp.rf_to_output(signal, rate)
    assert out["output_rate_hz"] == pytest.approx(ol.FOUR_FSC_HZ)
    # the output is baseband at 4 fsc, so far fewer samples than the RF
    assert out["output"].size < signal.size
    expected = (signal.size - 2 * out["trimmed_each_end"]) * ol.FOUR_FSC_HZ / rate
    assert out["output"].size == pytest.approx(expected, rel=1e-3)
    assert out["profile"]["bits"] == 10


def test_the_model_is_applied_while_the_signal_is_still_a_carrier():
    signal, _truth, rate = _planted()
    seen = {}

    def model(rf):
        seen["rate_looks_like_rf"] = rf.size == signal.size
        return rf

    out = rp.rf_to_output(signal, rate, model=model)
    assert seen["rate_looks_like_rf"]
    assert out["steps"][0] == "model applied at the RF rate"


def test_decimating_before_demodulating_is_refused_by_the_order_check():
    right = ["demodulated: the inverse Hilbert with the time differential",
             "decimated to 14.318182 MHz"]
    wrong = ["decimated to 14.318182 MHz",
             "demodulated: the inverse Hilbert with the time differential"]
    assert rp.order_is_respected(right)["respected"] is True
    assert rp.order_is_respected(wrong)["respected"] is False
    # a list that names neither cannot be judged, and says so
    assert rp.order_is_respected(["something else"])["respected"] is None


def test_the_baseband_decimation_keeps_the_band_and_trims_the_edges():
    rate, n = 40e6, 1 << 14
    t = np.arange(n) / rate
    tone = np.cos(2 * np.pi * (round(1e6 * n / rate) * rate / n) * t)
    out = rp.decimate_baseband(tone, rate, 4e6, guard=0.125)
    assert out["sample_rate_hz"] == 4e6
    assert out["trimmed_each_end"] == n // 8
    kept = n - 2 * (n // 8)
    assert out["samples"].size == pytest.approx(kept * 4e6 / rate, rel=1e-3)
    # a 1 MHz tone survives a 4 MSps grid with its amplitude
    assert out["samples"].std() == pytest.approx(tone.std(), rel=0.05)
    with pytest.raises(ValueError, match="decimates"):
        rp.decimate_baseband(tone, rate, 80e6)


def _recovered_error(n):
    """The end-to-end error on a planted carrier of `n` RF samples."""
    signal, truth, rate = _planted(n=n)
    out = rp.rf_to_output(signal, rate)
    trim = out["trimmed_each_end"]
    reference = truth[trim:signal.size - trim]
    count = out["output"].size
    spectrum = np.fft.rfft(reference)[:count // 2 + 1] * (count / reference.size)
    reference_at_output = np.fft.irfft(spectrum, n=count)
    recovered = out["output"] - out["output"].mean()
    error = np.sqrt(np.mean((recovered - reference_at_output) ** 2))
    return {
        "ire": error / 1e6 * 140.0,
        "floors": error / 1e6 * 140.0 / out["profile"]["quantisation_noise_ire"],
        "relative": error / np.sqrt(np.mean(reference_at_output ** 2)),
        "correlation": float(np.corrcoef(recovered, reference_at_output)[0, 1]),
    }


def test_the_recovered_baseband_tracks_the_planted_modulation():
    out = _recovered_error(1 << 16)
    assert out["correlation"] > 0.99999      # measured 1.000000
    assert out["relative"] < 0.01            # measured 0.0061
    assert out["floors"] < 8.0               # measured 6.58


def test_what_is_left_is_an_edge_effect_and_dilutes_with_the_record():
    """The interior is exact; the whole residual is the transform domain's
    wrap-around at the two ends, so it becomes a smaller fraction of a
    longer record. Measured: 9.30, 6.58, 2.31 and 1.16 floors at 2^15,
    2^16, 2^17 and 2^18 samples, with the correlation 1.000000 at every
    length."""
    short = _recovered_error(1 << 15)
    long = _recovered_error(1 << 17)
    assert short["correlation"] > 0.99999 and long["correlation"] > 0.99999
    assert long["floors"] < 0.5 * short["floors"]
    assert long["relative"] < 0.5 * short["relative"]
