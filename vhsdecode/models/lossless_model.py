"""MODEL-BASED LOSSLESS COMPRESSION of the RF capture, against FLAC and FFV1.

Ethan's theory: the capture's information is the MODELLED part - the
chain's deterministic response to the known sync and burst structure, the
luminance FM carrier, the colour-under - plus a truly random remainder. A
lossless coder that subtracts the model and entropy-codes the remainder
should beat a generic coder by the model's information. This module
measures what the model, as the arc has it today, is worth in bits per
sample, and it measures the two generic coders on the same windows so the
comparison is between numbers and not between claims.

THE PREDICTION, WRITTEN BEFORE THE MEASUREMENT. `interference_distributions
.carrier_fraction` read 92.9 per cent carrier power and a carrier-to-noise
ratio of 11.2 dB off a single moment of the off-air capture. A constant-
envelope carrier sampled at random phase is arcsine distributed and the
rest is Gaussian, so the raw marginal is the arcsine convolved with the
Gaussian and the remainder after a perfect carrier predictor is the
Gaussian alone. Both entropies are discretised at one 8-bit code and both
scale as log2 of the rms, so their difference depends on the carrier share
only (`predicted_carrier_saving_bits`, evaluated 2026-09-05 before any
predictor ran):

    carrier share 0.929 (C/N 11.2 dB)   saving 1.69 bits per sample
    carrier share 0.880 (C/N  8.7 dB)   saving 1.38 bits per sample
    carrier share 0.966 (C/N 14.5 dB)   saving 2.08 bits per sample
    carrier share 0.916 (C/N 10.4 dB)   saving 1.60 bits per sample

The second row is the 75-bars playback capture measured on its own
kurtosis, the third countdown at ten seconds, the fourth home at seventeen
seconds (0.1 s windows, this module's `read_window`). That is the ceiling
of the carrier stage's zeroth-order saving; the measured saving is below
it by the predictor's own error, which at these carrier-to-noise ratios is
the noise the predictor cannot average out of its window plus its lag at
every luminance transient. The remainder's floor is the Gaussian at the
noise rms: 5.13 bits on 75-bars (noise rms 8.5 codes), 5.25 on countdown
(9.2), 5.65 on home (12.2).

The colour-under's worth is bounded by its share of what the carrier stage
leaves: the 0.3-1.2 MHz band holds 8.3 per cent of the AC power on 75-bars,
0.7 on countdown, 0.55 on home, so removing it entirely would take the
remainder down by 2.3 dB (0.38 bits), 0.8 dB (0.14 bits) and 0.3 dB (0.05
bits) respectively. The line stage cannot beat the noise: a previous-line
prediction of noise adds the previous line's noise to this line's, so on
samples the carrier stage already predicts to the floor it can only lose,
and its worth is confined to the samples the carrier stage mispredicts -
the transients - on static content, and to blanking elsewhere.

WHAT A GENERIC CODER SEES. FLAC is linear prediction of order up to 12 at
its highest level (32 with --lax), fitted per 4096-sample block, so on a
signal with ten samples per carrier cycle it already predicts the carrier
well: the whole question is whether tracking the carrier's INSTANTANEOUS
frequency beats a fixed spectrum. Kolmogorov's formula says a linear
predictor's error is the geometric mean of the spectrum, and an FM carrier
spreads its power over the deviation band, so the linear predictor cannot
see the carrier as the narrowband thing it locally is. The gap between the
fixed-spectrum bound and the tracking predictor is the FM model's
information, and it is the number this module puts beside FLAC's.

DECODABILITY. Every predictor here is CAUSAL - it uses samples the decoder
has already reconstructed - or it uses parameters that are counted as side
information in the ladder. The carrier stage's instantaneous frequency is
read from a delayed analytic band-pass of the past; the line stage's
per-line gains are fitted on the current line but quantised and counted.
The code length is the IDEAL adaptive arithmetic coder's, in closed form,
so no coder is implemented and none is needed: the Krichevsky-Trofimov
mixture length is exact to within two bits of what such a coder writes.

EVERY QUANTITY IS MEASURED OR CITED. Carrier band, line period, sync width
and blanking geometry come from the decoder's own parameter sets through
`noise_budget.format_figures` and `lddecode.core.SysParams_NTSC`; the
colour-under carrier is `colour_under.COLOUR_UNDER_LINE_MULTIPLE` times the
LINE RATE MEASURED on the window; the line period in samples is measured by
`noise_budget.sample_rate_from_line_period`; predictor window lengths and
smoothings are chosen by the residual entropy on a calibration prefix of
the window, and the choice is reported.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import fftconvolve, firwin
from scipy.special import erf, gammaln

from vhsdecode.models import colour_under
from vhsdecode.models import interference_distributions
from vhsdecode.models import noise_budget


# The capture headers carry the rate in kilohertz where they mean megahertz
# (40 kHz for 40 MSps, 50 kHz for 50 MSps): the same rule as
# tools/ringing_measure/modulation_noise_measure.py, confirmed for the two
# hour captures by a least-squares fit of file position against field index
# (40.0053 +/- 0.014 MHz) and for the test patterns by their sample count
# (24,000,512 samples over 0.48 s).
RATE_IS_KILOHERTZ_BELOW_HZ = 1e5

# The 8-bit codes reach libsndfile as int16 left-shifted by eight; ffmpeg's
# pcm_s16le conversion does the same. Both were checked on every capture
# used here (all values multiples of 256).
CODE_SCALE = 256

LN2 = float(np.log(2.0))


# --------------------------------------------------------------------------
# reading the captures
# --------------------------------------------------------------------------

def capture_rate_hz(path: str) -> float:
    import soundfile as sf
    header = float(sf.info(path).samplerate)
    return header * 1000.0 if header < RATE_IS_KILOHERTZ_BELOW_HZ else header


def read_window(path: str, position_s: float, seconds: float
                ) -> Tuple[np.ndarray, float]:
    """A window of the capture as its 8-bit codes (int32), and its rate.

    soundfile's seek where the file carries a frame count (the test
    patterns); ffmpeg's `-ss` where it does not (the streamed two-hour
    captures), with seconds counted at the header's nominal rate, which
    tools/ringing_measure/content_independence.py verified sample-exact
    against a sequential read.
    """
    import soundfile as sf

    rate = capture_rate_hz(path)
    info = sf.info(path)
    start = int(round(position_s * rate))
    count = int(round(seconds * rate))
    data = None
    if info.frames < (1 << 62):
        try:
            with sf.SoundFile(path) as handle:
                handle.seek(start)
                data = handle.read(count, dtype="int16", always_2d=False)
            data = np.asarray(data).ravel()
        except (sf.LibsndfileError, RuntimeError):
            data = None
    if data is None:
        header_rate = float(info.samplerate)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-ss", repr(start / header_rate), "-i", path,
                   "-t", repr(count / header_rate),
                   "-c:a", "pcm_s16le", "-f", "s16le", "-"]
        raw = subprocess.run(command, capture_output=True, check=True).stdout
        data = np.frombuffer(raw, "<i2")[:count]
    data = np.asarray(data, dtype=np.int32)
    if data.size and np.all(data % CODE_SCALE == 0):
        data = data // CODE_SCALE
    return data, rate


# --------------------------------------------------------------------------
# entropies and the ideal code length
# --------------------------------------------------------------------------

def zeroth_order_entropy_bits(values) -> float:
    """The empirical entropy of the integer codes: the bits per sample an
    ideal memoryless coder would spend on this window."""
    codes = np.asarray(values).ravel().astype(np.int64)
    if codes.size == 0:
        return float("nan")
    counts = np.bincount(codes - codes.min())
    probability = counts[counts > 0] / codes.size
    return float(-(probability * np.log2(probability)).sum())


def linear_prediction_residual(values, order: int,
                               coefficient_precision_bits: int = 15
                               ) -> np.ndarray:
    """The integer residual of an order-`order` linear predictor fitted by
    least squares on the window, its coefficients quantised to the
    precision FLAC allows (15 bits), so the residual is what a transmitted
    predictor leaves and not what an unattainable one would."""
    codes = np.asarray(values, dtype=np.float64).ravel()
    order = int(order)
    if order <= 0 or codes.size <= 2 * order:
        return np.asarray(values).ravel().astype(np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(codes[:-1], order)
    target = codes[order:]
    coefficients = np.linalg.lstsq(windows, target, rcond=None)[0]
    step = 2.0 ** -int(coefficient_precision_bits)
    coefficients = np.round(coefficients / step) * step
    prediction = np.rint(windows @ coefficients)
    residual = np.asarray(values).ravel().astype(np.int64).copy()
    residual[order:] -= prediction.astype(np.int64)
    return residual


def entropy_bits(samples, order: int = 32) -> Dict[str, float]:
    """Zeroth-order entropy, and the conditional entropy under a linear
    predictor of the stated order (32, FLAC's --lax maximum; 12 is what its
    highest standard level uses, reported alongside)."""
    codes = np.asarray(samples).ravel().astype(np.int64)
    result = {
        "zeroth_order": zeroth_order_entropy_bits(codes),
        "linear_prediction": zeroth_order_entropy_bits(
            linear_prediction_residual(codes, order)),
        "order": int(order),
        "linear_prediction_order_12": zeroth_order_entropy_bits(
            linear_prediction_residual(codes, 12)),
        "samples": int(codes.size),
    }
    result["prediction_gain_bits"] = (result["zeroth_order"]
                                      - result["linear_prediction"])
    return result


def code_length(residual, alphabet: Optional[Tuple[int, int]] = None
                ) -> Dict[str, float]:
    """THE IDEAL ADAPTIVE ARITHMETIC CODER'S LENGTH, in closed form.

    An arithmetic coder driven by the Krichevsky-Trofimov estimator over an
    alphabet of K symbols writes, for a sequence with symbol counts c_s,

        -log2 [ Gamma(K/2) / Gamma(N + K/2) * prod_s Gamma(c_s + 1/2) /
                Gamma(1/2) ]

    bits, to within two bits of termination. No coder is needed to know
    that number, so none is implemented. It exceeds the empirical entropy by
    the cost of learning the distribution, about (K - 1)/2 log2 N bits in
    total, which over a window of millions of samples is thousandths of a
    bit per sample. Given as both the total and the per-sample figure.
    """
    codes = np.asarray(residual).ravel().astype(np.int64)
    n = int(codes.size)
    if n == 0:
        return {"bits_total": float("nan"), "bits_per_sample": float("nan")}
    low, high = (int(codes.min()), int(codes.max())) if alphabet is None \
        else (int(alphabet[0]), int(alphabet[1]))
    size = high - low + 1
    counts = np.bincount(codes - low, minlength=size)
    log_probability = (gammaln(0.5 * size) - gammaln(n + 0.5 * size)
                       + np.sum(gammaln(counts + 0.5) - gammaln(0.5)))
    bits_total = float(-log_probability / LN2)
    entropy = zeroth_order_entropy_bits(codes)
    return {
        "bits_total": bits_total,
        "bits_per_sample": bits_total / n,
        "entropy_bits_per_sample": entropy,
        "adaptivity_cost_bits_per_sample": bits_total / n - entropy,
        "alphabet": int(size),
        "samples": n,
        "why": ("the Krichevsky-Trofimov mixture is what an ideal adaptive "
                "arithmetic coder writes, to within two bits; no coder is "
                "implemented because the bound is exact"),
    }


# --------------------------------------------------------------------------
# the prediction
# --------------------------------------------------------------------------

def _gaussian_code_entropy_bits(sigma: float) -> float:
    """Entropy of a Gaussian quantised to unit codes."""
    reach = int(np.ceil(8.0 * sigma)) + 2
    k = np.arange(-reach, reach + 1, dtype=np.float64)
    upper = 0.5 * (1.0 + erf((k + 0.5) / (sigma * np.sqrt(2.0))))
    lower = 0.5 * (1.0 + erf((k - 0.5) / (sigma * np.sqrt(2.0))))
    p = upper - lower
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def predicted_carrier_saving_bits(carrier_share: float,
                                  total_rms: float = 32.0,
                                  phases: int = 4096) -> Dict[str, float]:
    """The bits per sample a perfect carrier predictor saves, from the
    carrier share alone.

    The raw marginal is the arcsine of a carrier of rms sqrt(share) times
    the total, convolved with a Gaussian of the rest, both quantised to
    unit codes; the remainder is the Gaussian. `total_rms` only fixes the
    quantisation and the result is independent of it once the noise spans
    a few codes (checked: 1.690, 1.691, 1.690 bits at 10, 20, 40 codes for
    a share of 0.929).
    """
    share = float(carrier_share)
    total_rms = float(total_rms)
    carrier_rms = total_rms * np.sqrt(share)
    noise_rms = total_rms * np.sqrt(max(1.0 - share, 1e-12))
    amplitude = carrier_rms * np.sqrt(2.0)
    phase = (np.arange(int(phases)) + 0.5) / float(phases) * 2.0 * np.pi
    carrier = amplitude * np.cos(phase)
    reach = int(np.ceil(amplitude + 8.0 * noise_rms)) + 2
    k = np.arange(-reach, reach + 1, dtype=np.float64)[:, None]
    scale = noise_rms * np.sqrt(2.0)
    upper = 0.5 * (1.0 + erf((k + 0.5 - carrier[None, :]) / scale))
    lower = 0.5 * (1.0 + erf((k - 0.5 - carrier[None, :]) / scale))
    p = (upper - lower).mean(axis=1)
    p = p[p > 0] / p.sum()
    raw_bits = float(-(p * np.log2(p)).sum())
    remainder_bits = _gaussian_code_entropy_bits(noise_rms)
    return {
        "carrier_share": share,
        "carrier_to_noise_db": float(10.0 * np.log10(share / max(1.0 - share, 1e-12))),
        "raw_bits": raw_bits,
        "remainder_bits": remainder_bits,
        "saving_bits": raw_bits - remainder_bits,
        "noise_rms": float(noise_rms),
        "why": ("fourth cumulants gave the share; arcsine convolved with "
                "Gaussian is the raw marginal and the Gaussian alone is the "
                "remainder; both scale as log2 rms so only the share matters"),
    }


# --------------------------------------------------------------------------
# the analytic band-pass, and the causal frequency estimate
# --------------------------------------------------------------------------

def analytic_kernel(half_length: int, sample_rate_hz: float,
                    band_hz: Tuple[float, float]) -> np.ndarray:
    """A complex FIR whose output is the analytic signal of one band: a
    windowed low-pass prototype of the band's half-width, shifted to the
    band's centre and doubled, so a cosine in the band comes out as a unit
    complex exponential. Length 2 * half_length + 1, group delay
    half_length."""
    rate = float(sample_rate_hz)
    low, high = float(band_hz[0]), float(band_hz[1])
    centre = 0.5 * (low + high)
    half_width = 0.5 * (high - low)
    taps = 2 * int(half_length) + 1
    prototype = firwin(taps, half_width, fs=rate)
    k = np.arange(taps) - int(half_length)
    return 2.0 * prototype * np.exp(2j * np.pi * centre * k / rate)


def analytic_response(kernel: np.ndarray, frequency_hz, sample_rate_hz: float
                      ) -> np.ndarray:
    """The kernel's complex response at the given frequencies."""
    k = np.arange(kernel.size)
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    phase = -2j * np.pi * np.outer(f, k) / float(sample_rate_hz)
    return np.exp(phase) @ kernel


def hilbert_half_length(sample_rate_hz: float, band_hz: Tuple[float, float],
                        tolerance: float = 0.01,
                        candidates: Sequence[int] = (7, 11, 15, 23, 31, 47,
                                                     63, 95, 127, 191, 255)
                        ) -> int:
    """The shortest analytic band-pass whose magnitude is within
    `tolerance` of one across the band and whose image rejection is within
    `tolerance` of zero. One per cent is chosen because a one per cent
    quadrature error is a phase error of 0.01 radian, twenty times below
    the per-sample phase noise at these carrier-to-noise ratios."""
    rate = float(sample_rate_hz)
    low, high = float(band_hz[0]), float(band_hz[1])
    grid = np.linspace(low, high, 64)
    for half in candidates:
        kernel = analytic_kernel(half, rate, band_hz)
        magnitude = np.abs(analytic_response(kernel, grid, rate))
        image = np.abs(analytic_response(kernel, -grid, rate))
        if (np.all(np.abs(magnitude - 1.0) <= tolerance)
                and np.all(image <= tolerance)):
            return int(half)
    return int(candidates[-1])


def causal_frequency_estimate(x, sample_rate_hz: float,
                              band_hz: Tuple[float, float],
                              smoothing_samples: int,
                              half_length: Optional[int] = None
                              ) -> Dict[str, object]:
    """The phase advance per sample of the band's analytic signal, read
    from the PAST only: omega[n] depends on x[:n]. The analytic sample at
    time m is available at time m + half_length (the kernel's group delay),
    and the advance is the angle of the sum of `smoothing_samples`
    consecutive products z[m] conj(z[m-1]), so the estimate lags by the
    delay plus half the smoothing. Radians per sample."""
    values = np.asarray(x, dtype=np.float64).ravel()
    n = values.size
    rate = float(sample_rate_hz)
    half = int(half_length if half_length is not None
               else hilbert_half_length(rate, band_hz))
    kernel = analytic_kernel(half, rate, band_hz)
    # full convolution index m holds the analytic sample centred at m - half
    # and uses x[m - 2 half .. m]
    analytic = fftconvolve(values, kernel, mode="full")[:n]
    advance = analytic[1:] * np.conj(analytic[:-1])      # index j uses x[:j+2]
    smoothing = max(int(smoothing_samples), 1)
    cumulative = np.cumsum(advance)
    summed = cumulative.copy()
    summed[smoothing:] -= cumulative[:-smoothing]        # index j uses x[:j+2]
    centre = 2.0 * np.pi * 0.5 * (band_hz[0] + band_hz[1]) / rate
    omega = np.full(n, centre, dtype=np.float64)
    # to predict x[n] use summed[n - 2], which used x[:n]
    lead = 2 + smoothing + 2 * half
    if n > lead:
        omega[lead:] = np.angle(summed[lead - 2:n - 2])
    return {"omega": omega, "half_length": half, "smoothing": smoothing,
            "lag_samples": half + 1 + 0.5 * (smoothing - 1),
            "analytic": analytic}


# --------------------------------------------------------------------------
# the sinusoid extrapolation predictor
# --------------------------------------------------------------------------

def sinusoid_predictor_bank(omegas, length: int, constant: bool
                            ) -> np.ndarray:
    """Least-squares one-step extrapolation coefficients for a sinusoid of
    each frequency, fitted over the `length` past samples, with or without
    a constant term. Row g, column j multiplies x[n - length + j]."""
    omegas = np.asarray(omegas, dtype=np.float64).ravel()
    lags = np.arange(int(length), 0, -1, dtype=np.float64)   # L .. 1
    cos = np.cos(np.outer(omegas, lags))
    sin = np.sin(np.outer(omegas, lags))
    columns = [cos, sin] + ([np.ones_like(cos)] if constant else [])
    basis = np.stack(columns, axis=-1)                     # (G, L, B)
    gram = np.einsum("glb,glc->gbc", basis, basis)
    # the prediction at lag zero is A cos 0 + B sin 0 + C = A + C
    target = np.zeros(basis.shape[-1])
    target[0] = 1.0
    if constant:
        target[-1] = 1.0
    weights = np.linalg.solve(gram, np.broadcast_to(
        target, (omegas.size, target.size)).copy()[..., None])[..., 0]
    return np.einsum("gb,glb->gl", weights, basis)


def sinusoid_extrapolation(x, omega, length: int,
                           omega_range: Tuple[float, float],
                           grid_points: Optional[int] = None,
                           chunk: int = 1 << 18) -> np.ndarray:
    """x_hat[n] from x[n - length .. n - 1] at the per-sample frequency
    omega[n], quantised to a grid. The grid step is set so that the phase
    error at the fit's centroid, half the window back, stays below 0.01
    radian - the same level `hilbert_half_length` holds the quadrature to.
    The first `length` samples are predicted as zero."""
    values = np.asarray(x, dtype=np.float64).ravel()
    n = values.size
    length = int(length)
    prediction = np.zeros(n, dtype=np.float64)
    if length <= 0 or n <= length:
        return prediction
    low, high = float(omega_range[0]), float(omega_range[1])
    if grid_points is None:
        step_wanted = 0.01 / max(0.5 * length, 1.0) * 2.0
        grid_points = max(int(np.ceil((high - low) / step_wanted)) + 1, 1)
    grid_points = int(grid_points)
    grid = np.linspace(low, high, grid_points) if grid_points > 1 \
        else np.array([0.5 * (low + high)])
    # the constant term is well conditioned only over at least one cycle
    constant = bool(length * low >= 2.0 * np.pi)
    bank = sinusoid_predictor_bank(grid, length, constant).astype(np.float32)
    omega = np.asarray(omega, dtype=np.float64).ravel()
    if grid_points > 1:
        index = np.rint((omega - low) / (high - low) * (grid_points - 1))
        index = np.clip(index, 0, grid_points - 1).astype(np.intp)
    else:
        index = np.zeros(n, dtype=np.intp)
    windows = np.lib.stride_tricks.sliding_window_view(
        values.astype(np.float32), length)            # row m = x[m .. m+L-1]
    for start in range(length, n, int(chunk)):
        stop = min(start + int(chunk), n)
        rows = windows[start - length:stop - length]
        prediction[start:stop] = np.einsum(
            "nk,nk->n", bank[index[start:stop]], rows, dtype=np.float64)
    return prediction


def carrier_band_rad(sample_rate_hz: float, figures: Dict[str, float]
                     ) -> Tuple[float, float]:
    """Where the luminance carrier can be, in radians per sample: from the
    sync tip to peak white, widened by the format's deviation figure to
    cover pre-emphasised overshoot (JVC VTG82063 section 3 via
    `capture_profile`)."""
    rate = float(sample_rate_hz)
    low = float(figures["tip_hz"]) - float(figures["deviation_hz"])
    high = float(figures["white_hz"]) + float(figures["deviation_hz"])
    return (2.0 * np.pi * low / rate, 2.0 * np.pi * high / rate)


def carrier_stage(x, sample_rate_hz: float,
                  band_hz: Tuple[float, float],
                  omega_range: Tuple[float, float],
                  lengths: Sequence[int] = (8, 16, 24, 32, 48, 64, 96),
                  smoothings: Sequence[int] = (4, 8, 16, 32, 64),
                  fixed_omega: Optional[float] = None,
                  calibration_samples: Optional[int] = None
                  ) -> Dict[str, object]:
    """ONE CARRIER PREDICTOR STAGE: the analytic signal's phase advance and
    envelope, read from the past, extrapolated one sample.

    The window length and the frequency smoothing are chosen by the
    residual's zeroth-order entropy on a calibration prefix of the window
    (the first eighth, at least half a million samples where the window
    allows), and the choice of "no prediction" is on the table so the
    stage can never make the ladder worse. With `fixed_omega` the
    frequency is not tracked - the colour-under at forty times the line
    rate is run that way as well as tracked, and the better is kept.
    """
    codes = np.asarray(x).ravel().astype(np.int64)
    n = codes.size
    calibration = int(calibration_samples if calibration_samples
                      else min(n, max(n // 8, 1 << 19)))
    prefix = codes[:calibration]
    before = zeroth_order_entropy_bits(prefix)
    best = {"entropy": before, "length": 0, "smoothing": 0, "tracked": False}
    trials: List[Tuple[int, int, bool, float]] = []
    modes: List[Tuple[Optional[int], bool]] = []
    if fixed_omega is not None:
        modes.append((0, False))
    modes.extend((int(q), True) for q in smoothings)
    for smoothing, tracked in modes:
        if tracked:
            estimate = causal_frequency_estimate(prefix, sample_rate_hz,
                                                 band_hz, smoothing)
            omega = np.clip(estimate["omega"], omega_range[0], omega_range[1])
            span = omega_range
        else:
            omega = np.full(prefix.size, float(fixed_omega))
            span = (float(fixed_omega), float(fixed_omega))
        for length in lengths:
            prediction = sinusoid_extrapolation(prefix, omega, length, span)
            residual = prefix - np.rint(prediction).astype(np.int64)
            entropy = zeroth_order_entropy_bits(residual)
            trials.append((int(length), int(smoothing), tracked, entropy))
            if entropy < best["entropy"]:
                best = {"entropy": entropy, "length": int(length),
                        "smoothing": int(smoothing), "tracked": tracked}
    if best["length"] == 0:
        prediction = np.zeros(n)
        omega_full = None
    elif best["tracked"]:
        estimate = causal_frequency_estimate(codes, sample_rate_hz, band_hz,
                                             best["smoothing"])
        omega_full = np.clip(estimate["omega"], omega_range[0], omega_range[1])
        prediction = sinusoid_extrapolation(codes, omega_full, best["length"],
                                            omega_range)
    else:
        omega_full = np.full(n, float(fixed_omega))
        prediction = sinusoid_extrapolation(
            codes, omega_full, best["length"],
            (float(fixed_omega), float(fixed_omega)))
    residual = codes - np.rint(prediction).astype(np.int64)
    return {
        "residual": residual,
        "prediction": prediction,
        "omega": omega_full,
        "entropy_before": zeroth_order_entropy_bits(codes),
        "entropy_after": zeroth_order_entropy_bits(residual),
        "length": best["length"],
        "smoothing": best["smoothing"],
        "tracked": best["tracked"],
        "calibration_samples": calibration,
        "trials": trials,
        # three small integers describe the choice; negligible per sample
        "side_information_bits": 3 * 8,
    }


# --------------------------------------------------------------------------
# the sync structure: line starts, blanking geometry
# --------------------------------------------------------------------------

def blanking_geometry(tape_speed: int = 0) -> Dict[str, float]:
    """The line's blanking interval from the decoder's own NTSC VHS
    parameters: front porch, sync width, and the start of active video
    measured from the sync's leading edge. Seconds."""
    from lddecode.core import SysParams_NTSC
    from vhsdecode.format_defs.vhs import get_sysparams_ntsc_vhs
    system = get_sysparams_ntsc_vhs(SysParams_NTSC, int(tape_speed))
    return {
        "front_porch_s": float(system["frontPorchUS"]) * 1e-6,
        "sync_s": float(system["hsyncPulseUS"]) * 1e-6,
        "active_start_s": float(system["activeVideoUS"][0]) * 1e-6,
        "line_period_s": float(system["line_period"]) * 1e-6,
    }


def sync_starts(x, sample_rate_hz: float, figures: Dict[str, float]
                ) -> np.ndarray:
    """Sample indices where the carrier drops to the sync tip for about a
    line sync's width: `noise_budget`'s tip mask and dwell finder on the
    band-limited analytic signal of the whole window. These positions are
    side information for the decoder and are costed as such."""
    codes = np.asarray(x, dtype=np.float64).ravel()
    analytic = noise_budget.analytic_signal(codes, sample_rate_hz,
                                            figures["fm_band_low_hz"],
                                            figures["fm_band_high_hz"])
    frequency = noise_budget.instantaneous_frequency(analytic, sample_rate_hz)
    mask = noise_budget.tip_mask(frequency, figures)
    dwells = noise_budget.carrier_dwells(mask, sample_rate_hz,
                                         figures["line_sync_s"])
    return np.array([start for start, _ in dwells], dtype=np.int64)


def line_period_samples(x, sample_rate_hz: float, figures: Dict[str, float]
                        ) -> Dict[str, float]:
    """The line period in samples, measured from the tip mask's
    periodicity by `noise_budget.sample_rate_from_line_period`, and the
    sample rate that period implies against the format's line period."""
    return noise_budget.sample_rate_from_line_period(
        np.asarray(x, dtype=np.float64), sample_rate_hz, figures)


def line_stage(x, starts, line_period: float, sample_rate_hz: float,
               bands_hz: Sequence[Tuple[float, float]],
               geometry: Dict[str, float],
               coefficient_bits: int = 12) -> Dict[str, object]:
    """THE LINE-PERIODIC STAGE: predict this line from the previous one.

    An FM carrier's waveform does not repeat line to line even on a static
    picture: the modulator's phase is continuous, so each line is the
    previous one ROTATED by the phase the carrier advanced over a line,
    plus whatever the time base moved. The right previous-line predictor
    is therefore a complex gain on the previous line's analytic signal in
    each band - luminance and colour-under separately, because the
    colour-under at an integer multiple of the line rate rotates by
    nothing while the luminance carrier rotates by a fraction of a cycle
    that the time-base error changes line to line. The gains are fitted by
    least squares per line and per region (blanking, active), quantised
    to `coefficient_bits`, and a region keeps its prediction only where
    the Gaussian coding gain 0.5 n log2(SSE before / SSE after) exceeds
    the bits the coefficients cost - which is what confines the stage to
    blanking on moving pictures and lets it take the whole line on static
    ones. Sync positions and coefficients are counted as side information.
    The analytic band-pass reaches `half_length` samples past the previous
    line's sample, which is still inside the decoder's past because a line
    is thousands of samples long.
    """
    codes = np.asarray(x).ravel().astype(np.int64)
    values = codes.astype(np.float64)
    n = values.size
    rate = float(sample_rate_hz)
    starts = np.asarray(starts, dtype=np.int64).ravel()
    period = float(line_period)
    residual = codes.copy()
    columns = []
    max_half = 0
    for band in bands_hz:
        half = hilbert_half_length(rate, band)
        max_half = max(max_half, half)
        kernel = analytic_kernel(half, rate, band)
        centred = fftconvolve(values, kernel, mode="full")[half:half + n]
        columns.append(centred.real)
        columns.append(centred.imag)
    regressors = np.stack(columns, axis=1)                  # (n, 2 * bands)
    front = int(round(geometry["front_porch_s"] * rate))
    active = int(round(geometry["active_start_s"] * rate))
    regions = [("blanking", -front, active), ("active", active,
                                              int(round(period)) - front)]
    scale = 2.0 ** (int(coefficient_bits) - 2)             # gains within +-2
    side_bits = 0.0
    kept = {name: 0 for name, _, _ in regions}
    seen = {name: 0 for name, _, _ in regions}
    lines_used = 0
    position_bits = float(np.ceil(np.log2(max(2.0 * period, 2.0))))
    for i in range(1, starts.size):
        delta = int(starts[i] - starts[i - 1])
        side_bits += position_bits
        # the equalising pulses halve the spacing, so anything nearer the
        # period than half of it is an ordinary line
        if abs(delta - period) > 0.25 * period or delta <= max_half:
            continue
        lines_used += 1
        for name, low, high in regions:
            lo = max(starts[i] + low, delta + max_half + 1, 0)
            hi = min(starts[i] + high, n)
            if hi - lo < 2 * regressors.shape[1] + 2:
                continue
            seen[name] += 1
            side_bits += 1.0                                # the keep flag
            design = regressors[lo - delta:hi - delta]
            target = values[lo:hi]
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
            coefficients = np.clip(np.rint(coefficients * scale) / scale,
                                   -2.0, 2.0)
            prediction = np.rint(design @ coefficients)
            before = float(np.sum(target ** 2))
            after = float(np.sum((target - prediction) ** 2))
            cost = coefficient_bits * coefficients.size
            gain = 0.5 * (hi - lo) * np.log2(before / after) \
                if after > 0 and before > 0 else 0.0
            if gain > cost:
                residual[lo:hi] = codes[lo:hi] - prediction.astype(np.int64)
                side_bits += cost
                kept[name] += 1
    return {
        "residual": residual,
        "entropy_before": zeroth_order_entropy_bits(codes),
        "entropy_after": zeroth_order_entropy_bits(residual),
        "side_information_bits": side_bits,
        "side_information_bits_per_sample": side_bits / max(n, 1),
        "lines_used": lines_used,
        "regions_kept": kept,
        "regions_seen": seen,
        "half_length": max_half,
    }


# --------------------------------------------------------------------------
# the generic coders on the same window
# --------------------------------------------------------------------------

def flac_bits_per_sample(codes, header_rate_hz: int = 50000,
                         compression_level: float = 1.0) -> Dict[str, float]:
    """FLAC through soundfile (libsndfile), 8-bit subtype, highest level,
    on the window as written to a file; verified exact on read-back."""
    import soundfile as sf
    codes = np.asarray(codes).ravel().astype(np.int16)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "window.flac")
        sf.write(path, codes * CODE_SCALE, int(header_rate_hz),
                 subtype="PCM_S8", format="FLAC",
                 compression_level=float(compression_level))
        size = os.path.getsize(path)
        back = sf.read(path, dtype="int16")[0]
    exact = bool(np.array_equal(back, codes * CODE_SCALE))
    return {"bits_per_sample": 8.0 * size / codes.size, "bytes": int(size),
            "exact": exact, "samples": int(codes.size),
            "level": "libFLAC 8 (soundfile compression_level 1.0)"}


def ffv1_bits_per_sample(codes, width: int, ffmpeg: str = "ffmpeg"
                         ) -> Dict[str, float]:
    """FFV1 through ffmpeg on the window packed as a gray8 raster, one line
    per row at `width` samples, so its two-dimensional context sees the
    line above; level 3, the large context, the range coder. The 8-bit
    codes are offset by 128 into gray8, a bijection."""
    codes = np.asarray(codes).ravel().astype(np.int64)
    width = int(width)
    height = codes.size // width
    raster = (codes[:width * height] + 128)
    if raster.min() < 0 or raster.max() > 255:
        raise ValueError("codes outside the 8-bit range")
    raw = raster.astype(np.uint8).tobytes()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "window.mkv")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "gray8",
                   "-s", f"{width}x{height}", "-i", "-",
                   "-c:v", "ffv1", "-level", "3", "-context", "1",
                   "-coder", "1", "-f", "matroska", path]
        subprocess.run(command, input=raw, check=True)
        size = os.path.getsize(path)
    return {"bits_per_sample": 8.0 * size / (width * height),
            "bytes": int(size), "width": width, "height": height,
            "samples": int(width * height)}


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def ladder(codes, sample_rate_hz: float, figures: Dict[str, float],
           tape_speed: int = 0, starts: Optional[np.ndarray] = None,
           line_period: Optional[float] = None) -> Dict[str, object]:
    """Every stage alone on the raw codes and cumulatively on the previous
    stage's remainder, with the remainder's entropy, ideal code length and
    distribution. Returns the rows and the measurements they rest on."""
    codes = np.asarray(codes).ravel().astype(np.int64)
    rate = float(sample_rate_hz)
    geometry = blanking_geometry(tape_speed)
    if line_period is None:
        measured = line_period_samples(codes, rate, figures)
        line_period = float(measured["line_period_samples"])
    else:
        measured = {"line_period_samples": float(line_period)}
    if starts is None:
        starts = sync_starts(codes, rate, figures)
    line_rate_hz = rate / float(line_period)
    colour_under_hz = colour_under.COLOUR_UNDER_LINE_MULTIPLE * line_rate_hz
    luma_band = (figures["fm_band_low_hz"], figures["fm_band_high_hz"])
    omega_range = carrier_band_rad(rate, figures)
    chroma_half_width = colour_under_hz - float(figures.get(
        "colour_under_low_hz", colour_under_hz / 2.0))
    chroma_band = (colour_under_hz - chroma_half_width,
                   colour_under_hz + chroma_half_width)
    omega_cu = 2.0 * np.pi * colour_under_hz / rate

    stages = []

    def run_carrier(signal):
        return carrier_stage(signal, rate, luma_band, omega_range)

    def run_colour_under(signal):
        return carrier_stage(signal, rate, chroma_band,
                             (omega_cu / 2.0, omega_cu * 1.5),
                             lengths=(64, 128, 256, 512),
                             smoothings=(16, 64, 256),
                             fixed_omega=omega_cu)

    def run_line(signal):
        return line_stage(signal, starts, line_period, rate,
                          [luma_band, chroma_band], geometry)

    raw_entropy = zeroth_order_entropy_bits(codes)
    current = codes
    for name, runner in (("carrier", run_carrier),
                         ("colour_under", run_colour_under),
                         ("line", run_line)):
        alone = runner(codes)
        cumulative = runner(current)
        alone_side = alone.get("side_information_bits", 0.0) / codes.size
        cumulative_side = cumulative.get("side_information_bits", 0.0) / codes.size
        stages.append({
            "name": name,
            "alone_bits": alone["entropy_after"] + alone_side,
            "alone_saving": raw_entropy - alone["entropy_after"] - alone_side,
            "cumulative_bits": cumulative["entropy_after"] + cumulative_side,
            "cumulative_saving": (zeroth_order_entropy_bits(current)
                                  - cumulative["entropy_after"] - cumulative_side),
            "side_information_bits_per_sample": cumulative_side,
            "parameters": {k: cumulative[k] for k in ("length", "smoothing",
                                                       "tracked", "lines_used",
                                                       "regions_kept",
                                                       "regions_seen")
                           if k in cumulative},
        })
        current = cumulative["residual"]
    remainder = current
    return {
        "raw_bits": 8.0,
        "zeroth_order_bits": raw_entropy,
        "stages": stages,
        "remainder": remainder,
        "remainder_entropy": entropy_bits(remainder),
        "remainder_code_length": code_length(remainder),
        "remainder_distribution": interference_distributions.classify(
            remainder.astype(np.float64)),
        "line_period_samples": float(line_period),
        "line_rate_hz": line_rate_hz,
        "colour_under_hz": colour_under_hz,
        "sync_starts": int(starts.size),
        "measured_rate": measured,
    }
