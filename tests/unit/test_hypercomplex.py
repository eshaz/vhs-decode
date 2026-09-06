"""The hypercomplex Hilbert traversal: exact in, exact out, spec at zero."""

import numpy as np
import pytest

from vhsdecode.models import hypercomplex as hc
from vhsdecode.models import tesseract as T


def _cube(seed=0, bins=64):
    rng = np.random.default_rng(seed)
    axes = ("head", "half")
    values = rng.standard_normal((2, 2, bins)) + 1j * rng.standard_normal((2, 2, bins))
    return T.Cube(axes, values, np.full(values.shape, 1e-4))


def test_bode_relation_recovers_a_planted_minimum_phase_response():
    # a DISCRETE one-pole system 1 / (1 - a z^-1), |a| < 1, is minimum phase
    # on the sampled grid (a continuous-time pole sampled is not: its phase
    # at Nyquist is not zero, and no real sequence has that)
    n = 256
    f = np.fft.rfftfreq(2 * (n - 1))
    H = 1.0 / (1.0 - 0.9 * np.exp(-2j * np.pi * f))
    phase = hc.minimum_phase(np.log(np.abs(H)))
    assert np.max(np.abs(phase - np.unwrap(np.angle(H)))) < 1e-9
    # a pure delay is all excess phase
    delayed = np.log(np.abs(H)) + 1j * (np.unwrap(np.angle(H)) - 2 * np.pi * f * 3.0)
    ex = hc.excess_phase(delayed)
    assert np.allclose(ex["excess"], -2 * np.pi * f * 3.0, atol=1e-9)


def test_expansion_doubles_per_axis_and_the_inverse_is_exact():
    cube = _cube()
    components = hc.expand(cube)
    assert len(components) == 2 ** (cube.order + 1)
    back = hc.collapse_back(cube, components)
    assert np.abs(back - cube.values).max() < 1e-12


def test_a_specification_met_exactly_leaves_zero_everywhere():
    cube = _cube(1)
    spec = hc.expand(cube)
    related = hc.relate_to_spec(hc.expand(cube), spec)
    assert all(np.abs(v).max() < 1e-12 for v in related.values())


def test_deconvolve_returns_a_real_kernel_and_a_floor_residual():
    rng = np.random.default_rng(2)
    bins = 64
    axes = ("head", "tape")
    grid = np.linspace(0, 1, bins)
    head = 0.8 * np.exp(-2 * grid)
    values = np.zeros((2, 2, bins), dtype=complex)
    for v in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        values[v] = (head if v[0] == 0 else -head) + 0.001 * (
            rng.standard_normal(bins) + 1j * rng.standard_normal(bins))
    cube = T.Cube(axes, values, np.full(values.shape, 2e-6), bins=grid * 1e6)
    out = hc.deconvolve(cube, sample_rate_hz=4e6)
    assert out["kernel_is_real"]
    assert out["round_trip_error"] < 1e-12
    assert "head" in out["identified"] and "tape" not in out["identified"]
    assert 0.3 < out["residual_over_floor"] < 3.0


def test_nested_fourier_slices_equal_the_direct_transform():
    rng = np.random.default_rng(3)
    a, b, c = rng.standard_normal(8), rng.standard_normal(6), rng.standard_normal(5)
    separable = np.einsum("i,j,k->ijk", a, b, c)
    out = hc.fourier_slices(separable, axes=(0, 1, 2))
    assert out["nested_equals_direct"] < 1e-10
    assert out["separable_share"] > 0.999999
    mixed = separable + rng.standard_normal(separable.shape)
    assert hc.fourier_slices(mixed, axes=(0, 1, 2))["separable_share"] < 0.9


def test_causality_tells_a_zero_phase_shape_from_a_minimum_phase_one_and_a_delay():
    # a two-head cube on a 4 f_sc sub-band: head A carries a MINIMUM-PHASE
    # one-pole departure, head B the same magnitude with NO phase (a
    # symmetric, zero-phase shape) plus a pure delay
    rate = 14.318e6
    grid = np.fft.rfftfreq(4096, d=1.0 / rate)
    band = (grid > 0.2e6) & (grid < 4.0e6)
    f = grid[band]
    pole = 1.0 / (1.0 - 0.6 * np.exp(-2j * np.pi * grid / rate))
    minimum = np.log(np.abs(pole[band])) + 1j * np.unwrap(np.angle(pole[band]))
    delay_s = 100e-9
    zero_phase = np.log(np.abs(pole[band])) - 2j * np.pi * f * delay_s
    values = np.zeros((2, 2, f.size), dtype=complex)
    values[0, :, :] = minimum          # head A, both tapes
    values[1, :, :] = zero_phase       # head B, both tapes
    cube = T.Cube(("head", "tape"), values, np.full(values.shape, 1e-8), bins=f)
    read = hc.causality(cube, rate)
    # the head contrast is half the difference: half a minimum-phase term
    # minus half a zero-phase-plus-delay term, so its magnitude is zero and
    # its phase is pure excess
    head = read["head"]
    assert head["magnitude_rms"] < 1e-9
    assert head["excess_rms"] > 0.05 and head["minimum_phase_rms"] < 1e-9
    # the sign convention, on a pure delay: state 1 delayed by tau reads
    # -tau/2 on the contrast (half the difference, state 0 minus state 1)
    pure = np.zeros((2, 2, f.size), dtype=complex)
    pure[0, :, :] = minimum
    pure[1, :, :] = minimum - 2j * np.pi * f * delay_s
    pure_cube = T.Cube(("head", "tape"), pure, np.full(pure.shape, 1e-8), bins=f)
    assert abs(hc.causality(pure_cube, rate)["head"]["delay_s"] + 0.5 * delay_s) < 1e-12
    # the grand-mean-like parent read per head: a minimum-phase departure
    # has a share near one, a zero-phase one has a share at or below zero
    # the fill outside the band costs a true minimum-phase departure about
    # 0.06 rad rms of apparent excess (the cepstrum is global); the
    # zero-phase-plus-delay one carries far more than that
    a_full, _g, index = hc.on_full_grid(values[0, 0], f, rate)
    a = hc.excess_phase(a_full)
    assert np.sqrt(np.mean(a["excess"][index] ** 2)) < 0.1
    b_full, _g, index = hc.on_full_grid(values[1, 0], f, rate)
    b = hc.excess_phase(b_full)
    assert np.sqrt(np.mean(b["excess"][index] ** 2)) > 0.5


def test_a_sub_band_measurement_is_placed_on_its_full_grid_before_the_cepstrum():
    # a discrete minimum-phase response on a 4 f_sc rfft grid, observed only
    # on 0.2-4 MHz: placed back on its grid the Bode phase is recovered to
    # the fill's own error (0.15 rad max, 0.06 rms, measured); fed as if the
    # band were the whole axis it is wrong by more than the phase itself
    rate = 14.318e6
    grid = np.fft.rfftfreq(4096, d=1.0 / rate)
    H = 1.0 / (1.0 - 0.7 * np.exp(-2j * np.pi * grid / rate))
    band = (grid > 0.2e6) & (grid < 4.0e6)
    log_H = np.log(np.abs(H[band])) + 1j * np.unwrap(np.angle(H[band]))
    full, _g, index = hc.on_full_grid(log_H.real, grid[band], rate)
    placed = hc.minimum_phase(full)[index]
    assert np.max(np.abs(placed - log_H.imag)) < 0.2
    naive = hc.minimum_phase(log_H.real)
    placed_rms = np.sqrt(np.mean((placed - log_H.imag) ** 2))
    naive_rms = np.sqrt(np.mean((naive - log_H.imag) ** 2))
    assert naive_rms > 3.0 * placed_rms          # measured 0.276 against 0.062


def test_the_transform_is_its_own_inverse_on_every_axis_count():
    """Ethan: 'transform it by itself essentially, to get the 2d real
    signal back out ... like FFT and an IFFT, but 3D'. It is stronger than
    that: H H = -I, so the analysis operator IS the synthesis operator."""
    rng = np.random.default_rng(4)
    shape = (16, 32, 8)
    spectrum = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    keep = np.ones(shape, dtype=bool)
    for axis, size in enumerate(shape):          # no DC, no Nyquist
        f = np.fft.fftfreq(size)
        ok = (np.abs(f) > 0) & (np.abs(f) < 0.5)
        keep &= ok.reshape([-1 if a == axis else 1 for a in range(3)])
    signal = np.fft.ifftn(spectrum * keep).real
    for axes in ((0,), (1,), (0, 1), (0, 1, 2)):
        out = hc.self_inverse(signal, axes)
        assert out["sign_after_two"] == (-1.0) ** len(axes)
        assert out["error_after_two"] < 1e-12
        assert out["error_after_four"] < 1e-12
    # and every component of the full set returns by its own transform
    for bits in range(8):
        axes = tuple(a for a in range(3) if (bits >> a) & 1)
        component = hc.partial_hilbert(signal, axes)
        back = hc.partial_hilbert(component, axes)
        assert np.abs(back - (-1.0) ** len(axes) * signal).max() < 1e-12


def test_a_partial_model_is_refused_because_the_phase_lives_in_its_partners():
    cube = _cube(11)
    components = hc.expand(cube)
    magnitude_only = {key: value for key, value in components.items()
                      if key[1] == "real"}
    with pytest.raises(ValueError, match="unsubtracted"):
        hc.relate_to_spec(components, magnitude_only, strict=True)
    loose = hc.relate_to_spec(components, magnitude_only)
    assert all(np.abs(v).max() < 1e-12 for k, v in loose.items() if k[1] == "real")
    assert any(np.abs(v).max() > 1e-9 for k, v in loose.items() if k[1] == "hilbert")


def test_the_time_base_shift_is_exact_as_a_phase_and_approximate_as_a_sinc():
    """Ethan: 'the timebase is also sinc, which can be made complex'. The
    phase ramp is the shift's definition; a windowed sinc approaches it."""
    rng = np.random.default_rng(6)
    n = 1024
    spectrum = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    spectrum[np.abs(np.fft.fftfreq(n)) > 0.4] = 0
    x = np.fft.ifft(spectrum).real
    # shifting by a whole sample is a roll, exactly
    assert np.abs(hc.fractional_delay(x, 1.0) - np.roll(x, 1)).max() < 1e-10
    # shifting forward then back returns the signal
    assert np.abs(hc.fractional_delay(hc.fractional_delay(x, 0.37), -0.37) - x).max() < 1e-10
    # a windowed sinc of 15 taps sits about 45 dB below, not at zero error
    half = 7
    k = np.arange(-half, half + 1, dtype=float)
    kernel = np.sinc(k - 0.37) * np.hamming(15)
    kernel /= kernel.sum()
    padded = np.concatenate([x[-half:], x, x[:half]])
    sinc_shifted = np.array([np.dot(kernel[::-1], padded[i:i + 15]) for i in range(n)])
    error = sinc_shifted - hc.fractional_delay(x, 0.37)
    decibels = -20 * np.log10(np.sqrt(np.mean(error ** 2)) / np.std(x))
    assert 40.0 < decibels < 55.0


def test_a_time_base_error_is_corrected_by_rotation_without_resampling():
    """Ethan: 'we may not actually need to timebase correct that way'. True,
    provided the rotation follows the instantaneous frequency rather than
    the carrier: at full luma deviation the carrier version is ten times the
    ten-bit output floor and the instantaneous one is five times under it."""
    from vhsdecode.models import output_limit as ol

    rate, n = 40e6, 8192
    t = np.arange(n) / rate
    carrier, deviation = 3.9e6, 0.5e6
    signal = np.cos(2 * np.pi * carrier * t
                    + (deviation / 15734.0) * np.sin(2 * np.pi * 15734.0 * t))
    tau = 5e-9
    exact = hc.fractional_delay(signal, tau * rate)
    guard = slice(400, -400)
    scale = 40.0 / np.sqrt(np.mean(signal[guard] ** 2))
    floor = ol.output_profile(bits=10)["quantisation_noise_ire"]

    by_carrier = hc.time_base_by_rotation(signal, tau, rate, carrier_hz=carrier)
    by_instantaneous = hc.time_base_by_rotation(signal, tau, rate)
    carrier_error = np.sqrt(np.mean((by_carrier[guard] - exact[guard]) ** 2)) * scale
    instant_error = np.sqrt(np.mean((by_instantaneous[guard] - exact[guard]) ** 2)) * scale
    assert carrier_error > 5.0 * floor          # measured about 10x
    assert instant_error < 0.5 * floor          # measured about 0.15x
    # and the correction undoes itself, because the operator is its own inverse
    there_and_back = hc.time_base_by_rotation(
        hc.time_base_by_rotation(signal, tau, rate), -tau, rate)
    assert np.sqrt(np.mean((there_and_back[guard] - signal[guard]) ** 2)) * scale < floor


def test_the_demodulator_is_the_inverse_hilbert_with_an_exact_time_differential():
    """Ethan's demodulator, and the two ways to get it wrong. The exact
    derivative reaches a fifth of the ten-bit output floor; a central
    difference inside the cross-multiply scales the answer by a sinc and
    misses by three orders of magnitude."""
    from vhsdecode.models import output_limit as ol

    rate, n = 40e6, 1 << 14
    t = np.arange(n) / rate
    # a whole number of cycles in the record, because the exact derivative
    # is a transform-domain operator and assumes the record repeats; on
    # material that does not, guard an eighth at each end (measured in the
    # function's docstring)
    carrier = round(3.9e6 * n / rate) * rate / n
    modulation = round(100e3 * n / rate) * rate / n
    deviation = 0.5e6
    signal = np.cos(2 * np.pi * carrier * t
                    + (deviation / modulation) * np.sin(2 * np.pi * modulation * t))
    truth = carrier + deviation * np.cos(2 * np.pi * modulation * t)
    guard = slice(2000, -2000)
    floor = ol.output_profile(bits=10)["quantisation_noise_ire"]
    per_hz = 140.0 / 1.0e6                      # the deviation spans 140 IRE

    exact = hc.demodulate(signal, rate)
    naive = hc.demodulate(signal, rate, exact=False)
    exact_ire = np.sqrt(np.mean((exact[guard] - truth[guard]) ** 2)) * per_hz
    naive_ire = np.sqrt(np.mean((naive[guard] - truth[guard]) ** 2)) * per_hz
    assert exact_ire < 0.5 * floor              # measured about 0.18x
    assert naive_ire > 100.0 * floor            # measured about 769x
    # the sinc is the whole of the naive error: it reads the carrier low
    w = 2 * np.pi * carrier / rate
    assert np.mean(naive[guard]) == pytest.approx(carrier * np.sin(w) / w, rel=0.02)


def test_the_exact_derivative_beats_a_central_difference_on_a_known_slope():
    rate, n = 40e6, 4096
    t = np.arange(n) / rate
    frequency = round(3.9e6 * n / rate) * rate / n      # an exact bin
    tone = np.cos(2 * np.pi * frequency * t)
    truth = -2 * np.pi * frequency * np.sin(2 * np.pi * frequency * t)
    guard = slice(64, -64)
    exact = hc.derivative(tone, rate)
    naive = np.gradient(tone) * rate
    assert np.abs(exact[guard] - truth[guard]).max() < 1e-6 * np.abs(truth).max()
    assert np.abs(naive[guard] - truth[guard]).max() > 0.05 * np.abs(truth).max()
