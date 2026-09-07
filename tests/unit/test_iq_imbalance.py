"""The chroma's quadrature imbalance, seen through the burst itself.

Ethan: 'The vectorscope looks like an IQ imbalance on the chroma channel',
and then the method: 'I think we can see the IQ imbalance by using the
entire complex shape of the burst itself in all dimensions.'
"""

import math

import numpy as np
import pytest

from vhsdecode.models import iq_imbalance as iq


def _series(lines=480, rotation_deg=90.0, imbalance=0.0, phase=0.3, seed=0):
    index = np.arange(lines, dtype=np.float64)
    ideal = np.exp(1j * (math.radians(rotation_deg) * index + phase))
    return ideal + imbalance * np.conj(ideal)


def test_a_half_turn_a_line_cannot_separate_the_image_and_a_quarter_can():
    """One phasor fits any alpha, so the reference has to sweep.

    At a half turn a line the signal and its image both change sign every
    line and are the same sequence; at a quarter turn they rotate opposite
    ways and are orthogonal. This is why the measurement belongs on the
    colour-under and not on the up-converted burst.
    """
    half = iq.separability(180.0)
    quarter = iq.separability(90.0)
    assert not half["separable"]
    assert quarter["separable"]
    assert math.isclose(half["coherence"], 1.0, abs_tol=1e-9)
    assert quarter["coherence"] < 1e-9
    assert half["condition"] > 1e10
    assert math.isclose(quarter["condition"], 1.0, rel_tol=1e-6)


def test_a_balanced_pair_shows_no_image():
    out = iq.from_burst_series(_series(imbalance=0.0))
    assert out["usable"]
    assert out["image_ratio"] < 1e-9
    assert out["rotation_is_a_quarter_turn"]


def test_a_planted_imbalance_is_recovered_to_a_third_of_a_per_cent():
    """And the small shortfall is a real second-order bias, not slack.

    The rotation each run is de-rotated by is estimated from the same data
    that carries the image, so the image biases its own estimate a little.
    Measured: 0.020 comes back 0.019947, 0.050 as 0.049866 and 0.150 as
    0.149581 - a shortfall of 0.27, 0.27 and 0.28 per cent, which is flat
    in the imbalance and so is a fixed fractional bias rather than a
    growing error.
    """
    for planted in (0.02, 0.05, 0.15):
        out = iq.from_burst_series(_series(imbalance=planted))
        assert math.isclose(out["image_ratio"], planted, rel_tol=5e-3)
        shortfall = 1.0 - out["image_ratio"] / planted
        assert 0.0 < shortfall < 0.01


def test_the_rotation_is_taken_from_the_run_and_not_assumed():
    """The standard advances on one track and retards on the other.

    A fixed sign is wrong for half of any capture spanning a head switch,
    and using one returned an image stronger than the signal, which is not
    a thing.
    """
    for rotation in (90.0, -90.0):
        out = iq.from_burst_series(_series(rotation_deg=rotation,
                                           imbalance=0.05))
        assert math.isclose(out["rotation_deg_per_line"], rotation, abs_tol=0.5)
        assert math.isclose(out["image_ratio"], 0.05, rel_tol=5e-3)
        assert out["image_ratio"] < 1.0


def test_the_imbalance_is_not_visible_in_the_magnitude():
    """Which is the whole reason it needs the complex shape.

    Both terms have the same modulus, so the per-line amplitude of an
    imbalanced burst barely moves while its image ratio is a fifth.
    """
    balanced = np.abs(_series(imbalance=0.0))
    skewed = np.abs(_series(imbalance=0.2))
    assert abs(balanced.mean() - skewed.mean()) / balanced.mean() < 0.05
    assert iq.from_burst_series(_series(imbalance=0.2))["image_ratio"] > 0.19


def test_the_control_sits_where_no_imbalance_can_put_anything():
    out = iq.from_burst_series(_series(imbalance=0.10))
    assert out["control_ratio"] < 0.02
    assert out["over_control"] > 5.0


def test_a_short_record_is_refused_rather_than_fitted():
    with pytest.raises(ValueError):
        iq.from_burst_series(np.ones(50, dtype=complex))


def test_the_two_physical_readings_follow_from_the_image_ratio():
    out = iq.from_burst_series(_series(imbalance=0.1067))
    ratio = out["image_ratio"]
    assert math.isclose(out["gain_imbalance_fraction"], 2.0 * ratio,
                        rel_tol=1e-12)
    assert math.isclose(out["quadrature_error_deg"],
                        math.degrees(2.0 * ratio), rel_tol=1e-12)
    # and both follow the measured ratio, which is itself within a third of
    # a per cent of what was planted
    assert math.isclose(ratio, 0.1067, rel_tol=5e-3)


# --------------------------------------------------------------------------
# THE INVERSE. Planted lines at four times the subcarrier, a burst on every
# line, the picture a band-limited complex baseband gated to zero through
# sync and the porches as a chroma line is. Line width 910 is 4 x 227.5
# cycles (SMPTE 170M clause 8.4); the burst window is the decoder's own 74
# to 110, nine cycles (burst_instrument); the 6-sample raised-cosine edges
# are the shortest the table under `correct` puts below -55 dB.
# --------------------------------------------------------------------------

from vhsdecode.models import composite_channel, hypercomplex, output_limit

WIDTH = 910
BURST = (74, 110)
ACTIVE = (130, 880)
EDGE = 6
KAPPA = 0.1067 * np.exp(0.7j)          # the playback tap's 0.1067, at an angle


def _gate(index, start, stop, edge):
    shape = np.zeros(index.size)
    shape[(index >= start) & (index < stop)] = 1.0
    for k in range(edge):
        weight = 0.5 * (1.0 - math.cos(math.pi * (k + 0.5) / edge))
        shape[start + k] = weight
        shape[stop - 1 - k] = weight
    return shape


def _planted_baseband(lines, seed, bandwidth=0.04, gated=True, edge=EDGE):
    # 0.04 cycles a sample is 0.57 MHz at 4 fsc: the chroma separator's
    # half-width, SMPTE 32M 3.9.2.1.1 (composite_channel.SEPARATOR_MINUS_3DB_HZ,
    # 3.08 to 4.08 MHz about 3.58), rounded up so the picture fills its band
    rng = np.random.default_rng(seed)
    index = np.arange(WIDTH)
    frequency = np.fft.fftfreq(WIDTH)
    out = np.zeros((lines, WIDTH), dtype=np.complex128)
    for line in range(lines):
        spectrum = np.fft.fft(rng.standard_normal(WIDTH)
                              + 1j * rng.standard_normal(WIDTH))
        spectrum[np.abs(frequency) > bandwidth] = 0.0
        picture = np.fft.ifft(spectrum)
        picture *= 0.5 / np.abs(picture).std()
        if gated:
            picture = picture * _gate(index, ACTIVE[0], ACTIVE[1], edge)
        out[line] = picture + _gate(index, BURST[0], BURST[1], edge)
    return out


def _rotating(lines, seed, phase=0.4, **kwargs):
    """A colour-under-like field: the burst turns a quarter turn a line."""
    turn = math.radians(composite_channel.PHASE_ROTATION_DEG_PER_LINE)
    return _planted_baseband(lines, seed, **kwargs) * np.exp(
        1j * (turn * np.arange(lines) + phase))[:, None]


def _real_lines(baseband):
    index = np.arange(WIDTH)
    return np.ascontiguousarray(np.real(
        baseband * np.exp(2j * math.pi * iq.FSC_CYCLES_PER_SAMPLE * index)))


def _baseband_of(lines):
    index = np.arange(WIDTH)
    analytic = lines + 1j * hypercomplex.hilbert(lines, axis=-1)
    return analytic * np.exp(-2j * math.pi * iq.FSC_CYCLES_PER_SAMPLE * index)


def _relative_to_burst(baseband):
    burst = baseband[:, BURST[0] + 8:BURST[1] - 8].mean(axis=1)
    return baseband * np.conj(burst / np.abs(burst))[:, None]


def _db(error, reference):
    return 20.0 * math.log10(np.linalg.norm(error) / np.linalg.norm(reference))


def _complex_ratio(phasors):
    """beta/alpha the module's way, with its phase: the period-two component
    over the mean, after de-rotation by the run's own rotation."""
    step = float(np.angle(complex((phasors[1:] * np.conj(phasors[:-1])).sum())))
    index = np.arange(phasors.size)
    turned = phasors * np.exp(-1j * step * index)
    return complex((turned * (-1.0) ** index).mean() / turned.mean())


def _floor_db(baseband):
    """The analytic signal's own error on these lines, relative to the burst,
    with nothing planted: the floor any correction is judged against."""
    lines = _real_lines(baseband)
    return _db(_relative_to_burst(_baseband_of(lines))
               - _relative_to_burst(baseband), baseband)


def test_the_grid_constants_come_from_the_specification_and_not_from_typing():
    """Four samples a cycle is the 4 fsc grid's, a quarter turn a line is
    SMPTE 32M's, and both are bound to their homes rather than retyped."""
    assert iq.FSC_CYCLES_PER_SAMPLE == 0.25
    assert iq.FSC_CYCLES_PER_SAMPLE == (
        output_limit.NTSC_SUBCARRIER_HZ / output_limit.FOUR_FSC_HZ)
    assert iq.BURST_ROTATION_DEG_PER_LINE == composite_channel.PHASE_ROTATION_DEG_PER_LINE
    assert iq.BURST_ROTATION_DEG_PER_LINE == 90.0


def test_correct_baseband_inverts_a_planted_pair_to_machine_precision():
    """Measured: 4.97e-16 with alpha = 1, 6.28e-16 with alpha = 0.9 e^{0.4j};
    the module's own instrument reads 0.106691 before and 4.35e-16 after on
    a rotating burst series, against a control of 5.79e-16."""
    b = _planted_baseband(8, 1)
    for alpha in (1.0, 0.9 * np.exp(0.4j)):
        z = alpha * b + KAPPA * np.conj(b)
        assert np.abs(iq.correct_baseband(z, alpha, KAPPA, 1.0) - b).max() < 1e-12
    series_true = _series(rotation_deg=90.0, imbalance=0.0, phase=0.3)
    series = series_true + KAPPA * np.conj(series_true)
    before = iq.from_burst_series(series)
    after = iq.from_burst_series(iq.correct_baseband(series, 1.0, KAPPA, 1.0))
    assert math.isclose(before["image_ratio"], abs(KAPPA), rel_tol=5e-3)
    assert after["image_ratio"] < 1e-12
    assert after["image_ratio"] < 10.0 * max(after["control_ratio"], 1e-16)


def test_half_the_amount_halves_the_image_and_is_linear_in_between():
    """The amount scales the CHANGE, as the correction gain law has it, so
    the residual image is exactly (1 - amount) |beta/alpha|: measured
    0.106700, 0.080025, 0.053350, 0.026675 at 0, 0.25, 0.5, 0.75. Half is
    6.02 dB below the uncorrected image - not half of the way in decibels,
    which is undefined against a floor at machine precision."""
    b = _planted_baseband(8, 1)
    z = b + KAPPA * np.conj(b)
    for amount in (0.0, 0.25, 0.5, 0.75):
        out = iq.correct_baseband(z, 1.0, KAPPA, amount)
        ratio = np.linalg.norm(out - b) / np.linalg.norm(b)
        assert math.isclose(ratio, (1.0 - amount) * abs(KAPPA), rel_tol=1e-9)
    half = np.linalg.norm(iq.correct_baseband(z, 1.0, KAPPA, 0.5) - b)
    full = np.linalg.norm(z - b)
    assert math.isclose(20.0 * math.log10(half / full), -6.0206, abs_tol=1e-3)


def test_the_map_is_refused_where_it_is_singular():
    z = np.ones(4, dtype=complex)
    for alpha, beta in ((1.0, 1.0), (1.0, 1.2), (0.5, 0.5j)):
        with pytest.raises(ValueError):
            iq.correct_baseband(z, alpha, beta)
        with pytest.raises(ValueError):
            iq.correct(np.zeros((4, WIDTH)), np.ones(4), alpha, beta)
    with pytest.raises(ValueError):
        iq.correct_baseband(z, 0.0, 0.1)
    with pytest.raises(ValueError):
        iq.correct_baseband(z, 1.0, 0.1, amount=float("nan"))


def test_real_lines_are_corrected_to_the_analytic_signals_floor():
    """The whole chain on real lines, judged relative to each line's burst.

    240 lines, burst turning a quarter turn a line, the pair measured from
    the observed bursts the module's way (|kappa| 0.10669 for 0.1067
    planted, the documented 0.27 per cent short). Measured: -19.38 dB
    before, -55.60 after, against the analytic signal's own floor of -55.72
    on the same lines with nothing planted - the gate edges' error, recorded
    by edge length under `correct`. The improvement is the planted amount
    less that floor, and a phase given in radians reads the same as the
    phasor."""
    lines = 240
    b = _rotating(lines, 5)
    z = b + KAPPA * np.conj(b)
    original = _real_lines(z)
    floor = _floor_db(b)
    assert -58.0 < floor < -53.0
    reference = iq.line_reference(original, BURST)
    kappa = _complex_ratio(reference[:lines // 2])
    assert math.isclose(abs(kappa), abs(KAPPA), rel_tol=5e-3)
    before = _db(_relative_to_burst(_baseband_of(original))
                 - _relative_to_burst(b), b)
    assert math.isclose(before, 20.0 * math.log10(abs(KAPPA)), abs_tol=0.3)

    corrected = original.copy()
    out = iq.correct(corrected, reference, 1.0, kappa, amount=1.0)
    after = _db(_relative_to_burst(_baseband_of(corrected))
                - _relative_to_burst(b), b)
    assert out["lines"] == lines and out["skipped"] == 0
    assert after <= floor + 0.5
    assert before - after >= (before - floor) - 0.5
    assert math.isclose(out["reference_wobble_deg"], 0.68, abs_tol=0.05)

    by_phase = original.copy()
    iq.correct(by_phase, np.angle(reference), 1.0, kappa, amount=1.0)
    assert np.abs(by_phase - corrected).max() < 1e-9


def test_half_the_amount_on_real_lines_is_six_decibels():
    """Measured -25.40 dB at a half against -19.38 before: 6.02 dB."""
    lines = 240
    b = _rotating(lines, 5)
    original = _real_lines(b + KAPPA * np.conj(b))
    reference = iq.line_reference(original, BURST)
    kappa = _complex_ratio(reference[:lines // 2])
    before = _db(_relative_to_burst(_baseband_of(original))
                 - _relative_to_burst(b), b)
    half = original.copy()
    out = iq.correct(half, reference, 1.0, kappa, amount=0.5)
    after = _db(_relative_to_burst(_baseband_of(half)) - _relative_to_burst(b), b)
    assert math.isclose(after, before - 6.0206, abs_tol=0.15)
    assert math.isclose(out["residual_image_ratio"], 0.5 * abs(kappa), rel_tol=1e-12)


def test_the_bursts_frame_alternates_the_coefficient_every_line():
    """The trap, measured: a constant coefficient in the burst's frame lands
    at -16.2 dB, worse than the -19.4 uncorrected; the wrong parity at
    -13.2, the image doubled."""
    lines = 240
    b = _rotating(lines, 5)
    original = _real_lines(b + KAPPA * np.conj(b))
    reference = iq.line_reference(original, BURST)
    kappa = _complex_ratio(reference[:lines // 2])
    before = _db(_relative_to_burst(_baseband_of(original))
                 - _relative_to_burst(b), b)
    constant = original.copy()
    iq.correct(constant, reference, 1.0, kappa, amount=1.0,
               rotation_deg_per_line=0.0)
    worse = _db(_relative_to_burst(_baseband_of(constant))
                - _relative_to_burst(b), b)
    assert worse > before
    assert math.isclose(worse, -16.2, abs_tol=0.5)
    parity = original.copy()
    iq.correct(parity, reference, 1.0, kappa, amount=1.0, line_offset=1)
    doubled = _db(_relative_to_burst(_baseband_of(parity))
                  - _relative_to_burst(b), b)
    assert math.isclose(doubled, before + 6.0206, abs_tol=0.3)


def test_the_reference_is_cleared_of_the_image_it_carries():
    """The observed burst is (alpha + beta_n) times the true one. A pure
    quadrature error of the playback tap's size wobbles it 6.09 degrees at
    period two; with the division `correct` makes, -55.65 dB, and without
    it -32.73. A pure gain imbalance has no wobble: -55.87 dB either way."""
    lines = 240
    for planted, expected_wobble, expected_without in (
            (0.1067j, 6.09, -32.7), (0.1067, 0.0, -55.9)):
        b = _rotating(lines, 6, phase=0.0)
        original = _real_lines(b + planted * np.conj(b))
        reference = iq.line_reference(original, BURST)
        corrected = original.copy()
        out = iq.correct(corrected, reference, 1.0, planted, amount=1.0)
        assert math.isclose(out["reference_wobble_deg"], expected_wobble,
                            abs_tol=0.05)
        with_division = _db(_relative_to_burst(_baseband_of(corrected))
                            - _relative_to_burst(b), b)
        assert with_division <= _floor_db(b) + 0.5
        # the same correction with the observed direction taken as the frame
        unit = reference / np.abs(reference)
        alternating = planted * (-1.0) ** np.arange(lines)
        framed = _baseband_of(original) * np.conj(unit)[:, None]
        naive = iq.correct_baseband(framed, 1.0, alternating[:, None], 1.0)
        without = _db(_relative_to_burst(naive * unit[:, None])
                      - _relative_to_burst(b), b)
        assert math.isclose(without, expected_without, abs_tol=0.5)


def test_the_decoders_locked_and_inverted_lines():
    """The decoder's own case: the up-conversion's local oscillator sits
    above the colour-under, so the baseband is conjugated, and each line is
    then locked so its observed burst sits on the composite's alternating
    target. Passing conj(beta) with the lock's target as the reference goes
    from -19.37 dB to -55.56 at amount 1 and -25.39 at a half, against a
    floor of -55.72, with the burst held on the target to 0.001 degrees.
    The two traps: beta unconjugated lands at -30.1 dB, and a constant
    coefficient at -16.3, worse than uncorrected."""
    lines = 240
    b = _rotating(lines, 7)
    z = b + KAPPA * np.conj(b)
    colour_under_burst = z[:, BURST[0]:BURST[1]].mean(axis=1)
    kappa = _complex_ratio(colour_under_burst[:lines // 2])
    # SMPTE 170M clause 8.4: the composite burst alternates half a turn
    target = math.pi * (np.arange(lines) % 2) + 1.0
    lock = target - np.angle(np.conj(colour_under_burst))
    locked = np.conj(z) * np.exp(1j * lock)[:, None]
    true_burst = b[:, BURST[0]:BURST[1]].mean(axis=1)
    ideal = np.conj(b) * np.exp(
        1j * (target - np.angle(np.conj(true_burst))))[:, None]
    original = _real_lines(locked)
    reference = np.exp(1j * target)
    floor = _db(_baseband_of(_real_lines(ideal)) - ideal, ideal)
    before = _db(_baseband_of(original) - ideal, ideal)
    assert math.isclose(before, 20.0 * math.log10(abs(KAPPA)), abs_tol=0.3)

    corrected = original.copy()
    iq.correct(corrected, reference, 1.0, np.conj(kappa), amount=1.0)
    after = _db(_baseband_of(corrected) - ideal, ideal)
    assert after <= floor + 0.5
    held = iq.line_reference(corrected, BURST)
    assert np.degrees(np.abs(np.angle(held / reference))).max() < 0.01

    half = original.copy()
    iq.correct(half, reference, 1.0, np.conj(kappa), amount=0.5)
    assert math.isclose(_db(_baseband_of(half) - ideal, ideal),
                        before - 6.0206, abs_tol=0.15)

    unconjugated = original.copy()
    iq.correct(unconjugated, reference, 1.0, kappa, amount=1.0)
    assert math.isclose(_db(_baseband_of(unconjugated) - ideal, ideal),
                        -30.1, abs_tol=0.5)
    constant = original.copy()
    iq.correct(constant, reference, 1.0, np.conj(kappa), amount=1.0,
               rotation_deg_per_line=0.0)
    assert _db(_baseband_of(constant) - ideal, ideal) > before


def test_a_heterodyne_above_the_signal_conjugates_the_pair():
    """The decoder's local oscillator runs at pi/2 (1 + f_cu/f_sc) radians a
    sample - f_sc + f_cu on the 4 fsc grid, above the colour-under. On a
    periodic tone (four lines: 910 subcarrier cycles and 160 colour-under
    cycles, both whole) the up-converted baseband is conj(z) to -252.7 dB,
    and the inverse needs conj(beta): -251.7 dB, where beta as measured
    leaves -16.7."""
    length = 4 * WIDTH
    index = np.arange(length)
    colour_under = composite_channel.COLOUR_UNDER_HZ / output_limit.FOUR_FSC_HZ
    assert math.isclose(colour_under * length, 160.0, abs_tol=1e-9)
    assert math.isclose(iq.FSC_CYCLES_PER_SAMPLE * length, 910.0)
    local_oscillator = math.pi / 2 * (
        1.0 + composite_channel.COLOUR_UNDER_HZ / composite_channel.NTSC_SUBCARRIER_HZ)
    assert math.isclose(local_oscillator / (2 * math.pi),
                        iq.FSC_CYCLES_PER_SAMPLE + colour_under)
    b = np.full((1, length), 0.8 * np.exp(0.9j))
    z = b + KAPPA * np.conj(b)
    tape = np.real(z * np.exp(2j * math.pi * colour_under * index))
    product = 2.0 * tape * np.cos(local_oscillator * index)
    frequency = np.fft.fftfreq(length)
    keep = np.abs(np.abs(frequency) - iq.FSC_CYCLES_PER_SAMPLE) < 0.06
    up = np.real(np.fft.ifft(np.fft.fft(product, axis=-1) * keep, axis=-1))
    analytic = up + 1j * hypercomplex.hilbert(up, axis=-1)
    baseband = analytic * np.exp(-2j * math.pi * iq.FSC_CYCLES_PER_SAMPLE * index)
    assert _db(baseband - np.conj(z), z) < -200.0
    assert _db(iq.correct_baseband(baseband, 1.0, np.conj(KAPPA)) - np.conj(b), b) < -200.0
    assert _db(iq.correct_baseband(baseband, 1.0, KAPPA) - np.conj(b), b) > -20.0


def test_the_corrected_burst_stays_where_the_reference_put_it():
    """Measured: the phase kept to 2.8e-14 degrees, the amplitude wobble
    0.1062 of the mean before and 0.0000 after. `image_of_lines` then reads
    the kept phase wobble as |Im(beta/alpha)|: 0.10669 before, 0.01068
    after at a planted 0.01056 - and reports the decoder's locked lines,
    whose burst alternates a half turn, as not separable."""
    lines = 240
    b = _rotating(lines, 5)
    original = _real_lines(b + KAPPA * np.conj(b))
    reference = iq.line_reference(original, BURST)
    kappa = _complex_ratio(reference[:lines // 2])
    before = iq.image_of_lines(original, BURST)
    assert before["usable"]
    assert math.isclose(before["image_ratio"], abs(KAPPA), rel_tol=5e-3)
    corrected = original.copy()
    iq.correct(corrected, reference, 1.0, kappa, amount=1.0)
    held = iq.line_reference(corrected, BURST)
    assert np.degrees(np.abs(np.angle(held / reference))).max() < 1e-9
    assert np.abs(reference).std() / np.abs(reference).mean() > 0.1
    assert np.abs(held).std() / np.abs(held).mean() < 1e-3
    after = iq.image_of_lines(corrected, BURST)
    assert after["usable"]
    assert math.isclose(after["image_ratio"], abs(kappa.imag), rel_tol=0.05)
    target = math.pi * (np.arange(lines) % 2)
    locked = _real_lines(b * np.exp(1j * (target - np.angle(
        b[:, BURST[0]:BURST[1]].mean(axis=1))))[:, None])
    assert not iq.image_of_lines(locked, BURST)["usable"]


def test_lines_without_a_reference_are_left_untouched():
    lines = np.random.default_rng(0).standard_normal((4, WIDTH))
    kept = lines.copy()
    reference = np.array([1.0, 0.0, np.nan, 1.0]) * np.exp(0.2j)
    out = iq.correct(lines, reference, 1.0, 0.1, amount=1.0)
    assert out["lines"] == 2 and out["skipped"] == 2
    assert np.array_equal(lines[1], kept[1]) and np.array_equal(lines[2], kept[2])
    assert not np.array_equal(lines[0], kept[0])
    with pytest.raises(TypeError):
        iq.correct(np.zeros((4, WIDTH), dtype=np.int16), reference, 1.0, 0.1)
