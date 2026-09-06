"""The Laplace eigenbasis, and the conventions that are easy to get wrong.

Two of these tests exist because the published formulation is internally
inconsistent, and copying it produces a model that looks plausible and is
wrong by a factor of 2 pi in the lengthscale.
"""

import numpy as np
import pytest

from vhsdecode.models import hilbert_space_gp as hsgp

FS = 4 * 315e6 / 88


def test_eigenfunctions_are_orthonormal():
    """The property the whole construction rests on: a fixed ORTHONORMAL
    basis is what turns the fit into a linear solve and keeps the normal
    matrix near the identity."""
    # integrated over the WHOLE segment: they are orthonormal on
    # [-L, L] and nothing smaller
    positions = np.linspace(-10.0, 10.0, 4001)
    basis = hsgp.eigenfunctions(positions, 12, 10.0)
    step = positions[1] - positions[0]
    gram = basis.T @ basis * step
    assert np.allclose(gram, np.eye(12), atol=2e-3)


def test_eigenfunctions_vanish_at_the_boundary():
    """Dirichlet conditions - which is why the boundary must sit clear of
    the data, or the fit is dragged to zero at the last samples."""
    edges = hsgp.eigenfunctions(np.array([-10.0, 10.0]), 8, 10.0)
    assert np.allclose(edges, 0.0, atol=1e-12)


def test_eigenvalues_match_the_closed_form():
    got = hsgp.eigenvalues(5, 7.0)
    want = (np.arange(1, 6) * np.pi / 14.0) ** 2
    assert np.allclose(got, want)


def test_kernel_reconstruction_pins_the_angular_convention():
    """THE CONVENTION TEST, and it is not academic.

    sqrt(lambda_j) leaves the eigenproblem as radians per unit, so the
    spectral density must be written for ANGULAR frequency. The reference
    write-up prints an exponent for ORDINARY frequency and evaluates it at
    sqrt(lambda_j) anyway. Measured here: the angular form reconstructs a
    known squared-exponential kernel to 4e-15, the printed form is off by
    2.4 against an amplitude of 2.89 - it is not a small error, it is a
    different kernel.
    """
    positions = np.linspace(-10.0, 10.0, 81)
    amplitude, lengthscale = 1.7, 2.3
    exact = amplitude ** 2 * np.exp(
        -0.5 * (positions[:, None] - positions[None, :]) ** 2 / lengthscale ** 2)
    approximate = hsgp.approximate_kernel(
        positions, 150, 30.0, hsgp.spectral_density_squared_exponential,
        amplitude, lengthscale)
    assert np.max(np.abs(approximate - exact)) < 1e-12

    def as_printed(angular, amp, ell):
        return (amp ** 2 * ell * np.sqrt(2 * np.pi)
                * np.exp(-2 * np.pi ** 2 * ell ** 2 * angular ** 2))

    wrong = hsgp.approximate_kernel(positions, 150, 30.0, as_printed,
                                    amplitude, lengthscale)
    assert np.max(np.abs(wrong - exact)) > 1.0


def test_boundary_factor_is_the_measured_one():
    """1.5 leaves a boundary-bias floor that no number of modes removes;
    2.0 reaches machine precision. The constant records the measurement."""
    positions = np.linspace(-10.0, 10.0, 81)
    exact = np.exp(-0.5 * (positions[:, None] - positions[None, :]) ** 2 / 4.0)
    tight = hsgp.approximate_kernel(
        positions, 400, 10.0 * 1.5, hsgp.spectral_density_squared_exponential,
        1.0, 2.0)
    generous = hsgp.approximate_kernel(
        positions, 400, 10.0 * hsgp.SUGGESTED_BOUNDARY_FACTOR,
        hsgp.spectral_density_squared_exponential, 1.0, 2.0)
    assert np.max(np.abs(tight - exact)) > 1e-6      # a floor, not noise
    assert np.max(np.abs(generous - exact)) < 1e-9


def test_matern_tail_is_a_power_law():
    """The reason it is the default here: a squared-exponential density
    says the transient is infinitely smooth, and a recovery tail built
    from poles is not."""
    angular = np.array([1.0, 2.0, 4.0])
    matern = hsgp.spectral_density_matern(angular, 1.0, 1.0, 1.5)
    ratios = matern[:-1] / matern[1:]
    # doubling frequency divides the density by ~2^(2nu+1) = 16 in the tail
    assert ratios[1] > ratios[0]
    gaussian = hsgp.spectral_density_squared_exponential(angular, 1.0, 1.0)
    assert matern[-1] > gaussian[-1]                  # heavier tail


def test_mode_count_and_frequencies_come_from_the_format():
    """The point of the whole module: the model's size is a format
    constant, not a choice. Mode j sits at j/(4L) cycles per sample."""
    frame = hsgp.from_format(52, FS, band_limit_hz=3.0e6)
    # 32 at the FITTING boundary factor of 1.5; the mode count follows L
    assert frame["modes"] == 32
    top = frame["mode_frequencies_hz"][-1]
    assert top <= 3.0e6
    assert top > 2.9e6                                # and it uses the band
    per_mode = FS / (4.0 * frame["half_width"])
    assert frame["mode_frequencies_hz"][0] == pytest.approx(per_mode)


def test_the_fitting_factor_is_conditioned_where_the_other_is_not():
    """Two criteria pull opposite ways and each was measured: kernel
    reconstruction wants the boundary well clear of the data, fitting
    wants the basis conditioned on it. At 52 samples over a 3 MHz band the
    reconstruction factor conditions at 3.5e15 - within 1.3x of 1/eps,
    where lstsq silently discards three directions - for an answer
    identical to four decimals."""
    positions = np.arange(52.0) - 25.5
    fitting = hsgp.eigenfunctions(positions, 32, 26.0 * hsgp.FITTING_BOUNDARY_FACTOR)
    reconstructing = hsgp.eigenfunctions(
        positions, 43, 26.0 * hsgp.SUGGESTED_BOUNDARY_FACTOR)
    assert np.linalg.cond(fitting) < 1e8
    assert np.linalg.cond(reconstructing) > 1e14
    assert hsgp.FITTING_BOUNDARY_FACTOR < hsgp.SUGGESTED_BOUNDARY_FACTOR


def test_from_format_uses_the_fitting_factor(self=None):
    """Because from_format exists to set up a fit."""
    frame = hsgp.from_format(52, FS, band_limit_hz=3.0e6)
    assert frame["half_width"] == pytest.approx(
        26.0 * hsgp.FITTING_BOUNDARY_FACTOR)


def test_a_wider_band_admits_more_modes():
    narrow = hsgp.from_format(52, FS, band_limit_hz=1.5e6)["modes"]
    wide = hsgp.from_format(52, FS, band_limit_hz=3.0e6)["modes"]
    assert wide == pytest.approx(2 * narrow, abs=1)


def test_fit_recovers_a_planted_shape():
    frame = hsgp.from_format(64, FS, band_limit_hz=3.0e6)
    positions = frame["positions"]
    truth = (2.0 * np.exp(-np.arange(64) / 9.0)
             * np.cos(2 * np.pi * 0.13 * np.arange(64)))
    model = hsgp.fit(positions, truth, frame["modes"], frame["half_width"],
                     amplitude=2.0, lengthscale=1.0, noise=1e-4,
                     density=hsgp.spectral_density_matern)
    assert model["residual_rms"] < 0.02 * np.std(truth)


def test_predict_interpolates_between_samples():
    """The basis is continuous, so the fit is defined off-grid - which is
    what lets one model serve sites at different sample rates."""
    frame = hsgp.from_format(48, FS, band_limit_hz=2.0e6)
    positions = frame["positions"]
    values = np.sin(2 * np.pi * np.arange(48) / 16.0)
    model = hsgp.fit(positions, values, frame["modes"], frame["half_width"],
                     1.0, 2.0, 1e-3)
    midpoints = 0.5 * (positions[:-1] + positions[1:])
    between = hsgp.predict(model, midpoints)
    expected = np.sin(2 * np.pi * (np.arange(47) + 0.5) / 16.0)
    assert np.max(np.abs(between - expected)) < 0.05


def test_evidence_prefers_the_true_lengthscale():
    generator = np.random.default_rng(20260906)
    positions = np.linspace(-20.0, 20.0, 201)
    truth_scale = 3.0
    kernel = np.exp(-0.5 * (positions[:, None] - positions[None, :]) ** 2
                    / truth_scale ** 2)
    sample = generator.multivariate_normal(np.zeros(len(positions)),
                                           kernel + 1e-8 * np.eye(len(positions)))
    scores = {}
    for lengthscale in (0.5, 1.5, 3.0, 8.0, 20.0):
        scores[lengthscale] = hsgp.log_marginal_likelihood(
            positions, sample, 60, 40.0, 1.0, lengthscale, 0.05)
    best = max(scores, key=scores.get)
    assert best in (1.5, 3.0)


def test_group_delay_of_a_pure_delay_is_that_delay():
    """The relation Ethan is invoking: a transient's trace and its band's
    delay-versus-frequency are one object. A minimum-phase magnitude with
    a single pole has a known group delay at DC, tau."""
    # THE BAND MUST CONTAIN THE ROLL-OFF, which is the point of the
    # second half of this test: at 5 MHz this same pole reads 0.83 of its
    # true delay and looks perfectly reasonable doing it.
    tau = 2.0e-7
    frequency = np.linspace(0.0, 80e6, 4096)
    magnitude = 1.0 / np.sqrt(1.0 + (2 * np.pi * frequency * tau) ** 2)
    delay = hsgp.group_delay_s(magnitude, frequency)
    assert delay[1] == pytest.approx(tau, rel=0.05)
    assert delay[2000] < delay[1]


def test_group_delay_refuses_a_truncated_band():
    """The truncation failure is invisible - smooth, positive, the right
    order of magnitude, and 17 per cent low."""
    tau = 2.0e-7
    frequency = np.linspace(0.0, 5e6, 2048)
    magnitude = 1.0 / np.sqrt(1.0 + (2 * np.pi * frequency * tau) ** 2)
    short = hsgp.group_delay_s(magnitude, frequency)
    assert short[1] < 0.9 * tau                       # quietly wrong
    with pytest.raises(ValueError, match="roll-off lies outside"):
        hsgp.group_delay_s(magnitude, frequency, strict=True)


def test_group_delay_is_zero_for_a_flat_magnitude():
    frequency = np.linspace(0.0, 5e6, 1024)
    delay = hsgp.group_delay_s(np.ones_like(frequency), frequency)
    assert np.nanmax(np.abs(delay)) < 1e-9
