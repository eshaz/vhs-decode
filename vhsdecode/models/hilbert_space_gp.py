"""ONE FUNCTION for every transient shape, in the Laplace eigenbasis.

Ethan, having watched the shapes trace out on a line scope: "is this just
the same function that happens with different format characteristics as
the constant model." This module is the answer, and the answer is yes.

THE CONSTRUCTION (Solin & Sarkka's Hilbert-space approximation to a
Gaussian process, the reference Ethan linked). On a bounded segment
[-L, L] the Laplace operator's eigenproblem

    d2 phi / dx2 = -lambda phi,      phi(-L) = phi(L) = 0

has eigenvalues and eigenfunctions that are FIXED - they know nothing
about the signal:

    lambda_j = (j pi / 2L)^2
    phi_j(x) = sqrt(1/L) sin(pi j (x + L) / 2L)

and any stationary covariance is approximated by weighting them with that
kernel's own spectral density evaluated at the eigenvalues:

    k(x, x') ~= sum_j S(sqrt(lambda_j)) phi_j(x) phi_j(x')

THE WHOLE POINT IS WHERE THE PARAMETERS LIVE. The basis is format-blind;
every dependence on amplitude, bandwidth and smoothness sits in the
DIAGONAL S(sqrt(lambda_j)). So fitting a transient stops being a search
over frequencies and decays and becomes a linear solve on a fixed basis
with a diagonal prior - which is exactly the shape of the derivation this
project already runs: the basis names WHICH frequencies are available,
and the spectral density carries their amplitude.

WHAT IT REPLACES HERE. The back porch has been fitted as one real pole
(the relaxation, tau about 1.25 us) plus one complex pair (the ring, near
2.43 MHz on home and 1.35 on bars) - four nonlinear parameters found by
scanning, with the scan railing whenever a shape had no time constant.
Both are single points of one smooth S(omega). Fitting the density
instead of enumerating poles removes the scan, the rails, and the
question of how many components to allow.

AND IT IS THE SAME FUNCTION AT EVERY SITE. The luma transient, the chroma
transient, the sync pulse's own residual and the back porch differ only
in the segment length L and the band the format allows - the constants.
`from_format` builds the arguments from those, so the caller states a
format and gets the same estimator.
"""

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np

__all__ = [
    "eigenvalues", "eigenfunctions", "mode_frequencies_hz",
    "spectral_density_squared_exponential", "spectral_density_matern",
    "approximate_kernel", "fit", "predict", "log_marginal_likelihood",
    "choose_hyperparameters", "group_delay_s", "from_format",
    "SUGGESTED_BOUNDARY_FACTOR",
]

SUGGESTED_BOUNDARY_FACTOR = 2.0
"""How far past the data the segment runs, as a multiple of its half width.

The eigenfunctions are pinned to zero at +-L, so a segment ending at the
data forces the fit to zero there and bends the last samples to reach it.

MEASURED rather than taken from the guidance. Reconstructing a known
squared-exponential kernel over the data span, the error floors at
2.3e-4 for a factor of 1.5 no matter how many modes are added - that is
boundary bias, and more modes cannot remove it - while a factor of 2.0
reaches 4.0e-15, machine precision, by 60 modes. The published range of
1.3 to 4 is right at its upper half and optimistic at its lower.
"""


def eigenvalues(modes: int, half_width: float) -> np.ndarray:
    """lambda_j = (j pi / 2L)^2, the Dirichlet Laplacian's spectrum."""
    orders = np.arange(1, int(modes) + 1, dtype=np.float64)
    return (orders * np.pi / (2.0 * float(half_width))) ** 2


def eigenfunctions(positions, modes: int, half_width: float) -> np.ndarray:
    """phi_j(x), orthonormal on [-L, L]: an (n x m) design matrix.

    FIXED. It does not depend on the kernel, the amplitude, the
    lengthscale or the format - which is what makes the fit a linear
    solve and lets one basis serve every site.
    """
    x = np.asarray(positions, dtype=np.float64).ravel()
    orders = np.arange(1, int(modes) + 1, dtype=np.float64)
    half = float(half_width)
    return np.sqrt(1.0 / half) * np.sin(
        np.pi * orders[None, :] * (x[:, None] + half) / (2.0 * half))


def mode_frequencies_hz(modes: int, half_width: float,
                        sample_rate_hz: float) -> np.ndarray:
    """What each mode means in hertz, so a density can be read as a
    spectrum rather than as a mode index.

    sqrt(lambda_j) is an ANGULAR frequency in radians per sample, so the
    ordinary frequency is sqrt(lambda_j) / (2 pi) cycles per sample, and
    the rate carries it to hertz. Mode j is j / (4L) cycles per sample -
    note the FOUR: the segment is 2L long and the eigenfunctions are half
    cycles on it, which is the same half-cycle convention as a DCT-II
    index and the reason both count in halves.
    """
    return (np.sqrt(eigenvalues(modes, half_width)) / (2.0 * np.pi)
            * float(sample_rate_hz))


def spectral_density_squared_exponential(angular, amplitude: float,
                                         lengthscale: float) -> np.ndarray:
    """S(w) for the squared-exponential kernel, in the ANGULAR convention.

    S(w) = a^2 l sqrt(2 pi) exp(-l^2 w^2 / 2)

    The convention matters and is the one thing worth checking rather
    than copying: sqrt(lambda_j) comes out of the eigenproblem as radians
    per sample, so the density must be written for angular frequency or
    the lengthscale is wrong by 2 pi. `approximate_kernel` is the test
    that settles it - it reconstructs the kernel only when the two agree.
    """
    w = np.asarray(angular, dtype=np.float64)
    ell = float(lengthscale)
    return (float(amplitude) ** 2 * ell * math.sqrt(2.0 * math.pi)
            * np.exp(-0.5 * (ell * w) ** 2))


def spectral_density_matern(angular, amplitude: float, lengthscale: float,
                            smoothness: float = 1.5) -> np.ndarray:
    """S(w) for a Matern kernel of order `smoothness` (nu), angular.

    Its tail falls as a POWER of frequency rather than a Gaussian, which
    matters here: a squared-exponential density says the transient is
    infinitely smooth, and a recovery tail with a corner is not. nu = 3/2
    is once differentiable and is the honest default for a shape built
    from poles.
    """
    w = np.asarray(angular, dtype=np.float64)
    nu = float(smoothness)
    ell = float(lengthscale)
    lead = (float(amplitude) ** 2 * 2.0 * math.sqrt(math.pi)
            * math.gamma(nu + 0.5) / math.gamma(nu)
            * (2.0 * nu) ** nu / ell ** (2.0 * nu))
    return lead * (2.0 * nu / ell ** 2 + w ** 2) ** (-(nu + 0.5))


def approximate_kernel(positions, modes: int, half_width: float,
                       density: Callable, *density_args) -> np.ndarray:
    """The approximation itself, for checking it against the real kernel.

    Kept in the module rather than the tests because it is the only thing
    that pins the angular-versus-ordinary convention, and getting that
    wrong scales every lengthscale by 2 pi while still looking plausible.
    """
    basis = eigenfunctions(positions, modes, half_width)
    weights = density(np.sqrt(eigenvalues(modes, half_width)), *density_args)
    return (basis * weights[None, :]) @ basis.T


def fit(positions, values, modes: int, half_width: float,
        amplitude: float, lengthscale: float, noise: float,
        density: Callable = spectral_density_squared_exponential,
        density_args: Sequence = ()) -> Dict[str, object]:
    """The linear solve: coefficients under the diagonal prior.

    beta ~ Normal(0, D) with D = diag(S(sqrt(lambda_j))), so the posterior
    mean is a ridge whose penalty is PER MODE - modes the density says are
    unlikely are shrunk hard, and the ones it favours are left alone. That
    is the whole estimator; there is no frequency search and nothing to
    rail.

    Solved through the Cholesky of the normal matrix, which is safe here
    for a reason worth stating: the basis is ORTHONORMAL, so Phi^T Phi is
    near the identity and the prior only adds to its diagonal. This is not
    the ill-conditioned normal-equation case - conditioning is bounded by
    the ratio of the largest to the smallest admitted density.
    """
    x = np.asarray(positions, dtype=np.float64).ravel()
    y = np.asarray(values, dtype=np.float64).ravel()
    basis = eigenfunctions(x, modes, half_width)
    weights = density(np.sqrt(eigenvalues(modes, half_width)),
                      amplitude, lengthscale, *density_args)
    weights = np.maximum(weights, 1e-300)
    variance = max(float(noise), 1e-12) ** 2
    normal = basis.T @ basis + variance * np.diag(1.0 / weights)
    try:
        factor = np.linalg.cholesky(normal)
        solved = np.linalg.solve(factor.T, np.linalg.solve(factor, basis.T @ y))
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(normal, basis.T @ y, rcond=None)[0]
    fitted = basis @ solved
    return {
        "coefficients": solved,
        "fitted": fitted,
        "residual": y - fitted,
        "residual_rms": float(np.sqrt(np.mean((y - fitted) ** 2))),
        "basis": basis,
        "prior_variance": weights,
        "modes": int(modes),
        "half_width": float(half_width),
        "amplitude": float(amplitude),
        "lengthscale": float(lengthscale),
        "noise": float(noise),
        # the per-mode POWER the data actually put there, which is the
        # measured spectrum on this basis - the thing to compare against
        # the density rather than against a list of fitted poles
        "mode_power": solved ** 2,
    }


def predict(model: Dict[str, object], positions) -> np.ndarray:
    """The fitted function anywhere on the segment, including between
    samples - the basis is continuous, so this interpolates rather than
    resamples."""
    basis = eigenfunctions(positions, model["modes"], model["half_width"])
    return basis @ np.asarray(model["coefficients"], dtype=np.float64)


def log_marginal_likelihood(positions, values, modes: int, half_width: float,
                            amplitude: float, lengthscale: float, noise: float,
                            density: Callable
                            = spectral_density_squared_exponential) -> float:
    """The evidence, which is how the hyperparameters are chosen.

    Because the basis is fixed, the marginal covariance is
    Phi D Phi^T + sigma^2 I, and the matrix determinant lemma turns its
    determinant and inverse into m-sized work rather than n-sized. This is
    what makes choosing the lengthscale cheap enough to do per field.
    """
    x = np.asarray(positions, dtype=np.float64).ravel()
    y = np.asarray(values, dtype=np.float64).ravel()
    n = len(y)
    basis = eigenfunctions(x, modes, half_width)
    weights = np.maximum(
        density(np.sqrt(eigenvalues(modes, half_width)), amplitude,
                lengthscale), 1e-300)
    variance = max(float(noise), 1e-12) ** 2
    inner = np.diag(variance / weights) + basis.T @ basis
    try:
        factor = np.linalg.cholesky(inner)
    except np.linalg.LinAlgError:
        return -np.inf
    projected = basis.T @ y
    quadratic = (y @ y - projected @ np.linalg.solve(
        factor.T, np.linalg.solve(factor, projected))) / variance
    log_determinant = (2.0 * np.sum(np.log(np.diag(factor)))
                       + (n - modes) * math.log(variance)
                       + np.sum(np.log(weights)))
    return float(-0.5 * (quadratic + log_determinant
                         + n * math.log(2.0 * math.pi)))


def choose_hyperparameters(positions, values, modes: int, half_width: float,
                           amplitudes=None, lengthscales=None, noise=None,
                           density: Callable
                           = spectral_density_squared_exponential
                           ) -> Dict[str, object]:
    """Amplitude and lengthscale by evidence, on a grid.

    A grid rather than an optimiser on purpose: the evidence is cheap
    here, a grid cannot run away to a boundary and report the boundary as
    a measurement (the failure that produced a railed time constant), and
    its resolution is a stated number rather than a convergence
    tolerance.
    """
    y = np.asarray(values, dtype=np.float64).ravel()
    spread = float(np.std(y)) or 1.0
    if amplitudes is None:
        amplitudes = np.geomspace(spread * 0.05, spread * 20.0, 24)
    if lengthscales is None:
        # from a couple of samples to the whole segment: shorter cannot be
        # resolved and longer is a constant, which the fit carries anyway
        lengthscales = np.geomspace(0.5, float(half_width), 32)
    if noise is None:
        noise = max(spread * 0.05, 1e-9)
    best = None
    for amplitude in np.atleast_1d(amplitudes):
        for lengthscale in np.atleast_1d(lengthscales):
            evidence = log_marginal_likelihood(
                positions, y, modes, half_width, float(amplitude),
                float(lengthscale), float(noise), density)
            if best is None or evidence > best[0]:
                best = (evidence, float(amplitude), float(lengthscale))
    evidence, amplitude, lengthscale = best
    return {"log_evidence": evidence, "amplitude": amplitude,
            "lengthscale": lengthscale, "noise": float(noise),
            "railed": bool(
                amplitude in (float(np.min(amplitudes)), float(np.max(amplitudes)))
                or lengthscale in (float(np.min(lengthscales)),
                                   float(np.max(lengthscales))))}


GROUP_DELAY_RESIDUAL_LIMIT = 0.1
"""How far the magnitude must have fallen by the top of the band before a
group delay read off it can be trusted.

The Bode relation is an integral over ALL frequencies. Given a truncated
band it silently returns a delay that is too SHORT, because the phase the
missing tail would have contributed is simply absent. Measured on a
one-pole low-pass of 200 ns: a 5 MHz band returns 0.83 of the true delay,
20 MHz returns 0.95, and 80 MHz returns 0.99. Nothing about the 5 MHz
answer looks wrong - it is smooth, positive and the right order.

So the caller is told. A magnitude still above this fraction of its peak
at the band edge means the roll-off is outside the window and the delay
is an underestimate.
"""


def group_delay_s(magnitude, frequency_hz, strict: bool = False):
    """The group delay of the MINIMUM-PHASE response with this magnitude.

    Ethan's reading of the shapes on the line scope - "essentially a group
    delay" - is the statement that a transient's trace and its band's
    delay-versus-frequency are the same object seen twice. For a
    minimum-phase channel they are: Bode's relation fixes the phase from
    the log magnitude, and the group delay is minus its derivative.

    So a fitted spectral density is not only an amplitude answer. It
    carries the delay the band imposes, which is what the eye sees smear.

    `strict` refuses rather than returning a truncation-shortened answer;
    see GROUP_DELAY_RESIDUAL_LIMIT for why that failure is invisible.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    size = len(grid)
    if size < 4:
        return np.full(size, np.nan)
    values = np.asarray(magnitude, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(values))) or 1.0
    truncated = abs(float(values[-1])) / peak > GROUP_DELAY_RESIDUAL_LIMIT
    if truncated and strict:
        raise ValueError(
            "the magnitude is still %.2f of its peak at the top of the band, "
            "so the roll-off lies outside it and the group delay would be an "
            "underestimate - widen the band or pass strict=False knowingly"
            % (abs(float(values[-1])) / peak))
    log_magnitude = np.log(np.maximum(values, 1e-300))
    # the Hilbert transform of the log magnitude, via the real cepstrum:
    # fold the negative quefrencies onto the positive ones and the result
    # is the minimum-phase partner
    doubled = np.concatenate([log_magnitude, log_magnitude[-2:0:-1]])
    cepstrum = np.fft.ifft(doubled).real
    folded = np.zeros_like(cepstrum)
    half = len(cepstrum) // 2
    folded[0] = cepstrum[0]
    folded[1:half] = 2.0 * cepstrum[1:half]
    if len(cepstrum) % 2 == 0:
        folded[half] = cepstrum[half]
    phase = np.imag(np.fft.fft(folded))[:size]
    step = float(np.mean(np.diff(grid)))
    if step <= 0:
        return np.full(size, np.nan)
    return -np.gradient(np.unwrap(phase), 2.0 * np.pi * step)


FITTING_BOUNDARY_FACTOR = 1.5
"""The boundary factor for FITTING data, which is not the one for
reconstructing a kernel - and the difference is five orders of magnitude of
conditioning.

TWO CRITERIA PULL OPPOSITE WAYS and each was measured.

  RECONSTRUCTING A KERNEL over the whole domain wants the boundary WELL
  clear of the data: 1.5 leaves a 2.3e-4 bias floor no number of modes
  removes, and 2.0 reaches machine precision. That is what
  `SUGGESTED_BOUNDARY_FACTOR` records.

  FITTING SAMPLED DATA is a different problem, because the data occupy
  only part of [-L, L] and eigenfunctions restricted to a sub-interval are
  NOT orthogonal. The mode count follows L (modes = 4 L f_band / f_s), so
  a larger boundary buys more modes over the same band and pays for them
  in conditioning. At 52 samples over a 3 MHz band:

      factor 1.25   L 32.5   27 modes   cond 3.4e+02
      factor 1.50   L 39.0   32 modes   cond 4.6e+06
      factor 1.75   L 45.5   38 modes   cond 5.1e+11
      factor 2.00   L 52.0   43 modes   cond 3.5e+15   <- within 1.3x of 1/eps

MEASURED ON REAL DATA, held out on independent lines:

      case      floor     1.25     1.50     1.75     2.00
      home A   0.0252   0.0375   0.0357   0.0356   0.0356
      home B   0.0285   0.0422   0.0404   0.0404   0.0404
      bars A   0.0245   0.0357   0.0347   0.0347   0.0347
      bars B   0.0162   0.0248   0.0230   0.0229   0.0229

1.50 is within 0.3 per cent of the best answer at NINE ORDERS OF
MAGNITUDE better conditioning, and 1.25 gives up 5 per cent. A factor of
1.0 was also tried, at perfect conditioning, and is 40 to 80 per cent
worse - the Dirichlet pinning at the data's own edges costs far more than
the conditioning buys.

`from_format` uses this one, because `from_format` exists to set up a fit.
"""


def from_format(segment_samples: int, sample_rate_hz: float,
                band_limit_hz: Optional[float] = None,
                boundary_factor: float = FITTING_BOUNDARY_FACTOR
                ) -> Dict[str, object]:
    """The estimator's arguments FROM FORMAT CONSTANTS, nothing fitted.

    This is the part that answers "the same function with different format
    characteristics". A site supplies how long its segment is, how fast it
    is sampled, and what the format's band limit is; everything else -
    where the boundary sits, how many modes are worth carrying - follows.

    The mode count is set by the band, not chosen: mode j sits at
    j / (4L) cycles per sample, so the highest mode the format can carry
    is 4 L f_band / f_sample. Asking for more is asking the fit to
    describe frequencies the channel cannot deliver, and they come back
    as whatever the noise put there.
    """
    samples = int(segment_samples)
    half_width = 0.5 * samples * float(boundary_factor)
    if band_limit_hz is None:
        band_limit_hz = 0.5 * float(sample_rate_hz)
    modes = int(math.floor(4.0 * half_width * float(band_limit_hz)
                           / float(sample_rate_hz)))
    modes = max(1, min(modes, samples))
    centre = 0.5 * (samples - 1)
    return {
        "positions": np.arange(samples, dtype=np.float64) - centre,
        "half_width": half_width,
        "modes": modes,
        "sample_rate_hz": float(sample_rate_hz),
        "band_limit_hz": float(band_limit_hz),
        "mode_frequencies_hz": mode_frequencies_hz(modes, half_width,
                                                   sample_rate_hz),
    }
