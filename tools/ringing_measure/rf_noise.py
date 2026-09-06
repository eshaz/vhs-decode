"""Random RF noise in the complex plane, and what it becomes downstream.

Ethan: "I think I can represent the final residual using a random noise
distribution theorem. If we have a residual after all of our modeling steps
what remains, if above [the] chain's result, is unknown and likely random.
I want to remove the model of random RF noise in the complex plane."

The point is that the noise floor should be PREDICTED, not fitted. Every
stopping rule in this arc so far has compared the residual against an
empirical scalar - the accumulated cross-line variance - which is a
measurement of the residual by another name and therefore cannot say
whether what is left is noise or merely small. The noise has a known
distribution, that distribution has a known image after the demodulator,
and what the residual must be compared against is THAT.

THE DISTRIBUTION. Thermal and front-end noise on the RF is circularly
symmetric complex Gaussian: n = n_I + j n_Q with the two parts independent
and of equal variance. Three consequences follow without any fitting.

  * the envelope |n| is Rayleigh, and |n|^2 exponential with mean 2 sigma^2
  * the phase is uniform on the circle, so the noise carries no preferred
    direction and cannot be removed by a phase-sensitive correction
  * added to a carrier of amplitude A well above it, the noise resolves
    into an AMPLITUDE part (the component in phase with the carrier,
    variance sigma^2) and a PHASE part (the quadrature component, variance
    sigma^2 / A^2) - which is why the same noise reaches the envelope and
    the demodulated luma with different sizes and different spectra.

THE IMAGE AFTER THE DEMODULATOR. The FM demodulator differentiates the
phase, and differentiation multiplies by 2 pi f, so a flat phase noise
density becomes a demodulated density rising as f^2 - the triangular noise
every FM system has. That is the shape the residual's own spectrum must be
compared against, band by band, rather than against one number.

WHAT "REMOVE THE MODEL" MEANS HERE. Not subtracting a noise waveform -
there is no such thing to subtract, the phase being uniform is exactly the
statement that no realisation is knowable. It means subtracting the
PREDICTED POWER from the residual's measured power in each band, and
treating what remains as the only part that can be modelled at all. A band
whose residual sits at or under its predicted noise is finished, and the
process ends there because random noise cannot be estimated.
"""

import numpy as np

BOLTZMANN = 1.380649e-23
"""Exact by the 2019 SI redefinition."""


def resolve_components(carrier_amplitude, noise_sigma):
    """The amplitude and phase parts a complex Gaussian noise resolves into
    against a carrier.

    Returns (amplitude_sigma, phase_sigma_radians). Valid above threshold,
    where A is well clear of the noise; below it the demodulator's own
    non-linearity takes over and neither figure means anything, which the
    caller checks with `above_threshold`.
    """
    amplitude = np.asarray(carrier_amplitude, dtype=np.float64)
    sigma = float(noise_sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = np.where(amplitude > 0, sigma / np.maximum(amplitude, 1e-30),
                         np.nan)
    return np.full_like(amplitude, sigma), phase


def above_threshold(carrier_amplitude, noise_sigma, margin_db=10.0):
    """Whether the carrier is far enough above the noise for the resolution
    above to hold. The FM threshold is where clicks begin and the noise
    stops being small perturbations of the phase; below it none of this
    applies and the honest answer is that the model does not reach."""
    amplitude = np.asarray(carrier_amplitude, dtype=np.float64)
    ratio = amplitude / max(float(noise_sigma), 1e-30)
    return 20.0 * np.log10(np.maximum(ratio, 1e-30)) >= margin_db


def demodulated_density(frequencies_hz, carrier_amplitude, noise_sigma,
                        rate_hz):
    """The demodulated noise's power density per frequency, PREDICTED.

    The phase noise is flat at sigma^2 / A^2 over the RF band; the
    demodulator differentiates, which multiplies the density by (2 pi f)^2.
    Returned in the demodulated signal's own units squared per hertz, so a
    residual expressed as a frequency deviation compares to it directly.

    This is the f^2 law, and it matters that it is a law rather than a fit:
    a residual that follows it is noise however large it is, and one that
    does not is structure however small.
    """
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    amplitude = float(np.mean(np.asarray(carrier_amplitude, dtype=np.float64)))
    sigma = float(noise_sigma)
    if amplitude <= 0 or rate_hz <= 0:
        return np.full_like(frequency, np.nan)
    # one-sided phase noise density, flat across the band it occupies
    phase_density = (sigma / amplitude) ** 2 / (0.5 * rate_hz)
    # Differentiation multiplies the density by (2 pi f)^2, and the
    # demodulated quantity is a FREQUENCY in hertz - (1/2 pi) dphi/dt - so
    # the same (2 pi)^2 divides straight back out. Leaving it in overstates
    # the predicted floor by 39.5x, which on a first run read as the law
    # being wrong by two orders of magnitude when it was the units.
    return frequency ** 2 * phase_density


def quantization_density(step, rate_hz, frequencies_hz):
    """The capture's own floor: a uniform quantizer's error has variance
    d^2/12 spread flat over the band, and it is exact rather than measured.
    Any predicted floor must include it - it is the one part of the noise
    that is not the tape's."""
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    if rate_hz <= 0:
        return np.full_like(frequency, np.nan)
    return np.full_like(frequency, (float(step) ** 2 / 12.0) / (0.5 * rate_hz))


MEDIAN_OVER_MEAN = float(np.log(2.0))
"""A periodogram bin of Gaussian noise is EXPONENTIALLY distributed, so its
median is ln(2) = 0.693 of its mean.

Recorded as a constant because comparing a measured MEDIAN against a
predicted MEAN reads as the law being wrong by exactly this factor, and it
took a while to see: the ratio came out 0.690, 0.690, 0.691 across three
noise levels - flat, which should have said "a constant, not physics"
immediately. Compare means to means. Measured on the realisation the bins
are exponential to within a per cent: normalised power sd 0.991 against 1,
and 4.82 per cent above three times the mean against exp(-3) = 4.98."""


def spectrum_confidence(averages, confidence=0.95):
    """How far a measured power spectrum may sit above its own mean by
    chance.

    A periodogram bin of Gaussian noise is exponential - chi-square with
    two degrees of freedom - so averaging N of them gives chi-square with
    2N, and the upper bound is that distribution's quantile over 2N. This
    is the number that decides whether a band standing above the predicted
    floor is real or is the tail of a distribution with a long one.
    """
    from scipy import stats

    dof = 2 * max(int(averages), 1)
    return float(stats.chi2.ppf(confidence, dof) / dof)


def remove(measured_power, predicted_power, averages, confidence=0.95):
    """Subtract the noise MODEL from a measured spectrum.

    Not a waveform subtraction - the phase of the noise is uniform, which
    is precisely the statement that no realisation of it is knowable, so
    there is nothing to subtract in the time domain. What is subtracted is
    the predicted POWER, band by band.

    Returns, per band: the excess power that the noise model does not
    account for, and whether that excess clears the chance bound. A band
    that does not clear it is at its limit and no further component should
    be fitted there.
    """
    measured = np.asarray(measured_power, dtype=np.float64)
    predicted = np.asarray(predicted_power, dtype=np.float64)
    bound = spectrum_confidence(averages, confidence)
    excess = measured - predicted
    explained = np.where(predicted > 0, measured / np.maximum(predicted, 1e-30),
                         np.inf)
    return {
        "excess_power": np.maximum(excess, 0.0),
        "at_limit": explained <= bound,
        "ratio": explained,
        "chance_bound": bound,
    }


def verdict(measured_power, predicted_power, averages, confidence=0.95):
    """One line on whether the residual is finished.

    The process ends where the residual is indistinguishable from the noise
    the model predicts, because random noise cannot be estimated. This says
    how much of the spectrum has got there.
    """
    result = remove(measured_power, predicted_power, averages, confidence)
    at_limit = np.asarray(result["at_limit"])
    usable = np.isfinite(result["ratio"])
    if not usable.any():
        return {"finished": False, "share_at_limit": 0.0, **result}
    share = float(np.count_nonzero(at_limit & usable) / np.count_nonzero(usable))
    return {"finished": share >= 1.0, "share_at_limit": share, **result}


def through_deemphasis(density, frequencies_hz, tau_s, gain_db):
    """Carry the predicted noise through the de-emphasis, which is the
    whole reason the f^2 law is not what a decoded residual shows.

    Pre-emphasis at record and de-emphasis at playback exist precisely to
    flatten the demodulator's triangular noise: the recorder lifts the high
    frequencies before modulation and the player puts them back down,
    taking the f^2 noise down with them. A prediction made at the
    demodulator's output and compared against a residual measured AFTER
    de-emphasis is therefore comparing two different points in the chain,
    and will read the noise as far larger than it is at high frequency.

    Measured on the home tape's sync residual, the spectrum falls with
    frequency - local slopes of -6.2, +0.1, +1.1, -0.9 across the band -
    where the unshaped law predicts a steady +2. That is not the law
    failing; it is the law being evaluated at the wrong place in the chain.

    A single-pole shelf, which is what the format specifies: unity at DC,
    falling to `gain_db` above the corner set by `tau_s`.
    """
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    if tau_s <= 0:
        return np.asarray(density, dtype=np.float64)
    corner = 1.0 / (2.0 * np.pi * float(tau_s))
    floor = 10.0 ** (float(gain_db) / 20.0)
    ratio = frequency / corner
    # one zero at the corner, one pole where the shelf flattens
    # This shelf runs from unity at DC down to `floor` above the corner,
    # which IS the de-emphasis. Inverting it here gave a NINE DECIBEL BOOST
    # at high frequency - the pre-emphasis, the record side, the wrong
    # direction - and the tell was that the "de-emphasis" made the
    # predicted noise larger exactly where de-emphasis exists to make it
    # smaller.
    response = np.sqrt((1.0 + ratio ** 2)
                       / (1.0 + (ratio / max(floor, 1e-9)) ** 2))
    return np.asarray(density, dtype=np.float64) * response ** 2


def occupied_band(measured_power, frequencies_hz, law_slope=2.0,
                  tolerance=0.5):
    """Where the measured spectrum actually follows the law, measured
    rather than assumed.

    The band the noise occupies decides how a measured variance inverts
    into sigma/A, and getting it wrong is not a small error: under the f^2
    law the variance is dominated by the top of the band, so assuming
    Nyquist when the signal is limited to a fifth of it understates sigma
    badly. Rather than choose, this reports where the local slope is within
    `tolerance` of the law - and where nothing is, it says so.
    """
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    power = np.asarray(measured_power, dtype=np.float64)
    good = (frequency > 0) & (power > 0)
    if good.sum() < 8:
        return None
    logf = np.log10(frequency[good])
    logp = np.log10(power[good])
    window = max(len(logf) // 8, 4)
    following = []
    for start in range(0, len(logf) - window, max(window // 2, 1)):
        piece = slice(start, start + window)
        slope = np.polyfit(logf[piece], logp[piece], 1)[0]
        if abs(slope - law_slope) <= tolerance:
            following.append((10 ** logf[piece][0], 10 ** logf[piece][-1]))
    if not following:
        return None
    return (min(low for low, _ in following), max(high for _, high in following))
