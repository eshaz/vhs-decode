"""Which DISTRIBUTION each interference type draws from, and how to tell.

Ethan: *"I think the key is finding the right distribution to fit against.
Let's analyse to see if there are any distributions that make sense given the
various types of interference components I have identified, and the random one
for random noise."*

THERE ARE, AND THEY ARE NOT ALL THE SAME ONE - which is the whole point.
Everything in this arc so far has modelled a component's SHAPE: how it moves
the response against frequency. A shape says what a mechanism does on average.
It says nothing about how its departures are DISTRIBUTED, and that is a second,
independent axis on which mechanisms differ - often more sharply than their
shapes do. Two mechanisms with the same roll-off can have entirely different
tails, and the tail is frequently the easier measurement.

THE TAXONOMY, WITH THE PHYSICS THAT FIXES EACH ONE. Every entry is derived
from what the mechanism IS, not chosen for convenience:

  GAUSSIAN - thermal noise, receiver front end, preamplifier
      A sum of very many independent contributions, so the central limit
      theorem applies and there is nothing to choose. Excess kurtosis 0.

  POISSON, TENDING TO GAUSSIAN - particle noise on the tape
      A resolution cell holds a FINITE number of magnetic particles and the
      count fluctuates. That is Poisson by construction, and its signature is
      not its shape but that its VARIANCE TRACKS ITS MEAN - which no additive
      Gaussian does. With the particle counts a real coating carries it is
      numerically Gaussian, so the variance-mean relation is the only thing
      that separates them.

  LOG-NORMAL - modulation noise
      Modulation noise is MULTIPLICATIVE: it is the medium's own magnetisation
      fluctuating in proportion to what was written. A product of many small
      independent factors is log-normal, which is to say normal after a
      logarithm - and this arc already works in the log domain for exactly
      that reason.

  COMPOUND POISSON, HEAVY TAILED - dropouts
      Rare arrivals of large events. The arrivals are Poisson in time and the
      amplitudes are broad, so the marginal has a sharp peak and heavy tails:
      high excess kurtosis, and a variance dominated by a few samples.

  UNIFORM - quantisation
      Bounded by construction between plus and minus half a code, and flat
      within it provided the signal exercises more than a code or two. Excess
      kurtosis -1.2, which is the FLATTEST of anything here and therefore the
      easiest to recognise.

  ARCSINE - beat, co-channel, any surviving carrier
      The marginal distribution of a sinusoid sampled at random phase is
      `1 / (pi sqrt(A^2 - x^2))`: U-SHAPED, with its mass at the extremes
      rather than the centre. Excess kurtosis -1.5, the extreme opposite of a
      heavy tail. A residual carrier is the one interference type whose
      distribution is unmistakable at a glance.

  SPARSE, NOT A DISTRIBUTION - multipath, echo, ghosting
      A handful of discrete reflections. The right description is a count and
      a set of delays, not a density, and treating it as noise is exactly the
      error that makes a ghost unrecoverable. Its statistic is SPARSITY.

  DETERMINISTIC - head switching, the drum, anything periodic
      Once per field, at a known place. Not a distribution at all, and the
      honest handling is to exclude the samples rather than to model them.

EXCESS KURTOSIS IS THE PRIMARY DISCRIMINATOR AND IT ORDERS THEM CLEANLY:

    arcsine   -1.5   |   uniform  -1.2   |   Gaussian  0   |   dropouts  >> 0

which is a single number separating a carrier from a converter from thermal
noise from a dropout. That is a better-conditioned question than any shape fit
in this repository, and it needs no model of the chain at all.

WHAT THIS CANNOT DO, STATED FIRST. A distribution is a marginal: it is blind
to ORDER. White Gaussian noise and Gaussian noise through any filter have the
same marginal, so the distribution identifies the MECHANISM and never its
spectrum, exactly as the spectrum identifies the shape and never the mechanism.
The two axes are complementary and neither substitutes for the other - which is
the useful part, because a mechanism that is degenerate in one may be
separable in the other.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

# Excess kurtosis, so a Gaussian is 0. These are exact analytic values, not
# fitted, and they are what makes the classification a comparison against
# known constants rather than against a trained set.
EXCESS_KURTOSIS = {
    "arcsine": -1.5,
    "uniform": -1.2,
    "gaussian": 0.0,
    "laplace": 3.0,
}

DISTRIBUTIONS: Dict[str, Dict[str, object]] = {
    "thermal noise": {
        "distribution": "gaussian",
        "excess_kurtosis": 0.0,
        "why": ("a sum of very many independent contributions, so the central "
                "limit theorem applies and there is nothing to choose"),
        "discriminator": "excess kurtosis near zero",
        "additive": True,
    },
    "particle noise": {
        "distribution": "poisson",
        "excess_kurtosis": 0.0,
        "why": ("a resolution cell holds a finite number of particles and the "
                "count fluctuates; numerically Gaussian at real particle "
                "counts, so only the variance-mean relation separates them"),
        "discriminator": "variance proportional to mean",
        "additive": True,
    },
    "modulation noise": {
        "distribution": "lognormal",
        "excess_kurtosis": None,
        "why": ("multiplicative: the medium's magnetisation fluctuating in "
                "proportion to what was written, so a product of small "
                "factors and therefore normal after a logarithm"),
        "discriminator": "normal in the log domain, skewed in the linear one",
        "additive": False,
    },
    "dropout": {
        "distribution": "compound poisson",
        "excess_kurtosis": None,
        "why": ("rare arrivals of large events - Poisson in time with broad "
                "amplitudes, so a sharp peak and heavy tails"),
        "discriminator": "large positive excess kurtosis; variance carried by "
                        "a few samples",
        "additive": True,
    },
    "quantisation": {
        "distribution": "uniform",
        "excess_kurtosis": -1.2,
        "why": ("bounded by construction to plus and minus half a code and "
                "flat within it, provided the signal exercises more than a "
                "code or two"),
        "discriminator": "excess kurtosis near -1.2, and a hard bound",
        "additive": True,
    },
    "beat / co-channel": {
        "distribution": "arcsine",
        "excess_kurtosis": -1.5,
        "why": ("the marginal of a sinusoid at random phase is U-shaped, with "
                "its mass at the extremes; a surviving carrier is the one "
                "interference type that is unmistakable at a glance"),
        "discriminator": "excess kurtosis near -1.5, bimodal density",
        "additive": True,
    },
    "echo": {
        "distribution": "sparse",
        "excess_kurtosis": None,
        "why": ("a handful of discrete reflections; the right description is "
                "a count and a set of delays, not a density, and treating it "
                "as noise is what makes a ghost unrecoverable"),
        "discriminator": "sparsity - a few large coefficients, not a spread",
        "additive": True,
    },
    "head switch": {
        "distribution": "deterministic",
        "excess_kurtosis": None,
        "why": ("once per field at a known place; not a distribution, and the "
                "honest handling is to exclude the samples rather than model "
                "them"),
        "discriminator": "periodic at a known rate",
        "additive": True,
    },
}


def excess_kurtosis(samples) -> float:
    """The fourth standardised moment less three, so a Gaussian is zero."""
    values = np.asarray(samples, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float("nan")
    centred = values - values.mean()
    variance = float(np.mean(centred ** 2))
    if not variance > 0:
        return float("nan")
    return float(np.mean(centred ** 4) / variance ** 2 - 3.0)


def sparsity(samples) -> float:
    """The share of the total energy carried by the largest one per cent.

    A sparse residual - a few reflections - concentrates its energy; a noise
    residual spreads it. For Gaussian noise this sits near 0.03, and for a
    handful of discrete arrivals it approaches one.
    """
    values = np.abs(np.asarray(samples, dtype=np.float64).ravel()) ** 2
    values = values[np.isfinite(values)]
    if values.size < 100:
        return float("nan")
    ordered = np.sort(values)[::-1]
    top = max(int(round(0.01 * ordered.size)), 1)
    total = float(ordered.sum())
    return float(ordered[:top].sum() / total) if total > 0 else float("nan")


def variance_tracks_mean(blocks) -> Dict[str, float]:
    """Poisson's signature: the variance rises with the mean, in proportion.

    `blocks` is a sequence of sample groups taken at different signal levels.
    Returns the slope of variance against mean and the correlation, because a
    Poisson process gives a slope of one in its own units while any additive
    process gives zero.
    """
    means, variances = [], []
    for block in blocks:
        values = np.asarray(block, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size > 3:
            means.append(float(values.mean()))
            variances.append(float(values.var()))
    if len(means) < 3:
        return {"slope": float("nan"), "correlation": float("nan"),
                "why": "too few blocks to see a relation"}
    means, variances = np.array(means), np.array(variances)
    slope = float(np.polyfit(means, variances, 1)[0])
    correlation = float(np.corrcoef(means, variances)[0, 1])
    return {
        "slope": slope,
        "correlation": correlation,
        "poisson_like": bool(correlation > 0.8 and slope > 0),
        "why": ("a Poisson count's variance rises with its mean; any purely "
                "additive process gives a slope of zero"),
    }


def classify(samples, candidates: Optional[Sequence[str]] = None
             ) -> Dict[str, object]:
    """WHICH DISTRIBUTION A RESIDUAL LOOKS LIKE, and how confidently.

    The primary statistic is the excess kurtosis, because it orders the
    candidates cleanly - arcsine -1.5, uniform -1.2, Gaussian 0, heavy tails
    well above - and needs no model of the chain. Sparsity is reported
    alongside it because a sparse residual is not a distribution at all and
    would otherwise be classified as whichever density it least resembles.

    The margin to the runner-up is returned, because a kurtosis that sits
    between two candidates has not chosen between them and saying so is the
    useful answer.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 32:
        return {"identified": False, "why": "too few samples to classify"}

    kurtosis = excess_kurtosis(values)
    concentration = sparsity(values)
    names = list(candidates) if candidates else list(EXCESS_KURTOSIS)
    distances = {name: abs(kurtosis - EXCESS_KURTOSIS[name])
                 for name in names if name in EXCESS_KURTOSIS}
    if not distances:
        return {"identified": False, "why": "no candidate with a known moment"}

    best = min(distances, key=distances.get)
    ordered = sorted(distances.values())
    margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else np.inf
    # the standard error of the excess kurtosis of a Gaussian sample
    error = float(np.sqrt(24.0 / values.size))
    close = distances[best] < 2.0 * error

    # NAMING THE NEAREST CANDIDATE IS WRONG WHEN NOTHING IS NEAR. A dropout
    # residual measures an excess kurtosis of 530 and a sparse echo 3877;
    # both are hundreds of times outside every closed-form candidate here,
    # and reporting "laplace" because it is the least distant of four is a
    # label that would be believed. Where nothing is close the verdict says
    # so, and the sparsity says which way it failed - heavy-tailed noise and
    # a handful of discrete arrivals are different objects and neither is a
    # density in this table.
    verdict = best if close else (
        "sparse" if (np.isfinite(concentration) and concentration > 0.30)
        else "heavy tailed, none of these")
    return {
        "identified": bool(close and margin > error),
        "distribution": verdict,
        "nearest": best,
        "nearest_distance": float(distances[best]),
        "excess_kurtosis": kurtosis,
        "expected": EXCESS_KURTOSIS[best],
        "standard_error": error,
        "margin": margin,
        "sparsity": concentration,
        "looks_sparse": bool(np.isfinite(concentration)
                             and concentration > 0.30),
        "samples": int(values.size),
        "why": ("excess kurtosis orders the candidates cleanly and needs no "
                "model of the chain; sparsity is reported beside it because a "
                "sparse residual is not a density at all"),
    }


def separable_at(sample_count: int,
                 pair: Sequence[str] = ("arcsine", "uniform")
                 ) -> Dict[str, object]:
    """CAN THESE TWO BE TOLD APART AT THIS SAMPLE COUNT? The control.

    The excess kurtosis of a sample has standard error `sqrt(24/n)`, so two
    distributions whose true excess kurtoses differ by less than a few of
    those cannot be separated however carefully the fit is done. This states
    the requirement rather than discovering it after a fit has been believed.

    The hardest pair in the table is arcsine against uniform - they differ by
    0.3 - and the easiest is arcsine against Gaussian at 1.5.
    """
    first, second = pair
    gap = abs(EXCESS_KURTOSIS[first] - EXCESS_KURTOSIS[second])
    error = float(np.sqrt(24.0 / max(int(sample_count), 1)))
    return {
        "pair": (first, second),
        "gap": gap,
        "standard_error": error,
        "sigma": float(gap / error) if error > 0 else np.inf,
        "separable": bool(gap > 3.0 * error),
        "samples_needed": int(np.ceil(24.0 * 9.0 / max(gap ** 2, 1e-30))),
        "why": ("the excess kurtosis of a sample has standard error "
                "sqrt(24/n), so a gap smaller than a few of those cannot be "
                "resolved however carefully the fit is done"),
    }


def carrier_fraction(samples) -> Dict[str, float]:
    """WHAT SHARE OF THE POWER IS A CARRIER, from the kurtosis alone.

    An FM capture is a constant-envelope sinusoid plus noise, and those two
    have known and opposite fourth cumulants: arcsine at -1.5 and Gaussian at
    0. For independent contributions the FOURTH CUMULANTS ADD, so

        excess kurtosis = -1.5 * (carrier power / total power)^2

    and the share inverts directly. No spectrum, no model of the chain, and
    no assumption about where the carrier sits - which is what makes it worth
    having beside the spectral estimate rather than instead of it.

    Measured on the off-air capture, 2.1 M samples at 40 MSps: an excess
    kurtosis of -1.2954 gives a carrier share of 0.929, so the capture is
    92.9 per cent carrier and 7.1 per cent everything else. That is a
    carrier-to-noise ratio of 11.2 dB read off a single moment.

    Returns not-a-number where the kurtosis is outside the reachable range,
    which is the honest answer for a residual that is not a carrier plus
    noise at all.
    """
    kurtosis = excess_kurtosis(samples)
    if not np.isfinite(kurtosis) or kurtosis > 0 or kurtosis < -1.5:
        return {"carrier_share": float("nan"), "excess_kurtosis": kurtosis,
                "why": ("outside the range a carrier-plus-noise mixture can "
                        "produce, which is -1.5 to 0")}
    share = float(np.sqrt(kurtosis / EXCESS_KURTOSIS["arcsine"]))
    noise = 1.0 - share
    return {
        "carrier_share": share,
        "noise_share": noise,
        "carrier_to_noise_db": (10.0 * np.log10(share / noise)
                                if noise > 0 else float("inf")),
        "excess_kurtosis": kurtosis,
        "why": ("fourth cumulants add for independent contributions, and a "
                "constant-envelope carrier is -1.5 while noise is 0"),
    }


def for_component(name: str) -> Optional[Dict[str, object]]:
    """The distribution a named interference component draws from.

    Matches on a prefix so that the key's own entry names - which carry
    qualifiers like "(dark)" or "(playback)" - resolve to their mechanism.
    """
    if name in DISTRIBUTIONS:
        return DISTRIBUTIONS[name]
    for key, entry in DISTRIBUTIONS.items():
        if name.startswith(key):
            return entry
    return None
