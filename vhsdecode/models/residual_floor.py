"""ZERO, OR AN UNRECOVERABLE RESIDUAL: the test Ethan asked for, runnable.

Ethan: *"Given a knowable multi dimensional field of coefficients ... what
remains is truly random, and should be divided by 2, the final division over
the complex fully connected graph. I want to take this work so I can test it
and what the actual result is, will it be 0 or will there be an
unrecoverable residual."*

And, on the shape of the space the differentials live in: *"the tesseract
is asymmetric, and you never reach the full residual, since a hypercube has
a finite number of edges."* That second statement is given its own module,
`tesseract`, which measures it; this one measures the first.

WHAT THE TEST IS. A measured complex transfer `H(f)` with a per-bin standard
error is written in the log domain, where the chain's components add. The
model - every entry of the key, orthogonalised and admitted as
`interference.admission` leaves it - is projected out with REAL coefficients
on the stacked real and imaginary parts (the arc's rule: a Hermitian inner
product is blind to a quarter-turn, and that blindness has produced five
defects in this work). What remains is judged three ways, none of which is
"is it small":

  1. against the FLOOR: the reduced chi-square of the residual over the
     per-bin noise. One means the residual is the instrument's own noise;
     the error bar is root(2 / dof). Above one is excess, in decibels.
  2. WHITENESS across frequency: noise is uncorrelated from bin to bin,
     structure is not. The lag-one autocorrelation against its null of
     1/root(N), and a portmanteau sum over the first lags. This does not
     depend on the floor's absolute level, so it catches a wrong floor.
  3. DISTRIBUTION: `interference_distributions.classify` on the residual
     scaled to unit variance. Gaussian is the floor; a carrier's arcsine, a
     sparse or heavy tail, is structure with a name.

HELD OUT, WITH THE PARAMETERS FROZEN. The exports carry the response over
the first half of the fields and over the second half separately. The
coefficients are fitted on the first and applied unchanged to the second,
so a model that has learned its own noise is caught: its in-sample residual
falls below the floor while its held-out residual does not.

THE DIVISION BY TWO. Whatever is left above the floor after every knowable
component is removed is, by construction, not distinguishable from noise by
any dimension the model has. The best linear estimate of a signal of power
S buried in noise of power N is the Wiener gain S / (S + N), and its
residual error is S N / (S + N). At the point where the unexplained
structure equals the floor - S = N - the gain is exactly one half and the
error is half of either. That is the division by two: not a convention but
the optimum for a remainder about which nothing more is known. Below S = N
the gain falls further, and at S = 0 nothing is applied at all.

PREDICTION, WRITTEN BEFORE THE RUN. The half-split difference on the cd and
home exports is 2.2 to 5.1 times what four times the standard error
predicts (pnb: 0.8 to 1.3), so a slow term moves the response between the
halves of those two captures that the within-pool error does not see. The
held-out residual on cd and home is therefore expected NOT to reach the
fast floor for any frequency-axis key, because the key has no time axis;
it should sit near the slow floor. And the key's entries are smooth
mechanisms, while the remainder measured on dd's corrected fields was a
2.2 MHz ring, so the whiteness test is expected to fail on all three tapes:
an unrecoverable residual at this model order, with structure. The
measured answer follows.

THE MEASURED ANSWER (2026-09-05, sync exports, 0.2-4 MHz, fall view, first
half fitted and second half judged, every number a reduced chi-square over
the standard-error floor unless said):

    tape head   before   in-sample   held-out   lag-1 z   agreement
    cd   a      12238      1498        731        29.8      +0.884
    cd   b      13473      2383        912        29.9      +0.770
    home a      35784     15493       7502        29.9      +0.999
    home b      36107     17348       8399        30.0      +0.999
    pnb  a       2837       838        402        30.2      +0.993
    pnb  b       2122       536        259        30.3      +0.994

AND THE FLOOR THOSE FIGURES DIVIDE BY IS NOW ITSELF MEASURED, which changes
what they mean. `error_bar_audit` (2026-09-06) puts the export's quoted `se`
against the export's own independent-half estimates - a scatter WITHIN each
pool against a scatter BETWEEN two pools - and finds the disagreement running
with frequency rather than sitting at a factor: the ratio swings by two orders
of magnitude inside one measurement and crosses one in both directions, and
its rank correlation against frequency is negative on all twelve tape, head
and view combinations, a sign test at p = 2^-12. So the arc's two competing
numbers - the error bar overstated by 3.7 to 4.0 in variance on home, about 2
understated on countdown (`running_differential.is_constant_over_span`) - are
not two facts about two tapes but one fact about the instrument: the error bar
has no frequency dimension. A gain and a delay between the halves account for
a median of one per cent of the excess, so the "slow term" this module
predicted above is not either of the two terms `nuisance` carries.

Two consequences for the table above. The bins are oversampled fifty-fold
against the window's own 0.1812 MHz information spacing, so `verdict`'s
`error_bar` of root(2 / dof) counts 944 bins where 19 independent points
exist and is 7.0 times too small - the reduced chi-squares stand, the
confidence in them does not. And the "24 to 39 dB above the floor" is above a
floor that is wrong by a frequency-dependent amount, so it is a range and not
a number until the audit's shape is applied.

(the entry-wise admitted key of 55 entries; the base key of 14 is 1.6-2.4
times worse in sample). The answer to "zero or unrecoverable" is
UNRECOVERABLE WITH STRUCTURE on every tape and head: the residual after
the whole frequency-axis key is 24 to 39 dB above the standard-error floor,
it is not white (lag-one correlation thirty standard errors from zero),
and `structureless` finds it REPRODUCING on the half that did not build it
with an agreement of 0.77 to 1.00 - a component remains, by the arc's own
stopping rule. The prediction above held on both counts, and the tesseract
says what the component is not: not a filter-like shape on any axis, but
delays and all-pass terms (`tesseract`, `hypercomplex.causality`), so the
dimension the key lacks is the time axis. The division by two does not
arise yet - it applies to a remainder that does not reproduce, and this
one does.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from vhsdecode.models import information_extrapolation as extrapolation
from vhsdecode.models import interference
from vhsdecode.models import interference_distributions as distributions

# A complex measurement has two real parts, so a bin carries two degrees of
# freedom and a chi-square over N bins with p real parameters has 2N - p.
REAL_PARTS = 2

# Two independent halves of one pooled estimate: each half pools half the
# fields, so its variance is twice the pool's, and the variance of their
# difference is twice that again.
#
# AND THE ARC CARRIES TWO CONVENTIONS FOR `se`, EXACTLY TWO APART, WHICH IS
# THE SIZE OF SEVERAL OF THE DISCREPANCIES BEING ARGUED ABOUT. `log_domain`
# below reads `se` as the error of the complex estimate with both parts
# together, giving a difference variance of four; `tesseract.from_sync_exports`
# inserts a root two for the same export, and the instrument's own arithmetic
# (`sync_step_response`, `noise_power = 0.5 * sum(w^2 * derivative variance)`)
# matches the variance of ONE component, which makes the total twice `se^2`
# and the difference variance eight. The convention has never been pinned, so
# `error_bar_audit` takes it as an argument and reports both readings rather
# than choosing. Its SHAPE findings do not depend on the choice: a constant
# factor cannot make a ratio run with frequency or change sign inside one
# measurement.
HALF_DIFFERENCE_VARIANCE = 4.0
HALF_DIFFERENCE_VARIANCE_PER_COMPONENT = 2.0 * HALF_DIFFERENCE_VARIANCE

# The three-standard-error convention `whiteness` already uses for its
# lag-one null, named once so the audit and the whiteness test agree.
SIGMA_THRESHOLD = 3.0


def log_domain(H, se) -> Dict[str, np.ndarray]:
    """The complex log of a measured transfer, and its floor per bin.

    `se` is the standard error of the complex estimate with both parts
    together - calibrated on the pnb export, where the first-half minus
    second-half difference measures 0.82 to 1.25 times four `se` squared,
    which is what two independent halves of an estimate whose full-pool
    variance is `se` squared must give. The floor in the log domain is
    `se^2 / |H|^2`, the delta method on the complex logarithm.
    """
    H = np.asarray(H, dtype=np.complex128)
    se = np.asarray(se, dtype=np.float64)
    magnitude = np.maximum(np.abs(H), 1e-30)
    value = np.log(magnitude) + 1j * np.unwrap(np.angle(H))
    return {"log": value, "floor": (se / magnitude) ** 2}


def nuisance(frequency_hz) -> Dict[str, np.ndarray]:
    """A level, a phase reference and a delay: the terms that are not shapes.

    A constant in the log is a gain, a linear phase is a delay, and both
    belong to other estimators (levels to `standard_levels`, delay to the
    time base). They are fitted alongside the key so that neither is
    charged to the residual nor absorbed by an entry that resembles them.

    THE DELAY MUST PASS THROUGH ZERO FREQUENCY, and the first version of
    this did not. A true delay writes as `exp(-2 pi i f tau)`, whose phase
    is linear THROUGH THE ORIGIN; a column linear about the band's own
    centre differs from it by the constant `2 pi f_bar tau`, and with no
    imaginary constant anywhere in the basis that constant had nowhere to
    go and was charged to the residual. The repair is two columns rather
    than one: the delay taken through zero, and a free PHASE REFERENCE,
    which the unwrapping needs in any case since the branch it lands on is
    arbitrary.

    MEASURED, as the share of the departure the key explains, before and
    after the repair:

        countdown     72.6 -> 81.0 per cent
        home          46.8 -> 91.2
        pluge bars    67.0 -> 84.4

    On the home recording it nearly doubles, so this was not a refinement.
    Any figure computed against the earlier basis is not comparable with
    one computed against this.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    scale = float(np.max(np.abs(f))) or 1.0
    return {"level": np.ones(f.size, dtype=np.complex128),
            "phase reference": 1j * np.ones(f.size, dtype=np.complex128),
            "delay": 1j * f / scale}


def design(entries: Dict[str, np.ndarray], frequency_hz
           ) -> Dict[str, object]:
    """The regressors: each entry's log shape, plus the nuisance terms."""
    columns: List[np.ndarray] = []
    names: List[str] = []
    for name, value in nuisance(frequency_hz).items():
        columns.append(value)
        names.append(name)
    for name, value in entries.items():
        shape = interference._log_shape(np.asarray(value))
        if np.linalg.norm(shape) > 0:
            columns.append(shape)
            names.append(name)
    return {"matrix": np.column_stack(columns), "names": names}


def fit(log_value, floor, matrix) -> Dict[str, object]:
    """Weighted least squares with REAL coefficients on stacked parts."""
    w = 1.0 / np.sqrt(np.maximum(np.asarray(floor, dtype=np.float64),
                                 1e-300))
    A = np.vstack([np.real(matrix) * w[:, None],
                   np.imag(matrix) * w[:, None]])
    b = np.concatenate([np.real(log_value) * w, np.imag(log_value) * w])
    coefficients, _res, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    explained = matrix @ coefficients
    return {"coefficients": coefficients, "rank": int(rank),
            "explained": explained, "residual": log_value - explained}


def whiteness(residual, floor, lags: int = 8) -> Dict[str, float]:
    """Is what is left uncorrelated from bin to bin, as noise is?"""
    x = np.asarray(residual, dtype=np.complex128) / np.sqrt(
        np.maximum(np.asarray(floor, dtype=np.float64), 1e-300))
    x = x - x.mean()
    power = float(np.sum(np.abs(x) ** 2)) or 1e-300
    n = x.size
    rho = [complex(np.sum(x[:-k] * np.conj(x[k:])) / power)
           for k in range(1, min(lags, n - 1) + 1)]
    null = 1.0 / np.sqrt(max(n, 1))
    portmanteau = float(n * sum(abs(r) ** 2 for r in rho))
    return {"lag_one": float(abs(rho[0])) if rho else 0.0,
            "lag_one_null": float(null),
            "lag_one_z": float(abs(rho[0]) / null) if rho else 0.0,
            "portmanteau": portmanteau, "portmanteau_lags": len(rho),
            # a chi-square with `lags` degrees of freedom has mean `lags`
            # and standard deviation root(2 lags)
            "portmanteau_z": float((portmanteau - len(rho))
                                   / max(np.sqrt(2.0 * len(rho)), 1e-300)),
            "white": bool(abs(rho[0]) < 3 * null) if rho else True}


def verdict(residual, floor, parameters: int = 0) -> Dict[str, object]:
    """The three judgements on a residual, and the division by two."""
    r = np.asarray(residual, dtype=np.complex128)
    fl = np.maximum(np.asarray(floor, dtype=np.float64), 1e-300)
    n = r.size
    dof = max(REAL_PARTS * n - int(parameters), 1)
    chi2 = float(np.sum(np.abs(r) ** 2 / fl))
    reduced = chi2 / (dof / REAL_PARTS)
    error_bar = float(np.sqrt(2.0 / dof))
    excess = max(reduced - 1.0, 0.0)
    white = whiteness(r, fl)
    scaled = np.concatenate([np.real(r), np.imag(r)]) / np.sqrt(
        np.concatenate([fl, fl]) / REAL_PARTS)
    shape = distributions.classify(scaled)
    at_floor = bool(abs(reduced - 1.0) <= 3.0 * error_bar)
    structured = bool(not white["white"])
    if at_floor and not structured:
        reading = "zero: the residual is the instrument's own noise"
    elif at_floor and structured:
        reading = ("at the floor in power but not white: structure of the "
                   "floor's own size, unrecoverable by power alone")
    elif structured:
        reading = "unrecoverable residual with structure"
    else:
        reading = ("unrecoverable residual, white: the floor is larger "
                   "than the standard error says, a slow term or a wrong "
                   "noise estimate")
    return {
        "bins": int(n), "parameters": int(parameters), "dof": int(dof),
        "reduced_chi_square": float(reduced), "error_bar": error_bar,
        "excess_db": float(10.0 * np.log10(max(reduced, 1e-300))),
        "at_floor": at_floor, "whiteness": white,
        # `interference_distributions.classify` returns "distribution" and
        # "nearest"; asking it for "verdict" or "best" silently returned None
        # on every call this module has ever made, so the held-out pair had
        # nothing in this field to disagree about.
        "distribution": shape.get("distribution", shape.get("nearest")),
        "distribution_detail": shape,
        "half": divided_by_two(excess),
        "reading": reading,
    }


def divided_by_two(structured_excess: float) -> Dict[str, float]:
    """The Wiener treatment of the remainder, in units of the floor.

    `structured_excess` is S / N, the unexplained power over the floor.
    The gain that minimises the error is S / (S + N); its error is
    S N / (S + N). At S = N both are one half, which is the division by
    two: applying the whole remainder as a correction would double the
    error rather than halve it, because half of what is applied is noise.
    """
    s = max(float(structured_excess), 0.0)
    gain = s / (s + 1.0) if s > 0 else 0.0
    return {"structured_excess": s, "wiener_gain": gain,
            "error_after": s / (s + 1.0) if s > 0 else 0.0,
            "error_if_fully_applied": 1.0 if s > 0 else 0.0,
            "at_unity": bool(abs(s - 1.0) < 0.25)}


def resolution_cells(frequency_hz, resolution_hz) -> Dict[str, object]:
    """THE INSTRUMENT'S INDEPENDENT POINTS, WHICH ARE FEWER THAN ITS BINS.

    The sync export transforms a window of 79 samples (fall) or 124 (rise)
    on a 4096-point grid, so its bins are finer than the information it
    carries. The export says so in its own metadata: *"resolution_mhz per
    polarity is the true information spacing - bins are finer"*. Adjacent
    bins are therefore not independent draws, and a chi-square that counts
    them as if they were quotes an error bar root(bins per cell) times too
    small.

    MEASURED (2026-09-06, the three `ss_*_off` sync exports, 0.2-4 MHz):
    944 valid bins carry 19 independent points on the fall view (0.1812 MHz
    spacing) and 29 on the rise (0.1155 MHz) - an oversampling of 50 and
    33, so `verdict`'s `error_bar` of root(2 / dof) is 7.0 and 5.7 times too
    small on those views. The half-split difference confirms the
    oversampling directly rather than by assertion: its bin-to-bin lag-one
    correlation measures 0.998 to 0.9996 on all twelve tape/head/view
    combinations, which is what a fiftyfold oversampled band-limited noise
    must give and what independent bins could not.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    width = float(resolution_hz)
    if width <= 0:
        raise ValueError("the resolution must be positive; it is the "
                         "instrument's own information spacing")
    index = np.floor((f - f.min()) / width).astype(int)
    occupied = np.bincount(index)
    keep = occupied > 0
    return {"index": index, "count": int(keep.sum()),
            "bins": int(f.size), "resolution_hz": width,
            "oversampling": float(f.size) / max(int(keep.sum()), 1),
            "why": ("a chi-square counting bins rather than independent "
                    "points understates its error bar by the root of this")}


def _average_by_cell(index, values, cells: int):
    """The mean of `values` in each occupied resolution cell."""
    counts = np.bincount(index, minlength=cells).astype(np.float64)
    totals = np.bincount(index, weights=np.asarray(values, dtype=np.float64),
                         minlength=cells)
    keep = counts > 0
    return totals[keep] / counts[keep], keep


def _rank(values) -> np.ndarray:
    """Ranks with ties averaged, for a distribution-free trend test."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(x, return_inverse=True,
                                        return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks, minlength=unique.size)
        ranks = (sums / counts)[inverse]
    return ranks


def _strip(difference, columns, predicted):
    """Remove the given complex shapes with real coefficients, weighted by
    the predicted noise - the arc's rule that a Hermitian inner product is
    blind to a quarter-turn, so the parts are stacked."""
    if not columns:
        return np.asarray(difference)
    A = np.column_stack(columns)
    w = 1.0 / np.sqrt(np.maximum(predicted, 1e-300))
    M = np.vstack([np.real(A) * w[:, None], np.imag(A) * w[:, None]])
    b = np.concatenate([np.real(difference) * w, np.imag(difference) * w])
    coefficients, _r, _k, _s = np.linalg.lstsq(M, b, rcond=None)
    return difference - A @ coefficients


def error_bar_audit(frequency_hz, H, se, first, second, resolution_hz,
                    variance_factor: float = HALF_DIFFERENCE_VARIANCE
                    ) -> Dict[str, object]:
    """THE INSTRUMENT'S ERROR BAR, MEASURED AGAINST A SECOND MEASUREMENT.

    Every "times the floor" figure in this arc divides by the export's
    per-bin standard error, and that error bar had never itself been
    measured: `log_domain` calibrated it on the pnb export alone, and
    `running_differential.is_constant_over_span` found it disagreeing with
    the field-axis lag-one difference by 3.7 to 4.0 in variance on home and
    about 2 the other way on countdown, closing with *"the instrument's
    error bar needs its own audit"*. This is that audit.

    THE PAIR. The export carries two measurements of the same transfer that
    can disagree: the quoted `se`, a cross-line variance of the accumulated
    mean, and `H_first`/`H_second`, two independent-half estimates. Two
    halves of a pooled estimate whose variance is `se^2` differ by
    `variance_factor * se^2`, so the ratio of the measured difference power
    to that prediction is one when the error bar is right. The factor is an
    argument and not a decision: the tree carries two conventions for `se`
    exactly two apart (see `HALF_DIFFERENCE_VARIANCE`), and both readings are
    returned. NONE OF THE SHAPE FINDINGS BELOW DEPEND ON WHICH IS RIGHT - a
    constant factor cannot make a ratio run with frequency, and cannot make it
    cross one in opposite directions inside a single measurement.
    The two are independent in the way that matters: the quoted error bar is
    a scatter WITHIN each pool and the half-difference is a scatter BETWEEN
    two pools, so a noise the pooling shares is invisible to the first and
    plain in the second.

    WHAT IS MEASURED, AND IT IS NOT A SINGLE FACTOR (2026-09-06, the three
    `ss_*_off` sync exports, 0.2-4 MHz, both heads and both polarities, the
    ratio above per resolution cell):

        band (MHz)     cd            home          pnb
        0.05-0.2       0.03 - 6.00   0.25 - 0.54   0.07 - 0.22
        0.5-1.0        1.26 - 7.19   7.02 - 14.34  0.79 - 2.37
        1.0-2.0        1.65 - 8.48   2.96 - 12.34  2.18 - 3.36
        3.0-3.5        0.11 - 1.37   0.16 - 2.20   0.01 - 0.15

    The discrepancy swings by two orders of magnitude ACROSS FREQUENCY
    inside one and the same measurement, and it changes sign: the quoted
    error bar is too small in the middle of the band and too large at both
    ends. The rank correlation of the per-cell ratio against frequency is
    NEGATIVE on all twelve tape/head/view combinations (rho -0.01 to -0.65),
    a sign test at p = 2^-12 = 0.00024. So the arc's competing single
    numbers - overstated by 3.7 on home, understated by 2 on countdown - are
    not two facts about two tapes but one fact about the instrument: THE
    ERROR BAR HAS NO FREQUENCY DIMENSION, and a pooled ratio reports
    wherever the band weighting happens to land.

    WHAT IT IS NOT. The excess is not a slow gain or timing drift between
    the halves, which is how `residual_floor`'s own prediction read it. A
    gain change is `H` and a delay change is `j f H`; projecting both out
    removes 0 to 30 per cent of the excess, a median of 1 per cent, so
    whatever moves between the halves is a SHAPE and not the two terms
    `nuisance` already carries.

    THE PHASE DIMENSION, AND IT COULD NOT BE SETTLED. A single real `se` per
    bin asserts that the complex noise is circular - equal variance along
    and across `H`. Rotating the difference by `H`'s own phase splits it
    into a radial part (an amplitude error) and a tangential one (a phase or
    timing error); the measured variance ratio runs 0.70 to 2.06. At the
    honest degrees of freedom (`resolution_cells`, 19 to 29, not 944) that
    is at most 1.6 standard errors and never reaches three, so the
    circularity assumption is NOT falsified by this evidence - but neither
    is it confirmed, and the quantity is reported because it is the axis on
    which a single real `se` is silent. Counting the 944 bins instead would
    have called the same 2.06 significant, which is what the oversampling
    correction is for.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.complex128).ravel()
    se = np.asarray(se, dtype=np.float64).ravel()
    difference = (np.asarray(first, dtype=np.complex128).ravel()
                  - np.asarray(second, dtype=np.complex128).ravel())
    if not (f.size == H.size == se.size == difference.size):
        raise ValueError("the frequency grid, the transfer, the error bar "
                         "and the two halves must share one grid")
    predicted = float(variance_factor) * se ** 2
    power = np.abs(difference) ** 2

    cells = resolution_cells(f, resolution_hz)
    index = cells["index"]
    size = int(index.max()) + 1
    measured_cell, keep = _average_by_cell(index, power, size)
    predicted_cell, _ = _average_by_cell(index, predicted, size)
    centre_cell, _ = _average_by_cell(index, f, size)
    n = int(measured_cell.size)
    if n < 3:
        raise ValueError("too few independent points to audit an error bar")
    ratio_cell = measured_cell / np.maximum(predicted_cell, 1e-300)

    pooled = float(np.sum(power) / max(np.sum(predicted), 1e-300))
    # a trend across frequency is what a missing frequency dimension looks
    # like; the rank correlation needs no model of the shape
    rank_f, rank_r = _rank(centre_cell), _rank(ratio_cell)
    rho = float(np.corrcoef(rank_f, rank_r)[0, 1]) if n > 2 else 0.0
    trend_z = float(rho * np.sqrt(max(n - 1, 1)))
    split = n // 2
    low = float(np.mean(ratio_cell[:split]))
    high = float(np.mean(ratio_cell[split:]))

    # a gain change is H and a delay change is j f H: the only two smooth
    # terms `nuisance` carries, expressed in the linear domain
    centred = f - f.mean()
    scale = float(np.max(np.abs(centred))) or 1.0
    stripped = _strip(difference, [H, 1j * (centred / scale) * H], predicted)
    drift_share = float(1.0 - np.sum(np.abs(stripped) ** 2)
                        / max(np.sum(power), 1e-300))

    # the complex difference carries a phase, so it splits into an amplitude
    # error and a timing one; a single real `se` says these are equal
    unit = H / np.maximum(np.abs(H), 1e-300)
    rotated = difference * np.conj(unit)
    radial = float(np.var(rotated.real))
    tangential = float(np.var(rotated.imag))
    anisotropy = radial / max(tangential, 1e-300)
    # log F with n degrees of freedom on each side has standard deviation
    # root(2/n + 2/n)
    anisotropy_z = float(np.log(max(anisotropy, 1e-300))
                         / max(2.0 / np.sqrt(n), 1e-300))

    quoted_dof = REAL_PARTS * int(f.size)
    honest_dof = REAL_PARTS * n
    # a shape smaller than the convention ambiguity itself cannot be told
    # from a units error, so that ratio - and not a chosen number - is the
    # bar the band contrast must clear
    ambiguity = (HALF_DIFFERENCE_VARIANCE_PER_COMPONENT
                 / HALF_DIFFERENCE_VARIANCE)
    frequency_dependent = bool(abs(trend_z) > SIGMA_THRESHOLD
                               or (min(low, high) > 0
                                   and max(low, high) / min(low, high)
                                   > ambiguity))
    if frequency_dependent:
        reading = ("the error bar is wrong in SHAPE, not by a factor: its "
                   "ratio to the measured half-difference runs with "
                   "frequency, so no single correction is right")
    elif abs(np.log(max(pooled, 1e-300))) > SIGMA_THRESHOLD * 2.0 / np.sqrt(n):
        reading = ("the error bar is wrong by a factor of "
                   f"{pooled:.2f} in variance and the shape is flat, so a "
                   "single rescaling would correct it")
    else:
        reading = "the error bar agrees with the second measurement"
    return {
        "independent_points": n, "bins": int(f.size),
        "oversampling": cells["oversampling"],
        "quoted_dof": quoted_dof, "honest_dof": honest_dof,
        "error_bar_understated_by": float(np.sqrt(quoted_dof
                                                  / max(honest_dof, 1))),
        "pooled_ratio": pooled,
        "variance_factor": float(variance_factor),
        "pooled_ratio_other_convention": float(
            pooled * variance_factor
            / (HALF_DIFFERENCE_VARIANCE_PER_COMPONENT
               if variance_factor == HALF_DIFFERENCE_VARIANCE
               else HALF_DIFFERENCE_VARIANCE)),
        "se_should_scale_by": float(np.sqrt(max(pooled, 0.0))),
        "cell_frequency_hz": centre_cell, "cell_ratio": ratio_cell,
        "low_band_ratio": low, "high_band_ratio": high,
        "trend_rho": rho, "trend_z": trend_z,
        "band_contrast": float(max(low, high) / max(min(low, high), 1e-300)),
        "convention_ambiguity": float(ambiguity),
        "frequency_dependent": frequency_dependent,
        "drift_share": drift_share,
        "radial_variance": radial, "tangential_variance": tangential,
        "anisotropy": anisotropy, "anisotropy_z": anisotropy_z,
        "circular": bool(abs(anisotropy_z) <= SIGMA_THRESHOLD),
        "reading": reading,
        "why": ("the quoted error bar is a scatter within each pool and the "
                "half-difference is a scatter between two pools, so they "
                "are two measurements of one noise that can disagree"),
    }


def test(frequency_hz, H, se, entries: Dict[str, np.ndarray],
         held_out: Optional[Dict[str, np.ndarray]] = None
         ) -> Dict[str, object]:
    """Run the whole test on one measured transfer.

    `held_out`, when given, carries `H_fit`, `H_judge` and `se_half`: the
    coefficients are fitted on `H_fit` and the residual judged on
    `H_judge`, both halves of the same measurement, each with the half's
    standard error - which is root two times the full pool's.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    reg = design(entries, f)
    full = log_domain(H, se)
    fitted = fit(full["log"], full["floor"], reg["matrix"])
    p = reg["matrix"].shape[1]
    out = {
        "entries": len(entries), "parameters": p,
        "in_sample": verdict(fitted["residual"], full["floor"], p),
        "before": verdict(full["log"] - fit(
            full["log"], full["floor"],
            reg["matrix"][:, :len(nuisance(f))])["explained"],
            full["floor"], len(nuisance(f))),
        "coefficients": dict(zip(reg["names"], fitted["coefficients"])),
    }
    if held_out is not None:
        first = log_domain(held_out["H_fit"], held_out["se_half"])
        second = log_domain(held_out["H_judge"], held_out["se_half"])
        frozen = fit(first["log"], first["floor"], reg["matrix"])
        # the nuisance terms are re-fitted on the judged half: a level and a
        # delay are not the model's claim, and freezing them would charge
        # the second half's own gain and timing to the residual
        judged = second["log"] - reg["matrix"] @ frozen["coefficients"]
        nuisance_columns = reg["matrix"][:, :len(nuisance(f))]
        judged = fit(judged, second["floor"], nuisance_columns)["residual"]
        out["held_out"] = verdict(judged, second["floor"],
                                  len(nuisance(f)))
        out["held_out"]["fast_floor_ratio"] = float(
            np.mean(np.abs(judged) ** 2) / np.mean(second["floor"]))
        # THE STOPPING CONDITION THE ARC ALREADY HAS. Ethan: "we have all
        # this already, look at the code". `structureless` asks the
        # decisive question - does the residual REPRODUCE on evidence that
        # did not build it? - and that is what separates an unrecoverable
        # residual (structure, which reproduces) from the floor (noise,
        # which does not), whatever the size of what remains.
        own = fit(first["log"] - reg["matrix"] @ frozen["coefficients"],
                  first["floor"], nuisance_columns)["residual"]
        scale = np.sqrt(np.concatenate([first["floor"], first["floor"]]) / 2)
        stacked_first = np.concatenate([own.real, own.imag]) / scale
        stacked_second = np.concatenate([judged.real, judged.imag]) / scale
        out["structureless"] = extrapolation.structureless(
            stacked_first, stacked_second)
    return out


def load_export(path: str, head: str, band_hz=(0.2e6, 4.0e6),
                view: str = "fall") -> Dict[str, object]:
    """One head's measured transfer from a `sync_step_response` export."""
    data = np.load(path, allow_pickle=True)
    f = np.asarray(data["frequency_mhz"], dtype=np.float64) * 1e6
    base = f"head_{head}_{view}"
    valid = np.asarray(data[f"{base}_valid"], dtype=bool)
    valid &= (f >= band_hz[0]) & (f <= band_hz[1])
    H = np.asarray(data[f"{base}_H"], dtype=np.complex128)
    se = np.asarray(data[f"{base}_se"], dtype=np.float64)
    first = data.get(f"{base}_H_first")
    second = data.get(f"{base}_H_second")
    for extra in (first, second):
        if extra is not None:
            valid &= np.isfinite(np.asarray(extra))
    valid &= np.isfinite(H) & np.isfinite(se) & (se > 0)
    out = {"frequency_hz": f[valid], "H": H[valid], "se": se[valid],
           "valid": valid, "view": view}
    # the true information spacing, which is far coarser than the bin
    # spacing and is what `resolution_cells` counts by
    resolution = data.get(f"{base}_resolution_mhz")
    if resolution is not None:
        out["resolution_hz"] = float(np.asarray(resolution)) * 1e6
    if first is not None and second is not None:
        out["held_out"] = {
            "H_fit": np.asarray(first, dtype=np.complex128)[valid],
            "H_judge": np.asarray(second, dtype=np.complex128)[valid],
            # each half pools half the fields: root-two the full error
            "se_half": se[valid] * np.sqrt(2.0),
        }
    return out


def keys_for(frequency_hz, mechanics=None) -> Dict[str, Dict[str, np.ndarray]]:
    """The base key and the entry-wise admitted key on this grid."""
    base = interference.signatures(frequency_hz, mechanics=mechanics)
    admitted = interference.admission(frequency_hz, mechanics=mechanics,
                                      granularity="entry")
    full = interference.key_after_admission(frequency_hz, admitted,
                                            mechanics=mechanics)
    return {"base": base, "admitted": full, "admission": admitted}


# --------------------------------------------------------------------------
# R5's declared floor
# --------------------------------------------------------------------------
#
# "The limit terminates at the noise floor, the floor is DECLARED rather
# than assumed, and it is measured OUT OF SAMPLE" - rule R5 of
# docs/RESIDUAL_LIMIT_DESIGN.md, which is the stopping rule of the whole
# method: "'No change at all in the residual' is the stopping rule. What
# 'no change' means has to be a measured quantity: the bins' own standard
# error, 1.2533 x MAD / sqrt(n) per bin. A residual above that which stops
# moving is a stalled sequence, not a converged one."
#
# IT WAS IMPLEMENTED TWICE AND ASSERTED NOWHERE. `debug_plot.py` computes
# it for the response trace and `noise_budget.py` for a density's
# uncertainty, each with its own copy of the numbers, and no test held
# either to the relation. For the rule that decides when the algorithm has
# finished, that is the wrong place for it to live.
#
# Both factors are DERIVED here rather than typed, which the project's own
# standard requires and which also documents what they are.

def _normal_quantile(p: float) -> float:
    """The standard normal's inverse CDF, so the constants below are
    derived rather than quoted."""
    from scipy.stats import norm
    return float(norm.ppf(p))


MAD_TO_SIGMA = 1.0 / _normal_quantile(0.75)
"""1.4826. The median absolute deviation of a normal is Phi^-1(0.75) sigma
= 0.6745 sigma, so this is its reciprocal. Robust because it is set by the
middle of the distribution: dropouts sit in the tails and a plain standard
deviation lets them set the scale."""

IQR_TO_SIGMA = 1.0 / (_normal_quantile(0.75) - _normal_quantile(0.25))
"""1/1.349. The interquartile range of a normal is 1.349 sigma."""

MEDIAN_STANDARD_ERROR_FACTOR = float(np.sqrt(np.pi / 2.0))
"""1.2533 = sqrt(pi/2). The median of n normal samples has standard error
sqrt(pi/2) sigma / sqrt(n) - LARGER than the mean's sigma/sqrt(n), by 25
per cent, which is the price of using a robust centre. A floor that quotes
the mean's error while the estimator takes a median is 25 per cent too
low, and a residual sitting between the two reads as converged when it is
not."""


def robust_sigma(values, method: str = "mad") -> float:
    """The scale, set by the middle of the distribution rather than its
    tails - which is what makes it usable on real lines, where dropouts
    are a low tail that would otherwise define the noise."""
    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return float("nan")
    if method == "iqr":
        low, high = np.percentile(data, [25.0, 75.0])
        return float((high - low) * IQR_TO_SIGMA)
    return float(np.median(np.abs(data - np.median(data))) * MAD_TO_SIGMA)


def declared_floor(values, count: Optional[int] = None,
                   method: str = "mad") -> Dict[str, float]:
    """R5's floor for one bin: what "no change in the residual" means here.

    `count` overrides the sample count when the values passed are already
    an average of that many - the floor belongs to the ESTIMATE, and an
    estimate accumulated over N lines has a floor root-N below one line's.
    Getting this wrong by using the per-line scatter for an accumulated
    profile overstates the floor by exactly root N, which in this arc was
    a factor of 45 and stopped a derivation early.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    n = int(len(data) if count is None else count)
    sigma = robust_sigma(data, method)
    if n < 1 or not np.isfinite(sigma):
        return {"sigma": sigma, "count": n, "floor": float("nan"),
                "mean_floor": float("nan")}
    return {
        "sigma": sigma,
        "count": n,
        "floor": float(MEDIAN_STANDARD_ERROR_FACTOR * sigma / np.sqrt(n)),
        # what it would be for a mean, carried so the 25 per cent is
        # visible rather than implicit
        "mean_floor": float(sigma / np.sqrt(n)),
    }


def at_declared_floor(residual, values, count: Optional[int] = None,
                      method: str = "mad") -> Dict[str, object]:
    """Whether a residual has reached R5's floor, with the margin.

    Reports the RATIO as well as the verdict, because "eight to twelve
    times the floor" is the statement this arc has needed over and over,
    and a bare boolean throws it away.
    """
    floor = declared_floor(values, count, method)
    level = float(np.sqrt(np.mean(np.asarray(residual, dtype=np.float64) ** 2)))
    ratio = level / floor["floor"] if floor["floor"] > 0 else float("inf")
    return {**floor, "residual_rms": level, "ratio": ratio,
            "at_floor": bool(ratio <= 1.0)}
