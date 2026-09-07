"""ONE TRANSFORM, ALL DIMENSIONS AT ONCE, AND THE NULL SPACE TREATED BOTH WAYS.

Ethan, 2026-09-07:

    "I think I can do much less now than all the stages. I can make a
    singla picture stage that transforms luma chroma all up to the
    composite functions. There doesn't need to be sequencing, just all the
    dimensions execute at once in a single transform."

    "Additionally, I need to have a VHS RF stage transformed the same way.
    We do the full spherical shape, but, replace the noise with null
    space, which is a constant that we do not derive at all. We are
    extracting the signal we care about."

    "Those edges, which are null space do not get computed and therefore
    save computation."

    "Is is the concentric rings of dimensions, like a multi fold sphere
    that has a causality dimension fixed, since we process video RF data"

    "What if you differential down the null space out of this signal,
    rather than just add null space as the constant"

    "Essentially removing each indifvidual residual, instead of just
    substituting in null space, let's try both out."

    "I think there is just a generalized model for rf null space though,
    where we don't want the data to exist, like the sinc function, or it's
    counter part at the number of dimensions, since we are representing
    sinewaves" ... "Perhaps an edge that represents exactly the cutoff for
    the input data, i.e. 20mhz of possible data in a 40mhz file. A
    hypercube" ... "That is a constant I think for the rf file itself" ...
    "Perhaps that's how we remove the noise and keep the image" ...
    "Subtract out the hyper cube" ... "and discard it as noise"

    "This becomes a data transformation problem using our instruments"

    "Let's see what happens if we take this to hyperspace as the null space
    model, the exact inverse within our possible area of measure."

    "I believe there is still a wave underneath the convergence that itself
    can be differentialed as random noise and subtracted out, if we don't
    already have that"

WHY A SEQUENCE WAS NEVER NEEDED. Forty stages exist because each
measurement was built when it was understood and a pipeline was written to
order them. But this arc established early that THE FOLDS COMMUTE, and
operations that commute do not need ordering. What they need is one
transform. The instruments are the measurements, the fold is the
transformation, and the realisations write what remains.

LUMA AND CHROMA ARE ONE AXIS, NOT TWO STAGES. A fold takes a pair of
vertices to their PARENT, the mean, and their DIFFERENTIAL, half the
difference. Composite video is luma plus chroma, so the composite IS the
parent of that fold and the separation is its differential. The same holds
on the radio frequency, where the luminance frequency modulation and the
colour-under are the pair and the recorded signal is their parent. For
that to be an axis, what a vertex carries cannot itself be "luma" or
"chroma": it is the same three QUANTITIES at every vertex, a response, a
level series and a timing, concatenated as channels on one grid.

THE RINGS. With n axes the fold returns C(n, k) contrasts at grade k: the
grand mean at the centre, the single-axis differentials on the first ring,
the pairwise interactions on the second, out to the n-way. That grading is
the concentric rings, and it is the grading of the Clifford algebra whose
2^n components the fold is. The causal axis, frequency, is the one that is
NOT folded: the recorded radio frequency is causal, so every contrast
splits into a delay, a minimum-phase part fixed by its own magnitude, and
an all-pass excess, and each is inverted in its own form.

THE NULL SPACE, THREE WAYS, ALL THREE REPORTED. The kept set is a
declaration from the physics: a contrast is kept where a mechanism produces
it. What happens to the rest is Ethan's question above, and the two
answers are built side by side. SUBSTITUTE never computes a null contrast:
the pruned fold declines to descend where nothing below is kept, and the
correction is the vertex's departure rebuilt from the kept contrasts alone.
REMOVE folds everything and treats every contrast as a residual in its own
right, subtracted by the amount it REPRODUCES across two banks of fields -
the R5 agreement gate the decode already runs - so a residual describing
the path is removed in full and one describing the field it came from is
left alone, with no constant chosen anywhere. The third form is the
capture's own HYPERCUBE: the sample rate fixes where data can exist at all
and the format fixes where the signal may exist inside that box; the rest
is null space by construction and signal-free, so its noise density is
measured with no judge and subtracted from every contrast as the constant
it is. The three floors are reported together and must agree; where the
in-band floor stands above the file's constant, the difference is a
mechanism to name and not noise to discard.

THE WAVE UNDER THE CONVERGENCE. The latch is the constant the constants
rule asks for, and what is left after it, field by field, is not zero: on
the first real decode the response's in-band floor stood about eight
hundred times above the capture's own constant. That remainder is a series
over field time, and field time folds like every other axis - as the bits
of the field index (`tesseract.from_field_series`), where the drum, the
guides and the reels have already been found as lines offline. `Wave`
records each field's remainder after the latch, folds the run, judges each
time-scale contrast by whether the second half of the run reproduces what
the first half read, subtracts the part that reproduces per field, and
discards the rest as the random noise it is. The latched constant does not
move; the wave is a separate term with its own admission.

WHAT SUBSTITUTE COSTS, because it is not free: with no noise model there is
no whitening, so the projection is unweighted, and a component's admission
is a geometric verdict rather than a statistical one. REMOVE restores the
error bar from the data itself. Which is better is decided by the held-out
remainder on real decodes and recorded with the number both ways.
"""

import math
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from vhsdecode.models import hypercomplex, tesseract
from vhsdecode.models.chroma_head_switch import DETECTION_SIGMA
from vhsdecode.residual_limit import agreement

Subset = Tuple[str, ...]


# The axes each transform folds on, and what each one's differential MEANS.
# A contrast is kept only where a mechanism produces it, and the mechanism
# is named here rather than discovered by fitting - which is what makes the
# null space a declaration rather than a threshold.
PICTURE_AXES: Dict[str, str] = {
    "colour": "luma against chroma; the parent is composite video itself",
    "head": "the drum's two heads, which are two different paths",
    "polarity": "the pulse's fall against its rise, two landings",
    "field": "the two fields of a frame, which differ by a half line",
}
RF_AXES: Dict[str, str] = {
    "band": "the luma frequency modulation against the colour-under; the "
            "parent is the recorded signal",
    "head": "the drum's two heads, which are two different magnetic paths",
    "polarity": "the carrier's deviation above against below blanking",
    "tap": "the record tap against the playback tap",
}

# THE AXES ONE DECODE CANNOT REACH, and why. Head is field parity - the drum
# lays one field per half revolution, and `bool(field.isFirstField)` is the
# only head label a decode has - so the picture's field axis is the head
# axis under another name. A capture is taken at one tap. Neither is null
# space: each is a KNOWN CONSTANT of the decode (the constants rule), with
# one state present, and the transform carries the three live axes and
# reports these two as unreached.
UNREACHED: Dict[str, Dict[str, str]] = {
    "picture": {"field": "field parity IS the head label in one decode; the "
                         "two coincide and the axis has one state"},
    "rf": {"tap": "a capture is taken at one tap; the other tap exists "
                  "only as another capture, measured offline"},
}
LIVE_AXES: Dict[str, Subset] = {
    "picture": ("colour", "head", "polarity"),
    "rf": ("band", "head", "polarity"),
}

# The contrasts a mechanism actually produces, by order. Order zero is the
# grand mean and is always signal; the single-axis differentials are the
# mechanisms named above; a pairwise contrast is signal only where two
# mechanisms genuinely interact. Polarity is null at radio frequency because
# the channel ahead of the demodulator is linear and a rise cannot differ
# from a fall there; it is a picture quantity.
PICTURE_SIGNAL: Tuple[Subset, ...] = (
    (),
    ("colour",),
    ("head",),
    ("polarity",),
    ("colour", "head"),
    # ONLY THE LUMA HAS LANDINGS. The chroma is a carrier whose burst has no
    # fall or rise, so its two polarity faces carry one value and the whole
    # polarity mechanism is colour-specific: the fold puts half of the
    # luma's fall-to-rise difference into the polarity contrast and the
    # other half here. Measured with this contrast declared null, a planted
    # 2 IRE sync-depth deficit was realised at half under SUBSTITUTE (0.5
    # IRE left per face) and in full under REMOVE (0.0003 IRE), which is
    # the declaration being wrong, not the judge.
    ("colour", "polarity"),
)
RF_SIGNAL: Tuple[Subset, ...] = (
    (),
    ("band",),
    ("head",),
    ("tap",),
    ("band", "head"),
    ("band", "tap"),
)

TREATMENTS: Tuple[str, ...] = ("substitute", "remove")

# The normal consistency constant: a median absolute deviation is 0.6745 of
# a Gaussian's sigma, so sigma is 1.4826 times it (Rousseeuw and Croux,
# "Alternatives to the median absolute deviation", 1993). Used only to put
# the detection sigma `chroma_head_switch` already declares onto a
# running median of steps.
MAD_TO_SIGMA = 1.4826
# How many accepted fills a vertex needs before its running median of steps
# means anything; below it every fill is accepted. Four is the decision
# count the stages already use (`model_stages._LOCK_DECISION_FIELDS`).
STEPS_BEFORE_GATING = 4
STEP_HISTORY = 32


def declared(kind: str = "picture") -> Dict[str, object]:
    """The axes and the kept contrasts for one transform, with the reasons.

    The kept set is a DECLARATION from the physics, not a threshold on the
    data. That is the whole difference between this and fitting: a contrast
    absent from the list is null space because no mechanism produces it,
    and so it is never computed rather than computed and then rejected.

    The declaration names all four axes; `live` and `unreached` say which
    of them one decode can fold and which it carries as a known constant.
    """
    if kind not in ("picture", "rf"):
        raise ValueError("kind is 'picture' or 'rf', not %r" % (kind,))
    axes = PICTURE_AXES if kind == "picture" else RF_AXES
    signal = PICTURE_SIGNAL if kind == "picture" else RF_SIGNAL
    order = tuple(axes)
    kept = {tuple(sorted(s, key=order.index)) for s in signal}
    every = set()
    for size in range(len(order) + 1):
        for combination in _combinations(order, size):
            every.add(combination)
    live = LIVE_AXES[kind]
    return {
        "kind": kind,
        "axes": order,
        "meanings": dict(axes),
        "kept": kept,
        "null": every - kept,
        "total": len(every),
        "computed": len(kept),
        "live": live,
        "unreached": dict(UNREACHED[kind]),
        "live_kept": live_kept(kind, live),
        "rings": rings(order, kept),
        "fixed_axis": ("frequency, the causal axis: never folded; each "
                       "contrast on it splits into a delay, a minimum phase "
                       "and an all-pass"),
        "why": ("a contrast is kept where a mechanism produces it; the "
                "rest is null space and is never computed"),
    }


def rings(axes: Sequence[str], kept: Iterable[Subset]) -> Dict[int, Dict[str, object]]:
    """THE CONCENTRIC RINGS. Ethan: "Is is the concentric rings of
    dimensions, like a multi fold sphere that has a causality dimension
    fixed, since we process video RF data."

    Ring k holds the C(n, k) contrasts of grade k: the grand mean at the
    centre, the single-axis differentials on the first ring, the pairwise
    interactions on the second, out to the n-way. Each ring is listed with
    the contrasts kept on it and those that are null, so a reading shows
    the sphere fold by fold; the causal axis is not among them because it
    is the fixed one.
    """
    names = tuple(axes)
    n = len(names)
    kept_set = {tuple(sorted(s, key=names.index)) for s in kept}
    out: Dict[int, Dict[str, object]] = {}
    for grade in range(n + 1):
        members = [c for c in _combinations(names, grade)]
        out[grade] = {
            "contrasts": [":".join(c) or "mean" for c in members],
            "kept": [":".join(c) or "mean" for c in members if c in kept_set],
            "null": [":".join(c) or "mean" for c in members if c not in kept_set],
            "count": len(members),
        }
    return out


def live_kept(kind: str, axes: Sequence[str]) -> Set[Subset]:
    """The kept contrasts restricted to the axes a cube actually carries.

    A kept subset naming an axis the cube lacks - the field axis, which is
    the head axis in one decode, or the tap - is not null and not kept: it
    is unreachable, and `prune` would raise on it. It is dropped here and
    reported by `declared` under `unreached`.
    """
    signal = PICTURE_SIGNAL if kind == "picture" else RF_SIGNAL
    names = tuple(axes)
    out = set()
    for subset in signal:
        if all(axis in names for axis in subset):
            out.add(tuple(sorted(subset, key=names.index)))
    return out


def _combinations(items: Sequence[str], size: int) -> Iterable[Subset]:
    from itertools import combinations
    return combinations(items, size)


def fold_order(axes: Sequence[str], kept: Set[Subset]) -> Subset:
    """The order to fold in, so the null branches are dropped while they
    are still expensive.

    THE TREE'S COST IS AT ITS TOP. The first fold acts on the whole cube
    and every later one on something half the size, so a branch pruned at
    the bottom saves almost nothing while a branch pruned at the top saves
    half the remaining work. Measured before this ordering existed, pruning
    to four contrasts of sixteen ran only 1.33 times faster at four axes
    and 0.89 times - slower - at six, because the pruning all happened
    where the arrays were already small.

    So the axes are folded in order of how much of the tree they kill: an
    axis that appears in NO kept contrast has its whole differential half
    discarded immediately and goes first, and an axis every kept contrast
    needs goes last.
    """
    names = tuple(axes)
    demand = {axis: sum(1 for subset in kept if axis in subset)
              for axis in names}
    return tuple(sorted(names, key=lambda a: (demand[a], names.index(a))))


def prune(cube: tesseract.Cube, kept: Set[Subset]) -> Dict[Subset, Dict[str, np.ndarray]]:
    """THE FOLD WITH THE NULL BRANCHES NEVER TAKEN.

    `tesseract.walsh` folds every branch to the bottom and returns all 2^n
    contrasts. This walks the same tree and declines to descend where
    nothing below is kept, so the arithmetic is proportional to the
    contrasts asked for rather than to 2^n.

    The saving is real because the fold is a tree: a branch whose entire
    subtree is null costs nothing at all, not merely nothing at the leaf.

    The keys come back in the CUBE'S axis order, the same order `walsh`
    uses. The first version keyed them in fold order, which put the radio
    frequency's `("band", "head")` under `("head", "band")` where a
    consumer looking it up by the declaration would have missed it.
    """
    for subset in kept:
        for axis in subset:
            if axis not in cube.axes:
                raise ValueError("kept contrast %r names an axis the cube "
                                 "does not carry; use live_kept" % (subset,))
    order = fold_order(cube.axes, kept)
    wanted = {tuple(sorted(s, key=order.index)) for s in kept}
    products: Dict[Subset, tesseract.Cube] = {(): cube}
    for depth, axis in enumerate(order):
        remaining = order[depth + 1:]
        folded: Dict[Subset, tesseract.Cube] = {}
        for subset, part in products.items():
            for taken, branch in ((False, subset), (True, subset + (axis,))):
                # descend only where something at or below this branch is
                # wanted; a subset is reachable if some kept contrast
                # begins with it and adds only axes still to come
                if not _reachable(branch, wanted, remaining):
                    continue
                halves = fold_half(part, axis, taken)
                folded[branch] = halves
        products = folded
    out: Dict[Subset, Dict[str, np.ndarray]] = {}
    for subset, part in products.items():
        if subset in wanted:
            key = tuple(sorted(subset, key=cube.axes.index))
            out[key] = {"value": part.values.reshape(-1),
                        "variance": part.variance.reshape(-1),
                        "order": len(subset)}
    return out


def _reachable(branch: Subset, wanted: Set[Subset], remaining: Subset) -> bool:
    """Is any kept contrast reachable from here by adding later axes only?"""
    rest = set(remaining)
    prefix = set(branch)
    for target in wanted:
        chosen = set(target)
        if prefix <= chosen and (chosen - prefix) <= rest:
            return True
    return False


def fold_half(cube: tesseract.Cube, axis: str, differential: bool
              ) -> tesseract.Cube:
    """One half of a fold: the parent, or the differential, not both.

    `tesseract.fold` returns the pair because a full transform needs both.
    A pruned walk often needs only one, and computing the other is exactly
    the work the null space is supposed to save.
    """
    index = cube.axes.index(axis)
    first = np.take(cube.values, 0, axis=index)
    second = np.take(cube.values, 1, axis=index)
    variance = 0.25 * (np.take(cube.variance, 0, axis=index)
                       + np.take(cube.variance, 1, axis=index))
    values = 0.5 * (first - second) if differential else 0.5 * (first + second)
    rest = tuple(a for a in cube.axes if a != axis)
    return tesseract.Cube(rest, values, variance, cube.bins, cube.channels)


def signs(axes: Sequence[str], vertex: Sequence[int], subset: Subset) -> float:
    """The sign a contrast carries into one vertex: the fold puts the
    differential on the FIRST state and its negative on the second, so a
    vertex at state 1 on every axis of the subset with an odd count reads
    minus."""
    names = tuple(axes)
    parity = sum(int(vertex[names.index(axis)]) for axis in subset)
    return -1.0 if parity % 2 else 1.0


def unfold_kept(axes: Sequence[str],
                contrasts: Dict[Subset, Dict[str, np.ndarray]],
                keep: Optional[Iterable[Subset]] = None,
                amounts: Optional[Dict[Subset, np.ndarray]] = None
                ) -> np.ndarray:
    """THE FOLD RUN BACKWARDS OVER THE ADMITTED CONTRASTS ONLY.

    `tesseract.reconstruct` sums every contrast into every vertex at 4^n
    vector operations and the butterfly in `ringing_tesseract.unfold` does
    it at n 2^n, but both visit every subset. A pruned transform holds only
    the admitted ones, so the signed sum over those - 2^n times the count
    admitted - is the cheapest exact form and touches nothing null.

    `amounts` scales each contrast per bin: one everywhere under
    SUBSTITUTE, the two-bank agreement under REMOVE. A subset absent from
    `contrasts` contributes nothing, which is what "never computed" means.
    """
    names = tuple(axes)
    n = len(names)
    chosen = [s for s in contrasts] if keep is None else [
        tuple(sorted(s, key=names.index)) for s in keep]
    bins = next(iter(contrasts.values()))["value"].size if contrasts else 0
    out = np.zeros((2,) * n + (bins,), dtype=np.complex128)
    for vertex in np.ndindex(*((2,) * n)):
        total = np.zeros(bins, dtype=np.complex128)
        for subset in chosen:
            if subset not in contrasts:
                continue
            value = np.asarray(contrasts[subset]["value"], dtype=np.complex128)
            if amounts is not None and subset in amounts:
                value = value * amounts[subset]
            total += signs(names, vertex, subset) * value
        out[vertex] = total
    return out


# ---------------------------------------------------------------------------
# The causal axis: a contrast's delay, its minimum phase and its all-pass
# ---------------------------------------------------------------------------

def causal_split(value: np.ndarray, bins_hz: np.ndarray, sample_rate_hz: float
                 ) -> Dict[str, float]:
    """One contrast's departure split on the axis that is not folded.

    The real part is the log magnitude and the imaginary part the phase, on
    the measured band. The contrast is placed on the full grid the band was
    cut from (`tesseract.on_full_grid`: feeding the band alone to the
    cepstrum stretches it over the whole axis and mis-scales every delay,
    the defect the tesseract record names), the Bode relation gives the
    phase a causal minimum-phase response with this magnitude must have,
    and the excess over that is a delay (the slope against frequency, one
    number) plus what remains, the all-pass. The share is taken AFTER the
    delay is removed: a delay is not a failure of minimum phase.
    """
    x = np.asarray(value, dtype=np.complex128).reshape(-1)
    f = np.asarray(bins_hz, dtype=np.float64).reshape(-1)
    if x.size < 4 or x.size != f.size:
        raise ValueError("a causal split needs a series on a frequency axis")
    full, _grid, index = tesseract.on_full_grid(x.real, f, sample_rate_hz)
    minimum = hypercomplex.minimum_phase(full)[index]
    excess = x.imag - minimum
    # THE DELAY IS FITTED THROUGH ZERO FREQUENCY, WITH A FREE PHASE
    # REFERENCE BESIDE IT. A delay of tau is -2 pi f tau, and a ramp fitted
    # about the band's centre leaves the constant 2 pi f0 tau behind as a
    # false all-pass - measured here on a planted 40 ns delay as 0.528 rad
    # rms of "all-pass" that was nothing but the band centre's phase. The
    # constant is a phase reference, unwrapped from the first bin's principal
    # value and no more a failure of minimum phase than the delay is; the
    # modelled-residuals report records what fitting it was worth (home's
    # explained share rose from 46.8 to 91.2 per cent).
    basis = np.stack([np.ones(f.size), f], axis=1)
    coefficients, _res, _rank, _sv = np.linalg.lstsq(basis, excess, rcond=None)
    reference, slope = float(coefficients[0]), float(coefficients[1])
    residue = excess - reference - slope * f
    phase_without_delay = x.imag - reference - slope * f
    return {
        "magnitude_rms": float(np.sqrt(np.mean(x.real ** 2))),
        "phase_rms": float(np.sqrt(np.mean(x.imag ** 2))),
        "minimum_phase_rms": float(np.sqrt(np.mean(minimum ** 2))),
        "excess_rms": float(np.sqrt(np.mean(excess ** 2))),
        "delay_s": -slope / (2.0 * np.pi),
        "phase_reference_rad": reference,
        "allpass_rms": float(np.sqrt(np.mean(residue ** 2))),
        # a share of nothing is not a number: a zero-phase reading has no
        # phase for the minimum-phase part to explain, and dividing by its
        # vanishing power read -7.20 on a real probe before this guard
        "minimum_phase_share": (float(
            1.0 - np.mean(residue ** 2)
            / np.mean(phase_without_delay ** 2))
            if np.mean(phase_without_delay ** 2) > 0.0 else float("nan")),
    }


# ---------------------------------------------------------------------------
# The capture's hypercube: where the data must not exist
# ---------------------------------------------------------------------------

def hypercube(power: np.ndarray, bins_hz: np.ndarray,
              occupied: Sequence[Tuple[float, float]],
              edge_hz: Optional[float] = None) -> Dict[str, object]:
    """The generalized model for the null space: the box the file fixes,
    less the sub-box the format fills.

    Ethan: "an edge that represents exactly the cutoff for the input data,
    i.e. 20mhz of possible data in a 40mhz file. A hypercube ... That is a
    constant I think for the rf file itself ... Subtract out the hyper cube
    and discard it as noise."

    `bins_hz` runs to the file's own edge (half the sample rate); `occupied`
    lists the bands the format may put signal in. Every bin outside them is
    where the data must not exist, so what is there is noise by
    construction and is measured with no judge: its mean density per bin is
    the constant, and the bins inside the sub-box are what the correction
    keeps. Nothing here decides whether the in-band floor is HIGHER than
    this constant; that difference is reported by `Transform.floors` and is
    structure to name, never noise to discard.
    """
    p = np.asarray(power, dtype=np.float64).reshape(-1)
    f = np.asarray(bins_hz, dtype=np.float64).reshape(-1)
    if p.size != f.size:
        raise ValueError("power and bins differ in length")
    inside = np.zeros(f.size, dtype=bool)
    for low, high in occupied:
        inside |= (f >= float(low)) & (f <= float(high))
    if edge_hz is not None:
        inside |= f > float(edge_hz)
    outside = ~inside
    if not np.any(outside):
        raise ValueError("no bin lies outside the occupied bands; the "
                         "hypercube has no null region to measure")
    density = float(np.mean(p[outside]))
    return {
        "density": density,
        "bins_outside": int(np.count_nonzero(outside)),
        "bins_inside": int(np.count_nonzero(inside)),
        "edge_hz": float(f.max() if edge_hz is None else edge_hz),
        "occupied": [(float(a), float(b)) for a, b in occupied],
        "mask_outside": outside,
        "why": ("the file's sample rate fixes where data can exist and the "
                "format fixes where the signal may; the rest is null space "
                "by construction, signal-free, and its density is the "
                "constant subtracted and discarded"),
    }


# ---------------------------------------------------------------------------
# The accumulating transform
# ---------------------------------------------------------------------------

Latched = namedtuple("Latched", [
    "version", "kind", "axes", "treatment", "contrasts", "amounts",
    "departures", "causal", "floors", "evidence", "why"])


class Transform:
    """One transform: every measurement a vertex, one fold, one departure
    per vertex, and the null space treated the way the caller asks.

    THE VERTEX IS ACCUMULATED, because the response is a known constant
    over a capture span (the constants rule: average the whole capture with
    transients rejected, never track a drift). Each vertex and channel is a
    running mean with its own running second moment (Welford), so the
    variance of the mean is available at any time without storing fields.
    Two BANKS take alternate accepted fills, so a contrast can be judged on
    evidence it was not fitted on - the R5 rule, the only judgement that
    carries evidence.

    TRANSIENTS ARE REJECTED BY THEIR OWN WITNESSES first (the caller says
    `transient=True` when the decoder's sync confidence, the parity
    alternation or the colour lock refuses the field), and then by the
    running median of steps: a fill whose distance from the running mean
    exceeds the detection sigma `chroma_head_switch` declares, times the
    normal consistency constant, times the median of the recent steps, is a
    dropout or a splice and not evidence. Measured on this material, 31 per
    cent of consecutive same-head field pairs jump more than five times the
    median step, and with them counted a process variance read a thousand
    times too large.

    NOTHING HERE READS A FIELD. The adapters in `model_stages` fill the
    vertices from the instruments and realise the departures; this class is
    the mathematics between.
    """

    def __init__(self, kind: str, axes: Sequence[str],
                 channels: Dict[str, int], sample_rate_hz: float,
                 bins_hz: Optional[Dict[str, np.ndarray]] = None,
                 kept: Optional[Iterable[Subset]] = None):
        if kind not in ("picture", "rf"):
            raise ValueError("kind is 'picture' or 'rf', not %r" % (kind,))
        self.kind = kind
        self.axes: Subset = tuple(axes)
        if len(set(self.axes)) != len(self.axes) or not self.axes:
            raise ValueError("axes must be distinct and at least one")
        self.n = len(self.axes)
        if not channels:
            raise ValueError("a transform needs at least one channel")
        self.channel_names: Tuple[str, ...] = tuple(channels)
        self.channels: Dict[str, slice] = {}
        offset = 0
        for name, length in channels.items():
            length = int(length)
            if length < 1:
                raise ValueError("channel %r has no bins" % (name,))
            self.channels[name] = slice(offset, offset + length)
            offset += length
        self.bins = offset
        self.sample_rate_hz = float(sample_rate_hz)
        self.bins_hz: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in (bins_hz or {}).items()}
        for name, f in self.bins_hz.items():
            if name in self.channels and f.size != (
                    self.channels[name].stop - self.channels[name].start):
                raise ValueError("bins_hz for %r does not match its length"
                                 % (name,))
        self.kept: Set[Subset] = (live_kept(kind, self.axes) if kept is None
                                  else {tuple(sorted(s, key=self.axes.index))
                                        for s in kept})
        shape = (2,) * self.n + (self.bins,)
        cshape = (2,) * self.n + (len(self.channel_names),)
        self._banks = [self._empty(shape, cshape) for _ in range(3)]
        self._fills = np.zeros(cshape, dtype=np.int64)
        self._steps: Dict[Tuple[int, ...], List[float]] = {}
        self.accepted = 0
        self.rejected = 0
        self.rejected_by_witness = 0
        self.latched: Optional[Latched] = None
        self.version = 0
        self.external_floors: Dict[str, float] = {}
        self._waves: Dict[Tuple[int, ...], "Wave"] = {}

    @staticmethod
    def _empty(shape, cshape):
        return {"mean": np.zeros(shape, dtype=np.complex128),
                "m2": np.zeros(shape, dtype=np.float64),
                "count": np.zeros(cshape, dtype=np.float64),
                "squares": np.zeros(cshape, dtype=np.float64)}

    # -- addressing ---------------------------------------------------------

    def vertex_index(self, vertex: Dict[str, object]) -> Tuple[int, ...]:
        """A vertex named by axis, as the index tuple the cube uses.

        Every live axis must be named and nothing else may be: a vertex
        with an axis missing is not a vertex, and one naming an unreached
        axis is a category error the caller should hear about.
        """
        if set(vertex) != set(self.axes):
            raise ValueError("vertex names %r; the transform's axes are %r"
                             % (sorted(vertex), self.axes))
        return tuple(int(bool(vertex[axis])) for axis in self.axes)

    def _channel(self, channel: str) -> Tuple[int, slice]:
        if channel not in self.channels:
            raise ValueError("no channel %r; the channels are %r"
                             % (channel, self.channel_names))
        return self.channel_names.index(channel), self.channels[channel]

    # -- filling --------------------------------------------------------------

    def fill(self, vertex: Dict[str, object], channel: str, values,
             weight: float = 1.0, transient: bool = False) -> bool:
        """One field's measurement into one vertex and channel.

        Returns whether it was accepted. `weight` is the evidence behind
        the measurement in the caller's own currency (lines pooled, say);
        the banks alternate on accepted fills, not on weight.
        """
        index = self.vertex_index(vertex)
        c, sl = self._channel(channel)
        x = np.asarray(values, dtype=np.complex128).reshape(-1)
        if x.size != sl.stop - sl.start:
            raise ValueError("channel %r takes %d bins, got %d"
                             % (channel, sl.stop - sl.start, x.size))
        if not (weight > 0.0):
            raise ValueError("weight must be positive")
        if transient:
            self.rejected += 1
            self.rejected_by_witness += 1
            return False
        if not np.all(np.isfinite(x)) or self._is_transient(index, c, sl, x):
            self.rejected += 1
            return False
        fills = int(self._fills[index + (c,)])
        for bank in (0, 1 + (fills % 2)):
            self._welford(self._banks[bank], index, c, sl, x, float(weight))
        self._fills[index + (c,)] = fills + 1
        self.accepted += 1
        return True

    @staticmethod
    def _welford(bank, index, c, sl, x, w):
        n0 = bank["count"][index + (c,)]
        n = n0 + w
        mean = bank["mean"][index + (sl,)]
        delta = x - mean
        mean += (w / n) * delta
        bank["m2"][index + (sl,)] += w * np.real(np.conj(delta) * (x - mean))
        bank["count"][index + (c,)] = n
        bank["squares"][index + (c,)] += w * w

    @staticmethod
    def _variance_of_mean(bank, index, c, sl):
        """The variance of the weighted mean with RELIABILITY weights.

        A weight is how much a fill is worth, not how many samples it is:
        one fill of weight 245 is still one field and has no variance. With
        V1 the sum of weights and V2 the sum of their squares, the unbiased
        variance is m2 / (V1 - V2/V1) and the mean's is that times V2/V1^2;
        with every weight one this is m2 / (n (n - 1)), and with one fill
        it is undefined. The first form divided by W (W - 1) as though the
        weight were a count, and at weight 245 the instrument floor read
        zero against a null pool of 2.9e-4.
        """
        v1 = bank["count"][index + (c,)]
        v2 = bank["squares"][index + (c,)]
        m2 = bank["m2"][index + (sl,)]
        denominator = v1 - (v2 / v1 if v1 > 0.0 else 0.0)
        if not (denominator > 0.0) or not (v1 > 0.0):
            return np.full(m2.shape, np.inf)
        return (m2 / denominator) * (v2 / (v1 * v1))

    def _is_transient(self, index, c, sl, x) -> bool:
        key = index + (c,)
        history = self._steps.setdefault(key, [])
        fills = int(self._fills[key])
        mean = self._banks[0]["mean"][index + (sl,)]
        step = float(np.sqrt(np.mean(np.abs(x - mean) ** 2))) if fills else 0.0
        if fills >= STEPS_BEFORE_GATING and history:
            gate = DETECTION_SIGMA * MAD_TO_SIGMA * float(np.median(history))
            if gate > 0.0 and step > gate:
                return True
        history.append(step)
        del history[:-STEP_HISTORY]
        return False

    # -- evidence -------------------------------------------------------------

    def evidence(self) -> Dict[str, Dict[str, object]]:
        """Per channel: the fewest and most fills any vertex has, and
        whether the channel is in use at all."""
        out = {}
        for c, name in enumerate(self.channel_names):
            fills = self._fills[..., c]
            out[name] = {"used": bool(fills.max() > 0),
                         "fewest": int(fills.min()),
                         "most": int(fills.max()),
                         # keyed by the vertex's coordinates as text, so a
                         # reading can be written out as it stands
                         "per_vertex": {",".join(str(int(i)) for i in v):
                                        int(fills[v])
                                        for v in np.ndindex(*fills.shape)}}
        return out

    def ready(self, minimum: int, treatment: str = "substitute") -> bool:
        """Whether every vertex of every channel in use has the evidence
        the caller asks for - and, under REMOVE, whether both banks have
        something at every vertex, because a judge with one bank empty has
        nothing to compare."""
        if treatment not in TREATMENTS:
            raise ValueError("treatment is one of %r" % (TREATMENTS,))
        used = [c for c in range(len(self.channel_names))
                if self._fills[..., c].max() > 0]
        if not used:
            return False
        for c in used:
            fills = self._fills[..., c]
            if fills.min() < max(int(minimum), 1):
                return False
            if treatment == "remove" and fills.min() < 2:
                return False
        return True

    # -- the cube and its contrasts ------------------------------------------

    def cube(self, bank: int = 0) -> tesseract.Cube:
        """The accumulated vertices as a cube, with the variance of each
        mean. An unfilled vertex has no value to give and raises; `ready`
        is the guard. A channel nobody filled is carried as zero with an
        infinite variance so a verdict on it is refused rather than made."""
        b = self._banks[bank]
        values = b["mean"].copy()
        variance = np.full(values.shape, np.inf, dtype=np.float64)
        for c, name in enumerate(self.channel_names):
            sl = self.channels[name]
            fills = self._fills[..., c]
            if fills.max() == 0:
                continue
            # the BANK's own evidence, not the total: a bank with no fill at
            # a vertex has nothing to say there
            if np.any(b["count"][..., c] <= 0.0):
                raise ValueError("channel %r has a vertex with no evidence "
                                 "in bank %d" % (name, bank))
            for vertex in np.ndindex(*((2,) * self.n)):
                variance[vertex + (sl,)] = self._variance_of_mean(b, vertex, c, sl)
        return tesseract.Cube(self.axes, values, variance,
                              channels=dict(self.channels))

    def contrasts(self, treatment: str = "substitute"
                  ) -> Tuple[Dict[Subset, Dict[str, np.ndarray]],
                             Dict[Subset, np.ndarray]]:
        """The contrasts and the amount of each to apply, per bin.

        SUBSTITUTE: the pruned fold, kept contrasts only, amount one.
        REMOVE: the full fold, and every contrast's amount is its two-bank
        agreement per channel - the correlation of the bank-one and
        bank-two readings, which a residual describing the path reads near
        one on and a residual describing the field it came from reads near
        zero on. The grand mean is the departure itself and is applied in
        full under both.
        """
        if treatment not in TREATMENTS:
            raise ValueError("treatment is one of %r" % (TREATMENTS,))
        cube = self.cube(0)
        if treatment == "substitute":
            contrasts = prune(cube, self.kept)
            amounts = {s: np.ones(self.bins) for s in contrasts}
            return contrasts, amounts
        contrasts = tesseract.walsh(cube)
        first = tesseract.walsh(self.cube(1))
        second = tesseract.walsh(self.cube(2))
        amounts = {}
        for subset, entry in contrasts.items():
            amount = np.ones(self.bins)
            if subset:
                for c, name in enumerate(self.channel_names):
                    sl = self.channels[name]
                    if self._fills[..., c].max() == 0:
                        amount[sl] = 0.0
                        continue
                    a = first[subset]["value"][sl]
                    b = second[subset]["value"][sl]
                    amount[sl] = self._agreement(a, b)
            amounts[subset] = amount
            entry["amount"] = amount
        return contrasts, amounts

    @staticmethod
    def _agreement(a: np.ndarray, b: np.ndarray) -> float:
        """The R5 gate on the two halves of a complex contrast together:
        the real and imaginary parts are stacked so a phase-only residual
        is judged as fully as a magnitude-only one."""
        first = np.concatenate([a.real, a.imag])
        second = np.concatenate([b.real, b.imag])
        weight = np.ones(first.size)
        return float(agreement(first, second, weight, weight, 1.0))

    def departure(self, vertex: Dict[str, object],
                  treatment: Optional[str] = None) -> np.ndarray:
        """This vertex's whole departure on every channel, from the latched
        contrasts where a latch exists and from the running ones otherwise."""
        index = self.vertex_index(vertex)
        if self.latched is not None and (treatment is None
                                         or treatment == self.latched.treatment):
            return self.latched.departures[index].copy()
        contrasts, amounts = self.contrasts(treatment or "substitute")
        stack = unfold_kept(self.axes, contrasts, None, amounts)
        return stack[index]

    # -- floors ---------------------------------------------------------------

    def set_floor(self, channel: str, power_per_bin: float) -> None:
        """The capture's hypercube, in this channel's units: the noise power
        one contrast bin would carry from the region of the file where the
        data must not exist, measured once by the adapter and handed in."""
        self._channel(channel)
        self.external_floors[channel] = float(power_per_bin)

    def floors(self) -> Dict[str, Dict[str, object]]:
        """The three floors, per channel, in one currency: the power a
        contrast bin carries.

        `instrument`: from the Welford variance of the means, a contrast
        being the mean of 2^n vertices with signs. `null_pool`: the mean
        power of the contrasts the declaration calls null, from the full
        fold - noise samples if the declaration is right. `banks`: the two
        banks' disagreement, a quarter of the mean squared difference of
        their contrasts. `hypercube`: the file's own constant, where the
        adapter measured one. They must agree where the null space is
        noise; where `null_pool` or `banks` stand above `hypercube`, the
        excess is structure and is named as such.
        """
        cube = self.cube(0)
        full = tesseract.walsh(cube)
        out: Dict[str, Dict[str, object]] = {}
        both = all(self._fills[..., c].min() >= 2
                   for c in range(len(self.channel_names))
                   if self._fills[..., c].max() > 0)
        first = tesseract.walsh(self.cube(1)) if both else None
        second = tesseract.walsh(self.cube(2)) if both else None
        for c, name in enumerate(self.channel_names):
            sl = self.channels[name]
            if self._fills[..., c].max() == 0:
                out[name] = {"used": False}
                continue
            var = cube.variance[..., sl]
            instrument = float(np.mean(var[np.isfinite(var)]) / (2 ** self.n)) \
                if np.any(np.isfinite(var)) else float("nan")
            null = [s for s in full if s not in self.kept]
            null_pool = float(np.mean([np.mean(np.abs(full[s]["value"][sl]) ** 2)
                                       for s in null])) if null else float("nan")
            banks = float("nan")
            if first is not None:
                banks = float(np.mean([
                    np.mean(np.abs(first[s]["value"][sl]
                                   - second[s]["value"][sl]) ** 2) / 4.0
                    for s in full]))
            hyper = self.external_floors.get(name, float("nan"))
            reading = {
                "used": True,
                "instrument": instrument,
                "null_pool": null_pool,
                "banks": banks,
                "hypercube": hyper,
                "null_contrasts": [":".join(s) for s in null],
            }
            floor = min(v for v in (instrument, null_pool, banks, hyper)
                        if np.isfinite(v)) if any(
                np.isfinite(v) for v in (instrument, null_pool, banks, hyper)
            ) else float("nan")
            reading["floor"] = floor
            reading["excess_over_hypercube"] = (
                float(null_pool / hyper) if np.isfinite(hyper) and hyper > 0
                and np.isfinite(null_pool) else float("nan"))
            out[name] = reading
        return out

    # -- the latch --------------------------------------------------------------

    def latch(self, treatment: str, minimum: int) -> Optional[Latched]:
        """Freeze the correction, once.

        Rule 4 of the ringing rules: the measurement is accumulated over the
        decode and the APPLIED correction must not change from field to
        field. The latch fires the first time the evidence is there and the
        result is immutable; fills continue afterwards for the report, and
        a second latch is refused - a new capture is a new transform.
        """
        if self.latched is not None:
            return self.latched
        if not self.ready(minimum, treatment):
            return None
        contrasts, amounts = self.contrasts(treatment)
        stack = unfold_kept(self.axes, contrasts, None, amounts)
        departures = {v: stack[v] for v in np.ndindex(*((2,) * self.n))}
        causal = {}
        if "response" in self.channels and "response" in self.bins_hz:
            sl = self.channels["response"]
            f = self.bins_hz["response"]
            for subset, entry in contrasts.items():
                if not subset:
                    continue
                try:
                    causal[":".join(subset)] = causal_split(
                        entry["value"][sl], f, self.sample_rate_hz)
                except ValueError:
                    continue
        self.version += 1
        self.latched = Latched(
            version=self.version, kind=self.kind, axes=self.axes,
            treatment=treatment,
            contrasts={s: {"value": e["value"].copy(),
                           "variance": e["variance"].copy(),
                           "order": e["order"]} for s, e in contrasts.items()},
            amounts={s: a.copy() for s, a in amounts.items()},
            departures=departures, causal=causal, floors=self.floors(),
            evidence=self.evidence(),
            why=("latched once on %d fields per vertex at least, treatment "
                 "%s; the applied correction does not change from field to "
                 "field" % (int(minimum), treatment)))
        return self.latched

    # -- the wave under the convergence ------------------------------------------

    def record_remainder(self, vertex: Dict[str, object], field_number: int,
                         values, field_rate_hz: float) -> Optional[int]:
        """After the latch, this field's reading less the latched departure
        goes into the vertex's wave; before it there is no constant to be
        under."""
        if self.latched is None:
            return None
        index = self.vertex_index(vertex)
        x = np.asarray(values, dtype=np.complex128).reshape(-1)
        if x.size != self.bins:
            raise ValueError("a wave takes the whole vertex, %d bins" % self.bins)
        wave = self._waves.setdefault(index, Wave(self.bins, field_rate_hz))
        # one field's own scatter about the mean: the mean's variance times
        # the effective count, which the same weighted form gives
        variance = np.full(self.bins, np.inf)
        for c, name in enumerate(self.channel_names):
            sl = self.channels[name]
            b = self._banks[0]
            v1 = b["count"][index + (c,)]
            v2 = b["squares"][index + (c,)]
            effective = (v1 * v1 / v2) if v2 > 0.0 else 0.0
            variance[sl] = self._variance_of_mean(b, index, c, sl) * max(effective, 1.0)
        return wave.record(field_number, x - self.latched.departures[index], variance)

    def wave(self, vertex: Dict[str, object]) -> Optional[Dict[str, object]]:
        index = self.vertex_index(vertex)
        wave = self._waves.get(index)
        return None if wave is None else wave.fold()

    def wave_at(self, vertex: Dict[str, object], field_number: int) -> Optional[np.ndarray]:
        """The admitted wave to subtract at this field, or None when the
        vertex has no folded run yet."""
        index = self.vertex_index(vertex)
        wave = self._waves.get(index)
        if wave is None:
            return None
        folded = wave.fold()
        if folded is None or not folded["admitted"]:
            return None
        return wave.value_at(field_number, folded)

    # -- remainder and report ---------------------------------------------------

    def remainder(self, vertex: Dict[str, object], channel: str, values,
                  treatment: Optional[str] = None) -> Dict[str, float]:
        """One field's own reading against the departure the transform
        would remove from it: what is left, per channel, in the reading's
        units - the per-field number the report is built from."""
        c, sl = self._channel(channel)
        x = np.asarray(values, dtype=np.complex128).reshape(-1)
        expected = self.departure(vertex, treatment)[sl]
        before = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        after = float(np.sqrt(np.mean(np.abs(x - expected) ** 2)))
        return {"before": before, "after": after,
                "share_explained": (1.0 - (after / before) ** 2) if before > 0
                else float("nan")}

    def report(self) -> Dict[str, object]:
        """What the transform holds, for the field report and the plot."""
        out: Dict[str, object] = {
            "kind": self.kind,
            "axes": self.axes,
            "live": self.axes,
            "declared_live": LIVE_AXES[self.kind],
            "unreached": dict(UNREACHED[self.kind]),
            "channels": {k: (v.start, v.stop) for k, v in self.channels.items()},
            "kept": sorted(":".join(s) for s in self.kept),
            "rings": rings(self.axes, self.kept),
            "fixed_axis": "frequency, the causal axis, never folded",
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejected_by_witness": self.rejected_by_witness,
            "evidence": self.evidence(),
            "latched": None if self.latched is None else {
                "version": self.latched.version,
                "treatment": self.latched.treatment,
                "why": self.latched.why,
            },
        }
        if self.latched is not None:
            per = {}
            for subset, entry in self.latched.contrasts.items():
                name = ":".join(subset) or "mean"
                row = {"order": entry["order"]}
                for cname, sl in self.channels.items():
                    v = entry["value"][sl]
                    row[cname] = {
                        "magnitude_rms": float(np.sqrt(np.mean(v.real ** 2))),
                        "phase_rms": float(np.sqrt(np.mean(v.imag ** 2))),
                        "amount": float(np.mean(self.latched.amounts[subset][sl])),
                    }
                if name in self.latched.causal:
                    row["causal"] = dict(self.latched.causal[name])
                per[name] = row
            out["contrasts"] = per
            out["floors"] = self.latched.floors
            waves = {}
            for index, wave in self._waves.items():
                folded = wave.fold()
                if folded is None:
                    continue
                waves[",".join(str(i) for i in index)] = {
                    "fields_used": folded["fields_used"],
                    "admitted": folded["admitted"],
                    "amounts": {":".join(s) or "mean": float(a)
                                for s, a in folded["amounts"].items()},
                    "scales": folded["scales"],
                    "noise_floor": folded["noise_floor"],
                    "filled_fraction": folded["filled_fraction"],
                }
            out["waves"] = waves
        return out


# ---------------------------------------------------------------------------
# Dimensional depth: each measurement applied once, at its own grade
# ---------------------------------------------------------------------------

def depth_plan(axes: Sequence[str], kept: Iterable[Subset]) -> Dict[str, object]:
    """WHAT EACH MEASUREMENT COSTS TO APPLY, BY ITS OWN DEPTH.

    Ethan, 2026-09-07: "we can compress down measurements to only need to
    apply to their dimensional depth. I think the actual order doesn't
    matter, just the correct number of transformation in all dimensions."

    Both halves of that are properties of the fold rather than choices.

    THE DEPTH. A contrast of grade k is ONE array however many vertices it
    reaches: the fold puts it on every vertex with a sign that is the
    product of k bits, so the grand mean is one array applied with a
    constant sign, a single-axis differential is one array applied with one
    bit's sign, and the n-way is one array applied with the parity of all
    of them. Nothing of grade k needs 2^k arrays; it needs one, and the
    signs are free. So the whole model costs as many arrays as there are
    kept contrasts, and a vertex costs a signed sum over exactly those - it
    never costs 2^n of anything, which is the compression he is naming.

    THE ORDER. The folds commute, so the axes may be folded in any order
    and the same contrasts come back; what is fixed is that each axis is
    folded exactly ONCE. The count per dimension is the invariant and the
    sequence is not, which is the same reason the stages needed no
    ordering. `folds_per_axis` states that count and it is one for every
    axis, always.

    The saving against reconstructing each vertex separately is reported
    here rather than claimed: `naive_applications` is what a per-vertex sum
    over every kept contrast costs, `applications` is what applying each
    contrast once at its own depth costs, and `saving` is their ratio.
    """
    names = tuple(axes)
    n = len(names)
    kept_set = {tuple(sorted(s, key=names.index)) for s in kept}
    by_grade: Dict[int, List[str]] = {}
    for subset in sorted(kept_set, key=lambda s: (len(s), s)):
        by_grade.setdefault(len(subset), []).append(":".join(subset) or "mean")
    vertices = 2 ** n
    return {
        "axes": names,
        "folds_per_axis": {axis: 1 for axis in names},
        "depths": by_grade,
        "arrays": len(kept_set),
        "vertices": vertices,
        "applications": len(kept_set),
        "naive_applications": vertices * len(kept_set),
        "saving": (float(vertices) if kept_set else 1.0),
        "order_is_free": True,
        "why": ("a contrast of grade k is one array applied with the sign "
                "of k bits, so the model costs one array a contrast and "
                "never 2^n of anything; the folds commute, so only the "
                "count of folds per axis is fixed and their order is not"),
    }


def apply_at_depth(axes: Sequence[str],
                   contrasts: Dict[Subset, Dict[str, np.ndarray]],
                   amounts: Optional[Dict[Subset, np.ndarray]] = None,
                   keep: Optional[Iterable[Subset]] = None) -> np.ndarray:
    """EVERY CONTRAST APPLIED ONCE, AT ITS OWN DEPTH, AND UNFOLDED ONCE.

    The same departures `unfold_kept` returns, reached the other way about:
    instead of summing the kept contrasts into each vertex in turn, each
    contrast is written once into the folded array at the position its own
    axes name, and one butterfly carries the whole array back out to the
    vertices. The two agree to machine precision - the test asserts it -
    because they are the same sum in a different order, which is the whole
    of his point.

    The butterfly costs n 2^n and the writes cost one array each, against
    the signed sum's 2^n per contrast; at three axes and six contrasts that
    is 24 against 48, and the gap widens with every axis added.
    """
    names = tuple(axes)
    n = len(names)
    chosen = list(contrasts) if keep is None else [
        tuple(sorted(s, key=names.index)) for s in keep]
    if not chosen:
        raise ValueError("no contrast to apply")
    bins = next(iter(contrasts.values()))["value"].size
    stack = np.zeros((2,) * n + (bins,), dtype=np.complex128)
    for subset in chosen:
        entry = contrasts.get(subset)
        if entry is None:
            continue
        value = np.asarray(entry["value"], dtype=np.complex128)
        if amounts is not None and subset in amounts:
            value = value * amounts[subset]
        # THE DEPTH IS THE POSITION: a contrast of grade k is written at
        # the vertex whose bits are one exactly on its own axes, which is
        # where the butterfly will read it from and spread it with the
        # right signs. One write, whatever the grade.
        stack[tuple(1 if axis in subset else 0 for axis in names)] = value
    for position in range(n):
        parent = np.take(stack, 0, axis=position)
        differential = np.take(stack, 1, axis=position)
        stack = np.stack([parent + differential, parent - differential],
                         axis=position)
    return stack


# ---------------------------------------------------------------------------
# The wave under the convergence: the remainder folded on field time
# ---------------------------------------------------------------------------

class Wave:
    """One vertex's remainder after the latch, field by field, folded on
    the bits of the field index.

    Ethan: "I believe there is still a wave underneath the convergence that
    itself can be differentialed as random noise and subtracted out."

    THE SERIES IS PER VERTEX, so it is per head, and a head reads every
    second field: the ordinal of a field within its head is the field
    number halved, and the series' own rate is half the field rate. Bit 0
    of this series is therefore a two-frame alternation and not the head;
    the head is already an axis of the transform. The drum turns once per
    two fields and is a constant of a head's own series, which is the trap
    the tesseract record names and this construction sidesteps.

    THE JUDGE IS TIME ITSELF. The run is split into its first and second
    halves, each folded on its own bits; a contrast on a scale both halves
    contain is admitted by the amount the second half reproduces of the
    first (the R5 agreement), and a contrast on the run's top bit, which
    only the whole run contains, cannot be judged and is not admitted. A
    wave that reproduces is subtracted at each field by the signs of that
    field's bits; what does not reproduce is the random noise and is
    discarded, its power reported as the wave's floor.
    """

    def __init__(self, bins: int, field_rate_hz: float, fields_per_step: int = 2):
        self.bins = int(bins)
        self.field_rate_hz = float(field_rate_hz)
        self.step = int(fields_per_step)
        self.records: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def record(self, field_number: int, remainder, variance) -> int:
        """One field's remainder, by its ordinal within the head."""
        ordinal = int(field_number) // self.step
        x = np.asarray(remainder, dtype=np.complex128).reshape(-1)
        v = np.broadcast_to(np.asarray(variance, dtype=np.float64), x.shape).copy()
        if x.size != self.bins:
            raise ValueError("a remainder has %d bins, the wave takes %d"
                             % (x.size, self.bins))
        if np.all(np.isfinite(x)):
            self.records[ordinal] = (x, v)
        return ordinal

    def run(self) -> Optional[Dict[str, object]]:
        """The longest consecutive run ending at the latest field, with
        gaps filled from same-head neighbours and the filling reported."""
        if len(self.records) < 4:
            return None
        ordinals = sorted(self.records)
        last = ordinals[-1]
        first = ordinals[0]
        present = np.array(ordinals)
        total = last - first + 1
        values = np.stack([self.records[o][0] for o in ordinals])
        var = np.stack([self.records[o][1] for o in ordinals])
        filled = tesseract.fill_field_gaps(values, var, present - first, total)
        if filled["responses"].shape[0] < 4:
            return None
        return filled

    def fold(self, sigma: float = 3.0) -> Optional[Dict[str, object]]:
        """The run folded on its bits, each contrast judged by the halves."""
        run = self.run()
        if run is None:
            return None
        responses, variances = run["responses"], run["variances"]
        index = np.asarray(run["field_index"], dtype=int)
        consecutive = np.all(np.diff(index) == 1)
        if not consecutive:
            # keep the longest consecutive tail
            breaks = np.flatnonzero(np.diff(index) != 1)
            start = int(breaks[-1]) + 1
            responses, variances, index = responses[start:], variances[start:], index[start:]
        if index.size < 4:
            return None
        rate = self.field_rate_hz / self.step
        whole = tesseract.from_field_series(responses, variances, index, field_rate_hz=rate)
        used = whole["fields_used"]
        responses, variances, index = responses[-used:], variances[-used:], index[-used:]
        whole = tesseract.from_field_series(responses, variances, index, field_rate_hz=rate)
        half = used // 2
        first = tesseract.from_field_series(responses[:half], variances[:half],
                                            index[:half], field_rate_hz=rate)
        second = tesseract.from_field_series(responses[half:], variances[half:],
                                             index[half:], field_rate_hz=rate)
        contrasts = tesseract.walsh(whole["cube"])
        verdicts = tesseract.identified(contrasts, sigma)
        left = tesseract.walsh(first["cube"])
        right = tesseract.walsh(second["cube"])
        amounts: Dict[Subset, float] = {}
        for subset in contrasts:
            if not subset:
                amounts[subset] = 0.0        # the run's mean is the latch's business
                continue
            if subset in left and subset in right:
                a, b = left[subset]["value"], right[subset]["value"]
                fa = np.concatenate([a.real, a.imag])
                fb = np.concatenate([b.real, b.imag])
                w = np.ones(fa.size)
                amounts[subset] = float(agreement(fa, fb, w, w, 1.0))
            else:
                amounts[subset] = 0.0        # only the whole run holds it
        admitted = [s for s, a in amounts.items() if a > 0.0]
        noise = [s for s in contrasts if s and amounts[s] == 0.0]
        floor = float(np.mean([np.mean(np.abs(contrasts[s]["value"]) ** 2)
                               for s in noise])) if noise else float("nan")
        return {
            "cube": whole["cube"], "contrasts": contrasts, "amounts": amounts,
            "verdicts": verdicts, "scales": whole["scales"],
            "fields_used": used, "filled": run["filled"],
            "filled_fraction": run["filled_fraction"],
            "first_ordinal": int(index[0]),
            "admitted": [":".join(s) for s in admitted],
            "noise_floor": floor,
            "why": ("each time-scale contrast is admitted by the amount the "
                    "second half of the run reproduces of the first; the "
                    "run's mean belongs to the latch, its top bit cannot be "
                    "judged, and what is not admitted is the random noise"),
        }

    def value_at(self, field_number: int, folded: Dict[str, object]) -> np.ndarray:
        """The admitted wave at one field: the fold run backwards at the
        vertex this field's bits name, read periodically beyond the run."""
        ordinal = int(field_number) // self.step
        axes = folded["cube"].axes
        k = len(axes)
        n = (ordinal - int(folded["first_ordinal"])) % (2 ** k)
        vertex = tuple(int((n >> b) & 1) for b in range(k))
        total = np.zeros(self.bins, dtype=np.complex128)
        for subset, amount in folded["amounts"].items():
            if amount <= 0.0 or not subset:
                continue
            total += amount * signs(axes, vertex, subset) * folded["contrasts"][subset]["value"]
        return total


# ---------------------------------------------------------------------------
# The head schedule and the published snapshot, for the worker threads
# ---------------------------------------------------------------------------

class HeadSchedule(namedtuple("HeadSchedule",
                              ["anchor", "period", "parity", "version"])):
    """Which head a block of radio frequency belongs to, from its absolute
    sample position.

    The workers demodulate about five fields ahead of assembly and are not
    told which head is coming; the field thread knows, after the fact, where
    each head switch fell. So the schedule is anchored on a switch the
    decoder located (`head_switch.locate`), with the field period in samples
    as the known constant it is, and a block is labelled by extrapolation.
    `parity` is the head label of the field that BEGINS at the anchor.
    """
    __slots__ = ()

    def head_at(self, sample: float) -> bool:
        k = math.floor((float(sample) - self.anchor) / self.period)
        return bool(self.parity) ^ bool(k & 1)

    def majority(self, start: float, length: float) -> bool:
        """The head owning most of a span. A span straddling a switch takes
        the side with more samples; a switch sits in the vertical interval
        and a block is shorter than that interval, so what is mislabelled
        is blanking."""
        first = self.head_at(start)
        last = self.head_at(start + max(float(length) - 1.0, 0.0))
        if first == last:
            return first
        k = math.floor((float(start) - self.anchor) / self.period)
        boundary = self.anchor + (k + 1) * self.period
        before = boundary - float(start)
        return first if before >= float(length) / 2.0 else last


class Published(namedtuple("Published", [
        "version", "common", "per_head", "schedule", "n_fft", "why", "extras"])):
    """An immutable snapshot for the worker threads: one common table, one
    table per head, the schedule that chooses between them, and `extras`,
    a mapping of any further latched quantity the workers apply beside the
    multiply - the luma's quadrature image (`alpha`, `beta`, `carrier_hz`),
    which is not a multiply but the same conjugate map the chroma's
    correction uses, on the carrier's complex baseband. Built once at the
    latch and rebound atomically; the workers read it and never mutate it,
    which is what makes sharing it across threads safe."""
    __slots__ = ()

    def table(self, block_start, n_fft: int) -> Optional[np.ndarray]:
        if int(n_fft) != int(self.n_fft):
            return None
        if self.per_head is None or self.schedule is None or block_start is None:
            return self.common
        parity = self.schedule.majority(float(block_start), float(n_fft))
        return self.per_head.get(bool(parity), self.common)


# a snapshot built without `extras` carries none: the adapters that rebind
# the schedule under the same version construct the tuple positionally
Published.__new__.__defaults__ = ({},)


def publish(version: int, common: np.ndarray,
            per_head: Optional[Dict[bool, np.ndarray]],
            schedule: Optional[HeadSchedule], why: str = "",
            extras: Optional[Dict[str, object]] = None) -> Published:
    """Fold the per-head differential into one table per head, once, so a
    block costs a single multiply. `per_head` holds each head's OWN
    multiplier (the differential already applied); it is combined with the
    common table here."""
    common = np.asarray(common, dtype=np.complex128)
    tables = None
    if per_head is not None:
        tables = {}
        for parity, own in per_head.items():
            own = np.asarray(own, dtype=np.complex128)
            if own.shape != common.shape:
                raise ValueError("a head table must match the common table")
            tables[bool(parity)] = common * own
    return Published(version=int(version), common=common, per_head=tables,
                     schedule=schedule, n_fft=int(common.size), why=why,
                     extras=dict(extras or {}))
