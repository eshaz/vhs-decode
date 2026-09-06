"""The held-out harness: every estimator's residual on data it did not see,
with every parameter frozen.

Ethan: *"Report held-out residual with parameters frozen. In-sample residual
falls monotonically on pure noise - it can't distinguish a good model from an
over-parameterized one."*

THAT IS A STATEMENT ABOUT WHAT A RESIDUAL CAN CERTIFY, AND IT IS EXACT. Adding
a parameter to a least-squares model can only lower the residual it was fitted
to, whatever the data hold, so a falling in-sample residual is a property of
the arithmetic and not of the tape. The arc has this measured on its own
material: fitted and judged on the same seven fields, the limit drives the
response, sweep and product residuals to 0.0001 / 0.0000 / 0.0002 dB; fitted
on other fields of the same head and tape it stands at 0.0328 / 0.0079 /
0.0563 dB - three hundred times larger with nothing changed but which samples
judged it (docs/RESIDUAL_LIMIT_DESIGN.md, rule R5, 75 bars head A). An
in-sample product term put +0.0046 dB back into its own margin and stalled
where the projected form converged to 0.0001 dB (the same document, rule R3).
So the floor a component is measured against is its residual on data it was
not built from, and that is the only number this module reports as evidence.

TWO OUT-OF-SAMPLE MEASUREMENTS LIVE IN THIS TREE AND THEY ANSWER DIFFERENT
QUESTIONS; NEITHER REPLACES THE OTHER.

  `running_differential`   Each field is judged against a running window of
                           fields BEFORE it joins that window, so its
                           residual is out of sample by construction and no
                           split can be forgotten. Its window is bounded
                           below by root-N and above by the measured 0.293 Hz
                           transport drift, and a window outside them is
                           refused. WHAT IT ANSWERS: is the model, taken
                           whole, still tracking this tape as the decode
                           runs, and what gain has the remainder earned
                           (0.72 to 0.74 on 1023 home fields, 3.6 to 3.9
                           times the floor)? It cannot say which entry
                           earned its place: nothing in it is fitted per
                           entry, nothing is frozen, and the window moves.
  this module              The pooled offline exports, where the field axis
                           has already been averaged away. WHAT IT ANSWERS:
                           does THIS entry, carrying THIS many parameters,
                           lower the residual on members its parameters
                           never saw, by more than a wrong shape of the same
                           smoothness reaches? Parameters are frozen at the
                           values one bank fitted, the split is a fixed
                           block split, and each entry meets a surrogate
                           null of its own shape. It cannot say whether the
                           model tracks the tape over a thousand fields -
                           the exports it reads have pooled that axis into
                           two halves.

Both face the same drift and treat it oppositely, on purpose: the running
window must be SHORTER than the drift it means to follow, while the blocks
here alternate between the banks so that a drift across the record lands in
both and neither bank is judged by extrapolation.

THREE THINGS THE SPLIT MUST DO, each a measured trap rather than a preference:

  it stays within a head     Fields alternate heads on a two-head helical
                             drum, and the two heads' departures do not
                             correlate. Pooled across heads the vertical
                             interval's DC offset tracked the response tilt at
                             r = -0.694; within each head it vanished to
                             -0.025 and -0.144 (memory: vertical-interval
                             LF reference, home tape). A pooled correlation is
                             a between-group difference, not a relationship,
                             so no group here ever holds two heads.
  it is blocked in time      Time-base wander and the drum's rate variation
                             are slow, so alternate fields share them and an
                             interleaved split leaks them from train to test.
                             Members are taken in contiguous blocks, and the
                             blocks alternate between the banks so that a
                             trend across the record does not land wholly in
                             one bank (`information_extrapolation.
                             guess_credibility` notes that trap on the other
                             side). The coarsest such split is the one the
                             sync exports already carry - the first half of
                             the fields against the second - and `half_split`
                             is that split on shared bins.
  it never splits a field    Every line of a field goes to the same bank: a
                             field is one head pass and one time-base state.

THE VERDICT NEEDS A NULL, because "the held-out residual fell" is not enough
on its own: a single parameter fitted to noise lowers the held-out residual
half the time (the descent is sigma^2 U (2Z - U) for two standard normals,
which is positive with probability one half). `null_holdout` scrambles the
component's own signature and runs the same frozen fit, giving the
distribution of held-out descents a model with this many parameters and this
smoothness reaches when its shape is wrong. An entry is accepted only when
its descent stands above that distribution's median, floored at zero, by
`CHANCE_SIGMA` of its spreads. `ladder` applies that rule entry by entry in
the chain's known order, rolling back any entry it refuses.

WHAT THE SURROGATE MUST KEEP, AND WHAT IT MUST NOT. The chance descent of an
entry is set by its projection on the noise, whose variance is the sum over
lags of the entry's autocorrelation times the noise's. Lags beyond the
noise's own correlation length do not enter it. So the surrogate keeps the
signature's autocorrelation up to that length - the signature's power
spectrum along the member axis, smoothed over a window of that many bins,
with every phase randomised - and nothing beyond it. A surrogate that kept
the whole spectrum kept the whole shape whenever the spectrum was a line:
an echo's log shape is one ripple, and its phase-randomised copy was the
same ripple rotated, so the null carried the entry it was meant to test
(measured on a planted echo, 2026-09-05: the rotated null descended by 0.157
against the true entry's 0.295, and the true entry would have been refused).
A surrogate that kept nothing - a permutation - measured a smooth entry
against white noise it does not face, and on a correlated member axis it
understated the chance level. The correlation length is measured, never
assumed: `noise_correlation_length` reads it from two independent
measurements of the same members (their difference is noise alone), and the
sync exports carry exactly that pair.

WHAT THE FIRST BUILD OF THIS MODULE GOT WRONG, measured against its own
tests on 2026-09-05, and kept here because each was a way the harness could
deceive its reader:

  a percentile gate           the bar was the null's fifth percentile, which
                              with eight draws sits between the two smallest,
                              so a wrong entry cleared it about one time in
                              six: on pure noise the ladder accepted two of
                              eight echoes. The bar is now the null's median
                              plus three spreads. Measured on eight echo
                              entries against white complex noise, band
                              split (2026-09-05): 4 of 800 accepted at eight
                              draws (0.5%), 0 of 600 at twenty-four; against
                              noise correlated over sixteen members, with
                              the length measured and eight draws, 0 of 240.
  a null that broke the       on `half_split` an entry is one shape on both
  halves                      halves, and its surrogate was made over the
                              stacked members, so the null's two halves were
                              independent random shapes (correlation 0.003
                              on a planted echo). Its descent was then the
                              product of two independent chances - a weight
                              fitted to one random shape, applied to
                              another - which is symmetric about zero and
                              wide, where a wrong shape that IS repeated
                              descends by the positive share it captures of
                              whatever reproduces, and more narrowly. On cd
                              head a's `dropout` entry (sync export, 944
                              bins, 24 draws) the stacked null spread 3.0
                              floor units and set the bar at 8.99 against a
                              descent of 8.86 - refused; the repeated null
                              has median 0.54, spread 1.69, bar 5.61 -
                              accepted. Over the six head-tapes the counts
                              accepted went from 1/2/0/0/0/0 to
                              3/10/4/3/2/3. The surrogate is now made on
                              the underlying axis and mapped through
                              `member_of` exactly as the entry is.
  a null that did harm        a template of per-field labels predicts nothing
                              on held-out fields, so its descent is zero; its
                              label-permuted null adds noise, so the null's
                              descent is negative; zero lay below the null's
                              percentile and the control was accepted. The
                              chance level is floored at zero.
  the wrong sequence counted  the measure of blockedness counted runs in the
                              global member index, where one head's members
                              are never contiguous because the fields
                              alternate heads; a correctly blocked split of
                              ten fields read as ten runs. `runs_of` counts
                              against the sequence the bank was drawn from.

The estimator contract is small: `fit(values, index) -> parameters` and
`predict(index, parameters) -> the modelled contribution`. The residual after
an entry is the value minus its prediction, on train under the parameters it
fitted and on test under THE SAME parameters - frozen means nothing in the
prediction is re-estimated on test. Constructors are provided for the forms
the chain uses: a signature with one weight, a level, a template over a
recurring position (line number), a free parameter per member (the
over-parameterised control), and a scanned family (a relaxation or a ring).
`residual_floor` judges what the ladder leaves - the floor, the whiteness,
and whether the remainder reproduces - and the runner in
`tools/ringing_measure/holdout.py` reports both side by side.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# THE FIELD-TO-HEAD RULE. A two-head helical drum lays one field per head
# pass, so consecutive fields alternate heads - `information_extrapolation.
# field_cadence` places head parity at one half cycle per field. Which parity
# is head A cannot be read from the field count alone, so heads are LABELS
# here, kept apart and never named beyond that.
FIELDS_PER_HEAD_CYCLE = 2

# THE CHANCE LEVEL, in spreads of the null. An entry is accepted when its
# held-out descent stands above the null's median by this many of the null's
# own standard deviations. Three is the arc's standing convention for
# "stands above its noise" - `tesseract.identified` (sigma = 3.0),
# `residual_floor.verdict` (at the floor within three error bars),
# `residual_floor.whiteness` (lag one against three times 1 / root N) - and
# what it buys here is measured on this module's own control (2026-09-05,
# eight one-parameter echo entries per seed on pure complex noise, 512 bins
# in six bands, pooled over every seed run to date):
#
#     eight null draws        5 of 1280 entries falsely accepted (0.39%)
#     twenty-four draws       1 of 2280 (0.04%), of which 0 of 1200 on the
#                             150 seeds run last
#     noise correlated over   2 of 560 with eight draws (0.36%), the length
#     sixteen members         measured by `noise_correlation_length`
#
# The distribution of a chance descent is a product of two normals, so its
# tail is heavier than a Gaussian's and the rate at eight draws is the one
# to quote; twenty-four draws is what the runner uses by default and what
# these figures say it is worth.
CHANCE_SIGMA = 3.0

# Members held out for every block that is not in training: the default
# halves the record, and the block count below sets how finely the halves
# alternate. Six blocks give three contiguous runs per bank, so each bank
# spans the whole record while every run is long against a slow wander.
DEFAULT_FRACTION = 0.5
DEFAULT_BLOCKS = 6


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


@dataclass
class Split:
    """A held-out split: per group, contiguous blocks assigned to train or
    test. `groups` maps a head label to its index sets; nothing in it pools
    two heads. Each group also carries `members`, the ordered sequence its
    banks were drawn from, which is what blockedness is measured against."""
    by: str
    groups: Dict[object, Dict[str, object]]
    blocks: int
    fraction: float
    seed: int
    description: str

    def train(self, group=None) -> np.ndarray:
        return self._bank("train", group)

    def test(self, group=None) -> np.ndarray:
        return self._bank("test", group)

    def _bank(self, bank: str, group) -> np.ndarray:
        if group is None:
            if len(self.groups) != 1:
                raise ValueError(
                    "this split holds more than one head; name the group "
                    "rather than pooling them - the split must never pool "
                    "heads")
            group = next(iter(self.groups))
        return np.asarray(self.groups[group][bank], dtype=np.int64)


def _balanced_pattern(blocks: int, train_blocks: int, seed: int
                      ) -> np.ndarray:
    """Which blocks train: evenly spaced through the record so both banks
    span it, rotated by the seed so different seeds hold different runs out.

    Evenly spaced rather than random, because a random choice can put every
    training block at one end of the record and leave the other end wholly
    held out - which measures extrapolation across a trend, not the model.
    """
    blocks = int(blocks)
    train_blocks = int(train_blocks)
    if train_blocks <= blocks // 2:
        stride = blocks / float(train_blocks)
        chosen = {int(np.floor(k * stride)) for k in range(train_blocks)}
        pattern = np.array([i in chosen for i in range(blocks)])
    else:
        # choose the TEST blocks evenly instead, and take the complement
        stride = blocks / float(blocks - train_blocks)
        chosen = {int(np.floor(k * stride))
                  for k in range(blocks - train_blocks)}
        pattern = np.array([i not in chosen for i in range(blocks)])
    rotate = int(seed) % blocks
    return np.roll(pattern, rotate)


def _contiguous_blocks(ordered_members: np.ndarray, count: int
                       ) -> List[np.ndarray]:
    """Split an ordered member list into `count` contiguous runs of nearly
    equal size, dropping empty runs."""
    parts = np.array_split(np.asarray(ordered_members, dtype=np.int64),
                           max(int(count), 1))
    return [part for part in parts if part.size]


def _field_blocks(members: np.ndarray, fields: np.ndarray, count: int
                  ) -> List[np.ndarray]:
    """Contiguous blocks of whole FIELDS: every line of a field lands in the
    same block, and the blocks follow field order."""
    distinct = np.unique(fields[members])
    runs = _contiguous_blocks(np.arange(distinct.size), count)
    out = []
    for run in runs:
        wanted = distinct[run]
        chosen = members[np.isin(fields[members], wanted)]
        if chosen.size:
            out.append(chosen)
    return out


def split(count: Optional[int] = None, *, fields=None, heads=None,
          by: str = "field", within_head: bool = True,
          fraction: float = DEFAULT_FRACTION, blocks: int = DEFAULT_BLOCKS,
          seed: int = 0) -> Split:
    """A held-out split that stays within a head and is blocked in time.

    `count` is the number of members, or give `fields`: the field each
    member belongs to (one entry per member; for per-line data many members
    share a field). `heads` labels each member's head; for `by="field"` and
    `by="line"` it defaults to the field's parity, because fields alternate
    heads. `by="band"` is for data pooled over fields already - a per-head
    export over frequency, with no time axis left to block - where the
    members are bins and the blocks are contiguous bands; it takes no heads,
    and what it measures is interpolation across frequency, not
    reproduction in time. Where an export carries two halves in time, use
    `half_split` instead.

    `within_head=False` is refused whenever more than one head is present.
    That is not a missing feature: pooling heads is the measured trap this
    module exists to prevent, and a caller who wants a pooled number can
    average the per-head results and say so.

    Returns a `Split` whose groups are the heads, each with `train`, `test`,
    the blocks each bank was built from (for the block jackknife), and
    `members`, the ordered sequence the banks were drawn from.
    """
    if by not in ("field", "line", "band"):
        raise ValueError("by must be 'field', 'line' or 'band'")
    if fields is not None:
        fields = np.asarray(fields).ravel()
        size = fields.size
    elif count is not None:
        size = int(count)
        fields = np.arange(size) if by != "band" else None
    else:
        raise ValueError("give count or fields")
    if size < 2:
        raise ValueError("fewer than two members cannot be split")
    members = np.arange(size, dtype=np.int64)

    if by == "band":
        if heads is not None and np.unique(np.asarray(heads)).size > 1:
            raise ValueError(
                "a band split is for one head's pooled export; pass each "
                "head separately - the split must never pool heads")
        group_of = {"pooled": members}
    else:
        if heads is None:
            heads = (fields % FIELDS_PER_HEAD_CYCLE)
        heads = np.asarray(heads).ravel()
        if heads.size != size:
            raise ValueError("heads must label every member")
        labels = np.unique(heads)
        if not within_head and labels.size > 1:
            raise ValueError(
                "the split must never pool heads: pooled across heads the "
                "vertical interval's DC offset tracked the response tilt at "
                "r = -0.694 and within each head the relationship vanished "
                "(-0.025 and -0.144); evaluate per head and average the "
                "results if a pooled figure is wanted")
        group_of = {label: members[heads == label] for label in labels}

    blocks = max(int(blocks), 2)
    train_blocks = int(round(float(fraction) * blocks))
    train_blocks = min(max(train_blocks, 1), blocks - 1)
    pattern = _balanced_pattern(blocks, train_blocks, seed)

    groups: Dict[object, Dict[str, object]] = {}
    for label, chosen in group_of.items():
        if by == "band":
            runs = _contiguous_blocks(chosen, blocks)
        elif by == "field":
            runs = _field_blocks(chosen, fields, blocks)
        else:
            # by line: contiguous runs of lines inside each field, so that a
            # field is never wholly in one bank and the runs stay blocked
            runs = []
            for field_id in np.unique(fields[chosen]):
                lines = chosen[fields[chosen] == field_id]
                runs.extend(_contiguous_blocks(lines, blocks))
        assignment = np.resize(pattern, len(runs)) if runs else np.zeros(0, bool)
        if by == "line":
            # the pattern restarts in every field: pattern index = run's
            # position inside its field
            assignment = np.zeros(len(runs), dtype=bool)
            position = 0
            last_field = None
            for k, run in enumerate(runs):
                this_field = fields[run[0]]
                if this_field != last_field:
                    position = 0
                    last_field = this_field
                assignment[k] = pattern[position % blocks]
                position += 1
        train_runs = [run for run, is_train in zip(runs, assignment)
                      if is_train]
        test_runs = [run for run, is_train in zip(runs, assignment)
                     if not is_train]
        if not train_runs or not test_runs:
            # too few members for the requested blocks: halve
            halves = _contiguous_blocks(chosen, 2)
            train_runs, test_runs = halves[:1], halves[1:]
        groups[label] = {
            "train": (np.concatenate(train_runs) if train_runs
                      else np.zeros(0, np.int64)),
            "test": (np.concatenate(test_runs) if test_runs
                     else np.zeros(0, np.int64)),
            "train_blocks": train_runs,
            "test_blocks": test_runs,
            "members": np.asarray(chosen, dtype=np.int64),
        }

    unit = {"field": "fields", "line": "lines", "band": "bins"}[by]
    description = (
        f"split by {by}: {len(groups)} head group(s) kept apart, "
        f"{blocks} contiguous blocks of {unit} per group with "
        f"{train_blocks} in training, banks alternating, seed {seed}")
    return Split(by=by, groups=groups, blocks=blocks,
                 fraction=float(fraction), seed=int(seed),
                 description=description)


def half_split(count: int, blocks: int = DEFAULT_BLOCKS) -> Split:
    """The time-blocked split an export of two halves already carries.

    The first half of the fields trains and the second judges, both measured
    on the same members - the bins of one head's response. Members
    `0 .. count-1` are the first half and `count .. 2 count-1` the second,
    so the values handed to `frozen` are the two halves stacked in that
    order and an entry's basis is its shape repeated for each half
    (`entries_from_key` does that through `bin_of`). Two contiguous blocks
    of fields, one per bank: the coarsest time split there is, within one
    head by construction, and the one `residual_floor.test` and
    `tesseract.from_sync_exports` are built on. The jackknife blocks are
    contiguous bands within each half, so the error bar on a held-out
    residual is its variation across bands.
    """
    count = int(count)
    if count < 2:
        raise ValueError("fewer than two shared members cannot be judged")
    first = np.arange(count, dtype=np.int64)
    second = count + first
    groups = {"pooled": {
        "train": first, "test": second,
        "train_blocks": _contiguous_blocks(first, blocks),
        "test_blocks": _contiguous_blocks(second, blocks),
        "members": np.arange(2 * count, dtype=np.int64),
    }}
    return Split(by="half", groups=groups, blocks=int(blocks), fraction=0.5,
                 seed=0, description=(
                     f"split by half: the first half of the fields trains and "
                     f"the second judges, on {count} shared members; "
                     f"{int(blocks)} contiguous bands per half for the "
                     f"jackknife"))


def pools_heads(split_: Split, heads) -> bool:
    """Whether any bank of any group holds members of two heads. The tests
    hold this at False for every split this module makes."""
    heads = np.asarray(heads).ravel()
    for group in split_.groups.values():
        for bank in ("train", "test"):
            index = np.asarray(group[bank], dtype=np.int64)
            if index.size and np.unique(heads[index]).size > 1:
                return True
    return False


def runs_of(index, members=None) -> int:
    """How many contiguous runs an index set is made of, counted against
    the ordered sequence it was drawn from - the measure of blockedness the
    tests use. An interleaved split of n members has about n runs; a blocked
    one has as many as its blocks.

    `members` is that sequence (a group's `members`). Without it the
    sequence is taken to be every integer, which is right for a band split
    and wrong for a head group: one head's fields are every other field, so
    its members are never contiguous in the global index and a correctly
    blocked bank of ten fields counted as ten runs there.
    """
    index = np.unique(np.asarray(index, dtype=np.int64))
    if index.size == 0:
        return 0
    if members is None:
        return int(1 + np.sum(np.diff(index) > 1))
    members = np.unique(np.asarray(members, dtype=np.int64))
    place = np.searchsorted(members, index)
    inside = place < members.size
    if not inside.all() or not np.array_equal(members[place[inside]], index):
        raise ValueError("the index holds members outside the sequence it "
                         "is being counted against")
    return int(1 + np.sum(np.diff(place) > 1))


# --------------------------------------------------------------------------
# The noise's correlation along the member axis
# --------------------------------------------------------------------------


def noise_correlation_length(sample, other=None) -> Dict[str, object]:
    """The integral correlation length of the noise along the member axis,
    in members: one plus twice the sum of the autocorrelation over the lags
    where it stands above its own noise, which is the factor by which
    correlated members count for fewer than independent ones.

    `other`, when given, is a second independent measurement of the same
    members - the other half of a split export - and the difference of the
    two is taken, which removes whatever both share (the structure) and
    leaves the noise alone. That is the only way to read the noise's
    correlation from a residual whose structure is itself correlated, and
    it is why the sync exports carry `H_first` and `H_second`. Without
    `other`, `sample` must already be noise.

    The lag sum stops at the first lag whose autocorrelation is not above
    `1 / root N`, the level a white series reaches by chance and the null
    `residual_floor.whiteness` uses. Measured on the sync exports (fall
    view, 944 bins at 3.5 kHz over 0.2-3.5 MHz, whitened, the halves'
    difference): 74 and 58 members on cd, 70 and 51 on home, 69 and 52 on
    pnb (heads a and b), against the 52 the exports' stated resolution
    of 0.181 MHz (a 5.5 us window) implies. The lag-one autocorrelation is
    0.99 on all six: neighbouring bins are nearly one measurement.
    """
    x = np.asarray(sample).ravel()
    if other is not None:
        y = np.asarray(other).ravel()
        if y.size != x.size:
            raise ValueError("the two measurements must cover the same "
                             "members")
        x = x - y
    x = x[np.isfinite(x)]
    n = int(x.size)
    out: Dict[str, object] = {"members": n, "paired": other is not None}
    if n < 4:
        out.update(length=1.0, lags=0, null=float("nan"),
                   autocorrelation=np.zeros(0),
                   why="too few members to measure a correlation; white")
        return out
    x = x - x.mean()
    power = float(np.real(np.vdot(x, x)))
    if power <= 0:
        out.update(length=1.0, lags=0, null=float("nan"),
                   autocorrelation=np.zeros(0),
                   why="no variation to correlate; white")
        return out
    null = 1.0 / np.sqrt(n)
    profile = []
    total = 0.0
    lags = 0
    for k in range(1, n // 2):
        rho = float(np.real(np.vdot(x[:-k], x[k:]))) / power
        profile.append(rho)
        if rho <= null:
            break
        total += rho
        lags = k
    length = 1.0 + 2.0 * total
    out.update(length=float(length), lags=int(lags), null=float(null),
               autocorrelation=np.asarray(profile),
               why=(f"the autocorrelation stands above 1/root N for {lags} "
                    f"lag(s); {length:.1f} members share one noise draw"))
    return out


# --------------------------------------------------------------------------
# The estimator contract
# --------------------------------------------------------------------------


@dataclass
class Entry:
    """One estimator: how it fits, how it predicts, how many parameters it
    holds, and how to scramble its signature for the null. `scramble` takes
    the generator, the kind of surrogate, and the noise's correlation length
    along the member axis."""
    name: str
    fit: Callable[[np.ndarray, np.ndarray], object]
    predict: Callable[[np.ndarray, object], np.ndarray]
    size: int
    parameters: Callable[[object], int]
    scramble: Optional[Callable[..., "Entry"]] = None
    position: Optional[int] = None
    kind: str = "entry"
    notes: str = ""


def _rms(values) -> float:
    values = np.asarray(values)
    good = np.isfinite(values)
    if not good.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.abs(values[good]) ** 2)))


def _least_squares(basis: np.ndarray, target: np.ndarray):
    """Weights for `basis @ w ~ target`. A complex basis gets REAL weights
    through real/imaginary stacking - the arc's rule that a physical
    parameter scales a log-shape and does not rotate it, and the form
    `stage_boundaries.estimate` found a Hermitian product gets wrong. That
    solve IS `residual_floor.fit`'s, called with a unit floor because the
    basis and the values here are whitened before they arrive, so the rule
    lives in one place. A real basis with complex targets gets complex
    weights, one real solve per part."""
    from vhsdecode.models import residual_floor

    basis = np.asarray(basis)
    target = np.asarray(target)
    if basis.ndim == 1:
        basis = basis[:, None]
    if basis.shape[0] == 0:
        return None
    if np.iscomplexobj(basis):
        fitted = residual_floor.fit(np.asarray(target, dtype=np.complex128),
                                    np.ones(basis.shape[0]), basis)
        return np.asarray(fitted["coefficients"], dtype=np.float64)
    if np.iscomplexobj(target):
        real, *_ = np.linalg.lstsq(basis, target.real, rcond=None)
        imag, *_ = np.linalg.lstsq(basis, target.imag, rcond=None)
        return (real + 1j * imag).astype(np.complex128)
    weights, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return weights.astype(np.float64)


def _circular_mean(values: np.ndarray, width: int) -> np.ndarray:
    """A boxcar mean of `width` bins on a periodic axis; the total is
    preserved, so a spectrum smoothed this way keeps its power."""
    width = int(width)
    n = values.size
    if width <= 1 or n < 2:
        return values
    if width >= n:
        return np.full(n, values.mean())
    half = width // 2
    extended = np.concatenate([values[-half:], values,
                               values[:width - half - 1]])
    return np.convolve(extended, np.ones(width) / width, mode="valid")


def _phase_surrogate(x: np.ndarray, rng: np.random.Generator,
                     correlation_length: float) -> np.ndarray:
    """One column's surrogate: the power spectrum along the member axis,
    smoothed over `n / correlation_length` bins so that the autocorrelation
    survives only to the noise's correlation length, with every phase
    randomised. A real column stays real."""
    n = x.size
    if n < 2:
        return x.copy()
    width = int(np.clip(round(n / max(float(correlation_length), 1.0)),
                        1, n))
    spectrum = np.fft.fft(x)
    amplitude = np.sqrt(np.maximum(
        _circular_mean(np.abs(spectrum) ** 2, width), 0.0))
    if np.iscomplexobj(x):
        return np.fft.ifft(amplitude * np.exp(2j * np.pi * rng.random(n)))
    half = n // 2
    phases = np.exp(2j * np.pi * rng.random(half + 1))
    # the bins a real series must keep real take a sign, not a phase
    phases[0] = rng.choice([-1.0, 1.0])
    if n % 2 == 0:
        phases[half] = rng.choice([-1.0, 1.0])
    return np.fft.irfft(amplitude[: half + 1] * phases, n=n)


def surrogate(signature, rng: np.random.Generator, how: str = "phase",
              correlation_length: float = 1.0) -> np.ndarray:
    """A scrambled copy of a signature over the member axis.

    `phase` keeps the signature's autocorrelation up to
    `correlation_length` members - its power spectrum along that axis
    smoothed over `n / correlation_length` bins - and randomises every
    phase: a shape with the same chance projection on this noise and none
    of the actual shape. With a correlation length of one (a white member
    axis) that is a white shape of the same power; with the whole length it
    would be the exact spectrum, which for a single ripple is the ripple
    itself rotated, and that is the surrogate the first build used and the
    reason a planted echo's null descended with the echo.

    `permute` shuffles the members, which is white whatever the noise. It
    is exact on a white member axis and understates the chance level on a
    correlated one; it is offered as the control that needs no correlation
    length.
    """
    values = np.asarray(signature)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty_like(flat)
    for column in range(flat.shape[1]):
        x = flat[:, column]
        if how == "permute":
            out[:, column] = x[rng.permutation(x.size)]
        elif how == "phase":
            out[:, column] = _phase_surrogate(x, rng, correlation_length)
        else:
            raise ValueError("how must be 'phase' or 'permute'")
    return out.reshape(values.shape)


def _whitened(basis: np.ndarray, floor) -> np.ndarray:
    """The basis divided by the root of the per-member floor, so that an
    unweighted least squares on whitened values is `residual_floor.fit`'s
    weighted one, and every residual reads in units of its own floor."""
    if floor is None:
        return basis
    weight = 1.0 / np.sqrt(np.maximum(
        np.asarray(floor, dtype=np.float64).ravel(), 1e-300))
    if weight.size != basis.shape[0]:
        raise ValueError("the floor must give every member a value")
    return basis * weight[:, None]


def linear_entry(name: str, basis, position: Optional[int] = None,
                 notes: str = "", floor=None, member_of=None) -> Entry:
    """A signature (or a few) with one weight each, fitted by least squares.

    `basis` is (members,) or (members, columns) over every member of the
    data, real or complex. Complex columns get a real weight (see
    `_least_squares`). `floor`, when given, is the noise variance per
    member: the basis is whitened by it, and the values handed to `frozen`
    must be whitened the same way (`residual_floor.log_domain` gives both
    the log values and the floor).

    `member_of` says that the members are not the signature's own axis:
    `basis` is then given over the underlying positions (the bins of one
    response) and `member_of[m]` is the position member `m` carries, so
    the same shape is applied wherever a position recurs - the two halves
    of `half_split`, where it is the bin index repeated. THE NULL IS MADE
    ON THAT AXIS AND MAPPED THE SAME WAY, and that is the reason the axis
    is declared rather than the caller indexing the basis before it
    arrives: a surrogate made over the stacked members gives the first
    half one random shape and the second half another (their correlation
    measured 0.003 on a planted echo, 2026-09-05), so the null lost the
    one property the half split judges - that a frozen shape fitted on the
    first half is the shape applied to the second. Its descent was the
    product of two independent chances, symmetric about zero and wide;
    a repeated surrogate's is the positive share of the reproducing
    structure a wrong shape captures, and narrower: on cd head a's
    `dropout` entry the stacked null's spread was 3.0 floor units against
    the repeated null's 1.69, a bar of 8.99 against 5.61 on a descent of
    8.86 (the module docstring has the counts). The surrogate is made over
    the raw signature and whitened after, so a null column carries the
    whitening's structure exactly as the real column does; and it keeps
    the signature's own roughness, which for an entry of the entry-wise
    admitted key - the remainder of a smooth signature after the key
    before it is projected out - is white (`tape path clearance` measures
    a correlation length of one bin), so a rough entry is judged against
    random rough shapes of its own spectrum.
    """
    shape = np.asarray(basis)
    if shape.ndim == 1:
        shape = shape[:, None]
    if member_of is not None:
        member_of = np.asarray(member_of, dtype=np.int64).ravel()
        if member_of.min() < 0 or member_of.max() >= shape.shape[0]:
            raise ValueError("member_of names a position the basis does "
                             "not have")
        applied = shape[member_of]
    else:
        applied = shape
    applied = _whitened(applied, floor)
    size = int(applied.shape[0])

    def fit(values, index):
        return _least_squares(applied[index], values)

    def predict(index, weights):
        if weights is None:
            return np.zeros(len(index), dtype=applied.dtype)
        return applied[index] @ weights

    def scramble(rng, how, correlation_length=1.0):
        return linear_entry(f"{name} [scrambled]",
                            surrogate(shape, rng, how, correlation_length),
                            position=position, notes="null", floor=floor,
                            member_of=member_of)

    return Entry(name=name, fit=fit, predict=predict, size=size,
                 parameters=lambda w: int(applied.shape[1]),
                 scramble=scramble, position=position, kind="linear",
                 notes=notes)


def constant_entry(name: str, size: int, complex_valued: bool = False,
                   position: Optional[int] = None, floor=None,
                   member_of=None) -> Entry:
    """The level: one constant per group, which is the per-head DC term. On
    complex values the weight is complex (a gain and a rotation together).

    Its null is the same as every other entry's: a random shape of the
    level's power with the noise's correlation length, because the chance
    descent of one parameter on this noise does not depend on the parameter
    being a constant. The first build gave the level no null at all (the
    model with no parameter, descent zero), which accepted a level fitted
    to noise half the time. With `member_of` (see `linear_entry`) the
    constant is declared over the underlying positions, so its null is one
    random shape repeated wherever a position recurs.
    """
    dtype = np.complex128 if complex_valued else np.float64
    positions = (int(np.asarray(member_of).max()) + 1
                 if member_of is not None else int(size))
    entry = linear_entry(name, np.ones(positions, dtype=np.float64),
                         position=position, floor=floor, member_of=member_of)
    entry.kind = "level"
    entry.notes = f"dtype {np.dtype(dtype).name}"
    return entry


def template_entry(name: str, labels, position: Optional[int] = None,
                   notes: str = "") -> Entry:
    """One free parameter per LABEL: the mean of the members carrying it.

    Whether this is a legitimate model or the over-parameterised control
    depends only on whether its labels recur in the held-out data:

      labels that recur    a line number: every field carries every line, so
                           a template fitted on training fields is applied
                           on test fields with nothing re-estimated. If the
                           per-line structure is real the held-out residual
                           falls; if it was noise it rises.
      labels that do not   a field number: a test field has no parameter,
                           and a member whose label was never fitted gets
                           NO prediction - the template's claim is per
                           label, and what it has not seen it leaves alone.
                           In-sample falls by the whole per-field scatter;
                           held-out cannot fall at all. That is exactly the
                           over-parameterised model Ethan's rule refuses,
                           and `ladder` refuses it.

    The first build predicted the mean of the fitted parameters for an
    unseen label, which is a level in disguise: after a relaxation fitted
    without an offset had left a small level behind, the per-line control
    picked that level up on held-out fields and was accepted (its own test,
    2026-09-05). A control that estimates a level is not a control.

    The null deals the labels to the wrong members: the same parameter
    count with the structural claim removed.
    """
    labels = np.asarray(labels).ravel()
    size = int(labels.size)

    def fit(values, index):
        values = np.asarray(values)
        distinct, inverse, counts = np.unique(
            labels[index], return_inverse=True, return_counts=True)
        if distinct.size == 0:
            return None
        totals = np.zeros(distinct.size, dtype=values.dtype)
        np.add.at(totals, inverse, values)
        means = totals / counts
        return {"labels": distinct, "means": means}

    def predict(index, parameters):
        if not parameters:
            return np.zeros(len(index))
        distinct, means = parameters["labels"], parameters["means"]
        wanted = labels[index]
        place = np.searchsorted(distinct, wanted)
        place = np.minimum(place, distinct.size - 1)
        seen = distinct[place] == wanted
        return np.where(seen, means[place], 0.0)

    def scramble(rng, how, correlation_length=1.0):
        return template_entry(f"{name} [scrambled]",
                              labels[rng.permutation(size)],
                              position=position, notes="null")

    return Entry(name=name, fit=fit, predict=predict, size=size,
                 parameters=lambda p: int(p["labels"].size) if p else 0,
                 scramble=scramble, position=position, kind="template",
                 notes=notes)


def entries_from_key(key: Dict[str, np.ndarray],
                     chain: Optional[Dict[str, int]] = None,
                     order: Optional[Sequence[Tuple[int, str]]] = None,
                     bin_of=None, floor=None) -> List[Entry]:
    """The chain's key as ladder entries, in the order a correction removes
    them: `interference.ordered_key` - last applied, first removed - unless
    an `order` is given. Each signature enters as its log shape (log
    magnitude and unwrapped phase, means removed), the form the arc reads
    every ensemble in, with one real weight.

    `bin_of` maps each member to the key's bin, for members that are not
    the bins themselves - the two halves of `half_split` stacked, where it
    is the bin index repeated. It is handed to `linear_entry` as the
    entry's underlying axis rather than used to index the shape here, so
    that the null is one scrambled shape repeated on both halves (see
    `linear_entry`). `floor` whitens (see `linear_entry`).
    """
    from vhsdecode.models import interference as inf

    if order is None:
        order = inf.ordered_key(key, chain=chain)
    out = []
    for position, name in order:
        shape = inf._log_shape(np.asarray(key[name]))
        out.append(linear_entry(name, shape, position=int(position),
                                floor=floor, member_of=bin_of))
    return out


def scan_entry(name: str, family: Callable[[np.ndarray, float], np.ndarray],
               grid: Sequence[float], size: int, offset: bool = False,
               grid_position=None, position: Optional[int] = None,
               notes: str = "") -> Entry:
    """A one-parameter family scanned, with linear weights at each step.

    `family(positions, parameter)` returns the basis over `positions` - a
    relaxation `exp(-t / tau)` scanned over tau, a ring's cosine and sine
    pair scanned over decay and frequency (pack two parameters into one grid
    entry). The scan keeps the parameter whose linear fit has the least
    in-sample residual; `offset` adds a fitted constant. Both the scanned
    parameter and the weights are frozen for prediction.

    `grid_position` says which position on a SHARED grid each member sits
    at, for data where many members (the lines of a field) carry the same
    time axis. `family` is then called with grid positions rather than
    member indices, and the fit is done on the members' mean profile with
    each position weighted by how many members carry it - which is the same
    least squares as the full problem, exactly: the per-member scatter about
    the profile is orthogonal to every shape on the grid and only adds a
    constant to the sum of squares. It is the difference between one solve
    per scan step on a few hundred grid points and one on half a million
    samples.

    Exact in the sense `sync_geometry.relaxation_or_ringing` relies on: the
    fit is linear once the scanned parameter is fixed, so no optimiser and
    no starting point. The null surrogates the family's shapes along the
    grid with one phase pattern per draw, so it stays a one-parameter family
    of shapes of the noise's correlation length rather than a fresh random
    vector per step.
    """
    grid = list(grid)
    size = int(size)
    shared = (np.asarray(grid_position, dtype=np.int64).ravel()
              if grid_position is not None else None)
    everyone = (np.arange(int(shared.max()) + 1) if shared is not None
                else np.arange(size))

    def positions_of(index):
        return shared[index] if shared is not None else np.asarray(index)

    def design(positions, parameter):
        basis = np.asarray(family(positions, parameter))
        if basis.ndim == 1:
            basis = basis[:, None]
        if offset:
            basis = np.column_stack([basis, np.ones(len(positions))])
        return basis

    def fit(values, index):
        values = np.asarray(values)
        positions = positions_of(index)
        if shared is not None:
            # the mean profile over the shared grid, weighted by occupancy
            distinct, inverse, counts = np.unique(
                positions, return_inverse=True, return_counts=True)
            profile = np.zeros(distinct.size, dtype=values.dtype)
            np.add.at(profile, inverse, values)
            profile = profile / counts
            weight = np.sqrt(counts.astype(np.float64))
            target = profile * weight
        else:
            distinct, target, weight = positions, values, None
        best = None
        for parameter in grid:
            basis = design(distinct, parameter)
            weighted = basis * weight[:, None] if weight is not None else basis
            weights = _least_squares(weighted, target)
            if weights is None:
                continue
            if shared is not None:
                residual = _rms(values - basis[inverse] @ weights)
            else:
                residual = _rms(values - basis @ weights)
            if best is None or residual < best["rms"]:
                best = {"parameter": parameter, "weights": weights,
                        "rms": residual}
        return best

    def predict(index, parameters):
        if not parameters:
            return np.zeros(len(index))
        return (design(positions_of(index), parameters["parameter"])
                @ parameters["weights"])

    def scramble(rng, how, correlation_length=1.0):
        seed = int(rng.integers(0, 2 ** 31 - 1))

        def scrambled_family(positions, parameter):
            full = np.asarray(family(everyone, parameter))
            local = np.random.default_rng(seed)
            return surrogate(full, local, how,
                             correlation_length)[np.asarray(positions)]

        return scan_entry(f"{name} [scrambled]", scrambled_family, grid, size,
                          offset=offset, grid_position=shared,
                          position=position, notes="null")

    def count(parameters):
        if not parameters:
            return 0
        weights = np.asarray(parameters["weights"])
        return int(weights.size + 1)

    return Entry(name=name, fit=fit, predict=predict, size=size,
                 parameters=count, scramble=scramble, position=position,
                 kind="scan", notes=notes)


# --------------------------------------------------------------------------
# Frozen evaluation
# --------------------------------------------------------------------------


def _groups(split_or_train, test=None) -> Dict[object, Dict[str, object]]:
    if isinstance(split_or_train, Split):
        return split_or_train.groups
    if test is None:
        raise ValueError("give a Split, or both train and test index arrays")
    train = np.asarray(split_or_train, dtype=np.int64).ravel()
    test = np.asarray(test, dtype=np.int64).ravel()
    return {"given": {"train": train, "test": test,
                      "train_blocks": [train], "test_blocks": [test],
                      "members": np.union1d(train, test)}}


def _block_jackknife(statistic: Callable[[np.ndarray], float],
                     members: np.ndarray, blocks: Sequence[np.ndarray]
                     ) -> float:
    """The standard error of a statistic over `members`, by leaving out
    each block in turn. Blocks rather than samples, because neighbouring
    members are correlated and a sample jackknife would understate the
    error. `statistic` receives the mask of members kept."""
    blocks = [np.asarray(b, dtype=np.int64) for b in blocks if len(b)]
    if len(blocks) < 2:
        return float("nan")
    members = np.asarray(members, dtype=np.int64)
    leave_out = []
    for block in blocks:
        keep = ~np.isin(members, block)
        if keep.any():
            leave_out.append(float(statistic(keep)))
    if len(leave_out) < 2:
        return float("nan")
    leave_out = np.asarray(leave_out)
    count = leave_out.size
    return float(np.sqrt((count - 1) / count
                         * np.sum((leave_out - leave_out.mean()) ** 2)))


def _finite(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    return index[np.isfinite(values[index])]


def frozen(entry: Entry, values, split_or_train, test=None, group=None
           ) -> Dict[object, Dict[str, object]]:
    """Fit on train, freeze every parameter, evaluate on test.

    Per group: the in-sample residual (train, under the parameters it
    fitted), the held-out residual (test, under THE SAME parameters), the
    residual each bank carried before the entry, the descent on test with
    a block-jackknife error bar, a block-jackknife error bar on the held-out
    rms itself, and the ratio of held-out to in-sample. `lowered` says
    whether the held-out residual fell at all and `reading` says it in
    words with the descent against its own error bar; the verdict that
    also needs the null - whether the descent is more than a wrong shape
    reaches - is `verdict`, taken in `ladder`.
    """
    values = np.asarray(values)
    out: Dict[object, Dict[str, object]] = {}
    for label, bank in _groups(split_or_train, test).items():
        if group is not None and label != group:
            continue
        train = _finite(values, bank["train"])
        held = _finite(values, bank["test"])
        parameters = entry.fit(values[train], train) if train.size else None
        residual_train = values[train] - entry.predict(train, parameters)
        residual_test = values[held] - entry.predict(held, parameters)
        in_sample = _rms(residual_train)
        held_out = _rms(residual_test)
        before_test = _rms(values[held])
        blocks = bank.get("test_blocks", [held])
        descent = before_test - held_out
        descent_se = _block_jackknife(
            lambda keep: _rms(values[held][keep]) - _rms(residual_test[keep]),
            held, blocks)
        if not np.isfinite(descent):
            reading = "no held-out descent could be measured"
        elif descent <= 0:
            reading = "held-out did not fall: refused before any null"
        else:
            sigma = (descent / descent_se if np.isfinite(descent_se)
                     and descent_se > 0 else float("nan"))
            reading = (f"held-out fell by {descent:.4g} +- {descent_se:.2g} "
                       f"({sigma:.1f} sigma of its block jackknife); "
                       f"accepted only if that clears the null's chance "
                       f"level")
        out[label] = {
            "entry": entry.name,
            "parameters": parameters,
            "count": int(entry.parameters(parameters)),
            "in_sample_before": _rms(values[train]),
            "in_sample": in_sample,
            "held_out_before": before_test,
            "held_out": held_out,
            "held_out_se": _block_jackknife(
                lambda keep: _rms(residual_test[keep]), held, blocks),
            "descent": descent,
            "descent_se": descent_se,
            "ratio": (held_out / in_sample if in_sample > 0
                      else float("nan")),
            "lowered": bool(held_out < before_test),
            "reading": reading,
            "train_members": int(train.size),
            "test_members": int(held.size),
            "residual_train": residual_train,
            "residual_test": residual_test,
            "train_index": train,
            "test_index": held,
        }
    return out


def null_holdout(entry: Entry, values, split_or_train, test=None,
                 group=None, draws: int = 24, seed: int = 0,
                 scramble: str = "phase", correlation_length: float = 1.0
                 ) -> Dict[object, Dict[str, object]]:
    """The surrogate null: the same split, the same frozen fit, the entry's
    signature scrambled.

    Each draw scrambles the entry - its basis surrogated over the members
    to the noise's correlation length, a template's labels dealt to the
    wrong members, a scanned family's shapes surrogated with one phase
    pattern - and fits it on train exactly as the real entry was, then
    reads the held-out residual. What comes back is the distribution of
    held-out descents a model with this many parameters and this
    smoothness reaches when its shape is wrong: the level an
    over-parameterised model reaches on noise alone. `descent_median` and
    `descent_spread` (the standard deviation over draws) are what
    `chance_level` turns into the bar a real entry has to clear.
    """
    if entry.scramble is None:
        raise ValueError(f"{entry.name} has no scramble; give the entry one "
                         "or build it with a constructor from this module")
    values = np.asarray(values)
    out: Dict[object, Dict[str, object]] = {}
    for label, bank in _groups(split_or_train, test).items():
        if group is not None and label != group:
            continue
        rng = np.random.default_rng(seed)
        held_out, in_sample, descent = [], [], []
        for _ in range(max(int(draws), 2)):
            null_entry = entry.scramble(rng, scramble, correlation_length)
            got = frozen(null_entry, values, split_or_train, test, label)
            if label not in got:
                continue
            held_out.append(got[label]["held_out"])
            in_sample.append(got[label]["in_sample"])
            descent.append(got[label]["descent"])
        held_out = np.asarray(held_out, dtype=np.float64)
        in_sample = np.asarray(in_sample, dtype=np.float64)
        descent = np.asarray(descent, dtype=np.float64)

        def middle(x):
            return float(np.median(x)) if x.size else float("nan")

        out[label] = {
            "draws": int(held_out.size),
            "scramble": scramble,
            "correlation_length": float(correlation_length),
            "held_out_draws": held_out,
            "in_sample_draws": in_sample,
            "descent_draws": descent,
            "held_out_median": middle(held_out),
            "in_sample_median": middle(in_sample),
            "descent_median": middle(descent),
            "descent_spread": (float(np.std(descent, ddof=1))
                               if descent.size > 1 else float("nan")),
        }
    return out


def chance_level(null: Dict[str, object], sigma: float = CHANCE_SIGMA
                 ) -> float:
    """The held-out descent an entry must exceed: the null's median descent,
    floored at zero, plus `sigma` of the null's spreads. Floored, because a
    null that does harm (a negative median) must not license an entry that
    does nothing."""
    median = float(null["descent_median"])
    spread = float(null["descent_spread"])
    if not (np.isfinite(median) and np.isfinite(spread)):
        return float("nan")
    return max(median, 0.0) + float(sigma) * spread


def verdict(result: Dict[str, object], null: Dict[str, object],
            sigma: float = CHANCE_SIGMA) -> str:
    """Accepted only when the held-out residual fell AND fell further than
    the null's chance level. Anything else is refused, and the refusal names
    why."""
    descent = float(result["descent"])
    bar = chance_level(null, sigma)
    if not np.isfinite(descent):
        return "refused: no held-out descent could be measured"
    if not np.isfinite(bar):
        return "refused: the null has no spread to judge against"
    if descent <= 0:
        return "refused: held-out did not fall"
    if descent <= bar:
        return "refused: within the null"
    return "accepted"


def ladder(entries: Sequence[Entry], values, split_or_train, test=None,
           draws: int = 24, seed: int = 0, scramble: str = "phase",
           correlation_length: float = 1.0, sigma: float = CHANCE_SIGMA,
           enforce: bool = True) -> Dict[object, Dict[str, object]]:
    """The ordered chain, applied frozen entry by entry, with the held-out
    descent reported at every step and every refused entry rolled back.

    The order is the caller's, because it is known circuitry: the chain's
    declared positions through `interference.ordered_key` (last applied,
    first removed) for the frequency axis, and the consistency loop's order
    - the level, then timing, then the spectral parts, emphasis last - on
    the time axis. Ethan named the loop matching pursuit; the order being
    known is what makes it a traversal rather than a search.

    An entry that lowers the held-out residual beyond the null's chance
    level is accepted, and the residual it leaves - train under its own
    parameters, test under the same frozen parameters - is what the next
    entry sees. An entry that does not is refused and the residual is left
    exactly as it was, so a refused entry costs nothing downstream and can
    be read in the table for what it would have done.

    THE TRAVERSAL IS GREEDY, AND ON A COLLINEAR KEY THAT COSTS SOMETHING
    REAL. Each entry is offered alone against the residual its predecessors
    left, so an entry that carries nothing by itself but a great deal in
    company is refused. Measured on the sync exports (2026-09-05, fall view,
    944 bins, the entry-wise admitted key of 55 entries behind a level and a
    delay, first half fitted and second judged): the ladder accepts 2 to 10
    entries and leaves 28.1 to 129.9 floor units, where the SAME key fitted
    all at once and frozen the same way leaves 16.1 to 91.6 -

        head-tape   ladder leaves   whole key at once   accepted
        cd a            69.2             27.0             3 of 57
        cd b            58.9             30.2            10 of 57
        home a         127.3             86.6             4 of 57
        home b         129.9             91.6             3 of 57
        pnb a           35.3             20.0             2 of 57
        pnb b           28.1             16.1             3 of 57

    - and the gap is not over-fitting, because the joint figure is held out
    too. It is the key's collinearity against a member axis that carries
    only 13 to 19 independent members once the measured noise correlation
    length (51 to 74 bins) is allowed for: a one-parameter projection on so
    few independent members has a wide chance distribution, so each entry's
    bar is high and few clear it alone. The ladder answers "did this entry
    earn its own place"; `residual_floor.test` answers "does the key as a
    whole carry to the other half", and the runner prints both because
    neither is the other's answer.

    `correlation_length` is the noise's along the member axis
    (`noise_correlation_length`); one means a white member axis.
    `enforce=False` applies every entry whatever its verdict. That is the
    in-sample chain Ethan's rule warns about, kept available so the two
    descents can be printed side by side: with it the in-sample column falls
    at every entry on pure noise and the held-out column does not, which is
    the whole reason the frozen evaluation exists.
    """
    values = np.asarray(values)
    groups = _groups(split_or_train, test)
    out: Dict[object, Dict[str, object]] = {}
    for label, bank in groups.items():
        current = values.copy()
        rows: List[Dict[str, object]] = []
        accepted: List[str] = []
        refused: List[str] = []
        start_in_sample = _rms(values[_finite(values, bank["train"])])
        start_held_out = _rms(values[_finite(values, bank["test"])])
        for entry in entries:
            got = frozen(entry, current, split_or_train, test, label)[label]
            null = null_holdout(entry, current, split_or_train, test, label,
                                draws=draws, seed=seed, scramble=scramble,
                                correlation_length=correlation_length)[label]
            decision = verdict(got, null, sigma)
            bar = chance_level(null, sigma)
            row = {
                "entry": entry.name,
                "position": entry.position,
                "kind": entry.kind,
                "count": got["count"],
                "in_sample_before": got["in_sample_before"],
                "in_sample": got["in_sample"],
                "held_out_before": got["held_out_before"],
                "held_out": got["held_out"],
                "held_out_se": got["held_out_se"],
                "descent": got["descent"],
                "descent_se": got["descent_se"],
                "ratio": got["ratio"],
                "null_held_out": null["held_out_median"],
                "null_descent": null["descent_median"],
                "null_spread": null["descent_spread"],
                "null_low": got["held_out_before"] - bar,
                "null_in_sample": null["in_sample_median"],
                "null_draws": null["draws"],
                "verdict": decision,
            }
            rows.append(row)
            if decision == "accepted" or not enforce:
                (accepted if decision == "accepted" else refused).append(
                    entry.name)
                current[got["train_index"]] = got["residual_train"]
                current[got["test_index"]] = got["residual_test"]
            else:
                refused.append(entry.name)
        out[label] = {
            "rows": rows,
            "accepted": accepted,
            "refused": refused,
            "start_in_sample": start_in_sample,
            "start_held_out": start_held_out,
            "final_in_sample": _rms(current[_finite(current, bank["train"])]),
            "final_held_out": _rms(current[_finite(current, bank["test"])]),
            "residual": current,
            "train_members": int(len(bank["train"])),
            "test_members": int(len(bank["test"])),
            "correlation_length": float(correlation_length),
            "sigma": float(sigma),
        }
    return out


def format_table(rows: Sequence[Dict[str, object]], unit: str = "",
                 width: int = 44) -> str:
    """The descent table as text: entry, parameter count, in-sample,
    held-out with its block-jackknife error, the null's held-out (the
    median over draws), the chance level (the held-out an entry must get
    below), and the verdict. One line per entry, in the order offered."""
    lines = [f"  {'entry':<{width}} {'k':>4} {'in-sample':>11} "
             f"{'held-out':>11} {'+-':>8} {'null':>11} {'chance':>11} "
             f"{'verdict'}"]
    for row in rows:
        name = str(row["entry"])
        if len(name) > width:
            name = name[:width - 1] + "~"
        se = row.get("held_out_se")
        se_text = (f"{se:8.3g}" if se is not None and np.isfinite(se)
                   else f"{'-':>8}")
        lines.append(
            f"  {name:<{width}} {row['count']:>4d} {row['in_sample']:>11.5g} "
            f"{row['held_out']:>11.5g} {se_text} "
            f"{row['null_held_out']:>11.5g} {row['null_low']:>11.5g} "
            f"{row['verdict']}")
    if unit:
        lines.append(f"  (residual rms in {unit})")
    return "\n".join(lines)
