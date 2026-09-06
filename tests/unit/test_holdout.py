"""The held-out harness: frozen parameters, a split that never pools heads
and is blocked in time, and the null that separates a real component from an
over-parameterised one.

Ethan: *"Report held-out residual with parameters frozen. In-sample residual
falls monotonically on pure noise - it can't distinguish a good model from an
over-parameterized one."*

Every number these tests hold the module to was measured on 2026-09-05,
and where a test's own material had to change the reason is written at the
change: the first build's tests were sized against a null whose bar could
lie below zero, and a correct chance level needs evidence to clear.
"""

import numpy as np
import pytest

from vhsdecode.models import holdout as ho

GRID = np.linspace(1e6, 7e6, 512)


def _log_shape(value):
    logged = np.log(np.abs(value)) + 1j * np.unwrap(np.angle(value))
    return logged - logged.mean()


def _echo(cycles):
    """An echo whose ripple runs `cycles` times over the grid, as a log
    shape: one spectral line along the member axis, the shape a
    spectrum-preserving surrogate can only rotate."""
    resolution = 1.0 / (GRID[-1] - GRID[0])
    return _log_shape(1.0 + 0.2 * np.exp(-2j * np.pi * GRID * resolution
                                         * cycles))


def _echo_entries(count):
    """Smooth, distinct signatures: echoes at delays spread over the band's
    ghost window, entered as log shapes."""
    return [ho.linear_entry(f"echo {k}", _echo(k + 1)) for k in range(count)]


def test_on_pure_noise_in_sample_falls_at_every_entry_and_held_out_does_not():
    """THE STATEMENT THE HARNESS EXISTS FOR. Every added parameter lowers the
    residual it was fitted to, on noise as on signal; the held-out residual
    is what says there was nothing there. The first build accepted 'echo 0'
    and 'echo 7' here, because its bar was a percentile of eight draws."""
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(GRID.size) + 1j * rng.standard_normal(GRID.size)
    split_ = ho.split(GRID.size, by="band", blocks=6, seed=0)
    entries = _echo_entries(8)

    # applied unconditionally - the in-sample chain
    chain = ho.ladder(entries, noise, split_, draws=8, seed=0,
                      enforce=False)["pooled"]
    in_sample = [row["in_sample"] for row in chain["rows"]]
    assert all(row["in_sample"] < row["in_sample_before"]
               for row in chain["rows"]), "in-sample must fall every entry"
    assert in_sample == sorted(in_sample, reverse=True)
    # and the held-out did not descend with it
    assert chain["final_held_out"] >= chain["start_held_out"] - 0.02 * \
        chain["start_held_out"]

    # the enforcing ladder refuses them all
    judged = ho.ladder(entries, noise, split_, draws=8, seed=0)["pooled"]
    assert judged["accepted"] == []
    assert len(judged["refused"]) == len(entries)
    assert judged["final_held_out"] == judged["start_held_out"]
    assert all(row["verdict"].startswith("refused") for row in judged["rows"])


def test_planted_structure_is_accepted_for_the_right_entry_and_not_a_wrong_one():
    """A planted component lowers the held-out residual for its own entry
    and for nothing orthogonal to it, and the held-out residual it leaves is
    the noise that was planted with it."""
    rng = np.random.default_rng(5)
    right = _echo(3)
    # a wrong entry from another family: a monotone loss, pure magnitude,
    # orthogonal to the echo's ripple
    wrong = _log_shape(np.exp(-GRID / 4e6))
    noise_per_part = 0.3
    data = 3.0 * right + noise_per_part * (
        rng.standard_normal(GRID.size) + 1j * rng.standard_normal(GRID.size))
    split_ = ho.split(GRID.size, by="band", blocks=6, seed=0)
    result = ho.ladder([ho.linear_entry("wrong loss", wrong),
                        ho.linear_entry("right echo", right)],
                       data, split_, draws=12, seed=0)["pooled"]
    verdicts = {row["entry"]: row["verdict"] for row in result["rows"]}
    assert verdicts["right echo"] == "accepted"
    assert verdicts["wrong loss"] != "accepted"
    rows = {row["entry"]: row for row in result["rows"]}
    echo = rows["right echo"]
    assert echo["held_out"] < echo["held_out_before"]
    # The first build asked for the held-out residual to fall below HALF its
    # before-value. The planted noise is 0.3 per part, a complex rms of
    # 0.424, and the before-value is 0.710: half of it is 0.355, below the
    # floor the test itself planted, which no frozen estimator can reach -
    # the module's own doctrine says the floor is the held-out residual.
    # The right entry lands AT that floor (measured 0.414).
    floor = noise_per_part * np.sqrt(2.0)
    assert echo["held_out"] == pytest.approx(floor, rel=0.1)
    assert echo["held_out"] < echo["null_low"]
    assert echo["descent"] > echo["null_descent"] + 3 * echo["null_spread"]


def test_the_split_never_pools_heads():
    """Fields alternate heads; every bank of every group holds one head, and
    a request to pool is refused rather than honoured."""
    for seed in range(4):
        for blocks in (2, 4, 6):
            split_ = ho.split(29, by="field", blocks=blocks, seed=seed)
            heads = np.arange(29) % ho.FIELDS_PER_HEAD_CYCLE
            assert not ho.pools_heads(split_, heads)
            assert set(split_.groups) == {0, 1}
            for bank in split_.groups.values():
                assert bank["train"].size and bank["test"].size
                assert not np.intersect1d(bank["train"], bank["test"]).size
    # explicit head labels are honoured
    heads = np.array([0, 0, 1, 1] * 6)
    split_ = ho.split(24, by="field", heads=heads, blocks=4)
    assert not ho.pools_heads(split_, heads)
    with pytest.raises(ValueError, match="never pool heads"):
        ho.split(24, by="field", heads=heads, within_head=False)
    with pytest.raises(ValueError, match="never pool heads"):
        ho.split(24, by="band", heads=heads)


def test_the_split_is_blocked_in_time_and_keeps_fields_whole():
    """Contiguous runs, as many as the blocks and no more; an interleaved
    split would have about as many runs as members. And every line of a
    field lands in the same bank.

    Blockedness is counted against the sequence a bank was drawn from - a
    head's own fields - because one head's fields are every other field
    and are never contiguous in the global index. The first build counted
    runs in the global index and read a correctly blocked bank of ten
    fields as ten runs; the same bank counted against its head's sequence
    is three."""
    split_ = ho.split(40, by="field", blocks=6, seed=0)
    for bank in split_.groups.values():
        assert ho.runs_of(bank["train"], bank["members"]) <= 3
        assert ho.runs_of(bank["test"], bank["members"]) <= 3
        # an interleaved split of 20 members per head would have 10 runs
        assert (ho.runs_of(bank["train"], bank["members"])
                + ho.runs_of(bank["test"], bank["members"])) <= 6
        # the global count is the wrong sequence for a head group
        assert ho.runs_of(bank["train"]) == bank["train"].size
    # and an interleaved bank reads as interleaved against the same sequence
    head = np.arange(0, 40, 2)
    assert ho.runs_of(head[::2], head) == head[::2].size
    with pytest.raises(ValueError, match="outside the sequence"):
        ho.runs_of([1, 3], head)
    # both banks span the record rather than one bank taking one end
    for bank in split_.groups.values():
        assert bank["train"].min() < bank["test"].max()
        assert bank["test"].min() < bank["train"].max()

    fields = np.repeat(np.arange(12), 30)
    split_ = ho.split(fields=fields, by="field", blocks=4, seed=1)
    for bank in split_.groups.values():
        train_fields = set(np.unique(fields[bank["train"]]))
        test_fields = set(np.unique(fields[bank["test"]]))
        assert not train_fields & test_fields
        for field_id in train_fields | test_fields:
            members = np.flatnonzero(fields == field_id)
            in_train = np.isin(members, bank["train"])
            assert in_train.all() or not in_train.any()

    band = ho.split(300, by="band", blocks=6, seed=2).groups["pooled"]
    assert ho.runs_of(band["train"]) == 3
    assert ho.runs_of(band["test"]) == 3
    assert ho.runs_of(band["train"], band["members"]) == 3


def test_a_parameter_per_field_lowers_in_sample_and_cannot_lower_held_out():
    """The over-parameterised model, refused by the rule. A free gain per
    field has no parameter for a field it has not seen, so it predicts
    nothing there and the descent is exactly zero.

    Twenty-four fields rather than the first build's twelve. With twelve -
    three training fields per head, twenty labels fitted on sixty members
    against noise of 0.3 - the line template's held-out descent stood 3.0
    and 1.9 spreads above the null on the two heads, which the chance level
    rightly calls unproven; with twenty-four it stands 4.6 and 9.6 above.
    The first build's bar lay below zero, so twelve fields never had to
    carry evidence."""
    rng = np.random.default_rng(7)
    fields = np.repeat(np.arange(24), 20)
    lines = np.tile(np.arange(20), 24)
    values = 1.0 + 0.5 * np.sin(lines / 3.0) + 0.3 * rng.standard_normal(
        fields.size)
    split_ = ho.split(fields=fields, by="field", blocks=6, seed=0)
    entries = [ho.constant_entry("level", values.size),
               ho.template_entry("line template", lines),
               ho.template_entry("per-field gain", fields)]
    for group, got in ho.ladder(entries, values, split_, draws=8,
                                seed=0).items():
        rows = {row["entry"]: row for row in got["rows"]}
        assert rows["level"]["verdict"] == "accepted"
        assert rows["line template"]["verdict"] == "accepted"
        control = rows["per-field gain"]
        assert control["in_sample"] < control["in_sample_before"]
        assert control["held_out"] == control["held_out_before"]
        assert control["descent"] == 0.0
        assert control["verdict"] == "refused: held-out did not fall"
        assert got["accepted"] == ["level", "line template"]


def test_an_unseen_template_label_predicts_nothing():
    """A template's claim is per label; a member whose label it never fitted
    is left alone. The first build predicted the mean of the fitted
    parameters there - a level in disguise - and its per-line control
    picked up the level a relaxation without an offset had left behind."""
    labels = np.arange(8)
    entry = ho.template_entry("per-member", labels)
    parameters = entry.fit(np.array([2.0, 4.0, 6.0, 8.0]), np.arange(4))
    assert np.array_equal(parameters["labels"], np.arange(4))
    assert np.allclose(entry.predict(np.arange(4), parameters),
                       [2.0, 4.0, 6.0, 8.0])
    assert np.array_equal(entry.predict(np.arange(4, 8), parameters),
                          np.zeros(4))


def test_parameters_are_frozen_on_the_held_out_bank():
    """The prediction on test uses exactly the parameters fitted on train."""
    seen = []

    def fit(values, index):
        return {"mean": float(np.mean(values)), "from": index.copy()}

    def predict(index, parameters):
        seen.append((index.copy(), parameters))
        return np.full(len(index), parameters["mean"])

    entry = ho.Entry(name="recording level", fit=fit, predict=predict,
                     size=40, parameters=lambda p: 1)
    values = np.arange(40, dtype=np.float64)
    split_ = ho.split(40, by="band", blocks=4, seed=0)
    got = ho.frozen(entry, values, split_)["pooled"]
    train, test = split_.train("pooled"), split_.test("pooled")
    applied_on_test = [p for index, p in seen if np.array_equal(index, test)]
    assert applied_on_test, "the entry was never applied on test"
    assert applied_on_test[0]["mean"] == pytest.approx(float(values[train].mean()))
    assert np.array_equal(applied_on_test[0]["from"], train)
    assert got["held_out"] == pytest.approx(
        np.sqrt(np.mean((values[test] - values[train].mean()) ** 2)))
    assert got["in_sample"] == pytest.approx(float(values[train].std()))
    assert np.isfinite(got["descent_se"])


def test_the_null_keeps_the_noise_correlation_and_not_the_signature():
    """Two things the surrogate must do, each the repair of a measured
    defect of the first build.

    ONE: it must not carry the entry's own shape. Phase randomisation that
    keeps a signature's whole spectrum along the member axis keeps an echo
    - one spectral line - as the same ripple rotated. On a planted echo
    that null descended by 0.159 against the true entry's 0.295, and the
    true entry was refused. Kept only to the noise's correlation length
    (one member here: white noise), the null descends by nothing.

    TWO: it must keep what sets the chance level. On noise correlated over
    fourteen members a smooth wrong echo projects on the noise far more
    than a white shape does: the phase null's spread is an order of
    magnitude above the permutation null's, and over forty seeds the
    permutation null accepted the wrong entry eight times where the phase
    null accepted it never."""
    rng = np.random.default_rng(5)
    right = _echo(3)
    data = 3.0 * right + 0.3 * (rng.standard_normal(GRID.size)
                                + 1j * rng.standard_normal(GRID.size))
    split_ = ho.split(GRID.size, by="band", blocks=6, seed=0)
    entry = ho.linear_entry("right echo", right)
    got = ho.frozen(entry, data, split_)["pooled"]
    white = ho.null_holdout(entry, data, split_, draws=12, seed=0,
                            correlation_length=1.0)["pooled"]
    whole = ho.null_holdout(entry, data, split_, draws=12, seed=0,
                            correlation_length=GRID.size)["pooled"]
    assert white["draws"] == 12 and whole["draws"] == 12
    assert abs(white["descent_median"]) < 0.05 * got["descent"]
    assert whole["descent_median"] > 0.3 * got["descent"]
    assert ho.verdict(got, white) == "accepted"
    assert ho.verdict(got, whole) != "accepted"

    for seed in range(6):
        local = np.random.default_rng(100 + seed)
        smooth = np.convolve(local.standard_normal(GRID.size)
                             + 1j * local.standard_normal(GRID.size),
                             np.ones(16) / 4.0, mode="same")
        length = ho.noise_correlation_length(smooth)["length"]
        assert 8.0 < length < 24.0
        wrong = ho.linear_entry("wrong echo", _echo(1 + seed % 4))
        result = ho.frozen(wrong, smooth, split_)["pooled"]
        phase = ho.null_holdout(wrong, smooth, split_, draws=12, seed=seed,
                                correlation_length=length)["pooled"]
        permute = ho.null_holdout(wrong, smooth, split_, draws=12,
                                  seed=seed, scramble="permute")["pooled"]
        assert phase["descent_spread"] > 3.0 * permute["descent_spread"]
        assert ho.verdict(result, phase) != "accepted"


def test_the_noise_correlation_length_is_read_from_the_data():
    """One member for white noise, the boxcar's width for boxcar-smoothed
    noise, and from a PAIR of measurements the structure they share
    cancels and only the noise's length remains."""
    rng = np.random.default_rng(21)
    white = rng.standard_normal(4000) + 1j * rng.standard_normal(4000)
    assert ho.noise_correlation_length(white)["length"] == pytest.approx(
        1.0, abs=0.15)
    smooth = np.convolve(white, np.ones(8) / np.sqrt(8.0), mode="same")
    assert ho.noise_correlation_length(smooth)["length"] == pytest.approx(
        8.0, rel=0.2)
    structure = 5.0 * np.sin(np.arange(4000) / 40.0)
    first = structure + white
    second = structure + (rng.standard_normal(4000)
                          + 1j * rng.standard_normal(4000))
    alone = ho.noise_correlation_length(first)
    paired = ho.noise_correlation_length(first, second)
    assert alone["length"] > 10.0          # the structure, read as noise
    assert paired["length"] == pytest.approx(1.0, abs=0.15)
    assert paired["paired"]
    with pytest.raises(ValueError, match="same members"):
        ho.noise_correlation_length(first, second[:-1])
    # the surrogate keeps that length and nothing beyond it
    shape = _echo(3)
    kept = ho.surrogate(shape, np.random.default_rng(0), "phase", 8.0)
    assert np.linalg.norm(kept) == pytest.approx(np.linalg.norm(shape))
    assert 4.0 < ho.noise_correlation_length(kept)["length"] < 16.0
    overlap = abs(np.vdot(shape, kept)) / np.linalg.norm(shape) ** 2
    assert overlap < 0.3
    # a real signature stays real
    real = ho.surrogate(np.exp(-np.arange(60) / 20.0),
                        np.random.default_rng(1), "phase", 1.0)
    assert np.isrealobj(real)


def test_the_half_split_judges_the_second_half_with_the_first_halfs_fit():
    """The time-blocked split the exports carry: a shape common to both
    halves reproduces and is accepted; a shape present in the first half
    only cannot be predicted from it and is refused. Entries enter once per
    half through `bin_of`, with one frozen weight for both."""
    rng = np.random.default_rng(31)
    bins = 400
    grid = np.linspace(1e6, 7e6, bins)
    resolution = 1.0 / (grid[-1] - grid[0])
    key = {"common echo": 1.0 + 0.2 * np.exp(-2j * np.pi * grid * resolution * 3),
           "first-half only": 1.0 + 0.2 * np.exp(-2j * np.pi * grid
                                                 * resolution * 6)}
    common = 3.0 * _log_shape(key["common echo"])
    passing = 3.0 * _log_shape(key["first-half only"])
    noise = [0.3 * (rng.standard_normal(bins) + 1j * rng.standard_normal(bins))
             for _ in range(2)]
    values = np.concatenate([common + passing + noise[0], common + noise[1]])
    split_ = ho.half_split(bins, blocks=6)
    bank = split_.groups["pooled"]
    assert np.array_equal(bank["train"], np.arange(bins))
    assert np.array_equal(bank["test"], bins + np.arange(bins))
    assert len(bank["test_blocks"]) == 6
    assert split_.by == "half"
    # the halves' difference is the noise AND whatever does not reproduce:
    # the planted first-half-only shape sits in it and lengthens what the
    # paired estimator reads (as the slow term between halves does on the
    # real exports, 51-74 bins against a resolution of 52), while the noise
    # alone reads as white. Both are right, and the ladder is judged
    # against the length the halves actually differ by.
    length = ho.noise_correlation_length(values[:bins], values[bins:])
    assert length["length"] > 3.0
    assert ho.noise_correlation_length(noise[0], noise[1])["length"] < 1.3
    entries = ho.entries_from_key(key, order=[(1, "common echo"),
                                              (0, "first-half only")],
                                  bin_of=np.tile(np.arange(bins), 2))
    assert all(entry.size == 2 * bins for entry in entries)
    result = ho.ladder(entries, values, split_, draws=12, seed=0,
                       correlation_length=length["length"])["pooled"]
    verdicts = {row["entry"]: row["verdict"] for row in result["rows"]}
    assert verdicts["common echo"] == "accepted"
    assert verdicts["first-half only"] == "refused: held-out did not fall"
    assert result["accepted"] == ["common echo"]
    # the frozen weight was the first half's and the second half's residual
    # lands at its own noise
    assert result["final_held_out"] == pytest.approx(0.3 * np.sqrt(2.0),
                                                     rel=0.1)


def test_the_half_split_null_is_one_shape_on_both_halves():
    """An entry declared over the bins is one shape repeated on both halves,
    and so must its null be. Made over the stacked members, the surrogate's
    first and second halves were independent random shapes (their
    correlation measured 0.003 on this planted echo), and a null that
    cannot reproduce cannot stand for a wrong shape that does. Declared
    through `member_of`, the two halves of every null draw are identical,
    for the key's entries and for the level alike."""
    bins = 400
    grid = np.linspace(1e6, 7e6, bins)
    resolution = 1.0 / (grid[-1] - grid[0])
    key = {"echo": 1.0 + 0.2 * np.exp(-2j * np.pi * grid * resolution * 3)}
    bin_of = np.tile(np.arange(bins), 2)
    floor = np.concatenate([np.full(bins, 0.5), np.full(bins, 2.0)])
    entries = ho.entries_from_key(key, order=[(0, "echo")], bin_of=bin_of,
                                  floor=floor)
    entries.append(ho.constant_entry("level", 2 * bins, complex_valued=True,
                                     floor=floor, member_of=bin_of))
    first, second = np.arange(bins), bins + np.arange(bins)
    for entry in entries:
        assert entry.size == 2 * bins
        for draw in range(4):
            null = entry.scramble(np.random.default_rng(draw), "phase", 8.0)
            unit = np.ones(1)
            shape_first = null.predict(first, unit) * np.sqrt(floor[first])
            shape_second = null.predict(second, unit) * np.sqrt(floor[second])
            assert np.allclose(shape_first, shape_second)
            # and it is a scrambled shape, not the entry's own
            own = entry.predict(first, unit) * np.sqrt(floor[first])
            overlap = abs(np.vdot(own, shape_first)) / (
                np.linalg.norm(own) * np.linalg.norm(shape_first))
            assert overlap < 0.5
    with pytest.raises(ValueError, match="position the basis does not have"):
        ho.linear_entry("bad", np.ones(bins), member_of=np.arange(bins + 1))


def test_no_false_acceptance_over_seeds_at_twenty_four_draws():
    """The rate behind `CHANCE_SIGMA`: eight echo entries on white complex
    noise, band split, twenty-four null draws - 0 of 600 accepted over 75
    seeds when measured (2026-09-05; 4 of 800 at eight draws). Fifteen of
    those seeds here."""
    entries = _echo_entries(8)
    split_ = ho.split(GRID.size, by="band", blocks=6, seed=0)
    accepted = 0
    for seed in range(15):
        rng = np.random.default_rng(1000 + seed)
        noise = (rng.standard_normal(GRID.size)
                 + 1j * rng.standard_normal(GRID.size))
        got = ho.ladder(entries, noise, split_, draws=24, seed=seed)["pooled"]
        accepted += len(got["accepted"])
    assert accepted == 0


def test_the_scan_entry_on_a_shared_grid_recovers_a_planted_relaxation():
    """Members sharing a time grid: the fit on the occupancy-weighted mean
    profile is the full least squares, and the frozen tail is judged on
    held-out fields."""
    rng = np.random.default_rng(13)
    t = np.arange(60) * 0.02
    lines = 240
    y = (np.tile(2.0 * np.exp(-t / 0.9), lines)
         + 0.4 * rng.standard_normal(60 * lines)
         + np.repeat(0.2 * rng.standard_normal(lines), 60))
    fields = np.repeat(np.arange(lines) // 20, 60)
    position = np.tile(np.arange(60), lines)
    split_ = ho.split(fields=fields, by="field", blocks=4, seed=0)
    relax = ho.scan_entry("relaxation",
                          lambda p, tau: np.exp(-t[np.asarray(p)] / tau),
                          np.geomspace(0.1, 5.0, 40), y.size,
                          grid_position=position)
    result = ho.ladder([ho.constant_entry("level", y.size), relax,
                        ho.template_entry("per-line level (control)",
                                          np.repeat(np.arange(lines), 60))],
                       y, split_, draws=8, seed=0)
    for got in result.values():
        rows = {row["entry"]: row for row in got["rows"]}
        assert rows["relaxation"]["verdict"] == "accepted"
        assert rows["per-line level (control)"]["verdict"].startswith("refused")
        fitted = ho.frozen(relax, y, split_)[next(iter(result))]["parameters"]
        assert fitted["parameter"] == pytest.approx(0.9, rel=0.25)


def test_entries_from_the_key_follow_the_chain_order():
    """Last applied, first removed: the key's entries come out in descending
    declared position, and each is subtractable as a log shape."""
    from vhsdecode.models import interference as inf

    grid = np.linspace(1e6, 7e6, 256)
    key = inf.signatures(grid)
    entries = ho.entries_from_key(key)
    positions = [entry.position for entry in entries]
    assert positions == sorted(positions, reverse=True)
    assert {entry.name for entry in entries} == set(key)
    assert all(entry.size == grid.size for entry in entries)
