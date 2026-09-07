"""One transform, all dimensions at once, the null space never computed.

Ethan: 'I can make a singla picture stage that transforms luma chroma all
up to the composite functions. There doesn't need to be sequencing, just
all the dimensions execute at once in a single transform.' And: 'replace
the noise with null space, which is a constant that we do not derive at
all.' And: 'Those edges, which are null space do not get computed and
therefore save computation.'
"""

import math

import numpy as np
import pytest

from vhsdecode.models import single_transform as st
from vhsdecode.models import tesseract


def _cube(axes, bins=256, seed=0):
    rng = np.random.default_rng(seed)
    shape = (2,) * len(axes) + (bins,)
    return tesseract.Cube(tuple(axes), rng.normal(size=shape),
                          np.abs(rng.normal(size=shape)), np.arange(bins))


def test_luma_and_chroma_are_one_axis_whose_parent_is_composite():
    """The relation that removes three stages and leaves one axis.

    A fold takes a pair to their mean and half their difference, so if the
    two vertices are luma and chroma then the parent IS composite video
    and the differential is the separation. Composite is not a third thing
    to compute; it is what the fold already returns.
    """
    axes = ("colour",)
    bins = 64
    luma = np.linspace(0.0, 1.0, bins)
    chroma = np.linspace(1.0, 0.0, bins)
    values = np.stack([luma, chroma])
    cube = tesseract.Cube(axes, values, np.ones_like(values), np.arange(bins))
    out = st.prune(cube, {(), ("colour",)})
    assert np.allclose(out[()]["value"], 0.5 * (luma + chroma))
    assert np.allclose(out[("colour",)]["value"], 0.5 * (luma - chroma))
    # and the pair reconstructs the two vertices exactly
    assert np.allclose(out[()]["value"] + out[("colour",)]["value"], luma)
    assert np.allclose(out[()]["value"] - out[("colour",)]["value"], chroma)


def test_the_pruned_fold_agrees_with_the_full_one_on_what_it_keeps():
    axes = ("a", "b", "c", "d")
    cube = _cube(axes)
    full = tesseract.walsh(cube)
    kept = {(), ("a",), ("b",), ("a", "b")}
    pruned = st.prune(cube, kept)
    assert set(pruned) == kept
    for subset in kept:
        assert np.allclose(pruned[subset]["value"], full[subset]["value"],
                           atol=1e-12)


def test_the_null_contrasts_are_absent_rather_than_zero():
    """Never computed, which is the difference from computing and discarding."""
    cube = _cube(("a", "b", "c"))
    pruned = st.prune(cube, {(), ("a",)})
    assert set(pruned) == {(), ("a",)}
    assert ("b",) not in pruned
    assert ("a", "b", "c") not in pruned


def test_the_null_axes_are_folded_first_because_the_cost_is_at_the_top():
    """The ordering that makes the pruning worth anything.

    The first fold acts on the whole cube and every later one on something
    half the size, so a branch pruned at the bottom saves almost nothing.
    An axis appearing in no kept contrast loses its whole differential half
    at once and must go first.
    """
    axes = ("wanted", "unwanted", "also_wanted")
    kept = {(), ("wanted",), ("also_wanted",)}
    order = st.fold_order(axes, kept)
    assert order[0] == "unwanted"
    assert set(order) == set(axes)


def test_pruning_saves_more_as_the_dimension_rises():
    """Which is what makes going up in dimensions affordable at all."""
    import time

    def cost(n, bins):
        axes = tuple("abcdefg"[:n])
        cube = _cube(axes, bins)
        kept = {(), (axes[0],), (axes[1],), (axes[0], axes[1])}

        def timed(fn):
            # the fastest of several repeats, because a concurrent decode on
            # the same box inflates a mean and the question is the fold's
            # own cost, not the box's load at the moment
            fn()
            best = float("inf")
            for _ in range(7):
                start = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - start)
            return best

        return (timed(lambda: tesseract.walsh(cube))
                / max(timed(lambda: st.prune(cube, kept)), 1e-9))

    small = cost(4, 512)
    large = cost(6, 512)
    assert large > small
    assert large > 1.5


def test_the_declaration_names_a_mechanism_for_every_contrast_it_keeps():
    """The null space is a declaration from the physics, not a threshold.

    A contrast absent from the kept set is null because no mechanism makes
    it, which is why it can be skipped rather than computed and rejected.
    """
    for kind in ("picture", "rf"):
        out = st.declared(kind)
        assert out["computed"] < out["total"]
        assert out["kept"] | out["null"] == set(
            tuple(sorted(s, key=out["axes"].index))
            for size in range(len(out["axes"]) + 1)
            for s in st._combinations(out["axes"], size))
        for axis, why in out["meanings"].items():
            assert len(why) > 20, f"{axis} has no stated meaning"
        # every kept contrast is built only from declared axes
        for subset in out["kept"]:
            assert set(subset) <= set(out["axes"])


def test_the_picture_and_the_radio_frequency_are_the_same_construction():
    picture, rf = st.declared("picture"), st.declared("rf")
    assert len(picture["axes"]) == len(rf["axes"])
    assert () in picture["kept"] and () in rf["kept"]
    # each has a band-like axis whose parent is the combined signal
    assert "colour" in picture["axes"]
    assert "band" in rf["axes"]
    assert "composite" in picture["meanings"]["colour"]
    assert "recorded signal" in rf["meanings"]["band"]


def test_one_half_of_a_fold_can_be_taken_without_the_other():
    """Computing the half nobody wants is exactly the waste being removed."""
    cube = _cube(("a", "b"), bins=32)
    both = tesseract.fold(cube, "a")
    parent = st.fold_half(cube, "a", differential=False)
    differential = st.fold_half(cube, "a", differential=True)
    assert np.allclose(parent.values, both["parent"].values)
    assert np.allclose(differential.values, both["differential"].values)
    assert parent.axes == ("b",)


# ---------------------------------------------------------------------------
# The transform that takes a field: the live axes, both null treatments, the
# causal split, the floors, the latch, and the schedule for the workers
# ---------------------------------------------------------------------------

def _planted(kind="picture", bins=48, seed=1):
    """A transform with a known departure planted on named contrasts."""
    rng = np.random.default_rng(seed)
    axes = st.LIVE_AXES[kind]
    f = np.linspace(0.2e6, 4.0e6, bins)
    t = st.Transform(kind, axes, {"response": bins, "levels": 4},
                     sample_rate_hz=4 * 3579545.0, bins_hz={"response": f})
    return t, rng, f


def test_prune_keys_are_in_cube_axis_order_for_both_transforms():
    """The defect: fold order put ("head","band") where ("band","head")
    was declared, and a lookup by the declaration missed it."""
    for kind in ("picture", "rf"):
        out = st.declared(kind)
        cube = _cube(out["axes"], bins=16)
        pruned = st.prune(cube, out["kept"])
        assert set(pruned) == out["kept"]


def test_unreached_axes_are_declared_and_dropped_by_live_kept():
    picture = st.declared("picture")
    assert "field" in picture["unreached"] and "field" not in picture["live"]
    rf = st.declared("rf")
    assert "tap" in rf["unreached"] and "tap" not in rf["live"]
    assert ("tap",) not in rf["live_kept"] and ("band", "tap") not in rf["live_kept"]
    assert ("band", "head") in rf["live_kept"]
    with pytest.raises(ValueError):
        st.prune(_cube(rf["live"], bins=8), rf["kept"])


def test_unfold_over_the_admitted_contrasts_equals_reconstruct():
    axes = ("a", "b", "c")
    cube = _cube(axes, bins=32)
    full = tesseract.walsh(cube)
    keep = [(), ("a",), ("b",), ("a", "b")]
    ours = st.unfold_kept(axes, {s: full[s] for s in keep})
    theirs = tesseract.reconstruct(cube, full, keep)
    assert np.allclose(ours, theirs, atol=1e-12)
    # and the full set reproduces the vertices themselves
    assert np.allclose(st.unfold_kept(axes, full), cube.values, atol=1e-12)


def test_fill_then_contrasts_agrees_with_walsh_on_the_live_axes():
    t, rng, _ = _planted()
    truth = rng.normal(size=(2, 2, 2, t.bins)) + 1j * rng.normal(size=(2, 2, 2, t.bins))
    for v in np.ndindex(2, 2, 2):
        for _ in range(6):
            noise = 0.01 * (rng.normal(size=t.bins) + 1j * rng.normal(size=t.bins))
            x = truth[v] + noise
            vertex = dict(zip(t.axes, v))
            t.fill(vertex, "response", x[:48])
            t.fill(vertex, "levels", x[48:])
    assert t.ready(6)
    cube = t.cube()
    full = tesseract.walsh(cube)
    contrasts, amounts = t.contrasts("substitute")
    assert set(contrasts) == t.kept
    for subset in t.kept:
        assert np.allclose(contrasts[subset]["value"], full[subset]["value"])
        assert np.all(amounts[subset] == 1.0)


def test_a_face_duplicated_channel_has_zero_differential_and_is_refused():
    """Chroma has no fall or rise: both polarity faces carry the same value,
    so the polarity contrast of that channel is exactly zero with a
    nonzero variance and the verdict refuses it."""
    t, rng, _ = _planted()
    for v in np.ndindex(2, 2, 2):
        for k in range(4):
            colour, head, polarity = v
            base = rng.normal(size=4) if polarity == 0 else None
            # the same value on both polarity faces for the level channel
            vertex = dict(zip(t.axes, v))
            same = np.full(4, 1.0 + colour + 3 * head) + 0.01 * k
            t.fill(vertex, "levels", same)
            t.fill(vertex, "response", rng.normal(size=48))
    full = tesseract.walsh(t.cube())
    sl = t.channels["levels"]
    assert np.allclose(full[("polarity",)]["value"][sl], 0.0)
    verdict = tesseract.identified({("polarity",): {
        "value": full[("polarity",)]["value"][sl],
        "variance": full[("polarity",)]["variance"][sl], "order": 1}})
    assert not verdict[("polarity",)]["identified"]


def test_a_planted_head_departure_appears_at_the_head_contrast_only():
    t, rng, _ = _planted()
    shape = np.exp(1j * np.linspace(0, 1, 48)) * np.linspace(0.1, 0.3, 48)
    for v in np.ndindex(2, 2, 2):
        sign = 1.0 if v[1] == 0 else -1.0
        for _ in range(3):
            t.fill(dict(zip(t.axes, v)), "response", sign * shape)
            t.fill(dict(zip(t.axes, v)), "levels", np.zeros(4))
    contrasts, _ = t.contrasts("substitute")
    sl = t.channels["response"]
    assert np.allclose(contrasts[("head",)]["value"][sl], shape)
    for subset in contrasts:
        if subset != ("head",):
            assert np.allclose(contrasts[subset]["value"][sl], 0.0, atol=1e-12)


def test_the_two_null_treatments_on_noise_and_on_a_planted_mechanism():
    """Pure noise on the null contrasts: `remove` admits almost nothing of
    them and its floors agree. A mechanism planted on a declared-null
    contrast: `remove` admits it and lowers the remainder, `substitute`
    leaves it in the picture."""
    def build(plant, seed):
        t, rng, _ = _planted(seed=seed)
        mechanism = 0.5 * np.exp(1j * np.linspace(0, 2, 48))
        for v in np.ndindex(2, 2, 2):
            sign = st.signs(t.axes, v, ("head", "polarity"))
            for _ in range(8):
                noise = 0.05 * (rng.normal(size=48) + 1j * rng.normal(size=48))
                x = noise + (sign * mechanism if plant else 0.0)
                t.fill(dict(zip(t.axes, v)), "response", x)
                t.fill(dict(zip(t.axes, v)), "levels", np.zeros(4))
        return t, mechanism

    quiet, _ = build(False, 3)
    contrasts, amounts = quiet.contrasts("remove")
    null = [s for s in contrasts if s not in quiet.kept]
    assert null, "the picture declaration leaves null contrasts to judge"
    sl = quiet.channels["response"]
    assert max(float(np.mean(amounts[s][sl])) for s in null) < 0.5
    floors = quiet.floors()["response"]
    assert 0.25 < floors["null_pool"] / floors["instrument"] < 4.0
    assert 0.25 < floors["banks"] / floors["instrument"] < 4.0

    planted, mechanism = build(True, 4)
    contrasts, amounts = planted.contrasts("remove")
    assert float(np.mean(amounts[("head", "polarity")][sl])) > 0.9
    vertex = dict(zip(planted.axes, (0, 0, 0)))
    reading = mechanism + 0.0
    removed = planted.remainder(vertex, "response", reading, "remove")
    substituted = planted.remainder(vertex, "response", reading, "substitute")
    assert removed["after"] < 0.3 * substituted["after"]


def test_the_hypercube_reads_the_noise_where_the_data_must_not_exist():
    rng = np.random.default_rng(5)
    f = np.linspace(0.0, 20e6, 2001)
    noise = 2.0 * rng.chisquare(2, size=f.size) / 2.0     # mean 2.0
    signal = 100.0 * ((f > 1e6) & (f < 9e6))
    box = st.hypercube(noise + signal, f, [(1e6, 9e6)])
    assert abs(box["density"] - 2.0) < 0.15
    assert box["bins_inside"] + box["bins_outside"] == f.size
    with pytest.raises(ValueError):
        st.hypercube(noise, f, [(0.0, 20e6)])


def test_the_causal_split_recovers_a_planted_delay_on_the_right_grid():
    rate = 4 * 3579545.0
    f = np.linspace(0.2e6, 4.0e6, 200)
    delay = 40e-9
    value = np.zeros(f.size) + 1j * (-2 * np.pi * f * delay)
    out = st.causal_split(value, f, rate)
    assert abs(out["delay_s"] - delay) < 0.02 * delay
    assert out["allpass_rms"] < 1e-9
    # a one-pole minimum-phase shape carries no delay and little excess
    pole = 1.0 / (1.0 + 1j * f / 1.5e6)
    full_f = np.fft.rfftfreq(4096, 1.0 / rate)
    full = np.log(1.0 / (1.0 + 1j * full_f / 1.5e6))
    index = np.rint(f / (full_f[1] - full_f[0])).astype(int)
    out = st.causal_split(full[index], full_f[index], rate)
    assert out["minimum_phase_share"] > 0.8


def test_transients_are_rejected_by_witness_and_by_the_running_median():
    t, rng, _ = _planted()
    vertex = dict(zip(t.axes, (0, 0, 0)))
    for _ in range(8):
        assert t.fill(vertex, "response", rng.normal(size=48) * 0.01)
    assert not t.fill(vertex, "response", rng.normal(size=48) * 0.01, transient=True)
    assert t.rejected_by_witness == 1
    assert not t.fill(vertex, "response", 100.0 * np.ones(48))
    assert t.rejected == 2
    assert not t.fill(vertex, "response", np.full(48, np.nan))


def test_the_latch_fires_once_and_the_snapshot_never_changes():
    t, rng, _ = _planted()
    for v in np.ndindex(2, 2, 2):
        for _ in range(4):
            t.fill(dict(zip(t.axes, v)), "response", rng.normal(size=48))
            t.fill(dict(zip(t.axes, v)), "levels", rng.normal(size=4))
    assert t.latch("substitute", 4) is None or t.latched is not None
    first = t.latch("substitute", 4)
    assert first is not None and first.version == 1
    before = t.departure(dict(zip(t.axes, (1, 0, 1)))).copy()
    for v in np.ndindex(2, 2, 2):
        t.fill(dict(zip(t.axes, v)), "response", 10.0 * rng.normal(size=48))
    assert t.latch("remove", 4) is first
    assert np.array_equal(t.departure(dict(zip(t.axes, (1, 0, 1)))), before)
    report = t.report()
    assert report["latched"]["version"] == 1
    assert "head" in report["contrasts"] and "mean" in report["contrasts"]


def test_the_head_schedule_labels_by_absolute_sample_and_the_majority_rule():
    period = 667334.0
    schedule = st.HeadSchedule(anchor=1000.0, period=period, parity=True, version=1)
    assert schedule.head_at(1000.0) is True
    assert schedule.head_at(1000.0 + period) is False
    assert schedule.head_at(1000.0 + 2 * period) is True
    assert schedule.head_at(999.0) is False
    # a block straddling the switch takes the side with more samples
    assert schedule.majority(1000.0 + period - 100.0, 1000.0) is False
    assert schedule.majority(1000.0 + period - 900.0, 1000.0) is True


def test_the_published_snapshot_chooses_a_table_per_block_and_refuses_a_wrong_grid():
    n = 64
    common = np.full(n, 2.0 + 0j)
    per_head = {False: np.full(n, 1.5 + 0j), True: np.full(n, 0.5 + 0j)}
    schedule = st.HeadSchedule(anchor=0.0, period=1000.0, parity=False, version=1)
    published = st.publish(3, common, per_head, schedule)
    assert published.table(10.0, n)[0] == 3.0
    assert published.table(1010.0, n)[0] == 1.0
    assert published.table(10.0, n + 1) is None
    alone = st.publish(1, common, None, None)
    assert np.array_equal(alone.table(None, n), common)
    with pytest.raises(AttributeError):
        published.version = 4


def test_the_wave_under_the_convergence_admits_what_reproduces_and_discards_the_rest():
    """Ethan: 'a wave underneath the convergence that itself can be
    differentialed as random noise and subtracted out.' A planted eight-
    field alternation over sixty-four fields of one head is admitted by
    the second half of the run reproducing the first, and reproduced at
    each field; pure noise admits nothing."""
    rng = np.random.default_rng(11)
    bins = 16
    shape = 0.3 * np.exp(1j * np.linspace(0, 1.5, bins))

    def series(plant):
        wave = st.Wave(bins, field_rate_hz=60.0 / 1.001)
        for ordinal in range(64):
            field_number = 2 * ordinal          # one head's fields
            sign = 1.0 if ((ordinal >> 3) & 1) == 0 else -1.0
            noise = 0.02 * (rng.normal(size=bins) + 1j * rng.normal(size=bins))
            wave.record(field_number, noise + (sign * shape if plant else 0.0),
                        np.full(bins, 0.02 ** 2 * 2))
        return wave

    planted = series(True)
    folded = planted.fold()
    assert folded is not None and folded["fields_used"] == 64
    assert "field bit 3" in folded["admitted"]
    assert folded["amounts"][("field bit 3",)] > 0.9
    # the top bit only the whole run holds is not judged
    assert folded["amounts"][("field bit 5",)] == 0.0
    value = planted.value_at(2 * 5, folded)
    assert np.allclose(value, shape, atol=0.1)
    value = planted.value_at(2 * 12, folded)
    assert np.allclose(value, -shape, atol=0.1)

    quiet = series(False)
    folded = quiet.fold()
    assert folded is not None
    assert max(folded["amounts"][s] for s in folded["amounts"] if s) < 0.5


def test_the_transform_records_a_wave_only_after_the_latch():
    t, rng, _ = _planted()
    vertex = dict(zip(t.axes, (0, 1, 0)))
    assert t.record_remainder(vertex, 3, np.zeros(t.bins), 60.0) is None
    for v in np.ndindex(2, 2, 2):
        for _ in range(4):
            t.fill(dict(zip(t.axes, v)), "response", rng.normal(size=48))
            t.fill(dict(zip(t.axes, v)), "levels", rng.normal(size=4))
    assert t.latch("substitute", 4) is not None
    for k in range(8):
        assert t.record_remainder(vertex, 2 * k + 1, rng.normal(size=t.bins), 60.0) == k
    assert t.wave(vertex) is not None
    assert "waves" in t.report()


def test_a_weight_is_worth_not_a_count_in_the_variance_of_the_mean():
    """One fill of weight 245 is one field and has no variance; two fills of
    any equal weight have the variance two points have."""
    t, rng, _ = _planted()
    vertex = dict(zip(t.axes, (0, 0, 0)))
    t.fill(vertex, "response", np.ones(48), weight=245.0)
    for v in np.ndindex(2, 2, 2):
        if v == (0, 0, 0):
            continue
        t.fill(dict(zip(t.axes, v)), "response", np.ones(48))
    cube = t.cube()
    assert np.all(np.isinf(cube.variance[(0, 0, 0)][t.channels["response"]]))
    t.fill(vertex, "response", np.zeros(48), weight=245.0)
    cube = t.cube()
    # two points at 1 and 0: sample variance 1/2, the mean's variance 1/4
    assert np.allclose(cube.variance[(0, 0, 0)][t.channels["response"]], 0.25)


def test_the_rings_are_the_grades_of_the_fold_and_the_causal_axis_is_not_among_them():
    """Ethan: 'the concentric rings of dimensions, like a multi fold sphere
    that has a causality dimension fixed.'"""
    out = st.declared("picture")
    rings = out["rings"]
    assert [rings[k]["count"] for k in sorted(rings)] == [1, 4, 6, 4, 1]
    assert rings[0]["kept"] == ["mean"]
    assert "colour:head" in rings[2]["kept"]
    assert all("frequency" not in c for k in rings for c in rings[k]["contrasts"])
    assert "causal" in out["fixed_axis"]
    t, rng, _ = _planted()
    live = t.report()["rings"]
    assert [live[k]["count"] for k in sorted(live)] == [1, 3, 3, 1]


def test_a_measurement_costs_one_array_at_its_own_depth():
    """Ethan: 'we can compress down measurements to only need to apply to
    their dimensional depth. I think the actual order doesn't matter, just
    the correct number of transformation in all dimensions.'"""
    out = st.declared("picture")
    plan = st.depth_plan(out["axes"], out["kept"])
    assert plan["arrays"] == len(out["kept"])
    assert plan["applications"] == len(out["kept"])
    assert plan["naive_applications"] == 16 * len(out["kept"])
    assert set(plan["folds_per_axis"].values()) == {1}
    assert plan["depths"][0] == ["mean"]
    assert "colour:head" in plan["depths"][2]


def test_the_order_of_folding_does_not_change_the_contrasts():
    """The folds commute, which is why the stages needed no sequencing:
    only the count of folds per axis is fixed."""
    axes = ("a", "b", "c", "d")
    cube = _cube(axes, bins=32)
    reference = tesseract.walsh(cube)
    for order in (("d", "c", "b", "a"), ("b", "d", "a", "c")):
        kept = set(reference)
        got = st.prune(cube, kept)
        assert set(got) == set(reference)
        for subset in reference:
            assert np.allclose(got[subset]["value"], reference[subset]["value"],
                               atol=1e-12)
        # folding by hand in the named order reaches the same place
        part = cube
        for axis in order:
            part = st.fold_half(part, axis, False)
        assert np.allclose(part.values.reshape(-1),
                           reference[()]["value"], atol=1e-12)


def test_applying_at_depth_equals_summing_into_each_vertex():
    axes = ("a", "b", "c")
    cube = _cube(axes, bins=24)
    full = tesseract.walsh(cube)
    keep = [(), ("a",), ("b",), ("a", "b"), ("a", "b", "c")]
    amounts = {s: np.linspace(0.4, 0.9, 24) for s in keep}
    one = st.apply_at_depth(axes, {s: full[s] for s in keep}, amounts)
    other = st.unfold_kept(axes, {s: full[s] for s in keep}, None, amounts)
    assert np.allclose(one, other, atol=1e-12)
    # and with everything kept it is the cube itself
    assert np.allclose(st.apply_at_depth(axes, full), cube.values, atol=1e-12)


def test_a_wrong_specification_lands_entirely_on_the_grand_mean():
    """ATONALITY, AND IT IS WHY THE MODEL SURVIVES A WRONG SPECIFICATION.

    Ethan: 'atonal music is the same system that rejects the tonal model,
    it creates it's own.' A departure against a specification is TONAL: it
    needs a reference given in advance. A contrast between two vertices is
    ATONAL: it is a difference within the measured set and needs no
    reference at all. So any error COMMON to every vertex - a
    mis-specified pulse width, a mis-stated level, an edge convention -
    lands entirely on the grand mean and moves no contrast of grade one or
    above by anything.

    This is not decoration: an edge convention in `sync_geometry` narrows
    every specified pulse by one edge time, and this property is the reason
    that error could only ever reach the mean.
    """
    axes = ("colour", "head", "polarity")
    bins = 32
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(2, 2, 2, bins)) + 1j * rng.normal(size=(2, 2, 2, bins))
    variance = np.ones_like(truth, dtype=float)
    right = tesseract.walsh(tesseract.Cube(axes, truth, variance))

    # the specification is wrong by a shape common to every vertex
    error = 0.7 * np.exp(1j * np.linspace(0.0, 2.0, bins))
    wrong = tesseract.walsh(tesseract.Cube(axes, truth + error, variance))

    assert np.allclose(wrong[()]["value"] - right[()]["value"], error,
                       atol=1e-12)
    for subset, entry in right.items():
        if not subset:
            continue
        assert np.allclose(wrong[subset]["value"], entry["value"], atol=1e-12)


def test_the_two_null_treatments_are_the_tonal_and_the_atonal_reading():
    """The declared kept set is a hierarchy of admissible relations fixed
    before the data is seen, which is what a tonal system is; judging each
    residual by its own reproducibility declares nothing in advance and
    lets the measured set supply its own structure, which is what an atonal
    one is. Both are built, and the decode chooses between them."""
    picture = st.declared("picture")
    assert picture["kept"] and picture["null"]          # a declared hierarchy
    assert st.TREATMENTS == ("substitute", "remove")
    plan = st.depth_plan(picture["axes"], picture["kept"])
    assert plan["order_is_free"]
