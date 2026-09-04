"""The limit's law, tested where it is load bearing.

`docs/RESIDUAL_LIMIT_DESIGN.md` is the design. These are the properties the
rules stand on, not the numbers they were measured at.
"""
import numpy as np

from vhsdecode import residual_limit as rl


def test_cell_medians_matches_numpy_and_counts():
    rng = np.random.default_rng(0)
    index = rng.integers(0, 37, size=9000)
    values = rng.normal(size=9000)
    median, population = rl.cell_medians(index, values, 37)
    reference = np.array([np.median(values[index == k]) for k in range(37)])
    assert np.allclose(median, reference)
    assert np.array_equal(population, np.bincount(index, minlength=37))


def test_cell_medians_handles_empty_cells_and_no_samples():
    median, population = rl.cell_medians(np.array([], dtype=int),
                                         np.array([]), 5)
    assert median.shape == (5,) and not population.any()
    index = np.array([0, 0, 4, 4])
    median, population = rl.cell_medians(index, np.array([1.0, 3.0, 2.0, 8.0]), 5)
    assert median[1] == 0.0 and population[1] == 0.0
    assert median[0] == 2.0 and median[4] == 5.0


def test_orthogonal_removes_both_margins():
    """RULE R3. A product table that keeps a margin puts it back into the
    component that owns it, which is what stalls the sequence."""
    rng = np.random.default_rng(1)
    weight = rng.integers(1, 400, size=(23, 7)).astype(float)
    cells = (rng.normal(size=(23, 7))
             + rng.normal(size=(23, 1))      # a frequency margin
             + rng.normal(size=(1, 7)))      # a sweep margin
    out = rl.orthogonal(cells, weight)
    rows = (out * weight).sum(axis=1) / weight.sum(axis=1)
    cols = (out * weight).sum(axis=0) / weight.sum(axis=0)
    scale = np.sqrt(np.average(out ** 2, weights=weight))
    assert np.abs(rows).max() < 1e-4 * scale
    assert np.abs(cols).max() < 1e-4 * scale


def test_orthogonal_leaves_unpopulated_cells_alone():
    weight = np.ones((4, 3))
    weight[2] = 0.0
    out = rl.orthogonal(np.ones((4, 3)), weight)
    assert not out[2].any()


def test_agreement_separates_shared_structure_from_field_specific():
    """RULE R5. A component is admitted as far as it says the same thing on
    evidence it has not seen."""
    rng = np.random.default_rng(2)
    shape = (30, 8)
    weight = np.full((2,) + shape, 500.0)
    truth = rng.normal(size=shape)
    shared = np.stack([truth + 0.05 * rng.normal(size=shape),
                       truth + 0.05 * rng.normal(size=shape)])
    assert rl.agreement(shared[0], shared[1], weight[0], weight[1], 64) > 0.9
    independent = rng.normal(size=(2,) + shape)
    assert rl.agreement(independent[0], independent[1],
                        weight[0], weight[1], 64) < 0.2


def test_agreement_withholds_without_evidence():
    thin = np.full((3, 3), 4.0)
    assert rl.agreement(np.ones((3, 3)), np.ones((3, 3)), thin, thin, 64) == 0.0


def test_coupling_reports_a_diagonal_for_separable_components():
    """RULE R4. Off-diagonal mass is a relationship, not noise."""
    state = {"a": np.zeros(4), "b": np.zeros(3)}
    truth = {"a": np.array([1.0, -2.0, 0.5, 0.0]), "b": np.array([0.3, -0.3, 0.0])}
    weight = {"a": np.full(4, 100.0), "b": np.full(3, 100.0)}

    def residuals():
        return [(truth[k] - state[k], weight[k]) for k in ("a", "b")]

    def update(name):
        keep = state[name].copy()
        state[name] = state[name] + (truth[name] - state[name])
        return lambda: state.__setitem__(name, keep)

    before, matrix = rl.coupling(["a", "b"], residuals, update)
    assert before[0] > 0 and before[1] > 0
    assert matrix[0][0] < 0 and matrix[0][1] == 0.0
    assert matrix[1][1] < 0 and matrix[1][0] == 0.0


def test_order_is_feed_forward_and_deterministic():
    declared = {"node": [
        {"name": "late", "reads": ["mid_out"], "writes": ["late_out"],
         "measures": [{"node": "envelope", "as": "envelope"}]},
        {"name": "early", "reads": [], "writes": ["early_out"],
         "measures": [{"node": "envelope", "as": "envelope"}]},
        {"name": "mid", "reads": ["early_out"], "writes": ["mid_out"],
         "measures": [{"node": "envelope", "as": "envelope"}]},
        {"name": "no_measurement", "reads": [], "writes": ["x"]},
    ]}
    names = [n["name"] for n in rl.order(declared)]
    assert names == ["early", "mid", "late"]
    assert rl.order(declared) == rl.order(declared)


def test_order_follows_a_consumed_model_as_a_parent():
    """A node consuming another's MODEL is its child, and will not say so in
    `reads` - the model reaches it as a bundle, not as a channel."""
    declared = {"node": [
        {"name": "consumer", "reads": [], "writes": ["out"],
         "consumes_model": ["producer.block_model"],
         "measures": [{"node": "envelope", "as": "envelope"}]},
        {"name": "producer", "reads": [], "writes": ["model"],
         "measures": [{"node": "envelope", "as": "envelope"}]},
    ]}
    nodes = rl.order(declared)
    assert [n["name"] for n in nodes] == ["producer", "consumer"]
    assert nodes[1]["parents"] == ["producer"]


def test_order_ignores_a_consumed_model_that_is_not_a_node():
    declared = {"node": [
        {"name": "alone", "reads": [], "writes": ["out"],
         "consumes_model": ["something_undeclared.member"],
         "measures": [{"node": "envelope", "as": "envelope"}]},
    ]}
    assert rl.order(declared)[0]["parents"] == []


def test_order_reads_the_shipped_declaration():
    from vhsdecode import pipeline_graph

    nodes = rl.order(pipeline_graph.load())
    assert nodes, "the declaration should hold measurement nodes"
    placed = set()
    for node in nodes:
        assert all(parent in placed for parent in node["parents"])
        placed.add(node["name"])
