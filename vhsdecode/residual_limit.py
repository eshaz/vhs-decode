"""The limit of the residuals, as machinery every measurement can share.

`docs/RESIDUAL_LIMIT_DESIGN.md` is the design and the authority; this module
is its executable part. Five rules, and the three that need code are here:

  R1  a measurement is refined against its OWN residual, never against a
      quantity that merely resembles it;
  R2  the limit is taken in every measured dimension at once;
  R3  every PRODUCT of dimensions is a component in its own right and must
      be projected orthogonal to its own margins before it is applied -
      `orthogonal`;
  R4  the differential between residuals is measured every step, and
      off-diagonal mass is a relationship to model, not noise -
      `coupling`;
  R5  termination is a DECLARED noise floor, and it is measured out of
      sample, because a model fitted on the fields it is then judged on
      converges to nothing whatever it describes - `agreement`.

R5 is not a caution, it is a measurement. On this deck, fitted and judged on
the same seven fields, the limit drives response, sweep and product to
0.0001 / 0.0000 / 0.0002 dB. Fitted on other fields of the same head and
tape it stands at 0.0328 / 0.0079 / 0.0563 dB. Nothing about the sequence
changed between those two numbers except which samples were used to judge
it, so in-sample convergence carries no evidence at all.

The traversal `order` returns is read from `pipeline/stages.toml`, which is
the tree's single declaration of what each stage reads, writes, measures and
feeds forward. The measurement graph is not a second list: it is that
declaration, walked in the direction its edges already point.
"""

import numpy as np


def cell_medians(index, values, count):
    """Median and population of every cell, without a call per cell.

    One lexicographic sort puts the samples in cell order and, within each
    cell, in value order, after which the median is an index rather than a
    reduction. The spelling is load bearing: the same table built with a
    median call per cell cost 60 per cent of the decode rate, which is what
    kept the product component out of the runtime until it was written this
    way.
    """
    order = np.lexsort((values, index))
    idx = index[order]
    val = values[order]
    bounds = np.searchsorted(idx, np.arange(count + 1))
    lo, hi = bounds[:-1], bounds[1:]
    population = (hi - lo).astype(np.float64)
    if not len(val):
        return np.zeros(count), population
    middle = np.minimum((lo + hi) // 2, len(val) - 1)
    median = np.where(population > 0.0, val[middle], 0.0)
    even = (population > 0.0) & (population % 2 == 0)
    if np.any(even):
        lower = np.clip(middle - 1, 0, len(val) - 1)
        median = np.where(even, 0.5 * (median + val[lower]), median)
    return median, population


def orthogonal(cells, weight, passes=4):
    """RULE R3. A product table made orthogonal to both its margins.

    A cell table that still carries a margin puts that margin straight back
    into the component that owns it, the moment it is applied. The two then
    fight and the sequence stalls far above the noise floor. Measured:
    without this, updating the product term returns +0.0046 dB to the
    frequency residual - the size of what remains - and the sweep settles at
    0.0048 / 0.0006 / 0.0045 dB. With it the coupling matrix is diagonal by
    the first step and the limit reaches 0.0001 / 0.0000 / 0.0002 dB.

    Removing one margin disturbs the other, so the pair is swept until both
    are gone; four passes puts each below a millionth of the table's own rms.
    """
    out = np.where(weight > 0.0, cells, 0.0)
    for _ in range(passes):
        for axis in (1, 0):
            mass = weight.sum(axis=axis, keepdims=True)
            mean = np.where(
                mass > 0.0,
                (out * weight).sum(axis=axis, keepdims=True)
                / np.maximum(mass, 1e-30),
                0.0,
            )
            out = np.where(weight > 0.0, out - mean, 0.0)
    return out


def agreement(first, second, weight_first, weight_second, minimum):
    """RULE R5. How far a component says the same thing on evidence it has
    not seen, as a number between zero and one.

    The component is accumulated in two banks on alternating fields. What
    comes back is their population-weighted correlation, clipped: a
    component describing the path reads near one, a component describing the
    field it was measured from reads near zero, and scaling by it admits
    each on its own evidence with no constant chosen anywhere.

    Verified on synthetic banks: two banks of one truth read 0.98,
    independent banks read 0.005.
    """
    both = (weight_first >= minimum) & (weight_second >= minimum)
    if not np.any(both):
        return 0.0
    a = first[both]
    b = second[both]
    w = weight_first[both] + weight_second[both]
    scale = np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w))
    if not scale > 0.0:
        return 0.0
    return float(np.clip(np.average(a * b, weights=w) / scale, 0.0, 1.0))


def coupling(components, residuals, update):
    """RULE R4. How much correcting each component moves every other one.

    `components` names them, `residuals` returns the current residual of
    each as (values, weight), and `update(name)` applies that component's
    own residual to it and returns a callable that puts it back. What comes
    back is a square matrix of norm changes: entry [j][i] is how far
    updating component j moved component i's residual.

    A diagonal matrix means the components are separable and taking the
    limit one at a time is correct. Off-diagonal mass is the relationship
    Ethan's design calls the residual of the residuals, and it is either
    modelled or projected away - never ignored, because it is what stalls a
    sequence that would otherwise converge.
    """
    before = [norm(r) for r in residuals()]
    matrix = []
    for name in components:
        restore = update(name)
        after = [norm(r) for r in residuals()]
        matrix.append([a - b for a, b in zip(after, before)])
        restore()
    return before, matrix


def norm(residual):
    """Population-weighted rms of one component's residual."""
    values, weight = residual
    live = weight > 0
    if not np.any(live):
        return 0.0
    return float(np.sqrt(np.average(values[live] ** 2, weights=weight[live])))


def order(declared):
    """The measurement nodes of the declared pipeline, in traversal order.

    Feed-forward, in the direction the declaration's own edges point: a node
    is refined under the model its parents currently hold and never against
    a parent its own residual has already entered, which would be fitting
    rather than measuring. Stages that measure nothing are not nodes of this
    graph, however much correction they apply - a stage with no residual has
    nothing to take a limit of.

    Ties are broken by declared order, so the traversal is deterministic and
    two runs of the same decode refine in the same sequence.
    """
    stages = declared.get("node") or []
    nodes = []
    for position, stage in enumerate(stages):
        measures = stage.get("measures") or []
        if not measures:
            continue
        # a measurement edge names the node whose output is measured and
        # the name the consumer knows it by; the residual belongs to the
        # consumer's MODEL of that quantity, so the consumer is the node of
        # this graph and the source is its evidence
        sources = [edge.get("node") if isinstance(edge, dict) else str(edge)
                   for edge in measures]
        nodes.append({
            "name": stage.get("name", f"stage_{position}"),
            "position": position,
            "measures": sources,
            "reads": list(stage.get("reads") or []),
            "writes": list(stage.get("writes") or []),
            "feeds_forward": list(stage.get("feeds_forward") or []),
            # A node that consumes another's MODEL is its child as surely as
            # one that reads its output channel, and it will not say so in
            # `reads`: the model reaches it as a bundle on the decoder
            # rather than as a channel. Declared as "<node>.<member>", so
            # the producer is the part before the dot. Without this the
            # head-switch stage resolves as a root while plainly depending
            # on the amplitude measurement it takes its residual from.
            "consumes": sorted({
                str(entry).split(".", 1)[0]
                for entry in (stage.get("consumes_model") or [])
            }),
        })
    produced = {}
    for node in nodes:
        for written in node["writes"]:
            produced.setdefault(written, node["name"])
    known = {node["name"] for node in nodes}
    for node in nodes:
        parents = {
            produced[read] for read in node["reads"]
            if read in produced and produced[read] != node["name"]
        }
        parents |= {
            name for name in node["consumes"]
            if name in known and name != node["name"]
        }
        node["parents"] = sorted(parents)
    ready, placed, remaining = [], set(), list(nodes)
    while remaining:
        free = [n for n in remaining
                if all(p in placed for p in n["parents"])]
        if not free:
            # a cycle in the declaration is the declaration's bug, not this
            # traversal's: take the earliest declared and say so by leaving
            # the parents unmet rather than silently reordering the rest
            free = [min(remaining, key=lambda n: n["position"])]
        step = min(free, key=lambda n: n["position"])
        ready.append(step)
        placed.add(step["name"])
        remaining.remove(step)
    return ready
