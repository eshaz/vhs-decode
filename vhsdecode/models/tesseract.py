"""THE TESSERACT: every measured state a vertex, every fold on every axis.

Ethan, in order:

  *"Oh I think the tesseract is asymmetric, and you never reach the full
  residual, since a hypercube has a finite number of edges."*

  *"Build out this n dimensional tesseract that folds in onto itself, it is
  a graph that is connected to all of its dimensional siblings and parents,
  and all other relationships, the differentials are determined by the
  color and the luma together for the video part, and the sync pulses, and
  eq pulses, etc."*

  *"The folding happens on all dimensions, previously we were only
  connecting it in sequence three ways, not the full dense graph."*

WHAT IT IS. Each axis of the measurement has two states: head A or B, a
falling or a rising edge, the first or the second half of the fields, one
tape or another. A vertex is one combination of states and carries the
measurement made there - a complex transfer over frequency for the luma,
the burst's complex gain for the chroma, both together for the video. With
n axes there are 2^n vertices, and the hypercube's edges join the vertices
that differ in ONE state: the differentials the earlier machinery took, one
axis at a time, in a fixed order.

THE FOLD ON ALL DIMENSIONS. Folding the cube along an axis pairs each
vertex with its sibling across that axis and replaces the pair by its mean
(the PARENT, one axis fewer) and its half-difference (the DIFFERENTIAL
along that axis). Folding the differential again along a second axis gives
the differential of the differential, and so on down to a single number.
Done on every axis, in every order, this is the Walsh-Hadamard transform:
2^n - 1 CONTRASTS, one for every non-empty subset of the axes, and the
order of the folds does not matter because they commute. The contrasts
are the complete, orthogonal basis of everything the vertices can differ
by, so after the full fold nothing is left over: the cube reproduces
exactly from its contrasts. That is the dense graph - every vertex is
connected to every other, and the difference between any two vertices is
the sum of the contrasts on the axes they differ in (`dense_graph`).

WHY THE SEQUENTIAL THREE-WAY WAS LESS. Taking the axes in sequence reads
the order-one contrasts (the edges) and the one chain of higher orders
that the sequence happens to pass through; it never reads the interaction
between two axes that the sequence did not place next to each other, and
it charges that interaction to whichever residual it lands in. The full
fold reads all of them at once, each with its own noise.

THE ASYMMETRY. The contrasts are not equal: a head contrast is measured
against one noise, a tape contrast against another, and their sizes above
that noise differ by orders of magnitude. The cube is a hyper-rectangle in
the units of its own floors, and `asymmetry` measures its sides.

THE FINITE NUMBER OF EDGES, AND THE RESIDUAL THAT IS NEVER REACHED. A
contrast is identified when its power stands above its noise; below that
it cannot be told from noise and is left in the residual. Every contrast's
noise is fixed by how many vertices the cube has - its edge count - so the
residual after the fold is NOT zero: it is the sum of every contrast the
cube is too small to identify, and the only way to lower it is more
vertices (more fields, more tapes, another axis), never more folding of
the same ones. `unreached` measures exactly that: the power left in the
unidentified contrasts, against the total, per order. When the top-order
contrast itself stands above noise, the structure has more dimensions
than the cube has axes, and the missing dimension is a measurement not
yet made.

MEASURED (2026-09-05, the sync exports, 944 bins 0.2-4 MHz, three tape
pairs). Every one of the 15 contrasts stands above its noise, the top-order
head x polarity x half x tape included (z = 260 on cd/home, 324 on cd/pnb,
23 on home/pnb): the structure has more dimensions than the cube has axes,
which is the finite-edge statement measured. The sides are unequal by 5.7
to 55 times (tape 18-43 floors, polarity 17-39, half 0.7-9.9, head 2.4-4.0).
Every axis's departure is mostly EXCESS phase, not minimum phase
(`hypercomplex.causality`, the contrasts placed back on the 4 f_sc grid
before the cepstrum - the first run fed the 0.2-4 MHz band as if it were
the whole axis and its delays were wrong): the polarity contrast carries
a +48 to +128 ns delay, the tape contrast -126 to +78 ns, with 0.4 to 1.1
rad of all-pass left after the delay is removed. And the minimum-phase
share is at or BELOW zero on almost every contrast: the magnitude implies
more phase than is measured, which is what a ZERO-PHASE (symmetric) shape
does. Two explanations were tried and BOTH FAILED: a truncated-sinc time
base (its error is 45 dB down and its phase uniformly spread, so it is not
zero-phase at all) and the instrument's own LTI/non-LTI split (the even,
LTI part is minimum phase on home but not on cd or pnb).

WHAT IT IS, AND WHAT IT IS NOT. Removing the delay first - a delay is not
a failure of minimum phase - leaves genuine ALL-PASS content, and it is
real: a minimum-phase response built from poles and zeros on the full
grid, then observed only on 0.2-4 MHz, fakes 0.05 rad of all-pass at the
depth of a one-pole and 0.21 rad at |log H| rms 1.93, DEEPER than any of
the tapes. The measured 0.87 to 2.07 rad is four to ten times that, so it
is the channel and not the method.

    tape           source        all-pass, rad
    countdown      television      2.07, 2.39
    home           home vcr        0.93, 0.98
    bars           composite       0.87, 0.92

A CLAIM MADE HERE AND NOW WITHDRAWN. It said the size follows the source
and named a broadcast reflection as the mechanism, because countdown is
the only tape recorded off air and carries twice the all-pass. The
correlation is real; the mechanism is wrong, and two measurements refuse
it. First, a propagation echo WEAKER than the direct path is minimum phase
and contributes no all-pass at all - measured, an echo of amplitude 0.3 to
0.95 gives 0.09 to 0.22 rad, which is the fill error and nothing more -
while an echo STRONGER than the direct path gives about 7 rad. The
measured values sit between the two regimes and match neither. Second,
home and the bars tape never passed through a transmitter at all, and they
carry 0.87 to 0.93 rad of it, correlated with each other at +0.55 despite
sharing no tape and no source. So most of this is in the recorder or the
decoder, and countdown's excess of 1.76 rad over that common part remains
unexplained. What is certain is that it is not ordinary multipath.

THE VIDEO PART IS LUMA AND CHROMA TOGETHER. A video vertex carries both
channels stacked; a differential along an axis moves both, and a contrast
that appears in one and not the other is a channel-specific mechanism
(the depth split: chroma reads 1.5 um into the coating, luma 0.2 um). The
chroma has no polarity axis - a burst is not an edge - so it lives on a
FACE of the luma's cube, entered against the luma's LTI part
`(fall + rise) / 2`, the only part an LTI consumer may fold.

`from_sync_exports` fills the luma cube from the sync-pulse instrument;
`chroma_burst_cube` fills the chroma from a decoded field set; the sync
and equalising pulses enter as further vertices when their responses are
measured on the same grid (the 1/T law: the sync pulse resolves 213 kHz,
the equalising train 1.7 kHz), which `sync_geometry.probes` sizes.
"""

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Subset = Tuple[str, ...]


class Cube:
    """2^n vertices, each a complex vector, each with a noise variance.

    `values` has shape (2,) * n + (bins,); `variance` the same, holding the
    noise variance of each vertex's entries (both real parts together).
    """

    def __init__(self, axes: Sequence[str], values, variance, bins=None,
                 channels: Optional[Dict[str, slice]] = None):
        self.axes = tuple(axes)
        self.values = np.asarray(values, dtype=np.complex128)
        self.variance = np.asarray(variance, dtype=np.float64)
        expected = (2,) * len(self.axes)
        if self.values.shape[:len(self.axes)] != expected:
            raise ValueError(f"values must have shape {expected} + (bins,), "
                             f"got {self.values.shape}")
        if self.variance.shape != self.values.shape:
            raise ValueError("variance must match values")
        self.bins = (np.asarray(bins) if bins is not None
                     else np.arange(self.values.shape[-1]))
        self.channels = dict(channels or {"all": slice(None)})

    @property
    def order(self) -> int:
        return len(self.axes)

    @property
    def vertices(self) -> List[Tuple[int, ...]]:
        return [tuple(int(b) for b in np.unravel_index(i, (2,) * self.order))
                for i in range(2 ** self.order)]


def fold(cube: Cube, axis: str) -> Dict[str, Cube]:
    """ONE FOLD: the cube pairs each vertex with its sibling across `axis`.

    Returns the PARENT (the mean of the pair: the same measurement with
    that axis collapsed) and the DIFFERENTIAL (half their difference: what
    that axis changes). Both have one axis fewer. The noise variance of
    each is a quarter of the sum of the pair's, because each is half of a
    sum or difference of two independent measurements.
    """
    k = cube.axes.index(axis)
    first = np.take(cube.values, 0, axis=k)
    second = np.take(cube.values, 1, axis=k)
    var = 0.25 * (np.take(cube.variance, 0, axis=k)
                  + np.take(cube.variance, 1, axis=k))
    remaining = tuple(a for a in cube.axes if a != axis)
    return {"parent": Cube(remaining, 0.5 * (first + second), var, cube.bins,
                           cube.channels),
            "differential": Cube(remaining, 0.5 * (first - second), var,
                                 cube.bins, cube.channels)}


def walsh(cube: Cube) -> Dict[Subset, Dict[str, np.ndarray]]:
    """THE FOLD ON ALL DIMENSIONS: every contrast of every order at once.

    Implemented as the fold applied along each axis in turn to BOTH
    products of the previous fold - which is the in-place fast
    Walsh-Hadamard transform, the cube folding onto itself n times. The
    result is indexed by the subset of axes a contrast differentiates on;
    the empty subset is the grand mean, the vertex the cube collapses to.
    """
    products: Dict[Subset, Cube] = {(): cube}
    for axis in cube.axes:
        folded: Dict[Subset, Cube] = {}
        for subset, part in products.items():
            halves = fold(part, axis)
            folded[subset] = halves["parent"]
            folded[subset + (axis,)] = halves["differential"]
        products = folded
    out = {}
    for subset, part in products.items():
        out[tuple(sorted(subset, key=cube.axes.index))] = {
            "value": part.values.reshape(-1), "variance": part.variance.reshape(-1),
            "order": len(subset)}
    return out


def reconstruct(cube: Cube, contrasts: Dict[Subset, Dict[str, np.ndarray]],
                keep: Optional[Sequence[Subset]] = None) -> np.ndarray:
    """The vertices rebuilt from their contrasts - exactly, when all are
    kept - or from a chosen subset, which is what a model that has only
    identified some of them predicts."""
    out = np.zeros_like(cube.values)
    for vertex in cube.vertices:
        total = np.zeros(cube.values.shape[-1], dtype=np.complex128)
        for subset, entry in contrasts.items():
            if keep is not None and subset not in keep:
                continue
            sign = 1.0
            for axis in subset:
                sign *= -1.0 if vertex[cube.axes.index(axis)] else 1.0
            total = total + sign * entry["value"]
        out[vertex] = total
    return out


def identified(contrasts: Dict[Subset, Dict[str, np.ndarray]],
               sigma: float = 3.0) -> Dict[Subset, Dict[str, object]]:
    """Which contrasts stand above their own noise, and by how much.

    The power of a contrast is the mean of |value|^2 over its bins; its
    noise is the mean variance. Noise alone gives a ratio of one with a
    standard error of root(2 / bins) on a complex Gaussian, so a contrast
    is identified when its ratio clears one by `sigma` such errors.
    """
    out = {}
    for subset, entry in contrasts.items():
        value, var = entry["value"], np.maximum(entry["variance"], 1e-300)
        n = value.size
        ratio = float(np.mean(np.abs(value) ** 2 / var))
        error = float(np.sqrt(2.0 / max(n, 1)))
        excess = max(ratio - 1.0, 0.0)
        # the structured power is read through the same per-bin weighting
        # as the verdict: (ratio - 1) times the noise. A plain difference
        # of means let a few high-variance bins swallow it - the per-field
        # cube reported 0.000 for contrasts identified at z = 122.
        noise_power = float(np.mean(var))
        out[subset] = {
            "order": entry["order"], "ratio": ratio, "error": error,
            "z": (ratio - 1.0) / error,
            "identified": bool(ratio - 1.0 > sigma * error),
            "structured_power": float(excess * noise_power),
            "noise_power": noise_power,
            "excess_over_floor": excess,
        }
    return out


def budget(cube: Cube) -> Dict[str, object]:
    """THE CONSTANT THE FOLD CONSERVES: the total representable change.

    Ethan: *"If you continue the hilbert transform down until a constant,
    you should get a constant, which describes the total amount of
    representable change in your dataset."* That is exactly what happens,
    and it is Parseval for this transform.

    Every vertex is a signed sum of the contrasts, so

        sum over vertices |x_v|^2  =  2^n  sum over subsets |c_S|^2

    and the fold moves energy between contrasts without creating or
    destroying any. Measured on a random cube of four axes and 64 bins:
    2055.420320 over the vertices against 128.463770 x 2^4 = 2055.420320
    over the contrasts, a ratio of 1.000000000000, and the same total to
    nine decimals for every order the axes are folded in.

    AND THE HILBERT HALF CONSERVES IT TOO, EXCEPT THE LEVEL. The transform's
    multiplier has unit modulus everywhere except at zero frequency, where
    `sgn(0) = 0` kills it, so a signal and its transform carry the same
    energy less exactly the DC term: measured 262.533014 against 262.503939,
    a difference of 0.029075, which is the DC energy to six decimals. A
    constant level is not change, so what the operator conserves IS the
    representable change - Ethan's sentence, arrived at from the arithmetic.

    WHAT IT IS FOR. The total is a BUDGET fixed before any fitting, so every
    contrast's share is a fraction of a known constant and "how much is
    left" is a fraction rather than an open quantity. The same invariant
    appears in `information_extrapolation.ellipsoid` as the trace of the
    Gram, which is N whatever the components do: the total is conserved and
    only the SHAPE carries information, which is why the asymmetry is the
    measurement.

    The level (the empty subset) is reported separately from the change,
    because a decode's absolute gain is not something the fold is measuring.
    """
    contrasts = walsh(cube)
    order_count = cube.order
    scale = float(2 ** order_count)
    per_subset = {s: float(np.sum(np.abs(e["value"]) ** 2) * scale)
                  for s, e in contrasts.items()}
    level = per_subset.get((), 0.0)
    total = float(np.sum(np.abs(cube.values) ** 2))
    change = total - level
    noise = float(np.sum(cube.variance))
    by_order = {}
    for k in range(1, order_count + 1):
        by_order[k] = {
            "power": sum(v for s, v in per_subset.items() if len(s) == k),
            "contrasts": sum(1 for s in per_subset if len(s) == k),
        }
    for k, row in by_order.items():
        row["share_of_change"] = row["power"] / max(change, 1e-30)
    return {
        "total": total, "level": level, "change": change,
        "noise_floor": noise,
        "change_over_floor": change / max(noise, 1e-30),
        "by_order": by_order,
        "per_subset": {":".join(s) if s else "level": v
                       for s, v in per_subset.items()},
        "conserved": float(sum(per_subset.values()) / max(total, 1e-30)),
        "why": ("the fold is orthogonal, so the total is fixed before any "
                "fitting and every contrast's share is a fraction of it; "
                "the transform conserves everything but the level, which is "
                "why what it conserves is the CHANGE"),
    }


def unreached(cube: Cube, sigma: float = 3.0) -> Dict[str, object]:
    """THE RESIDUAL THE FOLD DOES NOT REACH, per order.

    The vertices are rebuilt from the identified contrasts only; what is
    left per vertex is the sum of the contrasts the cube could not
    identify - noise, plus any structure sitting under it. Reported per
    order, with the count of contrasts, how many were identified, their
    structured power, and the power left behind.
    """
    contrasts = walsh(cube)
    verdicts = identified(contrasts, sigma)
    keep = [s for s, v in verdicts.items() if v["identified"] or s == ()]
    model = reconstruct(cube, contrasts, keep)
    residual = cube.values - model
    floor = float(np.mean(cube.variance))
    left = float(np.mean(np.abs(residual) ** 2))
    orders = {}
    for k in range(1, cube.order + 1):
        members = [s for s in verdicts if len(s) == k]
        found = [s for s in members if verdicts[s]["identified"]]
        orders[k] = {
            "contrasts": len(members), "identified": len(found),
            "structured_power": float(sum(max(verdicts[s]["structured_power"], 0.0)
                                          for s in members)),
            "unidentified_power": float(sum(max(verdicts[s]["structured_power"], 0.0)
                                            for s in members if s not in found)),
            "names": [":".join(s) for s in found],
        }
    edges = cube.order
    total = 2 ** cube.order - 1
    top = tuple(cube.axes)
    return {
        "axes": cube.axes, "vertices": 2 ** cube.order,
        "contrasts": total, "edge_contrasts": edges,
        "edge_fraction_by_count": edges / total,
        "identified_total": len(keep) - 1,
        "orders": orders,
        "top_order_identified": bool(verdicts[top]["identified"]),
        "top_order_z": verdicts[top]["z"],
        "residual_power": left, "floor_power": floor,
        "residual_over_floor": left / max(floor, 1e-300),
        "reading": (
            "the top-order contrast stands above noise: the structure has "
            "more dimensions than the cube has axes, and the residual is "
            "not reachable without another measurement"
            if verdicts[top]["identified"] else
            "every unidentified contrast is at its noise: what is left is "
            "the floor the cube's own size sets, lowered only by more "
            "vertices"),
        "verdicts": verdicts,
    }


def asymmetry(cube: Cube) -> Dict[str, Dict[str, float]]:
    """THE SIDES OF THE TESSERACT, in units of their own floors.

    Each axis's edge length is the rms of its order-one contrast over the
    rms of that contrast's noise; equal sides would be a cube, and they are
    not.
    """
    contrasts = walsh(cube)
    out = {}
    for axis in cube.axes:
        entry = contrasts[(axis,)]
        power = float(np.mean(np.abs(entry["value"]) ** 2))
        noise = float(np.mean(entry["variance"]))
        out[axis] = {"rms": float(np.sqrt(power)),
                     "noise_rms": float(np.sqrt(noise)),
                     "side_in_floors": float(np.sqrt(power / max(noise, 1e-300)))}
    sides = [v["side_in_floors"] for v in out.values()]
    out["ratio_longest_to_shortest"] = {
        "value": float(max(sides) / max(min(sides), 1e-300)) if sides else 1.0}
    return out


def dense_graph(cube: Cube) -> Dict[str, object]:
    """EVERY VERTEX CONNECTED TO EVERY OTHER, and what each connection reads.

    The difference between two vertices is the sum of the contrasts on the
    axes they differ in, with signs - so the dense graph's 2^n (2^n - 1) / 2
    differentials are all combinations of the same 2^n - 1 contrasts.
    Siblings differ in one axis (the hypercube's edges); parents are the
    vertices with an axis collapsed (the folds); the rest are the
    diagonals, which the sequential machinery never took.
    """
    verts = cube.vertices
    contrasts = walsh(cube)
    pairs = []
    for u, v in combinations(range(len(verts)), 2):
        differ = tuple(cube.axes[i] for i in range(cube.order)
                       if verts[u][i] != verts[v][i])
        reads = [s for s in contrasts if s and len(set(s) & set(differ)) % 2 == 1]
        pairs.append({"from": verts[u], "to": verts[v], "differ_on": differ,
                      "hamming": len(differ),
                      "kind": "edge" if len(differ) == 1 else "diagonal",
                      "reads": reads})
    parents = {}
    for axis in cube.axes:
        parents[axis] = [(verts[u], verts[v]) for u, v in combinations(
            range(len(verts)), 2)
            if sum(verts[u][i] != verts[v][i] for i in range(cube.order)) == 1
            and verts[u][cube.axes.index(axis)] != verts[v][cube.axes.index(axis)]]
    return {"vertices": verts, "pairs": pairs,
            "edges": sum(1 for p in pairs if p["kind"] == "edge"),
            "diagonals": sum(1 for p in pairs if p["kind"] == "diagonal"),
            "parents_by_axis": parents,
            "contrasts": len(contrasts) - 1}


def on_full_grid(values, bins_hz, sample_rate_hz: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place values measured on a sub-band onto the rfft grid they came
    from, so that the cepstrum sees the right frequency axis.

    THE DEFECT THIS REMOVES. The cepstral relation assumes bin 0 is DC and
    the last bin is Nyquist. The tesseract's cubes carry only the band the
    measurement is valid on - 944 bins from 0.2 to 4 MHz of a 2049-bin
    4 f_sc grid - and feeding those straight to the cepstrum stretches the
    band over the whole axis, which scales every delay by the ratio of the
    two spans and mis-shapes every all-pass term. The first run of
    `causality` did exactly that, and its delays (2 to 112 ns) were
    reported before this was caught. Values outside the band are held flat
    at the band's ends, the export's convention.
    """
    f = np.asarray(bins_hz, dtype=np.float64)
    step = float(np.median(np.diff(f)))
    half = int(round(0.5 * sample_rate_hz / step)) + 1
    grid = np.arange(half) * step
    index = np.rint(f / step).astype(int)
    if index.min() < 0 or index.max() >= half:
        raise ValueError("bins fall outside the grid the sample rate implies")
    full = np.zeros(half, dtype=np.asarray(values).dtype)
    full[index] = values
    full[:index.min()] = values[0]
    full[index.max() + 1:] = values[-1]
    return full, grid, index


def from_sync_exports(paths: Dict[str, str], tapes: Sequence[str],
                      band_hz=(0.2e6, 4.0e6)) -> Cube:
    """The luma cube: head x polarity x half x tape, from the exports.

    Vertex noise is the half's standard error: root two times the pooled
    one, both parts together (calibrated in `residual_floor.log_domain`).
    Values are the complex log of the transfer, where the chain adds.
    """
    if len(tapes) != 2:
        raise ValueError("a binary tape axis takes exactly two tapes")
    loaded = {t: np.load(paths[t], allow_pickle=True) for t in tapes}
    f = np.asarray(loaded[tapes[0]]["frequency_mhz"], dtype=np.float64) * 1e6
    valid = (f >= band_hz[0]) & (f <= band_hz[1])
    keys = {}
    for t in tapes:
        data = loaded[t]
        if not np.array_equal(np.asarray(data["frequency_mhz"]) * 1e6, f):
            raise ValueError("the exports must share one frequency grid")
        for head in ("a", "b"):
            for view in ("fall", "rise"):
                base = f"head_{head}_{view}"
                valid &= np.asarray(data[f"{base}_valid"], dtype=bool)
                for half in ("first", "second"):
                    value = np.asarray(data[f"{base}_H_{half}"])
                    valid &= np.isfinite(value)
                    keys[(head, view, half, t)] = (value, data[f"{base}_se"])
    values = np.zeros((2, 2, 2, 2, int(valid.sum())), dtype=np.complex128)
    variance = np.zeros_like(values, dtype=np.float64)
    for (head, view, half, t), (value, se) in keys.items():
        index = (("a", "b").index(head), ("fall", "rise").index(view),
                 ("first", "second").index(half), tuple(tapes).index(t))
        H = np.asarray(value, dtype=np.complex128)[valid]
        magnitude = np.maximum(np.abs(H), 1e-30)
        values[index] = np.log(magnitude) + 1j * np.unwrap(np.angle(H))
        variance[index] = (np.sqrt(2.0) * np.asarray(se, dtype=np.float64)[valid]
                           / magnitude) ** 2
    return Cube(("head", "polarity", "half", "tape"), values, variance,
                bins=f[valid])


def chroma_burst_cube(tbc_path: str, json_path: str,
                      lines=(20, 250)) -> Dict[str, object]:
    """The chroma vertices: the burst's complex gain per head and half,
    read PER LINE against the phase the decoder imposed on that line.

    From a decoded chroma TBC (16-bit samples at four times the subcarrier,
    the geometry read from its JSON, never assumed): the complex envelope
    at the burst on every line (`chroma._complex_envelope`, the multiply
    the instrument was missing). A decoded chroma TBC does not carry the
    tape's burst phase: it carries the phase the decoder IMPOSED, which is
    the colour-framing base phase of the field's identifier
    (`chroma.ntsc_color_framing_map`, 0 or 180 degrees) less the standard's
    33 degree shift (`chroma.ntsc_color_framing_phase_shift`), alternating
    by 180 degrees from one line to the next because the subcarrier runs
    227.5 cycles a line. Measured on the pnb decode (16 fields): every line
    sits within an interquartile range of about one degree of -32.4 or
    +147.6 degrees, the assignment to even and odd rows flipping with the
    identifier exactly as the map says, and the amplitude within two per
    cent of its median. The first version of this averaged the envelope
    across lines WITHOUT removing that pattern, which averaged across a
    180 degree alternation and left a vertex variance of 0.5 to 1.2 -
    nothing identified. Divided line by line by the imposed phase, the
    residual gain is the measurement: its log, against the field-set mean,
    is the vertex value, and the between-field scatter of the mean is the
    variance.

    MEASURED THIS WAY (pnb, two decodes of the same tape from different
    seek points, 16 fields each): the per-field gain holds to 0.5 per cent
    in amplitude and 0.15 degrees in phase; the vertex variances are 1.4e-6
    to 2.9e-6; the HEAD contrast is identified at z = 12.7 and 11.5 with the
    same value on both decodes, 0.0030 + 0.0010j (head A reads the burst 0.6
    per cent larger and 0.06 degrees ahead of head B), and the half and
    head-by-half contrasts sit at their noise. A head parameter, reproduced
    on evidence that did not build it.
    """
    import json
    from vhsdecode import chroma as chroma_module

    meta = json.load(open(json_path))
    vp = meta["videoParameters"]
    width, height = int(vp["fieldWidth"]), int(vp["fieldHeight"])
    start, end = int(vp["colourBurstStart"]), int(vp["colourBurstEnd"])
    raw = np.fromfile(tbc_path, dtype=np.uint16)
    count = raw.size // (width * height)
    fields = raw[:count * width * height].reshape(count, height, width)
    identifiers = [int(fld["fieldPhaseID"]) for fld in meta["fields"]][:count]
    first_field = [bool(fld["isFirstField"]) for fld in meta["fields"]][:count]
    base_of = {identifier: base for (identifier, base)
               in chroma_module.ntsc_color_framing_map.values()}
    shift = float(chroma_module.ntsc_color_framing_phase_shift)
    rows = np.arange(lines[0], lines[1])
    per_field = []
    for index in range(count):
        block = fields[index, lines[0]:lines[1], start:end].astype(np.float64)
        env = chroma_module._complex_envelope(block, start)
        interior = env[:, 2:-2] if env.shape[1] > 4 else env
        per_line = interior.mean(axis=1)
        imposed = np.deg2rad(base_of[identifiers[index]] - shift
                             + 180.0 * (rows % 2))
        gain = per_line * np.exp(-1j * imposed)
        per_field.append(complex(np.mean(gain)))
    per_field = np.asarray(per_field)
    reference = np.mean(per_field)
    log_gain = np.log(per_field / reference)
    half = np.arange(count) >= count // 2
    values = np.zeros((2, 2, 1), dtype=np.complex128)
    variance = np.zeros_like(values, dtype=np.float64)
    for h, is_first in enumerate((True, False)):
        for s, in_second in enumerate((False, True)):
            pick = [i for i in range(count)
                    if first_field[i] == is_first and half[i] == in_second]
            sample = log_gain[pick]
            values[h, s, 0] = np.mean(sample)
            variance[h, s, 0] = (np.var(sample.real) + np.var(sample.imag)) / max(
                len(sample), 1)
    return {"cube": Cube(("head", "half"), values, variance),
            "per_field": per_field, "fields": count, "log_gain": log_gain,
            "imposed_deg": {identifier: base_of[identifier] - shift
                            for identifier in sorted(base_of)}}


def video_cube(luma: Cube, chroma: Cube) -> Cube:
    """LUMA AND CHROMA TOGETHER on the axes they share.

    The luma's polarity axis is folded to its parent (the LTI part) and its
    tape axis, if any, must already be collapsed; the chroma is appended to
    every vertex as extra bins, so a differential along a shared axis moves
    both channels and the contrasts report what each channel does.

    MEASURED (pnb: the luma's LTI part over 944 bins with the burst gain
    appended, head x half): the head contrast is identified in BOTH
    channels (luma z 389, chroma 19 times its noise); the half and
    head-by-half contrasts are identified in the luma (z 5 and 23) and at
    noise in the chroma. And the two channels' head contrasts have OPPOSITE
    signs: the luma reads head A 0.3 per cent lower than B in the 0.5-0.8
    MHz band of the sync response while the burst reads head A 0.6 per cent
    higher. A shared head-to-tape separation would move both the same way,
    so the per-head term the burst carries is channel-specific - the
    chroma path's own per-head gain - and not the spacing term. That is the
    invariance reading the joint cube exists to give, and it is why the
    channels are folded together rather than compared afterwards.
    """
    part = luma
    for axis in luma.axes:
        if axis not in chroma.axes:
            part = fold(part, axis)["parent"]
    if part.axes != chroma.axes:
        raise ValueError(f"the channels must end on the same axes, luma "
                         f"{part.axes} vs chroma {chroma.axes}")
    values = np.concatenate([part.values, chroma.values], axis=-1)
    variance = np.concatenate([part.variance, chroma.variance], axis=-1)
    n = part.values.shape[-1]
    return Cube(chroma.axes, values, variance,
                channels={"luma": slice(0, n), "chroma": slice(n, None)})


# --------------------------------------------------------------------------
# The path through the graph is known: the chain's stages give it depth
# --------------------------------------------------------------------------

# Ethan: *"but across stages we know the path through this graph a depth we
# can trace"* and *"Eventually it collapses down to one real signal which we
# can subtract to remove the residual."*
#
# The folds commute, so the CONTRASTS do not depend on the order they are
# taken in. What the chain supplies is where each contrast BELONGS: an axis
# is a property of one stage of the chain - the head of the playback
# machine, the polarity of the demodulated edge, the tape, the time - and
# the stage has a position (`interference.full_chain`). Tracing the fold in
# chain order, last applied first removed, is what makes the collapse a
# correction rather than a description: the non-commuting entries (a clip,
# an AGC) stay at their depth, and the linear ones fold to one spectrum.

# Which stage each axis of a cube is a property of. Positions are read from
# the chain, never typed here; the module named must declare one.
# Keys are entries of `interference.full_chain()`; the assignment of an
# axis to an entry is a labelled assumption about which stage first makes
# that axis matter, not a measurement.
AXIS_STAGE = {
    "head": "head differential phase",       # the playback head pair (24)
    "polarity": "input clipping",            # fall differs from rise at the
                                             # record-side clip first (9)
    "tape": "record level dependence",       # the recording on its tape (11)
    "half": "transport (playback)",          # the time base across fields (24)
    "channel": "colour-under envelope amplitude",   # luma against chroma (20)
}


def trace(cube: Cube, chain: Optional[Dict[str, int]] = None
          ) -> Dict[str, object]:
    """THE PATH: each contrast attributed to a depth in the chain.

    A contrast on several axes belongs to the DEEPEST of their stages - the
    one applied first in the chain and therefore removed last - because an
    interaction between a tape property and a head property cannot be
    undone until both are reached. The path is the contrasts in removal
    order: shallowest stage first, and within a stage the lowest order
    first, so an edge is subtracted before the interaction that rides on it.
    """
    from vhsdecode.models import interference

    chain = chain if chain is not None else interference.full_chain(strict=False)
    depth = {}
    for axis in cube.axes:
        module = AXIS_STAGE.get(axis)
        position = chain.get(module) if module else None
        depth[axis] = {"stage": module, "position": position}
    contrasts = walsh(cube)
    verdicts = identified(contrasts)
    path = []
    for subset, verdict in verdicts.items():
        if not subset:
            continue
        positions = [depth[a]["position"] for a in subset
                     if depth[a]["position"] is not None]
        owner = None
        if positions:
            deepest = min(positions)
            owner = next(depth[a]["stage"] for a in subset
                         if depth[a]["position"] == deepest)
        path.append({"contrast": ":".join(subset), "axes": subset,
                     "order": len(subset), "stage": owner,
                     "position": min(positions) if positions else None,
                     "identified": verdict["identified"], "z": verdict["z"]})
    # removal order: last applied (highest position) first, then low order
    path.sort(key=lambda p: (-(p["position"] if p["position"] is not None
                               else -1), p["order"]))
    return {"axes": depth, "path": path,
            "removal_order": [p["contrast"] for p in path],
            "why": ("the folds commute but the chain does not: a contrast "
                    "is removed at the depth of the deepest stage it "
                    "touches, and the shallowest stage goes first")}


def one_real_signal(cube: Cube, sample_rate_hz: float,
                    chain: Optional[Dict[str, int]] = None,
                    sigma: float = 3.0) -> Dict[str, object]:
    """THE COLLAPSE TO ONE REAL SIGNAL that is subtracted from the capture.

    The identified contrasts, traced in removal order, sum in the log
    domain to one complex spectrum per vertex (the vertex's whole
    departure, exact when every contrast is kept, and its identified part
    otherwise). The correction is `exp(-sum)`, the identity
    `correction_export.collapse` carries; and because the capture is a REAL
    sample stream the correction's impulse response is real, obtained by
    completing the spectrum with its Hermitian image. What returns is that
    real kernel, one per vertex, and the residual per vertex after it is
    applied in the log domain - the unidentified contrasts, at their floor.

    `sample_rate_hz` fixes the grid the real kernel is built on; the cube's
    bins are the positive frequencies of the measurement.
    """
    contrasts = walsh(cube)
    verdicts = identified(contrasts, sigma)
    order = trace(cube, chain)["removal_order"]
    keep = [()] + [tuple(name.split(":")) for name in order
                   if verdicts[tuple(name.split(":"))]["identified"]]
    departure = reconstruct(cube, contrasts, keep)
    residual = cube.values - departure
    kernels = {}
    f = np.asarray(cube.bins, dtype=np.float64)
    for vertex in cube.vertices:
        # the departure placed back on the rfft grid it was cut from, held
        # flat outside the measured band (a first version interpolated
        # onto a grid of the band's own size, 1890 taps for a 4096-point
        # measurement, which was a coarser kernel than the measurement)
        full, _grid, _index = on_full_grid(departure[vertex], f, sample_rate_hz)
        full[:_index.min()] = 0.0
        full[_index.max() + 1:] = 0.0
        spectrum = np.exp(-full)
        # a real kernel has a real spectrum at DC and at Nyquist; a
        # departure that reaches either bin can only be honoured there in
        # magnitude, and the measured bands never do (0.2-4 MHz)
        spectrum[0] = np.real(spectrum[0])
        spectrum[-1] = np.real(spectrum[-1])
        n_fft = int(2 * (full.size - 1))
        kernel = np.fft.irfft(spectrum, n=n_fft)      # real by construction
        kernels[vertex] = kernel
    return {
        "removal_order": order,
        "identified": [":".join(s) for s in keep if s],
        "departure": departure,
        "residual": residual,
        "residual_over_floor": float(np.mean(np.abs(residual) ** 2)
                                     / max(np.mean(cube.variance), 1e-300)),
        "kernels": kernels,
        "kernel_is_real": all(np.isrealobj(k) for k in kernels.values()),
        "why": ("a product of responses is a sum of logarithms, so every "
                "identified contrast folds to one exp(-sum); a real capture "
                "takes a real kernel, and what it cannot remove is the sum "
                "of the contrasts the cube could not identify"),
    }


# --------------------------------------------------------------------------
# Time enters as the bits of the field index: every scale folded at once
# --------------------------------------------------------------------------

def fill_field_gaps(responses, variances, field_index, total: int,
                    inflate: float = 4.0) -> Dict[str, object]:
    """Fill the fields a record is missing from their same-head neighbours.

    A dropped field (the instrument returned nothing for it) breaks the
    consecutive run the fold needs. Each missing field is replaced by the
    mean of the nearest measured fields two away on either side - the same
    head, so the head contrast is not blurred - and its variance is set to
    `inflate` times theirs, so the fold weighs it as the guess it is. The
    filled indices are returned and must be reported with any result; a
    record with more than a few per cent filled is not a measurement of
    the scales the gaps sit at.
    """
    values = np.asarray(responses, dtype=np.complex128)
    var = np.asarray(variances, dtype=np.float64)
    fields = np.asarray(field_index, dtype=int)
    bins = values.shape[-1]
    full = np.full((total, bins), np.nan, dtype=np.complex128)
    full_var = np.full((total, bins), np.nan, dtype=np.float64)
    full[fields] = values
    full_var[fields] = var
    present = np.zeros(total, dtype=bool)
    present[fields] = True
    filled = []
    for n in np.flatnonzero(~present):
        sources = [k for k in (n - 2, n + 2) if 0 <= k < total and present[k]]
        if not sources:
            sources = [k for k in (n - 1, n + 1) if 0 <= k < total and present[k]]
        if not sources:
            continue
        full[n] = np.mean([full[k] for k in sources], axis=0)
        full_var[n] = inflate * np.mean([full_var[k] for k in sources], axis=0)
        filled.append(int(n))
    keep = ~np.isnan(full[:, 0])
    return {"responses": full[keep], "variances": full_var[keep],
            "field_index": np.flatnonzero(keep), "filled": filled,
            "filled_fraction": len(filled) / max(total, 1)}


def from_field_series(responses, variances, field_index, bins=None,
                      field_rate_hz: float = 60.0 / 1.001) -> Dict[str, object]:
    """THE TIME AXIS AS A HIERARCHY OF BINARY AXES: the field index's bits.

    A per-field measurement over 2^k consecutive fields is a cube whose
    axes are the BITS of the field index. Bit 0 is the head (the heads
    alternate every field), bit 1 is the drum revolution (two fields a
    frame), and each higher bit is a time scale twice the last, up to the
    whole record. The fold on all of them at once is the Walsh-Hadamard
    (sequency) spectrum of the series over field time, and a contrast on a
    subset of bits is the response's variation at exactly that combination
    of scales - so the drum, the head switch, and every slower mechanism
    appear as contrasts on named axes, with the noise of each fixed by how
    many fields the record holds. This is what "each adjacent dimension of
    measurements gets another dimension added" means on the time axis:
    doubling the record adds one bit, one axis, and halves every contrast's
    noise.

    The fields must be consecutive and their count a power of two; a longer
    record is truncated to the largest power of two and the truncation
    reported. Mechanism rates from `transport_model.rotation_rates` say
    which bit each mechanism sits nearest; that mapping is reported, not
    assumed.

    MEASURED (countdown, 256 consecutive fields = 4.27 s decoded at the
    recorded flags `-s 500 -n -f 40 -t 3 --dctp --ire0_adjust --lti_gain 0`,
    the fall-edge sync response per field from the arc's own instrument,
    944 bins 0.2-4 MHz, no field missing):

        bit 0    1 field    59.94 Hz   power/noise  37.6   z   794   head
        bit 1    2 fields   29.97 Hz                 8.1   z   155   drum
        bit 2    4 fields   14.99 Hz                17.5   z   358
        bit 3    8 fields    7.49 Hz                17.6   z   361
        bit 4   16 fields    3.75 Hz                44.2   z   938
        bit 5   32 fields    1.87 Hz                41.9   z   888
        bit 6   64 fields    0.94 Hz                41.3   z   876
        bit 7  128 fields    0.47 Hz                48.0   z  1020

    Every scale carries structure, all 255 contrasts are identified and the
    top-order one stands at z 227. The head bit is the largest; the drum
    bit - the two-field parity, which is also the colour-frame A/B parity -
    is real at z 155 though its band-averaged mean cancels (the four
    identifiers' band means differ by 0.01 at a standard error of 0.009);
    its structure is a frequency shape, not a level. From a quarter of a
    second to two seconds the power over noise is flat at 40 to 48: the
    response drifts across the transport's whole band rather than at one
    mechanism's line. The field-time spectrum of the band-averaged log
    magnitude has its strongest lines at 19.7, 18.7 and 20.4 Hz (7 to 10
    times the median) and the capstan's 4.83 Hz at 3.4 times; none is a
    line yet at this record length.

    AND ON HOME, 1024 consecutive fields = 17.08 s at the same flags, no
    field missing, all ten bits identified and 993 of the 1023 contrasts:

        bit 0    1 field    59.94 Hz   power/noise 252.8   z  5470
        bit 1    2 fields   29.97 Hz                 7.6   z   143
        bit 3    8 fields    7.49 Hz                 2.4   z    31
        bit 5   32 fields    1.87 Hz                34.8   z   733
        bit 8  256 fields    0.23 Hz                78.8   z  1691
        bit 9  512 fields    0.12 Hz               252.6   z  5466

    THE REEL BAND IS A LINE AT THIS LENGTH. The field-time spectrum of the
    band-averaged log magnitude puts 33 times the median at 0.293 Hz, inside
    the 0.118-0.442 Hz band `transport_model.rotation_rates` gives the reels
    (five bins of the 0.0585 Hz grid fall in it), with the impedance roller
    at 6.1 and the pinch roller at 5.0 times. The four-second and
    eight-second bits carrying 79 and 253 times their noise are the same
    thing read as scales rather than lines. The 4.27 s record could not
    resolve this; 17 s can, which is the transport memory's "about five
    seconds of tape" confirmed.

    THE DRUM CANNOT BE READ THIS WAY, and the number that looks like it is
    a trap. A once-per-field series has its Nyquist at 29.970 Hz, and the
    drum turns once per two fields - EXACTLY 29.970 Hz. So the drum aliases
    onto the head alternation itself: the 72-times-median "line at 29.970
    Hz" is bit 0, the head, and no per-field measurement can separate the
    two. Separating them needs a within-field (per line) measurement.

    (A first run of this on home was made at `-f 50` where the capture is
    40 MSps; the decoder locked and produced plausible fields, and the
    figures were withdrawn when the flag memory was read.)
    """
    values = np.asarray(responses, dtype=np.complex128)
    var = np.asarray(variances, dtype=np.float64)
    fields = np.asarray(field_index, dtype=int)
    order = np.argsort(fields)
    values, var, fields = values[order], var[order], fields[order]
    if not np.all(np.diff(fields) == 1):
        raise ValueError("the fields must be consecutive")
    k = int(np.floor(np.log2(fields.size)))
    used = 2 ** k
    values, var, fields = values[:used], var[:used], fields[:used]
    axes = tuple(f"field bit {b}" for b in range(k))
    shape = (2,) * k + (values.shape[-1],)
    cube_values = np.zeros(shape, dtype=np.complex128)
    cube_var = np.zeros(shape, dtype=np.float64)
    for n in range(used):
        index = tuple(int((n >> b) & 1) for b in range(k))
        cube_values[index] = values[n]
        cube_var[index] = var[n]
    scales = {axis: {"fields": 2 ** b, "seconds": 2 ** b / field_rate_hz,
                     "rate_hz": field_rate_hz / 2 ** b}
              for b, axis in enumerate(axes)}
    scales[axes[0]]["mechanism"] = "head (alternates every field)"
    if k > 1:
        scales[axes[1]]["mechanism"] = "drum revolution (two fields)"
    return {"cube": Cube(axes, cube_values, cube_var, bins=bins),
            "fields_used": used, "fields_dropped": int(fields.size - used)
            if False else int(np.asarray(field_index).size - used),
            "scales": scales}
