"""Stacking tapes as a third axis: why a tensor fit is a measurement and a
matrix fit is not.

Ethan: *"Stacking tapes as a third axis breaks the rotational ambiguity of a
2D fit. Trilinear decompositions are essentially unique under Kruskal's
condition; matrix ones aren't. This makes fitted components readable as
measurements rather than as a convenient parameterization."*

And the invariance structure that makes the axes mean something:
*"Parameters separate because of different invariances: gap per head, preamp
per deck, spacing/contact per tape, timing per session."*

THE MATHEMATICAL FACT, and it is a fact rather than a heuristic. A matrix
`X = A B^T` is not determined by X: for ANY invertible Q, `(A Q)(Q^-1 B)^T`
is the same matrix, so a two-way fit identifies only the SUBSPACE the
components span and never the components themselves. Every "component" a
matrix decomposition returns is one arbitrary basis of that subspace, and the
singular vectors are merely the basis that happens to be orthogonal. Reading
them as physical mechanisms is the error this whole arc has been circling.

A three-way tensor `X_ijk = sum_r a_ir b_jr c_kr` is different in kind.
Kruskal (1977): if the k-ranks of the three factor matrices satisfy

    k_A + k_B + k_C >= 2R + 2

then the decomposition is unique up to permutation and scaling of the
components. No rotation is free. The components ARE determined by the data,
which is exactly what "readable as measurements" means. The k-rank of a
matrix is the largest k such that EVERY set of k columns is independent - a
stricter thing than rank, and the thing that makes the condition bite.

WHAT THE THIRD AXIS BUYS, IN THIS REPOSITORY'S OWN TERMS. The ensemble has
been a matrix: components against frequency. The magnetics collapsed to 1.58
effectively distinguishable of 6 because six mechanisms that are functions of
one dimensionless group span a one-dimensional subspace - and inside a
subspace, a matrix fit cannot say which mechanism is which. Add tapes and
heads as axes and each mechanism acquires a SIGNATURE ALONG THE NEW AXES that
the others do not share: a head parameter is constant across tapes, a tape
parameter is constant across heads, a deck parameter is constant across both.
Those signatures are independent by construction, so the k-ranks along the
new axes are what carry Kruskal's condition, and the count of identifiable
components is set by HOW MANY TAPES AND HEADS there are, not by how many
mechanisms the physics names.

With this repository's data - two heads and three tapes with sync-step
responses - the condition gives `R + 2 + 3 >= 2R + 2`, so `R <= 3`: three
components are uniquely identifiable, however many mechanisms exist. Each
tape added raises the ceiling by one. That is a statement about the data one
must collect, and it is derived rather than hoped.

THE DEMONSTRATION THIS MODULE CARRIES, because a claim of uniqueness has to
be shown failing where it should fail. `rotational_ambiguity` plants known
factors, fits them as a matrix and as a tensor, and reports how well each
recovers the PLANTED components - not the subspace, the components. The
matrix fit recovers the subspace to machine precision and the components only
up to a rotation; the tensor fit recovers the components. And with too few
tapes for the rank - Kruskal violated - the tensor fit fails too, which is
the control that keeps the claim honest.
"""

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def k_rank(matrix, tolerance: float = 1e-9) -> int:
    """Kruskal's k-rank: the largest k such that EVERY k columns are
    independent.

    Stricter than the rank. A matrix can have full rank and a k-rank of one -
    two identical columns anywhere reduce it to one - and it is the k-rank
    that Kruskal's condition needs. Computed exactly by enumeration, which is
    fine for the handful of components a physical model carries and is not
    meant for more.
    """
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] == 0:
        return 0
    columns = values.shape[1]
    scale = float(np.abs(values).max()) or 1.0
    for k in range(1, columns + 1):
        for chosen in combinations(range(columns), k):
            sub = values[:, list(chosen)]
            if np.linalg.matrix_rank(sub, tol=tolerance * scale) < k:
                return k - 1
    return columns


def kruskal(factors: Sequence[np.ndarray]) -> Dict[str, object]:
    """Whether Kruskal's condition holds for these factor matrices.

    `k_A + k_B + k_C >= 2R + 2` for a three-way decomposition of rank R.
    Returns the k-ranks, the margin, and the maximum rank the given axes
    could ever identify - which is the number to plan data collection around.
    """
    ranks = [k_rank(f) for f in factors]
    rank = int(np.asarray(factors[0]).shape[1])
    total = int(sum(ranks))
    needed = 2 * rank + 2
    # the ceiling the NON-frequency axes impose: with k_freq = R at best,
    # R + k_B + k_C >= 2R + 2  ->  R <= k_B + k_C - 2
    others = int(sum(ranks[1:]))
    return {
        "k_ranks": ranks,
        "rank": rank,
        "sum": total,
        "needed": needed,
        "margin": total - needed,
        "unique": bool(total >= needed),
        "ceiling_from_the_other_axes": max(others - 2, 0),
        "why": ("k_A + k_B + k_C >= 2R + 2 makes the decomposition unique up "
                "to permutation and scale; the ceiling is set by how many "
                "tapes and heads there are, not by how many mechanisms exist"),
    }


def _khatri_rao(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Column-wise Kronecker product, the workhorse of the ALS update."""
    r = first.shape[1]
    return np.einsum("ir,jr->ijr", first, second).reshape(-1, r)


def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def _rank_one_columns(columns: np.ndarray, shape: Tuple[int, int]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each column of a Khatri-Rao product back into its two factors.

    Column r of `B (.) C` is `vec(b_r c_r^T)` with row index j*K + k, so
    reshaping it to (J, K) gives a rank-one matrix whose leading singular
    pair is (b_r, c_r) up to the scale, which is returned separately.
    """
    j_size, k_size = shape
    rank = columns.shape[1]
    B = np.zeros((j_size, rank), dtype=columns.dtype)
    C = np.zeros((k_size, rank), dtype=columns.dtype)
    scale = np.zeros(rank, dtype=columns.dtype)
    for r in range(rank):
        # M = b c^T = u s v^H, so the row vh[0] = v^H is c^T/|c| as it
        # stands: no conjugate. Conjugating it here cost the complex start
        # its exactness (1.0e-4 after 500 rounds against 1.4e-10 in 3 for
        # the real case), the discarded-phase family once more.
        u, s, vh = np.linalg.svd(columns[:, r].reshape(j_size, k_size))
        B[:, r], C[:, r], scale[r] = u[:, 0], vh[0], s[0]
    return B, C, scale


def gevd_initial(tensor, rank: int, seed: int = 0
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A closed-form start for CP from a generalised eigenvalue problem.

    Sanchez and Kowalski (1990) and Leurgans, Ross and Abel (1993): take two
    slices of the tensor along one axis, `S_0 = A D_0 C^T` and
    `S_1 = A D_1 C^T` with the `D` diagonal, compress both to rank x rank
    with the leading singular vectors of the other two unfoldings, and the
    eigenvectors of `T_1 T_0^-1 = A' D_1 D_0^-1 A'^-1` ARE the columns of
    the compressed `A`. Noiseless, this is exact; the alternating least
    squares that follows only has to confirm it.

    WHY IT IS NEEDED HERE. The tensor this file exists for is frequency x
    head x tape with TWO heads, and a rank-three fit of a two-row factor is
    exactly the case where alternating least squares crawls: the head
    factor's Gram matrix is singular, its Hadamard product with the tape
    Gram is barely invertible, and from a singular-vector start the fit
    error on a NOISELESS planted tensor was still 3.2e-3 after 500 rounds
    (components 0.943). Two slices along the head axis are also exactly what
    the eigenvalue method wants, so the case that starves ALS feeds this.

    The slice axis is the shortest one with at least two entries; the two
    slices are random combinations of all of them, so the method does not
    depend on any particular pair being well conditioned. Returns None when
    no axis arrangement offers two slices with the other two sizes at least
    `rank` - the caller then falls back to the singular-vector start.
    """
    X = np.asarray(tensor)
    generator = np.random.default_rng(seed)
    order = np.argsort(X.shape)
    for slice_mode in order:
        if X.shape[slice_mode] < 2:
            continue
        others = [m for m in range(3) if m != slice_mode]
        if min(X.shape[m] for m in others) < rank:
            continue
        Y = np.moveaxis(X, slice_mode, 1)
        weights = generator.standard_normal((2, Y.shape[1]))
        slices = [np.tensordot(weights[i], Y, axes=([0], [1]))
                  for i in range(2)]
        u, _s, _v = np.linalg.svd(_unfold(Y, 0), full_matrices=False)
        v, _s, _v = np.linalg.svd(_unfold(Y, 2), full_matrices=False)
        U, V = u[:, :rank], v[:, :rank]
        T = [U.conj().T @ s @ V.conj() for s in slices]
        try:
            values, vectors = np.linalg.eig(T[1] @ np.linalg.inv(T[0]))
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(vectors)):
            continue
        A = U @ vectors
        if not np.iscomplexobj(X) and np.abs(A.imag).max() < 1e-6 * max(
                np.abs(A.real).max(), 1e-300):
            A = A.real
        # the remaining two factors from the Khatri-Rao columns A leaves
        columns = (np.linalg.pinv(A) @ _unfold(Y, 0)).T
        B, C, _scale = _rank_one_columns(columns, (Y.shape[1], Y.shape[2]))
        # Y's axes are (others[0], slice_mode, others[1]) of X
        factors = [None, None, None]
        factors[others[0]], factors[slice_mode], factors[others[1]] = A, B, C
        return tuple(factors)
    return None


def cp_decompose(tensor, rank: int, iterations: int = 500,
                 tolerance: float = 1e-10, seed: int = 0
                 ) -> Dict[str, object]:
    """CANDECOMP/PARAFAC by alternating least squares, for a three-way tensor.

    `X_ijk = sum_r a_ir b_jr c_kr`. Each factor is updated in closed form with
    the other two held, which is exact at each step and converges
    monotonically. Started from the closed-form `gevd_initial` where the
    tensor's shape allows it (it does for frequency x two heads x three
    tapes), and otherwise from the singular vectors of each unfolding, so a
    run is reproducible; `seed` fixes the random slice combinations and the
    small perturbation of the fallback start.

    Written plainly here rather than imported, because the arithmetic is
    short, the repository carries no tensor library, and a reader should be
    able to see that nothing beyond least squares is happening.
    """
    X = np.asarray(tensor, dtype=np.complex128 if np.iscomplexobj(tensor)
                   else np.float64)
    if X.ndim != 3:
        raise ValueError("cp_decompose takes a three-way tensor")
    generator = np.random.default_rng(seed)

    def initial(mode):
        u, _s, _v = np.linalg.svd(_unfold(X, mode), full_matrices=False)
        basis = u[:, :rank]
        if basis.shape[1] < rank:
            extra = generator.standard_normal((X.shape[mode],
                                               rank - basis.shape[1]))
            basis = np.column_stack([basis, extra.astype(basis.dtype)])
        return basis + 1e-3 * generator.standard_normal(basis.shape)

    def unit_columns(factor):
        scale = np.linalg.norm(factor, axis=0, keepdims=True)
        return factor / np.where(scale > 0, scale, 1.0)

    start = gevd_initial(X, rank, seed=seed)
    if start is None:
        start = (initial(0), initial(1), initial(2))
    A, B, C = (unit_columns(np.asarray(f, dtype=X.dtype)) for f in start)
    lam = np.ones(rank)
    norm = float(np.linalg.norm(X))
    history: List[float] = []

    for _ in range(int(iterations)):
        for mode in range(3):
            # THE KHATRI-RAO ORDER MUST MATCH THE UNFOLDING, and the first
            # version had it backwards - the fit error sat at exactly 1.0,
            # which is the signature of a solve against the wrong basis.
            # `_unfold(X, 0)` flattens the remaining axes (J, K) in C order,
            # so its column index is j*K + k, and the matching Khatri-Rao
            # product is B (over j) with C (over k), in that order.
            if mode == 0:
                kr, gram = _khatri_rao(B, C), (B.conj().T @ B) * (C.conj().T @ C)
            elif mode == 1:
                kr, gram = _khatri_rao(A, C), (A.conj().T @ A) * (C.conj().T @ C)
            else:
                kr, gram = _khatri_rao(A, B), (A.conj().T @ A) * (B.conj().T @ B)
            # A SMALL RIDGE, AND UNIT COLUMNS AFTER EVERY STEP. Bare ALS lets
            # the scale wander between factors - one grows while another
            # shrinks, the product unchanged - until the Gram matrix is
            # singular to machine precision and the pseudo-inverse's SVD
            # stops converging, which is exactly how the first version of
            # this died. The ridge is tied to the Gram's own trace so it is
            # scale-free, and the two factors held fixed are kept at unit
            # column norm so the scale has nowhere to hide.
            ridge = 1e-10 * float(np.abs(np.trace(gram))) / max(rank, 1)
            # min ||X_(m) - F kr^T|| gives F kr^T conj(kr) = X_(m) conj(kr),
            # and kr^T conj(kr) is the TRANSPOSE of the Gram kr^H kr. For
            # real data the two are equal; for complex data they are
            # conjugates, and inverting the untransposed Gram rotates every
            # component's phase by the Gram's own - the discarded-phase
            # family again, in the solver this time.
            solve = (_unfold(X, mode) @ kr.conj()) @ np.linalg.inv(
                gram.T + ridge * np.eye(rank))
            if not np.all(np.isfinite(solve)):
                break
            # THE SCALE MUST GO SOMEWHERE. A previous version normalised the
            # two factors being HELD, which changes the product they were
            # just solved against - the fit error then sat at 0.9 on a
            # noiseless tensor. The factor just solved absorbs all the scale;
            # it is normalised and the scale is carried in `lam`, so the
            # product A B C lam is preserved exactly at every step.
            lam = np.linalg.norm(solve, axis=0)
            lam = np.where(lam > 0, lam, 1.0)
            if mode == 0:
                A = solve / lam
            elif mode == 1:
                B = solve / lam
            else:
                C = solve / lam
        fitted = np.einsum("r,ir,jr,kr->ijk", lam, A, B, C)
        error = float(np.linalg.norm(X - fitted) / max(norm, 1e-30))
        history.append(error)
        if len(history) > 2 and abs(history[-2] - history[-1]) < tolerance:
            break

    # every factor is already unit-column; `lam` carries the scale
    weights = np.asarray(lam, dtype=X.dtype)
    order = np.argsort(-np.abs(weights))
    A, B, C, weights = A[:, order], B[:, order], C[:, order], weights[order]
    return {
        "factors": (A, B, C),
        "weights": weights,
        "relative_error": history[-1] if history else float("nan"),
        "iterations": len(history),
        "kruskal": kruskal((A, B, C)),
    }


def match_components(recovered: np.ndarray, planted: np.ndarray
                     ) -> Dict[str, object]:
    """How well a set of recovered components matches a PLANTED set - as
    components, not as a subspace.

    Each planted column is matched to its best recovered column by absolute
    normalised inner product, greedily and without reuse. A perfect recovery
    scores one on every component. A recovery that spans the right subspace
    but has rotated inside it scores well below one on every component while
    scoring one on the subspace - which is the distinction the whole module
    exists to make.
    """
    R = np.asarray(recovered)
    P = np.asarray(planted)
    Rn = R / np.maximum(np.linalg.norm(R, axis=0, keepdims=True), 1e-30)
    Pn = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-30)
    overlap = np.abs(Pn.conj().T @ Rn)
    taken, scores = set(), []
    for i in range(P.shape[1]):
        best, best_j = -1.0, None
        for j in range(R.shape[1]):
            if j not in taken and overlap[i, j] > best:
                best, best_j = float(overlap[i, j]), j
        taken.add(best_j)
        scores.append(best)
    # the SUBSPACE match: how much of each planted column lies in the span
    # of the recovered ones, which a rotated basis still gets right
    q, _ = np.linalg.qr(Rn)
    projected = q @ (q.conj().T @ Pn)
    subspace = [float(np.linalg.norm(projected[:, i]))
                for i in range(P.shape[1])]
    return {
        "component_scores": scores,
        "worst_component": float(min(scores)),
        "subspace_scores": subspace,
        "worst_subspace": float(min(subspace)),
        "components_recovered": bool(min(scores) > 0.99),
        "subspace_recovered": bool(min(subspace) > 0.99),
    }


def rotational_ambiguity(frequencies: int = 256, heads: int = 2,
                         tapes: int = 3, rank: int = 3, noise: float = 1e-3,
                         seed: int = 1) -> Dict[str, object]:
    """THE DEMONSTRATION: a matrix fit recovers the subspace and not the
    components; a tensor fit recovers the components - while Kruskal holds.

    Plants `rank` components with distinct signatures along the head and tape
    axes - the invariance structure Ethan names - builds the tensor, and fits
    it three ways:

        matrix    SVD of the frequency unfolding, the two-way fit this
                  repository has used throughout
        tensor    CP with `tapes` tapes
        control   CP again with the tapes reduced until Kruskal fails

    and scores each against the PLANTED components. The expected outcome, and
    the one this returns, is: matrix subspace one and components well below
    one; tensor components one; control components well below one again. The
    control is what keeps the claim honest - uniqueness is a property of the
    data's dimensions, not of the algorithm.
    """
    generator = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, int(frequencies))
    # three physically distinct shapes along frequency
    shapes = np.column_stack([np.exp(-3.0 * grid),
                              np.sin(2 * np.pi * 1.5 * grid),
                              grid ** 2 - 0.3])[:, :rank]
    if rank > 3:
        shapes = np.column_stack([shapes, generator.standard_normal(
            (grid.size, rank - 3))])
    # invariance signatures: one constant across tapes (a head parameter),
    # one constant across heads (a tape parameter), one varying on both
    head_sig = generator.standard_normal((heads, rank))
    tape_sig = generator.standard_normal((tapes, rank))
    if rank >= 1:
        tape_sig[:, 0] = 1.0            # head parameter: same on every tape
    if rank >= 2:
        head_sig[:, 1] = 1.0            # tape parameter: same on every head
    planted = (shapes, head_sig, tape_sig)
    X = np.einsum("ir,jr,kr->ijk", *planted)
    X = X + noise * generator.standard_normal(X.shape) * np.abs(X).max()

    # 1. the matrix fit: an unfolding's singular vectors
    u, _s, _v = np.linalg.svd(_unfold(X, 0), full_matrices=False)
    matrix = match_components(u[:, :rank], shapes)

    # 2. the tensor fit with all the tapes
    fit = cp_decompose(X, rank)
    tensor = match_components(fit["factors"][0], shapes)

    # 3. the control: strip tapes until Kruskal fails
    few = max(1, rank - 1) if tapes >= rank else 1
    reduced = cp_decompose(X[:, :, :few], rank)
    control = match_components(reduced["factors"][0], shapes)

    return {
        "planted_rank": rank,
        "matrix": {"components": matrix["worst_component"],
                   "subspace": matrix["worst_subspace"]},
        "tensor": {"components": tensor["worst_component"],
                   "subspace": tensor["worst_subspace"],
                   "kruskal": fit["kruskal"], "error": fit["relative_error"]},
        "control": {"tapes": few, "components": control["worst_component"],
                    "kruskal": reduced["kruskal"]},
        "claim_holds": bool(matrix["worst_subspace"] > 0.99
                            and matrix["worst_component"] < 0.95
                            and tensor["worst_component"] > 0.99
                            and control["worst_component"] < 0.95),
        "why": ("a matrix identifies the subspace and any rotation inside it "
                "fits equally; a tensor identifies the components while "
                "Kruskal holds, and stops when the tapes run out"),
    }


def invariances(factors: Sequence[np.ndarray],
                axes: Sequence[str] = ("frequency", "head", "tape"),
                flat: float = 0.05) -> List[Dict[str, object]]:
    """READ EACH RECOVERED COMPONENT AS A MEASUREMENT: what is it invariant to?

    Ethan: *"gap per head, preamp per deck, spacing/contact per tape, timing
    per session"*. A component's loading along an axis is flat when the
    mechanism does not depend on that axis. So a component flat across tapes
    and varying across heads is a HEAD parameter; flat across heads and
    varying across tapes is a TAPE parameter; flat across both is the DECK.
    That reading is only legitimate because the tensor fit is unique - the
    same question asked of a matrix fit's rotated basis would be meaningless.
    """
    out = []
    rank = np.asarray(factors[0]).shape[1]
    for r in range(rank):
        profile = {}
        for name, factor in zip(axes[1:], factors[1:]):
            column = np.asarray(factor)[:, r]
            magnitude = np.abs(column)
            spread = (float(magnitude.std() / magnitude.mean())
                      if magnitude.mean() > 0 else float("inf"))
            profile[name] = {"spread": spread, "flat": bool(spread < flat)}
        flat_axes = [name for name, p in profile.items() if p["flat"]]
        varying = [name for name, p in profile.items() if not p["flat"]]
        if not varying:
            reading = "the deck: constant across heads and tapes"
        elif varying == ["head"]:
            reading = "a head parameter: varies per head, same on every tape"
        elif varying == ["tape"]:
            reading = "a tape parameter: varies per tape, same on every head"
        else:
            reading = "varies on every axis: content, or a per-recording term"
        out.append({"component": r, "profile": profile,
                    "flat_across": flat_axes, "varies_across": varying,
                    "reading": reading})
    return out
