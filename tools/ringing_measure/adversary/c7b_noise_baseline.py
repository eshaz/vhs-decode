"""Claim 7 (b): the tool's twelve ensembles on pure noise.

The nine small ensembles go through the repository code unchanged.  The
three FIELD ensembles have 7140 rows, so their Gram is 7140x7140 and the
repository's eigh on it is too slow to sweep; for those the ellipsoid
arithmetic is reimplemented exactly (verified line for line against the
repository on the small cases first).
"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(2024)


def nested(v, axis="x"):
    n = len(v)
    o = {f"c{i}": {axis: v[i]} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            o[f"c{i} vs c{j}"] = {axis: v[i] - v[j]}
    return o


def nested_stack(v):
    """The same matrix as an array, rows unit-normalised."""
    n = len(v)
    rows = [v[i] for i in range(n)]
    rows += [v[i] - v[j] for i in range(n) for j in range(i + 1, n)]
    S = np.array(rows, dtype=np.float64)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    return S


def my_ellipsoid(S, removed=0):
    """ellipsoid() reimplemented; eigenvalues via the small side."""
    count, length = S.shape
    small = S.T @ S                       # length x length, same nonzero spec
    w = np.clip(np.linalg.eigvalsh(small)[::-1], 0.0, None)
    values = np.zeros(count)
    values[:min(count, length)] = w[:min(count, length)]
    floor = count * np.finfo(np.float64).eps ** 0.5
    rank = int(np.count_nonzero(values > floor))
    mp_edge = (1.0 + np.sqrt(count / max(length, 1))) ** 2
    significant = int(np.count_nonzero(values > mp_edge))
    effective = max(count - max(removed, 0), 1)
    live = values[:effective]
    total = live.sum()
    share = live / total if total > 0 else live
    part = 1.0 / np.sum(share ** 2) if np.any(share) else float(effective)
    asym = (effective - part) / (effective - 1) if effective > 1 else 0.0
    c = ie.sphere_floor(max(effective, 1), length)
    quiet = values[values <= mp_edge]
    bulk = float(np.median(quiet)) if quiet.size else 1.0
    resolved = float(np.sum(np.clip(values[:significant] - bulk, 0, None))) \
        if significant else 0.0
    return dict(count=count, length=length, rank=rank, significant=significant,
                bulk=bulk, resolved=resolved,
                resolved_fraction=resolved / max(count, 1e-30),
                asymmetry=asym, sphere_floor=c["asymmetry"],
                sigma=(asym - c["asymmetry"]) / c["scatter"], mp_edge=mp_edge)


print("=== verification: my reimplementation against the repository ===")
for n, L in ((6, 64), (13, 119), (13, 264)):
    v = [rng.standard_normal(L) for _ in range(n)]
    a = ie.ellipsoid(nested(v), "x")
    b = my_ellipsoid(nested_stack(v))
    ok = (a["rank"] == b["rank"] and a["significant"] == b["significant"]
          and abs(a["asymmetry"] - b["asymmetry"]) < 1e-9
          and abs(a["resolved"] - b["resolved"]) < 1e-8
          and abs(a["sigma"] - b["sigma"]) < 1e-6)
    print(f"  n={n} L={L}: match {ok}  (asym {a['asymmetry']:.6f} vs "
          f"{b['asymmetry']:.6f}; sigma {a['sigma']:.2f} vs {b['sigma']:.2f})")

SPEC = [
    ("frequency / head A", 13, 119, "13", "11", 92.9, "0.268->0.005", 39.3),
    ("frequency / head B", 14, 120, "14", "11", None, "0.190->0.025", None),
    ("frequency / both  ", 27, 119, "27", "14", 84.0, "0.710->0.081", None),
    ("time      / head A", 13, 264, "13", "13", 100.0, "0.185->0.000", 114.),
    ("time      / head B", 14, 262, "14", "14", None, "0.224->0.000", 118.),
    ("time      / both  ", 27, 258, "27", "19", 91.7, "0.893->0.032", None),
    ("amplitude / head A", 13, 9, "9", "2", 82.8, "0.832->0.832", None),
    ("amplitude / head B", 14, 9, "-", "-", None, "-", None),
    ("amplitude / both  ", 27, 9, "9", "1", None, "0.924->0.924", None),
    ("field     / head A", 119, 13, "13", "2", 54.7, "0.726->0.726", None),
    ("field     / head B", 120, 14, "-", "-", None, "-", None),
    ("field     / both  ", 119, 27, "27", "3", None, "0.857->0.219", None),
]

print()
print("PURE-NOISE BASELINE, report_ellipse view (removed=0, as the tool calls it)")
print(f"{'ensemble':20s} {'entries':>7} {'plc':>4} | {'pub rank':>8} "
      f"{'rank':>5} | {'pub dir':>7} {'dir':>4} | {'pub res%':>8} {'res%':>7} | "
      f"{'pub sigma':>9} {'sigma':>8}")
for label, n, L, prank, pdir, pres, ploop, psig in SPEC:
    v = [rng.standard_normal(L) for _ in range(n)]
    b = my_ellipsoid(nested_stack(v))
    print(f"{label:20s} {b['count']:>7} {L:>4} | {prank:>8} {b['rank']:>5} | "
          f"{pdir:>7} {b['significant']:>4} | "
          f"{('%.1f' % pres) if pres else '-':>8} "
          f"{100*b['resolved_fraction']:>6.1f}% | "
          f"{('%.0f' % psig) if psig else '-':>9} {b['sigma']:>+8.1f}")

print()
print("Repeated draws, to show these are not one lucky sample:")
for label, n, L in (("frequency / head A", 13, 119),
                    ("time / head A", 13, 264),
                    ("time / head B", 14, 262),
                    ("time / both", 27, 258),
                    ("field / head A", 119, 13),
                    ("amplitude / head A", 13, 9)):
    res, sig, sgn = [], [], []
    for _ in range(30):
        v = [rng.standard_normal(L) for _ in range(n)]
        b = my_ellipsoid(nested_stack(v))
        res.append(100 * b["resolved_fraction"]); sig.append(b["sigma"])
        sgn.append(b["significant"])
    print(f"  {label:20s} resolved {np.mean(res):6.2f} +- {np.std(res):.2f} %"
          f"   sigma {np.mean(sig):+8.1f} +- {np.std(sig):.1f}"
          f"   directions {np.mean(sgn):.1f}")
