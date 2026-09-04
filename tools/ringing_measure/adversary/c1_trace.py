"""Claim 1: trace(D D^H) = N.  Discovery or tautology?"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(0)

print("=== 1a. Is the trace identity a property of the DATA or of line 2499? ===")
# Deliberately pathological ensembles: wildly unequal norms, collinear,
# rank-1, with NaNs, huge dynamic range.
def show(label, vecs):
    res = {f"c{i}": {"frequency": v} for i, v in enumerate(vecs)}
    fit = ie.ellipsoid(res, "frequency")
    D = np.array([v / np.linalg.norm(v) for v in vecs])
    G = D @ D.conj().T
    print(f"  {label:38s} N={len(vecs):3d}  trace(G)={np.trace(G).real:.12f}"
          f"  ellipsoid['trace']={fit['trace']:.12f}")

L = 256
base = rng.standard_normal(L) + 1j * rng.standard_normal(L)
show("4 independent complex", [rng.standard_normal(L) + 1j*rng.standard_normal(L) for _ in range(4)])
show("4 views of ONE (rank 1)", [c * base for c in (1.0, -3.0, 0.01, 1e6)])
show("norms spanning 1e-9..1e9", [base * s + 1e-3*rng.standard_normal(L) for s in (1e-9, 1.0, 1e3, 1e9)])
show("17 collinear + noise", [base + 1e-6*rng.standard_normal(L) for _ in range(17)])

print("""
  VERDICT: trace(G) = N for every one of these, including a rank-1 ensemble
  and one spanning 18 orders of magnitude in norm.  It is enforced at
  information_extrapolation.py:2494-2499:
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        ... stack[keep] / norms[keep]
  Each row is divided by its own norm, so G_ii == 1 identically and
  trace(G) == N by definition of the trace.  No data can change it.
""")

print("=== 1b. Does ellipsoid['trace'] actually report N? ===")
res = {f"c{i}": {"frequency": rng.standard_normal(L)} for i in range(6)}
for removed in (0, 1, 3, 5, 6):
    fit = ie.ellipsoid(res, "frequency", removed=removed)
    print(f"  removed={removed}: information={fit['information']:.0f}"
          f"  reported trace={fit['trace']:.6f}"
          f"  asymmetry={fit['asymmetry']:.6f}")
print("""  The key 'trace' is sum(values[:effective]) (line 2643), NOT trace(G).
  It equals N only when removed==0.  The doc's table has removed=0 so it
  reads 4.0000000000; the same field on a deflated ensemble is < N.""")

print("=== 1c. The doc's own §2 table, reproduced ===")
def asym(vecs):
    res = {f"c{i}": {"frequency": v} for i, v in enumerate(vecs)}
    return ie.ellipsoid(res, "frequency")

shared = rng.standard_normal(L) + 1j*rng.standard_normal(L)
sets = {
 "four independent": [rng.standard_normal(L)+1j*rng.standard_normal(L) for _ in range(4)],
 "four views of ONE": [shared*c for c in (1, 2, 3, 4)],
 "shared plus private": [shared + (rng.standard_normal(L)+1j*rng.standard_normal(L))*2 for _ in range(4)],
}
for k, v in sets.items():
    f = asym(v)
    print(f"  {k:22s} rank {f['rank']} of 4  trace {f['trace']:.10f}"
          f"  asym {f['asymmetry']:.3f}  semi-axes "
          + " ".join(f"{x:.3f}" for x in f['semi_axes']))
