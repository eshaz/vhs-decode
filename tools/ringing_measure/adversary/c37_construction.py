"""Claims 3 and 7: the MP edge as a significance test, and whether the real
-data run's "100% of budget / asymmetry exactly 0.000" is constructed.

The nested differential matrix built by tools/ringing_measure/elliptical_collapse
.build_matrix has, for N fields, N diagonal entries d_i and N(N-1)/2
off-diagonal entries d_i - d_j.  All of them live in span{d_1..d_N}.
So the matrix has M = N(N+1)/2 rows of rank <= N.

Feed it PURE NOISE fields -- no structure whatsoever -- and see what the
transform reports.
"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(7)


def nested(d):
    """Exactly what build_matrix -> component_differentials produces."""
    N = len(d)
    out = {f"field {i}": {"frequency": d[i]} for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            out[f"field {i} vs field {j}"] = {"frequency": d[i] - d[j]}
    return out


print("=== 7a. PURE NOISE fields through the real tool's construction ===")
print("  d_i = iid Gaussian.  No shared structure, no departures, nothing.")
print(f"  {'fields':>6} {'places':>7} {'entries':>8} {'rank':>5} {'MPedge':>7} "
      f"{'signif':>7} {'bulk':>7} {'resolved':>9} {'share':>7} "
      f"{'asym0':>7} {'asymEnd':>8} {'passes':>7} {'reason'}")
for N, L in ((13, 264), (14, 262), (27, 258), (13, 119), (27, 119),
             (13, 1000), (6, 264), (40, 264)):
    d = [rng.standard_normal(L) for _ in range(N)]
    m = nested(d)
    fit = ie.ellipsoid(m, "frequency")
    loop = ie.differentiate_to_floor(m, "frequency")
    end = loop["trace"][-1]
    print(f"  {N:>6} {L:>7} {len(m):>8} {fit['rank']:>5} {fit['mp_edge']:>7.3f} "
          f"{fit['significant']:>7} {fit['bulk']:>7.4f} {fit['resolved']:>9.3f} "
          f"{100*fit['resolved_fraction']:>6.1f}% {loop['trace'][0]['asymmetry']:>7.4f} "
          f"{end['asymmetry']:>8.4f} {loop['passes']:>7} {loop['reason']}")

print("""
  READ: pure noise gives rank == field count, significant == rank,
  bulk == 0.0000, resolved == the entire budget (100.0%), and the loop
  terminates with asymmetry EXACTLY 0.0000 in two passes.  That is the
  document's headline real-data result, reproduced with no signal at all.
""")

print("=== 7b. WHY: the three lines that construct it ===")
N, L = 13, 264
d = [rng.standard_normal(L) for _ in range(N)]
m = nested(d)
fit = ie.ellipsoid(m, "frequency")
v = fit["eigenvalues"]
M = len(m)
print(f"  M={M} rows of rank {N} in R^{L}.  trace(G)=M={M} (rows unit-normed),")
print(f"  spread over {N} nonzero eigenvalues -> mean nonzero = M/N = {M/N:.2f}")
print(f"  MP edge (1+sqrt(M/L))^2 = {fit['mp_edge']:.3f}")
print(f"  nonzero eigenvalues: " + " ".join(f"{x:.2f}" for x in v[:N]))
print(f"  smallest nonzero {v[N-1]:.3f} vs edge {fit['mp_edge']:.3f} -> "
      f"{'ALL above' if v[N-1] > fit['mp_edge'] else 'some below'}")
print(f"  eigenvalues <= edge: {np.sum(v <= fit['mp_edge'])} of them, "
      f"median = {np.median(v[v <= fit['mp_edge']]):.3e}   <-- 'bulk'")
print("""  ellipsoid line 2663-2666:
        quiet_bulk = values[values <= mp_edge]
        bulk_level = median(quiet_bulk)              # == 0, they are the
        resolved   = sum(values[:significant] - bulk)#    structural zeros
  so resolved == sum of every nonzero eigenvalue == trace == the budget.
  '100% of the information budget' is arithmetically forced whenever the
  ensemble is rank-deficient enough that the structural zeros outnumber
  the true bulk -- which build_matrix guarantees, because it emits
  N(N+1)/2 rows spanning only N dimensions.
""")

print("=== 7c. The exact zero asymmetry ===")
loop = ie.differentiate_to_floor(m, "frequency")
print(f"  entries {M}, rank {N}, deficiency = {M - N} = {M-N}")
print(f"  after pass 1 removes {fit['significant']} directions:")
print(f"    removed = significant + deficiency = {fit['significant']} + {M-N} "
      f"= {fit['significant'] + M - N}")
print(f"    effective = max(count - removed, 1) = max({M} - "
      f"{fit['significant']+M-N}, 1) = {max(M - (fit['significant']+M-N), 1)}")
print("""  information_extrapolation.py:2647-2648:
        asymmetry = ((effective - participation) / (effective - 1)
                     if effective > 1 else 0.0)
  effective == 1 takes the else branch and RETURNS THE LITERAL 0.0.
  The 'asymmetry falls to exactly 0.000' in the document's time-axis rows
  is this constant, not a measurement.  It cannot be anything else once
  significant == rank.""")

print("\n=== 7d. Does ANY input make the time axis not reach 100%? ===")
print("  Trying to break it: heavy structure, one shared departure, ")
print("  identical fields, adversarial rank.")
cases = {}
base = rng.standard_normal(L)
cases["13 identical fields"] = [base.copy() for _ in range(13)]
cases["13 = one departure + tiny noise"] = [base + 1e-9*rng.standard_normal(L) for _ in range(13)]
cases["13 heavy-tailed"] = [rng.standard_t(1.5, L) for _ in range(13)]
cases["13 rank-3 mixtures"] = [
    sum(c*rng.standard_normal() for c in [rng.standard_normal(L) for _ in range(3)])
    if False else (np.array([rng.standard_normal(L) for _ in range(3)]).T @ rng.standard_normal(3))
    for _ in range(13)]
cases["13 smooth (low-order poly)"] = [
    np.polyval(rng.standard_normal(4), np.linspace(-1, 1, L)) for _ in range(13)]
for k, d in cases.items():
    m = nested(d)
    f = ie.ellipsoid(m, "frequency")
    lp = ie.differentiate_to_floor(m, "frequency")
    print(f"  {k:32s} rank {f['rank']:>3} signif {f['significant']:>3} "
          f"resolved {100*f['resolved_fraction']:6.1f}%  end asym "
          f"{lp['trace'][-1]['asymmetry']:.4f}  passes {lp['passes']}  "
          f"{lp['reason'][:34]}")

print("\n=== 3. Does the MP edge ever admit pure noise as a direction? ===")
print("  (a) On a PROPER iid ensemble (rows independent, L > N), as MP assumes:")
exceed = 0
trials = 0
for N in (4, 8, 16, 32):
    for L in (64, 128, 256, 1024):
        hits = 0
        T = 300
        for _ in range(T):
            D = rng.standard_normal((N, L))
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            v = np.linalg.eigvalsh(D @ D.T)[::-1]
            edge = (1 + np.sqrt(N / L)) ** 2
            if v[0] > edge:
                hits += 1
        exceed += hits
        trials += T
        if N in (16, 32) and L in (64, 1024):
            D = rng.standard_normal((N, L)); D /= np.linalg.norm(D, axis=1, keepdims=True)
            v = np.linalg.eigvalsh(D @ D.T)[::-1]
            print(f"    N={N:>3} L={L:>5}: lmax/edge typical "
                  f"{v[0]/((1+np.sqrt(N/L))**2):.3f}, P(lmax>edge) = {hits/T:.3f}")
print(f"    OVERALL false-direction rate on iid noise: {exceed}/{trials} = "
      f"{100*exceed/trials:.2f}%")
print("  (b) On the tool's own rank-deficient construction (7a above): 100%.")
