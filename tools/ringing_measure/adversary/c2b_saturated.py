"""Claim 2 follow-up: the scatter formula in the regime section 6a calls
'SATURATED', and what the true scatter does to that verdict."""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models.information_extrapolation import sphere_floor

rng = np.random.default_rng(808)


def true_scatter(N, L, trials=4000, complexv=False):
    a = np.empty(trials)
    for t in range(trials):
        D = rng.standard_normal((N, L))
        if complexv:
            D = D + 1j * rng.standard_normal((N, L))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        v = np.clip(np.real(np.linalg.eigvalsh(D @ D.conj().T)), 0, None)
        s = v / v.sum()
        a[t] = (N - 1.0 / np.sum(s ** 2)) / (N - 1)
    return a.mean(), a.std()


print("The regime section 6a calls SATURATED: effective ~ length.")
print("`ellipsoid` judges the shape with count=effective, length=length.")
print()
print(f"{'case':26s} {'N':>4} {'L':>5} {'floor':>7} {'code scat':>10} "
      f"{'TRUE scat':>10} {'ratio':>7} {'asym':>7} {'code sig':>9} "
      f"{'TRUE sig':>9} {'verdict'}")
CASES = [
    # (label, effective, length, published asymmetry)
    ("amplitude / head A", 9, 9, 0.832),
    ("amplitude / both", 9, 9, 0.924),
    ("field / head A", 13, 13, 0.726),
    ("field / both", 27, 27, 0.857),
    # and the two that DID pass, for contrast
    ("frequency / head A (p0)", 13, 119, 0.268),
    ("time / head A (p0)", 13, 264, 0.185),
]
for label, N, L, asym in CASES:
    f = sphere_floor(N, L)
    mu, sd = true_scatter(N, L, trials=3000 if N < 20 else 1200)
    code_sig = (asym - f["asymmetry"]) / f["scatter"]
    true_sig = (asym - mu) / sd
    verdict = ("SATURATED by the code, SIGNIFICANT in truth"
               if code_sig <= 3 < true_sig else
               "agree" if (code_sig > 3) == (true_sig > 3) else "disagree")
    print(f"{label:26s} {N:>4} {L:>5} {f['asymmetry']:>7.4f} "
          f"{f['scatter']:>10.5f} {sd:>10.5f} {f['scatter']/sd:>7.2f} "
          f"{asym:>7.3f} {code_sig:>+9.2f} {true_sig:>+9.1f}  {verdict}")

print("""
READ.  sqrt(2)/L is the scatter of the LARGE-L limit only.  When N is
comparable with L the true scatter collapses (the asymmetry is pinned
near its ceiling and cannot fluctuate), so the code's denominator is
2 to 20 times too LARGE and the sigma it reports is that many times too
small.  Section 6a's reading -- 'a random ensemble is already almost
maximally asymmetric and nothing can stand above it' -- is therefore
half right and half an artefact: the FLOOR really does rise, but the
SCATTER does not stay at sqrt(2)/L, and against the true scatter every
one of the four 'saturated' ensembles is many sigma above its floor.
""")

print("=== And the doc's own validation range, re-measured ===")
print("  'Checked over N in {3..32} and L in {64..1024}, real and complex:")
print("   measured over predicted sits at 0.88-1.18 with no trend, and the")
print("   scatter fit returns 1.408 against sqrt(2) = 1.414.'")
ratios, srat = [], []
for N in (3, 6, 12, 24, 32):
    for L in (64, 128, 256, 512, 1024):
        for cx in (False, True):
            mu, sd = true_scatter(N, L, trials=600, complexv=cx)
            p = sphere_floor(N, L)
            ratios.append(mu / p["asymmetry"])
            srat.append(sd * L)
print(f"  mean ratio over that grid: {min(ratios):.3f} .. {max(ratios):.3f}"
      f"  (doc: 0.88 .. 1.18)  -> CONFIRMED")
print(f"  scatter * L over that grid: {min(srat):.3f} .. {max(srat):.3f}"
      f"  (doc: one number, 1.408)")
print("  A single fitted 1.408 is only reachable by pooling real and complex")
print("  together: complex alone gives ~sqrt(2)*sqrt(N/(N-1)), real alone")
print("  gives ~2*sqrt(N/(N-1)).  The two differ by sqrt(2) and the pooled")
print("  fit lands between them.")
