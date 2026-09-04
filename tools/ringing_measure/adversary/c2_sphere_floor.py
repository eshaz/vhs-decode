"""Claim 2: sphere floor N/(L+N-1) and scatter sqrt(2)/L.

Independent re-derivation and a wider sweep than the doc used, including
N > L, N ~ L, real vs complex.
"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models.information_extrapolation import sphere_floor

rng = np.random.default_rng(12345)


def measure(N, L, complexv, trials=400):
    """Asymmetry of a pure-noise ensemble exactly as ellipsoid computes it."""
    out = np.empty(trials)
    for t in range(trials):
        if complexv:
            D = rng.standard_normal((N, L)) + 1j * rng.standard_normal((N, L))
        else:
            D = rng.standard_normal((N, L))
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        G = D @ D.conj().T
        G = 0.5 * (G + G.conj().T)
        v = np.clip(np.real(np.linalg.eigvalsh(G))[::-1], 0.0, None)
        share = v / v.sum()
        part = 1.0 / np.sum(share ** 2)
        out[t] = (N - part) / (N - 1)
    return out


print("=== 2a. Algebra of the closed form ===")
print("""  participation = (tr G)^2 / tr(G^2) = N^2 / (N + sum_{i!=j}|G_ij|^2).
  With E|G_ij|^2 = 1/L that is N^2 / (N + N(N-1)/L) = N*L/(L+N-1),
  so asymmetry = (N - part)/(N-1) = N(N-1) / ((N-1)(L+N-1)) = N/(L+N-1).
  ALGEBRA CONFIRMED -- but note it substitutes E[num]/E[den] for E[num/den],
  a first-order (delta-method) approximation, exact only as L -> inf.
""")

print("=== 2b. Wider sweep: measured / predicted ===")
rows = []
print(f"  {'N':>4} {'L':>5} {'kind':>8} {'pred':>9} {'meas':>9} {'ratio':>7} "
      f"{'pred scat':>10} {'meas scat':>10} {'s-ratio':>8}")
grid = []
for N in (3, 4, 8, 16, 32, 64, 128):
    for L in (8, 16, 32, 64, 128, 256, 1024):
        grid.append((N, L))
for N, L in grid:
    for complexv in (False, True):
        a = measure(N, L, complexv, trials=300)
        pred = sphere_floor(N, L)
        r = a.mean() / pred["asymmetry"]
        sr = a.std() / pred["scatter"]
        rows.append((N, L, complexv, r, sr, a.mean(), pred["asymmetry"],
                     a.std(), pred["scatter"]))
        if (N, L) in ((3, 8), (4, 64), (8, 8), (16, 16), (32, 8), (32, 32),
                      (64, 8), (128, 8), (16, 1024), (128, 1024)):
            print(f"  {N:>4} {L:>5} {'complex' if complexv else 'real':>8} "
                  f"{pred['asymmetry']:9.4f} {a.mean():9.4f} {r:7.3f} "
                  f"{pred['scatter']:10.5f} {a.std():10.5f} {sr:8.3f}")

import collections
print("\n=== 2c. Where the mean-ratio breaks, by regime ===")
for kind, cx in (("real", False), ("complex", True)):
    sel = [r for r in rows if r[2] == cx]
    for lbl, f in (("L >= 8N", lambda r: r[1] >= 8 * r[0]),
                   ("L ~ N (0.5..2x)", lambda r: 0.5 <= r[1] / r[0] <= 2),
                   ("N > 4L", lambda r: r[0] > 4 * r[1])):
        s = [r[3] for r in sel if f(r)]
        if s:
            print(f"  {kind:8s} {lbl:16s} n={len(s):3d} "
                  f"measured/predicted {min(s):.3f} .. {max(s):.3f}")

print("\n=== 2d. The scatter: is it sqrt(2)/L, and is it flat in N? ===")
print("  Prediction from second moments of the Gram off-diagonals:")
print("    real   : Var(g^2)=2/L^2 -> sd(asym) ~ 2*sqrt(N/(N-1))/L")
print("    complex: |g|^2~Exp     -> sd(asym) ~ sqrt(2)*sqrt(N/(N-1))/L")
print(f"  {'N':>4} {'L':>5} {'meas sd*L':>10} {'code sqrt2':>11} "
      f"{'my real':>9} {'my cplx':>9}")
for N in (3, 4, 8, 16, 32):
    for L in (256, 1024):
        for cx in (False, True):
            a = measure(N, L, cx, trials=1500)
            f = np.sqrt(N / (N - 1.0))
            print(f"  {N:>4} {L:>5} {'C' if cx else 'R'} {a.std()*L:10.4f} "
                  f"{np.sqrt(2):11.4f} {2*f:9.4f} {np.sqrt(2)*f:9.4f}")

print("\n=== 2e. Consequence for the sigma test ===")
print("  If the real-case scatter is ~2/L but the code divides by sqrt(2)/L,")
print("  every reported sigma on a REAL ensemble is inflated by 2/sqrt(2) =")
print(f"  {2/np.sqrt(2):.3f}x.  A true 3-sigma departure reads 4.24 sigma.")
