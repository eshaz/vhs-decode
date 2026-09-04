"""Claim 4: 'the halting count is two, flat'.

Reproduce the planted-departure experiment, then break the assumptions the
document's version quietly makes: orthogonal departures, equal strengths,
N well above the departure count.
"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(99)


def plant(N, L, k, strength=1.0, angle_deg=90.0, strengths=None, trials=30,
          complexv=False):
    """N components = noise + k planted departures shared across them."""
    passes, recovered, circle = [], [], []
    for _ in range(trials):
        def rv(*s):
            a = rng.standard_normal(s)
            if complexv:
                a = a + 1j * rng.standard_normal(s)
            return a
        # departures with a controlled mutual angle
        deps = []
        base = rv(L); base /= np.linalg.norm(base)
        deps.append(base)
        for i in range(1, k):
            w = rv(L); w -= (base.conj() @ w) * base
            w /= np.linalg.norm(w)
            t = np.radians(angle_deg)
            d = np.cos(t) * base + np.sin(t) * w
            deps.append(d / np.linalg.norm(d))
        amps = strengths if strengths is not None else [strength] * k
        comps = {}
        for i in range(N):
            v = rv(L) / np.sqrt(L)
            for d, a in zip(deps, amps):
                v = v + a * rng.standard_normal() * d
            comps[f"c{i}"] = {"x": v}
        out = ie.differentiate_to_floor(comps, "x", maximum_passes=12)
        passes.append(out["passes"])
        recovered.append(len(out["targets"]))
        circle.append(out["at_floor"])
    return np.mean(passes), np.mean(recovered), np.mean(circle)


print("=== 4a. The document's own table, reproduced (orthogonal, equal) ===")
print(f"{'N':>4} {'planted':>8} {'passes':>7} {'recovered':>10} {'circle':>7}")
for N in (6, 10, 16):
    for k in (0, 1, 2, 4, 5):
        if k >= N:
            continue
        p, r, c = plant(N, 512, k, strength=1.0)
        print(f"{N:>4} {k:>8} {p:>7.1f} {r:>10.1f} {100*c:>6.0f}%")

print("\n=== 4b. NON-ORTHOGONAL departures (the doc never tests this) ===")
print("  Two planted departures at a controlled angle, N=16, L=512.")
print(f"{'angle':>7} {'passes':>7} {'recovered':>10} {'circle':>7}")
for ang in (90, 60, 45, 30, 20, 10, 5):
    p, r, c = plant(16, 512, 2, strength=1.0, angle_deg=ang)
    print(f"{ang:>6}d {p:>7.1f} {r:>10.1f} {100*c:>6.0f}%")

print("\n=== 4c. UNEQUAL strengths ===")
print("  Three planted departures with a strength ratio, N=16, L=512.")
print(f"{'ratio':>10} {'passes':>7} {'recovered':>10} {'circle':>7}")
for ratio in (1, 3, 10, 30, 100, 1000):
    s = [1.0, 1.0 / ratio, 1.0 / ratio ** 2]
    p, r, c = plant(16, 512, 3, strengths=s)
    print(f"{ratio:>10} {p:>7.1f} {r:>10.1f} {100*c:>6.0f}%")

print("\n=== 4d. N close to the departure count ===")
print(f"{'N':>4} {'planted':>8} {'passes':>7} {'recovered':>10} {'circle':>7}")
for N, k in ((6, 3), (6, 4), (6, 5), (8, 5), (8, 6), (8, 7),
             (10, 7), (10, 8), (10, 9), (16, 12), (16, 14), (16, 15)):
    p, r, c = plant(N, 512, k)
    print(f"{N:>4} {k:>8} {p:>7.1f} {r:>10.1f} {100*c:>6.0f}%")

print("\n=== 4e. Weak departures (SNR near the edge) ===")
print("  N=16, L=512, 2 planted, strength swept.")
print(f"{'strength':>9} {'passes':>7} {'recovered':>10} {'circle':>7}")
for s in (3.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
    p, r, c = plant(16, 512, 2, strength=s)
    print(f"{s:>9.2f} {p:>7.1f} {r:>10.1f} {100*c:>6.0f}%")

print("\n=== 4f. Is 'passes' ever > 2 for anything the doc would call a pass? ===")
print("  Sweeping N and k widely, recording the maximum pass count seen.")
worst = (0, None)
for N in (6, 8, 10, 16, 24, 32):
    for k in (1, 2, 3, 5, 8):
        if k >= N - 1:
            continue
        p, r, c = plant(N, 512, k, trials=15)
        if p > worst[0]:
            worst = (p, (N, k, p, r, c))
print(f"  worst mean pass count seen: {worst[0]:.2f} at N,k,passes,rec,circle="
      f"{worst[1]}")
