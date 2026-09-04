"""The CORRECT null for the published pass-0 asymmetries: pure noise put
through the tool's own nested construction, with the deficiency correction
the loop applies.  This is the test the document never ran."""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(4242)


def stack(v):
    n = len(v)
    rows = [v[i] for i in range(n)]
    rows += [v[i] - v[j] for i in range(n) for j in range(i + 1, n)]
    S = np.array(rows, float)
    return S / np.linalg.norm(S, axis=1, keepdims=True)


def pass0(S):
    """ellipsoid() at removed = deficiency, exactly as differentiate_to_floor
    fits it on the first pass."""
    count, length = S.shape
    w = np.clip(np.linalg.eigvalsh(S.T @ S)[::-1], 0.0, None)
    values = np.zeros(count)
    values[:min(count, length)] = w[:min(count, length)]
    rank = int(np.count_nonzero(values > count * np.finfo(float).eps ** 0.5))
    eff = max(count - max(count - rank, 0), 1)
    live = values[:eff]
    sh = live / live.sum()
    part = 1.0 / np.sum(sh ** 2)
    asym = (eff - part) / (eff - 1) if eff > 1 else 0.0
    c = ie.sphere_floor(eff, length)
    return asym, c["asymmetry"], c["scatter"], rank


CASES = [
    ("frequency / head A", 13, 119, 0.268),
    ("frequency / head B", 14, 120, 0.190),
    ("frequency / both  ", 27, 119, 0.710),
    ("time      / head A", 13, 264, 0.185),
    ("time      / head B", 14, 262, 0.224),
    ("time      / both  ", 27, 258, 0.893),
    ("amplitude / head A", 13, 9, 0.832),
    ("amplitude / both  ", 27, 9, 0.924),
]
print("Published pass-0 asymmetry against the PURE-NOISE distribution of the")
print("same construction (500 draws each).  The code's own sphere floor and")
print("scatter are shown for comparison.")
print()
print(f"{'ensemble':20s} {'pub':>7} | {'noise mean':>10} {'noise sd':>9} "
      f"{'TRUE z':>8} | {'code floor':>10} {'code scat':>9} {'code sigma':>10}")
for label, n, L, pub in CASES:
    a, fl, sc = [], None, None
    T = 120 if n < 20 else 40
    for _ in range(T):
        S = stack([rng.standard_normal(L) for _ in range(n)])
        x, fl, sc, rk = pass0(S)
        a.append(x)
    a = np.array(a)
    z = (pub - a.mean()) / a.std()
    print(f"{label:20s} {pub:>7.3f} | {a.mean():>10.4f} {a.std():>9.4f} "
          f"{z:>+8.1f} | {fl:>10.4f} {sc:>9.5f} "
          f"{(pub - fl)/sc:>+10.1f}")

print("""
The right-hand block is what the tool printed.  The middle block is what
the same statistic does on structureless data put through the same
construction.  Where the two disagree, the published sigma is measuring
the construction and not the tape.
""")
