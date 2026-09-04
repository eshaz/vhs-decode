"""Claim 7: run the tool's TWELVE ensembles on pure noise and compare with
the published real-data run.

Sizes taken verbatim from docs/ELLIPTICAL_COLLAPSE.md sections 6 and 6a.
"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(2024)


def nested(vectors, axis="x"):
    n = len(vectors)
    out = {f"c{i}": {axis: vectors[i]} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            out[f"c{i} vs c{j}"] = {axis: vectors[i] - vectors[j]}
    return out


# (label, n_components, n_places, published rank, published directions,
#  published asym first -> last, published resolved %)
SPEC = [
    ("frequency / head A", 13, 119, 13, 11, 0.268, 0.005, 92.9),
    ("frequency / head B", 14, 120, 14, 11, 0.190, 0.025, None),
    ("frequency / both  ", 27, 119, 27, 14, 0.710, 0.081, 84.0),
    ("time      / head A", 13, 264, 13, 13, 0.185, 0.000, 100.0),
    ("time      / head B", 14, 262, 14, 14, 0.224, 0.000, None),
    ("time      / both  ", 27, 258, 27, 19, 0.893, 0.032, 91.7),
    ("amplitude / head A", 13, 9, 9, 2, 0.832, 0.832, 82.8),
    ("amplitude / head B", 14, 9, None, None, None, None, None),
    ("amplitude / both  ", 27, 9, 9, 1, 0.924, 0.924, None),
    ("field     / head A", 119, 13, 13, 2, 0.726, 0.726, 54.7),
    ("field     / head B", 120, 14, None, None, None, None, None),
    ("field     / both  ", 119, 27, 27, 3, 0.857, 0.219, None),
]

print("PURE-NOISE BASELINE for the tool's twelve ensembles")
print("(iid Gaussian components; NO structure of any kind)")
print()
print(f"{'ensemble':20s} {'entr':>5} {'plc':>4} {'rank':>4} {'sig':>4} "
      f"{'resolved%':>9} {'ell asym':>8} {'ell floor':>9} {'ell sig':>8} "
      f"{'loop first':>10} {'loop last':>9} {'dirs':>4} {'circle':>6}")
tot_dirs = 0
reached = 0
noise_rows = {}
for label, n, L, *pub in SPEC:
    v = [rng.standard_normal(L) for _ in range(n)]
    m = nested(v)
    fit = ie.ellipsoid(m, "x")                       # report_ellipse: removed=0
    loop = ie.differentiate_to_floor(m, "x", maximum_passes=8)
    first = loop["trace"][0]["asymmetry"]
    last = loop["trace"][-1]["asymmetry"]
    d = len(loop["targets"])
    tot_dirs += d
    reached += int(loop["at_floor"])
    noise_rows[label] = (fit, first, last, d)
    print(f"{label:20s} {len(m):>5} {L:>4} {fit['rank']:>4} "
          f"{fit['significant']:>4} {100*fit['resolved_fraction']:>8.1f}% "
          f"{fit['asymmetry']:>8.4f} {fit['sphere_floor']:>9.4f} "
          f"{fit['sigma']:>+8.1f} {first:>10.4f} {last:>9.4f} {d:>4} "
          f"{str(loop['at_floor']):>6}")
print()
print(f"REACHED THE FLOOR IN {reached} OF {len(SPEC)} ENSEMBLES; "
      f"{tot_dirs} directions recovered in total")
print("  (published on the real 27-field decode: 12 of 12, 86 directions)")

print()
print("SIDE BY SIDE with the published real-data numbers")
print(f"{'ensemble':20s} {'pub rank':>8} {'noise rank':>10} "
      f"{'pub dirs':>8} {'noise dirs':>10} {'pub res%':>8} {'noise res%':>10} "
      f"{'pub last':>8} {'noise last':>10}")
for label, n, L, *pub in SPEC:
    prank, pdir, pfirst, plast, pres = pub
    fit, first, last, d = noise_rows[label]
    print(f"{label:20s} {str(prank):>8} {fit['rank']:>10} {str(pdir):>8} "
          f"{fit['significant']:>10} "
          f"{('%.1f' % pres) if pres is not None else '-':>8} "
          f"{100*fit['resolved_fraction']:>9.1f}% "
          f"{('%.3f' % plast) if plast is not None else '-':>8} "
          f"{last:>10.3f}")

print()
print("=== The 114-118 sigma claim for the time axis ===")
for label, n, L in (("time / head A", 13, 264), ("time / head B", 14, 262)):
    sigs = []
    for _ in range(40):
        v = [rng.standard_normal(L) for _ in range(n)]
        f = ie.ellipsoid(nested(v), "x")
        sigs.append(f["sigma"])
    print(f"  {label}: pure noise reports {np.mean(sigs):.1f} +- "
          f"{np.std(sigs):.1f} sigma above the sphere floor "
          f"(published: 114-118)")
for label, n, L in (("frequency / head A", 13, 119),):
    sigs = [ie.ellipsoid(nested([rng.standard_normal(L) for _ in range(n)]),
                         "x")["sigma"] for _ in range(40)]
    print(f"  {label}: pure noise {np.mean(sigs):.1f} +- {np.std(sigs):.1f} "
          f"sigma (published: +39.3)")
