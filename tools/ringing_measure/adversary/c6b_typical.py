"""Claim 6, part 2: reproduce 1.58 from hm.TYPICAL and find what it depends on.

hm.TYPICAL sets azimuth_error_degrees = 0, contour_depth = 0, gain_db = 0.
A 5% perturbation of zero is zero, so those three mechanisms have to be
given a value the model does not supply.  That choice is the parameter the
answer turns on.
"""
import sys
import itertools
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import head_model as hm

MECH = ["spacing_m", "gap_m", "thickness_m", "azimuth_error_degrees",
        "contour_depth", "gain_db"]
LABEL = ["spacing", "gap", "thickness", "azimuth", "contour", "gain"]
SPEED, WIDTH = 5.8, 58e-6


def eff(base, band=(0.5e6, 8.0e6, 512), delta=0.05, demean=False):
    lo, hi, n = band
    f = np.linspace(lo, hi, n)
    rows = []
    for key in MECH:
        cur = float(base.get(key, 0.0) or 0.0)
        a = dict(base)
        b = dict(base)
        b[key] = cur * (1.0 + delta) if cur != 0 else delta
        sig = hm.log_response(f, SPEED, WIDTH, **b) - hm.log_response(f, SPEED, WIDTH, **a)
        rows.append(np.nan_to_num(sig))
    S = np.array(rows)
    if demean:
        S = S - S.mean(axis=1, keepdims=True)
    nn = np.linalg.norm(S, axis=1, keepdims=True)
    S = S / np.where(nn > 0, nn, 1)
    s = np.linalg.svd(S, compute_uv=False)
    share = s ** 2 / max((s ** 2).sum(), 1e-30)
    return 1.0 / np.sum(share ** 2), s, np.abs(S @ S.T)


BASE = dict(hm.TYPICAL)
print("=== 6a'. From hm.TYPICAL, with the three zero mechanisms given a value ===")
for az, cd, gd in ((0.1, 0.05, 1.0), (1.0, 0.05, 1.0), (6.0, 0.05, 1.0),
                   (0.01, 0.01, 0.1)):
    b = dict(BASE, azimuth_error_degrees=az, contour_depth=cd, gain_db=gd)
    p, s, C = eff(b)
    print(f"  azimuth {az:>5}deg contour {cd:>5} gain {gd:>4} dB -> "
          f"effective {p:.3f}   s = {np.round(s, 4)}")

print("\n  Published: 2.159 1.000 0.565 0.133 0.0078 0.0002, effective 1.58")
print("  Searching for the (azimuth, contour, gain) that reproduces it...")
best = None
for az in (0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 6.0, 10.0):
    for cd in (0.001, 0.01, 0.05, 0.1, 0.3):
        for gd in (0.01, 0.1, 1.0, 3.0):
            b = dict(BASE, azimuth_error_degrees=az, contour_depth=cd, gain_db=gd)
            p, s, C = eff(b)
            err = abs(p - 1.58)
            if best is None or err < best[0]:
                best = (err, az, cd, gd, p, s, C)
print(f"  closest: azimuth {best[1]} contour {best[2]} gain {best[3]} -> "
      f"effective {best[4]:.3f}")
print(f"           s = {np.round(best[5], 5)}")
print("  coherence matrix:")
for i, l in enumerate(LABEL):
    print(f"    {l:>10} " + " ".join(f"{best[6][i,j]:.3f}" for j in range(6)))

print("\n=== 6h. HOW MUCH DOES THE ANSWER MOVE WITH THOSE FREE CHOICES? ===")
vals = []
print(f"  {'azimuth deg':>12} {'contour':>8} {'gain dB':>8} {'effective':>10}")
for az in (0.001, 0.01, 0.1, 1.0, 6.0, 20.0):
    for cd in (0.01, 0.1):
        b = dict(BASE, azimuth_error_degrees=az, contour_depth=cd, gain_db=1.0)
        p, s, C = eff(b)
        vals.append(p)
        print(f"  {az:>12} {cd:>8} {1.0:>8} {p:>10.3f}")
print(f"  RANGE over the free choices: {min(vals):.2f} .. {max(vals):.2f}")

print("\n=== 6i. Is 'gap vs azimuth = 1.000' physics or a small-argument limit? ===")
print("  Both are sinc(k/lambda).  Their PERTURBATION signatures are")
print("  d/dk log|sinc(k f / v)|, which for small k f / v is -(pi k f/v)^2/3")
print("  for BOTH -- i.e. proportional to f^2 for both, hence collinear.")
print("  The collinearity is a property of operating far from the sinc null,")
print("  not of the two mechanisms being the same effect.")
print(f"  {'azimuth deg':>12} {'across/lambda@8MHz':>19} {'|coh(gap,azimuth)|':>19}")
for az in (0.01, 0.1, 1.0, 3.0, 6.0, 12.0, 25.0, 45.0, 70.0):
    b = dict(BASE, azimuth_error_degrees=az, contour_depth=0.05, gain_db=1.0)
    p, s, C = eff(b)
    across = WIDTH * np.tan(np.radians(az))
    lam = SPEED / 8.0e6
    print(f"  {az:>12} {across/lam:>19.4f} {C[1,3]:>19.6f}")
print("  So the 'exactly collinear' finding holds only while the azimuth")
print("  loss is negligible; at a real azimuth error the two separate.")

print("\n=== 6j. Removing the mean, on the TAPE set ===")
b = dict(BASE, azimuth_error_degrees=0.1, contour_depth=0.05, gain_db=1.0)
for dm in (False, True):
    p, s, C = eff(b, demean=dm)
    print(f"  mean removed = {str(dm):5s}: effective {p:.3f}  s = {np.round(s,5)}")
print("  (interference.distinguishable REMOVES the mean; the tape figure")
print("   quoted beside it does not, so the two are not the same measure.)")

print("\n=== 6k. Band dependence, from TYPICAL ===")
for lo, hi in ((0.5, 8.0), (0.5, 4.0), (0.1, 8.0), (0.5, 40.0), (0.02, 100.0)):
    p, s, C = eff(b, band=(lo*1e6, hi*1e6, 512))
    print(f"  {lo:>6.2f} - {hi:<7.2f} MHz: effective {p:.3f}")
