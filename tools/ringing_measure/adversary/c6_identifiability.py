"""Claim 6: 1.58 of 6 (tape), 6.89 of 12 (interference), 3.79 of 6
(propagation).  How much is physics and how much is parameterisation?"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import head_model as hm
from vhsdecode.models import interference as intf

# The document's procedure, reimplemented independently:
#   perturb each mechanism's own parameter by 5% about TYPICAL,
#   take the difference in log_response, unit-normalise, SVD,
#   report the participation ratio of the squared singular values.

TYPICAL = {k: v for k, v in getattr(hm, "TYPICAL", {}).items()} \
    if hasattr(hm, "TYPICAL") else None
print("hm.TYPICAL present:", TYPICAL is not None)
if TYPICAL is None:
    for name in dir(hm):
        if name.isupper():
            print("  ", name, "=", repr(getattr(hm, name))[:110])

MECH = {
    "spacing":   ("spacing_m", None),
    "gap":       ("gap_m", None),
    "thickness": ("thickness_m", None),
    "azimuth":   ("azimuth_error_degrees", None),
    "contour":   ("contour_depth", None),
    "gain":      ("gain_db", None),
}


def participation(stack, demean=True):
    stack = np.asarray(stack, dtype=np.float64)
    if demean:
        stack = stack - stack.mean(axis=1, keepdims=True)
    n = np.linalg.norm(stack, axis=1, keepdims=True)
    stack = stack / np.where(n > 0, n, 1)
    s = np.linalg.svd(stack, compute_uv=False)
    share = s ** 2 / (s ** 2).sum()
    return 1.0 / np.sum(share ** 2), s


def signatures(base, speed, width, band, delta, demean=True, log=True):
    lo, hi, npts = band
    f = np.linspace(lo, hi, npts)
    ref = hm.log_response(f, speed, width, **base)
    rows, names = [], []
    for label, (key, _) in MECH.items():
        p = dict(base)
        cur = float(p.get(key, 0.0) or 0.0)
        if cur == 0.0:
            cur = {"gain_db": 1.0, "contour_depth": 0.05}.get(key, 1e-7)
            p[key] = cur
            ref_local = hm.log_response(f, speed, width, **dict(base, **{key: cur}))
        p[key] = cur * (1.0 + delta)
        sig = hm.log_response(f, speed, width, **p) - ref
        if not np.all(np.isfinite(sig)) or np.linalg.norm(sig) == 0:
            p2 = dict(base); p2[key] = cur
            sig = hm.log_response(f, speed, width, **dict(p2, **{key: cur * (1 + delta)})) \
                - hm.log_response(f, speed, width, **p2)
        rows.append(np.nan_to_num(sig))
        names.append(label)
    return names, np.array(rows), f


BASE = dict(spacing_m=0.1e-6, gap_m=0.3e-6, thickness_m=0.5e-6,
            azimuth_error_degrees=6.0, contour_depth=0.05,
            contour_length_m=200e-6, gain_db=1.0)
SPEED, WIDTH = 5.8, 58e-6

print("\n=== 6a. Baseline reproduction (0.5-8.0 MHz, 5% perturbation) ===")
names, S, f = signatures(BASE, SPEED, WIDTH, (0.5e6, 8.0e6, 512), 0.05)
p, s = participation(S)
print("  singular values:", np.round(s, 4))
print(f"  EFFECTIVE DISTINGUISHABLE: {p:.2f} of 6   "
      f"(published 1.58; s = 2.159 1.000 0.565 0.133 0.0078 0.0002)")
print(f"  condition number {s[0]/s[-1]:.3e}")

print("\n=== 6b. SENSITIVITY TO THE PERTURBATION SIZE ===")
print("  The 5% is arbitrary.  If the answer is physics it must not move.")
print(f"  {'delta':>10} {'effective':>10} {'condition':>12}")
for d in (1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
    _, S, _ = signatures(BASE, SPEED, WIDTH, (0.5e6, 8.0e6, 512), d)
    p, s = participation(S)
    print(f"  {d:>10.6f} {p:>10.3f} {s[0]/max(s[-1],1e-30):>12.3e}")

print("\n=== 6c. SENSITIVITY TO THE BAND ===")
print(f"  {'band (MHz)':>16} {'effective':>10}")
for lo, hi in ((0.5, 8.0), (0.1, 8.0), (0.5, 4.0), (1.0, 5.0), (2.0, 6.0),
               (0.05, 20.0), (0.5, 40.0), (3.0, 4.0), (0.01, 100.0)):
    _, S, _ = signatures(BASE, SPEED, WIDTH, (lo*1e6, hi*1e6, 512), 0.05)
    p, s = participation(S)
    print(f"  {lo:>6.2f} - {hi:<7.2f} {p:>10.3f}")

print("\n=== 6d. SENSITIVITY TO REMOVING THE MEAN ===")
for demean in (True, False):
    _, S, _ = signatures(BASE, SPEED, WIDTH, (0.5e6, 8.0e6, 512), 0.05)
    p, s = participation(S, demean=demean)
    print(f"  mean removed = {str(demean):5s}: effective {p:.3f}, "
          f"singular {np.round(s,4)}")

print("\n=== 6e. SENSITIVITY TO THE OPERATING POINT (the TYPICAL values) ===")
print(f"  {'variation':>28} {'effective':>10}")
rng = np.random.default_rng(4)
vals = []
for _ in range(40):
    b = {k: (v * float(np.exp(rng.normal(0, 0.7))) if k != "gain_db" else v)
         for k, v in BASE.items()}
    try:
        _, S, _ = signatures(b, SPEED, WIDTH, (0.5e6, 8.0e6, 512), 0.05)
        p, _ = participation(S)
        if np.isfinite(p):
            vals.append(p)
    except Exception:
        pass
print(f"  {'log-normal x2 spread, n=%d' % len(vals):>28} "
      f"{np.min(vals):.2f} .. {np.max(vals):.2f}  (median {np.median(vals):.2f})")

print("\n=== 6f. SENSITIVITY TO THE GRID DENSITY / SAMPLING ===")
for npts in (16, 32, 64, 128, 512, 4096):
    _, S, _ = signatures(BASE, SPEED, WIDTH, (0.5e6, 8.0e6, npts), 0.05)
    p, _ = participation(S)
    print(f"  {npts:>6} points: effective {p:.3f}")
for scale in ("linear", "log"):
    lo, hi, n = 0.5e6, 8.0e6, 512
    f = np.linspace(lo, hi, n) if scale == "linear" else np.geomspace(lo, hi, n)
    ref = hm.log_response(f, SPEED, WIDTH, **BASE)
    rows = []
    for label, (key, _) in MECH.items():
        p2 = dict(BASE); p2[key] = float(BASE[key]) * 1.05
        rows.append(np.nan_to_num(hm.log_response(f, SPEED, WIDTH, **p2) - ref))
    pr, _ = participation(np.array(rows))
    print(f"  {scale:>6} frequency spacing: effective {pr:.3f}")

print("\n=== 6g. The interference set: 6.89 of 12 ===")
for lo, hi in ((0.5, 8.0), (0.1, 8.0), (1.0, 5.0), (0.5, 40.0), (3.0, 4.4)):
    f = np.linspace(lo * 1e6, hi * 1e6, 1024)
    out = intf.distinguishable(f)
    print(f"  band {lo:>5.1f}-{hi:<5.1f} MHz: effective "
          f"{out['effective']:.3f} of {out['count']}  condition "
          f"{out['condition']:.2f}")
print("  NOTE: signatures() picks its four echo delays as k/(span) for")
print("  k = 1,2,4,8 -- i.e. DEFINED to be near-orthogonal over the band.")
f = np.linspace(0.5e6, 8.0e6, 1024)
for delays in ([1e-6, 1.05e-6, 1.1e-6, 1.15e-6],
               [1e-7, 2e-7, 4e-7, 8e-7],
               None):
    out = intf.distinguishable(f, delays_s=delays)
    lab = "default k/span" if delays is None else str(np.round(np.array(delays)*1e6, 2))
    print(f"  echo delays {lab:>28}: effective {out['effective']:.3f} of "
          f"{out['count']}")
