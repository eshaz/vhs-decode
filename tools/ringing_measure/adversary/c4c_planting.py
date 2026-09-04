"""Claim 4, the decisive part: what the document's OWN planting helper
actually plants.

tests/unit/test_information_extrapolation.py::_ensemble

    for order, vector in enumerate(shared):
        weight = 1.0 if order == 0 else (
            (1.0 if index % 2 else -1.0) if order == 1
            else (index - (count - 1) / 2.0) / (count / 2.0))
        value = value + weight * vector

Orders 0 and 1 get distinct weight patterns across the components
(all-ones, and alternating).  EVERY order from 2 upwards gets the SAME
linear ramp.  So departures 2, 3, 4, ... enter the ensemble only through
their SUM, as a single direction.
"""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie


def _ensemble(rng, length, count, departures, noise_scale):
    """Copied verbatim from the repository's test file."""
    shared = [rng.standard_normal(length) + 1j * rng.standard_normal(length)
              for _ in range(departures)]
    out = {}
    for index in range(count):
        value = noise_scale * (rng.standard_normal(length)
                               + 1j * rng.standard_normal(length))
        for order, vector in enumerate(shared):
            weight = 1.0 if order == 0 else (
                (1.0 if index % 2 else -1.0) if order == 1
                else (index - (count - 1) / 2.0) / (count / 2.0))
            value = value + weight * vector
        out[f"c{index}"] = {"frequency": value}
    return out, shared


print("=== 4g. The rank of the planted structure, by construction ===")
print("  The weight MATRIX W (count x departures) decides how many")
print("  independent place-space directions the ensemble can contain.")
for count in (6, 10, 16):
    for dep in (1, 2, 3, 4, 5, 8):
        W = np.zeros((count, dep))
        for index in range(count):
            for order in range(dep):
                W[index, order] = (1.0 if order == 0 else
                                   ((1.0 if index % 2 else -1.0) if order == 1
                                    else (index - (count - 1) / 2.0)
                                    / (count / 2.0)))
        r = np.linalg.matrix_rank(W, tol=1e-9)
        print(f"  count {count:>3}, planted {dep}: rank of the weight matrix "
              f"= {r}   -> {r} independent direction(s), not {dep}")

print("""
  So for THREE OR MORE planted departures the ensemble contains exactly
  three independent shared directions no matter how many are 'planted'.
  The document's section 4 table reports 'recovered 4.0' for 4 planted
  and '5.0' for 5 planted.  Those cannot be four and five independent
  planted directions; at most three exist.
""")

print("=== 4h. Confirmed on the actual vectors ===")
rng = np.random.default_rng(91)
for count, dep in ((16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 8)):
    res, shared = _ensemble(rng, 512, count, dep, 1.0)
    stack = np.array([res[f"c{i}"]["frequency"] for i in range(count)])
    S = np.array(shared)
    # project the ensemble onto the planted span and measure its rank there
    Q, _ = np.linalg.qr(S.conj().T)
    coeff = stack @ Q                       # count x dep
    sv = np.linalg.svd(coeff, compute_uv=False)
    eff = int(np.count_nonzero(sv > sv[0] * 1e-6))
    out = ie.differentiate_to_floor(dict(res), "frequency", maximum_passes=12)
    print(f"  count {count}, planted {dep}: singular values of the ensemble's "
          f"coefficients on the planted span = {np.round(sv, 3)}")
    print(f"      -> {eff} genuine direction(s); loop reports "
          f"{len(out['targets'])} recovered in {out['passes']} pass(es), "
          f"{out['reason'][:38]}")

print("\n=== 4i. With departures planted INDEPENDENTLY (random weights) ===")
print("  the same experiment, with each component given its own random")
print("  coefficient on every departure -- k genuinely independent")
print("  directions.")


def honest(rng, length, count, dep, noise=1.0):
    shared = [rng.standard_normal(length) + 1j * rng.standard_normal(length)
              for _ in range(dep)]
    out = {}
    for i in range(count):
        v = noise * (rng.standard_normal(length)
                     + 1j * rng.standard_normal(length))
        for s in shared:
            v = v + (rng.standard_normal() + 1j * rng.standard_normal()) * s
        out[f"c{i}"] = {"frequency": v}
    return out


print(f"  {'count':>6} {'planted':>8} {'passes':>7} {'recovered':>10} "
      f"{'circle':>7}")
for count in (6, 10, 16):
    for dep in (0, 1, 2, 3, 4, 5):
        if dep >= count:
            continue
        p, r, c = [], [], []
        for t in range(25):
            res = honest(np.random.default_rng(1000 + t), 512, count, dep)
            o = ie.differentiate_to_floor(res, "frequency", maximum_passes=12)
            p.append(o["passes"]); r.append(len(o["targets"]))
            c.append(o["at_floor"])
        print(f"  {count:>6} {dep:>8} {np.mean(p):>7.1f} {np.mean(r):>10.1f} "
              f"{100*np.mean(c):>6.0f}%")
