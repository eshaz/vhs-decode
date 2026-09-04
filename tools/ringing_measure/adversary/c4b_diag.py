"""Why does a STRONGER planted departure produce MORE 'recovered'
directions?  Instrument one run."""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(5)
N, L, k = 16, 512, 2


def build(strength, seed):
    r = np.random.default_rng(seed)
    deps = []
    for _ in range(k):
        d = r.standard_normal(L)
        for e in deps:
            d -= (e @ d) * e
        deps.append(d / np.linalg.norm(d))
    comps = {}
    for i in range(N):
        v = r.standard_normal(L) / np.sqrt(L)
        for d in deps:
            v = v + strength * r.standard_normal() * d
        comps[f"c{i}"] = {"x": v}
    return comps, deps


for strength in (0.3, 1.0, 3.0, 10.0):
    comps, deps = build(strength, 11)
    working = {n: dict(a) for n, a in comps.items()}
    print(f"\n--- strength {strength} ---")
    out = ie.differentiate_to_floor(comps, "x", maximum_passes=12)
    for i, t in enumerate(out["trace"]):
        print(f"  pass {i}: rank {t['rank']:.0f} asym {t['asymmetry']:.5f} "
              f"floor {t['sphere_floor']:.5f} sigma {t['sigma']:+.1f}")
    print(f"  -> {out['reason']}, {len(out['targets'])} directions, "
          f"{out['passes']} passes")
    # what the first fit saw
    f0 = ie.ellipsoid(comps, "x")
    print(f"  pass-0 eigenvalues: " + " ".join(f"{x:.3f}" for x in
                                               f0["eigenvalues"][:6]))
    print(f"  mp_edge {f0['mp_edge']:.4f}  significant {f0['significant']}  "
          f"bulk {f0['bulk']:.5f}")
    w = np.clip((f0["eigenvalues"][:f0["significant"]] - f0["bulk"])
                / f0["eigenvalues"][:f0["significant"]], 0, 1)
    print(f"  Wiener weights: {np.round(w, 5)}")
    # after one pass, what is left of the planted directions?
    res = out["residuals"]
    for j, d in enumerate(deps):
        share = np.mean([abs(d @ np.asarray(res[n]["x"]))
                         / np.linalg.norm(res[n]["x"]) for n in comps])
        share0 = np.mean([abs(d @ np.asarray(comps[n]["x"]))
                          / np.linalg.norm(comps[n]["x"]) for n in comps])
        print(f"  planted dir {j}: |cos| before {share0:.4f} after {share:.4f}")
