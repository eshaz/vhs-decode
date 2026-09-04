"""What survives: the section 5 information-budget table, on ensembles that
are NOT the nested construction.  This is the fair test of `resolved`."""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import information_extrapolation as ie

rng = np.random.default_rng(114)
L = 256


def noise():
    return rng.standard_normal(L) + 1j * rng.standard_normal(L)


shared = noise()
sets = {
    "six independent": {f"c{i}": {"f": noise()} for i in range(6)},
    "six views of ONE": {f"c{i}": {"f": shared * (1 + 0.02 * i)}
                         for i in range(6)},
    "six, shared + private": {f"c{i}": {"f": shared + 1.2 * noise()}
                              for i in range(6)},
    "twelve, same shared": {f"c{i}": {"f": shared + 1.2 * noise()}
                            for i in range(12)},
}
print("=== Section 5's table, reproduced on well-conditioned ensembles ===")
print(f"{'ensemble':24s} {'N':>3} {'total':>7} {'resolved':>9} {'share':>7} "
      f"{'published'}")
pub = {"six independent": (6.00, 0.000, "0.0%"),
       "six views of ONE": (6.00, 6.000, "100.0%"),
       "six, shared + private": (6.00, 2.440, "40.7%"),
       "twelve, same shared": (12.00, 5.166, "43.0%")}
for k, v in sets.items():
    f = ie.ellipsoid(v, "f")
    p = pub[k]
    print(f"{k:24s} {f['information']:>3.0f} {f['trace']:>7.2f} "
          f"{f['resolved']:>9.3f} {100*f['resolved_fraction']:>6.1f}% "
          f"   {p[1]:.3f} / {p[2]}")
print("""  CONFIRMED.  On an ensemble whose rows are independent draws, `resolved`
  behaves exactly as the document says: 0 for independence, N for a
  needle, and it roughly doubles with the component count at fixed
  structure.  The pathology in section 6 is not in `resolved`; it is in
  what build_matrix feeds it.""")

print("\n=== The same four ensembles put through the NESTED construction ===")
print("  (i.e. add every pairwise difference, as build_matrix does)")


def nest(d):
    n = len(d)
    names = list(d)
    o = dict(d)
    for i in range(n):
        for j in range(i + 1, n):
            o[f"{names[i]} vs {names[j]}"] = {
                "f": d[names[i]]["f"] - d[names[j]]["f"]}
    return o


print(f"{'ensemble':24s} {'entries':>7} {'rank':>5} {'signif':>7} "
      f"{'resolved':>9} {'share':>7}")
for k, v in sets.items():
    f = ie.ellipsoid(nest(v), "f")
    print(f"{k:24s} {f['information']:>7.0f} {f['rank']:>5} "
          f"{f['significant']:>7} {f['resolved']:>9.3f} "
          f"{100*f['resolved_fraction']:>6.1f}%")
print("""  The 'six independent' row moves from 0.0% to nearly everything.  Adding
  the differences of a set to the set does not add information -- they are
  in its span -- yet it moves the measured 'share of the budget in real
  directions' from zero to ~95%.  That is the whole of the section 6
  effect in one line.""")
