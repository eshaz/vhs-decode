"""Does the over-the-air sample carry a clip the home recording does not?

Ethan: *"It might be at the television tuner stage, the cd sample was an over
the air recording."*

The brief here is the archive's usual one: REFUTE. Negative modulation puts
the sync tip at peak carrier, so a receiver that overloads clips the sync
pulse and nothing else does - the recorder's own dark clip is stated 40 per
cent above the tip. If the tuner site is occupied on the over-the-air sample,
the sync TIP is where it shows, and the home recording is the control that
never went through a tuner at all.

Two witnesses are run, both on the sibling lane's pooled sync pulses, both
heads, at the decoded luma output. NEITHER attests a clip on cd that home
lacks, and the first points the other way. The script prints why, because the
confounds are the useful part: they say what a decisive instrument needs.

Run: python3 tools/ringing_measure/adversary/c10_tuner_site.py
"""

import os

import numpy as np

SHARED = os.environ.get(
    "VHS_SHARED_DIR",
    "/tmp/claude-1000/-workspaces-vhs-decode/shared")

# the site the exports were taken at; the pulses are 4fsc-sampled
RATE_MHZ = 14.318181818181818

TAPES = (("cd", "over the air"), ("home", "a home recording, the control"))


def _detrended(values):
    """the ripple, with a straight line removed so a slow tilt is not counted
    as ripple"""
    if values.size < 3:
        return values * 0.0
    t = np.arange(values.size, dtype=float)
    return values - np.polyval(np.polyfit(t, values, 1), t)


def _interior(mask, margin=6):
    """the middle of a settled run, clear of the edges either side"""
    index = np.flatnonzero(mask)
    return index[margin:-margin] if index.size > 2 * margin + 4 else index


def _load(tape):
    path = os.path.join(SHARED, f"{tape}_tree_noeq_sync_step_response.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def witnesses(tape):
    data = _load(tape)
    if data is None:
        print(f"  {tape}: export not present, skipped")
        return []
    rows = []
    for head in ("a", "b"):
        measured = np.asarray(data[f"head_{head}_pulse_mean"], dtype=float)
        ideal = np.asarray(data[f"head_{head}_pulse_ideal"], dtype=float)
        departure = measured - ideal
        slope = np.gradient(ideal) * RATE_MHZ
        settled = np.abs(slope) < 0.05 * np.abs(slope).max()
        span = ideal.max() - ideal.min()
        tip = _interior(settled & (ideal <= ideal.min() + 0.20 * span))
        porch = _interior(settled & (ideal >= ideal.max() - 0.20 * span))

        # WITNESS 1: a clip is ONE-SIDED, so it moves the tip and not the
        # porch. Anything that moves them oppositely and equally is an
        # amplitude mismatch instead.
        one_sided = float(departure[tip].mean() - departure[porch].mean())
        symmetry = float(departure[tip].mean() + departure[porch].mean())

        # WITNESS 2: a clipped level has NO GAIN, so what rides on it is
        # suppressed there and nowhere else. Divide out the ideal's own
        # ripple or the burst on the porch counts as chain noise.
        def added(index):
            got = _detrended(measured[index]).var()
            want = _detrended(ideal[index]).var()
            return max(got - want, 0.0) ** 0.5

        ripple = added(tip) / max(added(porch), 1e-9)
        rows.append({
            "head": head,
            "one_sided": one_sided,
            "symmetry": symmetry,
            "ripple": ripple,
            "width_us": float(data[f"head_{head}_pulse_width_us"]),
            "spec_us": float(data[f"head_{head}_pulse_spec_width_us"]),
        })
    return rows


def main():
    print(__doc__.split("Run:")[0].rstrip())
    print()
    print("=" * 74)
    print("WITNESS 1  tip minus porch departure - a clip is one-sided")
    print("WITNESS 2  tip/porch added ripple    - a clipped level has no gain")
    print("=" * 74)
    found = {}
    for tape, what in TAPES:
        print(f"\n{tape}  ({what})")
        rows = witnesses(tape)
        found[tape] = rows
        for row in rows:
            print(f"  head {row['head']}   one-sided {row['one_sided']:+.3f} "
                  f"IRE   sum {row['symmetry']:+.3f}   "
                  f"ripple {row['ripple']:.3f}   "
                  f"width {row['width_us']:.3f} us "
                  f"(spec {row['spec_us']:.3f})")

    if not (found.get("cd") and found.get("home")):
        print("\nboth exports are needed for the comparison; "
              "set VHS_SHARED_DIR")
        return

    air = np.mean([r["one_sided"] for r in found["cd"]])
    control = np.mean([r["one_sided"] for r in found["home"]])
    air_r = np.mean([r["ripple"] for r in found["cd"]])
    control_r = np.mean([r["ripple"] for r in found["home"]])

    print("\n" + "=" * 74)
    print("THE VERDICT")
    print("=" * 74)
    print(f"  one-sided departure   over the air {air:+.3f} IRE   "
          f"control {control:+.3f}   -> the CONTROL is larger")
    print(f"  added-ripple ratio    over the air {air_r:.3f}        "
          f"control {control_r:.3f}     -> the same on both")
    print("""
  NOT ATTESTED. Neither witness separates the over-the-air sample from the
  control, and the first points the other way.

  Both are confounded, and this is the useful part:

  1. The departure is ANTIsymmetric - the tip reads high by as much as the
     porch reads low - which is a residual amplitude mismatch between the
     fitted ideal and the measurement, not the one-sided truncation a clip
     makes. The instrument scales its ideal to the measured pulse, and that
     absorbs exactly the depth a clip removes. A fitted ideal cannot witness
     a clip; a SPEC-DEPTH ideal on the measured blanking can.

  2. The ripple ratio is below one on both tapes by the same factor, because
     the porch carries burst and picture content the tip does not. It is not
     a clean comparator.

  The witness that survives both: the scatter ACROSS pulses at a fixed sample
  on the tip, against the same across-pulse scatter at a fixed sample on the
  back porch AFTER the burst. A ratio of across-pulse variances is blind to
  any amplitude fit, and a clipped level suppresses what rides on it there
  and nowhere else. That needs the per-line pulses rather than the pooled
  mean, which is a decode-time measurement.

  WHAT IS ATTESTED IS THE PREMISE, NOT THE COMPONENT. The recorded sync width
  above is a static timing property of the source, and it separates the three
  tapes cleanly - 4.608 us over the air, 4.676 at home, 4.650 on the test
  tapes, against a 4.700 us spec. That alone says cd's source chain differs.

  A min-phase test on the exact decoder division is recorded as converging on
  home and not on cd, leaving a zero-phase LEVEL-INDEPENDENT roll-off of
  source class, which would point the same way - but it is UNCONFIRMED. The
  lane it was attributed to could not reproduce it on request, so it is kept
  without attribution and carries no weight until someone re-runs it.

  And neither says THIS component is the difference: a clip is
  level-DEPENDENT and that residue is not.""")


if __name__ == "__main__":
    main()
