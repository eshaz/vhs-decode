"""Zero, or an unrecoverable residual - and the tesseract that says which.

Run:
    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/residual_floor.py

Reads the sync-pulse exports in the shared directory, applies the whole
frequency-axis key to each tape and head with the first half of the fields
fitted and the second judged, and prints the three readings of
`residual_floor` (the floor, the whiteness, the reproduction) beside the
tesseract's fold on all dimensions for each pair of tapes.

    --per-field <npz> [...]   also fold the TIME axis of a per-field
                              response file (keys frequency_mhz,
                              head_{a,b}_fall_H/_se/_valid, head_{a,b}_
                              first_field, as the per-field instrument
                              writes them) by the bits of the field index.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vhsdecode.models import hypercomplex, residual_floor, tesseract  # noqa: E402

SHARED = "/tmp/claude-1000/-workspaces-vhs-decode/shared"
TAPES = ("cd", "home", "pnb")
OUTPUT_RATE_HZ = 4.0 * 455.0 / 2.0 * 525.0 * 30.0 / 1.001


def export_path(tape):
    return os.path.join(SHARED, f"{tape}_tree_noeq_sync_step_response.npz")


def time_axis(path):
    """The per-field responses of one decode folded by field-index bits."""
    data = np.load(path)
    f = np.asarray(data["frequency_mhz"], dtype=np.float64)
    band = (f > 0.2) & (f < 4.0)
    H, SE, V = {}, {}, {}
    for head in ("a", "b"):
        for i, start in enumerate(data[f"head_{head}_first_field"]):
            H[int(start)] = data[f"head_{head}_fall_H"][i]
            SE[int(start)] = data[f"head_{head}_fall_se"][i]
            V[int(start)] = data[f"head_{head}_fall_valid"][i]
    fields = np.array(sorted(H))
    ok = band.copy()
    for n in fields:
        ok &= V[n] & np.isfinite(H[n]) & np.isfinite(SE[n]) & (SE[n] > 0)
    response = np.array([np.log(np.abs(H[n][ok])) + 1j * np.unwrap(np.angle(H[n][ok]))
                         for n in fields])
    variance = np.array([(SE[n][ok] / np.abs(H[n][ok])) ** 2 for n in fields])
    built = tesseract.from_field_series(response, variance, fields, bins=f[ok] * 1e6)
    cube, scales = built["cube"], built["scales"]
    out = tesseract.unreached(cube)
    print(f"{os.path.basename(path)}: {built['fields_used']} fields folded "
          f"({built['fields_used'] * 1.001 / 60:.2f} s), {built['fields_dropped']} dropped, "
          f"{ok.sum()} bins")
    for b, axis in enumerate(cube.axes):
        v = out["verdicts"][(axis,)]
        print(f"   bit {b}: {2 ** b:4d} fields {scales[axis]['rate_hz']:6.2f} Hz  "
              f"power/noise {v['ratio']:8.2f}  z {v['z']:8.1f}  "
              f"{'identified' if v['identified'] else 'noise'}"
              f"{'  ' + scales[axis]['mechanism'] if 'mechanism' in scales[axis] else ''}")
    print(f"   {out['identified_total']} of {out['contrasts']} contrasts identified; "
          f"top order z {out['top_order_z']:.1f}")


def main():
    warnings.simplefilter("ignore")
    print(__doc__.split("Run:")[0].rstrip())
    print()
    per_field = [a for a in sys.argv[2:]] if len(sys.argv) > 2 and sys.argv[1] == "--per-field" else []
    if per_field:
        print("THE TIME AXIS, folded by the bits of the field index")
        for path in per_field:
            time_axis(path)
        print()
    print("THE RESIDUAL AFTER THE WHOLE FREQUENCY-AXIS KEY (entry-wise admitted)")
    print("tape head   before   in-sample   held-out   lag-1 z   agreement   reading")
    for tape in TAPES:
        path = export_path(tape)
        if not os.path.exists(path):
            print(f"{tape}: export missing, skipped")
            continue
        for head in ("a", "b"):
            m = residual_floor.load_export(path, head)
            keys = residual_floor.keys_for(m["frequency_hz"])
            r = residual_floor.test(m["frequency_hz"], m["H"], m["se"],
                                    keys["admitted"], m.get("held_out"))
            st = r.get("structureless", {})
            print("%-4s %s   %8.0f   %9.0f   %8.0f   %7.1f   %+.3f      %s" % (
                tape, head, r["before"]["reduced_chi_square"],
                r["in_sample"]["reduced_chi_square"],
                r["held_out"]["reduced_chi_square"],
                r["held_out"]["whiteness"]["lag_one_z"],
                st.get("agreement", float("nan")),
                st.get("why", r["held_out"]["reading"])[:48]))
    print()
    print("THE TESSERACT: head x polarity x half x tape, the fold on all dimensions")
    paths = {t: export_path(t) for t in TAPES if os.path.exists(export_path(t))}
    for pair in (("cd", "home"), ("cd", "pnb"), ("home", "pnb")):
        if not all(t in paths for t in pair):
            continue
        cube = tesseract.from_sync_exports(paths, pair)
        out = tesseract.unreached(cube)
        sides = tesseract.asymmetry(cube)
        print(f"tapes {pair[0]} and {pair[1]}: {out['vertices']} vertices, "
              f"{out['contrasts']} contrasts, {out['identified_total']} identified; "
              f"top order z {out['top_order_z']:.1f}")
        print("   sides in floors: " + ", ".join(
            f"{ax} {sides[ax]['side_in_floors']:.1f}" for ax in cube.axes))
        causal = hypercomplex.causality(cube, OUTPUT_RATE_HZ)
        for name in ("head", "polarity", "half", "tape"):
            c = causal[name]
            print(f"   {name:9s} delay {c['delay_s'] * 1e9:+7.1f} ns, all-pass "
                  f"{c['allpass_rms']:.3f} rad, minimum-phase share "
                  f"{c['minimum_phase_share']:+.2f}")
        signal = tesseract.one_real_signal(cube, OUTPUT_RATE_HZ)
        print(f"   one real signal: {len(signal['identified'])} contrasts folded, "
              f"kernel real {signal['kernel_is_real']}, residual "
              f"{signal['residual_over_floor']:.2f}x floor")
        print("   " + out["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
