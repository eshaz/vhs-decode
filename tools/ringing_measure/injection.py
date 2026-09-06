"""The injection test, runnable: sweep a known component through the chain.

Ethan: *"Injection test: known signal, known amplitude, never fitted. Sweep
level, find recovery threshold. Gives a sensitivity curve and identifies
which stage eats what."*

Run:
    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/injection.py \\
        --out /some/directory [--tapes cd home pnb] [--heads a] \\
        [--capture /testdata/.../zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac] \\
        [--levels 0.125 ... 64] [--draws 6] [--amplitude-se 16] [--no-plots]

For every background it can find - each tape's sync-step export (the grid),
a short window of the RF capture demodulated through the decoder's own
chain (the series), and the tesseract over two tapes (the cube) - it plants
each kind at a sweep of amplitudes in units of the measured per-bin noise,
runs the chain's estimators unchanged, and prints:

    * per kind, per reader: the detect and recover thresholds in SE units
      and in the kind's natural amplitude, and the slope of the response
      well above threshold (the fraction the reader returns);
    * the matrix of kinds against stages at one amplitude well above the
      noise: the share of the injection each stage's entries return, the
      nuisance's and the residual's shares, which join's prior would claim
      the residual and which base-key direction `outlier_response` names;
    * the worst stage, and why.

Everything measured is also written to `<out>/injection_<label>.npz` and
the sensitivity curves to `<out>/sensitivity_<label>_<kind>.png`.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vhsdecode.models import injection as inj  # noqa: E402

SHARED = "/tmp/claude-1000/-workspaces-vhs-decode/shared"
CAPTURE = ("/testdata/test_patterns/vhs/playback/"
           "zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac")
# The RF capture's sample rate is not in its FLAC header (which can only say
# 50 kHz); the file name carries it, and the decoder is run with -f 50 on it.
CAPTURE_RATE_HZ = 50e6


def export_path(tape):
    return os.path.join(SHARED, f"{tape}_tree_noeq_sync_step_response.npz")


def fmt(value, width=8, digits=3):
    if value is None:
        return " " * (width - 1) + "-"
    if isinstance(value, float) and not np.isfinite(value):
        return " " * (width - 3) + "nan"
    return f"{value:{width}.{digits}g}"


def natural_units(kind, representation):
    return {
        "echo": "reflection", "ring": "IRE peak" if representation == "series"
        else "peak ripple", "level step": "IRE" if representation == "series"
        else "nepers", "clip": "IRE above the extreme",
        "frequency scaling": "fraction",
        "phase rotation": "radians",
    }[kind]


def print_sweep(result, readers=None):
    """The thresholds of one sweep, one line per reader."""
    kind, rep = result["kind"], result["representation"]
    planted = result["planted"]
    print(f"\n  {kind.upper()} on {result['label']} ({rep}): levels "
          f"{', '.join(f'{s:g}' for s in result['amplitudes_se'])} SE = "
          f"{result['amplitudes'][0]:.4g} .. {result['amplitudes'][-1]:.4g} "
          f"{natural_units(kind, rep)}; matched-filter SNR at the top "
          f"{planted[-1]['snr']:.0f}, peak {planted[-1]['peak_se']:.1f} SE")
    print(f"  {'reader':<58} {'detect SE':>9} {'recover SE':>10} "
          f"{'slope':>7}  {'stage' if rep == 'series' else 'identity'}")
    for name, curve in result["readers"].items():
        if readers is not None and not any(name.startswith(r) for r in readers):
            continue
        tag = (curve["stage"] if rep == "series"
               else ("identity" if curve["identity"] else
                     "flag" if curve["is_flag"] else "-"))
        print(f"  {name:<58} {fmt(curve['threshold_detect_se'], 9, 3)} "
              f"{fmt(curve['threshold_recover_se'], 10, 3)} "
              f"{fmt(curve['slope'], 7, 3)}  {tag}")


def print_grid_matrix(matrix):
    """Kinds against stages, per key: the share of the injection each stage
    returns, and where the rest goes."""
    stages = matrix["stages"]
    short = {"the source": "source", "the transmission path": "transmission",
             "the recording machine": "recorder", "the tape": "tape",
             "the playback machine": "playback", "the capture": "capture",
             "the picture stage": "picture", "undeclared": "undecl.",
             inj.NUISANCE_STAGE: "nuisance", inj.RESIDUAL_STAGE: "residual"}
    for label in matrix["keys"]:
        print(f"\n  WHICH STAGE EATS WHAT on {matrix['label']}, key '{label}', "
              f"at {matrix['amplitude_se']:g} SE - share of the injected amplitude")
        print("  " + f"{'kind':<18}" + "".join(f"{short[s]:>13}" for s in stages)
              + f"{'chi2 rise':>11}  claimed at (gain), family")
        for kind, row in matrix["rows"].items():
            entry = row[label]
            shares = "".join(f"{entry['shares'][s]:>13.3f}" for s in stages)
            claim = (f"{entry['claimed_at']} ({entry['claimed_gain']:+.3f}), "
                     f"{entry['family_winner']}"
                     f"{' identified' if entry['family_identified'] else ''}")
            print(f"  {kind:<18}{shares}{entry['chi_square_rise']:>11.1f}  {claim}")
        print("  the outliers' carrier after injection: " + ", ".join(
            f"{kind}: {row[label]['carried_by']}" for kind, row in matrix["rows"].items()))


def print_series_matrix(matrix):
    print(f"\n  CROSS-TALK on {matrix['label']} (series) at {matrix['amplitude_se']:g} SE"
          f" - each reader's response in its own error bars (z), and its flag")
    readers = matrix["readers"]
    print("  " + f"{'kind':<18}" + "".join(f"{r[:12]:>13}" for r in readers))
    for kind, row in matrix["rows"].items():
        cells = "".join(
            (f"{row['readers'][r]['z']:>9.1f}" if np.isfinite(row['readers'][r]['z'])
             else f"{'-':>9}")
            + f"{'*' if row['readers'][r]['flag'] > 0.5 else ' ':>4}"
            for r in readers)
        print(f"  {kind:<18}{cells}")
    print("  stages: " + ", ".join(f"{r}: {matrix['rows'][matrix['kinds'][0]]['readers'][r]['stage']}"
                                   for r in readers))
    print("  * = the estimator's flag is set after injection; baseline flags: "
          + ", ".join(f"{r} {matrix['rows'][matrix['kinds'][0]]['readers'][r]['baseline_flag']:.2f}"
                      for r in readers))


def print_cube_matrix(matrix):
    print(f"\n  THE TESSERACT on {matrix['label']} at {matrix['amplitude_se']:g} SE, "
          f"planted with the sign pattern of the contrast named")
    print(f"  {'kind':<18}{'contrast':>10}{'recovered':>11}{'worst cross-talk':>36}  stage charged")
    for kind, row in matrix["rows"].items():
        print(f"  {kind:<18}{':'.join(row['subset']):>10}{row['recovered']:>11.4f}"
              f"{str(row['worst_cross_talk']).replace('contrast[', '')[:26]:>28}"
              f" {row['worst_cross_talk_share']:+.1e}  {row['stage']}")


def worst_stage(grid_matrices):
    """The stage that eats the most of what is not its own: over every
    kind, the largest share returned by a stage that is not where the kind
    belongs - the transmission path for an echo, nothing in the frequency
    key for a ring, the nuisance for a level, the nuisance for a scaling,
    nothing for a phase rotation."""
    belongs = {"echo": "the transmission path", "ring": None,
               "level step": inj.NUISANCE_STAGE,
               "frequency scaling": inj.NUISANCE_STAGE, "phase rotation": None}
    worst = None
    for matrix in grid_matrices:
        for label in matrix["keys"]:
            for kind, row in matrix["rows"].items():
                for stage, share in row[label]["shares"].items():
                    if stage in (belongs.get(kind), inj.RESIDUAL_STAGE):
                        continue
                    if worst is None or abs(share) > abs(worst[0]):
                        worst = (share, stage, kind, label, matrix["label"])
    return worst


def plot_sweep(result, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    levels = result["amplitudes_se"]
    identity = {n: c for n, c in result["readers"].items() if c["identity"]}
    others = {n: c for n, c in result["readers"].items()
              if not c["identity"] and not c["is_flag"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for name, curve in identity.items():
        expected = curve["expected"]
        ok = np.abs(expected) > 0
        ax.errorbar(levels[ok], curve["response"][ok] / expected[ok],
                    yerr=curve["error"][ok] / np.abs(expected[ok]),
                    marker="o", ms=3, capsize=2, label=name[:48])
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="identity")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("injected amplitude (units of the per-bin noise SE)")
    ax.set_ylabel("recovered / injected")
    ax.set_title(f"{result['kind']} on {result['label']}: recovery")
    if identity:
        ax.legend(fontsize=6, loc="lower right")
    ax = axes[1]
    for name, curve in others.items():
        z = curve["response"] / np.where(curve["error"] > 0, curve["error"], np.nan)
        ax.plot(levels, z, marker=".", ms=3, lw=0.8, label=name[:48])
    ax.axhspan(-inj.DETECT_SIGMA, inj.DETECT_SIGMA, color="0.85", label="not detected")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_xlabel("injected amplitude (SE units)")
    ax.set_ylabel("response / error bar")
    ax.set_title("every other reader: detection")
    ax.legend(fontsize=5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save(out, label, sweeps, matrix, background_summary):
    """Everything measured, as one npz of arrays plus a json of the rest."""
    arrays, meta = {}, {"label": label, "background": background_summary,
                        "matrix": to_json(matrix), "sweeps": {}}
    for kind, result in sweeps.items():
        key = kind.replace(" ", "_")
        arrays[f"{key}/amplitudes_se"] = result["amplitudes_se"]
        arrays[f"{key}/amplitudes"] = result["amplitudes"]
        meta["sweeps"][kind] = {"readers": {}, "parameters": to_json(result["parameters"]),
                                "planted": to_json(result["planted"])}
        for name, curve in result["readers"].items():
            tag = name.replace(" ", "_").replace("/", "_")
            arrays[f"{key}/{tag}/response"] = curve["response"]
            arrays[f"{key}/{tag}/error"] = curve["error"]
            if curve["expected"] is not None:
                arrays[f"{key}/{tag}/expected"] = curve["expected"]
            meta["sweeps"][kind]["readers"][name] = {
                k: to_json(v) for k, v in curve.items()
                if k not in ("response", "error", "expected", "flag")}
    safe = label.replace(" ", "_")
    np.savez(os.path.join(out, f"injection_{safe}.npz"), **arrays)
    with open(os.path.join(out, f"injection_{safe}.json"), "w") as handle:
        json.dump(meta, handle, indent=1)


def to_json(value):
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_grid(tape, head, args):
    path = export_path(tape)
    if not os.path.exists(path):
        print(f"{tape}: export missing, skipped")
        return None
    background = inj.load_grid(path, head)
    started = time.time()
    inj.grid_keys(background)
    print(f"\n{'=' * 78}\nGRID: {background['label']}, {background['frequency_hz'].size} "
          f"bins, keys built in {time.time() - started:.0f} s; per-bin noise median "
          f"{np.median(background['sigma']):.4f}, 95th {np.percentile(background['sigma'], 95):.4f}")
    grid = background["frequency_hz"]
    parameters = {"echo": {"delay_s": inj.key_delay_s(grid, 2)}}
    sweeps = {}
    for kind in args.kinds:
        if kind not in inj.GRID_KINDS:
            continue
        started = time.time()
        sweeps[kind] = inj.sweep(kind, args.levels, background, draws=args.draws,
                                 **parameters.get(kind, {}))
        print_sweep(sweeps[kind], readers=("stage[", "entry[", "floor[", "join[", "correction["))
        print(f"  ({time.time() - started:.0f} s)")
        if not args.no_plots:
            plot_sweep(sweeps[kind], os.path.join(
                args.out, f"sensitivity_{background['label'].replace(' ', '_')}_{kind.replace(' ', '_')}.png"))
    # the matrix, with the echo BETWEEN the key's entries as a further row
    kinds = [k for k in args.kinds if k in inj.GRID_KINDS]
    matrix = inj.which_stage_eats_what(background, kinds, args.amplitude_se,
                                       parameters=parameters)
    between = inj.which_stage_eats_what(
        background, ["echo"], args.amplitude_se,
        parameters={"echo": {"delay_s": inj.default_delay_s(grid)}})
    matrix["rows"]["echo (between entries)"] = between["rows"]["echo"]
    matrix["kinds"] = list(matrix["rows"])
    print_grid_matrix(matrix)
    summary = {"bins": int(grid.size), "sigma_median": float(np.median(background["sigma"])),
               "band_hz": [float(grid.min()), float(grid.max())]}
    save(args.out, background["label"], sweeps, matrix, summary)
    return matrix


def run_series(args):
    if not os.path.exists(args.capture):
        print(f"\ncapture {args.capture} missing, the series is skipped")
        return None
    import soundfile
    from vhsdecode.formats import get_format_params, parse_tape_speed
    sysparams, rfparams = get_format_params(
        args.system, args.format, parse_tape_speed(args.speed),
        logging.getLogger(__name__))
    rate = CAPTURE_RATE_HZ
    frames = int(args.seconds * rate)
    samples = soundfile.read(args.capture, start=int(args.start_seconds * rate),
                             frames=frames, dtype="int16")[0]
    started = time.time()
    ire, out_rate = inj.demodulate_luma(samples, rate, sysparams, rfparams)
    lines, fall = inj.sync_lines(ire, out_rate, sysparams)
    label = os.path.basename(args.capture).split("-NTSC")[0].replace("zaroff-", "")
    background = inj.series_background(lines, out_rate, fall, sysparams, label)
    print(f"\n{'=' * 78}\nSERIES: {label}, {lines.shape[0]} horizontal sync lines of "
          f"{lines.shape[1]} samples from {args.seconds:g} s of the capture, demodulated "
          f"in {time.time() - started:.0f} s; tip {background['tip_level']:.2f} IRE, porch "
          f"{background['porch_level']:.2f}, width {background['width_us']:.3f} us, "
          f"single-line noise {background['sigma_line']:.2f} IRE, group-mean noise "
          f"{background['sigma_mean']:.3f}; tail opens {3 * sysparams['syncTransitionUS']:.2f} us "
          f"after the rise, slope settles {background['settle_time_us'] - background['rise_time_us']:.2f} us after it")
    sweeps = {}
    for kind in args.kinds:
        started = time.time()
        sweeps[kind] = inj.sweep(kind, args.levels, background)
        print_sweep(sweeps[kind])
        print(f"  ({time.time() - started:.0f} s)")
        if not args.no_plots:
            plot_sweep(sweeps[kind], os.path.join(
                args.out, f"sensitivity_{label}_{kind.replace(' ', '_')}.png"))
    matrix = inj.which_stage_eats_what(background, list(args.kinds), args.amplitude_se)
    print_series_matrix(matrix)
    summary = {"lines": int(lines.shape[0]), "tip_level": background["tip_level"],
               "porch_level": background["porch_level"],
               "sigma_line": background["sigma_line"], "sigma_mean": background["sigma_mean"],
               "width_us": background["width_us"]}
    save(args.out, label, sweeps, matrix, summary)
    return matrix


def run_cube(args):
    paths = {t: export_path(t) for t in args.tapes if os.path.exists(export_path(t))}
    if len(paths) < 2:
        print("\nfewer than two exports, the cube is skipped")
        return None
    tapes = tuple(list(paths)[:2])
    background = inj.cube_background(paths, tapes)
    print(f"\n{'=' * 78}\nCUBE: {background['label']}, axes {background['cube'].axes}, "
          f"{background['frequency_hz'].size} bins")
    sweeps = {}
    kinds = [k for k in args.kinds if k in inj.CUBE_KINDS]
    for kind in kinds:
        sweeps[kind] = inj.sweep(kind, args.levels, background, draws=args.draws)
        print_sweep(sweeps[kind], readers=("contrast[head]", "unreached"))
        if not args.no_plots:
            plot_sweep(sweeps[kind], os.path.join(
                args.out, f"sensitivity_cube_{kind.replace(' ', '_')}.png"))
    matrix = inj.which_stage_eats_what(background, kinds, args.amplitude_se, draws=args.draws)
    print_cube_matrix(matrix)
    save(args.out, "cube_" + "_".join(tapes), sweeps, matrix,
         {"tapes": list(tapes), "bins": int(background["frequency_hz"].size)})
    return matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description="the injection test")
    parser.add_argument("--out", required=True, help="where the npz, json and plots go")
    parser.add_argument("--tapes", nargs="+", default=["cd", "home", "pnb"])
    parser.add_argument("--heads", nargs="+", default=["a"])
    parser.add_argument("--capture", default=CAPTURE)
    parser.add_argument("--start-seconds", type=float, default=0.2)
    parser.add_argument("--seconds", type=float, default=0.05)
    parser.add_argument("--system", default="NTSC")
    parser.add_argument("--format", default="VHS")
    parser.add_argument("--speed", default="SP")
    parser.add_argument("--levels", type=float, nargs="+", default=None,
                        help="amplitudes in SE units (default an eighth to sixty-four)")
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--amplitude-se", type=float, default=16.0)
    parser.add_argument("--kinds", nargs="+", default=list(inj.KINDS))
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-series", action="store_true")
    parser.add_argument("--skip-cube", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")
    os.makedirs(args.out, exist_ok=True)
    args.levels = (np.asarray(args.levels, dtype=np.float64) if args.levels
                   else inj.default_levels())
    print(__doc__.split("Run:")[0].rstrip())
    grids = []
    if not args.skip_grid:
        for tape in args.tapes:
            for head in args.heads:
                got = run_grid(tape, head, args)
                if got is not None:
                    grids.append(got)
    if not args.skip_series:
        run_series(args)
    if not args.skip_cube:
        run_cube(args)
    if grids:
        worst = worst_stage(grids)
        if worst is not None:
            share, stage, kind, label, where = worst
            print(f"\n{'=' * 78}\nTHE WORST STAGE: '{stage}' returns {share:+.3f} of an "
                  f"injected {kind} on {where} under the '{label}' key - the largest "
                  "share any stage takes of a component that is not its own.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
