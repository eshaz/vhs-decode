#!/usr/bin/env python3
"""The cross-instrument transfer: the COMPLEX record-tap to playback-tap
response of one deck, per pattern, per head and per speed, with standard
errors from repeated fields.

    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/tap_transfer.py
        [--speed SP] [--pattern 75bars ...] [--jobs 8] [--out DIR] [--plot]

Per (pattern, speed) pair the instrument locates the fields and lines of
both captures from their sync structure, pairs lines of equal line number,
aligns each pair by whitened cross-correlation, and forms

    H(f) = Sxy(record -> playback) / Sxy(record -> record)

per playback field, per head (interlace parity), with the scatter across
fields as the standard error. See `vhsdecode.models.tap_transfer` for the
estimator and the evidence behind it. This script prints, per band, the
magnitude in dB referenced to the reference band, the excess phase in
degrees, both with SE, the coherence, the record tap's coherent fraction
and the magnitude-only power ratio on the same data; then the separation
across patterns (the content-dependence witness), the attribution against
`rf_stages`, and the comparisons with the 2026-09-02 export and the depth
agent's slopes. Results are written as one npz per pair plus a summary npz.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

from vhsdecode.models import tap_transfer as tt

ROOT = "/testdata/test_patterns/vhs"
DEFAULT_OUT = os.path.join(os.path.dirname(tt.SHARED_EXPORT), "tap_transfer")


def capture_pairs(root, speeds, patterns):
    """Every (pattern, speed) with both taps present."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "record", "*.flac"))):
        name = os.path.basename(path)
        try:
            pattern = name.split("zaroff-")[1].split("-NTSC-")[0]
            speed = name.split("-NTSC-")[1].split("-")[0]
        except IndexError:
            continue
        if speeds and speed not in speeds:
            continue
        if patterns and pattern not in patterns:
            continue
        playback = os.path.join(root, "playback", name.replace("-rf-rec.", "-rf-pb."))
        if os.path.exists(playback):
            out.append((pattern, speed, path, playback))
    return out


def analyse_pair(job):
    """One (pattern, speed): the transfer per head, saved to an npz."""
    pattern, speed, record_path, playback_path, out_dir = job
    target = os.path.join(out_dir, f"tap_transfer_{pattern}_{speed}.npz")
    rate = tt.sample_rate_from_name(record_path)
    params = tt.capture_parameters("NTSC", "VHS", speed)
    record = tt.read_capture(record_path)
    playback = tt.read_capture(playback_path)
    try:
        transfers = tt.complex_transfer(record, playback, rate, params)
    except ValueError as error:
        return pattern, speed, None, str(error)
    payload = {"freqs_hz": next(iter(transfers.values())).freqs,
               "pattern": pattern, "speed": speed, "sample_rate_hz": rate,
               "reference_band_hz": np.array(params["reference_band_hz"])}
    for head, t in transfers.items():
        tag = f"head{head}"
        payload[f"{tag}_transfer"] = t.transfer
        payload[f"{tag}_per_field"] = t.per_field
        payload[f"{tag}_log_magnitude"] = t.log_magnitude
        payload[f"{tag}_se_log_magnitude"] = t.se_log_magnitude
        payload[f"{tag}_se_log_magnitude_fields"] = t.se_log_magnitude_fields
        payload[f"{tag}_se_log_magnitude_reference"] = t.se_log_magnitude_reference
        payload[f"{tag}_bound_log_magnitude_incoherent"] = t.bound_log_magnitude_incoherent
        payload[f"{tag}_phase"] = t.phase
        payload[f"{tag}_se_phase"] = t.se_phase
        payload[f"{tag}_se_phase_fields"] = t.se_phase_fields
        payload[f"{tag}_se_phase_reference"] = t.se_phase_reference
        payload[f"{tag}_bound_phase_incoherent"] = t.bound_phase_incoherent
        payload[f"{tag}_reference"] = t.reference
        payload[f"{tag}_reference_per_pair"] = t.reference_per_pair
        payload[f"{tag}_reference_independent"] = t.reference_independent
        payload[f"{tag}_se_log_magnitude_random"] = t.se_log_magnitude_random
        payload[f"{tag}_se_phase_random"] = t.se_phase_random
        payload[f"{tag}_chroma_phase_constant_rad"] = t.chroma_phase_constant_rad
        payload[f"{tag}_coherence"] = t.coherence
        payload[f"{tag}_beta_record"] = t.beta_record
        payload[f"{tag}_record_coherent_power"] = t.record_coherent_power
        payload[f"{tag}_power_ratio"] = t.power_ratio
        payload[f"{tag}_fields"] = t.fields
        payload[f"{tag}_lines"] = t.lines
        payload[f"{tag}_line_coherence"] = t.line_coherence
        payload[f"{tag}_group_delay_reference_s"] = t.group_delay_reference_s
        for key, value in t.delay_track.items():
            payload[f"{tag}_delay_{key}"] = np.asarray(value)
    np.savez(target, **payload)
    return pattern, speed, target, None


def load_transfer(path):
    """Rebuild the per-head TapTransfer objects from a saved npz."""
    with np.load(path, allow_pickle=False) as data:
        freqs = data["freqs_hz"]
        ref = tuple(float(v) for v in data["reference_band_hz"])
        out = {}
        for head in (0, 1):
            tag = f"head{head}"
            if f"{tag}_transfer" not in data:
                continue
            track = {k[len(tag) + 7:]: data[k] for k in data.files
                     if k.startswith(f"{tag}_delay_")}
            track = {k: (v.item() if v.ndim == 0 else v) for k, v in track.items()}
            out[head] = tt.TapTransfer(
                head=head, label=tt.HEAD_LABELS[head], freqs=freqs,
                transfer=data[f"{tag}_transfer"], per_field=data[f"{tag}_per_field"],
                log_magnitude=data[f"{tag}_log_magnitude"],
                se_log_magnitude=data[f"{tag}_se_log_magnitude"],
                phase=data[f"{tag}_phase"], se_phase=data[f"{tag}_se_phase"],
                coherence=data[f"{tag}_coherence"], beta_record=data[f"{tag}_beta_record"],
                record_coherent_power=data[f"{tag}_record_coherent_power"],
                power_ratio=data[f"{tag}_power_ratio"],
                fields=int(data[f"{tag}_fields"]), lines=int(data[f"{tag}_lines"]),
                rejected_lines=0, line_coherence=float(data[f"{tag}_line_coherence"]),
                delay_track=track,
                group_delay_reference_s=float(data[f"{tag}_group_delay_reference_s"]),
                reference_band_hz=ref,
                se_log_magnitude_fields=data[f"{tag}_se_log_magnitude_fields"],
                se_log_magnitude_reference=data[f"{tag}_se_log_magnitude_reference"],
                bound_log_magnitude_incoherent=data[f"{tag}_bound_log_magnitude_incoherent"],
                se_phase_fields=data[f"{tag}_se_phase_fields"],
                se_phase_reference=data[f"{tag}_se_phase_reference"],
                bound_phase_incoherent=data[f"{tag}_bound_phase_incoherent"],
                reference=data[f"{tag}_reference"],
                reference_per_pair=data[f"{tag}_reference_per_pair"],
                reference_independent=int(data[f"{tag}_reference_independent"]),
                se_log_magnitude_random=data[f"{tag}_se_log_magnitude_random"],
                se_phase_random=data[f"{tag}_se_phase_random"],
                chroma_phase_constant_rad=float(data[f"{tag}_chroma_phase_constant_rad"]))
        return out


def band_label(low_hz, high_hz):
    return "%.1f-%.1f" % (low_hz / 1e6, high_hz / 1e6)


def print_transfer(pattern, speed, transfers, bands):
    print(f"\n=== {pattern} {speed} ===")
    for head, t in sorted(transfers.items()):
        track = t.delay_track
        print(f"  {t.label}: {t.fields} fields, {t.lines} lines, line coherence "
              f"{t.line_coherence:.4f} (record self ceiling "
              f"{track.get('record_self_coherence', float('nan')):.4f}), "
              f"reference group delay {t.group_delay_reference_s * 1e9:+.1f} ns; "
              f"time base: drift {track.get('drift_over_field_ns', float('nan')):+.0f} ns/field, "
              f"line step rms {track.get('line_step_rms_ns', float('nan')):.0f} ns, "
              f"largest step {track.get('largest_step_ns', float('nan')):.0f} ns at line "
              f"{track.get('largest_step_line', -1)}")
        print("    %-9s %9s %7s %6s %6s %6s %6s %9s %7s %6s %6s %9s %8s" % (
            "band MHz", "mag dB", "SE", "fld", "ref", "rnd", "bnd", "phase deg",
            "SE", "coh", "beta", "PSD dB", "tau ns"))
        for v in tt.band_summary(t, bands):
            print("    %-9s %+9.2f %7.2f %6.2f %6.2f %6.2f %6.2f %+9.1f %7.1f %6.3f %6.3f %+9.2f %+8.1f" % (
                band_label(v.low_hz, v.high_hz), v.magnitude_db, v.se_magnitude_db,
                v.se_magnitude_db_fields, v.se_magnitude_db_reference,
                v.se_magnitude_db_random, v.bound_magnitude_db_incoherent,
                v.phase_deg, v.se_phase_deg, v.coherence, v.beta_record,
                v.power_ratio_db, v.group_delay_ns))
        print("      (SE is fld + ref + rnd + bnd in quadrature: the numerators' scatter "
              "over fields, the denominator's over the record self-pairs, the random error "
              "the coherence and the line count imply, and the bound the record tap's "
              "coherent fraction beta puts on the ratio.  The colour-under band's phase "
              "carries a free constant, removed here: %+.1f deg)"
              % np.degrees(t.chroma_phase_constant_rad))


def print_separation(speed, separations, bands):
    print(f"\n=== separation across patterns, {speed}: common transfer and the content-dependence witness ===")
    print("    (chi2 is the reduced chi-square of the per-pattern values about the "
          "common one; an LTI chain gives about 1. The colour-under band's phase "
          "carries its own per-line rotation, so its phase chi2 is not a witness.)")
    for head, s in sorted(separations.items()):
        print(f"  {s.label}: patterns {', '.join(s.patterns)}")
        print("    %-9s %9s %7s %9s %8s %8s %8s %8s %4s" % (
            "band MHz", "common dB", "SE", "phase deg", "excess dB", "chi2 mag",
            "exc deg", "chi2 ph", "n"))
        for row in tt.band_separation(s, bands):
            print("    %-9s %+9.2f %7.2f %+9.1f %8.2f %8.1f %8.1f %8.1f %4d" % (
                band_label(row["low_hz"], row["high_hz"]), row["common_db"],
                row["se_common_db"], row["common_phase_deg"], row["content_excess_db"],
                row["chi2_magnitude"], row["content_excess_deg"], row["chi2_phase"],
                row["patterns"]))


def print_attribution(speed, separations, params):
    print(f"\n=== attribution against rf_stages, {speed} ===")
    for head, s in sorted(separations.items()):
        a = tt.stage_attribution(s, params)
        if not a.get("fitted"):
            print(f"  {s.label}: not fitted ({a.get('why')})")
            continue
        print(f"  {s.label}: {a['explained_fraction'] * 100:.1f}% of the common log-magnitude "
              f"shape explained by {len(a['entries'])} fused RF-stage entries "
              f"(residual {a['residual_rms_db']:.2f} dB rms); the entries span "
              f"{a['effective_directions']:.2f} effective directions of {a['of']} "
              f"at condition {a['condition']:.0f}")
        for name, coefficient in a["coefficients"].items():
            print(f"      {coefficient:+8.3f}  x  {name}")
        print(f"    excess phase measured {a['excess_phase_measured_rms_deg']:.1f} deg rms; "
              f"beyond minimum phase {a['non_minimum_phase_rms_deg']:.1f} deg rms "
              f"(interior of the measured band)")
    print("  not in the transfer at all: " + a["not_in_the_transfer"])
    print("  a head difference is: " + a["head_difference_is"])
    print("  one deck cannot separate: " + a["cannot_separate_with_one_deck"])


def print_comparisons(pattern, speed, transfers):
    export = tt.compare_with_export(transfers)
    if export is not None:
        print(f"\n=== {pattern} {speed} against the 2026-09-02 power-ratio export "
              f"({len(export['signals'])} luma-only signals) ===")
        print("    %-9s %10s %8s | %-32s | %-32s | %s" % (
            "band MHz", "export dB", "spread", "coherent dB per head",
            "own power ratio dB per head", "coherence per head"))
        for r in export["rows"]:
            coh = " ".join("%+7.2f" % v for v in r["coherent_db"].values())
            pr = " ".join("%+7.2f" % v for v in r["own_power_ratio_db"].values())
            g = " ".join("%5.3f" % v for v in r["coherence"].values())
            print("    %-9s %+10.2f %8.2f | %-32s | %-32s | %s" % (
                band_label(r["low_hz"], r["high_hz"]), r["export_mean_db"],
                r["export_spread_db"], coh, pr, g))
        print("    " + export["why"])
    depth = tt.compare_with_depth_slopes(transfers)
    print(f"\n=== {pattern} {speed} against the depth agent's slopes ===")
    ref = depth["depth_agent"]
    for label, values in depth.items():
        if label == "depth_agent":
            continue
        cs, cse = values["chroma_slope_db_per_octave"]
        ls, lse = values["luma_slope_db_per_mhz"]
        print(f"  {label}: chroma band {ref['chroma_band_hz'][0] / 1e6:.2f}-"
              f"{ref['chroma_band_hz'][1] / 1e6:.2f} MHz slope {cs:+.2f} +/- {cse:.2f} dB/oct "
              f"(depth agent SP: {ref['chroma_slope_db_per_octave_sp'][0]:+.1f} +/- "
              f"{ref['chroma_slope_db_per_octave_sp'][1]:.1f}); luma band "
              f"{ref['luma_band_hz'][0] / 1e6:.1f}-{ref['luma_band_hz'][1] / 1e6:.1f} MHz "
              f"slope {ls:+.2f} +/- {lse:.2f} dB/MHz (depth agent: flat to "
              f"{ref['luma_flat_to_hz'] / 1e6:.1f} MHz)")


def plot_speed(speed, results, separations, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    for head in (0, 1):
        for pattern, transfers in sorted(results.items()):
            if head not in transfers:
                continue
            t = transfers[head]
            f = t.freqs / 1e6
            ok = t.se_log_magnitude < 0.1
            axes[0, head].plot(f[ok], t.log_magnitude[ok] * tt.DB_PER_NEPER, lw=0.7, label=pattern)
            axes[1, head].plot(f[ok], np.degrees(t.phase[ok]), lw=0.7)
            axes[2, head].plot(f, t.coherence, lw=0.7)
        if head in separations:
            s = separations[head]
            good = np.isfinite(s.common_log_magnitude) & (s.se_common_log_magnitude < 0.1)
            axes[0, head].plot(s.freqs[good] / 1e6, s.common_log_magnitude[good] * tt.DB_PER_NEPER,
                               "k", lw=1.5, label="common")
            axes[1, head].plot(s.freqs[good] / 1e6, np.degrees(s.common_phase[good]), "k", lw=1.5)
        axes[0, head].set_title(f"{speed}: {tt.HEAD_LABELS[head]}: |H| dB (bins with SE < 0.1 Np)")
        axes[1, head].set_title("excess phase, degrees")
        axes[2, head].set_title("coherence record -> playback")
        axes[2, head].set_xlabel("MHz")
        axes[0, head].set_ylim(-30, 10)
        axes[1, head].set_ylim(-180, 180)
        for ax in axes[:, head]:
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 7)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, f"tap_transfer_{speed}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--speed", action="append", default=None)
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse saved per-pair results where present")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    pairs = capture_pairs(args.root, args.speed, args.pattern)
    if not pairs:
        print("no capture pairs found", file=sys.stderr)
        return 1
    jobs = []
    saved = {}
    for pattern, speed, rec, pb in pairs:
        target = os.path.join(args.out, f"tap_transfer_{pattern}_{speed}.npz")
        if args.reuse and os.path.exists(target):
            saved[(pattern, speed)] = target
        else:
            jobs.append((pattern, speed, rec, pb, args.out))
    if jobs:
        with Pool(min(args.jobs, len(jobs))) as pool:
            for pattern, speed, target, error in pool.imap_unordered(analyse_pair, jobs):
                if error:
                    print(f"  {pattern} {speed}: FAILED: {error}", file=sys.stderr)
                else:
                    saved[(pattern, speed)] = target
    bands = tt.reporting_bands()
    speeds = sorted({speed for _, speed in saved})
    summary = {}
    for speed in speeds:
        params = tt.capture_parameters("NTSC", "VHS", speed)
        results = {pattern: load_transfer(path) for (pattern, s), path in sorted(saved.items()) if s == speed}
        for pattern, transfers in sorted(results.items()):
            print_transfer(pattern, speed, transfers, bands)
        separations = tt.separate(results)
        print_separation(speed, separations, bands)
        print_attribution(speed, separations, params)
        for pattern in ("75bars", "ntc7composite-y-only"):
            if pattern in results:
                print_comparisons(pattern, speed, results[pattern])
        for head, s in separations.items():
            summary[f"{speed}_head{head}_common_log_magnitude"] = s.common_log_magnitude
            summary[f"{speed}_head{head}_se_common_log_magnitude"] = s.se_common_log_magnitude
            summary[f"{speed}_head{head}_common_phase"] = s.common_phase
            summary[f"{speed}_head{head}_content_excess_nepers"] = s.content_excess_nepers
            summary[f"{speed}_head{head}_content_excess_radians"] = s.content_excess_radians
            summary[f"{speed}_head{head}_chi2_magnitude"] = s.content_chi2_magnitude
            summary[f"{speed}_head{head}_patterns"] = np.array(s.patterns)
            summary["freqs_hz"] = s.freqs
        if args.plot:
            print("plot:", plot_speed(speed, results, separations, args.out))
    path = os.path.join(args.out, "tap_transfer_summary.npz")
    np.savez(path, **summary)
    print("\nwritten:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
