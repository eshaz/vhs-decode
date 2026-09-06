#!/usr/bin/env python3
"""THE RESIDUAL AGAINST AN INDEPENDENT REFERENCE, at both taps of one deck.

    PYTHONPATH=/workspaces/vhs-decode python3 \
        tools/ringing_measure/reference_residual.py \
        [--pattern 75bars ...] [--speed SP] [--samples N] [--no-clip]

videosynth generates the picture the recorder was given, `reference_signal`
carries it through the record chain, and this script differences it against
the RECORD tap and then against the PLAYBACK tap of the same pattern. The
record tap is the deck's own RF before the tape, so its residual is the
record chain's model error with nothing else in it; the playback tap's extra
residual is the tape and the playback chain. That decomposition is what an
external reference buys, and no measurement in this repository could make it
before - every ideal the arc has differenced against was drawn by the arc.

Every residual is reported in IRE and as a multiple of the output file's own
quantisation floor (`models/output_limit`), because a residual under that
floor cannot reach the file however large it looks in any other unit.

Read `vhsdecode/models/reference_signal.py` first: it records what the
generator gets right, what it gets wrong (its sync pulses are one transition
narrow at the half-amplitude points, which is why alignment uses the leading
edge alone), and which stages of the record chain are modelled.
"""

import argparse
import os
import sys

import numpy as np

from vhsdecode.models import output_limit, reference_signal as rs

ROOT = "/testdata/test_patterns/vhs"
ORDER = ("sync leading edge", "sync tip", "sync trailing", "back porch",
         "blanking", "active", "line")
# Which windows the sync-only law licenses a correction from. `active` is
# reported because it bounds the total, not because anything may be fitted
# to it - and against this reference it also carries a horizontal
# registration difference between two generators, measured at about one
# microsecond, which belongs to neither the deck nor the tape.
LICENSED = ("sync leading edge", "sync tip", "sync trailing", "back porch",
            "blanking")


def capture_path(tap: str, pattern: str, speed: str) -> str:
    suffix = "rec" if tap == "record" else "pb"
    return os.path.join(
        ROOT, tap,
        f"zaroff-{pattern}-NTSC-{speed}-SLV-778HF-50msps-rf-{suffix}.flac")


def show(title: str, result: dict) -> None:
    floor = output_limit.output_profile()["quantisation_noise_ire"]
    print(f"\n{title}")
    print(f"  {result['line_pairs']} line pairs from "
          f"{result['measured_fields']} measured fields, alignment "
          f"{result['delay_samples']['median']:+.3f} samples "
          f"(spread {result['delay_samples']['spread']:.3f})")
    if result["calibrations"]:
        gains = [c["gain"] for c in result["calibrations"]]
        offsets = [c["tip_offset_ire"] for c in result["calibrations"]]
        print(f"  sync anchoring: record level "
              f"{np.mean(gains):.5f} of the specified deviation, sync tip "
              f"{np.mean(offsets):+.3f} IRE off the specified carrier")
    print(f"  {'window':<18} {'profile IRE':>12} {'floors':>8} "
          f"{'scatter IRE':>12} {'per sample':>11} {'floors':>8}")
    for name in ORDER:
        if name not in result["windows"]:
            continue
        row = result["windows"][name]
        mark = " " if name in LICENSED else "*"
        print(f"  {name:<18}{mark}{row['profile']['effect_ire']:>11.4f} "
              f"{row['profile']['over_floor']:>8.1f} "
              f"{row['scatter_ire']:>12.4f} "
              f"{row['per_sample']['effect_ire']:>11.4f} "
              f"{row['per_sample']['over_floor']:>8.1f}")
    print(f"  (floor {floor:.4f} IRE rms; profile = the residual averaged "
          f"over {result['line_pairs']} line pairs, which is the model's own "
          "error;")
    print("   scatter = what the averaging removed; * = not licensed by the "
          "sync-only law, reported as a bound)")


def decompose(record: dict, playback: dict) -> None:
    """What the tape and the playback chain add over the record chain."""
    floor = output_limit.output_profile()["quantisation_noise_ire"]
    added = rs.tap_difference(record, playback)["windows"]
    print("\n  THE DECOMPOSITION: the playback tap's profile MINUS the record")
    print("  tap's, so the record chain's model error and the generator's own")
    print("  defects cancel and what is left is the tape and the playback chain.")
    print(f"  {'window':<18} {'record':>9} {'playback':>10} {'added IRE':>11} "
          f"{'floors':>8} {'over SE':>9}")
    for name in ORDER:
        if name not in added:
            continue
        row = added[name]
        print(f"  {name:<18} {row['record_ire'] / floor:>9.1f} "
              f"{row['playback_ire'] / floor:>10.1f} "
              f"{row['effect_ire']:>11.4f} {row['over_floor']:>8.1f} "
              f"{row['over_error']:>9.1f}")
    print("  (record and playback in floors; over SE = the added term against "
          "the two taps' pooled standard error, so under about one it is noise)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", nargs="*", default=["75bars"])
    parser.add_argument("--speed", default="SP")
    parser.add_argument("--samples", type=int, default=6_000_000,
                        help="samples of capture to read; the whole file is "
                             "24 000 512, about 0.48 s")
    parser.add_argument("--frames", type=int, default=3,
                        help="reference frames to generate; two fields a "
                             "frame and the locator drops partial ones")
    parser.add_argument("--no-clip", action="store_true",
                        help="leave the white and dark clips out of the "
                             "forward chain - the deck's measured dark clip "
                             "falsifies both readings of the clause")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="difference against the specification's law "
                             "with no free parameter at all")
    parser.add_argument("--clip-ire", nargs=2, type=float, default=None,
                        metavar=("DARK", "WHITE"),
                        help="a MEASURED clip pair in IRE in place of the "
                             "standard's; the SLV-778HF record tap's own "
                             "dark floor is -125.8 IRE and no white clip is "
                             "visible there")
    parser.add_argument("--tap", nargs="*", default=["record", "playback"])
    args = parser.parse_args(argv)

    tool = rs.available()
    if not tool["present"]:
        print(tool["why"], file=sys.stderr)
        return 1
    print(f"videosynth {tool['version']} at {tool['binary']}")

    for pattern in args.pattern:
        results = {}
        for tap in args.tap:
            path = capture_path(tap, pattern, args.speed)
            if not os.path.exists(path):
                print(f"missing: {path}", file=sys.stderr)
                continue
            results[tap] = rs.residual_against_capture(
                path, pattern=pattern, speed=args.speed, frames=args.frames,
                samples=args.samples, calibrate=not args.no_calibrate,
                clip=not args.no_clip,
                clip_ire=tuple(args.clip_ire) if args.clip_ire else None)
            show(f"{pattern} {args.speed} {tap} tap  "
                 f"({os.path.basename(path)})", results[tap])
        if "record" in results and "playback" in results:
            decompose(results["record"], results["playback"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
