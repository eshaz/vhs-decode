"""Rate constraints measured on the real per-line series.

Ethan: *"Constrain parameter rates to physical time constants. Drum
29.97 Hz, head switch 59.94 Hz, capstan servo tens of Hz. Nothing
mechanical varies above a few hundred Hz; a parameter floating per-line
will absorb things it shouldn't."*

This runs `vhsdecode.models.rate_constraints.rate_test` and `constrain`
on two kinds of real series, each parameter against the rate the
registry allows it:

  the hsync-impulse exports in the shared inbox (`<cond>_hs_impulse.npz`:
  per-line front porch, sync tip and back porch in IRE, the sync width in
  us, and the per-line complex response ln|H| at the producer's reference
  frequencies, per head, on 75 bars SP, bounce SP and 75 bars EP);

  series extracted from the 75-bars SP playback RF with the decoder's own
  front end (`tools/ringing_measure/content_independence.py`: sync tip,
  front and back porch, sync width, RF envelope at the tip, the line
  period against the standard's, and the burst's phase against its own
  sync fall with the format's per-line rotation removed as structure).
  Sync and blanking intervals only; nothing is read from active video.

Every series is judged in the structured form (fields by line positions,
alternating heads on their true time axis), the lines inside the
specification's head-switch window and the vertical interval excluded by
`sync_geometry.head_switch_window` and the exports' own first line.

Usage:
    PYTHONPATH=/workspaces/vhs-decode python3 -m tools.ringing_measure.rate_constraints_measure \\
        --npz-dir /tmp/claude-1000/-workspaces-vhs-decode/shared \\
        --capture /testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \\
        --out /tmp/claude-1000/-workspaces-vhs-decode/<session>/scratchpad/rate_verdicts.json
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

from vhsdecode.models import rate_constraints as rc
from vhsdecode.models import sync_geometry

# The hsync-impulse producer (shared/hs_impulse.py) exports, per head,
# `levels_h<head>` as rows of [active, blank, tip, back] in IRE over lines
# LINE_LO=10 to LINE_HI=262 (exclusive) of every field it saw, `width_h<head>`
# as the sync width in us on the same rows, and `tdim_H_M1f_h<head>` as the
# per-line complex response at REF_MHZ = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
# with its own `tdim_lines_*` axis. The "active" column is picture content
# and is not read here.
HS_IMPULSE_LINES = (10, 262)
HS_IMPULSE_COLUMNS = {"front_porch_level": 1, "sync_tip_level": 2,
                      "back_porch_level": 3}
HS_IMPULSE_REF_MHZ = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
HEAD_RESPONSE_MHZ = (0.5, 1.5, 3.0)
CONDITIONS = {"hb75": "75 bars SP", "hbnc": "bounce SP", "he75": "75 bars EP"}

DEFAULT_NPZ_DIR = "/tmp/claude-1000/-workspaces-vhs-decode/shared"
DEFAULT_CAPTURE = ("/testdata/test_patterns/vhs/playback/"
                   "zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac")

# Which registry entry each measured series is an instance of.
PARAMETER_OF = {
    "sync_tip_level": "sync_tip_level",
    "front_porch_level": "front_porch_level",
    "back_porch_level": "back_porch_level",
    "sync_width": "sync_width",
    "burst_phase": "burst_phase",
    "rf_envelope": "rf_envelope",
    "line_period": "time_base",
    "line_position": "time_base",
    "head_response": "head_response",
}


def usable_line_range(system: str = "NTSC") -> range:
    """Lines from the first normal line after the vertical interval up to
    the specification's head-switch window (SMPTE 32M, 5 to 8 H ahead of
    the V-sync edge plus the assumed transient), both from the tree."""
    window = sync_geometry.head_switch_window(system)
    ahead_low, ahead_high = window["specified_range_h"]
    field_lines = int(sync_geometry._system(system)) // 2 + 1
    last = int(field_lines - ahead_high - window["transient_h"])
    return range(HS_IMPULSE_LINES[0], last)


def nominal_rate(path: str, override: Optional[float]) -> float:
    if override:
        return float(override)
    found = re.search(r"(\d+(?:\.\d+)?)msps", os.path.basename(path))
    if not found:
        raise SystemExit(f"cannot read a sample rate from {path}; give --rate")
    return float(found.group(1)) * 1e6


# --------------------------------------------------------------------------
# series from the hsync-impulse exports
# --------------------------------------------------------------------------

def export_series(path: str, head: int) -> Dict[str, object]:
    data = np.load(path)
    levels = np.asarray(data[f"levels_h{head}"], dtype=np.float64)
    width = np.asarray(data[f"width_h{head}"], dtype=np.float64)
    per_field = HS_IMPULSE_LINES[1] - HS_IMPULSE_LINES[0]
    if len(levels) % per_field or len(width) != len(levels):
        raise ValueError(f"{path} head {head}: {len(levels)} rows is not a "
                         f"whole number of {per_field}-line fields")
    fields = len(levels) // per_field
    # heads alternate per field, so this head's fields are every other one
    field_index = 2 * np.repeat(np.arange(fields), per_field) + head
    line_index = np.tile(np.arange(*HS_IMPULSE_LINES), fields)
    out = {"field_index": field_index, "line_index": line_index, "series": {}}
    for name, column in HS_IMPULSE_COLUMNS.items():
        out["series"][name] = levels[:, column]
    out["series"]["sync_width"] = width
    response = np.asarray(data[f"tdim_H_M1f_h{head}"])
    lines = np.asarray(data[f"tdim_lines_M1f_h{head}"])
    out["response"] = {}
    for mhz in HEAD_RESPONSE_MHZ:
        k = HS_IMPULSE_REF_MHZ.index(mhz)
        magnitude = np.log(np.abs(response[:, :, k]))          # (lines, fields)
        out["response"][mhz] = {
            "values": magnitude.T.ravel(),
            "field_index": 2 * np.repeat(np.arange(magnitude.shape[1]), len(lines)) + head,
            "line_index": np.tile(lines, magnitude.shape[1]),
        }
    return out


# --------------------------------------------------------------------------
# series from the RF capture, with the decoder's own front end
# --------------------------------------------------------------------------

def capture_series(path: str, rate_hz: float, seconds: Optional[float],
                   system: str = "NTSC") -> Dict[str, object]:
    from tools.ringing_measure import content_independence as tool

    count = int(round(seconds * rate_hz)) if seconds else 1 << 40
    samples = tool.read_rf(path, 0, count)
    demod = tool.demodulate(samples, rate_hz, system)
    demod["sysparams"]["system"] = system
    pulses = tool.find_pulses(demod["luma_lowpass_ire"], rate_hz, demod["sysparams"])
    fields = tool.find_fields(pulses, rate_hz, demod["sysparams"])
    if len(fields) < 3:
        raise RuntimeError(f"{path}: found only {len(fields)} fields")
    fields = fields[:-1]                  # the last one is not closed
    table = tool.measure_lines(demod, pulses, fields)
    sysparams = demod["sysparams"]
    per_us = rate_hz * 1e-6
    line_samples = float(sysparams["line_period"]) * per_us
    colour_under = float(demod["colour_under_hz"])
    chroma = demod["chroma"]
    burst_us = tuple(float(v) for v in sysparams["colorBurstUS"])

    field = np.asarray(table["field"], dtype=int)
    line = np.asarray(table["line"], dtype=int)
    fall = np.asarray(table["fall_sample"], dtype=np.float64)
    head = np.asarray(table["half_line_offset"], dtype=int)

    # the burst's complex value against the line's own sync fall
    burst = np.zeros(len(fall), dtype=np.complex128)
    for k, anchor in enumerate(fall):
        low = int(round(anchor + burst_us[0] * per_us))
        high = int(round(anchor + burst_us[1] * per_us))
        if low < 0 or high > len(chroma) or high <= low + 8:
            burst[k] = np.nan
            continue
        t = (np.arange(low, high) - anchor) / rate_hz
        burst[k] = np.mean(chroma[low:high] * np.exp(-2j * np.pi * colour_under * t))
    amplitude = np.abs(burst)
    has_burst = amplitude > 0.5 * np.nanmedian(amplitude)

    # the format rotates the colour-under phase by a quarter turn per line,
    # one way on one head and the other way on the other
    # (format_defs.vhs NTSC_ROTATION); the rotation is read off each field
    # as the nearest quarter turn to the median line-to-line phase step and
    # removed as STRUCTURE, leaving the phase the time base moves
    phase = np.full(len(fall), np.nan)
    rotation_seen = {}
    for f in np.unique(field):
        picked = np.flatnonzero((field == f) & has_burst & np.isfinite(burst))
        if len(picked) < 8:
            continue
        order = picked[np.argsort(line[picked])]
        consecutive = np.diff(line[order]) == 1
        steps = np.angle(burst[order][1:][consecutive] * np.conj(burst[order][:-1][consecutive]))
        quarter = np.pi / 2.0
        turns = int(np.round(np.median(steps) / quarter))
        rotation_seen[int(f)] = turns
        unrotated = burst[order] * np.exp(-1j * turns * quarter * line[order])
        phase[order] = np.unwrap(np.angle(unrotated))

    # the time base two ways: the line period against the standard's, as
    # a deviation, for lines whose successor is the next line of the same
    # field (a first difference, whose noise is blue: lag-one -0.5 by
    # construction); and the sync fall's POSITION against the field's own
    # nominal grid in samples, the form the decoder's linelocs take
    period = np.full(len(fall), np.nan)
    same = (field[1:] == field[:-1]) & (line[1:] == line[:-1] + 1)
    period[:-1][same] = (fall[1:][same] - fall[:-1][same]) / line_samples - 1.0
    position = np.full(len(fall), np.nan)
    for f in np.unique(field):
        picked = np.flatnonzero(field == f)
        nominal = line[picked] * line_samples
        slope, intercept = np.polyfit(nominal, fall[picked], 1)
        position[picked] = fall[picked] - (slope * nominal + intercept)

    series = {
        "sync_tip_level": np.asarray(table["sync_tip_ire"], dtype=np.float64),
        "front_porch_level": np.asarray(table["front_porch_ire"], dtype=np.float64),
        "back_porch_level": np.asarray(table["back_porch_ire"], dtype=np.float64),
        "sync_width": np.asarray(table["sync_width_us"], dtype=np.float64),
        "rf_envelope": np.asarray(table["envelope_tip"], dtype=np.float64),
        "burst_phase": phase,
        "line_period": period,
        "line_position": position,
    }
    return {"field_index": field, "line_index": line, "head": head,
            "series": series, "fields": len(fields),
            "seconds": len(samples) / rate_hz,
            "rotation_turns_per_field": rotation_seen,
            "colour_under_hz": colour_under}


# --------------------------------------------------------------------------
# running the test
# --------------------------------------------------------------------------

def judge(values, field_index, line_index, parameter: str, rates,
          system: str, draws: int) -> Dict[str, object]:
    allowed = rc.allowed_rate(parameter, system=system)
    keep = np.isin(line_index, list(usable_line_range(system))) & np.isfinite(values)
    mechanisms = {name: entry["rate_hz"] for name, entry in rates.items()
                  if entry["kind"] == "mechanical" and entry["rate_hz"] > 0}
    verdict = rc.rate_test(values[keep], rc.line_rate_hz(system),
                           allowed["allowed_hz"],
                           field_rate_hz=rc.field_rate_hz(system),
                           field_index=field_index[keep],
                           line_index=line_index[keep],
                           mechanism_hz=mechanisms, draws=draws)
    verdict["allowed"] = {k: allowed[k] for k in ("allowed_hz", "set_by", "rule")}
    if verdict.get("usable"):
        # the constraint's cost: the low-pass at the allowed rate applied
        # field by field, what it removes and whether that is white
        laid = rc.structure(values[keep], field_index[keep], line_index[keep])
        removed_rms, lag_one = [], []
        for row in laid["table"]:
            out = rc.constrain(row, rc.line_rate_hz(system), allowed["allowed_hz"])
            removed_rms.append(out["evidence"]["removed_rms"])
            lag_one.append(out["evidence"]["removed_lag_one"])
        verdict["constraint"] = {
            "removed_rms": float(np.mean(removed_rms)),
            "removed_share": float(np.mean(removed_rms) ** 2 / max(verdict["variance"], 1e-300)),
            "removed_lag_one": float(np.mean(lag_one)),
        }
    return verdict


def summarise(label: str, head, parameter: str, verdict: Dict[str, object]
              ) -> Dict[str, object]:
    row = {"series": label, "head": head, "parameter": parameter,
           "usable": bool(verdict.get("usable"))}
    if not verdict.get("usable"):
        row["why"] = verdict.get("why")
        return row
    residual, locked, slow = verdict["residual"], verdict["line_locked"], verdict["slow"]
    half = slow["half_contrast"]
    strongest = slow.get("strongest_line")
    row.update({
        "fields": verdict["fields"], "lines": verdict["lines"],
        "allowed_hz": verdict["allowed_hz"],
        "rms": float(np.sqrt(verdict["variance"])),
        "white_rms": float(np.sqrt(max(residual["floor_per_bin"], 0.0)
                                   * (residual["bins_above"] + residual["bins_below"]))),
        "shares": {"field": verdict["field_term_share"], "sweep": verdict["sweep_share"],
                   "line_locked": verdict["line_term_share"],
                   "residual": verdict["residual_share"]},
        "residual_ratio": residual["excess_ratio"],
        "residual_bound": verdict["null"]["residual"]["excess_ratio_bound"],
        "residual_sigma": residual["sigma"],
        "floats_too_fast": bool(verdict["floats_too_fast"]),
        "excess_lives_hz": verdict.get("excess_lives_hz"),
        "line_locked_ratio": locked["excess_ratio"],
        "line_locked_bound": verdict["null"]["line_locked"]["excess_ratio_bound"],
        "line_locked_above_rate": bool(verdict["line_locked_above_rate"]),
        "line_locked_lives_in_lines": locked["lives_in_lines"],
        "line_locked_share_above_rate": locked["share_of_line_term_variance_above_rate"],
        "half_contrast": {k: half.get(k) for k in ("difference", "se", "sigma",
                                                     "identified", "lives_at_hz",
                                                     "record_s")},
        "strongest_slow_line": ({k: strongest[k] for k in ("name", "coefficient",
                                                            "rate_hz", "amplitude",
                                                            "sigma")}
                                if strongest else None),
        "constraint": verdict.get("constraint"),
        "signature": verdict["signature"],
    })
    return row


def print_table(rows: List[Dict[str, object]]) -> None:
    print()
    print("  %-14s %-4s %-18s %-8s %9s %9s | %-27s | %-31s | %-16s | %s" % (
        "series", "head", "parameter", "F x L", "rms", "white", "remainder above rate",
        "line-locked above rate", "half-contrast", "strongest slow line / constraint"))
    for r in rows:
        if not r["usable"]:
            print("  %-14s %-4s %-18s unusable: %s" % (r["series"], r["head"], r["parameter"], r["why"]))
            continue
        remainder = "%.3f (bound %.3f) %s" % (
            r["residual_ratio"], r["residual_bound"],
            "TOO FAST" if r["floats_too_fast"] else "white")
        locked = "%.2f (bound %.2f) %s" % (
            r["line_locked_ratio"], r["line_locked_bound"],
            ("lines %s" % (r["line_locked_lives_in_lines"],)
             if r["line_locked_above_rate"] else "white"))
        half = r["half_contrast"]
        contrast = "%+.2f sigma <= %.1f Hz" % (half["sigma"], half["lives_at_hz"])
        strongest = r["strongest_slow_line"]
        slow = ("%s %.1fs" % (strongest["name"], strongest["sigma"]) if strongest else "-")
        cost = r["constraint"]
        print("  %-14s %-4s %-18s %3dx%-4d %9.4g %9.4g | %-27s | %-31s | %-16s | %s; removed %.1f%% lag1 %+.2f" % (
            r["series"], r["head"], r["parameter"], r["fields"], r["lines"], r["rms"],
            r["white_rms"], remainder, locked, contrast, slow,
            100.0 * cost["removed_share"], cost["removed_lag_one"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--capture", default=DEFAULT_CAPTURE)
    parser.add_argument("--rate", type=float, default=None,
                        help="capture sample rate in Hz (default: from the file name)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="how much of the capture to read (default: all of it)")
    parser.add_argument("--system", default="NTSC")
    parser.add_argument("--draws", type=int, default=rc.NULL_DRAWS)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--skip-exports", action="store_true")
    parser.add_argument("--out", default=None, help="JSON file for the verdicts")
    args = parser.parse_args(argv)

    rates = rc.mechanism_rates("VHS", args.system, "SP", "SLV-778HF")
    ceiling = rc.mechanical_ceiling_hz(rates)
    lines = usable_line_range(args.system)
    print(f"ceiling {ceiling['ceiling_hz']:.1f} Hz set by {ceiling['set_by']}; "
          f"usable lines {lines.start}..{lines.stop - 1}; null {args.draws} draws")
    rows: List[Dict[str, object]] = []

    if not args.skip_exports:
        for cond in args.conditions.split(","):
            path = os.path.join(args.npz_dir, f"{cond}_hs_impulse.npz")
            if not os.path.exists(path):
                print(f"  {path}: missing, skipped")
                continue
            label = CONDITIONS.get(cond, cond)
            for head in (0, 1):
                export = export_series(path, head)
                for name, values in export["series"].items():
                    verdict = judge(values, export["field_index"], export["line_index"],
                                    PARAMETER_OF[name], rates, args.system, args.draws)
                    rows.append(summarise(label, head, name, verdict))
                for mhz, entry in export["response"].items():
                    verdict = judge(entry["values"], entry["field_index"], entry["line_index"],
                                    "head_response", rates, args.system, args.draws)
                    rows.append(summarise(label, head, f"ln|H| {mhz:g} MHz", verdict))
            print_table([r for r in rows if r["series"] == label])

    if not args.skip_capture:
        rate_hz = nominal_rate(args.capture, args.rate)
        capture = capture_series(args.capture, rate_hz, args.seconds, args.system)
        label = os.path.basename(args.capture).split("-NTSC")[0].replace("zaroff-", "RF ")
        print(f"\n  {os.path.basename(args.capture)}: {capture['fields']} fields in "
              f"{capture['seconds']:.3f} s; burst rotation per field (quarter turns): "
              f"{sorted(set(capture['rotation_turns_per_field'].values()))}")
        capture_rows = []
        for head in (0, 1):
            pick = capture["head"] == head
            for name, values in capture["series"].items():
                verdict = judge(values[pick], capture["field_index"][pick],
                                capture["line_index"][pick], PARAMETER_OF[name],
                                rates, args.system, args.draws)
                capture_rows.append(summarise(label, head, name, verdict))
        # both heads together: the only series on which the drum's own rate
        # is resolvable, at the cost of the head difference sharing its bin
        for name in ("sync_tip_level", "rf_envelope", "line_position", "burst_phase"):
            verdict = judge(capture["series"][name], capture["field_index"],
                            capture["line_index"], PARAMETER_OF[name], rates,
                            args.system, args.draws)
            capture_rows.append(summarise(label, "both", name, verdict))
        print_table(capture_rows)
        rows.extend(capture_rows)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1, default=lambda v: (
                v.tolist() if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, np.floating) else
                int(v) if isinstance(v, np.integer) else
                bool(v) if isinstance(v, np.bool_) else str(v)))
        print(f"\n  verdicts saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
