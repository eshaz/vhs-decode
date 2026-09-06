#!/usr/bin/env python3
"""The held-out ladder on the real exports: every entry of the chain judged
on data its parameters never saw.

Ethan: *"Report held-out residual with parameters frozen. In-sample residual
falls monotonically on pure noise - it can't distinguish a good model from an
over-parameterized one."*

This is the runnable form of `vhsdecode/models/holdout.py`. Two kinds of
material, each with the split its structure allows:

  the sync step-response exports   {cd,home,pnb}_tree_noeq_sync_step_response.npz
                                   one complex response per head, and the
                                   same response over the FIRST and SECOND
                                   halves of the fields. The first half
                                   trains, the second judges - blocked in
                                   time, within a head, on shared bins
                                   (`holdout.half_split`) - and the entries
                                   are the chain's entry-wise admitted key in
                                   its declared order, behind a level and a
                                   delay. Loaded and whitened exactly as
                                   `residual_floor` does, so every residual
                                   reads in units of the half's standard
                                   error and 1 is the floor.
  the hsync impulse exports        *_hs_impulse.npz: the pooled per-head
                                   response (no halves, so the split is over
                                   contiguous BANDS: interpolation across
                                   frequency, not reproduction in time); the
                                   per-line, per-field response at one
                                   reference frequency and the per-line
                                   levels, both split over contiguous blocks
                                   of FIELDS within a head.

Every table has the same columns: the entry, its parameter count, the
in-sample residual, the held-out residual with a block-jackknife error, the
held-out residual the scrambled-signature null reaches, the chance level
the entry had to get below, and the verdict. Nothing in a held-out column
was seen by the parameters that produced it. After each ladder on the sync
exports the remainder is judged the arc's way - `residual_floor.verdict`
(the floor, the whiteness, the distribution) and
`information_extrapolation.structureless` (does what is left reproduce on
the half that did not build it?) - beside `residual_floor.test`'s own
all-at-once held-out figure for the same key.

    PYTHONPATH=/workspaces/vhs-decode python3 -m tools.ringing_measure.holdout
    PYTHONPATH=/workspaces/vhs-decode python3 -m tools.ringing_measure.holdout --only sync --key base
    PYTHONPATH=/workspaces/vhs-decode python3 -m tools.ringing_measure.holdout --only impulse --draws 12

The first build of this tool carried a third section that demodulated an RF
capture itself. It is gone: it re-implemented the decoder's demodulation
outside the decoder and nothing here could verify it against the decoder's
own output (memory: runtime/offline parity is established per instrument,
never assumed). The decoder's sync-pulse instrument is what writes the
exports above, and that is the path from a capture to this tool.
"""

import argparse
import ast
import glob
import itertools
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vhsdecode.models import holdout as ho                         # noqa: E402
from vhsdecode.models import information_extrapolation as extrapolation  # noqa: E402
from vhsdecode.models import interference as inf                   # noqa: E402
from vhsdecode.models import residual_floor                        # noqa: E402
from vhsdecode.models import tesseract                             # noqa: E402

SHARED = "/tmp/claude-1000/-workspaces-vhs-decode/shared"
# the exports are discovered by these suffixes in the shared directory
SYNC_SUFFIX = "_tree_noeq_sync_step_response.npz"
IMPULSE_SUFFIX = "_hs_impulse.npz"
IMPULSE_PRODUCER = "hs_impulse.py"

# The impulse export does not carry three things its producer fixes, and the
# producer (shared/hs_impulse.py) is not in the tree. They are READ FROM THE
# PRODUCER'S SOURCE whenever it sits beside the exports (`producer_constants`
# parses it rather than importing it, because importing it pulls in the
# producer's own scratch dependencies):
#   REF_MHZ            the reference frequencies the time dimension was
#                      sampled at, in the order of the export's last axis
#   MODELS["M1f"]      the fall-edge window in microseconds about the sync
#                      fall, which fixes the frequency resolution of the
#                      pooled response (NFFT over the window's samples)
#   the level columns  the plateaus appended per line, in the order the
#                      producer appends them
# IMPULSE_RECORDED is the record of those values as read from the producer
# on 2026-09-05, used only when the producer is absent; the run says which
# source it used.
IMPULSE_RECORDED = {
    "reference_mhz": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    "fall_window_us": (-1.3, 3.6),
    "level_columns": ("active", "blank", "tip", "back"),
}


def producer_constants(shared):
    """The impulse producer's `REF_MHZ`, `MODELS["M1f"]` and the order in
    which it appends the level columns, read from its source. None when the
    producer is not beside the exports; an error naming what is missing
    when it is there but no longer in the form this reads, so a change in
    the producer cannot pass silently."""
    path = os.path.join(shared, IMPULSE_PRODUCER)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            target = node.targets[0].id
            if target == "REF_MHZ":
                found["reference_mhz"] = tuple(
                    float(v) for v in ast.literal_eval(node.value))
            elif target == "MODELS":
                found["fall_window_us"] = tuple(
                    float(v) for v in ast.literal_eval(node.value)["M1f"])
        # levels[head].append([lv["active"], lv["blank"], ...])
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Subscript)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "levels"
                and node.args and isinstance(node.args[0], ast.List)):
            columns = []
            for element in node.args[0].elts:
                key = getattr(element, "slice", None)
                # before Python 3.9 a subscript's slice was wrapped in Index
                if isinstance(key, getattr(ast, "Index", ())):
                    key = key.value
                if (isinstance(element, ast.Subscript)
                        and isinstance(element.value, ast.Name)
                        and element.value.id == "lv"
                        and isinstance(key, ast.Constant)):
                    columns.append(str(key.value))
            if columns and len(columns) == len(node.args[0].elts):
                found["level_columns"] = tuple(columns)
    missing = [name for name in IMPULSE_RECORDED if name not in found]
    if missing:
        raise SystemExit(f"{path} no longer carries {', '.join(missing)} in "
                         f"the form this tool reads; update "
                         f"producer_constants")
    return found


def discovered(shared, suffix):
    """The export names in `shared` ending in `suffix`, in name order."""
    paths = sorted(glob.glob(os.path.join(shared, f"*{suffix}")))
    return [os.path.basename(p)[: -len(suffix)] for p in paths]


def banner(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def show(label, result, unit):
    for group, got in result.items():
        print(f"\n  {label} - group {group}: {got['train_members']} training "
              f"and {got['test_members']} held-out members; noise "
              f"correlation length {got['correlation_length']:.1f} members")
        print(f"  start: in-sample {got['start_in_sample']:.5g}, "
              f"held-out {got['start_held_out']:.5g}; "
              f"final: in-sample {got['final_in_sample']:.5g}, "
              f"held-out {got['final_held_out']:.5g}")
        print(ho.format_table(got["rows"], unit=unit))
        print(f"  accepted ({len(got['accepted'])} of {len(got['rows'])}): "
              f"{', '.join(got['accepted']) or 'none'}")


# --------------------------------------------------------------------------
# 1. The sync step response: first half fitted, second half judged
# --------------------------------------------------------------------------


def sync_ladder(path, tape, head, args):
    m = residual_floor.load_export(path, head, view=args.view)
    if "held_out" not in m:
        print(f"\n  {tape} head {head}: the export carries no halves; skipped")
        return None
    f = m["frequency_hz"]
    bins = f.size
    keys = residual_floor.keys_for(f)
    key = keys[args.key]
    halves = m["held_out"]
    first = residual_floor.log_domain(halves["H_fit"], halves["se_half"])
    second = residual_floor.log_domain(halves["H_judge"], halves["se_half"])
    floor = np.concatenate([first["floor"], second["floor"]])
    values = np.concatenate([first["log"], second["log"]]) / np.sqrt(floor)
    bin_of = np.tile(np.arange(bins), 2)

    # the noise's correlation along the bins, from the halves' difference
    # (the structure cancels), against what the export's stated resolution
    # implies; and the halves' difference itself against the two the
    # standard error predicts, which is the se calibration residual_floor
    # relies on
    length = ho.noise_correlation_length(values[:bins], values[bins:])
    spacing = float(np.median(np.diff(f)))
    export = np.load(path, allow_pickle=True)
    stated = float(export[f"head_{head}_{args.view}_resolution_mhz"]) * 1e6
    difference = ho._rms(values[:bins] - values[bins:])
    # the part that REPRODUCES between the halves (their mean, level
    # removed) is what the key is asked to explain; its own correlation
    # length says how many independent members carry it
    shared_part = (values[:bins] + values[bins:]) / 2.0
    shared_part = shared_part - shared_part.mean()
    reproducing = ho.noise_correlation_length(shared_part)
    print(f"\n  {tape} head {head} ({args.view}): {bins} valid bins "
          f"{f[0] / 1e6:.3f}-{f[-1] / 1e6:.3f} MHz at {spacing / 1e3:.1f} kHz; "
          f"{len(key)} key entries ({args.key})")
    print(f"  noise correlation length {length['length']:.1f} bins measured "
          f"from the halves' difference (lag-one autocorrelation "
          f"{length['autocorrelation'][0]:.3f}), against {stated / spacing:.1f} "
          f"bins the stated resolution of {stated / 1e6:.3f} MHz implies; "
          f"about {bins / length['length']:.0f} independent members")
    print(f"  halves' whitened difference rms {difference:.2f} against 2.00 "
          f"if the standard error were the whole of it; the part common to "
          f"both halves has rms {ho._rms(shared_part):.1f} and a correlation "
          f"length of {reproducing['length']:.0f} bins, so about "
          f"{bins / reproducing['length']:.0f} independent members carry "
          f"what the key is asked to explain")

    # the level and the delay are declared over the BINS and mapped through
    # `bin_of`, as the key's entries are, so each null is one scrambled
    # shape repeated on both halves
    nuisance = residual_floor.nuisance(f)
    entries = [
        ho.constant_entry("level (nuisance: gain and phase offset)",
                          2 * bins, complex_valued=True, floor=floor,
                          member_of=bin_of),
        ho.linear_entry("delay (nuisance: linear phase)",
                        nuisance["delay"], floor=floor, member_of=bin_of),
    ]
    entries += ho.entries_from_key(key, chain=inf.full_chain(), bin_of=bin_of,
                                   floor=floor)
    split_ = ho.half_split(bins, blocks=args.blocks)
    print(f"  {split_.description}")
    result = ho.ladder(entries, values, split_, draws=args.draws,
                       seed=args.seed, scramble=args.scramble,
                       correlation_length=length["length"], sigma=args.sigma)
    show(f"{tape} head {head} sync {args.view} response", result,
         "floor units: 1 = the half's standard error")
    got = result["pooled"]

    # the remainder, judged the arc's way
    residual = got["residual"]
    left_first = residual[:bins] * np.sqrt(first["floor"])
    left_second = residual[bins:] * np.sqrt(second["floor"])
    parameters = sum(row["count"] for row in got["rows"]
                     if row["verdict"] == "accepted")
    judged = residual_floor.verdict(left_second, second["floor"], parameters)
    # `classify` names its answer "distribution"; residual_floor.verdict
    # looks for "verdict" and reports None, so the detail is read directly
    shape = judged["distribution_detail"].get(
        "distribution", judged["distribution_detail"].get("why", "?"))
    stacked_first = np.concatenate([residual[:bins].real, residual[:bins].imag])
    stacked_second = np.concatenate([residual[bins:].real,
                                     residual[bins:].imag])
    reproduces = extrapolation.structureless(stacked_first, stacked_second)
    print(f"  what the accepted entries leave on the second half: reduced "
          f"chi-square {judged['reduced_chi_square']:.0f} "
          f"(+{judged['excess_db']:.1f} dB over the floor), lag-one z "
          f"{judged['whiteness']['lag_one_z']:.1f}, distribution "
          f"{shape}; {judged['reading']}")
    print(f"  does it reproduce on the half that did not build it? "
          f"agreement {reproduces.get('agreement', float('nan')):+.3f}, "
          f"flatness {reproduces.get('flatness', float('nan')):.2f}: "
          f"{reproduces.get('why', '')}")
    # THE LADDER IS GREEDY, and on a collinear key that costs something the
    # reader must see. The same key fitted ALL AT ONCE on the first half and
    # frozen entire - the nuisance too, which `residual_floor.test` re-fits -
    # is the figure directly comparable with the ladder's final held-out
    # column, in the same floor units.
    reg = residual_floor.design(key, f)
    joint = residual_floor.fit(first["log"], first["floor"], reg["matrix"])
    joint_left = ho._rms((second["log"] - reg["matrix"] @ joint["coefficients"])
                         / np.sqrt(second["floor"]))
    columns = reg["matrix"][:, :len(nuisance)]
    only_nuisance = ho._rms(
        (second["log"] - columns @ residual_floor.fit(
            first["log"], first["floor"], columns)["coefficients"])
        / np.sqrt(second["floor"]))
    refit_nuisance = ho._rms(residual_floor.fit(
        second["log"], second["floor"], columns)["residual"]
        / np.sqrt(second["floor"]))
    print(f"  the ladder is greedy: {len(got['accepted'])} entries accepted "
          f"leave {got['final_held_out']:.1f} floor units, where the same key "
          f"fitted all at once on the\n  first half and frozen entire leaves "
          f"{joint_left:.1f} - held out just the same, so the gap is what the "
          f"entries carry in company and\n  cannot carry alone, on the "
          f"{bins / length['length']:.0f} independent members this export has")
    print(f"  freezing the nuisance costs nothing here: a level and delay "
          f"fitted on the first half leave {only_nuisance:.2f} on the second "
          f"against\n  {refit_nuisance:.2f} re-fitted there, so the two "
          f"halves share their gain and their timing and what the ladder "
          f"judges is shape")
    whole = residual_floor.test(f, m["H"], m["se"], key, halves)
    print(f"  for reference, residual_floor.test with the whole key at once "
          f"(nuisance re-fitted on the judged half): in-sample "
          f"{whole['in_sample']['reduced_chi_square']:.0f}, held-out "
          f"{whole['held_out']['reduced_chi_square']:.0f}, before "
          f"{whole['before']['reduced_chi_square']:.0f}")
    return got


def tesseract_reference(paths):
    """The same exports as the cube `tesseract.from_sync_exports` builds -
    head x polarity x half x tape - read by `tesseract.unreached`. The
    cube's HALF axis is this tool's split, so every identified contrast
    that involves the half is a part of the response no frozen parameter
    can carry from the first half to the second: the ceiling on what any
    ladder here can reach, measured by the fold rather than by a key."""
    tapes = list(paths)
    if len(tapes) < 2:
        return
    print("\n  for reference, the tesseract on each pair of tapes "
          "(tesseract.from_sync_exports, tesseract.unreached): the "
          "contrasts on the half axis,\n  as excess power over the floor, "
          "are what no parameter frozen on the first half can carry to the "
          "second")
    reading = ""
    for pair in itertools.combinations(tapes, 2):
        got = tesseract.unreached(tesseract.from_sync_exports(paths, pair))
        on_half = {":".join(subset): verdict
                   for subset, verdict in got["verdicts"].items()
                   if "half" in subset}
        text = ", ".join(
            f"{name} {verdict['excess_over_floor']:.0f}x"
            for name, verdict in sorted(
                on_half.items(),
                key=lambda item: -item[1]["excess_over_floor"]))
        print(f"    {pair[0]} x {pair[1]}: {got['identified_total']} of "
              f"{got['contrasts']} contrasts identified, top-order z "
              f"{got['top_order_z']:.0f}; on the half axis: {text}")
        reading = got["reading"]
    print(f"    {reading}")


def sync_ladders(args):
    banner("THE SYNC STEP RESPONSE: the first half of the fields fitted, "
           "the second judged, per head, the key in chain order")
    print("  the entries are a level and a delay (the nuisance terms) and "
          "then the chain's key in its declared order\n  (last applied, "
          "first removed), each as its log shape with one real weight; the "
          "values are ln|H| + j arg H\n  whitened by the half's standard "
          "error, so the residual rms reads in floor units and 1 is the "
          "floor")
    refused_everywhere = {}
    seen = 0
    paths = {}
    for tape in args.tapes:
        path = os.path.join(args.shared, f"{tape}{SYNC_SUFFIX}")
        if not os.path.exists(path):
            print(f"\n  {tape}: {path} not found, skipped")
            continue
        paths[tape] = path
        for head in ("a", "b"):
            got = sync_ladder(path, tape, head, args)
            if got is None:
                continue
            seen += 1
            for row in got["rows"]:
                if row["verdict"] != "accepted":
                    refused_everywhere.setdefault(row["entry"], []).append(
                        f"{tape}{head}")
    if refused_everywhere and seen:
        everywhere = sorted(name for name, places in refused_everywhere.items()
                            if len(places) == seen)
        somewhere = sorted((name, places) for name, places
                           in refused_everywhere.items()
                           if len(places) < seen)
        print(f"\n  refused on all {seen} head-tapes ({len(everywhere)} "
              f"entries): {', '.join(everywhere) or 'none'}")
        print("  accepted somewhere, refused elsewhere:")
        for name, places in somewhere:
            print(f"    {name:<52} refused on {' '.join(places)}")
        if not somewhere:
            print("    none")
    tesseract_reference(paths)


# --------------------------------------------------------------------------
# 2. The impulse exports
# --------------------------------------------------------------------------


def impulse_pooled_ladder(export, name, head, args):
    """The pooled per-head response over frequency, split over bands."""
    f = np.asarray(export["f_hz"], dtype=np.float64)
    tag = f"M1f_meas_h{head}"
    H = np.asarray(export[f"H_{tag}"], dtype=np.complex128)
    se = np.asarray(export[f"se_{tag}"], dtype=np.float64)
    coherence = np.asarray(export[f"coh_{tag}"], dtype=np.float64)
    valid = np.asarray(export[f"valid_{tag}"], dtype=bool)
    valid &= np.isfinite(H) & np.isfinite(se) & (se > 0)
    if valid.sum() < 2 * args.blocks:
        print(f"\n  {name} head {head} pooled: {valid.sum()} valid bins; "
              f"skipped")
        return
    grid, H, se = f[valid], H[valid], se[valid]
    log = residual_floor.log_domain(H, se)
    values = log["log"] / np.sqrt(log["floor"])
    # the resolution of the pooled response: the fall-edge window's samples
    # against the transform length, both the producer's
    rate = 2.0 * float(f[-1])
    n_fft = 2 * (f.size - 1)
    window_us = args.impulse_constants["fall_window_us"]
    window_samples = (window_us[1] - window_us[0]) * rate / 1e6
    length = n_fft / window_samples
    keys = residual_floor.keys_for(grid)
    key = keys[args.key]
    print(f"\n  {name} head {head} pooled response: {grid.size} valid bins "
          f"{grid[0] / 1e6:.2f}-{grid[-1] / 1e6:.2f} MHz (median coherence "
          f"{np.median(coherence[valid]):.2f}); resolution {length:.1f} bins "
          f"from a {window_samples / rate * 1e6:.1f} us window, about "
          f"{grid.size / length:.0f} independent members; {len(key)} key "
          f"entries ({args.key})")
    print("  no halves in this export: the split is over contiguous bands, "
          "which measures interpolation across frequency")
    nuisance = residual_floor.nuisance(grid)
    entries = [
        ho.constant_entry("level (nuisance: gain and phase offset)",
                          grid.size, complex_valued=True, floor=log["floor"]),
        ho.linear_entry("delay (nuisance: linear phase)", nuisance["delay"],
                        floor=log["floor"]),
    ]
    entries += ho.entries_from_key(key, chain=inf.full_chain(),
                                   floor=log["floor"])
    split_ = ho.split(grid.size, by="band", blocks=args.blocks, seed=args.seed)
    print(f"  {split_.description}")
    result = ho.ladder(entries, values, split_, draws=args.draws,
                       seed=args.seed, scramble=args.scramble,
                       correlation_length=length, sigma=args.sigma)
    show(f"{name} head {head} pooled fall response", result,
         "floor units: 1 = the standard error")


def impulse_time_ladder(export, name, head, args):
    """The per-line, per-field complex response at one reference frequency,
    split over contiguous blocks of fields."""
    reference = list(args.impulse_constants["reference_mhz"]).index(
        args.reference_mhz)
    lines = np.asarray(export[f"tdim_lines_M1f_h{head}"])
    matrix = np.asarray(export[f"tdim_H_M1f_h{head}"])
    per_field = matrix.shape[1]
    block = matrix[:, :, reference]
    table = np.where(np.isfinite(block),
                     np.log(np.abs(block) + 1e-300) + 1j * np.angle(block),
                     np.nan + 0j)
    values = table.ravel()
    field_of = np.tile(np.arange(per_field), lines.size)
    line_of = np.repeat(lines, per_field)
    heads = np.full(values.size, head)
    split_ = ho.split(fields=field_of, heads=heads, by="field",
                      blocks=args.blocks, seed=args.seed)
    # the noise's correlation along the member axis (line-major: consecutive
    # members are consecutive fields of one line), from the difference of
    # neighbouring lines' values - the same complex log the ladder sees -
    # which removes what the lines share
    length = ho.noise_correlation_length((table[1:] - table[:-1]).ravel())
    print(f"\n  {name} head {head} response at {args.reference_mhz} MHz: "
          f"{lines.size} lines x {per_field} fields (fall edge, measured "
          f"depth); noise correlation length {length['length']:.1f} members "
          f"from neighbouring lines' difference")
    print(f"  {split_.description}")
    droop = ho.scan_entry(
        "line droop along the field (A exp(-line/L))",
        lambda positions, span: np.exp(
            -(line_of[np.asarray(positions)] - lines.min()) / float(span)),
        np.geomspace(2.0, 4.0 * lines.size, 32), values.size)
    entries = [
        ho.constant_entry("level per head", values.size, complex_valued=True),
        droop,
        ho.template_entry("per-line template", line_of),
        ho.template_entry("per-field free gain (control)", field_of),
    ]
    result = ho.ladder(entries, values, split_, draws=args.draws,
                       seed=args.seed, scramble=args.scramble,
                       correlation_length=length["length"], sigma=args.sigma)
    show(f"{name} head {head} ln|H| + j arg H at {args.reference_mhz} MHz",
         result, "nepers + j radians")


def impulse_level_ladders(export, name, head, args):
    """The per-line levels, split over contiguous blocks of fields."""
    lines = np.asarray(export[f"tdim_lines_M1f_h{head}"])
    levels = np.asarray(export[f"levels_h{head}"], dtype=np.float64)
    if levels.shape[0] % lines.size:
        print(f"\n  {name} head {head} levels: {levels.shape[0]} rows is not "
              f"a whole number of {lines.size}-line fields; skipped")
        return
    fields_here = levels.shape[0] // lines.size
    field_of = np.repeat(np.arange(fields_here), lines.size)
    line_of = np.tile(lines, fields_here)
    heads = np.full(levels.shape[0], head)
    split_ = ho.split(fields=field_of, heads=heads, by="field",
                      blocks=args.blocks, seed=args.seed)
    column = {n: levels[:, k]
              for k, n in enumerate(args.impulse_constants["level_columns"])}
    centred = {n: v - np.mean(v) for n, v in column.items()}
    print(f"\n  {name} head {head} levels: {fields_here} fields x "
          f"{lines.size} lines (field-major); {split_.description}")

    def length_of(series):
        # consecutive fields at the same lines: their difference is noise
        # along the line axis, which is the member axis here
        table = series.reshape(fields_here, lines.size)
        return ho.noise_correlation_length(
            table[1:].ravel(), table[:-1].ravel())["length"]

    ladders = (
        # the front porch, with the preceding active level as its drive
        # (RINGING_RULES 19: the reaction to the preceding content is
        # legitimate calibration when the content is the stratification
        # variable) - the model of sync_geometry.front_porch_residual,
        # blank = settled + k (active - settled), as level + slope
        ("front porch", "blank",
         [ho.linear_entry("relaxation from the preceding active level "
                          "(slope on active)", centred["active"])]),
        # the tip against the back porch: clipping.clip_offset_from_porch
        # reads the slope of tip on porch; here that slope is frozen and
        # judged on held-out fields
        ("sync tip", "tip",
         [ho.linear_entry("follows the back porch (slope of tip on porch)",
                          centred["back"])]),
        ("back porch", "back", []),
    )
    for label, source, drives in ladders:
        series = column[source]
        length = length_of(series)
        entries = [ho.constant_entry("level per head", series.size)]
        entries += drives
        entries += [
            ho.template_entry("per-line template", line_of),
            ho.template_entry("per-field free gain (control)", field_of),
        ]
        result = ho.ladder(entries, series, split_, draws=args.draws,
                           seed=args.seed, scramble=args.scramble,
                           correlation_length=length, sigma=args.sigma)
        show(f"{name} head {head} {label} (IRE)", result, "IRE")


def impulse_ladders(args):
    banner("THE HSYNC IMPULSE EXPORTS: the pooled response over bands, and "
           "the per-line, per-field material over blocks of fields within a "
           "head")
    constants = args.impulse_constants
    print(f"  reference frequencies {constants['reference_mhz']} MHz, fall "
          f"window {constants['fall_window_us']} us, level columns "
          f"{constants['level_columns']}: "
          + (f"read from {IMPULSE_PRODUCER} beside the exports"
             if args.producer_read else
             "the producer is not beside the exports; the values recorded "
             "from it on 2026-09-05 are used"))
    for name in args.impulse:
        path = os.path.join(args.shared, f"{name}{IMPULSE_SUFFIX}")
        if not os.path.exists(path):
            print(f"\n  {name}: {path} not found, skipped")
            continue
        export = np.load(path)
        for head in (0, 1):
            if f"H_M1f_meas_h{head}" not in export.files:
                print(f"\n  {name} head {head}: no fall response; skipped")
                continue
            impulse_pooled_ladder(export, name, head, args)
            impulse_time_ladder(export, name, head, args)
            impulse_level_ladders(export, name, head, args)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="the held-out ladder on the real exports")
    parser.add_argument("--shared", default=SHARED,
                        help="directory holding the exports")
    parser.add_argument("--tapes", default=None,
                        help="step-response exports to run (comma list); "
                             f"default: every *{SYNC_SUFFIX} in --shared")
    parser.add_argument("--impulse", default=None,
                        help="impulse exports to run (comma list); default: "
                             f"every *{IMPULSE_SUFFIX} in --shared")
    parser.add_argument("--only", choices=("sync", "impulse"), action="append",
                        help="run only these sections (repeatable)")
    parser.add_argument("--key", choices=("admitted", "base"),
                        default="admitted",
                        help="the entry-wise admitted key (residual_floor."
                             "keys_for) or the base key")
    parser.add_argument("--view", choices=("fall", "rise"), default="fall",
                        help="which edge of the sync exports")
    parser.add_argument("--draws", type=int, default=24,
                        help="null draws per entry")
    parser.add_argument("--blocks", type=int, default=ho.DEFAULT_BLOCKS,
                        help="contiguous blocks per group")
    parser.add_argument("--sigma", type=float, default=ho.CHANCE_SIGMA,
                        help="spreads of the null an entry must clear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scramble", choices=("phase", "permute"),
                        default="phase")
    parser.add_argument("--reference-mhz", type=float, default=1.0,
                        help="which of the impulse export's reference "
                             "frequencies to read the time dimension at")
    args = parser.parse_args(argv)
    args.tapes = ([t for t in args.tapes.split(",") if t]
                  if args.tapes is not None
                  else discovered(args.shared, SYNC_SUFFIX))
    args.impulse = ([t for t in args.impulse.split(",") if t]
                    if args.impulse is not None
                    else discovered(args.shared, IMPULSE_SUFFIX))
    read = producer_constants(args.shared)
    args.producer_read = read is not None
    args.impulse_constants = read if read is not None else IMPULSE_RECORDED
    if args.reference_mhz not in args.impulse_constants["reference_mhz"]:
        raise SystemExit(f"--reference-mhz must be one of "
                         f"{args.impulse_constants['reference_mhz']}")
    warnings.simplefilter("ignore")
    wanted = set(args.only or ("sync", "impulse"))
    if "sync" in wanted:
        sync_ladders(args)
    if "impulse" in wanted:
        impulse_ladders(args)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
