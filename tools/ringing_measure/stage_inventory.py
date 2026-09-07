#!/usr/bin/env python3
"""EVERY STAGE WE HAVE DESIGNED, AND WHETHER IT IS ACTUALLY USED.

Ethan, 2026-09-06: *"Every stage that we have designed must be called and
used. Exhaustively go through the stages and make sure they are being
used. All of them need to be used and all of them need to model all
dimensions."*

And, on the same failure a moment earlier: *"This is very important and
you keep missing this part ... I can visibly see in the luma that this is
not happening."*

WHY THIS FILE EXISTS RATHER THAN A ONE-OFF CHECK. The gap between having
modelled something and having applied it is invisible from inside either
side: the model's own tests pass, the decode runs, and nothing anywhere
says the two never met. That is how the same gap was reported three times.
This makes the gap a measurement, so it can be printed, tested against and
watched, and `tests/unit/test_stage_inventory.py` fails when a module that
should reach a decode does not.

WHAT IT MEASURES, per module under `vhsdecode/models/`:

  REACHES      whether anything outside `vhsdecode/models/` imports it, which
               is the only definition of "used" that cannot be argued with.
  DECLARED     whether the pipeline declaration names it as a node, so it can
               be turned on and off by name as the graph requires.
  DIMENSIONS   which of amplitude, frequency and time the module names in its
               own returned keys. This is a coarse reading of the source and
               is meant to find the module that carries one axis where it
               should carry three, not to certify that three are correct.
  KIND         whether it is a STAGE, which belongs in a decode, or an
               INSTRUMENT, which measures and is offline by nature. The
               distinction is declared in `INSTRUMENTS` below rather than
               guessed, so that anything not named there is treated as a
               stage and has to be wired or explicitly excused.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "vhsdecode", "models")
DECLARATION = os.path.join(ROOT, "vhsdecode", "pipeline", "stages.toml")

# Modules that measure rather than correct, and are offline BY DESIGN. Each
# is named with the reason, because "it is only an instrument" is exactly
# the excuse that let a correction sit unwired for three sessions.
INSTRUMENTS: Dict[str, str] = {
    "algorithmic_information": "a definition of information, not a filter",
    "measurement_bound": "states what a measurement can resolve",
    "deck_log": "reads a hand-edited log of the bench",
    "tape_model": "a registry of names and confounds",
    "identifiability": "reports what a measurement cannot separate",
    "holdout": "the judge that decides what may be admitted",
    "injection": "plants a known signal to test an estimator",
    "content_independence": "checks a correction is not reading the picture",
    "loss_separation": "reports which mechanisms are distinguishable",
    "depth_split": "reports how a loss divides, offline",
    "modelable_subspace": "reports what remains when the model is exhausted",
    "lossless_model": "measures whether the residual compresses",
    "noise_budget": "accounts for the noise, offline",
    "rate_constraints": "predicts mechanical rates from the specification",
    "transport_dimensions": "reports what the transport can and cannot show",
    "vhs_specification": "the specification itself, a table of values",
    "profiles": "declares which stages exist for a source, not a stage",
    "stage_boundaries": "declares where one stage ends and the next begins",
    "reference_signal": "generates a reference to measure against",
    "correction_export": "writes measurements to a file",
    "hilbert_space_gp": "an interpolator used by other modules",
    "tesseract": "the fold, used by the stages rather than being one",
    "hypercomplex": "the transform, used by the stages rather than being one",
    "interference_distributions": "a table of distributions",
    "replay_split": "an offline experiment on repeated replay",
    "tap_transfer": "an offline measurement between two capture taps",
    "tape_tensor": "assembles measurements, offline",
    "pair_dimension": "checks a measurement has a partner",
    "chroma_stage_supersession": "evidence for retiring a stage, not a stage",
    "capture_chain_noise": "characterises the capture hardware",
    "modulation_noise": "an offline noise measurement",
    "hifi_carriers": "a separate audio path, deferred by direction",
    "capture_alignment": ("aligns two captures of one tape pass; the second "
                          "capture does not exist on this machine, and the "
                          "video side of it duplicates the head_switch_pair "
                          "node rather than adding a stage"),
    "information_extrapolation": "the component registry itself",
    "per_field_ringing": ("an offline PICTURE-domain gauge, and the standing "
                          "sync-only law forbids one from driving a "
                          "correction; its own subject is discharged - the "
                          "addon it named is deleted and the engine that "
                          "replaced it is one field at a time by "
                          "construction, measured as byte-identical output "
                          "at two different horizons"),
    "running_differential": "the constants rule, consulted by others",
    "sync_geometry": "the format's geometry, consulted by others",
    "output_limit": "states the output format's cap",
    "restoration_pipeline": "declares the order, does not correct",
    "edge_tracks": "asks whether the video head can read the linear audio "
                   "and control tracks; SMPTE 32M table 2 puts a 150 um "
                   "guard at both ends of the sweep and the measurement "
                   "found nothing across it, so there is nothing to correct "
                   "- see docs/EDGE_TRACKS.md",
}

AXIS_WORDS = {
    "amplitude": ("amplitude", "level", "gain", "ire", "_db", "magnitude"),
    "frequency": ("frequency", "_hz", "response", "spectrum", "band"),
    "time": ("delay", "seconds", "_s\"", "group_delay", "per_line",
             "per_field", "rate_hz", "time_"),
}


def modules() -> List[str]:
    return sorted(name[:-3] for name in os.listdir(MODELS)
                  if name.endswith(".py") and name != "__init__.py")


def _runtime_sources() -> List[str]:
    found = []
    for base, _, names in os.walk(os.path.join(ROOT, "vhsdecode")):
        if os.path.abspath(base).startswith(os.path.abspath(MODELS)):
            continue
        for name in names:
            if name.endswith(".py"):
                found.append(os.path.join(base, name))
    return found


def reached() -> Set[str]:
    """Which modules anything outside `models/` imports.

    Import is the only definition of "used" that cannot be argued with: a
    module nothing imports cannot run, whatever its docstring says.
    """
    text = ""
    for path in _runtime_sources():
        try:
            text += open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
    out = set()
    for name in modules():
        if re.search(r"models\s+import\s+[^\n]*\b%s\b" % re.escape(name), text) \
           or re.search(r"models\.%s\b" % re.escape(name), text):
            out.add(name)
    return out


def declared() -> Set[str]:
    """Which modules the pipeline declaration names, by entry point."""
    try:
        text = open(DECLARATION, encoding="utf-8").read()
    except OSError:
        return set()
    return {name for name in modules()
            if re.search(r"\b%s\b" % re.escape(name), text)}


def dimensions(name: str) -> Dict[str, bool]:
    """Which axes the module names in its own source, coarsely."""
    try:
        text = open(os.path.join(MODELS, name + ".py"), encoding="utf-8").read()
    except OSError:
        return {axis: False for axis in AXIS_WORDS}
    lowered = text.lower()
    return {axis: any(word in lowered for word in words)
            for axis, words in AXIS_WORDS.items()}


def survey() -> List[Dict[str, object]]:
    live, named = reached(), declared()
    rows = []
    for name in modules():
        axes = dimensions(name)
        rows.append({
            "module": name,
            "kind": "instrument" if name in INSTRUMENTS else "stage",
            "excuse": INSTRUMENTS.get(name),
            "reaches_the_runtime": name in live,
            "declared_as_a_node": name in named,
            "dimensions": axes,
            "axis_count": sum(axes.values()),
        })
    return rows


def unwired_stages() -> List[str]:
    """The stages that were designed and are not called. Ethan's question."""
    return [row["module"] for row in survey()
            if row["kind"] == "stage" and not row["reaches_the_runtime"]]


def one_dimensional() -> List[str]:
    """Modules naming fewer than all three axes, which he also asked about."""
    return [row["module"] for row in survey() if row["axis_count"] < 3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unwired", action="store_true",
                        help="print only the stages that are not called")
    args = parser.parse_args()
    rows = survey()
    if args.unwired:
        for name in unwired_stages():
            print(name)
        return 0
    stages = [r for r in rows if r["kind"] == "stage"]
    live = [r for r in stages if r["reaches_the_runtime"]]
    print(f"{len(rows)} modules: {len(stages)} stages, "
          f"{len(rows) - len(stages)} instruments")
    print(f"{len(live)} of {len(stages)} stages reach a decode\n")
    print(f"{'module':34s} {'kind':11s} {'runs':>5s} {'node':>5s} "
          f"{'A':>2s}{'F':>2s}{'T':>2s}")
    for row in rows:
        axes = row["dimensions"]
        print(f"{row['module']:34s} {row['kind']:11s} "
              f"{'yes' if row['reaches_the_runtime'] else 'NO':>5s} "
              f"{'yes' if row['declared_as_a_node'] else '-':>5s} "
              f"{'A' if axes['amplitude'] else '.':>2s}"
              f"{'F' if axes['frequency'] else '.':>2s}"
              f"{'T' if axes['time'] else '.':>2s}")
    missing = unwired_stages()
    if missing:
        print(f"\n{len(missing)} STAGES DESIGNED AND NOT CALLED:")
        for name in missing:
            print(f"   {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
