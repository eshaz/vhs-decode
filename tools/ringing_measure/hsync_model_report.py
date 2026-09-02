"""Debug plots for the hsync artifact model's measurement layer.

Feeds a decoded .tbc through the module's own measurement code
(ringing_cancellation.measure_field_lines - the identical path the decoder
runs) and renders one figure per decode:

  1. the accumulated mean hsync interval at full scale, with every
     measurement point shaded and its measured level marked
  2. the same mean zoomed to blanking level - the artifact view: ringing,
     smear and boundary structure riding on the flat regions
  3. the per-lag cross-line variance (log scale) - where picture content
     and the color burst live, and the noise floor of the flat regions
  4. the entry-amplitude strata: the active falling edge's aftermath per
     amplitude, absolute and normalized by amplitude (curves that collapse
     in the normalized view = ringing proportional to the transient)
  5. the per-field health timeline (accumulated / rejected / reset)

Run it against UNCORRECTED decodes (--inverse_eq -1): the runtime measures
before any correction is applied, and these plots must see the same
signal.

Figures are written as PNG files next to the decodes, in
<RC_WORK>/hsync_plots/.

Usage: RC_WORK=<dir> python3 hsync_model_report.py <prefix>...
"""

import os
import sys
import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# resolve the repository root from this file's own location, so the
# module measured is the one in THIS tree (the installed console script
# silently resolves stale site-packages - never rely on cwd)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from vhsdecode.addons import ringing_cancellation as model
from vhsdecode.formats import get_format_params, parse_tape_speed

WORK_DIRECTORY = os.environ.get("RC_WORK", "/output/claude_validation")
PLOT_DIRECTORY = os.path.join(WORK_DIRECTORY, "hsync_plots")

# the runtime default of --inverse_eq when enabled, so the report's
# averaging horizon matches the decoder's
AVERAGE_FIELDS = 4

REGION_STYLE = {
    # measurement point   shade color   label
    "entry_context": ("#c8b4e6", "active area"),
    "front_porch": ("#a8d8a8", "front porch"),
    "sync_tip": ("#a8c4e0", "sync tip"),
    "back_porch": ("#a8d8a8", "back porch"),
}


def load_decode(prefix):
    """One decode as (fields x rows x samples) in IRE, plus its params."""
    with open(f"{WORK_DIRECTORY}/{prefix}.tbc.json") as handle:
        video_parameters = json.load(handle)["videoParameters"]
    width = video_parameters["fieldWidth"]
    height = video_parameters["fieldHeight"]
    blanking = video_parameters["black16bIre"]
    white = video_parameters["white16bIre"]
    units_per_ire = (white - blanking) / 100.0
    samples = np.fromfile(f"{WORK_DIRECTORY}/{prefix}.tbc", dtype="<u2")
    field_count = samples.size // (width * height)
    fields_ire = (samples[: field_count * width * height]
                  .reshape(field_count, height, width)
                  .astype(np.float64) - blanking) / units_per_ire
    return fields_ire


def measure_decode(prefix, geometry, window_plan):
    """Run the module's measurement over every field of one decode.

    Returns the final state, the per-field health timeline, and - when the
    decode ended after a reset with nothing re-accumulated - the state as
    it stood before the reset, so the plot can still show what had been
    measured.  (A reset empties the shared dictionary in place, but the
    arrays inside a shallow copy survive because the accumulator rebinds
    them rather than writing into them.)
    """
    fields_ire = load_decode(prefix)
    state = {}
    field_health = []
    state_before_reset = None
    for field in fields_ire:
        if state.get("fields_accumulated", 0) > 0:
            state_before_reset = dict(state)
        diagnostics = model.measure_field_lines(
            field, geometry, window_plan, state, AVERAGE_FIELDS)
        if state.get("slow_fields_accumulated", 0) >= AVERAGE_FIELDS:
            model.fit_artifact_model(
                state, geometry, window_plan, AVERAGE_FIELDS)
        if diagnostics["reset"]:
            field_health.append("reset")
        elif diagnostics["healthy"]:
            field_health.append("healthy")
        else:
            field_health.append("rejected")
    if state.get("fields_accumulated", 0) == 0 and state_before_reset:
        resets = state.get("resets", 0)
        state = state_before_reset
        state["resets"] = resets
        state["showing_pre_reset"] = True
    return state, field_health


def plot_decode(prefix, geometry, window_plan, state, field_health):
    """Render the module's debug figure for one decode and save it."""
    state = dict(state)
    state["health_history"] = field_health
    title = (f"{prefix} - hsync artifact model   "
             f"({sum(1 for h in field_health if h == 'healthy')}"
             f"/{len(field_health)} fields healthy, "
             f"{state.get('resets', 0)} resets, "
             f"{state.get('usable_lines_average', 0.0):.0f} lines/field)")
    if state.get("showing_pre_reset"):
        title += "   [state accumulated BEFORE the reset]"
    figure = model.render_debug_figure(state, geometry, window_plan, title)
    os.makedirs(PLOT_DIRECTORY, exist_ok=True)
    path = os.path.join(PLOT_DIRECTORY, f"{prefix}.png")
    figure.savefig(path, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    sys_params, decoder_params = get_format_params(
        "NTSC", "VHS", parse_tape_speed("sp"), logging.getLogger(__name__))
    geometry = model.build_geometry(
        sys_params, decoder_params, sys_params["outlinelen"],
        line_offset=0, line_count=263)
    window_plan = model.plan_measurement_window(geometry)
    for prefix in prefixes:
        state, field_health = measure_decode(prefix, geometry, window_plan)
        path = plot_decode(prefix, geometry, window_plan, state, field_health)
        healthy = sum(1 for entry in field_health if entry == "healthy")
        print(f"{prefix:20s} {healthy}/{len(field_health)} fields healthy, "
              f"{state.get('resets', 0)} resets  ->  {path}")
