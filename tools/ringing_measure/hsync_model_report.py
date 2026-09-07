"""Debug plots for the hsync artifact model's measurement layer.

Feeds a decoded .tbc through the module's own measurement code
(`vhsdecode.models.ringing_tesseract` - the identical path the decoder
runs) and renders one figure per decode:

  1. the two collapsed kernels, the fall's landing on the sync tip against
     the rise's on blanking, with the colour burst's own lags shaded
  2. what the fold separated: every contrast's energy inside the burst
     lags against outside them, with the ones refused as chroma in red
  3. the held-out depth scan - what each extra line axis costs on lines
     the model was not fitted on
  4. the field's own summary

Run it against UNCORRECTED decodes (`--inverse_eq -1`): the runtime
measures before any correction is applied, and these plots must see the
same signal.

ONE FIELD AT A TIME. The model is per field by design, so this reports the
field the caller asks for (the first by default) rather than an average -
there is no accumulation left to show.

Figures are written as PNG files under <RC_WORK>/hsync_plots/.

Usage: RC_WORK=<dir> python3 hsync_model_report.py <prefix>...
"""

import json
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# resolve the repository root from this file's own location, so the
# module measured is the one in THIS tree (the installed console script
# silently resolves stale site-packages - never rely on cwd)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from vhsdecode.models import ringing_tesseract as model
from vhsdecode.formats import get_format_params, parse_tape_speed

WORK_DIRECTORY = os.environ.get("RC_WORK", "/output/claude_validation")
PLOT_DIRECTORY = os.path.join(WORK_DIRECTORY, "hsync_plots")


def load_decode(prefix):
    """One decode as (fields x rows x samples) in IRE, plus its geometry.

    The levels come from the decode's own black and white points, and the
    field is then re-anchored on the specification's only absolute scale -
    blanking at zero, the sync tip at minus the stated depth - measured
    from the field itself, which is what `process_field` receives at
    runtime from the decoder's reference levels.
    """
    with open(f"{WORK_DIRECTORY}/{prefix}.tbc.json") as handle:
        video_parameters = json.load(handle)["videoParameters"]
    width = video_parameters["fieldWidth"]
    height = video_parameters["fieldHeight"]
    black = video_parameters["black16bIre"]
    white = video_parameters["white16bIre"]
    units_per_ire = (white - black) / 100.0
    samples = np.fromfile(f"{WORK_DIRECTORY}/{prefix}.tbc", dtype="<u2")
    field_count = samples.size // (width * height)
    fields = (samples[: field_count * width * height]
              .reshape(field_count, height, width)
              .astype(np.float64) - black) / units_per_ire
    return fields, video_parameters


def geometry_for(video_parameters, height):
    system = video_parameters.get("system", "NTSC")
    sys_params, decoder_params = get_format_params(
        system, video_parameters.get("tapeFormat", "VHS"),
        parse_tape_speed("sp"), logging.getLogger(__name__))
    return model.build_geometry(
        sys_params, decoder_params, video_parameters["fieldWidth"],
        line_offset=0, line_count=height)


def anchor_field(field, geometry):
    """Blanking to zero and the sync tip to minus the stated depth."""
    edge = int(geometry.sync_transition_samples)
    tip = float(np.median(field[:, 2 * edge:
                                int(geometry.sync_pulse_samples) - edge]))
    porch = float(np.median(
        field[:, field.shape[1] - int(geometry.front_porch_samples):
              field.shape[1] - 2]))
    scale = (porch - tip) / geometry.sync_depth_ire
    return (field - porch) / scale


def measure_decode(prefix, field_number=0):
    """Run the module's own path over one field of one decode."""
    fields, video_parameters = load_decode(prefix)
    geometry = geometry_for(video_parameters, fields.shape[1])
    field = anchor_field(fields[min(field_number, len(fields) - 1)], geometry)
    events = model.measure_events(field, geometry)
    if events is None:
        return None
    depth = model.choose_depth(events, geometry)
    built = model.cube_from_events(events, geometry, depth["depth"])
    if built is None:
        return None
    cube, counts, bits = built
    collapse = model.collapse_to_one_signal(cube, geometry)
    return {
        "geometry": geometry, "events": events, "cube": cube,
        "counts": counts, "depth": depth, "collapse": collapse,
        "bank": model.kernel_bank(collapse, events, bits),
        "strength": model.correction_strength(depth["held_out_gain"]),
        "noise_ire": float(np.median(events.noise_ire)),
        "status": "measured", "head_name": "unknown",
    }


def plot_decode(prefix, state):
    """Render the module's debug figure for one decode and save it."""
    depth = state["depth"]
    title = (f"{prefix} - ringing on the tesseract graph   "
             f"({state['events'].count} lines, "
             f"{', '.join(depth['axes'])}, "
             f"held-out {depth['held_out_residual']:.4f})")
    figure = model.render_debug_figure(state, title)
    os.makedirs(PLOT_DIRECTORY, exist_ok=True)
    path = os.path.join(PLOT_DIRECTORY, f"{prefix}.png")
    figure.savefig(path, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    for prefix in sys.argv[1:]:
        state = measure_decode(prefix)
        if state is None:
            print(f"{prefix:20s} no usable sync events")
            continue
        path = plot_decode(prefix, state)
        print(f"{prefix:20s} {state['events'].count} lines, "
              f"held-out {state['depth']['held_out_residual']:.4f}, "
              f"strength {state['strength']:.3f}  ->  {path}")
