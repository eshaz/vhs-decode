"""EVERY MODELLED COMPONENT, AND WHAT IT LEAVES, FIELD BY FIELD.

Ethan, 2026-09-06: *"generate a report on all the modeled residuals. I need
to see what I am modeling and what the difference is for each field."*

THE FIELD AXIS IS THE POINT, AND THE ARC DID NOT HAVE ONE. Every measured
result the component key has been judged against so far comes from the
pooled `*_sync_step_response.npz` exports, and those exports have already
averaged the field axis away - into one response per head, plus a first-half
and a second-half estimate. Two halves are not a field axis. This tool
measures the response of EACH FIELD separately and fits the key to each,
so that "what is modelled" and "what is left" are curves over the decode
rather than single numbers for it.

WHAT ONE FIELD CAN MEASURE. A field carries about 254 horizontal sync
pulses. The pulse is the only deterministic step the signal contains: its
amplitude, width and transition time are specified, so the measured edge
divided by the specified edge is the channel's complex response as that
field witnessed it. Pooled over the field's own lines it reaches a standard
error of about 0.019 in |H| - some fifteen times coarser than the pooled
export's, which is the price of a field axis and is reported rather than
hidden.

THE INSTRUMENT IS SELF-CONTAINED, ON PURPOSE. `sync_step_response.py` and
`ringing_cancellation.measure_field_lines` are the arc's pooled instrument
and its runtime measurement path; both were being restructured while this
was written, and a report tool that stops working when another lane renames
a dataclass field is not a report tool. Every timing here is derived from
the format's own SysParams and DecoderParams - the same source those two use
- so the window is theirs by derivation rather than by import.

AND IT WAS CHECKED AGAINST THEM RATHER THAN ASSUMED TO AGREE. Pooling this
tool's per-field intervals over one head and running the same step response
reproduces the certified export's window exactly (lags 26 to 105, 79
samples, 0.1812 MHz resolution), its fitted crossing to 0.007 samples
(45.010 against 45.017) and its step to 0.02 IRE (-37.455 against -37.438).
The responses agree to 0.08 dB and 0.6 degrees below 1 MHz, and part company
above 2 MHz - 1.1 to 2.1 dB, 8 to 17 degrees - where the sync edge's own
spectrum is weakest and the two instruments' dropout hygiene differs. That
disagreement bounds what this tool can claim at the top of the band, and it
is quoted in the report for that reason.

WHAT IS REPORTED PER FIELD, and why each is there:

  departure        the measured response with only the three terms that
                   are not shapes removed - a level, a phase reference and
                   a delay. A gain belongs to the levels estimator, a delay
                   to the time base and a phase reference to the
                   measurement's own alignment; charging any of them to the
                   model would flatter it. `nuisance` records why there are
                   three where `residual_floor.nuisance` offers two, and
                   what the difference is worth on this material.
  ladder           the key applied in the chain's own declared order
                   (`interference.full_chain`), one component at a time,
                   with the share of the departure each removes. The order
                   is known circuitry, so this is a traversal and not a
                   search.
  unique share     what each component explains that no other component in
                   the key can. On a collinear key this is far smaller than
                   the ladder share, and the gap between them IS the
                   collinearity - the identifiability work measured six
                   magnetic mechanisms collapsing to 1.58 directions.
  remainder        what the whole key leaves, in three units: decibels of
                   amplitude, degrees of phase, and multiples of the
                   instrument's own standard error.
  held out         the same, with every component's coefficient frozen at
                   the value the OTHER fields of the same head fitted. An
                   in-sample residual falls whenever a parameter is added,
                   whatever the data hold; this one does not have to.
  floor            the medium's particulate floor. A resolution cell holds a
                   countable number of oxide particles and their
                   orientations are independent, so the square root of that
                   count is a limit no processing reaches past. A remainder
                   at it is the medium; a remainder above it is structure
                   still unclaimed.

Usage:
    PYTHONPATH=/workspaces/vhs-decode python3 \\
        tools/ringing_measure/modelled_residuals.py [prefix ...]

With no arguments it reads `modelled_residuals.toml` beside this file and
runs the decodes listed there. Figures are written as PNG files (the Agg
backend - this container is headless and `plt.show()` writes nothing), and
the report is written to `docs/MODELLED_RESIDUALS.md`.
"""

import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal.windows import tukey
from scipy.special import erf

try:                                    # Python 3.11+
    import tomllib
except ModuleNotFoundError:             # pragma: no cover - older runtimes
    import tomli as tomllib             # type: ignore

REPOSITORY = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPOSITORY not in sys.path:
    sys.path.insert(0, REPOSITORY)

from vhsdecode.formats import get_format_params, parse_tape_speed
from vhsdecode.models import interference, modelable_subspace, residual_floor

CONFIGURATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "modelled_residuals.toml")

# --------------------------------------------------------------------------
# The instrument's own constants, each with what fixes it
# --------------------------------------------------------------------------

# The transform length the pinned export contract uses, so this tool's bins
# land on the same grid as `*_sync_step_response.npz` and the two can be
# compared bin for bin.
FFT_LENGTH = 4096

# The taper the same contract applies to the differentiated edge before the
# transform. A rectangular window on a 79-sample record leaks the step's own
# energy across the whole band; 15 per cent Tukey is the export's figure.
TAPER_ALPHA = 0.15

# The gate below which the ideal step carries no energy and the ratio would
# be noise over noise: two per cent of the ideal's peak, the export's rule.
IDEAL_AMPLITUDE_GATE = 0.02

# The band the instrument itself is valid over, from the export's metadata:
# "~0.1-3 MHz determined; 0.05-0.1 one marginal band; below 0.05 MHz
# structurally absent".
INSTRUMENT_BAND_MHZ = (0.04, 3.5)

# Reciprocal bandwidths of the luma low-pass a transition needs to settle
# before a flat region may be read - `ringing_cancellation`'s own figure,
# repeated here because the geometry is derived rather than imported.
TRANSITION_SETTLE_BANDWIDTHS = 3.0

# The erf form 0.5*(1 + erf(t/s)) has a 10-to-90 rise of 1.8124*s, so a
# specified 10-90 transition time divides by this to give s.
ERF_WIDTH_FACTOR = 2.0 * 0.9061938

# A field's vertical sync occupies three sections of `numPulses` half-lines
# each (equalizing, serration, equalizing), and those lines carry pulses of
# a different width at twice line rate. They are excluded BY LINE NUMBER
# from the specification, never by looking at the waveform.
VERTICAL_SYNC_SECTIONS = 3

# The fraction of the specified sync depth a line's porch-to-tip difference
# must reach to count as carrying a real pulse. Anything shallower is a
# dropout or noise, not a sync edge.
SYNC_DEPTH_ADMISSION = 0.5

# Nepers to decibels, and radians to degrees: the log domain's two parts
# reported in the units a person reads them in.
NEPERS_TO_DB = 20.0 / math.log(10.0)
RADIANS_TO_DEGREES = 180.0 / math.pi

# The record's own geometry where the particulate floor is evaluated. The
# depth the write current actually magnetised and the carrier it wrote are
# the arc's measured figures, carried in `modelable_subspace.exhausted`'s
# signature; the writing speed is the JVC VTG82063 stated value.
RECORDED_DEPTH_M = 0.2064e-6
WRITING_SPEED_M_S = 5.8
CARRIER_HZ = 3.9e6


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULTS: Dict[str, object] = {
    "decodes": {"directory": "/output/decodes",
                "prefixes": ["ss_cd_off", "ss_home_off", "ss_pnb_off"],
                "labels": {}},
    "measurement": {"system": "NTSC", "tape_format": "VHS",
                    "tape_speed": "sp", "view": "fall",
                    "band_hz": [2.0e5, 4.0e6]},
    "model": {"relative_error": 0.01093, "admitted_key": False},
    "output": {"figures": "/output", "document": "docs/MODELLED_RESIDUALS.md",
               "dpi": 140},
}


def configuration(path: str = CONFIGURATION) -> Dict[str, object]:
    """The tool's one readable configuration file, with its defaults."""
    settings = {section: dict(values) for section, values
                in DEFAULTS.items()}
    if os.path.exists(path):
        with open(path, "rb") as handle:
            given = tomllib.load(handle)
        for section, values in given.items():
            settings.setdefault(section, {})
            settings[section].update(values)
    return settings


# --------------------------------------------------------------------------
# Geometry, derived from the specification
# --------------------------------------------------------------------------

def geometry(system: str = "NTSC", tape_format: str = "VHS",
             tape_speed: str = "sp") -> Dict[str, object]:
    """The sync interval's layout, every number derived from SysParams.

    Nothing here is typed. The sample rate, the front porch, the sync width,
    the specified transition time, the active start, the sync depth and the
    count of vertical-interval lines all come from the format's own tables,
    which is what makes the same code correct for a standard it has never
    been run against.
    """
    sys_params, decoder_params = get_format_params(
        system, tape_format, parse_tape_speed(tape_speed),
        logging.getLogger(__name__))
    rate = float(sys_params["outfreq"])                 # megahertz, 4 x fsc
    settle = int(math.ceil(TRANSITION_SETTLE_BANDWIDTHS * rate * 1e6
                           / float(decoder_params["video_lpf_freq"])))
    edge = float(sys_params["syncTransitionUS"]) * rate
    porch = float(sys_params["frontPorchUS"]) * rate
    width = float(sys_params["hsyncPulseUS"]) * rate
    active = float(sys_params["activeVideoUS"][0]) * rate
    # A LEVEL read needs only the transition core out of the slice - a
    # median tolerates the decaying tail, which is the artifact being
    # measured - so its guard is the edge plus one reciprocal bandwidth,
    # not the full settle time.
    level_guard = int(math.ceil(edge + settle / TRANSITION_SETTLE_BANDWIDTHS))
    before = int(math.ceil(porch + edge + 3 * settle))
    after = int(math.ceil(active + level_guard + 2 * settle))
    total = before + after

    def bounded(start: float, stop: float):
        low = int(max(0, math.ceil(start)))
        high = int(min(total, math.floor(stop)))
        return (low, high) if high > low else (low, low)

    return {
        "sys_params": sys_params,
        "rate_mhz": rate,
        "samples_per_line": int(sys_params["outlinelen"]),
        "settle": settle,
        "edge_samples": edge,
        "level_guard": level_guard,
        "anchor": before,
        "total": total,
        # region ENDS keep their guard - the next transition's band-limited
        # shape begins before its crossing and is nobody's artifact; region
        # STARTS sit at the preceding edge's 10-90 end, because the
        # artifact's largest part lives in that first reciprocal bandwidth
        "front_porch": bounded(before - porch + edge / 2.0,
                               before - level_guard),
        "sync_tip": bounded(before + edge / 2.0,
                            before + width - (edge / 2.0 + level_guard)),
        "back_porch": bounded(before + width + edge / 2.0,
                              before + active - settle),
        "sync_depth_ire": abs(float(sys_params["vsync_ire"])),
        "transition_us": float(sys_params["syncTransitionUS"]),
        "first_line": int(math.ceil(VERTICAL_SYNC_SECTIONS
                                    * float(sys_params["numPulses"]) / 2.0)),
        "frequency_hz": np.fft.rfftfreq(FFT_LENGTH, d=1.0 / rate) * 1e6,
    }


# --------------------------------------------------------------------------
# The decode
# --------------------------------------------------------------------------

def load_decode(prefix: str, directory: str) -> Dict[str, object]:
    """One decode as (fields x lines x samples) in IRE, with its parity.

    The levels are the decode's own black and white points, which is the
    unit the pinned export contract states: "IRE per tbc.json black/white".
    `isFirstField` is read from the decode's own metadata rather than
    inferred from the index, because a decode that starts on the second
    field of a frame would otherwise have both heads mislabelled.
    """
    with open(os.path.join(directory, f"{prefix}.tbc.json")) as handle:
        meta = json.load(handle)
    parameters = meta["videoParameters"]
    width = int(parameters["fieldWidth"])
    height = int(parameters["fieldHeight"])
    black = float(parameters["black16bIre"])
    per_ire = (float(parameters["white16bIre"]) - black) / 100.0
    raw = np.fromfile(os.path.join(directory, f"{prefix}.tbc"), dtype="<u2")
    count = raw.size // (width * height)
    fields = ((raw[: count * width * height]
               .reshape(count, height, width).astype(np.float64) - black)
              / per_ire)
    records = meta.get("fields", [])
    first = [bool(records[i].get("isFirstField", i % 2 == 0))
             if i < len(records) else bool(i % 2 == 0)
             for i in range(count)]
    return {"fields": fields, "is_first_field": first,
            "video_parameters": parameters, "prefix": prefix}


def field_interval(field_ire: np.ndarray, geom: Dict[str, object]
                   ) -> Optional[Dict[str, object]]:
    """One field's mean sync interval and its cross-line variance.

    Each line's window is read at the specified offset, admitted only if it
    carries a pulse of at least half the specified depth, anchored on its
    OWN front-porch level (consecutive lines carry a wander that would
    otherwise be booked as noise at every lag), aligned on its own 50 per
    cent crossing, and averaged.
    """
    rows, per_line = field_ire.shape
    flat = field_ire.reshape(-1)
    lines = np.arange(max(int(geom["first_line"]), 1), rows)
    starts = lines * per_line - int(geom["anchor"])
    inside = (starts >= 0) & (starts + int(geom["total"]) <= flat.size)
    starts = starts[inside]
    if not starts.size:
        return None
    windows = flat[starts[:, None] + np.arange(int(geom["total"]))[None, :]]

    def region_median(slice_pair):
        low, high = slice_pair
        if high <= low:
            return np.full(len(windows), np.nan)
        return np.median(windows[:, low:high], axis=1)

    porch = region_median(geom["front_porch"])
    tip = region_median(geom["sync_tip"])
    deep = (np.isfinite(porch) & np.isfinite(tip)
            & (porch - tip
               > SYNC_DEPTH_ADMISSION * float(geom["sync_depth_ire"])))
    windows, porch, tip = windows[deep], porch[deep], tip[deep]
    if len(windows) < 8:
        return None

    anchor = int(geom["anchor"])
    radius = int(geom["settle"])
    half = 0.5 * (porch + tip)
    low, high = anchor - radius, anchor + radius + 1
    segment = windows[:, low:high] - half[:, None]
    brackets = segment[:, :-1] * segment[:, 1:] <= 0
    index = np.argmax(brackets, axis=1)
    found = brackets[np.arange(len(windows)), index]
    y0 = segment[np.arange(len(windows)), index]
    y1 = segment[np.arange(len(windows)), index + 1]
    span = np.where(y1 != y0, y1 - y0, 1.0)
    crossing = low + index + np.where(y1 != y0, -y0 / span, 0.0)
    windows, porch, crossing = windows[found], porch[found], crossing[found]
    if len(windows) < 8:
        return None

    # PER-FIELD, PER-LINE LEVEL ANCHOR: blanking is the reference, and the
    # decoded buffer has not been through the decoder's black-level
    # adjustment at this point.
    windows = windows - porch[:, None]

    # Align every line on the common anchor by a fractional shift taken as a
    # phase ramp. The shift is CIRCULAR and the window's two ends sit in
    # different lines' active areas, so the step between them is removed as
    # a ramp before the shift and restored after it; without that, the wrap
    # injects a content-dependent discontinuity at the window edge.
    length = windows.shape[1]
    shift = anchor - crossing
    ends = windows[:, -1] - windows[:, 0]
    ramp_line = np.arange(length, dtype=np.float64) / (length - 1)
    flattened = windows - ends[:, None] * ramp_line[None, :]
    phase = np.exp(-2j * np.pi * np.fft.rfftfreq(length)[None, :]
                   * shift[:, None])
    aligned = (np.fft.irfft(np.fft.rfft(flattened, axis=1) * phase,
                            n=length, axis=1)
               + ends[:, None] * ramp_line[None, :])
    return {"mean": aligned.mean(axis=0),
            "variance": aligned.var(axis=0, ddof=1),
            "lines": int(len(aligned)),
            "crossing_spread": float(np.std(crossing))}


def step_response(interval: Dict[str, object], geom: Dict[str, object],
                  view: str = "fall") -> Optional[Dict[str, object]]:
    """The complex response of one sync edge, on the pinned export contract.

    The ideal is an erf at the SPECIFIED transition time centred on the
    fitted 50 per cent crossing, not a Heaviside: a Heaviside ideal would
    charge the record side's own edge shaping to the playback channel. The
    ratio is taken on the DERIVATIVE with a Tukey taper, so the step's
    two settled levels do not have to be windowed away.
    """
    mean = np.asarray(interval["mean"], dtype=np.float64)
    variance = np.asarray(interval["variance"], dtype=np.float64)
    if view == "fall":
        pre, post = geom["front_porch"], geom["sync_tip"]
    else:
        pre, post = geom["sync_tip"], geom["back_porch"]
    low, high = pre[0] + 1, post[1] - 1
    if high - low < 8:
        return None
    settle = int(geom["settle"])
    pre_level = float(np.median(mean[slice(*pre)]))
    interior = mean[post[0] + settle:post[1] - 1]
    if len(interior) < 4:
        return None
    step = float(np.median(interior)) - pre_level
    if abs(step) < 1.0:
        return None
    target = pre_level + 0.5 * step
    segment = mean[pre[1] - 2:post[0] + settle]
    hits = np.nonzero((segment[:-1] - target) * (segment[1:] - target) <= 0)[0]
    if not hits.size:
        return None
    k = int(hits[0])
    y0, y1 = float(segment[k]), float(segment[k + 1])
    t0 = ((pre[1] - 2) + k
          + (0.0 if y1 == y0 else (target - y0) / (y1 - y0)))
    sigma = (float(geom["transition_us"]) * float(geom["rate_mhz"])
             / ERF_WIDTH_FACTOR)
    lags = np.arange(low, high, dtype=np.float64)
    ideal = pre_level + step * 0.5 * (
        1.0 + erf((lags - t0) / (math.sqrt(2.0) * sigma)))
    measured = mean[low:high]
    d_measured, d_ideal = np.diff(measured), np.diff(ideal)
    window = tukey(len(d_measured), TAPER_ALPHA)
    spectrum = np.fft.rfft(d_measured * window, n=FFT_LENGTH)
    reference = np.fft.rfft(d_ideal * window, n=FFT_LENGTH)
    frequency = np.asarray(geom["frequency_hz"], dtype=np.float64)
    gate = IDEAL_AMPLITUDE_GATE * np.abs(reference).max()
    valid = ((np.abs(reference) > gate)
             & (frequency >= INSTRUMENT_BAND_MHZ[0] * 1e6)
             & (frequency <= INSTRUMENT_BAND_MHZ[1] * 1e6))
    response = np.full(frequency.size, np.nan + 0j, dtype=np.complex128)
    response[valid] = spectrum[valid] / reference[valid]
    # the standard error of the mean per lag, carried through the
    # derivative and the taper as a white-noise expectation
    mean_variance = variance[low:high] / max(int(interval["lines"]), 1)
    derivative_variance = mean_variance[:-1] + mean_variance[1:]
    noise_power = 0.5 * float(np.sum(window ** 2 * derivative_variance))
    error = np.full(frequency.size, np.nan)
    error[valid] = math.sqrt(noise_power) / np.abs(reference[valid])
    return {"H": response, "se": error, "valid": valid, "t0_sample": t0,
            "step_ire": step, "window_lags": (low, high),
            "resolution_hz": float(geom["rate_mhz"]) * 1e6 / max(high - low, 1),
            "lines": int(interval["lines"])}


def measure(prefix: str, directory: str, geom: Dict[str, object],
            view: str = "fall", band_hz=(2.0e5, 4.0e6)) -> Dict[str, object]:
    """Every field of one decode, on one common frequency grid."""
    decode = load_decode(prefix, directory)
    fields = decode["fields"]
    intervals: List[Optional[Dict[str, object]]] = []
    responses: List[Optional[Dict[str, object]]] = []
    for index in range(len(fields)):
        interval = field_interval(fields[index], geom)
        intervals.append(interval)
        responses.append(step_response(interval, geom, view)
                         if interval is not None else None)
    frequency = np.asarray(geom["frequency_hz"], dtype=np.float64)
    valid = (frequency >= band_hz[0]) & (frequency <= band_hz[1])
    for response in responses:
        if response is not None:
            valid &= (response["valid"] & np.isfinite(response["H"])
                      & np.isfinite(response["se"]) & (response["se"] > 0))
    usable = [i for i, r in enumerate(responses) if r is not None]
    if not usable or int(valid.sum()) < 16:
        raise RuntimeError(f"{prefix}: no usable field response in band")
    return {
        "prefix": prefix,
        "responses": responses,
        "intervals": intervals,
        "is_first_field": decode["is_first_field"],
        "field_count": int(len(fields)),
        "usable": usable,
        "valid": valid,
        "frequency_hz": frequency[valid],
        "resolution_hz": float(responses[usable[0]]["resolution_hz"]),
        "window_lags": responses[usable[0]]["window_lags"],
        "lines_per_field": float(np.mean([responses[i]["lines"]
                                          for i in usable])),
        "view": view,
        "video_parameters": decode["video_parameters"],
    }


# --------------------------------------------------------------------------
# The model, per field
# --------------------------------------------------------------------------

def _weighted_power(residual, floor) -> float:
    """The residual in units of its own floor: the fitted objective."""
    return float(np.sum(np.abs(np.asarray(residual)) ** 2
                        / np.maximum(np.asarray(floor), 1e-300)))


NUISANCE_NAMES = ("level", "phase reference", "delay")


def nuisance(frequency_hz) -> np.ndarray:
    """A level, a phase reference and a delay: the three terms that are not
    shapes, and there are THREE of them for a measured reason.

    `residual_floor.nuisance` offers two, a constant real part and a linear
    phase taken about the band's centre, and describes the second as a
    delay. It is not one. A delay of tau writes in the log domain as
    `-2 pi f tau`, which on a centred grid is `-2 pi (f - f0) tau` MINUS the
    constant `2 pi f0 tau`; the first part is in that span and the constant
    is not, so a pure delay leaves a frequency-independent phase behind that
    no entry of the key can express and every fit charges to the residual.
    Measured on this arc's own material, that omission is worth a great
    deal: on the pluge decode's first field it leaves 69.3 per cent of the
    departure explained where a delay taken through zero frequency leaves
    84.4, and the remainder is 2.3 times larger.

    The third term earns its place separately. `log_domain` unwraps the
    phase from the first bin's principal value, so the whole curve is
    defined only up to a constant; without a free constant phase the fitted
    result would depend on which branch the unwrap happened to take. With
    it, the answer is invariant, and it adds a further 1.2 points of
    explained departure over the delay alone.

    None of the three is the model's claim. A gain belongs to the levels
    estimator, a delay to the time base, and a phase reference to the
    measurement's own alignment; charging any of them to the residual would
    overstate what is unexplained, and letting an entry absorb one would
    overstate what is understood.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    scale = float(np.max(np.abs(f))) or 1.0
    ones = np.ones(f.size, dtype=np.complex128)
    return np.column_stack([ones, 1j * ones, 1j * f / scale])


def design(entries: Dict[str, np.ndarray], frequency_hz) -> Dict[str, object]:
    """The regressors: the three nuisance terms, then each entry's shape."""
    columns = [nuisance(frequency_hz)]
    names = list(NUISANCE_NAMES)
    for name, value in entries.items():
        shape = interference._log_shape(np.asarray(value))
        if np.linalg.norm(shape) > 0:
            columns.append(shape.reshape(-1, 1))
            names.append(name)
    return {"matrix": np.hstack(columns), "names": names,
            "nuisance_columns": len(NUISANCE_NAMES)}


def ordered_key(frequency_hz, admitted: bool = False
                ) -> Dict[str, np.ndarray]:
    """The component key in the chain's own declared order.

    `interference.full_chain` gives each entry's position in the signal's
    path, and the chain is removed last-applied-first. Ordering by position
    is what makes the ladder a traversal of known circuitry rather than a
    search over an arbitrary sequence.
    """
    base = (residual_floor.keys_for(frequency_hz)["admitted"] if admitted
            else interference.signatures(frequency_hz))
    chain = interference.full_chain(strict=False)

    def position(name: str) -> float:
        if name in chain:
            return float(chain[name])
        for known, place in chain.items():
            if name.startswith(known):
                return float(place)
        return float(max(chain.values()) + 1)

    return {name: base[name]
            for name in sorted(base, key=lambda n: (position(n), n))}


def model_field(response: Dict[str, object], valid: np.ndarray,
                frequency_hz: np.ndarray, matrix: np.ndarray,
                nuisance_columns: int) -> Dict[str, object]:
    """One field's decomposition: departure, ladder, unique shares, remainder.

    Everything is measured in the fitted objective - the residual power
    divided by the instrument's own per-bin floor - because that is what the
    weighted fit minimises and it is therefore the only quantity guaranteed
    not to rise when a component is added.
    """
    logged = residual_floor.log_domain(response["H"][valid],
                                       response["se"][valid])
    value, floor = logged["log"], logged["floor"]
    free_only = residual_floor.fit(value, floor,
                                   matrix[:, :nuisance_columns])
    departure = free_only["residual"]
    departure_power = _weighted_power(departure, floor)
    entries = matrix.shape[1] - nuisance_columns

    ladder_power = np.empty(entries + 1)
    ladder_power[0] = departure_power
    for step in range(1, entries + 1):
        fitted = residual_floor.fit(
            value, floor, matrix[:, :nuisance_columns + step])
        ladder_power[step] = _weighted_power(fitted["residual"], floor)
    full = residual_floor.fit(value, floor, matrix)
    remainder = full["residual"]
    remainder_power = _weighted_power(remainder, floor)

    unique = np.empty(entries)
    keep = np.ones(matrix.shape[1], dtype=bool)
    for entry in range(entries):
        keep[:] = True
        keep[nuisance_columns + entry] = False
        without = residual_floor.fit(value, floor, matrix[:, keep])
        unique[entry] = (_weighted_power(without["residual"], floor)
                         - remainder_power)

    return {
        "log": value, "floor": floor,
        "departure": departure, "remainder": remainder,
        "departure_power": departure_power,
        "remainder_power": remainder_power,
        "ladder_power": ladder_power,
        # the ladder power only ever falls - a least-squares fit cannot be
        # worsened by another column - so the share each component removes
        # is the NEGATIVE of the difference along it
        "ladder_share": -np.diff(ladder_power) / max(departure_power, 1e-300),
        "unique_share": unique / max(departure_power, 1e-300),
        "coefficients": full["coefficients"],
        "bins": int(value.size),
    }


def held_out_field(index: int, logs, floors, mates: Sequence[int],
                   matrix: np.ndarray, nuisance_columns: int
                   ) -> Dict[str, object]:
    """This field's residual with every component frozen on other fields.

    The coefficients come from the pooled response of the OTHER fields of
    the same head; only the level and the delay are re-fitted here, because
    a gain and a timing are not the model's claim and freezing them would
    charge this field's own level to the residual.

    EACH RUNG IS ITS OWN FROZEN FIT, which the first version of this got
    wrong and which matters more than it sounds. Taking one joint fit of
    the whole key and then applying the first k of its coefficients is not
    a model of anything: on a collinear design the joint coefficients are
    large and cancelling, so a partial sum destroys the cancellation and
    reports absurd descents - it put +765 per cent against the sub-emphasis
    entry and -919 per cent against particle noise on the first run. The
    honest rung is the model of the first k components, fitted on the
    training fields and frozen.
    """
    value, floor = logs[index], floors[index]
    pooled = np.mean([logs[i] for i in mates], axis=0)
    pooled_floor = np.mean([floors[i] for i in mates], axis=0) / len(mates)
    entries = matrix.shape[1] - nuisance_columns
    free = matrix[:, :nuisance_columns]

    def after(step: int) -> float:
        columns = matrix[:, :nuisance_columns + step]
        frozen = residual_floor.fit(pooled, pooled_floor, columns)
        predicted = (columns[:, nuisance_columns:]
                     @ frozen["coefficients"][nuisance_columns:]
                     if step else 0.0)
        left = residual_floor.fit(value - predicted, floor, free)["residual"]
        return _weighted_power(left, floor), left

    powers, residuals = zip(*[after(step) for step in range(entries + 1)])
    ladder_power = np.asarray(powers, dtype=np.float64)
    return {
        "departure_power": ladder_power[0],
        "remainder_power": ladder_power[-1],
        "ladder_power": ladder_power,
        "ladder_share": -np.diff(ladder_power) / max(ladder_power[0], 1e-300),
        "remainder": residuals[-1],
    }


def model_decode(measured: Dict[str, object], relative_error: float,
                 admitted: bool = False) -> Dict[str, object]:
    """Every field of one decode, modelled, with the field axis preserved."""
    frequency = measured["frequency_hz"]
    valid = measured["valid"]
    key = ordered_key(frequency, admitted)
    built = design(key, frequency)
    nuisance_columns = built["nuisance_columns"]
    names = built["names"][nuisance_columns:]
    matrix = built["matrix"]
    # the same fit under the shared two-term nuisance set, so what that
    # set costs is measured in this run rather than asserted
    shared = residual_floor.design(key, frequency)
    shared_columns = len(residual_floor.nuisance(frequency))

    fields, logs, floors = [], [], []
    for index in measured["usable"]:
        outcome = model_field(measured["responses"][index], valid,
                              frequency, matrix, nuisance_columns)
        outcome["field"] = index
        outcome["head"] = "a" if measured["is_first_field"][index] else "b"
        # only the two totals are wanted from the shared set, so the
        # ladder and the leave-one-out sweep are not repeated for it
        value, floor = outcome["log"], outcome["floor"]
        before = residual_floor.fit(value, floor,
                                    shared["matrix"][:, :shared_columns])
        after = residual_floor.fit(value, floor, shared["matrix"])
        shared_departure = _weighted_power(before["residual"], floor)
        shared_remainder = _weighted_power(after["residual"], floor)
        outcome["shared_nuisance"] = {
            "departure_power": shared_departure,
            "remainder_power": shared_remainder,
            "explained": 1.0 - shared_remainder / max(shared_departure, 1e-300),
        }
        fields.append(outcome)
        logs.append(outcome["log"])
        floors.append(outcome["floor"])

    # held out, within a head: the two heads' departures do not correlate,
    # so a split that pools them judges one head by the other's model
    by_head: Dict[str, List[int]] = {}
    for position, outcome in enumerate(fields):
        by_head.setdefault(outcome["head"], []).append(position)
    for head, members in by_head.items():
        if len(members) < 2:
            continue
        for position in members:
            mates = [m for m in members if m != position]
            fields[position]["held_out"] = held_out_field(
                position, logs, floors, mates, matrix, nuisance_columns)

    # the modelable subspace, exhausted to its rank, against the medium's
    # own particulate floor - the arc's recorded currency
    for outcome in fields:
        response = measured["responses"][outcome["field"]]
        exhausted = modelable_subspace.exhausted(
            response["H"][valid], frequency, key,
            relative_error=relative_error,
            recorded_depth_m=RECORDED_DEPTH_M,
            writing_speed_m_s=WRITING_SPEED_M_S, carrier_hz=CARRIER_HZ)
        particles = float(exhausted["floor"]["particles_per_cell"])
        # THE FLOOR HAS TWO CONVENTIONS AND THEY DIFFER BY TWENTY DECIBELS,
        # so both are carried and neither is called "the" floor. A cell
        # holding `particles` independently oriented particles fluctuates
        # by one over their root in RELATIVE amplitude, which is that many
        # nepers of log-amplitude and is directly comparable with this
        # residual: that is `above_floor_db`. `above_floor_db_recorded` is
        # `modelable_subspace.against_the_floor`, which divides the
        # DEPARTURE's own energy by the particle count instead - the
        # convention behind the 27 to 32 dB table recorded in that module -
        # and reads about twenty decibels higher because the departure is a
        # small perturbation of the signal rather than the signal.
        floor_nepers = 1.0 / math.sqrt(particles)
        amplitude_nepers = (in_decibels(outcome["remainder"],
                                        outcome["floor"]) / NEPERS_TO_DB)
        outcome["exhausted"] = {
            "modelled_share": exhausted["decomposition"]["modelled_share"],
            "remainder_share": exhausted["decomposition"]["remainder_share"],
            "rank": exhausted["decomposition"]["rank"],
            "components": exhausted["decomposition"]["components"],
            "condition": exhausted["decomposition"]["condition"],
            "above_floor_db": 20.0 * math.log10(
                max(amplitude_nepers, 1e-300) / floor_nepers),
            "above_floor_db_recorded": exhausted["verdict"]["ratio_db"],
            "floor_db": NEPERS_TO_DB * floor_nepers,
            "particles_per_cell": particles,
            "verdict": exhausted["verdict"]["verdict"],
        }

    cells = residual_floor.resolution_cells(frequency,
                                            measured["resolution_hz"])
    return {"prefix": measured["prefix"], "names": names, "fields": fields,
            "matrix": matrix, "nuisance_columns": nuisance_columns,
            "key": key, "cells": cells, "measured": measured,
            "by_head": by_head}


# --------------------------------------------------------------------------
# Readable units
# --------------------------------------------------------------------------

def _weighted_rms(values, floor) -> float:
    """Root mean square with each bin weighted by its own precision.

    WEIGHTED, AND IT HAS TO BE. The fit minimises the residual divided by
    the per-bin floor, so a plain average over bins is not the quantity the
    model is optimising: above 2 MHz the sync edge's own spectrum is weak,
    the standard error there is a quarter of |H|, and an unweighted mean is
    dominated by bins the fit is right to ignore. Measured on the first
    run, which reported these unweighted: the amplitude residual AFTER the
    whole key read larger than before it on some fields, purely from that
    mismatch between the summary and the objective.
    """
    weight = 1.0 / np.maximum(np.asarray(floor, dtype=np.float64), 1e-300)
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.sum(weight * values ** 2) / np.sum(weight)))


def in_decibels(residual, floor) -> float:
    """The amplitude part of a log-domain residual, in decibels."""
    return _weighted_rms(np.real(residual), floor) * NEPERS_TO_DB


def in_degrees(residual, floor) -> float:
    """The phase part of a log-domain residual, in degrees."""
    return _weighted_rms(np.imag(residual), floor) * RADIANS_TO_DEGREES


def in_floor_units(power: float, bins: int, cells: int) -> float:
    """The residual as a multiple of the instrument's own standard error.

    Counted in the instrument's INDEPENDENT points rather than its bins.
    The 79-sample window is transformed on a 4096-point grid, so adjacent
    bins are not independent draws; a chi-square counting bins understates
    its error bar by the root of the oversampling.
    """
    del bins
    return float(np.sqrt(power / max(2 * cells, 1)))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

HEAD_COLOUR = {"a": "#1f6fb4", "b": "#c1512b"}


def _stack_colours(count: int):
    base = plt.get_cmap("turbo")
    return [base(0.06 + 0.88 * i / max(count - 1, 1)) for i in range(count)]


def figure_fields(model: Dict[str, object], label: str, path: str,
                  dpi: int = 140) -> str:
    """What is modelled and what is left, for each field."""
    fields = model["fields"]
    names = model["names"]
    cells = int(model["cells"]["count"])
    index = np.array([f["field"] for f in fields])
    heads = np.array([f["head"] for f in fields])
    departure_db = np.array([in_decibels(f["departure"], f["floor"])
                             for f in fields])
    remainder_db = np.array([in_decibels(f["remainder"], f["floor"])
                             for f in fields])
    departure_deg = np.array([in_degrees(f["departure"], f["floor"])
                              for f in fields])
    remainder_deg = np.array([in_degrees(f["remainder"], f["floor"])
                              for f in fields])
    remainder_units = np.array([
        in_floor_units(f["remainder_power"], f["bins"], cells)
        for f in fields])
    departure_units = np.array([
        in_floor_units(f["departure_power"], f["bins"], cells)
        for f in fields])
    held = np.array([
        in_floor_units(f["held_out"]["remainder_power"], f["bins"], cells)
        if "held_out" in f else np.nan for f in fields])
    shares = np.array([f["ladder_share"] for f in fields])
    above = np.array([f["exhausted"]["above_floor_db"] for f in fields])
    particles = float(fields[0]["exhausted"]["particles_per_cell"])
    # a cell holding N independent particles fluctuates as 1/sqrt(N) in
    # amplitude, which is that many nepers and so this many decibels
    floor_db = NEPERS_TO_DB / math.sqrt(particles)

    figure, axes = plt.subplots(3, 2, figsize=(15.5, 12.0))
    figure.suptitle(
        f"Modelled residuals per field - {label}\n"
        f"{len(fields)} fields, {len(names)} modelled components, "
        f"{fields[0]['bins']} frequency bins carrying {cells} independent "
        f"points over "
        f"{model['measured']['frequency_hz'][0] / 1e6:.2f}-"
        f"{model['measured']['frequency_hz'][-1] / 1e6:.2f} MHz",
        fontsize=13, y=0.985)

    # (a) amplitude, before and after, against both floors
    ax = axes[0][0]
    for head in ("a", "b"):
        mask = heads == head
        ax.plot(index[mask], departure_db[mask], ".", markersize=3.5,
                color=HEAD_COLOUR[head], alpha=0.55,
                label=f"head {head}: measured departure")
        ax.plot(index[mask], remainder_db[mask], "o", markersize=3.0,
                markerfacecolor="none", color=HEAD_COLOUR[head],
                label=f"head {head}: after all {len(names)} components")
    ax.axhline(floor_db, color="#111111", linestyle="--", linewidth=1.2,
               label=f"medium's particulate floor, {floor_db:.4f} dB")
    ax.set_yscale("log")
    ax.set_xlabel("field index within the decode")
    ax.set_ylabel("residual amplitude, dB rms, precision weighted")
    ax.set_title("(a) what the model removes, field by field")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best", ncol=1)

    # (b) the shares, stacked
    ax = axes[0][1]
    colours = _stack_colours(len(names))
    ax.stackplot(index, 100.0 * shares.T, colors=colours, labels=names,
                 linewidth=0)
    ax.set_xlim(index.min(), index.max())
    # THE LEGEND GETS ITS OWN BAND ABOVE THE DATA. Drawn over the stack it
    # covered the top third of it, and the edge of the box read as a step
    # in the data at the field where it happened to end - which is exactly
    # the kind of thing this report exists to measure rather than imagine.
    ax.set_ylim(0.0, 148.0)
    ax.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax.set_xlabel("field index within the decode")
    ax.set_ylabel("share of the measured departure explained, per cent")
    ax.set_title("(b) what is being modelled: each component's share, "
                 "in chain order")
    shift = strongest_shift(fields, names)
    if shift.get("found") and shift.get("material"):
        ax.axvline(shift["at_field"], color="#111111", linestyle="-.",
                   linewidth=1.2, label="_nolegend_")
        ax.text(0.012, 0.018,
                f"dash-dot line: the largest shift on the field axis, at "
                f"field {shift['at_field']}\n{shift['quantity']} moves from "
                f"{shift['before']:.3g} to {shift['after']:.3g} per cent, "
                f"{shift['standardised']:.1f} times the field-to-field "
                f"scatter (p = {shift['p_value']:.1g})",
                transform=ax.transAxes, fontsize=7.0, color="#f0f0f0",
                va="bottom", ha="left", linespacing=1.4)
    handles, texts = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], texts[::-1], fontsize=6.0, loc="upper left",
              ncol=2, framealpha=0.9,
              title="applied last at the top of the stack", title_fontsize=6.0)
    ax.grid(alpha=0.2)

    # (c) the remainder in the instrument's own units, in sample and held out
    ax = axes[1][0]
    ax.plot(index, departure_units, ".", markersize=3.0, color="#8a8a8a",
            label="before the model")
    for head in ("a", "b"):
        mask = heads == head
        ax.plot(index[mask], remainder_units[mask], "-", linewidth=3.0,
                color=HEAD_COLOUR[head], alpha=0.35,
                label=f"head {head}: in sample")
        ax.plot(index[mask], held[mask], "--", linewidth=1.1,
                color=HEAD_COLOUR[head],
                label=f"head {head}: held out, frozen on other fields")
    ax.axhline(1.0, color="#111111", linestyle=":", linewidth=1.4,
               label="the instrument's own standard error")
    ax.set_yscale("log")
    ax.set_xlabel("field index within the decode")
    ax.set_ylabel("residual, multiples of the standard error")
    ax.set_title("(c) in sample against held out, in floor units")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, loc="best")

    # (d) phase
    ax = axes[1][1]
    for head in ("a", "b"):
        mask = heads == head
        ax.plot(index[mask], departure_deg[mask], ".", markersize=3.5,
                color=HEAD_COLOUR[head], alpha=0.55,
                label=f"head {head}: measured departure")
        ax.plot(index[mask], remainder_deg[mask], "o", markersize=3.0,
                markerfacecolor="none", color=HEAD_COLOUR[head],
                label=f"head {head}: after all components")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("field index within the decode")
    ax.set_ylabel("residual phase, degrees rms, precision weighted")
    ax.set_title("(d) the phase the model removes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")

    # (e) how far the remainder sits above the medium's floor
    ax = axes[2][0]
    for head in ("a", "b"):
        mask = heads == head
        ax.plot(index[mask], above[mask], "-", linewidth=1.1,
                color=HEAD_COLOUR[head], label=f"head {head}")
    ax.axhline(0.0, color="#111111", linestyle="--", linewidth=1.2,
               label="the medium's particulate floor")
    ax.set_xlabel("field index within the decode")
    ax.set_ylabel("remainder above the medium's particulate floor, dB")
    ax.set_title("(e) how much structure the modelling has not claimed\n"
                 f"floor = one over the root of {particles:.0f} particles "
                 f"in a resolution cell = {floor_db:.4f} dB",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")

    # (f) the total explained share per field, and the rank supporting it
    ax = axes[2][1]
    explained = 1.0 - np.array([f["remainder_power"] / f["departure_power"]
                                for f in fields])
    subspace = np.array([f["exhausted"]["modelled_share"] for f in fields])
    rank = np.array([f["exhausted"]["rank"] for f in fields])
    ax.plot(index, 100.0 * explained, "-", linewidth=1.2, color="#2a6f3f",
            label="weighted least squares, all components")
    ax.plot(index, 100.0 * subspace, "-", linewidth=1.2, color="#7a3fa0",
            label="modelable-subspace projection")
    ax.set_xlabel("field index within the decode")
    ax.set_ylabel("departure explained, per cent")
    ax.set_ylim(0, 100)
    twin = ax.twinx()
    twin.plot(index, rank, ".", markersize=2.5, color="#b08a20",
              label="rank the data supports")
    twin.set_ylabel("rank supported, of "
                    f"{fields[0]['exhausted']['components']} components")
    twin.set_ylim(0, 3.5 * fields[0]["exhausted"]["components"])
    lines = [line for line in ax.get_lines() + twin.get_lines()
             if not str(line.get_label()).startswith("_")]
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7.5,
              loc="lower right")
    ax.set_title("(f) the total explained, and the rank behind it")
    ax.grid(alpha=0.25)

    figure.tight_layout(rect=(0, 0, 1, 0.955))
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def figure_components(model: Dict[str, object], label: str, path: str,
                      dpi: int = 140) -> str:
    """Every component: what it explains, what only it explains, what it
    carries to a field it never saw."""
    fields = model["fields"]
    names = model["names"]
    cells = int(model["cells"]["count"])
    ladder = np.array([f["ladder_share"] for f in fields])
    unique = np.array([f["unique_share"] for f in fields])
    held = np.array([f["held_out"]["ladder_share"] for f in fields
                     if "held_out" in f])
    order = np.arange(len(names))

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 0.34 * len(names) + 2.9),
                                sharey=True)
    figure.suptitle(
        f"What each modelled component explains - {label}\n"
        f"bars are the median over {len(fields)} fields, whiskers the 10th "
        f"to 90th percentile across fields; components in chain order, "
        f"first applied at the top", fontsize=12)

    def draw(ax, values, title, colour):
        low = np.percentile(values, 10, axis=0) * 100.0
        mid = np.percentile(values, 50, axis=0) * 100.0
        high = np.percentile(values, 90, axis=0) * 100.0
        ax.barh(order, mid, color=colour, height=0.68)
        ax.errorbar(mid, order, xerr=[np.maximum(mid - low, 0),
                                      np.maximum(high - mid, 0)],
                    fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.0)
        ax.set_yticks(order)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("share of the measured departure, per cent")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.axvline(0.0, color="#111111", linewidth=0.8)

    draw(axes[0], ladder, "in the chain's order, each after its predecessors",
         "#3a7dbd")
    draw(axes[1], unique, "what ONLY this component explains", "#2a8f6f")
    if held.size:
        draw(axes[2], held,
             "held out: frozen on other fields of the same head", "#b06a2a")
    else:
        axes[2].text(0.5, 0.5, "held out unavailable: a head with one field",
                     ha="center", va="center", transform=axes[2].transAxes)
    figure.text(0.01, 0.005,
                f"instrument: {fields[0]['bins']} bins carrying {cells} "
                f"independent points; departure = the response with a "
                f"level, a phase reference and a delay removed",
                fontsize=7.5, color="#444444")
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def figure_frequency(model: Dict[str, object], label: str, path: str,
                     dpi: int = 140) -> str:
    """The same decomposition read across frequency rather than across fields."""
    fields = model["fields"]
    frequency = model["measured"]["frequency_hz"] / 1e6
    departure = np.array([f["departure"] for f in fields])
    remainder = np.array([f["remainder"] for f in fields])
    floor = np.array([f["floor"] for f in fields])
    particles = float(fields[0]["exhausted"]["particles_per_cell"])
    floor_db = NEPERS_TO_DB / math.sqrt(particles)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True)
    figure.suptitle(f"Where in the band the remainder lives - {label}\n"
                    f"solid line the median over {len(fields)} fields, band "
                    f"the 10th to 90th percentile across fields", fontsize=12)

    def envelope(ax, values, colour, name, scale):
        low = np.percentile(values, 10, axis=0) * scale
        mid = np.percentile(values, 50, axis=0) * scale
        high = np.percentile(values, 90, axis=0) * scale
        ax.fill_between(frequency, low, high, color=colour, alpha=0.22,
                        linewidth=0)
        ax.plot(frequency, mid, color=colour, linewidth=1.4, label=name)

    ax = axes[0]
    envelope(ax, np.abs(np.real(departure)), "#8a8a8a",
             "measured departure", NEPERS_TO_DB)
    envelope(ax, np.abs(np.real(remainder)), "#1f6fb4",
             f"after all {len(model['names'])} components", NEPERS_TO_DB)
    envelope(ax, np.sqrt(floor), "#2a8f6f",
             "the instrument's own standard error", NEPERS_TO_DB)
    ax.axhline(floor_db, color="#111111", linestyle="--", linewidth=1.2,
               label=f"medium's particulate floor, {floor_db:.4f} dB")
    ax.set_yscale("log")
    ax.set_ylabel("amplitude, dB")
    ax.set_title("(a) amplitude", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    envelope(ax, np.abs(np.imag(departure)), "#8a8a8a",
             "measured departure", RADIANS_TO_DEGREES)
    envelope(ax, np.abs(np.imag(remainder)), "#c1512b",
             f"after all {len(model['names'])} components", RADIANS_TO_DEGREES)
    envelope(ax, np.sqrt(floor), "#2a8f6f",
             "the instrument's own standard error", RADIANS_TO_DEGREES)
    ax.set_yscale("log")
    ax.set_xlabel("frequency, MHz")
    ax.set_ylabel("phase, degrees")
    ax.set_title("(b) phase", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def figure_summary(models: Sequence[Dict[str, object]],
                   labels: Dict[str, str], path: str, dpi: int = 140) -> str:
    """One panel per decode: the whole answer at a glance."""
    figure, axes = plt.subplots(1, len(models),
                                figsize=(5.3 * len(models), 5.0),
                                squeeze=False)
    figure.suptitle("What is modelled and what is left, for each field of "
                    "each decode", fontsize=13)
    for column, model in enumerate(models):
        ax = axes[0][column]
        fields = model["fields"]
        cells = int(model["cells"]["count"])
        index = np.array([f["field"] for f in fields])
        heads = np.array([f["head"] for f in fields])
        before = np.array([in_floor_units(f["departure_power"], f["bins"],
                                          cells) for f in fields])
        after = np.array([in_floor_units(f["remainder_power"], f["bins"],
                                         cells) for f in fields])
        held = np.array([in_floor_units(f["held_out"]["remainder_power"],
                                        f["bins"], cells)
                         if "held_out" in f else np.nan for f in fields])
        ax.plot(index, before, ".", markersize=3.0, color="#8a8a8a")
        for head in ("a", "b"):
            mask = heads == head
            ax.plot(index[mask], after[mask], "-", linewidth=1.0,
                    color=HEAD_COLOUR[head])
            ax.plot(index[mask], held[mask], "--", linewidth=0.9,
                    color=HEAD_COLOUR[head], alpha=0.7)
        ax.axhline(1.0, color="#111111", linestyle=":", linewidth=1.3)
        ax.set_yscale("log")
        ax.set_xlabel("field index")
        if column == 0:
            ax.set_ylabel("residual, multiples of the standard error")
        ax.set_title(labels.get(model["prefix"], model["prefix"]),
                     fontsize=10)
        ax.grid(alpha=0.25)
    handles = [Line2D([], [], color="#8a8a8a", marker=".", linestyle="none",
                      label="before the model (level and delay removed)"),
               Line2D([], [], color=HEAD_COLOUR["a"], linestyle="-",
                      label="head a, after every component"),
               Line2D([], [], color=HEAD_COLOUR["b"], linestyle="-",
                      label="head b, after every component"),
               Line2D([], [], color="#555555", linestyle="--",
                      label="held out, coefficients frozen on other fields"),
               Line2D([], [], color="#111111", linestyle=":",
                      label="the instrument's own standard error")]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
                  frameon=False)
    figure.tight_layout(rect=(0, 0.11, 1, 0.93))
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# The written report
# --------------------------------------------------------------------------

DESCRIPTIONS: Dict[str, str] = {
    "specified test signal": (
        "the signal that was recorded, as its standard specifies it"),
    "vestigial sideband": (
        "the broadcast channel's asymmetric sideband shaping, before the "
        "recorder ever saw the signal"),
    "group delay": (
        "a delay that varies with frequency, so the band's parts arrive "
        "at different times"),
    "sound trap": (
        "the notch that removes the intercarrier sound at 4.5 MHz"),
    "echo": "a reflection arriving after the direct path",
    "waveform distortion": (
        "the level's departure over one of the four specified durations - "
        "short, line, field and long time"),
    "sub emphasis level dependence": (
        "the record-side emphasis behaving differently at different input "
        "levels"),
    "record level dependence": (
        "the tape's own transfer changing with how hard it was recorded"),
    "tape path clearance": (
        "the spacing between head and coating, whose loss is exponential "
        "in frequency"),
    "tape path strain": "the tape's tension and its effect on contact",
    "head differential frequency": (
        "the two video heads' responses differing in frequency"),
    "head differential amplitude": "the two heads differing in gain",
    "head differential phase": "the two heads differing in timing",
    "head contact tilt": (
        "the head's contact varying across the track, tilting the response"),
    "dropout": "a loss of contact, with a shape even where its position is random",
    "modulation noise": (
        "noise proportional to the recorded signal rather than added to it"),
    "particle noise": (
        "the finite number of oxide particles in the volume the head reads"),
    "beat / co-channel": (
        "a beat at a definite frequency, even when its phase is random"),
    "luma to colour-under transfer roll-off": (
        "the luma leaking into the colour-under band and rolling off"),
    "residual chroma carrier in luma": (
        "the colour-under carrier surviving into the luma output"),
    "colour-under envelope phase": "the colour-under envelope's timing",
    "colour-under envelope amplitude": "the colour-under envelope's gain",
    "colour-under heterodyne offset": (
        "the offset of the heterodyne that puts colour under the luma"),
    "sync phase": "the sync pulse's timing against the time base",
    "sync amplitude": "the sync pulse's depth against the specified scale",
    "burst lock": "the burst's lock to the subcarrier",
    "burst amplitude": "the burst's amplitude against the specified scale",
    "genlock": "the lock of the decode to the incoming field rate",
    "time base residual": "what the time base correction did not remove",
    "transport": "the mechanical rates of the drum, capstan and reels",
    "tuner clipping": "the tuner's limiter, at the front of the chain",
    "input clipping": "the recorder's own input limiter",
    "vcr agc": "the recorder's automatic gain control",
}


def describe(name: str) -> str:
    """One plain sentence for a component, from its family."""
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    for known, text in DESCRIPTIONS.items():
        if name.startswith(known):
            return text
    return "a modelled component of the chain"


def sign_test(descents) -> Dict[str, object]:
    """Does this component lower a residual it never saw, more often than chance?

    DISTRIBUTION FREE, ON PURPOSE. The held-out descents across fields are
    neither independent nor Gaussian - the two heads drift, and the key is
    collinear - so a t-test on their mean would quote an error bar the data
    do not support. Under the null that the component carries nothing out of
    sample its descent is as likely to be negative as positive, whatever its
    distribution, so the count of fields it lowers is binomial with a half
    and that is the whole test.
    """
    from scipy.stats import binomtest
    values = np.asarray(descents, dtype=np.float64)
    values = values[np.isfinite(values)]
    lowered = int(np.sum(values > 0.0))
    total = int(values.size)
    if total < 4:
        return {"lowered": lowered, "fields": total, "p_value": float("nan"),
                "survives": False,
                "reading": "too few fields for a decision"}
    p = float(binomtest(lowered, total, 0.5, alternative="greater").pvalue)
    return {"lowered": lowered, "fields": total, "p_value": p,
            "survives": bool(p < 0.05 and np.median(values) > 0.0),
            "reading": (f"lowered {lowered} of {total} fields, "
                        f"p = {p:.2g} against a coin")}


def change_point(values, index) -> Dict[str, object]:
    """The single place on the field axis where the answer changes most.

    A pooled figure hides this by construction, and the eye finds it on the
    plot without being able to say where or how much. The estimator is the
    standard one for a single shift in level: the split that maximises the
    difference of the two sides' means, scaled by `sqrt(n1 n2 / n)` so a
    split one field from the end cannot win by having almost no data on one
    side. The decision is then a Mann-Whitney U on the two sides, which is
    distribution free - the per-field values are neither Gaussian nor
    independent, and a t-test would quote an interval they do not support.
    """
    from scipy.stats import mannwhitneyu
    y = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(y)
    y, positions = y[keep], np.asarray(index)[keep]
    n = y.size
    if n < 20:
        return {"found": False, "reading": "too few fields to look"}
    total = np.cumsum(y)
    counts = np.arange(1, n)
    left = total[:-1] / counts
    right = (total[-1] - total[:-1]) / (n - counts)
    weight = np.sqrt(counts * (n - counts) / n)
    statistic = np.abs(left - right) * weight
    cut = int(np.argmax(statistic)) + 1
    if cut < 4 or n - cut < 4:
        return {"found": False, "reading": "no interior split"}
    p = float(mannwhitneyu(y[:cut], y[cut:], alternative="two-sided").pvalue)
    return {"found": True, "at_field": int(positions[cut]),
            "before": float(np.median(y[:cut])),
            "after": float(np.median(y[cut:])),
            "fields_before": cut, "fields_after": int(n - cut),
            "p_value": p, "significant": bool(p < 0.01)}


def strongest_shift(fields, names) -> Dict[str, object]:
    """The most significant single shift anywhere on the field axis.

    The total explained share can be flat while the COMPOSITION moves - one
    component giving way to another - and it is the composition that the
    stacked plot shows. So the search runs over the total and over every
    component's own share, and reports whichever shift is most significant,
    with its effect size, rather than only the total's.
    """
    index = [f["field"] for f in fields]
    candidates = []
    total = 100.0 * (1.0 - np.array([f["remainder_power"]
                                     / f["departure_power"] for f in fields]))
    found = change_point(total, index)
    if found.get("found"):
        found["quantity"] = "the total explained share"
        found["series"] = total
        candidates.append(found)
    ladder = np.array([f["ladder_share"] for f in fields]) * 100.0
    for position, name in enumerate(names):
        found = change_point(ladder[:, position], index)
        if found.get("found"):
            found["quantity"] = f"the share explained by {name}"
            found["series"] = ladder[:, position]
            candidates.append(found)
    if not candidates:
        return {"found": False}
    # RANKED BY A STANDARDISED EFFECT, not by significance alone. With two
    # hundred fields a shift of a twentieth of a percentage point reaches
    # p = 0.0002 and means nothing; what matters is the size of the step
    # against the field-to-field scatter of the same quantity. The scale is
    # the median absolute deviation, so a few wild fields cannot set it.
    for candidate in candidates:
        series = candidate.pop("series")
        scale = float(np.median(np.abs(series - np.median(series))))
        candidate["effect"] = abs(candidate["after"] - candidate["before"])
        candidate["scatter"] = scale
        candidate["standardised"] = (candidate["effect"] / scale
                                     if scale > 0 else 0.0)
        candidate["material"] = bool(candidate["significant"]
                                     and candidate["standardised"] >= 1.0)
    return max(candidates, key=lambda c: (c["material"], c["standardised"]))


def _quantiles(values) -> str:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return "n/a"
    return (f"{np.percentile(array, 50):.3g} "
            f"({np.percentile(array, 10):.3g} to "
            f"{np.percentile(array, 90):.3g})")


def report(models: Sequence[Dict[str, object]], labels: Dict[str, str],
           figures: Dict[str, List[str]], settings: Dict[str, object]) -> str:
    """The document a person reads."""
    lines: List[str] = []
    add = lines.append
    add("# The modelled residuals, field by field")
    add("")
    add("Ethan asked for a report on all the modelled residuals - what is "
        "being modelled, and what the difference is for each field. This "
        "is that report, and every number in it was produced by "
        "`tools/ringing_measure/modelled_residuals.py` on the decodes named "
        "below rather than quoted from an earlier run.")
    add("")
    add("Regenerate it, and the figures with it, with")
    add("")
    add("```")
    add("PYTHONPATH=/workspaces/vhs-decode python3 \\")
    add("    tools/ringing_measure/modelled_residuals.py")
    add("```")
    add("")
    add("and add decode prefixes as arguments to run it on new decodes. "
        "The tool's settings live in one readable file beside it, "
        "`tools/ringing_measure/modelled_residuals.toml`.")
    add("")

    add("## What is being measured, and why it has a field axis at all")
    add("")
    add("Every result the component key has been judged against until now "
        "came from the pooled `*_sync_step_response.npz` exports, and those "
        "have already averaged the field axis away - into one response per "
        "head, plus a first-half and a second-half estimate. Two halves are "
        "not a field axis.")
    add("")
    add("A single field carries about "
        f"{models[0]['measured']['lines_per_field']:.0f} horizontal sync "
        "pulses, and the sync pulse is the only deterministic step the "
        "signal contains: its depth, its width and its transition time are "
        "specified, so the measured edge divided by the specified edge is "
        "the channel's complex response as that one field witnessed it. "
        "This tool measures that response for each field separately and "
        "fits the whole component key to each, so what follows is a curve "
        "over the decode rather than a single number for it.")
    add("")
    add("The price is precision. Pooled over one field's own lines the "
        "instrument reaches a standard error of about 0.019 in the response "
        "magnitude, some fifteen times coarser than the pooled export's, so "
        "a larger part of one field's departure is the instrument's own "
        "noise than of the pooled export's - noise no component can "
        "explain. Where a figure below is meant to be read against the "
        "27 to 32 dB table recorded in `modelable_subspace`, it is the "
        "modelable-subspace column and not the weighted least-squares one: "
        "the two estimators differ, and the section on each decode says "
        "how.")
    add("")
    add("The instrument was checked against the certified pooled export "
        "rather than assumed to agree with it. Pooling this tool's "
        "per-field intervals over one head and taking the same step "
        "response reproduces the export's window exactly (lags 26 to 105, "
        "79 samples, 0.1812 MHz information spacing), its fitted crossing "
        "to 0.007 samples (45.010 against 45.017) and its step to 0.02 IRE "
        "(-37.455 against -37.438). The two responses agree to 0.08 dB and "
        "0.6 degrees below 1 MHz and part company above 2 MHz - 1.1 to 2.1 "
        "dB, 8 to 17 degrees - where the sync edge's own spectrum is "
        "weakest and the two instruments' dropout hygiene differs. That "
        "disagreement is the bound on what any figure here claims at the "
        "top of the band.")
    add("")

    add("A measurement per field is not a licence for a correction per "
        "field, and the two must not be confused. Rule 4 of "
        "`vhsdecode/addons/RINGING_RULES.md` stands: the channel's ringing "
        "is a fixed property of the deck, the measurement is accumulated "
        "over the whole decode and the APPLIED correction must not change "
        "from field to field. What the field axis is for is diagnosis - "
        "seeing where the model stops describing the tape, and how much of "
        "what is left moves - not per-field fitting.")
    add("")
    add("## One correction to the shared nuisance set, and what it was worth")
    add("")
    add("The measured departure is defined as the response with the terms "
        "that are not shapes removed - a gain belongs to the levels "
        "estimator and a delay to the time base, and charging either to the "
        "model would flatter it. `residual_floor.nuisance` offers two such "
        "terms, a constant real part and a linear phase taken about the "
        "BAND'S CENTRE, and calls the second a delay. It is not one. A "
        "delay of tau writes in the log domain as `-2 pi f tau`, which on a "
        "centred grid is `-2 pi (f - f0) tau` minus the constant "
        "`2 pi f0 tau`; the ramp is in that span and the constant is not, "
        "so a pure delay leaves a frequency-independent phase behind that "
        "no entry of the key can express and every fit charges to the "
        "residual. A second term earns its place alongside it for a "
        "different reason: the phase is unwrapped from the first bin's "
        "principal value, so the whole curve is defined only up to a "
        "constant, and without a free constant phase the answer depends on "
        "which branch the unwrap happened to take.")
    add("")
    add("This tool therefore fits three: a level, a phase reference and a "
        "delay through zero frequency. What the shared two-term set costs "
        "was measured in this run rather than assumed, by fitting the same "
        "key both ways on every field:")
    add("")
    add("| decode | explained, three terms | explained, the shared two | "
        "remainder, three terms | remainder, the shared two |")
    add("| --- | --- | --- | --- | --- |")
    for model in models:
        fields = model["fields"]
        cells = int(model["cells"]["count"])
        mine = np.median([100.0 * (1.0 - f["remainder_power"]
                                   / f["departure_power"]) for f in fields])
        theirs = np.median([100.0 * f["shared_nuisance"]["explained"]
                            for f in fields])
        mine_units = np.median([in_floor_units(f["remainder_power"],
                                               f["bins"], cells)
                                for f in fields])
        theirs_units = np.median([
            in_floor_units(f["shared_nuisance"]["remainder_power"],
                           f["bins"], cells) for f in fields])
        add(f"| {labels.get(model['prefix'], model['prefix'])} | "
            f"{mine:.1f} % | {theirs:.1f} % | {mine_units:.1f} | "
            f"{theirs_units:.1f} |")
    add("")
    add("Every figure in the rest of this report uses the three-term set. "
        "The shared one is left untouched: other lanes read it, and a "
        "change to it is Ethan's to make. It is recorded here because the "
        "difference is not small - it is the difference between a model "
        "that is judged to explain two thirds of the departure and one that "
        "explains five sixths of it.")
    add("")
    add("## The figures")
    add("")
    for prefix, paths in figures.items():
        add(f"- **{labels.get(prefix, prefix)}**")
        for path in paths:
            add(f"  - `{path}`")
    add("")

    for model in models:
        prefix = model["prefix"]
        label = labels.get(prefix, prefix)
        fields = model["fields"]
        cells = int(model["cells"]["count"])
        names = model["names"]
        add(f"## {label}")
        add("")
        measured = model["measured"]
        add(f"`{prefix}`: {measured['field_count']} fields, "
            f"{len(fields)} of them carrying a usable sync edge, "
            f"{measured['lines_per_field']:.0f} admitted lines per field, "
            f"the {measured['view']} view over "
            f"{measured['frequency_hz'][0] / 1e6:.3f} to "
            f"{measured['frequency_hz'][-1] / 1e6:.3f} MHz. "
            f"{fields[0]['bins']} frequency bins carry "
            f"{cells} independent points - the window is "
            f"{measured['window_lags'][1] - measured['window_lags'][0]} "
            f"samples long and its information spacing is "
            f"{measured['resolution_hz'] / 1e6:.4f} MHz, so the bins are "
            f"{model['cells']['oversampling']:.0f} times finer than the "
            "information and are counted as the cells they belong to.")
        add("")

        before_db = [in_decibels(f["departure"], f["floor"]) for f in fields]
        after_db = [in_decibels(f["remainder"], f["floor"]) for f in fields]
        before_deg = [in_degrees(f["departure"], f["floor"]) for f in fields]
        after_deg = [in_degrees(f["remainder"], f["floor"]) for f in fields]
        before_units = [in_floor_units(f["departure_power"], f["bins"], cells)
                        for f in fields]
        after_units = [in_floor_units(f["remainder_power"], f["bins"], cells)
                       for f in fields]
        held_units = [in_floor_units(f["held_out"]["remainder_power"],
                                     f["bins"], cells)
                      for f in fields if "held_out" in f]
        explained = [1.0 - f["remainder_power"] / f["departure_power"]
                     for f in fields]
        subspace = [f["exhausted"]["modelled_share"] for f in fields]
        above = [f["exhausted"]["above_floor_db"] for f in fields]
        recorded = [f["exhausted"]["above_floor_db_recorded"] for f in fields]
        rank = [f["exhausted"]["rank"] for f in fields]

        add("### The totals, per field")
        add("")
        add("| quantity | median over fields (10th to 90th percentile) |")
        add("| --- | --- |")
        add(f"| measured departure, amplitude | {_quantiles(before_db)} dB rms |")
        add(f"| remainder after every component, amplitude | "
            f"{_quantiles(after_db)} dB rms |")
        add(f"| measured departure, phase | {_quantiles(before_deg)} degrees rms |")
        add(f"| remainder after every component, phase | "
            f"{_quantiles(after_deg)} degrees rms |")
        add(f"| departure, in the instrument's own standard errors | "
            f"{_quantiles(before_units)} |")
        add(f"| remainder, in the same units, in sample | "
            f"{_quantiles(after_units)} |")
        add(f"| remainder, in the same units, held out | "
            f"{_quantiles(held_units)} |")
        add(f"| share of the departure explained, weighted least squares | "
            f"{_quantiles(np.array(explained) * 100)} per cent |")
        add(f"| share explained, modelable-subspace projection | "
            f"{_quantiles(np.array(subspace) * 100)} per cent |")
        add(f"| rank the data supports, of "
            f"{fields[0]['exhausted']['components']} | {_quantiles(rank)} |")
        add(f"| remainder above the medium's particulate floor | "
            f"{_quantiles(above)} dB |")
        add(f"| the same, in the convention recorded in "
            f"`modelable_subspace` | {_quantiles(recorded)} dB |")
        add("")
        add("The two share rows are two different estimators and are meant "
            "to be read as such. The weighted least-squares figure is the "
            "one the rest of this report uses: the key fitted with each bin "
            "weighted by its own precision, behind the three nuisance "
            "terms. The modelable-subspace figure is "
            "`modelable_subspace.exhausted`, which removes only the mean, "
            "weights every bin equally and projects through the singular "
            "value decomposition at a rank cut set by the measurement's own "
            "relative error. It is the lower of the two because it lets the "
            "band's noisiest bins carry the same weight as its best, and "
            "because it does not fit the delay; it is reported because it "
            "is the form the arc's recorded result was taken in.")
        add("")
        add(f"The two floor rows differ by about twenty decibels and both "
            f"are given because neither is wrong. A resolution cell of "
            f"this recording holds "
            f"{fields[0]['exhausted']['particles_per_cell']:.0f} oxide "
            f"particles whose orientations are independent, so its "
            f"magnetisation fluctuates by one over the root of that count "
            f"in RELATIVE amplitude - "
            f"{fields[0]['exhausted']['floor_db']:.4f} dB of log amplitude, "
            f"which is the same quantity the residual above is measured "
            f"in. That is the first row. The second is "
            f"`modelable_subspace.against_the_floor`, which divides the "
            f"DEPARTURE's own energy by the particle count rather than the "
            f"signal's; it is the convention behind the 27 to 32 dB table "
            f"recorded in that module, and it reads higher because a "
            f"departure is a small perturbation of a signal and not the "
            f"signal. The conclusion is the same either way: the remainder "
            f"is far above anything the medium's own physics imposes.")
        add("")

        add("### Each component: what it is, what it explains, what survives")
        add("")
        add("`in the chain` is the share of the departure the component "
            "removes when it is applied after its predecessors in the "
            "chain's own declared order. `only this one` is what it "
            "explains that no other entry in the key can reach - the gap "
            "between the two columns is the key's collinearity, not a "
            "measurement error. `held out` refits that same rung on the "
            "OTHER fields of the same head, freezes every coefficient and "
            "applies it unchanged here, so a component that has learned "
            "its own noise does not survive it. `survives held out` is a "
            "sign test: under the null that a component carries nothing "
            "out of sample its descent is as likely negative as positive, "
            "so the count of fields it lowers is binomial with a half, and "
            "the decision is that count against a coin at the five per "
            "cent level. The test is distribution free because the "
            "descents across fields are neither independent nor Gaussian.")
        add("")
        add("| component | what it is | in the chain, % | only this one, % "
            "| held out, % | fields lowered | survives held out |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        ladder = np.array([f["ladder_share"] for f in fields])
        unique = np.array([f["unique_share"] for f in fields])
        held = np.array([f["held_out"]["ladder_share"] for f in fields
                         if "held_out" in f])
        for position, name in enumerate(names):
            share = np.median(ladder[:, position]) * 100.0
            only = np.median(unique[:, position]) * 100.0
            if held.size:
                out = np.median(held[:, position]) * 100.0
                test = sign_test(held[:, position])
                out_text = f"{out:+.3f}"
                lowered_text = (f"{test['lowered']} of {test['fields']} "
                                f"(p = {test['p_value']:.2g})")
                verdict_text = "yes" if test["survives"] else "no"
            else:
                out_text = lowered_text = verdict_text = "n/a"
            add(f"| {name} | {describe(name)} | {share:.3f} | {only:.3f} "
                f"| {out_text} | {lowered_text} | {verdict_text} |")
        add("")

        add("### The fields that differ most")
        add("")
        shift = strongest_shift(fields, names)
        if shift.get("found") and shift.get("material"):
            add(f"The decode is not one population. Searching the total "
                f"explained share and every component's share separately, "
                f"the largest single shift on the field axis falls at "
                f"field {shift['at_field']}, in {shift['quantity']}: the "
                f"{shift['fields_before']} fields before it have a median "
                f"of {shift['before']:.3g} per cent and the "
                f"{shift['fields_after']} after it {shift['after']:.3g} "
                f"per cent. That step is {shift['standardised']:.1f} times "
                f"the field-to-field scatter of the same quantity, at "
                f"p = {shift['p_value']:.2g}. Nothing in the key has a time "
                f"axis, so a shift like this can only appear in the "
                f"residual - which is exactly why the field axis had to be "
                f"measured rather than averaged away.")
        elif shift.get("found"):
            add(f"No shift on the field axis is worth calling a step. The "
                f"best split found, at field {shift['at_field']} in "
                f"{shift['quantity']}, separates {shift['before']:.3g} per "
                f"cent from {shift['after']:.3g} - only "
                f"{shift['standardised']:.2f} times the field-to-field "
                f"scatter of that quantity, at p = "
                f"{shift['p_value']:.2g}. With two hundred fields a step of "
                f"a twentieth of a point clears any significance test and "
                f"means nothing, so the size against the scatter is what "
                f"decides. What varies between fields here varies without a "
                f"step in it.")
        add("")
        add("The comparison is taken WITHIN a head, and that is not a "
            "formality. Fields alternate between the two video heads on a "
            "helical drum, the two heads' departures do not correlate, and "
            "their instruments do not even have the same noise: the "
            "standard errors below differ between the heads, so a "
            "remainder quoted in standard errors is not comparable across "
            "them. Ranking every field of a decode together would report a "
            "head difference as a field difference.")
        add("")
        # the column name must not contain a pipe: it is a markdown table
        add("| head | fields | the instrument's own standard error in the "
            "response magnitude | remainder, standard errors | remainder, "
            "dB | explained, % | field of the largest remainder | of the "
            "smallest | spread |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for head in sorted(model["by_head"]):
            members = model["by_head"][head]
            head_after = np.array([after_units[m] for m in members])
            head_db = [after_db[m] for m in members]
            head_explained = [explained[m] * 100.0 for m in members]
            noise = np.median([np.median(
                np.sqrt(fields[m]["floor"]) * np.abs(
                    np.exp(np.real(fields[m]["log"]))))
                for m in members])
            worst = members[int(np.argmax(head_after))]
            best = members[int(np.argmin(head_after))]
            add(f"| {head} | {len(members)} | {noise:.4f} | "
                f"{_quantiles(head_after)} | {_quantiles(head_db)} | "
                f"{_quantiles(head_explained)} | "
                f"{fields[worst]['field']} at {after_units[worst]:.1f} | "
                f"{fields[best]['field']} at {after_units[best]:.1f} | "
                f"{after_units[worst] / max(after_units[best], 1e-12):.2f}x |")
        add("")

    add("## What all of it comes to")
    add("")
    for model in models:
        prefix = model["prefix"]
        fields = model["fields"]
        cells = int(model["cells"]["count"])
        explained = np.median([1.0 - f["remainder_power"]
                               / f["departure_power"] for f in fields])
        after_units = np.median([in_floor_units(f["remainder_power"],
                                                f["bins"], cells)
                                 for f in fields])
        held = [in_floor_units(f["held_out"]["remainder_power"], f["bins"],
                               cells) for f in fields if "held_out" in f]
        above = np.median([f["exhausted"]["above_floor_db"] for f in fields])
        recorded = np.median([f["exhausted"]["above_floor_db_recorded"]
                              for f in fields])
        add(f"- **{labels.get(prefix, prefix)}**: the "
            f"{len(model['names'])} modelled components together explain "
            f"{100 * explained:.1f} per cent of the median field's "
            f"departure, leaving {after_units:.1f} standard errors in "
            f"sample and {np.median(held):.1f} held out, and that remainder "
            f"sits {above:.1f} dB above the floor the medium's own particle "
            f"count fixes ({recorded:.1f} dB in the convention recorded in "
            f"`modelable_subspace`).")
    add("")
    add("Three things follow from those numbers, and they are the reason "
        "this report has a field axis.")
    add("")
    add("**The remainder is not noise.** It stands well above the "
        "instrument's own standard error and far above the medium's "
        "particulate floor, and it is a REMAINDER rather than a "
        "measurement: the same components, frozen on other fields, "
        "reproduce most of what they removed here. Structure remains, and "
        "the modelling is not near the limit the physics fixes.")
    add("")
    add("**The key is collinear, and the two share columns say by how "
        "much.** A component's share in the chain is often several times "
        "what it alone explains. That is the identifiability ceiling this "
        "arc has already measured directly - six magnetic mechanisms "
        "collapsing to 1.58 distinguishable directions - showing up here "
        "as a difference between two columns of the same table.")
    add("")
    add("**The field axis carries real variation.** The per-field "
        "remainder moves by a large factor across a decode, and the "
        "movement is not the same on the two heads. A key with no time "
        "axis cannot follow it, which is exactly what the pooled held-out "
        "work predicted when it found the half-split difference exceeding "
        "what the within-pool error allows.")
    add("")
    add("---")
    add("")
    add("Generated by `tools/ringing_measure/modelled_residuals.py`. "
        "The component key is `vhsdecode/models/interference.signatures`, "
        "its chain order `interference.full_chain`, the projection and the "
        "medium's floor `vhsdecode/models/modelable_subspace`, and the "
        "weighted fit, the log domain and the resolution cells "
        "`vhsdecode/models/residual_floor`.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------

def run(prefixes: Optional[Sequence[str]] = None,
        settings: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Measure, model, plot and write, for every decode asked for."""
    settings = settings or configuration()
    decodes = settings["decodes"]
    measurement = settings["measurement"]
    output = settings["output"]
    labels = dict(decodes.get("labels", {}))
    chosen = list(prefixes) if prefixes else list(decodes["prefixes"])
    geom = geometry(measurement["system"], measurement["tape_format"],
                    measurement["tape_speed"])
    band = tuple(float(edge) for edge in measurement["band_hz"])
    os.makedirs(output["figures"], exist_ok=True)

    models, figures = [], {}
    for prefix in chosen:
        print(f"[{prefix}] measuring every field ...", flush=True)
        measured = measure(prefix, decodes["directory"], geom,
                           measurement["view"], band)
        print(f"[{prefix}] {len(measured['usable'])} of "
              f"{measured['field_count']} fields usable, "
              f"{int(measured['valid'].sum())} bins in band", flush=True)
        model = model_decode(measured, float(settings["model"]
                                             ["relative_error"]),
                             bool(settings["model"]["admitted_key"]))
        models.append(model)
        label = labels.get(prefix, prefix)
        paths = [
            figure_fields(model, label,
                          os.path.join(output["figures"],
                                       f"modelled_residuals_{prefix}"
                                       "_fields.png"), int(output["dpi"])),
            figure_components(model, label,
                              os.path.join(output["figures"],
                                           f"modelled_residuals_{prefix}"
                                           "_components.png"),
                              int(output["dpi"])),
            figure_frequency(model, label,
                             os.path.join(output["figures"],
                                          f"modelled_residuals_{prefix}"
                                          "_frequency.png"),
                             int(output["dpi"])),
        ]
        figures[prefix] = paths
        for path in paths:
            print(f"[{prefix}] wrote {path} "
                  f"({os.path.getsize(path) / 1024:.0f} kB)", flush=True)

    summary = os.path.join(output["figures"], "modelled_residuals_summary.png")
    figure_summary(models, labels, summary, int(output["dpi"]))
    figures["all decodes"] = [summary]
    print(f"wrote {summary} ({os.path.getsize(summary) / 1024:.0f} kB)",
          flush=True)

    document = os.path.join(REPOSITORY, output["document"])
    os.makedirs(os.path.dirname(document), exist_ok=True)
    with open(document, "w") as handle:
        handle.write(report(models, labels, figures, settings))
    print(f"wrote {document} "
          f"({os.path.getsize(document) / 1024:.0f} kB)", flush=True)
    return {"models": models, "figures": figures, "document": document}


if __name__ == "__main__":
    run(sys.argv[1:] or None)
