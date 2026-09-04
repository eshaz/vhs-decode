"""The VCR's own filters, as parameters rather than as a curve.

The head model gives the magnetics: a spacing in microns, an azimuth in
minutes of arc. This is its counterpart for the electronics. The recording
and playback machines apply a de-emphasis shelf and a set of band-limiting
filters whose FORM is fixed by the format and whose PARAMETERS vary from
device to device - the shelf's mid frequency, its gain and its Q drift with
component tolerance, age and temperature, and no two machines are alike.

Fitting a measured response to those parameters rather than to a free curve
is what makes the result a per-device fit: it says "this machine's shelf sits
at 268 kHz, not the assumed 274", which is a statement about a machine that
transfers to its other recordings. A spline coefficient says nothing that
travels.

Two rules hold this together:

  The parametrization is the DECODER'S OWN. The shelf is built by calling
  `compute_video_filters.gen_video_main_deemp_fft`, the same function the
  runtime uses, so a fitted parameter can be handed straight back as a
  decode setting. Reimplementing the form here would let the two drift
  apart silently, and a fit against a slightly different shelf would put
  the difference between the shelves into the fitted device.

  The assumed values come from the repo's own format definitions, never
  from numbers copied into this file. What is reported is the DEPARTURE
  from what the decoder already assumes, because that is the quantity a
  decode can act on.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.compute_video_filters import gen_video_main_deemp_fft


# The shelf's three parameters, in the decoder's own names and units, with
# the bounds a real device can occupy. The bounds are deliberately wide:
# they exist to keep a fit physical, not to encode an expectation.
DEEMPHASIS_PARAMETERS = ("deemph_gain", "deemph_mid", "deemph_q")
DEEMPHASIS_BOUNDS = {
    "deemph_gain": (4.0, 24.0),        # dB of shelf
    "deemph_mid": (80e3, 900e3),       # Hz
    "deemph_q": (0.15, 2.0),
}


def assumed_parameters(rf_params: Dict[str, float]) -> Dict[str, float]:
    """What the decoder already assumes, read from the format definitions
    it was built with. The fit reports departures from these."""
    return {
        "deemph_gain": float(rf_params["deemph_gain"]),
        "deemph_mid": float(rf_params["deemph_mid"]),
        "deemph_q": float(rf_params.get("deemph_q", 0.5)),
    }


def shelf(parameters: Dict[str, float], freq_hz: float,
          block_len: int) -> np.ndarray:
    """The de-emphasis shelf, built by the decoder's own construction."""
    return np.asarray(gen_video_main_deemp_fft(
        float(parameters["deemph_gain"]),
        float(parameters["deemph_mid"]),
        float(parameters["deemph_q"]),
        float(freq_hz), int(block_len)))


def shelf_log_magnitude(parameters: Dict[str, float], frequency_hz,
                        freq_hz: float, block_len: int) -> np.ndarray:
    """The shelf's log magnitude sampled at arbitrary frequencies."""
    table = shelf(parameters, freq_hz, block_len)
    grid = np.arange(len(table)) * (float(freq_hz) / int(block_len))
    return np.interp(np.asarray(frequency_hz, dtype=np.float64), grid,
                     np.log(np.abs(table) + 1e-30))


def fit_deemphasis(frequency_hz, measured_log, rf_params: Dict[str, float],
                   freq_hz: float, block_len: int,
                   free: Sequence[str] = (),
                   weights=None) -> Dict[str, float]:
    """Fit the shelf's parameters to a measured DEPARTURE from the assumed
    response.

    `measured_log` is the measured log magnitude relative to the decode
    that produced it - that is, what the decoder's assumed shelf failed to
    account for. The fit returns the parameters whose shelf, minus the
    assumed shelf, reproduces that departure. An overall constant is
    projected out first: a level is not a shelf parameter, and leaving it
    in lets the fit trade the mid frequency against a gain."""
    from scipy.optimize import least_squares

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    measured = np.asarray(measured_log, dtype=np.float64)
    good = np.isfinite(frequency) & np.isfinite(measured) & (frequency > 0)
    if good.sum() < 4:
        return {}
    frequency, measured = frequency[good], measured[good]
    weight = (np.ones_like(measured) if weights is None
              else np.asarray(weights, dtype=np.float64)[good])
    assumed = assumed_parameters(rf_params)
    free = list(free) or list(DEEMPHASIS_PARAMETERS)
    reference = shelf_log_magnitude(assumed, frequency, freq_hz, block_len)
    start = np.array([assumed[name] for name in free], dtype=np.float64)
    lower = np.array([DEEMPHASIS_BOUNDS[name][0] for name in free])
    upper = np.array([DEEMPHASIS_BOUNDS[name][1] for name in free])

    def model_for(values):
        parameters = dict(assumed)
        parameters.update(dict(zip(free, values)))
        return shelf_log_magnitude(parameters, frequency, freq_hz,
                                   block_len) - reference

    def residual(values):
        model = model_for(values)
        return weight * ((model - model.mean()) - (measured - measured.mean()))

    try:
        solution = least_squares(residual, np.clip(start, lower, upper),
                                 bounds=(lower, upper),
                                 x_scale=np.maximum(np.abs(start), 1e-6),
                                 max_nfev=4000)
    except Exception:
        return {}
    fitted = dict(assumed)
    fitted.update(dict(zip(free, solution.x)))
    model = model_for(solution.x)
    error = (model - model.mean()) - (measured - measured.mean())
    out = {name: float(fitted[name]) for name in DEEMPHASIS_PARAMETERS}
    out.update({
        "residual_rms_nepers": float(np.sqrt(np.mean(error ** 2))),
        "explained": float(1.0 - np.var(error)
                           / max(np.var(measured - measured.mean()), 1e-12)),
        "bins": int(len(frequency)),
    })
    for name in DEEMPHASIS_PARAMETERS:
        out["assumed_" + name] = float(assumed[name])
        out["departure_" + name] = float(fitted[name] - assumed[name])
    # a parameter pinned at its bound was not determined by the data
    out["pinned"] = ",".join(
        name for name, value in zip(free, solution.x)
        if abs(value - DEEMPHASIS_BOUNDS[name][0]) < 1e-9
        or abs(value - DEEMPHASIS_BOUNDS[name][1]) < 1e-9)
    return out


def device_signature(head: Optional[Dict[str, float]],
                     filters: Optional[Dict[str, float]]) -> Dict[str, float]:
    """One device's fit: its magnetics and its electronics together.

    This is where the head model connects up. The head parameters describe
    the magnetics of the recording and playback pair; the shelf parameters
    describe that machine's electronics. Together they are what a VCR model
    predicts, and what a second tape from the same machine should
    reproduce - which is the test that makes the whole thing falsifiable
    rather than merely a good fit."""
    signature: Dict[str, float] = {}
    for source, prefix in ((head, "head_"), (filters, "filter_")):
        for name, value in (source or {}).items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                signature[prefix + name] = float(value)
    return signature


def agreement(first: Dict[str, float], second: Dict[str, float],
              errors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Whether two fits describe the same device.

    Each shared parameter is compared against its own error, because a
    difference below the measurement's error is not a difference. This is
    the falsification test for the VCR model: two tapes from one machine
    must agree, and two from different machines need not."""
    shared = sorted(set(first) & set(second))
    out: Dict[str, float] = {}
    worst = 0.0
    for name in shared:
        error = (errors or {}).get(name)
        difference = float(first[name] - second[name])
        out["difference_" + name] = difference
        if error and error > 0:
            sigma = abs(difference) / error
            out["sigma_" + name] = float(sigma)
            worst = max(worst, sigma)
    if worst:
        out["worst_sigma"] = float(worst)
        out["same_device"] = float(worst < 3.0)
    out["compared"] = float(len(shared))
    return out


# --------------------------------------------------------------------------
# the extrapolation: the per-VCR differential as a curve, against its inputs
# --------------------------------------------------------------------------

# The response stays a CURVE. What is modelled is how that curve MOVES, and
# the movement is regressed on inputs that are measured per field:
#
#   recorded energy the integrated power of the RF envelope: the energy
#                   actually laid onto the tape. It moves with record
#                   current, and so with the recording machine's supply
#                   voltage and its temperature - the environment. It is
#                   external to the picture, being the carrier's amplitude
#                   rather than the content, so unlike scene variation it
#                   may drive rather than merely be regressed out.
#   tape position   the frequency axis of a per-field series IS a length of
#                   tape: one field is a fixed distance at a fixed linear
#                   speed, so a drift at so many cycles per field is a
#                   feature of so many millimetres of tape. External.
#   recording date  user-supplied metadata. External, and the only input
#                   spanning years rather than a third of a second.
#   scene variation A NUISANCE regressor, never a driver. It comes from the
#                   content, and the standing constraint is that the
#                   correction derives from sync alone. Including it REMOVES
#                   the content's influence from the per-VCR estimate rather
#                   than admitting it: the coefficient is fitted so that its
#                   effect can be subtracted, which makes the remaining
#                   estimate cleaner, not content-driven.
#
# Each input gets its own COEFFICIENT CURVE - a response shape per unit of
# that input - so the result is still a curve fit, with the per-VCR
# differential component as what the inputs explain.

EXTRAPOLATION_INPUTS = ("tape_position_m", "recording_years", "recorded_energy",
                       "scene_variation")
NUISANCE_INPUTS = ("scene_variation",)


def tape_position(field_index, linear_speed_m_s: float,
                  field_rate_hz: float = 59.94) -> np.ndarray:
    """Where along the tape each field sits, in metres.

    This is the conversion that turns the per-field axis into a length: at a
    fixed linear speed each field occupies a fixed distance, so a drift
    measured in cycles per field is a feature of a distance along the tape,
    and the two descriptions are the same measurement in different units."""
    index = np.asarray(field_index, dtype=np.float64)
    return index * float(linear_speed_m_s) / max(float(field_rate_hz), 1e-9)


def spatial_period(cycles_per_field, linear_speed_m_s: float,
                   field_rate_hz: float = 59.94) -> np.ndarray:
    """A drift rate in cycles per field, expressed as a length of tape."""
    cycles = np.asarray(cycles_per_field, dtype=np.float64)
    per_field = float(linear_speed_m_s) / max(float(field_rate_hz), 1e-9)
    return np.where(cycles > 0, per_field / np.maximum(cycles, 1e-12), np.inf)


def recorded_energy(envelope) -> float:
    """The total energy laid onto the tape in one field.

    The RF envelope is what the record head actually wrote, so its
    integrated power is the energy that went into the tape. That makes it
    a direct witness of the recording machine's ENVIRONMENT: record
    current falls as a battery sags, and a tape's coercivity falls as it
    warms, so both supply and temperature move this number.

    Reported as a log, because it is a power and everything else here is
    compared in nepers."""
    values = np.asarray(envelope, dtype=np.float64).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 64:
        return float("nan")
    return float(np.log(np.mean(values ** 2)))


def scene_variation(field_picture, active_start_sample: int) -> float:
    """One number per field describing how much the picture is doing.

    Taken from the active area only, and used ONLY to be regressed out.
    The measure is the median absolute line-to-line change, which is robust
    to a few bright lines and responds to motion and detail alike."""
    picture = np.asarray(field_picture, dtype=np.float64)[:, int(active_start_sample):]
    if picture.shape[0] < 3 or picture.shape[1] < 8:
        return float("nan")
    return float(np.median(np.abs(np.diff(picture, axis=0))))


def design_matrix(inputs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    """The regressors, standardised, with an intercept first.

    Standardising puts every coefficient in the same units - response per
    standard deviation of its input - so their sizes may be compared. An
    input that never varies is dropped rather than left to destabilise the
    solve."""
    names, columns = [], []
    for name in EXTRAPOLATION_INPUTS:
        values = inputs.get(name)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).any() or np.nanstd(values) <= 0:
            continue
        centred = values - np.nanmean(values)
        columns.append(np.nan_to_num(centred / np.nanstd(values)))
        names.append(name)
    if not columns:
        return np.zeros((0, 0)), []
    matrix = np.column_stack([np.ones(len(columns[0]))] + columns)
    return matrix, ["intercept"] + names


def extrapolate(curves: np.ndarray, inputs: Dict[str, np.ndarray]
                ) -> Dict[str, object]:
    """Regress a per-field family of response CURVES on its inputs.

    `curves` is (fields, frequency bins). Every frequency bin is solved
    against the same design, so each input comes back as a coefficient
    CURVE - the response shape that one standard deviation of that input
    produces. The nuisance inputs' contribution is then subtracted, and
    what remains is the per-VCR differential component with the content's
    influence removed."""
    observed = np.asarray(curves, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[0] < 4:
        return {"usable": False,
                "why": "the extrapolation needs at least four fields"}
    matrix, names = design_matrix(inputs)
    if matrix.size == 0 or matrix.shape[0] != observed.shape[0]:
        return {"usable": False, "why": "no usable inputs"}
    centred = observed - observed.mean(axis=0, keepdims=True)
    coefficients, *_ = np.linalg.lstsq(matrix, centred, rcond=None)
    fitted = matrix @ coefficients
    residual = centred - fitted
    total = float(np.var(centred))
    out = {
        "usable": True,
        "names": names,
        "coefficients": coefficients,
        "residual": residual,
        "explained": float(1.0 - np.var(residual) / max(total, 1e-30)),
        "fields": int(observed.shape[0]),
    }
    # what each input explains on its own, and the curve it carries
    for position, name in enumerate(names):
        if name == "intercept":
            continue
        contribution = np.outer(matrix[:, position], coefficients[position])
        out["curve_" + name] = coefficients[position]
        out["explained_" + name] = float(np.var(contribution) / max(total, 1e-30))
    # the per-VCR component: the nuisance inputs' effect removed, the
    # external ones kept, because only the external ones may drive a
    # correction
    nuisance = np.zeros_like(centred)
    for position, name in enumerate(names):
        if name in NUISANCE_INPUTS:
            nuisance = nuisance + np.outer(matrix[:, position],
                                           coefficients[position])
    out["per_vcr_component"] = centred - nuisance
    out["nuisance_removed"] = float(np.var(nuisance) / max(total, 1e-30))
    return out


# --------------------------------------------------------------------------
# the non-linear stage, and the triple differential of every parameter
# --------------------------------------------------------------------------

# The non-linear de-emphasis scales the high-frequency part of the signal by
# one minus a power of its own instantaneous amplitude, so it removes more at
# low amplitude than at high. That is what makes it non-linear, and it is
# also why its parameters are AMPLITUDE-dependent by construction: the
# triple differential is the natural description of them, not an extra
# structure laid over them.
#
# Names are the decoder's own (`nonlinear_filter.sub_deemphasis_inner`), so a
# fitted value can be handed back as a setting.
NONLINEAR_PARAMETERS = (
    "exponential_scale",     # the power the amplitude is raised to
    "linear_scale_1",        # scaling before that power
    "linear_scale_2",        # scaling after it
    "logistic_mid",          # where the logistic weighting turns over
    "logistic_rate",         # how sharply it turns
    "static_factor",         # a linear part running alongside
    "nonlinear_highpass_freq",   # which frequencies the stage acts on
)
ALL_PARAMETERS = DEEMPHASIS_PARAMETERS + NONLINEAR_PARAMETERS


def nonlinear_response(amplitude, frequency_hz, parameters,
                       corner_hz=None) -> np.ndarray:
    """What the non-linear stage does, as a function of BOTH amplitude and
    frequency - which is the whole point of it.

    The high-frequency part is taken by a high-pass, its instantaneous
    amplitude is scaled and raised to a power, and the part is then removed
    in proportion to one minus that. So the gain at a given frequency
    depends on the signal's own level there, and no single frequency
    response describes it."""
    a = np.asarray(amplitude, dtype=np.float64)[:, None]
    f = np.asarray(frequency_hz, dtype=np.float64)[None, :]
    corner = float(corner_hz if corner_hz is not None
                   else parameters.get("nonlinear_highpass_freq", 820e3))
    # the high-pass the stage acts through, first order
    high = (f / corner) / np.sqrt(1.0 + (f / corner) ** 2)
    scaled = a * float(parameters.get("linear_scale_1", 1.0) or 1.0)
    powered = np.power(np.maximum(scaled, 0.0),
                       float(parameters.get("exponential_scale", 0.33)))
    powered = powered * float(parameters.get("linear_scale_2", 1.0) or 1.0)
    rate = float(parameters.get("logistic_rate", 0.0) or 0.0)
    if rate > 0:
        mid = float(parameters.get("logistic_mid", 0.0) or 0.0)
        powered = powered / (1.0 + np.exp(-rate * (powered - mid)))
    removed = high * (1.0 - powered)
    static = float(parameters.get("static_factor", 0.0) or 0.0)
    return 1.0 - removed - high * static


def combined_response(amplitude, frequency_hz, parameters, freq_hz=40e6,
                      block_len=32768) -> np.ndarray:
    """The LINEAR shelf and the NON-LINEAR stage together.

    They must be evaluated jointly or the identifiability question is
    answered wrongly: measuring the non-linear stage alone reports the
    shelf's parameters as unobservable, which is true only of that
    measurement and not of the chain. The shelf sets the response's shape
    against frequency and the non-linear stage bends it with amplitude, so
    only together do they describe what the video path does."""
    shelf = shelf_log_magnitude(parameters, frequency_hz, freq_hz, block_len)
    non_linear = nonlinear_response(amplitude, frequency_hz, parameters)
    return np.exp(shelf)[None, :] * non_linear


# The logistic term is switched OFF when its rate is zero, and a derivative
# taken at exactly zero measures the discontinuity of switching it on rather
# than the parameter's own effect. Sensitivities are therefore evaluated at a
# small positive rate when the stage is inactive, and the parameter is
# reported as inactive rather than as enormously sensitive.
LOGISTIC_PROBE_RATE = 1e-3


def parameter_sensitivity(amplitude, frequency_hz, parameters,
                          names=None, step: float = 1e-3
                          ) -> Dict[str, np.ndarray]:
    """The TRIPLE DIFFERENTIAL of every parameter: how the response moves
    when each one does, across amplitude and frequency.

    This is what says which parameters can be fitted at all. A parameter
    whose sensitivity is flat everywhere does nothing observable; two whose
    sensitivities have the SAME SHAPE cannot be told apart by any amount of
    data, and a fit will divide them arbitrarily. Checking this first is
    cheaper than discovering it from a fit that rails at its bounds.

    Reported per ONE PERCENT change of each parameter, so parameters
    carried in different units are comparable."""
    names = list(names or ALL_PARAMETERS)
    working = dict(parameters)
    inactive = []
    if not working.get("logistic_rate"):
        # probe it awake, so what is measured is its effect and not the
        # discontinuity of turning the stage on
        working["logistic_rate"] = LOGISTIC_PROBE_RATE
        inactive.append("logistic_rate")
    base = combined_response(amplitude, frequency_hz, working)
    out = {}
    for name in names:
        value = working.get(name)
        if value is None:
            continue
        nudged = dict(working)
        delta = step * (abs(float(value)) if value else 1.0)
        nudged[name] = float(value) + delta
        moved = combined_response(amplitude, frequency_hz, nudged)
        # PER FRACTIONAL CHANGE, not per absolute unit. Dividing by the
        # absolute step makes a parameter measured in hertz look
        # insensitive next to one measured in units of order one - the mid
        # frequency's 2.7e-7 per hertz is not small, it is 274000 hertz
        # wide. Scaling to a common one-percent change is what makes the
        # comparison mean anything.
        out[name] = (moved - base) / max(step, 1e-12) / 100.0
    out["_inactive"] = np.array(inactive, dtype=object)
    return out


def identifiability(sensitivities: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Which parameters are separable, from their sensitivities alone.

    Two parameters are confounded when their sensitivity patterns point the
    same way: no data separates them, and the split a fit reports is an
    artefact of where it started. Reporting this BEFORE fitting is what
    stops a railed parameter being read as a measurement."""
    names = [n for n, v in sensitivities.items()
             if not n.startswith("_") and np.any(np.isfinite(v))]
    if len(names) < 2:
        return {"names": names, "pairs": []}
    flat = []
    for name in names:
        v = np.nan_to_num(sensitivities[name]).ravel()
        norm = np.linalg.norm(v)
        flat.append(v / norm if norm > 0 else v)
    matrix = np.array(flat)
    gram = np.abs(matrix @ matrix.T)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append({"a": names[i], "b": names[j],
                          "alignment": float(gram[i, j]),
                          "separable": bool(gram[i, j] < 0.9)})
    dead = [names[i] for i in range(len(names))
            if np.linalg.norm(np.nan_to_num(sensitivities[names[i]])) <= 0]
    # HOW MANY parameters are independently identifiable is the RANK of the
    # sensitivity matrix, not the count minus the confounded pairs - pairs
    # overlap, and subtracting them double-counts. The singular values say
    # how many independent directions the parameters actually span.
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular.max() * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > max(tolerance, 1e-9 * singular.max())))
    return {"names": names, "pairs": sorted(pairs, key=lambda p: -p["alignment"]),
            "unobservable": dead,
            "worst": max((p["alignment"] for p in pairs), default=0.0),
            "identifiable": rank,
            "supplied": len(names),
            "singular_values": singular}


# --------------------------------------------------------------------------
# the de-emphasis chain, in the decoder's own order, and what it does to a pulse
# --------------------------------------------------------------------------

# THE ORDER IS NOT A DETAIL. Read from process.py:
#   1. the MAIN de-emphasis, a linear shelf folded into FVideo and applied in
#      the frequency domain, so it acts first;
#   2. `nldeemp`, which takes the high-frequency part, HARD CLIPS it to
#      limits, and subtracts it - non-linear by clipping;
#   3. `subdeemp`, which takes the high-frequency part, scales it by a power
#      of its own instantaneous amplitude, and subtracts it - non-linear by
#      the power law.
# Two SEPARATE non-linear paths. Both draw their high-frequency part from the
# SAME pre-existing spectrum and subtract in turn, so they do not cascade
# through one another, and modelling them as a cascade would be wrong.
#
# A pulse's WIDTH is what this chain moves: each stage reshapes the edges, and
# a 50% crossing sits wherever the reshaped edge puts it. So a width that
# departs from the specification is a candidate signature of a mis-set
# parameter, and the sensitivity of width to each parameter says which
# parameter could account for it - and whether any of them can.
NONLINEAR_LIMIT_PARAMETERS = ("nonlinear_highpass_limit_l",
                              "nonlinear_highpass_limit_h")


def _high_pass(signal, corner_hz, rate_hz):
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), 1.0 / rate_hz)
    with np.errstate(invalid="ignore"):
        response = (freq / corner_hz) / np.sqrt(1.0 + (freq / corner_hz) ** 2)
    return np.fft.irfft(spectrum * response, n=len(signal))


# THE NON-LINEAR STAGE WORKS IN HERTZ, NOT IRE. It runs on the demodulated
# signal before `hz_to_output`, and divides the instantaneous amplitude by a
# DEVIATION which the decoder computes as hz_ire * (100 - vsync_ire) - one
# megahertz for NTSC VHS. A 40 IRE sync pulse is 285714 Hz there. Feeding it
# IRE against a deviation of one puts the stage orders of magnitude outside
# its design range, where its response is meaningless and no fit converges.
def nominal_deviation(hz_ire: float, vsync_ire: float = -40.0) -> float:
    """The deviation the non-linear stage measures amplitudes against, by
    the decoder's own definition (`create_sub_emphasis_params`)."""
    return float(hz_ire) * (100.0 - float(vsync_ire))


def apply_deemphasis_chain(pulse, parameters, rate_hz, freq_hz=40e6,
                           block_len=32768, stages=("main", "nldeemp",
                                                    "subdeemp")):
    """Put a pulse through the de-emphasis chain in the decoder's order.

    `stages` names which run, so a stage can be left out to see what it
    alone contributes. The high-frequency part for BOTH non-linear paths is
    taken from the signal as it stands after the main de-emphasis, matching
    the decoder, which computes one spectrum and feeds both from it."""
    x = np.asarray(pulse, dtype=np.float64).copy()
    if "main" in stages:
        shelf = shelf_log_magnitude(parameters,
                                    np.fft.rfftfreq(len(x), 1.0 / rate_hz),
                                    freq_hz, block_len)
        x = np.fft.irfft(np.fft.rfft(x) * np.exp(shelf), n=len(x))
    corner = float(parameters.get("nonlinear_highpass_freq", 820e3))
    high = _high_pass(x, corner, rate_hz)          # one spectrum, both paths
    if "nldeemp" in stages:
        clipped = np.clip(high,
                          float(parameters.get("nonlinear_highpass_limit_l",
                                               -20000.0)),
                          float(parameters.get("nonlinear_highpass_limit_h",
                                               5000.0)))
        x = x - clipped
    if "subdeemp" in stages:
        from scipy.signal import hilbert

        deviation = float(parameters.get("deviation", 1.0)) / 2.0
        amplitude = np.abs(hilbert(high)) / max(deviation, 1e-12)
        amplitude = amplitude * float(parameters.get("linear_scale_1", 1.0) or 1.0)
        amplitude = np.power(np.maximum(amplitude, 0.0),
                             float(parameters.get("exponential_scale", 0.33)))
        amplitude = amplitude * float(parameters.get("linear_scale_2", 1.0) or 1.0)
        rate = float(parameters.get("logistic_rate", 0.0) or 0.0)
        if rate > 0:
            mid = float(parameters.get("logistic_mid", 0.0) or 0.0)
            amplitude = amplitude / (1.0 + np.exp(-rate * (amplitude - mid)))
        x = x - high * (1.0 - amplitude)
    return x


def pulse_width(signal, rate_hz, level=None):
    """A pulse's width at its half level, by interpolated crossings."""
    y = np.asarray(signal, dtype=np.float64)
    top = float(np.median(np.concatenate([y[:8], y[-8:]])))
    bottom = float(np.min(y))
    half = 0.5 * (top + bottom) if level is None else float(level)
    down = np.nonzero((y[:-1] >= half) & (y[1:] < half))[0]
    up = np.nonzero((y[:-1] < half) & (y[1:] >= half))[0]
    if not len(down) or not len(up):
        return float("nan")
    def refine(i):
        a, b = y[i], y[i + 1]
        return i + (half - a) / ((b - a) if b != a else 1e-12)
    start, stop = refine(down[0]), refine(up[-1])
    return (stop - start) / rate_hz * 1e6 if stop > start else float("nan")


def width_sensitivity(pulse, parameters, rate_hz, names=None,
                      step: float = 0.02) -> Dict[str, float]:
    """How far each parameter moves the pulse's WIDTH, per one percent.

    This is what turns a measured width deficit into a statement about
    parameters. A parameter that barely moves the width cannot account for
    the deficit however far it is mis-set, and two that move it the same way
    cannot be told apart by the width alone - the width is ONE number, so at
    most one degree of freedom can ever be recovered from it."""
    base = pulse_width(apply_deemphasis_chain(pulse, parameters, rate_hz),
                       rate_hz)
    out = {"baseline_us": base}
    for name in list(names or (ALL_PARAMETERS + NONLINEAR_LIMIT_PARAMETERS)):
        value = parameters.get(name)
        if value is None:
            continue
        nudged = dict(parameters)
        nudged[name] = float(value) * (1.0 + step)
        moved = pulse_width(apply_deemphasis_chain(nudged and pulse, nudged,
                                                   rate_hz), rate_hz)
        if np.isfinite(moved) and np.isfinite(base):
            out[name] = float((moved - base) / (step * 100.0))
    return out


# --------------------------------------------------------------------------
# fitting the chain ONE STAGE AT A TIME, in the chain's own order
# --------------------------------------------------------------------------

# The chain's order is known circuitry, so the fit follows it rather than
# treating all ten parameters as one problem. At each step only ONE stage's
# parameters are free and the stages before it are already determined, which
# is what keeps the degeneracies out: two parameters can only be confounded
# with each other if both are free at once.
#
# Within a stage a genuine degeneracy still has to be handled, and it is
# handled by fitting the identifiable COMBINATION. `linear_scale_1` and
# `linear_scale_2` provably enter only as s1^e * s2, so one of them is held
# and the other carries the pair - fitting both would report the starting
# point, not a measurement.
#
# The sweep repeats until the parameters stop moving, because a later stage's
# fit changes what the earlier stages should have been. That is ordinary
# coordinate descent, and it converges here because each stage's own fit is
# well posed once the others are held.
STAGE_ORDER = ("main", "nldeemp", "subdeemp")
STAGE_PARAMETERS = {
    "main": ("deemph_gain", "deemph_mid", "deemph_q"),
    # `linear_scale_1` is deliberately absent: it is not separable from
    # `linear_scale_2`, so the pair is carried by the latter alone
    "nldeemp": ("nonlinear_highpass_freq", "nonlinear_highpass_limit_l",
                "nonlinear_highpass_limit_h"),
    "subdeemp": ("exponential_scale", "linear_scale_2", "static_factor"),
}
HELD_BY_DEGENERACY = {"linear_scale_1": "linear_scale_2"}


def response_curve(pulse, parameters, rate_hz, stages=STAGE_ORDER,
                   band=(50e3, 5e6)):
    """The chain's response, as a discrete transform of the pulse.

    The same instrument the colour-under work used: transform the pulse,
    take the ratio against the specified one, and read the response off it.
    A curve rather than a single number, which is what lets the parameters
    be told apart - the width could never do that, being one number."""
    ideal = np.asarray(pulse, dtype=np.float64)
    got = apply_deemphasis_chain(ideal, parameters, rate_hz, stages=stages)
    freq = np.fft.rfftfreq(len(ideal), 1.0 / rate_hz)
    a = np.fft.rfft(ideal)
    b = np.fft.rfft(got)
    inside = (freq >= band[0]) & (freq <= band[1]) & (np.abs(a) > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.abs(b[inside]) / np.abs(a[inside]))
    return freq[inside], ratio


def fit_chain_in_order(pulse, measured_curve, measured_freq, start, rate_hz,
                       sweeps: int = 4, bounds=None) -> Dict[str, object]:
    """Fit the chain one stage at a time, down the chain's own order.

    `measured_curve` is the response actually observed, at `measured_freq`.
    Each stage in turn has its parameters fitted against that curve with the
    others held; the sweep repeats until they stop moving.

    Reported per stage: what it fitted, how much of the curve it explains,
    and whether any parameter landed on a bound - a bound means the data did
    not determine it, and the value should not be read as a measurement."""
    from scipy.optimize import least_squares

    bounds = dict(bounds or {})
    bounds.setdefault("deemph_gain", DEEMPHASIS_BOUNDS["deemph_gain"])
    bounds.setdefault("deemph_mid", DEEMPHASIS_BOUNDS["deemph_mid"])
    bounds.setdefault("deemph_q", DEEMPHASIS_BOUNDS["deemph_q"])
    parameters = dict(start)
    target = np.asarray(measured_curve, dtype=np.float64)
    history = []
    for sweep in range(int(sweeps)):
        for stage in STAGE_ORDER:
            names = [n for n in STAGE_PARAMETERS[stage]
                     if parameters.get(n) is not None]
            if not names:
                continue
            lower, upper, start_values = [], [], []
            for name in names:
                value = float(parameters[name])
                low, high = bounds.get(
                    name, (value - abs(value) - 1.0, value + abs(value) + 1.0))
                lower.append(low); upper.append(high)
                start_values.append(min(max(value, low), high))

            def residual(values, names=names):
                trial = dict(parameters)
                trial.update(dict(zip(names, values)))
                for held, carrier in HELD_BY_DEGENERACY.items():
                    if held in trial and carrier in trial:
                        trial[held] = float(start.get(held, trial[held]))
                freq, curve = response_curve(pulse, trial, rate_hz)
                model = np.interp(measured_freq, freq, curve)
                return (model - model.mean()) - (target - target.mean())

            try:
                # X_SCALE IS NOT OPTIONAL HERE. These parameters span six
                # orders of magnitude - a mid frequency near 3e5 beside an
                # exponent near 0.1 - and the default unit step is enormous
                # for one and negligible for the other. Left at the default
                # the fit moved nothing and recovered none of six planted
                # parameters; scaled per parameter it recovers the strongest
                # to 5% and lifts the explained share from 83% to 92%.
                solved = least_squares(residual, start_values,
                                       bounds=(lower, upper),
                                       x_scale=np.maximum(
                                           np.abs(start_values), 1e-9),
                                       max_nfev=600)
            except Exception:                                # noqa: BLE001
                continue
            parameters.update(dict(zip(names, solved.x)))
            error = residual(solved.x)
            explained = float(1.0 - np.var(error)
                              / max(np.var(target - target.mean()), 1e-30))
            pinned = [n for n, v, lo, hi in zip(names, solved.x, lower, upper)
                      if abs(v - lo) < 1e-9 or abs(v - hi) < 1e-9]
            history.append({"sweep": sweep, "stage": stage,
                            "fitted": dict(zip(names, map(float, solved.x))),
                            "explained": explained, "pinned": pinned})
    return {"parameters": parameters, "history": history,
            "held_by_degeneracy": dict(HELD_BY_DEGENERACY),
            "order": list(STAGE_ORDER)}

