"""A physical model of a VHS video head, for fitting the measured response.

The per-head response is not an arbitrary curve to be splined: a magnetic
recording head has a known transfer function built from a small number of
mechanisms, each with a physical parameter and each established long
before this format existed. Fitting the measurement to THAT rather than to
a free curve is what makes the fitted numbers mean something - a spacing
in microns, an azimuth error in minutes of arc - and it is what allows
tuning, because a physical parameter can be adjusted where a spline
coefficient cannot.

The mechanisms, in the order they enter (standard magnetic recording
theory; Wallace 1951 for the spacing law, Mallinson and Jorgensen for the
rest):

  differentiation  playback voltage follows the rate of change of flux,
                   so the head's own response rises with frequency. This
                   is the term that makes a head a differentiator and it
                   is exact, not fitted.
  spacing loss     exp(-2*pi*d/lambda) - Wallace's law. The single
                   largest loss, and the one that worsens as a tape or a
                   head ages and dirt intervenes. d is a separation.
  gap loss         sinc(g/lambda) from the finite read gap, with a null
                   at a wavelength equal to the gap. g is a length.
  thickness loss   (1 - exp(-x))/x with x = 2*pi*delta/lambda, from the
                   depth of the coating that contributes.
  azimuth loss     sinc(W*tan(dtheta)/lambda) across the track width. THE
                   term that separates two VHS heads: the format records
                   adjacent tracks at OPPOSITE azimuth, so a systematic
                   azimuth error cannot affect both heads alike. Where a
                   measured head difference is real, this is the first
                   mechanism to suspect.
  contour effect   the "head bumps" - undulations at LOW frequency caused
                   by flux paths around the finite head core, at
                   wavelengths comparable to the core's dimensions. It is
                   a known per-head low-frequency phenomenon, which is
                   where the measured head difference actually sits.

All wavelengths are lambda = v/f with v the head-to-tape writing speed,
which the format's mechanics fix: the drum's circumference times its
rotation rate. Nothing here is a tuned constant - the mechanical figures
are the format's specification and are declared per format and speed.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# The format's own mechanics. These are specification figures, not tuning:
# the writing speed follows from the drum a machine of this format must
# have, and the azimuth and track width are what the format defines.
FORMAT_MECHANICS: Dict[Tuple[str, str, str], Dict[str, float]] = {
    # (tape format, system, tape speed)
    # JVC Video Technical Guide VTG82063, section 1. These are the
    # SPECIFICATION with its own tolerances, not typical figures.
    ("VHS", "NTSC", "SP"): {
        "drum_diameter_m": 0.062,                # 62 +/- 0.01 mm
        "drum_revolutions_per_second": 30.0,     # 30 Hz (NTSC)
        "linear_tape_speed_m_s": 0.03335,        # 33.35 mm/s +/- 0.5%
        # the guide gives the writing speed DIRECTLY as 5.80 m/s. It is not
        # the drum's surface speed: the head sweeps helically while the tape
        # moves, and the velocity along the TRACK is the vector sum at the
        # track angle. Deriving it as circumference times rotation gives
        # 5.877, which is 1.3% high and carries into every wavelength and so
        # into every fitted spacing.
        "writing_speed_m_s": 5.80,
        "track_angle_degrees": 5.0 + 58.0 / 60.0 + 9.9 / 3600.0,
        "azimuth_tolerance_degrees": 10.0 / 60.0,
        # fallback only: the repo's `video_track_width` overrides it
        "track_width_m": 58e-6,
        "azimuth_degrees": 6.0,                  # the format's +/- 6
    },
    ("VHS", "PAL", "SP"): {
        "drum_diameter_m": 0.062,
        "drum_revolutions_per_second": 25.0,     # 1500 rpm
        "linear_tape_speed_m_s": 0.02339,       # 23.39 mm/s +/- 0.5%
        "writing_speed_m_s": 4.85,              # stated by the guide
        "track_angle_degrees": 5.0 + 57.0 / 60.0 + 50.3 / 3600.0,
        "azimuth_tolerance_degrees": 10.0 / 60.0,
        "track_width_m": 49e-6,                 # 0.049 mm
        "azimuth_degrees": 6.0,
    },
}

# Starting points for the fitted parameters, in metres and degrees. These
# are typical construction figures for a head of this class and serve only
# as a starting guess and a physical bound; the fit moves them.
# PER-CHANNEL parameters. A drum carries two video heads and each has its
# own alignment, so these belong to a CHANNEL and not to the machine. The
# other model's simulator lists exactly these four per channel, which is an
# independent confirmation of the per-head structure this arc reached from
# the data.
#
# `protrusion_m` is the mechanism behind the spacing: a head stands proud of
# the drum by some radial extension, and how far it does sets how hard it
# presses into the tape and therefore the separation Wallace's law sees. A
# head that has worn back protrudes less, contacts less firmly, and reads as
# more spacing. Fitting spacing alone gave a number with no mechanism; this
# is the mechanism, and it is what wears.
CHANNEL_PARAMETERS = ("azimuth_error_degrees", "gain_db", "height_m",
                      "protrusion_m")
PROTRUSION_TO_SPACING = -1.0   # more protrusion, less separation


def spacing_from_protrusion(protrusion_m: float, reference_m: float = 0.0
                            ) -> float:
    """The separation a given head protrusion produces, relative to a
    reference head. Only DIFFERENCES are meaningful: the absolute
    separation also carries the tape's surface and its coating, which no
    measurement on one tape can split from the head's own contribution."""
    return PROTRUSION_TO_SPACING * (float(protrusion_m) - float(reference_m))


TYPICAL = {
    "spacing_m": 0.05e-6,
    "protrusion_m": 0.0,
    "height_m": 0.0,
    "gain_db": 0.0,
    "gap_m": 0.30e-6,
    "thickness_m": 0.20e-6,
    "azimuth_error_degrees": 0.0,
    "contour_length_m": 200e-6,
    "contour_depth": 0.0,
}
BOUNDS = {
    "spacing_m": (0.0, 1.0e-6),
    "protrusion_m": (-2.0e-6, 2.0e-6),
    "height_m": (-30.0e-6, 30.0e-6),
    "gain_db": (-6.0, 6.0),
    "gap_m": (0.05e-6, 1.5e-6),
    "thickness_m": (0.02e-6, 2.0e-6),
    "azimuth_error_degrees": (-2.0, 2.0),
    "contour_length_m": (20e-6, 2000e-6),
    "contour_depth": (-1.0, 1.0),
}
# the effective gap is a little wider than the physical one, a standard
# correction for the field's fringing either side of it
EFFECTIVE_GAP_FACTOR = 1.11

# one neper is 20/ln(10) decibels. The head model works in nepers because
# the loss mechanisms are exponentials; `gain_db` is stated in decibels
# because that is how a head's output difference is specified.
DB_PER_NEPER = 20.0 / np.log(10.0)

# The mechanisms the carrier-normalization law cannot see, and so the ones
# that must never be fitted against a demodulated video measurement.
#
# The law responds to [e(fc+f) + e(fc-f)]/2 - e(fc), which annihilates any
# constant and any linear function of frequency exactly. `gain_db` is a
# constant; Wallace's spacing loss is -2*pi*d*f/v, linear in frequency. Both
# therefore reach demodulated video as nothing at all - measured through
# `predicted_video_effect` as 0.0e+00 and 1.8e-15 nepers respectively,
# against 0.15 to 0.32 for gap, thickness and azimuth.
#
# They are perfectly measurable on the RF ENVELOPE, where a level is a level
# and a slope is a slope; `fit_difference_response` is where they belong.
CARRIER_LAW_NULL_SPACE = ("gain_db", "spacing_m")


def writing_speed(mechanics: Dict[str, float]) -> float:
    """The head-to-tape speed, which converts every frequency into a
    recorded wavelength.

    The SPECIFICATION states it directly, and where it does that figure
    wins. Deriving it from the drum's circumference and rotation rate gives
    5.877 m/s against the stated 5.80, because the head sweeps helically
    while the tape moves and the velocity along the TRACK is the vector sum
    at the track angle. That 1.3% would go straight into every wavelength
    and so into every fitted spacing."""
    stated = mechanics.get("writing_speed_m_s")
    if stated:
        return float(stated)
    return (np.pi * mechanics["drum_diameter_m"]
            * mechanics["drum_revolutions_per_second"]
            + mechanics.get("linear_tape_speed_m_s", 0.0))


def mechanics_for(tape_format: str, system: str, tape_speed: str,
                  rf_params: Optional[Dict[str, float]] = None
                  ) -> Optional[Dict[str, float]]:
    """The format's mechanics. Where the decoder's own RF parameters are
    supplied they WIN: the repo already carries the recorded track width
    per tape speed (`video_track_width`, in micrometres), and a second
    copy here could drift away from it silently. Only the figures the repo
    does not hold - the drum's diameter and rotation rate, and the
    format's azimuth - are declared in this table."""
    key = (str(tape_format).upper(), str(system).upper(), str(tape_speed).upper())
    mechanics = FORMAT_MECHANICS.get(key)
    if mechanics is None:
        return None
    mechanics = dict(mechanics)
    width = (rf_params or {}).get("video_track_width")
    if width:
        mechanics["track_width_m"] = float(width) * 1e-6
        mechanics["track_width_source"] = "the decoder's own rf parameters"
    return mechanics


def _sinc(x: np.ndarray) -> np.ndarray:
    """sin(pi x)/(pi x), finite at zero."""
    return np.sinc(np.asarray(x, dtype=np.float64))


def log_response(frequency_hz: np.ndarray, speed_m_s: float,
                 track_width_m: float, **parameters) -> np.ndarray:
    """The head's log magnitude response in nepers, mechanism by mechanism.

    Returned WITHOUT the differentiation term's overall scale, which is a
    gain rather than a shape; for ONE head the caller removes any constant.

    Between TWO heads a constant is not free: it is their gain difference,
    and `gain_db` carries it. Leaving it out is what made the spacing wrong.
    Measured on this tape the head difference is still 0.15-0.22 nepers at
    629 kHz, where a difference in clearance would have decayed to 0.01 -
    so the heads differ in flat GAIN, and a spacing-only fit had to invent
    0.45-0.49 um of clearance to reproduce a curve that is not a slope."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    # a head has no response at or below zero frequency, and a caller that
    # reaches there is asking for a sideband that has folded through the
    # carrier. Return not-a-number so it is masked rather than silently
    # becoming a negative infinity that poisons a fit.
    valid = frequency > 0
    wavelength = np.where(valid, speed_m_s / np.maximum(frequency, 1e-9), np.inf)
    total = np.zeros_like(frequency)
    total = total + np.where(valid, np.log(np.maximum(frequency, 1e-9)), np.nan)

    # a flat gain: no shape at all, so it is invisible in one head's fit and
    # separable in a difference. See the docstring for why it has to be here.
    gain_db = float(parameters.get("gain_db", 0.0) or 0.0)
    if gain_db:
        total = total + gain_db / DB_PER_NEPER

    # Wallace spacing loss
    spacing = float(parameters.get("spacing_m", 0.0))
    if spacing > 0:
        total = total - 2.0 * np.pi * spacing / wavelength

    # gap loss, with the standard effective-gap correction
    gap = float(parameters.get("gap_m", 0.0))
    if gap > 0:
        ratio = EFFECTIVE_GAP_FACTOR * gap / wavelength
        total = total + np.log(np.maximum(np.abs(_sinc(ratio)), 1e-12))

    # thickness loss
    thickness = float(parameters.get("thickness_m", 0.0))
    if thickness > 0:
        x = 2.0 * np.pi * thickness / wavelength
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(x > 1e-9, (1.0 - np.exp(-x)) / np.maximum(x, 1e-12), 1.0)
        total = total + np.log(np.maximum(term, 1e-12))

    # azimuth loss - the mechanism that can separate two VHS heads, because
    # the format writes adjacent tracks at opposite azimuth
    error = float(parameters.get("azimuth_error_degrees", 0.0))
    if error != 0.0 and track_width_m > 0:
        across = track_width_m * np.tan(np.radians(error))
        total = total + np.log(np.maximum(
            np.abs(_sinc(across / wavelength)), 1e-12))

    # the contour effect: undulations at LOW frequency from the finite head
    # core, at wavelengths comparable to its dimensions
    depth = float(parameters.get("contour_depth", 0.0))
    length = float(parameters.get("contour_length_m", 0.0))
    if depth != 0.0 and length > 0:
        total = total + depth * np.cos(2.0 * np.pi * length / wavelength) \
            * np.exp(-wavelength_ratio(wavelength, length))
    return total


def wavelength_ratio(wavelength: np.ndarray, length: float) -> np.ndarray:
    """How fast the contour undulations die away with frequency: they are a
    long-wavelength effect and vanish once the wavelength is short compared
    with the core."""
    return np.asarray(length / np.maximum(wavelength, 1e-12) * 0.15,
                      dtype=np.float64)


def group_delay_s(frequency_hz: np.ndarray, speed_m_s: float,
                  track_width_m: float, **parameters) -> np.ndarray:
    """The TIME axis of the same model: the group delay implied by the
    model's phase.

    The loss mechanisms are minimum phase, so the phase follows from the
    magnitude and the delay follows from the phase. The gap and azimuth
    terms are the exception - their sinc nulls flip sign rather than
    turning the phase smoothly - and they are handled by carrying the sign
    explicitly."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    magnitude = log_response(frequency, speed_m_s, track_width_m, **parameters)
    finite = np.isfinite(magnitude)
    phase = np.zeros_like(frequency)
    if finite.sum() > 3:
        # the minimum-phase relation, discretely: the Hilbert transform of
        # the log magnitude
        from scipy.signal import hilbert
        phase[finite] = -np.imag(hilbert(magnitude[finite]))
    delay = np.zeros_like(frequency)
    if len(frequency) > 2:
        delay[1:-1] = -(phase[2:] - phase[:-2]) / (
            2.0 * np.pi * (frequency[2:] - frequency[:-2] + 1e-12))
    return delay


def fit(frequency_hz: np.ndarray, measured_log: np.ndarray,
        mechanics: Dict[str, float], free: Sequence[str] = (),
        weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Fit the head model to a measured log magnitude.

    A constant is projected out first: an overall gain is not a head
    parameter, and leaving it in would let the fit trade real mechanisms
    against a level. What is fitted is the SHAPE."""
    from scipy.optimize import least_squares

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    measured = np.asarray(measured_log, dtype=np.float64)
    good = np.isfinite(frequency) & np.isfinite(measured) & (frequency > 0)
    if good.sum() < 4:
        return {}
    frequency, measured = frequency[good], measured[good]
    weight = (np.ones_like(measured) if weights is None
              else np.asarray(weights, dtype=np.float64)[good])
    free = list(free) or ["spacing_m", "gap_m", "azimuth_error_degrees"]
    speed = writing_speed(mechanics)
    width = mechanics["track_width_m"]
    start = np.array([TYPICAL[name] for name in free], dtype=np.float64)
    lower = np.array([BOUNDS[name][0] for name in free], dtype=np.float64)
    upper = np.array([BOUNDS[name][1] for name in free], dtype=np.float64)
    scale = np.maximum(np.abs(start), (upper - lower) / 100.0)

    def residual(values):
        parameters = dict(TYPICAL)
        parameters.update(dict(zip(free, values)))
        model = log_response(frequency, speed, width, **parameters)
        model = np.where(np.isfinite(model), model, 0.0)
        # an overall gain is not a head parameter: remove the constant
        return weight * ((model - model.mean()) - (measured - measured.mean()))

    try:
        solution = least_squares(residual, np.clip(start, lower, upper),
                                 bounds=(lower, upper), x_scale=scale,
                                 max_nfev=4000)
    except Exception:
        return {}
    fitted = dict(zip(free, solution.x))
    fitted["writing_speed_m_s"] = speed
    fitted["residual_rms_nepers"] = float(
        np.sqrt(np.mean(solution.fun ** 2)))
    reference = measured - measured.mean()
    fitted["explained"] = float(
        1.0 - np.var(solution.fun) / max(np.var(reference), 1e-12))
    return fitted


# --------------------------------------------------------------------------
# from the head to what the decoder actually sees
# --------------------------------------------------------------------------

def head_difference_log(frequency_hz, speed_m_s, track_width_m,
                        first: Dict[str, float],
                        second: Dict[str, float]) -> np.ndarray:
    """The two heads' log responses subtracted.

    Everything the two heads SHARE cancels here - the differentiation, the
    record-side path, the decoder's own filters, the de-emphasis - which is
    what makes the difference the honest quantity to fit. Only mechanisms
    that actually differ between the heads survive it, and for this format
    that is chiefly azimuth, since adjacent tracks are written at opposite
    azimuth and no systematic azimuth error can affect both heads alike."""
    return (log_response(frequency_hz, speed_m_s, track_width_m, **first)
            - log_response(frequency_hz, speed_m_s, track_width_m, **second))


def predicted_video_effect(baseband_hz: np.ndarray, rf_log_difference,
                           carrier_hz: float, speed_m_s: float,
                           track_width_m: float, first: Dict[str, float],
                           second: Dict[str, float]) -> np.ndarray:
    """What an RF-domain head difference does to the DEMODULATED video.

    The head acts on the FM carrier, not on baseband video, so a head
    model cannot be fitted to a video spectrum directly. The demodulated
    magnitude responds to the response either side of the carrier by the
    carrier-normalization law already measured on this chain: the sum of
    the two sidebands, halved, less the response at the carrier itself.
    A departure that is symmetric about the carrier therefore shows, and
    one that is a linear tilt does not - which is the null space this
    whole apparatus keeps running into."""
    baseband = np.asarray(baseband_hz, dtype=np.float64)
    # the lower sideband must stay above zero: a baseband frequency beyond
    # the carrier folds through it and has no meaning here
    reachable = baseband < carrier_hz
    upper = rf_log_difference(carrier_hz + baseband)
    lower = rf_log_difference(np.where(reachable, carrier_hz - baseband, np.nan))
    at_carrier = rf_log_difference(np.array([carrier_hz]))[0]
    out = 0.5 * (upper + lower) - at_carrier
    return np.where(reachable, out, np.nan)


def fit_difference_response(frequency_hz: np.ndarray,
                            measured_difference: np.ndarray,
                            mechanics: Dict[str, float],
                            free: Sequence[str] = ("gain_db", "spacing_m"),
                            weights: Optional[np.ndarray] = None
                            ) -> Dict[str, float]:
    """Fit a head difference measured on the RF ENVELOPE, constant kept.

    THE CONSTANT IS NOT PROJECTED OUT HERE, and that is the whole point.
    For one head an overall gain is not a head parameter, so `fit` removes
    it. For a DIFFERENCE the constant is the two heads' gain difference - a
    real and separable parameter - and removing it forces spacing to absorb
    something that is not spacing.

    It has to be fitted HERE rather than in `fit_head_difference`, because
    that one works in the video domain through the carrier-normalization
    law, and that law annihilates a constant exactly: half the sum of two
    equal sidebands, less an equal carrier, is zero. A flat gain difference
    is therefore INVISIBLE in demodulated video and measurable only on the
    RF envelope, where it is simply a level."""
    from scipy.optimize import least_squares

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    measured = np.asarray(measured_difference, dtype=np.float64)
    good = np.isfinite(frequency) & np.isfinite(measured) & (frequency > 0)
    if good.sum() < 4:
        return {}
    frequency, measured = frequency[good], measured[good]
    weight = (np.ones_like(measured) if weights is None
              else np.asarray(weights, dtype=np.float64)[good])
    free = list(free)
    speed = writing_speed(mechanics)
    width = mechanics["track_width_m"]
    start = np.array([TYPICAL[name] for name in free], dtype=np.float64)
    lower = np.array([BOUNDS[name][0] for name in free], dtype=np.float64)
    upper = np.array([BOUNDS[name][1] for name in free], dtype=np.float64)
    start = np.clip(np.where(start == 0.0, 1e-3, start), lower, upper)

    def residual(values):
        first = dict(TYPICAL)
        first.update(dict(zip(free, values)))
        model = head_difference_log(frequency, speed, width, first,
                                    dict(TYPICAL))
        model = np.where(np.isfinite(model), model, 0.0)
        return weight * (model - measured)          # NO constant removed

    try:
        solution = least_squares(residual, start, bounds=(lower, upper),
                                 x_scale=np.maximum(np.abs(start), 1e-9),
                                 max_nfev=4000)
    except Exception:
        return {}
    fitted = dict(zip(free, map(float, solution.x)))
    fitted["writing_speed_m_s"] = speed
    error = residual(solution.x) / np.maximum(weight, 1e-12)
    fitted["residual_rms_nepers"] = float(np.sqrt(np.mean(error ** 2)))
    # `explained` is against the VARIANCE, the same denominator `fit` uses,
    # so the two are comparable. Against the mean square instead - which the
    # constant now belongs to - the very same residual reads far higher, and
    # a fit that had not improved at all appeared to go from 82% to 99%.
    # Both are reported, named apart, so neither can stand in for the other.
    fitted["explained"] = float(
        1.0 - np.var(error) / max(np.var(measured), 1e-12))
    fitted["explained_with_constant"] = float(
        1.0 - np.mean(error ** 2) / max(np.mean(measured ** 2), 1e-12))
    fitted["bins"] = int(len(frequency))
    return fitted


def fit_head_difference(baseband_hz: np.ndarray, measured_log: np.ndarray,
                        carrier_hz: float, mechanics: Dict[str, float],
                        free: Sequence[str] = (),
                        weights: Optional[np.ndarray] = None
                        ) -> Dict[str, float]:
    """Fit the PHYSICAL head parameters to a measured per-head difference.

    The measurement is in the video domain; the model lives at RF; the
    carrier-normalization law connects them. What is fitted is the
    difference between the heads, so only parameters that can differ
    between two heads of one drum are free.

    TWO parameters are deliberately NOT free here, because both lie in this
    estimator's NULL SPACE. The carrier law annihilates any constant and
    any linear function of frequency, and:

      gain_db    is a constant, exactly;
      spacing_m  is Wallace's -2*pi*d*f/v, which is linear in frequency
                 exactly - so a spacing difference is annihilated too.

    Measured through this very function, a 1.5 dB gain difference reaches
    the video as 0.0e+00 nepers and a 0.40 um spacing difference as
    1.8e-15, while gap, thickness and azimuth come through at 0.15-0.32.
    A parameter free in a null direction does not merely fit poorly; it is
    unconstrained, and takes whatever value the noise asks for. Fit the
    curved mechanisms here, and the flat and linear ones on the RF
    envelope with `fit_difference_response`."""
    from scipy.optimize import least_squares

    baseband = np.asarray(baseband_hz, dtype=np.float64)
    measured = np.asarray(measured_log, dtype=np.float64)
    good = np.isfinite(baseband) & np.isfinite(measured)
    if good.sum() < 4:
        return {}
    baseband, measured = baseband[good], measured[good]
    weight = (np.ones_like(measured) if weights is None
              else np.asarray(weights, dtype=np.float64)[good])
    # the curved mechanisms only - see the docstring for why spacing and
    # gain cannot be among them
    free = list(free) or ["azimuth_error_degrees", "gap_m"]
    blind = [name for name in free if name in CARRIER_LAW_NULL_SPACE]
    if blind:
        raise ValueError(
            "%s cannot be fitted in the video domain: the carrier law "
            "annihilates it. Fit it on the RF envelope with "
            "fit_difference_response()." % ", ".join(blind))
    speed = writing_speed(mechanics)
    width = mechanics["track_width_m"]
    start = np.array([TYPICAL[name] for name in free])
    lower = np.array([BOUNDS[name][0] for name in free])
    upper = np.array([BOUNDS[name][1] for name in free])
    # the two heads differ by these parameters: one is held at the typical
    # value and the other moves, so what is fitted is a DIFFERENCE
    start = np.where(start == 0.0, 0.1, start)

    def residual(values):
        first = dict(TYPICAL)
        second = dict(TYPICAL)
        for name, value in zip(free, values):
            first[name] = value
        model = predicted_video_effect(
            baseband,
            lambda f: head_difference_log(f, speed, width, first, second),
            carrier_hz, speed, width, first, second)
        usable = np.isfinite(model)
        if usable.sum() < 4:
            return weight * (measured - measured.mean())
        # compared only where the model is defined, and with the constant
        # removed on the SAME set, so a shrinking set cannot flatter the fit
        difference = np.zeros_like(measured)
        difference[usable] = ((model[usable] - model[usable].mean())
                              - (measured[usable] - measured[usable].mean()))
        return weight * difference

    try:
        solution = least_squares(residual, np.clip(start, lower, upper),
                                 bounds=(lower, upper),
                                 x_scale=np.maximum(np.abs(start), 1e-9),
                                 max_nfev=4000)
    except Exception:
        return {}
    fitted = dict(zip(free, solution.x))
    fitted["writing_speed_m_s"] = speed
    fitted["carrier_hz"] = float(carrier_hz)
    # the quality is reported UNWEIGHTED. The solver's own residual carries
    # the fit weights, and comparing a weighted residual against an
    # unweighted reference is a units error that makes the explained share
    # meaningless - it can read in the millions of percent.
    best = dict(TYPICAL)
    best.update(dict(zip(free, solution.x)))
    model = predicted_video_effect(
        baseband, lambda f: head_difference_log(f, speed, width, best,
                                                dict(TYPICAL)),
        carrier_hz, speed, width, best, dict(TYPICAL))
    usable = np.isfinite(model)
    if usable.sum() >= 4:
        centred_model = model[usable] - model[usable].mean()
        centred_measured = measured[usable] - measured[usable].mean()
        error = centred_model - centred_measured
        fitted["residual_rms_nepers"] = float(np.sqrt(np.mean(error ** 2)))
        fitted["explained"] = float(
            1.0 - np.var(error) / max(np.var(centred_measured), 1e-12))
        fitted["bins"] = int(usable.sum())
    else:
        fitted["residual_rms_nepers"] = float("nan")
        fitted["explained"] = float("nan")
        fitted["bins"] = int(usable.sum())
    return fitted


# --------------------------------------------------------------------------
# the RF stage over time: the parameters that are NOT constant
# --------------------------------------------------------------------------

# The per-head loss is static given the recording and playback pair - it is
# fixed hardware. The RF STAGE's own parameters are not. The recording
# device's battery voltage sags and its temperature climbs while it
# records, and a camcorder does both far more than a mains-powered deck
# sitting on a shelf. So these parameters are expected to DRIFT along a
# tape, and that drift is the control variable rather than noise to be
# averaged away.
RF_STAGE_IS_TIME_VARYING = True


def lag_one(series: np.ndarray) -> float:
    """The lag-one autocorrelation of a series."""
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.0
    centred = values - values.mean()
    denominator = float(centred @ centred)
    if denominator <= 0:
        return 0.0
    return float(np.clip(float(centred[:-1] @ centred[1:]) / denominator,
                         -0.99, 0.99))


def effective_sample_size(series: np.ndarray) -> float:
    """How many INDEPENDENT samples a serially correlated series carries.

    A drift test that counts every sample as independent is badly
    overconfident: consecutive fields of one tape are not independent
    draws. With lag-one correlation r the effective count is n(1-r)/(1+r).

    IMPORTANT: this must be given the REGRESSION RESIDUALS, not the raw
    series. A genuine linear ramp is strongly autocorrelated by
    construction, so measuring the raw series would let a real trend
    deflate its own significance - the opposite of the error being guarded
    against. The count is also bounded by the sample size, which the bare
    formula does not guarantee when the correlation is negative."""
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return float(n)
    correlation = lag_one(values)
    estimate = n * (1.0 - correlation) / (1.0 + correlation)
    return float(np.clip(estimate, 1.0, float(n)))


# The critical value for the unit-root test is NOT a constant: it depends on
# how many samples there are. Rejecting a random walk from twenty points
# demands much stronger evidence than from two hundred, and a fixed
# threshold would be too lenient on short series and too strict on long
# ones. These are the standard response-surface coefficients (MacKinnon)
# for the trend-included case, which give the critical value as
#     C(n) = a + b/n + c/n^2
# and reduce to the asymptotic value as the count grows. The count is
# whatever the measurement actually has: fields within one capture, or
# TAPES when the series being tested is the VCR's aging profile across
# recordings.
UNIT_ROOT_SURFACE = {
    0.01: (-3.9638, -8.353, -47.44),
    0.05: (-3.4126, -4.039, -17.83),
    0.10: (-3.1279, -2.418, -7.58),
}
DEFAULT_UNIT_ROOT_LEVEL = 0.05


def unit_root_critical(count: int, level: float = DEFAULT_UNIT_ROOT_LEVEL
                       ) -> float:
    """The unit-root critical value for THIS many samples."""
    asymptotic, first, second = UNIT_ROOT_SURFACE[
        min(UNIT_ROOT_SURFACE, key=lambda p: abs(p - level))]
    n = max(int(count), 4)
    return float(asymptotic + first / n + second / (n * n))


# --------------------------------------------------------------------------
# what a capture of a given length can actually resolve
# --------------------------------------------------------------------------

def drift_resolution(field_count: int, field_rate_hz: float = 59.94
                     ) -> Dict[str, float]:
    """The band of drift rates a capture of this length can see at all.

    The parameters are sampled once per field, so the capture's length sets
    the measurement's own resolution exactly as a sample rate and a record
    length set a spectrum's. Nothing slower than one cycle across the whole
    capture can be distinguished from a constant, and nothing faster than
    half the field rate exists. Stating this first stops a search for a
    thermal drift in a capture far too short to hold one."""
    count = max(int(field_count), 1)
    duration = count / max(field_rate_hz, 1e-9)
    return {
        "fields": float(count),
        "duration_seconds": duration,
        "slowest_resolvable_hz": 1.0 / max(duration, 1e-9),
        "slowest_resolvable_period_seconds": duration,
        "fastest_resolvable_hz": field_rate_hz / 2.0,
        "unit_root_critical": unit_root_critical(count),
    }


def can_resolve(field_count: int, drift_period_seconds: float,
                field_rate_hz: float = 59.94) -> bool:
    """Whether a drift of this period is inside the capture's own band. A
    thermal drift measured in minutes needs a capture measured in minutes;
    no amount of care extracts it from a third of a second."""
    limits = drift_resolution(field_count, field_rate_hz)
    return bool(drift_period_seconds <= limits["duration_seconds"])


def unit_root(series: np.ndarray) -> Dict[str, float]:
    """Whether a series is a random walk rather than a trend.

    Regresses the change on the previous level with a trend term allowed:

        y[t] - y[t-1] = a + b*t + g*y[t-1] + e

    A stationary series pulls back toward its own mean, so g is negative. A
    random walk has g = 0: it never pulls back, and the straight line that
    fits it is an accident of where it wandered. Only a sufficiently
    negative statistic rejects the walk."""
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 8:
        return {"unit_root_t": float("nan"), "random_walk_risk": 1.0}
    change = np.diff(values)
    previous = values[:-1]
    index = np.arange(len(change), dtype=np.float64)
    design = np.column_stack([np.ones_like(previous), index, previous])
    try:
        coefficients, *_ = np.linalg.lstsq(design, change, rcond=None)
        residual = change - design @ coefficients
        degrees = max(len(change) - design.shape[1], 1)
        variance = float(residual @ residual) / degrees
        covariance = variance * np.linalg.pinv(design.T @ design)
        error = float(np.sqrt(max(covariance[2, 2], 1e-30)))
        statistic = float(coefficients[2] / max(error, 1e-30))
    except np.linalg.LinAlgError:
        return {"unit_root_t": float("nan"), "random_walk_risk": 1.0}
    critical = unit_root_critical(n)
    walk = statistic > critical
    out = {"unit_root_t": statistic, "unit_root_critical": critical,
           "random_walk_risk": float(walk)}
    if walk:
        out["warning"] = ("a random walk cannot be ruled out: it has no true "
                          "trend yet fits a line about half the time, so this "
                          "slope is not evidence of drift on its own - it "
                          "needs a longer span")
    return out


def drift(series: np.ndarray, error_bar: Optional[float] = None
          ) -> Dict[str, float]:
    """Whether a per-field series is DRIFTING or merely noisy.

    Reports the slope per field, its standard error, and a t statistic
    scaled to the EFFECTIVE sample size rather than the raw count. Also
    reports the variation against the measurement's own error bar, because
    a spread below the gauge's error is a warning and not a finding."""
    values = np.asarray(series, dtype=np.float64)
    good = np.isfinite(values)
    index = np.arange(len(values), dtype=np.float64)[good]
    values = values[good]
    n = len(values)
    out: Dict[str, float] = {"count": float(n)}
    if n < 4:
        return out
    slope, intercept = np.polyfit(index, values, 1)
    residual = values - (slope * index + intercept)
    spread = float(np.std(residual, ddof=2))
    span = float(np.sum((index - index.mean()) ** 2))
    naive_error = spread / np.sqrt(max(span, 1e-12))
    # the correlation is taken on the RESIDUALS: a real trend must not be
    # allowed to inflate its own error bar
    effective = effective_sample_size(residual)
    inflation = np.sqrt(max(n, 1) / max(effective, 1.0))
    out.update({
        "slope_per_field": float(slope),
        "slope_error": float(naive_error * inflation),
        "t": float(abs(slope) / max(naive_error * inflation, 1e-15)),
        "effective_samples": float(effective),
        "lag1_residual": lag_one(residual),
        "lag1_raw": lag_one(values),
        "standard_deviation": float(np.std(values)),
        "total_change": float(slope * (index[-1] - index[0])),
    })
    # A first-order correction handles correlated NOISE. It does not handle a
    # UNIT ROOT: a random walk has no true trend, yet wanders in a way that
    # fits a straight line about half the time even after the correction.
    # Over a span as short as one decode that is the likeliest way to be
    # fooled. The RAW autocorrelation cannot be used to detect it, because a
    # genuine trend is strongly autocorrelated too - that heuristic flags the
    # real ramp and misses the walk, which is the confound one level up. The
    # actual test regresses the CHANGE on the previous level.
    out.update(unit_root(values))
    if error_bar is not None and error_bar > 0:
        out["variation_over_error"] = float(np.std(values) / error_bar)
        out["resolved"] = float(np.std(values) > 3.0 * error_bar)
    return out


def track(responses_by_field: Sequence[Tuple[np.ndarray, np.ndarray]],
          mechanics: Dict[str, float], free: Sequence[str] = (),
          error_bars: Optional[Dict[str, float]] = None
          ) -> Dict[str, Dict[str, float]]:
    """Fit the head model to EACH field in turn and describe how each
    parameter moves along the tape.

    `responses_by_field` is one (frequency_hz, log_magnitude) pair per
    field, in tape order, with the decoder's own band-pass already divided
    out. Returns, per parameter, the drift description above plus the
    series itself, so a recording device warming up shows as a slope and a
    steady one shows as white scatter about a mean."""
    free = list(free) or ["spacing_m", "gap_m"]
    series: Dict[str, list] = {name: [] for name in free}
    series["explained"] = []
    for frequency, magnitude in responses_by_field:
        fitted = fit(frequency, magnitude, mechanics, free=free)
        for name in free:
            series[name].append(fitted.get(name, np.nan))
        series["explained"].append(fitted.get("explained", np.nan))
    out: Dict[str, Dict[str, float]] = {}
    for name, values in series.items():
        described = drift(np.array(values, dtype=np.float64),
                          (error_bars or {}).get(name))
        described["series"] = np.array(values, dtype=np.float64)
        out[name] = described
    return out


# --------------------------------------------------------------------------
# the aging profile: how far apart two recordings were made
# --------------------------------------------------------------------------

# Heads and tape wear, and debris accumulates, so the EFFECTIVE SPACING that
# the Wallace term measures grows with use. That makes the spacing a clock:
# two recordings made by one machine years apart differ by the wear between
# them. The rate is not universal - it depends on the machine, the tapes and
# the handling - so it must be CALIBRATED from recordings whose dates are
# known, which is exactly the metadata Ethan holds. Once calibrated on one
# VCR the same fit dates that VCR's other recordings, and the video content
# checks the answer without ever having entered it.
def aging_rate(dates_years: Sequence[float],
               spacings_m: Sequence[float],
               spacing_errors_m: Optional[Sequence[float]] = None
               ) -> Dict[str, float]:
    """Calibrate how fast the effective spacing grows, from recordings whose
    dates are known. Needs at least two, and is only meaningful within ONE
    recording machine - two machines differ by construction, not by age."""
    years = np.asarray(dates_years, dtype=np.float64)
    spacing = np.asarray(spacings_m, dtype=np.float64)
    good = np.isfinite(years) & np.isfinite(spacing)
    if good.sum() < 2:
        return {"usable": 0.0,
                "why": "an aging rate needs at least two dated recordings "
                       "from the same machine"}
    years, spacing = years[good], spacing[good]
    weights = None
    if spacing_errors_m is not None:
        errors = np.asarray(spacing_errors_m, dtype=np.float64)[good]
        weights = 1.0 / np.maximum(errors, 1e-15) ** 2
    slope, intercept = np.polyfit(years, spacing, 1, w=weights)
    predicted = slope * years + intercept
    residual = spacing - predicted
    degrees = max(len(years) - 2, 1)
    spread = float(np.sqrt(float(residual @ residual) / degrees))
    span = float(np.sum((years - years.mean()) ** 2))
    return {
        "usable": 1.0,
        "metres_per_year": float(slope),
        "metres_per_year_error": float(spread / np.sqrt(max(span, 1e-12))),
        "intercept_m": float(intercept),
        "residual_m": spread,
        "recordings": float(len(years)),
    }


def interval_years(spacing_a_m: float, spacing_b_m: float,
                   rate: Dict[str, float],
                   spacing_error_m: float = 0.0) -> Dict[str, float]:
    """How many years apart two recordings were made, from their fitted
    spacings and a calibrated wear rate.

    The uncertainty carries BOTH sources: the spacing measurement's own
    error and the rate calibration's. Reporting the interval without them
    would be the same mistake as reporting a residual below its gauge's
    error."""
    if not rate.get("usable"):
        return {"usable": 0.0, "why": rate.get("why", "no calibrated rate")}
    per_year = float(rate["metres_per_year"])
    if abs(per_year) < 1e-15:
        return {"usable": 0.0, "why": "the calibrated wear rate is zero"}
    difference = float(spacing_b_m) - float(spacing_a_m)
    years = difference / per_year
    measurement = np.sqrt(2.0) * float(spacing_error_m) / abs(per_year)
    calibration = (abs(years) * float(rate.get("metres_per_year_error", 0.0))
                   / abs(per_year))
    return {
        "usable": 1.0,
        "years": float(years),
        "years_error": float(np.hypot(measurement, calibration)),
        "from_measurement": float(measurement),
        "from_calibration": float(calibration),
    }


def datable_resolution(spacing_error_m: float,
                       metres_per_year: float) -> Dict[str, float]:
    """The finest interval this method could resolve at all, given how well
    the spacing is measured and how fast it grows. It is the information
    boundary of the aging clock, and it is worth stating BEFORE any dating
    is attempted, exactly as a capture's length is stated before a drift is
    looked for."""
    if abs(metres_per_year) < 1e-15:
        return {"years": float("inf")}
    smallest = np.sqrt(2.0) * float(spacing_error_m) / abs(metres_per_year)
    return {"smallest_resolvable_years": float(smallest),
            "three_sigma_years": float(3.0 * smallest)}

