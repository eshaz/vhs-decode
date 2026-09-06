"""The head pair as a differential: its amplitude, its phase, its frequency.

Ethan: *"It will be the amplitude, phase, and frequency of this differential
pair. We have three sets of three differential pairs, which form a matrix to
calculate over the residuals."* This module is the head pair's set of three.

A drum carries two video heads and they are not identical. Everything the two
share cancels in their difference - the record-side path, the tape, the
decoder's own filters, the de-emphasis - so the difference is the honest
quantity, and what survives it belongs to the heads alone. Three real
physical parameters can survive it, and each has its own closed form:

    amplitude   a head-to-tape SEPARATION difference, Wallace's law
                exp(-2 pi dd / lambda), the group being dd/lambda
    phase       a pure DELAY difference between the two heads' channels
                exp(-2 pi j f tau), the group being f tau
    frequency   a fractional SCALING of the frequency axis between them
                Rbar(f (1+s)) / Rbar(f), the group being s and the shared
                response's own log-log slope

WHAT WAS ALREADY ESTABLISHED, and what this module reproduces. From
`docs/RESIDUAL_LIMIT_DESIGN.md` sections 3a.1, 4.2g and 4.2h, measured on
consecutive head pairs - each pair is one drum revolution - over 26 and 27
pairs of two SP decodes:

    static differential, shape         0.0936 dB rms   0.0949 dB rms
    static differential, level        -0.5558 dB      -0.3380 dB
    varying part, per pair             0.0841 dB rms   0.1350 dB rms
    first half against second        r = +0.902      r = +0.958
    varying part a frequency shift explains   0.40%        0.66%

and the phase, from the sync-edge instrument's per-head complex response,
which is content-free by construction: every entry sits within plus or minus
1.7 degrees on all four captures and both edges once a delay of a few
nanoseconds is removed. That is the signature of a SPACING difference and not
of a gap or an azimuth difference, because Wallace's law is a real
attenuation and carries no phase while a gap or azimuth error would turn the
phase through its sinc null.

AND THE DIFFERENCE IS RECORD-SIDE. Two tapes played back on one deck give
head-difference vectors that do not correlate at all, r = +0.11 on 54 degrees
of freedom, so no shared playback component is resolved above the split
error. The constant is therefore held PER RECORDING and never carried across
tapes - `information_extrapolation.HEAD_DIFFERENCE_SCOPE` states the same
rule from the other side.

WHY AZIMUTH IS NOT ONE OF THE THREE, though it is the mechanism the format
seems to offer. Azimuth loss is `sinc(W tan(dtheta) / lambda)`, and `sinc` is
EVEN while `tan` is odd, so the loss is even in the azimuth error. VHS writes
adjacent tracks at opposite azimuth, so a single misalignment of the drum
reaches head A as `+delta` and head B as `-delta`, and evenness makes their
magnitude responses IDENTICAL. The whole of a systematic azimuth error
cancels in the head difference, exactly, and azimuth survives into the
difference only through the two heads' independent construction errors.
`azimuth_cancels_control` below is that statement as a runnable control, and
it is one of the two controls this module keeps as code.

THE MEASURED EVIDENCE THIS MODULE'S OWN RUNS PRODUCE, so that a reader after
a restart does not have to re-derive it. On the VHS luma RF band, 1 to 7 MHz
over 512 places, VHS NTSC SP mechanics:

    all three signatures        2.2505 effective of 3, condition 2.413
    singular values             1.3064, 1.0000, 0.5415
    worst coherent pair         amplitude against frequency, 0.70541
    amplitude and phase alone   2.0000 of 2, condition 1.000, coherence 8e-17
    amplitude and frequency     1.3355 of 2, condition 2.406
    phase and frequency         1.9961 of 2, condition 1.045
    a family of five SPACINGS   1.0000 of 5 - one direction, as the
                                separability law requires of a rate family
    azimuth control             a common misalignment cancels to 0.0e+00
                                nepers rms, while independent construction
                                errors of 0.25 and 0.15 degrees leave 4.9e-02
    Wallace round trip          -0.5558 dB -> 14.767 nm -> -0.5558 dB, 1.1e-16
    band dependence             2.2505 on 1-7 MHz, 2.2506 on 0.2-7,
                                2.2519 on 0.1-10, 2.2501 on the carrier alone

THESE NUMBERS ARE THE RECHARACTERISED ONES AND THE CONCLUSION THEY SUPPORT IS
THE OPPOSITE OF THE ONE THEY FIRST SUPPORTED. `frequency_signature` built
`exp(there - here)` from two log MAGNITUDES and cast the real result to
complex, so the entry arrived with no phase. It read 1.8035 of 3 at condition
28.84, the amplitude-frequency pair sat at 0.99760, and this module's central
finding was that the frequency scaling is another RATE and not another kind.

A FREQUENCY SCALING IS THE ONE CASE WHERE DROPPING THE PHASE IS NOT
RECOVERABLE. For most entries a minimum-phase response's phase is implied by
its magnitude, so a magnitude-only signature loses nothing that cannot be
reconstructed. Here the two sides are the SAME response at two different
frequencies, so what the entry carries is `phi(s f) - phi(f)`, and that is
not implied by the magnitude difference - it is a separate half of the same
operation. Measured at a fractional scaling of 1e-3, the magnitude difference
peaks at 0.00092 nepers and the phase difference at 0.00362 radians: THE
SIGNATURE WAS BLIND TO ABOUT FOUR FIFTHS OF ITS OWN EFFECT.

Carrying it, the amplitude-frequency pair falls from 0.99760 to 0.70541 and
the three entries read 2.2505 of 3 at condition 2.413. The frequency scaling
IS another kind after all, and the kind is a phase.

WHAT SURVIVES OF THE OLD READING. The separability argument below is still
correct about the MAGNITUDES - scaling the axis scales the same dimensionless
group a length does, so the magnitude halves really are one direction - and
the band dependence is still almost flat, 2.2501 to 2.2519 across a hundredfold
change of fractional bandwidth, which is the same insensitivity the magnetics
found. What was wrong was reading a magnitude-only collinearity as a statement
about the mechanism.

SO THE HONEST ANSWER IS TWO NEW KINDS AND THE MAGNITUDES SHARING ONE, and it
took the measurement twice to find that out. The delay is a genuinely new
kind: it is orthogonal to both magnitude quantities to 8e-17 once real and
imaginary parts are stacked, because a phase and a magnitude are two
mechanisms - which is why `real_parameters=True` is not optional here, and
why amplitude and phase together read a perfect 2.0000 of 2 at condition
1.000. The frequency scaling is a new kind TOO, and only its magnitude half is
not. It sits at 0.70541 with the separation and the two together read 1.3355
of 2 - not one direction wearing two names, but two directions sharing most
of one. Against the phase it reads 1.9961 of 2, essentially orthogonal.

The separability law predicts exactly this, and the prediction is worth
stating because it was not obvious in advance. Every mechanism in the shared
head response is a function of one dimensionless group, a length over the
recorded wavelength. Scaling the frequency axis scales that group - which is
the same thing scaling the length does. So a fractional frequency difference
between the heads acting on a response built out of rate mechanisms IS
another rate, and it lands inside the family that gave 1.58 effectively
distinguishable of 6 rather than beside it. What little separates the two is
the shared response's CURVATURE, contributed by the gap and thickness terms,
and widening the band from 7:1 to 100:1 moves the pair only from 0.99760 to
0.99110. Fractional bandwidth is the only variable that moves it at all,
which is the same finding the magnetics arrived at.

ADDING THE FREQUENCY QUANTITY TO THE KEY NOW RAISES THE COUNT: 2.0000 of 2 at
condition 1.000 becomes 2.2505 of 3 at condition 2.413. It used to LOWER it,
to 1.8035 at condition 28.84, and that was the independent reason given here
for keeping it out.

THAT REASON IS GONE AND THE QUANTITY IS STILL OUT, on a narrower argument
that has to be stated rather than inherited. It buys +0.2505 of a direction
for 2.41 times the conditioning. The admission gate this arc applies
elsewhere - `interference.admission` - asks for more than half a direction of
gain at no more than 1.25 times the cost, and this clears neither. So it is
held, as it was, but because it does not pay rather than because it adds
nothing; and the size argument below is now the primary reason rather than
the secondary one.

THE FREQUENCY QUANTITY IS REPORTED AS A BOUND, NOT AS A CORRECTION. Three
things say so and they agree. The first is the prior measurement: a frequency
shift explains 0.40 to 0.66 per cent of the varying part, and the varying
part is itself 0.084 to 0.135 dB rms against a per-bin measurement noise
independently estimated at about 0.096 dB, so the varying part is consistent
with being noise entire. The second is the conditioning just described. The
third is geometry, and it is much the sharpest: a head that stands proud of
the drum by `dr` more than its partner turns at a tangential speed larger by
`dr/R`, and `dr` is the same protrusion difference Wallace's law already
measured from the level split. From this module's own run,

    spacing difference from the level      14.767 nm
    drum radius, JVC VTG82063 section 1    31.0 mm
    implied fractional speed difference    4.764e-07
    the log-magnitude shape it produces    1.642e-06 dB rms on this band

against a varying part of 0.0841 dB rms. The prediction is FIFTY-ONE THOUSAND
times below the noise. Even the drum's own measured off-axis error - 1.5 um
of arc, section 4.2g - gives 4.839e-05 and a shape of 1.668e-04 dB rms, still
five hundred and four times below. And the estimator's own error bar agrees
from the other side: with 0.0841 dB rms of noise over 512 places and a
variance inflation of 208.5 from the collinearity, the standard error on the
log shift is 1.55e-02, so the data alone bound the scaling only at about
5e-02 three sigma - a hundred thousand times looser than geometry does.

So the frequency quantity is real physics with a computable size, and that
size is far beneath what any measurement in this arc can see. It is not
merely unmeasured; it is unmeasurable at this evidence level, and by the
square-root law the fields needed to see the drum-off-axis case would be
about 2.5e+05 times what was used. `frequency_verdict` returns a bound and
refuses to promote it without an out-of-sample test that clears.

THE TWO STANDING RULES THE ESTIMATOR RESPECTS. A DIFFERENCE, NEVER A RATIO:
spectral division blows up wherever the denominator is small and that is the
mechanism that manufactured phantom features in this project before
(`docs/FREQUENCY_DERIVATION.md` section 2), so the estimator works on
`log|A| - log|B|`, a subtraction, throughout. NO SEEDING FROM SPECTRAL PEAKS:
the shift is not found by locating a feature in either response and matching
it, but by one linear projection onto the shift direction, jointly with the
spacing direction so that the two cannot steal from one another.

THE WAY IN IS THE LOG-MAGNITUDE DOMAIN. On `u = ln f` a scaling of the
frequency axis is a SHIFT, `Lbar(f(1+s))` is `Lbar` evaluated at `u + ln(1+s)`,
and to first order the difference it makes is `ln(1+s)` times `dLbar/du`. So
the estimator is a projection onto one known vector, solved jointly with the
spacing direction so that neither can steal from the other, and the whole
question becomes how far those two vectors are apart - measured at 0.99760,
a variance inflation of 208.5. Noiseless, the estimator recovers a planted
scaling of 1.0e-03 as 1.0015e-03, the 0.15 per cent being the first-order
linearisation and nothing else.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from vhsdecode.models import head_model


# --------------------------------------------------------------------------
# What has been measured. Every figure here is a MEASUREMENT with its source,
# never a tuning constant: the caller supplies its own tape's numbers and the
# defaults below only make the module runnable on the evidence in hand.
# --------------------------------------------------------------------------

# docs/RESIDUAL_LIMIT_DESIGN.md section 4.2h, table "The head differential,
# split as Ethan predicted", 75 bars SP and chromanoise SP, 26 and 27
# consecutive head pairs.
MEASURED_SHAPE_DB_RMS = (0.0936, 0.0949)
MEASURED_LEVEL_DB = (-0.5558, -0.3380)
MEASURED_VARYING_DB_RMS = (0.0841, 0.1350)
MEASURED_HALF_AGAINST_HALF_R = (0.902, 0.958)
MEASURED_SHIFT_EXPLAINED = (0.0040, 0.0066)

# docs/RESIDUAL_LIMIT_DESIGN.md section 3a.1: the per-head complex response
# from the sync-edge instrument, as A/B with a pure delay removed. Four
# captures, falling edge.
MEASURED_PHASE_BOUND_DEGREES = 1.7
MEASURED_DELAYS_S = (1.5e-9, 0.0, -1.0e-9, -7.1e-9)

# docs/RESIDUAL_LIMIT_DESIGN.md section 4.2g: the drum's own off-axis error,
# 0.0028 degrees of rotation, which on the 62 mm drum is 1.5 um of arc.
MEASURED_DRUM_OFF_AXIS_M = 1.5e-6

# information_extrapolation.HEAD_DIFFERENCE_SCOPE, and the measurement behind
# it: two tapes on one deck give head-difference vectors correlating at
# r = +0.11, t = 0.8 on 54 degrees of freedom.
MEASURED_TWO_TAPE_R = 0.11
SCOPE = "per recording"

# The band the head difference was measured over, and the band this module's
# own evidence is quoted on: the VHS luma RF band, which is what the rest of
# this arc uses (docs/COMPONENT_MAPPINGS.md section 1).
EVIDENCE_BAND_HZ = (1e6, 7e6)
EVIDENCE_PLACES = 512

# Wallace 1951, "The Reproduction of Magnetically Recorded Signals", Bell
# System Technical Journal 30, 1145-1173: the separation loss is
# exp(-2 pi d / lambda), so one wavelength of separation costs 2 pi nepers.
# In decibels that is 54.59, which is the figure section 4.2g inverts to turn
# the measured level split into a spacing. It is DERIVED here rather than
# written down, so a wrong conversion cannot hide in it.
NEPERS_PER_WAVELENGTH_OF_SEPARATION = 2.0 * np.pi
DB_PER_WAVELENGTH_OF_SEPARATION = (NEPERS_PER_WAVELENGTH_OF_SEPARATION
                                   * head_model.DB_PER_NEPER)


def _grid(frequency_hz) -> np.ndarray:
    return np.asarray(frequency_hz, dtype=np.float64).ravel()


def mechanics(system: str = "NTSC", tape_speed: str = "SP",
              rf_params: Optional[Dict[str, float]] = None
              ) -> Dict[str, float]:
    """The format's mechanics, taken from the head model rather than restated.

    A second copy of the drum diameter or the writing speed in this file
    could drift away from the head model's silently, and both this module's
    geometric bound and its wavelengths depend on them.
    """
    found = head_model.mechanics_for("VHS", system, tape_speed, rf_params)
    if found is None:
        raise ValueError(f"no VHS mechanics for {system!r} at {tape_speed!r}")
    return found


# --------------------------------------------------------------------------
# 1. AMPLITUDE - a separation difference, by Wallace's law
# --------------------------------------------------------------------------


def amplitude_signature(frequency_hz, spacing_difference_m: float,
                        writing_speed_m_s: float) -> np.ndarray:
    """The two heads' separation difference, as a complex signature.

    Wallace 1951: a head standing `d` from the medium loses
    `exp(-2 pi d / lambda)`, and with `lambda = v/f` the whole dependence is
    on the single dimensionless group `dd f / v`, the separation difference
    measured in recorded wavelengths.

    IT IS PURELY REAL, and that is the measurement rather than a
    simplification. Wallace's law is an attenuation with no phase, and the
    per-head complex response measured on four captures sits within plus or
    minus 1.7 degrees of zero phase once a delay is removed. A gap-length or
    an azimuth difference would have turned the phase through its sinc null,
    so the absence of phase is what identifies this as a separation.

    SUBTRACTABLE by construction: an exponential of a finite quantity is
    strictly positive, so its logarithm is finite everywhere on the band. It
    is a `1 - d*shape` style loss written in its exact form rather than
    approximated, since the exact form is already bounded.

    THE SIDE IS THE RECORD SIDE, and only a two-tape test says so. A
    separation at playback has an identical signature - the law does not know
    which machine the gap was in - and the two are told apart only by whether
    the difference travels with the deck or with the tape. Measured, it
    travels with the tape: r = +0.11 across two tapes on one deck.
    """
    f = _grid(frequency_hz)
    wavelengths = float(spacing_difference_m) * f / float(writing_speed_m_s)
    return np.exp(-NEPERS_PER_WAVELENGTH_OF_SEPARATION
                  * wavelengths).astype(np.complex128)


def spacing_from_level(level_db: float, frequency_hz: float,
                       writing_speed_m_s: float) -> float:
    """Invert Wallace's law: what separation difference makes this level.

    Section 4.2g's own arithmetic, as code. A level split of `L` decibels at
    a frequency whose wavelength is `lambda` is `L / 54.59` wavelengths of
    separation, and 54.59 is `2 pi` nepers expressed in decibels rather than
    a figure to be looked up.

    Measured: the -0.5558 dB level of the 75 bars SP decode, read at the
    4 MHz carrier on a 5.80 m/s head, gives 14.767 nm - inside the 13.5 to
    39.8 nm the level split gave for the wider -0.51 to -1.50 dB range, which
    this function reproduces as 13.55 and 39.85 nm.
    """
    wavelength = float(writing_speed_m_s) / max(float(frequency_hz), 1.0)
    return abs(float(level_db)) / DB_PER_WAVELENGTH_OF_SEPARATION * wavelength


def wallace_consistency(level_db: float, shape_db_rms: float,
                        centre_hz: float) -> Dict[str, float]:
    """Is the measured level and shape consistent with ONE separation?

    A test the measurement can fail. If the head difference is a pure
    separation then its log magnitude is a straight line through the origin
    in frequency, so over a band of half-width `w` about `fc` the shape - the
    departure from the band's mean - is a ramp whose root mean square is
    `(w/fc)/sqrt(3)` times the level. The measured level and the measured
    shape therefore IMPLY the band the measurement was made over, and if that
    implied band is not the band the instrument actually used, the difference
    is not a pure separation.

    Measured on 75 bars SP: level -0.5558 dB, shape 0.0936 dB rms, centre
    4 MHz gives an implied half-width of 1.1668 MHz, that is a band of 2.833
    to 5.167 MHz. That is the VHS SP luma carrier band with its first
    sidebands,
    which is where the response was measured, so the pure-separation reading
    survives its own consistency test. It could have failed: a half-width of
    5 MHz or of 100 kHz would both have refuted it.
    """
    level = abs(float(level_db))
    if not level > 0:
        return {"implied_half_width_hz": float("nan"),
                "why": "a zero level implies no band"}
    ratio = float(shape_db_rms) / level
    half_width = ratio * np.sqrt(3.0) * float(centre_hz)
    return {
        "implied_half_width_hz": float(half_width),
        "implied_band_hz": (float(centre_hz - half_width),
                            float(centre_hz + half_width)),
        "shape_over_level": float(ratio),
        "why": ("a pure separation is a straight line through the origin, so "
                "its shape over its level fixes the band it was measured on"),
    }


# --------------------------------------------------------------------------
# 2. PHASE - a pure delay between the two heads' channels
# --------------------------------------------------------------------------


def phase_signature(frequency_hz, delay_difference_s: float) -> np.ndarray:
    """The two heads' delay difference, as a complex signature.

    `exp(-2 pi j f tau)`: unit magnitude, and a phase linear in frequency.
    The dimensionless group is `f tau`, the delay measured in periods.

    THIS IS THE ONLY ONE OF THE THREE WITH ANY PHASE, and that is what makes
    the set worth stacking as real parameters. Under the Hermitian inner
    product a pure-amplitude signature and a pure-phase signature of the same
    shape are the SAME direction, since `|<r, jr>| = ||r||^2`; stacked as
    `[Re; Im]` they are orthogonal. A separation and a delay are two
    mechanisms and only the stacked form says so.

    THE SIZE IS MEASURED, THE SIDE IS NOT. The four captures give delays of
    +1.5, 0.0, -1.0 and -7.1 nanoseconds, and after removing them the residual
    phase is within plus or minus 1.7 degrees everywhere - so the delay is
    real and everything else about the phase is zero. Which machine it lives
    in is not determined: the same tape at SP and at EP gives -1.0 and -7.1 ns,
    and a playback amplifier's path length does not change with tape speed,
    which argues for the record side; but the two-tape test that settled the
    amplitude's side has never been run on the delay.

    IT DOES NOT MATTER MUCH, and it is worth saying why. A pure delay is
    diagonal in frequency and commutes with every other entry in the key,
    all of which are also diagonal in frequency, so its position in the chain
    changes nothing about the inverse. The order still has to be DECLARED -
    an undeclared entry is refused by `ordered_key` - but of the three
    quantities here this is the one whose declaration is least consequential.
    """
    f = _grid(frequency_hz)
    return np.exp(-2j * np.pi * f * float(delay_difference_s))


def residual_phase_bound(measured_phase_rad, frequency_hz
                         ) -> Dict[str, float]:
    """Remove the best-fitting pure delay and report what phase is left.

    The measurement the module rests on, as an instrument rather than a
    quotation. A delay is a phase linear in frequency through the origin, so
    the fit is one projection; what remains is EXCESS phase, and it is the
    quantity that would have shown a gap or an azimuth difference.

    Measured on the four captures the design document reports, the excess is
    within plus or minus 1.7 degrees - which is the evidence that the
    amplitude difference is a separation.
    """
    f = _grid(frequency_hz)
    phase = np.asarray(measured_phase_rad, dtype=np.float64).ravel()
    good = np.isfinite(f) & np.isfinite(phase)
    if good.sum() < 2:
        return {"delay_s": 0.0, "excess_degrees": float("nan"),
                "why": "too few places to remove a delay"}
    # phase = -2 pi f tau, so tau is one projection onto f. No constant is
    # allowed: a constant phase is not a delay, and letting one in would let
    # the fit absorb half an excess.
    design = f[good]
    slope = float(design @ phase[good]) / float(design @ design)
    excess = phase[good] - slope * design
    return {
        "delay_s": float(-slope / (2.0 * np.pi)),
        "excess_rms_degrees": float(np.degrees(np.sqrt(np.mean(excess ** 2)))),
        "excess_peak_degrees": float(np.degrees(np.max(np.abs(excess)))),
        "why": ("a delay is a phase linear in frequency; what is left after "
                "removing it is excess phase, and a separation has none"),
    }


# --------------------------------------------------------------------------
# 3. FREQUENCY - a fractional scaling of the axis, never measured before
# --------------------------------------------------------------------------


def shared_log_response(frequency_hz, writing_speed_m_s: float,
                        track_width_m: float, **parameters) -> np.ndarray:
    """The response the two heads SHARE, in nepers.

    The frequency quantity is a rescaling of the axis, and a rescaling of the
    axis does nothing at all unless there is a shape to move. So this
    signature, alone of the three, needs the shared response as an input: it
    is the derivative of that response, not a shape in its own right.

    Taken from the head model with its own typical parameters unless the
    caller supplies fitted ones, which is what a fitted per-tape run would do.
    """
    values = dict(head_model.TYPICAL)
    values.update(parameters)
    return head_model.log_response(_grid(frequency_hz), writing_speed_m_s,
                                   track_width_m, **values)


def shared_phase(frequency_hz, writing_speed_m_s: float,
                 track_width_m: float, **parameters) -> np.ndarray:
    """The phase of the response the two heads share, in radians.

    The partner to `shared_log_response`, and needed for the same reason the
    frequency-scaling signature exists at all: a rescaling of the axis moves
    whatever shape is there, and the phase is part of the shape.
    """
    values = dict(head_model.TYPICAL)
    values.update(parameters)
    return head_model.phase_rad(_grid(frequency_hz), writing_speed_m_s,
                                track_width_m, **values)


def shift_direction(frequency_hz, writing_speed_m_s: float,
                    track_width_m: float, **parameters) -> np.ndarray:
    """`dLbar/du` with `u = ln f`: the direction a frequency shift moves in.

    THE LOG-MAGNITUDE DOMAIN MAKES A SCALING A SHIFT. On `u = ln f`,
    `Lbar(f(1+s))` is `Lbar(u + ln(1+s))`, so to first order the difference a
    scaling makes is `ln(1+s)` times this vector. That is the whole way in,
    and it is why the estimator is one projection rather than a search.

    Computed by differencing the shared response on a uniform `u` grid of the
    same places, so it needs no analytic derivative of a model that may be
    replaced.
    """
    f = _grid(frequency_hz)
    good = f > 0
    u = np.log(np.maximum(f, 1e-30))
    response = shared_log_response(f, writing_speed_m_s, track_width_m,
                                   **parameters)
    response = np.where(np.isfinite(response), response, 0.0)
    derivative = np.gradient(response, u)
    return np.where(good, derivative, 0.0)


def frequency_signature(frequency_hz, fractional_scaling: float,
                        writing_speed_m_s: float, track_width_m: float,
                        **parameters) -> np.ndarray:
    """The two heads' frequency-axis scaling, as a complex signature.

    Head B's response is head A's evaluated at `f (1+s)`, so the signature is
    the exact difference of the two shared responses in the log domain,
    exponentiated:

        exp( Lbar(f (1+s)) - Lbar(f) )

    A DIFFERENCE, never a ratio, even here where a ratio would have been the
    obvious way to write it - the two forms are the same only when nothing is
    small, and the whole reason for the rule is what happens when something is.

    WHAT PHYSICALLY SCALES THE AXIS. Every loss in the head model is a
    function of a length over the recorded wavelength, `l f / v`. A head
    turning at a slightly different tangential speed therefore sees every one
    of them at a scaled frequency, and nothing else changes. A head that
    stands proud of the drum by `dr` more than its partner turns at
    `dr / R` more speed, and that is the same protrusion difference the
    amplitude quantity already measures as a separation - which is why the two
    are not independent and why `geometric_bound` can predict this one's size
    from the other one's measurement.

    THE SCALING IS EXACTLY DEGENERATE with a fractional change of the writing
    speed and with a simultaneous fractional change of EVERY length in the
    head model, because only the ratio appears. Nothing measured on one head
    pair separates those three readings; they are one parameter under three
    names.

    SUBTRACTABLE: a difference of two finite log responses is finite, so the
    exponential is strictly positive. Where the shared response is not finite
    - at or below zero frequency, or at a gap null if one falls inside the
    band - the difference is undefined and the signature returns unity there,
    which asserts no difference where nothing is known rather than clamping a
    logarithm.
    """
    f = _grid(frequency_hz)
    scale = 1.0 + float(fractional_scaling)
    here = shared_log_response(f, writing_speed_m_s, track_width_m,
                               **parameters)
    there = shared_log_response(f * scale, writing_speed_m_s, track_width_m,
                                **parameters)
    exponent = there - here
    exponent = np.where(np.isfinite(exponent), exponent, 0.0)

    # AND THE PHASE, which this returned without. It built `exp(there - here)`
    # from two log MAGNITUDES and cast the real result to complex - a real
    # number wearing a complex dtype.
    #
    # A FREQUENCY SCALING IS EXACTLY THE CASE WHERE THAT IS NOT HARMLESS. For
    # most operations a minimum-phase response's phase is implied by its
    # magnitude, so carrying only the magnitude loses nothing that cannot be
    # recovered. Here the two sides are the SAME response at two different
    # frequencies, so the phase difference is `phi(s f) - phi(f)`, which is
    # not zero for any response with a shape - it is the very quantity a
    # scaling moves. Dropping it made the signature blind to the half of the
    # scaling that shows as a delay.
    phase_here = shared_phase(f, writing_speed_m_s, track_width_m,
                              **parameters)
    phase_there = shared_phase(f * scale, writing_speed_m_s, track_width_m,
                               **parameters)
    turn = phase_there - phase_here
    turn = np.where(np.isfinite(turn), turn, 0.0)
    return np.exp(exponent + 1j * turn)


def geometric_bound(spacing_difference_m: float,
                    drum_diameter_m: float) -> Dict[str, float]:
    """How large the frequency scaling CAN be, from geometry alone.

    The bound that decides the question, and it is arithmetic rather than a
    fit. The heads sit on a drum of radius `R` and sweep at `omega R`. A head
    standing proud by `dr` more than its partner sweeps at `omega (R + dr)`,
    a fractional difference of `dr / R`. And `dr` is not a free quantity: a
    head standing proud presses harder and sits closer, so it is the same
    protrusion difference the amplitude quantity measures as a separation -
    `head_model.spacing_from_protrusion` states that relation, with the sign,
    from the other side.

    So the amplitude measurement PREDICTS the frequency quantity's size, and
    the prediction is decisive. Measured: 14.767 nm of separation on a
    31.0 mm drum radius gives 4.764e-07, which on the luma RF band moves the
    log magnitude by 1.642e-06 dB rms against a varying part of 0.0841 dB
    rms. Fifty-one thousand times below the noise.

    The drum's own measured off-axis error gives the looser bound and is
    reported beside it: 1.5 um of arc on the same radius is 4.839e-05, and
    1.668e-04 dB rms, still five hundred and four times below.
    """
    radius = 0.5 * float(drum_diameter_m)
    if not radius > 0:
        raise ValueError("a drum radius must be positive")
    return {
        "from_the_measured_separation": abs(float(spacing_difference_m)) / radius,
        "from_the_drum_off_axis": MEASURED_DRUM_OFF_AXIS_M / radius,
        "drum_radius_m": radius,
        "separation_m": abs(float(spacing_difference_m)),
        "why": ("a head standing proud by dr sweeps at (R+dr)/R times the "
                "speed, and dr is the protrusion the separation measures"),
    }


def estimate_frequency_shift(frequency_hz, log_magnitude_difference,
                             writing_speed_m_s: float, track_width_m: float,
                             **parameters) -> Dict[str, object]:
    """Estimate the frequency scaling from a measured head difference.

    THE TWO STANDING RULES, and how each is kept.

    A DIFFERENCE, NEVER A RATIO. The input is `log|A| - log|B|` and every
    operation on it is linear. Nothing divides one spectrum by another at any
    point, so there is no denominator to be small.

    NO SEEDING FROM SPECTRAL PEAKS. The shift is not found by locating a
    feature in either response and matching it. It is one linear projection
    onto a direction the physics fixes in advance - `dLbar/du`, the shared
    response's own log-log slope - so the estimate does not depend on the
    measurement having any feature at all.

    AND THE SPACING IS FITTED JOINTLY, which is the part that makes the
    answer mean anything. A separation difference contributes `-2 pi dd f / v`
    and a frequency shift contributes `ln(1+s) dLbar/du`, and where the shared
    response is spacing-dominated those two are the SAME vector: `dLbar/du` is
    then itself proportional to `f`, and no measurement whatever can separate
    them. What separates them is the shared response's CURVATURE, which comes
    from the gap and thickness terms. Measured on the luma band with the head
    model's typical parameters the two directions sit at 0.99760, a variance
    inflation of 208.5, and the reported error carries that inflation - which
    is why the standard error on the log shift is 1.55e-02 where the same
    noise on an orthogonal direction would have given 1.08e-03.

    A level is removed from both before the solve, because a flat gain
    difference between the heads is neither a spacing nor a shift and would
    otherwise be split between them.

    THE ERROR BAR ASSUMES INDEPENDENT PLACES and is therefore optimistic:
    neighbouring bins of a measured response are correlated, so the true
    error is larger than the one returned by however much correlation there
    is. That makes the BOUND this function reports conservative in the wrong
    direction, which is why `frequency_verdict` refuses to promote on it
    alone.
    """
    f = _grid(frequency_hz)
    measured = np.asarray(log_magnitude_difference, dtype=np.float64).ravel()
    good = np.isfinite(f) & (f > 0) & np.isfinite(measured)
    if good.sum() < 4:
        return {"usable": False, "why": "too few finite places to solve"}
    f, measured = f[good], measured[good]

    # the two directions, both mean-removed: a level is not a mechanism
    spacing = -NEPERS_PER_WAVELENGTH_OF_SEPARATION * f / float(writing_speed_m_s)
    shift = shift_direction(f, writing_speed_m_s, track_width_m, **parameters)
    spacing = spacing - spacing.mean()
    shift = shift - shift.mean()
    target = measured - measured.mean()

    norms = np.array([np.linalg.norm(spacing), np.linalg.norm(shift)])
    if not np.all(norms > 0):
        return {"usable": False,
                "why": "one of the directions has no shape on this band"}
    coherence = abs(float(spacing @ shift) / float(norms[0] * norms[1]))
    inflation = 1.0 / max(1.0 - coherence ** 2, 1e-30)

    # THE COLUMNS MUST BE SCALED BEFORE THE SOLVE. A spacing is in metres and
    # a shift is dimensionless, so the two columns' norms differ by thirteen
    # orders of magnitude; the normal matrix formed from them is numerically
    # singular and its pseudo-inverse silently discards the shift's own
    # direction, returning an error bar of 1e-17 on a quantity that is not
    # determined at all. Solved on unit columns and rescaled afterwards, the
    # same fit returns 1.55e-02, which is the truth and is what the noise and
    # the variance inflation together predict.
    design = np.column_stack([spacing, shift])
    scale = np.linalg.norm(design, axis=0)
    unit = design / scale
    solution, *_ = np.linalg.lstsq(unit, target, rcond=None)
    solution = solution / scale
    residual = target - design @ solution
    places = int(len(target))
    degrees = max(places - design.shape[1], 1)
    variance = float(residual @ residual) / degrees
    try:
        normal = np.linalg.inv(unit.T @ unit)
    except np.linalg.LinAlgError:
        # exactly collinear directions: the shift is not determined at all,
        # and the pseudo-inverse's zero error bar would say the opposite
        return {"usable": False,
                "why": "the spacing and shift directions are exactly "
                       "collinear on this band, so the shift is not "
                       "determined"}
    covariance = variance * normal
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0)) / scale

    shift_log = float(solution[1])                 # this is ln(1+s)
    fractional = float(np.expm1(shift_log))
    explained = 1.0 - float(residual @ residual) / max(float(target @ target),
                                                       1e-30)
    return {
        "usable": True,
        "fractional_scaling": fractional,
        "log_shift": shift_log,
        "log_shift_error": float(errors[1]),
        "spacing_difference_m": float(solution[0]),
        "spacing_difference_error_m": float(errors[0]),
        "explained": float(explained),
        "coherence_with_spacing": coherence,
        "variance_inflation": float(inflation),
        "sigma": float(abs(shift_log) / max(float(errors[1]), 1e-30)),
        # the honest upper bound on the size of the scaling: the estimate plus
        # three of its standard errors, not the estimate itself
        "three_sigma_bound": float(np.expm1(abs(shift_log)
                                            + 3.0 * float(errors[1]))),
        "places": places,
        "why": ("one projection onto the shared response's log-log slope, "
                "solved jointly with the spacing so neither steals from the "
                "other; a difference throughout and no peak was sought"),
    }


def out_of_sample_shift(frequency_hz, first_half_difference,
                        second_half_difference, writing_speed_m_s: float,
                        track_width_m: float, **parameters
                        ) -> Dict[str, object]:
    """Fit the shift on one half of a decode and test it on the other.

    THE ONLY THING THAT CAN PROMOTE THE FREQUENCY QUANTITY from a bound to a
    correction. The ellipse is fitted, so its directions must be earned on
    evidence they have not seen; the same applies one level down to a single
    coefficient. A shift fitted on the first half and applied unchanged to the
    second either explains some of it or it does not, and a coefficient that
    explains nothing out of sample was describing that half's noise.

    The static part of the head difference is known to reproduce across the
    halves - r = +0.902 and +0.958 - so the test is well posed: there IS a
    reproducible object to be found. What is at issue is whether the shift is
    part of it.

    Reports both directions, because a single split is one draw and the two
    orders disagreeing is itself informative.
    """
    forward = estimate_frequency_shift(frequency_hz, first_half_difference,
                                       writing_speed_m_s, track_width_m,
                                       **parameters)
    backward = estimate_frequency_shift(frequency_hz, second_half_difference,
                                        writing_speed_m_s, track_width_m,
                                        **parameters)
    if not (forward.get("usable") and backward.get("usable")):
        return {"usable": False, "why": "a half could not be solved"}

    def applied(fitted, other):
        f = _grid(frequency_hz)
        target = np.asarray(other, dtype=np.float64).ravel()
        good = np.isfinite(f) & (f > 0) & np.isfinite(target)
        f, target = f[good], target[good]
        shift = shift_direction(f, writing_speed_m_s, track_width_m,
                                **parameters)
        spacing = (-NEPERS_PER_WAVELENGTH_OF_SEPARATION * f
                   / float(writing_speed_m_s))
        shift = shift - shift.mean()
        spacing = spacing - spacing.mean()
        target = target - target.mean()
        # the spacing is refitted on the held-out half, because it is not the
        # coefficient under test; only the shift is carried across
        carried = fitted["log_shift"] * shift
        left = target - carried
        alone, *_ = np.linalg.lstsq(spacing[:, None], left, rcond=None)
        residual = left - spacing * float(alone[0])
        spacing_only, *_ = np.linalg.lstsq(spacing[:, None], target,
                                           rcond=None)
        baseline = target - spacing * float(spacing_only[0])
        return (1.0 - float(residual @ residual)
                / max(float(baseline @ baseline), 1e-30))

    gains = (applied(forward, second_half_difference),
             applied(backward, first_half_difference))
    separation = abs(forward["log_shift"] - backward["log_shift"])
    combined = float(np.hypot(forward["log_shift_error"],
                              backward["log_shift_error"]))
    agree = bool(separation <= 2.0 * max(combined, 1e-30))
    pooled = 0.5 * (forward["log_shift"] + backward["log_shift"])
    # the mean of two independent estimates: the variances add and the sum is
    # halved twice, so the error is HALF the quadrature sum and not the
    # quadrature sum over root two - which would be too large by root two and
    # would make the promotion test quietly harder to pass than it states
    pooled_error = 0.5 * combined
    standing = abs(pooled) / max(pooled_error, 1e-30)

    # THREE CONDITIONS, and the last of them is why there are three. A
    # held-out gain positive in both directions happens about one time in
    # four when the truth is zero, because two independent noise draws
    # sometimes agree by accident, so a positive gain alone is far too weak a
    # bar. The pooled estimate standing three sigma from zero is the bar
    # every other admission in this arc is held to, and it is imported from
    # `information_extrapolation.SPHERE_SIGMA` rather than restated.
    from vhsdecode.models import information_extrapolation as ie
    clears = bool(min(gains) > 0.0 and agree and standing > ie.SPHERE_SIGMA)
    return {
        "usable": True,
        "first_half": forward,
        "second_half": backward,
        "held_out_gain": tuple(float(g) for g in gains),
        "halves_agree": agree,
        "halves_separation": float(separation),
        "halves_combined_error": combined,
        "pooled_log_shift": float(pooled),
        "pooled_error": float(pooled_error),
        "pooled_sigma": float(standing),
        "clears": clears,
        "why": ("a coefficient earns promotion by explaining evidence it was "
                "not fitted on, by the two halves agreeing within their "
                "errors, and by the pooled estimate standing three sigma "
                "from zero"),
    }


def frequency_verdict(frequency_hz, log_magnitude_difference,
                      writing_speed_m_s: float, track_width_m: float,
                      out_of_sample: Optional[Dict[str, object]] = None,
                      drum_diameter_m: Optional[float] = None,
                      spacing_difference_m: Optional[float] = None,
                      **parameters) -> Dict[str, object]:
    """A BOUND unless the out-of-sample test clears - never a correction.

    The frequency quantity has never been measured, one prior attempt
    explained 0.40 to 0.66 per cent of the varying part, and geometry says the
    true size is four to five orders of magnitude below the noise. So the
    honest output is an upper bound, and this function will not return
    anything else without an out-of-sample result that clears.

    Reports the bound from the data, the bound from geometry, and which of
    the two binds. On the evidence in hand geometry binds by a very wide
    margin, which is the useful finding: the quantity is not merely
    unmeasured, it is unmeasurable at this evidence level, and no amount of
    care with the estimator changes that. More fields would be needed by a
    factor of the square of the ratio.
    """
    fit = estimate_frequency_shift(frequency_hz, log_magnitude_difference,
                                   writing_speed_m_s, track_width_m,
                                   **parameters)
    out: Dict[str, object] = {"fit": fit}
    if not fit.get("usable"):
        return {"promoted": False, "why": fit.get("why", "unusable"), **out}

    from_data = abs(float(fit["three_sigma_bound"]))
    geometry = None
    if drum_diameter_m is not None and spacing_difference_m is not None:
        geometry = geometric_bound(spacing_difference_m, drum_diameter_m)
    cleared = bool((out_of_sample or {}).get("clears", False))
    out.update({
        "promoted": cleared,
        "bound_from_the_data": from_data,
        "bound_from_geometry": (None if geometry is None
                                else geometry["from_the_measured_separation"]),
        "binds": ("geometry" if geometry is not None
                  and geometry["from_the_measured_separation"] < from_data
                  else "the data"),
        "geometry": geometry,
        "out_of_sample": out_of_sample,
        "report_as": "a correction" if cleared else "a bound",
        "why": ("the fitted coefficient is reported as an upper bound until "
                "it explains evidence it was not fitted on; geometry bounds "
                "it far more tightly than the data can"),
    })
    return out


# --------------------------------------------------------------------------
# The set, and how distinguishable it is
# --------------------------------------------------------------------------


def default_parameters(system: str = "NTSC", tape_speed: str = "SP",
                       level_db: Optional[float] = None,
                       carrier_hz: float = 4e6,
                       rf_params: Optional[Dict[str, float]] = None
                       ) -> Dict[str, float]:
    """The three parameters' sizes, derived from the measurements in hand.

    None of these is a tuning constant. The separation is Wallace's law
    inverted on the measured level split; the delay is the root mean square
    of the four measured delays; the scaling is the geometric bound the drum's
    own measured off-axis error gives, which is the larger of the two bounds
    and so the safer size at which to look at the shape.

    The SHAPE of each signature is what enters the key, and it is scale free
    to first order, so these sizes decide only whether the signature is
    numerically alive - not what direction it points in.
    """
    format_mechanics = mechanics(system, tape_speed, rf_params)
    speed = head_model.writing_speed(format_mechanics)
    level = MEASURED_LEVEL_DB[0] if level_db is None else float(level_db)
    separation = spacing_from_level(level, carrier_hz, speed)
    delays = np.asarray(MEASURED_DELAYS_S, dtype=np.float64)
    bound = geometric_bound(separation, format_mechanics["drum_diameter_m"])
    return {
        "spacing_difference_m": separation,
        "delay_difference_s": float(np.sqrt(np.mean(delays ** 2))),
        "fractional_scaling": bound["from_the_drum_off_axis"],
        "writing_speed_m_s": speed,
        "track_width_m": float(format_mechanics["track_width_m"]),
        "drum_diameter_m": float(format_mechanics["drum_diameter_m"]),
    }


def signatures(frequency_hz, system: str = "NTSC", tape_speed: str = "SP",
               sizes: Optional[Dict[str, float]] = None,
               rf_params: Optional[Dict[str, float]] = None,
               **parameters) -> Dict[str, np.ndarray]:
    """The head pair's three quantities on one frequency grid.

    Ready to be merged into `interference.signatures` and fitted with
    `ellipsoid(..., real_parameters=True)`, which is required rather than
    preferred here: two of the three are pure magnitude and one is pure phase,
    and the Hermitian inner product would report the delay and the separation
    as one direction.

    The names carry the stage each acts at, because the key's order is by
    stage and the three do not share one - two are written onto the tape and
    the third is read off it.
    """
    f = _grid(frequency_hz)
    sizes = dict(default_parameters(system, tape_speed,
                                    rf_params=rf_params), **(sizes or {}))
    speed = float(sizes["writing_speed_m_s"])
    width = float(sizes["track_width_m"])
    return {
        "head differential frequency": frequency_signature(
            f, sizes["fractional_scaling"], speed, width, **parameters),
        "head differential amplitude": amplitude_signature(
            f, sizes["spacing_difference_m"], speed),
        "head differential phase": phase_signature(
            f, sizes["delay_difference_s"]),
    }


def _shapes(entries: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Log magnitude and unwrapped phase, each with its level removed.

    The same reduction `interference.distinguishable` uses, so the counts are
    comparable: a gain is a constant in the log domain rather than a scale,
    and a constant is a level and not a shape.
    """
    out: Dict[str, np.ndarray] = {}
    for name, value in entries.items():
        value = np.asarray(value)
        magnitude = np.log(np.maximum(np.abs(value), 1e-12))
        phase = (np.unwrap(np.angle(value)) if np.iscomplexobj(value)
                 else np.zeros_like(magnitude))
        vector = ((magnitude - magnitude.mean())
                  + 1j * (phase - phase.mean()))
        if np.linalg.norm(vector) > 0:
            out[name] = vector
    return out


def _participation(shapes: Dict[str, np.ndarray]) -> Dict[str, object]:
    """The participation ratio of a set of signatures, and its conditioning.

    Real and imaginary parts stacked before the singular values are taken,
    because each entry is the signature of a REAL physical parameter and a
    gain is a different mechanism from a delay of the same shape. Measured
    elsewhere in this arc, that distinction moves a propagation result from
    2.20 to 3.79.
    """
    names = list(shapes)
    if not names:
        return {"effective": 0.0, "count": 0, "names": []}
    stack = np.array([np.concatenate([shapes[name].real, shapes[name].imag])
                      for name in names])
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    alive = (norms > 0).ravel()
    if not np.all(alive):
        # a signature with no shape is not a direction, and normalising it
        # would manufacture one out of rounding - the error `magnetic.py`
        # records, which read 4.65 of 5 before it was found
        names = [name for name, ok in zip(names, alive) if ok]
        stack, norms = stack[alive], norms[alive]
    if not names:
        return {"effective": 0.0, "count": 0, "names": []}
    stack = stack / norms
    values = np.linalg.svd(stack, compute_uv=False)
    share = values ** 2 / max(float((values ** 2).sum()), 1e-30)
    gram = np.abs(stack @ stack.T)
    np.fill_diagonal(gram, 0.0)
    worst = np.unravel_index(int(np.argmax(gram)), gram.shape)
    return {
        "names": names,
        "count": len(names),
        "effective": float(1.0 / np.sum(share ** 2)),
        "singular_values": values,
        "condition": float(values[0] / max(values[-1], 1e-30)),
        "worst_pair": (names[worst[0]], names[worst[1]]),
        "worst_coherence": float(gram[worst]),
    }


def distinguishable(frequency_hz, **kwargs) -> Dict[str, object]:
    """How many of the three this band can tell apart.

    The measure the whole arc uses, asked of the head pair. Measured on the
    VHS luma RF band, 1 to 7 MHz over 512 places: 1.8035 effective of 3 at
    condition 28.84, singular values 1.4134, 1.0000 and 0.0490, and the worst
    coherent pair is amplitude against frequency at 0.99760.

    Set against the tape magnetics' 1.58 of 6, the honest reading is ONE new
    kind and two members of the same collinear family. The delay is the new
    kind: it is a phase where the other two are magnitudes, and their
    coherence measures 8e-17. Amplitude and phase alone therefore read a
    perfect 2.0000 of 2 at condition 1.000. The frequency scaling is not a new
    kind at all - it and the separation together read 1.0024 of 2.

    The separability law explains it. Every mechanism in the shared response
    is a function of one dimensionless group, a length over the recorded
    wavelength; scaling the frequency axis scales that group, which is the
    same thing scaling the length does. A rescaling of the axis acting on a
    family of rates is therefore another rate. Only the shared response's
    curvature separates the two, and fractional bandwidth is the sole variable
    that moves it: 0.99760 over 7:1, 0.99110 over 100:1, 0.99960 over the
    carrier band alone.
    """
    return _participation(_shapes(signatures(frequency_hz, **kwargs)))


def key_distinguishable(frequency_hz, **kwargs) -> Dict[str, object]:
    """The same measure over only the entries that earn a place in the key.

    Measured on the VHS luma RF band: 2.0000 effective of 2 at condition
    1.000, the two being orthogonal to 8e-17. That is the number to compare
    with the tape magnetics' 1.58 of 6, since it is the count over the
    entries actually offered to the key.
    """
    return _participation(_shapes(key_signatures(frequency_hz, **kwargs)))


# --------------------------------------------------------------------------
# THE CONTROLS. Kept as code because a control that cannot fail is not a
# control, and because an earlier component in this arc reported 4.65
# effective of 5 and what exposed it was a control that should have read 1.00
# and read 4.97 instead.
# --------------------------------------------------------------------------


def collinear_control(frequency_hz, system: str = "NTSC",
                      tape_speed: str = "SP",
                      separations_m: Sequence[float] = (),
                      rf_params: Optional[Dict[str, float]] = None
                      ) -> Dict[str, object]:
    """A family of SEPARATIONS must be exactly ONE direction. Expected 1.00.

    The separability law's own statement, run as a control: a family whose
    members differ only in a RATE is collinear. Wallace's law in the log
    domain is `-2 pi dd f / v`, a straight line through the origin whose only
    freedom is its slope, so five different separations are five multiples of
    one vector and the participation ratio must be 1.000 whatever the
    separations are.

    THIS CONTROL CAN FAIL, and it is built to fail the way this arc has
    actually gone wrong. The separations used are the ones the measurement
    gives - a few tens of nanometres - so the log shapes are genuinely small,
    a few hundredths of a neper, which is exactly the regime in which
    normalising a near-zero vector measures rounding instead of shape. That is
    the failure `magnetic.py` records: a construction whose signature had
    collapsed to a constant read 4.97 where 1.00 was required. If the
    normalisation guard here were wrong, or if a level were left in, or if the
    real and imaginary stacking were mismatched, this would read well above
    one.

    Measured: 1.0000 of 5 over separations of 10, 15, 20, 30 and 40 nm, at a
    condition number of 3.5e+15 - which is the singular values collapsing to
    machine precision, exactly as five multiples of one vector should.
    """
    format_mechanics = mechanics(system, tape_speed, rf_params)
    speed = head_model.writing_speed(format_mechanics)
    if not len(separations_m):
        # the range the level split gave: 13.5 to 39.8 nm, with a smaller and
        # a larger value either side of it
        separations_m = (10e-9, 15e-9, 20e-9, 30e-9, 40e-9)
    entries = {f"{value * 1e9:.0f} nm": amplitude_signature(frequency_hz,
                                                            value, speed)
               for value in separations_m}
    result = _participation(_shapes(entries))
    effective = float(result.get("effective", 0.0))
    result.update({
        "expected": 1.0,
        "passes": bool(abs(effective - 1.0) < 0.05),
        "separations_m": tuple(float(v) for v in separations_m),
        "why": ("a family differing only in a rate is one direction; anything "
                "else means a level was left in, or a near-zero vector was "
                "normalised, or the stacking is wrong"),
    })
    return result


def azimuth_cancels_control(frequency_hz, misalignment_degrees: float = 0.25,
                            system: str = "NTSC", tape_speed: str = "SP",
                            rf_params: Optional[Dict[str, float]] = None
                            ) -> Dict[str, object]:
    """A single azimuth misalignment must cancel EXACTLY. Expected zero.

    The second control, and it tests the reasoning that kept azimuth out of
    the set rather than the arithmetic that built it.

    VHS writes adjacent tracks at opposite azimuth, so a drum rotated by
    `delta` reaches one head as `+delta` and the other as `-delta` relative to
    the track each reads. The azimuth loss is `sinc(W tan(dtheta) / lambda)`,
    `tan` is odd and `sinc` is even, so the composition is EVEN and the two
    heads' magnitude responses are identical. The head difference is
    therefore exactly zero, to machine precision, and azimuth reaches the head
    difference only through the two heads' independent construction errors.

    THIS CONTROL CAN FAIL. It fails if the azimuth error is entered with one
    sign for both heads, which is the natural mistake and would make a
    systematic misalignment look like a head difference; it fails if the loss
    is written with `sin` rather than `sinc`, which is not even; and it fails
    if the track width is taken per head rather than as the format's, since
    the width and the tangent enter only as a product. It is checked against a
    NON-zero case in the same breath - two heads with independent construction
    errors DO differ - so it cannot pass by returning zero for everything.

    Measured: the common misalignment cancels to 0.0e+00 nepers rms - exactly,
    since evenness makes the two terms bit-for-bit identical - while
    independent errors of 0.25 and 0.15 degrees leave 4.9e-02 nepers rms.
    """
    format_mechanics = mechanics(system, tape_speed, rf_params)
    speed = head_model.writing_speed(format_mechanics)
    width = float(format_mechanics["track_width_m"])
    f = _grid(frequency_hz)
    delta = float(misalignment_degrees)

    def response(error_degrees):
        values = dict(head_model.TYPICAL)
        values["azimuth_error_degrees"] = float(error_degrees)
        return head_model.log_response(f, speed, width, **values)

    common = response(+delta) - response(-delta)
    # and the case that must NOT be zero: two heads built differently
    independent = response(+delta) - response(0.6 * delta)
    common_rms = float(np.sqrt(np.nanmean(np.asarray(common) ** 2)))
    independent_rms = float(np.sqrt(np.nanmean(np.asarray(independent) ** 2)))
    return {
        "common_misalignment_rms_nepers": common_rms,
        "independent_errors_rms_nepers": independent_rms,
        "expected": 0.0,
        "passes": bool(common_rms < 1e-12 and independent_rms > 1e-4),
        "misalignment_degrees": delta,
        "why": ("azimuth loss is even in the azimuth error, so opposite "
                "track azimuths give identical magnitudes and a systematic "
                "misalignment cancels; only independent construction errors "
                "survive the difference"),
    }


def wallace_round_trip_control(level_db: float = MEASURED_LEVEL_DB[0],
                               carrier_hz: float = 4e6,
                               system: str = "NTSC", tape_speed: str = "SP",
                               rf_params: Optional[Dict[str, float]] = None
                               ) -> Dict[str, object]:
    """Invert Wallace's law and put the answer back. Expected the input.

    The third control, and the one that checks the constant nobody looks at.
    `spacing_from_level` divides by 54.59 decibels per wavelength and
    `amplitude_signature` multiplies by `2 pi` nepers, and those are the same
    number only if the neper-to-decibel conversion is right. A factor of two -
    the difference between a field ratio and a power ratio, which is the
    classic way to get this wrong - would show here and nowhere else, because
    every other use in this module is of a shape rather than of a size.

    The independently known answer is the input level itself, and the
    published cross-check is section 4.2g's own arithmetic: -0.51 to -1.50 dB
    gave 13.5 to 39.8 nm on a 1.45 um wavelength, which this function
    reproduces to the digit.

    Measured: -0.5558 dB gives 14.767 nm and returns -0.5558 dB, a round-trip
    error of 1.1e-16 dB, on 54.575 dB per wavelength against the 54.6 the
    design document states. And the published range reproduces to the digit:
    -0.51 and -1.50 dB give 13.55 and 39.85 nm against the document's 13.5
    and 39.8 nm.
    """
    format_mechanics = mechanics(system, tape_speed, rf_params)
    speed = head_model.writing_speed(format_mechanics)
    separation = spacing_from_level(level_db, carrier_hz, speed)
    back = amplitude_signature(np.array([float(carrier_hz)]), separation,
                               speed)
    returned = float(head_model.DB_PER_NEPER
                     * np.log(np.abs(complex(back[0]))))
    # section 4.2g's own published range, recomputed rather than quoted: the
    # -0.51 to -1.50 dB level split read at the 4 MHz carrier
    published = tuple(spacing_from_level(level, 4e6, speed)
                      for level in (-0.51, -1.50))
    error = abs(returned - (-abs(float(level_db))))
    return {
        "level_db": float(level_db),
        "separation_m": float(separation),
        "returned_db": returned,
        "round_trip_error_db": float(error),
        "db_per_wavelength": float(DB_PER_WAVELENGTH_OF_SEPARATION),
        "published_range_m": published,
        "passes": bool(error < 1e-9
                       and abs(DB_PER_WAVELENGTH_OF_SEPARATION - 54.6) < 0.05),
        "why": ("54.59 dB per wavelength is 2 pi nepers converted, and the "
                "round trip is the only place in this module where the size "
                "rather than the shape of the law is exercised"),
    }


# --------------------------------------------------------------------------
# Where each of the three belongs in the chain
# --------------------------------------------------------------------------

# The positions this module asks the coordinator to declare in
# `interference.COMPONENT_ORDER`. Position is the stage the modification
# happens AT, counting from the source, in that table's own numbering: the
# transmission path is 1 to 4, the recording machine 10 to 11, the tape and
# the head 20 to 23, and the capture 30.
#
# THE FREQUENCY SCALING IS FIRST OF THE THREE, at 12. It is a property of the
# writing geometry: it sets the wavelength that every length-dependent loss
# downstream then sees. It cannot commute with them, because rescaling the
# axis and then applying a frequency-dependent loss is not the same as
# applying the loss and then rescaling, so its position is a real statement
# and not bookkeeping.
#
# THE SEPARATION IS SECOND, at 13. It acts on the wavelength the scaling has
# just fixed, at the record head, before the medium's own losses at 20. Its
# side is settled by measurement rather than assumed: two tapes on one deck
# give head-difference vectors correlating at r = +0.11, so the difference
# travels with the tape and not with the deck.
#
# THE DELAY IS ON THE PLAYBACK SIDE, at 24 - after the tape and its losses,
# before the capture. That placement is the conservative one and the evidence
# is genuinely divided: the same tape at SP and at EP gives -1.0 and -7.1 ns,
# and an amplifier's path length does not change with tape speed, which
# argues for the record side instead. It matters less than the other two,
# and for a reason worth stating: a pure delay is diagonal in frequency and
# commutes with every other entry in the key, so no inverse is misapplied by
# placing it on the wrong side. The declaration is still required.
#
# WHICH OF THE THREE SHOULD ACTUALLY BE REGISTERED. Two of them: the
# amplitude and the phase, which together read 2.0000 of 2 at condition
# 1.000. The frequency scaling is declared here and given its position -
# because an entry with no declared position cannot be inverted, and because
# a later measurement on a wider band or a longer capture may promote it -
# but it should not go into the key as it stands. It adds no direction and it
# costs the conditioning: 2.0000 of 2 at condition 1.000 becomes 1.8035 of 3
# at condition 28.84. `KEY_ENTRIES` names the two that earn their place, and
# `frequency_verdict` is what would change that.
#
# WHAT MERGING THE TWO INTO `interference.signatures` COSTS, measured on the
# VHS luma RF band before the coordinator does it:
#
#     interference alone            8.4249 of 14, condition    7.41
#     with the amplitude only       8.0057 of 15, condition   92.79
#     with the phase only           8.1911 of 15, condition   39.37
#     with both                     7.9234 of 16, condition   93.60
#     with all three                7.4176 of 17, condition  815.95
#
# THE FIRST COLUMN IS NOT STABLE and the coordinator should re-measure it
# rather than quote it. The modelled set is being changed in a parallel lane
# while this is written - the same table read 7.8864 of 14 at condition 7.76
# an hour earlier, before the record-level entry was orthogonalised - so the
# absolute counts move with that lane's work. What does not move is the pair
# coherences below, which involve only this module's entries and two
# unchanged ones.
#
# So the count falls, and the reason is a shape coincidence rather than a
# physical one, which is why it is reported here rather than acted on. The
# head delay is 0.9817 coherent with the modelled `group delay`, because a
# phase linear in frequency and one quadratic in it are nearly the same
# vector over a 7:1 band once the level is removed. The head separation is
# 0.9722 coherent with `particle noise`, for the same reason one step along:
# both are monotone in the same dimensionless group.
#
# THOSE PAIRS ARE SEPARATED BY THEIR STAGE AND NOT BY THEIR SHAPE, and the
# participation ratio is a shape measure that knows nothing about stage. The
# transmission group delay acts at position 2, before the recorder saw the
# signal; the head delay acts at 24, inside the playback deck. They are
# distinguished by head parity - a tape defect alternates with the head and a
# transmission defect does not - which is the first discriminator
# `COMPONENT_MAPPINGS.md` section 6 lists. The same holds for the separation
# against the particle noise, which is not a correctable term at all but the
# floor the others are judged against. So this is a case where the span
# measure and the physics disagree, and the physics is right; the coordinator
# should register both entries and carry these numbers rather than let the
# count decide.
COMPONENT_POSITIONS: Dict[str, int] = {
    "head differential frequency": 12,
    "head differential amplitude": 13,
    "head differential phase": 24,
}

KEY_ENTRIES = ("head differential amplitude", "head differential phase")
BOUND_ONLY = ("head differential frequency",)


def key_signatures(frequency_hz, **kwargs) -> Dict[str, np.ndarray]:
    """The entries that earn a place in the key: the amplitude and the phase.

    `signatures` returns all three, because the third is a real physical
    quantity whose size this module bounds. This returns the two that add a
    direction, which is what a key should be given: measured, 2.0000 effective
    of 2 at condition 1.000, against 1.8035 of 3 at condition 28.84 with the
    third included.
    """
    every = signatures(frequency_hz, **kwargs)
    return {name: every[name] for name in KEY_ENTRIES if name in every}


# --------------------------------------------------------------------------
# The two on-tape bands as two measurements of the same head pair
# --------------------------------------------------------------------------

COLOUR_UNDER_BAND_HZ = (0.4e6, 0.9e6)
LUMA_BAND_HZ = (3.4e6, 4.4e6)
"""The two bands the same head wrote in the same pass.

The colour-under sits where the luma FM does not, and both were sourced
from one signal by one head - so they are two measurements of that head at
wavelengths eleven times apart, taken under identical everything else.
"""


def band_ratio_discriminator(luma_hz: float = 3.9e6,
                             colour_under_hz: float = 0.65e6
                             ) -> Dict[str, float]:
    """GAIN OR CLEARANCE - decided by one ratio, with no fitting.

    Ethan: "We can use the difference between the color under and luma
    components like we did before to measure the properties of the head.
    The luma and chroma components are band measurements of each head."

    The head difference in log magnitude is

        dlog|H|(f) = g - 2 pi (dd) f / v

    a CONSTANT if the heads differ in gain and LINEAR IN f if they differ
    in clearance. So the ratio of the departure measured in the two bands
    is the whole test:

        a pure clearance difference  ->  ratio = f_luma / f_cu = 6.000
        a pure gain difference       ->  ratio = 1.000

    Nothing is fitted and nothing is assumed; the two mechanisms simply
    predict different numbers. This is the reasoning that established the
    head difference IS a flat gain rather than a clearance - the departure
    was still 0.15 to 0.22 nepers at 629 kHz where a clearance would have
    decayed to about 0.01 - and it is written here as an estimator rather
    than left as an argument.
    """
    ratio = float(luma_hz) / max(float(colour_under_hz), 1e-30)
    return {
        "clearance_predicts": ratio,
        "gain_predicts": 1.0,
        "luma_hz": float(luma_hz),
        "colour_under_hz": float(colour_under_hz),
        "how": "measure the head difference in each band and divide; the "
               "answer lands on one prediction or between them",
    }


def two_band_difference(luma_frequency_hz, luma_log_difference,
                        colour_under_frequency_hz, colour_under_log_difference,
                        writing_speed_m_s: float = 5.8709
                        ) -> Dict[str, object]:
    """Solve the head difference for a GAIN and a CLEARANCE together.

    Two bands, two unknowns - the construction Ethan describes, applied to
    the head pair rather than to the absolute response. The model is
    linear in both, so this is one least-squares solve and the error bars
    come out of it directly.

    WHY THE PAIR IS WORTH IT, measured over the two bands against either
    alone:

        luma band alone     cond 27.0   coherence 0.997
                            sigma(clearance) 186 nm   sigma(gain) 0.78 dB
        colour-under alone  cond  9.1   coherence 0.976
                            sigma(clearance) 373 nm   sigma(gain) 0.27 dB
        BOTH BANDS          cond  3.1   coherence 0.811
                            sigma(clearance)  23 nm   sigma(gain) 0.070 dB

    ELEVEN TIMES the precision on the gain and eight on the clearance,
    because over one band a constant and a term linear in frequency are
    nearly the same shape and over eleven wavelengths they are not.

    EVERYTHING ELSE STILL CANCELS. This is a difference between the two
    heads, so the tape, the record-side path, the decoder's filters and
    the de-emphasis are common to both and gone - which is what makes the
    residual belong to the heads alone. The two bands do not weaken that;
    they add a second lever to the same cancellation.
    """
    luma_f = np.asarray(luma_frequency_hz, dtype=np.float64).ravel()
    luma_y = np.asarray(luma_log_difference, dtype=np.float64).ravel()
    cu_f = np.asarray(colour_under_frequency_hz, dtype=np.float64).ravel()
    cu_y = np.asarray(colour_under_log_difference, dtype=np.float64).ravel()
    if len(luma_f) != len(luma_y) or len(cu_f) != len(cu_y):
        raise ValueError("each band needs one departure per frequency")
    grid = np.concatenate([cu_f, luma_f])
    values = np.concatenate([cu_y, luma_y])
    good = np.isfinite(grid) & np.isfinite(values)
    grid, values = grid[good], values[good]
    if len(grid) < 4:
        return {"gain_nepers": float("nan"), "clearance_m": float("nan"),
                "usable": False}
    design = np.column_stack([np.ones_like(grid),
                              -2.0 * np.pi * grid / float(writing_speed_m_s)])
    solution, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ solution
    dof = max(len(grid) - 2, 1)
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.inv(design.T @ design)
    gain, clearance = float(solution[0]), float(solution[1])
    # A SIGNIFICANCE TEST NEEDS A MEANINGFUL ERROR BAR. On noiseless data
    # the residual is zero, so sigma is zero, and "three sigma" fires on
    # floating-point dust - a planted pure gain reported a significant
    # clearance of 1e-17 metres. The floor is the departure's own scale
    # times the double's epsilon, below which nothing is a measurement.
    scale = float(np.max(np.abs(values))) or 1.0
    floor = scale * np.finfo(np.float64).eps * 16.0
    # which mechanism dominates, by the share of the departure each explains
    gain_only = np.full_like(grid, gain)
    clearance_only = -2.0 * np.pi * grid / float(writing_speed_m_s) * clearance
    total = float(np.sum(values ** 2)) or 1.0
    return {
        "gain_nepers": gain,
        "gain_db": gain * 20.0 / np.log(10.0),
        "gain_sigma_nepers": float(np.sqrt(covariance[0, 0])),
        "clearance_m": clearance,
        "clearance_sigma_m": float(np.sqrt(covariance[1, 1])),
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "gain_share": float(np.sum(gain_only ** 2) / total),
        "clearance_share": float(np.sum(clearance_only ** 2) / total),
        "gain_significant": bool(
            abs(gain) > max(3.0 * float(np.sqrt(covariance[0, 0])), floor)),
        "clearance_significant": bool(
            abs(clearance)
            > max(3.0 * float(np.sqrt(covariance[1, 1])), floor)),
        "significance_floor": floor,
        "usable": True,
        "cancels": "the tape, the record-side path, the decoder's filters "
                   "and the de-emphasis - all common to both heads",
    }


# --------------------------------------------------------------------------
# Is each head's response FIXED? The test, and what it can and cannot see
# --------------------------------------------------------------------------

def fixed_response_verdict(head_a_halves, head_b_halves,
                           span_s: Optional[float] = None
                           ) -> Dict[str, object]:
    """Whether each head's response is fixed, and over what span.

    Ethan: "The model of the VCR is fixed throughout recording and playback
    depending on which circuit path is enabled on the VCR so a fixed
    response for each head confirms this though testing."

    THE TEST, made explicit. If the model is fixed and the only switch is
    which head reads, then the BETWEEN-head difference must exceed the
    WITHIN-head variation - otherwise "each head has its own fixed
    response" is a distinction the data does not support. Each argument is
    a pair of profiles from disjoint halves of that head's own fields, so
    their difference IS the within-head variation with no model in it.

    THE PULSE MUST COME OUT FIRST, and this is why the function takes
    profiles rather than a correlation. Two sync profiles correlate at
    +1.0000 whatever the channel does, because both are dominated by the
    same -40 IRE step; the correlation measures the pulse, not the
    response. So the comparison is made on the LEVEL and the SHAPE
    separately, after the common offset is removed.

    MEASURED on 16 fields of two tapes, sync interval, in IRE:

                    within-head             between heads
        home   level 0.023  shape 0.069   level -0.147 (6.3x)  shape 0.315 (4.6x)
        bars   level 0.006  shape 0.105   level +0.048 (7.8x)  shape 0.246 (2.4x)

    So the per-head LEVEL is resolved on both tapes and the SHAPE on one.
    And the level's SIGN differs between them, which is the head model's
    own statement arriving from a third direction: the per-head constant
    belongs to the (recording, playback) PAIR, not to a machine.

    WHAT THIS TEST CANNOT SEE, and it is the important limit. Sixteen
    fields is 0.27 seconds. Over that span the response is fixed; it says
    nothing about "throughout", and the tesseract's field-series fold -
    256 and 1024 fields, 4.3 and 17 seconds - measures the response
    DRIFTING across the transport's whole band, flat at 40 to 48 times the
    noise from a quarter second upward.

    THE TWO ARE NOT IN CONFLICT, and the reconciliation is the useful
    result: THE TIME SCALE SEPARATES THE CIRCUIT FROM THE MECHANICS. The
    circuit does not change - Ethan's claim, and nothing in the
    electronics has a second-scale time constant - so the fixed part is
    the circuit's. What varies does so at the drum and reel rates, which
    are mechanical, and belongs to the head-to-tape interface rather than
    to the model. A response that were fixed at ALL scales would actually
    refute the transport model, not confirm it.
    """
    def split(first, second):
        first = np.asarray(first, dtype=np.float64).ravel()
        second = np.asarray(second, dtype=np.float64).ravel()
        if len(first) != len(second):
            raise ValueError("both halves must be the same profile length")
        level = float(np.mean(first - second))
        shape = (first - second) - level
        return level, float(np.sqrt(np.mean(shape ** 2)))

    # each half averages half the fields, so a whole-head figure is
    # root-two better than the difference between halves
    root_two = float(np.sqrt(2.0))
    within = {}
    wholes = {}
    for name, halves in (("A", head_a_halves), ("B", head_b_halves)):
        first, second = halves
        level, shape = split(first, second)
        within[name] = {"level": abs(level) / root_two,
                        "shape": shape / root_two}
        wholes[name] = 0.5 * (np.asarray(first, dtype=np.float64).ravel()
                              + np.asarray(second, dtype=np.float64).ravel())

    level, shape = split(wholes["A"], wholes["B"])
    typical_level = 0.5 * (within["A"]["level"] + within["B"]["level"])
    typical_shape = 0.5 * (within["A"]["shape"] + within["B"]["shape"])
    level_ratio = abs(level) / max(typical_level, 1e-30)
    shape_ratio = shape / max(typical_shape, 1e-30)
    return {
        "within_head": within,
        "between_level": level,
        "between_shape_rms": shape,
        "level_ratio": level_ratio,
        "shape_ratio": shape_ratio,
        "level_resolved": bool(level_ratio > 3.0),
        "shape_resolved": bool(shape_ratio > 3.0),
        "fixed_per_head": bool(level_ratio > 3.0 or shape_ratio > 3.0),
        "span_s": span_s,
        "span_covers_drift": bool(span_s is not None and span_s > 1.0),
        "limit": "a short span tests reproducibility, not stability - the "
                 "transport's drift lives from a quarter second upward, so "
                 "a verdict from under a second speaks for the CIRCUIT "
                 "only, which is exactly the part that should be fixed",
    }
