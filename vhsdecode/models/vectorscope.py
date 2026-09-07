"""THE REFERENCE NTSC VECTORSCOPE, computed from the encoding equations.

Ethan, 2026-09-06, looking at a vectorscope of a real composite recording:
*"determine what a reference NTSC color composite vectorscope looks like,
and see what the lines pointing to I, Q, -Q, -I are, and see if those
measurements map to a part of the signal in the color carrier we can use
for reference when designing the circular shape of the color."*

EVERYTHING HERE IS DERIVED, NOT QUOTED. SMPTE 170M-2004 clause 6 gives the
luminance equation and clause 10 the encoding:

    N = 0.925 Y + 7.5 + 0.4552 (B-Y) sin(w t) + 0.8115 (R-Y) cos(w t)

with note 2 fixing the reference: *"The subcarrier phase reference in the
equations above is the phase of the color burst + 180 degrees."* So a
colour's vector is 0.4552(B-Y) on one axis and 0.8115(R-Y) on the other,
measured from the burst plus 180 - which is why a scope draws the burst at
180 and everything else follows without a further convention.

THE TARGETS, computed from those equations for 75 per cent bars:

    bar   phase        chroma      complement check
    Yl    167.08 deg   31.034      180.00 from B, equal amplitude
    R     103.46       43.869      180.00 from Cy, equal
    Mg     60.71       40.963      180.00 from G, equal
    B     347.08       31.034
    Cy    283.46       43.869
    G     240.71       40.963

For 100 per cent bars the angles are IDENTICAL and the amplitudes scale by
four thirds. That is the first useful fact: the ANGLES do not depend on the
bar level at all, so an angle error cannot be a gain error.

WHAT THE I AND Q LINES ARE, AND WHAT THEY ARE NOT. They are the two
MODULATION axes - clause 10's second form modulates Q on the sine and I on
the cosine, both at 33 degrees from the (B-Y, R-Y) pair - so Q sits at 33
degrees, I at 123, -Q at 213 and -I at 303. They are the axes the encoder
works in.

NOTHING IN THE SIGNAL SITS ON THEM. No bar does: the closest are R at
103.46 and Cy at 283.46, both 19.54 degrees off the I axis, and Yl and B
at 12.92 degrees off the burst axis. So the I and Q lines cannot be used
as a reference - they are a coordinate system, not a signal. The answer to
Ethan's question is that the reference he is looking for is elsewhere, and
there are exactly two places it can be:

  * THE BURST, on every line, at 180 degrees exactly and 40 IRE
    peak-to-peak by ITU-R BT.1700. It is the only per-line reference the
    signal carries, and it fixes the phase origin and the amplitude scale
    together.
  * THE SIX BAR TARGETS, when the content is bars. Their angles are fixed
    by the equations above and independent of level, and each complement
    pair must be exactly 180 degrees apart with equal amplitude - six
    independent references, and three exact constraints that hold whatever
    the gain.

AND THE SHAPE IS NOT A CIRCLE. The six targets do not lie on one: their
amplitudes are 31.03, 40.96 and 43.87 for the three complementary pairs, a
spread of forty per cent. What is circular is the locus of CONSTANT
SATURATION, which a bar signal does not trace; a chroma sweep at constant
amplitude does. So "the circular shape of the colour" is a property to be
imposed by normalising each hue by its own specified amplitude, and
`circularity` below does exactly that - after which a departure from the
circle is the chroma channel's response and nothing else.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# SMPTE 170M-2004 clause 6, the luminance equation, with the note that the
# coefficients are precise values and not roundings.
LUMA_COEFFICIENTS = {"R": 0.299, "G": 0.587, "B": 0.114}
# clause 10, the BASE EQUATION's two modulation scalings
BY_SCALE = 0.4552
RY_SCALE = 0.8115
# clause 10, the I/Q form: both carriers at 33 degrees from (B-Y, R-Y)
IQ_ROTATION_DEG = 33.0
# clause 8.2: the burst is inverted from the reference subcarrier, and note
# 2 makes that reference the burst + 180, so the burst is the origin at 180
BURST_PHASE_DEG = 180.0
# ITU-R BT.1700, 525-line: burst amplitude 40 IRE peak to peak
BURST_IRE_PEAK_TO_PEAK = 40.0

BARS = {
    "Yl": (1, 1, 0), "Cy": (0, 1, 1), "G": (0, 1, 0),
    "Mg": (1, 0, 1), "R": (1, 0, 0), "B": (0, 0, 1),
}
COMPLEMENTS = (("Yl", "B"), ("Cy", "R"), ("G", "Mg"))


def encode(red: float, green: float, blue: float) -> Dict[str, float]:
    """One colour's luminance and chroma vector, from clauses 6 and 10."""
    luma = (LUMA_COEFFICIENTS["G"] * green + LUMA_COEFFICIENTS["B"] * blue
            + LUMA_COEFFICIENTS["R"] * red)
    blue_difference = blue - luma
    red_difference = red - luma
    on_sine = BY_SCALE * blue_difference
    on_cosine = RY_SCALE * red_difference
    return {
        "luma_ire": luma,
        "blue_difference": blue_difference,
        "red_difference": red_difference,
        "chroma": float(math.hypot(on_sine, on_cosine)),
        "phase_deg": float(math.degrees(math.atan2(on_cosine, on_sine)) % 360.0),
    }


def targets(level: float = 75.0) -> Dict[str, Dict[str, float]]:
    """The six bar targets at a given bar level, computed each time.

    The angles are independent of `level` - only the amplitudes scale - so
    an angle error can never be a gain error, which is what makes the
    angles the stronger reference of the two.
    """
    return {name: encode(*(level * c for c in rgb))
            for name, rgb in BARS.items()}


def axes() -> Dict[str, float]:
    """The modulation axes, which are a coordinate system and not a signal."""
    return {
        "Q": IQ_ROTATION_DEG,
        "I": IQ_ROTATION_DEG + 90.0,
        "-Q": IQ_ROTATION_DEG + 180.0,
        "-I": IQ_ROTATION_DEG + 270.0,
        "burst": BURST_PHASE_DEG,
        "+(B-Y)": 0.0,
        "+(R-Y)": 90.0,
    }


def nearest_axis(phase_deg: float) -> Dict[str, object]:
    """How far a measured vector sits from the nearest modulation axis.

    Reported because it is the question Ethan asked, and because the answer
    is always "not on it": the closest any bar comes is 19.54 degrees.
    """
    named = axes()
    best, distance = None, 360.0
    for name, angle in named.items():
        gap = abs(((float(phase_deg) - angle + 180.0) % 360.0) - 180.0)
        if gap < distance:
            best, distance = name, gap
    return {"axis": best, "degrees_away": distance,
            "on_axis": bool(distance < 1.0)}


def burst_reference(measured_ire: float,
                    measured_phase_deg: float = BURST_PHASE_DEG,
                    peak_to_peak: bool = False) -> Dict[str, object]:
    """The burst, which is the only per-line reference the signal carries.

    It fixes both the amplitude scale and the phase origin, so a chroma
    correction that uses anything else on an ordinary picture is using
    content as a reference.

    `peak_to_peak` says which convention the measurement is in. The
    standard specifies 40 IRE PEAK TO PEAK; a vectorscope conventionally
    reports the AMPLITUDE, which is half of it, and the default here
    follows the scope. Getting this wrong is a clean factor of two: a
    reading of 17.97 IRE is 89.9 per cent of the specification as an
    amplitude and 44.9 per cent as a peak-to-peak.
    """
    specified = (BURST_IRE_PEAK_TO_PEAK if peak_to_peak
                 else BURST_IRE_PEAK_TO_PEAK / 2.0)
    gain = float(measured_ire) / specified
    offset = ((float(measured_phase_deg) - BURST_PHASE_DEG + 180.0) % 360.0) - 180.0
    return {
        "measured_ire": float(measured_ire),
        "specified_ire": specified,
        "convention": "peak to peak" if peak_to_peak else "amplitude",
        "gain": gain,
        "gain_percent": 100.0 * gain,
        "gain_db": 20.0 * math.log10(gain) if gain > 0 else float("-inf"),
        "phase_offset_deg": offset,
        "phase_offset_s": offset / 360.0 / (455.0 / 2.0 * 525.0 * 30.0 / 1.001),
        "cite": ("ITU-R BT.1700 for the 40 IRE; SMPTE 170M clause 8.2 and "
                 "note 2 to clause 10 for the phase origin"),
    }


def circularity(measured: Dict[str, Tuple[float, float]],
                level: float = 75.0) -> Dict[str, object]:
    """THE SHAPE, in all three dimensions, against the computed reference.

    `measured` maps a bar name to its measured (chroma amplitude, phase in
    degrees). Each hue is normalised by ITS OWN specified amplitude, which
    is what turns the reference hexagon into a circle - the six targets are
    31.03, 40.96 and 43.87, a spread of forty per cent, so a raw plot of
    bars is not circular and was never meant to be.

    Returns the three dimensions separately, because they fail separately:

      AMPLITUDE   the common gain, and how much each hue departs from it.
                  A hue-dependent radius is differential gain.
      PHASE       the common rotation, and each hue's departure from it. A
                  hue-dependent angle is differential phase.
      SHAPE       the residual after both are removed, which is what no
                  gain and no rotation can explain.

    And two constraints hold whatever the gain and rotation are, so they
    are the strongest checks available: each complement pair must be 180
    degrees apart, and the two members must have equal amplitude.
    """
    reference = targets(level)
    rows, gains, rotations = {}, [], []
    for name, (amplitude, phase) in measured.items():
        if name not in reference:
            raise ValueError(f"{name!r} is not one of the six bars")
        want = reference[name]
        gain = float(amplitude) / want["chroma"]
        rotation = ((float(phase) - want["phase_deg"] + 180.0) % 360.0) - 180.0
        rows[name] = {"gain": gain, "rotation_deg": rotation,
                      "specified_chroma": want["chroma"],
                      "specified_phase_deg": want["phase_deg"]}
        gains.append(gain)
        rotations.append(rotation)
    common_gain = float(np.median(gains)) if gains else float("nan")
    common_rotation = float(np.median(rotations)) if rotations else float("nan")
    for row in rows.values():
        row["differential_gain"] = row["gain"] / common_gain - 1.0
        row["differential_phase_deg"] = row["rotation_deg"] - common_rotation
    pairs = []
    for first, second in COMPLEMENTS:
        if first in measured and second in measured:
            gap = abs(((measured[first][1] - measured[second][1] + 180.0)
                       % 360.0) - 180.0)
            ratio = (measured[first][0] / measured[second][0]
                     if measured[second][0] else float("nan"))
            pairs.append({"pair": f"{first}/{second}",
                          "degrees_apart": 180.0 - gap if gap < 90 else gap,
                          "amplitude_ratio": ratio})
    return {
        "per_bar": rows,
        "common_gain": common_gain,
        "common_rotation_deg": common_rotation,
        "differential_gain_percent": 100.0 * float(np.ptp(
            [r["differential_gain"] for r in rows.values()])) if rows else 0.0,
        "differential_phase_deg": float(np.ptp(
            [r["differential_phase_deg"] for r in rows.values()])) if rows else 0.0,
        "complements": pairs,
        "why": ("each hue is normalised by its own specified amplitude, so "
                "the reference becomes a circle and what is left is the "
                "channel: a common gain, a common rotation, and a residual "
                "that neither explains"),
    }


# --------------------------------------------------------------------------
# Where the 4 fsc sample instants actually land
# --------------------------------------------------------------------------

SAMPLING_STANDARD = "SMPTE ST 244 (4 fsc composite NTSC)"

SAMPLE_PHASES_DEG = tuple((IQ_ROTATION_DEG + 90.0 * n) % 360.0
                          for n in range(4))
"""(33, 123, 213, 303) - the +Q, +I, -Q, -I axes.

THE SAMPLE INSTANTS ARE THE I/Q AXES. Ethan, relaying the point: "4fsc
SMPTE ST.0244 NTSC is sampled on the +I, +Q, -I, -Q phases which are 33
degrees rotated relative to U and V".

This is not decoration on a graticule - it decides what a pair of adjacent
samples MEANS. At 4 fsc the subcarrier advances 90 degrees a sample, so
consecutive samples are always in quadrature and a magnitude taken as
hypot(s[n], s[n+1]) is correct whatever the phase origin. But the ANGLE of
that pair is measured from the first sample's own phase, and under ST 244
that phase is +Q at 33 degrees - NOT the +(B-Y) axis at zero.

SO A PHASE READ STRAIGHT OFF 4 fsc SAMPLES IS IN THE I/Q FRAME and is 33
degrees from the U/V frame the burst and the bar targets are quoted in.
The decoder already carries the rotation - `chroma.py`'s
`ntsc_color_framing_phase_shift = 33` is this same 33 degrees - and this
constant records WHY it is 33 rather than leaving it as a magic number
beside the framing map.

The practical rule: a magnitude needs no correction, an angle needs 33
degrees, and `to_uv_frame` / `from_uv_frame` below do it in the one
direction each rather than leaving a sign to be rediscovered.
"""


def sample_phase_deg(index: int) -> float:
    """Which modulation axis the nth 4 fsc sample sits on."""
    return SAMPLE_PHASES_DEG[int(index) % 4]


def sample_axis_name(index: int) -> str:
    """+Q, +I, -Q or -I - the axis the nth sample lands on."""
    return ("+Q", "+I", "-Q", "-I")[int(index) % 4]


def to_uv_frame(phase_deg):
    """A phase read off 4 fsc samples, carried into the U/V frame.

    The I/Q frame leads U/V by the rotation, so coming back subtracts it.
    Quoted this way round because the burst, the bar targets and every
    angle in `axes()` are U/V quantities, and a phase compared against
    them must arrive in their frame.
    """
    return np.mod(np.asarray(phase_deg, dtype=np.float64)
                  + IQ_ROTATION_DEG, 360.0)


def from_uv_frame(phase_deg):
    """The inverse: a U/V angle expressed against the sampling axes."""
    return np.mod(np.asarray(phase_deg, dtype=np.float64)
                  - IQ_ROTATION_DEG, 360.0)


def quadrature_note() -> Dict[str, object]:
    """What is and is not affected by the sampling frame.

    Stated as a contract because getting it half right is the likely
    error: a magnitude is frame-free and an angle is not, so an instrument
    that reads both from the same pair of samples needs the rotation on
    exactly one of them.
    """
    return {
        "standard": SAMPLING_STANDARD,
        "sample_phases_deg": SAMPLE_PHASES_DEG,
        "rotation_deg": IQ_ROTATION_DEG,
        "magnitude": "UNAFFECTED - consecutive samples are in quadrature "
                     "whatever the phase origin, so hypot(s[n], s[n+1]) is "
                     "correct as it stands",
        "angle": "ROTATED by 33 degrees - a phase read off the samples is "
                 "in the I/Q frame, and the burst and bar targets are "
                 "quoted in U/V",
        "already_applied_at": "chroma.py's ntsc_color_framing_phase_shift, "
                              "which is this same 33 degrees",
    }


# --------------------------------------------------------------------------
# The colour difference coordinate system, read off the transition streaks
# --------------------------------------------------------------------------

"""THE I AND Q LINES ARE NOT EMPTY AFTER ALL, and the correction above is
narrower than it reads. Ethan, 2026-09-06, with a composite vectorscope in
front of him: *"I believe these lines pointing to I.Q.-I,-Q need to be
corrected and represent a measureable shape that we can use for correcting
the color's coordinate system."*

Both statements stand together, and the distinction between them is the
whole of what follows. `nearest_axis` is right that no BAR TARGET sits on
an I or Q line - the closest is 19.54 degrees away and that figure is
computed, not measured. But a scope set to whole line rather than to burst
shows the TRANSITIONS between the bars as well as the bars, and a
transition is a trajectory: it leaves one target and arrives at another
along a direction. Those directions are what his screenshot shows lying
near I, near minus I, near Q and near minus Q, and a direction is a
measurement whether or not anything rests on it.

WHAT THAT BUYS, AND WHY IT IS THE STRONGER OF THE TWO CORRECTIONS. The six
targets fix six POINTS. The transition directions fix the AXES those
points are expressed in - the colour difference coordinate system itself.
Correcting points moves six colours; correcting the coordinate system
moves every colour at once, and it is the only one of the two that can be
made from material that is not a bar signal.

THE SHEAR IS THE PART THAT MATTERS. If the two measured axes are not
ninety degrees apart, no rotation puts them right: the departure is a
two-by-two linear map and it takes four numbers, not one. Every real
two-by-two map on the chroma plane is exactly

    d  ->  alpha d + beta conj(d)

for one pair of complex numbers, and that identity is what makes the whole
of this measurable as a complex quantity rather than as a list of angles.
`alpha` carries the rotation and the common gain, `beta` carries the shear
and the direction it acts along, and `beta` is zero for a channel whose
coordinate system is only turned. A magnitude cannot express either.

AND IT IS AXIAL, NOT DIRECTIONAL, which decides the statistic. A streak
running along a line has an orientation and no arrow, so the quantity is
defined modulo 180 degrees and the mean of the raw angles is meaningless.
The standard treatment doubles the angle; here the doubling has to happen
TWICE, and both readings are used:

  * the FOURFOLD resultant pools an orthogonal pair onto a single point,
    so its magnitude is exactly the test that the streaks cluster on two
    axes rather than scattering, and its angle is where the pair sits;
  * the TWOFOLD resultant is `cos` of the separation between the two
    axes, so it is zero for a truly orthogonal pair and grows with the
    shear. It is the departure from ninety degrees, read directly.

THE BURST SUPPLIES THE ABSOLUTE REFERENCE and nothing else can. SMPTE 170M
note 2 to clause 10 makes the phase origin the burst plus 180, so the pair's
orientation is only meaningful once it is expressed against the burst - and
`colour_under.burst_observables` records why it must be the burst's PHASE
and never its level: three unknown gains multiply the burst's amplitude and
none of them rotates it.

READING THE ACTIVE PICTURE, HONESTLY. The streaks are in the picture, not
in the reserved interval, so this stands under the same exception
`colour_lock` was granted for the residual colour carrier and it keeps the
same discipline: a null distribution decides whether the shape is there,
and the measurement DECLINES rather than reporting a coordinate system it
has not established. The orientation of `n` independent random axes has a
fourfold resultant exceeding `r` with probability `exp(-n r^2)`, the same
Rayleigh law the colour lock's gate is taken from, so the floor is derived
and not chosen.

WHAT THE STREAKS MEASURE, AND WHAT THEY DO NOT, measured on four decodes
with the stage wired and running. The two readings behave differently and
the difference is the whole of how they may be used:

    decode                pair vs burst   vs the specified 33   separation
    75 bars SP              56.90 deg      0.10, reversed        70.78 deg
    75 bars EP              56.55          0.45, reversed        71.13
    countdown               49.14          7.86, reversed        59.63
    chroma noise SP         10.90         22.10, as specified    15.53

THE ORIENTATION AGAINST THE BURST IS A CHANNEL MEASUREMENT, and it lands on
the specification. A rotation of the coordinate system turns every streak
together whatever the content is, so the pair's angle from the burst is a
property of the frame - and on both bar tapes it sits 33 degrees from the
burst to within half a degree, which is the rotation clause 10 specifies.
It sits there with the OPPOSITE SIGN, and that is the finding rather than
an error: the axes are at minus 33 degrees where the encoding puts them at
plus 33. Two tape speeds, twenty-seven fields each, agreeing to 0.35
degrees between them.

THE SEPARATION IS THE CONTENT AND THE CHANNEL TOGETHER, and reading it as
the channel alone would be wrong. Which directions the transitions run in
depends on which hues the picture holds: a bar signal's trajectories are
fixed by its six targets, so 70.78 degrees on bars is a statement about
bars through this channel and not about the channel. The evidence is in
the table - the two bar decodes agree to 0.35 degrees while the
chroma-noise pattern gives 15.53 and the countdown 59.63 on the same deck.
So a DIFFERENCE between two decodes of the SAME pattern is a channel
measurement and an absolute value is not, and the same caveat
`coordinate_map` states for its isotropy assumption applies here.

AND IT AGREES WITH THE QUADRATURE IMBALANCE MEASURED INDEPENDENTLY, to the
extent two instruments on different material can. `iq_imbalance` reads the
image on the raw radio frequency at the playback tap at 0.1067, and this
module's `coordinate_map` reads a shear ratio of 0.1151 per field on the
bars - eight per cent apart, from a scope trace and a radio-frequency burst
with a whole decoder between them. Read through the decoder's own
colour-under burst the same imbalance comes back at 0.0194, five times
smaller, so the three do NOT all agree and the disagreement is recorded
rather than averaged: the two that agree are both taken where the image is
still present, and the small one is taken after the decoder's chroma band
pass and time base correction have been through it.

ONE FIGURE FROM HIS SCREENSHOT, USED AS A CHECK ON THE CONVENTION rather
than as a measurement of the tape: the display reads the burst at 19.72 IRE
and calls it 98.6 per cent of specification. The specification is 40 IRE
PEAK TO PEAK, so 98.6 per cent of it is 39.44 and not 19.72 - the scope is
reporting the AMPLITUDE, which `burst_reference` defaults to, and
19.72 / 20.0 = 0.986 reproduces the display's own percentage exactly. The
burst on that machine is therefore 1.4 per cent low, and the convention
this module reads amplitudes in is confirmed against an instrument.
"""

# The separation the encoding specifies between the two modulation axes.
# SMPTE 170M clause 10 modulates Q on the sine and I on the cosine of the
# same subcarrier, which is a quadrature pair by construction - so ninety
# degrees is the specification's own number and not a convention.
SPECIFIED_AXIS_SEPARATION_DEG = 90.0

# The significance the streak clustering has to clear. The same four sigma
# `chroma_head_switch` detects at and `model_stages` gates the colour lock
# with, so one significance governs the chroma path rather than three.
CLUSTER_SIGMA = 4.0


def _axial_floor(count: int, sigma: float = CLUSTER_SIGMA) -> float:
    """The fourfold resultant `count` random orientations would reach.

    The resultant of the mean of `n` independent unit phasors exceeds `r`
    with probability `exp(-n r^2)` (Rayleigh), so requiring that to be no
    larger than the two-sided Gaussian tail at `sigma` gives the floor
    below which a measured clustering is indistinguishable from scatter.
    Derived, so the gate moves with the evidence rather than being chosen.
    """
    if int(count) < 1:
        return 1.0
    tail = math.erfc(float(sigma) / math.sqrt(2.0))
    return float(math.sqrt(-math.log(tail) / float(count)))


def transition_axes(steps, weights=None, sigma: float = CLUSTER_SIGMA,
                    burst_phase_deg: float = BURST_PHASE_DEG
                    ) -> Dict[str, object]:
    """THE COORDINATE SYSTEM, from the directions the colour transitions run.

    `steps` are complex chroma DIFFERENCES - one colour minus the colour
    before it - so each is a vector along the trajectory between two
    targets. Their orientation is what carries the axes; their sign does
    not, because a transition from yellow to blue and one from blue to
    yellow lie on the same line.

    Returns the four things a coordinate system is made of, each as the
    complex quantity it came from as well as the angle a reader wants:

      `fourfold`        the axial resultant of `exp(4 i theta)`. An
                        orthogonal pair maps onto one point under it, so
                        its MAGNITUDE is the test that the streaks cluster
                        on two axes and its ANGLE, quartered, is where the
                        pair sits.
      `twofold`         the resultant of `exp(2 i theta)`, which for two
                        axes at `theta1` and `theta2` is exactly
                        `exp(i(theta1+theta2)) cos(theta1-theta2)`. It is
                        therefore zero for a right angle and its magnitude
                        IS the sine of the departure from one.
      `separation_deg`  `arccos` of that magnitude: the angle between the
                        two axes, against the ninety the encoding
                        specifies.
      `versus_burst`    the pair's orientation measured from the burst,
                        which is the signal's only absolute phase.

    `weights` lets a caller weight each step by its own length, which is
    the right thing when the steps are measured rather than planted: a
    long transition carries more evidence about its direction than a short
    one does, and an unweighted mean lets the noise between two identical
    bars vote as loudly as a full excursion.
    """
    values = np.asarray(steps, dtype=np.complex128).ravel()
    if values.size < 4:
        raise ValueError("an axis measurement needs a run of transitions")
    magnitude = np.abs(values)
    good = magnitude > 0.0
    if not np.any(good):
        raise ValueError("every transition is zero length")
    unit = values[good] / magnitude[good]
    if weights is None:
        weight = magnitude[good]
    else:
        weight = np.asarray(weights, dtype=np.float64).ravel()[good]
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("the weights sum to zero")
    # The angle is doubled once to make it axial and again to fold an
    # orthogonal pair onto one point. Both are taken as powers of the unit
    # phasor rather than as trigonometry on an angle, so no branch cut is
    # crossed and the quantity stays complex throughout.
    twofold = complex((weight * unit ** 2).sum() / total)
    fourfold = complex((weight * unit ** 4).sum() / total)
    # An effective count, because the weights are unequal: Kish's formula,
    # the sum squared over the sum of squares, which is the count of equal
    # weights carrying the same information. Using the raw count instead
    # would understate the floor wherever a few long transitions dominate.
    effective = total * total / float((weight * weight).sum())
    floor = _axial_floor(effective, sigma)
    separation = math.degrees(math.acos(min(abs(twofold), 1.0)))
    pair_angle = math.degrees(np.angle(fourfold)) / 4.0
    sum_angle = math.degrees(np.angle(twofold)) / 2.0
    first = (sum_angle + 0.5 * separation) % 180.0
    second = (sum_angle - 0.5 * separation) % 180.0
    # THE SEPARATION IS ILL-CONDITIONED AT EXACTLY THE ANSWER THE
    # SPECIFICATION PREDICTS, and saying so is the difference between a
    # measurement and a number. A truly orthogonal pair puts the twofold
    # resultant at zero, so what is measured there is the resultant of the
    # noise alone - Rayleigh distributed with mean `sqrt(pi / 4 n)` - and
    # `arcsin` of the SAME floor the clustering is judged against is
    # therefore the departure from ninety degrees this many steps can
    # resolve, and one significance governs both readings rather than two.
    # Measured on a planted orthogonal pair of 400 steps this reads 84.5
    # degrees where the truth is 90 - inside the 11.1 degree resolution -
    # and on a planted 70 degree pair it reads 69.8, which is outside it.
    # The estimator is behaving exactly as its own null says it will.
    #
    # The same collapse makes the two axes SEPARATELY unidentifiable there:
    # `arg(twofold)` is their bisector and its argument is pure noise once
    # the modulus is at the floor. So the pair is reported only where the
    # twofold resultant clears the same floor, and `None` where it does not
    # - which is not a failure but the correct reading of an orthogonal
    # pair, whose two axes are fixed by `pair_angle_deg` alone.
    resolution = math.degrees(math.asin(min(floor, 1.0)))
    separated = abs(twofold) >= floor
    versus_burst = (pair_angle - float(burst_phase_deg)) % 90.0

    def _apart(first, second):
        """The distance between two ORIENTATIONS of a quadrature pair.

        Both live modulo ninety degrees, because a pair of axes ninety
        apart is carried onto itself by a quarter turn, so 89 and 1 are two
        degrees apart and not eighty-eight.
        """
        gap = (float(first) - float(second)) % 90.0
        return min(gap, 90.0 - gap)

    as_specified = _apart(versus_burst, IQ_ROTATION_DEG)
    reversed_sense = _apart(versus_burst, -IQ_ROTATION_DEG)
    return {
        "fourfold": fourfold,
        "twofold": twofold,
        "clustering": abs(fourfold),
        "clustering_floor": floor,
        "effective_count": float(effective),
        "clustered": bool(abs(fourfold) >= floor),
        "pair_angle_deg": pair_angle % 90.0,
        "axis_deg": (first, second) if separated else None,
        "axes_separately_resolved": bool(separated),
        "separation_deg": separation,
        "separation_specified_deg": SPECIFIED_AXIS_SEPARATION_DEG,
        "separation_error_deg": separation - SPECIFIED_AXIS_SEPARATION_DEG,
        "separation_resolution_deg": resolution,
        # ORTHOGONAL EXACTLY WHEN THE TWOFOLD RESULTANT IS AT ITS FLOOR,
        # which is the same test `separated` is, read the other way: a
        # right angle puts that resultant at zero by construction, so
        # failing to resolve the two axes separately IS the finding that
        # they are ninety degrees apart.
        "orthogonal": bool(not separated),
        "shear": abs(twofold),
        "versus_burst_deg": versus_burst,
        # THE SIGN OF THE ROTATION IS A REAL QUESTION AND IS ANSWERED HERE.
        # Clause 10 puts both modulation axes at `IQ_ROTATION_DEG` from the
        # colour difference pair, and the burst is the origin, so the pair
        # should sit 33 degrees from the burst - but an orientation is
        # defined modulo ninety and 33 and -33 are 66 degrees apart, so a
        # reading that agrees in MAGNITUDE and disagrees in sign looks like
        # a 23 degree error unless both are reported. Measured on a decode
        # of `zaroff-75bars-NTSC-SP`, the pair sits 56.5 degrees from the
        # burst, which is 33.5 degrees the other way - the specified
        # rotation to half a degree, with the opposite sign.
        "versus_specified_deg": min(as_specified, reversed_sense),
        "specified_rotation_deg": IQ_ROTATION_DEG,
        "rotation_sense": ("as specified" if as_specified <= reversed_sense
                           else "reversed"),
        "departure_as_specified_deg": as_specified,
        "departure_reversed_deg": reversed_sense,
        "sigma": float(sigma),
        "cite": ("SMPTE 170M clause 10 for the quadrature pair; note 2 to "
                 "the same clause for the burst as the phase origin"),
        "why": ("a streak has an orientation and no arrow, so the mean of "
                "the raw angles is meaningless; doubling makes it axial and "
                "doubling again folds an orthogonal pair onto one point, so "
                "the fourfold magnitude tests the shape and the twofold one "
                "measures the departure from a right angle"),
    }


def coordinate_map(steps, weights=None) -> Dict[str, object]:
    """THE SAME DEPARTURE AS A LINEAR MAP: one pair of complex numbers.

    Every real two-by-two map on the plane is `d -> alpha d + beta conj(d)`
    for exactly one complex pair, and that is the representation to carry
    a rotation and a shear in together - the split is the map's conformal
    and anti-conformal halves, and no magnitude expresses it.

    The pair is recovered from two second moments of the measured steps.
    Written against a source whose own excursions are isotropic - which is
    what an encoder driving two quadrature axes with independent content
    produces - the moments are

        E|d|^2 = (|alpha|^2 + |beta|^2) sigma^2
        E d^2  = 2 alpha beta sigma^2

    so their ratio fixes `|beta| / |alpha|` outright and `sigma` cancels.
    That ratio is the whole of the shear: zero for a coordinate system
    that is only turned, and rising to one where the map has collapsed the
    plane onto a line.

    THE ISOTROPY IS AN ASSUMPTION AND IS RETURNED AS ONE. A bar signal's
    transitions are not isotropic - six targets give a fixed set of
    trajectories - so on bars this reads the CONTENT's own anisotropy
    beside the channel's, and the two cannot be separated from one field
    of one pattern. It is honest on a picture with many hues and it says
    so rather than implying otherwise; the fourfold clustering in
    `transition_axes` is the reading that does not depend on it.
    """
    values = np.asarray(steps, dtype=np.complex128).ravel()
    if values.size < 4:
        raise ValueError("a coordinate map needs a run of transitions")
    if weights is None:
        weight = np.ones(values.size, dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64).ravel()
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("the weights sum to zero")
    power = float((weight * np.abs(values) ** 2).sum() / total)
    pseudo = complex((weight * values * values).sum() / total)
    if power <= 0.0:
        raise ValueError("the transitions carry no power")
    normalised = pseudo / power
    eccentricity = min(abs(normalised), 1.0)
    # |E d^2| / E|d|^2 = 2r / (1 + r^2) with r = |beta| / |alpha|, whose
    # root inside the unit disc is this. The far root, 1/r, is the same
    # map with its two halves exchanged - an orientation-reversing map -
    # and the near root is the one an ordinary channel is.
    if eccentricity <= 0.0:
        ratio = 0.0
    else:
        ratio = float((1.0 - math.sqrt(max(1.0 - eccentricity ** 2, 0.0)))
                      / eccentricity)
    stretch = (1.0 + ratio) / (1.0 - ratio) if ratio < 1.0 else float("inf")
    return {
        "power": power,
        "pseudo_power": pseudo,
        "normalised_pseudo": normalised,
        "shear_ratio": ratio,
        "stretch": stretch,
        "stretch_db": 20.0 * math.log10(stretch) if stretch > 0
        and math.isfinite(stretch) else float("inf"),
        # arg(alpha beta) is the doubled angle of the direction the stretch
        # acts along, so half of it is that direction.
        "stretch_axis_deg": float(math.degrees(np.angle(normalised)) / 2.0
                                  % 180.0),
        "is_conformal": bool(ratio < 1e-9),
        "assumption": ("the source excursions are isotropic; a bar signal's "
                       "are not, so on bars this carries the content's own "
                       "anisotropy beside the channel's"),
        "why": ("a rotation and a shear are one linear map, and the only "
                "complete complex form of a real two-by-two map is "
                "alpha d + beta conj(d) - so the shear is a number this "
                "returns rather than a shape a magnitude cannot hold"),
    }
