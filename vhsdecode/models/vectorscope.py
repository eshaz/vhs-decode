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
