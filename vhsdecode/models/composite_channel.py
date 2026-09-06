"""THE COMPOSITE VIDEO CHANNEL, from the specifications that define it.

Ethan, 2026-09-06: *"Look at the video spec for the frequency response of
the luma signal, it should be scaled according to that."* and *"The
modeling needs to consider the RF channel response of the composite
signal, just as we do with the other stages with the VCR."* and *"Same
with the chroma and color under carrier response."*

THE FIRST ANSWER IS A NEGATIVE ONE, AND IT MATTERS. SMPTE 170M-2004 clause
7.1: *"This standard does not impose a bandwidth restriction on the
luminance part of the NTSC signal."* So there is no specified luma
frequency response to scale by. What the standard specifies instead is
that the luma should be FLAT and that any limit belongs to the CHANNEL
rather than to the signal - the note to clause 7 puts the familiar 4.2 MHz
where it belongs: *"When the composite NTSC signal is transmitted (or
recorded on some types of video tape recorders), the overall bandwidth is
normally limited to less than 5 MHz, typically 4.2 MHz as for
broadcasting."*

That is exactly the distinction this arc's profile system already draws.
The 4.2 MHz is the television channel's, so it belongs to a `television`
source and not to a `camera` or a `composite` one, and a model that
attaches it to the signal would apply it where it does not exist.

WHERE THE LUMA RESPONSE IS SPECIFIED, then, is in the MEASUREMENT: the
multiburst carries six packets of equal amplitude, and the specification
is that they come back equal. Any departure is the channel.

WHAT IS SPECIFIED WITH NUMBERS is the chrominance, at both ends of the
chain, and all of it is cited below rather than fitted:

  the encoder            SMPTE 170M 7.2      colour difference, before
                                             modulation: under 2 dB down at
                                             1.3 MHz, at least 20 dB down at
                                             3.6 MHz
  the 1953 Q channel     SMPTE 170M note 7   under 2 dB at 0.4 MHz, under
                                             6 dB at 0.5, at least 6 dB at 0.6
  the VHS separator      SMPTE 32M 3.9.2.1.1 band-pass centred 3.58 MHz,
                                             -3 dB at 3.08 and 4.08 MHz
  the burst doubler      SMPTE 32M 3.9.2.1.3 the burst is raised 6.0 +/- 0.5
                                             dB BEFORE recording
  the down-conversion    SMPTE 32M 3.9.2.1.4 to 40 f_H exactly, 629.371 kHz
  the phase rotation     SMPTE 32M 3.9.2.1.5 +90 degrees a line on track 1
  the record level       SMPTE 32M 3.9.2.1.2 played back 7 to 10 dB below
                                             the saturation level
  luma left in the chroma SMPTE 32M 7.5.1.2.1 attenuated more than 20 dB near
                                             1.2 MHz

THE BURST DOUBLER IS THE ONE TO WATCH. Six decibels means the recorded
burst is twice the recorded chroma, so a level taken from the burst and
applied to the chroma is out by a factor of two unless the doubler is
undone - and it is a RECORD-side step, so it is present on every tape and
absent from every composite reference.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

# Exact ratios, not microsecond figures (see sync_geometry for why that
# distinction matters).
NTSC_LINE_RATE_HZ = 525.0 * 30.0 / 1.001
NTSC_SUBCARRIER_HZ = 455.0 / 2.0 * NTSC_LINE_RATE_HZ
COLOUR_UNDER_HZ = 40.0 * NTSC_LINE_RATE_HZ          # SMPTE 32M 3.9.2.1.4

# SMPTE 170M-2004 clause 7.2, the colour-difference limit before modulation
COLOUR_DIFFERENCE_POINTS = (
    (1.3e6, -2.0, "less than 2 dB down"),
    (3.6e6, -20.0, "at least 20 dB down"),
)
# SMPTE 170M-2004 note to clause 7, the 1953 narrowband Q channel
Q_CHANNEL_POINTS = (
    (0.4e6, -2.0, "less than 2 dB down"),
    (0.5e6, -6.0, "less than 6 dB down"),
    (0.6e6, -6.0, "at least 6 dB down"),
)
# SMPTE 32M-2004 clause 3.9.2.1.1, the VHS chroma separation band-pass
SEPARATOR_CENTRE_HZ = 3.58e6
SEPARATOR_MINUS_3DB_HZ = (3.08e6, 4.08e6)
# SMPTE 32M-2004 clause 3.9.2.1.3
BURST_DOUBLER_DB = 6.0
BURST_DOUBLER_TOLERANCE_DB = 0.5
# SMPTE 32M-2004 clause 3.9.2.1.5
PHASE_ROTATION_DEG_PER_LINE = 90.0
# SMPTE 32M-2004 clause 3.9.2.1.2
RECORD_LEVEL_BELOW_SATURATION_DB = (7.0, 10.0)
# SMPTE 32M-2004 clause 7.5.1.2.1
LUMA_IN_CHROMA_ATTENUATION_DB = 20.0
LUMA_IN_CHROMA_NEAR_HZ = 1.2e6
# The transmission channel's limit, which belongs to the channel and not
# to the signal (SMPTE 170M note to clause 7)
BROADCAST_LIMIT_HZ = 4.2e6
# ITU-R BT.1700, 525-line: the burst is 9 +/- 1 cycles of subcarrier at
# 40 IRE peak-to-peak, on every line, on every field
BURST_CYCLES = 9.0
BURST_CYCLES_TOLERANCE = 1.0
BURST_IRE_PEAK_TO_PEAK = 40.0
# SMPTE 170M-2004 clause 8.4: 227.5 subcarrier cycles a line, so the burst
# phase alternates by exactly half a cycle from one line to the next
SUBCARRIER_CYCLES_PER_LINE = 455.0 / 2.0

# ITU-R J.63 Table III, the 525-line multiburst, and EBU Tech 3209 7.2.9
MULTIBURST_HZ = (0.5e6, 1.0e6, 2.0e6, 3.0e6, 3.58e6, 4.2e6)
MULTIBURST_IRE = 50.0
MULTIBURST_TOLERANCE_IRE = 0.5


def luma_response(frequency_hz, source: str = "composite") -> Dict[str, object]:
    """What the standard says the luma response should be: FLAT.

    `source` decides whether the transmission channel's limit applies, and
    that is the profile system's distinction rather than this module's: a
    camera or a line input has no 4.2 MHz anywhere, and attaching one to
    the signal would apply a broadcast channel to a tape that never saw a
    broadcast.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    response = np.ones(f.size, dtype=np.float64)
    limited = source == "television"
    if limited:
        # the channel's own limit, not the signal's; a brick wall is the
        # specification's language ("normally limited to") and not a shape
        # it gives, so this marks the edge rather than modelling a skirt
        response = np.where(f <= BROADCAST_LIMIT_HZ, 1.0, np.nan)
    return {
        "frequency_hz": f, "response": response,
        "specified_flat": True,
        "limit_applies": bool(limited),
        "limit_hz": BROADCAST_LIMIT_HZ if limited else None,
        "cite": ("SMPTE 170M-2004 clause 7.1: the standard imposes no "
                 "bandwidth restriction on the luminance; the note to "
                 "clause 7 puts the 4.2 MHz on the transmission channel"),
        "why": ("there is no specified luma response to scale by - the "
                "specification is flatness, and the departure from it is "
                "what the multiburst measures"),
    }


def multiburst_reference() -> Dict[str, object]:
    """The six equal packets that MEASURE the luma response.

    The specification is that they come back equal, so the measured
    departure from equality is the channel's response at those six
    frequencies, with the standard's own tolerance as the error bar.
    """
    return {
        "frequency_hz": np.asarray(MULTIBURST_HZ),
        "amplitude_ire": MULTIBURST_IRE,
        "tolerance_ire": MULTIBURST_TOLERANCE_IRE,
        "tolerance_db": 20.0 * math.log10(
            (MULTIBURST_IRE + MULTIBURST_TOLERANCE_IRE) / MULTIBURST_IRE),
        "cite": ("ITU-R J.63 Table III (525-line multiburst, line 17 field "
                 "2) and EBU Tech 3209 section 7.2.9; 50 +/- 0.5 IRE "
                 "peak-to-peak on the pedestal"),
        "why": ("equal amplitudes are the specification, so the departure "
                "from equal IS the response"),
    }


def _points_to_response(frequency_hz, points) -> np.ndarray:
    """Interpolate a response through the specification's own dB points.

    The standard gives limits at named frequencies and says nothing between
    them, so this joins them in the log domain and marks the extrapolation
    beyond the last point as unspecified rather than continuing a slope the
    standard never gave.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    named = np.array([p[0] for p in points], dtype=np.float64)
    levels = np.array([p[1] for p in points], dtype=np.float64)
    out = np.interp(f, named, levels, left=0.0, right=np.nan)
    return out


def colour_difference_limit(frequency_hz, channel: str = "equiband"
                            ) -> Dict[str, object]:
    """The encoder's colour-difference limit, SMPTE 170M clause 7.2."""
    points = Q_CHANNEL_POINTS if channel == "Q1953" else COLOUR_DIFFERENCE_POINTS
    decibels = _points_to_response(frequency_hz, points)
    return {
        "frequency_hz": np.asarray(frequency_hz, dtype=np.float64).ravel(),
        "decibels": decibels,
        "points": points,
        "channel": channel,
        "cite": ("SMPTE 170M-2004 clause 7.2" if channel != "Q1953"
                 else "SMPTE 170M-2004, note to clause 7 (the 1953 Q channel)"),
        "shape_note": ("clause 7.3 asks for a minimum of ringing and "
                       "overshoot, naming a Gaussian characteristic as an "
                       "example, so the standard constrains the SHAPE only "
                       "by that preference"),
    }


def separator_response(frequency_hz) -> Dict[str, object]:
    """The VHS chroma separation band-pass, SMPTE 32M clause 3.9.2.1.1.

    Centred at 3.58 MHz with its half-power points at 3.08 and 4.08 MHz.
    The standard gives three points and no order, so a single-pole
    band-pass is fitted THROUGH them and its order is labelled an
    assumption rather than a specification.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    low, high = SEPARATOR_MINUS_3DB_HZ
    centre = math.sqrt(low * high)
    bandwidth = high - low
    quality = centre / bandwidth
    with np.errstate(divide="ignore", invalid="ignore"):
        detuning = quality * (f / centre - centre / np.maximum(f, 1e-30))
    response = 1.0 / np.sqrt(1.0 + detuning ** 2)
    return {
        "frequency_hz": f, "response": response,
        "centre_hz": centre, "specified_centre_hz": SEPARATOR_CENTRE_HZ,
        "minus_3db_hz": SEPARATOR_MINUS_3DB_HZ,
        "quality_factor": quality,
        "cite": "SMPTE 32M-2004 clause 3.9.2.1.1",
        "assumption": ("the standard names a centre and two half-power "
                       "points and no order; a single pole pair is fitted "
                       "through them, and the geometric centre it implies "
                       "is 3.546 MHz against the stated 3.58"),
    }


def chroma_chain() -> Dict[str, object]:
    """Every specified step of the VHS chrominance path, in order.

    THE BURST DOUBLER IS THE TRAP. Six decibels before recording means the
    recorded burst is twice the recorded chroma, so a level taken from the
    burst and applied to the chroma is out by a factor of two unless it is
    undone - and it is a record-side step, present on every tape and absent
    from every composite reference.
    """
    return {
        "steps": [
            {"step": "separation band-pass", "cite": "SMPTE 32M 3.9.2.1.1",
             "centre_hz": SEPARATOR_CENTRE_HZ,
             "minus_3db_hz": SEPARATOR_MINUS_3DB_HZ},
            {"step": "record level", "cite": "SMPTE 32M 3.9.2.1.2",
             "below_saturation_db": RECORD_LEVEL_BELOW_SATURATION_DB},
            {"step": "burst amplitude doubler", "cite": "SMPTE 32M 3.9.2.1.3",
             "gain_db": BURST_DOUBLER_DB,
             "tolerance_db": BURST_DOUBLER_TOLERANCE_DB,
             "gain_linear": 10.0 ** (BURST_DOUBLER_DB / 20.0)},
            {"step": "down-conversion", "cite": "SMPTE 32M 3.9.2.1.4",
             "carrier_hz": COLOUR_UNDER_HZ,
             "exact_ratio": "40 x f_H, so one line is exactly 40 cycles"},
            {"step": "carrier phase rotation", "cite": "SMPTE 32M 3.9.2.1.5",
             "degrees_per_line": PHASE_ROTATION_DEG_PER_LINE,
             "track": "advance on track 1"},
            {"step": "luma left in the chroma", "cite": "SMPTE 32M 7.5.1.2.1",
             "attenuation_db": LUMA_IN_CHROMA_ATTENUATION_DB,
             "near_hz": LUMA_IN_CHROMA_NEAR_HZ},
        ],
        "burst_doubler_warning": (
            "the recorded burst is 6 dB - a factor of two - above the "
            "recorded chroma, so a burst-referenced level must undo it "
            "before it is applied to the chroma"),
        "why": ("every number here is cited to a clause; nothing in this "
                "chain is fitted"),
    }


def burst_amplitude_response(offset_hz=None, cycles: float = BURST_CYCLES,
                             points: int = 2001) -> Dict[str, object]:
    """THE EXACT SHAPE THE SPECIFICATION GIVES THE BURST'S AMPLITUDE.

    Ethan, 2026-09-06: *"composite channel specifies an exact shape, i.e.
    frequency response of the amplitude of the color carrier burst, that
    should be the differential to use for correcting it's amplitude."*

    The burst is not a level, it is a band. Nine cycles of subcarrier is
    2.5143 microseconds, so by the same 1/T law the sync pulse obeys the
    burst occupies the carrier plus or minus 397.7 kHz, and across that
    band the specified chroma separator is not flat. This returns that
    shape and, more to the point, its DIFFERENTIAL, which is the quantity
    the burst's amplitude has to be corrected by.

    WHY A DIFFERENTIAL AND NOT A LEVEL, which is the whole of it. The
    separator's three specified numbers - centred 3.58 MHz, half power at
    3.08 and 4.08 MHz (SMPTE 32M 3.9.2.1.1) - put its geometric centre at
    3.5449 MHz, which is 34.6 kHz BELOW the subcarrier. The carrier
    therefore sits on the falling side of the specified band-pass, and the
    response has a slope there of -1.1803 dB per MHz. Over the burst's own
    half-band that is -0.4695 dB, and across the whole main lobe the upper
    null comes back 0.2118 dB below the lower one.

    WHAT THE TILT DOES, measured rather than assumed, because the first
    version of this docstring got it wrong. A slope across the burst's
    spectrum is an ODD perturbation about the carrier, so it maps to a
    quadrature term that is odd about the burst's CENTRE in time - measured
    odd rms 4.75e-03 against an even rms of 2.03e-17. An odd term sums to
    zero, so it moves neither the burst's bulk phase nor its envelope
    centroid:

        phase of the burst's sum          0.000000 degrees
        envelope centroid shift          -0.000000 samples

    It appears instead as a phase RAMP ACROSS the burst, from -2.3871
    degrees at the first sample to +2.3871 at the last, a span of 4.7743
    degrees and a peak excursion of 1.85 nanoseconds. Read through this
    module's own instruments, an ideal burst passed through the specified
    response gives:

        `burst_instrument.timing` bulk phase      moved 0.000000 degrees
        `burst_instrument.timing` scatter                0.9307 degrees
        `burst_instrument.amplitude` peak-to-peak       -0.7391 dB

    So the bulk phase reading is completely blind to the tilt, and the
    scatter reading sees it but reads it as NOISE rather than as a
    specified shape. The magnitude sees only the band-pass's even part.
    That is why the correction has to be a differential across the band:
    neither of the two scalar readings can carry it.

    AND IT IS NOT CONFIRMED ON TAPE. The prediction was tested against the
    within-burst phase span measured on three decodes, 241 active lines a
    field:

        decode                    measured span        predicted
        wide 75bars SP           -0.7336 +/- 0.066     +4.7743
        wide chromanoise SP      +0.1899 +/- 0.058     +4.7743
        wide 75bars EP           +6.4542 +/- 0.141     +4.7743

    The two SP decodes show essentially none of it and the EP decode shows
    rather more, so this is published as a BOUND and not as a correction to
    apply. The standard gives the separator no order (see
    `separator_response`), and the order cannot bridge the gap: sweeping it
    from one to four moves the predicted span only from +4.7743 to +2.2290,
    never near zero. The likelier reading is that the decoder's own chroma
    path has already equalised the separator before the chroma time-base
    file is written, which would put this measurement on the RF side of the
    chroma filtering rather than after it.

    THE ENCODER'S LIMIT DOES NOT APPLY HERE. SMPTE 170M clause 7.2 shapes
    the colour DIFFERENCE signals before modulation; the burst is the
    unmodulated reference carrier itself and passes no such filter. Only
    the separator shapes it, so only the separator is used.
    """
    carrier = NTSC_SUBCARRIER_HZ
    duration = float(cycles) / carrier
    null_hz = 1.0 / duration
    if offset_hz is None:
        offset = np.linspace(-null_hz, null_hz, int(points))
    else:
        offset = np.asarray(offset_hz, dtype=np.float64).ravel()
    frequency = carrier + offset
    separator = separator_response(frequency)
    response = np.asarray(separator["response"], dtype=np.float64)
    decibels = 20.0 * np.log10(np.maximum(response, 1e-300))
    # the differential at the carrier, taken on the specification's own
    # closed form rather than on a difference of samples
    step = null_hz * 1e-4
    high = float(separator_response([carrier + step])["response"][0])
    low = float(separator_response([carrier - step])["response"][0])
    slope_db_per_hz = (20.0 * math.log10(high) - 20.0 * math.log10(low)) / (2.0 * step)
    at_carrier = float(separator_response([carrier])["response"][0])
    edge_low = float(separator_response([carrier - null_hz])["response"][0])
    edge_high = float(separator_response([carrier + null_hz])["response"][0])
    return {
        "frequency_hz": frequency,
        "offset_hz": offset,
        "response": response,
        "decibels": decibels,
        "carrier_hz": carrier,
        "cycles": float(cycles),
        "duration_s": duration,
        "half_band_hz": null_hz,
        "at_carrier": at_carrier,
        "at_carrier_db": 20.0 * math.log10(at_carrier),
        "slope_db_per_hz": slope_db_per_hz,
        "slope_db_per_mhz": slope_db_per_hz * 1e6,
        "half_band_db": slope_db_per_hz * null_hz,
        "tilt_db": 20.0 * math.log10(edge_high / edge_low),
        "centre_offset_hz": float(separator["centre_hz"]) - carrier,
        "cite": ("SMPTE 32M-2004 clause 3.9.2.1.1 for the separator, "
                 "ITU-R BT.1700 for the 9-cycle burst that sets the band"),
        "why": ("the burst occupies a band, the specified response is not "
                "flat across it, and the slope is the differential the "
                "burst's amplitude has to be corrected by"),
    }


def burst_quadrature_prediction(cycles: float = BURST_CYCLES,
                                sample_rate_hz: Optional[float] = None,
                                length: int = 4096) -> Dict[str, object]:
    """What the specified tilt does to an ideal burst, computed not asserted.

    Builds the burst the standard describes - a rectangle of `cycles`
    cycles - passes it through `separator_response` about the carrier, and
    reports what comes back. The control is the same response symmetrised
    about the carrier, which must show no tilt at all and does, to 1.04e-14
    degrees of span against the 4.7743 the full response gives.

    READ `phase_span_deg`, NOT A BULK PHASE. The tilt is odd about the
    burst's centre, so it leaves the burst's total phase exactly where it
    was (`bulk_phase_deg`, measured 0.000000) and shows only as a ramp
    across the burst. Anything that sums the burst before taking an angle
    is blind to it.
    """
    carrier = NTSC_SUBCARRIER_HZ
    rate = 4.0 * carrier if sample_rate_hz is None else float(sample_rate_hz)
    width = int(round(float(cycles) / carrier * rate))
    if width < 4 or width >= length:
        raise ValueError("the burst must fit inside the transform length")
    envelope = np.zeros(int(length), dtype=np.complex128)
    start = (int(length) - width) // 2
    envelope[start:start + width] = 1.0
    offset = np.fft.fftfreq(int(length), d=1.0 / rate)
    spectrum = np.fft.fft(envelope)
    full = np.asarray(separator_response(carrier + offset)["response"])
    mirrored = np.asarray(separator_response(carrier - offset)["response"])
    passed = np.fft.ifft(spectrum * full)
    control = np.fft.ifft(spectrum * 0.5 * (full + mirrored))
    window = slice(start, start + width)

    def share(values):
        in_phase = math.sqrt(float(np.mean(values.real ** 2)))
        quadrature = math.sqrt(float(np.mean(values.imag ** 2)))
        return quadrature / in_phase if in_phase > 0 else float("nan")

    def span(values):
        angle = np.degrees(np.unwrap(np.angle(values[window])))
        return float(angle[-1] - angle[0]), float(np.abs(angle).max())

    quadrature = share(passed)
    phase_span, peak = span(passed)
    control_span, _ = span(control)
    return {
        "quadrature_share": quadrature,
        "phase_span_deg": phase_span,
        "peak_excursion_deg": peak,
        "peak_excursion_s": peak / 360.0 / carrier,
        "bulk_phase_deg": math.degrees(float(np.angle(passed.sum()))),
        "centroid_shift_samples": float(
            (np.arange(length) * np.abs(passed)).sum() / np.abs(passed).sum()
            - (np.arange(length) * np.abs(envelope)).sum()
            / np.abs(envelope).sum()),
        "magnitude_db": 20.0 * math.log10(
            float(np.abs(passed).sum() / np.abs(envelope).sum())),
        "control_quadrature_share": share(control),
        "control_span_deg": control_span,
        "sample_rate_hz": rate,
        "burst_samples": width,
        "why": ("an odd perturbation of the spectrum about the carrier is "
                "a quadrature term odd about the burst's centre, so it "
                "moves no bulk phase and no centroid and shows only as a "
                "ramp across the burst"),
    }
