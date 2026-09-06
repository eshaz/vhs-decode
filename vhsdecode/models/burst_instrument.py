"""THE BURST AS THE CHROMA'S INSTRUMENT, in all three dimensions.

Ethan, 2026-09-06: *"It's going to be based on the up-heterodyned burst
shape, the entire three dimensions, how the color stretches, the active
area is contents we want to keep."*

THE SYMMETRY WITH THE LUMA IS EXACT, and it is why this module is short.
The sync pulse is a known shape sent through the luma channel, so its
measured shape divided by its ideal IS that channel's response - frequency
from the edge's spectrum, amplitude from the two edges landing on two
different carriers, time from the phase and from the per-field series. The
burst is the same object for the chrominance: a known number of cycles at
a known amplitude and a known phase, so what comes back measures the
chroma channel in the same three dimensions.

  FREQUENCY   the burst envelope's shape. Nine cycles of subcarrier is
              2.5 microseconds (ITU-R BT.1700), so by the same 1/T law the
              burst resolves 400 kHz and no better - which is why it can
              say how the chroma band is shaped but not where a narrow
              notch sits. "How the colour stretches", in Ethan's words: a
              chroma channel that has lost bandwidth returns a burst whose
              envelope has spread, and the spreading IS the response.
  AMPLITUDE   the burst's own level against its specification, remembering
              that SMPTE 32M clause 3.9.2.1.3 RAISES it 6.0 +/- 0.5 dB
              before recording, so the recorded burst is twice the recorded
              chroma and a level read from one and applied to the other is
              out by a factor of two.
  TIME        the burst's phase, which is the finest timing instrument in
              the signal: one degree at the subcarrier is 0.776 nanoseconds
              against the 14 nanoseconds a sync edge affords at a tenth of
              its 140 nanosecond rise (`sync_geometry`).

UP-HETERODYNED, which is the qualifier that matters. On tape the burst
lives at 40 f_H, and it has been through the record-side rotation of 90
degrees a line (SMPTE 32M 3.9.2.1.5) and the doubler. The instrument
measures it after the up-conversion, where it is comparable with the
specification, and `chroma._complex_envelope` is what recovers it without
discarding the phase - the defect that made the earlier burst instrument
rank two of four.

THE TWO CORRECTIONS DO NOT SUBSTITUTE FOR EACH OTHER, which is the one
thing to carry away from `amplitude_differential` and `field_constant`
standing side by side. The specified shape is a fixed filter: identical on
every line of every field, so it moves the burst's level and its within-
burst shape and cannot account for a single part of the line-to-line or
field-to-field variation. Applying it line by line on a real decode moves
the level +0.0284 dB and leaves the field-to-field excess at 1.62 against
1.69 and the lag-one correlation at +0.4782 against +0.4787 - both inside
the estimator's own noise. So the specified differential is one term and
the per-field residual is another, and neither can be spent on the other.

AND THE ACTIVE AREA IS THE CONTENT, not the measurement. The burst is in
the reserved interval; the correction it yields is applied to the picture.
That is the same division the luma side already keeps
(`sync_geometry.frame_mask`), and it is why this module measures on the
burst alone and never on the colour in the picture.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

from vhsdecode.models import composite_channel, hypercomplex

# ITU-R BT.1700, 525-line: the burst is 9 cycles of subcarrier. Bound to
# `composite_channel` rather than retyped, so the specification has one home.
BURST_CYCLES = composite_channel.BURST_CYCLES
BURST_CYCLES_TOLERANCE = composite_channel.BURST_CYCLES_TOLERANCE
BURST_IRE_PEAK_TO_PEAK = composite_channel.BURST_IRE_PEAK_TO_PEAK
BURST_DURATION_S = BURST_CYCLES / composite_channel.NTSC_SUBCARRIER_HZ
# SMPTE 170M clause 8.4 gives 227.5 cycles a line, so the composite burst
# turns half a cycle a line; SMPTE 32M 3.9.2.1.5 gives the record-side
# rotation the colour-under burst turns instead.
LINE_ROTATION_RAD = {
    "composite": 2.0 * math.pi * (
        composite_channel.SUBCARRIER_CYCLES_PER_LINE % 1.0),
    "colour_under": math.radians(composite_channel.PHASE_ROTATION_DEG_PER_LINE),
}


def probe_resolution_hz(cycles: float = BURST_CYCLES) -> Dict[str, float]:
    """What the burst can resolve, by the same 1/T law the sync pulse obeys.

    Nine cycles at 3.579545 MHz is 2.514 microseconds, so the burst carries
    nothing finer than 397.7 kHz. It describes the shape of the chroma band
    and cannot locate a narrow feature inside it.
    """
    duration = float(cycles) / composite_channel.NTSC_SUBCARRIER_HZ
    return {
        "cycles": float(cycles),
        "duration_s": duration,
        "duration_us": duration * 1e6,
        "resolution_hz": 1.0 / duration,
        "cite": "ITU-R BT.1700, 525-line: burst duration 9 +/- 1 cycles",
        "why": ("a probe of duration T carries nothing below 1/T, and the "
                "burst is short"),
    }


def envelope_response(envelope, sample_rate_hz: float,
                      ideal: Optional[Sequence] = None) -> Dict[str, object]:
    """THE FREQUENCY DIMENSION: how the colour stretches.

    The complex envelope of the measured burst against the envelope the
    specification implies - a rectangle of `BURST_CYCLES` cycles, softened
    only by the standard's own envelope rise. Their ratio in the transform
    domain is the chroma channel's response about the subcarrier, and its
    spread in time is the same statement read the other way.

    Returns both readings, because they are the same measurement and a
    disagreement between them is a fault in the window rather than in the
    channel.
    """
    measured = np.asarray(envelope, dtype=np.complex128).ravel()
    if measured.size < 4:
        raise ValueError("a burst envelope needs more than a few samples")
    if ideal is None:
        want = int(round(BURST_DURATION_S * float(sample_rate_hz)))
        reference = np.zeros(measured.size, dtype=np.complex128)
        start = max((measured.size - want) // 2, 0)
        reference[start:start + want] = 1.0
    else:
        reference = np.asarray(ideal, dtype=np.complex128).ravel()
    spectrum = np.fft.fft(measured)
    reference_spectrum = np.fft.fft(reference)
    good = np.abs(reference_spectrum) > 1e-9 * np.abs(reference_spectrum).max()
    transfer = np.full(spectrum.shape, np.nan, dtype=np.complex128)
    transfer[good] = spectrum[good] / reference_spectrum[good]
    # the spread, read in time: the envelope's second moment about its own
    # centroid, against the reference's
    def spread(values):
        """The envelope's second moment, weighted by AMPLITUDE not power.

        A rectangle of width W convolved with a Gaussian of width sigma has
        an amplitude profile whose second moment is W^2/12 + sigma^2, so
        the moment grows with the loss, which is the whole point. Weighting
        by power instead makes it FALL - squaring de-emphasises exactly the
        gradual tails that the spreading creates - and the first version of
        this did that and reported a stretch below one for a channel that
        had lost bandwidth.
        """
        weight = np.abs(values)
        total = weight.sum()
        if total <= 0:
            return float("nan")
        index = np.arange(values.size)
        centre = float((index * weight).sum() / total)
        return float(np.sqrt(((index - centre) ** 2 * weight).sum() / total))
    measured_spread = spread(measured)
    reference_spread = spread(reference)
    return {
        "frequency_hz": np.fft.fftfreq(measured.size, d=1.0 / sample_rate_hz),
        "transfer": transfer,
        "measured_spread_samples": measured_spread,
        "reference_spread_samples": reference_spread,
        "stretch": measured_spread / reference_spread
        if reference_spread else float("nan"),
        "resolution": probe_resolution_hz(),
        "why": ("a chroma channel that has lost bandwidth returns a burst "
                "whose envelope has spread, so the stretch and the transfer "
                "are one measurement seen two ways"),
    }


def amplitude(envelope, doubler_applied: bool = True) -> Dict[str, object]:
    """THE AMPLITUDE DIMENSION, with the record-side doubler undone.

    SMPTE 32M clause 3.9.2.1.3 raises the burst 6.0 +/- 0.5 dB before
    recording, so a level taken from a played-back burst is twice the level
    of the chroma beside it. `doubler_applied` says whether the measurement
    still carries that gain; the returned `chroma_referred` is what the
    picture's colour should be scaled against either way.
    """
    values = np.asarray(envelope, dtype=np.complex128).ravel()
    peak_to_peak = 2.0 * float(np.abs(values).mean())
    gain = 10.0 ** (composite_channel.BURST_DOUBLER_DB / 20.0)
    chroma_referred = peak_to_peak / gain if doubler_applied else peak_to_peak
    return {
        "burst_peak_to_peak": peak_to_peak,
        "doubler_applied": bool(doubler_applied),
        "doubler_db": composite_channel.BURST_DOUBLER_DB,
        "doubler_linear": gain,
        "chroma_referred": chroma_referred,
        "specified_ire": BURST_IRE_PEAK_TO_PEAK,
        "cite": "SMPTE 32M-2004 clause 3.9.2.1.3; ITU-R BT.1700 for the 40 IRE",
        "why": ("the recorded burst is a factor of two above the recorded "
                "chroma, so a burst-referenced level must undo the doubler "
                "before it is applied to the picture"),
    }


def timing(envelope, reference_phase_rad: float = 0.0) -> Dict[str, object]:
    """THE TIME DIMENSION: the burst's phase, the finest clock in the signal.

    One degree at the subcarrier is 0.776 nanoseconds, against roughly 14
    nanoseconds for a sync edge located to a tenth of its specified rise.
    So the burst is about eighteen times the finer instrument, and it is
    the one to carry the time base once the sync edge has supplied the
    integer line.
    """
    values = np.asarray(envelope, dtype=np.complex128).ravel()
    total = values.sum()
    phase = float(np.angle(total)) - float(reference_phase_rad)
    per_degree_s = 1.0 / (360.0 * composite_channel.NTSC_SUBCARRIER_HZ)
    scatter = float(np.std(np.angle(values * np.exp(-1j * np.angle(total)))))
    return {
        "phase_rad": phase,
        "phase_deg": math.degrees(phase),
        "seconds": math.degrees(phase) * per_degree_s,
        "per_degree_s": per_degree_s,
        "scatter_rad": scatter,
        "scatter_seconds": math.degrees(scatter) * per_degree_s,
        "why": ("a degree of subcarrier is 0.776 nanoseconds where a sync "
                "edge affords about fourteen, so the burst carries the "
                "fraction and the edge carries the integer line"),
    }


def three_dimensions(envelope, sample_rate_hz: float,
                     doubler_applied: bool = True) -> Dict[str, object]:
    """All three at once, which is what the chroma stage needs to align on."""
    return {
        "frequency": envelope_response(envelope, sample_rate_hz),
        "amplitude": amplitude(envelope, doubler_applied),
        "time": timing(envelope),
        "why": ("the same three dimensions the sync pulse gives the luma, "
                "measured on the burst for the chroma, so the two stages "
                "can be aligned against each other"),
    }


def amplitude_differential(envelope, sample_rate_hz: float,
                           doubler_applied: bool = True) -> Dict[str, object]:
    """THE AMPLITUDE CORRECTION AS A DIFFERENTIAL, not as a scale factor.

    Ethan, 2026-09-06: *"composite channel specifies an exact shape, i.e.
    frequency response of the amplitude of the color carrier burst, that
    should be the differential to use for correcting it's amplitude."*

    `amplitude` above returns a level, and a level cannot carry a shape.
    The specification gives an exact one: the separator's response across
    the burst's own band (`composite_channel.burst_amplitude_response`),
    which slopes -1.1803 dB per MHz at the carrier because the separator's
    geometric centre sits 34.6 kHz below the subcarrier. This divides that
    specified shape back out of the measured burst across the whole band,
    which is the differential correction, and returns the complex amplitude
    that results.

    WHAT THE DIFFERENTIAL BUYS OVER THE LEVEL. The specified slope is odd
    about the carrier, so what it puts into the envelope is odd about the
    burst's centre in time - which means it moves neither the bulk phase
    (measured 0.000000 degrees) nor the envelope's centroid (-0.000000
    samples), and appears only as a phase ramp across the burst, -2.3871
    degrees at the first sample to +2.3871 at the last. Neither of this
    module's two scalar readings can carry that: `timing` sums before it
    takes an angle and so moves by 0.000000 degrees, and `amplitude` takes
    a modulus and sees only the band-pass's even part, -0.7391 dB. `timing`
    does see it in its `scatter`, at 0.9307 degrees, but reads it there as
    noise rather than as a shape the standard specifies.

    THE BAND LIMIT IS NECESSARY, NOT PRUDENCE. Dividing the specified
    response out across the whole transform returns 3.68e+20 rms error,
    because far from the carrier the separator's response is near zero and
    the division is by nothing. Restricting the correction to the burst's
    own main lobe holds 90.307 per cent of the burst's energy and leaves an
    rms residual of 4.80e-02 on a unit burst, which is the other 9.7 per
    cent still carrying the response. Inside the band the correction is a
    division and therefore exact.

    AND SEE `composite_channel.burst_amplitude_response` BEFORE APPLYING
    IT. The predicted ramp is not confirmed on tape - two SP decodes
    measure -0.73 and +0.19 degrees of span against +4.77 predicted - so
    the shape is currently a bound rather than a correction.

    The doubler is undone exactly as `amplitude` undoes it, because it is a
    gain and not a shape, and the two must not disagree; that is checked,
    and both give 0.501187.
    """
    measured = np.asarray(envelope, dtype=np.complex128).ravel()
    if measured.size < 4:
        raise ValueError("a burst envelope needs more than a few samples")
    rate = float(sample_rate_hz)
    offset = np.fft.fftfreq(measured.size, d=1.0 / rate)
    shape = composite_channel.burst_amplitude_response(offset_hz=offset)
    response = np.asarray(shape["response"], dtype=np.float64)
    # only where the specification actually reaches: outside the burst's own
    # band the division would amplify what the standard never described
    half_band = float(shape["half_band_hz"])
    inside = np.abs(offset) <= half_band
    corrected_spectrum = np.fft.fft(measured).copy()
    corrected_spectrum[inside] /= response[inside]
    corrected = np.fft.ifft(corrected_spectrum)
    gain = 10.0 ** (composite_channel.BURST_DOUBLER_DB / 20.0)
    scale = 1.0 / gain if doubler_applied else 1.0
    before = complex(measured.sum())
    after = complex(corrected.sum())
    turn = after / before if abs(before) > 0 else complex("nan")
    return {
        "corrected": corrected * scale,
        "corrected_amplitude": after * scale,
        "measured_amplitude": before * scale,
        "differential": turn,
        "magnitude_db": 20.0 * math.log10(abs(turn)) if abs(turn) > 0
        else float("nan"),
        "phase_deg": math.degrees(np.angle(turn)),
        "seconds": float(np.angle(turn)) / (
            2.0 * math.pi * composite_channel.NTSC_SUBCARRIER_HZ),
        "bins_corrected": int(inside.sum()),
        "half_band_hz": half_band,
        "slope_db_per_mhz": shape["slope_db_per_mhz"],
        "doubler_applied": bool(doubler_applied),
        "is_complex": True,
        "cite": shape["cite"],
        "why": ("the specified response is a shape across the burst's band "
                "and not a number, so the correction that undoes it is a "
                "differential, and its odd part is a rotation a magnitude "
                "cannot see"),
    }


def per_line(amplitudes, phases=None, domain: str = "composite"
             ) -> Dict[str, object]:
    """THE SPEC'S PER-LINE MODEL, and what the measurement leaves after it.

    Ethan, 2026-09-06: *"the same spec derives the burst per line and
    constant per field. which should be consistent and follow the model
    build on the expected shape in hilbert space."*

    Both readings come from one specification. ITU-R BT.1700 puts the burst
    at 40 IRE peak-to-peak on EVERY line, so the amplitude's model is a
    constant. SMPTE 170M clause 8.4 puts 227.5 subcarrier cycles in a line,
    so the composite burst's phase turns exactly half a cycle a line; on
    tape it is SMPTE 32M 3.9.2.1.5's 90 degrees instead. Neither is fitted.

    The residual is carried into Hilbert space by its own analytic form,
    because that is the only representation in which a real per-line series
    has a phase to relate to anything else.

    MEASURED ON FOUR DECODES, 241 active lines a field, burst window 74 to
    110 at 4fsc which is exactly 9.00 subcarrier cycles - the specified
    number, read off the decoder's own geometry:

        decode                  departure from 180 deg/line   scatter  sigma
        wide 75bars SP          +0.002676 +/- 0.000424 deg    2.34 deg   6.3
        filt 75bars SP          +0.002678 +/- 0.000424        2.34       6.3
        wide chromanoise SP     +0.003030 +/- 0.000400        2.29       7.6
        wide 75bars EP          +0.000999 +/- 0.000366        2.06       2.7

    The first two are one tape decoded two ways and agree to five decimal
    places, so this is three measurements and not four. The specified
    rotation is right to three decimal places of a degree a line, but the
    departure is six to eight times its own error bar on all three SP
    readings and shares their sign, so it is a real term rather than a
    rounding one - 2.1 picoseconds a line, half a nanosecond across a
    field. The EP decode's departure is a third the size and inside three
    error bars, which is the one place the specification is confirmed
    outright. The scatter about the model is the instrument's own, at 2.1
    to 2.3 degrees.

    The amplitude model is the weaker of the two: see `field_constant` for
    how much weaker, and for the sense in which the two readings of the
    same specification do not agree.
    """
    values = np.asarray(amplitudes, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.shape[-1] < 4:
        raise ValueError("a per-line model needs a run of lines, not a few")
    if domain not in LINE_ROTATION_RAD:
        raise ValueError("domain is 'composite' or 'colour_under'")
    turn = LINE_ROTATION_RAD[domain]
    level = values.mean(axis=-1, keepdims=True)
    amplitude_residual = values - level
    out = {
        "domain": domain,
        "rotation_rad_per_line": turn,
        "rotation_deg_per_line": math.degrees(turn),
        "level": np.squeeze(level),
        "amplitude_residual": amplitude_residual,
        "amplitude_residual_rms": float(np.sqrt(np.mean(amplitude_residual ** 2))),
        "amplitude_complex": hypercomplex.complex_form(
            values, axis=-1, name="burst amplitude per line"),
        "specified_ire": BURST_IRE_PEAK_TO_PEAK,
        "cite": ("ITU-R BT.1700 for the constant amplitude; SMPTE 170M "
                 "clause 8.4 (227.5 cycles a line) for the composite "
                 "rotation, SMPTE 32M 3.9.2.1.5 for the colour-under one"),
        "why": ("one specification gives both the per-line model and the "
                "per-field constant, so the two must agree, and the "
                "residual is what is left when the model is removed"),
    }
    if phases is not None:
        angle = np.asarray(phases, dtype=np.float64)
        if angle.ndim == 1:
            angle = angle[np.newaxis, :]
        lines = np.arange(angle.shape[-1], dtype=np.float64)
        # De-rotate by the specified model, then remove the field's own
        # absolute burst phase as a CIRCULAR MEAN. Anchoring on the first
        # line instead - which this did at first - makes every line's
        # residual carry that one line's noise, and reported a mean of
        # +2.67 degrees on material whose line-to-line residual is +0.07.
        turned = np.angle(np.exp(1j * (angle - turn * lines)))
        offset = np.angle(np.exp(1j * turned).sum(axis=-1, keepdims=True))
        residual = np.angle(np.exp(1j * (turned - offset)))
        # the departure from the SPECIFIED rotation rate is the residual's
        # own slope along the line axis, which is the quantity the model is
        # actually on trial for
        centred = lines - lines.mean()
        slope = ((centred * residual).sum(axis=-1)
                 / max(float((centred * centred).sum()), 1e-300))
        scatter = residual.std(axis=-1)
        stderr = scatter / math.sqrt(float((centred * centred).sum()))
        out["phase_residual"] = residual
        out["phase_offset_rad"] = np.squeeze(offset)
        out["phase_residual_sd_deg"] = float(np.degrees(residual.std()))
        out["phase_residual_seconds"] = float(
            residual.std() / (2.0 * math.pi
                              * composite_channel.NTSC_SUBCARRIER_HZ))
        departure = float(np.degrees(slope.mean()))
        error = float(np.degrees(stderr.mean() / math.sqrt(max(slope.size, 1))))
        # A noiseless series gives a zero error bar, and then the verdict is
        # a comparison of two numbers at the rounding floor. The guard is
        # numerical rather than physical: a departure a trillionth of the
        # specified rotation is arithmetic, not a measurement.
        floor = 1e-12 * math.degrees(turn)
        out["rotation_departure_deg_per_line"] = departure
        out["rotation_departure_error_deg"] = error
        out["rotation_confirmed"] = bool(abs(departure) <= max(3.0 * error, floor))
        out["phase_complex"] = hypercomplex.complex_form(
            residual, axis=-1, name="burst phase residual per line")
        out["relation"] = hypercomplex.relate(
            out["amplitude_complex"], out["phase_complex"])
    return out


def field_constant(amplitudes) -> Dict[str, object]:
    """THE SAME SPECIFICATION READ PER FIELD, and whether the two agree.

    The burst is specified constant, so a field has one burst amplitude and
    the per-line series is that constant plus the instrument. If that were
    the whole story the field-to-field scatter would be the per-line
    scatter divided by the root of the line count, and this reports the
    ratio by which it is not.

    MEASURED, 241 active lines a field:

        decode                 within field   between fields   white would
                                     (IRE)            (IRE)      give (IRE)
        wide 75bars SP              0.5770           0.1014        0.0372
        filt 75bars SP              0.5774           0.1016        0.0372
        wide chromanoise SP         0.5282           0.1142        0.0340
        wide 75bars EP              0.6659           0.2144        0.0429

    So the field-to-field term is 2.73 to 5.00 times what independent line
    noise would produce, on all four. The two readings of one specification
    therefore do NOT agree under a white assumption.

    BUT THE LINES ARE NOT INDEPENDENT, and that is measured, not supposed:
    the per-line series has a lag-one correlation of +0.42 to +0.56 against
    a white bar of 0.0644, six to nine times the bar. A correlated series
    averages more slowly than the root of its length, so the right
    comparison inflates the prediction by the root of (1+rho)/(1-rho) and
    asks what is left:

        decode                  excess over white   over the correlated bar
        wide 75bars SP                       2.73                      1.62
        filt 75bars SP                       2.73                      1.62
        wide chromanoise SP                  3.36                      2.14
        wide 75bars EP                       5.00                      2.67

    The correlation therefore accounts for a little over half the excess on
    every decode and none of them for all of it. Against a bar of 1.42 -
    three times the sampling error of a standard deviation read from 26
    fields, and no chosen number - all four are refused, so a genuine
    field-to-field term remains at 1.6 to 2.7 times what the lines can
    explain. That term is what the specification does not contain, and no
    per-line average removes it.
    """
    values = np.asarray(amplitudes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("field_constant needs fields x lines, at least "
                         "two fields and a run of lines")
    fields, lines = values.shape
    within = float(values.std(axis=1, ddof=1).mean())
    field_means = values.mean(axis=1)
    between = float(field_means.std(ddof=1))
    white = within / math.sqrt(lines)
    centred = values - values.mean(axis=1, keepdims=True)
    numerator = (centred[:, :-1] * centred[:, 1:]).sum(axis=1)
    denominator = (centred * centred).sum(axis=1)
    lag_one = float(np.mean(numerator / np.maximum(denominator, 1e-300)))
    # A correlated series averages more slowly than the root of its length.
    # For a first-order process with lag-one rho the variance of the mean is
    # inflated by (1+rho)/(1-rho), so the standard deviation is inflated by
    # its root, and THAT is the honest prediction to compare against - not
    # the white one, and not a chosen threshold.
    inflation = math.sqrt(max((1.0 + lag_one) / max(1.0 - lag_one, 1e-12), 0.0))
    correlated = white * inflation
    return {
        "fields": int(fields),
        "lines": int(lines),
        "constant": float(values.mean()),
        "per_field": field_means,
        "within_field_sd": within,
        "between_field_sd": between,
        "white_prediction_sd": white,
        "correlated_prediction_sd": correlated,
        "correlation_inflation": inflation,
        "excess": between / white if white > 0 else float("nan"),
        "excess_over_correlated": (between / correlated if correlated > 0
                                   else float("nan")),
        "lag_one": lag_one,
        "white_bar": 1.0 / math.sqrt(lines),
        # The bar is the sampling error of the between-field standard
        # deviation itself, which is estimated from `fields` numbers: a
        # chi distribution's relative spread is 1/sqrt(2(n-1)), so three of
        # those is the bar. It does NOT carry the error in rho, so it is
        # the optimistic bar and a ratio that clears it clears it honestly.
        "verdict_bar": 1.0 + 3.0 / math.sqrt(2.0 * max(fields - 1, 1)),
        "consistent": bool(
            between <= correlated * (1.0 + 3.0
                                     / math.sqrt(2.0 * max(fields - 1, 1)))),
        "field_complex": hypercomplex.complex_form(
            field_means, axis=-1, name="burst constant per field")
        if fields >= 4 else None,
        "specified_ire": BURST_IRE_PEAK_TO_PEAK,
        "why": ("the specification makes the burst constant on both axes, "
                "so a field-to-field scatter larger than the per-line "
                "scatter can average to is a term the specification does "
                "not contain"),
    }
