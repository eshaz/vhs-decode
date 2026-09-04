"""A DIMENSION is the Wiener transfer between two co-registered channels.

Ethan: *"The 'dimensions' are the Weiner transform of each complex pair
(i.e. luma frequency vs. luma amplitude). ... The eliptical curve is the
differentiation between the dimensions. You only 'guess' the curve, you
don't know it."*

An axis is not a dimension. An axis is one measured channel. A dimension is
manufactured by holding TWO channels together on one abscissa and taking
the Wiener transfer between them:

    T(f)      = S_yx(f) / S_xx(f)                  the transfer, complex
    gamma2(f) = |S_yx|^2 / (S_xx S_yy)             how believable it is
    se(f)     = sqrt( (S_yy/S_xx) (1 - gamma2) / n )

`|T|`, `arg T` and the frequency it lives at are Ethan's "amplitude, phase,
and frequency of this differential pair".

WHY THE GRID PROBLEM DISSOLVES. Two channels measured on different axes
would normally have no common abscissa, and an inner product across two
abscissae is memory layout rather than measurement. But this repository
already built the shared grid for another purpose: `residual_channels`
places amplitude, frequency, carrier and time on the SAME 4fsc grid of the
final time base correction, with identical shape. There is an index that
means the same instant in all of them.

THE ONE WORKED EXAMPLE ALREADY IN THE TREE. `head_switch.measured_transfer`
returns a complex `g` in "Hz of demodulated frequency per neper of residual
log-amplitude" - which is luma frequency per luma amplitude, complex, per
band. Ethan's own parenthetical, already implemented, on one pair.

FOUR THINGS THIS FILE GETS RIGHT THAT A FIRST ATTEMPT DID NOT. Each was
found by adversarial review of the construction, and each is a way the
measurement lies rather than a matter of taste.

1. **The ellipse is fitted over the TRANSFERS, not over their differences.**
   Differencing two dimensions, `D_ij = T_i - T_j`, imports the form of the
   nested differential without its content: there the subtraction is
   expected-relation minus actual-relation and both sides are the same kind
   of thing, whereas here there is no synthetic transfer, so the difference
   is bare and CANCELS exactly the shared mechanism the ellipse exists to
   find. Three channels also give only three differences, and they satisfy
   `D_12 + D_23 + D_31 = 0`, so the ensemble arrives rank-deficient by one.

2. **Coherence must be debiased or it certifies its own arithmetic.** With
   `n` averaged segments an entirely uncorrelated pair still returns
   `gamma2 = 1/n`, and at `n = 1` it returns exactly 1. An undebiased
   coherence is a tautology, not evidence.

3. **Never multiply the pair by `W`.** `W` has units of y per x, so applying
   it to both halves of `x + jy` is dimensionally incoherent, and for real
   halves it acts as the same real kernel on each and cannot separate them.
   Carry the coherence-weighted pair instead.

4. **Correlated rows shorten the effective length.** `sphere_floor` rests on
   `E|G_ij|^2 = 1/L`, which assumes independent places. A transfer estimated
   on a smooth grid is correlated between neighbours, so the count that
   belongs in the floor is `L_eff = (sum w)^2 / sum w^2` over the weights.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def _segments(channel: np.ndarray, count: int) -> np.ndarray:
    """A channel cut into `count` equal segments, as rows."""
    values = np.asarray(channel).ravel()
    width = values.size // max(int(count), 1)
    if width < 4:
        raise ValueError("too few samples per segment to estimate a spectrum")
    return values[:width * count].reshape(count, width)


def effective_length(weights: np.ndarray) -> float:
    """The independent-place count a weighted grid is worth.

    `sphere_floor` assumes every place is independent. Weighted or smoothed
    places are not, and the count that belongs in the floor is the
    participation ratio of the weights - the same statistic the ellipse uses
    on its own eigenvalues, applied one level down.
    """
    w = np.abs(np.asarray(weights, dtype=np.float64).ravel())
    total = float(w.sum())
    if not total > 0:
        return 0.0
    return float(total ** 2 / max(float((w ** 2).sum()), 1e-30))


def pair_transfer(x: np.ndarray, y: np.ndarray, segments: int = 8,
                  window: bool = True) -> Dict[str, np.ndarray]:
    """THE DIMENSION: the Wiener transfer of y from x, with its coherence.

    `x` and `y` are two channels co-registered on one abscissa. Returns the
    complex transfer per frequency, the DEBIASED magnitude-squared
    coherence, the standard error, and the effective place count the
    coherence weights are worth.

    The transfer is the H1 estimator `S_yx / S_xx`, which is the Wiener
    filter when the noise is on `y`. It is a ratio, and this project has a
    standing refutation of spectral division - but that refutation is of
    dividing two SPECTRA of the same quantity to recover a response, which
    manufactures phantoms wherever the denominator is small. Here the
    denominator is the input's own power, the numerator is a CROSS
    spectrum, and the coherence states exactly where the estimate is
    trustworthy, which is the thing the refuted construction lacked.
    """
    count = max(int(segments), 2)
    rows_x = _segments(x, count)
    rows_y = _segments(y, count)
    if window:
        taper = np.hanning(rows_x.shape[1])
        rows_x = rows_x * taper
        rows_y = rows_y * taper
    # A COMPLEX PAIR NEEDS THE WHOLE SPECTRUM. `rfft` folds the negative
    # frequencies onto the positive ones, which is exact for a real signal
    # and wrong for a complex one: a carrier at +f and one at -f are
    # different channels, and the burst dimension exists precisely to tell
    # them apart. The real path is unchanged, so no existing measurement
    # moves.
    if np.iscomplexobj(rows_x) or np.iscomplexobj(rows_y):
        fx = np.fft.fft(rows_x.astype(np.complex128), axis=1)
        fy = np.fft.fft(rows_y.astype(np.complex128), axis=1)
    else:
        fx = np.fft.rfft(rows_x, axis=1)
        fy = np.fft.rfft(rows_y, axis=1)
    sxx = np.mean(np.abs(fx) ** 2, axis=0)
    syy = np.mean(np.abs(fy) ** 2, axis=0)
    syx = np.mean(fy * np.conj(fx), axis=0)

    safe = np.maximum(sxx, 1e-30)
    transfer = syx / safe
    raw = np.abs(syx) ** 2 / np.maximum(sxx * syy, 1e-30)
    # DEBIASED. An uncorrelated pair returns 1/n with n segments, and
    # exactly 1 at n = 1; without this the coherence certifies the
    # arithmetic rather than the physics.
    coherence = np.clip((raw - 1.0 / count) / (1.0 - 1.0 / count), 0.0, 1.0)
    error = np.sqrt(np.maximum(syy / safe, 0.0)
                    * (1.0 - coherence) / max(count, 1))
    return {
        "transfer": transfer,
        "coherence": coherence,
        "raw_coherence": raw,
        "standard_error": error,
        "segments": count,
        "input_power": sxx,
        "output_power": syy,
        "effective_length": effective_length(coherence),
        "bias": 1.0 / count,
    }


def coherent_pair(x: np.ndarray, y: np.ndarray, segments: int = 8
                  ) -> np.ndarray:
    """The pair kept only where the two channels agree.

    `gamma(f) Z(f)`, never `W(f) Z(f)`: `W` carries units of y per x, so
    multiplying the whole complex pair by it is dimensionally incoherent,
    and for real halves it is one real kernel acting on both at once, which
    cannot suppress what only one half saw. Weighting by the coherence
    keeps what the halves agree on and attenuates what they do not, which
    is the thing the step is for.
    """
    result = pair_transfer(x, y, segments)
    count = result["segments"]
    rows_x = _segments(x, count)
    rows_y = _segments(y, count)
    # THE SAME COMPLEX BRANCH ITS SIBLING TWELVE LINES ABOVE ALREADY HAS.
    # `pair_transfer` was given this and `coherent_pair` was not, so a complex
    # pair worked in one and raised TypeError in the other - numpy refuses
    # `rfft` on complex input rather than dropping the imaginary part
    # silently, which is the only reason this was a crash and not a wrong
    # answer. A fix that lands in one place and not in its sibling is the
    # pattern behind most of what a sweep for discarded phase turns up.
    if np.iscomplexobj(rows_x) or np.iscomplexobj(rows_y):
        fx = np.fft.fft(rows_x.astype(np.complex128), axis=1)
        fy = np.fft.fft(rows_y.astype(np.complex128), axis=1)
    else:
        fx = np.fft.rfft(rows_x, axis=1)
        fy = np.fft.rfft(rows_y, axis=1)
    pair = fx + 1j * fy
    weight = np.sqrt(result["coherence"])
    return np.mean(pair * weight, axis=0)


# The horizontal sync pulse, as the standard specifies it. SMPTE 170M-2004
# table 2 and ITU-R BT.1700: the pulse is 4.7 us wide at half amplitude, its
# edges rise in 140 ns between the 10 and 90 per cent points, and it sits
# 40 IRE below blanking. These are the SYNTHETIC side of the last dimension
# and they are specified, not fitted.
SYNC_WIDTH_S = 4.7e-6
SYNC_RISE_S = 140e-9
SYNC_DEPTH_IRE = -40.0


def spec_sync(sample_rate_hz: float, length: int,
              width_s: float = SYNC_WIDTH_S,
              rise_s: float = SYNC_RISE_S,
              depth_ire: float = SYNC_DEPTH_IRE) -> np.ndarray:
    """The sync pulse the standard prescribes, on a sample grid.

    A rectangle of the specified width whose edges are raised cosines of the
    specified rise time - the shape a band-limited system produces from an
    ideal step, and the shape every generator is measured against.

    This is the STIMULUS, not a modification: the channel's effect on it is
    what the differential recovers, exactly as `pair_transfer` recovers a
    transfer from an input and an output.
    """
    t = (np.arange(int(length), dtype=np.float64)
         / float(sample_rate_hz))
    centre = t[-1] / 2.0 if len(t) else 0.0
    half = 0.5 * float(width_s)
    # THE STANDARD SPECIFIES A 10-90 TIME, NOT A TRANSITION WIDTH. For a
    # raised cosine spanning `w`, inside = 0.5(1 - cos(pi x)) reaches 0.1 at
    # x = 0.2048 and 0.9 at x = 0.7952, so the 10-90 time is 0.5904 w and
    # the transition must be widened by its reciprocal or the pulse comes
    # out too fast - measured 100 ns against the specified 140.
    ten_ninety = (np.arccos(-0.8) - np.arccos(0.8)) / np.pi
    rise = max(float(rise_s) / ten_ninety, 1.0 / float(sample_rate_hz))

    # One expression rather than two edges combined, because combining them
    # is where a first attempt went wrong: it returned a pulse the width of
    # the whole window with no rise at all. Distance from the pulse's own
    # edge, in units of the rise time, is 1 deep inside and 0 outside, and a
    # raised cosine between.
    offset = np.abs(t - centre)
    across = np.clip((half + 0.5 * rise - offset) / rise, 0.0, 1.0)
    inside = 0.5 * (1.0 - np.cos(np.pi * across))
    return (float(depth_ire) * inside).astype(np.float64)


def sync_dimension(measured_sync: np.ndarray, sample_rate_hz: float,
                   segments: int = 8, **kwargs) -> Dict[str, np.ndarray]:
    """THE FINAL DIMENSION: the sync as specified against the sync as it
    came back.

    Ethan: *"The shape of the final complex differential, sync estimation
    vs. actual shape"*, and *"add that extra dimensional layer to the
    existing flow"*. It needs no new machinery, because it IS a pair: the
    specified pulse is the input channel, the measured pulse is the output,
    and their Wiener transfer is a dimension like any other.

    It is the best-conditioned dimension available, for three reasons no
    other pair has together. Its synthetic side is SPECIFIED rather than
    fitted, so nothing is estimated from the data it is then judged on. It
    is CONTENT-FREE, so it satisfies the standing constraint that correction
    derives from sync and reserved intervals and never from picture. And the
    sync pulse exists on BOTH sides of the demodulator, so a residual
    measured on one is a prediction about the other and the difference
    belongs to the demodulator alone.
    """
    measured = np.asarray(measured_sync, dtype=np.float64).ravel()
    ideal = spec_sync(sample_rate_hz, measured.size, **kwargs)
    result = pair_transfer(ideal, measured, segments)
    result["ideal"] = ideal
    result["why"] = ("the specified pulse against the measured one; the "
                     "synthetic side is a standard, not a fit")
    return result


def dimensions(channels: Dict[str, np.ndarray], segments: int = 8,
               pairs: Optional[Tuple[Tuple[str, str], ...]] = None
               ) -> Dict[str, Dict[str, np.ndarray]]:
    """Every dimension the channels support, as an ensemble for the ellipse.

    With three channels there are three ordered pairs worth taking, and
    each is one dimension. What is returned feeds `ellipsoid` DIRECTLY -
    the transfers themselves, not their differences, for the reason in the
    module docstring.
    """
    names = list(channels)
    if pairs is None:
        pairs = tuple((a, b) for index, a in enumerate(names)
                      for b in names[index + 1:])
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for source, target in pairs:
        if source not in channels or target not in channels:
            continue
        try:
            result = pair_transfer(channels[source], channels[target],
                                   segments)
        except ValueError:
            continue
        value = np.log(np.maximum(np.abs(result["transfer"]), 1e-30)) \
            + 1j * np.unwrap(np.angle(result["transfer"]))
        # weighted by how believable each place is, which is what makes the
        # ellipse fitted over evidence rather than over arithmetic
        value = value * np.sqrt(result["coherence"])
        live = np.isfinite(value)
        out[f"{target} from {source}"] = {
            "frequency": np.where(live, value, 0.0),
            "coherence": result["coherence"],
            "effective_length": result["effective_length"],
        }
    return out


# --------------------------------------------------------------------------
# The colour burst, which is specified in all three dimensions at once
# --------------------------------------------------------------------------
#
# Ethan: *"Additionally, the color burst can be modeled using the spec. That
# satisfies all three dimensions, and adds another set of residuals that may
# relate to the color under, and eventually the color subcarrier."*
#
# It does satisfy all three, and it is the ONLY element of the signal that
# does. The sync pulse pins amplitude and time and says nothing about phase,
# because it carries no carrier. The burst pins:
#
#     FREQUENCY   f_sc = 455/2 x f_H, an exact rational multiple of the line
#                 rate, so it is not a measured number at all
#     AMPLITUDE   40 IRE peak to peak, 285.7 mV +- 2 per cent
#     PHASE       SC/H 0 degrees +- 5, which is what makes it the only
#                 ABSOLUTE phase reference anywhere in the signal
#
# and its position and duration are specified in whole subcarrier cycles -
# 19 cycles from the sync datum, 9 cycles long - so the three are tied to
# each other rather than stated independently.
#
# SMPTE 170M-2004 and ITU-R BT.1700 for the levels and timings; the TSG-130A
# generator specification in docs/VERTICAL_INTERVAL_COLLAPSE.md gives the
# same values with the generator's own tolerances, which is what the captures
# were made against.
NTSC_LINE_RATE_HZ = 30000.0 / 1001.0 * 525.0
SUBCARRIER_RATIO = 455.0 / 2.0
BURST_CYCLES = 9.0
BURST_DELAY_CYCLES = 19.0
BURST_AMPLITUDE_IRE = 40.0
BURST_ENVELOPE_RISE_S = 300e-9
# The recorded burst is NOT at the subcarrier. VHS heterodynes the chroma
# down before it reaches the tape, so what the medium holds is the burst at
# the colour-under carrier - which is why this dimension probes the
# colour-under path directly and the subcarrier only through it.
# DERIVED, NOT WRITTEN DOWN. It was a literal 629371.0 and that was wrong by
# 0.3706 Hz - negligible across one burst and 2.2 DEGREES across a field,
# which is a real error in the one dimension this module exists to supply.
# The colour-under carrier is EXACTLY forty line periods (SMPTE 32M), so it
# is a ratio and never a measured constant:
#
#     f_cu = 40 f_H = 40 x 525 x 30000/1001 = 629370.629371... Hz
#     f_cu / f_sc = 40 / (455/2) = 16/91, exactly
#
# and the same identity is what `correction_export.clock_relation` reads the
# capture crystal against.
COLOUR_UNDER_MULTIPLE = 40.0
NTSC_COLOUR_UNDER_HZ = COLOUR_UNDER_MULTIPLE * NTSC_LINE_RATE_HZ

# --------------------------------------------------------------------------
# 625/50, from ITU-R BT.1700 Part B Tables 1-3.
#
# THE BURST IS NOT THE SAME OBJECT IN EVERY SYSTEM, and two of the
# differences would silently corrupt a model ported from NTSC:
#
#   BURST AMPLITUDE IS NOT ALWAYS THE SYNC AMPLITUDE. In NTSC they are equal
#   at 40 IRE / 285.7 mV, and it is tempting to treat that as a law. It is
#   not: 625 PAL gives burst 300 +- 30 mV against a sync amplitude of 300 mV,
#   a ratio of exactly 1.000, but 525 PAL gives burst 316 to 317 mV against a
#   sync amplitude of 286 mV - a ratio of 1.107. A level model that carries
#   NTSC's "burst equals sync" into PAL-M is 10.7 per cent low.
#
#   THE CYCLE COUNT IS NOT A PROPERTY OF PAL. It is 10 +- 1 for standard
#   625 PAL and 9 +- 1 for 525 PAL and for Argentina's 625 variant, with the
#   durations following - 2.25, 2.52 and 2.51 us respectively.
#
# One cross-standard confirmation worth recording: 525 PAL's burst starts
# 5.3 us after the datum, which is 18.95 subcarrier cycles - within 0.05 of
# the 19 whole cycles modelled for NTSC above, so that constant is attested
# twice rather than once.
PAL_LINE_RATE_HZ = 15625.0
# f_sc = (1135/4 + 1/625) f_H, exactly 4433618.75 Hz +- 1 Hz
PAL_SUBCARRIER_RATIO = 1135.0 / 4.0 + 1.0 / 625.0
PAL_BURST_CYCLES = 10.0
PAL_BURST_START_US = 5.6                 # +- 0.1 us from the sync datum
PAL_BURST_AMPLITUDE_MV = 300.0           # +- 30, equal to the sync amplitude
PAL_SYNC_AMPLITUDE_MV = 300.0
# The burst alternates +-135 degrees from the E'U axis line to line, which is
# the whole of PAL's phase-alternation and is why an NTSC model that carries
# ONE burst phase cannot represent it.
PAL_BURST_PHASE_DEG = 135.0


def pal_subcarrier_hz(line_rate_hz: float = PAL_LINE_RATE_HZ) -> float:
    """`f_sc = (1135/4 + 1/625) f_H` - 4433618.75 Hz, exactly."""
    return float(PAL_SUBCARRIER_RATIO * float(line_rate_hz))


def spec_burst_pal(sample_rate_hz: float, length: int, line: int = 0,
                   amplitude_mv: float = PAL_BURST_AMPLITUDE_MV,
                   rise_s: float = BURST_ENVELOPE_RISE_S) -> np.ndarray:
    """The 625-line burst, WITH the line-to-line phase alternation.

    ITU-R BT.1700 Part B: ten cycles at `+-135` degrees from the E'U axis,
    starting 5.6 us after the sync datum. `line` selects the alternation -
    the sign of the 135 degrees flips every line, and that alternation is the
    thing an NTSC-derived model has no way to express.

    Returned complex for the same reason `spec_burst` is: the phase is the
    dimension this exists to supply.
    """
    rate = float(sample_rate_hz)
    carrier = pal_subcarrier_hz()
    duration = PAL_BURST_CYCLES / carrier
    start = PAL_BURST_START_US * 1e-6
    t = np.arange(int(length), dtype=np.float64) / rate
    rising = np.clip((t - start) / max(rise_s, 1e-12), 0.0, 1.0)
    falling = np.clip((start + duration - t) / max(rise_s, 1e-12), 0.0, 1.0)
    envelope = (0.5 - 0.5 * np.cos(np.pi * rising)) * \
               (0.5 - 0.5 * np.cos(np.pi * falling))
    swing = -1.0 if int(line) % 2 else 1.0
    phase = np.deg2rad(swing * PAL_BURST_PHASE_DEG)
    return (0.5 * float(amplitude_mv) * envelope
            * np.exp(1j * (2.0 * np.pi * carrier * t + phase)))


def subcarrier_hz(line_rate_hz: float = NTSC_LINE_RATE_HZ) -> float:
    """`f_sc = 455/2 x f_H`, exactly. A ratio, not a measurement."""
    return float(SUBCARRIER_RATIO * float(line_rate_hz))


def spec_burst(sample_rate_hz: float, length: int,
               line_rate_hz: float = NTSC_LINE_RATE_HZ,
               amplitude_ire: float = BURST_AMPLITUDE_IRE,
               phase_rad: float = np.pi,
               carrier_hz: Optional[float] = None,
               rise_s: float = BURST_ENVELOPE_RISE_S) -> np.ndarray:
    """THE BURST THE STANDARD PRESCRIBES, as a complex-valued stimulus.

    Nine cycles of the subcarrier under a raised-cosine envelope, at the
    specified amplitude and at the specified phase - 180 degrees from the
    +U axis for NTSC, which is what `phase_rad` defaults to.

    Returned COMPLEX, carrying the carrier's own phase, because that is the
    dimension the burst exists to supply and a magnitude-only stimulus would
    throw away the only absolute phase reference in the signal.

    `carrier_hz` overrides the subcarrier with the colour-under carrier where
    the burst is being read off the tape rather than off the baseband.
    """
    rate = float(sample_rate_hz)
    carrier = float(carrier_hz) if carrier_hz else subcarrier_hz(line_rate_hz)
    duration = BURST_CYCLES / max(carrier, 1.0)
    t = np.arange(int(length), dtype=np.float64) / rate
    start = BURST_DELAY_CYCLES / max(carrier, 1.0)
    rising = np.clip((t - start) / max(rise_s, 1e-12), 0.0, 1.0)
    falling = np.clip((start + duration - t) / max(rise_s, 1e-12), 0.0, 1.0)
    envelope = (0.5 - 0.5 * np.cos(np.pi * rising)) * \
               (0.5 - 0.5 * np.cos(np.pi * falling))
    return (0.5 * float(amplitude_ire) * envelope
            * np.exp(1j * (2.0 * np.pi * carrier * t + float(phase_rad))))


def burst_dimension(measured_burst: np.ndarray, sample_rate_hz: float,
                    segments: int = 8, **kwargs) -> Dict[str, np.ndarray]:
    """THE BURST AS SPECIFIED AGAINST THE BURST AS IT CAME BACK.

    The same pair the sync dimension forms, and it needs no new machinery
    for the same reason: the specified burst is the input channel, the
    measured burst is the output, and their Wiener transfer is a dimension.

    WHAT IT ADDS THAT THE SYNC DIMENSION CANNOT. The sync pulse is real and
    carries no carrier, so its transfer's phase is only the channel's group
    delay measured against an arbitrary datum. The burst carries a carrier
    at a specified phase, so its transfer's phase is ABSOLUTE - the one
    place in the signal where a measured phase can be compared with a
    specified one rather than with another measurement.

    AND IT LANDS ON THE COLOUR-UNDER PATH, NOT THE SUBCARRIER. What the tape
    holds is the burst heterodyned to 629 kHz, so this dimension measures the
    colour-under channel first and the subcarrier only through the
    heterodyne that produced it. That ordering is the useful part: a residual
    here is attributable to the downconversion and the medium before anything
    is said about the chroma the viewer sees.

    The standing constraint is satisfied for the same reason the sync
    dimension satisfies it - the burst is in the blanking interval and
    carries no picture - with one caveat that is not optional: the burst sits
    in the BACK PORCH, and the back porch's own residual has to be taken out
    first or it is read as burst phase.
    """
    measured = np.asarray(measured_burst)
    if not np.iscomplexobj(measured):
        measured = measured.astype(np.complex128)
    reference = spec_burst(sample_rate_hz, measured.size, **kwargs)
    transfer = pair_transfer(reference, measured, segments=segments)
    transfer["reference"] = reference
    transfer["absolute_phase"] = True
    transfer["why"] = ("the only pair whose synthetic side specifies a "
                       "PHASE, so its transfer's phase is absolute rather "
                       "than relative to another measurement")
    return transfer
