"""The sync pulse's SHAPE as one component carried on all three axes.

Ethan, having had to say it twice: "the luma one about noise and it's
relationship to the sync pulse shape. the shape is a component with all
parts, amplitude, frequency, time".

And earlier, the half that says where the noise comes from: "The luma and
chroma are band limited, so the data outside this bandlimited area is
noise, that can use as the differential".

WHAT THIS CORRECTS. This lane has been treating the pulse shape as an
amplitude profile that happens to have a spectrum - fitting a relaxation
here, a ring there, each a scalar amplitude with a time constant. That is
one axis with the other two implied. The statement is that the shape is
ONE component present on amplitude, frequency and time simultaneously,
which is the same three-axis model `docs/THE_ALGORITHM.md` already states
for everything else, applied to the pulse itself.

THE BAND IS WHAT SEPARATES SHAPE FROM NOISE, and it is a FORMAT CONSTANT
rather than a fitted knee. The luma is band limited, so:

    what lies INSIDE the band is the shape
    what lies OUTSIDE it is noise, BY CONSTRUCTION

That is not a threshold anyone chooses and not a signal-to-noise judgment.
It follows from the format, and it makes the noise a MEASUREMENT rather
than a leftover - which is Ethan's "can use as the differential".

HOW THE THREE AXES COME OUT OF ONE FIT. The Laplace eigenbasis
(`hilbert_space_gp`) is the natural frame for this, because its modes are
indexed by frequency and truncating it at the format's band limit IS the
split above:

    FREQUENCY  the mode frequencies, fixed by the segment and the format,
               nothing fitted - mode j sits at j/(4L) cycles per sample
    AMPLITUDE  the coefficient on each mode, which is the shape's spectrum
    TIME       the group delay implied by that spectrum, through Bode's
               relation, plus where the shape's energy sits in the segment

They are three readings of ONE fitted object, not three fits. That is the
part the previous treatment lost: a relaxation fitted for its amplitude
and a ring fitted for its frequency were two objects describing one shape,
and they disagreed because nothing made them share a frame.

MEASURED, 2026-09-06: held out on independent lines, this basis truncated
at the 3.0 MHz VHS luma band reaches 1.4x the two-measurement floor on
both tapes and both heads, where the relaxation-plus-ring enumeration sits
at 5.4 to 12.4x. The held-out error falls monotonically to that cap and
flatlines past it, so the band constant is confirmed by the data rather
than imposed on it.

WHERE THE TIME AXIS ABOVE FALLS SHORT, and what was added on 2026-09-06.
`shape_components` reports its TIME axis as `group_delay_s`, and that
reading is Bode's relation applied to the shape's own coefficient
magnitudes: it is the delay a MINIMUM-PHASE channel of that magnitude
would have. It is therefore a function of the amplitude axis and carries
nothing the amplitude does not, so the three axes as first written were
two. `transit_delay` and `edge_pair` are the third axis MEASURED: the
delay one shape carries against another, taken from the phase of the
response between them, split into the part the magnitude already fixes
and the part it does not, and checked on the pulse's two transients
separately because a delay moves them together and a change of width
moves them apart.
"""

from typing import Dict, Optional

import numpy as np

from vhsdecode.models import hilbert_space_gp as hsgp
from vhsdecode.models import hypercomplex as hc
from vhsdecode.models import pair_dimension as pdim

__all__ = ["shape_components", "differential", "locate_transients",
           "transit_delay", "edge_pair", "VHS_LUMA_BAND_HZ",
           "SVHS_LUMA_BAND_HZ", "UMATIC_LUMA_BAND_HZ", "BROADCAST_LUMA_BAND_HZ"]

VHS_LUMA_BAND_HZ = 3.0e6
SVHS_LUMA_BAND_HZ = 5.0e6
UMATIC_LUMA_BAND_HZ = 3.5e6
BROADCAST_LUMA_BAND_HZ = 4.2e6
"""The band limits that decide where shape ends and noise begins.

FORMAT CONSTANTS, and the whole point is that they are not fitted. The
broadcast figure is the channel's, not the signal's - BTS-3 states no
luminance bandwidth of its own, it states the I and Q channels (2.4.2)
and the chrominance sidebands (2.4.8), so 4.2 MHz belongs to the
transmission channel and is quoted as such.
"""


def shape_components(profile, sample_rate_hz: float, band_limit_hz: float,
                     boundary_factor: Optional[float] = None
                     ) -> Dict[str, object]:
    """The shape on all three axes, and the noise outside the band.

    `profile` is the accumulated pulse shape - one segment, already
    aligned, in whatever amplitude unit the caller works in (IRE here).

    Returns the three axes as readings of ONE fit, plus the out-of-band
    remainder as a measurement in its own right.
    """
    values = np.asarray(profile, dtype=np.float64).ravel()
    samples = len(values)
    if samples < 8:
        raise ValueError("a shape needs more than a handful of samples")
    # the FITTING factor, not the kernel-reconstruction one: at 52 samples
    # and 43 modes the latter conditions at 3.5e15 and rcond silently
    # discards three directions, for an identical held-out answer
    factor = (hsgp.FITTING_BOUNDARY_FACTOR if boundary_factor is None
              else float(boundary_factor))
    frame = hsgp.from_format(samples, sample_rate_hz, band_limit_hz, factor)
    positions = frame["positions"]
    basis = hsgp.eigenfunctions(positions, frame["modes"], frame["half_width"])

    # THE FIT IS PLAIN LEAST SQUARES, and that is a measured choice rather
    # than a simplification. A smoothness prior over these modes was tried
    # and earns nothing: the evidence drives its lengthscale to about 0.02
    # samples - a flat density, i.e. no prior - and plain least squares
    # scores as well or better on held-out lines. The BASIS does the work;
    # the band limit is the only regularisation, and it is a constant.
    centre = float(np.mean(values))
    coefficients, *_ = np.linalg.lstsq(basis, values - centre, rcond=None)
    in_band = basis @ coefficients + centre
    out_of_band = values - in_band

    frequencies = frame["mode_frequencies_hz"]
    amplitude = np.abs(coefficients)
    # the group delay Bode's relation implies for this magnitude. The band
    # is by construction NOT wide enough to contain the roll-off - it IS
    # the roll-off - so this is a within-band reading and the truncation
    # guard is left off deliberately rather than by oversight.
    delay = hsgp.group_delay_s(np.maximum(amplitude, 1e-12), frequencies)

    energy = amplitude ** 2
    total = float(energy.sum()) or 1.0
    centroid_hz = float(np.sum(frequencies * energy) / total)

    # THE POSITION COMES FROM THE DFT PHASE, and it has to. The
    # eigenfunctions are sines pinned at the boundary, so this basis has NO
    # SHIFT THEOREM - a translated shape is a different set of
    # coefficients, not the same set rotated - and any position read off
    # the coefficients moves unpredictably. Measured on a shape shifted
    # three, six and ten samples right, an amplitude centroid in this basis
    # moved LEFT, monotonically.
    #
    # The DFT does have the theorem: a shift of k samples is exactly a
    # phase ramp of -2 pi f k, so the slope of the unwrapped phase IS the
    # position, weighted by magnitude squared so the bins carrying the
    # shape decide it. The same rule already stands for the edge position
    # in the collapsed-model test, for the same reason.
    spectrum = np.fft.rfft(values - centre)
    bins = np.fft.rfftfreq(samples)
    phase = np.unwrap(np.angle(spectrum))
    weights = np.abs(spectrum) ** 2
    weights[0] = 0.0                     # DC carries no position
    if float(weights.sum()) > 0:
        mean_f = float(np.sum(weights * bins) / weights.sum())
        mean_p = float(np.sum(weights * phase) / weights.sum())
        spread = float(np.sum(weights * (bins - mean_f) ** 2))
        slope = (float(np.sum(weights * (bins - mean_f) * (phase - mean_p)))
                 / spread) if spread > 0 else 0.0
        position_samples = -slope / (2.0 * np.pi)
    else:
        position_samples = float("nan")

    return {
        # ---- the three axes, all read off the same fit ----
        "frequency_hz": frequencies,
        "amplitude": amplitude,
        "group_delay_s": delay,
        # ---- scalar summaries of each axis ----
        "amplitude_rms": float(np.sqrt(np.mean((in_band - centre) ** 2))),
        "centroid_hz": centroid_hz,
        "position_samples": position_samples,
        "mean_group_delay_s": float(np.nanmean(delay)),
        # ---- the split the format decides ----
        "in_band": in_band,
        "out_of_band": out_of_band,
        "noise_rms": float(np.sqrt(np.mean(out_of_band ** 2))),
        "band_limit_hz": float(band_limit_hz),
        "modes": int(frame["modes"]),
        "half_width": float(frame["half_width"]),
        "coefficients": coefficients,
        "level": centre,
        # the effective number of modes actually carrying the shape, which
        # is the honest count of its degrees of freedom rather than the
        # number the basis offers
        "effective_rank": float(total ** 2 / float(np.sum(energy ** 2)))
        if float(np.sum(energy ** 2)) > 0 else 0.0,
    }


def differential(measured: Dict[str, object],
                 reference: Dict[str, object]) -> Dict[str, object]:
    """Measured against expected, on all three axes at once.

    Ethan's process is a DIFFERENCE, never a ratio - spectral division is
    what manufactured phantoms in this tree twice - so every axis here is
    subtractive.

    The out-of-band term is carried through as a difference too, which is
    the second half of his statement: the noise is not discarded, it is
    itself a channel of the differential. Two shapes whose in-band parts
    agree and whose out-of-band parts do not have a real difference, and
    it is one a band-limited comparison would report as none.
    """
    for name, side in (("measured", measured), ("reference", reference)):
        if int(side["modes"]) != int(measured["modes"]):
            raise ValueError(
                "both sides must be measured on the same basis - %s carries "
                "%d modes against %d, and a comparison is only a comparison "
                "when both sides are computed over the same set"
                % (name, int(side["modes"]), int(measured["modes"])))
    return {
        "frequency_hz": np.asarray(measured["frequency_hz"]),
        "amplitude": np.asarray(measured["amplitude"])
        - np.asarray(reference["amplitude"]),
        "group_delay_s": np.asarray(measured["group_delay_s"])
        - np.asarray(reference["group_delay_s"]),
        "time_s": float(measured["mean_group_delay_s"])
        - float(reference["mean_group_delay_s"]),
        "position_samples": float(measured["position_samples"])
        - float(reference["position_samples"]),
        "noise_rms": float(measured["noise_rms"]) - float(reference["noise_rms"]),
        "out_of_band": np.asarray(measured["out_of_band"])
        - np.asarray(reference["out_of_band"]),
        "residual": np.asarray(measured["in_band"])
        - np.asarray(reference["in_band"]),
    }


# --------------------------------------------------------------------------
# THE TIME AXIS AS A MEASUREMENT RATHER THAN A CONSEQUENCE OF THE AMPLITUDE
#
# `shape_components` reports a `group_delay_s`, and that reading is Bode's
# relation applied to the shape's own magnitude - it is the delay a
# MINIMUM-PHASE channel of that magnitude would have. It therefore carries
# no information the amplitude axis does not already carry, and a module
# claiming three axes on that basis has two.
#
# What follows is the third axis measured instead of derived: the delay one
# shape carries against another, taken from the PHASE of the response
# between them, and then split into the part the magnitude already fixes and
# the part it does not. The second is the only half that is new.
# --------------------------------------------------------------------------


def locate_transients(profile, sample_rate_hz: float,
                      width_s: float = pdim.SYNC_WIDTH_S,
                      tolerance: float = 0.25) -> Dict[str, int]:
    """The pulse's two transients, and the point that separates them.

    The steepest fall and the steepest rise, which for a sync pulse are its
    leading and trailing edges. Their separation is checked against SMPTE
    170M table 2's 4.7 microsecond sync width - transcribed once, in
    `pair_dimension.SYNC_WIDTH_S` - so a segment that does not contain a
    whole pulse is refused rather than measured.

    `tolerance` is the fractional width departure allowed. It is an
    ASSUMPTION and is labelled one: the standard states a 4.7 microsecond
    pulse and states no tolerance for the recovered width after a tape
    channel, so a quarter is a guard against a mis-framed segment and not a
    reading of anything.
    """
    values = np.asarray(profile, dtype=np.float64).ravel()
    if len(values) < 8:
        raise ValueError("a pulse needs more than a handful of samples")
    slope = np.diff(values)
    fall = int(np.argmin(slope))
    rise = int(np.argmax(slope))
    if not rise > fall:
        raise ValueError(
            "the steepest rise (sample %d) does not follow the steepest fall "
            "(sample %d), so this segment does not hold one whole pulse"
            % (rise, fall))
    measured_s = (rise - fall) / float(sample_rate_hz)
    if abs(measured_s - width_s) > tolerance * width_s:
        raise ValueError(
            "the transients are %.3f us apart against the specified %.3f us, "
            "outside the %.0f per cent guard - reframe the segment"
            % (measured_s * 1e6, width_s * 1e6, tolerance * 100))
    return {"fall": fall, "rise": rise, "midpoint": (fall + rise) // 2,
            "width_s": measured_s, "specified_width_s": float(width_s)}


def _band_response(measured, reference, sample_rate_hz: float,
                   band_limit_hz: float):
    """The complex response between two shapes, and the bins that carry it.

    MADE PERIODIC BY ITS ENDS, NOT BY A WINDOW, and that is a measured
    choice rather than a preference. A transform assumes its segment
    repeats; a segment cut out of a line does not, and the usual remedy is
    a taper. A taper cannot be used here: multiplying both sides by the
    same window does not commute with a shift, so it pulls the delay
    towards zero by a fixed fraction of itself. Measured on planted delays
    of 2, 5, 12, 20 and 50 ns in a 130-sample segment, a Hann window read
    -4.9, -5.0, -5.2, -5.3 and -5.0 per cent low - a scale error, not a
    noise - and a Tukey window at taper fractions 0.5, 0.25 and 0.1 read
    -4.3, -6.3 and -14.4 per cent, so the bias grows as the window narrows
    and no setting escapes it.

    What is subtracted instead is the STRAIGHT LINE JOINING THE TWO ENDS.
    That makes the segment periodic exactly, it is what a step between two
    flat levels is, and because both ends sit in flat signal a shift of the
    transient between them leaves it unchanged. Measured on the pulse's two
    half-segments, where one end is blanking and the other is the sync tip
    and the step is therefore the whole 40 IRE, the same planted delays of
    2, 5, 12 and 20 ns came back as 2.02, 5.05, 12.09 and 20.11 ns on the
    leading transient and 1.99, 4.98, 11.93 and 19.86 on the trailing one.
    Left raw the same halves read 0.86, 2.16, 5.15 and 8.54 - a 57 per cent
    under-read - and tapered they read 1.55, 3.87, 9.28 and 15.45, low and
    unequal between the two transients, which would have put a false width
    change into every pair.

    The size of what was removed is reported as `end_step`, because a
    segment framed in blanking should have almost none of it and one framed
    anywhere else is being measured on an assumption the caller should see.

    WHICH BINS COUNT IS THE FORMAT'S DECISION, not a threshold anyone
    chooses - the same statement this module already makes about the shape
    and its noise. The luma is band limited, so the power above the limit is
    noise by construction; a bin below that level carries no shape, its
    phase is the noise's, and it is excluded.
    """
    m = np.asarray(measured, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    if len(m) != len(r):
        raise ValueError("both shapes must be sampled over the same segment")
    join = float(np.sqrt(np.mean(np.asarray(
        [m[0] - m[-1], r[0] - r[-1]], dtype=np.float64) ** 2)))
    ramp = np.linspace(0.0, 1.0, len(m))
    m = m - (m[0] + (m[-1] - m[0]) * ramp)
    r = r - (r[0] + (r[-1] - r[0]) * ramp)
    spectrum_m = np.fft.rfft(m - m.mean())
    spectrum_r = np.fft.rfft(r - r.mean())
    frequencies = np.fft.rfftfreq(len(m), 1.0 / float(sample_rate_hz))
    outside = frequencies > float(band_limit_hz)
    if not outside.any():
        raise ValueError(
            "the segment's Nyquist is inside the %.2f MHz band limit, so "
            "there is no out-of-band noise to measure the floor with"
            % (band_limit_hz / 1e6))
    floor = float(np.mean(np.abs(spectrum_r[outside]) ** 2))
    power = np.abs(spectrum_r) ** 2
    carried = (frequencies > 0) & (~outside) & (power > floor)
    if carried.sum() < 4:
        raise ValueError("fewer than four bins carry the shape")
    # how large the straight line removed above was, against the segment's
    # own noise. For a length-N real transform of noise of amplitude sigma
    # the expected out-of-band power is N sigma^2 / 2, so the floor the
    # segment already measured inverts to the amplitude to compare with.
    noise_amplitude = float(np.sqrt(2.0 * floor / len(r)))
    ends_match = bool(join <= max(4.0 * noise_amplitude, 1e-12))
    response = np.zeros_like(spectrum_m)
    response[carried] = (spectrum_m[carried] * np.conj(spectrum_r[carried])
                         / power[carried])
    return (response, frequencies, carried, power, floor, ends_match, join)


def _weighted_slope(frequencies, angle, weight) -> float:
    """The delay a small residual phase implies, weighted, through zero."""
    denominator = float(np.sum(weight * frequencies ** 2))
    if not denominator > 0:
        return 0.0
    return -float(np.sum(weight * frequencies * angle)) / (
        2.0 * np.pi * denominator)


def transit_delay(measured, reference, sample_rate_hz: float,
                  band_limit_hz: float,
                  maximum_lag_s: Optional[float] = None) -> Dict[str, object]:
    """THE DELAY ONE SHAPE CARRIES AGAINST ANOTHER, and what fixes it.

    The response is formed as a cross spectrum divided by the reference's
    own power - a DIFFERENCE of phases rather than a division of two
    measured numbers - so a reference bin near a null contributes its
    weight, which is near zero, instead of a large ratio.

    THE DELAY IS FOUND WITHOUT UNWRAPPING ANYTHING. A matched filter over a
    bounded lag grid takes the coarse value and a weighted slope through the
    origin refines it, because after the coarse lag is removed the residual
    phase is small everywhere. Unwrapping across a pulse's spectral nulls is
    what made an earlier attempt at this measurement read a scatter of 417
    nanoseconds where the field-to-field spread was 130 - the estimator's
    noise larger than the effect, from the nulls alone.

    `maximum_lag_s` defaults to SMPTE 170M table 2's 140 nanosecond sync
    rise time, taken from `pair_dimension.SYNC_RISE_S`: a displacement
    larger than the edge's own rise is not a perturbation of this shape, and
    within that bound no bin below the band limit can wrap.

    THE SPLIT THAT MATTERS. The delay is reported three ways:

      `delay_s`                 what the phase says
      `minimum_phase_delay_s`   what the response's own MAGNITUDE says it
                                must be, by Bode's relation
      `excess_delay_s`          the difference, which is the only part that
                                is a separate term

    AND WHETHER THE ONE NUMBER IS A DELAY AT ALL is reported beside it. A
    delay is a phase LINEAR in frequency; whatever survives the linear fit
    is not one. `equivalent_residual_delay_s` is the weighted root mean
    square of that survivor expressed as a time at the band's weighted
    centre, and `is_a_delay` is the honest comparison of the two. A caller
    that quotes `delay_s` while `is_a_delay` is False is quoting a
    projection of a dispersive response onto a straight line.

    The per-bin `group_delay_s` is returned for inspection and is NOT a
    reading to quote: differentiating a measured phase bin by bin amplifies
    the noise, and on the tapes below it swings over hundreds of
    nanoseconds where the fitted delay is tens.

    MEASURED, 2026-09-06, by this function on the head A against head B
    mean sync profile of three decoded tapes in /tmp/claude-1000 - fields
    pooled by isFirstField, lines 25 to 239, the segment 130 samples from
    activeVideoEnd, 27 carried bins below 3 MHz:

        tape                   delay   min-phase     excess  residual  delay?
        dod_wide_75bars_SP    -4.823     -13.414     +8.591     6.972   False
        dod_wide_75bars_EP   -16.686      +4.833    -21.519    11.693   True
        dod_wide_home         +3.350      -3.644     +6.994     7.091   False

    all in nanoseconds. TWO THINGS FOLLOW, and both are the point of the
    function.

    First, the magnitude does not predict the phase. It disagrees in SIGN
    on the EP and home recordings and over-predicts by a factor of 2.8 on
    the SP one, and on all three the excess is the larger part. So the
    head-to-head delay is not the minimum-phase partner of the head-to-head
    amplitude difference: a model that derives one from the other - which is
    what this module's own `group_delay_s` does, and what
    `head_model.group_delay_s` does - gets its size wrong at best and its
    sign wrong at worst.

    Second, only the EP recording carries a delay large enough to survive
    its own residual. On the other two the scalar is smaller than the
    dispersion it was drawn from, and calling either a per-head delay would
    be a mistake the number itself cannot reveal.

    ON THESE SEGMENTS `ends_match` READS FALSE, and it should: the 130
    samples from activeVideoEnd end inside the back porch while the
    ringing is still running, so the straight line removed is 2.3, 3.3 and
    6.5 IRE against noise amplitudes of 0.060, 0.117 and 0.242 IRE. The
    joining line is doing real work there rather than none, which is why
    its size is reported rather than assumed away.
    """
    if maximum_lag_s is None:
        maximum_lag_s = pdim.SYNC_RISE_S
    (response, frequencies, carried, power, floor, ends_match,
     join) = _band_response(measured, reference, sample_rate_hz,
                            band_limit_hz)
    weight = np.where(carried, power, 0.0)
    f = frequencies[carried]
    w = weight[carried]
    h = response[carried]

    lags = np.linspace(-float(maximum_lag_s), float(maximum_lag_s), 4097)
    score = np.abs(np.exp(2j * np.pi * np.outer(lags, f)) @ (h * w))
    coarse = float(lags[int(np.argmax(score))])
    residual = np.angle(h * np.exp(2j * np.pi * f * coarse))
    delay = coarse + _weighted_slope(f, residual, w)

    # what the magnitude alone fixes. The cepstrum is global, so the bins
    # that carry no shape are marked rather than filled with a number they
    # never had - the `valid` contract `hypercomplex.minimum_phase`
    # documents, and the reason it exists.
    log_magnitude = np.zeros_like(frequencies)
    log_magnitude[carried] = np.log(np.maximum(np.abs(response[carried]),
                                               1e-12))
    minimum = hc.minimum_phase(log_magnitude, valid=carried)
    minimum_delay = _weighted_slope(f, minimum[carried], w)

    # WHETHER THE ONE NUMBER IS A DELAY AT ALL. A delay is a phase linear in
    # frequency; what is left after the linear term is removed is not. The
    # residual is small by construction, so no unwrapping is needed to
    # differentiate it, and the per-bin group delay comes out of the same
    # quantity rather than from a second computation.
    left = np.angle(h * np.exp(2j * np.pi * f * delay))
    step = float(np.mean(np.diff(f))) if len(f) > 1 else 1.0
    group = delay - np.gradient(left, 2.0 * np.pi * step)
    centre = float(np.sum(w * f) / max(float(np.sum(w)), 1e-300))
    spread = float(np.sqrt(np.sum(w * left ** 2) / max(float(np.sum(w)),
                                                       1e-300)))
    residual_delay = spread / (2.0 * np.pi * max(centre, 1e-300))

    group_delay = np.full_like(frequencies, np.nan)
    group_delay[carried] = group
    return {
        # the response itself, COMPLEX, because the phase is the whole point
        "response": response,
        "frequency_hz": frequencies,
        "carried": carried,
        "weight": weight,
        "noise_floor_power": floor,
        # how far the segment was from periodic before the joining line
        # was removed, and whether that was within its own noise
        "ends_match": ends_match,
        "end_step": join,
        # the three readings of the time axis
        "delay_s": float(delay),
        "minimum_phase_delay_s": float(minimum_delay),
        "excess_delay_s": float(delay - minimum_delay),
        # and whether the first of them means what its name says
        "group_delay_s": group_delay,
        "residual_phase_rad": spread,
        "equivalent_residual_delay_s": float(residual_delay),
        "is_a_delay": bool(abs(delay) > residual_delay),
        "weighted_centre_hz": centre,
        "bins": int(carried.sum()),
        "maximum_lag_s": float(maximum_lag_s),
        "band_limit_hz": float(band_limit_hz),
    }


def edge_pair(measured, reference, sample_rate_hz: float,
              band_limit_hz: float,
              maximum_lag_s: Optional[float] = None) -> Dict[str, object]:
    """THE MEASUREMENT PAIR: the pulse's two transients, read separately.

    A whole-pulse delay is one number from one computation, and one number
    cannot be checked. The pulse offers two transients on DISJOINT samples -
    the leading edge and the trailing edge, split at the midpoint between
    them - and a channel delay must move both the same way. A change in the
    pulse's WIDTH moves them in opposite directions and is not a delay at
    all. So the pair is not decoration: it is the only thing separating the
    two, and this lane has already been caught by exactly that confusion
    once, when a fall time stood in for a rise that had never been measured
    independently.

        common_delay_s   (lead + trail) / 2    a delay
        width_change_s    trail - lead         a width, not a delay

    MEASURED, 2026-09-06, by this function on three decoded tapes in
    /tmp/claude-1000. Every field is read against the pooled reference of
    its own tape and the fields are then grouped by isFirstField, so the
    uncertainty quoted is the field-to-field standard error of the two head
    means and the figure in brackets is the ratio of one to the other:

        tape                 flds    whole pulse   common delay   width change
        dod_wide_75bars_SP     26   -4.84+-0.37    -3.78+-0.46    +2.82+-0.38
                                        (13.2)         (8.2)          (7.4)
        dod_wide_75bars_EP     27  -16.77+-0.86   -18.18+-0.89    -3.00+-0.62
                                        (19.5)        (20.5)          (4.9)
        dod_wide_home          40   +3.35+-0.55    +2.22+-0.51    +0.28+-0.30
                                         (6.1)         (4.4)          (0.9)

    all in nanoseconds. THE PAIR GIVES THE THREE TAPES THREE DIFFERENT
    ANSWERS, which one number could not have done.

    The HOME recording carries a delay and nothing else: 2.22 ns at 4.4
    standard errors, with the width change at 0.9 and consistent with zero.
    The EP recording is delay-dominated too and far larger, 18.18 ns at
    20.5 standard errors - and `transit_delay` agrees independently, this
    being the only one of the three whose `is_a_delay` reads True. The SP
    recording carries BOTH terms at once, 3.78 ns of delay and 2.82 ns of
    width at 8.2 and 7.4 standard errors, on the same deck and the same
    pattern as the EP one; its whole-pulse figure of 4.84 ns is neither of
    them and is what a single reading would have reported.

    THE PAIR ALSO CAUGHT THE ESTIMATOR. Read through a Hann window - which
    is what this measurement did first - the same three tapes gave width
    changes of +1.90, -1.51 and +1.85 ns at 5.7, 2.8 and 7.3 standard
    errors, and the home recording appeared to be a width change with no
    delay. The window under-reads a shift, and it under-reads the two
    transients by different amounts because they sit at different distances
    from the window's centre, so it manufactures a width change out of a
    delay. That is why `_band_response` removes the joining line instead.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    measured = np.asarray(measured, dtype=np.float64).ravel()
    where = locate_transients(reference, sample_rate_hz)
    cut = where["midpoint"]
    leading = transit_delay(measured[:cut], reference[:cut], sample_rate_hz,
                            band_limit_hz, maximum_lag_s)
    trailing = transit_delay(measured[cut:], reference[cut:], sample_rate_hz,
                             band_limit_hz, maximum_lag_s)
    whole = transit_delay(measured, reference, sample_rate_hz, band_limit_hz,
                          maximum_lag_s)
    lead_s = leading["delay_s"]
    trail_s = trailing["delay_s"]
    return {
        "leading": leading,
        "trailing": trailing,
        "whole": whole,
        "transients": where,
        "leading_delay_s": lead_s,
        "trailing_delay_s": trail_s,
        "common_delay_s": 0.5 * (lead_s + trail_s),
        "width_change_s": trail_s - lead_s,
        "whole_delay_s": whole["delay_s"],
        "why": ("a channel delay moves both transients the same way and a "
                "width change moves them oppositely, so one number from the "
                "whole pulse cannot tell them apart"),
    }
