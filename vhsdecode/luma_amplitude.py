"""Luma FM carrier amplitude variation.

An FM carrier carries no amplitude information - its amplitude should be
constant. Every deviation from that constant is imposed by the tape, and is
measurable on the raw RF, before the time base correction has run and before
any scaling has been applied.

The amplitude is taken as the decoder's own envelope channel, at the full
bandwidth the demodulator produced it in. Nothing narrows it: no smoothing, no
band limit, no windowing. The tape imposes its amplitude noise at whatever rate
it imposes it, and a filter would decide in advance which of that the correction
is allowed to see. The dropout detector low passes the same channel for its own
purposes, in `doc.py`, because a wide envelope crosses a threshold repeatedly
inside one damaged region - but that is the detector's business and it is not
imposed here.

Two causes of amplitude change, only one of them noise
------------------------------------------------------
The received carrier amplitude is not constant, and the reasons split in two.

  Frequency driven.  The carrier sweeps with picture content - sync tip to peak
  white - and the amplitude response of the path varies across that sweep. The
  same carrier frequency gives the same amplitude every time, so this is the
  path's response and not tape noise, and it is not variation to report.

  Random.  Head-to-medium separation modulating the signal. This is the tape's
  own noise, and it is what the measurement is for.

Separating them is the whole job of this module, and it happens here at the
measurement rather than at whatever acts on it: the part of the carrier
amplitude that is a function of the luma's own frequency is divided out once,
leaving only the amplitude changes worth acting on. Downstream the two are no
longer separable.

Part of that response is known exactly and is simply divided out - the decoder's
own RF path, which on formats using the linear ramp boost tilts the band by
several dB across the carrier sweep on its own. The remainder - tape channel,
head response, record and playback equalization - is measured by binning the
flattened amplitude against carrier frequency and fitting a curve.

Both parts are functions of the carrier's frequency and of nothing else, so they
are combined into a single table over frequency and divided out together, once
per sample. That is what keeps this affordable: the response is exponentiated
where it is described, over the few thousand points of the table, rather than
over the million samples it is applied to.

What comes back is a property of the luma carrier and of the tape, and of
nothing else. It carries no assumption about what will consume it.
"""

from collections import namedtuple

import numpy as np
from numba import njit

import scipy.fft as sps_fft
from scipy.ndimage import maximum_filter1d


# What `measure_amplitude_deviation` yields. `deviation` is unity where the
# carrier holds the amplitude its own frequency predicts, and departs from unity
# by the tape's multiplicative noise. `carrier_hz` is the instantaneous carrier
# frequency conditioned the same way the deviation was measured against, so a
# consumer scaling by wavelength uses the frequency that actually applies.
# `steadiness` withdraws belief where the carrier is sweeping too fast for the
# path to follow.
#
# All are single precision, which is what the demodulator produced them in;
# widening them here would cost a full copy of the field per field and buy no
# precision that was ever recorded.
AmplitudeDeviation = namedtuple(
    "AmplitudeDeviation",
    ["deviation", "carrier_hz", "steadiness", "response"],
)


# The response the deviation was measured against, as its components rather
# than as a table. `separation` is the straight line the amplitude follows
# against carrier frequency - the frequency it is centred on, the log amplitude
# there, and the slope per Hz, about -0.4 per MHz on this deck. What that slope
# physically is remains open: the record head capture, taken before the tape
# exists, has the same slope to within 4%, so whatever it is, it is not the
# medium.
#
# `residual_hz` and `residual` are what the binned medians still departed from
# that line by. It looks like the rest of the path and it is NOT: it is a
# function of the picture. The record capture reproduces the same residual for
# the same picture at r = +0.89 with no tape involved at all, while two record
# captures of DIFFERENT pictures through the same electronics anticorrelate at
# r = -0.29. What it measures is how the picture's own transitions populate the
# level bins - a sweeping carrier's amplitude has collapsed through the path's
# band limit and lands in whatever bin the transition crossed.
#
# So do not fit it, spline it, or weigh by it. Measured, every use loses:
# subtracted from the model it costs +0.0595 on the record-referenced 75bars
# metric, weighed against the deviation as a Wiener confidence +0.0041. It is
# carried for inspection only, and it is the same reason nothing freer than a
# line survives here - every degree of freedom past the line absorbs picture and
# hands it to whatever consumes the deviation.
#
# `described_hz` and `described` are the accumulated dense response actually in
# force - the steadiest samples only, pooled per head across the decode. It is
# what the model IS, so it is what a plot should be judged against; the median
# of every sample is a different population and the two are not expected to
# agree where a level is reached mostly in passing.
# The fields from `described_hz` onward are the accumulated dense response, and
# there is none of it until a head has been seen at least once - the first field
# of a decode, and the second, describe nothing yet. They default to empty rather
# than being filled in at the one place the model is built, so that how many of
# them there are stays the field list's own business: this arity has already
# gone wrong once, when the dense response grew a third component and the
# construction site still passed two.
_RESPONSE_FIELDS = [
    "separation", "residual_hz", "residual",
    "described_hz", "described", "described_weight",
]
ResponseModel = namedtuple(
    "ResponseModel",
    _RESPONSE_FIELDS,
    defaults=(np.zeros(0),)
    * (len(_RESPONSE_FIELDS) - _RESPONSE_FIELDS.index("described_hz")),
)


# Resolution the response curve is described at, spaced evenly across the
# carrier's deviation range. One point per IRE: that is the video standard's own
# quantization of level, the carrier frequency is a linear function of level,
# and the recorded RF carries eight bits against a luma signal-to-noise around
# 43 dB, so a level distinction finer than an IRE is not in the data to be
# resolved. `carrier_frequency_bins` already bins at exactly this rate, so the
# curve simply keeps those bins rather than merging them.
#
# Equal width, not equal population. Equal population would put every point at
# the same confidence and put the points where the picture actually is, which is
# what a fit wants - but this curve is interpolated through its points rather
# than fitted, and equal population crowds them to within 9 kHz of each other
# wherever the picture dwells. Interpolating through closely spaced medians
# turns their noise into steep wiggles, which are then divided back out of the
# amplitude as though they were response.
#
# Resolution alone does not make the curve sharper. What a point can resolve is
# set by how many samples stand behind it, which is the budget below; the
# smoothing widens until it has them. Raising the resolution buys a finer
# description of whatever the budget can already determine, and costs nothing.
RESPONSE_CURVE_RESOLUTION_IRE = 1.0

# Independent features a physical channel response has across the carrier's
# range - tape channel, head, record and playback equalization, all smooth in
# frequency. This is not the resolution; it is how much evidence the curve
# needs, and so how much of the field is walked to fit it.
CURVE_INDEPENDENT_POINTS = 16

# Samples standing behind each point of the curve, after smoothing. A median's
# standard error is about 1.25 * sigma / sqrt(m); the amplitude deviation runs
# about 5% rms and the correction acts at about 1% rms, so determining the curve
# an order of magnitude finer than it acts needs (1.25 * 0.05 / 0.001)^2, near
# enough four thousand samples a point. This is the knob that trades the curve's
# sharpness against its noise: the smoothing pools neighbouring points until it
# reaches this, so a smaller number resolves finer structure and admits more
# wiggle.
CURVE_SAMPLES_PER_POINT = 4096

# Below this a point carries no measurement at all and is dropped rather than
# placed, for the interpolation to span.
CURVE_MINIMUM_POPULATION = 64


# Points the accumulated response needs before it stands in for the line.
CURVE_DENSE_MINIMUM_POINTS = 8


# Natural log amplitude to decibels, so the instrumentation reports in the unit
# the rest of the signal path is stated in.
_LOG_TO_DB = 20.0 / np.log(10.0)

# How far past the nominal deviation range the curve is measured, in IRE. The
# carrier does not stay between sync tip and peak white: sync edges undershoot
# hard, and about a tenth of a field lands outside. The response there is real
# and worth measuring rather than extrapolating - measured, the frequency left
# in the deviation outside the nominal range falls from 0.10 to well under that
# once the curve reaches it. The range is still fixed and derived from the
# format, so the curve means the same thing from field to field.
CURVE_RANGE_MARGIN_IRE = 120



@njit(cache=True, nogil=True, fastmath=True)
def _flatten_by_response(amplitude, carrier_hz, response, frequency_step):
    """Divide the decoder's own RF response out of the carrier amplitude.

    The response is tabulated on a uniform frequency grid, so the sample's bin
    is an index rather than a search.

    Only the samples the response curve is fitted from come through here - a
    few tens of thousands of the field's million. Applying the finished model
    to the whole field is `_deviation_from_inverse_response`, which divides out
    this response and the fitted curve together.
    """
    count = len(amplitude)
    # Carrier amplitude is physically positive, so zero
    # cannot be mistaken for a measurement.
    flattened = np.zeros(count)
    last = len(response) - 1
    for i in range(count):
        position = carrier_hz[i] / frequency_step
        if position <= 0.0:
            value = response[0]
        elif position >= last:
            value = response[last]
        else:
            lower = int(position)
            fraction = position - lower
            value = response[lower] * (1.0 - fraction) + response[lower + 1] * fraction
        if value > 0.0:
            flattened[i] = amplitude[i] / value
    return flattened


def rf_path_response(rf):
    """The decoder's own RF amplitude response, and the grid it is tabulated on.

    `Filters["RFVideo"]` is the magnitude response actually applied to the
    signal before the carrier amplitude is taken, so consuming it directly keeps
    this correct for every format and every combination of ramp boost, peaking
    and notch options.

    Fixed for the life of the decoder, so it is derived once and kept on the
    decoder rather than rebuilt every field.
    """
    cached = getattr(rf, "_luma_amplitude_response", None)
    if cached is not None:
        return cached

    response = np.abs(rf.Filters["RFVideo"])
    # The stored response covers the whole spectrum with the negative
    # frequencies mirrored above Nyquist; only the positive half is meaningful.
    half = len(response) // 2
    cached = (response[:half], rf.freq_hz / len(response))
    rf._luma_amplitude_response = cached
    return cached


def path_transient_scale(rf):
    """How fast the carrier may sweep before its amplitude stops meaning anything.

    A band limited path cannot follow a carrier that moves quickly: the
    sidebands of the sweep fall outside the passband and the envelope collapses
    for as long as the path takes to settle. That is frequency modulation
    arriving as amplitude modulation, and it is not tape loss - the record head
    capture shows four fifths of it already present before the tape ever
    touched the signal, and the color-under, which does not change frequency,
    never suffered it at all. Correcting the chroma for it imposes the luma's
    transitions on the chroma.

    Both numbers come from the decoder's own RF filter rather than from
    fitting. The scale is the passband width divided by the length of the
    path's impulse response: the sweep rate at which the carrier crosses its
    whole passband while the path is still settling from where it started.
    """
    cached = getattr(rf, "_luma_transient_scale", None)
    if cached is not None:
        return cached

    magnitude = np.abs(rf.Filters["RFVideo"])
    half = len(magnitude) // 2
    band = magnitude[:half]
    step = rf.freq_hz / len(magnitude)
    inside = np.flatnonzero(band > band.max() / np.sqrt(2.0))
    width = (inside[-1] - inside[0]) * step if len(inside) > 1 else rf.freq_hz / 4

    impulse = np.abs(np.fft.ifft(rf.Filters["RFVideo"]))
    energy = np.cumsum(np.roll(impulse, len(impulse) // 2) ** 2)
    energy /= energy[-1]
    span = max(int(np.searchsorted(energy, 0.98) - np.searchsorted(energy, 0.02)), 1)

    cached = (width / span, span)
    rf._luma_transient_scale = cached
    return cached


def carrier_frequency_bins(sys_params, rf):
    """Frequency bin edges across the carrier's deviation range, one bin per IRE.

    One bin per IRE because that is the video standard's own quantization of
    level, and the carrier frequency is a linear function of level.

    Anchored on the format's specified carrier frequencies, not on the levels
    the decoder has measured. This measurement is taken on the raw RF - the
    carrier's own envelope against its own instantaneous frequency - and the
    decoder's running `ire0` and `hz_ire` are derived from sync tip and
    blanking levels found in the DEMODULATED luma, which is downstream of here.
    Anchoring on them imports a later, level-tracked estimate into an earlier
    measurement and makes the frequency axis move field to field.

    Measured, it moves a long way: on a second-generation tape decoded with
    `--ire0_adjust`, the running anchor put the whole scale 36 IRE from the
    specified one (`ire0` 3.4277 MHz against 3.6857 MHz), so a bin meant a
    different frequency in every field and the accumulation averaged unlike
    with unlike.

    What the specified anchor buys is that it is CONSTANT, not that it is
    right. A deck or a tape whose carrier sits off the nominal deviation simply
    uses a different part of the range, and nothing downstream cares: the
    response is divided out of the amplitude that measured it, so every
    quantity here is relative to itself and a fixed offset cancels. The margin
    either side is what absorbs the offset. A moving anchor is the only thing
    that does real damage, because it is what stops a bin meaning one
    frequency.

    Samples outside this range are dropped rather than clipped into the end
    bins; see `measure_amplitude_by_frequency`.
    """
    sync_tip_hz = rf.iretohz(
        sys_params["vsync_ire"] - CURVE_RANGE_MARGIN_IRE, spec=True
    )
    peak_white_hz = rf.iretohz(100 + CURVE_RANGE_MARGIN_IRE, spec=True)
    span_ire = (peak_white_hz - sync_tip_hz) / sys_params["hz_ire"]
    bin_count = int(round(span_ire / RESPONSE_CURVE_RESOLUTION_IRE))
    return sync_tip_hz, peak_white_hz, max(bin_count, CURVE_INDEPENDENT_POINTS)


@njit(cache=True, nogil=True, fastmath=True)
def measure_amplitude_by_frequency(
    amplitude, carrier_hz, sync_tip_hz, hz_ire, bin_count, group_count
):
    """Carrier amplitude against carrier frequency, as points on a curve.

    Returns one point per group: the frequency the group covers, the median
    amplitude in it, and how many samples it holds. The median is what makes
    the estimate robust - amplitude noise is not symmetric, dropouts sit in the
    low tail, so a median tracks the level the carrier normally has at that
    frequency rather than the average of level and damage.

    The two ends are treated differently, deliberately and not obviously.
    Samples below the range are dropped; samples above it are kept, pooled into
    the top bin. That is a real bias: 0.9% of samples, averaging 5.26 MHz, land
    in a bin centred near 4.8, and being at the fit's longest lever arm they
    steepen the fitted slope by about +0.039 /MHz - 75bars reads -0.400 with
    them and -0.362 without.

    Dropping them as well was tried, which is what this docstring used to
    describe, and it is measurably worse: +0.085 on the record-referenced
    75bars metric, enough to put the correction back to net harmful, with
    chromanoise unchanged. Widening the range so they bin at their true
    frequency does nothing - those bins fall below the minimum population and
    are dropped regardless.

    The reason the biased fit wins is not principled. Those samples are sync
    undershoots and transition overshoots whose amplitude has collapsed through
    the path's band limit rather than through any response, and understating
    their frequency lets that collapse steepen the line - which evidently
    cancels part of the same transient's contribution to the deviation. It is
    an empirical choice, recorded so it is not silently "fixed" again.

    The groups are equal-width across the deviation range, so the points the
    curve is interpolated through stay evenly spaced. Groups the picture barely
    visits come back with little population and the caller drops them; the
    interpolation simply spans the gap, which is a better answer than a point
    placed on a handful of samples.

    The caller passes a subsample of the field, already gathered; the curve has
    three parameters and even a subsample carries tens of thousands of points,
    so it is not short of evidence. The correction itself still applies to every
    sample.
    """
    count = len(amplitude)
    fine = np.zeros(bin_count, dtype=np.int64)
    index = np.full(count, -1, dtype=np.int64)

    for i in range(count):
        value = amplitude[i]
        if value <= 0.0:
            continue
        bin_number = int(np.floor((carrier_hz[i] - sync_tip_hz) / hz_ire))
        if bin_number < 0:
            continue
        if bin_number >= bin_count:
            # Kept, at the top bin, rather than dropped - see above.
            bin_number = bin_count - 1
        index[i] = bin_number
        fine[bin_number] += 1

    total = 0
    for b in range(bin_count):
        total += fine[b]
    if total <= 0:
        return (
            np.zeros(0), np.zeros(0), np.zeros(0)
        )

    # merge adjacent fine bins into equal spans of the deviation range
    group_of = np.zeros(bin_count, dtype=np.int64)
    for b in range(bin_count):
        g = (b * group_count) // bin_count
        if g > group_count - 1:
            g = group_count - 1
        group_of[b] = g
    groups = group_count

    population = np.zeros(groups)
    centre = np.zeros(groups)
    for b in range(bin_count):
        g = group_of[b]
        population[g] += fine[b]
        centre[g] += fine[b] * (sync_tip_hz + (b + 0.5) * hz_ire)
    for g in range(groups):
        if population[g] > 0.0:
            centre[g] /= population[g]

    start = np.zeros(groups + 1, dtype=np.int64)
    for g in range(groups):
        start[g + 1] = start[g] + np.int64(population[g])
    cursor = start[:groups].copy()
    grouped = np.empty(start[groups], dtype=amplitude.dtype)
    for i in range(count):
        b = index[i]
        if b >= 0:
            g = group_of[b]
            grouped[cursor[g]] = amplitude[i]
            cursor[g] += 1

    levels = np.zeros(groups)
    for g in range(groups):
        first, last = start[g], start[g + 1]
        if last > first:
            levels[g] = np.median(grouped[first:last])

    return levels, population, centre


def response_curve(levels, population, centre_hz):
    """The path's amplitude response across the carrier deviation range.

    One component, because one is all the measurement supports. In the
    logarithm it is a straight line against carrier frequency - about
    -0.4 per MHz on this deck, fitted through the binned medians
    `measure_amplitude_by_frequency` returns - a median because amplitude noise
    is not symmetric and dropouts sit in the low tail.

    What the slope physically is remains open. Separation loss would give
    exactly this shape - exp(-2*pi*d/lambda) with the recorded wavelength going
    as 1/f is a line in the logarithm - but the record head capture, taken
    before the tape, has the same slope to within 4% (-0.385 against -0.400 on
    75bars, -0.394 against -0.366 on chromanoise). Whatever it is, it is
    already there before the medium, so it is not head-to-medium separation.

    Nothing freer than a line survives contact with the measurement. Measured
    as the share of the deviation's variance that carrier frequency still
    explains - which is picture left behind, to be corrected onto the chroma as
    though it were tape noise - a straight line leaves 0.075 / 0.048 / 0.046 on
    75bars SP, chromanoise SP and 75bars EP, against 0.14 / 0.035 / 0.057 for a
    parabola, 0.27 / 0.16 / 0.070 for an interpolated curve through the medians,
    and 0.34 and 0.50 for cubics and quartics. Every degree of freedom past the
    line fits noise and hands it back as though it were response, and the
    deviation's own size grows with it - 5.8% rms at a line against 9.2% at a
    quartic. A wrong-but-plausible control, the same line with its slope
    reversed, scores 0.91.

    Returned as the line in the logarithm, because the response is
    multiplicative: the frequency it is centred on, the log level there, and the
    slope per Hz. None if the field does not visit enough distinct levels to
    place a line.
    """
    # A bin standing on a handful of samples is noise wearing the shape of a
    # response, and would pull the line by its lever arm; it is dropped.
    valid = (population >= CURVE_MINIMUM_POPULATION) & (levels > 0)
    if np.count_nonzero(valid) < 2:
        return None

    frequency = centre_hz[valid]
    log_level = np.log(levels[valid])
    weight = population[valid]

    # Weighted by population, and centred before the fit so the intercept is
    # determined where the evidence is rather than extrapolated back to zero
    # frequency, which is millions of Hz outside the measurement.
    mass = weight.sum()
    if not mass > 0.0:
        return None
    centre = float((frequency * weight).sum() / mass)
    offset = frequency - centre
    spread = float((weight * offset * offset).sum())
    if not spread > 0.0:
        return None
    level = float((log_level * weight).sum() / mass)
    slope = float((weight * offset * (log_level - level)).sum() / spread)
    return centre, level, slope


def _inverse_response_table(response, frequency_step, curve, dense=None):
    """The whole modelled response, inverted, on the grid it is tabulated on.

    The decoder's own RF path and the line measured on top of it are both
    functions of carrier frequency alone, so they are combined here rather than
    divided out one after the other. It turns the correction's per-sample
    work into a table lookup and a multiply, and confines the exponential to the
    few thousand points that describe the response rather than the million it is
    applied to.

    The line covers the whole grid, including the frequencies the picture never
    visited. That is not extrapolation past the evidence but the model itself:
    a line is a line everywhere, and about a
    tenth of a field - sync undershoot especially - lands outside the nominal
    deviation range and would otherwise be corrected against a response held
    artificially flat.

    Zero marks a frequency at which the model says nothing usable - the response
    vanishes. That is the same sentinel the amplitude itself uses, and it leaves
    the deviation at unity.
    """
    centre_hz, level, slope = curve
    grid_hz = np.arange(len(response), dtype=np.float64) * frequency_step
    log_level = level + slope * (grid_hz - centre_hz)
    if dense is not None:
        # Shrunk towards the line by how much evidence each point stands on.
        # `CURVE_MINIMUM_POPULATION` is the weight at which a bin is believed
        # half on its own measurement and half on the line, so a point with far
        # more than that is essentially its own measurement and a point with far
        # less is essentially the line.
        described_hz, described, described_weight = dense
        inside = (grid_hz >= described_hz[0]) & (grid_hz <= described_hz[-1])
        measured = np.interp(grid_hz[inside], described_hz, described)
        confidence = np.interp(grid_hz[inside], described_hz, described_weight)
        share = confidence / (confidence + CURVE_MINIMUM_POPULATION)
        log_level[inside] = share * measured + (1.0 - share) * log_level[inside]

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        modelled = response * np.exp(log_level)
        # Narrowed here rather than after, so a reciprocal that is finite in
        # double and infinite in single is caught by the same test as the rest.
        inverse = np.reciprocal(modelled).astype(np.float32)

    inverse[~np.isfinite(inverse)] = 0.0
    inverse[modelled <= 0.0] = 0.0
    return inverse


@njit(cache=True, nogil=True, fastmath=True)
def _deviation_from_inverse_response(
    amplitude, carrier_hz, inverse_response, frequency_step, low, high
):
    """How far the amplitude departs from the response modelled for it.

    One pass over the field, and no transcendental in it: the modelled response
    was inverted where it was tabulated, so what is left per sample is one
    interpolation off a computed index, a multiply and a bound.
    """
    count = len(amplitude)
    deviation = np.empty(count, dtype=np.float32)
    last = len(inverse_response) - 1
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for i in range(count):
        position = carrier_hz[i] / frequency_step
        if position <= zero:
            inverse = inverse_response[0]
        elif position >= last:
            inverse = inverse_response[last]
        else:
            lower = np.int32(position)
            fraction = position - np.float32(lower)
            base = inverse_response[lower]
            inverse = base + (inverse_response[lower + 1] - base) * fraction
        ratio = amplitude[i] * inverse
        if ratio <= zero:
            # no measurement here, or no usable model - leave the sample alone
            ratio = one
        elif ratio < low:
            ratio = low
        elif ratio > high:
            ratio = high
        deviation[i] = ratio
    return deviation


@njit(cache=True, nogil=True, fastmath=True)
def _static_response(carrier_hz, response, frequency_step, out):
    """The tabulated response at each sample's own frequency.

    The table is on a uniform grid, so this is an index and a lerp rather than
    the binary search per sample that a general interpolation would do.
    """
    last = len(response) - 1
    for i in range(len(carrier_hz)):
        position = carrier_hz[i] / frequency_step
        if position <= 0.0:
            out[i] = response[0]
        elif position >= last:
            out[i] = response[last]
        else:
            lower = int(position)
            fraction = position - lower
            base = response[lower]
            out[i] = base + (response[lower + 1] - base) * fraction
    return out


def _predicted_sweep_collapse(rf, carrier_hz, stride):
    """The amplitude a constant carrier loses purely by MOVING, predicted.

    The channel is specified - `Filters["RFVideo"]` is built from the format's
    own parameters and is what the signal went through - so the FM-to-AM a sweep
    suffers can be predicted instead of inferred. Synthesise a unit amplitude
    carrier following the measured trajectory, filter it, take the analytic
    magnitude.

    Then divide out the STATIC response at the same instantaneous frequency.
    What the filter does to a carrier standing still is already removed by
    `_flatten_by_response`; leaving it in here applies it a second time, and
    that is not small - |RFVideo| is 0.157 at 3.1 MHz against 0.268 at 3.9, so
    squaring it costs 4.6 dB at the bottom of the range and nothing at the top.
    What is left is unity for a stationary carrier and below it for a moving one.

    Returned on `stride`, which is the grid the caller reads it on. The
    transforms have to span the whole field - the trajectory is what it is - but
    nothing after them does, so the magnitude, the static response divided out
    of it and the median that normalises the pair all run on one sample in
    `stride`. That still leaves tens of thousands of samples behind the median.

    Worth -0.0122 on the record-referenced 75bars SP metric and -0.0125 at EP
    against gating alone, both resolved; +0.0005 on chromanoise, which has few
    transients for it to act on. Removing the collapse does NOT make the gate
    redundant - the prediction is a model of the path, not a measurement of this
    field, so the steadiest samples are still the ones worth describing the
    response with. Correcting without gating is measurably worse than gating
    without correcting (-0.036 against -0.125 on 75bars SP).
    """
    # The same band and grid `_flatten_by_response` divides out, which is held
    # on the decoder rather than rebuilt from the filter every field.
    band, step = rf_path_response(rf)
    count = len(carrier_hz)

    # Keyed on the length it was built for. Fields are not all the same length,
    # and a cache that only remembers the last one rebuilds a few hundred
    # thousand interpolated points every time the length changes - which on real
    # content is most fields. A handful of lengths recur, so a few are kept.
    cache = rf.__dict__.setdefault("_luma_sweep_response", {})
    cached = cache.get(count)
    if cached is None:
        cached = np.interp(np.fft.rfftfreq(count, 1.0 / rf.freq_hz),
                           np.arange(len(band)) * step, band, left=band[0], right=0.0)
        if len(cache) >= 8:
            cache.clear()
        cache[count] = cached

    decoder = getattr(rf, "decoder", None)
    workers = max(int(getattr(decoder, "numthreads", 1) or 1), 1)

    phase = np.cumsum(np.asarray(carrier_hz, dtype=np.float64))
    phase *= 2.0 * np.pi / rf.freq_hz
    spectrum = sps_fft.rfft(np.cos(phase, out=phase), workers=workers)
    spectrum *= cached
    # a one sided spectrum inverts straight to the analytic signal
    # The transforms are the largest cost here and scipy will thread them, on
    # the decoder's own thread budget rather than helping themselves to the
    # machine: a decode asked to run on one thread gets one.
    analytic = sps_fft.ifft(
        np.concatenate([spectrum * 2.0, np.zeros(count - len(spectrum))]),
        workers=workers,
    )
    predicted = np.abs(analytic[::stride])

    sampled_hz = np.asarray(carrier_hz[::stride], dtype=np.float64)
    static = _static_response(sampled_hz, band, step, np.empty(len(sampled_hz)))
    alive = static > 0.0
    collapse = np.zeros(len(static))
    np.divide(predicted, static, out=collapse, where=alive)
    reference = np.median(collapse[alive]) if alive.any() else 0.0
    return collapse / reference if reference > 0 else None


def wants_averaging_probe(rf):
    """Whether anything is going to read the averaging instrumentation.

    The probes below and their opposite numbers in `chroma.py` cost several
    walks of the field between them, and feed one debug plot.
    """
    debug_plot = getattr(rf, "debug_plot", None)
    return bool(debug_plot) and debug_plot.is_plot_requested("luma_averaging")


def _instrument_averaging(
    rf, key, curve, dense, deviation, levels, population, bin_count, clamp_low, clamp_high
):
    """Report what the averaging is doing, without changing what it does.

    Only run when `--debug_plot luma_averaging` asked for it: this walks the
    whole field several more times - a mask, a widening, a logarithm and two
    comparisons - and everything it produces is read by that plot and by
    nothing else.

    Four things are measured here and none of them is acted on. They exist
    because the accumulation makes claims that have never been checked against
    the decode it runs on:

    - the step the model takes at the two edges of the described range, which
      is what the blend leaves behind when the dense term carries a decode
      average level and the line outside it carries this field's;
    - the shrinkage actually in force, against the constant that is supposed to
      set it;
    - whether the response is stationary, which is what an unbounded running
      mean assumes, measured as the lag one autocovariance of the field to
      field difference - a quantity a variance alone cannot separate;
    - how often the deviation reaches its clamps.

    Everything is accumulated per head, because the heads differ.
    """
    probe = rf.__dict__.setdefault("_luma_averaging_probe", {})
    state = probe.setdefault(
        key,
        {
            "fields": 0,
            "edge_low_db": [],
            "edge_high_db": [],
            "deviation_db": [],
            "clamp_low": [],
            "clamp_high": [],
            "previous": None,
            "previous_population": None,
            "previous_difference": None,
            "difference_square": np.zeros(bin_count),
            "difference_lag": np.zeros(bin_count),
            "difference_count": np.zeros(bin_count),
        },
    )
    if len(state["difference_square"]) != bin_count:
        return
    state["fields"] += 1

    # The deviation's own spread, as the scale everything else is read against.
    alive = deviation > 0.0
    if np.count_nonzero(alive) > 0:
        spread = float(np.std(np.log(np.asarray(deviation[alive], dtype=np.float64))))
        state["deviation_db"].append(_LOG_TO_DB * spread)
    state["clamp_low"].append(float(np.count_nonzero(deviation <= clamp_low)) / len(deviation))
    state["clamp_high"].append(float(np.count_nonzero(deviation >= clamp_high)) / len(deviation))

    # The step at each edge of the described range: what the blend puts there
    # and what stands one bin outside it are two different statements, and this
    # is how far apart they are.
    if dense is not None:
        described_hz, described, described_weight = dense
        centre_of_line, level_of_line, slope_of_line = curve
        share = described_weight / (described_weight + CURVE_MINIMUM_POPULATION)
        line = level_of_line + slope_of_line * (described_hz - centre_of_line)
        step = share * (described - line)
        state["edge_low_db"].append(_LOG_TO_DB * float(step[0]))
        state["edge_high_db"].append(_LOG_TO_DB * float(step[-1]))
        state["share"] = share
        state["share_hz"] = described_hz

    # Stationary or drifting. The field to field difference of a bin's level is
    # a moving average of order one if the level itself is a random walk plus
    # measurement noise, so its lag one autocovariance is minus the measurement
    # variance and its own variance is the walk's step plus twice that. A
    # variance alone cannot separate the two; the lag term is what does.
    described_now = (population >= CURVE_MINIMUM_POPULATION) & (levels > 0)
    current = np.where(described_now, np.log(np.maximum(levels, 1e-30)), 0.0)
    previous, previous_population = state["previous"], state["previous_population"]
    if previous is not None:
        both = described_now & (previous_population >= CURVE_MINIMUM_POPULATION)
        difference = np.where(both, current - previous, 0.0)
        earlier = state["previous_difference"]
        if earlier is not None:
            paired = both & (earlier != 0.0)
            state["difference_lag"][paired] += difference[paired] * earlier[paired]
        state["difference_square"][both] += difference[both] * difference[both]
        state["difference_count"][both] += 1.0
        state["previous_difference"] = difference
    state["previous"] = current
    state["previous_population"] = np.where(described_now, population, 0.0)


def measure_amplitude_deviation(field):
    '''How far the luma carrier's amplitude departs from the constant it should
    hold, measured across the whole field on the raw RF sample grid.

    Returns an `AmplitudeDeviation`, or None if the field cannot support a
    measurement.

    Demodulation happens per block and is not repeated here: this works on the
    assembled field, so the blocks are concatenated first and the part of the
    amplitude that follows the carrier's own frequency is removed afterwards,
    once, over the whole field. The tape imposes its amplitude noise as the
    field is read, continuously, so the reference it is measured against has to
    be continuous too - fitting each demodulation block separately puts a step
    at every block seam, on a carrier amplitude that has none.

    The response curve is fitted from a subsample and applied to every sample.
    The curve has three parameters and the subsample carries tens of thousands
    of points, so the fit is not the limiting factor; walking the whole field
    twice to determine it would be.

    Nothing here knows about the color-under. What comes back is a property of
    the luma carrier alone - the tape's multiplicative noise, with the path's
    response to the carrier's own frequency already removed - so it can drive
    any correction that shares the same head and instant.
    '''
    rf = field.rf
    video = field.data["video"]
    if video is None or "demod_raw" not in video:
        return None

    # Read as recorded. Both channels are already single precision, so these are
    # views onto the field's record array rather than copies of it.
    amplitude = video["envelope"]
    carrier_hz = video["demod_raw"]
    if len(amplitude) != len(carrier_hz) or len(amplitude) == 0:
        return None

    sys_params = rf.SysParams
    sync_tip_hz, peak_white_hz, bin_count = carrier_frequency_bins(sys_params, rf)
    response, frequency_step = rf_path_response(rf)

    # The two channels are half a sample apart. `unwrap_hilbert` writes the
    # frequency of the interval i-1 to i at index i, so the frequency there
    # describes the instant i - 0.5, while the amplitude at i describes i.
    # Recentring costs a two tap average - the shortest half sample
    # interpolator there is - and lets the response be divided out against the
    # frequency that actually applied to the sample it explains.
    #
    # Measured rather than assumed: the response curve is a fixed map from
    # frequency to amplitude, so a timing error between the two shows up in the
    # deviation as the curve's own slope times the sweep rate. Regressed on
    # that, alongside the sweep loss it would otherwise be confused with, the
    # offset comes to -0.507 +- 0.078 samples.
    centred_hz = np.empty_like(carrier_hz)
    np.add(carrier_hz[:-1], carrier_hz[1:], out=centred_hz[:-1])
    centred_hz[:-1] *= np.float32(0.5)
    centred_hz[-1] = carrier_hz[-1]

    # How far the amplitude can be believed, given how fast the carrier is
    # sweeping. The collapse lags the sweep and outlasts it, so the sweep rate
    # is spread over the settling time before it is weighed.
    knee, span = path_transient_scale(rf)
    slew = np.abs(np.diff(np.asarray(carrier_hz, dtype=np.float32), prepend=carrier_hz[0]))
    slew = maximum_filter1d(slew, size=3 * span, origin=-span, mode="nearest")
    relative = slew / np.float32(knee)
    steadiness = (1.0 / (1.0 + relative * relative)).astype(np.float32)

    # The part of the amplitude that follows the carrier's own frequency,
    # measured once over the concatenated field. Demodulation stays per block
    # and is not repeated here; what is measured is the assembled result, so
    # neither the curve nor the level it implies has a block boundary in it.
    #
    # A sweeping carrier is deliberately NOT kept out of this. Its amplitude has
    # collapsed through the path's band limit rather than through anything the
    # tape did, and it biases the median of whatever level bin the transition
    # passed through - which is why the residual about the fitted line follows
    # the picture (r = +0.81 between the same pattern at two tape speeds, only
    # +0.24 between different patterns on the same tape). Excluding those
    # samples does make the deviation flatter, but it also narrows the level
    # distribution the line is fitted from, and measured against the record
    # reference it costs +0.042 on 75bars. Flatness is not the objective.
    stride = max(
        1, len(amplitude) // (CURVE_INDEPENDENT_POINTS * CURVE_SAMPLES_PER_POINT)
    )
    sampled_amplitude = np.asarray(amplitude[::stride], dtype=np.float64)
    sampled_carrier_hz = np.asarray(centred_hz[::stride], dtype=np.float64)
    sampled_flattened = _flatten_by_response(
        sampled_amplitude, sampled_carrier_hz, response, frequency_step
    )
    levels, population, centre_hz = measure_amplitude_by_frequency(
        sampled_flattened,
        sampled_carrier_hz,
        sync_tip_hz,
        sys_params["hz_ire"] * RESPONSE_CURVE_RESOLUTION_IRE,
        bin_count,
        bin_count,
    )
    curve = response_curve(levels, population, centre_hz)
    if curve is None:
        return None

    # What the binned medians still depart from the line by. Carried for
    # inspection, and deliberately not acted on - see `ResponseModel`.
    centre_of_line, level_of_line, slope_of_line = curve
    measured = (population >= CURVE_MINIMUM_POPULATION) & (levels > 0)
    residual_hz = centre_hz[measured]
    residual = np.log(levels[measured]) - (
        level_of_line + slope_of_line * (residual_hz - centre_of_line)
    )

    # The response read off the measurement itself, accumulated per head across
    # the decode. A single field determines it far too poorly - that is why
    # fitting densely per field loses - but a decode's worth of one head's
    # fields determines it well, and the two heads genuinely differ.
    #
    # Binned a second time, from every sample, with the sweep collapse divided
    # out first so a carrier that was moving contributes the amplitude it would
    # have had standing still. What comes back is the amplitude's own median
    # against carrier frequency - the curve the debug plot draws as "carrier
    # amplitude / median" - and it is used as the model directly.
    #
    # A steadiness gate stood here for a long time and is measurably better on
    # the bar patterns - about 0.11 on 75bars SP and 0.10 at EP against the
    # record reference, and nothing on chromanoise, which has few transitions
    # for it to act on. It is absent by decision: the gate makes the model
    # describe a sub-population rather than the amplitude, and on real content
    # the deviation is then not flat where it should be.
    compensated = sampled_flattened
    sampled_collapse = _predicted_sweep_collapse(rf, carrier_hz, stride)
    if sampled_collapse is not None:
        alive = sampled_collapse > 0.05
        compensated = np.where(
            alive, compensated / np.where(alive, sampled_collapse, 1.0), 0.0
        )
    dense_levels, dense_population, dense_centre_hz = measure_amplitude_by_frequency(
        compensated,
        sampled_carrier_hz,
        sync_tip_hz,
        sys_params["hz_ire"] * RESPONSE_CURVE_RESOLUTION_IRE,
        bin_count,
        bin_count,
    )
    store = rf.__dict__.setdefault("_luma_dense_response", {})
    key = bool(field.isFirstField)
    total, weight, centre_total = store.setdefault(
        key, (np.zeros(bin_count), np.zeros(bin_count), np.zeros(bin_count))
    )
    # Every bin that saw anything contributes, weighted by how much it saw, and
    # the population test is applied to the ACCUMULATED weight rather than to
    # each field's share of it. Testing per field defeats the accumulation: a
    # bin holding thirty samples a field never qualifies, though thirty fields
    # of them would describe it well. That is what starved the ends of the
    # range, where the model then fell back on the line - and the line is a poor
    # description exactly there, because the amplitude arches rather than falls
    # monotonically.
    described = (dense_population > 0.0) & (dense_levels > 0)
    _measure_reliability(rf, key, dense_levels, dense_population, described,
                         total, weight)
    total[described] += np.log(dense_levels[described]) * dense_population[described]
    weight[described] += dense_population[described]
    # The frequency each bin stands at is accumulated the same way its level is,
    # weighted by how many samples went into it. Taking it from THIS field's
    # centres instead - which is what happened - puts a zero wherever a bin
    # accumulated enough from earlier fields but the picture did not visit it
    # this time. The frequencies then stop ascending, `np.interp` reads them as
    # nonsense, and the model swings by 2.5x where the amplitude it describes
    # varies by 1.14x. It bites on material whose levels roam rather than sit at
    # a few discrete bars, which is most real content.
    centre_total[described] += (
        dense_centre_hz[described] * dense_population[described]
    )
    # Only frequencies with real evidence describe themselves, and the test is
    # on the ACCUMULATED weight. Bins the carrier barely touched - the far
    # undershoot below sync tip, the overshoot past peak white - are left to the
    # line, which extrapolates through them along the slope it measured where
    # the picture actually was. Admitting them instead, on whatever handful of
    # samples they hold, stretches the interpolation over uncorrelated
    # excursions and drags the model away from the amplitude it follows.
    #
    # Past that admission there is no second threshold: `_inverse_response_table`
    # blends each admitted point towards the line by its own weight, so a bin
    # with little behind it is mostly line and one with a decode's worth of
    # evidence stands on its own measurement. That is what stops one field's
    # sparse accumulation producing a model that swings 3x where the amplitude
    # it describes varies 1.2x, which is what the early fields of a decode used
    # to get.
    accumulated = weight >= CURVE_MINIMUM_POPULATION
    dense = (
        (
            centre_total[accumulated] / weight[accumulated],
            total[accumulated] / weight[accumulated],
            weight[accumulated],
        )
        if np.count_nonzero(accumulated) >= CURVE_DENSE_MINIMUM_POINTS
        else None
    )

    dropout_fraction = rf.dod_options.dod_threshold_p
    deviation = _deviation_from_inverse_response(
        amplitude,
        centred_hz,
        _inverse_response_table(response, frequency_step, curve, dense),
        np.float32(frequency_step),
        np.float32(dropout_fraction),
        np.float32(1.0 / dropout_fraction),
    )

    if wants_averaging_probe(rf):
        _instrument_averaging(
            rf,
            key,
            curve,
            dense,
            deviation,
            levels,
            population,
            bin_count,
            np.float32(dropout_fraction),
            np.float32(1.0 / dropout_fraction),
        )

    return AmplitudeDeviation(
        deviation,
        carrier_hz,
        steadiness,
        ResponseModel(
            curve,
            residual_hz,
            residual,
            *(dense if dense is not None else ()),
        ),
    )


# How much of the measured path response the equalizer takes out.
#
# There is no single right answer and this is not one. A correction's benefit
# against the amount applied is `2a - a^2 (1 + rho)`, peaking at `1/(1 + rho)`
# and reaching zero at twice that, where rho is the noise-to-signal ratio of
# the estimate driving it. Measured against the record reference, that optimum
# is not a constant of the format: 0.5 on 75bars SP, 1.05 on the same pattern
# at EP, 1.58 on chromanoise - a factor of three.
#
# Nor can it be derived per decode. rho is set by MODEL error, and model error
# is invisible to any resampling of the data the model was fitted from: on this
# very correction, split-half returns rho = 0.00 where the truth is 2.47.
#
# So this is chosen to maximise the WORST case rather than any average, over
# the three conditions where the optimum is known. As a fraction of each
# condition's own best: 0.5 gives 100/75/52%, this gives 97/94/74%, 1.0 gives
# 54/100/88%, and 1.5 costs -146% on 75bars SP, which turns harmful past about
# 1.1. Three quarters is the amount whose weakest showing is strongest, and it
# stays clear of the one condition that has a near harm threshold.
#
# A lead, not yet a law: the optimum ranks in the same order as the fraction of
# the response that is head-specific rather than shared (0.26 -> 0.50,
# 0.32 -> 1.05, 0.52 -> 1.58). Three points, so one chance in six of ordering
# that way by luck. If it holds over more material this constant can be
# replaced by a measurement.
LUMA_EQ_AMOUNT = 0.75


def luma_path_equalizer(rf, curve, amount, dense=None):
    """The inverse of the measured path response, on the RF filter's own grid.

    Despite the name, this does not equalize the response that steady content
    experiences - it cannot. Measured on a synthetic constant-amplitude carrier
    through a known sloped magnitude, the induced frequency error is EXACTLY
    ZERO wherever the carrier is not sweeping: the demodulator is a limiter, so
    an amplitude change on a steady carrier is discarded before it can reach the
    picture. Everything this stage does, it does at transitions, where the
    carrier's sidebands are wide enough for a sloped magnitude to weight them
    asymmetrically and convert some frequency modulation into phase.

    Steady content still moves, but through TIMING rather than through the
    response. Measured on real decode pairs the change concentrates at
    transitions as the synthetic law predicts - transition-span rms 5.3 times
    the steady-span rms, both linear in amount - and yet the steady floor is
    0.09 to 0.16 IRE rms rather than zero. The path is the time base: this stage
    sharpens the sync edges, about 7 and 6 samples down to 6 and 5, the line
    location refinement then lands differently, and the whole line resamples on
    a slightly shifted grid. It is a cross-coupling into timing rather than a
    response the demodulator passed, and it is neither harm nor benefit in
    itself - but it means nothing here is confined to the transitions it acts on.

    That conversion follows the SIGN of the sweep, so a rise and a fall are
    pushed opposite ways - measured at +4.73 kHz against -2.87 kHz on a quarter
    microsecond edge, and falling as roughly edge^-3.4 as edges soften. Being
    linear and time invariant, this stage applies the same magnitude to both and
    so cannot have the right sign for both at once. It improves whichever
    direction it happens to suit and degrades the other, which is why the best
    amount is a property of the PICTURE - the balance of rises and falls in the
    content - rather than of the path, and why no measurement of the path can
    derive it.

    Those two kHz figures are from a BARE sloped magnitude, with neither
    `Filters["RFVideo"]` nor the pre/de-emphasis round trip in the chain, and
    they overstate how one-sided the effect is. Simulated over the whole chain
    against the measured per-head response - a rise and a fall over the same two
    levels, each divided by its own signed step - the part that is linear in
    sweep rate runs 0.29 to 0.48 of the part that RECTIFIES, on cd, pnb and home
    head A, rising toward parity as edges soften. The argument above survives:
    a linear time invariant stage still cannot suit both directions. But the
    asymmetry it trades on is the minority of what the response does to a
    transient, which is worth knowing before reading too much into the amount.

    `Filters["RFVideo"]` is already a frequency domain multiply over the same
    block, so this costs one more multiply of an array that is built once.

    The accumulated dense response is inverted, not just the line. Its
    departure from the line is the path's own shape, measured: across two
    different pictures on the same tape it reproduces at r = +0.87 and +0.89 on
    one head and +0.54 and +0.77 on the other, while across the same picture at
    two tape speeds - a different path - it falls to +0.57 and +0.53. High
    between pictures and lower between paths is the signature this module
    already uses to admit a component.

    The often-quoted +0.24 between pictures is the PER-FIELD residual, which is
    a different object: one field's binned medians against one field's line.
    Accumulating per head across a decode is what separates them, and it is
    worth 0.17 to 0.29 dB rms of real shape.

    Each point is still shrunk toward the line by its own accumulated evidence,
    so a bin the carrier barely visited contributes almost nothing rather than
    contributing noise.

    Held flat outside the range the carrier visits, because a line is not a
    measurement where nothing was measured: at -3 dB/MHz, extrapolated across
    the whole RF band, it would reach tens of dB. Full strength over the
    format's nominal deviation range and tapering to flat across the margin
    either side, which is the region the curve already treats as thinly
    evidenced. The taper is a raised cosine rather than a step because a step
    in a magnitude response rings in the time domain, which is the opposite of
    the point.
    """
    sys_params = rf.SysParams
    inner_low = rf.iretohz(sys_params["vsync_ire"], spec=True)
    inner_high = rf.iretohz(100, spec=True)
    outer_low = rf.iretohz(sys_params["vsync_ire"] - CURVE_RANGE_MARGIN_IRE, spec=True)
    outer_high = rf.iretohz(100 + CURVE_RANGE_MARGIN_IRE, spec=True)

    length = len(rf.Filters["RFVideo"])
    half = length // 2
    frequency = np.arange(half, dtype=np.float64) * (rf.freq_hz / length)

    centre_hz, level, slope = curve
    reference_hz = 0.5 * (inner_low + inner_high)
    log_level = slope * (frequency - reference_hz)
    if dense is not None:
        described_hz, described, described_weight = dense
        inside = (frequency >= described_hz[0]) & (frequency <= described_hz[-1])
        measured = np.interp(frequency[inside], described_hz, described)
        confidence = np.interp(frequency[inside], described_hz, described_weight)
        share = confidence / (confidence + CURVE_MINIMUM_POPULATION)
        # Both are absolute log levels, so the line's own level comes back in
        # before they are mixed, and the result is stated against the middle of
        # the range so the equalizer carries no overall gain.
        line_here = level + slope * (frequency[inside] - centre_hz)
        blended = share * measured + (1.0 - share) * line_here
        log_level[inside] = blended - (level + slope * (reference_hz - centre_hz))

    weight = np.zeros(half)
    weight[(frequency >= inner_low) & (frequency <= inner_high)] = 1.0
    for low, high, rising in ((outer_low, inner_low, True),
                              (inner_high, outer_high, False)):
        edge = (frequency > low) & (frequency < high)
        position = (frequency[edge] - low) / max(high - low, 1.0)
        cosine = np.cos(np.pi * position)
        weight[edge] = 0.5 * (1.0 - cosine) if rising else 0.5 * (1.0 + cosine)

    correction = np.exp(-amount * weight * log_level)
    # The stored filter mirrors the negative frequencies above Nyquist.
    return np.concatenate((correction, np.flip(correction)))[:length]


def update_luma_equalizer(field, amount):
    """Fit the next field's equalizer from this head's accumulated response.

    Fields come off the tape alternating between the two heads, so once this
    field is placed the next one's head is known - which is what makes a
    per-head correction possible at all in a demodulator that runs before
    fields are assembled. A break in the alternation means a field was lost, to
    a gap in the recording or a dropout large enough to swallow one. The
    equalizer holds what it has for that field rather than rebuilding on a
    prediction the signal has just contradicted, and `rf._luma_eq_breaks`
    counts how often that happened - a decode with many breaks is one where
    the head assignment, and so a per-head correction, should not be trusted.

    The response is accumulated per head across the decode. One field
    determines the line well enough (its slope's standard deviation is under
    1% of the slope) but accumulating costs nothing and rides out a bad field.
    """
    measured = measured_amplitude_deviation(field)
    if measured is None or measured.response is None:
        return
    rf = field.rf
    store = rf.__dict__.setdefault("_luma_eq_lines", {})
    head = bool(field.isFirstField)
    centre_hz, level, slope = measured.response.separation
    total, weight = store.get(head, (np.zeros(3), 0.0))
    store[head] = (total + np.array([centre_hz, level, slope]), weight + 1.0)

    expected = rf.__dict__.get("_luma_eq_expected")
    rf._luma_eq_expected = not head
    if expected is not None and expected != head:
        # The alternation just failed, so the basis for naming the next
        # field's head has failed with it. Whatever filter is in place
        # stays there for this field rather than being rebuilt on a
        # prediction the signal has just contradicted.
        rf._luma_eq_breaks = rf.__dict__.get("_luma_eq_breaks", 0) + 1
        return

    # Both heads pooled, deliberately, rather than the head the next field
    # will come from.
    #
    # The demodulator runs in its own thread and ahead of field assembly, so a
    # filter swapped in here reaches whichever blocks that thread happens not
    # to have started yet. With two different filters alternating, which one a
    # block gets is a matter of scheduling, and the decode stops being
    # reproducible - measured, two identical runs produced different output
    # and the correction's size moved by about a tenth of itself.
    #
    # Pooling costs nothing measurable: the two heads' lines differ by about
    # 7%, and a per-head filter was measured to give the same result as a
    # pooled one to within the noise. What is left varying is successive
    # versions of one filter as the response converges, which is a far smaller
    # difference than between two heads.
    total = sum(entry[0] for entry in store.values())
    weight = sum(entry[1] for entry in store.values())
    # Held back where the response cannot be trusted. On a tape whose
    # measurement is noise-dominated an equalizer built from it was measured
    # to make the picture worse at every amount, so the correction is scaled
    # by how well the response describes itself. The weaker of the two heads
    # decides, because one pooled filter is applied to both.
    trust = rf.__dict__.get("_luma_eq_trust", 1.0)
    rf.Filters["LumaPathEQ"] = luma_path_equalizer(
        rf,
        tuple(total / weight),
        amount * trust,
        _pooled_dense(rf),
    ).astype(np.float64)



def _measure_reliability(rf, head, levels, population, seen, total, weight):
    """How far this head's response can be trusted, from the data alone.

    Two failures have to be caught and neither statistic catches both.

    A head whose response is noise rather than shape shows it in the SLOPE of
    each new field regressed on the model built without it: consistent shape
    regresses at one, noise regresses low. And a head whose response is real but
    poorly determined shows it in the SCATTER about that regression, which is
    the measurement's own noise against the shape it is trying to describe.
    Measured on the one tape where an equalizer built from this was found to
    harm the picture, the slope caught one head (0.28) and the scatter caught
    the other (a noise-to-signal of 0.32); on four tapes where it helped, both
    read unity and near zero.

    The noise floor is pooled across the heads because it is a property of the
    RF chain rather than of either head, so both estimates rest on twice the
    evidence.

    Leaves the factor on the decoder as `_luma_eq_trust` - one where the
    response is solid, below one where it is not.
    """
    probe = rf.__dict__.setdefault("_luma_eq_reliability", {})
    state = probe.setdefault(head, {"num": 0.0, "den": 0.0, "resid": 0.0, "weight": 0.0})
    prior = weight >= CURVE_MINIMUM_POPULATION
    both = prior & seen
    if np.count_nonzero(both) >= CURVE_INDEPENDENT_POINTS * 2:
        model = total[both] / weight[both]
        observed = np.log(np.maximum(levels[both], 1e-30))
        mass = population[both]
        model = model - np.average(model, weights=mass)
        observed = observed - np.average(observed, weights=mass)
        state["num"] += float(np.sum(mass * model * observed))
        state["den"] += float(np.sum(mass * model * model))
        slope = state["num"] / max(state["den"], 1e-30)
        departure = observed - slope * model
        state["resid"] += float(np.sum(mass * departure * departure))
        state["weight"] += float(np.sum(mass))
    if state["den"] <= 0.0 or state["weight"] <= 0.0:
        return
    slope = state["num"] / state["den"]
    floor = sum(s["resid"] for s in probe.values()) / max(
        sum(s["weight"] for s in probe.values()), 1e-30
    )
    signal = state["den"] / state["weight"]
    noise_to_signal = floor / max(signal, 1e-30)
    state["trust"] = float(np.clip(slope / (1.0 + noise_to_signal), 0.0, 1.0))
    rf._luma_eq_trust = min(
        entry.get("trust", 1.0) for entry in probe.values()
    )

def _pooled_dense(rf):
    """Both heads' accumulated responses together, on the model's contract."""
    store = rf.__dict__.get("_luma_dense_response")
    if not store:
        return None
    total = sum(entry[0] for entry in store.values())
    weight = sum(entry[1] for entry in store.values())
    centre_total = sum(entry[2] for entry in store.values())
    described = weight >= CURVE_MINIMUM_POPULATION
    if np.count_nonzero(described) < CURVE_DENSE_MINIMUM_POINTS:
        return None
    return (
        centre_total[described] / weight[described],
        total[described] / weight[described],
        weight[described],
    )

def attach_luma_deviation(field):
    """Carry the carrier's amplitude deviation as a video channel.

    An FM carrier is recorded at constant amplitude, so its departure from
    constant is the path's doing and not the signal's - which makes this a
    measurement of the path taken through a known-flat input, available
    wherever the carrier is rather than only where a reference level is.

    The channel is the measured amplitude over the amplitude the carrier's own
    instantaneous frequency predicts, on the RF sample grid, so unity means the
    carrier sits where the model puts it. The model it was measured against is
    left on the field as `luma_response`, in components rather than as a table.

    Nothing in the decoder reads either. Returns whether it attached anything.
    """
    measured = measured_amplitude_deviation(field)
    if measured is None:
        return False
    field.data["video"]["luma_deviation"] = measured.deviation
    field.luma_response = measured.response
    return True


def measured_amplitude_deviation(field):
    """`measure_amplitude_deviation`, computed once per field.

    Two consumers want the same measurement and it walks the field several
    times, so the second one reads the first one's answer.
    """
    cached = getattr(field, "_luma_amplitude_measured", None)
    if cached is None:
        cached = measure_amplitude_deviation(field)
        field._luma_amplitude_measured = cached
    return cached


