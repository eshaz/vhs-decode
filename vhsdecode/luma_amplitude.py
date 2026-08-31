"""Luma FM carrier amplitude variation.

An FM carrier carries no amplitude information - its amplitude should be
constant. Every deviation from that constant is imposed by the tape, and is
measurable on the raw RF, before the time base correction has run and before
any scaling has been applied.

The amplitude is taken as the decoder's own envelope channel, which is
band limited to 700 kHz by the demodulator's envelope detector - a single pole
applied forward and backward, so 12 dB per octave. Nothing here narrows it
further: no smoothing, no additional band limit, no windowing. The tape imposes
its amplitude noise at whatever rate it imposes it, and a filter here would
decide in advance which of that the correction is allowed to see.

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
# path to follow. `noise_scale` is what a consumer's own Wiener weight should be
# multiplied by at this sample: the measurement is not equally noisy across the
# carrier's range, and this carries how far it departs from its average there.
#
# Both are single precision, which is what the demodulator produced them in;
# widening them here would cost two full copies of the field per field and buy
# no precision that was ever recorded.
AmplitudeDeviation = namedtuple(
    "AmplitudeDeviation",
    ["deviation", "carrier_hz", "steadiness", "noise_scale", "response"],
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
ResponseModel = namedtuple(
    "ResponseModel",
    ["separation", "residual_hz", "residual", "residual_population",
     "described_hz", "described"],
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


# Stands in for the noise profile when nothing has asked for it, so the
# per-sample loop needs no second code path.
_UNIT_TABLE = np.ones(1, dtype=np.float32)

# How much of the field is steady enough to describe the response.
#
# A sweeping carrier's amplitude has collapsed through the path's own band
# limit rather than through anything the path does at that frequency, and it
# lands in whatever level bin the transition happened to cross. Left in, it is
# what makes a dense fit fail: measured against the record reference, a dense
# per-head response fitted from everything costs +0.056 on 75bars, and the same
# fit gated here gains -0.083. The gate is what separates the two.
#
# The value is empirical, and far tighter than the sweep loss alone would ask
# for - a threshold set where the transient's contribution reaches the
# deviation's own noise would keep most of the field - so the gate is evidently
# rejecting samples whose binning is unreliable for other reasons too. At this
# fraction some forty bins and ten thousand samples still describe the
# response.
# Expressed as the fraction of the field that survives, not as a steadiness
# value, so it means the same thing at every tape speed. An absolute threshold
# does not: EP's narrower RF passband gives a smaller knee, so the same number
# rejects nearly everything there and starves the fit entirely.
CURVE_DENSE_SAMPLE_FRACTION = 0.16

# Points the accumulated response needs before it stands in for the line.
CURVE_DENSE_MINIMUM_POINTS = 8

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

    Anchored at sync tip - the format's own fixed reference, the one level that
    is defined rather than pictorial - and taken from the decoder's running
    parameters, so the anchor sits at the sync tip the signal actually has
    rather than the one the specification nominates.

    Samples outside this range are dropped rather than clipped into the end
    bins; see `measure_amplitude_by_frequency`. The range itself stays fixed, so
    the curve means the same thing from one field to the next - letting it
    follow the highest sample instead lets a single demodulator excursion set
    the axis, and the fit is then conditioned differently in every field.
    """
    sync_tip_hz = rf.iretohz(sys_params["vsync_ire"] - CURVE_RANGE_MARGIN_IRE)
    peak_white_hz = rf.iretohz(100 + CURVE_RANGE_MARGIN_IRE)
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
    -0.4 per MHz on this deck. The binned medians are what the line is fitted
    through, a median because amplitude noise is not symmetric and dropouts sit
    in the low tail, so it tracks the level the carrier normally holds at that
    frequency rather than the average of level and damage.

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
        # Inside the frequencies actually measured, the accumulated response
        # itself; outside them the line carries on, as it must.
        described_hz, described = dense
        inside = (grid_hz >= described_hz[0]) & (grid_hz <= described_hz[-1])
        log_level[inside] = np.interp(grid_hz[inside], described_hz, described)

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
    amplitude, carrier_hz, inverse_response, noise_table, frequency_step, low, high
):
    """How far the amplitude departs from the response modelled for it, and how
    noisy the measurement is where it was taken.

    One pass over the field, and no transcendental in it: the modelled response
    was inverted where it was tabulated, so what is left per sample is two
    interpolations off the same index, a multiply and a bound.
    """
    count = len(amplitude)
    deviation = np.empty(count, dtype=np.float32)
    noise_scale = np.empty(count, dtype=np.float32)
    last = len(inverse_response) - 1
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for i in range(count):
        position = carrier_hz[i] / frequency_step
        if position <= zero:
            inverse = inverse_response[0]
            scale = noise_table[0]
        elif position >= last:
            inverse = inverse_response[last]
            scale = noise_table[len(noise_table) - 1]
        else:
            lower = np.int32(position)
            fraction = position - np.float32(lower)
            base = inverse_response[lower]
            inverse = base + (inverse_response[lower + 1] - base) * fraction
            if lower + 1 < len(noise_table):
                base = noise_table[lower]
                scale = base + (noise_table[lower + 1] - base) * fraction
            else:
                scale = noise_table[len(noise_table) - 1]
        ratio = amplitude[i] * inverse
        if ratio <= zero:
            # no measurement here, or no usable model - leave the sample alone
            ratio = one
        elif ratio < low:
            ratio = low
        elif ratio > high:
            ratio = high
        deviation[i] = ratio
        noise_scale[i] = scale
    return deviation, noise_scale


def _noise_scale_table(rf, sampled_flattened, sampled_carrier_hz, curve, frequency_step, points):
    """How the measurement's own noise varies across the carrier's range.

    The track-width Wiener weight already says how much of the luma's deviation
    belongs on the color-under, but as one number for the whole sweep. The
    measurement is not equally noisy across it: taken as the robust spread of
    the deviation in each frequency bin, it varies by 1.4 to 2.2 times within a
    single decode.

    That variation is the path's and not the picture's, which is what makes it
    safe to act on where the residual mean is not. Measured across two patterns
    and two tape speeds, the profile's shape agrees at r = +0.81 between
    different pictures on the same tape, and only +0.31 between the same picture
    at two speeds - the exact reverse of the residual, which follows the picture.

    Returned as the factor the trust weight scales by. Writing the weight as
    S / (S + N) with S the shared loss and N the measurement's noise, a global
    trust fixes S = trust * sigma_bar^2, so at a frequency where the spread is
    sigma the weight becomes trust * (sigma_bar / sigma)^2. Nothing is fitted:
    the spread is measured and the rest is the weight the module already uses.
    """
    total = getattr(rf, "_luma_noise_total", None)
    if total is None or len(total) != points:
        total = np.zeros(points)
        rf._luma_noise_total = total
        rf._luma_noise_weight = np.zeros(points)
    weight = rf._luma_noise_weight

    centre, level, slope = curve
    keep = sampled_flattened > 0.0
    if np.count_nonzero(keep) > 256:
        residual = np.log(sampled_flattened[keep]) - (
            level + slope * (sampled_carrier_hz[keep] - centre)
        )
        index = np.clip(
            (sampled_carrier_hz[keep] / frequency_step).astype(np.int64), 0, points - 1
        )
        order = np.argsort(index)
        index, residual = index[order], residual[order]
        edges = np.flatnonzero(np.diff(index)) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(index)]):
            if hi - lo < 64:
                continue
            segment = residual[lo:hi]
            spread = np.median(np.abs(segment - np.median(segment))) * 1.4826
            if spread > 0.0:
                total[index[lo]] += spread * (hi - lo)
                weight[index[lo]] += hi - lo

    seen = weight > 0.0
    if np.count_nonzero(seen) < 8:
        return np.ones(points, dtype=np.float32)
    profile = total[seen] / weight[seen]
    mean_spread = float((profile * weight[seen]).sum() / weight[seen].sum())
    grid = np.arange(points, dtype=np.float64)
    spread = np.interp(grid, grid[seen], profile)
    return ((mean_spread * mean_spread) / (spread * spread)).astype(np.float32)



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


def _predicted_sweep_collapse(rf, carrier_hz):
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

    Worth -0.0122 on the record-referenced 75bars SP metric and -0.0125 at EP
    against gating alone, both resolved; +0.0005 on chromanoise, which has few
    transients for it to act on. Removing the collapse does NOT make the gate
    redundant - the prediction is a model of the path, not a measurement of this
    field, so the steadiest samples are still the ones worth describing the
    response with. Correcting without gating is measurably worse than gating
    without correcting (-0.036 against -0.125 on 75bars SP).
    """
    magnitude = np.abs(rf.Filters["RFVideo"])
    half = len(magnitude) // 2
    band = magnitude[:half]
    step = rf.freq_hz / len(magnitude)
    count = len(carrier_hz)

    cached = getattr(rf, "_luma_sweep_response", None)
    if cached is None or len(cached) != count // 2 + 1:
        cached = np.interp(np.fft.rfftfreq(count, 1.0 / rf.freq_hz),
                           np.arange(half) * step, band, left=band[0], right=0.0)
        rf._luma_sweep_response = cached

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
    predicted = np.abs(analytic)

    static = _static_response(
        np.asarray(carrier_hz, dtype=np.float64), band, step, np.empty(count)
    )
    alive = static > 0.0
    collapse = np.zeros(count)
    np.divide(predicted, static, out=collapse, where=alive)
    reference = np.median(collapse[alive]) if alive.any() else 0.0
    return collapse / reference if reference > 0 else None


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

    # Nothing in the decoder reads the noise profile, and building it costs a
    # sort and a robust spread per bin, so it is measured only when something
    # has asked to see it.
    wants_noise_scale = bool(getattr(rf.options, "luma_deviation", False))
    # The response read off the measurement itself, accumulated per head across
    # the decode. A single field determines it far too poorly - that is why
    # fitting densely per field loses - but a decode's worth of one head's
    # fields determines it well, and the two heads genuinely differ.
    #
    # Binned again, from the steadiest samples only. The line above keeps every
    # sample: it has two parameters, a transient barely moves it, and it must
    # never fail to fit or there is no correction at all. The dense response is
    # the opposite - it follows whatever it is shown, so it is shown only
    # carriers that are barely moving.
    steady_sample = steadiness[::stride]
    threshold = np.quantile(steady_sample, 1.0 - CURVE_DENSE_SAMPLE_FRACTION)
    gated = sampled_flattened * (steady_sample >= threshold)
    # The sweep collapse taken off first, so what the gate then selects on is a
    # carrier already compensated for moving.
    collapse = _predicted_sweep_collapse(rf, carrier_hz)
    if collapse is not None:
        sampled_collapse = collapse[::stride]
        alive = sampled_collapse > 0.05
        gated = np.where(alive, gated / np.where(alive, sampled_collapse, 1.0), 0.0)
    steady_levels, steady_population, steady_centre_hz = measure_amplitude_by_frequency(
        gated,
        sampled_carrier_hz,
        sync_tip_hz,
        sys_params["hz_ire"] * RESPONSE_CURVE_RESOLUTION_IRE,
        bin_count,
        bin_count,
    )
    store = rf.__dict__.setdefault("_luma_dense_response", {})
    key = bool(field.isFirstField)
    total, weight = store.setdefault(key, (np.zeros(bin_count), np.zeros(bin_count)))
    # Every bin that saw anything contributes, weighted by how much it saw, and
    # the population test is applied to the ACCUMULATED weight rather than to
    # each field's share of it. Testing per field defeats the accumulation: a
    # bin holding thirty steady samples a field never qualifies, though thirty
    # fields of them would describe it well. That is what starved the ends of
    # the range, where the model then fell back on the line - and the line is a
    # poor description exactly there, because the amplitude arches rather than
    # falls monotonically.
    described = (steady_population > 0.0) & (steady_levels > 0)
    total[described] += np.log(steady_levels[described]) * steady_population[described]
    weight[described] += steady_population[described]
    accumulated = weight >= CURVE_MINIMUM_POPULATION
    dense = (
        (steady_centre_hz[accumulated], total[accumulated] / weight[accumulated])
        if np.count_nonzero(accumulated) >= CURVE_DENSE_MINIMUM_POINTS
        else None
    )

    dropout_fraction = rf.dod_options.dod_threshold_p
    deviation, noise_scale = _deviation_from_inverse_response(
        amplitude,
        centred_hz,
        _inverse_response_table(response, frequency_step, curve, dense),
        _noise_scale_table(
            rf,
            sampled_flattened,
            sampled_carrier_hz,
            curve,
            frequency_step,
            len(response),
        )
        if wants_noise_scale
        else _UNIT_TABLE,
        np.float32(frequency_step),
        np.float32(dropout_fraction),
        np.float32(1.0 / dropout_fraction),
    )

    return AmplitudeDeviation(
        deviation,
        carrier_hz,
        steadiness,
        noise_scale,
        ResponseModel(
            curve,
            residual_hz,
            residual,
            population[measured],
            *(dense if dense is not None else (np.zeros(0), np.zeros(0))),
        ),
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
