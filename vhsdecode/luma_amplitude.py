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
measurement rather than at whatever acts on it. One equation carries all of
it: per sample, the measured amplitude factors as

    A(t)  =  S(f(t)) * C(t) * R(f(t)) * n(t)

with f the carrier's instantaneous frequency. S is the decoder's own static
RF response, known exactly - on formats using the linear ramp boost it tilts
the band by several dB across the carrier sweep on its own. C is the sweep
collapse: the amplitude a carrier loses purely by MOVING, predicted from the
same filter. (That prediction is quasi-stationary FM analysis; the modelled
response is zero phase, so the series' first-order term is pure quadrature
and the magnitude loss is second order in the sweep - which is why it acts
near the one per cent scale and concentrates at transitions.) R is what
remains - tape channel, head response, record and playback equalization -
measured by binning the flattened amplitude against carrier frequency, per
head. And n is the tape's multiplicative noise, the deliverable: the
deviation solves the equation for n, and the luma equalizer applies R to the
RF with its sign reversed. Downstream the factors are no longer separable.

S and R are functions of the carrier's frequency alone, so they are combined
into tables over frequency and divided out together, once per sample; C rides
along as the synthesized magnitude with S cancelled (see `_response_tables`).
That is what keeps this affordable: the response is exponentiated where it is
described, over the few thousand points of the table, rather than over the
million samples it is applied to.

What comes back is a property of the luma carrier and of the tape, and of
nothing else. It carries no assumption about what will consume it.
"""

from collections import namedtuple

import numpy as np
from numba import njit

import scipy.fft as sps_fft


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
# What the binned medians still departed from that line by - the RESIDUAL - was
# carried here as two more fields and is not, because nothing ever read them.
# The measurement behind that decision is worth keeping, because the residual
# looks like the rest of the path and is NOT: it is a function of the picture.
# The record capture reproduces the same residual for the same picture at
# r = +0.89 with no tape involved at all, while two record
# captures of DIFFERENT pictures through the same electronics anticorrelate at
# r = -0.29. What it measures is how the picture's own transitions populate the
# level bins - a sweeping carrier's amplitude has collapsed through the path's
# band limit and lands in whatever bin the transition crossed.
#
# So do not fit it, spline it, or weigh by it. Measured, every use loses:
# subtracted from the model it costs +0.0595 on the record-referenced 75bars
# metric, weighed against the deviation as a Wiener confidence +0.0041. That is
# the same reason nothing freer than a line survives here - every degree of
# freedom past the line absorbs picture and hands it to whatever consumes the
# deviation. Recompute it if you want to look at it; it is four lines.
#
# `described_hz` and `described` are the accumulated dense response actually in
# force - every sample, pooled per head across the decode. (A steadiness gate
# stood here once and is gone; see `measure_amplitude_deviation`.) It is
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
    "separation",
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
    to the whole field is `_deviation_from_collapse_magnitude`, which divides
    out this response and the fitted curve together.
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

    Everything the signal passed before the carrier amplitude was taken:
    `Filters["RFVideo"]` - which already carries the format's band, the ramp
    boost and the peaking options - and, when `--notch` is in use, the video
    notch, which the demodulator applies to the block right before it. The
    notch is a separate filter rather than part of `RFVideo`, and leaving it
    out here books its dip into the measured tape response - from where the
    equalizer would faithfully re-boost the interference the notch exists to
    remove.

    Fixed for the life of the decoder, so it is derived once and kept on the
    decoder rather than rebuilt every field.
    """
    cached = getattr(rf, "_luma_amplitude_response", None)
    if cached is not None:
        return cached

    response = np.abs(rf.Filters["RFVideo"])
    notch = rf.Filters.get("FVideoNotchF")
    if notch is not None:
        # Already a magnitude, tabulated over the same block.
        response = response * notch
    # The stored response covers the whole spectrum with the negative
    # frequencies mirrored above Nyquist; only the positive half is meaningful.
    half = len(response) // 2
    cached = (response[:half], rf.freq_hz / len(response))
    rf._luma_amplitude_response = cached
    return cached


# The factors of the model, as tables. `A = S*C*R*n`: S is the static
# response `rf_path_response` returns; the tables below carry C - the sweep
# collapse - as coefficient curves over frequency, so that C becomes a local
# formula of the carrier's own trajectory instead of a synthesized signal.
_SweepTables = namedtuple(
    "_SweepTables",
    [
        # per-frequency coefficient tables, float64 masters and float32 casts
        "l1", "q", "q2", "rho", "ceiling",
        "l1_32", "q_32", "q2_32", "rho_32", "ceiling_32",
        # trajectory smoothing scales, in samples
        "span", "track_width",
    ],
)


def sweep_collapse_tables(rf):
    """Coefficient tables of the closed-form sweep collapse, from the filter.

    The collapse of a moving carrier is a local property of the band's shape
    at the carrier's own frequency, and every coefficient here is a
    log-derivative of that shape:

        track   L1 = d ln H / df          first-order tracking
        q       (L2^2 floored) fs^2/4pi^2 the resummation's theta^2 per
                                          (Hz/sample)^2 of sweep rate
        q2      -(H''/H) fs / 4pi         the quadrature (phase) coefficient
        rho     L1^2 / (2 sqrt(Qcurv))    the regain exponent's scale
        ceiling Hmax / H(f)               the physical bound: a swept
                                          envelope cannot exceed the band's
                                          peak response

    The floor under q is the band's own energy-weighted variance - on the
    log-flat stretch of a ramp-boosted band L2 passes through zero, and the
    global width is what enforces collapse there.

    Built from the RF band WITHOUT the video notch, deliberately, on the same
    grounds as `path_transient_scale`: a high-Q notch's log-derivatives would
    predict absurd collapse over a feature a few kilohertz wide, while its
    real effect on a sweeping carrier is confined to the moment the carrier
    crosses it. The static response S keeps the notch; C's coefficients must
    not.

    `track_width` is the window whose variance matches the filter's own
    impulse magnitude over its 2-98% energy support: the envelope at an
    instant is formed over that support, so the frequency it responds to is
    the carrier averaged there - which is also what makes the estimator quiet
    against per-sample carrier noise. The window realizes the series' odd
    (sweep-acceleration) term as well: a centered mean minus the centre IS a
    second difference, and regressing the residual against the explicit
    series term measures what is left of it at +0.06 of its coefficient
    (b75; the window carries the rest).

    Fixed for the life of the decoder; cached like `rf_path_response`.

    The synthesized predecessor and both closed forms are measured history,
    kept so no choice is reopened on a guess. The synthesis - a unit carrier
    on the measured trajectory through the band, transform pair at full rate,
    ~21 ms a field - was exact for the zero-phase model and needed a median
    (measured 1.0003) to state its own unity; it also had to divide the
    static response back out at the sample's own frequency, without which
    |RFVideo|'s span (0.157 at 3.1 MHz against 0.268 at 3.9) is applied
    twice, 4.6 dB at the bottom of the range. A RAW pointwise series was
    rejected first (quasi-static p50 2.8e-2 from differentiating trajectory
    noise; settling p99 7.5e-1 from having no memory). THIS form - the same
    series with the sweep rate read at the filter's own support, resummed,
    ceiling-bounded - measures against the synthesis at quasi-static p50
    2.5e-3 / 5.9e-3 / 4.4e-3 and post-clamp deviation added-variance
    2.9 / 4.1 / 2.2 per cent on 75bars SP / chromanoise / 75bars EP; what it
    does not reproduce is the synthesis's response to its own trajectory
    noise, and whether that share was signal is what the record-referenced
    decode comparison judges.
    """
    cached = getattr(rf, "_luma_sweep_tables", None)
    if cached is not None:
        return cached

    raw = np.abs(rf.Filters["RFVideo"])
    half = len(raw) // 2
    band = raw[:half]
    step = rf.freq_hz / len(raw)

    mask = band > band.max() * 1e-6
    log_band = np.where(mask, np.log(np.maximum(band, 1e-300)), 0.0)
    l1 = np.gradient(log_band, step)
    l2 = np.gradient(l1, step)
    # the finite-difference stencil widens each derivative's invalid margin
    erode = mask.copy()
    for k in (1, 2, 3):
        erode[k:] &= mask[:-k]
        erode[:-k] &= mask[k:]
    l1 = np.where(erode, l1, 0.0)
    l2 = np.where(erode, l2, 0.0)
    h2_over_h = np.where(erode, l2 + l1 * l1, 0.0)

    grid = np.arange(half, dtype=np.float64) * step
    weight = band * band
    centre = (weight * grid).sum() / weight.sum()
    band_variance = (weight * (grid - centre) ** 2).sum() / weight.sum()

    curvature = np.maximum(l2 * l2, 1.0 / band_variance**2)
    q = curvature * (rf.freq_hz**2) / (4.0 * np.pi**2)
    q2 = -h2_over_h * rf.freq_hz / (4.0 * np.pi)
    rho = (l1 * l1) / (2.0 * np.sqrt(curvature))
    ceiling = np.where(mask, band.max() / np.maximum(band, 1e-300), 1.0)

    knee, span = path_transient_scale(rf)
    impulse = np.abs(np.fft.ifft(rf.Filters["RFVideo"]))
    impulse = np.roll(impulse, len(impulse) // 2)
    energy = np.cumsum(impulse**2)
    energy /= energy[-1]
    low = int(np.searchsorted(energy, 0.02))
    high = int(np.searchsorted(energy, 0.98))
    segment = impulse[low:high]
    offsets = np.arange(len(segment)) - (len(segment) - 1) / 2.0
    variance = float((segment * offsets**2).sum() / segment.sum())
    track_width = max(int(round(np.sqrt(12.0 * variance + 1.0))) | 1, 3)

    cached = _SweepTables(
        l1, q, q2, rho, ceiling,
        l1.astype(np.float32), q.astype(np.float32), q2.astype(np.float32),
        rho.astype(np.float32), ceiling.astype(np.float32),
        int(span), int(track_width),
    )
    rf._luma_sweep_tables = cached
    return cached


@njit(cache=True, nogil=True, fastmath=True)
def _sweep_arrays(carrier_hz, span, track_width, track_arg, slope):
    """The carrier trajectory's two smoothed channels, one pass each.

    `track_arg[i]` = (centered mean over `track_width`) - carrier[i]: what the
    envelope-forming support saw minus the instant's own frequency.
    `slope[i]` = the centered mean slope over `span`, in Hz per sample - the
    sweep rate at the scale the path settles on. Both accumulate in double
    and narrow on the way out; edges replicate.

    The window is the variance-matched boxcar, and the hsync-witnessed
    attempt to better it is measured history: folding the arm difference
    against the exact synthesis on refined per-line sync edges, the
    filter's own tau-centred |h| window moved the fall/rise coherent
    difference from 24.2/24.4 mNp only to 23.7/23.1, a shifted window's
    apparent 15.6 was an alignment artifact (a half-tap centroid error
    reads as ~18 mNp of spurious edge track), and shift sweeps improve
    BOTH directions off centre - the signature of the surviving structure
    being the zero-phase channel's pre-ring and settling oscillation,
    convolution memory no pointwise window can carry.
    """
    count = len(carrier_hz)
    half_w = track_width // 2
    running = 0.0
    for i in range(track_width):
        running += carrier_hz[i if i < count else count - 1]
    for i in range(count):
        low = i - half_w
        high = i + half_w
        if low < 0 or high >= count:
            total = 0.0
            for j in range(max(low, 0), min(high, count - 1) + 1):
                total += carrier_hz[j]
            width = min(high, count - 1) - max(low, 0) + 1
            track_arg[i] = np.float32(total / width - carrier_hz[i])
        else:
            if i == half_w:
                running = 0.0
                for j in range(track_width):
                    running += carrier_hz[j]
            elif i > half_w:
                running += carrier_hz[high] - carrier_hz[low - 1]
            track_arg[i] = np.float32(running / track_width - carrier_hz[i])
    half_s = span // 2 if span // 2 > 0 else 1
    for i in range(count):
        low = i - half_s
        high = i + half_s
        if low < 0:
            low = 0
        if high > count - 1:
            high = count - 1
        if high > low:
            slope[i] = np.float32(
                (carrier_hz[high] - carrier_hz[low]) / (high - low)
            )
        else:
            slope[i] = np.float32(0.0)
    return track_arg, slope


@njit(cache=True, nogil=True, fastmath=True)
def _collapse_at(l1, q, q2, rho, ceiling, position, fraction, lower, upper,
                 track, slope):
    """The closed-form collapse at one resolved table index. See
    `sweep_collapse_tables` for the terms; `_deviation_from_response` and the
    strided `_collapse_closed` both stand on this so the two paths cannot
    drift apart."""
    a = l1[lower]
    l1_v = a + (l1[upper] - a) * fraction
    a = q[lower]
    q_v = a + (q[upper] - a) * fraction
    a = q2[lower]
    q2_v = a + (q2[upper] - a) * fraction
    a = rho[lower]
    rho_v = a + (rho[upper] - a) * fraction
    a = ceiling[lower]
    ceil_v = a + (ceiling[upper] - a) * fraction
    theta2 = slope * slope * q_v
    damp = 1.0 / (1.0 + theta2)
    real = 1.0 + l1_v * track * damp
    if real < 0.05:
        real = 0.05
    imag = q2_v * slope * damp
    collapse = np.sqrt(real * real + imag * imag)
    collapse *= np.sqrt(np.sqrt(damp))
    collapse *= np.exp(rho_v * theta2 * damp)
    if collapse > ceil_v:
        collapse = ceil_v
    # The phase is the same complex number read the other way: the even and
    # ceiling factors are positive reals and cannot turn it.
    return collapse, np.arctan2(imag, real)


@njit(cache=True, nogil=True, fastmath=True)
def _collapse_closed(carrier_hz, track_arg, slope, l1, q, q2, rho, ceiling,
                     frequency_step, out):
    """The collapse on a sample set, float64 tables, float32 out - the strided
    companion of the deviation kernel, for the compensated binning."""
    last = len(l1) - 1
    for i in range(len(carrier_hz)):
        position = carrier_hz[i] / frequency_step
        if position <= 0.0:
            lower = 0
            upper = 0
            fraction = 0.0
        elif position >= last:
            lower = last
            upper = last
            fraction = 0.0
        else:
            lower = int(position)
            upper = lower + 1
            fraction = position - lower
        value, _ = _collapse_at(
            l1, q, q2, rho, ceiling, position, fraction, lower, upper,
            np.float64(track_arg[i]), np.float64(slope[i]),
        )
        out[i] = np.float32(value)
    return out


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

    The video notch, when active, is deliberately NOT part of this scale even
    though `rf_path_response` models it: a high-Q notch's impulse response
    rings far longer than the band's own settling, and folding it in would
    stretch the measured span - and depress the steadiness - across the whole
    band, for a feature a few kilohertz wide.
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

    Samples below this range are dropped; above it they pool into the top
    bin - see `_frequency_bin_of`.
    """
    sync_tip_hz = rf.iretohz(
        sys_params["vsync_ire"] - CURVE_RANGE_MARGIN_IRE, spec=True
    )
    peak_white_hz = rf.iretohz(100 + CURVE_RANGE_MARGIN_IRE, spec=True)
    span_ire = (peak_white_hz - sync_tip_hz) / sys_params["hz_ire"]
    bin_count = int(round(span_ire / RESPONSE_CURVE_RESOLUTION_IRE))
    return sync_tip_hz, peak_white_hz, max(bin_count, CURVE_INDEPENDENT_POINTS)


@njit(cache=True, nogil=True, fastmath=True)
def _frequency_bin_of(carrier_hz, sync_tip_hz, hz_ire, bin_count):
    """The per-IRE bin each sample's carrier frequency lands in, values unseen.

    Computed once per field and read by both binnings: which bin a sample
    belongs to depends only on its frequency, while which samples COUNT
    depends on the values being binned - so the index is shared and the
    admission is `_binned_median`'s own.

    Samples outside the range are dropped at BOTH ends - and the top end's
    history is the caution this docstring exists to carry. For a long time
    samples above the range were pooled into the top bin instead: a real,
    admitted bias (0.9% of samples averaging 5.26 MHz booked into a bin near
    4.8, steepening the slope by ~+0.039 /MHz) that was kept because
    dropping them cost +0.085 on the record-referenced 75bars metric - in
    the era when the sweep collapse was synthesized OUTSIDE the line's model,
    and the biased slope evidently cancelled part of the transients'
    contribution to the deviation. An unprincipled crutch, recorded so it
    would not be silently "fixed".

    Re-adjudicated non-silently once the closed-form collapse entered the
    expectation: with C modelled explicitly the crutch's job is gone -
    dropping the pooling reads NEUTRAL on the record-referenced metric on
    all three conditions (-0.0003/+0.0000/-0.0018 paired, the last at its
    2-sigma edge in favour) while the transition-domain model error, folded
    at the hsync witness, falls 16% at the sync fall and 8% at the rise, and
    the bar-plateau structure improves. The slope shallows from -0.385 to
    -0.366 /MHz, matching the recorded bias magnitude.
    """
    count = len(carrier_hz)
    index = np.full(count, -1, dtype=np.int64)
    for i in range(count):
        bin_number = int(np.floor((carrier_hz[i] - sync_tip_hz) / hz_ire))
        if bin_number < 0:
            continue
        if bin_number >= bin_count:
            # Dropped, like the low end - see above for the pooling era.
            continue
        index[i] = bin_number
    return index


@njit(cache=True, nogil=True, fastmath=True)
def _binned_median(values, index, sync_tip_hz, hz_ire, bin_count):
    """Carrier amplitude against carrier frequency, as points on a curve.

    One point per bin: the frequency the bin's samples average to, the median
    amplitude in it, and how many samples it holds. The median is what makes
    the estimate robust - amplitude noise is not symmetric, dropouts sit in
    the low tail, so a median tracks the level the carrier normally has at
    that frequency rather than the average of level and damage.

    Only samples carrying a measurement count: zero is the sentinel every
    stage upstream writes for "nothing usable here", and such samples keep
    their bin in `index` but contribute nothing. The bins are equal-width
    across the deviation range, so the points the curve is interpolated
    through stay evenly spaced. Bins the picture barely visits come back with
    little population and the caller drops them; the interpolation simply
    spans the gap, which is a better answer than a point placed on a handful
    of samples.

    The caller passes a subsample of the field, already gathered; the curve
    has three parameters and even a subsample carries tens of thousands of
    points, so it is not short of evidence. The correction itself still
    applies to every sample.
    """
    count = len(values)
    fine = np.zeros(bin_count, dtype=np.int64)
    for i in range(count):
        if values[i] <= 0.0:
            continue
        b = index[i]
        if b >= 0:
            fine[b] += 1

    total = 0
    for b in range(bin_count):
        total += fine[b]
    if total <= 0:
        return (
            np.zeros(0), np.zeros(0), np.zeros(0)
        )

    population = np.zeros(bin_count)
    centre = np.zeros(bin_count)
    for b in range(bin_count):
        population[b] += fine[b]
        centre[b] += fine[b] * (sync_tip_hz + (b + 0.5) * hz_ire)
    for b in range(bin_count):
        if population[b] > 0.0:
            centre[b] /= population[b]

    start = np.zeros(bin_count + 1, dtype=np.int64)
    for b in range(bin_count):
        start[b + 1] = start[b] + np.int64(population[b])
    cursor = start[:bin_count].copy()
    grouped = np.empty(start[bin_count], dtype=values.dtype)
    for i in range(count):
        if values[i] <= 0.0:
            continue
        b = index[i]
        if b >= 0:
            grouped[cursor[b]] = values[i]
            cursor[b] += 1

    levels = np.zeros(bin_count)
    for b in range(bin_count):
        first, last = start[b], start[b + 1]
        if last > first:
            levels[b] = np.median(grouped[first:last])

    return levels, population, centre


def response_curve(levels, population, centre_hz):
    """The path's amplitude response across the carrier deviation range.

    One component, because one is all the measurement supports. In the
    logarithm it is a straight line against carrier frequency - about
    -0.4 per MHz on this deck, fitted through the binned medians
    `_binned_median` returns - a median because amplitude noise
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


def _low_end_slope(described_hz, described, weight, sync_band, fallback):
    """The roll-off the response follows across the SYNC AREAS.

    The band is not one straight line, and the low end has its own slope. The
    place to measure it is between sync tip and blanking - the format's own two
    fixed levels, -40 and 0 IRE - because that is the one stretch of the low end
    the carrier visits on every line and so the one that is densely determined:
    it holds a tenth to a quarter of a field's samples.

    Measured there it is steeper than the line fitted across the whole range on
    ten of twelve channels, by a median 0.79 dB/MHz and by as much as 2.80 on
    75bars at EP and 2.24 on home's second head. Only two channels come back
    marginally shallower, both within 0.21. So extrapolating BELOW the band with
    the global line understates the loss, and it does so exactly where the sync
    undershoot lands.

    Falls back to the global slope when the sync band is too thinly described to
    fit one, which is the answer that stood before this existed.
    """
    low_hz, high_hz = sync_band
    inside = (described_hz >= low_hz) & (described_hz <= high_hz)
    if np.count_nonzero(inside) < CURVE_INDEPENDENT_POINTS:
        return fallback
    x, y, w = described_hz[inside], described[inside], weight[inside]
    mass = w.sum()
    if not mass > 0.0:
        return fallback
    centre = float((x * w).sum() / mass)
    spread = float((w * (x - centre) ** 2).sum())
    if not spread > 0.0:
        return fallback
    level = float((y * w).sum() / mass)
    return float((w * (x - centre) * (y - level)).sum() / spread)


def _low_end_curve(frequency, edge_hz, edge_level, slope):
    """Continue below the measured band on a FILTER's curve, not on a line.

    A straight line in dB does not stop. At three dB per MHz it reaches tens of
    dB across the RF band, and whatever the response is doing down there, it is
    not that: the channel has a passband, and a magnitude that keeps climbing as
    the frequency falls is not a channel. A low-pass magnitude does stop - it
    flattens toward its own passband - and it is the shape the path actually
    has.

    n poles, so log|H| = -0.5 * ln(1 + (f/fc)^(2n)), anchored to pass through
    the measurement's own edge. Matching the slope measured across the sync
    areas fixes the corner:

        u = (f0/fc)^(2n),    u / (1 + u) = -slope * f0 / n

    which has a solution only for n > -slope * f0, so the pole count is the
    smallest the measurement admits rather than a number chosen here. On the
    twelve channels it comes out three, and four on one.

    Tested by fitting on the sync areas and predicting the measured response
    BELOW sync tip, which the fit never saw: the curve beats the straight line
    on ALL TWELVE channels, typically by a quarter of the error - 0.39 dB to
    0.30 on pulse and bar, 0.42 to 0.30 on chromanoise, 0.41 to 0.29 on 75bars.

    Falls back to the line when the response does not fall with frequency at the
    edge at all, which no channel measured here does but a damaged one might.
    """
    straight = edge_level + slope * (frequency - edge_hz)
    if not (edge_hz > 0.0) or not (slope < 0.0):
        return straight
    poles = max(2, int(np.ceil(-slope * edge_hz)) + 1)
    share = -slope * edge_hz / poles
    if not (0.0 < share < 1.0):
        return straight
    # u = (f0/fc)^(2n), inverted for the corner
    ratio = share / (1.0 - share)
    corner = edge_hz / ratio ** (0.5 / poles)
    if not (corner > 0.0):
        return straight

    def shape(hz):
        return -0.5 * np.log1p((hz / corner) ** (2 * poles))

    return edge_level + shape(frequency) - shape(edge_hz)


def _described_level(
    described_hz, described, weight, frequency, sync_band, fallback
):
    """The response where it was measured, believed in full; extrapolated below.

    A bin is admitted only once it has accumulated `CURVE_MINIMUM_POPULATION`
    samples, and past that admission the measurement stands on its own. Nothing
    pulls it back toward the line. The objective is a FLAT corrected response,
    and any pull is response knowingly left in - at the median bin the graded
    blend that stood here kept an eighth of it.

    The line survives OUTSIDE the described range, and only there. About a tenth
    of a field - sync undershoot especially - lands beyond the frequencies the
    picture visits, and a response held flat across that mis-corrects it, while
    a line is a line everywhere. So the model is the measurement where there is
    one and the trend where there is not, with nothing in between.

    Two graded alternatives were built and measured before this ruling and are
    gone from the code; the numbers are kept only so the choice is not reopened
    on a guess. Scored out of sample across six heads - model built on half a
    head's fields, judged on the other half - a fixed half weight of
    `CURVE_MINIMUM_POPULATION` reached 0.801, the Wiener weight that briefly
    replaced it 0.826, and believing the measurement outright 0.803. All three
    sit within a few points of each other and the flatness they deliver differs
    by less, which is why the decision rests on what the correction is FOR
    rather than on those numbers.

    BELOW the band the extrapolation leaves the measurement's own edge on the
    slope measured across the sync areas and follows a filter's curve down from
    there, rather than jumping to wherever the global line happens to pass and
    running off on a straight one. Both halves of that matter: the line's height
    at the edge
    is not the measurement's: the step between them runs -0.13 to -0.83 dB
    across the channels and reaches -5.6 dB on home's second head, and every
    sample below the band was corrected against that step. And the slope
    differs too - see `_low_end_slope`.

    ABOVE the band the line still stands, unanchored, because the same case has
    not been made for it.
    """
    level = np.interp(frequency, described_hz, described)
    below = frequency < described_hz[0]
    if np.any(below):
        slope = _low_end_slope(described_hz, described, weight, sync_band,
                               fallback)
        level[below] = _low_end_curve(
            frequency[below], described_hz[0], described[0], slope
        )
    return level


def _model_log_level(grid_hz, curve, dense, sync_band):
    """`ln R` on a frequency grid: the measurement inside the described range,
    the low end's own roll-off continued below it, the fitted line above."""
    centre_hz, level, slope = curve
    log_level = level + slope * (grid_hz - centre_hz)
    if dense is not None:
        described_hz, described, described_weight = dense
        reach = grid_hz <= described_hz[-1]
        log_level[reach] = _described_level(
            described_hz, described, described_weight,
            grid_hz[reach], sync_band, slope,
        )
    return log_level


def _response_tables(response, frequency_step, curve, dense, sync_band):
    """The modelled response `S * R`, inverted, on the grid it is tabulated on.

    The decoder's own RF path and the response measured on top of it are both
    functions of carrier frequency alone, so they are combined here rather than
    divided out one after the other. It turns the correction's per-sample
    work into a table lookup and a multiply, and confines the exponential to the
    few thousand points that describe the response rather than the million it is
    applied to.

    The grid covers frequencies the picture never visited, and the model has a
    different answer on each side of what it measured. Inside the described
    range it is the measurement. BELOW it, `_low_end_curve` continues on a
    filter's own shape from the measured edge. ABOVE it, the fitted line still
    stands, because the case made for the low end has not been made for the
    high one.

    How much rides on that is worth stating, since it is easy to over-invest
    here: about a tenth of a field lands outside the NOMINAL deviation range
    (-40 to 100 IRE), but the described range reaches far past that, and only
    0.03 per cent of a field's samples fall below its low edge - all of them
    within 0.1 MHz of it. That is the ceiling on anything the extrapolation can
    be worth.

    Zero marks a frequency at which the model says nothing usable - the response
    vanishes. That is the same sentinel the amplitude itself uses, and it leaves
    the deviation at unity. The collapse C is not in this table: it is a
    function of the trajectory, not of frequency alone, and the kernel forms
    it per sample from `sweep_collapse_tables`.
    """
    grid_hz = np.arange(len(response), dtype=np.float64) * frequency_step
    log_level = _model_log_level(grid_hz, curve, dense, sync_band)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        modelled = response * np.exp(log_level)
        # Narrowed here rather than after, so a reciprocal that is finite in
        # double and infinite in single is caught by the same test as the rest.
        inverse = np.reciprocal(modelled).astype(np.float32)

    inverse[~np.isfinite(inverse)] = 0.0
    inverse[modelled <= 0.0] = 0.0
    return inverse


@njit(cache=True, nogil=True, fastmath=True)
def _deviation_from_response(
    amplitude, carrier_hz, fail_inv, l1, q, q2, rho, ceiling,
    track_arg, slope, frequency_step, low, high
):
    """The residual `n`: how far the amplitude departs from `S * C * R`.

    One pass over the field. Per sample the model is one table lookup for the
    frequency-only factors and the closed-form collapse for the trajectory
    factor:

        ratio = A * fail_inv(f) / C(f, track, slope)

    with the division skipped where the collapse predicts nothing believable
    (C at or below the guard). The collapse belongs in that denominator: it
    is the amplitude a carrier loses purely by MOVING, predicted from the
    decoder's own filter, and the model was binned from the amplitude with it
    already taken out. Leaving it in states the two sides of the division
    against different references and books the loss as tape noise. It is not
    tape noise - it is the picture, and a sweeping carrier is exactly where
    the picture is. Taking it out flattens the corrected response by a third
    on 75bars and two thirds on chromanoise; the full-rate prediction is
    worth -0.0122 on the record-referenced 75bars SP metric and -0.0125 at EP
    against gating alone, +0.0005 on chromanoise.

    The collapse arithmetic runs in double against the float64 tables - the
    same spelling the strided binning path uses - so the two cannot drift
    apart; the ratio narrows back to single where the old kernel narrowed.
    """
    count = len(amplitude)
    deviation = np.empty(count, dtype=np.float32)
    last = len(fail_inv) - 1
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for i in range(count):
        position = carrier_hz[i] / frequency_step
        # One index resolution serves the response table and the coefficient
        # tables alike; at the ends the interpolation degenerates to the held
        # entry exactly.
        if position <= zero:
            lower = 0
            upper = 0
            fraction = zero
        elif position >= last:
            lower = last
            upper = last
            fraction = zero
        else:
            lower = int(position)
            upper = lower + 1
            fraction = position - np.float32(lower)
        base = fail_inv[lower]
        inverse = base + (fail_inv[upper] - base) * fraction
        ratio = amplitude[i] * inverse
        collapse, _ = _collapse_at(
            l1, q, q2, rho, ceiling, np.float64(position),
            np.float64(fraction), lower, upper,
            np.float64(track_arg[i]), np.float64(slope[i]),
        )
        if collapse > 0.05:
            ratio = np.float32(ratio / collapse)
        if ratio <= zero:
            # no measurement here, or no usable model - leave the sample alone
            ratio = one
        elif ratio < low:
            ratio = low
        elif ratio > high:
            ratio = high
        deviation[i] = ratio
    return deviation


# fastmath is load bearing, not decoration. The lerp below used to stand in a
# function of its own compiled with it, and the fused multiply-add that enables
# is part of the answer: measured, dropping the flag moves a third of the
# samples in the last bit and the decoded output with them.
@njit(cache=True, nogil=True)
def _running_max(values, size, origin, out):
    """Sliding maximum, in two passes and three comparisons a sample.

    van Herk / Gil-Werman: a forward running max within each block of `size` and
    a backward one, after which any window of that width is the larger of two
    already-computed ends. The cost stops depending on the window width, which
    matters because the width here is three settling times and grows with the
    format's band.

    Bit-identical to `scipy.ndimage.maximum_filter1d(..., mode="nearest")` -
    verified over spans of 37, 128 and 401 - because a maximum is comparisons
    and nothing else, so there is no arithmetic to reassociate. Measured at a
    field's length it is 4.9 ms against scipy's 10.7.

    The edges replicate, which is what mode="nearest" means. They are scanned
    directly rather than padded: padding the input to avoid the scan was tried
    and is SLOWER (6.9 ms), because the copy costs more than the few edge
    samples do.
    """
    count = len(values)
    forward = np.empty(count, dtype=values.dtype)
    backward = np.empty(count, dtype=values.dtype)
    for i in range(count):
        if i % size == 0:
            forward[i] = values[i]
        else:
            previous = forward[i - 1]
            forward[i] = previous if previous > values[i] else values[i]
    for i in range(count - 1, -1, -1):
        if i == count - 1 or (i + 1) % size == 0:
            backward[i] = values[i]
        else:
            later = backward[i + 1]
            backward[i] = later if later > values[i] else values[i]

    half = size // 2
    last = count - 1
    for i in range(count):
        low = i - half - origin
        high = low + size - 1
        if low < 0 or high > last:
            # the window hangs off an end, so the replicated edge value stands
            # in for everything outside
            best = values[0] if low < 0 else values[last]
            start = low if low > 0 else 0
            stop = high if high < last else last
            for j in range(start, stop + 1):
                if values[j] > best:
                    best = values[j]
            out[i] = best
        else:
            a, b = backward[low], forward[high]
            out[i] = a if a > b else b
    return out


# The model as one bundle a demodulation block can read without any field
# context - the joint interface with the head-switch arc. `expected_ln` is
# `ln(S * R)` pooled across heads (per-head tables swapped under the demod
# thread were measured irreproducible; one field of latency is the accepted
# pattern), absolute in envelope units so the residual is centred without a
# DC fixup; the coefficient tables are `sweep_collapse_tables`' float32
# casts. `version` increments per refresh. None until the first measured
# field.
BlockModel = namedtuple(
    "BlockModel",
    [
        "expected_ln", "frequency_step",
        "l1", "q", "q2", "rho", "ceiling",
        "span", "track_width", "version",
        # appended (contract growth is append-only): the measured sync-region
        # level anchors - the carrier frequency at blanking and at sync tip,
        # and the scale they imply, pooled per the same determinism law as
        # everything else in the bundle. 0.0 until measured.
        "porch_hz", "tip_hz", "hz_ire_measured",
    ],
)


def block_model(rf):
    """The current `BlockModel`, or None until a field has been measured."""
    return rf.__dict__.get("_luma_block_model")


@njit(cache=True, nogil=True, fastmath=True)
def _block_residual_kernel(envelope, carrier_hz, expected_ln, l1, q, q2, rho,
                           ceiling, track_arg, slope, frequency_step, out):
    last = len(expected_ln) - 1
    for i in range(len(envelope)):
        position = carrier_hz[i] / frequency_step
        if position <= np.float32(0.0):
            lower = 0
            upper = 0
            fraction = np.float32(0.0)
        elif position >= last:
            lower = last
            upper = last
            fraction = np.float32(0.0)
        else:
            lower = int(position)
            upper = lower + 1
            fraction = position - np.float32(lower)
        base = expected_ln[lower]
        expected = base + (expected_ln[upper] - base) * fraction
        collapse, _ = _collapse_at(
            l1, q, q2, rho, ceiling, np.float64(position),
            np.float64(fraction), lower, upper,
            np.float64(track_arg[i]), np.float64(slope[i]),
        )
        value = envelope[i]
        if value > np.float32(0.0):
            out[i] = np.log(value) - expected - np.float32(np.log(collapse))
        else:
            # no carrier at all: NO EVIDENCE, not extreme evidence. A fixed
            # out-of-range marker the consumer must EXCLUDE - clamping it
            # into range manufactures a spike where nothing was measured
            # (found the hard way by the first consumer). Any real residual
            # is within a few nepers; anything at or below the marker is
            # absence.
            out[i] = np.float32(-100.0)
    return out


@njit(cache=True, nogil=True, fastmath=True)
def _block_phase_kernel(carrier_hz, l1, q, q2, rho, ceiling, track_arg, slope,
                        frequency_step, out):
    last = len(l1) - 1
    for i in range(len(carrier_hz)):
        position = carrier_hz[i] / frequency_step
        if position <= np.float32(0.0):
            lower = 0
            upper = 0
            fraction = np.float32(0.0)
        elif position >= last:
            lower = last
            upper = last
            fraction = np.float32(0.0)
        else:
            lower = int(position)
            upper = lower + 1
            fraction = position - np.float32(lower)
        _, psi = _collapse_at(
            l1, q, q2, rho, ceiling, np.float64(position),
            np.float64(fraction), lower, upper,
            np.float64(track_arg[i]), np.float64(slope[i]),
        )
        out[i] = np.float32(psi)
    return out


def block_log_residual(model, envelope, carrier_hz):
    """`d = ln A - ln(S*R) - ln C` for one demodulation block, float32.

    The residual is the model's `ln n` - what the path did that the carrier's
    own trajectory does not predict. Unclamped; the consumer clamps to its
    own bounds - EXCEPT the no-evidence marker: samples with no carrier at
    all return exactly -100.0, to be excluded, never clamped (absence of
    measurement is not extreme measurement). `carrier_hz` is the block's raw
    instantaneous frequency
    straight out of the demodulator; the trajectory channels are formed here
    at the model's own scales, so the block needs no field context.
    """
    track_arg = np.empty_like(carrier_hz)
    slope = np.empty_like(carrier_hz)
    _sweep_arrays(carrier_hz, model.span, model.track_width, track_arg, slope)
    return _block_residual_kernel(
        envelope, carrier_hz, model.expected_ln,
        model.l1, model.q, model.q2, model.rho, model.ceiling,
        track_arg, slope, np.float32(model.frequency_step),
        np.empty(len(envelope), dtype=np.float32),
    )


def block_sweep_phase(model, carrier_hz):
    """`psi = arg(E/H)` for one block: the deterministic phase the sweep
    imposes through the band's shape, radians, float32. The causal partner
    of the residual's amplitude is the CONSUMER's Hilbert completion; this is
    the modelled, sweep-locked part that has no amplitude signature."""
    track_arg = np.empty_like(carrier_hz)
    slope = np.empty_like(carrier_hz)
    _sweep_arrays(carrier_hz, model.span, model.track_width, track_arg, slope)
    return _block_phase_kernel(
        carrier_hz, model.l1, model.q, model.q2, model.rho, model.ceiling,
        track_arg, slope, np.float32(model.frequency_step),
        np.empty(len(carrier_hz), dtype=np.float32),
    )


@njit(cache=True, nogil=True, fastmath=True)
def _refine_sync_edges(carrier, anchors, search, porch_lo, porch_hi,
                       tip_lo, tip_hi, out):
    """Per-line sub-sample sync falling edge from the carrier itself.

    The line positions this stage sees are an integer, exact-period
    interpolation carrying none of the tape's timing - the carrier holds the
    truth. Per nominal anchor: this line's own porch and tip levels by
    median, then the 50% falling crossing by linear interpolation, the
    candidate nearest the nominal winning. -1 marks a line with no clean
    crossing. Ported from the offline fold instrument, whose folded edge
    width on refined anchors is the validation that the convention is
    sound.
    """
    n = len(carrier)
    for ix in range(len(anchors)):
        out[ix] = -1.0
        a = int(np.round(anchors[ix]))
        lo = a - search
        hi = a + search
        if lo <= tip_hi or hi + tip_hi >= n or lo < -porch_lo:
            continue
        porch = np.median(carrier[a - porch_lo:a - porch_hi])
        tip = np.median(carrier[a + tip_lo:a + tip_hi])
        if not porch > tip:
            continue
        mid = 0.5 * (porch + tip)
        best = -1.0
        best_dist = 1e18
        for k in range(lo, hi):
            y0 = carrier[k]
            y1 = carrier[k + 1]
            if y0 > mid and y1 <= mid and y1 != y0:
                pos = k + (mid - y0) / (y1 - y0)
                dist = abs(pos - a)
                if dist < best_dist:
                    best_dist = dist
                    best = pos
        out[ix] = best
    return out


def sync_edge_trace(field):
    """Per-line refined sync-edge positions for this field - the time base's
    own trace, measured from the carrier.

    Returns (edges, period) or None: `edges` is float64, one entry per line
    position the decoder carries, the absolute sample index of that line's
    50% sync falling crossing, -1 where no clean crossing exists; `period`
    is the median line spacing of the valid entries. Raw positions,
    deliberately - wow and flutter live in the trace's low frequencies, and
    what to detrend is the consumer's call, not this stage's.

    Computed on demand and memoized on the field: the measurement pass never
    pays for it. Consumers so far: the head-switch arc's wow/flutter side
    task.
    """
    cached = getattr(field, "_luma_sync_trace", None)
    if cached is not None:
        return cached
    linelocs = getattr(field, "linelocs2", None)
    video = field.data.get("video") if hasattr(field, "data") else None
    if linelocs is None or video is None or "demod_raw" not in video:
        return None
    rf = field.rf
    fs_us = rf.freq_hz / 1e6
    carrier = np.asarray(video["demod_raw"], dtype=np.float64)
    edges = _refine_sync_edges(
        carrier,
        np.asarray(linelocs, dtype=np.float64),
        int(round(6.0 * fs_us)),
        int(round(1.2 * fs_us)), int(round(0.2 * fs_us)),
        int(round(1.0 * fs_us)), int(round(3.5 * fs_us)),
        np.empty(len(linelocs), dtype=np.float64),
    )
    valid = edges > 0
    period = float(np.median(np.diff(edges[valid]))) if valid.sum() > 2 else 0.0
    cached = (edges, period)
    field._luma_sync_trace = cached
    return cached


def _measure_level_anchors(field, centred_hz):
    """The carrier frequency at the format's two fixed levels, measured.

    Per usable line, the median carrier over the sync tip and over the two
    porches (the burst span excluded), anchored on the decoder's line
    positions with guard margins from the format's own transition width and
    the path's settling span. The medians are robust to the anchor grid's
    few-sample jitter, which is why plain linelocs plus spec offsets
    suffice here where the fold instrument needed refined edges - a LEVEL
    is flat across its span; a TIMING is not.

    Returns (porch_hz, tip_hz) medians-of-lines for this field, or None.
    The spec anchor in `carrier_frequency_bins` is deliberately NOT fed by
    this: the binning's constancy argument stands (a moving axis is what
    does damage there); this is an EXPORT for consumers that need the
    measured axis - the head-switch kernel's absolute level and slope
    anchors first among them.
    """
    linelocs = getattr(field, "linelocs2", None)
    if linelocs is None:
        return None
    rf = field.rf
    fs_us = rf.freq_hz / 1e6
    sys_params = rf.SysParams
    # spans in samples relative to each line's sync falling edge
    edge = 0.140 * fs_us * 3.0
    knee, span = path_transient_scale(rf)
    # Two guards, the ringing lane's convention: a LEVEL span needs only the
    # transition width plus a third of the settling (levels are flat once
    # near-settled), while the tip's interior - entered through a full
    # transition - gets the whole settling span. One guard for everything
    # was measured to eat every porch whole.
    level_guard = int(np.ceil(edge + span / 3.0))
    tip_guard = int(np.ceil(edge + span))
    sync_samples = int(round(4.7 * fs_us))
    front = int(round(1.5 * fs_us))
    burst_lo = int(round(5.3 * fs_us))
    burst_hi = int(round(7.8 * fs_us))
    active = int(round(9.45 * fs_us))
    n = len(centred_hz)
    L = np.asarray(linelocs, dtype=np.float64)
    inner = L[9:min(len(L) - 1, 261)]
    if len(inner) < CURVE_INDEPENDENT_POINTS:
        return None
    period = np.median(np.diff(inner))
    tips = []
    porches = []
    linecount = int(getattr(field, "linecount", 0)) or (len(L) - 10)
    for i in range(9, min(linecount - 1, len(L) - 1)):
        if abs(L[i] - L[i - 1] - period) > 5.0 or abs(L[i + 1] - L[i] - period) > 5.0:
            continue
        a = int(round(L[i]))
        lo = a + tip_guard
        hi = a + sync_samples - tip_guard
        if lo < 0 or hi + active >= n or hi <= lo:
            continue
        tips.append(np.median(centred_hz[lo:hi]))
        spans_p = []
        f_lo, f_hi = a - front + level_guard, a - level_guard
        if f_hi > f_lo:
            spans_p.append(centred_hz[f_lo:f_hi])
        b_lo = a + sync_samples + level_guard
        b_hi = a + burst_lo - level_guard
        if b_hi > b_lo:
            spans_p.append(centred_hz[b_lo:b_hi])
        b2_lo = a + burst_hi + level_guard
        b2_hi = a + active - level_guard
        if b2_hi > b2_lo:
            spans_p.append(centred_hz[b2_lo:b2_hi])
        if spans_p:
            porches.append(np.median(np.concatenate(spans_p)))
    if len(tips) < CURVE_INDEPENDENT_POINTS or len(porches) < CURVE_INDEPENDENT_POINTS:
        return None
    return float(np.median(porches)), float(np.median(tips))


def _refresh_block_model(rf, response, frequency_step, tables, deviation,
                         stride, dropout_fraction, sync_band):
    """Rebuild the pooled block bundle from this field's accumulated state.

    The noise scale that lived here is measured history: a strided-MAD
    noise_ln shipped with the first bundle, was measured to read TWICE the
    honest sync-witnessed floor (the excess being picture-locked structure,
    not noise - 36-45 vs 73-83 mNp on SP), lost its only consumer when the
    head-switch garrote became a regression-measured gain, and was removed
    on Ethan's ruling. The honest floor and the noise's frequency response
    live in the shared inbox (luma_noise_response) and this lane's fold
    instruments; a future noise consumer starts from those, not from a MAD
    over picture.
    """
    store = rf.__dict__.get("_luma_eq_lines")
    if not store:
        return
    line_total = sum(entry[0] for entry in store.values())
    line_weight = sum(entry[1] for entry in store.values())
    pooled_curve = tuple(line_total / line_weight)
    grid_hz = np.arange(len(response), dtype=np.float64) * frequency_step
    # The block expectation KEEPS the accumulated table even though the
    # deviation's own model dropped it. Different consumers, opposite
    # optima, measured: the deviation is better off leaving the ripple in
    # (the line-only model won on all three conditions), while the block
    # residual's consumer differentiates in time - and the table's ripple,
    # traversed by the carrier, is picture-locked structure worth
    # 0.019 nepers rms along a real trajectory (a quarter of the noise
    # floor) that would otherwise be booked as path events. The head-switch
    # signature itself is indifferent to this choice: it lives in the
    # per-head levels and the handoff lines (measured: the assembled
    # field's first ~13 lines, 0.12-0.14 nepers against the field's own
    # model, 2.4x the static head split of 0.055), and no R(f) can absorb
    # a 13-line block.
    log_level = _model_log_level(grid_hz, pooled_curve, _pooled_dense(rf),
                                 sync_band)
    expected = np.where(
        response > 0,
        np.log(np.maximum(response, 1e-300)) + log_level,
        # no response here at all: state a floor so the residual lands at the
        # consumer's clamp rather than at an infinity
        np.log(1e-300),
    ).astype(np.float32)
    levels = rf.__dict__.get("_luma_level_state", (0.0, 0.0, 0.0))
    porch_hz = levels[0] / levels[2] if levels[2] > 0 else 0.0
    tip_hz = levels[1] / levels[2] if levels[2] > 0 else 0.0
    vsync_ire = float(rf.SysParams["vsync_ire"])
    scale = (porch_hz - tip_hz) / (0.0 - vsync_ire) if levels[2] > 0 else 0.0
    previous = rf.__dict__.get("_luma_block_model")
    rf._luma_block_model = BlockModel(
        expected, frequency_step,
        tables.l1_32, tables.q_32, tables.q2_32, tables.rho_32,
        tables.ceiling_32,
        tables.span, tables.track_width,
        (previous.version + 1) if previous is not None else 1,
        porch_hz, tip_hz, scale,
    )


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

    - the distance between the model and the line at each edge of the described
      range - a real discontinuity at the high end, where the line takes over,
      and at the low end the size of what `_low_end_curve` is measured against;
    - how much evidence stands behind each described bin, against the
      population that admits it;
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

    # How far the model stands from the line at each end of the described range.
    # At the HIGH end that is the model's real discontinuity: the measurement
    # stops there and the line takes over at whatever height it happens to have.
    # At the LOW end it is not a step at all - `_low_end_curve` continues from
    # the measured edge - so it is the distance the low-end extrapolation is
    # measured against rather than an error.
    #
    # Not scaled by any shrinkage. One stood here and understated both by that
    # factor; the correction has applied the measurement in full since.
    if dense is not None:
        described_hz, described, described_weight = dense
        centre_of_line, level_of_line, slope_of_line = curve
        line = level_of_line + slope_of_line * (described_hz - centre_of_line)
        step = described - line
        state["edge_low_db"].append(_LOG_TO_DB * float(step[0]))
        state["edge_high_db"].append(_LOG_TO_DB * float(step[-1]))
        state["evidence"] = described_weight
        state["evidence_hz"] = described_hz

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
    slew = _running_max(slew, 3 * span, -span, np.empty_like(slew))
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
    bin_hz = sys_params["hz_ire"] * RESPONSE_CURVE_RESOLUTION_IRE
    sample_bin = _frequency_bin_of(
        sampled_carrier_hz, sync_tip_hz, bin_hz, bin_count
    )
    levels, population, centre_hz = _binned_median(
        sampled_flattened, sample_bin, sync_tip_hz, bin_hz, bin_count
    )
    curve = response_curve(levels, population, centre_hz)
    if curve is None:
        return None

    # The fitted line, accumulated per head where the measurement is made.
    # One owner for the model's factors: the equalizer and the block bundle
    # both read this store, and neither accumulates it themselves.
    line_store = rf.__dict__.setdefault("_luma_eq_lines", {})
    line_key = bool(field.isFirstField)
    line_total, line_weight = line_store.get(line_key, (np.zeros(3), 0.0))
    line_store[line_key] = (line_total + np.array(curve), line_weight + 1.0)

    # The response read off the measurement itself, accumulated per head across
    # the decode. A single field determines it far too poorly - that is why
    # fitting densely per field loses - but a decode's worth of one head's
    # fields determines it well, and the two heads genuinely differ.
    #
    # Binned a second time, from the same subsample, with the sweep collapse divided
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
    tables = sweep_collapse_tables(rf)
    track_arg = np.empty_like(centred_hz)
    slope = np.empty_like(centred_hz)
    _sweep_arrays(centred_hz, tables.span, tables.track_width, track_arg, slope)
    # The collapse on the subsample, for the compensated binning: a moving
    # carrier contributes the amplitude it would have had standing still.
    # Same float64 formula the deviation kernel applies per sample, so the
    # two paths cannot drift apart; float32 out, the width the binning gate
    # has always read.
    sampled_collapse = _collapse_closed(
        sampled_carrier_hz, track_arg[::stride], slope[::stride],
        tables.l1, tables.q, tables.q2, tables.rho, tables.ceiling,
        np.float64(frequency_step),
        np.empty(len(sampled_carrier_hz), dtype=np.float32),
    )
    alive = sampled_collapse > 0.05
    compensated = np.where(
        alive, compensated / np.where(alive, sampled_collapse, 1.0), 0.0
    )
    dense_levels, dense_population, dense_centre_hz = _binned_median(
        compensated, sample_bin, sync_tip_hz, bin_hz, bin_count
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
    # undershoot below sync tip, the overshoot past peak white - are left to
    # what `_response_tables` extrapolates. Admitting them instead, on
    # whatever handful of samples they hold, stretches the interpolation over
    # uncorrelated excursions and drags the model away from the amplitude it
    # follows.
    #
    # This admission is the ONLY gate. Past it a bin is believed outright, and
    # the graded blends that used to sit downstream of it are gone - see
    # `_described_level` for the measurements that ended them.
    #
    # A PARAMETRIC form of the accumulated response was built and rejected, and
    # it is worth saying so because it is an attractive thing to rebuild. The
    # ripple was fitted as damped sinusoids by matrix pencil, which on planted
    # signals is exact where peak picking leaves a third of the ripple behind.
    # On real data it does not hold: the singular values of the Hankel matrix
    # decay smoothly - 1, 0.69, 0.20, 0.17, 0.17, 0.14 - where a true sum of
    # sinusoids drops twelve orders of magnitude after its last component, so
    # there is no rank to find, the answer moves with the pencil parameter, and
    # the strongest component disagrees between the two heads by microseconds
    # rather than the nanoseconds a real echo would. The test before trying
    # again is that singular value spectrum: no knee, no parametric form.
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
    # The deviation divides by the LINE'S R, not the accumulated table -
    # measured, resolved, on all three conditions: withholding the table is
    # worth -0.016/-0.037/-0.041 per cent against keeping it, and
    # -0.012/-0.025/-0.043 against the synthesized-collapse build it
    # replaced (record-referenced, paired). Under the closed-form collapse
    # the table's ripple prices NEGATIVE here: what it absorbs from the
    # deviation is structure the correction is better off leaving. The table
    # is still measured and accumulated - the equalizer inverts it on its own
    # evidence, the plots draw it, and the block model's expectation carries
    # what its consumer rules (see `_refresh_block_model`).
    fail_inv = _response_tables(
        response, frequency_step, curve, None,
        # sync tip and blanking, the format's own two fixed levels
        (rf.iretohz(sys_params["vsync_ire"], spec=True),
         rf.iretohz(0.0, spec=True)),
    )
    deviation = _deviation_from_response(
        amplitude,
        centred_hz,
        fail_inv,
        tables.l1, tables.q, tables.q2, tables.rho, tables.ceiling,
        track_arg,
        slope,
        np.float32(frequency_step),
        np.float32(dropout_fraction),
        np.float32(1.0 / dropout_fraction),
    )

    anchors = _measure_level_anchors(field, centred_hz)
    if anchors is not None:
        porch_sum, tip_sum, count = rf.__dict__.get(
            "_luma_level_state", (0.0, 0.0, 0.0))
        rf._luma_level_state = (
            porch_sum + anchors[0], tip_sum + anchors[1], count + 1.0)

    _refresh_block_model(
        rf, response, frequency_step, tables, deviation, stride,
        dropout_fraction,
        (rf.iretohz(sys_params["vsync_ire"], spec=True),
         rf.iretohz(0.0, spec=True)),
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
        centred_hz,
        steadiness,
        ResponseModel(
            curve,
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

    A bin the carrier barely visited contributes nothing, because it is never
    admitted: `CURVE_MINIMUM_POPULATION` is a gate, not a blend. Past it the
    measurement stands on its own - see `_described_level` for the measurements
    that ended the graded alternatives.

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
        inside = frequency <= described_hz[-1]
        # An absolute log level either way, and the result is stated against the
        # middle of the range so the equalizer carries no overall gain.
        blended = _described_level(
            described_hz, described, described_weight, frequency[inside],
            (inner_low, rf.iretohz(0.0, spec=True)), slope,
        )
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
    prediction the signal has just contradicted.

    The response is accumulated per head across the decode. One field
    determines the line well enough (its slope's standard deviation is under
    1% of the slope) but accumulating costs nothing and rides out a bad field.
    """
    measured = measured_amplitude_deviation(field)
    if measured is None or measured.response is None:
        return
    rf = field.rf
    # The line is accumulated where the measurement is made, in
    # `measure_amplitude_deviation`; this reads the store it shares with the
    # block bundle.
    store = rf.__dict__.get("_luma_eq_lines")
    if not store:
        return
    head = bool(field.isFirstField)

    expected = rf.__dict__.get("_luma_eq_expected")
    rf._luma_eq_expected = not head
    if expected is not None and expected != head:
        # The alternation just failed, so the basis for naming the next
        # field's head has failed with it. Whatever filter is in place
        # stays there for this field rather than being rebuilt on a
        # prediction the signal has just contradicted.
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


