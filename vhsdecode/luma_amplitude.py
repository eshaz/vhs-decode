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

from vhsdecode import residual_limit
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


# The sweep residue's bins, in the collapse formula's OWN argument theta^2
# (= q * slope^2), so nothing about the axis is invented here: the ladder is
# anchored on the knee at theta^2 = 1, where the formula's own damping
# 1/(1 + theta^2) halves, and each step is a factor of four - one octave in
# theta. It reaches eight octaves below the knee because that is where the
# carrier's sweep actually lives: measured over four captures the median
# theta is 0.02-0.04 and the 99th percentile 0.33, so almost every sample
# sits at theta^2 below 0.11, and a ladder centred on the knee would put the
# whole picture into one bin.
SWEEP_RESIDUE_OCTAVES = 8
SWEEP_RESIDUE_RATIO = 4.0



# The sync transition's 10-90% width as the video standard sends it. Used to
# stand a guard off each blanking edge, never to model the edge itself - the
# recorded edge is not this shape (the sync-edge instrument reads a delay of
# 45-55 ns against an ideal centred on the same crossing).
SYNC_EDGE_US = 0.140


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
def _collapse_at(l1, q, q2, rho, ceiling, fraction, lower, upper,
                 track, slope):
    """The closed-form collapse at one resolved table index. See
    `sweep_collapse_tables` for the terms; `_deviation_from_response` and the
    the deviation kernel stands on this alone now. Its strided companion
    `_collapse_closed` was retired with the compensated binning that was its
    only caller - the response is measured as the deviation's own residual,
    so there is no longer a second path to keep in step.

    Takes the index already RESOLVED - lower, upper and the fraction between
    them. The unresolved `position` was taken here too and never read, every
    caller having resolved it before the call."""
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
    # ceiling factors are positive reals and cannot turn it. theta^2 comes
    # back with them because it is the formula's own sweep argument and the
    # measured sweep residue is indexed on it - recomputing it in the caller
    # would be a second spelling of the same lerp, which is exactly how two
    # paths drift apart.
    return collapse, np.arctan2(imag, real), theta2


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


def sweep_residue_edges():
    """Bin edges in theta^2 for the measured sweep residue.

    The first bin holds everything below the ladder and the last everything
    at or above the knee, so every sweep rate lands somewhere and no sample
    is dropped from the measurement that corrects it.
    """
    ladder = SWEEP_RESIDUE_RATIO ** np.arange(
        -SWEEP_RESIDUE_OCTAVES, 1, dtype=np.float64)
    return np.concatenate(([0.0], ladder))


# RULE R3 and the statistic that carries it live in `residual_limit`, the
# shared law - see `docs/RESIDUAL_LIMIT_DESIGN.md`. They are named here so
# the call sites below read as the design does.
_cell_medians = residual_limit.cell_medians
_orthogonal = residual_limit.orthogonal


def joint_residue_of(rf, head, bin_count):
    """The product component: what neither margin can carry, over carrier
    frequency AND the sweep argument together.

    Held orthogonal to both margins by construction (see `_orthogonal`), so
    the response table stays a response and the collapse residue stays a
    function of sweep rate alone.

    ADMITTED ONLY AS FAR AS IT GENERALISES, and the test is the component's
    own: it is accumulated in two banks on alternating fields, and what is
    applied is scaled by how far the banks agree - the population-weighted
    correlation between them, floored at zero. A component that describes
    the path reads near one and passes through; a component that describes
    the field it was measured from reads near zero and is withheld, with no
    constant chosen anywhere to do it.

    That test exists because this term needs it. Fitted on the fields it is
    then measured on, the limit drives every component to nothing -
    0.0001 / 0.0000 / 0.0002 dB for response, sweep and product. Fitted on
    other fields it stands at 0.0328 / 0.0079 / 0.0563 dB, and the product
    is the WORST of the three: the least reproducible from one field to the
    next, which is what a term describing the picture rather than the path
    looks like. In-sample convergence is not evidence; see rule R5 in
    `docs/RESIDUAL_LIMIT_DESIGN.md`.

    Ones wherever nothing has been measured, so an unvisited cell is left
    to the formula rather than to an extrapolation across one.
    """
    edges = sweep_residue_edges()
    store = rf.__dict__.get("_luma_joint_residue") or {}
    entry = store.get(bool(head))
    factor = np.ones((bin_count, len(edges)))
    if entry is None:
        return edges, factor
    banks, weights = entry
    weight = weights.sum(axis=0)
    described = weight >= CURVE_MINIMUM_POPULATION
    if not np.any(described):
        return edges, factor
    cells = np.zeros_like(weight)
    cells[described] = banks.sum(axis=0)[described] / weight[described]
    cells = _orthogonal(cells, np.where(described, weight, 0.0))
    # the two banks, each on its own evidence, projected the same way
    pair = []
    for bank in (0, 1):
        w = weights[bank]
        ok = w >= CURVE_MINIMUM_POPULATION
        half = np.zeros_like(w)
        half[ok] = banks[bank][ok] / w[ok]
        pair.append(_orthogonal(half, np.where(ok, w, 0.0)))
    trust = residual_limit.agreement(
        pair[0], pair[1], weights[0], weights[1], CURVE_MINIMUM_POPULATION)
    rf._luma_joint_trust = trust
    return edges, np.exp(trust * cells)


def sweep_residue_of(rf, head):
    """The accumulated multiplicative correction to the collapse, per bin.

    C is predicted from the decoder's own filter and is right to about a
    twentieth of a decibel. What it is not right about is measurable, is a
    function of the sweep rate alone, and reproduces across pictures on the
    formula's own theta axis - r = +0.94 and +0.78 between two pictures of
    one head, where the same residue read on per-condition sweep DECILES
    cannot be compared across conditions at all, the deciles standing at
    different sweep rates on each.

    THE PRODUCT TERM IS NOT CARRIED, and was built and measured before that
    was decided. A joint table over frequency AND sweep rate does reduce the
    residual in its own cells - 0.836 to 0.666 per cent rms on chromanoise,
    1.487 to 1.427 on bars - but it does nothing at all for the
    frequency-binned residual the debug panel draws, which it cannot: that
    margin is divided out before this is accumulated, so the two are
    orthogonal by construction. Against that it reproduces across pictures
    at only r = +0.25 and +0.19, which is the signature of describing the
    picture, and the spelling that measured it cost 60 per cent of the
    decode rate. Restoring it is a table shape and a lookup; the case for it
    has to come from a consumer that reads the joint residual, not from the
    panel.

    Ones wherever nothing has been measured yet, so an unvisited sweep rate
    is left to the formula rather than to an extrapolation.
    """
    edges = sweep_residue_edges()
    store = rf.__dict__.get("_luma_sweep_residue") or {}
    entry = store.get(bool(head))
    residue = np.ones(len(edges))
    if entry is not None:
        total, weight = entry
        described = weight >= CURVE_MINIMUM_POPULATION
        residue[described] = np.exp(total[described] / weight[described])
    return edges, residue


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
    track_arg, slope, frequency_step, low, high,
    residue_edges, residue, joint, sync_tip_hz, bin_hz
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
        collapse, _, theta2 = _collapse_at(
            l1, q, q2, rho, ceiling,
            np.float64(fraction), lower, upper,
            np.float64(track_arg[i]), np.float64(slope[i]),
        )
        # The measured half of the collapse: what the formula predicts times
        # what the residue says it gets wrong at this sweep rate. The bins
        # are few and the mass sits in the first of them, so the scan costs
        # less on average than the division below.
        b = 0
        while b + 1 < len(residue_edges) and theta2 >= residue_edges[b + 1]:
            b += 1
        f_bin = int((carrier_hz[i] - sync_tip_hz) / bin_hz)
        if f_bin < 0:
            f_bin = 0
        elif f_bin >= len(joint):
            f_bin = len(joint) - 1
        # the sweep margin and the product that neither margin can carry
        collapse *= residue[b] * joint[f_bin, b]
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
        collapse, _, _ = _collapse_at(
            l1, q, q2, rho, ceiling,
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
        _, psi, _ = _collapse_at(
            l1, q, q2, rho, ceiling,
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

    Per usable line, the median carrier over the sync tip and over the back
    porch after the burst, anchored on the decoder's line positions with
    guard margins from the format's own transition width and the path's
    settling span. The medians are robust to the anchor grid's few-sample
    jitter, which is why plain linelocs plus spec offsets suffice here
    where the fold instrument needed refined edges - a LEVEL is flat across
    its span; a TIMING is not.

    THE FRONT PORCH IS NOT A LEVEL and is deliberately excluded. It stood
    here pooled with the back porch, on geometry alone, until the
    consuming lane asked whether the anchors move with picture content.
    They did. The front porch is 1.5 us long and follows the active-video
    transition, whose carrier overshoot needs about 1 us to settle at this
    path's transient scale, so no guard leaves a settled span inside it:
    measured per line over four captures of ONE deck - four contents, two
    speeds, both heads; the mechanism is a porch shorter than the path's
    settling and should hold wherever that is true, but the SIZE of the
    error is a property of the path and is not established off this deck.
    The per-line regression below is the test to repeat elsewhere. The
    front span's own median sat
    at -0.04 / -5.41 / +2.63 / +5.40 IRE across them (sd 3.99) while every
    back-porch window agreed to sd 0.3-0.5, and the front-minus-back
    difference regressed on the PRECEDING line's active level at
    -0.21 +- 0.005 IRE per IRE (41-66 sigma on the two flat-field tapes,
    2.3 IRE of anchor swing). Pooling the two therefore averaged a settled
    level with a content-driven tail. The back porch after the burst
    carries none of it: slope 0.001-0.009 IRE per IRE, under 0.06 IRE of
    swing, and its median holds to sd 0.27 IRE across the same four
    captures. A span between the sync rise and the burst was also carried
    here and was DEAD CODE - at 40 MSps the guards leave it negative width
    - and the rise's own overshoot occupies it in any case (+64 IRE at
    5.0 us, settled by 5.8).

    Do not extend the span to the start of active video. lddecode's own
    blanking convention (burst end to active start) was measured on the
    same lines and picks up the approach to the active transition: slope
    +0.010 to +0.052 IRE per IRE, 11-49 sigma, up to 0.48 IRE of swing.

    That last ranking is SMALL in absolute terms and was checked across
    domains and decks after the consuming lane failed to reproduce it. In
    the decoded 4fsc luma of this deck's flat-field capture it does
    reproduce - the windows reaching active start regress at +0.0040 to
    +0.0042 IRE per IRE against +0.0001 to +0.0012 for the rest, 32-36
    sigma on the line-correlation-corrected count - so the video domain is
    not blind to it. What differs between decks is not the ranking but the
    COMMON term every window shares: about 0.001 IRE per IRE here against
    about 0.021 on the consuming lane's tape, where the same ~0.002-0.004
    best-to-worst spread is a 9% effect instead of a large ratio. Choose
    the window on the spread, never on the ratio, and expect the common
    term - a settling tail, not a window artefact - to dominate on tapes
    whose porch has one.

    Returns (porch_hz, tip_hz) medians-of-lines for this field, or None.
    The spec anchor in `carrier_frequency_bins` is deliberately NOT fed by
    this: the binning's constancy argument stands (a moving axis is what
    does damage there); this is an EXPORT for consumers that need the
    measured axis - the head-switch kernel's absolute level and slope
    anchors first among them.

    READ THE TIP WITH CARE. The sync pulse is spec-ideal in TIMING as
    recorded but NOT in DEPTH after the recorder: on these decks the tip
    sits shallow (this measurement reads porch-minus-tip 5.8% under the
    spec deviation; the decoded picture reads the tip near -34 IRE while
    black and white land on spec). So `porch_hz` is a sound LEVEL anchor,
    but a scale derived from porch minus tip inherits the recorder's sync
    processing and would stretch a spec-level picture - measured by the
    consuming lane on the source tape. `hz_ire_measured` is exported as
    what it is, a tip-referenced scale, and should not replace the spec
    scale for picture levels.
    """
    linelocs = getattr(field, "linelocs2", None)
    if linelocs is None:
        return None
    rf = field.rf
    fs_us = rf.freq_hz / 1e6
    sys_params = rf.SysParams
    # spans in samples relative to each line's sync falling edge, from the
    # format's own timings rather than NTSC numbers written out here
    edge = SYNC_EDGE_US * fs_us * 3.0
    knee, span = path_transient_scale(rf)
    # Two guards, the ringing lane's convention: a LEVEL span needs only the
    # transition width plus a third of the settling (levels are flat once
    # near-settled), while the tip's interior - entered through a full
    # transition - gets the whole settling span. One guard for everything
    # was measured to eat every porch whole.
    level_guard = int(np.ceil(edge + span / 3.0))
    tip_guard = int(np.ceil(edge + span))
    sync_samples = int(round(sys_params["hsyncPulseUS"] * fs_us))
    burst_hi = int(round(sys_params["colorBurstUS"][1] * fs_us))
    active = int(round(sys_params["activeVideoUS"][0] * fs_us))
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
        p_lo = a + burst_hi + level_guard
        p_hi = a + active - level_guard
        if p_hi > p_lo:
            porches.append(np.median(centred_hz[p_lo:p_hi]))
    if len(tips) < CURVE_INDEPENDENT_POINTS or len(porches) < CURVE_INDEPENDENT_POINTS:
        return None
    return float(np.median(porches)), float(np.median(tips))


def _refresh_block_model(rf, response, frequency_step, tables, sync_band):
    """Rebuild the pooled block bundle from this field's accumulated state.

    The noise scale that lived here is measured history: a strided-MAD
    noise_ln shipped with the first bundle, was measured to read TWICE the
    honest sync-witnessed floor (the excess being picture-locked structure,
    not noise - 36-45 vs 73-83 mNp on SP), lost its only consumer when the
    head-switch garrote became a regression-measured gain, and was removed
    on Ethan's ruling. The honest floor and the noise's frequency response
    live in the shared inbox (luma_noise_response) and this lane's fold
    instruments; a future noise consumer starts from those, not from a MAD
    over picture. Its inputs - the deviation, the stride it was sampled on
    and the dropout fraction - were still taken here after it went, and are
    now gone with it.
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
    # The response's own reading of which head wrote this field, against
    # the parity the decoder assigned. Disagreement means the accumulations
    # are mixing two heads, which no amount of limit-taking can undo.
    matched = head_from_response(rf, levels, population)
    if matched is not None:
        agree, seen = rf.__dict__.get("_luma_head_check", (0, 0))
        rf._luma_head_check = (
            agree + int(matched == bool(field.isFirstField)), seen + 1)

    line_store = rf.__dict__.setdefault("_luma_eq_lines", {})
    line_key = bool(field.isFirstField)
    line_total, line_weight = line_store.get(line_key, (np.zeros(3), 0.0))
    line_store[line_key] = (line_total + np.array(curve), line_weight + 1.0)

    # The response read off the measurement itself, accumulated per head across
    # the decode. A single field determines it far too poorly - that is why
    # fitting densely per field loses - but a decode's worth of one head's
    # fields determines it well, and the two heads genuinely differ.
    #
    # Measured below as the deviation's own residual rather than by binning
    # the subsample a second time with the collapse divided out. That second
    # binning stood here for a long time and its failure is recorded at the
    # measurement itself: it estimated a quantity related to the residual but
    # not identical to it, and the table it produced could not cancel the
    # residual it described.
    #
    # A steadiness gate stood here for a long time and is measurably better on
    # the bar patterns - about 0.11 on 75bars SP and 0.10 at EP against the
    # record reference, and nothing on chromanoise, which has few transitions
    # for it to act on. It is absent by decision: the gate makes the model
    # describe a sub-population rather than the amplitude, and on real content
    # the deviation is then not flat where it should be.
    tables = sweep_collapse_tables(rf)
    track_arg = np.empty_like(centred_hz)
    slope = np.empty_like(centred_hz)
    _sweep_arrays(centred_hz, tables.span, tables.track_width, track_arg, slope)

    key = bool(field.isFirstField)
    store = rf.__dict__.setdefault("_luma_dense_response", {})
    banks, weights, centre_total = store.setdefault(
        key,
        (np.zeros((2, bin_count)), np.zeros((2, bin_count)), np.zeros(bin_count)),
    )
    # Banked on THIS HEAD's own field count, not on the decode's. Fields
    # alternate heads, so a bank index that flips per field puts every one
    # of a head's fields in the same bank and leaves the other empty - the
    # agreement test then has nothing to compare and reads zero for a
    # reason that has nothing to do with the component. That is a mistake
    # this file made and the measurement it produced was worthless.
    seen = rf.__dict__.setdefault("_luma_field_count", {})
    bank = int(seen.get(key, 0)) % 2
    seen[key] = int(seen.get(key, 0)) + 1
    total = banks.sum(axis=0)
    weight = weights.sum(axis=0)
    # THE MODEL IN FORCE, from the fields already seen and not from this one.
    # Measuring a field through a model its own residual has already entered
    # is fitting, not measuring; keeping this field out is what makes the
    # sequence below an iteration.
    dense = _head_model(rf, key, bin_count)
    sync_band = (rf.iretohz(sys_params["vsync_ire"], spec=True),
                 rf.iretohz(0.0, spec=True))
    residue_edges, residue = sweep_residue_of(rf, key)
    _, joint = joint_residue_of(rf, key, bin_count)
    dropout_fraction = rf.dod_options.dod_threshold_p
    fail_inv = _response_tables(
        response, frequency_step, curve, dense, sync_band,
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
        residue_edges,
        residue,
        joint,
        np.float64(sync_tip_hz),
        np.float64(bin_hz),
    )

    # THE LIMIT. The response is measured as the deviation's OWN residual
    # under the model in force, and that measurement goes back into the
    # model, so each field is one step of an iteration that the decode's own
    # field sequence carries to its limit.
    #
    # It replaces a different estimator, and the difference is the whole
    # point. What stood here binned the flattened subsample with the collapse
    # divided out - a quantity RELATED to the residual but not the same one -
    # and the table it produced could not cancel the residual it was supposed
    # to describe: re-imposing it on the deviation made the trace worse, not
    # better, which is why the table was withheld from the deviation and the
    # withholding measured as an improvement. Measured against itself instead,
    # the residual closes. Offline, driving this to convergence on captured
    # fields takes the binned residual from 24-41 times the bins' own standard
    # error to below it in one pass on chromanoise and two on bars, and the
    # peak-to-peak swing from 7-9.5 per cent to 0.2-0.5.
    #
    # The roll-off survives its own audit under that limit: re-derived from
    # the converged model it lands at -3.296 against -3.303 dB/MHz fitted
    # directly on 75bars head A, -3.299 against -3.323 and -2.956 against
    # -2.988 on chromanoise. The line is not carrying an error the residual
    # was hiding.
    #
    # WHY THE SWEEP AXIS IS NOT OPTIONAL. Taken over frequency alone the
    # limit converges onto the picture: the converged table repeats within
    # one picture at r = +0.98 and across two pictures at only +0.49, and it
    # agrees BETTER between the two heads of one picture than between two
    # pictures of one head - the signature of one axis absorbing what belongs
    # to another. Separating the sweep dimension lifts the frequency
    # component to +0.77 and +0.86. Both main effects are taken; the
    # interaction between them is deliberately left alone, being mostly
    # picture (r = +0.25 and +0.19 across pictures).
    # The frequency table is accumulated on a residual with the TRACK
    # component already removed, so the two do not describe the same
    # structure twice. Without this the frequency table absorbs whatever of
    # the track's profile happens to project onto carrier frequency, and
    # then applies it to every field as though the path's response had that
    # shape. The track model is divided out HERE and nowhere else: the
    # deviation the correction rides on keeps it, because it is the tape.
    residual_samples = np.asarray(deviation[::stride], dtype=np.float64)
    along_track = track_correction(rf, field, deviation, stride, sampled_carrier_hz)
    if along_track is not None:
        residual_samples = residual_samples / np.where(
            along_track > 0.0, along_track, 1.0)
    # And the amplitude this field's OWN time-base error carries, at the
    # measured coupling. The track term above removes the profile the two
    # share along the track; this removes what is particular to this field,
    # which no accumulated profile can hold. Both are the same rule - a
    # component is accumulated on a residual with the others already gone -
    # applied once across the track axis and once within the field.
    coupling = time_base_coupling(rf, key)
    if coupling != 0.0:
        timing = field_time_base(field)
        if timing is not None:
            values, live = timing
            linelocs = np.asarray(getattr(field, "linelocs", ()), dtype=np.float64)
            if len(linelocs) >= len(values):
                index = np.arange(len(residual_samples), dtype=np.float64) * stride
                line_of = np.searchsorted(linelocs, index, side="right") - 1
                inside = (line_of >= 0) & (line_of < len(values))
                place = np.clip(line_of, 0, len(values) - 1)
                carried = np.where(inside & live[place], coupling * values[place], 0.0)
                residual_samples = residual_samples / np.exp(carried)
    ratio_levels, ratio_population, ratio_centre_hz = _binned_median(
        residual_samples, sample_bin, sync_tip_hz, bin_hz, bin_count,
    )
    described = (ratio_population > 0.0) & (ratio_levels > 0)
    # The level this field says the response has: the model it was measured
    # through, times how far the residual departed from unity.
    grid_hz = np.arange(len(response), dtype=np.float64) * frequency_step
    model_ln = _model_log_level(grid_hz, curve, dense, sync_band)
    seen_ln = np.zeros(bin_count)
    seen_ln[described] = np.interp(
        ratio_centre_hz[described], grid_hz, model_ln
    ) + np.log(ratio_levels[described])
    levels_seen = np.zeros(bin_count)
    levels_seen[described] = np.exp(seen_ln[described])
    _measure_reliability(rf, key, levels_seen, ratio_population, described,
                         total, weight)
    banks[bank][described] += seen_ln[described] * ratio_population[described]
    weights[bank][described] += ratio_population[described]
    # The frequency each bin stands at is accumulated the same way its level
    # is, weighted by how many samples went into it. Taking it from THIS
    # field's centres instead - which is what happened - puts a zero wherever
    # a bin accumulated enough from earlier fields but the picture did not
    # visit it this time. The frequencies then stop ascending, `np.interp`
    # reads them as nonsense, and the model swings by 2.5x where the
    # amplitude it describes varies by 1.14x.
    centre_total[described] += ratio_centre_hz[described] * ratio_population[described]

    # THE SECOND DIMENSION, measured on the same residual with the frequency
    # dimension divided out first, so neither axis is handed the other's
    # structure. Indexed on the collapse formula's own theta^2, which is what
    # makes the table the same object from tape to tape.
    #
    # A PARAMETRIC form of the frequency table was built and rejected, and it
    # is worth saying so because it is an attractive thing to rebuild. The
    # ripple was fitted as damped sinusoids by matrix pencil, which on planted
    # signals is exact where peak picking leaves a third of the ripple behind.
    # On real data it does not hold: the singular values of the Hankel matrix
    # decay smoothly - 1, 0.69, 0.20, 0.17, 0.17, 0.14 - where a true sum of
    # sinusoids drops twelve orders of magnitude after its last component, so
    # there is no rank to find, the answer moves with the pencil parameter,
    # and the strongest component disagrees between the two heads by
    # microseconds rather than the nanoseconds a real echo would. The test
    # before trying again is that singular value spectrum: no knee, no
    # parametric form.
    flat_ratio = np.asarray(deviation[::stride], dtype=np.float64)
    scale = np.where(described, ratio_levels, 1.0)[sample_bin]
    usable = (flat_ratio > 0.0) & (scale > 0.0)
    theta2 = np.interp(sampled_carrier_hz, grid_hz, tables.q) * (
        np.asarray(slope[::stride], dtype=np.float64) ** 2
    )
    residue_store = rf.__dict__.setdefault("_luma_sweep_residue", {})
    res_total, res_weight = residue_store.setdefault(
        key, (np.zeros(len(residue_edges)), np.zeros(len(residue_edges)))
    )
    place = np.clip(
        np.searchsorted(residue_edges, theta2, side="right") - 1,
        0, len(residue_edges) - 1,
    )
    logged = np.log(np.where(usable, flat_ratio / np.maximum(scale, 1e-30), 1.0))
    sweep_med, sweep_pop = _cell_medians(
        np.where(usable, place, len(residue_edges)), logged,
        len(residue_edges) + 1,
    )
    keep = sweep_pop[:-1] >= CURVE_INDEPENDENT_POINTS
    res_total[keep] += sweep_med[:-1][keep] * sweep_pop[:-1][keep]
    res_weight[keep] += sweep_pop[:-1][keep]

    # THE PRODUCT COMPONENT, rule R3 of docs/RESIDUAL_LIMIT_DESIGN.md: what
    # neither margin carries, over frequency and sweep rate together. Stored
    # as measured and made orthogonal to both margins where it is read, so
    # the response table stays a response and this cannot put a frequency
    # mean back into it - the coupling that stalls the sequence otherwise.
    joint_store = rf.__dict__.setdefault("_luma_joint_residue", {})
    joint_total, joint_weight = joint_store.setdefault(
        key,
        (np.zeros((2, bin_count, len(residue_edges))),
         np.zeros((2, bin_count, len(residue_edges)))),
    )
    # alternating banks, so the component can be asked whether it says the
    # same thing on evidence it has not seen
    bank = (int(rf.__dict__.get("_luma_field_count", {}).get(key, 1)) - 1) % 2
    cells = np.where(usable, sample_bin * len(residue_edges) + place,
                     bin_count * len(residue_edges))
    cell_med, cell_pop = _cell_medians(
        cells, logged, bin_count * len(residue_edges) + 1)
    cell_med = cell_med[:-1].reshape(bin_count, len(residue_edges))
    cell_pop = cell_pop[:-1].reshape(bin_count, len(residue_edges))
    keep_cell = cell_pop >= CURVE_INDEPENDENT_POINTS
    joint_total[bank][keep_cell] += cell_med[keep_cell] * cell_pop[keep_cell]
    joint_weight[bank][keep_cell] += cell_pop[keep_cell]

    anchors = _measure_level_anchors(field, centred_hz)
    if anchors is not None:
        porch_sum, tip_sum, count = rf.__dict__.get(
            "_luma_level_state", (0.0, 0.0, 0.0))
        rf._luma_level_state = (
            porch_sum + anchors[0], tip_sum + anchors[1], count + 1.0)

    _refresh_block_model(
        rf, response, frequency_step, tables,
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


def _head_model(rf, head, bin_count):
    """One head's response, as the SHARED response plus its own departure
    from it - the head treated as a measured dimension rather than as two
    models kept apart.

    Two independent per-head tables halve the evidence for everything the
    heads have in common, which is most of the response: they are the same
    tape, the same electronics and the same band, differing in the gap that
    wrote them. Decomposed instead, the shared component carries both
    heads' evidence and the head-specific part is measured explicitly and
    admitted on its own - which is also the only form in which the variance
    BETWEEN the heads is described rather than merely duplicated.

    The head term is orthogonal to the shared one by construction (it is
    the departure from the population-weighted mean over heads), so R3
    holds here as it does for the sweep product: applying it cannot put a
    shared mean back into the shared component.

    Admitted by R5's own test, banked on each head's own field count. The
    shared component is not gated - it stands on both heads' evidence and
    is what the line already assumed.
    """
    store = rf.__dict__.get("_luma_dense_response") or {}
    if not store:
        return None
    weight_by_head = {}
    total_by_head = {}
    centre_by_head = {}
    for which, (banks, weights, centre_total) in store.items():
        weight_by_head[which] = weights.sum(axis=0)
        total_by_head[which] = banks.sum(axis=0)
        centre_by_head[which] = centre_total
    mass = sum(weight_by_head.values())
    shared = np.zeros(bin_count)
    lit = mass > 0
    shared[lit] = sum(total_by_head.values())[lit] / mass[lit]
    key = bool(head)
    if key not in store:
        described = mass >= CURVE_MINIMUM_POPULATION
        if np.count_nonzero(described) < CURVE_DENSE_MINIMUM_POINTS:
            return None
        centres = sum(centre_by_head.values())[described] / mass[described]
        return centres, shared[described], mass[described]
    own_weight = weight_by_head[key]
    own = np.zeros(bin_count)
    here = own_weight > 0
    own[here] = total_by_head[key][here] / own_weight[here]
    delta = np.where(here, own - shared, 0.0)
    banks, weights, centre_total = store[key]
    pair = []
    for bank in (0, 1):
        w = weights[bank]
        ok = w > 0
        half = np.zeros(bin_count)
        half[ok] = banks[bank][ok] / w[ok]
        pair.append(np.where(ok, half - shared, 0.0))
    trust = residual_limit.agreement(
        pair[0], pair[1], weights[0], weights[1], CURVE_MINIMUM_POPULATION)
    rf.__dict__.setdefault("_luma_head_trust", {})[key] = trust
    described = own_weight >= CURVE_MINIMUM_POPULATION
    if np.count_nonzero(described) < CURVE_DENSE_MINIMUM_POINTS:
        return None
    return (
        centre_total[described] / own_weight[described],
        (shared + trust * delta)[described],
        own_weight[described],
    )


def measured_response(rf, head):
    """The accumulated luma path response of one head: (curve, dense).

    The public form of what `update_luma_equalizer` reads for itself, so a
    consumer outside this module (the channel identification stage) does
    not depend on the private stores. `head` is the field's isFirstField
    truth value. `curve` is the fitted line (centre_hz, level, slope) with
    the line's own weight folded in; `dense` is (centre_hz, log_level,
    weight) over the bins whose ACCUMULATED population has reached
    CURVE_MINIMUM_POPULATION, or None while fewer than
    CURVE_DENSE_MINIMUM_POINTS have - the same admission the pooled
    consumer applies. Both are natural log of the flattened amplitude, the
    decoder's own path response already divided out. Returns (None, None)
    before the head has been seen.
    """
    key = bool(head)
    lines = rf.__dict__.get("_luma_eq_lines") or {}
    curve = None
    if key in lines and lines[key][1] > 0:
        curve = tuple(lines[key][0] / lines[key][1])
    store = rf.__dict__.get("_luma_dense_response") or {}
    dense = None
    if key in store:
        dense = _head_model(rf, key, len(store[key][2]))
    return curve, dense


def _pooled_dense(rf):
    """Both heads' accumulated responses together, on the model's contract."""
    store = rf.__dict__.get("_luma_dense_response")
    if not store:
        return None
    total = sum(entry[0].sum(axis=0) for entry in store.values())
    weight = sum(entry[1].sum(axis=0) for entry in store.values())
    centre_total = sum(entry[2] for entry in store.values())
    described = weight >= CURVE_MINIMUM_POPULATION
    if np.count_nonzero(described) < CURVE_DENSE_MINIMUM_POINTS:
        return None
    return (
        centre_total[described] / weight[described],
        total[described] / weight[described],
        weight[described],
    )


def _track_axis(field, deviation):
    """The track axis: which bin of the track each strided sample sits in.

    Returns (place, usable, bins, stride) or None. Shared by every
    component measured on this axis, so the amplitude residual and the time
    base's cannot end up on two axes that only look alike.
    """
    from vhsdecode import head_switch

    regions = head_switch.locate(field)
    if not regions:
        return None
    # THE LAST sustained excursion in the field, not the first. `locate`
    # identifies excursions and returns them in positional order; the first
    # of them is not the switch and does not sit still - measured over 25
    # fields it wanders from line 4 to line 261, a standard deviation of
    # 107-115 lines out of 262. The LAST one is the switch: the field's
    # data overhangs its own end, so the switch sits at the boundary, and
    # measured the same way it stands at line 259.6 +- 0.8 on one head and
    # 258.9 +- 0.3 on the other - stable to under a line, and agreeing with
    # the head-switch arc's own reading of the onset at line 259-260.
    #
    # The rule is structural rather than a fitted line number: the switch
    # is where the field ends, so it is the last of them, and no constant
    # is written down here. An axis anchored on the first region is not a
    # track axis at all - it is a different axis every field.
    origin = int(max(region[0] for region in regions))
    if origin < 0 or origin >= len(deviation):
        return None
    linelocs = np.asarray(getattr(field, "linelocs", ()), dtype=np.float64)
    if len(linelocs) < CURVE_INDEPENDENT_POINTS:
        return None
    bins = int(getattr(field, "linecount", 0)) or (len(linelocs) - 1)
    if bins < CURVE_INDEPENDENT_POINTS:
        return None
    switch_line = int(np.searchsorted(linelocs, origin, side="right")) - 1
    # how wide the switch's own excursion is, in lines, measured rather than
    # written down: both heads touch the tape across it, so those lines
    # belong to neither track
    first = int(np.searchsorted(linelocs, min(r[0] for r in regions), side="right"))
    last = int(np.searchsorted(linelocs, max(r[1] for r in regions), side="right"))
    switch_span = max(last - first, 1)
    return origin, bins, switch_line, linelocs, switch_span


def _accumulate(rf, name, key, bins, place, values, usable):
    """One component's cells into its own two-bank store.

    WEIGHTED BY FIELDS, not by samples, and the two components sharing this
    axis are why. The amplitude residual brings thousands of samples to a
    bin and the time base brings one line; weighted by sample count the
    first outvotes the second by three orders of magnitude, and a per-field
    population gate written for the first never admits the second at all -
    which is what happened when this was first written, and the time-base
    component simply never appeared.

    The field is the honest independent unit in any case: samples within a
    field are correlated, the field is what the agreement test resamples,
    and it is what the rest of the tree jackknifes over. So a bin takes
    this field's median once, at unit weight, and its accumulated weight
    counts the FIELDS that have described it. Every component then reads in
    one currency.

    A bin that saw anything contributes. A per-field population test here
    defeats the accumulation, which is the lesson already recorded at the
    dense table: a bin holding a handful of samples a field never
    qualifies, though thirty fields of them describe it well.
    """
    median, population = residual_limit.cell_medians(
        np.where(usable, place, bins), np.where(usable, values, 0.0), bins + 1
    )
    store = rf.__dict__.setdefault(name, {})
    banks, weights = store.setdefault(key, (np.zeros((2, bins)), np.zeros((2, bins))))
    if banks.shape[1] != bins:
        return
    bank = (int(rf.__dict__.get("_luma_field_count", {}).get(key, 1)) - 1) % 2
    seen = population[:bins] > 0
    banks[bank][seen] += median[:bins][seen]
    weights[bank][seen] += 1.0


def head_from_response(rf, levels, population):
    """Which head this field's own response matches, or None.

    Ethan: the COMMON frequency response between the heads determines which
    head is being modelled. Remove what the two share and correlate what is
    left against each head's own departure from it - the head that wrote the
    field is the one it matches. Measured: the right head on 25 of 25 fields
    of 75 bars and 25 of 26 of chromanoise, the own-head correlation +0.587
    and +0.508 against −0.563 and −0.451 for the other, cleanly separated
    and opposite in sign.

    Used here as a CHECK, not as the key. Field parity is reliable in this
    scope and keying on a correlation would be a worse bargain than the
    problem it solves; but parity failing silently would mix the two heads
    into one accumulation, and this notices. It is also the piece
    `head_switch` needs, whose calibration pools both heads because blocks
    in RF-block scope carry no field parity at all.
    """
    store = rf.__dict__.get("_luma_dense_response") or {}
    if len(store) < 2:
        return None
    rows = {}
    for which, (banks, weights, _centre) in store.items():
        weight = weights.sum(axis=0)
        described = weight >= CURVE_MINIMUM_POPULATION
        if np.count_nonzero(described) < CURVE_DENSE_MINIMUM_POINTS:
            return None
        row = np.zeros(len(weight))
        row[described] = banks.sum(axis=0)[described] / weight[described]
        rows[which] = (row, described, weight)
    here = (population > 0) & (levels > 0)
    (a_row, a_ok, a_w), (b_row, b_ok, b_w) = rows[True], rows[False]
    both = a_ok & b_ok & here
    if np.count_nonzero(both) < CURVE_INDEPENDENT_POINTS:
        return None
    w = (a_w + b_w)[both]
    shared = (a_row[both] * a_w[both] + b_row[both] * b_w[both]) / np.maximum(
        a_w[both] + b_w[both], 1e-30)
    mine = np.log(levels[both]) - shared
    scores = {}
    for which, row in ((True, a_row), (False, b_row)):
        theirs = row[both] - shared
        x = mine - np.average(mine, weights=w)
        y = theirs - np.average(theirs, weights=w)
        spread = np.sqrt(np.average(x * x, weights=w) * np.average(y * y, weights=w))
        scores[which] = float(np.average(x * y, weights=w) / spread) if spread > 0 else 0.0
    return max(scores, key=scores.get)


def time_base_coupling(rf, head):
    """How much amplitude a unit of time-base error carries, per head.

    THE SIBLING RELATIONSHIP PUT TO WORK. The amplitude residual and the
    time base's own residual sit on one axis - position along the track -
    and they are coupled: measured there, the slope is −0.0026 on one head
    and +0.0165 on the other (75 bars), −0.0063 on the second head of
    chromanoise. `carrier_tbc` fits the same coupling independently, per
    field and in the drum band alone, and reads −0.0040, +0.0062 and
    −0.0059. Two instruments sharing no arithmetic, agreeing in sign on
    every case and closely in magnitude on one head, and reproducing the
    sign flip between the heads that `carrier_tbc`'s own docstring calls the
    head-contact structure of the wobble.

    So the amplitude a time-base error carries is a MEASURED term of the
    model, not an assumption, and the frequency table is accumulated with it
    removed - otherwise the table absorbs amplitude that belongs to the
    tape's motion and applies it as though the path had that response.

    Zero until both components have described the same bins, which is the
    only condition under which the slope means anything.
    """
    amplitude = (rf.__dict__.get("_luma_track_residue") or {}).get(bool(head))
    timing = (rf.__dict__.get("_luma_time_base_residue") or {}).get(bool(head))
    if amplitude is None or timing is None:
        return 0.0
    values = []
    for banks, weights in (amplitude, timing):
        weight = weights.sum(axis=0)
        described = weight >= CURVE_DENSE_MINIMUM_POINTS
        row = np.zeros(len(weight))
        row[described] = banks.sum(axis=0)[described] / weight[described]
        values.append((row, described, weight))
    (a_row, a_ok, a_w), (t_row, t_ok, _t_w) = values
    both = a_ok & t_ok
    if np.count_nonzero(both) < CURVE_INDEPENDENT_POINTS:
        return 0.0
    w = a_w[both]
    x = t_row[both] - np.average(t_row[both], weights=w)
    y = a_row[both] - np.average(a_row[both], weights=w)
    spread = float(np.average(x * x, weights=w))
    if not spread > 0.0:
        return 0.0
    return float(np.average(x * y, weights=w) / spread)


def field_time_base(field):
    """This field's own per-line time-base residual, against the SPEC period.

    The reference is Ethan's: a perfectly flat and constant time base. Not
    the field's own median, which removes the constant speed error by
    construction - and the constant speed error is exactly what a spec
    reference keeps. Reads `sync_edge_trace`, which `carrier_tbc` also
    reads, so the two cannot drift about where the edges are.

    Returns (values, live) per line interval, or None.
    """
    trace = sync_edge_trace(field)
    if trace is None:
        return None
    rf = field.rf
    edges = np.asarray(trace[0], dtype=np.float64)
    spec = float(rf.SysParams["line_period"]) * rf.freq_hz / 1e6
    if len(edges) < CURVE_INDEPENDENT_POINTS + 1 or not spec > 0.0:
        return None
    live = (edges[:-1] > 0) & (edges[1:] > 0)
    values = np.where(live, (edges[1:] - edges[:-1]) / spec - 1.0, 0.0)
    return values, live


def track_correction(rf, field, deviation, stride, sampled_carrier_hz):
    """The track component's correction for this field's strided samples.

    RULE R3 ACROSS COMPONENTS, not just within one. The frequency table and
    the track table are accumulated from the SAME residual, so without this
    they describe the same structure twice: whatever of the track's profile
    projects onto carrier frequency is absorbed by the frequency table and
    then applied to every field as though it were frequency response. The
    components have to be measured on a residual with the others already
    removed, which is what this returns - the track model, ready to divide
    out before the frequency table is accumulated.

    It is NOT divided out of the deviation itself, and that distinction is
    the whole of the attribution. A frequency response is a property of the
    path and belongs in the model; an amplitude that varies with position
    along the track is the tape and the head reading it, the colour-under
    written beside it took the same loss, and it must reach the correction
    rather than be explained away. The measurement this rests on is that the
    speed signature explains 0.01% of the amplitude residual: the residual
    is not the model read at the wrong frequency, it is real amplitude.

    Ones where the track model has not been measured yet.
    """
    seen = (rf.__dict__.get("_luma_switch_line") or {}).get(bool(field.isFirstField))
    if seen is None or seen[1] <= 0.0:
        return None
    linelocs = np.asarray(getattr(field, "linelocs", ()), dtype=np.float64)
    bins = int(getattr(field, "linecount", 0)) or (len(linelocs) - 1)
    if bins < CURVE_INDEPENDENT_POINTS or len(linelocs) < CURVE_INDEPENDENT_POINTS:
        return None
    residue, trust = track_residue_of(rf, bool(field.isFirstField), bins)
    if trust <= 0.0 or not np.any(residue):
        return None
    switch_line = seen[0] / seen[1]
    index = np.arange(len(sampled_carrier_hz), dtype=np.float64) * stride
    line_of = np.searchsorted(linelocs, index, side="right") - 1
    place = np.mod(line_of - switch_line, bins).astype(np.int64)
    place = np.clip(place, 0, bins - 1)
    # the same bound as the accumulation: past the last line location there
    # is no line, so there is no track position either
    last_line = min(bins, len(linelocs) - 1)
    inside = (line_of >= 0) & (line_of < last_line)
    return np.where(inside, np.exp(residue[place]), 1.0)


def track_residue_of(rf, head, bins, name="_luma_track_residue"):
    """A residual along the TRACK, accumulated per head.

    Ones-equivalent (zeros in the log) wherever nothing has been measured,
    and admitted only as far as it says the same thing on evidence it has
    not seen - the two-bank test of rule R5, banked on this head's own
    field count.
    """
    store = rf.__dict__.get(name) or {}
    entry = store.get(bool(head))
    residue = np.zeros(bins)
    if entry is None:
        return residue, 0.0
    banks, weights = entry
    weight = weights.sum(axis=0)
    # in FIELDS, the unit `_accumulate` counts in, and at the module's own
    # bar for an accumulated table standing on its own evidence
    described = weight >= CURVE_DENSE_MINIMUM_POINTS
    if not np.any(described):
        return residue, 0.0
    residue[described] = banks.sum(axis=0)[described] / weight[described]
    pair = []
    for bank in (0, 1):
        w = weights[bank]
        ok = w > 0
        half = np.zeros(bins)
        half[ok] = banks[bank][ok] / w[ok]
        pair.append(half)
    trust = residual_limit.agreement(
        pair[0], pair[1], weights[0], weights[1],
        CURVE_DENSE_MINIMUM_POINTS / 2.0)
    rf.__dict__.setdefault(name + "_trust", {})[bool(head)] = trust
    return trust * residue, trust


def accumulate_track_residual(field):
    """Both residuals on the track axis: the amplitude's, and the time base's.

    THE AXIS IS THE TRACK, NOT THE FIELD. Each head writes a new track, so
    the head switch is the physical origin and the field boundary is not:
    the field's data overhangs into the next, and `head_switch.locate`
    records that a field can therefore hold two switch regions - its own at
    the first lines and the next field's at the tail. Binning on the field
    would smear the start of one track against the end of another.

    The switch is not declared by the format - `head_switches_per_field` is,
    its position is not - so it is measured, and measured ONCE: this reads
    `head_switch.locate`, which finds the switch from this stage's own
    per-field deviation at tens of sigma. The two therefore cannot disagree
    about where the switch is, which a second locator here would eventually
    do.

    BINS ARE LINE-ALIGNED, and that is load bearing rather than tidy.
    Uniform divisions of the track beat against the line structure: the
    residual's line-locked profile is large - largest at the sync phases,
    where the response's low end is extrapolated - so bins each covering a
    DIFFERENT phase mix turn that profile into a structured wander of the
    track axis. Measured both ways on 75bars SP over 25 fields: uniform
    bins read 0.43 dB rms at an agreement of 0.977, line-aligned bins
    0.186 dB at 0.757 - more than half of the first figure was the beat,
    and it agreed with itself so well because it is deterministic geometry
    rather than tape. `head_switch.locate` records the same trap burying a
    130 mNp switch under a 50-70 mNp wander.

    TWO COMPONENTS, ONE AXIS. The amplitude residual is what this stage
    measures; the time base's is the measured line period against the SPEC
    period, which is Ethan's reference - a perfectly flat and constant time
    base. Both are binned here, together, because a coupling between two
    components can only be measured where they share an axis.

    The spec reference is deliberate and differs from `carrier_tbc`, which
    divides by the field's own median period: relative to the median, the
    constant speed error is gone by construction, and the constant speed
    error is precisely what a spec reference keeps. Both read the same
    `sync_edge_trace`, so the two cannot drift apart about where the edges
    are.

    Called AFTER the memo is set, and that is structural: `locate` asks for
    the measurement, so a call from inside `measure_amplitude_deviation`
    would recurse.

    Measurement only. Nothing here reaches the model yet.
    """
    measured = measured_amplitude_deviation(field)
    if measured is None:
        return None
    deviation = np.asarray(measured.deviation, dtype=np.float64)
    axis = _track_axis(field, deviation)
    if axis is None:
        return None
    origin, bins, switch_line, linelocs, switch_span = axis
    rf = field.rf
    key = bool(field.isFirstField)
    # Remember where the switch was, so the MEASUREMENT can place samples on
    # the track axis without asking the locator again - which it cannot do,
    # the locator asking for the measurement in turn. A running mean is
    # enough: the switch stands at line 259.6 +- 0.8 and 258.9 +- 0.3, under
    # a line of scatter, and the format's own guide puts it 6.014 H ahead of
    # V-sync, which is the same place.
    seen_switch = rf.__dict__.setdefault("_luma_switch_line", {})
    total_line, count_line = seen_switch.get(key, (0.0, 0.0))
    seen_switch[key] = (total_line + float(switch_line), count_line + 1.0)

    # the amplitude residual, on the strided subsample
    stride = max(
        1, len(deviation) // (CURVE_INDEPENDENT_POINTS * CURVE_SAMPLES_PER_POINT)
    )
    sampled = deviation[::stride]
    index = np.arange(len(sampled), dtype=np.float64) * stride
    line_of = np.searchsorted(linelocs, index, side="right") - 1
    place = (line_of - switch_line) % bins
    # PAST THE LAST LINE LOCATION THERE IS NO LINE. `linelocs` stops short
    # of the field's data - 273 entries ending near sample 761,000 of
    # 983,040 - and `searchsorted` gives every sample beyond it the same
    # index, so a quarter of the field piles into ONE bin. Measured: 26.8%
    # of samples in one place, worth +0.56 dB on one head and -0.83 on the
    # other, and the whole track residual moved 0.150/0.222 -> 0.128/0.181
    # dB when they were dropped. They are not a line and they are not this
    # bin's evidence.
    last_line = min(bins, len(linelocs) - 1)
    usable = (sampled > 0.0) & (line_of >= 0) & (line_of < last_line)
    _accumulate(rf, "_luma_track_residue", key, bins, place,
                np.log(np.where(usable, sampled, 1.0)), usable)

    # THE SWITCH'S OWN REGION IS EXCLUDED, and it is not a detail: both
    # heads touch the tape across the switch, so the lines there belong to
    # neither track. Measured, excluding it takes the time-base residual on
    # this axis from 0.109/0.127 per cent to 0.0136/0.0169 - an order of
    # magnitude - with single bins inside the window carrying 1.49 and 2.01
    # per cent on their own, and the alternating-field agreement falling to
    # nothing once the window is widened. What was being measured there was
    # the switch, not the track.
    #
    # The window comes from the located region itself rather than from a
    # line count written down here: `locate` returns the excursion's own
    # start and end, which is the measurement, and the format's overlap is
    # about eight lines on this deck.
    clear_of_switch = np.minimum(place, (bins - place) % bins) > switch_span
    usable &= clear_of_switch

    # the time base's own residual, per line, against the spec period
    trace = sync_edge_trace(field)
    if trace is not None:
        edges = np.asarray(trace[0], dtype=np.float64)
        spec = float(rf.SysParams["line_period"]) * rf.freq_hz / 1e6
        count = min(len(edges) - 1, len(linelocs) - 1)
        if count >= CURVE_INDEPENDENT_POINTS and spec > 0.0:
            lengths = edges[1:count + 1] - edges[:count]
            live = (edges[:count] > 0) & (edges[1:count + 1] > 0)
            periods = np.where(live, lengths / spec - 1.0, 0.0)
            lines = (np.arange(count) - switch_line) % bins
            live = live & (
                np.minimum(lines, (bins - lines) % bins) > switch_span)
            _accumulate(rf, "_luma_time_base_residue", key, bins, lines,
                        periods, live)
    return track_residue_of(rf, key, bins)


def measured_amplitude_deviation(field):
    """`measure_amplitude_deviation`, computed once per field.

    Two consumers want the same measurement and it walks the field several
    times, so the second one reads the first one's answer.
    """
    cached = getattr(field, "_luma_amplitude_measured", None)
    if cached is None:
        cached = measure_amplitude_deviation(field)
        field._luma_amplitude_measured = cached
        # The track axis, here rather than at either call site, because the
        # measurement is triggered from two of them - the field's own hook
        # and the chroma path - and the residual must be accumulated once
        # per field however it was asked for. It runs AFTER the memo is
        # set: the switch locator asks for the measurement, and the guard
        # makes the re-entry that follows a no-op rather than a recursion.
        if cached is not None and not getattr(field, "_luma_track_done", False):
            field._luma_track_done = True
            accumulate_track_residual(field)
    return cached


