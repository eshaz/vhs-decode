import math
import numpy as np
import lddecode.utils as lddu
import lddecode.core as ldd
import scipy.signal as sps
import scipy.fft as sps_fft
from vhsdecode.rust_utils import sosfiltfilt_rust
from vhsdecode import luma_amplitude
from vhsdecode import luma_beat

import numba
from numba import njit
from numba.experimental import jitclass
from functools import cache

BURST_START_LINE=10


# ---------------------------------------------------------------------------
# Color-under amplitude correction from the luma carrier's amplitude
#
# An FM carrier is recorded at constant amplitude, so every departure from
# constant is loss the tape imposed. The luma carrier and the color-under were
# written by the same head, onto the same oxide, at the same instant, so they
# shared that loss. `vhsdecode.luma_amplitude` measures the departure;
# everything here scales it across to the color-under and reverses it.
#
# Both happen on the raw RF, before the time base correction and
# before anything has been measured from the color-under: the tape imposed the
# loss at that rate, so it is undone at that rate.

WAVELENGTH_INDEPENDENT_NOISE = 0.194

# How far the measurement can be trusted.
#
# The reference is itself noisy, so the correction is scaled back by the Wiener
# weight - signal power over signal plus noise power:
#
#     weight = track width / (track width + half-coupling width)
#
# The noise is particulate, and a wider track reproduces from proportionally
# more particles, so the half-coupling width is the track width at which the
# two powers meet. This is why the slower speeds need the correction held back
# so much harder: EP lays down a third of SP's track. Estimated alongside the
# fraction above.
HALF_COUPLING_TRACK_WIDTH = 18.29

# Used only when a format supplies no track width of its own.
REFERENCE_TRACK_WIDTH = 58.0

# How far the transfer above survives as the loss gets faster.
#
# The trust weight settles how much of the luma's deviation belongs on the
# color-under, but not how quickly. Head to medium separation is mechanical and
# therefore slow; the luma envelope's own noise is broadband. So the fraction of
# the deviation worth transferring is a Wiener weight in frequency as well, and
# it falls away as the modulation gets faster - measured against the
# color-under's own amplitude it is the whole of the modelled transfer at a few
# tens of kHz and about half of it by 150 kHz, on both speeds and both test
# patterns.
#
# Applied flat, as it was, the gain therefore spends most of its excursion where
# it predicts nothing, and the only defence is to scale the whole correction
# back - which is why the slow part, the part that is real, ends up under
# applied. Measuring the roll-off and imposing it lets the rest come up: the
# best flat amount measured 0.23-0.72 across the test patterns and rises to
# 0.38-1.12 once shaped.
#
# The shape is not a chosen curve. No single pole order describes it (chi2/dof
# 2-24 for every one tried) and it differs between tapes and speeds, so it is
# measured per field by regressing the color-under's own amplitude on the luma's
# deviation, band by band.

# Bands the transfer is measured in, across the color-under's modulation range.
# The response is read off these rather than fitted through them, so they are
# kept well above what a roll-off needs.
TRANSFER_BANDS = 12

# Burst amplitude below which the color killer decides the field carries no
# color-under at all. It is the decoder's one definition of "is there a burst",
# and anything that needs to ask reads it here rather than measuring its own.
#
# TODO Expose as option, possible this needs to be relative to the sync pulse and level detection
BURST_MAGNITUDE_THRESHOLD = 2.5e4

# Fields the measured response is averaged over. The regression is noisy in a
# single field - the color-under's own picture content is the larger part of
# what it sees - but the transfer is a property of the head and the tape, so it
# holds across a decode.
TRANSFER_AVERAGE_FIELDS = 30

# Fields between measurements, once that average has something in it. Measuring
# every field resolves a quantity that is constant over the whole averaging
# window several times over; on this stride the window still turns over inside
# two seconds of tape. The shaping is applied every field regardless - only the
# regression behind it is strided.
TRANSFER_MEASURE_STRIDE = 4


@njit(cache=True, nogil=True, fastmath=True)
def _transfer_exponent(
    deviation,
    carrier_hz,
    steadiness,
    color_under_carrier_hz,
    wavelength_independent,
    trust,
    dropout_fraction,
    exponent,
):
    """The power the luma's amplitude deviation is raised to, per sample.

    The transfer above, evaluated against the luma carrier's own instantaneous
    frequency since that is what set its wavelength at the moment in question,
    and scaled by how far this track width lets the measurement be trusted.

    Confidence is a second Wiener weight, on the measurement rather than the
    medium: one where the carrier sits at its normal level, falling away where
    it has collapsed, because scaling the chroma up there would restore noise
    rather than saturation. Steadiness withdraws it again wherever the carrier
    is sweeping too fast for the path to follow, since the amplitude there is
    reporting the path rather than the tape - see `path_transient_scale`.

    Arithmetic only, into a caller-supplied array; raising the deviation to it
    is left to the ufunc that can do eight samples at a time.
    """
    count = len(deviation)
    half = dropout_fraction * dropout_fraction
    scale = np.float32(1.0) + half
    unity = np.float32(1.0)
    for i in range(count):
        value = deviation[i]
        squared = value * value
        confidence = squared * scale / (squared + half) * steadiness[i]
        wavelength_ratio = color_under_carrier_hz / carrier_hz[i]
        transfer = (
            wavelength_independent
            + (unity - wavelength_independent) * wavelength_ratio
        ) * trust
        exponent[i] = -transfer * confidence


def _envelope_gain_from_deviation(
    deviation,
    carrier_hz,
    steadiness,
    color_under_carrier_hz,
    wavelength_independent,
    trust,
    dropout_fraction,
):
    """The color-under gain that reverses the luma's measured amplitude loss.

    `deviation ** exponent`, split in two. The exponent is assembled in a
    compiled loop, which is what branchless per-sample arithmetic is for, and
    the power is handed to numpy, whose single precision loop is vectorised
    where numba's is not - numba can only vectorise transcendentals through a
    vector math library this build cannot reach. The same arithmetic either
    way, an order of magnitude apart in cost.
    """
    gain = np.empty(len(deviation), dtype=np.float32)
    _transfer_exponent(
        deviation,
        carrier_hz,
        steadiness,
        np.float32(color_under_carrier_hz),
        np.float32(wavelength_independent),
        np.float32(trust),
        np.float32(dropout_fraction),
        gain,
    )
    return np.power(deviation, gain, out=gain)


@njit(cache=True, nogil=True, fastmath=True)
def _scale_color_under_complex(chroma, chroma_quad, gain, phase, amount,
                               phase_amount):
    """Scale AND rotate the color-under: the correction's complete form.

    The amplitude disturbance the tape imposed has a causal phase partner
    (the same Hilbert relation the head-switch correction stands on), and a
    narrowband signal is rotated by phi via its own quadrature:

        out = G * (chroma * cos(p*phi) - H{chroma} * sin(p*phi))

    with G the existing gain envelope. At phase_amount = 0 this is exactly
    the real path. `chroma_quad` is the Hilbert transform of the color-under
    along time, computed once per field by the caller.
    """
    one = np.float32(1.0)
    for i in range(len(gain)):
        g = one + (gain[i] - one) * amount
        rot = phase[i] * phase_amount
        chroma[i] = g * (
            chroma[i] * np.cos(rot) - chroma_quad[i] * np.sin(rot)
        )


@njit(cache=True, nogil=True, fastmath=True)
def _scale_color_under(chroma, gain, amount):
    """Scale the color-under by the correction, in place, on the RF grid.

    A plain multiply suffices: the color-under has been band passed and had its
    block mean subtracted, so it carries no DC term for a time varying gain to
    turn into a real component at the subcarrier frequency.
    """
    one = np.float32(1.0)
    for i in range(len(gain)):
        chroma[i] *= one + (gain[i] - one) * amount


@njit(cache=True, nogil=True, fastmath=True)
def _line_comb_pair(deviation, power, linelocs, first_line, last_line, mean_power, x, y):
    """Both signals, less the same phase one line earlier, in one pass.

    On a repeating picture both the luma deviation and the color-under's
    amplitude are dominated by content locked to the line, which shares its
    harmonics with the other and so produces high coherence at arbitrary phase.
    The transfer cannot be read through that. Subtracting the previous line
    removes everything picture-locked and leaves the tape's own contribution.
    It is linear and applied to both signals identically, so the transfer it
    measures is unchanged.

    The previous line is found from the decoder's own line locations rather
    than a nominal line length, so the comb follows the time base instead of
    assuming there isn't one.
    """
    out = 0
    limit = len(x)
    for line in range(first_line, last_line):
        this_start = linelocs[line]
        this_length = linelocs[line + 1] - this_start
        previous_start = linelocs[line - 1]
        previous_length = this_start - previous_start
        if this_length <= 0.0 or previous_length <= 0.0:
            continue
        scale = previous_length / this_length
        begin = int(np.ceil(this_start))
        stop = int(np.floor(linelocs[line + 1]))
        for position in range(begin, stop):
            if out >= limit:
                return out
            source = previous_start + (position - this_start) * scale
            lower = int(source)
            fraction = np.float32(source - lower)
            back = deviation[lower] + (deviation[lower + 1] - deviation[lower]) * fraction
            x[out] = (deviation[position] - back)
            back = power[lower] + (power[lower + 1] - power[lower]) * fraction
            y[out] = (power[position] - back) / mean_power
            out += 1
    return out


def _isotonic(values, weight):
    """Weighted least squares fit under a non-increasing constraint.

    Pool adjacent violators. The transfer is a Wiener weight against a noise
    that grows with frequency, so it can only fall; imposing that is what makes
    a per-field estimate usable without choosing a curve for it to follow.
    """
    level = list(values)
    mass = list(weight)
    count = [1] * len(level)
    i = 0
    while i < len(level) - 1:
        if level[i] >= level[i + 1]:
            i += 1
            continue
        total = mass[i] + mass[i + 1]
        level[i] = (level[i] * mass[i] + level[i + 1] * mass[i + 1]) / total
        mass[i] = total
        count[i] += count[i + 1]
        del level[i + 1], mass[i + 1], count[i + 1]
        if i > 0:
            i -= 1
    return np.repeat(np.array(level), count)


def _amount_probe(field, **measured):
    """Record what the transfer regression saw, without acting on any of it.

    Gated by the caller on the plot that reads it, since assembling this is
    several arrays and a second isotonic fit per measurement.

    The regression already produces, per field and per head, the level of the
    transfer and its standard error - the very number the track width model
    predicts - and then normalises it away. This keeps a copy so the two can be
    compared, and counts how often the measurement fails to resolve at all,
    which is how often the unshaped fallback is reached.
    """
    probe = field.rf.__dict__.setdefault("_chroma_amount_probe", {})
    state = probe.setdefault(bool(field.isFirstField), [])
    state.append(measured)


def _project_level(transfer, error, shape):
    """The transfer's level along a shape, and the variance of that level.

    Inverse variance weighted least squares of the measured band transfers on
    the shape they are supposed to follow, which is the level the shape is
    normalised by before it is stored. Taken over every band that could be
    measured rather than over the bands that passed a significance test:
    selecting bands on the size of their own estimate and then reading a level
    off the survivors biases the level upward by the selection.
    """
    usable = (error > 0.0) & (shape > 0.0)
    if np.count_nonzero(usable) < 2:
        return float("nan"), float("nan")
    precision = 1.0 / (error[usable] * error[usable])
    denominator = float((shape[usable] * shape[usable] * precision).sum())
    if not denominator > 0.0:
        return float("nan"), float("nan")
    level = float((transfer[usable] * shape[usable] * precision).sum()) / denominator
    return level, 1.0 / denominator


def _measure_transfer_response(field, deviation, half_width_hz):
    """How much of the modelled transfer survives, band by band.

    Regresses the color-under's own amplitude on the luma's deviation. The
    color-under is squared rather than enveloped - no analytic signal, no
    filter - which puts its amplitude at baseband alongside a copy around twice
    the under-carrier that is far outside the range this looks at.

    Returns the band centres and the response normalised to unity where the
    transfer is strongest, or None where the field cannot support it. Nothing
    about the level is taken from here: that is the track width's business, and
    this carries only the shape.
    """
    video = field.data["video"]
    linelocs = np.asarray(field.linelocs, dtype=np.float64)
    if len(linelocs) < 20:
        return None

    # The active field. There is no color-under in the vertical interval, so a
    # ratio measured there is noise over nothing.
    first = field.lineoffset + 12
    last = min(len(linelocs) - 2, field.lineoffset + field.linecount)
    if last <= first:
        return None
    start = int(np.ceil(linelocs[first]))
    end = min(int(np.floor(linelocs[last])), len(deviation))

    # Long enough that the comb's nulls, which sit at multiples of the line
    # rate, fall inside a segment rather than across it.
    samples_per_line = float(np.median(np.diff(linelocs)))
    segment = 1 << int(np.ceil(np.log2(max(2.0 * samples_per_line, 256.0))))
    if end - start < 4 * segment:
        return None

    burst = np.asarray(video["demod_burst"], dtype=np.float32)
    power = burst * burst
    mean_power = float(power[start:end].mean())
    if not mean_power > 0.0:
        return None

    span = end - start
    x = np.empty(span, dtype=np.float32)
    y = np.empty(span, dtype=np.float32)
    kept = _line_comb_pair(
        np.asarray(deviation, dtype=np.float32),
        power,
        linelocs,
        first,
        last,
        np.float32(mean_power),
        x,
        y,
    )
    if kept < 4 * segment:
        return None

    # Abutting rather than overlapping segments. Half the transforms, and the
    # response is averaged over whole fields anyway.
    count = kept // segment
    taper = np.hanning(segment).astype(np.float32)
    fx = sps_fft.rfft(x[: count * segment].reshape(count, segment) * taper, axis=1)
    fy = sps_fft.rfft(y[: count * segment].reshape(count, segment) * taper, axis=1)
    sxx = np.einsum("ij,ij->j", fx.conj(), fx).real.astype(np.float64)
    sxy = np.einsum("ij,ij->j", fx.conj(), fy).real.astype(np.float64)
    syy = np.einsum("ij,ij->j", fy.conj(), fy).real.astype(np.float64)

    frequency = np.fft.rfftfreq(segment, 1.0 / field.rf.freq_hz)
    edges = np.linspace(0.0, half_width_hz, TRANSFER_BANDS + 1)
    centre = 0.5 * (edges[:-1] + edges[1:])
    response = np.zeros(TRANSFER_BANDS)
    weight = np.zeros(TRANSFER_BANDS)
    raw_transfer = np.zeros(TRANSFER_BANDS)
    raw_error = np.zeros(TRANSFER_BANDS)
    band_sxx = np.zeros(TRANSFER_BANDS)
    band_sxy = np.zeros(TRANSFER_BANDS)
    band_syy = np.zeros(TRANSFER_BANDS)
    band_coherence = np.zeros(TRANSFER_BANDS)
    # The transform's frequencies ascend, so which band a bin belongs to is a
    # position in them rather than a mask over all of them.
    bounds = np.searchsorted(frequency, edges)
    for b in range(TRANSFER_BANDS):
        low, high = bounds[b], bounds[b + 1]
        a, c, d = sxx[low:high].sum(), sxy[low:high].sum(), syy[low:high].sum()
        if not (a > 0.0 and d > 0.0):
            continue
        transfer = c / a
        coherence = transfer * transfer * a / d
        samples = max(int(high - low) * count - 1, 1)
        error = np.sqrt(max((d / a) * (1.0 - coherence), 0.0) / samples)
        raw_transfer[b] = transfer
        raw_error[b] = error
        band_sxx[b], band_sxy[b], band_syy[b] = a, c, d
        band_coherence[b] = coherence
        response[b] = transfer
        weight[b] = 1.0 / (error * error)

    # Every band that could be measured takes part, carrying the weight of how
    # well it was measured. There is no significance test in front of this, and
    # removing it is what makes the level below usable.
    #
    # A band that resolves nothing already contributes almost nothing: the fit
    # is inverse-variance weighted, so its weight is its own precision. The
    # test that used to stand here dropped it entirely instead, which cost two
    # ways. It discarded a usable shape - measured, 75bars EP and home.flac
    # never once cleared three sigma in three bands, so the transfer never
    # resolved at all and every field fell back on no roll-off with the whole
    # amount applied wideband. And selecting bands on the size of their own
    # estimate, then reading a level off the survivors, biases that level
    # upward by the selection - which matters now that the level is read.
    used = weight > 0.0
    if luma_amplitude.wants_averaging_probe(field.rf):
        # What the same bands say with no significance gate in front of them,
        # and what the gate then leaves. Reported only - the gate still decides.
        measurable = raw_error > 0.0
        ungated_shape = np.zeros(TRANSFER_BANDS)
        if np.count_nonzero(measurable) >= 2:
            pooled = _isotonic(
                raw_transfer[measurable], 1.0 / (raw_error[measurable] ** 2)
            )
            if pooled[0] > 0.0:
                ungated_shape[measurable] = np.clip(pooled / pooled[0], 0.0, 1.0)
        ungated_level, ungated_var = _project_level(
            raw_transfer, raw_error, ungated_shape
        )
        _amount_probe(
            field,
            resolved=bool(np.count_nonzero(used) >= 2),
            measurable=int(np.count_nonzero(measurable)),
            # How many bands would have stood three standard errors clear on
            # their own. Reported so the gate's removal can be seen, not used.
            passing=int(np.count_nonzero(raw_transfer > 3.0 * raw_error)),
            band_hz=centre.copy(),
            raw_transfer=raw_transfer.copy(),
            raw_error=raw_error.copy(),
            ungated_level=ungated_level,
            ungated_var=ungated_var,
            model_exponent=float(
                getattr(field, "chroma_model_exponent", float("nan"))
            ),
            sxx=band_sxx.copy(),
            sxy=band_sxy.copy(),
            syy=band_syy.copy(),
            coherence=band_coherence.copy(),
        )
    if np.count_nonzero(used) < 2:
        return None

    fitted = _isotonic(response[used], weight[used])
    peak = fitted[0]
    if not peak > 0.0:
        return None
    fitted = np.clip(fitted / peak, 0.0, 1.0)

    # Onto the fixed grid, so a field's answer means the same thing as every
    # other field's and they can simply be averaged. Below the lowest band that
    # resolved, the response is held: the transfer cannot rise with frequency.
    # Above the highest, it runs down to zero at the half-width - the bands in
    # between showed no transfer to measure, and the color-under cannot carry
    # amplitude modulation past the half-width at all, so falling to zero there
    # is the mildest continuation that respects both.
    measured_hz = centre[used]
    filled = np.interp(centre, measured_hz, fitted, left=fitted[0], right=np.nan)
    beyond = centre > measured_hz[-1]
    if beyond.any():
        span = half_width_hz - measured_hz[-1]
        filled[beyond] = fitted[-1] * np.clip(
            1.0 - (centre[beyond] - measured_hz[-1]) / max(span, 1.0), 0.0, 1.0
        )

    # WAS THE SHAPE RESOLVED AT ALL? Recorded, never gated - see below.
    #
    # The per-band three-sigma test that used to stand in front of the fit
    # was removed for good reasons stated above, and this is not it coming
    # back: it selects no bands and changes no result. It asks the whole
    # measurement one question instead - does the coupling stand clear of
    # what the same pipeline returns when the coupling is destroyed?
    #
    # That question is worth asking because the inverse-variance argument
    # for dropping the gate holds only while SOME band resolves. When none
    # does, a weighted combination of unresolved bands is still a confident
    # looking shape, and nothing downstream can tell.
    #
    # Measured with a matched null - the chroma power's segment index
    # rolled, which destroys the pairing while leaving both spectra
    # bit-identical - the two regimes are two orders of magnitude apart:
    #
    #   chromanoise   peak coherence 0.307 against a null of 0.021   14x
    #   75bars SP                    0.133             of 0.029      4.6x
    #   home.flac                    0.0008            of 0.0009     1x
    #
    # So the instrument is sound and resolves the coupling wherever the
    # coupling is driven; home simply does not drive it, which is the same
    # material the comment above already names as never clearing three
    # sigma. What is new is that on that material the retained shape is
    # indistinguishable from noise, and `_project_level` reads the applied
    # LEVEL off it. Reported so the decision can be made on evidence.
    field.chroma_envelope_coherence = float(np.max(band_coherence))
    return centre, filled


def _shape_gain(gain, band_hz, band_response, freq_hz, half_width_hz):
    """Impose the measured response on the correction, in place, zero phase.

    The gain is shaped as `gain - 1` rather than in the logarithm: the
    correction runs at a few percent, so the two agree to second order, and
    this is the form the scaling below already consumes. Above the frequency
    the color-under can carry amplitude modulation at all the response is zero,
    so the gain is band limited whatever the regression found.

    Applied over the whole field at once as a real, even mask, which is zero
    phase by construction - a causal filter here would slide the correction off
    the loss it is meant to reverse.
    """
    excursion = gain - np.float32(1.0)
    spectrum = sps_fft.rfft(excursion)
    frequency = np.fft.rfftfreq(len(excursion), 1.0 / freq_hz)
    # Anchored flat below the first band and at zero on the half-width, so the
    # mask runs down to nothing rather than stepping off wherever the
    # measurement ran out.
    knots_hz = np.concatenate(([0.0], band_hz, [half_width_hz]))
    knots = np.concatenate(([band_response[0]], band_response, [0.0]))
    mask = np.interp(frequency, knots_hz, knots, left=knots[0], right=0.0)
    spectrum *= mask.astype(spectrum.real.dtype)
    gain[:] = np.float32(1.0) + sps_fft.irfft(spectrum, n=len(excursion)).astype(
        np.float32
    )
    return gain


def apply_chroma_envelope_gain(field):
    """Reverse the tape's amplitude noise on the color-under, in place.

    Whatever in the luma carrier's amplitude does not follow its own frequency
    is tape noise rather than picture, and the color-under written beside it
    took the same loss. The reverse is applied to the field's own copy of the
    color-under, once, before the bursts are measured - so everything
    downstream reads a signal that noise has already been taken out of.

    Returns whether it did anything.
    """
    if getattr(field, "chroma_envelope_gain_applied", False):
        return False

    rf = field.rf
    decoder_params = rf.DecoderParams
    phase_amount = float(getattr(rf.options, "chroma_env_phase", 0.0) or 0.0)
    if (rf.options.chroma_env_gain <= 0 and phase_amount == 0.0) or rf.do_cafc:
        # Under chroma AFC this channel is still raw RF, carrying the ADC's own
        # offset, and the band pass that makes it zero mean has not run yet.
        return False
    if (
        "color_under_carrier" not in decoder_params
        or "chroma_bpf_upper" not in decoder_params
    ):
        return False

    video = field.data["video"]
    if video is None or "demod_burst" not in video:
        return False

    measured = luma_amplitude.measured_amplitude_deviation(field)
    if measured is None:
        return False
    deviation, carrier_hz = measured.deviation, measured.carrier_hz

    debug_plot = getattr(rf, "debug_plot", None)
    if debug_plot and debug_plot.is_plot_requested("luma_noise"):
        # The profile the chroma is actually scaled by, kept for the plot so it
        # can be shown against the luma it was supposed to be freed of, along
        # with the response model it was measured against.
        field.chroma_envelope_deviation = deviation
        field.chroma_envelope_response = measured.response

    # How far the measurement can be trusted at this track width.
    track_width = decoder_params.get("video_track_width", REFERENCE_TRACK_WIDTH)
    trust = track_width / (track_width + HALF_COUPLING_TRACK_WIDTH)

    # Belief is withdrawn per sample where the carrier was sweeping too fast for
    # the path to follow, since the amplitude there is reporting the path rather
    # than the tape.
    belief = measured.steadiness

    gain = _envelope_gain_from_deviation(
        deviation,
        carrier_hz,
        belief,
        decoder_params["color_under_carrier"],
        WAVELENGTH_INDEPENDENT_NOISE,
        trust,
        rf.dod_options.dod_threshold_p,
    )

    # The amount the model asks for, kept so the regression's own level can be
    # read against it. Recomputed on a stride rather than captured from the
    # exponent above, which `np.power` has already overwritten in place - and
    # only where the plot that reads it was asked for.
    if luma_amplitude.wants_averaging_probe(rf):
        probe_stride = 64
        probe_carrier = np.asarray(carrier_hz[::probe_stride], dtype=np.float64)
        probe_deviation = np.asarray(deviation[::probe_stride], dtype=np.float64)
        probe_belief = np.asarray(belief[::probe_stride], dtype=np.float64)
        half = float(rf.dod_options.dod_threshold_p) ** 2
        squared = probe_deviation * probe_deviation
        confidence = squared * (1.0 + half) / (squared + half) * probe_belief
        ratio = decoder_params["color_under_carrier"] / np.maximum(probe_carrier, 1.0)
        field.chroma_model_exponent = float(
            np.mean(
                (
                    WAVELENGTH_INDEPENDENT_NOISE
                    + (1.0 - WAVELENGTH_INDEPENDENT_NOISE) * ratio
                )
                * trust
                * confidence
            )
        )

    # The band the color-under can carry amplitude modulation in at all. Beyond
    # it a gain does not correct the chroma, it modulates chroma out of its own
    # band for the final band pass to discard.
    half_width_hz = (
        decoder_params["chroma_bpf_upper"] - decoder_params["color_under_carrier"]
    )
    # A format whose two chroma parameters do not describe a color-under band -
    # they mean something else there - has no modulation range to shape within,
    # and is left as it was.
    if half_width_hz > 0.0:
        # Kept per head: the two differ, and pooling them averages a real
        # distinction away. Same split, and the same reason, as the chroma gain.
        history = rf.field_averages.chroma_transfer_for(field.isFirstField)
        counters = rf.__dict__.setdefault("_chroma_transfer_fields", {})
        seen = counters.get(field.isFirstField, 0)
        counters[field.isFirstField] = seen + 1
        # Measured on every field until the average has something to say, on a
        # stride after that.
        if len(history) < TRANSFER_BANDS or seen % TRANSFER_MEASURE_STRIDE == 0:
            resolved = _measure_transfer_response(field, deviation, half_width_hz)
            if resolved is not None:
                history.append(resolved)
        # This head's own measured shape, or the other head's until it has one.
        #
        # A field contributes nothing unless its fitted transfer has a positive
        # part, which is right - a field whose measurement says the coupling
        # runs the wrong way carries no usable shape. But that test turns on the
        # sign of the lowest band, and on difficult material that sign is not
        # resolved: measured on an off-air tape, band zero came out positive on
        # 71% of one head's fields and 10% of the other's, at 0.6 and 1.6 sigma.
        # One head was then shaped on every field and the other on a fifth of
        # them, from a distinction the data does not support.
        #
        # The two heads' transfers differ by about 15% of the transfer's own
        # level, so standing in for one with the other is far closer than the
        # alternative, which is applying no correction at all.
        borrowed = False
        if not history:
            history = rf.field_averages.chroma_transfer_for(not field.isFirstField)
            borrowed = bool(history)
        if history:
            # Every field answers on the same grid, so this is a plain mean.
            band_hz = history[-1][0]
            band_response = np.mean([response for _, response in history], axis=0)
            _shape_gain(gain, band_hz, band_response, rf.freq_hz, half_width_hz)
            field.chroma_envelope_shaped = True
            field.chroma_envelope_borrowed = borrowed
        else:
            # Neither head has resolved yet, so there is nothing to shape by and
            # the correction is not applied. Holding the shape flat instead -
            # band limiting only - applies the whole amount across the whole
            # band on the strength of a transfer that was never measured, and
            # measured, that is worse than leaving the color-under alone.
            gain[:] = np.float32(1.0)
            field.chroma_envelope_shaped = False
            field.chroma_envelope_borrowed = False

        if (
            history
            and debug_plot
            and debug_plot.is_plot_requested("luma_noise")
        ):
            field.chroma_envelope_transfer = (band_hz, band_response)

    amount = np.float32(rf.options.chroma_env_gain)

    if (debug_plot and debug_plot.is_plot_requested("luma_noise")) or getattr(
        rf, "_residual_channels_dir", None
    ):
        # What actually multiplies the color-under, wet/dry mix included. The
        # scaling below applies this without materialising it, so it is built
        # here only when something is going to look at it - the plot, or the
        # residual channel export, which carries it as `chroma_amplitude`.
        field.chroma_envelope_correction = 1.0 + (gain - 1.0) * amount

    if phase_amount != 0.0:
        # The phase half of the same correction, driven by MEASUREMENT: the
        # per-band quadrature transfer from the luma residual to the
        # color-under's complex error, estimated on this decode's own fields
        # (accumulated and pooled - one-field latency, the bundle's own
        # pattern) and applied as a rotation of the band-split residual.
        # The causal-partner form that stood here (phi = H_t{ln g}) was
        # falsified by the chroma-side transfer instrument: the measured
        # quadrature transfer is near-flat through 0.6 MHz, not a
        # Hilbert-of-amplitude shape, and the rotation it produced left the
        # measured phase error untouched while leaking into amplitude.
        # Judged by the chroma-side transfer instrument before shipping
        # default-on: on chromanoise the measured quadrature error falls
        # from +0.061/+0.058/+0.062 to +0.039/+0.010/+0.033 across
        # 0.05-0.6 MHz with the amplitude transfer unchanged in every band;
        # y-only content moves the chroma energy by 0.015% (no injection).
        # On the decode's converged tail (fields 16-27, where the pooled
        # gains have settled at 0/+0.051/+0.054/+0.031 across the four
        # runtime bands) the same instrument reads +0.007/-0.008/+0.009
        # against an uncorrected +0.036/+0.041/+0.043. The top band
        # (0.6-1.2 MHz) is under-estimated at runtime against the
        # instrument (0.04 vs 0.195) and stays open. Cost, interleaved on
        # the same box: 2.7 FPS off, 2.2 FPS on.
        # At phase_amount 0 none of this runs and the real path is untouched.
        import scipy.fft as _sps_fft

        workers = max(int(getattr(getattr(rf, "decoder", None), "numthreads", 1) or 1), 1)
        chroma64 = video["demod_burst"].astype(np.float64)
        n_samples = len(chroma64)
        # analytic signal via one transform pair (scipy's hilbert spells the
        # same thing but single-threaded and with an extra copy)
        spectrum = _sps_fft.rfft(chroma64, workers=workers)
        padded = np.zeros(n_samples, dtype=np.complex128)
        np.multiply(spectrum, 2.0, out=padded[: len(spectrum)])
        padded[0] *= 0.5
        if n_samples % 2 == 0:
            padded[len(spectrum) - 1] *= 0.5
        analytic = _sps_fft.ifft(padded, workers=workers)
        quad64 = np.imag(analytic)
        # The complex envelope against a LOCAL reference - a narrow low-pass
        # of the envelope itself. A line-locked reference is invalid here
        # twice over: the under-carrier sits off nominal (a fold averages
        # wound phasors) and the format rotates chroma phase per line by
        # design. The local reference tracks both; what remains is the fast
        # relative error, which is the correction's whole subject.
        cc_hz = float(decoder_params["color_under_carrier"])
        top_hz = float(decoder_params["chroma_bpf_upper"]) - cc_hz
        # octave bands anchored on the chroma band's own edge
        edges = [top_hz / 16.0, top_hz / 8.0, top_hz / 4.0, top_hz / 2.0, top_hz]
        # The heterodyne to the under-carrier is a circular shift of the
        # one-sided spectrum - no million-point complex exponential - and the
        # band-pass has already confined the chroma to [0, chroma_bpf_upper],
        # so after the shift everything the estimator can use lies within
        # +-max(cc, top) of zero. The analysis therefore runs on the
        # spectrally TRUNCATED grid: the same bins, a fraction of the
        # samples (measured: the full-rate spelling cost 40% of the decode
        # rate; this one is a small fraction of that). Truncation is exact
        # inside the band - nothing is approximated, only the empty
        # spectrum outside it is dropped.
        bin_hz = rf.freq_hz / n_samples
        shift_bins = int(round(cc_hz / bin_hz))
        shifted = np.roll(padded, -shift_bins)
        half_bins = int(np.ceil(max(cc_hz, top_hz) / bin_hz))
        band = np.concatenate((shifted[: half_bins + 1], shifted[n_samples - half_bins:]))
        m_dec = len(band)
        fs_dec = m_dec * bin_hz
        fz = np.fft.fftfreq(m_dec, 1.0 / fs_dec)
        z = _sps_fft.ifft(band, workers=workers)
        zref = _sps_fft.ifft(np.where(np.abs(fz) < edges[0] / 2.0, band, 0.0),
                             overwrite_x=True, workers=workers)
        ref_mag = np.abs(zref)
        alive = ref_mag > 0.3 * np.median(ref_mag)
        err = np.zeros(m_dec, dtype=complex)
        err[alive] = z[alive] / zref[alive] - 1.0
        residual = np.log(np.clip(deviation.astype(np.float64), 1e-6, 1e6))
        residual -= residual.mean()
        # real transforms: the residual is real and the rotation built from
        # it is Hermitian, so half the bins carry everything
        D = _sps_fft.rfft(residual, workers=workers)
        # the residual on the truncated grid, rescaled so a unit of
        # log-residual stays a unit (the inverse transform divides by its
        # own length)
        residual_dec = _sps_fft.irfft(D[: half_bins + 1], n=m_dec, workers=workers) * (m_dec / n_samples)
        # ESTIMATION on Hann-windowed segments, the instrument's own hygiene:
        # a field-length rectangular transform leaks the line-locked picture
        # structure across bins far beyond a two-bin exclusion, and the
        # estimate then reads picture as path (measured: the wrong sign in
        # the low bands before this existed). Segments of a power of two
        # give a clean exclusion; line harmonics are picture-locked, not
        # path, and are excluded from the ESTIMATE only - never from the
        # applied rotation's band split. The segment keeps its DURATION on
        # the truncated grid, so its bin spacing is unchanged.
        seg = int(round(16384 * m_dec / n_samples))
        hann = np.hanning(seg)
        fseg = np.fft.fftfreq(seg, 1.0 / fs_dec)
        line_rate = 1e6 / float(rf.SysParams["line_period"])
        nearest = np.round(fseg / line_rate) * line_rate
        clear = np.abs(fseg - nearest) > 2.0 * (fs_dec / seg)
        sels = [clear & (np.abs(fseg) >= lo) & (np.abs(fseg) < hi)
                for lo, hi in zip(edges[:-1], edges[1:])]
        dd_row = [0.0] * len(sels)
        ed_row = [0.0 + 0.0j] * len(sels)
        err_filled = np.where(alive, err, 0.0)
        # Half-overlapped segments, the instrument's own stride. A stride of
        # two segments was tried while this ran at full rate (four times
        # fewer windows) and it MOVED the estimate, not just its error bar:
        # the 71-143 kHz band went from +0.050+-0.007 to +0.019+-0.013 on
        # the same 27 fields and fell to the garrote, while the chroma
        # instrument's residual in its 0.05-0.15 MHz band rose from +0.010
        # to +0.038 (uncorrected +0.036). Spectral truncation, by contrast,
        # left every band's estimate unchanged to the third decimal. On the
        # truncated grid the segments are short and the full window count
        # costs nothing that shows.
        for w0 in range(0, m_dec - seg, seg // 2):
            sl = slice(w0, w0 + seg)
            occ = float(alive[sl].mean())
            if occ < 0.5:
                continue
            Dw = _sps_fft.fft((residual_dec[sl] - residual_dec[sl].mean()) * hann, workers=workers)
            Ew = _sps_fft.fft(err_filled[sl] * hann, workers=workers)
            for b, sel in enumerate(sels):
                dd_row[b] += float((np.abs(Dw[sel]) ** 2).sum())
                ed_row[b] += complex((Ew[sel] * np.conj(Dw[sel])).sum() / occ)
        state = rf.__dict__.setdefault("_chroma_phase_state", {"dd": [], "ed": []})
        state["dd"].append(dd_row)
        state["ed"].append(ed_row)
        DD = np.array(state["dd"])
        ED = np.array(state["ed"])
        # per-band quadrature gain, pooled across the decode's fields, with a
        # jackknife standard error over fields; a band not resolved at two
        # sigma is withheld rather than applied (the garrote)
        gains = np.zeros(len(edges) - 1)
        if len(DD) >= 2:
            for b in range(len(gains)):
                tot_dd = DD[:, b].sum()
                tot_ed = ED[:, b].sum()
                if tot_dd <= 0.0:
                    continue
                g = (tot_ed / tot_dd).imag
                jack = np.array([
                    ((tot_ed - ED[k, b]) / max(tot_dd - DD[k, b], 1e-30)).imag
                    for k in range(len(DD))
                ])
                se = np.sqrt((len(DD) - 1) / len(DD) * np.sum((jack - jack.mean()) ** 2))
                gains[b] = g if abs(g) > 2.0 * se else 0.0
        field.chroma_phase_gains = gains
        # the rotation that cancels the measured quadrature error: the
        # band-split residual times minus the measured gain, one transform
        fr = np.fft.rfftfreq(n_samples, 1.0 / rf.freq_hz)
        Phi = np.zeros(len(D), dtype=complex)
        for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            sel = (fr >= lo) & (fr < hi)
            Phi[sel] = -gains[b] * D[sel]
        phase = _sps_fft.irfft(Phi, n=n_samples, overwrite_x=True, workers=workers).astype(np.float32)
        # BURST-SPARING: the rotation must taper to zero across each line's
        # colour burst, or the burst-referenced time base reads the applied
        # phase as line timing and fights the correction - measured on
        # y-only content as a 100x-class luma timing echo before this
        # existed. The burst then reports true timing; the correction
        # applies where the chroma content lives. Half-cosine tapers avoid
        # phase steps.
        linelocs = getattr(field, "linelocs2", None)
        if linelocs is not None:
            fs_us = rf.freq_hz / 1e6
            burst_us = rf.SysParams["colorBurstUS"]
            # The spared span covers the burst FIT windows, not just the
            # spec burst: the level/phase fits read margins beyond it (-4
            # and +8 output px plus an adjustment - lddecode/core.py's
            # burst slice and this file's burst_start/burst_end), and a
            # taper inside their reach was measured to leave half the
            # timing fight standing.
            out_us = 8.0 / (4.0 * rf.SysParams["fsc_mhz"])
            b_lo = int(round((burst_us[0] - 2.0 * out_us) * fs_us))
            b_hi = int(round((burst_us[1] + 2.0 * out_us) * fs_us))
            taper = int(round(0.25 * fs_us))
            ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
            n_ph = len(phase)
            for loc in np.asarray(linelocs, dtype=np.float64):
                a = int(round(loc))
                lo = a + b_lo - taper
                hi = a + b_hi + taper
                if lo < 0 or hi >= n_ph:
                    continue
                phase[lo:lo + taper] *= ramp.astype(np.float32)
                phase[lo + taper:hi - taper] = 0.0
                phase[hi - taper:hi] *= ramp[::-1].astype(np.float32)
        quad = quad64.astype(video["demod_burst"].dtype)
        _scale_color_under_complex(
            video["demod_burst"], quad, gain, phase,
            amount, np.float32(phase_amount),
        )
    else:
        _scale_color_under(video["demod_burst"], gain, amount)
    field.chroma_envelope_gain_applied = True
    return True


@njit(cache=True, nogil=True, fastmath=True)
def chroma_to_u16(chroma):
    """
    Scale the chroma output array to a 16-bit value for output.
    """
    S16_ABS_MAX = 32767.0
    U16_MAX = 65535.0
    N = len(chroma)

    out = np.empty(N, dtype=np.uint16)

    for i in range(N):
        value = chroma[i] + S16_ABS_MAX
        # Saturate rather than let the cast wrap. Chroma that runs past the
        # scale is chroma at its limit, and wrapping sends it to the opposite
        # limit instead - a sample eight counts below the bottom comes out
        # eight counts from the top, which reads as the complementary hue at
        # full saturation rather than as the clipped sample it is.
        if value < 0.0:
            value = 0.0
        elif value > U16_MAX:
            value = U16_MAX
        out[i] = np.uint16(value)

    return out

@njit(cache=False, nogil=True, fastmath=True)
def chroma_automatic_gain(
    chroma,
    burst_abs_ref,
    phase_sequence,
    burst_detected_line,
    sync_tip_len,
    decode_average,
    decode_average_count,
    smoothing_window=8,
    k=2.0
):
    burst_count = len(phase_sequence)

    raw_gains = np.empty(burst_count, dtype=np.float64)
    valid_gains = np.empty(burst_count, dtype=np.float64)
    valid_amps = np.empty(burst_count, dtype=np.float64)
    valid_count = 0

    # scale the reference gain based on the average previous decodes
    # decode average is the uncorrected average amplitude, burst abs ref is the target reference.
    #   For example, with a decode average consisting of 8 fields, and this is current processing field 9, 
    #   Scale the amount of gain applied to use 8/9 the ratio of these values and 1/9 the ratio of the actual measurement
    if decode_average_count > 0:
        decode_average_numerator = decode_average_count / (decode_average_count + 1)
        decode_average_denominator = 1 / (decode_average_count + 1)
        decode_average_gain = (burst_abs_ref / decode_average) * decode_average_numerator
    else:
        decode_average_gain = 0
        decode_average_denominator = 1

    # extract gain values and track valid amplitudes
    for i in range(burst_count):
        current_burst = phase_sequence[i]
        current_amp = current_burst.amplitude if current_burst.amplitude != 0 else 1e-8

        raw_gain = burst_abs_ref / current_amp
        raw_gains[i] = raw_gain

        if current_burst.line_number >= burst_detected_line:
            valid_gains[valid_count] = raw_gain
            valid_amps[valid_count] = current_amp
            valid_count += 1

    # calculate MAD threshold for gain adjustment
    if valid_count > 0:
        active_gains = valid_gains[:valid_count]
        median_gain = np.median(active_gains)
        mad_gain = np.median(np.abs(active_gains - median_gain))
        max_allowable_gain = median_gain + (k * mad_gain)
        min_allowable_gain = median_gain - k * mad_gain
    else:
        max_allowable_gain = 1
        min_allowable_gain = 0

    # clamp gains
    clamped_gains = np.clip(raw_gains, min_allowable_gain, max_allowable_gain)

    # apply gain averaging across fields
    clamped_gains = decode_average_gain + (clamped_gains * decode_average_denominator)

    # calculate smoothing
    smoothed_gains = np.empty(burst_count, dtype=np.float64)
    half_w = smoothing_window // 2
    for i in range(burst_count):
        start = max(0, i - half_w)
        end = min(burst_count, i + half_w + 1)
        smoothed_gains[i] = np.sum(clamped_gains[start:end]) / (end - start)

    # apply gain, calculate noise floor
    noise_sum = 0
    noise_samples = 0
    for i in range(burst_count):
        current_burst = phase_sequence[i]
        current_burst_start = current_burst.start

        if i < burst_count - 1:
            next_burst_start = phase_sequence[i + 1].start
        else:
            next_burst_start = len(chroma)

        if current_burst.line_number < burst_detected_line:
            chroma[current_burst_start:next_burst_start] = 0.0
        else:
            gain_start = smoothed_gains[i]
            if i < burst_count - 1:
                gain_end = smoothed_gains[i + 1] 
            else:
                gain_end = smoothed_gains[i]

            length = next_burst_start - current_burst_start
            if length > 0:
                gain_increment = (gain_end - gain_start) / length
                gain = gain_start

                # apply gain
                for j in range(current_burst_start, next_burst_start):
                    chroma[j] = chroma[j] * gain
                    gain += gain_increment

                # get noise floor of sync tip area in the chroma channel
                sync_tip = chroma[next_burst_start + 4 - sync_tip_len : next_burst_start - 4]

                # MAD
                med = np.median(sync_tip)
                abs_dev = np.abs(sync_tip - med)
                mad = np.median(abs_dev)

                # Accumulate the noise floor
                noise_sum += mad * 1.4826
                noise_samples += 1
            
    # Calculate the average noise floor for the entire processed region
    noise_floor = noise_sum / noise_samples if noise_samples > 0 else 0.0

    # return mean of means
    if valid_count > 0:
        return np.mean(valid_amps[:valid_count]), noise_floor

    return 0.0, noise_floor


@njit(cache=True, nogil=True)
def comb_c_pal(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 2H,
    and one advanced by 2H
    line by line. VCRs do this to reduce crosstalk.
    Helps chroma stability on LP tapes in particular.
    (VCRs only adds delayed by 1h instead)
    """

    # TODO: Compensate for PAL quarter cycle offset
    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        adv2h = data2[(line_num + 2) * line_len : (line_num + 3) * line_len]
        delayed2h = data2[(line_num - 2) * line_len : (line_num - 1) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/4 and advanced 1/4.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice * 2) - (delayed2h) - adv2h
        ) / 4
    return data


@njit(cache=True, nogil=True)
def comb_c_ntsc(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 1H,
    and one advanced by 1h
    line by line. VCRs do this to reduce crosstalk.
    (VCRs only adds delayed by 1h instead)
    """

    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        advanced1h = data2[(line_num + 1) * line_len : (line_num + 2) * line_len]
        delayed1h = data2[(line_num - 1) * line_len : (line_num) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/3.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice * 2) - advanced1h - delayed1h
        ) / 4
    return data


@jitclass({
    'line_number': numba.int32,
    'start': numba.int32,
    'end': numba.int32,
    'phase_deg': numba.float64,
    'phase_offset_deg': numba.float64,
    'amplitude': numba.float64,
    'magnitude': numba.float64,
    'frequency': numba.float64,
    'dc': numba.float64,
    'I': numba.float64,
    'Q': numba.float64,
    'phase_rotation': numba.int8,
})
class BurstInfo:
    line_number: int
    start: int
    end: int
    center: int
    phase_deg: float
    amplitude: float
    magnitude: float
    frequency: float
    dc: float
    I: float
    Q: float
    phase_rotation: int

    def __init__(
        self,
        line_number,
        burst_start,
        burst_end,
        burst_center,
        burst_phase_deg,
        burst_amplitude,
        burst_magnitude,
        burst_dc,
        burst_frequency,
        I,
        Q
    ):
        self.line_number = line_number
        self.start = burst_start
        self.end = burst_end
        self.center = burst_center
        self.phase_deg = burst_phase_deg
        self.amplitude = burst_amplitude
        self.magnitude = burst_magnitude
        self.dc = burst_dc
        self.frequency = burst_frequency
        self.I = I
        self.Q = Q
        self.phase_rotation = -1 # this is set later


@njit(nogil=True, inline='always')
def _solve_4x4(A, b):
    """
    Inlined 4x4 Gaussian elimination solver with partial pivoting.
    """
    M = np.empty((4, 5))
    for r in range(4):
        M[r, 0] = A[r, 0]
        M[r, 1] = A[r, 1]
        M[r, 2] = A[r, 2]
        M[r, 3] = A[r, 3]
        M[r, 4] = b[r]
        
    for i in range(4):
        # Find pivot row
        max_row = i
        max_val = abs(M[i, i])
        for r in range(i + 1, 4):
            val = abs(M[r, i])
            if val > max_val:
                max_val = val
                max_row = r
                
        # Swap rows if necessary
        if max_row != i:
            for c in range(i, 5):
                tmp = M[i, c]
                M[i, c] = M[max_row, c]
                M[max_row, c] = tmp
                
        if abs(M[i, i]) < 1e-12:
            return np.zeros(4), False  # Singular matrix protection
            
        # Eliminate below
        for r in range(i + 1, 4):
            factor = M[r, i] / M[i, i]
            for c in range(i, 5):
                M[r, c] -= factor * M[i, c]
                
    # Back substitution
    x = np.empty(4)
    for i in range(3, -1, -1):
        sum_ax = 0.0
        for j in range(i + 1, 4):
            sum_ax += M[i, j] * x[j]
        x[i] = (M[i, 4] - sum_ax) / M[i, i]
        
    return x, True


@njit(nogil=True, fastmath=False, cache=True, inline='always')
def _tune_burst_measurements(
    burst, burst_start, amp_guess, phi_guess, dc_guess, fsc, f_weight=1e4, max_iter=32, max_precision=1e-10
):
    """
    Gauss-Newton optimization for tuning color burst measurements.

    Fits an NTSC/PAL analytical subcarrier model to digitized video samples:
        Model(t) = A * cos(2 * pi * f * t - phi) + dc

    Utilizes an iterative least-squares (Gauss-Newton) algorithm to extract 
    four critical parameters from the color burst window:
        1. Amplitude (A)       -> Subcarrier strength / chroma saturation
        2. Phase (phi)         -> Subcarrier phase / chroma hue
        3. DC Offset (dc)      -> Local luma/black level offset
        4. Frequency (f)       -> Local subcarrier frequency drift

    Includes a Bayesian frequency prior (f_weight) acting as a regularizer 
    to prevent the frequency parameter from diverging on noisy lines.
    """
    A = amp_guess
    phi = phi_guess
    dc = dc_guess
    f = fsc
    f_target = fsc

    N = len(burst)
    
    # Precompute time array
    t = np.empty(N, dtype=np.float64)
    for k in range(N):
        t[k] = (k + burst_start) / (4.0 * fsc)

    theta = np.empty(N, dtype=np.float64)
    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    r = np.empty(N, dtype=np.float64)
    
    # Pre-allocate Jacobian components
    j0 = np.empty(N, dtype=np.float64)
    j1 = np.empty(N, dtype=np.float64)
    # Note: j2 is constant 1.0, so we do not need an array for it
    j3 = np.empty(N, dtype=np.float64)

    # Pre-allocate normal equation containers
    J_JT = np.empty((4, 4), dtype=np.float64)
    J_r = np.empty(4, dtype=np.float64)

    minus_two_pi = -2.0 * np.pi

    for _ in range(max_iter):
        two_pi_f = 2.0 * np.pi * f

        theta = (two_pi_f * t) - phi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Compute residual vector
        r = burst - (A * cos_theta + dc)
        
        # Build Jacobian components
        j0 = cos_theta
        j1 = A * sin_theta
        j3 = (minus_two_pi * t) * j1

        # np.dot utilizes BLAS-like instruction sets (SSE, AVX, or AVX-512)
        J_JT[0, 0] = np.dot(j0, j0)
        J_JT[0, 1] = np.dot(j0, j1)
        J_JT[0, 2] = np.sum(j0)          # since j2 is 1.0
        J_JT[0, 3] = np.dot(j0, j3)
        
        J_JT[1, 1] = np.dot(j1, j1)
        J_JT[1, 2] = np.sum(j1)          # since j2 is 1.0
        J_JT[1, 3] = np.dot(j1, j3)
        
        J_JT[2, 2] = float(N)            # np.dot(1.0, 1.0) for N elements
        J_JT[2, 3] = np.sum(j3)          # since j2 is 1.0
        
        J_JT[3, 3] = np.dot(j3, j3)
        
        # Compute J_r vector elements using SIMD-accelerated dot products
        J_r[0] = np.dot(j0, r)
        J_r[1] = np.dot(j1, r)
        J_r[2] = np.sum(r)               # since j2 is 1.0
        J_r[3] = np.dot(j3, r)

        # Mirror the upper triangle to the lower triangle (Exploit Symmetry)
        J_JT[1, 0] = J_JT[0, 1]
        J_JT[2, 0] = J_JT[0, 2]
        J_JT[2, 1] = J_JT[1, 2]
        J_JT[3, 0] = J_JT[0, 3]
        J_JT[3, 1] = J_JT[1, 3]
        J_JT[3, 2] = J_JT[2, 3]

        # Apply Bayesian frequency centering prior/penalty
        J_JT[3, 3] += f_weight
        J_r[3] += f_weight * (f_target - f)

        # Diagonal regularization
        J_JT[0, 0] += 1e-6
        J_JT[1, 1] += 1e-6
        J_JT[2, 2] += 1e-6
        J_JT[3, 3] += 1e-6

        # Solve system
        delta, success = _solve_4x4(J_JT, J_r)
        if not success:
            break

        delta_A = delta[0]
        delta_phi = delta[1]
        delta_dc = delta[2]
        delta_f = delta[3]

        # Apply updates
        A += delta_A
        phi += delta_phi
        dc += delta_dc
        f += delta_f

        # Break early if updates converge to tiny changes
        if (abs(delta_A) < max_precision) and \
           (abs(delta_phi) < max_precision) and \
           (abs(delta_dc) < max_precision) and \
           (abs(delta_f) < max_precision):
            break

    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    return A, phi, dc, f


@njit(cache=True, nogil=True, fastmath=False)
def _demod_burst(
    burst,
    burst_start,
    burst_len,
    burst_sin,
    burst_cos,
    fsc
):
    # get initial burst measurements
    I = 0.0
    Q = 0.0
    burst_sum = 0.0

    for i in range(burst_len):
        burst_sample = burst[i]
        burst_sum += burst_sample
        carrier_idx = i + burst_start
        I += burst_sample * burst_cos[carrier_idx]
        Q += burst_sample * burst_sin[carrier_idx]


    # build starting point for refinement
    phi_guess = (math.atan2(Q, I) + math.pi) % (2.0 * math.pi) - math.pi
    dc_guess = burst_sum / burst_len
    amp_guess = (2.0 * math.sqrt(I * I + Q * Q)) / burst_len

    # refine burst measurements
    burst_amplitude, fit_phi, burst_dc, burst_frequency = _tune_burst_measurements(
        burst, burst_start, amp_guess, phi_guess, dc_guess, fsc
    )

    # Convert the absolute fitted phase shift (radians) into a fractional sample offset
    phase_sample_offset = (fit_phi % (2.0 * math.pi)) * (2.0 / math.pi)

    # Combine the geometric window midpoint with the phase shift
    burst_center_relative = (burst_len - 1) / 2.0 + phase_sample_offset
    burst_center = burst_start + burst_center_relative
    burst_phase_deg = math.degrees(fit_phi) % 360.0
    burst_magnitude = burst_amplitude * (burst_len / 2.0)

    return burst_center, burst_phase_deg, burst_amplitude, burst_magnitude, burst_dc, burst_frequency, I, Q

def _get_upconverted_burst(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    current_phase,
    burst_area,
    burst_sin,
    burst_cos,
    line_number,
    line_offset,
    outwidth,
    fsc
):
    burst_filter_padding = burst_area[0]
    line_start = (line_number - line_offset) * outwidth
    burst_start = max(0, line_start + burst_area[0] - burst_filter_padding)
    burst_end = min(len(chroma), line_start + burst_area[1] + burst_filter_padding)

    upconverted_burst = (
        chroma_heterodyne[current_phase][burst_start:burst_end]
        * chroma[burst_start:burst_end]
    )

    # filter out noise so only the color burst is present
    filtered_padded = sosfiltfilt_rust(chroma_filter, upconverted_burst)
    filtered = filtered_padded[burst_filter_padding:-burst_filter_padding]

    burst_len = len(filtered)
    burst_results = _demod_burst(
        filtered, burst_start + burst_filter_padding, burst_len, burst_sin, burst_cos, fsc
    )

    return BurstInfo(
        line_number,
        burst_start,
        burst_end,
        *burst_results
    )

def _get_phase_sequence(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    chroma_rotation,
    chroma_rotation_starting_index,
    burstarea,
    burst_sin,
    burst_cos,
    fsc,
    lineoffset,
    outwidth,
    last_line,
    detect_chroma_track_phase,
    rotation_check_start_line,
    track_change_threshold,
    color_system
):
    do_phase_rotation_check = (
        detect_chroma_track_phase
        and chroma_rotation is not None
        and chroma_heterodyne is not None
    )

    phase_sequence = []

    if chroma_rotation_starting_index is None:
        # first field
        chroma_rotation_starting_index = 0
        chroma_rotation_index = 0

    if chroma_rotation:
        # color under format that uses a phase rotated heterodyne to down convert the composite chroma
        chroma_rotation_index = chroma_rotation_starting_index
        track_rotation = chroma_rotation[chroma_rotation_index]
    else:
        # format that uses a fixed heterodyne phase, or does not rotate
        chroma_rotation_index = 0
        track_rotation = chroma_rotation_starting_index
    """
    "...a signal that represents phase zero with respect to the chroma signal phase 
    +90°, +180°, +270° etc. or a phase 0°, -90°. —180°. —270°. etc., 
    depending upon which head is on the tape at the particular time.

    The direction of phase rotation, being related to which head is on the tape at a given time,
    can be determined and preset by sensing whether the PG (pulse generator) pulse is positive-going or negative-going."
     - https://archive.org/details/rca-vcr-1-red-book-w-cover/page/n25/mode/2up?q=phase

    See also: https://archive.org/details/video-technical-guide/page/1-9/mode/2up?q=phase

    The phase rotation switch is determined at record time depending on which video head is on the tape.
    This rotation switch can occur in the middle of a line, causing a small phase artifact
    TODO: It may be possible to detect where this happens on the line and correct the phase issue mid-line
          Possibly a 2D aware detection could be used to determine where the color phase is rotated +-90 degrees relative to the lines above and below
    """

    current_phase = 0
    use_next_phase = False
    for linenumber in range(lineoffset, last_line):
        if use_next_phase:
            # reuse the calculated phase from the previous iteration
            current_phase = next_phase
            current_burst = next_burst

            use_next_phase = False
        else:
            current_phase = (current_phase + track_rotation) % 4
            current_burst = _get_upconverted_burst(
                chroma,
                chroma_heterodyne,
                chroma_filter,
                current_phase,
                burstarea,
                burst_sin,
                burst_cos,
                linenumber,
                lineoffset,
                outwidth,
                fsc
            )

        # check if the track has rotated around the head switching area
        if (
            do_phase_rotation_check
            and linenumber >= rotation_check_start_line
            and linenumber < last_line - 1
        ):
            # get the next burst using the phase rotation for the current track
            next_phase = (current_phase + track_rotation) % 4
            next_burst = _get_upconverted_burst(
                chroma,
                chroma_heterodyne,
                chroma_filter,
                next_phase,
                burstarea,
                burst_sin,
                burst_cos,
                linenumber + 1,
                lineoffset,
                outwidth,
                fsc
            )

            if color_system == "NTSC":
                # check one line back
                comparison_burst: BurstInfo = current_burst
            else: # color_system in ("PAL", "PAL_M", "NLINHA", "MESECAM")
                # check two lines back
                comparison_burst: BurstInfo = phase_sequence[-1]

            phase_delta_quadrant = abs(
                (next_burst.phase_deg - comparison_burst.phase_deg + 180) % 360 - 180
            )
            if phase_delta_quadrant > track_change_threshold:
                # burst is more in phase than out of phase, flip rotation so it remains out of phase
                chroma_rotation_index = (chroma_rotation_index + 1) % 2
                track_rotation = chroma_rotation[chroma_rotation_index]
            else:
                use_next_phase = True

        current_burst.phase_rotation = current_phase
        phase_sequence.append(current_burst)

    if chroma_rotation and chroma_rotation_index == chroma_rotation_starting_index:
        # rotate the phase for the next field, if rotation was not detected
        chroma_rotation_index = (chroma_rotation_index + 1) % 2

    return chroma_rotation_index, phase_sequence


def get_phase_rotation_sequence(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    chroma_rotation,
    chroma_rotation_index,
    lineoffset,
    linesout,
    outwidth,
    burstarea,
    burst_sin,
    burst_cos,
    fsc,
    detect_chroma_track_phase,
    rotation_check_start_line,
    enable_color_killer,
    prev_burst_detected_line,
    color_system,
):
    # Detects the correct color-under heterodyne starting phase and rotation direction
    # Additional for NTSC, this function calculates the color burst average for burst-locked TBC later on
    track_change_threshold = 90
    burst_check_skip_lines = 16


    end = linesout + lineoffset

    chroma_rotation_index, phase_sequence = _get_phase_sequence(
        chroma,
        chroma_heterodyne,
        chroma_filter,
        chroma_rotation,
        chroma_rotation_index,
        burstarea,
        burst_sin,
        burst_cos,
        fsc,
        lineoffset,
        outwidth,
        end,
        detect_chroma_track_phase,
        rotation_check_start_line,
        track_change_threshold,
        color_system
    )

    burst_check_start = burst_check_skip_lines
    burst_check_end = end - burst_check_skip_lines
    burst_detected_line = 0 # color enabled by default

    if chroma_rotation:
        # detect relative phase difference between lines
        delta_0 = 0
        delta_90 = 0
        delta_180 = 0
        delta_270 = 0

        for i in range(1, len(phase_sequence)):
            previous_burst = phase_sequence[i-1]
            current_burst = phase_sequence[i]

            if current_burst.line_number > burst_check_start and current_burst.line_number < burst_check_end:
                delta = (current_burst.phase_deg - previous_burst.phase_deg) % 360
                bucket = int((delta + 45) // 90) % 4

                if bucket == 0:
                    delta_0 += 1
                elif bucket == 1:
                    delta_90 += 1
                elif bucket == 2:
                    delta_180 += 1
                else:
                    delta_270 += 1

        if color_system == "NTSC":
            # if the bursts are out of phase with each other, the track was miss-detected, flip phase and recalculate sequence
            flip_track_phase = delta_0 < delta_180
        else:  # color_system in ("PAL", "PAL_M", "NLINHA", "MESECAM")
            # each line should alternate phase, if there are repeated sequences of phase, recalculate
            alt1 = delta_90 + delta_270
            alt2 = delta_0 + delta_180

            # choose whichever pattern dominates
            flip_track_phase = alt1 < alt2
    else:
        # no difference between track phases, do not flip
        flip_track_phase = False

    if flip_track_phase:
        # recalculate with the corrected track rotation
        chroma_rotation_index, phase_sequence = _get_phase_sequence(
            chroma,
            chroma_heterodyne,
            chroma_filter,
            chroma_rotation,
            chroma_rotation_index,
            burstarea,
            burst_sin,
            burst_cos,
            fsc,
            lineoffset,
            outwidth,
            end,
            detect_chroma_track_phase,
            rotation_check_start_line,
            track_change_threshold,
            color_system
        )

    # calculate the average color phase for even and odd lines
    even_I_total = 0
    even_Q_total = 0
    odd_I_total = 0
    odd_Q_total = 0

    avg_count = 0
    burst_magnitude_avg = 0

    for burst in phase_sequence:
        if burst.line_number > burst_check_start and burst.line_number < burst_check_end:
            I = burst.I
            Q = burst.Q

            if burst.magnitude != 0:
                I /= burst.magnitude
                Q /= burst.magnitude

                avg_count += 1
                burst_magnitude_avg += burst.magnitude

                if enable_color_killer:
                    # find the first line that might have a valid burst if the previous field had the burst disabled
                    # broadcasters would sometime turn on the burst mid-field, so attempt to detect that transition here
                    if (
                        prev_burst_detected_line == -1 # previous field had color killer activated
                        and burst_detected_line == 0 and burst.magnitude > BURST_MAGNITUDE_THRESHOLD # first burst that exceeds threshold
                    ):
                        # first burst that exceeds threshold
                        # color killer will be active until this line, then it deactivates
                        # it is only reactivated after an entire field is without color (below)
                        burst_detected_line = burst.line_number
            
                if burst.line_number % 2:
                    odd_I_total += I
                    odd_Q_total += Q
                else:
                    even_I_total += I
                    even_Q_total += Q
    
    burst_magnitude_avg /= avg_count

    if enable_color_killer:
        if burst_magnitude_avg < BURST_MAGNITUDE_THRESHOLD:
            # (re)activate color killer for the entire field
            burst_detected_line = -1

    burst_phase_avg = np.degrees(np.arctan2(even_Q_total + odd_Q_total, even_I_total + odd_I_total)) % 360
    even_burst_phase_avg = np.degrees(np.arctan2(even_Q_total, even_I_total)) % 360
    odd_burst_phase_avg = np.degrees(np.arctan2(odd_Q_total, odd_I_total)) % 360

    return chroma_rotation_index, phase_sequence, burst_detected_line, burst_magnitude_avg, burst_phase_avg, even_burst_phase_avg, odd_burst_phase_avg


@njit(cache=False, nogil=True, fastmath=False)
def upconvert_chroma(
    chroma,
    uphet,
    lineoffset,
    outwidth,
    phase_rotation_sequence,
    chroma_heterodyne,
):
    for burst in phase_rotation_sequence:
        linestart = (burst.line_number - lineoffset) * outwidth
        lineend = linestart + outwidth

        heterodyne = chroma_heterodyne[burst.phase_rotation][linestart:lineend]
        c = chroma[linestart:lineend]
        uphet[linestart:lineend] = c * heterodyne


@njit(nogil=True, cache=False, fastmath=False)
def upconvert_chroma_phase_comp(
    chroma,
    lineoffset,
    outwidth,
    phase_rotation_sequence,
    color_under_carrier_fs,
    fsc,
    target_phase_even,
    target_phase_odd
):
    deg2rad_scale = np.pi / 180.0
    pi_over_two = np.pi / 2.0

    # Initial nominal reference coefficient
    het_hz_nominal = color_under_carrier_fs
    het_coefficient = pi_over_two * (1.0 + het_hz_nominal / fsc)

    target_phase_even_rad = target_phase_even * deg2rad_scale
    target_phase_odd_rad = target_phase_odd * deg2rad_scale
    num_bursts = len(phase_rotation_sequence)

    coeff_step_factor = pi_over_two / (fsc * outwidth)

    # Pre-generate a local pixel coordinate array to help Numba vectorize
    # Computing on a local range [0, outwidth) helps the compiler reason about alignment
    local_idx = np.arange(outwidth, dtype=np.float64)

    for idx in range(num_bursts):
        current_burst = phase_rotation_sequence[idx]

        # Solve for current line's active het_hz
        k_current = current_burst.frequency / fsc
        het_hz_current = k_current * color_under_carrier_fs

        # Solve for next line's active het_hz
        if idx < num_bursts - 1:
            next_burst = phase_rotation_sequence[idx + 1]
            k_next = next_burst.frequency / fsc
            het_hz_next = k_next * color_under_carrier_fs
        else:
            het_hz_next = het_hz_current

        linestart = (current_burst.line_number - lineoffset) * outwidth
        lineend = linestart + outwidth

        # Determine target phase
        if current_burst.line_number % 2 != 0:
            target_phase_rad = target_phase_odd_rad
        else:
            target_phase_rad = target_phase_even_rad

        # Initial starting phase
        theta_0 = het_coefficient * linestart + (
            current_burst.phase_rotation * pi_over_two
            + target_phase_rad 
            + current_burst.phase_deg * deg2rad_scale
        )

        # Coefficient step parameters
        alpha = pi_over_two * (1.0 + het_hz_current / fsc)
        delta_coeff = (het_hz_next - het_hz_current) * coeff_step_factor
        beta = 0.5 * delta_coeff
        dc_val = current_burst.dc

        # Slice target and source arrays to provide direct contiguous memory views
        chroma_slice = chroma[linestart:lineend]

        # No outer dependencies so LLVM can vectorize
        for k in range(outwidth):
            # Compute closed-form phase directly (no dependency on previous k)
            theta_k = theta_0 + alpha * local_idx[k] + beta * (local_idx[k] * local_idx[k])

            # Vectorized trigonometric and arithmetic execution
            chroma_slice[k] = chroma_slice[k] * -np.cos(theta_k) - dc_val

@njit(cache=True, nogil=True)
def burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea):
    for line in range(lineoffset, linesout + lineoffset):
        linestart = (line - lineoffset) * outwidth
        lineend = linestart + outwidth

        chroma[linestart + burstarea[1] + 4 : lineend] *= 2

    return chroma


@njit(cache=True, nogil=True, fastmath=False)
def shift_chroma_and_remove_dc(out_chroma, move):
    n = len(out_chroma)
    move %= n
    
    mean_acc = 0

    # save wrapped values
    tmp = np.empty(move, dtype=out_chroma.dtype)

    for i in range(move):
        tmp[i] = out_chroma[n - move + i]

    # single pass shift
    for i in range(n - move - 1, -1, -1):
        mean_acc += out_chroma[i]
        out_chroma[i + move] = out_chroma[i]

    # small wrap-around copy
    for i in range(move):
        mean_acc += tmp[i]
        out_chroma[i] = tmp[i]

    mean_acc /= n

    # crude DC offset removal
    for i in range(n):
        out_chroma[i] -= mean_acc


def chroma_color_under_filter(
    data, filter, blocklen, notch, do_notch=None, move=10, audio_notch=None
):
    out_chroma = sosfiltfilt_rust(filter, data[:blocklen])

    if audio_notch is not None:
        out_chroma = sps.filtfilt(
            audio_notch[0],
            audio_notch[1],
            out_chroma,
        )

    if do_notch is not None and do_notch:
        out_chroma = sps.filtfilt(
            notch[0],
            notch[1],
            out_chroma,
        )

    # Move chroma to compensate for Y filter delay.
    # value needs tweaking, ideally it should be calculated if possible.
    # TODO: Not sure if we need this after hilbert filter change, needs check.
    shift_chroma_and_remove_dc(out_chroma, move)

    return out_chroma


def decode_chroma_phase_rotation(
    field,
    disable_tracking_cafc=False,
    chroma_rotation=None,
    detect_chroma_track_phase=False,
):
    chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst")

    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    burstarea = get_burst_area(field)
    rotation_check_start_line = lineoffset + linesout - 16

    # Rotation per track
    # VHS PAL:      Track1 0,   Track2 -90
    # VHS NTSC:     Track1 +90, Track2 -90
    # Betamax PAL:  None - uses frequency offset instead
    # Betamax NTSC: Track1 180, Track2 0
    # Video8 PAL:   Track1 0,   Track2 -90
    # Video8 NTSC:  Track1 0,   Track2 180

    chroma_heterodyne = (
        field.rf.chroma_afc.getChromaHet()
        if (field.rf.do_cafc and not disable_tracking_cafc)
        else field.rf.chroma_heterodyne
    )

    prev_burst_detected_line = 0
    if field.prevfield is not None:
        prev_burst_detected_line = field.prevfield.burst_detected_line

    track_phase, phase_sequence, burst_detected_line, burst_magnitude_avg, burst_phase_avg, even_burst_phase_avg, odd_burst_phase_avg = get_phase_rotation_sequence(
        chroma,
        chroma_heterodyne,
        field.rf.Filters["FChromaFinal"],
        chroma_rotation,
        field.rf.track_phase, # index for chroma rotation, and static if there is no chroma rotation
        lineoffset,
        linesout,
        outwidth,
        burstarea,
        field.rf.fsc_wave,
        field.rf.fsc_cos_wave,
        field.rf.SysParams['fsc_mhz'] * 1e6,
        detect_chroma_track_phase,
        rotation_check_start_line, # check for track phase rotation around the headswitching area (bottom of field)
        field.rf.options.enable_color_killer,
        prev_burst_detected_line,
        field.rf.color_system,
    )

    return track_phase, phase_sequence, burst_detected_line, burst_magnitude_avg, burst_phase_avg, even_burst_phase_avg, odd_burst_phase_avg


def measure_secam_under_carrier_offset(
    chroma,
    linesout,
    outwidth,
    window,
    samp_rate,
    pair_center,
    separation_range=(90e3, 230e3),
):
    """Measure how far a SECAM colour-under rest carrier pair sits from
    its nominal position, using the undeviated subcarrier on the late back
    porch of each line (the early porch is still sweeping from the previous
    line's carrier switch). For ME-SECAM this picks up the recording VCR's
    down-conversion crystal error, which otherwise ends up as an offset of
    both restored subcarriers. (SECAM method 1 has no conversion crystal, so
    for it this is only useful as a diagnostic and to recognise which of the
    two recording methods a tape actually used.)

    separation_range bounds the accepted distance between the two carrier
    clusters: the pair is nominally 156.25 kHz apart for ME-SECAM and
    39.0625 kHz apart for method 1 (a quarter of foR - foB).

    Returns the offset in Hz of the measured pair midpoint from pair_center,
    or None if no reliable measurement could be made. Single-field accuracy
    is on the order of +-100 Hz (ringing from the per-line carrier switch
    beats across the short porch window); averaging across fields washes
    this out. Crystal errors being chased are in the kHz range.
    """
    # Stay clear of the vertical interval and head switch area.
    SKIP_LINES = 20
    MIN_LINES_PER_CLUSTER = 8
    # Reject measurements where the two clusters land somewhere else
    # entirely.
    MIN_SEPARATION, MAX_SEPARATION = separation_range

    window_start, window_end = window
    freq_scale = samp_rate / (2 * np.pi)

    # Analytic signal over the whole field so the short per-line windows are
    # free of transform edge effects (a windowed transform of just the porch
    # would bias the frequency estimate by hundreds of Hz).
    n_fft = sps_fft.next_fast_len(len(chroma))
    analytic = sps.hilbert(chroma, N=n_fft)[: len(chroma)]

    freqs = []
    envs = []

    for linenumber in range(SKIP_LINES, linesout - SKIP_LINES):
        line_start = linenumber * outwidth
        start = line_start + window_start
        end = line_start + window_end
        if start < 0 or end > len(chroma):
            continue

        window_analytic = analytic[start:end]
        # Instantaneous frequency; median rejects FM clicks and noise spikes.
        f_inst = np.diff(np.unwrap(np.angle(window_analytic))) * freq_scale
        freqs.append(np.median(f_inst))
        envs.append(np.median(np.abs(window_analytic)))

    if not freqs:
        return None

    freqs = np.asarray(freqs)
    envs = np.asarray(envs)

    # Ignore lines where the porch carrier is too weak to measure
    # (dropouts, colour killed lines).
    valid = envs > (np.median(envs) * 0.25)
    low = freqs[valid & (freqs < pair_center)]
    high = freqs[valid & (freqs >= pair_center)]

    if len(low) < MIN_LINES_PER_CLUSTER or len(high) < MIN_LINES_PER_CLUSTER:
        return None

    low_carrier = np.median(low)
    high_carrier = np.median(high)
    separation = high_carrier - low_carrier
    if separation < MIN_SEPARATION or separation > MAX_SEPARATION:
        return None

    return ((low_carrier + high_carrier) / 2) - pair_center


# SECAM subcarrier rest frequencies and HF ("cloche"/bell) pre-emphasis
# constants from ITU-R BT.470-6 table 2 / BT.1700: the subcarrier amplitude
# follows G = M0 * |1 + j16F| / |1 + j1.26F| with F = f/f0 - f0/f.
SECAM_FOR = 4406250.0
SECAM_FOB = 4250000.0
SECAM_BELL_F0 = 4286000.0
# Colour-under rest carrier pair midpoints, used to tell the two VHS SECAM
# recording methods apart from the porch carriers.
SECAM_M1_UNDER_PAIR_CENTER = (SECAM_FOB / 4 + SECAM_FOR / 4) / 2  # 1082031.25
MESECAM_UNDER_PAIR_CENTER = 5060571.875 - (SECAM_FOR + SECAM_FOB) / 2  # 732446.875
# Nominal pair separation is 39.0625 kHz; the gates are loose because this is
# measured over active video (see _secam_method_diagnostic) where content
# deviation biases the per-line medians.
SECAM_M1_SEPARATION_RANGE = (18e3, 60e3)
MESECAM_SEPARATION_RANGE = (90e3, 230e3)  # nominal pair separation 156.25 kHz


def secam_bell_gain(freq_hz):
    """Relative SECAM subcarrier HF pre-emphasis (bell) gain at the given
    instantaneous frequency, normalized to 1.0 at f0 (BT.470-6)."""
    f = freq_hz / SECAM_BELL_F0
    bell_f = f - 1.0 / f
    return np.sqrt((1.0 + (16.0 * bell_f) ** 2) / (1.0 + (1.26 * bell_f) ** 2))


def upconvert_secam_method1(
    chroma, samp_rate, under_bpf, carrier_mult, rest_amplitude, return_envelope=False
):
    """Restore the studio SECAM chroma block from a method 1 colour-under
    signal (IEC 60774-1 6.4.1: recorded through a divide-by-4 counter) by
    multiplying the carrier phase back up.

    Unlike the heterodyne formats this scales carrier and deviation together,
    so tape timebase error self-corrects and there is no LO to servo. The
    divider outputs a constant-amplitude signal, so the BT.470 bell
    pre-emphasis is regenerated here from the restored instantaneous
    frequency to put the amplitude envelope back on spec for downstream
    SECAM decoders.

    Returns (restored, inst_freq): the restored chroma block signal and the
    smoothed restored-domain instantaneous frequency it was shaped with
    (the latter is reused for line identification). With return_envelope the
    band-passed under-carrier envelope is returned as a third element (used
    by regenerate_secam_blanking for local amplitude matching).
    """
    filtered = sosfiltfilt_rust(under_bpf, chroma)

    # Analytic signal over the whole field so short-window edge effects don't
    # bias the phase.
    n_fft = sps_fft.next_fast_len(len(filtered))
    analytic = sps.hilbert(filtered, N=n_fft)[: len(filtered)]
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    # Clamp under-carrier deviation to SECAM's legal window before x4
    # multiplication.
    # BT.470-6 Table 2 item 2.12 and BT.1700 Part C Table 4 item 10e limit the
    # carrier to 3.900-4.756 MHz: the per-component deviation maxima
    # (D′B -350/+506 kHz, D′R -506/+350 kHz) mirror across the carrier pair, so
    # both components share the corridor (fOB/4 - 87.5 kHz to fOR/4 + 87.5 kHz
    # in the under-carrier domain). The studio or broadcast limiter ideally
    # clipped the pre-corrected color difference to these bounds, so anything
    # outside the window is noise or a tape channel-truncation transient; x4
    # multiply would scale such excursions x4 and downstream de-emphasis smears
    # them into streaks at saturated transitions ("fire"). Clipping
    # instantaneous frequency and reintegrating keeps the carrier segment
    # smooth: clipped spans become constant-frequency stretches.
    f_under = np.gradient(phase) * (samp_rate / (2.0 * np.pi))
    np.clip(
        f_under,
        SECAM_FOB / 4 - 350e3 / 4,
        SECAM_FOR / 4 + 350e3 / 4,
        out=f_under,
    )
    phase = np.cumsum(f_under) * (2.0 * np.pi / samp_rate)

    # Restored instantaneous frequency for the bell shaping. Central
    # difference plus a short moving average keeps sample-level phase noise
    # from ending up as amplitude noise; the bell curve itself is smooth so
    # this doesn't blunt legitimate deviation.
    inst_freq = np.gradient(phase) * (carrier_mult * samp_rate / (2 * np.pi))
    smooth_len = 9
    inst_freq = np.convolve(
        inst_freq, np.full(smooth_len, 1.0 / smooth_len), mode="same"
    )
    # Keep the gain lookup inside the legal carrier excursion (BT.470:
    # 3.900 to 4.756 MHz) so noise and carrier switch transients don't get
    # boosted by the bell skirts.
    np.clip(inst_freq, 3.9e6, 4.756e6, out=inst_freq)
    gain = secam_bell_gain(inst_freq)

    # Scale by the normalized under-carrier envelope (capped just above
    # nominal). Where the carrier is healthy this is ~unity, so the average
    # amplitude stays on the bell curve; where it dips or disappears
    # (dropouts, FM clicks, no colour) the dip is passed through to the
    # output instead of being hard-limited away. Downstream SECAM decoders
    # key their click/dropout concealment off exactly those envelope
    # collapses, so preserving them matters more than emulating the
    # constant-amplitude divider chain of a real deck - and it doubles as
    # the squelch that keeps carrier-free noise from becoming full-scale
    # splatter.
    env_med = np.median(envelope)
    if env_med > 0:
        limited = np.minimum(envelope / env_med, 1.25)
    else:
        limited = np.zeros_like(envelope)

    restored = rest_amplitude * gain * limited * np.cos(carrier_mult * phase)
    if return_envelope:
        return restored, inst_freq, envelope
    return restored, inst_freq


SECAM_IDENT_MIN_CONFIDENCE = 0.7


def fit_secam_line_alternation(inst_freq, linesout, outwidth, first_line, porch_end_px):
    """Fit the field's D'R/D'B line alternation from the active-region median
    restored frequency of each line: D'R lines sit in the top half of the
    chroma block, D'B in the bottom.

    The sequence alternates strictly (BT.470), so fit the better of the two
    possible parities; per-line deviation medians can land on the wrong side
    on heavily saturated lines, the majority never does.

    Returns (dr_on_even, confidence) where confidence is the fraction of
    lines whose measured identity matches the fitted alternation, or None if
    there are too few lines to fit.
    """
    n_lines = linesout - first_line
    if n_lines < 32:
        return None

    active_start = porch_end_px + 30
    active_end = outwidth - 40
    freq_lines = inst_freq[first_line * outwidth : linesout * outwidth].reshape(
        n_lines, outwidth
    )
    line_medians = np.median(freq_lines[:, active_start:active_end], axis=1)
    is_dr = line_medians > (SECAM_FOR + SECAM_FOB) / 2

    line_index = np.arange(first_line, linesout)
    even_is_dr = np.count_nonzero(is_dr == (line_index % 2 == 0))
    confidence = max(even_is_dr, n_lines - even_is_dr) / n_lines
    return (even_is_dr >= (n_lines - even_is_dr)), confidence


class SecamParityFlywheel:
    """Carry the fitted D'R/D'B alternation across fields.

    Each TBC field is 312.5 line periods, so the alternation phase of
    consecutive fields walks a strict 4-field cycle:

        dr_on_even(n) = base ^ (((n + 1) >> 1) & 1)

    (verified on all method 1 fixture tapes: TFFT/FTTF sequences). A single
    bit therefore locks the parity of every field in the recording. Fields
    whose own alternation fit is confident teach `base`; fields whose content
    can't be fitted (near-neutral pictures, noisy tape) inherit the predicted
    parity instead of losing their blanking regeneration.

    The lock requires MIN_LOCK agreeing confident fields, expires after
    MAX_AGE fields without confirmation, and a confident contradiction resets
    it - a dropped field upstream shifts the cycle phase, and re-learning is
    cheaper than trusting a stale lock.
    """

    MIN_LOCK = 4
    MAX_AGE = 32

    def __init__(self):
        self._index = -1
        self._last_readloc = None
        self._base = None
        self._agree = 0
        self._last_confirm = None

    @staticmethod
    def _flip(index):
        return ((index + 1) >> 1) & 1

    def resolve(self, readloc, fit):
        """Advance to the field identified by readloc and resolve its parity.

        fit is (dr_on_even, confidence) or None. Returns (dr_on_even, source)
        with source "measured" or "flywheel", or (None, "unlocked") when
        neither the fit nor the lock can identify the field.
        """
        if readloc != self._last_readloc:
            self._last_readloc = readloc
            self._index += 1
        n = self._index
        flip = self._flip(n)

        if fit is not None and fit[1] >= SECAM_IDENT_MIN_CONFIDENCE:
            base = bool(fit[0]) ^ bool(flip)
            if base == self._base:
                self._agree += 1
            else:
                self._base = base
                self._agree = 1
            self._last_confirm = n
            return bool(fit[0]), "measured"

        if (
            self._base is not None
            and self._agree >= self.MIN_LOCK
            and self._last_confirm is not None
            and n - self._last_confirm <= self.MAX_AGE
        ):
            return bool(self._base ^ bool(flip)), "flywheel"
        return None, "unlocked"


def _measure_under_carrier(chroma, samp_rate, start, length, f_rot):
    """Narrowband frequency/phase estimate of the colour-under carrier over
    chroma[start:start+length] by correlation against a rotor at f_rot.

    Correlation projects out everything away from f_rot, so this stays usable
    on the raw (pre-band-pass) chroma channel where luma crosstalk would bias
    a broadband analytic-signal measurement. Two half-window correlations
    give the frequency offset from the rotor; the pooled correlation gives
    the phase. Returns (freq, phase_at) with phase_at(t) evaluating the
    carrier phase at absolute sample t, or None if there is no carrier.
    """
    x = chroma[start : start + length]
    if len(x) < length:
        return None
    t_abs = np.arange(start, start + length)
    xz = x * np.exp(-2j * np.pi * (f_rot / samp_rate) * t_abs)
    z1 = np.sum(xz[: length // 2])
    z2 = np.sum(xz[length // 2 :])
    zf = z1 + z2
    if np.abs(z1) == 0 or np.abs(z2) == 0 or np.abs(zf) == 0:
        return None
    dphi = np.angle(z2 * np.conj(z1))
    df = dphi / (2 * np.pi * (length / 2) / samp_rate)
    # Keep runaway estimates (no real carrier in the window) inside the
    # format's legal deviation.
    df = np.clip(df, -130e3, 130e3)
    freq = f_rot + df
    t_mid = start + (length - 1) / 2.0
    phase_mid = np.angle(zf)

    def phase_at(t):
        return (
            2 * np.pi * (f_rot / samp_rate) * t
            + phase_mid
            + 2 * np.pi * (df / samp_rate) * (t - t_mid)
        )

    return freq, phase_at


def regenerate_secam_blanking(
    chroma,
    envelope,
    samp_rate,
    linesout,
    outwidth,
    blank_start_px,
    porch_end_px,
    first_line,
    dr_on_even,
    carrier_mult,
):
    """Replace each line's horizontal blanking interval - front porch, sync
    and back porch in one continuous run - with a synthesized undeviated
    colour-under rest carrier, phase-continuous with the active video on both
    sides.

    On method 1 tapes the whole blanking interval carries the record chain's
    divide-by-4 counter settling transient (blanking edges / SECAM subcarrier
    phase reversals upset the divider), not the undeviated reference BT.470
    promises. Two things go wrong if it is left in place:

    - the zero-phase filters in this chain (the under-carrier band-pass here,
      FChromaFinal later) and the linear-phase cloche filters in downstream
      SECAM decoders smear the end-of-line transient BACKWARDS into the last
      ~2 us of active video, which demodulates as a magenta band down the
      right edge of the picture (D'R deviates negative, D'B positive, so the
      transient reads red on D'R lines and blue on D'B lines);
    - decoders calibrate their discriminator zeros and line identification
      from the back porch, and transient energy ringing into that window
      biases the zeros, which shows up as a full-field colour cast.

    This runs in the colour-under domain BEFORE the band-pass/analytic-signal
    restoration pass, so the zero-phase filtering never sees the transient.
    One continuous synthesis per blanking interval, phase-aligned to the
    outgoing active carrier at its start and the incoming one at its end,
    with no interior splices: an earlier version that spliced the back porch
    separately left an unaligned interior seam whose click rang into the
    decoders' porch measurement window.

    The synthesized frequency ramps from the measured outgoing carrier to the
    outgoing line's rest frequency across the front porch, steps to the
    incoming line's rest over the sync tip, and holds it through the back
    porch and the fade-out; the phase is the integral of that profile, so it
    is continuous throughout. The random phase difference
    between the two lines' carriers is closed by a frequency bump over the
    sync region plus a small constant offset across the hold (see the
    closure comment below). All disturbances are anchored to the line
    structure so that everything from ~90 px into the next line onwards - in
    particular the back-porch window decoders calibrate their discriminator
    zeros from (~65..5 px before active video) - stays within a few kHz of
    the incoming rest frequency, with margin for the ~25 px ring of the
    zero-phase band-passes.

    Returns a float64 copy of chroma with the blanking intervals replaced.
    """
    FADE_LEN = 8
    RAMP_LEN = 20
    MEAS_LEN = 32
    # Rest-to-rest frequency step position within the NEXT line (px from its
    # start): over the sync tip.
    STEP_PX = (8, 40)
    # Phase closure: the outgoing and incoming carriers are independent
    # oscillator segments, so the synthesis must absorb a uniformly random
    # phase difference of up to +-pi (x4 by the restoration - a step is not
    # an option anywhere near the picture or the porch). It goes into a
    # cosine-tapered flat-top frequency excursion across the sync region,
    # sized to the error but capped inside the under band-pass (BUMP_MAX_HZ
    # around either rest carrier stays within the 550..1300 kHz pass band),
    # and finished early enough that the band-pass ring stays out of the
    # porch reference window. Any spill past the cap (degenerately short
    # blanking only) becomes a constant offset across the rest-frequency
    # hold - never more than a few kHz, too small to bias the per-field
    # porch cluster medians or flip a line identity label.
    BUMP_END_PX = 88
    BUMP_MAX_HZ = 170e3
    BUMP_TAPER = 12

    f_rest = {
        True: SECAM_FOR / carrier_mult,
        False: SECAM_FOB / carrier_mult,
    }

    cleaned = np.array(chroma, dtype=np.float64, copy=True)
    fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(FADE_LEN) / FADE_LEN)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(RAMP_LEN) / RAMP_LEN)
    mid_step = 0.5 - 0.5 * np.cos(np.pi * np.arange(32) / 32)

    for linenumber in range(first_line, linesout - 1):
        line_is_dr = (linenumber % 2 == 0) == dr_on_even
        f_out_rest = f_rest[line_is_dr]
        f_in_rest = f_rest[not line_is_dr]
        start = linenumber * outwidth + blank_start_px
        end = (linenumber + 1) * outwidth + porch_end_px
        span = end - start
        if start - 2 * MEAS_LEN - FADE_LEN < 0 or end + 2 * MEAS_LEN > len(cleaned):
            continue

        out_meas = _measure_under_carrier(
            cleaned, samp_rate, start - MEAS_LEN, MEAS_LEN, f_out_rest
        )
        in_meas = _measure_under_carrier(
            cleaned, samp_rate, end, MEAS_LEN, f_in_rest
        )
        if out_meas is None or in_meas is None:
            continue
        f_out, out_phase_at = out_meas
        f_in, in_phase_at = in_meas

        # Local amplitudes from the band-passed envelope: narrowband
        # correlation under-reads a deviating FM carrier, and an amplitude
        # step at the splice would read as a click downstream. Measured a
        # little away from the splice points, where the pass-1 envelope is
        # still inflated by the band-pass smear of the adjacent transient.
        amp_out = np.median(envelope[start - 2 * MEAS_LEN : start - MEAS_LEN])
        amp_in = np.median(envelope[end + MEAS_LEN : end + 2 * MEAS_LEN])

        # Frequency profile: measured outgoing -> outgoing rest (over the
        # front porch) -> incoming rest (step over the sync tip) -> measured
        # incoming (final ramp), all raised-cosine.
        next_line_p = span - porch_end_px  # px offset of the next line start
        step0 = min(max(next_line_p + STEP_PX[0], RAMP_LEN), span - RAMP_LEN - 96)
        step1 = step0 + (STEP_PX[1] - STEP_PX[0])
        # The write extends FADE_LEN beyond `start` on the outside, so the
        # fade-in sits OVER the phase-matched measured outgoing carrier
        # (before the transient sets in) instead of over raw transient next
        # to the picture. The incoming side gets NO fade at all: the synth
        # ends phase-closed against the incoming carrier model at `end`, and
        # the raw signal (blanking edge, colour turn-on, picture) takes over
        # with its natural continuity. Fading out over the raw porch tail
        # mixes settling transient back in next to the decoders' porch
        # measurement window; fading out past `end` (over the incoming
        # picture) leaves a synthetic-to-content seam whose phase and
        # amplitude mismatch rings green/red fire down the left edge. Both
        # were measurably worse than the phase-closed hard handover.
        q = FADE_LEN  # profile offset of `start`
        span_ext = span + FADE_LEN
        f_prof = np.empty(span_ext)
        f_prof[:q] = f_out
        f_prof[q : q + RAMP_LEN] = f_out + (f_out_rest - f_out) * ramp
        f_prof[q + RAMP_LEN : q + step0] = f_out_rest
        f_prof[q + step0 : q + step1] = (
            f_out_rest + (f_in_rest - f_out_rest) * mid_step
        )
        # Rest frequency holds right through the back porch AND the fade-out:
        # the porch is the decoders' discriminator-zero reference, and the
        # undeviated carrier is also zero colour difference, so the fade-out
        # region (over the incoming line's low-amplitude colour turn-on
        # strip) decodes as neutral instead of as a per-line click - ramping
        # toward the measured content frequency there put green/red fire down
        # the left edge of the picture.
        f_prof[q + step1 :] = f_in_rest

        phase = out_phase_at(start - q) + (
            2 * np.pi * np.concatenate(([0.0], np.cumsum(f_prof[:-1]))) / samp_rate
        )
        phase_at_end = phase[-1] + 2 * np.pi * f_prof[-1] / samp_rate
        err = np.angle(np.exp(1j * (in_phase_at(end) - phase_at_end)))

        # Flat-top frequency excursion over the sync region: area = absorbed
        # phase. Starts no earlier than just before the next line (its
        # band-pass ring must stay out of the outgoing picture) and ends
        # early enough that the ring stays out of the porch reference window.
        b0 = max(RAMP_LEN + 4, next_line_p - 16)
        b1 = min(next_line_p + BUMP_END_PX, span - RAMP_LEN - 4)
        hold_len = (span - RAMP_LEN) - b1
        if b1 - b0 < 2 * BUMP_TAPER + 8 or hold_len < 24:
            continue
        bump_area = b1 - b0 - BUMP_TAPER  # in units of amplitude * samples
        bump_capacity = 2 * np.pi * BUMP_MAX_HZ * bump_area / samp_rate
        err_bump = np.clip(err, -bump_capacity, bump_capacity)
        bump_amp = err_bump * samp_rate / (2 * np.pi * bump_area)
        bump = np.full(b1 - b0, bump_amp)
        taper = 0.5 - 0.5 * np.cos(np.pi * np.arange(BUMP_TAPER) / BUMP_TAPER)
        bump[:BUMP_TAPER] *= taper
        bump[-BUMP_TAPER:] *= taper[::-1]
        f_prof[q + b0 : q + b1] += bump
        # Constant-offset spill over the rest-frequency hold (usually zero).
        f_prof[q + b1 : q + span - RAMP_LEN] += (
            (err - err_bump) * samp_rate / (2 * np.pi * hold_len)
        )
        phase = out_phase_at(start - q) + (
            2 * np.pi * np.concatenate(([0.0], np.cumsum(f_prof[:-1]))) / samp_rate
        )

        synth = np.linspace(amp_out, amp_in, span_ext) * np.cos(phase)
        blend = np.ones(span_ext)
        blend[:FADE_LEN] = fade
        cleaned[start - q : end] = cleaned[start - q : end] * (1.0 - blend) + synth * blend

    return cleaned


ntsc_color_framing_phase_shift = 33
# The BASE phase per colour frame, before the constant shift above is applied.
#
# THE BASE IS KEPT SEPARATE BECAUSE A BUG DEPENDED ON IT BEING FOLDED IN. The
# map used to store `0 - 33` and `180 - 33` directly, and the call site read
# the group-delay compensation's direction as `1 if target_phase else -1`.
# With the shift folded in, both stored values are -33 and 147 - BOTH NON-ZERO
# - so the test was always true and the -1 branch was unreachable. The
# compensation therefore never reversed with the colour frame, which is a
# phase error that alternates with framing.
#
# The code was evidently written when the shift was zero, where `0` is falsy
# and the test worked. Keeping the base separate makes the direction
# recoverable whatever the shift is set to, and makes the shift a pure
# constant offset rather than something the control flow depends on.
ntsc_color_framing_map = {
    # Color Frame I
    (1, 0): (1, 0),
    (0, 1): (2, 180),
    # Color Frame II
    (1, 1): (3, 180),
    (0, 0): (4, 0),
}


def colour_frame_parity(field):
    """The colour-frame parity. THE COUNTER, and the measurement it replaced
    was physically impossible.

    Ethan asked for this to be derived from the colour carrier's relationship
    to hsync rather than counted, and an attempt was made and shipped that
    read `burst_phase_avg` and anchored the counter to it. IT WAS WRONG, and
    the proof is arithmetic rather than a judgement.

    THE COLOUR-UNDER CANNOT CARRY THE COLOUR FRAME. In units of the line rate
    the subcarrier is 455/2 and the record heterodyne oscillator sits at
    `fsc + f_cu = 227.5 + 40 = 267.5`. Over one field of 262.5 lines,

        subcarrier   262.5 x 227.5 = 59718.75 cycles, fractional 3/4 -> 270 deg
        record LO    262.5 x 267.5 = 70218.75 cycles, fractional 3/4 -> 270 deg
        difference                              10500 cycles, fractional 0

    The two fractional parts are IDENTICAL, so the 270 degrees per field that
    carries the four-field sequence cancels exactly in the down-conversion
    and the colour-under advances a whole number of cycles every field. That
    is not an accident: 40 f_H was chosen so that it would.

    So no measurement of the recorded burst against the sync datum can
    recover the colour frame, and the quantity the shipped attempt actually
    read was the DECODER'S OWN rotation index - `current_phase`, reset each
    field and advanced per line - fed back as though it were a property of
    the tape.

    WHAT IT COST, MEASURED. A/B against this counter on identical code, 11 of
    20 fields came out exactly 180 degrees apart, the NTSC run-of-two burst
    structure was destroyed, and `fieldPhaseID` went from 1,2,3,4 ascending to
    an irregular 1,4,3,2,3,2,1,4. The threshold sat at exactly 90 degrees with
    10 of 19 fields landing within 2 degrees of it, so half the decode was
    decided inside its own noise. The withdrawn docstring claimed the counter
    produced a DESCENDING sequence; on this material the counter ascends and
    the measured version is what descends. It did the opposite of what it
    claimed.

    WHAT WOULD ACTUALLY WORK is named rather than guessed: the framing has to
    be read from something the record heterodyne does not cancel. The
    surviving candidates are the luma side's own sync-to-subcarrier
    relationship before the chroma is split off, or an external reference.
    `vhsdecode/models/colour_framing.py` holds the specification arithmetic -
    270 degrees per field, four fields - and is correct; what it lacks, and
    now says it lacks, is a measurable input.
    """
    return (field.field_number // 2) % 2, False


def _secam_method_diagnostic(field, chroma, linesout, outwidth):
    """For the first fields of a SECAM method 1 decode, check that the
    colour-under energy actually sits where method 1 puts it, and warn if the
    tape looks like it was recorded with the ME-SECAM heterodyne method
    instead (the two methods are mutually incompatible in chroma).

    The methods are told apart by band energy: the ME-SECAM carrier pair
    lives around 654/811 kHz while the method 1 pair lives around
    1062.5/1101.6 kHz, and unlike a carrier cluster measurement this works
    regardless of content saturation. (The back porch is not a usable rest
    carrier reference here as it is for the ME-SECAM servo: the deck's
    divide-by-4 counter output takes most of the porch to settle after the
    blanking edges / SECAM subcarrier phase reversals.)"""
    diag = getattr(field.rf, "secam_method_diag", None)
    if diag is None or diag["done"]:
        return

    samp_rate = field.rf.chroma_afc.true_samp_rate
    freqs, power = sps.welch(chroma, fs=samp_rate, nperseg=16384)

    mesecam_band = power[(freqs >= 550e3) & (freqs < 900e3)].mean()
    method1_band = power[(freqs >= 1000e3) & (freqs < 1300e3)].mean()

    if method1_band > 3 * mesecam_band:
        diag["method1"] += 1
    elif mesecam_band > 3 * method1_band:
        diag["mesecam"] += 1

    diag["fields"] += 1
    if diag["fields"] >= 20:
        diag["done"] = True
        ldd.logger.debug(
            "SECAM recording method check: %d/%d fields matched method 1, "
            "%d looked like ME-SECAM"
            % (diag["method1"], diag["fields"], diag["mesecam"])
        )
        if diag["mesecam"] > diag["method1"] and diag["mesecam"] >= 5:
            ldd.logger.warning(
                "The colour-under carriers look like ME-SECAM "
                "(pair around 654/811 kHz) rather than SECAM method 1 "
                "(1062.5/1101.6 kHz). If the colour comes out wrong, "
                "decode with --system MESECAM instead."
            )


def _process_chroma_secam_method1(field, chroma, linesout, outwidth, burstarea):
    """SECAM method 1 chroma restoration: x4 phase multiplication instead of
    a heterodyne mix, plus BT.470 bell amplitude regeneration."""
    _secam_method_diagnostic(field, chroma, linesout, outwidth)

    afc = field.rf.chroma_afc
    # Peak amplitude such that the undeviated carrier lands near the same
    # porch RMS level the other formats' chroma AGC normalizes to.
    rest_amplitude = field.rf.SysParams["burst_abs_ref"] * np.sqrt(2.0)
    restored, inst_freq, envelope = upconvert_secam_method1(
        chroma,
        afc.true_samp_rate,
        field.rf.Filters["FSecamUnder"],
        afc.carrier_mult,
        rest_amplitude,
        return_envelope=True,
    )

    STARTING_LINE = 16
    first_line = max(STARTING_LINE, field.burst_detected_line)

    # Give downstream decoders the undeviated blanking-interval reference the
    # standard promises them; what comes off tape there is the record
    # divider's settling transient (see regenerate_secam_blanking). This is a
    # two-pass restore: the first pass above identifies the lines, then the
    # blanking is replaced in the colour-under domain and the restoration is
    # run again on the cleaned signal, so the zero-phase filtering never gets
    # to smear the transient into the picture or the porch reference. The
    # interval runs right up to active video, so the fade-out lands on the
    # picture's own (band-limited, desaturated) colour turn-on strip rather
    # than next to the decoders' porch measurement window.
    porch_end_px = int(field.usectooutpx(field.rf.SysParams["activeVideoUS"][0]))
    fit = fit_secam_line_alternation(
        inst_freq, linesout, outwidth, first_line, porch_end_px
    )
    flywheel = getattr(field.rf, "secam_parity_flywheel", None)
    if flywheel is None:
        flywheel = SecamParityFlywheel()
        field.rf.secam_parity_flywheel = flywheel
    dr_on_even, parity_source = flywheel.resolve(field.readloc, fit)

    if dr_on_even is not None:
        # The record chain's blanking-edge transient sets in slightly before
        # the nominal end of active video (the source's own blanking edge
        # lands inside the TBC active window), so the splice starts a little
        # early.
        blank_start_px = int(
            field.usectooutpx(field.rf.SysParams["activeVideoUS"][1] - 0.85)
        )
        cleaned = regenerate_secam_blanking(
            chroma,
            envelope,
            afc.true_samp_rate,
            linesout,
            outwidth,
            blank_start_px,
            porch_end_px,
            first_line,
            dr_on_even,
            afc.carrier_mult,
        )
        restored, inst_freq = upconvert_secam_method1(
            cleaned,
            afc.true_samp_rate,
            field.rf.Filters["FSecamUnder"],
            afc.carrier_mult,
            rest_amplitude,
        )
        ldd.logger.debug(
            "SECAM blanking reference regenerated (%s, fit confidence %s)"
            % (parity_source, "%.02f" % fit[1] if fit is not None else "n/a")
        )
    else:
        ldd.logger.debug(
            "SECAM blanking left as-is (line ident confidence too low, "
            "no parity lock)"
        )

    uphet = restored[: linesout * outwidth]

    # Block-anchored final band-pass (same band as ME-SECAM).
    uphet = sosfiltfilt_rust(field.rf.Filters["FChromaFinal"], uphet)

    # No per-line chroma AGC here: the amplitude envelope was synthesised
    # from the BT.470 bell above, and normalizing every line to its porch
    # level would flatten the intended foR/foB rest amplitude difference.
    # Just blank the vertical interval / colour-killed lines and log the
    # porch level like acc() does for the other formats.
    uphet[: first_line * outwidth] = 0

    porch_rms_total = 0.0
    for linenumber in range(STARTING_LINE, linesout):
        linestart = linenumber * outwidth
        porch_rms_total += lddu.rms(
            uphet[linestart + burstarea[0] : linestart + burstarea[1]]
        )

    return uphet


@cache
def _gen_chroma_fft_filter(
    filter_len: int,
    fsc: float,
    color_under_carrier_f: float,
    bw_lower_hz: float,
    heterodyne_attenuation_db: float,
    order: int,
) -> np.ndarray:
    """
    Generate asymmetric Super-Gaussian bandpass mask in the frequency domain.
    """

    # calculate upper bandwidth limit that is required to remove the heterodyne up conversion product
    # at the supplied attenuation
    A = 10.0 ** (-abs(heterodyne_attenuation_db) / 20.0)
    delta_f = 2.0 * color_under_carrier_f
    exponent = 1.0 / (2.0 * order)
    bw_upper_hz = delta_f / ((-np.log(A)) ** exponent)

    freqs_up = sps_fft.rfftfreq(filter_len, d=1.0 / (fsc * 4.0))
    mask = np.zeros_like(freqs_up, dtype=np.float64)

    lower_idx = freqs_up <= fsc
    upper_idx = freqs_up > fsc

    # asymmetric Super-Gaussian evaluation around subcarrier (fsc)
    mask[lower_idx] = np.exp(-((freqs_up[lower_idx] - fsc) / bw_lower_hz) ** (2 * order))
    mask[upper_idx] = np.exp(-((freqs_up[upper_idx] - fsc) / bw_upper_hz) ** (2 * order))

    return mask


def filter_chroma_fft(
    uphet: np.ndarray, 
    fsc: float,
    color_under_carrier_f: float,
    bw_lower_hz: float,               # Lower chroma bandwidth (1.3 MHz below fsc)
    heterodyne_attenuation_db: float, # Rejection target at sum product (dB)
    order: int = 2,                   # filter order
    pad_samples: int = 256,
) -> np.ndarray:
    """
    Zero-phase FFT bandpass filter using a Super-Gaussian mask.
    Dynamically computes bw_upper_hz from color_under_carrier_f to guarantee
    stopband attenuation at the heterodyne sum product.
    """
    N_raw = len(uphet)

    # pad to nearest fast FFT length (combines 2, 3, 5, 7 prime factors)
    min_pad_len = N_raw + 2 * max(1, int(pad_samples))
    N_up = sps_fft.next_fast_len(min_pad_len)

    # Symmetric pad calculation to center the signal in the fast length array
    pad_left = (N_up - N_raw) // 2
    pad_right = N_up - N_raw - pad_left

    x_padded = np.pad(uphet, (pad_left, pad_right), mode='reflect')

    mask = _gen_chroma_fft_filter(
        N_up,
        fsc,
        color_under_carrier_f,
        bw_lower_hz,
        heterodyne_attenuation_db,
        order
    )

    # apply filter against Forward Real FFT
    F_filtered = sps_fft.rfft(x_padded)
    F_filtered *= mask

    # return to real signal
    chroma_padded = sps_fft.irfft(F_filtered, n=N_up)

    # remove padding
    return chroma_padded[pad_left : pad_left + N_raw]


def luma_beat_wanted(field):
    """Whether the color-under beat correction should run on this field.

    Gated on the decoder's own burst detection rather than on a test of its own.
    The color killer already decides whether a field carries color-under, and
    where it does not there is no beat to cancel - only luma that leaked through
    the chroma band pass, which is the one thing the fit must not lock onto.
    Measured on this deck the burst stands at 26k to 33k on recordings with
    colour and at 595 on one recorded without it, so the killer's own threshold
    separates them by a wide margin.

    `burst_magnitude_avg` is measured on every field whether or not `--ck` is
    given, so this holds without it. With `--ck` the killer stops the chroma
    earlier still, and the two agree.
    """
    return (
        field.rf.options.luma_beat != 0
        and getattr(field, "burst_magnitude_avg", 0.0) >= BURST_MAGNITUDE_THRESHOLD
    )


def process_chroma(
    field,
    disable_deemph=False,
    disable_comb=False,
    disable_tracking_cafc=False,
    do_chroma_deemphasis=False,
):
    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    if field.burst_detected_line == -1:
        # skip chroma if the color killer is active for the whole field
        return np.zeros((linesout * outwidth), dtype=np.float32)
    
    if (
        not field.rf.options.disable_phase_correction
        and field.rf.color_system == "NTSC"
    ):
        parity, measured = colour_frame_parity(field)
        field.fieldPhaseID, base_phase = ntsc_color_framing_map[
            (field.isFirstField, parity)
        ]
        field.colour_framing_measured = measured
        target_phase = base_phase - ntsc_color_framing_phase_shift
        # BEHAVIOUR PRESERVED DELIBERATELY, AND A FINDING RECORDED WITH IT.
        # This reads the SHIFTED target, and with the 33 degree shift folded
        # in both possible values (-33 and 147) are non-zero - so the test is
        # always true and the `-1` branch is unreachable. The compensation
        # therefore never reverses with the colour frame.
        #
        # That is left exactly as it is. Ethan has ruled that the shift is
        # part of the output standard and is to be kept, and that this is NOT
        # the source of the rainbowing it was briefly suspected of - so
        # whether the `-1` branch was ever meant to fire is an open question
        # about intent, not a defect to be silently repaired. Reading the base
        # instead would make it fire on half of all fields and move every
        # decode's output.
        chroma_shift_direction = 1 if target_phase else -1
    else:
        chroma_shift_direction = 0

    # Run TBC/downscale on chroma (if new field, else uses cache)
    # Cached if chroma process is run multiple times on one field due to track detection.
    if field.chroma_tbc_buffer is None:
        # shift the chroma to reverse group delay caused by the color under heterodyne filter
        # this is dependent on color framing, and is disabled if color framing is disabled
        # TODO: shift amount may need tuning / needs validation
        chroma_subcarrier_delay_cycles = field.rf.SysParams['fsc_mhz'] * 1e6 / (2.0 * np.pi * field.rf.DecoderParams["color_under_carrier"])
        chroma_subcarrier_delay_samples = chroma_subcarrier_delay_cycles * 4
        chroma_downscale_shift = chroma_subcarrier_delay_samples * chroma_shift_direction

        chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst", shift=chroma_downscale_shift)

        # If chroma AFC is enabled
        if field.rf.do_cafc:
            # it does the chroma filtering AFTER the TBC
            chroma = chroma_color_under_filter(
                chroma,
                field.rf.chroma_afc.get_chroma_bandpass(),
                len(chroma),
                field.rf.Filters["FVideoNotch"],
                field.rf.notch,
                move=(int(10 * (field.rf.sys_params["outfreq"] / 40))),
                audio_notch=field.rf.Filters.get("FChromaAudioNotch", None),
            )

            if not disable_tracking_cafc:
                spec, meas, offset, cphase = field.rf.chroma_afc.freqOffset(chroma)
                ldd.logger.debug(
                    "Chroma under AFC: %.02f kHz, Offset (long term): %.02f Hz, Phase: %.02f deg"
                    % (meas / 1e3, offset, cphase * 360 / (2 * np.pi))
                )

        if (
            field.rf.color_system == "MESECAM"
            and field.rf.options.secam_carrier_servo
        ):
            # Measure the rest carrier pair on the late back porch,
            # 3.7 to 0.3 us before active video starts.
            active_start_px = field.usectooutpx(field.rf.SysParams["activeVideoUS"][0])
            porch_window = (int(active_start_px) - 65, int(active_start_px) - 5)

            carrier_offset = measure_secam_under_carrier_offset(
                chroma,
                linesout,
                outwidth,
                porch_window,
                field.rf.chroma_afc.true_samp_rate,
                field.rf.DecoderParams["color_under_carrier"],
            )
            if carrier_offset is not None:
                field.rf.secam_servo_avg.push(carrier_offset)
                ldd.logger.debug(
                    "SECAM carrier servo: measured offset %.02f Hz" % carrier_offset
                )

        field.rf.chroma_tbc_buffer = chroma
        field.chroma_tbc_buffer = chroma
    else:
        chroma = field.chroma_tbc_buffer

    burstarea = get_burst_area(field)

    if field.rf.color_system == "SECAM":
        # Method 1 restores the chroma block by phase multiplication rather
        # than by mixing against a heterodyne, so it skips the shared
        # up-conversion path below entirely.
        return _process_chroma_secam_method1(
            field, chroma, linesout, outwidth, burstarea
        )

    # For NTSC, the color burst amplitude is doubled when recording, so we have to undo that.
    if field.rf.color_system == "NTSC":
        if not disable_deemph:
            chroma = burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea)

    if (
        not field.rf.options.disable_phase_correction
        and field.rf.color_system == "NTSC"
    ):
        target_phase_even = target_phase
        target_phase_odd = target_phase

        # TODO: PAL color framing is disabled for now.
        #       need to find a reliable way to detect if this is field 1,2 vs 3,4
        # if field.rf.color_system == "PAL":
        #     line_6_burst_present = field.phase_sequence[4 + lineoffset][3] > field.burst_magnitude_avg / 3
        #     field.fieldPhaseID, target_phase_even, target_phase_odd = pal_color_framing_map[
        #         (field.isFirstField, line_6_burst_present, (field.field_number // 4) % 2)
        #     ]

        # this uses the burst measurements to interpolate the correct phase of the color under heterodyne
        # phase issues are corrected continiously for each sample using a linear spline interpolated from the burst measurements
        # the mixing is performed on the upsampled signal to avoid aliasing introduced from the up-heterodyne mixing product
        if luma_beat_wanted(field):
            # The color-under as the time base correction leaves it, before the
            # up-conversion rewrites its phase. The beat in the luma carries the
            # phase the TAPE held, which the burst-locked interpolation below is
            # about to replace with a target one.
            field.chroma_under_tbc = np.array(chroma, dtype=np.float64)
        upconvert_chroma_phase_comp(
            chroma, # modifies this in place
            lineoffset,
            outwidth,
            field.phase_sequence,
            field.rf.DecoderParams["color_under_carrier"],
            field.rf.SysParams["fsc_mhz"] * 1e6,
            target_phase_even,
            target_phase_odd,
        )
        uphet = chroma
    else:
        if field.rf.chroma_afc.conversion_lo is not None:
            # Explicit conversion LO (ME-SECAM): trim it by the smoothed
            # measured carrier offset (cancelling the recording VCR's
            # converter crystal error), and keep the heterodyne phase
            # continuous across fields.
            lo_trim = 0.0
            # Holds either live servo measurements or a seeded/fixed trim
            # (secam_lo_trim); with the servo disabled and no seed it's empty.
            if field.rf.secam_servo_avg.has_values():
                # Quantize so measurement noise doesn't dither the LO.
                lo_trim = np.clip(
                    round(field.rf.secam_servo_avg.pull() / 10.0) * 10.0,
                    -10e3,
                    10e3,
                )
            field.rf.chroma_afc.updateConversion(
                lo_trim, field.field_number * linesout * outwidth
            )
            chroma_heterodyne = field.rf.chroma_afc.getChromaHet()
        else:
            chroma_heterodyne = (
                field.rf.chroma_afc.getChromaHet()
                if (field.rf.do_cafc and not disable_tracking_cafc)
                else field.rf.chroma_heterodyne
            )

        uphet = np.zeros((linesout * outwidth), dtype=np.float32)
        upconvert_chroma(
            chroma,
            uphet,
            lineoffset,
            outwidth,
            field.phase_sequence,
            chroma_heterodyne
        )

    # Filter out unwanted frequencies from the final chroma signal.
    # Mixing the signals will produce waves at the difference and sum of the
    # frequencies. We only want the difference wave which is at the correct color
    # carrier frequency here.
    if field.rf.color_system == "MESECAM":
        # The restored SECAM FM block is anchored at conversion_lo -
        # color_under (4.328125 MHz), not fsc, so the fsc-anchored FFT mask
        # sits ~106 kHz high on it and loses the tight top edge that
        # suppresses high-side FM splatter from saturated transitions. Keep
        # the block-anchored Butterworth here.
        uphet = sosfiltfilt_rust(field.rf.Filters["FChromaFinal"], uphet)
    else:
        uphet = filter_chroma_fft(
            uphet,
            field.rf.SysParams["fsc_mhz"] * 1e6,
            field.rf.DecoderParams["color_under_carrier"],
            1.3e6, # lower chroma bandwidth (roughly this for PAL / NTSC)
            80.0   # heterodyne up-mixing attenuation
        )

    if do_chroma_deemphasis:
        b, a = field.rf.Filters["chroma_deemphasis"]
        uphet = sps.lfilter(b, a, uphet)

    # Basic comb filter for NTSC to calm the color a little.
    if not disable_comb:
        if field.rf.color_system == "NTSC":
            uphet = comb_c_ntsc(uphet, outwidth)
        else:
            uphet = comb_c_pal(uphet, outwidth)

    # Chroma AGC
    # average ACG over each field
    # save a separate gain per field
    chroma_average_state = (
        field.rf.field_averages.chroma_level_even
        if field.field_number % 2 == 0 else
        field.rf.field_averages.chroma_level_odd
    )

    decode_average_count = len(chroma_average_state)
    if decode_average_count > 0:
        decode_average = np.mean([c[0] for c in chroma_average_state])
    else:
        decode_average = 0

    field_average, chroma_noise_floor = chroma_automatic_gain(
        uphet,
        field.rf.SysParams["burst_abs_ref"],
        field.phase_sequence,
        field.burst_detected_line,
        math.floor(field.usectooutpx(field.rf.SysParams["hsyncPulseUS"])),
        decode_average,
        decode_average_count
    )

    chroma_average_state.append((field_average, chroma_noise_floor))

    # CTI does NOT run here. It is a cosmetic sharpener - it accelerates the
    # sweep between colour states - so anything that measures a physical
    # property of the chroma must read the chroma before it, not after.
    # `luma_beat` is the case in point: it scales its correction by the
    # chroma's saturation, and CTI reshapes amplitude at exactly the
    # transitions that measurement is taken across. The noise floor the
    # gate needs is carried forward so CTI can run last, in `decode_chroma`.
    field.chroma_noise_floor = chroma_noise_floor

    return uphet


EDGE_SPAN_FROM_RISE = 1.815
"""The 1%-99% span of a smooth edge as a multiple of its 10%-90% rise.
Exact for a Gaussian edge, which is what a cascade of band limits
approaches. The sweep anchors have to straddle the whole transition, so the
radius is half of this - 0.907 of the measured rise."""

MAX_SWEEP_CYCLES = 8
"""The widest sweep the operator may be asked for, in subcarrier cycles.
Also the lead-in the rise measurement reaches back by, so a chroma path
slow enough to want the widest radius is still measurable."""

SUBCARRIER_QUADRATURE = 4
"""The radius must be a whole number of subcarrier cycles. CTI reads
`(x[s], x[s-1])` as a quadrature pair, and at 4fsc the frame rotates 90 deg
per sample, so `x[s +/- R]` lies in the same frame only when R is a
multiple of 4. A radius that is not is a rotated vector, not a neighbour."""


BURST_GATE_RISE_US = 0.300
"""RS-170A's own burst gate transition. The reference the measured burst is
differenced against is the specification's shape, not a fitted one - which
is what makes the difference a residual rather than a comparison of two
measurements."""


def decoder_envelope_response(rf, frequencies_hz):
    """What the DECODER'S OWN chroma band-pass does to the colour-under's
    envelope, per baseband modulation frequency.

    This has to be divided out of the burst measurement before anything is
    derived from it, and the reason is the same one `rf_path_response`
    already records on the luma side: leaving the decoder's own filter in
    the reference books its roll-off into the measured tape response, from
    where a correction would faithfully undo the separation filter that
    exists to keep the chroma apart.

    It is most of what is there. `FVideoBurst` is a fourth-order band-pass
    from 60 kHz to 1.2 MHz about a 629 kHz carrier, so it passes envelope
    components only to about 571 kHz - and the roll-off measured on the
    burst is strongest at 573 kHz. Nearly all of that is ours.

    An envelope component at `f` rides the carrier as the sideband pair
    `carrier +/- f`, so what the envelope sees is the mean of the filter's
    magnitude at the two - the same carrier-normalization structure the RF
    side uses. Returns None where the filter or the carrier is unavailable.
    """
    from scipy.signal import sosfreqz

    sos = rf.Filters.get("FVideoBurst")
    carrier = rf.DecoderParams.get("color_under_carrier")
    rate = float(getattr(rf, "freq_hz", 0.0) or 0.0)
    if sos is None or not carrier or rate <= 0:
        return None
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    upper = carrier + frequencies
    lower = np.abs(carrier - frequencies)
    reachable = (upper < 0.5 * rate) & (lower < 0.5 * rate)
    if not np.any(reachable):
        return None
    try:
        _, h_up = sosfreqz(sos, worN=2.0 * np.pi * upper / rate)
        _, h_down = sosfreqz(sos, worN=2.0 * np.pi * lower / rate)
    except Exception:                                        # noqa: BLE001
        return None
    response = 0.5 * (np.abs(h_up) + np.abs(h_down))
    return np.where(reachable, response, np.nan)


def _complex_envelope(block, first_column):
    """THE MULTIPLY THE CHROMA INSTRUMENT WAS MISSING.

    A real band-pass signal sampled at four times its carrier is
    `c_n = Re{b_n j^n}`, so the complex envelope comes back by rotating:

        y_n = 2 c_n (-j)^n = b_n + (-1)^n conj(b_n)

    and the second term alternates at Nyquist, where a symmetric 3-tap
    `(1, 2, 1)/4` has response `(1 - 2 + 1)/4 = 0` EXACTLY. So one multiply
    and one three-tap recover `b_n`, and the filter is symmetric so it adds
    no phase of its own.

    The annihilation is exact in the interior and NOT at the two ends, which
    are carried through unfiltered so the window keeps its length and the
    caller's indexing is unchanged. Measured on a planted envelope, that
    leaves 1.0 per cent of the conjugate image - a hundredfold suppression,
    not an infinite one. Trimming two samples instead would make it exact;
    keeping the length was judged worth more than the last two decades on a
    term that is already 40 dB down.

    Measured against a planted complex envelope, interior samples only: the
    magnitude comes back to 2.2e-03 and the PHASE to 1.7e-03 radians. The
    magnitude alone is 19 times better than `hypot`'s 4.2e-02 on the same
    signal - so this is not a trade of accuracy for phase, it is better at
    both.

    WHY THIS AND NOT THE `(i, q)` PAIR BESIDE IT. `hypot(c_n, c_{n-1})` reads
    a magnitude from two DIFFERENT samples, so its result is centred half a
    sample early - 34.9 ns at 4fsc, which is the same size as the group delay
    the chroma measurements are trying to resolve - and it discards the angle
    between them, which is the whole of the phase.

    WHAT THE DISCARDED HALF WAS CARRYING, measured: taking the magnitude
    makes the burst instrument EXACTLY RANK 2 OF 4 in the complex response's
    four symmetry classes, singular values [71.86, 4.123, 0, 0]. The two
    unobserved directions are the anti-Hermitian half of `log H` - the
    sideband amplitude TILT, which is the tape's own roll-off across the
    colour-under band, and the even phase term. A real envelope passes
    through them into quadrature, where a magnitude sees them only at second
    order. With the complex envelope the same instrument reads
    [145.45, 145.45, 13.64, 13.64], condition 10.67 against 7.2e301.

    `first_column` is the block column the window starts at, because the
    rotation is referenced to the LINE's own origin and not to the window's -
    an offset there is a constant phase error on every measurement taken
    from it.
    """
    values = np.asarray(block, dtype=np.float64)
    columns = np.arange(first_column, first_column + values.shape[1])
    rotation = (-1j) ** columns
    rotated = 2.0 * values * rotation[None, :]
    if rotated.shape[1] < 3:
        return rotated
    # (1, 2, 1)/4 along the sample axis: zero at Nyquist, so the (-1)^n
    # conjugate image is removed exactly. The ends are carried through
    # unfiltered rather than trimmed, so the window keeps its length and the
    # caller's indexing is unchanged.
    smoothed = np.empty_like(rotated)
    smoothed[:, 1:-1] = 0.25 * (rotated[:, :-2] + 2.0 * rotated[:, 1:-1]
                                + rotated[:, 2:])
    smoothed[:, 0] = rotated[:, 0]
    smoothed[:, -1] = rotated[:, -1]
    return smoothed


def burst_complex_envelope(field, uphet):
    """The burst's COMPLEX envelope, folded over the field's lines.

    The same window and the same fold as `burst_envelope`, carrying the phase
    that one throws away. Returned separately rather than replacing it,
    because the magnitude profile feeds the shipped rise-time and frequency
    response measurements and changing those would move decode output.

    The fold is a median of the real and imaginary parts taken separately,
    which is robust in the same way the magnitude's median is. That is only
    valid because the burst's phase is CONSTANT across a field at this point
    in the chain - `upconvert_chroma_phase_comp` has already applied the
    burst lock, so the output burst sits at minus the field's target phase on
    every line - and it would be wrong on the colour-under signal before that
    lock, where the head rotation alternates the phase by 90 degrees a line.
    """
    rate = getattr(field, "outlinelen", 0)
    if not rate:
        return None
    try:
        burst_start, burst_end = get_burst_area(field)
    except Exception:                                        # noqa: BLE001
        return None
    start = (field.lineoffset + 1) * rate
    lines = (len(uphet) - start) // rate
    if lines < 8 or burst_end <= burst_start:
        return None
    lead_in = MAX_SWEEP_CYCLES * SUBCARRIER_QUADRATURE
    burst_start = max(burst_start - lead_in, 1)
    block = np.asarray(uphet[start:start + lines * rate]).reshape(lines, rate)
    window = _complex_envelope(block[:, burst_start:burst_end], burst_start)
    if window.shape[1] < 8:
        return None
    profile = (np.median(window.real, axis=0)
               + 1j * np.median(window.imag, axis=0))
    if not np.all(np.isfinite(profile)):
        return None
    return profile, burst_start


def burst_envelope(field, uphet):
    """The burst's envelope, folded over every line of the field, and the
    sample it starts at.

    One instrument, two consumers: the rise time and the frequency
    response are both read off this, so they cannot disagree about what
    the burst looks like.
    """
    rate = getattr(field, "outlinelen", 0)
    if not rate:
        return None
    try:
        burst_start, burst_end = get_burst_area(field)
    except Exception:                                        # noqa: BLE001
        return None
    start = (field.lineoffset + 1) * rate
    lines = (len(uphet) - start) // rate
    if lines < 8 or burst_end <= burst_start:
        return None
    lead_in = MAX_SWEEP_CYCLES * SUBCARRIER_QUADRATURE
    burst_start = max(burst_start - lead_in, 1)
    block = np.asarray(uphet[start:start + lines * rate]).reshape(lines, rate)
    in_phase = block[:, burst_start:burst_end].astype(np.float64)
    quadrature = block[:, burst_start - 1:burst_end - 1].astype(np.float64)
    window = np.hypot(in_phase, quadrature)
    if window.shape[1] < 8:
        return None
    profile = np.median(window, axis=0)
    if not np.all(np.isfinite(profile)):
        return None
    return profile, burst_start


def measured_chroma_rise(field, uphet):
    """The chroma path's 10%-90% step response, in output samples, measured
    from the COLOUR BURST - or None if it cannot be measured.

    The burst is the instrument here because it is a gated subcarrier at a
    fixed place on every line, so its leading envelope IS the step response
    of everything the chroma has been through, with no edge detection and
    no dependence on picture content. Averaged over the field's lines it is
    a far steadier estimate than any transition in the picture.

    Two instruments were tried before this one and both are wrong for the
    job, which is worth recording so they are not reached for again.
    `luma_beat.path_shaping()` is the RF path's sideband imbalance about the
    luma carrier, on an axis of RF offset - not a chroma response at all.
    And `_measure_transfer_response` in this module is a Wiener weight in
    frequency, on an axis of baseband MODULATION RATE, describing how much
    of the luma envelope's noise is shared with the colour-under as the loss
    gets faster; its level is discarded by design. Neither predicts a rise
    time. Measured on the home tape the transfer also fails its own null
    control - a matched null with the coupling destroyed reproduces the
    stored shape band for band, peak coherence 8e-4 against a null of
    1-9e-4 - so on that material it resolves no roll-off to shape anything
    with.
    """
    rate = getattr(field, "outlinelen", 0)
    if not rate:
        return None
    try:
        burst_start, burst_end = get_burst_area(field)
    except Exception:                                        # noqa: BLE001
        return None
    # `uphet` holds the whole field and CTI starts a line in, so the rows
    # available are what remains AFTER that offset - not `outlinecount`,
    # which is the buffer's full height and overruns it by exactly the
    # offset. Getting this wrong makes the measurement decline on every
    # real field while still passing a fixture built one line too long.
    start = (field.lineoffset + 1) * rate
    lines = (len(uphet) - start) // rate
    if lines < 8 or burst_end <= burst_start:
        return None
    # `get_burst_area` leaves only a few samples ahead of the burst, which
    # is enough to gate it but not to SEE it start: a rise of order 17
    # samples never reaches its own 10% point inside that window, and the
    # measurement then declines on every field. Reach back into the
    # breezeway instead, which is blanking and therefore the honest floor.
    # The allowance is the widest rise the operator can be asked for, so a
    # slow chroma path is still measurable rather than silently falling back.
    lead_in = MAX_SWEEP_CYCLES * SUBCARRIER_QUADRATURE
    burst_start = max(burst_start - lead_in, 1)
    block = np.asarray(uphet[start:start + lines * rate]).reshape(lines, rate)
    if burst_start < 1:
        return None
    # The chroma here is a REAL modulated signal at 4fsc, so its envelope is
    # the quadrature magnitude of a sample and the one before it - the same
    # pair CTI itself reads as (i, q). Taking `abs` of the real signal would
    # measure the rectified subcarrier, not the envelope.
    in_phase = block[:, burst_start:burst_end].astype(np.float64)
    quadrature = block[:, burst_start - 1:burst_end - 1].astype(np.float64)
    window = np.hypot(in_phase, quadrature)
    if window.shape[1] < 8:
        return None
    profile = np.median(window, axis=0)
    if not np.all(np.isfinite(profile)):
        return None
    floor, ceiling = profile.min(), profile.max()
    if ceiling - floor <= 0:
        return None
    normalised = (profile - floor) / (ceiling - floor)
    peak = int(np.argmax(normalised))
    if peak < 2:
        return None
    leading = normalised[:peak + 1]
    low = np.flatnonzero(leading <= 0.1)
    high = np.flatnonzero(leading >= 0.9)
    if len(low) == 0 or len(high) == 0 or high[0] <= low[-1]:
        return None

    def crossing(index, level):
        """Where the profile passes `level` between two samples. Reading
        the bracketing indices alone biases the rise a whole sample long,
        which on a 17 sample rise is 6% and always in the same direction."""
        if index <= 0 or index >= len(leading):
            return float(index)
        before, after = leading[index - 1], leading[index]
        if after == before:
            return float(index)
        return (index - 1) + (level - before) / (after - before)

    return float(crossing(high[0], 0.9) - crossing(low[-1] + 1, 0.1))


def chroma_path_response(field, uphet):
    """The TAPE's roll-off on the chroma envelope, per frequency.

    Ethan: "I need to shape the slope of the improvement based on the
    entire burst envelope ... Expected is the spec derived shape of the
    burst, and actual is the measured shape of the burst."

    So this is `docs/FREQUENCY_DERIVATION.md` applied to the burst, and it
    replaces the single rise time the width used to come from. A 10-90 rise
    collapses the envelope to one scalar; the envelope has a shape, and the
    shape is the response.

    Steps 1 and 2 of that algorithm, in order and for the reasons given
    there:

      ALIGN FIRST. The measured burst is delayed relative to the spec
      window and a delay is not a roll-off. Measured, aligning first drops
      the residual from 0.2328 to 0.1504 rms - a third of what looks like
      response is position.

      DIFFERENCE, NEVER A RATIO. At 430 kHz the ratio reads -12.55 dB and
      the difference -1.76, because the expected magnitude there is small.
      That is the spectral division this project has refuted twice. Bins
      where the expected carries no energy are reported as unusable rather
      than as a deep roll-off measured on nothing - on this material that
      is 41 bins of 50.

    Returns (frequencies_hz, log_response, usable) or None.
    """
    profile = burst_envelope(field, uphet)
    if profile is None:
        return None
    measured, first_sample = profile
    rate = float(field.rf.SysParams["outfreq"]) * 1e6
    reference = _spec_burst_envelope(field, first_sample, len(measured))
    if reference is None:
        return None

    # The flat top comes from the SPECIFICATION, not from a fraction of the
    # array. The window reaches back into the breezeway so the rise can be
    # seen, which puts the middle third on the rising EDGE - normalising
    # there divides by a point on the transition and inflates the whole
    # envelope. Caught by a planted burst reading 12.8 at its own flat top.
    flat = _spec_flat_span(field, first_sample, len(measured))
    if flat is None:
        return None
    scale = np.median(measured[flat])
    if not scale > 0:
        return None
    measured = measured / scale
    reference = reference / max(np.median(reference[flat]), 1e-12)

    grid = np.arange(len(measured), dtype=np.float64)
    shift = _alignment_shift(measured, reference)
    aligned = np.interp(grid + shift, grid, measured)

    spectrum_measured = np.abs(np.fft.rfft(aligned - aligned.mean()))
    spectrum_expected = np.abs(np.fft.rfft(reference - reference.mean()))
    frequencies = np.fft.rfftfreq(len(aligned), d=1.0 / rate)

    # The decoder's own band-pass is divided out of BOTH, so what is left is
    # the tape's contribution and not ours.
    ours = decoder_envelope_response(field.rf, frequencies)
    if ours is not None:
        keep = np.isfinite(ours) & (ours > 1e-3)
        spectrum_expected = np.where(keep, spectrum_expected * ours,
                                     spectrum_expected)

    floor = TRANSFER_EVIDENCE_FRACTION * spectrum_expected.max()
    usable = spectrum_expected > floor
    usable[0] = False
    log_response = np.zeros_like(frequencies)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_response[usable] = np.log(
            np.maximum(spectrum_measured[usable], 1e-9)
            / np.maximum(spectrum_expected[usable], 1e-9))
    return frequencies, log_response, usable


TRANSFER_EVIDENCE_FRACTION = 0.05
"""How much energy the expected spectrum must carry in a bin before that
bin is allowed to speak. Below it the comparison has no evidence either
way, and reporting it anyway is how a ratio manufactures a roll-off."""


def _alignment_shift(measured, reference):
    """Where the measured envelope sits relative to the reference, to
    sub-sample, from the correlation peak."""
    correlation = np.correlate(measured - measured.mean(),
                               reference - reference.mean(), mode="same")
    peak = int(np.argmax(correlation))
    if peak <= 0 or peak >= len(correlation) - 1:
        return 0.0
    y0, y1, y2 = correlation[peak - 1], correlation[peak], correlation[peak + 1]
    denominator = y0 - 2.0 * y1 + y2
    offset = 0.0 if denominator == 0 else 0.5 * (y0 - y2) / denominator
    return (peak - len(measured) // 2) + offset


def _spec_burst_envelope(field, first_sample, count):
    """The specification's own burst envelope on the measured grid: flat
    across the spec window, with the spec gate transition at each end."""
    from scipy.special import erf

    sys_params = field.rf.SysParams
    burst = sys_params.get("colorBurstUS")
    if not burst:
        return None
    samples = first_sample + np.arange(count, dtype=np.float64)
    microseconds = samples / float(sys_params["outfreq"])
    # erf's 10-90 width is 2.5631 sigma
    sigma = BURST_GATE_RISE_US / 2.563103
    rising = 0.5 * (1.0 + erf((microseconds - burst[0]) / (sigma * np.sqrt(2.0))))
    falling = 0.5 * (1.0 - erf((microseconds - burst[1]) / (sigma * np.sqrt(2.0))))
    return rising * falling


def _spec_flat_span(field, first_sample, count):
    """Where the burst is flat, in window indices, from the spec window
    with the gate's own transition guarded off each end."""
    sys_params = field.rf.SysParams
    burst = sys_params.get("colorBurstUS")
    if not burst:
        return None
    per_us = float(sys_params["outfreq"])
    guard = 2.0 * BURST_GATE_RISE_US
    low = int(np.ceil((burst[0] + guard) * per_us)) - first_sample
    high = int(np.floor((burst[1] - guard) * per_us)) - first_sample
    low = max(low, 0)
    high = min(high, count)
    if high - low < 4:
        return None
    return slice(low, high)


def sharpener_passes(field, uphet, radius):
    """Per-pass radii and weights for the sharpener, solved JOINTLY.

    The operator used one radius and a fixed geometric decay of 0.25 across
    four passes, so its emphasis had a shape nothing measured. Ethan asked
    for the slope of the improvement to follow the burst's own roll-off,
    which needs passes at DIFFERENT radii - a pass of radius R emphasises
    features of that scale, so a set of them is a small filter bank - and
    weights chosen so the bank's combined response inverts what was
    measured.

    The weights are solved TOGETHER, by least squares over the usable
    frequencies, because each pass's contribution changes the residual the
    others are fitted against. That is the governing rule of the derivation
    and it is why this is not a per-pass fit.

    Falls back to the historical geometric decay when the response cannot
    be measured, so a field the burst cannot be read on is sharpened
    exactly as it always was.
    """
    # A bank of DISTINCT radii. Halving until it reaches the quadrature
    # floor produced duplicates - two passes at radius 4 are one pass with
    # twice the weight, and they make the basis rank-deficient, so the
    # joint solve is fitting a direction that does not exist.
    wanted, seen = [], set()
    for step in range(SHARPENER_PASSES * 2):
        cycles = max(1, int(round(radius / (2 ** step) / SUBCARRIER_QUADRATURE)))
        candidate = cycles * SUBCARRIER_QUADRATURE
        if candidate not in seen:
            seen.add(candidate)
            wanted.append(candidate)
        if len(wanted) == SHARPENER_PASSES:
            break
    radii = np.array(wanted, dtype=np.int64)
    fallback = np.array([SHARPENER_DECAY ** p for p in range(len(radii))],
                        dtype=np.float64)

    measured = chroma_path_response(field, uphet)
    if measured is None:
        return radii, fallback
    frequencies, log_response, usable = measured
    if np.count_nonzero(usable) < 3:
        return radii, fallback

    rate = float(field.rf.SysParams["outfreq"]) * 1e6
    # What each pass does per frequency: a symmetric two-anchor sweep of
    # radius R is a raised-cosine emphasis, unity at DC and peaking where
    # half a cycle spans the sweep.
    angle = 2.0 * np.pi * frequencies[usable, None] * radii[None, :] / rate
    basis = 1.0 - np.cos(np.clip(angle, 0.0, np.pi))
    # The target: undo the measured loss, never more than the operator is
    # allowed to add, and err low - a boost that overshoots posterises.
    target = np.clip(-log_response[usable], 0.0, MAX_SHARPENER_BOOST)
    try:
        weights, *_ = np.linalg.lstsq(basis, target, rcond=None)
    except Exception:                                        # noqa: BLE001
        return radii, fallback
    if not np.all(np.isfinite(weights)):
        return radii, fallback
    # A pass may not pull the other way; the sharpener sharpens.
    weights = np.clip(weights, 0.0, SHARPENER_WEIGHT_LIMIT)
    if not weights.sum() > 0:
        return radii, fallback
    return radii, weights


SHARPENER_PASSES = 4
SHARPENER_DECAY = 0.25
"""The historical geometric decay, kept as the fallback for a field whose
burst cannot be measured."""

MAX_SHARPENER_BOOST = 1.5
"""The most the operator may be asked to add at any frequency, in nepers.
Past the derivation's own ceiling the median content transition starts
collapsing, which is posterisation of detail the tape really carries."""

SHARPENER_WEIGHT_LIMIT = 1.0


def chroma_sweep_radius(field, uphet):
    """The CTI radius in output samples, measured where possible.

    `--cti_width` sets it directly when the user asks for a number; "auto"
    derives it from the chroma's own measured rise, which is the point -
    the right radius is a property of the tape and the machine, not a knob.

    On the home tape the measurement gives a 16.8 sample rise and so a
    radius of 16, against the 8 the fixed default asked for. Swept on real
    content through the decoder's own operator, the fast tail of the
    transition distribution has an interior minimum at exactly 16, and the
    operator never overshoots at any radius, so the width was short by a
    factor of two. Past 16 the median content transition starts collapsing
    - that is posterisation of detail the tape really carries - which is
    where the derivation's ceiling and the sweep's agree.
    """
    width = getattr(field.rf.options, "cti_width", "auto")
    if width != "auto":
        return int(max(SUBCARRIER_QUADRATURE, int(width) * SUBCARRIER_QUADRATURE))
    rise = measured_chroma_rise(field, uphet)
    if rise is None or not np.isfinite(rise) or rise <= 0:
        return 2 * SUBCARRIER_QUADRATURE          # the historical default
    cycles = int(round(EDGE_SPAN_FROM_RISE * 0.5 * rise / SUBCARRIER_QUADRATURE))
    return int(np.clip(cycles, 1, MAX_SWEEP_CYCLES)) * SUBCARRIER_QUADRATURE


def apply_chroma_transient_improvement(field, uphet):
    """Sharpen the chroma's transitions, in place. Runs LAST, after every
    stage that measures the chroma, and returns whether it did anything."""
    if field.rf.options.cti_mix == 0:
        return False
    noise_floor = getattr(field, "chroma_noise_floor", None)
    if noise_floor is None:
        return False
    radius = chroma_sweep_radius(field, uphet)
    # The passes and their weights, solved against the burst's own measured
    # roll-off - the slope of the improvement rather than one radius and a
    # fixed decay.
    radii, weights = sharpener_passes(field, uphet, radius)
    # The gate is the noise floor's own quantity and no longer moves with
    # the radius: what a median absolute deviation has to clear to be a
    # real colour change is a property of the noise, not of how far the
    # operator reaches. Scaled by the widest sweep so a wider reach does
    # not lower the bar.
    threshold = noise_floor * math.sqrt(
        float(np.max(radii)) / SUBCARRIER_QUADRATURE)
    chroma_transient_improvement(
        uphet,
        (field.lineoffset + 1) * field.outlinelen,
        field.outlinelen,
        threshold,
        np.asarray(radii, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        field.rf.options.cti_mix,
    )
    return True


@njit(cache=True, fastmath=True, nogil=True)
def chroma_transient_improvement(
    chroma_data: np.ndarray,
    line_start: int,
    line_length: int,
    mad_threshold: float,
    sweep_radii: np.ndarray,
    pass_weights: np.ndarray,
    cti_mix: float,
) -> np.ndarray:
    """
    Accelerates the sweep rate between color states without warping phase.
    Operates symmetrically by measuring forward/backward vector neighbors simultaneously.
    """
    # THE SLOPE OF THE IMPROVEMENT. The passes used one radius and a fixed
    # geometric decay of 0.25, so the operator's emphasis had a shape that
    # nothing had measured. Each pass now has its OWN radius - a pass of
    # radius R emphasises features of that scale, so the set is a small
    # filter bank - and the weights arrive already solved, jointly, so the
    # bank's combined response follows the roll-off measured on the burst.
    num_passes = len(sweep_radii)
    mix_factors = np.empty(num_passes, dtype=np.float32)
    for p in range(num_passes):
        mix_factors[p] = cti_mix * pass_weights[p]

    # Establish spatial boundaries
    remaining_samples = chroma_data.shape[0] - line_start
    line_count = remaining_samples // line_length
    
    # The radius and the gate arrive already decided. They used to be
    # derived here from one knob - radius = cti_width * 4 and threshold =
    # noise * sqrt(cti_width) - which meant a change of width silently
    # moved the noise gate as well, so a width change was never a pure
    # width change. They are separate quantities and are now passed
    # separately: the radius comes from the measured chroma rise, the gate
    # from the measured noise floor.
    widest = 4
    for p in range(num_passes):
        if sweep_radii[p] > widest:
            widest = int(sweep_radii[p])

    # Protect against edge bleeding, against the WIDEST pass
    start_s = widest + 1
    end_s = line_length - (widest + 1)

    line_buffer = np.empty(line_length, dtype=chroma_data.dtype)

    # --- Optimized Vector-Ready Loop ---
    for l in range(line_count):
        curr_line_offset = line_start + (l * line_length)

        for p in range(num_passes):
            current_mix = mix_factors[p]
            sweep_radius = int(max(4, sweep_radii[p]))
            line_buffer[:] = chroma_data[curr_line_offset : curr_line_offset + line_length]

            for s in range(start_s, end_s):
                idx = curr_line_offset + s
                
                i_curr = line_buffer[s]
                q_curr = line_buffer[s - 1]
                
                i_past = line_buffer[s - sweep_radius]
                q_past = line_buffer[s - sweep_radius - 1]
                
                i_future = line_buffer[s + sweep_radius]
                q_future = line_buffer[s + sweep_radius - 1]
                
                # Measure vector distances
                i_delta_back = i_curr - i_past
                q_delta_back = q_curr - q_past
                dist_back = math.sqrt(i_delta_back * i_delta_back + q_delta_back * q_delta_back)
                
                i_delta_forw = i_future - i_curr
                q_delta_forw = q_future - q_curr
                dist_forw = math.sqrt(i_delta_forw * i_delta_forw + q_delta_forw * q_delta_forw)
                
                total_sweep_distance = dist_back + dist_forw
                
                # Noise Gate
                gate_mask = 1.0 if total_sweep_distance > mad_threshold else 0.0
                
                # Inherent Boundary (Naturally bounds to [0.0, 1.0) because distances are >= 0)
                norm_progress = dist_back / total_sweep_distance if total_sweep_distance != 0 else 0
                
                # Transform the progress via sigmoidal sweep-acceleration function
                is_lower = norm_progress < 0.5
                
                inv_prog = 1.0 - norm_progress
                t_low = 4.0 * (norm_progress * norm_progress)
                t_high = 1.0 - 4.0 * (inv_prog * inv_prog)
                
                # Select interpolation weights and anchor to the closer sample (before or after)
                t = t_low if is_lower else t_high
                anchor_a = i_past if is_lower else i_curr
                anchor_b = i_curr if is_lower else i_future
                
                # 4. Final vector mix
                # If gate_mask is 0.0, the delta cancels out and i_curr is cleanly written back
                i_target = anchor_a + t * (anchor_b - anchor_a)
                chroma_data[idx] = i_curr + (current_mix * gate_mask) * (i_target - i_curr)



class _FieldChromaStage:
    """One field-level chroma stage: its declared name and how to run it."""

    __slots__ = ("name", "run")

    def __init__(self, name, run):
        self.name = name
        self.run = run


def _run_luma_beat(field, uphet):
    """The colour-under's beat in the luma, taken out where the DECODED
    chroma first exists - its amplitude is the saturation the beat scales
    with."""
    if not luma_beat_wanted(field):
        return False
    beat = luma_beat.correct(field, uphet, field.rf.options.luma_beat)
    field.chroma_under_tbc = None
    debug_plot = getattr(field.rf, "debug_plot", None)
    if debug_plot and debug_plot.is_plot_requested("luma_noise"):
        # Kept for the plot only. Dropout detection, which draws it, runs
        # after this point.
        field.luma_beat = beat
    return True


FIELD_CHROMA_STAGES = (
    _FieldChromaStage("luma_beat", _run_luma_beat),
    _FieldChromaStage("cti", apply_chroma_transient_improvement),
)
"""In declared order. `cti` is LAST because it is a cosmetic sharpener:
anything measuring a physical property of the chroma must read the chroma
before it."""


def _stage_enabled(field, name):
    """Whether `--stages` has turned this stage off by name.

    The stage's own flag still gates it as before - this only answers the
    selection, so there is one gate per stage and not two."""
    selection = getattr(field.rf.options, "stage_selection", None)
    if not selection:
        return True
    return selection.get("stages", {}).get(name, True)


def decode_chroma(field, do_chroma_deemphasis=False):
    if field.rf.options.write_chroma:
        """Do track detection if needed and upconvert the chroma signal"""
        field.chroma_tbc_buffer = None

        uphet = process_chroma(
            field,
            disable_comb=field.rf.options.disable_comb,
            disable_tracking_cafc=False,
            do_chroma_deemphasis=do_chroma_deemphasis,
        )
        field.uphet_temp = uphet

        # THE FIELD-LEVEL CHROMA STAGES, run as a declared list rather than
        # as two hand-placed calls. `luma_beat` is a stage like any other -
        # gated by name, ordered by the declaration, and able to be turned
        # off with `--stages -luma_beat` exactly as the rest are - which is
        # what "folded in as a normal stage" means here. Keeping the list in
        # one place rather than moving the calls up to `field.py` also keeps
        # it out of the seven format subclasses that each call this.
        #
        # The order is the declaration's and is not free: `cti` declares
        # `must_follow = [luma_beat, decode_chroma]`, because the sharpener
        # reshapes amplitude at exactly the transitions the beat correction
        # takes its saturation across.
        for stage in FIELD_CHROMA_STAGES:
            if not _stage_enabled(field, stage.name):
                continue
            stage.run(field, uphet)

        # Release to avoid keeping this im memory - should do this in a cleaner manner.
        field.chroma_tbc_buffer = None
        return chroma_to_u16(uphet)

    return None


def get_burst_area(field):
    burst_start = math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])) - 4
    burst_end = math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])) + 8

    # burst length must be multiple of 4
    burst_end = burst_end - ((burst_end - burst_start) % 4)

    return burst_start, burst_end
