"""Transient artifact of the luma path, corrected where it is created.

The path's frequency response does nothing to steady content - the demodulator
is a limiter, and an amplitude change on a carrier that is not sweeping is
discarded before it can reach the picture. Everything the response does, it does
at TRANSITIONS, by weighting a sweeping carrier's sidebands asymmetrically.

That is why the RF-domain equalizer this replaces could not win. A frequency
domain multiply applies one magnitude to both sweep directions, while the
conversion it induces follows the SIGN of the sweep, so the same filter helps
one polarity and harms the other. Measured on four tapes in the 0.9-1.8 MHz
ring band, `--luma_eq` at 0.75 removed 4.0-7.3% of the ring from active picture
and INJECTED 13.4-20.7% at the sync fall, about three to one; on countdown the
two signs are symmetric, -0.75 giving -15.95% at the fall against +7.47% in
active. No scalar amount fixes a sign that differs by polarity.

What replaces it
----------------
The artifact is synthesised rather than fitted: a constant amplitude carrier on
a known trajectory, twice, once through the head's own measured response and
once through a flat one, everything else identical. The difference is what the
response does to a transition, and it is indexed by where the transition LANDS
and by its POLARITY, because the channel is linear only at a fixed operating
point and the operating point moves with the signal.

Measured over that family, the whole bank collapses. A single value decomposition
puts 88-97% of it in ONE waveform per polarity, scaled by a straight line in
landing level (R^2 0.88-0.98) - and that line crosses zero, near blanking for
rises and up near white for falls, on every tape and both heads. So the
correction is

    a * (level - L0) * w(t)

two waveforms and four numbers per head, not a bank of kernels. The waveform is
held as its cosine coefficients, which is where the compactness lives: 64 of
them carry it.

Where it is applied
-------------------
Per RF BLOCK, inside `_demodulate_to_video`, immediately after `unwrap_hilbert`
and before anything else touches the signal - so BEFORE the time base
correction, not on the corrected field. The transitions are therefore detected
on the RAW demodulated luma, the frequency the demodulator produced, before
de-emphasis.

That domain is the point. `sub_deemphasis` is nonlinear and is enabled for VHS
LP and EP and for all PAL, so a model derived before it does not compose with a
correction applied after it. Detecting on the raw channel keeps the model where
it is clean, and the waveform is synthesised once at build time so nothing has
to be filtered at runtime.
"""

import numpy as np
import numpy.fft as npfft
import scipy.signal as sps
from lddecode.utils import unwrap_hilbert

from vhsdecode import luma_amplitude


# Landing levels the artifact is synthesised at, in IRE, and the step it is
# synthesised with. One bin per 16 IRE across the format's own range spans it
# with enough points to fit a line through and few enough to synthesise quickly;
# the step is the format's sync-to-blanking excursion, which is the one
# transition size the signal is guaranteed to contain.
LANDING_STEP_IRE = 16.0
SYNTHESIS_SPAN_US = 5.1

# The waveform is held as a cosine series and truncated - a plain DCT-II
# low pass, which is the ordinary way to compact a smooth waveform. Where to cut
# needs no choosing: the waveform is a video signal and the format states its
# own bandwidth, so the series is kept up to `video_lpf_freq` and dropped above
# it, where the chain carries nothing anyway.

# Nothing here sets a threshold on a transition's size. Every same-sign run of
# the level is an event and each is weighed against the events before it, so
# what a transition has to clear is the shadow of its neighbours rather than a
# number.


def _synthesise(rf, response, start_ire, end_ire, lpf, span):
    """The artifact of one transition, per unit signed step, raw and filtered.

    Two arms differing in nothing but the head's response, so the difference is
    the response's doing and not the chain's. Returned twice: in the RAW
    demodulated frequency, which is what the detector sees, and through the
    de-emphasis and video low pass, which is what the output carries.
    """
    length = rf.blocklen
    video = np.full(length, float(start_ire))
    video[length // 2: 3 * length // 4] = float(end_ire)
    # A block that ended on a different level from the one it starts on puts a
    # step at the wrap, and that step's own artifact is larger than the one
    # being measured; returning three quarters along keeps the block continuous.
    mean = video.mean()
    video = npfft.irfft(npfft.rfft(video - mean) * lpf, n=length) + mean

    deemp = rf.Filters["FDeemp"]
    pre = npfft.irfft(npfft.rfft(video - mean) / deemp, n=length) + mean
    carrier = np.cos(
        2.0 * np.pi * np.cumsum(rf.iretohz(pre, spec=True)) / rf.freq_hz
    )
    spectrum = npfft.fft(carrier) * rf.Filters["RFVideo"]

    # The RAW demodulated frequency, which is what `demodblock` holds at the
    # point the correction goes in: before the spike replacement, the video
    # equalizer, the chroma trap and every de-emphasis stage. Correcting there
    # needs a model built there, and it is also the only domain in which the
    # model is unconditionally valid - `sub_deemphasis` is nonlinear and is
    # enabled for VHS LP, EP and all PAL, so anything downstream of it does not
    # compose with a model derived upstream.
    raw = []
    for arm in (np.ones_like(response), response):
        demod = np.asarray(
            unwrap_hilbert(npfft.ifft(spectrum * arm * rf.Filters["hilbert"]), rf.freq_hz),
            dtype=np.float64,
        )
        raw.append(rf.hztoire(demod, spec=True))
    shaped = raw
    step = float(end_ire - start_ire)
    window = slice(length // 2, length // 2 + span)
    # The channel's own 10-90 edge width, measured on the arm that had no
    # response applied to it. The detector needs a settling span and this is it,
    # derived from what the chain actually passes rather than chosen.
    flat = shaped[0]
    low, high = flat[length // 2 - 400], flat[length // 2 + 400]
    profile = (flat[length // 2 - 200: length // 2 + 200] - low) / (high - low)
    rising = np.flatnonzero(profile > 0.1)
    settled = np.flatnonzero(profile > 0.9)
    width = int(settled[0] - rising[0]) if len(rising) and len(settled) else span // 17
    return ((raw[1] - raw[0])[window] / step,
            (shaped[1] - shaped[0])[window] / step,
            max(width, 2))


def _family(rf, response, lpf, polarity, span):
    """The artifact at every landing the format can hold a full step at."""
    sys_params = rf.SysParams
    floor, ceiling = float(sys_params["vsync_ire"]), 100.0
    # The sync-to-blanking excursion: the one transition size every field is
    # guaranteed to contain, so the model is synthesised at a step the signal
    # actually makes rather than at a chosen one.
    step = -float(sys_params["vsync_ire"])
    levels, raws, shapeds, widths = [], [], [], []
    land = floor + LANDING_STEP_IRE
    while land <= ceiling + LANDING_STEP_IRE:
        start = land - step if polarity > 0 else land + step
        if floor <= start <= ceiling:
            raw, shaped, width = _synthesise(rf, response, start, land, lpf, span)
            levels.append(land)
            raws.append(raw)
            shapeds.append(shaped)
            widths.append(width)
        land += LANDING_STEP_IRE
    return (np.array(levels), np.array(raws), np.array(shapeds),
            int(np.median(widths)) if widths else 2)


def _collapse(levels, rows, cutoff=None):
    """One waveform and a straight line in level, from the family.

    The bank is rank one to within a few per cent, so the leading singular pair
    IS the model: its right vector is the waveform every landing shares and its
    left vector is how that waveform scales with where the transition lands.

    The waveform is then held as a truncated cosine series. `cutoff` is the
    index the format's own video bandwidth falls at, so what is dropped is what
    the chain could not have carried.
    """
    left, values, right = np.linalg.svd(rows, full_matrices=False)
    waveform, amplitude = right[0], left[:, 0] * values[0]
    if amplitude[-1] < 0.0:
        waveform, amplitude = -waveform, -amplitude
    slope, intercept = np.polyfit(levels, amplitude, 1)
    from scipy.fft import dct, idct

    coefficients = dct(waveform, type=2, norm="ortho")
    if cutoff is not None and 0 < cutoff < len(coefficients):
        coefficients = coefficients.copy()
        coefficients[cutoff:] = 0.0
    shape = idct(coefficients, type=2, norm="ortho")

    # Faded to zero over the tail. The artifact has decayed to 0.3-1.2% of its
    # peak by the end of the synthesis span and only about 1% of its energy
    # lies past it, so the truncation seam is small - but every copy that is
    # subtracted would otherwise end on that step, and the seams accumulate
    # where transitions are dense. A raised cosine over the last tenth removes
    # it for nothing.
    tail = max(len(shape) // 10, 2)
    shape[-tail:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(tail) / tail))
    return shape, float(slope), float(intercept)


def _events(level_ire, settle, release):
    """Every transition, weighed rather than selected.

    There is no entry threshold and nothing is discarded. The derivative is cut
    into same-sign runs, so everything is an event, and each one's magnitude is
    the level it actually MOVED across its run - noise and texture have plenty
    of derivative but almost no net displacement, so they arrive weighing
    nothing without ever having to be rejected.

    What decides an event's weight is the events before it. A running envelope
    holds each accepted event's size and releases over the settling span, so a
    transition's own ring half-cycles - which are real same-sign runs arriving
    just after it - sit under its shadow and weigh nothing. That is what stops a
    single edge being corrected fifteen times over.

    Selecting instead of weighing is what a TRAINING pool wants, where quiet
    flanks buy a clean population at the cost of most of it. Applied to
    delivery it corrects almost nothing: measured, an isolation test admitted
    445 transitions across thirty fields where this admits every one of them at
    its own weight. Overlapping corrections sum the way the overlapping
    artifacts do, so neighbours need no special handling.
    """
    if len(level_ire) < 4 * settle:
        return []
    change = np.diff(level_ire)
    # The signal's own noise, robustly: half the median absolute step between
    # neighbouring samples. An event has to clear this before it is a
    # transition at all - without it every one-sample wiggle of the noise is a
    # same-sign run with nothing standing in front of it, and measured, that
    # painted 1503 IRE of "correction" over more than half the field.
    sigma = float(np.median(np.abs(change))) * 1.4826
    sign = np.signbit(change)
    edges = np.flatnonzero(np.diff(sign)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(change)]))
    # net displacement across each run, which is what a transition is
    carried = np.concatenate(([0.0], np.cumsum(change)))
    moved = carried[stops] - carried[starts]

    found = []
    envelope = 0.0
    previous = 0
    release = float(max(release, settle))
    for begin, finish, step in zip(starts, stops, moved):
        envelope *= np.exp(-(begin - previous) / release)
        previous = begin
        size = abs(step)
        if size <= envelope + sigma:
            continue
        # A soft ramp out of both shadows - the standing envelope of what came
        # before, and the noise the level carries anyway. An event just clear of
        # them contributes a little, one far clear contributes fully; nothing is
        # rejected by a threshold, it simply weighs nothing.
        weight = min((size - envelope - sigma) / size, 1.0)
        envelope = size
        # The step the MODEL means is the settled level difference across the
        # transition, not the run's own displacement. On the raw demodulated
        # luma those differ by the record pre-emphasis, whose overshoot runs
        # three to nine times the step it belongs to - scaling the correction by
        # the overshoot instead of the step is an order-of-magnitude error, and
        # it is measured: it painted 1500 IRE where the artifact is worth ten.
        if begin < settle or finish + 2 * settle >= len(level_ire):
            continue
        before = float(np.median(level_ire[begin - settle: begin]))
        after = float(np.median(level_ire[finish + settle: finish + 2 * settle]))
        settled = after - before
        if settled == 0.0 or np.sign(settled) != np.sign(step):
            continue
        found.append((int(finish), settled, after, float(weight)))
    return found


def _noise_floor(rf):
    """The chain's own log-amplitude noise, pooled across the heads.

    Already measured: `luma_amplitude` fits it while it accumulates the
    response, as the scatter of each field about the model built without it.
    Read here only as a settling test - it is zero until the accumulation has
    enough to compare a field against.
    """
    probe = rf.__dict__.get("_luma_eq_reliability")
    if not probe:
        return 0.0
    mass = sum(entry.get("weight", 0.0) for entry in probe.values())
    return (sum(entry.get("resid", 0.0) for entry in probe.values()) / mass
            if mass > 0.0 else 0.0)


def build(rf):
    """The decoder's transient model, synthesised once and then frozen.

    Pooled across the heads, and built exactly once. `demodblock` runs in its
    own thread and ahead of field assembly, so it cannot know which head wrote
    the block it is holding, and a model that kept changing would be read at a
    different version by every block - which is precisely the defect that made
    `--luma_eq` irreproducible at one thread. Synthesised once and never
    touched again, there is nothing left to race on after the warm-up.

    Returns None until the carrier's own amplitude has been measured over
    enough of the decode to describe the response, which is the honest answer
    for the opening fields rather than a model built on nothing.
    """
    if "_luma_transient_model" in rf.__dict__:
        return rf.__dict__["_luma_transient_model"]
    dense = luma_amplitude._pooled_dense(rf)
    line = rf.__dict__.get("_luma_transient_line")
    if dense is None or line is None:
        return None
    # Not until the response has settled. The reliability statistic only starts
    # accumulating once a field's own bins overlap the model built without it,
    # so a nonzero noise floor is the signal that the accumulation has something
    # to describe. The model is built once and then frozen, so building it early
    # would freeze whatever the opening fields happened to say - measured, the
    # fitted zero crossings move by several IRE between the two.
    if _noise_floor(rf) <= 0.0:
        return None

    length = rf.blocklen
    lpf = np.abs(
        sps.sosfreqz(
            sps.butter(rf.DecoderParams["video_lpf_order"],
                       rf.DecoderParams["video_lpf_freq"] / (rf.freq_hz / 2),
                       "lowpass", output="sos"),
            length, whole=True)[1][: length // 2 + 1]
    )
    span = int(SYNTHESIS_SPAN_US * 1e-6 * rf.freq_hz)

    # Each described frequency is a mean of however many samples the carrier
    # left there, so its own standard error is the chain's noise floor over that
    # population. Frequencies the picture dwells at are determined; ones it only
    # passes through are not, and the difference is what decides which cosines
    # of the waveform survive.
    # Where the cosine series is cut: DCT index k spans k * rate / (2 * span),
    # so the format's own video bandwidth fixes the index directly.
    cutoff = int(2 * span * rf.DecoderParams["video_lpf_freq"] / rf.freq_hz) + 1
    response = luma_amplitude.luma_path_equalizer(rf, line, -1.0, dense)

    built = {}
    for polarity in (1, -1):
        levels, raws, shapeds, width = _family(rf, response, lpf, polarity, span)
        if len(levels) < 4:
            return None
        shape, slope, intercept = _collapse(levels, raws, cutoff)
        built[polarity] = {
            "waveform": shape,
            "slope": slope,
            "zero": -intercept / slope if slope else 0.0,
            "certified": (float(levels.min()), float(levels.max())),
            "edge_samples": width,
            "components": cutoff,
        }
    rf.__dict__["_luma_transient_model"] = built
    return built


def observe(field):
    """Keep the line the model is synthesised from, from this field's own fit.

    Called where the head is known and the measurement has been taken. The
    accumulated dense response is `luma_amplitude`'s already; this only holds
    the straight line it is stated against, which one field determines to
    within a per cent of its slope.
    """
    measured = luma_amplitude.measured_amplitude_deviation(field)
    if measured is None or measured.response is None:
        return
    field.rf.__dict__.setdefault("_luma_transient_line",
                                 tuple(measured.response.separation))


def correct(rf, demod, amount):
    """Subtract the modelled transient artifact from the demodulated luma.

    In place, on the raw instantaneous frequency, before anything else has
    shaped it. Returns how many transitions were corrected.
    """
    model = build(rf)
    if model is None or amount == 0:
        return 0
    hz_ire = rf.iretohz(1.0, spec=True) - rf.iretohz(0.0, spec=True)
    level = (np.asarray(demod, dtype=np.float64) - rf.iretohz(0.0, spec=True)) / hz_ire
    settle = max(model[1]["edge_samples"], 2)
    release = len(model[1]["waveform"])

    applied = 0
    for start, step, landing, weight in _events(level, settle, release):
        entry = model[1 if step > 0 else -1]
        low, high = entry["certified"]
        if not low <= landing <= high:
            # Past the span the model was synthesised over, its own level law
            # reverses sign, and that is where the response measurement is
            # thinnest. Held rather than extrapolated.
            continue
        shape = entry["waveform"]
        end = min(start + len(shape), len(demod))
        if end <= start:
            continue
        scale = amount * weight * entry["slope"] * (landing - entry["zero"]) * step
        demod[start:end] -= (scale * hz_ire) * shape[: end - start]
        applied += 1
    return applied
