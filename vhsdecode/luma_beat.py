"""Color-under beat in the demodulated luma, cancelled against the chroma itself.

The luma FM carrier and the color-under are written by the same head at the same
instant, and the color-under amplitude modulates the carrier. That is measurable
directly: the luma envelope carries a component at the color-under frequency,
which needs sidebands at `carrier +- color_under` inside the luma passband and so
cannot be the color-under leaking in additively. Measured against a recording
with nothing in the chroma band at all, the coherence between the color-under
channel and the luma envelope goes from 0.07 to 0.55-0.64 as soon as the tape
carries chroma.

An amplitude modulation on the carrier is not supposed to survive an FM
demodulator, and most of it does not. What reaches the picture lands in the
demodulated luma at the color-under frequency, where it is signal energy that
has nothing to do with the luma's own image. Nothing else in the chain removes
it: the subcarrier trap is off by default and is tuned to fsc, which on a
color-under format is not where this is.

It arrives as FREQUENCY rather than as phase, and that is measured rather than
assumed. Phase interference would be differentiated by the demodulator, so its
transfer would have to rise across the color-under band; fitting transfers shaped
as `f` and as `f^2` against the flat one moved nothing outside a per cent either
way (74.1 / 73.1 / 71.4 on 75% bars, 88.9 / 89.6 / 89.9 on chroma noise). A flat
transfer is what the signal wants, and it is what is applied.

The interference ratio's other half - that the beat should go inversely with the
carrier amplitude it competes against - was built and measured and is NOT here.
On the time base corrected picture, where this correction is applied, the carrier
amplitude varies by only about 3% rms across a field once the response to the
carrier's own frequency is averaged out, and that is far too little to show
against the model's own residual. Dividing by it changed 73.7% removed to 73.8%,
and dividing by it smoothed any shorter than a line drives the correction hardest
through sync, where there is no beat at all.

What the correction is
----------------------
The color-under is QAM, so its carrier's amplitude IS the picture's saturation,
and the beat follows it - not proportionally. Fitted separately inside each bar
of a 75% colour bar field the transfer runs monotonically with saturation,
0.00904 at the least saturated bar against 0.02167 at the most. Yellow settles
that it is saturation rather than level: it is the brightest of the six, and low
in both. So the model is

    beat(t) = Re{ h * c(t) } * saturation(t) ** SATURATION_EXPONENT

with `c` the color-under the tape held and `h` two numbers for the whole decode.
The transfer is a property of the deck rather than of the picture, and within a
decode it holds to a few per cent and a few degrees from field to field, which is
what makes it safe to freeze.

Where it is allowed to run
--------------------------
Only on fields the decoder itself says are carrying color-under. The correction
subtracts whatever part of the luma correlates with the chroma channel, and on a
monochrome recording that channel is not carrying color-under at all - it is
carrying luma that leaked through the chroma band pass. Those are the same
correlation and no statistic on the field separates them, so the question has to
be answered somewhere else, and the decoder already answers it: the color
killer's own burst measurement. `chroma.luma_beat_wanted` holds that test, and
nothing here measures a burst of its own.

Where the transfer is fitted
----------------------------
After the chroma is processed, because that is the only place the decoded
saturation exists. The RF envelope is not saturation - it is saturation times
whatever the head was doing - and weighting by it was measured to make the
correction worse, 70.2% of the beat removed against 68.6%.

Measured on the decoded picture it takes out 74% of the beat on 75% bars and 83%
on a chroma noise pattern. The amount is at its own optimum already - half of it
and one and a half times it both remove less - so what is left is the single
gain's shape error rather than anything an amount can reach. On material with no
color-under at all it moves the picture by 0.04 IRE, twenty times under the line
noise floor - and with the burst gate above it does not run there at all.

Every field is corrected, the first one included - a single field already fixes
the transfer to within a few per cent, so the estimate is applied while it is
still being refined rather than after. Measured field by field, the opening field
has 71% of its beat removed and nothing steps later on.
"""

import numpy as np
import scipy.fft as fft
from scipy.ndimage import uniform_filter1d


# Fields averaged into the transfer before it is frozen.
#
# What averaging can remove is the field to field scatter, a few per cent in
# magnitude and a few degrees in phase, and it removes it as the square root of
# the count. What it cannot remove is the slow drift the transfer shows along a
# tape - about ten degrees over fourteen fields on one of the patterns measured.
# Eight fields brings the scatter below that drift, and past there averaging buys
# back nothing the drift does not take away again.
#
# It costs nothing at the start of a decode, because the running estimate is
# applied from the first field rather than waited for. This is only where it
# stops moving.
ACCUMULATED_FIELDS = 8


# How the beat grows with the color-under's amplitude.
#
# One means the beat goes as the square of saturation, since the reference it
# multiplies is already proportional to it. Measured against the beat left in
# the picture, on the pattern whose saturation varies: 67.7% of it removed with
# no exponent, 72.6% at a half, 74.1% at one. On a chroma noise pattern the
# exponent changes nothing at all, which is the check rather than a
# disappointment - that pattern sits at a single saturation across the whole
# field, so there is nothing for an exponent to act on.
SATURATION_EXPONENT = 1.0



def _saturation(field, uphet):
    """The decoded chroma's own amplitude, sample by sample.

    The output grid runs at four times the subcarrier, so the up-converted
    chroma lands on it as I, Q, -I, -Q repeating and any two neighbouring
    samples ARE the quadrature pair. Their root sum of squares is the amplitude
    exactly - no filter, no transform.

    Smoothed over the shortest interval the color-under's modulation band can
    resolve; anything shorter describes the carrier rather than what it carries.
    """
    values = np.asarray(uphet, dtype=np.float64)
    amplitude = np.empty_like(values)
    np.hypot(values[:-1], values[1:], out=amplitude[:-1])
    amplitude[-1] = amplitude[-2] if len(amplitude) > 1 else 0.0

    decoder = field.rf.DecoderParams
    half_hz = decoder["chroma_bpf_upper"] - decoder["color_under_carrier"]
    span = max(int(field.rf.SysParams["fsc_mhz"] * 4e6 / (2.0 * half_hz)), 1)
    if span > 1:
        uniform_filter1d(amplitude, span, mode="nearest", output=amplitude)
    return amplitude



def _rf_response(rf):
    """The pre-demodulator response as the demodulator sees it.

    Falls back to the stored filter if the channel equalizer is not present, so
    this module does not depend on that stage existing.
    """
    try:
        from vhsdecode import channel_eq
    except ImportError:
        return rf.Filters["RFVideo"]
    return channel_eq.effective_rf_response(rf)


def _ramp_shaping(rf, frequencies_hz):
    """How the RF path's own tilt shapes the beat across the color-under band.

    The beat does not ride on the color-under itself - it rides on the sidebands
    the color-under puts either side of the luma carrier, at
    `carrier +- color_under`. What turns their amplitude modulation into
    something an FM demodulator reads is the IMBALANCE between the two, and the
    linear ramp creates that imbalance by construction: it is a tilt, and
    `start_rf_linear` is the color-under carrier itself, so the lower sideband
    falls down the ramp while the upper one does not.

    Measured on the decoder's own RFVideo magnitude at blanking, the imbalance
    runs from 0.14 of its band-centre value at the bottom of the color-under band
    to 1.34 at the top - a factor of nine and a half that a flat correction
    cannot follow.

    Taken from the decoder's own filters rather than rebuilt, so it stays correct
    for every format and every combination of ramp, peaking and notch options, and
    costs no fitted parameter. Read through `channel_eq.effective_rf_response`,
    which is `RFVideo` alone until the channel equalizer is active and `RFVideo`
    times its table once it is: the beat is carried by the demodulated luma, so
    the path this must model is the one the DEMODULATOR saw, not the one the
    envelope saw - those two stopped being the same response when that stage
    landed. Evaluated at blanking because that is where the burst sits, and the
    burst is what the color carrier is phase locked to.

    Normalised at the carrier, so the transfer keeps its own scale.
    """
    response = np.abs(np.asarray(_rf_response(rf), dtype=np.float64))
    half = len(response) // 2
    response = response[:half]
    step_hz = rf.freq_hz / (2 * half)

    def at(hz):
        index = np.clip((np.asarray(hz) / step_hz).astype(int), 0, half - 1)
        return response[index]

    operating_hz = rf.DecoderParams["ire0"]
    imbalance = at(operating_hz + frequencies_hz) - at(operating_hz - frequencies_hz)
    middle = at(operating_hz + rf.DecoderParams["color_under_carrier"]) - at(
        operating_hz - rf.DecoderParams["color_under_carrier"]
    )
    if not abs(middle) > 0.0:
        return None
    return imbalance / middle


def _reference(field, under, saturation):
    """The color-under, band limited, split into quadrature and weighted.

    Two real arrays rather than one complex one: `Re{h * c} * w` is
    `Re(h) * w * c_real - Im(h) * w * c_imag`, so the pair carries everything an
    analytic signal would and nothing has to allocate complex.

    The color-under here is the one the TAPE held, taken before the
    up-conversion rewrites its phase, because that is the phase the beat has.

    Band limited twice, once before the weighting and once after. Multiplying by
    saturation spreads the band by the width of saturation's own modulation, and
    what leaves the band is no longer orthogonal to the picture - so the fit
    would reach out and take hold of luma. The correction belongs inside the band
    the color-under occupies and is held there.

    Saturation goes in unnormalised: it only has to mean the same thing every
    field, and whatever scale it carries is absorbed into the transfer.

    The band is shaped by what the RF path does to the sidebands the beat
    actually rode in on - see `_ramp_shaping`. That shaping is computed from the
    decoder's own filter, not fitted, so it adds nothing to estimate.
    """
    rate_hz = field.rf.SysParams["fsc_mhz"] * 4e6
    carrier_hz = field.rf.DecoderParams["color_under_carrier"]
    half_hz = field.rf.DecoderParams["chroma_bpf_upper"] - carrier_hz

    # A field is `outlinelen * outlinecount` samples, and on NTSC that is 263
    # lines - a prime, which is the worst length a transform can be handed.
    # Padding to the next fast length costs a tenth of a per cent of the field
    # and takes each transform from 6.6 ms to 1.6 ms. What the padding rings
    # against is a step at the very end of the field, and the band's own impulse
    # response is about a dozen samples long, so it reaches nothing that is kept.
    length = len(under)
    padded = fft.next_fast_len(length)
    per_bin_hz = rate_hz / padded
    low = max(int(np.ceil((carrier_hz - half_hz) / per_bin_hz)), 1)
    high = min(int((carrier_hz + half_hz) / per_bin_hz) + 1, padded // 2 + 1)
    if high <= low:
        return None, None

    def to_band(values):
        spectrum = fft.rfft(values, n=padded)
        spectrum[:low] = 0.0
        spectrum[high:] = 0.0
        return spectrum

    def from_band(spectrum):
        return fft.irfft(spectrum, n=padded)[:length]

    spectrum = to_band(np.asarray(under, dtype=np.float64))

    # Once, on the color-under itself. The second pass through `to_band` below
    # only holds the weighted result inside the band; shaping there would apply
    # the path's tilt twice.
    shaping = _ramp_shaping(
        field.rf, np.arange(low, high, dtype=np.float64) * per_bin_hz
    )
    if shaping is not None:
        spectrum[low:high] *= shaping

    weight = saturation ** SATURATION_EXPONENT
    in_phase = from_band(spectrum) * weight
    quadrature = from_band(spectrum * -1j) * weight
    return from_band(to_band(in_phase)), from_band(to_band(quadrature))


def _units_per_hz(field):
    """Output code values per hertz of demodulated frequency."""
    return field.out_scale / field.rf.SysParams["hz_ire"]


def correct(field, uphet, amount):
    """Take the beat out of the time base corrected luma, against the chroma.

    Called where the processed chroma exists, which is the only place the decoded
    saturation does and where both planes already sit on one grid. The picture is
    in output units by then, so the correction is converted into them and clipped
    back into range.

    Returns what it subtracted, in IRE and on the picture's own grid, or None if
    it did nothing. The caller carries that where the debug plot asked for it;
    nothing else reads it.
    """
    picture = getattr(field, "dspicture", None)
    under = getattr(field, "chroma_under_tbc", None)
    if not amount or picture is None or under is None or uphet is None:
        return None

    length = min(len(picture), len(under), len(uphet))
    if length < 2:
        return None

    saturation = _saturation(field, uphet[:length])
    in_phase, quadrature = _reference(
        field, np.asarray(under[:length], dtype=np.float64), saturation
    )
    if in_phase is None:
        return None

    transfer = _transfer(field, picture, in_phase, quadrature)
    if transfer is None:
        return None

    transfer *= amount
    beat = transfer.real * in_phase - transfer.imag * quadrature

    corrected = picture[:length].astype(np.float64) - beat * _units_per_hz(field)
    np.clip(corrected, 0.0, 65535.0, out=corrected)
    picture[:length] = corrected.astype(picture.dtype)
    return beat / field.rf.SysParams["hz_ire"]


def _transfer(field, picture, in_phase, quadrature):
    """The transfer as everything seen so far describes it.

    Every field is corrected, the first one included. A single field already
    fixes the transfer to within a few per cent, so there is nothing to gain by
    waiting and something to lose: a model that arrives part way through a decode
    steps the picture where it lands. The running estimate is applied instead and
    tightens as fields go by.

    Nothing here fits against its own output. This runs before the field it
    measured is corrected, so every field contributes the beat it actually had.

    Frozen once enough fields have averaged in, so the rest of the decode is not
    corrected by an estimate that keeps moving under it.
    """
    rf = field.rf
    frozen = rf.__dict__.get("_luma_beat_model")
    if frozen is not None:
        return frozen

    luma = picture[: len(in_phase)].astype(np.float64) / _units_per_hz(field)
    luma -= luma.mean()

    store = rf.__dict__.setdefault(
        "_luma_beat_sums", {"cross": 0.0 + 0.0j, "power": 0.0, "fields": 0}
    )
    store["cross"] += np.dot(in_phase, luma) - 1j * np.dot(quadrature, luma)
    store["power"] += float(np.dot(in_phase, in_phase) + np.dot(quadrature, quadrature))
    store["fields"] += 1
    if store["power"] <= 0.0:
        return None

    # The least squares coefficient of `Re(h * c)` against a real signal is twice
    # the ordinary projection: half of its energy sits in the conjugate term,
    # which contributes nothing to the correlation because a narrowband signal
    # has no second harmonic for it to correlate with.
    transfer = 2.0 * store["cross"] / store["power"]
    if store["fields"] >= ACCUMULATED_FIELDS:
        rf.__dict__["_luma_beat_model"] = transfer
    return transfer


def measured_response(rf):
    """What this correction measured, for a consumer in another stage.

    Read only. Nothing here is used by the correction itself, and calling it
    cannot change a decode - it reports the state the fit has already reached.

    The color-under's footprint on the luma is one of the few places in the
    decoder where a path property is measured against a phase locked reference,
    so the numbers are worth exporting even though this stage does not need them
    exported. Returns None before the fit has anything to say.

    `transfer` is the complex coefficient of `Re{h . c}` against the demodulated
    luma at the color-under carrier, in luma frequency units per unit of decoded
    chroma. Its MAGNITUDE is well determined - within a few per cent from a
    single field. Its ARGUMENT is well determined within a decode, and stable to
    a tenth of a degree once frozen, but it is NOT established as a property of
    the path: measured across captures from one deck and tape family it spans
    tens of degrees, including twenty between two captures whose fits are both
    well conditioned. A consumer wanting a channel phase should treat that spread
    as the uncertainty until it is explained, rather than reading `transfer`'s
    angle as the path's.

    `fields` says how much has averaged in, and whether the estimate is still
    moving - it stops at `ACCUMULATED_FIELDS`.
    """
    transfer = rf.__dict__.get("_luma_beat_model")
    sums = rf.__dict__.get("_luma_beat_sums")
    if transfer is None and sums is not None and sums["power"] > 0.0:
        # Not yet frozen, so report the running estimate on the same terms the
        # correction is currently using rather than nothing at all.
        transfer = 2.0 * sums["cross"] / sums["power"]
    if transfer is None:
        return None
    return {
        "transfer": complex(transfer),
        "fields": int(sums["fields"]) if sums else 0,
        "frozen": rf.__dict__.get("_luma_beat_model") is not None,
        "carrier_hz": float(rf.DecoderParams["color_under_carrier"]),
    }


def path_shaping(rf, frequencies_hz):
    """The RF path's own tilt across the color-under band, for a consumer.

    Read only, and the same curve `_reference` applies - exported rather than
    recomputed so a consumer cannot drift from what the correction actually did.
    See `_ramp_shaping` for what it is and why it is taken from the decoder's
    own filter rather than rebuilt.

    Real valued today. It is the seam where a MEASURED complex low frequency
    response would enter this stage: the same curve carrying phase would make
    the correction phase aware without changing anything else about it.
    """
    return _ramp_shaping(rf, np.asarray(frequencies_hz, dtype=np.float64))


# ---------------------------------------------------------------------------
# The same footprint, at the RF site.
#
# This stage removes the color-under's mark on the picture by subtracting it
# from the DEMODULATED luma, which is where it is cheapest: FM demodulation of a
# carrier plus a small interferer is linear to first order, so a subtraction
# after the demodulator is equivalent to one before it and needs no carrier to
# be tracked. That is why the correction lives where it does and it is not
# moving.
#
# What follows is for a different purpose, not a better version of that one. The
# channel identification loop needs the luma's own residual near 1.4 MHz
# separated from the color-under's footprint, and that entanglement is physical
# and lives in the RF - so the separation has to happen there. These two
# functions supply the model; the RF site and the subtraction belong to the
# stage that calls them.
#
# The construction, and the measurement behind it: the color-under amplitude
# modulates the luma carrier, so its footprint is the PRODUCT of the carrier and
# the color-under, which puts the synthesized energy at `carrier +- f_cu`
# automatically - including as the carrier moves with the picture, which is what
# made a re-synthesis from a tracked carrier frequency unattractive. Measured on
# raw RF blocks, the magnitude squared coherence between the luma envelope and
# the color-under taken from the same RF runs 0.46-0.57 in band against a
# 0.05-0.06 shoulder floor, so the footprint is coherent and identifiable per
# block.
#
# The scale is COMPLEX, and that is not decoration. Fitted as a real scalar with
# the alignment left to a delay, the same measurement reads 0.009 - nothing at
# all - and scanning the delay recovers only a part of it while returning a best
# lag that disagrees between blocks. The footprint has a phase against the
# color-under; a complex scale carries it, and a delay cannot.


def _color_under(rf, data, length):
    """The color-under as the tape holds it, from the raw RF block.

    Taken from the block's own samples rather than from the chroma path's
    output, which has a deliberate time shift applied to compensate the luma
    filter's delay and has had its DC removed. Deriving it here keeps the
    footprint's alignment a measured quantity - it lands in the fitted scale's
    phase - instead of an agreement between two stages about a shift.
    """
    carrier_hz = rf.DecoderParams["color_under_carrier"]
    half_hz = rf.DecoderParams["chroma_bpf_upper"] - carrier_hz
    if not half_hz > 0.0:
        return None
    padded = fft.next_fast_len(length)
    per_bin_hz = rf.freq_hz / padded
    low = max(int(np.ceil((carrier_hz - half_hz) / per_bin_hz)), 1)
    high = min(int((carrier_hz + half_hz) / per_bin_hz) + 1, padded // 2 + 1)
    if high <= low:
        return None
    spectrum = fft.rfft(np.asarray(data[:length], dtype=np.float64), n=padded)
    spectrum[:low] = 0.0
    spectrum[high:] = 0.0
    # Both quadratures of the analytic signal, so the fitted scale can rotate it.
    return (
        fft.irfft(spectrum, n=padded)[:length],
        fft.irfft(spectrum * -1j, n=padded)[:length],
    )


def _rf_store(rf, head):
    """Per head, because the two heads are two different paths.

    A pooled scale hides a head alternating term rather than averaging it away:
    this stage's own baseband fit alternated by a factor of two between heads
    when it was estimated on one reference interval per line, and survived two
    changes of estimator before that was found.
    """
    return rf.__dict__.setdefault("_luma_beat_rf", {}).setdefault(
        head, {"cross": 0.0 + 0.0j, "power": 0.0, "fields": 0, "model": None}
    )


def fit_rf_footprint(rf, y_rf, data, head):
    """Accumulate the complex scale relating the color-under to its footprint.

    The witness is the luma carrier's own envelope. An amplitude modulation by
    the color-under writes `|y| * Re{m * c}` into it, so the envelope's in band
    part measures `m` directly, without demodulating anything or knowing where
    the carrier sits.

    Accumulated across fields and frozen once enough have averaged in, the same
    way the baseband transfer is, and for the same reason: an estimate that
    keeps moving under the signal it corrects steps the result where it lands.

    Fitted over the whole block, never on a per line reference interval.
    """
    length = min(len(y_rf), len(data))
    if length < 2:
        return None
    store = _rf_store(rf, head)
    if store["model"] is not None:
        return store["model"]

    reference = _color_under(rf, data, length)
    if reference is None:
        return None
    in_phase, quadrature = reference

    carrier = np.asarray(y_rf[:length], dtype=np.float64)
    # The carrier's envelope, from the same transform pair the rest of this
    # module uses: multiplying the spectrum by -1j is the Hilbert transform, and
    # the magnitude of the pair is the envelope.
    padded = fft.next_fast_len(length)
    quadrature_rf = fft.irfft(
        fft.rfft(carrier, n=padded) * -1j, n=padded
    )[:length]
    envelope = np.hypot(carrier, quadrature_rf)
    level = envelope.mean()
    if not level > 0.0:
        return None
    envelope -= level

    # Divided by the carrier's own level, which makes the scale a MODULATION
    # INDEX rather than an envelope amplitude. The distinction is not
    # cosmetic: the envelope measures `|y| * Re{m * c}`, so a scale fitted
    # against the envelope directly carries a factor of `|y|` inside it, and
    # multiplying by the carrier again when the footprint is built would apply
    # that factor twice. Measured, that overshoot does not look like a small
    # error - the residual stays perfectly correlated with the color-under and
    # in band coherence RISES, from 0.18 to 0.50, because too much has been
    # taken out rather than too little.
    store["cross"] += (
        np.dot(in_phase, envelope) - 1j * np.dot(quadrature, envelope)
    ) / level
    store["power"] += float(
        np.dot(in_phase, in_phase) + np.dot(quadrature, quadrature)
    )
    store["fields"] += 1
    if store["power"] <= 0.0:
        return None
    scale = 2.0 * store["cross"] / store["power"]
    if store["fields"] >= ACCUMULATED_FIELDS:
        store["model"] = scale
    return scale


def rf_footprint(rf, y_rf, data, head=None):
    """The color-under's footprint on the luma carrier, for the caller to subtract.

    `y_rf` is the band limited luma RF and `data` the raw block, both at the RF
    rate. Returns an array the length of the shorter of them, or None before the
    scale has anything to say.

    The envelope carries `|y| * Re{m * c}`, so the footprint riding on the
    carrier is `y * Re{m * c}` - the modulation multiplying the carrier it rode
    in on. No carrier frequency appears anywhere: multiplying by the carrier
    itself places the result at `carrier +- f_cu` wherever the carrier happens
    to be at that instant.

    What this can buy, so it is not oversold: the in band coherence averages
    0.20 across a block even though it peaks at 0.55, and a coherent subtraction
    can only take out the color-under correlated share. The rest of the
    envelope's in band energy is the luma's own content and stays.
    """
    scale = fit_rf_footprint(rf, y_rf, data, head)
    if scale is None:
        return None
    length = min(len(y_rf), len(data))
    reference = _color_under(rf, data, length)
    if reference is None:
        return None
    in_phase, quadrature = reference
    modulation = scale.real * in_phase - scale.imag * quadrature
    return np.asarray(y_rf[:length], dtype=np.float64) * modulation
