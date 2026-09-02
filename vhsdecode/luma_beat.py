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

    Taken from `Filters["RFVideo"]` rather than rebuilt, so it stays correct for
    every format and every combination of ramp, peaking and notch options, and
    costs no fitted parameter. Evaluated at blanking because that is where the
    burst sits, and the burst is what the color carrier is phase locked to.

    Normalised at the carrier, so the transfer keeps its own scale.
    """
    response = np.abs(np.asarray(rf.Filters["RFVideo"], dtype=np.float64))
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
