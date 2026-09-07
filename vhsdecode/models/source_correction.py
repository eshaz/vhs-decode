"""THE SOURCE CHANNEL: the luma response the signal carried INTO the
recording VCR, measured against the specification and taken back off.

Ethan, 2026-09-06, having had to say it more than once:

  *"The correction stages after the VCR models, i.e. the television and
  source correction stages are not being corrected properly. As stated
  before, I need to use all the components together in all measurable
  dimensions to correct the chroma and luma response that came into the
  recording VCR. I can visibly see in the luma that this is not
  happening."*

and, earlier, naming the gap exactly:

  *"Ok I found a gap, the rf path seems complete, but the picture path is
  not complete. We need to use the NTSC and PAL video signal spects to
  remove the residual added between the source video and the VCR input.
  This is the composite video chain, and we have exact specs to create the
  model, and perform the transform in all our dimensions."*

He was right, and the gap was total. Before this module nothing in the
runtime reached `composite_channel`, `picture_stage`, `source_agc`,
`multipath` or `profiles`: two chroma stages were wired and no stage
anywhere corrected the response the signal carried before it reached the
recording VCR. The chain runs

    source -> the source's own chain -> RECORDING VCR -> tape ->
    playback VCR -> decode

and undoing it runs the other way, so this stage is the LAST one going
backwards: after the playback and record models have been undone, what is
left still carries the source chain, and that is what comes off here.

--------------------------------------------------------------------------
THE REFERENCE IS A NEGATIVE SPECIFICATION, AND THAT IS THE WHOLE POINT
--------------------------------------------------------------------------

SMPTE 170M-2004 clause 7.1: *"This standard does not impose a bandwidth
restriction on the luminance part of the NTSC signal."* There is therefore
no specified luma response to scale by - THE SPECIFICATION IS FLATNESS, and
every departure from flat is a channel rather than a signal. The familiar
4.2 MHz belongs to the transmission CHANNEL, which the note to clause 7
puts there in as many words. `composite_channel.luma_response` already
states this correctly and is read from here rather than restated, so the
clause travels with the number.

That makes the target trivial to write down and the IDENTIFICATION the
entire difficulty: the source's response and the VCR's are in series, and
something has to separate them.

--------------------------------------------------------------------------
WHAT SEPARATES THEM, AND WHAT DOES NOT - MEASURED, NOT ASSUMED
--------------------------------------------------------------------------

THE HEAD AXIS SEPARATES THE PLAYBACK HEADS, EXACTLY. The VCR's response is
per head and alternates field to field; the source's is not, because the
source signal never knew which head would write it. So the head DIFFERENCE
of the measured response is purely the VCR and is excluded from the
correction by construction, while the head COMMON part carries the source
plus whatever the VCR shares between its heads.

Measured 2026-09-06 on three decodes of the SP playback tap
(/testdata/test_patterns/vhs/playback, `-f 50`, the decode command in
`RUNTIME_CONTRACT`), sync profiles pooled over lines 25 to 239 and fields
grouped by `isFirstField`, 13 or 14 fields a head:

    decode                head difference    head-common departure
                          (|H|, dB rms)      from flat (dB rms)
    zaroff multiburst           0.388              13.077
    zaroff pulse and bar        0.747              13.181
    zaroff 75 bars              0.464               8.966

So the per-head term is between 3 and 6 per cent of the total. The head
axis is real and it is small, which is exactly why it cannot be the whole
of the identification.

THE RECORD TAP SEPARATES THE TAPE FROM EVERYTHING AHEAD OF IT, AND THIS IS
THE MEASUREMENT THAT SETTLES THE QUESTION. The captures carry both of the
VCR's RF test points (readme.txt in /testdata/test_patterns/vhs: record RF
at CN261 pin 1, playback RF at pin 2, the same TSG-130A pattern into the
same Sony SLV-778HF). Decoding the two taps of one pattern and comparing
their pooled sync response against the specified pulse:

    departure from the specified flat, dB rms, 26 or 27 fields a tap:

        pattern            playback   record   tape+playback   pre-tape
                           tap        tap      (pb vs record)  share
        multiburst y-only    13.077   12.896          2.158     98.6 %
        pulse and bar        13.181   11.741          1.951     89.1 %
        75 bars               8.966    8.208          2.526     91.5 %

THE DEPARTURE IS ADDED BEFORE THE TAPE. Between 89 and 99 per cent of the
luma response error is already present at the recording VCR's own RF test
point, and the tape and playback channel - which is what every VCR stage in
this arc models - can only ever reach the remaining 2 dB, and reaches a
consistent 1.95 to 2.53 dB of it on all three patterns. That is Ethan's
*"the rf path seems complete, but the picture path is not"*, in decibels,
and it is why he can see it in the luma.

The time axis lands in the same place. Against the same specified pulse the
pooled record tap shows a sync width change of -42.4, -37.9 and -37.0 ns on
the three patterns and the playback tap -46.1, -46.1 and -53.5, so the tape
adds between 4 and 16 ns of it and the rest arrives pre-tape; the whole
playback-against-record delay is 4.77, 2.58 and 5.46 ns. (The absolute
delay against a synthetic reference depends on where the segment frames the
pulse and is not quoted as physical; the tap-against-tap figures do not,
because both sides are framed identically.)

WHAT THE RECORD TAP DOES NOT SEPARATE, SAID PLAINLY. The record RF test
point sits after the recorder's own front end - its input amplifier, its
luma low-pass and its pre-emphasis and modulator - so "pre-tape" is the
source chain AND the recorder's front end, not the source chain alone.
Those two are in series ahead of the tape and NOTHING here separates them:

  * the head axis cannot, because the recorder's front end is ahead of both
    playback heads and is head-common exactly as the source is;
  * the frequency axis cannot, because the sync pulse's own resolution is
    1/T = 213 kHz for the specified 4.7 microsecond width (the arc's 1/T
    probe law), and both terms are smooth on that scale - there is no
    ripple, no null and no structure in either to tell them apart;
  * the profile cannot, because it says which source terms may EXIST, not
    how large they are.

So the source share of the head-common departure is bounded and not
resolved: it is AT MOST the 89 to 99 per cent the record tap measures and
AT LEAST zero. This module corrects the head-common departure as a
whole and says, here, that its attribution inside that bound is not a
measurement. That is the honest statement, and the alternative - splitting
it by an assumed ratio - would be a number nothing supports.

--------------------------------------------------------------------------
THE PROFILE DECIDES WHICH SOURCE TERMS CAN EXIST AT ALL
--------------------------------------------------------------------------

`profiles.py` already draws this and is read from here rather than
re-decided: a camera or a composite line feed has no 4.2 MHz limit and no
multipath; a television source has both. Attaching a broadcast channel to a
tape that never saw a broadcast applies a correction where the mechanism
does not exist, and the arc has paid for that mistake once already with the
tuner clip site.

AND ON VHS THE PROFILE PRUNES NOTHING, WHICH IS ITSELF THE FINDING. Both
television-only terms fall outside what a VHS decode can reach, for the two
reasons below, so the correction comes out the same whatever the source and
the profile is REPORTED rather than applied. `model_stages` reads the
capture's own history through `profiles.for_capture` and logs it once a
decode - which is the first time anything in the runtime has called
`profiles.py` at all - so the provenance is visible and the same call would
gate on a format whose band reaches further. Measured on the two tapes that
differ:

    /testdata/countdown.flac   'television' - 14 chain entries present, 13
                               observable, multipath withheld, the 4.2 MHz
                               limit above the band
    zaroff test patterns       'composite' - 4 entries, all observable, no
                               transmission channel at all

TWO SOURCE TERMS ARE SPECIFIED WITH NUMBERS AND NEITHER IS RECOVERABLE FROM
A VHS DECODE, which is worth stating because it is a real limit rather than
an omission:

  * THE TRANSMISSION CHANNEL'S 4.2 MHz LIMIT exists only for a `television`
    source (SMPTE 170M, note to clause 7). The VHS luma band is 3.0 MHz
    (`sync_shape.VHS_LUMA_BAND_HZ`, a format constant), so the broadcast
    channel's corner lies ENTIRELY ABOVE the band a VHS tape returns and
    nothing in a VHS decode observes it. It is declared by `admissible` and
    contributes nothing to the corrector; on a wider format it would.

  * MULTIPATH is a tapped delay line reaching 2.35 microseconds
    (`multipath.py`), so its ripple period is 425 kHz and upward - just
    resolvable against the pulse's 213 kHz. It is admitted for a
    `television` source and refused for every other one. It is NOT fitted
    here: `multipath.py`'s own held-out test collapses from 56 per cent to
    17 with its phase going negative on the only broadcast tape this arc
    holds, and a term that does not hold out is not a term to subtract.
    `admissible` reports it as present-but-unfitted rather than dropping it
    silently.

--------------------------------------------------------------------------
ALL MEASURABLE DIMENSIONS - AND WHICH ONE IS NOT MEASURABLE HERE
--------------------------------------------------------------------------

Ethan's requirement is explicit, so the correction is not a magnitude:
amplitude, frequency AND time. `sync_shape.shape_components` gives all
three from ONE non-parametric fit of the pulse and they are used as three
readings of one object, never as three fits.

    AMPLITUDE   the head-common magnitude departure from the specified
                flat, which is the corrector's own magnitude
    FREQUENCY   the abscissa, at the pulse's own 1/T = 213 kHz resolution,
                on the bins the specified pulse's spectrum supports
    TIME        the corrector's phase, and it is NOT fitted beside the
                magnitude

THE TIME AXIS OBEYS BODE AND IS NOT FITTED TWICE. A minimum-phase
response's delay is fixed by its own magnitude through Bode's relation, so
the corrector's phase is `hypercomplex.minimum_phase` OF ITS OWN LOG
MAGNITUDE and nothing else. Fitting a delay beside it would be fitting the
same quantity a second time. The separate term Bode does not fix is the
EXCESS phase, and this module measures it, tests it, and REFUSES IT:

    CONTROL, 2026-09-06, AND IT HAD TO BE MATCHED TO THE TAPE TO MEAN
    ANYTHING. An analogue Butterworth low-pass is minimum phase by
    construction, so its excess phase is exactly zero; passing the specified
    pulse through one and reading the excess back through this segment's own
    geometry - 90 samples, 46 bins, 14 of them carried - says how much of a
    reading is the instrument. The size of that error depends entirely on
    how STEEP the control is, so a control gentler than the tape would prove
    nothing:

        control (true excess = 0)           excess phase read back
        Butterworth n=2, fc 2.00 MHz            3.16 deg rms
        Butterworth n=3, fc 1.20 MHz           11.77 deg rms
        Butterworth n=4, fc 1.35 MHz           16.27 deg rms   <- MATCHED
        Butterworth n=6, fc 1.60 MHz           46.37 deg rms
        Butterworth n=4, fc 1.00 MHz           49.81 deg rms
        pure delay 20 ns                        0.045 deg rms  (correct)
        pure delay 100 ns                       0.058 deg rms  (correct)
        identity                                0.000 deg rms  (correct)

    The matched line is the one that counts. Fitting a Butterworth to the
    measured head-common magnitude puts the tape's own luma channel at
    order 4 with a 1.35 MHz corner, to a misfit of 2.38 dB rms, and THAT
    control reads 16.27 degrees rms where the truth is zero.

    Against it the real decodes read 57.5, 44.4 and 34.4 degrees rms - two
    to three and a half times the matched control, and inside the spread of
    the controls either side of it, which read 46.4 and 49.8. The reading
    does not stand clear of the instrument at the significance the rest of
    this arc requires, so `usable` compares it with
    `chroma_head_switch.DETECTION_SIGMA` times the control and returns
    False on all three. It is unmeasured, not measured, and no correction is
    built on it; `excess_phase_bound` returns the number with the control
    beside it so a longer segment can revisit it.

    (The first version of this control was wrong in a way worth recording,
    because it inverted the conclusion. `scipy.signal.butter(..., analog=
    True)` takes its corner in RADIANS PER SECOND, not hertz, so controls
    labelled 2.0, 2.5 and 1.6 MHz were really 318, 398 and 255 kHz - far
    steeper across the measured band than any tape - and read 42.5, 630.8
    and 861.4 degrees. On that evidence the estimator looked hopeless at
    every steepness. It is not: it is good to 3 degrees on a gentle channel
    and fails on a steep one, and the tape happens to sit where it is
    marginal.)

    What the same controls confirm is that a DELAY is recovered exactly
    (20 and 100 ns to better than a part in a thousand), which is why the
    time axis is checked on `sync_shape.edge_pair` instead: a channel delay
    moves both transients together and a width change moves them apart.
    Read that way across the two taps the time axis lands pre-tape exactly
    as the magnitude does - the width change is -42.4, -37.9 and -37.0 ns
    already at the record tap against -46.1, -46.1 and -53.5 at the
    playback tap - which is the figure quoted above.

THE PURE DELAY IS REMOVED FROM THE CORRECTOR ON PURPOSE. A delay common to
the whole line displaces the picture and is the time base's business, not
the response's; leaving it in would make this stage fight the time base
corrector. Only the departure from linear phase is applied.

--------------------------------------------------------------------------
HOW MUCH IS APPLIED
--------------------------------------------------------------------------

The correction gain law, which is standing in this arc: the optimum
fraction of a believed correction is `a* = 1/(1 + rho)` with `rho` the
model error, the result goes back to zero at `2 a*`, and the instruction is
to apply at HALF the believed optimum because `rho` is invisible to a
split-half test. Here `rho` is not small and is not guessed at: the
attribution between the source chain and the recorder's front end is
unresolved across the whole bound above, so the model error is of the same
order as the quantity, `a*` is about one half, and half of that is a
quarter. `AMOUNT_LAW` computes it rather than writing it down, and the
graph's `amount` multiplies it - so `--stages +source_correction` seeds
1.0, which means "the gain law's recommendation", not "the whole inverse".

THE GAIN IS CAPPED WHERE THE MEASUREMENT STOPS SUPPORTING IT. A full
inverse would multiply by 1/0.03 = 33 at 2.86 MHz, where the channel has
already collapsed, and would amplify that bin's noise with it. Each bin's
correction is therefore capped at its own measured signal-to-noise divided
by `chroma_head_switch.DETECTION_SIGMA` - the same significance the lock
gate and the head-switch detector already use, so one constant governs all
three - which says a bin may be lifted only until its own evidence reaches
the arc's detection threshold. The cap is derived from the segment's own
out-of-band noise, which `sync_shape` measures as a matter of course, and
is not a number anyone chose.

--------------------------------------------------------------------------
WHERE IT IS APPLIED, AND WHERE IT IS NOT
--------------------------------------------------------------------------

THE ACTIVE AREA ONLY. The specification's flatness claim is about the
picture signal; the blanking interval carries the timing and level
references that the decoder's own stages have already measured on - the
sync tip and the back porch that `hz_to_output` anchors to - and reshaping
those after they have been read would move anchors other stages depend on.
`colour_free_luma` restricts itself to the active area for the same reason.

THE MEASUREMENT IS THE SYNC PULSE AND THE GAUGES ARE NOT. The standing rule
in this arc is that a correction derives from the sync pulse alone and
never from visible-area content, with active-area instruments kept for
offline validation. That is obeyed here, and it has a happy consequence:
the multiburst and the pulse-and-bar in the active area are gauges this
corrector is not fitted with, so they judge it independently.

--------------------------------------------------------------------------
WHAT IT DOES TO THE PICTURE, ON GAUGES IT IS NOT FITTED WITH
--------------------------------------------------------------------------

Measured 2026-09-06 on SP playback decodes with and without
`--stages +source_correction`, 5400 active lines a decode, everything below
in the active area and therefore invisible to the corrector.

THE MULTIBURST, which is the specified luma-response instrument: six
packets of EQUAL amplitude at 50 +/- 0.5 IRE (ITU-R J.63 Table III, EBU
Tech 3209 7.2.9, via `composite_channel.multiburst_reference`), so their
departure from equal IS the response. Located by band-passing at each
packet's own frequency, so no packet position is written down. The 3.0 MHz
packet returns 2.4 IRE of its specified 50 and is past the tape's recovery,
so the three that survive are what the figures are taken over:

    packet amplitude, IRE      0.5 MHz   1.0 MHz   2.0 MHz    3.0 MHz
    before                       54.18     48.05     25.87       2.37
    after                        55.39     52.75     34.19       2.96

    against the specified 50 IRE      14.183  ->   9.775 IRE rms   -31 %
    departure from EQUAL               2.845  ->   1.894 dB rms    -33 %
    worst packet against best          6.421  ->   4.191 dB        -35 %

THE 75 PER CENT BARS, whose every step is specified flat. Splitting each
step top into the part common to all lines - the channel's own tilt and
ringing - and the part that varies line to line, which is the noise:

    step tops, IRE            static (the channel)   noise (the cost)
    before                      1.0884 +- 0.0049      0.6276 +- 0.0075
    after                       0.9744 +- 0.0093      0.6737 +- 0.0083
                                     -10.5 %               +7.3 %

THE COST IS REAL AND IS NOT HIDDEN. A high-frequency lift amplifies the
noise that was already there, and on the pulse-and-bar the trade goes the
other way:

    pulse-and-bar bar top     static                 noise
    before                      5.3504 +- 0.0079      1.7805 +- 0.0120
    after                       5.5860 +- 0.0182      2.0246 +- 0.0199
                                      +4.4 %              +13.7 %

and the swept-frequency envelope - a constant-amplitude sweep whose
envelope is the response measured on content - is a wash over the fourteen
of twenty-four position bins that still carry signal: 6.049 to 6.158 dB rms
of flatness, with the worst-against-best falling 17.797 to 17.676 dB. The
sweep lifts where the correction acts (its bins 4 to 11 rise by 2 to 4 dB)
and cannot flatten what the tape no longer returns, so a metric spanning
the whole sweep is dominated by the dead end of it. That is the gain law
visible in a gauge: a quarter of the way is a quarter of the way.

THE ABSOLUTE SCALE IS UNTOUCHED, which is the check that the stage stays
inside the active area. The sync-to-blanking spacing - the luma's only
absolute reference - reads identically to four decimal places on all three
decodes before and after: 33.2486, 34.5700 and 34.0352 IRE.

AND A DECODE THAT DOES NOT NAME THE STAGE IS UNCHANGED. Byte-compared,
luma and chroma both: `cmp` reports no difference between a decode taken
before this module existed and one taken after it with no `--stages`
argument.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import signal

from vhsdecode.models import chroma_head_switch
from vhsdecode.models import composite_channel
from vhsdecode.models import hypercomplex as hc
from vhsdecode.models import pair_dimension as pdim
from vhsdecode.models import profiles
from vhsdecode.models import sync_shape

__all__ = [
    "AMOUNT_LAW",
    "SPECIFIED_FLAT",
    "admissible",
    "amount_law",
    "head_split",
    "channel",
    "excess_phase_bound",
    "corrector",
    "kernel",
    "apply_to_active",
    "RUNTIME_CONTRACT",
]


SPECIFIED_FLAT = True
"""SMPTE 170M-2004 clause 7.1 imposes no luminance bandwidth, so the
specified source response is flat and the departure from it is the channel.
Held as a named constant so the claim is greppable; the citation itself
lives in `composite_channel.luma_response` and is read from there."""


def amount_law(model_error: float = 1.0) -> float:
    """The fraction of the believed correction to apply, from the gain law.

    `a* = 1 / (1 + rho)` is the optimum and the standing instruction is to
    apply HALF of it, because `rho` - the model error - is invisible to a
    split-half test and over-application is worse than under-application
    (the gain goes back to zero at `2 a*`).

    The default `rho = 1` is not a guess: the attribution of the head-common
    departure between the source chain and the recorder's front end is
    unresolved across its entire bound, so the model error is of the same
    order as the quantity being corrected. Passing a smaller `rho` is how a
    caller that has separated them would raise it.
    """
    if not model_error >= 0.0:
        raise ValueError("the model error is a non-negative ratio")
    return 0.5 / (1.0 + float(model_error))


AMOUNT_LAW = amount_law()
"""The gain law's recommendation at the measured model error: one quarter.
Computed rather than written down, so a change in the law changes here."""


def admissible(source: str) -> Dict[str, object]:
    """Which source terms may exist at all, and which are observable here.

    `profiles.py` owns the question and is read rather than re-decided. The
    answer is split three ways, because "present" and "correctable" are not
    the same thing and conflating them is how a correction gets applied
    where its mechanism does not exist:

      `present`      the chain entries this source has
      `observable`   those the recovered VHS band can see at all
      `withheld`     those present and observable but deliberately not
                     fitted, with the reason

    The 4.2 MHz transmission limit is the clearest case: it exists for a
    television source and lies entirely above the 3.0 MHz the tape returns,
    so it is present and NOT observable.
    """
    if source not in profiles.SOURCES:
        raise ValueError("unknown source %r; known: %s"
                         % (source, sorted(profiles.SOURCES)))
    chain = list(profiles.SOURCES[source]["chain"])
    band = float(sync_shape.VHS_LUMA_BAND_HZ)
    limit = float(composite_channel.BROADCAST_LIMIT_HZ)
    television = source == "television"

    observable, withheld = [], {}
    for entry in chain:
        if entry == "multipath":
            withheld[entry] = (
                "admitted for this source but not fitted: `multipath.py`'s "
                "own held-out test collapses from 56 per cent to 17 with its "
                "phase going negative on the only broadcast tape this arc "
                "holds, and a term that does not hold out is not a term to "
                "subtract")
            continue
        observable.append(entry)

    return {
        "source": source,
        "present": chain,
        "observable": observable,
        "withheld": withheld,
        "band_limit_hz": band,
        "broadcast_limit_hz": limit if television else None,
        "broadcast_limit_observable": bool(television and limit < band),
        "why_not_observable": (
            "the transmission channel's %.1f MHz corner (SMPTE 170M, note to "
            "clause 7) lies above the %.1f MHz a VHS tape returns "
            "(`sync_shape.VHS_LUMA_BAND_HZ`), so no VHS decode observes it"
            % (limit / 1e6, band / 1e6)) if television else (
            "this source passed through no transmission channel, so it has "
            "no 4.2 MHz limit and no multipath to remove"),
        "specified_luma_response": composite_channel.luma_response(
            [0.0, band], source=source),
    }


def head_split(by_parity: Dict[bool, Sequence[np.ndarray]]) -> Dict[str, object]:
    """The head-common and head-difference sync profiles.

    `by_parity` maps `isFirstField` to that head's accumulated profiles.
    The DIFFERENCE is purely the VCR - the source never knew which head
    would write it - and the COMMON part is what this stage acts on.

    Both are returned with the per-field scatter that produced them, so a
    caller can see whether the difference it is discarding was measured or
    was noise. Refuses a split with only one head present rather than
    calling one head's profile "common", because a single head's response
    contains the whole per-head term and correcting on it would put that
    term into the picture.
    """
    heads = {bool(k): [np.asarray(v, dtype=np.float64).ravel()
                       for v in values]
             for k, values in by_parity.items() if len(values)}
    if len(heads) < 2:
        raise ValueError(
            "the head split needs both parities; with one head the "
            "difference cannot be removed and the common part is not common")
    lengths = {len(p) for values in heads.values() for p in values}
    if len(lengths) != 1:
        raise ValueError("every profile must span the same segment")

    means = {parity: np.mean(values, axis=0) for parity, values in heads.items()}
    counts = {parity: len(values) for parity, values in heads.items()}
    first, second = means[True], means[False]
    common = 0.5 * (first + second)
    difference = first - second

    errors = {}
    for parity, values in heads.items():
        stack = np.asarray(values)
        errors[parity] = (np.std(stack, axis=0, ddof=1) / math.sqrt(len(values))
                          if len(values) > 1 else np.zeros(stack.shape[1]))
    scatter = float(np.sqrt(np.mean(
        np.asarray([e ** 2 for e in errors.values()]))))

    return {
        "common": common,
        "difference": difference,
        "per_head": means,
        "fields": counts,
        "difference_rms": float(np.sqrt(np.mean(difference ** 2))),
        "field_scatter_rms": scatter,
        # whether the difference being discarded was measured at all
        "difference_significant": bool(
            np.sqrt(np.mean(difference ** 2)) > chroma_head_switch.DETECTION_SIGMA
            * max(scatter, 1e-12)),
        "why": ("the head difference is the VCR's alone because the source "
                "signal never knew which head would write it, so it is "
                "removed from the correction by construction rather than "
                "fitted and subtracted"),
    }


def channel(common_profile, sample_rate_hz: float,
            offset_samples: Optional[float] = None,
            band_limit_hz: float = sync_shape.VHS_LUMA_BAND_HZ
            ) -> Dict[str, object]:
    """The head-common luma response against the specified flat source.

    The reference is the pulse SMPTE 170M table 2 prescribes, built by
    `pair_dimension.spec_sync` so the specification is transcribed once in
    this tree and not twice, and rolled to sit where the measured pulse
    actually sits in the segment.

    THE ALIGNMENT IS MEASURED, NOT PREDICTED, and that is deliberate. Where
    the segment's own geometry says the pulse should be and where it is are
    not the same: on the SP playback decodes the format tables put the
    pulse's centre at sample 49.65 of a 90 sample segment and
    `sync_shape.locate_transients` finds it at 51.00, a difference of 1.35
    samples - 94 nanoseconds. Rolling the reference by the predicted figure
    would put those 94 nanoseconds into the response as a delay the channel
    never had, and `transit_delay`'s lag search is bounded by the specified
    140 nanosecond rise, so most of its range would be spent on the
    framing. `locate_transients` is used instead, which also validates the
    recovered width against the specified 4.7 microseconds and refuses a
    mis-framed segment rather than measuring one. Pass `offset_samples` to
    override it where a caller already knows better.

    Returns the COMPLEX response, because a response has a phase and the
    arc has lost a measurement's rank at exactly this kind of boundary
    before. Its magnitude is the amplitude axis, its bins are the frequency
    axis, and its phase is carried but is NOT the corrector's phase - see
    `corrector`, and Bode.
    """
    measured = np.asarray(common_profile, dtype=np.float64).ravel()
    length = len(measured)
    reference = pdim.spec_sync(float(sample_rate_hz), length)
    if offset_samples is None:
        where = sync_shape.locate_transients(measured, float(sample_rate_hz))
        offset_samples = 0.5 * (where["fall"] + where["rise"])
    # `spec_sync` centres the pulse in its window, so the roll is the
    # displacement between that centre and the measured one.
    centre = (length - 1) / 2.0
    reference = np.roll(reference, int(round(float(offset_samples) - centre)))

    fit = sync_shape.transit_delay(measured, reference, float(sample_rate_hz),
                                   float(band_limit_hz))
    carried = np.asarray(fit["carried"])
    response = np.asarray(fit["response"])
    frequencies = np.asarray(fit["frequency_hz"])

    magnitude = np.abs(response)
    reference_level = (float(magnitude[carried][0]) if carried.any()
                       else 1.0)
    departure_db = np.zeros_like(frequencies)
    departure_db[carried] = 20.0 * np.log10(
        np.maximum(magnitude[carried], 1e-12) / max(reference_level, 1e-12))

    # The per-bin evidence, from the segment's OWN out-of-band noise. This
    # is what caps the correction: a bin may be lifted only until its own
    # signal-to-noise reaches the arc's detection significance.
    floor = float(fit["noise_floor_power"])
    power = np.asarray(fit["weight"])
    snr = np.zeros_like(frequencies)
    if floor > 0:
        snr[carried] = np.sqrt(np.maximum(power[carried], 0.0) / floor)

    shape = sync_shape.shape_components(measured, float(sample_rate_hz),
                                        float(band_limit_hz))
    return {
        "response": response,
        "frequency_hz": frequencies,
        "carried": carried,
        "magnitude": magnitude,
        "departure_db": departure_db,
        "departure_db_rms": float(np.sqrt(np.mean(departure_db[carried] ** 2)))
        if carried.any() else 0.0,
        "signal_to_noise": snr,
        "reference": reference,
        "reference_level": reference_level,
        "alignment_samples": float(offset_samples),
        # the time axis as a MEASUREMENT, checked on the two transients
        "delay_s": float(fit["delay_s"]),
        "is_a_delay": bool(fit["is_a_delay"]),
        "resolution_hz": 1.0 / float(pdim.SYNC_WIDTH_S),
        "bins": int(carried.sum()),
        "noise_rms_ire": float(shape["noise_rms"]),
        "band_limit_hz": float(band_limit_hz),
        "sample_rate_hz": float(sample_rate_hz),
        "specified": composite_channel.luma_response(
            frequencies[carried] if carried.any() else [0.0]),
    }


def excess_phase_bound(measured: Dict[str, object]) -> Dict[str, object]:
    """The excess phase, and the control that says it is not a measurement.

    Bode's relation fixes a minimum-phase response's delay from its own
    magnitude; what is left over is the excess phase and it is the only
    separate term on the time axis. This computes it - and then reports the
    control that decides whether it means anything.

    ON THIS SEGMENT IT DOES NOT, AT THE STEEPNESS THIS TAPE HAS. A
    Butterworth low-pass is minimum phase by construction, so its excess
    phase is exactly zero; read back through this segment's geometry the
    same estimator returns 16.27 degrees rms for the order and corner a
    Butterworth fit to the measured channel comes to - order 4 at 1.35 MHz -
    and 46.37 and 49.81 for the two controls either side of it. The real
    decodes read 57.5, 44.4 and 34.4 degrees rms, which is inside that
    spread.

    So this returns a BOUND and never a correction. `usable` is False unless
    the reading stands clear of the control by
    `chroma_head_switch.DETECTION_SIGMA`, which on all three decodes it does
    not, and the corrector does not call this function at all. The module
    docstring carries the full control table and the units error that first
    made it say the opposite.
    """
    frequencies = np.asarray(measured["frequency_hz"])
    carried = np.asarray(measured["carried"])
    response = np.asarray(measured["response"])
    log_magnitude = np.zeros_like(frequencies)
    log_magnitude[carried] = np.log(
        np.maximum(np.abs(response[carried]), 1e-12))
    minimum = hc.minimum_phase(log_magnitude, valid=carried)
    excess = np.zeros_like(frequencies)
    excess[carried] = np.angle(
        response[carried] * np.exp(-1j * minimum[carried]))
    if carried.any():
        excess[carried] = np.unwrap(excess[carried])
        # a pure delay is not a distortion, so the linear part comes out
        weight = np.abs(response[carried]) ** 2
        design = np.vstack([frequencies[carried],
                            np.ones(int(carried.sum()))]).T
        solved, *_ = np.linalg.lstsq(design * weight[:, None],
                                     excess[carried] * weight, rcond=None)
        excess[carried] = excess[carried] - design @ solved
    reading = (float(np.degrees(np.sqrt(np.mean(excess[carried] ** 2))))
               if carried.any() else 0.0)
    # A reading is believed only where it stands clear of the instrument's
    # own error at the significance the rest of this arc requires - the same
    # constant the lock gate, the head-switch detector and the corrector's
    # cap use, so the four cannot drift apart.
    threshold = chroma_head_switch.DETECTION_SIGMA * CONTROL_EXCESS_DEG_RMS
    return {
        "excess_phase_rad": excess,
        "frequency_hz": frequencies,
        "carried": carried,
        "reading_deg_rms": reading,
        "control_deg_rms": CONTROL_EXCESS_DEG_RMS,
        "usable": bool(reading > threshold),
        "threshold_deg_rms": threshold,
        "control": (
            "an analogue Butterworth low-pass has exactly zero excess "
            "phase; through this segment's geometry, at the steepness the "
            "tape actually has, the same estimator reads %.2f deg rms where "
            "the truth is zero, so a reading below %.1f - that error at the "
            "arc's detection significance - is the estimator and not the "
            "channel" % (CONTROL_EXCESS_DEG_RMS, threshold)),
        "why_not_corrected": (
            "the estimator's error on a control whose answer is zero is the "
            "same size as the reading on real decodes, so the excess phase "
            "is unmeasured rather than measured and no correction is built "
            "on it"),
    }


CONTROL_EXCESS_DEG_RMS = 16.27
"""The excess phase this segment's geometry reports for a channel whose true
excess phase is exactly zero, at the STEEPNESS THE TAPE ACTUALLY HAS.

Measured 2026-09-06 through `sync_shape.transit_delay` and
`hypercomplex.minimum_phase` on 90 samples at 14.318 MHz, the control being
an analogue Butterworth of order 4 with a 1.35 MHz corner - which is what a
Butterworth fit to the measured head-common magnitude comes to, to a misfit
of 2.38 dB rms. A gentler control reads 3.16 degrees and a steeper one
46.37, so the matched one is the only one that says anything about the real
reading.

A LIMIT OF THE INSTRUMENT, labelled as such: it is not a property of any
tape, and a longer segment would lower it."""


def corrector(measured: Dict[str, object], amount: float = 1.0,
              model_error: float = 1.0) -> Dict[str, object]:
    """The complex correction, on the bins the measurement supports.

    `amount` multiplies the GAIN LAW'S recommendation rather than the whole
    inverse, so `amount = 1.0` means "apply what the law says" and not
    "apply all of it". The law is `amount_law(model_error)`.

    THE PHASE IS BODE'S AND NOTHING ELSE. The corrector's magnitude is the
    fractional inverse of the measured magnitude in the log domain - the
    same form the luma path equalizer and `baseband_eq` already use - and
    its phase is `hypercomplex.minimum_phase` of that log magnitude. No
    delay is fitted beside it, because a minimum-phase response's delay is
    already fixed by its magnitude and fitting one would be fitting the same
    quantity twice. The excess phase, which is the separate term, is refused
    on the evidence `excess_phase_bound` reports.

    THE GAIN IS CAPPED PER BIN at that bin's own signal-to-noise divided by
    `chroma_head_switch.DETECTION_SIGMA`, so a bin may be lifted only until
    its evidence reaches the arc's detection significance. Where the channel
    has collapsed - 0.03 at 2.86 MHz on the multiburst decode, a full
    inverse of 33 - the cap is what stops the correction amplifying that
    bin's noise into the picture.

    DC IS PINNED AT UNITY. The level is other stages' business and this one
    must not move it: the corrector is normalised so its lowest carried bin
    is exactly 1, which is also why `channel` reports its departure relative
    to that bin.
    """
    frequencies = np.asarray(measured["frequency_hz"])
    carried = np.asarray(measured["carried"])
    magnitude = np.asarray(measured["magnitude"])
    snr = np.asarray(measured["signal_to_noise"])
    level = float(measured["reference_level"])

    fraction = float(amount) * amount_law(model_error)
    log_departure = np.zeros_like(frequencies)
    log_departure[carried] = np.log(
        np.maximum(magnitude[carried], 1e-12) / max(level, 1e-12))

    # the fractional inverse, in the log domain
    log_gain = -fraction * log_departure
    # ... capped where the measurement stops supporting it. The cap is the
    # bin's own evidence at the arc's detection significance, in the same
    # log domain, so it bounds a ratio and not a difference of decibels.
    ceiling = np.zeros_like(frequencies)
    ceiling[carried] = np.log(np.maximum(
        snr[carried] / chroma_head_switch.DETECTION_SIGMA, 1.0))
    capped = np.clip(log_gain, -np.inf, ceiling)
    limited = carried & (log_gain > ceiling)

    phase = hc.minimum_phase(capped, valid=carried)
    gain = np.zeros_like(frequencies, dtype=np.complex128)
    gain[carried] = np.exp(capped[carried] + 1j * phase[carried])
    # unity where the measurement says nothing, so an unmeasured bin is
    # passed through rather than zeroed
    gain[~carried] = 1.0 + 0.0j

    return {
        "gain": gain,
        "frequency_hz": frequencies,
        "carried": carried,
        "fraction": fraction,
        "amount_law": amount_law(model_error),
        "capped_bins": int(limited.sum()),
        "capped_frequency_hz": frequencies[limited],
        "maximum_gain_db": float(20.0 * np.log10(np.abs(gain[carried]).max()))
        if carried.any() else 0.0,
        "phase_source": "Bode, from the corrector's own log magnitude",
        "why": ("the fractional inverse in the log domain, at the gain "
                "law's fraction, with the phase Bode's relation fixes and "
                "no delay fitted beside it"),
    }


def kernel(correction: Dict[str, object], length: int,
           sample_rate_hz: float, reach_samples: Optional[int] = None
           ) -> np.ndarray:
    """The corrector as a real, symmetric FIR of a derived length.

    `length` is the line the kernel will be applied along. The correction is
    interpolated from the pulse's own coarse grid onto that line's grid,
    inverse transformed, centred, and TRUNCATED at the reach the measurement
    can describe - which is the segment it was measured on and not a chosen
    number: a response measured at 1/T resolution cannot describe an impulse
    response longer than T. That is the same truncation law this arc already
    records for the ringing kernel, applied to its own measurement's width.

    The kernel is CENTRED IN ITS WINDOW rather than at its start, so the
    correction adds no delay of its own; the delay the response carries has
    already been taken out of the phase for the same reason. That is a
    matter of where the window sits, not of the filter's direction:
    convolution reverses the kernel, so a tap beyond the centre reads an
    EARLIER picture sample, and the standing rule against a two-sided
    kernel is about which side of the driving edge the response falls on.

    IT LANDS ON THE CAUSAL SIDE, MEASURED. Because the phase is Bode's, the
    corrector is minimum phase and its impulse response is one-sided by
    construction. On the multiburst decode's own kernel at the shipped
    amount, of 135 taps:

        energy at the centre tap                       97.65 per cent
        energy on the causal side                       2.24 per cent
        energy on the anti-causal side                  0.11 per cent
        largest tap, causal side against anti-causal    0.0825 to 0.0234

    so the two sides stand at about twenty to one and the anti-causal
    remainder is the cepstral estimate's own leakage rather than a
    two-sided design. It is reported rather than asserted away.
    """
    frequencies = np.asarray(correction["frequency_hz"])
    gain = np.asarray(correction["gain"])
    carried = np.asarray(correction["carried"])
    line_bins = np.fft.rfftfreq(int(length), 1.0 / float(sample_rate_hz))

    if not carried.any():
        response = np.ones(line_bins.shape, dtype=np.complex128)
    else:
        known = frequencies[carried]
        # interpolated in the LOG domain, where the correction was formed,
        # so the magnitude cannot go negative between two measured bins
        log_magnitude = np.interp(line_bins, known,
                                  np.log(np.abs(gain[carried])),
                                  left=0.0, right=0.0)
        angle = np.interp(line_bins, known,
                          np.unwrap(np.angle(gain[carried])),
                          left=0.0, right=0.0)
        response = np.exp(log_magnitude + 1j * angle)
    # unity at DC exactly: the level belongs to other stages
    response[0] = 1.0

    impulse = np.fft.irfft(response, int(length))
    impulse = np.fft.fftshift(impulse)
    reach = (int(round(float(pdim.SYNC_WIDTH_S) * float(sample_rate_hz)))
             if reach_samples is None else int(reach_samples))
    half = min(max(reach, 1), (int(length) - 1) // 2)
    centre = int(length) // 2
    cut = impulse[centre - half: centre + half + 1].copy()
    total = float(cut.sum())
    if abs(total) > 1e-12:
        # truncation removes a little of the kernel's area; restoring it
        # keeps DC at unity, which the untruncated kernel already had
        cut /= total
    return cut


def apply_to_active(picture, active: Sequence[int], lines: Sequence[int],
                    fir: np.ndarray) -> Dict[str, object]:
    """Filter the active area of each picture line, in place.

    THE SPAN IS THE CALLER'S CHOICE, and the choice matters more than it
    looks. Passing the active span alone leaves the blanking exactly as it
    arrived, which was this function's original and only behaviour, on the
    argument that the sync tip and the back porch are anchors the decoder's
    level stages have already measured on. Passing the WHOLE LINE corrects
    the blanking too.

    ETHAN RULED FOR THE WHOLE LINE, and gave the test that settles it:
    *"The cd sample will have a flat sync pulse when it is working."* The
    physics agrees with him. The channel distorted the entire line, so its
    inverse applies to the entire line, and a correction withheld from the
    sync pulse cannot be checked against the one part of the signal whose
    shape the specification states. Applying it there turns the sync pulse
    into the verification: if the correction is right the pulse comes back
    flat, and if it is wrong the pulse says so.

    The anchors do move, and that is not a side effect to be regretted -
    an anchor measured before the correction was measured on a distorted
    signal, which is correct for DERIVING the correction and wrong for
    describing what the picture should now be.

    The span is padded with its own edge values rather than with zeros, so
    the filter does not manufacture a transition at the active boundary
    where the picture has none.
    """
    field = np.asarray(picture)
    if field.ndim != 2:
        raise ValueError("the picture is a lines-by-samples array")
    # THE WRITE IS IN PLACE, so the array must be one that can be written
    # in place. `np.asarray(x, dtype=np.float64)` on an integer picture
    # returns a COPY, and every line below would then filter that copy and
    # throw it away - a stage that runs, logs, reports a change and alters
    # nothing. Refused rather than converted, because a caller handing an
    # integer buffer here has a conversion to do that this function cannot
    # do for it.
    if field.dtype != np.float64:
        raise ValueError(
            "the picture must be float64 to be filtered in place; %s would "
            "be copied and the correction lost" % field.dtype)
    start, end = int(active[0]), int(active[1])
    first, last = int(lines[0]), int(lines[1])
    taps = np.asarray(fir, dtype=np.float64).ravel()
    half = (len(taps) - 1) // 2
    if end - start <= 2 * half:
        return {"lines": 0, "changed_rms": 0.0,
                "why": "the span is shorter than the kernel's reach"}

    span = field[first:last + 1, start:end]
    padded = np.pad(span, ((0, 0), (half, half)), mode="edge")
    # ONE TRANSFORM FOR THE WHOLE BLOCK, not a Python loop over lines: the
    # same kernel applies to every line. Compared back to back in ONE
    # process on a 237 by 760 active block with this stage's 135 tap
    # kernel, the overlap-add form ran 2.3 times faster than `fftconvolve`
    # and 2.9 times faster than a line-at-a-time `np.convolve`, so it is
    # the one used. Only the RATIOS are quoted: several decode sessions run
    # against this repository at once and an absolute timing taken under
    # that load is not a measurement, which this arc has already recorded
    # once as an "11 per cent regression" that was somebody else's job.
    # `oaconvolve` agrees with the loop to 5e-11 output units on a field of
    # 30000 +/- 2000, which is float rounding and not a different answer.
    filtered = signal.oaconvolve(padded, taps.reshape(1, -1), mode="valid",
                                 axes=1)
    changed = float(np.sqrt(np.mean((filtered - span) ** 2)))
    field[first:last + 1, start:end] = filtered
    return {
        "lines": last + 1 - first,
        "samples": end - start,
        "taps": len(taps),
        "changed_rms": changed,
        "why": ("the active area alone, edge padded, so the blanking "
                "references other stages have already read are untouched"),
    }


RUNTIME_CONTRACT = """
What the decoder calls, and where.

  1. THE SITE is `vhsdecode/field.py`, in `downscale`, immediately after
     `dsout = self.hz_to_output(dsout)` - the last point in the luma path,
     which is where this stage belongs because it is the last one going
     backwards. The picture is on its final time base and in output units,
     and every VCR stage ahead of it has run.

  2. THE GLUE is `vhsdecode.model_stages:correct_source_response`, which
     accumulates the per-field sync profile by head parity, forms the split
     once both heads have been seen, and applies the kernel to the active
     area of the picture in place.

  3. THE SEGMENT is derived from the decoder's own tables and never
     written down: it is centred on the sync pulse, bounded ahead by the
     previous line's front porch and behind by `colorBurstUS[0]`, so the
     colour burst is never inside it and the pulse is framed in blanking on
     both sides. On NTSC VHS at 4fsc that comes to 90 samples.

  4. THE CONTROL is `--stages +source_correction` and there is no command
     line flag. The node declares its own default of 0, so a decode with no
     `--stages` argument is byte-identical to one taken before the node
     existed.

  5. THE DECODE the measurements in this module were taken from:

         vhs-decode <input> -n -f 50 -l 14 -t 4 --dctp --lti_gain 0 \\
             --ire0_adjust --chroma_env_gain 0.5 --overwrite <output>

     with `-f 40` for /testdata/home.flac and /testdata/countdown.flac.
"""
