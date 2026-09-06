"""Clipping, its two possible sites, and why it is not a filter.

Ethan: *"the sync pulse of the video side may be clipped, additionally upper
frequencies in the luma may be clipped (but this is much less likely) ...
Clipping will happen from the source -> recording VCR, if it exists."*

Both halves of that are right, and the second is the reason this module is
separate from every other one here.

THERE ARE TWO SITES, NOT ONE, AND THE SECOND IS ETHAN'S: *"It might be at the
television tuner stage, the cd sample was an over the air recording."*

    the tuner's, position 5     the receiver, before the recorder saw
                                anything; clips at the SYNC TIP
    the recorder's, position 9  the recording machine's input, before the
                                emphasis; clips 40 per cent ABOVE the tip

The recorder's site is a property of the MACHINE and is in every chain. The
tuner's is a property of the SOURCE and is in the chain of material that came
over the air and no other - which is why `signatures` emits it only when the
caller declares a broadcast source. Provenance decides whether a component
exists, and that is the only honest way to carry one that is real on one tape
and absent on another.

WHY THE TWO ARE SEPARABLE, AND IT IS NOT A MATTER OF OPINION. Two hard
limiters acting on the same excursion differ only in threshold - a RATE - and
`composition_control` shows what the separability law costs there: they
compose into ONE limiter at the tighter threshold, and no measurement can
split them. These two are separable only because they do not act on the same
excursion. Negative modulation puts the sync tip at peak carrier (ITU-R
BT.470-6 Table 3 System M items 7 and 8), so a receiver that runs out of
headroom meets the sync pulse first and clips the TIP; the recorder's dark
clip is stated at 40 per cent of the sync-tip-to-peak-white span measured
from the tip, so it sits above the tip and takes the emphasized undershoot
instead. The sync tip's own settled level lies in the window between them,
and is therefore the tuner's witness and nothing else's.

WHAT THE MEASUREMENT SAYS SO FAR, AND IT DOES NOT YET SETTLE IT. Two
witnesses were run on the sibling lane's pooled sync pulses for the
over-the-air sample (cd) against the home recording, both heads, at the
decoded luma output:

    tip minus porch departure   cd +0.19 / +0.09 IRE   home +0.51 / +0.63
    tip/porch added ripple      cd  0.236 / 0.266      home  0.234 / 0.199

NEITHER ATTESTS A CLIP ON CD THAT HOME LACKS, and the first points the other
way. Both are confounded, and the confounds are worth recording because they
say what a decisive instrument would need. The departure is ANTIsymmetric -
the tip reads high by as much as the porch reads low - which is a residual
amplitude mismatch between the fitted ideal and the measurement, not the
one-sided truncation a clip makes; that instrument scales its ideal to the
measured pulse, which absorbs exactly the depth a clip removes. The ripple
ratio is below one on both tapes by the same factor, because the porch
carries burst and picture content the tip does not, so it is not a clean
comparator.

The witness that survives both confounds is the scatter ACROSS pulses at a
fixed sample on the tip, against the same across-pulse scatter at a fixed
sample on the back porch after the burst: a clipped level has no gain, so
what rides on it is suppressed there and nowhere else, and a ratio of
across-pulse variances is blind to any amplitude fit. That needs the per-line
pulses rather than the pooled mean, which is a decode-time measurement.

WHAT IS ATTESTED IS THAT CD'S SOURCE CHAIN DIFFERS, which is Ethan's premise
and not the clip itself. Its recorded sync is 4.608 us wide against home's
4.676 and the test tapes' 4.650, on a 4.700 us spec - a static timing
property of the source, measured the same way on all three.

A second finding points the same way and is marked UNCONFIRMED: a min-phase
test on the exact decoder division is recorded as converging on home and not
on cd, leaving a zero-phase, LEVEL-INDEPENDENT roll-off of source class. The
lane it was attributed to could not reproduce it on request, so it is kept
without attribution and carries no weight until someone re-runs it. Even if
it holds it is not evidence for THIS component, because a clip is
level-DEPENDENT and that residue is not.

WHAT IS CLIPPED, AND WHY THE SYNC PULSE FIRST. The standard fixes the levels
it is applied at. SMPTE 32M-2004 gives VHS a white clipping level of 160 per
cent and a dark clipping level of 40 per cent, both measured from sync tip
with the sync-tip-to-peak-white span as 100 per cent; IEC 60774-3 gives S-VHS
210 per cent and MINUS 70, the dark clip changing sign between the formats.
The sync pulse is the largest excursion in the signal and it is the one that
sits at the dark end, so it meets the dark clip first: on VHS at 40 per cent
of a span whose sync tip is at zero, the clip is above the tip and the pulse
is what it takes. The luma's upper frequencies are clipped only when the
pre-emphasis overshoot carries them past the white clip, which needs a
transient steep enough to matter and is correspondingly rarer.

WHY IT CANNOT BE ENTERED AS A RESPONSE, AND WHAT REPLACES IT. A first
attempt at this module claimed a hard limit fails `subtractable` because its
logarithm is undefined where it flattens the signal. Running the control
refuted that: a clipped signal is bounded away from zero as often as an
unclipped one, and the check passed on both.

The real objection is stronger. A hard limit is MEMORYLESS AND NON-LINEAR,
so it has no frequency response to enter at all - the ratio of output to
input is a function of the sample's own value rather than of frequency, and
two signals of different amplitude see different apparent gains from the
same clipper. Measured, at drives of 0.5, 1, 2 and 4: 0.639, 0.437, 0.282,
0.172. What replaces it is the describing function, the quasi-linear gain a
limiter presents to a sinusoid of a given amplitude, which is a response
because it is stated at a level.

HOW MUCH IT CARRIES, MEASURED, AND IT IS LESS THAN THE RECORD LEVEL. Two
numbers in an earlier draft of this docstring were written before they were
run and were wrong; these are the measured ones.

    describing-function gain at 0.5, 1.0, 1.5, 2.0, 3.0 of the limit
        1.000, 1.000, 0.781, 0.609, 0.416
    the level axis, levels spanning the crossover     1.22 effective of 5
    the same with the limit removed (the control)     1.00 of 5
    the level axis at levels far above the limit      1.00 of 5

So clipping is specified, real, and carries BARELY A DIRECTION on the
frequency axis. The reason is structural rather than a shortcoming of the
model: a memoryless non-linearity has no frequency shape at all, and the
only frequency dependence it acquires is through the emphasis lift that
decides where the drive reaches the limit first. That lift spans about two
to one across the band, so the crossover sweeps a narrow range of levels -
measured, only levels near 0.2 of the sync-tip span put it inside the band
at all - where the record level's cap crossover sweeps the whole band from
12.31 down to 2.05 MHz and earns 2.90 of 5.

THE CONSEQUENCE THAT MATTERS IS NOT A DIRECTION. It is that a detected clip
truncates the IDEAL, not merely the measured signal - see `truncated_ideal`.
A synthetic reference that was never clipped, compared against a recording
that was, reports the clip as a defect of everything downstream of it.

ETHAN'S REVISION, 2026-09-06: *"I no longer think it is possible to clip
the sync pulse."* The evidence gathered since this module was written
supports him on the RECORD side, from three directions that do not share
an assumption:

  * MEASURED. The videosynth reference agent found the SLV-778HF's dark
    clip at 2.787 MHz, which is -125.8 IRE - eighty-six IRE BELOW the
    sync tip at -40 IRE. The sync pulse never descends far enough to meet
    it. No white clip was visible at the record tap at all.
  * THE SPECIFICATION, read the other way. SMPTE 32M-2004 clause
    3.9.1.1.3 puts the dark clip at 40 per cent of the sync-to-white
    excursion measured FROM the tip, which is forty IRE ABOVE it. Either
    reading places the clip out of the pulse's reach; they differ on which
    side, and both agree it cannot be touched.
  * THE CIRCUIT. The keyed AGC of JVC VTG82063 section 3 clamps ON the
    sync tip, which is precisely what holds it away from either level.

So `detect_tip_clip` and `tip_clip_from_lines` should be read as measuring
whether a tip clip is PRESENT, and on this deck the answer is that it is
not and cannot be. What remains open is only the TUNER site at chain
position 5, which is a different machine whose evidence was a
tip-flatness and noise detector rather than a level measurement, and which
no capture in this repository can settle without a tuner to measure.
"""

from typing import Dict, Optional

import numpy as np

# SMPTE 32M-2004: the clipping levels are stated as percentages measured
# from sync tip, with sync tip to peak white as 100 per cent.
VHS_WHITE_CLIP_PERCENT = 160.0
VHS_DARK_CLIP_PERCENT = 40.0
# IEC 60774-3:1993 section 6.1.2, and note the dark clip changes SIGN
# between the two formats rather than merely moving.
SVHS_WHITE_CLIP_PERCENT = 210.0
SVHS_DARK_CLIP_PERCENT = -70.0

# THE RADIATED LEVEL MAP, ATTESTED TWICE AND NOW COMPLETE.
#
# ITU-R BT.1701-1 Table 1, System M, items 7 and 8, read from the document
# itself - and it states the same values as BT.470-6 Table 3 items 7 and 8
# quoted at docs/PROPAGATION_COLLAPSE.md:51. Two independent recommendations
# agreeing to the digit is why these can stop being re-checked.
#
#     type and polarity of vision modulation    C3F NEGATIVE
#     synchronizing level                       100 per cent of peak carrier
#     blanking level                            72.5 to 77.5
#     black level above blanking                2.88 to 6.75 (0 to 6.75 Japan)
#     PEAK WHITE LEVEL                          10 to 15
#
# The peak white figure is NEW here and it is what completes the map. With
# only the sync and blanking levels the tuner's clip could be placed at the
# tip and nowhere else; with peak white the whole sync-tip-to-peak-white span
# is expressible in carrier terms, which is the span the recorder's own
# clipping levels are stated against.
BROADCAST_SYNC_PERCENT_OF_PEAK_CARRIER = 100.0
BROADCAST_BLANKING_PERCENT_OF_PEAK_CARRIER = (72.5, 77.5)
BROADCAST_PEAK_WHITE_PERCENT_OF_PEAK_CARRIER = (10.0, 15.0)
BROADCAST_BLACK_ABOVE_BLANKING_PERCENT = (2.88, 6.75)

CHAIN_PREFIX = "input clipping"
CHAIN_POSITION = 9

# The tuner's own site, in the transmission band and four positions ahead of
# the recorder's input. See `sites()` for why it is a SEPARATE entry rather
# than the same clipper moved.
TUNER_PREFIX = "tuner clipping"
TUNER_CHAIN_POSITION = 5

COMPONENT_POSITIONS: Dict[str, int] = {
    CHAIN_PREFIX: CHAIN_POSITION,
    "input clipping (dark)": CHAIN_POSITION,
    "input clipping (white)": CHAIN_POSITION,
    "input clipping level dependence": CHAIN_POSITION,
    TUNER_PREFIX: TUNER_CHAIN_POSITION,
    "tuner clipping (sync tip)": TUNER_CHAIN_POSITION,
    "tuner clipping level dependence": TUNER_CHAIN_POSITION,
}


def clip_levels(format_name: str = "VHS") -> Dict[str, float]:
    """The two clipping levels the standard states, as fractions of the
    sync-tip-to-peak-white span."""
    name = str(format_name).upper().replace("-", "")
    if name in ("SVHS", "SUPERVHS"):
        white, dark = SVHS_WHITE_CLIP_PERCENT, SVHS_DARK_CLIP_PERCENT
    elif name == "VHS":
        white, dark = VHS_WHITE_CLIP_PERCENT, VHS_DARK_CLIP_PERCENT
    else:
        raise ValueError(f"no stated clipping levels for {format_name!r}")
    return {"white": white / 100.0, "dark": dark / 100.0,
            "span_percent": 100.0,
            "why": ("measured from sync tip, with sync tip to peak white "
                    "as 100 per cent")}


def broadcast_levels() -> Dict[str, object]:
    """The over-the-air level map, and the one fact that matters in it.

    System M is negatively modulated, so the SYNC TIP IS PEAK CARRIER. Every
    other level in the picture sits at a lower carrier amplitude. Anything in
    the receiver that runs out of headroom - the IF stages, the video
    detector, the AGC that is referenced to the sync tip in the first place -
    therefore meets the sync pulse before it meets any part of the picture.
    """
    low, high = BROADCAST_BLANKING_PERCENT_OF_PEAK_CARRIER
    white_low, white_high = BROADCAST_PEAK_WHITE_PERCENT_OF_PEAK_CARRIER
    return {
        "modulation": "C3F negative",
        "sync_percent_of_peak_carrier":
            BROADCAST_SYNC_PERCENT_OF_PEAK_CARRIER,
        "blanking_percent_of_peak_carrier": (low, high),
        "peak_white_percent_of_peak_carrier": (white_low, white_high),
        "black_above_blanking_percent": BROADCAST_BLACK_ABOVE_BLANKING_PERCENT,
        # the whole span the recorder's clipping levels are stated against,
        # expressed in the carrier terms the tuner works in
        "span_percent_of_peak_carrier": (
            BROADCAST_SYNC_PERCENT_OF_PEAK_CARRIER
            - 0.5 * (white_low + white_high)),
        "source": ("ITU-R BT.1701-1 Table 1 System M items 7 and 8, read from "
                   "the document; the same values appear in BT.470-6 Table 3 "
                   "items 7 and 8 quoted at docs/PROPAGATION_COLLAPSE.md:51"),
        "why": ("negative modulation puts the sync tip at peak carrier, so a "
                "receiver that overloads clips the sync pulse first"),
    }


def tuner_limit_in_span(overload_depth: float = 0.0) -> float:
    """Where a tuner overload lands in the RECORDER's own level span.

    The recorder's clipping levels are stated as percentages of the
    sync-tip-to-peak-white span measured FROM SYNC TIP, so that span is the
    common scale on which the two sites can be compared at all.

    A receiver that overloads flattens the top `overload_depth` of the
    carrier. Negative modulation puts the sync tip at the top, so in the
    recorder's span - which counts upward from the tip - the tuner's limit
    sits at exactly `overload_depth`, and at ZERO for a receiver just on the
    edge of overload. The recorder's dark clip sits at +0.40 of the same
    span.

    That difference is the entire separability argument, and it is stated as
    a number rather than an opinion: the two sites act on the same waveform
    at levels 0.40 of the span apart, so an excursion between them is
    touched by exactly ONE of them.
    """
    return float(overload_depth)


def sites(format_name: str = "VHS", overload_depth: float = 0.0
          ) -> Dict[str, Dict[str, object]]:
    """The two places a clip can be, with what distinguishes them.

    Ethan: *"It might be at the television tuner stage, the cd sample was an
    over the air recording."*

    It might, and the chain has to be able to say so, which is why this is a
    second SITE rather than the same entry moved. The recorder's input clip
    is a property of the machine; the tuner's is a property of the SOURCE,
    so it is present on material that came over the air and absent on
    material that did not - the countdown sample against the test-pattern
    tapes. `signatures` therefore emits the tuner entries only when the
    caller declares a broadcast source, because provenance decides whether
    the component exists at all.
    """
    levels = clip_levels(format_name)
    return {
        TUNER_PREFIX: {
            "position": TUNER_CHAIN_POSITION,
            "stage": "the receiver, before the recorder saw the signal",
            "limit_in_span": tuner_limit_in_span(overload_depth),
            "reaches_the_sync_tip": True,
            "why": ("negative modulation puts the sync tip at peak carrier, "
                    "so the receiver's headroom runs out there first"),
        },
        CHAIN_PREFIX: {
            "position": CHAIN_POSITION,
            "stage": "the recording machine's input",
            "limit_in_span": float(levels["dark"]),
            "reaches_the_sync_tip": False,
            "why": ("the standard's dark clip is 40 per cent ABOVE the tip, "
                    "so it takes the emphasized undershoot and never the "
                    "tip's own settled level"),
        },
    }


def site_separation(format_name: str = "VHS", overload_depth: float = 0.0
                    ) -> Dict[str, object]:
    """Whether the two sites can be told apart, as a measurement.

    THE SEPARABILITY LAW APPLIES HERE THE SAME WAY IT APPLIES EVERYWHERE
    ELSE IN THIS ARC: two members of a family are collinear when they differ
    in a RATE and separable when they differ in KIND. Two hard limiters
    acting on the SAME excursion differ only in threshold, which is a rate,
    and `composition_control` shows what that costs - they compose into one
    limiter at the tighter threshold and no measurement can split them.

    What makes these two separable is that they do not act on the same
    excursion. The recorder's dark clip sits 0.40 of the span above the sync
    tip; the tuner's sits at the tip. Everything between the two levels is
    touched by exactly one of them, and the sync pulse's own settled tip is
    in that window - which is why the tip is the tuner's witness and nothing
    else's.

    The separation closes only if the receiver overloads by more than the
    recorder's own dark clip, which would mean an overload deep enough to
    remove the whole sync pulse and 40 per cent of the picture range with it.
    """
    levels = clip_levels(format_name)
    recorder = float(levels["dark"])
    tuner = tuner_limit_in_span(overload_depth)
    window = recorder - tuner
    return {
        "tuner_limit_in_span": tuner,
        "recorder_limit_in_span": recorder,
        "window": window,
        "separable": bool(window > 0.0),
        "the_tip_is_in_the_window": bool(tuner <= 0.0 < recorder),
        "closes_at_overload_depth": recorder,
        "why": ("they act at levels 0.40 of the span apart, so an excursion "
                "between them is touched by exactly one; they become "
                "inseparable only at an overload that removes the whole "
                "sync pulse"),
    }


def composition_control(signal, first: float, second: float
                        ) -> Dict[str, object]:
    """THE CONTROL, and it is the one that can refuse the whole design.

    Two hard limiters in series are ONE hard limiter at the tighter
    threshold. If that is true then two clips at the same site, or at two
    sites acting on the same excursion, are not separately observable at
    all, however many fields are averaged - and any claim to have located a
    clip has to rest on the two acting at DIFFERENT levels rather than on
    fitting a second one.

    This checks the identity itself, so a design that quietly assumed the
    two could be split by fitting would fail here rather than in the
    measurement it corrupted.
    """
    values = np.asarray(signal, dtype=np.float64)
    a, b = abs(float(first)), abs(float(second))
    chained = np.clip(np.clip(values, -a, a), -b, b)
    single = np.clip(values, -min(a, b), min(a, b))
    worst = float(np.max(np.abs(chained - single))) if values.size else 0.0
    return {
        "tighter": min(a, b),
        "worst_difference": worst,
        "composes_to_the_tighter": bool(worst <= 1e-12),
        "why": ("two limiters in series are one limiter at the tighter "
                "threshold, so two clips on the same excursion cannot be "
                "separated by fitting - only by acting at different levels"),
    }


def describing_function(amplitude_over_limit) -> np.ndarray:
    """The effective gain a hard limiter presents to a sinusoid.

    For a limiter of threshold `L` driven by a sinusoid of amplitude `a`,
    the fundamental's gain is

        N(a/L) = 1                                             a <= L
        N(a/L) = (2/pi) [ asin(L/a) + (L/a) sqrt(1 - (L/a)^2) ] a >  L

    which is the standard quasi-linear result and is what makes a clipper
    enterable at all: below the limit it is the identity, above it a gain
    that falls with the drive.

    Returned as a real gain in [0, 1], never zero, so its logarithm exists
    and the entry is subtractable.
    """
    ratio = np.asarray(amplitude_over_limit, dtype=np.float64)
    inverse = np.divide(1.0, np.maximum(ratio, 1e-12))
    inside = np.clip(inverse, 0.0, 1.0)
    gain = np.where(
        ratio <= 1.0, 1.0,
        (2.0 / np.pi) * (np.arcsin(inside)
                         + inside * np.sqrt(np.maximum(1.0 - inside ** 2,
                                                       0.0))))
    return np.clip(gain, 1e-6, 1.0)


def soft_limit(signal, limit: float, sharpness: float = 4.0) -> np.ndarray:
    """A clipper as a real one is: a smooth saturation, not `min(x, c)`.

    A real clipper is a saturating amplifier rather than an ideal limiter,
    and a smooth saturation is differentiable everywhere, which a hard
    corner is not. This reaches the same asymptote; `sharpness` says how
    abruptly.
    """
    values = np.asarray(signal, dtype=np.float64)
    scale = float(limit)
    if not scale > 0:
        raise ValueError("the limit must be positive")
    return scale * np.tanh(sharpness * values / scale) / np.tanh(sharpness)


def signature(frequency_hz, level: float = 1.0,
              format_name: str = "VHS", edge: str = "dark") -> np.ndarray:
    """The clipper's signature at one drive level, as a complex response.

    Flat in frequency by construction, because a memoryless non-linearity
    has no frequency shape of its own - what it has is a LEVEL shape, and
    that is what `level_signature` below carries. This entry exists so the
    key holds the clipper's presence and its position in the chain; on its
    own it adds no direction, and it says so.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    levels = clip_levels(format_name)
    limit = abs(levels["dark" if edge == "dark" else "white"])
    gain = float(describing_function(float(level) / max(limit, 1e-12)))
    return np.full(grid.shape, gain, dtype=np.complex128)


def level_signature(frequency_hz, level: float = 1.0,
                    format_name: str = "VHS", edge: str = "dark",
                    step: float = 0.05,
                    reference_hz: Optional[float] = None) -> np.ndarray:
    """How the clipper's gain MOVES with the drive level: the informative
    part.

    A memoryless limiter is flat in frequency, so on the frequency axis
    alone it is nothing. What it has is a level dependence, and because the
    describing function is strongly curved through the limit that dependence
    changes shape as the level crosses it - which is what makes it a
    direction rather than a gain.

    The frequency dependence enters through the DRIVE: the pre-emphasis
    lifts the high frequencies, so a transient reaches the limit there
    first. `reference_hz` is the frequency at which the stated level applies,
    and above it the effective drive rises with the emphasis, which is why
    the luma's upper frequencies clip at all.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    levels = clip_levels(format_name)
    limit = abs(levels["dark" if edge == "dark" else "white"])
    reference = (float(reference_hz) if reference_hz
                 else float(np.median(grid)))
    # the emphasis lift, as the drive the clipper actually sees
    lift = np.sqrt(1.0 + (grid / max(reference, 1.0)) ** 2)
    above = describing_function((level + step) * lift / max(limit, 1e-12))
    below = describing_function((level - step) * lift / max(limit, 1e-12))
    return (np.maximum(above, 1e-12)
            / np.maximum(below, 1e-12)).astype(np.complex128)


def signatures(frequency_hz, level: float = 1.0,
               format_name: str = "VHS", source: str = "unknown",
               overload_depth: float = 0.0) -> Dict[str, np.ndarray]:
    """The clipper's entries for the key.

    The level dependence is the one that carries a direction; the flat
    entries carry each clipper's presence and its chain position.

    `source` IS PROVENANCE, AND IT DECIDES WHICH ENTRIES EXIST. A tuner is
    in the chain of material that came over the air and is not in the chain
    of material that did not, so the tuner entries are emitted only for
    `source="broadcast"`. This is the same discipline the rest of the key
    follows - an entry is declared when the physics that makes it is present,
    not because a slot exists for it - and it is the only honest way to hold
    a component that is real on one tape and absent on another.
    """
    entries = {
        "input clipping (dark)": signature(frequency_hz, level, format_name,
                                           "dark"),
        "input clipping (white)": signature(frequency_hz, level, format_name,
                                            "white"),
        "input clipping level dependence": level_signature(
            frequency_hz, level, format_name, "dark"),
    }
    if str(source).lower() in ("broadcast", "off-air", "over the air", "air"):
        grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
        limit = max(tuner_limit_in_span(overload_depth), 1e-6)
        gain = float(describing_function(float(level) / limit))
        entries["tuner clipping (sync tip)"] = np.full(
            grid.shape, gain, dtype=np.complex128)
        entries["tuner clipping level dependence"] = level_signature(
            frequency_hz, level, format_name, "dark")
    return entries


def level_axis(frequency_hz, levels=(0.5, 1.0, 1.5, 2.0, 3.0),
               format_name: str = "VHS", limited: bool = True
               ) -> Dict[str, object]:
    """How many independent directions the clipper's level axis provides.

    With `limited` false the describing function is replaced by unity, which
    is a clipper that never clips: the axis must then collapse to a single
    direction, and that is the control.
    """
    shapes = []
    for level in levels:
        if limited:
            value = level_signature(frequency_hz, level, format_name)
        else:
            grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
            value = np.ones_like(grid, dtype=np.complex128)
        logged = np.log(np.maximum(np.abs(value), 1e-12))
        logged = logged - logged.mean()
        norm = float(np.linalg.norm(logged))
        shapes.append(logged / norm if norm > 1e-9 else np.zeros_like(logged))
    stack = np.array(shapes)
    if not float(np.linalg.norm(stack)) > 0:
        return {"effective": 1.0, "levels": list(levels),
                "why": "a clipper that never clips has one shape: none"}
    singular = np.linalg.svd(stack, compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    return {
        "effective": float(1.0 / np.sum(share ** 2)),
        "count": len(levels),
        "levels": list(levels),
        "singular_values": singular,
        "gains": [float(describing_function(
            level / abs(clip_levels(format_name)["dark"])))
            for level in levels],
        "why": ("the describing function is strongly curved through the "
                "limit, so the level dependence changes shape as the level "
                "crosses it"),
    }


def truncated_ideal(ideal, limit: float, edge: str = "dark") -> np.ndarray:
    """THE IDEAL, TRUNCATED THE WAY THE RECORDING WAS.

    Ethan: *"If clipping is detected, the ideal shape will be truncated at
    the bottom."*

    This is the operational consequence, and it matters more than any
    direction the clipper contributes. Every differential in this method is
    `synthetic minus measured`. If the recording was clipped and the
    synthetic side was not, the difference contains the clip - and it is
    then attributed to whatever component is being measured, which is
    everything downstream of the recorder's input.

    The sync pulse is the case that bites. It is the largest excursion in
    the signal, it sits at the dark end, and it is therefore what a dark
    clip takes first. The specified pulse is the synthetic side of the final
    dimension, so a clipped recording compared against an unclipped spec
    reports a sync distortion that is really the recorder's input stage
    doing exactly what the standard says it does.

    `edge` says which end was truncated: "dark" clips from below, which is
    what takes the sync tip, and "white" from above.
    """
    values = np.asarray(ideal, dtype=np.float64).copy()
    bound = float(limit)
    if edge == "dark":
        return np.maximum(values, bound)
    if edge == "white":
        return np.minimum(values, bound)
    raise ValueError(f"edge must be 'dark' or 'white', not {edge!r}")


def truncation_error(ideal, limit: float, edge: str = "dark"
                     ) -> Dict[str, float]:
    """What comparing against an UNtruncated ideal would cost.

    Reports the share of the ideal that the clip removed and the size of the
    error a differential would carry if the synthetic side were left whole.
    That error is not noise: it is the same on every field, so it survives
    every average and reads as a real, reproducible defect.
    """
    values = np.asarray(ideal, dtype=np.float64)
    truncated = truncated_ideal(values, limit, edge)
    difference = values - truncated
    energy = float(np.vdot(values, values).real)
    return {
        "truncated_fraction": float(np.mean(difference != 0.0)),
        "error_share": (float(np.vdot(difference, difference).real / energy)
                        if energy > 0 else 0.0),
        "peak_error": float(np.max(np.abs(difference))),
        "why": ("the same on every field, so it survives every average and "
                "reads as a reproducible defect of whatever is measured"),
    }


def hard_limit_control(signal, limit: float) -> Dict[str, object]:
    """THE CONTROL: a hard limit is not a response at all.

    An earlier version of this control asserted that a hard clip fails
    `interference.subtractable` because its logarithm is undefined. That was
    wrong, and running it said so: a clipped SIGNAL is bounded away from
    zero as often as an unclipped one, and the check passed on both. The
    real objection is different and stronger.

    A hard limit is MEMORYLESS AND NON-LINEAR, so it has no frequency
    response to enter at all: the ratio of output to input is not a function
    of frequency but of the sample's own value, and two signals of different
    amplitude see different "responses" from the same clipper. What this
    control checks is that - that the apparent response depends on the drive
    - because an entry that passes it is not a clipper and an entry that
    fails it must be entered through its describing function instead.
    """
    values = np.asarray(signal, dtype=np.float64)
    bound = abs(limit)
    responses = []
    for drive in (0.5, 1.0, 2.0, 4.0):
        scaled = values * drive
        hard = np.clip(scaled, -bound, bound)
        live = np.abs(scaled) > 1e-9
        responses.append(float(np.mean(np.abs(hard[live] / scaled[live]))))
    spread = float(max(responses) - min(responses))
    return {
        "responses_by_drive": responses,
        "response_depends_on_drive": bool(spread > 0.05),
        "clipped_fraction": float(np.mean(np.abs(values) > bound)),
        "soft_is_subtractable": bool(_subtractable_soft(values, bound)),
        "why": ("a memoryless non-linearity has no frequency response; its "
                "apparent gain depends on the drive, which is why it must "
                "enter through a describing function"),
    }


def _subtractable_soft(values, bound: float) -> bool:
    from vhsdecode.models import interference as inf
    soft = soft_limit(values, bound)
    return bool(inf.subtractable((soft + 2.0 * bound).astype(np.complex128)))


# --------------------------------------------------------------------------
# Detecting a clip, measuring how deep it was, and handing that to the levels
# --------------------------------------------------------------------------


def detect_tip_clip(samples, reference, tolerance: float = 0.05,
                    min_run: int = 3, suppression: float = 2.0
                    ) -> Dict[str, object]:
    """IS THE SYNC TIP CLIPPED? The witness is noise, not flatness.

    AN UNCLIPPED SYNC TIP IS ALSO FLAT. Its whole job is to sit at one level
    between two edges, so a flat run at the extreme proves nothing at all,
    and a detector built on flatness would report every pulse ever recorded.

    What a clip does that nothing else does is remove the GAIN at that level:
    beyond the threshold the output does not move when the input does, so
    everything riding there - noise, ringing, residual content - is
    suppressed at that level AND NOWHERE ELSE. So the test is a ratio of
    scatter between the candidate run and a `reference` level of the same
    waveform that is known not to be clipped, and it is blind to any gain or
    offset applied to the whole pulse, which a depth test is not.

    The repository's own reading on the test tapes is the negative control:
    the bars sync tip is NOT clipped, at a tip/porch noise ratio of 1.31.

    `reference` is the unclipped comparison level - the back porch after the
    burst, per the standing constraint that the front porch is not a level.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    other = np.asarray(reference, dtype=np.float64).ravel()
    if values.size < min_run + 2 or other.size < 2:
        return {"clipped": False, "why": "too few samples to decide"}
    extreme = float(values.min())
    span = float(np.ptp(values)) or 1.0
    at_extreme = values <= extreme + float(tolerance) * span
    # the longest contiguous run sitting at the extreme
    best = run = 0
    for flag in at_extreme:
        run = run + 1 if flag else 0
        best = max(best, run)

    def scatter(block):
        if block.size < 3:
            return 0.0
        t = np.arange(block.size, dtype=float)
        return float((block - np.polyval(np.polyfit(t, block, 1), t)).std())

    tip_noise = scatter(values[at_extreme])
    reference_noise = scatter(other)
    ratio = tip_noise / max(reference_noise, 1e-12)
    clipped = bool(best >= int(min_run)
                   and reference_noise > 0
                   and ratio < 1.0 / float(suppression))
    return {
        "clipped": clipped,
        "run": int(best),
        "noise_ratio": float(ratio),
        "tip_noise": tip_noise,
        "reference_noise": reference_noise,
        "level": extreme,
        "mask": at_extreme,
        "why": ("a clipped level has no gain, so what rides on it is "
                "suppressed there and nowhere else; flatness alone is not a "
                "witness because an unclipped sync tip is flat too"),
    }


def extrapolate_through_clip(measured, model, clipped) -> Dict[str, object]:
    """HOW MUCH WAS CLIPPED, by continuing the model through the flat run.

    Ethan: *"We may be able to determine how much was clipped by
    extrapolating the model down to where the waves were clipped."*

    We can, and this is the whole of it. The model is fitted to the
    UNCLIPPED samples only - a gain and an offset, which is all a clip
    leaves free - and then evaluated inside the run the clip flattened. The
    depth is how far the fit goes past the level the recording stopped at.

    THE FIT MUST NOT SEE THE CLIPPED SAMPLES. Including them pulls the fit
    toward the flat level and shrinks the very depth being measured, which
    would make every clip look shallower the deeper it actually was. That is
    the failure this function is arranged to avoid, and `mask` is what
    avoids it.

    `model` is the unclipped shape the chain would have produced - the
    specified pulse through the measured response, not the specified pulse
    alone, because the overshoot on the approach to the tip is the channel's
    and a model without it under-predicts the excursion.
    """
    y = np.asarray(measured, dtype=np.float64).ravel()
    m = np.asarray(model, dtype=np.float64).ravel()
    flat = np.asarray(clipped, dtype=bool).ravel()
    if y.size != m.size or y.size != flat.size:
        raise ValueError("the measured shape, the model and the mask must be "
                         "on the same grid")
    free = ~flat
    if free.sum() < 3:
        return {"depth": 0.0, "why": "not enough unclipped samples to fit"}
    design = np.column_stack([m[free], np.ones(int(free.sum()))])
    (gain, offset), *_ = np.linalg.lstsq(design, y[free], rcond=None)
    recovered = gain * m + offset
    stopped_at = float(y[flat].mean()) if flat.any() else float(y.min())
    reached = float(recovered[flat].min()) if flat.any() else float(
        recovered.min())
    depth = float(stopped_at - reached)
    residual = float((y[free] - recovered[free]).std())
    return {
        "depth": max(depth, 0.0),
        "recovered": recovered,
        "stopped_at": stopped_at,
        "would_have_reached": reached,
        "gain": float(gain),
        "offset": float(offset),
        "fit_residual": residual,
        "fitted_on": int(free.sum()),
        "why": ("the model is fitted on the unclipped samples alone and "
                "evaluated inside the run; fitting it on the clipped ones "
                "would shrink the depth in proportion to the clip"),
    }


def ire0_correction(porch_level: float, tip_level: float, depth: float,
                    vsync_ire: float = -40.0) -> Dict[str, float]:
    """WHAT A DETECTED TIP CLIP COSTS THE LEVELS, AND THE FACTOR THAT FIXES
    IT.

    Ethan: *"If clipping is detected on the sync tip, transfer that knowledge
    to the ire_adjust code so it can correct the levels."*

    The decoder's `--ire0_adjust` in `hsync` mode derives its scale from two
    measured levels, at `vhsdecode/field.py:1170`:

        hz_ire = (porch - tip) / -vsync_ire

    A CLIPPED TIP MAKES THAT SCALE WRONG IN A DEFINITE DIRECTION. The clip
    moves the measured tip TOWARDS the porch, so the measured separation is
    short by the clip's depth, so `hz_ire` comes out too SMALL and every
    level derived from it is stretched. The correction is the depth put
    back:

        hz_ire_true = (porch - tip + depth) / -vsync_ire

    which is the measured scale times `1 + depth/(porch - tip)`. It is a
    multiplicative factor, so a caller can apply it without re-deriving
    anything, and it is exactly 1.0 when no clip is detected.

    The porch is the right reference for the second level, and not only by
    convention: the back porch after the burst is the one level in the line
    that a dark clip provably cannot reach, since the standard's dark clip
    sits 40 per cent of the span ABOVE the tip.
    """
    separation = float(porch_level) - float(tip_level)
    if not np.isfinite(separation) or separation == 0:
        return {"factor": 1.0, "hz_ire": float("nan"),
                "why": "degenerate separation; the decoder falls back"}
    measured = separation / -float(vsync_ire)
    corrected = (separation + float(depth)) / -float(vsync_ire)
    return {
        "hz_ire_measured": measured,
        "hz_ire_corrected": corrected,
        "factor": float(corrected / measured),
        "depth": float(depth),
        "separation": separation,
        "why": ("a clipped tip shortens the measured tip-to-porch "
                "separation, so hz_ire reads small and every level is "
                "stretched; the factor puts the clipped depth back"),
    }


def tip_clip_from_lines(tip_block, porch_block, suppression: float = 2.0,
                        min_lines: int = 8) -> Dict[str, object]:
    """THE CLIP WITNESS ON A WHOLE FIELD, per line and then pooled.

    The decoder has the field as a `(lines x samples)` view and measures its
    levels from it, so this takes the same two windows and answers the same
    question the single-pulse detector answers - but ACROSS PULSES, which is
    the form that survives the confounds.

    The scatter is measured WITHIN each line's window and then pooled by
    MEDIAN across lines, not by raveling the two blocks. Raveling would run a
    single detrend across line boundaries and count the line-to-line level
    variation as scatter, which is a different quantity and a larger one; the
    median pooling also discards the lines a dropout or a head switch moved,
    which is the same reasoning the decoder's own middle-third trim uses.

    Returns the ratio of tip scatter to porch scatter. A clipped level has no
    gain, so a ratio well below one is the witness.

    THE DEFAULT THRESHOLD IS DELIBERATELY NOT THE STATISTICALLY OPTIMAL ONE,
    and the reason is the whole difficulty of this measurement. On a
    synthetic null - equal Gaussian noise on both windows, 200 lines, 60
    samples - the ratio is extremely tight: mean 1.0006, standard deviation
    0.0107 over 200 draws, so 0.94 would be a six-sigma test. Calibrating on
    that would be wrong, because THE REAL UNCLIPPED RATIO IS NOT ONE. FM
    noise is level-dependent, so the tip and the porch do not carry the same
    noise even when neither is clipped, and the porch carries burst and
    content the tip does not. The repository's only measured unclipped
    control, the bars tape, reads 1.31 rather than 1.00.

    So the default `suppression` of 2.0 - a ratio below 0.5 - is set well
    clear of that unmodelled spread rather than at the edge of the synthetic
    null. It is not sensitive: measured against clip depth on the synthetic
    case, a ratio of 0.5 corresponds to a clip already touching about 60 per
    cent of the tip's samples, while a clip touching 16 per cent reads 0.86
    and would not fire. That is the correct trade until an unclipped
    population is measured on real tape and its spread is known - a false
    clip detection would put a depth correction into the levels of a
    recording that never lost anything, which is worse than missing a
    shallow clip.
    """
    tip = np.asarray(tip_block, dtype=np.float64)
    porch = np.asarray(porch_block, dtype=np.float64)
    if tip.ndim != 2 or porch.ndim != 2:
        raise ValueError("both blocks must be (lines x samples)")
    if tip.shape[0] < int(min_lines) or tip.shape[1] < 3 \
            or porch.shape[1] < 3:
        return {"clipped": False, "why": "too few lines or samples to decide"}

    def scatter(block):
        t = np.arange(block.shape[1], dtype=np.float64)
        # one least-squares line per row, vectorised
        centred = t - t.mean()
        denominator = float(np.dot(centred, centred)) or 1.0
        slope = (block - block.mean(axis=1, keepdims=True)) @ centred \
            / denominator
        fit = (block.mean(axis=1, keepdims=True)
               + slope[:, None] * centred[None, :])
        return np.std(block - fit, axis=1)

    tip_noise = scatter(tip)
    porch_noise = scatter(porch)
    pooled_tip = float(np.median(tip_noise))
    pooled_porch = float(np.median(porch_noise))
    ratio = pooled_tip / max(pooled_porch, 1e-12)
    return {
        "clipped": bool(pooled_porch > 0 and ratio < 1.0 / float(suppression)),
        "noise_ratio": float(ratio),
        "tip_noise": pooled_tip,
        "porch_noise": pooled_porch,
        "lines": int(tip.shape[0]),
        "why": ("scatter within each line's window, pooled by median across "
                "lines; a clipped level has no gain so what rides on it is "
                "suppressed there and nowhere else"),
    }


def clip_offset_from_porch(tip_levels, porch_levels,
                           minimum: int = 8) -> Dict[str, object]:
    """ETHAN'S RULE: THE BACK PORCH IS THE ANCHOR AND THE TIP IS THE SUSPECT.

    Ethan: *"the important thing is that the back porch is centered at 0 IRE.
    The sync tip may be clipped, and if flatness is detected that does not
    correlate with the back porch, this is the amount of offset needed to
    know the relative levels of the luma signal."*

    THIS IS A BETTER DETECTOR THAN THE NOISE RATIO ABOVE, and the reason is
    worth stating because it is not obvious. `tip_clip_from_lines` asks
    whether the tip is quiet; that is sound, but it needs the porch to be a
    fair comparator and the porch carries burst and content the tip does not.
    This asks instead whether the tip MOVES WITH the porch, which is immune
    to both channels having different noise, and immune to any gain applied
    to the whole line.

    THE LOGIC. Whatever moves the blanking level - deck AGC, tape speed,
    dropouts, the demodulator - moves the tip and the porch TOGETHER, because
    both are levels in the same signal. A clip breaks exactly that: beyond
    the limiter the tip cannot follow, so its correlation with the porch
    collapses while the porch goes on moving. Flatness alone is not the
    witness, because an unclipped tip is flat too; flatness THAT DOES NOT
    CORRELATE WITH THE PORCH is.

    AND THE SLOPE IS THE ANSWER, not merely the correlation. Regressing the
    tip on the porch gives a slope of 1 when the tip follows and 0 when it is
    pinned, so `1 - slope` is the fraction of the porch's movement the clip
    is swallowing, and the offset is what that costs at the working point.
    That offset is the quantity the levels need: with the porch anchored at
    0 IRE, it is what makes the luma's scale right - and, through the colour
    residual that survives in the luma, the chroma's with it.

    THE SLOPE AND NOT THE CORRELATION, and a measured case says why. Four
    populations of 200 paired levels, the porch wandering by 1.5 IRE rms:

        case                       slope   correlation   offset   verdict
        tip follows the porch      0.998      0.999      0.004    not clipped
        tip pinned (hard clip)     0.001      0.046      1.580    CLIPPED
        tip follows half           0.495      0.997      0.800    CLIPPED
        a gain change, not a clip  1.298      0.999     -0.471    not clipped

    THE PARTIAL CLIP IS THE CASE THAT DECIDES THE DESIGN. Its correlation is
    0.997 - indistinguishable from the unclipped population - so a detector
    reading correlation alone would call it clean. The slope reads 0.495 and
    catches it. Correlation asks whether the tip moves WITH the porch; the
    slope asks HOW MUCH, and only the second is the quantity a clip changes.

    The last row is the control that keeps it honest: a change of gain moves
    the tip MORE than the porch, slope 1.298, and is correctly not flagged,
    because a limiter can only ever reduce how much the tip follows and never
    increase it. A detector that fired there would be reporting the deck's
    AGC as a clip.

    Pass per-field or per-line level pairs measured the same way; both must
    come from the same fields in the same order.
    """
    tip = np.asarray(tip_levels, dtype=np.float64).ravel()
    porch = np.asarray(porch_levels, dtype=np.float64).ravel()
    if tip.size != porch.size:
        raise ValueError("the tip and porch levels must be paired")
    if tip.size < int(minimum):
        return {"clipped": False, "why": "too few paired levels to decide"}

    tip_c = tip - tip.mean()
    porch_c = porch - porch.mean()
    tip_spread = float(tip_c.std())
    porch_spread = float(porch_c.std())
    if not porch_spread > 0:
        return {"clipped": False,
                "why": "the porch does not move, so nothing can fail to "
                       "follow it"}
    correlation = float(np.corrcoef(tip, porch)[0, 1]) \
        if tip_spread > 0 else 0.0
    slope = float(np.dot(tip_c, porch_c) / np.dot(porch_c, porch_c))
    # the movement the tip failed to follow, at the porch's own working point
    unfollowed = (1.0 - slope) * porch_spread
    return {
        "clipped": bool(slope < 0.5 or abs(correlation) < 0.5),
        "slope": slope,
        "correlation": correlation,
        "tip_spread": tip_spread,
        "porch_spread": porch_spread,
        "followed_fraction": slope,
        "offset_ire": float(unfollowed),
        "samples": int(tip.size),
        "why": ("everything that moves a level moves both; a clip breaks "
                "that, so the slope of tip on porch is the fraction the tip "
                "still follows and 1 - slope is what the limiter swallows"),
    }
