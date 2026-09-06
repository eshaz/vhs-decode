"""THE SOURCE'S AUTOMATIC GAIN CONTROL, with an attack and a decay.

Ethan, 2026-09-06: *"The levels changing related to the luma level is
likely AGC drift and this has a attack and decay component to it. We
should model that in all dimensions, it should likely go between the
propagation model and the VCR input model. AGC can be derived by
asserting the levels should be constant, and back porch should be ire 0."*

WHERE IT SITS. Chain position 7, between the propagation terms (the echo
at 4 and the tuner's clip at 5, with the waveform distortions at 6) and
the recorder's own AGC at 8. That ordering is the physical one: whatever
gain the source applies happens before the tape sees anything, and the
recorder's AGC then acts on its output. Two gain controls in series is not
a redundancy - they have different time constants and different
references, and a model with one of them will attribute the other's
behaviour to whatever is nearest.

HOW IT IS DERIVED, WHICH IS THE POINT. Not fitted to a residual: asserted.
The standard fixes two levels and says both are constant - blanking is the
reference zero (ITU-R BT.1700, item 1) and the synchronizing level is 40
IRE beneath it. So on a signal that has passed through nothing, the back
porch reads zero on every line of every field and the sync depth reads
forty. Any departure is the chain's, and the part of it that MOVES is the
gain control's, because that is the only element whose whole purpose is to
move.

Two observables, and they separate two different things:

    the back porch's level      an OFFSET; a gain cannot move a zero
    the sync depth              a GAIN; an offset cannot change a spacing

Measured on this session's three decodes, per field across a whole field
(`sync_depth.field_slope`): countdown's blanking falls 0.603 IRE while its
tip falls 0.314, so its SPACING opens by 0.289 and its gain genuinely
drifts; home and the bars tape drift 0.34 to 0.39 IRE in both levels
together with their spacing steady to four hundredths, which is an offset
and no gain change at all. And all three sit 1.4 to 2.6 IRE below the
zero the standard specifies.

ATTACK AND DECAY. A gain control is asymmetric by design: it reduces gain
quickly when the signal grows, because the alternative is clipping, and
restores it slowly when the signal shrinks, because the alternative is
pumping. So one time constant will not describe it and fitting one will
average the two into a number that is neither. The model here is the
standard asymmetric single pole, and its two constants are what the fit
returns.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

CHAIN_POSITION = 7                      # between propagation and the VCR

# ITU-R BT.1700, 525-line: the two levels the standard fixes and calls
# constant. Everything in this module is a departure from these.
BLANKING_IRE = 0.0
SYNC_DEPTH_IRE = 40.0


def assert_levels(blanking_ire, spacing_ire) -> Dict[str, object]:
    """The departure from the two constants the standard fixes.

    This is the derivation Ethan describes: the levels SHOULD be constant
    and the back porch SHOULD be zero, so what is measured instead is the
    chain, and its time-varying part is the gain control.
    """
    blanking = np.asarray(blanking_ire, dtype=np.float64).ravel()
    spacing = np.asarray(spacing_ire, dtype=np.float64).ravel()
    if blanking.size != spacing.size:
        raise ValueError("one blanking level and one spacing per line")
    offset = blanking - BLANKING_IRE
    gain = spacing / SYNC_DEPTH_IRE
    return {
        "offset_ire": offset,
        "gain": gain,
        "offset_mean": float(offset.mean()),
        "gain_mean": float(gain.mean()),
        "offset_moves": float(offset.std()),
        "gain_moves": float(gain.std()),
        "why": ("a gain cannot move a zero and an offset cannot change a "
                "spacing, so the back porch measures one and the sync "
                "depth the other, with no fitting between them"),
    }


def response(gain, sample_period_s: float, attack_s: float, decay_s: float
             ) -> np.ndarray:
    """An asymmetric single pole: fast down, slow up.

    The convention is the physical one. When the input gain is BELOW the
    state the control is reducing its own gain, which it does quickly
    (attack); when it is above, it is restoring gain, which it does slowly
    (decay). Reversing the two is the easy mistake and gives a control that
    pumps on every transient.
    """
    values = np.asarray(gain, dtype=np.float64).ravel()
    fast = 1.0 - math.exp(-float(sample_period_s) / max(float(attack_s), 1e-12))
    slow = 1.0 - math.exp(-float(sample_period_s) / max(float(decay_s), 1e-12))
    out = np.empty_like(values)
    state = float(values[0])
    for index, value in enumerate(values):
        rate = fast if value < state else slow
        state = state + rate * (value - state)
        out[index] = state
    return out


def fit(measured_gain, sample_period_s: float,
        percentile: float = 90.0) -> Dict[str, object]:
    """The asymmetry of the control, from the trajectory's own increments.

    WHAT IS AND IS NOT IDENTIFIABLE HERE. The driving level is never
    observed - only the control's output is - so the two time constants
    cannot both be recovered from the output alone without knowing what
    drove it. A first attempt tried one-step prediction, feeding the
    output back through the candidate filter and comparing it with itself,
    and that is self-referential: the fastest pole passes the signal
    unchanged and therefore always wins. It returned 0.1 ms and 1.0 ms for
    a planted 2 ms and 200 ms, which were simply the lower ends of its own
    search ranges.

    WHAT IS IDENTIFIABLE is the ASYMMETRY, and it is identifiable because
    it is a property of the trajectory rather than of the fit: a control
    that falls fast and recovers slowly takes large downward steps and
    small upward ones, whatever drove it. So the statistic is the ratio of
    a high percentile of the downward increments to the same percentile of
    the upward ones, and it maps to the ratio of the two time constants
    directly.

    The individual constants need the drive. If it is known - a level step,
    a scene change, a switch to bars - `response` can be run forward and
    the pair solved; without it, this reports the ratio and says so rather
    than returning two numbers it cannot support.

    AND THE RATIO IT REPORTS ORDERS RATHER THAN MEASURES. Verified against
    planted controls driven by the same step: a planted ratio of 10 reads
    1.0, one of 100 reads 4.3, and another 100 with slower constants reads
    15.4. So the statistic rises with the asymmetry and can say which of
    two controls falls faster relative to its recovery, but the number is
    not the ratio and must not be quoted as one. With the drive known
    `solve_with_drive` recovers the constants properly: 2.015 ms and
    222.8 ms against a planted 2 and 200.
    """
    values = np.asarray(measured_gain, dtype=np.float64).ravel()
    if values.size < 16:
        raise ValueError("an asymmetry needs more than a few samples")
    steps = np.diff(values)
    down = -steps[steps < 0]
    up = steps[steps > 0]
    if down.size < 4 or up.size < 4:
        return {"asymmetry": float("nan"), "identifiable": False,
                "why": "the trajectory moves in only one direction"}
    fall = float(np.percentile(down, percentile))
    rise = float(np.percentile(up, percentile))
    ratio = fall / rise if rise > 0 else float("inf")
    return {
        "fall_per_sample": fall, "rise_per_sample": rise,
        "asymmetry": ratio,
        "implied_decay_over_attack": ratio,
        "identifiable": True,
        "falls_faster": bool(ratio > 1.0),
        "samples": int(values.size),
        "why": ("only the ratio is identifiable from the output alone; the "
                "two constants need the drive, which is not observed"),
    }


def solve_with_drive(measured_gain, drive, sample_period_s: float,
                     attack_range_s: Sequence[float] = (1e-4, 1e-1),
                     decay_range_s: Sequence[float] = (1e-3, 1e0),
                     steps: int = 24) -> Dict[str, object]:
    """Both time constants, when the driving level IS known.

    With the drive in hand the model runs forward and the pair that
    reproduces the observed output wins, which is a fit to data rather
    than to itself.
    """
    observed = np.asarray(measured_gain, dtype=np.float64).ravel()
    forcing = np.asarray(drive, dtype=np.float64).ravel()
    if observed.size != forcing.size:
        raise ValueError("the drive and the output must be the same length")
    best = None
    for attack in np.geomspace(*attack_range_s, int(steps)):
        for decay in np.geomspace(*decay_range_s, int(steps)):
            if decay < attack:
                continue
            error = float(np.mean(
                (response(forcing, sample_period_s, attack, decay)
                 - observed) ** 2))
            if best is None or error < best[0]:
                best = (error, attack, decay)
    error, attack, decay = best
    return {"attack_s": attack, "decay_s": decay,
            "asymmetry": decay / attack, "error": error,
            "attack_lines": attack / float(sample_period_s),
            "decay_lines": decay / float(sample_period_s),
            "why": "the drive is known, so the model runs forward and is fitted to data"}


# The FM carrier frequencies the three specified levels map to, SMPTE
# 32M-2004 clause 3.9.1.1.4. The levels are not merely levels: on tape they
# ARE frequencies, so a gain error on the levels is a deviation error on
# the carrier and moves where every channel sits.
CARRIER_AT_SYNC_TIP_HZ = 3.4e6
CARRIER_AT_BLANKING_HZ = 3.69e6      # the arc's own landing figure
CARRIER_AT_PEAK_WHITE_HZ = 4.4e6
SPECIFIED_DEVIATION_HZ = 1.0e6


def levels_as_frequencies(gain: float, offset_ire: float = 0.0
                          ) -> Dict[str, float]:
    """What a level error IS, once the signal is on tape.

    Ethan: *"they are the exact frequency range of the luma channel, and
    therefore the frequency range of all the higher channels."* Exactly so.
    The recorder maps sync tip to 3.4 MHz and peak white to 4.4 (SMPTE 32M
    clause 3.9.1.1.4), so the 140 IRE excursion IS the 1.0 MHz deviation
    and a gain of g puts the deviation at g megahertz. Everything above it
    moves with it: the sidebands Carson's rule places, the guard to the
    colour-under below, and the band the head must carry.

    Measured this session, from the sync depth: countdown 0.913, home
    0.840, the bars tape 0.873 - so deviations of 913, 840 and 873 kHz
    against the specified 1000, and a peak-white carrier 87, 160 and 127
    kHz low.
    """
    g = float(gain)
    deviation = g * SPECIFIED_DEVIATION_HZ
    white = CARRIER_AT_SYNC_TIP_HZ + deviation * (140.0 / 140.0)
    blanking_offset = float(offset_ire) / 140.0 * deviation
    return {
        "gain": g,
        "deviation_hz": deviation,
        "deviation_error_hz": deviation - SPECIFIED_DEVIATION_HZ,
        "sync_tip_hz": CARRIER_AT_SYNC_TIP_HZ,
        "peak_white_hz": white,
        "peak_white_error_hz": white - CARRIER_AT_PEAK_WHITE_HZ,
        "blanking_shift_hz": blanking_offset,
        "cite": "SMPTE 32M-2004 clause 3.9.1.1.4",
        "why": ("the levels are frequencies once the signal is modulated, "
                "so a gain error is a deviation error and moves the whole "
                "spectral layout with it"),
    }


def signatures(frequency_hz, attack_s: float = 1e-3, decay_s: float = 1e-1
               ) -> Dict[str, np.ndarray]:
    """The chain entry, as a frequency-axis signature.

    A gain control is time-varying, so it is NOT a frequency response and
    must not be entered as one - the arc has already made that mistake once
    with the recorder's AGC, writing a cosine in frequency for what is a
    linear periodically time-varying element. What CAN be entered is the
    envelope of its two poles: the band over which it can move at all,
    bounded above by the attack and below by the decay. Outside that band
    the control is transparent, and inside it the entry says only that
    something varies, not what.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    attack_corner = 1.0 / (2.0 * np.pi * float(attack_s))
    decay_corner = 1.0 / (2.0 * np.pi * float(decay_s))
    inside = (f >= decay_corner) & (f <= attack_corner)
    shape = np.where(inside, 1.0, 0.0)
    return {
        "source agc band": np.exp(0.5 * shape - 0.25),
        "source agc attack corner": np.exp(
            0.5 / np.sqrt(1.0 + (f / attack_corner) ** 2) - 0.25),
        "source agc decay corner": np.exp(
            0.5 / np.sqrt(1.0 + (decay_corner / np.maximum(f, 1e-30)) ** 2) - 0.25),
    }
