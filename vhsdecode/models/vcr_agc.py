"""The VCR's AGC: a gain referenced to a level that may be clipped.

Ethan: *"Remember the clipping constraint, and AGC may also be a concern
here, since this shifts the frequency response of the composite channel as it
relates to the television standard ... and the underlying video standard."*

Both halves are right and they are the same mechanism seen twice.

WHERE IT IS, CORRECTED. Ethan: *"AGC happens in the VCR."* This module first
placed it in the receiver on the strength of `docs/PROPAGATION_COLLAPSE.md`
row 13, which describes a receiver AGC following the sync tip. That row is
real but it is not the loop that matters here, and two things settle it. The
specifications are silent - BT.1700 defers every radio-frequency
characteristic to BT.1701, and BT.1701-1 Table 1, read in full, gives the
radiated level map, the sidebands, the sound carrier, the pre-emphasis and
the group-delay precorrection while stating NOTHING about receiver AGC,
clipping, limiting or overload. And the VCR's own AGC is the one in the
signal path this repository actually decodes.

Position 8: the recording machine's automatic gain, acting on the incoming
video AFTER the transmission path at 1 to 7 and BEFORE the input clip at 9.
That order is the physical one - input, gain, limiting, emphasis - and it is
why the two cannot be folded together: THE AGC SETS THE LEVEL AT WHICH THE
CLIP IS REACHED.

WHY IT STILL FOLLOWS THE SYNC TIP. Negative modulation puts the tip at peak
carrier (BT.1701-1 Table 1 item 8, and BT.470-6 Table 3 items 7 and 8), so
the tip is the one part of the signal whose amplitude is known a priori. That
makes it the natural reference wherever the loop sits, and every consequence
below follows from the reference and not from the location.

THE FIRST CONSEQUENCE IS A LEVEL ERROR, AND IT IS WHERE THE CLIPPING
CONSTRAINT BITES. An AGC referenced to the sync tip drives the tip to its
nominal amplitude. IF THE TIP IS CLIPPED, THE LOOP DRIVES A LEVEL THAT IS NOT
THE TIP, and it succeeds: the tip comes out looking exactly right while
everything else in the picture is scaled by the reciprocal of what the clip
removed. The error is invisible in the one place it is measured. That is why
a clip detector cannot rest on the tip's DEPTH - the AGC has already
normalised it - and has to rest on the tip's NOISE, which the loop does not
touch. See `vhsdecode.models.clipping.detect_tip_clip`.

The size is exact and it is not small: a tip clipped by a fraction `d` of the
sync-tip-to-peak-white span leaves the AGC setting a gain of `1/(1-d)`, so
every level above blanking is stretched by that factor while the standard's
own levels are unmoved. At the 40 per cent the format's dark clip sits at,
that would be a 67 per cent error - which is why a clip anywhere near that
depth is not a subtle defect but a broken picture, and why the shallow clips
this arc is looking for are the interesting case.

THE SECOND CONSEQUENCE IS THE FREQUENCY SHIFT ETHAN NAMES, AND IT IS REAL
RATHER THAN LOOSE LANGUAGE. A gain that varies in time is not a filter, but
its effect on a periodic signal is a filter's: the loop's own response is a
HIGH-PASS on the level, because it suppresses exactly the slow amplitude
changes it is built to remove, with a corner at `1/(2 pi tau)`. A receiver
whose AGC time constant is comparable to a field therefore removes real
field-rate picture content along with the fading it was meant to track, and
that removal IS a change in the composite channel's low-frequency response.

The corner lands in a band this arc has only one other instrument for.
Measured against the probes in `sync_geometry`:

    AGC time constant     corner        reachable by
    1 us              159154.9 Hz       one line, the equalizing sequence,
                                        the whole interval
    20 us               7957.7 Hz       the equalizing sequence, the
                                        whole interval
    100 us              1591.5 Hz       NOTHING
    1 ms                 159.2 Hz       NOTHING
    50 ms                  3.2 Hz       NOTHING

Two things in that table are worth reading carefully, because both are the
opposite of the intuition and both are `where_the_corner_lands`'s own output
rather than an assertion here.

A SINGLE SYNC PULSE REACHES NONE OF THEM, not even the 1 us loop: the pulse
resolves 212.8 kHz and that loop corners at 159.2 kHz, which is below it. And
the vertical interval, the longest probe available, resolves 1748 Hz and so
STOPS SHORT OF A 100 us LOOP at 1591 Hz. Every AGC slow enough to be doing
the job it exists for - tracking fading, which is a matter of milliseconds -
is invisible to every probe in the sync area. That is a real limit of the
instrument and not a gap to be filled by a longer fit: the information is not
there. It also means an AGC cannot currently be blamed for a residual, which
cuts both ways.

WHERE IT SITS, AND WHY THE POSITION IS A SIMPLIFICATION STATED AS ONE. The
AGC is a FEEDBACK LOOP: its gain is applied at IF, ahead of the overload that
clips the tip, while its reference is taken after the detector, behind it. A
loop does not have a chain position, and pretending otherwise would be the
kind of quiet assumption this arc keeps catching. Position 7 is where its
OBSERVABLE effect enters - a gain modulation on the detected video - and the
docstring says plainly that the gain itself acts earlier.
"""

from typing import Dict, Optional

import numpy as np

# ITU-R BT.470-6 Table 3 System M items 7 and 8, quoted at
# docs/PROPAGATION_COLLAPSE.md:51 - the same citation `clipping` uses, and
# for the same reason: the tip is at peak carrier, so it is the reference.
SYNC_PERCENT_OF_PEAK_CARRIER = 100.0

CHAIN_PREFIX = "vcr agc"
CHAIN_POSITION = 8

COMPONENT_POSITIONS: Dict[str, int] = {
    CHAIN_PREFIX: CHAIN_POSITION,
    "vcr agc level response": CHAIN_POSITION,
}

# A representative span of receiver AGC time constants, used only to report
# where the corner lands.
#
# NO STANDARD STATES ONE, AND THAT IS NOW ESTABLISHED RATHER THAN ASSUMED.
# BT.1700 defines the baseband signal and defers every radio-frequency
# characteristic to BT.1701; BT.1701-1 Table 1 was then read in full and
# gives the radiated level map, the sidebands, the sound carrier, the
# pre-emphasis and the group-delay precorrection - and states NOTHING about
# receiver AGC, clipping, limiting or overload. The two documents that would
# carry it are both silent, so these remain an assumption and are labelled
# one everywhere they appear.
ASSUMED_TIME_CONSTANTS_US = (1.0, 20.0, 100.0, 1000.0, 50000.0)


def corner_hz(time_constant_us: float) -> float:
    """The loop's corner, `1/(2 pi tau)`. Below it the AGC removes level
    changes; above it they pass."""
    return float(1.0 / (2.0 * np.pi * max(float(time_constant_us), 1e-9)
                        * 1e-6))


def level_response(frequency_hz, time_constant_us: float) -> np.ndarray:
    """THE HIGH-PASS THE LOOP IMPOSES ON THE LEVEL.

    An AGC suppresses the amplitude changes it can track and passes the ones
    it cannot, so as a response on the level it is a first-order high-pass at
    `1/(2 pi tau)`. This is the "shift in the frequency response of the
    composite channel" - not a metaphor, the loop's own closed-loop transfer.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    corner = corner_hz(time_constant_us)
    ratio = grid / max(corner, 1e-12)
    return ((1j * ratio) / (1.0 + 1j * ratio)).astype(np.complex128)


def reference_error(clip_depth_in_span: float) -> Dict[str, float]:
    """WHAT A CLIPPED TIP COSTS WHEN THE AGC IS REFERENCED TO IT.

    The loop drives the measured tip to nominal. A clip removes `d` of the
    sync-tip-to-peak-white span from that tip, so the loop compensates by
    `1/(1-d)` and every other level is stretched by the same factor - while
    the tip itself comes out looking correct, which is what makes the error
    invisible where it is measured.

    THE CONSEQUENCE FOR THE DETECTOR IS THE POINT: a clip cannot be found by
    comparing the tip's depth against the standard, because the loop has
    already removed that difference. It has to be found by the tip's NOISE,
    which no gain can restore.
    """
    depth = float(clip_depth_in_span)
    if depth >= 1.0:
        return {"gain_error": float("inf"), "depth": depth,
                "why": "a clip removing the whole span leaves no reference"}
    gain = 1.0 / (1.0 - depth)
    return {
        "gain_error": gain,
        "percent_error": 100.0 * (gain - 1.0),
        "depth": depth,
        "tip_looks_correct": True,
        "why": ("the loop drives the measured tip to nominal, so the error "
                "is exactly absent from the one level it is read at; find "
                "the clip by the tip's noise, which no gain restores"),
    }


def line_rate_sidebands(time_constant_us: float,
                        line_rate_hz: float = 15734.264,
                        depth: float = 0.01, orders: int = 3
                        ) -> Dict[str, object]:
    """WHAT A WITHIN-LINE GAIN VARIATION ACTUALLY DOES - AND IT IS NOT A
    FREQUENCY RESPONSE.

    THIS REPLACES A FUNCTION THAT WAS WRONG, and the error is worth keeping
    because it is a whole class of mistake. That function returned
    `1 + strength * cos(2 pi f / f_H)` - a RIPPLE IN FREQUENCY - while its own
    docstring correctly argued that a gain varying in time convolves the
    spectrum. Those are different objects. A ripple of period `f_H` in the
    response is what an ECHO at delay `1/f_H` produces; a periodic gain
    produces REPLICAS of the whole spectrum at multiples of `f_H`. The prose
    described one thing and the code computed the other, and neither matched
    what the loop does.

    A gain that varies within the line is LINEAR PERIODICALLY TIME-VARYING,
    and an LPTV operator has no single `H(f)` at all - the same objection that
    keeps the clip out of the key as a response. Multiplying by a periodic
    `g(t)` convolves with its Fourier series, so the honest description is the
    set of REPLICATION WEIGHTS, which is what this returns.

    The depth falls as the loop slows: a time constant much longer than a line
    smooths the variation away, and a loop that cannot move within a line
    contributes nothing here. That is why this is separate from
    `level_response`, which IS a genuine LTI transfer on the level and is the
    only part of the AGC that belongs in the key as a response.
    """
    line_period_us = 1e6 / max(float(line_rate_hz), 1.0)
    survives = float(np.exp(-line_period_us
                            / max(float(time_constant_us), 1e-9)))
    strength = float(depth) * (1.0 - survives)
    # a first-order loop's periodic gain is dominated by its fundamental, so
    # the weights fall geometrically with the harmonic order
    weights = {0: 1.0}
    for order in range(1, max(int(orders), 1) + 1):
        weights[order] = 0.5 * strength * survives ** (order - 1)
    return {
        "replication_weights": weights,
        "offsets_hz": {order: order * float(line_rate_hz)
                       for order in weights},
        "depth": strength,
        "smoothed_away": survives,
        "is_a_frequency_response": False,
        "why": ("a periodic gain is linear PERIODICALLY TIME-VARYING and has "
                "no single H(f); it replicates the spectrum at multiples of "
                "the line rate, which is a different object from the ripple "
                "an echo makes"),
    }


def signatures(frequency_hz, time_constant_us: Optional[float] = None,
               line_rate_hz: float = 15734.264,
               depth: float = 0.01) -> Dict[str, np.ndarray]:
    """The AGC's entries for the key - ONE, not two.

    `time_constant_us` is NOT specified by any standard held here, so it has
    no default that could be mistaken for one: called without it, this
    returns nothing and says so through an empty result, the same way
    `colour_under.signatures` declines a band that cannot see its carrier.

    THE LINE-RATE MODULATION IS NO LONGER EMITTED, and that is the correction
    rather than an omission. It is linear periodically time-varying, so it has
    no frequency response to enter - see `line_rate_sidebands`, which returns
    what it actually is. Only the loop's closed-loop transfer on the LEVEL is
    an LTI object, and only that belongs in a key of frequency responses.
    """
    if time_constant_us is None:
        return {}
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    return {
        "vcr agc level response": level_response(grid, time_constant_us),
    }


def where_the_corner_lands(time_constants_us=ASSUMED_TIME_CONSTANTS_US
                           ) -> Dict[str, object]:
    """Which probe, if any, can see a loop of each time constant.

    Reported against `sync_geometry.probes`, so the answer is the same 1/T
    arithmetic the rest of that module rests on rather than a separate claim.
    """
    from vhsdecode.models import sync_geometry

    reach = sync_geometry.probes("NTSC")
    rows = []
    for tau in time_constants_us:
        corner = corner_hz(tau)
        visible = [name for name, probe in reach.items()
                   if probe["resolution_hz"] <= corner]
        rows.append({
            "time_constant_us": float(tau),
            "corner_hz": corner,
            "reachable_by": visible or ["nothing here"],
        })
    return {
        "rows": rows,
        "assumed": True,
        "why": ("a probe resolves no finer than 1/T, so a loop is visible "
                "only to a probe at least as long as its own time constant; "
                "the time constants are assumed, not specified"),
    }
