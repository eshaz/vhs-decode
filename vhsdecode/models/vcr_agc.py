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

The corner lands in a band this arc long had only one other instrument for.
Measured against the probes in `sync_geometry`, and then against the
field-length probe those four stop short of (`field_probe`):

    AGC time constant     corner        reachable by
    1 us              159154.9 Hz       one line, the equalizing sequence,
                                        the whole interval, a field
    20 us               7957.7 Hz       the equalizing sequence, the
                                        whole interval, a field
    100 us              1591.5 Hz       A FIELD
    1 ms                 159.2 Hz       A FIELD
    50 ms                  3.2 Hz       NOTHING

Two things in that table are worth reading carefully, because both are the
opposite of the intuition and both are `where_the_corner_lands`'s own output
rather than an assertion here.

A SINGLE SYNC PULSE REACHES NONE OF THEM, not even the 1 us loop: the pulse
resolves 212.8 kHz and that loop corners at 159.2 kHz, which is below it. And
the vertical interval, the longest CONTIGUOUS run of sync, resolves 1748 Hz
and so stops short of a 100 us loop at 1591 Hz.

THE LAST THREE ROWS USED TO READ "NOTHING", AND THAT WAS THIS MODULE'S OWN
BLIND SPOT RATHER THAN THE SIGNAL'S. The reasoning was that no probe in the
sync area is longer than the vertical interval, and it was wrong because it
counted only contiguous runs. The sync tip and the back porch are read on
EVERY line, so a level series across a field is a probe of the field's own
length - 236 lines of this session's decodes is 15.0 ms, resolving 66.7 Hz,
twenty-six times the vertical interval's reach - and it is still sync-only,
because both levels sit in the reserved intervals. Two of the three "NOTHING"
rows are reachable after all.

What survives of the old reading is the last row and the caution behind it.
An AGC slow enough to track fading over tens of milliseconds is still
invisible, and the field probe's own verdict on this session's tapes is that
every gain drift it can see is SLOWER than it can resolve
(`time_constant_from_drift`: tau exceeds 11.27, 2.08 and 1.26 ms on the three
tapes, so on every one of them only the 50 ms constant survives). So an AGC still cannot be blamed for
that drift - but that is now a measured exclusion rather than an absence of
instrument, which is the difference between a limit and a gap.

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


def field_probe(lines_measured: int, system: str = "NTSC") -> Dict[str, float]:
    """THE PROBE THE SYNC AREA'S OWN PROBES STOP SHORT OF.

    `sync_geometry.probes` ends at the vertical interval, 572 us, because
    that is the longest CONTIGUOUS run of sync in the signal. But the level
    read on the sync tip and the back porch is available on EVERY line, so
    a level series across a field is a probe of the field's own length -
    and it is still sync-only, because both levels are read in the
    reserved intervals and never in the active picture.

    The extension is large. Measured on this session's three decodes, whose
    fields are 263 lines and whose level series run over the 236 lines from
    line 25 to the field's end (`sync_depth.field_slope`'s own window):

        probe                        length     resolves    reaches tau
        the line-sync pulse           4.70 us   212766 Hz    <= 0.75 us
        one whole line               63.56 us    15734 Hz    <= 10.1 us
        the equalizing sequence     190.67 us     5245 Hz    <= 30.3 us
        the whole vertical interval   572 us      1748 Hz    <= 91.0 us
        236 lines of a field       14999 us      66.67 Hz    <= 2387 us

    That is 26 times the vertical interval's reach, and it is the first
    probe in this arc that covers the 1 ms row of
    `ASSUMED_TIME_CONSTANTS_US` - which `where_the_corner_lands` reported
    as reachable by NOTHING for as long as the sync area's four probes
    were the whole list.

    The line period comes from `sync_geometry.LINE_PERIOD_US`, the same
    ruler `probes` uses, so the new row is commensurable with the four it
    joins rather than being a separate claim on a separate clock.
    """
    from vhsdecode.models import sync_geometry

    key = sync_geometry._system(system)
    count = int(lines_measured)
    if count < 3:
        raise ValueError("a drift needs at least three lines to be a slope")
    duration_us = count * sync_geometry.LINE_PERIOD_US[key]
    resolution = 1e6 / duration_us
    return {
        "lines": float(count),
        "duration_us": float(duration_us),
        "resolution_hz": float(resolution),
        "reaches_time_constant_us": float(1e6 / (2.0 * np.pi * resolution)),
    }


def relaxation_curvature(window_over_time_constant,
                         points: int = 2001) -> np.ndarray:
    """HOW FAR A ONE-POLE RELAXATION BENDS INSIDE A WINDOW, per unit of the
    FITTED SLOPE it shows there.

    This is the forward law the identification inverts. A loop relaxing
    with time constant `tau`, watched for `T`, traces a saturating
    exponential over `u` in [0, 1] with `x = T/tau`. Fit a straight line
    through it and what is left is the CURVATURE; divide by the fitted
    line's own drift and the amplitude - which is the disturbance, and is
    never observed - cancels. What survives is a function of `x` alone.

    THE NORMALISATION IS PER FITTED SLOPE, AND THAT IS FORCED rather than
    chosen. The quantity measured on a field is a fitted slope, so the law
    must be expressed against the same thing or the two do not compare. Per
    TOTAL excursion the law turns over near `x = 8` and falls away, because
    a saturating curve's excursion runs ahead of its slope; per fitted
    slope it is MONOTONE, so a measured sensitivity inverts to exactly one
    `x` and the two-branch ambiguity never arises. Measured on a 236-point
    window: 0.0038 at `x = 0.1`, 0.0374 at 1, 0.1428 at 4, 0.2562 at 8,
    0.9163 at 64.

    The small-window limit is exactly `x / (2 sqrt(180))` under either
    normalisation, because as `x` goes to zero the relaxation IS its own
    straight line. That is why a slow loop watched briefly looks perfectly
    straight: the bend falls off linearly in `T/tau` while the drift does
    not.

    `points` is the sampling the template is built on, and it matters at
    the fast end, where a bend confined to one sample is one sample wide
    whatever the grid. A caller comparing against measured data should pass
    that data's own length, which `time_constant_from_drift` does.

    Nothing here is fitted or tabulated: the ratio is computed from the
    exponential itself on each call, so the law carries no constant that
    could drift away from the physics it came from.
    """
    x = np.atleast_1d(np.asarray(window_over_time_constant,
                                 dtype=np.float64))
    out = np.zeros(x.shape, dtype=np.float64)
    for index, value in np.ndenumerate(x):
        if value <= 0.0:
            out[index] = 0.0
            continue
        out[index] = float(np.sqrt(np.mean(
            _bend_template(float(value), points) ** 2)))
    return out


def _bend_template(window_over_time_constant: float,
                   points: int) -> np.ndarray:
    """The SHAPE of the bend a one-pole relaxation leaves behind after a
    straight line is taken off it, PER UNIT OF THE FITTED SLOPE it shows.

    The normalisation is per fitted slope and not per total excursion, and
    that choice is forced: the quantity actually measured on a field is the
    straight line's own drift, so the law has to be expressed against the
    same thing or the two are not comparable. For a slow loop the two
    normalisations agree - as `x` goes to zero the relaxation IS its
    straight line - and the small-window limit is unaffected. For a fast
    one they part company sharply, because a saturating curve's fitted
    slope is far smaller than its total excursion.

    Getting this wrong is a sign error as well as a scale one: a relaxation
    that FALLS has a rising exponential's residual, so a template built on
    `exp(-x u)` and divided by a positive excursion predicts the bend
    upside down. That fault was present in the first version of this and
    was caught by asking the diagnostic to find a planted relaxation, which
    it reported at -2.04 where +1 was due.

    Its root-mean-square is `relaxation_curvature` of the same argument, so
    this is that scalar's vector form - what a fit needs in order to
    measure the bend rather than assume it absent."""
    x = float(window_over_time_constant)
    u = np.linspace(0.0, 1.0, int(points))
    if x <= 0.0:
        return np.zeros(u.size)
    shape = -np.expm1(-x * u) / (1.0 - np.exp(-x))
    design = np.vstack([np.ones_like(u), u]).T
    coefficients = np.linalg.lstsq(design, shape, rcond=None)[0]
    residual = shape - design @ coefficients
    return residual / (coefficients[1] if abs(coefficients[1]) > 1e-300
                       else 1e-300)


def curvature_limit(points: int = 2001) -> Dict[str, float]:
    """THE FASTEST LOOP THE WINDOW CAN STILL BE SAID TO WATCH, and the bend
    it would leave.

    Per fitted slope the curvature law is monotone, so any measured
    sensitivity inverts to some `x` and there is always a bound. What ends
    the scale is sampling, not the law: a loop with `x` equal to the sample
    count has relaxed inside ONE line, and past that the window is no
    longer watching a relaxation at all - it is watching a step, and the
    absence of a step is a different and stronger observation than the
    absence of a bend. So the search stops there and says so."""
    x = float(max(int(points), 4))
    return {"window_over_time_constant": x,
            "curvature_over_drift": float(relaxation_curvature(x, points)[0]),
            "why": ("a relaxation this fast is over within one line, so what "
                    "the window holds is a step rather than a bend")}


def time_constant_from_drift(spacing_ire, blanking_ire,
                             system: str = "NTSC",
                             sigma: float = 3.0) -> Dict[str, object]:
    """THE MEASUREMENT PAIR, AND THE TIME AXIS IT BUYS.

    Until this function the module's central quantity was assumed:
    `ASSUMED_TIME_CONSTANTS_US` lists five candidates spanning five decades
    and no measurement narrowed them, because `where_the_corner_lands`
    consulted only the sync area's four short probes and reported that
    nothing reached past 20 us. `field_probe` reaches 2.4 ms, and the pair
    below decides what is seen there.

    THE PAIR IS THE TWO LEVELS, AND THEY CAN DISAGREE. Both are specified
    constants - blanking at zero, the synchronizing level 40 IRE beneath it
    - so both are read on every line, in the reserved intervals only, and a
    slope in either is an error. But they are errors of DIFFERENT
    MECHANISMS, and that is what makes them a pair rather than one number
    read twice:

      the two drifting TOGETHER is a level shift. The spacing is unmoved,
      so the gain is unmoved, and the drift belongs to the direct-current
      restoration and not to this loop.

      the SPACING drifting is a gain drift, and a gain drift is what an
      automatic gain control is. This is the loop's own trajectory,
      observed.

    MEASURED, on this session's three decodes, over the 236 lines of
    `sync_depth.field_slope`'s window. The drift is stated ACROSS THAT
    WINDOW rather than scaled to a nominal field, because the window is
    what the curvature law below is watching; `field_slope` scales the same
    slopes by the full 263 lines, which is the factor 0.8935 between the
    two tables and not a disagreement.

        tape          blanking  sigma   spacing  scatter  sigma   gain drift
        countdown      -0.5386   19.5   -0.2577   0.0656   17.5   -0.706 %
        home           +0.3067    4.7   -0.0420   0.0508    3.7   -0.125 %
        pulse-and-bar  +0.3344   30.7   +0.0336   0.0590    2.5   +0.096 %

    The pair does disagree, and it disagrees differently on different
    tapes. Every tape moves its LEVEL, and significantly. Only countdown
    moves its GAIN by more than a fraction of a per cent, and its blanking
    moves in the opposite direction to the other two. On pulse-and-bar the
    spacing is not resolved at all at 2.5 sigma while its blanking is
    resolved at 30.7 - the sharpest form of the disagreement, one probe
    shouting and the other silent. The two probes are read on the same
    lines of the same fields by the same arithmetic, and they still part
    company - which is what a pair exists to produce, and what a single
    level read twice could never do.

    NOTE ON A SIGN. `sync_depth.field_slope`'s docstring table prints the
    spacing column as +0.289 / +0.047 / -0.037 and reads countdown's gain
    as opening; its code returns -0.2884 / -0.0470 / +0.0376, and the code
    is right - the spacing is `blanking - tip`, and -0.603 - (-0.314) is
    negative. Countdown's gain FALLS across the field by 0.72 per cent.
    The prose in that module and in `source_agc` has the sign inverted; the
    numbers here are the code's.

    WHAT THE CURVATURE THEN SAYS, and it is a bound rather than a value.
    A loop relaxing inside the window BENDS its trajectory, by
    `relaxation_curvature` of `T/tau` times the drift it shows. The
    measured spacing series are straight, so the bend is not resolved, and
    the exclusion runs the other way: any `tau` whose bend WOULD have stood
    `sigma` above the scatter is ruled out, which bounds `tau` BELOW.

    The test is on a curvature the data could actually resolve, not on the
    raw scatter. A template carried at the measured drift stands
    `|drift| * ratio * sqrt(N) / scatter` above the noise, so the smallest
    resolvable ratio is `sigma` divided by that:

        tape         resolvable ratio   T/tau at most   tau at least
        countdown        0.0497             1.331         11.27 ms
        home             0.2366             7.226          2.08 ms
        pulse-and-bar    0.3430            11.950          1.26 ms

    NONE of the three shows a bend, so on all three the time constant is
    bounded below - and every bound clears the 1 ms row. **Of the five
    assumed constants only the 50 ms one survives, on all three tapes
    independently.** That is a real narrowing of a five-decade assumption to
    a single decade, and it is the first measurement this module has ever
    had of its own central quantity.

    The three bounds differ by a factor of nine, and the ordering is the
    instrument's rather than the tape's: countdown's drift is six times
    home's against a similar scatter, so countdown resolves a bend six times
    finer and bounds six times harder. A tape with more drift is a better
    probe of the loop, which is the ordinary behaviour of a signal-to-noise
    ratio, and a useful check that the bound is not arriving from elsewhere.

    The reading is therefore not that a loop was measured. It is that
    whatever moves these levels across a field is SLOWER than a field can
    resolve, so it is not a loop relaxing inside the window - which is why
    this bound cuts the same way the module's opening argument does, and
    still does not let an AGC be blamed for the drift.

    The bound is conservative by construction: the scatter charged to the
    curvature is the whole residual, measurement noise included, so the
    true time constant can only be longer.
    """
    spacing = np.asarray(spacing_ire, dtype=np.float64).ravel()
    blanking = np.asarray(blanking_ire, dtype=np.float64).ravel()
    if spacing.size != blanking.size:
        raise ValueError("the two levels must be read on the same lines")
    probe = field_probe(spacing.size, system)
    index = np.arange(spacing.size, dtype=np.float64)
    span = index[-1] - index[0]

    def straight(series):
        slope, intercept = np.polyfit(index, series, 1)
        residual = series - (slope * index + intercept)
        scatter = float(residual.std(ddof=2))
        drift = float(slope * span)
        # the standard error of the fitted drift across the whole window
        centred = index - index.mean()
        error = scatter * span / np.sqrt(float(centred @ centred))
        return {"drift_ire": drift, "scatter_ire": scatter,
                "standard_error_ire": float(error),
                "sigma": float(abs(drift) / max(error, 1e-300))}

    differential = straight(spacing)
    common = straight(blanking)

    # WHICH RELAXATIONS THE SERIES RULES OUT, and the test is deliberately
    # the CONSERVATIVE one. A template of unit curvature-to-drift ratio,
    # carried at the measured drift, stands `|drift| * ratio * sqrt(N) /
    # scatter` above the noise, because the scatter is a per-sample figure
    # and the template is fitted over all N points. The smallest ratio that
    # would have shown at `sigma` is the reciprocal of that, and every
    # relaxation bending harder is excluded.
    #
    # THE WHOLE RESIDUAL IS CHARGED TO NOISE, which understates the
    # sensitivity on purpose: a real bend inflates the scatter it is
    # measured against, so it can only WEAKEN the bound, never strengthen
    # it. A lower bound that can only be too generous is the right shape
    # for the claim being made. Fitting the bend out instead was tried and
    # rejected - it turns a bound into a hypothesis test, and the
    # hypothesis it tests (a relaxation decaying from this window's own
    # start, carrying the whole drift) is narrower than the question.
    points = spacing.size
    limit = curvature_limit(points)
    drift = differential["drift_ire"]
    detectable = (abs(drift) * np.sqrt(float(points))
                  / max(differential["scatter_ire"], 1e-300))
    threshold = float(sigma) / max(detectable, 1e-300)
    if threshold >= limit["curvature_over_drift"]:
        x = None
    else:
        low, high = 1e-6, limit["window_over_time_constant"]
        for _ in range(120):
            middle = 0.5 * (low + high)
            if float(relaxation_curvature(middle, points)[0]) < threshold:
                low = middle
            else:
                high = middle
        x = 0.5 * (low + high)
    seconds = probe["duration_us"] * 1e-6
    bound_us = None if x is None else float(seconds / x * 1e6)

    survivors = [float(tau) for tau in ASSUMED_TIME_CONSTANTS_US
                 if bound_us is None or tau >= bound_us]
    gain_moves = differential["sigma"] >= float(sigma)
    level_moves = common["sigma"] >= float(sigma)
    if gain_moves and level_moves:
        moves = "both the gain and the level"
    elif gain_moves:
        moves = "the gain"
    elif level_moves:
        moves = "the level"
    else:
        moves = "neither, at this significance"
    return {
        "probe": probe,
        "differential": differential,
        "common_mode": common,
        "gain_drift_fraction": (differential["drift_ire"]
                                / float(np.mean(spacing))),
        "the_pair_disagrees": bool(gain_moves != level_moves),
        "gain_moves": bool(gain_moves),
        "level_moves": bool(level_moves),
        "which_moves": moves,
        "curvature_detectable_at": float(threshold),
        "curvature_limit": limit,
        "window_over_time_constant_at_most": x,
        "time_constant_us_at_least": bound_us,
        "assumed_constants_surviving_us": survivors,
        "why": ("blanking and the tip-to-blanking spacing are two probes of "
                "the same field that answer to different mechanisms - a "
                "level shift moves both together and leaves the spacing "
                "alone, a gain drift moves the spacing - so their "
                "disagreement separates the direct-current restoration "
                "from this loop; the spacing's straightness then bounds "
                "the loop's time constant from below, because a loop that "
                "relaxed inside the window would have bent it"),
    }


def where_the_corner_lands(time_constants_us=ASSUMED_TIME_CONSTANTS_US,
                           lines_measured: Optional[int] = None
                           ) -> Dict[str, object]:
    """Which probe, if any, can see a loop of each time constant.

    Reported against `sync_geometry.probes`, so the answer is the same 1/T
    arithmetic the rest of that module rests on rather than a separate claim.

    `lines_measured` adds the field-length probe, which is not in that list
    because it is not a contiguous run of sync - it is the same two levels
    read on every line of a field (`field_probe`). Without it the four
    sync-area probes stop at 20 us and the 100 us, 1 ms and 50 ms rows all
    read "nothing here", which is the state this module was in before the
    level series was recognised as a probe. With the 236 lines this
    session's decodes measure, the 1 ms row becomes reachable.
    """
    from vhsdecode.models import sync_geometry

    reach = dict(sync_geometry.probes("NTSC"))
    if lines_measured is not None:
        probe = field_probe(lines_measured)
        reach[f"{int(lines_measured)} lines of a field"] = {
            "duration_us": probe["duration_us"],
            "resolution_hz": probe["resolution_hz"],
            "time_scale": "long time",
        }
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
        "field_probe_included": lines_measured is not None,
        "why": ("a probe resolves no finer than 1/T, so a loop is visible "
                "only to a probe at least as long as its own time constant; "
                "the time constants are assumed, not specified"),
    }
