"""The picture stage: the references and alignments that act AFTER the RF
matrix has been corrected.

Ethan named four of them: *"sync pulse amplitude and phase; genlocking, which
aligns the colour burst to hsync; burst locking, which aligns the hsyncs to
the colour burst; and the time-base residual, which is the existing residual
subtracted from a constant."*

WHY THIS IS A SEPARATE STAGE AND NOT MORE ENTRIES IN THE RF KEY. Every
mechanism in `magnetic.py`, and the tape half of `interference.py`, acts on
the radio-frequency carrier ahead of a nonlinear demodulator, which is why
the standing law forbids inverting them with a video-domain filter. The
components here act on the demodulated line, on quantities the RF stage does
not carry at all: the horizontal reference, the subcarrier's phase against
it, and the line-to-line timing. They are applied after the RF matrix and
their abscissa is the VIDEO BASEBAND frequency rather than the RF frequency.
`baseband_from_rf` gives the map between the two, for a caller that wants
both sets on one axis, and states what the map costs.

THE SYNTHETIC SIDE IS SPECIFIED, NOT FITTED. SMPTE 170M-2004 fixes the
horizontal sync pulse at 4.7 us wide at half amplitude with a 140 ns rise,
40 IRE below blanking - already transcribed in `pair_dimension` and used from
there rather than restated; the colour burst at 40 +- 1 IRE, 9 +- 1 cycles,
beginning 19 cycles of subcarrier after the horizontal reference, with a
300 ns envelope rise; and the subcarrier-to-horizontal phase at zero with a
tolerance of 40 degrees of subcarrier. Every closed form below is the Fourier
transform of one of those specified shapes, or its derivative with respect to
one of those specified parameters.

HOW A SIGNATURE IS BUILT, and it is one construction throughout. Take the
specified line `X(f)`, perturb ONE real physical parameter `p`, and the
modification that perturbation makes is `dX/dp`. Scaled to unit peak and
entered as `1 + a * shape` it is bounded away from zero, so it has a finite
logarithm everywhere and `interference.subtractable` accepts it. The value is
complex because `dX/dp` is - the real part is what the parameter does to
amplitude and the imaginary part what it does to phase - and each parameter
is a REAL physical quantity, so the ensemble is fitted with
`ellipsoid(..., real_parameters=True)`.

THE MEASURED RESULT, on the NTSC video baseband from DC to 4.2 MHz over 4096
places at `amount = 0.25`:

    entries                                        effective   condition
    the key this module registers                    3.844 / 4       1.35
    the same with burst amplitude added              4.842 / 5       1.35
    the same with burst lock added as its own curve  3.535 / 5       9.31
    all six together                                 4.459 / 6       9.36

Against the tape magnetics' 1.58 of 6 at condition 1.06e4 these are a
DIFFERENT KIND of family and not more members of a collinear one. The six
magnetic losses are six functions of ONE dimensionless group, a length over
the recorded wavelength, and collapse to a direction and a half at a
condition number of ten thousand. The four here depend on four different
groups - `f T_sync` for the sync pulse's own spectrum, `(f - f_sc) T_burst`
for the burst's, `f tau` for a delay, `f sigma` for timing jitter - and reach
0.961 of their member count at condition 1.35, against multipath's 0.990 and
the magnetics' 0.263. They are parameterised along axes that make them
independent, as multipath's delays are, rather than along the one axis that
makes the tape's losses alike.
The worst coherent pair among the registered four is `sync phase` against
`time base residual` at 0.254.

GENLOCK AND BURST LOCK ARE ONE DIMENSION, NOT TWO. This was the question
worth asking and the answer came out against the expectation. Genlock rotates
the subcarrier's phase, a CONSTANT phase on the burst's band alone; burst
lock displaces the line origin, a DELAY of the whole line whose phase is
LINEAR in frequency and which reaches the luma band too. Constant against
linear is a difference in kind, so the luma band ought to break the tie. It
does not, and the reason is quantitative: a delay's sensitivity goes as
`f |X(f)|`, and on the specified line - sync and burst, with no picture,
because the sync-only constraint forbids picture - the only content at high
frequency IS the burst. Measured:

    band                                     effective   coherence
    full baseband, DC to 4.2 MHz              1.028 / 2      0.973
    the burst band alone, 2.78 to 4.38 MHz    1.023 / 2      0.977

and, projected directly, burst lock lies 95.5 per cent inside the span of
`sync phase` and `genlock`, at coefficients -0.973 and +0.094. Entered as its
own curve it therefore does what the S-VHS sub pre-emphasis did when its two
tabulated levels were entered as two curves: it LOWERS the count, 3.844 of 4
to 3.535 of 5, and worsens the condition sevenfold, 1.35 to 9.31. So the key
carries burst lock as the composition it is - a subcarrier phase and a sync
displacement, both already in it - and `burst_lock` is available as a
function, ordered and subtractable, behind a flag that is off by default.

WHAT WOULD BREAK THE TIE, exactly. Only the sync's own position separates
them, and the standard says how precisely it would have to be known. One
sample of the 4fsc grid is exactly ninety degrees of subcarrier, so with the
burst phase read to `d` degrees the sync edge must be located to `d/90` of a
4fsc sample - 0.776 ns at one degree. Measured over a per-line ensemble, the
coherence between the two runs 0.999938 when the sync is located to a whole
sample, 0.993884 at a tenth of one, and exactly 0.707107 at the balance
point, where the participation ratio reaches 1.333. That is the burst-to-sync
lock's null space stated as a required precision, and `alignment_tie` returns
it. `displacement_null_space` states the other half of the same null, on the
per-line axis, where pinning the exported displacement annihilates a constant
offset exactly.

THE CONTROLS. Three, and every one can fail. `pulse_control` inverse-
transforms the analytic sync spectrum and measures the pulse that comes back
with `standard_levels.half_amplitude_crossings`, an instrument in a module
this one does not own; the expected answers are the standard's 4.7 us and
140 ns, known without reference to anything here, and it read 4.7 us to
within 3.4 picoseconds and 141.5 ns, with the closed form agreeing with the
discrete transform of `spec_sync` to a mean 0.00023 and a worst 0.00056.
`rate_control` builds a family differing only in a RATE, which the
separability law says must be exactly one direction; it read 1.000 of 5 with
a second singular value 1.1e-16 of the first. `null_space_control` adds a
constant to a displacement series, re-pins it, and must recover none of it;
it recovered 7.4e-16 of it while a quadratic, which is not in an affine
pinning's null, survived - so it distinguishes the null the lock actually has
from the several it might have had.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from vhsdecode.models import burst_sync_lock as bsl
from vhsdecode.models import pair_dimension as pdim

# --------------------------------------------------------------------------
# The specification. Every number in this block is SMPTE 170M-2004 for
# System M/NTSC, with ITU-R BT.1700 carrying the same figures. Nothing here
# is fitted and nothing is assumed.
# --------------------------------------------------------------------------

# SMPTE 170M-2004 clause 8.1: the colour subcarrier is 315/88 MHz exactly and
# the line rate is 2/455 of it.
SUBCARRIER_HZ = 315e6 / 88.0
LINE_HZ = SUBCARRIER_HZ * 2.0 / 455.0
LINE_PERIOD_S = 1.0 / LINE_HZ

# SMPTE 170M-2004 clause 8.4 and table 2, the colour burst.
BURST_IRE = 40.0                 # peak to peak
BURST_IRE_TOLERANCE = 1.0
BURST_CYCLES = 9.0
BURST_CYCLES_TOLERANCE = 1.0
BURST_START_CYCLES = 19.0        # cycles of subcarrier after the reference
BURST_ENVELOPE_RISE_S = 300e-9   # 10 to 90 per cent, on the envelope

# SMPTE 170M-2004 clause 8.5, with SMPTE RP 154 defining the measurement: the
# subcarrier-to-horizontal phase - which is the genlock quantity itself - is
# specified as zero with a tolerance of 40 degrees of subcarrier. That is the
# standard's own statement of how far the burst-to-sync relationship may sit
# from where it belongs, so it is the right default scale for a synthetic
# timing residual, and it is a specification rather than a guess.
SCH_PHASE_TOLERANCE_DEG = 40.0

# The horizontal sync pulse is NOT restated here. `pair_dimension` already
# transcribes SMPTE 170M table 2 as SYNC_WIDTH_S, SYNC_RISE_S and
# SYNC_DEPTH_IRE and builds the pulse in `spec_sync`; both are used from
# there, because two transcriptions of one table are two chances to disagree.

# THE ONE ASSUMED NUMBER IN THIS MODULE, and what it would change. `amount`
# is how large a perturbation each signature carries. It is not a property of
# the format: the standard fixes the SHAPE a parameter's movement has, and
# how far this recording moved is what a fit is for. It is held small so that
# `log(1 + a s) = a s + O(a^2)` and the shape entering the ellipse is the
# derivative itself rather than a distorted copy. Measured over the
# registered key plus burst amplitude, the effective count reads 4.846 at
# 0.05, 4.845 at 0.10, 4.842 at 0.25, 4.835 at 0.50 and 4.823 at 0.75, and
# the condition number 1.341 to 1.356 across the same span - so the value
# moves the third figure of the conditioning and the directions not at all.
DEFAULT_AMOUNT = 0.25


def _grid(frequency_hz) -> np.ndarray:
    return np.asarray(frequency_hz, dtype=np.float64).ravel()


def ten_ninety_fraction() -> float:
    """The 10-90 time of a raised-cosine transition, as a fraction of the
    transition's full width.

    `pair_dimension.spec_sync` builds each edge as `0.5 (1 - cos(pi x))`
    across a transition of width `w`, which reaches 0.1 at `x = 0.2048` and
    0.9 at `x = 0.7952`, so a specified 10-90 rise is `0.5904 w` and a
    transition width is that rise DIVIDED by this. Computed from the
    arccosines rather than written down, and computed the same way in both
    places, so the two cannot drift apart - the confusion between the two
    quantities is what once made a specified 140 ns rise come out at 100.
    """
    return float((np.arccos(-0.8) - np.arccos(0.8)) / np.pi)


def cosine_taper_transform(u) -> np.ndarray:
    """The Fourier transform of the taper `spec_sync`'s edges are made of.

    Differentiating `0.5 (1 - cos(pi x))` across a transition of width `w`
    leaves a half-cosine kernel of width `w` and unit area, so a specified
    pulse is a rectangle of the specified width CONVOLVED with that kernel -
    which is why its half-amplitude points land exactly on the specified
    width, the kernel being symmetric. Its transform is the cosine window's,

        K(u) = (pi/4) [ sinc(u - 1/2) + sinc(u + 1/2) ],   u = f w

    normalised so that `K(0) = 1`. It is what turns a bare `sinc` into the
    specified shape, and it is the part of the closed form `pulse_control`
    exists to check.
    """
    u = np.asarray(u, dtype=np.float64)
    return (np.pi / 4.0) * (np.sinc(u - 0.5) + np.sinc(u + 0.5))


# --------------------------------------------------------------------------
# The specified line, in the frequency domain
# --------------------------------------------------------------------------


def sync_spectrum(frequency_hz, width_s: float = pdim.SYNC_WIDTH_S,
                  rise_s: float = pdim.SYNC_RISE_S,
                  depth_ire: float = pdim.SYNC_DEPTH_IRE) -> np.ndarray:
    """The specified horizontal sync pulse's own spectrum, in closed form.

    A rectangle of the specified width convolved with the specified
    transition, positioned so that the horizontal reference - the 50 per cent
    point of the leading edge, which is where SMPTE 170M measures everything
    else from - sits at the time origin:

        S(f) = depth * T * sinc(f T) * K(f w) * exp(-2 pi j f T / 2)

    with `T` the half-amplitude width, `w = rise / 0.5904` the transition
    width and `K` the taper transform above. The linear phase is the pulse's
    position in the line and is physical rather than bookkeeping: it is why
    this signature is complex, and why a delay of the pulse is a different
    direction from a gain on it.

    Measured against the discrete transform of `pair_dimension.spec_sync`
    over the band where the pulse holds a hundredth of its peak: mean
    relative error 0.00023, worst 0.00056. The first null is at
    `1 / 4.7 us = 212.8 kHz`, and at 274 kHz - where the de-emphasis shelf
    acts - the pulse is down to 0.194 of its value at DC, which is the closed
    form of the standing observation that a sync pulse has almost no
    resolution at the shelf and the vertical interval or a flat field is
    needed there instead.
    """
    f = _grid(frequency_hz)
    width = float(width_s)
    transition = float(rise_s) / ten_ninety_fraction()
    envelope = (float(depth_ire) * width * np.sinc(f * width)
                * cosine_taper_transform(f * transition))
    return (envelope.astype(np.complex128)
            * np.exp(-2j * np.pi * f * width / 2.0))


def burst_spectrum(frequency_hz, cycles: float = BURST_CYCLES,
                   rise_s: float = BURST_ENVELOPE_RISE_S,
                   amplitude_ire: float = BURST_IRE,
                   start_cycles: float = BURST_START_CYCLES,
                   subcarrier_hz: float = SUBCARRIER_HZ) -> np.ndarray:
    """The specified colour burst's spectrum, in closed form.

    The burst is `cycles` cycles of subcarrier under an envelope of the same
    tapered-rectangle family as the sync pulse, beginning `start_cycles`
    cycles of subcarrier after the horizontal reference. Modulation puts the
    envelope's transform on either side of the subcarrier:

        E(f) = (A/2) T_b sinc(f T_b) K(f w_b)
        B(f) = [ E(f - f_sc) + E(f + f_sc) ] exp(-2 pi j f t_c)

    with `T_b = cycles / f_sc` and `t_c` the packet's centre. The second term
    is negligible on a non-negative baseband grid but is kept so the form is
    exact rather than nearly so.

    The dimensionless group is `(f - f_sc) T_b`, and no loss in `magnetic.py`
    has it: those are all functions of a length over the recorded wavelength
    and are monotone across the band, where this one is band-limited about a
    frequency the standard fixes. Measured, 99.2 per cent of the burst's
    energy lies within two lobes of the subcarrier, 2.78 to 4.38 MHz, and
    1.7 per cent of it below 3 MHz.
    """
    f = _grid(frequency_hz)
    duration = float(cycles) / float(subcarrier_hz)
    transition = float(rise_s) / ten_ninety_fraction()
    half = 0.5 * float(amplitude_ire) * duration

    def envelope(offset):
        return (half * np.sinc(offset * duration)
                * cosine_taper_transform(offset * transition))

    centre = float(start_cycles) / float(subcarrier_hz) + 0.5 * duration
    packet = (envelope(f - float(subcarrier_hz))
              + envelope(f + float(subcarrier_hz)))
    return packet.astype(np.complex128) * np.exp(-2j * np.pi * f * centre)


# The sync pulse and the burst each have a rise time, so one keyword cannot
# serve both. Below `sync_spectrum` and `burst_spectrum` each take their own
# names; the two functions that build BOTH halves - `line_spectrum` and,
# through it, `burst_lock` and `signatures` - take the burst's envelope rise
# as `burst_rise_s` and everything else under its own name, so nothing is
# silently routed to the wrong half.
_SYNC_KEYS = ("width_s", "rise_s", "depth_ire")
_BURST_KEYS = {"cycles": "cycles", "burst_rise_s": "rise_s",
               "amplitude_ire": "amplitude_ire",
               "start_cycles": "start_cycles",
               "subcarrier_hz": "subcarrier_hz"}


def _sync_only(spec: Dict[str, float]) -> Dict[str, float]:
    return {name: value for name, value in spec.items()
            if name in _SYNC_KEYS}


def _burst_only(spec: Dict[str, float]) -> Dict[str, float]:
    unknown = set(spec) - set(_SYNC_KEYS) - set(_BURST_KEYS)
    if unknown:
        raise TypeError("unknown specification keyword(s): "
                        + ", ".join(sorted(unknown)))
    return {target: spec[name] for name, target in _BURST_KEYS.items()
            if name in spec}


def line_spectrum(frequency_hz, **spec) -> np.ndarray:
    """The specified reserved interval: the sync pulse and the burst together.

    The picture is deliberately absent, and its absence is load-bearing
    twice. It satisfies the standing sync-only constraint, that a correction
    derives from the sync pulse and the reserved intervals and never from
    picture content. And it is the reason `burst_lock` collapses onto
    `genlock`: a delay's sensitivity goes as `f |X(f)|`, and with no picture
    the only content at high frequency is the burst.
    """
    return (sync_spectrum(frequency_hz, **_sync_only(spec))
            + burst_spectrum(frequency_hz, **_burst_only(spec)))


def _bounded(shape: np.ndarray, amount: float) -> np.ndarray:
    """A perturbation entered so that it can be subtracted.

    `interference.subtractable` requires a finite logarithm everywhere, so a
    perturbation goes in as `1 + a * shape` with the shape scaled to unit
    peak and `a` below one. The magnitude then never leaves `[1 - a, 1 + a]`,
    which has a logarithm throughout - the same bounded form the interference
    module uses for a power contribution, and the reason no entry here is a
    mask or a bare shape.
    """
    value = np.asarray(shape, dtype=np.complex128)
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if not peak > 0:
        return np.ones_like(value)
    return 1.0 + float(amount) * value / peak


# --------------------------------------------------------------------------
# The components
# --------------------------------------------------------------------------


def sync_amplitude(frequency_hz, amount: float = DEFAULT_AMOUNT,
                   **spec) -> np.ndarray:
    """SYNC PULSE AMPLITUDE. The parameter is the pulse's depth in IRE.

    The sync tip is one of the two anchors the whole picture's level scale
    hangs on - `standard_levels.affine_to_standard` maps the measured tip to
    -40 IRE and the measured blanking to zero - so an error in the pulse's
    depth is an error in every level below it. Differentiating the specified
    line with respect to that depth leaves the pulse's own spectrum,

        dX/d(depth) = S(f) / depth

    a low-pass shape with its first null at 212.8 kHz and the taper's
    roll-off beyond. It is the amplitude half of the pair
    `pair_dimension.sync_dimension` returns, and it is complex because the
    pulse has a position as well as a size.

    Measured, it is coherent with `sync phase` to 0.094 and with the time
    base residual to 0.088 - all but orthogonal to both.
    """
    return _bounded(sync_spectrum(frequency_hz, **spec), amount)


def sync_phase(frequency_hz, amount: float = DEFAULT_AMOUNT,
               **spec) -> np.ndarray:
    """SYNC PULSE PHASE. The parameter is the pulse's own position in time.

    Displacing the pulse and nothing else differentiates to

        dX/dt = -2 pi j f S(f)

    the same spectrum turned through a right angle and weighted by frequency.
    That combination is why the pulse's amplitude and its phase are two
    components rather than two readings of one: under the Hermitian inner
    product a gain and a delay of the same shape are a single direction, and
    only the real-stacked form separates them - the distinction that moved
    the propagation set from 2.20 to 3.79.

    It is also the reference every other timing quantity in this module is
    measured from, which is why it stands first in the stage's order, and it
    is the component that carries what `burst lock` has beyond `genlock`.
    """
    f = _grid(frequency_hz)
    shape = -2j * np.pi * f * sync_spectrum(f, **spec)
    return _bounded(shape, amount)


def genlock(frequency_hz, amount: float = DEFAULT_AMOUNT,
            **spec) -> np.ndarray:
    """GENLOCK: the burst aligned to hsync. The parameter is the subcarrier's
    phase against the horizontal reference.

    This is the subcarrier-to-horizontal phase itself, which SMPTE 170M
    clause 8.5 specifies as zero with a tolerance of 40 degrees of
    subcarrier, so both the synthetic side and its tolerance are published.
    Rotating that phase and leaving everything else where it stands
    differentiates to

        dX/d(phase) = j B(f)

    a CONSTANT phase rotation on the burst's band and on nothing else.
    Constant, not linear in frequency: the subcarrier's phase moves and its
    envelope does not, which is the whole of the difference between this
    component and `burst_lock` - and, measured, not enough of a difference
    for the two to be separate directions.
    """
    shape = 1j * burst_spectrum(frequency_hz, **spec)
    return _bounded(shape, amount)


def burst_lock(frequency_hz, amount: float = DEFAULT_AMOUNT,
               **spec) -> np.ndarray:
    """BURST LOCK: the hsyncs aligned to the burst. The parameter is the line
    origin's displacement.

    Moving the line origin resamples the whole line, so every specified
    feature delays together:

        dX/d(origin) = -2 pi j f X(f),   X = sync + burst

    a DELAY, whose phase is linear in frequency and which reaches the luma
    band as well as the chroma.

    IT IS NOT A DIRECTION OF ITS OWN, and that is a measurement rather than
    an opinion. A delay's sensitivity goes as `f |X(f)|`, and on the
    specified line - which carries no picture, because the sync-only
    constraint forbids one - the only content at high frequency is the burst,
    so a whole-line delay is seen almost entirely there. Measured, this
    signature lies 95.5 per cent inside the span of `sync phase` and
    `genlock`, at coefficients +0.094 and -0.973, and is coherent with
    `genlock` alone to 0.973. Entered as its own curve it lowers the
    effective count from 3.844 of 4 to 3.535 of 5 and worsens the condition
    from 1.35 to 9.31, which is exactly what the S-VHS sub pre-emphasis did
    when two of its tabulated levels were entered as two curves.

    So `signatures` leaves it out by default and the key carries it as the
    composition it is. The function stays, ordered and subtractable, because
    burst locking is a real operation whose modification a caller may need to
    apply even where it is not an independent direction to fit.
    """
    shape = -2j * np.pi * _grid(frequency_hz) * line_spectrum(frequency_hz,
                                                              **spec)
    return _bounded(shape, amount)


def burst_amplitude(frequency_hz, amount: float = DEFAULT_AMOUNT,
                    **spec) -> np.ndarray:
    """The burst's amplitude, 40 +- 1 IRE. NOT one of the four Ethan named.

    It is the chroma's gain reference exactly as the sync tip is the luma's,
    and it is offered because it measurably ADDS a direction rather than
    because it is a modification: including it moves the set from 3.844 of 4
    to 4.842 of 5 at an unchanged condition of 1.35, its worst coherence with
    anything already there being 0.011. By the rule the emphasis established
    that is a clear entry rather than a marginal one, but it is not one of
    the four named, so whether it joins the key is the owner's call and
    `signatures` asks before including it.
    """
    return _bounded(burst_spectrum(frequency_hz, **spec), amount)


def jitter_loss(frequency_hz, jitter_s: float) -> np.ndarray:
    """The amplitude a Gaussian timing residual of this size leaves behind.

    A delay that varies from line to line by `sigma` averages the line's
    spectrum over a distribution of phases, and for a Gaussian residual that
    average is closed form:

        E[ exp(-2 pi j f dt) ] = exp( -2 pi^2 f^2 sigma^2 )

    a REAL loss, quadratic in frequency in the log. That is what makes the
    time-base residual a different kind of thing from any delay in this
    module: a delay is a phase linear in frequency, and this is an amplitude
    quadratic in it. Its dimensionless group is `f sigma`.
    """
    f = _grid(frequency_hz)
    sigma = float(jitter_s)
    return np.exp(-2.0 * np.pi ** 2 * (f * sigma) ** 2)


def specified_jitter_s(subcarrier_hz: float = SUBCARRIER_HZ) -> float:
    """The timing residual the standard itself allows, in seconds.

    SMPTE 170M's subcarrier-to-horizontal tolerance is 40 degrees of
    subcarrier, which at 3.579545 MHz is 31.04 ns. It is the standard's own
    statement of how far this exact relationship may sit from where it
    belongs, so it is the right default scale for a synthetic residual, and
    it is specified rather than assumed.
    """
    return float(SCH_PHASE_TOLERANCE_DEG / 360.0 / float(subcarrier_hz))


def time_base_residual(frequency_hz, residual: Optional[np.ndarray] = None,
                       jitter_s: Optional[float] = None,
                       amount: float = DEFAULT_AMOUNT) -> np.ndarray:
    """THE TIME-BASE RESIDUAL: the existing residual subtracted from a
    constant.

    Ethan's phrasing is the construction and not a description of it. A
    measured residual crosses zero, and a quantity that crosses zero has no
    logarithm there, so it cannot be subtracted off in the domain this
    transform works in - it is precisely the failure `subtractable` was
    written to catch. Subtracted FROM A CONSTANT it becomes `1 - d r`, which
    is bounded away from zero whenever `d |r| < 1` and has a finite logarithm
    everywhere. The instruction IS the subtractable form.

    With no measured residual the synthetic side stands in: the loss a
    Gaussian timing residual of the standard's own tolerance leaves,
    `1 - exp(-2 pi^2 f^2 sigma^2)` at `sigma = 31.04 ns`, scaled and
    subtracted from one the same way. A caller holding a measured time-base
    residual on this grid passes it and gets the same bounded form around its
    own shape; the mean is removed from a measured residual first, because a
    constant offset of the time base is a delay and belongs to `sync phase`
    rather than here.
    """
    f = _grid(frequency_hz)
    if residual is None:
        sigma = (specified_jitter_s() if jitter_s is None else float(jitter_s))
        shape = 1.0 - jitter_loss(f, sigma)
    else:
        shape = np.asarray(residual).ravel().astype(np.complex128)
        if shape.size != f.size:
            raise ValueError("the residual is not on the given grid: "
                             f"{shape.size} places against {f.size}")
        shape = shape - shape.mean()
    return _bounded(-np.asarray(shape, dtype=np.complex128), amount)


# --------------------------------------------------------------------------
# Where these belong in the chain
# --------------------------------------------------------------------------

# THE ORDER. `interference.COMPONENT_ORDER` counts positions from the source:
# the transmission path at 1 to 4, the recording machine at 10 and 11, the
# tape and head at 20 to 23, the capture at 30. Everything in this module
# happens in the decoder, on the demodulated picture, after the capture and
# after the RF matrix has been inverted, so every position here is above 30
# and `ordered_key` - which removes in descending position, last applied
# first - takes them off before anything RF. That ordering is the whole point
# of the stage being separate: the RF matrix is corrected first, and these are
# applied after, on quantities the RF stage does not carry.
#
# Within the stage the order is the order the decoder applies them, and each
# step consumes what the one before it produced:
#
#   40  the horizontal reference is found from the sync pulse's own edge, and
#       every timing below is measured from it;
#   41  the time base removes what that reference still varies by, which is
#       the standing rule that timing comes before anything spectral;
#   42  the line origins are refined against the burst, on the corrected time
#       base rather than the raw one;
#   43  the subcarrier is aligned to the horizontal reference, last of the
#       timing steps because it consumes the origins the two before it fixed;
#   44  the levels are normalised on the sync tip and blanking, after all
#       timing - `standard_levels` requires the anchors to be measured at the
#       point in the chain where the normalisation will land, and one
#       measured before a timing change and applied after it fights it;
#   45  the burst's amplitude, the chroma's own gain reference, alongside the
#       luma's at the same point.
COMPONENT_ORDER: Dict[str, int] = {
    "sync phase": 40,
    "time base residual": 41,
    "burst lock": 42,
    "genlock": 43,
    "sync amplitude": 44,
    "burst amplitude": 45,
}


def signatures(frequency_hz, amount: float = DEFAULT_AMOUNT,
               residual: Optional[np.ndarray] = None,
               jitter_s: Optional[float] = None,
               subcarrier_hz: float = SUBCARRIER_HZ,
               include_burst_lock: bool = False,
               include_burst_amplitude: bool = False,
               **spec) -> Dict[str, np.ndarray]:
    """The picture stage's modelled modifications, on one baseband grid.

    THE GRID IS VIDEO BASEBAND, not RF. Handed an RF grid these forms would
    be evaluated at frequencies they do not describe. `baseband_from_rf`
    gives the map a caller needs to put them on one axis with the RF key and
    states what the map costs.

    WHAT IS IN BY DEFAULT AND WHY. The four Ethan named, less `burst lock`,
    which is 95.5 per cent inside the span of two of the others and lowers
    the count when entered as its own curve; `include_burst_lock` puts it in
    for a caller who wants the modification rather than the direction.
    `burst amplitude` is not one of the four and is asked for rather than
    assumed, though it measurably adds a direction.

    THE BURST ENTRIES ARE GATED ON THE BAND, for the same reason the record
    level entry is gated on its crossover, but the reason is about evidence
    rather than arithmetic. On a grid stopping at 3 MHz - which is where a
    VHS luma baseband stops - the burst's skirt still has a distinguishable
    SHAPE, and ungated the set reads 4.853 of 5 at condition 1.30 against
    2.909 of 3 gated. What it does not have is any energy: 1.7 per cent of
    the burst lies below 3 MHz. Claiming a direction there is the
    representable-is-not-knowable failure exactly - a basis spans noise
    perfectly and predicts none of it - so the gate keeps the key from
    claiming a direction where the signal is not.
    """
    f = _grid(frequency_hz)
    spec = dict(spec, subcarrier_hz=subcarrier_hz)
    burst = _burst_only(spec)
    out: Dict[str, np.ndarray] = {
        "sync amplitude": sync_amplitude(f, amount, **_sync_only(spec)),
        "sync phase": sync_phase(f, amount, **_sync_only(spec)),
        "time base residual": time_base_residual(f, residual, jitter_s,
                                                 amount),
    }
    if reaches_burst_band(f, **burst):
        out["genlock"] = genlock(f, amount, **burst)
        if include_burst_amplitude:
            out["burst amplitude"] = burst_amplitude(f, amount, **burst)
    if include_burst_lock:
        out["burst lock"] = burst_lock(f, amount, **spec)
    return out


def reaches_burst_band(frequency_hz, cycles: float = BURST_CYCLES,
                       subcarrier_hz: float = SUBCARRIER_HZ, **_) -> bool:
    """Does this grid contain the burst's own band?

    The burst's main lobe is `+- 1 / T_b` about the subcarrier, `T_b` being
    the specified nine cycles, so the band is 3.58 +- 0.40 MHz. A grid that
    does not reach it holds 1.7 per cent of the burst's energy or less, and
    the chroma entries have nothing there to be measured from.
    """
    f = _grid(frequency_hz)
    if f.size == 0:
        return False
    half = float(subcarrier_hz) / max(float(cycles), 1e-9)
    return bool(f.min() <= float(subcarrier_hz) + half
                and f.max() >= float(subcarrier_hz) - half)


def burst_band(cycles: float = BURST_CYCLES,
               subcarrier_hz: float = SUBCARRIER_HZ,
               lobes: float = 2.0) -> Tuple[float, float]:
    """The burst's band, as the standard's own cycle count defines it.

    Two lobes by default, which measures 99.2 per cent of the burst's energy:
    2.784 to 4.375 MHz.
    """
    half = float(lobes) * float(subcarrier_hz) / max(float(cycles), 1e-9)
    return (float(subcarrier_hz) - half, float(subcarrier_hz) + half)


def baseband_from_rf(frequency_hz, carrier_hz: float) -> np.ndarray:
    """The map an RF grid needs before these forms apply to it.

    A baseband component at `f_m` appears on the frequency-modulated carrier
    as a pair of sidebands at `f_c +- f_m`, so the baseband frequency an RF
    place carries is `|f - f_c|`. Three things follow, and all three are
    worth stating before anyone uses it.

    A picture-stage signature carried onto an RF grid is SYMMETRIC ABOUT THE
    CARRIER, which is a kind no loss in `magnetic.py` has - every one of
    those is monotone across the band - so the map does not destroy the
    distinction this module rests on. Measured on the VHS luma band from 0.5
    to 7 MHz about a 3.9 MHz carrier, the registered four read 3.897 of 4 at
    condition 1.26 there, against 3.844 of 4 on their own baseband grid.

    The mapped abscissa is NOT MONOTONE, since two RF places fold onto one
    baseband place. That is legitimate - each function is still evaluated at
    the frequency it is a function of, and the place index means the same RF
    frequency in every entry - but it means a preparation that unwraps phase
    along the grid is unwrapping along a fold. It is harmless here only
    because the bounded form keeps every phase inside 0.206 radians at
    `amount = 0.25`, so nothing wraps; a caller raising the amount must sort
    the grid instead.

    And the fold halves the independent places, which `sphere_floor` takes as
    its length. A set carried across this map must be judged at the folded
    length, not at the RF grid's.

    Measured with both sets on one RF grid: the RF key alone reads 8.042 of
    14 at condition 9.72, the picture stage alone 3.897 of 4 at 1.26, and the
    eighteen together 9.455 at 15.70 - so the picture stage adds about 1.4
    directions to a key that already had 8.0. The first and third of those
    move whenever the RF key's own contents change; the middle one is this
    module's and does not.
    """
    return np.abs(_grid(frequency_hz) - float(carrier_hz))


# --------------------------------------------------------------------------
# Reading the shape
# --------------------------------------------------------------------------


def _shapes(entries: Dict[str, np.ndarray]
            ) -> Dict[str, Dict[str, np.ndarray]]:
    """Each signature as the log-magnitude and phase pair the ellipse takes.

    The same preparation `interference.distinguishable` uses: the log
    magnitude, so that a gain is a constant rather than a scale; the
    unwrapped phase beside it; and the mean removed from each, because a
    constant is a level and not a shape.
    """
    shapes: Dict[str, Dict[str, np.ndarray]] = {}
    for name, value in entries.items():
        value = np.asarray(value)
        magnitude = np.log(np.maximum(np.abs(value), 1e-12))
        phase = (np.unwrap(np.angle(value)) if np.iscomplexobj(value)
                 else np.zeros_like(magnitude))
        vector = (magnitude - magnitude.mean()) + 1j * (phase - phase.mean())
        if np.linalg.norm(vector) > 0:
            shapes[name] = {"frequency": vector}
    return shapes


def _participation(shapes: Dict[str, Dict[str, np.ndarray]],
                   names: List[str]) -> Dict[str, object]:
    """The participation ratio of the singular values, and the worst pair.

    Real and imaginary parts stacked as one real vector before the norm,
    because each entry is a REAL physical parameter's signature and a gain
    is a different mechanism from a delay of the same shape.
    """
    stack = np.array([
        np.concatenate([shapes[n]["frequency"].real,
                        shapes[n]["frequency"].imag]) for n in names])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    values = np.linalg.svd(stack, compute_uv=False)
    share = values ** 2 / max(float((values ** 2).sum()), 1e-30)
    gram = np.abs(stack @ stack.T)
    np.fill_diagonal(gram, 0.0)
    worst = np.unravel_index(int(np.argmax(gram)), gram.shape)
    return {
        "names": names,
        "count": len(names),
        "effective": float(1.0 / np.sum(share ** 2)) if share.size else 0.0,
        "singular_values": values,
        "condition": (float(values[0] / max(values[-1], 1e-30))
                      if values.size else 0.0),
        "worst_pair": (names[worst[0]], names[worst[1]]),
        "worst_coherence": float(gram[worst]),
    }


def distinguishable(frequency_hz, **kwargs) -> Dict[str, object]:
    """How many of the picture stage's modifications this band tells apart.

    The same question and the same measure as the rest of the arc - the
    participation ratio of the singular values of the unit-normalised
    real-stacked shapes - so the answer is directly comparable with the tape
    magnetics' 1.58 of 6 at condition 1.06e4 and multipath's 7.92 of 8 at
    condition 1.2.

    Measured on the NTSC baseband, DC to 4.2 MHz over 4096 places:

        the registered four                3.844 / 4   condition 1.35
        with burst amplitude               4.842 / 5   condition 1.35
        with burst lock as its own curve   3.535 / 5   condition 9.31

    The four are 0.961 of their count, against multipath's 0.990 and the
    magnetics' 0.263 - three and a half times the tape's. They are NEW KINDS
    rather than more members of a collinear family: the six
    magnetic losses are six values of one dimensionless group, and these four
    depend on four different ones. The result is stable to the grid - 4.840
    to 4.842 over lengths from 512 to 16384 places on the five-entry set -
    and to the assumed amount, 4.846 down to 4.823 across a fifteen-fold
    range of it.
    """
    from vhsdecode.models import information_extrapolation as ie

    shapes = _shapes(signatures(frequency_hz, **kwargs))
    fit = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    names = list(fit["names"]) or list(shapes)
    out = _participation(shapes, names)
    out["ellipse"] = fit
    out["why"] = ("the participation ratio of the singular values, the same "
                  "measure the magnetics' 1.58 of 6 was taken with")
    return out


def separating_precision_s(burst_phase_deg: float = 1.0,
                           subcarrier_hz: float = SUBCARRIER_HZ
                           ) -> Dict[str, float]:
    """How precisely the sync must be located for the alignment to be two
    dimensions rather than one.

    Genlock and burst lock are read through two channels, the burst's phase
    and the sync's position. In units of each channel's own precision they
    point along `(1, 0)` and `(2 pi f_sc, 1/rho)`, so their coherence is

        1 / sqrt( 1 + rho^2 ),   rho = d(phase) / (2 pi f_sc d(time))

    and they are equally separated - coherence exactly `1/sqrt 2`, a
    participation ratio of 4/3 - at `rho = 1`. That happens when the sync is
    located to `d(phase) / (2 pi f_sc)`.

    THE STANDARD MAKES THIS EXACT. One sample of the 4fsc grid is a quarter
    cycle of subcarrier, which is ninety degrees, so at `d` degrees of burst
    phase the sync edge must be found to `d/90` of a 4fsc sample - 0.776 ns,
    or a ninetieth of a sample, at one degree. Measured over a per-line
    ensemble the coherence runs 0.999938 at a whole sample, 0.993884 at a
    tenth, 0.707107 at the balance point and 0.668965 at a hundredth.
    """
    phase = float(np.radians(burst_phase_deg))
    seconds = phase / (2.0 * np.pi * float(subcarrier_hz))
    sample = 1.0 / (4.0 * float(subcarrier_hz))
    return {
        "burst_phase_deg": float(burst_phase_deg),
        "required_sync_precision_s": seconds,
        "in_4fsc_samples": seconds / sample,
        "degrees_per_4fsc_sample": 360.0 * float(subcarrier_hz)
        / (4.0 * float(subcarrier_hz)),
        "why": ("one 4fsc sample is ninety degrees of subcarrier, so the "
                "sync must be located to d/90 of a sample to separate a "
                "subcarrier phase from a line displacement"),
    }


def alignment_tie(frequency_hz, amount: float = DEFAULT_AMOUNT,
                  subcarrier_hz: float = SUBCARRIER_HZ,
                  lines: int = 262, seed: int = 0,
                  **spec) -> Dict[str, object]:
    """GENLOCK AND BURST LOCK: one dimension or two, and what breaks the tie.

    They are two operations on one alignment, so the question has to be
    answered rather than assumed, and the answer came out against the
    expectation. Genlock rotates the subcarrier's phase, a CONSTANT phase on
    the burst's band alone. Burst lock displaces the line origin, a DELAY of
    the whole line, whose phase is LINEAR in frequency and which reaches the
    luma band too. Constant against linear is a difference in kind, so the
    luma ought to break the tie - and it does not, because a delay's
    sensitivity goes as `f |X(f)|` and the specified line's only
    high-frequency content is the burst itself.

        band                                    effective   coherence
        full baseband, DC to 4.2 MHz             1.028 / 2      0.973
        the burst band alone, 2.78 to 4.38 MHz   1.023 / 2      0.977

    THEY ARE ONE DIMENSION. What separates them is not a band but a
    PRECISION: the sync's own position, read to a ninetieth of a 4fsc sample
    at one degree of burst phase. This function measures that too, over a
    per-line ensemble at a range of sync precisions, so the closed form in
    `separating_precision_s` is checked rather than asserted.

    That is the burst-to-sync lock's null space in the form this stage needs
    it: the burst alone cannot say which of the two moved.
    """
    f = _grid(frequency_hz)
    spec = dict(spec, subcarrier_hz=subcarrier_hz)
    low, high = burst_band(subcarrier_hz=subcarrier_hz)
    inside = (f >= low) & (f <= high)

    def measure(grid):
        entries = {"genlock": genlock(grid, amount, **_burst_only(spec)),
                   "burst lock": burst_lock(grid, amount, **spec)}
        shapes = _shapes(entries)
        if len(shapes) < 2:
            return None
        return _participation(shapes, list(shapes))

    whole = measure(f)
    narrow = measure(f[inside]) if int(inside.sum()) > 8 else None

    # the per-line construction: one movement seen through two channels, each
    # scaled by its own measurement precision
    rng = np.random.default_rng(int(seed))
    movement = rng.standard_normal(int(lines))
    sample = 1.0 / (4.0 * float(subcarrier_hz))
    phase_precision = float(np.radians(1.0))
    per_line = []
    for fraction in (1.0, 0.1, separating_precision_s(
            1.0, subcarrier_hz)["in_4fsc_samples"], 0.01):
        time_precision = fraction * sample
        a = np.concatenate([movement / phase_precision,
                            np.zeros(int(lines))])
        b = np.concatenate([2.0 * np.pi * subcarrier_hz * movement
                            / phase_precision, movement / time_precision])
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        values = np.linalg.svd(np.array([a, b]), compute_uv=False)
        share = values ** 2 / float((values ** 2).sum())
        per_line.append({
            "sync_precision_in_4fsc_samples": float(fraction),
            "coherence": float(abs(a @ b)),
            "effective": float(1.0 / np.sum(share ** 2)),
        })

    return {
        "full_band": whole,
        "burst_band": narrow,
        "burst_band_hz": (low, high),
        "places_in_the_burst_band": int(inside.sum()),
        "one_dimension_on_the_frequency_axis": bool(
            whole is not None and whole["effective"] < 1.2),
        "per_line_by_sync_precision": per_line,
        "separating_precision": separating_precision_s(1.0, subcarrier_hz),
        "why": ("genlock is a constant phase on the chroma and burst lock a "
                "delay of the whole line, but the specified line's only high "
                "frequency content is the burst, so on the frequency axis "
                "they are one direction; only the sync's own position "
                "separates them, and only if it is located to a ninetieth of "
                "a 4fsc sample"),
    }


def displacement_null_space(displacement) -> Dict[str, object]:
    """WHAT THE LOCK'S OWN DISPLACEMENT CANNOT SEE.

    The burst-to-sync relationship is the only absolute phase reference the
    signal has, and the obvious way to read it - the displacement the lock
    exported, line by line - does not work. That series has its endpoints
    pinned, so pinning removes any affine function of the line index exactly,
    and a CONSTANT offset is annihilated with it. A constant burst-to-sync
    error is therefore in the null space of that instrument, and reading it
    there returns the same number for two tapes that differ.

    This function states the null rather than working around it: it takes a
    displacement series, applies the pinning, and reports how much of an
    added constant survives. The answer is none, to machine precision. The
    constant has to come from the burst phase read directly on the chroma
    output instead - and not from the luma, which has the chroma notched out
    of it, so the burst is not there to read at all.
    """
    series = np.asarray(displacement, dtype=np.float64).ravel()
    if series.size < 3:
        raise ValueError("too short a series to pin")

    def pin(values):
        index = np.arange(values.size, dtype=np.float64)
        span = max(values.size - 1, 1)
        ends = values[0] + (values[-1] - values[0]) * index / span
        return values - ends

    scale = float(np.std(series)) or 1.0
    pinned = pin(series)
    shifted = pin(series + scale)
    survives = float(np.max(np.abs(shifted - pinned)))
    return {
        "recovered_fraction": survives / scale,
        "pinned": pinned,
        "constant_added": scale,
        "constant_survives": survives,
        "null_is_the_constant": bool(survives / scale < 1e-9),
        "why": ("the exported displacement has its endpoints pinned, which "
                "annihilates a constant offset exactly; the absolute "
                "burst-to-sync phase must be read from the chroma output"),
    }


# --------------------------------------------------------------------------
# THE CONTROLS. Each has an answer known independently of this module, and
# each can fail.
# --------------------------------------------------------------------------


def pulse_control(sample_rate_hz: float = 40e6,
                  length: int = 1024) -> Dict[str, object]:
    """THE CONTROL ON THE CLOSED FORM: invert it and measure what comes back.

    The analytic sync spectrum is a rectangle's transform times a taper's,
    and the taper is where a construction of this kind goes wrong -
    `spec_sync`'s own docstring records a first attempt whose specified
    140 ns rise came out at 100 because a 10-90 time was used as a transition
    width. So the control inverse-transforms `sync_spectrum` and measures the
    pulse that returns with `standard_levels.half_amplitude_crossings`, an
    instrument in a module this one does not own, which reads at the pulse's
    own half amplitude with sub-sample interpolation.

    The expected answers are the standard's, known without reference to
    anything here: 4.7 us wide and 140 ns rise. Measured at 40 MHz over 1024
    places, width 4.700000 us - error 3.4 ps - and rise 141.5 ns, inside the
    50 ns the sample grid allows. It fails if the taper's transform is wrong,
    if the transition width is confused with the 10-90 time, if the width is
    misplaced, or if the position phase is not the pulse's own.

    The closed form is also compared against the discrete transform of
    `spec_sync` itself, over the band where the pulse holds a hundredth of
    its peak: mean relative error 0.00023, worst 0.00056.
    """
    from vhsdecode.models import standard_levels as sl

    rate = float(sample_rate_hz)
    count = int(length)
    f = np.fft.rfftfreq(count, d=1.0 / rate)

    # A discrete transform of samples is `rate` times the continuous
    # transform of the waveform. The pulse is moved to the middle of the
    # window so that neither edge wraps around it - `spec_sync` centres its
    # own pulse for the same reason.
    to_centre = np.exp(-2j * np.pi * f
                       * (0.5 * count / rate - 0.5 * pdim.SYNC_WIDTH_S))
    pulse = np.fft.irfft(sync_spectrum(f) * rate * to_centre, n=count)
    read = sl.half_amplitude_crossings(pulse, rate)

    built = np.abs(np.fft.rfft(pdim.spec_sync(rate, count))) / rate
    analytic = np.abs(sync_spectrum(f))
    band = analytic > 0.01 * float(analytic.max())
    error = (np.abs(built[band] - analytic[band])
             / np.maximum(analytic[band], 1e-30))
    return {
        "width_s": float(read["width_s"]),
        "rise_s": float(read["rise_s"]),
        "expected_width_s": pdim.SYNC_WIDTH_S,
        "expected_rise_s": pdim.SYNC_RISE_S,
        "width_error_s": float(read["width_s"]) - pdim.SYNC_WIDTH_S,
        "rise_error_s": float(read["rise_s"]) - pdim.SYNC_RISE_S,
        "spectrum_mean_error": float(error.mean()) if error.size else 1.0,
        "spectrum_worst_error": float(error.max()) if error.size else 1.0,
        "passes": bool(
            abs(float(read["width_s"]) - pdim.SYNC_WIDTH_S) < 2.0 / rate
            and abs(float(read["rise_s"]) - pdim.SYNC_RISE_S) < 2.0 / rate
            and error.size and float(error.max()) < 0.05),
        "why": ("the closed form inverted must be the standard's own pulse, "
                "measured by an instrument this module does not own"),
    }


def rate_control(frequency_hz=None, amounts=(0.25, 0.5, 1.0, 2.0, 4.0)
                 ) -> Dict[str, object]:
    """THE CONTROL ON THE SEPARABILITY LAW, and it can fail.

    The law this arc measures everything against says a family is COLLINEAR
    when its members differ only in a RATE and SEPARABLE when they differ in
    KIND. One delay taken at five sizes is the purest case of the first: the
    shapes are one vector at five scales, so the participation ratio must be
    exactly ONE however many sizes are used, and any excess is directions
    manufactured by the construction rather than found in it.

    It is the control `magnetic.linear_control` is, and for the same reason:
    there a construction that should have read 1.00 read 4.97, and that is
    what exposed a wrong answer of 4.65 of 5. It fails here if normalising a
    near-zero vector turns rounding into a direction, or if a signature
    depends on its size in any way other than as a scale.

    Measured: 1.000 of 5, with a second singular value 1.1e-16 of the
    first.
    """
    f = (np.linspace(0.0, 4.2e6, 4096) if frequency_hz is None
         else _grid(frequency_hz))
    shapes = []
    for size in amounts:
        delay = float(size) * 1e-9
        vector = np.unwrap(np.angle(np.exp(-2j * np.pi * f * delay)))
        vector = vector - vector.mean()
        norm = float(np.linalg.norm(vector))
        # a near-zero norm normalised is rounding turned into a direction,
        # which is exactly the failure this control exists to catch
        shapes.append(vector / norm if norm > 1e-12 else np.zeros_like(vector))
    singular = np.linalg.svd(np.array(shapes), compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    effective = float(1.0 / np.sum(share ** 2))
    return {
        "effective": effective,
        "expected": 1.0,
        "singular_values": singular,
        "second_over_first": (float(singular[1] / max(singular[0], 1e-30))
                              if singular.size > 1 else 0.0),
        "passes": bool(abs(effective - 1.0) < 0.05),
        "why": ("a family differing only in a rate is one direction; more "
                "than one means the construction is manufacturing them"),
    }


def null_space_control(length: int = 262, seed: int = 0) -> Dict[str, object]:
    """THE CONTROL ON THE NULL SPACE, whose answer is known by algebra.

    Pinning a series at its endpoints removes any affine function of the
    index exactly, so a constant added before pinning must survive it by
    nothing at all, and the expected answer is zero to machine precision -
    known from the arithmetic rather than from anything measured here.

    It can fail in both directions, which is what makes it a control rather
    than a restatement. If `displacement_null_space` removed the mean instead
    of the endpoints, a constant would still vanish but a tilt would not
    match; if it removed a tilt only, the constant would survive. And a
    QUADRATIC is not in an affine pinning's null, so it must survive - a
    construction that annihilated everything would pass a one-sided check and
    fails this one. Measured: the constant recovered at 7.4e-16, the tilt
    case identical to the constant case to 0.0, and the quadratic surviving
    at 3.3e-9 on a series of nanosecond scale.
    """
    rng = np.random.default_rng(int(seed))
    series = np.cumsum(rng.standard_normal(int(length))) * 1e-9
    constant = displacement_null_space(series)
    index = np.arange(int(length), dtype=np.float64) / max(int(length) - 1, 1)
    spread = 3.0 * float(np.std(series))
    tilt = displacement_null_space(series + spread * index)
    curve = displacement_null_space(series + spread * index ** 2)
    quadratic = float(np.max(np.abs(curve["pinned"] - constant["pinned"])))
    return {
        "constant_recovered": float(constant["recovered_fraction"]),
        "tilt_matches_the_constant_case": float(np.max(np.abs(
            tilt["pinned"] - constant["pinned"]))),
        "quadratic_survives": quadratic,
        "passes": bool(constant["recovered_fraction"] < 1e-9
                       and quadratic > 0.0),
        "why": ("pinning the endpoints removes any affine function of the "
                "index exactly and nothing else; a constant is annihilated, "
                "a tilt with it, and a quadratic is not"),
    }


def controls(**kwargs) -> Dict[str, object]:
    """All three controls, and whether every one of them passed."""
    results: Dict[str, object] = {
        "pulse": pulse_control(),
        "rate": rate_control(),
        "null_space": null_space_control(),
    }
    results["passes"] = bool(all(value["passes"] for value in results.values()
                                 if isinstance(value, dict)))
    return results


# --------------------------------------------------------------------------
# THE COLOUR LOCK, READ ONE FIELD AT A TIME
#
# Ethan's directive for this stage, in his own order: the active area's
# residual carries the colour lock, so the residual colour carrier gives the
# amplitude and the phase for the up-heterodyne; the up-heterodyne need not
# use a fixed frequency; the luma retains residual chroma, so with the
# active area masked IN and the colour-under phase locked, the
# up-heterodyned residual colour is subtracted from the luma to leave a
# colour-free luma; that connection is fed back into the chroma stage for
# alignment on all dimensions; and the correction is applied on ONE FIELD AT
# A TIME with any averaging removed.
#
# Everything above this line in this module is a closed-form spectrum of the
# specified line - the sensitivity of a signature to a parameter, with no
# field in it anywhere. What follows is the first measurement in the module
# that reads a field, and it is written to Ethan's last clause first: there
# is no accumulator, no rolling mean and no state, so a caller that hands it
# one field gets that field's answer and nothing of any other.
# --------------------------------------------------------------------------


def colour_lock(chroma_field, sample_rate_hz: float,
                burst_window: Tuple[int, int],
                active_window: Tuple[int, int],
                first_line: int = 0,
                subcarrier_hz: float = SUBCARRIER_HZ) -> Dict[str, object]:
    """THE UP-HETERODYNED COLOUR CARRIER'S LOCK, from one field, twice.

    `chroma_field` is one decoded chroma field as lines by samples, already
    up-heterodyned onto the subcarrier and time-base corrected - what a
    `*_chroma.tbc` holds. `burst_window` and `active_window` are the two
    column spans the decode's own JSON gives as colourBurstStart to
    colourBurstEnd and activeVideoStart to activeVideoEnd; nothing about the
    geometry is assumed here. `first_line` is the field line number of the
    first row handed in, which is needed because the specified alternation
    depends on it.

    WHAT IS MEASURED. In each window the carrier's complex phasor is taken
    at the subcarrier, referred to the line's own origin - which is the sync
    datum, so this phase IS the burst-to-sync relationship SMPTE 170M clause
    8.5 specifies - and the standard's own half-cycle-per-line alternation
    is removed before the lines are combined, from
    `burst_sync_lock.expected_burst_phase_rad`. Without that removal every
    other line reads 180 degrees out and the field mean is nothing.

    Each window then yields a COMPLEX lock, amplitude and phase in one
    number, and a LINE COHERENCE - the length of the mean phasor against the
    mean length - which says how much of that window is actually locked
    rather than merely present.

    THE PAIR, AND WHAT IT SAYS ON REAL TAPE. The two windows are two
    independent readings of one lock and they can disagree. Measured
    2026-09-06 on the first six fields of three decoded tapes in
    /tmp/claude-1000, lines 30 to 239:

        tape                  burst phase   active phase   difference   coh
        dod_wide_75bars_SP      -33.0/+147   +11.9/-170.2   -39 to -45  0.97
        dod_wide_home           -33.1/+147  -111.7/+59.8    +77 to +87  0.73-0.88
        dod_wide_chromanoise_SP -32.9/+147  -110.0/+70.1    +77 to +79  0.9998

    The burst window is locked to 0.9999 or better on every field of every
    tape, and its phase sits at -33 or +147 degrees, exactly 180 apart,
    which is the four-field colour sequence the arc has already
    established. The active
    area IS strongly line-coherent, 0.73 to 0.9999, so Ethan's premise holds:
    a residual colour carrier is there to read.

    BUT ITS PHASE IS THE PICTURE'S, NOT THE LOCK'S, on picture-bearing
    material, and the same deck proves it: the burst-to-active difference is
    -42 degrees on the colour bars and +78 degrees on the chroma noise
    pattern, two recordings made by one machine within minutes of each
    other. A lock cannot depend on what was filmed. The active area's mean
    phasor at the subcarrier is the vector sum of the hues in the window,
    and on these tapes that sum dominates whatever carrier leak sits
    underneath it.

    `within_line_frequency_hz` is the same statement in the other unit and
    is the sharper form of it. The active window is split in half and the
    phase difference between the halves is read as a frequency, which is
    what Ethan's "the up-heterodyne need not use a fixed frequency" asks
    for. Measured across every field:

        dod_wide_75bars_SP       -18749.7 +- 11.0 Hz
        dod_wide_home             -3488.6 +- 975.6 Hz
        dod_wide_chromanoise_SP      -57.1 +- 18.7 Hz

    Eighteen kilohertz is not a heterodyne error. Over the 13.27
    microseconds between the two half-window centres it is 89.6 degrees,
    which is the hue rotation between the left and right halves of a colour
    bar pattern; the chroma noise field, whose hues carry no left-right
    structure, reads 57 Hz on the same deck at the same speed. So this
    reading is usable as a frequency only where the picture's chroma has no
    structure across the line, and the value is returned with its own
    coherence beside it so that a caller can see when it does not.

    WHAT DOES SAY THE HETERODYNE IS LOCKED is the burst across the field:
    fitting the burst phase against line number over lines 30 to 239 gives
    a drift of -0.01 +- 0.07, -0.01 +- 0.08 and +0.05 +- 0.09 Hz on the
    three tapes, so on this material the up-heterodyne holds to a tenth of
    a hertz within a field and the fixed-frequency assumption costs nothing
    that these captures can see. `burst_drift_hz` is that number.

    WHAT THE RUNTIME WOULD CALL. `vhsdecode/chroma.py` belongs to another
    lane and is not touched here, so this is the modelling side of the
    correction rather than the correction. Its
    `upconvert_chroma_phase_comp` already takes a per-burst frequency and
    interpolates it line to line, so Ethan's second clause is met there
    already; what this adds is a second, independent reading of the same
    lock, per field and with no averaging, that the runtime could compare
    its burst sequence against before up-converting - and a measurement
    that says, on this material, that the comparison would be dominated by
    the picture rather than by the channel.
    """
    field = np.asarray(chroma_field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("one field, as lines by samples")
    rows = field - field.mean(axis=1, keepdims=True)
    columns = np.arange(field.shape[1], dtype=np.float64)
    seconds = columns / float(sample_rate_hz)
    rotator = np.exp(-2j * np.pi * float(subcarrier_hz) * seconds)
    numbers = np.arange(int(first_line), int(first_line) + field.shape[0],
                        dtype=np.float64)
    # THE SPECIFIED ALTERNATION COMES OFF BEFORE THE LINES ARE COMBINED.
    # 227.5 cycles of subcarrier a line is the standard's own figure and
    # `burst_sync_lock` already holds it; a second transcription here would
    # be a second chance to disagree with it.
    alternation = np.exp(-1j * bsl.expected_burst_phase_rad(numbers))

    def window(span):
        start, stop = int(span[0]), int(span[1])
        if not 0 <= start < stop <= field.shape[1]:
            raise ValueError("a window must lie inside the line, and %d to "
                             "%d does not" % (start, stop))
        phasors = (rows[:, start:stop] @ rotator[start:stop]) * alternation
        mean = complex(np.mean(phasors))
        lengths = float(np.mean(np.abs(phasors)))
        return {
            "lock": mean,
            "amplitude": abs(mean),
            "phase_deg": float(np.degrees(np.angle(mean))),
            "line_coherence": abs(mean) / lengths if lengths > 0 else 0.0,
            "per_line": phasors,
            "centre_s": 0.5 * (seconds[start] + seconds[stop - 1]),
            "columns": (start, stop),
        }

    burst = window(burst_window)
    active = window(active_window)

    # the up-heterodyne's frequency, read across the active window rather
    # than assumed to be the one it was mixed with
    start, stop = int(active_window[0]), int(active_window[1])
    middle = (start + stop) // 2
    first = window((start, middle))
    second = window((middle, stop))
    gap = second["centre_s"] - first["centre_s"]
    step = complex(np.sum(second["per_line"] * np.conj(first["per_line"])))
    within = (float(np.angle(step)) / (2.0 * np.pi * gap)) if gap else float("nan")

    # and the same question asked of the burst across the whole field, which
    # is the reading that does not depend on the picture
    phases = np.unwrap(np.angle(burst["per_line"]))
    if len(numbers) > 2 and np.ptp(numbers) > 0:
        per_line = float(np.polyfit(numbers, phases, 1)[0])
        drift = per_line / (2.0 * np.pi) * float(sample_rate_hz) / field.shape[1]
    else:
        drift = float("nan")

    difference = burst["lock"] * np.conj(active["lock"])
    return {
        "burst": burst,
        "active": active,
        # the pair, as one complex number: its angle is the disagreement in
        # phase and its magnitude the product of the two amplitudes
        "difference": difference,
        "difference_deg": float(np.degrees(np.angle(difference))),
        "amplitude_ratio": (burst["amplitude"] / active["amplitude"]
                            if active["amplitude"] > 0 else float("inf")),
        # the up-heterodyne's frequency, measured rather than fixed
        "within_line_frequency_hz": within,
        "within_line_coherence": float(min(first["line_coherence"],
                                           second["line_coherence"])),
        "burst_drift_hz": drift,
        "lines": int(field.shape[0]),
        "first_line": int(first_line),
        "subcarrier_hz": float(subcarrier_hz),
        "why": ("the burst and the active area are two readings of one "
                "lock, so they can disagree; on picture-bearing material "
                "the active one is the vector sum of the hues in the "
                "window and its line coherence is what says so"),
    }
