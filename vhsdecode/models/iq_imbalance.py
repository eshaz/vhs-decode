"""THE CHROMA'S QUADRATURE IMBALANCE, seen through the burst itself.

Ethan, 2026-09-06, looking at a composite vectorscope: *"The vectorscope
looks like an IQ imbalance on the chroma channel"*, and then the method:
*"I think we can see the IQ imbalance by using the entire complex shape of
the burst itself in all dimensions."*

WHAT AN IMBALANCE IS, and why a magnitude cannot see it. A balanced
quadrature pair maps a phasor to `alpha b`. An imbalanced one, whose two
arms differ in gain or are not exactly ninety degrees apart, maps it to

    b  ->  alpha b + beta conj(b)

and that conjugate term is a second phasor counter-rotating against the
first. A circle becomes an ellipse, a hue rotates the wrong way, and the
whole colour coordinate system shears. `|beta/alpha|` is the image
rejection, and it is invisible to `abs()` because both terms have the same
modulus.

ONE PHASOR CANNOT SHOW IT, WHICH IS WHY THE BURST'S OWN VARIATION IS THE
INSTRUMENT. With a single measured phasor, any `alpha` fits for some
`beta`; the two are one unknown. The reference has to SWEEP, and the burst
sweeps by specification: SMPTE 32M clause 3.9.2.1.5 rotates the
colour-under carrier ninety degrees a line. Measured on the two candidate
references:

    reference                             coherence   condition
    composite burst, 180 degrees a line      1.0000    1.331e+14
    colour-under burst, 90 degrees a line    0.0000    1.000e+00

At a half turn a line, `b` and `conj(b)` BOTH change sign every line, so
they are the same sequence and no fit can separate them. At a quarter turn
they rotate opposite ways with period four and are exactly orthogonal. So
the measurement must be made on the COLOUR-UNDER, before the decoder's
up-conversion, and the record-side rotation is not an obstacle to it - it
is the only thing that makes it possible.

THE SIGNATURE SITS AT PERIOD TWO, which is what makes it immune to the
time base. De-rotating the measured burst by the rotation it carries turns
`alpha i^n b + beta (-i)^n conj(b)` into `alpha b + beta (-1)^n conj(b)`,
so the imbalance lands at exactly period two along the line axis - the
fastest structure that axis has - while the time base's drift is the
slowest. They cannot be confused.

AND THE ROTATION'S SIGN ALTERNATES WITH THE TRACK, which was the trap. The
standard advances on one track and retards on the other, so a single fixed
sign is wrong for half of any capture spanning a head switch, and using
one returned `|beta/alpha|` above two - an image stronger than the signal,
which is not a thing. Taking each run's rotation from the run itself
returns the specification's own value and a physical answer:

    capture           tap        runs   rotation   |beta/alpha|   image    control
    75 bars           record       31    -88.97       0.0063     -44.06 dB   1.97x
    75 bars           playback     29    +90.92       0.1067     -19.44      1.77x
    chroma noise      record       31    +88.90       0.1758     -15.10      5.77x

The measured rotations of -88.97, +90.92 and +88.90 degrees a line confirm
the clause directly, on the radio frequency, without a decoder in the way.

WHAT IT SAYS. On the bars the record tap rejects its image by 44 dB and
the playback tap by only 19, on the same signal through the same deck - so
the imbalance is introduced on the PLAYBACK side and is not on the tape.
An image rejection of -19.4 dB is a gain imbalance of about 21 per cent or
a quadrature error of about 12 degrees, or some mixture, and 12 degrees of
shear is the size of thing a vectorscope shows plainly.

ONE SMALL BIAS, MEASURED RATHER THAN IGNORED. Each run is de-rotated by a
rotation estimated from the same data that carries the image, so the image
biases its own estimate a little. On planted values of 0.020, 0.050 and
0.150 the method returns 0.019947, 0.049866 and 0.149581 - short by 0.27,
0.27 and 0.28 per cent. The shortfall is flat in the imbalance, so it is a
fixed fractional bias and not a growing error, and at that size it changes
no conclusion here.

WHAT IS NOT YET SETTLED. The margin over the control is 1.77 to 1.97 on
the bars, which is present but slim; only the chroma-noise record tap
reaches 5.77. And the chroma-noise playback capture returns a run rotation
of -18.75 degrees rather than a quarter turn, which means the run detector
failed there and its row is not usable. Both are reported rather than
averaged away.

THE INVERSE, AND WHERE IT IS APPLIED. The map is a real-linear map of the
plane and it inverts in closed form:

    b = (conj(alpha) z - beta conj(z)) / (|alpha|^2 - |beta|^2)

singular only when the image is as strong as the signal. `correct_baseband`
is that line alone; `correct` is the same line reached from what the
decoder actually holds - real lines at four times the subcarrier, one
burst a line as the phase reference - by way of the analytic signal. Three
pieces of bookkeeping stand between the two, each measured rather than
assumed, and each recorded in its own docstring: the burst's frame is NOT
the frame in which the image coefficient is constant (it alternates sign
every line, because the burst turns a quarter turn a line and the image
turns twice as fast the other way); the burst itself carries the image, so
its measured phase wobbles at period two and the frame has to be taken from
the imbalance-free burst, which the pair itself supplies; and a heterodyne
whose local oscillator sits above the signal conjugates the image
coefficient, so a pair measured on the colour-under is passed conjugated to
the up-converted lines.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

from vhsdecode.models import composite_channel, hypercomplex, output_limit

# The decoder's output grid is four times the colour subcarrier
# (`output_limit.FOUR_FSC_HZ`, "the arc's output grid", derived from SMPTE
# 170M's line rate), so on it the subcarrier turns exactly a quarter of a
# cycle a sample - four samples a cycle, the geometry `chroma._complex_envelope`
# and `burst_sync_lock.ambiguity_interval_s` both rest on. A property of the
# grid and not of the system: a 4 fsc PAL grid has the same quarter.
FSC_CYCLES_PER_SAMPLE = output_limit.NTSC_SUBCARRIER_HZ / output_limit.FOUR_FSC_HZ

# SMPTE 32M-2004 clause 3.9.2.1.5: the colour-under burst turns a quarter
# turn a line on tape. The imbalance is on the playback side (the record tap
# rejects its image by 44 dB, the playback tap by 19), so it sees the burst
# turning, and everything referenced to the burst sees the image coefficient
# turn twice as fast the other way.
BURST_ROTATION_DEG_PER_LINE = composite_channel.PHASE_ROTATION_DEG_PER_LINE


def separability(degrees_per_line: float, lines: int = 64
                 ) -> Dict[str, object]:
    """Whether a reference rotating this fast can separate an imbalance.

    A half turn a line cannot, because the signal and its image become the
    same sequence. A quarter turn can, and the specification supplies one.
    """
    index = np.arange(int(lines), dtype=np.float64)
    ideal = np.exp(1j * math.radians(float(degrees_per_line)) * index)
    image = np.conj(ideal)
    norm = np.linalg.norm(ideal) * np.linalg.norm(image)
    coherence = float(abs(np.vdot(ideal, image)) / max(norm, 1e-300))
    stacked = np.column_stack([
        np.concatenate([ideal.real, ideal.imag]),
        np.concatenate([image.real, image.imag])])
    singular = np.linalg.svd(stacked, compute_uv=False)
    return {
        "degrees_per_line": float(degrees_per_line),
        "coherence": coherence,
        "condition": float(singular.max() / max(singular.min(), 1e-300)),
        "separable": bool(coherence < 0.5),
        "why": ("the signal and its image must be distinguishable "
                "sequences, and at a half turn a line they are not"),
    }


def from_burst_series(phasors, run_lines: int = 120,
                      control_fraction: float = 0.8) -> Dict[str, object]:
    """The imbalance from a run of colour-under burst phasors.

    Each run's rotation is taken FROM THAT RUN rather than assumed, because
    the standard's rotation reverses with the track and a fixed sign is
    wrong for half a capture. The statistic is the period-two component
    after de-rotation, against a control taken a little away from period
    two where no imbalance can put anything.
    """
    values = np.asarray(phasors, dtype=np.complex128).ravel()
    span = int(run_lines)
    if values.size < 2 * span:
        raise ValueError("an imbalance needs at least two runs of lines")
    rotations, ratios, controls = [], [], []
    for start in range(0, values.size - span, span):
        run = values[start:start + span]
        step = float(np.angle(complex((run[1:] * np.conj(run[:-1])).sum())))
        index = np.arange(run.size, dtype=np.float64)
        turned = run * np.exp(-1j * step * index)
        level = abs(complex(turned.mean()))
        if level <= 0.0:
            continue
        rotations.append(math.degrees(step))
        ratios.append(abs(complex((turned * (-1.0) ** index).mean())) / level)
        controls.append(abs(complex(
            (turned * np.exp(-1j * math.pi * float(control_fraction)
                             * index)).mean())) / level)
    if not ratios:
        return {"usable": False,
                "why": "no run carried a measurable burst level"}
    ratio = float(np.median(ratios))
    control = float(np.median(controls))
    return {
        "usable": True,
        "runs": len(ratios),
        "rotation_deg_per_line": float(np.median(rotations)),
        "image_ratio": ratio,
        "image_rejection_db": 20.0 * math.log10(max(ratio, 1e-12)),
        "control_ratio": control,
        "over_control": ratio / max(control, 1e-12),
        "gain_imbalance_fraction": 2.0 * ratio,
        "quadrature_error_deg": math.degrees(2.0 * ratio),
        "rotation_is_a_quarter_turn": bool(
            abs(abs(float(np.median(rotations))) - 90.0) < 5.0),
        "why": ("the imbalance is the period-two component after each run "
                "is de-rotated by the rotation it actually carries"),
    }


def _pair(alpha, beta):
    """The pair as complex numbers, refused when the map is singular."""
    a = complex(alpha)
    b = np.asarray(beta, dtype=np.complex128)
    if not (math.isfinite(a.real) and math.isfinite(a.imag)
            and np.all(np.isfinite(b))):
        raise ValueError("alpha and beta must be finite")
    if abs(a) <= 0.0:
        raise ValueError("alpha must not be zero: the map would have no "
                         "signal term to recover")
    worst = float(np.abs(b).max()) if b.size else 0.0
    if worst >= abs(a):
        raise ValueError(
            "|beta| >= |alpha| (|beta/alpha| = %.4f): the image is as strong "
            "as the signal and b -> alpha b + beta conj(b) is singular"
            % (worst / abs(a)))
    return a, b


def correct_baseband(z, alpha, beta, amount: float = 1.0) -> np.ndarray:
    """THE INVERSE MAP on a complex baseband, at a fraction of its change.

    An imbalance maps `b` to `z = alpha b + beta conj(b)`, and because that
    is a real-linear map of the plane it inverts in closed form:

        b = (conj(alpha) z - beta conj(z)) / (|alpha|^2 - |beta|^2)

    singular exactly when the image is as strong as the signal, and refused
    there. `alpha`'s modulus and argument are the chroma's gain and hue,
    which this stage does not own: a caller holding only the measured ratio
    passes `alpha = 1` and `beta` equal to it. `beta` may be one number or
    an array that broadcasts against `z`, which is how the real-line form
    hands each line its own coefficient.

    Measured on a planted band-limited baseband with |beta/alpha| = 0.1067
    (the playback tap's value) at an arbitrary angle: the recovery is exact
    to 4.97e-16, and to 6.28e-16 with a complex alpha of 0.9 e^{0.4j}. On a
    rotating burst series the module's own instrument reads an image ratio
    of 0.106691 before and 4.35e-16 after, against a control of 5.79e-16 -
    the floor.

    `amount` applies that fraction of the change the full inverse would
    make, `z + amount (b - z)`. That is exactly the linear application the
    correction gain law describes, and the residual image is exactly
    `(1 - amount) |beta/alpha|`: measured 0.106700, 0.080025, 0.053350 and
    0.026675 at amounts 0, 0.25, 0.5 and 0.75. Half the amount therefore
    halves the image, 6.02 dB below the uncorrected one - not half of the
    way in decibels, which has no meaning when the full inverse reaches the
    floor.
    """
    values = np.asarray(z, dtype=np.complex128)
    a, b = _pair(alpha, beta)
    gain = float(amount)
    if not math.isfinite(gain):
        raise ValueError("amount must be finite")
    full = (np.conj(a) * values - b * np.conj(values)) / (
        abs(a) ** 2 - np.abs(b) ** 2)
    return values + gain * (full - values)


def _baseband(block: np.ndarray, cycles_per_sample: float) -> np.ndarray:
    """Each line's observed complex baseband: the analytic signal along the
    line, de-rotated by the subcarrier from the line's first sample."""
    index = np.arange(block.shape[1], dtype=np.float64)
    analytic = block + 1j * hypercomplex.hilbert(block, axis=-1)
    return analytic * np.exp(-2j * math.pi * float(cycles_per_sample)
                             * index)[None, :]


def _carrier(width: int, cycles_per_sample: float) -> np.ndarray:
    index = np.arange(int(width), dtype=np.float64)
    return np.exp(2j * math.pi * float(cycles_per_sample) * index)


def line_reference(lines, window, fsc_cycles_per_sample: float = FSC_CYCLES_PER_SAMPLE
                   ) -> np.ndarray:
    """The burst phasor of every line, in the convention `correct` reads.

    The analytic signal along each line is de-rotated by the subcarrier
    counted from the line's FIRST sample and averaged over `window`, a
    `(start, stop)` pair in samples, so the phasor's argument is the burst's
    phase at that first sample and its modulus the burst's amplitude. A
    caller with the decoder's own burst fit converts to this; one holding
    these lines simply calls it. Measured at zero imbalance on lines gated
    with 6-sample raised-cosine edges, the phasor is within 1.3e-3 degrees
    and 0.03 per cent of the planted burst; with 4-sample edges 7.3e-3
    degrees and 0.10 per cent - the analytic signal's edge error, of which
    more under `correct`.

    WHAT THE PHASOR IS NOT. It is the burst as OBSERVED, and the observed
    burst carries the image: `(alpha + beta_n)` times the true one, so its
    phase wobbles at period two by `arg(1 + beta_n / alpha)` - 0.68 degrees
    at the planted 0.1067 e^{0.7j}, 6.09 degrees for the same modulus as a
    pure quadrature error. `correct` removes that wobble from the reference
    itself before using it, which is why it is passed the observed phasor
    and not something cleaner.
    """
    block = np.asarray(lines, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError("lines must have shape (lines, width)")
    start, stop = int(window[0]), int(window[1])
    if not 0 <= start < stop <= block.shape[1]:
        raise ValueError("the window must lie inside the line")
    return _baseband(block, fsc_cycles_per_sample)[:, start:stop].mean(axis=1)


def image_of_lines(lines, window, run_lines: Optional[int] = None,
                   fsc_cycles_per_sample: float = FSC_CYCLES_PER_SAMPLE
                   ) -> Dict[str, object]:
    """The image as the module's own instrument reads it on these lines.

    The line bursts from `line_reference`, in the lines' own frame, through
    `from_burst_series` in two runs of half the lines (`run_lines` overrides
    that). Separable only where the burst sweeps a quarter turn a line - on
    the colour-under, or on lines still in the frame the imbalance lives in
    - and reported as not usable otherwise. The decoder's locked,
    up-converted lines return exactly that: their burst alternates a half
    turn, and the module proves at the top why a half turn separates
    nothing.

    READ THE AFTER-NUMBER WITH CARE. `correct` keeps every burst where the
    reference put it, so on unlocked lines the corrected burst still carries
    the observed burst's phase wobble, and this instrument reads that wobble
    as a remaining image of `|Im(beta / alpha)|`: measured 0.10669 before
    and 0.01068 after at a planted imaginary part of 0.01056, with the
    amplitude wobble gone (0.1062 to 0.0000 of the mean). The picture
    relative to its burst is at the floor, and a lock removes the wobble
    from burst and picture together.
    """
    phasors = line_reference(lines, window, fsc_cycles_per_sample)
    run = int(run_lines) if run_lines else phasors.size // 2
    if run < 8 or phasors.size < 2 * run:
        return {"usable": False,
                "why": "too few lines for two runs of at least eight"}
    found = from_burst_series(phasors, run_lines=run)
    if found.get("usable") and not found["rotation_is_a_quarter_turn"]:
        found = dict(found)
        found["usable"] = False
        found["why"] = ("the burst does not sweep a quarter turn on these "
                        "lines, so the image is not separable here")
    return found


def _unit_reference(reference, count: int):
    """The per-line reference as unit phasors, and which lines carry one."""
    values = np.asarray(reference)
    if values.ndim != 1 or values.size != count:
        raise ValueError("one reference is needed for each line")
    if np.iscomplexobj(values):
        magnitude = np.abs(values)
        usable = np.isfinite(magnitude) & (magnitude > 0.0)
        unit = np.where(usable, values / np.where(usable, magnitude, 1.0), 0.0)
    else:
        phase = values.astype(np.float64)
        usable = np.isfinite(phase)
        unit = np.where(usable, np.exp(1j * np.where(usable, phase, 0.0)), 0.0)
    return unit.astype(np.complex128), usable


def correct(lines, reference, alpha, beta, amount: float = 0.5, *,
            rotation_deg_per_line: float = BURST_ROTATION_DEG_PER_LINE,
            line_offset: int = 0,
            fsc_cycles_per_sample: float = FSC_CYCLES_PER_SAMPLE
            ) -> Dict[str, object]:
    """THE INVERSE ON REAL LINES, one burst a line as the phase reference.

    What the decoder holds is the up-converted chroma at four times the
    subcarrier: real, a line at a time, a burst on every line. This reaches
    the complex baseband from that - the analytic signal along the line
    (`hypercomplex.hilbert`), de-rotated by the subcarrier at
    `fsc_cycles_per_sample` (a quarter of a cycle a sample on the 4 fsc
    grid) and by the line's reference - applies the inverse there,
    re-rotates, remodulates and writes the result back into `lines` in
    place. Lines whose reference is zero or not finite are left untouched
    and counted as skipped.

    `reference` is one phasor a line, or its phase in radians: the burst as
    OBSERVED on these lines, in `line_reference`'s convention - for the
    decoder's locked lines that is simply the lock's target phase.
    `alpha, beta` are the pair as `from_burst_series` measures it, in the
    frame where the imbalance-free burst of the run's first line is real
    and positive: `alpha = 1`, `beta` the complex ratio of the period-two
    component to the mean. `amount` is the fraction of the change applied,
    and the caller's default is a half by the correction gain law: the
    optimum is a* = 1/(1 + rho) with rho the model error, which no
    resampling of the fit's own data can see, and applied at half the
    believed optimum a correction keeps 75 per cent of its benefit anywhere
    in the safe range. The believed optimum here is 1, the image stands
    only 1.8 to 2.0 times over its control on the bars, and so a half is
    what the law prescribes.

    THREE PIECES OF BOOKKEEPING sit between the pure inverse and these
    lines, each with a measured cost of getting it wrong, on 240 planted
    lines at |beta/alpha| = 0.1067 (the playback tap) where the correction
    itself reaches -55.6 dB from -19.4:

    1. THE BURST'S FRAME IS NOT THE FRAME THE IMAGE IS CONSTANT IN. The
       imbalance is fixed in the playback chain's frame; the burst turns
       `rotation_deg_per_line` in that frame (SMPTE 32M 3.9.2.1.5, a quarter
       turn), and a frame turned by phi carries beta to beta e^{-2j phi}, so
       referenced to the burst the coefficient turns twice as fast the other
       way - at the quarter turn it alternates sign every line, which is the
       module's period-two signature seen from the other side. Each line
       gets `beta e^{-2j rotation (n + line_offset)}`, `line_offset` being
       the index of `lines[0]` counted from the first line of the run the
       pair was measured on. A constant coefficient instead lands at -16.2
       dB, WORSE than uncorrected; the wrong parity at -13.2, the image
       doubled. The decoder's up-conversion undoes the tape's rotation to
       give the composite's half turn, and the alternation survives that,
       because it is the up-conversion's own quarter turn a line that
       carries it across.

    2. THE BURST CARRIES THE IMAGE TOO. What is observed on line n is
       `(alpha + beta_n)` times the true burst, so the observed phase
       wobbles at period two - 0.68 degrees at the planted angle, 6.09
       degrees for a pure quadrature error of the same size - and a frame
       taken from it is wrong by that. The pair supplies the wobble, so the
       reference is divided by `(alpha + beta_n)` before it is used: with
       the division the quadrature case reaches -55.7 dB, without it -32.7.
       A pure gain imbalance has no wobble and does not care (-55.9 dB
       either way).

    3. A HETERODYNE ABOVE THE SIGNAL CONJUGATES. The decoder's
       `upconvert_chroma_phase_comp` runs its local oscillator at
       pi/2 (1 + f_cu / f_sc) radians a sample - f_sc + f_cu on the 4 fsc
       grid, above the colour-under - so the difference product carries
       conj(b), and with it conj(beta). Measured on a periodic tone, the
       up-converted baseband is conj(z) to -252.7 dB and the correction
       needs conj(beta) (-251.7 dB) where beta as measured leaves -16.7. A
       pair measured on the colour-under is therefore passed as
       `conj(beta)` for the up-converted lines: unconjugated it lands at
       -30.1 dB on the decoder's lines, and conjugated where there was no
       inversion it costs the same (-30.0). At the quarter turn the
       rotation's sign is immaterial, e^{-2j pi/2 n} and e^{+2j pi/2 n}
       being the same alternation.

    THE CORRECTED LINE KEEPS ITS BURST WHERE THE REFERENCE PUT IT (to
    2.8e-14 degrees), with the amplitude wobble removed (0.1062 to 0.0000
    of the mean), because the burst is what every later stage demodulates
    against: on the decoder's locked lines that is the target phase, and on
    unlocked lines a later lock removes the observed phase wobble from burst
    and picture together. The picture is corrected RELATIVE TO ITS BURST,
    and that is how it is judged. On the decoder's own case - lines locked
    to an alternating target after an inverting up-conversion, `conj(beta)`
    passed - the direct comparison against the true locked lines goes from
    -19.4 dB to -55.6 at amount 1 and -25.4 at a half, against a floor of
    -55.7, with the burst held at the target to 0.001 degrees.

    THE FLOOR IS THE ANALYTIC SIGNAL'S, AND IT IS THE GATE'S, NOT THE
    WRAP'S. `hypercomplex.hilbert` is periodic, and a chroma line is gated -
    zero through sync and the porches - so the line has no wrap
    discontinuity; what remains is the gate transitions' spectral spread
    across zero frequency, which no analytic signal can assign. Measured at
    zero imbalance, a band-limited picture gated with raised-cosine edges:

        edge, samples at 4 fsc    baseband error    real round trip
              2   (0.14 us)          -33.3 dB           -58.7 dB
              4   (0.28 us)          -44.7              -67.7
              6   (0.42 us)          -55.6              -84.7
              8   (0.56 us)          -59.2              -83.8
             12   (0.84 us)          -67.1              -93.8
             16   (1.12 us)          -72.8             -100.1

    concentrated at the transitions (6-sample edges: 1.6e-2 peak in the
    burst, 1.3e-2 in the picture, 5.4e-4 in sync). A line with chroma
    running to its ends wraps instead, and the periodic transform then costs
    -29.0 dB, 0.81 and 0.67 peak in the first and last samples with the
    interior at 4.5e-3 - which is why lines are taken whole and never in
    pieces. The real round trip is far better than the baseband because the
    error sits mostly in the quadrature the real part discards.

    Returns what it did: the lines written and skipped, the amount, the pair
    and its image rejection, the residual the model predicts,
    `(1 - amount) |beta/alpha|`, and the largest reference wobble removed.
    An independent before-and-after is `image_of_lines`, where the burst
    sweeps.
    """
    if not isinstance(lines, np.ndarray) or lines.ndim != 2:
        raise ValueError("lines must be a two-dimensional array, (lines, width)")
    if not np.issubdtype(lines.dtype, np.floating):
        raise TypeError("lines must be a floating-point array, written in place")
    count, width = lines.shape
    a, b = _pair(alpha, beta)
    if b.ndim != 0:
        raise ValueError("beta is one number here; the per-line alternation "
                         "is derived from the burst's rotation")
    b = complex(b)
    gain = float(amount)
    if not math.isfinite(gain):
        raise ValueError("amount must be finite")
    observed, usable = _unit_reference(reference, count)

    # THE IMAGE COEFFICIENT IN EACH LINE'S BURST FRAME. The burst turns
    # `rotation` a line relative to the frame the imbalance lives in, and a
    # frame turned by phi carries beta to beta exp(-2j phi): at the
    # standard's quarter turn the coefficient simply alternates sign.
    turn = math.radians(float(rotation_deg_per_line))
    index = np.arange(count, dtype=np.float64) + float(int(line_offset))
    beta_line = b * np.exp(-2j * turn * index)

    # THE BURST CARRIES THE IMAGE TOO: what is observed on line n is
    # (alpha + beta_n) times the true burst, so the imbalance-free direction
    # is the observed one divided by that, and the pair supplies it.
    wobble = a + beta_line
    true_direction = observed / wobble
    magnitude = np.abs(true_direction)
    true_direction = np.where(usable, true_direction
                              / np.where(usable, magnitude, 1.0), 0.0)

    baseband = _baseband(lines.astype(np.float64, copy=False),
                         fsc_cycles_per_sample)
    framed = baseband * np.conj(true_direction)[:, None]
    restored = correct_baseband(framed, a, beta_line[:, None], 1.0)
    # the corrected line keeps its burst where the reference put it
    full = restored * observed[:, None]
    corrected = baseband + gain * (full - baseband)

    carrier = _carrier(width, fsc_cycles_per_sample)
    rows = np.flatnonzero(usable)
    if rows.size:
        lines[rows] = np.real(corrected[rows] * carrier[None, :]).astype(
            lines.dtype, copy=False)

    ratio = abs(b) / abs(a)
    return {
        "lines": int(rows.size),
        "skipped": int(count - rows.size),
        "amount": gain,
        "alpha": a,
        "beta": b,
        "rotation_deg_per_line": float(rotation_deg_per_line),
        "image_ratio": ratio,
        "image_rejection_db": 20.0 * math.log10(max(ratio, 1e-12)),
        "residual_image_ratio": abs(1.0 - gain) * ratio,
        "residual_image_rejection_db": 20.0 * math.log10(
            max(abs(1.0 - gain) * ratio, 1e-12)),
        "reference_wobble_deg": float(np.degrees(np.abs(
            np.angle(wobble[usable]))).max()) if rows.size else 0.0,
        "why": ("the inverse of alpha b + beta conj(b), applied in each "
                "line's imbalance-free burst frame with the coefficient the "
                "burst's rotation gives that line, at `amount` of the change"),
    }
