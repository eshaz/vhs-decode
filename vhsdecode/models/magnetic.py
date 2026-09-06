"""The magnetic residual: the shape the tape exhibits against record level.

Ethan: *"I am modeling the shape of the observed magnetic properties of the
tape as the information sits on the tape ... I think the phsical models we
used are extra dimensions into the magnetic information, such as record level
for example. There should be a shape that defines that against the curve the
tape will exhibit."*

There is, and this module is it. Two results, both measured, and the second
is the useful one.

RECORD LEVEL IS NOT A NEW FREQUENCY SHAPE. It reaches the response through
two physical paths and both land inside the family that already collapsed to
1.58 effectively distinguishable of 6:

    through the recording depth      coherence 0.9995 with thickness loss
    through the transition length    coherence 1.0000 with spacing loss

Adding both to the five-mechanism family moves the effective count from 1.56
to 1.40 - it LOWERS it, because a second copy of a direction already in the
span only worsens the conditioning.

ITS LEVEL DEPENDENCE IS A NEW AXIS, AND THE SELF-DEMAGNETISATION CAP IS THE
ENTIRE REASON. Differentiating the response with respect to level, over five
levels:

    depth law                  with the cap    without it
    linear in level                2.90 of 5         1.01
    saturating, exponential        2.28              1.01
    saturating, tanh               1.93                -

Removing the cap collapses the axis to exactly one direction. So the whole
information content of the level axis comes from `d <= lambda/(2 pi)` binding
at DIFFERENT FREQUENCIES FOR DIFFERENT LEVELS: the crossover between
level-limited and cap-limited recording moves along the band as the level
changes, and that movement is the shape.

And magnetic SATURATION REDUCES the axis, 2.90 to 1.93-2.28, by compressing
the range of depths the levels explore. That is the opposite of the
intuition, and it is why `saturation` defaults to None here.

THE ERROR THIS MODULE EXISTS TO NOT REPEAT. A first estimate read 4.65 of 5
and was wrong. The depth had been written as `cap * g(level)` with
`cap = lambda/(2 pi)`, which makes `x = 2 pi d / lambda = g(level)`
INDEPENDENT OF FREQUENCY - so the response was a constant, its log-shape was
undefined, and normalising a near-zero vector measured numerical noise. What
exposed it was a linear control that should have read 1.00 and read 4.97
instead. `linear_control` below is that control, kept as a function rather
than as a comment, because a control that cannot fail is not a control.
"""

from typing import Callable, Dict, Optional

import numpy as np

# The self-demagnetisation limit, established in docs/MATHEMATICS.md section
# 8a: a coating of depth d holds no wavelength shorter than 2 pi d, so
# recording lambda needs d <= lambda/(2 pi). This is SHALLOWER than the
# lambda/4 sensitivity optimum the format's guide states, so it binds
# everywhere and the optimum is never reachable.
DEMAGNETISATION_DIVISOR = 2.0 * np.pi
OPTIMUM_DIVISOR = 4.0

# A depth in metres that a nominal record level of 1.0 would produce if
# nothing capped it. It is a SCALE, not a constant of the format: the caller
# supplies the tape's own where it is known, and every result below depends
# on the level only through the ratio to this.
NOMINAL_DEPTH_M = 0.15e-6


def _grid(frequency_hz) -> np.ndarray:
    return np.asarray(frequency_hz, dtype=np.float64).ravel()


def demagnetisation_cap(frequency_hz, writing_speed_m_s: float) -> np.ndarray:
    """The deepest recording each wavelength can hold: `lambda / (2 pi)`."""
    wavelength = writing_speed_m_s / np.maximum(_grid(frequency_hz), 1.0)
    return wavelength / DEMAGNETISATION_DIVISOR


def recording_depth(frequency_hz, level: float, writing_speed_m_s: float,
                    nominal_depth_m: float = NOMINAL_DEPTH_M,
                    saturation: Optional[Callable[[float], float]] = None
                    ) -> np.ndarray:
    """The depth actually recorded at this level, in metres.

    The level sets a depth; the cap binds it. THE DEPTH MUST BE A LENGTH
    BEFORE THE CAP IS APPLIED - writing it as a fraction of the cap makes
    `2 pi d / lambda` independent of frequency and the whole response
    constant, which is the error this module's docstring records.

    `saturation` maps a level to a multiplier and defaults to none, because
    it was measured to REDUCE the axis's information rather than create it.
    """
    grid = _grid(frequency_hz)
    drive = float(level) if saturation is None else float(saturation(level))
    wanted = float(nominal_depth_m) * drive * np.ones_like(grid)
    return np.minimum(wanted, demagnetisation_cap(grid, writing_speed_m_s))


def crossover_hz(level: float, writing_speed_m_s: float,
                 nominal_depth_m: float = NOMINAL_DEPTH_M,
                 saturation: Optional[Callable[[float], float]] = None
                 ) -> float:
    """Where the cap takes over from the level, in hertz.

    THE SHAPE ETHAN ASKED FOR, in one number. Below it the level sets the
    depth; above it the medium does. Setting `nominal_depth * level =
    lambda/(2 pi)` and solving,

        f = writing speed / (2 pi * nominal depth * level)

    so the crossover moves DOWN the band as the level rises - and that
    movement along the frequency axis is what makes the level a dimension
    rather than another member of the collapsed family.
    """
    drive = float(level) if saturation is None else float(saturation(level))
    return float(writing_speed_m_s
                 / (DEMAGNETISATION_DIVISOR * float(nominal_depth_m)
                    * max(drive, 1e-12)))


def level_response(frequency_hz, level: float, writing_speed_m_s: float,
                   **kwargs) -> np.ndarray:
    """The thickness-loss response at one record level.

    `(1 - exp(-x)) / x` with `x = 2 pi d / lambda`, the standard thickness
    loss, with the depth taken from `recording_depth` so the cap is present.
    """
    grid = _grid(frequency_hz)
    depth = recording_depth(grid, level, writing_speed_m_s, **kwargs)
    wavelength = writing_speed_m_s / np.maximum(grid, 1.0)
    x = 2.0 * np.pi * depth / wavelength
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 1e-9, (1.0 - np.exp(-x)) / np.maximum(x, 1e-12),
                        1.0)


def level_signature(frequency_hz, level: float, writing_speed_m_s: float,
                    step: float = 0.05, **kwargs) -> np.ndarray:
    """How the response MOVES when the record level moves: the new axis.

    The shape of `d(log response)/d(level)`, which is what carries the
    information the frequency response alone does not. Complex-typed so it
    enters the key beside every other signature, and bounded so it is
    subtractable.
    """
    grid = _grid(frequency_hz)
    above = level_response(grid, level + step, writing_speed_m_s, **kwargs)
    below = level_response(grid, level - step, writing_speed_m_s, **kwargs)
    ratio = np.maximum(above, 1e-12) / np.maximum(below, 1e-12)
    return ratio.astype(np.complex128)


def level_axis(frequency_hz, writing_speed_m_s: float,
               levels=(0.5, 1.0, 1.5, 2.0, 3.0), **kwargs) -> Dict[str, object]:
    """How many independent directions a set of record levels provides.

    Returns the participation ratio of the signatures' singular values - the
    same measure the rest of this arc uses - so it is directly comparable
    with the magnetics' 1.58 of 6.
    """
    shapes = []
    for level in levels:
        value = level_signature(frequency_hz, level, writing_speed_m_s,
                                **kwargs)
        logged = np.log(np.maximum(np.abs(value), 1e-12))
        logged = logged - logged.mean()
        norm = float(np.linalg.norm(logged))
        # A near-zero norm means the response did not move with level, and
        # normalising it would manufacture a direction out of rounding.
        shapes.append(logged / norm if norm > 1e-9 else np.zeros_like(logged))
    stack = np.array(shapes)
    alive = float(np.linalg.norm(stack))
    if not alive > 0:
        return {"effective": 0.0, "levels": list(levels), "signal": 0.0,
                "why": "the response does not move with level at all"}
    singular = np.linalg.svd(stack, compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    return {
        "effective": float(1.0 / np.sum(share ** 2)),
        "count": len(levels),
        "levels": list(levels),
        "singular_values": singular,
        "condition": float(singular[0] / max(singular[-1], 1e-30)),
        "signal": alive,
        "crossovers_hz": [crossover_hz(level, writing_speed_m_s, **kwargs)
                          for level in levels],
        "why": ("the cap binds at a different frequency for every level, and "
                "that movement is the axis"),
    }


def orthogonal_level_signature(frequency_hz, level: float,
                               writing_speed_m_s: float,
                               against, step: float = 0.05,
                               **kwargs) -> np.ndarray:
    """The level signature with the collinear part taken out first.

    THE CORRECTION THREE INDEPENDENT DERIVATIONS AGREED ON. Entered raw the
    entry makes the key WORSE - measured on the real carrier axis, thirteen
    entries give 6.996 effective at condition 11.66 and adding the raw entry
    gives 6.370 at condition 159.02 - because most of its shape duplicates
    the magnetics it sits beside. Projected orthogonal to them first it gives
    7.593 at condition 16.52, and the part that survives is subtractable with
    a minimum magnitude of 0.9935.

    This is the same rule the arc's products already obey: a component that
    still carries a margin puts that margin back into the component that owns
    it the moment it is applied, and the two then fight. What is new here is
    only that the margin belongs to a different family - the level axis
    against the magnetics - rather than to the entry's own axes.

    `against` is the family to project out, as an iterable of complex
    signatures on the same grid.
    """
    def complex_log(values):
        """`log|H| + j arg H`, so a projection removes the direction the
        family actually occupies rather than its magnitude's shadow.

        This took `np.abs` of both sides. That is harmless while every entry
        is phase-free and wrong the moment one is not: `head contact tilt`
        carries pi of phase - a SIGN CHANGE - and projecting onto the
        magnitude of a signature that goes negative removes the wrong
        direction wherever it does.
        """
        values = np.asarray(values).ravel()
        magnitude = np.log(np.maximum(np.abs(values), 1e-12))
        if not np.iscomplexobj(values):
            return magnitude.astype(np.complex128)
        return (magnitude + 1j * np.unwrap(np.angle(values))
                ).astype(np.complex128)

    value = level_signature(frequency_hz, level, writing_speed_m_s,
                            step=step, **kwargs)
    logged = complex_log(value)
    logged = logged - logged.mean()
    for other in against:
        reference = np.asarray(other).ravel()
        if reference.size != logged.size:
            continue
        reference = complex_log(reference)
        reference = reference - reference.mean()
        norm = float(np.linalg.norm(reference))
        if norm > 1e-12:
            unit = reference / norm
            logged = logged - (unit.conj() @ logged) * unit
    # back to a bounded multiplicative signature, so it stays subtractable
    scale = float(np.max(np.abs(logged)))
    if scale > 0:
        logged = logged / scale * 0.5
    return np.exp(logged)


def linear_control(frequency_hz, writing_speed_m_s: float, **kwargs) -> Dict:
    """THE CONTROL, kept as code because it caught a wrong answer once.

    With the cap removed, a depth that is linear in level must give exactly
    ONE direction however many levels are used: the response is then a fixed
    function of a single scaled variable and every derivative has the same
    shape. If this returns anything but one, the construction has gone wrong
    somewhere - which is precisely what happened when the depth was written
    as a fraction of the cap and the control read 4.97 instead of 1.00.
    """
    def uncapped(frequencies, level, speed, **rest):
        grid = _grid(frequencies)
        wavelength = speed / np.maximum(grid, 1.0)
        depth = NOMINAL_DEPTH_M * float(level) * np.ones_like(grid)
        x = 2.0 * np.pi * depth / wavelength
        return np.where(x > 1e-9, (1.0 - np.exp(-x)) / np.maximum(x, 1e-12),
                        1.0)
    shapes = []
    for level in (0.5, 1.0, 1.5, 2.0, 3.0):
        above = uncapped(frequency_hz, level + 0.05, writing_speed_m_s)
        below = uncapped(frequency_hz, level - 0.05, writing_speed_m_s)
        logged = np.log(np.maximum(above, 1e-12)
                        / np.maximum(below, 1e-12))
        logged = logged - logged.mean()
        norm = float(np.linalg.norm(logged))
        shapes.append(logged / norm if norm > 1e-9 else np.zeros_like(logged))
    singular = np.linalg.svd(np.array(shapes), compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    return {
        "effective": float(1.0 / np.sum(share ** 2)),
        "expected": 1.0,
        "passes": bool(abs(1.0 / np.sum(share ** 2) - 1.0) < 0.15),
        "why": ("without the cap a linear depth law has one derivative "
                "shape; anything else means the construction is wrong"),
    }
