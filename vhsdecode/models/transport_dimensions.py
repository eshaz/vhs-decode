"""The transport's rates as TIME-AXIS dimensions, and what a pack that
unwinds does to their identifiability.

`transport_model` predicts eight mechanical parts, each turning at a rate its
own diameter fixes, and says for each whether it disturbs the tape's TENSION,
its SPEED, or both. This module turns those predictions into signatures on the
time axis - a modulation at each rate - so that the same measure the rest of
this arc uses can be asked of them: how many of the eight are actually
distinguishable, and how that number depends on how much tape was observed.

WHY THE ANSWER IS NOT THE MAGNETICS' ANSWER. The tape's six loss mechanisms
are 1.58 effectively distinguishable of 6 at condition 1.06e4, because every
one of them is a function of one dimensionless group - a length over the
recorded wavelength - so they differ only in a RATE and the separability law
calls them collinear. The transport's parts differ in KIND twice over. They
sit at different rates on an axis where a rate difference is resolved rather
than absorbed, and they act on DIFFERENT QUANTITIES: a tension disturbance
moves the RF envelope through the head-to-tape spacing and Wallace's law,
while a speed disturbance moves the time base and barely touches the
envelope. Those two land on the real and the imaginary parts of the same
complex time-axis residual, so they are orthogonal before any rate is
considered.

    measured, five seconds of tape at the head of a T-120, every part's
    phase referenced to the start of the window

        effective 6.3906 of 8    singular values 1.4160, 1.0210, 1.0051,
        rank 7                   1.0016, 0.9997, 0.9948, 0.9747, 0.0000
        condition over the rank 1.4528    worst coherent pair 1.0000

    against the magnetics' 1.58 of 6 at condition 1.06e4.

These are NEW KINDS, not more members of a collinear family. Six of the eight
singular values sit within five per cent of one another, which is what a
nearly orthogonal family looks like; the condition number over the directions
that exist is 1.45, four orders of magnitude better than the magnetics; and
the whole of the shortfall from eight is ONE pair that is exactly degenerate
rather than a family that is merely similar.

THE DEGENERATE PAIRS, AND WHY ONLY ONE OF THEM IS PERMANENT. The entry and
exit guides are declared with the same diameter, so they turn at the same
rate and both act on tension: their signatures are identical, and no amount
of tape separates them. The two reels are the interesting case. They share a
nominal BAND rather than a rate, because the pack radius changes as tape
winds across, and the established result treated that as a second permanent
degeneracy. It is not one:

    the supply pack UNWINDS, so its radius falls and its rate RISES;
    the take-up pack winds ON, so its radius grows and its rate FALLS.

The two rates are therefore equal at exactly one place on the tape - the
midpoint, where both packs hold the same length - and far apart everywhere
else. Measured on a T-120 at SP, with the specified hub, thickness and
length, using a five-second window:

    tape position     supply     take-up    difference  coherence  separable
    0 s (the head)    0.1304 Hz  0.4083 Hz  0.2779 Hz   0.142      yes
    1843 s            0.1481 Hz  0.2283 Hz  0.0802 Hz   0.314      yes
    2764 s            0.1601 Hz  0.1969 Hz  0.0368 Hz   0.791      no
    3686 s (middle)   0.1756 Hz  0.1757 Hz  0.0001 Hz   1.000      no
    5529 s            0.2281 Hz  0.1481 Hz  0.0799 Hz   0.315      yes
    7371 s (the end)  0.4071 Hz  0.1304 Hz  0.2766 Hz   0.138      yes

So the reel pair is separable over the outer three-quarters of the tape in
five seconds, and only within about a quarter of an hour of the midpoint does
it need the drift itself. There the two rates sweep in OPPOSITE directions at
1.94e-05 Hz per second each, so their difference opens at twice that and the
pair separates once

    2 |df/dt| T > 1/T,   i.e.   T > sqrt( 1 / (2 |df/dt|) ) = 160.5 s

Confirmed against the signatures: at the midpoint the two packs' coherence
reads 1.0000 at five seconds, 0.9813 at sixty, 0.3865 at a hundred and sixty
and 0.1886 at two hundred, crossing a half where the closed form puts it, and
the effective count of all eight rises 5.32, 5.37, 6.22, 6.36 across the
same windows.

A RATE THAT MOVES IS AN EASIER IDENTIFICATION PROBLEM THAN ONE THAT IS
FIXED, not a harder one. A pair at a fixed common rate never separates
however long the observation, because the signatures are identical and the
limit of their coherence is one; a pair whose rates drift apart separates
after a finite and computable time that goes as the square root of the
inverse drift. `docs/ELLIPTICAL_COLLAPSE.md` section 7.3 named this drift as
an identifying signature it could not then predict; the pack law below
predicts it in closed form from four specified numbers.

WHAT THIS CONFIRMS AND WHAT IT CORRECTS. The published table - 9 of 28 pairs
at 0.23 s, 13 at 0.5 s, 17 at 1.0 s, 25 at 2.0 s, 26 at 5.0 s and saturating
there - is reproduced EXACTLY by `rayleigh_pairs`, under the criterion it was
computed with, which is Rayleigh's on the nominal rates with both reels held
at the middle of their band. Every figure agrees and so do the two pairs left
over. Two things extend it.

    THE AXIS A PART ACTS ON SEPARATES PAIRS NO RATE DIFFERENCE REACHES. Ten
    of the twenty-eight pairs put a tension part against a speed part, and
    those are orthogonal in the shortest window there is. Measured at 0.23 s:
    9 separable by rate, 10 by axis, 2 in both, and 9 + 10 - 2 = 17 is the
    measured signature count exactly.

    SATURATION AT 26 BELONGS TO THE FIXED-RATE APPROXIMATION. With the pack
    law the reels separate too, so the count reaches 27 of 28 from two
    seconds on, and the effective count 5.32 of 8 under the approximation
    becomes 6.39. What remains is the guide pair, which is degenerate by
    construction and needs a different witness rather than more tape.

WHAT NO OBSERVATION OF THIS KIND CAN DO. `after_capstan` is not identifiable
here. The capstan servo regulates an upstream disturbance and passes a
downstream one, but at any single rate that is a scalar gain, and every
measure in this arc unit-normalises, so the gain is absorbed. Separating the
two sides needs the servo's loop response, which neither SMPTE 32M nor the
JVC guide states.

SOURCES. The reel geometry, tape thickness and tape length are specification
figures and are cited at their constants. The eight parts, their diameters
and their `acts_on` and `after_capstan` declarations are taken from
`transport_model.TRANSPORT` unchanged, where six of the eight are marked
`certain: False` and are typical values rather than specified ones - so every
rate below except the drum's inherits that status.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import transport_model


# --------------------------------------------------------------------------
# the specification, cited
# --------------------------------------------------------------------------

# JVC Video Technical Guide VTG82063 section 1, as carried by
# head_model.FORMAT_MECHANICS: VHS NTSC SP runs the tape past the head at
# 33.35 mm/s +/- 0.5%.
LINEAR_TAPE_SPEED_M_S = 0.03335

# SMPTE 32M-2004 table 1: tape thickness 19.0 +1/-2 um for T-120 through
# T-30, and a T-120's length is 246 m +3/-0. This is the PACK PITCH - one
# revolution of the reel adds one tape thickness to the pack radius - and it
# is the whole tape including its base, not the recording depth.
TAPE_THICKNESS_M = 19.0e-6
T120_TAPE_LENGTH_M = 246.0

# JVC VTG82063 section 1.1.25 sub-section 3, and SMPTE 32M-2004 table 1: the
# reel's outside diameter is 89 +/- 0.2 mm and the hub over 60 minutes is
# 26 +/- 0.15 mm. Cassettes of 60 minutes or less use a 62 or 70 mm hub, so
# the hub radius is a property of the cassette and not of the format.
HUB_RADIUS_M = 0.013
FLANGE_RADIUS_M = 0.0445

# JVC VTG82063 section 4: DRUM FG is 240 Hz for NTSC against a drum rotation
# of 30 Hz, so the drum's tachometer has eight poles. It is named here
# because it is the natural fine clock for a transport measurement and
# because any drum-servo ripple lands on it rather than on the drum rate.
DRUM_FG_POLES = 8

# How finely the time axis is sampled, as a multiple of the fastest rate
# being modelled. Eight is a working choice and not a specification: the
# rates here top out at the drum's 29.97 Hz while the decoder's own time-base
# residual is sampled once per line at 15.734 kHz, so nothing in this module
# is anywhere near a sampling limit and the multiple only sets arithmetic
# cost. `distinguishable` is measured to move by 3.0e-04 between four samples
# of the fastest cycle and a hundred and twenty-eight.
SAMPLES_PER_FASTEST_CYCLE = 8

# The size of the modulation, in nepers of log envelope and radians of
# log-signature phase. ASSUMED: no specification states how much envelope a
# given eccentricity produces, and it would take a measurement on a
# particular deck to fix it. It sets the SIZE of a signature and not its
# shape, and every measure below unit-normalises, so no reported figure
# depends on it: measured, the effective count moves by less than 1e-06 over
# three decades of it, from 0.005 to 2.0.
MODULATION_DEPTH = 0.05

# For the head drum, which `transport_model` declares as acting on BOTH
# tension and speed, the ratio of its timing modulation to its envelope
# modulation. ASSUMED equal. It is the only figure in this module that
# could in principle affect a reported count, because for a part acting on
# one quantity alone the scale is absorbed by normalisation. Measured, it
# does not: moving it over four decades, from 0.01 to 100, changes the
# effective count of the eight by less than 1e-04, because the drum's
# 29.97 Hz is resolved from every other part at every duration examined and
# its internal balance therefore never enters an overlap. It would matter
# only if some other part sat within `1/T` of the drum rate.
DRUM_TIMING_RATIO = 1.0


# WHERE THESE ACT IN THE CHAIN. The eight parts exist on BOTH machines and
# the two sides are at different positions, which is why the names carry the
# machine and why one entry would be wrong.
#
# The RECORDING machine's transport disturbed the tape while it was being
# written, so it acts after the record electronics and before the medium -
# between `record level dependence` at 11 and `head contact tilt` at 20 in
# `interference.COMPONENT_ORDER`. Its disturbance is frozen into the tape and
# no playback adjustment reaches it.
#
# The PLAYBACK machine's transport acts as the tape is read, after everything
# the medium did to the signal and before the capture at 30. It is the one a
# correction can undo, and a correction must undo it FIRST, since
# `ordered_key` removes the last-applied entry first.
#
# `interference.position_of` matches on a prefix, so the two entries below
# cover all sixteen names that `signatures` produces.
CHAIN_POSITIONS: Dict[str, int] = {
    "transport (record)": 12,
    "transport (playback)": 24,
}


def _grid(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).ravel()


def _shaped(result: np.ndarray, source):
    """A scalar back for a scalar argument, an array for an array.

    The pack laws are called both ways - once for a position, and once for a
    whole window - and a caller who passed one number should get one number
    rather than a length-one array it has to unpack.
    """
    return float(result[0]) if np.ndim(source) == 0 else result


# --------------------------------------------------------------------------
# the pack, in closed form
# --------------------------------------------------------------------------


def full_pack_radius_m(hub_radius_m: float = HUB_RADIUS_M,
                       tape_length_m: float = T120_TAPE_LENGTH_M,
                       tape_thickness_m: float = TAPE_THICKNESS_M) -> float:
    """The radius a wound pack reaches, from the three specified figures.

    A wound pack is an annulus of tape seen edge on, so its area is the tape's
    length times its thickness:

        pi (r_full^2 - r_hub^2) = L t   hence   r_full = sqrt(r_hub^2 + Lt/pi)

    Measured on a T-120 at the specified hub, thickness and length: 40.70 mm,
    which sits inside the specified 44.5 mm flange radius with 3.8 mm to
    spare. That the three independently specified numbers land inside the
    fourth is `winding_control`'s first check.
    """
    return float(np.sqrt(float(hub_radius_m) ** 2
                         + float(tape_length_m) * float(tape_thickness_m)
                         / np.pi))


def pack_radius_m(position_s, winding: str = "take-up",
                  linear_speed_m_s: float = LINEAR_TAPE_SPEED_M_S,
                  hub_radius_m: float = HUB_RADIUS_M,
                  tape_length_m: float = T120_TAPE_LENGTH_M,
                  tape_thickness_m: float = TAPE_THICKNESS_M) -> np.ndarray:
    """The pack radius at a position along the tape, in metres.

    `position_s` is where the transport has reached, in seconds of play from
    the start of the tape, which is the axis a long capture is indexed by.
    The take-up pack starts at the hub and grows; the supply pack starts full
    and shrinks. Both follow the same area law with the sign reversed:

        r(p) = sqrt( r_hub^2   + t v p / pi )      taking up
        r(p) = sqrt( r_full^2  - t v p / pi )      supplying
    """
    position = _grid(position_s)
    swept = (float(tape_thickness_m) * float(linear_speed_m_s) * position
             / np.pi)
    if str(winding).lower().startswith("supply"):
        full = full_pack_radius_m(hub_radius_m, tape_length_m,
                                  tape_thickness_m)
        inside = np.maximum(full ** 2 - swept, float(hub_radius_m) ** 2)
    else:
        inside = float(hub_radius_m) ** 2 + swept
    return _shaped(np.sqrt(inside), position_s)


def reel_revolutions(position_s, winding: str = "take-up",
                     **kwargs) -> np.ndarray:
    """How many turns the reel has made by this position, in closed form.

    The rate is `v / (2 pi r)` and the radius moves under the area law, and
    the integral of the one over the other collapses completely:

        Phi(p) = integral of v / (2 pi r(u)) du  =  |r(p) - r(0)| / t

    which is the statement that ONE REVOLUTION ADDS ONE TAPE THICKNESS TO THE
    PACK RADIUS. That identity is known independently of the integral, so it
    is what `winding_control` checks the integral against - and a factor of
    two or of pi mislaid in the area law fails it immediately.
    """
    start = pack_radius_m(0.0, winding, **kwargs)
    here = pack_radius_m(position_s, winding, **kwargs)
    thickness = float(kwargs.get("tape_thickness_m", TAPE_THICKNESS_M))
    return np.abs(here - float(start)) / thickness


def reel_rate_hz(position_s, winding: str = "take-up",
                 linear_speed_m_s: float = LINEAR_TAPE_SPEED_M_S,
                 **kwargs) -> np.ndarray:
    """The reel's rotation rate at this position: `v / (2 pi r(p))`."""
    radius = pack_radius_m(position_s, winding,
                           linear_speed_m_s=linear_speed_m_s, **kwargs)
    return float(linear_speed_m_s) / (2.0 * np.pi * radius)


def reel_drift_hz_per_s(position_s, winding: str = "take-up",
                        linear_speed_m_s: float = LINEAR_TAPE_SPEED_M_S,
                        tape_thickness_m: float = TAPE_THICKNESS_M,
                        **kwargs) -> np.ndarray:
    """How fast the reel's rate is moving, in hertz per second.

    Differentiating `f = v / (2 pi r)` under `dr/dp = t v / (2 pi r)`:

        df/dp = - t v^2 / (4 pi^2 r^3)

    positive for the supply pack, whose radius falls, and negative for the
    take-up pack. Measured at a T-120's midpoint radius of 30.21 mm:
    1.94e-05 Hz per second, so the two reels' difference opens at 3.88e-05 Hz
    per second and the pair separates after 160 s of observation.
    """
    radius = pack_radius_m(position_s, winding,
                           linear_speed_m_s=linear_speed_m_s,
                           tape_thickness_m=tape_thickness_m, **kwargs)
    size = (float(tape_thickness_m) * float(linear_speed_m_s) ** 2
            / (4.0 * np.pi ** 2 * np.asarray(radius) ** 3))
    signed = size if str(winding).lower().startswith("supply") else -size
    return float(signed) if np.ndim(position_s) == 0 else signed


def pack_midpoint_s(linear_speed_m_s: float = LINEAR_TAPE_SPEED_M_S,
                    tape_length_m: float = T120_TAPE_LENGTH_M,
                    **kwargs) -> float:
    """Where on the tape the two packs hold the same length, and therefore
    the same radius and the same rate. It is the tape's own midpoint, and it
    is the only place the reel pair is degenerate at a fixed rate."""
    return 0.5 * float(tape_length_m) / float(linear_speed_m_s)


# --------------------------------------------------------------------------
# the eight parts, with the reels' rate taken from the pack
# --------------------------------------------------------------------------


def parts(position_s: float = 0.0,
          linear_speed_m_s: float = LINEAR_TAPE_SPEED_M_S,
          field_rate_hz: float = 59.94, fixed_rate: bool = False,
          **kwargs) -> List[Dict[str, object]]:
    """The eight mechanical parts at one position along the tape.

    `transport_model.rotation_rates` supplies every part, its order in the
    path, whether it acts on tension or speed, whether the capstan's servo
    stands between it and the head, and whether its diameter is a
    specification or a typical value. All this adds is the reels' rate, which
    that model can only give as the middle of a band because it has no pack
    law; with the specified hub, thickness and length the rate at a stated
    position is exact, and its drift with it.

    `fixed_rate` leaves the reels at the middle of their band, which is the
    approximation the established saturation result was computed under. It is
    kept so the difference can be measured rather than asserted.
    """
    out: List[Dict[str, object]] = []
    for entry in transport_model.rotation_rates(linear_speed_m_s,
                                                field_rate_hz):
        item = dict(entry)
        item["winding"] = None
        item["drift_hz_per_s"] = 0.0
        if item.get("kind") == "reel" and not fixed_rate:
            winding = ("supply" if "supply" in str(item["name"])
                       else "take-up")
            item["winding"] = winding
            item["rate_hz"] = float(reel_rate_hz(
                position_s, winding, linear_speed_m_s=linear_speed_m_s,
                **kwargs))
            item["drift_hz_per_s"] = float(reel_drift_hz_per_s(
                position_s, winding, linear_speed_m_s=linear_speed_m_s,
                **kwargs))
            item["pack_radius_m"] = float(pack_radius_m(
                position_s, winding, linear_speed_m_s=linear_speed_m_s,
                **kwargs))
            item["rate_note"] = ("from the pack radius at this position; the "
                                 "rate MOVES, and the movement identifies it")
        item["position_s"] = float(position_s)
        out.append(item)
    return out


def part_names(**kwargs) -> List[str]:
    return [str(entry["name"]) for entry in parts(**kwargs)]


# --------------------------------------------------------------------------
# the signature: a modulation at each rate, on the quantity the part acts on
# --------------------------------------------------------------------------


def time_grid(duration_s: float, sample_rate_hz: Optional[float] = None,
              fastest_rate_hz: float = 29.97) -> np.ndarray:
    """The time axis a measurement of this length provides.

    Given no sampling rate, eight samples of the fastest modelled cycle. The
    physical grid is the decoder's own per-line time-base residual at
    15.734 kHz, which is five hundred times faster than anything modelled
    here, so this choice is arithmetic economy and not a limit.
    """
    if sample_rate_hz is None:
        sample_rate_hz = SAMPLES_PER_FASTEST_CYCLE * float(fastest_rate_hz)
    count = max(int(round(float(duration_s) * float(sample_rate_hz))), 8)
    return np.arange(count, dtype=np.float64) / float(sample_rate_hz)


def cycles(time_s, rate_hz: float, drift_hz_per_s: float = 0.0,
           winding: Optional[str] = None, position_s: float = 0.0,
           exact_pack: bool = True, **kwargs) -> np.ndarray:
    """The accumulated turns of a part over the observation.

    A part at a fixed rate accumulates `f t`. A reel does not, and the
    difference is the whole of the reel result, so it is taken exactly rather
    than as a linear chirp wherever the pack law applies: the closed form of
    `reel_revolutions` is used, referenced to the window's start. Setting
    `exact_pack` False falls back to the linear chirp `f t + (df/dt) t^2 / 2`,
    which is what the exact law reduces to over a short window and which the
    test suite checks it against.
    """
    t = _grid(time_s)
    if winding and exact_pack:
        return (reel_revolutions(float(position_s) + t, winding, **kwargs)
                - reel_revolutions(float(position_s), winding, **kwargs))
    return float(rate_hz) * t + 0.5 * float(drift_hz_per_s) * t ** 2


def log_signature(time_s, rate_hz: float, acts_on: str,
                  phase_rad: float = 0.0,
                  depth: float = MODULATION_DEPTH,
                  drum_timing_ratio: float = DRUM_TIMING_RATIO,
                  turns: Optional[np.ndarray] = None,
                  **kwargs) -> np.ndarray:
    """The LOGARITHM of a part's signature: envelope in the real part, time
    base in the imaginary part.

    This is the complex time-axis residual the arc already uses - `time as
    the hsync deviation plus j the burst-phase deviation` - with the two
    halves carrying the two things a transport can do:

    A TENSION disturbance changes the head-to-tape pressure, hence the
    spacing, hence the recovered level by Wallace's law, and it does that
    without moving the time base. It is a real log-envelope modulation, in
    nepers, and it enters as a cosine.

    A SPEED disturbance moves the time base and barely touches the envelope.
    It enters as a SINE and not as a cosine, and that quadrature is derived
    rather than chosen: the time-base displacement a measurement sees is the
    INTEGRAL of the speed error, so a fractional speed error
    `e cos(2 pi f t)` displaces the time base by `e sin(2 pi f t) / (2 pi f)`.
    The same integral also says a slow part displaces the time base far more
    than a fast one at the same fractional error - a factor of 107 between
    the reels and the drum - which matters for whether a part is DETECTED but
    not for whether it is DISTINGUISHED, since every measure here
    unit-normalises.

    The head drum acts on both, so its signature carries a cosine in the
    envelope and a sine in the time base, and it is the one part whose two
    halves stand in a ratio no specification fixes. `drum_timing_ratio` is
    that ratio, assumed equal.

    PHASE. Every part's disturbance has an arbitrary phase - wherever the
    eccentricity happened to be when the window opened - and `phase_rad`
    defaults to zero for all of them. That is the CONSERVATIVE convention and
    not a convenience: two sinusoids at one rate sum to a single sinusoid, so
    two parts at a common rate cannot be attributed however they are phased,
    and giving them independent phases would report them as two directions
    when no measurement could ever separate them. Measured: independent
    random phases raise the effective count of the eight from
    5.3231 of 8 at rank 6 to 6.2646 at rank 8, under the fixed-rate
    approximation where two pairs genuinely share a rate, and the whole of
    that rise is this false separation. `resolution_control` catches it on
    two parts alone, where the expected answer is known without this module.
    """
    t = _grid(time_s)
    turns = cycles(t, rate_hz, **kwargs) if turns is None else _grid(turns)
    angle = 2.0 * np.pi * turns + float(phase_rad)
    acts = str(acts_on).lower()
    envelope = (float(depth) * np.cos(angle)
                if acts in ("tension", "both") else np.zeros_like(t))
    scale = float(drum_timing_ratio) if acts == "both" else 1.0
    timing = (float(depth) * scale * np.sin(angle)
              if acts in ("speed", "both") else np.zeros_like(t))
    return envelope + 1j * timing


def signature(time_s, rate_hz: float, acts_on: str, **kwargs) -> np.ndarray:
    """A part's signature as a SUBTRACTABLE multiplicative modification.

    `exp(log_signature)`, whose magnitude is `exp(envelope)` and is therefore
    bounded away from zero everywhere, so its logarithm is finite everywhere
    and it can be taken off by subtraction in the log domain. A bare
    modulation `depth * cos(...)` would cross zero twice per cycle and fail
    `interference.subtractable`; a bounded exponential does not.
    """
    return np.exp(log_signature(time_s, rate_hz, acts_on, **kwargs))


def signatures(duration_s: float, position_s: float = 0.0,
               sample_rate_hz: Optional[float] = None,
               machine: str = "playback",
               phases: Optional[Dict[str, float]] = None,
               fixed_rate: bool = False,
               **kwargs) -> Dict[str, np.ndarray]:
    """Every mechanical part's signature on one time grid.

    The names carry the machine, because the eight parts exist on BOTH decks
    and act at different points in the chain: the recording machine's
    transport disturbed the tape as it was written and its disturbance is
    frozen into the medium, while the playback machine's is added as the tape
    is read. `interference.COMPONENT_ORDER` needs one position for each side
    and the names below are prefixed so that `position_of` finds them.

    `fixed_rate` holds the reels at their nominal rate instead of following
    the pack, which is what the established saturation result assumed; it is
    kept so the difference can be measured rather than asserted.
    """
    _, logs = log_signatures(duration_s, position_s, sample_rate_hz, machine,
                             phases, fixed_rate, **kwargs)
    return {name: np.exp(value) for name, value in logs.items()}


def log_signatures(duration_s: float, position_s: float = 0.0,
                   sample_rate_hz: Optional[float] = None,
                   machine: str = "playback",
                   phases: Optional[Dict[str, float]] = None,
                   fixed_rate: bool = False,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """The same set, as the logarithms, with the time grid beside them.

    Returned separately from `signatures` because the identifiability measure
    works on the logarithm and recovering it through `log|s| + j arg s` would
    wrap the phase wherever the timing modulation exceeds pi radians.
    """
    shape = {key: kwargs.pop(key) for key in
             ("depth", "drum_timing_ratio", "exact_pack")
             if key in kwargs}
    entries = parts(position_s, fixed_rate=fixed_rate, **kwargs)
    fastest = max(float(entry["rate_hz"]) for entry in entries)
    t = time_grid(duration_s, sample_rate_hz, fastest)
    prefix = f"transport ({machine}) "
    out: Dict[str, np.ndarray] = {}
    for entry in entries:
        name = str(entry["name"])
        out[prefix + name] = log_signature(
            t, float(entry["rate_hz"]), str(entry["acts_on"]),
            phase_rad=float((phases or {}).get(name, 0.0)),
            drift_hz_per_s=float(entry.get("drift_hz_per_s", 0.0)),
            winding=entry.get("winding"), position_s=position_s,
            **shape, **kwargs)
    return t, out


# --------------------------------------------------------------------------
# how many are distinguishable, and after how much tape
# --------------------------------------------------------------------------


def _stacked(entries: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """Unit directions in the real space of dimension 2L.

    Real and imaginary parts stacked, because each of these is the signature
    of a REAL physical parameter - the size of one part's eccentricity - and
    an envelope modulation is a different mechanism from a time-base
    modulation of the same shape. The mean is removed from each part first,
    since a constant is a level and not a shape.
    """
    names, rows = [], []
    for name, value in entries.items():
        vector = np.asarray(value).ravel()
        real = np.real(vector) - np.real(vector).mean()
        imag = np.imag(vector) - np.imag(vector).mean()
        row = np.concatenate([real, imag])
        norm = float(np.linalg.norm(row))
        if norm > 1e-12:
            names.append(name)
            rows.append(row / norm)
    return names, (np.array(rows) if rows else np.zeros((0, 0)))


def distinguishable(duration_s: float, position_s: float = 0.0,
                    rank_floor: float = 1e-9,
                    **kwargs) -> Dict[str, object]:
    """How many of the eight parts a measurement of this length tells apart.

    The same question and the same measure as everywhere else in this arc -
    the participation ratio of the singular values of the unit-normalised
    signatures - so the answer stands directly beside the magnetics' 1.58 of
    6 at condition 1.06e4.

    Measured over five seconds at the head of a T-120:

        effective   6.3906 of 8          rank 7
        singular    1.4160, 1.0210, 1.0051, 1.0016, 0.9997, 0.9948,
                    0.9747, 0.0000
        worst pair  entry guide against exit guide, coherence 1.0000
        condition   infinite over all eight; 1.4528 over the rank

    and at the tape's midpoint, where the two packs momentarily share a rate:

        effective   5.3245 of 8          rank 7 (the seventh is 1.1e-03,
        singular    1.4340, 1.3966, 1.0051, 1.0001, 0.9965, 0.9948,
                    0.0011, 0.0000       arithmetic rather than evidence)

    and under the fixed-rate approximation the established result used, which
    holds both reels at the middle of their band everywhere:

        effective   5.3231 of 8          rank 6      condition 1.4432

    against the magnetics' 1.58 of 6 at condition 1.06e4.

    `condition` is reported twice on purpose. Over all eight it is infinite,
    because the guide pair is EXACTLY degenerate and the family's rank is at
    most seven; reporting that alone would say the set is hopeless when in
    fact the directions it does span are conditioned four orders of magnitude
    better than the magnetics. `condition_of_the_rank` is the honest figure
    and the infinity is the correct statement about the degenerate pair.

    `rank` is the numerical rank at `rank_floor` relative to the largest
    singular value, so it counts a direction the arithmetic can see whether
    or not the evidence supports it. `effective` and `separable_pairs` are
    the figures to read; the midpoint's seventh singular value of 1.1e-03 is
    exactly the case where the rank flatters the answer.

    THE FIGURES ABOVE DO NOT DEPEND ON THE ASSUMED CONSTANTS. Measured:
    changing `depth` over three decades moves the effective count by less
    than 1e-06, since every row is unit-normalised; changing
    `drum_timing_ratio` over four decades moves it by less than 1e-04,
    because the drum's rate is resolved from every other part at every
    duration examined and its internal balance therefore never enters an
    overlap; changing the sampling from four to a hundred and twenty-eight
    samples of the fastest cycle moves it by 3.0e-04.
    """
    _, entries = log_signatures(duration_s, position_s, **kwargs)
    names, stack = _stacked(entries)
    if stack.size == 0:
        return {"names": [], "count": 0, "effective": 0.0}
    values = np.linalg.svd(stack, compute_uv=False)
    share = values ** 2 / max(float((values ** 2).sum()), 1e-30)
    live = values[values > float(rank_floor) * max(float(values[0]), 1e-30)]
    gram = np.abs(stack @ stack.T)
    np.fill_diagonal(gram, 0.0)
    worst = np.unravel_index(int(np.argmax(gram)), gram.shape)
    return {
        "names": names,
        "count": len(names),
        "effective": float(1.0 / np.sum(share ** 2)),
        "rank": int(live.size),
        "singular_values": values,
        "condition": float(values[0] / values[-1]) if values[-1] > 0
        else float("inf"),
        "condition_of_the_rank": float(live[0] / live[-1]) if live.size
        else 0.0,
        "worst_pair": (names[worst[0]], names[worst[1]]),
        "worst_coherence": float(gram[worst]),
        "duration_s": float(duration_s),
        "position_s": float(position_s),
        "why": ("the transport's parts differ in KIND - in rate on an axis "
                "that resolves rates, and in the quantity they disturb - "
                "where the tape's losses differ only in a rate of decay"),
    }


def identifiability_vs_duration(durations_s: Sequence[float] = (
        0.23, 0.5, 1.0, 2.0, 5.0, 30.0, 120.0),
        position_s: float = 0.0, **kwargs) -> List[Dict[str, object]]:
    """The effective count against how much tape was observed.

    This is the practical table: for a decode of a given length, how many of
    the eight mechanical parts are separable at all. Measured at the head of
    a T-120, with the reels following the pack, beside the rate-only count
    the established result was stated in:

        observed  resolution  effective of 8  rank  pairs  by rate alone
        0.23 s    4.348 Hz    2.235           7     17/28  9/28
        0.50 s    2.000 Hz    3.830           7     24/28  13/28
        1.00 s    1.000 Hz    4.778           7     25/28  17/28
        2.00 s    0.500 Hz    6.041           7     27/28  25/28
        5.00 s    0.200 Hz    6.391           7     27/28  26/28
        30.0 s    0.033 Hz    6.399           7     27/28  26/28
        120. s    0.008 Hz    6.400           7     27/28  26/28

    THE PRACTICAL READING. Every decode this arc has examined is about
    0.22 seconds, the first row, where the rate-only count resolves nothing
    mechanical at all and even the signature count owes ten of its seventeen
    pairs to the axis split rather than to any rate being resolved. Two
    seconds reaches 27 of 28 and 6.04 of 8, and five seconds is where the
    effective count settles. Beyond that nothing changes at the head of the
    tape, and what more tape buys is only the reel pair near the midpoint,
    which needs 160 s.

    It saturates at 6.40 of 8 and 27 of 28, and what is left is the guide
    pair, which is degenerate by construction and needs a different witness.
    """
    out = []
    for duration in durations_s:
        fit = distinguishable(duration, position_s, **kwargs)
        pairs = separable_pairs(duration, position_s, **kwargs)
        out.append({
            "duration_s": float(duration),
            "resolution_hz": 1.0 / float(duration),
            "effective": fit["effective"],
            "rank": fit["rank"],
            "count": fit["count"],
            "separable_pairs": pairs["separable"],
            "pairs": pairs["total"],
            "worst_pair": fit["worst_pair"],
        })
    return out


# --------------------------------------------------------------------------
# the pairwise question the established result was stated in
# --------------------------------------------------------------------------


def rayleigh_pairs(duration_s: float, position_s: float = 0.0,
                   fixed_rate: bool = True, **kwargs) -> Dict[str, object]:
    """The established count, under the criterion it was computed with.

    Two rates separate once the observation is long enough to resolve their
    difference, `1/T < |f1 - f2|`, and nothing else is considered.

    THE ESTABLISHED RESULT IS CONFIRMED, EXACTLY. Reproduced: 9 of 28 at
    0.23 s, 13 at 0.5 s, 17 at 1.0 s, 25 at 2.0 s, 26 at 5.0 s, and 26 still
    at 120 s - saturating, as `docs/ELLIPTICAL_COLLAPSE.md` section 7.3 and
    `docs/BROADCAST_VTR_COLLAPSE.md` section 5 both state it, and with the
    same two pairs left over. The unresolved pairs at five seconds are the
    two guides at a rate difference of exactly zero and the two reels, also
    at exactly zero - because this criterion holds the reel rate at the
    MIDDLE OF ITS BAND, which is all `transport_model` can report and what
    that table used. `fixed_rate` defaults True here for that reason.

    Set it False and the reels take their rate from the pack at
    `position_s`, at which point they are 0.2779 Hz apart at the head of the
    tape and the count at 0.23 s is unchanged at 9 while five seconds reaches
    27 of 28. The saturation at 26 belongs to the approximation, not to the
    transport.
    """
    entries = parts(position_s, fixed_rate=fixed_rate, **kwargs)
    rates = [float(entry["rate_hz"]) for entry in entries]
    names = [str(entry["name"]) for entry in entries]
    resolution = 1.0 / max(float(duration_s), 1e-12)
    separable, unresolved = 0, []
    for i in range(len(rates)):
        for j in range(i + 1, len(rates)):
            if abs(rates[i] - rates[j]) > resolution:
                separable += 1
            else:
                unresolved.append((names[i], names[j],
                                   abs(rates[i] - rates[j])))
    total = len(rates) * (len(rates) - 1) // 2
    return {"separable": separable, "total": total,
            "resolution_hz": resolution, "unresolved": unresolved,
            "why": "Rayleigh's criterion on the nominal rates alone"}


def separable_pairs(duration_s: float, position_s: float = 0.0,
                    coherence_limit: float = 0.5,
                    **kwargs) -> Dict[str, object]:
    """The same count taken on the SIGNATURES rather than on the rates.

    Two parts are separable when their unit signatures are not nearly
    parallel, and that is a weaker requirement than resolving their rates for
    two independent reasons.

    THE AXIS COSTS NO TAPE AT ALL. A part acting on tension and one acting on
    speed are orthogonal whatever their rates, because one modulates the
    envelope and the other the time base. Ten of the twenty-eight pairs are
    of that kind, and they are separable in the shortest window there is.

    AND THE OVERLAP CRITERION IS SOFTER THAN RAYLEIGH'S. Two sinusoids
    separated by `d` over a window `T` overlap as `sinc(dT)`, which passes a
    half at `dT` near 0.6 rather than at 1, so the signatures separate about
    1.7 times sooner than the rate test admits.

    The two decompose the count exactly at the shortest window. At 0.23 s:
    9 pairs separable by rate, 10 orthogonal by axis, 2 counted in both, and
    9 + 10 - 2 = 17, which is what is measured. At 1.0 s the union of the two
    predicts 22 and 25 are measured, the extra three being the softer
    criterion. From two seconds on it is 27 of 28 and stays there.

    `coherence_limit` is the largest overlap two signatures may have and
    still be called separable; a half is the value at which one signature
    explains a quarter of the other's variance. Measured, the count is flat
    from 0.3 to 0.6 and moves by one pair at 0.7.
    """
    _, entries = log_signatures(duration_s, position_s, **kwargs)
    names, stack = _stacked(entries)
    gram = np.abs(stack @ stack.T)
    total = len(names) * (len(names) - 1) // 2
    separable, unresolved = 0, []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if gram[i, j] < float(coherence_limit):
                separable += 1
            else:
                unresolved.append((names[i], names[j], float(gram[i, j])))
    return {"separable": separable, "total": total,
            "unresolved": unresolved,
            "coherence_limit": float(coherence_limit),
            "why": ("two parts separate by their rate OR by the quantity "
                    "they disturb, and the second costs no tape at all")}


# --------------------------------------------------------------------------
# the reels: a rate that moves
# --------------------------------------------------------------------------


def reel_pair(position_s: float, duration_s: float,
              sample_rate_hz: Optional[float] = None,
              **kwargs) -> Dict[str, object]:
    """Whether the two reels are separable here, and by what.

    Three answers, and they do not always agree, which is the point. The
    Rayleigh test asks whether the two rates differ by more than the window
    resolves. The drift test asks whether the two rates SWEEP apart within
    the window by more than it resolves, which is a different question and
    the one that matters at the tape's midpoint where the rates are equal.
    The measured coherence between the two drifting signatures answers both
    at once and is what `separable_pairs` uses.

    Measured on a T-120 at SP with a five-second window: separable at every
    position except a stretch of roughly a quarter of an hour either side of
    the midpoint, and separable there too once the window reaches 160 s. At
    the midpoint the coherence falls 1.0000 at five seconds, 0.9813 at sixty,
    0.3865 at a hundred and sixty, 0.1886 at two hundred - crossing the half
    within a couple of seconds of where the closed form puts it.

    `beyond_the_tape` marks a window that would run past the end of the reel,
    where the supply pack clamps at its hub and the rate stops moving. The
    figures are then an artefact of the clamp and not a measurement.
    """
    supply = float(reel_rate_hz(position_s, "supply", **kwargs))
    takeup = float(reel_rate_hz(position_s, "take-up", **kwargs))
    drift = abs(float(reel_drift_hz_per_s(position_s, "supply", **kwargs))) \
        + abs(float(reel_drift_hz_per_s(position_s, "take-up", **kwargs)))
    resolution = 1.0 / max(float(duration_s), 1e-12)
    if sample_rate_hz is None:
        sample_rate_hz = SAMPLES_PER_FASTEST_CYCLE * max(supply, takeup)
    t = time_grid(duration_s, sample_rate_hz)
    rows = {}
    for winding, rate in (("supply", supply), ("take-up", takeup)):
        rows[winding] = log_signature(
            t, rate, "tension", winding=winding, position_s=position_s,
            **kwargs)
    _, stack = _stacked(rows)
    coherence = float(abs(stack[0] @ stack[1])) if stack.shape[0] == 2 else 1.0
    played = pack_midpoint_s(**{
        k: v for k, v in kwargs.items()
        if k in ("linear_speed_m_s", "tape_length_m")}) * 2.0
    return {
        "position_s": float(position_s),
        "duration_s": float(duration_s),
        "supply_hz": supply,
        "take_up_hz": takeup,
        "difference_hz": abs(supply - takeup),
        "drift_hz_per_s": drift,
        "resolution_hz": resolution,
        "beyond_the_tape": bool(float(position_s) + float(duration_s)
                                > played),
        "separated_by_rate": bool(abs(supply - takeup) > resolution),
        "separated_by_drift": bool(drift * float(duration_s) > resolution),
        "coherence": coherence,
        "separable": bool(coherence < 0.5),
        "why": ("the packs move in opposite directions, so the pair is "
                "degenerate at one point on the tape and nowhere else"),
    }


def reel_separation_s(position_s: Optional[float] = None,
                      **kwargs) -> Dict[str, object]:
    """How long the reel pair must be watched, in closed form.

    At the tape's midpoint the two rates are equal and only their opposite
    drift separates them. Their difference after a window of length `T` is
    `2 |df/dt| T`, and it is resolved when that exceeds `1/T`:

        T > sqrt( 1 / (2 |df/dt|) )

    where `|df/dt|` is taken as the SUM of the two packs' drifts, since they
    move in opposite directions and the difference opens at their sum.

    Measured on a T-120 at SP: 63.0 s at either end of the tape, where the
    packs' drifts are 2.516e-04 Hz per second between them, and 160.5 s at
    the midpoint, where they are 3.881e-05. The midpoint is the hard case and
    it is also the only place the closed form is needed, since everywhere
    else the two rates already differ by more than a short window resolves.
    Checked against the signatures themselves: at the midpoint the measured
    coherence crosses a half between 60 s and 160 s, which is where 160.5 s
    puts it.

    THE SHAPE OF THIS RESULT IS THE USEFUL PART. A pair of parts at a fixed
    common rate never separates, however long the observation: their
    signatures are identical and the limit of the coherence is one. A pair
    whose rates DRIFT apart separates after a finite time that goes as the
    square root of the inverse drift. So a moving rate is not a harder
    identification problem than a fixed one; it is the only version of this
    problem that has a solution at all.
    """
    if position_s is None:
        position_s = pack_midpoint_s(**{
            k: v for k, v in kwargs.items()
            if k in ("linear_speed_m_s", "tape_length_m")})
    drift = abs(float(reel_drift_hz_per_s(position_s, "supply", **kwargs))) \
        + abs(float(reel_drift_hz_per_s(position_s, "take-up", **kwargs)))
    return {
        "position_s": float(position_s),
        "drift_hz_per_s": drift,
        "seconds": float(np.sqrt(1.0 / max(drift, 1e-30))),
        "why": ("2 |df/dt| T > 1/T; a fixed pair at a common rate never "
                "separates, a drifting pair separates in finite time"),
    }


def across_the_tape(positions_s: Optional[Sequence[float]] = None,
                    duration_s: float = 5.0, **kwargs
                    ) -> List[Dict[str, object]]:
    """The reel pair's separability as a function of tape position.

    The axis a two-hour capture is actually indexed by, and the one the
    established result never had. Measured on a T-120 at SP with a
    five-second window, the two packs' coherence against how far the
    transport has run:

        position   supply     take-up    difference  coherence  separable
        0 s        0.1304 Hz  0.4083 Hz  0.2779 Hz   0.142      yes
        921 s      0.1384 Hz  0.2818 Hz  0.1434 Hz   0.084      yes
        1843 s     0.1481 Hz  0.2283 Hz  0.0802 Hz   0.314      yes
        2764 s     0.1601 Hz  0.1969 Hz  0.0368 Hz   0.791      no
        3686 s     0.1756 Hz  0.1757 Hz  0.0001 Hz   1.000      no
        4607 s     0.1968 Hz  0.1602 Hz  0.0366 Hz   0.791      no
        5529 s     0.2281 Hz  0.1481 Hz  0.0799 Hz   0.315      yes
        6450 s     0.2814 Hz  0.1384 Hz  0.1429 Hz   0.082      yes
        7371 s     0.4071 Hz  0.1304 Hz  0.2766 Hz   0.138      yes

    So five seconds anywhere in the outer three-quarters of the tape
    separates the reels, and only a stretch of roughly plus or minus fifteen
    minutes about the midpoint needs the drift. The coherence is not monotone
    in the difference - it follows `sinc(dT)` and passes through the sinc's
    first sidelobe - which is why the criterion is taken on the signatures
    rather than on the rate difference.

    The window is held inside the tape: the last position returned leaves
    `duration_s` of tape to run, since a pack law asked past the end of the
    reel would clamp at the hub and report a rate that has stopped moving.
    """
    if positions_s is None:
        end = 2.0 * pack_midpoint_s(**{
            k: v for k, v in kwargs.items()
            if k in ("linear_speed_m_s", "tape_length_m")})
        positions_s = np.linspace(0.0, max(end - float(duration_s), 0.0), 9)
    return [reel_pair(float(position), duration_s, **kwargs)
            for position in positions_s]


# --------------------------------------------------------------------------
# the controls
# --------------------------------------------------------------------------


def resolution_control(rate_hz: float = 1.0, separation_cycles: float = 0.02,
                       resolved_cycles: float = 4.0,
                       duration_s: float = 20.0) -> Dict[str, object]:
    """THE CONTROL. Rayleigh's criterion, whose answer is known without this
    module and which a construction that manufactures directions fails.

    Two parts acting on the same quantity, given the same phase, whose rates
    differ by `d`. Over a window of length `T` Fourier resolution says that
    when `d T` is far below one the two are one direction and when it is well
    above one they are two - and neither number comes from anything here.
    Expected 1.00 and 2.00.

    THIS IS WHAT IT CATCHES. The failure mode a time-axis construction has is
    manufacturing separation out of bookkeeping: spectral leakage at the ends
    of the window, a mean left in, a normalisation applied per sample rather
    than per component, or - the one that actually bites - giving each part
    an independent phase, which makes two parts at the SAME rate read as two
    directions when two sinusoids at one rate sum to a single sinusoid and no
    measurement could ever attribute them. Any of those raises the
    unresolved figure above one. The magnetics module records a linear
    control that should have read 1.00 and read 4.97, and this is the same
    instrument on this module's axis.

    Measured: unresolved 1.0026 against an expected 1.00 at 0.02 cycles of
    separation, resolved 2.0000 against an expected 2.00 at 4 cycles, and
    with the two given phases a quarter turn apart the unresolved figure
    rises to 1.9922 - the false separation, caught.

    That last figure is why the phase convention in `log_signature` is not a
    convenience. Applied to the whole set it costs the same: with independent
    random phases and the reels held at the middle of their band, the
    effective count of the eight reads 6.26 of 8 at rank 8 where the true
    answer under that approximation is 5.32 at rank 6, and every part of the
    difference is two parts at one rate being counted twice.
    """
    def effective(gap_cycles, phases=(0.0, 0.0)):
        gap = float(gap_cycles) / float(duration_s)
        t = time_grid(duration_s, SAMPLES_PER_FASTEST_CYCLE
                      * (float(rate_hz) + abs(gap)))
        rows = {
            "a": log_signature(t, float(rate_hz), "tension",
                               phase_rad=phases[0]),
            "b": log_signature(t, float(rate_hz) + gap, "tension",
                               phase_rad=phases[1]),
        }
        _, stack = _stacked(rows)
        values = np.linalg.svd(stack, compute_uv=False)
        share = values ** 2 / max(float((values ** 2).sum()), 1e-30)
        return float(1.0 / np.sum(share ** 2))

    unresolved = effective(separation_cycles)
    resolved = effective(resolved_cycles)
    phased = effective(separation_cycles, phases=(0.0, 0.5 * np.pi))
    return {
        "unresolved": unresolved,
        "unresolved_expected": 1.0,
        "resolved": resolved,
        "resolved_expected": 2.0,
        "unresolved_with_independent_phases": phased,
        "passes": bool(abs(unresolved - 1.0) < 0.05
                       and abs(resolved - 2.0) < 0.05),
        "phase_convention_matters": bool(phased > 1.5),
        "why": ("two rates closer than the window resolves are one "
                "direction and two well beyond it are two; neither figure "
                "comes from this module, and a construction that "
                "manufactures separation fails the first"),
    }


def winding_control(**kwargs) -> Dict[str, object]:
    """THE SECOND CONTROL, on the pack law rather than on the signatures.

    Three checks, each against something known without the integral.

    ONE REVOLUTION PER TAPE THICKNESS. The closed form for the accumulated
    turns, `Phi = |r(p) - r(0)| / t`, was obtained by integrating
    `v / (2 pi r)` under the area law. It is also true for a reason that owes
    nothing to the integral: each turn lays one thickness of tape onto the
    pack, so the radius grows by one thickness per turn. Checked against a
    numerical integration of the instantaneous rate. A factor of two or of pi
    mislaid in the area law fails this and nothing else would show it.
    Measured over ten minutes of take-up: 212.10114599946863 turns by the
    closed form against 212.10114599957 numerically over two hundred thousand
    points, a relative departure of 4.8e-13.

    THE PACK FITS INSIDE THE REEL. The hub radius, the tape thickness and the
    tape length are three independently specified figures and the flange
    radius is a fourth, and the pack the first three wind must fit inside it.
    Measured: 40.70 mm against a 44.5 mm flange.

    THE TAPE IS AS LONG AS THE PLAY TIME SAYS. Two hours at 33.35 mm/s is
    240.12 m of tape against the 246 m SMPTE 32M states, 2.39% apart, the
    surplus being leader and the length's own +3/-0 tolerance.
    """
    thickness = float(kwargs.get("tape_thickness_m", TAPE_THICKNESS_M))
    speed = float(kwargs.get("linear_speed_m_s", LINEAR_TAPE_SPEED_M_S))
    length = float(kwargs.get("tape_length_m", T120_TAPE_LENGTH_M))
    span = 600.0
    closed = float(reel_revolutions(span, "take-up", **kwargs))
    grid = np.linspace(0.0, span, 200001)
    numeric = float(np.trapezoid(reel_rate_hz(grid, "take-up", **kwargs),
                                 grid))
    full = full_pack_radius_m(
        float(kwargs.get("hub_radius_m", HUB_RADIUS_M)), length, thickness)
    played = length / speed
    return {
        "turns_closed_form": closed,
        "turns_numerical": numeric,
        "turns_departure": float(abs(closed - numeric) / max(numeric, 1e-30)),
        "full_pack_radius_m": full,
        "flange_radius_m": FLANGE_RADIUS_M,
        "fits_inside_the_flange": bool(full < FLANGE_RADIUS_M),
        "play_time_s": played,
        "length_from_two_hours_m": speed * 7200.0,
        "length_specified_m": length,
        "length_departure": float(abs(speed * 7200.0 - length) / length),
        "passes": bool(abs(closed - numeric) / max(numeric, 1e-30) < 1e-4
                       and full < FLANGE_RADIUS_M
                       and abs(speed * 7200.0 - length) / length < 0.05),
        "why": ("one revolution adds one tape thickness to the pack radius, "
                "which is known independently of the integral that produced "
                "the closed form"),
    }
