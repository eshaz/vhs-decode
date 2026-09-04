"""The two RF stages: the machine that wrote the tape and the machine that
read it, as separate components.

Ethan named the chain as four links, and two of them are RF:

    Picture chain (TV to VCR); VCR chain: VCR -> Tape; Tape -> VCR;
    Capture chain: VCR -> capture card; capture card profile -> complete
    limits of the sample rate i.e. 8bits, and 40mhz

`capture_profile` is the fourth link and `picture_components` the first. The
two in the middle are two DIFFERENT MACHINES - the recorder has its own
emphasis, its own amplifier, its own head and its own record current; the
player has its own head, its own preamplifier, its own equalisation and its
own de-emphasis - and a single measured response is their product. This
module builds them apart, and then measures whether taking them apart was
possible at all.

THE ANSWER TO THE CENTRAL QUESTION IS NO, AND THE MEASUREMENTS SAY WHY.
Four separate results, each reproducible from a function below:

  * THE EMPHASIS PAIR IS EXACTLY ONE DIRECTION. The format specifies the
    playback de-emphasis as the inverse of the record pre-emphasis, so their
    log magnitudes are exact negatives - measured through `cascade_control`,
    `|log|pre * de|| = 2.2e-16` nepers. A shape and its negation are one
    direction, so a departure in the emphasis fits the recording machine and
    the playback machine equally well. This is the emphasis's analogue of
    the carrier law annihilating `gain_db` and `spacing_m`.

  * THE WAVELENGTH LOSSES COMPOSE IN THE EXPONENT, SO ONLY THEIR SUM IS
    IDENTIFIABLE. The record head's transition-length loss is `exp(-2 pi
    a/lambda)` and the playback head's spacing loss is `exp(-2 pi
    d/lambda)`; their product is `exp(-2 pi (a+d)/lambda)`. Measured
    coherence of the two signatures: 1.000000000. The difference `a - d` is
    annihilated exactly.

  * A BAND LIMIT AT EITHER END OF THE CHAIN IS ONE DIRECTION. The record
    amplifier's upper corner and the playback preamplifier's upper corner
    both sit above the RF band, where a single pole's log-magnitude
    derivative goes as `f^2` whatever the corner. Measured coherence 0.9994.
    This is the same small-argument collapse that makes gap and azimuth read
    as one mechanism far from the sinc null - the only one of the four that
    is a limit rather than an identity, and the only one a wider band would
    break.

  * PHASE ADDS NOTHING PER MACHINE. Every mechanism here except a transport
    delay is minimum phase, and the Hilbert transform is unitary on
    mean-removed signals, so stacking `[log|H| ; arg H]` doubles the Gram
    and leaves the participation ratio unchanged - measured, 5.899435
    against 5.899485 on a six-row ensemble. Phase buys one direction for the
    PAIR (the total delay) and none for either machine, because a record
    delay and a playback delay are also only ever observed as their sum.

WHAT THE TEST CAPTURES DO SEPARATE, MEASURED. `/testdata/test_patterns/vhs/`
holds both a record-side and a playback-side RF tap of every zaroff pattern,
at both SP and EP, from one Sony SLV-778HF (its readme names the taps as
CN261 pin 1 and pin 2). That is one of the separators the question named - an
RF tap at the record side - and it does give a real measurement, but it cuts
the chain in the wrong place and it gives less than it first appears:

    the RATIO of the two taps IS a response.        0.46 nepers rms at SP,
    Same content through two points, so             0.74 at EP, of which 0.80
    `pb - rec` is the transfer from the record      and 0.84 repeats across
    head's terminals to the preamplifier's output   eight different patterns
    the record tap's OWN spectrum IS NOT a          its 1.67 nepers rms is
    response. It is the FM spectrum of the          the modulated carrier's
    picture through the record chain, and           own shape far more than
    nothing divides the two apart without a         the electronics', and its
    reference                                       0.91 agreement across
                                                    patterns says only that
                                                    an FM spectrum is
                                                    dominated by its carrier
    the record HEAD and the playback HEAD stay      and on this dataset they
    inside the ratio                                are the same physical
                                                    head, one machine having
                                                    done both

So the pairing splits four links into two blocks, not four; it reduces the
playback tap's shape by 3.3x at SP and 1.65x at EP; and the block it isolates
is the one the tap sits at the end of, which is not the same as isolating the
record machine's own response. It does NOT separate the two machines by
itself. `separation_design` puts each proposed experiment
through the two-way model `y = R[deck] + T[tape] + P[deck]` and reports which
contrasts are estimable, and the answer is uncomfortable and clear:

    one deck, one tape                      nothing
    a second tape on the same deck          only how the tapes differ
    a second deck playing the same tape     only how the PLAYBACK machines
                                            differ
    the same stock recorded on two decks    only how the RECORD machines
                                            differ
    an RF tap at the record side            the record stage alone, up to
                                            the grand level
    a known reference recorded on the tape  the tape alone

    R against P on ONE deck                 NONE of the five, on its own

The two machines' contributions appear in every playback together, so no
number of tapes and no number of decks reaches `R - P` for a single machine.
It takes TWO of the five at once, and exactly one pairing does it: A RECORD
TAP AND A KNOWN REFERENCE ON THE TAPE. The tap gives the record stage, the
reference gives the medium, and the playback stage is what is left. The test
directory already holds the first half.

THE SPEED LEVER IS 96.5% RECORD PROCESSING AND 3.5% MAGNETICS, so it is
confounded almost totally. `speed_confound` computes both arms: the tape-side
change a VHS speed switch makes is a track width of 58 -> 19.3 um and a
writing speed of 5.800 -> 5.822 m/s (0.384%), which the head model turns into
0.0104 nepers rms of log response, against 0.2978 nepers rms measured between
SP and EP at the playback tap. The remaining 96.5% is the record machine's
own emphasis switch - the very stage one arm has and the other does not.
LP against EP is worse still, 0.0018 nepers. What breaks it is the record
tap, which observes the processing arm with no tape in it at all.

THE EFFECTIVE COUNT. Against the magnetics baseline of 1.58 effectively
distinguishable of 6 (reproduced here by `magnetics_baseline` as 1.576 at
condition 3.0e4), adding all six RF stage entries gives 1.803 of 12. Six more
members buy 0.23 of a direction. THE TWO STAGES ARE THE SAME COLLINEAR FAMILY
SEEN TWICE, with two exceptions that are worth their place: the record
current's level dependence (coherence at most 0.42 with anything else, the
axis `magnetic` already established) and the playback equalisation's corner
inside the band (at most 0.65). Everything else is another value of the one
dimensionless group `length / recorded wavelength`.

THE CONTROL IS `cascade_control`, and it can fail. One mechanism entered once
on each side is physically one shape, so the pair must span exactly ONE
direction; and the specified emphasis round trip must be unity exactly.
Measured: correct construction 1.0000 and 2.2e-16 nepers; the same with one
side's wavelength taken from the linear tape speed instead of the writing
speed, 1.9996; with the de-emphasis returning a logarithm where a response is
wanted, 2.06 nepers; with it returned in decibels rather than nepers, 1.41
nepers. Three realistic construction errors, three failures.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import head_model, interference, magnetic

# --------------------------------------------------------------------------
# The specification. Every figure below is a standard's or the repository's
# own, named where it comes from. Anything that is an EXAMPLE for a worked
# signature is called a default and the caller is expected to replace it with
# the measurement; it is never a constant of the format.
# --------------------------------------------------------------------------

# IEC 60774-1 gives the VHS luma FM carriers as sync tip and peak white, with
# the deviation their difference. Both line standards carry 1.0 MHz.
VHS_CARRIER_HZ: Dict[str, Tuple[float, float]] = {
    "525": (3.4e6, 4.4e6),
    "625": (3.8e6, 4.8e6),
}
# IEC 60774-3 gives S-VHS as 5.4 to 7.0 MHz with 1.6 MHz of deviation,
# IDENTICALLY for both line standards - which is why the format needs no
# per-system entry here where VHS does.
SVHS_CARRIER_HZ: Tuple[float, float] = (5.4e6, 7.0e6)


def deviation_hz(carriers: Tuple[float, float]) -> float:
    """The FM deviation: the span between sync tip and peak white."""
    return float(carriers[1]) - float(carriers[0])


# IEC 774-1 (1994) page 67, as transcribed by the repository's own
# `vhsdecode/format_defs/vhs.py`: the main video emphasis is a FIRST-ORDER
# shelf with an RC time constant of 1.3 us and a resistor divider ratio of
# 4:1, so a gain factor of 5. PAL and NTSC share it. The de-emphasis is the
# same network read the other way, which is why the pair is exactly one
# direction and not two.
MAIN_EMPHASIS_TIME_CONSTANT_S = 1.30e-6
MAIN_EMPHASIS_GAIN_FACTOR = 5.0

# Which tape speeds carry the EXTRA record-side emphasis, from the
# repository's own speed table (`fill_rfparams_vhs_shared`,
# `use_sub_deemphasis = [False, True, True, True]` indexed by tape speed):
# VHS applies it at LP, EP and SLP and not at SP, and S-VHS applies it always
# (`fill_rfparams_svhs_shared`). This is the confound in the speed lever: the
# two arms of an SP-versus-EP comparison differ by a whole processing stage.
EXTRA_EMPHASIS_BY_SPEED: Dict[str, Dict[str, bool]] = {
    "VHS": {"SP": False, "LP": True, "EP": True, "SLP": True},
    "SVHS": {"SP": True, "LP": True, "EP": True, "SLP": True},
}

# The non-linear emphasis in the DECODER'S OWN parametrization, which is the
# only place its numbers exist: a first-order high pass whose output is
# clipped between two limits and then subtracted (`filter_model
# .apply_deemphasis_chain`, and `vhsdecode/format_defs/vhs.py`). These are
# that file's NTSC VHS values, carried here as the starting point a fit
# moves, not as constants of the format - the file's own comment says they
# are "probably not correct".
NONLINEAR_HIGHPASS_HZ = 820e3
NONLINEAR_LIMIT_LOW = -20000.0
NONLINEAR_LIMIT_HIGH = 5000.0

# The recording chain applies its emphases in the reverse of the order the
# decoder undoes them. The decoder's de-emphasis chain is main, then the
# non-linear stage, then the sub stage (`filter_model.STAGE_ORDER`, and
# `process.py` at the three call sites), and a chain is undone last-applied-
# first, so the record machine applied sub, then non-linear, then main.
RECORD_EMPHASIS_ORDER: Tuple[str, str, str] = ("sub", "nonlinear", "main")

# One neper is 20/ln(10) decibels. Stated here rather than imported so that a
# reader can check the unit of every return value against one place; it is
# the same figure `head_model.DB_PER_NEPER` and `interference.DB_PER_NEPER`
# carry.
DB_PER_NEPER = 20.0 / np.log(10.0)

# The NTSC VHS SP writing speed, READ BACK from `head_model
# .FORMAT_MECHANICS` rather than repeated as a literal. JVC VTG82063 section
# 1 states it directly as 5.80 m/s - it is not the drum's surface speed, and
# deriving it from the circumference gives 5.877, which is 1.3% high and
# carries into every wavelength - and a second copy of it here could drift
# away from the first silently. It is a DEFAULT for the worked signatures; a
# caller at another format or speed passes `mechanics_at_speed`'s result.
DEFAULT_WRITING_SPEED_M_S = head_model.writing_speed(
    head_model.mechanics_for("VHS", "NTSC", "SP") or {})

# A stop-band rejection no real amplifier exceeds, used to keep the pole and
# zero forms SUBTRACTABLE at the edge of a band. `interference
# .vestigial_sideband` takes the same precaution for the same reason: a
# response clipped to zero has no logarithm, and a real filter has a finite
# rejection rather than an infinite one.
STOP_BAND_FLOOR = 1e-3


def _grid(frequency_hz) -> np.ndarray:
    return np.asarray(frequency_hz, dtype=np.float64).ravel()


def _positive(frequency_hz) -> np.ndarray:
    """The grid with zero excluded. A head has no response at DC and a
    coupled amplifier has none either, so a caller reaching there is asking
    for a value no stage in this chain has."""
    return np.maximum(_grid(frequency_hz), 1.0)


# --------------------------------------------------------------------------
# The mechanics a speed switch changes, which is less than it looks
# --------------------------------------------------------------------------


def mechanics_at_speed(tape_speed: str = "SP", system: str = "NTSC",
                       rf_params: Optional[Dict[str, float]] = None
                       ) -> Dict[str, float]:
    """The head-to-tape mechanics at one tape speed.

    `head_model.FORMAT_MECHANICS` tabulates SP, where the guide states the
    writing speed directly. The slower speeds are DERIVED from it rather than
    given a second table, because only one thing changes: the drum turns at
    the same rate whatever the tape speed, so the writing speed is the drum's
    surface speed less the tape's own, and the track pitch follows the tape
    speed. The repository already carries the recorded track width per speed
    (`video_track_width`, 58.0 / 29.0 / 19.3 um on NTSC VHS, from JVC
    VTG82063 section 1), so the linear tape speed is read back from that
    ratio and no new figure is introduced.

    The result is the quantitative reason the speed lever is confounded: from
    SP to EP the writing speed moves 5.800 to 5.822 m/s, which is 0.384%, and
    `docs/COMPONENT_MAPPINGS.md` section 3 has already established that
    writing speed is inert to identifiability and track width inert to three
    decimals. Both of the tape-side changes a speed switch makes are in the
    inert set.
    """
    base = head_model.mechanics_for("VHS", system, "SP", rf_params)
    if base is None:
        raise ValueError("no tabulated mechanics for VHS %s SP" % system)
    widths = {"SP": 58.0, "LP": 29.0, "EP": 19.3, "SLP": 19.3}
    speed = str(tape_speed).upper()
    if speed not in widths:
        raise ValueError("unknown VHS tape speed %r" % tape_speed)
    mechanics = dict(base)
    drum = (np.pi * base["drum_diameter_m"]
            * base["drum_revolutions_per_second"])
    linear = base["linear_tape_speed_m_s"] * widths[speed] / widths["SP"]
    mechanics["linear_tape_speed_m_s"] = float(linear)
    # the stated SP figure wins at SP; the others move with it by the tape's
    # own speed, which is the only term that differs
    mechanics["writing_speed_m_s"] = float(
        base["writing_speed_m_s"] + (base["linear_tape_speed_m_s"] - linear))
    mechanics["track_width_m"] = float(widths[speed]) * 1e-6
    mechanics["tape_speed"] = speed
    mechanics["drum_surface_speed_m_s"] = float(drum)
    return mechanics


# --------------------------------------------------------------------------
# THE RECORD SIDE. Pre-emphasis, the record amplifier, the record head's own
# response, and the record current that sets the depth.
# --------------------------------------------------------------------------


def main_pre_emphasis(baseband_hz,
                      time_constant_s: float = MAIN_EMPHASIS_TIME_CONSTANT_S,
                      gain_factor: float = MAIN_EMPHASIS_GAIN_FACTOR
                      ) -> np.ndarray:
    """The main record pre-emphasis, as the standard's network gives it.

    IEC 774-1 (1994) page 67: a first-order shelf built from an RC of 1.3 us
    and a 4:1 resistor divider, so a gain factor `G` of 5. A divider with a
    capacitor across the series arm has a zero at `1/tau` and a pole at
    `G/tau`, hence

        H(f) = (1 + j 2 pi f tau) / (1 + j 2 pi f tau / G)

    which is unity at DC, `G` well above the shelf, and whose geometric mean
    corner is `sqrt(G) / (2 pi tau)`. That is 273.756 kHz and 13.979 dB, the
    two figures `vhsdecode/format_defs/vhs.py` carries as `deemph_mid` and
    `deemph_gain` - so this form and the decoder's shelf are the same network
    in two parametrizations, and either can check the other.

    SUBTRACTABLE by construction: the magnitude runs between 1 and `G` and
    never approaches zero, so its logarithm is finite everywhere.

    THIS SIGNATURE IS ON THE BASEBAND VIDEO ABSCISSA, not the RF one. The
    emphasis acts on the modulating signal, ahead of the FM modulator, so an
    inner product against a head loss would be memory layout rather than
    coherence - the trap `docs/COMPONENT_MAPPINGS.md` section 9 records.
    `ABSCISSA` states which grid every entry belongs on.
    """
    w = 2j * np.pi * _grid(baseband_hz)
    tau = float(time_constant_s)
    return (1.0 + w * tau) / (1.0 + w * tau / float(gain_factor))


def saturation_gain(ratio) -> np.ndarray:
    """The incremental gain of a symmetric limiter, as a describing function.

    A limiter of half-range `L` driven by a sinusoid of amplitude `a` passes
    the fundamental with gain

        N(a) = 1                                            a <= L
        N(a) = (2/pi) [ arcsin(L/a) + (L/a) sqrt(1-(L/a)^2) ]  a > L

    which is the standard sinusoidal-input describing function of a
    saturation element (Gelb and Vander Velde). `ratio` is `a/L`.

    It is what makes the non-linear emphasis a LEVEL AXIS rather than another
    frequency shape: the limit binds at a different frequency for every
    drive, and that movement along the band is the shape - exactly the
    structure `magnetic` found for the self-demagnetisation cap, where
    removing the cap collapsed the level axis from 2.90 directions to 1.01.
    """
    value = np.asarray(ratio, dtype=np.float64)
    out = np.ones_like(value)
    beyond = value > 1.0
    if np.any(beyond):
        inverse = 1.0 / np.maximum(value[beyond], 1.0)
        out[beyond] = (2.0 / np.pi) * (
            np.arcsin(inverse) + inverse * np.sqrt(
                np.maximum(1.0 - inverse * inverse, 0.0)))
    return out


def _high_pass(baseband_hz, corner_hz: float) -> np.ndarray:
    w = 2j * np.pi * _grid(baseband_hz) / float(corner_hz)
    return w / (1.0 + w)


def nonlinear_pre_emphasis(baseband_hz, drive: float,
                           corner_hz: float = NONLINEAR_HIGHPASS_HZ,
                           limit: Optional[float] = None) -> np.ndarray:
    """The record-side non-linear emphasis, at one drive level.

    The decoder's own form, read backwards. On playback it high-passes the
    signal at `corner_hz`, clips the result between two limits, and subtracts
    it; on record the same high-passed part was ADDED, so

        H(f; A) = 1 + N( A |HP(f)| / L ) HP(f)

    with `HP` the first-order high pass and `N` the limiter's describing
    function. Below the limit `N` is one and the stage is an ordinary
    high-frequency boost; above it the boost fades, and the frequency where
    it starts to fade moves DOWN the band as the drive rises.

    THE ASYMMETRY IS AN ASSUMPTION HERE, AND IT IS LABELLED. The decoder's
    limits are not symmetric - -20000 and +5000 in its own units - and an
    asymmetric clipper also produces a mean shift and even harmonics, which a
    describing function of the fundamental does not carry. `limit` therefore
    defaults to the half-range `(high - low)/2`, which is the symmetric
    element closest to it; a fit that needs the offset must carry it
    separately.

    IT IS EP AND LP ONLY ON VHS, which is `EXTRA_EMPHASIS_BY_SPEED`, and it
    is the whole confound in the speed lever - see `speed_confound`.
    """
    if limit is None:
        limit = 0.5 * (NONLINEAR_LIMIT_HIGH - NONLINEAR_LIMIT_LOW)
    high = _high_pass(baseband_hz, corner_hz)
    gain = saturation_gain(abs(float(drive)) * np.abs(high) / max(float(limit),
                                                                 1e-12))
    return 1.0 + gain * high


def nonlinear_emphasis_level_dependence(baseband_hz, low_drive: float,
                                        high_drive: float, **kwargs
                                        ) -> np.ndarray:
    """How the non-linear emphasis MOVES between two drive levels.

    Entered as the level dependence and not as the response at a level, for
    the reason `docs/COMPONENT_MAPPINGS.md` section 8a states and measured
    twice already: two curves of nearly the same shape are one direction, and
    a second copy of a direction already in the span lowers the effective
    count and worsens the conditioning. What is new about this stage is that
    it varies with LEVEL, so that is what enters.

    A ratio of two responses, so it is strictly positive and SUBTRACTABLE.
    """
    above = nonlinear_pre_emphasis(baseband_hz, high_drive, **kwargs)
    below = nonlinear_pre_emphasis(baseband_hz, low_drive, **kwargs)
    return above / below


def sub_pre_emphasis_level_dependence(baseband_hz, speed: str = "EP",
                                      low_level_db: float = -20.0,
                                      high_level_db: float = 0.0
                                      ) -> np.ndarray:
    """The S-VHS sub pre-emphasis, entered as its level dependence.

    The table is IEC 60774-3 table 3, already transcribed in
    `interference.SUB_EMPHASIS_DB` with its tolerances recorded in that
    function's docstring - six frequencies, four input levels, separately for
    SP and for EP/LP, with tolerances from +-0.30 dB at the low frequencies
    to +-0.70 dB at 3 and 5 MHz. It is read here rather than transcribed
    again.

    RETURNED AS A RESPONSE, NOT AS A LOGARITHM. `interference.sub_emphasis`
    returns nepers - the log - while every other entry in that module returns
    the response itself, so a caller that mixes them takes the logarithm of a
    logarithm. Exponentiating the difference here keeps this module's
    convention single: every signature is a complex RESPONSE whose magnitude
    is bounded away from zero. `cascade_control` fails if that convention is
    ever broken, which is what it is for.
    """
    above = interference.sub_emphasis(baseband_hz, high_level_db, speed).real
    below = interference.sub_emphasis(baseband_hz, low_level_db, speed).real
    return np.exp(np.asarray(below - above, dtype=np.float64)
                  ).astype(np.complex128)


def record_amplifier(rf_hz, low_corner_hz: float = 0.3e6,
                     high_corner_hz: float = 8.0e6,
                     floor: float = STOP_BAND_FLOOR) -> np.ndarray:
    """The record amplifier, as a band pass with per-device corners.

    NOT A SPECIFIED SHAPE, and it is important to say so. The standard fixes
    the recording characteristic - the carriers, the deviation, the emphasis
    - and leaves the amplifier that drives the head to the manufacturer. So
    the FORM here is the format's (one coupling zero at the bottom, one
    band-limiting pole at the top, which is what a record driver into a
    rotary transformer is) and the CORNERS are per-device parameters a fit
    supplies. The defaults are examples on the scale of the VHS luma band,
    not constants.

    ITS UPPER CORNER IS IN THE PLAYBACK PREAMPLIFIER'S NULL SPACE. Both
    corners sit above the RF band, and a single pole's log-magnitude
    derivative with respect to its corner goes as `f^2` whatever the corner
    is, so the two are collinear to three decimals - `null_space` measures
    0.9994. A band limit at either end of the chain is one direction, not
    two, and this is the only one of the module's nulls that is a limit
    rather than an identity: a band wide enough to reach either corner would
    break it.
    """
    f = _grid(rf_hz)
    w = 1j * f
    response = (w / float(low_corner_hz)) / (1.0 + w / float(low_corner_hz)) \
        / (1.0 + w / float(high_corner_hz))
    return _floored(response, floor)


def _floored(response: np.ndarray, floor: float) -> np.ndarray:
    """Hold a response's magnitude above a stated stop-band rejection.

    SUBTRACTABLE FORM, the same precaution `interference.vestigial_sideband`
    takes: a response that reaches zero has no logarithm there, so it cannot
    be subtracted, and a real amplifier has a finite rejection anyway. The
    phase is carried through untouched.
    """
    magnitude = np.abs(response)
    kept = np.maximum(magnitude, float(floor))
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = np.where(magnitude > 0, response / np.maximum(magnitude,
                                                                  1e-300),
                             1.0)
    return (kept * direction).astype(np.complex128)


def record_head_write(rf_hz, transition_length_m: float = 0.10e-6,
                      writing_speed_m_s: float = DEFAULT_WRITING_SPEED_M_S
                      ) -> np.ndarray:
    """The record head's own response: the transition it can write.

    Writing is not reading. A reproduce head reads through a gap and loses
    `sinc(g/lambda)`; a record head lays down a magnetisation reversal of
    finite length `a`, set by the trailing-edge field gradient and the
    coating's demagnetising field, and the shortest wavelength it can write
    is bounded by it. The loss is the same Fourier statement as Wallace's,

        exp(-2 pi a / lambda),      lambda = writing speed / f

    THIS IS THE RECORD STAGE'S CENTRAL NULL SPACE RESULT. It is the same
    functional form as the playback head's spacing loss, and the two compose
    in the exponent: `exp(-2 pi a/lambda) exp(-2 pi d/lambda) = exp(-2 pi
    (a+d)/lambda)`. Only the SUM is identifiable, the difference is
    annihilated exactly, and `null_space` measures their coherence as
    1.000000000. `magnetic` had already found the transition length 1.0000
    coherent with spacing loss through the record-level path; this is the
    same statement reached from the record side.
    """
    wavelength = float(writing_speed_m_s) / _positive(rf_hz)
    return np.exp(-2.0 * np.pi * float(transition_length_m) / wavelength
                  ).astype(np.complex128)


def record_current_level_dependence(
        rf_hz, level: float = 1.0,
        writing_speed_m_s: float = DEFAULT_WRITING_SPEED_M_S,
        **kwargs) -> np.ndarray:
    """The record current that sets the depth, as its level dependence.

    Delegated to `magnetic.level_signature` rather than rebuilt: that module
    established the whole result, including that the record level's frequency
    shape is 0.9995 coherent with thickness loss and 1.0000 with spacing
    loss - another member of the collapsed family - while its MOVEMENT with
    level is a direction nothing else supplies, and that the movement exists
    only because the self-demagnetisation cap binds at a different frequency
    for every level.

    It is the record stage's best entry by a wide margin. Measured against
    every other entry in this module and the six magnetics, its coherence
    never exceeds 0.42, where every other electronic entry reaches 0.99.
    """
    return magnetic.level_signature(rf_hz, level, writing_speed_m_s, **kwargs)


# --------------------------------------------------------------------------
# THE PLAYBACK SIDE. The head's response, the preamplifier, the equalisation
# and the de-emphasis.
# --------------------------------------------------------------------------


def playback_head_differentiation(rf_hz, reference_hz: Optional[float] = None
                                  ) -> np.ndarray:
    """The reproduce head differentiates: THE ONE MECHANISM WITH NO RECORD-
    SIDE COUNTERPART.

    A reproduce head's output is the rate of change of the flux it links, so
    its response rises as `j omega` - six decibels per octave in magnitude
    and exactly a quarter turn of phase, with no output at all at DC. The
    record head has nothing like it: the recorded magnetisation follows the
    record CURRENT, and a current is not a derivative of anything.

    So this is the asymmetry the whole component is looking for, and it is
    real. It is also, on its own, weak: its log magnitude is `log f`, whose
    demeaned coherence with a flat gain over the VHS luma band is 0.950,
    because a logarithm over one octave and a half is nearly a constant plus
    a slope. It separates on the WIDEST band the RF occupies, which is the
    actionable rule `docs/COMPONENT_MAPPINGS.md` section 3 already gives.

    `reference_hz` only sets where the response is unity; it is a gain, and a
    gain is in the null space of everything here, so it defaults to the
    band's own geometric mean purely to keep the magnitudes near one.
    """
    f = _positive(rf_hz)
    if reference_hz is None:
        reference_hz = float(np.sqrt(float(f.min()) * float(f.max())))
    return ((f / float(reference_hz)) * np.exp(1j * np.pi / 2.0)
            ).astype(np.complex128)


def playback_head_losses(rf_hz, mechanics: Optional[Dict[str, float]] = None,
                         minimum_phase: bool = True, **parameters
                         ) -> np.ndarray:
    """The reproduce head's wavelength losses, as a complex response.

    `head_model.log_response` is the model and is not duplicated here; what
    this adds is the conversion to the module's convention - a complex
    response rather than a log magnitude - and the removal of the
    differentiation term, which `log_response` includes as `log f` and which
    is entered separately above because it is the playback stage's only
    one-sided mechanism.

    The losses are MINIMUM PHASE, so their phase is the Hilbert transform of
    their log magnitude and carries no direction of its own. That is not a
    modelling convenience but a measured identity: the Hilbert transform is
    unitary on mean-removed signals, so stacking magnitude and phase doubles
    the Gram matrix and leaves the participation ratio where it was - 5.899435
    against 5.899485 on a six-row ensemble. Phase earns a direction only
    where a mechanism is NOT minimum phase, which in this chain means a
    transport delay, and a record delay and a playback delay are themselves
    only ever observed as their sum.
    """
    mechanics = mechanics or mechanics_at_speed("SP")
    speed = head_model.writing_speed(mechanics)
    width = mechanics["track_width_m"]
    used = dict(head_model.TYPICAL)
    used.update(parameters)
    logged = head_model.log_response(_positive(rf_hz), speed, width, **used)
    # the differentiation is its own entry; what remains is the losses
    logged = np.nan_to_num(logged - np.log(_positive(rf_hz)))
    phase = minimum_phase_of(logged) if minimum_phase else np.zeros_like(logged)
    return (np.exp(logged) * np.exp(1j * phase)).astype(np.complex128)


def playback_preamplifier(rf_hz, low_corner_hz: float = 0.3e6,
                          high_corner_hz: float = 7.0e6,
                          floor: float = STOP_BAND_FLOOR) -> np.ndarray:
    """The playback preamplifier, as a band pass with per-device corners.

    The same form and the same status as `record_amplifier`: the standard
    does not print it, the FORM is what a head preamplifier is, and the
    corners are per-device. It is entered separately because the two machines
    have two of them - and then measured to be the same direction, which is
    the point.
    """
    return record_amplifier(rf_hz, low_corner_hz, high_corner_hz, floor)


def playback_equalisation(rf_hz, corner_hz: float = 1.2e6,
                          floor: float = STOP_BAND_FLOOR) -> np.ndarray:
    """The playback equaliser, as a zero inside the RF band.

    The playback chain has to undo a head whose response falls at the top of
    the band and rises at the bottom, so its equaliser puts a zero INSIDE the
    band rather than a corner outside it. That is what makes it the one
    electronic entry here that is not collinear with the rest: its coherence
    never exceeds 0.65 against anything else, where the two amplifier band
    limits read 0.9994 against each other.

    `corner_hz` is a per-device parameter; the default is an example on the
    scale of the VHS luma band's lower half, not a constant of the format.
    """
    f = _grid(rf_hz)
    w = 1j * f / float(corner_hz)
    return _floored(w / (1.0 + w), floor)


def de_emphasis(baseband_hz,
                time_constant_s: float = MAIN_EMPHASIS_TIME_CONSTANT_S,
                gain_factor: float = MAIN_EMPHASIS_GAIN_FACTOR) -> np.ndarray:
    """The playback de-emphasis: the record pre-emphasis, exactly inverted.

    The format specifies it that way - the same network read the other way -
    and that specification is a NULL SPACE, not a convenience. In the log
    domain the two are exact negatives, so they are one direction and not
    two, and no measurement of a fused response can say which machine a
    departure in the emphasis belongs to. `cascade_control` measures the
    identity at 2.2e-16 nepers and fails if it is ever broken.
    """
    return 1.0 / main_pre_emphasis(baseband_hz, time_constant_s, gain_factor)


def minimum_phase_of(log_magnitude) -> np.ndarray:
    """The phase a minimum-phase response of this magnitude must have.

    The Hilbert transform of the log magnitude, the same relation
    `head_model.group_delay_s` uses to turn its own model's magnitude into a
    delay. Kept here so that every loss-type entry in this module carries the
    phase physics gives it rather than a zero that would make it look like a
    mechanism it is not.
    """
    from scipy.signal import hilbert

    values = np.asarray(log_magnitude, dtype=np.float64).ravel()
    finite = np.isfinite(values)
    phase = np.zeros_like(values)
    if int(finite.sum()) > 3:
        phase[finite] = -np.imag(hilbert(values[finite]))
    return phase


# --------------------------------------------------------------------------
# The two stages as ordered, subtractable key entries
# --------------------------------------------------------------------------

# WHICH GRID EACH ENTRY LIVES ON. An inner product across two abscissae is
# memory layout and not coherence, which is one of the recorded measurement
# traps, and this component is where the trap bites: the emphases act on the
# MODULATING VIDEO, ahead of the modulator and behind the demodulator, while
# the amplifiers and the heads act on the RF. Nothing here ever crosses them.
ABSCISSA: Dict[str, str] = {
    "record main pre-emphasis": "baseband",
    "sub emphasis level dependence": "baseband",
    "record non-linear emphasis level dependence": "baseband",
    "playback de-emphasis": "baseband",
    "record amplifier": "rf",
    "record head write response": "rf",
    "record level dependence": "rf",
    "playback head differentiation": "rf",
    "playback head losses": "rf",
    "playback preamplifier": "rf",
    "playback equalisation": "rf",
}

# WHERE EACH ENTRY SITS IN THE CHAIN, to be merged into
# `interference.COMPONENT_ORDER`. Position counts from the source, so a
# correction undoes them in DESCENDING order - `interference.ordered_key`.
#
# The blocks the existing map already fixes are the transmission path at 1-4,
# the recording machine at 10-11, the tape at 20-23 and the capture at 30.
# The recording machine's entries therefore go below 20 and the playback
# machine's between the tape and the capture, at 24-27, which is the position
# block nothing has claimed and the only one the chain allows: the playback
# machine acted after the medium and before the converter.
#
# THE ORDER WITHIN THE RECORDING MACHINE IS THE DECODER'S OWN, READ
# BACKWARDS. `filter_model.STAGE_ORDER` is main, non-linear, sub on playback,
# and a chain is undone last-applied-first, so the record machine applied the
# sub emphasis first, the non-linear stage next and the main emphasis last.
# The existing map's "sub emphasis level dependence" at 10 is therefore
# already in the right place and is reused rather than duplicated.
#
# ONE RECOMMENDATION FOR THE COORDINATOR, not a change made here: the
# existing "record level dependence" at 11 is the magnetic WRITE process -
# the record current setting the depth the tape holds - so it belongs after
# the record amplifier rather than before the main emphasis. Moving it from
# 11 to 16 would make the record block strictly ordered. It is left alone
# because this component does not own that file, and because the entry is a
# level rather than a filter, so its position only matters against the
# non-linear stage.
RF_STAGE_ORDER: Dict[str, int] = {
    # the recording machine, in the order it applied them
    "record non-linear emphasis level dependence": 12,
    "record main pre-emphasis": 13,
    "record amplifier": 14,
    "record head write response": 15,
    # the playback machine: after the tape at 20-23, before the capture at 30
    #
    # "playback head" is declared as a PREFIX as well as the two exact names,
    # because `interference.position_of` falls back to prefix matching and
    # the head is the one entry a caller may legitimately break into its
    # mechanisms - `magnetics_baseline` names its six rows "playback head
    # spacing_m" and so on, and all of them act at the same point and commute
    # exactly, being multiplicative factors of one transducer's transfer.
    "playback head": 24,
    "playback head differentiation": 24,
    "playback head losses": 24,
    "playback preamplifier": 25,
    "playback equalisation": 26,
    "playback de-emphasis": 27,
}


def record_signatures(baseband_hz, rf_hz,
                      mechanics: Optional[Dict[str, float]] = None,
                      speed: str = "EP", tape_format: str = "SVHS",
                      drives: Tuple[float, float] = (8000.0, 24000.0)
                      ) -> Dict[str, np.ndarray]:
    """Every modelled record-side modification, as complex responses.

    Up to three on the baseband abscissa and three on the RF one; `ABSCISSA`
    says which is which and nothing may be inner-producted across them.

    THE SPEED AND THE FORMAT DECIDE WHICH EMPHASES ARE PRESENT AT ALL, and
    that is the whole confound in the speed lever rather than a detail of the
    interface. `EXTRA_EMPHASIS_BY_SPEED` carries the repository's own table:
    VHS runs the extra record-side emphasis at LP, EP and SLP and not at SP,
    and S-VHS runs it always. The S-VHS SUB emphasis is included on top of
    that only for S-VHS, because IEC 60774-3 table 3 is the only place any
    standard prints the shape.

    `drives` are the two levels the non-linear emphasis is differenced
    between, in the decoder's own units for its high-pass limits. They are an
    EXAMPLE spanning the limit, not a constant: the point of the entry is
    that the limit binds at a different frequency for each, so a caller with
    a measured drive range should supply it.
    """
    mechanics = mechanics or mechanics_at_speed("SP")
    writing = head_model.writing_speed(mechanics)
    fmt = str(tape_format).upper()
    if fmt not in EXTRA_EMPHASIS_BY_SPEED:
        raise ValueError("no emphasis speed table for format %r" % tape_format)
    extra = EXTRA_EMPHASIS_BY_SPEED[fmt].get(str(speed).upper(), False)
    out = {
        "record main pre-emphasis": main_pre_emphasis(baseband_hz),
        "record amplifier": record_amplifier(rf_hz),
        "record head write response": record_head_write(
            rf_hz, writing_speed_m_s=writing),
        "record level dependence": record_current_level_dependence(
            rf_hz, 1.0, writing),
    }
    if extra:
        out["record non-linear emphasis level dependence"] = \
            nonlinear_emphasis_level_dependence(baseband_hz, *drives)
    if extra and fmt == "SVHS":
        out["sub emphasis level dependence"] = \
            sub_pre_emphasis_level_dependence(baseband_hz, speed)
    return out


def playback_signatures(rf_hz, baseband_hz,
                        mechanics: Optional[Dict[str, float]] = None,
                        **head_parameters) -> Dict[str, np.ndarray]:
    """Every modelled playback-side modification, as complex responses.

    One on the baseband abscissa - the de-emphasis - and four on the RF one.
    The head is entered as two members because they are two different kinds
    of thing: the differentiation is fixed by physics and has no free
    parameter and no record-side counterpart, while the losses are the
    six-mechanism family that already collapsed to 1.58 of 6.
    """
    mechanics = mechanics or mechanics_at_speed("SP")
    return {
        "playback head differentiation": playback_head_differentiation(rf_hz),
        "playback head losses": playback_head_losses(rf_hz, mechanics,
                                                     **head_parameters),
        "playback preamplifier": playback_preamplifier(rf_hz),
        "playback equalisation": playback_equalisation(rf_hz),
        "playback de-emphasis": de_emphasis(baseband_hz),
    }


def stage_signatures(baseband_hz, rf_hz, **kwargs) -> Dict[str, np.ndarray]:
    """Both stages on their own grids, as one key.

    Every value is a complex RESPONSE whose magnitude is bounded away from
    zero, so `interference.subtractable` holds on all of them, and every name
    has a position in `RF_STAGE_ORDER`, so `interference.ordered_key` can
    invert them once the coordinator has merged that map.
    """
    speed = kwargs.pop("speed", "EP")
    tape_format = kwargs.pop("tape_format", "SVHS")
    mechanics = kwargs.pop("mechanics", None)
    drives = kwargs.pop("drives", (8000.0, 24000.0))
    out = record_signatures(baseband_hz, rf_hz, mechanics, speed, tape_format,
                            drives)
    out.update(playback_signatures(rf_hz, baseband_hz, mechanics, **kwargs))
    return out


# --------------------------------------------------------------------------
# What can never be attributed to one machine rather than the other
# --------------------------------------------------------------------------


def _shape(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values - values.mean()
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-30 else values


def _coherence(first, second) -> float:
    return float(abs(_shape(first) @ _shape(second)))


def null_space(rf_hz, baseband_hz,
               mechanics: Optional[Dict[str, float]] = None
               ) -> Dict[str, object]:
    """WHAT IS IN EACH STAGE'S NULL SPACE, measured rather than asserted.

    The established precedent is the carrier-normalization law, which
    annihilates a constant and a linear term exactly, so `head_model
    .fit_head_difference` refuses `gain_db` and `spacing_m` outright and
    names them in `CARRIER_LAW_NULL_SPACE`. This is the equivalent for the
    two RF stages, and there are four entries rather than two.

    Three of the four are DEMONSTRATED rather than asserted, by building two
    different splits of the same total between the two machines and showing
    that the composed response is identical to machine precision. A null
    space is exactly the statement that the split cannot be recovered from
    the composition, so that is the check that belongs here rather than a
    coherence of something with itself:

      the emphasis pair       the format specifies de-emphasis as the inverse
                              of pre-emphasis, so their log magnitudes are
                              exact negatives - one direction, not two
      the wavelength losses   record transition length and playback spacing
                              compose in the exponent, so only their SUM is
                              identifiable
      any flat gain           record current, record amplifier gain, head
                              output, preamplifier gain and the capture's own
                              vertical scale are one constant between them
      any pure delay          a record delay and a playback delay appear only
                              as their sum

    A fifth is not exact but is measured to three decimals: the two
    amplifiers' upper band limits, because a single pole far above the band
    has a log-magnitude derivative proportional to `f^2` whatever its corner.
    """
    mechanics = mechanics or mechanics_at_speed("SP")
    writing = head_model.writing_speed(mechanics)
    f = _positive(rf_hz)

    pre = np.log(np.abs(main_pre_emphasis(baseband_hz)))
    post = np.log(np.abs(de_emphasis(baseband_hz)))
    transition = np.log(np.abs(record_head_write(
        f, 0.10e-6, writing_speed_m_s=writing)))
    spacing = -2.0 * np.pi * 0.05e-6 * f / writing

    def corner_shape(builder, corner, relative=BASELINE_PERTURBATION):
        below = np.log(np.abs(builder(f, high_corner_hz=corner)))
        above = np.log(np.abs(builder(f, high_corner_hz=corner
                                      * (1.0 + relative))))
        return np.nan_to_num(above - below)

    band_record = corner_shape(record_amplifier, 8.0e6)
    band_playback = corner_shape(playback_preamplifier, 7.0e6)

    def split_is_invisible(compose, first, second):
        """Two different splits of the same total, composed."""
        one = np.log(np.abs(compose(*first)))
        two = np.log(np.abs(compose(*second)))
        return float(np.max(np.abs(one - two)))

    def by_length(record_m, playback_m):
        return (record_head_write(f, record_m, writing)
                * np.exp(-2.0 * np.pi * playback_m / (writing / f)))

    def by_gain(record_db, playback_db):
        return np.exp((record_db + playback_db) / DB_PER_NEPER
                      ) * np.ones_like(f)

    def by_delay(record_s, playback_s):
        return np.exp(-2j * np.pi * f * (record_s + playback_s))

    return {
        "emphasis_pair_residual_nepers":
            float(np.max(np.abs(pre + post))),
        "emphasis_pair_coherence": _coherence(pre, post),
        "wavelength_loss_coherence": _coherence(transition, spacing),
        "wavelength_split_residual_nepers": split_is_invisible(
            by_length, (0.10e-6, 0.05e-6), (0.02e-6, 0.13e-6)),
        "flat_gain_split_residual_nepers": split_is_invisible(
            by_gain, (2.0, -0.5), (-3.0, 4.5)),
        "pure_delay_split_residual_radians": float(np.max(np.abs(
            np.angle(by_delay(20e-9, 5e-9) * np.conj(by_delay(1e-9, 24e-9)))))),
        "band_limit_coherence": _coherence(band_record, band_playback),
        "refused_for_the_record_stage": (
            "the emphasis shape, any flat gain, any pure delay, and the "
            "wavelength-loss exponent, all of which the playback stage can "
            "carry equally well"),
        "refused_for_the_playback_stage": (
            "the same four; the head's differentiation is the only mechanism "
            "either stage owns outright, and only because the record head "
            "writes a current rather than a derivative"),
        "why": ("a null space is not a poor fit but an unconstrained one: a "
                "parameter in a null direction takes whatever value the "
                "noise asks for, which is why fit_head_difference refuses "
                "gain_db and spacing_m rather than fitting them badly"),
    }


# --------------------------------------------------------------------------
# What data would separate them
# --------------------------------------------------------------------------


def separation_design(recordings: Sequence[Tuple[str, str]],
                      playbacks: Sequence[Tuple[int, str]],
                      record_taps: Sequence[str] = (),
                      known_tapes: Sequence[str] = ()) -> Dict[str, object]:
    """WHAT AN EXPERIMENT CAN IDENTIFY, from the estimable contrasts of its
    design.

    A playback of tape `t` on deck `m`, where tape `t` was recorded on deck
    `d(t)`, has a log response

        y = R[d(t)] + T[t] + P[m]

    which is an ordinary two-way additive model. RANK ALONE IS THE WRONG
    STATISTIC HERE - a two-way model is always deficient by the levels it
    cannot place, and reading that deficiency as a failure would condemn
    every design including the ones that work. What is asked instead is
    whether a NAMED CONTRAST lies in the row space, which is estimability,
    and that is what this returns.

    `recordings` lists the tapes as `(tape, recording deck)`, `playbacks`
    lists the observations as `(row into recordings, playback deck)`,
    `record_taps` names the decks whose record-side RF is also observed
    directly - an extra row that sees `R[d]` and nothing else - and
    `known_tapes` names the tapes whose own response is known independently,
    which is what a reference recording buys.

    The five candidate separators, run through it:

      one deck, one tape          nothing. `R + T + P` is one sum
      a second TAPE, same deck    the tapes' difference `T1 - T2`, and no
                                  more; the machines never move
      a second DECK playing the   `P[A] - P[B]`, the playback machines'
      same tape                   difference - so the playback stage
                                  separates BETWEEN decks
      the same source recorded    `R[A] - R[B]` once the tape stock is
      on two decks                shared, so the record stage separates
                                  between decks
      an RF tap at the record     `R[A]` itself, up to the grand level -
                                  and then `T + P` by subtraction, but NOT
                                  `T` from `P`
      a known reference on tape   `T` alone, and on its own that still does
                                  not split `R` from `P`

    AND THE ONE THAT NEVER BECOMES ESTIMABLE ON A SINGLE DECK IS `R[A] -
    P[A]` - not with more tapes, not with more decks, not with a record tap
    on its own, and not with a reference on its own. The recording machine's
    contribution and the playback machine's occur in every playback together.
    That is the formal statement of this component's central answer, and it
    is why the honest answer to "are the two separable from one playback of
    one tape" is no.

    IT TAKES TWO OF THEM TOGETHER. A record tap AND a known reference on the
    tape does it: the tap gives `R`, the reference gives `T`, and `P` is what
    is left of the playback. That is the only pairing among the five that
    reaches `R - P` on a single deck, and the test directory holds one half
    of it already.
    """
    tapes = [str(row[0]) for row in recordings]
    record_decks = [str(row[1]) for row in recordings]
    tape_names = list(dict.fromkeys(tapes))
    tape_index = {name: i for i, name in enumerate(tape_names)}
    deck_names = list(dict.fromkeys(record_decks
                                    + [str(p[1]) for p in playbacks]
                                    + [str(d) for d in record_taps]))
    deck_index = {name: i for i, name in enumerate(deck_names)}
    decks, kinds = len(deck_index), len(tape_index)
    columns = decks * 2 + kinds          # R per deck, P per deck, T per tape

    def blank():
        return np.zeros(columns)

    design = []
    for row, deck in playbacks:
        line = blank()
        line[deck_index[record_decks[int(row)]]] = 1.0                  # R
        line[decks + deck_index[str(deck)]] = 1.0                       # P
        line[2 * decks + tape_index[tapes[int(row)]]] = 1.0             # T
        design.append(line)
    for deck in record_taps:
        line = blank()
        line[deck_index[str(deck)]] = 1.0        # the tap sees R alone
        design.append(line)
    for tape in known_tapes:
        if str(tape) not in tape_index:
            continue
        line = blank()
        line[2 * decks + tape_index[str(tape)]] = 1.0   # the reference
        design.append(line)
    matrix = np.array(design) if design else np.zeros((0, columns))
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0

    def estimable(contrast: np.ndarray) -> bool:
        """A contrast is estimable when it lies in the design's row space."""
        if not matrix.size or not np.any(contrast):
            return False
        residual = contrast - matrix.T @ np.linalg.lstsq(
            matrix.T, contrast, rcond=None)[0]
        return bool(np.linalg.norm(residual)
                    <= 1e-9 * max(np.linalg.norm(contrast), 1e-30))

    def slot(block: str, name: str) -> int:
        if block == "R":
            return deck_index[name]
        if block == "P":
            return decks + deck_index[name]
        return 2 * decks + tape_index[name]

    contrasts: Dict[str, bool] = {}
    if decks > 1:
        first, second = deck_names[0], deck_names[1]
        for block, label in (("R", "record machines differ"),
                             ("P", "playback machines differ")):
            c = blank()
            c[slot(block, first)] = 1.0
            c[slot(block, second)] = -1.0
            contrasts[label] = estimable(c)
    if kinds > 1:
        c = blank()
        c[slot("T", tape_names[0])] = 1.0
        c[slot("T", tape_names[1])] = -1.0
        contrasts["tapes differ"] = estimable(c)
    # the one that matters: on one named deck, is the record path separable
    # from the playback path?
    here = deck_names[0]
    c = blank()
    c[slot("R", here)] = 1.0
    c[slot("P", here)] = -1.0
    contrasts["record against playback, one deck"] = estimable(c)
    # and is the record stage reachable at all, up to the grand level?
    c = blank()
    c[slot("R", here)] = 1.0
    contrasts["the record stage alone"] = estimable(c)

    return {
        "observations": int(matrix.shape[0]),
        "unknowns": columns,
        "rank": rank,
        "deficiency": columns - rank,
        "decks": decks,
        "tapes": kinds,
        "estimable": contrasts,
        "record_and_playback_separate": bool(
            contrasts.get("record against playback, one deck", False)),
        "why": ("R and P occur in every playback together, so on one deck "
                "they are one quantity however many tapes are used; a second "
                "deck makes them differ across observations, and only a "
                "record tap reaches R on its own"),
    }


# The measured record/playback tap comparison, from the eight zaroff patterns
# in /testdata/test_patterns/vhs/ at 50 MS/s, Welch periodogram of 4096
# points over 0.4-9.0 MHz, log magnitude in nepers with the mean removed
# because the scope's per-capture vertical scale destroys an overall gain
# before any modelling question is asked (the directory's readme says the
# vertical scale was adjusted per capture for full screen, and the coupling
# was AC).
TAP_MEASUREMENT: Dict[str, float] = {
    "record_tap_rms_nepers_sp": 1.6748,
    "record_tap_rms_nepers_ep": 1.6811,
    "playback_tap_rms_nepers_sp": 1.4383,
    "playback_tap_rms_nepers_ep": 1.1877,
    "round_trip_rms_nepers_sp": 0.4644,
    "round_trip_rms_nepers_ep": 0.7444,
    "record_tap_repeatable_sp": 0.910,
    "record_tap_repeatable_ep": 0.925,
    "round_trip_repeatable_sp": 0.798,
    "round_trip_repeatable_ep": 0.835,
    "playback_over_record_coherence_sp": 0.9724,
    "playback_over_record_coherence_ep": 0.9402,
    "playback_on_record_regression_sp": 0.835,
    "playback_on_record_regression_ep": 0.652,
    "speed_lever_record_tap_rms_nepers": 0.1618,
    "speed_lever_playback_tap_rms_nepers": 0.2978,
    "speed_lever_repeatable_record_tap": 0.268,
    "speed_lever_repeatable_playback_tap": 0.816,
    "patterns": 8.0,
}


def tap_separation(record_log=None, playback_log=None) -> Dict[str, object]:
    """WHAT THE RECORD TAP BUYS, and where it cuts the chain.

    The test directory holds both taps of every pattern from one Sony
    SLV-778HF, which is the third of the four separators - an RF tap at the
    record side. It works, and it cuts at the RECORD HEAD'S TERMINALS: the
    tap is the drive to the head, so everything before it is observed
    directly and everything after it stays fused.

    Measured, and the numbers are in `TAP_MEASUREMENT`:

      the RATIO of the two taps is a response, because the same content
      passed through both: subtracting the record tap reduces the playback
      tap's log-magnitude shape by 3.31x at SP and 1.65x at EP, and what
      remains repeats 0.80 and 0.84 across eight different test patterns,
      which is what says it belongs to the chain rather than to the picture;

      the record tap's OWN spectrum is NOT a response. Its 1.67 nepers rms is
      the modulated carrier's own shape at least as much as the record
      chain's, and its 0.91 agreement across patterns says only that an FM
      spectrum is dominated by its carrier. Turning it into the record
      stage's response needs a reference - the synthesised FM spectrum of the
      pattern the generator sent - which is the second half of the pairing
      `separation_design` finds;

      and the regression of the playback tap on the record tap is 0.835 at SP
      and 0.652 at EP rather than unity, WHICH IS A WARNING AND NOT A
      RESPONSE. The playback tap carries an additive noise floor - particle
      noise and the preamplifier's own - that fills the FM spectrum's deep
      valleys, and a log-spectral ratio taken where the numerator is near the
      floor is biased toward zero. The ratio is a response only where the
      record spectrum stands well above that floor.

    Called with two arrays it recomputes the comparison on them; called with
    none it reports the measurement above.
    """
    if record_log is None or playback_log is None:
        result = dict(TAP_MEASUREMENT)
        result["what_it_separates"] = (
            "the record electronics from everything downstream of the record "
            "head's terminals")
        result["what_stays_fused"] = (
            "the record head, the tape, the playback head and the "
            "preamplifier - and on this dataset the two heads are the same "
            "physical part, one machine having done both")
        result["why"] = (
            "a tap splits the chain where it is taken, and this one is taken "
            "at the head drive, so it makes four links into two blocks "
            "rather than four")
        return result
    record = np.asarray(record_log, dtype=np.float64).ravel()
    playback = np.asarray(playback_log, dtype=np.float64).ravel()
    record = record - record.mean()
    playback = playback - playback.mean()
    trip = playback - record
    return {
        "record_rms_nepers": float(record.std()),
        "playback_rms_nepers": float(playback.std()),
        "round_trip_rms_nepers": float(trip.std()),
        "reduction": float(playback.std() / max(trip.std(), 1e-30)),
        "coherence": _coherence(record, playback),
        "regression": float((record @ playback) / max(record @ record, 1e-30)),
        "why": ("a regression below one is the playback noise floor filling "
                "the FM spectrum's valleys, not a response"),
    }


def speed_confound(rf_hz, system: str = "NTSC",
                   fast: str = "SP", slow: str = "EP",
                   measured_rms_nepers: Optional[float] = None
                   ) -> Dict[str, object]:
    """THE SPEED LEVER, AND HOW MUCH OF IT IS THE RECORD MACHINE'S PROCESSING.

    The record side varies with tape speed and the playback side does not, so
    SP against EP looks like a controlled experiment on the recording
    machine. It is confounded, and the confound is nearly total, for a reason
    the format's own mechanics give:

      A VHS SPEED SWITCH BARELY MOVES THE TAPE SIDE. The drum turns at one
      rate whatever the tape speed, so the writing speed changes only by the
      tape's own contribution - 5.800 to 5.822 m/s on NTSC, 0.384% - and what
      really changes is the track pitch, 58 to 19.3 um. Both are in the inert
      set: `docs/COMPONENT_MAPPINGS.md` section 3 measured writing speed as
      inert and slightly adverse over an eightfold range, and track width as
      inert to three decimals, because it enters `log_response` only as the
      product with the tangent of the azimuth.

      Measured through the head model, the whole tape-side change is 0.0104
      nepers rms of log response over 0.5-8 MHz, against 0.2978 nepers rms
      measured between SP and EP at the playback tap. THE TAPE ARM IS 3.5% OF
      THE LEVER AND THE RECORD MACHINE'S PROCESSING IS 96.5%.

      And the processing is exactly the stage one arm has and the other does
      not: `EXTRA_EMPHASIS_BY_SPEED` says the extra record-side emphasis runs
      at LP, EP and SLP and not at SP. So the lever moves one whole
      processing block and essentially nothing else, which means it can
      measure that block and can measure nothing else at all - in particular
      it cannot measure the record machine's MAGNETICS.

    WHAT BREAKS IT. Not LP against EP: that is worse, 0.0018 nepers modelled,
    because both arms carry the extra emphasis and the mechanics move less
    still. What breaks it is the RECORD TAP, which observes the processing
    arm with no tape in it at all - and the test directory has one. Failing
    that, an EP recording made with the extra emphasis defeated.

    THE TRAP IN USING THE TAP FOR THIS. The record tap's own SP-minus-EP
    difference repeats only 0.268 across eight patterns, against 0.816 for
    the playback tap's. Three quarters of what looks like a record-side speed
    effect is the picture, because it is a small difference between two large
    and nearly identical FM spectra. It needs many patterns averaged before
    it says anything.
    """
    quick = mechanics_at_speed(fast, system)
    slowly = mechanics_at_speed(slow, system)
    typical = dict(head_model.TYPICAL)
    typical.update(azimuth_error_degrees=0.1, contour_depth=0.05, gain_db=1.0)
    f = _positive(rf_hz)
    one = head_model.log_response(f, head_model.writing_speed(quick),
                                  quick["track_width_m"], **typical)
    two = head_model.log_response(f, head_model.writing_speed(slowly),
                                  slowly["track_width_m"], **typical)
    tape_arm = np.nan_to_num(one - two)
    tape_arm = tape_arm - tape_arm.mean()
    # the two halves of the tape arm, separately
    width_only = head_model.log_response(f, head_model.writing_speed(quick),
                                         slowly["track_width_m"], **typical)
    speed_only = head_model.log_response(f, head_model.writing_speed(slowly),
                                         quick["track_width_m"], **typical)
    width_arm = np.nan_to_num(one - width_only)
    speed_only_arm = np.nan_to_num(one - speed_only)
    measured = (TAP_MEASUREMENT["speed_lever_playback_tap_rms_nepers"]
                if measured_rms_nepers is None else float(measured_rms_nepers))
    share = float(tape_arm.std()) / max(measured, 1e-30)
    return {
        "writing_speed_fast_m_s": head_model.writing_speed(quick),
        "writing_speed_slow_m_s": head_model.writing_speed(slowly),
        "writing_speed_change_fraction": abs(
            head_model.writing_speed(quick) - head_model.writing_speed(slowly)
        ) / max(head_model.writing_speed(quick), 1e-30),
        "track_width_fast_m": quick["track_width_m"],
        "track_width_slow_m": slowly["track_width_m"],
        "tape_arm_rms_nepers": float(tape_arm.std()),
        "track_width_arm_rms_nepers": float((width_arm
                                             - width_arm.mean()).std()),
        "writing_speed_arm_rms_nepers": float((speed_only_arm
                                               - speed_only_arm.mean()).std()),
        "measured_lever_rms_nepers": measured,
        "tape_share_of_the_lever": share,
        "processing_share_of_the_lever": 1.0 - share,
        "extra_emphasis_fast": EXTRA_EMPHASIS_BY_SPEED["VHS"].get(
            str(fast).upper(), False),
        "extra_emphasis_slow": EXTRA_EMPHASIS_BY_SPEED["VHS"].get(
            str(slow).upper(), False),
        "confounded": bool(
            EXTRA_EMPHASIS_BY_SPEED["VHS"].get(str(fast).upper(), False)
            != EXTRA_EMPHASIS_BY_SPEED["VHS"].get(str(slow).upper(), False)),
        "what_breaks_it": (
            "the record RF tap, which sees the processing arm with no tape "
            "in it; or an EP recording made with the extra emphasis "
            "defeated. NOT LP against EP, which moves the mechanics less "
            "still and holds the emphasis switch in both arms"),
        "why": ("the drum turns at one rate whatever the tape speed, so a "
                "speed switch moves the track pitch and almost nothing "
                "else, and both of the things it moves are already measured "
                "as inert to identifiability"),
    }


# --------------------------------------------------------------------------
# The count, on the same measure the rest of the arc uses
# --------------------------------------------------------------------------


def effective_count(rows: Sequence[np.ndarray], demean: bool = False
                    ) -> Dict[str, object]:
    """The participation ratio of the singular values, as the arc measures it.

    `demean` is FALSE by default, because that is the convention the
    published 1.58 of 6 was measured with - the same rows demeaned read 2.07,
    and the two are not the same measure. `interference.distinguishable`
    removes the mean; the tape figure quoted beside it does not.
    """
    stack = np.array([np.asarray(r, dtype=np.float64).ravel() for r in rows])
    if demean:
        stack = stack - stack.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    stack = stack / np.where(norms > 0, norms, 1.0)
    singular = np.linalg.svd(stack, compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    return {
        "effective": float(1.0 / np.sum(share ** 2)),
        "count": int(stack.shape[0]),
        "singular_values": singular,
        "condition": float(singular[0] / max(singular[-1], 1e-30)),
        "unit_rows": stack,
    }


# The six mechanisms and the operating point that reproduce the published
# 1.58 of 6. The band and the perturbation are the ones the adversary sheet
# used, `tools/ringing_measure/adversary/c6b_typical.py`: `head_model
# .TYPICAL` with the three mechanisms it leaves at zero given small values,
# 0.5 to 8.0 MHz over 512 points, a 5% perturbation, and the mean NOT
# removed. Reproduced here at 1.576 with singular values 2.1592, 1.0000,
# 0.5657, 0.1327, 0.0088, 0.0001.
MAGNETICS_MECHANISMS = ("spacing_m", "gap_m", "thickness_m",
                        "azimuth_error_degrees", "contour_depth", "gain_db")
MAGNETICS_OPERATING_POINT = {"azimuth_error_degrees": 0.1,
                             "contour_depth": 0.05, "gain_db": 1.0}
BASELINE_BAND_HZ = (0.5e6, 8.0e6)
BASELINE_POINTS = 512
BASELINE_PERTURBATION = 0.05


def magnetics_baseline(rf_hz=None, mechanics: Optional[Dict[str, float]] = None
                       ) -> Dict[str, object]:
    """The 1.58 of 6 this component is measured against, recomputed.

    Quoted rather than recomputed it would be a number from another run on
    another band; recomputed here it is the same procedure applied to the
    same model, so a difference between it and the combined count is the
    stages' doing and not the measure's.
    """
    if rf_hz is None:
        rf_hz = np.linspace(BASELINE_BAND_HZ[0], BASELINE_BAND_HZ[1],
                            BASELINE_POINTS)
    mechanics = mechanics or mechanics_at_speed("SP")
    speed = head_model.writing_speed(mechanics)
    width = mechanics["track_width_m"]
    point = dict(head_model.TYPICAL)
    point.update(MAGNETICS_OPERATING_POINT)
    reference = head_model.log_response(_positive(rf_hz), speed, width, **point)
    rows, names = [], []
    for key in MAGNETICS_MECHANISMS:
        moved = dict(point)
        moved[key] = point[key] * (1.0 + BASELINE_PERTURBATION)
        rows.append(np.nan_to_num(
            head_model.log_response(_positive(rf_hz), speed, width, **moved)
            - reference))
        names.append("playback head %s" % key)
    result = effective_count(rows)
    result["names"] = names
    result["rows"] = rows
    result["why"] = ("every tape loss is a function of one dimensionless "
                     "group, a length over the recorded wavelength, so six "
                     "mechanisms are six values of one variable")
    return result


def _stage_rows(rf_hz, mechanics) -> Tuple[List[str], List[np.ndarray]]:
    """The RF-abscissa stage entries as parameter-derivative log shapes.

    A signature enters the count as the SHAPE ITS OWN PARAMETER MOVES THE
    RESPONSE IN, which is what the magnetics baseline measures and therefore
    the only thing comparable with it. The differentiation is the exception
    and has no free parameter - it is `f` exactly, fixed by Faraday - so its
    derivative with respect to the exponent is taken instead, which is
    `log f`.
    """
    writing = head_model.writing_speed(mechanics)
    f = _positive(rf_hz)
    step = 1.0 + BASELINE_PERTURBATION

    def moved(builder, value):
        below = np.log(np.abs(builder(value)))
        above = np.log(np.abs(builder(value * step)))
        return np.nan_to_num(above - below)

    names = ["record amplifier", "record head write response",
             "record level dependence", "playback head differentiation",
             "playback preamplifier", "playback equalisation"]
    rows = [
        moved(lambda v: record_amplifier(f, high_corner_hz=v), 8.0e6),
        moved(lambda v: record_head_write(f, v, writing), 0.10e-6),
        np.nan_to_num(np.log(
            np.abs(record_current_level_dependence(f, 1.0 * step, writing))
            / np.abs(record_current_level_dependence(f, 1.0, writing)))),
        np.log(f),
        moved(lambda v: playback_preamplifier(f, high_corner_hz=v), 7.0e6),
        moved(lambda v: playback_equalisation(f, v), 1.2e6),
    ]
    return names, rows


def distinguishable(rf_hz=None, mechanics: Optional[Dict[str, float]] = None
                    ) -> Dict[str, object]:
    """HOW MANY DIRECTIONS THE TWO STAGES ADD TO THE MAGNETICS FAMILY.

    THE RF ABSCISSA ONLY. The emphases live on the baseband one and are
    counted by `baseband_distinguishable`; pooling the two would be the inner
    product across two abscissae that the recorded traps forbid.

    The measure and the procedure are the baseline's, so the comparison is
    like for like. Measured on the VHS RF band, 0.5 to 8.0 MHz:

        magnetics alone                 1.576 of  6, condition 3.0e4
        with the record stage           1.676 of  9
        with the playback stage         1.743 of  9
        with both                       1.803 of 12, condition 6.8e13

    SIX MORE MEMBERS BUY 0.23 OF A DIRECTION. The plain reading is the right
    one: THE TWO STAGES ARE THE SAME COLLINEAR FAMILY SEEN TWICE. Every
    entry except two is another monotone function of one length over the
    recorded wavelength, which is the law `docs/COMPONENT_MAPPINGS.md`
    section 2 states - a family is collinear when its members differ only in
    a rate, and separable when they differ in kind.

    The two exceptions earn their places by the direction they add, which is
    the rule the sub emphasis and the record level both taught:

        record level dependence     coherence at most 0.42 with anything else
        playback equalisation       at most 0.65, because its corner is INSIDE
                                    the band where every other electronic
                                    corner is outside it

    And the condition number is the honest warning: 6.8e13 says the combined
    ensemble is rank deficient in practice, which it is - by the emphasis
    pair and the wavelength-loss sum, both of which `null_space` measures as
    exactly collinear.
    """
    if rf_hz is None:
        rf_hz = np.linspace(BASELINE_BAND_HZ[0], BASELINE_BAND_HZ[1],
                            BASELINE_POINTS)
    mechanics = mechanics or mechanics_at_speed("SP")
    baseline = magnetics_baseline(rf_hz, mechanics)
    names, rows = _stage_rows(rf_hz, mechanics)
    record = [i for i, n in enumerate(names) if n.startswith("record")]
    playback = [i for i, n in enumerate(names) if n.startswith("playback")]

    def with_extra(indices):
        return effective_count(list(baseline["rows"])
                               + [rows[i] for i in indices])

    both = with_extra(range(len(rows)))
    unit = both["unit_rows"]
    coherence = np.abs(unit @ unit.T)
    labels = list(baseline["names"]) + names
    return {
        "names": labels,
        "baseline": {"effective": baseline["effective"],
                     "count": baseline["count"],
                     "condition": baseline["condition"]},
        "with_record": {"effective": with_extra(record)["effective"],
                        "count": len(baseline["rows"]) + len(record)},
        "with_playback": {"effective": with_extra(playback)["effective"],
                          "count": len(baseline["rows"]) + len(playback)},
        "with_both": {"effective": both["effective"], "count": both["count"],
                      "condition": both["condition"]},
        "effective": both["effective"],
        "count": both["count"],
        "condition": both["condition"],
        "singular_values": both["singular_values"],
        "coherence": coherence,
        "gain_over_the_baseline": both["effective"] - baseline["effective"],
        "stages_alone": effective_count(rows)["effective"],
        "why": ("six more members buy a quarter of a direction, so the two "
                "stages are the same collinear family seen twice; only the "
                "record current's level dependence and the playback "
                "equaliser's in-band corner add anything"),
    }


def baseband_distinguishable(baseband_hz=None, speed: str = "EP"
                             ) -> Dict[str, object]:
    """The same count on the BASEBAND abscissa, where the emphases live.

    Reported separately and never pooled with the RF one, because an inner
    product across two abscissae is memory layout rather than coherence.

    Measured over 50 kHz to 3.0 MHz, which is the VHS luma baseband:

        four entries, 1.455 effective, condition 1.4e14

    and the condition number is the emphasis null arriving as arithmetic: the
    pre-emphasis and de-emphasis rows are exact negatives, so the matrix is
    rank deficient by one exactly. Their coherence is 1.0000.

    The second finding here is that the two LEVEL-DEPENDENT record-side
    emphases are nearly one direction as well - the S-VHS sub emphasis's
    level dependence against the non-linear stage's, coherence 0.9864. Both
    are a high-frequency boost whose amount falls as the drive rises, so the
    record machine's two non-linear stages are one axis and not two.
    """
    if baseband_hz is None:
        baseband_hz = np.linspace(50e3, 3.0e6, BASELINE_POINTS)
    step = 1.0 + BASELINE_PERTURBATION
    tau = MAIN_EMPHASIS_TIME_CONSTANT_S
    rows = [
        np.log(np.abs(main_pre_emphasis(baseband_hz, tau * step)))
        - np.log(np.abs(main_pre_emphasis(baseband_hz, tau))),
        np.log(np.abs(de_emphasis(baseband_hz, tau * step)))
        - np.log(np.abs(de_emphasis(baseband_hz, tau))),
        np.log(np.abs(sub_pre_emphasis_level_dependence(baseband_hz, speed))),
        np.log(np.abs(nonlinear_emphasis_level_dependence(
            baseband_hz, 8000.0, 24000.0))),
    ]
    names = ["record main pre-emphasis", "playback de-emphasis",
             "sub emphasis level dependence",
             "record non-linear emphasis level dependence"]
    result = effective_count(rows)
    unit = result.pop("unit_rows")
    result["names"] = names
    result["coherence"] = np.abs(unit @ unit.T)
    result["emphasis_pair_coherence"] = float(abs(unit[0] @ unit[1]))
    result["level_dependent_pair_coherence"] = float(abs(unit[2] @ unit[3]))
    result["why"] = ("the pre-emphasis and the de-emphasis are one direction "
                     "by specification, and the two level-dependent stages "
                     "are very nearly one more")
    return result


# --------------------------------------------------------------------------
# THE CONTROL
# --------------------------------------------------------------------------


def _reproduce_gap_loss(
        rf_hz, gap_m: float,
        writing_speed_m_s: float = DEFAULT_WRITING_SPEED_M_S) -> np.ndarray:
    """A reproduce head's gap loss, `sinc(g/lambda)`, bounded below.

    Used only by the control. `sinc` is chosen over an exponential precisely
    because its shape is NOT scale free: a wrong wavelength changes the shape
    and not merely the size, so the control can see it. An exponential loss
    could not be used here - its logarithm is linear in frequency whatever
    the constant, so every wrong constant gives the same direction and the
    control would pass on a broken construction.
    """
    wavelength = float(writing_speed_m_s) / _positive(rf_hz)
    ratio = head_model.EFFECTIVE_GAP_FACTOR * float(gap_m) / wavelength
    return np.maximum(np.abs(np.sinc(ratio)), 1e-6).astype(np.complex128)


def cascade_control(rf_hz=None, baseband_hz=None,
                    record=_reproduce_gap_loss,
                    playback=_reproduce_gap_loss,
                    pre_emphasis=main_pre_emphasis,
                    post_emphasis=de_emphasis,
                    gap_m: float = 0.30e-6) -> Dict[str, object]:
    """THE CONTROL, kept as code because a control that cannot fail is not one.

    An earlier component in this arc reported 4.65 effectively distinguishable
    of 5 and was wrong, and what exposed it was a linear control that should
    have read 1.00 and read 4.97. This is the equivalent for a component
    whose whole risk is the opposite direction - manufacturing two directions
    where the physics has one, and then reporting the two machines as
    separable because the construction split them.

    So the control asserts two things whose answers are known independently
    of anything measured here:

      ONE MECHANISM ENTERED ONCE ON EACH SIDE IS ONE DIRECTION. The same gap
      loss, at the same parameter, built through the record path and through
      the playback path, must span exactly one direction: the participation
      ratio must be 1.00. If the two paths differ in the wavelength they use,
      in the unit they return, or in the convention of what a signature is,
      it reads 2.00.

      THE SPECIFIED EMPHASIS ROUND TRIP IS UNITY EXACTLY. The format defines
      de-emphasis as the inverse of pre-emphasis, so `log|pre * de|` must be
      zero to machine precision. If either returns a logarithm where a
      response is wanted, or decibels where nepers are wanted, it does not.

    Measured, with the module's own builders and with three realistic
    substitutions in their place:

        correct                                    1.0000    2.2e-16 nepers
        one side's wavelength from the LINEAR tape
          speed instead of the writing speed       1.9996    2.2e-16
        de-emphasis returning a logarithm          1.0000    2.06 nepers
        de-emphasis returned in decibels           1.0000    1.41 nepers

    The first substitution is the mistake `head_model.writing_speed` exists
    to prevent, and the second is the convention mismatch that already exists
    in the tree, `interference.sub_emphasis` returning nepers where every
    other signature there returns a response. Both are caught, and by
    different halves of the control.
    """
    if rf_hz is None:
        rf_hz = np.linspace(BASELINE_BAND_HZ[0], BASELINE_BAND_HZ[1],
                            BASELINE_POINTS)
    if baseband_hz is None:
        baseband_hz = np.linspace(50e3, 3.0e6, BASELINE_POINTS)
    rows = []
    for builder in (record, playback):
        below = np.log(np.abs(builder(rf_hz, gap_m)))
        above = np.log(np.abs(builder(rf_hz, gap_m
                                      * (1.0 + BASELINE_PERTURBATION))))
        rows.append(np.nan_to_num(above - below))
    shape = effective_count(rows, demean=True)
    trip = float(np.max(np.abs(np.log(np.abs(
        np.asarray(pre_emphasis(baseband_hz))
        * np.asarray(post_emphasis(baseband_hz)))))))
    effective = float(shape["effective"])
    return {
        "effective": effective,
        "expected": 1.0,
        "round_trip_nepers": trip,
        "round_trip_expected": 0.0,
        "one_mechanism_is_one_direction": bool(abs(effective - 1.0) < 0.05),
        "emphasis_inverts_exactly": bool(trip < 1e-9),
        "passes": bool(abs(effective - 1.0) < 0.05 and trip < 1e-9),
        "why": ("one mechanism entered on both sides is one shape, and the "
                "specified emphasis round trip is unity; anything else means "
                "the construction split a direction that physics did not"),
    }
