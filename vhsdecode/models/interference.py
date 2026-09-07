"""The interference types, as complex components the transform can
differentiate over.

Ethan's reading, and it is the right one:

    randomness is a residual here that we can effectively differentiate over
    since we can model different types of radio interference, maybe that's
    all the RF stage boils down to using all of our measured components

    I have enough known types of interference measured that I now can derive
    exactly what types of interference map to all of my existing properties,
    since I know exactly how the vcr was built, and exactly how the test
    signals are

THE MEASUREMENT THAT MOTIVATES THIS FILE. Five disturbances were given a
RANDOM PHASE per field, so that none of them could correlate trivially, and
the ellipse was fitted across fields:

    particle noise      0 directions   asymmetry 0.0301 vs floor 0.0304   -0.1 sigma
    beat / co-channel   1 direction    1.0000 vs 0.0304                 +351   sigma
    head clog           1 direction    1.0000 vs 0.0304                 +351   sigma
    dropout             4 directions   0.3978 vs 0.0304                 +133   sigma
    modulation noise    4 directions   0.2773 vs 0.0304                  +89   sigma

Four of the five are not random to the transform at all. Only particle noise
sits at the sphere floor, and it sits there to within a tenth of a sigma.
A beat has a frequency even when its phase is random; a dropout has a shape
even when its position is not. So what a single measurement lumps together
as noise is mostly modellable interference, and the irreducible remainder is
one specific mechanism - the finite number of particles in the volume the
head reads.

WHY THE SIGNATURES ARE COMPLEX AND WHY THAT MATTERS. Each function returns a
complex signature over frequency: the real part is what the mechanism does to
amplitude and the imaginary part what it does to phase. Fitted with
`ellipsoid(..., real_parameters=True)`, because each of these is the effect
of a REAL physical parameter and a gain is therefore a different mechanism
from a delay of the same shape - which the Hermitian inner product cannot
see. Measured on the propagation set, that distinction moves the effective
count from 2.20 to 3.79.

WHY PROPAGATION SEPARATES AND TAPE DOES NOT. Every tape loss is `exp(-kf)` or
`sinc(kf)` - a family of monotone decays that is nearly one-dimensional over
a finite band, which is why six tape mechanisms collapse to 1.58. An echo
family is `exp(-2*pi*j*f*tau)`, a Fourier basis over delay, orthogonal by
construction: eight echoes spaced at or above `1/B` measure 7.92 of 8
distinguishable. The tape's mechanisms are parameterised along the axis that
makes them alike; multipath's along the axis that makes them independent.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from vhsdecode.models import magnetic

# Where a figure below is a standard's, it is named. Where it is an example
# for a worked signature it is a DEFAULT the caller is expected to replace
# with the measurement, never a constant of the format.

# ITU-R BT.1700 / SMPTE 170M: the NTSC intercarrier sound is 4.5 MHz above
# the vision carrier, so the trap that removes it sits there.
INTERCARRIER_SOUND_HZ = 4.5e6

# ATSC A/49 states the ghost-cancelling reference's window as -3 to +45 us,
# which is the range of echo delays a receiver is expected to correct.
GHOST_WINDOW_S = (-3e-6, 45e-6)


def _grid(frequency_hz) -> np.ndarray:
    return np.asarray(frequency_hz, dtype=np.float64).ravel()


# --------------------------------------------------------------------------
# The tape's own, which is where the floor is
# --------------------------------------------------------------------------


def particle_noise(frequency_hz, writing_speed_m_s: float,
                   track_width_m: float,
                   recording_depth_m: Optional[float] = None,
                   particle_volume_m3: float = 1e-21,
                   packing: float = 0.4) -> np.ndarray:
    """The one mechanism that is genuinely random.

    A head reads a finite volume of coating holding a finite number of
    particles, each with a randomly oriented moment. The signal adds
    coherently and the noise as a square root, and the volume shrinks with
    the wavelength, so the noise RISES with frequency where a converter's
    does not.

    THE DEPTH IS NOT A CONSTANT. JVC VTG82063 section 7.2: "the optimum
    playback sensitivity is obtained when recording is performed at a depth
    equivalent to 1/4th the recording wavelength", and the guide works it
    through to 0.26-0.36 um across the video carrier. So `d = lambda/4`,
    the read volume goes as lambda SQUARED rather than lambda, and the
    particle noise rises with frequency TWICE as fast as a fixed depth
    implies - measured, -20 dB per decade against -10. Passing an explicit
    `recording_depth_m` overrides the law with a constant, which is what a
    flat-field measurement of a particular stock would supply.

    Returned as a real amplitude profile because it has no phase to have:
    it is a power spectrum, not a transfer function. This is the component
    that the transform will always leave in the eigenvalue bulk, and the
    thing every other entry in this module is measured against.
    """
    f = _grid(frequency_hz)
    wavelength = writing_speed_m_s / np.maximum(f, 1.0)
    depth = (wavelength / 4.0 if recording_depth_m is None
             else np.full_like(wavelength, float(recording_depth_m)))
    volume = np.maximum(wavelength, 0.0) * track_width_m * depth
    particles = np.maximum(volume * packing / particle_volume_m3, 1.0)
    return (1.0 / np.sqrt(particles)).astype(np.complex128)


def modulation_noise(frequency_hz, carrier_hz: float,
                     width_hz: float, amount: float = 0.5) -> np.ndarray:
    """The coating's own variation, which SCALES WITH THE SIGNAL.

    Unlike particle noise it is not additive: it rides the carrier as
    sidebands, so it appears where the carrier is and nowhere else. A simple
    particle count omits it entirely, which is part of why the measured
    excess over the particle prediction is 7.7 dB rather than zero.
    """
    f = _grid(frequency_hz)
    shape = np.exp(-((f - carrier_hz) / max(width_hz, 1.0)) ** 2)
    # SUBTRACTABLE FORM. Modulation noise ADDS IN POWER; entered as a bare
    # Gaussian its magnitude reaches zero away from the carrier, its
    # logarithm is minus infinity there, and the fit sees a clamped constant
    # instead of a shape. As a power contribution on the carrier it is
    # `1 + a*shape`, whose logarithm is finite everywhere and whose effects
    # compose by addition - which is what the transform requires.
    return (1.0 + amount * shape).astype(np.complex128)


def dropout(frequency_hz, span_hz: float, centre_hz: float,
            depth: float = 0.9) -> np.ndarray:
    """Oxide shed and head clog: a loss over a band, impulsive in time.

    Broad in frequency and short in time, which is the opposite of a beat
    and the reason the two never share a direction.
    """
    f = _grid(frequency_hz)
    inside = np.abs(f - centre_hz) <= 0.5 * span_hz
    # SUBTRACTABLE FORM: a bounded attenuation, not a zero-one mask. A mask
    # has no logarithm where it is zero, so it cannot be subtracted off; a
    # dropout is in any case a loss of some depth, never a total one.
    return (1.0 - depth * inside).astype(np.complex128)


def head_contact_tilt(frequency_hz, low_hz: float,
                      high_hz: float) -> np.ndarray:
    """A slow tilt across the band from head-to-tape separation changing.

    One direction, and a strong one: it is the same shape every time and
    only its size varies, so it stands far above the sphere floor.
    """
    f = _grid(frequency_hz)
    span = max(high_hz - low_hz, 1.0)
    return (2.0 * (f - low_hz) / span - 1.0).astype(np.complex128)


# --------------------------------------------------------------------------
# Radio-borne, and the reason propagation separates where tape does not
# --------------------------------------------------------------------------


def echo(frequency_hz, delay_s: float, amplitude: float = 0.1) -> np.ndarray:
    """Multipath: a reflection at one delay.

    `1 + a exp(-2 pi j f tau)` - a ripple of period `1/tau` in the response,
    in amplitude AND phase together. This is the well-conditioned family:
    delays separated by more than `1/B` give near-orthogonal signatures, and
    the result is scale free, identical in units of `1/B` at any bandwidth.

    A PRE-echo, `tau < 0`, is legitimate here and only here. The ringing
    arc's standing rule forbids pre-echo absolutely, and that is right for
    the tape - a physical channel cannot respond before its stimulus. It is
    wrong for a transmission component, where ATSC A/49 admits echoes from
    -3 us, and where a non-causal echo is EVIDENCE OF TRANSMISSION rather
    than grounds for rejection. The rule needs a stage qualifier; this
    function does not enforce one, and the caller must not apply a
    transmission echo to a tape stage.
    """
    f = _grid(frequency_hz)
    return 1.0 + amplitude * np.exp(-2j * np.pi * f * delay_s)


def beat(frequency_hz, beat_hz: float, width_hz: float = 5e3,
         amount: float = 1.0) -> np.ndarray:
    """Co-channel or a converter birdie: a LINE in the spectrum.

    Narrow where every tape mechanism is broad, so it is the easiest thing
    in this module to separate and the first to show as its own direction.
    """
    f = _grid(frequency_hz)
    shape = np.exp(-((f - beat_hz) / max(width_hz, 1.0)) ** 2)
    # SUBTRACTABLE FORM, as for the modulation noise: an interfering carrier
    # adds in power, so it enters as `1 + a*shape` rather than as a bare
    # line that is zero everywhere else.
    return (1.0 + amount * shape).astype(np.complex128)


def vestigial_sideband(frequency_hz, carrier_hz: float,
                       vestige_hz: float, floor: float = 1e-3) -> np.ndarray:
    """The VSB Nyquist slope about the vision carrier.

    The transmitter suppresses most of one sideband and shapes what is left
    so that `R(f) + R(-f) = 1` across the carrier. That identity is why a
    SYNCHRONOUS detector recovers the video by a LINEAR operation - and
    therefore why a video-domain filter is the correct instrument for a
    transmission defect, where it is forbidden for a tape defect. The "no
    video-domain inverse" law's premise is a nonlinear demodulator, and the
    transmission chain does not have one.
    """
    f = _grid(frequency_hz)
    offset = (f - carrier_hz) / max(vestige_hz, 1.0)
    # SUBTRACTABLE FORM: the Nyquist slope bounded away from zero. Clipped
    # to zero at the stop-band edge it has no logarithm there, and a real
    # filter has a finite stop-band rejection rather than an infinite one.
    return np.clip(0.5 * (1.0 - offset), floor, 1.0).astype(np.complex128)


def group_delay(frequency_hz, seconds_per_hz: float) -> np.ndarray:
    """An IF or SAW filter's group delay: PHASE ONLY, no amplitude.

    The component that makes `real_parameters` matter. Its magnitude
    profile can match a gain change exactly while being a different
    mechanism, and only the stacked form tells them apart.
    """
    f = _grid(frequency_hz)
    return np.exp(-2j * np.pi * f * (seconds_per_hz * f))


def sound_trap(frequency_hz, notch_hz: float = INTERCARRIER_SOUND_HZ,
               width_hz: float = 200e3, depth: float = 0.9) -> np.ndarray:
    """The intercarrier sound trap, and its skirt reaching into the video.

    Its NOTCH is unreachable through a colour-under recording - VHS's luma
    baseband stops at 3 MHz, well below 4.5 MHz - so a trap seen in a tape
    decode was applied before recording. That makes it one of the
    landmarks that separate the transmission chain from the tape.
    """
    f = _grid(frequency_hz)
    shape = 1.0 / (1.0 + ((f - notch_hz) / max(width_hz, 1.0)) ** 2)
    return (1.0 - depth * shape).astype(np.complex128)


# --------------------------------------------------------------------------
# SPECIFIED MODIFICATIONS. Not interference - things the format's own
# standard says the recorder DOES to the signal. Ethan: "We have the key
# which is all possible modifications to our video signal, that's the key."
# A modification whose shape the standard prints is the strongest kind of
# key entry, because its synthetic side needs no fitting at all.
# --------------------------------------------------------------------------

# IEC 60774-3:1993 table 3, the S-VHS sub pre-emphasis: a LEVEL-DEPENDENT
# and SPEED-DEPENDENT record-side characteristic, in dB, tabulated against
# input level (0 dB = 0.4 Vpp) and frequency, with tolerances the standard
# states alongside. The main pre-emphasis is unchanged from VHS and is
# referenced rather than restated, so this table is the whole of what S-VHS
# added. Transcribed from the standard; the tolerances are dropped here and
# kept in the docstring, because a signature needs the shape and a fit needs
# the tolerance separately.
SUB_EMPHASIS_HZ = (200e3, 500e3, 1e6, 2e6, 3e6, 5e6)
SUB_EMPHASIS_INPUT_DB = (0.0, -10.0, -20.0, -30.0)
SUB_EMPHASIS_DB = {
    "SP": (
        (-1.73, -1.60, -1.04, -0.37, -0.07, -0.06),
        (-1.30, -0.73, +0.69, +1.75, +2.10, +2.02),
        (-0.65, +1.09, +2.86, +4.16, +4.60, +4.43),
        (-0.49, +2.35, +5.30, +7.14, +7.64, +7.34),
    ),
    "EP": (
        (-1.86, -0.76, +0.81, +1.60, +1.72, +1.54),
        (-1.46, +0.11, +2.50, +3.73, +3.89, +3.59),
        (-0.82, +1.93, +4.70, +6.14, +6.36, +5.97),
        (-0.63, +3.21, +7.14, +9.15, +9.43, +8.89),
    ),
}
SUB_EMPHASIS_DB["LP"] = SUB_EMPHASIS_DB["EP"]

DB_PER_NEPER = 8.685889638065035


def sub_emphasis(frequency_hz, input_level_db: float = 0.0,
                 speed: str = "SP") -> np.ndarray:
    """The record-side sub pre-emphasis, as the standard prints it.

    IEC 60774-3:1993 section 6.1.1 and table 3. Interpolated in frequency on
    a logarithmic axis, because the table's own points are logarithmically
    spaced, and in input level linearly between the tabulated 0, -10, -20
    and -30 dB rows. Outside the tabulated band the nearest tabulated point
    is held, since the standard says nothing beyond 5 MHz and extrapolating
    a shape a standard does not state would be inventing it.

    Tolerances the standard gives alongside each value run from +-0.30 dB at
    the low frequencies and high input levels to +-0.70 dB at 3 and 5 MHz on
    the -30 dB row; a fit against this shape should be weighted by them.

    THIS IS A DIFFERENT KIND OF SIGNATURE from every loss in this module.
    The losses are monotone decays in one dimensionless group and collapse
    together; this one is level-dependent, so it varies along an axis none
    of them touch, and it is the sort of entry that makes a key more
    complete rather than merely larger.
    """
    f = _grid(frequency_hz)
    table = SUB_EMPHASIS_DB.get(str(speed).upper())
    if table is None:
        raise ValueError(f"no sub pre-emphasis table for speed {speed!r}")
    levels = np.asarray(SUB_EMPHASIS_INPUT_DB, dtype=np.float64)
    wanted = float(np.clip(input_level_db, levels.min(), levels.max()))
    rows = np.asarray(table, dtype=np.float64)
    # linear between the tabulated input levels
    curve = np.array([np.interp(wanted, levels[::-1], rows[::-1, column])
                      for column in range(rows.shape[1])])
    grid = np.log10(np.asarray(SUB_EMPHASIS_HZ, dtype=np.float64))
    shape = np.interp(np.log10(np.maximum(f, 1.0)), grid, curve)
    return (shape / DB_PER_NEPER).astype(np.complex128)


# --------------------------------------------------------------------------
# The set, and what to do with it
# --------------------------------------------------------------------------


# THE ORDER OF THE COMPONENTS MUST BE KNOWN.
#
# Ethan: *"The order of the components MUST be know though, and the shape of
# the keys MUST be subtractable."* The second is enforced by `subtractable`
# below; this is the first.
#
# A chain can only be inverted in the reverse of the order it was applied,
# and the standards state that order rather than leaving it to be inferred -
# SMPTE RP 86 section 2.1 gives the normative recording chain as burst
# amplitude modifier, then pre-emphasis, then ONE FM modulator, and the
# playback side is its mirror. Each entry below carries its position, and
# `ordered_key` returns them in the sequence a correction must undo them.
#
# Position is the stage the modification happens AT, counting from the
# source. Two entries may share a position when they act at the same point
# and commute; nothing may be left without one.
COMPONENT_ORDER: Dict[str, int] = {
    # THIS MAP DECLARES ONLY THIS MODULE'S OWN ENTRIES, and `full_chain()`
    # below composes it with every other component module's.
    #
    # Copying theirs in here was tried and was wrong. Each module's tests
    # assert that IT owns its declarations and that its entries are NOT
    # resolvable until a caller registers them - an opt-in, so that adding a
    # module cannot silently move numbers its other callers have measured.
    # A second copy here makes the ownership ambiguous and lets the two
    # drift apart.
    #
    # The positions leave room for the rest, and `full_chain` refuses any
    # name two modules place differently.

    # the transmission path, if the source came over the air - these act
    # BEFORE the recorder saw the signal at all
    "vestigial sideband": 1,
    "group delay": 2,
    "sound trap": 3,
    "echo": 4,                       # every delay shares this position

    # the recording machine, before the tape
    "sub emphasis level dependence": 10,
    # The record head's own current, which sets the depth the tape holds:
    # after the emphasis, and before the head-to-tape interface the path
    # modules occupy - the current is set by the electronics before the
    # geometry it is delivered through is reached.
    "record level dependence": 11,

    # the tape and the head
    "head contact tilt": 20,
    "dropout": 21,
    "modulation noise": 22,
    "particle noise": 23,            # the floor, and last

    # the capture
    "beat / co-channel": 30,
}


def full_chain(strict: bool = True) -> Dict[str, int]:
    """Every component module's declared positions, composed into one chain.

    Ethan: *"The order of the components MUST be know though."* Each module
    owns its own declarations - that is what its tests assert - so the chain
    is ASSEMBLED here rather than copied. A module that is not installed
    contributes nothing, and a caller that wants a module's entries orderable
    registers them itself, which is the opt-in those tests require.

    The bands, and the four links Ethan named:

        0        the source's own inserted test signals
        1 - 7    the transmission path: the tuner's own clip at 5,
                 because the receiver is the last thing the SOURCE chain
                 does to the video and it is present only when the
                 material came over the air, then the clamp's waveform
                 distortion at 6 - which is below the recorder's clip at
                 9, where Ethan localises the back-porch residual - and the
                 VCR's own AGC at 8, which sets the level at which the
                 input clip at 9 is reached - Ethan: "AGC happens in the VCR"
         9 - 19  the RECORDING machine       (TV -> VCR, VCR -> tape),
                 opening at 9 with its input stage, where a clip is
                 applied to the incoming video before the emphasis
        20 - 23  the tape                    (the medium itself)
        24 - 29  the PLAYBACK machine        (tape -> VCR)
        30       the capture                 (VCR -> capture card)
        40 - 49  the picture stage, applied after the RF matrix

    With `strict`, a name two modules place differently raises, because a
    chain with an ambiguous order cannot be inverted.
    """
    chain: Dict[str, int] = dict(COMPONENT_ORDER)
    sources: Dict[str, str] = {name: "interference" for name in chain}
    for module_name in ("clipping", "magnetic", "tape_path",
                        "transport_dimensions",
                        "head_differential", "rf_stages", "colour_under",
                        "sync_geometry", "vcr_agc",
                        "picture_stage", "vertical_interval",
                        # the Hi-Fi audio carriers, if a video-tap capture
                        # ever shows one. The entry is declared so that a
                        # detection goes through the same door as every
                        # other component rather than round it; on present
                        # evidence it is refused for want of a detection
                        # (capture_alignment.carrier_route_bound).
                        "capture_alignment"):
        try:
            module = __import__(f"vhsdecode.models.{module_name}",
                                fromlist=[module_name])
        except Exception:
            continue
        declared = getattr(module, "COMPONENT_ORDER", None)
        if not isinstance(declared, dict):
            declared = getattr(module, "COMPONENT_POSITIONS", None)
        if declared is None:
            prefix = getattr(module, "CHAIN_PREFIX", None)
            position = getattr(module, "CHAIN_POSITION", None)
            declared = ({prefix: position}
                        if prefix is not None and position is not None
                        else None)
        if not isinstance(declared, dict):
            declared = getattr(module, "CHAIN_POSITIONS", None)
        if not isinstance(declared, dict):
            continue
        for name, position in declared.items():
            if strict and name in chain and chain[name] != position:
                raise ValueError(
                    f"{module_name} places {name!r} at {position} where "
                    f"{sources[name]} places it at {chain[name]}; a chain "
                    "with an ambiguous order cannot be inverted")
            chain[name] = int(position)
            sources.setdefault(name, module_name)
    return chain




def position_of(name: str,
                chain: Optional[Dict[str, int]] = None) -> Optional[int]:
    """Where in the chain an entry acts, or None if it is undeclared.

    `chain` is the map to look in, and defaults to THIS MODULE'S OWN
    declarations. Every other component module's entries are opt-in - that
    is what their tests assert - so a caller that has registered them passes
    the composed map from `full_chain()` here. Without that a composed chain
    could be assembled and not ordered, which is the one thing the order
    exists for.
    """
    table = COMPONENT_ORDER if chain is None else chain
    if name in table:
        return table[name]
    for key, place in table.items():
        if name.startswith(key):
            return place
    return None


def ordered_key(entries: Dict[str, np.ndarray],
                chain: Optional[Dict[str, int]] = None
                ) -> List[Tuple[int, str]]:
    """The entries in the order a correction must undo them: LAST APPLIED
    FIRST REMOVED, which is descending position.

    Raises if any entry has no declared position, because a key whose order
    is unknown cannot be inverted - the requirement is not advisory. Pass
    `chain=full_chain()` to order entries belonging to the opt-in modules.
    """
    undeclared = [name for name in entries
                  if position_of(name, chain) is None]
    if undeclared:
        raise ValueError(
            "these key entries have no declared position in the chain, so "
            "the key cannot be inverted: " + ", ".join(sorted(undeclared)))
    return sorted(((position_of(name, chain), name) for name in entries),
                  key=lambda pair: (-pair[0], pair[1]))


def subtractable(signature: np.ndarray) -> bool:
    """THE REQUIREMENT ON EVERY KEY ENTRY.

    Ethan: *"the shape of the keys MUST be subtractable."* The transform
    works in the log domain, where a multiplicative modification becomes an
    additive one and a key can be taken off by subtraction. An entry whose
    magnitude reaches zero has no logarithm there, so it cannot be
    subtracted; entered anyway, the clamp turns it into a large constant and
    the fit sees that constant instead of the shape.

    Two ways an entry fails and both were present: a MASK, which is zero
    outside its support, and a POWER contribution, which is zero where the
    interferer is not. Both have subtractable forms - a bounded attenuation
    `1 - d*shape` and a power addition `1 + a*shape` - and those are what
    this module now returns.
    """
    magnitude = np.abs(np.asarray(signature).ravel())
    if not np.all(magnitude > 1e-12):
        return False
    logs = np.log(magnitude)
    return bool(np.all(np.isfinite(logs)))


def signatures(frequency_hz, mechanics: Optional[Dict[str, float]] = None,
               carrier_hz: float = 3.9e6,
               delays_s: Optional[List[float]] = None,
               include: Optional[List[str]] = None
               ) -> Dict[str, np.ndarray]:
    """Every modelled interference type on one frequency grid.

    Feed the result to `ellipsoid(..., real_parameters=True)` to read how
    many are distinguishable given the band, or to `component_differentials`
    to enter them as components in their own right. The particle-noise
    entry is the floor the others are judged against, not a correctable
    term.

    `delays_s` defaults to four echoes spread across the ghost window, which
    is where the well-conditioned dimensions are.

    `include` names opt-in component modules whose entries join the key,
    each ORTHOGONALISED against everything already in it. It defaults to
    none, so nothing a caller has already measured moves when a module is
    added to the tree. `admission` measures which of them earn a place on a
    given band and returns the list to pass here.
    """
    f = _grid(frequency_hz)
    low, high = float(f.min()), float(f.max())
    mechanics = mechanics or {}
    speed = mechanics.get("writing_speed_m_s", 5.8)
    width = mechanics.get("track_width_m", 58e-6)
    # None means the specification's own law, d = lambda/4
    depth = mechanics.get("recording_depth_m")
    if delays_s is None:
        span = high - low
        delays_s = [1.0 / max(span, 1.0) * k for k in (1, 2, 4, 8)]
    out = {
        "particle noise": particle_noise(f, speed, width, depth),
        "modulation noise": modulation_noise(f, carrier_hz, 0.05 * (high - low)),
        "dropout": dropout(f, 0.4 * (high - low), 0.5 * (low + high)),
        "head contact tilt": head_contact_tilt(f, low, high),
        "beat / co-channel": beat(f, 0.5 * (low + high)),
        "vestigial sideband": vestigial_sideband(f, low, 0.25 * (high - low)),
        "group delay": group_delay(f, 1.0 / max(high, 1.0) ** 2),
        "sound trap": sound_trap(f),
        # A SPECIFIED modification. Entered as its LEVEL DEPENDENCE - the
        # difference between two tabulated input levels - and not as the
        # curves themselves. Measured: adding the 0 dB and -20 dB curves as
        # two entries LOWERED the effective count from 6.92 to 6.00 and
        # worsened the condition from 10.5 to 28.5, because two curves of
        # nearly the same shape are one direction and a second copy of a
        # direction already in the span only makes the ensemble worse
        # conditioned. THE KEY MUST SPAN, NOT MERELY BE LARGE. What is new
        # about this modification is that it varies with LEVEL, an axis no
        # loss mechanism varies along, so that is what enters.
        "sub emphasis level dependence":
            sub_emphasis(f, -20.0) - sub_emphasis(f, 0.0),
    }
    for index, delay in enumerate(delays_s):
        out[f"echo at {delay * 1e6:.2f} us"] = echo(f, delay)

    # THE MAGNETIC RESIDUAL'S OWN AXIS, included only where the band can
    # see it.
    #
    # Entered as the LEVEL DEPENDENCE and not as the response at a level:
    # record level's frequency shape is 0.9995 coherent with thickness loss
    # and 1.0000 with spacing loss, so the response itself is another member
    # of the family that collapsed to 1.58 of 6, while its movement WITH
    # level is a direction nothing else supplies.
    #
    # AND ITS DIRECTION LIVES AT THE CROSSOVER - the frequency where the
    # self-demagnetisation cap takes over from the record level in setting
    # the depth. A band that does not contain it sees a nearly featureless
    # signature, and including it there costs conditioning without adding a
    # direction. Measured on the VHS luma band, which contains the 6.15 MHz
    # crossover: 7.91 effective of 14 at condition 7.8, against 7.66 of 13
    # without it. On the carrier band alone, which does not: 6.38 at
    # condition 180.8. The rule is the same one the emphasis taught - an
    # entry earns its place by the direction it adds.
    # ORTHOGONALISED AGAINST THE MAGNETICS BEFORE IT ENTERS. Raw, most of
    # its shape duplicates the losses it sits beside and it makes the key
    # worse - 6.370 effective at condition 159.02 against 6.996 at 11.66
    # without it. Projected orthogonal to them, 7.593 at condition 16.52.
    # Three independent derivations agreed on this, and it is the same rule
    # the arc's product terms already obey.
    losses = [out[name] for name in
              ("head contact tilt", "dropout", "modulation noise",
               "particle noise") if name in out]
    out["record level dependence"] = magnetic.orthogonal_level_signature(
        f, 1.0, speed, losses)

    # Each module's entries are projected against the key AS IT STOOD BEFORE
    # that module, not against a key that already holds its own siblings.
    # A module's entries carry relationships its own tests assert, and
    # orthogonalising them against each other would break those; more to the
    # point, `admission` measures the key this way, and a builder that
    # assembled it differently would make that measurement describe
    # something other than what it built. The two read 15.18 of 23 at
    # condition 10.8 and 14.72 at 9.8 before this was made to agree.
    for module_name in (include or ()):
        established = list(out.values())
        for name, value in _candidate_signatures(
                module_name, f, mechanics).items():
            out[name] = orthogonalised(value, established)
    return out


# --------------------------------------------------------------------------
# The opt-in modules, and the measurement that decides which of them join
# --------------------------------------------------------------------------

# Every component module that emits FREQUENCY-axis signatures and is not in
# the key by default. They are opt-in for the reason `COMPONENT_ORDER` gives:
# adding a module must not silently move numbers a caller has already
# measured. `admission` below says which of them earn a place on a given
# band, and `signatures(include=...)` puts them there.
#
# `transport_dimensions` is deliberately absent: its signatures are on the
# TIME axis, and a time-axis member cannot join a frequency-axis ensemble.
# Matching is on the dimension and the period, never on the name.
CANDIDATE_MODULES = ("colour_under", "head_differential", "tape_path",
                     "picture_stage", "sync_geometry", "vertical_interval")


def orthogonalised(value, against) -> np.ndarray:
    """A signature with the part already spanned by `against` removed.

    THE RULE PROOF 31 ESTABLISHED, applied to every entry that joins the key
    rather than to one of them. Entered raw, a new entry whose shape
    duplicates the family it joins costs conditioning without adding a
    direction - measured on the record-level entry, 6.370 effective at
    condition 159.02 against 6.996 at 11.66 without it. Projected orthogonal
    to that family first, 7.593 at 16.52.

    The projection is done on the LOG shape, which is where the ensemble is
    read, and the result is mapped back through `exp` and bounded so it is
    still subtractable.
    """
    vector = _log_shape(np.asarray(value))
    for other in against:
        reference = _log_shape(np.asarray(other))
        if reference.size != vector.size:
            continue
        norm = float(np.linalg.norm(reference))
        if norm > 1e-12:
            unit = reference / norm
            vector = vector - (unit.conj() @ vector) * unit
    scale = float(np.max(np.abs(vector)))
    if scale > 0:
        vector = vector / scale * 0.5
    return np.exp(vector)


def _log_shape(value: np.ndarray) -> np.ndarray:
    """The log magnitude and unwrapped phase, each with its mean removed -
    the form the whole arc reads an ensemble in, because a constant is a
    level and not a shape."""
    magnitude = np.log(np.maximum(np.abs(value), 1e-12))
    phase = (np.unwrap(np.angle(value)) if np.iscomplexobj(value)
             else np.zeros_like(magnitude))
    return ((magnitude - magnitude.mean())
            + 1j * (phase - phase.mean())).astype(np.complex128)


def _candidate_signatures(module_name: str, frequency_hz,
                          mechanics: Optional[Dict[str, float]] = None
                          ) -> Dict[str, np.ndarray]:
    """One opt-in module's frequency-axis entries, called the way it asks.

    Each module owns its own call convention, so the adaptation lives here
    rather than forcing nine modules to share a signature they did not
    choose.
    """
    from importlib import import_module

    if module_name not in CANDIDATE_MODULES:
        raise ValueError(f"{module_name!r} is not an opt-in component module;"
                         f" the candidates are {CANDIDATE_MODULES}")
    module = import_module(f"vhsdecode.models.{module_name}")
    mechanics = mechanics or {}
    speed = mechanics.get("writing_speed_m_s", 5.8)
    if module_name == "tape_path":
        return module.signatures(frequency_hz, speed)
    if module_name == "colour_under":
        return module.signatures(
            frequency_hz, mechanics={"writing_speed_m_s": speed})
    return module.signatures(frequency_hz)


def _conditioning(entries: Dict[str, np.ndarray]):
    """The effective count and the condition of an assembled key."""
    shapes = [_log_shape(np.asarray(v)) for v in entries.values()]
    shapes = [s for s in shapes if np.linalg.norm(s) > 0]
    if not shapes:
        return 0.0, 0, 0.0
    stack = np.array([np.concatenate([s.real, s.imag]) for s in shapes])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    values = np.linalg.svd(stack, compute_uv=False)
    share = values ** 2 / max(float((values ** 2).sum()), 1e-30)
    return (float(1.0 / np.sum(share ** 2)), len(shapes),
            float(values[0] / max(values[-1], 1e-30)))


def admission(frequency_hz, modules=None,
              mechanics: Optional[Dict[str, float]] = None,
              gain: float = 0.5, cost: float = 1.25,
              passes: int = 4, granularity: str = "module",
              total_cost: float = 4.0, **kwargs) -> Dict[str, object]:
    """WHICH OPT-IN MODULES EARN A PLACE IN THE KEY, ON THIS BAND, MEASURED.

    Ethan's rule, and the arc's: an entry earns its place by the DIRECTION
    it adds. A module is offered to the key with its entries orthogonalised
    against everything already there, and is admitted if the effective count
    rises by more than `gain` and the condition rises by no more than a
    factor of `cost`.

    RUN TO A FIXED POINT, AND THAT IS NOT A REFINEMENT. A single pass in any
    fixed order can wrongly hold a module, because orthogonalising against a
    RICHER key removes more of the shape a module duplicates. Measured on
    the VHS luma band, `head_differential` offered three times:

        to the bare key            +0.77 effective, condition +40.8   held
        after picture_stage        +1.42 effective, condition  +2.1   held
        after tape_path as well    +2.22 effective, condition  +0.2   ADMITTED

    The same module, the same band, the same gate - and the trade turns from
    bad to good purely because the key it is projected against got richer.
    A single pass would have discarded it. Held modules are therefore
    re-offered until a pass admits nothing.

    Measured, three bands, all three from the same 14-entry key:

        VHS luma band   8.45 of 14 at   7.4  ->  14.72 of 23 at   9.8
        full RF         8.81 of 14 at  16.8  ->  17.67 of 25 at  18.8
        carrier only    6.88 of 14 at  19.9  ->   6.88 of 14 at  19.9

    On the two bands that admit anything the key roughly DOUBLES its
    effective directions while its conditioning improves relative to the
    count - 8.45 to 14.72 for a condition of 9.8 against 7.4.

    AND OFFERING A MODULE AS ONE BLOCK IS ITSELF A LIMITATION, which is what
    `granularity="entry"` removes. A module is held when its entries are
    collinear WITH EACH OTHER, whatever they would each be worth alone.
    `vertical_interval` is the case: its 25 test-signal entries measure 8.62
    effective of 25 at condition 1518 BEFORE they ever meet the key, and the
    offer's condition of 1616 is essentially that intrinsic figure. The cause
    is narrower than "too many entries" - eight of the colour bars share one
    6 microsecond window, so in a band that sees none of their energy
    directly they enter as sinc tails with the same lobe spacing, and grey
    against black measures 0.999947 coherent. Twenty-five entries do not
    wreck a key: twenty-five orthogonalised ECHOES added to the same key give
    33.06 of 48 at condition 15.08 and pass comfortably.

    Offered entry by entry to the same fixed point, every one of them is
    admitted:

        VHS luma band   8.45 of 14 at   7.4  ->  41.67 of 52 at  17.9
        full RF         8.81 of 14 at  16.8  ->  45.93 of 55 at  25.5

    THAT IS NOT THE GATE BEING GAMED, AND `total_cost` IS WHY IT CAN BE
    CHECKED. Thirty-eight separate offers at 1.25 each would license a 4815x
    rise in the condition; the measured rise is 2.417x on the luma band and
    1.517x on the full RF. Each entry stops duplicating once its siblings are
    in the key, so the later offers cost almost nothing - and the total
    ceiling bounds the whole admission against where it started rather than
    leaving that to be incidentally true.

    The default granularity stays "module", so no number a caller has already
    measured moves when this is used.

    The carrier band admits NOTHING, and that is the measurement working
    rather than failing: 2.5 to 5.3 MHz is too narrow to tell these shapes
    apart, so every candidate costs more conditioning than the direction it
    brings. `vertical_interval` is held on every band - its 25 entries are
    specified test SIGNALS rather than modifications of the chain, and they
    take the condition past 1500.
    """
    if granularity not in ("module", "entry"):
        raise ValueError("granularity must be 'module' or 'entry'")
    candidates = list(modules if modules is not None else CANDIDATE_MODULES)
    key = dict(signatures(frequency_hz, mechanics=mechanics, **kwargs))
    effective, count, condition = _conditioning(key)
    start = (effective, count, condition)
    admitted: List[str] = []
    trace: List[Dict[str, object]] = []

    offers: List[Tuple[str, Dict[str, np.ndarray]]] = []
    for module_name in list(candidates):
        entries = _candidate_signatures(module_name, frequency_hz, mechanics)
        if not entries:
            trace.append({"module": module_name, "verdict": "empty",
                          "why": "emits nothing on this band"})
            candidates.remove(module_name)
            continue
        if granularity == "entry":
            offers.extend((f"{module_name}:{name}", {name: value})
                          for name, value in entries.items())
        else:
            offers.append((module_name, entries))
    if granularity == "entry":
        candidates = [label for label, _ in offers]

    for _ in range(max(int(passes), 1)):
        moved = False
        for module_name in list(candidates):
            entries = dict(next(e for label, e in offers
                                if label == module_name))
            trial = dict(key)
            for name, value in entries.items():
                trial[name] = orthogonalised(value, key.values())
            new_effective, new_count, new_condition = _conditioning(trial)
            # TWO CEILINGS, AND THE SECOND IS WHAT MAKES ENTRY-WISE
            # OFFERING HONEST. The per-offer one limits a single step; on
            # its own, 38 separate offers at 1.25 each would license a 4815x
            # rise, which is a ratchet rather than a gate. The total ceiling
            # bounds the whole admission against where it started, so the
            # guarantee is auditable instead of incidentally small.
            earns = (new_effective > effective + float(gain)
                     and new_condition <= max(condition * float(cost),
                                              condition + 1e-9)
                     and new_condition <= start[2] * float(total_cost))
            trace.append({
                "module": module_name, "entries": len(entries),
                "effective": new_effective, "condition": new_condition,
                "delta_effective": new_effective - effective,
                "delta_condition": new_condition - condition,
                "verdict": "admitted" if earns else "held",
            })
            if earns:
                key, effective, count, condition = (
                    trial, new_effective, new_count, new_condition)
                admitted.append(module_name)
                candidates.remove(module_name)
                moved = True
        if not moved:
            break
    return {
        "admitted": admitted,
        "held": candidates,
        "granularity": granularity,
        "condition_rise": float(condition / max(start[2], 1e-30)),
        "condition_ceiling": float(start[2] * float(total_cost)),
        "licensed_by_per_offer_gate": float(float(cost) ** len(admitted)),
        "start": {"effective": start[0], "count": start[1],
                  "condition": start[2]},
        "end": {"effective": effective, "count": count,
                "condition": condition},
        "trace": trace,
        "why": ("an entry earns its place by the direction it adds; held "
                "modules are re-offered because orthogonalising against a "
                "richer key removes more of what they duplicate"),
    }


def key_after_admission(frequency_hz, verdict: Dict[str, object],
                        mechanics: Optional[Dict[str, float]] = None,
                        **kwargs) -> Dict[str, np.ndarray]:
    """The key exactly as `admission` left it, rebuilt from its verdict.

    `signatures(include=...)` takes whole modules, so an ENTRY-WISE verdict
    could not be turned back into a key without re-running the admission.
    This walks the admitted labels in the order they were admitted and
    orthogonalises each against the key as it stood at that moment, which
    is the same construction `admission` measured, so the effective count
    and condition of the result are the ones its verdict reports.
    """
    key = dict(signatures(frequency_hz, mechanics=mechanics, **kwargs))
    for label in verdict.get("admitted", []):
        if verdict.get("granularity") == "entry":
            module_name, name = str(label).split(":", 1)
            entries = {name: _candidate_signatures(
                module_name, frequency_hz, mechanics)[name]}
        else:
            entries = _candidate_signatures(str(label), frequency_hz,
                                            mechanics)
        established = list(key.values())
        for name, value in entries.items():
            key[name] = orthogonalised(value, established)
    return key


def span_completeness(residuals: Dict[str, np.ndarray],
                      frequency_hz, **kwargs) -> Dict[str, object]:
    """IS THE KEY COMPLETE? The share of a residual the modelled set spans.

    Ethan's framing, and it is the right one:

        it can be solved by finding as many dimensions as exist and being
        able to model them exactly, which you cannot since they encode
        information. That is the part that you cannot know unless you have
        the key. We have the key which is all possible modifications to our
        video signal, that's the key.

    The key is the set of every modification the chain can make. If that set
    SPANS the space, any residual decomposes into it uniquely and nothing is
    left to guess. If it does not, the part outside the span is
    unrecoverable no matter how the fit is arranged - not because the method
    is weak but because nothing in the model can represent it.

    So completeness is a measurement, not an assumption: orthonormalise the
    modelled signatures, project each residual onto that basis, and report
    the share of its energy that lands inside. `inside` near one means the
    key covers what the tape did. `outside` is the part the key is missing,
    and it is the honest upper bound on what any amount of further fitting
    can recover.

    The measure is charitable to the key by construction - a basis of `k`
    vectors captures `k/L` of ANY residual by chance - so `excess` reports
    the share above that null, which is what says the key covers the
    residual rather than merely being large.
    """
    signature = signatures(frequency_hz, **kwargs)
    basis = []
    for value in signature.values():
        vector = np.asarray(value).ravel().astype(np.complex128)
        vector = vector - vector.mean()
        for earlier in basis:
            vector = vector - (earlier.conj() @ vector) * earlier
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            basis.append(vector / norm)
    if not basis:
        return {"inside": 0.0, "outside": 1.0, "basis": 0, "excess": 0.0}
    length = int(np.asarray(next(iter(signature.values()))).size)
    inside_total, energy_total, shares = 0.0, 0.0, []
    for value in residuals.values():
        vector = np.nan_to_num(np.asarray(value).ravel().astype(np.complex128))
        vector = vector - vector.mean()
        energy = float(np.vdot(vector, vector).real)
        if not energy > 0:
            continue
        captured = sum(abs(complex(direction.conj() @ vector)) ** 2
                       for direction in basis
                       if direction.size == vector.size)
        inside_total += captured
        energy_total += energy
        shares.append(captured / energy)
    if not energy_total > 0:
        return {"inside": 0.0, "outside": 1.0, "basis": len(basis),
                "excess": 0.0}
    inside = inside_total / energy_total
    # what a basis of this size would capture from anything at all
    null = len(basis) / max(length, 1)
    return {
        "inside": float(inside),
        "outside": float(1.0 - inside),
        "basis": len(basis),
        "length": length,
        "null": float(null),
        "excess": float(inside - null),
        "per_component": np.asarray(shares),
        "why": ("the share of each residual lying inside the span of the "
                "modelled modifications; what is outside cannot be "
                "recovered by any fit, because nothing models it"),
    }


def outlier_response(residual, frequency_hz, noise_floor,
                     threshold: float = 3.0, **kwargs) -> Dict[str, object]:
    """OUTLIERS ABOVE THE NOISE FLOOR THAT THE MODEL DOES NOT EXPLAIN, AND
    WHICH MODELLED DIMENSION CARRIES THEIR RESPONSE.

    Ethan: *"outliers that do not explain the model, but are more
    significant than the noise floor, should be corrected at the residual
    step. We need to measure the response of the outlier, which should be
    contained in one of the many dimensions we have."*

    The instruction is precise and this implements it literally. An entry is
    an outlier when it stands above the noise floor by more than `threshold`
    - below that there is nothing to explain, because the floor is the total
    information the sample holds. What makes it worth correcting rather than
    discarding is that its RESPONSE should already lie in the span of the
    dimensions the key holds: an outlier is a modification the chain made,
    and the key is the set of every modification the chain can make.

    So the outlier's own shape is projected onto the key and this reports
    which dimension carries it, how much of it the key spans, and the
    residual with that part removed. THREE OUTCOMES, AND THEY MEAN DIFFERENT
    THINGS:

        the key spans it              the outlier is a known modification
                                      read at an unusual size; correct it
                                      and attribute it to the dimension named
        the key spans part of it      correct that part; the rest is the
                                      honest bound
        the key does not span it      NOTHING models it, so no amount of
                                      fitting recovers it - and the right
                                      response is a new dimension, not a
                                      larger coefficient on an old one

    The share is reported against the null a basis of this size captures
    from ANY vector by chance, `basis/length`, because a large key spans a
    lot of everything and the excess above that null is what says the key
    covers the outlier rather than merely being large.
    """
    values = np.nan_to_num(np.asarray(residual).ravel().astype(np.complex128))
    grid = _grid(frequency_hz)
    floor = np.abs(np.broadcast_to(np.asarray(noise_floor, dtype=np.float64),
                                   values.shape))
    significance = np.abs(values) / np.maximum(floor, 1e-30)
    is_outlier = significance > float(threshold)

    key = signatures(grid, **kwargs)
    basis, names = [], []
    for name, entry in key.items():
        vector = np.nan_to_num(
            np.asarray(entry).ravel().astype(np.complex128))
        if vector.size != values.size:
            continue
        vector = vector - vector.mean()
        # Gram-Schmidt against what is already there, so a dimension is not
        # credited with a share another dimension already explained
        for direction in basis:
            vector = vector - (direction.conj() @ vector) * direction
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            basis.append(vector / norm)
            names.append(name)

    outlying = np.where(is_outlier, values, 0.0)
    outlying = outlying - outlying.mean()
    energy = float(np.vdot(outlying, outlying).real)
    if not basis or not energy > 0:
        return {
            "outliers": int(is_outlier.sum()),
            "inside": 0.0, "outside": 1.0, "excess": 0.0,
            "carried_by": None, "corrected": values,
            "why": ("nothing modelled is on this grid, or no entry stands "
                    "above the floor"),
        }

    projection = np.zeros_like(outlying)
    shares = {}
    for name, direction in zip(names, basis):
        weight = complex(direction.conj() @ outlying)
        projection = projection + weight * direction
        shares[name] = float(abs(weight) ** 2 / energy)

    inside = float(np.vdot(projection, projection).real / energy)
    null = len(basis) / max(values.size, 1)
    carried = max(shares, key=shares.get)
    corrected = values.copy()
    corrected[is_outlier] = (values - projection)[is_outlier]
    return {
        "outliers": int(is_outlier.sum()),
        "outlier_fraction": float(is_outlier.mean()),
        "threshold": float(threshold),
        "peak_significance": float(significance.max()),
        "inside": inside,
        "outside": float(1.0 - inside),
        "null": float(null),
        "excess": float(inside - null),
        "basis": len(basis),
        "shares": shares,
        "carried_by": carried,
        "carried_share": shares[carried],
        "projection": projection,
        "corrected": corrected,
        "spanned": bool(inside - null > 0.0),
        "why": ("an outlier is a modification the chain made, and the key is "
                "the set of every modification the chain can make; what the "
                "key cannot span is not correctable by any fit, and calls "
                "for a new dimension rather than a larger coefficient"),
    }


def split_residual(residuals: Dict[str, np.ndarray], frequency_hz,
                   draws: int = 24, seed: int = 0,
                   **kwargs) -> Dict[str, object]:
    """THE THREE-WAY SPLIT of a measured residual.

    Ethan: *"There are a fixed number of components that exist in a video
    signal that represent video and noise randomness."* The count is fixed
    by band and duration alone - `2BT` degrees of freedom, 116,783 complex
    over the 7 MHz Carson band in one NTSC field - and everything else is a
    partition of it. Three parts, and only the middle one is worth work:

    | part | what it is | how it is told |
    |---|---|---|
    | inside the key | the modelled modifications | projection onto the span |
    | outside, still structured | video and mechanisms not yet modelled | stands above the surrogate null |
    | outside, unstructured | noise; nothing models it, ever | sits at the null |

    The second and third could not be told apart until there was a null
    built by generating the same construction on structureless input.
    A mechanism the key is missing still leaves a preferred direction; noise
    does not. That distinction is the whole value of the split, because the
    second part is the only thing that is both unreached and reachable.

    Returns the three shares, and for the outside part the asymmetry it
    shows against a null of its own size.
    """
    from vhsdecode.models import information_extrapolation as ie

    cover = span_completeness(residuals, frequency_hz, **kwargs)
    if not cover.get("basis"):
        return {"inside": 0.0, "structured": 0.0, "noise": 0.0,
                "why": "no modelled basis"}

    signature = signatures(frequency_hz, **kwargs)
    basis = []
    for value in signature.values():
        vector = np.asarray(value).ravel().astype(np.complex128)
        vector = vector - vector.mean()
        for earlier in basis:
            vector = vector - (earlier.conj() @ vector) * earlier
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            basis.append(vector / norm)

    # what the key does NOT reach, per component
    outside: Dict[str, Dict[str, np.ndarray]] = {}
    for name, value in residuals.items():
        vector = np.nan_to_num(np.asarray(value).ravel().astype(np.complex128))
        vector = vector - vector.mean()
        if not float(np.vdot(vector, vector).real) > 0:
            continue
        remainder = vector
        for direction in basis:
            if direction.size == remainder.size:
                remainder = remainder - (direction.conj() @ remainder) * direction
        outside[name] = {"frequency": remainder}
    if len(outside) < 3:
        return {"inside": cover["inside"], "structured": 0.0,
                "noise": cover["outside"],
                "why": "too few components to test the remainder"}

    # is what is left still structured, or is it the floor?
    fit = ie.ellipsoid(outside, "frequency")
    null = ie.surrogate_null(outside, "frequency", draws=draws, seed=seed)
    z = float(null.get("asymmetry_z", 0.0))
    # the share of the outside part that stands above the null, taken as the
    # resolved fraction the ellipse finds there, and zero when it does not
    # stand above it at all
    reaches = z > ie.SPHERE_SIGMA
    share = float(fit["resolved_fraction"]) if reaches else 0.0
    return {
        "inside": float(cover["inside"]),
        "structured": float(cover["outside"] * share),
        "noise": float(cover["outside"] * (1.0 - share)),
        "outside_total": float(cover["outside"]),
        "outside_asymmetry": float(fit["asymmetry"]),
        "outside_null_mean": float(null.get("asymmetry_mean", 0.0)),
        "outside_z": z,
        "outside_stands_above_the_null": bool(reaches),
        "basis": cover["basis"],
        "excess": float(cover["excess"]),
        "why": ("inside the key, outside but structured, and outside and "
                "unstructured; the middle part is the only one that is both "
                "unreached and reachable"),
    }


def interpolate_unmodelled(residuals: Dict[str, np.ndarray], frequency_hz,
                           band: float = 0.02, draws: int = 16,
                           seed: int = 0, **kwargs) -> Dict[str, object]:
    """WHAT CANNOT BE MODELLED CAN STILL BE RECONSTRUCTED.

    Ethan: *"If those cannot be modeled, that defines the variance in shape
    of this multidimentional residual and this can be substrituted with an
    upscaling algorithm, like sinc, but on the residual dimentions."*

    The middle part of the three-way split - structured, but outside the key
    - is real and has a shape, and nothing in the modelled set reaches it.
    But a structured quantity does not need a parametric model to be carried:
    if it is band-limited on the abscissa it is *determined* by its samples,
    and the band-limited interpolation reconstructs it exactly. So where the
    key cannot supply a shape, the residual's own samples can, and what is
    substituted is an interpolation rather than a mechanism.

    THE CONDITION THIS RESTS ON, and it is checkable rather than assumed:
    the remainder must be band-limited on the abscissa. Structure is smooth
    - a departure that varies as fast as the grid is noise by definition -
    so the test is whether the remainder's energy concentrates in the low
    part of its own transform. `band` is the fraction of the transform kept,
    and `concentration` reports how much of the remainder that fraction
    holds. A remainder that is genuinely structured concentrates; noise
    spreads evenly and reports its own equal share.

    This is the same reasoning the sphere floor rests on, one level down: an
    equal share is what having nothing to say looks like.
    """
    entries = signatures(frequency_hz, **kwargs)
    basis = []
    for value in entries.values():
        vector = np.asarray(value).ravel().astype(np.complex128)
        vector = vector - vector.mean()
        for earlier in basis:
            vector = vector - (earlier.conj() @ vector) * earlier
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            basis.append(vector / norm)

    kept = max(int(round(band * len(np.asarray(frequency_hz).ravel()))), 2)
    reconstructed: Dict[str, np.ndarray] = {}
    concentrations, gains = [], []
    for name, value in residuals.items():
        vector = np.nan_to_num(np.asarray(value).ravel().astype(np.complex128))
        vector = vector - vector.mean()
        remainder = vector
        for direction in basis:
            if direction.size == remainder.size:
                remainder = remainder - (direction.conj() @ remainder) * direction
        energy = float(np.vdot(remainder, remainder).real)
        if not energy > 0:
            continue
        # the band-limited reconstruction: keep the low part of the
        # remainder's own transform and return it to the abscissa, which is
        # sinc interpolation written in the domain where it is a projection
        spectrum = np.fft.fft(remainder)
        low = np.zeros_like(spectrum)
        low[:kept] = spectrum[:kept]
        low[-kept + 1:] = spectrum[-kept + 1:]
        smooth = np.fft.ifft(low)
        held = float(np.vdot(smooth, smooth).real) / energy
        concentrations.append(held)
        gains.append(held / (2.0 * kept / remainder.size))
        reconstructed[name] = smooth
    concentration = float(np.mean(concentrations)) if concentrations else 0.0
    equal_share = 2.0 * kept / max(len(np.asarray(frequency_hz).ravel()), 1)
    return {
        "reconstructed": reconstructed,
        "band": float(band),
        "kept": kept,
        "concentration": concentration,
        "equal_share": float(equal_share),
        "gain_over_equal_share": (float(np.mean(gains)) if gains else 0.0),
        "band_limited": bool(concentration > 2.0 * equal_share),
        "why": ("what the key cannot model is substituted by a band-limited "
                "reconstruction of the residual's own samples; it is "
                "structured only if it concentrates well above the equal "
                "share that having nothing to say produces"),
    }


def ideal_tape_response(frequency_hz, writing_speed_m_s: float = 5.8,
                        track_width_m: float = 58e-6,
                        particle_volume_m3: float = 1e-21,
                        packing: float = 0.4) -> Dict[str, np.ndarray]:
    """THE MAXIMUM POSSIBLE BANDPASS OF MEANINGFUL INFORMATION.

    Ethan: *"do an interpolation modeled over the ideal response of the VHS
    tape, i.e. the maximum possible bandpass of meaningful information."*

    Two limits bound it and both come from the medium rather than from any
    machine, so this is the tape's own ceiling and not a particular deck's.

    **The depth is capped by self-demagnetisation, not by the sensitivity
    optimum.** The guide gives the optimum recording depth as a quarter
    wavelength, and the shortest wavelength a coating of depth `d` can hold
    at all is `2*pi*d`, so recording wavelength `lambda` requires
    `d <= lambda/(2*pi)`. Since `1/(2*pi) = 0.159` is smaller than `1/4`,
    **the demagnetisation limit binds everywhere and the optimum is never
    reachable.** The two have no fixed point - substituting one into the
    other gives `lambda = 1.57 lambda` - and that is the physical statement,
    not a defect in either.

    The cost is exactly `10 log10((1/4)/(1/(2 pi))) = 1.96 dB`, everywhere,
    because both laws are proportional to the wavelength: the noise slope
    stays at twenty decibels per decade and only its level moves.

    **The head's own response bounds the other end.** A head reads the rate
    of change of flux, so its output rises with frequency and there is no
    output at all at DC.

    Returns the response in nepers, the signal-to-noise the medium allows at
    each frequency, and the band where meaningful information can exist.
    """
    f = _grid(frequency_hz)
    wavelength = writing_speed_m_s / np.maximum(f, 1.0)
    # the depth actually available, and the noise it implies
    depth = wavelength / (2.0 * np.pi)
    volume = wavelength * track_width_m * depth
    particles = np.maximum(volume * packing / particle_volume_m3, 1.0)
    snr_db = 10.0 * np.log10(particles)
    # the head's differentiation: output proportional to frequency
    response = np.log(np.maximum(f, 1.0) / max(float(np.max(f)), 1.0))
    meaningful = snr_db > 0.0
    return {
        "frequency_hz": f,
        "depth_m": depth,
        "snr_db": snr_db,
        "log_response": response,
        "meaningful": meaningful,
        "optimum_depth_m": wavelength / 4.0,
        "cost_of_the_limit_db": float(10.0 * np.log10(0.25 / (1.0 / (2.0 * np.pi)))),
        "why": ("the depth is capped at lambda/(2 pi) by self-"
                "demagnetisation, which is shallower than the lambda/4 "
                "sensitivity optimum, so the optimum is never reachable"),
    }


def interpolate_over_ideal(residuals: Dict[str, np.ndarray], frequency_hz,
                           writing_speed_m_s: float = 5.8,
                           track_width_m: float = 58e-6,
                           **kwargs) -> Dict[str, object]:
    """THE CLOSING STEP: the final unknown residual, interpolated over the
    tape's ideal response.

    Ethan: *"Using the final unknown residual, do an interpolation modeled
    over the ideal response of the VHS tape ... doing the process over that
    difference, which leads us to the absolute noise floor, since our video
    tape contains less information that what is possible to be stored on the
    capture data."*

    The last sentence is the reason it works. The capture holds more than
    the tape has to say - measured, five to nine times more - so the
    capture's band is WIDER than the tape's, and any part of the residual
    that the tape's own response could never have carried is not tape
    information at all. Weighting the remainder by the ideal response and
    reconstructing it there is therefore not a smoothing choice: it is
    discarding what the medium could not have recorded.

    What is left after that has nowhere else to be. It is inside the band,
    it is not modelled, and it does not reconstruct - so it is the absolute
    floor.
    """
    ideal = ideal_tape_response(frequency_hz, writing_speed_m_s,
                                track_width_m)
    weight = np.maximum(ideal["snr_db"], 0.0)
    weight = weight / max(float(weight.max()), 1e-30)

    # what the key does not reach
    entries = signatures(frequency_hz, **kwargs)
    basis = []
    for value in entries.values():
        vector = np.asarray(value).ravel().astype(np.complex128)
        vector = vector - vector.mean()
        for earlier in basis:
            vector = vector - (earlier.conj() @ vector) * earlier
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            basis.append(vector / norm)

    total = inside = reconstructed = 0.0
    for value in residuals.values():
        vector = np.nan_to_num(np.asarray(value).ravel().astype(np.complex128))
        vector = vector - vector.mean()
        energy = float(np.vdot(vector, vector).real)
        if not energy > 0:
            continue
        remainder = vector
        for direction in basis:
            if direction.size == remainder.size:
                remainder = remainder - (direction.conj() @ remainder) * direction
        total += float(np.vdot(remainder, remainder).real)
        # the part the tape's response could have carried
        within = remainder * weight
        inside += float(np.vdot(within, within).real)
        # and of that, the part a band-limited reconstruction recovers
        spectrum = np.fft.fft(within)
        keep = max(int(round(0.02 * within.size)), 2)
        low = np.zeros_like(spectrum)
        low[:keep] = spectrum[:keep]
        low[-keep + 1:] = spectrum[-keep + 1:]
        smooth = np.fft.ifft(low)
        reconstructed += float(np.vdot(smooth, smooth).real)
    if not total > 0:
        return {"why": "no residual with any energy"}
    return {
        "outside_the_tape_band": float(1.0 - inside / total),
        "inside_the_tape_band": float(inside / total),
        "reconstructable": float(reconstructed / total),
        # inside the band, unmodelled, and not reconstructable
        "absolute_floor": float((inside - reconstructed) / total),
        "band_weight_mean": float(weight.mean()),
        "cost_of_the_limit_db": ideal["cost_of_the_limit_db"],
        "why": ("what the tape's response could not have carried is not tape "
                "information; what it could, and that reconstructs, is "
                "recoverable; what remains is the absolute floor"),
    }


def noise_shape_differential(residuals: Dict[str, np.ndarray], frequency_hz,
                             writing_speed_m_s: float = 5.8,
                             track_width_m: float = 58e-6,
                             **kwargs) -> Dict[str, object]:
    """THE NOISE HAS A SHAPE, SO THE DIFFERENTIAL EXTENDS TO IT.

    Ethan: *"Since we expect there to be RF noise, which follows a certain
    shape, we can extend this same complex differential to the residual in
    this manner."*

    Right, and it changes what "at the floor" means. Particle noise is not
    shapeless: the read volume shrinks with the wavelength and the
    specification fixes the recording depth at a quarter wavelength, so the
    noise rises at twenty decibels per decade and nothing else in the
    modelled set does. That is a SYNTHETIC side for the noise itself, and
    the same differential applies - expected shape against measured shape.

    So a remainder is only the floor if it has the floor's SHAPE. One that
    is flat, or that falls, is not particle noise however small it is, and
    the departure is another component rather than the end.

    Returns the measured slope of the remainder against the predicted one,
    both in decibels per decade, and how far apart they are.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    predicted = np.abs(particle_noise(f, writing_speed_m_s, track_width_m))
    logf = np.log10(np.maximum(f, 1.0))
    def slope(values):
        y = 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))
        design = np.vstack([logf, np.ones_like(logf)]).T
        return float(np.linalg.lstsq(design, y, rcond=None)[0][0])
    expected = slope(predicted)
    measured = [slope(np.asarray(v).ravel()) for v in residuals.values()
                if np.any(np.abs(np.asarray(v)) > 0)]
    if not measured:
        return {"expected_slope": expected, "measured_slope": 0.0,
                "departure": 0.0, "has_the_floor_shape": False,
                "why": "no residual with any energy"}
    got = float(np.mean(measured))
    spread = float(np.std(measured)) if len(measured) > 1 else 0.0
    departure = got - expected
    return {
        "expected_slope": expected,
        "measured_slope": got,
        "measured_spread": spread,
        "departure": departure,
        # within a few dB per decade of the predicted rise
        "has_the_floor_shape": bool(abs(departure) < 5.0),
        "why": ("particle noise rises at twenty decibels per decade because "
                "the read volume goes as the wavelength squared; a remainder "
                "without that slope is not the floor, whatever its size"),
    }


def derive_against_zero(residuals: Dict[str, np.ndarray], frequency_hz,
                        draws: int = 24, seed: int = 0,
                        **kwargs) -> Dict[str, object]:
    """THE LAST SUBTRACTION: the final residual differentiated against zero.

    Ethan: *"Then once we get the complex value, we derive that against zero,
    so the final residual that is at the end is also subtracted."*

    Every earlier differential was `synthetic minus measured`, with the
    synthetic side taken from a standard or a physical model. For the last
    one the synthetic side is ZERO - nothing should remain once everything
    knowable has been removed - so the differential IS the residual, and
    applying it subtracts what is left.

    WHAT MUST NOT BE SUBTRACTED, AND WHY THIS IS NOT SIMPLY `x - x`. A
    remainder contains two populations, and only one of them may go. The
    part that stands above a null built by the same construction is
    structure the key did not model, and subtracting it is a correction. The
    part sitting at the null is noise, and subtracting THAT is fitting it -
    it would drive the reported residual to zero while making the signal
    worse, which is exactly the failure the arc's own withdrawal records.

    So the operation is: differentiate against zero, keep the part that
    survives the null, and subtract only that. What comes back is the
    corrected residual and an honest statement of how much was taken.
    """
    from vhsdecode.models import information_extrapolation as ie

    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    entries = signatures(grid, **kwargs)
    basis = []
    for value in entries.values():
        vector = np.asarray(value).ravel().astype(np.complex128)
        vector = vector - vector.mean()
        for earlier in basis:
            vector = vector - (earlier.conj() @ vector) * earlier
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            basis.append(vector / norm)

    # what the key does not reach: the final residual
    remainder: Dict[str, Dict[str, np.ndarray]] = {}
    for name, value in residuals.items():
        vector = np.nan_to_num(np.asarray(value).ravel().astype(np.complex128))
        vector = vector - vector.mean()
        left = vector
        for direction in basis:
            if direction.size == left.size:
                left = left - (direction.conj() @ left) * direction
        remainder[name] = {"frequency": left}
    if len(remainder) < 3:
        return {"why": "too few components to test what remains"}

    # DOES ANYTHING IN IT STAND ABOVE THE NOISE? Two corrections, both
    # found by a control that used pure noise as its input.
    #
    # First, the test is `significant`, not the null's z. Projecting the
    # key's basis out of the remainder leaves it in a SMALLER SPACE than a
    # freshly generated null of the same nominal size, so the remainder is
    # more spherical than the null by construction and the z runs hugely
    # NEGATIVE - measured -2347 on pure noise. The z is reported as a
    # diagnostic and no longer decides anything.
    #
    # Second, the test must be ONE-SIDED here. A departure makes a raw
    # remainder MORE asymmetric, so only a positive excursion is evidence;
    # a negative one means more spherical than random, which is not
    # structure. (The two-sided reading belongs to the nested matrix, where
    # a shared departure cancels in the differences and reads negative.)
    null = ie.surrogate_null(remainder, "frequency", draws=draws, seed=seed)
    fit = ie.ellipsoid(remainder, "frequency")
    stands = int(fit["significant"]) > 0

    corrected: Dict[str, np.ndarray] = {}
    taken = kept = 0.0
    if stands:
        # the directions the remainder itself carries, and ONLY those - no
        # `max(significant, 1)`, which forced a removal even where nothing
        # was significant and subtracted a tenth of pure noise
        names = fit["names"]
        stack = np.array([np.nan_to_num(remainder[n]["frequency"])
                          for n in names])
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        unit = np.divide(stack, norms, out=np.zeros_like(stack),
                         where=norms > 0)
        vectors = np.asarray(fit["axes"])[:, :int(fit["significant"])]
        target = vectors.conj().T @ unit
        keep = []
        for row in target:
            for earlier in keep:
                row = row - (earlier.conj() @ row) * earlier
            norm = float(np.linalg.norm(row))
            if norm > 0:
                keep.append(row / norm)
        for name in names:
            vector = np.asarray(remainder[name]["frequency"]).ravel()
            before = float(np.vdot(vector, vector).real)
            after = vector
            for direction in keep:
                after = after - (direction.conj() @ after) * direction
            corrected[name] = after
            taken += before - float(np.vdot(after, after).real)
            kept += float(np.vdot(after, after).real)
    else:
        for name, value in remainder.items():
            corrected[name] = value["frequency"]
            kept += float(np.vdot(value["frequency"],
                                  value["frequency"]).real)
    total = taken + kept
    return {
        "corrected": corrected,
        "subtracted": float(taken / total) if total > 0 else 0.0,
        "left": float(kept / total) if total > 0 else 0.0,
        "stood_above_the_null": bool(stands),
        "significant": int(fit["significant"]),
        "null_z": float(null.get("asymmetry_z", 0.0)),
        "why": ("the final synthetic side is zero, so the differential is "
                "the residual itself; only the part standing above the null "
                "is subtracted, because subtracting the rest is fitting "
                "noise"),
    }


def information_used(residuals: Dict[str, np.ndarray], frequency_hz,
                     draws: int = 24, seed: int = 0,
                     **kwargs) -> Dict[str, object]:
    """HAVE WE USED ALL THE INFORMATION AVAILABLE?

    Ethan: *"I think this collapses into a point where we have used all
    possible informaiton available to restore the video signal"* and *"That
    is testable using all of the tests we made to get to this point."* It is,
    and this is that test - the instruments built along the way, assembled
    into one verdict.

    Four conditions, all of which must hold, and each of which was built
    separately for its own reason:

    1. **the key is ordered** - `ordered_key` refuses an entry with no
       declared position, because a chain can only be inverted in the
       reverse of the order it was applied;
    2. **every entry is subtractable** - `subtractable`, because the
       transform works in the log domain and an entry that reaches zero has
       no logarithm to subtract;
    3. **nothing structured remains outside the key** - `split_residual`,
       whose middle part is the only thing both unreached and reachable;
    4. **what does remain sits at the null** - `surrogate_null`, generated by
       the same construction on structureless input, because an analytic
       null does not apply to a constructed ensemble.

    Conditions 3 and 4 are the substance and 1 and 2 are the preconditions
    for them to mean anything: an unordered or unsubtractable key can appear
    to leave nothing behind simply by being unable to remove anything
    properly in the first place.

    The verdict is deliberately conservative. `exhausted` is True only when
    all four hold, and `remaining` states what is still there to find.
    """
    from vhsdecode.models import information_extrapolation as ie

    entries = signatures(frequency_hz, **kwargs)
    try:
        order = ordered_key(entries)
        ordered = True
        order_note = f"{len(order)} entries, positions {order[0][0]} down to {order[-1][0]}"
    except ValueError as error:
        ordered, order = False, []
        order_note = str(error)
    unsubtractable = [name for name, value in entries.items()
                      if not subtractable(value)]

    split = split_residual(residuals, frequency_hz, draws=draws, seed=seed,
                           **kwargs)
    structured_left = bool(split.get("structured", 0.0) > 0.01)

    # what remains after the key, judged against a null built the same way
    at_the_null = not split.get("outside_stands_above_the_null", False)

    exhausted = (ordered and not unsubtractable
                 and not structured_left and at_the_null)
    return {
        "exhausted": bool(exhausted),
        "ordered": bool(ordered),
        "order_note": order_note,
        "unsubtractable": unsubtractable,
        "inside": split.get("inside", 0.0),
        "structured": split.get("structured", 0.0),
        "noise": split.get("noise", 0.0),
        "outside_z": split.get("outside_z", 0.0),
        "structured_remains": structured_left,
        "at_the_null": at_the_null,
        "remaining": ("nothing the present key can reach"
                      if exhausted else
                      "; ".join(filter(None, [
                          "" if ordered else "the key has no complete order",
                          "" if not unsubtractable else
                          f"{len(unsubtractable)} entries are not subtractable",
                          "" if not structured_left else
                          f"{100*split.get('structured', 0):.1f}% of the "
                          f"residual is structured and outside the key",
                      ]))),
        "why": ("ordered key, subtractable entries, nothing structured left "
                "outside, and what remains at the null"),
    }


def distinguishable(frequency_hz, **kwargs) -> Dict[str, object]:
    """How many of the modelled types this band can tell apart.

    The same question that gave 1.58 of 6 for the tape's magnetics and 3.79
    of 6 for propagation, asked of whatever set the caller assembles. The
    log magnitude is used for the amplitude part so that a gain is a
    constant rather than a scale, and the mean is removed from each because
    a constant is a level and not a shape.
    """
    from vhsdecode.models import information_extrapolation as ie

    shapes = {}
    for name, value in signatures(frequency_hz, **kwargs).items():
        value = np.asarray(value)
        magnitude = np.log(np.maximum(np.abs(value), 1e-12))
        phase = np.unwrap(np.angle(value)) if np.iscomplexobj(value) \
            else np.zeros_like(magnitude)
        vector = (magnitude - magnitude.mean()) + 1j * (phase - phase.mean())
        if np.linalg.norm(vector) > 0:
            shapes[name] = {"frequency": vector}
    fit = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    stack = np.array([
        np.concatenate([shapes[n]["frequency"].real,
                        shapes[n]["frequency"].imag]) for n in fit["names"]])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    values = np.linalg.svd(stack, compute_uv=False)
    share = values ** 2 / max(float((values ** 2).sum()), 1e-30)
    return {
        "names": fit["names"],
        "count": len(fit["names"]),
        "effective": float(1.0 / np.sum(share ** 2)) if share.size else 0.0,
        "singular_values": values,
        "condition": float(values[0] / max(values[-1], 1e-30))
        if values.size else 0.0,
        "ellipse": fit,
    }
