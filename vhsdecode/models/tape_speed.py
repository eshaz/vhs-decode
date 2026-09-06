"""The tape's speed against its expected speed, as a component.

Ethan: "Let's use the tape speed difference from its expected as another
component, this is what ties in the tapes mechanical model we built
earlier."

WHY IT IS A COMPONENT AND NOT A CALIBRATION. Speed enters the magnetic
model only through the wavelength, `lambda = v / f`, and every loss in
`head_model` is a function of a length over that wavelength. So a speed
error of `eps` multiplies every one of those ratios by `1 + eps` - it does
not shift the response, it RESCALES the frequency axis the losses live on.
That makes it a first-class departure with its own axis, not a constant to
be folded into a gain.

THE DEGENERACY, MEASURED, AND IT IS TOTAL ON THE FREQUENCY AXIS.
Over 0.5 to 6 MHz, in the log response:

    a +2% speed error against a -2% scaling of ALL magnetic lengths
        coherence 1.000000, worst difference 6.7e-16 nepers
    a +2% speed error against a +2% SPACING error alone
        coherence -0.999

The first is not an approximation - it is an identity. `exp(-2 pi d f / v)`
depends on `d/v`, `sinc(g f / v)` on `g/v`, the thickness term on `t/v`, so
scaling `v` and scaling every length together are the same operation
written twice. The second says the near-degeneracy survives even against
spacing alone, because over one decade of band the three losses are all
smooth monotone functions of the same ratio.

SO THE FREQUENCY AXIS CANNOT MEASURE THIS, AND THAT IS THE POINT. Speed is
observable on the TIME axis, directly and with no model at all: the line
period is the sync-to-sync spacing, and the drum rate is a line in the
envelope's own spectrum. That is what "ties in the mechanical model" - the
transport supplies a measurement on an axis where the magnetics are blind,
and pinning speed there removes a whole direction from the frequency
axis's null space.

THE TWO SPEEDS ARE DIFFERENT QUANTITIES and confusing them is the trap
this module exists to prevent:

    WRITING SPEED   head against tape, about 5.8 m/s, set almost entirely
                    by the DRUM, which is servo-locked to the field rate.
                    This is the one that sets wavelength.
    LONGITUDINAL    tape against the deck, 33.35 mm/s for SP, set by the
                    CAPSTAN. It contributes 0.57 per cent of the writing
                    speed, so a capstan error barely moves a wavelength -
                    what it moves is the track geometry and the running
                    time.

A deck whose capstan is one per cent fast has a one per cent tracking
error and a 0.006 per cent wavelength error. A deck whose drum is one per
cent fast has a one per cent wavelength error and no tracking error at
all. They are separate components with separate witnesses, and
`attribute` says which transport element can produce which.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

from vhsdecode.models import transport_model

__all__ = [
    "NOMINAL_LONGITUDINAL_M_S", "STATED_WRITING_M_S", "DRUM_DIAMETER_M",
    "expected_speeds", "from_line_period", "from_drum_line",
    "wavelength_scaling", "loss_departure", "degeneracy_with_lengths",
    "attribute", "component",
]

NOMINAL_LONGITUDINAL_M_S = {"SP": 33.35e-3, "LP": 16.67e-3, "EP": 11.12e-3}
"""VHS NTSC linear tape speeds. SP is the specified 33.35 mm/s; LP and EP
are it divided by two and three, which is how the format defines them."""

STATED_WRITING_M_S = 5.80
"""JVC VTG82063's stated head-to-tape speed. The geometry gives 5.8709
from a 62 mm drum at 29.97 rev/s plus the tape's contribution - 1.2 per
cent higher - and the stated figure is the one used, with the discrepancy
carried rather than resolved (see `expected_speeds`)."""

DRUM_DIAMETER_M = 0.0620


def expected_speeds(tape_speed: str = "SP",
                    field_rate_hz: float = 60000.0 / 1001.0,
                    drum_diameter_m: float = DRUM_DIAMETER_M
                    ) -> Dict[str, float]:
    """What the format says the two speeds are, and where they disagree.

    The drum rate is NOT a free parameter: one revolution lays two fields,
    so it is half the field rate exactly, and the servo holds it there.
    That is why the writing speed is the more trustworthy of the two - it
    is locked to a quantity the signal itself carries.

    ETHAN'S REFINEMENT, and it names the lock's reference: "Drum rate is
    servo locked to the head switching area record and playback." So the
    drum is not merely locked to the field rate in the abstract - the
    HEAD-SWITCH POSITION is the error signal, on both passes. Two things
    follow. A drum-rate departure is a SERVO error and not a mechanical
    drift, so it should be small and correlated with the switch position
    rather than with the reels. And because the same lock operates at
    record and at playback, a switch-position anomaly can have been
    written in as easily as read in - which is why SMPTE 3.6 permits the
    switch anywhere in a three-line window rather than fixing it.
    """
    drum_hz = transport_model.drum_rate_hz(field_rate_hz)
    circumference = math.pi * float(drum_diameter_m)
    head = circumference * drum_hz
    longitudinal = NOMINAL_LONGITUDINAL_M_S.get(str(tape_speed).upper(),
                                                NOMINAL_LONGITUDINAL_M_S["SP"])
    derived = head + longitudinal
    return {
        "drum_rate_hz": drum_hz,
        "drum_circumference_m": circumference,
        "head_speed_m_s": head,
        "longitudinal_m_s": longitudinal,
        "writing_derived_m_s": derived,
        "writing_stated_m_s": STATED_WRITING_M_S,
        "writing_m_s": STATED_WRITING_M_S,
        "spec_disagreement": derived / STATED_WRITING_M_S - 1.0,
        "longitudinal_share": longitudinal / derived,
    }


def from_line_period(measured_period_s, expected_period_s: float
                     ) -> Dict[str, object]:
    """THE TIME-AXIS WITNESS, and the one that breaks the degeneracy.

    The sync-to-sync spacing is the line period. A tape played faster than
    it was recorded delivers lines sooner, so the period is short by the
    same fraction the speed is long:

        eps = expected_period / measured_period - 1

    NOTE WHAT THIS DOES AND DOES NOT SEE. It measures the ratio of
    playback speed to RECORD speed, because the recorded line period is
    whatever the recording deck laid down. A source whose own sync
    generator ran fast is indistinguishable here from a playback deck
    running slow - `attribute` carries that, and only a second recording
    on the same deck separates them.

    IT IS ALSO NULLED BY THE TIME BASE CORRECTOR, which locks sync to
    sync. So this must be read from the linelocs BEFORE correction, or
    from what the corrector had to remove - never from the corrected
    output, where it is zero by construction. That is the same null-space
    trap the burst lock has.
    """
    period = np.asarray(measured_period_s, dtype=np.float64).ravel()
    period = period[np.isfinite(period) & (period > 0)]
    if not period.size:
        return {"epsilon": float("nan"), "lines": 0}
    expected = float(expected_period_s)
    ratio = expected / period - 1.0
    return {
        "epsilon": float(np.median(ratio)),
        "epsilon_rms": float(np.sqrt(np.mean(ratio ** 2))),
        "spread": float(np.std(ratio)),
        "lines": int(period.size),
        "measures": "playback speed against RECORD speed, not against spec",
        "read_before_the_tbc": True,
    }


def from_drum_line(measured_drum_hz: float,
                   field_rate_hz: float = 60000.0 / 1001.0
                   ) -> Dict[str, float]:
    """The envelope's own witness: the drum-rate line in its spectrum.

    Independent of the time base, because it is an AMPLITUDE modulation -
    the two heads' sensitivities differ, so the envelope carries a line at
    the drum rate whatever the timing does. Measured in this arc on both
    tapes.
    """
    expected = transport_model.drum_rate_hz(field_rate_hz)
    return {
        "expected_hz": expected,
        "measured_hz": float(measured_drum_hz),
        "epsilon": float(measured_drum_hz) / expected - 1.0,
        "axis": "amplitude - a line in the envelope, so the time base "
                "cannot null it",
    }


def wavelength_scaling(epsilon: float) -> float:
    """A speed error of `eps` scales every wavelength by `1 + eps`.

    One line, and it is the whole coupling between this component and the
    magnetics.
    """
    return 1.0 + float(epsilon)


def loss_departure(frequency_hz, epsilon: float, spacing_m: float = 0.05e-6,
                   gap_m: float = 0.30e-6, thickness_m: float = 0.20e-6,
                   writing_speed_m_s: float = STATED_WRITING_M_S
                   ) -> np.ndarray:
    """What a speed error does to the log response, in nepers.

    Computed as the difference of the three losses at the two speeds, so
    it carries the exact shape rather than a linearisation - which matters
    because the spacing term is an exponential and the gap term has a
    null.
    """
    f = np.asarray(frequency_hz, dtype=np.float64)

    def logged(speed):
        lam = speed / np.maximum(f, 1e-30)
        spacing = -2.0 * np.pi * float(spacing_m) / lam
        gap = np.log(np.maximum(np.abs(np.sinc(float(gap_m) / lam)), 1e-30))
        ratio = 2.0 * np.pi * float(thickness_m) / lam
        thickness = np.log(np.maximum((1.0 - np.exp(-ratio))
                                      / np.maximum(ratio, 1e-30), 1e-30))
        return spacing + gap + thickness

    return logged(writing_speed_m_s * wavelength_scaling(epsilon)) \
        - logged(writing_speed_m_s)


def degeneracy_with_lengths(frequency_hz, epsilon: float = 0.02,
                            **geometry) -> Dict[str, float]:
    """Is a speed error distinguishable from the head's geometry? Measured.

    The answer is no on this axis, and stating it as a number is what
    keeps a transform from claiming a speed correction it cannot witness.
    """
    f = np.asarray(frequency_hz, dtype=np.float64)
    spacing_m = geometry.get("spacing_m", 0.05e-6)
    gap_m = geometry.get("gap_m", 0.30e-6)
    thickness_m = geometry.get("thickness_m", 0.20e-6)
    speed = loss_departure(f, epsilon, spacing_m, gap_m, thickness_m)
    # the common scaling: shrink every length by the same factor the speed
    # grew, which is the operation the speed error is claimed to equal
    scale = 1.0 / (1.0 + epsilon)

    def logged(speed_m_s, d, g, t):
        lam = speed_m_s / np.maximum(f, 1e-30)
        ratio = 2.0 * np.pi * t / lam
        return (-2.0 * np.pi * d / lam
                + np.log(np.maximum(np.abs(np.sinc(g / lam)), 1e-30))
                + np.log(np.maximum((1.0 - np.exp(-ratio))
                                    / np.maximum(ratio, 1e-30), 1e-30)))

    base = logged(STATED_WRITING_M_S, spacing_m, gap_m, thickness_m)
    common = logged(STATED_WRITING_M_S, spacing_m * scale, gap_m * scale,
                    thickness_m * scale) - base
    spacing_only = logged(STATED_WRITING_M_S, spacing_m * (1.0 + epsilon),
                          gap_m, thickness_m) - base

    def coherence(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denominator = math.sqrt(float(a @ a) * float(b @ b))
        return float(a @ b / denominator) if denominator > 0 else float("nan")

    return {
        "against_common_scaling": coherence(speed, common),
        "worst_difference_nepers": float(np.max(np.abs(speed - common))),
        "against_spacing_alone": coherence(speed, spacing_only),
        "separable_on_frequency": False,
        "separable_on_time": True,
        "why": "exp(-2 pi d f / v), sinc(g f / v) and the thickness term "
               "all depend on a length over the speed, so scaling the "
               "speed and scaling every length are one operation. The time "
               "axis measures the speed directly and breaks it.",
    }


def attribute(epsilon: float, drum_epsilon: Optional[float] = None
              ) -> Dict[str, object]:
    """WHICH TRANSPORT ELEMENT can produce this error - the tie-in.

    `transport_model.TRANSPORT` already declares, per element, whether it
    acts on speed, on tension, or on both, and whether its geometry is
    certain. Only two elements act on SPEED - the capstan and the pinch
    roller - and only the drum acts on the writing speed. So a speed
    departure has at most three candidates and they are distinguishable by
    which witness moved:

        the LINE PERIOD moved but the DRUM LINE did not
            the capstan or pinch roller: longitudinal speed, tracking and
            running time, essentially no wavelength change
        the DRUM LINE moved
            the drum servo: wavelength changes with it
        BOTH moved together
            the source's own sync generator, or a playback rate applied
            to everything alike
    """
    speed_parts = [p["name"] for p in transport_model.TRANSPORT
                   if p["acts_on"] in ("speed", "both")]
    drum = [p for p in transport_model.TRANSPORT if p["kind"] == "drum"][0]
    moved = abs(float(epsilon)) > 1e-6
    drum_moved = (drum_epsilon is not None
                  and abs(float(drum_epsilon)) > 1e-6)
    if moved and drum_moved:
        cause = "both witnesses moved: the source's sync generator, or a " \
                "rate applied to everything alike"
    elif drum_moved:
        cause = "the drum servo - the writing speed and so every wavelength"
    elif moved:
        cause = "the capstan or pinch roller - longitudinal only, so " \
                "tracking and running time but almost no wavelength change"
    else:
        cause = "no departure"
    return {
        "candidates": speed_parts,
        "drum_is_certain": bool(drum["certain"]),
        "cause": cause,
        "longitudinal_share_of_writing": NOMINAL_LONGITUDINAL_M_S["SP"]
        / expected_speeds()["writing_derived_m_s"],
        "note": "only the drum's geometry is marked certain in "
                "transport_model, so a capstan-attributed error carries "
                "the capstan's own diameter uncertainty with it",
    }


def component() -> Dict[str, object]:
    """The component's declaration: ideal, witness, axes, and its limits.

    Shaped to what `information_extrapolation.Component` needs - the
    synthetic side, the criterion that identifies it, and the axes it is
    carried on - so it can be registered rather than described.
    """
    return {
        "name": "tape speed against its expected",
        "ideal": "the format's own figures: the drum at half the field "
                 "rate exactly, and 33.35 mm/s longitudinal for SP",
        "axes": ("time", "frequency"),
        "criterion":
            "the line period read BEFORE the time base correction (or what "
            "the corrector had to remove), and the drum-rate line in the "
            "envelope. The time axis is the one that identifies it: on the "
            "frequency axis a speed error is EXACTLY degenerate with a "
            "common scaling of every magnetic length - coherence 1.000000, "
            "worst difference 6.7e-16 nepers - and 0.999 collinear with "
            "spacing alone, so nothing spectral can witness it.",
        "stage": "the transport, which is upstream of both RF stages and "
                 "is a property of the deck rather than of the tape",
        "measures": "playback speed against RECORD speed. A source whose "
                    "sync generator ran fast is indistinguishable from a "
                    "playback deck running slow without a second recording "
                    "on the same deck.",
        "couples_to": "every magnetic loss, through lambda = v / f - which "
                      "is the whole reason it belongs in the key rather "
                      "than in a calibration",
        "null_space": "the time base corrector locks sync to sync, so it "
                      "nulls this by construction; read it before "
                      "correction or from the correction itself",
    }


# --------------------------------------------------------------------------
# The field's physical size on the tape, and the amplitude that gives it
# --------------------------------------------------------------------------

def field_geometry(tape_speed: str = "SP",
                   field_rate_hz: float = 60000.0 / 1001.0,
                   drum_diameter_m: float = DRUM_DIAMETER_M,
                   lines_per_field: float = 262.5) -> Dict[str, float]:
    """THE TOTAL SIZE OF ONE FIELD ON THE TAPE, in millimetres of track.

    Ethan: "if we can derive amplitude of the luma we also know the time
    base, which is expected to be the line rate frequency, and the total
    overall size of the field on the VHS tape itself."

    One head lays one field, and a field is half a drum revolution, so the
    track a field occupies is the writing speed times the field period:

        track = v / f_field = 97.95 mm     (relative speed 5.8709 m/s)
        per line = track / 262.5 = 0.373 mm

    THE 180 DEGREE CLOSURE IS ALGEBRAIC, NOT EVIDENCE, and saying so
    matters. Half a drum circumference plus the tape's creep during that
    half revolution comes to 97.9458 mm, which equals the track length
    above to four decimals - but it must, because the derived speed was
    itself `pi D f_drum + v_tape` and the field period is half the drum
    period. The two are the same statement. It is a consistency check on
    the arithmetic and not a confirmation of the geometry.

    WHAT IT DOES SETTLE IS THE 1.2 PER CENT SPEC DISAGREEMENT. The head's
    own motion is 5.8375 m/s and the relative speed including the tape's
    creep is 5.8709; JVC states 5.80. The stated figure is the HEAD-ONLY
    speed to two significant figures - 0.65 per cent away - and not the
    relative speed, which is 1.22 per cent away. Wavelength is set by the
    RELATIVE speed, so `lambda = v/f` should use 5.8709 and the stated
    figure should not be read as a measurement of it.
    """
    drum_hz = transport_model.drum_rate_hz(field_rate_hz)
    circumference = math.pi * float(drum_diameter_m)
    longitudinal = NOMINAL_LONGITUDINAL_M_S.get(str(tape_speed).upper(),
                                                NOMINAL_LONGITUDINAL_M_S["SP"])
    relative = circumference * drum_hz + longitudinal
    field_period = 1.0 / float(field_rate_hz)
    track = relative * field_period
    contact = 0.5 / drum_hz
    half_turn = 0.5 * circumference + longitudinal * contact
    return {
        "track_length_m": track,
        "track_per_line_m": track / max(float(lines_per_field), 1e-9),
        "field_period_s": field_period,
        "contact_time_s": contact,
        "half_revolution_m": half_turn,
        # THE TRACK IN UNITS OF DRUM CIRCUMFERENCE, which is NOT the
        # mechanical wrap angle and must not be read as one. The head
        # sweeps exactly 180 degrees of drum per field; the track it lays
        # is longer than 180 degrees of circumference because the tape
        # moved under it during the sweep. 181.03 degrees is that excess,
        # and it is the tape's creep expressed in the drum's angle.
        "track_in_circumference_degrees": 360.0 * track / circumference,
        "mechanical_sweep_degrees": 180.0,
        "relative_speed_m_s": relative,
        "head_only_m_s": circumference * drum_hz,
        "stated_m_s": STATED_WRITING_M_S,
        "stated_is_head_only_rounded": abs(
            circumference * drum_hz / STATED_WRITING_M_S - 1.0) < 0.01,
        "line_rate_hz": float(field_rate_hz) * float(lines_per_field),
    }


def time_base_from_amplitude(envelope_field_rate_hz: float,
                             lines_per_field: float = 262.5,
                             tape_speed: str = "SP") -> Dict[str, object]:
    """The chain Ethan names: AMPLITUDE -> time base -> line rate -> size.

    The luma's amplitude alone carries the transport's timing, without
    decoding a single sync pulse:

      * the HEAD SWITCH is an amplitude discontinuity twice per frame,
        because the two heads' sensitivities differ - so the field
        boundaries are in the envelope;
      * the DRUM RATE is a line in the envelope's spectrum for the same
        reason, and this arc has found it on both tapes.

    From the field rate the rest follows arithmetically: the line rate is
    the field rate times the lines a field carries (262.5 for 525/60, the
    half being what makes the interlace), and the track length is the
    writing speed times the field period.

    WHY THIS IS WORTH HAVING WHEN SYNC ALREADY GIVES THE LINE RATE: it is
    an INDEPENDENT witness, on a different axis, that the time base
    corrector cannot null. The corrector locks sync to sync, so a speed
    error is zero by construction in the corrected output; the envelope's
    drum line is an amplitude modulation and survives it. Two witnesses on
    two axes are what make the speed identifiable at all - see
    `degeneracy_with_lengths`, where the frequency axis is blind to it.

    THE LIMIT, stated so it is not assumed away: this gives the PLAYBACK
    transport's rate. It cannot see the record-side rate, because the
    envelope was written by a head whose own speed is not recoverable from
    what it wrote.
    """
    field_rate = float(envelope_field_rate_hz)
    geometry = field_geometry(tape_speed, field_rate,
                              lines_per_field=lines_per_field)
    expected = 60000.0 / 1001.0
    return {
        "field_rate_hz": field_rate,
        "line_rate_hz": field_rate * float(lines_per_field),
        "expected_line_rate_hz": expected * float(lines_per_field),
        "epsilon": field_rate / expected - 1.0,
        "track_length_m": geometry["track_length_m"],
        "track_per_line_m": geometry["track_per_line_m"],
        "witnesses": ("the head switch, an amplitude step twice per frame",
                      "the drum-rate line in the envelope's spectrum"),
        "survives_the_tbc": True,
        "measures": "the PLAYBACK transport's rate only - the record-side "
                    "rate is not recoverable from what the head wrote",
    }


# --------------------------------------------------------------------------
# The reels, from the specification's own length law
# --------------------------------------------------------------------------

HUB_RADIUS_M = 0.012
"""The reel hub, from `transport_model.TRANSPORT`'s stated 0.012 to 0.045 m
range. The lower bound is the hub itself, which is a fixed piece of the
cassette; the upper bound is a full pack and is what the length law below
replaces with a per-tape figure."""

NTSC_LENGTH_LAW = (2.2, 2.0)
"""L = 2.2 t + 2 metres for t minutes (JVC VTG82063 s.1.1.25-1, quoted as
`[2.2t + 2]` +3/-0). PAL/SECAM is 1.42 t + 2."""

TAPE_THICKNESS_M = {"T-30": 19e-6, "T-60": 19e-6, "T-90": 19e-6,
                    "T-120": 19e-6, "T-160": 15.6e-6}
"""SMPTE 32M-1998 table 1: 19.0 +1/-2 um for T-120 and below, and 15.6
+-0.5 um for T-160 - the long tape is THINNER, which is how it fits."""


def tape_length_m(minutes: float, system: str = "NTSC") -> float:
    """The specification's own length law."""
    slope, intercept = (NTSC_LENGTH_LAW if str(system).upper().startswith("N")
                        else (1.42, 2.0))
    return slope * float(minutes) + intercept


def reel_pack_radius_m(length_wound_m: float, thickness_m: float = 19e-6,
                       hub_radius_m: float = HUB_RADIUS_M) -> float:
    """The pack radius after winding a given length - an Archimedean spiral.

    The wound tape is an annulus of area `pi (r^2 - r0^2)` and of area
    `L t`, so `r = sqrt(r0^2 + L t / pi)`. Nothing fitted: the hub is a
    piece of the cassette and the length and thickness are specified.
    """
    return math.sqrt(float(hub_radius_m) ** 2
                     + max(float(length_wound_m), 0.0) * float(thickness_m)
                     / math.pi)


def reel_rate_hz(radius_m: float, tape_speed_m_s: float = 33.35e-3) -> float:
    """A reel turning at the tape speed: f = v / (2 pi r)."""
    return float(tape_speed_m_s) / (2.0 * math.pi * max(float(radius_m), 1e-9))


def reel_band_hz(tape_type: str = "T-120", system: str = "NTSC",
                 tape_speed_m_s: float = 33.35e-3) -> Dict[str, float]:
    """THE REEL'S FREQUENCY BAND, PER TAPE TYPE - Ethan's derivation.

    "Reel speed can be derived using the VHS specs against the different
    types of possible VHS tapes."

    It can, and it is sharper than a fixed radius range. The reel turns
    slowest at a full pack and fastest at the bare hub, so the band runs
    from `v/(2 pi r_full)` to `v/(2 pi r_hub)` - and `r_full` is set by
    the tape's own length and thickness, both specified:

        T-30    68 m, 19.0 um   full 23.56 mm   0.2253 - 0.4423 Hz
        T-60   134 m, 19.0 um   full 30.89 mm   0.1718 - 0.4423 Hz
        T-90   200 m, 19.0 um   full 36.79 mm   0.1443 - 0.4423 Hz
        T-120  266 m, 19.0 um   full 41.87 mm   0.1268 - 0.4423 Hz
        T-160  354 m, 15.6 um   full 43.61 mm   0.1217 - 0.4423 Hz

    The upper edge is common to every tape because it is the bare hub. The
    lower edge is the tape's signature, and the tree's fixed 12-45 mm
    range gives 0.1180 Hz - below every real tape's, so it admits a region
    no cassette can reach.
    """
    thickness = TAPE_THICKNESS_M.get(str(tape_type), 19e-6)
    minutes = float(str(tape_type).split("-")[-1])
    length = tape_length_m(minutes, system)
    full = reel_pack_radius_m(length, thickness)
    return {
        "tape_type": str(tape_type),
        "length_m": length,
        "thickness_m": thickness,
        "full_radius_m": full,
        "hub_radius_m": HUB_RADIUS_M,
        "slowest_hz": reel_rate_hz(full, tape_speed_m_s),
        "fastest_hz": reel_rate_hz(HUB_RADIUS_M, tape_speed_m_s),
    }


def position_from_reel_rate(rate_hz: float, tape_type: str = "T-120",
                            system: str = "NTSC",
                            tape_speed_m_s: float = 33.35e-3
                            ) -> Dict[str, object]:
    """WHERE ON THE TAPE WE ARE, from the reel's own rate.

    This is what makes the reel worth deriving rather than merely
    bounding. The rate gives the pack radius directly, `r = v/(2 pi f)`,
    and the radius gives the length wound by inverting the spiral. So a
    line in the envelope's spectrum is a POSITION measurement - the axis
    this arc's tape-position sweep had to construct by hand.

    Measured on home over 1024 fields the reel line sits at 0.293 Hz,
    which is a pack radius of 18.1 mm - near the hub, so near one end of
    the tape.
    """
    band = reel_band_hz(tape_type, system, tape_speed_m_s)
    radius = float(tape_speed_m_s) / (2.0 * math.pi * max(float(rate_hz), 1e-9))
    wound = max(math.pi * (radius ** 2 - HUB_RADIUS_M ** 2)
                / band["thickness_m"], 0.0)
    return {
        "rate_hz": float(rate_hz),
        "radius_m": radius,
        "length_wound_m": wound,
        "fraction": min(wound / max(band["length_m"], 1e-9), 1.0),
        "minutes_in": wound / max(float(tape_speed_m_s), 1e-9) / 60.0,
        "within_band": bool(band["slowest_hz"] <= float(rate_hz)
                            <= band["fastest_hz"]),
        "band": band,
    }


# --------------------------------------------------------------------------
# Tracking mismatch
# --------------------------------------------------------------------------

def tracking_mismatch(offset_m: float, track_width_m: float = 58e-6,
                      azimuth_deg: float = 6.0,
                      writing_speed_m_s: float = 5.8709,
                      frequency_hz=(0.629e6, 3.4e6, 4.4e6)
                      ) -> Dict[str, object]:
    """What a tracking error costs, and what it lets in.

    Ethan: "Variations between tape speed and head speed, and tracking
    miss-match will be components within this set of data."

    THERE IS NO GUARD BAND. SMPTE table 2 gives pitch equal to width and
    the JVC glossary says so outright - "a non-recording band between
    tracks to prevent interference from adjacent tracks. This is not
    provided for VHS." So a tracking offset does two things at once and
    they are not the same component:

      LOSS       the head overlaps its own track by less, and the wanted
                 signal falls in proportion;
      CROSSTALK  the uncovered part sits over the NEIGHBOUR, whose signal
                 arrives suppressed only by the azimuth.

    AND THE AZIMUTH PROTECTS THE TWO BANDS VERY DIFFERENTLY. The two heads
    differ by 2 alpha = 12 degrees, so the gap sweeps `w tan(2 alpha)` =
    12.33 um across a 58 um track, and the suppression is
    `sinc(across / lambda)`:

        colour-under  9.33 um    13.8 dB
        luma sync tip 1.73 um    34.5 dB
        luma white    1.33 um    32.6 dB

    TWENTY DECIBELS LESS PROTECTION FOR THE CHROMA - the same eleven-fold
    wavelength lever that makes the two bands a useful pair for measuring
    the head, working against us here. It is why VHS chroma crosstalk is
    the visible problem it is, and it follows from the specification's own
    +-6 degrees and 58 um with nothing fitted.
    """
    offset = abs(float(offset_m))
    width = float(track_width_m)
    overlap = max(width - offset, 0.0) / width
    across = width * math.tan(math.radians(2.0 * float(azimuth_deg)))
    grid = np.atleast_1d(np.asarray(frequency_hz, dtype=np.float64))
    wavelength = float(writing_speed_m_s) / np.maximum(grid, 1e-9)
    suppression = np.abs(np.sinc(across / wavelength))
    leaked = min(offset / width, 1.0)
    return {
        "offset_m": offset,
        "overlap_fraction": overlap,
        "loss_db": -20.0 * math.log10(max(overlap, 1e-9)),
        "frequency_hz": grid,
        "azimuth_suppression_db": -20.0 * np.log10(
            np.maximum(suppression, 1e-12)),
        "crosstalk_db": (20.0 * math.log10(max(leaked, 1e-12))
                         + 20.0 * np.log10(np.maximum(suppression, 1e-12))),
        "across_track_m": across,
        "no_guard_band": True,
        "why": "pitch equals width in SMPTE table 2, so the uncovered part "
               "of the head sits over the neighbour and the azimuth is the "
               "only thing between them",
    }
