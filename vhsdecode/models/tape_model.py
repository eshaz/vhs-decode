"""The tape itself: the medium between the two machines.

The head model describes the magnetics of a head, the transport model the
mechanics of a deck, the filter model its electronics. All three describe a
MACHINE. The tape is not a machine - it is written by one and read by the
other, so its parameters belong to neither and are carried here.

Each variation below is listed with the one thing that matters for using
it: WHAT MEASUREMENT SEES IT. A parameter that changes nothing observable
cannot be fitted, and a parameter two measurements see identically cannot
be separated from its twin. Both cases are marked, because the failure mode
in this project has repeatedly been fitting something the data cannot
determine and reading the result as physics.

The tape enters the chain twice, and asymmetrically. At RECORD its
coercivity sets what magnetisation a given head current produces; at
PLAYBACK its remanence and surface set what flux the head recovers. A
change in the tape between those two moments - years of storage, a
temperature swing, oxide shed - therefore breaks the symmetry, and that
asymmetry is the only way some of these can be told apart at all.
"""

from typing import Dict, List, Optional

import numpy as np


# Which measurement witnesses each variation. These are the instruments this
# project actually has, named so a parameter cannot be introduced without
# saying how it would be seen.
WITNESSES = {
    "envelope_level": "the RF envelope's overall level",
    "envelope_tilt": "the RF envelope's slope across the carrier band",
    "envelope_events": "brief collapses of the RF envelope",
    "time_base": "the per-line and per-field timing",
    "azimuth": "the head difference's frequency dependence",
    "noise_floor": "the residual noise once every component has converged",
    "none": "nothing this project measures",
}


# Every variation a tape can carry, with its parameter, its physical effect,
# and the witness that would see it. `separable_from` names any parameter it
# is confounded with - a pair with the same witness and the same frequency
# dependence cannot be told apart by that witness alone.
TAPE_VARIATIONS: List[Dict[str, object]] = [
    {
        "name": "coating thickness",
        "parameter": "thickness_m",
        # NOT the tape's physical thickness, which the guide gives as
        # 19 +/- 2 um for the whole tape including its base. This is the
        # EFFECTIVE RECORDING DEPTH: how far into the coating the FM
        # carrier's short wavelengths actually penetrate, a small fraction
        # of the coating and far smaller again than the tape.
        "typical": 0.20e-6,
        "spec": "tape 19 +/- 2 um overall (JVC VTG82063 s.1) - this "
                "parameter is the recording depth, not that",
        "effect": "thickness loss (1 - exp(-x))/x with x = 2 pi delta / lambda: "
                  "only the top of the coating contributes at short wavelengths",
        "witness": "envelope_tilt",
        "confounded_with": ("spacing_m",),
        "note": "a TAPE property, though it sits in the head model's loss "
                "stack because that is where the loss acts",
    },
    {
        "name": "coercivity",
        "parameter": "coercivity_a_m",
        # JVC VTG82063 s.1: "600 oersted class (nominal)".
        # 600 Oe x 1000/(4 pi) = 47.7 kA/m
        "typical": 600.0 * 1000.0 / (4.0 * np.pi),
        "spec": "600 oersted class (JVC VTG82063 s.1)",
        "effect": "sets what magnetisation a given record current produces, so "
                  "it moves the RECORD side only. Falls as the tape warms, "
                  "which is why a warm tape behaves like an over-driven one",
        "witness": "envelope_tilt",
        "confounded_with": ("record_current",),
        "note": "record-side: it cannot be separated from record current by a "
                "playback measurement, only by knowing one of the two",
    },
    {
        "name": "remanence",
        "parameter": "remanence_t",
        "typical": 0.15,
        "effect": "the magnetisation the tape retains, so it scales the "
                  "recovered flux at every wavelength alike",
        "witness": "envelope_level",
        "confounded_with": ("head_efficiency", "preamp_gain"),
        "note": "a PURE GAIN, and the head fit removes an overall constant, "
                "so this is invisible to it by construction",
    },
    {
        "name": "surface roughness and debris",
        "parameter": "spacing_m",
        "typical": 0.05e-6,
        "effect": "adds to the head-to-tape separation, so it enters Wallace's "
                  "law exactly as head spacing does",
        "witness": "envelope_tilt",
        "confounded_with": ("thickness_m",),
        "note": "indistinguishable from the head's own spacing by any "
                "measurement on one tape: the two add",
    },
    {
        "name": "oxide shed and dropouts",
        "parameter": "dropout_rate_per_second",
        "typical": 1.0,
        "effect": "brief total loss of contact - the envelope collapses and "
                  "the demodulator has nothing to lock to",
        "witness": "envelope_events",
        "confounded_with": (),
        "note": "the one variation that is EVENTS rather than a parameter, so "
                "it is counted and located, never fitted as a level",
    },
    {
        "name": "binder degradation",
        "parameter": "shed_severity",
        "typical": 0.0,
        "effect": "progressive loss of surface: rising spacing AND rising "
                  "dropout rate together, worsening along a tape as the "
                  "heads load with shed oxide",
        "witness": "envelope_tilt",
        "confounded_with": ("spacing_m",),
        "note": "told apart from plain spacing by its TREND: sticky shed grows "
                "along a pass and recovers after cleaning, a fixed spacing "
                "does neither. Needs a long capture to see",
    },
    {
        "name": "mistracking",
        "parameter": "track_offset_m",
        "typical": 0.0,
        "effect": "the head reads partly off its track and partly onto the "
                  "neighbour, which was written at the OPPOSITE azimuth: the "
                  "wanted signal falls and an azimuth-suppressed crosstalk "
                  "rises with it",
        "witness": "azimuth",
        "confounded_with": (),
        "note": "distinguishable because it is the only variation that changes "
                "the two heads in OPPOSITE directions, the format's azimuth "
                "being +/- alternating",
    },
    {
        "name": "stretch and deformation",
        "parameter": "stretch_fraction",
        "typical": 0.0,
        "effect": "the track's geometry no longer matches the head's sweep, so "
                  "timing drifts along the line and the head leaves the track "
                  "toward one end of its sweep",
        "witness": "time_base",
        "confounded_with": ("transport_speed",),
        "note": "separable from a transport fault by its SHAPE: stretch is "
                "fixed to the tape and repeats at the same place on replay, a "
                "transport fault is fixed to the machine and does not",
    },
    {
        "name": "print-through",
        "parameter": "print_through_db",
        "typical": -55.0,
        "effect": "the adjacent wrap's magnetisation copied faintly into this "
                  "one during storage",
        "witness": "noise_floor",
        "confounded_with": (),
        "note": "far below the FM threshold on a video track: it matters on "
                "linear audio, and is listed for completeness rather than "
                "because this chain can see it",
    },
]


def variations() -> List[Dict[str, object]]:
    return [dict(entry) for entry in TAPE_VARIATIONS]


def by_witness() -> Dict[str, List[str]]:
    """Which variations each instrument can see. Two variations sharing a
    witness AND a frequency dependence cannot be separated by it."""
    out: Dict[str, List[str]] = {}
    for entry in TAPE_VARIATIONS:
        out.setdefault(str(entry["witness"]), []).append(str(entry["name"]))
    return out


def confounded() -> List[Dict[str, object]]:
    """The pairs a single measurement cannot separate, stated up front.

    This is the list that stops a fitted number being reported as physics:
    where two parameters add into the same observable with the same shape,
    a fit apportions them arbitrarily and the split means nothing."""
    out = []
    for entry in TAPE_VARIATIONS:
        for other in entry.get("confounded_with", ()):
            out.append({"parameter": entry["parameter"], "with": other,
                        "witness": entry["witness"], "name": entry["name"]})
    return out


def separable(first: str, second: str) -> bool:
    """Whether two named variations can be told apart by the instruments
    this project has."""
    index = {str(e["name"]): e for e in TAPE_VARIATIONS}
    a, b = index.get(first), index.get(second)
    if a is None or b is None:
        return True
    if a["witness"] != b["witness"]:
        return True
    return (b["parameter"] not in a.get("confounded_with", ())
            and a["parameter"] not in b.get("confounded_with", ()))


def tape_head_spacing(tape_spacing_m: float, head_spacing_m: float) -> float:
    """The spacing Wallace's law actually sees: the tape's surface and the
    head's own separation ADD. No measurement on a single tape can split
    them, which is why the head model's fitted figure is an EFFECTIVE
    spacing for the whole path and not a head's clearance."""
    return float(tape_spacing_m) + float(head_spacing_m)


# --------------------------------------------------------------------------
# the tape's magnetic limits - which bind BEFORE the capture's
# --------------------------------------------------------------------------

# The order matters and it is not the order the chain is written in. The RF
# capture's arithmetic floor is a hard bound, but it is not the BINDING one:
# a magnetic tape's own noise sits far above an 8-bit converter's
# quantization, so what can be recovered is set by the tape long before the
# capture has any say. The tape's limits are therefore removed first, and
# only then is it worth asking what the capture adds.
#
# These are FREQUENCY and PHASE limits, and barely TIME limits at all. A
# tape's magnetics decide which wavelengths survive and what phase they
# carry; they say nothing about when a line arrives, which is the
# transport's business. So the tape's bound applies to the frequency axis
# and the phase that goes with it, and the time axis is left to the
# mechanics.
PARTICLE_VOLUME_M3 = 1.0e-21          # a typical gamma-ferric oxide particle
PACKING_FRACTION = 0.4                # of the coating that is particle


def magnetic_limits(writing_speed_m_s: float, track_width_m: float,
                    recording_depth_m: float,
                    particle_volume_m3: float = PARTICLE_VOLUME_M3,
                    packing: float = PACKING_FRACTION) -> Dict[str, float]:
    """What the tape's own magnetics allow, before any converter is asked.

    PARTICLE NOISE. A head reads a finite volume of coating, holding a
    finite number of particles, each contributing a randomly oriented
    moment. The signal adds coherently and the noise as a square root, so
    the tape's own signal-to-noise ratio is set by how many particles are
    in the volume - and that volume shrinks as the wavelength does, which
    is why tape noise rises with frequency while an ADC's does not.

    SELF-DEMAGNETISATION. Short wavelengths demagnetise themselves: the
    poles of a transition sit close enough to oppose it. This sets the
    shortest wavelength the tape can hold at all, and therefore a frequency
    above which nothing was ever recorded to recover.

    Both are FREQUENCY limits. Neither says anything about time."""
    speed = float(writing_speed_m_s)
    width = float(track_width_m)
    depth = float(recording_depth_m)

    def volume_at(frequency_hz):
        wavelength = speed / max(float(frequency_hz), 1e-9)
        # the volume one wavelength of track occupies, to the recording depth
        return wavelength * width * depth

    def particles_at(frequency_hz):
        return (volume_at(frequency_hz) * float(packing)
                / max(float(particle_volume_m3), 1e-30))

    def snr_db_at(frequency_hz):
        count = particles_at(frequency_hz)
        return float(10.0 * np.log10(max(count, 1e-30)))

    # self-demagnetisation: a transition cannot be narrower than about the
    # recording depth, so the shortest wavelength is a few times that
    shortest_wavelength = 2.0 * np.pi * depth
    return {
        "writing_speed_m_s": speed,
        "shortest_wavelength_m": shortest_wavelength,
        "highest_frequency_hz": speed / max(shortest_wavelength, 1e-12),
        "particles_per_wavelength": particles_at,
        "snr_db_at": snr_db_at,
        "axes": "frequency and phase; not time",
    }


def binding_limit(tape_snr_db: float, capture_snr_db: float
                  ) -> Dict[str, object]:
    """Which limit actually binds, the tape's or the capture's.

    Two floors in series are not two constraints: the higher one decides,
    and the lower one is irrelevant until the higher is removed. Stating
    which binds is what stops effort going into the wrong one - refining
    against a converter's arithmetic floor while the tape's own noise sits
    twenty decibels above it accomplishes nothing."""
    tape = float(tape_snr_db)
    capture = float(capture_snr_db)
    binds = "tape" if tape < capture else "capture"
    return {
        "tape_snr_db": tape,
        "capture_snr_db": capture,
        "binds": binds,
        "headroom_db": abs(capture - tape),
        "why": ("the tape's magnetics are the limit; the capture's "
                "arithmetic floor is %.1f dB below it and cannot be reached "
                "until the tape's is removed" % abs(capture - tape)
                if binds == "tape" else
                "the capture is the limit here, which is unusual and worth "
                "checking before acting on"),
    }

