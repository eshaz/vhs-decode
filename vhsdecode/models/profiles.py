"""WHICH MODEL APPLIES: the deck, the source, and the television system.

Ethan, 2026-09-05:

  *"I think do a VCR profile specific to your tape deck, as an option,
  otherwise the ideally calibrated VCR. And a profile for source type, like
  camera, or home vcr, for more refined approach to modeling the fully
  calibrated VCR. Additionally, I will need models for NTSC television, or
  NTSC composite, and same for PAL."*

THREE INDEPENDENT CHOICES, and the point of separating them is that they
fail differently. The SYSTEM is a specification and is exact. The SOURCE
decides which stages of the chain exist at all - a camera never passes
through a tuner, so a tuner's clip and a broadcast ghost are not small on
a camera tape, they are ABSENT, and a model that carries them anyway will
fit them to noise. The DECK is the only one of the three that is measured
rather than specified, and it defaults to the ideal: every parameter at
its specified nominal with the specification's own tolerance as its error
bar, so a decode with no deck measurement is honest rather than wrong.

WHAT A PROFILE IS FOR. It answers three questions before any fitting
starts: which chain entries are PRESENT (`chain_entries`), which
quantities are KNOWN CONSTANTS and may not be given a dimension
(`running_differential.constants_for`, extended here per source), and what
each free parameter's specified nominal and tolerance are. The first
prunes the model, the second prunes it further, and the third supplies the
prior.

THE SOURCE IS THE STRONGEST PRUNE, and this arc has already paid for
getting it wrong once: the tuner clip site was added as a second clipping
position because countdown is an off-air recording, and it is separable
from the recorder's own input clip only because negative modulation puts
the sync tip at peak carrier. On a camera or composite tape that site does
not exist, and admitting it there would let it absorb the recorder's clip.

WHAT IS SPECIFIED AND WHAT IS MEASURED, kept apart on purpose. Every
number below is either cited to a standard, taken from a module that cites
one, or labelled as a measurement of one particular deck with the date and
the instrument. Nothing here is a fitted convenience.
"""

from typing import Dict, List, Optional, Sequence

from vhsdecode.models import running_differential

# --------------------------------------------------------------------------
# The television systems: specifications, exact by definition
# --------------------------------------------------------------------------

SYSTEMS: Dict[str, Dict[str, object]] = {
    "NTSC-M": {
        "lines": 525,
        "field_rate_hz": 60.0 / 1.001,
        "line_rate_hz": 525.0 * 30.0 / 1.001,
        "subcarrier_cycles_per_line": (455, 2),
        "colour_frame_fields": 4,
        "burst_cycles": 9,
        "cite": ("ITU-R BT.1700 (525-line NTSC section) and SMPTE 170M; the "
                 "colour-frame sequence and the 227.5 cycles a line are "
                 "definitions, not measurements"),
    },
    "PAL-B/G/I": {
        "lines": 625,
        "field_rate_hz": 50.0,
        "line_rate_hz": 15625.0,
        "subcarrier_cycles_per_line": (1135, 4),
        "colour_frame_fields": 8,
        "burst_cycles": 10,
        "burst_phase_deg": (135.0, -135.0),
        "cite": ("ITU-R BT.1700 (625-line PAL section), items 10f and the "
                 "burst-blanking sequence; the eight-field sequence follows "
                 "from the quarter-line offset and the PAL switch"),
    },
}

# --------------------------------------------------------------------------
# The sources: what the signal passed through before it reached the recorder
# --------------------------------------------------------------------------
#
# `chain` names the entries of `interference.full_chain` that EXIST for this
# source. An entry absent here is not merely small: it must not be offered
# to the key at all.

SOURCES: Dict[str, Dict[str, object]] = {
    "television": {
        "what": "off air through a tuner, the countdown tape's own history",
        "chain": ["vestigial sideband", "group delay", "sound trap", "echo",
                  "multipath", "source agc band",
                  "tuner clipping", "tuner clipping (sync tip)",
                  "tuner clipping level dependence", "waveform distortion",
                  "waveform distortion (short time)",
                  "waveform distortion (line time)",
                  "waveform distortion (field time)",
                  "waveform distortion (long time)"],
        "constants": {
            "vision carrier offset": (
                "the channel's carrier is fixed by the broadcast plan; the "
                "tuner's local oscillator error is a per-session constant, "
                "not a per-field one"),
        },
        "multipath": (
            "a television signal reaches the aerial by more than one path, "
            "so the channel is a tapped delay line: the direct path plus a "
            "reflection at every delay the terrain supplies. "
            "`models/multipath.py` carries it, fitted as complex taps on "
            "the log departure because a delayed copy moves magnitude and "
            "phase together. MEASURED AND NOT CONFIRMED: fitted on the "
            "first half of countdown's fields and judged on the second it "
            "collapses from 56 per cent to 17 with its phase going "
            "negative, while the two tapes that never passed a transmitter "
            "hold out at 52 and 38 - so the structure a tapped delay "
            "describes here is not in the air. The entry stays because the "
            "physics belongs to this source and a better capture may show "
            "it; it is declared unconfirmed rather than quietly dropped."),
        "notes": ("negative modulation puts the sync tip at peak carrier, so "
                  "a tuner overload clips the TIP while the recorder's own "
                  "input clip is a dark clip 40 per cent above it (SMPTE 32M "
                  "clause 3.9.1.1.3) - which is what makes the two sites "
                  "separable at all"),
    },
    "composite": {
        "what": "a line input: a generator, a disc player, another deck",
        "chain": ["group delay", "waveform distortion",
                  "waveform distortion (short time)",
                  "waveform distortion (line time)"],
        "constants": {
            "source time base": (
                "a studio generator is genlocked and crystal-referenced; "
                "measured on the TSG-130A captures at 0 +/- 2 degrees of "
                "SC/H with a drift under 0.02 degrees a field, so the "
                "source's own time base is a constant and every departure "
                "belongs to the recorder or the tape"),
        },
        "notes": ("no tuner, so NO tuner clip, no vestigial sideband, no "
                  "sound trap and no broadcast ghost; the pnb tape is this"),
    },
    "camera": {
        "what": "a camera recording direct to tape, no transmission at all",
        "chain": ["waveform distortion", "waveform distortion (short time)"],
        "constants": {
            "source time base": (
                "the camera's own sync generator drives the recorder, so "
                "there is no time-base difference between source and "
                "recorder to model"),
        },
        "notes": ("the shortest chain there is: everything between the "
                  "camera's encoder and the tape is inside one machine, so "
                  "the only pre-record terms are that machine's own"),
    },
    "home vcr": {
        "what": "a recording made from another VCR's output",
        "chain": ["group delay", "waveform distortion",
                  "waveform distortion (short time)",
                  "waveform distortion (line time)",
                  "waveform distortion (field time)",
                  "waveform distortion (long time)",
                  "vcr agc", "vcr agc level response"],
        "constants": {},
        "notes": ("the playing deck's whole chain is upstream of the "
                  "recording deck's, so its time base is NOT a constant - a "
                  "tape's time base is the one thing a second generation "
                  "cannot inherit clean, and both decks' transports appear"),
    },
}

# --------------------------------------------------------------------------
# The decks: the ideal one is specified, a named one is measured
# --------------------------------------------------------------------------

IDEAL_DECK = {
    "name": "ideally calibrated VCR",
    "measured": False,
    "cite": "SMPTE 32M-2004 and JVC VTG82063; every value its specified nominal",
    "parameters": {
        # (nominal, tolerance, unit, citation)
        "carrier_sync_tip_hz": (3.4e6, 0.1e6, "Hz",
                                "SMPTE 32M-2004 clause 3.9.1.1.4"),
        "carrier_peak_white_hz": (4.4e6, 0.1e6, "Hz",
                                  "SMPTE 32M-2004 clause 3.9.1.1.4"),
        "deviation_hz": (1.0e6, 0.1e6, "Hz",
                         "SMPTE 32M-2004 clause 3.9.1.1.4"),
        "colour_under_hz": (629.371e3, None, "Hz",
                            "SMPTE 32M-2004 clause 3.9.2, 40 x f_H, exact"),
        "white_clip_percent": (160.0, None, "% of sync-to-white",
                               "SMPTE 32M-2004 clause 3.9.1.1.3, 155 min "
                               "200 max"),
        "dark_clip_percent": (40.0, 10.0, "% of sync-to-white",
                              "SMPTE 32M-2004 clause 3.9.1.1.3"),
        "writing_speed_m_s": (5.80, None, "m/s",
                              "SMPTE 32M-2004 clause 3.1.4, stated nominal"),
        "track_width_m": (58e-6, None, "m",
                          "SMPTE 32M-2004 table, video track width 0.058 mm"),
        "azimuth_deg": (6.0, None, "degrees",
                        "SMPTE 32M-2004 clause 3.3, +6 and -6"),
        "drum_rate_hz": (30.0 / 1.001, None, "Hz",
                         "one revolution lays two fields; derived from the "
                         "field rate, not measured"),
    },
}

DECKS: Dict[str, Dict[str, object]] = {
    "ideal": IDEAL_DECK,
    "SLV-778HF": {
        "name": "Sony SLV-778HF",
        "measured": True,
        "cite": ("the deck that played every capture in "
                 "/testdata/test_patterns/vhs/; schematic in the same "
                 "directory"),
        "parameters": dict(IDEAL_DECK["parameters"]),
        "measurements": {
            "head_to_tape_separation_m": (
                0.220e-6, 0.052e-6, "m",
                "measured 2026-09-05 from the record and playback taps of "
                "eleven patterns, `models/depth_split.py`; absorbs the deck "
                "equalisation and the record transition, so it is an "
                "effective separation and not a geometric one"),
            "chroma_read_depth_m": (
                1.17e-6, 0.33e-6, "m",
                "same measurement, SP; 1.87 +/- 0.42 at EP"),
        },
    },
}


def profile(system: str = "NTSC-M", source: str = "television",
            deck: str = "ideal") -> Dict[str, object]:
    """Compose the three choices into one model description.

    Returns the system's exact numbers, the chain entries that exist for
    this source, the constants that may not become components, and the
    deck's parameters with their tolerances - everything a fit needs to
    know before it starts, and nothing it should discover by fitting.
    """
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; known: {sorted(SYSTEMS)}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {sorted(SOURCES)}")
    if deck not in DECKS:
        raise ValueError(f"unknown deck {deck!r}; known: {sorted(DECKS)}")
    constants = dict(running_differential.constants_for("VHS"))
    constants.update(SOURCES[source].get("constants", {}))
    entry = DECKS[deck]
    return {
        "system": {"name": system, **SYSTEMS[system]},
        "source": {"name": source, **SOURCES[source]},
        "deck": {"name": entry["name"], "measured": entry["measured"],
                 "cite": entry["cite"],
                 "parameters": entry["parameters"],
                 "measurements": entry.get("measurements", {})},
        "chain_entries": list(SOURCES[source]["chain"]),
        "constants": constants,
        "why": ("the system is exact, the source decides which stages exist "
                "at all, and the deck is the only measured one of the three "
                "- so an unmeasured deck falls back to the specification "
                "rather than to a guess"),
    }


def absent_entries(source: str) -> List[str]:
    """Chain entries that do NOT exist for this source and must not be
    offered to the key.

    The pre-record part of the chain is positions 0 to 8; anything there
    that this source does not have is absent, not small.
    """
    from vhsdecode.models import interference

    chain = interference.full_chain(strict=False)
    present = set(SOURCES[source]["chain"])
    return sorted(name for name, position in chain.items()
                  if position <= 8 and name not in present
                  and name != "specified test signal")


def for_capture(name: str) -> Dict[str, object]:
    """The profile of a capture this arc actually uses, by its own history.

    countdown is off air (Ethan: *"cd = countdown.flac was an over-the-air
    recording"*), home is a home recording of unknown origin, and the
    zaroff patterns came from a Tektronix TSG-130A generator into the
    SLV-778HF over composite.
    """
    known = {
        "cd": ("NTSC-M", "television", "ideal"),
        "countdown": ("NTSC-M", "television", "ideal"),
        "home": ("NTSC-M", "home vcr", "ideal"),
        "pnb": ("NTSC-M", "composite", "SLV-778HF"),
        "zaroff": ("NTSC-M", "composite", "SLV-778HF"),
    }
    for key, (system, source, deck) in known.items():
        if key in name.lower():
            return profile(system, source, deck)
    raise ValueError(f"no recorded history for {name!r}; name the system, "
                     f"source and deck explicitly rather than guessing")
