"""Aligning two radio-frequency captures of one tape pass.

Ethan, 2026-09-06: *"Additionally, I can do the same with the hifi tracks
where I can use the residual carriers in the video track to synchronize a
separate hifi rf capture with the video rf capture."* And, on what the pair
is eventually for: *"if we detect hifi, that needs to be matched the same
way we are for all the other components and subtracted out. Eventually I
have another project that I will be combining into this process that
decodes the hifi. In a later session we will merge these two projects to
use both RF captures as a complex pair video and hifi."*

THE ROUTE HE NAMES IS BOUNDED CLOSED, AND THIS FILE DOES NOT RE-OPEN IT.
`hifi_carriers.detection_bound` looked for the audio carriers inside the
video-head radio frequency and found none. The bound is restated in
`carrier_route_bound` below, on the only records that could carry the
signal, so that nobody proposes the synchronisation-by-residual-carrier
route again from scratch. What follows is the answer to the question the
route was FOR, which is how to put a separate audio capture on the same
time axis as a video one, using things that are demonstrably shared.

TWO ALIGNMENTS, AND CONFUSING THEM IS THE FIRST TRAP.

  THE TEMPORAL alignment answers "which audio sample was taken at the same
  INSTANT as this video sample". It is what a shared time base needs, and
  it is what the eventual complex pair needs, because a fold may only pair
  two measurements of the same moment.

  THE TAPE-POSITION alignment answers "which audio sample read the same
  POINT OF TAPE as this video sample". It is what lip-sync needs, and what
  attributing a defect to a place on the tape needs.

They are not the same number and they differ by a mechanical constant,
because the audio and video heads are at different angular positions on
one drum and therefore touch different tape at any one instant. SMPTE
32M-2004 clause 5.1: *"Separate FM audio heads shall be mounted on the
scanning drum for this purpose"*, and 5.3: *"The FM audio signal shall be
recorded prior to the video signal which is recorded at the same location
on the tape."*

THE SPECIFICATION FIXES THAT THE CONSTANT EXISTS AND REFUSES TO FIX ITS
VALUE, and this is the central result of the file. SMPTE 32M table 5,
"Recording time sequence", permits

    SP   audio recorded 0 to 2 fields earlier than the associated video
    LP   1/3 to 2-1/3 fields earlier
    EP   1-1/3 to 3-1/3 fields earlier

Every one of those windows is TWO FIELD PERIODS WIDE, and two field
periods is exactly one revolution of the drum, because the drum carries
two video heads and turns once per frame. So the standard permits the
audio head pair to sit at ANY angle whatever relative to the video pair:
the permitted range is the whole circle. `offset_is_not_specified` does
that arithmetic. Nothing can be predicted from the format; the offset is a
constant OF THE DECK and has to be measured once, per deck and per tape
speed - per speed because a deck with separate SP and EP video heads
mounts them at different angles, which is what makes the three windows in
table 5 differ at all (`head_pair_placement` derives those angles from the
window centres and labels the derivation).

WHAT IS ACTUALLY SHARED, in the order it is worth using.

  1. THE DRUM. Both captures are of one rotating drum, so the head-switch
     event in each is the same rotation seen twice. This is the strong
     method and `align_by_switches` is it. Measured on the video side at
     365 ns an event on the deck's own tape, against an instrument floor
     of 100 to 250 ns, with the residual WHITE - so it averages down with
     no floor in sight, and thirty seconds predicts 12.2 ns. Three things that were
     assumptions when this file was started are now read rather than
     assumed: that the audio path switches heads at all (it does -
     `SWITCHING_TOPOLOGY`, IC340's two chains meeting at a changeover
     switch); that both switches are timed from one drum tachometer by one
     controller against one crystal (they are, and the format's own guide
     says the audio switcher is drum-derived generally -
     `JVC_CORROBORATION`); and what the audio marker looks like - it is a
     step in the CARRIER'S PHASE, which is a better marker than the video
     side's amplitude step and cannot be hidden, because an audio track has
     no blanking to hide it in (`afm_switch_marker`).
  2. THE TAPE. Both heads cross the same physical defects, so a dropout is
     a mark at a fixed TAPE POSITION common to both. This is MEASURED here
     between the two video heads and it works: 159 coincidences against a
     uniform-random null of 0.20 +- 0.46, at the lag the transport's
     geometry predicts and not at the field period, one mark locating the
     tape to 1.2-1.7 us (`MEASURED`, `tape_recurrence_lag`). It offers
     fewer marks a second than the head switch and is weaker in the audio
     channel for the reason `dropout_visibility` computes - but it measures
     a DIFFERENT quantity, which is why it is needed as well and not
     instead.
  3. THE CAPTURE CLOCK, if the two converters share one. Then the temporal
     alignment is a single constant skew rather than a drifting one, and
     `rate_precision` says how well a capture of a given length bounds any
     residual drift.

WHY BOTH METHODS ARE NEEDED: TWO UNKNOWNS, TWO EQUATIONS. Let s be the
capture skew (the audio file's clock origin against the video file's) and
L the tape-position lead of the audio head over the video head. A
head-switch comparison sees only their SUM - an audio switch observed
early is equally explained by a card started late or by a head mounted
ahead - so it yields one equation in two unknowns. A defect seen in both
captures yields L on its own, because a defect is at a place, not at a
time. `separate_skew_from_lead` closes the pair. With a shared converter
clock s is predicted to be small and constant, so recovering it and
finding it small is a TEST of the whole scheme rather than a free
parameter.

WHAT THE PAIR IS FOR, IN THIS ARC'S OWN TERMS. Ethan intends to merge a
Hi-Fi decoder and treat the two captures as a complex pair. In the
tesseract that is not a figure of speech: video-against-audio is a binary
measurement axis exactly like head, polarity, tape, half and channel, and
adding it raises the cube from 32 vertices to 64. `fold_axis` declares it.
The fold's parent is what the two share - the tape, the drum, the capstan,
the tension - and its child is what differs: the head, the azimuth (30 deg
30 min against 6 deg, clauses 5.4 and table 2), the depth read (0.710 and
0.543 um against 0.272 and 0.210 um, `hifi_carriers.read_depth_m`) and the
band. The last of those is the one that matters most, because this arc's
measured identifiability ceiling says only fractional bandwidth moves it:
`fold_axis` computes that adding the audio band takes the observed span
from a ratio of 1.294 to one of 3.385, which is 4.73 times as many decades
for separating exponential losses that are collinear over the video band
alone. So the alignment is not a convenience. It is what makes the axis
exist, and its precision sets what the contrast can resolve
(`resolution_from_alignment`).

WHAT COULD NOT BE MEASURED HERE, STATED PLAINLY. There is no audio
radio-frequency capture on this machine. Every figure below that concerns
the audio side is therefore a PREDICTION conditional on the audio side
behaving as the video side measurably does, and is labelled as one;
nothing here is a demonstration of alignment, and no synthesis in this
file is offered as a measurement. `refusals` lists what is withheld and
`the_capture_that_would_prove_it` states the one capture that would settle
all of it.
"""

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np

from vhsdecode.models import (hifi_carriers, interference, magnetic_circuit,
                              rate_constraints, rf_stages, transport_model)


# --------------------------------------------------------------------------
# The specification, quoted with its clause
# --------------------------------------------------------------------------

def _entry(value, unit: str, clause: Optional[str], quote: str,
           note: str = "") -> Dict[str, object]:
    return {"value": value, "unit": unit, "clause": clause,
            "quote": quote, "note": note}


# SMPTE 32M-2004, fetched from decode-orc/analogue-video-specifications,
# docs/vhs/SMPTE-32M-2004/markdown.md, on 2026-09-06. Everything this file
# treats as normative is here with the clause it came from; everything it
# derives says so.
SPECIFICATION: Dict[str, Dict[str, object]] = {
    "separate_audio_heads": _entry(
        True, "1", "5.1",
        "Separate FM audio heads shall be mounted on the scanning drum for "
        "this purpose.",
        note="So the two signals are never on one wire, and a single "
             "wideband tap cannot reach both. This is why an audio capture "
             "is a SECOND capture and why it needs aligning at all."),
    "audio_recorded_first": _entry(
        True, "1", "5.3",
        "The FM audio heads shall be arranged so that the FM audio signal "
        "is recorded within the time-difference limits specified in table "
        "5. The FM audio signal shall be recorded prior to the video "
        "signal which is recorded at the same location on the tape.",
        note="'Prior' fixes the SIGN of the tape-position lead: the audio "
             "head reaches a point of tape BEFORE the video head does, at "
             "record and equally at replay, because it is the same "
             "geometry."),
    # table 5, "Recording time sequence", in field periods. The pair is
    # (earliest, latest) that the standard permits.
    "audio_lead_fields_sp": _entry(
        (Fraction(0), Fraction(2)), "field", "table 5",
        "Audio recorded 0 to 2 fields earlier than the associated video "
        "recording"),
    "audio_lead_fields_lp": _entry(
        (Fraction(1, 3), Fraction(7, 3)), "field", "table 5",
        "Audio recorded 1/3 to 2-1/3 fields earlier than the associated "
        "video recording"),
    "audio_lead_fields_ep": _entry(
        (Fraction(4, 3), Fraction(10, 3)), "field", "table 5",
        "Audio recorded 1-1/3 to 3-1/3 fields earlier than the associated "
        "video recording"),
    "audio_azimuth_deg": _entry(
        (30.5, -30.5), "degree", "5.4",
        "The azimuth angles of adjacent FM audio tracks shall be +30 deg "
        "30 min and -30 deg 30 min, respectively."),
    "azimuth_direction_vs_video": _entry(
        {"SP": "opposite", "LP": "opposite", "EP": "same"}, "1", "table 5",
        "Azimuth angle direction: audio head vs video head - Opposite "
        "direction (SP), Opposite direction (LP), Same direction (EP)",
        note="This is what pairs a given audio head with a given video "
             "head, and therefore what stops the lead being ambiguous by "
             "half a drum revolution: an audio head may only lay its track "
             "where a video head of the other azimuth will later record."),
    "head_switch_ahead_of_vsync_lines": _entry(
        (5.0, 8.0), "line", "3.6",
        "The switching position between the two heads during playback "
        "shall lie between 5 and 8 horizontal lines ahead of the leading "
        "edge of the vertical sync signal",
        note="THREE LINES WIDE, so the switch's placement against the "
             "picture is a deck constant too, not a format one. It does "
             "not enter the alignment: both captures are of one deck at "
             "one moment, so whatever the servo chose is common."),
    "drum_diameter_m": _entry(
        0.0620, "m", "3.1.5",
        "The video head drum diameter shall be 62.00 mm +- 0.01 mm."),
    "drum_rate_hz": _entry(
        None, "Hz", None,
        "NOT STATED - the standard gives no rpm and no rev/s",
        note="Derived from clause 1's '525 lines, 59.94 fields per second' "
             "and the two-head 180 degree scan of 3.4: one revolution per "
             "two fields. `transport_model.drum_rate_hz` owns the "
             "derivation and this file calls it rather than re-deriving."),
    "audio_control_head_distance_m": _entry(
        79.244e-3, "m", "3.4",
        "The distance X on the tape pattern, from the end of the 180 deg "
        "scan of a video head to the audio and control track head "
        "position, shall be 79.244 mm",
        note="THIS ONE IS SPECIFIED, unlike the rotary audio head's angle. "
             "It is the LINEAR audio and control track head, a stationary "
             "head downstream of the drum, and it is relevant here only "
             "because it shows the standard fixes the geometry it means to "
             "fix - the silence about the rotary FM audio head's angle is "
             "deliberate, not an omission."),
    "carrier_centres_hz": _entry(
        (1.3e6, 1.7e6), "Hz", "5.8",
        "Channel 1: 1.3 MHz +- 10 kHz; Channel 2: 1.7 MHz +- 10 kHz"),
    "carrier_deviation_hz": _entry(
        {"maximum": 150e3, "reference": 50e3}, "Hz", "5.9",
        "Maximum frequency deviation: +- 150 kHz; Reference frequency "
        "deviation: +- 50 kHz at 400 Hz"),
    "noise_reduction": _entry(
        {"ratio": 2.0, "attack_s": (3e-3, 10e-3), "recovery_s": (0.056, 0.084)},
        "1", "5.6",
        "Compression ratio: 2:1 logarithmic compression; Attack time: 3 to "
        "10 ms; Recovery time: 70 ms +- 14 ms",
        note="AT SILENCE THE ENCODER'S GAIN IS MAXIMAL, so whatever noise "
             "sits at the deck's audio input is amplified before it "
             "modulates the carrier. A recording made with nothing plugged "
             "into the audio inputs therefore does NOT guarantee a clean "
             "unmodulated line, and the narrow-line detector is not "
             "automatically the operative one. This is why "
             "`carrier_route_bound` quotes the WEAKER of the two "
             "statistics' bounds as the conservative figure."),
    "recording_level": _entry(
        None, "1", "5.11",
        "The FM audio rf signals shall be recorded such that their "
        "playback rf levels are approximately equal and maximum.",
        note="'Maximum' means there is no headroom to be gained by asking "
             "for a louder recording: the crosstalk bound cannot be beaten "
             "by turning the audio up."),
}


# The deck that made every test-pattern capture in this repository. Its
# figures are read from its own schematic and are carried by
# `rate_constraints.DECKS`; this file cites that rather than re-reading the
# sheet, and names the two signals the alignment argument depends on.
DECK = "SLV-778HF"


# --------------------------------------------------------------------------
# The rates, all delegated so that no number is typed twice
# --------------------------------------------------------------------------

def rates(system: str = "NTSC") -> Dict[str, float]:
    """Field, drum and line rates, and their periods.

    None of these is typed here. The field and line rates come from
    `rate_constraints`, which reads them from the format's own arithmetic,
    and the drum rate from `transport_model.drum_rate_hz`, which derives
    one revolution per two fields from the two-head 180 degree scan of
    SMPTE 32M clause 3.4. For NTSC that is 59.94 fields a second, a drum at
    29.97 rev/s and a drum period of 33.3667 ms.
    """
    field_hz = rate_constraints.field_rate_hz(system)
    drum_hz = transport_model.drum_rate_hz(field_hz)
    line_hz = rate_constraints.line_rate_hz(system)
    return {"field_rate_hz": float(field_hz),
            "drum_rate_hz": float(drum_hz),
            "line_rate_hz": float(line_hz),
            "field_period_s": 1.0 / float(field_hz),
            "drum_period_s": 1.0 / float(drum_hz),
            "line_period_s": 1.0 / float(line_hz)}


def audio_lead_window(tape_speed: str = "SP", system: str = "NTSC"
                      ) -> Dict[str, object]:
    """What SMPTE 32M table 5 permits for the audio head's tape-position
    lead, in seconds as well as in fields.

    The lead L is the time by which the audio head reaches a point of tape
    ahead of the video head. Clause 5.3 fixes its sign - audio first - and
    table 5 its range. The range is returned with the quantity that makes
    it interesting, `width_in_drum_revolutions`, which is 1.0 at every tape
    speed.
    """
    key = "audio_lead_fields_" + str(tape_speed).strip().lower()
    if key not in SPECIFICATION:
        raise ValueError(
            f"table 5 states a recording time sequence for SP, LP and EP "
            f"only; {tape_speed!r} is not one of them")
    entry = SPECIFICATION[key]
    low, high = entry["value"]
    r = rates(system)
    return {
        "tape_speed": str(tape_speed).upper(),
        "clause": entry["clause"],
        "quote": entry["quote"],
        "fields": (low, high),
        "seconds": (float(low) * r["field_period_s"],
                    float(high) * r["field_period_s"]),
        "width_fields": float(high - low),
        "width_s": float(high - low) * r["field_period_s"],
        "width_in_drum_revolutions": (float(high - low) * r["field_period_s"]
                                      / r["drum_period_s"]),
        "sign": "the audio head reaches a point of tape BEFORE the video "
                "head (clause 5.3, 'recorded prior to')",
    }


def offset_is_not_specified(system: str = "NTSC") -> Dict[str, object]:
    """THE CENTRAL RESULT: the format fixes that the constant exists and
    refuses to fix its value, and it refuses by exactly a whole revolution.

    Each of table 5's three windows is two field periods wide. The drum
    carries two video heads and turns once per two fields, so two field
    periods is one full revolution: the standard permits the audio head
    pair to sit at any angle whatever relative to the video pair. A number
    that may be anything on the circle is not a prediction, so no amount of
    reading the standard will align two captures - the offset is a constant
    OF THE DECK and must be measured once.

    That is not a defeat. A mechanical angle between two head assemblies
    bolted to one drum does not change with time, temperature or tape, so
    one measurement serves for every capture that deck ever makes at that
    tape speed. What it does mean is that the measurement cannot be
    skipped, and that a deck with separate SP and EP video heads needs it
    once per speed.
    """
    r = rates(system)
    windows = {speed: audio_lead_window(speed, system)
               for speed in ("SP", "LP", "EP")}
    return {
        "windows": windows,
        "width_in_drum_revolutions": {
            speed: window["width_in_drum_revolutions"]
            for speed, window in windows.items()},
        "drum_period_s": r["drum_period_s"],
        "permitted_angle_deg": 360.0 * max(
            window["width_in_drum_revolutions"] for window in windows.values()),
        "verdict": ("the permitted range is one whole drum revolution at "
                    "every tape speed, so the specification determines "
                    "nothing about the offset and it must be measured per "
                    "deck and per tape speed"),
        "what_is_fixed": ("that the offset is a fixed mechanical angle "
                          "between two head assemblies on one drum "
                          "(clause 5.1), hence a constant of the deck"),
    }


def head_pair_placement(system: str = "NTSC") -> Dict[str, object]:
    """Where the LP and EP video head pairs sit, DERIVED from table 5.

    THIS IS A DERIVATION, NOT A QUOTATION, and it is labelled one. The
    three windows in table 5 are all two fields wide and differ only in
    where they are centred: 1, 1-1/3 and 2-1/3 fields for SP, LP and EP.
    The audio heads do not move between tape speeds - a deck has one FM
    audio head pair - so a difference in the window's centre can only come
    from the VIDEO heads being somewhere else, which is what a deck with
    separate SP and EP video heads has. Converting the centre difference to
    an angle at 180 degrees a field gives where the other pairs sit
    relative to the SP pair.

    The assumption this rests on, stated because it is doing work: that
    table 5's three windows are centred on the actual head placements
    rather than being three independently drawn tolerance bands. The
    standard does not say so. What supports it is that all three have the
    same width, which is what a common tolerance about three different
    nominals looks like.
    """
    r = rates(system)
    centres = {}
    for speed in ("SP", "LP", "EP"):
        low, high = SPECIFICATION["audio_lead_fields_" + speed.lower()]["value"]
        centres[speed] = (low + high) / 2
    reference = centres["SP"]
    out = {}
    for speed, centre in centres.items():
        fields = centre - reference
        # 180 degrees of drum rotation per field, from the two-head scan
        degrees = float(fields) * 180.0
        out[speed] = {
            "centre_fields": centre,
            "fields_from_sp": fields,
            "degrees_from_sp": degrees,
            "degrees_from_sp_mod_180": degrees % 180.0,
            "seconds_from_sp": float(fields) * r["field_period_s"],
        }
    return {
        "placements": out,
        "note": ("LP and EP both come out 60 degrees from the SP pair "
                 "modulo 180, which is what one extra head pair mounted at "
                 "60 degrees would give, the two differing by which head "
                 "of that pair leads. That is a consistency check on the "
                 "derivation and not an independent measurement."),
        "status": "DERIVED from table 5's window centres, not quoted",
    }


# --------------------------------------------------------------------------
# METHOD 1 - the drum, seen twice
# --------------------------------------------------------------------------

def switch_phase(times_s: Sequence[float],
                 index: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """The drum's phase and rate, from a run of head-switch event times.

    The head switch is a once-per-field event locked to the drum's
    rotation, so a straight line through (event number, event time) has the
    field period for its slope and the phase for its intercept. What makes
    it worth fitting rather than merely differencing is the RESIDUAL: it is
    the whole of the estimator's noise plus whatever the drum did that a
    constant rate does not describe, and it is the only honest measure of
    how well one event locates the rotation.

    `index` is the event's field number, which need not be consecutive: a
    field whose switch could not be located is simply left out rather than
    silently renumbering the rest, because renumbering turns a gap into a
    rate error.

    The intercept is reported at the run's own CENTROID rather than at
    index zero. At the centroid the intercept and the slope are
    uncorrelated, so the phase's error is sigma / sqrt(N) with no term in
    the rate's error; quoted at index zero instead it inherits the rate's
    lever arm and looks worse than it is.
    """
    t = np.asarray(times_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 3:
        raise ValueError("a rate and a phase need at least three events; "
                         f"got {t.size}")
    k = (np.arange(t.size, dtype=np.float64) if index is None
         else np.asarray(index, dtype=np.float64))
    if k.shape != t.shape:
        raise ValueError("index must have one entry per event time")
    kbar = float(k.mean())
    sxx = float(((k - kbar) ** 2).sum())
    if sxx <= 0.0:
        raise ValueError("every event carries the same index, so no rate "
                         "can be fitted")
    period = float(((k - kbar) * (t - t.mean())).sum() / sxx)
    epoch = float(t.mean())                      # the time at index kbar
    residual = t - (epoch + (k - kbar) * period)
    dof = max(t.size - 2, 1)
    variance = float((residual ** 2).sum() / dof)
    return {
        "count": int(t.size),
        "index_centroid": kbar,
        "epoch_s": epoch,
        "period_s": period,
        "rate_hz": (1.0 / period) if period else float("nan"),
        "residual_rms_s": float(np.sqrt((residual ** 2).mean())),
        "residual_s": residual,
        "epoch_se_s": float(np.sqrt(variance / t.size)),
        "period_se_s": float(np.sqrt(variance / sxx)),
        "span_s": float(t.max() - t.min()),
    }


def align_by_switches(video: Dict[str, object], audio: Dict[str, object],
                      cycle_s: float,
                      reference_time_s: Optional[float] = None
                      ) -> Dict[str, object]:
    """The temporal alignment of two captures, from their head switches.

    Both captures see one drum. Fit each capture's switch events with
    `switch_phase`, evaluate the two phases at ONE reference instant, and
    the difference is how far the audio capture's switch sequence sits from
    the video capture's. It is returned wrapped into plus or minus half a
    cycle, because a run of switch events cannot say which cycle it is in.

    WHAT `cycle_s` MUST BE, and getting it wrong is the easiest mistake
    here. If the two heads of a pair can be told apart - in the video
    capture by the field's parity and by the step in the blanking radio
    frequency level that `head_switch_pair.luma_switch` measures, in an
    audio capture by the two heads' opposed azimuths - then the sequence
    repeats once per DRUM revolution and the cycle is two field periods.
    If they cannot, it repeats once per FIELD and the ambiguity doubles.
    Pass `rates()["drum_period_s"]` for the first and `field_period_s` for
    the second; nothing here guesses.

    AND THE AUDIO SIDE'S HANDLE MAY NOT SURVIVE THE DECK. The obvious way
    to tell two heads apart is their level, and VTG82063 7.3.2 puts an FM
    AGC in the playback chain "to compensate for level fluctuations due to
    variations in head to tape contact" - IC340 carries its own as well -
    so a tap at `CN341` pin 3 may already be levelled. Capturing `AF SWP`
    settles it outright, because its two half-cycles ARE the two heads.

    WHAT THIS NUMBER IS NOT. It is the sum of the capture skew and the
    audio head's tape-position lead, not either alone; see
    `separate_skew_from_lead`.
    """
    cycle = float(cycle_s)
    if not cycle > 0.0:
        raise ValueError("the switch cycle must be a positive duration")
    if reference_time_s is None:
        reference_time_s = 0.5 * (float(video["epoch_s"]) + float(audio["epoch_s"]))
    t_ref = float(reference_time_s)

    def phase_at(fit: Dict[str, object]) -> float:
        # how far past a switch the reference instant sits, in seconds
        elapsed = t_ref - float(fit["epoch_s"])
        return elapsed - cycle * math.floor(elapsed / cycle)

    raw = phase_at(audio) - phase_at(video)
    wrapped = raw - cycle * round(raw / cycle)
    # the two fits' rates, as a fractional difference: this is the relative
    # error of the two capture clocks if the drum is common, which it is
    v_period, a_period = float(video["period_s"]), float(audio["period_s"])
    relative_rate = (a_period - v_period) / v_period if v_period else float("nan")
    v_se, a_se = float(video["epoch_se_s"]), float(audio["epoch_se_s"])
    return {
        "offset_s": float(wrapped),
        "offset_se_s": float(math.hypot(v_se, a_se)),
        "cycle_s": cycle,
        "reference_time_s": t_ref,
        "relative_clock_rate": float(relative_rate),
        "relative_clock_rate_se": float(math.hypot(
            float(video["period_se_s"]), float(audio["period_se_s"])) / v_period)
        if v_period else float("nan"),
        "ambiguity_s": cycle,
        "meaning": ("the audio capture's switch sequence leads the video "
                    "capture's by this much, modulo one cycle; it is the "
                    "SUM of the capture skew and the audio head's "
                    "tape-position lead"),
    }


def jitter_survival(jitter_hz, lead_s: float) -> np.ndarray:
    """How much of the drum's own jitter survives the DIFFERENCE of the two
    switch times - which is the reason the method is better than the raw
    scatter of either capture suggests.

    Both captures watch one drum, so an angular error common to the two
    events cancels when they are subtracted. It does not cancel exactly,
    because the audio and video switches happen at different drum angles
    and therefore at instants `lead_s` apart, so the difference samples the
    error twice with that delay. For an error at frequency f the surviving
    factor is |2 sin(pi f lead)|, which goes to zero as the lead does and
    which is exactly zero when the lead is a whole number of the error's
    own periods.

    The consequence worth knowing before designing a capture: a lead near a
    whole drum revolution cancels the drum's own wow completely, and a lead
    near half a revolution DOUBLES it. Since the lead is a property of the
    deck and cannot be chosen, this is a thing to measure and report, not
    to assume.
    """
    f = np.asarray(jitter_hz, dtype=np.float64)
    return np.abs(2.0 * np.sin(np.pi * f * float(lead_s)))


def field_sampling_alias(jitter_hz, system: str = "NTSC") -> Dict[str, object]:
    """Where a mechanical disturbance lands once the switch is sampled once
    a field - and why the drum's own wow costs the offset nothing.

    A head-switch sequence is one measurement per field, so it samples the
    transport at the field rate. `transport_model` already records what
    that does: the drum turns at exactly half the field rate, so the drum's
    own once-per-revolution disturbance sits exactly at the Nyquist
    frequency of this sampling and folds to an alternation between the two
    heads rather than to scatter. It becomes a per-head CONSTANT, which a
    fit that keeps the two heads apart absorbs and which cancels again in
    the difference of two captures that sample the same drum the same way.

    So the term that looks like it should dominate does not, and what is
    left to limit the offset is the locator's own noise plus the transport
    at rates that are NOT harmonics of the drum: the capstan band and the
    reel band, both slower, both largely cancelled by
    `jitter_survival` for any lead short of half a revolution.
    """
    r = rates(system)
    f = np.asarray(jitter_hz, dtype=np.float64)
    nyquist = 0.5 * r["field_rate_hz"]
    folded = np.abs(f - r["field_rate_hz"] * np.round(f / r["field_rate_hz"]))
    return {
        "sample_rate_hz": r["field_rate_hz"],
        "nyquist_hz": float(nyquist),
        "folded_hz": folded,
        "drum_is_at_nyquist": bool(
            abs(r["drum_rate_hz"] - nyquist) < 1e-9 * nyquist),
        "note": ("the drum rate is exactly the Nyquist of once-a-field "
                 "sampling, so drum wow appears as a fixed difference "
                 "between the two heads and not as scatter"),
    }


def offset_precision(locator_sigma_video_s: float,
                     locator_sigma_audio_s: float,
                     count: int,
                     transport_jitter_rms_s: float = 0.0,
                     transport_jitter_hz: float = 0.0,
                     lead_s: float = 0.0) -> Dict[str, object]:
    """What the offset costs in seconds, derived rather than asserted.

    Three terms, and only the first two are usually there.

      the LOCATOR noise of each capture, which is the residual scatter
      `switch_phase` reports once the transport's coherent part is out of
      it. Two captures, two independent locators, so they add in quadrature
      and then fall as one over the square root of the number of events;

      the TRANSPORT's surviving common mode. A disturbance shared by both
      captures cancels in their difference to the factor `jitter_survival`
      gives. It is coherent from event to event, so unlike the locator
      noise it does NOT fall with the count and it has to be beaten by the
      cancellation alone;

      and nothing else, because everything the two captures share exactly -
      the drum's mean rate, the servo's placement of the switch against the
      picture, the tape's speed - is common mode with zero lag and cancels
      completely.

    Pass `transport_jitter_rms_s` as the timing amplitude of a disturbance,
    not its fractional speed error: a fractional speed error e0 at
    frequency f displaces the time base by e0 / (2 pi f), which
    `timing_from_speed_error` converts.
    """
    n = int(count)
    if n < 3:
        raise ValueError("at least three events are needed for a rate and "
                         "a phase; got %d" % n)
    locator = math.hypot(float(locator_sigma_video_s),
                         float(locator_sigma_audio_s)) / math.sqrt(n)
    survival = float(jitter_survival(transport_jitter_hz, lead_s)) \
        if transport_jitter_hz else 0.0
    transport = survival * float(transport_jitter_rms_s)
    return {
        "count": n,
        "locator_term_s": float(locator),
        "transport_term_s": float(transport),
        "jitter_survival": float(survival),
        "offset_sigma_s": float(math.hypot(locator, transport)),
        "note": ("the locator term falls as one over the square root of "
                 "the count and the transport term does not, so past the "
                 "crossing point more events buy nothing and only a "
                 "shorter lead or a quieter transport does"),
    }


def timing_from_speed_error(fractional_speed_error: float,
                            frequency_hz: float) -> float:
    """A fractional speed error of amplitude e0 at frequency f displaces
    the time base by e0 / (2 pi f) seconds.

    The time base is the integral of the speed error, and integrating a
    sinusoid divides it by its own angular frequency. This is why a slow
    disturbance costs far more displacement than a fast one of the same
    fractional size, and why the reel band rather than the drum band sets
    the scale of what has to cancel.
    """
    f = float(frequency_hz)
    if f <= 0.0:
        raise ValueError("a speed error at zero frequency is a rate error, "
                         "not a displacement")
    return float(fractional_speed_error) / (2.0 * math.pi * f)


def events_for_precision(target_s: float, locator_sigma_video_s: float,
                         locator_sigma_audio_s: float) -> Dict[str, object]:
    """How many head-switch events a wanted offset precision costs, and how
    long that is in seconds of capture.

    Inverting the locator term of `offset_precision`: N is the squared
    ratio of the combined per-event scatter to the target. Events arrive at
    the field rate, so the duration follows.
    """
    combined = math.hypot(float(locator_sigma_video_s),
                          float(locator_sigma_audio_s))
    target = float(target_s)
    if target <= 0.0:
        raise ValueError("a precision target must be a positive duration")
    n = (combined / target) ** 2
    r = rates()
    return {"target_s": target, "per_event_sigma_s": combined,
            "events": float(n),
            "seconds_of_capture": float(n) / r["field_rate_hz"],
            "fields_per_second": r["field_rate_hz"]}


def rate_precision(offset_sigma_per_event_s: float, duration_s: float,
                   system: str = "NTSC") -> Dict[str, object]:
    """How well a capture of a given length bounds a DRIFT between the two
    clocks, as a fractional rate.

    Fitting the per-event offset against time with a straight line, the
    slope is the fractional difference between the two capture clocks. For
    N points spread uniformly over a span T the slope's standard error is
    sigma sqrt(12) / (sqrt(N) T), and with events at the field rate
    N = f_field T, so the error falls as T to the three-halves. That steep
    exponent is the argument for a long capture rather than a careful one:
    doubling the length buys 2.83 times, where doubling the per-event
    precision buys 2.

    If the two converters run from one clock there is nothing to drift and
    this is a TEST rather than a measurement: a slope consistent with zero
    is the evidence that the shared clock did its job, and a slope that is
    not is the evidence that it did not.
    """
    r = rates(system)
    t = float(duration_s)
    if t <= 0.0:
        raise ValueError("a capture has a positive duration")
    n = r["field_rate_hz"] * t
    sigma = float(offset_sigma_per_event_s)
    return {
        "duration_s": t, "events": float(n),
        "fractional_rate_sigma": float(sigma * math.sqrt(12.0 / n) / t),
        "parts_per_million": float(sigma * math.sqrt(12.0 / n) / t * 1e6),
        "scaling": "falls as the duration to the three-halves",
    }


# --------------------------------------------------------------------------
# METHOD 2 - the tape, crossed twice
# --------------------------------------------------------------------------

def spacing_loss_ratio(system: str = "NTSC", tape_speed: str = "SP"
                       ) -> Dict[str, object]:
    """How much less a dropout costs the audio channel than the video one.

    A dropout is head-to-tape separation, and separation costs 54.6 d over
    lambda decibels (Wallace; `magnetic_circuit.spacing_loss_db` owns the
    constant and derives it as 20 log10(e) times 2 pi). The cost therefore
    scales as one over the recorded wavelength, and the two channels record
    at very different wavelengths - `hifi_carriers.wavelength_clusters`
    gives 4.4615 and 3.4118 um for the audio carriers against 1.7059 and
    1.3182 um for the luminance carrier at sync tip and peak white.

    So the SAME lump of dirt is between two and three and a half times
    weaker in the audio channel. That is not an inconvenience to be
    apologised for; it is the mechanism that makes Hi-Fi audio survive
    dropouts that wipe the picture, and it is a hard limit on this method:
    a defect must be large enough to clear the audio capture's own noise
    after being divided by that ratio before it can be used as a common
    mark at all.
    """
    lam = hifi_carriers.wavelength_clusters(system, tape_speed)
    out = {}
    for video_name, video_lambda in lam["luma"].items():
        for audio_name, audio_lambda in lam["hifi"].items():
            out[(video_name, audio_name)] = float(audio_lambda / video_lambda)
    return {
        "wavelengths_m": lam,
        # video dB divided by audio dB for one separation, by pair
        "video_over_audio_db": {f"{v} / {a}": r for (v, a), r in out.items()},
        "range": (float(min(out.values())), float(max(out.values()))),
        "law": "54.6 d / lambda dB, magnetic_circuit.spacing_loss_db",
    }


def dropout_visibility(video_depth_db, system: str = "NTSC",
                       tape_speed: str = "SP",
                       video_carrier: str = "sync_tip",
                       audio_channel: str = "channel_1") -> Dict[str, object]:
    """The same separation event, in the audio channel, given its depth in
    the video one.

    The separation is recovered from the video depth by inverting Wallace
    at the video wavelength, then applied at the audio wavelength. The
    result is the depth the audio capture would show, in decibels, and the
    separation itself in micrometres so that it can be sanity-checked
    against what a particle actually is.
    """
    lam = hifi_carriers.wavelength_clusters(system, tape_speed)
    video_lambda = lam["luma"][video_carrier]
    audio_lambda = lam["hifi"][audio_channel]
    depth = np.asarray(video_depth_db, dtype=np.float64)
    # invert 54.6 d / lambda for d, using the module that owns the constant
    per_metre = magnetic_circuit.spacing_loss_db(1.0, video_lambda)
    separation = depth / per_metre
    audio_depth = magnetic_circuit.spacing_loss_db(separation, audio_lambda)
    return {
        "video_depth_db": depth,
        "separation_m": separation,
        "separation_um": separation * 1e6,
        "audio_depth_db": audio_depth,
        "ratio": float(video_lambda and audio_lambda / video_lambda),
        "video_carrier": video_carrier, "audio_channel": audio_channel,
        "assumption": ("that the event is pure head-to-tape separation and "
                       "the same separation at both heads. A defect that "
                       "removes the top of the coating rather than lifting "
                       "the head hurts the surface-recorded video and "
                       "spares the buried audio, and is not this; such an "
                       "event is video-only and simply fails to appear as "
                       "a coincidence"),
    }


def dropout_timing_precision(edge_time_s: float, amplitude_snr: float,
                             ) -> Dict[str, object]:
    """How well a dropout's time can be read, from its edge and its depth.

    An edge of rise time `edge_time_s` observed with amplitude
    signal-to-noise `amplitude_snr` locates its own position to about the
    edge time divided by that ratio: the slope at the edge is the
    amplitude over the edge time, and a noise of amplitude A over that
    slope is a time error of A times the edge time over the amplitude.
    This is the standard threshold-crossing result and it is the same
    quantity `sync_geometry` uses for a sync edge.

    Two captures, two independent readings, so the coincidence's precision
    is the quadrature sum - and the audio capture's own signal-to-noise is
    the video one's divided by `spacing_loss_ratio`, which is where this
    method loses to the head-switch one.
    """
    edge = float(edge_time_s)
    snr = float(amplitude_snr)
    if edge <= 0.0 or snr <= 0.0:
        raise ValueError("an edge has a positive duration and a positive "
                         "signal-to-noise ratio")
    return {"edge_time_s": edge, "amplitude_snr": snr,
            "timing_sigma_s": edge / snr,
            "law": "sigma_t = t_rise / (A / sigma_A)"}


def separate_skew_from_lead(switch_offset_s: float,
                            dropout_lead_s: float,
                            switch_offset_sigma_s: float = 0.0,
                            dropout_lead_sigma_s: float = 0.0,
                            tape_speed: str = "SP", system: str = "NTSC"
                            ) -> Dict[str, object]:
    """TWO UNKNOWNS, TWO EQUATIONS - and a test that comes free with them.

    The head-switch comparison measures the sum: an audio switch seen early
    is equally well explained by a converter started late (the skew s) or
    by a head mounted ahead (the tape-position lead L). A defect measured
    in both captures measures L alone, because a defect is at a place and
    not at a time. So

        switch offset  =  s + L        (`align_by_switches`)
        dropout lead   =      L        (a coincidence in tape position)
        hence  s = switch offset - dropout lead.

    THE TEST. If the two converters share a clock, s is a small constant
    and nothing more: an s that comes out large, or that changes between
    captures, says the sharing failed. And L must land inside the window
    SMPTE 32M table 5 permits - `audio_lead_window` - which is an
    independent check on the whole arrangement that costs nothing to make.
    A lead outside the permitted window means the coincidence was matched
    to the wrong defect or the wrong drum revolution.
    """
    skew = float(switch_offset_s) - float(dropout_lead_s)
    window = audio_lead_window(tape_speed, system)
    low, high = window["seconds"]
    inside = bool(low - 1e-12 <= float(dropout_lead_s) <= high + 1e-12)
    return {
        "capture_skew_s": skew,
        "capture_skew_sigma_s": float(math.hypot(float(switch_offset_sigma_s),
                                                 float(dropout_lead_sigma_s))),
        "tape_position_lead_s": float(dropout_lead_s),
        "permitted_window_s": (low, high),
        "lead_inside_the_permitted_window": inside,
        "verdict": ("consistent with the standard" if inside else
                    "the lead is outside what SMPTE 32M table 5 permits, "
                    "so the coincidence is matched to the wrong defect or "
                    "the wrong drum revolution"),
    }


# --------------------------------------------------------------------------
# What the pair is for: the sixth axis of the fold
# --------------------------------------------------------------------------

def resolution_from_alignment(sigma_s: float, system: str = "NTSC",
                              tape_speed: str = "SP",
                              time_base_error: float = 0.0
                              ) -> Dict[str, object]:
    """What a given alignment precision lets the paired fold resolve.

    A fold may only pair two measurements of the same moment. A residual
    misalignment of sigma seconds therefore enters the contrast as a phase
    error of 2 pi f sigma at every frequency, and any component of the
    contrast whose own phase is smaller than that is unrecoverable. That is
    the whole of the translation from a timing figure to a physical one,
    and it is why the alignment's precision and not its mere existence is
    what matters.

    Four readings of the same sigma:

      the PHASE it costs at the top of the luminance band and at each audio
      carrier, in radians, which bounds any all-pass or delay term the
      contrast could carry;
      the fraction of one OUTPUT SAMPLE at four times the colour
      subcarrier, which is this arc's own delivery grid;
      the fraction of one LINE, which is what a per-line correction needs;
      and, when a time-base error is given, the fraction of that error's
      own per-field displacement, which is the figure to quote when the
      pair is being used as a time-base probe - the point of the exercise.
    """
    sigma = float(sigma_s)
    r = rates(system)
    tip, white = rf_stages.VHS_CARRIER_HZ[
        "525" if str(system).upper().startswith("NTSC") else "625"]
    ch1, ch2 = SPECIFICATION["carrier_centres_hz"]["value"]
    fsc = colour_subcarrier_hz(system)
    out = {
        "sigma_s": sigma,
        "phase_rad": {
            "luma_peak_white": 2.0 * math.pi * white * sigma,
            "luma_sync_tip": 2.0 * math.pi * tip * sigma,
            "audio_channel_1": 2.0 * math.pi * ch1 * sigma,
            "audio_channel_2": 2.0 * math.pi * ch2 * sigma,
        },
        "output_samples_at_4fsc": sigma * 4.0 * fsc,
        "fraction_of_a_line": sigma / r["line_period_s"],
        "fraction_of_a_field": sigma / r["field_period_s"],
    }
    if time_base_error:
        displacement = abs(float(time_base_error)) * r["field_period_s"]
        out["time_base_displacement_per_field_s"] = displacement
        out["fraction_of_the_time_base_error"] = (
            sigma / displacement if displacement else float("inf"))
    return out


def colour_subcarrier_hz(system: str = "NTSC") -> float:
    """The colour subcarrier, taken from the colour-under module rather
    than typed, so one number has one owner."""
    from vhsdecode.models import colour_under as _cu
    return float(_cu.subcarrier_hz(
        "NTSC" if str(system).upper().startswith("NTSC") else "PAL"))


def fold_axis(system: str = "NTSC", tape_speed: str = "SP"
              ) -> Dict[str, object]:
    """THE SIXTH AXIS: video capture against audio capture.

    Ethan: *"In a later session we will merge these two projects to use
    both RF captures as a complex pair video and hifi."* In this arc's
    framework that is a literal statement. `tesseract` folds a cube on
    binary measurement axes - head, polarity, tape, half, channel - and
    which tap a measurement came from is exactly such an axis. Adding it
    takes the cube from thirty-two vertices to sixty-four and makes the
    contrast between the two taps a measurement in its own right.

    WHAT THE FOLD SEPARATES. The parent is what the two captures share and
    the child is what differs, and here that split is unusually clean,
    because the two signals travel the same tape on the same drum and part
    company only at the head:

        common      the tape and its coating, the defects in it, the drum's
                    rotation and its wow, the capstan, the tape tension,
                    the transport's whole time base
        different   the head and its gap, the azimuth (30 deg 30 min
                    against 6 deg, clauses 5.4 and table 2, so 36.5 deg
                    apart in SP), the depth read into the coating, the
                    band, and the playback amplifier

    WHY THAT IS WORTH HAVING, IN NUMBERS. This arc has measured that its
    magnetic mechanisms collapse - they are all monotone decays over a
    narrow band and are therefore nearly collinear - and that only
    fractional bandwidth moves that. The video band alone spans a frequency
    ratio of 1.29 from sync tip to peak white. With the audio carriers the
    observed span runs from 1.3 MHz to 4.4 MHz, a ratio of 3.39, which is
    4.7 times as many decades for telling one exponential decay from
    another. And the DEPTH sampling goes from two clusters to four:
    0.210 and 0.272 um for the luminance, 0.543 and 0.710 for the audio,
    1.466 for the colour under. Two clusters cannot split three losses;
    four can be asked to.

    THE STAGE ASSIGNMENT IS A LABELLED ASSUMPTION, as `tesseract.AXIS_STAGE`
    requires of every axis. The axis is proposed to first matter at the
    head-to-tape interface, because that is where clause 5.1 puts the
    separation - separate heads, different gaps, different azimuths, the
    audio written into the depth and the video onto the surface. It is NOT
    proposed at the capture, because by the capture the two have been
    different signals for the whole of the chain.
    """
    lam = hifi_carriers.wavelength_clusters(system, tape_speed)
    tip, white = rf_stages.VHS_CARRIER_HZ[
        "525" if str(system).upper().startswith("NTSC") else "625"]
    ch1, ch2 = SPECIFICATION["carrier_centres_hz"]["value"]
    video_ratio = white / tip
    paired_ratio = white / min(ch1, ch2)
    depths = {
        "luma_peak_white": lam["luma"]["peak_white"] / magnetic_circuit_divisor(),
        "luma_sync_tip": lam["luma"]["sync_tip"] / magnetic_circuit_divisor(),
        "audio_channel_2": lam["hifi"]["channel_2"] / magnetic_circuit_divisor(),
        "audio_channel_1": lam["hifi"]["channel_1"] / magnetic_circuit_divisor(),
        "colour_under": lam["chroma"]["colour_under"] / magnetic_circuit_divisor(),
    }
    return {
        "axis": "capture",
        "vertices": {"0": "the video tap (CN261 pin 2 'PB RF')",
                     "1": "the audio tap (CN341 pin 3 'FM PB')"},
        "common": ["the tape and its coating", "tape defects", "the drum",
                   "the drum's wow", "the capstan", "tape tension",
                   "the transport's time base"],
        "different": ["the head and its gap", "the azimuth",
                      "the depth read", "the band",
                      "the playback amplifier"],
        "relative_azimuth_deg": hifi_carriers.relative_azimuth_degrees(tape_speed),
        "frequency_ratio_video_only": float(video_ratio),
        "frequency_ratio_with_audio": float(paired_ratio),
        "decades_gained": float(math.log10(paired_ratio) / math.log10(video_ratio)),
        "depth_clusters_m": depths,
        "depth_cluster_count": len(depths),
        "proposed_axis_stage": "head contact tilt",
        "axis_stage_status": ("a labelled assumption about which stage "
                              "first makes the axis matter, as "
                              "tesseract.AXIS_STAGE requires; the evidence "
                              "for it is SMPTE 32M 5.1, which puts the "
                              "separation at the heads"),
        "cost": ("the axis exists only if the two captures can be put on "
                 "one time axis, and the contrast can resolve only what "
                 "`resolution_from_alignment` allows at the achieved "
                 "precision"),
    }


def magnetic_circuit_divisor() -> float:
    """lambda / (2 pi) is the depth a playback head reads from; the divisor
    is owned by `magnetic.DEMAGNETISATION_DIVISOR` and taken from there so
    that 2 pi is not typed in two places."""
    from vhsdecode.models import magnetic as _magnetic
    return float(_magnetic.DEMAGNETISATION_DIVISOR)


# --------------------------------------------------------------------------
# The route Ethan named, restated closed - on the valid subjects only
# --------------------------------------------------------------------------

# WHICH RECORDS COULD CARRY AN AUDIO CARRIER AT ALL. This is a fact about
# the tapes, and the provenance is the person who owns them.
#
# Ethan, 2026-09-06: "These sample may not have hifi audio. The zaroff
# samples should have hifi carriers, home and cd will not."
#
# The zaroff test-pattern tape was recorded on a Sony SLV-778HF, whose HF
# suffix denotes a Hi-Fi machine, and a Hi-Fi deck lays its audio carriers
# down whenever it records - frequency modulation of silence is the carrier
# at its centre. The capture readme records only a composite video
# connection ("TSG composite video out -> Sony VCR L1 composite video in"),
# so silence at the audio input is the likely case, not absence.
CARRIER_BEARING_RECORDS: Dict[str, Dict[str, object]] = {
    "zaroff SP playback": {
        "carries_hifi": True,
        "why": "recorded on the SLV-778HF, a Hi-Fi deck, which lays the "
               "carriers down whenever it records",
        "role": "subject"},
    "zaroff EP playback": {
        "carries_hifi": True, "why": "same deck, same tape, EP mode",
        "role": "subject"},
    "zaroff SP record (control)": {
        "carries_hifi": True,
        "why": "the same recording, but the tap is the VIDEO record "
               "current, where the audio carriers cannot appear because "
               "the audio record signal goes to IC340 and the audio "
               "windings and never to this pin",
        "role": "control"},
    "countdown": {
        "carries_hifi": False,
        "why": "Ethan, 2026-09-06: 'home and cd will not' have Hi-Fi audio",
        "role": "WITHDRAWN - searched for a signal that was never recorded"},
    "home": {
        "carries_hifi": False,
        "why": "Ethan, 2026-09-06: 'home and cd will not' have Hi-Fi audio",
        "role": "WITHDRAWN - searched for a signal that was never recorded"},
}

# THE CORRECTION, KEPT VISIBLE. This arc's rule is that a withdrawn claim
# stays visible with its reason rather than being edited away.
WITHDRAWN: Dict[str, object] = {
    "date": "2026-09-06",
    "claim": ("'no AFM carrier reaches the video tap in any capture in "
              "this repository, to a bound of -45 dB ... and -45 dB "
              "(reference-modulated, both channels, the 2-hour tapes)' - "
              "docs/HIFI_CAPTURE_PATH.md as written on 2026-09-05, and "
              "the -50 dB figures in its results table"),
    "withdrawn_because": ("`countdown` and `home` carry no Hi-Fi audio, so "
                          "their rows tested for a signal that was never "
                          "recorded. An absent signal found absent is not "
                          "a bound on leakage, and four of the ten "
                          "(record, channel) pairs in that table are those "
                          "two records"),
    "what_survives": ("the zaroff rows, which are the only valid subjects, "
                      "and the record-tap control, which is unaffected and "
                      "is in fact the strongest thing in the original "
                      "measurement"),
    "restated_by": "capture_alignment.carrier_route_bound",
}


def carrier_route_bound() -> Dict[str, object]:
    """THE ROUTE ETHAN NAMED, RESTATED CLOSED ON THE VALID SUBJECTS ONLY.

    His proposal was to synchronise the two captures by finding the audio
    carriers inside the video track. `hifi_carriers.detection_bound`
    measured whether they are there. This function reads that measurement
    rather than restating its numbers, splits it by whether the record
    could carry a carrier at all, and reports the bound on the subjects
    that could.

    WHY THE SPLIT MATTERS. The original wrote the bound over five records,
    two of which - `countdown` and `home` - have no Hi-Fi audio on them.
    Their rows tested for a signal that was never recorded, so their -50 dB
    figures are not evidence about leakage and are withdrawn; see
    `WITHDRAWN`. What is left is one tape at two speeds, plus a control.

    WHY THE CONTROL IS UNAFFECTED, AND IS THE BEST PART. The audio carriers
    go to separate heads on the drum (clause 5.1), so they can never appear
    at the VIDEO record-current tap. That makes the record tap a proper
    negative control rather than a coincidence, and it is what disposes of
    the apparent detection: the unplanted hump statistic is 4 to 8.6 on the
    zaroff records, which looks like a signal, and the record tap shows the
    SAME excess in the SAME bins with the playback LOWER. The excess is the
    video spectrum's own curvature.

    WHICH STATISTIC IS OPERATIVE, and this is the one place the correction
    makes the result WEAKER rather than merely narrower. The original
    argued that the zaroff recordings were made with no audio connected, so
    the carrier is unmodulated and the narrow-line bound applies. Clause
    5.6 puts a 2:1 logarithmic compressor in the record path, whose gain at
    silence is maximal, so whatever noise sits on an open audio input is
    amplified before it modulates the carrier. A "no audio connected"
    recording therefore does not guarantee a clean line, and the
    conservative figure to quote is the WEAKER of the two statistics'
    bounds, not the narrow-line one alone.
    """
    measured = hifi_carriers.MEASURED
    results = measured["results"]
    subjects, controls, withdrawn = {}, {}, {}
    for name, rows in results.items():
        where = CARRIER_BEARING_RECORDS.get(name)
        if where is None:
            continue
        if not where["carries_hifi"]:
            withdrawn[name] = rows
        elif where["role"] == "control":
            controls[name] = rows
        else:
            subjects[name] = rows
    # line_sigma is the first entry of each row and the bounds the last two
    line_sigmas = [row[0] for rows in subjects.values() for row in rows.values()]
    hump_sigmas = [row[1] for rows in subjects.values() for row in rows.values()]
    per_channel: Dict[int, Dict[str, float]] = {}
    for name, rows in subjects.items():
        for channel, row in rows.items():
            entry = per_channel.setdefault(channel, {})
            # the conservative bound is the WEAKER (less negative) of the
            # narrow-line and hump bounds, for the compander reason above
            entry[name] = max(row[5], row[6])
    return {
        "subjects": sorted(subjects),
        "control": sorted(controls),
        "withdrawn": sorted(withdrawn),
        "pairs": len(line_sigmas),
        "line_sigma_range": (min(line_sigmas), max(line_sigmas)),
        "hump_sigma_range": (min(hump_sigmas), max(hump_sigmas)),
        "detection_threshold_sigma": hifi_carriers.DETECTION_SIGMA,
        "detected": bool(max(line_sigmas) >= hifi_carriers.DETECTION_SIGMA),
        "conservative_bound_db": {
            channel: max(entry.values()) for channel, entry in per_channel.items()},
        "best_bound_db": {
            channel: min(entry.values()) for channel, entry in per_channel.items()},
        "verdict": ("no narrow line at either carrier on any record that "
                    "could carry one; the route is closed at the bound "
                    "below, which is WEAKER than the withdrawn -50 dB and "
                    "is the honest figure"),
        "not_evidence": ("the -50 dB figures came from `countdown` and "
                         "`home`, which have no Hi-Fi audio; see WITHDRAWN"),
        "what_would_lower_it": "see `bound_lowering`",
    }


# --------------------------------------------------------------------------
# If the carriers were ever found, they are a COMPONENT and not a special case
# --------------------------------------------------------------------------

# Ethan, 2026-09-06: *"if we detect hifi, that needs to be matched the same
# way we are for all the other components and subtracted out."* So the door
# is built before the detection rather than after it, and it is the same
# door: a declared chain position, a complex signature of the shape
# `interference` already carries, and admission by the held-out judge.
#
# WHERE IT ACTS. An audio carrier reaching the video tap does so by being
# picked up from the buried layer of the track the video head is on, so it
# is ADDED at the head's output, alongside the medium's own particle noise.
# It shares position 23 with that entry for the reason the map allows a
# share - the two act at the same point and commute, both being additive
# terms at one node. It is NOT placed in the capture band at 30: it would
# be there only if the coupling were electrical between the two amplifiers,
# and the record-tap control bounds that path, because the audio record
# drive is present during a record-tap capture and left no excess.
COMPONENT_ORDER: Dict[str, int] = {
    "afm carrier crosstalk": 23,
}


def silent_carrier_width_hz(time_base_error: float = 4e-4, channel: int = 1,
                            system: str = "NTSC") -> float:
    """How wide a SILENT carrier's line actually is, which the standard does
    not state and the transport decides.

    With nothing at the audio input the modulator sits at its centre, so
    clause 5.9's deviation figures do not apply and there is no specified
    width at all. What is left is the tape's own time base, which drags
    every recorded frequency by the fractional speed error: a carrier at
    1.3 MHz with the measured drum-rate wow of about four parts in ten
    thousand is a line some 520 Hz wide, not a delta function. That is the
    number to use, and it is a measurement of the transport rather than a
    quotation from the standard.

    The compander caveat stands over this: clause 5.6's 2:1 logarithmic
    encoder is at maximum gain when the input is silent, so noise on an
    open input is amplified before it modulates the carrier and can widen
    the line well past this figure. This is a FLOOR on the width.
    """
    centres = SPECIFICATION["carrier_centres_hz"]["value"]
    if int(channel) not in (1, 2):
        raise ValueError("SMPTE 32M clause 5.5 specifies two audio "
                         f"channels; {channel!r} is not one of them")
    return abs(float(time_base_error)) * float(centres[int(channel) - 1])


def afm_carrier_signature(frequency_hz, channel: int = 1,
                          width_hz: Optional[float] = None,
                          system: str = "NTSC",
                          amount: float = 1.0) -> np.ndarray:
    """The key entry an audio carrier would make, if one were ever found.

    It DELEGATES to `interference.beat` rather than inventing a shape,
    because a Hi-Fi carrier at the video tap is exactly what that entry
    describes - a line in the spectrum at a known frequency with a known
    origin - and because `beat` already returns the subtractable form
    `1 + a * shape` that `interference.subtractable` requires.

    THE WIDTH IS NOT GUESSED. It defaults to the reference-modulated reach
    of clause 5.9, `hifi_carriers.reference_half_width_hz`, because that is
    the only width the standard states. For a silent recording - which is
    what the zaroff captures are - pass `silent_carrier_width_hz`, which is
    a property of the transport and not of the format. Passing neither and
    hoping is what would put a specification number on a signal the
    specification says nothing about.
    """
    centres = SPECIFICATION["carrier_centres_hz"]["value"]
    if int(channel) not in (1, 2):
        raise ValueError("SMPTE 32M clause 5.5 specifies two audio "
                         f"channels; {channel!r} is not one of them")
    centre = float(centres[int(channel) - 1])
    width = (hifi_carriers.reference_half_width_hz() if width_hz is None
             else float(width_hz))
    if width <= 0.0:
        raise ValueError("a spectral line has a positive width")
    return interference.beat(frequency_hz, centre, width_hz=width,
                             amount=float(amount))


def admission_route() -> Dict[str, object]:
    """How a positive detection would have to be admitted - stated so that
    a future session does not build a bespoke path for it.

    The steps are the ones every other component in this arc takes, and
    none of them may be skipped on the grounds that this one is obvious:

      1. the signature is `afm_carrier_signature`, which is
         `interference.beat` and therefore already subtractable;
      2. the position is `COMPONENT_ORDER` above, composed into the chain
         by `interference.full_chain` once this module is registered there;
      3. admission is `interference.admission` followed by the held-out
         judge in `holdout`, which decides on out-of-sample evidence and
         not on whether the residual fell;
      4. subtraction happens only if it is admitted, and at the gain the
         correction-gain law gives rather than at the full fitted amount.

    On present evidence the entry would be REFUSED at step 3 for want of a
    detection: `carrier_route_bound` finds no line on any subject that
    could carry one. The entry exists so that a future positive goes
    through this door and not around it.
    """
    return {
        "signature": "capture_alignment.afm_carrier_signature",
        "position": dict(COMPONENT_ORDER),
        "composed_by": "interference.full_chain",
        "judge": "interference.admission, then holdout",
        "subtract_only_if": "admitted out of sample",
        "present_verdict": "refused for want of a detection",
        "evidence": carrier_route_bound()["verdict"],
    }


def bound_lowering(capture_duration_s: float = 0.480,
                   time_base_error: float = 4e-4,
                   channel: int = 1, system: str = "NTSC"
                   ) -> Dict[str, object]:
    """WHAT WOULD LOWER THE CARRIER BOUND, stated as arithmetic so the
    question is settled rather than left ajar.

    THE LARGEST AVAILABLE GAIN IS NOT MORE CAPTURE, IT IS COHERENCE. The
    search is a transform over the whole record, and a carrier is a line in
    it only if its frequency holds still. It does not: the tape's time-base
    error e drags every recorded frequency by e f, so a carrier at 1.3 MHz
    with the measured drum-rate wow of about 4 parts in ten thousand
    (`hifi_carriers` records it from `carrier_tbc`) is smeared over some
    hundreds of hertz while the transform's own bin is one over the
    duration. The signal is spread over that many bins and the detector
    sees a fraction of it.

    The remedy costs no new capture. If the carrier reaches the video tap
    magnetically - the video head reading the buried layer of the track it
    is on - then it is dragged by the SAME time base as the video, which
    this arc already measures. Correcting the record's time base before the
    transform therefore concentrates the whole carrier into one bin, and
    the gain is the ratio of the smeared width to the bin width. That
    number is returned as `coherence_gain_db` and it is large.

    WHAT IT DOES NOT HELP. If the coupling were electrical rather than
    magnetic the carrier would carry the AUDIO head's time base, which the
    video capture does not measure, and the correction would smear it
    instead of sharpening it. So the two mechanisms respond oppositely to
    this treatment, which makes the treatment a discriminator as well as a
    gain.

    LONGER CAPTURE, SECOND. Once coherent, the line grows over the floor as
    the duration, so a thirty-second capture beats a half-second one by
    18 dB. Incoherent, it buys nothing at all: the smear widens with
    nothing to concentrate it.

    MORE BITS, THIRD AND CONDITIONALLY. The zaroff captures are eight-bit
    from an oscilloscope. Six decibels a bit is available only if the floor
    at the carrier is the converter's and not the tape's, which
    `capture_chain_noise` is the instrument to decide and which is NOT
    settled here; it is listed with that condition attached.
    """
    ch1, ch2 = SPECIFICATION["carrier_centres_hz"]["value"]
    centre = float(ch1 if int(channel) == 1 else ch2)
    duration = float(capture_duration_s)
    if duration <= 0.0:
        raise ValueError("a capture has a positive duration")
    bin_hz = 1.0 / duration
    # peak-to-peak smear: the frequency swings +- e f about the centre
    smear_hz = 2.0 * abs(float(time_base_error)) * centre
    gain = max(smear_hz / bin_hz, 1.0)
    return {
        "carrier_hz": centre,
        "capture_duration_s": duration,
        "transform_bin_hz": bin_hz,
        "time_base_error": float(time_base_error),
        "smear_hz": smear_hz,
        "coherence_gain_db": float(10.0 * math.log10(gain)),
        "then_duration_gain_db_per_decade": 10.0,
        "ranked": [
            ("time-base correct the record before the transform",
             float(10.0 * math.log10(gain)),
             "no new capture; also discriminates magnetic coupling from "
             "electrical, because it sharpens one and smears the other"),
            ("a longer capture, AFTER the time base is corrected",
             float(10.0 * math.log10(30.0 / duration)),
             "10 dB a decade of duration, and exactly nothing before the "
             "time base is corrected"),
            ("a chrominance-free recording",
             None,
             "removes the colour-under's second harmonic at 1.258741 MHz, "
             "which sits 41 kHz below carrier 1 and inside the hump band; "
             "the zaroff set contains y-only recordings and the Hi-Fi path "
             "is untouched by the absence of chroma, so this costs nothing "
             "but choosing a different existing file"),
            ("more converter bits",
             None,
             "6 dB a bit, but ONLY if the floor at the carrier is the "
             "converter's rather than the tape's, which is not settled "
             "here; `capture_chain_noise` is the instrument for it"),
            ("a capture at CN341 pin 3 'FM PB'",
             None,
             "not a way of lowering this bound at all - it is the signal "
             "itself, and it is what the alignment actually needs"),
        ],
    }


# --------------------------------------------------------------------------
# What is withheld, and what would settle it
# --------------------------------------------------------------------------

def refusals() -> List[Dict[str, str]]:
    """What this module will not report, and why - each with what would
    lift the refusal.

    A thing that cannot be measured is refused with its reason rather than
    estimated, because an estimate presented beside measurements is
    indistinguishable from one afterwards.
    """
    return [
        {"refused": "the alignment itself, end to end",
         "because": "there is no audio radio-frequency capture on this "
                    "machine. Nothing here demonstrates two captures being "
                    "aligned; the estimator is exercised on the video side "
                    "and on planted signals only",
         "lifted_by": "one capture at CN341 pin 3 'FM PB' taken "
                      "simultaneously with one at CN261 pin 2 'PB RF'"},
        {"refused": "the audio head's tape-position lead L for this deck",
         "because": "SMPTE 32M table 5 permits a whole drum revolution "
                    "(`offset_is_not_specified`), so it cannot be read "
                    "from the standard, and measuring it needs both "
                    "captures",
         "lifted_by": "the same pair of captures, through "
                      "`separate_skew_from_lead`"},
        {"refused": "the locator noise of an AUDIO head-switch event",
         "because": "no audio capture exists, so the audio side of every "
                    "precision figure is a prediction conditional on the "
                    "audio locator matching the video one. It is labelled "
                    "as such wherever it appears and is not a measurement. "
                    "What is NO LONGER refused is whether there is an "
                    "audio head switch to locate at all: the schematic "
                    "shows IC340's two head amplifiers meeting at a "
                    "changeover switch driven by 'AF SWP' from IC160 pin "
                    "19, so the event exists and an AFM capture carries it",
         "lifted_by": "one audio capture of any length over three fields, "
                      "or - for the OFFSET rather than the locator noise - "
                      "an oscilloscope on CN261 pin 3 against JL345, which "
                      "needs no capture at all"},
        {"refused": "whether a dropout is genuinely common to both heads",
         "because": "it can be tested BETWEEN the two video heads on the "
                    "material here, which is a necessary condition, but "
                    "the video-to-audio case adds a different azimuth, a "
                    "different depth and a different band, and passing the "
                    "video-to-video test does not establish it",
         "lifted_by": "the paired capture, looking for the same defect in "
                      "both at the lead the head switch predicts"},
        {"refused": "the residual skew between two clock-synchronised "
                    "converter cards",
         "because": "no two-card capture exists here. A shared clock "
                    "removes drift and not a fixed skew, and the size of "
                    "that skew is exactly what `separate_skew_from_lead` "
                    "would return",
         "lifted_by": "one signal split to both cards and cross-correlated, "
                      "which is also the calibration this method needs"},
        {"refused": "whether the carrier bound is converter-limited or "
                    "tape-limited",
         "because": "`bound_lowering` lists more bits as worth six "
                    "decibels each ONLY under that condition, and the "
                    "condition is not tested here",
         "lifted_by": "`capture_chain_noise` run at the carrier "
                      "frequencies on captures at two bit depths"},
    ]


def the_capture_that_would_prove_it() -> Dict[str, object]:
    """The one capture that settles every open item above.

    It is a single experiment and it is within reach of the bench that made
    the existing set: the same deck, the same tape, the same procedure, one
    more channel.
    """
    return {
        "what": ("two simultaneous radio-frequency captures of ONE tape "
                 "pass, on a shared sample clock"),
        "channel_a": ("CN261 pin 2 'PB RF' - the video playback tap, which "
                      "is where every existing capture in this repository "
                      "was taken"),
        "channel_b": ("CN341 pin 3 'FM PB' - the AFM playback tap, the "
                      "audio equivalent of pin 2 on its own test connector"),
        "clock": ("one clock to both converters. The project's own answer "
                  "is two CX cards on an external clock (the clockgen "
                  "mod), which is what makes the two streams comparable "
                  "sample by sample"),
        "duration_s": 30.0,
        "why_that_long": ("the rate bound falls as the duration to the "
                          "three-halves (`rate_precision`), so thirty "
                          "seconds is about 200 times better than the "
                          "half-second captures here; and it is long "
                          "enough to contain dropouts large enough to "
                          "clear the audio channel's reduced sensitivity "
                          "(`dropout_visibility`)"),
        "also_capture": (
            "if further channels are available, the two head-switch square "
            "waves themselves: CN261 pin 3 'RF SWP' (a connector pin, "
            "ground on pin 4) and 'AF SWP' at the solder pad JL345. They "
            "turn the switch locator from an inference off a noisy "
            "radio-frequency envelope into a five-volt logic edge, which "
            "is the single largest improvement available to method one"),
        "do_first_because_it_is_free": (
            "read RF SWP against AF SWP on two oscilloscope channels with "
            "the deck merely in playback - no tape of interest, no "
            "synchronised converters, no radio frequency. That measures "
            "the switch offset directly and is the check the paired "
            "capture must reproduce. See `switch_offset_on_the_bench`"),
        "what_it_settles": [
            "the deck's audio-to-video head offset, once and for ever",
            "the capture skew, and whether the shared clock held",
            "whether a dropout is common to the audio and video heads",
            "the AFM radio-frequency level against the luma's, which is "
            "the bridge the crosstalk envelope has always been missing",
            "whether the sixth axis of the fold can be filled at all",
        ],
        "what_it_does_not_settle": (
            "whether the audio carriers leak into the video track. That is "
            "already answered - see `carrier_route_bound` - and this "
            "capture would only make the answer's reference level "
            "measurable"),
    }


# --------------------------------------------------------------------------
# The wide re-measurement, made when the subjects were corrected
# --------------------------------------------------------------------------

# Measured 2026-09-06/07 with `hifi_carriers.carrier_presence` and
# `hifi_carriers.detection_bound`, using the call convention that produced
# `hifi_carriers.MEASURED` (whole capture, demeaned; resolution
# 2 * CARRIER_TOLERANCE_HZ / SPECTRUM_BINS_PER_TOLERANCE = 833.33 Hz, bin
# 762.94 Hz; the colour-under harmonic excluded by name where it falls
# inside the hump band). The published SP ntc7composite row was reproduced
# to four figures first (-0.5327 / +0.1208 line, +7.5867 / +4.1088 hump)
# and the run refused otherwise.
#
# WHY THIS EXISTS. The original bound was written over five records, two of
# which cannot carry a Hi-Fi carrier at all, and it sampled ONE programme
# of the thirty-two zaroff playback captures. Correcting the subjects meant
# re-measuring, and widening the subject set from two records to
# thirty-two changed three things.
WIDE_SWEEP: Dict[str, object] = {
    "date": "2026-09-07",
    "subjects": "all 32 zaroff playback captures, 64 (file, channel) pairs",
    "control": "all 32 zaroff record-current captures, a further 64 pairs",

    # 1. THE NARROW-LINE STATISTIC IS NOT SAFE, and the wide sweep is what
    # exposed it. It normalises a peak by the SCATTER of its own
    # neighbourhood, so a flat neighbourhood manufactures significance out
    # of an ordinary peak.
    "line_sigma_range_playback": (-1.23, 12.40),
    "line_sigma_over_threshold_playback": 10,
    "line_sigma_range_record_control": (-2.14, 1.11),
    "line_sigma_over_threshold_record_control": 0,
    "why_the_ten_are_not_detections": (
        "all ten are EP records whose surround scatter is 0.81 to 2.18 dB "
        "rather than records with tall peaks - their peaks are 9.5 to "
        "12.1 dB over their own floor, BELOW the record-tap control's "
        "median of 13.35 dB. The AFM-free control routinely shows 13 to "
        "27 dB bumps in the same window and never fires, because its "
        "scatter is 3.8 to 9.0 dB. The statistic is reading surround "
        "smoothness, not carrier presence"),
    "ensemble_test": (
        "averaging the 16 captures of each (tap, speed) group, "
        "playback-EP channel 1 peaks +9.57 dB at 1297760 Hz while the "
        "AFM-free record-SP channel 1 peaks HIGHER at +15.70 dB and "
        "record-EP at +12.85 dB. No candidate survives the control"),

    # 2. A NAMED CONFOUND IS MISATTRIBUTED. The prior work names the
    # colour-under's second harmonic at 1.258741 MHz as the confound 41 kHz
    # below channel 1. The y-only recordings are the experiment that tests
    # it, and it fails.
    "colour_under_fundamental_removed_db": (-29.57, -10.12),
    "colour_under_fundamental_removed_mean_db": -18.93,
    "second_harmonic_removed_mean_db": -0.14,
    "second_harmonic_verdict": (
        "removing 18.93 dB of colour-under fundamental removes 0.14 dB at "
        "1.258741 MHz, where a square-law product would have lost about "
        "38 dB. WHAT SITS THERE IS NOT THE COLOUR-UNDER SECOND HARMONIC - "
        "it is the luma lower-sideband continuum and the tape's own noise, "
        "present whether or not the record carries chrominance. The "
        "confound named in docs/HIFI_CAPTURE_PATH.md is therefore "
        "misattributed, and excluding that frequency by name buys nothing"),
    "y_only_floor_advantage_db": {"channel_1": -0.25, "channel_2": +0.24},

    # 3. THE OPERATIVE STATISTIC REVERSES, which makes the bound weaker.
    "planted_bounds_db": {
        "zaroff-ntc7composite-NTSC-SP-...-rf-pb": {
            1: {"line": -45.0, "hump": -40.0},
            2: {"line": -35.0, "hump": -35.0}},
        "zaroff-ntc7composite-y-only-NTSC-SP-...-rf-pb": {
            1: {"line": -40.0, "hump": -45.0},
            2: {"line": -35.0, "hump": -35.0}},
    },
    "compander_test": (
        "planting at the -45 dB line bound on the mandatory subject and "
        "requiring the 3 sigma rise: an UNMODULATED carrier is detected "
        "(line rise +3.033), and 500 Hz of deviation is not (+2.744). "
        "Clause 5.6's 2:1 logarithmic compression turns an input 84 dB "
        "below the rotation point into exactly that 500 Hz, so the "
        "narrow-line bound survives only for a carrier that is perfectly "
        "unmodulated. The hump statistic is nearly indifferent to "
        "deviation over the whole range 0 to 50 kHz (rise 2.556 to 2.697)"),
    "conservative_bound_db": {1: -40.0, 2: -35.0},
    "best_hump_bound_db": {1: -45.0, 2: -35.0},
    "best_hump_bound_from": "zaroff-ntc7composite-y-only-NTSC-SP-...-rf-pb",
    "verdict": (
        "no AFM carrier is attributable to any zaroff playback capture. "
        "The bound is the HUMP bound, not the narrow-line one the prior "
        "write-up quoted: -40 dB on channel 1 for the chroma-bearing SP "
        "record and -45 dB for the y-only one, -35 dB on channel 2. "
        "Channel 2 is 10 dB weaker than channel 1 throughout"),
}


def tape_recurrence_lag(tape_speed: str = "SP", system: str = "NTSC",
                        field_period_s: Optional[float] = None
                        ) -> Dict[str, object]:
    """THE DISCRIMINATOR THAT SEPARATES A TAPE MARK FROM A DECK MARK, and
    the thing that makes the dropout method work at all.

    A defect fixed to the TAPE does not recur one field later at the field
    period. The tape has crept forward while the drum turned, so the next
    head has to travel that much further along its own scan before it
    reaches the same piece of oxide:

        L1 = T_field * (1 + v_tape / (v_writing * cos(theta)))

    with theta the track angle, which is itself set by the geometry:
    sin(theta) = pitch / (v_tape * T_field). A defect fixed to the DECK or
    to the PICTURE - a head-switch transient, a vertical-interval event, a
    servo artefact - recurs at exactly T_field instead. The two differ by
    96.5 microseconds at SP against a 16.68 millisecond period, which is a
    clean separation and costs nothing to apply.

    MEASURED, and this is what turns it from a proposal into a method. On
    two consumer tapes the pair-lag histogram is a single clean peak at
    +92 to +100 us with NOTHING at zero: `home` shows 159 coincidences at
    L1 against a uniform-random null of 0.20 +- 0.46 over 500 draws, and
    `countdown` 39 against 0.02 +- 0.13. The EP capture is the independent
    confirmation - its predicted offset is three times smaller because the
    tape crawls three times slower, and the peak lands at +32.5 us. The
    tape-speed ratio is written into the answer.

    AND IT PAYS FOR ITSELF IMMEDIATELY. On `home`, 88.2 per cent of what
    the shipped detector calls a dropout is a once-per-field event at a
    fixed field phase - it recurs at exactly T_field and is not a tape
    defect at all. Any method that took those for tape marks would be
    aligning to the deck. This test removes them for free.
    """
    spec = SPECIFICATION
    r = rates(system)
    period = float(field_period_s) if field_period_s else r["field_period_s"]
    from vhsdecode.models import vhs_specification as _spec
    speeds = {"SP": "tape_speed_sp_m_s", "LP": "tape_speed_lp_m_s",
              "EP": "tape_speed_ep_m_s"}
    key = speeds.get(str(tape_speed).upper())
    if key is None:
        raise ValueError(f"{tape_speed!r} is not a specified tape speed")
    v_tape = float(_spec.SMPTE_32M[key]["value"])
    v_write = float(_spec.SMPTE_32M["writing_speed_m_s"]["value"])
    v_tape_sp = float(_spec.SMPTE_32M["tape_speed_sp_m_s"]["value"])
    # THE PITCH FOLLOWS THE TAPE SPEED, and getting this wrong is what
    # makes the track angle come out different at each speed when it must
    # not. The heads sweep the same path whatever the tape does, so a tape
    # running three times slower lays its tracks three times closer: the
    # pitch scales with the tape speed and the ANGLE is a constant of the
    # drum. Table 2 confirms it, giving 5 deg 58 min 9.9 sec for the moving
    # tape and 5 deg 56 min 7.4 sec for a stationary one - two arc minutes
    # apart across the whole range of speeds.
    pitch = (float(_spec.SMPTE_32M["track_pitch_sp_m"]["value"])
             * v_tape / v_tape_sp)
    ratio = pitch / (v_tape * period)
    theta = math.asin(max(-1.0, min(1.0, ratio)))
    lag = period * (1.0 + v_tape / (v_write * math.cos(theta)))
    return {
        "tape_speed": str(tape_speed).upper(),
        "field_period_s": period,
        "track_angle_deg": math.degrees(theta),
        # table 2's own figure, as a check on the derivation: 5 deg 58 min
        # 9.9 sec for the moving tape. The derived angle sits 0.014 deg
        # above it, which is the nominal 58 um pitch's rounding (table
        # note 1: "Values are nominal where there are no tolerances
        # specified") and is 0.2 per cent - far below anything the
        # discriminator cares about.
        "table_2_track_angle_deg": 5.0 + 58.0 / 60.0 + 9.9 / 3600.0,
        "tape_locked_lag_s": lag,
        "deck_locked_lag_s": period,
        "discriminator_s": lag - period,
        "discriminator_in_lines": (lag - period) / r["line_period_s"],
        "why": ("a defect fixed to the tape recurs at the longer lag "
                "because the tape crept while the drum turned; one fixed "
                "to the deck recurs at the field period exactly"),
        "note": ("`field_period_s` should be MEASURED per capture rather "
                 "than taken from the format, because the discriminator is "
                 "96.5 us on a 16.68 ms period and a 6 parts-per-thousand "
                 "error in the period would swallow it"),
    }


# --------------------------------------------------------------------------
# What was measured here, and on what
# --------------------------------------------------------------------------

# Measured 2026-09-06/07 on the raw radio frequency of four captures, with
# the shipped detector's own state machine (`vhsdecode/doc.py`
# `detect_dropouts_rf` / `find_dropouts_rf`, threshold a fraction of the
# envelope, hysteresis 1.25, merge 30 samples, minimum 10) reproduced on a
# 2.0-6.0 MHz band-pass and an analytic-signal envelope, with the field
# mean replaced by a running median over one field period so that no field
# locking is needed and the dropouts do not bias their own threshold.
#
# EVERY FIGURE HERE IS VIDEO HEAD AGAINST VIDEO HEAD. There is no audio
# capture on this machine, so the audio-against-video case is REFUSED; see
# `refusals`. What these establish is the necessary condition - that a
# defect is a mark shared between two head passes at all - and the
# precision one such mark carries.
MEASURED: Dict[str, object] = {
    "date": "2026-09-07",
    "captures": {
        "sp75": "zaroff-75bars-NTSC-SP-...-rf-pb.flac, 50 MSps, 0.480 s",
        "ep75": "zaroff-75bars-NTSC-EP-...-rf-pb.flac, 50 MSps, 0.480 s",
        "countdown": "/testdata/countdown.flac, 40 MSps, 300-360 s",
        "home": "/testdata/home.flac, 40 MSps, 20-80 s",
    },

    # THE RESULT. A defect IS a common mark between two head passes.
    "coincidence": {
        "home": {"observed": 159, "null_mean": 0.20, "null_sd": 0.46,
                 "null_max_of_500_draws": 3, "z": 345.0,
                 "recurrence_fraction": 0.3596, "se": 0.0293},
        "countdown": {"observed": 39, "null_mean": 0.02, "null_sd": 0.13,
                      "null_max_of_500_draws": 1, "z": 311.0,
                      "recurrence_fraction": 0.3364, "se": 0.0553},
        "at_the_field_period_instead": ("zero on every capture, -1.3 to "
                                        "-0.2 sigma: the coincidence is at "
                                        "the tape-locked lag and not at "
                                        "the deck-locked one"),
        "ep_confirmation": ("the EP capture's predicted offset is three "
                            "times smaller because the tape crawls three "
                            "times slower, and the peak lands at +32.5 us "
                            "against a prediction of 32.16"),
    },

    # THE PRECISION ONE MARK CARRIES, by cross-correlating the two heads'
    # full-rate envelope traces of the same defect.
    "mark_precision": {
        "correlation_median": {"countdown": 0.937, "home": 0.881},
        "robust_sigma_s": {"countdown": 1.70e-6, "home": 1.73e-6},
        "robust_sigma_best_s": {"countdown": 0.71e-6, "home": 1.22e-6},
        "in_tape_micrometres": {"countdown": 9.8, "home": 10.1},
        "per_head_systematic_s": {"countdown": -0.876e-6, "home": -0.913e-6},
        "systematic_is": ("common to both head orders and agrees between "
                          "two unrelated tapes to 0.04 us, so it is a "
                          "fixed deck constant and is calibratable"),
        "noise_floor_s": (27e-9, 64e-9),
        "floor_note": ("the Cramer-Rao bound on the envelope is 27 to "
                       "64 ns, forty to sixty times below what is "
                       "achieved, so the 1.2-1.7 us is PHYSICAL - a "
                       "different cut through the defect, plus the "
                       "transport's own time base - and not measurement "
                       "noise"),
        "yield_per_second": {"countdown": 0.65, "home": 2.65},
    },

    # THE TRAP, which is the reason `tape_recurrence_lag` exists.
    "deck_locked_population": {
        "home": 0.882, "ep75": 0.177, "countdown": 0.051, "sp75": 0.0,
        "what_it_is": ("a once-per-field event at a fixed field phase, "
                       "spread 9.7 us, median 5.17 us long - it recurs at "
                       "exactly the field period and is not a tape defect. "
                       "A method that took these for tape marks would be "
                       "aligning to the deck"),
    },

    # THE DEFECTS' SHAPE, which says what the method has to work with.
    "defect_shape": {
        "recurrence_at_one_track": (0.30, 0.36),
        "recurrence_at_two_tracks": (0.027, 0.054),
        "recurrence_at_three_tracks": "consistent with zero, -1.6 to +2.5 sigma",
        "verdict": ("point defects on the scale of the track pitch, not "
                    "creases spanning many tracks"),
        "along_track_extent_um": {"median": (60.0, 119.0),
                                  "p90": (273.0, 448.0)},
        "edge_steepest_ramp_s": (260e-9, 820e-9),
        "instrument_step_response_s": (220e-9, 240e-9),
        "shipped_envelope_filter_step_response_s": 810e-9,
        "edge_note": ("the steepest part of a dropout edge is 1.1 to 3.6 "
                      "times the instrument's own step response, so "
                      "dropout edges are at the bandwidth limit the "
                      "carrier itself sets - one cycle at 3.8 MHz is "
                      "263 ns"),
    },

    "select_for": ("recurrence rises steeply with severity: 69-71 per cent "
                   "for the deepest events and 67-75 per cent for those "
                   "over 100 us, against 13-18 per cent for events under "
                   "10 us. Selecting on depth or length is what turns a "
                   "one-in-three mark into a two-in-three one"),

    # A SIDE FINDING, recorded because it bears on a figure this tree
    # already flags rather than because it was looked for.
    # THE DRUM-PHASE MARKER, measured on the same day and the same
    # material. Two markers were tried and one of them is rejected.
    "drum_phase": {
        # REJECTED. The drum's line in the RF envelope is found easily
        # enough - 1429 to 7044 times the local background at 0.48 s, and
        # 84 044 to 383 691 times at 30 s - and it is still not a usable
        # clock. Its analytic phase error is 41 to 92 us at 0.48 s and 6 to
        # 13 us at 30 s, but its SPLIT-HALF reproducibility is 60 to 300 us,
        # four to twenty times worse, because a small error in the fitted
        # frequency becomes phase drift. The reason is structural: the line
        # is mostly the head-A against head-B level difference (3.2 to 11.3
        # per cent of the mean), which is a two-state SQUARE WAVE, so
        # fitting its fundamental as a sinusoid is systematically
        # compromised. Recorded so it is not proposed again.
        "envelope_line_rejected": {
            "line_over_background": {"0.48 s": (1429, 7044),
                                     "30 s": (84044, 383691)},
            "analytic_sigma_s": {"0.48 s": (40.9e-6, 92.4e-6),
                                 "30 s": (6.05e-6, 12.7e-6)},
            "split_half_disagreement_s": (61e-6, 296e-6),
            "why": ("the line is mostly the head-to-head level difference, "
                    "a two-state square wave rather than a sinusoid"),
            "verdict": "not a competitive alignment clock",
        },
        # THE MARKER THAT WORKS: the playback head switch, located as the
        # maximum-likelihood change point of the log envelope after the
        # channel response and the field-median profile are divided out.
        "playback_switch_residual_s": {
            "zaroff-75bars SP": 365e-9,
            "home": 3642e-9,
            "countdown": 8150e-9,
        },
        "fields": {"zaroff-75bars SP": 28, "home": 363, "countdown": 336},
        "drum_period_s": {"zaroff-75bars SP": 33365.4284e-6},
        "drum_period_se_s": {"zaroff-75bars SP": 19.2e-9},
        "instrument_floor_s": (100e-9, 250e-9),
        "floor_note": ("a gain step of known size planted in the raw radio "
                       "frequency away from any real switch, and the same "
                       "estimator run on it: 100 to 250 ns on a strong "
                       "edge. So the 365 ns is within a factor of 1.5 to 3 "
                       "of the instrument, and is conservative, because a "
                       "planted pure step lacks the real switch's own "
                       "transient"),
        # WHY THE TWO CONSUMER TAPES ARE TEN AND TWENTY TIMES WORSE, and it
        # is not the tapes' fault.
        "switch_position_lines_ahead_of_vsync": {
            "zaroff-75bars SP": 6.334, "home": -1.082, "countdown": 0.00},
        "position_note": ("on both consumer tapes the PLAYBACK switch lands "
                          "on or just after the vertical sync leading edge, "
                          "where the broad pulses dominate the envelope and "
                          "there is nothing to see a step against. On the "
                          "deck's own tape it sits six lines clear. SMPTE "
                          "32M 3.6 permits 5 to 8 lines, so the deck that "
                          "made those tapes was placing its RECORD switch "
                          "elsewhere and the playing deck's own switch "
                          "lands where it lands"),
        # THE PREMISE OF THE WHOLE METHOD, CONFIRMED.
        "switch_minus_vsync_rms_s": {"zaroff-75bars SP": 159.5e-9,
                                     "home": 4846e-9},
        "premise_confirmed": (
            "on the deck's own tape the drum and the tape move together to "
            "within the instrument's floor. On `home` the SWITCH IS "
            "STEADIER IN ABSOLUTE TIME THAN THE VERTICAL SYNC - 3642 ns "
            "against 5202 - which is the expected physics and the "
            "method's whole premise: the playback drum is servo-locked to "
            "the deck's own reference while the tape flutters. A drum-"
            "referenced alignment is therefore better founded than a "
            "tape-referenced one"),
        # AND THE TRANSPORT TERM IS NOT RESOLVABLE.
        "drum_wander": {
            "verdict": "white to the measurement's noise floor",
            "running_mean_over_white_ratio": (0.93, 1.27),
            "upper_bound_s": {"zaroff over 0.48 s": 435e-9,
                              "home over 0.15 to 1.35 s": (1e-6, 2e-6)},
            "consequence": ("no coherent transport term is detected, so "
                            "`offset_precision`'s second term has no "
                            "measured value to carry and the scatter "
                            "averages down as one over the root of the "
                            "count with no floor in sight. A single "
                            "measured drum-phase offset stays valid across "
                            "a capture"),
        },
        "tape_wanders_far_more": (
            "`home`'s vertical sync scatters 5202 ns with excursions to "
            "15 152 ns and 99.2 per cent of its power below 10 Hz, peaking "
            "at 1.9 to 4.8 Hz; `countdown`'s at 0.17 to 1.3 Hz with "
            "running-mean ratios up to 3.8. So an alignment expressed as "
            "DRUM PHASE holds across a capture and one expressed against "
            "anything tape-locked does not"),
        "pooling_is_empirical": (
            "measured by pooling disjoint groups of consecutive fields "
            "rather than assumed: countdown falls 8150 -> 5016 -> 3305 -> "
            "1834 -> 643 -> 315 ns from 1 to 64 fields, BEATING the root-N "
            "projection because the single-field figure is dominated by "
            "outright estimator failures that averaging suppresses"),
        "half_a_second_suffices": (
            "sub-microsecond drum phase on every capture tested, including "
            "the worst"),
        # THE RECORD MARK IS BETTER LOCATED AND IS THE WRONG MARKER.
        "record_mark_residual_s": {"zaroff-75bars SP": 296e-9,
                                   "home": 6929e-9, "countdown": 2602e-9},
        "record_mark_is_not_the_drum": (
            "the colour-under phase reversal is frozen on the tape by the "
            "RECORDING machine, so it reaches an audio capture of the same "
            "pass only THROUGH THE TAPE and not through the shared drum. "
            "It is the more precisely locatable marker - 296 ns against "
            "365 on zaroff with none of 28 rejected against six, and three "
            "times better on countdown - and it is the wrong one for a "
            "drum-based common clock. It belongs to method two's family, "
            "not method one's"),
        "record_mark_traps": (
            "the mark carries an exact HALF-LINE alternation, 0.4991 H "
            "peak to peak, which must be modelled or it swamps everything "
            "(15832 ns without, 296 ns with); and countdown needs a "
            "FOUR-field term rather than a two-field one, because the "
            "recorder's mark moves with the colour frame (19517 -> 2602 ns)"),
        "estimator_failure_mode": (
            "a first version that searched for the largest step anywhere in "
            "the sixteen lines before vertical sync LOCKED ONTO THE "
            "VERTICAL INTERVAL ITSELF on both consumer tapes, giving a "
            "switch-minus-vsync residual of exactly zero. Subtracting the "
            "field-median profile on a vsync-locked grid is what removes "
            "it, and is load-bearing rather than a refinement"),
    },

    "writing_speed_side_finding": (
        "the measured per-field tape offset is 95.55 us against the 96.46 "
        "the specification's numbers give, so the ratio of tape speed to "
        "head longitudinal speed is 0.91 to 0.95 per cent below nominal. "
        "Taking 33.35 mm/s as given, the along-track writing speed comes "
        "out 5.853 to 5.855 m/s against clause 3.1.4's 5.80 - the "
        "direction and roughly the size of the discrepancy "
        "`vhs_specification.writing_speed_m_s` already flags. Only the "
        "RATIO is measured; the two speeds are degenerate here"),
}


# --------------------------------------------------------------------------
# The deck's own switching topology, read from its schematic
# --------------------------------------------------------------------------

# Read 2026-09-07 from `/testdata/test_patterns/vhs/Sony Slv777Hf 778Hf
# 788Hf Schematic.pdf`, board MA-327, rendered at 700 to 4200 dpi and with
# the vector geometry traced segment by segment so that a net's terminal
# COUNT is known rather than its path merely followed by eye. Every name
# below is as printed on the sheet.
#
# WHY IT MATTERS, AND IT DECIDES METHOD ONE. The head-switch method assumes
# that the video and audio head switches are two views of one rotation
# rather than two independent events. On this deck that is not an
# assumption any more:
#
#   * `RF SWP` and `AF SWP` are pins 18 and 19 of ONE microcontroller,
#     IC160 M37777M7A235GP-C, adjacent outputs carrying the same 5 V
#     peak-to-peak 30 Hz square wave (waveforms 17 and 18, which share one
#     photograph);
#   * the machine holds exactly ONE rotation sensor. CN101 from M901 DRUM
#     MOTOR carries `D PG` and `D FG` through R177 and R178 to IC160 pins
#     86 and 87, and NO drum signal of any kind reaches the AFM section -
#     sheet 5/8's complete cross-sheet inventory contains no PG, no FG and
#     no tachometer line;
#   * IC160 runs on two crystals, X160 at 16 MHz (pins 38/39) and X161 at
#     32.768 kHz (pins 41/42), both confirmed running by their waveforms.
#
# So both switches are timed by one controller from one tachometer against
# one crystal. That their offset is therefore a FIRMWARE CONSTANT is an
# inference from the topology and is labelled one; what is verified is that
# nothing else in the drawing could set it.
SWITCHING_TOPOLOGY: Dict[str, object] = {
    "deck": DECK,
    "source": ("Sony SLV-777HF/778HF/788HF schematic, board MA-327, read "
               "2026-09-07 at 700-4200 dpi with the vector geometry traced"),
    "controller": "IC160 M37777M7A235GP-C, 'SERVO/SYSTEM CONTROL', sheet 3/8",
    "video_switch": {
        "origin": "IC160 pin 18, printed 'RF SWP', 2.5 V DC",
        "waveform": "17, 'IC260 (17) REC/PB', 5 Vpp, 30 Hz square",
        "destinations": ["CN261 pin 3 'RF SWP' (test point JL270)",
                         "IC260 pin 17 via R270 22k, the VIDEO REC/PB AMP"],
        "terminals": 3,
        "capturable_at": "CN261 pin 3, with ground on CN261 pin 4",
    },
    "audio_switch": {
        "origin": "IC160 pin 19, printed 'AF SWP', 2.5 V DC",
        "waveform": ("18, qualified '(HiFi STEREO MODEL)', sharing one "
                     "photograph with waveform 17 - the same 5 Vpp 30 Hz "
                     "square"),
        "destinations": ["IC340 pin 1 'HSW', the AFM REC/PB AMP's head "
                         "select - direct trace, no series component",
                         "IC360 pin 40 'AF SW P' with an overbar, the AFM "
                         "AUDIO PROCESS"],
        "terminals": 3,
        "capturable_at": ("JL345 on sheet 1/8 or JL375 on sheet 5/8 - "
                          "SOLDER-SIDE LINK PADS, not a connector. There "
                          "is no AF SWP pin on CN341"),
        "direction_evidence": (
            "the inter-sheet terminal RS7 carries opposite chevrons on the "
            "sheets that send and receive it - leaving 3/8 at IC160 and "
            "entering both 1/8 and 5/8 - and the sheet 3/8 net has exactly "
            "three terminals with a junction dot, so IC160 pin 19 drives "
            "both. `AF ENV` runs the other way, into IC160 pin 9, which is "
            "the master and slave pattern this implies"),
    },
    "afm_head_switch_is_real": (
        "IC340 LA7256 carries two identical amplifier chains on nodes "
        "printed 'Ch1' and 'Ch2' meeting at a changeover switch, and the "
        "control arrow into that switch traces inside the outline to pin 1 "
        "'HSW'. So the AFM playback path DOES alternate between two heads "
        "once a field, and an AFM capture carries that switch"),
    "driven_by": ("AF SWP and NOT RF SWP - the two are separate controller "
                  "outputs, so the audio switch is not simply the video "
                  "one reused"),
    "tachometer": {
        "connector": "CN101 5P to M901 DRUM MOTOR",
        "drum_pg": "pin 3 'D PG' -> wire S7 -> R177 470 -> IC160 pin 86, "
                   "waveform 25, 4.5 Vpp 30 Hz narrow pulses",
        "drum_fg": "pin 4 'D FG' -> wire S6 -> R178 470 -> IC160 pin 87, "
                   "waveform 26, 4 Vpp 360 Hz",
        "count": 1,
        "reaches_the_afm_section": False,
    },
    "controller_time_base": {
        "X160": "16 MHz, pins 38/39, R167 1M, C167 12p, C168 10p, R168 390",
        "X161": "32.768 kHz, pins 41/42, R169 10M, C169 22p, C170 20p",
    },
    "polarity_caution": (
        "IC360 pin 40 is printed 'AF SW P' WITH AN OVERBAR while IC340 pin "
        "1 is plain 'HSW', so the AFM processor appears to expect the "
        "inverse sense. A capture of AF SWP must therefore fix its own "
        "polarity convention rather than assume it matches RF SWP's"),
    "inference": (
        "that the RF SWP to AF SWP offset is a firmware constant with the "
        "16 MHz crystal's stability. This is an inference FROM the "
        "topology - one controller, one tachometer, one crystal, adjacent "
        "output pins - and the schematic does not state it"),
    "not_legible": ("which internal LOGIC port of IC260 pin 17 the video "
                    "switch lands on; the block's port names could not be "
                    "matched to its right-edge ports even at 4200 dpi"),
}


def switch_offset_on_the_bench() -> Dict[str, object]:
    """THE MEASUREMENT THAT NEEDS NO TAPE PASS AND NO RADIO FREQUENCY.

    The head-switch method compares where the switch falls in each capture.
    On this deck both switches are brought out as logic signals - `RF SWP`
    at CN261 pin 3 with ground on pin 4, and `AF SWP` at the solder pad
    JL345 - so their phase difference can be read directly on two channels
    of an oscilloscope with the deck simply in playback. It does not need
    two synchronised converters, a tape of interest, or any radio frequency
    at all.

    WHAT IT GIVES AND WHAT IT DOES NOT. It gives the SWITCH offset, which
    is the quantity `align_by_switches` would otherwise have to infer from
    two noisy envelopes - and it gives it from a five-volt logic edge
    rather than from an amplitude discontinuity in a noisy carrier, which
    is a far better locator. It does NOT give the capture skew, because no
    capture is involved, and it does NOT give the tape-position lead, which
    is `separate_skew_from_lead`'s business.

    ITS REAL VALUE IS AS A CHECK. Measured on the bench and again from a
    paired capture, the two must agree; a disagreement is the evidence that
    the switch locator in one of the two RF envelopes is biased, which is
    exactly the failure that would otherwise go unnoticed.

    THE ONE CARE. `SWITCHING_TOPOLOGY["polarity_caution"]`: IC360's pin is
    printed with an overbar and IC340's is not, so the polarity convention
    has to be fixed by observation rather than assumed to match.
    """
    video = SWITCHING_TOPOLOGY["video_switch"]
    audio = SWITCHING_TOPOLOGY["audio_switch"]
    return {
        "channel_a": video["capturable_at"],
        "channel_b": audio["capturable_at"],
        "both_are": "5 V peak-to-peak 30 Hz square waves from IC160",
        "measures": "the switch offset, directly",
        "does_not_measure": ["the capture skew", "the tape-position lead"],
        "needs": "the deck in playback, and two oscilloscope channels",
        "does_not_need": ["two synchronised converters", "a paired capture",
                          "any radio frequency"],
        "caution": SWITCHING_TOPOLOGY["polarity_caution"],
        "value": ("a locator on a logic edge rather than on an amplitude "
                  "discontinuity in a noisy carrier, and an independent "
                  "check on the switch locator used in each RF capture"),
    }


# The format's own manufacturer guide corroborates the schematic reading,
# which matters because the schematic is one deck and this is the format.
# JVC Video Technical Guide VTG82063 section 7.3.2, fetched 2026-09-07 from
# decode-orc/analogue-video-specifications:
#
#   "The playback signals from the rotary audio heads are sent through the
#    rotary transformers, then preamplified approximately 60 dB and supplied
#    to the channel switcher. This uses the DRUM FLIPFLOP SIGNAL to produce
#    continuous signals."
#
# So the audio channel switcher is driven from the drum on every deck built
# to the guide, not merely on the one whose schematic was read.
JVC_CORROBORATION: Dict[str, str] = {
    "source": "JVC Video Technical Guide VTG82063 section 7.3.2",
    "channel_switcher": ("'This uses the drum flipflop signal to produce "
                         "continuous signals' - the AFM head switch is "
                         "drum-derived at the FORMAT level, not just on the "
                         "deck whose schematic was read"),
    "the_marker": ("'the instantaneous phase of the carrier changes at the "
                   "head switching point. This results in a frequency "
                   "deviation change that appears as pulse type noise in "
                   "the demodulated signal'"),
    "no_blanking": ("'In the case of the video signal, head switching is "
                    "performed outside the picture region. However, since "
                    "blanking is absent from the audio signal, processing "
                    "is required in order to remove the noise.' The audio "
                    "switch is therefore ALWAYS present in the signal - "
                    "there is no interval it can be hidden in"),
    "heads_are_independent": ("7.2.1: 'the video and audio heads are "
                              "independent of each other. First the audio "
                              "signal is recorded and then the video signal "
                              "is recorded on top of the audio signal'"),
    "azimuth": ("7.2.2 gives +-30 degrees for the audio heads against +-6 "
                "for the video, where SMPTE 32M 5.4 gives +-30 deg 30 min. "
                "Both are carried; the standard's is the normative one"),
    "silent_on_the_offset": ("the guide states the audio head's azimuth, "
                             "its depth, its wavelengths and its switching "
                             "source, and says NOTHING about its angular "
                             "position relative to the video heads. So both "
                             "normative sources decline to fix the offset, "
                             "which is what `offset_is_not_specified` "
                             "concludes from the standard alone"),
}


def afm_switch_marker(system: str = "NTSC", channel: int = 1,
                      sample_rate_hz: float = 40e6,
                      amplitude_snr: Optional[float] = None
                      ) -> Dict[str, object]:
    """THE AUDIO HEAD SWITCH IS A PHASE STEP, NOT AN AMPLITUDE EDGE - which
    makes it a better marker than the video side's, not a worse one.

    The guide says so in as many words (VTG82063 7.3.2, quoted in
    `JVC_CORROBORATION`): *"the instantaneous phase of the carrier changes
    at the head switching point"*. Two heads reading the same track cannot
    hand over in phase, because nothing constrains their relative gap
    positions to a fraction of a 4.5 micrometre wavelength, so the carrier's
    phase jumps at the changeover. That is a discontinuity in the ANALYTIC
    signal, and this arc's standing rule is to work complex wherever a phase
    exists.

    WHY IT IS THE BETTER MARKER. The video switch is found from an amplitude
    step in a noisy envelope, and an envelope is a magnitude - it has thrown
    its phase away. A carrier phase step is a discontinuity in a quantity
    that is measured to a fraction of a cycle, and the audio carrier's cycle
    is 769 ns at channel 1 and 588 ns at channel 2. Divided by the
    amplitude signal-to-noise ratio in the usual way, that is tens of
    nanoseconds - and it is bounded below only by the capture's own sample
    period.

    AND IT CANNOT BE HIDDEN. The guide again: *"since blanking is absent
    from the audio signal, processing is required in order to remove the
    noise"*. Decks conceal the switch in the DEMODULATED audio by holding
    the previous sample, but a capture at `CN341` pin 3 is taken at the head
    amplifier's output, ahead of that processing, so the step is intact in
    the raw radio frequency. There is no interval in an audio track for the
    switch to hide in, which is exactly the property a timing marker wants.

    The returned `timing_sigma_s` is `carrier_period / amplitude_snr`, the
    same threshold-crossing law `dropout_timing_precision` uses, floored at
    the capture's sample period because no estimator resolves a
    discontinuity finer than its own grid without interpolation. Passing no
    signal-to-noise ratio returns the unfloored law's inputs and no figure,
    because a precision quoted without a measured ratio would be invented.
    """
    centres = SPECIFICATION["carrier_centres_hz"]["value"]
    if int(channel) not in (1, 2):
        raise ValueError("SMPTE 32M clause 5.5 specifies two audio "
                         f"channels; {channel!r} is not one of them")
    centre = float(centres[int(channel) - 1])
    period = 1.0 / centre
    grid = 1.0 / float(sample_rate_hz)
    out: Dict[str, object] = {
        "carrier_hz": centre,
        "carrier_period_s": period,
        "sample_period_s": grid,
        "samples_per_carrier_cycle": float(sample_rate_hz) / centre,
        "marker": "a step in the carrier's instantaneous phase",
        "source": JVC_CORROBORATION["the_marker"],
        "always_present": JVC_CORROBORATION["no_blanking"],
        "law": "sigma_t = carrier_period / (A / sigma_A), floored at the "
               "sample period",
    }
    if amplitude_snr is None:
        out["timing_sigma_s"] = None
        out["refused"] = ("no signal-to-noise ratio was given, and there is "
                          "no audio capture on this machine to measure one "
                          "from. A precision quoted without it would be "
                          "invented")
        return out
    snr = float(amplitude_snr)
    if snr <= 0.0:
        raise ValueError("a signal-to-noise ratio is positive")
    out["amplitude_snr"] = snr
    out["timing_sigma_s"] = float(max(period / snr, grid))
    out["limited_by"] = ("the capture grid" if period / snr < grid
                         else "the carrier and the noise")
    return out
