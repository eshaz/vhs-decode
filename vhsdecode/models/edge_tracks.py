"""THE LINEAR AUDIO AND CONTROL TRACKS, AS SEEN BY THE VIDEO HEAD.

Ethan, 2026-09-06: *"On VHS there are other tracks contained on the tape.
They may overlap with the video track to some degree. These are the linear
audio and control tracks. I believe I can use the geometry of the video
heads, and helical scan to extract out data from these potentially
overlapping tracks, these may be useful for synchronizing audio to video."*

THE IDEA IS SOUND AND THE GEOMETRY REFUSES IT, and the refusal is worth
more than a vague yes because it is a number. SMPTE 32M table 2 fixes the
whole transverse layout, and the layout closes exactly: the control track
occupies 0 to 0.75 mm from the reference edge, the longitudinal audio track
11.65 to 12.65 mm, and the video head's recording area is 10.60 mm wide
centred at 6.20 mm - which is, to the micrometre, the midpoint of the span
the two edge tracks leave free. So the head stops 150 um short of each edge
track at BOTH ends of its sweep. There is no overlap. The premise of the
idea - "they may overlap to some degree" - is false at nominal dimensions,
and no tolerance the standard states closes the gap: stacked worst case it
narrows to 50 um and does not vanish.

WHAT REMAINS IS FRINGING, and that is a real mechanism, not a dodge. A
magnetisation of wavelength lambda produces a field outside the medium
that satisfies Laplace's equation, and in the transverse plane that is
(d2/dy2 + d2/dz2 - k2) phi = 0 with k = 2 pi / lambda, whose Green's
function K0(k r) decays as exp(-k r). The decay length is lambda / 2 pi in
EVERY direction away from the source, sideways included. So the 150 um
guard is not a wall, it is an exponential attenuator whose strength depends
entirely on wavelength - and the edge tracks were written at 33.35 mm/s, so
their wavelengths are enormous.

THE SPEED RATIO IS THE WHOLE REASON THIS IS TESTABLE. The head crosses the
tape at 5.80 m/s along a track inclined 5 deg 58' 9.9" to the tape's length,
so its longitudinal component is 5.7686 m/s against the 33.35 mm/s at which
the edge tracks were laid down: a ratio of 172.970. The control track's one
pulse per frame therefore returns at 5183.9 Hz, and its spatial period of
1.112778 mm becomes 1.11881 mm along the head's path - a decay length of
178.1 um, or 4.64 lines of transverse travel, 1.871 dB a line. At the 150
um guard that is exp(-0.842) = 0.431, only 7.3 dB down. The control track
is the ONE edge signal whose wavelength is long enough to fringe across the
guard at all.

The audio track is the opposite case. Fringing at 150 um costs 8.686 dB per
decay length, so it admits only on-tape components below 245.8 Hz at -60 dB
- a 1 kHz tone is 244.1 dB down before anything else touches it. Whatever
the video head can see of the linear audio, it is the rumble and nothing
else, and it arrives spread from zero to 42.5 kHz.

AZIMUTH IS NAMED IN THE PREMISE AND IS NOT THE LIMIT, which is worth
saying because it inverts the expected discriminator. The +-6 degree head
reading a 0 degree linear track loses |sinc(w tan alpha / lambda)|, and
with a 58 um track width the azimuth scale length is 6.096 um: the first
null falls at a reproduced 951.5 kHz, an on-tape 5.50 kHz. At everything
the guard admits the azimuth loss is 0.99995. Lateral separation refuses
this idea; azimuth never gets a chance to.

AND THE HEAD IS A DIFFERENTIATOR, which is what closes the case. Reproduce
voltage goes as dPhi/dt, so 5183.9 Hz arrives 57.0 dB below the 3.6857 MHz
blanking carrier from that factor alone, before the guard's 7.3 dB and
before the deck's playback path - which SMPTE 32M does not specify at all
(`vhs_specification.SMPTE_32M["playback_response"]`: "ABSENT ENTIRELY.
Every video response clause is record-side"). The two terms this module can
compute give an upper bound of -64.4 dBc, and it is generous: it credits
the control track with the coupling an on-track read would give and charges
nothing for the deck's band-pass around the FM carrier.

MEASURED, and this is the part that makes it evidence rather than
arithmetic. The estimator below is a matched filter in three variables at
once - position along the sweep (the fringing exponential anchored at the
head switch), frequency (the reproduce frequency the ratio predicts), and
field parity (the tape advances 0.556389 mm per field against a control
period of 1.112778 mm, EXACTLY half, so a control-track fringe must reverse
sign every field while nothing locked to the drum does). Run by
`tools/ringing_measure/edge_tracks_measure.py` over 442 fields of the
sixteen `zaroff-*-NTSC-SP` playback captures against 439 fields of the
sixteen matched RECORD-tap captures, which carry the drive going to the
head and can hold nothing read off the tape.

THE POSITION SCAN IS WHAT DECIDED IT, at 5183.9 Hz with a four-line probe,
alternating, in dB relative to the carrier's peak:

    line from V-sync   playback   record tap   what is there
        -30              -91.4      -89.0      nothing, either tap
        -22              -62.9      -94.9
        -18              -59.3     -104.0      a hump, PLAYBACK ONLY
        -14              -64.5      -96.7
        -10              -72.2      -90.8
         -6              -39.3      -66.8      the head switch, both taps
         -2              -85.3      -77.8      the tape edge: nothing
      field median       -79.0      -90.5

A CONTROL-TRACK FRINGE MUST BE LARGEST AT THE TAPE EDGE and fall inward at
1.871 dB a line. What is there is the opposite: the playback tap's
alternating energy near 5183.9 Hz is at its WEAKEST at the edge, -85.3 dBc
at line -2, and peaks eighteen lines earlier, in the middle of the picture.
The arithmetic refuses the identification rather than merely disliking it:
by line -18 the leaving head is 0.449 mm past the edge even on the most
favourable anchoring, where a fringe is 21.9 dB below its own edge value,
so the hump implies -37.4 dBc at the edge - 27 dB ABOVE the generous
bound. It cannot be an edge track. See `distance_to_track_m`.

THE BOUND, then, is the reading at line -2, the closest line to the tape
edge that the head switch does not occupy: -85.3 dBc with a two-sigma
bound of -85.5 dBc, against a prediction there of -67.2 dBc - the -64.4
bound plus the 2.8 dB of extra guard that a line and a half of travel
buys. The measurement sits 18.3 dB below the most generous level the
geometry allows, and found nothing.

THE OTHER CONTROLS, in the order they were run. The RECORD TAP is flat at
-80 dBc from 3.4 to 7 kHz with every ratio at or below 1.44, and its
position scan is flat everywhere but the switch - so the mid-picture hump
is read off the tape and is not in the drive. MID-FIELD: four millimetres
from any edge track, where the fringing factor is 10^-11.6, the filter
still read within 8 dB of the sweep end, which is how the tap's own
low-frequency noise was caught impersonating a fringe. And TAPE SPEED,
which was to have been the decisive control - EP writes the same one pulse
per frame at 11.12 mm/s, so the line must move to 15628.5 Hz - IS
INCONCLUSIVE AND IS REPORTED AS SUCH: over 442 EP fields nothing appears at
15628.5 Hz, but the EP captures' own floor there is -54.2 dBc, above the
level an SP-sized fringe would reach, so those captures cannot test their
own prediction.

ONE ESTIMATOR DEFECT IS RECORDED HERE because it changed an answer. The
first templates were normalised to unit norm and not orthogonalised against
a constant. A one-sided exponential of time constant 295 us still passes
zero frequency at a tenth of its peak gain, so the filter read whatever
offset sat under the window: it returned -46.6 dBc at EVERY frequency from
1 to 30 kHz and at EVERY line from the switch to line +24, to a tenth of a
decibel, and that was very nearly written down as a detection at the
predicted frequency. `matched_reference` now subtracts the template's mean,
which costs 0.03 per cent of the profile's norm and makes the reading blind
to an offset. The unit-norm scaling was the second half of the same
mistake: it understates a level by the ratio of the window to the profile,
13.8 dB at SP, and tilts a frequency sweep by 3 dB an octave.

THE VERDICT IS A BOUNDED NEGATIVE: no linear-audio and no control-track
signal at the video tap, on three tapes and two tape speeds, to -85.5 dBc
at the tape edge on the deck's own tape - 18.3 dB below the most generous
level the geometry allows. The one thing that looked like a find is 27 dB
too strong for its own position. See `docs/EDGE_TRACKS.md` for the full
record.

ONE THING THE NEGATIVE DOES NOT COVER, and it is stated rather than
glossed: the head's closest approach to each edge track happens inside the
head-switch overlap, where the tap shows the other head. `visible_window`
computes it - the two heads are both on the tape for 0.53 mm, 13.82 lines,
and the electronic switch sits somewhere inside that - and those are
exactly the lines where a fringe would be strongest. Worse, the lines that
follow the switch, where the entering head is nearest the control track,
are the vertical interval, whose own structure is the largest
field-alternating thing in the record. The bound above is over the lines
that are VISIBLE and quiet. A capture of one head's preamp output taken
AHEAD of the switching amplifier would remove that reservation, and is the
one capture that could still change this answer.

THE ANSWER TO WHAT HE ASKED FOR ANYWAY - audio-to-video synchronisation -
does not need any of this, and the standard hands it over directly.
`audio_displacement` reads clauses 3.4 and 3.5: the audio and control head
sits 79.244 mm downstream of the end of the video head's 180 degree scan,
so program audio time-coincident with a point on the video track is written
79.244 mm further along the tape, which at 33.35 mm/s is 2.376132 s, or
71.213 frames. That is an exact, specified, machine-independent offset
between a separately captured linear audio track and the video record, its
precision set by the tape speed's own +- 0.5 per cent, and it is worth more
than the fringe would have been.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import sync_geometry

__all__ = [
    "TRACK_GEOMETRY", "MEASURED", "layout", "speed_ratio",
    "reproduce_hz", "control_track_reproduce_hz", "apparent_wavelength_m",
    "fringing_decay_length_m", "fringing_factor", "azimuth_loss",
    "admitted_tape_band_hz", "differentiation_db", "geometric_upper_bound_db",
    "sweep_position_m", "distance_to_track_m", "visible_window",
    "matched_reference", "project", "Projection", "frequency_sweep",
    "excess_over_trend", "audio_displacement", "synchronisation_precision",
]


def _entry(value, unit: str, clause: str, quote: str, tolerance=None,
           note: str = "") -> Dict[str, object]:
    return {"value": value, "unit": unit, "clause": clause, "quote": quote,
            "tolerance": tolerance, "note": note}


# --------------------------------------------------------------------------
# The transverse layout. SMPTE 32M table 2 states every one of these and
# `vhs_specification` carries none of them - it took the head's parameters
# and the signal's, and the tape's cross-section was never asked for.
# --------------------------------------------------------------------------

TRACK_GEOMETRY: Dict[str, Dict[str, object]] = {
    "tape_width_m": _entry(
        12.65e-3, "m", "32M table 2 (A), 32M 3.2.1.2",
        "A, Tape width, 12.65 +- 0.01 ... Width fluctuation shall not "
        "exceed 6 um", tolerance=0.01e-3,
        note="JVC VTG82063 table 1-1-1 item 1 states the same."),
    "video_recording_area_m": _entry(
        10.60e-3, "m", "32M table 2 (B)", "B, Video recording area width, 10.6",
        note="NOMINAL, no tolerance. This is the transverse width the head "
             "actually records over, wider than the 180 degree scan by the "
             "head-switch overlap. JVC table 1-1-1 item 6 calls it 'Video "
             "Width'."),
    "video_effective_area_m": _entry(
        10.07e-3, "m", "32M table 2 (W)", "W, Video recording effective area, 10.07",
        note="One field's worth: 10.07 mm at the 5.9694 degree track angle "
             "is a track 96.76 mm long, which is 5.80 m/s for 16.683 ms - "
             "the 180 degree scan exactly. The 0.53 mm by which B exceeds "
             "it is the overlap."),
    "control_track_width_m": _entry(
        0.75e-3, "m", "32M table 2 (C)", "C, Control track width, 0.75 +- 0.10",
        tolerance=0.10e-3),
    "audio_track_reference_line_m": _entry(
        11.65e-3, "m", "32M table 2 (F)", "F, Audio track reference line, 11.65 +- 0.05",
        tolerance=0.05e-3,
        note="The audio track's edge nearest the video area. 11.65 plus the "
             "1.0 mm monophonic width is 12.65, the tape width, which is "
             "how this reads as an edge and not a centre line. JVC table "
             "1-1-1 item 14 gives the tighter +- 0.03."),
    "audio_track_width_mono_m": _entry(
        1.0e-3, "m", "32M table 2 (R)", "R, Audio track width (monophonic), 1.0 +- 0.1",
        tolerance=0.1e-3,
        note="Stereophonic is two 0.35 mm tracks with a 0.3 mm guard "
             "between them (D, E, H), which spans the same 1.0 mm, so the "
             "edge nearest the video area is 11.65 either way."),
    "video_track_centre_m": _entry(
        6.20e-3, "m", "32M table 2 (L)",
        "L, Video track center from reference edge of tape, 6.2",
        note="THE CONSISTENCY CHECK THAT FIXES WHICH EDGE IS WHICH. The "
             "span the edge tracks leave free runs 0.75 to 11.65 mm and its "
             "midpoint is 6.20 mm exactly. Put the audio at the reference "
             "edge instead and the midpoint is 6.45. So the control track "
             "is at the reference edge and the audio track at the far edge, "
             "and that is arithmetic from the standard's own numbers rather "
             "than a reading of figure 7."),
    "track_angle_running_rad": _entry(
        (5 + 58/60 + 9.9/3600) * math.pi / 180, "rad", "32M table 2 (theta)",
        "theta, Video track angle, 5 deg 58' 9.9\"",
        note="SP. WITH the tape running, which is the one that matters "
             "here: the edge tracks are fixed in tape coordinates, so the "
             "relative motion is what sets the reproduce frequency. The "
             "stationary-tape angle theta_0 is 5 deg 56' 7.4\" and is the "
             "same for every speed. Per-speed values in `SPEED_GEOMETRY`."),
    "azimuth_tolerance_rad": _entry(
        10.0 / 60.0 * math.pi / 180, "rad", "JVC VTG82063 table 1-1-1 item 18",
        "(alpha) Video Head Gap Azimuth Angle, 6 deg +- 10'",
        note="SMPTE 32M table 2 gives the +6/-6 with NO tolerance; the JVC "
             "guide supplies the +- 10 arc-minutes. Recorded here because "
             "`vhs-mechanical-spec` notes this project's fitted azimuth "
             "exceeds it."),
    "audio_control_head_position_m": _entry(
        79.244e-3, "m", "32M 3.4 / table 2 (X)",
        "The distance X on the tape pattern, from the end of the 180 degree "
        "scan of a video head to the audio and control track head position, "
        "shall be 79.244 mm",
        note="SP; the standard gives 79.251 for LP and 79.253 for EP. "
             "Clause 3.5 turns it into a time: audio time-coincident with a "
             "point on video track 2 is recorded X downstream of it."),
    "control_pulse_per_frame": _entry(
        1, "1", "JVC VTG82063 1.1.21",
        "Phase of the control signal is the same as the vertical sync "
        "signal rise component of the CH-1 track. The positive pulse "
        "voltage is the reference 30 Hz (NTSC) ... The control signal is "
        "recorded on the control track above the saturation recording level.",
        note="SMPTE 32M states no control-signal waveform at all - it fixes "
             "the track's width and position and nothing about what is on "
             "it. The rate, the phase and the above-saturation level are "
             "the JVC guide's."),
    "video_guard_to_edge_track_m": _entry(
        None, "m", None,
        "NOT STATED as a dimension. It is 0.150 mm at both edges by "
        "subtraction - (0.900 - 0.750) below and (11.650 - 11.500) above - "
        "and the standard neither names it nor gives it a tolerance.",
        note="So the single number this whole question turns on is a "
             "DERIVED one, and the tolerances that could move it are the "
             "control track's +- 0.10 mm width and the audio reference "
             "line's +- 0.05 mm. Worst case in both directions the guard "
             "is 0.10 mm, not zero: even stacked, the tolerances do not "
             "produce an overlap."),
    "head_gap_length_m": _entry(
        None, "m", "32M 5.1",
        "NOT STATED. `vhs_specification.SMPTE_32M['gap_length_m']` records "
        "the same absence.",
        note="Which is why `fringing_factor` returns a BOUND and not a "
             "coupling: the absolute off-track sensitivity needs the head's "
             "sensitivity function, and the standard does not supply one."),
}


# The three quantities that change with tape speed, and only those three.
# SMPTE 32M table 3 carries the angle and the track centre for LP and EP;
# it does NOT carry a writing speed for either, and JVC VTG82063 table 1-1-1
# item 4 supplies the EP one. There is no NTSC LP column in the JVC guide,
# so the LP writing speed is refused rather than guessed - which makes LP
# the one speed at which this module cannot state a reproduce frequency.
SPEED_GEOMETRY: Dict[str, Dict[str, object]] = {
    "SP": {
        "track_angle_rad": (5 + 58/60 + 9.9/3600) * math.pi / 180,
        "video_track_centre_m": 6.20e-3,
        "writing_speed_m_s": 5.80,
        "clause": "32M table 2; writing speed 32M 3.1.4",
    },
    "LP": {
        "track_angle_rad": (5 + 57/60 + 8.5/3600) * math.pi / 180,
        "video_track_centre_m": 6.20e-3,
        "writing_speed_m_s": None,
        "clause": "32M table 3; NO writing speed is stated for LP in either "
                  "document, and the JVC guide's NTSC columns are SP and EP",
    },
    "EP": {
        "track_angle_rad": (5 + 56/60 + 8.1/3600) * math.pi / 180,
        "video_track_centre_m": 6.195e-3,
        "writing_speed_m_s": 5.83,
        "clause": "32M table 3; writing speed JVC VTG82063 table 1-1-1 "
                  "item 4, 'EP mode 5.83 m/sec'",
    },
}


def _value(name: str) -> float:
    entry = TRACK_GEOMETRY[name]
    if entry["value"] is None:
        raise KeyError(
            "%s is not stated: %s" % (name, entry["quote"]))
    return float(entry["value"])


# Tape speeds live in `vhs_specification`; they are read from there rather
# than restated, so a correction there reaches here.
def _transport(speed: str) -> Tuple[float, float]:
    from vhsdecode.models import vhs_specification as spec

    name = str(speed).upper()
    key = {"SP": "tape_speed_sp_m_s", "LP": "tape_speed_lp_m_s",
           "EP": "tape_speed_ep_m_s"}.get(name)
    if key is None:
        raise ValueError("speed must be SP, LP or EP, got %r" % (speed,))
    head = SPEED_GEOMETRY[name]["writing_speed_m_s"]
    if head is None:
        raise KeyError("no writing speed is stated for %s: %s"
                       % (name, SPEED_GEOMETRY[name]["clause"]))
    return float(spec.SMPTE_32M[key]["value"]), float(head)


def _line_period_s(system: str = "NTSC") -> float:
    key = "525" if str(system).upper().startswith("NTSC") else "625"
    return sync_geometry.LINE_PERIOD_US[key] * 1e-6


def _frame_hz(system: str = "NTSC") -> float:
    """The control pulse rate: one per frame, JVC 1.1.21."""
    return 1.0 / (_field_lines(system) * 2.0 * _line_period_s(system))


def _field_lines(system: str = "NTSC") -> float:
    return 262.5 if str(system).upper().startswith("NTSC") else 312.5


# --------------------------------------------------------------------------
# The layout, and the guard that decides the whole question
# --------------------------------------------------------------------------


def layout(speed: str = "SP") -> Dict[str, object]:
    """Where everything sits across the tape, in metres from the reference
    edge, with the guard that separates the head's sweep from each edge track.

    MEASURED BY ARITHMETIC from SMPTE 32M table 2 and nothing else:

        control track      0.000 .. 0.750 mm
        head's sweep       0.900 .. 11.500 mm   (B, centred on L)
        one field's scan   1.165 .. 11.235 mm   (W, centred on L)
        audio track       11.650 .. 12.650 mm
        guard, both ends   0.150 mm

    The two guards come out equal, which is the check that the reading is
    right: L = 6.20 mm is the midpoint of the free span and the standard
    put it there deliberately.
    """
    width = _value("tape_width_m")
    ctl = _value("control_track_width_m")
    audio_edge = _value("audio_track_reference_line_m")
    centre = float(SPEED_GEOMETRY[str(speed).upper()]["video_track_centre_m"])
    recording = _value("video_recording_area_m")
    effective = _value("video_effective_area_m")
    sweep = (centre - recording / 2.0, centre + recording / 2.0)
    scan = (centre - effective / 2.0, centre + effective / 2.0)
    return {
        "tape_width_m": width,
        "control_track_m": (0.0, ctl),
        "audio_track_m": (audio_edge, width),
        "sweep_m": sweep,
        "scan_m": scan,
        "guard_control_m": sweep[0] - ctl,
        "guard_audio_m": audio_edge - sweep[1],
        "overlap_m": recording - effective,
        "free_span_midpoint_m": (ctl + audio_edge) / 2.0,
    }


def transverse_rate_m_per_line(system: str = "NTSC") -> float:
    """How far across the tape the head moves in one line: W over the lines
    of a field, 38.362 um for NTSC SP. The head's sweep is 276.32 lines long
    where a field is 262.5, the difference being the overlap."""
    return _value("video_effective_area_m") / _field_lines(system)


# --------------------------------------------------------------------------
# The speed ratio, which is the signature
# --------------------------------------------------------------------------


def speed_ratio(speed: str = "SP") -> float:
    """How much faster the head crosses a longitudinal pattern than the head
    that wrote it: 172.970 at SP.

    The head's velocity relative to the tape has magnitude equal to the
    writing speed and lies along the track, at the running track angle to
    the tape's length; the edge tracks run along the tape's length, so it is
    the LONGITUDINAL COMPONENT that matters, 5.80 cos(5.96942 deg) = 5.7686
    m/s. Against 33.35 mm/s that is 172.970, not the 174 a reader gets by
    forgetting the cosine.
    """
    tape, head = _transport(speed)
    angle = float(SPEED_GEOMETRY[str(speed).upper()]["track_angle_rad"])
    return head * math.cos(angle) / tape


def reproduce_hz(tape_hz: float, speed: str = "SP") -> float:
    """What a component recorded at `tape_hz` on an edge track comes back as."""
    return float(tape_hz) * speed_ratio(speed)


def control_track_reproduce_hz(speed: str = "SP", system: str = "NTSC") -> float:
    """The control track's one-per-frame pulse, read by a video head:
    5183.9 Hz at NTSC SP.

    Not an event but a rate: over the 0.29 ms during which the head is
    within a decay length of the track it passes 1.45 control periods, so
    what the head could see is a cycle and a half of a 5.18 kHz oscillation,
    not a single isolated pulse.
    """
    return reproduce_hz(_frame_hz(system), speed)


def apparent_wavelength_m(reproduce_frequency_hz: float,
                          speed: str = "SP") -> float:
    """The spatial period along the head's own path of something the head
    reproduces at this frequency. This, not the period along the tape, is
    what sets the fringing decay: 1.11881 mm for the control track, against
    1.112778 mm along the tape, the two differing by 1/cos(theta)."""
    _tape, head = _transport(speed)
    return head / float(reproduce_frequency_hz)


# --------------------------------------------------------------------------
# Fringing: the only mechanism the geometry leaves open
# --------------------------------------------------------------------------


def fringing_decay_length_m(reproduce_frequency_hz: float,
                            speed: str = "SP") -> float:
    """lambda / 2 pi, the ONLY length in the problem.

    Outside the magnetised layer the scalar potential of a magnetisation
    varying as exp(i k x) satisfies (d2/dy2 + d2/dz2 - k2) phi = 0, whose
    Green's function is K0(k r). K0 falls as exp(-k r), so the field decays
    with length 1/k = lambda / 2 pi away from the source in EVERY direction,
    across the tape exactly as much as above it. There is no second length
    scale to appeal to: 178.1 um for the control track, 46.2 um at 20 kHz
    reproduced, 9.2 um at 100 kHz.
    """
    return apparent_wavelength_m(reproduce_frequency_hz, speed) / (2.0 * math.pi)


def fringing_factor(distance_m: float, reproduce_frequency_hz: float,
                    speed: str = "SP") -> float:
    """An UPPER BOUND on off-track pickup relative to an on-track read.

    exp(-distance / decay length). It is a bound and not a value because
    the K0 kernel carries a further 1/sqrt(k r) that this drops, and
    because the absolute coupling needs the head's sensitivity function -
    which needs the gap length, which the standard does not state
    (`TRACK_GEOMETRY['head_gap_length_m']`). Erring high is the right
    direction for a refutation: a signal absent below a generous bound is
    absent.

    At the 150 um guard: 0.431 for the control track (-7.3 dB), 0.039 at
    20 kHz reproduced (-28.2 dB), 9.2e-8 at 100 kHz (-140.7 dB).
    """
    d = max(float(distance_m), 0.0)
    return math.exp(-d / fringing_decay_length_m(reproduce_frequency_hz, speed))


def azimuth_loss(reproduce_frequency_hz: float, speed: str = "SP",
                 track_width_m: Optional[float] = None,
                 azimuth_rad: Optional[float] = None) -> float:
    """The gap-azimuth loss of a +-6 degree video head reading a 0 degree
    linear track: |sinc(w tan(alpha) / lambda)|.

    NAMED IN THE PREMISE AND NOT THE LIMIT, which is worth saying plainly.
    The head is 58 um wide and tan 6 deg is 0.10510, so the azimuth scale
    length is 6.096 um and the first null falls at a reproduced 951.5 kHz -
    an on-tape 5.50 kHz. Everything the guard admits is far below that:
    at the control track's 5183.9 Hz the loss is 0.99995, six parts in a
    hundred thousand. Lateral separation is what refuses this idea;
    azimuth never gets a chance to.
    """
    from vhsdecode.models import vhs_specification as spec

    if track_width_m is None:
        track_width_m = float(spec.SMPTE_32M["track_width_sp_m"]["value"])
    if azimuth_rad is None:
        azimuth_rad = abs(spec.SMPTE_32M["azimuth_deg"]["value"][0]) * math.pi / 180
    lam = apparent_wavelength_m(reproduce_frequency_hz, speed)
    x = math.pi * track_width_m * math.tan(azimuth_rad) / lam
    return 1.0 if x == 0.0 else abs(math.sin(x) / x)


def admitted_tape_band_hz(distance_m: Optional[float] = None,
                          floor_db: float = -60.0,
                          speed: str = "SP") -> float:
    """The highest ON-TAPE frequency whose fringe clears `floor_db` across
    the guard - 170.5 Hz at the 150 um guard and -60 dB.

    This is the sentence the audio half of the idea reduces to. Solving
    exp(-2 pi d f_tape r / v_head) = 10^(floor/20) for f_tape, with r the
    speed ratio, gives a hard low-pass on what the video head can possibly
    see of the linear audio: at 150 um only the bass below about 170 Hz
    survives at all, and it arrives between 3.5 kHz and 29 kHz. A 1 kHz
    tone is 246 dB down. There is no audio there to synchronise with, only
    its rumble.
    """
    if distance_m is None:
        distance_m = layout(speed)["guard_audio_m"]
    _tape, head = _transport(speed)
    attenuation_nepers = -float(floor_db) / (20.0 * math.log10(math.e))
    return attenuation_nepers * head / (2.0 * math.pi * float(distance_m)
                                        * speed_ratio(speed))


def differentiation_db(reproduce_frequency_hz: float,
                       carrier_hz: Optional[float] = None) -> float:
    """The head is a flux DIFFERENTIATOR, so its output falls with frequency.

    Reproduce voltage is proportional to N eta w dPhi/dt, so relative to
    the luma carrier the ratio is simply the frequency ratio: 5183.9 Hz
    against the blanking carrier is -57.5 dB. This term owes nothing to the
    tape or the guard and cannot be argued away.
    """
    if carrier_hz is None:
        from vhsdecode.models import vhs_specification as spec
        carrier_hz = spec.carrier_hz_for_ire(0.0)
    return 20.0 * math.log10(float(reproduce_frequency_hz) / float(carrier_hz))


def geometric_upper_bound_db(speed: str = "SP", system: str = "NTSC"
                             ) -> Dict[str, float]:
    """The most generous level a control-track fringe could reach at the tap,
    counting only the terms this module can compute from the standard.

    Two terms, both hard: the differentiation, and the guard. Everything
    omitted makes the true figure SMALLER - the deck's playback band-pass
    around the FM carrier, the rotary transformer's own low-frequency
    corner, the K0 kernel's dropped prefactor, and the fact that the head
    reads 58 um of a 750 um track. Nothing omitted makes it larger. So a
    measurement that reaches below this number and finds nothing has
    settled the question for this deck and these tapes.

    NTSC SP: -57.5 dB differentiation, -7.3 dB guard, -64.8 dBc total.
    """
    f = control_track_reproduce_hz(speed, system)
    guard = layout(speed)["guard_control_m"]
    diff_db = differentiation_db(f)
    guard_db = 20.0 * math.log10(fringing_factor(guard, f, speed))
    return {"reproduce_hz": f,
            "differentiation_db": diff_db,
            "guard_db": guard_db,
            "azimuth_db": 20.0 * math.log10(azimuth_loss(f, speed)),
            "total_db": diff_db + guard_db}


# --------------------------------------------------------------------------
# Where the head is, line by line
# --------------------------------------------------------------------------


def sweep_position_m(line: float, anchor_line: float, system: str = "NTSC",
                     speed: str = "SP", head: str = "entering") -> float:
    """The head's transverse position, in metres from the reference edge, at
    `line` lines from the V-sync leading edge.

    TWO HEADS AND TWO ANSWERS, which is why `head` is not optional in
    spirit. After the switch the tap shows the ENTERING head, climbing away
    from the edge it touched down on at `anchor_line`; before the switch it
    shows the LEAVING head, running toward the far edge it lifts off at.
    Feeding a pre-switch line into the entering branch is how a distance
    comes out clamped at the guard for the whole picture, which is wrong by
    millimetres.

    Both are the OPTIMISTIC anchoring: they assume the head touched down or
    lifted off exactly at the switch, when `visible_window` shows it may do
    either up to 13.82 lines away, and be that much further from the edge.
    """
    place = layout(speed)
    per_line = transverse_rate_m_per_line(system)
    if head == "entering":
        return place["sweep_m"][0] + per_line * (float(line) - float(anchor_line))
    if head == "leaving":
        return place["sweep_m"][1] - per_line * (float(anchor_line) - float(line))
    raise ValueError("head must be 'entering' or 'leaving', got %r" % (head,))


def distance_to_track_m(line: float, anchor_line: float,
                        track: str = "control", system: str = "NTSC",
                        speed: str = "SP", head: str = "entering") -> float:
    """How far the head is from an edge track's near edge at a given line.

    MEASURED CONSEQUENCE, and the reason this function exists separately
    from the estimator: the field-alternating component that the frequency
    sweep found near 5183.9 Hz peaks at line -18 to -22, not at the sweep
    edge. At line -18 the leaving head is 0.449 mm past the edge it will
    lift off at, even on the most favourable anchoring, so a fringe there
    is 21.9 dB below its own value at the edge - which would put the edge
    value 27 dB ABOVE `geometric_upper_bound_db`. That is the arithmetic
    that refuses the identification, and it does not depend on which edge
    the head leaves by.
    """
    place = layout(speed)
    y = sweep_position_m(line, anchor_line, system, speed, head)
    if track == "control":
        return max(y - place["control_track_m"][1], 0.0)
    if track == "audio":
        return max(place["audio_track_m"][0] - y, 0.0)
    raise ValueError("track must be 'control' or 'audio', got %r" % (track,))


def visible_window(system: str = "NTSC", speed: str = "SP") -> Dict[str, object]:
    """What the switched tap can and cannot show, and why part of this
    question is REFUSED rather than answered.

    The head's contact is 10.60 mm wide and one field's scan is 10.07 mm,
    so the two heads are both on the tape for 0.53 mm - 13.82 lines - and
    the electronic switch happens somewhere inside that. During the overlap
    the incoming head is at its closest to the control track and the
    outgoing head at its closest to the audio track, and the tap shows only
    one of them. So the strongest samples of the effect are behind the
    switch, and where exactly the switch sits inside the overlap is not
    determinable from a switched capture.

    SMPTE 32M 3.6 puts the switch 5 to 8 lines ahead of the V-sync leading
    edge - a three-line permitted window, itself wider than nothing.
    MEASURED on `zaroff-75bars-NTSC-SP` playback, 27 fields, from the FM
    envelope: the disturbance runs from line -6.3 to -3.5 with its minimum
    at -5.29 and a depth of 11.7 per cent, so this deck sits inside the
    permitted range.
    """
    overlap_m = layout(speed)["overlap_m"]
    per_line = transverse_rate_m_per_line(system)
    return {
        "overlap_m": overlap_m,
        "overlap_lines": overlap_m / per_line,
        "specified_switch_ahead_h": sync_geometry.HEAD_SWITCH_AHEAD_OF_VSYNC_H,
        "measured_switch_lines": (-6.3, -3.5),
        "measured_switch_minimum_line": -5.29,
        "measured_switch_depth": 0.117,
        "refused": "the head's closest approach to either edge track falls "
                   "inside the head-switch overlap, where the switched tap "
                   "shows the other head; a bound from a switched capture "
                   "covers the VISIBLE lines only",
        "capture_that_would_settle_it": "one head's preamp output taken "
                                        "ahead of the switching amplifier",
    }


# --------------------------------------------------------------------------
# The estimator: a matched filter in position, frequency and field parity
# --------------------------------------------------------------------------


@dataclass
class Projection:
    """One matched-filter reading, complex because the fringe has a phase.

    THE CALIBRATION IS THE POINT OF THIS CLASS, and getting it wrong once
    cost a whole round of readings. A projection onto a UNIT-NORM template
    is not a level: the template's profile is an exponential whose length
    is lambda / 2 pi, so it shortens as the frequency rises, and a sweep
    quoted against the window's own square root therefore carries a
    spurious 3 dB per octave tilt and understates every level by the ratio
    of the window to the profile - 13.8 dB for the 28-line window at SP.

    So the template is kept UNNORMALISED, as profile times phasor, and the
    reading is turned into the fringe's PEAK amplitude, which is the
    quantity `geometric_upper_bound_db` bounds:

        <p e^{iwt}, A p cos(wt + phi)> = (A/2) ||p||^2

    so A = 2 |projection| / ||p||^2, and `dbc` is that against the
    carrier's own peak, rms times root two.

    IT IS GOOD TO A FEW DECIBELS AND NOT BETTER, which is stated here
    rather than discovered later. The profile holds only 1.53 cycles at the
    control track's frequency - 4.64 lines, 295 us, times 5183.9 Hz - so
    the negative-frequency image of the projection does not average away,
    and the reading depends on the fringe's phase against the V-sync edge.
    MEASURED over twelve planted phases at 60 fields: a spread of 5.40 dB,
    1.90 dB rms, at most 2.91 dB from the planted level. A bound quoted
    from this estimator therefore carries a systematic +-3 dB that no
    number of fields removes, and the bounds below are stated with it.
    """
    amplitude: complex
    standard_error: float
    fields: int
    reference_rms: float
    window_samples: int
    template_norm: float = 1.0

    @property
    def peak_amplitude(self) -> float:
        """The fringe's peak amplitude at the tape edge, in capture units."""
        return 2.0 * abs(self.amplitude) / (self.template_norm ** 2)

    @property
    def peak_error(self) -> float:
        return 2.0 * self.standard_error / (self.template_norm ** 2)

    @property
    def dbc(self) -> float:
        """The peak amplitude against the carrier's peak, in dB."""
        return 20.0 * math.log10(self.peak_amplitude
                                 / (math.sqrt(2.0) * self.reference_rms))

    @property
    def error_dbc(self) -> float:
        return 20.0 * math.log10(self.peak_error
                                 / (math.sqrt(2.0) * self.reference_rms))

    @property
    def ratio(self) -> float:
        """Amplitude over its own standard error. Below about 2 there is
        nothing; the frequency and position nulls decide the rest."""
        return abs(self.amplitude) / self.standard_error

    @property
    def bound_dbc(self) -> float:
        """The two-sigma level below which a fringe would have been missed."""
        return 20.0 * math.log10(2.0 * self.peak_error
                                 / (math.sqrt(2.0) * self.reference_rms))


def matched_reference(sample_rate_hz: float, first_line: float,
                      last_line: float, switch_line: float,
                      reproduce_frequency_hz: float, track: str = "control",
                      speed: str = "SP", system: str = "NTSC"
                      ) -> np.ndarray:
    """The unit-norm complex template: the fringing exponential in position
    times the reproduce frequency's phasor.

    Complex throughout, because the fringe's phase against the V-sync edge
    is set by where the tape happens to be and is not known in advance -
    only its magnitude is a prediction. Taking the magnitude of a complex
    projection keeps the phase free without letting it be fitted.

    Returned UNNORMALISED - profile times phasor, the profile peaking at 1
    at the tape edge - because the profile's own norm is what turns a
    projection into a level, and because a level measured AT THE EDGE is
    the one `geometric_upper_bound_db` predicts. See `Projection`.
    """
    line = _line_period_s(system) * float(sample_rate_hz)
    a, b = int(first_line * line), int(last_line * line)
    n = b - a
    if n <= 0:
        raise ValueError("the window is empty: %g to %g lines"
                         % (first_line, last_line))
    index = np.arange(a, b, dtype=np.float64)
    lines = index / line
    decay = fringing_decay_length_m(reproduce_frequency_hz, speed)
    per_line = transverse_rate_m_per_line(system)
    place = layout(speed)
    if track == "control":
        distance = place["guard_control_m"] + per_line * (lines - switch_line)
    elif track == "audio":
        distance = place["guard_audio_m"] + per_line * (switch_line - lines)
    else:
        raise ValueError("track must be 'control' or 'audio', got %r" % (track,))
    # normalised to ONE AT THE TAPE EDGE, not at zero distance: the
    # guard's own attenuation belongs to the prediction
    # (`geometric_upper_bound_db`), not to the estimator's calibration, and
    # dividing it out here is what stops it being counted twice.
    guard = place["guard_control_m" if track == "control" else "guard_audio_m"]
    profile = np.exp(-(np.maximum(distance, 0.0) - guard) / decay)
    phasor = np.exp(2j * np.pi * float(reproduce_frequency_hz)
                    * index / float(sample_rate_hz))
    template = profile * phasor
    # ORTHOGONALISED AGAINST A CONSTANT, and this is not a refinement.
    # The profile is a one-sided exponential of time constant tau, so its
    # transform at the reproduce frequency is 1/sqrt(1 + (w tau)^2) of its
    # value at zero - only 19.6 dB down at the control track's 5183.9 Hz
    # with tau = 295 us. A template that leaks a fifth of its gain to DC
    # reads whatever offset sits under the window and reads it EVERYWHERE,
    # which is exactly what it did: before this line the position scan
    # returned the same level to a tenth of a decibel at every line from
    # the head switch to line +24, and the same level at every frequency
    # from 1 to 30 kHz. Removing the mean makes <template, constant> = 0
    # and costs 0.03 per cent of the profile's norm.
    return template - template.mean()


def project(capture: np.ndarray, field_starts: Sequence[int],
            sample_rate_hz: float, reference: np.ndarray,
            first_line: float, reference_rms: float,
            alternate: bool = True, system: str = "NTSC") -> Projection:
    """Project every field's window onto the template and average.

    `alternate` is the parity discriminator and it is not a knob. The tape
    advances v_tape / field rate per field and the control pulses are
    v_tape / frame rate apart, EXACTLY twice that, so a control-track fringe
    reverses sign from one field to the next while anything locked to the
    drum does not. Averaging with the alternation keeps the fringe and
    cancels the switch transient; averaging without it does the opposite.
    Both are reported, and a reading that does not care which is used is not
    a fringe.

    The standard error is the scatter of the per-field projections over the
    root of their number, which is the honest one here: the fields are the
    repeats, and their spread already contains the tape's own variation.
    """
    line = _line_period_s(system) * float(sample_rate_hz)
    a = int(first_line * line)
    n = len(reference)
    values: List[complex] = []
    for index, start in enumerate(field_starts):
        begin = int(start) + a
        if begin < 0 or begin + n > len(capture):
            continue
        sign = (-1.0) ** index if alternate else 1.0
        values.append(sign * np.vdot(reference, capture[begin:begin + n]))
    if len(values) < 3:
        raise ValueError("a projection needs at least three fields, got %d"
                         % len(values))
    array = np.asarray(values, dtype=np.complex128)
    mean = complex(array.mean())
    spread = math.hypot(float(array.real.std(ddof=1)),
                        float(array.imag.std(ddof=1))) / math.sqrt(len(array))
    return Projection(amplitude=mean, standard_error=spread,
                      fields=len(array), reference_rms=float(reference_rms),
                      window_samples=n,
                      template_norm=float(np.linalg.norm(reference)))


def frequency_sweep(capture: np.ndarray, field_starts: Sequence[int],
                    sample_rate_hz: float, frequencies: Sequence[float],
                    switch_line: float, reference_rms: float,
                    track: str = "control", span_lines: float = 28.0,
                    alternate: bool = True, speed: str = "SP",
                    system: str = "NTSC") -> Dict[float, Projection]:
    """The same filter at a ladder of frequencies, which is the control that
    turns a reading into evidence.

    A genuine control-track fringe is a feature AT the predicted frequency.
    A tap's own low-frequency noise is a smooth trend through it. Only the
    sweep separates them, and on the test tape it is the sweep that
    dissolved an apparent -48.3 dBc detection.
    """
    if track == "control":
        first, last = switch_line, switch_line + float(span_lines)
    else:
        first, last = switch_line - float(span_lines), switch_line
    out: Dict[float, Projection] = {}
    for frequency in frequencies:
        reference = matched_reference(sample_rate_hz, first, last, switch_line,
                                      float(frequency), track, speed, system)
        out[float(frequency)] = project(capture, field_starts, sample_rate_hz,
                                        reference, first, reference_rms,
                                        alternate, system)
    return out


def excess_over_trend(sweep: Dict[float, Projection], predicted_hz: float
                      ) -> Dict[str, float]:
    """The bound: how far the predicted frequency stands above the smooth
    trend its neighbours define.

    The tap's low-frequency noise is smooth in log frequency - measured, a
    straight -5.44 dB per octave from 500 Hz to 40 kHz on the playback tap -
    so a straight line through the OTHER points predicts what the predicted
    point would read with no fringe present, and the residual is the
    finding. The scatter of the fit's own residuals is the error bar, which
    is the conservative choice: it charges the bound for the trend's
    departure from a straight line as well as for the noise.
    """
    ordered = sorted(sweep.items())
    others = [(math.log2(f), p.dbc) for f, p in ordered
              if abs(f - predicted_hz) > 1.0]
    if len(others) < 3:
        raise ValueError("a trend needs at least three other frequencies")
    x = np.array([o[0] for o in others])
    y = np.array([o[1] for o in others])
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    scatter = float(residuals.std(ddof=2))
    here = sweep[min(sweep, key=lambda f: abs(f - predicted_hz))]
    expected = slope * math.log2(predicted_hz) + intercept
    return {
        "predicted_hz": float(predicted_hz),
        "measured_dbc": here.dbc,
        "trend_dbc": float(expected),
        "excess_db": here.dbc - float(expected),
        "trend_scatter_db": scatter,
        "trend_slope_db_per_octave": float(slope),
        "sigma": (here.dbc - float(expected)) / scatter if scatter else float("inf"),
        "bound_dbc": float(expected) + 2.0 * scatter,
    }


# --------------------------------------------------------------------------
# What he actually asked for
# --------------------------------------------------------------------------


def audio_displacement(speed: str = "SP") -> Dict[str, float]:
    """The linear audio's offset from the video, straight out of the standard.

    SMPTE 32M 3.5: "Program audio or other information which is time
    coincident with video information recorded at a point S0 of the video 2
    track shall be recorded on either audio track at a distance X downstream
    from that point", and 3.4 fixes X at 79.244 mm. Divided by the tape
    speed that is a TIME, and it is the whole of the synchronisation answer:

        79.244 mm / 33.35 mm/s = 2.376131 s = 71.213 frames at NTSC SP

    A machine that records and replays with the same geometry cancels it,
    which is why nobody notices. It does not cancel when the linear audio is
    captured by anything other than that machine's own audio head at the
    same time - a separate audio pass, a different deck, a tape scanned for
    its audio track - and then this is the number that puts the two back
    together, with no external clock and nothing measured.

    ITS PRECISION IS THE TAPE SPEED'S, and the standard states that too:
    +- 0.5 per cent on 33.35 mm/s (3.1.3), so +- 11.9 ms, +- 0.356 frames.
    The displacement X itself carries no tolerance in table 2.
    """
    tape, _head = _transport(speed)
    from vhsdecode.models import vhs_specification as spec

    key = {"SP": "tape_speed_sp_m_s", "LP": "tape_speed_lp_m_s",
           "EP": "tape_speed_ep_m_s"}[str(speed).upper()]
    tolerance = float(spec.SMPTE_32M[key]["tolerance"])
    distance = _value("audio_control_head_position_m")
    seconds = distance / tape
    return {
        "distance_m": distance,
        "seconds": seconds,
        "frames": seconds * _frame_hz("NTSC"),
        "speed_tolerance": tolerance,
        "seconds_tolerance": seconds * tolerance,
        "frames_tolerance": seconds * _frame_hz("NTSC") * tolerance,
    }


def synchronisation_precision(system: str = "NTSC") -> Dict[str, object]:
    """What timing precision the idea WOULD have bought, and what is left.

    Had the control-track pulse been visible in the video head's own record,
    its position in the sweep would have been an absolute frame reference.
    The precision that reference could reach is set by the pulse's own
    slope: the fringing at the 150 um guard smooths the recorded transition
    over one decay length of tape, 178.1 um, which the head crosses in
    30.9 us - so a single field's reading would locate the frame to about
    that, and pooling N fields to 30.9 us / sqrt(N). Over a second's 59.94
    fields that is 4.0 us, a fortieth of a line.

    NONE OF IT IS AVAILABLE, because the fringe is not there to be timed.
    What IS available needs no fringe: the head switch is servo-locked to
    the control track, so the switch's own position already carries the
    frame reference, and `audio_displacement` converts it to the linear
    audio's position through a specified 79.244 mm. The precision of THAT
    route is the head switch's own repeatability - measured on
    `zaroff-75bars-NTSC-SP`, the field-to-field scatter of the V-sync
    leading edge against a uniform field grid is 23.5 samples at 50 MSps,
    0.470 us - against a tape-speed tolerance of +- 0.5 per cent that
    dominates everything at +- 11.9 ms.
    """
    decay = fringing_decay_length_m(control_track_reproduce_hz("SP", system))
    per_line = transverse_rate_m_per_line(system)
    line_s = _line_period_s(system)
    smear_s = decay / per_line * line_s
    return {
        "fringe_route": {
            "smear_seconds": smear_s,
            "per_field_seconds": smear_s,
            "one_second_of_fields_seconds": smear_s / math.sqrt(1.0 / (line_s * _field_lines(system))),
            "available": False,
            "because": "no fringe was found at the predicted frequency, "
                       "position and parity, to -80.7 dBc pooled",
        },
        "switch_route": {
            "measured_vsync_scatter_seconds": 23.5 / 50e6,
            "specified_speed_tolerance_seconds":
                audio_displacement("SP")["seconds_tolerance"],
            "available": True,
            "because": "the head switch is servo-locked to the control "
                       "track and SMPTE 32M 3.4 fixes the audio head's "
                       "displacement from the scan's end",
        },
    }


# --------------------------------------------------------------------------
# The record of what was run
# --------------------------------------------------------------------------

MEASURED: Dict[str, object] = {
    "date": "2026-09-06",
    "tool": "tools/ringing_measure/edge_tracks_measure.py",
    "document": "docs/EDGE_TRACKS.md",
    "verdict": "no linear-audio and no control-track signal at the video "
               "tap, on three tapes and two tape speeds. At the tape edge "
               "on the deck's own tape, -85.3 dBc with a two-sigma bound of "
               "-85.5, against a prediction there of -67.2 - 18.3 dB below "
               "the most generous level the geometry allows.",
    "captures": {
        "zaroff *-NTSC-SP playback": "16 captures, 50 MSps, 442 fields",
        "zaroff *-NTSC-SP record tap": "16 captures, 439 fields - the "
                                       "control, which carries the drive "
                                       "going TO the head and can hold "
                                       "nothing read off the tape",
        "zaroff *-NTSC-EP playback": "16 captures, 442 fields - the speed "
                                     "control, INCONCLUSIVE",
        "countdown.flac": "40 MSps, 90 fields at each of 2 s and 400 s",
        "home.flac": "40 MSps, 90 fields at each of 2 s and 400 s",
    },
    "geometry": {
        "guard_control_um": 150.0,
        "guard_audio_um": 150.0,
        "worst_case_guard_um": 50.0,
        "speed_ratio_sp": 172.970,
        "control_reproduce_hz_sp": 5183.9,
        "control_reproduce_hz_ep": 15628.5,
        "decay_length_um": 178.1,
        "decay_db_per_line": 1.871,
        "transverse_rate_um_per_line": 38.362,
        "upper_bound_dbc_sp": -64.4,
        "upper_bound_dbc_ep": -68.7,
        "audio_admitted_tape_hz_at_60_db": 245.8,
    },
    # 5183.9 Hz, four-line probe, alternating, dBc: line -> playback, record
    "position_scan_sp": {
        -30: (-91.4, -89.0), -22: (-62.9, -94.9), -18: (-59.3, -104.0),
        -14: (-64.5, -96.7), -10: (-72.2, -90.8), -6: (-39.3, -66.8),
        -2: (-85.3, -77.8), "median": (-79.0, -90.5),
    },
    "controls": {
        "position": "THE ONE THAT DECIDED IT. A fringe must be largest at "
                    "the tape edge and fall inward at 1.871 dB a line; the "
                    "alternating energy near 5183.9 Hz is WEAKEST at the "
                    "edge and peaks eighteen lines earlier. By line -18 the "
                    "leaving head is 0.449 mm past its edge, so the hump "
                    "implies -37.4 dBc at the edge, 27 dB above the bound.",
        "record tap": "flat at -80 dBc from 3.4 to 7 kHz, every ratio at or "
                      "below 1.44; its position scan flat but for the "
                      "switch - so the mid-picture hump is read off the "
                      "tape and is not in the drive",
        "mid-field": "four millimetres from any edge track, where the "
                     "fringing factor is 1e-11.6, the filter still read "
                     "within 8 dB of the sweep end",
        "parity": "the non-alternating average differs by about 15 dB, "
                  "placing the energy in the per-head branch",
        "tape speed": "INCONCLUSIVE. Nothing at the EP prediction of "
                      "15628.5 Hz over 442 EP fields, but the EP floor "
                      "there is -54.2 dBc, above the level an SP-sized "
                      "fringe would reach - those captures cannot test "
                      "their own prediction.",
    },
    "estimator_defects_found": (
        "templates that leaked DC at a tenth of their peak gain, reading "
        "-46.6 dBc at every frequency and every line alike; and a "
        "unit-norm scaling that understated levels by 13.8 dB and tilted a "
        "sweep by 3 dB an octave. Both fixed; the residual +-3 dB from the "
        "profile's 1.53 cycles is stated in `Projection`."),
    "reservation": "the closest approach to either edge track falls inside "
                   "the head-switch overlap, and the lines just after the "
                   "switch are the vertical interval; see `visible_window`",
    "open_observation": "a field-alternating component near 5 kHz peaking "
                        "16 to 22 lines before V-sync, -59 dBc on the "
                        "deck's own tape and -38 dBc on countdown, absent "
                        "from the record tap. NOT an edge track; belongs "
                        "with `head_switch_pair`'s different-deck record "
                        "mark and the modulation-noise lane.",
}
