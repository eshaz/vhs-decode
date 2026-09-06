"""What the VHS standards actually fix, with the clause beside each value.

Ethan: "Let's use the VHS specs to get the needed values for the heads."

THE ANSWER IS LARGELY A NEGATIVE ONE, and it is worth stating before the
table rather than leaving a reader to infer it: SMPTE 32M does not contain
the head. Of the six quantities a magnetic head model needs - gap length,
head-to-tape spacing, coating thickness, remanence, saturation, and the
coil's turns - the standard states NONE. It fixes coercivity, and loosely
("approximately"). Everything else in this project's `head_model` is
sourced from the JVC guide or fitted, and this module exists so that
distinction is visible at every value rather than buried.

PROVENANCE, which changes the weight of every figure below. The repository
folder is labelled `SMPTE-32M-2004`, but the document's own title block
reads "SMPTE 32M-1998, Revision of ANSI/SMPTE 32M-1993", approved 23
September 1998. 32M was reaffirmed in 2004 and nothing inside the file
says so, so these are cited as 32M-1998.

AND THE STANDARD SPECIFIES NO PLAYBACK-SIDE VIDEO RESPONSE AT ALL. Every
video response clause is record-side: the preemphasis, the FM high-pass,
the subpreemphasis tables, the separation filters. There is no playback
deemphasis, no playback equalisation, no head response. That is the formal
justification for treating the reproduce chain as machine-specific and
unidentifiable from the standard - it is not an omission in our modelling,
it is an absence in the specification.
"""

import math
from typing import Dict, Optional

__all__ = ["SMPTE_32M", "value_of", "absent_from_the_standard",
           "head_values_present", "carrier_hz_for_ire"]


def _entry(value, unit: str, clause: str, quote: str,
           tolerance=None, note: str = "") -> Dict[str, object]:
    return {"value": value, "unit": unit, "clause": clause, "quote": quote,
            "tolerance": tolerance, "note": note}


SMPTE_32M: Dict[str, Dict[str, object]] = {
    # ---- transport ----
    "writing_speed_m_s": _entry(
        5.80, "m/s", "3.1.4",
        "The nominal video writing speed shall be 5.80 m/s.",
        note="NOMINAL, no tolerance. The standard does NOT derive it from "
             "the drum, and the geometric value (pi x 62 mm x 29.97 rev/s "
             "plus the tape's creep) is 5.8709 - 1.2 per cent higher. The "
             "stated figure matches the HEAD-ONLY speed of 5.8375 to two "
             "significant figures, so it is most likely that quantity "
             "rounded; wavelength is set by the relative speed."),
    "drum_diameter_m": _entry(
        0.0620, "m", "3.1.5",
        "The video head drum diameter shall be 62.00 mm +- 0.01 mm.",
        tolerance=1e-5),
    "drum_rate_hz": _entry(
        None, "Hz", None,
        "NOT STATED - no rpm, no r/s, no 1800, no 29.97 anywhere",
        note="Only implied by clause 1's '525 lines, 59.94 fields per "
             "second'. So half the field rate is a derivation of ours, not "
             "a quotation."),
    "wrap_angle_deg": _entry(
        None, "degree", "3.4",
        "NOT STATED as an angle. The nearest is 'the end of the 180 degree "
        "scan of a video head'.",
        note="A 180 degree SCAN is stated; the mechanical wrap, which is "
             "larger to give the head-switch overlap, is not."),
    "tape_speed_sp_m_s": _entry(
        33.35e-3, "m/s", "3.1.3",
        "The tape speed shall be 33.35 mm/s +- 0.5%.", tolerance=0.005),
    "tape_speed_lp_m_s": _entry(16.67e-3, "m/s", "4.1", "16.67 mm/s +- 0.5%",
                                tolerance=0.005),
    "tape_speed_ep_m_s": _entry(11.12e-3, "m/s", "4.1", "11.12 mm/s +- 0.5%",
                                tolerance=0.005),
    "tape_tension_n": _entry(
        (0.30, 0.45), "N", "3.7",
        "normally from 0.30 N to 0.45 N at the entrance of the drum"),

    # ---- track geometry ----
    "track_width_sp_m": _entry(
        58e-6, "m", "table 2", "T, Video track width, 0.058 mm",
        note="NOMINAL - table note 1: 'Values are nominal where there are "
             "no tolerances specified.'"),
    "track_pitch_sp_m": _entry(58e-6, "m", "table 2",
                               "P, Video track pitch, 0.058 mm"),
    "video_guard_band_m": _entry(
        0.0, "m", "table 2",
        "NOT STATED, and zero by construction - pitch equals width, so "
        "adjacent video tracks abut."),
    "azimuth_deg": _entry(
        (6.0, -6.0), "degree", "table 2",
        "alpha, Video head azimuth angle, +6 degrees and -6 degrees",
        note="THIS IS NOT AN AZIMUTH ERROR. The two heads are deliberately "
             "opposed to suppress crosstalk between abutting tracks; a head "
             "reading its own track sees no azimuth loss. `head_model`'s "
             "azimuth_error_degrees defaulting to 0 is therefore correct "
             "for the matched case, and the +-6 belongs to a crosstalk "
             "model instead. No tolerance is stated on it."),

    # ---- the medium ----
    "coercivity_a_m": _entry(
        50e3, "A/m", "3.2.1.4",
        "The coercivity shall be approximately 50 x 10^3 A/m.",
        note="'APPROXIMATELY', no tolerance. 50 kA/m is about 628 Oe, "
             "which differs by 4.6 per cent from the 600 Oe class the JVC "
             "guide states and which this project has been using (47.7 "
             "kA/m). Both are hedged; carry both."),
    "coercivity_svhs_a_m": _entry(
        70e3, "A/m", "7.3.3",
        "The coercivity shall be approximately 70 x 10^3 A/m."),
    "tape_thickness_m": _entry(
        19e-6, "m", "table 1", "T-120/T-90/T-60/T-30, 19.0 +1 -2 um",
        note="TOTAL tape thickness, base plus coating undecomposed. The "
             "magnetic layer's own depth is NOT stated, despite clause 5.1 "
             "relying on one for depth multiplexing."),
    "remanence_t": _entry(None, "T", None, "NOT STATED - 'remanen' does not "
                          "occur in the document"),
    "saturation_t": _entry(None, "T", None, "NOT STATED as a magnetic "
                           "property; the one 'saturation' is a signal "
                           "level (3.9.2.1.2)"),
    "coating_thickness_m": _entry(None, "m", None, "NOT STATED"),

    # ---- the head: absent ----
    "gap_length_m": _entry(
        None, "m", "5.1",
        "NOT STATED. The only mention of a head gap is qualitative: 'The "
        "FM audio and video head gaps shall be of different azimuth angles "
        "to minimize crosstalk'.",
        note="So `head_model`'s 0.30 um is not normative."),
    "head_to_tape_spacing_m": _entry(
        None, "m", None,
        "NOT STATED - the word 'spacing' does not occur in the document"),
    "coil_turns": _entry(None, "turn", None, "NOT STATED"),
    "core_permeability_r": _entry(None, "1", None, "NOT STATED"),
    "core_saturation_t": _entry(None, "T", None, "NOT STATED"),
    "record_current_a": _entry(
        None, "A", "3.9.1.1.6",
        "The record current shall be set to the optimum value over the "
        "entire bandwidth of the FM carrier. Optimum record current shall "
        "be that which returns the maximum output signal level during "
        "playback.",
        note="DEFINED ONLY OPERATIONALLY - no value, no units, no "
             "tolerance, no per-head matching. The standard makes record "
             "level a free parameter per machine, which is exactly why "
             "this project treats it as an AXIS rather than a fixed "
             "shape."),

    # ---- the signal ----
    "fm_peak_white_hz": _entry(4.4e6, "Hz", "3.9.1.1.4",
                               "Reference white level (100 IRE units) 4.4 "
                               "MHz +- 0.1 MHz", tolerance=0.1e6),
    "fm_sync_tip_hz": _entry(3.4e6, "Hz", "3.9.1.1.4",
                             "Reference sync level (-40 IRE units) 3.4 MHz "
                             "+- 0.1 MHz", tolerance=0.1e6),
    "fm_deviation_hz": _entry(1.0e6, "Hz", "3.9.1.1.4",
                              "Frequency deviation, ref white to ref sync "
                              "(140 IRE units) 1.0 MHz +- 0.1 MHz",
                              tolerance=0.1e6),
    "fm_blanking_hz": _entry(
        None, "Hz", None,
        "NOT STATED - only white and sync tip are given as reference "
        "points",
        note="So the blanking carrier this decoder uses is derived by "
             "linear interpolation, which the standard licenses only "
             "through 'a linear frequency modulator, having constant "
             "deviation'."),
    "colour_under_hz": _entry(
        629.371e3, "Hz", "3.9.2.1.4",
        "the new carrier frequency to be recorded equals the horizontal "
        "scanning rate, multiplied by 40 (629.371 kHz)",
        note="40 x f_H. The printed 629.371 kHz uses f_H = 15.734 kHz; "
             "40 x the exact 15734.2657 is 629370.63 Hz."),
    "burst_doubler_db": _entry(
        6.0, "dB", "3.9.2.1.3",
        "The amplitude of the color burst part of the chrominance signal "
        "shall be increased by 6.0 dB +- 0.5 dB prior to recording.",
        tolerance=0.5),
    "chroma_line_phase_deg": _entry(
        (90.0, -90.0), "degree", "3.9.2.1.5",
        "video track 1: Advance +90 degrees ...; video track 2: Retard -90 "
        "degrees from the carrier phase of the previous horizontal line"),
    "preemphasis_tau_s": _entry(
        1.3e-6, "s", "3.9.1.1.2 / figure 9",
        "T = C x Rb = 1.3 us +- 0.05 us", tolerance=0.05e-6),
    "preemphasis_ratio": _entry(
        4.0, "1", "3.9.1.1.2 / figure 9", "X = Rb/Ra = 4 +- 0.3",
        tolerance=0.3,
        note="A first-order shelf (1+sT)/((1+X)+sT): zero at 122 kHz, pole "
             "at 612 kHz, total boost 1+X = 5 = 13.98 dB, matching the "
             "13.3-14.3 dB plateau plotted in figure 9."),
    "head_switch_window_h": _entry(
        (5.0, 8.0), "lines", "3.6",
        "The switching position between the two heads during playback "
        "shall lie between 5 and 8 horizontal lines ahead of the leading "
        "edge of the vertical sync signal",
        note="A THREE-LINE PERMITTED WINDOW, not a fixed position - so a "
             "machine-specific switch onset is conformant rather than "
             "anomalous."),
    "carrier_interleave_hz": _entry(
        None, "Hz", "4.3.1.2",
        "The FM carrier frequency to be recorded using the channel 1 video "
        "head shall be 1/2 f_H higher than that using the channel 2 video "
        "head",
        note="7.867 kHz, and LP AND EP ONLY - no interleave is stated for "
             "SP. A real per-head difference, but not on the SP tapes this "
             "arc measures."),
    "head_matching_tolerance": _entry(
        None, None, None,
        "ABSENT ENTIRELY. No permitted difference in gap length, track "
        "width, output amplitude, frequency response, azimuth magnitude or "
        "effective spacing between the two heads of a pair.",
        note="So a measured head-pair asymmetry cannot be checked against "
             "a conformance limit - the standard neither permits nor "
             "forbids it."),
    "playback_response": _entry(
        None, None, None,
        "ABSENT ENTIRELY. Every video response clause is record-side. "
        "There is no playback deemphasis, no playback equalisation and no "
        "head response anywhere in the standard.",
        note="The formal justification for treating the reproduce chain as "
             "machine-specific and unidentifiable from the standard."),
}


def value_of(name: str) -> Optional[object]:
    """The value, or None where the standard does not state one."""
    entry = SMPTE_32M.get(name)
    return None if entry is None else entry["value"]


def absent_from_the_standard() -> Dict[str, str]:
    """Every quantity the standard does NOT fix, with what it says instead.

    Kept as a first-class query because the absences are the finding: a
    model built on them is built on the JVC guide or on a fit, and should
    say so.
    """
    return {name: entry["quote"] for name, entry in SMPTE_32M.items()
            if entry["value"] is None}


def head_values_present() -> Dict[str, bool]:
    """Which of the head model's own parameters the standard supplies.

    The answer for SMPTE 32M is: the track width and the azimuth, and
    nothing else. Not the gap, not the spacing, not the coating, not the
    remanence, not the saturation, and none of the electrical side.
    """
    wanted = ("gap_length_m", "head_to_tape_spacing_m", "coating_thickness_m",
              "remanence_t", "saturation_t", "coil_turns",
              "core_permeability_r", "core_saturation_t",
              "track_width_sp_m", "azimuth_deg", "coercivity_a_m")
    return {name: SMPTE_32M[name]["value"] is not None for name in wanted}


def carrier_hz_for_ire(ire: float) -> float:
    """The FM carrier at a given IRE, by the standard's own two anchors.

    Sync tip is -40 IRE at 3.4 MHz and peak white +100 IRE at 4.4 MHz, so
    the scale is 1.0 MHz over 140 IRE = 7142.857 Hz an IRE. The BLANKING
    carrier is not stated and comes out of this interpolation, which the
    standard licenses only through its 'linear frequency modulator, having
    constant deviation'.
    """
    sync = SMPTE_32M["fm_sync_tip_hz"]["value"]
    white = SMPTE_32M["fm_peak_white_hz"]["value"]
    return sync + (white - sync) * (float(ire) + 40.0) / 140.0
