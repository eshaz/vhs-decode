"""Tape bias: the luminance FM as the AC bias the chrominance is recorded on.

Ethan: *"look through the VHS specs and magnetic tape recording principles to
check if tape bias is a component to this measurement. Findings here may be
able to relate the tape magnetic properties to the head magnetic properties."*

IT IS A COMPONENT, AND IT IS NOT AN INFERENCE - IT IS WRITTEN INTO THE
FORMAT'S OWN DOCUMENTS. The arc has been treating the luma and the chroma as
two channels that share a track and interfere by accident. They do not. The
chrominance is recorded THROUGH the luminance, and both normative sources say
so in as many words:

    SMPTE 32M 7.5.1.2.2   "The chrominance signal shall be recorded with the
                          luminance FM signal acting as bias."
    JVC VTG82063 §1.1.6   "The down converted color signal is then recorded
                          directly, using the FM luminance signal as AC bias."

That single fact reorganises three things this tree already holds. It says the
colour under's record drive was never meant to reach its own optimum, which
disposes of `magnetic_circuit.optimum_record_drive`'s unreachable 15.934 times
the coercivity; it says the chroma's recorded DEPTH is set by the luma's write
depth rather than by the chroma's wavelength, which supplies the justification
`recorded_layer` was missing for applying the luma's drive to the colour
under's row; and it predicts a luma-level-dependent chroma gain and a
third-order product, both of which are measured below and neither of which a
two-independent-channels account allows.

THE SCOPING CAVEAT, BECAUSE IT MATTERS. 7.5.1.2.2 lives in clause 7, the
high-performance (S-VHS) system, and its parent 7.5.1.2 says "The
specifications of 3.9 shall apply to all items except those stated below" - so
for S-VHS the bias clause REPLACES 3.9.2.1.2. For baseline VHS the standard
never uses the word: clause 3.9.3.1 says only "The FM luminance carrier and AM
chrominance carrier shall be combined before recording". The explicit
statement for baseline VHS is the JVC guide's, four times over. So: STATED for
S-VHS, stated by the manufacturer's guide for VHS, and only implied by the
standard's own clause 3 - which is exactly the provenance `vhs_specification`
exists to keep visible.

WHAT THE STANDARD DOES NOT SAY, AND THE ABSENCES CARRY THE ARGUMENT.
`absent_from_the_standard` below is the list. The word "bias" occurs EXACTLY
ONCE in the whole of SMPTE 32M and never in an audio clause, although the
linear audio track is conventional AC-bias recording and the JVC guide
describes it as such. There is no chroma record-current clause at all - the
chroma is specified by a PLAYBACK level and never by a current. There is no
luma-to-chroma level ratio for VHS. There is no FM carrier amplitude anywhere,
and the standard never says the carrier is constant-amplitude; its one
"constant" is "constant deviation". And "saturation" occurs once in the entire
document, in the chroma record-level clause, with its referent undefined.

THE PHYSICS, AND WHAT IT PREDICTS THAT A NO-BIAS ACCOUNT DOES NOT.
Unbiased, a magnetic coating records through its remanence curve, whose
switching-field distribution is centred on the coercivity: the transfer is
flat through the origin and steep at Hc, so a small signal leaves almost
nothing and what it leaves is grossly distorted. Biased, the medium is walked
down through decreasing minor loops and follows the ANHYSTERETIC curve
instead, which is linear through the origin. The two accounts therefore
disagree about the SIGN of the distortion-against-level slope, and the
standard settles it: clause 3.9.2.1.2 orders the chroma recorded 7 to 10 dB
BELOW saturation. Computed in `operating_point` and `unbiased_distortion`:

    biased, on the anhysteretic curve   third harmonic 24.8% at the luma
                                        band centre's own 3.278 Hc drive,
                                        2.94% at 7 dB below saturation and
                                        1.41% at 10 dB below
    unbiased, on the remanence curve    11.3% at 3 Hc, 55.8% at 2 Hc,
                                        149.3% at 1 Hc - it DIVERGES as the
                                        level falls

A clause ordering a 7-to-10 dB back-off is only intelligible under the first.
That is a specification-level falsification of the no-bias account, and it
needed no capture.

BUT IT IS A FEW-CYCLE BIAS, NOT THE TEXTBOOK IDEAL, and this is the honest
limit of the mechanism. Anhysteretic magnetisation is the limit of an
alternating field whose envelope decays SLOWLY - many cycles per decade of
field. `bias_cycles` computes what a tape element actually sees from the
arc's own Karlqvist field, at the guide's 0.30 um gap (JVC VTG82063 7.2.1;
SMPTE 32M states no gap) and the luma's own optimum drive: 0.302 bias
cycles across the switching band from twice the
coercivity to half of it, 1.867 down to a tenth, 3.954 down to a twentieth,
and the field changes by a factor of 5.045 PER BIAS CYCLE at the coercivity
crossing. The ideal wants that factor near unity. So the linearisation is
partial by construction, and the format compensates twice over: it backs the
chroma off 7 to 10 dB (3.9.2.1.2), and it places the residual third-order
product where the eye cannot see it (`interleave`).

THE MEASUREMENTS ARE IN `MEASURED`, taken this session on the matched record
and playback taps of one Sony SLV-778HF. Two of them decide the question.

  * THE TAPE IMPOSES A LUMA-LEVEL-DEPENDENT CHROMA GAIN OF +2.084 +- 0.048 dB
    PER MHz OF LUMA CARRIER (SP, 43.3 sigma), while the FM's own envelope
    passes through unchanged at -0.095 +- 0.065 dB/MHz. The playback chain
    sees the chroma at ONE frequency, so its gain is one number and cannot
    depend on the luma; the dependence is therefore written at record time.
    A y-only recording - the FM with nothing to bias - puts 44.5 dB less in
    the chroma band, so the reading is chroma and not the FM's skirt.

    AND THE ONE ALTERNATIVE ACCOUNT IS EXCLUDED BY THE SAME PAIR OF
    NUMBERS. A compressive playback amplifier would make the chroma's gain
    depend on the FM's amplitude with the observed sign, but a memoryless
    compression moves a small signal's gain EXACTLY TWICE as far as it
    moves the large signal's own: it would have to shift the FM envelope by
    +1.042 dB/MHz and the measurement says -0.095 +- 0.065, the wrong sign
    and a sixteenth of the size, 17 standard errors away
    (`playback_compression_bounded`).

  * THE THIRD-ORDER PRODUCT THE JVC GUIDE NAMES IS PRESENT AT THE PLAYBACK
    TAP AND ABSENT AT THE RECORD TAP. In the demodulated luma the beat at
    2 f_c stands 9.28 dB above its own floor at playback and 0.91 dB - the
    floor - at record, and 0.21 dB in the y-only playback. The nonlinearity
    that makes it is between the record head's drive and the playback head's
    output, which is where the medium is.

BOTH FIGURES REPRODUCE AT EP - 8.57 dB at the playback tap against 0.17 at
the record tap and -0.03 in the y-only playback - so the finding is not a
property of one tape speed.

AND THE SECOND-ORDER TERM IS NOT CLEAN, WHICH IS WORTH SAYING BECAUSE THE
FIRST PASS HERE CLAIMED IT WAS. At SP the beat at f_c is already at the
RECORD tap (4.45 dB) and does not grow through the tape (3.10 dB), which
reads as a purely electrical coupling; at EP it is at the FLOOR at the record
tap (0.17 dB) and stands at 3.22 dB at playback, so the medium makes one
there too. The order that separates cleanly at both speeds is the THIRD -
0.91 and 0.17 dB at the record tap against 9.28 and 8.57 at playback - and it
is the third order that both documents name.

THE HEAD AND THE TAPE, WHICH IS WHAT ETHAN ASKED THE FINDINGS TO REACH. Bias
is the missing term between them. Without it a head field and a tape
magnetisation are joined only by the coercivity - the field either exceeds it
or does not - and the chroma, whose own field never approaches it, cannot be
recorded at all. With it the join is `head_to_tape_link` below: the chroma's
remanence is the ANHYSTERETIC SUSCEPTIBILITY, a tape property, multiplying the
chroma's own head field, over the depth to which the LUMA's field exceeded the
coercivity - a head property. The tape supplies the slope, the head supplies
the field and the depth, and the product is what the playback head reads.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import colour_under, magnetic_circuit, vhs_specification

__all__ = [
    "SPECIFICATION", "MEASURED", "AUDIO_BIAS_RULE",
    "specification", "absent_from_the_standard",
    "bias_frequency_ratio", "bias_ratio_over_deviation",
    "bias_cycles", "anhysteretic_distortion", "unbiased_distortion",
    "operating_point", "record_current_scope", "chroma_recorded_layer",
    "intermodulation_products", "interleave", "beat_hz",
    "head_to_tape_link", "differential_gain_refused",
    "product_order_refused", "playback_compression_bounded",
    "line_amplitudes", "alternating_split", "line_rate_offset",
    "product_in_the_luma",
]


# --------------------------------------------------------------------------
# What the two documents say, with the clause beside each
# --------------------------------------------------------------------------

def _clause(source: str, clause: str, quote: str, note: str = "",
            scope: str = "VHS") -> Dict[str, str]:
    return {"source": source, "clause": clause, "quote": quote, "note": note,
            "scope": scope}


SPECIFICATION: Dict[str, Dict[str, str]] = {
    "bias_stated": _clause(
        "SMPTE 32M-1998", "7.5.1.2.2",
        "The chrominance signal shall be recorded with the luminance FM "
        "signal acting as bias. The chrominance signal recording level shall "
        "be set so that the playback level of the spurious components "
        "measured at a frequency f_y - 2f_c is attenuated between 20 dB and "
        "25 dB with reference to the level of the output signal at a "
        "frequency f_y",
        scope="S-VHS",
        note="THE ONE OCCURRENCE OF THE WORD 'BIAS' IN THE STANDARD. Clause 7 "
             "is the high-performance system and 7.5.1.2 opens 'The "
             "specifications of 3.9 shall apply to all items except those "
             "stated below', so this REPLACES 3.9.2.1.2 for S-VHS. For "
             "baseline VHS the standard says only 3.9.3.1."),
    "bias_stated_for_vhs": _clause(
        "JVC VTG82063", "1.1.6",
        "The down converted color signal is then recorded directly, using "
        "the FM luminance signal as AC bias.",
        note="Stated four times in the guide - 1.1.6 (NTSC), 1.1.8 (PAL), "
             "1.1.10 (SECAM) and 3.2.2, the last as 'This circuit records the "
             "down converted color signal by using the FM luminance signal as "
             "recording bias.' This is the explicit statement for BASELINE "
             "VHS, which the standard itself does not make."),
    "combination": _clause(
        "SMPTE 32M-1998", "3.9.3.1",
        "The FM luminance carrier and AM chrominance carrier shall be "
        "combined before recording.",
        note="The whole of what the standard says about the two channels "
             "meeting, for baseline VHS. No ratio, no order, no mechanism."),
    "chroma_record_level": _clause(
        "SMPTE 32M-1998", "3.9.2.1.2",
        "The chrominance signal is recorded directly on the video tape as an "
        "amplitude modulated (AM) rf carrier. Record level shall be such "
        "that the amplitude of the played back chrominance signal is 7 dB to "
        "10 dB below the saturation level of the recording chrominance "
        "signal.",
        note="THE ONE OCCURRENCE OF 'SATURATION' IN THE STANDARD, and its "
             "referent - 'the saturation level of the recording chrominance "
             "signal' - is self-referential and nowhere defined. No test "
             "signal, no bandwidth, no instrument, no reference tape."),
    "record_current": _clause(
        "SMPTE 32M-1998", "3.9.1.1.6",
        "The record current shall be set to the optimum value over the "
        "entire bandwidth of the FM carrier. Optimum record current shall be "
        "that which returns the maximum output signal level during playback.",
        note="SCOPED TO THE FM CARRIER, and that scope is the finding. There "
             "is no chroma record-current clause anywhere in the standard."),
    "chroma_record_current": _clause(
        "JVC VTG82063", "1.1.6",
        "Constant current characteristics are possessed by the "
        "down-converted color signal recording current.",
        note="Repeated per system at 1.1.8, 1.1.10, 1.1.15 and 1.1.18. The "
             "chroma is driven at a CONSTANT CURRENT, deliberately not at a "
             "per-frequency optimum - which is what a bias signal's companion "
             "is, and is not what a saturation recording would be."),
    "luma_record_current_shape": _clause(
        "JVC VTG82063", "3.2.2",
        "Above 3.4 MHz : Optimum recording saturation current / 2 MHz : "
        "3 +- 1 dB 0 dB at 3.4 MHz / 1 MHz : 6 +- 1.5 dB / Below 1 MHz : "
        "Flat current response. There are two types of recording amplifier: "
        "fixed voltage and fixed current.",
        note="The record amplifier's own frequency shape, which is NOWHERE in "
             "SMPTE 32M. It is why the FM's amplitude at the head is not "
             "constant, and therefore why the bias level moves with the "
             "picture - see MEASURED['fm_envelope_slope_db_per_mhz']."),
    "head_current_tolerance": _clause(
        "JVC VTG82063", "1.1.4",
        "Specified to be within +-1.5 dB of 4 MHz optimum recording current.",
        note="The only tolerance on the field that does the biasing."),
    "product_named": _clause(
        "JVC VTG82063", "1.1.6",
        "When recorded and played back using magnetic tape, which possesses "
        "3-dimensional distortion and nonlinearity, interference in the form "
        "of F_D + 2F_DC (Fo : FM carrier; Fdc : down converted color signal) "
        "is introduced and cannot be ignored. When the 2F dc component is "
        "detected and demodulated, a beat is produced with respect to the "
        "luminance signal and appears in the picture.",
        note="THE GUIDE NAMES THE MEDIUM AS THE NONLINEARITY, and names the "
             "product this module measures. SMPTE 32M contains none of the "
             "words nonlinear, distortion, intermodulation, moire or beat."),
    "interleave_reason": _clause(
        "JVC VTG82063", "1.1.6",
        "Therefore, as with the color signal, Fc (down converted color "
        "subcarrier frequency) must be selected so that the frequency of the "
        "2Fdc component becomes interleaved (1/2 offset) in relation to the "
        "luminance signal. ... The 2 Fdc components for both CH-1 and CH-2 "
        "become interleaved (1/2 line offset) with respect to the luminance "
        "signal and thereby visually reduced. The 629.371 kHz value was "
        "selected for both reducing noise and in consideration of color "
        "bandwidth.",
        note="THE COLOUR UNDER'S FREQUENCY WAS CHOSEN FOR THIS PRODUCT. The "
             "residual the bias fails to remove is the reason 40 f_H is 40 "
             "f_H, which makes the bias mechanism load-bearing on the "
             "format's central number rather than incidental to it."),
    "down_conversion": _clause(
        "SMPTE 32M-1998", "3.9.2.1.4",
        "The chrominance signal shall be (down) converted such that the new "
        "carrier frequency to be recorded equals the horizontal scanning "
        "rate, multiplied by 40 (629.371 kHz)."),
    "deviation": _clause(
        "SMPTE 32M-1998", "3.9.1.1.4",
        "Reference white level (100 IRE units) 4.4 MHz +- 0.1 MHz; Reference "
        "sync level (-40 IRE units) 3.4 MHz +- 0.1 MHz; Frequency deviation, "
        "ref white to ref sync (140 IRE units) 1.0 MHz +- 0.1 MHz",
        note="'a linear frequency modulator, having constant deviation'. "
             "CONSTANT DEVIATION, not constant amplitude - the standard "
             "never says the carrier's amplitude is constant, and the "
             "measurement below says it is not."),
    "audio_bias_rule": _clause(
        "JVC VTG82063", "7.1.1",
        "The audio signal is recorded together with AC bias of at least "
        "twice the highest audio frequency (typically 3 to 5 times). AC bias "
        "recording features good linearity and naturalness, plus low "
        "non-linear distortion.",
        note="THE FORMAT'S OWN CRITERION FOR A VALID BIAS, stated for the "
             "linear audio track. `bias_ratio_over_deviation` tests the "
             "video system against it. No numeric audio bias frequency is "
             "given in either document, so the ratio is all there is."),
    "tape_conformance": _clause(
        "JVC VTG82063", "1.1.25",
        "Coercivity : 600 oersted class (nominal) ... Optimum recording "
        "current shall not differ from the standard tape.",
        note="The entire tape-to-current conformance requirement. It fixes "
             "the BIAS the tape will see by fixing the tape, which is the "
             "only place the two magnetic properties are tied together in "
             "any document."),
}
"""Every statement either document makes that bears on bias, with its clause.

Fetched 2026-09-06 from `decode-orc/analogue-video-specifications`, files
`docs/vhs/SMPTE-32M-2004/markdown.md` (whose title block reads "SMPTE
32M-1998, Revision of ANSI/SMPTE 32M-1993") and
`docs/vhs/JVC-Video-Technical-Guide-VTG82063/section-{1,3,7}.md`.
"""


AUDIO_BIAS_RULE = (2.0, 3.0, 5.0)
"""The guide's bias-frequency rule at 7.1.1: at least 2x the highest signal
frequency, typically 3 to 5x. Held as the format's own criterion for what
counts as a bias, because it is the only such criterion either document
states and the alternative would be to import one from outside."""


def specification() -> Dict[str, Dict[str, str]]:
    """The clause table, so a caller can quote rather than paraphrase."""
    return dict(SPECIFICATION)


def absent_from_the_standard() -> Dict[str, str]:
    """What was searched for in SMPTE 32M and NOT found.

    Kept first-class for the same reason `vhs_specification` keeps its own:
    a model built on an absence is built on the JVC guide or on a fit, and
    every one of these absences was verified by searching the document text
    rather than by failing to remember it.
    """
    return {
        "video_bias_in_clause_3":
            "the word 'bias' does not occur anywhere in clause 3, the "
            "baseline VHS system; its one occurrence is 7.5.1.2.2, S-VHS",
        "audio_bias":
            "no audio bias frequency, level or current, and the word does "
            "not occur in any audio clause - although 3.10.1 and 3.10.2 do "
            "specify the audio recording level and its time constants",
        "chroma_record_current":
            "no clause. The chroma is specified by a PLAYBACK level "
            "(3.9.2.1.2) and never by a current, in any unit",
        "chroma_preemphasis_and_acc":
            "no chroma preemphasis clause and no automatic-chroma-control "
            "clause in the normative text; the only chroma emphasis is "
            "informative annex A's vertical emphasis",
        "luma_to_chroma_level_ratio":
            "none for VHS. The only quantitative relation between the two "
            "carriers in the whole document is the S-VHS f_y - 2f_c "
            "criterion at 7.5.1.2.2",
        "fm_carrier_amplitude":
            "no amplitude at the record head, in any unit, and no statement "
            "that the carrier is constant-amplitude - 'constant deviation' "
            "is the only 'constant' in 3.9.1.1.4, and the words 'limiter' "
            "and 'constant amplitude' do not occur",
        "saturation_referent":
            "'saturation' occurs once, in 3.9.2.1.2, and what the 7 to 10 dB "
            "is measured against is not defined - not the tape, not the FM "
            "carrier, not a test condition",
        "nonlinearity":
            "the words nonlinear, non-linear, distortion, intermodulation, "
            "third-order, moire and beat do not occur. The one 'spurious' is "
            "7.5.1.2.2, which IS a third-order limit without naming itself "
            "as one",
        "record_amplifier_response":
            "3.9.3.2 is the whole of it - 'An amplifier shall be provided to "
            "supply recording signal current drive to the record heads.' The "
            "current-versus-frequency shape is in the JVC guide alone",
    }


# --------------------------------------------------------------------------
# Is the FM a valid bias at all, by the format's own criterion
# --------------------------------------------------------------------------

def bias_frequency_ratio(luma_hz, system: str = "NTSC") -> np.ndarray:
    """The bias-to-signal frequency ratio, f_y / f_c.

    The colour under is the signal being biased and the FM carrier is the
    bias, so this is the quantity the guide's audio rule at 7.1.1 is about.
    """
    carrier = colour_under.carrier_hz(system)
    return np.asarray(luma_hz, dtype=np.float64) / carrier


def bias_ratio_over_deviation(system: str = "NTSC") -> Dict[str, object]:
    """THE FM IS A VALID BIAS BY THE FORMAT'S OWN RULE, ACROSS THE WHOLE
    DEVIATION - and the ratio MOVES, which is the first prediction.

    The guide states the criterion for the linear audio track and states no
    other: bias "of at least twice the highest audio frequency (typically 3
    to 5 times)". Measured against the standard's own deviation and the
    standard's own 40 f_H:

        sync tip   3.4 MHz     5.402
        blanking   3.686 MHz   5.856
        centre     3.868 MHz   6.146
        peak white 4.4 MHz     6.991

    Never below the stated minimum of 2, and above the typical 3-to-5 band
    everywhere - so the mechanism is not marginal on the frequency axis.

    BUT THE RATIO SWINGS BY 29 PER CENT FROM SYNC TIP TO PEAK WHITE, and
    that is the point. A bias whose frequency is a function of the picture
    is a recording sensitivity that is a function of the picture, which is
    differential gain, which is what `MEASURED` finds. A two-independent-
    channels account has nothing to offer here at all.
    """
    band = magnetic_circuit.spec_band_centres()
    carrier = colour_under.carrier_hz(system)
    speed = float(band["writing_speed_m_s"])
    points = {
        "sync_tip": vhs_specification.SMPTE_32M["fm_sync_tip_hz"]["value"],
        "blanking": vhs_specification.carrier_hz_for_ire(0.0),
        "geometric_centre": float(band["luma_geometric_centre_hz"]),
        "peak_white": vhs_specification.SMPTE_32M["fm_peak_white_hz"]["value"],
    }
    rows = {}
    for name, hz in points.items():
        rows[name] = {
            "luma_hz": float(hz),
            "ratio": float(hz) / carrier,
            "bias_wavelength_m": speed / float(hz),
        }
    ratios = [row["ratio"] for row in rows.values()]
    minimum, typical_low, typical_high = AUDIO_BIAS_RULE
    return {
        "colour_under_hz": carrier,
        "signal_wavelength_m": speed / carrier,
        "points": rows,
        "lowest_ratio": min(ratios),
        "highest_ratio": max(ratios),
        "swing": max(ratios) / min(ratios),
        "meets_the_minimum": bool(min(ratios) >= minimum),
        "above_the_typical_band": bool(min(ratios) > typical_high),
        "rule": "JVC VTG82063 7.1.1: at least %gx, typically %g to %gx"
                % (minimum, typical_low, typical_high),
    }


def bias_cycles(depth_m: Optional[float] = None,
                gap_m: float = 0.30e-6,
                spacing_m: float = 0.05e-6,
                bias_hz: Optional[float] = None,
                coercivity_a_m: Optional[float] = None,
                drive_ratio: Optional[float] = None,
                writing_speed_m_s: Optional[float] = None,
                bands: Sequence[Tuple[float, float]] = ((2.0, 0.5),
                                                        (1.0, 0.1),
                                                        (1.0, 0.05),
                                                        (2.0, 0.02)),
                points: int = 2000001,
                reach_m: float = 60e-6) -> Dict[str, object]:
    """HOW MANY BIAS CYCLES A TAPE ELEMENT SEES ON ITS WAY OUT - the test the
    textbook account has to pass and does not.

    Anhysteretic magnetisation is a LIMIT: the alternating field's envelope
    must decay slowly enough that the element traverses many decreasing minor
    loops. The criterion is the fractional change per bias cycle,
    `|d ln H / dx| * lambda_bias`, which the ideal wants far below one. The
    field is the arc's own `karlqvist_field`; nothing here is fitted.

    MEASURED at the guide's 0.30 um gap (JVC VTG82063 7.2.1; the standard
    states no gap), this project's 0.05 um separation, the standard's
    5.80 m/s and the luma band's own optimum drive of 3.278 times the
    coercivity, for an element at the top of the coating:

        peak field at 0.05 um            2.606 Hc
        2.0 Hc -> 0.50 Hc                0.302 bias cycles
        1.0 Hc -> 0.10 Hc                1.867 bias cycles
        1.0 Hc -> 0.05 Hc                3.954 bias cycles
        2.0 Hc -> 0.02 Hc               10.314 bias cycles
        fractional change per cycle at the coercivity crossing   5.045

    SO THE MECHANISM IS REAL AND THE IDEAL IS NOT REACHED. Through the
    switching band proper - twice the coercivity down to half of it, where
    the particles that carry the signal actually reverse - the element sees
    THREE TENTHS OF A CYCLE, and the field changes by a factor of five per
    cycle where the ideal wants a few per cent. Ten cycles appear only if
    the far tail down to a fiftieth of the coercivity is counted, and almost
    nothing switches there.

    AND THAT IS NOT A FAILURE OF THE ACCOUNT, IT IS THE REASON THE FORMAT
    LOOKS THE WAY IT DOES. A partial anhysteretic leaves residual
    nonlinearity, and the format answers it twice: 3.9.2.1.2 backs the
    chroma off 7 to 10 dB so the residual curvature is small, and 40 f_H is
    chosen so the residual PRODUCT interleaves (`interleave`). Both of those
    clauses are otherwise unexplained.

    The count cannot be raised by choosing a longer wavelength for the bias,
    because the bias is the luma and its wavelength is what makes the luma
    recordable: the recording zone must be shorter than half a luma
    wavelength or no luma transition survives. The few-cycle bias is
    structural to putting both signals on one track.
    """
    band = magnetic_circuit.spec_band_centres()
    speed = (float(band["writing_speed_m_s"]) if writing_speed_m_s is None
             else float(writing_speed_m_s))
    bias = (float(band["luma_geometric_centre_hz"]) if bias_hz is None
            else float(bias_hz))
    coercivity = (vhs_specification.SMPTE_32M["coercivity_a_m"]["value"]
                  if coercivity_a_m is None else float(coercivity_a_m))
    drive = (magnetic_circuit.optimum_record_drive(
        gap_m=gap_m, spacing_m=spacing_m)["drive_ratio"]
        if drive_ratio is None else float(drive_ratio))
    depth = float(spacing_m) if depth_m is None else float(depth_m)
    wavelength = speed / bias

    x = np.linspace(0.0, float(reach_m), int(points))
    field = magnetic_circuit.karlqvist_field(
        x, np.full_like(x, depth), gap_m,
        drive * coercivity)["magnitude_a_m"]
    log_slope = np.abs(np.gradient(np.log(np.maximum(field, 1e-30)), x))

    def at(level: float) -> int:
        return int(np.argmin(np.abs(field - level * coercivity)))

    crossing = at(1.0)
    counts = {}
    for high, low in bands:
        a, b = at(high), at(low)
        counts["%g_to_%g_Hc" % (high, low)] = abs(x[b] - x[a]) / wavelength
    return {
        "bias_hz": bias,
        "bias_wavelength_m": wavelength,
        "depth_m": depth,
        "drive_ratio": drive,
        "peak_field_over_coercivity": float(field[0]) / coercivity,
        "coercivity_crossing_m": float(x[crossing]),
        "fractional_change_per_cycle": float(log_slope[crossing] * wavelength),
        "cycles": counts,
        "ideal_wants": "the fractional change per cycle far below one, so "
                       "that the element traverses many decreasing minor "
                       "loops; `fractional_change_per_cycle` beside this is "
                       "what this geometry actually delivers, and at the "
                       "format's own figures it is 5.045",
        "reaches_the_ideal": bool(log_slope[crossing] * wavelength < 1.0),
    }


# --------------------------------------------------------------------------
# The transfer, biased and not, and where the standard puts the operating point
# --------------------------------------------------------------------------

def _harmonics(drive: float, order: int = 5, points: int = 40001
               ) -> Dict[int, float]:
    """Odd harmonics of `tanh(drive * sin)`, the anhysteretic curve driven
    by a sinusoid. Odd only, because the curve is odd."""
    theta = np.linspace(0.0, 2.0 * math.pi, int(points))[:-1]
    y = np.tanh(float(drive) * np.sin(theta))
    return {k: float(2.0 * np.mean(y * np.sin(k * theta)))
            for k in range(1, order + 1, 2)}


def anhysteretic_distortion(drive_ratio) -> Dict[str, np.ndarray]:
    """What a biased recording leaves, as a function of the signal's own
    drive in units of the coercivity.

    The curve is `magnetic_circuit.anhysteretic_b`'s - `B = Bs tanh(H / Hc)`,
    a Langevin-class two-state limit with no free parameter beyond the two
    the datasheet states - driven by a sinusoid, which is what an AM carrier
    is. The fundamental saturates at `4 / pi` times the remanence, the
    square-wave limit, and that is the "saturation level" 3.9.2.1.2 has to
    mean since it names no other.
    """
    drives = np.atleast_1d(np.asarray(drive_ratio, dtype=np.float64))
    rows = [_harmonics(float(d)) for d in drives]
    first = np.array([r[1] for r in rows])
    third = np.array([r[3] for r in rows])
    fifth = np.array([r[5] for r in rows])
    return {
        "drive_ratio": drives,
        "fundamental": first,
        "third_harmonic": np.abs(third),
        "fifth_harmonic": np.abs(fifth),
        "third_harmonic_fraction": np.abs(third) / np.maximum(np.abs(first),
                                                              1e-300),
        "saturated_fundamental": 4.0 / math.pi,
    }


def unbiased_distortion(drive_ratio) -> Dict[str, np.ndarray]:
    """What the SAME signal leaves with no bias, and it goes the other way.

    Without bias the medium records on its remanence curve, whose switching
    field distribution is centred on the coercivity - that is what a
    coercivity IS - so the transfer is `tanh((|H| - Hc) / Hc)` in the sense
    of the field, flat through the origin and steep at Hc. The scale of the
    distribution is taken as the coercivity itself rather than fitted,
    which is the same one-parameter choice `anhysteretic_b` makes and the
    only one either document supports.

    MEASURED, third harmonic as a fraction of the fundamental:

        H_pk = 3 Hc     11.3 per cent
        H_pk = 2 Hc     55.8 per cent
        H_pk = 1 Hc    149.3 per cent

    IT DIVERGES AS THE LEVEL FALLS, because the operating point moves into
    the dead zone the coercivity makes. Below one coercivity the model
    stops meaning anything at all - the remanence no longer follows the
    field - and the table is cut there rather than extrapolated.

    THIS IS THE DISCRIMINATING PREDICTION. Biased, distortion falls as the
    level falls; unbiased, it rises. SMPTE 32M 3.9.2.1.2 orders the chroma
    recorded 7 to 10 dB BELOW saturation, which is the right thing to do
    under the first account and the wrong thing under the second.
    """
    coercivity = vhs_specification.SMPTE_32M["coercivity_a_m"]["value"]
    drives = np.atleast_1d(np.asarray(drive_ratio, dtype=np.float64))
    theta = np.linspace(0.0, 2.0 * math.pi, 40001)[:-1]
    first, third, usable = [], [], []
    for drive in drives:
        field = float(drive) * coercivity * np.sin(theta)
        remanent = np.sign(field) * np.tanh(
            (np.abs(field) - coercivity) / coercivity)
        f1 = float(2.0 * np.mean(remanent * np.sin(theta)))
        f3 = float(2.0 * np.mean(remanent * np.sin(3.0 * theta)))
        first.append(f1)
        third.append(abs(f3))
        usable.append(bool(drive >= 1.0))
    first = np.array(first)
    third = np.array(third)
    return {
        "drive_ratio": drives,
        "fundamental": first,
        "third_harmonic": third,
        "third_harmonic_fraction": third / np.maximum(np.abs(first), 1e-300),
        "usable": np.array(usable),
        "why_cut_at_one_coercivity":
            "below the coercivity the remanence does not follow the field "
            "and the sign of this model's fundamental inverts; it is a dead "
            "zone, not a transfer, and reporting a distortion figure there "
            "would be reporting an artefact of the parametrisation",
    }


def operating_point(system: str = "NTSC") -> Dict[str, object]:
    """WHERE 3.9.2.1.2 PUTS THE CHROMA, AND WHAT THAT BUYS.

    The clause asks for the played-back chroma 7 to 10 dB below saturation.
    Saturation of the anhysteretic transfer is the square-wave limit, `4 /
    pi` times the remanence, so the clause names a fundamental amplitude and
    the drive that produces it follows.

    MEASURED on `anhysteretic_distortion`:

        7 dB below saturation    H_pk = 0.6221 Hc, third harmonic 2.94%
        10 dB below saturation   H_pk = 0.4201 Hc, third harmonic 1.41%
        the luma band centre's own optimum, 3.278 Hc      24.8%

    So the back-off the clause orders buys a factor of 8.4 to 17.6 in
    third-harmonic distortion, and puts the chroma at four to six tenths of
    the coercivity - a field that on its own, unbiased, would record
    NOTHING. That is the whole argument in one line: the standard's chosen
    operating point is below the medium's own threshold, and is only a
    working operating point because the luma FM is there.
    """
    saturated = 4.0 / math.pi
    out = {}
    for db in (7.0, 10.0):
        target = saturated * 10.0 ** (-db / 20.0)
        low, high = 1e-4, 1e4
        for _ in range(200):
            mid = math.sqrt(low * high)
            if _harmonics(mid)[1] < target:
                low = mid
            else:
                high = mid
        drive = math.sqrt(low * high)
        harmonic = _harmonics(drive)
        out["%g_db_below_saturation" % db] = {
            "drive_ratio": drive,
            "fundamental": harmonic[1],
            "third_harmonic_fraction": abs(harmonic[3]) / abs(harmonic[1]),
            "below_the_coercivity": bool(drive < 1.0),
        }
    luma = magnetic_circuit.optimum_record_drive()["drive_ratio"]
    luma_harmonic = _harmonics(luma)
    return {
        "clause": SPECIFICATION["chroma_record_level"]["clause"],
        "saturated_fundamental": saturated,
        "window": out,
        "luma_drive_ratio": luma,
        "luma_third_harmonic_fraction":
            abs(luma_harmonic[3]) / abs(luma_harmonic[1]),
        "colour_under_hz": colour_under.carrier_hz(system),
        "why": "the clause's operating point is below the coercivity, where "
               "an unbiased recording leaves nothing; it is a working point "
               "only because the luma FM biases it",
    }


# --------------------------------------------------------------------------
# What bias does to the record-current question this tree already asked
# --------------------------------------------------------------------------

def record_current_scope(system: str = "NTSC", **kwargs) -> Dict[str, object]:
    """THE 15.934 RESOLVED: it is not a requirement that fails, it is the
    measure of why the bias exists.

    `magnetic_circuit.optimum_record_drive` implements SMPTE 32M 3.9.1.1.6
    and finds the colour under's optimum drive at 15.934 times the
    coercivity against the luma band's 3.000 to 3.598 - 4.86 times as much,
    and above the 7.50 of headroom `head_transformer` measures before the
    core itself saturates. The module's own conclusion was that the chroma's
    optimum is "not merely unchosen but unreachable".

    THE BIAS ACCOUNT SAYS THAT COMPUTATION ASKS THE CHROMA THE WRONG
    QUESTION, and the standard's own scoping says so first. Clause 3.9.1.1.6
    fixes the optimum "over the entire bandwidth of the FM CARRIER" and
    there is no chroma record-current clause anywhere in the document; the
    JVC guide says the chroma is driven at a CONSTANT CURRENT and recorded
    "using the FM luminance signal as AC bias". A biased signal is not
    written by its own field reaching the coercivity - it is written by the
    BIAS field, on the anhysteretic curve, with its own field only tilting
    that process. So it does not need 15.934 Hc; it needs to be SMALL, and
    3.9.2.1.2 makes it small.

    WHAT THE 15.934 STILL IS, AND IT IS WORTH MORE THAN IT WAS: the drive a
    9.2156 um wavelength would need if it were recorded the way the luma is.
    It is 4.86 times the luma's and 2.12 times the core's whole headroom, so
    saturation recording of the colour under is not merely inconvenient but
    impossible on this head - which is the quantitative reason the format
    had to bias it. The figure moves from an anomaly to the argument.
    """
    carrier = colour_under.carrier_hz(system)
    chroma = magnetic_circuit.optimum_record_drive(carrier, **kwargs)
    luma = magnetic_circuit.optimum_record_drive(**kwargs)
    head = magnetic_circuit.head_transformer()
    return {
        "colour_under_optimum_drive": chroma["drive_ratio"],
        "luma_optimum_drive": luma["drive_ratio"],
        "ratio": chroma["drive_ratio"] / luma["drive_ratio"],
        "core_headroom": head["headroom"],
        "exceeds_the_core": bool(chroma["drive_ratio"] > head["headroom"]),
        "over_the_core_headroom": chroma["drive_ratio"] / head["headroom"],
        "clause_scope": SPECIFICATION["record_current"]["quote"],
        "chroma_has_no_current_clause": True,
        "verdict": "the colour under's own optimum is unreachable, and under "
                   "the bias account it is also unwanted: the chroma is a "
                   "small signal on the luma's bias, not a saturation "
                   "recording of its own",
    }


def chroma_recorded_layer(drive_ratio: Optional[float] = None,
                          system: str = "NTSC", **kwargs) -> Dict[str, object]:
    """THE CHROMA IS WRITTEN TO THE LUMA'S DEPTH, and bias is why.

    `magnetic_circuit.recorded_layer` already computes the colour under's
    layer as write-bound at the luma's drive rather than read-bound at its
    own `lambda / (2 pi)`, and says so: "the chroma's effective thickness is
    set by the RECORD CURRENT and not by its wavelength at all". What it
    could not say was WHY the luma's drive is the right one to put in the
    chroma's row, since the two signals have different currents.

    Bias supplies it. The anhysteretic process happens where the BIAS field
    exceeds the coercivity, so the chroma is recorded exactly and only in
    the layer the luma wrote. Its depth is a property of the LUMA's record
    current and the head's geometry, and of nothing belonging to the chroma.

    That is a head-side quantity governing a chroma-side amplitude, which is
    the relation Ethan asked whether these findings could reach.
    """
    drive = (magnetic_circuit.optimum_record_drive(
        **{k: v for k, v in kwargs.items()
           if k in ("gap_m", "spacing_m", "writing_speed_m_s")})["drive_ratio"]
        if drive_ratio is None else float(drive_ratio))
    carrier = colour_under.carrier_hz(system)
    chroma = magnetic_circuit.recorded_layer(drive, frequency_hz=carrier,
                                             **kwargs)
    luma = magnetic_circuit.recorded_layer(
        drive, frequency_hz=float(
            magnetic_circuit.spec_band_centres()["luma_geometric_centre_hz"]),
        **kwargs)
    return {
        "drive_ratio": drive,
        "chroma_layer": chroma,
        "luma_layer": luma,
        "same_bottom": bool(abs(chroma["layer_bottom_m"]
                                - luma["layer_bottom_m"]) < 1e-12),
        "chroma_read_depth_m": chroma["read_depth_m"],
        "chroma_is_write_bound": chroma["bound_by"] == "write",
        "why": "the anhysteretic process runs where the BIAS field exceeds "
               "the coercivity, so the colour under occupies the layer the "
               "luma wrote and nothing of its own wavelength enters",
    }


def head_to_tape_link(system: str = "NTSC", **kwargs) -> Dict[str, object]:
    """THE TERM THAT JOINS A HEAD FIELD TO A TAPE MAGNETISATION.

    Without bias the join is a threshold: the field either exceeds the
    coercivity or it does not, and a signal at the chroma's level does not.
    Nothing in `head_model`, `magnetic_circuit` or `tape_path` can carry a
    small signal across that boundary, which is why the colour under has
    never had a magnetic account in this tree at all.

    With bias the join is a SLOPE, and it factorises cleanly:

        M_chroma(y)  =  chi_ar  x  H_chroma(y)      over the depth where
                                                    H_luma(y) > Hc

    The tape owns `chi_ar`, the anhysteretic susceptibility - the initial
    slope of the anhysteretic curve, which `magnetic_circuit
    .differential_permeability` already computes at H = 0 as `Bs / Hc`. The
    head owns `H_chroma(y)`, through `head_efficiency` and
    `karlqvist_field`, and it owns the DEPTH through `write_depth_m` applied
    to the LUMA's drive. So the chroma's recovered amplitude is a tape slope
    times a head field integrated over a head-determined depth, and every
    one of those three is already in the tree.

    THE CONSEQUENCE THAT CAN BE TESTED: the chroma's recovered amplitude
    follows the LUMA's record current, because that current sets the depth
    and the bias level. The luma's current at the head is not constant - the
    JVC guide's record amplifier shapes it and the standard does not mention
    it - so the chroma's gain moves with the picture. Measured on the tape,
    +2.084 +- 0.048 dB per MHz of luma carrier; see `MEASURED`.

    NOT DERIVABLE FROM THE STANDARD, and the honest list is short. The
    standard states the coercivity, "approximately", and states neither the
    remanence nor the coating thickness nor the gap nor the turns
    (`vhs_specification.head_values_present`), so `chi_ar` here is
    `remanence / coercivity` with the remanence taken from the JVC guide's
    600-oersted class through `magnetic_circuit`'s own default. The SHAPE of
    the relation is specification-borne; its SCALE is not, and a number
    quoted from it is a number quoted from the guide.
    """
    coercivity = vhs_specification.SMPTE_32M["coercivity_a_m"]["value"]
    remanence = kwargs.pop("remanence_t", 0.15)
    drive = magnetic_circuit.optimum_record_drive(**{
        k: v for k, v in kwargs.items()
        if k in ("gap_m", "spacing_m", "writing_speed_m_s")})["drive_ratio"]
    gap = kwargs.get("gap_m", 0.30e-6)
    slope = float(magnetic_circuit.differential_permeability(
        0.0, remanence, coercivity))
    return {
        "anhysteretic_susceptibility_t_per_a_m": slope,
        "coercivity_a_m": coercivity,
        "remanence_t": remanence,
        "luma_drive_ratio": drive,
        "write_depth_m": magnetic_circuit.write_depth_m(drive, gap),
        "tape_owns": ("the anhysteretic susceptibility, the slope the small "
                      "signal rides"),
        "head_owns": ("the chroma's own field through the efficiency and the "
                      "Karlqvist field, and the DEPTH through the luma's "
                      "write depth"),
        "specification_supplies": ("the coercivity, 'approximately' "
                                   "(3.2.1.4), and nothing else of these"),
        "assumption": ("the remanence is the JVC guide's 600-oersted class "
                       "value carried by `magnetic_circuit`; SMPTE 32M does "
                       "not state a remanence and the word does not occur"),
    }


# --------------------------------------------------------------------------
# The residual: the product the bias does not remove
# --------------------------------------------------------------------------

def beat_hz(system: str = "NTSC") -> float:
    """The frequency the third-order product demodulates to: 2 f_c.

    A component at `f_y - 2 f_c` is, to the FM demodulator, the carrier with
    one sideband `2 f_c` away, and that demodulates as a tone at `2 f_c`.
    The guide says the same in its own words: "When the 2F dc component is
    detected and demodulated, a beat is produced with respect to the
    luminance signal and appears in the picture."
    """
    return 2.0 * colour_under.carrier_hz(system)


def intermodulation_products(luma_hz: Optional[float] = None,
                             system: str = "NTSC") -> Dict[str, object]:
    """Which products a residual nonlinearity makes of the two carriers, and
    which one matters.

    An odd memoryless nonlinearity acting on the sum of the FM carrier and
    the colour under makes `f_y +- 2 f_c` at third order and `2 f_y +- f_c`
    beside it. The guide names `F_D + 2F_DC` and the standard bounds
    `f_y - 2 f_c`, and what makes that one matter is not where the product
    sits but where it DEMODULATES: a sideband `2 f_c` from the carrier
    arrives in the picture at `2 f_c`, 1.2587 MHz, in the middle of the
    luma's own band. `2 f_y - f_c` demodulates to 3.238 MHz, at the edge of
    what a VHS luma channel carries at all. Neither third-order product
    lies inside the 3.4 to 4.4 MHz deviation - `f_y - 2 f_c` sits 0.79 MHz
    below the sync tip - which is why `inside_the_deviation` is reported
    rather than assumed.

    The even-order products - `f_y +- f_c`, second order - are a different
    animal and the taps separate them: measured, the beat at `f_c` is
    already at the RECORD tap and does not grow through the tape, while the
    beat at `2 f_c` is only at playback. Second order electrical, third
    order magnetic.
    """
    band = magnetic_circuit.spec_band_centres()
    carrier = colour_under.carrier_hz(system)
    luma = (float(band["luma_geometric_centre_hz"]) if luma_hz is None
            else float(luma_hz))
    low = float(band["luma_low_hz"])
    high = float(band["luma_high_hz"])
    rows = {}
    for name, hz, order in (("f_y - 2 f_c", luma - 2 * carrier, 3),
                            ("f_y + 2 f_c", luma + 2 * carrier, 3),
                            ("f_y - f_c", luma - carrier, 2),
                            ("f_y + f_c", luma + carrier, 2),
                            ("2 f_y - f_c", 2 * luma - carrier, 3)):
        rows[name] = {
            "hz": hz,
            "order": order,
            "inside_the_deviation": bool(low <= hz <= high),
            "demodulates_to_hz": abs(hz - luma),
        }
    return {
        "luma_hz": luma,
        "colour_under_hz": carrier,
        "products": rows,
        "the_one_that_matters": "f_y - 2 f_c",
        "clause": SPECIFICATION["bias_stated"]["clause"],
        "bound_db": (20.0, 25.0),
        "bound_scope": "S-VHS; the standard sets no such bound for baseline "
                       "VHS and the JVC guide sets none either",
    }


def interleave(system: str = "NTSC",
               rotation_per_line_degrees: float = 90.0
               ) -> Dict[str, object]:
    """WHY 40 f_H, AND THE SIGNATURE THAT MAKES THE PRODUCT MEASURABLE.

    The guide's argument, checked here in the format's own arithmetic. The
    colour under is rotated a quarter turn a line for crosstalk
    cancellation, so the carrier it records is `40 f_H +- f_H / 4`. The
    third-order product is at twice that: `80 f_H +- f_H / 2`. Eighty is an
    even multiple of the line rate, which is where the luminance's own
    spectrum sits, and the half-line offset puts the product exactly between
    those lines - "interleaved (1/2 line offset) ... and thereby visually
    reduced", which is the guide's stated reason for choosing 629.371 kHz.

    AND IT IS NOT HALF A LINE RATE EVERYWHERE. The offset is derived here
    rather than typed, as `2 (f_c / f_H + rotation)` reduced into a half
    line either side of zero, and for 625-line VHS the nominal carrier
    already carries an eighth of a line rate: 40.125 f_H doubles to 80.25,
    so the product lands a QUARTER of a line rate off and cycles over four
    lines rather than two. The half-line result belongs to the 525-line
    system, and the estimator MEASURES the offset (`line_rate_offset`)
    instead of assuming either.

    AND HALF A LINE RATE IS AN ALTERNATION FROM LINE TO LINE, which is a
    signature nothing in the picture shares. `alternating_split` uses it,
    and it is what makes a measurement possible on delivered material where
    a spectral line at 1.2587 MHz would otherwise sit under the picture's
    own content. MEASURED on the chroma-noise SP decode: the alternating
    part stands at 2.2651 IRE against a neighbouring-bin control of 0.0323
    IRE - 36.9 dB - while the STILL part, 0.4246 IRE, equals its control's
    0.4193 exactly. The product is entirely in the alternation, which is the
    guide's claim tested and confirmed on tape.
    """
    line_rate = colour_under.line_rate_hz(system)
    carrier = colour_under.carrier_hz(system)
    harmonic = carrier / line_rate
    rotation = float(rotation_per_line_degrees) / 360.0
    offset = (2.0 * (harmonic + rotation)) % 1.0
    if offset > 0.5:
        offset -= 1.0
    return {
        "line_rate_hz": line_rate,
        "colour_under_hz": carrier,
        "harmonic_of_line_rate": harmonic,
        "rotation_per_line_degrees": float(rotation_per_line_degrees),
        "carrier_offset_lines": rotation,
        "product_harmonic_of_line_rate": 2.0 * harmonic,
        "product_offset_lines": offset,
        "product_hz": 2.0 * carrier,
        # THE TOLERANCE IS NOT COSMETIC. `colour_under` carries the 625-line
        # carrier as the rounded 626 953 Hz rather than the exact 40.125 f_H,
        # so the derived offset is 0.250016 and not 0.25; a tolerance tight
        # enough to reject that would report "no cycle" for a system that
        # plainly has one.
        "alternates_line_to_line": bool(abs(abs(offset) - 0.5) < 1e-4),
        "lines_in_the_cycle": (2 if abs(abs(offset) - 0.5) < 1e-4
                               else (4 if abs(abs(offset) - 0.25) < 1e-4
                                     else None)),
        "why": "a half-line-rate offset is a sign change from line to line, "
               "which no picture content shares and which therefore makes "
               "the product separable from it",
        "clause": SPECIFICATION["interleave_reason"]["clause"],
        "verified_for": "525-line VHS, where the measured offset is -0.4978 "
                        "cycles a line on twelve fields of twelve; the "
                        "625-line rotation is NOT verified here and the "
                        "estimator measures the offset rather than assuming "
                        "this one",
    }


# --------------------------------------------------------------------------
# The estimator: the product in a delivered picture
# --------------------------------------------------------------------------

def line_amplitudes(lines, hz: float, sample_rate_hz: float,
                    start: int, stop: int) -> np.ndarray:
    """The COMPLEX amplitude at one frequency over each line's span.

    Complex because the product's phase is what carries the alternation, and
    a magnitude taken first destroys exactly the axis the measurement stands
    on. Each line's own mean is removed over the same span the sum runs on,
    so the window's response to a constant multiplies a constant that is
    zero - the trap `chroma_leakage.window_response_to_a_constant` documents.

    THE COEFFICIENT IS HALF THE PEAK of a real sinusoid, the usual one-sided
    convention, so a caller wanting the amplitude in the picture's own units
    doubles it. `product_in_the_luma` returns both rather than leaving the
    factor of two to be remembered.
    """
    block = np.asarray(lines, dtype=np.float64)[..., int(start):int(stop)]
    block = block - block.mean(axis=-1, keepdims=True)
    turn = np.exp(-2j * math.pi * float(hz)
                  * np.arange(block.shape[-1]) / float(sample_rate_hz))
    return block @ turn / block.shape[-1]


def alternating_split(amplitudes) -> Dict[str, object]:
    """Split a per-line complex series into its still and alternating parts.

    The still part is the mean and the alternating part is the mean against
    `(-1)**n`; they are the two ends of the line-rate transform and are
    orthogonal, so a component in one contributes nothing to the other. The
    product lives entirely in the alternating part (`interleave`), and the
    picture's own content lives in the still part, which is why the still
    part doubles as an in-band control.
    """
    values = np.asarray(amplitudes, dtype=np.complex128)
    index = np.arange(values.shape[-1])
    sign = (-1.0) ** index
    still = values.mean(axis=-1)
    alternating = (values * sign).mean(axis=-1)
    return {
        "still": still,
        "alternating": alternating,
        "still_magnitude": np.abs(still),
        "alternating_magnitude": np.abs(alternating),
        "lines": int(values.shape[-1]),
        "is_complex": True,
    }


def line_rate_offset(amplitudes) -> Dict[str, object]:
    """Where the per-line series actually sits, in cycles of the line rate.

    The transform along the LINE index, whose bin `k / lines` is an offset
    of `k / lines` line rates from the frequency the amplitudes were taken
    at. This is measured rather than assumed because the offset is a
    property of the format's chroma rotation and this arc has only verified
    the rotation for NTSC: the interleave argument predicts exactly one half
    for 525-line VHS, and reading that back out is the test.
    """
    values = np.asarray(amplitudes, dtype=np.complex128)
    lines = int(values.shape[-1])
    spectrum = np.fft.fft(values, axis=-1) / lines
    magnitude = np.abs(spectrum)
    peak = int(np.argmax(magnitude[..., :].reshape(-1, lines).mean(axis=0)))
    offset = peak / lines
    return {
        "spectrum": spectrum,
        "peak_bin": peak,
        "offset_cycles_per_line": offset if offset <= 0.5 else offset - 1.0,
        "peak_magnitude": float(
            magnitude.reshape(-1, lines).mean(axis=0)[peak]),
        "lines": lines,
        "is_complex": True,
    }


def product_in_the_luma(lines, sample_rate_hz: float, start: int, stop: int,
                        system: str = "NTSC",
                        carrier_hz: Optional[float] = None,
                        control_offsets: Sequence[int] = (-4, -3, -2,
                                                          2, 3, 4)
                        ) -> Dict[str, object]:
    """The tape-bias residual in a delivered luma field, with its control.

    THE WINDOW IS CHOSEN SO THE PRODUCT LANDS ON AN EXACT BIN. The output
    runs at 910 f_H, so 91 output samples hold exactly eight cycles of
    80 f_H; a span that is a whole multiple of 91 puts the product on a
    transform bin and the neighbouring bins 1.25 f_H away carry no product
    at all. That makes the control LOCAL - the same picture content, a few
    kilohertz away - rather than remote, and it is a better control than a
    fixed fraction of the frequency because it does not assume the picture's
    spectrum is flat over a wide span.

    Returns the alternating and still parts at the product's frequency and
    at each control bin. The reading is the ALTERNATING part against the
    control's alternating part; the still part is reported beside it and
    should equal its own control, which is the second, in-band check.
    `offset_cycles_per_line` is where the per-line series actually sits,
    MEASURED rather than assumed, so the half-line interleave the guide
    claims is read back out of the material instead of being built into the
    projection.

    WHAT THIS CANNOT DO ALONE, and it is refused rather than glossed: a
    delivered picture cannot say whether the product was written on the tape
    or made inside the decoder by squaring a colour-under leak, because both
    land at the same frequency with the same alternation. The record and
    playback taps settle it and nothing in a decode does; see
    `MEASURED['product_record_tap_db']`.
    """
    span = int(stop) - int(start)
    group = int(round(sample_rate_hz / colour_under.line_rate_hz(system)))
    if span <= 0:
        raise ValueError("the span must be positive")
    hertz = (beat_hz(system) if carrier_hz is None
             else 2.0 * float(carrier_hz))
    cycles = span * hertz / sample_rate_hz
    on_bin = abs(cycles - round(cycles)) < 1e-9
    bin_hz = sample_rate_hz / span
    predicted = abs(interleave(system)["product_offset_lines"])
    amplitudes = line_amplitudes(lines, hertz, sample_rate_hz, start, stop)
    reading = alternating_split(amplitudes)
    found = line_rate_offset(amplitudes)
    controls = [alternating_split(
        line_amplitudes(lines, hertz + offset * bin_hz, sample_rate_hz,
                        start, stop)) for offset in control_offsets]
    control_alt = float(np.mean([np.abs(c["alternating"]) for c in controls]))
    control_still = float(np.mean([np.abs(c["still"]) for c in controls]))
    alternating = float(np.abs(reading["alternating"]))
    return {
        "frequency_hz": hertz,
        "bin_hz": bin_hz,
        "samples_per_line": group,
        "span": span,
        "lands_on_a_bin": bool(on_bin),
        "cycles_in_the_span": cycles,
        "alternating": complex(reading["alternating"]),
        "still": complex(reading["still"]),
        "alternating_magnitude": alternating,
        "still_magnitude": float(np.abs(reading["still"])),
        "control_alternating": control_alt,
        "control_still": control_still,
        "alternating_peak": 2.0 * alternating,
        "control_alternating_peak": 2.0 * control_alt,
        "offset_cycles_per_line": found["offset_cycles_per_line"],
        "offset_magnitude": found["peak_magnitude"],
        # AGAINST THE OFFSET THIS SYSTEM PREDICTS, not against a half. The
        # 625-line product sits a quarter of a line rate off rather than a
        # half (`interleave`), so a flag hard-wired to one half would report
        # a correct 625-line measurement as a failure.
        "predicted_offset_lines": predicted,
        "interleaved": bool(min(abs(found["offset_cycles_per_line"]
                                    - predicted),
                                abs(found["offset_cycles_per_line"]
                                    + predicted))
                            < 1.0 / max(found["lines"], 1)),
        "ratio": alternating / max(control_alt, 1e-300),
        "still_ratio": float(np.abs(reading["still"])) / max(control_still,
                                                             1e-300),
        "lines": reading["lines"],
        "is_complex": True,
    }


def differential_gain_refused() -> Dict[str, str]:
    """WHY THE LUMA-DEPENDENT CHROMA GAIN IS NOT CORRECTED HERE.

    The tape imposes +2.084 dB of chroma gain per MHz of luma carrier, which
    over the standard's own 1.0 MHz deviation is a differential gain of 2.08
    dB and over the picture's 0 to 100 IRE is 1.49 dB, or 18.8 per cent. It
    is large and it is a real impairment. It is not corrected, for a reason
    that is about evidence rather than effort:

    A DECODE CANNOT TELL THE TAPE'S SHARE FROM THE SOURCE'S. The measurement
    above is a DIFFERENCE between two taps - the record tap reads -0.750
    dB/MHz on this material and the playback tap +1.334, and only their
    difference belongs to the tape. A decode holds the playback side alone,
    so the number it could measure is the sum of the source's differential
    gain, the record electronics' and the tape's, and correcting that sum
    would take out the source's own colour as though it were an artefact.

    AND THE TWO CANDIDATE MECHANISMS ARE COLLINEAR ON THIS DECK, so even
    with both taps the effect cannot be attributed. The luma's carrier
    frequency and its amplitude at the head move together, because the
    record amplifier's response is one fixed function of frequency: measured
    -3.133 +- 0.002 dB/MHz at the record tap in SP and -3.171 +- 0.003 in
    EP, the same shape in both modes. Whether the chroma's gain follows the
    bias FREQUENCY, which rises 13 per cent across the sweep, or the bias
    AMPLITUDE, which falls 1.57 dB across it, is not identifiable from any
    capture in this set.

    WHAT WOULD SETTLE IT: one recording of a flat field at a fixed luma
    level, made twice with the luma record current deliberately changed and
    the chroma current held - which separates the amplitude axis from the
    frequency axis by moving one of them alone. That is a bench operation on
    the recording deck, not a decode setting, and until it exists the size
    of the effect is measured and its cause is not.
    """
    return {
        "refused": "a luma-dependent chroma gain correction",
        "because": "a decode holds one tap, and the tape's share is a "
                   "difference between two",
        "and": "the bias frequency and the bias amplitude are collinear on "
               "this deck, so even the tap pair cannot attribute it",
        "would_settle_it": "a flat field recorded twice with the luma record "
                           "current changed and the chroma current held",
    }


def playback_compression_bounded(chroma_slope_db_per_mhz: float = 2.084,
                                 fm_slope_db_per_mhz: float = -0.095,
                                 fm_slope_error: float = 0.065
                                 ) -> Dict[str, object]:
    """THE ONE ALTERNATIVE TO A RECORD-TIME COUPLING, AND WHAT BOUNDS IT.

    The measured chroma gain rises as the luma carrier rises, and the luma
    carrier's own AMPLITUDE falls as it rises (the record amplifier's shape,
    JVC VTG82063 3.2.2). So a compressive playback preamplifier would
    produce the same sign without any tape involved: a large FM tone
    compresses the small chroma riding with it, and as the FM weakens the
    chroma is let up. That is the account this measurement has to exclude.

    IT IS EXCLUDED BY A FACTOR OF TWO THAT IS NOT NEGOTIABLE. For any
    memoryless compression - write it `y = x - e x^3`, which is the leading
    term of every one - a large tone of amplitude A suppresses its OWN
    fundamental by `(3/4) e A^2` and suppresses a small tone sharing the
    path by `(3/2) e A^2`. The small signal's gain moves EXACTLY TWICE as
    far as the large signal's, and both are measured here in the same
    capture. A compression large enough to move the chroma by the measured
    +2.084 dB/MHz must move the FM's own envelope by half of that, +1.042,
    and in the same direction.

    MEASURED, the FM's own envelope through the tape moves by -0.095 +-
    0.065 dB/MHz - the wrong sign and a sixteenth of the size. The
    compression account is out by sixteen standard errors, and it is out on
    a quantity taken from the same lines of the same two captures as the
    chroma reading, so nothing about the instrument is being asked to be
    the same twice.

    THE SAME ARGUMENT BOUNDS ANY MEMORYLESS NONLINEARITY ANYWHERE BETWEEN
    THE TAPS, not only the preamplifier, because the factor of two follows
    from the nonlinearity and not from where it sits. What it does NOT
    bound is a nonlinearity with memory that acts on the two bands
    differently - which is what the write process is, and which is why the
    finding is a record-time coupling rather than a bench fault.
    """
    required = 0.5 * float(chroma_slope_db_per_mhz)
    observed = float(fm_slope_db_per_mhz)
    error = max(float(fm_slope_error), 1e-12)
    return {
        "small_signal_over_large_signal": 2.0,
        "why_two": "a memoryless cubic suppresses a large tone's own "
                   "fundamental by (3/4) e A^2 and a small tone sharing the "
                   "path by (3/2) e A^2",
        "required_fm_slope_db_per_mhz": required,
        "observed_fm_slope_db_per_mhz": observed,
        "observed_error": error,
        "sigma_away": abs(required - observed) / error,
        "wrong_sign": bool(required * observed < 0.0),
        "excluded": bool(abs(required - observed) / error > 5.0),
        "not_bounded": "a nonlinearity WITH MEMORY that treats the two bands "
                       "differently - which is what a write process is",
    }


def product_order_refused() -> Dict[str, str]:
    """WHY THE PRODUCT'S ORDER IN THE CHROMA IS NOT REPORTED.

    A third-order product goes as `A_y * A_c^2`, so its level rises 6 dB for
    every 3 dB of chroma; a second-order AM coupling goes as `A_c` and rises
    3 dB. Measuring that exponent would settle which the beat at 2 f_c is
    without appealing to the guide at all, and it cannot be measured on this
    capture set. Two attempts, both recorded because the reason each failed
    is the same identifiability and not a defect in the estimator:

      * SPECTRALLY, per bar of 75 bars. A bar is 5 to 7 microseconds, whose
        transform bin is 250 kHz wide, so the reading at 1.2587 MHz is a
        quarter-megahertz slab of the demodulated luma containing the bar
        edges themselves. Measured exponent 0.594 +- 0.138 with an r-squared
        of 0.49 - a fit to the picture's own transients.

      * BY THE ALTERNATION, which is what separates the product from the
        picture in a delivered field. It does not work on the raw taps: the
        line starts carry the tape's time-base error, and a tenth of a
        microsecond is 45 degrees at 1.2587 MHz, so the per-line phases do
        not add. Measured, the alternating part at the PLAYBACK tap stood at
        1.4 times its control and the RECORD tap read higher still, which is
        the estimator reporting jitter.

    THE TWO PATTERNS THAT COULD ANSWER IT EACH LACK THE OTHER HALF. The
    chroma-noise capture shows the product at 145 times its control but
    holds one chroma level; 75 bars sweeps the chroma by a factor of five
    but chops each level into 109 output samples, and the product there
    stands at 1.46 times its control.

    WHAT WOULD SETTLE IT: a recording of a chrominance amplitude STAIRCASE
    with steps of a third of a line or longer, at one luma level - so the
    chroma sweeps while the picture and the time base do not. Nothing in
    this set has that shape.
    """
    return {
        "refused": "the product's order in the chroma amplitude",
        "because": "no capture in this set sweeps the chroma amplitude in "
                   "steps long enough to resolve the product against the "
                   "picture",
        "spectral_attempt": "exponent 0.594 +- 0.138, r-squared 0.49, a fit "
                            "to the bar edges",
        "alternation_attempt": "1.4 times its control at the playback tap "
                               "and higher at the record tap - the raw taps "
                               "carry the time-base error the alternation "
                               "needs removed",
        "would_settle_it": "a chrominance amplitude staircase with steps of "
                           "a third of a line or longer at one luma level",
    }


# --------------------------------------------------------------------------
# The measurements, with what took them
# --------------------------------------------------------------------------

MEASURED: Dict[str, object] = {
    "captures": "the matched record and playback taps of one Sony SLV-778HF "
                "(CN261 pin 1 recording, pin 2 playing back) at 50 MS/s, "
                "/testdata/test_patterns/vhs/{record,playback}/",
    "instrument": "the colour under's envelope and the FM's instantaneous "
                  "frequency read from the same capture with "
                  "`tap_transfer.analytic_in_band` and "
                  "`tap_transfer.locate_fields`, one microsecond of "
                  "averaging at 1.5 us steps across 14 to 59 us of every "
                  "usable line, per-field medians so a dropout cannot lever "
                  "a fit",
    "pattern": "zaroff-modulatedramp - a luminance ramp carrying a constant "
               "chrominance, which is the signal differential gain is "
               "defined on and which puts the luma's whole deviation inside "
               "one line",

    "chroma_gain_slope_db_per_mhz": {
        "SP": {"record": -0.750, "record_se": 0.009,
               "playback": 1.334, "playback_se": 0.047,
               "tape": 2.084, "tape_se": 0.048, "sigma": 43.3, "fields": 8},
        "EP": {"record": -0.341, "record_se": 0.218,
               "playback": 1.434, "playback_se": 0.108,
               "tape": 1.775, "tape_se": 0.243, "sigma": 7.3, "fields": 9},
        "per_head_SP_playback": {"first-field head": 1.272,
                                 "second-field head": 1.397},
    },
    "fm_envelope_slope_db_per_mhz": {
        "SP": {"record": -3.133, "playback": -3.228, "tape": -0.095,
               "tape_se": 0.065, "sigma": 1.4},
        "EP": {"record": -3.171, "playback": -3.248, "tape": -0.077,
               "tape_se": 0.044, "sigma": 1.7},
        "why_it_is_the_control": "the FM's own within-line envelope passes "
                                 "through the tape unchanged, so the "
                                 "instrument has no within-line systematic "
                                 "of its own at the 0.1 dB/MHz level while "
                                 "the chroma moves 2.08",
    },
    "y_only_control_db": {
        "record": -44.5, "playback": -33.2,
        "what": "the chroma band's level in a recording made with no "
                "chrominance at all, against the same band in the "
                "chroma-carrying recording; its own variation across the "
                "ramp moves the chroma reading by at most 0.03 dB",
    },
    "product_playback_tap_db": 9.28,
    "product_record_tap_db": 0.91,
    "product_y_only_playback_db": 0.21,
    "product_control_db": (0.45, 0.53),
    "product_playback_ire": 4.1823,
    "product_record_ire": 0.3065,
    "product_what": "the beat at 2 f_c in the demodulated luma, against the "
                    "median of its own neighbourhood; the control is the "
                    "same statistic at 0.83 x 2 f_c and reads 0.45 to 0.53 "
                    "dB in every one of the four cells",
    "product_playback_tap_db_ep": 8.57,
    "product_record_tap_db_ep": 0.17,
    "product_y_only_playback_db_ep": -0.03,
    "second_order_db": {
        "SP": {"record": 4.45, "playback": 3.10},
        "EP": {"record": 0.17, "playback": 3.22},
        "what": "the beat at f_c. AT SP it is already at the record tap and "
                "does not grow through the tape, which reads as an "
                "electrical coupling; AT EP it is at the floor at the record "
                "tap and stands at 3.22 dB at playback, so the medium makes "
                "one there too. The second order is therefore NOT a clean "
                "electrical-versus-magnetic split and is not reported as "
                "one; the third order is, at both speeds."},
    "delivered_picture_ire": {
        "chromanoise_SP": {"alternating": 2.2651, "still": 0.4246,
                           "control_alternating": 0.0323,
                           "control_still": 0.4193},
        "75bars_SP": {"alternating": 0.3758, "control_alternating": 0.2572},
        "75bars_EP": {"alternating": 0.5333, "control_alternating": 0.3770},
        "home": {"alternating": 0.0283, "control_alternating": 0.0241},
        "what": "the 80 f_H product in the delivered luma of "
                "/tmp/claude-1000/dod_fix_*.tbc, split into its still and "
                "alternating parts over 12 fields, in IRE through "
                "`precursor.output_scale` so `--level_adjust` is taken back "
                "out; 75 bars dilutes the product because each bar is only "
                "109 output samples long and the line is chopped into seven "
                "of them, and `home` carries no steady chroma at all",
    },
    "runtime": {
        "pooled_alternating_ire": 2.1746,
        "pooled_control_ire": 0.0150,
        "ratio": 145.12,
        "interleaved_fields": "4 of 4 at the first colour frame, and every "
                              "field of sixteen thereafter",
        "offset_cycles_per_line": 0.4980,
        "per_head_ire": (1.95, 2.40),
        "what": "the same statistic taken by the stage itself inside a "
                "decode of the chroma-noise SP capture with `--stages "
                "+tape_bias`, over 245 picture lines and a 728-sample span "
                "in groups of 91. RUNTIME AND OFFLINE AGREE to 4 per cent - "
                "2.1746 IRE against the offline 2.2651 on a decode taken "
                "with different flags - which is the parity check this arc "
                "has been caught missing before.",
        "per_head_note": "the two heads differ by 20 per cent, 1.95 against "
                         "2.40 IRE, alternating field by field. The product "
                         "is written by the head that wrote the track, so a "
                         "per-head difference is a difference between two "
                         "record heads' fields and is exactly the kind of "
                         "thing a bias mechanism should show; it is "
                         "reported and not explained.",
    },
}
"""Everything measured this session, with the capture and the instrument.

Reproduced by the scripts this arc ran on 2026-09-06; the tap figures come
from `zaroff-modulatedramp-NTSC-{SP,EP}` at both taps and the delivered-
picture figures from the decodes already in `/tmp/claude-1000`.
"""
