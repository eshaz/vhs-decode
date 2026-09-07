"""The VHS Hi-Fi (AFM) carriers as a timing probe independent of the video
path, and what the deck and the captures say about reaching them.

CORRECTION, 2026-09-07 - READ THIS BEFORE THE MEASUREMENT BELOW. Two of the
five records `MEASURED` reports on carry no Hi-Fi audio at all. Ethan,
2026-09-06: *"These sample may not have hifi audio. The zaroff samples
should have hifi carriers, home and cd will not."* So the `countdown` and
`home` rows searched for a signal that was never recorded, and their -50 dB
and -45 dB bounds are NOT evidence about leakage. They are withdrawn. Four
of the ten (record, channel) pairs in that table are those two records, and
the summary line "no AFM carrier at the video tap in any capture" overstates
what the valid subjects support.

WHAT SURVIVES, and it is weaker than what was withdrawn. The zaroff records
are the only valid subjects - the tape was recorded on a Sony SLV-778HF,
a Hi-Fi deck, which lays its carriers down whenever it records - and the
record-tap control is unaffected and is the strongest part of the original
measurement, because the audio carriers go to separate heads (clause 5.1)
and can never appear at the video record-current tap. Re-measured over all
32 zaroff playback captures and all 32 record captures rather than the one
programme sampled here, the honest bound is the HUMP bound and not the
narrow-line one: -40 dB on channel 1 for the chroma-bearing SP record,
-45 dB for the y-only one, and -35 dB on channel 2.

THREE THINGS THE WIDER RE-MEASUREMENT FOUND, all recorded in
`capture_alignment.WIDE_SWEEP` with the arithmetic:

  * THE NARROW-LINE STATISTIC IS NOT SAFE. It divides a peak by the scatter
    of its own neighbourhood, so a flat neighbourhood manufactures
    significance. Ten of 64 playback pairs cross 3 sigma and none of 64
    record-tap control pairs does - yet the control shows LARGER bumps
    (up to 27.1 dB over its floor against the firing pairs' 11.2 dB
    median), and only its rougher surround stops it firing.
  * THE COLOUR-UNDER SECOND HARMONIC IS MISATTRIBUTED. The confound named
    below at 1.258741 MHz was tested on the y-only recordings, which carry
    no chrominance: removing 18.93 dB of colour-under fundamental removed
    0.14 dB there, where a square-law product would have lost about 38 dB.
    What sits at that frequency is the luma lower-sideband continuum and
    the tape's own noise. Excluding it by name buys nothing.
  * THE OPERATIVE STATISTIC REVERSES. The argument below - that the zaroff
    recordings had no audio connected, so the carrier is unmodulated and
    the narrow line is the bound to quote - does not hold, because clause
    5.6's 2:1 logarithmic compressor is at maximum gain on silence. Planted
    at -45 dB, an unmodulated carrier is detected (line rise +3.03) and one
    with 500 Hz of deviation is not (+2.74); 500 Hz is what that compressor
    makes of an input 84 dB down. The hump statistic is nearly indifferent
    to deviation and is the conservative one.

This block is kept rather than the numbers being edited away, because this
arc's rule is that a withdrawn claim stays visible with its reason. See
`capture_alignment.WITHDRAWN`, `capture_alignment.carrier_route_bound` and
docs/CAPTURE_ALIGNMENT.md.

Ethan (2026-09-05): *"HiFi carriers - independent timing probe. AFM carriers
~1.3/1.7 MHz, different heads, different azimuth, deeper penetration.
Wavelengths ~3.4-4.5 um, between chroma (9.2 um) and luma (1.45 um). Third
wavelength cluster helps the loss-mechanism separation that two clusters
can't resolve. Carriers derive from the same crystal as video at different
divider ratios, so the ratio is fixed by design. Departure from nominal is
mechanical, not electronic - a TBE readout independent of the video path.
Cross-check video-derived TBE against HiFi-derived TBE to catch systematic
error in either."*

WHAT THE FORMAT SPECIFIES. SMPTE 32M-2004 clause 5, "FM audio recording
characteristics", for the 525-line system (text fetched from
decode-orc/analogue-video-specifications, docs/vhs/SMPTE-32M-2004); JVC
Video Technical Guide VTG82063 section 7 for the 625-line pair, which SMPTE
32M does not cover:

  5.1   depth multiplex: the FM audio is recorded INTO the magnetic layer by
        separate heads on the drum, a specified number of fields before the
        video, which is then recorded over it on the surface; the audio and
        video gaps have different azimuths to keep the two apart.
  5.4   FM audio azimuth +/- 30 degrees 30 minutes; table 5: the audio
        head's azimuth direction is OPPOSITE to the coincident video head's
        in SP and LP, and the SAME in EP.
  5.7   pre-emphasis 56 us +/- 11 us and 20 us +/- 4 us.
  5.8   centre carriers: channel 1 = 1.3 MHz +/- 10 kHz, channel 2 = 1.7 MHz
        +/- 10 kHz. JVC 7.2.3 (625-line): 1.4 and 1.8 MHz +/- 10 kHz.
  5.9   deviation: +/- 150 kHz maximum, +/- 50 kHz reference at 400 Hz.
  5.10  the two channels are modulated with the same polarity.
  5.12  channel 1 carries the left (main) signal, channel 2 the right (sub).
  table 5: audio track width A1, A2 = 0.010 to 0.029 mm in SP (B at least
        0.012 mm LP, 0.016 mm EP); the audio is recorded 0 to 2 fields
        earlier than the video in SP.

The two carriers stand in the nominal ratio 17:13 (525) and 9:7 (625). The
repository's own `vhsdecode/format_defs/vhs.py` carries the same four
frequencies as `fm_audio_channel_0_freq` and `fm_audio_channel_1_freq`
(lines 207-208 and 340-341), and `vhsdecode/hifi/HiFiDecode.py` as
`AFEParamsNTSCVHS.LCarrierRef`/`RCarrierRef` and the PAL pair, where the
carrier tracker is clamped to +/- 10 kHz of them - the tolerance of 5.8.

WHAT THE SPECIFICATION DOES NOT SAY. Nothing in SMPTE 32M-2004 states how
the carriers are generated: the words crystal, oscillator and divider do
not appear in clause 5, which specifies only the centre frequencies and
their tolerance. Ethan's premise that the carriers "derive from the same
crystal as video at different divider ratios" is therefore a claim about a
DECK, not about the format, and has to be settled deck by deck. For the
deck that made this repository's test captures it is settled below, and
the answer is no.

WHAT THE DECK DOES, read from the Sony SLV-777HF/778HF/788HF service manual
schematic (/testdata/test_patterns/vhs/Sony Slv777Hf 778Hf 788Hf
Schematic.pdf; the deck that made every test-pattern capture in this
repository). Page numbers are the manual's printed ones; the PDF page is
given in brackets. Every item below was read off the rendered drawing on
2026-09-05:

  page 4-17/4-18 [PDF 5], MA-327 (5/8) "AFM AUDIO": IC360 is a Philips
        TDA9615H/N1 "AFM AUDIO PROCESS". Its block diagram runs, for each
        channel, NR - LPF - VCO - LPF into a summing node that feeds pin 36
        "AFM REC"; pin 37 "AFM PB" enters through a BPF to the same VCO
        blocks (as PLL demodulators) and a LEVEL DET that drives pin 39
        "AF ENV". The VCOs are timed by external parts on pins 23 to 33,
        the "Rch" block on 33-31 and "Lch" on 26-23, symmetric pair for
        pair: C399 and C398 (3900 pF) at the two VCO pins, R385 and R384
        (56 kilohm), R372 and R370 (printed "4700", so 4.7 kilohm - the
        sheet's note reads "All resistor are in ohms ... k-ohm: 1000
        ohm"); R371 (39 kilohm +/- 0.5 percent) feeds pin 28 "I ref" and
        pin 27 is the ground return, C379 (2.2 uF) sits at "V ref"
        (pin 29). Pins 41/42 are SDA/SCL (R391/R392 1 kilohm) from the
        system controller's "I2C DATA 1" and "I2C CLOCK 1" (page 3/8);
        pin 40 "AF SW P-bar" takes the audio head-switching pulse "AF SWP"
        from the same page. Of the 44 pins, 1 to 11 are the baseband audio
        inputs (TU/L1 to L4, left and right) and 12 to 22 the outputs,
        35 is 12 V and 43 "VSS D". NO CRYSTAL AND NO CLOCK LINE ENTERS THE AFM SECTION -
        verified by reading the whole sheet at 350 and 600 dpi on
        2026-09-05, the only signals crossing IC360's boundary being FM
        REC/FM PB to the head amplifier (page 1/8), AF ENV / AF SWP /
        I2C DATA 1 / I2C CLOCK 1 to the servo and system controller
        (page 3/8), the baseband audio, and 12 V. The carrier frequencies
        are therefore set by the VCOs' RC timing and current reference and
        by I2C configuration, which is how the +/- 10 kHz of 5.8 is held.
  page 4-8/4-9/4-10 [PDF 2], MA-327 (2/8): the video processor IC201 (Sanyo
        LA71534M-MPB, "Y/C process, normal audio process") has ITS OWN
        crystal, X202 3.579545 MHz; the waveform table marks IC201 pin 56
        at 3.579545 MHz in both REC and PB.
  page 4-11/4-12/4-13 [PDF 3], MA-327 (3/8): the servo/system controller
        IC160 (Mitsubishi M37777M7A235GP-C) runs from a 16 MHz crystal
        (pin 38 waveform) and a 32.768 kHz one (pin 41), drives the drum
        and capstan servos and talks to IC360 over I2C. Its pin 19 is
        marked "(HiFi STEREO MODEL) 5 Vp-p 30 Hz": the AFM heads' own
        head-switching pulse, separate from the video one.
  page 4-5/4-6/4-7 [PDF 1], MA-327 (1/8) "REC/PB AMP": connector CN260
        (13 pins) brings the drum's rotary-transformer windings in - pins 6
        to 13 are the four video heads (SP CH2 S/F, EP CH1 S/F, SP CH1 S/F,
        EP CH2 S/F), pins 1 to 3 are "AUDIO CH2", "AUDIO REC", "AUDIO CH1",
        pin 4 "FE" (the flying erase head), pin 5 "GND(SW)". The video
        windings go ONLY to IC260 (Hitachi HA118195NT "VIDEO REC/PB AMP");
        the audio windings go ONLY to IC340 (Sanyo LA7256 "AFM REC/PB
        AMP", with its own Ch1/Ch2 head amplifiers, SP/EP switching, record
        current amplifier and AGC). The test connector CN261 "FOR CHECK"
        carries 1 REC CURR, 2 PB RF, 3 RF SWP, 4 GND from the VIDEO
        amplifier; the AFM amplifier's own test connector is CN341 (1 HF
        ADJ, 2 GND, 3 FM PB). The test-pattern captures were taken at
        CN261 pin 1 (record) and pin 2 (playback) by a Rigol DS1202Z-E
        oscilloscope at 50 MS/s, 8 bit, with no buffer or amplifier
        (/testdata/test_patterns/vhs/readme.txt).

So on this deck the premise "same crystal, fixed ratio" does NOT hold for
recordings it makes: the two AFM carriers are free-running RC-timed VCOs,
each within the +/- 10 kHz the standard allows, and their ratio is a
per-recording constant to be measured, not 17:13 by construction. What
survives is the part that matters for timing: an RC oscillator drifts
thermally over minutes, so everything at the transport's rates - the drum
at 29.97 Hz, the capstan near 4.8 Hz, their harmonics - is mechanical, while
the MEAN departure from nominal is electronic (the recording machine's
oscillator setting) and must be reported separately rather than read as a
speed error. `time_base_error` does exactly that split.

And the video/audio split is at the HEADS: separate windings, separate
amplifiers, separate test points. There is no wideband tap ahead of the
split on this deck - nor on any deck built to 5.1, which requires separate
audio heads - so the AFM carriers can reach the video tap only by crosstalk:
the video head reading the deeper layer through the azimuth difference, or
coupling in the rotary transformer or between amplifiers. That is a
measurement, and it was made: see `MEASURED` and docs/HIFI_CAPTURE_PATH.md.

CONVENTIONS. The time-base error epsilon here is the fractional PERIOD
error of the playback, positive when the tape runs slow: a line then lasts
(1 + epsilon) of its nominal period and every recorded frequency f plays
back as f / (1 + epsilon). Both readouts, the sync-crossing periods and the
AFM carrier frequencies, are expressed in that one convention so they can
be subtracted.

THE AUDIO PROBLEM, stated rather than hidden. The audio deviation is 50 kHz
at the reference level - 3.8 percent of a 1.3 MHz carrier - against a wow
of about 0.04 percent (the measured drum-rate wow of this transport,
`carrier_tbc`). Audio above the flutter band is removed by averaging over a
line and low-passing; audio INSIDE the flutter band (bass at the drum rate
and its harmonics) is not, and no single-carrier estimator can tell a 30 Hz
bass note from a 30 Hz wow. Two carriers can, because the speed error
scales with the carrier (delta f = -epsilon f0) while the audio deviation is
the same number of hertz on both (5.9, 5.10): the DIFFERENCE of the two
carriers' deviations is free of any audio common to both channels.
`time_base_error_pair` is that estimator; its residual error is the stereo
difference (R - L) inside the flutter band divided by the 400 kHz carrier
spacing, and `audio_leakage_bound` states the single-carrier leakage for a
given tone so a test can hold it.
"""

import inspect
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from vhsdecode.models import colour_under, head_model, magnetic, rf_stages
from vhsdecode.models import sync_geometry, transport_model
from vhsdecode.models.loss_separation import SPEC_RECORDED_DEPTH_M


# --------------------------------------------------------------------------
# The specification, with provenance. Nothing below is a tuning constant.
# --------------------------------------------------------------------------

# SMPTE 32M-2004 5.8 (525-line); JVC VTG82063 7.2.3 (625-line). The
# repository's format_defs/vhs.py carries the same figures.
CARRIER_HZ: Dict[str, Tuple[float, float]] = {
    "525": (1.3e6, 1.7e6),
    "625": (1.4e6, 1.8e6),
}
CARRIER_TOLERANCE_HZ = 10e3            # 5.8: "+/- 10 kHz"
MAX_DEVIATION_HZ = 150e3               # 5.9
REFERENCE_DEVIATION_HZ = 50e3          # 5.9: "+/- 50 kHz at 400 Hz"
REFERENCE_MODULATION_HZ = 400.0        # 5.9
AZIMUTH_DEGREES = 30.0 + 30.0 / 60.0   # 5.4: "+30 deg 30' and -30 deg 30'"
# table 5, "azimuth angle direction: audio head vs video head"
AZIMUTH_DIRECTION_VS_VIDEO: Dict[str, str] = {
    "SP": "opposite", "LP": "opposite", "EP": "same"}
# table 5, "FM audio track width", in metres (the standard prints mm)
TRACK_WIDTH_M: Dict[str, Tuple[float, Optional[float]]] = {
    "SP": (10e-6, 29e-6), "LP": (12e-6, None), "EP": (16e-6, None)}
# table 5, "recording time sequence": fields the audio precedes the video by
RECORDED_FIELDS_EARLIER: Dict[str, Tuple[float, float]] = {
    "SP": (0.0, 2.0), "LP": (1.0 / 3.0, 2.0 + 1.0 / 3.0),
    "EP": (1.0 + 1.0 / 3.0, 3.0 + 1.0 / 3.0)}
PRE_EMPHASIS_S: Tuple[float, float] = (56e-6, 20e-6)   # 5.7
CHANNEL_USE: Dict[int, str] = {1: "left (main)", 2: "right (sub)"}   # 5.12
# JVC VTG82063 7.1: the audible range the system was designed to carry,
# "20 Hz to 20 kHz". SMPTE 32M states no audio bandwidth in its text.
AUDIO_BAND_HZ: Tuple[float, float] = (20.0, 20e3)

# JVC VTG82063 7.2.1: the video head's 0.3 um gap records to a depth of about
# 0.3 um. That layer lies OVER the audio, so a video head reading the audio
# sees it through this much extra spacing. Imported so it cannot drift from
# the loss-separation module's copy.
VIDEO_LAYER_DEPTH_M = SPEC_RECORDED_DEPTH_M

# SMPTE 32M 3.9.1.1.4: reference white (100 IRE) and reference sync (-40
# IRE) name the luma carriers; blanking is 0 IRE by the definition of the
# scale, so the blanking carrier sits 40/140 of the deviation above the tip.
SYNC_IRE = -40.0
WHITE_IRE = 100.0
BLANKING_IRE = 0.0

# What the schematic records for the deck the test-pattern captures came
# from. A description, not a parameter; every entry was read off the drawing
# (module docstring, "WHAT THE DECK DOES").
DECK_CARRIER_GENERATION: Dict[str, object] = {
    "deck": "Sony SLV-777HF/778HF/788HF",
    "source": "Sony service manual schematic, MA-327 board",
    "afm_processor": "IC360 Philips TDA9615H/N1-557, 44 pins, marked 'AFM "
                     "AUDIO PROCESS' (page 4-17/4-18, PDF page 5)",
    "afm_carrier_source": ("two RC-timed VCOs: C398/C399 3900 pF, R384/R385 "
                           "56 k, R370/R372 4.7 k on pins 23-33, R371 39 k "
                           "+/- 0.5% at pin 28 'I ref' (pin 27 ground), set "
                           "over I2C (pins 41/42 SDA/SCL); no crystal, no "
                           "clock line anywhere on the sheet"),
    "afm_head_switching_pulse": "IC160 pin 19 'AF SWP' -> IC360 pin 40 'AF SW P-bar'",
    "video_reference": "IC201 Sanyo LA71534M-MPB with X202 3.579545 MHz "
                       "(page 4-8/4-9/4-10, PDF page 2; pin 56 waveform)",
    "system_controller": "IC160 Mitsubishi M37777M7A235GP-C, 16 MHz (pin 38) "
                         "and 32.768 kHz (pin 41) crystals (page 4-11/4-12/4-13, "
                         "PDF page 3)",
    "ratio_fixed_by_design": False,
    "video_head_amplifier": "IC260 Hitachi HA118195NT, heads on CN260 pins "
                            "6-13, test point CN261 pin 2 'PB RF' (pin 1 'REC "
                            "CURR', pin 3 'RF SWP')",
    "afm_head_amplifier": "IC340 Sanyo LA7256, heads on CN260 pins 1-3, "
                          "test point CN341 pin 3 'FM PB'",
    "split": "at the heads: separate rotary-transformer windings, separate "
             "amplifiers, separate test points; no tap carries both",
    "test_pattern_capture": "CN261 pins 1/2 into a Rigol DS1202Z-E at 50 MS/s, "
                            "8 bit, no buffer (test_patterns/vhs/readme.txt)",
}


def _system(system: str) -> str:
    name = str(system).upper()
    if name in ("NTSC", "525", "M", "M/NTSC"):
        return "525"
    if name in ("PAL", "625", "SECAM", "B", "G", "I"):
        return "625"
    raise ValueError(f"no Hi-Fi carriers declared for system {system!r}")


def _colour_system(system: str) -> str:
    return "NTSC" if _system(system) == "525" else "PAL"


def carriers_hz(system: str = "NTSC") -> Tuple[float, float]:
    """The two centre carriers, channel 1 then channel 2."""
    return CARRIER_HZ[_system(system)]


def nominal_ratio(system: str = "NTSC") -> Fraction:
    """The carriers' ratio as the specification's integers: 17/13 for the
    525-line pair, 9/7 for the 625-line pair. This is what a common-crystal
    divider WOULD fix; whether a deck fixes it is `DECK_CARRIER_GENERATION`'s
    question, and for the SLV-778HF the answer is no."""
    low, high = carriers_hz(system)
    return Fraction(int(round(high)), int(round(low)))


def carrier_half_width_hz() -> float:
    """How far either side of a centre carrier the recorded signal can
    reach: Carson's rule for the maximum deviation and the top of the
    audio band, plus the centre tolerance. 180 kHz."""
    return MAX_DEVIATION_HZ + AUDIO_BAND_HZ[1] + CARRIER_TOLERANCE_HZ


def reference_half_width_hz() -> float:
    """The reach of a carrier modulated at the REFERENCE level (5.9): the
    reference deviation plus the top of the audio band, 70 kHz. This is the
    width of the hump a normally modulated carrier makes in a spectrum."""
    return REFERENCE_DEVIATION_HZ + AUDIO_BAND_HZ[1]


def luma_band_hz(system: str = "NTSC") -> Tuple[float, float]:
    """The luma FM band used as the power reference for every level in this
    module: sync tip minus the deviation to peak white plus the deviation
    (`rf_stages.VHS_CARRIER_HZ`), which holds the first-order sidebands."""
    tip, white = rf_stages.VHS_CARRIER_HZ[_system(system)]
    dev = rf_stages.deviation_hz((tip, white))
    return tip - dev, white + dev


def mechanics(system: str = "NTSC", tape_speed: str = "SP") -> Dict[str, float]:
    mech = head_model.mechanics_for("VHS", _colour_system(system), tape_speed)
    if mech is None:
        # the format's writing speed does not change with tape speed - the
        # drum turns at the same rate - so the SP mechanics serve
        mech = head_model.mechanics_for("VHS", _colour_system(system), "SP")
    return mech


def wavelength_m(system: str = "NTSC", tape_speed: str = "SP",
                 speed_m_s: Optional[float] = None) -> Tuple[float, float]:
    """Recorded wavelength of each carrier, lambda = v / f, at the stated
    writing speed (5.80 m/s NTSC, 4.85 m/s PAL; JVC VTG82063 section 1 as
    carried by `head_model`). 4.4615 and 3.4118 um for the 525-line pair:
    Ethan's "~3.4-4.5 um", between the colour-under's 9.2 um and the luma's
    1.3-1.7 um. JVC 7.2.1 rounds them to "about 4 um" and "about 3.4 um"."""
    speed = float(speed_m_s) if speed_m_s else head_model.writing_speed(
        mechanics(system, tape_speed))
    low, high = carriers_hz(system)
    return speed / low, speed / high


def read_depth_m(system: str = "NTSC", tape_speed: str = "SP",
                 speed_m_s: Optional[float] = None) -> Tuple[float, float]:
    """The depth each carrier is read from, lambda / (2 pi): the playback
    head weights a layer at depth y by exp(-2 pi y / lambda), and it is
    also the self-demagnetisation limit of docs/MATHEMATICS.md 8a
    (`magnetic.DEMAGNETISATION_DIVISOR`). 0.710 and 0.543 um for the
    525-line pair - two to three times the luma's 0.21-0.27 um and below
    the chroma's 1.47 um, which is the "deeper penetration" the depth
    multiplex relies on. JVC 7.2.4 states the design optimum as lambda / 4,
    0.68 to 0.89 um for the 625-line pair; the 2 pi figure is the cap."""
    low, high = wavelength_m(system, tape_speed, speed_m_s)
    return (low / magnetic.DEMAGNETISATION_DIVISOR,
            high / magnetic.DEMAGNETISATION_DIVISOR)


def wavelength_clusters(system: str = "NTSC", tape_speed: str = "SP"
                        ) -> Dict[str, Dict[str, float]]:
    """The three wavelength clusters a Hi-Fi tape offers, in metres, with
    the Hi-Fi pair sitting between the other two. The luma pair is the FM
    carrier at sync tip and peak white (`rf_stages.VHS_CARRIER_HZ`), the
    chroma the colour-under carrier (`colour_under.carrier_hz`)."""
    sysname = _system(system)
    speed = head_model.writing_speed(mechanics(system, tape_speed))
    tip, white = rf_stages.VHS_CARRIER_HZ[sysname]
    cu = colour_under.carrier_hz(_colour_system(system))
    low, high = carriers_hz(system)
    return {
        "chroma": {"colour_under": speed / cu},
        "hifi": {"channel_1": speed / low, "channel_2": speed / high},
        "luma": {"sync_tip": speed / tip, "peak_white": speed / white},
    }


# --------------------------------------------------------------------------
# What a VIDEO head would read of the audio layer: the crosstalk expectation
# --------------------------------------------------------------------------

def relative_azimuth_degrees(tape_speed: str = "SP") -> float:
    """The angle between a video head's gap and the audio track's
    recorded azimuth: 30.5 + 6 = 36.5 degrees where the directions are
    opposite (SP, LP), 30.5 - 6 = 24.5 where they are the same (EP)."""
    video = head_model.FORMAT_MECHANICS[("VHS", "NTSC", "SP")]["azimuth_degrees"]
    direction = AZIMUTH_DIRECTION_VS_VIDEO[str(tape_speed).upper()]
    return AZIMUTH_DEGREES + video if direction == "opposite" else AZIMUTH_DEGREES - video


def azimuth_crosstalk_db(system: str = "NTSC", tape_speed: str = "SP",
                         channel: int = 1,
                         overlap_width_m: Optional[float] = None
                         ) -> Dict[str, float]:
    """The attenuation with which a video head reads the audio carrier,
    relative to a head at the audio azimuth, from the two mechanisms the
    depth multiplex relies on (JVC 7.2.2 and 7.2.4; the forms are
    `head_model`'s): the azimuth loss sinc(w tan(theta) / lambda) across the
    overlap width w, and the spacing loss exp(-2 pi d / lambda) through the
    video layer of depth d recorded over it.

    The overlap width defaults to the audio track width's bounds (table 5),
    because that, not the wider video head, is where audio flux exists;
    both bounds are returned. Computed: in SP at 1.3 MHz, -15.5 dB (10 um)
    to -28.6 dB (29 um) of azimuth loss and -3.7 dB of spacing loss, so a
    video head would read the channel-1 carrier 19 to 32 dB below what the
    audio head reads; at 1.7 MHz, -22.6 to -28.0 dB and -4.8 dB, 27 to 33
    dB in all; in EP (16 um, 24.5 degrees) 19 and 29 dB. All of it IF the
    buried layer survived the overwrite intact, which JVC 7.2.4 says it
    does. That is an upper envelope on the crosstalk, and the measurement
    (`MEASURED`) sits under it by 20 dB or more."""
    lam = wavelength_m(system, tape_speed)[int(channel) - 1]
    theta = np.deg2rad(relative_azimuth_degrees(tape_speed))
    depth = VIDEO_LAYER_DEPTH_M
    spacing_db = -head_model.DB_PER_NEPER * 2.0 * np.pi * depth / lam
    widths = ([float(overlap_width_m)] if overlap_width_m else
              [w for w in TRACK_WIDTH_M[str(tape_speed).upper()] if w])
    out = {"wavelength_m": lam, "relative_azimuth_degrees": float(np.degrees(theta)),
           "spacing_loss_db": float(spacing_db)}
    for w in widths:
        x = w * np.tan(theta) / lam
        s = np.sinc(x)   # numpy's sinc is sin(pi x) / (pi x)
        az = 20.0 * np.log10(max(abs(float(s)), 1e-12))
        out[f"azimuth_loss_db_{w*1e6:.0f}um"] = float(az)
        out[f"total_db_{w*1e6:.0f}um"] = float(az + spacing_db)
    return out


# --------------------------------------------------------------------------
# Is a carrier there? The presence detector and its planted calibration
# --------------------------------------------------------------------------

# Bins a median needs inside the +/- 10 kHz centre tolerance. The resulting
# bin (about 830 Hz) is coarser than the drum-rate wow's broadening of a
# carrier (4e-4 x 1.3 MHz = 520 Hz), so a real unmodulated line stays in one
# bin instead of being split across several.
SPECTRUM_BINS_PER_TOLERANCE = 24
# The significance a planted carrier must reach, over what the record
# already shows, to count as detectable at that level.
DETECTION_SIGMA = 3.0


def line_harmonic_offset_hz(frequency_hz: float, system: str = "NTSC"
                            ) -> Dict[str, float]:
    """How far a frequency sits from the nearest multiple of the line rate.

    Anything the picture repeats every line - the sync pulses, a static test
    pattern's content, the colour-under's line-locked burst - appears as a
    picket fence of harmonics of f_H (15.734 kHz for 525 lines, SMPTE 32M
    4.3.1.2). Those pickets are the narrow-line statistic's competition, and
    a peak that sits ON one is not evidence of a carrier.

    The AFM carriers are not locked to f_H - on the SLV-778HF they come from
    free-running RC VCOs, `DECK_CARRIER_GENERATION` - so where a peak lands
    relative to the grid is informative. `offset_hz` is the FREQUENCY minus
    the nearest picket, so it is negative below one. Computed for 525
    lines: the nearest picket to 1.3 MHz is the 83rd at 1.305944 MHz
    (offset -5944 Hz); the nearest to 1.7 MHz is the 108th at 1.699301 MHz
    (offset +699 Hz). A picket therefore falls inside the +/- 10 kHz
    tolerance window at BOTH nominal carriers, so on a static pattern the
    narrow-line statistic must be read together with this offset."""
    rate = colour_under.line_rate_hz(_colour_system(system))
    order = int(round(float(frequency_hz) / rate))
    return {"harmonic": order, "harmonic_hz": order * rate,
            "offset_hz": float(frequency_hz) - order * rate,
            "spacing_hz": float(rate)}


def colour_under_harmonic_offset_hz(frequency_hz: float, system: str = "NTSC"
                                    ) -> Dict[str, float]:
    """The same for harmonics of the colour-under carrier (629.371 kHz,
    SMPTE 32M 3.9.2.1.4), which any nonlinearity in the record or playback
    chain produces. Computed for 525 lines: the SECOND harmonic is at
    1.258741 MHz, 41.26 kHz below the 1.3 MHz AFM carrier - outside the
    +/- 10 kHz tolerance window but well inside the +/- 70 kHz band the hump
    statistic integrates, so it is a confound for channel 1 and must be
    excluded or reported. The nearest to 1.7 MHz is the third at 1.888 MHz,
    188 kHz away, outside even the +/- 180 kHz reach: channel 2 is clear of
    it."""
    carrier = colour_under.carrier_hz(_colour_system(system))
    order = max(1, int(round(float(frequency_hz) / carrier)))
    return {"harmonic": order, "harmonic_hz": order * carrier,
            "offset_hz": float(frequency_hz) - order * carrier,
            "carrier_hz": float(carrier)}


def confounds(centre_hz: float, system: str = "NTSC",
              reach_hz: Optional[float] = None,
              hump_half_width_hz: Optional[float] = None,
              tolerance_hz: Optional[float] = None) -> List[Dict[str, object]]:
    """The deterministic neighbours inside a carrier's reach, named, with
    where each falls: inside the centre tolerance (it competes with the
    narrow line), inside the hump band (it competes with the hump), or only
    in the surround (it inflates the floor's scatter but a median rejects
    it). The colour-under harmonics and the line-harmonic pickets are the
    two families; the luma lower sideband is a continuum, not a line, and
    enters the floor rather than this list."""
    tol = float(tolerance_hz or CARRIER_TOLERANCE_HZ)
    hump = float(hump_half_width_hz or reference_half_width_hz())
    reach = float(reach_hz or carrier_half_width_hz())
    found: List[Dict[str, object]] = []
    cu = colour_under.carrier_hz(_colour_system(system))
    line = colour_under.line_rate_hz(_colour_system(system))
    for name, spacing in (("colour-under harmonic", cu), ("line harmonic", line)):
        low = int(np.floor((float(centre_hz) - reach) / spacing))
        high = int(np.ceil((float(centre_hz) + reach) / spacing))
        for order in range(max(low, 1), high + 1):
            offset = order * spacing - float(centre_hz)
            if abs(offset) > reach:
                continue
            found.append({
                "what": name, "harmonic": order, "at_hz": order * spacing,
                "offset_hz": float(offset),
                "in_tolerance": bool(abs(offset) <= tol),
                "in_hump": bool(abs(offset) <= hump),
            })
    return sorted(found, key=lambda item: abs(item["offset_hz"]))


def power_spectrum(samples, sample_rate_hz: float, resolution_hz: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power density (Hann, half overlap) at about `resolution_hz`
    per bin; the segment is the power of two that reaches it."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    nperseg = 1 << int(np.ceil(np.log2(max(float(sample_rate_hz) / float(resolution_hz), 16.0))))
    nperseg = int(min(nperseg, len(x)))
    return signal.welch(x, fs=float(sample_rate_hz), nperseg=nperseg,
                        noverlap=nperseg // 2, window="hann",
                        scaling="density", detrend=False)


def band_power(freq: np.ndarray, density: np.ndarray, low_hz: float, high_hz: float) -> float:
    sel = (freq >= float(low_hz)) & (freq <= float(high_hz))
    return float(np.trapezoid(density[sel], freq[sel]))


def carrier_presence(samples, sample_rate_hz: float, centre_hz: float,
                     spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     tolerance_hz: Optional[float] = None,
                     hump_half_width_hz: Optional[float] = None,
                     reach_hz: Optional[float] = None,
                     exclude: Optional[Sequence[Tuple[float, float]]] = None,
                     system: str = "NTSC") -> Dict[str, float]:
    """Two statistics for a carrier at `centre_hz`, each against the same
    local floor.

    The floor is the MEDIAN density of the carrier's reach (+/- 180 kHz,
    `carrier_half_width_hz`) outside the hump band (+/- 70 kHz,
    `reference_half_width_hz`). A median is what separates the carrier
    from its neighbours: the colour-under's sidebands and harmonics and the
    luma's lower sideband are, on a static pattern, a picket fence of line
    harmonics 15.734 kHz apart - one bin in twenty - and on moving
    picture a smooth continuum; neither moves a median taken over 300 kHz,
    while a carrier does one of two things.

    NARROW LINE - an unmodulated carrier (the deck's own recording of a
    silent input) is one bin within the +/- 10 kHz tolerance of 5.8. The
    tallest bin in the tolerance is compared with the tallest a noise
    neighbourhood of that many bins would show (median + scatter x
    sqrt(2 ln N), the Gaussian extreme in decibels); `line_sigma` is the
    excess in units of the neighbourhood's scatter.

    HUMP - a modulated carrier at reference level fills +/- 50 kHz (5.9)
    and, at the audio band's top, +/- 70 kHz; it raises the MEAN density
    of the hump band over the floor. `hump_sigma` is that excess over its
    standard error, the neighbourhood's relative scatter divided by the
    square root of the hump's bin count.

    Neither statistic can call a carrier present on its own against
    structured neighbours; the calibration `detection_bound` plants
    carriers and asks at what level each statistic moves.

    `exclude` removes named bands - (centre, half width) pairs, from
    `confounds` - from BOTH the hump band and the surround, for neighbours
    that are deterministic rather than noise. The one that matters at 1.3
    MHz is the colour-under's second harmonic, 41.26 kHz below the carrier
    and so inside the hump band (`colour_under_harmonic_offset_hz`).

    Returns both statistics, the floor, the peak's frequency, and where
    that peak sits relative to the line-harmonic grid, which is what says
    whether a narrow line is a carrier or a picket."""
    tol = float(tolerance_hz or CARRIER_TOLERANCE_HZ)
    hump = float(hump_half_width_hz or reference_half_width_hz())
    reach = float(reach_hz or carrier_half_width_hz())
    if spectrum is None:
        spectrum = power_spectrum(samples, sample_rate_hz, 2.0 * tol / SPECTRUM_BINS_PER_TOLERANCE)
    freq, density = spectrum
    offset = np.abs(freq - float(centre_hz))
    keep = np.ones(len(freq), dtype=bool)
    for band_centre, band_half in (exclude or ()):
        keep &= np.abs(freq - float(band_centre)) > float(band_half)
    window = offset <= tol
    hump_band = (offset <= hump) & keep
    surround = (offset <= reach) & (offset > hump) & (density > 0) & keep
    floor = float(np.median(density[surround]))
    scatter_db = float(np.std(10.0 * np.log10(density[surround])))
    relative_scatter = float(np.std(density[surround]) / max(floor, 1e-300))
    peak_index = int(np.argmax(np.where(window, density, -np.inf)))
    peak_db = float(10.0 * np.log10(density[peak_index] / floor))
    n_window = int(window.sum())
    expected_peak_db = scatter_db * float(np.sqrt(2.0 * np.log(max(n_window, 2))))
    hump_mean = float(np.mean(density[hump_band]))
    excess = hump_mean / floor - 1.0
    hump_error = relative_scatter / float(np.sqrt(max(int(hump_band.sum()), 1)))
    return {
        "centre_hz": float(centre_hz), "bin_hz": float(freq[1] - freq[0]),
        "floor_density": floor, "scatter_db": scatter_db,
        "peak_db": peak_db, "peak_at_hz": float(freq[peak_index]),
        "expected_peak_db": expected_peak_db,
        "line_sigma": float((peak_db - expected_peak_db) / max(scatter_db, 1e-12)),
        "hump_excess": float(excess), "hump_error": float(hump_error),
        "hump_sigma": float(excess / max(hump_error, 1e-12)),
        "bins_in_tolerance": n_window, "bins_in_hump": int(hump_band.sum()),
        "bins_in_surround": int(surround.sum()),
        "bins_excluded": int((~keep).sum()),
        "peak_line_harmonic_offset_hz":
            line_harmonic_offset_hz(float(freq[peak_index]), system)["offset_hz"],
    }


def plant_fm_carrier(count: int, sample_rate_hz: float, centre_hz: float,
                     amplitude: float, deviation_hz: float = REFERENCE_DEVIATION_HZ,
                     modulation_hz: float = REFERENCE_MODULATION_HZ) -> np.ndarray:
    """An FM carrier with the reference modulation of 5.9 (a 400 Hz tone at
    +/- 50 kHz); `deviation_hz` of zero gives the unmodulated carrier.

    The modulation term is the INTEGRAL of the instantaneous frequency, so
    the peak phase excursion is the modulation index deviation/modulation
    and carries no further factor of 2 pi: differentiating
    (dev/mod) sin(2 pi mod t) and dividing by 2 pi returns dev cos(2 pi mod
    t), the deviation in hertz. Writing 2 pi (dev/mod) sin(...) instead - as
    this function did until 2026-09-05 - multiplies the deviation by 2 pi:
    measured 314.2 kHz of peak deviation for a requested 50 kHz, past the
    150 kHz maximum of 5.9, which spread the planted power over +/- 314 kHz
    so that only 14 percent of it (0.00042 of 0.00295 in the calibration
    record) fell inside the +/- 70 kHz hump band the detector integrates.
    The hump statistic then could not see a carrier planted 20 dB below the
    record."""
    t = np.arange(int(count), dtype=np.float64) / float(sample_rate_hz)
    phase = 2.0 * np.pi * float(centre_hz) * t
    if deviation_hz:
        phase = phase + (float(deviation_hz) / float(modulation_hz)) * np.sin(
            2.0 * np.pi * float(modulation_hz) * t)
    return float(amplitude) * np.cos(phase)


def detection_bound(samples, sample_rate_hz: float, centre_hz: float,
                    levels_db: Sequence[float], system: str = "NTSC",
                    reference_power: Optional[float] = None,
                    exclude: Optional[Sequence[Tuple[float, float]]] = None
                    ) -> Dict[str, object]:
    """Calibrate `carrier_presence` on this record by planting carriers.

    At each level (decibels relative to the record's own luma-band power,
    `luma_band_hz`, integrated from the same spectrum) an unmodulated
    carrier is planted for the narrow-line statistic and a reference-
    modulated one for the hump statistic. A level is DETECTABLE when the
    statistic rises `DETECTION_SIGMA` above what the unplanted record
    shows, so structured neighbours already present cannot be mistaken for
    sensitivity. Returns the unplanted statistics, the per-level table, and
    the weakest detectable level for each statistic (None if none of the
    levels tried was) - the bound on any carrier the record can hide."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    resolution = 2.0 * CARRIER_TOLERANCE_HZ / SPECTRUM_BINS_PER_TOLERANCE
    spectrum = power_spectrum(x, sample_rate_hz, resolution)
    if reference_power is None:
        reference_power = band_power(*spectrum, *luma_band_hz(system))
    unplanted = carrier_presence(x, sample_rate_hz, centre_hz, spectrum,
                                 exclude=exclude, system=system)
    table: List[Dict[str, float]] = []
    for level in levels_db:
        amplitude = float(np.sqrt(2.0 * reference_power * 10.0 ** (float(level) / 10.0)))
        line = carrier_presence(
            x + plant_fm_carrier(len(x), sample_rate_hz, centre_hz, amplitude, 0.0),
            sample_rate_hz, centre_hz, exclude=exclude, system=system)
        hump = carrier_presence(
            x + plant_fm_carrier(len(x), sample_rate_hz, centre_hz, amplitude),
            sample_rate_hz, centre_hz, exclude=exclude, system=system)
        table.append({
            "level_db": float(level),
            "line_sigma": line["line_sigma"], "peak_db": line["peak_db"],
            "peak_at_hz": line["peak_at_hz"],
            "line_detectable": bool(line["line_sigma"] - unplanted["line_sigma"] >= DETECTION_SIGMA
                                    and abs(line["peak_at_hz"] - centre_hz) <= line["bin_hz"]),
            "hump_sigma": hump["hump_sigma"], "hump_excess": hump["hump_excess"],
            "hump_detectable": bool(hump["hump_sigma"] - unplanted["hump_sigma"] >= DETECTION_SIGMA),
        })

    def weakest(key: str) -> Optional[float]:
        found = [row["level_db"] for row in table if row[key]]
        return min(found) if found else None

    return {"centre_hz": float(centre_hz), "reference_power": float(reference_power),
            "unplanted": unplanted, "levels": table,
            "confounds": confounds(centre_hz, system),
            "excluded": [(float(a), float(b)) for a, b in (exclude or ())],
            "line_bound_db": weakest("line_detectable"),
            "hump_bound_db": weakest("hump_detectable")}


# The measurement (2026-09-05, this module's `detection_bound` on the
# video-tap captures; the full table is in docs/HIFI_CAPTURE_PATH.md). Levels
# are decibels relative to the luma-band power of the same record. The
# zaroff captures are the SLV-778HF's own recordings of a Tektronix TSG-130A
# with NO audio connected, so any carrier the deck laid down would be
# unmodulated and the narrow-line bound is the operative one; the 2-hour
# tapes were recorded elsewhere with unknown audio, so the hump bound is.
MEASURED: Dict[str, object] = {
    "date": "2026-09-05",
    "verdict": "no AFM carrier at the video tap in any capture",
    "windows": {
        "zaroff SP/EP playback, SP record": "whole 0.48 s capture, 50 MSps",
        "countdown, home": "0.5 s from 16.68 s (the -s 500 decode window), 40 MSps",
    },
    "rates_note": "countdown/home are 40 MSps (the FLAC header reads 40 000 "
                  "Hz - the number is right and the UNIT is wrong by 1000); "
                  "the zaroff files 50 MSps (header 50 000 Hz, 24 000 512 "
                  "frames = 0.480 s). home.flac carries no seek table, so "
                  "its window is reached by reading and discarding.",
    "statistic_note": "line_sigma and hump_sigma below are the UNPLANTED "
                      "record's; the bounds are the weakest planted level "
                      "at which the statistic rose DETECTION_SIGMA above "
                      "them, in dB relative to that record's own luma-band "
                      "power (`luma_band_hz`).",
    # record -> channel -> (line_sigma, hump_sigma, peak offset from the
    # carrier in Hz, peak offset from the nearest line picket in Hz, the
    # floor over the hump band in dB re the luma band, line bound, hump bound)
    "results": {
        "zaroff SP playback": {
            1: (-0.53, 7.59, 812, -5132, -43.6, -45.0, -40.0),
            2: (0.12, 4.11, -5511, -4812, -42.7, -35.0, -35.0)},
        "zaroff EP playback": {
            1: (0.66, 8.59, -1477, -7421, -36.0, -40.0, -40.0),
            2: (1.02, 7.39, 2118, 2817, -36.3, -35.0, -35.0)},
        "zaroff SP record (control)": {
            1: (-0.55, 7.79, 812, -5132, -44.5, -30.0, -35.0),
            2: (0.41, 4.76, -5511, -4812, -43.1, -20.0, -30.0)},
        "countdown": {
            1: (-0.02, 4.33, 1880, -4064, -36.1, -50.0, -45.0),
            2: (-2.64, -7.00, -1392, -692, -33.6, -50.0, -45.0)},
        "home": {
            1: (-2.62, -2.90, 1880, -4064, -38.9, -50.0, -45.0),
            2: (-1.73, 1.08, -9937, 6497, -34.1, -45.0, -45.0)},
    },
    # THE CONTROL is what makes the null result readable. The unplanted
    # hump_sigma is 4 to 8.6 on the zaroff records, over DETECTION_SIGMA,
    # because the video spectrum has curvature across +/- 180 kHz and a
    # +/- 70 kHz band inside it sits above the surround's median whatever
    # is or is not recorded. The record-side capture (CN261 pin 1, the
    # VIDEO record current, which cannot carry AFM at all: the AFM record
    # signal goes to IC340 and the audio windings) shows the SAME excess -
    # 7.79 and 4.76 against the playback's 7.59 and 4.11, the playback
    # LOWER by 0.20 and 0.65 - and its strongest bin lands in the same two
    # spectrum bins, 1300811.8 and 1694488.5 Hz. So the hump excess is a
    # property of the video signal's own spectral shape, and there is no
    # playback-only component at either carrier.
    # ADDED 2026-09-07 with the correction in this module's docstring. The
    # rows above are left exactly as measured; these two keys say which of
    # them are evidence about leakage and which are not.
    "subjects_corrected": {
        "valid": ["zaroff SP playback", "zaroff EP playback"],
        "control": ["zaroff SP record (control)"],
        "withdrawn": ["countdown", "home"],
        "why_withdrawn": ("Ethan, 2026-09-06: 'home and cd will not' have "
                          "Hi-Fi audio, so those rows searched for a "
                          "signal that was never recorded"),
    },
    "bound_restated": ("the HUMP bound on the valid subjects: -40 dB on "
                       "channel 1 (chroma-bearing SP), -45 dB (y-only SP), "
                       "-35 dB on channel 2. The -50 dB figures above came "
                       "from the withdrawn records. See "
                       "capture_alignment.WIDE_SWEEP"),
    "control": "CN261 pin 1 (video record current) shows the same hump "
               "excess and the same peak bins as CN261 pin 2 (playback)",
    "narrow_line": "line_sigma runs -2.64 to +1.02 across all ten "
                   "(record, channel) pairs, against DETECTION_SIGMA 3: no "
                   "narrow line at either carrier in any capture",
    "envelope_comparison": ("`azimuth_crosstalk_db` puts the azimuth plus "
                            "spacing loss at 19 to 33 dB, but relative to "
                            "what a head AT THE AUDIO AZIMUTH reads. "
                            "Converting that to the luma-band reference "
                            "these bounds use needs the AFM RF level at the "
                            "audio head against the luma RF level at the "
                            "video head, and NO capture of the AFM tap "
                            "(CN341 pin 3 'FM PB') exists in this "
                            "repository, so the bridge is unmeasured. "
                            "Stated separately rather than combined."),
}


# --------------------------------------------------------------------------
# The readouts
# --------------------------------------------------------------------------

DECIMATION_GUARD = 2.0   # the decimated Nyquist is kept at twice the half-width
STOP_BAND_DB = 80.0      # the mixer's anti-alias filter, so the colour-under
                         # (up to 40 dB above the AFM floor) cannot fold in


def instantaneous_frequency(samples, sample_rate_hz: float, centre_hz: float,
                            half_width_hz: Optional[float] = None
                            ) -> Tuple[np.ndarray, float, float]:
    """The instantaneous frequency of the carrier nearest `centre_hz`.

    The record is mixed to complex baseband about the centre, low-passed to
    the half-width with a Kaiser FIR of `STOP_BAND_DB` attenuation, and
    decimated so that the new Nyquist sits at `DECIMATION_GUARD` times the
    half-width; the phase derivative of what remains is the frequency. The
    decimation is what makes a two-hour 40 MSps capture affordable: a 1.3
    MHz carrier with a +/- 180 kHz reach comes down by a factor of 55.

    The FIR's start-up and tail - half its length at each end, where the
    record is implicitly zero-padded - are discarded, so the series begins
    at `start_s`, not at zero.

    Returns the frequency series in hertz, its sample rate, and the time of
    its first sample."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    x = x - x.mean()
    rate = float(sample_rate_hz)
    width = float(half_width_hz or carrier_half_width_hz())
    n = len(x)
    z = x * np.exp(-2j * np.pi * float(centre_hz) * np.arange(n) / rate)
    factor = max(1, int(np.floor(rate / (2.0 * width * DECIMATION_GUARD))))
    taps, beta = signal.kaiserord(STOP_BAND_DB, width / (rate / 2.0))
    taps = int(taps) | 1
    h = signal.firwin(taps, width, fs=rate, window=("kaiser", beta))
    if factor > 1:
        z = signal.resample_poly(z, 1, factor, window=h)
        rate = rate / factor
    else:
        z = signal.fftconvolve(z, h, mode="same")
    guard = int(np.ceil((taps // 2) / factor))
    if len(z) > 2 * guard + 2:
        z = z[guard: len(z) - guard]
    else:
        guard = 0
    phase = np.unwrap(np.angle(z))
    frequency = float(centre_hz) + np.diff(phase) * rate / (2.0 * np.pi)
    # the difference sits between samples: the first value is at guard + 1/2
    return frequency, rate, (guard + 0.5) / rate


def block_average(values: np.ndarray, rate_hz: float, average_hz: float
                  ) -> Tuple[np.ndarray, float]:
    """Average consecutive blocks so the series runs at about `average_hz`:
    "averaging over a line" when that is the line rate."""
    block = max(1, int(round(float(rate_hz) / float(average_hz))))
    count = len(values) // block
    if count < 1:
        return np.asarray(values, dtype=np.float64), float(rate_hz)
    out = np.asarray(values[: count * block], dtype=np.float64).reshape(
        count, block).mean(axis=1)
    return out, float(rate_hz) / block


def lowpass_kernel(rate_hz: float, band_hz: float) -> np.ndarray:
    """A zero-phase FIR passing `band_hz` with the transition as wide as the
    band and `STOP_BAND_DB` of rejection; unity gain at zero frequency."""
    width = min(float(band_hz), float(rate_hz) / 2.0 - float(band_hz))
    width = max(width, float(rate_hz) / 1e4)
    taps, beta = signal.kaiserord(STOP_BAND_DB, width / (float(rate_hz) / 2.0))
    taps = int(taps) | 1
    h = signal.firwin(taps, float(band_hz), fs=float(rate_hz),
                      window=("kaiser", beta))
    return h / h.sum()


NOISE_SLOPE_LIMITS = (-1.0, 2.0)   # red (1/f) to blue (differenced phase noise)
NOISE_FIT_OCTAVES = (2.0, 8.0)     # the fit spans two to eight times the band:
                                   # above the band, where the signal is not,
                                   # yet near enough to extrapolate one octave


def noise_law(values: np.ndarray, rate_hz: float, band_hz: float
              ) -> Dict[str, float]:
    """The series' own noise density JUST ABOVE the band, as a power law.

    The density between two and eight times the band is fitted as
    S(f) = level x (f / band)^slope, the slope held between 1/f and f
    squared: white noise gives a flat law, and the blue noise a frequency
    discriminator or a period difference makes (white phase or crossing
    noise differenced, rising as f squared) gives the far smaller in-band
    level a white assumption would over-state sixty-fold. The level is
    re-anchored with the slope held, so clipping cannot bias it. Where the
    record is too short to resolve the band, the upper half of the
    spectrum's mean is returned as a flat law."""
    x = np.asarray(values, dtype=np.float64)
    rate = float(rate_hz)
    band = float(band_hz)
    nperseg = int(min(len(x), 2 ** int(np.ceil(np.log2(max(8.0 * rate / band, 16))))))
    f, s = signal.welch(x - x.mean(), fs=rate, nperseg=nperseg, window="hann",
                        detrend="constant")
    lo, hi = NOISE_FIT_OCTAVES
    sel = (f >= lo * band) & (f <= min(hi * band, rate / 2.0)) & (s > 0)
    if sel.sum() < 4:
        upper = (f >= rate / 4.0) & (s > 0)
        level = float(np.mean(s[upper])) if upper.any() else float(np.mean(s[1:]))
        return {"level": level, "slope": 0.0, "reference_hz": band}
    slope, _ = np.polyfit(np.log(f[sel] / band), np.log(s[sel]), 1)
    slope = float(np.clip(slope, *NOISE_SLOPE_LIMITS))
    intercept = float(np.mean(np.log(s[sel]) - slope * np.log(f[sel] / band)))
    return {"level": float(np.exp(intercept)), "slope": slope, "reference_hz": band}


def filtered_noise_error(law: Dict[str, float], kernel: np.ndarray,
                         rate_hz: float, duration_s: float) -> float:
    """The standard deviation of the noise a unity-gain kernel passes:
    the square root of the integral of S(f) |H(f)|^2 from the record's own
    resolution 1/T (anything slower is the mean, already removed) to the
    Nyquist frequency. For white noise this is the density times the
    kernel's equivalent noise bandwidth exactly; for a sloped law it is the
    exact filtered variance, which no single-frequency evaluation gives."""
    rate = float(rate_hz)
    points = max(8192, 16 * len(kernel))   # resolves the passband of a kernel this long
    w, response = signal.freqz(kernel, worN=points)
    f = w * rate / (2.0 * np.pi)
    f_low = 1.0 / max(float(duration_s), 1.0 / rate)
    keep = f >= f_low
    density = law["level"] * (f[keep] / law["reference_hz"]) ** law["slope"]
    variance = float(np.trapezoid(density * np.abs(response[keep]) ** 2, f[keep]))
    return float(np.sqrt(max(variance, 0.0)))


def band_limit(values: np.ndarray, rate_hz: float, band_hz: float
               ) -> Tuple[np.ndarray, float, np.ndarray]:
    """Low-pass a series to `band_hz` and state the standard error of what
    passes, from the series' own noise law (`noise_law`) integrated against
    the kernel (`filtered_noise_error`). Edges are reflected so the ends are
    not pulled to zero; within half a kernel of either end the result rests
    on that reflection and should not be judged.

    Returns the passed series, its standard error, and the kernel."""
    x = np.asarray(values, dtype=np.float64)
    h = lowpass_kernel(rate_hz, band_hz)
    pad = min(len(h) // 2, max(len(x) - 1, 0))
    padded = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]]) if pad else x
    passed = signal.fftconvolve(padded, h, mode="same")[pad: pad + len(x)]
    law = noise_law(x, rate_hz, band_hz)
    error = filtered_noise_error(law, h, rate_hz, len(x) / float(rate_hz))
    return passed, error, h


def time_base_error(carrier_samples, sample_rate_hz: float,
                    carrier_hz: Optional[float] = None, system: str = "NTSC",
                    channel: int = 1, half_width_hz: Optional[float] = None,
                    average_hz: Optional[float] = None,
                    band_hz: Optional[float] = None,
                    reference: str = "mean") -> Dict[str, object]:
    """The time-base error read from ONE AFM carrier.

    The carrier's instantaneous frequency is averaged over a line (the
    line rate by default, `colour_under.line_rate_hz`), which removes audio
    above about 8 kHz, and converted to the fractional period error
    epsilon = f_ref / f - 1 (positive = slow). With `band_hz` given, the
    series is low-passed to the flutter band and carries a standard error;
    audio inside that band is NOT removed - `audio_leakage_bound` says how
    much a tone leaks, and `time_base_error_pair` removes what the two
    carriers share.

    `reference` is what epsilon is measured against: "mean" uses the
    record's own mean carrier frequency, so a recorder whose oscillator sat
    off nominal (allowed 10 kHz, and on the SLV-778HF an RC oscillator) does
    not read as a constant speed error; "nominal" uses the specification's
    centre. Either way `electronic_offset` reports the mean's departure
    from nominal, which is the oscillator's and not the transport's.

    Returns a dict: `tbe` (per-line fractional period error), `time_s`,
    `rate_hz` (the per-line rate), `frequency_hz` (the per-line carrier
    frequency), `reference_hz`, `carrier_hz`, `electronic_offset`; and with
    `band_hz`, `tbe_band` and `standard_error`."""
    sysname = _system(system)
    centre = float(carrier_hz) if carrier_hz else CARRIER_HZ[sysname][int(channel) - 1]
    f, rate, start = instantaneous_frequency(carrier_samples, sample_rate_hz, centre,
                                             half_width_hz)
    if average_hz is None:
        average_hz = colour_under.line_rate_hz(_colour_system(system))
    f_line, line_rate = block_average(f, rate, float(average_hz))
    block = int(round(rate / line_rate))
    f_line = np.maximum(f_line, 1e-9)
    if reference == "mean":
        f_ref = float(f_line.mean())
    elif reference == "nominal":
        f_ref = centre
    else:
        f_ref = float(reference)
    tbe = f_ref / f_line - 1.0
    out: Dict[str, object] = {
        "tbe": tbe, "rate_hz": line_rate,
        "time_s": start + (np.arange(len(tbe)) * block + (block - 1) / 2.0) / rate,
        "frequency_hz": f_line, "reference_hz": f_ref, "carrier_hz": centre,
        "electronic_offset": float(f_line.mean() / centre - 1.0),
        "convention": "fractional period error, positive = tape slow",
    }
    if band_hz:
        passed, error, _ = band_limit(tbe, line_rate, float(band_hz))
        out["tbe_band"] = passed
        out["standard_error"] = error
        out["band_hz"] = float(band_hz)
    return out


def time_base_error_pair(samples, sample_rate_hz: float, system: str = "NTSC",
                         carriers: Optional[Sequence[float]] = None,
                         half_width_hz: Optional[float] = None,
                         average_hz: Optional[float] = None,
                         band_hz: Optional[float] = None) -> Dict[str, object]:
    """The time-base error read from BOTH carriers, with the audio they share
    removed.

    Each carrier's deviation from its own mean is d_i = -epsilon f_i + a_i,
    the speed term scaling with the carrier and the audio term a_i the same
    hertz on both for whatever is common to L and R (5.9, 5.10). So

        epsilon = -(d_2 - d_1) / (f_2 - f_1),    a = d_1 + epsilon f_1

    the second being the common-mode audio deviation, returned as
    `audio_hz` for inspection (per line, and band-limited as
    `audio_hz_band` when a band is given - the per-line series carries the
    two carriers' frequency noise scaled by 4.25 and 3.25). What is NOT
    removed is the stereo difference (R - L) inside the band, which enters
    as (a_2 - a_1) / (f_2 - f_1) - a 5 kHz difference costs 1.25 percent,
    so the estimator is honest only where the channels carry the same
    low-frequency content. Returns as `time_base_error`, plus `audio_hz`,
    `carriers_hz`, `measured_ratio`."""
    low, high = carriers if carriers else carriers_hz(system)
    one = time_base_error(samples, sample_rate_hz, low, system, 1,
                          half_width_hz, average_hz, None, "mean")
    two = time_base_error(samples, sample_rate_hz, high, system, 2,
                          half_width_hz, average_hz, None, "mean")
    count = min(len(one["tbe"]), len(two["tbe"]))
    f1 = one["frequency_hz"][:count]
    f2 = two["frequency_hz"][:count]
    r1 = one["reference_hz"]
    r2 = two["reference_hz"]
    d1 = f1 - r1
    d2 = f2 - r2
    eps = -(d2 - d1) / (r2 - r1)
    audio = d1 + eps * r1
    out: Dict[str, object] = {
        "tbe": eps, "rate_hz": one["rate_hz"], "time_s": one["time_s"][:count],
        "audio_hz": audio, "carriers_hz": (low, high),
        "reference_hz": (r1, r2), "measured_ratio": float(r2 / r1),
        "nominal_ratio": float(nominal_ratio(system)),
        "electronic_offset": (one["electronic_offset"], two["electronic_offset"]),
        "convention": one["convention"],
    }
    if band_hz:
        passed, error, _ = band_limit(eps, one["rate_hz"], float(band_hz))
        out["tbe_band"] = passed
        out["standard_error"] = error
        out["band_hz"] = float(band_hz)
        out["audio_hz_band"] = band_limit(audio, one["rate_hz"], float(band_hz))[0]
    return out


def audio_leakage_bound(audio_hz: float, deviation_hz: float,
                        carrier_hz: float, line_rate_hz: float,
                        band_hz: Optional[float] = None) -> float:
    """How much of a single audio tone survives into a single-carrier
    time-base error, as a fractional amplitude: the deviation over the
    carrier, times the line average's response |sinc(f / line rate)|, times
    the band low-pass's response at the tone where a band is given. A 400
    Hz tone at the reference deviation on a 1.3 MHz carrier leaks 3.8
    percent into an unfiltered readout and 3.8 percent x |H(400 Hz)| after
    the low-pass - which is why a band must be stated and why in-band bass
    needs the pair."""
    level = float(deviation_hz) / float(carrier_hz)
    average = abs(float(np.sinc(float(audio_hz) / float(line_rate_hz))))
    response = 1.0
    if band_hz:
        h = lowpass_kernel(line_rate_hz, float(band_hz))
        w, resp = signal.freqz(h, worN=[2.0 * np.pi * float(audio_hz) / float(line_rate_hz)])
        response = float(np.abs(resp[0]))
    return level * average * response


# --------------------------------------------------------------------------
# The video path's readout, for the cross-check: sync-edge crossings
# --------------------------------------------------------------------------

ANALYTIC_CHUNK = 1 << 21
# Samples discarded at every chunk boundary and at the record's ends. The
# brick-wall band mask rings as 1 / (pi B t); over 4096 samples at 40 MSps
# (102 us) with the 3 MHz luma band that has fallen to 1e-3, below the
# per-line crossing noise, and at 16 MSps further still.
ANALYTIC_OVERLAP = 1 << 12


def analytic_band(samples, sample_rate_hz: float, low_hz: float, high_hz: float
                  ) -> np.ndarray:
    """The analytic signal of the band `low_hz` to `high_hz`, by an FFT mask
    applied in overlapping chunks so memory stays bounded on long
    captures. The overlap is discarded on both sides of every chunk."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = len(x)
    chunk, overlap = ANALYTIC_CHUNK, ANALYTIC_OVERLAP
    if n <= chunk + 2 * overlap:
        chunk, overlap = n, 0
    out = np.empty(n, dtype=np.complex128)
    start = 0
    while start < n:
        lo = max(0, start - overlap)
        hi = min(n, start + chunk + overlap)
        piece = x[lo:hi]
        spectrum = np.fft.fft(piece)
        freq = np.fft.fftfreq(len(piece), 1.0 / float(sample_rate_hz))
        mask = (freq >= float(low_hz)) & (freq <= float(high_hz))
        spectrum = np.where(mask, spectrum * 2.0, 0.0)
        analytic = np.fft.ifft(spectrum)
        keep_lo = start - lo
        keep_hi = keep_lo + min(chunk, n - start)
        out[start: start + (keep_hi - keep_lo)] = analytic[keep_lo:keep_hi]
        start += chunk
    return out


def luma_instantaneous_frequency(rf, sample_rate_hz: float, system: str = "NTSC",
                                 band_hz: Optional[Tuple[float, float]] = None
                                 ) -> np.ndarray:
    """The luma FM carrier's instantaneous frequency, from the analytic
    signal of the carrier band (`luma_band_hz`: sync tip minus the
    deviation to peak white plus the deviation), which holds the
    first-order sidebands the sync edges need."""
    lo, hi = band_hz if band_hz else luma_band_hz(system)
    z = analytic_band(rf, sample_rate_hz, lo, hi)
    return np.diff(np.unwrap(np.angle(z))) * float(sample_rate_hz) / (2.0 * np.pi)


def sync_crossing_time_base(rf, sample_rate_hz: float, system: str = "NTSC",
                            carriers: Optional[Tuple[float, float]] = None
                            ) -> Dict[str, object]:
    """The time-base error read from the VIDEO path: the periods between
    successive line-sync falling edges.

    ONE VALUE PER LINE, at 15.734 kHz - not per field. The distinction
    decides what can be seen: at the field rate the Nyquist frequency is
    29.970 Hz, which is exactly the drum's rotation rate (one revolution
    per two fields, `transport_model.FIELDS_PER_DRUM_REVOLUTION`), so a
    per-field series aliases the drum onto the head alternation and cannot
    separate the two. Per line the drum is sampled 262 times a revolution
    and the capstan near 3260, so both are ordinary low-frequency
    components of this series.

    The luma carrier frequency is smoothed over one pulse edge
    (`sync_geometry.PULSE_EDGE_NS`) and thresholded midway between the sync
    tip and blanking carriers (3.9.1.1.4); the 50 percent falling crossings
    are interpolated to sub-sample precision. Crossings inside the band
    mask's edge transient at either end of the record (`ANALYTIC_OVERLAP`)
    are discarded. A pulse is a LINE sync only if its width lies nearer the
    line sync (4.70 us) than the equalizing pulse (2.30 us) on the short
    side and as far on the long side, which rejects the equalizing and
    broad pulses of the vertical interval; a period is kept only if it is
    within half a line of nominal, which rejects the gap across the
    interval. The period's fractional error is epsilon, positive = slow, on
    the same convention as the carrier readouts.

    Returns `tbe` per accepted line, `time_s` at each period's midpoint,
    `line_rate_hz`, the per-line `noise` (the white part of the
    line-to-line scatter, from the first difference), and the counts."""
    sysname = _system(system)
    rate = float(sample_rate_hz)
    tip, white = carriers if carriers else rf_stages.VHS_CARRIER_HZ[sysname]
    blank = tip + (BLANKING_IRE - SYNC_IRE) / (WHITE_IRE - SYNC_IRE) * (white - tip)
    threshold = 0.5 * (tip + blank)
    dev = rf_stages.deviation_hz((tip, white))
    f = luma_instantaneous_frequency(rf, rate, system, (tip - dev, white + dev))
    edge = max(1, int(round(sync_geometry.PULSE_EDGE_NS[sysname] * 1e-9 * rate)))
    if edge > 1:
        f = np.convolve(f, np.ones(edge) / edge, mode="same")
    below = f < threshold
    fall = np.nonzero(~below[:-1] & below[1:])[0]
    rise = np.nonzero(below[:-1] & ~below[1:])[0]

    def crossing(i):
        a, b = f[i], f[i + 1]
        frac = 0.0 if a == b else (a - threshold) / (a - b)
        return (i + frac + 0.5) / rate   # +0.5: the difference sits between samples

    fall_t = np.array([crossing(i) for i in fall])
    rise_t = np.array([crossing(i) for i in rise])
    duration = len(f) / rate
    edge_s = ANALYTIC_OVERLAP / rate if duration > 3 * ANALYTIC_OVERLAP / rate else 0.0
    inside = lambda t: (t >= edge_s) & (t <= duration - edge_s)
    fall_t = fall_t[inside(fall_t)] if len(fall_t) else fall_t
    rise_t = rise_t[inside(rise_t)] if len(rise_t) else rise_t
    line_sync = sync_geometry.LINE_SYNC_US[sysname] * 1e-6
    equalizing = sync_geometry.EQUALIZING_PULSE_US[sysname] * 1e-6
    half = 0.5 * (line_sync - equalizing)
    accepted = []
    j = 0
    for t in fall_t:
        while j < len(rise_t) and rise_t[j] <= t:
            j += 1
        if j >= len(rise_t):
            break
        width = rise_t[j] - t
        if abs(width - line_sync) <= half:
            accepted.append(t)
    accepted = np.asarray(accepted)
    period = 1.0 / colour_under.line_rate_hz(_colour_system(system))
    if len(accepted) < 3:
        return {"tbe": np.array([]), "time_s": np.array([]), "accepted": int(len(accepted)),
                "rejected": int(len(fall_t) - len(accepted)), "line_rate_hz": 1.0 / period}
    periods = np.diff(accepted)
    ok = np.abs(periods / period - 1.0) <= 0.5
    tbe = periods[ok] / period - 1.0
    time = (accepted[:-1] + accepted[1:])[ok] / 2.0
    noise = float(np.std(np.diff(tbe)) / np.sqrt(2.0)) if len(tbe) > 2 else float("nan")
    return {"tbe": tbe, "time_s": time, "line_rate_hz": 1.0 / period,
            "noise": noise, "accepted": int(ok.sum()),
            "rejected": int(len(fall_t) - ok.sum()),
            "threshold_hz": threshold,
            "convention": "fractional period error, positive = tape slow"}


def on_uniform_grid(time_s: np.ndarray, values: np.ndarray, rate_hz: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an irregular per-line series (lines the vertical interval
    took are missing) onto a uniform grid by linear interpolation, so it can
    be band-limited and compared sample by sample."""
    t = np.asarray(time_s, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if len(t) < 2:
        return t, v
    grid = np.arange(t[0], t[-1], 1.0 / float(rate_hz))
    return grid, np.interp(grid, t, v)


# --------------------------------------------------------------------------
# The cross-check
# --------------------------------------------------------------------------

# What `transport_model.find_lines` needs before it will look for a rate at
# all, restated here so this module can say WHY a rate was not searched
# instead of handing back an empty list - the distinction
# `transport_model.observable` exists to draw, carried one step further. The
# tolerance is read from that function's own signature so the two cannot
# drift apart; the two bin counts are the minimums its body enforces
# (`if near.sum() < 1: continue` and `if around.sum() < 4: continue`), and
# `find_lines` also refuses a series shorter than 16 samples.
LINE_SEARCH_TOLERANCE = float(inspect.signature(
    transport_model.find_lines).parameters["tolerance"].default)
LINE_SEARCH_MIN_NEAR_BINS = 1
LINE_SEARCH_MIN_BACKGROUND_BINS = 4
LINE_SEARCH_MIN_SAMPLES = 16


def line_search_duration_s(rate_hz: float, tolerance: Optional[float] = None
                           ) -> Dict[str, float]:
    """The shortest record in which `transport_model.find_lines` can look
    for a line at `rate_hz`, which is several times longer than one cycle.

    The spectrum's resolution is the record's own, 1/T. The candidate band
    is +/- tolerance x rate and so holds 2 tau r T bins, of which at least
    one is required; the local background is the annulus from tolerance to
    four times it, 6 tau r T bins, of which at least four are required
    because the median of fewer is not a background. The background is the
    binding requirement, T >= 4 / (6 tau r).

    Measured on the drum at 29.97 Hz with the default tolerance 0.08: the
    search needs 0.278 s, where `transport_model.observable` - which asks
    only for one cycle in the record - is satisfied by 0.033 s. In a 0.249 s
    record the drum's candidate band holds 2 bins and its background 2, so
    the search returns nothing whatever the transport did. That is a
    property of the record, not of the transport, and this function is what
    lets `cross_check` say so."""
    tau = float(tolerance if tolerance is not None else LINE_SEARCH_TOLERANCE)
    rate = float(rate_hz)
    near = LINE_SEARCH_MIN_NEAR_BINS / (2.0 * tau * rate)
    background = LINE_SEARCH_MIN_BACKGROUND_BINS / (6.0 * tau * rate)
    return {"rate_hz": rate, "tolerance": tau,
            "candidate_band_s": float(near), "background_s": float(background),
            "duration_s": float(max(near, background))}


def searchable_rates(rates: Sequence[Dict[str, object]], duration_s: float,
                     samples: int, tolerance: Optional[float] = None
                     ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Split predicted rates into those a record of this length and sample
    count can actually be searched for and those it cannot, each of the
    latter carrying the reason and the record length it would need.

    A rate already declared unresolvable by `transport_model.observable`
    keeps that verdict; the rest are put to `line_search_duration_s`."""
    searchable: List[Dict[str, object]] = []
    refused: List[Dict[str, object]] = []
    for entry in rates:
        rate = float(entry.get("rate_hz", np.nan))
        item = {"name": entry.get("name"), "rate_hz": rate}
        if int(samples) < LINE_SEARCH_MIN_SAMPLES:
            refused.append(dict(item, why="fewer samples than the search accepts",
                                needs_s=float(LINE_SEARCH_MIN_SAMPLES) *
                                float(duration_s) / max(int(samples), 1)))
            continue
        if not np.isfinite(rate) or rate <= 0.0:
            refused.append(dict(item, why="no rate predicted", needs_s=float("nan")))
            continue
        # the binding requirement is whichever is longer: a cycle in the
        # record (transport_model.observable) or the search's background
        need = line_search_duration_s(rate, tolerance)
        needs_s = max(need["duration_s"], 1.0 / rate)
        if not entry.get("resolvable", True):
            refused.append(dict(item, why=entry.get("why_not", "not resolvable"),
                                needs_s=float(needs_s), have_s=float(duration_s)))
            continue
        if need["duration_s"] > float(duration_s):
            refused.append(dict(
                item, why=("the record is shorter than the local background "
                           "the search needs"),
                needs_s=need["duration_s"], have_s=float(duration_s)))
        else:
            searchable.append(entry)
    return searchable, refused


def cross_check(video: Dict[str, object], hifi: Dict[str, object],
                band_hz: float, rate_hz: Optional[float] = None,
                field_rate_hz: float = 59.94,
                linear_speed_m_s: Optional[float] = None) -> Dict[str, object]:
    """Compare the video-derived and Hi-Fi-derived time-base errors.

    Both are put on one uniform grid (the video line rate unless `rate_hz`
    is given), band-limited to `band_hz` with the standard error each
    carries from its own noise law, trimmed by half a kernel at each end
    (where the low-pass rests on reflected data), and compared as
    hifi = intercept + slope x video. Agreement means slope 1, intercept 0
    and a chi-square per degree of freedom near 1 against the combined
    errors. A slope away from 1 is a SCALE error - a wrong nominal
    frequency or line rate in one readout; an intercept is an offset one
    of them carries; a chi-square far above 1 with slope 1 is noise one
    of them under-states or a component only one of them sees. Which
    readout is wrong the pair cannot say by itself, so the difference is
    also searched at the transport's predicted rates
    (`transport_model.rotation_rates`): a mechanical line that appears in
    the DIFFERENCE is one readout's systematic error, since both must see
    the same transport.

    THE GRID IS PER LINE, not per field, and that is what makes the drum
    searchable at all. `sync_crossing_time_base` returns one value per
    accepted line-sync period, so the series runs at 15.734 kHz and the
    drum's 29.97 Hz is sampled 262 times a revolution. A series sampled
    once per FIELD would put its Nyquist frequency at 29.970 Hz, exactly
    the drum rate, and the drum would alias onto the head alternation and
    could not be told from it; nothing in this module produces such a
    series, and `rate_hz` should not be set to the field rate.

    Only rates the record is long enough to search are put to
    `find_lines`; the rest are returned in `lines_not_searched` with the
    reason and the record length each would need (`searchable_rates`), so
    an empty `lines_video` means the search looked and found nothing rather
    than that it never looked.

    Returns the fitted slope, intercept and their errors, the correlation,
    the rms difference, chi-square per degree of freedom, a verdict, the
    lines found in each series and in the difference, and what was not
    searched."""
    rate = float(rate_hz or video.get("line_rate_hz") or video.get("rate_hz"))
    tv, vv = on_uniform_grid(video["time_s"], video["tbe"], rate)
    th, hv = on_uniform_grid(hifi["time_s"], hifi["tbe"], rate)
    if len(tv) < 2 or len(th) < 2:
        return {"usable": False, "why": "a readout is empty"}
    lo, hi = max(tv[0], th[0]), min(tv[-1], th[-1])
    grid = np.arange(lo, hi, 1.0 / rate)
    if len(grid) < 8:
        return {"usable": False, "why": "no overlap between the two readouts"}
    v = np.interp(grid, tv, vv)
    h = np.interp(grid, th, hv)
    v_band, v_err, kernel = band_limit(v, rate, band_hz)
    h_band, h_err, _ = band_limit(h, rate, band_hz)
    trim = len(kernel) // 2
    if len(grid) - 2 * trim < 8:
        return {"usable": False, "why": "the overlap is shorter than the band's kernel",
                "overlap_s": float(grid[-1] - grid[0]), "kernel_s": len(kernel) / rate}
    grid, v_band, h_band = grid[trim:-trim], v_band[trim:-trim], h_band[trim:-trim]
    v_band = v_band - v_band.mean()
    h_band = h_band - h_band.mean()
    # independent samples in the band: the record's length times the band
    n_eff = max(2.0, 2.0 * band_hz * (grid[-1] - grid[0]))
    design = np.column_stack([np.ones_like(v_band), v_band])
    coef, *_ = np.linalg.lstsq(design, h_band, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    resid = h_band - design @ coef
    combined = float(np.sqrt(v_err ** 2 + h_err ** 2))
    chi2 = float(np.mean(resid ** 2) / max(combined ** 2, 1e-30))
    var_v = float(np.var(v_band))
    slope_err = float(np.sqrt(np.mean(resid ** 2) / max(var_v, 1e-30) / n_eff))
    intercept_err = float(np.sqrt(np.mean(resid ** 2) / n_eff))
    r = float(np.corrcoef(v_band, h_band)[0, 1]) if var_v > 0 and np.var(h_band) > 0 else 0.0
    dof = max(n_eff - 2.0, 1.0)
    consistent = (abs(slope - 1.0) <= 3.0 * slope_err
                  and abs(intercept) <= 3.0 * intercept_err
                  and chi2 <= 1.0 + 3.0 * np.sqrt(2.0 / dof))
    out: Dict[str, object] = {
        "usable": True, "samples": int(len(grid)), "independent": float(n_eff),
        "slope": slope, "slope_error": slope_err,
        "intercept": intercept, "intercept_error": intercept_err,
        "correlation": r, "rms_difference": float(np.sqrt(np.mean((h_band - v_band) ** 2))),
        "video_error": v_err, "hifi_error": h_err, "chi2_per_dof": chi2,
        "consistent": bool(consistent),
        "verdict": ("consistent" if consistent else
                    "scale error" if abs(slope - 1.0) > 3.0 * slope_err else
                    "offset" if abs(intercept) > 3.0 * intercept_err else
                    "excess disagreement"),
        "band_hz": float(band_hz), "rate_hz": rate,
        "span_s": float(grid[-1] - grid[0]),
    }
    speed = float(linear_speed_m_s) if linear_speed_m_s else (
        head_model.FORMAT_MECHANICS[("VHS", "NTSC", "SP")]["linear_tape_speed_m_s"])
    # the length the SEARCH sees, on find_lines' own terms: it takes the
    # series and the rate, so its resolution is len(series) / rate
    search_s = len(v_band) / rate
    rates = transport_model.observable(
        transport_model.rotation_rates(speed, field_rate_hz), search_s, rate)
    rates, refused = searchable_rates(rates, search_s, len(v_band))
    out["line_search_s"] = float(search_s)
    out["lines_searched"] = [entry["name"] for entry in rates]
    out["lines_not_searched"] = refused
    out["lines_video"] = transport_model.find_lines(v_band, rate, rates)
    out["lines_hifi"] = transport_model.find_lines(h_band, rate, rates)
    out["lines_difference"] = transport_model.find_lines(h_band - v_band, rate, rates)
    return out
