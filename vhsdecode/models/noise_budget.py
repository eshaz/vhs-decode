"""The noise budget: three predicted floors, summed, against the measured one.

Ethan: *"Noise budget - arithmetic, one evening. Compute kTB (thermal),
quantization from effective bits, and particulate from 10 log10(N).
Particulate estimate: track width ~58 um SP, read depth ~ lambda / 2 pi ~
0.23 um at luma, along-track ~1.45 um -> ~19 um^3 -> ~7000 particles ->
~38 dB. Measured floor is 45 dB. Sum the three, compare to measured. Budget
closes -> nothing unexplained. Gap -> the size of the gap points at what's
missing. Write down expected values before measuring."*

Everything here is stated as a CARRIER-TO-NOISE DENSITY, C/N0 in dB-Hz:
the carrier's power over the noise's one-sided power density per hertz.
That is the one currency the three terms share - each mechanism has its own
natural bandwidth (kTB is quoted over a band, a particle count over one
wavelength, a quantiser over Nyquist), and adding them as powers is only
legitimate once they are all per hertz. Every other figure this arc has
quoted - "C/N in Carson's band", "per bin", "the tape's 21.6 dB", a video
signal-to-noise - is C/N0 with a bandwidth label attached, and
`definitions` writes the same floor under every label so that two numbers
can be recognised as the same measurement or as different ones.

THE THREE TERMS, EACH DERIVED

  1. THERMAL. A source of resistance R at temperature T makes available
     k T per hertz (Nyquist-Johnson), and an amplifier of noise factor F
     raises it to k T F. The head's signal makes available V^2 / (4 R), so

         C/N0 = V^2 / (4 R k T F).

     k is exact (SI 2019). T is an assumption - no deck temperature has
     been measured - at 293 K. V, R and F are assumptions too, labelled in
     `ASSUMPTIONS` with the schematic evidence there is, read from the Sony
     SLV-777HF/778HF/788HF service manual in /testdata/test_patterns/vhs
     (11 pages; PDF page 1 = manual pages 4-5 to 4-7, "MA-327 (REC/PB AMP)
     SCHEMATIC DIAGRAM", sheet MA-327 (1/8); PDF page 2 = manual pages 4-8
     to 4-10, "MA-327 (VIDEO, NORMAL AUDIO)", sheet MA-327 (2/8)). The head
     amplifier is IC260, marked HA118195NT, "VIDEO REC/PB AMP" (page 1).
     The four video heads (EP CH1/CH2 and SP CH1/CH2, each F and S) arrive
     through the 13-pin connector CN260 with 620 ohm damping resistors
     R260 and R261 across the head lines (page 1). The check connector
     CN261 is marked FOR CHECK: pin 1 REC CURR, pin 2 PB RF, pin 3 RF SWP,
     pin 4 GND (page 1), and the readme beside the captures states the
     playback captures were taken at CN261 pin 2 and the record captures at
     pin 1, unbuffered, into a Rigol DS1202Z-E oscilloscope. Waveform 7 on
     page 2, taken at "IC201 pin 15 PB / CN261 pin 2 PB" at 200 mV per
     division and 0.1 us per division, shows the playback RF spanning about
     3.2 graticule divisions, so the RF at the capture point is roughly
     0.6 V peak to peak - read off the graticule at eight-fold
     magnification of the page, not printed. (The pin numbers in this
     manual's waveform captions are set in a symbol font whose glyphs are
     not their characters; they were decoded from the page's own text
     layer, where waveform 7 reads IC201 pin 15 and waveform 13, beside
     it, IC201 pin 38. An earlier draft of this module read pin 19 off the
     rendered page and was wrong.) That is an AMPLIFIED level: the manual states
     neither IC260's gain nor its automatic gain control, no head impedance,
     no input noise figure, and no data sheet for the HA118195NT is in the
     tree, so nothing in it fixes the head's own output. The thermal term
     is therefore the least certain of the three, and its range is reported
     alongside its central value rather than hidden inside it.

  2. QUANTISATION. A uniform quantiser of step d adds d^2 / 12 spread flat
     over the Nyquist band (`capture_profile.quantization_floor`). The
     word length is READ from the code lattice of the capture
     (`capture_profile.measure_bit_depth`), never assumed, and the
     effective number of bits comes from the lowest floor the capture's
     spectrum shows anywhere - where the analogue chain contributes least,
     which on a filtered capture is above its anti-alias roll-off and on
     an oscilloscope capture (no anti-alias filter at 50 MS/s on a 200 MHz
     front end, per the readme) is wherever the front end's own noise is
     thinnest. Each lost effective bit doubles the noise voltage, so the
     density is scaled by 4^(bits - effective bits). The record-side
     capture of the same scope, which carries the modulator and the record
     amplifier but no tape, is measured as a CONTROL on the recording
     chain rather than folded into the term.

  3. PARTICULATE. A head reads a volume of coating holding N particles of
     random moment; the signal adds coherently and the noise as a square
     root, so the carrier-to-noise is N. Ethan's count: track width 58 um
     (JVC VTG82063 section 1 table 1-1-1 item 9, and the section 3 format
     table, "Video Track Width SP 0.058 mm"), read depth lambda / 2 pi (the
     self-demagnetisation bound, docs/MATHEMATICS.md proposition 8a.1) =
     0.23 um at the luma carrier, along-track length one wavelength,
     1.45 um at 4.0 MHz (lambda = v / f with v = 5.80 m/s, VTG82063 table
     1-1-1 item 4 "Writing speed 5.80 m/sec"; section 7.2.1 states the
     same law's results, "approximately 1.3 um" at 4.4 MHz and
     "approximately 1.7 um" at 3.4 MHz, against 1.32 and 1.71 um here; and
     the track width is table 1-1-1 item 9, 0.058 mm) - about 19 um^3,
     about 7000 particles,
     about 38 dB. Reproduced here with the repository's own particle
     constants (`tape_model.PARTICLE_VOLUME_M3` 1e-21 m^3 and
     `PACKING_FRACTION` 0.4, both labelled there as typical rather than
     measured; no specification in the collection gives the stock's
     particle volume or packing, and the stock of these tapes is not
     known): 19.41 um^3, 7763 particles, 38.9 dB. WHERE THIS DIFFERS FROM
     ETHAN'S FIGURE: only in the particle density. His ~7000 in ~19 um^3
     is about 370 particles per um^3; the repository's constants give 400
     per um^3; the volume is the same to three figures. The 0.45 dB between
     38.45 and 38.9 dB is that assumption and nothing else.

     WHAT BANDWIDTH 10 log10(N) BELONGS TO, derived rather than chosen.
     Particles at n per unit length along the track, each carrying a moment
     that follows the recorded sinusoid, are a marked Poisson process. Its
     mean is the signal, n m sin(k x), of power n^2 m^2 / 2; its shot noise
     is white with two-sided density n E[mark^2] = n m^2 / 2 per unit
     wavenumber, which the head sweeps past at v to make a one-sided
     density of n m^2 / v per hertz. Hence

         C/N0 = n v / 2 = N_lambda f_c / 2,

     where N_lambda is the count per wavelength and f_c = v / lambda. So
     10 log10(N_lambda) is the carrier-to-noise in a band of f_c / 2 - two
     megahertz at the luma carrier - and the same density read in Carson's
     eight megahertz is six decibels lower. Counting particles over one
     SAMPLE SPACING instead (v / f_s = 0.12 to 0.15 um) gives the
     carrier-to-noise over the Nyquist band, by the same law: the choice of
     along-track length sets only the label on the bandwidth; the density
     is invariant, which is why the density is what is summed. The unit
     tests check the law on a simulated track. A saturation recording with
     square-wave marks - which is what an unbiased FM recording is - has a
     fundamental of 4 / pi times the mark and a mark-square of m^2, giving
     4 / pi^2 in place of 1 / 2: 0.9 dB less, available as `saturation`.

THE PREDICTION, WRITTEN BEFORE THE MEASUREMENT (2026-09-05, from the
arithmetic alone, before any capture was read for this module; the carrier
amplitude of 35.2 codes is the figure the residual-limit record carries for
the 8-bit 50 MSps bars capture)

  thermal        106.9 dB-Hz   central: 0.25 mV rms head output, 20 ohm,
                               noise figure 6 dB, 293 K; the labelled range
                               (0.125 to 0.5 mV) spans 100.8 to 112.9
  quantisation   112.7 dB-Hz   8 bits at 50 MSps against a carrier of
                               35.2 codes amplitude, 8.0 effective bits
  particulate    101.9 dB-Hz   7763 particles at 4.0 MHz, 38.9 dB in
                               f_c / 2 = 2.0 MHz
  sum            100.4 dB-Hz   i.e. 31.4 dB in Carson's 8 MHz band, and a
                               demodulated luma signal-to-noise of about
                               41 dB after the format's de-emphasis and
                               about 51 dB with the unified weighting
                               network (this weighted figure was revised
                               from 46 dB before any capture was read, when
                               the network was corrected from a single pole
                               to the cited constant-resistance equaliser)

  Expected reading of "45 dB": the converter's own floor in Carson's band
  is 43.7 dB (8.0 bits) to 46.7 dB (the docs' figure, which counts only the
  amplitude quadrature), so if 45 dB was measured it is the CAPTURE CARD's
  floor and not the tape's; the tape's own carrier-to-noise in that band
  was expected near the sum above, in the low thirties.

THE MEASUREMENT is made by `measure_floor`, described there, and the
numbers it returned on the real captures are recorded under MEASURED at the
end of this docstring, after the prediction and separately from it.

MEASURED (2026-09-06, `tools/ringing_measure/noise_budget.py`, 0.25 s
windows; the in-band floor is the median over field-sync broad pulses of
the per-pulse median density between the tip tone's main lobe and the
peak-white carrier plus the same lobe, against the carrier power on the
same pulses)

  capture   rate     ENOB  carrier  MEASURED  sum    gap    weighted video
  bars_sp   49.99 M  7.50  38.0     93.8      100.3  +6.44  44.3 dB
  cd        40.00 M  7.90  75.7     95.6      100.6  +4.98  46.1 dB
  home      39.99 M  7.85  72.5     93.2      100.6  +7.40  43.6 dB

  (C/N0 in dB-Hz. bars_sp is the zaroff 75 % bars SP playback capture, two
  windows in its 0.48 s; cd and home are the two-hour captures, four
  windows each at 60, 1800, 3600 and 5400 seconds. The rate is MEASURED
  from the line period and settles the standing trap: both two-hour
  captures run at 40 MSps, not 50, and the tip lands inside the format's
  3.4 +/- 0.1 MHz at every window. The sum moves between captures only
  because the quantisation term follows the capture's own carrier and
  effective bits.)

  THE BUDGET DOES NOT CLOSE. The gap is +5.0 to +7.4 dB, which is 68 to
  82 per cent of the measured noise power standing outside the three
  terms. Four measurements say what it is not:

  1. NOT THE CONVERTER OR THE RECORDING CHAIN. The record-side capture of
     the same signal through the same oscilloscope - the modulator and the
     record amplifier, no tape and no head amplifier - reads 109.2 dB-Hz,
     13.6 to 16.0 dB better than any playback capture. Its spectrum is at
     the arithmetic quantisation floor (8.01 effective bits of 8, -0.1 dB
     over arithmetic).

  2. NOT THE THERMAL TERM, and the one bound that exists points the wrong
     way. The lowest density anywhere in the playback spectrum, less the
     arithmetic quantisation floor, bounds the electronics from BELOW at
     C/N0 >= 113.4 (bars), 126.9 (cd) and 124.4 (home) dB-Hz. The thermal
     term ASSUMES 106.9. So the electronics are quieter than assumed, not
     noisier, and closing the gap by moving the thermal assumptions makes
     the prediction worse rather than better.

  3. NOT ADDITIVE AT ALL: THE FLOOR RISES TOWARD THE CARRIER. Read in
     rings of offset from the tip tone, the floor is (bars / cd / home, in
     dB-Hz) 96.9 / 95.9 / 98.0 at 0.7-1.3 MHz out, 95.3 / 95.5 / 95.3 at
     0.4-0.7 MHz, and 89.0 / 91.2 / 88.4 at 0.2-0.4 MHz - 4.7, 7.9 and
     9.6 dB worse within a few hundred kilohertz of the tone. Every one
     of the three terms is white by construction: thermal is kT, quantisation is
     d^2 / 12 over Nyquist, and the particle count is the shot noise of a
     Poisson process. None of them can rise toward a tone. Noise that
     rides the carrier is multiplicative, and the same reading on the
     record side is flat to within 1.7 dB, so it is made in the tape and
     head path and not in the electronics.

  4. THE PARTICULATE TERM'S OWN SCALING LAW FAILS THE ONE TEST AVAILABLE.
     NTSC VHS EP keeps the drum speed and so the writing speed, the
     wavelength, the read depth and the along-track length, and keeps the
     carriers exactly (the decoder returns the same ire0, hz_ire, line
     period and vsync width); it changes the recorded track width from
     58 to 19.3 um. The count is proportional to that width and to nothing
     else that moves, so the particulate term predicts the EP floor worse
     by 10 log10(58 / 19.3) = 4.78 dB. On the same deck, head amplifier,
     oscilloscope, test signal and capture session, the SP and EP playbacks
     of the zaroff 75 % bars measure 93.81 and 80.30 dB-Hz: a difference of
     13.52 dB where 4.78 was predicted. The record-side pair of the same
     two recordings differs by 2.85 dB, which is the recording chain's own
     EP penalty and mostly its smaller carrier (26.8 against 35.8 codes);
     net of that control the playback difference is 10.67 dB, still 5.9 dB
     more than the count allows. And the excess is again concentrated at
     the carrier: the EP skirt is 15.4 dB from the 0.7-1.3 MHz ring to the
     0.2-0.4 MHz ring where SP's is 7.9 dB.

  WHAT THE GAP POINTS AT, therefore, is not a mis-set constant in any of
  the three terms but a MECHANISM none of them contains: multiplicative
  noise made where the head meets the coating - contact and spacing
  modulation, surface roughness, and at EP the adjacent-track leakage that
  no guard band separates - which appears as a skirt on the carrier rather
  than as a floor under it. The measurement that would settle its share
  is a tape-stopped capture at the same tap, which reads the head
  amplifier alone with no coating moving past it; nothing in the tree is
  such a capture, so the electronics' contribution is still only bounded
  and not measured.

  "45 dB": WHICH DEFINITION. The same measured floor, written under every
  label (`definitions`), reads 24.2 to 26.6 dB in Carson's 8 MHz band,
  20.2 to 22.6 dB over the Nyquist band, and 43.6 to 46.1 dB as a
  demodulated luma video signal-to-noise with the format's de-emphasis and
  the unified weighting network - peak-to-peak signal over rms noise, the
  way a deck's video signal-to-noise is conventionally quoted. Only the
  last of those brackets 45 dB, and it does so on all three captures at
  once with nothing free to adjust, so Ethan's 45 dB is a DEMODULATED,
  DE-EMPHASISED, WEIGHTED VIDEO figure. It is not the full-band
  carrier-to-noise: `interference_distributions.carrier_fraction`, which
  counts everything that is not the carrier as noise across the whole
  Nyquist band, reads a carrier share of 0.878 to 0.973 and 8.6 to 15.6 dB
  on these same windows - a different quantity in a different bandwidth,
  and not comparable to 45. A per-bin reading also passes through 45 dB,
  but only at a chosen bin width (66 kHz on home, 76 on bars, 116 on cd),
  so it matches only with a free parameter and is the weaker reading.
  Against the video figure the budget's own prediction was 41 dB
  de-emphasised and 51 dB weighted, so the prediction was 5 to 7 dB
  optimistic there too - the same gap, in the units of the quoted number.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import capture_profile, head_model, tape_model
from vhsdecode.models.interference_distributions import carrier_fraction

# --------------------------------------------------------------------------
# constants, each with its provenance
# --------------------------------------------------------------------------

# Exact by the 2019 SI redefinition; the same value tools/ringing_measure/
# rf_noise.py carries as BOLTZMANN.
BOLTZMANN_J_PER_K = 1.380649e-23

# ASSUMPTION: no temperature has been measured on the deck or in the room
# where these captures were made. Room temperature; docs/ELLIPTICAL_COLLAPSE
# .md section 7.5 used the same figure for its kTB (-104.9 dBm over 8 MHz).
ROOM_TEMPERATURE_K = 293.0

# ASSUMPTIONS for the thermal term, each labelled with what is and is not
# known about it. None is a measurement. `thermal_term` takes them as
# arguments so a measured value replaces the assumption without editing.
ASSUMPTIONS: Dict[str, Dict[str, object]] = {
    "head_output_rms_v": {
        "value": 0.25e-3,
        "range": (0.125e-3, 0.5e-3),
        "source": (
            "assumption: a typical VHS SP video head's open-circuit output at "
            "the luma carrier, about 0.7 mV peak to peak. Not measured on this "
            "deck. What is on record is the level at the CAPTURE POINT: the "
            "Sony SLV-777HF/778HF/788HF service manual's waveform 7 (PDF page "
            "2, sheet MA-327 (2/8), 'IC201 pin 15 PB / CN261 pin 2 PB', 200 mV "
            "per division) shows the playback RF filling about three "
            "divisions, and the oscilloscope screenshot beside the captures "
            "(DS1202Z-E_PB_RF_example.png: 50.0 MSa/s, 24.0 Mpts, channel 1 "
            "AC-coupled, 100 mV per division, bandwidth limit off, probe 1X) "
            "shows it filling about six of the eight divisions - both about "
            "0.6 V peak to peak, read off graticules. That is after IC260's "
            "gain and whatever gain control it has, neither of which the "
            "manual states (all eleven PDF pages were read; pages 3 to 11 "
            "are the servo, on-screen display, audio, tuner, power and "
            "switch boards and carry nothing on the head or IC260), so it "
            "does not fix the head's own level"),
    },
    "source_resistance_ohm": {
        "value": 20.0,
        "range": (5.0, 100.0),
        "source": (
            "assumption: the real part of a VHS video head's impedance at the "
            "carrier - coil resistance and core loss. The manual gives no head "
            "impedance. The 620 ohm damping resistors across the head lines "
            "(R260, R261, PDF page 1, sheet MA-327 (1/8)) add little through "
            "the head's low impedance"),
    },
    "noise_figure_db": {
        "value": 6.0,
        "range": (2.0, 10.0),
        "source": (
            "assumption: a head-amplifier input stage of about 1 nV per root "
            "hertz against a 20 ohm source. The amplifier is IC260, marked "
            "HA118195NT, 'VIDEO REC/PB AMP' (PDF page 1); no data sheet for it "
            "is in the tree and the manual states no noise figure"),
    },
    "temperature_k": {
        "value": ROOM_TEMPERATURE_K,
        "range": (283.0, 313.0),
        "source": "assumption: room temperature, no measurement exists",
    },
}

# The unified noise weighting network, as docs/SPECIFICATION_INVENTORY.md
# section 3.10 transcribes it from ITU-R BT.1439-1: a constant-resistance
# equaliser with L = Z0 tau, C = tau / Z0, R1 = a Z0, R2 = Z0 / a, tau =
# 245 ns and a = 4.5, whose loss tends to 20 log10(1 + a) = 14.8 dB. Its
# insertion loss follows from those formulae alone: the series arm is R1 in
# parallel with L and the shunt arm R2 in series with C, their product is
# Z0^2, and the response into Z0 is (a + j w tau) / (a + j w tau (1 + a)).
# BT.1439's own check figures - 7.4 dB for flat noise and 12.2 dB for
# triangular noise over a 5 MHz band - are reproduced by that response to
# 0.05 dB (`weighting_noise_factors`, and the unit test).
WEIGHTING_TIME_CONSTANT_S = 245e-9
WEIGHTING_RATIO = 4.5
WEIGHTING_CHECK_BAND_HZ = 5.0e6
WEIGHTING_CHECK_FLAT_DB = 7.4
WEIGHTING_CHECK_TRIANGULAR_DB = 12.2

# The measurement's own structural figures, with their derivations.
#
# The analytic band-pass follows the format: the playback chain's high-pass
# "extracts the FM signal component above 1.4 MHz from the preamplified
# signal" (JVC VTG82063 section 3.2.3, "2) Highpass filter"; the decoder's
# `video_hpf_extra` puts its own at 1.2 MHz) and the decoder's luma
# band-pass tops out at `video_bpf_high`; both are read from the format
# parameters at run time in `format_figures`, not copied here.
#
# The floor is read during the field-sync broad pulses because they are the
# longest dwell of the carrier at one frequency the format offers
# (`vsyncPulseUS` = 27.1 us against 4.7 us for a line sync), and resolution
# is what a floor read beside a tone needs: a Blackman-Harris window over
# the interior of one pulse has a main lobe four bins to either side of the
# tone, 0.2 MHz at 20 us, and sidelobes below -90 dB, so the tone
# contaminates nothing beyond its main lobe. Trimming 2 us at each end of a
# pulse leaves the serration edges and the band-pass's settling outside the
# window; that is about ten settling times of the 5.3 MHz analytic band.
SPECTRAL_WINDOW_S = 20e-6
PULSE_TRIM_S = 2e-6
MAIN_LOBE_BINS = 4.0
# JVC VTG82063 section 3 format table: the sync-tip carrier is specified as
# 3.4 +/- 0.1 MHz for NTSC. A dwell's median frequency must sit inside it.
CARRIER_TOLERANCE_HZ = 0.1e6
# A dwell may be shorter or longer than the nominal pulse by the same
# settling allowance the trim uses: the smoothing blurs each edge by half
# the trim, and the format's own width tolerance is far tighter than that.
DWELL_LENGTH_TOLERANCE_S = PULSE_TRIM_S
# The instantaneous frequency is smoothed over half the trim so that the
# blur it puts on each dwell edge lies wholly inside the trimmed region.
FREQUENCY_SMOOTHING_S = PULSE_TRIM_S / 2.0
# How long a window the sample-rate estimate needs: enough lines for the
# autocorrelation peak to stand clear, which a few thousand give; capped so
# the correlation's FFT stays small.
RATE_ESTIMATE_WINDOW_S = 0.1


def db(power_ratio) -> float:
    return float(10.0 * np.log10(max(float(power_ratio), 1e-300)))


def undb(decibels: float) -> float:
    return float(10.0 ** (float(decibels) / 10.0))


# --------------------------------------------------------------------------
# the format's figures, read from the decoder's own definitions
# --------------------------------------------------------------------------


def format_figures(tape_speed: int = 0) -> Dict[str, float]:
    """The NTSC VHS figures the budget needs, from the decoder's parameter
    sets rather than restated. The import is deferred because
    `lddecode.core` is the decoder's own module and costs seconds to load.

    tip and white carriers: `ire0` and `hz_ire` (4.4 MHz at peak white,
    1 MHz over 140 IRE); line period from the subcarrier; sync widths from
    the system parameters; deviation and baseband from `capture_profile`
    (JVC VTG82063 section 3); de-emphasis from IEC 60774-1 as
    `format_defs/vhs.py` cites it; track width and writing speed from
    `head_model.mechanics_for` (JVC VTG82063 section 1).

    ONLY SP IS AVAILABLE. `head_model.FORMAT_MECHANICS` holds NTSC SP and
    PAL SP and nothing else, so the writing speed - which every wavelength
    and therefore the whole particulate term rests on - exists for SP
    alone. The decoder does carry `video_track_width` for all three speeds
    (58, 29 and 19.3 um), but a track width without its writing speed does
    not make a read volume, and deriving an LP or EP writing speed here
    would be inventing a figure the guide's section 1 table does not state
    in this tree. A speed with no mechanics therefore refuses rather than
    failing obscurely inside `head_model`. The carriers and timings are the
    same at every speed (checked: EP returns the identical ire0, hz_ire,
    line period and vsync pulse width), so an EP capture's FLOOR can be
    measured with the SP figures; only its BUDGET cannot be predicted."""
    from lddecode.core import SysParams_NTSC
    from vhsdecode.format_defs.vhs import (get_rfparams_ntsc_vhs,
                                           get_sysparams_ntsc_vhs)

    system = get_sysparams_ntsc_vhs(SysParams_NTSC, tape_speed)
    rf = get_rfparams_ntsc_vhs({}, tape_speed)
    speed_name = ("SP", "LP", "EP")[min(int(tape_speed), 2)]
    mechanics = head_model.mechanics_for("VHS", "NTSC", speed_name, rf)
    if mechanics is None:
        raise ValueError(
            "no mechanics for VHS NTSC %s: head_model.FORMAT_MECHANICS "
            "carries NTSC SP and PAL SP only, and the writing speed the "
            "particulate term needs is not stated there for %s"
            % (speed_name, speed_name))
    hz_ire = float(system["hz_ire"])
    ire0 = float(system["ire0"])
    return {
        "tip_hz": ire0 + float(system["vsync_ire"]) * hz_ire,
        "blanking_hz": ire0,
        "white_hz": ire0 + 100.0 * hz_ire,
        "hz_ire": hz_ire,
        "luma_span_hz": 100.0 * hz_ire,
        "line_period_s": float(system["line_period"]) * 1e-6,
        "line_sync_s": float(system["hsyncPulseUS"]) * 1e-6,
        "field_sync_pulse_s": float(system["vsyncPulseUS"]) * 1e-6,
        "deviation_hz": capture_profile.NTSC_VHS_SP_DEVIATION_HZ,
        "baseband_hz": capture_profile.NTSC_VHS_SP_BASEBAND_HZ,
        "carson_hz": capture_profile.carson_bandwidth(
            capture_profile.NTSC_VHS_SP_DEVIATION_HZ,
            capture_profile.NTSC_VHS_SP_BASEBAND_HZ),
        "deemphasis_tau_s": float(rf["deemph_tau"]),
        "deemphasis_gain_db": float(rf["deemph_gain"]),
        "fm_band_low_hz": float(rf["video_hpf_extra"]),
        "fm_band_high_hz": float(rf["video_bpf_high"]),
        "writing_speed_m_s": head_model.writing_speed(mechanics),
        "track_width_m": float(mechanics["track_width_m"]),
    }


# --------------------------------------------------------------------------
# term 1: thermal
# --------------------------------------------------------------------------


def thermal_term(carrier_rms_v: Optional[float] = None,
                 source_resistance_ohm: Optional[float] = None,
                 noise_figure_db: Optional[float] = None,
                 temperature_k: Optional[float] = None,
                 band_hz: float = 8.0e6) -> Dict[str, float]:
    """kTB, referred to the carrier as a density.

    The available noise density of a resistive source is k T, raised by the
    amplifier's noise factor F; the carrier's available power is
    V^2 / (4 R). Nothing after the input stage changes the ratio, so the
    C/N0 at the head is the C/N0 at the tap unless a later stage is noisier
    than the first, which is what forty decibels of head-amplifier gain
    exists to prevent. Arguments default to the labelled ASSUMPTIONS; the
    result carries two ranges: over the head output alone (the dominant
    unknown) and over all three electrical assumptions at once."""
    v = (ASSUMPTIONS["head_output_rms_v"]["value"] if carrier_rms_v is None
         else float(carrier_rms_v))
    r = (ASSUMPTIONS["source_resistance_ohm"]["value"]
         if source_resistance_ohm is None else float(source_resistance_ohm))
    nf = (ASSUMPTIONS["noise_figure_db"]["value"] if noise_figure_db is None
          else float(noise_figure_db))
    t = (ASSUMPTIONS["temperature_k"]["value"] if temperature_k is None
         else float(temperature_k))
    noise_factor = undb(nf)
    kt = BOLTZMANN_J_PER_K * t
    noise_density_w_per_hz = kt * noise_factor
    carrier_w = v * v / (4.0 * r)
    c_over_n0 = carrier_w / noise_density_w_per_hz

    def ratio(voltage, resistance, figure_db):
        return db(voltage ** 2 / (4.0 * resistance) / (kt * undb(figure_db)))

    low_v, high_v = ASSUMPTIONS["head_output_rms_v"]["range"]
    low_r, high_r = ASSUMPTIONS["source_resistance_ohm"]["range"]
    low_nf, high_nf = ASSUMPTIONS["noise_figure_db"]["range"]
    return {
        "c_over_n0_db_hz": db(c_over_n0),
        "c_over_n_band_db": db(c_over_n0 / band_hz),
        "band_hz": float(band_hz),
        "kt_dbm_per_hz": db(kt * 1e3),
        "ktb_dbm": db(kt * band_hz * 1e3),
        "noise_density_w_per_hz": noise_density_w_per_hz,
        "equivalent_input_noise_v_per_root_hz": float(
            np.sqrt(4.0 * kt * r * noise_factor)),
        "carrier_available_w": carrier_w,
        "carrier_rms_v": v,
        "source_resistance_ohm": r,
        "noise_figure_db": nf,
        "temperature_k": t,
        # the same term across the labelled range of the least-known input
        "c_over_n0_range_db_hz": (ratio(low_v, r, nf), ratio(high_v, r, nf)),
        # and across every electrical assumption at its extremes at once
        "c_over_n0_full_range_db_hz": (ratio(low_v, high_r, high_nf),
                                       ratio(high_v, low_r, low_nf)),
    }


# --------------------------------------------------------------------------
# term 2: quantisation from effective bits
# --------------------------------------------------------------------------


def effective_bits(measured_density: float, bits: float, sample_rate_hz: float,
                   step: float) -> float:
    """The word length the floor actually delivers. Each effective bit lost
    doubles the noise voltage, so the loss is half the log2 of the measured
    density over the arithmetic one; the same formula
    `information_extrapolation.floor_differential` uses."""
    arithmetic = capture_profile.quantization_floor(
        bits, sample_rate_hz, step * 2.0 ** bits)["density_per_hz"]
    ratio = max(float(measured_density), 1e-300) / max(arithmetic, 1e-300)
    return float(bits) - 0.5 * float(np.log2(max(ratio, 1e-300)))


def quantisation_term(carrier_power: float, bits: float, step: float,
                      sample_rate_hz: float,
                      effective: Optional[float] = None,
                      band_hz: float = 8.0e6) -> Dict[str, float]:
    """The converter's floor against the carrier, in the carrier's units.

    `carrier_power` is A^2 / 2 in the same units the step is measured in.
    With `effective` bits given, the arithmetic density is scaled by
    4^(bits - effective); without it the term is the word length's own."""
    floor = capture_profile.quantization_floor(bits, sample_rate_hz,
                                               step * 2.0 ** bits)
    enob = float(bits) if effective is None else float(effective)
    density = floor["density_per_hz"] * 4.0 ** (float(bits) - enob)
    c_over_n0 = float(carrier_power) / max(density, 1e-300)
    return {
        "c_over_n0_db_hz": db(c_over_n0),
        "c_over_n_band_db": db(c_over_n0 / band_hz),
        "band_hz": float(band_hz),
        "bits": float(bits),
        "effective_bits": enob,
        "step": float(step),
        "arithmetic_density_per_hz": floor["density_per_hz"],
        "density_per_hz": density,
        "carrier_power": float(carrier_power),
    }


def code_histogram(samples, step: float) -> Dict[str, float]:
    """The converter's code use: occupancy, missing codes, and a
    differential non-linearity figure from the histogram's roughness
    against its own local mean. These are diagnostics of the converter -
    an effective-bits figure is NOT derived from them here, because the
    signal's own distribution shapes the histogram and no assumption about
    it is made; the effective bits come from the measured floor instead."""
    codes = np.rint(np.asarray(samples, dtype=np.float64).ravel() / float(step))
    low, high = int(codes.min()), int(codes.max())
    counts = np.bincount((codes - low).astype(np.int64))
    present = int(np.count_nonzero(counts))
    span = high - low + 1
    # local mean over five neighbours, the narrowest window that still
    # averages a code's neighbours on both sides
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(counts, kernel, mode="same")
    interior = (counts > 0) & (smooth > 0)
    interior[:2] = False
    interior[-2:] = False
    dnl = counts[interior] / smooth[interior] - 1.0
    return {
        "codes_present": float(present),
        "codes_spanned": float(span),
        "missing_codes_in_span": float(span - present),
        "dnl_rms": float(np.sqrt(np.mean(dnl ** 2))) if dnl.size else float("nan"),
        "lowest_code": float(low),
        "highest_code": float(high),
    }


# --------------------------------------------------------------------------
# term 3: particulate, 10 log10(N)
# --------------------------------------------------------------------------

# the marked-Poisson law's factor for sinusoidal marks, and for the
# square-wave marks of a saturation recording (see the module docstring)
PARTICLE_LAW_SINUSOIDAL = 0.5
PARTICLE_LAW_SATURATION = 4.0 / np.pi ** 2


def particulate_term(carrier_hz: float, writing_speed_m_s: float,
                     track_width_m: float,
                     read_depth_m: Optional[float] = None,
                     along_track_m: Optional[float] = None,
                     particle_volume_m3: float = tape_model.PARTICLE_VOLUME_M3,
                     packing: float = tape_model.PACKING_FRACTION,
                     saturation: bool = False,
                     band_hz: float = 8.0e6) -> Dict[str, float]:
    """N particles in the read volume, and the density that count implies.

    Defaults are Ethan's construction: read depth lambda / 2 pi (the
    self-demagnetisation bound; JVC VTG82063 section 7.2.1 states the
    recorded depth as "about 0.3 um", which `read_depth_m` accepts as the
    alternative), along-track length one wavelength. The count converts to
    a density by the marked-Poisson law in the module docstring,
    C/N0 = N_lambda f_c / 2, which makes the along-track choice a label on
    the bandwidth and nothing else: any length L gives the carrier-to-noise
    over v / (2 L). `saturation` swaps the sinusoidal-mark factor 1/2 for
    the square-wave factor 4 / pi^2."""
    wavelength = float(writing_speed_m_s) / float(carrier_hz)
    depth = wavelength / (2.0 * np.pi) if read_depth_m is None else float(read_depth_m)
    length = wavelength if along_track_m is None else float(along_track_m)
    volume = float(track_width_m) * depth * length
    count = volume * float(packing) / float(particle_volume_m3)
    factor = PARTICLE_LAW_SATURATION if saturation else PARTICLE_LAW_SINUSOIDAL
    # the count per unit length, and the law C/N0 = factor * n * v
    per_length = count / length
    c_over_n0 = factor * per_length * float(writing_speed_m_s)
    # the band 10 log10(N) is a carrier-to-noise over, by the same law
    count_band_hz = c_over_n0 / count
    return {
        "wavelength_m": wavelength,
        "read_depth_m": depth,
        "along_track_m": length,
        "volume_m3": volume,
        "particles": count,
        "particles_per_m": per_length,
        "ten_log_n_db": db(count),
        "count_band_hz": count_band_hz,
        "law_factor": factor,
        "c_over_n0_db_hz": db(c_over_n0),
        "c_over_n_band_db": db(c_over_n0 / band_hz),
        "band_hz": float(band_hz),
        "particle_volume_m3": float(particle_volume_m3),
        "packing": float(packing),
    }


# --------------------------------------------------------------------------
# the sum, and the same floor under every definition
# --------------------------------------------------------------------------


def sum_of_terms(terms: Dict[str, float]) -> Dict[str, float]:
    """Noise densities add; carrier-to-noise densities therefore combine as
    the reciprocal sum of their linear values. Returns the total and each
    term's share of the total noise."""
    linear = {name: undb(-float(value)) for name, value in terms.items()}
    total = sum(linear.values())
    return {
        "c_over_n0_db_hz": -db(total),
        "shares": {name: value / total for name, value in linear.items()},
    }


def deemphasis_power_response(frequency_hz, tau_s: float, gain_db: float) -> np.ndarray:
    """The format's de-emphasis as a power response: unity at DC, falling
    to -gain_db above the corner. A first-order shelf with RC time constant
    tau and gain factor g = 10^(gain/20): corners at 1/(2 pi tau) and g
    times that (IEC 60774-1 as `format_defs/vhs.py` cites it). Checked
    against the decoder's own filter, `addons/FMdeemph.FMDeEmphasisB` built
    from the same parameter set's `deemph_mid` and `deemph_q` at 50 MSps:
    -0.00, -2.83, -10.26, -12.66 and -13.81 dB at 1, 122, 500, 1000 and
    3000 kHz from the decoder, against -0.00, -2.83, -10.25, -12.66 and
    -13.81 dB from this shelf (2026-09-05). The direction matters:
    `tools/ringing_measure/rf_noise.py`'s `through_deemphasis`, as
    written, tends to +gain_db at high frequency - the pre-emphasis - and
    is not reused here for that reason; it has no caller in the tree."""
    f = np.asarray(frequency_hz, dtype=np.float64)
    lower = 1.0 / (2.0 * np.pi * float(tau_s))
    upper = lower * 10.0 ** (float(gain_db) / 20.0)
    return (1.0 + (f / upper) ** 2) / (1.0 + (f / lower) ** 2)


def weighting_power_response(frequency_hz, tau_s: float = WEIGHTING_TIME_CONSTANT_S,
                             ratio: float = WEIGHTING_RATIO) -> np.ndarray:
    """The unified weighting network's power response,
    |a + j w tau|^2 / |a + j w tau (1 + a)|^2, from the component formulae
    the inventory transcribes (see the constants above)."""
    x = 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64) * float(tau_s)
    a = float(ratio)
    return (a * a + x * x) / (a * a + x * x * (1.0 + a) ** 2)


def weighting_noise_factors(band_hz: float = WEIGHTING_CHECK_BAND_HZ,
                            points: int = 200001) -> Dict[str, float]:
    """The network's noise weighting factors over a band, for flat and for
    triangular noise - the two figures BT.1439 publishes for 5 MHz, which
    check the derived response against the standard."""
    f = np.linspace(0.0, float(band_hz), int(points))
    response = weighting_power_response(f)
    flat = np.trapezoid(response, f) / float(band_hz)
    triangular = np.trapezoid(f ** 2 * response, f) / np.trapezoid(f ** 2, f)
    return {
        "flat_db": -db(flat),
        "triangular_db": -db(triangular),
        "asymptotic_loss_db": float(20.0 * np.log10(1.0 + WEIGHTING_RATIO)),
        "published_flat_db": WEIGHTING_CHECK_FLAT_DB,
        "published_triangular_db": WEIGHTING_CHECK_TRIANGULAR_DB,
    }


def video_signal_to_noise(c_over_n0_db_hz: float, figures: Dict[str, float],
                          deemphasis: bool = True, weighting: bool = False,
                          points: int = 20000) -> float:
    """The demodulated luma signal-to-noise an RF floor implies.

    Around a carrier of power C, additive noise of one-sided density N0
    makes a phase noise of density N0 / C (rad^2/Hz, from the quadrature
    component: variance N0 B over an RF band B, a baseband of B / 2, on a
    carrier of amplitude root(2C)). The demodulator reads frequency,
    (1 / 2 pi) d phi / dt, so the frequency-noise density is f^2 N0 / C in
    Hz^2 per Hz - the triangular noise every FM system has, the law
    `rf_noise.py` states - and integrating it to the luma baseband with the
    de-emphasis, and the weighting if asked, gives the noise in hertz of
    deviation. The signal is the 100 IRE luma span in hertz (`hz_ire` times
    100), and video signal-to-noise is quoted peak-to-peak signal over rms
    noise."""
    f = np.linspace(0.0, float(figures["baseband_hz"]), int(points))
    density = f ** 2 * undb(-float(c_over_n0_db_hz))
    if deemphasis:
        density = density * deemphasis_power_response(
            f, figures["deemphasis_tau_s"], figures["deemphasis_gain_db"])
    if weighting:
        density = density * weighting_power_response(f)
    noise_rms_hz = float(np.sqrt(np.trapezoid(density, f)))
    return float(20.0 * np.log10(float(figures["luma_span_hz"]) / max(noise_rms_hz, 1e-300)))


def definitions(c_over_n0_db_hz: float, figures: Dict[str, float],
                sample_rate_hz: float, bin_hz: Optional[float] = None,
                target_db: float = 45.0) -> Dict[str, float]:
    """One floor, under every label this arc has used for a floor."""
    nyquist = float(sample_rate_hz) / 2.0
    out = {
        "per_hz_db_hz": float(c_over_n0_db_hz),
        "carson_band_db": float(c_over_n0_db_hz) - db(figures["carson_hz"]),
        "deviation_band_db": float(c_over_n0_db_hz) - db(figures["deviation_hz"]),
        "nyquist_band_db": float(c_over_n0_db_hz) - db(nyquist),
        "video_unweighted_no_deemphasis_db": video_signal_to_noise(
            c_over_n0_db_hz, figures, deemphasis=False),
        "video_deemphasised_db": video_signal_to_noise(c_over_n0_db_hz, figures),
        "video_deemphasised_weighted_db": video_signal_to_noise(
            c_over_n0_db_hz, figures, weighting=True),
        # the bin width at which a per-bin reading would show the target
        "bin_width_reading_target_hz": undb(float(c_over_n0_db_hz) - float(target_db)),
    }
    if bin_hz:
        out["per_bin_db"] = float(c_over_n0_db_hz) - db(bin_hz)
        out["bin_hz"] = float(bin_hz)
    return out


def labels_near(defined: Dict[str, float], target_db: float,
                within_db: float) -> List[str]:
    """Which definitions of a floor land within `within_db` of a quoted
    figure - the reconciliation of one number against another."""
    return [name for name, value in defined.items()
            if name.endswith("_db") and np.isfinite(value)
            and abs(float(value) - float(target_db)) <= float(within_db)]


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------


def analytic_signal(samples, sample_rate_hz: float, low_hz: float,
                    high_hz: float) -> np.ndarray:
    """The band-limited analytic signal: the real capture's spectrum kept
    between the two edges, negative frequencies dropped, positive ones
    doubled, so the envelope is the carrier's amplitude and the phase
    derivative its frequency."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size
    spectrum = np.fft.rfft(x)
    frequency = np.fft.rfftfreq(n, 1.0 / float(sample_rate_hz))
    keep = (frequency >= float(low_hz)) & (frequency <= float(high_hz))
    spectrum[~keep] = 0.0
    full = np.zeros(n, dtype=np.complex128)
    full[:spectrum.size] = spectrum * 2.0
    return np.fft.ifft(full)


def instantaneous_frequency(analytic: np.ndarray, sample_rate_hz: float,
                            smoothing_s: float = FREQUENCY_SMOOTHING_S) -> np.ndarray:
    """Cycles per second from the unwrapped phase, smoothed over
    `smoothing_s` so that a dwell is judged on the carrier and not on the
    sample-to-sample phase noise, which at these carrier-to-noise ratios is
    hundreds of kilohertz per sample."""
    phase = np.unwrap(np.angle(analytic))
    frequency = np.diff(phase) * float(sample_rate_hz) / (2.0 * np.pi)
    smoothing = int(round(float(smoothing_s) * float(sample_rate_hz)))
    if smoothing > 1:
        frequency = np.convolve(frequency, np.ones(smoothing) / smoothing, mode="same")
    return np.r_[frequency, frequency[-1]]


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if mask.size and mask[0]:
        starts = np.r_[0, starts]
    if mask.size and mask[-1]:
        ends = np.r_[ends, mask.size]
    return starts, ends


def close_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
    """Fill holes shorter than `gap_samples` in a boolean mask: the phase
    noise punches single-sample holes in a dwell, and a hole shorter than
    the smoothing blur is noise, not a serration."""
    closed = mask.copy()
    if gap_samples <= 0:
        return closed
    starts, ends = _runs(~mask)
    for start, end in zip(starts, ends):
        if end - start < gap_samples and start > 0 and end < mask.size:
            closed[start:end] = True
    return closed


def tip_mask(frequency_hz: np.ndarray, figures: Dict[str, float]) -> np.ndarray:
    """Where the carrier is at the sync tip: nearer the tip frequency than
    the blanking level, the same rule a sync slicer applies at half the
    sync amplitude. Nothing legitimate sits below the tip, so no lower
    bound is needed."""
    threshold = (float(figures["tip_hz"]) + float(figures["blanking_hz"])) / 2.0
    return np.asarray(frequency_hz) < threshold


def carrier_dwells(mask: np.ndarray, sample_rate_hz: float, duration_s: float,
                   length_tolerance_s: float = DWELL_LENGTH_TOLERANCE_S,
                   close_gaps_s: float = FREQUENCY_SMOOTHING_S
                   ) -> List[Tuple[int, int]]:
    """Runs of the mask about `duration_s` long: the line syncs at 4.7 us,
    the field-sync broad pulses at 27.1 us. Gaps shorter than the smoothing
    blur are closed first."""
    rate = float(sample_rate_hz)
    closed = close_gaps(np.asarray(mask, dtype=bool), int(round(close_gaps_s * rate)))
    starts, ends = _runs(closed)
    lengths = (ends - starts) / rate
    low = float(duration_s) - float(length_tolerance_s)
    high = float(duration_s) + float(length_tolerance_s)
    return [(int(s), int(e)) for s, e, length in zip(starts, ends, lengths)
            if low <= length <= high]


def dwell_median_frequency(frequency_hz: np.ndarray, dwells: Sequence[Tuple[int, int]],
                           sample_rate_hz: float, trim_s: float = PULSE_TRIM_S) -> float:
    """The carrier's resting frequency over the interiors of the dwells."""
    trim = int(round(float(trim_s) * float(sample_rate_hz)))
    interiors = [frequency_hz[s + trim:e - trim] for s, e in dwells if e - s > 2 * trim]
    if not interiors:
        return float("nan")
    return float(np.median(np.concatenate(interiors)))


def sample_rate_from_line_period(samples, nominal_rate_hz: float,
                                 figures: Dict[str, float],
                                 passes: int = 3,
                                 window_s: float = RATE_ESTIMATE_WINDOW_S
                                 ) -> Dict[str, float]:
    """The capture's true sample rate, from the format's line period.

    The file header of these captures is a placeholder (it says 40 kHz),
    so the rate is measured: the carrier rests at the sync-tip frequency
    once per line, and the period of that resting, in samples, over the
    line period in seconds, is the rate. The tip frequency read at the
    measured rate is then checked against the format's 3.4 +/- 0.1 MHz - a
    second anchor sharing nothing with the first.

    The first pass must not trust the nominal rate at all: a wrong nominal
    scales every frequency, so a threshold at the tip level would slice
    the wrong thing. It therefore correlates the instantaneous-frequency
    waveform itself, whose line periodicity survives any scale, for a
    coarse rate; later passes rebuild the band-pass at that rate and
    correlate the tip mask, which is sharp. A peak at either edge of the
    search window is not a peak and is refused. Only the first `window_s`
    of the samples is used."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    x = x[:int(round(float(window_s) * float(nominal_rate_hz)))]
    rate = float(nominal_rate_hz)
    period_samples = float("nan")
    tip_measured = float("nan")
    refused = False
    for pass_index in range(max(int(passes), 1)):
        analytic = analytic_signal(x, rate, figures["fm_band_low_hz"],
                                   figures["fm_band_high_hz"])
        frequency = instantaneous_frequency(analytic, rate)
        if pass_index == 0:
            indicator = np.array(frequency, dtype=np.float64)
        else:
            indicator = tip_mask(frequency, figures).astype(np.float64)
        indicator -= indicator.mean()
        n = int(2 ** np.ceil(np.log2(2 * indicator.size)))
        spectrum = np.fft.rfft(indicator, n)
        correlation = np.fft.irfft(np.abs(spectrum) ** 2, n)[:indicator.size]
        nominal_period = figures["line_period_s"] * rate
        low = int(0.7 * nominal_period)
        high = int(1.3 * nominal_period)
        if high >= correlation.size or low < 2:
            break
        peak = low + int(np.argmax(correlation[low:high]))
        if peak <= low or peak >= high - 1:
            refused = True
            break
        # parabolic refinement about the peak
        left, centre, right = correlation[peak - 1:peak + 2]
        denominator = left - 2.0 * centre + right
        offset = 0.5 * (left - right) / denominator if denominator != 0 else 0.0
        period_samples = peak + float(offset)
        rate = period_samples / figures["line_period_s"]
        if pass_index > 0:
            dwells = carrier_dwells(tip_mask(frequency, figures), rate,
                                    figures["line_sync_s"])
            if dwells:
                tip_measured = dwell_median_frequency(frequency, dwells, rate,
                                                      trim_s=FREQUENCY_SMOOTHING_S)
    return {
        "sample_rate_hz": rate,
        "line_period_samples": period_samples,
        "tip_frequency_measured_hz": tip_measured,
        "tip_frequency_spec_hz": figures["tip_hz"],
        "tip_within_tolerance": bool(np.isfinite(tip_measured) and abs(
            tip_measured - figures["tip_hz"]) <= CARRIER_TOLERANCE_HZ),
        "nominal_rate_hz": float(nominal_rate_hz),
        "peak_refused": refused,
    }


def chi_square_median_factor(averages: int) -> float:
    """The median of an average of N periodogram bins of Gaussian noise
    over its mean: chi-square with 2N degrees of freedom, so ln 2 = 0.693
    for one bin (`rf_noise.MEDIAN_OVER_MEAN`) and tending to one as N
    grows. A median read from an averaged spectrum is divided by this to
    compare with a predicted mean."""
    from scipy import stats

    dof = 2 * max(int(averages), 1)
    return float(stats.chi2.median(dof) / dof)


def dwell_spectrum(samples, sample_rate_hz: float,
                   dwells: Sequence[Tuple[int, int]],
                   window_s: float = SPECTRAL_WINDOW_S,
                   trim_s: float = PULSE_TRIM_S) -> Dict[str, np.ndarray]:
    """The one-sided power density of each dwell, windowed (Blackman-
    Harris) over `window_s` centred in its trimmed interior, and their
    mean. Scaling is the periodogram's: 2 |X|^2 / (f_s sum w^2) per hertz."""
    from scipy.signal import windows

    x = np.asarray(samples, dtype=np.float64).ravel()
    rate = float(sample_rate_hz)
    length = int(round(window_s * rate))
    trim = int(round(trim_s * rate))
    window = windows.blackmanharris(length)
    scale = 2.0 / (rate * float(np.sum(window ** 2)))
    per_dwell = []
    for start, end in dwells:
        interior_start, interior_end = start + trim, end - trim
        if interior_end - interior_start < length:
            continue
        centre = (interior_start + interior_end) // 2
        segment = x[centre - length // 2: centre - length // 2 + length]
        if segment.size < length:
            continue
        segment = segment - segment.mean()
        per_dwell.append(np.abs(np.fft.rfft(segment * window)) ** 2 * scale)
    frequency = np.fft.rfftfreq(length, 1.0 / rate)
    stack = np.array(per_dwell) if per_dwell else np.zeros((0, frequency.size))
    density = stack.mean(axis=0) if per_dwell else np.zeros(frequency.size)
    return {"frequency_hz": frequency, "density": density, "per_dwell": stack,
            "averages": np.array(len(per_dwell)), "bin_hz": np.array(rate / length),
            "main_lobe_hz": np.array(MAIN_LOBE_BINS * rate / length),
            "window": window}


def effective_bins(window, bins: int) -> float:
    """How many independent estimates `bins` adjacent periodogram bins hold
    under a window.

    A tapered window correlates neighbouring bins. For Gaussian noise the
    correlation of two bins' powers at lag m is rho(m)^2, with rho the
    transform of the window's square normalised to one at zero lag, so the
    mean of n adjacent bins has a relative variance of
    (1 / n^2) sum over k, l of rho(k - l)^2 per periodogram, and the
    effective count is the reciprocal of that. A rectangular window gives
    n; Blackman-Harris over six bins gives 2.5 - derived here and checked
    by Monte Carlo (relative variance 0.39 against the derived 0.40; the
    unit test repeats the check). The bins had been treated as
    independent, which made every chance bound in `converter_floor` about
    40 per cent too tight, and on a synthetic capture whose noise was
    white everywhere the flat-region walk broke within a few blocks of
    Nyquist by chance alone."""
    w = np.asarray(window, dtype=np.float64)
    n = max(int(bins), 1)
    rho = np.abs(np.fft.rfft(w * w))
    rho = rho / max(float(rho[0]), 1e-300)
    total = 0.0
    for lag in range(n):
        weight = 1.0 if lag == 0 else 2.0
        value = float(rho[lag]) if lag < rho.size else 0.0
        total += weight * (n - lag) * value * value
    return float(n * n / max(total, 1e-300))


def band_median(frequency_hz: np.ndarray, density: np.ndarray,
                low_hz: float, high_hz: float, averages: int = 1) -> float:
    """A median, never a mean: discrete interference doubles a mean and
    leaves a median alone (`information_extrapolation.out_of_band_noise`).
    Divided by the chi-square median factor for `averages`, so the result
    compares with a predicted mean density."""
    inside = (frequency_hz >= float(low_hz)) & (frequency_hz < float(high_hz))
    if np.count_nonzero(inside) < 3:
        return float("nan")
    return float(np.median(density[inside])) / chi_square_median_factor(averages)


def exponential_sample_median_factor(count: int) -> float:
    """The expected sample median of `count` independent exponential
    variables, over their mean: ln 2 in the limit, and above it for small
    samples (0.719 for twenty), from the order statistics of the
    exponential, E[X_(k)] = sum of 1/i for i from n - k + 1 to n."""
    n = max(int(count), 1)

    def order(k):
        return float(np.sum(1.0 / np.arange(n - k + 1, n + 1)))

    if n % 2:
        return order((n + 1) // 2)
    return 0.5 * (order(n // 2) + order(n // 2 + 1))


def per_dwell_floor(spectrum: Dict[str, np.ndarray], low_hz: float, high_hz: float
                    ) -> Dict[str, float]:
    """The floor in a band read separately on every dwell, then the median
    across dwells: unbiased in frequency (a mean over the band's bins, the
    band having been chosen clear of the tone) and robust in time (a
    median over dwells excludes dropouts and switching). That is the TOTAL
    in the band, `density`. A second reading, `density_continuum`, takes
    the median over the band's bins on each dwell, corrected by the
    exponential's small-sample median factor for the band's independent
    bin count (`effective_bins`; the factor assumes independence, so with
    correlated bins it is an approximation of a tenth of a decibel), and
    then the median across dwells: a median is blind to discrete lines and
    to the loud end of a sloped band, so it reads the continuum. The total
    standing above the continuum beyond chance means the band holds
    structure - a line, or a skirt rising toward the tone - which additive
    white noise is not. The uncertainty is the standard error of the
    median from the spread across dwells, with the spread taken from the
    interquartile range so an outlying dwell does not inflate it."""
    f = spectrum["frequency_hz"]
    stack = spectrum["per_dwell"]
    inside = (f >= float(low_hz)) & (f < float(high_hz))
    bins = int(np.count_nonzero(inside))
    if stack.shape[0] == 0 or bins < 3:
        return {"density": float("nan"), "density_continuum": float("nan"),
                "uncertainty_db": float("nan"), "dwells": 0, "bins": bins,
                "independent_bins": float("nan")}
    readings = stack[:, inside].mean(axis=1)
    independent = effective_bins(spectrum["window"], bins)
    robust = (np.median(stack[:, inside], axis=1)
              / exponential_sample_median_factor(int(round(independent))))
    count = readings.size
    median = float(np.median(readings))
    if count >= 4:
        quartiles = np.percentile(readings, [25, 75])
        # a normal's interquartile range is 1.349 sigma; the median's
        # standard error is 1.2533 sigma over root N
        sigma = float(quartiles[1] - quartiles[0]) / 1.349
        standard_error = 1.2533 * sigma / np.sqrt(count)
        uncertainty_db = db(1.0 + standard_error / max(median, 1e-300))
    else:
        uncertainty_db = float("nan")
    continuum = float(np.median(robust))
    return {"density": median, "density_continuum": continuum,
            "uncertainty_db": float(uncertainty_db),
            "dwells": int(count), "bins": bins, "independent_bins": float(independent),
            "mean_over_median_db": db(median / max(continuum, 1e-300))}


def expected_minimum_factor(blocks: int, dof: int, points: int = 4000) -> float:
    """The expected minimum of `blocks` independent chi-square estimates
    of the same density, each with `dof` degrees of freedom, over that
    density: the integral of the survival function to the power of the
    count. The minimum of many noisy estimates of a flat floor sits below
    the floor by exactly this, and dividing by it removes the bias - which
    on a synthetic capture was three decibels."""
    from scipy import stats

    x = np.linspace(0.0, 3.0, int(points))
    survival = stats.chi2.sf(x * dof, dof)
    return float(np.trapezoid(survival ** max(int(blocks), 1), x))


def converter_floor(spectrum: Dict[str, np.ndarray], block_hz: float,
                    confidence: float = 0.95) -> Dict[str, float]:
    """The lowest density anywhere in the capture's band, and the region
    that shares it. Quantisation is white and present everywhere, so the
    minimum of the spectrum bounds the converter's floor from above; the
    analogue chain contributes least where the minimum sits. The spectrum
    is read in blocks of `block_hz` - the mean of each block's bins after
    dropping any bin standing above the block's median by more than the
    chance bound, which removes spurs - the minimum over blocks is
    corrected for the order-statistic bias (`expected_minimum_factor`),
    and the flat top region is every block from the top downward that
    stays within the chance bound of the corrected minimum; its lower edge
    is the capture's roll-off when it has one. Both the bias correction
    and the bound use the block's INDEPENDENT bin count under the window
    (`effective_bins`), not its bin count."""
    from scipy import stats

    f = spectrum["frequency_hz"]
    density = spectrum["density"]
    averages = int(spectrum["averages"])
    bin_hz = float(spectrum["bin_hz"])
    per_block = max(int(round(float(block_hz) / bin_hz)), 3)
    blocks = int(f.size // per_block)
    if blocks < 3 or averages < 1:
        return {"density": float("nan"), "at_hz": float("nan"),
                "flat_from_hz": float("nan")}
    # the window correlates the bins of a block: a block estimate has the
    # degrees of freedom of its INDEPENDENT bins, not of its bins
    independent = effective_bins(spectrum["window"], per_block)
    # a single averaged bin is chi-square with 2 * averages degrees of
    # freedom; a bin above the block's median by more than its 99.9 per
    # cent point is a spur
    bin_dof = 2 * averages
    spur = float(stats.chi2.ppf(0.999, bin_dof) / bin_dof) / chi_square_median_factor(averages)
    estimates = np.zeros(blocks)
    kept = np.zeros(blocks, dtype=np.int64)
    centres = np.zeros(blocks)
    for i in range(blocks):
        block = density[i * per_block:(i + 1) * per_block]
        keep = block <= np.median(block) * spur
        estimates[i] = float(block[keep].mean()) if keep.any() else float(block.mean())
        kept[i] = int(np.count_nonzero(keep))
        centres[i] = float(f[i * per_block:(i + 1) * per_block].mean())
    # the top block sits against the Nyquist edge, whose bins are folded,
    # and the bottom block against zero frequency
    usable = np.arange(1, blocks - 1)
    index = int(usable[np.argmin(estimates[usable])])
    independent_kept = independent * float(kept[index]) / per_block
    dof = bin_dof * independent_kept
    correction = expected_minimum_factor(usable.size, dof)
    minimum = float(estimates[index]) / correction
    bound = _chance_bound(averages * independent_kept, confidence)
    # the flat region: walking down from the top, a block above the
    # corrected minimum by more than the chance bound is a chance
    # exceedance unless the next block is too - one in twenty blocks of
    # a flat floor clears a 95 per cent bound - so two in a row end it
    flat_from = centres[index]
    for i in range(blocks - 2, 0, -1):
        if estimates[i] <= minimum * bound:
            flat_from = centres[i]
        elif i - 1 >= 1 and estimates[i - 1] <= minimum * bound:
            continue
        else:
            break
    return {"density": minimum, "raw_minimum": float(estimates[index]),
            "minimum_correction": float(correction),
            "at_hz": float(centres[index]), "flat_from_hz": float(flat_from),
            "block_hz": per_block * bin_hz, "chance_bound": float(bound),
            "independent_bins_per_block": float(independent)}


def _chance_bound(averages: float, confidence: float) -> float:
    """The formula `tools/ringing_measure/rf_noise.py` carries as
    `spectrum_confidence` (a unit test holds the two equal; that module
    is a script beside its tool, not a package, so it is not imported
    here), accepting a fractional count so an effective number of
    independent bins passes through - chi-square is defined for
    fractional degrees of freedom."""
    from scipy import stats

    dof = 2.0 * max(float(averages), 1.0)
    return float(stats.chi2.ppf(confidence, dof) / dof)


def measure_floor(samples, sample_rate_hz: float, figures: Dict[str, float],
                  bits: Optional[float] = None, step: Optional[float] = None
                  ) -> Dict[str, object]:
    """The RF floor of one capture window, under the carrier.

    HOW. The carrier rests at the sync-tip frequency for 27.1 us in each
    field-sync broad pulse (six per field). Inside those dwells the FM
    carrier is one tone, so every other bin of a windowed spectrum is
    noise, and the noise UNDER THE FM BAND - which content otherwise
    covers - can be read directly: on each pulse the median density from
    the tip tone's main lobe up to the peak-white carrier plus the same
    lobe, and the median of those across pulses. The carrier's power is the
    square of the analytic envelope over the same pulses, halved. The tip
    tone's frequency is measured, so the read region follows the tape's
    carrier and not the nominal one, and it is checked against the
    format's tolerance.

    Also returned: the floor in rings of offset about the tone (noise that
    rides the carrier shows there and not further out), the floor in the
    upper sideband region above peak white, the converter's floor as the
    spectrum's minimum with its flat region, the envelope's own spread over
    the dwells as a second reading (an upper bound: it also holds the
    multiplicative modulation noise and the channel's own level variation),
    and the full-band carrier share from the kurtosis
    (`interference_distributions.carrier_fraction`), which is the C/N over
    the whole Nyquist band with everything that is not the carrier counted
    as noise - the definition under which one capture read 11.2 dB."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    rate = float(sample_rate_hz)
    nyquist = rate / 2.0
    if bits is None or step is None:
        lattice = capture_profile.measure_bit_depth(np.asarray(samples))
    else:
        lattice = {"bits": float(bits), "step": float(step)}
    analytic = analytic_signal(x, rate, figures["fm_band_low_hz"],
                               figures["fm_band_high_hz"])
    frequency = instantaneous_frequency(analytic, rate)
    envelope = np.abs(analytic)
    at_tip = tip_mask(frequency, figures)
    broad = carrier_dwells(at_tip, rate, figures["field_sync_pulse_s"])
    lines = carrier_dwells(at_tip, rate, figures["line_sync_s"])
    out: Dict[str, object] = {
        "sample_rate_hz": rate,
        "samples": int(x.size),
        "broad_pulses": len(broad),
        "line_syncs": len(lines),
        "bits": float(lattice["bits"]),
        "step": float(lattice["step"]),
        "carrier_power_all": float(np.mean(envelope ** 2) / 2.0),
    }
    if not broad:
        out["why"] = "no field-sync broad pulses found; the floor needs them"
        return out
    trim = int(round(PULSE_TRIM_S * rate))
    interiors = [envelope[s + trim:e - trim] for s, e in broad if e - s > 2 * trim]
    amplitude = float(np.median([seg.mean() for seg in interiors]))
    spread = float(np.median([seg.std() / seg.mean() for seg in interiors]))
    carrier_power = amplitude ** 2 / 2.0
    tip_measured = dwell_median_frequency(frequency, broad, rate)
    spectrum = dwell_spectrum(x, rate, broad)
    f, density = spectrum["frequency_hz"], spectrum["density"]
    averages = int(spectrum["averages"])
    lobe = float(spectrum["main_lobe_hz"])
    tip = tip_measured if np.isfinite(tip_measured) else figures["tip_hz"]
    white = tip + figures["deviation_hz"]
    region = (tip + lobe, white + lobe)
    in_band = per_dwell_floor(spectrum, *region)
    carson_top = (tip + white) / 2.0 + figures["carson_hz"] / 2.0
    rings = {}
    for low, high in ((1.0, 2.0), (2.0, 3.5), (3.5, 6.5)):
        above = band_median(f, density, tip + low * lobe, tip + high * lobe, averages)
        below = band_median(f, density, tip - high * lobe, tip - low * lobe, averages)
        rings["%.1f-%.1f MHz" % (low * lobe / 1e6, high * lobe / 1e6)] = float(
            np.nanmean([above, below]))
    upper_sideband = band_median(f, density, white + lobe, min(carson_top, nyquist - lobe),
                                 averages)
    converter = converter_floor(spectrum, block_hz=CARRIER_TOLERANCE_HZ + lobe)
    arithmetic = capture_profile.quantization_floor(
        lattice["bits"], rate, lattice["step"] * 2.0 ** lattice["bits"])["density_per_hz"]
    # the envelope's spread reads the in-phase quadrature over the analytic
    # band: sigma^2 = N0 B, so N0 / C = 2 (sigma / A)^2 / B
    band = figures["fm_band_high_hz"] - figures["fm_band_low_hz"]
    envelope_c_over_n0 = -db(2.0 * spread ** 2 / band) if spread > 0 else float("nan")
    share = carrier_fraction(x)
    floor = in_band["density"]
    continuum = float(in_band.get("density_continuum", float("nan")))
    # what stands above the arithmetic floor at the spectrum's minimum is
    # the electronics' noise at the top of the band: a LOWER BOUND on the
    # electronic floor's carrier-to-noise, since nothing says the head
    # amplifier's noise gain is flat out there
    top_excess = (float(converter["density"]) - arithmetic
                  if np.isfinite(converter["density"]) else float("nan"))
    out.update({
        "carrier_amplitude_tip": amplitude,
        "carrier_amplitude_tip_codes": amplitude / max(float(lattice["step"]), 1e-300),
        "carrier_power_tip": carrier_power,
        "tip_frequency_measured_hz": tip_measured,
        "tip_within_tolerance": bool(np.isfinite(tip_measured) and abs(
            tip_measured - figures["tip_hz"]) <= CARRIER_TOLERANCE_HZ),
        "in_band_density": floor,
        "in_band_region_hz": region,
        "in_band_dwells": in_band["dwells"],
        "in_band_bins": in_band["bins"],
        "in_band_mean_over_median_db": in_band.get("mean_over_median_db", float("nan")),
        "c_over_n0_db_hz": db(carrier_power / floor) if floor > 0 else float("nan"),
        "c_over_n0_uncertainty_db": in_band["uncertainty_db"],
        "c_over_n0_continuum_db_hz": (db(carrier_power / continuum)
                                      if continuum > 0 else float("nan")),
        "in_band_independent_bins": float(in_band.get("independent_bins", float("nan"))),
        "top_of_band_excess_density": top_excess,
        "top_of_band_excess_c_over_n0_db_hz": (
            db(carrier_power / top_excess) if top_excess > 0 else float("inf")),
        "c_over_n0_from_mean_spectrum_db_hz": db(
            carrier_power / band_median(f, density, *region, averages=averages)),
        "rings_c_over_n0_db_hz": {name: db(carrier_power / value) if value > 0 else float("nan")
                                  for name, value in rings.items()},
        "upper_sideband_density": upper_sideband,
        "upper_sideband_c_over_n0_db_hz": (db(carrier_power / upper_sideband)
                                           if upper_sideband > 0 else float("nan")),
        "converter_density": converter["density"],
        "converter_at_hz": converter["at_hz"],
        "converter_flat_from_hz": converter["flat_from_hz"],
        "converter_over_arithmetic_db": (db(converter["density"] / arithmetic)
                                         if converter["density"] > 0 else float("nan")),
        "converter_c_over_n0_db_hz": (db(carrier_power / converter["density"])
                                      if converter["density"] > 0 else float("nan")),
        "effective_bits": effective_bits(converter["density"], lattice["bits"], rate,
                                         lattice["step"]),
        "arithmetic_density": arithmetic,
        "envelope_spread": spread,
        "envelope_c_over_n0_db_hz": envelope_c_over_n0,
        "kurtosis_carrier_share": share.get("carrier_share", float("nan")),
        "kurtosis_c_over_n_full_band_db": share.get("carrier_to_noise_db", float("nan")),
        "spectrum_averages": averages,
        "spectrum_bin_hz": float(spectrum["bin_hz"]),
        "spectrum_main_lobe_hz": lobe,
        "frequency_hz": f,
        "density": density,
    })
    return out


# --------------------------------------------------------------------------
# the budget
# --------------------------------------------------------------------------


def budget(figures: Dict[str, float], sample_rate_hz: float,
           carrier_power: float, bits: float, step: float,
           effective: Optional[float] = None,
           carrier_hz: float = 4.0e6,
           measured_c_over_n0_db_hz: Optional[float] = None,
           measured_uncertainty_db: Optional[float] = None,
           thermal_arguments: Optional[Dict[str, float]] = None,
           particulate_arguments: Optional[Dict[str, float]] = None
           ) -> Dict[str, object]:
    """Predict the three terms for this capture, sum them, and if a
    measured floor is given, report the gap and what it points at.

    `carrier_hz` defaults to the 4.0 MHz of Ethan's arithmetic; the
    particulate term moves by 0.6 dB across the deviation range. The
    budget closes on the central assumptions when the gap is inside the
    measurement's own uncertainty, and within the assumptions when the
    labelled range of the thermal term reaches the measured floor."""
    band = figures["carson_hz"]
    thermal = thermal_term(band_hz=band, **(thermal_arguments or {}))
    quantisation = quantisation_term(carrier_power, bits, step, sample_rate_hz,
                                     effective, band_hz=band)
    particulate = particulate_term(float(carrier_hz), figures["writing_speed_m_s"],
                                   figures["track_width_m"], band_hz=band,
                                   **(particulate_arguments or {}))
    terms = {"thermal": thermal["c_over_n0_db_hz"],
             "quantisation": quantisation["c_over_n0_db_hz"],
             "particulate": particulate["c_over_n0_db_hz"]}
    total = sum_of_terms(terms)
    def total_with_thermal(value):
        return sum_of_terms({**terms, "thermal": value})["c_over_n0_db_hz"]

    total_range = tuple(total_with_thermal(v) for v in thermal["c_over_n0_range_db_hz"])
    total_full_range = tuple(total_with_thermal(v)
                             for v in thermal["c_over_n0_full_range_db_hz"])
    out: Dict[str, object] = {
        "thermal": thermal,
        "quantisation": quantisation,
        "particulate": particulate,
        "terms_db_hz": terms,
        "predicted_c_over_n0_db_hz": total["c_over_n0_db_hz"],
        "predicted_c_over_n0_range_db_hz": total_range,
        "predicted_c_over_n0_full_range_db_hz": total_full_range,
        "predicted_carson_db": total["c_over_n0_db_hz"] - db(band),
        "shares": total["shares"],
        "band_hz": band,
    }
    if measured_c_over_n0_db_hz is not None:
        measured = float(measured_c_over_n0_db_hz)
        uncertainty = (float(measured_uncertainty_db)
                       if measured_uncertainty_db is not None
                       and np.isfinite(measured_uncertainty_db) else 0.0)
        gap = total["c_over_n0_db_hz"] - measured
        gap_range = (total_range[0] - measured, total_range[1] - measured)
        gap_full_range = (total_full_range[0] - measured, total_full_range[1] - measured)
        out.update({
            "measured_c_over_n0_db_hz": measured,
            "measured_uncertainty_db": uncertainty,
            "measured_carson_db": measured - db(band),
            "gap_db": gap,
            "gap_range_db": gap_range,
            "gap_full_range_db": gap_full_range,
            "unexplained_share": 1.0 - undb(-gap) if gap > 0 else 0.0,
            "closes": bool(abs(gap) <= uncertainty),
            "closes_within_thermal_range": bool(
                gap_range[0] - uncertainty <= 0.0 <= gap_range[1] + uncertainty),
            "closes_within_full_range": bool(
                gap_full_range[0] - uncertainty <= 0.0 <= gap_full_range[1] + uncertainty),
            "verdict": _verdict(gap, gap_range, gap_full_range, uncertainty,
                                total["shares"]),
        })
    return out


def _verdict(gap_db: float, gap_range_db: Tuple[float, float],
             gap_full_range_db: Tuple[float, float],
             uncertainty_db: float, shares: Dict[str, float]) -> str:
    """What the size of the gap points at, in words."""
    dominant = max(shares, key=shares.get)
    low, high = gap_range_db
    full_low, full_high = gap_full_range_db
    if abs(gap_db) <= uncertainty_db:
        return ("the budget closes: the measured floor is within the "
                "measurement's own uncertainty (%.2f dB) of the sum, and "
                "nothing is unexplained at this precision" % uncertainty_db)
    if gap_db < 0:
        return ("the measured floor is BETTER than the sum by %.1f dB, which "
                "means a term is overstated - the %s term dominates the "
                "prediction and its assumptions are the place to look"
                % (-gap_db, dominant))
    if low - uncertainty_db <= 0.0:
        return ("the measured noise exceeds the sum by %.1f dB on the central "
                "assumptions, but the thermal term's labelled head-output "
                "range reaches it (gap %.1f to %.1f dB): the head amplifier's "
                "input - its signal level and noise figure - is the first "
                "unknown to pin down, by measuring the head output or by a "
                "tape-stopped capture; modulation noise, which rides the "
                "carrier and which an additive particle count omits, is the "
                "second" % (gap_db, low, high))
    if full_low - uncertainty_db <= 0.0:
        return ("the measured noise exceeds the sum by %.1f dB; halving the "
                "assumed head output does not reach it (gap %.1f to %.1f dB, "
                "the thermal term being a minority of the sum), and only the "
                "joint extremes of the head output, source resistance and "
                "noise figure do (gap %.1f to %.1f dB): the head amplifier's "
                "input is unmeasured in every respect, and a tape-stopped "
                "capture, which reads that amplifier's noise alone at the "
                "tap, is the measurement that decides whether the gap is the "
                "electronics or the coating's modulation noise"
                % (gap_db, low, high, full_low, full_high))
    return ("the measured noise exceeds the sum by %.1f dB and no labelled "
            "assumption reaches it (gap %.1f to %.1f dB over every "
            "assumption): the unexplained power is %.0f%% of the floor, and "
            "points at mechanisms outside the three terms - modulation noise "
            "of the coating, head-amplifier excess noise beyond its noise "
            "figure, or the tape stock's particle constants - in that order "
            "of what the data can separate"
            % (gap_db, full_low, full_high, 100.0 * (1.0 - undb(-gap_db))))
