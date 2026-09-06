"""The depth split: one shared head-to-tape separation, and a coating term
that each band reads at its own depth.

Ethan, 2026-09-05: *"4·fsc → 69.8 ns/sample → 0.405 µm of tape per sample
at 5.8 m/s. λ = v/f: 1.71 µm at sync tip (3.4 MHz), 1.32 µm at peak white
(4.4 MHz), 9.22 µm at chroma (629 kHz). Read depth ≈ λ/2π → luma reads
~0.2 µm down, chroma ~1.5 µm. The bands sample different depths.
Consequence: timing is genuinely common-mode across bands; amplitude is
not. Split the current spacing term into shared head-to-tape separation
plus a band-specific depth term. Thickness loss (1−e^(−kδ))/kδ with
δ≈4–5 µm → ≈−26 dB luma, −10 dB chroma. Most of the luma↔chroma band
mismatch, and modelable given δ."*

HIS NUMBERS AND THE REPRODUCTION, side by side. Writing speed 5.80 m/s is
JVC VTG82063 section 1, stated (head_model.FORMAT_MECHANICS); the FM
carrier is SMPTE 32M / JVC table 1-1-2, sync tip 3.4 MHz and peak white
4.4 MHz (vhs.py: hz_ire = 1e6/140, ire0 = 4.4e6 - 100 hz_ire); the
colour-under carrier is 40 f_H = 629 370.63 Hz (colour_under.carrier_hz,
derived); 4 fsc = 14.318 18 MHz (SMPTE 170M).

    quantity                        Ethan        reproduced        note
    tape per 4fsc sample            0.405 um     0.4051 um         exact
    lambda, sync tip 3.4 MHz        1.71 um      1.706 um          exact
    lambda, peak white 4.4 MHz      1.32 um      1.318 um          exact
    lambda, chroma 629 kHz          9.22 um      9.216 um          exact
    read depth lambda/2pi, luma     ~0.2 um      0.237 um at 3.9   0.210-0.271 over the deviation
    read depth lambda/2pi, chroma   ~1.5 um      1.467 um          exact
    thickness loss, luma            ~-26 dB      -25.6 dB          at delta = 4.5 um, 3.9 MHz
    thickness loss, chroma          ~-10 dB      -10.2 dB          at delta = 4.5 um
    one delta for both figures      4-5 um       4.37 um           residuals +0.7 / +0.1 dB
    EP writing speed                             5.83 m/s          JVC 1-1-1: SP->EP moves lambda 0.5%, NOT 3x

Every one of his figures reproduces from the specification. The two things
that do NOT follow are the ones this module exists to record, and both are
measured rather than argued.

WHERE THE DEPTH LAWS DISAGREE, AND WHAT THE TAPE SAYS. Three depths are in
play and they give three different thickness losses:

    depth law                          luma 3.9 MHz   chroma 629 kHz   difference
    whole coating, delta = 4.5 um      -25.6 dB       -10.2 dB         15.4 dB   <- Ethan's figures
    JVC 7.2.1 record depth 0.30 um     -3.98 (capped) -0.87 dB          3.1 dB   <- magnetic.recording_depth
    read depth lambda/2pi (x = 1)      -3.98 dB       -3.98 dB          0.0 dB   <- the cap binding in both

The whole-coating figures need the video head to magnetise 4.5 um of
coating. JVC VTG82063 section 7.2.1 states the video *"is recorded to a
depth of about 0.3 um ... by the 0.3 um gap"*, and the format's Hi-Fi depth
multiplex depends on it: the audio survives BENEATH the video, which it
could not if the video head wrote through the coating. And the coating
thickness itself is in no document the repository holds
(docs/SPECIFICATION_INVENTORY.md 2.3: "no ... coating thickness figure
anywhere in the collection"); 4-5 um is the literature's typical VHS oxide
layer on a 14-15 um base for the 19.0 um tape SMPTE 32M table 1 specifies,
and is entered here as a LABELLED ASSUMPTION, not a specification.

MEASURED ON THE SONY SLV-778HF RECORD AND PLAYBACK TAPS (CN261 pins 1 and
2, 50 MS/s, /testdata readme), eleven chroma-carrying test patterns at each
speed, the playback/record power ratio of smoothed spectra (24.4 kHz bins,
wider than one line spacing, because the format's +-90 deg/line chroma
rotation puts the chroma comb at (n +- 1/4) f_H and the EP FM combs
interleave by f_H/2 - a harmonic picker at n f_H reads shoulders and
scattered the first pass 12 dB rms):

    chroma band 0.30-1.00 MHz, ratio slope      SP +3.2 +- 0.4 dB/oct   EP +2.6 +- 0.3
      (head differentiation alone is +6.02; the round trip LOSES
       2.8 dB/oct on SP against the differentiator)
    luma band 3.2-5.0 MHz, effective spacing    SP 0.220 +- 0.052 um    EP 0.14 +- 0.11
      (the ratio's own tilt with the differentiation removed; it
       absorbs the deck's equaliser and the record transition length,
       exactly as the 0.487 um of the envelope fit did)
    cross-band level, ratio(3.9 MHz) - ratio(629 kHz)
                                                SP +4.02 +- 0.33 dB     EP +1.72 +- 0.54
      (differentiation alone would give +15.84: the round trip loses
       11.8 dB more in the luma band than in the chroma band)

    THE SHAPE TEST. With the shared separation held at each capture's own
    luma-band value and only the chroma band's depth free:
        delta_chroma = 1.17 +- 0.33 um (SP), 1.87 +- 0.42 (EP), rms 0.2-0.7 dB
    With delta = 4.5 um held instead, the separation pins at its lower
    bound and the rms is 4-6 TIMES WORSE (1.00 against 0.17 dB on the
    bars, 0.93 against 0.24 on bounce, 0.97 against 0.20 on the ramp):
    the whole-coating curve is the wrong shape for the chroma band.
    With delta = 0.30 um held, the rms is 0.6-0.7 dB: the chroma band
    shows 1.4 +- 0.4 dB/oct (SP) MORE loss than the luma band's own
    separation predicts through Wallace's law. That excess IS the
    band-specific term, and it is the size of a uniformly magnetised
    layer of about 1.2 um - between the 0.3 um record depth and the
    1.47 um read depth - or, indistinguishably in shape, 0.35 um more
    effective separation for the chroma band than for the luma band.

    THE LEVEL TEST. Predicted ratio(3.9 MHz) - ratio(629 kHz) with the
    measured shared separation 0.22 um (`predicted_band_mismatch_db`;
    differentiation +15.84, separation -6.77, gap -0.71, then the depth
    term), the luma at its lambda/2pi cap unless said otherwise:
        whole coating 4.5 um, both bands       -7.1 dB   11.1 dB below the measured +4.0
        chroma 1.17 um (the shape's), luma cap +7.6 dB    3.6 above
        chroma 1.17 um, luma the same layer    -2.3 dB    6.4 below
        JVC 0.30 um, capped / uncapped         +5.3 / +4.3 dB   1.2 / 0.3 above  <- matches
    The whole-coating thickness loss would need 11 dB of luma-band
    boost from somewhere the luma band's own flat ratio does not show.
    So: the mismatch Ethan points at is REAL (11.8 dB against the
    differentiator) and it is NOT mostly thickness. On this deck it
    decomposes as shared separation 6.8 dB, thickness 3.1-4.1 dB (JVC
    depth), gap 0.7 dB. The shape and the level are not reconciled by
    ONE (d, delta) pair: the shape wants the chroma depth near 1.2 um,
    which over-predicts the level by 3.6 dB; the level wants the JVC
    depth, which under-predicts the chroma tilt by 1.4 dB/oct; the gap
    between them is the size of the confounds named below.

    CONFOUNDS, unbounded by this measurement: the head-amplifier's own
    response between 0.63 and 3.9 MHz (the luma band is flat to 4.7 MHz
    then rolls off, which is an equalised response and not a Wallace
    line, so extrapolating its tilt to the chroma band is the weakest
    step); whether CN261 pin 1 is the head current or a voltage upstream
    of the driver; the FM being saturation-recorded while the colour-
    under is bias-recorded by it (a flux-per-current difference that
    only makes the level HARDER for the whole-coating law, since
    compression lowers the luma band); and the FM's lower sidebands,
    which saturation recording regenerates and which the y-only
    patterns show at +5 to +15 dB in the chroma band, 10-20 dB under
    the colour-under in the chroma-carrying patterns.

TIMING IS COMMON-MODE, MEASURED. The two pilots - sync from the luma, burst
from the chroma - share one mechanical time base, and the band losses'
minimum-phase delays are the only way the bands could disagree. Wallace's
law gives tau = -(2d/(pi v))(ln w + 1), so the two bands differ by
(2d/(pi v)) ln(f_l/f_c) = 10 ns per 0.05 um of separation (97 ns for the
0.487 um effective figure). On the sync-to-burst export (bars SP, 20
fields, 4 fsc output samples; 69.84 ns per sample), active lines 40-250:

    per-line disagreement, per field    10.9 / 9.2 ns rms (heads A / B)
    the burst pilot's own residual      11.4 / 10.3 ns rms
    the sync pilot's own residual       23.2 / 21.5 ns rms
    per-field mean disagreement         within +-1.5 ns over ten fields
    per-head fixed offset               +1.3 / +3.2 ns, paired -2.0 +- 0.4 ns

The disagreement is at the pilots' own noise: the two time bases agree
within error (`common_mode_timing`: 10.9/9.2 ns against a combined pilot
error of 25.9/23.8), and the per-head difference of 2 ns bounds a per-head
separation difference at 0.010 um. A CONSTANT delay between the bands is
invisible (the chroma path's filters set one too), so the test is on what
varies - per field, per head, per line - and it passes. The 30 ns on
lines 0-40 is the burst lock re-acquiring after the vertical interval, and
the 40 ns rms on lines 200-263 is the head switch; neither is the tape.

AMPLITUDE IS NOT COMMON-MODE, MEASURED (memory: vhs-head-model). The
per-head amplitude difference reads +0.17 nepers in the FM band and +0.22
at 629 kHz, where a shared separation predicts 0.011: the chroma band does
not take the luma's amplitude deviation in the ratio f_c/f_l, which is what
`WAVELENGTH_INDEPENDENT_NOISE = 0.194` in vhsdecode/chroma.py already
carries empirically. The band-specific term above is its mechanism.

THE SPLIT ITSELF, and where it is identifiable. In the luma band the
coating term is a pure gain whatever the depth: at x = 2 pi delta/lambda
around 19 its derivative to ln(delta) is -1.000 flat, and at the cap it is
zero; so the luma band determines the SHARED separation (as a tilt) and
never the depth. In the chroma band the depth term has a shape only in the
transition regime x ~ 1-5: `identifiability_control` reads condition 1.8
to 5.7 there with the per-band gains unknown, against 14 to 110 at the
JVC depth, where x << 1 and the thickness term collapses EXACTLY onto a
Wallace spacing of delta/2 (`small_x_control`, coherence 0.9999). The
split is therefore determined only when the chroma band reads a depth of
the order of its own lambda/2pi, which is what the shape measurement
says it does. The other lane (loss_separation.py) owns spacing against gap
against thickness across clusters; this module owns the two-band depth
split and the timing test.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# specification, every figure with its source
# --------------------------------------------------------------------------

WRITING_SPEED_M_S = {"SP": 5.80,     # JVC VTG82063 section 1, stated
                     "EP": 5.83}     # JVC VTG82063 table 1-1-1 (SPECIFICATION_INVENTORY 2.2a)
NTSC_LINE_RATE_HZ = 525.0 * 30.0 / 1.001                 # SMPTE 170M
NTSC_SUBCARRIER_HZ = 455.0 / 2.0 * NTSC_LINE_RATE_HZ     # SMPTE 170M
COLOUR_UNDER_HZ = 40.0 * NTSC_LINE_RATE_HZ               # SMPTE 32M / JVC 1-1-2, derived
FM_SYNC_TIP_HZ = 3.4e6                                   # SMPTE 32M / JVC 1-1-2
FM_PEAK_WHITE_HZ = 4.4e6                                 # SMPTE 32M / JVC 1-1-2
FM_PEAK_HZ = 3.9e6                                       # decoder video_rf_peak_freq, band centre
OUTPUT_RATE_HZ = 4.0 * NTSC_SUBCARRIER_HZ                # 4 fsc, 14.318 18 MHz
RECORD_DEPTH_M = 0.30e-6         # JVC VTG82063 7.2.1: "recorded to a depth of about 0.3 um"
GAP_M = 0.30e-6                  # same sentence
EFFECTIVE_GAP_FACTOR = 1.11      # head_model.EFFECTIVE_GAP_FACTOR
# NOT a specification. No document in the collection gives a coating
# thickness (SPECIFICATION_INVENTORY 2.3). 4-5 um is the literature's
# typical VHS oxide layer on a 14-15 um base for the 19.0 +1/-2 um T-120
# tape of SMPTE 32M table 1. Labelled, bounded, and used only to reproduce
# Ethan's -26/-10 dB; the measurement above does not support it as the
# depth either band reads.
COATING_THICKNESS_ASSUMED_M = (4.0e-6, 5.0e-6)
DB_PER_NEPER = 20.0 / np.log(10.0)

# Ethan's reference figures, verbatim, for the side-by-side.
ETHAN_REFERENCE: Dict[str, float] = {
    "tape_per_sample_um": 0.405,
    "lambda_sync_tip_um": 1.71,
    "lambda_peak_white_um": 1.32,
    "lambda_chroma_um": 9.22,
    "read_depth_luma_um": 0.2,
    "read_depth_chroma_um": 1.5,
    "thickness_loss_luma_db": -26.0,
    "thickness_loss_chroma_db": -10.0,
    "coating_thickness_um": 4.5,
}

# What this module measured, with the instrument named.
MEASURED: Dict[str, float] = {
    # Sony SLV-778HF record/playback taps, 11 chroma-carrying patterns each
    "chroma_slope_db_per_octave_sp": 3.2, "chroma_slope_sd_sp": 0.4,
    "chroma_slope_db_per_octave_ep": 2.6, "chroma_slope_sd_ep": 0.3,
    "luma_effective_spacing_um_sp": 0.220, "luma_effective_spacing_sd_sp": 0.052,
    "luma_effective_spacing_um_ep": 0.141, "luma_effective_spacing_sd_ep": 0.108,
    "chroma_depth_given_luma_spacing_um_sp": 1.17, "chroma_depth_sd_sp": 0.33,
    "chroma_depth_given_luma_spacing_um_ep": 1.87, "chroma_depth_sd_ep": 0.42,
    "cross_band_level_db_sp": 4.02, "cross_band_level_sd_sp": 0.33,
    "cross_band_level_db_ep": 1.72, "cross_band_level_sd_ep": 0.54,
    "rms_db_whole_coating_bars_sp": 1.00, "rms_db_fitted_depth_bars_sp": 0.17,
    # sync-to-burst export, bars SP, 20 fields, active lines 40-250
    "timing_disagreement_rms_ns_head_a": 10.9, "timing_disagreement_rms_ns_head_b": 9.2,
    "burst_pilot_rms_ns_head_a": 11.4, "burst_pilot_rms_ns_head_b": 10.3,
    "sync_pilot_rms_ns_head_a": 23.2, "sync_pilot_rms_ns_head_b": 21.5,
    "per_head_timing_offset_ns": -2.0, "per_head_timing_offset_se_ns": 0.4,
}


def _grid(frequency_hz) -> np.ndarray:
    return np.asarray(frequency_hz, dtype=np.float64).ravel()


# --------------------------------------------------------------------------
# lengths
# --------------------------------------------------------------------------

def wavelength_m(frequency_hz, writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> np.ndarray:
    """The recorded wavelength, lambda = v / f."""
    return writing_speed_m_s / np.maximum(_grid(frequency_hz), 1.0)


def read_depth_m(frequency_hz, writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> np.ndarray:
    """The depth a wavelength is read from: lambda / (2 pi).

    The playback head weights a layer at depth y by exp(-2 pi y / lambda),
    so the 1/e depth of what it reads is lambda / (2 pi); it is also the
    self-demagnetisation limit of docs/MATHEMATICS.md 8a. Luma at 3.9 MHz
    reads 0.237 um, the colour-under 1.467 um: Ethan's 0.2 and 1.5.
    """
    return wavelength_m(frequency_hz, writing_speed_m_s) / (2.0 * np.pi)


def tape_per_sample_m(writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                      sample_rate_hz: float = OUTPUT_RATE_HZ) -> float:
    """Metres of track one output sample spans: 0.405 um at 4 fsc and 5.80 m/s."""
    return float(writing_speed_m_s) / float(sample_rate_hz)


# --------------------------------------------------------------------------
# the two terms of the split
# --------------------------------------------------------------------------

def shared_separation_nepers(frequency_hz, separation_m: float,
                             writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> np.ndarray:
    """Wallace's law, -2 pi d / lambda: THE SHARED TERM. One d for every band,
    so two bands' deviations are in the exact ratio of their frequencies
    (colour_under.wavelength_sharing_control)."""
    return -2.0 * np.pi * float(separation_m) / wavelength_m(frequency_hz, writing_speed_m_s)


def thickness_nepers(x) -> np.ndarray:
    """log[(1 - e^-x) / x], the loss of a layer magnetised uniformly to a
    depth delta read at wavelength lambda, x = 2 pi delta / lambda."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 1e-9, np.log((1.0 - np.exp(-x)) / np.maximum(x, 1e-12)), 0.0)


def effective_depth_m(frequency_hz, depth_m: float,
                      writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                      capped: bool = True) -> np.ndarray:
    """The depth a band's coating term uses: the recorded depth, capped at
    lambda/(2 pi) where the cap binds (magnetic.recording_depth's law).
    `capped=False` is the whole-layer reading that gives Ethan's figures."""
    wanted = float(depth_m) * np.ones_like(_grid(frequency_hz))
    if not capped:
        return wanted
    return np.minimum(wanted, read_depth_m(frequency_hz, writing_speed_m_s))


def band_specific_depth_nepers(frequency_hz, depth_m: float,
                               writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                               capped: bool = True) -> np.ndarray:
    """THE BAND-SPECIFIC TERM: the thickness loss at the depth THIS band
    reads. Each band carries its own `depth_m`; the luma's is cap-limited
    at 0.24 um whatever the coating, the chroma's is measured at about
    1.2 um on the SLV-778HF (MEASURED above)."""
    depth = effective_depth_m(frequency_hz, depth_m, writing_speed_m_s, capped)
    return thickness_nepers(2.0 * np.pi * depth / wavelength_m(frequency_hz, writing_speed_m_s))


def gap_nepers(frequency_hz, gap_m: float = GAP_M,
               writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> np.ndarray:
    ratio = EFFECTIVE_GAP_FACTOR * float(gap_m) / wavelength_m(frequency_hz, writing_speed_m_s)
    return np.log(np.maximum(np.abs(np.sinc(ratio)), 1e-12))


def split_response(frequency_hz, separation_m: float, depth_m: float,
                   writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                   capped: bool = True, differentiation: bool = False,
                   gap_m: float = GAP_M) -> np.ndarray:
    """One band's log response in nepers: shared separation + its own depth
    term (+ gap, + the head's differentiation if asked). Complex-typed
    magnitude is the caller's business; this is the log magnitude."""
    grid = _grid(frequency_hz)
    total = (shared_separation_nepers(grid, separation_m, writing_speed_m_s)
             + band_specific_depth_nepers(grid, depth_m, writing_speed_m_s, capped)
             + gap_nepers(grid, gap_m, writing_speed_m_s))
    if differentiation:
        total = total + np.log(np.maximum(grid, 1.0))
    return total


def thickness_loss_db(depth_m: float, frequency_hz: float,
                      writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                      capped: bool = False) -> float:
    """The thickness loss in decibels at one frequency, uncapped by default
    because that is the reading Ethan's -26 / -10 dB come from."""
    return float(DB_PER_NEPER * band_specific_depth_nepers(
        [frequency_hz], depth_m, writing_speed_m_s, capped)[0])


def depth_for_loss_m(loss_db: float, frequency_hz: float,
                     writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> float:
    """The uniformly magnetised depth that gives `loss_db` at one frequency."""
    from scipy.optimize import brentq
    return float(brentq(lambda dl: thickness_loss_db(dl, frequency_hz, writing_speed_m_s) - loss_db,
                        1e-9, 1e-3))


def one_depth_for_both(luma_db: float = ETHAN_REFERENCE["thickness_loss_luma_db"],
                       chroma_db: float = ETHAN_REFERENCE["thickness_loss_chroma_db"],
                       luma_hz: float = FM_PEAK_HZ, chroma_hz: float = COLOUR_UNDER_HZ,
                       writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> Dict[str, float]:
    """Does ONE depth reproduce both of Ethan's figures? Yes: 4.37 um, with
    residuals +0.7 and +0.1 dB. The figures are mutually consistent under
    a single whole-coating depth of the size the literature gives."""
    from scipy.optimize import minimize_scalar
    cost = lambda dl: ((thickness_loss_db(dl, luma_hz, writing_speed_m_s) - luma_db) ** 2
                       + (thickness_loss_db(dl, chroma_hz, writing_speed_m_s) - chroma_db) ** 2)
    best = float(minimize_scalar(cost, bounds=(1e-7, 2e-5), method="bounded").x)
    return {"depth_m": best,
            "luma_db": thickness_loss_db(best, luma_hz, writing_speed_m_s),
            "chroma_db": thickness_loss_db(best, chroma_hz, writing_speed_m_s),
            "luma_residual_db": thickness_loss_db(best, luma_hz, writing_speed_m_s) - luma_db,
            "chroma_residual_db": thickness_loss_db(best, chroma_hz, writing_speed_m_s) - chroma_db,
            "depth_from_luma_alone_m": depth_for_loss_m(luma_db, luma_hz, writing_speed_m_s),
            "depth_from_chroma_alone_m": depth_for_loss_m(chroma_db, chroma_hz, writing_speed_m_s)}


def reproduction(writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                 coating_m: float = ETHAN_REFERENCE["coating_thickness_um"] * 1e-6
                 ) -> Dict[str, Tuple[float, float]]:
    """Ethan's figure beside the specification's, as (his, reproduced)."""
    um = 1e6
    return {
        "tape_per_sample_um": (ETHAN_REFERENCE["tape_per_sample_um"], tape_per_sample_m(writing_speed_m_s) * um),
        "lambda_sync_tip_um": (ETHAN_REFERENCE["lambda_sync_tip_um"], float(wavelength_m(FM_SYNC_TIP_HZ, writing_speed_m_s)[0]) * um),
        "lambda_peak_white_um": (ETHAN_REFERENCE["lambda_peak_white_um"], float(wavelength_m(FM_PEAK_WHITE_HZ, writing_speed_m_s)[0]) * um),
        "lambda_chroma_um": (ETHAN_REFERENCE["lambda_chroma_um"], float(wavelength_m(COLOUR_UNDER_HZ, writing_speed_m_s)[0]) * um),
        "read_depth_luma_um": (ETHAN_REFERENCE["read_depth_luma_um"], float(read_depth_m(FM_PEAK_HZ, writing_speed_m_s)[0]) * um),
        "read_depth_chroma_um": (ETHAN_REFERENCE["read_depth_chroma_um"], float(read_depth_m(COLOUR_UNDER_HZ, writing_speed_m_s)[0]) * um),
        "thickness_loss_luma_db": (ETHAN_REFERENCE["thickness_loss_luma_db"], thickness_loss_db(coating_m, FM_PEAK_HZ, writing_speed_m_s)),
        "thickness_loss_chroma_db": (ETHAN_REFERENCE["thickness_loss_chroma_db"], thickness_loss_db(coating_m, COLOUR_UNDER_HZ, writing_speed_m_s)),
    }


# --------------------------------------------------------------------------
# the band mismatch: predicted against measured
# --------------------------------------------------------------------------

def predicted_band_mismatch_db(separation_m: float, luma_depth_m: float, chroma_depth_m: float,
                               writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                               luma_hz: float = FM_PEAK_HZ, chroma_hz: float = COLOUR_UNDER_HZ,
                               capped: bool = True) -> Dict[str, float]:
    """ratio(luma) - ratio(chroma) of the playback/record transfer, in dB,
    term by term. Measured +4.02 +- 0.33 dB at SP; differentiation alone
    +15.84."""
    def term(fn):
        return float(DB_PER_NEPER * (fn(luma_hz)[0] - fn(chroma_hz)[0]))
    differentiation = 20.0 * np.log10(luma_hz / chroma_hz)
    spacing = term(lambda f: shared_separation_nepers([f], separation_m, writing_speed_m_s))
    gap = term(lambda f: gap_nepers([f], GAP_M, writing_speed_m_s))
    thickness = float(DB_PER_NEPER * (
        band_specific_depth_nepers([luma_hz], luma_depth_m, writing_speed_m_s, capped)[0]
        - band_specific_depth_nepers([chroma_hz], chroma_depth_m, writing_speed_m_s, capped)[0]))
    return {"differentiation_db": differentiation, "shared_separation_db": spacing,
            "gap_db": gap, "band_specific_depth_db": thickness,
            "total_db": differentiation + spacing + gap + thickness}


def mismatch_verdict(measured_db: float = MEASURED["cross_band_level_db_sp"],
                     separation_m: float = MEASURED["luma_effective_spacing_um_sp"] * 1e-6,
                     writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> Dict[str, object]:
    """Each depth law against the measured cross-band level."""
    laws = {
        "whole coating 4.5 um": (4.5e-6, 4.5e-6, False),
        "chroma depth fitted on shape 1.17 um": (RECORD_DEPTH_M, 1.17e-6, True),
        "JVC record depth 0.30 um, capped": (RECORD_DEPTH_M, RECORD_DEPTH_M, True),
        "JVC record depth 0.30 um, uncapped": (RECORD_DEPTH_M, RECORD_DEPTH_M, False),
    }
    out = {}
    for name, (dl, dc, capped) in laws.items():
        p = predicted_band_mismatch_db(separation_m, dl, dc, writing_speed_m_s, capped=capped)
        out[name] = {"predicted_db": p["total_db"], "error_db": p["total_db"] - measured_db,
                     "thickness_share": p["band_specific_depth_db"]}
    return {"measured_db": measured_db, "laws": out,
            "why": ("the mismatch against the differentiator is real, 11.8 dB, and on this "
                    "deck it is mostly the shared separation; a whole-coating thickness "
                    "loss overshoots it by 11 dB")}


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def wallace_delay_difference_s(separation_m: float, luma_hz: float = FM_PEAK_HZ,
                               chroma_hz: float = COLOUR_UNDER_HZ,
                               writing_speed_m_s: float = WRITING_SPEED_M_S["SP"]) -> float:
    """The minimum-phase group-delay difference between two bands from a
    shared separation. log|H| = -(d/v)|w| has minimum-phase partner
    phi = (2d/(pi v)) w ln|w|, so tau = -(2d/(pi v))(ln w + 1) and the
    bands differ by -(2d/(pi v)) ln(f_l/f_c): -10.0 ns per 0.05 um, which
    a wide-grid Hilbert transform reproduces to 0.01 ns."""
    return float(-(2.0 * separation_m / (np.pi * writing_speed_m_s)) * np.log(luma_hz / chroma_hz))


def common_mode_timing(fixed_first, fixed_second, perfield_first, perfield_second,
                       burst_first, burst_second, sync_first, sync_second,
                       line_index, active=(40, 250), seconds_per_sample: float = 1.0 / OUTPUT_RATE_HZ
                       ) -> Dict[str, float]:
    """THE COMMON-MODE TEST on the sync-to-burst export's own arrays.

    Passes when the per-field disagreement between the two pilots is no
    larger than their own position residuals added in quadrature - two
    independent readings of ONE time base can disagree by no less and no
    more than that, so the criterion is the pilots' combined error with
    the rms's own sampling allowance, 3/sqrt(2n). It also reports the
    per-head fixed offset, which a shared separation would turn into
    10 ns per 0.05 um of per-head difference. Measured on bars SP:
    10.9/9.2 ns against a combined 25.9/23.8 (burst 11.4/10.3, sync
    23.2/21.5); per-head -2.0 +- 0.4 ns, bounding a per-head separation
    difference at 0.010 um; passes."""
    li = np.asarray(line_index); mask = (li >= active[0]) & (li < active[1])
    a = np.asarray(fixed_first, dtype=np.float64); b = np.asarray(fixed_second, dtype=np.float64)
    paired = mask & np.isfinite(a) & np.isfinite(b)
    ns = seconds_per_sample * 1e9
    per_first = np.asarray(perfield_first, dtype=np.float64)[:, mask]
    per_second = np.asarray(perfield_second, dtype=np.float64)[:, mask]
    dis = np.array([np.nanstd(per_first), np.nanstd(per_second)]) * ns
    count = min(int(np.isfinite(per_first).sum()), int(np.isfinite(per_second).sum()))
    burst = np.array([np.nanstd(burst_first), np.nanstd(burst_second)]) * ns
    sync = np.array([np.nanstd(sync_first), np.nanstd(sync_second)]) * ns
    combined = np.hypot(burst, sync)
    allowance = 1.0 + 3.0 / np.sqrt(max(2.0 * count, 1.0))
    diff = (a[paired] - b[paired]) * ns
    # seconds of per-head offset over seconds of delay per metre of separation
    per_metre = abs(wallace_delay_difference_s(1.0))
    return {
        "disagreement_rms_ns": dis, "burst_pilot_rms_ns": burst, "sync_pilot_rms_ns": sync,
        "combined_pilot_rms_ns": combined,
        "disagreement_over_combined": dis / np.maximum(combined, 1e-12),
        "per_head_offset_ns": float(diff.mean()),
        "per_head_offset_se_ns": float(diff.std() / np.sqrt(max(len(diff), 1))),
        "per_head_separation_bound_m": float(abs(diff.mean()) * 1e-9 / per_metre),
        "passes": bool(np.all(dis <= allowance * combined)),
        "why": "the two pilots' disagreement is at their own combined noise: one time base",
    }


# --------------------------------------------------------------------------
# controls that can fail
# --------------------------------------------------------------------------

def sharing_control(writing_speed_m_s: float = WRITING_SPEED_M_S["SP"], places: int = 256) -> Dict[str, float]:
    """With the depth term off, two bands' deviations from a shared
    separation must be in the ratio f_c / f_l to machine precision. Any
    depth term leaking into the shared one breaks this."""
    luma = np.linspace(FM_SYNC_TIP_HZ, FM_PEAK_WHITE_HZ, places)
    worst = 0.0
    for d in (0.02e-6, 0.05e-6, 0.5e-6):
        ratio = (shared_separation_nepers([COLOUR_UNDER_HZ], d, writing_speed_m_s)[0]
                 / shared_separation_nepers(luma, d, writing_speed_m_s))
        worst = max(worst, float(np.max(np.abs(ratio - COLOUR_UNDER_HZ / luma))))
    return {"worst_departure": worst, "passes": bool(worst < 1e-12)}


def small_x_control(depth_m: float = RECORD_DEPTH_M, writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                    places: int = 256) -> Dict[str, float]:
    """At x << 1 the thickness term is -x/2 = a Wallace spacing of delta/2,
    so on the chroma band at the JVC depth the two terms MUST be collinear;
    if they are not, the thickness term has been written wrongly. This is
    the identifiability ceiling of the split at small depth, kept as code."""
    grid = np.linspace(0.30e6, 1.00e6, places)
    t = band_specific_depth_nepers(grid, depth_m, writing_speed_m_s, capped=False)
    s = shared_separation_nepers(grid, depth_m / 2.0, writing_speed_m_s)
    t = t - t.mean(); s = s - s.mean()
    coherence = float(abs(t @ s) / (np.linalg.norm(t) * np.linalg.norm(s)))
    return {"coherence": coherence, "x_max": float(2 * np.pi * depth_m / wavelength_m(grid[-1], writing_speed_m_s)[0]),
            "passes": bool(coherence > 0.999)}


def identifiability_control(chroma_depth_m: float, separation_m: float = 0.05e-6,
                            writing_speed_m_s: float = WRITING_SPEED_M_S["SP"],
                            places: int = 64) -> Dict[str, float]:
    """The two-band Jacobian of (ln d, ln delta) with each band's gain
    unknown: singular values, effective count and condition. At delta 4.5
    um: 1.59 of 2 at condition 1.8; at the JVC 0.30 um: 1.01 of 2 at 14 to
    19. The split is determined only where the chroma band's depth is in
    its transition regime."""
    luma = np.linspace(1.2e6, 6.5e6, places); chroma = np.linspace(0.23e6, 1.03e6, places)
    def cols(grid):
        lam = wavelength_m(grid, writing_speed_m_s)
        x = 2 * np.pi * chroma_depth_m / lam
        cd = -2 * np.pi * separation_m / lam
        ct = x * np.exp(-x) / (1 - np.exp(-x)) - 1.0
        J = np.column_stack([cd, ct]); return J - J.mean(axis=0)
    J = np.vstack([cols(luma), cols(chroma)])
    s = np.linalg.svd(J, compute_uv=False); share = s ** 2 / (s ** 2).sum()
    return {"singular_values": s, "effective": float(1.0 / np.sum(share ** 2)),
            "condition": float(s[0] / max(s[-1], 1e-300))}
