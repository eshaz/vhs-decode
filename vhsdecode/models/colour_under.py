"""The colour-under channel, as components the transform can differentiate
over: the sibling that measures the LOW END of the same RF path.

VHS records the chrominance heterodyned down to a low carrier and lays it
under the frequency-modulated luminance on the same track, written by the same
head at the same instant. That sentence carries everything this module models,
and it carries one warning which turned out to be the module's main result.

WHAT THE COLOUR-UNDER IS FOR. The carrier sits at 629 kHz where the luma
carrier - 3.4 to 4.4 MHz, band-passed to 6.5 - never goes, and its band
reaches down to 59 kHz. COMPONENT_MAPPINGS section 3 established that
fractional bandwidth is the ONLY variable that moves how many mechanisms a
band can tell apart, writing speed and track width being inert, so a carrier
six times lower is the cheapest identifiability available. That is confirmed
here for the family the claim was made about, and CORRECTED for the family it
was not; see `band_extension`.

WHAT EARNS A PLACE IN THE KEY, AND WHAT DOES NOT. Four components were built.
Measured on the colour-under band, 59 kHz to 1.2 MHz, 1024 places:

    colour-under envelope phase             coherence 0.0000 with everything
    luma to colour-under transfer roll-off  0.2658 worst against the magnetics
    ---------------------------------------------------------------------
    colour-under envelope amplitude         0.99998 with spacing loss
    colour-under heterodyne offset          0.99999 with the amplitude

The first two are NEW KINDS and they are registered. Together they measure
2.000 effective directions of 2 at condition 1.000, with a mutual coherence of
5.7e-18 - a perfectly orthogonal pair, against the tape magnetics' 1.58 of 6
at condition 1.06e4.

The last two are NOT new kinds and they are NOT registered. Once the deck's
playback equalisation is taken off - which is what the decoder's envelope
actually shows - the colour-under's amplitude response over its own band is
0.99998 coherent with Wallace's spacing loss and 0.99993 with thickness loss.
It is not a mechanism the key is missing; it is the family that already
collapsed to 1.58 of 6, read at a wavelength six times longer. Entered anyway
it moves the magnetics from 1.4978 to 1.4157 effective of 7 - it LOWERS the
count, which
is the S-VHS sub pre-emphasis lesson of COMPONENT_MAPPINGS section 8a arriving
again: a second copy of a direction already in the span only worsens the
conditioning.

That is not a disappointment, and stating it as one would misread what a
second carrier is worth. The colour-under's amplitude response is a
CONSTRAINT on parameters the key already has, not a direction the key lacks -
it measures the same spacing, gap, thickness and azimuth at a wavelength of
9.216 micrometres instead of 1.487, where the total head loss is 0.1036 nepers
instead of 0.6880. Constraining a parameter and adding a dimension are
different services and the participation ratio only measures the second.

WHY THE PHASE IS A DELAY AND NOT THE AMPLITUDE'S CAUSAL PARTNER. Because the
causal partner was tried in this repository's runtime and FALSIFIED: the
measured quadrature transfer is near-flat through 0.6 MHz rather than
Hilbert-shaped, and the rotation the causal form produced left the measured
phase error untouched while leaking into amplitude. The record of that sits in
`vhsdecode/chroma.py` beside the estimator that replaced it. A near-flat
quadrature transfer is what a DELAY looks like, and a delay is parameterised
on the axis conjugate to frequency - which is why it is orthogonal to every
loss rather than nearly parallel to them.

TWO ESTABLISHED RESULTS, ONE REPRODUCED AND ONE STRENGTHENED. The
luma-to-colour-under amplitude transfer halves by about 150 kHz of modulation
rate, measured in this repository on both tape speeds and both test patterns,
and it is the half-point that `transfer_rolloff` carries. And no PER-HEAD
AMOUNT difference survives proper pooling - which `per_head_amount` reproduces
with its mechanism, an amount being a scale on one direction and a family
differing only in a scale being exactly one direction, and then strengthens:
even a per-head HALF-POINT, which is a shape and not a scale, is only 1.011
effective of 2 at 150 against 300 kHz and 1.048 at 150 against 1200. So the
per-head distinction the runtime keeps is real but it is not a direction, and
the pooling result is the correct one for a stronger reason than it was made.

WHAT THE BAND EXTENSION BUYS, AND THE RULE IT FORCES. With the grid density
held fixed so that only the band changes, the luma band being 1.2 to 6.5 MHz
and the whole band adding 58.7 kHz to 1.2 MHz below it:

    the tape's six magnetic mechanisms
        luma band alone                1.4419 of 6,  condition 4.65e+05
        plus the colour-under band     1.5528 of 6,  condition 1.98e+05
        and the two registered entries 2.3065 of 8,  condition 2.43e+05

    the modelled interference set
        luma band alone                7.6863 of 14, condition 10.6
        plus the colour-under band     7.1953 of 14, condition 13.5
        and the two registered entries 7.2993 of 16, condition 22.2

The extension buys the magnetics 0.111 of a direction and improves their
conditioning 2.3 times, reproducing the U-matic reading of COMPONENT_MAPPINGS
section 3 - 1.514 to 1.551 for the same move, at no cost - on a different
format; the two entries then buy 0.754 more. But the same move COSTS the
localised interference set 0.491, because a beat, a dropout, a sound trap and
a set of echoes are all featureless below the chroma ceiling and entries that
say nothing in the same place are more alike than they were.

So: A BAND EARNS ITS PLACE BY THE DIRECTIONS IT DISTINGUISHES, which is the
modification rule of COMPONENT_MAPPINGS section 8a applied to the abscissa
instead of to the key. "Fit over the widest band the RF occupies" is right for
the head model, whose mechanisms are parameterised across the whole band, and
it is not general.

THE RULING THAT GOVERNS THE HETERODYNE, kept as a control. A correction
expressed as a COMPLEX GAIN PER FREQUENCY crosses the mix with no scale factor
at all. `heterodyne_control` establishes that by actually performing the mix -
a tone at `f_cu + m`, the decoder's own oscillator, and a projection at the
output - and reads an amplitude slope of 1.000000 and a phase slope of
-1.000000, the sign being the band's mirror and not a scale. Applying
`heterodyne_timing_scale`'s 5.6875 to a gain, which is the mistake the ruling
exists to prevent, would read -5.6875 there.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from vhsdecode.models import head_model


# --------------------------------------------------------------------------
# The specification. Every figure below is either printed in a standard, or
# derived from a printed relation and marked so, or measured in this
# repository with the measurement named. Nothing is assumed silently.
# --------------------------------------------------------------------------

# SMPTE 170M: the M/NTSC line rate is 525 lines at 30/1.001 fields per second.
NTSC_LINE_RATE_HZ = 525.0 * (30.0 / 1.001)          # 15734.2657 Hz
# ITU-R BT.1700: the 625/50 line rate is exact.
PAL_LINE_RATE_HZ = 625.0 * 25.0                     # 15625 Hz

# SMPTE 170M fixes the NTSC colour subcarrier at 455/2 times the line rate;
# BT.1700 prints the PAL subcarrier directly.
NTSC_SUBCARRIER_HZ = (455.0 / 2.0) * NTSC_LINE_RATE_HZ    # 3579545.4545 Hz
PAL_SUBCARRIER_HZ = 4433618.75                            # printed exactly

# THE COLOUR-UNDER CARRIER, and the provenance of each figure.
#
# The specification states the carrier as a multiple of the horizontal
# scanning rate: forty times it for NTSC, and forty times it plus 1.953 kHz
# for PAL, the offset being what makes the PAL line-to-line phase relation
# come out right. For PAL the resulting figure is PRINTED - 626.953 kHz - and
# can be quoted. For NTSC it is not printed: it is DERIVED from the stated
# relation, and 629370.63 Hz is that derivation rather than a quotation. The
# distinction is kept in `carrier_provenance` rather than in a comment,
# because a derived figure inherits the tolerance of everything it came from
# and a printed one carries the standard's own.
COLOUR_UNDER_LINE_MULTIPLE = 40.0
PAL_COLOUR_UNDER_OFFSET_HZ = 1953.0
PAL_COLOUR_UNDER_PRINTED_HZ = 626953.0

# How far the colour-under's band reaches. These are the DECODER's own
# band-pass ceilings, `chroma_bpf_upper` in `vhsdecode/format_defs/vhs.py`:
# 1.2 MHz on NTSC and 1.3 MHz on PAL. They are FILTER SETTINGS and not
# figures of the format, and `sideband_fold_headroom_hz` shows the difference
# mattering - PAL's ceiling reaches further below the carrier than the
# carrier itself, which is a statement about the band-pass and not about the
# signal. That is why every function below takes the ceiling as an argument
# rather than reading a constant.
CHROMA_BAND_UPPER_HZ: Dict[str, float] = {"NTSC": 1.2e6, "PAL": 1.3e6}

# MEASURED IN THIS REPOSITORY, not assumed. `vhsdecode/chroma.py` states it
# from the transfer regression: the share of the luma's amplitude deviation
# worth carrying across to the colour-under "is the whole of the modelled
# transfer at a few tens of kHz and about half of it by 150 kHz, on both
# speeds and both test patterns". The half-point is the parameter; the SHAPE
# is explicitly NOT determined - the same file records that no single pole
# order fits it, chi-squared per degree of freedom running 2 to 24 for every
# order tried - which is why the runtime measures it per field band by band.
TRANSFER_HALF_HZ = 150e3

# MEASURED IN THIS REPOSITORY: `WAVELENGTH_INDEPENDENT_NOISE` in
# `vhsdecode/chroma.py`. The share of the luma's amplitude deviation that does
# NOT scale with wavelength, estimated there alongside the trust weight.
# Wallace's law alone would make it zero; it measures 0.194, and what the
# remainder is has not been established - a gain wobble in the record
# amplifier and a surface variation slower than the read volume would both do
# it. Labelled measured-but-unattributed rather than given a mechanism it has
# not earned.
WAVELENGTH_INDEPENDENT = 0.194

# An oscillator error of exactly one line rate. The colour-under local
# oscillator is a LINE-RATE MULTIPLE by specification, so a counter that lands
# on the neighbouring multiple is off by exactly one line frequency; that is
# the natural quantum rather than a chosen size. It is a SCALE and not a
# shape - `heterodyne_offset` is the response's derivative to first order, so
# the normalised signature does not depend on it. Measured over offsets from
# a tenth of a line rate to ten line rates, the normalised shapes agree to
# 3.5e-08 in the worst case and to 2.9e-10 at one line rate.
OFFSET_QUANTUM_LINES = 1.0


def _grid(frequency_hz) -> np.ndarray:
    return np.asarray(frequency_hz, dtype=np.float64).ravel()


def line_rate_hz(system: str = "NTSC") -> float:
    """The horizontal scanning rate the colour-under carrier is a multiple
    of."""
    name = str(system).upper()
    if name in ("NTSC", "M", "M/NTSC", "525"):
        return NTSC_LINE_RATE_HZ
    if name in ("PAL", "625", "B", "G", "I"):
        return PAL_LINE_RATE_HZ
    raise ValueError(f"no line rate declared for system {system!r}")


def subcarrier_hz(system: str = "NTSC") -> float:
    """The colour subcarrier the colour-under was heterodyned down from."""
    name = str(system).upper()
    if name in ("NTSC", "M", "M/NTSC", "525"):
        return NTSC_SUBCARRIER_HZ
    if name in ("PAL", "625", "B", "G", "I"):
        return PAL_SUBCARRIER_HZ
    raise ValueError(f"no subcarrier declared for system {system!r}")


def carrier_hz(system: str = "NTSC") -> float:
    """The colour-under carrier: forty times the line rate, plus the PAL
    offset where the system has one.

    NTSC gives 629370.63 Hz and PAL 626953.00 Hz, the latter agreeing with the
    printed 626.953 kHz exactly.
    """
    rate = line_rate_hz(system)
    offset = (PAL_COLOUR_UNDER_OFFSET_HZ if rate == PAL_LINE_RATE_HZ else 0.0)
    return COLOUR_UNDER_LINE_MULTIPLE * rate + offset


def carrier_provenance(system: str = "NTSC") -> Dict[str, object]:
    """Where the carrier figure comes from, and whether it is PRINTED.

    The PAL figure is printed as 626.953 kHz and the derivation reproduces it
    to 0.00 Hz; the NTSC figure is not printed anywhere in the specification
    and is marked derived. The distinction is not pedantry - a derived figure
    inherits the tolerance of the line rate it came from, and the whole method
    depends on knowing which of its numbers a standard fixed and which it
    merely implies.
    """
    value = carrier_hz(system)
    pal = line_rate_hz(system) == PAL_LINE_RATE_HZ
    return {
        "system": str(system).upper(),
        "carrier_hz": float(value),
        "printed": bool(pal),
        "printed_hz": (float(PAL_COLOUR_UNDER_PRINTED_HZ) if pal else None),
        "agreement_hz": (abs(value - PAL_COLOUR_UNDER_PRINTED_HZ)
                         if pal else None),
        "relation": ("40 f_H + 1953 Hz" if pal else "40 f_H"),
        "line_rate_hz": float(line_rate_hz(system)),
        "why": ("the standard states the carrier as a multiple of the line "
                "rate; PAL's resulting figure is printed and NTSC's is not, "
                "so the NTSC value is a derivation and is marked as one"),
    }


def band_upper_for(system: str = "NTSC",
                   band_upper_hz: Optional[float] = None) -> float:
    """The colour-under band-pass ceiling for a system, or the caller's.

    The decoder's own `chroma_bpf_upper`, which differs between the systems -
    1.2 MHz on NTSC, 1.3 MHz on PAL - and is a setting rather than a
    specification, so a caller with a measurement of its own overrides it.
    """
    if band_upper_hz is not None:
        return float(band_upper_hz)
    name = str(system).upper()
    if name in ("NTSC", "M", "M/NTSC", "525"):
        return CHROMA_BAND_UPPER_HZ["NTSC"]
    if name in ("PAL", "625", "B", "G", "I"):
        return CHROMA_BAND_UPPER_HZ["PAL"]
    raise ValueError(f"no chroma band-pass ceiling declared for {system!r}")


def modulation_half_width_hz(system: str = "NTSC",
                             band_upper_hz: Optional[float] = None) -> float:
    """How far either side of the carrier the colour-under's band-pass
    reaches: 570.63 kHz on NTSC and 673.05 kHz on PAL."""
    return band_upper_for(system, band_upper_hz) - carrier_hz(system)


def sideband_fold_headroom_hz(system: str = "NTSC",
                              band_upper_hz: Optional[float] = None
                              ) -> Dict[str, object]:
    """HOW CLOSE THE LOWER SIDEBAND COMES TO FOLDING THROUGH ZERO.

    The colour-under is a double-sideband carrier, so modulation faster than
    the carrier frequency itself folds through zero and returns as its own
    conjugate. That is a real mechanism rather than a modelling nicety - it is
    why the format cannot simply put the carrier lower to buy more separation
    from the luma - and it is the one thing that would stop the colour-under
    band from being a faithful image of the chroma band.

    Measured against the decoder's own band-pass ceilings, and the two systems
    do NOT agree:

        NTSC   carrier 629.37 kHz, ceiling 1.2 MHz    headroom  +58.74 kHz
        PAL    carrier 626.95 kHz, ceiling 1.3 MHz    headroom  -46.09 kHz

    THE PAL FIGURE IS A STATEMENT ABOUT THE BAND-PASS, NOT ABOUT THE SIGNAL,
    and reading it as a fold would be the mistake this function exists to make
    visible. `chroma_bpf_upper` is a filter setting that separates the
    colour-under from the luma; it is deliberately wider than the recorded
    chroma, and on PAL it is wider than the carrier, so its notional lower
    edge lies 46 kHz below zero. What actually folds is decided by how far the
    RECORDED modulation reaches, which no figure in this repository states -
    pass `band_upper_hz` from a measurement to answer that. On NTSC the
    question does not arise: even the generous ceiling clears zero.

    So `folds` reports whether the band as given would fold, and
    `from_the_band_pass` says whether the band as given was a filter setting
    rather than a measurement, which is the qualification a reader needs.
    """
    carrier = carrier_hz(system)
    half = modulation_half_width_hz(system, band_upper_hz)
    return {
        "carrier_hz": float(carrier),
        "half_width_hz": float(half),
        "lowest_hz": float(carrier - half),
        "headroom_hz": float(carrier - half),
        "folds": bool(half > carrier),
        "from_the_band_pass": bool(band_upper_hz is None),
        "band_upper_hz": band_upper_for(system, band_upper_hz),
        "why": ("modulation faster than the carrier folds through zero and "
                "returns as its own conjugate; NTSC's band-pass clears it by "
                "58.74 kHz, PAL's does not clear it at all - but PAL's is a "
                "filter setting wider than the recorded chroma, so the "
                "measured extent decides"),
    }


def heterodyne_local_oscillator_hz(system: str = "NTSC") -> float:
    """The heterodyne's local oscillator: subcarrier plus colour-under
    carrier, 4208916.08 Hz on NTSC.

    The decoder's own figure and not a choice made here.
    `chroma.upconvert_chroma_phase_comp` advances its oscillator by
    `(pi/2)(1 + f_cu/f_sc)` per sample on a grid of four times the subcarrier,
    which is `2 pi f_LO / (4 f_sc)` with `f_LO = f_sc + f_cu`.
    """
    return subcarrier_hz(system) + carrier_hz(system)


def up_converted_hz(colour_under_hz, system: str = "NTSC") -> np.ndarray:
    """Where each colour-under place lands after the mix: `f_LO - f`.

    THE BAND IS MIRRORED, and this is arithmetic rather than an opinion. The
    up-conversion multiplies by a real oscillator at `f_LO = f_sc + f_cu` and
    keeps the difference product, so a component at `f_cu + m` lands at
    `f_LO - (f_cu + m) = f_sc - m`: the modulation offset changes sign and the
    phase is conjugated. The colour-under's LOWER sideband therefore carries
    the chroma's UPPER one.

    Nothing is SCALED by that, which is what `heterodyne_control` establishes
    by performing the mix: the magnitude crosses untouched and the phase
    crosses with unit slope. A correction carried as a complex gain per
    frequency needs the mirror and needs no scale factor at all, and the
    5.6875 ratio of `information_extrapolation.heterodyne_timing_scale`
    belongs only where a phase is read as a TIME.
    """
    return heterodyne_local_oscillator_hz(system) - _grid(colour_under_hz)


# --------------------------------------------------------------------------
# The four components. Two of them earn a place in the key and two do not,
# and each says which in its own docstring.
# --------------------------------------------------------------------------


def _head_log(frequency_hz: np.ndarray, writing_speed_m_s: float,
              track_width_m: float, equalised: bool,
              parameters: Dict[str, float]) -> np.ndarray:
    """The head's log response, with the deck's playback equalisation
    optionally taken off.

    `head_model.log_response` includes the head's own differentiation, a
    `log f` term, because a head reads the rate of change of flux and that is
    physically what it does. The DECK equalises it, so what the decoder's
    envelope shows - and therefore what any measurement in this project can
    compare against - is the loss stack without it. Over the colour-under band
    the difference is not cosmetic: `log f` runs 3.0 nepers across 59 kHz to
    1.2 MHz while every loss together runs 0.10, so leaving it in makes the
    signature a measurement of the equaliser rather than of the tape.
    """
    logged = head_model.log_response(frequency_hz, float(writing_speed_m_s),
                                     float(track_width_m), **parameters)
    if equalised:
        logged = logged - np.log(frequency_hz)
    return logged


def envelope_amplitude(frequency_hz, writing_speed_m_s: float = 5.8,
                       track_width_m: float = 58e-6,
                       equalised: bool = True, **head: float) -> np.ndarray:
    """THE CHROMA ENVELOPE'S AMPLITUDE RESPONSE, as the head's own loss stack
    read at colour-under wavelengths.

    Nothing new is invented here, and that is the finding rather than a
    shortcut. The colour-under went through the same head and the same coating
    as the luma carrier, so its amplitude response is `head_model.log_response`
    evaluated at 629 kHz instead of at 3.9 MHz. At the NTSC SP writing speed
    of 5.80 m/s - the figure JVC VTG82063 states directly, not the 5.877 the
    drum's circumference implies - the wavelengths are 9.216 and 1.487
    micrometres, and every loss shrinks in that ratio:

        spacing loss, 0.05 um       0.0341 nepers against 0.2112
        gap loss, 0.30 um           0.0022 against 0.0839
        thickness loss, 0.20 um     0.0674 against 0.3929
        all of them together        0.1036 against 0.6880

    THIS ENTRY DOES NOT EARN A PLACE IN THE KEY, and the measurement says so
    without room for argument. Over the colour-under band its log-shape is
    0.99998 coherent with Wallace's spacing loss and 0.99993 with thickness
    loss: it is not a mechanism the key is missing but the family that already
    collapsed to 1.58 of 6, read at a longer wavelength. Added to the six
    magnetic mechanisms it moves them from 1.4978 effective of 6 to 1.4157 of
    7 - it LOWERS the count, exactly as the S-VHS sub pre-emphasis did when it
    was entered as two of its own curves.

    WHAT IT IS FOR IS DIFFERENT. A second measurement of the same parameters
    at a wavelength six times longer CONSTRAINS them; it does not add a
    direction, and the participation ratio measures only the second service.
    The function is kept, and exercised by `wavelength_sharing_control`,
    because the constraint is the whole reason a second carrier is worth
    having - see `band_extension`, where the six mechanisms go from 1.4419 of
    6 at condition 4.65e+05 on the luma band to 1.5528 at 1.98e+05 once the
    colour-under band is included.

    Returned as a bounded positive magnitude with the mean removed, so that
    `interference.subtractable` accepts it and a level is not mistaken for a
    shape.
    """
    grid = _grid(frequency_hz)
    if np.any(grid <= 0.0):
        raise ValueError("a head has no response at or below zero frequency")
    parameters = dict(head_model.TYPICAL)
    parameters.update(head)
    logged = _head_log(grid, writing_speed_m_s, track_width_m, equalised,
                       parameters)
    return np.exp(logged - float(np.mean(logged))).astype(np.complex128)


def envelope_phase(frequency_hz, delay_s: float,
                   system: str = "NTSC") -> np.ndarray:
    """THE CHROMA ENVELOPE'S PHASE, as a DELAY across the colour-under band.
    REGISTERED.

    A delay and not the minimum-phase partner of the amplitude, and that is a
    measured correction rather than a preference. The causal-partner form -
    phase as the Hilbert transform of the log amplitude - stood in this
    repository's runtime and was FALSIFIED by the chroma-side transfer
    instrument: the measured quadrature transfer is near-flat through 0.6 MHz
    rather than Hilbert-shaped, and the rotation the causal form produced left
    the measured phase error untouched while leaking into amplitude. The note
    recording that sits in `vhsdecode/chroma.py` beside the estimator that
    replaced it, which reads +0.007/-0.008/+0.009 across the lower bands
    against an uncorrected +0.036/+0.041/+0.043.

    A near-flat quadrature transfer across the band is what a DELAY looks
    like, and the physics agrees about where it comes from. The head's own
    phase here is the minimum-phase partner of an amplitude tilt of 0.10
    nepers, which is negligible; the record-side and playback-side chroma
    band-pass filters are 1.14 MHz wide about a 629 kHz carrier and their
    group delay is not. So the modelled source is the recorder's chroma path,
    and that is where the chain position places it.

    WHY IT EARNS ITS PLACE. Delay is the axis conjugate to frequency, and a
    family parameterised along it is a Fourier basis - which is why eight
    echoes measure 7.92 of 8 where six tape losses measure 1.58 of 6. Measured
    here, this signature's coherence with every other entry in this module and
    with all six magnetic mechanisms is 0.000 to five decimals. It is the
    single cleanest direction the colour-under supplies.

    Referred to the carrier, so the constant phase the burst lock removes is
    not part of the signature. Unit magnitude, therefore trivially
    subtractable, and PURE PHASE - which is why the fit must use
    `real_parameters=True`: under the Hermitian inner product a delay and a
    gain of the same shape are one direction, and here they are two
    mechanisms.
    """
    grid = _grid(frequency_hz)
    return np.exp(-2j * np.pi * (grid - carrier_hz(system)) * float(delay_s))


def heterodyne_offset(frequency_hz, offset_hz: Optional[float] = None,
                      system: str = "NTSC", writing_speed_m_s: float = 5.8,
                      track_width_m: float = 58e-6,
                      equalised: bool = True,
                      delay_s: float = 0.0, **head: float) -> np.ndarray:
    """THE HETERODYNE OFFSET: the channel's response, read at the wrong place.

    The recorder mixes the chroma down with an oscillator locked to its own
    line rate. If that oscillator is in error by `d` the chroma is written at
    `f + d` where the model says `f`, the channel acts on it there, and the
    playback up-conversion - locked to the PLAYER's line rate - brings it back
    to `f`. What survives is the response evaluated at the wrong frequency:

        E(f) = H(f + d) / H(f) = exp( d * d(log H)/df ) + O(d^2)

    so the signature is the response's own DERIVATIVE. That is the structure
    `vhsdecode/models/magnetic.py` found for record level, and it carries the
    same warning, which here turns out to be fatal.

    THIS ENTRY DOES NOT EARN A PLACE IN THE KEY. On a channel whose log
    response is LINEAR in frequency the derivative is a constant, the
    signature is a flat gain, and the offset adds no direction whatever.
    Wallace's spacing loss is exactly `-2 pi d f / v`, linear, and measured on
    a pure Wallace channel this signature's log-shape has a norm of 1.8e-15
    nepers, which is machine zero. On the head's whole equalised loss stack it
    reaches 1.2e-03, and its coherence with `envelope_amplitude` on the
    colour-under band is 0.99999. Added to the six magnetic mechanisms it
    moves them from 1.4978 to 1.4158 of 7, the same way the amplitude does and
    for the same reason - it IS the amplitude, differentiated.

    WHERE THE OFFSET IS IDENTIFIABLE IS THE TIME AXIS, not this one. The
    burst lock nulls the error by moving line positions, so it never survives
    AS a residual phase, and `information_extrapolation.chroma_components`
    already records the right instrument: read it from what the lock had to
    MOVE, a per-line period deviation, where the line-locked part and the
    wobbling part separate. This function exists so that the frequency-axis
    reading is on record as inert rather than assumed to be useful.

    THE ONE PLACE IT IS NOT INERT is the unequalised response, where the
    head's own differentiation contributes `1/f` - 1.589e-06 per hertz at the
    colour-under carrier against 2.564e-07 at the luma carrier, SIX TIMES
    LARGER, which is a real statement about where a slope is visible. With
    `equalised=False` the log-shape norm rises to 1.243, but that is the deck's
    equaliser being measured and not the tape.

    `offset_hz` defaults to exactly one line rate, the oscillator being a
    line-rate multiple by specification so that a counter landing on the
    neighbouring multiple is off by exactly that. It is a SCALE and not a
    shape: the normalised signature agrees with itself to 3.5e-08 over two
    decades of offsets, from a tenth of a line rate to ten.

    THE CONFOUND, stated rather than hidden: the record-side oscillator error
    and the playback-side one produce the same frequency-domain signature and
    a single decode cannot separate them. Only their DIFFERENCE reaches the
    burst; their common part does not reach it at all. What separates them is
    a second decode of the same tape on a different player, where the
    playback-side error changes and the record-side one does not.
    """
    grid = _grid(frequency_hz)
    if offset_hz is None:
        offset_hz = OFFSET_QUANTUM_LINES * line_rate_hz(system)
    offset = float(offset_hz)
    if np.any(grid + offset <= 0.0) or np.any(grid <= 0.0):
        raise ValueError("the offset moves the band through zero frequency")
    parameters = dict(head_model.TYPICAL)
    parameters.update(head)
    here = _head_log(grid, writing_speed_m_s, track_width_m, equalised,
                     parameters)
    there = _head_log(grid + offset, writing_speed_m_s, track_width_m,
                      equalised, parameters)
    logged = there - here
    logged = logged - float(np.mean(logged))
    signature = np.exp(logged).astype(np.complex128)
    if delay_s:
        # a delay in the chroma path turns the same offset into a CONSTANT
        # phase, which is a level rather than a shape and goes with the mean
        signature = signature * np.exp(-2j * np.pi * offset * float(delay_s))
    return signature


def transfer_rolloff(frequency_hz, system: str = "NTSC",
                     half_hz: float = TRANSFER_HALF_HZ,
                     band_upper_hz: Optional[float] = None) -> np.ndarray:
    """THE LUMA-TO-COLOUR-UNDER TRANSFER'S ROLL-OFF WITH MODULATION RATE.
    REGISTERED.

    The luma carrier and the colour-under were written by the same head at the
    same instant, so head-to-medium separation loss is common to both and the
    luma envelope's departure from constant amplitude predicts the
    colour-under's. But only the SLOW part of it does. Separation is
    mechanical and therefore slow; the luma envelope's own measurement noise
    is broadband; so the share worth carrying across is a Wiener weight in
    modulation rate,

        W(m) = S(m) / (S(m) + N)  =  1 / (1 + (m / m_half)^2)

    with `m` the offset from the colour-under carrier. The Lorentzian is the
    weight a one-pole mechanical spectrum against a flat noise floor produces,
    written that way rather than fitted: the ONE parameter is the half-point,
    and the weight is one half there by construction.

    THE HALF-POINT IS MEASURED, NOT ASSUMED - about 150 kHz, from this
    repository's own transfer regression on both tape speeds and both test
    patterns. The same measurement is explicit that the SHAPE is not
    determined, no single pole order fitting better than chi-squared per
    degree of freedom of 2 to 24, which is why the runtime measures it per
    field band by band. So the Lorentzian is a one-parameter stand-in whose
    half-point is measured and whose curvature is not, and a fit should read
    nothing into its tails.

    Beyond the colour-under's own modulation half-width the weight is held at
    its edge value, because outside that band there is no colour-under for the
    transfer to be a transfer TO. That keeps the shape where the mechanism is
    instead of letting a meaningless tail four megahertz away dominate a
    normalised fit, and it keeps the entry subtractable - the Lorentzian never
    reaching zero, its smallest value on the colour-under band being 0.0646.

    WHY IT EARNS ITS PLACE, AND IT IS THE ONLY ENTRY HERE THAT IS NEW IN
    KIND WITHOUT BEING NEW IN AXIS. Every magnetic loss is monotone across the
    whole band; this one is PEAKED at the colour-under carrier and flat
    outside it. Measured against the six magnetic mechanisms its worst
    coherence is 0.2658 - against the head's own differentiation, not
    against a loss - and added to them it moves 1.4978 of 6 to 1.9320 of 7, a
    gain of 0.4342 of a direction, where the amplitude and the offset each
    lose 0.0821.
    """
    grid = _grid(frequency_hz)
    carrier = carrier_hz(system)
    edge = modulation_half_width_hz(system, band_upper_hz)
    modulation = np.minimum(np.abs(grid - carrier), max(edge, 0.0))
    return (1.0 / (1.0 + (modulation / max(float(half_hz), 1.0)) ** 2)
            ).astype(np.complex128)


def wavelength_sharing(frequency_hz, system: str = "NTSC",
                       wavelength_independent: float = WAVELENGTH_INDEPENDENT
                       ) -> np.ndarray:
    """THE SHARE OF THE LUMA'S LOSS THE COLOUR-UNDER TOOK, against frequency.

    Wallace's spacing loss is `exp(-2 pi d / lambda)` and `lambda = v / f`, so
    the loss in NEPERS is `2 pi d f / v`, proportional to the frequency. Two
    carriers written at the same instant through the same separation therefore
    took losses in the exact ratio of their frequencies. Reading the luma
    carrier at `f` and the colour-under at `f_cu`, the fraction of the luma's
    log-deviation that belongs on the colour-under is `f_cu / f` - a
    prediction with NO FREE PARAMETER, and the one relation in this module
    that a construction error would silently break, which is why
    `wavelength_sharing_control` checks it to machine precision.

    It is not the whole story, and this repository measured the remainder
    rather than assuming it away. `WAVELENGTH_INDEPENDENT_NOISE` in
    `vhsdecode/chroma.py` is 0.194: about a fifth of the luma envelope's
    departure does not scale with wavelength at all, and the runtime therefore
    carries `w + (1 - w) f_cu / f`, which is the law reproduced here.

    THIS ENTRY DOES NOT EARN A PLACE IN THE KEY EITHER. On the luma band its
    log-shape is 0.948 coherent with Wallace's spacing loss and on the
    colour-under band 0.927; against the six magnetic mechanisms its worst
    coherence is 0.9988, and adding it to them moves 1.4978 to 1.4762. It is
    the same monotone decay in the same dimensionless group. Kept as a
    function because the runtime's law needs a
    closed form to be checked against, and because it is the relation the
    whole shared-loss argument rests on.
    """
    grid = _grid(frequency_hz)
    share = float(wavelength_independent)
    ratio = carrier_hz(system) / np.maximum(grid, 1.0)
    return (share + (1.0 - share) * ratio).astype(np.complex128)


# --------------------------------------------------------------------------
# Where each entry acts in the chain
# --------------------------------------------------------------------------

# Position is the stage the modification happens AT, counting from the source,
# and `interference.ordered_key` returns entries last-applied-first-removed,
# which is descending position. The existing chain places transmission at 1 to
# 4, the recording machine's own processing at 10 and 11, the tape and head at
# 20 to 23, and the capture at 30.
#
#   10  the recording machine, before the record head. The chroma is split
#       off, down-heterodyned and band-limited HERE, in parallel with the
#       luma's sub pre-emphasis which already sits at 10. The two act on
#       disjoint bands of disjoint paths and commute, which is exactly the
#       case `interference.COMPONENT_ORDER`'s own comment allows a shared
#       position for.
#   20  the head-to-medium interface, alongside head contact tilt. The
#       colour-under's amplitude response and the transfer's roll-off are both
#       properties of the separation, which is where the loss the two channels
#       SHARE is imposed - the roll-off being the rate at which that
#       separation can change, which is mechanical.
#
# All four are declared, because `ordered_key` refuses an entry without a
# position and a component that might be entered later must have one. Only
# the two that `signatures` returns are recommended for registration; the
# other two are measured to add no direction and their docstrings say so.
COMPONENT_ORDER: Dict[str, int] = {
    "colour-under heterodyne offset": 10,
    "colour-under envelope phase": 10,
    "colour-under envelope amplitude": 20,
    "luma to colour-under transfer roll-off": 20,
    # The chroma carrier that survives in the LUMA. It sits at 20 with the
    # other shared-path entries because the coupling is AM and an amplitude
    # coupling needs a stage the two channels are in together - which on VHS
    # is the record amplifier, the head and the medium, not the band-split
    # at 10 where they are already disjoint.
    "residual chroma carrier in luma": 20,
}

# The entries `signatures` returns, in the order they are applied.
REGISTERED: Tuple[str, ...] = ("colour-under envelope phase",
                               "luma to colour-under transfer roll-off",
                               "residual chroma carrier in luma")

# What the shipped --luma_beat correction removes, measured: 87 to 89 per
# cent of the beat, by a field-fitted tap under a burst-presence gate. The
# residual entry carries what is LEFT, so the default is the worse of the
# two measured figures - erring low on a removal means erring high on the
# residual, which is the safe direction for a correction.
LUMA_BEAT_REMOVED = 0.87


def signatures(frequency_hz, system: str = "NTSC",
               mechanics: Optional[Dict[str, float]] = None,
               band_upper_hz: Optional[float] = None,
               delay_s: Optional[float] = None) -> Dict[str, np.ndarray]:
    """The colour-under entries for the key, on one frequency grid.

    TWO ENTRIES AND NOT FOUR. The colour-under envelope's amplitude response
    and the heterodyne offset were built, measured, and found to be the
    collapsed magnetic family read at a longer wavelength - 0.99998 coherent
    with spacing loss and 0.99999 with each other - so they are not returned.
    Their functions remain, with the numbers that disqualified them in their
    docstrings, because the measurement is the result and deleting it would
    lose it. A modification earns its place by the direction it adds.

    RETURNED ONLY WHERE THE BAND CONTAINS THE CARRIER, which is the rule
    `interference.signatures` already applies to the record-level entry and
    for the same reason: a signature whose whole shape lies outside the grid
    is a featureless constant there, and entering it costs conditioning
    without adding a direction. The guard is not tidiness - measured on the
    luma band alone, where the transfer roll-off is clamped flat across the
    whole grid, the pair's condition number is 1.0e+30 against 1.000 on the
    colour-under band.

    `delay_s` defaults to a quarter cycle across the modulation half-width,
    which is the delay a band-pass of that width and a half-cycle of
    transition produces. It is a SCALE on a pure-phase signature and the
    normalised shape does not depend on it at all.

    `mechanics` is accepted and unused, and that is itself the finding: the
    two entries that survived are a delay and a Wiener weight, and neither is
    a function of the writing speed or the track width. Every entry that DID
    depend on the mechanics turned out to be the magnetic family read at
    another wavelength. The argument is kept so that a caller forwarding one
    set of keyword arguments to every module's `signatures` does not have to
    special-case this one.
    """
    grid = _grid(frequency_hz)
    low, high = float(grid.min()), float(grid.max())
    carrier = carrier_hz(system)
    if not low <= carrier <= high:
        return {}
    half = modulation_half_width_hz(system, band_upper_hz)
    if delay_s is None:
        delay_s = 0.25 / max(half, 1.0)
    return {
        "colour-under envelope phase":
            envelope_phase(grid, delay_s, system),
        "luma to colour-under transfer roll-off":
            transfer_rolloff(grid, system, band_upper_hz=band_upper_hz),
        "residual chroma carrier in luma":
            residual_carrier(grid, system),
    }


def residual_carrier(frequency_hz, system: str = "NTSC",
                     removed: float = LUMA_BEAT_REMOVED,
                     width_hz: Optional[float] = None) -> np.ndarray:
    """THE CHROMA CARRIER THAT SURVIVES IN THE LUMA, after the correction.

    Ethan: *"the residual chroma carrier in the luma like we discussed
    before"*, and *"use existing examples we have already done that match"*.

    The matching example is already in the key: a carrier leaking into a
    channel that should not carry it is a BEAT - a discrete line at a known
    offset rather than a response shape - and `interference.beat` is that
    model. This is that beat placed at the colour-under carrier, which for
    NTSC VHS is 629 kHz and NOT the 3.58 MHz subcarrier, because what the
    medium holds is the down-heterodyned chroma.

    THE SIZE IS MEASURED, NOT ASSUMED. The coupling is a real AM coupling
    between the two channels, and the shipped `--luma_beat` correction - a
    field-fitted tap under a burst-presence gate - removes 87 to 89 per cent
    of it. This entry carries the 11 to 13 per cent that is left, which is
    what a residual step still has to account for.

    The burst-presence gate is the reason this is a colour-under entry
    rather than a general interference one: the beat is there when the
    chroma is there and absent when it is not, so it is the chroma's
    footprint in the luma and belongs with the chroma's other entries.
    """
    from vhsdecode.models import interference

    grid = _grid(frequency_hz)
    centre = carrier_hz(system)
    # A LINE, NOT A BAND, and the width matters more than it looks. What
    # survives a field-fitted tap is the CARRIER, whose width is set by how
    # far it drifts from field to field - not by the chroma's modulation,
    # which the tap removes along with the rest. Given the chroma's
    # modulation half-width instead, the "line" is 500 kHz wide, covers the
    # whole colour-under grid, and the entry becomes a constant that adds no
    # direction: measured, its minimum went to 1.048 with no null at the
    # carrier at all. `interference.beat`'s own default width is the right
    # order and is what this uses.
    line = (interference.beat(grid, centre) if width_hz is None
            else interference.beat(grid, centre, width_hz=float(width_hz)))
    survives = max(0.0, 1.0 - float(removed))
    # a beat is a multiplicative departure from unity; scaling the DEPARTURE
    # rather than the entry keeps it a response and keeps it subtractable
    return (1.0 + (np.asarray(line) - 1.0) * survives).astype(np.complex128)


def candidates(frequency_hz, system: str = "NTSC",
               mechanics: Optional[Dict[str, float]] = None,
               band_upper_hz: Optional[float] = None,
               delay_s: Optional[float] = None,
               **head: float) -> Dict[str, np.ndarray]:
    """All four colour-under components, registered or not.

    What `signatures` would have returned before the marginal contributions
    were measured. Kept so that the disqualification can be reproduced rather
    than taken on trust: `distinguishable` reads 2.667 effective of 4 here at
    condition 1.49e+03, against 2.000 of 2 at condition 1.000 for the two that
    survived, and the difference is two entries whose coherence with each
    other is 0.99999.
    """
    grid = _grid(frequency_hz)
    mechanics = mechanics or {}
    speed = float(mechanics.get("writing_speed_m_s", 5.8))
    width = float(mechanics.get("track_width_m", 58e-6))
    half = modulation_half_width_hz(system, band_upper_hz)
    if delay_s is None:
        delay_s = 0.25 / max(half, 1.0)
    return {
        "colour-under envelope amplitude":
            envelope_amplitude(grid, speed, width, **head),
        "colour-under envelope phase":
            envelope_phase(grid, delay_s, system),
        "colour-under heterodyne offset":
            heterodyne_offset(grid, None, system, speed, width, **head),
        "luma to colour-under transfer roll-off":
            transfer_rolloff(grid, system, band_upper_hz=band_upper_hz),
    }


# --------------------------------------------------------------------------
# Reading the shape: the same measure the rest of the arc uses
# --------------------------------------------------------------------------


def _shapes(entries: Dict[str, np.ndarray]) -> np.ndarray:
    """The signatures as unit rows of `[log magnitude; phase]`.

    The treatment `interference.distinguishable` applies, kept here so the
    numbers are directly comparable: the log magnitude so that a gain is a
    constant rather than a scale, the unwrapped phase beside it, and the mean
    removed from each because a constant is a level and not a shape. Real and
    imaginary parts are STACKED rather than kept complex, which is
    `real_parameters=True` written out - each of these is the signature of a
    real physical parameter, so a gain and a delay of the same shape are two
    mechanisms and not one.
    """
    rows: List[np.ndarray] = []
    for value in entries.values():
        value = np.asarray(value).ravel()
        magnitude = np.log(np.maximum(np.abs(value), 1e-12))
        phase = (np.unwrap(np.angle(value)) if np.iscomplexobj(value)
                 else np.zeros_like(magnitude))
        vector = np.concatenate([magnitude - magnitude.mean(),
                                 phase - phase.mean()])
        norm = float(np.linalg.norm(vector))
        # a signature that did not move on this grid has no direction, and
        # normalising it would manufacture one out of rounding
        rows.append(vector / norm if norm > 1e-9 else np.zeros_like(vector))
    return np.array(rows) if rows else np.zeros((0, 0))


def _participation(stack: np.ndarray) -> Dict[str, object]:
    if stack.size == 0:
        return {"effective": 0.0, "condition": 0.0,
                "singular_values": np.zeros(0)}
    singular = np.linalg.svd(stack, compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    return {
        "effective": float(1.0 / np.sum(share ** 2)),
        "condition": float(singular[0] / max(singular[-1], 1e-30)),
        "singular_values": singular,
    }


def spread(entries: Dict[str, np.ndarray]) -> Dict[str, object]:
    """The participation reading for any set of signatures at all.

    How many independent directions the set spans, its condition number, and
    its singular values, on the same measure the whole arc uses. Public
    because the useful question about a candidate entry is a MARGINAL one -
    what a set reads with it and without it - and that needs the reading of
    an arbitrary set rather than of this module's own.
    """
    return _participation(_shapes(entries))


def coherence(first: np.ndarray, second: np.ndarray) -> float:
    """How closely two signatures agree, on [0, 1].

    One is a second copy of the same direction; zero is orthogonal. The
    magnetics' gap against azimuth reads 0.99997 here and the two registered
    colour-under entries read 5.7e-18.
    """
    stack = _shapes({"first": first, "second": second})
    return abs(float(stack[0] @ stack[1]))


def worst_pair(entries: Dict[str, np.ndarray]) -> Tuple[str, str, float]:
    """The two entries that agree most closely, and how closely.

    The magnetics' answer is gap against azimuth at 1.000 - two mechanisms
    that are the same sinc of the same group and are not distinguishable at
    all. Reproduced here from `head_model` over the whole band at 0.99997.
    Anything above about 0.99 is a second copy of a direction the family has.
    """
    names = list(entries)
    stack = _shapes(entries)
    worst = ("", "", 0.0)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = abs(float(stack[i] @ stack[j]))
            if value > worst[2]:
                worst = (names[i], names[j], value)
    return worst


def distinguishable(frequency_hz, all_four: bool = False,
                    **kwargs) -> Dict[str, object]:
    """How many independent directions the colour-under family spans.

    THE MEASURE AND THE STANDARD. Six tape magnetic loss mechanisms span 1.58
    effective directions at condition 1.06e+04, because every one of them is a
    function of a single dimensionless group, a length over the recorded
    wavelength. Eight multipath echoes span 7.92 of 8 at condition 1.2,
    because delay is the axis that makes signatures independent. A family is
    COLLINEAR when its members differ only in a rate and SEPARABLE when they
    differ in kind.

    Measured on the colour-under band, 58.74 kHz to 1.2 MHz, 1024 places:

        the two registered entries   2.000 of 2, condition 1.000
                                     singular values 1.000, 1.000
                                     mutual coherence 5.7e-18
        all four candidates          2.667 of 4, condition 1.49e+03
                                     worst pair the envelope amplitude
                                     against the heterodyne offset, 0.99999

    The registered pair is orthogonal to machine precision, which is what
    "new in kind" means when it is true rather than hoped: a delay lives on
    the axis conjugate to frequency and a Lorentzian peak at the carrier lives
    where no monotone decay has any feature. The two that were dropped are the
    demonstration that this measure can also say no.
    """
    entries = (candidates(frequency_hz, **kwargs) if all_four
               else signatures(frequency_hz, **kwargs))
    result = _participation(_shapes(entries))
    result["names"] = list(entries)
    result["count"] = len(entries)
    if len(entries) > 1:
        first, second, coherence = worst_pair(entries)
        result["worst_pair"] = (first, second)
        result["worst_coherence"] = coherence
    result["magnetics_baseline"] = "1.58 of 6 at condition 1.06e+04"
    return result


def magnetic_mechanisms(frequency_hz, writing_speed_m_s: float = 5.8,
                        track_width_m: float = 58e-6) -> Dict[str, np.ndarray]:
    """The tape's six loss mechanisms, one at a time, from `head_model`.

    The baseline every reading in this arc is measured against, taken from the
    head model itself rather than restated, so the comparison cannot drift
    away from the model it claims to be comparing with. Each is
    `head_model.log_response` with one parameter set and the head's own
    differentiation removed, plus the differentiation as its own entry.

    Reproduces the published figures: worst pair gap against azimuth at
    0.99997 over 59 kHz to 6.5 MHz - the same two mechanisms and the same
    1.000 the magnetics collapse reported - and 1.442 effective of 6 over the
    luma band alone.
    """
    grid = _grid(frequency_hz)
    typical = head_model.TYPICAL

    def one(**only: float) -> np.ndarray:
        parameters = {name: 0.0 for name in typical}
        parameters.update(only)
        logged = _head_log(grid, writing_speed_m_s, track_width_m, True,
                           parameters)
        return np.exp(logged - float(np.mean(logged))).astype(np.complex128)

    return {
        "spacing loss": one(spacing_m=typical["spacing_m"]),
        "gap loss": one(gap_m=typical["gap_m"]),
        "thickness loss": one(thickness_m=typical["thickness_m"]),
        "azimuth loss": one(azimuth_error_degrees=0.2),
        "contour effect": one(contour_length_m=typical["contour_length_m"],
                              contour_depth=0.02),
        "head differentiation": (grid / float(grid.max())).astype(
            np.complex128),
    }


def band_extension(system: str = "NTSC",
                   band_upper_hz: Optional[float] = None,
                   luma_upper_hz: float = 6.5e6,
                   step_hz: float = 6.5e3) -> Dict[str, object]:
    """DOES THE COLOUR-UNDER EXTEND THE IDENTIFIABLE BAND DOWNWARD, AND BY
    HOW MUCH? Yes for one family, NO for the other, and the difference is the
    finding.

    The question is asked the only way it can be asked honestly: with the grid
    DENSITY held fixed, so that the two readings differ in the band and in
    nothing else. A linear grid of a fixed number of points changes its own
    resolution when the band changes and the comparison would be reading the
    grid, which `grid_control` exists to rule out. The luma band is 1.2 to 6.5
    MHz - the decoder's own chroma ceiling to its own video band-pass high -
    and the whole band adds 58.74 kHz to 1.2 MHz below it.

    THE TAPE'S SIX MAGNETIC MECHANISMS, which is the family COMPONENT_MAPPINGS
    section 3 made its claim about:

        luma band alone       1.4419 effective of 6, condition 4.65e+05
        plus the colour-under 1.5528 of 6,            condition 1.98e+05
        and the two entries   2.3065 of 8,            condition 2.43e+05

    The extension buys 0.1109 of a direction and improves the conditioning by
    2.3 times, which reproduces the U-matic reading - 1.514 to 1.551 for the
    same move, at no cost - on a different format. The two colour-under
    entries then buy 0.7537 more, so they are seven times the larger part.

    THE MODELLED INTERFERENCE SET, and here the same move goes the OTHER WAY:

        luma band alone       7.6863 effective of 14, condition 10.6
        plus the colour-under 7.1953 of 14,           condition 13.5
        and the two entries   7.2993 of 16,           condition 22.2

    Extending the band COSTS 0.491 of a direction. The reason is not subtle
    once seen: a beat, a dropout, a sound trap, a vestigial-sideband slope and
    a set of echoes all live in the luma band and are FEATURELESS below 1.2
    MHz, so the added stretch is a region where several entries all say
    nothing, and entries that all say nothing in the same place are more alike
    than they were.

    THE RULE THIS GIVES, which is the modification rule of COMPONENT_MAPPINGS
    section 8a applied to the abscissa instead of to the key: A BAND EARNS ITS
    PLACE BY THE DIRECTIONS IT DISTINGUISHES. "Fit over the widest band the RF
    occupies" is correct for the head model, whose mechanisms are
    parameterised across the whole band, and it is NOT general - for a family
    of localised features a wider band dilutes rather than separates. Which
    family is being fitted decides.

    The set used for the interference reading is built here with every
    parameter fixed in absolute hertz, because `interference.signatures`
    derives several of its defaults from the grid's own edges and a comparison
    across two bands would otherwise be changing the components as well as the
    band. Read with those defaults instead, the same move reads 8.424 to 7.796
    - the same verdict, and the same 0.63 of a direction, reached less
    cleanly. The absolute levels of that second reading move whenever the
    interference set is edited; the sign and the size of the CHANGE do not,
    which is why the fixed-parameter set is what the docstring quotes and what
    the test asserts on.
    """
    from vhsdecode.models import interference as inf

    carrier = carrier_hz(system)
    half = modulation_half_width_hz(system, band_upper_hz)
    ceiling = band_upper_for(system, band_upper_hz)
    luma = np.arange(ceiling, luma_upper_hz + step_hz, step_hz)
    whole = np.arange(carrier - half, luma_upper_hz + step_hz, step_hz)

    def interference_set(grid: np.ndarray) -> Dict[str, np.ndarray]:
        """The modelled interference types with every parameter fixed in
        absolute hertz, so that changing the band changes only which
        frequencies are sampled. The figures are the decoder's own: the luma
        carrier's peak at 3.9 MHz (`video_rf_peak_freq`), the band's edges at
        1.2 and 6.5 MHz, and the intercarrier sound at 4.5 MHz, which is the
        standard's."""
        from vhsdecode.models import magnetic
        peak, low, high = 3.9e6, ceiling, luma_upper_hz
        span = high - low
        out = {
            "particle noise": inf.particle_noise(grid, 5.8, 58e-6),
            "modulation noise": inf.modulation_noise(grid, peak, 0.05 * span),
            "dropout": inf.dropout(grid, 0.4 * span, 0.5 * (low + high)),
            "head contact tilt": inf.head_contact_tilt(grid, low, high),
            "beat / co-channel": inf.beat(grid, 0.5 * (low + high)),
            "vestigial sideband": inf.vestigial_sideband(grid, low,
                                                         0.25 * span),
            "group delay": inf.group_delay(grid, 1.0 / high ** 2),
            "sound trap": inf.sound_trap(grid),
            "sub emphasis level dependence":
                inf.sub_emphasis(grid, -20.0) - inf.sub_emphasis(grid, 0.0),
            "record level dependence": magnetic.level_signature(grid, 1.0,
                                                                5.8),
        }
        for index in (1, 2, 4, 8):
            delay = index / span
            out[f"echo at {delay * 1e6:.2f} us"] = inf.echo(grid, delay)
        return out

    readings: Dict[str, Dict[str, object]] = {}
    for family, build in (("tape magnetics", magnetic_mechanisms),
                          ("modelled interference", interference_set)):
        alone = build(luma)
        extended = build(whole)
        with_ours = dict(extended)
        with_ours.update(signatures(whole, system,
                                    band_upper_hz=band_upper_hz))
        rows = {}
        for label, entries in (("luma band alone", alone),
                               ("plus the colour-under band", extended),
                               ("and the colour-under entries", with_ours)):
            value = _participation(_shapes(entries))
            value["count"] = len(entries)
            rows[label] = value
        rows["gained_by_the_band"] = (
            rows["plus the colour-under band"]["effective"]
            - rows["luma band alone"]["effective"])
        rows["gained_by_the_entries"] = (
            rows["and the colour-under entries"]["effective"]
            - rows["plus the colour-under band"]["effective"])
        readings[family] = rows
    return {
        "readings": readings,
        "luma_band_hz": (float(luma[0]), float(luma[-1])),
        "whole_band_hz": (float(whole[0]), float(whole[-1])),
        "places": (int(luma.size), int(whole.size)),
        "step_hz": float(step_hz),
        "why": ("a band earns its place by the directions it distinguishes: "
                "the magnetic mechanisms are parameterised across the whole "
                "band and gain, the localised interference entries are "
                "featureless below the chroma ceiling and are diluted"),
    }


def per_head_amount(frequency_hz, system: str = "NTSC",
                    amounts: Tuple[float, float] = (1.0, 1.15),
                    half_points_hz: Tuple[float, float] = (150e3, 300e3),
                    **kwargs) -> Dict[str, object]:
    """WHY NO PER-HEAD AMOUNT DIFFERENCE SURVIVES POOLING, with the mechanism.

    The established result is that the two heads' luma-to-chroma transfers
    differ by about 15 per cent of the transfer's own level - resolved at 3.4
    sigma in the mid bands, `vhsdecode/chroma.py` - but that no per-head
    AMOUNT difference survives proper pooling. This function shows why those
    two statements are consistent rather than asserting it.

    An amount is a SCALE on one direction. Two heads entered at two amounts
    are one shape twice, and the participation ratio of a family differing
    only in a scale is exactly one however far apart the scales are. Measured:

        two amounts, 1.00 and 1.15         1.000 effective of 2
        two half-points, 150 and 300 kHz   1.011 of 2
        two half-points, 150 and 1200 kHz  1.048 of 2

    The first row is the established result with its reason. THE OTHER TWO
    STRENGTHEN IT beyond what was claimed: even a per-head SHAPE, which is not
    a scale and could in principle be a direction, is worth 0.011 of one at
    the difference the runtime actually sees and 0.048 at eight times that.
    So the per-head distinction the runtime keeps is real as a correction and
    is not a direction to the key, and pooling could not have destroyed what
    was never there. This is the S-VHS sub pre-emphasis lesson of
    COMPONENT_MAPPINGS section 8a arriving from the other side.
    """
    grid = _grid(frequency_hz)
    scaled = {f"amount {value:.2f}":
              transfer_rolloff(grid, system, **kwargs) ** value
              for value in amounts}
    shaped = {f"half point {value / 1e3:.0f} kHz":
              transfer_rolloff(grid, system, half_hz=value, **kwargs)
              for value in half_points_hz}
    return {
        "by_amount": _participation(_shapes(scaled)),
        "by_shape": _participation(_shapes(shaped)),
        "amounts": tuple(float(v) for v in amounts),
        "half_points_hz": tuple(float(v) for v in half_points_hz),
        "why": ("an amount is a scale on one direction and adds none; a "
                "half-point is a shape and adds a fiftieth of one"),
    }


# --------------------------------------------------------------------------
# THE CONTROLS. Kept as code because a control that cannot fail is not a
# control - an earlier component in this arc reported 4.65 effective
# directions of 5, and what exposed the error was a linear control that
# should have read 1.00 and read 4.97 instead.
# --------------------------------------------------------------------------


def _tone_at(signal: np.ndarray, frequency_hz: float,
             sample_rate_hz: float) -> complex:
    """The complex amplitude of one tone in a real record, by projection.

    A Hann window and an inner product with `exp(-2 pi j f t)`, normalised so
    that a real cosine of unit amplitude returns unit modulus and its own
    starting phase. The control below chooses the record length so that every
    other product of the mix is hundreds of bins away and the leakage into
    this projection is negligible.
    """
    count = signal.size
    window = np.hanning(count)
    time = np.arange(count) / float(sample_rate_hz)
    kernel = window * np.exp(-2j * np.pi * float(frequency_hz) * time)
    return complex(2.0 * (kernel @ signal) / float(window.sum()))


def heterodyne_control(system: str = "NTSC", places: int = 1 << 16,
                       modulation_hz: float = 200e3,
                       amplitudes: Tuple[float, ...] = (0.4, 0.7, 1.0, 1.3),
                       phases_rad: Tuple[float, ...] = (-1.1, -0.4, 0.3, 1.0)
                       ) -> Dict[str, object]:
    """THE CONTROL, and its answer is known before it is run.

    THE RULING IT GUARDS. Heterodyning is multiplication by a local
    oscillator, so a correction expressed as a COMPLEX GAIN PER FREQUENCY
    crosses the mix with NO SCALE FACTOR AT ALL, magnitude and phase alike.
    The 5.6875 ratio of `information_extrapolation.heterodyne_timing_scale` is
    real and necessary, but only where a phase is read as a TIME - the chroma
    is time-base corrected while it is still the colour-under, so a residual
    grid displacement imprints its phase at `f_cu` and reading that phase at
    the subcarrier understates the displacement by that factor. Applying the
    ratio to a GAIN is the mistake the ruling exists to prevent, and it is a
    mistake that leaves every shape looking perfectly reasonable.

    IT IS COMPUTED, NOT ASSERTED. The control does not compare a signature
    with a copy of itself; that is a control that cannot fail. It builds a
    real colour-under tone at `f_cu + m` with a known amplitude and a known
    starting phase, multiplies it by the decoder's own oscillator -
    `-cos(2 pi f_LO t)` with `f_LO = f_sc + f_cu`, which is what
    `chroma.upconvert_chroma_phase_comp` does - and measures what comes out at
    the subcarrier by projection. Then it regresses the measured amplitude on
    the amplitude sent and the measured phase on the phase sent.

    THE INDEPENDENTLY KNOWN ANSWERS. The amplitude slope is 1, the mix not
    scaling a gain, and the phase slope is exactly -1: unit magnitude, which
    is the ruling, and negative because the difference product conjugates.
    The landing frequency is `f_sc - m` and not `f_sc + m`, the band being
    mirrored. Measured: amplitude slope 1.0000000, phase slope -1.0000000, and
    the unmirrored image at `f_sc + m` at 2.9e-11 of the signal.

    HOW IT FAILS. A phase slope of -5.6875 is the timing ratio wrongly applied
    to a gain. A slope of +1 is the mirror forgotten. An amplitude slope away
    from 1 is a scale factor that does not exist. Any of the three would leave
    the signatures in this module looking well behaved while making every
    correction carried across the mix wrong, which is exactly the class of
    error this arc has been caught by before.
    """
    from vhsdecode.models import information_extrapolation as ie

    carrier = carrier_hz(system)
    subcarrier = subcarrier_hz(system)
    oscillator = heterodyne_local_oscillator_hz(system)
    # the decoder's own output rate, four times the subcarrier
    sample_rate = 4.0 * subcarrier
    count = int(places)
    time = np.arange(count) / sample_rate
    local = -np.cos(2.0 * np.pi * oscillator * time)

    modulation = float(modulation_hz)
    lands_at = subcarrier - modulation
    unmirrored = subcarrier + modulation

    sent_amplitude, got_amplitude = [], []
    sent_phase, got_phase = [], []
    image = 0.0
    for amplitude in amplitudes:
        for phase in phases_rad:
            recorded = amplitude * np.cos(
                2.0 * np.pi * (carrier + modulation) * time + phase)
            crossed = recorded * local
            measured = _tone_at(crossed, lands_at, sample_rate)
            # the difference product carries half the amplitude
            sent_amplitude.append(amplitude)
            got_amplitude.append(2.0 * abs(measured))
            sent_phase.append(phase)
            got_phase.append(np.angle(measured))
            image = max(image,
                        abs(_tone_at(crossed, unmirrored, sample_rate))
                        / max(abs(measured), 1e-30))

    def slope(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.unwrap(np.asarray(y, dtype=np.float64))
        design = np.vstack([x, np.ones_like(x)]).T
        return float(np.linalg.lstsq(design, y, rcond=None)[0][0])

    amplitude_slope = slope(sent_amplitude, got_amplitude)
    phase_slope = slope(sent_phase, got_phase)
    scale = ie.heterodyne_timing_scale(subcarrier, carrier)
    return {
        "amplitude_slope": amplitude_slope,
        "amplitude_expected": 1.0,
        "phase_slope": phase_slope,
        "phase_expected": -1.0,
        "lands_at_hz": float(lands_at),
        "unmirrored_image_share": float(image),
        "timing_scale": float(scale),
        "phase_slope_if_the_ratio_were_applied": float(-scale),
        "passes": bool(abs(amplitude_slope - 1.0) < 1e-3
                       and abs(phase_slope + 1.0) < 1e-3
                       and image < 1e-3),
        "why": ("a complex gain per frequency crosses the heterodyne with no "
                "scale factor: the amplitude slope is one and the phase slope "
                "is minus one, the sign being the mirror and not a scale. A "
                "phase slope of -5.6875 would be the timing ratio wrongly "
                "applied to a gain"),
    }


def wavelength_sharing_control(system: str = "NTSC",
                               spacings_m: Tuple[float, ...] =
                               (0.02e-6, 0.05e-6, 0.20e-6),
                               places: int = 256) -> Dict[str, object]:
    """THE SECOND CONTROL: the shared loss must be in the frequency ratio,
    exactly.

    Wallace's spacing loss is `exp(-2 pi d / lambda)` with `lambda = v / f`,
    so the loss in nepers is `2 pi d f / v` - LINEAR IN FREQUENCY with no
    constant term. Two carriers through the same separation therefore took
    losses in the exact ratio `f_cu / f_luma`, and `wavelength_sharing` with
    its wavelength-independent share set to zero must reproduce that ratio to
    machine precision at every frequency and for every separation.

    The answer is known independently of anything in this module and it is
    1.000000, with no tolerance to negotiate. IT CAN FAIL: writing the
    wavelength as `v * f` instead of `v / f`, dropping the `2 pi`, or letting
    the wavelength-independent share leak into the ratio all move it off one,
    and each of those would leave the amplitude signatures looking perfectly
    reasonable while making every fitted separation wrong.

    Measured: worst departure from unity 2.2e-16 over three separations and
    256 frequencies, and 0.644 when the measured wavelength-independent share
    is left in - which is the check noticing the difference it exists to
    notice, and incidentally the size of the departure that share represents.
    """
    speed = 5.8
    grid = np.linspace(1.2e6, 6.5e6, int(places))
    carrier = carrier_hz(system)
    worst = 0.0
    for spacing in spacings_m:
        luma_loss = 2.0 * np.pi * float(spacing) * grid / speed
        chroma_loss = 2.0 * np.pi * float(spacing) * carrier / speed
        predicted = np.abs(wavelength_sharing(grid, system, 0.0))
        worst = max(worst,
                    float(np.max(np.abs((chroma_loss / luma_loss)
                                        / predicted - 1.0))))
    with_share = np.abs(wavelength_sharing(grid, system))
    leaked = float(np.max(np.abs((carrier / grid) / with_share - 1.0)))
    return {
        "worst_departure": worst,
        "expected": 0.0,
        "departure_if_the_share_leaked": leaked,
        "passes": bool(worst < 1e-12),
        "why": ("the loss in nepers is proportional to frequency, so two "
                "carriers through one separation take losses in the exact "
                "ratio of their frequencies - a prediction with no free "
                "parameter, which a construction error would break"),
    }


def linear_channel_control(system: str = "NTSC", places: int = 1024,
                           spacing_m: float = 0.05e-6) -> Dict[str, object]:
    """THE THIRD CONTROL: a heterodyne offset on a linear channel must be a
    CONSTANT, and therefore no direction at all.

    `heterodyne_offset` is `H(f + d) / H(f)`, so its log is `log H(f + d) -
    log H(f)`. If `log H` is linear in frequency that difference is a
    constant, its mean-removed shape is identically zero, and the entry
    carries no direction however large the offset. Wallace's spacing loss is
    exactly `-2 pi d f / v`, linear, so the answer is known before the code
    runs: a shape norm of zero.

    This is the control that disqualified the entry, and it is the reason the
    disqualification is a result rather than an opinion. Measured:

        a pure Wallace channel            1.8e-15 nepers - machine zero
        the head's equalised loss stack   1.2e-03
        the unequalised head response     1.243

    The middle row is the whole loss stack of a real head and it is three
    orders below anything a fit could use; the last row is the head's own
    differentiation, `1/f`, which the deck equalises, so it measures the
    equaliser rather than the tape.

    IT CAN FAIL. Writing the ratio as a product, or as a difference of
    responses rather than of their logarithms, or forgetting to remove the
    mean, all give a non-zero norm on a channel where the answer is provably
    zero - and any of them would make an inert entry look like a direction,
    which is precisely the error that produced a 4.65-of-5 reading elsewhere
    in this arc.
    """
    carrier = carrier_hz(system)
    half = modulation_half_width_hz(system)
    grid = np.linspace(carrier - half, carrier + half, int(places))
    offset = OFFSET_QUANTUM_LINES * line_rate_hz(system)
    speed = 5.8

    def norm(values: np.ndarray) -> float:
        logged = np.log(np.maximum(np.abs(values), 1e-300))
        return float(np.linalg.norm(logged - logged.mean()))

    wallace = (np.exp(-2.0 * np.pi * spacing_m * (grid + offset) / speed)
               / np.exp(-2.0 * np.pi * spacing_m * grid / speed))
    return {
        "pure_wallace_shape_norm": norm(wallace),
        "expected": 0.0,
        "equalised_loss_stack_shape_norm":
            norm(heterodyne_offset(grid, offset, system, speed)),
        "unequalised_response_shape_norm":
            norm(heterodyne_offset(grid, offset, system, speed,
                                   equalised=False)),
        "passes": bool(norm(wallace) < 1e-12),
        "why": ("a difference of a linear function is a constant, so an "
                "offset on a Wallace channel has no shape; anything but zero "
                "means the ratio was not formed in the log domain"),
    }


def grid_control(system: str = "NTSC",
                 places: Tuple[int, ...] = (512, 1024, 2048)
                 ) -> Dict[str, object]:
    """THE FOURTH CONTROL: the count must be a property of the shapes, not of
    the sampling.

    A participation ratio over normalised shapes is a property of the shapes.
    Sampling them more finely on the same band must not move it, and if it
    does then either a signature has a feature narrower than the grid or the
    normalisation is reading the number of places. The headline reading of
    this module is a comparison between two BANDS, and that comparison is
    worthless if the measure responds to the grid at all.

    Expected: the same effective count at 512, 1024 and 2048 places, to three
    decimals. Measured on the registered pair: 2.000, 2.000, 2.000, a spread
    of 0.0e+00 - the pair being exactly orthogonal, which is itself the
    strongest form of the answer. Measured on all four candidates, where the
    shapes are not orthogonal and the test has something to do: 2.666647,
    2.666647, 2.666647, a spread of 5.0e-08.
    """
    counts, counts_all = {}, {}
    low = carrier_hz(system) - modulation_half_width_hz(system)
    for count in places:
        grid = np.linspace(low, band_upper_for(system), int(count))
        counts[count] = distinguishable(grid, system=system)["effective"]
        counts_all[count] = distinguishable(grid, all_four=True,
                                            system=system)["effective"]
    spread = float(max(counts.values()) - min(counts.values()))
    spread_all = float(max(counts_all.values()) - min(counts_all.values()))
    return {
        "effective_by_places": counts,
        "all_four_by_places": counts_all,
        "spread": spread,
        "spread_all_four": spread_all,
        "expected": 0.0,
        "passes": bool(spread < 1e-3 and spread_all < 1e-3),
        "why": ("a participation ratio over normalised shapes is a property "
                "of the shapes; if it moved with the sampling, the band "
                "comparison would be reading the grid"),
    }
