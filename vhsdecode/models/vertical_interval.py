"""The vertical interval test signals, whose SYNTHETIC SIDE IS SPECIFIED.

Ethan: *"the complex convolusion over n dimensions along with their known
shapes, which we have modeled using the sync pulse, vits, rf, colro carrier,
genlock, etc. Those are all known and all have an order and strict compliance
to them."*

WHY THIS IS THE MOST VALUABLE INPUT THE METHOD CAN BE GIVEN. Every other
component's synthetic side is a fitted curve, and a curve fitted to data
describes that data by construction - which is the whole reason the arc
carries an out-of-sample rule. An insertion test signal has no such problem.
Its bar amplitude, its pulse half-amplitude durations, its multiburst
frequencies, its staircase levels and its subcarrier phases are printed in a
standard to a tenth of an IRE, with tolerances beside them. Nothing here is
estimated from a decode; the differential is against the standard.

WHAT IS BUILT HERE, and the citation for every number:

    NTC-7 composite      ITU-T J.63 (06/90) Annex II section 2, line 17 f1
    NTC-7 combination    ITU-T J.63 (06/90) Annex II section 3, line 17 f2
    75/7.5/75/7.5 bars   SMPTE EG 27-2004 table 2, restated to 1e-4 IRE as
                         SMPTE 170M-2004 Annex A table A.3
    T, the pulse ladder  ITU-R BT.1439-1 Annex 4 section 2: T = 1/(2 Fc),
                         with Fc = 4 MHz for the 525-line system, so
                         T = 125 ns, 2T = 250 ns and 12.5T = 1.5625 us

THE NUMBERS THIS MODULE PRODUCED ON ITS OWN RUNS. Every one is regenerated
by `line_dimensions`, `survival`, `amplitude_axis` and `gain_control`
below, and each is quoted again in the docstring of the function that makes
it.

    signal                 readings  effective  per reading  condition
    NTC-7 combination         26       10.53       0.405        123
    NTC-7 composite           27        5.61       0.208       1.2e4
    75% colour bars           26        4.76       0.183       5.7e3
    (for comparison) tape magnetics    1.58 of 6   0.263       1.1e4

THE COMPOSITE LINE IS CONFIRMED AS INEFFICIENT, at 0.208 independent
dimensions per reading against `docs/VERTICAL_INTERVAL_COLLAPSE.md` section
4.3's 0.186 from a construction that shares no code with this one, and for
the reason that document gives: its bar, its 2T pulse, its 12.5T luminance
half and its staircase treads are four views of one low-frequency luminance
channel, so a gain change moves all four together. The combination line is
twice as efficient because its multiburst is parameterised along the axis
that makes its parts different - the same reason echoes separate where tape
losses do not.

ONE CORRECTION TO THAT DOCUMENT. It records the composite as the worst of
eleven insertion signals. Measured here against the colour bars, which that
table does not contain, the composite is NOT the worst: the 75 per cent bars
return 0.183 per reading, below the composite's 0.208. The bars' eight
luminance levels are eight readings of one level, so on the frequency axis
they collapse harder than anything in the composite - which is the other
side of the coin to their value on the AMPLITUDE axis, where nothing else
competes with them.

TWO READING DEFECTS THAT MANUFACTURE FINDINGS OUT OF ARITHMETIC, both
avoided here and both measured in `read_pulse`. A 2T pulse read from the
4fsc grid at a threshold fixed in absolute IRE comes back 279.4 ns against a
specified 250 - nearly 12 per cent wide, three times the standard's whole
+-10 ns tolerance, from arithmetic alone - where the same data read at the
pulse's own half amplitude with a band-limited interpolation returns
252.3 ns. And a threshold fixed in absolute IRE stops being a width
measurement the moment the amplitude has changed, which is the condition
every tape decode is in. `standard_levels.half_amplitude_crossings` reads at
the pulse's own half amplitude with sub-sample interpolation and is what is
used throughout; `gain_control` below is the control that catches the
alternative, and it catches it.

WHAT THE SIGNATURES ARE. An insertion test element is a STIMULUS, not a
modification: it does not act on the signal, it IS the signal. What it
contributes to the key is therefore where in the band it puts energy, and
hence which part of the channel it can measure at all. Each element enters
as its own prescribed spectrum in the bounded form the transform requires,

    W_i(f) = 1 + a * S_i(f) / max|S_i|,     |a| < 1

so the magnitude never reaches zero, the logarithm is finite everywhere and
`interference.subtractable` returns True. The spectrum is referenced to the
element's OWN CENTRE and not to the line's origin, deliberately: an element's
position within the line is a layout convention rather than a specified
quantity, and a construction whose separation came from the layout would be
reporting an assumption as a measurement.

WHERE A DECODE PLUGS IN. Everything here is the synthetic side alone, and
nothing in this module reads a capture. `line_dimension` is the one hook:
hand it a decoded line resampled onto the specification's own time axis and
it returns the channel, as `pair_dimension.pair_transfer` returns any other
dimension, with the debiased coherence that says where the estimate is
believable. `pair_dimension.sync_dimension` is the same object built from the
sync pulse; this one has energy across the whole video band instead of at one
edge.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import pair_dimension
from vhsdecode.models import standard_levels


# ==========================================================================
# The format. Every constant below is a standard's, named where it is used.
# ==========================================================================

# SMPTE 170M-2004 clause 8.1: the colour subcarrier frequency is exactly
# 315/88 MHz, and the line frequency is 2/455 of it.
SUBCARRIER_HZ = 315.0e6 / 88.0
LINE_RATE_HZ = SUBCARRIER_HZ * 2.0 / 455.0
LINE_S = 1.0 / LINE_RATE_HZ

# The decoder's own output grid is four times the subcarrier, which is 910
# samples on one line exactly. This is not a choice: it is the grid the time
# base correction writes, so a synthetic side built on it needs no
# interpolation to meet a decode.
SAMPLE_RATE_HZ = 4.0 * SUBCARRIER_HZ
SAMPLES_PER_LINE = 910

# ITU-R BT.1439-1 Annex 4 section 2 defines T = 1/(2 Fc) with Fc the nominal
# bandwidth of the channel under test. ITU-T J.63 Annex II section 2.2 fixes
# the 525-line value at 2T = 250 +- 10 ns, which is Fc = 4 MHz.
#
# ONE INFERENCE, LABELLED AS SUCH. ITU-R BT.1701-1 table 1 item 4 gives the
# M/N nominal main sideband as 4.2 MHz, which would make T = 119 ns. The
# 125 ns figure is a rounded convention and no document states the
# reconciliation; `nyquist_interval_s` takes the cut-off as an argument so
# the convention is visible rather than buried.
NOMINAL_CUTOFF_HZ = 4.0e6

# ITU-T J.63 Annex II section 2.2 and section 4.2: the composite pulse for
# the 525-line system is 12.5T and the narrow pulse is 2T.
NARROW_PULSE_T = 2.0
COMPOSITE_PULSE_T = 12.5

# SMPTE 170M-2004 table 2 and clause 8.4, carried by `pair_dimension`: sync
# is 4.7 us wide at half amplitude with 140 ns edges and sits 40 IRE below
# blanking; the burst is 40 +- 1 IRE peak to peak, 9 +- 1 cycles, beginning
# 19 subcarrier cycles after the sync leading edge.
SYNC_WIDTH_S = pair_dimension.SYNC_WIDTH_S
SYNC_RISE_S = pair_dimension.SYNC_RISE_S
SYNC_DEPTH_IRE = pair_dimension.SYNC_DEPTH_IRE
BURST_CYCLES = 9.0
BURST_START_CYCLES = 19.0
BURST_IRE_PP = 40.0
BURST_PHASE_DEG = 180.0          # the burst is on the -(B-Y) axis

# Tektronix TSG-130A instruction manual table 3-2, the generator that made
# this project's test captures: front porch 1.5 +- 0.1 us, luminance rise
# 250 +- 25 ns, chrominance rise 400 +- 40 ns. The front porch is used as
# the line's origin so that the sync pulse's leading edge is renderable in
# full rather than clipped at the array boundary.
FRONT_PORCH_S = 1.5e-6
LUMINANCE_RISE_S = 250e-9
CHROMINANCE_RISE_S = 400e-9

# This project's standing colour-under figures, the same ones
# `interference.sound_trap` relies on when it notes that a 4.5 MHz trap is
# unreachable through a VHS recording.
VHS_LUMA_HZ = 3.0e6
VHS_CHROMA_HALF_HZ = 0.4e6
UMATIC_LUMA_HZ = 3.5e6
UMATIC_CHROMA_HALF_HZ = 0.5e6


def nyquist_interval_s(cutoff_hz: float = NOMINAL_CUTOFF_HZ) -> float:
    """T, the Nyquist interval of the channel under test.

    ITU-R BT.1439-1 Annex 4 section 2: `T = 1/(2 Fc)`. At the 525-line
    system's 4 MHz this is 125 ns, so 2T is 250 ns and 12.5T is 1.5625 us -
    derived here rather than written down, because the ladder is a
    consequence of the cut-off and not a set of independent numbers.
    """
    return 1.0 / (2.0 * float(cutoff_hz))


# ==========================================================================
# Waveform primitives. A sine-squared pulse and a sine-squared edge, which
# is what every element in an insertion test signal is made of.
# ==========================================================================


def _edge_ten_ninety_fraction() -> float:
    """What fraction of a sine-squared edge's full width its 10-90 time is.

    The edge is `sin^2(pi (x + 1) / 4)` on `x` in [-1, 1], so it reaches 0.1
    and 0.9 at `x = (4/pi) arcsin(sqrt(p)) - 1`, and the 10-90 span is

        (2/pi) (arcsin(sqrt(0.9)) - arcsin(sqrt(0.1))) = 0.5904

    of the full width. Computed rather than written down: a specification
    states a 10-90 TIME, and rendering it as a transition width instead
    makes every edge in the line 1.69 times too fast.
    """
    return float(2.0 / np.pi * (np.arcsin(np.sqrt(0.9))
                                - np.arcsin(np.sqrt(0.1))))


def _edge(x: np.ndarray) -> np.ndarray:
    """A sine-squared edge on `x` in [-1, 1]: 0 at one end, 1 at the other."""
    return np.sin(np.pi * (np.clip(x, -1.0, 1.0) + 1.0) / 4.0) ** 2


def sine_squared_pulse(t_s: np.ndarray, centre_s: float, had_s: float,
                       height: float) -> np.ndarray:
    """A sine-squared pulse of the stated HALF-AMPLITUDE DURATION.

    `cos^2(pi t / (2 HAD))` on `|t| <= HAD` is exactly 0.5 at `t = HAD/2`,
    so HAD is the full width at half maximum - which is what a
    half-amplitude duration specification means, and it is the quantity
    ITU-T J.63 Annex II section 2.2 bounds at 250 +- 10 ns.
    """
    x = (np.asarray(t_s, dtype=np.float64) - float(centre_s)) / float(had_s)
    return float(height) * np.where(np.abs(x) <= 1.0,
                                    np.cos(np.pi * x / 2.0) ** 2, 0.0)


def bar(t_s: np.ndarray, start_s: float, stop_s: float, height: float,
        rise_s: float) -> np.ndarray:
    """A bar with sine-squared edges of the stated 10-90 rise time."""
    half = float(rise_s) / (2.0 * _edge_ten_ninety_fraction())
    t = np.asarray(t_s, dtype=np.float64)
    return float(height) * np.minimum(_edge((t - float(start_s)) / half),
                                      _edge((float(stop_s) - t) / half))


def subcarrier(t_s: np.ndarray, phase_deg: float,
               frequency_hz: float = SUBCARRIER_HZ) -> np.ndarray:
    """One cycle-continuous carrier at the stated phase.

    The phase reference is the line's own time origin, and the burst is
    rendered at 180 degrees, so every phase quoted here is the standard's
    angle from the +(B-Y) axis exactly as SMPTE 170M-2004 Annex A tabulates
    it. Nothing in this module ever measures an absolute phase against
    anything but the burst.
    """
    return np.cos(2.0 * np.pi * float(frequency_hz)
                  * np.asarray(t_s, dtype=np.float64)
                  + np.deg2rad(float(phase_deg)))


def packet(t_s: np.ndarray, start_s: float, stop_s: float,
           frequency_hz: float, amplitude_ire: float,
           rise_s: float, phase_deg: float = 0.0) -> np.ndarray:
    """A multiburst packet: a carrier under a sine-squared envelope.

    `amplitude_ire` is the carrier's PEAK, so a packet specified at 50 IRE
    peak to peak is built with 25.
    """
    envelope = bar(t_s, start_s, stop_s, 1.0, rise_s)
    return float(amplitude_ire) * envelope * subcarrier(t_s, phase_deg,
                                                        frequency_hz)


# ==========================================================================
# The three specified lines.
#
# AMPLITUDES, DURATIONS, FREQUENCIES AND PHASES ARE THE STANDARD'S.
# ELEMENT POSITIONS WITHIN THE ACTIVE LINE ARE THE CONVENTIONAL LAYOUT AND
# ARE AN ASSUMPTION - J.63 draws the elements in order without fixing every
# start to the nanosecond. Nothing measured in this module depends on them:
# the signatures are referenced to each element's own centre (see
# `element_signatures`) and the readings window each element by its own
# declared span.
# ==========================================================================

# Positions are quoted from the sync pulse's leading edge, which is J.63's
# own timing reference ("timing is measured from the H-sync leading edge").
_LAYOUT_ORIGIN_S = FRONT_PORCH_S


def _at(offset_us: float) -> float:
    """A position quoted from the sync leading edge, as absolute line time."""
    return _LAYOUT_ORIGIN_S + offset_us * 1e-6


def _line_grid(samples: int = SAMPLES_PER_LINE,
               rate_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.arange(int(samples), dtype=np.float64) / float(rate_hz)


def _sync_and_burst(t_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sync and burst, present on every line whatever it carries.

    The pulse comes from `pair_dimension.spec_sync`, which already carries
    the standard's 4.7 us width and 140 ns 10-90 edges and already corrects
    the transition width for the raised cosine's own 10-90 fraction. It is
    generated on its own centred window and placed at the front porch, so
    the leading edge is present in full.
    """
    rate = 1.0 / float(t_s[1] - t_s[0])
    guard = 4.0 * SYNC_RISE_S
    window = int(round((SYNC_WIDTH_S + 2.0 * guard) * rate))
    pulse = pair_dimension.spec_sync(rate, window, SYNC_WIDTH_S,
                                     SYNC_RISE_S, SYNC_DEPTH_IRE)
    luma = np.zeros_like(t_s)
    start = int(round((_LAYOUT_ORIGIN_S - guard) * rate))
    stop = min(start + pulse.size, luma.size)
    if start >= 0 and stop > start:
        luma[start:stop] = pulse[:stop - start]

    begin = _LAYOUT_ORIGIN_S + BURST_START_CYCLES / SUBCARRIER_HZ
    chroma = bar(t_s, begin, begin + BURST_CYCLES / SUBCARRIER_HZ,
                 BURST_IRE_PP / 2.0, CHROMINANCE_RISE_S)
    chroma = chroma * subcarrier(t_s, BURST_PHASE_DEG)
    return luma, chroma


def _line(name: str, citation: str, elements: Dict[str, Dict[str, object]],
          geometry: Dict[str, object],
          samples: int = SAMPLES_PER_LINE,
          rate_hz: float = SAMPLE_RATE_HZ) -> Dict[str, object]:
    """Assemble a rendered line from its elements.

    LUMINANCE AND CHROMINANCE ARE RENDERED ON SEPARATE PLANES AND SUMMED AT
    THE END. This is not tidiness: a modulated staircase is a luminance
    element that REPLACES the level laid over a chrominance element that
    ADDS across the same window, and rendering them into one array in
    sequence erases the subcarrier entirely - which is the whole element.
    """
    t = _line_grid(samples, rate_hz)
    luma, chroma = _sync_and_burst(t)
    for entry in elements.values():
        luma = luma + entry.get("luma", 0.0)
        chroma = chroma + entry.get("chroma", 0.0)
    return {
        "name": name,
        "citation": citation,
        "time_s": t,
        "sample_rate_hz": float(rate_hz),
        "luma_ire": luma,
        "chroma_ire": chroma,
        "composite_ire": luma + chroma,
        "elements": elements,
        "geometry": geometry,
    }


def ntc7_composite(samples: int = SAMPLES_PER_LINE,
                   rate_hz: float = SAMPLE_RATE_HZ,
                   cutoff_hz: float = NOMINAL_CUTOFF_HZ) -> Dict[str, object]:
    """The NTC-7 composite line - ITU-T J.63 Annex II section 2, line 17 f1.

    Four elements and what each is for, from J.63 Annex II table IV, which
    is the authoritative element-to-characteristic map:

    | element | what it measures |
    |---|---|
    | B2, the 100 IRE white bar, 18 us, 125 ns rise | insertion gain, line-time waveform distortion, step response |
    | B1, the 2T pulse, 100 IRE, HAD 250 ns | short-time distortion, pulse response |
    | F, the 12.5T modulated pulse, HAD 1.5625 us, 50 IRE luminance and 100 IRE p-p chrominance at 60.8 degrees | chrominance-luminance gain and delay inequality |
    | D2, the five-step modulated staircase, 18 IRE per step to 90 IRE, 40 IRE p-p chrominance on every step | luminance non-linearity, differential gain, differential phase |

    TEN OF THE TWELVE CHARACTERISTICS J.63 TABLE IV NAMES ARE READ FROM
    THIS ONE LINE, and `line_dimensions` measures 5.61 independent
    directions behind the 27 readings its own elements name - so the
    conventional reading is between two and five times as many quantities as
    the line can independently determine, depending on which of them are
    written down.

    The chrominance on the staircase is rendered at the burst phase. J.63
    fixes its amplitude at 40 IRE peak to peak and not its angle, so that
    angle is an ASSUMPTION; it moves the differential-phase reading's
    origin and nothing else.
    """
    t = _line_grid(samples, rate_hz)
    unit = nyquist_interval_s(cutoff_hz)
    elements: Dict[str, Dict[str, object]] = {}

    # B2: 100 IRE white bar, 18 us long, 125 ns rise. The rise is the
    # TSG-130A's own figure for this pattern (manual table 3-4) and equals
    # one T, which is why the bar and the 2T pulse share a bandwidth.
    bar_start, bar_stop = _at(12.0), _at(30.0)
    elements["B2 white bar"] = {
        "luma": bar(t, bar_start, bar_stop, 100.0, unit),
        "window_s": (bar_start, bar_stop),
        "centre_s": 0.5 * (bar_start + bar_stop),
        "band_hz": 1.0 / unit,
    }

    # B1: the 2T pulse at 100 IRE, half-amplitude duration 250 +- 10 ns.
    two_t = NARROW_PULSE_T * unit
    pulse_centre = _at(34.0)
    elements["B1 2T pulse"] = {
        "luma": sine_squared_pulse(t, pulse_centre, two_t, 100.0),
        "window_s": (pulse_centre - 1.5 * two_t, pulse_centre + 1.5 * two_t),
        "centre_s": pulse_centre,
        "had_s": two_t,
        "band_hz": 1.0 / two_t,
    }

    # F: the 12.5T modulated pulse. J.63 requires its peak to sit within
    # +-0.5 IRE of B2, which is the construction that makes the luminance
    # half 50 IRE and the chrominance 100 IRE peak to peak with its base
    # line at blanking. The chrominance angle is J.63's 60.8 degrees.
    wide_t = COMPOSITE_PULSE_T * unit
    wide_centre = _at(37.0)
    envelope = sine_squared_pulse(t, wide_centre, wide_t, 1.0)
    elements["F 12.5T pulse, luminance"] = {
        "luma": 50.0 * envelope,
        "window_s": (wide_centre - wide_t, wide_centre + wide_t),
        "centre_s": wide_centre,
        "had_s": wide_t,
        "band_hz": 1.0 / wide_t,
    }
    elements["F 12.5T pulse, chrominance"] = {
        "chroma": 50.0 * envelope * subcarrier(t, 60.8),
        "window_s": (wide_centre - wide_t, wide_centre + wide_t),
        "centre_s": wide_centre,
        "band_hz": SUBCARRIER_HZ,
    }

    # D2: five steps of 18 IRE from a 0 IRE base to 90 IRE, each carrying
    # 40 IRE peak to peak of subcarrier.
    steps = 5
    top = 90.0
    tread = 4.0
    first = 40.0
    levels = np.arange(1, steps + 1, dtype=np.float64) * (top / steps)
    treads = []
    staircase = np.zeros_like(t)
    for index, level in enumerate(levels):
        start, stop = _at(first + index * tread), _at(first
                                                      + (index + 1) * tread)
        staircase = staircase + bar(t, start, stop, level, LUMINANCE_RISE_S)
        treads.append((start, stop, float(level)))
    elements["D2 staircase, luminance"] = {
        "luma": staircase,
        "window_s": (treads[0][0], treads[-1][1]),
        "centre_s": 0.5 * (treads[0][0] + treads[-1][1]),
        "band_hz": 1.0 / LUMINANCE_RISE_S,
    }
    chroma_start, chroma_stop = treads[0][0], treads[-1][1]
    elements["D2 staircase, chrominance"] = {
        "chroma": (40.0 / 2.0) * bar(t, chroma_start, chroma_stop, 1.0,
                                     CHROMINANCE_RISE_S)
        * subcarrier(t, BURST_PHASE_DEG),
        "window_s": (chroma_start, chroma_stop),
        "centre_s": 0.5 * (chroma_start + chroma_stop),
        "band_hz": SUBCARRIER_HZ,
    }

    geometry = {
        "bar": (bar_start, bar_stop),
        "narrow_pulse": (pulse_centre, two_t),
        "wide_pulse": (wide_centre, wide_t),
        "treads": treads,
        "chroma_reference_ire_pp": 40.0,
    }
    return _line("NTC-7 composite",
                 "ITU-T J.63 (06/90) Annex II section 2; element map "
                 "Annex II table IV", elements, geometry, samples, rate_hz)


def ntc7_combination(samples: int = SAMPLES_PER_LINE,
                     rate_hz: float = SAMPLE_RATE_HZ
                     ) -> Dict[str, object]:
    """The NTC-7 combination line - ITU-T J.63 Annex II section 3, line 280.

    | element | what it measures |
    |---|---|
    | C1, the 4 us white flag at 100 IRE | insertion gain, on a line whose rest sits at a 50 IRE pedestal |
    | C2, six multiburst packets at 0.5, 1.0, 2.0, 3.0, 3.58 and 4.2 MHz, each 50 IRE p-p on a 50 IRE pedestal | amplitude/frequency response, one packet per frequency |
    | G, the three-level modulated pedestal at 20, 40 and 80 IRE p-p, 90 degrees | chrominance non-linearity, chrominance-luminance intermodulation |

    THE MULTIBURST IS WHY THIS IS THE EFFICIENT LINE. Measured, 10.53
    effective directions of 26 readings against the composite's 5.61 of 27 -
    0.405 per reading against 0.208, and a condition of 123 against
    12 000 - and for the reason
    `interference.py`'s header gives about echoes: each packet interrogates
    the response at its own frequency and at no other, so six packets are
    close to six independent numbers, where the composite's four luminance
    elements are four views of one.

    The envelope rise is 140 ns on the first two packets and 400 ns on the
    rest, which is the TSG-130A's own figure for the pattern and is what
    keeps the 0.5 MHz packet's envelope from being wider than the packet.
    """
    t = _line_grid(samples, rate_hz)
    elements: Dict[str, Dict[str, object]] = {}

    pedestal_start, pedestal_stop = _at(12.0), _at(60.0)
    elements["pedestal"] = {
        "luma": bar(t, pedestal_start, pedestal_stop, 50.0,
                    LUMINANCE_RISE_S),
        "window_s": (pedestal_start, pedestal_stop),
        "centre_s": 0.5 * (pedestal_start + pedestal_stop),
        "band_hz": 1.0 / LUMINANCE_RISE_S,
    }
    flag_start, flag_stop = _at(12.0), _at(16.0)
    elements["C1 white flag"] = {
        # the flag is the pedestal lifted to 100 IRE, so it enters as the
        # 50 IRE difference and the two sum to the specified level
        "luma": bar(t, flag_start, flag_stop, 50.0, LUMINANCE_RISE_S),
        "window_s": (flag_start, flag_stop),
        "centre_s": 0.5 * (flag_start + flag_stop),
        "band_hz": 1.0 / LUMINANCE_RISE_S,
    }

    # C2, ITU-T J.63 Annex II table III: the 525-line packet set is
    # 0.5 / 1.0 / 2.0 / 3.0 / 3.58 / 4.2 MHz. (The 4.1 MHz set circulating
    # as "FCC multiburst" has no authoritative source and is not used.)
    packets = ((0.5e6, 140e-9), (1.0e6, 140e-9), (2.0e6, 400e-9),
               (3.0e6, 400e-9), (SUBCARRIER_HZ, 400e-9), (4.2e6, 400e-9))
    spans = ((18.0, 23.0), (24.0, 27.0), (28.0, 31.0), (32.0, 35.0),
             (36.0, 39.0), (40.0, 43.0))
    packet_geometry = []
    for (frequency, rise), (start_us, stop_us) in zip(packets, spans):
        start, stop = _at(start_us), _at(stop_us)
        name = f"C2 packet {frequency / 1e6:.2f} MHz"
        elements[name] = {
            "luma": packet(t, start, stop, frequency, 50.0 / 2.0, rise),
            "window_s": (start, stop),
            "centre_s": 0.5 * (start + stop),
            "band_hz": frequency,
        }
        packet_geometry.append((name, frequency, start, stop))

    # G, the three-level modulated pedestal: 20, 40 and 80 IRE peak to peak
    # at 90 degrees, on the same 50 IRE pedestal.
    zones = ((20.0, 46.0, 50.0), (40.0, 50.0, 54.0), (80.0, 54.0, 60.0))
    zone_geometry = []
    for amplitude, start_us, stop_us in zones:
        start, stop = _at(start_us), _at(stop_us)
        name = f"G chrominance {amplitude:.0f} IRE p-p"
        elements[name] = {
            "chroma": (amplitude / 2.0) * bar(t, start, stop, 1.0,
                                              CHROMINANCE_RISE_S)
            * subcarrier(t, 90.0),
            "window_s": (start, stop),
            "centre_s": 0.5 * (start + stop),
            "band_hz": SUBCARRIER_HZ,
        }
        zone_geometry.append((name, amplitude, start, stop))

    geometry = {
        "flag": (flag_start, flag_stop),
        "pedestal": (pedestal_start, pedestal_stop),
        # the pedestal is read where nothing rides on it: after the last
        # packet and before the first chrominance zone
        "pedestal_window": (packet_geometry[-1][3], zone_geometry[0][2]),
        "pedestal_ire": 50.0,
        "packets": packet_geometry,
        "chroma_zones": zone_geometry,
    }
    return _line("NTC-7 combination",
                 "ITU-T J.63 (06/90) Annex II section 3; packet set "
                 "Annex II table III", elements, geometry, samples, rate_hz)


# SMPTE EG 27-2004 table 2, the 75/7.5/75/7.5 set, restated to four decimal
# places as SMPTE 170M-2004 Annex A table A.3. Luminance and chrominance in
# IRE, phase in degrees from the +(B-Y) axis.
#
# THE PHASES ARE IDENTICAL BETWEEN THE 75 AND 100 PER CENT SETS - saturation
# scales the amplitude and not the angle - so the phase column is a
# saturation-independent reference.
COLOUR_BARS_75: Tuple[Tuple[str, float, float, Optional[float]], ...] = (
    ("grey",    76.9, 0.0, None),
    ("yellow",  69.0, 62.1, 167.1),
    ("cyan",    56.1, 87.7, 283.5),
    ("green",   48.2, 81.9, 240.7),
    ("magenta", 36.2, 81.9, 60.7),
    ("red",     28.2, 87.7, 103.5),
    ("blue",    15.4, 62.1, 347.1),
    ("black",    7.5, 0.0, None),
)


def colour_bars(samples: int = SAMPLES_PER_LINE,
                rate_hz: float = SAMPLE_RATE_HZ,
                table: Sequence[Tuple[str, float, float, Optional[float]]]
                = COLOUR_BARS_75) -> Dict[str, object]:
    """The 75/7.5/75/7.5 EIA colour bars - SMPTE EG 27-2004 table 2.

    EIGHT EXACTLY-KNOWN LUMINANCE LEVELS AND SIX EXACTLY-KNOWN SUBCARRIER
    PHASES ON ONE LINE. That is what makes this line the amplitude axis's
    answer: `docs/ELLIPTICAL_COLLAPSE.md` section 6a records the amplitude
    axis as saturated for want of places, nine sweep bins against ninety-one
    entries, and those nine bins were binned out of a fitted sweep while
    these eight levels are printed in a standard. `amplitude_axis` measures
    what they are worth.

    The bars occupy equal spans of the active line, which is the layout
    convention and an ASSUMPTION; the levels, the chrominance amplitudes
    and the phases are the standard's.
    """
    t = _line_grid(samples, rate_hz)
    elements: Dict[str, Dict[str, object]] = {}
    first, last = _at(12.0), _at(60.0)
    width = (last - first) / len(table)
    geometry: List[Tuple[str, float, float, Optional[float], float,
                         float]] = []
    for index, (name, luminance, chroma_pp, phase) in enumerate(table):
        start = first + index * width
        stop = start + width
        entry: Dict[str, object] = {
            "luma": bar(t, start, stop, luminance, LUMINANCE_RISE_S),
            "window_s": (start, stop),
            "centre_s": 0.5 * (start + stop),
            "band_hz": 1.0 / LUMINANCE_RISE_S,
        }
        if chroma_pp > 0.0 and phase is not None:
            entry["chroma"] = ((chroma_pp / 2.0)
                               * bar(t, start, stop, 1.0, CHROMINANCE_RISE_S)
                               * subcarrier(t, phase))
        elements[f"bar {name}"] = entry
        geometry.append((name, luminance, chroma_pp, phase, start, stop))
    return _line("75/7.5/75/7.5 colour bars",
                 "SMPTE EG 27-2004 table 2; SMPTE 170M-2004 Annex A "
                 "table A.3 to 1e-4 IRE",
                 elements, {"bars": geometry}, samples, rate_hz)


SIGNALS: Dict[str, Callable[..., Dict[str, object]]] = {
    "NTC-7 composite": ntc7_composite,
    "NTC-7 combination": ntc7_combination,
    "75% colour bars": colour_bars,
}


# ==========================================================================
# Reading a rendered line. Every pulse is read at its OWN half amplitude,
# with sub-sample interpolation.
# ==========================================================================


def oversample(values: np.ndarray, factor: int = 4) -> np.ndarray:
    """Band-limited interpolation by an integer factor.

    4fsc Nyquist is 7.16 MHz and everything in a 525-line composite signal
    is band-limited well below it, so the information is on the grid and
    only the reading loses it - see `read_pulse` for how much.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    factor = max(int(factor), 1)
    if factor == 1:
        return values
    length = values.size
    spectrum = np.fft.rfft(values)
    if length % 2 == 0 and spectrum.size:
        spectrum = spectrum.copy()
        spectrum[-1] *= 0.5
    padded = np.zeros(length * factor // 2 + 1, dtype=np.complex128)
    padded[:spectrum.size] = spectrum
    return np.fft.irfft(padded, length * factor) * factor


# A reading window is the pulse's own support plus this margin at each end,
# and the margin is not free. `half_amplitude_crossings` takes its baseline
# from the 90th percentile and its extreme from the 5th, so the window must
# hold more than a tenth of its samples at the baseline for the first to
# land there, and as few as possible for the second to land on the peak.
# Twenty per cent each side satisfies both: the baseline is a sixth of the
# window, and the fifth percentile falls at 0.991 of the peak of a
# sine-squared pulse, which puts the half-amplitude threshold half a per
# cent high and reads a 2T pulse 2.3 ns wide - under a quarter of the
# standard's +-10 ns tolerance, and a constant of the reading rather than a
# property of the signal, so it cancels in a differential.
PULSE_WINDOW_MARGIN = 0.2


def read_pulse(values: np.ndarray, rate_hz: float,
               window_s: Tuple[float, float], rising: bool = True,
               interpolate: int = 8) -> Dict[str, float]:
    """A pulse's amplitude, half-amplitude duration and rise, read properly.

    The width and the rise come from
    `standard_levels.half_amplitude_crossings`, which reads at the pulse's
    own half amplitude with sub-sample interpolation, given a band-limited
    interpolation of the window first. A positive-going pulse is negated
    before the call, because that function is written for a pulse below its
    baseline. The amplitude is taken from the window's own extreme against
    that function's baseline percentile, because the fifth percentile of a
    sine-squared pulse sits at 0.991 of its peak and is a poor amplitude
    even where it is an excellent threshold.

    MEASURED, ON THE IDEAL 2T PULSE THIS MODULE RENDERS:

        the specification (J.63 Annex II 2.2)  250.00 ns  100.000 IRE
        4fsc grid, nearest sample, fixed IRE   279.37 ns   98.328 IRE
        4fsc data, band-limited x4, own half   252.56 ns  100.160 IRE
        4fsc data, band-limited x8, own half   252.33 ns  100.235 IRE

    A gauge that reads the sample grid at a fixed threshold reports the
    pulse 11.7 per cent wide, which is nearly three times the standard's
    whole +-10 ns tolerance, from arithmetic alone. The 0.9 per cent that
    remains after interpolation is this reader's own floor and is stated in
    `PULSE_WINDOW_MARGIN`; it is a constant of the reading, so it cancels in
    a differential, which is the only way any of it is used.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    fine = oversample(values, interpolate)
    fine_rate = float(rate_hz) * max(int(interpolate), 1)
    low = int(round(float(window_s[0]) * fine_rate))
    high = int(round(float(window_s[1]) * fine_rate))
    low, high = max(low, 0), min(high, fine.size)
    window = fine[low:high]
    if window.size < 4:
        return {"peak": 0.0, "plateau": 0.0, "width_s": 0.0, "rise_s": 0.0,
                "baseline": 0.0}
    edges = standard_levels.half_amplitude_crossings(
        -window if rising else window, fine_rate)
    baseline = -float(edges["top"]) if rising else float(edges["top"])
    extreme = float(np.max(window)) if rising else float(np.min(window))
    # TWO AMPLITUDES, AND WHICH ONE IS RIGHT DEPENDS ON THE PULSE. A pulse
    # with a flat tip - sync, a bar - has its level AT the plateau, and the
    # fifth percentile finds it robustly where the extreme picks up the
    # interpolation's own overshoot. A pulse with no plateau - a 2T pulse is
    # a single sine-squared lobe - has its level at its peak, and there the
    # percentile reads 0.9 per cent low by construction.
    return {"peak": abs(extreme - baseline),
            "plateau": abs(float(edges["depth"])),
            "width_s": float(edges["width_s"]),
            "rise_s": float(edges["rise_s"]),
            "baseline": baseline}


def _pulse_window(centre_s: float, had_s: float) -> Tuple[float, float]:
    """The reading window for a sine-squared pulse of this duration."""
    half = float(had_s) * (1.0 + PULSE_WINDOW_MARGIN)
    return (float(centre_s) - half, float(centre_s) + half)


# The threshold a defective gauge uses: a level fixed in absolute IRE. It is
# the number `docs/VERTICAL_INTERVAL_COLLAPSE.md` section 8.2 records
# reading a decoded sync 111 ns narrow, outside the generator's tolerance,
# where the pulse's own half amplitude read it 44 ns and inside.
NAIVE_THRESHOLD_IRE = 20.0


def naive_read_pulse(values: np.ndarray, rate_hz: float,
                     window_s: Tuple[float, float], rising: bool = True,
                     interpolate: int = 8) -> Dict[str, float]:
    """A DELIBERATELY DEFECTIVE PULSE READING, kept as code so that the
    control can be shown to fail.

    Two defects, and this arc has made both. The width is read at a
    threshold fixed in absolute IRE, which stops being a width measurement
    the moment the amplitude has changed - the condition every tape decode
    is in. And every crossing is taken to the nearest sample of the 4fsc
    grid, which quantises to 69.8 ns, more than a quarter of a 2T pulse's
    whole half-amplitude duration.

    Its signature matches `read_pulse` so it can be substituted for it, and
    nothing in this module calls it except `naive_readings`.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    low = max(int(round(float(window_s[0]) * rate_hz)), 0)
    high = min(int(round(float(window_s[1]) * rate_hz)), values.size)
    window = values[low:high]
    if window.size < 4:
        return {"peak": 0.0, "plateau": 0.0, "width_s": 0.0, "rise_s": 0.0,
                "baseline": 0.0}
    sign = 1.0 if rising else -1.0
    baseline = float(np.percentile(sign * window, 10.0))
    extreme = float(np.max(sign * window))
    depth = extreme - baseline

    def nearest(level: float) -> Tuple[float, float]:
        above = np.flatnonzero(sign * window > level)
        if above.size == 0:
            return (0.0, 0.0)
        return (float(above[0]) / rate_hz, float(above[-1]) / rate_hz)

    first, last = nearest(baseline + NAIVE_THRESHOLD_IRE)
    ten_first, _ = nearest(baseline + 0.1 * depth)
    ninety_first, _ = nearest(baseline + 0.9 * depth)
    return {"peak": abs(depth), "plateau": abs(depth),
            "width_s": abs(last - first),
            "rise_s": abs(ninety_first - ten_first),
            "baseline": sign * baseline}


# AN INSTRUMENT'S MEASUREMENT BAND MUST BE SOFT, AND THIS IS NOT A MATTER
# OF TASTE. A brick-wall band has a sinc impulse response whose tails decay
# as one over time and therefore reach across the whole line, so a 100 IRE
# flag leaks into a pedestal reading four microseconds away and one
# multiburst packet leaks into the next. A Gaussian band's response decays
# as a Gaussian: at the two microseconds that separate adjacent packets the
# leakage is sixty thousand times down. `through_colour_under` deliberately
# keeps the hard edge, because there it is modelling a channel and not
# reading one.
def _shape(frequency: np.ndarray, centre_hz: float, half_hz: float,
           soft: bool) -> np.ndarray:
    if not soft:
        return (np.abs(frequency - centre_hz) <= half_hz).astype(np.float64)
    return np.exp(-np.log(2.0)
                  * ((frequency - centre_hz) / max(half_hz, 1.0)) ** 2)


def _lowpass(values: np.ndarray, rate_hz: float, cutoff_hz: float,
             soft: bool = True) -> np.ndarray:
    frequency = np.fft.rfftfreq(values.size, d=1.0 / rate_hz)
    keep = (_shape(frequency, 0.0, cutoff_hz, True) if soft
            else (frequency <= cutoff_hz).astype(np.float64))
    return np.fft.irfft(np.fft.rfft(values) * keep, values.size)


def _bandpass(values: np.ndarray, rate_hz: float, centre_hz: float,
              half_hz: float, soft: bool = True) -> np.ndarray:
    frequency = np.fft.rfftfreq(values.size, d=1.0 / rate_hz)
    return np.fft.irfft(np.fft.rfft(values)
                        * _shape(frequency, centre_hz, half_hz, soft),
                        values.size)


def _notch(values: np.ndarray, rate_hz: float, centre_hz: float,
           half_hz: float, soft: bool = True) -> np.ndarray:
    frequency = np.fft.rfftfreq(values.size, d=1.0 / rate_hz)
    keep = 1.0 - _shape(frequency, centre_hz, half_hz, soft)
    return np.fft.irfft(np.fft.rfft(values) * keep, values.size)


def _analytic(values: np.ndarray) -> np.ndarray:
    length = values.size
    spectrum = np.fft.fft(values)
    weight = np.zeros(length)
    weight[0] = 1.0
    if length % 2 == 0:
        weight[length // 2] = 1.0
        weight[1:length // 2] = 2.0
    else:
        weight[1:(length + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * weight)


def _mean_between(t_s, values, start_s, stop_s, guard_s=0.0) -> float:
    inside = ((t_s >= start_s + guard_s) & (t_s <= stop_s - guard_s))
    return float(np.mean(values[inside])) if np.any(inside) else 0.0


def readings(line: Dict[str, object], values: Optional[np.ndarray] = None,
             interpolate: int = 4,
             pulse_reader: Callable[..., Dict[str, float]] = read_pulse
             ) -> Dict[str, float]:
    """Every quantity the line's own elements name, read as an instrument
    reads it.

    Each entry is a number an operator writes down. What matters for the
    ellipse is not the number but how it MOVES when the channel changes:
    two readings that respond identically to every departure are one
    measurement wearing two names, which is what `line_dimensions` counts.
    """
    t = np.asarray(line["time_s"], dtype=np.float64)
    rate = float(line["sample_rate_hz"])
    signal = (np.asarray(line["composite_ire"], dtype=np.float64)
              if values is None else np.asarray(values, dtype=np.float64))
    geometry = line["geometry"]
    out: Dict[str, float] = {}

    # the luminance an instrument sees: the subcarrier notched out. NOT a
    # low-pass - a 2T pulse reaches 4 MHz and a low-pass destroys it.
    luma = _notch(signal, rate, SUBCARRIER_HZ, VHS_CHROMA_HALF_HZ + 0.2e6)
    chroma = _bandpass(signal, rate, SUBCARRIER_HZ, 1.2e6)
    envelope = np.abs(_analytic(chroma))
    reference = np.exp(-2j * np.pi * SUBCARRIER_HZ * t)
    turned = _analytic(chroma) * reference

    # --- present on every line: the sync pulse and the burst, which the
    # standing sync-only constraint already permits. The test elements are
    # judged on what they add BEYOND these.
    blanking = _mean_between(t, luma, _at(8.0), _at(10.0))
    out["blanking level"] = blanking
    # the sync window runs from the line's start to just before the burst.
    # It must exclude the burst, because a subcarrier inside the window
    # moves both percentiles, and it must be read on the COMPOSITE, because
    # notching the subcarrier out of a 140 ns edge slows it tenfold.
    begin = _LAYOUT_ORIGIN_S + BURST_START_CYCLES / SUBCARRIER_HZ
    sync = pulse_reader(signal, rate, (0.0, begin - CHROMINANCE_RISE_S),
                        rising=False, interpolate=interpolate)
    out["sync amplitude"] = -sync["plateau"]
    out["sync width"] = sync["width_s"]
    out["sync rise"] = sync["rise_s"]
    inside = (t >= begin + CHROMINANCE_RISE_S) & (
        t <= begin + BURST_CYCLES / SUBCARRIER_HZ - CHROMINANCE_RISE_S)
    out["burst amplitude"] = float(np.mean(envelope[inside])) * 2.0
    out["burst phase"] = float(np.angle(np.mean(turned[inside])))

    # --- the luminance bar, where the line has one. READ ON THE COMPOSITE
    # AND NOT ON THE NOTCHED LUMINANCE: a bar with a 125 ns edge and a 2T
    # pulse both reach 4 MHz, so notching 3.0 to 4.2 MHz out of them
    # broadens the pulse by 8 per cent and slows the edge by a third. An
    # instrument reads these elements on the waveform, and it is right to.
    span = geometry.get("bar") or geometry.get("flag")
    if span is not None:
        start, stop = span
        length = stop - start
        guard = 0.15 * length
        out["bar amplitude"] = _mean_between(t, signal, start, stop,
                                             guard) - blanking
        head = _mean_between(t, signal, start + guard,
                             start + length / 3.0)
        tail = _mean_between(t, signal, stop - length / 3.0, stop - guard)
        out["bar tilt"] = head - tail
        # the edge window must clear the burst: the burst swings +-20 IRE,
        # so a 10 per cent crossing found inside it measures the burst and
        # returns a rise time thirty times too slow
        after_burst = begin + BURST_CYCLES / SUBCARRIER_HZ + CHROMINANCE_RISE_S
        edge = pulse_reader(signal, rate,
                            (max(start - 0.25 * length, after_burst),
                             start + 0.75 * length),
                            rising=True, interpolate=interpolate)
        out["bar rise"] = edge["rise_s"]
        after = (t > start) & (t < start + 0.4 * length)
        flat = _mean_between(t, signal, start + guard, stop - guard)
        out["bar overshoot"] = (float(np.max(signal[after])) - flat
                                if np.any(after) else 0.0)

    pulse = geometry.get("narrow_pulse")
    if pulse is not None:
        centre, had = pulse
        read = pulse_reader(signal, rate, _pulse_window(centre, had),
                            rising=True, interpolate=interpolate)
        out["2T pulse amplitude"] = read["peak"]
        out["2T pulse HAD"] = read["width_s"]
        out["2T pulse rise"] = read["rise_s"]
        # the aftermath, stopping short of the next element
        after = (t > centre + 1.2 * had) & (t < centre + 4.0 * had)
        out["2T pulse ringing"] = float(np.std(signal[after] - blanking))

    pulse = geometry.get("wide_pulse")
    if pulse is not None:
        centre, had = pulse
        # THE TWO HALVES OF A MODULATED PULSE ARE SEPARATED BY THE
        # SUBCARRIER BAND AND NOT BY A LOW-PASS. The luminance half is what
        # remains once the chrominance band is taken out; a low-pass wide
        # enough to pass a 1.5625 us envelope also passes the pulse's own
        # subcarrier sidebands, and one narrow enough to stop them takes 5
        # per cent off the envelope it was meant to measure.
        near = (t > centre - had) & (t < centre + had)
        centred = (t > centre - 0.2 * had) & (t < centre + 0.2 * had)
        # The luminance half is read as a pulse, interpolated and at its own
        # half amplitude, on the notched waveform - the envelope of a 12.5T
        # pulse has no energy at the subcarrier, so the notch does not touch
        # it. Averaging over a window instead reads a cos-squared peak 2 per
        # cent low, which is the window's average and not the pulse.
        luminance = signal - chroma
        wide_read = pulse_reader(luminance, rate,
                                 _pulse_window(centre, had), rising=True,
                                 interpolate=interpolate)
        out["12.5T luminance"] = wide_read["peak"]
        out["12.5T chrominance"] = 2.0 * float(np.max(envelope[centred]))
        # the base line of a modulated pulse: a chrominance-luminance GAIN
        # inequality lifts or drops it, a DELAY inequality tips it
        lobe = (luminance[near] - blanking) - envelope[near]
        out["12.5T base line (C/Y gain)"] = float(np.mean(lobe))
        half = lobe.size // 2
        out["12.5T base line tilt (C/Y delay)"] = float(
            np.mean(lobe[:half]) - np.mean(lobe[half:]))

    treads = geometry.get("treads")
    if treads:
        levels, gains, phases = [], [], []
        for index, (start, stop, _level) in enumerate(treads):
            inside = (t > start + 0.6e-6) & (t < stop - 0.6e-6)
            if not np.any(inside):
                continue
            levels.append(float(np.mean(luma[inside])) - blanking)
            gains.append(float(np.mean(envelope[inside])))
            phases.append(float(np.angle(np.mean(turned[inside]))))
            out[f"staircase tread {index + 1}"] = levels[-1]
        levels = np.asarray(levels)
        gains = np.asarray(gains)
        phases = np.unwrap(np.asarray(phases))
        if levels.size >= 3:
            risers = np.diff(levels)
            out["luminance non-linearity"] = float(risers.max()
                                                   - risers.min())
        if gains.size >= 3 and gains.mean() > 0:
            out["differential gain"] = float((gains.max() - gains.min())
                                             / gains.mean())
            out["staircase chrominance"] = float(2.0 * gains.mean())
        if phases.size >= 3:
            out["differential phase"] = float(np.rad2deg(phases.max()
                                                         - phases.min()))

    packets = geometry.get("packets")
    if packets:
        quiet_start, quiet_stop = geometry["pedestal_window"]
        out["pedestal level"] = _mean_between(
            t, luma, quiet_start, quiet_stop, 0.5e-6) - blanking
        for name, frequency, start, stop in packets:
            inside = (t > start + 0.9e-6) & (t < stop - 0.9e-6)
            if inside.sum() < 8:
                continue
            band = _bandpass(signal, rate, frequency, 0.35e6)
            # the packet's own envelope, not its standard deviation: a
            # sine-squared envelope inside a +-350 kHz measurement band
            # makes the second moment read several IRE high
            out[f"{name} amplitude"] = 2.0 * float(
                np.mean(np.abs(_analytic(band))[inside]))
            out[f"{name} phase"] = float(np.angle(np.mean(
                _analytic(band)[inside]
                * np.exp(-2j * np.pi * frequency * t[inside]))))

    zones = geometry.get("chroma_zones")
    if zones:
        for name, _amplitude, start, stop in zones:
            inside = (t > start + CHROMINANCE_RISE_S * 2.0) & (
                t < stop - CHROMINANCE_RISE_S * 2.0)
            if not np.any(inside):
                continue
            out[f"{name} amplitude"] = 2.0 * float(np.mean(envelope[inside]))

    bars = geometry.get("bars")
    if bars:
        for name, _luminance, chroma_pp, phase, start, stop in bars:
            inside = (t > start + 0.8e-6) & (t < stop - 0.8e-6)
            if not np.any(inside):
                continue
            out[f"bar {name} luminance"] = (float(np.mean(luma[inside]))
                                            - blanking)
            if chroma_pp > 0.0 and phase is not None:
                out[f"bar {name} chrominance"] = 2.0 * float(
                    np.mean(envelope[inside]))
                out[f"bar {name} phase"] = float(np.rad2deg(np.angle(
                    np.mean(turned[inside]))))
    return out


# ==========================================================================
# The key entries: the specified spectra, in subtractable form.
# ==========================================================================

# WHERE THESE SIT IN THE CHAIN. `interference.COMPONENT_ORDER` counts a
# position from the source: the transmission path is 1 to 4, the recording
# machine 10 and 11, the tape 20 to 23, the capture 30. An insertion test
# signal is inserted at the SOURCE, before the transmission path and long
# before the recorder saw anything, so it sits at the low-numbered end -
# position 0, which is before everything already declared.
#
# Every entry this module produces is named with the prefix below, so
# `interference.position_of` resolves all of them from one declaration.
CHAIN_PREFIX = "specified test signal"
CHAIN_POSITION = 0

# All of them share one position because they act at the same point and
# trivially commute: they are stimuli rather than modifications, so no one
# of them acts on another.
COMPONENT_ORDER: Dict[str, int] = {CHAIN_PREFIX: CHAIN_POSITION}

# How far the bounded form departs from unity. Any value below one keeps the
# magnitude away from zero, so the logarithm is finite everywhere and
# `interference.subtractable` returns True; this is the same construction
# `interference.beat` and `interference.echo` use, and for the same reason.
SIGNATURE_DEPTH = 0.5


def _spectrum_at(t_s: np.ndarray, values: np.ndarray, frequency_hz,
                 centre_s: float) -> np.ndarray:
    """The element's own spectrum on an arbitrary frequency grid.

    A direct transform rather than an FFT, because the caller's grid is
    whatever band it is asking about and need not be the line's own bin
    spacing. Referenced to the element's centre, so the phase carried is the
    element's INTERNAL structure - a modulated pulse's subcarrier angle, a
    packet's own carrier - and not the delay implied by where the layout
    convention happens to place it.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    live = np.abs(values) > 0.0
    if not np.any(live):
        return np.zeros(f.size, dtype=np.complex128)
    time = t_s[live] - float(centre_s)
    kernel = np.exp(-2j * np.pi * np.outer(f, time))
    return kernel @ values[live].astype(np.complex128)


def element_signatures(frequency_hz, line: Dict[str, object],
                       depth: float = SIGNATURE_DEPTH
                       ) -> Dict[str, np.ndarray]:
    """Every element of one line as a subtractable complex signature.

        W_i(f) = 1 + a * S_i(f) / max|S_i|

    with `S_i` the element's prescribed spectrum. Bounded into
    `[1 - a, 1 + a]`, so it has a finite logarithm everywhere and can be
    subtracted; complex, so the real part carries what the element does to
    amplitude and the imaginary part what it does to phase.

    MEASURED over 1 to 7 MHz on 512 places: 25 elements across the three
    lines, every one subtractable, magnitudes 0.516 to 1.500. Per line, and
    the conditioning is as instructive as the count:

        NTC-7 composite      6 elements    5.71 effective   condition 1.4
        NTC-7 combination   11 elements    7.82 effective   condition 7.2
        75% colour bars      8 elements    2.87 effective   condition 688

    The bars collapse because eight bars are eight copies of one spectrum at
    eight levels, and a level is not a shape. The composite's six elements
    are nearly orthogonal as SPECTRA even though its readings are not,
    which is the distinction `line_dimensions` exists to make: what the
    elements occupy and what the conventional readings can determine are
    different questions with different answers.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    t = np.asarray(line["time_s"], dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for name, entry in line["elements"].items():
        values = np.zeros_like(t)
        for part in ("luma", "chroma"):
            piece = entry.get(part)
            if piece is not None:
                values = values + np.asarray(piece, dtype=np.float64)
        spectrum = _spectrum_at(t, values, f, float(entry["centre_s"]))
        peak = float(np.max(np.abs(spectrum)))
        if not peak > 0.0:
            continue
        shape = spectrum / peak
        out[f"{CHAIN_PREFIX}: {line['name']}, {name}"] = (
            1.0 + float(depth) * shape).astype(np.complex128)
    return out


def signatures(frequency_hz, signals: Optional[Sequence[str]] = None,
               depth: float = SIGNATURE_DEPTH) -> Dict[str, np.ndarray]:
    """Every specified element of every specified line, on one grid.

    Feed the result to `ellipsoid(..., real_parameters=True)` to read how
    many the band can tell apart, or register it beside
    `interference.signatures` as the source-side entries of the key.
    """
    names = list(SIGNALS) if signals is None else list(signals)
    out: Dict[str, np.ndarray] = {}
    for name in names:
        builder = SIGNALS.get(name)
        if builder is None:
            raise ValueError(f"no specified line named {name!r}")
        out.update(element_signatures(frequency_hz, builder(), depth))
    return out


def amplitude_signatures(table: Sequence[Tuple[str, float, float,
                                               Optional[float]]]
                         = COLOUR_BARS_75,
                         depth: float = SIGNATURE_DEPTH
                         ) -> Dict[str, np.ndarray]:
    """The colour bars as a component ON THE AMPLITUDE AXIS.

    The places are the eight specified luminance levels and the entries are
    the three quantities the standard fixes at each of them: the level
    itself, the chrominance amplitude that rides on it, and that
    chrominance's subcarrier phase. Complex throughout - the real part is
    the amplitude and the imaginary part the phase - and bounded into
    `[1 - a, 1 + a]` so every entry is subtractable.

    This is the amplitude axis's synthetic side, and it needs no fitting at
    all. A sweep's bins are estimated from the data they are then judged on;
    these eight numbers are printed in SMPTE EG 27-2004 table 2.
    """
    levels = np.array([row[1] for row in table], dtype=np.float64)
    chroma = np.array([row[2] for row in table], dtype=np.float64)
    phase = np.array([0.0 if row[3] is None else row[3] for row in table],
                     dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    scale = float(np.max(np.abs(levels))) or 1.0
    out[f"{CHAIN_PREFIX}: 75% bars, luminance level"] = (
        1.0 + depth * (levels / scale)).astype(np.complex128)
    scale = float(np.max(np.abs(chroma))) or 1.0
    out[f"{CHAIN_PREFIX}: 75% bars, chrominance amplitude"] = (
        1.0 + depth * (chroma / scale)).astype(np.complex128)
    # the phase entry is a unit phasor scaled by whether there is any
    # chrominance to carry an angle at all
    present = (chroma > 0.0).astype(np.float64)
    out[f"{CHAIN_PREFIX}: 75% bars, subcarrier phase"] = (
        1.0 + depth * present * np.exp(1j * np.deg2rad(phase))
    ).astype(np.complex128)
    return out


# ==========================================================================
# How many independent dimensions a line actually provides.
# ==========================================================================


def departure_basis(modes: int = 20, level_orders: int = 4) -> List[str]:
    """The channel departures a line is interrogated with.

    Two families, and a line's worth is how many of them it can tell apart.

    **Over frequency**, cosine modes across 0 to 4.2 MHz - the band a
    525-line composite signal occupies - in amplitude and in phase
    separately, because a gain and a delay are different mechanisms. The two
    are PAIRED into one complex place per mode, so the real part of a
    reading's signature is its sensitivity to a gain and the imaginary part
    its sensitivity to a delay.

    **Over level**, Chebyshev polynomials of the luminance the chrominance
    rides on, acting on the chrominance gain, on the chrominance phase and
    on the luminance itself. This is the amplitude axis, and it needs more
    than the one mode each that "differential gain" and "differential
    phase" amount to: a five-step staircase can determine four orders of a
    level-dependent law and a colour bar line eight, and a basis with one
    mode cannot show the difference.
    """
    names = ([f"|H| mode {k}" for k in range(int(modes))]
             + [f"arg H mode {k}" for k in range(int(modes))])
    for order in range(1, int(level_orders) + 1):
        names.append(f"level gain order {order}")
    for order in range(1, int(level_orders) + 1):
        names.append(f"level phase order {order}")
    for order in range(1, int(level_orders) + 1):
        names.append(f"level luminance order {order}")
    return names


def apply_departure(line: Dict[str, object], values: np.ndarray, name: str,
                    epsilon: float, band_hz: float = 4.2e6) -> np.ndarray:
    """Disturb the channel along one departure and return the line."""
    values = np.asarray(values, dtype=np.float64)
    rate = float(line["sample_rate_hz"])
    length = values.size
    if name.startswith("|H| mode") or name.startswith("arg H mode"):
        index = int(name.split()[-1])
        frequency = np.fft.rfftfreq(length, d=1.0 / rate)
        shape = np.cos(np.pi * index * np.clip(frequency / band_hz, 0.0,
                                               1.0))
        shape = shape * (frequency <= band_hz)
        spectrum = np.fft.rfft(values)
        if name.startswith("|H|"):
            spectrum = spectrum * (1.0 + epsilon * shape)
        else:
            spectrum = spectrum * np.exp(1j * epsilon * shape)
        return np.fft.irfft(spectrum, length)
    # the level-dependent departures act on the chrominance as a function
    # of the luminance it rides on, which is what a modulated staircase and
    # a colour bar line are built to measure. The level runs from the sync
    # tip to reference white, so it is the standard's own axis and not a
    # normalisation of this particular waveform.
    order = int(name.split()[-1])
    low = _lowpass(values, rate, 1.0e6)
    level = np.clip((low - SYNC_DEPTH_IRE)
                    / (100.0 - SYNC_DEPTH_IRE), 0.0, 1.0)
    shape = np.cos(order * np.arccos(np.clip(2.0 * level - 1.0, -1.0, 1.0)))
    if name.startswith("level luminance"):
        return values + epsilon * 100.0 * shape
    chroma = _bandpass(values, rate, SUBCARRIER_HZ, 1.2e6)
    rest = values - chroma
    if name.startswith("level gain"):
        return rest + chroma * (1.0 + epsilon * shape)
    if name.startswith("level phase"):
        turned = _analytic(chroma) * np.exp(1j * epsilon * shape)
        return rest + np.real(turned)
    return values


# A ROW IS DEAD ONLY AGAINST ITS OWN SIZE. Readings are in different units
# - IRE, seconds, degrees - so comparing their row norms to one another
# drops every duration reading on a video line for being small in seconds,
# which is a units error and not a measurement. The floor below is the
# numerical resolution of a central difference of a reading of magnitude
# `|base|`: about the machine epsilon on the reading, divided by the step.
# The thousand-fold allowance covers the interpolation's own error.
DEAD_ROW_FLOOR = 1e3 * float(np.finfo(np.float64).eps)


def reading_jacobian(line: Dict[str, object], modes: int = 20,
                     epsilon: float = 1e-3,
                     channel: Optional[Callable[[np.ndarray], np.ndarray]]
                     = None, interpolate: int = 4,
                     level_orders: int = 4
                     ) -> Tuple[Dict[str, np.ndarray], List[str],
                                Dict[str, float]]:
    """How every reading moves when the channel departs.

    Central differences, so the leading error is second order in `epsilon`
    and a reading that is smooth in the departure returns its derivative.
    Each row is that reading's SIGNATURE over the departure basis, complex:
    the gain modes in the real part and the delay modes in the imaginary
    part, at each place - which is the convention every signature in this
    arc uses.
    """
    through = (lambda v: v) if channel is None else channel
    base = np.asarray(line["composite_ire"], dtype=np.float64)
    reference = readings(line, through(base), interpolate)
    names = departure_basis(modes, level_orders)
    rows: Dict[str, List[float]] = {name: [] for name in reference}
    for mode in names:
        plus = readings(line, through(apply_departure(line, base, mode,
                                                      +epsilon)), interpolate)
        minus = readings(line, through(apply_departure(line, base, mode,
                                                       -epsilon)),
                         interpolate)
        for name in reference:
            rows[name].append((plus.get(name, 0.0) - minus.get(name, 0.0))
                              / (2.0 * epsilon))
    count = int(modes)
    orders = int(level_orders)
    packed: Dict[str, np.ndarray] = {}
    floors: Dict[str, float] = {}
    for name, row in rows.items():
        row = np.asarray(row, dtype=np.float64)
        # frequency: gain paired with delay; level: gain paired with phase;
        # the level-luminance modes have no phase partner and stay real
        frequency = row[:count] + 1j * row[count:2 * count]
        gain = row[2 * count:2 * count + orders]
        turn = row[2 * count + orders:2 * count + 2 * orders]
        luminance = row[2 * count + 2 * orders:]
        packed[name] = np.concatenate([frequency, gain + 1j * turn,
                                       luminance.astype(np.complex128)])
        floors[name] = (DEAD_ROW_FLOOR * max(abs(reference[name]), 1e-12)
                        / float(epsilon))
    return packed, names, floors


def _participation(stack: np.ndarray) -> Dict[str, object]:
    """The participation ratio and condition of a set of unit signatures."""
    if stack.size == 0:
        return {"effective": 0.0, "condition": 0.0,
                "singular_values": np.zeros(0), "span_90": 0, "span_99": 0}
    singular = np.linalg.svd(stack, compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    cumulative = np.cumsum(share)
    return {
        "effective": float(1.0 / np.sum(share ** 2)),
        "condition": float(singular[0] / max(singular[-1], 1e-30)),
        "singular_values": singular,
        "span_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "span_99": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def line_dimensions(line: Dict[str, object], modes: int = 20,
                    epsilon: float = 1e-3,
                    channel: Optional[Callable[[np.ndarray], np.ndarray]]
                    = None, interpolate: int = 4,
                    level_orders: int = 4) -> Dict[str, object]:
    """HOW MANY INDEPENDENT DIMENSIONS ONE TEST LINE PROVIDES.

    The same question `ELLIPTICAL_COLLAPSE.md` section 7.2 asks of the
    tape's magnetics - 1.58 effectively distinguishable of 6 - and
    `interference.distinguishable` asks of the modelled interference -
    6.89 of 12 - asked of a specified test line. The statistic is the same
    one: the participation ratio of the singular values, which is what makes
    the figures comparable across the arc.

    MEASURED, on the lines this module renders, over 52 departures - 20
    gain modes and 20 delay modes across the video band, paired into 20
    complex places, plus 12 level-dependent ones in four more:

        signal              readings  effective  per reading  condition
        NTC-7 combination      26      10.53        0.405        123
        NTC-7 composite        27       5.61        0.208        1.2e4
        75% colour bars        26       4.76        0.183        5.7e3
        (tape magnetics)        6       1.58        0.263        1.1e4

    with 90 per cent of each span carried by 13, 10 and 5 directions and 99
    per cent by 19, 16 and 11, and every one standing 12 to 21 sigma above
    its own sphere floor - so the structure is real and not the
    finite-ensemble scatter `sphere_floor` predicts.

    THE COMPOSITE LINE IS INEFFICIENT, at 0.208 against
    `docs/VERTICAL_INTERVAL_COLLAPSE.md` section 4.3's 0.186 from a
    construction sharing no code with this one: its bar, its 2T pulse, its
    12.5T luminance half and its staircase treads are four views of one
    low-frequency luminance channel, so a gain change moves all four
    together. The combination is twice as efficient because its multiburst
    is parameterised along the axis that makes its parts different.

    AND THE COLOUR BARS ARE LOWER STILL, at 0.183, which corrects that
    document's "worst of eleven" for the composite - the bars were not in
    its table. Eight bar luminances are eight readings of one level.

    THE RECORDING DOES NOT DESTROY THE IDENTIFIABILITY, IT MOVES IT.
    Measured through a VHS colour-under channel the same three lines return
    5.88, 10.84 and 4.71, every reading still responding. What changes is
    not how many directions there are but which channel they belong to -
    see `survival`, and the 3.58 MHz packet in particular.
    """
    from vhsdecode.models import information_extrapolation as ie

    rows, mode_names, floors = reading_jacobian(line, modes, epsilon,
                                                channel, interpolate,
                                                level_orders)
    # A reading that does not respond at all has no direction, and
    # normalising its row would measure the rounding of a zero - which is
    # exactly the error `magnetic.py` records. Judged against the reading's
    # OWN numerical floor, because these readings are in different units.
    live = {name: {"frequency": row} for name, row in rows.items()
            if float(np.linalg.norm(row)) > floors[name]}
    dead = sorted(name for name, row in rows.items()
                  if float(np.linalg.norm(row)) <= floors[name])
    if len(live) < 2:
        return {"name": line["name"], "readings": len(rows),
                "responding": len(live), "effective": 0.0,
                "insensitive": dead,
                "why": "too few responding readings to fit an ellipse"}
    fit = ie.ellipsoid(live, "frequency", real_parameters=True)
    stack = np.array([np.concatenate([live[name]["frequency"].real,
                                      live[name]["frequency"].imag])
                      for name in fit["names"]])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    result = _participation(stack)
    result.update({
        "name": line["name"],
        "citation": line["citation"],
        "readings": len(rows),
        "responding": len(fit["names"]),
        "insensitive": dead,
        "modes": len(mode_names),
        "per_reading": result["effective"] / max(len(rows), 1),
        "asymmetry": float(fit["asymmetry"]),
        "sphere_floor": float(fit["sphere_floor"]),
        "sigma": float(fit["sigma"]),
        "names": list(fit["names"]),
        "why": ("two readings that respond identically to every departure "
                "are one measurement wearing two names"),
    })
    return result


def distinguishable(frequency_hz, signals: Optional[Sequence[str]] = None,
                    depth: float = SIGNATURE_DEPTH) -> Dict[str, object]:
    """How many of the SPECIFIED ELEMENTS a band can tell apart.

    The companion to `line_dimensions`, asked of the signatures rather than
    of the readings, and built exactly as `interference.distinguishable`
    builds its own: the log magnitude for the amplitude part so a gain is a
    constant rather than a scale, the unwrapped angle for the phase part,
    and the mean removed from each because a constant is a level and not a
    shape.

    MEASURED over 1 to 7 MHz, 512 places, all three lines together: 25
    elements resolving 8.62 independent directions at condition 1.5e3, with
    12 directions carrying 90 per cent of the span and 16 carrying 99.
    """
    from vhsdecode.models import information_extrapolation as ie

    shapes: Dict[str, Dict[str, np.ndarray]] = {}
    for name, value in signatures(frequency_hz, signals, depth).items():
        magnitude = np.log(np.maximum(np.abs(value), 1e-12))
        phase = np.unwrap(np.angle(value))
        vector = ((magnitude - magnitude.mean())
                  + 1j * (phase - phase.mean()))
        if float(np.linalg.norm(vector)) > 0.0:
            shapes[name] = {"frequency": vector}
    if len(shapes) < 2:
        return {"names": list(shapes), "count": len(shapes),
                "effective": 0.0}
    fit = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    stack = np.array([np.concatenate([shapes[name]["frequency"].real,
                                      shapes[name]["frequency"].imag])
                      for name in fit["names"]])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    result = _participation(stack)
    result.update({"names": list(fit["names"]), "count": len(fit["names"]),
                   "ellipse": fit})
    return result


# ==========================================================================
# What survives a recording.
# ==========================================================================


def through_colour_under(line: Dict[str, object], values: np.ndarray,
                         luma_hz: float = VHS_LUMA_HZ,
                         chroma_half_hz: float = VHS_CHROMA_HALF_HZ
                         ) -> np.ndarray:
    """A colour-under recording: luminance band-limited, chrominance apart.

    A CAUTION ON THE MODEL, and it matters for reading the numbers below:
    the band edges here are brick walls where a real record-side splitting
    filter rolls off. That the 3.0 MHz packet loses about half and the
    4.2 MHz packet is destroyed is not in doubt; the exact percentages are
    qualitative.
    """
    values = np.asarray(values, dtype=np.float64)
    rate = float(line["sample_rate_hz"])
    chroma = _bandpass(values, rate, SUBCARRIER_HZ, chroma_half_hz,
                       soft=False)
    luma = _lowpass(_notch(values, rate, SUBCARRIER_HZ, chroma_half_hz,
                           soft=False), rate, luma_hz, soft=False)
    return luma + chroma


def survival(line: Dict[str, object], luma_hz: float = VHS_LUMA_HZ,
             chroma_half_hz: float = VHS_CHROMA_HALF_HZ,
             places: int = 2048) -> Dict[str, object]:
    """Which elements survive a recording, element by element.

    Two questions, and both are answered here because they disagree. The
    first is spectral: what share of the element's own prescribed energy
    lies inside the channel that will carry it. The second is the reading:
    what the line's own readings do once the line has been through the
    recorder.

    MEASURED THROUGH A VHS COLOUR-UNDER CHANNEL - luminance to 3 MHz,
    chrominance carried apart at +-400 kHz about the subcarrier:

    | element | share of its energy the two paths carry | path | verdict |
    |---|---|---|---|
    | C2 packet 0.50 / 1.00 / 2.00 MHz | 99.99 / 99.87 / 99.91% | luminance | survives |
    | C2 packet 3.00 MHz | 57.16% | both, half each | **marginal** |
    | C2 packet 3.58 MHz | 95.76% | **chrominance** | survives, measuring a different channel |
    | C2 packet 4.20 MHz | 4.37% | - | **structurally lost** |
    | B2 white bar, D2 staircase, pedestal, white flag | 99.99 to 100% | luminance | survives |
    | B1 2T pulse | 99.69% | luminance | survives as a TAPE measurement, not as a measurement of the source |
    | F 12.5T luminance half | 100% | luminance | survives |
    | F 12.5T chrominance half, D2 and G chrominance | 97.4 to 99.5% | chrominance | survives |
    | the eight colour bars | 98.8 to 100% | both, five of eight | survives |

    and U-matic's wider luminance band keeps FOUR packets rather than three:
    the 3.00 MHz packet rises from 57.2 to 99.9 per cent carried, while the
    4.20 MHz packet is still lost at 16.8.

    and the readings agree with the energy accounting: the 3.0 MHz packet
    reads 49.0 IRE direct and 26.6 recorded, a loss of 45.7 per cent
    against the 45.3 per cent `docs/VERTICAL_INTERVAL_COLLAPSE.md` section
    6.1 records from an independent construction; the 4.2 MHz packet reads
    49.0 and 0.86, a loss of 98.2 per cent; and the 3.58 MHz packet reads
    49.0 and 49.2, unchanged.

    THE 3.58 MHz PACKET IS THE ROW THAT CATCHES PEOPLE OUT. It lands inside
    the chrominance passband, so on a colour-under recording it is recorded
    through the chroma path and comes back intact - measuring a channel that
    is not the luminance one. Anyone reading a flat 3.58 MHz packet off a
    VHS decode as evidence of luminance bandwidth is reading the chroma
    channel.

    AND THE 2T PULSE STILL RESPONDS AFTER THE RECORDING BUT NO LONGER
    MEASURES THE SOURCE. Its name says 4 MHz but 99.5 per cent of its
    energy is below 3 MHz, so the recorder passes nearly all of it and the
    reading keeps its sensitivity. What that reading now reports is the
    recorder's own filter. It is a measurement of the recording only when
    differenced against a direct feed, which is what the record-tap and
    playback-tap pair in `/testdata/test_patterns/vhs/` is for.
    """
    t = np.asarray(line["time_s"], dtype=np.float64)
    rate = float(line["sample_rate_hz"])
    frequency = np.linspace(0.0, rate / 2.0, int(places))
    # THE TWO CHANNELS ARE DISJOINT, AND THE RECORDER MAKES THEM SO. The
    # chrominance path takes a band about the subcarrier; the luminance path
    # takes what is left below the cut-off, the chrominance band having been
    # notched out of it first. Mirrored exactly from `through_colour_under`,
    # so an element's energy is never counted on both paths - which matters
    # at U-matic's wider luminance band, where the two would otherwise
    # overlap by 400 kHz.
    chroma_band = np.abs(frequency - SUBCARRIER_HZ) <= chroma_half_hz
    luma_band = (frequency <= luma_hz) & ~chroma_band
    verdicts: Dict[str, Dict[str, object]] = {}
    for name, entry in line["elements"].items():
        values = np.zeros_like(t)
        for part in ("luma", "chroma"):
            piece = entry.get(part)
            if piece is not None:
                values = values + np.asarray(piece, dtype=np.float64)
        spectrum = np.abs(_spectrum_at(t, values, frequency,
                                       float(entry["centre_s"]))) ** 2
        total = float(spectrum.sum())
        if not total > 0.0:
            continue
        in_luma = float(spectrum[luma_band].sum()) / total
        in_chroma = float(spectrum[chroma_band].sum()) / total
        # THE ELEMENT IS ROUTED BY FREQUENCY AND NOT BY WHICH PLANE IT WAS
        # AUTHORED ON. That is the whole point of the 3.58 MHz multiburst
        # packet: it is a luminance probe by intent and a chrominance signal
        # by frequency, and the recorder only sees the frequency.
        carried = in_luma + in_chroma
        if min(in_luma, in_chroma) > 0.1 * carried:
            path = "both"
        else:
            path = "luminance" if in_luma >= in_chroma else "chrominance"
        if carried >= 0.9:
            verdict = "survives"
        elif carried >= 0.25:
            verdict = "marginal"
        else:
            verdict = "structurally lost"
        verdicts[name] = {"in_luma_band": in_luma,
                          "in_chroma_band": in_chroma,
                          "carried": carried, "path": path,
                          "verdict": verdict}
    direct = readings(line)
    recorded = readings(line, through_colour_under(
        line, np.asarray(line["composite_ire"], dtype=np.float64), luma_hz,
        chroma_half_hz))
    change = {name: (recorded.get(name, 0.0) - value)
              for name, value in direct.items()}
    return {"name": line["name"], "elements": verdicts, "direct": direct,
            "recorded": recorded, "change": change,
            "luma_hz": float(luma_hz),
            "chroma_half_hz": float(chroma_half_hz),
            "why": ("an element survives when the channel that will carry "
                    "it holds its energy; the 3.58 MHz packet survives "
                    "through the CHROMA path and therefore stops being a "
                    "luminance measurement")}


def line_dimension(line: Dict[str, object], measured: np.ndarray,
                   segments: int = 8) -> Dict[str, object]:
    """THE DIMENSION: the specified line against the line as it came back.

    `pair_dimension.sync_dimension` does exactly this for the sync pulse,
    and states why it is the best-conditioned dimension available: its
    synthetic side is SPECIFIED rather than fitted, so nothing is estimated
    from the data it is then judged on, and it is CONTENT-FREE, so it
    satisfies the standing constraint that correction derives from sync and
    the reserved intervals and never from picture. An insertion test line
    has both properties and adds the one thing a sync pulse cannot give:
    energy spread across the whole video band instead of concentrated at
    one edge.

    No new machinery is needed, because it IS a pair. The specified line is
    the input channel, the line as it came back is the output, and their
    Wiener transfer is a dimension like any other -
    `pair_dimension.pair_transfer` supplies it together with the debiased
    coherence that says where the estimate is trustworthy.

    THIS IS THE HOOK A DECODE PLUGS INTO. Everything else in this module is
    the synthetic side alone; hand this function a decoded line resampled
    onto the specification's own time axis and it returns the channel.

    MEASURED, on the combination line with a modelled VHS colour-under
    recording as the measured side:

        1.01 MHz    -0.00 dB   coherence 1.0000
        2.03 MHz    +0.00 dB             1.0000
        3.04 MHz    -8.67 dB             0.7212
        3.55 MHz    -0.00 dB             1.0000
        4.05 MHz   -25.47 dB             0.0860

    and the fourth row is the survival finding arriving from a completely
    different direction: at 3.55 MHz the transfer is flat and the coherence
    perfect, in the middle of a channel that has been cut off since 3 MHz,
    because the recorder carries the subcarrier down the CHROMA path. The
    recorder's cut-off shows up as coherence falling away rather than as a
    phantom of spectral division, which is what the coherence is for.

    AND THE IDENTITY IS A CONTROL. Given the specified line as its own
    measured side the transfer must be exactly one at every place; measured,
    `|T|` spans 0.9999999999999996 to 1.0000000000000004.
    """
    specified = np.asarray(line["composite_ire"], dtype=np.float64).ravel()
    got = np.asarray(measured, dtype=np.float64).ravel()
    if got.size != specified.size:
        raise ValueError("the measured line must be on the specified line's "
                         "own grid; resample it by its sync edge first")
    result = pair_dimension.pair_transfer(specified, got, segments)
    rate = float(line["sample_rate_hz"])
    width = specified.size // max(int(segments), 1)
    result["frequency_hz"] = np.fft.rfftfreq(width, d=1.0 / rate)
    result["specified"] = specified
    result["why"] = ("the specified line against the line as it came back; "
                     "the synthetic side is a standard, not a fit, and the "
                     "coherence says where the transfer is believable")
    return result


# ==========================================================================
# The amplitude axis.
# ==========================================================================


def amplitude_axis(table: Sequence[Tuple[str, float, float,
                                         Optional[float]]] = COLOUR_BARS_75,
                   sweep_bins: int = 9, entries: int = 91,
                   orders: int = 6) -> Dict[str, object]:
    """DO THE COLOUR BARS RELIEVE THE AMPLITUDE AXIS'S SATURATION?

    `docs/ELLIPTICAL_COLLAPSE.md` section 6a records the amplitude axis as
    *saturated for want of places* - 91 entries over 9 sweep bins, a sphere
    floor of 0.919 that nothing can stand above. The 75 per cent bars put
    eight exactly-known luminance levels and six exactly-known subcarrier
    phases on one line. This measures what that is worth, and the answer has
    three parts, only one of which is a rescue.

    MEASURED:

    | | places | sphere floor | scatter |
    |---|---|---|---|
    | the sweep alone | 9 | 0.9192 | 0.157 |
    | the sweep plus the bars' levels | 17 | 0.8505 | 0.083 |
    | what a floor below 0.5 would need | 92 | 0.4973 | 0.015 |

    **The count alone does not rescue it.** Eight more places move the floor
    from 0.919 to 0.851 and 91 entries still cannot stand above that. The
    floor is `N/(L + N - 1)` and it is `N` that is large: a floor below one
    half needs 92 places, which is ten colour bar lines' worth of distinct
    levels and not one.

    **Nor are eight specified levels worth more than nine fitted bins as
    LEVELS.** Over a Chebyshev basis of six level-dependent laws the nine
    sweep bins support 5.600 independent directions and the eight bar levels
    5.640. That is the same axis with the same reach.

    **What the bars supply that a sweep bin cannot is PHASE, and that is the
    relief.** A sweep bin is an amplitude and nothing else. Every coloured
    bar carries a subcarrier phase fixed by SMPTE 170M-2004 Annex A to four
    decimal places, so at that level a gain error and a rotation are two
    different mechanisms rather than one. Measured on the same basis: 5.640
    directions from the levels alone, and **7.674 with the phase column** -
    2.03 more directions, a 36 per cent increase, on exactly the same eight
    places.

    **And the levels are exactly known where the bins were fitted.** Nine
    sweep bins are estimated from the data they are then judged on, which is
    the problem the out-of-sample rule exists to contain; eight bar levels
    are printed in SMPTE EG 27-2004 table 2. That is not a bigger axis, it
    is an axis whose synthetic side needs no estimation.

    ONE CORRECTION THE CALLER MUST CARRY. `docs/MATHEMATICS.md` section 10.3
    has already withdrawn the "saturated, not empty" reading: when `N`
    approaches `L` the asymmetry is pinned near its ceiling and cannot
    fluctuate, so the true scatter collapses and those ensembles do clear
    three sigma against it. The bars are therefore not rescuing a dead axis;
    they are supplying known places and a phase reference to an axis that
    was already reading structure.
    """
    from vhsdecode.models import information_extrapolation as ie

    levels = np.array([row[1] for row in table], dtype=np.float64)
    chroma = np.array([row[2] for row in table], dtype=np.float64)
    phases = np.array([np.nan if row[3] is None else row[3]
                       for row in table], dtype=np.float64)
    with_phase = np.isfinite(phases)

    def floor_at(places: int) -> Dict[str, float]:
        circle = ie.sphere_floor(int(entries), int(places))
        return {"places": int(places), "asymmetry": circle["asymmetry"],
                "scatter": circle["scatter"]}

    needed = int(entries) + 1          # N/(L+N-1) < 0.5 needs L > N - 1

    # How many level-dependent laws the places can tell apart. The
    # candidate laws are Chebyshev polynomials of the normalised level -
    # the standard well-conditioned basis for exactly this question - and
    # the count is the participation ratio, the same statistic as
    # everywhere else in this arc.
    #
    # A LUMINANCE law is observable at every place, because every place has
    # a level. A CHROMINANCE law is observable only where there is
    # chrominance to carry it, and it is complex, because at a place with a
    # specified subcarrier phase a gain error and a rotation are two
    # different mechanisms. A sweep bin has an amplitude and nothing else,
    # so it supports the first family and not the second.
    def directions(places: np.ndarray,
                   carried: Optional[np.ndarray] = None,
                   angles: Optional[np.ndarray] = None) -> Dict[str, float]:
        span = float(places.max() - places.min()) or 1.0
        x = 2.0 * (places - places.min()) / span - 1.0
        shapes = [np.cos(order * np.arccos(np.clip(x, -1.0, 1.0)))
                  for order in range(int(orders))]
        rows = [shape.astype(np.complex128) for shape in shapes]
        if carried is not None and angles is not None:
            live = (carried > 0.0) & np.isfinite(angles)
            turn = np.where(live, np.exp(1j * np.deg2rad(
                np.nan_to_num(angles))), 0.0)
            rows += [(shape * turn).astype(np.complex128)
                     for shape in shapes]
        stack = np.array([np.concatenate([row.real, row.imag])
                          for row in rows])
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        keep = norms.ravel() > 0
        result = _participation(stack[keep] / norms[keep])
        return {"rows": int(keep.sum()),
                "effective": float(result["effective"]),
                "condition": float(result["condition"])}

    sweep_levels = np.linspace(float(levels.min()), float(levels.max()),
                               int(sweep_bins))
    return {
        "bar_levels_ire": levels,
        "bar_phases_deg": phases,
        "levels_with_a_phase": int(with_phase.sum()),
        "sweep_alone": floor_at(sweep_bins),
        "sweep_plus_bars": floor_at(int(sweep_bins) + levels.size),
        "floor_below_half_needs": floor_at(needed),
        "entries": int(entries),
        "sweep_directions": directions(sweep_levels),
        "bar_directions": directions(levels),
        "bar_directions_with_phase": directions(levels, chroma, phases),
        "why": ("the count of places alone does not lift the floor; what "
                "the bars supply that a sweep bin cannot is a specified "
                "subcarrier PHASE at a specified level"),
    }


# ==========================================================================
# The control.
# ==========================================================================


# How large a relative change counts as a response. A reading is in
# whatever unit it is in, so the test has to be a ratio; a millionth of the
# reading's own size is far above any rounding and far below any real
# sensitivity to a gain of a few per cent.
RELATIVE_RESPONSE = 1e-6


def gain_control(line: Optional[Dict[str, object]] = None,
                 amounts: Sequence[float] = (0.05, 0.10, 0.20, 0.35, 0.50),
                 reader: Optional[Callable[..., Dict[str, float]]] = None,
                 interpolate: int = 4) -> Dict[str, object]:
    """THE CONTROL, kept as code because a control that cannot fail is not
    one.

    `magnetic.linear_control` earned its place by reading 4.97 where the
    answer was 1.00, and that is what exposed a published 4.65-of-5. This is
    the same device for this component, and it catches the specific way a
    reading of a test line goes wrong.

    A FLAT INSERTION-GAIN CHANGE IS ONE PARAMETER, SO IT IS ONE DIRECTION.
    Probe the line with the same flat gain at several different sizes and
    take the central difference at each. Any reading that is smooth in the
    gain returns its own derivative at every size, so every row is that
    derivative times the same vector of ones, every normalised row is the
    same unit vector, and the participation ratio is exactly 1.00 however
    many readings the line has and however many sizes are used. Anything
    else means the readings are not measuring what they claim to.

    WHAT IT CATCHES, AND IT IS THE DEFECT THIS SUBJECT IS PRONE TO. A width
    read at a threshold fixed in absolute IRE is not smooth in the gain -
    the amplitude moves and the threshold does not - and a crossing taken to
    the nearest sample on the 4fsc grid quantises to 70 ns, which is more
    than a quarter of a 2T pulse's whole half-amplitude duration, so its
    central differences step in whole samples rather than following a
    derivative. Both defects move the participation ratio off one, and both
    break the homogeneity test below outright, which is the sharper of the
    two and is why this control has two halves rather than one.

    AND A SECOND, SHARPER TEST ON THE SAME PROBE. A flat gain is a scale, so
    a correct reading is HOMOGENEOUS in it: degree one for every level,
    amplitude and excursion, which scale with it, and degree zero for every
    duration, phase and ratio, which do not move at all. A reading that is
    neither is not measuring a property of the waveform; it is measuring its
    own threshold or its own sample grid. This needs no threshold in any
    unit and no list of which readings ought to be which.

    MEASURED, on the NTC-7 composite over gains of 5 to 50 per cent:

        reader                          effective  homogeneous  worst misfit
        own half amplitude, x4 interp     1.0000      24 of 24      6.2e-12
        fixed IRE, nearest sample         1.1647      22 of 24      1.0

    and the two readings the defective reader fails on are named: the 2T
    pulse's half-amplitude duration and the sync width - exactly the two
    quantities a fixed threshold cannot measure once the amplitude has
    moved. The worst misfit of 1.0 means one of them departed from
    homogeneity by its own entire size.

    So the control does fail when the construction is broken, which is the
    only thing that makes it evidence. `naive_readings` is the broken
    construction, kept in the module for that purpose.
    """
    line = ntc7_composite() if line is None else line
    read = readings if reader is None else reader
    base = np.asarray(line["composite_ire"], dtype=np.float64)
    reference = read(line, base, interpolate)
    rows: Dict[str, List[float]] = {name: [] for name in reference}
    change: Dict[str, float] = {name: 0.0 for name in reference}
    size: Dict[str, float] = {name: abs(value)
                              for name, value in reference.items()}
    seen: Dict[str, List[Tuple[float, float]]] = {name: []
                                                  for name in reference}
    for amount in amounts:
        plus = read(line, base * (1.0 + amount), interpolate)
        minus = read(line, base * (1.0 - amount), interpolate)
        for name in reference:
            high, low = plus.get(name, 0.0), minus.get(name, 0.0)
            rows[name].append((high - low) / (2.0 * amount))
            change[name] = max(change[name], abs(high - low))
            size[name] = max(size[name], abs(high), abs(low))
            seen[name].append((1.0 + amount, high))
            seen[name].append((1.0 - amount, low))
    # WHICH READINGS A GAIN IS ENTITLED TO MOVE, decided by relative change
    # and not by a threshold in any unit. A reading whose ideal value is
    # exactly zero - a tilt, a ringing, an overshoot on a line that has
    # none - has no size for a relative change to be measured against, so
    # the control does not judge it and does not claim to.
    judged = [name for name, value in reference.items() if value != 0.0]
    live = [name for name in judged
            if change[name] > RELATIVE_RESPONSE * max(size[name], 1e-300)]
    still = [name for name in judged if name not in live]
    # THE SHARPER HALF OF THE CONTROL: EVERY READING MUST BE HOMOGENEOUS IN
    # THE GAIN. A flat gain is a scale, so a correct reading of the
    # waveform is either degree one - a level, an amplitude, an excursion,
    # all of which scale with it - or degree zero - a duration read at the
    # pulse's own half amplitude, a phase, a ratio, none of which move at
    # all. A reading that is NEITHER is not measuring a property of the
    # waveform; it is measuring its own threshold, or its own sample grid.
    misfit: Dict[str, float] = {}
    for name in judged:
        scale_of = max(abs(reference[name]), 1e-300)
        degree_zero = max(abs(value - reference[name])
                          for _gain, value in seen[name]) / scale_of
        degree_one = max(abs(value / gain - reference[name])
                         for gain, value in seen[name]) / scale_of
        misfit[name] = min(degree_zero, degree_one)
    inhomogeneous = sorted(name for name in judged
                           if misfit[name] > RELATIVE_RESPONSE)

    if len(live) < 2:
        return {"effective": 0.0, "expected": 1.0, "passes": False,
                "why": "no reading responded to a gain change"}
    stack = np.array([np.asarray(rows[name], dtype=np.float64)
                      for name in live])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    result = _participation(stack)
    effective = result["effective"]
    return {
        "effective": effective,
        "expected": 1.0,
        "responding": len(live),
        "responded": live,
        "invariant": still,
        "judged": judged,
        "homogeneous": len(judged) - len(inhomogeneous),
        "inhomogeneous": inhomogeneous,
        "worst_misfit": (max(misfit.values()) if misfit else 0.0),
        "not_judged": [name for name in reference if name not in judged],
        "amounts": list(amounts),
        "passes": bool(abs(effective - 1.0) < 0.05
                       and not inhomogeneous),
        "why": ("a flat gain is one parameter, so every reading that is "
                "smooth in it must lie along one direction; a threshold "
                "fixed in IRE or a crossing taken to the nearest sample "
                "is not smooth in it and the control says so"),
    }


def naive_readings(line: Dict[str, object],
                   values: Optional[np.ndarray] = None,
                   interpolate: int = 4) -> Dict[str, float]:
    """THE SAME READING SET, taken with the defective pulse reader.

    Identical to `readings` in every other respect, so the difference
    between what this returns and what `readings` returns is exactly the
    two defects of `naive_read_pulse` and nothing else. It exists to be
    handed to `gain_control`, and nothing else in this module calls it.
    """
    return readings(line, values, interpolate,
                    pulse_reader=naive_read_pulse)


def readback_control(line: Dict[str, object]) -> Dict[str, object]:
    """The rendered line read back against the specification that made it.

    The first check any synthetic side must pass, and it is not circular:
    the line is built from amplitudes and durations, and it is read back by
    an independent instrument that finds edges and crossings without being
    told where they are. A construction error - an edge rendered as a
    transition width instead of a 10-90 time, a chrominance plane summed
    over a luminance one that replaced it - moves these numbers immediately.

    MEASURED on the NTC-7 composite this module renders:

        blanking level            0 IRE       reads  -0.082
        sync amplitude          -40 IRE       reads -40.063
        sync width, 50% points  4.700 us      reads   4.7009 us
        sync rise, 10-90        0.140 us      reads   0.1415 us
        burst                    40 IRE p-p   reads  39.918
        B2 white bar            100 IRE       reads 100.082
        B1 2T pulse amplitude   100 IRE       reads 100.235
        B1 2T half-amplitude    250 +- 10 ns  reads 252.3 ns
        D2 staircase top         90 IRE       reads  90.085

    and on the 75 per cent colour bars, where the whole point is the table:
    every one of the eight luminance levels within 0.09 IRE, every one of
    the six chrominance amplitudes within 0.01 IRE, and every one of the six
    subcarrier phases within 0.05 degrees of SMPTE EG 27-2004 table 2.

    THE RESIDUALS ARE THE READER'S OWN FLOOR AND NOT THE SIGNAL'S. Each is a
    constant of the reading method - the 90th and 5th percentiles a
    half-amplitude reading takes its baseline and its extreme from, and the
    Gaussian measurement bands - so each cancels in a differential, which is
    the only way any of them is used. Nothing measured through this reader
    below about 0.1 IRE or 3 ns is a finding.
    """
    got = readings(line, interpolate=8)
    checks: List[Tuple[str, float, float, float]] = [
        ("blanking level", 0.0, got.get("blanking level", 0.0), 0.5),
        ("sync amplitude", SYNC_DEPTH_IRE, got.get("sync amplitude", 0.0),
         1.0),
        ("sync width", SYNC_WIDTH_S, got.get("sync width", 0.0), 50e-9),
        ("sync rise", SYNC_RISE_S, got.get("sync rise", 0.0), 20e-9),
        ("burst amplitude", BURST_IRE_PP, got.get("burst amplitude", 0.0),
         1.0),
    ]
    if "bar amplitude" in got:
        checks.append(("bar amplitude", 100.0, got["bar amplitude"], 0.5))
    if "2T pulse HAD" in got:
        unit = nyquist_interval_s()
        checks.append(("2T pulse HAD", NARROW_PULSE_T * unit,
                       got["2T pulse HAD"], 10e-9))
        checks.append(("2T pulse amplitude", 100.0,
                       got["2T pulse amplitude"], 0.5))
    treads = line["geometry"].get("treads")
    if treads:
        checks.append(("staircase top", float(treads[-1][2]),
                       got.get(f"staircase tread {len(treads)}", 0.0), 1.0))
    zones = line["geometry"].get("chroma_zones")
    for name, amplitude, _start, _stop in (zones or ()):
        checks.append((name, float(amplitude),
                       got.get(f"{name} amplitude", 0.0), 1.0))
    bars = line["geometry"].get("bars")
    for name, luminance, chroma_pp, phase, _start, _stop in (bars or ()):
        checks.append((f"bar {name} luminance", float(luminance),
                       got.get(f"bar {name} luminance", 0.0), 0.5))
        if chroma_pp > 0.0 and phase is not None:
            checks.append((f"bar {name} chrominance", float(chroma_pp),
                           got.get(f"bar {name} chrominance", 0.0), 1.0))
            # the angle, wrapped: the standard's phases run past 180 and a
            # measured angle comes back on (-180, 180]
            read = got.get(f"bar {name} phase", 0.0)
            turned = float(phase) + ((read - float(phase) + 180.0) % 360.0
                                     - 180.0)
            checks.append((f"bar {name} phase", float(phase), turned, 1.0))
    rows = [{"reading": name, "specified": want, "read": have,
             "tolerance": tolerance, "passes": bool(abs(have - want)
                                                    <= tolerance)}
            for name, want, have, tolerance in checks]
    return {"name": line["name"], "rows": rows,
            "passes": all(row["passes"] for row in rows),
            "why": ("the rendered line read back by an instrument that was "
                    "not told where anything is")}
