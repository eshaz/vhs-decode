"""THE INVERSION: read the levels from the carrier's frequency, not the level.

Ethan, 2026-09-06: *"the frequency will have been refined to the point
where this will perfectly resolve the drift and shape of the luma channel
to remove the agc and other level inconsistencies"*, and then: *"Let's
build that inversion."*

WHY IT IS AN INVERSION AND NOT A SECOND OPINION. The recorder maps the
video's levels onto the carrier's frequency - SMPTE 32M-2004 clause
3.9.1.1.4 puts the sync tip at 3.4 MHz and peak white at 4.4, so the
140 IRE excursion IS the 1.0 MHz deviation. A level and a frequency are
therefore the same quantity in two units, and the arc has until now
measured it in the harder of the two. Measuring the level means finding a
flat interval in a demodulated, de-emphasised, time-base-corrected picture
and averaging it, which inherits every stage in between; measuring the
frequency means reading the instantaneous frequency of the RF itself,
which inherits none of them.

AND THE FREQUENCY IS THE FINER MEASUREMENT. The demodulator is exact on a
record holding a whole number of cycles and reaches a fifth of the ten-bit
output floor on one that does not (`hypercomplex.demodulate`), while the
level-domain sync depth scatters 0.65 to 2.1 IRE line to line. In the
frequency domain 1 IRE is 1.0 MHz / 140 = 7.14 kHz, so the two are
directly comparable and the comparison is the point of this module.

THE MEASUREMENT, AND IT MOVES THE FAULT. Read from the RF itself, band
limited to the luma and gated on the sync pulses the specified duty cycle
finds:

    tape         window   tip MHz   blanking   spacing IRE   the decode's
    bars         0.10 s    3.4302     3.7236        40.82        34.925
    countdown   16.70 s    3.4326     3.7498        43.62        36.518
    countdown   30.00 s    3.4292     3.7305        42.14        36.518

The tape holds the specified forty IRE and a little over - 40.8 and 42 to
44 - while the decoded picture delivers 34.9 and 36.5. The ratio is 1.17
on the bars tape and 1.15 to 1.19 on countdown, the same figure on two
tapes recorded on different equipment years apart.

SO THE LEVEL DEFICIT IS NOT ON THE TAPE. Every earlier reading of it -
gains of 0.913, 0.840 and 0.873, deficits of three to six IRE - was taken
on the decoded picture and attributed to the chain. The carrier says the
chain delivered its levels correctly and something after the demodulator
loses about fifteen per cent of them. That is a decoder calibration, not a
tape defect, and it is exactly what the inversion was built to find: the
frequency measurement resolves what the level measurement could not,
because it is taken before every stage the level measurement inherits.

TWO CAUTIONS ON READING THAT. The window at countdown's twenty second mark
returns 61.8 IRE with a tip scatter of 74 kHz against 7 to 10 kHz in the
good windows, so it is a dropout or an unlocked passage and not a
measurement; the scatter is what tells them apart. And home could not be
measured at all, because libsndfile cannot seek that file (it carries no
seek table and declares no length) - so this rests on two tapes, not
three.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import hypercomplex

# SMPTE 32M-2004 clause 3.9.1.1.4
SYNC_TIP_HZ = 3.4e6
PEAK_WHITE_HZ = 4.4e6
DEVIATION_HZ = PEAK_WHITE_HZ - SYNC_TIP_HZ
# ITU-R BT.1700: the excursion the deviation spans, sync tip to peak white
EXCURSION_IRE = 140.0
# the arc's own landing figure for blanking, which is 40 IRE above the tip
BLANKING_HZ = SYNC_TIP_HZ + DEVIATION_HZ * 40.0 / EXCURSION_IRE


def hz_per_ire() -> float:
    """The exchange rate between the two units, from the specification."""
    return DEVIATION_HZ / EXCURSION_IRE


def instantaneous(samples, sample_rate_hz: float, guard: float = 0.125,
                  band_hz: Tuple[float, float] = (2.6e6, 5.4e6)
                  ) -> Dict[str, object]:
    """The LUMA carrier's frequency, sample by sample, with the ends guarded.

    THE BAND LIMIT IS NOT OPTIONAL. A VHS track carries the luma's
    frequency-modulated carrier and the chrominance's amplitude-modulated
    one added together (SMPTE 32M clause 3.9.3), and the instantaneous
    frequency of a SUM of two carriers is neither of theirs. The back
    porch is where this bites hardest, because the colour burst lives
    there: a first version of this read blanking at 3.4857 MHz instead of
    3.6857 and returned a level gain of 0.22 where the level domain says
    0.87. Band-limiting to the luma's own 3.4 to 4.4 MHz first, with room
    for the deviation's sidebands, removes the 629 kHz carrier entirely.

    `hypercomplex.demodulate` is then the inverse Hilbert transform with
    the time differential; it is a transform-domain operator and assumes
    the record repeats, so the guard removes the wrap-around at the ends.
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    spectrum = np.fft.rfft(x)
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate_hz))
    spectrum[(frequencies < band_hz[0]) | (frequencies > band_hz[1])] = 0.0
    x = np.fft.irfft(spectrum, n=x.size)
    frequency = hypercomplex.demodulate(x, float(sample_rate_hz))
    trim = int(round(float(guard) * x.size))
    interior = slice(trim, x.size - trim) if trim and 2 * trim < x.size else slice(None)
    return {"frequency_hz": frequency[interior], "trimmed_each_end": trim,
            "sample_rate_hz": float(sample_rate_hz), "band_hz": band_hz,
            "band_limited": True}


def find_sync(frequency_hz, sample_rate_hz: float,
              pulse_s: float = 4.7e-6, line_s: float = 1.0 / 15734.264
              ) -> Dict[str, object]:
    """Locate the line sync pulses as the sustained LOW-frequency intervals.

    Negative modulation puts the sync tip at the bottom of the deviation,
    so the pulses are the only places the carrier stays low for 4.7
    microseconds together. Smoothing over half a pulse and thresholding
    below the median finds them without needing a demodulated picture or a
    time base.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    width = max(int(round(0.5 * pulse_s * float(sample_rate_hz))), 3)
    kernel = np.ones(width) / width
    smooth = np.convolve(f, kernel, mode="same")
    # THE THRESHOLD IS THE DUTY CYCLE, which the standard gives: a 4.7
    # microsecond pulse on a 63.5555 microsecond line is 7.39 per cent of
    # the time, so the sync tip is the lowest 7.39 per cent of the smoothed
    # frequency and the threshold is that percentile, widened a little to
    # take the edges. A first version set it a quarter of the way from the
    # median to the second percentile, which put 31 per cent of samples
    # below it and found no pulse at all.
    duty = 100.0 * float(pulse_s) / float(line_s)
    threshold = float(np.percentile(smooth, duty * 1.1))
    low = smooth < threshold
    edges = np.diff(low.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    stops = np.flatnonzero(edges == -1) + 1
    if starts.size and stops.size and stops[0] < starts[0]:
        stops = stops[1:]
    count = min(starts.size, stops.size)
    starts, stops = starts[:count], stops[:count]
    want = pulse_s * float(sample_rate_hz)
    keep = (stops - starts > 0.6 * want) & (stops - starts < 1.6 * want)
    return {"starts": starts[keep], "stops": stops[keep],
            "found": int(keep.sum()),
            "duty_percent": duty,
            "expected_per_second": 1.0 / line_s,
            "threshold_hz": threshold,
            "why": ("negative modulation puts the tip at the bottom of the "
                    "deviation, so a sustained low interval of the right "
                    "length is a sync pulse and needs no time base")}


def levels(samples, sample_rate_hz: float,
           interior: Tuple[float, float] = (0.3, 0.7),
           porch_offset_s: float = 1.5e-6,
           porch_width_s: float = 1.5e-6) -> Dict[str, object]:
    """The blanking and sync-tip carriers, and the level gain they imply.

    The tip is read on the pulse's flat interior, the same window the
    level-domain measurement uses and for the same reason; blanking is
    read on the back porch, which begins after the burst and is located
    from the pulse's own trailing edge.
    """
    got = instantaneous(samples, sample_rate_hz)
    f = got["frequency_hz"]
    found = find_sync(f, sample_rate_hz)
    if found["found"] < 8:
        raise ValueError(f"only {found['found']} sync pulses found; give a "
                         f"longer window or check the sample rate")
    rate = float(sample_rate_hz)
    tip_values, porch_values = [], []
    for start, stop in zip(found["starts"], found["stops"]):
        span = stop - start
        a = start + int(interior[0] * span)
        b = start + int(interior[1] * span)
        if b > a:
            tip_values.append(float(np.median(f[a:b])))
        p0 = stop + int(round(porch_offset_s * rate))
        p1 = p0 + int(round(porch_width_s * rate))
        if p1 < f.size:
            porch_values.append(float(np.median(f[p0:p1])))
    tip = np.asarray(tip_values)
    porch = np.asarray(porch_values)
    n = min(tip.size, porch.size)
    tip, porch = tip[:n], porch[:n]
    gap = porch - tip
    specified_gap = BLANKING_HZ - SYNC_TIP_HZ
    gain = float(np.median(gap)) / specified_gap
    return {
        "pulses": int(n),
        "tip_hz": float(np.median(tip)), "tip_scatter_hz": float(tip.std()),
        "blanking_hz": float(np.median(porch)),
        "blanking_scatter_hz": float(porch.std()),
        "gap_hz": float(np.median(gap)), "gap_scatter_hz": float(gap.std()),
        "specified_gap_hz": specified_gap,
        "gain": gain,
        "spacing_ire": float(np.median(gap)) / hz_per_ire(),
        "per_pulse_gap_hz": gap,
        "scatter_in_ire": float(gap.std()) / hz_per_ire(),
        "cite": "SMPTE 32M-2004 clause 3.9.1.1.4; ITU-R BT.1700 for the 40 IRE",
        "why": ("a level and a frequency are the same quantity in two "
                "units, and the frequency is read from the RF itself "
                "rather than through every stage that follows it"),
    }


def agreement(frequency_gain: float, level_gain: float) -> Dict[str, object]:
    """The two measurements of one quantity, and their disagreement.

    This is the measurement PAIR for the luma's scale: one taken on the RF
    before anything, one on the decoded picture after everything. They
    share no stage between them, so their agreement bounds both.
    """
    difference = float(frequency_gain) - float(level_gain)
    return {
        "frequency_gain": float(frequency_gain),
        "level_gain": float(level_gain),
        "difference": difference,
        "difference_ire": difference * EXCURSION_IRE * 40.0 / EXCURSION_IRE,
        "relative": difference / max(float(level_gain), 1e-30),
        "why": ("one measurement is taken on the RF before any stage and "
                "the other on the picture after all of them, so they share "
                "nothing and their agreement bounds both"),
    }
