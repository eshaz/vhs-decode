"""THE CAPTURE CHAIN'S OWN NOISE, and which of it reaches the picture.

Ethan, 2026-09-06: *"I think some of the remaining component is the noise
profile of the RF capture chain. The adc that I have does have some noise
and energy leaking from the computer and the clock crystal. The clock
crystal should be derivable from the cxadc spec. I have replaced the
crystal with a 40MHz crystal."* His reference is
`gitlab.com/wolfre/cx25800-11z-cxadc-rework-measurements`, which measures
the same card.

TWO FAMILIES, AND THEY ARE TOLD APART BY A CONTROL HE HIMSELF CREATED.
Replacing the crystal moves everything the crystal generates and leaves
everything else where it was, so his own captures at the stock 28.63636
MHz part and at the 40 MHz part are a two-point experiment on the origin
of every spur. Measured on both, as decibels above each spur's own local
noise floor:

    spur                stock 28.63636 MHz      replaced 40 MHz     verdict
    crystal / 8         3.5795 MHz  +50 to +54  5.0000 MHz  +9/+11  MOVED
    crystal / 6         4.7727 MHz  +41 to +43  6.6667 MHz  +33/+35 MOVED
    crystal / 4              -                 10.0000 MHz +11/+22  moved
    6.0000 MHz          6.0000 MHz  +9 to +11   6.0000 MHz  +41/+46 STAYED
    12.0000 MHz        12.0000 MHz  +10 to +11 12.0000 MHz  +47/+48 STAYED

The predicted crystal sub-harmonics land on their predicted frequencies to
the resolution of a 105 millisecond record, and the two spurs at exactly
6.000000 and 12.000000 MHz do not move at all. So one family is the
converter's own clock and the other is the computer, which is the
distinction he drew, established by measurement rather than assumed.

THE SPURS ARE REAL AND THEY DO NOT EXPLAIN THE SYNC-EDGE RESIDUAL, which
is a negative result with a clean control and is recorded here so it is
not proposed again. The response this arc measures is taken ON THE SYNC
PULSE, where the carrier sits at the tip, so a spur beats with it to a
FIXED frequency and should raise the residual exactly there: crystal / 6
to 3.2667 MHz, the computer's 6.0000 to 2.6000, crystal / 8 to 1.6000.
The excess in the measured relative error at each, as a multiple of that
export's own median:

    export   head   instrument      3.2667   1.6000   2.6000   elsewhere
    cd        a     CX card          10.29     0.60     4.58      245.67
    cd        b     CX card           9.57     0.62     4.04      315.79
    home      a     CX card           4.59     0.62     2.10       15.96
    home      b     CX card           4.14     0.55     1.83       16.61
    pnb       a     Rigol scope       5.00     0.63     3.44        5.01
    pnb       b     Rigol scope       5.80     0.63     3.46        5.99

The pulse-and-bar capture was taken on a Rigol oscilloscope with no CX
card and no crystal anywhere in it, and it shows the SAME elevation at
both beat frequencies. So whatever raises the residual there is common to
all three instruments and is not this. The largest excess in band is not
at a beat frequency at all - it is 245 to 315 times the median somewhere
else entirely on the countdown export.

One caveat that keeps the question open rather than closed: the export is
a per-field mean, and a spur that is not locked to the line rate averages
down in it. So this test bounds how much of the SYNC-EDGE residual the
capture chain explains, and it does not bound what the spurs do to a
single field or to the active picture. The spurs themselves are not in
doubt; they are measured directly in the raw radio frequency above.

AND THE STOCK CRYSTAL PUTS A SPUR EXACTLY ON THE COLOUR. 28.63636 MHz is
EIGHT TIMES the NTSC subcarrier - the ratio is 1.000000 - so the stock
part's eighth sub-harmonic sits at 3.579545 MHz, on the colour
subcarrier, and it is the strongest spur in the whole capture at +50 to
+54 dB over its own floor. Replacing the crystal with a 40 MHz part moves
that spur to 5.0000 MHz and off the colour entirely. Whatever the reason
for the change, this is one of its consequences and it is a large one.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# The stock part on a CX white card, and the divisions the reference
# measured. Its own text: the sample rate is the clock divided by one in
# eight-bit mode and by two in ten-bit mode.
STOCK_CRYSTAL_HZ = 28.63636e6
STOCK_CRYSTAL_STATUS = ("the cxadc reference's stated stock value; the "
                        "card's own marking was not read")
EIGHT_BIT_DIVISOR = 1.0
TEN_BIT_DIVISOR = 2.0

# The sub-harmonics that were found. The reference names the sixth
# explicitly ("1/6th of crystal frequency: 4.773 MHz"); the fourth and the
# eighth were found by looking for the family on Ethan's own captures.
SUB_HARMONICS = (4, 6, 8)

# The spurs that do NOT move with the crystal, measured at exactly these
# frequencies on captures taken with two different crystals. 12 MHz is the
# universal serial bus's full-speed reference and 6 MHz is its half, which
# is a LABELLED IDENTIFICATION and not a measurement: what is measured is
# that they are fixed, not what generates them.
FIXED_SPURS_HZ = (6.0e6, 12.0e6)
FIXED_SPUR_STATUS = ("measured fixed under a crystal change; the "
                     "attribution to a 12 MHz computer clock is a label")


def sample_rate(crystal_hz: float, bits: int = 8) -> Dict[str, object]:
    """The rate the converter runs at, from the crystal it is clocked by.

    The reference states the division plainly: eight-bit mode takes the
    clock and ten-bit mode takes half of it. Ethan's 40 MHz part in
    eight-bit mode therefore gives the 40 MSps his captures carry, which is
    the arithmetic that confirms which crystal a capture was taken with.
    """
    divisor = EIGHT_BIT_DIVISOR if int(bits) == 8 else TEN_BIT_DIVISOR
    rate = float(crystal_hz) / divisor
    return {
        "crystal_hz": float(crystal_hz),
        "bits": int(bits),
        "divisor": divisor,
        "sample_rate_hz": rate,
        "nyquist_hz": 0.5 * rate,
        "cite": ("gitlab.com/wolfre/cx25800-11z-cxadc-rework-measurements: "
                 "eight-bit mode is the clock, ten-bit mode is half"),
    }


def crystal_spurs(crystal_hz: float,
                  divisions: Sequence[int] = SUB_HARMONICS
                  ) -> Dict[str, object]:
    """Where the converter's own clock leaves marks, and how to falsify it.

    Every entry scales with the crystal, which is exactly what makes the
    family testable: change the crystal and they all move together by the
    same ratio. Nothing here is fitted.
    """
    lines = {f"crystal / {int(n)}": float(crystal_hz) / int(n)
             for n in divisions}
    return {
        "crystal_hz": float(crystal_hz),
        "lines_hz": lines,
        "scales_with_the_crystal": True,
        "falsified_by": ("a crystal change that leaves any of them where "
                         "it was"),
        "cite": ("the sixth is named by the cxadc reference; the fourth "
                 "and eighth were found in the same family on the arc's "
                 "own captures"),
    }


def lands_on_the_subcarrier(crystal_hz: float, subcarrier_hz: float,
                            tolerance_hz: float = 1e3) -> Dict[str, object]:
    """Does this crystal put one of its own sub-harmonics on the colour?

    The stock part does, and not by accident: 28.63636 MHz is eight times
    3.579545 MHz to a ratio of 1.000000, because the card was designed
    around the colour subcarrier. The consequence is that its strongest
    spur sits exactly where the colour is.
    """
    hits = {}
    for name, hz in crystal_spurs(crystal_hz)["lines_hz"].items():
        offset = hz - float(subcarrier_hz)
        if abs(offset) <= float(tolerance_hz):
            hits[name] = {"hz": hz, "offset_hz": offset}
    return {
        "crystal_hz": float(crystal_hz),
        "subcarrier_hz": float(subcarrier_hz),
        "hits": hits,
        "any": bool(hits),
        "ratio_to_subcarrier": float(crystal_hz) / float(subcarrier_hz),
        "why": ("a spur on the subcarrier is indistinguishable from colour "
                "by any filter that keeps the colour"),
    }


def beats_into_the_band(spur_hz, carrier_hz, video_band_hz: float
                        ) -> Dict[str, object]:
    """Which spurs reach the picture, and by what route.

    A spur inside the frequency-modulated band is captured with the
    carrier and needs no help. A spur outside it still reaches the picture
    because the demodulator is NOT linear: its difference with the carrier
    is a baseband term, and a spur six megahertz away from a four megahertz
    carrier lands squarely in the video. A spur whose difference exceeds
    the video band reaches nothing at first order.
    """
    spurs = np.atleast_1d(np.asarray(spur_hz, dtype=np.float64))
    carriers = np.atleast_1d(np.asarray(carrier_hz, dtype=np.float64))
    low, high = float(carriers.min()), float(carriers.max())
    rows = []
    for value in spurs:
        differences = np.abs(value - carriers)
        rows.append({
            "spur_hz": float(value),
            "inside_the_fm_band": bool(low <= value <= high),
            "beat_hz": differences,
            "closest_beat_hz": float(differences.min()),
            "reaches_the_video": bool(
                (low <= value <= high)
                or float(differences.min()) < float(video_band_hz)),
        })
    return {
        "spurs": rows,
        "carrier_span_hz": (low, high),
        "video_band_hz": float(video_band_hz),
        "why": ("the demodulator is nonlinear, so an out-of-band spur "
                "arrives as its difference with the carrier"),
    }
