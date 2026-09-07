"""The colour lock: the fixed offset between the colour-under carrier and the
colour lock derived from the luma.

Ethan, this session: *"There is a fixed offset between the colour-under carrier
and the colour lock derived from the luma; use it as another measurement
component."* And, in the same breath, the law that gives the component its
meaning: *"Colour-under carrier is constant (locked to chroma), colour carrier
is constant (locked to luma), time is constant; any deviation is residual."*

This module is the instrument for both sentences. It reads the SAME physical
quantity - the colour-under phase the tape holds - through two channels that
share nothing but the tape, forms their difference, shows that the difference
is a constant, and hands the departure from that constant back as a residual
channel.

THE TWO CHANNELS.

  * The DECODED CHROMA, at four times subcarrier. Its phase is the tape's
    colour-under phase plus the phase of the up-conversion oscillator the
    decoder synthesises and locks to the burst.

  * The DEMODULATED LUMA, at the colour-under carrier. The colour-under
    amplitude-modulates the luma FM carrier - `vhsdecode/luma_beat.py` records
    the coherence going from 0.07 to 0.55-0.64 as soon as the tape carries
    chroma - so the demodulated luma carries a small replica of the
    colour-under at its own frequency, with NO up-conversion applied to it.

Their difference is therefore the decoder's own up-conversion phase, read out
through a channel the decoder never touched. That is what makes it a
measurement rather than a tautology, and it is why a departure from it means
the tape moved rather than the decoder did.

WHY THE OFFSET REPEATS EVERY FOUR LINES AND NOT EVERY ONE. The colour-under is
recorded with the heterodyne phase advanced ninety degrees a line (RCA VCR-1
red book, quoted in `vhsdecode/chroma.py`), and the decoder's up-conversion
undoes that rotation line by line. At four times subcarrier a line is 227.5
cycles, so the sampling frame itself turns a hundred and eighty degrees a
line. Ninety against a hundred and eighty closes after four lines, which is
the period this module fits and the period the measurement shows.

MEASURED, AND WITH A CONTROL THAT FAILS THE RIGHT WAY. Run at this module's
own defaults on the NTSC decodes in `/tmp/claude-1000`, over picture lines 40
to 240 of eight fields each, `lock_offset` reports the mean resultant length of
the four residues - one for a perfectly fixed offset, zero for none:

    capture                lock resultant   residual rms   luma colour-under
    75 % bars SP           0.985 - 0.992     7.4 - 10.2 deg      0.565 IRE
    75 % bars EP           0.904 - 0.966    15.1 - 26.3 deg      0.627 IRE
    chroma noise SP        0.980 - 0.993     6.9 - 11.4 deg      0.446 IRE
    home recording         0.106 - 0.367    78.1 - 96.0 deg      1.233 IRE

and on a matched pair decoded here from the playback-tap captures, eight frames
each, `zaroff-multiburst-NTSC-SP` against its `y-only` twin - the same tape,
the same deck, the same pattern, with nothing recorded in the chroma band:

    multiburst SP          0.975 - 0.987     9.4 - 12.8 deg      1.131 IRE
    multiburst SP y-only   0.068 - 0.202    91.4 - 100.2 deg     1.135 IRE

The control is decisive in the way an amplitude test would not have been. The
AMPLITUDE at the colour-under carrier is the same to four parts in a thousand
on the two captures, because most of that amplitude is the luma's own picture.
What collapses when the chroma is taken off the tape is the LOCK, from 0.98 to
0.13. The quantity this module measures is a phase relation and the control
says so.

The home recording sits with the control at 0.11 - 0.37, and that is reported
rather than smoothed. `lock_offset` returns the resultant length so a caller
can decline; `residual` on a field whose lock is that weak is measuring the
luma's own content and must not be believed. Where the threshold belongs is a
decision for the runtime and is not taken here - what the two rows above
establish is that the statistic separates a tape carrying colour from one that
is not, which is the only property a gate needs.

WHAT DIRECTIVE 7 ASKS THIS MODULE TO ASSERT. Three quantities are declared
constant, each locked to the other channel:

    the colour-under carrier   40 f_H (NTSC), 40 f_H + 1953 Hz (PAL)
    the colour subcarrier      455/2 f_H (NTSC), 1135/4 f_H + 25 Hz (PAL)
    time                       the line period, 1 / f_H

`constants` states all three with their provenance, and `residual` is the
only thing this module reports as varying. The arc's standing treatment of a
known constant - `vhsdecode/models/running_differential.py`, Ethan: *"Time
over the span of the RF capture is constant so that is not on our component to
extract out"* - is what makes that split the right one: a constant is not a
component, and the departure from it is.

RELATION TO `colour_framing`. That module proves the colour FRAME cannot be
recovered from the recorded burst, because the subcarrier and the record
oscillator share a three-quarter-cycle fractional advance per field which
cancels in the down-conversion. Nothing here contradicts it. The offset
measured here is not the colour frame; it is the decoder's up-conversion phase
against the tape's colour-under phase, which is a per-line quantity and
survives precisely because it is not the per-field one.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from vhsdecode.models import colour_under

# The demodulated luma's colour-under replica is a narrow-band quantity about
# the carrier: measured on the decodes named in the module docstring, the share
# of the band that the decoded chroma can predict falls from 3.2-19.9 per cent
# at a 30 kHz half-width to 1.6-13.4 per cent at 150 kHz and 0.1-1.3 per cent
# at 500 kHz, while the y-only control holds at 0.1 per cent throughout. The
# default is the narrowest of the three because that is where the coupling is,
# and it is exposed as an argument because it is an analysis choice and not a
# figure of the format.
DEFAULT_HALF_WIDTH_HZ = 30e3

# The colour-under heterodyne advances ninety degrees a line and the four-times
# subcarrier sampling frame a hundred and eighty, so their difference closes
# after four lines. Derived in the module docstring; not a fitted period.
OFFSET_PERIOD_LINES = 4


def _lowpass(signal: np.ndarray, half_width_hz: float,
             sample_rate_hz: float, taps: int = 129) -> np.ndarray:
    """A linear-phase low-pass along the line axis.

    Linear phase matters here and nothing else about the window does: the
    whole measurement is a phase difference between two channels, so any
    filter delay has to be identical on both sides, which a symmetric
    finite-impulse-response filter applied to both guarantees by construction.
    """
    from scipy import signal as sps

    if taps % 2 == 0:  # a symmetric filter needs an odd length to be exact
        taps += 1
    kernel = sps.firwin(taps, half_width_hz / (sample_rate_hz / 2.0))
    return sps.filtfilt(kernel, [1.0], signal, axis=-1)


def chroma_envelope(chroma_field: np.ndarray,
                    sample_rate_hz: float,
                    subcarrier_hz: Optional[float] = None,
                    half_width_hz: float = DEFAULT_HALF_WIDTH_HZ,
                    centre: float = 32768.0) -> np.ndarray:
    """The decoded chroma as a complex envelope, one row per line.

    The time base corrected chroma is a real signal at the subcarrier; this
    returns its analytic envelope in the subcarrier's own rotating frame, so a
    steady hue is a steady complex number rather than a sinusoid.
    """
    field = np.asarray(chroma_field, dtype=np.float64) - float(centre)
    if field.ndim != 2:
        raise ValueError("a chroma field is two dimensional (lines, samples)")
    if subcarrier_hz is None:
        subcarrier_hz = sample_rate_hz / 4.0
    index = np.arange(field.shape[1], dtype=np.float64)
    local = np.exp(-2j * np.pi * float(subcarrier_hz) * index / sample_rate_hz)
    return _lowpass(field * local[None, :], half_width_hz, sample_rate_hz)


def luma_colour_under(luma_field: np.ndarray,
                      sample_rate_hz: float,
                      carrier_hz: Optional[float] = None,
                      half_width_hz: float = DEFAULT_HALF_WIDTH_HZ,
                      black_level: float = 0.0,
                      system: str = "NTSC") -> np.ndarray:
    """The colour-under replica the demodulated luma carries, as an envelope.

    Same construction as `chroma_envelope`, at the colour-under carrier
    instead of the subcarrier, so the two envelopes are the same physical
    vector expressed in two frames and their difference is a pure phase.
    """
    field = np.asarray(luma_field, dtype=np.float64) - float(black_level)
    if field.ndim != 2:
        raise ValueError("a luma field is two dimensional (lines, samples)")
    if carrier_hz is None:
        carrier_hz = colour_under.carrier_hz(system)
    index = np.arange(field.shape[1], dtype=np.float64)
    local = np.exp(-2j * np.pi * float(carrier_hz) * index / sample_rate_hz)
    return _lowpass(field * local[None, :], half_width_hz, sample_rate_hz)


def line_product(chroma: np.ndarray, luma: np.ndarray,
                 active: Tuple[int, int]) -> np.ndarray:
    """One complex number a line: the chroma against the luma's colour-under.

    THE ACTIVE AREA IS MASKED IN, not out. Blanking carries no colour-under
    modulation on either channel, so a sum that includes it adds noise to both
    sides of the product and nothing else; the burst is excluded for the same
    reason it is excluded from a saturation measurement - it is a reference
    the decoder has already used, so its phase is the decoder's and not the
    tape's.
    """
    first, last = int(active[0]), int(active[1])
    if not 0 <= first < last:
        raise ValueError("the active area must be a non-empty forward span")
    return (np.asarray(chroma)[:, first:last]
            * np.conj(np.asarray(luma)[:, first:last])).sum(axis=1)


def lock_offset(product: np.ndarray,
                lines: Optional[Tuple[int, int]] = None,
                period: int = OFFSET_PERIOD_LINES) -> Dict[str, object]:
    """THE FIXED OFFSET, as a table of unit complex numbers and its strength.

    One entry per line residue modulo `period`. The entries are unit modulus
    because the offset is a phase: the amplitude of the product is the product
    of two amplitudes and says nothing about the lock, which is exactly the
    trap the y-only control in the module docstring exposes.

    `resultant` is the mean resultant length of the unit phasors within each
    residue, averaged over residues - one for a perfectly fixed offset, zero
    for a uniform scatter. It is the number a caller should gate on.
    """
    product = np.asarray(product)
    first, last = (0, product.shape[0]) if lines is None else (int(lines[0]),
                                                               int(lines[1]))
    table = np.zeros(period, dtype=np.complex128)
    strength = np.zeros(period, dtype=np.float64)
    counts = np.zeros(period, dtype=np.int64)
    for residue in range(period):
        rows = np.arange(first, last)
        rows = rows[(rows % period) == residue]
        rows = rows[np.abs(product[rows]) > 0.0]
        counts[residue] = rows.size
        if rows.size == 0:
            continue
        unit = product[rows] / np.abs(product[rows])
        mean = unit.mean()
        strength[residue] = float(np.abs(mean))
        table[residue] = mean / np.abs(mean) if np.abs(mean) > 0 else 1.0
    return {
        "table": table,
        "degrees": np.degrees(np.angle(table)),
        "resultant": float(strength.mean()),
        "resultant_per_residue": strength,
        "lines_per_residue": counts,
        "period": int(period),
    }


def residual(product: np.ndarray, offset: Dict[str, object]) -> np.ndarray:
    """ANY DEVIATION IS RESIDUAL: the per-line departure, in degrees.

    Directive 7's second half, as a channel. The constant has been removed;
    what is returned is everything that is not the constant, one number a
    line, positive or negative, and it is the quantity the head-switch
    detector in `chroma_head_switch` reads.
    """
    product = np.asarray(product)
    table = np.asarray(offset["table"])
    period = int(offset["period"])
    magnitude = np.abs(product)
    unit = np.where(magnitude > 0.0, product / np.where(magnitude > 0.0,
                                                        magnitude, 1.0), 0.0)
    residues = np.arange(product.shape[0]) % period
    return np.degrees(np.angle(unit * np.conj(table[residues])))


def residual_scale(deviation: np.ndarray,
                   lines: Tuple[int, int]) -> Dict[str, float]:
    """How big the residual is where nothing is happening.

    The scale a departure has to be judged against. Taken over the caller's
    quiet span - the picture lines away from the head switch - and reported as
    both a root mean square and a median absolute deviation, because the first
    is what a Gaussian argument wants and the second is what survives the one
    or two lines that are the whole point of the measurement.
    """
    span = np.asarray(deviation)[int(lines[0]):int(lines[1])]
    span = span[np.isfinite(span)]
    if span.size == 0:
        raise ValueError("the quiet span holds no finite residual")
    median = float(np.median(span))
    return {
        "rms_deg": float(np.sqrt(np.mean(span ** 2))),
        "median_deg": median,
        "mad_deg": float(np.median(np.abs(span - median))),
        "lines": int(span.size),
    }


def constants(system: str = "NTSC") -> Dict[str, object]:
    """THE THREE CONSTANTS OF DIRECTIVE 7, each with where it comes from.

    Ethan: *"Colour-under carrier is constant (locked to chroma), colour
    carrier is constant (locked to luma), time is constant; any deviation is
    residual."*

    Nothing here is fitted. The colour-under carrier and the subcarrier are
    both stated by the format as multiples of the line rate, and the line
    period is the reciprocal of the same line rate - which is why the three
    are one constant seen three ways, and why a deviation in any of them is a
    deviation in the time base first and in a carrier only afterwards. That is
    the arithmetic behind the measurement already recorded in
    `docs/CHANNEL_LOOP_PROPOSAL.md`, where the raw colour-under sits 5752 Hz
    high and the time base removes 99.94 per cent of it.
    """
    line_rate = colour_under.line_rate_hz(system)
    return {
        "system": str(system).upper(),
        "line_rate_hz": float(line_rate),
        "colour_under_hz": float(colour_under.carrier_hz(system)),
        "colour_under_relation": colour_under.carrier_provenance(
            system)["relation"],
        "colour_under_locked_to": "chroma",
        "subcarrier_hz": float(colour_under.subcarrier_hz(system)),
        "subcarrier_relation": ("455/2 f_H" if str(system).upper()
                                in ("NTSC", "M", "M/NTSC", "525")
                                else "1135/4 f_H + 25 Hz"),
        "subcarrier_locked_to": "luma",
        "line_period_s": float(1.0 / line_rate),
        "time_locked_to": "the line rate both carriers are multiples of",
        "deviation_is": "residual",
    }


def measure(luma_field: np.ndarray, chroma_field: np.ndarray,
            sample_rate_hz: float, active: Tuple[int, int],
            quiet_lines: Tuple[int, int],
            black_level: float = 0.0,
            system: str = "NTSC",
            half_width_hz: float = DEFAULT_HALF_WIDTH_HZ) -> Dict[str, object]:
    """One field in, the colour lock and its residual out.

    The single entry point a caller needs: the fixed offset table, the
    strength that says whether to believe it, the per-line residual, and the
    scale that residual has to be judged against.
    """
    chroma = chroma_envelope(chroma_field, sample_rate_hz,
                             half_width_hz=half_width_hz)
    luma = luma_colour_under(luma_field, sample_rate_hz,
                             half_width_hz=half_width_hz,
                             black_level=black_level, system=system)
    product = line_product(chroma, luma, active)
    offset = lock_offset(product, quiet_lines)
    deviation = residual(product, offset)
    return {
        "offset": offset,
        "resultant": offset["resultant"],
        "residual_deg": deviation,
        "scale": residual_scale(deviation, quiet_lines),
        "chroma_envelope": chroma,
        "luma_colour_under": luma,
        "line_product": product,
    }
