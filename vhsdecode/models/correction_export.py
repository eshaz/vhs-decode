"""The collapsed residual, exported once, applied as one multiply.

Ethan: *"I need to export a model that I can use against my hifi capture as
well, to synchronize them together ... build a method that exports the full
collapsed residual that I would need to apply. I think we can use calculus to
simplify this away from a limit potentially, and to an identity so we can
apply this correction mathematically simplified and computationally
performant."*

THE LIMIT DOES COLLAPSE TO AN IDENTITY, AND THE REASON IS ONE LINE. Every
linear component in the chain multiplies in the frequency domain, so

    measured(f) = ideal(f) * prod_k H_k(f)

and taking logarithms turns the product into a sum

    log measured - log ideal = sum_k log H_k(f)

which is the whole of it. The correction is `exp(- sum_k w_k log H_k(f))`,
ONE complex array over frequency, and applying it is one forward transform,
one multiply and one inverse. The iteration this arc runs was never the
correction - it was the FIT, the search for the weights. Once the weights are
known the limit is spent and what remains is an identity.

THE COST IS O(N log N) ONCE, not per component and not per pass. A chain of
forty-seven declared positions collapses to a single spectrum before any
sample is touched, so adding a component to the model costs nothing at apply
time. That is the performance Ethan is asking for and it is free: it is a
property of logarithms, not an optimisation.

WHAT DOES NOT COLLAPSE, AND WHY THE EXPORT CARRIES IT SEPARATELY. The
identity holds for components that COMMUTE, which linear ones do. Two
components that do not commute cannot be folded into a product, and this
chain has exactly two kinds that do not:

    a memoryless non-linearity     the clip: its output depends on the
                                   sample's own value, so it has no
                                   frequency response to multiply at all
    a time-varying gain            the receiver AGC: a gain that moves
                                   within a line does not commute with a
                                   filter

So the export is a PAIR: one collapsed spectrum for everything linear, and a
short ordered list of the operations that must still be applied in sequence.
Pretending the second list is empty would make the export faster and wrong.

SYNCHRONISING TWO CAPTURES CLOCKED BY DIFFERENT CRYSTALS. Ethan: *"the
relation of the color carrier to the rf data rate, so I can derive drift
since the two signals were clocked with different crystals."*

The relation is exact and needs no measurement on the specification side.
For NTSC VHS the colour-under carrier is EXACTLY forty times the line rate:

    f_H  = 30000/1001 * 525      = 15734.265734... Hz
    f_cu = 40 f_H                = 629370.629371... Hz
    f_sc = 455/2 f_H             = 3579545.454545... Hz
    f_cu / f_sc = 40/(455/2)     = 16/91, exactly

so the colour carrier is rigidly tied to the tape's own time base and to
nothing else. Its position measured IN UNITS OF THE CAPTURE'S SAMPLE RATE is
therefore `40 / (samples per line)`, and any departure from the nominal is
the sum of two errors and only two: the tape's speed error and the capture
crystal's error.

THAT SUM IS WHAT MAKES TWO CAPTURES SEPARABLE. The tape speed error is COMMON
to the video capture and the HiFi capture - it is the same tape moving past
the same drum at the same instant - while the crystal errors are independent.
So the difference of the two measured ratios is the relative crystal drift
with the tape's own speed cancelled exactly, which is the quantity needed to
resample one onto the other.

TWO OBVIOUS ESTIMATORS OF IT WERE TRIED AND BOTH FAIL, which is why this
module takes the measurement as an argument rather than computing it:

  - THE COLOUR-UNDER BAND'S CENTROID IS NOT THE CARRIER. Measured at three
    tape positions of the off-air capture it gave -2953, +15608 and +20273
    ppm - swings of two per cent, because the band carries modulated chroma
    and the centroid follows the picture rather than the carrier.
  - `fileLoc` IN THE DECODED METADATA IS BLOCK-QUANTISED. Over 25415 fields
    of `countdown_on` the field-to-field step is 675840 samples with a
    standard deviation of EXACTLY ZERO, and 675840 is 660 * 1024. A real
    tape's field length jitters; a number that does not is a read-block
    boundary, not a measurement. Taken at face value it implies +12747 ppm,
    which is an artefact of the quantisation.

So the clock relation must be measured at decode time from the line locations
IN INPUT SAMPLES, before any resampling to the 4fsc output grid, and no
artefact this repository currently exports carries it. `clock_relation` is
written to consume that measurement the moment it exists.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np

# Exact rationals, so nothing here is a rounded constant. NTSC 525/60.
NTSC_LINE_RATE = Fraction(30000, 1001) * 525
NTSC_SUBCARRIER = Fraction(455, 2) * NTSC_LINE_RATE
# SMPTE 32M: the VHS NTSC colour-under carrier is 40 times the line rate.
COLOUR_UNDER_MULTIPLE = Fraction(40)
NTSC_COLOUR_UNDER = COLOUR_UNDER_MULTIPLE * NTSC_LINE_RATE

# PAL VHS is NOT an integer multiple - it is 40 + 1/8 line periods, which is
# why its heterodyne advances 45 degrees per line on an eight-line cycle.
PAL_LINE_RATE = Fraction(15625)
PAL_COLOUR_UNDER_MULTIPLE = Fraction(321, 8)


def specified_relations(system: str = "NTSC") -> Dict[str, object]:
    """The exact tie between the colour carrier and the line rate.

    Returned as `Fraction`s as well as floats, because the whole value of
    this relation is that it is EXACT - a rounded ratio would put a
    parts-per-million error into a measurement whose entire purpose is to
    resolve parts per million.
    """
    name = str(system).upper()
    if name in ("NTSC", "525", "M", "525/60"):
        line, multiple = NTSC_LINE_RATE, COLOUR_UNDER_MULTIPLE
        subcarrier = NTSC_SUBCARRIER
    elif name in ("PAL", "625", "625/50"):
        line, multiple = PAL_LINE_RATE, PAL_COLOUR_UNDER_MULTIPLE
        subcarrier = Fraction(4433618750, 1000)
    else:
        raise ValueError(f"no specified colour-under relation for {system!r}")
    carrier = multiple * line
    return {
        "system": name,
        "line_rate_hz": float(line),
        "line_rate_exact": line,
        "colour_under_hz": float(carrier),
        "colour_under_exact": carrier,
        "colour_under_multiple": multiple,
        "subcarrier_hz": float(subcarrier),
        "carrier_over_subcarrier": carrier / subcarrier,
        "integer_multiple_of_line_rate": multiple.denominator == 1,
        "why": ("the colour-under carrier is rigidly tied to the line rate, "
                "so its position in units of the sample rate is a direct "
                "reading of the capture clock against the tape's time base"),
    }


def clock_relation(samples_per_line: float, sample_rate_hz: float,
                   system: str = "NTSC") -> Dict[str, float]:
    """THE CAPTURE CLOCK AGAINST THE TAPE'S OWN TIME BASE.

    `samples_per_line` must be measured from the line locations IN INPUT
    SAMPLES - the raw capture's own grid - and NOT from a decoded product,
    which has already been resampled onto a nominal 4fsc grid and so carries
    no information about the crystal at all.

    Returns the colour carrier's position in cycles per sample, measured and
    nominal, and their difference in parts per million. That difference is
    the SUM of the tape's speed error and the crystal's error; neither is
    separable from a single capture, and separating them is exactly what a
    second capture of the same tape provides.
    """
    spec = specified_relations(system)
    measured_line_hz = float(sample_rate_hz) / max(float(samples_per_line), 1e-9)
    multiple = float(spec["colour_under_multiple"])
    measured_carrier = multiple * measured_line_hz
    nominal_cycles = float(spec["colour_under_hz"]) / float(sample_rate_hz)
    measured_cycles = multiple / max(float(samples_per_line), 1e-9)
    return {
        "samples_per_line": float(samples_per_line),
        "sample_rate_hz": float(sample_rate_hz),
        "measured_line_hz": measured_line_hz,
        "specified_line_hz": spec["line_rate_hz"],
        "measured_carrier_hz": measured_carrier,
        "specified_carrier_hz": spec["colour_under_hz"],
        "measured_cycles_per_sample": measured_cycles,
        "nominal_cycles_per_sample": nominal_cycles,
        "error_ppm": (measured_line_hz / spec["line_rate_hz"] - 1.0) * 1e6,
        "why": ("the departure is the tape's speed error PLUS the crystal's, "
                "and only a second capture of the same tape can split them"),
    }


def relative_drift(video: Dict[str, float], hifi: Dict[str, float]
                   ) -> Dict[str, float]:
    """THE QUANTITY THAT SYNCHRONISES TWO CAPTURES, with the tape cancelled.

    Both captures see the SAME tape at the same instant, so the tape's speed
    error is common to both and subtracts out exactly. What is left is the
    relative crystal drift, which is the resampling ratio one capture needs
    to land on the other's time base.

    Pass two `clock_relation` results. The tape's own speed error is not
    recovered and cannot be - it is the common mode this deliberately
    removes - which is the right trade, because synchronising the two does
    not require knowing how fast the tape ran, only that they agree.
    """
    ratio = (video["measured_line_hz"] / max(hifi["measured_line_hz"], 1e-30))
    return {
        "video_error_ppm": video["error_ppm"],
        "hifi_error_ppm": hifi["error_ppm"],
        "relative_drift_ppm": video["error_ppm"] - hifi["error_ppm"],
        "resample_ratio": ratio,
        "samples_to_resample_per_second": (ratio - 1.0)
        * float(hifi["sample_rate_hz"]),
        "why": ("the tape's speed error is common to both captures and "
                "cancels; what survives is the crystals' disagreement, which "
                "is the resampling ratio"),
    }


def collapse(entries: Dict[str, np.ndarray], frequency_hz,
             weights: Optional[Dict[str, float]] = None,
             chain: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    """COLLAPSE EVERY LINEAR COMPONENT INTO ONE COMPLEX SPECTRUM.

    The identity: a product of responses is a sum of their logarithms, so the
    correction that undoes all of them is a single `exp(-sum)`. This does
    that, and separates out the components that cannot be folded because they
    do not commute.

    THE ORDER IS STILL CHECKED even though the linear part does not need it.
    `interference.ordered_key` refuses an entry with no declared position,
    and that refusal is kept here rather than skipped as an optimisation:
    the ordering requirement exists because a chain whose order is unknown
    cannot be inverted, and the fact that the linear part happens to commute
    does not make an UNDECLARED entry safe - it makes it unnoticed.
    """
    from vhsdecode.models import interference as inf

    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    chain = chain if chain is not None else inf.full_chain()
    order = inf.ordered_key(entries, chain=chain)

    total = np.zeros(grid.size, dtype=np.complex128)
    folded: List[str] = []
    for _position, name in order:
        value = np.asarray(entries[name]).ravel()
        if value.size != grid.size:
            continue
        magnitude = np.log(np.maximum(np.abs(value), 1e-30))
        phase = (np.unwrap(np.angle(value)) if np.iscomplexobj(value)
                 else np.zeros_like(magnitude))
        weight = float((weights or {}).get(name, 1.0))
        total = total + weight * (magnitude + 1j * phase)
        folded.append(name)

    correction = np.exp(-total)
    return {
        "correction": correction,
        "log_sum": total,
        "frequency_hz": grid,
        "folded": folded,
        "order": order,
        "count": len(folded),
        "why": ("a product of responses is a sum of logarithms, so the whole "
                "linear chain is one exp(-sum) and applying it is one "
                "multiply rather than one pass per component"),
    }


def apply_correction(samples, correction, frequency_hz,
                     sample_rate_hz: float) -> np.ndarray:
    """Apply the collapsed spectrum to a block of samples: one FFT, one
    multiply, one inverse.

    THE INTERPOLATION IS DONE IN THE LOG DOMAIN, and that is not a detail.
    Interpolating a complex value linearly between grid points chords across
    the arc it should follow, so a correction whose phase turns loses
    magnitude wherever it turns fastest - the error grows with the phase step
    and is worst exactly where the correction is doing the most work.
    Interpolating the log magnitude and the UNWRAPPED phase separately
    follows the arc instead, and both are smooth by construction.

    Measured against the same analytic model evaluated exactly on the
    block's own grid, phase steps of 0.016 radians per bin: complex
    interpolation 1.45e-04, log interpolation 1.14e-04. A 1.3x improvement,
    which is modest BECAUSE the phase turns slowly here - the advantage grows
    with the phase step, and is the reason to prefer the log form rather than
    a result to quote on its own.

    THE IDENTITY ITSELF IS EXACT TO 4.8e-15; it is only ever the regridding
    that costs anything. And a caller who can build the model on the block's
    own grid should do that instead of interpolating at all.

    A TRAP WORTH KNOWING BEFORE COMPARING TWO GRIDS. `interference.signatures`
    derives several of its parameters FROM THE GRID'S OWN EXTENT - the echo
    delays default to `k / (f_max - f_min)`, and the dropout width, the beat
    centre and the vestigial edge are all fractions of the span. So the same
    call on two grids returns two DIFFERENT MODELS: on 1-7 MHz the echoes are
    at 0.17, 0.33, 0.67 and 1.33 us, and on 0-20 MHz the same call gives
    0.05, 0.10, 0.20 and 0.40 us. An attempt to validate this function by
    rebuilding the key on the finer grid measured a relative error of 105,
    which was two different models being compared rather than any defect in
    the regridding.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    value = np.asarray(correction).ravel()
    target = np.fft.rfftfreq(x.size, 1.0 / float(sample_rate_hz))
    magnitude = np.log(np.maximum(np.abs(value), 1e-30))
    phase = np.unwrap(np.angle(value))
    got_mag = np.interp(target, grid, magnitude, left=0.0, right=0.0)
    got_phase = np.interp(target, grid, phase, left=0.0, right=0.0)
    shaped = np.exp(got_mag + 1j * got_phase)
    # outside the modelled band the correction is exactly unity, not an
    # extrapolation of it
    shaped[(target < grid[0]) | (target > grid[-1])] = 1.0
    return np.fft.irfft(shaped * np.fft.rfft(x), n=x.size)


def export(path: str, frequency_hz, correction, sample_rate_hz: float,
           clock: Optional[Dict[str, float]] = None,
           ordered: Optional[Sequence] = None,
           uncollapsed: Optional[Sequence[str]] = None,
           metadata: str = "") -> Dict[str, object]:
    """Write the collapsed residual and everything needed to apply it.

    Deliberately self-describing: a consumer that has only this file must be
    able to apply the correction and to check the clock, without reading this
    repository. That is what "export a model I can use against my hifi
    capture" requires.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    value = np.asarray(correction).ravel()
    payload = {
        "frequency_hz": grid,
        "correction_real": value.real,
        "correction_imag": value.imag,
        "sample_rate_hz": float(sample_rate_hz),
        "folded": np.array(list(ordered or []), dtype=object),
        "must_be_applied_in_order": np.array(list(uncollapsed or []),
                                             dtype=object),
        "metadata": metadata or (
            "Collapsed linear correction: multiply the block's rfft by "
            "(correction_real + 1j*correction_imag) interpolated onto the "
            "block's own grid, then inverse transform. Entries listed in "
            "must_be_applied_in_order do NOT commute and are not folded in."),
    }
    if clock:
        for key, item in clock.items():
            if isinstance(item, (int, float)):
                payload[f"clock_{key}"] = float(item)
    np.savez_compressed(path, **payload)
    return {"path": path, "bins": int(grid.size),
            "folded": len(list(ordered or [])),
            "uncollapsed": len(list(uncollapsed or []))}
