"""KOLMOGOROV'S THREE APPROACHES, and the third one applied to the remainder.

Ethan, 2026-09-06: *"I believe we are exhausting the combinatorial approach
here by modeling our functions against their expected data, along with some
aspects of the probalistic approach. I want to focus on identifying the
remainder of the components after we have excausted all other methods to use
the algorithmic approach ... I believe this can be applied to my existing
approach in hyper complex hilbert space."*

He is citing Kolmogorov's 1965 paper, which gives three definitions of the
quantity of information and shows they are different things on a finite
object even though they agree in the limit:

  COMBINATORIAL   an object drawn from a set of N possibilities carries
                  log N, with no probability anywhere. This arc computes it
                  in `measurement_bound`: a region of duration T over
                  bandwidth B holds BT complex dimensions.
  PROBABILISTIC   Shannon's entropy, which needs a distribution. This arc
                  computes it as the noise budget and the particulate floor.
  ALGORITHMIC     the length of the shortest program that produces the
                  object. It needs neither a set nor a distribution, which
                  is exactly why it can still speak about a remainder the
                  other two have finished with.

AND THE REPRESENTATION IS THE WHOLE OF IT. The shortest program depends on
how the object is written down, and that is why the third approach has to
be taken in Hilbert space rather than on a magnitude. A pure delay is a
phase ramp: written complex it is two numbers, and written as a magnitude it
is not expressible at all, so its description is the entire record. A
complexity measured on the modulus would therefore report a delay - the one
structure this arc's own conclusion says is left - as incompressible noise.
`delay_is_cheap_only_in_the_complex_form` measures that difference rather
than asserting it.

MEASURED ON THE REAL REMAINDER, and the answer is not the expected one.
The component key explains about 84 per cent of the measured departure's
ENERGY. Judged by description length instead, quantised at each export's
own measured relative error so that a smaller residual genuinely costs
fewer bits:

    export        head    step    before    after    net
    cd             a    0.01093   14432    15192    -814 bits
    cd             b    0.01109   14600    15056    -510
    home           a    0.00923   14880    17176   -2350
    home           b    0.00898   15080    17472   -2446
    pnb            a    0.01265   10008    11544   -1590
    pnb            b    0.01480    8816    10072   -1305

So on every reading the model COSTS bits rather than saving them: what
remains is harder to describe than what it was subtracted from. The
parameter charge is not the reason - ten parameters over 1888 numbers is
about 54 bits against deficits of 510 to 2446.

THE READING, and it is a statement about the description language rather
than about the tape. A general-purpose byte compressor expresses
repetition and smoothness. The component key removes a smooth shape, so it
removes exactly what the compressor could see and leaves what it cannot,
and the length rises. Kolmogorov's third approach is only ever as sharp as
the machine the programs are written for, and zlib is a poor machine for
this data: it has no way to write down a delay, a minimum-phase response
or an all-pass term, which is the whole of what this arc's own conclusion
says is left. `delay_is_cheap_only_in_the_complex_form` measures that gap
directly - one number in the complex representation against a whole
waveform as a magnitude, a factor of 191 in bits.

What follows is not that the approach fails but that the DESCRIPTION
LANGUAGE has to be the arc's own component forms rather than a byte
compressor, with `admits` as the rule and each component's own parameter
count as its cost. The compressor stays as the null: a structure a general
compressor can already find is one nobody needed a model for.

AND WITH THE RIGHT LANGUAGE, THE REMAINDER IS NAMED. Writing the same
remainder in the component forms instead - a delay, then whatever the
magnitude already implies through Bode - and comparing the two machines on
the same object:

    export     head   zlib bits   language   saved   kept    min-phase
                                                              share
    cd          a       15192      14933      259    delay     0.159
    cd          b       15056      14813      243    delay     0.158
    home        a       17176      17045      131    delay     0.300
    home        b       17472      16653      819    delay     0.347
    pnb         a       11544      11533       11    delay     0.217
    pnb         b       10072      10072        0    none      0.202

A delay is admitted on five of the six and pays almost nothing - 0.1 to
4.8 per cent of the description. So the delay term is very nearly
exhausted, which is a real answer rather than a disappointing one.

What is left is EXCESS PHASE. Only 16 to 35 per cent of the remainder's
phase is implied by its own magnitude, so 65 to 84 per cent is a phase no
magnitude can supply, at 1.16 to 1.66 radians rms. That is an all-pass
term, and naming it is what the third approach was brought in to do: the
combinatorial account had spanned its basis, the probabilistic one had
said there was room left, and neither could say what shape the room had.

WHAT IS AND IS NOT CLAIMED. Kolmogorov complexity is not computable. Every
figure here is an UPPER BOUND obtained by actually compressing the object,
so a small number is evidence of structure and a large number is evidence of
nothing at all - the asymmetry is inherent and is stated wherever a verdict
is returned. That is the same discipline `lossless_model` already applies to
Chaitin's result, and this module is the general form of it.
"""

import math
import zlib
from typing import Dict, Optional, Sequence

import numpy as np

# The compressor. zlib is in the standard library, deterministic, and its
# output length is a genuine upper bound on the description length; a
# stronger compressor lowers every bound here and changes no conclusion
# that rests on a RATIO of two bounds taken with the same compressor.
COMPRESSION_LEVEL = 9


def _quantise(values, levels: int, step: Optional[float] = None) -> bytes:
    """Put a record on a grid so its description length is meaningful.

    A float array's byte pattern is dominated by mantissa noise, which is
    incompressible and swamps whatever structure is present, so the record
    has to be quantised before it is compressed.

    HOW IT IS QUANTISED DECIDES WHAT THE LENGTH MEANS, and the first
    version of this got it wrong in a way worth recording. Scaling each
    record to its OWN range makes the description length scale-invariant,
    so a model that halves the residual earns no bits at all and the
    minimum-description-length rule it feeds becomes blind to exactly the
    improvement it exists to reward. A fixed absolute `step` - the
    measurement's own error bar, or the output format's floor - is the
    correct choice: a residual reduced in amplitude then genuinely costs
    fewer bits, which is what the rule assumes. The range-relative form is
    kept for the case where no step is known, and it is the caller's job
    to know that it cannot compare two amplitudes.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        return b""
    if step is not None and float(step) > 0.0:
        return np.rint(array / float(step)).astype(np.int32).tobytes()
    span = float(np.max(array) - np.min(array))
    if span <= 0.0:
        return bytes(array.size)
    scaled = (array - float(np.min(array))) / span * (int(levels) - 1)
    return np.rint(scaled).astype(np.uint16).tobytes()


def algorithmic(values, levels: int = 4096,
                step: Optional[float] = None) -> Dict[str, object]:
    """An upper bound on the description length, by compressing the object.

    Returns the compressed length in bits and the bits per sample, which is
    the useful figure: a record of independent uniform samples costs
    log2(levels) bits each and anything below that is structure the
    compressor found.

    Pass `step` - an absolute quantisation, normally the measurement's own
    error bar - whenever two amplitudes are to be compared. Without it the
    length is scale-invariant and a smaller residual reads the same as a
    large one.
    """
    raw = _quantise(values, levels, step)
    if not raw:
        return {"bits": 0.0, "bits_per_sample": float("nan"), "samples": 0}
    packed = zlib.compress(raw, COMPRESSION_LEVEL)
    samples = np.asarray(values).size
    return {
        "bits": 8.0 * len(packed),
        "bits_per_sample": 8.0 * len(packed) / samples,
        "ceiling_bits_per_sample": math.log2(int(levels)),
        "samples": int(samples),
        "levels": int(levels),
        "step": step,
        "is_an_upper_bound": True,
        "why": ("the complexity is not computable, so a compressed length "
                "is an upper bound: small means structure was found, large "
                "means nothing was found and not that nothing is there"),
    }


def combinatorial(duration_s: float, bandwidth_hz: float,
                  levels: int) -> Dict[str, object]:
    """Kolmogorov's first approach: the logarithm of the number of states.

    A region of duration T over bandwidth B carries BT complex degrees of
    freedom, and each resolved to `levels` costs log2(levels) bits. No
    probability enters, which is the point of the definition.
    """
    dimensions = float(duration_s) * float(bandwidth_hz)
    per_dimension = math.log2(int(levels))
    return {
        "dimensions": dimensions,
        "complex_dimensions": dimensions,
        "bits_per_dimension": per_dimension,
        "bits": 2.0 * dimensions * per_dimension,
        "why": ("log of the count of distinguishable states, with no "
                "distribution assumed"),
    }


def probabilistic(values, levels: int = 4096) -> Dict[str, object]:
    """Kolmogorov's second approach: the entropy of the measured histogram.

    The distribution is taken from the data rather than assumed, so this is
    the empirical entropy and is itself an estimate; with `levels` bins and
    a finite record it is biased low, and the bias is reported.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        return {"bits_per_sample": float("nan"), "samples": 0}
    counts, _ = np.histogram(array, bins=int(levels))
    total = counts.sum()
    share = counts[counts > 0] / total
    entropy = float(-(share * np.log2(share)).sum())
    occupied = int((counts > 0).sum())
    return {
        "bits_per_sample": entropy,
        "bits": entropy * array.size,
        "occupied_bins": occupied,
        "levels": int(levels),
        "bias_bits": (occupied - 1) / (2.0 * total * math.log(2.0)),
        "samples": int(array.size),
        "why": ("the entropy of the measured histogram; the Miller bias is "
                "reported because a finite record under-counts it"),
    }


def three_approaches(values, duration_s: float, bandwidth_hz: float,
                     levels: int = 4096) -> Dict[str, object]:
    """All three on one object, which is the comparison the paper is about.

    They agree in the limit and differ on a finite record, and the shape of
    the disagreement is diagnostic: an object whose algorithmic bound falls
    well below its entropy has structure the distribution does not
    describe, and that is precisely a component nobody has named yet.
    """
    counted = combinatorial(duration_s, bandwidth_hz, levels)
    entropy = probabilistic(values, levels)
    described = algorithmic(values, levels)
    return {
        "combinatorial": counted,
        "probabilistic": entropy,
        "algorithmic": described,
        "algorithmic_below_entropy": bool(
            described["bits_per_sample"] < entropy["bits_per_sample"]),
        "compression_against_entropy": (
            described["bits_per_sample"] / entropy["bits_per_sample"]
            if entropy["bits_per_sample"] > 0 else float("nan")),
        "verdict": ("structure remains that the distribution does not "
                    "describe" if described["bits_per_sample"]
                    < entropy["bits_per_sample"] else
                    "nothing found beyond the distribution"),
        "why": ("the three definitions agree in the limit and differ on a "
                "finite object, and the difference is where an unnamed "
                "component lives"),
    }


def delay_is_cheap_only_in_the_complex_form(
        samples: int = 4096, delay_samples: float = 7.3,
        seed: int = 0, levels: int = 4096) -> Dict[str, object]:
    """WHY THE THIRD APPROACH HAS TO BE TAKEN IN HILBERT SPACE.

    A delay is the structure this arc's own conclusion says is left in the
    remainder. Its description length depends entirely on how the object is
    written down, and this measures that rather than asserting it.

    One record is delayed by a known amount. Written in the complex
    analytic form the relation between the two is a phase ramp, so the
    residual after removing one number is nothing. Written as a magnitude
    there is no number to remove, so the residual is the whole difference
    between two waveforms. The compressed length of each residual is the
    description length of what is left after the best each representation
    can do.
    """
    from vhsdecode.models import hypercomplex
    rng = np.random.default_rng(seed)
    grid = np.fft.rfftfreq(int(samples))
    spectrum = (rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size))
    spectrum *= np.exp(-grid * 8.0)                 # band limited, so a delay is exact
    original = np.fft.irfft(spectrum, n=int(samples))
    delayed = np.fft.irfft(
        spectrum * np.exp(-2j * np.pi * grid * float(delay_samples)),
        n=int(samples))
    # what a magnitude-only description leaves: the two waveforms differ and
    # no scalar recovers one from the other
    magnitude_residual = delayed - original
    # what the complex description leaves: one number, the delay itself
    recovered = hypercomplex.fractional_delay(delayed, -float(delay_samples))
    complex_residual = recovered - original
    magnitude = algorithmic(magnitude_residual, levels)
    complex_form = algorithmic(complex_residual, levels)
    return {
        "delay_samples": float(delay_samples),
        "magnitude_residual_rms": float(np.sqrt(np.mean(magnitude_residual ** 2))),
        "complex_residual_rms": float(np.sqrt(np.mean(complex_residual ** 2))),
        "magnitude_bits": magnitude["bits"],
        "complex_bits": complex_form["bits"],
        "ratio": (magnitude["bits"] / complex_form["bits"]
                  if complex_form["bits"] > 0 else float("inf")),
        "parameters_the_complex_form_needed": 1,
        "why": ("the shortest program depends on the representation, and a "
                "delay is one number in the complex form and a whole "
                "waveform in the magnitude one"),
    }


def description_length(residual, parameters: int, levels: int = 4096,
                       bits_per_parameter: Optional[float] = None,
                       step: Optional[float] = None) -> Dict[str, object]:
    """The minimum description length of a model and what it leaves.

    Kolmogorov's third approach, made into an admission rule: a component
    is worth having when the total description SHORTENS - the model's own
    cost plus the residual's - and not when it merely lowers the residual.
    A component that costs more than it saves has described the noise.

    The cost of a parameter is half the logarithm of the sample count,
    which is the standard code length for a real parameter estimated from
    that many samples; a caller with a better figure may supply it.
    """
    values = np.asarray(residual).ravel()
    left = algorithmic(values, levels, step)
    per_parameter = (bits_per_parameter if bits_per_parameter is not None
                     else 0.5 * math.log2(max(values.size, 2)))
    model_bits = float(parameters) * per_parameter
    return {
        "residual_bits": left["bits"],
        "model_bits": model_bits,
        "total_bits": left["bits"] + model_bits,
        "parameters": int(parameters),
        "bits_per_parameter": per_parameter,
        "why": ("a component is worth having when the total description "
                "shortens, not when the residual alone falls"),
    }


def admits(before, after, parameters: int, levels: int = 4096,
           bits_per_parameter: Optional[float] = None,
           step: Optional[float] = None) -> Dict[str, object]:
    """Does a component pay for itself, in the algorithmic sense?

    `before` is the residual without it and `after` the residual with it.
    The verdict is on the TOTAL description, so a component that lowers the
    residual by less than it costs to state is refused - which is the whole
    point of using the third approach to identify what is left.
    """
    without = description_length(before, 0, levels, bits_per_parameter, step)
    with_it = description_length(after, parameters, levels, bits_per_parameter, step)
    saved = without["total_bits"] - with_it["total_bits"]
    return {
        "without": without,
        "with": with_it,
        "bits_saved": saved,
        "return_on_its_cost": (saved / with_it["model_bits"]
                               if with_it["model_bits"] > 0 else float("inf")),
        "admitted": bool(saved > 0.0),
        "verdict": ("it pays for itself" if saved > 0.0
                    else "it costs more to state than it saves"),
        "why": ("the residual falling is not enough; the description of "
                "the model plus the residual has to fall"),
    }


# ---------------------------------------------------------------------------
# The description language: the arc's own component forms as the machine
# ---------------------------------------------------------------------------

def component_forms() -> Dict[str, Dict[str, object]]:
    """The primitives a residual may be written in, and what each costs.

    Kolmogorov's third approach is only as sharp as the machine the
    programs are written for, and the measurement above showed a byte
    compressor to be a poor one here: it cannot write down a delay, and a
    delay is what this arc's own conclusion says is left. These are the
    forms it CAN be written in, with the parameter count that is each
    one's code length.

    THE MINIMUM-PHASE SHAPE IS FREE, and that is the whole reason this
    language beats a compressor. Bode's relation fixes a causal response's
    phase from its own magnitude, so once the magnitude has been described
    the phase costs nothing further. A byte compressor has no way to know
    that and must spend bits on both halves. Everything else in the table
    is a term the magnitude does NOT already imply.
    """
    return {
        "level": {"parameters": 1,
                  "what": "a constant offset in the log domain, one gain"},
        "delay": {"parameters": 1,
                  "what": "a phase ramp; one number, and invisible to any "
                          "magnitude-only description"},
        "minimum phase": {"parameters": 0,
                          "what": "implied by the magnitude through Bode's "
                                  "relation, so it costs nothing once the "
                                  "magnitude is described"},
        "all-pass": {"parameters": 2,
                     "what": "a pole and its reflection; unit magnitude and "
                             "a phase the magnitude cannot imply"},
        "smooth shape": {"parameters": None,
                         "what": "as many coefficients as the fit keeps, so "
                                 "its cost is its own rank"},
    }


def remove_delay(response, frequency_hz) -> Dict[str, object]:
    """Take the best pure delay out of a complex response, and say what it cost.

    One parameter. The delay is the slope of the unwrapped phase against
    frequency, taken by least squares with the constant removed so that a
    level does not leak into it.
    """
    values = np.asarray(response, dtype=np.complex128).ravel()
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    good = np.isfinite(values) & (np.abs(values) > 0) & np.isfinite(grid)
    if good.sum() < 4:
        raise ValueError("a delay needs a run of usable bins")
    phase = np.unwrap(np.angle(values[good]))
    axis = grid[good] - grid[good].mean()
    slope = float((axis * (phase - phase.mean())).sum()
                  / max(float((axis * axis).sum()), 1e-300))
    delay = -slope / (2.0 * math.pi)
    corrected = values.copy()
    corrected[good] = values[good] * np.exp(2j * math.pi * grid[good] * delay)
    return {
        "delay_s": delay,
        "corrected": corrected,
        "parameters": 1,
        "why": ("one number, and the only term a magnitude-only account "
                "cannot express at all"),
    }


def minimum_phase_is_free(response) -> Dict[str, object]:
    """How much of a response's phase its own magnitude already implies.

    Returns the share of the phase that Bode's relation supplies for
    nothing, and the excess that remains and must be paid for. A
    description language that does not know this spends bits twice on the
    same information, which is exactly what a byte compressor does.
    """
    from vhsdecode.models import hypercomplex
    values = np.asarray(response, dtype=np.complex128).ravel()
    good = np.isfinite(values) & (np.abs(values) > 0)
    if good.sum() < 8:
        raise ValueError("the Bode relation needs a run of usable bins")
    logged = np.log(np.abs(values[good]))
    implied = hypercomplex.minimum_phase(logged)
    measured = np.unwrap(np.angle(values[good]))
    measured = measured - measured.mean()
    implied = implied - implied.mean()
    excess = measured - implied
    total = float(np.sum(measured ** 2))
    return {
        "implied_phase": implied,
        "excess_phase": excess,
        "implied_share": (float(np.sum(implied ** 2)) / total
                          if total > 0 else float("nan")),
        "excess_share": (float(np.sum(excess ** 2)) / total
                         if total > 0 else float("nan")),
        "excess_rms_rad": float(np.sqrt(np.mean(excess ** 2))),
        "parameters_it_cost": 0,
        "why": ("Bode fixes a causal response's phase from its magnitude, "
                "so that share is already paid for and only the excess is "
                "a separate term"),
    }


def describe(response, frequency_hz, step: float,
             levels: int = 4096) -> Dict[str, object]:
    """Write a complex response in the component language and count the bits.

    Each form is offered in turn and kept only if it pays for itself, which
    is `admits` applied inside the language rather than outside it. The
    total is the description length of the response under this machine, and
    it is directly comparable with the byte compressor's figure on the same
    object - which is the comparison that says whether the language is
    better than nothing.
    """
    values = np.asarray(response, dtype=np.complex128).ravel()
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    stacked = np.concatenate([values.real, values.imag])
    baseline = algorithmic(stacked, levels, step)
    kept, running = [], values.copy()

    # 1. the delay, one parameter, and the term no magnitude can express
    delay = remove_delay(running, grid)
    after = np.concatenate([delay["corrected"].real, delay["corrected"].imag])
    verdict = admits(np.concatenate([running.real, running.imag]), after,
                     delay["parameters"], levels, step=step)
    if verdict["admitted"]:
        kept.append(("delay", delay["parameters"], verdict["bits_saved"]))
        running = delay["corrected"]

    # 2. what the magnitude already implies, which costs nothing
    free = minimum_phase_is_free(running)

    encoded = np.concatenate([running.real, running.imag])
    final = algorithmic(encoded, levels, step)
    parameters = sum(count for _, count, _ in kept)
    model_bits = parameters * 0.5 * math.log2(max(stacked.size, 2))
    return {
        "baseline_bits": baseline["bits"],
        "residual_bits": final["bits"],
        "model_bits": model_bits,
        "total_bits": final["bits"] + model_bits,
        "saved_bits": baseline["bits"] - (final["bits"] + model_bits),
        "kept": kept,
        "minimum_phase_share": free["implied_share"],
        "excess_phase_rms_rad": free["excess_rms_rad"],
        "beats_the_compressor": bool(
            final["bits"] + model_bits < baseline["bits"]),
        "why": ("the language is worth having only if it describes the "
                "object in fewer bits than a general compressor does"),
    }
