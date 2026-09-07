"""CHROMA LEFT IN THE LUMA, and the test that asserts there is none.

Ethan, 2026-09-06: *"I am still seeing chroma leakage in the luma ... The
test should be written that asserts no chroma leakage from the color under
and the upconverted color exists in the luma channel."*

TWO CHANNELS, NOT ONE, and he names both. The colour-under sits at forty
times the line rate, 629.371 kHz (SMPTE 32M clause 3.9.2.1.4), and the
up-converted colour sits at the subcarrier, 3.579545 MHz. They are
different leaks with different causes: the first is the recorded carrier
surviving the decoder's separation, the second is the reconstructed colour
finding its way back into the luma after up-conversion. A luma channel is
clean only when neither is there.

THE FRAMING HAS TO BE ACCOUNTED FOR OR THE TEST LIES. Ethan, on being told
the coupling measured near zero: *"No, it does contain the color framing,
we have all the parts related together, and the color framing is spec
driven."* He is right, and the arithmetic is in `colour_framing.sequence`:
the subcarrier advances 227.5 cycles a line and so 270 degrees a field,
which closes after four fields. A leak carrying that sequence has a phase
that turns three quarters of a turn between one field and the next, so
averaging fields blindly drives it toward zero and reports a clean channel
that is not clean. This module therefore pools under each of the four
framing rotations and keeps the largest, and the difference between the
blind figure and that one is itself diagnostic.

AND THE COLOUR-UNDER IS THE EXCEPTION THAT PROVES IT.
`colour_framing.colour_under_field_advance` shows that forty times the
line rate advances exactly 10500 cycles over a field of 262.5 lines, a
fractional part of zero, so the colour-under carrier has no per-field
phase to carry. That is true of the colour-under and of nothing else; the
up-converted channel is where the sequence lives, and reading the
colour-under's result as though it settled the question for both is the
mistake this module exists to prevent.

MEASURED ON THREE DECODES, active lines only, the luma against the chroma
as a normalised complex resultant:

    decode              channel          per field   blind   framed   control
    bars SP             subcarrier         0.0736   0.0575   0.0575    0.0155
    bars SP             colour-under       0.0469   0.0197   0.0197    0.0155
    home                subcarrier         0.0298   0.0073   0.0073    0.0042
    home                colour-under       0.0530   0.0089   0.0089    0.0042
    chroma noise SP     subcarrier         0.2098   0.0225   0.2080    0.0468
    chroma noise SP     colour-under       0.0520   0.0259   0.0324    0.0468

So the leak is real: on the bars decode the subcarrier channel pools to
3.7 times its control. And the chroma-noise decode shows the framing doing
exactly what it was predicted to do - blind pooling collapses 0.2098 to
0.0225 while allowing the framing recovers 0.2080, a factor of nine that a
blind test would have thrown away and called cleanliness.

THE COLOUR-UNDER COLUMN OF THAT TABLE IS WITHDRAWN, and the reason is worth
more than the figures were. Those readings were taken before `_amplitude_at`
removed each line's own mean, and the sum is a rectangular window whose
response to a CONSTANT is 6.95 at the colour-under carrier over a 760 sample
active span - see `window_response_to_a_constant`. An offset of a single
count, which is exactly what the sixteen-bit chroma conversion leaves behind,
therefore arrived multiplied by seven and was read as colour. Re-measured with
the mean removed, on a decode of `zaroff-75bars-NTSC-SP` taken for this
purpose, the colour-under channel falls from 4.04 times its control to 0.78
and the subcarrier channel is unmoved at 4.04. The SUBCARRIER column stands -
760 samples is 190 whole subcarrier cycles, so that channel sat on an exact
transform bin and its response to a constant was 2.8e-15 - and the
colour-under figures are to be taken again.

That is one number changing the sign of a finding, which is what an estimator
whose conditioning is not reported does. Every reading now carries its own
window response, so the next one cannot do it quietly.

AND THE ROTATION IT TURNS BY IS NOT THE COLOUR FRAME'S, which was not
expected and is the more useful half of the result. Reading which of the
four states wins on each decode:

    decode              channel        best state   turn per field
    bars SP             subcarrier          0       none
    bars SP             colour-under        0       none
    home                subcarrier          0       none
    home                colour-under        0       none
    chroma noise SP     subcarrier          2       180 degrees
    chroma noise SP     colour-under        2       180 degrees
    bars EP             subcarrier          2       180 degrees
    bars EP             colour-under        0       none

(The colour-under rows are subject to the withdrawal above; the subcarrier
rows are not, and they are the ones the finding rests on.)

The colour frame would put the resultant on state one, since the statistic
is a conjugate product and turns by the negative of the chroma's own 270
degrees. Nothing lands there. Two decodes land on state two instead, which
is half a turn a field - the rate at which the HEAD alternates, one drum
revolution being two fields. So on those decodes the leak is keyed to the
head and not to the colour sequence, and the head switch is where to look
for it.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

from vhsdecode.models import colour_framing, colour_under

# The control offset. It has to be far enough from both carriers to carry
# none of either and inside the band the luma actually occupies, and it
# must not be a harmonic or a sub-harmonic of the line rate that the
# picture's own structure would fill. 0.83 of the subcarrier is 2.97 MHz,
# which satisfies all three, and it is an ASSUMPTION in the sense that the
# standard nominates no such frequency.
CONTROL_FRACTION_OF_SUBCARRIER = 0.83
CONTROL_STATUS = "assumed: 0.83 of the subcarrier, nominated by no standard"


def channels(sample_rate_hz: float, system: str = "NTSC"
             ) -> Dict[str, float]:
    """The two leaks Ethan names, and the control, each from its clause.

    At four times the subcarrier the container's own quarter-rate IS the
    subcarrier, so it is taken from the sample rate rather than retyped.
    """
    return {
        "subcarrier": float(sample_rate_hz) / 4.0,
        "colour-under": colour_under.carrier_hz(system),
        "control": CONTROL_FRACTION_OF_SUBCARRIER * float(sample_rate_hz) / 4.0,
    }


def window_response_to_a_constant(hz: float, samples: int,
                                  sample_rate_hz: float) -> float:
    """WHAT A CONSTANT CONTRIBUTES to a reading taken over this window.

    The statistic is one transform bin summed over a rectangle, so its
    response to a constant is the Dirichlet kernel `|sin(pi f N / fs) /
    sin(pi f / fs)|`. That is ZERO only where the frequency lands on an
    exact bin of an `N` sample window, and it is the conditioning the
    reading carries otherwise.

    IT IS NOT SMALL, AND THAT IS WHY THIS IS EXPORTED. Over a 525-line
    active span of 760 samples at four times the subcarrier the response is
    2.8e-15 at the subcarrier - 760 samples is 190 whole cycles, so the
    subcarrier is exactly on a bin - but 6.95 at the colour-under carrier
    and 1.33 at the control. Measured on a real decode of
    `zaroff-75bars-NTSC-SP`, an offset of ONE COUNT left in the chroma by
    the sixteen-bit conversion moved the colour-under reading from 0.77
    times its control to 4.04 times it, which would have been reported as a
    leak five times over. Hence the mean removal in `_amplitude_at`, and
    hence this function beside it so a caller can see what its window is
    doing rather than trusting that it does nothing.
    """
    turn = math.pi * float(hz) / float(sample_rate_hz)
    denominator = abs(math.sin(turn))
    if denominator <= 0.0:
        return float("inf")
    return float(abs(math.sin(turn * float(samples))) / denominator)


def _amplitude_at(lines, hz: float, sample_rate_hz: float,
                  start: int, stop: int) -> np.ndarray:
    """The complex amplitude at one frequency over each line's active span.

    EACH LINE'S OWN MEAN OVER THAT SPAN IS REMOVED FIRST, and it is not a
    nicety. The sum is a rectangular window, so a constant reaches the
    reading multiplied by `window_response_to_a_constant` above - 6.95 at
    the colour-under carrier over a 760 sample span - and a channel offset
    of a single count is then enough to manufacture a leak. Removing the
    mean over the SAME span the sum is taken on makes that product exactly
    zero whatever the window is, so the reading is of the line's alternating
    content and of nothing else. It changes no planted test here, because
    planted material is already centred, and it changes the reading on real
    material by a factor of five.
    """
    values = np.asarray(lines, dtype=np.float64)
    index = np.arange(values.shape[-1], dtype=np.float64)
    mixed = values * np.exp(-2j * np.pi * float(hz) * index
                            / float(sample_rate_hz))
    segment = mixed[..., start:stop]
    centred = np.asarray(values)[..., start:stop].mean(axis=-1, keepdims=True)
    offset = (centred * np.exp(-2j * np.pi * float(hz) * index[start:stop]
                               / float(sample_rate_hz))).sum(axis=-1)
    return segment.sum(axis=-1) - offset


def resultant(luma_lines, chroma_lines, hz: float, sample_rate_hz: float,
              active: Sequence[int]) -> Dict[str, object]:
    """How much of the chroma is in the luma at one frequency, per field.

    A normalised complex inner product per field: its magnitude is the
    share and its angle is the phase the leak arrives at. Complex, because
    the angle is what the framing acts on and a magnitude alone cannot be
    pooled correctly.
    """
    start, stop = int(active[0]), int(active[1])
    luma = _amplitude_at(luma_lines, hz, sample_rate_hz, start, stop)
    chroma = _amplitude_at(chroma_lines, hz, sample_rate_hz, start, stop)
    product = (luma * np.conj(chroma)).sum(axis=-1)
    norm = np.sqrt((np.abs(luma) ** 2).sum(axis=-1)
                   * (np.abs(chroma) ** 2).sum(axis=-1))
    per_field = product / np.maximum(norm, 1e-300)
    return {
        "per_field": per_field,
        "magnitude": np.abs(per_field),
        "mean_magnitude": float(np.abs(per_field).mean()),
        "frequency_hz": float(hz),
        "is_complex": True,
    }


def pool(per_field, system: str = "NTSC") -> Dict[str, object]:
    """Pool the per-field resultants blindly and under the framing.

    The blind average is what a test that ignores the colour frame would
    report. The framed one tries each of the sequence's rotations and
    keeps the largest, which is the honest figure when the leak turns with
    the frame. Their ratio says whether the framing was load-bearing.
    """
    values = np.asarray(per_field, dtype=np.complex128).ravel()
    if values.size < 2:
        raise ValueError("pooling needs more than one field")
    order = colour_framing.sequence(system)
    states = int(order["fields_in_sequence"])
    index = np.arange(values.size, dtype=np.float64)
    blind = complex(values.mean())
    trials = {}
    for state in range(states):
        turn = 2.0 * math.pi * state / states
        trials[state] = complex((values * np.exp(-1j * turn * index)).mean())
    best_state = max(trials, key=lambda k: abs(trials[k]))
    spec_state = int(round(float(order["degrees_per_field"]) / 360.0 * states)) % states
    # THE RESULTANT TURNS THE OTHER WAY, and the sign is not a detail. The
    # statistic is luma times the CONJUGATE of the chroma, so if the chroma
    # advances by the sequence's 270 degrees a field the product advances
    # by minus that, which is state one of four and not state three. A
    # test written against the chroma's own state fails on a correct
    # measurement, which is how this was found.
    resultant_state = (states - spec_state) % states
    return {
        "blind": blind,
        "blind_magnitude": abs(blind),
        "framed": trials[best_state],
        "framed_magnitude": abs(trials[best_state]),
        "best_state": best_state,
        "specified_state": spec_state,
        "expected_resultant_state": resultant_state,
        "matches_specified": bool(best_state == resultant_state),
        "by_state": trials,
        "fields_in_sequence": states,
        "degrees_per_field": float(order["degrees_per_field"]),
        "framing_was_load_bearing": bool(
            abs(trials[best_state]) > 2.0 * abs(blind)),
        "why": ("a leak that turns with the colour frame averages toward "
                "zero when the fields are pooled blindly, so a blind test "
                "reports a clean channel that is not clean"),
    }


def assert_clean(luma_lines, chroma_lines, sample_rate_hz: float,
                 active: Sequence[int], system: str = "NTSC",
                 margin: float = 2.0) -> Dict[str, object]:
    """THE TEST: is there any chroma in this luma, from either channel?

    Both of Ethan's channels are measured against the same control, under
    the framing, and the verdict is that the luma is clean only when
    neither exceeds the control by `margin`. The control is what makes the
    threshold a measurement rather than a number: it is the same statistic
    computed where no chroma can be, so it carries the estimator's own
    bias and the picture's own structure, and a leak has to beat it.
    """
    rates = channels(sample_rate_hz, system)
    readings, verdicts = {}, {}
    control = pool(resultant(luma_lines, chroma_lines, rates["control"],
                             sample_rate_hz, active)["per_field"],
                   system)
    floor = max(control["framed_magnitude"], control["blind_magnitude"])
    for name in ("subcarrier", "colour-under"):
        found = resultant(luma_lines, chroma_lines, rates[name],
                          sample_rate_hz, active)
        pooled = pool(found["per_field"], system)
        level = max(pooled["framed_magnitude"], pooled["blind_magnitude"])
        readings[name] = {"resultant": found, "pooled": pooled,
                          "level": level,
                          "over_control": level / max(floor, 1e-300)}
        verdicts[name] = bool(level <= float(margin) * floor)
    return {
        "channels": readings,
        "control_level": floor,
        "clean": all(verdicts.values()),
        "clean_by_channel": verdicts,
        "margin": float(margin),
        "control_status": CONTROL_STATUS,
        "why": ("the luma is clean only when neither the colour-under nor "
                "the up-converted colour beats the control, and the "
                "control is measured on the same material rather than "
                "assumed"),
    }
