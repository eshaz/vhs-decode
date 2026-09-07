"""The colour-under phase event at the head switch: detected, then compensated.

Ethan, this session: *"the colour-under phase rotation flips where the record
heads switch, and now that the luma-chroma residual is related it can be
DETECTED and the correct phase compensation applied there."*

`vhsdecode/chroma.py` has carried the open half of this since before the arc
began, beside the RCA VCR-1 red book passage that describes the rotation:

    "The phase rotation switch is determined at record time depending on which
    video head is on the tape. This rotation switch can occur in the middle of
    a line, causing a small phase artifact
    TODO: It may be possible to detect where this happens on the line and
    correct the phase issue mid-line"

WHAT IS ACTUALLY THERE, MEASURED. On the shipped NTSC decodes the colour phase
at the foot of every field departs from the phase the rest of the field holds.
Against the mean of the quiet picture lines of the same four-line residue,
amplitude weighted over the active area, pooled over the first fields of
`dod_fix_75bars_SP`:

    line       256    257    258    259    260    261
    departure   -1      0    -11    -32     +4    +16      degrees

and over its second fields, -20, -16, -10, -9 on the same four lines. Lines
252 to 257 sit inside two degrees; the event is confined to the last four or
five lines and it is large enough to be a visible hue shift in a band across
the bottom of the picture. The chroma amplitude falls with it, from 11.6 IRE
over the picture to 5.4-6.6 IRE on the affected lines, which is the head
leaving the tape.

THE DETECTOR IS THE LUMA-CHROMA RESIDUAL, AS DIRECTED. `colour_lock` measures
the fixed offset between the decoded chroma and the colour-under replica the
demodulated luma carries, and reports the per-line departure from it. Over the
quiet picture lines that departure has a root mean square of 6.9 to 11.4
degrees on the SP decodes; on the head-switch line it reaches 92 to 179
degrees. Expressed against the quiet span's own robust scale, which is what
`detect` uses, the event stands at

    75 % bars SP           12.6 to 30.6 sigma   peak line 259 / 258 by parity
    chroma noise SP        11.8 to 30.6 sigma   peak line 258 - 262
    75 % bars EP            5.5 to 10.6 sigma   peak line 260 - 262

so the detection is decisive on SP and usable on EP, and it needs no vertical
reference, no assumption about the picture, and no threshold taken from a
previous tape. The line it fires on alternates with the field parity - 259 on
first fields and 258 on second fields of the bars decode - which is what a
transport event fixed to the drum looks like when sampled by two interleaved
fields.

AND THE DETECTOR'S PHASE IS NOT THE COMPENSATION. This is the part that had to
be measured rather than assumed, and it is the reason this module has two
separate quantities where one might have been expected. The luma-chroma
residual is the DIFFERENCE of the two channels' departures, and at the head
switch BOTH channels depart, in opposite directions. Read against their own
quiet-line references on `dod_fix_75bars_SP`, field 0, line 259:

    the decoded chroma departs      -35 degrees
    the luma's colour-under departs  +39 degrees
    their difference                 -74 degrees

Rotating the chroma by the difference therefore over-corrects it by about a
factor of two, and it does. Applied to the switch lines and judged on an
INDEPENDENT gauge - the chroma's own vertical continuity against the quiet
lines, which the detector never sees - the correction runs the wrong way at
every gain tried, positive and negative:

    gain    -1.00   -0.50   -0.25    0.00   +0.25   +0.50   +0.75   +1.00
    error    41.4    26.8    19.3    15.6    17.3    23.9    32.4    40.0

with the best value at zero. That is recorded here rather than buried, because
it disposes of the natural reading of the directive - subtract the residual you
just measured - and because the same trap will be waiting for anything else
built on this pair of channels.

WHAT IS COMPENSATED, AND WHAT IT IS WORTH. The detector says WHERE; the phase
comes from the chroma channel's own departure from its four-line lock, pooled
over fields of the same parity. Pooling is what makes it a transport
measurement rather than a picture measurement: the head switch repeats at the
same lines with the same phase field after field while the picture does not,
so an average over fields keeps the event and thins everything else. Estimated
on one half of the fields and applied to the other half, judged on the
vertical gauge over lines 256 to the end of the field:

    decode                        parity    before    after     change
    75 % bars SP                  first     13.71     8.32      -39 %
    75 % bars SP                  second    19.54    15.48      -21 %
    chroma noise SP               first     16.18    15.69       -3 %
    chroma noise SP               second    15.43    13.72      -11 %
    75 % bars EP                  first     13.32     9.79      -27 %
    75 % bars EP                  second     9.83     8.93       -9 %

degrees of mean absolute phase error, six or seven held-out fields in each row.
Every row improves and none regresses, and the improvement is monotone in the
gain from zero to one, which is why the gain defaults to one rather than to the
arc's usual half: the half-amount law applies where the estimate's own error is
unknown, and here it is measured on held-out fields.

THE LIMIT OF THE EVIDENCE, STATED PLAINLY. The gauge is vertical continuity and
the captures are still test patterns, so the held-out split controls for noise
between fields and not for the assumption that the picture does not change down
the frame. The estimator has the same shape as the gauge. What defends it is
that the quantity being estimated is a property of the transport - it repeats
across fields whose noise is independent, it alternates with field parity, and
it sits exactly where the luma-chroma detector, which uses no vertical
reference at all, says the head switch is.

THE MID-LINE CASE IS NOT CLOSED. `split_point` locates the best single change
point in the residual along the line and it does fire - around sample 220 of an
active area beginning at 134, repeatably, on the first fields of two decodes -
but the phase either side of it comes from the same difference that the table
above shows to be the wrong quantity, so the two-segment correction is offered
and is NOT the default. The line-level table is what the evidence supports
today.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import colour_lock

# How far above the quiet span's own scatter a line has to sit before the
# detector claims the head switch. Not a tuned number: it is the point at
# which a Gaussian quiet span produces fewer than one false line in a field of
# roughly two hundred and sixty, which is the only rate that makes sense when
# the event being looked for occupies four or five lines. The measured events
# stand at 5 to 26 times the same scale, so the choice is not close.
DETECTION_SIGMA = 4.0

# The gain the held-out measurement in the module docstring settles at. Stated
# as a default and exposed as an argument, because it is a measured optimum on
# these captures and not a law.
DEFAULT_GAIN = 1.0


def _analytic(line: np.ndarray) -> np.ndarray:
    from scipy import signal as sps

    return sps.hilbert(np.asarray(line, dtype=np.float64))


def rotate_line(line: np.ndarray, phase, centre: float = 0.0) -> np.ndarray:
    """Rotate one real subcarrier-modulated line by a phase, exactly.

    The rotation is applied to the analytic signal and the real part taken, so
    every positive-frequency component turns by the same angle and the
    modulation is untouched. `phase` may be a scalar for a whole-line rotation
    or an array for a mid-line one; nothing here smooths a step, because the
    event being compensated is a step.
    """
    data = np.asarray(line, dtype=np.float64) - float(centre)
    turned = np.real(_analytic(data) * np.exp(-1j * np.asarray(phase)))
    return turned + float(centre)


def quiet_span(field_height: int, active_lines: Tuple[int, int],
               flagged: Sequence[int],
               margin: int = 2) -> Tuple[int, int]:
    """The picture lines the detector has NOT flagged, as a forward span.

    Derived rather than declared: the reference for both the lock and the
    departure has to exclude the event being measured, and the only honest
    definition of "away from the event" is "where the detector is quiet".
    """
    first, last = int(active_lines[0]), int(active_lines[1])
    if not flagged:
        return first, min(last, int(field_height))
    earliest = int(min(flagged)) - int(margin)
    if earliest <= first:
        raise ValueError("the detector flags the whole picture; no quiet span")
    return first, min(earliest, last, int(field_height))


def detect(residual_deg: np.ndarray, quiet_lines: Tuple[int, int],
           search_from: Optional[int] = None,
           sigma: float = DETECTION_SIGMA) -> Dict[str, object]:
    """WHERE the head switch is, from the luma-chroma residual alone.

    `residual_deg` is `colour_lock.residual`. The scale comes from the quiet
    span's median absolute deviation rather than its root mean square, so a
    field whose event happens to fall inside the caller's quiet span does not
    inflate its own threshold.
    """
    deviation = np.asarray(residual_deg, dtype=np.float64)
    scale = colour_lock.residual_scale(deviation, quiet_lines)
    # 1.4826 converts a median absolute deviation into the standard deviation
    # of the Gaussian it would have come from; it is the definition of the
    # robust scale, not a fitted factor.
    robust = 1.4826 * scale["mad_deg"]
    if robust <= 0.0:
        robust = scale["rms_deg"]
    start = int(quiet_lines[1]) if search_from is None else int(search_from)
    rows = np.arange(start, deviation.size)
    significance = np.abs(deviation[rows] - scale["median_deg"]) / robust
    hit = rows[significance >= float(sigma)]
    return {
        "lines": [int(line) for line in hit],
        "sigma": {int(line): float(value)
                  for line, value in zip(rows, significance)
                  if value >= float(sigma)},
        "scale_deg": float(robust),
        "quiet_lines": (int(quiet_lines[0]), int(quiet_lines[1])),
        "peak_line": (int(rows[int(np.argmax(significance))])
                      if rows.size else None),
        "peak_sigma": (float(significance.max()) if rows.size else 0.0),
    }


def chroma_departure(envelope: np.ndarray, active: Tuple[int, int],
                     line: int, quiet_lines: Tuple[int, int],
                     period: int = colour_lock.OFFSET_PERIOD_LINES) -> complex:
    """One line's chroma phase against the quiet lines of the same residue.

    The reference is the mean over quiet lines whose index shares this line's
    residue modulo the four-line period, which is the period the colour-under
    rotation and the four-times-subcarrier sampling frame close on together.
    Taking the reference from the same residue is what removes the rotation
    from the comparison; taking it from adjacent lines would not.
    """
    first, last = int(active[0]), int(active[1])
    rows = np.arange(int(quiet_lines[0]), int(quiet_lines[1]))
    rows = rows[(rows % period) == (int(line) % period)]
    if rows.size == 0:
        raise ValueError("the quiet span holds no line of this residue")
    reference = np.asarray(envelope)[rows][:, first:last].mean(axis=0)
    return complex((np.asarray(envelope)[int(line), first:last]
                    * np.conj(reference)).sum())


def accumulate(table: Optional[Dict[int, complex]],
               envelope: np.ndarray, active: Tuple[int, int],
               lines: Iterable[int],
               quiet_lines: Tuple[int, int],
               period: int = colour_lock.OFFSET_PERIOD_LINES
               ) -> Dict[int, complex]:
    """Fold one field's departures into a running table, as unit phasors.

    UNIT phasors, not the raw products: the head switch is a phase event and
    the amplitude of a line's chroma is the picture's saturation, so summing
    the products would let a saturated field outvote a pale one on a quantity
    neither of them disagrees about.
    """
    running = dict(table or {})
    for line in lines:
        value = chroma_departure(envelope, active, int(line), quiet_lines,
                                 period)
        if value == 0:
            continue
        running[int(line)] = running.get(int(line), 0j) + value / abs(value)
    return running


def phases(table: Dict[int, complex]) -> Dict[int, float]:
    """The accumulated table as one phase a line, in radians."""
    return {int(line): float(np.angle(value))
            for line, value in table.items() if value != 0}


def confidence(table: Dict[int, complex], fields: int) -> Dict[int, float]:
    """How consistent each line's departure was across the fields pooled.

    The mean resultant length of the unit phasors: one where every field
    agreed, near zero where the fields disagreed.

    It is a CONSISTENCY check and not a detector, and the difference matters.
    On the still captures every line reads 0.91 to 1.00, the quiet lines
    included, because a departure of zero degrees is as repeatable as a
    departure of thirty. What separates the head switch from the picture is the
    SIZE of the phase; what this answers is whether that size was the same
    field after field. Use it to refuse a table built on fields that disagreed,
    not to decide which lines to correct - `detect` decides that.
    """
    if fields <= 0:
        return {}
    return {int(line): float(abs(value) / fields)
            for line, value in table.items()}


def correct_field(chroma_field: np.ndarray, table: Dict[int, float],
                  gain: float = DEFAULT_GAIN,
                  centre: float = 32768.0) -> np.ndarray:
    """Apply the table to one time base corrected chroma field.

    Returns a new array; the input is not modified. Lines absent from the
    table are copied through untouched, which is the whole of the correction's
    reach - it is four or five lines at the foot of the field and nothing else.
    """
    out = np.array(chroma_field, dtype=np.float64)
    for line, phase in table.items():
        index = int(line)
        if not 0 <= index < out.shape[0]:
            continue
        out[index] = rotate_line(out[index], float(gain) * float(phase),
                                 centre=centre)
    return out


def split_point(product: np.ndarray) -> Dict[str, object]:
    """The best single change point in a line's residual, and its two phases.

    OFFERED, NOT DEFAULT. The location is stable - pooled over the first fields
    of `dod_fix_75bars_SP` it lands at sample 226 of an active area beginning
    at 134, and at 232 on `dod_fix_chromanoise_SP` - but the phases either side
    come from the luma-chroma difference, which the module docstring shows is
    not the chroma's own error. Use it to say WHERE the line breaks, not by how
    much.

    The score is the sum of the two segments' coherent magnitudes; `gain` is
    that score against the single-segment magnitude, so a line with no change
    point scores one and a clean split scores well above it.
    """
    values = np.asarray(product)
    if values.size < 3:
        raise ValueError("a change point needs at least three samples")
    running = np.cumsum(values)
    total = running[-1]
    left, right = running, total - running
    score = np.abs(left) + np.abs(right)
    index = int(np.argmax(score[:-1])) + 1
    whole = abs(total)
    return {
        "sample": index,
        "before_deg": float(np.degrees(np.angle(left[index - 1]))),
        "after_deg": float(np.degrees(np.angle(right[index - 1]))),
        "score": float(score[index - 1]),
        "gain": float(score[index - 1] / whole) if whole > 0 else float("inf"),
    }


RUNTIME_CONTRACT = """
What the decoder would have to call, and where.

`vhsdecode/chroma.py` is another lane's file and is not edited here; this is
the call it would take.

  1. In `process_chroma`, after `upconvert_chroma_phase_comp` and after the
     final chroma band-pass - the correction is a phase rotation of the
     up-converted chroma and has to see the signal the picture will get:

         lock = colour_lock.measure(
             field.dspicture.reshape(linesout, outwidth),
             uphet.reshape(linesout, outwidth),
             field.rf.SysParams["outfreq"] * 1e6, active_samples,
             quiet_lines, field.rf.DecoderParams["ire0"])

     `field.dspicture` is the time base corrected luma picture, set at
     `vhsdecode/field.py:1417` and already read in place by
     `vhsdecode/luma_beat.py:284`, so the buffer this needs is present at the
     site and no new plumbing is required. `active_samples` is the decode's
     `activeVideoStart`/`activeVideoEnd`.

  2. `found = chroma_head_switch.detect(lock["residual_deg"], quiet_lines)`
     and, on the first field of a decode, `quiet_lines` from
     `chroma_head_switch.quiet_span(...)` so the reference excludes what was
     found.

  3. Accumulate across fields of the same parity, in a store on `field.rf`
     keyed by `field.isFirstField` - which rule 38 of `RINGING_RULES.md` fixes
     as the head identity:

         store[parity] = chroma_head_switch.accumulate(
             store.get(parity), lock["chroma_envelope"], active_samples,
             found["lines"], quiet_lines)

  4. Apply, once the store holds more than one field:

         uphet = chroma_head_switch.correct_field(
             uphet.reshape(linesout, outwidth),
             chroma_head_switch.phases(store[parity])).reshape(-1)

  5. Gate it as a named stage in `vhsdecode/pipeline/stages.toml` under the
     colour block, so `--stages -chroma_head_switch` turns it off by name, as
     Ethan's on-off rule requires.

The first field of a decode has no table and is left alone. That is the
correct behaviour and not a limitation: the quantity is a transport constant
and one field cannot separate it from that field's picture.
"""
