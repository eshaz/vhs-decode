"""A colour-free luma: the up-heterodyned residual chroma, locked and removed.

Ethan, this session, two sentences that are one design:

    *"Subtract the up-heterodyned residual chroma from the luma, with the
    active area masked IN and the colour-under phase locked, to leave a
    colour-free luma."*

    *"Feed that luma connection BACK into the chroma stage for alignment on all
    dimensions."*

WHAT IS ALREADY THERE AND WHAT IS NOT. `vhsdecode/luma_beat.py` ships the
first half in a different form: it takes the colour-under as the time base
correction leaves it - `field.chroma_under_tbc`, captured BEFORE the
up-conversion rewrites its phase - fits two numbers for the whole decode, and
subtracts. Three things in the directive are not in that: the estimate is
taken from the DOWN-converted chroma rather than the up-heterodyned one, the
transfer is frozen for the decode rather than locked per field, and the fit
runs over whole lines rather than over the active area alone. This module is
the directive's form, and the differences are not cosmetic - the phase lock is
what lets the estimate be taken from the channel the picture is actually
made of.

THE LOCK IS WHAT MAKES THE SUBTRACTION POSSIBLE. The decoded chroma and the
luma's colour-under replica are the same physical vector in two frames, and
`colour_lock` measures the fixed offset between them: a four-line-periodic
table whose mean resultant length is 0.98 to 0.99 on the SP decodes. Turning
the decoded chroma back through that table produces a prediction of the luma's
colour-under content that is aligned sample for sample, which is the whole of
"the colour-under phase locked".

THE ACTIVE AREA IS MASKED IN. Sync and blanking carry no colour-under
modulation, so including them adds two noisy factors to a product and nothing
else; the burst is excluded with them because its phase is the decoder's
reference and not the tape's picture. Masked in, the fit has 760 samples a
line on a 910-sample NTSC field.

MEASURED, WITH THE CONTROL THE CLAIM NEEDS. `zaroff-multiburst-NTSC-SP` was
decoded here from the playback-tap capture together with its `y-only` twin -
the same tape, deck and pattern with nothing recorded in the chroma band -
eight frames each. One complex transfer a field, fitted on even picture lines
and measured on odd ones, so the figure is held out:

    capture                transfer |h|   removed   lock resultant
    multiburst SP              1.3238      9.94 %       0.981
    multiburst SP, y-only      0.0073      0.07 %       0.135

The transfer differs by a factor of 181 between a tape that carries colour and
one that does not, and the estimator declines on the control by itself, with no
gate and no burst test - which is the property `luma_beat` has to obtain from
the colour killer instead. On the shipped decodes, same held-out split:

    75 % bars SP        5.38 %       chroma noise SP      20.00 %
    75 % bars EP        4.84 %

THE SHARE IS SMALL BECAUSE THE BAND IS MOSTLY LUMA, and the number has to be
read that way. The luma carries 0.45 to 1.13 IRE at the colour-under carrier,
and the y-only capture carries 1.135 IRE against its twin's 1.131 - the same
to four parts in a thousand. Nearly all of that amplitude is the picture's own
content at 629 kHz; what the chroma can predict is only the coupled part. The
share also falls away from the carrier, which is the coupling telling us where
it lives:

    analysis half-width      30 kHz    150 kHz    500 kHz
    multiburst SP             9.94 %     4.38 %     0.05 %
    75 % bars SP              5.38 %     2.28 %     0.22 %
    chroma noise SP          20.00 %    13.41 %     1.35 %
    y-only control            0.07 %     0.06 %     0.13 %

so the default half-width is the narrowest of the three and the argument is
exposed. A wider window does not remove more; it dilutes what there is.

DIRECTIVE 5, AND WHAT IT HONESTLY DELIVERS. The connection carries back to the
chroma stage two dimensions that stand up to a control and one that does not:

    PHASE      the fixed offset table and the per-line departure from it -
               `colour_lock.lock_offset` and `colour_lock.residual`. Resultant
               0.98-0.99 on SP and 0.135 on the y-only control.

    AMPLITUDE  the fitted transfer, one complex number a field. 1.3238 against
               0.0073 on the matched pair.

    RATE       REFUSED. The phase advance along a line - the lag-one product of
               `chroma_envelope` against `luma_colour_under`, summed over the
               picture lines - would be a frequency offset between the two
               channels. It reads 18.0 kHz on 75 % bars, 1.0 kHz on chroma
               noise and 33.1 kHz on multiburst, and 62.7 kHz on the y-only
               capture, which has no colour-under at all. It is measuring the
               chroma envelope's own modulation and the control says so, so it
               is reported and not used.

Two dimensions of three, with the third named and refused, is what "alignment
on all dimensions" comes to on this evidence. `feedback` returns exactly that
and marks the refused axis in the result rather than omitting it.

AND IT IS A PER-FIELD REFERENCE, NOT A PER-LINE CORRECTION. This was measured
rather than assumed, because the tempting next step is to rotate each line of
the chroma by that line's own residual. Done on the shipped 75 % bars decode
and judged on the chroma's own vertical consistency over the quiet picture
lines - each line against the mean of the quiet lines sharing its four-line
residue - it is a large regression at every gain:

    as decoded                        0.775 degrees
    per-line residual, gain 0.25      1.740
    per-line residual, gain 0.50      3.288
    per-line residual, gain 1.00      6.546

The arithmetic behind that is the whole reason for the caution. The chroma's
own per-line phase error on those lines is 0.775 degrees; the residual's root
mean square is 7 to 11. Ten parts in eleven of the residual is the luma
replica's noise, and rotating the chroma by it injects that noise into the
picture. Averaged over a field - or over fields of one parity, which is what
`chroma_head_switch` does - the noise thins as the root of the count and the
reference is sound. Used line by line it is not.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from vhsdecode.models import colour_lock

DEFAULT_HALF_WIDTH_HZ = colour_lock.DEFAULT_HALF_WIDTH_HZ


def prediction(chroma_envelope: np.ndarray,
               offset: Dict[str, object]) -> np.ndarray:
    """The luma's colour-under content as the decoded chroma predicts it.

    The up-heterodyned chroma turned back through the fixed offset, so the
    result is in the colour-under's own frame and aligned with the luma's
    replica sample for sample. This is "the up-heterodyned residual chroma"
    with "the colour-under phase locked", and it is one multiplication
    because the lock is a phase table and nothing more.
    """
    envelope = np.asarray(chroma_envelope)
    table = np.asarray(offset["table"])
    period = int(offset["period"])
    residues = np.arange(envelope.shape[0]) % period
    return envelope * np.conj(table[residues])[:, None]


def transfer(predicted: np.ndarray, measured: np.ndarray,
             active: Tuple[int, int], lines: Tuple[int, int],
             stride: int = 1, offset_line: int = 0) -> complex:
    """One complex transfer a field, least squares, active area masked in.

    `stride` and `offset_line` exist so a caller can fit on one half of the
    picture lines and measure on the other - the held-out split every figure in
    the module docstring is taken under. A transfer fitted and judged on the
    same samples would fall whatever the coupling was.
    """
    first, last = int(active[0]), int(active[1])
    rows = np.arange(int(lines[0]) + int(offset_line), int(lines[1]),
                     int(stride))
    model = np.asarray(predicted)[rows][:, first:last].ravel()
    data = np.asarray(measured)[rows][:, first:last].ravel()
    energy = np.vdot(model, model)
    if energy == 0:
        return 0j
    return complex(np.vdot(model, data) / energy)


def removed_share(predicted: np.ndarray, measured: np.ndarray,
                  gain: complex, active: Tuple[int, int],
                  lines: Tuple[int, int], stride: int = 1,
                  offset_line: int = 0) -> Dict[str, float]:
    """How much of the band the subtraction takes out, over a given set.

    Power before and after, and their ratio. Call it with the lines the
    transfer was NOT fitted on and the answer is held out.
    """
    first, last = int(active[0]), int(active[1])
    rows = np.arange(int(lines[0]) + int(offset_line), int(lines[1]),
                     int(stride))
    model = np.asarray(predicted)[rows][:, first:last]
    data = np.asarray(measured)[rows][:, first:last]
    before = float(np.mean(np.abs(data) ** 2))
    after = float(np.mean(np.abs(data - gain * model) ** 2))
    return {
        "before": before,
        "after": after,
        "removed": (1.0 - after / before) if before > 0 else 0.0,
        "lines": int(rows.size),
    }


def colour_free(luma_field: np.ndarray, predicted: np.ndarray,
                gain: complex, active: Tuple[int, int],
                sample_rate_hz: float,
                carrier_hz: Optional[float] = None,
                system: str = "NTSC") -> np.ndarray:
    """THE COLOUR-FREE LUMA: the locked estimate re-modulated and subtracted.

    The estimate is carried back up to the colour-under carrier and taken off
    the real luma. THE SUBTRACTION IS CONFINED TO THE ACTIVE AREA - masked in,
    exactly as directed - so no correction can reach sync, blanking or the
    burst, and every level the rest of the decoder measures on those is
    untouched by construction.

    Returns a new array.
    """
    field = np.array(luma_field, dtype=np.float64)
    first, last = int(active[0]), int(active[1])
    if carrier_hz is None:
        from vhsdecode.models import colour_under
        carrier_hz = colour_under.carrier_hz(system)
    index = np.arange(field.shape[1], dtype=np.float64)
    # The same local oscillator `colour_lock.luma_colour_under` demodulated
    # with, conjugated - so the estimate returns to exactly the frame it was
    # taken from and no phase convention has to be agreed twice.
    local = np.exp(2j * np.pi * float(carrier_hz) * index / sample_rate_hz)
    estimate = np.real(gain * np.asarray(predicted) * local[None, :])
    # The demodulation took a one-sided envelope, so the real waveform it
    # represents is twice its real part.
    field[:, first:last] -= 2.0 * estimate[:, first:last]
    return field


def feedback(lock: Dict[str, object], gain: complex,
             share: Dict[str, float]) -> Dict[str, object]:
    """DIRECTIVE 5: what the luma connection hands back to the chroma stage.

    Phase and amplitude, each with the number that says whether to believe it,
    and the rate axis named and refused. A consumer should read `dimensions`
    and act only on the entries marked usable, which is what stops a refused
    axis from being used by a caller that did not read the docstring.
    """
    offset = lock["offset"]
    return {
        "dimensions": {
            "phase": {
                "usable": True,
                "offset_table": np.asarray(offset["table"]),
                "offset_deg": np.asarray(offset["degrees"]),
                "residual_deg": np.asarray(lock["residual_deg"]),
                "confidence": float(offset["resultant"]),
                "what": ("the fixed colour-under-to-subcarrier offset and the "
                         "per-line departure from it"),
            },
            "amplitude": {
                "usable": True,
                "transfer": complex(gain),
                "magnitude": float(abs(gain)),
                "degrees": float(np.degrees(np.angle(gain))),
                "removed": float(share["removed"]),
                "what": ("the luma-side replica's amplitude against the "
                         "decoded chroma, one complex number a field"),
            },
            "rate": {
                "usable": False,
                "why": ("the phase advance along a line reads 62.7 kHz on a "
                        "y-only capture that carries no colour-under, so it "
                        "measures the chroma envelope's own modulation"),
            },
        },
        "lock_resultant": float(offset["resultant"]),
    }


def process(luma_field: np.ndarray, chroma_field: np.ndarray,
            sample_rate_hz: float, active: Tuple[int, int],
            lines: Tuple[int, int], black_level: float = 0.0,
            system: str = "NTSC",
            half_width_hz: float = DEFAULT_HALF_WIDTH_HZ,
            held_out: bool = True) -> Dict[str, object]:
    """One field in; the colour-free luma, the transfer, and the feedback out.

    With `held_out` the transfer is fitted on even picture lines and its share
    measured on odd ones, which is how every figure in the module docstring was
    produced. The subtraction itself always uses every line - the split is a
    property of the MEASUREMENT, not of the correction.
    """
    lock = colour_lock.measure(luma_field, chroma_field, sample_rate_hz,
                               active, lines, black_level=black_level,
                               system=system, half_width_hz=half_width_hz)
    predicted = prediction(lock["chroma_envelope"], lock["offset"])
    measured = lock["luma_colour_under"]
    if held_out:
        gain = transfer(predicted, measured, active, lines, stride=2)
        share = removed_share(predicted, measured, gain, active, lines,
                              stride=2, offset_line=1)
    else:
        gain = transfer(predicted, measured, active, lines)
        share = removed_share(predicted, measured, gain, active, lines)
    return {
        "luma": colour_free(luma_field, predicted, gain, active,
                            sample_rate_hz, system=system),
        "transfer": gain,
        "share": share,
        "lock": lock,
        "feedback": feedback(lock, gain, share),
    }


RUNTIME_CONTRACT = """
What the decoder would have to call, and where.

`vhsdecode/chroma.py`, `vhsdecode/field.py` and `vhsdecode/luma_beat.py`
belong to other lanes and are not edited here.

  1. The site is where `luma_beat.correct` already stands: inside
     `chroma.decode_chroma`, after `process_chroma` has produced `uphet`,
     because that is the only place the decoded chroma and the time base
     corrected luma picture exist at once. `luma_beat` reaches
     `field.dspicture` in place there, so the buffer is available.

  2. Replace the `chroma_under_tbc` estimate with this one:

         result = colour_free_luma.process(
             field.dspicture.reshape(linesout, outwidth),
             uphet.reshape(linesout, outwidth),
             outfreq, (active_start_px, active_end_px),
             picture_lines, black_level)
         field.dspicture = result["luma"].reshape(-1)

     `picture_lines` is the active picture line span, and it must exclude the
     head-switch lines that `chroma_head_switch.detect` flags, for the reason
     `colour_lock` gives: the lock is fitted there.

  3. Hand `result["feedback"]` back to the chroma stage. Today nothing consumes
     it; the two entries marked usable are a per-field chroma phase reference
     and a per-field chroma amplitude reference, and they are the "alignment on
     all dimensions" the directive asks the connection to carry.

  4. Gate it as a named stage in `vhsdecode/pipeline/stages.toml`, ordered
     after `decode_chroma` and before `cti`, with `luma_beat` as the stage it
     supersedes rather than runs beside - both subtract the same coupling and
     running both would subtract it twice.
"""
