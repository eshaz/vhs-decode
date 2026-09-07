"""One field at a time in the ringing path: the directive, and what it cost.

Ethan, 2026-08: *"Remove the averaging in the ringing path and work ONE FIELD
AT A TIME."*

THIS MODULE'S SUBJECT IS DISCHARGED, AND THAT IS RECORDED HERE RATHER THAN
LEFT FOR A READER TO DISCOVER. What it originally described was a change still
to be made inside `vhsdecode/addons/ringing_cancellation.py` - a
`SLOW_HORIZON_FACTOR = 8` that multiplied the field-tracking horizon at five
accumulation sites. That file no longer exists; the ringing engine is
`vhsdecode/models/ringing_tesseract.py`, whose own third standing law reads
"ONE FIELD AT A TIME. `average_fields` is accepted and NOT used to blend
anything across fields", and which deliberately supersedes rule 4 of
`vhsdecode/addons/RINGING_RULES.md` on the directive quoted above.

MEASURED, 2026-09-07, AND THE HORIZON IS NOW PROVABLY INERT. Three decodes of
`zaroff-pulseandbar-NTSC-SP-SLV-778HF-50msps-rf-pb`, eight frames each, with
`-n -f 50 -l 8 -t 4 --dctp --lti_gain 0 --ire0_adjust --chroma_env_gain 0.5`
and nothing else changed, gauged by `edge_ripple` below on the PICTURE and not
on the sync region - the ripple following the bar's own rising edge, picture
lines 60 to 230, sixteen fields, a window from twelve to a hundred and twenty
samples past the edge with the level and any slow tilt taken out:

    decode                                  coherent ripple    total ripple
    --inverse_eq -1   correction off            0.1654 IRE      0.7171 IRE
    --inverse_eq  4   four-field horizon        0.2249 IRE      0.7419 IRE
    --inverse_eq  0   one field at a time       0.2249 IRE      0.7419 IRE

The last two decodes are BYTE-IDENTICAL - md5 aa7c901037d48e31b062e73bbb0353ec
on both luma files - so the horizon argument reaches nothing at all and the
engine is one field at a time whatever it is given. The directive is
implemented and there is no remaining change for this module to name.

WHAT SURVIVES, AND IT IS THE PART WORTH KEEPING. The warning the original
measurement carried is still true on the new engine: the correction raises the
coherent ripple from 0.1654 to 0.2249 IRE, which is 36 per cent WORSE than
applying no correction at all on this gauge. The first measurement of this,
taken on the old engine, read 0.1357 off against 0.2605 at a one-field
horizon - a different engine, different flags and different absolute numbers,
and the same sign and the same order. That is what an estimator does when its
own variance is no longer thinned by averaging: the correction stops being a
correction and becomes an injection of its own estimation noise, which is the
mechanism rule 4 was written against.

The arc already owns the two candidate answers, and neither is a running mean
over eight fields:

  * `residual_limit`'s gain law, `a* = 1 / (1 + rho)`, with `rho` the ratio of
    the estimate's own error variance to the signal it is correcting. A
    single-field estimate has a larger `rho`, so the same measurement applied
    at the gain its variance earns would shrink rather than amplify.

  * The model-based filtering Ethan named: the steady-state Kalman gain from
    two measured variances. A one-field measurement blended with the model at
    that gain changes field to field, which is what the directive asks, while
    still being defended against its own noise.

THE FREQUENCY AXIS WAS MISSING AND IS NOW MEASURED - AND IT REPORTS A LIMIT
RATHER THAN A RESONANCE. `edge_ripple` now returns where the coherent ripple
sits and what share of its power is there. Measured on the same three decodes,
the peak lands at 0.133 MHz on all of them with a share of 0.301 with the
correction off and 0.367 with it on. That 0.133 MHz is the FIRST BIN of a
108-sample window at four times the subcarrier, which is 7.5 microseconds and
by the 1/T probe law resolves nothing below 133 kHz - so the reading is the
window's own floor and says the coherent ripple is SLOWER than this gauge can
resolve, not that it sits at 133 kHz. What the pair of shares does say is that
the correction concentrates the coherent power further into that slowest bin
rather than removing it, which is consistent with the amplitude figures above.
Resolving the frequency needs a longer window than the pulse-and-bar pattern
leaves between features.

WHY THIS IS AN INSTRUMENT AND NOT A STAGE, which is the other question this
file has to answer. `edge_ripple` reads the ACTIVE PICTURE. The standing law
of this arc is that a correction derives from the sync pulse alone and that
active-area gauges are offline validation only, so this can never be wired
into a decode as a stage without breaking the rule it exists to check. It is
named in `tools/ringing_measure/stage_inventory.py`'s `INSTRUMENTS` table with
that reason.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# Where the gauge window starts and ends after the edge, in samples at four
# times subcarrier. The near edge clears the decoder's own video low-pass
# transient - `compute_video_filters` builds an order-9 supergaussian whose
# ring is over inside about ten samples - and the far edge stops before the
# next feature of the pulse-and-bar pattern. Both are stated as arguments; the
# defaults are the window the figures in the module docstring were taken in.
WINDOW_START = 12
WINDOW_END = 120

# The rate the gauge's samples arrive at, which the FREQUENCY axis needs and
# the two figures above do not. A decoded luma field is written at four times
# the colour subcarrier - `colour_under`'s own, derived from the line rate the
# standard fixes, not a second copy - and a caller reading a 625-line decode
# passes its own rate.
SUBCARRIER_MULTIPLE = 4.0


def edge_sample(picture: np.ndarray, active: Tuple[int, int]) -> int:
    """Where the bar's rising edge is, from the field's own mean line.

    The steepest rise of the line-averaged picture inside the active area. No
    threshold and no template: the pattern supplies the edge.
    """
    profile = np.asarray(picture, dtype=np.float64)
    if profile.ndim == 3:
        profile = profile.reshape(-1, profile.shape[-1])
    mean_line = profile.mean(axis=0)
    first, last = int(active[0]), int(active[1])
    return first + int(np.argmax(np.diff(mean_line[first:last])))


def output_rate_hz(system: str = "NTSC") -> float:
    """The decoded field's sample rate, from the specification's subcarrier.

    Read through `colour_under`, which derives the subcarrier from the line
    rate the standard fixes, rather than restated here - so the frequency
    this module reports and the one every other module in the arc reports sit
    on the same number.
    """
    from vhsdecode.models import colour_under

    return SUBCARRIER_MULTIPLE * float(colour_under.subcarrier_hz(system))


def edge_ripple(picture: np.ndarray, active: Tuple[int, int],
                black_level: float, ire_per_unit: float,
                edge: Optional[int] = None,
                window: Tuple[int, int] = (WINDOW_START, WINDOW_END),
                sample_rate_hz: Optional[float] = None
                ) -> Dict[str, float]:
    """THE GAUGE: ripple after a picture edge, on all three axes.

    AMPLITUDE is two numbers. `coherent` is the ripple of the LINE-AVERAGED
    trace, which is what a fixed channel resonance produces and what the
    correction is aimed at; `total` includes each line's own noise. Both have
    the window's mean level and its slow tilt removed, so neither can be moved
    by a level shift or a droop - only by ripple.

    TIME is the window itself: where the edge is and how far past it the
    ripple is read, which is what makes this a transient gauge rather than a
    spectrum of the whole line.

    FREQUENCY is where the coherent ripple actually sits, and it was missing.
    A ringing gauge that reports only a size cannot tell a channel resonance
    from a de-emphasis error, and the two want different corrections. The
    largest component of the tilt-removed mean trace is reported with its
    share of the coherent power, so a ripple with one clear frequency and one
    with none are told apart. The rate defaults to the decoded field's own,
    four times the specified subcarrier; a caller on another system passes
    its own rather than having one assumed for it.
    """
    profile = np.asarray(picture, dtype=np.float64)
    if profile.ndim == 3:
        profile = profile.reshape(-1, profile.shape[-1])
    profile = (profile - float(black_level)) / float(ire_per_unit)
    if edge is None:
        edge = edge_sample(profile, active)
    span = profile[:, edge + int(window[0]):edge + int(window[1])]
    if span.shape[1] < 4:
        raise ValueError("the ripple window is too short to fit a tilt")
    span = span - span.mean(axis=1, keepdims=True)
    mean_trace = span.mean(axis=0)
    index = np.arange(span.shape[1], dtype=np.float64)
    tilt = np.polyval(np.polyfit(index, mean_trace, 1), index)
    ripple = mean_trace - tilt
    rate = float(output_rate_hz() if sample_rate_hz is None
                 else sample_rate_hz)
    spectrum = np.abs(np.fft.rfft(ripple)) ** 2
    # the zero bin is the mean, which has already been removed, and reporting
    # it as a frequency would name direct current as the ripple
    spectrum[0] = 0.0
    total_power = float(spectrum.sum())
    peak = int(np.argmax(spectrum))
    return {
        "edge": int(edge),
        "coherent_ire": float(np.sqrt(np.mean(ripple ** 2))),
        "total_ire": float(np.sqrt(np.mean(span ** 2))),
        "lines": int(span.shape[0]),
        "ripple_hz": float(np.fft.rfftfreq(ripple.size, d=1.0 / rate)[peak]),
        "ripple_share": (float(spectrum[peak] / total_power)
                         if total_power > 0 else float("nan")),
        "sample_rate_hz": rate,
    }


MEASURED = {
    "capture": "zaroff-pulseandbar-NTSC-SP-SLV-778HF-50msps-rf-pb, 8 frames",
    "gauge": ("edge_ripple on the bar's rising edge, picture lines 60-230, "
              "16 fields, window 12-120 samples past the edge"),
    # THE LIVE ENGINE'S FIGURES ARE THE TOP-LEVEL ONES, because a reader
    # taking the headline numbers should be taking the ones that describe
    # what a decode does today. 2026-09-07, on
    # `vhsdecode/models/ringing_tesseract.py`, with `-n -f 50 -l 8 -t 4
    # --dctp --lti_gain 0 --ire0_adjust --chroma_env_gain 0.5`.
    "date": "2026-09-07",
    "engine": "vhsdecode/models/ringing_tesseract.py",
    "flags": ("-n -f 50 -l 8 -t 4 --dctp --lti_gain 0 --ire0_adjust "
              "--chroma_env_gain 0.5"),
    "correction_off": {"flag": "--inverse_eq -1",
                       "coherent_ire": 0.1654, "total_ire": 0.7171},
    "four_field": {"flag": "--inverse_eq 4",
                   "coherent_ire": 0.2249, "total_ire": 0.7419},
    "one_field": {"flag": "--inverse_eq 0",
                  "coherent_ire": 0.2249, "total_ire": 0.7419},
    "horizon_is_inert": True,
    "identical_md5": "aa7c901037d48e31b062e73bbb0353ec",
    # THE FREQUENCY AXIS, and it is a bound rather than a measurement: the
    # peak lands on the window's own first bin on every decode, so the ripple
    # is slower than 7.5 microseconds of window can resolve.
    "ripple_hz": 133000.0,
    "ripple_hz_is_the_window_floor": True,
    "ripple_share_correction_off": 0.301,
    "ripple_share_correction_on": 0.367,
    "on_the_retired_engine": {
        "engine": "vhsdecode/addons/ringing_cancellation.py, since deleted",
        "correction_off": {"flag": "--inverse_eq -1",
                           "coherent_ire": 0.1357, "total_ire": 0.6414},
        "four_field": {"flag": "--inverse_eq 4",
                       "coherent_ire": 0.1339, "total_ire": 0.6435},
        "one_field": {"flag": "--inverse_eq 0",
                      "coherent_ire": 0.2605, "total_ire": 0.8117},
    },
}
"""Both A/Bs, as data, and the two engines are kept apart rather than pooled.

The finding is that `one_field` is worse than `correction_off` on both numbers
and on BOTH engines. What changed with the engine is the four-field row: on
the retired one a four-field horizon was marginally better than no correction
at all, and on the current one it is the SAME DECODE as the one-field mode -
byte-identical output - because the horizon argument no longer reaches
anything."""


RUNTIME_CONTRACT = """
DISCHARGED. What the decoder had to change, and what became of each part.

  1. THE FLAG. `--inverse_eq 0` was already documented in `vhsdecode/main.py`
     as "0 disables averaging", and on the current engine the argument is
     accepted and ignored: `--inverse_eq 0` and `--inverse_eq 4` produce
     byte-identical output, so the engine is one field at a time whatever it
     is given. Nothing to do.

  2. THE STRUCTURAL PIECE IS GONE WITH ITS FILE. `SLOW_HORIZON_FACTOR = 8`
     lived in `vhsdecode/addons/ringing_cancellation.py`, which has been
     deleted; `vhsdecode/models/ringing_tesseract.py` replaced it and states
     one field at a time as its third standing law. There is no factor left
     to set.

  3. AND THE WARNING STANDS. On the current engine the correction reads
     0.2249 IRE coherent against 0.1654 with it off, so it is 36 per cent
     worse than doing nothing on this gauge. The acceptance test is
     unchanged - `edge_ripple` on the capture in `MEASURED`, reaching the
     correction-off figure or below - and it is a test of whichever of the
     two gains in the module docstring the ringing lane chooses, not of the
     horizon, which no longer exists.
"""
