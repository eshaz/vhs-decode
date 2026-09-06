"""The injection test: a known component, at a known amplitude, never fitted.

Ethan: *"Injection test: known signal, known amplitude, never fitted. Sweep
level, find recovery threshold. Gives a sensitivity curve and identifies
which stage eats what."*

WHAT THIS IS. Every estimator in the chain reports what it finds in a
residual. None of them can say, on its own, how small a component it would
still have found, or which OTHER estimator would have claimed the component
first. The only way to learn that is to plant a component whose kind and
size are known, run the chain unchanged, and read back what each stage
reports. The planted component is never fitted: the estimators are not told
it is there, nothing here tunes them to it, and the only quantity ever
calibrated is the SIZE of the injection in the measurement's own noise
units, which is a property of the injection and not of any estimator.

WHAT IS PLANTED. Six kinds, each with one known amplitude and, where the kind
needs them, known parameters:

    echo                a reflection at a known delay
    ring                a damped resonance at a known frequency and decay
                        (the arc measured 2.290 and 2.213 MHz, 0.46 and
                        0.54 us, on the home tape's back-porch remainder -
                        `sync_geometry.relaxation_or_ringing`; the default
                        is the centre of that pair)
    level step          a change of level: nepers on the grid, IRE in the
                        series
    clip                a limiter at a known depth relative to the sync tip;
                        memoryless, so it exists only in the time series
    frequency scaling   the frequency axis stretched by a known factor, which
                        is what a tape-speed error does to every signature at
                        once: a tape played fast by the fraction `a` puts every
                        recorded feature at `f (1 + a)` and compresses the
                        waveform in time by the same factor, and both
                        representations use that one sign convention
    phase rotation      a constant rotation of the complex transfer, or of
                        the analytic signal in time; phase only, no magnitude

WHERE IT IS PLANTED. Into the representation each judge consumes, and on a
REAL background rather than a synthetic one, so the injection sits on the
tape's own noise:

  * the GRID: the measured complex sync-step transfer with its per-bin SE,
    exactly as `residual_floor.load_export` reads it, with the two half-field
    splits so the held-out judgement is exercised too. An LTI modification is
    a multiplication of the transfer, an addition of `log K(f)` in the log
    domain where the chain adds;
  * the SERIES: horizontal sync pulses cut from luma demodulated here by the
    decoder's own chain (its band-pass, its discriminator, its de-emphasis
    and its low-pass, built from its parameters), aligned on the sync fall,
    with every window placed from the format's timing table and the pulse's
    own measured crossings, and never from the picture;
  * the CUBE: the tesseract's vertices (head x polarity x half x tape), with
    the component given a stated sign pattern over the axes, so it is known
    which contrast should carry it.

HOW RECOVERY IS READ. Each reader is an existing estimator called as it
stands. Its reading on the injected data less its reading on the same noise
realisation of the background is the reader's RESPONSE, and two error
quantities go with it, because they answer two different questions:

    noise     the reader's own scatter over realisations of the BACKGROUND -
              what one measurement of this size would show with nothing
              planted. A response is DETECTED when it clears twice this and
              stays clear at every larger amplitude; below that the reading
              is consistent with the null;
    error     the standard error of the paired response - how precisely the
              response itself is known, which for a linear reader is close to
              nothing because the noise cancels in the pair.

    detect    the smallest amplitude from which every larger one is detected
    recover   the smallest detected amplitude at which the response is ALSO
              within twice the larger of the two errors of the injected
              amplitude, defined only for a reader whose reading is in the
              injected quantity's own units - the reader reports the
              component at its true size. The band can be LEFT again at large
              amplitude when a reader returns a fixed fraction of the
              component, and `tracks_to` says where

and the slope of the response against the amplitude, well above threshold,
is the fraction of the component the reader returns. A reader that returns
half of it has a slope of one half and never recovers it, however large it
is made; that is the number that says a stage EATS part of a component.

WHICH STAGE EATS WHAT. On the grid the arc's judge, `residual_floor.test`,
fits every entry of the key with real coefficients on the stacked real and
imaginary parts, plus a level and a delay it does not charge to the residual.
Each entry carries a declared chain position (`interference.position_of`)
and `stage_boundaries.LINKS` maps a position to a link of the chain. The
fitted part belonging to each link is projected, in the fit's own weighted
inner product, onto the injected shape per unit amplitude; by the linearity
of least squares the link shares, the nuisance share and the residual share
SUM TO THE INJECTED AMPLITUDE EXACTLY, so the matrix of kinds against stages
is a partition and not a set of overlapping claims. What lands in the
residual is offered, unchanged, to every join through
`stage_boundaries.survey` and `search_families`, and to
`interference.outlier_response`, which say which join's prior would claim
it and which base-key direction would carry it. The whole thing is run with
the default key and with the key `interference.admission(granularity=
"entry")` admits on the band. In the series the stages are the modules the
readers belong to, at their declared positions; on the cube they are the
contrasts, and `tesseract.trace` names the stage each contrast is charged
to, `hypercomplex.causality` says whether what was planted reads as minimum
phase or as excess phase there, and `hypercomplex.deconvolve` whether the
real kernel that would be subtracted carries it.

UNITS. An amplitude is expressed relative to the measured noise and never as
a bare number. The unit is the rms PER-BIN SIGNIFICANCE of the injection:
the root mean square over bins (or samples) of the injected change divided
by that bin's own standard error. An injection at k SE therefore raises a
reduced chi-square by k squared when nothing absorbs it and has a
matched-filter signal-to-noise of k times the root of the bin count, which
is reported beside it. In the series the per-sample noise is that of the
group mean the readers consume, measured per sample across the lines, so a
component planted where the reserved interval is noisier - the colour-under
burst's tail, below - is measured against that noise and not against the
tip's. The clip has no linear regime, so its depth is found by bisection to
sit at the requested significance; every sweep also carries the natural
amplitude (reflection coefficient, IRE, nepers, fraction, radians) so
nothing is hidden behind the unit.

MEASURED. The numbers each function rests on are quoted in its own
docstring, with the source; the thresholds per kind per stage on the three
tapes' exports and on the 75-bar SP capture are in the tool's report
(`tools/ringing_measure/injection.py`) and asserted in
`tests/unit/test_injection.py`.
"""

import inspect
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import clipping
from vhsdecode.models import correction_export
from vhsdecode.models import hypercomplex
from vhsdecode.models import interference as inf
from vhsdecode.models import interference_distributions as dist
from vhsdecode.models import residual_floor
from vhsdecode.models import stage_boundaries as sb
from vhsdecode.models import standard_levels
from vhsdecode.models import sync_geometry as sg
from vhsdecode.models import tesseract

KINDS: Tuple[str, ...] = ("echo", "ring", "level step", "clip",
                          "frequency scaling", "phase rotation")
GRID_KINDS: Tuple[str, ...] = ("echo", "ring", "level step",
                               "frequency scaling", "phase rotation")
SERIES_KINDS: Tuple[str, ...] = KINDS
CUBE_KINDS: Tuple[str, ...] = GRID_KINDS

# The ring the arc measured on the remainder of the home tape's back porch:
# 2.290 and 2.213 MHz on the two heads, decays 0.46 and 0.54 us, amplitudes
# +0.717 and +0.708 IRE (`sync_geometry.relaxation_or_ringing`, its
# docstring). The default ring is the centre of that pair - a MEASURED
# default, not a constant of the format; a caller states its own where it
# knows better.
MEASURED_RING_HZ = 2.25e6
MEASURED_RING_DECAY_S = 0.5e-6

# A first-order expansion is used to calibrate a kind's change per unit of
# its natural amplitude (an echo of reflection coefficient a is a log-domain
# ripple of amplitude a only to first order in a). The expansion point is
# small enough that the second-order term, a^2 / 2, is five parts in ten
# thousand of the first - below anything the calibration is used for.
CALIBRATION_AMPLITUDE = 1e-3

# The detection gate: a response clears the reader's own noise by this
# factor. Two standard deviations is the conventional two-sided 95 per cent
# gate and it is a statistical convention, not a property of any tape.
DETECT_SIGMA = 2.0

# The label given to the two regressors `residual_floor.nuisance` fits
# beside the key: a level and a delay. They belong to the levels and the
# time base of the picture stage and are reported as their own column so a
# level step is seen to go where the judge sends it.
NUISANCE_STAGE = "nuisance (level, delay)"
RESIDUAL_STAGE = "residual"


# --------------------------------------------------------------------------
# The representations the chain consumes
# --------------------------------------------------------------------------


def significance(delta, sigma) -> Dict[str, float]:
    """The size of a change in the measurement's own noise units.

    `rms` is the root mean square of the per-bin change over the per-bin
    standard error - the unit every amplitude in this module is expressed in;
    `peak` its largest per-bin value; `snr` the matched-filter
    signal-to-noise, `rms` times the root of the count, which is what an
    ideal detector that knew the shape would see. `sigma` may be one number,
    one value per bin, or a full array the shape of `delta`.
    """
    change = np.asarray(delta)
    error = np.asarray(sigma, dtype=np.float64)
    if error.ndim == 0:
        error = np.full(change.shape, float(error))
    elif error.shape != change.shape:
        error = np.broadcast_to(error, change.shape)
    ratio = (np.abs(change) / np.maximum(error, 1e-300)).ravel()
    return {"rms": float(np.sqrt(np.mean(ratio ** 2))) if ratio.size else 0.0,
            "peak": float(ratio.max()) if ratio.size else 0.0,
            "snr": float(np.sqrt(np.sum(ratio ** 2))),
            "count": int(ratio.size)}


def grid_background(frequency_hz, transfer, se, label: str = "",
                    held_out: Optional[Dict[str, np.ndarray]] = None
                    ) -> Dict[str, object]:
    """A measured complex transfer on a frequency grid, with its per-bin SE,
    as a background for the grid injection.

    `se` is the standard error of the complex estimate with both parts
    together, as `residual_floor.log_domain` calibrates it on the pnb
    export; the per-bin noise of the log residual is then `se / |H|` (both
    parts together), which is the root of that function's `floor`.
    `held_out`, when given, carries `H_fit`, `H_judge` and `se_half` exactly
    as `residual_floor.load_export` returns them.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    values = np.asarray(transfer, dtype=np.complex128).ravel()
    error = np.asarray(se, dtype=np.float64).ravel()
    if not (grid.size == values.size == error.size):
        raise ValueError("the grid, the transfer and its SE must share a length")
    order = np.argsort(grid)
    grid, values, error = grid[order], values[order], error[order]
    domain = residual_floor.log_domain(values, error)
    out = {
        "representation": "grid",
        "label": label,
        "frequency_hz": grid,
        "H": values,
        "se": error,
        "log": domain["log"],
        "sigma": np.sqrt(domain["floor"]),
        "held_out": None,
    }
    if held_out is not None:
        out["held_out"] = {
            "H_fit": np.asarray(held_out["H_fit"], dtype=np.complex128)[order],
            "H_judge": np.asarray(held_out["H_judge"], dtype=np.complex128)[order],
            "se_half": np.asarray(held_out["se_half"], dtype=np.float64)[order],
        }
    return out


def load_grid(path: str, head: str = "a", band_hz=(0.2e6, 4.0e6),
              view: str = "fall") -> Dict[str, object]:
    """The arc's own measured sync-step transfer as a background, read by
    `residual_floor.load_export` unchanged so the injection sits on exactly
    the bins, band and half-splits the floor judge uses.

    On the three tapes' `tree_noeq` exports, fall view, 0.2 to 4 MHz: 944
    bins on every tape; the per-bin noise of the log residual has a median
    of 0.0036 (home A), 0.0054 (cd A) and 0.0106 (pnb A) and a 95th
    percentile of 0.023, 0.032 and 0.046, the largest values at the top of
    the band where the transfer is smallest (measured 2026-09-05).
    """
    measured = residual_floor.load_export(path, head, band_hz, view)
    tape = path.split("/")[-1].split("_")[0]
    return grid_background(measured["frequency_hz"], measured["H"],
                           measured["se"], f"{tape} head {head.upper()} {view}",
                           measured.get("held_out"))


def grid_keys(background: Dict[str, object], mechanics=None
              ) -> Dict[str, Dict[str, np.ndarray]]:
    """The default key and the entry-wise admitted key on the background's
    grid, from `residual_floor.keys_for`, built once and kept on the
    background: 14 and 55 entries on the 944-bin exports, the admission
    costing about seven seconds per grid (measured 2026-09-05)."""
    if "keys" not in background:
        keys = residual_floor.keys_for(background["frequency_hz"], mechanics)
        background["keys"] = {"default": keys["base"],
                              "admitted": keys["admitted"]}
    return background["keys"]


def demodulate_luma(rf_samples, sample_rate_hz: float,
                    sysparams: Dict[str, object], rfparams: Dict[str, object]
                    ) -> Tuple[np.ndarray, float]:
    """A MEASUREMENT INSTRUMENT built from the decoder's own chain, not a
    decoder: the luma FM read to IRE the way `process.py` reads it.

    Every stage is the decoder's, built by its own builder from its own
    parameters and applied in its order: the RF band-pass
    (`gen_bpf_supergauss` on `video_bpf_low/high/order`, the same call
    `_computevideofilters_b` makes for `Filters["RFVideo"]`), the analytic
    signal, the conjugate-product discriminator (`lddecode.utils.
    unwrap_hilbert`), then the main de-emphasis and the video low-pass
    (`gen_video_main_deemp_fft_params`, `gen_video_lpf_params`, the product
    the decoder keeps as `Filters["FVideo"]`) at the RF rate, then a
    polyphase resampling to the output rate `outfreq` and the map
    `(f - ire0) / hz_ire`. Nothing here is a correction and nothing reads
    the picture.

    THE DE-EMPHASIS IS NOT OPTIONAL. Read without it, the 75-bar capture's
    mean sync pulse carries the recording's pre-emphasised edges and a
    limiter placed above the tip clips the emphasised undershoot first -
    the recorder-input dark clip of the arc's clip-two-sites finding, not
    the tip clip an injection claims.

    WHAT THE INSTRUMENT PASSES, AND THE DECODER WITH IT. The band-pass
    `video_bpf_low` of 0.5 MHz is a supergauss of order 8 six megahertz
    wide, and at the NTSC VHS colour-under carrier of 629 kHz it still
    passes 0.71: the down-converted chroma reaches the discriminator, and
    wherever chroma is present the demodulated luma carries its beat with
    the carrier. On the 75-bar SP capture the beat spectrum peaks at 3.05
    and 2.82 MHz - the blanking carrier 3.69 MHz and the tip carrier 3.40
    MHz less 0.63 - and not at the 3.58 MHz burst, which is how it is known
    to be the colour-under and not a burst leaking through the record-side
    trap (measured 2026-09-05; the arc's `--luma_beat` correction is this
    coupling seen from the decoder). `series_background` measures where it
    sits in the line.
    """
    from fractions import Fraction

    from lddecode.utils import unwrap_hilbert
    from scipy.signal import resample_poly
    from vhsdecode.compute_video_filters import (
        gen_bpf_supergauss, gen_video_lpf_params,
        gen_video_main_deemp_fft_params)

    samples = np.asarray(rf_samples, dtype=np.float64).ravel()
    samples = samples - samples.mean()
    rate = float(sample_rate_hz)
    count = samples.size
    nyquist = rate / 2.0
    band = gen_bpf_supergauss(float(rfparams["video_bpf_low"]),
                              float(rfparams["video_bpf_high"]),
                              int(rfparams["video_bpf_order"]), nyquist, count)
    half_spectrum = np.fft.rfft(samples) * band
    # the analytic signal from the one-sided spectrum: the positive
    # frequencies doubled, the two real bins left alone, the negative
    # frequencies zero - the same construction as the decoder's hilbert mask
    full = np.zeros(count, dtype=np.complex128)
    full[:half_spectrum.size] = half_spectrum
    full[1:(count + 1) // 2] *= 2.0
    analytic = np.fft.ifft(full)
    instantaneous_hz = unwrap_hilbert(analytic, rate)
    deemphasis = gen_video_main_deemp_fft_params(rfparams, rate,
                                                 instantaneous_hz.size)
    low_pass = gen_video_lpf_params(rfparams, nyquist, instantaneous_hz.size)[1]
    video = np.fft.irfft(np.fft.rfft(instantaneous_hz) * deemphasis * low_pass,
                         n=instantaneous_hz.size)
    out_rate = float(sysparams["outfreq"]) * 1e6
    ratio = Fraction(out_rate / rate).limit_denominator(4096)
    luma = resample_poly(video, ratio.numerator, ratio.denominator)
    ire = (luma - float(sysparams["ire0"])) / float(sysparams["hz_ire"])
    return ire, rate * ratio.numerator / ratio.denominator


def sync_lines(ire, sample_rate_hz: float, sysparams: Dict[str, object]
               ) -> Tuple[np.ndarray, int]:
    """HORIZONTAL SYNC PULSES ONLY, cut from demodulated luma and aligned on
    the sync fall.

    A pulse is the interval below half the sync depth (`vsync_ire / 2`); it
    is kept when its width is within three transition times
    (`syncTransitionUS`) of the specified line sync (`hsyncPulseUS`), which
    rejects the equalizing pulses and the serrations by their widths alone,
    per the standing rule that the calibration reads horizontal sync and
    nothing from the vertical interval. Each kept line runs from one front
    porch before the fall (`frontPorchUS`) to the start of active video
    (`activeVideoUS[0]`), so the whole back porch is in the window and the
    windows placed on it end before the picture. The alignment is to the
    integer sample of the crossing; the sub-sample remainder is left as
    jitter, which the group spread then contains.

    On 0.05 s of the 75-bar SP capture from 0.2 s in: 760 lines of 156
    samples at the output rate, every one a horizontal sync by width, out
    of the 787 line periods the interval holds - the difference is the two
    vertical intervals (measured 2026-09-05).

    Returns the lines and the index of the fall inside each.
    """
    values = np.asarray(ire, dtype=np.float64).ravel()
    rate = float(sample_rate_hz)
    half = float(sysparams["vsync_ire"]) / 2.0
    width = float(sysparams["hsyncPulseUS"])
    tolerance = 3.0 * float(sysparams["syncTransitionUS"])
    before = int(round(float(sysparams["frontPorchUS"]) * rate / 1e6))
    after = int(round(float(sysparams["activeVideoUS"][0]) * rate / 1e6))
    below = (values < half).astype(np.int8)
    edges = np.diff(below)
    falls = np.where(edges == 1)[0] + 1
    rises = np.where(edges == -1)[0] + 1
    lines = []
    for fall in falls:
        following = rises[rises > fall]
        if following.size == 0:
            break
        pulse_us = (following[0] - fall) / rate * 1e6
        if abs(pulse_us - width) > tolerance:
            continue
        if fall - before < 0 or fall + after > values.size:
            continue
        lines.append(values[fall - before:fall + after])
    if not lines:
        raise ValueError("no horizontal sync pulse of the specified width found")
    return np.array(lines), before


def series_background(lines, sample_rate_hz: float, fall_index: int,
                      sysparams: Dict[str, object], label: str = "",
                      groups: int = 8) -> Dict[str, object]:
    """Demodulated sync-pulse lines, aligned on the sync fall, as a background
    for the time-series injection.

    `lines` is `(lines x samples)` in IRE; `fall_index` the sample at which
    the fall crosses half amplitude. The rise is MEASURED, not assumed: the
    mean pulse's trailing half-amplitude crossing
    (`standard_levels.half_amplitude_crossings`), because the tapes' sync is
    not the specified 4.70 us (the 75-bar tape reads 4.686). Every window is
    then placed from the format's timing table and from that measured edge,
    and nothing is read from the picture:

        front porch   from one front porch before the fall to one transition
                      before it (`frontPorchUS`, `syncTransitionUS`)
        tip           inside the pulse, three transition times clear of the
                      fall and of the measured rise
        tail          the back porch from three transition times after the
                      measured rise crossing - clear of the edge itself,
                      rule 45 - to the end margin. It deliberately contains
                      the edge's aftermath, the ringing and relaxation the
                      ring reader exists to read
        porch         the back porch AFTER the burst's specified position
                      (`colorBurstUS[1]`), the standing constraint on what
                      may be read as a LEVEL
        wide          the whole line window

    THE END MARGIN IS THREE TRANSITIONS BEFORE THE TABLE'S ACTIVE START, and
    it is measured rather than chosen: on the 75-bar SP capture the first
    bar's leading edge begins 0.2 us before `activeVideoUS[0]` (its
    pre-shoot at 9.15 to 9.22 us, the edge at 9.29), so a window ending one
    transition before the table's 9.45 us read the picture. Three
    transitions, 0.42 us, ends every window at 9.03 us, where the mean
    pulse is still at its porch level (2026-09-05).

    Lines are dealt into `groups` interleaved groups, so every group spans
    the whole window and the groups' readings are the estimator's own
    spread. THE NOISE IS PER SAMPLE: the scatter across lines at each
    sample of the window, and the group mean's noise is that over the root
    of the lines per group. It is per sample because the reserved interval
    is not uniformly quiet, measured on the 75-bar SP capture, 760 lines:

        sync tip                       1.1 IRE across lines
        front porch                    1.2 to 1.6
        burst position, 5.3 to 7.8 us  rising to 13.4 at 6.15 us
        porch after the burst          7.5 at 7.82 us falling to 1.5 at 8.9

    What sits at the burst position is the colour-under beat
    (`demodulate_luma`): the burst's phase alternates line to line, so it
    is absent from the mean pulse and present in every line. Its tail is
    still 2.6 times the tip's scatter at the table's burst end and reaches
    the tip's level only at `burst_tail_extinction_us` (8.80 us here),
    which is why the after-burst porch's within-line scatter, 4.10 IRE, is
    3.4 times the tip's 1.19 - `porch_contamination` - and why the field
    clip witness that uses that porch as its comparator fires on the
    unclipped tip (`read_series`). The windows are NOT moved for it: they
    stay where the standing constraint puts them, and the contamination is
    reported as a measurement of that constraint on this tape.

    On the capture: tip -35.0 IRE, porch +3.9, the sync width 4.686 us, the
    tip's cross-line scatter 1.10 IRE and the group mean's 0.113; the mean
    pulse overshoots to +17.8 IRE 0.21 us after the rise and rings back
    through 7.5, 7.9, 4.6, 5.5, 4.0 IRE over the next microsecond before
    settling at 4.3 (2026-09-05).
    """
    block = np.asarray(lines, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] < 2 * int(groups):
        raise ValueError("lines must be (lines x samples) with at least two "
                         "lines per group")
    rate = float(sample_rate_hz)
    count, length = block.shape
    time_us = (np.arange(length, dtype=np.float64) - int(fall_index)) / rate * 1e6
    transition = float(sysparams["syncTransitionUS"])
    active_start = float(sysparams["activeVideoUS"][0])
    burst_end = float(sysparams["colorBurstUS"][1])
    front_porch = float(sysparams["frontPorchUS"])
    end_margin = active_start - 3.0 * transition
    mean = block.mean(axis=0)

    crossings = standard_levels.half_amplitude_crossings(mean, rate)
    rise_sample = _trailing_crossing(mean, crossings["half_amplitude"])
    rise_time = (rise_sample - int(fall_index)) / rate * 1e6

    def window(start_us, stop_us):
        inside = np.where((time_us >= start_us) & (time_us <= stop_us))[0]
        if inside.size < 4:
            raise ValueError(f"window {start_us:.2f}-{stop_us:.2f} us holds "
                             "too few samples")
        return slice(int(inside[0]), int(inside[-1]) + 1)

    windows = {
        "front porch": window(-front_porch + transition, -transition),
        "tip": window(3.0 * transition, rise_time - 3.0 * transition),
        "tail": window(rise_time + 3.0 * transition, end_margin),
        "porch": window(burst_end, end_margin),
        "wide": window(-front_porch + transition, end_margin),
    }
    membership = np.arange(count) % int(groups)
    per_group = count / float(groups)

    # the noise, per sample: the scatter across lines
    sigma_line = block.std(axis=0, ddof=1)
    sigma_mean = sigma_line / np.sqrt(per_group)
    tip_noise = float(np.median(sigma_line[windows["tip"]]))

    # the within-line scatter the clip witness measures, per window
    def within_line(rows):
        t = np.arange(rows.shape[1], dtype=np.float64)
        centred = t - t.mean()
        denominator = float(centred @ centred) or 1.0
        slope = (rows - rows.mean(axis=1, keepdims=True)) @ centred / denominator
        fit = rows.mean(axis=1, keepdims=True) + slope[:, None] * centred[None, :]
        return float(np.median(np.std(rows - fit, axis=1)))

    scatter = {name: within_line(block[:, w]) for name, w in windows.items()
               if name != "wide"}
    contamination = scatter["porch"] / max(scatter["tip"], 1e-300)

    # where the burst's tail has gone: the first sample after the table's
    # burst end at which the cross-line scatter is within its own standard
    # error (a standard deviation over N lines carries 1 / root(2 (N - 1)))
    # of the level it holds over the last quarter of the porch window
    porch = windows["porch"]
    quarter = max((porch.stop - porch.start) // 4, 1)
    end_level = float(np.median(sigma_line[porch.stop - quarter:porch.stop]))
    relative_se = 1.0 / np.sqrt(2.0 * max(count - 1, 1))
    settled = np.where(sigma_line[porch] <= end_level
                       * (1.0 + DETECT_SIGMA * relative_se))[0]
    extinction = (porch.start + int(settled[0])) if settled.size else porch.stop
    extinction_us = float(time_us[min(extinction, length - 1)])

    # a DIAGNOSTIC, not a window: where the mean pulse's slope first stays
    # below what the noise alone produces (a difference of two samples has
    # root two times the mean's noise) for three consecutive samples
    slope = np.abs(np.diff(mean))
    floor_index = int(round(rise_sample + 3.0 * transition * rate / 1e6))
    settle = floor_index
    for index in range(floor_index, length - 4):
        gate = DETECT_SIGMA * np.sqrt(2.0) * sigma_line[index] / np.sqrt(count)
        if np.all(slope[index:index + 3] < gate):
            settle = index
            break
    settle_time = (settle - int(fall_index)) / rate * 1e6

    return {
        "representation": "series",
        "label": label,
        "lines": block,
        "time_us": time_us,
        "sample_rate_hz": rate,
        "fall_index": int(fall_index),
        "rise_index": int(round(rise_sample)),
        "rise_time_us": float(rise_time),
        "settle_time_us": float(settle_time),
        "width_us": float(crossings["width_s"]) * 1e6,
        "windows": windows,
        "groups": int(groups),
        "membership": membership,
        "model": mean,
        "tip_level": float(np.median(block[:, windows["tip"]])),
        "porch_level": float(np.median(block[:, windows["porch"]])),
        # the level a clip's depth is counted from: what the mean pulse
        # reaches inside the tip, which is what `extrapolate_through_clip`
        # continues the unclipped model down to
        "clip_floor": float(mean[windows["tip"]].min()),
        "sigma_line": sigma_line,
        "sigma_mean": sigma_mean,
        "tip_noise": tip_noise,
        "tip_noise_mean": tip_noise / np.sqrt(per_group),
        "within_line_scatter": scatter,
        "porch_contamination": float(contamination),
        "burst_tail_extinction_us": extinction_us,
        "sysparams": dict(sysparams),
    }


def _trailing_crossing(pulse: np.ndarray, level: float) -> float:
    """The last crossing of `level`, with sub-sample interpolation - the
    rise of a sync pulse aligned on its fall."""
    values = np.asarray(pulse, dtype=np.float64)
    for index in range(values.size - 1, 0, -1):
        first, second = values[index - 1], values[index]
        if (first - level) * (second - level) <= 0 and first != second:
            return (index - 1) + (first - level) / (first - second)
    raise ValueError("the pulse has no trailing crossing")


def cube_background(paths: Dict[str, str], tapes: Sequence[str],
                    band_hz=(0.2e6, 4.0e6)) -> Dict[str, object]:
    """The tesseract's vertices as a background: `tesseract.from_sync_exports`
    unchanged, head x polarity x half x tape over two tapes' exports, with
    the grid's own sample rate read from the export (`rate_mhz`, the 4 f_sc
    output rate whose 4096-point transform the 2049-bin grid is) so that
    `hypercomplex.causality` places the contrasts back on the full axis.

    On the cd and home exports, 944 bins, every one of the fifteen
    contrasts already stands above its noise (`tesseract.unreached`, top
    order z 260), so a detection threshold on the cube is saturated by the
    real structure and what an injection can test there is RECOVERY:
    whether the contrast it was planted on returns it at its size, whether
    any other contrast takes a share, which stage `tesseract.trace` charges
    it to, and how `causality` reads it (2026-09-05).
    """
    cube = tesseract.from_sync_exports(paths, tapes, band_hz)
    rates = set()
    for tape in tapes:
        data = np.load(paths[tape], allow_pickle=True)
        if "rate_mhz" in data.files:
            rates.add(float(data["rate_mhz"]) * 1e6)
    if len(rates) > 1:
        raise ValueError("the exports disagree on their sample rate")
    step = float(np.median(np.diff(cube.bins)))
    rate = rates.pop() if rates else None
    return {
        "representation": "cube",
        "label": " and ".join(tapes),
        "cube": cube,
        "frequency_hz": np.asarray(cube.bins, dtype=np.float64),
        "sigma": np.sqrt(cube.variance),
        "sample_rate_hz": rate,
        "bin_step_hz": step,
    }


# --------------------------------------------------------------------------
# The injection
# --------------------------------------------------------------------------


def ring_response(frequency_hz, ring_hz: float, decay_s: float) -> np.ndarray:
    """The transfer of a damped sinusoid `exp(-t/tau) sin(w0 t)` starting at
    the transient - the Laplace transform `w0 / ((s + 1/tau)^2 + w0^2)` on
    the imaginary axis, normalised to unit peak so its amplitude means the
    peak ripple it adds to the transfer. Its poles are in the left half
    plane and it has no zeros, so its logarithm is minimum phase: a planted
    ring should read as such to `hypercomplex.causality`."""
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    omega = 2.0 * np.pi * float(ring_hz)
    s = 1.0 / max(float(decay_s), 1e-12) + 2j * np.pi * grid
    response = omega / (s ** 2 + omega ** 2)
    return response / max(float(np.max(np.abs(response))), 1e-30)


def default_delay_s(frequency_hz) -> float:
    """An echo delay the band resolves as a ripple: three ripple periods
    across the band, so it lies BETWEEN the key's own echo entries, which
    `interference.signatures` places at 1, 2, 4 and 8 periods. A caller who
    wants the entry itself to be tested passes one of those."""
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    return 3.0 / max(float(grid.max() - grid.min()), 1.0)


def key_delay_s(frequency_hz, periods: int = 2) -> float:
    """The delay of one of the key's own echo entries: `periods` ripple
    periods across the band, the rule `interference.signatures` uses."""
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    return float(periods) / max(float(grid.max() - grid.min()), 1.0)


def grid_delta(kind: str, amplitude: float, frequency_hz, log_value=None,
               **parameters) -> np.ndarray:
    """The log-domain change a kind makes on a grid, complex over the bins:
    `log K(f)` for an LTI kind, the resampling difference for a frequency
    scaling (which needs the log transfer it stretches). This is what is
    ADDED to the log transfer, and it is the only place the injection's
    shape is written down.

    A frequency scaling of `a` is a tape played fast by that fraction: every
    recorded feature arrives at `f (1 + a)`, so the stretched transfer is
    `H(f / (1 + a))` - the same convention as the series, where the
    waveform is compressed in time by `(1 + a)`.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    a = float(amplitude)
    if kind == "echo":
        delay = float(parameters.get("delay_s", default_delay_s(grid)))
        return np.log(1.0 + a * np.exp(-2j * np.pi * grid * delay))
    if kind == "ring":
        shape = ring_response(grid, parameters.get("ring_hz", MEASURED_RING_HZ),
                              parameters.get("decay_s", MEASURED_RING_DECAY_S))
        return np.log(1.0 + a * shape)
    if kind == "level step":
        return np.full(grid.size, a, dtype=np.complex128)
    if kind == "phase rotation":
        return np.full(grid.size, 1j * a, dtype=np.complex128)
    if kind == "frequency scaling":
        if log_value is None:
            raise ValueError("a frequency scaling needs the log transfer it "
                             "stretches")
        value = np.asarray(log_value, dtype=np.complex128).ravel()
        source = grid / (1.0 + a)
        # log magnitude and unwrapped phase interpolated separately - the
        # log-domain interpolation `correction_export.apply_correction`
        # argues for; the edges are held, so a bin pushed past the band
        # keeps the band's edge value
        magnitude = np.interp(source, grid, value.real)
        phase = np.interp(source, grid, value.imag)
        return (magnitude - value.real) + 1j * (phase - value.imag)
    if kind == "clip":
        raise ValueError("a clip is memoryless and has no transfer to plant "
                         "on a grid; inject it into a series")
    raise ValueError(f"unknown kind {kind!r}; the kinds are {KINDS}")


def series_lines(kind: str, amplitude: float, background: Dict[str, object],
                 **parameters) -> np.ndarray:
    """The lines of a series background with a kind planted, per line.

    A clip is `max(line, clip_floor + depth)`: a limiter a known depth above
    the level the mean pulse reaches inside the tip, which is the level
    `clipping.extrapolate_through_clip` continues the unclipped model down
    to, so the reader's depth and the planted depth are one quantity. A
    negative depth sits below what the mean reaches and touches only the
    noise excursions. A ring is attached to the START of the transient that
    drives the back porch - the measured rise - per the arc's
    transient-start model. An echo is a delayed copy at a known reflection
    coefficient, padded before the window with the line's first sample. A
    frequency scaling of `a` compresses the time axis about the fall by
    `(1 + a)`, a tape played fast: the fall stays put and the rise arrives
    early. A phase rotation rotates each line's analytic signal about its
    mean.
    """
    lines = np.array(background["lines"], dtype=np.float64, copy=True)
    t = background["time_us"]
    rate = background["sample_rate_hz"]
    a = float(amplitude)
    if kind == "clip":
        return np.maximum(lines, background["clip_floor"] + a)
    if kind == "level step":
        return lines + a
    if kind == "ring":
        ring_hz = float(parameters.get("ring_hz", MEASURED_RING_HZ))
        decay_us = float(parameters.get("decay_s", MEASURED_RING_DECAY_S)) * 1e6
        lag = t - background["rise_time_us"]
        after = lag >= 0.0
        wave = np.zeros_like(t)
        wave[after] = (np.exp(-lag[after] / max(decay_us, 1e-9))
                       * np.cos(2.0 * np.pi * ring_hz * lag[after] * 1e-6))
        return lines + a * wave[None, :]
    if kind == "echo":
        delay = float(parameters.get(
            "delay_s", default_delay_s(np.fft.rfftfreq(t.size, 1.0 / rate))))
        shift = int(round(delay * rate))
        if shift >= lines.shape[1]:
            raise ValueError("the echo delay exceeds the window")
        delayed = np.empty_like(lines)
        delayed[:, shift:] = lines[:, :lines.shape[1] - shift]
        delayed[:, :shift] = lines[:, :1]
        return lines + a * delayed
    if kind == "frequency scaling":
        source = t * (1.0 + a)
        return np.array([np.interp(source, t, row) for row in lines])
    if kind == "phase rotation":
        from scipy.signal import hilbert
        centred = lines - lines.mean(axis=1, keepdims=True)
        quadrature = np.imag(hilbert(centred, axis=1))
        return (lines.mean(axis=1, keepdims=True)
                + np.cos(a) * centred + np.sin(a) * quadrature)
    raise ValueError(f"unknown kind {kind!r}; the kinds are {KINDS}")


def group_means(lines, membership, groups: int) -> np.ndarray:
    return np.array([lines[membership == g].mean(axis=0) for g in range(groups)])


def cube_sign(cube: "tesseract.Cube", subset: Sequence[str]) -> np.ndarray:
    """The sign each vertex gives a component that differentiates on
    `subset`: +1 where the vertex has state 0 on every axis of the subset,
    flipped once per axis in state 1 - the pattern the contrast on that
    subset reads with weight one and every other contrast with weight zero."""
    signs = np.ones((2,) * cube.order)
    for vertex in cube.vertices:
        sign = 1.0
        for axis in subset:
            if vertex[cube.axes.index(axis)]:
                sign = -sign
        signs[vertex] = sign
    return signs


def cube_deltas(kind: str, amplitude: float, background: Dict[str, object],
                **parameters) -> np.ndarray:
    """The per-vertex log change of a kind planted on the cube with the
    sign pattern of `subset` (default `("head",)`, a head difference).

    For an LTI kind the change is the same at every vertex up to its sign,
    so by the linearity of the fold it lands on the subset's contrast alone.
    A frequency scaling stretches each vertex's OWN transfer, and a vertex's
    transfer is the sum of every contrast with that vertex's signs, so the
    change on the head pattern carries the stretch of the grand mean onto
    the head contrast and the stretch of every other contrast onto the
    contrast that differs from it by the head axis: real cross-talk, which
    is what a speed error confined to one head does to the cube.
    """
    cube = background["cube"]
    subset = tuple(parameters.get("subset", ("head",)))
    signs = cube_sign(cube, subset)
    deltas = np.zeros_like(cube.values)
    for vertex in cube.vertices:
        deltas[vertex] = signs[vertex] * grid_delta(
            kind, amplitude, cube.bins, cube.values[vertex],
            **{k: v for k, v in parameters.items() if k != "subset"})
    return deltas


def unit_shape(kind: str, background: Dict[str, object], **parameters):
    """The injected change per unit natural amplitude, to first order, on
    the background - the direction the stage readers project onto. None for
    a series, where the readers speak natural units directly."""
    small = CALIBRATION_AMPLITUDE
    if background["representation"] == "grid":
        return grid_delta(kind, small, background["frequency_hz"],
                          background["log"], **parameters) / small
    if background["representation"] == "cube":
        return cube_deltas(kind, small, background, **parameters) / small
    return None


def unit_response(kind: str, background: Dict[str, object],
                  **parameters) -> Optional[float]:
    """The rms per-bin significance per unit of the kind's natural
    amplitude, to first order: the conversion between the natural amplitude
    and the amplitude in SE units. None for a clip, which has no linear
    regime and is placed by `clip_depth_at` instead."""
    small = CALIBRATION_AMPLITUDE
    if background["representation"] in ("grid", "cube"):
        return significance(unit_shape(kind, background, **parameters),
                            background["sigma"])["rms"]
    if kind == "clip":
        return None
    before = group_means(background["lines"], background["membership"],
                         background["groups"])
    after = group_means(series_lines(kind, small, background, **parameters),
                        background["membership"], background["groups"])
    return significance((after - before) / small, background["sigma_mean"])["rms"]


def clip_depth_at(amplitude_se: float, background: Dict[str, object]
                  ) -> float:
    """THE CLIP'S DEPTH AT A GIVEN SIGNIFICANCE, by bisection.

    A limiter's change to the group means grows monotonically with its
    level: from nothing while it sits below every tip sample, through the
    noise excursions, to the whole pulse once it reaches the porch. The
    depth above `clip_floor` whose rms change over the per-sample noise
    equals `amplitude_se` is therefore unique, and it puts the clip on the
    SAME unit as every other kind. NaN when even a limiter at the porch
    level - the pulse gone entirely, which is a dropout and not a clip -
    cannot reach the requested significance.

    On the 75-bar SP capture the smallest depth that touches a line is
    -5.0 IRE below the mean's floor; one SE is a limiter 3.0 IRE below the
    floor, touching the deepest noise excursions of a few lines per group;
    the mean's own floor is reached at about 12 SE; 64 SE is a limiter 8.4
    IRE above the floor, 6 IRE above the tip's median (2026-09-05).
    """
    lines = background["lines"]
    tip = background["windows"]["tip"]
    membership, groups = background["membership"], background["groups"]
    before = group_means(lines, membership, groups)
    floor = background["clip_floor"]

    def size(level):
        after = group_means(np.maximum(lines, level), membership, groups)
        return significance(after - before, background["sigma_mean"])["rms"]

    low = float(lines[:, tip].min())
    high = float(background["porch_level"])
    target = float(amplitude_se)
    if size(high) < target:
        return float("nan")
    for _ in range(60):
        middle = 0.5 * (low + high)
        if size(middle) < target:
            low = middle
        else:
            high = middle
        if high - low < 1e-6 * max(abs(high), 1.0):
            break
    return 0.5 * (low + high) - floor


def natural_amplitude(kind: str, amplitude_se: float,
                      background: Dict[str, object], **parameters) -> float:
    """The natural amplitude that puts a kind at `amplitude_se` times the
    measured noise (see the module docstring for the unit)."""
    if kind == "clip" and background["representation"] == "series":
        return clip_depth_at(amplitude_se, background)
    unit = unit_response(kind, background, **parameters)
    return float(amplitude_se) / max(unit, 1e-300)


def inject(kind: str, amplitude: float, target: Dict[str, object],
           **parameters) -> Dict[str, object]:
    """PLANT A KNOWN COMPONENT AT A KNOWN AMPLITUDE. Never fitted.

    `amplitude` is in the kind's natural units: the reflection coefficient
    of an echo, the peak ripple of a ring (IRE in a series), nepers for a
    level step on the grid and IRE in a series, the fractional speed error
    of a frequency scaling, radians for a phase rotation, and for a clip the
    depth in IRE of the limiter above the level the mean pulse reaches in
    the tip.

    On the grid the transfer is multiplied by `exp(delta)`, and so are both
    held-out halves (`halves="both"`; `"fit"` or `"judge"` plants it in one
    half only, a component that does NOT reproduce). On the cube the
    component is given the sign pattern of `subset` over the axes.

    Returns a copy of the target with the component planted and an
    `injected` record stating exactly what was planted and how large it is
    in SE units, so a caller can never lose track of the truth the recovery
    is judged against.
    """
    representation = target["representation"]
    out = dict(target)
    a = float(amplitude)
    if representation == "grid":
        delta = grid_delta(kind, a, target["frequency_hz"], target["log"],
                           **{k: v for k, v in parameters.items()
                              if k != "halves"})
        factor = np.exp(delta)
        out["H"] = target["H"] * factor
        out["log"] = target["log"] + delta
        halves = str(parameters.get("halves", "both"))
        if target.get("held_out") is not None:
            held = dict(target["held_out"])
            if halves in ("both", "fit"):
                held["H_fit"] = held["H_fit"] * factor
            if halves in ("both", "judge"):
                held["H_judge"] = held["H_judge"] * factor
            out["held_out"] = held
        size = significance(delta, target["sigma"])
    elif representation == "series":
        if kind not in SERIES_KINDS:
            raise ValueError(f"unknown kind {kind!r}; the kinds are {KINDS}")
        lines = series_lines(kind, a, target, **parameters)
        out["lines"] = lines
        before = group_means(target["lines"], target["membership"], target["groups"])
        after = group_means(lines, target["membership"], target["groups"])
        size = significance(after - before, target["sigma_mean"])
        if kind == "clip":
            size["depth_over_tip_noise"] = a / max(target["tip_noise"], 1e-300)
            size["lines_touched"] = float(np.mean(np.any(
                lines[:, target["windows"]["tip"]]
                != target["lines"][:, target["windows"]["tip"]], axis=1)))
    elif representation == "cube":
        deltas = cube_deltas(kind, a, target, **parameters)
        cube = target["cube"]
        out["cube"] = tesseract.Cube(cube.axes, cube.values + deltas,
                                     cube.variance, cube.bins, cube.channels)
        size = significance(deltas, target["sigma"])
    else:
        raise ValueError("the target must be a grid, a series or a cube")
    out["injected"] = {"kind": kind, "amplitude": a,
                       "parameters": dict(parameters),
                       "amplitude_se": size["rms"], "peak_se": size["peak"],
                       "snr": size["snr"], "count": size["count"],
                       **{k: v for k, v in size.items()
                          if k not in ("rms", "peak", "snr", "count")}}
    return out


# --------------------------------------------------------------------------
# Reading the chain: the grid, through the arc's own judge
# --------------------------------------------------------------------------


def stage_of(name: str, chain: Optional[Dict[str, int]] = None) -> str:
    """The link of the chain a key entry acts in, from its declared position
    and `stage_boundaries.LINKS`; `undeclared` where it has none."""
    chain = chain if chain is not None else inf.full_chain()
    position = inf.position_of(name, chain)
    if position is None:
        return "undeclared"
    for link, low, high in sb.LINKS:
        if low <= position <= high:
            return link
    return "undeclared"


def stage_names() -> List[str]:
    """Every column of the grid matrix: the chain's links, the nuisance and
    the residual."""
    return [link for link, _low, _high in sb.LINKS] + [
        "undeclared", NUISANCE_STAGE, RESIDUAL_STAGE]


def weighted_projection(shape, values, sigma) -> float:
    """The real coefficient of `values` on `shape` in the fit's own inner
    product - stacked real and imaginary parts, each bin weighted by its
    inverse variance. Linear in `values`, so responses subtract exactly."""
    s = np.asarray(shape).ravel()
    v = np.asarray(values).ravel()
    w = 1.0 / np.maximum(np.asarray(sigma, dtype=np.float64).ravel() ** 2, 1e-300)
    denominator = float(np.sum(np.abs(s) ** 2 * w))
    if not denominator > 0:
        return 0.0
    return float(np.sum((s.real * v.real + s.imag * v.imag) * w) / denominator)


def echo_entry_amplitude() -> float:
    """The reflection coefficient of the key's own echo entries, read from
    `interference.echo`'s signature rather than typed: a fitted coefficient
    of one on an echo entry is an echo of this size, to first order."""
    return float(inspect.signature(inf.echo).parameters["amplitude"].default)


def read_grid(target: Dict[str, object],
              keys: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
              chain: Optional[Dict[str, int]] = None,
              shape=None) -> Dict[str, object]:
    """EVERY GRID READER, RUN UNCHANGED, on one target.

    Per key: `residual_floor.test` (in-sample and held-out reduced
    chi-square, held-out lag-one whiteness, the half-split agreement of
    `structureless`, the fitted coefficients); `stage_boundaries.survey` and
    `search_families` on the residual the key leaves; `interference.
    outlier_response` on the same residual; and `correction_export.collapse`
    of the fitted key, whose collapsed log-sum is what a correction would
    subtract.

    With `shape` (the injected change per unit amplitude) the fit is split
    by stage: the projection onto `shape` of each link's fitted part, of the
    nuisance terms and of the residual, which sum to the projection of the
    whole target. Entry readers in natural units are emitted for the echo
    entries (coefficient times the entry's own reflection coefficient), the
    level (nepers) and the delay (the coefficient's own units, phase at the
    band's edge).

    THE CHI-SQUARE READERS RESPOND LINEARLY, NOT QUADRATICALLY, and can fall.
    Every export's residual after the key is hundreds of times its floor
    (`residual_floor`, its docstring), so the change the injection makes to
    the chi-square is dominated by the cross term between what it leaves in
    the residual and what was already there, which is linear in the
    amplitude and of either sign: on pnb head A an echo at the key's own
    delay raises the reduced chi-square by 5 per SE unit, and a phase
    rotation LOWERS it by 10 per SE unit because the re-fit of the
    collinear delay and short-echo entries moves them through the existing
    residual (2026-09-05).

    Returns `readers` (name -> number) and `detail` (name -> anything).
    """
    grid = target["frequency_hz"]
    chain = chain if chain is not None else inf.full_chain()
    keys = keys if keys is not None else {"default": dict(inf.signatures(grid))}
    sigma = target["sigma"]
    readers: Dict[str, float] = {}
    detail: Dict[str, object] = {}
    echo_unit = echo_entry_amplitude()
    for label, key in keys.items():
        result = residual_floor.test(grid, target["H"], target["se"], key,
                                     target.get("held_out"))
        design = residual_floor.design(key, grid)
        names = design["names"]
        matrix = design["matrix"]
        coefficients = np.array([result["coefficients"][n] for n in names])
        explained = matrix @ coefficients
        residual = target["log"] - explained
        prefix = f"floor[{label}]"
        readers[f"{prefix}: in-sample reduced chi-square"] = \
            result["in_sample"]["reduced_chi_square"]
        readers[f"{prefix}: in-sample lag-one z"] = \
            result["in_sample"]["whiteness"]["lag_one_z"]
        readers[f"{prefix}: at floor"] = float(result["in_sample"]["at_floor"])
        readers[f"{prefix}: white"] = float(result["in_sample"]["whiteness"]["white"])
        detail[f"{prefix}: distribution"] = result["in_sample"]["distribution"]
        detail[f"{prefix}: reading"] = result["in_sample"]["reading"]
        if "held_out" in result:
            readers[f"{prefix}: held-out reduced chi-square"] = \
                result["held_out"]["reduced_chi_square"]
            readers[f"{prefix}: held-out lag-one z"] = \
                result["held_out"]["whiteness"]["lag_one_z"]
            readers[f"{prefix}: agreement"] = float(
                result["structureless"].get("agreement", np.nan))
        for name, coefficient in zip(names, coefficients):
            if name.startswith("echo at"):
                readers[f"entry[{label}]: {name}"] = float(coefficient) * echo_unit
            elif name in ("level", "delay"):
                readers[f"entry[{label}]: {name}"] = float(coefficient)
        if shape is not None:
            parts: Dict[str, np.ndarray] = {}
            for name, coefficient, column in zip(names, coefficients, matrix.T):
                stage = (NUISANCE_STAGE if name in ("level", "delay")
                         else stage_of(name, chain))
                parts[stage] = parts.get(stage, 0.0) + coefficient * column
            for stage in stage_names():
                if stage == RESIDUAL_STAGE:
                    continue
                part = parts.get(stage)
                readers[f"stage[{label}]: {stage}"] = (
                    weighted_projection(shape, part, sigma) if part is not None
                    else 0.0)
            readers[f"stage[{label}]: {RESIDUAL_STAGE}"] = \
                weighted_projection(shape, residual, sigma)
            readers[f"stage[{label}]: whole"] = \
                weighted_projection(shape, target["log"], sigma)
            # the correction the fitted key would apply: the collapse of the
            # key with the fitted weights, compared on the injected shape
            key_weights = {n: float(c) for n, c in zip(names, coefficients)
                           if n not in ("level", "delay")}
            collapsed = correction_export.collapse(key, grid, key_weights, chain)
            log_sum = collapsed["log_sum"]
            readers[f"correction[{label}]: carries"] = weighted_projection(
                shape, log_sum - log_sum.mean(), sigma)
        # the residual the key leaves, offered to every join and to the
        # outlier estimator, unchanged
        offered = {str(join["boundary"]): residual for join in sb.boundaries(chain)}
        survey = sb.survey(offered, grid, chain)
        for join in survey["joins"]:
            got = join.get("estimate") or {}
            readers[f"join[{label}]: {join['boundary']}"] = \
                float(got.get("explained", 0.0) or 0.0)
        families = sb.search_families(residual, grid)
        readers[f"join[{label}]: identified"] = float(families.get("identified", False))
        detail[f"join[{label}]: winner"] = families.get("winner")
        detail[f"join[{label}]: winning join"] = families.get("winning_join")
        detail[f"join[{label}]: value"] = families.get("value")
        outliers = inf.outlier_response(residual, grid, sigma)
        readers[f"outliers[{label}]: inside"] = float(outliers.get("inside", 0.0))
        readers[f"outliers[{label}]: fraction"] = float(
            outliers.get("outlier_fraction", 0.0))
        detail[f"outliers[{label}]: carried by"] = outliers.get("carried_by")
    return {"readers": readers, "detail": detail}


# --------------------------------------------------------------------------
# Reading the chain: the time-series readers
# --------------------------------------------------------------------------


def _ring_scan(series: Dict[str, object], counts: Tuple[int, int]):
    """A coarser scan than `relaxation_or_ringing`'s default, spanning the
    same ranges it derives from the window - the full 120 x 90 scan costs
    about a second per call and a sweep makes hundreds of them."""
    t = series["time_us"][series["windows"]["tail"]]
    span = float(t.max() - t.min())
    step = float(np.median(np.diff(t)))
    decays = np.geomspace(0.05 * span, 4.0 * span, int(counts[0]))
    frequencies = np.linspace(1.0 / span, 0.5 / step, int(counts[1])) * 1e6
    return decays, frequencies


SERIES_READERS: Tuple[str, ...] = (
    "clip", "clip (front porch comparator)", "clip depth", "tip follows porch",
    "level", "ring", "ring frequency", "ring decay", "timing",
    "distribution (tip)", "distribution (tail)", "front porch")

# The readers whose reading is an absolute quantity that should land ON the
# injected parameter rather than a difference that should track the
# amplitude: judged by agreement with the parameter, not by a threshold.
ABSOLUTE_READERS: Tuple[str, ...] = ("ring frequency", "ring decay")


def series_stage(reader: str, chain: Optional[Dict[str, int]] = None) -> str:
    """The link of the chain a series reader's estimator is declared in:
    the clip readers at `clipping.CHAIN_POSITION`, the ring and the front
    porch at `sync_geometry.CHAIN_POSITION`, the level at the picture
    stage's sync amplitude, the timing at its time base residual, and the
    distribution at the capture, whose beat entry is the only one there."""
    chain = chain if chain is not None else inf.full_chain()
    owner = {
        "clip": clipping.CHAIN_PREFIX,
        "clip (front porch comparator)": clipping.CHAIN_PREFIX,
        "clip depth": clipping.CHAIN_PREFIX,
        "tip follows porch": clipping.CHAIN_PREFIX,
        "ring": sg.CHAIN_PREFIX, "ring frequency": sg.CHAIN_PREFIX,
        "ring decay": sg.CHAIN_PREFIX, "front porch": sg.CHAIN_PREFIX,
        "level": "sync amplitude", "timing": "time base residual",
        "distribution (tip)": "beat / co-channel",
        "distribution (tail)": "beat / co-channel",
    }.get(reader)
    return stage_of(owner, chain) if owner else "undeclared"


def read_series(series: Dict[str, object],
                ring_scan: Tuple[int, int] = (30, 45)) -> Dict[str, object]:
    """Every time-domain reader, run unchanged, per group of lines.

    Each reader returns one `amount` per group in the natural units of the
    kind it targets, and a `flag` where the estimator gives a verdict:

        clip                `clipping.tip_clip_from_lines` on the group's
                            tip and after-burst porch blocks - the noise
                            witness with the comparator the arc uses;
                            amount = tip/porch noise ratio
        clip (front porch   the same witness with the FRONT porch as the
        comparator)         comparator - not a level, per the standing
                            constraint, but a fair scatter comparator
        clip depth          `clipping.extrapolate_through_clip` on the group
                            mean's line window, the background's own mean
                            pulse as the unclipped model and the mask the
                            samples of the tip window sitting within the
                            detector's own tolerance of its extreme;
                            amount = depth in IRE
        tip follows porch   `clipping.clip_offset_from_porch` on the per-line
                            tip and porch levels - the control that a level
                            change is not read as a clip; amount = slope
        level               the after-burst porch's departure from the
                            standard's blanking (`standard_levels`);
                            amount = IRE
        ring                `sync_geometry.relaxation_or_ringing` on the
                            group mean's back-porch tail, with time counted
                            FROM THE MEASURED RISE, so the fitted amplitude
                            is the ring's amplitude at the transient it is
                            attached to; amount = IRE
        ring frequency,     the same fit's frequency (Hz) and decay (s) -
        ring decay          ABSOLUTE readers, judged by whether they land on
                            the planted parameter
        timing              `standard_levels.half_amplitude_crossings` on
                            the group mean; amount = the fractional speed
                            error that would explain the width,
                            `spec / width - 1`
        distribution (tip)  `interference_distributions.classify` on the
        distribution (tail) group's lines about their mean over the tip and
                            over the tail; amount = excess kurtosis
        front porch         `sync_geometry.front_porch_residual` on the
                            group mean's front porch, the porch's first
                            sample as the level it enters from and the
                            after-burst porch as the settled level; amount =
                            residual rms in IRE

    THREE READERS ARE SATURATED ON THE UNCLIPPED 75-BAR BACKGROUND, and each
    is a measurement of its comparator rather than of the tip. `clip` reads
    a tip/porch ratio of 0.29 and fires on every group, because the
    after-burst porch carries the colour-under burst's tail
    (`series_background`); with the front porch as the comparator the same
    witness reads 0.72 and does not fire. `tip follows porch` reads a slope
    of -0.02 to +0.01 and fires, because line to line the porch's level
    moves with that same tail, which the tip does not share, and the tip's
    own movement is its noise: its correlation with the porch is -0.1.
    `distribution (tail)` reads an excess kurtosis of -0.49, between a
    Gaussian and the arcsine of a carrier, which is the beat's own
    distribution, and `classify` labels that platykurtic residual "heavy
    tailed, none of these" - the label is the estimator's. The per-line
    clip detector is not used at all: at this tape's noise its extreme-run
    mask selects one or two samples per line and its scatter is then zero
    (all 2026-09-05).
    """
    lines = series["lines"]
    t = series["time_us"]
    windows = series["windows"]
    model = series["model"]
    membership = series["membership"]
    rate = series["sample_rate_hz"]
    decays, frequencies = _ring_scan(series, ring_scan)
    spec_width = float(series["sysparams"]["hsyncPulseUS"])
    tolerance = float(inspect.signature(clipping.detect_tip_clip)
                      .parameters["tolerance"].default)
    tip, porch, tail, wide = (windows["tip"], windows["porch"],
                              windows["tail"], windows["wide"])
    front = windows["front porch"]
    out: Dict[str, Dict[str, list]] = {
        name: {"amount": [], "flag": [], "detail": []} for name in SERIES_READERS}

    for g in range(series["groups"]):
        rows = lines[membership == g]
        mean = rows.mean(axis=0)

        witness = clipping.tip_clip_from_lines(rows[:, tip], rows[:, porch])
        out["clip"]["amount"].append(float(witness.get("noise_ratio", np.nan)))
        out["clip"]["flag"].append(bool(witness.get("clipped", False)))
        out["clip"]["detail"].append({"tip_noise": witness.get("tip_noise"),
                                      "porch_noise": witness.get("porch_noise")})

        against_front = clipping.tip_clip_from_lines(rows[:, tip], rows[:, front])
        out["clip (front porch comparator)"]["amount"].append(
            float(against_front.get("noise_ratio", np.nan)))
        out["clip (front porch comparator)"]["flag"].append(
            bool(against_front.get("clipped", False)))
        out["clip (front porch comparator)"]["detail"].append(
            {"front_porch_noise": against_front.get("porch_noise")})

        extreme = float(mean[tip].min())
        span = float(np.ptp(mean[tip])) or 1.0
        flat = np.zeros(mean.size, dtype=bool)
        flat[tip] = mean[tip] <= extreme + tolerance * span
        depth = clipping.extrapolate_through_clip(mean[wide], model[wide], flat[wide])
        out["clip depth"]["amount"].append(float(depth.get("depth", 0.0)))
        out["clip depth"]["flag"].append(bool(against_front.get("clipped", False)))
        out["clip depth"]["detail"].append({"stopped_at": depth.get("stopped_at"),
                                            "would_have_reached":
                                            depth.get("would_have_reached"),
                                            "masked": int(flat.sum())})

        tips = rows[:, tip].mean(axis=1)
        porches = rows[:, porch].mean(axis=1)
        follow = clipping.clip_offset_from_porch(tips, porches)
        out["tip follows porch"]["amount"].append(float(follow.get("slope", np.nan)))
        out["tip follows porch"]["flag"].append(bool(follow.get("clipped", False)))
        out["tip follows porch"]["detail"].append(
            {"correlation": follow.get("correlation"),
             "offset_ire": follow.get("offset_ire")})

        anchors = standard_levels.affine_to_standard(float(tips.mean()),
                                                     float(porches.mean()))
        out["level"]["amount"].append(float(porches.mean())
                                      - standard_levels.BLANKING_IRE)
        out["level"]["flag"].append(bool(
            abs(float(porches.mean()) - standard_levels.BLANKING_IRE)
            > standard_levels.IRE_TOLERANCE))
        out["level"]["detail"].append({"gain": anchors["gain"],
                                       "offset": anchors["offset"]})

        ring = sg.relaxation_or_ringing(mean[tail], t[tail] - series["rise_time_us"],
                                        decays_us=decays,
                                        frequencies_hz=frequencies)
        got = ring.get("ringing", {}) if ring.get("decided") else {}
        prefers = bool(ring.get("prefers_ringing", False))
        out["ring"]["amount"].append(float(got.get("amplitude", 0.0)))
        out["ring"]["flag"].append(prefers)
        out["ring"]["detail"].append({"frequency_hz": got.get("frequency_hz"),
                                      "tau_us": got.get("tau_us"),
                                      "improvement": ring.get("improvement")})
        out["ring frequency"]["amount"].append(float(got.get("frequency_hz", np.nan)))
        out["ring frequency"]["flag"].append(prefers)
        out["ring frequency"]["detail"].append({})
        out["ring decay"]["amount"].append(float(got.get("tau_us", np.nan)) * 1e-6)
        out["ring decay"]["flag"].append(prefers)
        out["ring decay"]["detail"].append({})

        try:
            crossings = standard_levels.half_amplitude_crossings(mean[wide], rate)
            width = float(crossings["width_s"]) * 1e6
        except ValueError:
            width = float("nan")
        out["timing"]["amount"].append(spec_width / width - 1.0
                                       if width > 0 else float("nan"))
        out["timing"]["flag"].append(bool(
            np.isfinite(width)
            and abs(width - spec_width) > sg.EQUALIZING_PULSE_TOLERANCE_US["525"]))
        out["timing"]["detail"].append({"width_us": width})

        for name, region in (("distribution (tip)", tip),
                             ("distribution (tail)", tail)):
            about = (rows[:, region] - mean[region][None, :]).ravel()
            verdict = dist.classify(about)
            out[name]["amount"].append(float(verdict.get("excess_kurtosis", np.nan)))
            out[name]["flag"].append(str(verdict.get("distribution")) != "gaussian")
            out[name]["detail"].append(
                {"distribution": verdict.get("distribution"),
                 "standard_error": verdict.get("standard_error")})

        isolated = sg.front_porch_residual(mean[front], rate,
                                           float(mean[front][0]),
                                           float(porches.mean()))
        out["front porch"]["amount"].append(float(np.std(isolated["residual"])))
        out["front porch"]["flag"].append(bool(isolated.get("explained", 0.0) > 0.5))
        out["front porch"]["detail"].append({"explained": isolated.get("explained")})
    return {name: {"amount": np.array(v["amount"], dtype=np.float64),
                   "flag": np.array(v["flag"], dtype=bool),
                   "detail": v["detail"]} for name, v in out.items()}


# --------------------------------------------------------------------------
# Reading the chain: the tesseract
# --------------------------------------------------------------------------


def read_cube(target: Dict[str, object], shape=None,
              subset: Sequence[str] = ("head",),
              chain: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    """The tesseract's readers, unchanged: `walsh`, `identified`,
    `unreached`, `trace`, and `hypercomplex.causality` and `deconvolve` with
    the export's own sample rate.

    With `shape` (the per-vertex change per unit amplitude) each contrast's
    value is projected onto the contrast that `shape` should occupy, in the
    contrast's own noise weighting, so the reader on `subset` carries the
    recovered amplitude and every other contrast its cross-talk; the real
    departure `deconvolve` would subtract is projected the same way at the
    vertices, and `causality` reports the injected contrast's delay, excess
    phase, all-pass remainder and minimum-phase share.
    """
    cube = target["cube"]
    contrasts = tesseract.walsh(cube)
    verdicts = tesseract.identified(contrasts)
    left = tesseract.unreached(cube)
    subset = tuple(sorted(subset, key=cube.axes.index))
    name_of = ":".join(subset)
    readers: Dict[str, float] = {
        "unreached: residual over floor": float(left["residual_over_floor"]),
        "unreached: identified": float(left["identified_total"]),
        "unreached: top order z": float(left["top_order_z"]),
    }
    detail: Dict[str, object] = {}
    rate = target.get("sample_rate_hz")
    causal = hypercomplex.causality(cube, rate)
    if name_of in causal:
        for quantity in ("delay_s", "excess_rms", "allpass_rms",
                         "minimum_phase_share"):
            readers[f"causality[{name_of}]: {quantity}"] = float(
                causal[name_of][quantity])
    if shape is not None:
        # the component on the subset's contrast, per unit amplitude: the
        # fold weights each vertex by its sign over 2^n, so a component with
        # the subset's own sign pattern arrives with weight one
        expected = tesseract.walsh(tesseract.Cube(
            cube.axes, shape, cube.variance, cube.bins))
        reference = expected[subset]["value"]
        for name, entry in contrasts.items():
            label = ":".join(name) if name else "mean"
            readers[f"contrast[{label}]: carries"] = weighted_projection(
                reference, entry["value"], np.sqrt(entry["variance"]))
        readers[f"contrast[{name_of}]: z"] = float(verdicts[subset]["z"])
        path = tesseract.trace(cube, chain)
        detail["stage of the injected contrast"] = next(
            (p["stage"] for p in path["path"] if tuple(p["axes"]) == subset), None)
        if rate is not None:
            traversal = hypercomplex.deconvolve(cube, rate)
            readers["deconvolve: departure carries"] = weighted_projection(
                shape, traversal["departure"], target["sigma"])
            readers["deconvolve: residual over floor"] = float(
                traversal["residual_over_floor"])
    for name, verdict in verdicts.items():
        label = ":".join(name) if name else "mean"
        readers[f"identified[{label}]"] = float(verdict["identified"])
    return {"readers": readers, "detail": detail}


# --------------------------------------------------------------------------
# recover: the estimators, unchanged, read back
# --------------------------------------------------------------------------


def perturb(background: Dict[str, object], rng: np.random.Generator
            ) -> Dict[str, object]:
    """Another realisation of the same measurement, at the measured SE.

    On the grid the export's `se` is the standard error of the complex
    estimate with both parts together (`residual_floor.log_domain`), so each
    part receives `se / root 2`; the half-splits receive their own
    `se_half` independently and the full transfer their mean, which has
    variance `se` squared as it should. The draw is ADDED to the measured
    transfer, which already carries one realisation of the same noise, so
    every reader's scatter over draws is its scatter under noise of the
    measured size, and the paired difference removes both. A series is not
    perturbed - its groups are the realisations. On the cube each vertex
    receives its own variance, both parts together.
    """
    out = dict(background)
    if background["representation"] == "grid":
        count = background["H"].size

        def draw(scale):
            return scale / np.sqrt(2.0) * (rng.standard_normal(count)
                                           + 1j * rng.standard_normal(count))

        held = background.get("held_out")
        if held is not None:
            first = draw(held["se_half"])
            second = draw(held["se_half"])
            out["held_out"] = {"H_fit": held["H_fit"] + first,
                               "H_judge": held["H_judge"] + second,
                               "se_half": held["se_half"]}
            noise = 0.5 * (first + second)
        else:
            noise = draw(background["se"])
        out["H"] = background["H"] + noise
        out["log"] = residual_floor.log_domain(out["H"], background["se"])["log"]
        return out
    if background["representation"] == "cube":
        cube = background["cube"]
        noise = np.sqrt(cube.variance / 2.0) * (
            rng.standard_normal(cube.values.shape)
            + 1j * rng.standard_normal(cube.values.shape))
        out["cube"] = tesseract.Cube(cube.axes, cube.values + noise,
                                     cube.variance, cube.bins, cube.channels)
        return out
    return out


def recover(target: Dict[str, object],
            keys: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
            chain: Optional[Dict[str, int]] = None,
            ring_scan: Tuple[int, int] = (30, 45),
            shape=None, subset: Sequence[str] = ("head",)) -> Dict[str, object]:
    """RUN THE EXISTING ESTIMATORS UNCHANGED AND READ BACK WHAT EACH REPORTS.

    On a grid: `read_grid` under each supplied key (the default key and the
    entry-wise admitted key when none are given). On a series: the readers
    of `read_series`, pooled over the groups. On a cube: `read_cube`.
    Returns the readers as name -> number, with the estimators' verdicts in
    `detail` and, for a series, the per-group amounts and flags.
    """
    representation = target["representation"]
    if representation == "series":
        per_group = read_series(target, ring_scan)
        readers = {name: float(np.nanmean(v["amount"])) for name, v in per_group.items()}
        detail = {name: {"flag": float(np.mean(v["flag"])),
                         "detail": v["detail"][0]} for name, v in per_group.items()}
        return {"representation": "series", "readers": readers,
                "detail": detail, "per_group": per_group}
    if representation == "cube":
        got = read_cube(target, shape, subset, chain)
        return {"representation": "cube", **got}
    chain = chain if chain is not None else inf.full_chain()
    keys = keys if keys is not None else grid_keys(target)
    got = read_grid(target, keys, chain, shape)
    return {"representation": "grid", **got}


# --------------------------------------------------------------------------
# The sweep and the sensitivity curve
# --------------------------------------------------------------------------


def _thresholds(amplitudes: np.ndarray, response: np.ndarray, error: np.ndarray,
                noise: np.ndarray, expected: Optional[np.ndarray] = None,
                absolute: bool = False) -> Dict[str, object]:
    """The thresholds and the slope from a sweep's arrays, in ascending
    order of amplitude (see the module docstring for the definitions).

    `noise` is the reader's scatter on the background, `error` the standard
    error of the paired response. For an ABSOLUTE reader the response is
    the reading itself less its baseline and `expected` the parameter less
    the baseline, so agreement means the reading landed on the parameter;
    it has no detect threshold, and `recover` is the first amplitude from
    which every larger one agrees within twice the paired error.
    """
    a = np.asarray(amplitudes, dtype=np.float64)
    r = np.asarray(response, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    if a.size > 1 and np.any(np.diff(a) < 0):
        raise ValueError("the sweep must be in ascending amplitude")
    usable = np.isfinite(r) & np.isfinite(e) & np.isfinite(n)
    detect = recover = tracks_to = None
    x = np.asarray(expected, dtype=np.float64) if expected is not None else None
    if absolute:
        if x is not None:
            within = usable & np.isfinite(x) & (np.abs(r - x) <= DETECT_SIGMA * e)
            for i in range(a.size):
                if within[i] and np.all(within[i:][usable[i:]]):
                    recover = int(i)
                    break
            tracks_to = int(np.where(within)[0][-1]) if within.any() else None
        return {"detect": None, "recover": recover, "tracks_to": tracks_to,
                "slope": float("nan")}
    clear = usable & (np.abs(r) > DETECT_SIGMA * n)
    for i in range(a.size):
        if clear[i] and np.all(clear[i:][usable[i:]]):
            detect = int(i)
            break
    if x is not None:
        within = clear & np.isfinite(x) & (
            np.abs(r - x) <= DETECT_SIGMA * np.maximum(n, e))
        if detect is not None:
            for i in range(detect, a.size):
                if within[i]:
                    recover = int(i)
                    break
            if recover is not None:
                tracks_to = recover
                for i in range(recover, a.size):
                    if within[i]:
                        tracks_to = int(i)
                    elif usable[i]:
                        break
    else:
        x = a
    top = usable & (a >= (a[usable].max() / 4.0 if usable.any() else np.inf))
    slope = (float(np.sum(r[top] * x[top]) / np.sum(x[top] ** 2))
             if top.any() and np.sum(x[top] ** 2) > 0 else float("nan"))
    return {"detect": detect, "recover": recover, "tracks_to": tracks_to,
            "slope": slope}


def expected_curve(reader: str, kind: str, natural: np.ndarray,
                   background: Dict[str, object], baseline: Dict[str, float],
                   parameters: Dict[str, object]) -> Optional[np.ndarray]:
    """The identity line in the reader's own units, where the reader speaks
    them; None where it does not and no identity exists.

    Stage and correction readers are projections per unit amplitude, so the
    line is the amplitude itself. An echo entry reader is already a
    reflection coefficient when its delay is the injected one. The level
    entry reads nepers. The delay entry's response to a speed error `a` of
    a transfer whose own delay coefficient is `c0` is `-a c0`, because
    `H(f / (1 + a))` scales the phase slope down. On the cube the contrast
    and the deconvolved departure carry the amplitude. In the series: the
    clip depth, the level and the ring amplitude are each the injected
    quantity; the timing reader's width shrinks by `(1 + a)` so its reading
    `spec / width - 1` moves by `a (1 + b0)` from its baseline `b0`; the
    ring's frequency and decay should land on the planted parameters.
    """
    representation = background["representation"]
    if representation == "grid":
        if reader.startswith("stage[") or reader.startswith("correction["):
            return natural
        if reader.startswith("entry[") and kind == "echo":
            grid = background["frequency_hz"]
            delay = float(parameters.get("delay_s", default_delay_s(grid)))
            name = reader.split("]: ", 1)[1]
            if name.startswith("echo at"):
                entry_delay = float(name.split("at")[1].split("us")[0]) * 1e-6
                return natural if abs(entry_delay - delay) < 0.005 * delay else None
            return None
        if reader.startswith("entry[") and reader.endswith(": level"):
            return natural if kind == "level step" else None
        if reader.startswith("entry[") and reader.endswith(": delay"):
            return (-natural * baseline.get(reader, 0.0)
                    if kind == "frequency scaling" else None)
        return None
    if representation == "cube":
        if reader.startswith("contrast[") and reader.endswith(": carries"):
            return natural
        if reader == "deconvolve: departure carries":
            return natural
        return None
    if reader == "timing" and kind == "frequency scaling":
        return natural * (1.0 + baseline.get(reader, 0.0))
    if reader == "ring frequency" and kind == "ring":
        return np.full(natural.shape, float(parameters.get(
            "ring_hz", MEASURED_RING_HZ)) - baseline.get(reader, 0.0))
    if reader == "ring decay" and kind == "ring":
        return np.full(natural.shape, float(parameters.get(
            "decay_s", MEASURED_RING_DECAY_S)) - baseline.get(reader, 0.0))
    identity = {"clip": "clip depth", "level step": "level", "ring": "ring"}.get(kind)
    return natural if reader == identity else None


def default_levels() -> np.ndarray:
    """A sweep from a sixty-fourth of the noise to sixty-four times it, in
    factors of two: thirteen levels spanning the four decades a threshold
    can sit in. The low end is where a reader matched to the shape sits: an
    injection at one SE per bin is a matched-filter signal-to-noise of the
    root of the bin count, 31 on the 944-bin exports, so such a reader
    detects well below one SE."""
    return np.geomspace(1.0 / 64.0, 64.0, 13)


def sweep(kind: str, amplitudes_se: Optional[Sequence[float]] = None,
          background: Optional[Dict[str, object]] = None,
          keys: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
          draws: int = 6, seed: int = 0, chain: Optional[Dict[str, int]] = None,
          ring_scan: Tuple[int, int] = (30, 45), **parameters) -> Dict[str, object]:
    """SWEEP THE LEVEL, FIND THE RECOVERY THRESHOLD, PRODUCE THE SENSITIVITY
    CURVE.

    `amplitudes_se` are in units of the measured noise (see the module
    docstring). For each, the kind is planted at the corresponding natural
    amplitude and every reader is run; on a grid or a cube each amplitude is
    read over `draws` realisations of the measured SE, paired with the
    background's reading on the SAME realisation, so the response is the
    mean paired difference, its error the standard error of that
    difference, and the reader's noise its scatter over the background's
    realisations; in a series the disjoint line groups are the
    realisations. Per reader the sweep records the response, the error, the
    noise, the estimator's flag, the identity line where the reader speaks
    the kind's units, and the thresholds and the slope from `_thresholds`,
    each threshold given both in SE units and in the natural amplitude.
    Levels a kind cannot reach (a clip past the porch) are dropped.
    """
    if background is None:
        raise ValueError("a background is required")
    levels = (np.asarray(amplitudes_se, dtype=np.float64).ravel()
              if amplitudes_se is not None else default_levels())
    natural = np.array([natural_amplitude(kind, s, background, **parameters)
                        for s in levels])
    reachable = np.isfinite(natural)
    levels, natural = levels[reachable], natural[reachable]
    representation = background["representation"]
    rng = np.random.default_rng(seed)
    subset = tuple(parameters.get("subset", ("head",)))
    shape = unit_shape(kind, background, **parameters)
    curves: Dict[str, Dict[str, object]] = {}
    planted: List[Dict[str, object]] = []

    if representation in ("grid", "cube"):
        chain = chain if chain is not None else inf.full_chain()
        if representation == "grid" and keys is None:
            keys = grid_keys(background)
        realisations = [perturb(background, rng) for _ in range(int(draws))]
        baseline = [recover(r, keys, chain, ring_scan, shape, subset)["readers"]
                    for r in realisations]
        names = list(baseline[0])
        base_mean = {n: float(np.mean([b[n] for b in baseline])) for n in names}
        noise = {n: (float(np.std([b[n] for b in baseline], ddof=1))
                     if len(baseline) > 1 else float("nan")) for n in names}
        table = {n: {"response": [], "error": [], "flag": []} for n in names}
        for a in natural:
            injected = [inject(kind, a, r, **parameters) for r in realisations]
            planted.append(injected[0]["injected"])
            readings = [recover(i, keys, chain, ring_scan, shape, subset)["readers"]
                        for i in injected]
            for n in names:
                paired = np.array([r[n] - b[n] for r, b in zip(readings, baseline)],
                                  dtype=np.float64)
                table[n]["response"].append(float(np.nanmean(paired)))
                table[n]["error"].append(
                    float(np.nanstd(paired, ddof=1) / np.sqrt(paired.size))
                    if paired.size > 1 else float("nan"))
                table[n]["flag"].append(bool(np.nanmean([r[n] for r in readings]) > 0.5))
        count = int(draws)
        flag_readers = ("identified", ": at floor", ": white", "identified[")
    else:
        base = read_series(background, ring_scan)
        names = list(base)
        table = {n: {"response": [], "error": [], "flag": []} for n in names}
        for a in natural:
            target = inject(kind, a, background, **parameters)
            planted.append(target["injected"])
            got = read_series(target, ring_scan)
            for n in names:
                paired = got[n]["amount"] - base[n]["amount"]
                table[n]["response"].append(float(np.nanmean(paired)))
                table[n]["error"].append(
                    float(np.nanstd(paired, ddof=1) / np.sqrt(paired.size))
                    if paired.size > 1 else float("nan"))
                table[n]["flag"].append(bool(np.mean(got[n]["flag"]) > 0.5))
        base_mean = {n: float(np.nanmean(base[n]["amount"])) for n in names}
        noise = {n: float(np.nanstd(base[n]["amount"], ddof=1)) for n in names}
        count = int(background["groups"])
        flag_readers = ()

    for n in names:
        response = np.array(table[n]["response"])
        error = np.array(table[n]["error"])
        spread = np.full(response.shape, noise[n])
        expected = expected_curve(n, kind, natural, background, base_mean, parameters)
        is_flag = any(n.startswith(f) or n.endswith(f) or f in n
                      for f in flag_readers) if flag_readers else False
        absolute = n in ABSOLUTE_READERS
        found = _thresholds(natural, response, error, spread,
                            expected=None if is_flag else expected,
                            absolute=absolute)
        flags = np.array(table[n]["flag"])
        turns = None
        if is_flag or representation == "series":
            # the first level from which the flag holds its final state
            final = bool(flags[-1]) if flags.size else False
            for i in range(flags.size):
                if np.all(flags[i:] == final):
                    turns = int(i)
                    break
        curves[n] = {
            "response": response, "error": error, "noise": spread,
            "flag": flags,
            "expected": expected,
            "identity": expected is not None and not is_flag,
            "absolute": absolute,
            "is_flag": is_flag,
            "baseline": base_mean[n],
            "stage": (series_stage(n, chain) if representation == "series"
                      else None),
            "threshold_detect_se": (float(levels[found["detect"]])
                                    if found["detect"] is not None else None),
            "threshold_detect_natural": (float(natural[found["detect"]])
                                         if found["detect"] is not None else None),
            "threshold_recover_se": (float(levels[found["recover"]])
                                     if found["recover"] is not None else None),
            "threshold_recover_natural": (float(natural[found["recover"]])
                                          if found["recover"] is not None else None),
            "tracks_to_se": (float(levels[found["tracks_to"]])
                             if found["tracks_to"] is not None else None),
            "flag_turns_se": (float(levels[turns]) if turns is not None
                              and turns > 0 else None),
            "flag_final": bool(flags[-1]) if flags.size else None,
            "slope": found["slope"],
        }
    return {"kind": kind, "representation": representation,
            "label": background.get("label", ""),
            "amplitudes_se": levels, "amplitudes": natural,
            "planted": planted, "parameters": dict(parameters),
            "realisations": count, "readers": curves}


# --------------------------------------------------------------------------
# Which stage eats what
# --------------------------------------------------------------------------


def which_stage_eats_what(background: Dict[str, object],
                          kinds: Optional[Sequence[str]] = None,
                          amplitude_se: float = 16.0,
                          keys: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                          chain: Optional[Dict[str, int]] = None,
                          draws: int = 4, seed: int = 0,
                          ring_scan: Tuple[int, int] = (30, 45),
                          parameters: Optional[Dict[str, Dict[str, object]]] = None
                          ) -> Dict[str, object]:
    """THE MATRIX OF KINDS AGAINST STAGES, at one amplitude well above the
    noise (`amplitude_se` times the measured SE; sixteen by default, where
    every grid reader is past its detect threshold on the three exports).

    On a grid, per kind and per key: the SHARE of the injected amplitude
    each link's fitted entries return, the nuisance's share, the residual's
    share (they sum to one, up to the log domain's second order), the join
    whose prior gains most on the residual and the family `search_families`
    names there, and the base-key direction `outlier_response` says carries
    the residual's outliers. In a series: each reader's response in units
    of its own noise, and its flag, at that amplitude - a cross-talk matrix
    of kinds against readers, each reader labelled with the link its
    estimator is declared in. On a cube: the share the injected contrast
    returns, the largest share any other contrast takes, the stage
    `tesseract.trace` charges the contrast to, and how `causality` and
    `deconvolve` read it.

    Shares are responses (injected less background on the same realisation)
    divided by the injected amplitude, averaged over `draws` realisations.
    """
    parameters = parameters or {}
    representation = background["representation"]
    chain = chain if chain is not None else inf.full_chain()
    rng = np.random.default_rng(seed)
    rows: Dict[str, Dict[str, object]] = {}

    if representation == "series":
        kinds = list(kinds) if kinds is not None else list(SERIES_KINDS)
        base = read_series(background, ring_scan)
        for kind in kinds:
            extra = parameters.get(kind, {})
            a = natural_amplitude(kind, amplitude_se, background, **extra)
            if not np.isfinite(a):
                continue
            target = inject(kind, a, background, **extra)
            got = read_series(target, ring_scan)
            rows[kind] = {"amplitude": a, "amplitude_se": float(amplitude_se),
                          "planted": target["injected"], "readers": {}}
            for reader in base:
                paired = got[reader]["amount"] - base[reader]["amount"]
                error = (float(np.nanstd(paired, ddof=1) / np.sqrt(paired.size))
                         if paired.size > 1 else float("nan"))
                noise = float(np.nanstd(base[reader]["amount"], ddof=1))
                response = float(np.nanmean(paired))
                rows[kind]["readers"][reader] = {
                    "response": response, "error": error, "noise": noise,
                    "z": response / noise if noise > 0 else float("nan"),
                    "flag": float(np.mean(got[reader]["flag"])),
                    "baseline_flag": float(np.mean(base[reader]["flag"])),
                    "stage": series_stage(reader, chain),
                }
        return {"representation": "series", "label": background.get("label", ""),
                "kinds": list(rows), "readers": list(base),
                "amplitude_se": float(amplitude_se), "rows": rows}

    if representation == "grid" and keys is None:
        keys = grid_keys(background)
    kinds = list(kinds) if kinds is not None else list(
        GRID_KINDS if representation == "grid" else CUBE_KINDS)
    realisations = [perturb(background, rng) for _ in range(int(draws))]
    for kind in kinds:
        extra = parameters.get(kind, {})
        subset = tuple(extra.get("subset", ("head",)))
        a = natural_amplitude(kind, amplitude_se, background, **extra)
        shape = unit_shape(kind, background, **extra)
        baseline = [recover(r, keys, chain, ring_scan, shape, subset)
                    for r in realisations]
        injected = [inject(kind, a, r, **extra) for r in realisations]
        readings = [recover(i, keys, chain, ring_scan, shape, subset)
                    for i in injected]
        names = list(baseline[0]["readers"])
        response = {n: float(np.mean([r["readers"][n] - b["readers"][n]
                                      for r, b in zip(readings, baseline)]))
                    for n in names}
        row: Dict[str, object] = {"amplitude": a, "amplitude_se": float(amplitude_se),
                                  "planted": injected[0]["injected"]}
        if representation == "grid":
            for label in keys:
                shares = {stage: response[f"stage[{label}]: {stage}"] / a
                          for stage in stage_names()}
                joins = {n.split("]: ", 1)[1]: response[n] for n in names
                         if n.startswith(f"join[{label}]: ")
                         and not n.endswith("identified")}
                best = max(joins, key=joins.get) if joins else None
                row[label] = {
                    "shares": shares,
                    "whole": response[f"stage[{label}]: whole"] / a,
                    "correction carries": response[f"correction[{label}]: carries"] / a,
                    "claimed_at": best,
                    "claimed_gain": joins[best] if best else 0.0,
                    "family_winner": readings[0]["detail"].get(f"join[{label}]: winner"),
                    "family_winner_before": baseline[0]["detail"].get(
                        f"join[{label}]: winner"),
                    "family_identified": bool(
                        np.mean([r["readers"][f"join[{label}]: identified"]
                                 for r in readings]) > 0.5),
                    "carried_by": readings[0]["detail"].get(
                        f"outliers[{label}]: carried by"),
                    "chi_square_rise": response[f"floor[{label}]: in-sample reduced chi-square"],
                    "held_out_rise": response.get(
                        f"floor[{label}]: held-out reduced chi-square"),
                    "agreement_change": response.get(f"floor[{label}]: agreement"),
                }
        else:
            own = f"contrast[{':'.join(subset)}]: carries"
            others = {n: response[n] / a for n in names
                      if n.startswith("contrast[") and n != own}
            worst = max(others, key=lambda n: abs(others[n])) if others else None
            prefix = f"causality[{':'.join(subset)}]: "
            row.update({
                "subset": subset,
                "recovered": response[own] / a,
                "worst_cross_talk": worst,
                "worst_cross_talk_share": others[worst] if worst else 0.0,
                "stage": readings[0]["detail"].get("stage of the injected contrast"),
                "residual_over_floor_change":
                    response["unreached: residual over floor"],
                "departure carries": (response.get("deconvolve: departure carries", np.nan)
                                      / a),
                "causality": {n[len(prefix):]: {
                    "after": float(np.mean([r["readers"][n] for r in readings])),
                    "change": response[n]} for n in names if n.startswith(prefix)},
            })
        rows[kind] = row
    return {"representation": representation, "label": background.get("label", ""),
            "kinds": kinds, "stages": stage_names(),
            "keys": list(keys) if keys else [],
            "amplitude_se": float(amplitude_se), "rows": rows}


# Where each kind BELONGS on the grid: the stage whose entries should carry
# it. An echo is the transmission path's; a level step is the nuisance's; a
# speed error moves the delay nuisance; a ring and a phase rotation have no
# entry of their own in the key, so the honest home of both is the residual.
BELONGS: Dict[str, str] = {
    "echo": "the transmission path",
    "level step": NUISANCE_STAGE,
    "frequency scaling": NUISANCE_STAGE,
    "ring": RESIDUAL_STAGE,
    "phase rotation": RESIDUAL_STAGE,
}


def worst_stage(grid_matrices: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """THE STAGE THAT EATS THE MOST OF WHAT IS NOT ITS OWN, and why.

    Over every grid matrix, key and kind: the largest share returned by a
    stage other than the one the kind belongs to (`BELONGS`). The reason is
    read from the same row: where another stage returns a share of the
    opposite sign, the two are fitting one direction between them - the
    entries are collinear on the injected shape under the judge's weights -
    and the sum of the absolute shares says how much larger than the
    component the two claims are.
    """
    worst = None
    for matrix in grid_matrices:
        for label in matrix["keys"]:
            for kind, row in matrix["rows"].items():
                home = BELONGS.get(kind.split(" (")[0])
                for stage, share in row[label]["shares"].items():
                    if stage in (home, RESIDUAL_STAGE):
                        continue
                    if worst is None or abs(share) > abs(worst["share"]):
                        others = {s: v for s, v in row[label]["shares"].items()
                                  if s != stage}
                        partner = (min(others, key=others.get) if share > 0
                                   else max(others, key=others.get))
                        worst = {
                            "share": float(share), "stage": stage, "kind": kind,
                            "key": label, "background": matrix["label"],
                            "home": home,
                            "partner": partner,
                            "partner_share": float(others[partner]),
                            "gross": float(sum(abs(v) for v in
                                               row[label]["shares"].values())),
                            "correction_carries": float(
                                row[label]["correction carries"]),
                        }
    if worst is None:
        return None
    opposite = worst["partner_share"] * worst["share"] < 0
    worst["why"] = (
        f"'{worst['stage']}' returns {worst['share']:+.3f} of an injected "
        f"{worst['kind']} on {worst['background']} under the '{worst['key']}' "
        f"key, whose home is '{worst['home']}'"
        + (f"; '{worst['partner']}' returns {worst['partner_share']:+.3f} at "
           "the same time, so the two are fitting one direction between them "
           "- their entries are collinear on this shape under the judge's "
           f"weights - and the gross claim is {worst['gross']:.2f} times the "
           "component" if opposite else
           f"; the gross claim over every stage is {worst['gross']:.2f} times "
           "the component")
        + f". The collapsed correction would carry {worst['correction_carries']:+.3f} "
        "of it.")
    return worst
