#!/usr/bin/env python3
"""Run the content-independence check on a capture's sync-derived series.

Ethan: *"Content-independence check: compute variance on blanking lines and
test-pattern segments, compare against full-field. Mismatch quantifies
picture-statistics bleed-through."*

This is the runnable form of `vhsdecode/models/content_independence.py`. It
extracts the per-line series itself from a short window of RF, using the
decoder's OWN demodulator (`VHSRFDecode.demodblock`) so that no carrier
frequency, deviation or filter response is copied here; finds each field's
vertical interval by classifying every sync pulse against the three widths
the specification states (equalizing 2.3 us, line 4.7 us, field sync
27.1 us, `sync_geometry`) and then LOCKING the interval's start to the
specification's own pulse train (`sync_geometry.pulse_train`, correlated
against the demodulated luma), so the line numbers are the specification's
counted from that anchor; measures each series in windows built from the
sysparams timings; and prints one table per series per capture.

The series, per line:

    sync tip         median of the demodulated luma over the tip plateau
    front porch      median over the front porch - the KNOWN bleed-through,
                     kept as the check's positive control
    back porch       median after the burst, before active video
    sync width       50% fall crossing to 50% rise crossing
    burst amplitude  the colour-under burst's analytic magnitude
    burst phase      its phase referenced to this line's sync fall, folded
                     by the format's quarter-turn-per-line rotation
    noise floor      RMS about the tip plateau's median
    envelope (tip)   the RF envelope over the tip plateau, the envelope-
                     derived response at the tip's carrier frequency

and, from a `*_hs_impulse.npz` export (`--npz`), the per-line ln|H| at the
export's reference frequencies (the envelope-derived response the ringing
lane measures per line from the sync edge) and the export's own per-line
levels and sync width (`levels_h*`, `width_h*`).

    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/content_independence.py \\
        --control /testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \\
        --capture /testdata/test_patterns/vhs/playback/zaroff-bounce1-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \\
        --rate 50e6
    ... --capture /testdata/home.flac --rate 40e6 --start 60 --seconds 3
    ... --npz /tmp/claude-1000/-workspaces-vhs-decode/shared/hb75_hs_impulse.npz

The zaroff captures are 0.48 s long (24 000 512 samples at 50 MSps), so
`--start 0` and the whole file are the defaults; the two-hour captures are
read with ffmpeg's `-ss`, which was verified sample-exact against a
sequential read (lag 0, no difference) and takes 0.07 s an hour in.

MEASURED 2026-09-05 - the tables are reproduced in docs/DECK_LOG.md's
companion section and in the final report of that date. In one line each:
the front porch is bleed-through on every capture with content ahead of it
(the positive control holds); the sync tip, the back porch after the burst
and the sync width react to the preceding line's level by an order of
magnitude less; the burst amplitude and phase are reacquired after the
vertical interval, so on them the blanking lines are not a floor; and on
the home capture the burst phase and the envelope climb along the field on
the static control too, which reads as the time base, not the picture.
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from vhsdecode.models import content_independence as ci      # noqa: E402
from vhsdecode.models import sync_geometry                    # noqa: E402

# A settled span begins this many specified edge times after a transition.
# The specification's edge is 10 to 90 per cent; three of them is where a
# first-order settling has reached about 99 per cent. A structural choice,
# not a measurement, and the same one the fold instruments used.
SETTLING_EDGES = 3.0

# The active-video windows are trimmed by one microsecond at each end so a
# line's own blanking transitions do not enter its content level. Labelled
# as an assumption: it is a guard, not a specified figure.
ACTIVE_GUARD_US = 1.0

# How finely the interval template is sampled when it is correlated against
# the luma: four samples across the narrowest pulse in it (the equalizing
# pulse). The anchor only has to place the interval's start within a
# quarter of a line, and the lowpass luma it is matched against is band-
# limited far below this rate, so nothing is lost by the decimation.
TEMPLATE_SAMPLES_PER_EQUALIZING_PULSE = 4.0

# The colour-under phase is rotated a quarter turn per line by the format
# (VHS's phase-shift crosstalk cancellation), and the NTSC burst itself
# alternates a half turn per line (227.5 cycles per line); the largest
# period both fold into is a quarter turn. The decoder carries the same
# rotation in its chroma stage; it is stated here in degrees because the
# fold is a property of the format and not of any recording.
PHASE_FOLD_DEG = 90.0

# The frequencies at which the ringing lane's export samples its per-line
# response - REF_MHZ in the producer (shared/hs_impulse.py). The export
# does not carry them, so they are stated here and can be overridden.
EXPORT_REFERENCE_MHZ = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


# --------------------------------------------------------------------------
# reading RF
# --------------------------------------------------------------------------

def read_rf(path: str, start_sample: int, count: int) -> np.ndarray:
    """Samples as int16. soundfile where the file seeks; ffmpeg where it
    does not. The 8-bit zaroff captures and the streamed two-hour captures
    (no frame count in their STREAMINFO) both refuse libsndfile's seek;
    ffmpeg's `-ss` counts the header's nominal rate as its seconds and was
    verified sample-exact against a sequential read (2026-09-05: best lag
    0, maximum difference 0 over 4000 samples)."""
    try:
        with sf.SoundFile(path) as handle:
            handle.seek(int(start_sample))
            data = handle.read(int(count), dtype="int16", always_2d=False)
        return np.asarray(data).ravel()
    except (sf.LibsndfileError, RuntimeError):
        pass
    header_rate = sf.info(path).samplerate
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-ss", repr(start_sample / header_rate), "-i", path,
               "-t", repr(count / header_rate),
               "-c:a", "pcm_s16le", "-f", "s16le", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<i2")[: int(count)]


# --------------------------------------------------------------------------
# demodulating with the decoder's own front end
# --------------------------------------------------------------------------

def demodulate(samples: np.ndarray, rate_hz: float, system: str,
               tape_format: str = "VHS") -> Dict[str, object]:
    """The decoder's demodulator, block by block with its own overlap."""
    import lddecode.core as ldd
    ldd.logger = logging.getLogger("content_independence")
    ldd.logger.setLevel(logging.ERROR)
    import vhsdecode.process as process

    decoder = process.VHSRFDecode(inputfreq=rate_hz / 1e6, system=system,
                                  tape_format=tape_format)
    blocklen = int(decoder.blocklen)
    cut_start = int(decoder.blockcut)
    cut_end = int(decoder.blockcut_end)
    stride = blocklen - cut_start - cut_end
    ire0 = float(decoder.SysParams["ire0"])
    hz_ire = float(decoder.SysParams["hz_ire"])

    luma, luma_lowpass, chroma, envelope = [], [], [], []
    data = np.asarray(samples, dtype=np.float64)
    for start in range(0, len(data) - blocklen + 1, stride):
        video = decoder.demodblock(data=data[start: start + blocklen],
                                   cut=True)["video"]
        luma.append((video["demod"] - ire0) / hz_ire)
        luma_lowpass.append((video["demod_05"] - ire0) / hz_ire)
        chroma.append(video["demod_burst"])
        envelope.append(video["envelope"])
    sysparams = dict(decoder.SysParams)
    sysparams["system"] = system
    return {
        "luma_ire": np.concatenate(luma),
        "luma_lowpass_ire": np.concatenate(luma_lowpass),
        "chroma": np.concatenate(chroma),
        "envelope": np.concatenate(envelope),
        "offset": cut_start,        # output sample k is input sample k+offset
        "rate_hz": float(rate_hz),
        "sysparams": sysparams,
        "colour_under_hz": float(decoder.DecoderParams["color_under_carrier"]),
    }


# --------------------------------------------------------------------------
# sync pulses, classified by the specification's widths
# --------------------------------------------------------------------------

def find_pulses(luma_lowpass_ire: np.ndarray, rate_hz: float,
                sysparams: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Every sync pulse's threshold crossings and width, classified.

    The threshold sits half the specified sync depth above the measured
    tip level; the tip is the low percentile the sync occupies. Widths at
    that crossing are classified at the midpoints between the specified
    equalizing, line and field-sync pulse widths, so nothing about the
    classification is tuned to a recording.
    """
    depth = -float(sysparams["vsync_ire"])
    tip = float(np.percentile(luma_lowpass_ire, 2.0))
    threshold = tip + 0.5 * depth
    below = luma_lowpass_ire < threshold
    edges = np.diff(below.astype(np.int8))
    falls = np.flatnonzero(edges == 1) + 1
    rises = np.flatnonzero(edges == -1) + 1
    if len(rises) and len(falls) and rises[0] < falls[0]:
        rises = rises[1:]
    count = min(len(falls), len(rises))
    falls, rises = falls[:count], rises[:count]
    width_us = (rises - falls) / rate_hz * 1e6
    key = sync_geometry._system(str(sysparams.get("system", "NTSC")))
    equalizing = sync_geometry.EQUALIZING_PULSE_US[key]
    line_sync = sync_geometry.LINE_SYNC_US[key]
    field_sync = sync_geometry.FIELD_SYNC_PULSE_US[key]
    kind = np.full(count, "other", dtype=object)
    kind[width_us < 0.5 * (equalizing + line_sync)] = "equalizing"
    kind[(width_us >= 0.5 * (equalizing + line_sync))
         & (width_us < 0.5 * (line_sync + field_sync))] = "line"
    kind[width_us >= 0.5 * (line_sync + field_sync)] = "field sync"
    # a crossing narrower than half an equalizing pulse is noise
    kind[width_us < 0.5 * equalizing] = "noise"
    return {"fall": falls, "rise": rises, "width_us": width_us, "kind": kind,
            "threshold_ire": threshold, "tip_ire": tip}


def interval_template(rate_hz: float, sysparams: Dict[str, object]
                      ) -> Dict[str, object]:
    """The specification's pulse train, decimated for the correlation."""
    key = sync_geometry._system(str(sysparams.get("system", "NTSC")))
    equalizing_us = sync_geometry.EQUALIZING_PULSE_US[key]
    decimation = max(1, int(np.floor(
        rate_hz * equalizing_us * 1e-6 / TEMPLATE_SAMPLES_PER_EQUALIZING_PULSE)))
    train = sync_geometry.pulse_train(key, rate_hz / decimation,
                                      depth=float(sysparams["vsync_ire"]),
                                      level=0.0)
    waveform = np.asarray(train["waveform"], dtype=np.float64)
    waveform = waveform - waveform.mean()
    return {"waveform": waveform, "decimation": decimation,
            "norm": float(np.sqrt(waveform @ waveform)),
            "duration_us": float(train["duration_us"])}


def lock_interval(luma_lowpass_ire: np.ndarray, candidate: int,
                  template: Dict[str, object], line_samples: float
                  ) -> Dict[str, float]:
    """Correlate the specification's train against the luma within a
    QUARTER line of the candidate start and return the best-matching start
    with its normalised correlation (1 is a perfect match of shape).

    A quarter line, not half: the equalizing sequences repeat at half-line
    spacing, so a half-line lag is an alias where twelve of the eighteen
    pulses still match. Measured on the home capture before this was
    narrowed, 68 of 277 fields locked onto that alias at a correlation of
    0.54 against 0.96 for a true lock, which would have renumbered one
    parity's lines by one. The candidate from the classified pulses is
    never a quarter line out when it is right at all; when it is wrong the
    lock's correlation says so and the field is excluded.
    """
    decimation = int(template["decimation"])
    waveform = template["waveform"]
    length = len(waveform)
    reach = int(np.ceil(0.25 * line_samples / decimation))
    best_lag, best_corr = 0, np.nan
    for lag in range(-reach, reach + 1):
        low = candidate + lag * decimation
        high = low + length * decimation
        if low < 0 or high > len(luma_lowpass_ire):
            continue
        segment = luma_lowpass_ire[low:high:decimation][:length]
        if len(segment) != length:
            continue
        centred = segment - segment.mean()
        norm = float(np.sqrt(centred @ centred))
        if norm <= 0:
            continue
        corr = float(centred @ waveform) / (norm * template["norm"])
        if not np.isfinite(best_corr) or corr > best_corr:
            best_lag, best_corr = lag, corr
    return {"start": candidate + best_lag * decimation,
            "correlation": best_corr, "lag_samples": best_lag * decimation}


def find_fields(pulses: Dict[str, np.ndarray], rate_hz: float,
                sysparams: Dict[str, object],
                luma_lowpass_ire: Optional[np.ndarray] = None
                ) -> List[Dict[str, object]]:
    """Each field's interval start. The candidate is the first equalizing
    pulse of the run that precedes the field-sync block (or, when the run
    is incomplete, the specification's three lines ahead of the block);
    the start is then locked to the specification's pulse train."""
    line_samples = float(sysparams["line_period"]) * 1e-6 * rate_hz
    key = sync_geometry._system(str(sysparams.get("system", "NTSC")))
    first_lines = sync_geometry.SEQUENCE_LINES[key][0]
    pulses_per_run = int(round(first_lines * sync_geometry.PULSES_PER_LINE))
    template = (interval_template(rate_hz, sysparams)
                if luma_lowpass_ire is not None else None)
    kind = pulses["kind"]
    fall = pulses["fall"]
    fields = []
    index = 0
    while index < len(kind):
        if kind[index] != "field sync":
            index += 1
            continue
        block_start = index
        while index < len(kind) and kind[index] == "field sync":
            index += 1
        run_start = block_start
        while (run_start - 1 >= 0 and kind[run_start - 1] == "equalizing"
               and block_start - run_start < pulses_per_run):
            run_start -= 1
        complete = (block_start - run_start) == pulses_per_run
        if complete:
            candidate = int(fall[run_start])
        else:
            candidate = int(round(fall[block_start] - first_lines * line_samples))
        lock = {"start": candidate, "correlation": np.nan, "lag_samples": 0}
        if template is not None:
            lock = lock_interval(luma_lowpass_ire, candidate, template,
                                 line_samples)
        start = int(lock["start"])
        # the half-line offset against the preceding line syncs tells the
        # field parity; which physical head it is stays unknown
        previous = np.flatnonzero((kind[:run_start] == "line"))
        if len(previous):
            last_line = fall[previous[-1]]
            phase = ((start - last_line) / line_samples) % 1.0
            half_line = 0.25 < phase < 0.75
        else:
            half_line = False
        fields.append({"start": start, "block_index": block_start,
                       "run_complete": bool(complete),
                       "half_line_offset": bool(half_line),
                       "template_correlation": float(lock["correlation"]),
                       "template_shift_samples": int(lock["lag_samples"])})
    return fields


# --------------------------------------------------------------------------
# per-line measurement
# --------------------------------------------------------------------------

def crossing(trace: np.ndarray, low: int, high: int, level: float,
             falling: bool) -> float:
    """The sub-sample 50% crossing nearest the middle of [low, high)."""
    segment = trace[low:high]
    if len(segment) < 2:
        return np.nan
    below = segment <= level
    if falling:
        hits = np.flatnonzero(~below[:-1] & below[1:])
    else:
        hits = np.flatnonzero(below[:-1] & ~below[1:])
    if len(hits) == 0:
        return np.nan
    middle = 0.5 * (high - low)
    hit = hits[np.argmin(np.abs(hits - middle))]
    a, b = segment[hit], segment[hit + 1]
    if b == a:
        return np.nan
    return low + hit + (level - a) / (b - a)


def measure_lines(demod: Dict[str, object], pulses: Dict[str, np.ndarray],
                  fields: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Every series, one entry per normal-sync line, with the line's
    field-relative number from the specification's numbering.

    The number is floor((t - t0) / H + 1.25) with t0 the locked interval
    start: the first normal sync follows the nine interval lines at 9 H in
    one field parity and 9.5 H in the other, and both read 10.
    """
    sysparams = demod["sysparams"]
    rate = float(demod["rate_hz"])
    per_us = rate * 1e-6
    luma = demod["luma_ire"]
    chroma = demod["chroma"]
    envelope = demod["envelope"]
    colour_under = float(demod["colour_under_hz"])
    line_us = float(sysparams["line_period"])
    line_samples = line_us * per_us
    edge_us = float(sysparams["syncTransitionUS"])
    guard = SETTLING_EDGES * edge_us
    front_us = float(sysparams["frontPorchUS"])
    sync_us = float(sysparams["hsyncPulseUS"])
    burst_us = tuple(float(v) for v in sysparams["colorBurstUS"])
    active_us = tuple(float(v) for v in sysparams["activeVideoUS"])
    field_lines = tuple(int(v) for v in sysparams["field_lines"])

    def span(anchor: float, start_us: float, stop_us: float):
        low = int(round(anchor + start_us * per_us))
        high = int(round(anchor + stop_us * per_us))
        return max(low, 0), min(high, len(luma))

    def median_in(anchor, start_us, stop_us):
        low, high = span(anchor, start_us, stop_us)
        return float(np.median(luma[low:high])) if high > low + 1 else np.nan

    columns: Dict[str, list] = {k: [] for k in (
        "field", "line", "half_line_offset", "fall_sample", "sync_width_us",
        "sync_tip_ire", "front_porch_ire", "back_porch_ire",
        "burst_amplitude", "burst_phase_deg", "noise_floor_ire",
        "envelope_tip", "active_mean_ire", "active_rms_ire")}

    fall_all = pulses["fall"]
    kind = pulses["kind"]
    starts = [f["start"] for f in fields] + [len(luma)]
    for field_number, field in enumerate(fields):
        start, stop = field["start"], starts[field_number + 1]
        picked = np.flatnonzero((fall_all >= start) & (fall_all < stop)
                                & (kind == "line"))
        for pulse_index in picked:
            nominal = float(fall_all[pulse_index])
            line = int(np.floor((nominal - start) / line_samples + 1.25))
            if line < 1 or line > max(field_lines) + 1:
                continue
            low, high = span(nominal, -front_us - 1.0, front_us + 1.0)
            if low <= 0 or high >= len(luma):
                continue
            porch = median_in(nominal, -front_us + guard, -guard)
            tip = median_in(nominal, guard, sync_us - guard)
            if not (np.isfinite(porch) and np.isfinite(tip) and porch > tip):
                continue
            level = 0.5 * (porch + tip)
            fall = crossing(luma, *span(nominal, -1.0, 1.0), level, True)
            if not np.isfinite(fall):
                continue
            rise = crossing(luma, *span(fall, sync_us - 1.0, sync_us + 1.0),
                            level, False)
            if not np.isfinite(rise):
                continue
            tip = median_in(fall, guard, sync_us - guard)
            porch = median_in(fall, -front_us + guard, -guard)
            back = median_in(fall, burst_us[1] + guard, active_us[0] - guard)
            low, high = span(fall, guard, sync_us - guard)
            noise = float(np.sqrt(np.mean((luma[low:high] - tip) ** 2)))
            envelope_tip = float(np.mean(envelope[low:high]))
            low, high = span(fall, burst_us[0], burst_us[1])
            t = (np.arange(low, high) - fall) / rate
            analytic = np.mean(chroma[low:high]
                               * np.exp(-2j * np.pi * colour_under * t))
            amplitude = 2.0 * float(np.abs(analytic))
            phase = float(np.degrees(np.angle(analytic)) % PHASE_FOLD_DEG)
            low, high = span(fall, active_us[0] + ACTIVE_GUARD_US,
                             active_us[1] - ACTIVE_GUARD_US)
            active = luma[low:high]
            for name, value in (
                    ("field", field_number), ("line", line),
                    ("half_line_offset", field["half_line_offset"]),
                    ("fall_sample", fall),
                    ("sync_width_us", (rise - fall) / per_us),
                    ("sync_tip_ire", tip), ("front_porch_ire", porch),
                    ("back_porch_ire", back), ("burst_amplitude", amplitude),
                    ("burst_phase_deg", phase), ("noise_floor_ire", noise),
                    ("envelope_tip", envelope_tip),
                    ("active_mean_ire", float(np.mean(active))),
                    ("active_rms_ire", float(np.std(active)))):
                columns[name].append(value)
    return {name: np.asarray(values) for name, values in columns.items()}


def preceding_content(table: Dict[str, np.ndarray]) -> np.ndarray:
    """The previous line's active level, the covariate that found the porch
    bias. NaN where the previous measured line is not the previous line."""
    field, line = table["field"], table["line"]
    out = np.full(len(line), np.nan)
    for index in range(1, len(line)):
        if field[index] == field[index - 1] and line[index] == line[index - 1] + 1:
            out[index] = table["active_mean_ire"][index - 1]
    return out


def field_levels(table: Dict[str, np.ndarray], lines: Sequence[int],
                 minimum_lines: int = 8) -> Dict[int, float]:
    """The mean active level of the given lines, per field."""
    picked_lines = np.isin(table["line"], lines)
    out = {}
    for field in np.unique(table["field"]):
        picked = picked_lines & (table["field"] == field)
        if picked.sum() >= minimum_lines:
            out[int(field)] = float(np.mean(table["active_mean_ire"][picked]))
    return out


def fold_phase(phase_deg: np.ndarray, period_deg: float = PHASE_FOLD_DEG
               ) -> np.ndarray:
    """A phase folded modulo `period_deg`, re-expressed as the deviation from
    the population's circular mean and wrapped into +-period/2.

    A fold alone leaves the boundary inside the data: a burst sitting near
    0/90 degrees reads as 2 on one line and 88 on the next, and a linear
    variance calls that a swing of 86. Measured before this was added, the
    reference lines read eight decibels NOISIER than the picture lines for
    exactly that reason. The circular mean is taken on the full turn
    (phase times 360/period) and the deviation wrapped, which is the
    correct linearisation for a variance."""
    turns = np.asarray(phase_deg, dtype=np.float64) * (360.0 / period_deg)
    good = np.isfinite(turns)
    if not good.any():
        return np.asarray(phase_deg, dtype=np.float64)
    mean_turn = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(turns[good])))))
    centre = mean_turn * (period_deg / 360.0)
    deviation = (np.asarray(phase_deg, dtype=np.float64) - centre
                 + 0.5 * period_deg) % period_deg - 0.5 * period_deg
    return deviation


def subset(table: Dict[str, np.ndarray], mask: np.ndarray
           ) -> Dict[str, np.ndarray]:
    return {name: values[mask] for name, values in table.items()}


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

SERIES = (
    ("sync tip (IRE)", "sync_tip_ire"),
    ("front porch (IRE)", "front_porch_ire"),
    ("back porch after burst (IRE)", "back_porch_ire"),
    ("sync width (us)", "sync_width_us"),
    ("burst amplitude", "burst_amplitude"),
    ("burst phase (deg, folded)", "burst_phase_deg"),
    ("noise floor on tip (IRE)", "noise_floor_ire"),
    ("RF envelope on tip", "envelope_tip"),
)

HEADER = "  ".join([f"{'series':<30}", f"{'blanking':>10}", f"{'pattern':>10}",
                    f"{'full field':>10}", f"{'counts':>7}",
                    f"{'excess dB':>15}", f"{'z':>6}", f"{'verdict':<24}"])
RULE = "-" * len(HEADER)


def format_row(name: str, result: Dict[str, object],
               control: Optional[Dict[str, object]] = None) -> str:
    full = result["full_field"]
    picture = result["picture"]
    level = result["level"]
    pattern = result.get("pattern")
    cells = [f"{name:<30}",
             f"{full['reference_variance']:>10.4g}",
             (f"{pattern['test_variance']:>10.4g}" if pattern
              else f"{'-':>10}"),
             f"{full['test_variance']:>10.4g}",
             f"{full['predicted_db']:>+7.2f}",
             f"{full['excess_db']:>+7.2f} +-{full['error_db']:.2f}",
             f"{full['z']:>+6.1f}",
             f"{full['verdict']:<24}"]
    lines = ["  ".join(cells)]
    detail = (f"      picture only {picture['excess_db']:+.2f} +-"
              f"{picture['error_db']:.2f} dB (z {picture['z']:+.1f})")
    if pattern:
        detail += (f"; static pattern {pattern['excess_db']:+.2f} +-"
                   f"{pattern['error_db']:.2f} dB (z {pattern['z']:+.1f})")
    thirds = " / ".join(f"{p['excess_db']:+.2f}" for p in result["profile"])
    detail += (f"; along the field {thirds} dB "
               f"({result['drift']['verdict']})")
    switch = result["head_switch"]
    if np.isfinite(switch.get("excess_db", np.nan)):
        detail += f"; head-switch lines {switch['excess_db']:+.2f} dB"
    lines.append(detail)
    detail = (f"      level shift {level['shift']:+.3g} +-{level['error']:.2g}"
              f" ({level['verdict']})")
    if "content_slope" in result:
        s = result["content_slope"]
        detail += (f"; slope on preceding line {s['slope']:+.3g}/IRE "
                   f"t {s['t']:+.1f} ({s['verdict']})")
    if "field_rate" in result:
        f = result["field_rate"]
        detail += f"; field-rate r {f['r']:+.2f} t {f['t']:+.1f} ({f['verdict']})"
    lines.append(detail)
    if control is not None:
        attribution = ci.control_attribution(result, control, "full_field")
        if np.isfinite(attribution["above_control_db"]):
            lines.append(
                f"      above the static control "
                f"{attribution['above_control_db']:+.2f} +-"
                f"{attribution['error_db']:.2f} dB (z {attribution['z']:+.1f}"
                f", control {attribution['control_excess_db']:+.2f}): "
                f"{attribution['verdict']}")
    return "\n".join(lines)


def report(label: str, table: Dict[str, np.ndarray], populations,
           series=SERIES, control: Optional[Dict[str, Dict[str, object]]] = None
           ) -> Dict[str, Dict[str, object]]:
    table = dict(table)
    table["burst_phase_deg"] = fold_phase(table["burst_phase_deg"])
    line_index = table["line"]
    occupancy = ci.occupied_lines(table["active_rms_ire"], line_index,
                                  populations)
    reference_lines = [n for n in populations["reference"]
                       if n not in occupancy["occupied"]]
    levels = field_levels(table, populations["picture"])
    # the field-level noise is the same statistic on the blank lines
    blank_levels = field_levels(table, reference_lines, minimum_lines=3)
    level_noise = ci.robust_spread(np.array(list(blank_levels.values())))
    plateaus = ci.plateau_fields(levels, level_noise)
    pattern_mask = np.isin(table["field"], plateaus["fields"])
    covariate = preceding_content(table)
    fields = len(np.unique(table["field"]))
    print("=" * len(HEADER))
    parities = np.unique(table["half_line_offset"], return_counts=True)
    print(f"{label}: {len(line_index)} lines in {fields} fields "
          f"(parity split {dict(zip(parities[0].tolist(), parities[1].tolist()))}"
          f"); reference lines {populations['reference'][0]}-"
          f"{populations['reference'][-1]}, picture "
          f"{populations['picture'][0]}-{populations['picture'][-1]}, head "
          f"switch from {populations['head_switch'][0]}")
    print(f"  occupied reference lines (inserted signals, measured): "
          f"{occupancy['occupied'] or 'none'}; active RMS floor "
          f"{occupancy['floor']:.2f} IRE")
    detail = ", ".join(f"{n}:{occupancy['per_line'][n]:.2f}"
                       for n in populations["reference"])
    print(f"  active RMS per reference line (IRE): {detail}")
    if levels:
        values = np.array(list(levels.values()))
        summary = (f"  picture level per field: {values.min():+.1f} to "
                   f"{values.max():+.1f} IRE (field-level noise on blank "
                   f"lines {level_noise:.2f} IRE); static pattern = "
                   f"{len(plateaus['fields'])} of {len(levels)} fields")
        if not plateaus["static_everywhere"]:
            summary += (" on plateaus at " + ", ".join(
                f"{p['level']:+.1f} IRE ({len(p['fields'])} fields)"
                for p in plateaus["plateaus"])
                + f", {len(plateaus['in_transition'])} in transition")
        print(summary)
    print("=" * len(HEADER))
    print(HEADER)
    print(RULE)
    results = {}
    for name, column in series:
        result = ci.check_series(
            table[column], table["field"], line_index, populations,
            exclude_lines=occupancy["occupied"],
            preceding_content=covariate, field_content=levels,
            pattern_mask=pattern_mask,
            group_index=table["half_line_offset"])
        results[column] = result
        print(format_row(name, result,
                         control.get(column) if control else None))
        if len(plateaus["plateaus"]) > 1:
            # each plateau alone: static content within it, so its excess
            # is the floor and the difference between plateaus is the step
            parts = []
            for plateau in plateaus["plateaus"]:
                mask = (np.isin(table["field"], plateau["fields"])
                        & np.isin(line_index, populations["picture"]))
                channel = ci.variance_channel(
                    result["residual"], mask,
                    np.isin(line_index, result["reference_lines"]),
                    list(result["decomposition"]["reference_per_field"].values()))
                parts.append(f"{plateau['level']:+.0f} IRE alone "
                             f"{channel['excess_db']:+.2f} +-"
                             f"{channel['error_db']:.2f} dB")
            print("      plateaus: " + "; ".join(parts))
    return results


def run_capture(path: str, rate_hz: float, start_seconds: float,
                seconds: float, system: str, out: Optional[str],
                control: Optional[Dict[str, Dict[str, object]]] = None):
    start_sample = int(round(start_seconds * rate_hz))
    count = int(round(seconds * rate_hz))
    samples = read_rf(path, start_sample, count)
    if len(samples) < count:
        print(f"  {os.path.basename(path)}: {len(samples)} of {count} samples "
              f"available from {start_seconds:g} s "
              f"({len(samples) / rate_hz:.3f} s used)")
        seconds = len(samples) / rate_hz
    if len(samples) < 4 * 32768:
        raise RuntimeError(f"{path}: too few samples ({len(samples)}) at "
                           f"{start_seconds:g} s")
    demod = demodulate(samples, rate_hz, system)
    pulses = find_pulses(demod["luma_lowpass_ire"], rate_hz, demod["sysparams"])
    fields = find_fields(pulses, rate_hz, demod["sysparams"],
                         demod["luma_lowpass_ire"])
    if len(fields) < 3:
        raise RuntimeError(f"{path}: found only {len(fields)} fields")
    fields = fields[:-1]                  # the last one is not closed
    correlations = np.array([f["template_correlation"] for f in fields])
    shifts = np.array([f["template_shift_samples"] for f in fields])
    incomplete = sum(not f["run_complete"] for f in fields)
    # A field whose interval does not match the specification's train is a
    # field whose line numbers cannot be trusted - a dropout or the head
    # switch on the equalizing pulses. The gate is the lock quality's own
    # distribution: below the median by the detection convention times the
    # robust spread. This selects on the interval's shape, never on any
    # series under test.
    good = np.isfinite(correlations)
    gate = np.nan
    if good.sum() >= 4:
        gate = (np.median(correlations[good])
                - ci.DETECTION_SIGMA * ci.robust_spread(correlations[good]))
        good &= correlations >= gate
    kept = [f for f, ok in zip(fields, good) if ok]
    print(f"  interval lock to the specification's train: {len(fields)} "
          f"fields, {incomplete} with an incomplete equalizing run; "
          f"correlation median {np.nanmedian(correlations):.3f}, gate "
          f"{gate:.3f}, {len(fields) - len(kept)} field(s) excluded; shift "
          f"from the pulse candidate median "
          f"{np.median(np.abs(shifts)) / rate_hz * 1e6:.2f} us, maximum "
          f"{np.max(np.abs(shifts)) / rate_hz * 1e6:.2f} us")
    if len(kept) < 3:
        raise RuntimeError(f"{path}: only {len(kept)} usable fields")
    table = measure_lines(demod, pulses, kept)
    if len(table["line"]) == 0:
        raise RuntimeError(f"{path}: no lines measured; is --rate right for "
                           f"this capture?")
    populations = ci.line_populations(system, max(demod["sysparams"]["field_lines"]))
    label = (f"{os.path.basename(path)} at {start_seconds:g} s for {seconds:g} s"
             f" ({rate_hz/1e6:g} MSps)")
    results = report(label, table, populations, control=control)
    if out:
        np.savez_compressed(out, **table, populations_reference=np.array(
            populations["reference"]), populations_picture=np.array(
            populations["picture"]), source=os.path.basename(path),
            start_seconds=start_seconds, seconds=seconds, rate_hz=rate_hz,
            template_correlation=correlations)
        print(f"  per-line table saved to {out}")
    return results


def run_npz(path: str, system: str, reference_mhz=EXPORT_REFERENCE_MHZ,
            field_lines: int = 263):
    """The ringing lane's per-line export. Its line index is the decoder's
    zero-based count from the first equalizing pulse (linelocs), so the
    specification's number is one more; each head is its own table, which
    is the within-head decomposition by construction. `levels_h*` rows are
    [active, blanking before sync, tip, back porch] per line in field-major
    order and are aligned with the response grid when their count matches
    it exactly (verified on hb75: 14 x 252 and 13 x 252, lines 10-261)."""
    export = np.load(path)
    results = {}
    for head in (0, 1):
        lines_key = f"tdim_lines_M1f_h{head}"
        response_key = f"tdim_H_M1f_h{head}"
        if lines_key not in export.files:
            continue
        lines = export[lines_key] + 1
        response = export[response_key]           # (line, field, frequency)
        line_count, field_count, frequency_count = response.shape
        line_index = np.tile(lines, field_count)
        field_index = np.repeat(np.arange(field_count), line_count)
        levels = export.get(f"levels_h{head}")
        widths = export.get(f"width_h{head}")
        grid = None
        if levels is not None and levels.shape[0] == field_count * line_count:
            grid = levels.reshape(field_count, line_count, 4)
        populations = ci.line_populations(system, field_lines)
        label = (f"{os.path.basename(path)} head {head}: per-line ln|H| from "
                 f"the sync fall edge, {field_count} fields, lines "
                 f"{lines.min()}-{lines.max()}")
        print("=" * len(HEADER))
        print(label)
        print("=" * len(HEADER))
        print(HEADER)
        print(RULE)
        table = {"field": field_index, "line": line_index}
        covariate, levels_by_field, pattern_mask = None, None, None
        if grid is not None:
            table["active_mean_ire"] = grid[:, :, 0].T.reshape(-1)
            covariate = preceding_content(table)
            levels_by_field = field_levels(table, populations["picture"])
            blank = field_levels(table, populations["reference"], minimum_lines=3)
            noise = ci.robust_spread(np.array(list(blank.values())))
            plateaus = ci.plateau_fields(levels_by_field, noise)
            pattern_mask = np.isin(field_index, plateaus["fields"])
        for bin_index, mhz in enumerate(list(reference_mhz)[:frequency_count]):
            values = np.log(np.abs(response[:, :, bin_index])).reshape(-1)
            result = ci.check_series(values, field_index, line_index,
                                     populations, preceding_content=covariate,
                                     field_content=levels_by_field,
                                     pattern_mask=pattern_mask)
            results[(head, mhz)] = result
            print(format_row(f"ln|H| at {mhz:g} MHz (Np)", result))
        if grid is not None:
            named = [("sync tip (IRE, export)", grid[:, :, 2]),
                     ("back porch (IRE, export)", grid[:, :, 3]),
                     ("front porch (IRE, export)", grid[:, :, 1])]
            if widths is not None and widths.shape[0] == field_count * line_count:
                named.append(("sync width (us, export)",
                              widths.reshape(field_count, line_count)))
            for name, values in named:
                result = ci.check_series(values.T.reshape(-1), field_index,
                                         line_index, populations,
                                         preceding_content=covariate,
                                         field_content=levels_by_field,
                                         pattern_mask=pattern_mask)
                results[(head, name)] = result
                print(format_row(name, result))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", default=None,
                        help="a static test-pattern capture run first; every "
                             "other capture's excess is then read against it")
    parser.add_argument("--capture", action="append", default=[])
    parser.add_argument("--rate", type=float, default=50e6,
                        help="sample rate in Hz (50e6 for the test-pattern "
                             "captures, 40e6 for home/countdown)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="seconds into the capture")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="window length; a shorter file is used whole")
    parser.add_argument("--control-rate", type=float, default=None,
                        help="the control's sample rate (default: --rate)")
    parser.add_argument("--control-start", type=float, default=0.0)
    parser.add_argument("--control-seconds", type=float, default=3.0)
    parser.add_argument("--system", default="NTSC")
    parser.add_argument("--npz", action="append", default=[],
                        help="a *_hs_impulse.npz export to check")
    parser.add_argument("--npz-mhz", type=float, nargs="*",
                        default=list(EXPORT_REFERENCE_MHZ),
                        help="the export's reference frequencies")
    parser.add_argument("--out", default=None,
                        help="save the per-line table (one capture only)")
    args = parser.parse_args(argv)
    if not args.capture and not args.npz and not args.control:
        parser.error("give --control, --capture or --npz")
    control = None
    if args.control:
        control = run_capture(args.control, args.control_rate or args.rate,
                              args.control_start, args.control_seconds,
                              args.system, None)
    for path in args.capture:
        run_capture(path, args.rate, args.start, args.seconds, args.system,
                    args.out if len(args.capture) == 1 else None,
                    control=control)
    for path in args.npz:
        run_npz(path, args.system, args.npz_mhz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
