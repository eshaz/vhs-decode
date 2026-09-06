"""The cross-instrument transfer: the COMPLEX response between the record tap
and the playback tap of one deck, per pattern and per head.

WHAT THE TWO TAPS ARE. The test-pattern captures hold the RF at the record
head's drive (CN261 pin 1: after the emphasis, the clipping and the FM
modulator, before the head) and the RF from the playback head amplifier
(CN261 pin 2) of one Sony SLV-778HF, for sixteen patterns at SP and EP. The
ratio of the two is everything BETWEEN the taps - the record head's write
process, the tape, the playback head, its preamplifier and its equalisation
- with no demodulator in the path and no carrier-coverage limit.

WHY THIS MODULE EXISTS BESIDE `tools/ringing_measure/record_playback_transfer
.py` AND `depth_split.py`. Both of those take the RATIO OF POWER SPECTRA,
which is a magnitude-only instrument, and the arc's record is that
magnitude-only instruments were the source of a family of defects
(`chroma_complex_envelope`: the burst instrument was rank 2 of 4 until its
phase was carried). A power ratio has a second, quieter defect that matters
more here: it equals |H|^2 / coherence, so wherever the playback holds
power that is NOT a copy of the record - tape noise, the HiFi audio
carriers the video head picks up, head-switch and dropout events - the
ratio reads the noise floor and calls it response. Measured on 75 bars SP,
the power ratio reads +1.8 dB at 1.7-2.0 MHz and +2.3 dB at 6.0-6.5 MHz
where the coherent transfer reads -2.7 and -4.1 dB (second-field head);
see `compare_with_export`.

THE CAPTURES DO NOT OVERLAP, AND THE MEASUREMENT IS COHERENT ANYWAY. Each
file is one 24 000 512-sample acquisition (the scope's 24 Mpt memory), which
at 50 MS/s is 0.48 s - the FLAC header's 50 kHz is a placeholder. The record
capture and the playback capture are therefore two 0.48 s windows of a
recording that lasted minutes, and cross-correlating the two files finds
nothing. What makes a coherent measurement possible is the pattern being
STATIC: two lines of identical video are modulated onto the identical FM
waveform up to a constant rotation of the carrier, because the modulator's
phase advance over a line of the same content is the same number every
line. So a record line and a playback line of the same line number are the
same waveform through the channel, up to one delay and one rotation for the
luma FM and one more rotation for the colour-under (whose phase the format
rotates by 90 degrees per line, so its rotation is a different constant).
All three are estimated per line pair and taken off; the spectral SHAPE of
the transfer, magnitude and phase, survives. Measured: the record tap's own
line-to-line coherence is 0.9992 over the luma FM band (the ceiling), and a
record line against a playback line of the same content is 0.995.

ONE ROTATION PER BAND IS ALL THERE IS, AND WHERE THE TWO BANDS OVERLAP THE
INSTRUMENT IS BLIND. The luma FM and the colour-under advance by different
angles per line, so a bin that carries both cannot be brought into line by
either rotation: whichever is applied, the other component is misrotated and
partly cancels in the sum over lines. The witness is the record tap's own
coherent fraction beta, which falls from 0.99 inside the luma FM band to
0.39 just above the chroma band's top on the synthetic taps of
`tests/unit/test_tap_transfer.py`, and the cost is carried in the error bar
rather than corrected - see `complex_transfer`, THE ERROR BAR MUST CARRY THE
DENOMINATOR.

THE ESTIMATOR IS A RATIO OF TWO CROSS-SPECTRA, NOT A CROSS-SPECTRUM OVER A
POWER SPECTRUM. The textbook H1 = Sxy / Sxx is unbiased only when the INPUT
is noise-free, and the record tap is not: scope noise, quantisation and the
modulator's phase-noise skirts are in it, and they scale H1 by the record
tap's coherent fraction beta(f). Measured with two record fields against
each other - a channel that is exactly unity - H1 read -5 dB at 1.1-1.4 MHz
and -7 dB at 6.0-6.5 MHz. The cross-spectrum of two record fields measures
beta directly, so

    H(f) = Sxy(record -> playback) / Sxy(record -> record)

with each side divided by the number of line pairs that made it, is exactly
1 for a copy, needs no noise model, and has an unbiased phase.
Per playback field the numerator is that field's own cross-spectrum against
a record field of the same interlace parity, the denominator is pooled over
far-apart same-parity record pairs, and the standard error is the scatter
of the per-field estimates: SE from repeated fields, per bin.

PER HEAD. The heads alternate per field and the switch lands 5 to 8 H ahead
of the vertical sync (SMPTE 32M, `sync_geometry.HEAD_SWITCH_AHEAD_OF_VSYNC_H`),
so a field's body from the end of the vertical interval to a line ahead of
the switch is one head. The head is labelled by the INTERLACE PARITY of the
field it reads - `first-field head` and `second-field head` - which on this
deck is a fixed physical identity, because the head that wrote a track is
the only head that can read it (the azimuths differ) and one machine did
both. Which physical head is which is not knowable from the taps.

THE PLAYBACK-ONLY PAIRS ARE CONTAMINATED AND ARE NOT USED FOR HEAD RATIOS.
The HiFi audio FM carriers the video head picks up are tones, and a tone is
line-coherent - its phase advances by the same angle every line, exactly
like the luma FM - so it survives a playback-to-playback cross-spectrum.
Measured, the two playback-only head ratios came out +1.2 dB and -10.3 dB
where reciprocals must sum to zero. The record tap has no audio carrier, so
the record-to-playback transfer is clean, and the head ratio is taken as
the ratio of the two heads' record-to-playback transfers.

The evidence for every figure above and for the module's results is
recorded on the functions that produce it; the runnable instrument is
`tools/ringing_measure/tap_transfer.py`.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.formats import get_format_params, parse_tape_speed
from vhsdecode.models import head_model, rf_stages, sync_geometry


DB_PER_NEPER = 20.0 / np.log(10.0)

# Where the shared exports of this arc live, so the reporting grid and the
# reference band can be read from the 2026-09-02 magnitude-only export and
# the comparison made on its own grid. Read only; never written by this module.
SHARED_EXPORT = ("/tmp/claude-1000/-workspaces-vhs-decode/shared/"
                 "record_playback_transfer_2026-09-02.npz")


# --------------------------------------------------------------------------
# The parameters, all read from the decoder's own tables
# --------------------------------------------------------------------------


def sample_rate_from_name(path: str) -> Optional[float]:
    """The capture's rate from its file name (`...-50msps-...`).

    The FLAC header cannot hold 50 MHz and carries 50 kHz instead, so the
    name is the only record of the rate the file itself keeps."""
    match = re.search(r"(\d+(?:\.\d+)?)msps", os.path.basename(path).lower())
    return float(match.group(1)) * 1e6 if match else None


def capture_parameters(system: str = "NTSC", tape_format: str = "VHS",
                       speed: str = "SP") -> Dict[str, object]:
    """Everything the instrument needs, derived from `vhsdecode.formats`.

    The sync-tip and blanking carriers come from `ire0`, `hz_ire` and
    `vsync_ire`; the pulse widths from the system's timing table; the luma
    FM band is the decoder's own chroma cut to its own RF band-pass top, and
    the colour-under band is the decoder's chroma band-pass mirrored about
    the colour-under carrier. The usable lines of a field exclude the
    vertical interval (`sync_geometry.SEQUENCE_LINES`) and the head-switch
    window (`HEAD_SWITCH_AHEAD_OF_VSYNC_H` plus the assumed transient).
    """
    sys_params, rf_params = get_format_params(system, tape_format,
                                              parse_tape_speed(speed), None)
    line_us = float(sys_params["line_period"])
    hz_ire = float(sys_params["hz_ire"])
    ire0 = float(sys_params["ire0"])
    vsync_ire = float(sys_params["vsync_ire"])
    sync_tip_hz = ire0 + vsync_ire * hz_ire
    key = "525" if int(sys_params["frame_lines"]) == 525 else "625"
    interval_lines = sync_geometry.SEQUENCE_LINES[key]
    # the field's first usable line: the interval ends `middle + last` lines
    # after the V-sync leading edge (line 0 here), and the first full line
    # after it is the next one
    first_usable = int(np.ceil(interval_lines[1] + interval_lines[2])) + 1
    # the last usable line: the switch is 5 to 8 H ahead of the NEXT V-sync
    # and its transient (assumed, not specified) follows the onset
    shortest_field = int(min(sys_params["field_lines"]))
    last_usable = int(np.floor(
        shortest_field - max(sync_geometry.HEAD_SWITCH_AHEAD_OF_VSYNC_H)
        - sync_geometry.ASSUMED_SWITCH_TRANSIENT_H)) - 1
    chroma_top = float(rf_params["chroma_bpf_upper"])
    colour_under = float(rf_params["color_under_carrier"])
    widths_us = {"horizontal": float(sys_params["hsyncPulseUS"]),
                 "equalizing": float(sys_params["eqPulseUS"]),
                 "field": float(sys_params["vsyncPulseUS"])}
    # the classes are told apart by width, at half the smallest gap
    gaps = np.diff(sorted(widths_us.values()))
    tolerance_us = float(gaps.min()) / 2.0
    audio = [float(rf_params[k]) for k in sorted(rf_params)
             if k.startswith("fm_audio_channel_") and k.endswith("_freq")]
    return {
        "system": system, "tape_format": tape_format, "speed": speed,
        "line_period_s": line_us * 1e-6,
        "sync_tip_hz": sync_tip_hz,
        "blanking_hz": ire0,
        # halfway between sync tip and blanking, the decoder's own sync level
        "sync_threshold_hz": ire0 + 0.5 * vsync_ire * hz_ire,
        "peak_white_hz": ire0 + 100.0 * hz_ire,
        "pulse_widths_us": widths_us,
        "pulse_tolerance_us": tolerance_us,
        "sync_transition_us": float(sys_params["syncTransitionUS"]),
        "field_lines": tuple(int(v) for v in sys_params["field_lines"]),
        "first_usable_line": first_usable,
        "last_usable_line": last_usable,
        # the luma FM band: above the decoder's chroma cut, below its RF top
        "fm_band_hz": (chroma_top, float(rf_params["video_bpf_high"])),
        # the colour-under band: the decoder's chroma band-pass, mirrored
        "chroma_band_hz": (max(2.0 * colour_under - chroma_top, 0.0),
                           chroma_top),
        "colour_under_hz": colour_under,
        "fm_audio_hz": audio,
        # the reference band for the magnitude and the phase convention: the
        # export's own if it exists, else sync tip to the decoder's RF peak
        "reference_band_hz": _reference_band(
            sync_tip_hz, float(rf_params["video_rf_peak_freq"])),
        "mechanics": rf_stages.mechanics_at_speed(speed, system),
    }


def _reference_band(low_hz: float, high_hz: float) -> Tuple[float, float]:
    if os.path.exists(SHARED_EXPORT):
        try:
            with np.load(SHARED_EXPORT) as export:
                ref = export["reference"]
                return (float(ref[0]) * 1e6, float(ref[1]) * 1e6)
        except (OSError, KeyError, ValueError):
            pass
    return (float(low_hz), float(high_hz))


def reporting_bands() -> List[Tuple[float, float]]:
    """The banding grid, in Hz: the 2026-09-02 export's own so the two
    instruments are compared like for like, else the same fifteen bands."""
    if os.path.exists(SHARED_EXPORT):
        try:
            with np.load(SHARED_EXPORT) as export:
                return [(float(a) * 1e6, float(b) * 1e6)
                        for a, b in export["bands"]]
        except (OSError, KeyError, ValueError):
            pass
    edges = (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 3.4, 3.9, 4.4, 5.0,
             5.5, 6.0, 6.5)
    return [(a * 1e6, b * 1e6) for a, b in zip(edges[:-1], edges[1:])]


# --------------------------------------------------------------------------
# Reading the captures
# --------------------------------------------------------------------------


def read_capture(path: str, start: int = 0, count: Optional[int] = None
                 ) -> np.ndarray:
    """A window of a capture as float32, positioned with soundfile's seek."""
    import soundfile as sf

    with sf.SoundFile(path) as handle:
        if start:
            handle.seek(int(start))
        frames = -1 if count is None else int(count)
        data = handle.read(frames, dtype="float32", always_2d=False)
    return np.ascontiguousarray(np.asarray(data).ravel())


# --------------------------------------------------------------------------
# The sync structure: lines and fields from the instantaneous frequency
# --------------------------------------------------------------------------


def analytic_in_band(x: np.ndarray, sample_rate_hz: float,
                     band_hz: Tuple[float, float]) -> np.ndarray:
    """The analytic signal of one band, by the FFT."""
    n = len(x)
    spectrum = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    spectrum[(f < band_hz[0]) | (f > band_hz[1])] = 0.0
    full = np.zeros(n, dtype=np.complex128)
    full[:len(spectrum)] = spectrum
    full[1:len(spectrum) - 1] *= 2.0
    return np.fft.ifft(full)


def instantaneous_frequency(x: np.ndarray, sample_rate_hz: float,
                            band_hz: Tuple[float, float],
                            smooth_s: float, block: int = 1 << 20
                            ) -> np.ndarray:
    """The luma FM's instantaneous frequency, block-wise with overlap, then
    smoothed by a boxcar shorter than any sync pulse."""
    out = np.empty(len(x))
    overlap = 4096
    step = block - 2 * overlap
    for start in range(0, len(x), step):
        a = max(0, start - overlap)
        b = min(len(x), start + block - overlap)
        z = analytic_in_band(x[a:b], sample_rate_hz, band_hz)
        f = np.gradient(np.unwrap(np.angle(z))) * sample_rate_hz / (2 * np.pi)
        lo = start - a
        hi = lo + (min(len(x), start + step) - start)
        out[start:start + (hi - lo)] = f[lo:hi]
        if start + step >= len(x):
            break
    k = max(int(round(smooth_s * sample_rate_hz)), 1)
    return np.convolve(out, np.ones(k) / k, mode="same")


@dataclass
class Field:
    """One field of a capture: where its V-sync leading edge is, the sample
    of every horizontal sync it holds by line number counted from that edge,
    and its interlace parity (0 first field, 1 second)."""
    start: int
    lines: Dict[int, int]
    parity: int
    index: int = 0


def locate_fields(x: np.ndarray, sample_rate_hz: float,
                  params: Dict[str, object]) -> List[Field]:
    """Fields and their lines from the sync pulses in the FM.

    Pulses are the intervals where the instantaneous frequency sits below
    the sync threshold; they are classed by width as horizontal, equalizing
    or field (broad), and a field begins at the first broad pulse after a
    non-broad one - the V-sync leading edge. Line numbers count from that
    edge; the first field's syncs fall at whole-line offsets from it and the
    second field's at half-line offsets, which is how the parity is read.
    Fields with fewer than half their lines are dropped rather than
    numbered, which is what a capture's partial first and last fields are.
    """
    fs = float(sample_rate_hz)
    line = float(params["line_period_s"]) * fs
    widths = params["pulse_widths_us"]
    tolerance = float(params["pulse_tolerance_us"]) * fs * 1e-6
    smooth_s = widths["horizontal"] * 1e-6 / 10.0
    f = instantaneous_frequency(x, fs, params["fm_band_hz"], smooth_s)
    low = f < float(params["sync_threshold_hz"])
    starts = np.flatnonzero(low[1:] & ~low[:-1]) + 1
    ends = np.flatnonzero(~low[1:] & low[:-1]) + 1
    if len(starts) == 0 or len(ends) == 0:
        return []
    ends = ends[ends > starts[0]]
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    width = ends - starts
    broad = np.abs(width - widths["field"] * 1e-6 * fs) < tolerance
    horizontal = np.abs(width - widths["horizontal"] * 1e-6 * fs) < tolerance
    v_edges = [int(starts[i]) for i in range(1, n) if broad[i] and not broad[i - 1]]
    h_starts = starts[horizontal]
    fields: List[Field] = []
    for k in range(len(v_edges) - 1):
        a, b = v_edges[k], v_edges[k + 1]
        inside = h_starts[(h_starts > a) & (h_starts < b)]
        if len(inside) < min(params["field_lines"]) // 2:
            continue
        offset = (inside - a) / line
        number = np.floor(offset + 0.25).astype(int)
        fraction = offset - np.floor(offset + 0.25)
        parity = int(np.round(float(np.median(fraction)) * 2.0)) % 2
        fields.append(Field(start=a, lines={int(m): int(s) for m, s in
                                             zip(number, inside)},
                            parity=parity, index=len(fields)))
    return fields


# --------------------------------------------------------------------------
# One line pair: delay, rotations, spectra
# --------------------------------------------------------------------------


@dataclass
class Grid:
    """The spectral grid one analysis runs on."""
    sample_rate_hz: float
    line_samples: int
    nfft: int
    freqs: np.ndarray
    window: np.ndarray
    fm_bins: np.ndarray
    chroma_bins: np.ndarray
    lead_samples: int
    lag_guard: int


def make_grid(sample_rate_hz: float, params: Dict[str, object]) -> Grid:
    """One line per window, Hann-tapered, zero-padded to the next power of
    two; the window starts a sync transition's worth ahead of the sync edge
    so the edge itself is inside it."""
    fs = float(sample_rate_hz)
    line_samples = int(round(float(params["line_period_s"]) * fs))
    nfft = 1 << int(np.ceil(np.log2(line_samples)))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    lo, hi = params["fm_band_hz"]
    clo, chi = params["chroma_band_hz"]
    # the search for the integer lag is bounded by one sync pulse width: the
    # two captures' lines were each located from their own sync edge, so
    # the residual is the channel's delay and the locator's error only
    guard = int(round(params["pulse_widths_us"]["horizontal"] * 1e-6 * fs))
    lead = int(round(params["pulse_widths_us"]["horizontal"] * 1e-6 * fs))
    return Grid(fs, line_samples, nfft, freqs, np.hanning(line_samples),
                (freqs >= lo) & (freqs <= hi), (freqs >= clo) & (freqs <= chi),
                lead, guard)


def _line_spectrum(x: np.ndarray, start: int, grid: Grid) -> Optional[np.ndarray]:
    a = start - grid.lead_samples
    if a < 0 or a + grid.line_samples > len(x):
        return None
    return np.fft.rfft(x[a:a + grid.line_samples] * grid.window, grid.nfft)


def _whitened_correlation(X: np.ndarray, Y: np.ndarray, grid: Grid
                          ) -> np.ndarray:
    """The generalised cross-correlation over the luma FM band with a
    floored phase transform: bins above the band's median product are
    whitened, bins below it are kept in proportion, so a blank line - one
    carrier tone, whose plain correlation repeats every carrier cycle - still
    yields one peak at the delay, anchored by its sync edge's sidebands."""
    product = np.conj(X) * Y
    weight = np.abs(product[grid.fm_bins])
    floor = float(np.median(weight)) if weight.size else 1.0
    spectrum = np.zeros(grid.nfft, dtype=np.complex128)
    spectrum[:len(grid.freqs)] = np.where(
        grid.fm_bins, product / (np.abs(product) + floor + 1e-300), 0.0)
    return np.fft.ifft(spectrum)


@dataclass
class LinePair:
    X: np.ndarray
    Y: np.ndarray
    delay_samples: float
    coherence: float
    rotation_fm: float
    rotation_chroma: float


def align_line_pair(x_a: np.ndarray, start_a: int, x_b: np.ndarray,
                    start_b: int, grid: Grid) -> Optional[LinePair]:
    """Line b aligned to line a: the integer lag from the whitened
    correlation, the fraction by parabolic interpolation of its peak applied
    as a linear phase, one rotation for the luma FM band, one for the
    colour-under band. Returns None when the lag exceeds the guard, which is
    the misalignment witness the caller counts.

    SIGN CONVENTION. `delay_samples` is the offset to ADD to `start_b` to
    bring b's window onto a's content - the correction, not the displacement:
    a window read five samples LATER than a's holds content five samples
    EARLIER, so `align_line_pair(x, s, x, s + 5, grid)` returns -5, which is
    the number `_line_spectrum(x_b, start_b + lag)` needs to get back to
    `start_a`."""
    X = _line_spectrum(x_a, start_a, grid)
    Y0 = _line_spectrum(x_b, start_b, grid)
    if X is None or Y0 is None:
        return None
    c = _whitened_correlation(X, Y0, grid)
    k = int(np.argmax(np.abs(c)))
    lag = k if k < grid.nfft // 2 else k - grid.nfft
    if abs(lag) > grid.lag_guard:
        return None
    Y = _line_spectrum(x_b, start_b + lag, grid)
    if Y is None:
        return None
    c = _whitened_correlation(X, Y, grid)
    y0, y1, y2 = np.abs(c[-1]), np.abs(c[0]), np.abs(c[1])
    curvature = y0 - 2.0 * y1 + y2
    fraction = float(np.clip(0.5 * (y0 - y2) / curvature, -1.0, 1.0)) \
        if curvature < 0 else 0.0
    fm = grid.fm_bins
    coherence = float(np.abs(np.sum(np.conj(X[fm]) * Y[fm]))
                      / np.sqrt(np.sum(np.abs(X[fm]) ** 2)
                                * np.sum(np.abs(Y[fm]) ** 2) + 1e-300))
    Y = Y * np.exp(-2j * np.pi * grid.freqs * fraction / grid.sample_rate_hz)
    rotation = float(np.angle(np.sum(np.conj(X[fm]) * Y[fm])))
    Y = Y * np.exp(-1j * rotation)
    ch = grid.chroma_bins
    rotation_chroma = float(np.angle(np.sum(np.conj(X[ch]) * Y[ch])))
    Y = Y.copy()
    Y[ch] *= np.exp(-1j * rotation_chroma)
    return LinePair(X, Y, lag + fraction, coherence, rotation, rotation_chroma)


# --------------------------------------------------------------------------
# Field pairs: accumulated cross-spectra
# --------------------------------------------------------------------------


@dataclass
class CrossSpectrum:
    Sxy: np.ndarray
    Sxx: np.ndarray
    Syy: np.ndarray
    lines: int
    delays: Dict[int, float] = field(default_factory=dict)
    coherences: Dict[int, float] = field(default_factory=dict)
    rejected: int = 0


def field_pair_cross_spectrum(x_a: np.ndarray, field_a: Field,
                              x_b: np.ndarray, field_b: Field, grid: Grid,
                              first_line: int, last_line: int,
                              track_lines: Optional[Tuple[int, int]] = None
                              ) -> CrossSpectrum:
    """The cross-spectrum of two fields, line by line at equal line numbers.

    `track_lines` extends the DELAY track (not the spectra) beyond the
    usable range, so the head switch can be located from the timing step
    it leaves without letting it into the transfer."""
    nb = len(grid.freqs)
    out = CrossSpectrum(np.zeros(nb, np.complex128), np.zeros(nb),
                        np.zeros(nb), 0)
    lo, hi = first_line, last_line
    if track_lines is not None:
        lo, hi = min(lo, track_lines[0]), max(hi, track_lines[1])
    for number in range(lo, hi + 1):
        if number not in field_a.lines or number not in field_b.lines:
            continue
        pair = align_line_pair(x_a, field_a.lines[number], x_b,
                               field_b.lines[number], grid)
        if pair is None:
            out.rejected += 1
            continue
        out.delays[number] = pair.delay_samples + (
            field_b.lines[number] - field_a.lines[number])
        out.coherences[number] = pair.coherence
        if first_line <= number <= last_line:
            out.Sxy += np.conj(pair.X) * pair.Y
            out.Sxx += np.abs(pair.X) ** 2
            out.Syy += np.abs(pair.Y) ** 2
            out.lines += 1
    return out


def same_parity_pairs(fields: Sequence[Field], minimum_separation: int
                      ) -> List[Tuple[int, int]]:
    """Index pairs of equal parity at least `minimum_separation` apart."""
    pairs = []
    for i, a in enumerate(fields):
        for j in range(i + 1, len(fields)):
            if fields[j].parity == a.parity and j - i >= minimum_separation:
                pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------
# The transfer per head
# --------------------------------------------------------------------------


HEAD_LABELS = {0: "first-field head", 1: "second-field head"}


@dataclass
class TapTransfer:
    """The complex transfer of one head, with its per-field scatter."""
    head: int
    label: str
    freqs: np.ndarray
    transfer: np.ndarray            # complex, pooled over fields, phase convention applied
    per_field: np.ndarray           # complex, (fields, bins), same convention
    log_magnitude: np.ndarray       # nepers, mean over fields
    se_log_magnitude: np.ndarray    # nepers, TOTAL: max(fields, random) with reference and bound in quadrature
    phase: np.ndarray               # radians, excess phase (linear removed over the reference band)
    se_phase: np.ndarray            # radians, TOTAL, the same four terms
    coherence: np.ndarray           # pooled gamma^2 record->playback
    beta_record: np.ndarray         # the record tap's coherent fraction
    record_coherent_power: np.ndarray   # |Sxy(record->record)| per line, pooled: the band weight
    power_ratio: np.ndarray         # Syy/Sxx pooled: the magnitude-only estimator, for comparison
    fields: int
    lines: int
    rejected_lines: int
    line_coherence: float
    delay_track: Dict[str, object]
    group_delay_reference_s: float
    reference_band_hz: Tuple[float, float]
    # the four terms of the error bar, kept apart so each can be read
    se_log_magnitude_fields: np.ndarray     # nepers, scatter of the numerators over fields
    se_log_magnitude_reference: np.ndarray  # nepers, scatter of the denominator over record self-pairs
    bound_log_magnitude_incoherent: np.ndarray   # nepers, the -log(beta) bound
    se_phase_fields: np.ndarray             # radians
    se_phase_reference: np.ndarray          # radians
    bound_phase_incoherent: np.ndarray      # radians
    reference: np.ndarray                   # complex Sxy(record -> record) PER LINE, pooled: the denominator
    reference_per_pair: np.ndarray          # complex, (record self-pairs, bins), per line
    reference_independent: int              # record fields behind those pairs
    se_log_magnitude_random: np.ndarray     # nepers, the coherence-and-lines term
    se_phase_random: np.ndarray             # radians, the same term
    chroma_phase_constant_rad: float        # the free colour-under constant, removed


def _linear_phase_fit(freqs: np.ndarray, phase: np.ndarray,
                      band: np.ndarray) -> Tuple[float, float]:
    """Slope and intercept of the unwrapped phase over a band."""
    f = freqs[band]
    columns = np.vstack([f, np.ones_like(f)]).T
    slope, intercept = np.linalg.lstsq(columns, phase[band], rcond=None)[0]
    return float(slope), float(intercept)


def _phase_convention(H: np.ndarray, freqs: np.ndarray,
                      reference_band_hz: Tuple[float, float]
                      ) -> Tuple[np.ndarray, float]:
    """Remove the delay and the constant the reference band defines.

    A pure delay is not measurable between two captures with no common
    clock, and a constant rotation is not measurable at all for a static
    pattern, so the phase is reported as EXCESS phase: what remains after
    the best linear fit over the reference band is taken off. The removed
    slope is returned as the reference group delay so it can be quoted."""
    band = (freqs >= reference_band_hz[0]) & (freqs < reference_band_hz[1])
    unwrapped = np.unwrap(np.angle(H))
    slope, intercept = _linear_phase_fit(freqs, unwrapped, band)
    corrected = H * np.exp(-1j * (slope * freqs + intercept))
    return corrected, -slope / (2.0 * np.pi)


def _chroma_phase_convention(H: np.ndarray, freqs: np.ndarray,
                             chroma_band_hz: Tuple[float, float],
                             weight: np.ndarray) -> Tuple[np.ndarray, float]:
    """Remove the colour-under band's own free constant, and return it.

    `align_line_pair` takes a rotation of its OWN off the colour-under band
    of every line pair, and it has to: the format rotates the chroma by 90
    degrees per line, the two captures are different moments of tape, and the
    number of lines between them is not knowable. So the chroma band's phase
    is determined only up to an additive constant per line pair, and what
    reaches the ratio is the difference between the numerator's constant and
    the denominator's - a number set by which fields were paired, not by the
    channel.

    Measured on the synthetic identity taps of
    `tests/unit/test_tap_transfer.py`, where the true transfer is unity: the
    offset holds at -20.2 degrees across every admitted chroma bin, moves by
    at most 1 degree when the noise realisation changes, and moves to -27.8
    degrees when one record field is dropped from the denominator. It is
    removed rather than reported, exactly as `_phase_convention` removes the
    delay and the constant it cannot measure either; what survives in this
    band is the phase's VARIATION across it, and that is what
    `band_summary`'s group delay and `separate` compare.

    The weight is the record tap's coherent power, so the constant is set by
    the bins that carry the information."""
    band = (freqs >= chroma_band_hz[0]) & (freqs < chroma_band_hz[1])
    if not band.any():
        return H, 0.0
    constant = float(np.angle(np.sum(weight[band] * H[band])))
    out = H.copy()
    out[band] = out[band] * np.exp(-1j * constant)
    return out, constant


def _random_error(coherence: np.ndarray, averages: int) -> np.ndarray:
    """The random error of a cross-spectral ratio, per bin, from the measured
    coherence and the number of line pairs averaged.

    With y = H x + n and n independent line to line, the numerator's noise
    term sums incoherently while its signal term sums coherently, so the
    fractional error of the numerator is sqrt(sigma_n^2 / (N |H|^2 Sxx)) =
    sqrt((1 - gamma^2) / (N gamma^2)); half that variance lands on the
    magnitude and half on the phase, which is the standard normalised random
    error sqrt((1 - gamma^2) / (2 N gamma^2)) - the same family as
    `pair_dimension.coherent_transfer`'s `standard_error`, written relatively
    rather than absolutely. It returns nepers for the log magnitude and
    radians for the phase, the two being equal.

    THIS TERM IS NOT OPTIONAL WITH FEW FIELDS. The scatter over fields is a
    3-sample or 4-sample estimate of the same quantity on a 0.48 s capture,
    and it under-reports: measured on the synthetic identity taps of
    `tests/unit/test_tap_transfer.py`, the field scatter read 0.29 dB at
    5.42 MHz where this term reads 0.50 dB and the estimate was 0.86 dB from
    unity."""
    gamma2 = np.clip(coherence, 0.0, 1.0)
    n = max(int(averages), 1)
    return np.sqrt((1.0 - gamma2) / np.maximum(2.0 * n * gamma2, 1e-300))


def _circular_stats(per_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean phase and its SE across fields, wrap-safe."""
    unit = per_field / np.maximum(np.abs(per_field), 1e-300)
    mean = np.angle(unit.mean(axis=0))
    residual = np.angle(unit * np.exp(-1j * mean)[None, :])
    n = per_field.shape[0]
    se = residual.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 \
        else np.full(per_field.shape[1], np.inf)
    return mean, se


def _reference_scatter(per_pair: np.ndarray, independent: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """The denominator's OWN standard error, per bin, in log magnitude and in
    phase, from the scatter of the record self-pairs that make it.

    The pairs share fields - on the synthetic taps of
    `tests/unit/test_tap_transfer.py` nine pairs come from seven fields, and
    on a real 0.48 s capture about fifty come from twenty-seven - so they are
    not that many independent draws. What the scatter is divided by is the
    number of distinct record FIELDS behind the pairs, each field being one
    realisation of the record tap. That is the smaller, and therefore the
    honest, denominator."""
    bins = per_pair.shape[1] if per_pair.ndim == 2 else 0
    if per_pair.shape[0] < 2 or independent < 2:
        return np.full(bins, np.inf), np.full(bins, np.inf)
    root = np.sqrt(float(independent))
    log = np.log(np.maximum(np.abs(per_pair), 1e-300))
    unit = per_pair / np.maximum(np.abs(per_pair), 1e-300)
    residual = np.angle(unit * np.exp(-1j * np.angle(unit.mean(axis=0)))[None, :])
    return (log.std(axis=0, ddof=1) / root,
            residual.std(axis=0, ddof=1) / root)


def _incoherent_bounds(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """What the record tap's non-reproducing fraction can do to the ratio.

    Split the record's line spectrum at a bin into the part that reproduces
    from field to field, S, and the part that does not, U. The record self
    cross-spectrum keeps only the first, so beta = |S|^2 / sqrt(Sxx Syy) and
    |U|^2 / |S|^2 = (1 - beta) / beta. The numerator holds H |S|^2 plus
    whatever of U survives its own pairing, which is at most |H| |U|^2 in
    magnitude, so the ratio lies within a factor 1 + (1 - beta)/beta = 1/beta
    of the truth: a log-magnitude bound of -log(beta) and a phase bound of
    arcsin(min(1, (1 - beta)/beta)).

    The bound is nearly attained whenever the non-reproducing part is not
    noise but a coherent component under the wrong rotation. Measured on the
    synthetic taps of `tests/unit/test_tap_transfer.py`, where the
    colour-under's 90-degree-per-line rotation and the luma FM's own share
    the bins just above the chroma band's top: beta = 0.39 at 1.58 MHz, the
    bound is 8.2 dB and the estimator's actual error there is 7.2 dB."""
    ratio = (1.0 - np.minimum(beta, 1.0)) / np.maximum(beta, 1e-300)
    return np.log1p(ratio), np.arcsin(np.minimum(ratio, 1.0))


def _delay_track_summary(tracks: List[Dict[int, float]], first_line: int,
                         last_line: int, sample_rate_hz: float
                         ) -> Dict[str, object]:
    """The playback time base against the record's, per line.

    Each field's track has its own arbitrary offset (the two captures are
    different frames), so each is centred on its own usable-line median.
    The largest line-to-line step per field is where the head switch's
    timing discontinuity falls; the drift across the usable lines is the
    time-base error over a field."""
    steps, drifts, jitters, switch_lines = [], [], [], []
    for track in tracks:
        if len(track) < 4:
            continue
        numbers = np.array(sorted(track))
        values = np.array([track[n] for n in numbers], dtype=np.float64)
        usable = (numbers >= first_line) & (numbers <= last_line)
        if usable.sum() < 4:
            continue
        values = values - float(np.median(values[usable]))
        drifts.append(float(values[usable][-1] - values[usable][0]))
        jitters.append(float(np.std(np.diff(values[usable]))))
        differences = np.abs(np.diff(values))
        k = int(np.argmax(differences))
        steps.append(float(differences[k]))
        switch_lines.append(int(numbers[k + 1]))
    if not drifts:
        return {"fields": 0}
    fs = float(sample_rate_hz)
    return {
        "fields": len(drifts),
        "drift_over_field_ns": float(np.median(drifts)) / fs * 1e9,
        "line_step_rms_ns": float(np.median(jitters)) / fs * 1e9,
        "largest_step_ns": float(np.median(steps)) / fs * 1e9,
        "largest_step_line": int(np.median(switch_lines)),
        "largest_step_lines": switch_lines,
    }


def complex_transfer(record: np.ndarray, playback: np.ndarray,
                     sample_rate_hz: float,
                     params: Optional[Dict[str, object]] = None,
                     record_fields: Optional[List[Field]] = None,
                     playback_fields: Optional[List[Field]] = None,
                     minimum_line_coherence: Optional[float] = None
                     ) -> Dict[int, TapTransfer]:
    """THE COMPLEX TRANSFER between the two taps, per head.

    `record` and `playback` are the two captures (or windows of them) at
    `sample_rate_hz`; `params` from `capture_parameters`. Fields are located
    from the sync structure in each, lines are paired at equal line numbers
    and aligned by whitened cross-correlation, and

        H_k(f) = Sxy(record field j(k) -> playback field k)
                 / Sxy(record -> record, pooled over far-apart same-parity pairs)

    per playback field k, with j(k) a record field of the same parity taken
    in rotation. The head is the parity; the statistics are over the
    playback fields of that parity. The phase is the excess phase under
    `_phase_convention`, with the SAME linear term applied to every field of
    a head so the per-field scatter is the channel's and not the fit's.

    THE ERROR BAR MUST CARRY THE DENOMINATOR. `se_log_magnitude` is not the
    scatter of the per-field numerators alone. The denominator is COMMON to
    every field of a head, so nothing it does appears in a scatter across
    fields, and the bins where the record tap does not reproduce itself - the
    bins where the estimate is worst - were the bins the scatter called best.
    Measured on the synthetic taps of `tests/unit/test_tap_transfer.py`, the
    field scatter read 0.06 dB at 1.575 MHz where the estimate was 7.2 dB
    wrong, and admitting bins on that scatter alone left an error of 2.0 dB
    rms against a channel recovered to 0.3 dB rms elsewhere. The reported
    error is therefore four terms, each kept separately on the result: the LARGER of the numerators' scatter over fields and the random
    error the measured coherence and the line count imply (`_random_error`) -
    the two being estimates of the same thing, one on three or four samples
    and one on every line - then in quadrature the denominator's own scatter
    over the record self-pairs (`_reference_scatter`) and the bound the record
    tap's coherent fraction puts on the ratio (`_incoherent_bounds`). The same
    four, banded, are in `band_summary`.

    MISALIGNMENT WITNESS. A line pair whose whitened correlation peaks
    beyond one sync width is rejected; a field pair whose mean line
    coherence over the luma FM band falls below `minimum_line_coherence`
    (default: half the record tap's own line-to-line coherence, measured
    here from the record self-pairs) is rejected as not the same content.
    If no field pair survives, ValueError: the captures are not the same
    pattern, or the sync structure was not found.
    """
    params = params or capture_parameters()
    grid = make_grid(sample_rate_hz, params)
    first, last = int(params["first_usable_line"]), int(params["last_usable_line"])
    track = (first, max(params["field_lines"]) - 1)
    if record_fields is None:
        record_fields = locate_fields(record, sample_rate_hz, params)
    if playback_fields is None:
        playback_fields = locate_fields(playback, sample_rate_hz, params)
    if len(record_fields) < 3 or len(playback_fields) < 1:
        raise ValueError("too few fields located: record %d, playback %d"
                         % (len(record_fields), len(playback_fields)))

    # the denominator: the record tap against itself, far apart
    separation = max(2, (len(record_fields) // 2) - (len(record_fields) // 2) % 2)
    pairs = same_parity_pairs(record_fields, separation)
    if not pairs:
        pairs = same_parity_pairs(record_fields, 2)
    nb = len(grid.freqs)
    self_xy = np.zeros(nb, np.complex128)
    self_xx = np.zeros(nb)
    self_yy = np.zeros(nb)
    self_lines = 0
    self_coherence = []
    reference_pairs: List[np.ndarray] = []
    reference_fields = set()
    for i, j in pairs:
        cs = field_pair_cross_spectrum(record, record_fields[i], record,
                                       record_fields[j], grid, first, last)
        if cs.lines == 0:
            continue
        self_xy += cs.Sxy
        self_xx += cs.Sxx
        self_yy += cs.Syy
        self_lines += cs.lines
        reference_pairs.append(cs.Sxy / cs.lines)
        reference_fields.update((i, j))
        self_coherence.append(float(np.mean(list(cs.coherences.values()))))
    if not self_coherence:
        raise ValueError("no record self-pairs aligned: sync structure not found")
    ceiling = float(np.mean(self_coherence))
    if minimum_line_coherence is None:
        minimum_line_coherence = 0.5 * ceiling
    # the coherent fraction against the two fields' own powers, so it is the
    # magnitude coherence and cannot exceed one (Cauchy-Schwarz); dividing by
    # the first field's power alone let it read 1.001
    beta = np.abs(self_xy) / np.maximum(np.sqrt(self_xx * self_yy), 1e-300)
    # PER LINE, both sides. The denominator sums over every self-pair's every
    # line and a numerator over one field pair's lines, so the raw ratio is
    # low by the number of self-pairs - nine on the synthetic taps of
    # `tests/unit/test_tap_transfer.py`, about fifty on a real capture. It
    # cancels in every referenced comparison, which is why only an absolute
    # one finds it: there an identity channel read -18.8 dB, 1/9, before this
    # division.
    reference = self_xy / max(self_lines, 1)
    reference_per_pair = np.array(reference_pairs)
    independent = len(reference_fields)
    se_reference_log, se_reference_phase = _reference_scatter(
        reference_per_pair, independent)
    bound_log, bound_phase = _incoherent_bounds(beta)

    out: Dict[int, TapTransfer] = {}
    for head in (0, 1):
        candidates = [f for f in record_fields if f.parity == head]
        targets = [f for f in playback_fields if f.parity == head]
        if not candidates or not targets:
            continue
        numerators, tracks, coherences = [], [], []
        pooled = CrossSpectrum(np.zeros(nb, np.complex128), np.zeros(nb),
                               np.zeros(nb), 0)
        rejected = 0
        for k, target in enumerate(targets):
            source = candidates[k % len(candidates)]
            cs = field_pair_cross_spectrum(record, source, playback, target,
                                           grid, first, last, track)
            if cs.lines == 0:
                continue
            mean_coherence = float(np.mean(
                [cs.coherences[n] for n in cs.coherences if first <= n <= last]))
            if mean_coherence < minimum_line_coherence:
                rejected += cs.lines
                continue
            numerators.append(cs.Sxy / cs.lines)
            tracks.append(cs.delays)
            coherences.append(mean_coherence)
            pooled.Sxy += cs.Sxy
            pooled.Sxx += cs.Sxx
            pooled.Syy += cs.Syy
            pooled.lines += cs.lines
            pooled.rejected += cs.rejected
        if not numerators:
            continue
        per_field = np.array(numerators) / np.maximum(np.abs(reference), 1e-300) \
            * np.exp(-1j * np.angle(reference))
        pooled_ratio = (pooled.Sxy / max(pooled.lines, 1)) / reference
        H_pooled, reference_delay = _phase_convention(
            pooled_ratio, grid.freqs, params["reference_band_hz"])
        H_pooled, chroma_constant = _chroma_phase_convention(
            H_pooled, grid.freqs, params["chroma_band_hz"], np.abs(reference))
        # the same convention for every field, as a unit-modulus factor so a
        # bin where the pooled ratio vanishes cannot divide by zero
        linear = np.exp(1j * (np.angle(H_pooled) - np.angle(pooled_ratio)))
        per_field = per_field * linear[None, :]
        n = per_field.shape[0]
        log_mag = np.log(np.maximum(np.abs(per_field), 1e-300))
        se_log_fields = log_mag.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 \
            else np.full(nb, np.inf)
        phase, se_phase_fields = _circular_stats(per_field)
        gamma2 = np.abs(pooled.Sxy) ** 2 / np.maximum(pooled.Sxx * pooled.Syy, 1e-300)
        random = _random_error(gamma2, pooled.lines)
        # the field scatter and the coherence term are two routes to the SAME
        # random error - one empirical on three or four fields, one from every
        # line pair - so the LARGER is taken, not the quadrature sum, which
        # would charge the same noise twice. The denominator's scatter and the
        # coherent-fraction bound are different quantities and do add.
        se_log = np.sqrt(np.maximum(se_log_fields, random) ** 2
                         + se_reference_log ** 2 + bound_log ** 2)
        se_phase = np.sqrt(np.maximum(se_phase_fields, random) ** 2
                           + se_reference_phase ** 2 + bound_phase ** 2)
        out[head] = TapTransfer(
            head=head, label=HEAD_LABELS[head], freqs=grid.freqs,
            transfer=H_pooled, per_field=per_field,
            log_magnitude=log_mag.mean(axis=0), se_log_magnitude=se_log,
            phase=phase, se_phase=se_phase, coherence=gamma2,
            beta_record=beta,
            record_coherent_power=np.abs(reference),
            power_ratio=pooled.Syy / np.maximum(pooled.Sxx, 1e-300),
            fields=n, lines=pooled.lines, rejected_lines=rejected + pooled.rejected,
            line_coherence=float(np.mean(coherences)),
            delay_track=_delay_track_summary(tracks, first, last, sample_rate_hz),
            group_delay_reference_s=reference_delay,
            reference_band_hz=tuple(params["reference_band_hz"]),
            se_log_magnitude_fields=se_log_fields,
            se_log_magnitude_reference=se_reference_log,
            bound_log_magnitude_incoherent=bound_log,
            se_phase_fields=se_phase_fields,
            se_phase_reference=se_reference_phase,
            bound_phase_incoherent=bound_phase,
            reference=reference, reference_per_pair=reference_per_pair,
            reference_independent=independent,
            se_log_magnitude_random=random, se_phase_random=random,
            chroma_phase_constant_rad=chroma_constant)
    if not out:
        raise ValueError(
            "no field pair reached the coherence floor %.3f (record self "
            "ceiling %.4f): the captures are not the same content, or they "
            "are misaligned" % (minimum_line_coherence, ceiling))
    for transfer in out.values():
        transfer.delay_track["record_self_coherence"] = ceiling
    return out


# --------------------------------------------------------------------------
# Banding, and the per-band statistics from repeated fields
# --------------------------------------------------------------------------


@dataclass
class BandValue:
    low_hz: float
    high_hz: float
    magnitude_db: float
    se_magnitude_db: float
    phase_deg: float
    se_phase_deg: float
    coherence: float
    beta_record: float          # the band's energy-weighted coherent fraction
    power_ratio_db: float
    group_delay_ns: float
    bins: int
    # the four terms of the band's error bar, as `complex_transfer` reports
    # them per bin, banded on the same energy weighting
    se_magnitude_db_fields: float
    se_magnitude_db_reference: float
    se_magnitude_db_random: float
    bound_magnitude_db_incoherent: float
    se_phase_deg_fields: float
    bound_phase_deg_incoherent: float


def band_summary(transfer: TapTransfer,
                 bands: Optional[Sequence[Tuple[float, float]]] = None
                 ) -> List[BandValue]:
    """Per band: the energy-weighted complex value per field, then the mean
    and SE across fields; magnitudes referenced to the reference band.

    Energy weighting within a band (the sum of the per-bin cross-spectra
    over the sum of the record self cross-spectra, which is what dividing
    the pooled cross-spectra bin by bin and then weighting by the record's
    coherent power amounts to) is the unbiased band estimate - the same
    reason the export banded power rather than decibels.

    The error bar carries the same four terms as the per-bin one: the
    numerators' scatter over fields, the denominator's scatter over the
    record self-pairs - banded through the SAME energy weighting, so a band
    whose bins are individually uncertain but jointly well determined is not
    charged twice - the per-bin random error propagated through those same
    weights, and the bound from the band's energy-weighted coherent fraction.
    See `complex_transfer`, THE ERROR BAR MUST CARRY THE DENOMINATOR.

    The random term treats the bins as independent, which is mildly
    optimistic: the analysis window is one line long, so the instrument's
    resolution is the line rate while the grid is 1.29 times finer (3178
    samples of line zero-padded to 4096), and neighbouring bins share
    information."""
    bands = list(bands or reporting_bands())
    f = transfer.freqs
    ref = (f >= transfer.reference_band_hz[0]) & (f < transfer.reference_band_hz[1])
    # the weight is the record's coherent power per bin, |Sxy(record->record)|,
    # which `per_field` was divided by: the sum recovers the band's ratio of
    # summed cross-spectra
    weight = transfer.record_coherent_power
    out = []
    ref_value = np.array([np.sum(weight[ref] * row[ref]) / np.sum(weight[ref])
                          for row in transfer.per_field])
    ref_mag = np.mean(np.log(np.abs(ref_value)))
    unwrapped = np.unwrap(np.angle(transfer.transfer))
    # the denominator banded the same way: each record self-pair's own band
    # value, on the pooled denominator's phase
    pairs = transfer.reference_per_pair
    pooled_reference = pairs.mean(axis=0) if pairs.size else np.zeros_like(f)
    derotate = np.exp(-1j * np.angle(pooled_reference))
    for low, high in bands:
        inside = (f >= low) & (f < high)
        if not inside.any():
            continue
        per_field = np.array([np.sum(weight[inside] * row[inside])
                              / max(np.sum(weight[inside]), 1e-300)
                              for row in transfer.per_field])
        n = len(per_field)
        log_mag = np.log(np.maximum(np.abs(per_field), 1e-300)) - ref_mag
        se_log_fields = log_mag.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
        unit = per_field / np.maximum(np.abs(per_field), 1e-300)
        mean_phase = float(np.angle(unit.mean()))
        residual = np.angle(unit * np.exp(-1j * mean_phase))
        se_phase_fields = residual.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
        per_pair = (pairs[:, inside] * derotate[inside]).sum(axis=1) \
            / max(float(np.sum(np.abs(pooled_reference[inside]))), 1e-300)
        se_reference, se_reference_phase = _reference_scatter(
            per_pair[:, None], transfer.reference_independent)
        se_reference = float(se_reference[0])
        se_reference_phase = float(se_reference_phase[0])
        # the random error through the same weights, bins treated as
        # independent
        band_weight = weight[inside]
        total_weight = max(float(np.sum(band_weight)), 1e-300)
        band_random = float(np.sqrt(np.sum(
            (band_weight * transfer.se_log_magnitude_random[inside]) ** 2))
            / total_weight)
        band_beta = float(np.sum(weight[inside] * transfer.beta_record[inside])
                          / max(np.sum(weight[inside]), 1e-300))
        band_bound_log, band_bound_phase = _incoherent_bounds(
            np.array([band_beta]))
        band_bound_log = float(band_bound_log[0])
        band_bound_phase = float(band_bound_phase[0])
        # the larger of the two routes to the random error, as per bin
        se_log = float(np.sqrt(max(se_log_fields, band_random) ** 2
                               + se_reference ** 2 + band_bound_log ** 2))
        se_phase = float(np.sqrt(max(se_phase_fields, band_random) ** 2
                                 + se_reference_phase ** 2 + band_bound_phase ** 2))
        slope, _ = _linear_phase_fit(f, unwrapped, inside) if inside.sum() > 2 \
            else (np.nan, 0.0)
        out.append(BandValue(
            low_hz=float(low), high_hz=float(high),
            magnitude_db=float(log_mag.mean() * DB_PER_NEPER),
            se_magnitude_db=float(se_log * DB_PER_NEPER),
            phase_deg=float(np.degrees(mean_phase)),
            se_phase_deg=float(np.degrees(se_phase)),
            coherence=float(np.mean(transfer.coherence[inside])),
            beta_record=band_beta,
            power_ratio_db=float(10.0 * np.log10(
                np.sum(transfer.power_ratio[inside] * 1.0) / inside.sum() + 1e-300)
                - 10.0 * np.log10(np.mean(transfer.power_ratio[ref]) + 1e-300)),
            group_delay_ns=float(-slope / (2.0 * np.pi) * 1e9),
            bins=int(inside.sum()),
            se_magnitude_db_fields=float(se_log_fields * DB_PER_NEPER),
            se_magnitude_db_reference=float(se_reference * DB_PER_NEPER),
            se_magnitude_db_random=float(band_random * DB_PER_NEPER),
            bound_magnitude_db_incoherent=float(band_bound_log * DB_PER_NEPER),
            se_phase_deg_fields=float(np.degrees(se_phase_fields)),
            bound_phase_deg_incoherent=float(np.degrees(band_bound_phase))))
    return out


# --------------------------------------------------------------------------
# Separation: what is common to every pattern, and what depends on content
# --------------------------------------------------------------------------


@dataclass
class Separation:
    """Per head: the common transfer across patterns and the content
    dependence beyond the per-pattern standard errors."""
    head: int
    label: str
    freqs: np.ndarray
    common_log_magnitude: np.ndarray
    se_common_log_magnitude: np.ndarray
    common_phase: np.ndarray
    se_common_phase: np.ndarray
    content_chi2_magnitude: np.ndarray     # per bin, chi^2 / dof across patterns
    content_chi2_phase: np.ndarray
    content_excess_nepers: np.ndarray      # rms scatter beyond SE, per bin
    content_excess_radians: np.ndarray
    patterns_per_bin: np.ndarray
    patterns: List[str]


def _combine(values: np.ndarray, errors: np.ndarray, finite: np.ndarray,
             extra: np.ndarray, circular: bool
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The inverse-variance mean with `extra` scatter added to every error,
    its residuals and the weight total."""
    weight = np.where(finite, 1.0 / (np.where(finite, errors, 1.0) ** 2
                                     + extra[None, :] ** 2), 0.0)
    total = weight.sum(axis=0)
    clean = np.nan_to_num(values)
    if circular:
        unit = np.where(finite, np.exp(1j * clean), 0.0)
        mean = np.angle((weight * unit).sum(axis=0))
        residual = np.where(finite, np.angle(np.exp(1j * (clean - mean[None, :]))), 0.0)
    else:
        mean = np.where(total > 0, (weight * clean).sum(axis=0)
                        / np.maximum(total, 1e-300), np.nan)
        residual = np.where(finite, clean - mean[None, :], 0.0)
    return mean, residual, total


def _weighted_common(values: np.ndarray, errors: np.ndarray, circular: bool
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance mean across patterns per bin, its SE, the reduced
    chi-square and the excess scatter beyond the SEs.

    THE EXCESS IS THE SCATTER THAT MAKES THE CHI-SQUARE COME OUT AT ONE, not
    the between-pattern variance minus the mean of the squared error bars.
    The difference matters whenever the patterns are measured to different
    precisions, which here they always are: on the sixteen real SP patterns
    the per-pattern bars at 1.7-2.0 MHz run from hundredths of a dB on the
    luma-only patterns to 6 dB on the ones carrying colour-under, so the mean
    of the squared bars was 30 times the actual spread and the subtraction
    floored at zero - a band whose reduced chi-square was 34 reported no
    content dependence at all. Solving for the scatter that would make the
    weighted chi-square one is insensitive to that heterogeneity: a pattern
    with a 6 dB bar carries almost no weight in either quantity. With equal
    bars the two definitions agree.

    The mean and its standard error are then taken WITH that scatter in the
    weights, which is the same convention: where the patterns disagree beyond
    their bars, the common value is not entitled to the precision of its best
    member."""
    finite = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    count = finite.sum(axis=0)
    bins = values.shape[1]
    zero = np.zeros(bins)
    mean, residual, total = _combine(values, errors, finite, zero, circular)
    chi2 = ((1.0 / np.where(finite, errors, 1.0) ** 2) * residual ** 2
            * finite).sum(axis=0)
    dof = np.maximum(count - 1, 1)
    reduced = np.where(count > 1, chi2 / dof, np.nan)

    def reduced_chi2(extra):
        m, r, _ = _combine(values, errors, finite, extra, circular)
        w = np.where(finite, 1.0 / (np.where(finite, errors, 1.0) ** 2
                                    + extra[None, :] ** 2), 0.0)
        return (w * r ** 2).sum(axis=0) / dof

    # bisect for the scatter that brings the reduced chi-square to one; it
    # falls monotonically with the scatter, so a bracket is enough
    low = np.zeros(bins)
    high = np.where(count > 1, np.max(np.abs(residual), axis=0), 0.0)
    for _ in range(8):
        needs = (reduced_chi2(high) > 1.0) & (count > 1)
        if not needs.any():
            break
        high = np.where(needs, np.maximum(high * 2.0, 1e-12), high)
    for _ in range(40):
        middle = 0.5 * (low + high)
        over = reduced_chi2(middle) > 1.0
        low = np.where(over, middle, low)
        high = np.where(over, high, middle)
    excess = np.where((count > 1) & (reduced > 1.0), 0.5 * (low + high), 0.0)
    mean, residual, total = _combine(values, errors, finite, excess, circular)
    se = np.where(total > 0, 1.0 / np.sqrt(np.maximum(total, 1e-300)), np.nan)
    return mean, se, reduced, excess, count


def separate(transfers: Dict[str, Dict[int, TapTransfer]]) -> Dict[int, Separation]:
    """What is common to all patterns and what varies with them.

    `transfers` maps pattern name -> head -> TapTransfer, all at one speed.
    For a linear time-invariant chain the transfer is the same whatever
    signal carries it, so the between-pattern scatter beyond the per-pattern
    standard errors is the NONLINEARITY WITNESS: content dependence at a
    frequency means the mechanism carrying that frequency differs between
    patterns - and there is a physical reason to expect one, because the
    luma FM is recorded in saturation while the colour-under rides on it as
    on a bias, so the FM's lower sideband and the chroma's upper sideband
    reach the same RF frequency by different mechanisms.

    Magnitudes are compared as each pattern's own referenced shape (the
    scope's scale differs per capture); phases as excess phase under the
    common convention. The colour-under band's phase carries a per-line
    rotation of its own, so only its slope is comparable there and its
    chi-square is reported for what it is."""
    out: Dict[int, Separation] = {}
    names = sorted(transfers)
    for head in (0, 1):
        rows = [(name, transfers[name][head]) for name in names
                if head in transfers[name]]
        if not rows:
            continue
        freqs = rows[0][1].freqs
        ref = rows[0][1].reference_band_hz
        band = (freqs >= ref[0]) & (freqs < ref[1])
        mags, mag_se, phases, ph_se = [], [], [], []
        for _, t in rows:
            # reference each pattern's magnitude to its own reference band
            level = float(np.mean(t.log_magnitude[band]))
            mags.append(t.log_magnitude - level)
            mag_se.append(t.se_log_magnitude)
            phases.append(t.phase)
            ph_se.append(t.se_phase)
        mags, mag_se = np.array(mags), np.array(mag_se)
        phases, ph_se = np.array(phases), np.array(ph_se)
        m_mean, m_se, m_chi, m_excess, count = _weighted_common(mags, mag_se, False)
        p_mean, p_se, p_chi, p_excess, _ = _weighted_common(phases, ph_se, True)
        out[head] = Separation(
            head=head, label=HEAD_LABELS[head], freqs=freqs,
            common_log_magnitude=m_mean, se_common_log_magnitude=m_se,
            common_phase=p_mean, se_common_phase=p_se,
            content_chi2_magnitude=m_chi, content_chi2_phase=p_chi,
            content_excess_nepers=m_excess, content_excess_radians=p_excess,
            patterns_per_bin=count, patterns=[n for n, _ in rows])
    return out


def band_separation(separation: Separation,
                    bands: Optional[Sequence[Tuple[float, float]]] = None
                    ) -> List[Dict[str, float]]:
    """The separation's figures per reporting band, each bin weighted by the
    information it carries.

    A BIN THAT MEASURED NOTHING MUST NOT SET THE BAND'S WITNESS. Averaged
    flat, the content-dependence figure of a band is decided by its worst
    bin: measured on the synthetic taps of `tests/unit/test_tap_transfer.py`,
    where two patterns were given IDENTICAL channels below 4 MHz, the
    2.0-3.0 MHz band read 0.69 dB of content dependence, of which 0.66 dB
    came from the single bin at 2.710 MHz - the bottom of the planted notch,
    coherence 0.023 and 0.008, where the two patterns' own error bars were
    1.3 and 5.3 dB. The weight here is the inverse variance of the common
    value, 1 / se_common^2, which is what an inverse-variance mean already
    uses bin by bin inside `_weighted_common`; carrying it into the band
    keeps one uninformative bin from speaking for fifty informative ones.
    The chi-squares stay medians, which are robust already."""
    bands = list(bands or reporting_bands())
    f = separation.freqs
    out = []
    for low, high in bands:
        inside = (f >= low) & (f < high) & np.isfinite(separation.common_log_magnitude)
        inside = inside & np.isfinite(separation.se_common_log_magnitude) \
            & (separation.se_common_log_magnitude > 0)
        if not inside.any():
            continue
        se = separation.se_common_log_magnitude[inside]
        weight = 1.0 / se ** 2
        total = float(np.sum(weight))

        def mean(values):
            return float(np.sum(weight * values) / total)

        def rms(values):
            return float(np.sqrt(np.sum(weight * values ** 2) / total))

        out.append({
            "low_hz": float(low), "high_hz": float(high),
            "common_db": mean(separation.common_log_magnitude[inside]) * DB_PER_NEPER,
            # a typical bin's bar, on the same weights - not the standard
            # error of the band's mean, which would treat neighbouring bins
            # of a one-line-resolution instrument as independent
            "se_common_db": rms(se) * DB_PER_NEPER,
            "common_phase_deg": float(np.degrees(mean(separation.common_phase[inside]))),
            "content_excess_db": rms(separation.content_excess_nepers[inside]) * DB_PER_NEPER,
            "content_excess_deg": float(np.degrees(rms(np.nan_to_num(
                separation.content_excess_radians[inside])))),
            "chi2_magnitude": float(np.nanmedian(separation.content_chi2_magnitude[inside])),
            "chi2_phase": float(np.nanmedian(separation.content_chi2_phase[inside])),
            "patterns": int(np.min(separation.patterns_per_bin[inside])),
        })
    return out


# --------------------------------------------------------------------------
# Attribution against the two-stage model
# --------------------------------------------------------------------------


def minimum_phase_prediction(freqs: np.ndarray, log_magnitude: np.ndarray,
                             valid: np.ndarray) -> np.ndarray:
    """The excess phase a MINIMUM-PHASE response of this magnitude would
    have, on the measured band, with the same linear term removed.

    Outside the measured band the magnitude is unknown; it is held at the
    edge values, and the prediction is trusted away from the edges only."""
    from scipy.signal import hilbert

    shape = np.array(log_magnitude, dtype=np.float64)
    if valid.any():
        first, last = int(np.argmax(valid)), len(valid) - 1 - int(np.argmax(valid[::-1]))
        shape[:first] = shape[first]
        shape[last:] = shape[last]
        inside = np.arange(len(shape))
        shape = np.interp(inside, inside[valid], shape[valid])
    # a real spectrum's log magnitude is even in frequency; the transform
    # over the two-sided grid keeps the phase odd, as a real filter's is
    two_sided = np.concatenate([shape[:0:-1], shape])
    phase = -np.imag(hilbert(two_sided))[len(shape) - 1:]
    return phase


def stage_attribution(separation: Separation,
                      params: Optional[Dict[str, object]] = None,
                      minimum_coherent_share: float = 0.5
                      ) -> Dict[str, object]:
    """WHICH STAGE THE MEASURED SHAPE BELONGS TO, against `rf_stages`, and
    what one deck cannot separate.

    The record tap sits at the record head's terminals, so nothing before
    it - the emphases, the clipping, the modulator, the record amplifier's
    shaping of the drive - is in the transfer at all (`rf_stages
    .tap_separation`). What the transfer holds is the RF stage chain from
    the head drive to the playback amplifier's output, and `rf_stages`
    already measured which of its members are one direction:

      * record head transition loss, playback spacing loss and the tape's
        surface spacing compose in one exponent, exp(-2 pi (a + d)/lambda):
        only the SUM is measurable (coherence 1.000000000);
      * the record amplifier's upper corner and the playback preamplifier's
        are collinear to 0.9994: a band limit at either end is one direction;
      * gains - remanence, head efficiency, preamplifier gain, the scope's
        scale - are one constant, removed by the reference band;
      * the playback head's differentiation (j omega) has no record-side
        counterpart, and the playback equaliser's in-band corner is the
        other one-sided entry (coherence at most 0.65 with anything).

    So the attribution regresses the common log magnitude on the RF-side
    entries in their fused form, and reports what each explains, with the
    per-head DIFFERENCE attributed to what is not shared: the record
    electronics are common to both heads through the one tap, so a head
    difference is head-plus-track (write process, read process, contact),
    fused per head. Two things this deck cannot separate are stated as
    such: the record head's write from the playback head's read, being the
    same physical head, and the tape's spacing from the head's.

    The minimum-phase test is the one attribution the phase adds: a chain
    of losses and filters is minimum phase, so measured excess phase beyond
    the Hilbert transform of the measured magnitude is a non-minimum-phase
    element - dispersion between bands that a loss cannot produce.
    """
    params = params or capture_parameters()
    f = separation.freqs
    # ONLY WHERE THE DECK PUTS SIGNAL ON THE TAPE. The grid runs to Nyquist,
    # 25 MHz, and outside the recorded band there is nothing to measure: on
    # 75bars SP those bins carried the whole median that the precision rule
    # below is scaled by, so every bad bin passed it, the phase was unwrapped
    # across them and the excess phase read 2273 degrees rms. The band is the
    # decoder's own: the colour-under band's floor to the luma FM band's top.
    recorded = ((f >= float(params["chroma_band_hz"][0]))
                & (f <= float(params["fm_band_hz"][1])))
    valid = (np.isfinite(separation.common_log_magnitude)
             & (separation.patterns_per_bin > 0)
             & (f > 0) & recorded)
    # restrict to bins measured with usable precision
    se = separation.se_common_log_magnitude
    if valid.any():
        precise = np.isfinite(se) & (se < float(np.nanmedian(se[valid])) * 4.0)
        valid = valid & precise
    if valid.sum() < 8:
        return {"fitted": False, "why": "too few measured bins"}
    rf = f[valid]
    target = separation.common_log_magnitude[valid]
    mechanics = params["mechanics"]
    writing = head_model.writing_speed(mechanics)
    # the fused directions, as log-magnitude rows
    rows = {
        "wavelength exponent (record transition + playback spacing + tape surface, the SUM)":
            np.log(np.abs(rf_stages.record_head_write(rf, 0.10e-6, writing, minimum_phase=False))),
        "playback head differentiation (j omega)":
            np.log(np.abs(rf_stages.playback_head_differentiation(rf))),
        "playback head gap and thickness losses (head_model.TYPICAL, no spacing)":
            np.log(np.abs(rf_stages.playback_head_losses(
                rf, mechanics, minimum_phase=False, spacing_m=0.0))),
        "band limit (record amplifier top = playback preamplifier top, ONE direction)":
            np.log(np.abs(rf_stages.playback_preamplifier(rf))),
        "playback equalisation (in-band zero)":
            np.log(np.abs(rf_stages.playback_equalisation(rf))),
    }
    names = list(rows)
    design = np.vstack([np.ones_like(rf)] + [rows[n] - rows[n].mean() for n in names]).T
    weights = 1.0 / np.maximum(se[valid], 1e-6)
    coefficients, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
    fitted = design @ coefficients
    residual = target - fitted
    explained = 1.0 - float(np.sum(residual ** 2) / max(np.sum((target - target.mean()) ** 2), 1e-300))
    count = rf_stages.effective_count([rows[n] for n in names], demean=True)
    # the minimum-phase test
    predicted = minimum_phase_prediction(f, np.where(valid, separation.common_log_magnitude, np.nan), valid)
    ref = params["reference_band_hz"]
    band = (f >= ref[0]) & (f < ref[1])
    # unwrap ONLY across the measured bins: a run of unmeasured ones between
    # two measured ones makes np.unwrap manufacture turns that were never in
    # the data
    measured = np.array(separation.common_phase, dtype=np.float64)
    measured[~np.isfinite(measured)] = 0.0
    if valid.any():
        measured[valid] = np.unwrap(measured[valid])
    slope_m, icpt_m = _linear_phase_fit(f, measured, band)
    slope_p, icpt_p = _linear_phase_fit(f, predicted, band)
    excess_measured = measured - (slope_m * f + icpt_m)
    excess_predicted = predicted - (slope_p * f + icpt_p)
    interior = valid.copy()
    # trust the prediction away from the band edges: drop the outer eighth
    # of the valid span at each end
    where = np.flatnonzero(valid)
    margin = max(len(where) // 8, 1)
    interior[where[:margin]] = False
    interior[where[-margin:]] = False
    non_minimum = excess_measured - excess_predicted
    return {
        "fitted": True,
        "entries": names,
        "coefficients": {n: float(c) for n, c in zip(names, coefficients[1:])},
        "explained_fraction": explained,
        "residual_rms_nepers": float(np.sqrt(np.mean(residual ** 2))),
        "residual_rms_db": float(np.sqrt(np.mean(residual ** 2)) * DB_PER_NEPER),
        "effective_directions": float(count["effective"]),
        "of": int(count["count"]),
        "condition": float(count["condition"]),
        "non_minimum_phase_rms_deg": float(np.degrees(np.sqrt(np.mean(non_minimum[interior] ** 2)))),
        "excess_phase_measured_rms_deg": float(np.degrees(np.sqrt(np.mean(excess_measured[interior] ** 2)))),
        "excess_phase_predicted": excess_predicted,
        "excess_phase_measured": excess_measured,
        "interior": interior,
        "not_in_the_transfer": ("the record emphases, the clipping, the FM "
                                "modulator and the record amplifier's drive "
                                "shaping: all before the record tap"),
        "head_difference_is": ("head plus track, fused: the record head's "
                               "write, the tape track, the playback head's "
                               "read and its contact; NOT the record "
                               "electronics, which both heads share through "
                               "the one tap"),
        "cannot_separate_with_one_deck": (
            "the record head's write process from the playback head's read "
            "process (the same physical head, the same gap, one exponent); "
            "the tape's surface spacing from the head's own; the record "
            "amplifier's band limit from the preamplifier's; any gain from "
            "any other gain"),
    }


# --------------------------------------------------------------------------
# Comparisons with the two magnitude-only measurements
# --------------------------------------------------------------------------


def compare_with_export(transfers: Dict[int, TapTransfer],
                        export_path: str = SHARED_EXPORT
                        ) -> Optional[Dict[str, object]]:
    """The coherent magnitude against the 2026-09-02 power-ratio export,
    band by band, with this module's OWN power ratio on the same data
    beside it - so the disagreement is attributed to the estimator rather
    than to the captures."""
    if not os.path.exists(export_path):
        return None
    with np.load(export_path) as export:
        bands = [(float(a) * 1e6, float(b) * 1e6) for a, b in export["bands"]]
        exported = np.asarray(export["transfer_db"], dtype=np.float64)
        signals = [str(s) for s in export["signals"]]
    rows = []
    heads = sorted(transfers)
    summaries = {h: band_summary(transfers[h], bands) for h in heads}
    for index, (low, high) in enumerate(bands):
        coherent = [next((v.magnitude_db for v in summaries[h] if v.low_hz == low), np.nan) for h in heads]
        power = [next((v.power_ratio_db for v in summaries[h] if v.low_hz == low), np.nan) for h in heads]
        gamma = [next((v.coherence for v in summaries[h] if v.low_hz == low), np.nan) for h in heads]
        rows.append({
            "low_hz": low, "high_hz": high,
            "export_mean_db": float(exported[:, index].mean()),
            "export_spread_db": float(exported[:, index].std()),
            "coherent_db": {HEAD_LABELS[h]: v for h, v in zip(heads, coherent)},
            "own_power_ratio_db": {HEAD_LABELS[h]: v for h, v in zip(heads, power)},
            "coherence": {HEAD_LABELS[h]: v for h, v in zip(heads, gamma)},
        })
    return {"signals": signals, "rows": rows,
            "why": ("a power ratio is |H|^2 / coherence: wherever the "
                    "playback holds power that is not a copy of the record "
                    "it reads the floor as response")}


def slope_db_per_octave(transfer: TapTransfer, low_hz: float, high_hz: float
                        ) -> Tuple[float, float]:
    """The magnitude slope over a band in dB per octave, with its SE from
    the per-field scatter - the figure the depth agent quoted."""
    f = transfer.freqs
    inside = (f >= low_hz) & (f <= high_hz) & (transfer.beta_record > 0)
    x = np.log2(f[inside] / 1e6)
    slopes = []
    for row in transfer.per_field:
        y = DB_PER_NEPER * np.log(np.maximum(np.abs(row[inside]), 1e-300))
        w = transfer.beta_record[inside]
        columns = np.vstack([x, np.ones_like(x)]).T
        slope = np.linalg.lstsq(columns * w[:, None], y * w, rcond=None)[0][0]
        slopes.append(slope)
    slopes = np.array(slopes)
    return float(slopes.mean()), float(slopes.std(ddof=1) / np.sqrt(len(slopes))) if len(slopes) > 1 else np.inf


def slope_db_per_mhz(transfer: TapTransfer, low_hz: float, high_hz: float
                     ) -> Tuple[float, float]:
    f = transfer.freqs
    inside = (f >= low_hz) & (f <= high_hz) & (transfer.beta_record > 0)
    x = f[inside] / 1e6
    slopes = []
    for row in transfer.per_field:
        y = DB_PER_NEPER * np.log(np.maximum(np.abs(row[inside]), 1e-300))
        w = transfer.beta_record[inside]
        columns = np.vstack([x, np.ones_like(x)]).T
        slopes.append(np.linalg.lstsq(columns * w[:, None], y * w, rcond=None)[0][0])
    slopes = np.array(slopes)
    return float(slopes.mean()), float(slopes.std(ddof=1) / np.sqrt(len(slopes))) if len(slopes) > 1 else np.inf


# The depth agent's figures from the same taps (depth_split, 2026-09-05):
# smoothed 24.4 kHz-bin power-ratio slopes.
DEPTH_AGENT = {
    "chroma_slope_db_per_octave_sp": (3.2, 0.4),
    "chroma_band_hz": (0.30e6, 1.00e6),
    "luma_flat_to_hz": 4.7e6,
    "luma_band_hz": (3.2e6, 5.0e6),
}


def compare_with_depth_slopes(transfers: Dict[int, TapTransfer]) -> Dict[str, object]:
    """The coherent slopes against the depth agent's power-ratio slopes."""
    out = {}
    lo, hi = DEPTH_AGENT["chroma_band_hz"]
    llo, lhi = DEPTH_AGENT["luma_band_hz"]
    for head, t in sorted(transfers.items()):
        out[HEAD_LABELS[head]] = {
            "chroma_slope_db_per_octave": slope_db_per_octave(t, lo, hi),
            "luma_slope_db_per_mhz": slope_db_per_mhz(t, llo, lhi),
        }
    out["depth_agent"] = dict(DEPTH_AGENT)
    return out
