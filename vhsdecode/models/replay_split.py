"""Align-and-difference for two captures of one tape: the replay split.

Ethan's two directives, verbatim, that this module serves:

(A) *"Cross-instrument capture - the experiment to run first. One tape, two
    different capture chains. Same magnetization, different hardware. Not two
    tapes on one device - that varies the wrong thing and leaves the card's
    contribution in both captures. Align, difference. What reproduces across
    both chains is the tape. What doesn't is the capture chain. Answers
    directly: is your -45 dB floor particulate-limited or electronics-limited?
    Practical payoff for restoration: tells you whether more electronics/
    filter work will improve the picture, or whether you're already at the
    medium's limit. Weaker fallback if only one card is available: one tape,
    two decks. Resolution limit of the split is where the two chains'
    uncorrelated noise dominates - roughly 10-20 dB below the floor with
    averaging."*

(B) *"Multi-pass replay - splits the floor. Replay one tape N times, align,
    average. Electronics/quantization/thermal fall as 1/sqrt(N). Frozen
    particulate does not - same physical magnetization every pass. Cross-track
    decorrelation between passes is a complication; sweeping tracking offset
    deliberately turns it into a measurement of the coating's cross-track
    correlation length."*

Both are the same machinery: align two or more captures of ONE magnetisation
line by line, and split every frequency band into the power that REPRODUCES
across captures and the power that does not. What reproduces is whatever sits
upstream of the point where the captures diverge - the tape when the chains
diverge at the deck's RF output, the tape and the deck when they diverge at
the capture card. What does not reproduce is everything downstream of that
point, plus whatever changed between passes.

===========================================================================
HOW THE ALIGNMENT IS DONE, AND WHY IN THIS ORDER
===========================================================================

1. COARSE, FROM THE SYNC-PULSE TRAIN. The RF is demodulated to its
   instantaneous frequency (the analytic signal's phase derivative), and the
   sync pulses are the runs where that frequency sits below the midpoint of
   the blanking and sync-tip carriers - both figures the decoder's own format
   table supplies (NTSC VHS: 3.686 and 3.400 MHz, JVC VTG82063). Pulses are
   classified by width against the format's equalising, horizontal and broad
   pulse widths, which gives the field starts and a line count inside every
   field. Candidate line offsets between the two captures are those that put
   field starts on field starts with matching field lengths.

2. FINE, FROM THE BLANKING INTERVAL'S RF ONLY. For every line pair the
   complex analytic RF over the blanking interval - front porch, sync pulse,
   back porch, a window the format defines - is cross-correlated. The GROUP
   DELAY comes from the slope of the cross-spectrum's phase against
   frequency, weighted by the cross-power, which is the sub-sample estimator
   that does not depend on the carrier phase. The active picture is never
   used for the alignment (standing constraint: sync and reserved intervals
   only).

3. CARRIER-PHASE REFINEMENT, WHERE IT IS LEGITIMATE. The residual phase of
   the cross-correlation at the group-delay lag is the carrier's phase
   difference between the captures. For two captures of ONE recording it is
   a constant (the chains' phase difference at the carrier) plus a term
   proportional to any residual delay, and that term resolves the delay a
   hundred times finer than the group delay can. For two DIFFERENT record
   events it is random line to line, because the modulator's phase at a
   given video instant is arbitrary. The concentration of the per-line phase
   across lines (a Rayleigh statistic) therefore says whether the two
   captures come from the same FM modulation event, and the refinement is
   applied only when it does.

4. PER-LINE TIME BASE, NOT ONE LAG. Two playbacks have independent time
   bases: capstan and drum flutter of a part in ten thousand is a third of a
   sample per line and tens of samples per field at 50 MSps, so a single lag
   is wrong within a field. The per-line lags form a piecewise-linear map,
   the second capture is resampled onto the first's time base with a
   windowed-sinc interpolator, and the residual misalignment is REPORTED:
   the estimator's own noise (disagreement of two independent sub-sample
   estimators on the same line), the wander (line-to-line change of the
   lag), the rate difference, and the lines that failed to lock.

5. THE SPLIT IS TRANSFER-INVARIANT. Reproduced power is the cross-spectrum
   between captures, not the power of their difference: a fixed transfer
   function between the two chains (the record-to-playback channel, or two
   cards' anti-alias filters) rotates and scales every bin and would read as
   non-reproduction in a plain difference. The coherence per bin ignores any
   fixed transfer and counts only what is not linearly related.

6. THE SIGNAL IS REMOVED BY THE FORMAT'S OWN PERIODICITY. Inside the
   carrier's band the reproduced power is dominated by the signal, which of
   course reproduces. To split the NOISE floor there, each capture's line is
   subtracted from its neighbour (aligned and phase-rotated, from the
   blanking window) - on line-periodic content the signal cancels and only
   the noise is left, and the coherence of those line differences across
   captures is the frozen share of the noise itself. Lines where the content
   changes are excluded by their own line-to-line scatter. The colour-under
   band is exempt: its phase is rotated line to line by the format, so it is
   not line-periodic and is reported by the direct estimator only.

===========================================================================
PREDICTIONS, WRITTEN BEFORE ANY MEASUREMENT (kept verbatim)
===========================================================================

Inputs to the arithmetic, with their sources:
  * measured capture floor, out of band:  46.5 dB below the carrier per
    sample over the 25 MHz Nyquist span (memory: capture-length-correction;
    8-bit, 3.4 dB over the arithmetic quantisation floor of 49.9 dB). Ethan
    rounds this to "the -45 dB floor".
  * particulate estimate: 37.9 dB signal-to-noise at 4.4 MHz from about
    7000 particles in about 19 um^3 of read volume (noise-budget arithmetic,
    `tape_model.magnetic_limits` / `interference.particle_noise`). That is a
    per-cell figure, i.e. over the carrier's own band (Carson: 8 MHz).

Bandwidth reference matters and is the main uncertainty in the prediction:
the electronics floor per sample over 25 MHz is 46.5 dB; over the 8 MHz
Carson band it is 46.5 + 10 log10(25/8) = 51.4 dB.

PREDICTION 1 - bounce1 against bounce2 (playback captures).
  Their record-side taps exist separately and their bounce transitions run
  in opposite directions (dark-to-bright at 0.22 s in bounce1, bright-to-
  dark at 0.26 s in bounce2; whole-file band energies), so they are two
  windows on DIFFERENT sections of tape. Predicted: no shared magnetisation.
  Direct coherence in the blanking window will still be high (the sync's RF
  is deterministic given the video), the carrier phase will NOT be shared
  (Rayleigh concentration near 1/sqrt(lines)), the comb-differenced frozen
  share of the in-band noise will be 0 within the resolution limit, and the
  unreproduced share of every noise band will be 1.0.

PREDICTION 2 - if a pair DID share magnetisation (the experiment Ethan
  should run), the frozen share of the in-band noise, measured by the comb
  estimator in the carrier band, would be
      tape / (tape + electronics)
      = 1 / (1 + 10^((37.9 - 46.5)/10)) = 0.88   (per-sample reference)
      = 1 / (1 + 10^((37.9 - 51.4)/10)) = 0.96   (Carson-band reference)
  i.e. 88-96 per cent of the floor frozen if the particulate estimate is
  right, and the in-band total floor would sit near -37.5 dB. If the tape
  were noiseless the in-band floor would sit at the electronics' -45 to
  -51 dB and the frozen share would be near zero. Above the tape's shortest
  recordable wavelength (1.26 um = 4.62 MHz; nothing survives the head above
  about 6 MHz) the frozen share is predicted to be zero except for
  deterministic deck spurs, which reproduce because the deck is the same.

PREDICTION 3 - record tap against playback tap of the same pattern.
  Same recording by construction, so the carrier phase should be shared
  (concentration near 1) up to the channel's constant phase. The direct
  coherence in the carrier band gives the signal as reproduced and, as
  unreproduced, the sum of tape noise, playback electronics, the record
  tap's own capture noise (its out-of-band floor is 11-13 dB below the
  playback capture's, so it is small) and the channel's non-linear
  products. That unreproduced power IS the in-band playback floor a single
  capture cannot see. Predicted -37.5 dB relative to the carrier if the
  particulate estimate holds, -45 dB or lower if the electronics dominate.

PREDICTION 4 - resolution. With one 0.48 s capture pair and a 1 MHz band
  the time-bandwidth product is 4.8e5 independent estimates, so a frozen
  share of 3 sigma is resolvable at 10 log10(3 sqrt(2) / (2 sqrt(4.8e5)))
  = -25 dB below the floor. Ethan's 10-20 dB corresponds to 10^2 to 10^4
  averages; whole captures do better.

===========================================================================
THE EXPERIMENT TO RUN (design note)
===========================================================================

One tape, two capture chains, from the SAME playback if the hardware allows
(a T on the deck's RF output into two cards: the deck's contribution is then
common and the split is exactly the card's), otherwise two playbacks with
one card each - which also folds the deck's non-frozen part into the
unreproduced side and is the weaker fallback Ethan names. Capture at least
the 0.48 s the existing windows have; every additional second lowers the
resolution floor by 10 log10 of the ratio, 5 dB per factor of ten. Two
passes resolve a frozen share 25 dB below the floor from 0.48 s; N passes
gain a further 10 log10(N/ (2 sqrt(1 + 1/(N-1)) / sqrt(2))) - the exact
figure is `resolution_limit`. For the cross-track sweep, step the deck's
tracking control: the tape moves longitudinally by the capstan's phase and
the head crosses the track by that shift times the sine of the track angle
(5 deg 58' NTSC, JVC VTG82063), so a whole 58 um track pitch is 557 um of
tape, one field's worth of motion; offsets inside a few micrometres are the
ones that resolve the correlation length, and offsets out to the track pitch
check the head-overlap triangle.

Measurements on the real data are recorded at the end of this docstring,
after the predictions, and never edit them.
"""

import itertools
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import interference_distributions

# --------------------------------------------------------------------------
# Structural numbers. Each carries its derivation; none is a format figure.
# --------------------------------------------------------------------------

# The instantaneous-frequency smoother used to find sync edges, as a fraction
# of the horizontal sync width. Shorter than the pulse by a wide margin so the
# edge is not smeared into its neighbours, long enough to average the FM
# demodulator's per-sample noise; both captures use the same smoother, so its
# residual bias cancels in the lag and the fine stage removes it anyway.
EDGE_SMOOTHING_FRACTION_OF_SYNC = 1.0 / 8.0

# A candidate pulse narrower than half the format's narrowest pulse (the
# equalising pulse) is not a pulse of the format; it is a dropout or noise.
NARROWEST_PULSE_FRACTION = 0.5

# The FM carrier inside a pulse is flat; a run whose frequency wanders by more
# than a quarter of the sync-to-white deviation is a dropout wearing the
# threshold, not a sync pulse.
PULSE_FLATNESS_FRACTION_OF_DEVIATION = 0.25

# The fine-lag search half-range as a fraction of the carrier period at the
# sync tip. The coarse edge estimate is sub-sample, so the search need only
# cover its error; a range wider than half a carrier period would admit the
# correlation's periodic side peaks.
SEARCH_FRACTION_OF_CARRIER_PERIOD = 0.45

# Windowed-sinc interpolator: 16 taps each side under a Kaiser window with
# beta 8 is flat to better than 1e-4 up to a quarter of the sample rate,
# which covers the 8 MHz Carson band at 50 MSps with a factor of three in
# hand. Verified on the planted data in the unit test.
INTERPOLATOR_HALF_WIDTH = 16
INTERPOLATOR_KAISER_BETA = 8.0
INTERPOLATOR_PHASES = 512

# The two sub-sample estimators (group delay from the phase slope, peak from
# the parabola) must agree to within this many samples for a line to count
# as locked. The parabola on a narrow-band correlation is the coarser of the
# two, so this bounds ITS error, not the group delay's.
ESTIMATOR_AGREEMENT_SAMPLES = 0.5

# A normalised correlation peak below one half says the two blanking windows
# do not describe the same waveform (a dropout, a head switch, a lost pulse).
MINIMUM_PEAK_CORRELATION = 0.5

# Carrier phase is judged SHARED when the Rayleigh concentration of the
# per-line phase difference exceeds this. Random phases over L lines give a
# concentration of order 1/sqrt(L); a shared carrier gives one near unity.
# The test is applied against the L-dependent null, this is only the floor.
MINIMUM_PHASE_CONCENTRATION = 0.5

# Lines whose content differs from their neighbour by more than this many
# median absolute deviations of the line-to-line scatter are excluded from
# the comb split (a content change is an outlier against a flat field's own
# line-to-line variation).
CONTENT_CHANGE_MAD_MULTIPLE = 6.0

# Jackknife blocks for the error bars: contiguous, so serially correlated
# segments stay together. Twenty blocks give a variance with 19 degrees of
# freedom, enough for a two-sigma bar to mean what it says.
JACKKNIFE_BLOCKS = 20


# --------------------------------------------------------------------------
# Format parameters, from the decoder's own tables
# --------------------------------------------------------------------------


def format_parameters(system: str = "NTSC", tape_format: str = "VHS",
                      tape_speed: int = 0) -> Dict[str, object]:
    """Every format figure this module uses, read from the decoder.

    The carrier frequencies come from the decoder's `ire0` and `hz_ire`
    (JVC VTG82063: sync tip 3.4 MHz, peak white 4.4 MHz for NTSC VHS), the
    timing from its `SysParams` (line period, sync, equalising and broad
    pulse widths, blanking end, active end), the track width from its
    `RFParams` per tape speed, and the track angle from `head_model`'s
    transcription of the same guide.
    """
    from vhsdecode.formats import get_format_params
    sysparams, rfparams = get_format_params(system, tape_format, tape_speed,
                                            logging.getLogger(__name__))
    hz_ire = float(sysparams["hz_ire"])
    ire0 = float(sysparams["ire0"])
    line_period_s = float(sysparams["line_period"]) * 1e-6
    active_start_us, active_end_us = sysparams["activeVideoUS"]
    parameters = {
        "system": system,
        "tape_format": tape_format,
        "tape_speed": int(tape_speed),
        "sync_tip_hz": ire0 + hz_ire * float(sysparams["vsync_ire"]),
        "blanking_hz": ire0,
        "white_hz": ire0 + 100.0 * hz_ire,
        "hz_per_ire": hz_ire,
        "line_period_s": line_period_s,
        "sync_width_s": float(sysparams["hsyncPulseUS"]) * 1e-6,
        "equalizing_width_s": float(sysparams["eqPulseUS"]) * 1e-6,
        "broad_width_s": float(sysparams["vsyncPulseUS"]) * 1e-6,
        "front_porch_s": (float(sysparams["line_period"])
                          - float(active_end_us)) * 1e-6,
        "blanking_end_s": float(active_start_us) * 1e-6,
        "field_lines": tuple(int(v) for v in sysparams["field_lines"]),
        "frame_lines": int(sysparams["frame_lines"]),
        "fields_per_second": 2.0 * float(sysparams["FPS"]),
        "colour_under_hz": float(rfparams.get("color_under_carrier", 0.0)),
        "track_width_m": float(rfparams.get("video_track_width", 0.0)) * 1e-6,
        "source": ("vhsdecode.formats.get_format_params: carrier figures per "
                   "JVC VTG82063 section 3, timing per the video standard"),
    }
    parameters["sync_to_white_hz"] = (parameters["white_hz"]
                                      - parameters["sync_tip_hz"])
    parameters["sync_depth_hz"] = (parameters["blanking_hz"]
                                   - parameters["sync_tip_hz"])
    try:
        from vhsdecode.models import head_model
        speed_name = {0: "SP", 1: "LP", 2: "EP"}.get(int(tape_speed), "SP")
        mechanics = head_model.FORMAT_MECHANICS.get(
            (tape_format, system, speed_name),
            head_model.FORMAT_MECHANICS.get((tape_format, system, "SP")))
        if mechanics:
            parameters["track_angle_degrees"] = float(
                mechanics["track_angle_degrees"])
            parameters["writing_speed_m_s"] = float(
                mechanics["writing_speed_m_s"])
    except Exception:                                       # pragma: no cover
        pass
    return parameters


def default_bands(parameters: Dict[str, object], sample_rate_hz: float
                  ) -> List[Tuple[str, float, float]]:
    """The bands the split is reported in, from the format's own figures.

    Colour-under: the down-converted chroma sits at the format's colour-under
    carrier with the chroma's own half-megahertz sidebands. Lower and upper
    sidebands and the carrier band are bounded by the sync-tip and white
    carriers, widened by the sync depth (the carrier rests between the tip
    and blanking for most of a line). Carson's band for the luma is
    2 x (deviation + baseband) = 8 MHz about the carrier's centre, so the
    upper sideband ends at the carrier's centre plus half of that, and the
    band above it is where the tape's shortest recordable wavelength has
    already given out and only the chains remain.
    """
    chroma_half_width_hz = 0.5e6            # NTSC chroma, +/- 0.5 MHz
    colour_under = float(parameters["colour_under_hz"])
    tip = float(parameters["sync_tip_hz"])
    white = float(parameters["white_hz"])
    depth = float(parameters["sync_depth_hz"])
    deviation = float(parameters["sync_to_white_hz"])
    baseband = 3.0e6                        # NTSC VHS luma baseband, JVC guide
    centre = 0.5 * (tip + white)
    carson_half = deviation + baseband
    nyquist = 0.5 * float(sample_rate_hz)
    bands = [
        ("colour-under", max(colour_under - chroma_half_width_hz, 0.0),
         colour_under + chroma_half_width_hz),
        ("lower sideband", colour_under + chroma_half_width_hz, tip - depth),
        ("carrier", tip - depth, white + depth),
        ("upper sideband", white + depth, centre + carson_half),
        ("beyond the tape", centre + carson_half,
         min(2.0 * (centre + carson_half), nyquist)),
    ]
    if 2.0 * (centre + carson_half) < nyquist:
        bands.append(("capture only", 2.0 * (centre + carson_half), nyquist))
    return bands


# --------------------------------------------------------------------------
# Analytic signal and instantaneous frequency
# --------------------------------------------------------------------------


def analytic_signal(samples: np.ndarray, sample_rate_hz: float,
                    band_hz: Optional[Tuple[float, float]] = None,
                    block: int = 1 << 20, guard: int = 1 << 12) -> np.ndarray:
    """The one-sided (analytic) signal, block-wise so a two-hour capture's
    window and a synthetic line both go through the same code.

    Each block is transformed with a guard on either side that is discarded,
    so the circular wrap of the FFT never reaches the kept samples. The
    optional band limit uses a raised-cosine edge two per cent of the band
    wide, because an abrupt edge rings across the guard.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    count = values.size
    out = np.empty(count, dtype=np.complex64)
    if count == 0:
        return out
    if count + 2 * guard <= block:
        block = int(2 ** np.ceil(np.log2(count + 2 * guard)))
    core = block - 2 * guard
    frequencies = np.fft.fftfreq(block, d=1.0 / float(sample_rate_hz))
    weight = np.zeros(block, dtype=np.float64)
    weight[frequencies > 0] = 2.0
    weight[0] = 1.0
    if block % 2 == 0:
        weight[block // 2] = 1.0
    if band_hz is not None:
        low, high = float(band_hz[0]), float(band_hz[1])
        edge = 0.02 * (high - low)
        shape = np.clip((frequencies - low) / max(edge, 1e-9), 0.0, 1.0) \
            * np.clip((high - frequencies) / max(edge, 1e-9), 0.0, 1.0)
        shape = 0.5 - 0.5 * np.cos(np.pi * shape)
        weight = weight * shape
    for start in range(0, count, core):
        stop = min(start + core, count)
        lo = start - guard
        hi = lo + block
        piece = np.zeros(block, dtype=np.float64)
        src_lo, src_hi = max(lo, 0), min(hi, count)
        piece[src_lo - lo:src_hi - lo] = values[src_lo:src_hi]
        spectrum = np.fft.fft(piece) * weight
        transformed = np.fft.ifft(spectrum)
        out[start:stop] = transformed[guard:guard + (stop - start)]
    return out


def instantaneous_frequency(analytic: np.ndarray, sample_rate_hz: float
                            ) -> np.ndarray:
    """The phase derivative, in hertz, from the product of neighbours - no
    unwrapping needed, and exact for any step under half the sample rate."""
    z = np.asarray(analytic)
    if z.size < 2:
        return np.zeros(z.size, dtype=np.float32)
    turn = np.angle(z[1:] * np.conj(z[:-1]))
    frequency = np.empty(z.size, dtype=np.float32)
    frequency[:-1] = turn * (float(sample_rate_hz) / (2.0 * np.pi))
    frequency[-1] = frequency[-2]
    return frequency


def moving_average(values: np.ndarray, length: int) -> np.ndarray:
    """Centred box average; the edges repeat the nearest full average."""
    length = max(int(length), 1)
    if length == 1:
        return np.asarray(values, dtype=np.float64)
    padded = np.pad(np.asarray(values, dtype=np.float64),
                    (length // 2, length - 1 - length // 2), mode="edge")
    total = np.cumsum(np.insert(padded, 0, 0.0))
    return (total[length:] - total[:-length]) / length


# --------------------------------------------------------------------------
# Sync pulses and the line table
# --------------------------------------------------------------------------


def sync_pulses(samples: np.ndarray, sample_rate_hz: float,
                parameters: Dict[str, object],
                analytic: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Every pulse in the capture whose carrier sits at the sync tip.

    Finds the runs of instantaneous frequency below the midpoint of the
    blanking and sync-tip carriers, classifies each by width against the
    format's three pulse widths, and places the leading edge at the
    sub-sample crossing of that midpoint. Dropouts are refused twice: a run
    narrower than half an equalising pulse is not a pulse, and a run whose
    carrier is not flat is not a pulse either.
    """
    rate = float(sample_rate_hz)
    tip = float(parameters["sync_tip_hz"])
    blanking = float(parameters["blanking_hz"])
    white = float(parameters["white_hz"])
    deviation = float(parameters["sync_to_white_hz"])
    if analytic is None:
        values = np.asarray(samples, dtype=np.float64)
        analytic = analytic_signal(values - values.mean(), rate,
                                   band_hz=(tip - deviation, white + deviation))
    frequency = instantaneous_frequency(analytic, rate)
    smoothing = max(int(round(float(parameters["sync_width_s"]) * rate
                              * EDGE_SMOOTHING_FRACTION_OF_SYNC)), 1)
    smooth = moving_average(frequency, smoothing)
    threshold = 0.5 * (blanking + tip)
    below = smooth < threshold
    change = np.diff(below.astype(np.int8))
    starts = np.flatnonzero(change == 1) + 1
    ends = np.flatnonzero(change == -1) + 1
    if below[0]:
        starts = np.insert(starts, 0, 0)
    if below[-1]:
        ends = np.append(ends, below.size)
    count = min(starts.size, ends.size)
    starts, ends = starts[:count], ends[:count]

    widths = {"equalizing": float(parameters["equalizing_width_s"]),
              "horizontal": float(parameters["sync_width_s"]),
              "broad": float(parameters["broad_width_s"])}
    names = list(widths)
    log_widths = np.log(np.array([widths[n] for n in names]))
    narrowest = widths["equalizing"] * NARROWEST_PULSE_FRACTION * rate
    envelope = np.abs(analytic)
    envelope_reference = float(np.median(envelope[::max(envelope.size // 100000, 1)]))

    edges, trailing, kinds, tip_hz, flat, env = [], [], [], [], [], []
    for start, end in zip(starts, ends):
        width = end - start
        if width < narrowest or start < 1 or end >= below.size:
            continue
        inside = smooth[start:end]
        middle = inside[width // 4: max(width // 4 + 1, 3 * width // 4)]
        wander = float(np.std(middle)) if middle.size else np.inf
        is_flat = wander < PULSE_FLATNESS_FRACTION_OF_DEVIATION * deviation
        kind = names[int(np.argmin(np.abs(np.log(width / rate) - log_widths)))]
        if not is_flat:
            kind = "dropout"
        # leading edge: the crossing between start-1 (above) and start (below)
        before, after = smooth[start - 1], smooth[start]
        lead = start - 1 + (before - threshold) / max(before - after, 1e-9)
        before, after = smooth[end - 1], smooth[end]
        trail = end - 1 + (threshold - before) / max(after - before, 1e-9)
        edges.append(lead)
        trailing.append(trail)
        kinds.append(kind)
        tip_hz.append(float(np.mean(middle)) if middle.size else np.nan)
        flat.append(is_flat)
        env.append(float(np.median(envelope[start:end])))
    return {
        "edge": np.array(edges, dtype=np.float64),
        "trailing_edge": np.array(trailing, dtype=np.float64),
        "width_s": (np.array(trailing) - np.array(edges)) / rate
        if edges else np.zeros(0),
        "kind": np.array(kinds, dtype=object),
        "tip_hz": np.array(tip_hz, dtype=np.float64),
        "envelope": np.array(env, dtype=np.float64),
        "envelope_reference": envelope_reference,
        "threshold_hz": threshold,
        "smoothing_samples": smoothing,
        "analytic": analytic,
        "frequency_hz": frequency,
        "sample_rate_hz": rate,
    }


def line_table(pulses: Dict[str, object], parameters: Dict[str, object]
               ) -> Dict[str, object]:
    """Field starts and a line count inside every field, from the pulses.

    A field starts at the first broad pulse of a broad-pulse block; every
    horizontal pulse after it is numbered by its position in that field.
    Lines before the first field start belong to a field whose start the
    capture did not see and are numbered from a notional start so that the
    field index is -1 and the count still runs.
    """
    rate = float(pulses["sample_rate_hz"])
    line_samples = float(parameters["line_period_s"]) * rate
    edge = pulses["edge"]
    kind = pulses["kind"]
    is_broad = kind == "broad"
    is_h = kind == "horizontal"
    field_starts = []
    last_broad = -np.inf
    for index in np.flatnonzero(is_broad):
        if edge[index] - last_broad > 2.0 * line_samples:
            field_starts.append(index)
        last_broad = edge[index]
    field_start_edges = edge[field_starts] if field_starts else np.zeros(0)
    h_index = np.flatnonzero(is_h)
    field_of = np.searchsorted(field_start_edges, edge[h_index],
                               side="right") - 1
    line_in_field = np.zeros(h_index.size, dtype=np.int64)
    for field in np.unique(field_of):
        members = np.flatnonzero(field_of == field)
        if field >= 0:
            # lines since the field start, counted in line periods so that a
            # missed pulse does not shift every later line by one
            since = (edge[h_index[members]] - field_start_edges[field]) / line_samples
            line_in_field[members] = np.rint(since).astype(np.int64)
        else:
            since = (edge[h_index[members]] - edge[h_index[members[0]]]) / line_samples
            line_in_field[members] = np.rint(since).astype(np.int64)
    # a global line number: field start positions in line periods plus the
    # in-field count, anchored on the first horizontal pulse
    global_line = np.rint((edge[h_index] - edge[h_index[0]]) / line_samples
                          ).astype(np.int64) if h_index.size else np.zeros(0, np.int64)
    field_lengths = (np.diff(field_start_edges) / line_samples
                     if field_start_edges.size > 1 else np.zeros(0))
    return {
        "pulse_index": h_index,
        "edge": edge[h_index],
        "field": field_of,
        "line_in_field": line_in_field,
        "global_line": global_line,
        "field_start_edges": field_start_edges,
        "field_start_lines": (np.rint((field_start_edges - edge[h_index[0]])
                                      / line_samples).astype(np.int64)
                              if h_index.size and field_start_edges.size
                              else np.zeros(0, np.int64)),
        "field_lengths_lines": np.rint(field_lengths).astype(np.int64),
        "line_samples": line_samples,
        "tip_hz": pulses["tip_hz"][h_index],
        "count": int(h_index.size),
    }


# --------------------------------------------------------------------------
# Interpolation on an arbitrary time map
# --------------------------------------------------------------------------


def _interpolator_table() -> np.ndarray:
    """Polyphase table of the Kaiser-windowed sinc, one row per fractional
    phase; rows are normalised to unit gain at DC."""
    half = INTERPOLATOR_HALF_WIDTH
    taps = np.arange(-half + 1, half + 1, dtype=np.float64)
    phases = np.arange(INTERPOLATOR_PHASES + 1) / INTERPOLATOR_PHASES
    argument = taps[None, :] - phases[:, None]
    window = np.i0(INTERPOLATOR_KAISER_BETA
                   * np.sqrt(np.clip(1.0 - (argument / half) ** 2, 0.0, 1.0))
                   ) / np.i0(INTERPOLATOR_KAISER_BETA)
    kernel = np.sinc(argument) * window
    return kernel / kernel.sum(axis=1, keepdims=True)


_INTERPOLATOR = None


def resample_on_map(values: np.ndarray, positions: np.ndarray,
                    chunk: int = 1 << 19) -> np.ndarray:
    """values(positions), band-limited: a windowed sinc with linear
    interpolation between polyphase rows. Positions outside the array give
    zero. Works for real and complex input; the lag convention throughout
    is that output sample n reads the input at positions[n]."""
    global _INTERPOLATOR
    if _INTERPOLATOR is None:
        _INTERPOLATOR = _interpolator_table()
    table = _INTERPOLATOR
    half = INTERPOLATOR_HALF_WIDTH
    x = np.asarray(values)
    pos = np.asarray(positions, dtype=np.float64)
    complex_input = np.iscomplexobj(x)
    out = np.zeros(pos.shape, dtype=np.complex128 if complex_input else np.float64)
    padded = np.concatenate([np.zeros(half, dtype=x.dtype), x,
                             np.zeros(half + 1, dtype=x.dtype)])
    taps = np.arange(-half + 1, half + 1)
    for start in range(0, pos.size, chunk):
        p = pos[start:start + chunk]
        valid = np.isfinite(p) & (p >= 0) & (p <= x.size - 1)
        p = np.where(valid, p, 0.0)
        base = np.floor(p).astype(np.int64)
        fraction = p - base
        row = fraction * INTERPOLATOR_PHASES
        row_low = np.minimum(np.floor(row).astype(np.int64), INTERPOLATOR_PHASES - 1)
        weight_high = row - row_low
        kernels = (table[row_low] * (1.0 - weight_high)[:, None]
                   + table[row_low + 1] * weight_high[:, None])
        indices = base[:, None] + taps[None, :] + half
        gathered = padded[indices]
        result = np.sum(gathered * kernels, axis=1)
        out[start:start + chunk] = np.where(valid, result, 0.0)
    return out


# --------------------------------------------------------------------------
# Fine lag between two blanking windows
# --------------------------------------------------------------------------


def fine_lag(window_a: np.ndarray, window_b: np.ndarray, search: int
             ) -> Dict[str, object]:
    """The sub-sample lag of `window_b` relative to `window_a`.

    `window_b` is wider than `window_a` by `search` samples each side. The
    complex cross-correlation of the two analytic windows is evaluated at
    every integer lag in the search range; its MAGNITUDE is the envelope of
    a band-pass correlation and varies on the scale of the reciprocal
    bandwidth (about 17 samples for the 3 MHz sync band at 50 MSps), so a
    parabola through the three samples about the peak locates the maximum
    to better than a hundredth of a sample - verified on a planted smooth
    fractional delay (0.37 -> 0.371, -1.25 -> -1.249, 2.6 -> 2.601). A
    five-sample least-squares parabola is returned beside it as a shape
    check: the two disagree only where the peak is not a single smooth
    maximum (a dropout, a double pulse), and the disagreement is REPORTED,
    not used as a gate.

    The phase-slope (group delay) estimator that stood here before was
    measured to recover only thirty per cent of a fractional delay on this
    narrow-band waveform and has been removed.

    The residual carrier phase is the phase of `b` relative to `a` at the
    aligned instant, `b(t) = a(t - d) exp(j phi)`, corrected for the
    fraction of a sample between the integer peak and the true lag at the
    cross-power-weighted mean frequency of the window.

    Lag convention: an event at sample n of `a` is at n + lag of `b`, both
    counted from the start of the un-widened windows.
    """
    a = np.asarray(window_a, dtype=np.complex128)
    b = np.asarray(window_b, dtype=np.complex128)
    length = a.size
    empty = {"lag": np.nan, "peak": 0.0, "locked": False,
             "residual_phase": np.nan, "carrier_cycles_per_sample": np.nan,
             "parabolic_lag": np.nan, "estimator_disagreement": np.nan,
             "integer_lag": 0, "at_edge": True}
    if b.size < length + 2 * search or length < 8:
        return empty
    lags = np.arange(-search, search + 1)
    energy_a = float(np.sum(np.abs(a) ** 2))
    correlation = np.empty(lags.size, dtype=np.complex128)
    magnitude = np.empty(lags.size)
    for k, l in enumerate(lags):
        piece = b[search + l: search + l + length]
        correlation[k] = np.vdot(piece, a)             # sum(conj(piece) * a)
        energy_b = float(np.sum(np.abs(piece) ** 2))
        magnitude[k] = (abs(correlation[k]) / np.sqrt(energy_a * energy_b)
                        if energy_a > 0 and energy_b > 0 else 0.0)
    best = int(np.argmax(magnitude))
    at_edge = best == 0 or best == lags.size - 1
    peak = float(magnitude[best])
    integer_lag = int(lags[best])
    if at_edge or peak <= 0:
        return {**empty, "peak": peak, "integer_lag": integer_lag,
                "at_edge": bool(at_edge)}
    left, centre, right = magnitude[best - 1], magnitude[best], magnitude[best + 1]
    curvature = left - 2.0 * centre + right
    fraction = float(0.5 * (left - right) / curvature) if curvature < 0 else 0.0
    fraction = float(np.clip(fraction, -1.0, 1.0))
    # five-point least-squares parabola as the shape check
    lo, hi = max(best - 2, 0), min(best + 3, lags.size)
    x = (lags[lo:hi] - integer_lag).astype(np.float64)
    coefficients = np.polyfit(x, magnitude[lo:hi], 2)
    five_point = (float(-coefficients[1] / (2.0 * coefficients[0]))
                  if coefficients[0] < 0 else fraction)
    five_point = float(np.clip(five_point, -1.5, 1.5))
    # the window's carrier: the cross-power-weighted mean frequency
    taper = np.hanning(length)
    aligned_b = b[search + integer_lag: search + integer_lag + length]
    cross = np.conj(np.fft.fft(a * taper)) * np.fft.fft(aligned_b * taper)
    weight = np.abs(cross)
    cycles = np.fft.fftfreq(length)
    f0 = float(np.sum(weight * cycles) / max(np.sum(weight), 1e-300))
    # c(best) = exp(-j phi) R_aa(-fraction), and R_aa(tau) turns as
    # exp(-2 pi j f0 tau) for a carrier at f0, so the angle of c(best) is
    # -phi + 2 pi f0 fraction
    residual = float(-np.angle(correlation[best]) + 2.0 * np.pi * f0 * fraction)
    residual = float(np.angle(np.exp(1j * residual)))
    return {
        "lag": float(integer_lag + fraction),
        "parabolic_lag": float(integer_lag + five_point),
        "integer_lag": integer_lag,
        "peak": peak,
        "at_edge": False,
        "residual_phase": residual,
        "carrier_cycles_per_sample": f0,
        "estimator_disagreement": float(abs(fraction - five_point)),
        "locked": bool(peak >= MINIMUM_PEAK_CORRELATION),
    }


# --------------------------------------------------------------------------
# Windows, matching and line differences
# --------------------------------------------------------------------------


def _blanking_window(parameters: Dict[str, object], sample_rate_hz: float
                     ) -> Tuple[int, int]:
    """Samples before and after the sync edge that the blanking interval
    spans: the front porch before, the back porch to the blanking end
    after. Format figures."""
    rate = float(sample_rate_hz)
    before = int(round(float(parameters["front_porch_s"]) * rate))
    after = int(round(float(parameters["blanking_end_s"]) * rate))
    return before, after


def _search_range(parameters: Dict[str, object], sample_rate_hz: float) -> int:
    period = float(sample_rate_hz) / float(parameters["sync_tip_hz"])
    return max(int(np.floor(SEARCH_FRACTION_OF_CARRIER_PERIOD * period)), 2)


def _match(analytic_a, centre_a: int, analytic_b, centre_b: int,
           before: int, after: int, search: int):
    """fine_lag between the blanking windows centred on two sync edges.
    The returned lag is between window starts; the absolute lag between the
    two captures at this line is (centre_b - centre_a) + lag."""
    if centre_a - before < 0 or centre_a + after > analytic_a.size:
        return None
    lo, hi = centre_b - before - search, centre_b + after + search
    if lo < 0 or hi > analytic_b.size:
        return None
    return fine_lag(analytic_a[centre_a - before: centre_a + after],
                    analytic_b[lo:hi], search)


def _ramped_positions(start: int, length: int, lag_start: float,
                      lag_end: float, ramp_length: float) -> np.ndarray:
    """Positions in the other signal for `length` samples from `start`,
    with the lag ramping linearly from `lag_start` at the start to
    `lag_end` a `ramp_length` further on - the time base between two
    sync edges is taken as linear, which the drum and capstan rates make
    exact to a part in a hundred thousand over a line."""
    offset = np.arange(length, dtype=np.float64)
    lag = lag_start + (lag_end - lag_start) * offset / max(float(ramp_length), 1.0)
    return start + offset + lag


def consecutive_line_lags(analytic: np.ndarray, table: Dict[str, object],
                          parameters: Dict[str, object], sample_rate_hz: float
                          ) -> Dict[str, np.ndarray]:
    """Within ONE capture: the lag and carrier-phase advance from every line's
    sync to the next line's, from the blanking windows alone."""
    before, after = _blanking_window(parameters, sample_rate_hz)
    search = _search_range(parameters, sample_rate_hz)
    centres = np.rint(table["edge"]).astype(np.int64)
    count = centres.size
    lag = np.full(count, np.nan)
    phase = np.full(count, np.nan)
    peak = np.zeros(count)
    for n in range(count - 1):
        fit = _match(analytic, centres[n], analytic, centres[n + 1],
                     before, after, search)
        if fit is None or not fit["locked"]:
            continue
        lag[n] = (centres[n + 1] - centres[n]) + fit["lag"]
        phase[n] = fit["residual_phase"]
        peak[n] = fit["peak"]
    return {"lag": lag, "phase": phase, "peak": peak, "centre": centres,
            "before": before, "after": after, "search": search}


def line_difference(analytic: np.ndarray, consecutive: Dict[str, np.ndarray],
                    index: int, length: int) -> Optional[np.ndarray]:
    """This line minus the next, the next aligned onto this one's time base
    and rotated to its carrier phase - the comb that removes line-periodic
    signal and keeps the noise. Needs the lag at this line and at the next
    (the ramp's two ends)."""
    lag = consecutive["lag"]
    if index + 1 >= lag.size or not (np.isfinite(lag[index]) and np.isfinite(lag[index + 1])):
        return None
    before = consecutive["before"]
    start = int(consecutive["centre"][index]) - before
    if start < 0 or start + length > analytic.size:
        return None
    ramp = float(consecutive["centre"][index + 1] - consecutive["centre"][index])
    positions = _ramped_positions(start, length, lag[index], lag[index + 1], ramp)
    if positions[-1] > analytic.size - 1:
        return None
    following = resample_on_map(analytic, positions)
    rotation = np.exp(1j * consecutive["phase"][index])
    return np.asarray(analytic[start:start + length], dtype=np.complex128) \
        - rotation * following


# --------------------------------------------------------------------------
# Alignment of two captures
# --------------------------------------------------------------------------


def _active_feature(pulses: Dict[str, object], table: Dict[str, object],
                    parameters: Dict[str, object]) -> np.ndarray:
    """The mean carrier frequency over each line's active interval - a
    CONTENT signature used only to choose between sync-consistent coarse
    candidates (the bounce transition, a moving pattern). It never enters the
    fine alignment or the split."""
    rate = float(pulses["sample_rate_hz"])
    frequency = pulses["frequency_hz"]
    start = int(round(float(parameters["blanking_end_s"]) * rate))
    stop = int(round((float(parameters["line_period_s"])
                      - float(parameters["front_porch_s"])) * rate))
    centres = np.rint(table["edge"]).astype(np.int64)
    feature = np.full(centres.size, np.nan)
    for n, centre in enumerate(centres):
        lo, hi = centre + start, centre + stop
        if 0 <= lo < hi <= frequency.size:
            feature[n] = float(np.mean(frequency[lo:hi]))
    return feature


def _line_pairs_for_offset(table_a, table_b, offset_lines: int
                           ) -> List[Tuple[int, int]]:
    """Pairs of horizontal-pulse indices whose global lines differ by
    `offset_lines` (line_a - line_b)."""
    lookup = {int(line): index for index, line in enumerate(table_b["global_line"])}
    pairs = []
    for index_a, line_a in enumerate(table_a["global_line"]):
        index_b = lookup.get(int(line_a) - int(offset_lines))
        if index_b is not None:
            pairs.append((index_a, index_b))
    return pairs


def _coarse_candidates(table_a, table_b) -> List[int]:
    """Line offsets that put field starts on field starts with matching
    field lengths; when a capture has no field start, the offset that
    aligns the first lines is the only candidate."""
    starts_a = table_a["field_start_lines"]
    starts_b = table_b["field_start_lines"]
    lengths_a = table_a["field_lengths_lines"]
    lengths_b = table_b["field_lengths_lines"]
    candidates = set()
    for i, line_a in enumerate(starts_a):
        for j, line_b in enumerate(starts_b):
            if i < lengths_a.size and j < lengths_b.size \
                    and lengths_a[i] != lengths_b[j]:
                continue
            candidates.add(int(line_a) - int(line_b))
    if not candidates and table_a["count"] and table_b["count"]:
        candidates.add(int(table_a["global_line"][0]) - int(table_b["global_line"][0]))
    return sorted(candidates)


def _content_score(feature_a, feature_b, pairs) -> float:
    """Correlation of the two content signatures over the paired lines;
    zero where either is constant (a static pattern says nothing)."""
    if len(pairs) < 4:
        return 0.0
    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])
    x, y = feature_a[ia], feature_b[ib]
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 4:
        return 0.0
    x, y = x[keep] - x[keep].mean(), y[keep] - y[keep].mean()
    norm = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / norm) if norm > 0 else 0.0


def _cross_line_coherence(delta_a: np.ndarray, analytic_b, consecutive_b,
                          index_b: int, lag_start: float, lag_end: float,
                          ramp: float, length: int) -> Optional[float]:
    """Coherence of capture a's line difference with capture b's, b's being
    moved onto a's time base by the cross-capture lag ramp."""
    delta_b = line_difference(analytic_b, consecutive_b, index_b, length)
    if delta_b is None:
        return None
    # delta_b is on b's own sample grid starting at b's window start; a's
    # sample k corresponds to b's sample k + lag(k) relative to the two
    # window starts, so the lag ramp is applied here as positions into delta_b
    positions = _ramped_positions(0, length, lag_start, lag_end, ramp)
    moved = resample_on_map(delta_b, positions)
    inside = (positions >= 0) & (positions <= length - 1)
    if inside.sum() < length // 2:
        return None
    x, y = delta_a[inside], moved[inside]
    norm = np.sqrt(np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2))
    return float(abs(np.vdot(y, x)) / norm) if norm > 0 else None


def _fingerprint(analytic_a, analytic_b, table_a, table_b, consecutive_a,
                 consecutive_b, pairs, lines: int, control_shift: int = 3
                 ) -> Dict[str, float]:
    """The content-blind test of shared magnetisation on a few lines: the
    coherence of the LINE DIFFERENCES between the captures, beside the same
    statistic on deliberately wrong partners (`control_shift` lines away) -
    the built-in null, so a lock is judged against what no-lock looks like
    on this very data."""
    length = int(round(table_a["line_samples"]))
    before, after, search = (consecutive_a["before"], consecutive_a["after"],
                             consecutive_a["search"])
    chosen = pairs[:: max(len(pairs) // lines, 1)][:lines]
    matched, control, phasors = [], [], []
    for index_a, index_b in chosen:
        delta_a = line_difference(analytic_a, consecutive_a, index_a, length)
        if delta_a is None:
            continue
        for shift, sink in ((0, matched), (control_shift, control)):
            jb = index_b + shift
            if jb + 2 >= table_b["count"] or index_a + 2 >= table_a["count"]:
                continue
            fits = []
            for step in (0, 1):
                fit = _match(analytic_a, int(consecutive_a["centre"][index_a + step]),
                             analytic_b, int(consecutive_b["centre"][jb + step]),
                             before, after, search)
                if fit is None or not fit["locked"]:
                    fits = []
                    break
                fits.append(fit)
            if not fits:
                continue
            # lag between window starts at this line and at the next
            centre_a0 = int(consecutive_a["centre"][index_a])
            centre_b0 = int(consecutive_b["centre"][jb])
            ramp = float(consecutive_a["centre"][index_a + 1] - centre_a0)
            lag0 = (centre_b0 - centre_a0) + fits[0]["lag"] - (centre_b0 - centre_a0)
            lag1 = ((int(consecutive_b["centre"][jb + 1]) - int(consecutive_a["centre"][index_a + 1]))
                    + fits[1]["lag"] - (centre_b0 - centre_a0))
            value = _cross_line_coherence(delta_a, analytic_b, consecutive_b, jb,
                                          lag0, lag1, ramp, length)
            if value is not None:
                sink.append(value)
                if shift == 0:
                    phasors.append(np.exp(1j * fits[0]["residual_phase"]))
    result = {
        "coherence": float(np.mean(matched)) if matched else 0.0,
        "lines": len(matched),
        "control_coherence": float(np.mean(control)) if control else 0.0,
        "control_spread": float(np.std(control)) if len(control) > 1 else 0.0,
        "control_lines": len(control),
        "carrier_concentration": float(abs(np.mean(phasors))) if phasors else 0.0,
    }
    # five standard errors above the null, with the null's own spread
    spread = result["control_spread"] / np.sqrt(max(len(control), 1))
    result["shared"] = bool(matched and control and
                            result["coherence"] > result["control_coherence"]
                            + 5.0 * max(spread, 1e-3))
    return result


def align(a: np.ndarray, b: np.ndarray, sample_rate_hz: float,
          parameters: Optional[Dict[str, object]] = None,
          content_lock: bool = True, fingerprint_lines: int = 16,
          use_carrier_phase: Optional[bool] = None) -> Dict[str, object]:
    """Line-by-line alignment of capture `b` onto capture `a`.

    Coarse from the sync-pulse train (field starts on field starts), the
    choice between sync-consistent candidates by the content-blind noise
    fingerprint where the captures share magnetisation and by the content
    signature where they do not, fine from the blanking-interval RF of
    every line, and the carrier-phase refinement where the per-line carrier
    phase is concentrated. Returns everything the later stages need and a
    verdict that says how the lock was obtained - or that it was not.
    """
    rate = float(sample_rate_hz)
    parameters = parameters or format_parameters()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pulses_a = sync_pulses(a - a.mean(), rate, parameters)
    pulses_b = sync_pulses(b - b.mean(), rate, parameters)
    table_a = line_table(pulses_a, parameters)
    table_b = line_table(pulses_b, parameters)
    analytic_a, analytic_b = pulses_a["analytic"], pulses_b["analytic"]
    result = {"sample_rate_hz": rate, "parameters": parameters,
              "table_a": table_a, "table_b": table_b,
              "pulses_a": pulses_a, "pulses_b": pulses_b,
              "aligned": False, "reason": "", "candidates": []}
    if table_a["count"] < 4 or table_b["count"] < 4:
        result["reason"] = "too few horizontal sync pulses found"
        return result
    feature_a = _active_feature(pulses_a, table_a, parameters)
    feature_b = _active_feature(pulses_b, table_b, parameters)
    consecutive_a = consecutive_line_lags(analytic_a, table_a, parameters, rate)
    consecutive_b = consecutive_line_lags(analytic_b, table_b, parameters, rate)
    table_a["active_hz"], table_b["active_hz"] = feature_a, feature_b

    scored = []
    for offset in _coarse_candidates(table_a, table_b):
        pairs = _line_pairs_for_offset(table_a, table_b, offset)
        if len(pairs) < 4:
            continue
        fingerprint = _fingerprint(analytic_a, analytic_b, table_a, table_b,
                                   consecutive_a, consecutive_b, pairs,
                                   fingerprint_lines)
        scored.append({"offset_lines": offset, "pairs": len(pairs),
                       "content": _content_score(feature_a, feature_b, pairs),
                       **fingerprint})
    result["candidates"] = scored
    if not scored:
        result["reason"] = "no sync-consistent overlap between the captures"
        return result

    shared = [c for c in scored if c["shared"]]
    if shared:
        chosen = max(shared, key=lambda c: c["coherence"])
        how = "shared noise fingerprint"
    elif content_lock:
        ordered = sorted(scored, key=lambda c: c["content"], reverse=True)
        best = ordered[0]
        runner = ordered[1]["content"] if len(ordered) > 1 else -1.0
        # half the maximum correlation, and clear of the runner-up by a
        # tenth: a static pattern ties every frame and must not be called
        if best["content"] > 0.5 and best["content"] - runner > 0.1:
            chosen, how = best, "content signature (no shared noise)"
        else:
            result["reason"] = ("no shared noise between the captures and "
                                "no decisive content lock: nothing to align "
                                "beyond the sync structure")
            result["best_candidate"] = max(scored, key=lambda c: c["coherence"])
            return result
    else:
        result["reason"] = "no shared noise between the captures"
        result["best_candidate"] = max(scored, key=lambda c: c["coherence"])
        return result

    offset = chosen["offset_lines"]
    pairs = _line_pairs_for_offset(table_a, table_b, offset)
    before, after = _blanking_window(parameters, rate)
    search = _search_range(parameters, rate)
    count = len(pairs)
    index_a = np.array([p[0] for p in pairs])
    index_b = np.array([p[1] for p in pairs])
    centre_a = np.rint(table_a["edge"][index_a]).astype(np.int64)
    centre_b = np.rint(table_b["edge"][index_b]).astype(np.int64)
    lag = np.full(count, np.nan)
    parabolic = np.full(count, np.nan)
    peak = np.zeros(count)
    phase = np.full(count, np.nan)
    carrier = np.full(count, np.nan)
    locked = np.zeros(count, dtype=bool)
    for k in range(count):
        fit = _match(analytic_a, centre_a[k], analytic_b, centre_b[k],
                     before, after, search)
        if fit is None:
            continue
        lag[k] = (centre_b[k] - centre_a[k]) + fit["lag"]
        parabolic[k] = (centre_b[k] - centre_a[k]) + fit["parabolic_lag"]
        peak[k] = fit["peak"]
        phase[k] = fit["residual_phase"]
        carrier[k] = fit["carrier_cycles_per_sample"]
        locked[k] = fit["locked"]

    # the carrier phase across lines: concentrated means the same
    # modulation event; its null over L lines is of order 1/sqrt(L)
    phasors = np.exp(1j * phase[locked])
    concentration = float(abs(np.mean(phasors))) if phasors.size else 0.0
    null = float(np.sqrt(np.pi / (4.0 * max(phasors.size, 1))))
    phase_shared = bool(phasors.size >= 8 and concentration
                        > max(MINIMUM_PHASE_CONCENTRATION, 5.0 * null))
    refine = phase_shared if use_carrier_phase is None else bool(use_carrier_phase)
    refined = lag.copy()
    if refine and phasors.size:
        common = np.angle(np.mean(phasors))
        # a residual delay d leaves the carrier phase at -2 pi f0 d
        residual_delay = -np.angle(np.exp(1j * (phase - common))) / (2.0 * np.pi * carrier)
        refined = lag + np.where(locked, residual_delay, 0.0)

    good = locked & np.isfinite(refined)
    # a lag that jumps by more than half a carrier period between adjacent
    # lines is not a time base, it is a lock failure
    period = rate / float(parameters["sync_tip_hz"])
    if good.sum() > 2:
        jump = np.abs(np.diff(refined[good]))
        bad_jump = np.flatnonzero(jump > 0.5 * period)
        good_indices = np.flatnonzero(good)
        for j in bad_jump:
            good[good_indices[j + 1]] = False
    statistics = _misalignment_statistics(refined, lag, parabolic, good,
                                          centre_a, rate, period)
    result.update({
        "aligned": bool(good.sum() >= 4),
        "reason": how if good.sum() >= 4 else "fewer than four locked lines",
        "how": how,
        "chosen": chosen,
        "offset_lines": offset,
        "index_a": index_a, "index_b": index_b,
        "edge_a": table_a["edge"][index_a], "edge_b": table_b["edge"][index_b],
        "lag": refined, "group_lag": lag, "parabolic_lag": parabolic,
        "peak": peak, "residual_phase": phase, "locked": good,
        "carrier_phase_shared": phase_shared,
        "carrier_phase_concentration": concentration,
        "carrier_phase_null": null,
        "carrier_phase_used": refine,
        "consecutive_a": consecutive_a, "consecutive_b": consecutive_b,
        "analytic_a": analytic_a, "analytic_b": analytic_b,
        "statistics": statistics,
    })
    return result


def _misalignment_statistics(refined, group, parabolic, good, centre_a,
                             rate, period) -> Dict[str, float]:
    """What is left after the alignment, in samples: the estimators' own
    noise, the wander, the rate difference and the failures."""
    if good.sum() < 3:
        return {"locked_lines": int(good.sum()), "failed_lines": int((~good).sum())}
    lag = refined[good]
    t = centre_a[good].astype(np.float64)
    slope, intercept = np.polyfit(t, lag, 1)
    detrended = lag - (slope * t + intercept)
    # second difference of a smooth series is dominated by the estimator
    # noise; sqrt(6) is its gain on white noise
    second = np.diff(lag, 2)
    return {
        "locked_lines": int(good.sum()),
        "failed_lines": int((~good).sum()),
        "estimator_disagreement_rms": float(np.sqrt(np.mean(
            (group[good] - parabolic[good]) ** 2))),
        "per_line_noise_rms": float(np.std(second) / np.sqrt(6.0)) if second.size else np.nan,
        "wander_peak_to_peak": float(np.ptp(detrended)),
        "wander_rms": float(np.std(detrended)),
        "rate_difference_ppm": float(slope * 1e6),
        "mean_lag": float(np.mean(lag)),
        "carrier_period_samples": float(period),
    }


# --------------------------------------------------------------------------
# Resampling onto the reference time base, and the difference
# --------------------------------------------------------------------------


def resample_onto(b: np.ndarray, alignment: Dict[str, object],
                  length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Capture `b` on capture `a`'s time base, from the per-line lags. The
    map is piecewise linear between locked sync edges; outside the locked
    span the mask is false. Returns (resampled, valid mask)."""
    good = alignment["locked"]
    edges = alignment["edge_a"][good]
    lags = alignment["lag"][good]
    length = int(length) if length is not None else int(alignment["analytic_a"].size)
    valid = np.zeros(length, dtype=bool)
    out = np.zeros(length, dtype=np.float64)
    if edges.size < 2:
        return out, valid
    before, _ = _blanking_window(alignment["parameters"], alignment["sample_rate_hz"])
    line = float(alignment["table_a"]["line_samples"])
    lo = max(int(np.floor(edges[0])) - before, 0)
    hi = min(int(np.ceil(edges[-1] + line)) - before, length)
    n = np.arange(lo, hi, dtype=np.float64)
    positions = n + np.interp(n, edges, lags)
    moved = resample_on_map(np.asarray(b, dtype=np.float64), positions)
    inside = (positions >= 0) & (positions <= len(b) - 1)
    out[lo:hi] = np.where(inside, moved, 0.0)
    valid[lo:hi] = inside
    return out, valid


def difference(a: np.ndarray, b_on_a: np.ndarray, alignment: Dict[str, object],
               valid: Optional[np.ndarray] = None) -> Dict[str, object]:
    """a minus b, b already on a's time base, with a real gain per field
    fitted on the blanking windows (the sync tip's carrier is nominally
    constant, so its level is the legitimate gain reference).

    This plain difference over-counts non-reproduction by whatever fixed
    transfer separates the chains (it cannot rotate a bin); the split in
    `floor_split` is transfer-invariant and is the number to quote. The
    difference is for looking at.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b_on_a, dtype=np.float64)
    before, after = _blanking_window(alignment["parameters"], alignment["sample_rate_hz"])
    good = alignment["locked"]
    centres = np.rint(alignment["edge_a"][good]).astype(np.int64)
    fields = alignment["table_a"]["field"][alignment["index_a"][good]]
    gains = {}
    for field in np.unique(fields):
        num = den = 0.0
        for centre in centres[fields == field]:
            lo, hi = centre - before, centre + after
            if lo < 0 or hi > a.size:
                continue
            num += float(np.dot(a[lo:hi], b[lo:hi]))
            den += float(np.dot(b[lo:hi], b[lo:hi]))
        gains[int(field)] = num / den if den > 0 else 1.0
    gain = np.ones(a.size)
    for field, value in gains.items():
        members = centres[fields == field]
        if members.size:
            gain[max(members.min() - before, 0): members.max() + int(alignment["table_a"]["line_samples"])] = value
    d = a - gain * b
    if valid is not None:
        d = np.where(valid, d, 0.0)
    return {"difference": d, "gain_per_field": gains, "gain": gain}


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


def _band_slices(frequencies_hz: np.ndarray, bands) -> List[np.ndarray]:
    return [(frequencies_hz >= low) & (frequencies_hz < high)
            for _, low, high in bands]


def _cross_matrices_from_spectra(spectra: np.ndarray, slices) -> np.ndarray:
    """spectra: [passes, bins] complex. Returns [bands, passes, passes] of
    real cross powers summed over the band's bins."""
    n = spectra.shape[0]
    out = np.zeros((len(slices), n, n))
    for k, inside in enumerate(slices):
        block = spectra[:, inside]
        out[k] = np.real(block @ np.conj(block.T))
    return out


def _split_from_matrices(matrices: np.ndarray, blocks: int = JACKKNIFE_BLOCKS
                         ) -> Dict[str, np.ndarray]:
    """matrices: [segments, bands, passes, passes]. The reproduced power is
    the mean off-diagonal cross power, the total the mean diagonal, the
    unreproduced their difference; the 1/N law is fitted to the power of
    subset means; every figure carries a block-jackknife error."""
    segments, bands, n, _ = matrices.shape
    subsets = {size: list(itertools.combinations(range(n), size))
               for size in range(1, n + 1)}
    for size in subsets:
        if len(subsets[size]) > 64:
            rng = np.random.default_rng(size)
            picks = rng.choice(len(subsets[size]), 64, replace=False)
            subsets[size] = [subsets[size][i] for i in picks]
    ones_over_n = np.array([1.0 / size for size in range(1, n + 1)])
    design = np.column_stack([np.ones(n), ones_over_n])

    def estimate(mean_matrix):
        diagonal = np.einsum("bii->bi", mean_matrix).mean(axis=1)
        if n > 1:
            off = (mean_matrix.sum(axis=(1, 2)) - np.einsum("bii->b", mean_matrix)) / (n * (n - 1))
        else:
            off = np.full(bands, np.nan)
        power_of_mean = np.zeros((bands, n))
        for size in range(1, n + 1):
            values = []
            for subset in subsets[size]:
                idx = np.array(subset)
                values.append(mean_matrix[:, idx][:, :, idx].sum(axis=(1, 2)) / size ** 2)
            power_of_mean[:, size - 1] = np.mean(values, axis=0)
        if n > 1:
            coefficients = np.linalg.lstsq(design, power_of_mean.T, rcond=None)[0]
            frozen_fit, falling_fit = coefficients[0], coefficients[1]
            fitted = design @ coefficients
            misfit = np.sqrt(np.mean((fitted - power_of_mean.T) ** 2, axis=0))
        else:
            frozen_fit = falling_fit = misfit = np.full(bands, np.nan)
        return {
            "total": diagonal, "reproduced": off,
            "unreproduced": diagonal - off,
            "share": np.where(diagonal > 0, off / np.maximum(diagonal, 1e-300), np.nan),
            "power_of_mean": power_of_mean,
            "one_over_n_frozen": frozen_fit,
            "one_over_n_falling": falling_fit,
            "one_over_n_misfit": misfit,
        }

    full = estimate(matrices.mean(axis=0))
    blocks = max(min(int(blocks), segments), 2)
    edges = np.linspace(0, segments, blocks + 1).astype(int)
    leave_out = []
    for k in range(blocks):
        keep = np.ones(segments, dtype=bool)
        keep[edges[k]:edges[k + 1]] = False
        if keep.sum() == 0:
            continue
        leave_out.append(estimate(matrices[keep].mean(axis=0)))
    errors = {}
    m = len(leave_out)
    for key in ("total", "reproduced", "unreproduced", "share",
                "one_over_n_frozen", "one_over_n_falling"):
        stack = np.array([e[key] for e in leave_out])
        errors[key + "_error"] = np.sqrt((m - 1) / m * np.sum(
            (stack - stack.mean(axis=0)) ** 2, axis=0)) if m > 1 else np.full(bands, np.nan)
    full.update(errors)
    full["segments"] = segments
    full["jackknife_blocks"] = m
    return full


def direct_split(passes: Sequence[np.ndarray], sample_rate_hz: float, bands,
                 valid: Optional[np.ndarray] = None, segment: int = 1 << 14
                 ) -> Dict[str, object]:
    """The split on the real signals as they are: Hann-windowed segments at
    half overlap, cross powers per band. Reproduced power here includes the
    signal, which reproduces; the estimator is transfer-invariant."""
    rate = float(sample_rate_hz)
    x = [np.asarray(p, dtype=np.float64) for p in passes]
    length = min(len(p) for p in x)
    hop = segment // 2
    window = np.hanning(segment)
    frequencies = np.fft.rfftfreq(segment, d=1.0 / rate)
    slices = _band_slices(frequencies, bands)
    matrices = []
    for start in range(0, length - segment + 1, hop):
        if valid is not None and not valid[start:start + segment].all():
            continue
        spectra = np.array([np.fft.rfft(p[start:start + segment] * window) for p in x])
        matrices.append(_cross_matrices_from_spectra(spectra, slices))
    if not matrices:
        return {"segments": 0}
    matrices = np.array(matrices)
    # normalise to power density per hertz: Hann window energy, and the
    # one-sided spectrum of a real signal
    scale = 2.0 / (rate * np.sum(window ** 2))
    widths = np.array([high - low for _, low, high in bands])
    result = _split_from_matrices(matrices * scale)
    for key in ("total", "reproduced", "unreproduced", "one_over_n_frozen",
                "one_over_n_falling", "total_error", "reproduced_error",
                "unreproduced_error", "one_over_n_frozen_error",
                "one_over_n_falling_error", "one_over_n_misfit"):
        result[key + "_density"] = result[key] / widths
    result["power_of_mean_density"] = result["power_of_mean"] / widths[:, None]
    result["bands"] = list(bands)
    result["duration_s"] = matrices.shape[0] * hop / rate
    result["independent_averages"] = widths * result["duration_s"]
    return result


def comb_split(analytic_passes: Sequence[np.ndarray], table: Dict[str, object],
               parameters: Dict[str, object], sample_rate_hz: float, bands,
               consecutive: Optional[Sequence[Dict[str, np.ndarray]]] = None,
               content_feature: Optional[np.ndarray] = None
               ) -> Dict[str, object]:
    """The split of the NOISE inside the signal's band: every pass's lines
    are differenced against their neighbours (each aligned and rotated from
    its own blanking windows, on the reference's line table), which removes
    line-periodic signal, and the cross powers of those differences are
    split exactly as the direct estimator's. Lines whose content changes
    are excluded by their own line-to-line scatter."""
    rate = float(sample_rate_hz)
    length = int(round(table["line_samples"]))
    passes = [np.asarray(p) for p in analytic_passes]
    if consecutive is None:
        consecutive = [consecutive_line_lags(p, table, parameters, rate) for p in passes]
    # content changes: lines whose active feature moves by more than the
    # flat field's own line-to-line scatter
    steady = np.ones(table["count"], dtype=bool)
    if content_feature is not None and table["count"] > 3:
        step = np.abs(np.diff(content_feature))
        finite = step[np.isfinite(step)]
        if finite.size:
            median = np.median(finite)
            mad = np.median(np.abs(finite - median)) or (np.std(finite) + 1e-9)
            changing = np.flatnonzero(step > median + CONTENT_CHANGE_MAD_MULTIPLE * mad)
            for n in changing:
                steady[max(n - 1, 0):n + 2] = False
    window = np.hanning(length)
    frequencies = np.fft.fftfreq(length, d=1.0 / rate)
    slices = _band_slices(frequencies, bands)
    matrices, kept, cancellation = [], [], []
    for n in range(table["count"] - 2):
        if not steady[n] or not steady[n + 1]:
            continue
        deltas = [line_difference(p, c, n, length) for p, c in zip(passes, consecutive)]
        if any(d is None for d in deltas):
            continue
        spectra = np.array([np.fft.fft(d * window) for d in deltas])
        matrices.append(_cross_matrices_from_spectra(spectra, slices))
        kept.append(n)
        start = int(consecutive[0]["centre"][n]) - consecutive[0]["before"]
        line_power = float(np.sum(np.abs(passes[0][start:start + length]) ** 2))
        cancellation.append(float(np.sum(np.abs(deltas[0]) ** 2)) / max(line_power, 1e-300))
    if not matrices:
        return {"segments": 0}
    matrices = np.array(matrices)
    # analytic signal: one-sided already, so no factor of two; and the line
    # difference doubles independent noise power (two lines' worth), which
    # is divided back out so densities compare with the direct estimator's
    scale = 1.0 / (rate * np.sum(window ** 2)) / 2.0
    widths = np.array([high - low for _, low, high in bands])
    result = _split_from_matrices(matrices * scale)
    for key in ("total", "reproduced", "unreproduced", "one_over_n_frozen",
                "one_over_n_falling", "total_error", "reproduced_error",
                "unreproduced_error", "one_over_n_frozen_error",
                "one_over_n_falling_error", "one_over_n_misfit"):
        result[key + "_density"] = result[key] / widths
    result["bands"] = list(bands)
    result["lines"] = np.array(kept)
    result["excluded_content_lines"] = int((~steady).sum())
    result["signal_cancellation"] = np.array(cancellation)
    result["duration_s"] = len(kept) * length / rate
    result["independent_averages"] = widths * result["duration_s"]
    return result


def floor_split(passes: Sequence[np.ndarray], sample_rate_hz: float,
                parameters: Optional[Dict[str, object]] = None,
                bands=None, valid: Optional[np.ndarray] = None,
                table: Optional[Dict[str, object]] = None,
                content_feature: Optional[np.ndarray] = None,
                segment: int = 1 << 14) -> Dict[str, object]:
    """Reproduced (frozen) and unreproduced power per band, with errors,
    across N aligned passes on one time base.

    `passes[0]` is the reference; every other pass must already be on its
    time base (`resample_onto`). With `table` (the reference's line table)
    the comb estimator is run too, which is the one that splits the NOISE
    inside the carrier's band. Both carry the 1/N fit: the power of the
    mean of n passes is frozen + falling / n, and the two coefficients are
    the frozen and the per-pass unreproduced power by a route that shares
    nothing with the cross-power estimate.
    """
    rate = float(sample_rate_hz)
    parameters = parameters or format_parameters()
    bands = bands or default_bands(parameters, rate)
    result = {"bands": list(bands), "n_passes": len(passes),
              "parameters": parameters, "sample_rate_hz": rate}
    result["direct"] = direct_split(passes, rate, bands, valid, segment)
    if table is not None:
        analytic = [analytic_signal(np.asarray(p, dtype=np.float64)
                                    - float(np.mean(p)), rate) for p in passes]
        result["comb"] = comb_split(analytic, table, parameters, rate, bands,
                                    content_feature=content_feature)
    return result


def characterise_residual(residual: np.ndarray, sample_rate_hz: float,
                          band_hz: Tuple[float, float]) -> Dict[str, object]:
    """What the unreproduced part looks like in one band, by its
    distribution: Gaussian for electronics, uniform for quantisation,
    arcsine for a surviving carrier, heavy-tailed for dropouts."""
    analytic = analytic_signal(np.asarray(residual, dtype=np.float64),
                               float(sample_rate_hz), band_hz=band_hz)
    real = np.real(analytic)
    verdict = interference_distributions.classify(real)
    verdict["carrier"] = interference_distributions.carrier_fraction(real)
    return verdict


# --------------------------------------------------------------------------
# Resolution of the split: Ethan's 10-20 dB, derived
# --------------------------------------------------------------------------


def independent_averages(band_hz: float, duration_s: float) -> float:
    """The time-bandwidth product: a band of width B observed for T holds
    B T independent complex spectral estimates, whatever estimator is used.
    A Hann-windowed half-overlap Welch estimate reaches about this many;
    it cannot exceed it."""
    return float(band_hz) * float(duration_s)


def resolution_limit(floor_db: float, n_passes: int, averages: float,
                     sigma: float = 3.0) -> Dict[str, float]:
    """How far below the unreproduced floor a frozen component can be seen.

    Derivation. In a band with M independent estimates, N passes, frozen
    power F and unreproduced power U per pass, the estimator is the power
    of the N-pass mean less the unreproduced power over N:
        F^ = <|mean|^2> - U^/N.
    The mean's power per bin is exponential with expectation F + U/N, so
    its variance over M estimates is (F + U/N)^2 / M; the unreproduced
    estimate has N-1 degrees of freedom per bin, variance U^2 / (M (N-1)),
    and is independent of the mean. Near the limit (F << U):
        var F^ = (U/N)^2 / M * (1 + 1/(N-1)),
        se  F^ = U sqrt(1 + 1/(N-1)) / (N sqrt(M)).
    A frozen power is resolved at `sigma` standard errors when
        F_min / U = sigma sqrt(1 + 1/(N-1)) / (N sqrt(M)),
    which for two passes is sigma / sqrt(2 M): ten to twenty decibels
    below the floor for M of 10^2 to 10^4, and further for whole captures.
    For N = 1 there is no split and the limit is the floor itself.
    """
    n = int(n_passes)
    m = float(averages)
    if n < 2 or m <= 0:
        return {"floor_db": float(floor_db), "n_passes": n, "averages": m,
                "resolvable_share": 1.0, "below_floor_db": 0.0,
                "resolvable_level_db": float(floor_db), "sigma": float(sigma)}
    share = float(sigma) * np.sqrt(1.0 + 1.0 / (n - 1)) / (n * np.sqrt(m))
    below = 10.0 * np.log10(share)
    return {
        "floor_db": float(floor_db),
        "n_passes": n,
        "averages": m,
        "sigma": float(sigma),
        "resolvable_share": float(share),
        "below_floor_db": float(below),
        "resolvable_level_db": float(floor_db) + float(below),
        "why": ("standard error of the frozen estimate near the limit is "
                "U sqrt(1 + 1/(N-1)) / (N sqrt(M)); averaging over M "
                "independent estimates and N passes"),
    }


def passes_for_resolution(target_below_floor_db: float, averages: float,
                          sigma: float = 3.0, maximum: int = 1024) -> int:
    """The smallest N that reaches a target resolution, from the same law."""
    for n in range(2, maximum + 1):
        if resolution_limit(0.0, n, averages, sigma)["below_floor_db"] <= target_below_floor_db:
            return n
    return maximum


# --------------------------------------------------------------------------
# Cross-track correlation from a tracking-offset sweep
# --------------------------------------------------------------------------


def tracking_offset_to_cross_track(longitudinal_m, track_angle_degrees: float
                                   ) -> np.ndarray:
    """The tracking control shifts the tape longitudinally (the capstan's
    phase); the head crosses the track by that shift times the sine of the
    track angle."""
    return np.asarray(longitudinal_m, dtype=np.float64) \
        * np.sin(np.radians(float(track_angle_degrees)))


def cross_track_model(offsets_m, correlation_length_m: float,
                      head_width_m: float) -> np.ndarray:
    """Reproduced share of frozen noise against cross-track offset.

    Two reads of the same coating offset by `delta` share the noise from
    the coating both heads covered. For a head of width w reading noise
    whose cross-track correlation is c(u), the normalised correlation of
    the two reads is the head's overlap triangle Lambda_w(u) = 1 - |u|/w
    convolved with c, normalised at zero: a bare triangle for a coating
    whose correlation length is far below the head width, a rounded one
    whose rounding measures the correlation length otherwise. c is taken
    Gaussian with standard deviation `correlation_length_m`.
    """
    offsets = np.abs(np.asarray(offsets_m, dtype=np.float64))
    w = float(head_width_m)
    lam = max(float(correlation_length_m), 1e-12)
    span = w + 6.0 * lam
    step = min(w, lam) / 64.0
    grid = np.arange(-span, span + step, step)
    triangle = np.clip(1.0 - np.abs(grid) / w, 0.0, None)
    kernel = np.exp(-0.5 * (grid / lam) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(triangle, kernel, mode="same")
    smoothed /= smoothed[np.argmin(np.abs(grid))]
    return np.interp(offsets, grid, smoothed, left=0.0, right=0.0)


def cross_track_correlation(offsets_m, reproduced_share, track_width_m: float,
                            head_width_m: Optional[float] = None,
                            share_error=None) -> Dict[str, object]:
    """Fit the coating's cross-track correlation length to a sweep of the
    reproduced share against tracking offset.

    The head width is the track width by default (a VHS head of the format's
    own width) and may be fitted instead by passing None for both; the
    correlation length's error comes from the Jacobian at the solution.
    Design: offsets inside a few correlation lengths resolve the rounding of
    the cusp; offsets out to the head width check the triangle, and
    reproduced share falling to zero beyond it is the confirmation that
    what was being measured was frozen in the coating.
    """
    from scipy.optimize import least_squares

    offsets = np.asarray(offsets_m, dtype=np.float64)
    share = np.asarray(reproduced_share, dtype=np.float64)
    weight = (1.0 / np.maximum(np.asarray(share_error, dtype=np.float64), 1e-6)
              if share_error is not None else np.ones_like(share))
    fit_width = head_width_m is None
    width0 = float(track_width_m) if fit_width else float(head_width_m)

    def residual(p):
        lam = p[0]
        w = p[1] if fit_width else width0
        return (cross_track_model(offsets, lam, w) - share) * weight

    start = [0.1 * width0] + ([width0] if fit_width else [])
    lower = [1e-3 * width0] + ([0.2 * width0] if fit_width else [])
    upper = [5.0 * width0] + ([5.0 * width0] if fit_width else [])
    solution = least_squares(residual, start, bounds=(lower, upper))
    jacobian = solution.jac
    dof = max(offsets.size - solution.x.size, 1)
    variance = float(np.sum(solution.fun ** 2)) / dof
    try:
        covariance = variance * np.linalg.inv(jacobian.T @ jacobian)
        errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    except np.linalg.LinAlgError:
        errors = np.full(solution.x.size, np.nan)
    lam = float(solution.x[0])
    width = float(solution.x[1]) if fit_width else width0
    return {
        "correlation_length_m": lam,
        "correlation_length_error_m": float(errors[0]),
        "head_width_m": width,
        "head_width_error_m": float(errors[1]) if fit_width else 0.0,
        "resolved": bool(np.isfinite(errors[0]) and errors[0] < 0.5 * lam),
        "model": cross_track_model(offsets, lam, width),
        "residual_rms": float(np.sqrt(np.mean((cross_track_model(offsets, lam, width) - share) ** 2))),
        "why": ("the reproduced share against offset is the head's overlap "
                "triangle convolved with the coating's correlation; the "
                "rounding of the cusp at zero offset is the correlation "
                "length"),
    }


# --------------------------------------------------------------------------
# Planted data: one recording, N replays
# --------------------------------------------------------------------------


def synthesise_passes(n_passes: int, lines: int = 30, sample_rate_hz: float = 50e6,
                      parameters: Optional[Dict[str, object]] = None,
                      frozen_noise_db: float = -38.0,
                      independent_noise_db: float = -46.5,
                      frozen_band_hz: Tuple[float, float] = (0.3e6, 6.0e6),
                      wander_samples: float = 3.0, rate_error: float = 1e-4,
                      gain_spread: float = 0.1, carrier_codes: float = 35.0,
                      dc_offset_codes: float = -33.0, quantise: bool = True,
                      active_ire: float = 50.0, bar_ire: Optional[float] = None,
                      frozen_components: int = 400, seed: int = 0
                      ) -> Dict[str, object]:
    """One FM recording with a FROZEN noise realisation, replayed N times
    through independent time-base wander, gain, DC offset, independent
    noise and 8-bit quantisation.

    The field carries the format's nine-line vertical interval (three lines
    of equalising pulses, three of broad pulses, three of equalising, at
    half-line spacing) followed by ordinary lines at `active_ire`, with an
    optional bar. The FM follows the format's carrier figures. The frozen
    noise is a sum of random sinusoids so it can be evaluated exactly at
    the warped sample times; each pass's wander is a sinusoid over the
    field plus a rate error, applied by evaluating the recording's phase at
    the warped time - no interpolator of this module is used to make the
    data it is later tested on. Noise levels are relative to the carrier
    power, per sample over the Nyquist span for the independent noise and
    over its own band for the frozen noise.
    """
    rate = float(sample_rate_hz)
    parameters = parameters or format_parameters()
    rng = np.random.default_rng(seed)
    line = int(round(float(parameters["line_period_s"]) * rate))
    total = lines * line
    time = np.arange(total) / rate

    def pulse(ire, start_s, width_s, rise_s=0.14e-6):
        # SMPTE 170M sync rise 140 ns: a raised-cosine edge of that width
        t = time - start_s
        edge_in = np.clip((t + 0.5 * rise_s) / rise_s, 0.0, 1.0)
        edge_out = np.clip((t - width_s + 0.5 * rise_s) / rise_s, 0.0, 1.0)
        shape = 0.5 - 0.5 * np.cos(np.pi * edge_in)
        shape *= 0.5 + 0.5 * np.cos(np.pi * edge_out)
        ire += float(parameters["hz_per_ire"] and -40.0) * shape * 0.0  # placeholder no-op
        return shape

    ire = np.zeros(total)
    sync_depth_ire = -40.0                      # the format's sync level
    period = float(parameters["line_period_s"])
    sync_w = float(parameters["sync_width_s"])
    eq_w = float(parameters["equalizing_width_s"])
    broad_w = float(parameters["broad_width_s"])
    blanking_end = float(parameters["blanking_end_s"])
    front_porch = float(parameters["front_porch_s"])
    vertical = 9 if lines > 12 else 0
    for n in range(lines):
        start = n * period
        if n < vertical:
            width = eq_w if (n < 3 or n >= 6) else broad_w
            for half in (0.0, 0.5 * period):
                ire += sync_depth_ire * pulse(None, start + half, width)
        else:
            ire += sync_depth_ire * pulse(None, start, sync_w)
            active = pulse(None, start + blanking_end, period - blanking_end - front_porch)
            ire += active_ire * active
            if bar_ire is not None:
                ire += (bar_ire - active_ire) * pulse(None, start + blanking_end + 0.3 * period,
                                                      0.3 * period)
    frequency = float(parameters["blanking_hz"]) + float(parameters["hz_per_ire"]) * ire
    phase = 2.0 * np.pi * np.cumsum(frequency) / rate

    # frozen noise as random sinusoids in the band, exact at any time
    k = int(frozen_components)
    frozen_hz = rng.uniform(frozen_band_hz[0], frozen_band_hz[1], k)
    frozen_phase = rng.uniform(0.0, 2.0 * np.pi, k)
    carrier_power = 0.5
    frozen_amplitude = np.sqrt(2.0 * carrier_power * 10 ** (frozen_noise_db / 10.0) / k)

    def frozen_at(positions_s):
        out = np.zeros(positions_s.size)
        for j in range(0, k, 64):
            out += np.sum(frozen_amplitude * np.cos(
                2.0 * np.pi * frozen_hz[j:j + 64, None] * positions_s[None, :]
                + frozen_phase[j:j + 64, None]), axis=0)
        return out

    independent_sigma = np.sqrt(carrier_power * 10 ** (independent_noise_db / 10.0))
    passes, warps, gains, offsets = [], [], [], []
    sample_index = np.arange(total, dtype=np.float64)
    for i in range(int(n_passes)):
        amplitude = wander_samples * rng.uniform(0.5, 1.0)
        cycles = rng.uniform(0.5, 1.5)
        start_phase = rng.uniform(0.0, 2.0 * np.pi)
        drift = rate_error * rng.uniform(-1.0, 1.0)
        warp = amplitude * np.sin(2.0 * np.pi * cycles * sample_index / total + start_phase) \
            + drift * sample_index
        warped = sample_index + warp                      # recording time, samples
        signal = np.cos(np.interp(warped, sample_index, phase))
        signal += frozen_at(warped / rate)
        gain = 1.0 + gain_spread * rng.uniform(-1.0, 1.0)
        noise = rng.normal(0.0, independent_sigma, total)
        codes = carrier_codes * gain * (signal + noise) + dc_offset_codes
        if quantise:
            codes = np.rint(codes)
        passes.append(codes)
        warps.append(warp)
        gains.append(gain)
        offsets.append(dc_offset_codes)
    scale = carrier_codes ** 2
    return {
        "passes": passes,
        "warps": warps,
        "gains": gains,
        "sample_rate_hz": rate,
        "parameters": parameters,
        "lines": lines,
        "line_samples": line,
        "vertical_interval_lines": vertical,
        "frozen_hz": frozen_hz,
        "frozen_power_codes2": scale * frozen_amplitude ** 2 / 2.0 * np.ones(k),
        "independent_density_codes2_per_hz": scale * independent_sigma ** 2 / (0.5 * rate),
        "carrier_power_codes2": scale * carrier_power,
        "signal_phase": phase,
        "true_lag": lambda i, j, n: _true_lag(warps[i], warps[j], n),
    }


def _true_lag(warp_i: np.ndarray, warp_j: np.ndarray, n) -> np.ndarray:
    """Pass j's sample position for pass i's sample n, from the planted
    warps: m + warp_j(m) = n + warp_i(n), solved by two fixed-point steps."""
    n = np.asarray(n, dtype=np.float64)
    index = np.arange(warp_i.size, dtype=np.float64)
    target = n + np.interp(n, index, warp_i)
    m = target - np.interp(n, index, warp_j)
    for _ in range(3):
        m = target - np.interp(m, index, warp_j)
    return m - n


def frozen_power_in_band(planted: Dict[str, object], low_hz: float, high_hz: float
                         ) -> float:
    """The planted frozen power inside a band, in codes squared, from the
    sinusoids that were planted there."""
    inside = (planted["frozen_hz"] >= low_hz) & (planted["frozen_hz"] < high_hz)
    return float(np.sum(planted["frozen_power_codes2"][inside]))
