"""THE DECODER'S OWN PRECURSOR, SUBTRACTED AGAINST THE MEASURED EDGE.

Ethan: convolve the precursor against the MEASURED edge - the sync edge as
measured carries the channel's own shaping - rather than an ideal step.

WHAT THE PRECURSOR IS, AND WHY IT CANNOT BE GUARDED OUT. The decoder's video
low-pass is a supergaussian of order 9 (`video_lpf_order`) on the demodulated
luminance, applied by FFT multiplication with a REAL, non-negative response.
A real response is a zero-phase filter, and a zero-phase filter's impulse
response is symmetric about lag zero: half of it acts BEFORE the input
arrives. Every edge the decoder outputs therefore carries a ripple ahead of
itself that was never on the tape - the anticausal half of the kernel
convolved with the edge - and for the sync edge that ripple lands in the
front porch, the reference level every level measurement in this arc rests
on. Built from the decoder's own constructor (`VHSRFDecode`, NTSC VHS, the
filter taken as `FVideo / FDeemp` and checked against the builder that made
it, `compute_video_filters.gen_video_lpf_params`):

    the response is real to 1.0e-16 and lies in [0, 1]
    the kernel is symmetric about lag zero to 6.9e-18 of its peak
    the anticausal and causal halves each weigh 0.4346 (the centre 0.1308)
    the half-amplitude point is 3.30 MHz - `video_lpf_freq` is the FULL
        width of the passband, 6.6 MHz, which the builder's own TODO notes
    the deemphasis applied alongside it is causal to 7.6e-16 of its peak
        (anticausal energy 8.8e-28) and contributes no precursor at all

Convolved with an IDEAL step the anticausal half predicts, as a fraction of
the step, -6.4% at 0.2 us before the edge, +4.2% at 0.3 us, -2.1% at 0.5 us,
-1.2% at 0.75 us and +0.2% at 1.5 us. On a 40 IRE sync edge that is 2.6 IRE
at 0.2 us and still 0.08 IRE at 1.5 us; the standard's front porch is 1.5 us
long, so THE WHOLE PORCH SITS INSIDE THE PRECURSOR'S SUPPORT. There is no
precursor-free part of the porch to retreat to; it has to be subtracted.

WHY THE MEASURED EDGE AND NOT A STEP. The precursor is linear in the drive,
and the ring at the band edge - the part that is largest - comes from the
drive's energy at 2.5 to 3.5 MHz. An ideal step has full energy there; the
recorded sync edge, shaped by the tape channel, has very little. So the
ideal-step precursor is wrong by however much the channel has already
rolled the edge off, and the measured edge carries exactly that roll-off.
The derivation is in `precursor`; the comparison is in `residual_check`.

THE ONE THING THE MEASURED EDGE GETS WRONG, AND THE OPTION THAT REPAIRS IT.
What is measured is the DECODED edge, which has been through the low-pass
once already; using it as the drive multiplies the drive's spectrum by the
response a second time, so the predicted ring is short by the response
itself at each frequency (a factor 0.5 at 3.3 MHz). Where the measured
edge's spectrum stands above its own noise the response can be divided
back out (`drive="deconvolved"`), restoring the drive inside the band the
average can see and leaving it as measured beyond. Both are reported.

MEASURED ON REAL TAPE - see `analyse` and the record at the end of this
docstring, which is filled in from the run.

THE CHECK AGAINST THE HYPERCOMPLEX READINGS. `tesseract` found every axis's
departure dominated by EXCESS phase - delays and all-pass. The precursor is
a ZERO-PHASE term: against the Bode relation the zero-phase low-pass carries
an excess phase of minus its own minimum phase, which is anticipation, an
all-pass in the causality decomposition. But it is the SAME kernel for every
head, every polarity, every tape, so it is common to every vertex and drops
out of every contrast - `hypercomplex.causality` reads contrasts only, and
the readings there are not this term. `minimum_phase_counterpart` builds the
causal filter with the same magnitude by the cepstral Bode relation and
checks that it has no precursor, which is the same statement made the other
way round: the precursor IS the zero-phase filter's anticipatory excess.

WHERE IT SITS. After the capture (position 30): the decoder is the last
thing that touches the signal and the first thing an inversion undoes. It
is not registered in `interference.full_chain` from here, because that
would mean editing another module; the position is declared for whoever
does.

CONVENTIONS. Sync lines only: the horizontal sync pulses of the lines
outside the vertical interval and clear of the head switch, populated from
the specification's line numbers (`sync_geometry`), never from the
waveform. Per head by field parity (`isFirstField`; which parity is head A
is not knowable from a TBC, so the labels are a convention). Levels in the
TBC's own units, converted to IRE by the JSON's `blanking16bIre` (0 IRE)
and `white16bIre` (100 IRE), which is how the decoder wrote them; the sync
depth is MEASURED, not taken as 40.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import json
import numpy as np

from vhsdecode.models import sync_geometry

CHAIN_PREFIX = "decoder precursor"
# After the capture low-pass (30): the decoder's own filtering is the last
# thing that happens to the signal.
CHAIN_POSITION = 31

COMPONENT_POSITIONS: Dict[str, int] = {
    CHAIN_PREFIX: CHAIN_POSITION,
    "decoder video low-pass precursor": CHAIN_POSITION,
}

# The irfft of a real spectrum is symmetric by construction up to rounding
# accumulated over log2(n) butterfly stages; a thousand machine epsilons is
# far above any rounding and far below any asymmetry a filter could have.
# Measured: 5.3e-17 relative on the decoder's own kernel.
SYMMETRY_TOLERANCE = 1000.0 * np.finfo(np.float64).eps

# A response smaller than this fraction of its peak cannot be told from zero
# in a 16-bit output (2^-16 = 1.5e-5), so flooring the log magnitude there
# for the cepstral construction changes nothing the output can show.
RESPONSE_FLOOR = 1e-8

# Fine-grid factor for the sub-sample crossing search: at four times the
# subcarrier one sample is 70 ns against a specified 140 ns edge, so a
# sixteenth of a sample (4.4 ns) puts thirty-two steps across the edge, and
# the linear interpolation between fine steps then resolves the crossing
# well below the noise-limited precision (about 0.08 sample per line).
CROSSING_UPSAMPLE = 16


# --------------------------------------------------------------------------
# The decoder's own low-pass, from its own constructor
# --------------------------------------------------------------------------


def decoder_video_lowpass(rf=None, *, inputfreq: float = 40,
                          system: str = "NTSC", tape_format: str = "VHS",
                          tape_speed: str = "sp") -> Dict[str, object]:
    """THE VIDEO LOW-PASS THE DECODER ACTUALLY APPLIES, from its constructor.

    `VHSRFDecode` builds `FVideo = FDeemp * lpf * FCustomVideo`; the low-pass
    is recovered by dividing the deemphasis (and any custom filter) back
    out, and checked against the builder the constructor called with the
    same parameters. The deemphasis is a causal IIR and the custom filters,
    where a format has them, are causal too, so the zero-phase factor is the
    low-pass alone - the check reports the largest imaginary part left over.
    Pass an existing decoder or let one be built.
    """
    if rf is None:
        import logging
        import lddecode.core as ldd
        if getattr(ldd, "logger", None) is None:
            ldd.logger = logging.getLogger("precursor")
        from vhsdecode import process
        rf = process.VHSRFDecode(inputfreq=inputfreq, system=system,
                                 tape_format=tape_format,
                                 rf_options={"tape_speed": tape_speed})
    from vhsdecode.compute_video_filters import gen_video_lpf_params

    full = np.asarray(rf.Filters["FVideo"], dtype=np.complex128)
    deemp = np.asarray(rf.Filters["FDeemp"], dtype=np.complex128)
    custom = rf.Filters.get("FCustomVideo", 1.0)
    recovered = full / np.where(np.abs(deemp) > 0, deemp, 1.0)
    custom_is_array = np.ndim(custom) > 0
    if custom_is_array:
        custom = np.asarray(custom, dtype=np.complex128)
        recovered = recovered / np.where(np.abs(custom) > 0, custom, 1.0)
    _, builder = gen_video_lpf_params(rf.DecoderParams, rf.freq_hz_half,
                                      rf.blocklen)
    builder = np.asarray(builder, dtype=np.float64)
    params = rf.DecoderParams
    return {
        "response": builder,
        "frequency_hz": np.linspace(0.0, rf.freq_hz_half, rf.blocklen // 2 + 1),
        "sample_rate_hz": float(rf.freq_hz),
        "blocklen": int(rf.blocklen),
        "decoder_params": params,
        "video_lpf_freq": float(params["video_lpf_freq"]),
        "video_lpf_order": float(params["video_lpf_order"]),
        "supergauss": bool(params.get("video_lpf_supergauss", False)),
        "imaginary_part": float(np.abs(recovered.imag).max()),
        "builder_deviation": float(np.abs(recovered.real - builder).max()),
        "custom_filters": bool(custom_is_array),
        "why": ("the zero-phase factor of the decoder's video filter is the "
                "low-pass alone: the deemphasis and custom filters are causal"),
    }


def filter_kernel(lowpass: Dict[str, object],
                  sample_rate_hz: Optional[float] = None,
                  length: Optional[int] = None) -> Dict[str, object]:
    """THE KERNEL, SPLIT INTO ITS ANTICAUSAL AND CAUSAL HALVES, VERIFIED.

    On the decoder's own grid by default, or on another rate's - the TBC's
    four-times-subcarrier grid is what the subtraction runs on. The response
    on a new grid comes from the decoder's OWN builder with the decoder's
    own parameters, and is checked against the decoder's response
    interpolated onto that grid (`resample_deviation`): the filter passes
    nothing above about 4 MHz, well inside either grid's Nyquist, so the two
    are the same continuous filter sampled twice.

    The split point is lag zero: the anticausal half is every lag before
    it, the causal half lag zero and after. Verified numerically rather
    than assumed - the peak is at lag zero, the two sides agree to
    `symmetry_error`, and the two halves weigh the same. The anticausal
    half's running sum, `step_response_anticausal`, is what an ideal step
    produces ahead of itself and is the convolution kernel `precursor` uses.
    """
    own_rate = float(lowpass["sample_rate_hz"])
    rate = own_rate if sample_rate_hz is None else float(sample_rate_hz)
    if length is None:
        # the same duration as the decoder's own block, as a power of two
        length = 1 << int(np.ceil(np.log2(
            int(lowpass["blocklen"]) * rate / own_rate)))
    length = int(length)
    if rate == own_rate and length == int(lowpass["blocklen"]):
        response = np.asarray(lowpass["response"], dtype=np.float64)
    else:
        from vhsdecode.compute_video_filters import gen_video_lpf_params
        _, response = gen_video_lpf_params(lowpass["decoder_params"],
                                           rate / 2.0, length)
        response = np.asarray(response, dtype=np.float64)
    grid = np.linspace(0.0, rate / 2.0, length // 2 + 1)
    own = np.interp(grid, np.asarray(lowpass["frequency_hz"]),
                    np.asarray(lowpass["response"]), right=0.0)
    kernel = np.fft.irfft(response, n=length)
    shifted = np.fft.fftshift(kernel)
    centre = length // 2
    peak = float(np.abs(shifted).max())
    peak_lag = int(np.argmax(np.abs(shifted))) - centre
    left = shifted[centre - 1:0:-1]
    right = shifted[centre + 1:2 * centre]
    symmetry = float(np.abs(left - right).max() / peak)
    anticausal = shifted[:centre]
    causal = shifted[centre:]
    return {
        "kernel": shifted,
        "lags": np.arange(length) - centre,
        "centre": centre,
        "sample_rate_hz": rate,
        "length": length,
        "response": response,
        "frequency_hz": grid,
        "anticausal": anticausal,
        "causal": causal,
        "step_response_anticausal": np.cumsum(anticausal),
        "anticausal_weight": float(anticausal.sum()),
        "causal_weight": float(causal[1:].sum()),
        "centre_weight": float(causal[0]),
        "peak_lag": peak_lag,
        "symmetry_error": symmetry,
        "wrap": float(np.abs(shifted[0]) / peak),
        "resample_deviation": float(np.abs(own - response).max()),
        "split_point_verified": bool(peak_lag == 0
                                     and symmetry < SYMMETRY_TOLERANCE),
        "why": ("a real response is a zero-phase filter whose kernel is "
                "symmetric about lag zero; the lags before zero act ahead "
                "of the input and are the precursor's whole cause"),
    }


def support_samples(kernel: Dict[str, object], amplitude: float,
                    noise_rms: float) -> int:
    """HOW FAR AHEAD OF AN EDGE THE PRECURSOR STANDS ABOVE A GIVEN NOISE.

    The window's length, from the kernel and the noise and nothing else:
    the largest lead at which an edge of `amplitude` still produces a
    precursor larger than `noise_rms`. Zero when the noise swamps it.
    """
    step = np.asarray(kernel["step_response_anticausal"]) * float(amplitude)
    above = np.nonzero(np.abs(step) > float(noise_rms))[0]
    if above.size == 0:
        return 0
    return int(kernel["centre"] - above[0])


# --------------------------------------------------------------------------
# The precursor from a drive
# --------------------------------------------------------------------------


def precursor(drive, kernel: Dict[str, object],
              onset: Optional[int] = None) -> np.ndarray:
    """THE ANTICAUSAL HALF CONVOLVED AGAINST THE DRIVE'S DERIVATIVE.

    Derivation. The decoder's output is `y = g * u` with `g` the zero-phase
    kernel and `u` the signal entering the low-pass. Split `g` at lag zero
    into `g_a` (lags before zero) and `g_c` (lag zero and after). At a time
    `t` ahead of an edge, `g_c` sees only the past, where `u` is at its
    reference level, so everything `y` shows there beyond that level comes
    from `g_a` reaching forward into the edge:

        p[t] = sum_{tau < 0} g[tau] (u[t - tau] - u[t])

    Writing the forward difference as a sum of the drive's increments
    `d[s] = u[s] - u[s-1]` and exchanging the sums,

        p[t] = sum_{s > t} d[s] G_a[t - s],   G_a[v] = sum_{tau <= v} g[tau]

    which is the derivative of the drive at every LATER time, weighted by
    the anticausal half's step response at the negative lag between them.
    An ideal unit step at `s0` gives `p[t] = G_a[t - s0]` directly. The drive
    beyond the end of the record is taken to hold its last value, which is
    the same as having no increments there.

    THE DRIVE IS THE TRANSITION AND WHAT FOLLOWS IT, NOT THE PORCH ITSELF.
    With `onset` given, only increments at `s >= onset` drive the sum. This
    is not a convenience. The measured edge is the DECODED edge, and ahead
    of its transition it already carries the porch - the precursor, the
    porch's own settling, the channel's ring - so feeding that region back
    in as drive predicts the anticausal half's response to the ripple being
    predicted: the nearest lag's weight (0.27 on the four-times-subcarrier
    grid) times the ripple's own slope, a quarter cycle out of phase with
    it. Measured on the 75-bar SP decode with the whole record as drive the
    free gain came out -1.3 to -2.6; restricted to the transition it is the
    positive value the record below reports. The onset is the transition's
    10 per cent point, where the drive's own increments begin.

    `drive` is on the kernel's grid. Returns `p` over the whole record; the
    caller applies it ahead of the edge (`subtract`).
    """
    u = np.asarray(drive, dtype=np.float64).ravel()
    n = u.size
    increments = np.diff(u)                       # increments[s - 1] = d[s]
    first = 1 if onset is None else max(1, int(onset))
    step = np.asarray(kernel["step_response_anticausal"])
    centre = int(kernel["centre"])
    out = np.zeros(n, dtype=np.float64)
    for t in range(n - 1):
        later = np.arange(max(t + 1, first), n)   # s > t, s >= onset
        if later.size == 0:
            continue
        lags = t - later + centre                 # G_a index for v = t - s
        keep = lags >= 0
        out[t] = float(np.dot(increments[later[keep] - 1], step[lags[keep]]))
    return out


def postcursor(drive, kernel: Dict[str, object],
               cutoff: Optional[int] = None) -> np.ndarray:
    """THE SAME KERNEL'S TAIL BEHIND THE TRANSITION THAT OPENS THE PORCH.

    The front porch has a transition at each end: the sync edge that closes
    it, whose PRECURSOR the anticausal half puts into the porch from ahead,
    and the blanking edge that opens it, whose POSTCURSOR the causal half
    leaves in the porch from behind. By the kernel's symmetry the two are
    mirror images: a step's output at lag `v` after it is short of its
    settled value by the kernel's tail beyond `v`,

        q[t] = - sum_{s <= t} d[s] T[t - s],   T[v] = sum_{tau > v} g[tau]

    and `T[v]` equals the anticausal step response at `-v-1`. Together with
    `precursor` this is the whole of `(g - delta) * u`, the filter's entire
    departure from identity; the two are kept apart because they are driven
    from opposite sides of the porch and the sync-only rule treats the
    sides differently (the preceding content is admitted only as the drive
    of a reference-level reaction). With `cutoff` given only increments at
    `s <= cutoff` drive the sum - the transition that opens the porch and
    whatever precedes it, never the porch itself, for the reason given at
    `precursor`.
    """
    u = np.asarray(drive, dtype=np.float64).ravel()
    n = u.size
    increments = np.diff(u)
    last = n - 1 if cutoff is None else min(n - 1, int(cutoff))
    step = np.asarray(kernel["step_response_anticausal"])
    centre = int(kernel["centre"])
    out = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        earlier = np.arange(1, min(t, last) + 1)  # s <= t, s <= cutoff
        if earlier.size == 0:
            continue
        tail_index = centre - (t - earlier) - 1   # T[v] = G_a[-v-1]
        keep = tail_index >= 0
        out[t] = -float(np.dot(increments[earlier[keep] - 1],
                               step[tail_index[keep]]))
    return out


def precursor_direct(drive, kernel: Dict[str, object]) -> np.ndarray:
    """The same quantity by the direct sum over the anticausal lags, kept
    as the identity check on `precursor`: the two agree to rounding."""
    u = np.asarray(drive, dtype=np.float64).ravel()
    n = u.size
    anticausal = np.asarray(kernel["anticausal"])
    centre = int(kernel["centre"])
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        lags = np.arange(-centre, 0)              # tau < 0
        ahead = t - lags                          # t - tau > t
        values = u[np.minimum(ahead, n - 1)]      # hold beyond the record
        out[t] = float(np.dot(anticausal, values - u[t]))
    return out


def specified_edge(length: int, crossing: float, depth: float,
                   sample_rate_hz: float, edge_us: float,
                   level: float = 0.0) -> np.ndarray:
    """THE IDEAL: the standard's own sync edge, a raised cosine of the
    specified transition time (BT.1700 row s, `sync_geometry.PULSE_EDGE_NS`),
    from `level` to `level + depth` with its half-amplitude point at
    `crossing` (fractional samples). With `edge_us = 0` it is a pure step."""
    t = (np.arange(int(length), dtype=np.float64) - float(crossing)) \
        / float(sample_rate_hz) * 1e6
    if float(edge_us) <= 0.0:
        gate = (t >= 0.0).astype(np.float64)
    else:
        x = np.clip(t / float(edge_us) + 0.5, 0.0, 1.0)
        gate = 0.5 - 0.5 * np.cos(np.pi * x)
    return float(level) + float(depth) * gate


def subtract(record, predicted, before_index: int) -> np.ndarray:
    """Remove the predicted precursor from the samples AHEAD of the edge -
    every sample before `before_index` - and leave the edge and everything
    after it untouched. The precursor is defined ahead of the edge; behind
    it the anticausal half's action is part of the filter's passband shape,
    which is not what is being removed."""
    out = np.array(record, dtype=np.float64, copy=True)
    p = np.asarray(predicted, dtype=np.float64)
    stop = max(0, min(int(before_index), out.size))
    out[:stop] -= p[:stop]
    return out


# --------------------------------------------------------------------------
# The drive restored inside the band the average can see
# --------------------------------------------------------------------------


def deconvolved_drive(edge, kernel: Dict[str, object], noise_rms: float
                      ) -> Dict[str, object]:
    """THE MEASURED EDGE WITH THE LOW-PASS DIVIDED BACK OUT, where it can be.

    The decoded edge is the drive times the response; dividing by the
    response restores the drive, but only where the edge's spectrum stands
    above its noise - beyond that the division amplifies noise by the
    inverse response, which grows without bound. The band is taken
    CONTIGUOUS from DC up to the first bin whose power fails the test, so
    a single noise bin standing up by chance where the response is tiny
    cannot be inverted. The threshold is set by the record length: a
    pure-noise bin exceeds `k` times its expected power with probability
    `exp(-k)`, and `k = ln(bins / 0.1)` keeps the expected number of false
    bins across the record below a tenth. The record's ends are joined by
    a linear ramp before the transform (an edge is not periodic) and the
    ramp is put back after.
    """
    y = np.asarray(edge, dtype=np.float64).ravel()
    n = y.size
    ramp = y[0] + (y[-1] - y[0]) * np.arange(n) / max(n - 1, 1)
    spectrum = np.fft.rfft(y - ramp)
    bins = spectrum.size
    from vhsdecode.compute_video_filters import gen_video_lpf_params
    # the decoder's own builder on the record's own grid; a record shorter
    # than a few samples has no band to restore
    _, response = gen_video_lpf_params(
        kernel.get("decoder_params") or {}, kernel["sample_rate_hz"] / 2.0, n) \
        if kernel.get("decoder_params") else (None, np.interp(
            np.linspace(0.0, kernel["sample_rate_hz"] / 2.0, bins),
            kernel["frequency_hz"], kernel["response"]))
    response = np.asarray(response, dtype=np.float64)
    noise_power = float(noise_rms) ** 2 * n          # per rfft bin, white
    threshold = np.log(bins / 0.1)
    above = np.abs(spectrum) ** 2 > threshold * noise_power
    band = 0
    while band < bins and above[band] and response[band] > RESPONSE_FLOOR:
        band += 1
    restored = spectrum.copy()
    restored[:band] = spectrum[:band] / response[:band]
    drive = np.fft.irfft(restored, n=n) + ramp
    return {
        "drive": drive,
        "restored_bins": int(band),
        "restored_to_hz": float(band / bins * kernel["sample_rate_hz"] / 2.0),
        "threshold": float(threshold),
        "why": ("the low-pass is divided back out of the measured edge where "
                "its spectrum stands above the noise, contiguous from DC, "
                "and left as measured beyond"),
    }


# --------------------------------------------------------------------------
# The TBC and the sync-line population
# --------------------------------------------------------------------------


def load_tbc(stem: str) -> Dict[str, object]:
    """A decoded luma TBC and its geometry, from the JSON beside it - never
    assumed. Samples are 16-bit unsigned; the field count is what the file
    holds. Levels: 0 IRE at `blanking16bIre`, 100 IRE at `white16bIre`,
    which is how the decoder wrote them (`iretohz(0)` and `iretohz(100)`)."""
    meta = json.load(open(stem + ".tbc.json"))
    vp = meta["videoParameters"]
    width, height = int(vp["fieldWidth"]), int(vp["fieldHeight"])
    raw = np.fromfile(stem + ".tbc", dtype=np.uint16)
    count = raw.size // (width * height)
    fields = raw[:count * width * height].reshape(count, height, width)
    records = meta["fields"][:count]
    dropouts = []
    for entry in records:
        d = entry.get("dropOuts") or {}
        dropouts.append(list(zip(d.get("fieldLine", []), d.get("startx", []),
                                 d.get("endx", []))))
    return {
        "fields": fields,
        "count": count,
        "width": width,
        "height": height,
        "sample_rate_hz": float(vp["sampleRate"]),
        "system": str(vp.get("system", "NTSC")),
        "units_per_ire": (float(vp["white16bIre"])
                          - float(vp["blanking16bIre"])) / 100.0,
        "blanking": float(vp["blanking16bIre"]),
        "active_end": int(vp["activeVideoEnd"]),
        "burst": (int(vp["colourBurstStart"]), int(vp["colourBurstEnd"])),
        "first_field": [bool(r["isFirstField"]) for r in records],
        "dropouts": dropouts,
        "source": stem,
    }


def sync_rows(height: int, system: str = "NTSC") -> np.ndarray:
    """THE ROWS WHOSE HORIZONTAL SYNC IS USED, from the specification's own
    line numbers. The vertical interval's rows (its `total_lines` from
    `sync_geometry.interval_geometry`) carry equalizing and field-sync
    pulses and are off limits by Ethan's rule; the head switch is
    specified 5 to 8 H ahead of V-sync (`HEAD_SWITCH_AHEAD_OF_VSYNC_H`)
    with an assumed one-line transient, so the last rows the switch can
    reach are dropped. The first usable row's front porch is the last
    equalizing line's trailing blanking, which is ordinary blanking.
    """
    geometry = sync_geometry.interval_geometry(system)
    first = int(np.ceil(float(geometry["total_lines"])))
    ahead = float(max(sync_geometry.HEAD_SWITCH_AHEAD_OF_VSYNC_H))
    reach = int(np.ceil(ahead + sync_geometry.ASSUMED_SWITCH_TRANSIENT_H))
    return np.arange(first, int(height) - reach)


def timing(tbc: Dict[str, object]) -> Dict[str, float]:
    """The specification's sync timing on the TBC's grid, in samples."""
    key = sync_geometry._system(tbc["system"])
    rate = float(tbc["sample_rate_hz"])
    return {
        "porch": sync_geometry.FRONT_PORCH_US[key] * 1e-6 * rate,
        "sync": sync_geometry.LINE_SYNC_US[key] * 1e-6 * rate,
        "edge": sync_geometry.PULSE_EDGE_NS[key] * 1e-9 * rate,
        "edge_us": sync_geometry.PULSE_EDGE_NS[key] * 1e-3,
        "line": sync_geometry.LINE_PERIOD_US[key] * 1e-6 * rate,
    }


# --------------------------------------------------------------------------
# Sub-sample alignment
# --------------------------------------------------------------------------


def shift_band_limited(record, delta: float) -> np.ndarray:
    """Move a record LATER by `delta` samples (fractional) with a Fourier
    phase ramp, exact for a band-limited record. An edge is not periodic,
    so the line joining the record's ends is removed first and put back
    shifted analytically; the Nyquist bin keeps only the real part of its
    factor, as a real output must."""
    x = np.asarray(record, dtype=np.float64).ravel()
    n = x.size
    i = np.arange(n, dtype=np.float64)
    slope = (x[-1] - x[0]) / max(n - 1, 1)
    ramp = x[0] + slope * i
    spectrum = np.fft.rfft(x - ramp)
    f = np.fft.rfftfreq(n)
    factor = np.exp(-2j * np.pi * f * float(delta))
    if n % 2 == 0:
        factor[-1] = factor[-1].real
    moved = np.fft.irfft(spectrum * factor, n=n)
    return moved + x[0] + slope * (i - float(delta))


def crossing(record, before: Tuple[int, int], after: Tuple[int, int],
             search: Tuple[int, int]) -> Dict[str, float]:
    """THE EDGE'S HALF-AMPLITUDE CROSSING, to a fraction of a sample.

    The levels either side are medians of the windows given (the reference
    level ahead of the edge, the settled level behind it); the crossing is
    where the sinc-interpolated record first passes their midpoint inside
    `search`, refined linearly between fine-grid points. Sinc interpolation
    is exact here because the low-pass leaves nothing near Nyquist.
    """
    x = np.asarray(record, dtype=np.float64).ravel()
    n = x.size
    pre = float(np.median(x[before[0]:before[1]]))
    post = float(np.median(x[after[0]:after[1]]))
    mid = 0.5 * (pre + post)
    i = np.arange(n, dtype=np.float64)
    slope = (x[-1] - x[0]) / max(n - 1, 1)
    ramp = x[0] + slope * i
    spectrum = np.fft.rfft(x - ramp)
    up = CROSSING_UPSAMPLE
    padded = np.zeros(n * up // 2 + 1, dtype=np.complex128)
    padded[:spectrum.size] = spectrum
    if n % 2 == 0:
        padded[spectrum.size - 1] *= 0.5
    fine = np.fft.irfft(padded, n=n * up) * up
    fine_i = np.arange(n * up, dtype=np.float64) / up
    fine += x[0] + slope * fine_i
    lo, hi = int(search[0]) * up, int(search[1]) * up
    segment = fine[lo:hi]
    sign = 1.0 if post < pre else -1.0
    passed = np.nonzero(sign * (segment - mid) < 0)[0]
    if passed.size == 0 or passed[0] == 0:
        return {"found": False, "pre": pre, "post": post}
    k = passed[0]
    a, b = segment[k - 1], segment[k]
    fraction = (mid - a) / (b - a) if b != a else 0.0
    return {"found": True, "crossing": (lo + k - 1 + fraction) / up,
            "pre": pre, "post": post, "depth": post - pre}


def transition_width(record, crossing_at: float, pre_level: float,
                     post_level: float, reach: int) -> float:
    """The edge's 10-to-90 per cent width in samples, on the fine grid, from
    the last 10 per cent point ahead of the crossing to the first 90 per
    cent point behind it, within `reach` samples of the crossing."""
    x = np.asarray(record, dtype=np.float64).ravel()
    n = x.size
    up = CROSSING_UPSAMPLE
    i = np.arange(n, dtype=np.float64)
    slope = (x[-1] - x[0]) / max(n - 1, 1)
    spectrum = np.fft.rfft(x - (x[0] + slope * i))
    padded = np.zeros(n * up // 2 + 1, dtype=np.complex128)
    padded[:spectrum.size] = spectrum
    if n % 2 == 0:
        padded[spectrum.size - 1] *= 0.5
    fine = np.fft.irfft(padded, n=n * up) * up
    fine += x[0] + slope * np.arange(n * up) / up
    fraction = (fine - pre_level) / (post_level - pre_level)
    c = int(round(float(crossing_at) * up))
    lo, hi = max(c - int(reach) * up, 0), min(c + int(reach) * up, fine.size)
    ahead = np.nonzero(fraction[lo:c] < 0.1)[0]
    behind = np.nonzero(fraction[c:hi] > 0.9)[0]
    if ahead.size == 0 or behind.size == 0:
        return float(reach)
    return float((c + behind[0]) - (lo + ahead[-1])) / up


def edge_records(tbc: Dict[str, object], kernel: Dict[str, object],
                 which: str = "fall", rows: Optional[Sequence[int]] = None
                 ) -> Dict[str, object]:
    """EVERY SYNC EDGE OF THE POPULATION, CUT OUT AND ALIGNED.

    The falling edge sits at the head of each TBC row, so its record is the
    tail of the previous row joined to the head of this one - the rows are
    contiguous by construction, each being one line resampled end to end.
    The rising edge is one sync width later in the same row. Ahead of the
    edge the record reaches the specified porch plus the kernel's support
    at the measured noise (so the porch window is a full support away from
    the record's end and the shift's end effects); behind it, to just short
    of the next edge's own support.

    Per line: the crossing (`crossing`), then a band-limited shift that
    puts every line's crossing on one integer index, the median over the
    tape. Lines are dropped when the crossing is not found, when the shift
    needed is more than a sample (a line the TBC did not lock), when the
    depth is more than four pooled sigmas from the median (a dropout or a
    switch remnant), or when a logged dropout touches the record.
    """
    fields = tbc["fields"]
    width, height = tbc["width"], tbc["height"]
    rate = float(tbc["sample_rate_hz"])
    spec = timing(tbc)
    per_ire = float(tbc["units_per_ire"])
    blanking = float(tbc["blanking"])
    rows = np.asarray(rows if rows is not None
                      else sync_rows(height, tbc["system"]))
    n_porch = int(round(spec["porch"]))
    n_sync = int(round(spec["sync"]))
    n_edge = int(max(1, round(spec["edge"])))

    # a first look at the pooled row: where the edge is, how deep, how noisy
    joined = np.concatenate([fields[:, rows - 1, :], fields[:, rows, :]],
                            axis=-1).astype(np.float64)
    joined = (joined - blanking) / per_ire
    mean_row = joined.reshape(-1, 2 * width).mean(axis=0)
    if which == "fall":
        expected = width
    elif which == "rise":
        expected = width + n_sync
    else:
        raise ValueError("which must be 'fall' or 'rise'")
    # the pooled edge: coarse crossing near the expected position
    # the settled level behind the edge: the sync tip's middle for the fall;
    # for the rise the back porch THROUGH the burst, whose median is the
    # porch level because a burst is symmetric about it
    behind_window = ((expected + n_edge + n_porch, expected + n_sync - n_porch)
                     if which == "fall"
                     else (expected + n_edge + 2, expected + 2 * n_porch))
    coarse = crossing(mean_row, (expected - n_porch, expected - n_edge),
                      behind_window, (expected - n_porch, expected + n_porch))
    if not coarse["found"]:
        return {"found": False, "why": "no pooled edge near the expected place"}
    c0 = int(round(coarse["crossing"]))
    depth = float(coarse["depth"])
    # per-line noise from the settled level's first difference
    settled = joined[..., c0 + n_edge + n_porch:c0 + n_sync - n_porch] \
        if which == "fall" else joined[..., c0 - n_porch:c0 - n_edge]
    noise = float(np.median(np.std(np.diff(settled, axis=-1), axis=-1)
                            / np.sqrt(2.0)))
    lines_total = int(joined.shape[0] * joined.shape[1])
    pooled_noise = noise / np.sqrt(max(lines_total / 2.0, 1.0))
    support = max(support_samples(kernel, depth, pooled_noise), n_edge)

    # the measured transition width, 10 to 90 per cent of the pooled edge
    # on the fine grid: the window is trimmed by it at both ends, because
    # the porch's own two transitions are the drive, not the porch
    transition = transition_width(mean_row, coarse["crossing"],
                                  coarse["pre"], coarse["post"], n_porch)
    trim = int(np.ceil(transition))

    # ahead of the edge: the porch and the kernel's support for the fall;
    # the whole sync pulse and the support for the rise, so the transition
    # that OPENS the window is inside the record in both cases. Behind it,
    # to just short of the next transition for the fall; the back porch,
    # burst included, for the rise.
    pre = (n_porch + support) if which == "fall" else (n_sync + support)
    post = (n_sync - 2 * n_edge) if which == "fall" else (n_porch + support)
    start = c0 - pre
    length = pre + post

    # THE TRANSITION THAT OPENS THE WINDOW: the blanking edge at the porch's
    # start (a sync width back for the rise). Found on the pooled row when
    # the levels either side differ by more than a line's noise; otherwise
    # taken at the specification's position, which is all a flat field can
    # offer.
    open_expected = float(coarse["crossing"]) - (spec["porch"] if which == "fall"
                                                 else spec["sync"])
    o = int(round(open_expected))
    if which == "fall":
        open_levels = ((o - support, o - trim), (o + trim, c0 - n_edge))
    else:
        open_levels = ((o - n_porch, o - trim), (o + trim + n_porch, c0 - n_porch))
    opening = crossing(mean_row, open_levels[0], open_levels[1],
                       (o - n_porch // 2, o + n_porch // 2))
    opening_found = bool(opening["found"]
                         and abs(float(opening.get("depth", 0.0))) > noise)
    opening_at = float(opening["crossing"]) if opening_found else open_expected
    records, meta = [], []
    for fi in range(fields.shape[0]):
        bad = set()
        for line, x0, x1 in tbc["dropouts"][fi]:
            bad.add(int(line))
        for row in rows:
            if row in bad or (row - 1) in bad:
                continue
            seq = np.concatenate([fields[fi, row - 1], fields[fi, row]]
                                 ).astype(np.float64)
            seq = (seq - blanking) / per_ire
            found = crossing(seq, (c0 - n_porch, c0 - n_edge),
                             (c0 + n_edge + n_porch, c0 + n_sync - n_porch)
                             if which == "fall" else
                             (c0 + n_edge + 2, c0 + 2 * n_porch),
                             (c0 - n_porch // 2, c0 + n_porch // 2))
            if not found["found"]:
                continue
            records.append(seq[start:start + length])
            meta.append((fi, int(row), float(found["crossing"]) - start,
                         float(found["depth"]), float(found["pre"])))
    if not records:
        return {"found": False, "why": "no line yielded a crossing"}
    records = np.asarray(records)
    meta = np.asarray(meta, dtype=np.float64)
    crossings, depths = meta[:, 2], meta[:, 3]
    reference = int(round(float(np.median(crossings))))
    shifts = reference - crossings
    depth_sigma = float(np.std(depths))
    keep = (np.abs(shifts) <= 1.0) \
        & (np.abs(depths - np.median(depths)) <= 4.0 * max(depth_sigma, 1e-12))
    aligned = np.array([shift_band_limited(r, s)
                        for r, s in zip(records[keep], shifts[keep])])
    # the drive boundaries: the closing edge's 10 per cent point (half the
    # 10-90 width ahead of its crossing) and the opening transition's 90 per
    # cent point; the window is what lies strictly between them
    onset = int(np.floor(reference - 0.5 * transition))
    opening_index = opening_at - start + (reference - float(np.median(crossings)))
    cutoff = int(np.ceil(opening_index + 0.5 * transition))
    return {
        "found": True,
        "records": aligned,
        "field": meta[keep, 0].astype(int),
        "row": meta[keep, 1].astype(int),
        "head": np.array([tbc["first_field"][int(f)] for f in meta[keep, 0]]),
        "crossing_index": reference,
        "crossing_scatter": float(np.std(crossings[keep])),
        "shifts": shifts[keep],
        "depth": float(np.median(depths[keep])),
        "depth_scatter": depth_sigma,
        "line_noise": noise,
        "support": int(support),
        "pre": pre,
        "post": post,
        "transition": float(transition),
        "onset": onset,
        "cutoff": cutoff,
        "opening_index": float(opening_index),
        "opening_found": opening_found,
        "porch_window": (cutoff + 1, onset),
        "tip_window": (reference + n_edge + n_porch, reference + post - n_porch)
        if which == "fall" else (reference + n_edge + n_porch // 2,
                                 reference + post),
        "kept": int(keep.sum()),
        "dropped": int((~keep).sum()),
        "sample_rate_hz": rate,
        "spec": spec,
        "which": which,
        "why": ("each line's crossing is found on the sinc-interpolated "
                "record and the line shifted so every crossing sits on one "
                "integer index; the window ahead of the edge is the "
                "specified porch, a full kernel support from the record's end"),
    }


def measured_edge(records: np.ndarray) -> Dict[str, object]:
    """The mean of aligned records with the standard error of each sample -
    the average that is the drive, and the floor it is known to."""
    r = np.asarray(records, dtype=np.float64)
    n = r.shape[0]
    mean = r.mean(axis=0)
    se = r.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(
        r.shape[1], np.inf)
    return {"edge": mean, "se": se, "lines": int(n)}


# --------------------------------------------------------------------------
# The residual ahead of the edge, before and after, against the floor
# --------------------------------------------------------------------------


def _fit_baseline(values, se, window: Tuple[int, int], slope: bool):
    lo, hi = window
    y = np.asarray(values, dtype=np.float64)[lo:hi]
    w = 1.0 / np.maximum(np.asarray(se, dtype=np.float64)[lo:hi], 1e-12)
    t = np.arange(hi - lo, dtype=np.float64)
    columns = [np.ones_like(t)] + ([t - t.mean()] if slope else [])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
    return y - design @ coefficients


def residual_check(test: Dict[str, object], window: Tuple[int, int],
                   predictions: Dict[str, np.ndarray],
                   slope: bool = False) -> Dict[str, object]:
    """THE FRONT-PORCH RESIDUAL BEFORE AND AFTER, AGAINST THE NOISE FLOOR AND
    AGAINST EACH OTHER, on a held-out mean.

    `test` is `measured_edge` of the held-out lines; `predictions` maps a
    name to a precursor built WITHOUT those lines. In the window each
    residual has its level fitted (and its slope, if asked - the porch
    relaxes from the preceding line, which is the channel's and not this
    term's) and nothing else: the precursor enters with NO free amplitude.
    Reported per prediction: the rms after the fit, its ratio to the floor
    (the rms standard error of the held-out mean over the window), the
    largest per-sample departure in sigmas, and the free gain the residual
    would have preferred - one within its error is the model confirmed with
    nothing left to tune, anything else says by how much it is wrong.
    """
    lo, hi = window
    edge = np.asarray(test["edge"], dtype=np.float64)
    se = np.asarray(test["se"], dtype=np.float64)
    floor = float(np.sqrt(np.mean(se[lo:hi] ** 2)))
    out: Dict[str, object] = {"window": (int(lo), int(hi)), "floor": floor,
                              "slope_fitted": bool(slope)}
    before = _fit_baseline(edge, se, window, slope)
    out["before"] = {"rms": float(np.sqrt(np.mean(before ** 2))),
                     "over_floor": float(np.sqrt(np.mean(before ** 2)) / floor)
                     if floor > 0 else np.inf,
                     "max_sigma": float(np.max(np.abs(before / se[lo:hi])))}
    for name, p in predictions.items():
        p = np.asarray(p, dtype=np.float64)
        after = _fit_baseline(edge - p, se, window, slope)
        rms = float(np.sqrt(np.mean(after ** 2)))
        # the free gain: regress the fitted 'before' residual on the fitted
        # precursor shape, weighted by the held-out errors
        shape = _fit_baseline(p, se, window, slope)
        w = 1.0 / np.maximum(se[lo:hi], 1e-12) ** 2
        denominator = float(np.sum(w * shape ** 2))
        gain = float(np.sum(w * shape * before) / denominator) \
            if denominator > 0 else np.nan
        gain_se = float(1.0 / np.sqrt(denominator)) if denominator > 0 else np.nan
        out[name] = {
            "rms": rms,
            "over_floor": rms / floor if floor > 0 else np.inf,
            "max_sigma": float(np.max(np.abs(after / se[lo:hi]))),
            "improvement": float(1.0 - rms / out["before"]["rms"])
            if out["before"]["rms"] > 0 else 0.0,
            "free_gain": gain,
            "free_gain_se": gain_se,
            "predicted_rms": float(np.sqrt(np.mean(shape ** 2))),
        }
    return out


# --------------------------------------------------------------------------
# The check against the hypercomplex readings
# --------------------------------------------------------------------------


def minimum_phase_counterpart(kernel: Dict[str, object]) -> Dict[str, object]:
    """THE CAUSAL FILTER WITH THE SAME MAGNITUDE, by the cepstral Bode
    relation (`hypercomplex.minimum_phase`), and what it says about the
    precursor.

    The zero-phase low-pass has phase zero; a causal filter with its
    magnitude must have the minimum phase, so the zero-phase filter carries
    an excess of minus that - anticipation, which is where the precursor
    comes from. The counterpart's kernel should have no anticausal part
    beyond the floor the log magnitude was held at, and its equivalent
    delay and all-pass residue are reported in the units `tesseract` used,
    for comparison with those readings: the term is common to every vertex
    and so cancels from every contrast, which is why it is not among them.
    """
    from vhsdecode.models import hypercomplex

    response = np.asarray(kernel["response"], dtype=np.float64)
    peak = float(response.max())
    floored = np.maximum(response, RESPONSE_FLOOR * peak)
    phase = hypercomplex.minimum_phase(np.log(floored))
    counterpart = floored * np.exp(1j * phase)
    k = np.fft.irfft(counterpart, n=int(kernel["length"]))
    shifted = np.fft.fftshift(k)
    centre = int(kernel["centre"])
    energy = float(np.sum(shifted ** 2))
    anticausal = float(np.sum(shifted[:centre] ** 2) / energy) if energy else 0.0
    grid = np.asarray(kernel["frequency_hz"], dtype=np.float64)
    passband = response > 0.5 * peak
    f = grid[passband]
    excess = -phase[passband]
    centred = f - f.mean()
    slope = float(centred @ excess / max(centred @ centred, 1e-300))
    residue = excess - slope * centred
    return {
        "response": counterpart,
        "kernel": shifted,
        "anticausal_energy_fraction": anticausal,
        "excess_phase_rms": float(np.sqrt(np.mean(excess ** 2))),
        "delay_s": -slope / (2.0 * np.pi),
        "allpass_rms": float(np.sqrt(np.mean(residue ** 2))),
        "passband_to_hz": float(f.max()),
        "why": ("the zero-phase filter's excess phase is minus its own "
                "minimum phase - anticipation - and is common to every "
                "vertex, so it is absent from every contrast the readings "
                "were made on"),
    }


def signatures(frequency_hz, kernel: Dict[str, object]) -> Dict[str, np.ndarray]:
    """The precursor's entry for the key: the anticausal half's transfer,
    `G_a(f) = sum_{tau<0} g[tau] exp(-2 pi i f tau / fs)`, the part of the
    filter that acts ahead of its input."""
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    lags = np.arange(-int(kernel["centre"]), 0, dtype=np.float64)
    g = np.asarray(kernel["anticausal"], dtype=np.float64)
    # only lags that carry anything, for the sum's size
    live = np.abs(g) > np.finfo(np.float64).eps * np.abs(g).max()
    phase = np.exp(-2j * np.pi * np.outer(grid, lags[live])
                   / float(kernel["sample_rate_hz"]))
    return {"decoder video low-pass precursor": phase @ g[live]}


# --------------------------------------------------------------------------
# The whole thing on a decode
# --------------------------------------------------------------------------


def analyse(stem: str, lowpass: Optional[Dict[str, object]] = None,
            which: str = "fall", slope: bool = False,
            tape_speed: str = "sp", inputfreq: float = 40
            ) -> Dict[str, object]:
    """PER HEAD, ON HELD-OUT LINES: the edge measured on one half of a head's
    lines (alternate rows), the precursor predicted from it three ways -
    the measured edge, the measured edge with the low-pass divided out, and
    the standard's ideal edge - and each subtracted from the other half's
    mean. Both folds are run and reported.
    """
    tbc = load_tbc(stem)
    if lowpass is None:
        lowpass = decoder_video_lowpass(inputfreq=inputfreq, system=tbc["system"],
                                        tape_speed=tape_speed)
    kernel = filter_kernel(lowpass, tbc["sample_rate_hz"])
    kernel["decoder_params"] = lowpass["decoder_params"]
    cut = edge_records(tbc, kernel, which=which)
    if not cut["found"]:
        return {"found": False, "why": cut["why"], "source": stem}
    reference = int(cut["crossing_index"])
    window = cut["porch_window"]
    spec = cut["spec"]
    out: Dict[str, object] = {
        "found": True, "source": stem, "which": which, "kernel": kernel,
        "crossing_index": reference, "window": window,
        "crossing_scatter": cut["crossing_scatter"], "depth": cut["depth"],
        "line_noise": cut["line_noise"], "support": cut["support"],
        "transition": cut["transition"], "onset": cut["onset"],
        "cutoff": cut["cutoff"], "opening_found": cut["opening_found"],
        "opening_index": cut["opening_index"],
        "kept": cut["kept"], "dropped": cut["dropped"], "heads": {},
    }
    for label, parity in (("A", True), ("B", False)):
        pick = cut["head"] == parity
        if pick.sum() < 4:
            continue
        rows = cut["row"][pick]
        records = cut["records"][pick]
        folds = {}
        for name, train_parity in (("even->odd", 0), ("odd->even", 1)):
            train = records[rows % 2 == train_parity]
            test = records[rows % 2 != train_parity]
            if train.shape[0] < 2 or test.shape[0] < 2:
                continue
            drive = measured_edge(train)
            held = measured_edge(test)
            noise = float(np.sqrt(np.mean(drive["se"] ** 2)))
            restored = deconvolved_drive(drive["edge"], kernel, noise)
            level = float(np.median(drive["edge"][window[0]:window[1]]))
            tip = cut["tip_window"]
            depth = float(np.median(drive["edge"][tip[0]:tip[1]]) - level)
            ideal_spec = specified_edge(records.shape[1], reference, depth,
                                        cut["sample_rate_hz"], spec["edge_us"],
                                        level)
            ideal_step = specified_edge(records.shape[1], reference, depth,
                                        cut["sample_rate_hz"], 0.0, level)
            onset, cutoff = cut["onset"], cut["cutoff"]
            measured = precursor(drive["edge"], kernel, onset)
            behind = postcursor(drive["edge"], kernel, cutoff)
            predictions = {
                "measured": measured,
                "deconvolved": precursor(restored["drive"], kernel, onset),
                "ideal_spec": precursor(ideal_spec, kernel, onset),
                "ideal_step": precursor(ideal_step, kernel, onset),
                "measured+post": measured + behind,
                "post_only": behind,
            }
            check = residual_check(held, window, predictions, slope=slope)
            reach = support_samples(kernel, depth, check["floor"])
            # the NEAR window: the part of the porch where the closing edge's
            # own precursor stands above the held-out floor
            near_window = (max(window[0], onset - reach), onset)
            check["near"] = (residual_check(held, near_window, predictions,
                                            slope=slope)
                             if near_window[1] - near_window[0] >= 3 else None)
            check.update({
                "train_lines": drive["lines"], "test_lines": held["lines"],
                "restored_to_hz": restored["restored_to_hz"],
                "depth": depth,
                "support_above_floor": reach,
                "held_out": held, "drive": drive, "predictions": predictions,
            })
            folds[name] = check
        out["heads"][label] = {"lines": int(pick.sum()), "folds": folds}
    return out


def report(result: Dict[str, object]) -> str:
    """The run as a table, in IRE and in multiples of the floor."""
    if not result.get("found"):
        return f"{result.get('source')}: {result.get('why')}"
    lines = [f"{result['source']}  edge={result['which']}  crossing at index "
             f"{result['crossing_index']} (scatter {result['crossing_scatter']:.3f} "
             f"samples)  depth {result['depth']:.2f} IRE  line noise "
             f"{result['line_noise']:.3f} IRE  support {result['support']} samples  "
             f"transition {result['transition']:.2f} samples  opening at "
             f"{result['opening_index']:.2f} ({'found' if result['opening_found'] else 'spec'})"
             f"  cutoff {result['cutoff']} onset {result['onset']}  window "
             f"{result['window']}  kept {result['kept']} dropped {result['dropped']}"]
    names = ("measured", "deconvolved", "ideal_spec", "ideal_step",
             "measured+post", "post_only")

    def block(check, label):
        b = check["before"]
        rows = [f"    {label} {check['window']}  floor {check['floor']:.4f} IRE"]
        rows.append(f"      before        rms {b['rms']:.4f}  x{b['over_floor']:.1f} "
                    f"floor  max {b['max_sigma']:.1f} sigma")
        for name in names:
            c = check[name]
            rows.append(f"      {name:13s} rms {c['rms']:.4f}  x{c['over_floor']:.1f} "
                        f"floor  max {c['max_sigma']:.1f} sigma  improvement "
                        f"{100*c['improvement']:+.1f}%  free gain "
                        f"{c['free_gain']:.3f} +/- {c['free_gain_se']:.3f}")
        return rows

    for head, entry in result["heads"].items():
        for fold, check in entry["folds"].items():
            lines.append(f"  head {head} {fold}: train {check['train_lines']} "
                         f"test {check['test_lines']}  restored to "
                         f"{check['restored_to_hz']/1e6:.2f} MHz  precursor above "
                         f"floor to {check['support_above_floor']} samples")
            lines += block(check, "window")
            if check.get("near") and tuple(check["near"]["window"]) != tuple(check["window"]):
                lines += block(check["near"], "near  ")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stems", nargs="+")
    parser.add_argument("--which", default="fall", choices=("fall", "rise"))
    parser.add_argument("--slope", action="store_true")
    parser.add_argument("--tape_speed", default="sp")
    parser.add_argument("--inputfreq", type=float, default=40)
    args = parser.parse_args()
    shared = decoder_video_lowpass(inputfreq=args.inputfreq,
                                   tape_speed=args.tape_speed)
    for stem in args.stems:
        print(report(analyse(stem, shared, which=args.which, slope=args.slope)))
