"""Residual channels, downscaled to 4fsc on the final time base.

Each residual of the multidimensional information extrapolation is a
CHANNEL: computed where its stage runs, at the capture rate, it rides the
video dict to the field like the picture and is resampled to 4fsc on the
FINAL time base correction - the corrected linelocs, after the carrier
refinement - so that every residual sits on the output grid the offline
gauges read, time-aligned with the picture, with the residual RF phase
removed by the same sync-distance resampling that removes it from the
picture. "Keep the existing residuals, but downscaled to 4fsc; each channel
is a residual that we downscale based on the final TBC."

The channels (`information_extrapolation.RESIDUAL_CHANNELS`):

  amplitude         the carrier envelope, resampled on the TIMING ALONE. The
                    downscale's wow level adjust corrects demodulated
                    FREQUENCY (speed scales the carrier frequency); an
                    amplitude has no such term, and multiplying it in would
                    write the drum-rate speed wobble - the very witness the
                    time-base node reads from the amplitude - back into the
                    channel.
  frequency         what the RF equalizer changed in the demodulated luma:
                    the demodulated field minus the unequalized one, both
                    through the full downscale (both are frequency; the
                    level adjust belongs to both alike).
  time              the final time base's own per-pixel speed deviation, the
                    wow factor minus one over the output span - and per
                    line, the line period's deviation from nominal.
  chroma_amplitude  the color-under gain the chroma stage applied - the
                    chroma channel's identified component - timing alone.
  head_switch,      what those stages changed, in IRE, where they run.
  baseband_eq

One npz per field in the directory named by --residual_channels; nothing
here runs without it.
"""

import os

import numpy as np

import lddecode.core as ldd
from lddecode.utils import scale_field
from vhsdecode import carrier_tbc


SITE = "4fsc on the final time base correction"
DECODER_FILTERS = "decoder_filters.npz"


def _half(table):
    """The positive half of a full-length FFT-grid table."""
    table = np.asarray(table)
    return table[: len(table) // 2 + 1]


def export_decoder_filters(rf, directory):
    """The decoder's own filters, written once beside the channels: known
    exactly, so the offline identification divides them out analytically
    and never estimates them. RF-side tables on the block's positive
    half-grid, video-side on the rfft grid, the downscale's interpolation
    taps, and the levels the picture is stated in."""
    path = os.path.join(directory, DECODER_FILTERS)
    if os.path.exists(path):
        return path
    n = int(rf.blocklen)
    filters = rf.Filters
    declared = {
        "freq_hz": float(rf.freq_hz),
        "blocklen": n,
        "rf_grid_hz": np.arange(n // 2 + 1) * (float(rf.freq_hz) / n),
        "rf_video": _half(filters["RFVideo"]).astype(np.complex128),
        "system": str(getattr(rf, "system", "")),
        "tape_format": str(getattr(rf, "tape_format", "")),
        "tape_speed": str(getattr(rf, "tape_speed", "")),
        "ire0": float(rf.DecoderParams["ire0"]),
        "hz_ire": float(rf.DecoderParams["hz_ire"]),
        "vsync_ire": float(rf.DecoderParams["vsync_ire"]),
        "line_period_us": float(rf.SysParams["line_period"]),
        "fsc_mhz": float(rf.SysParams.get("fsc_mhz", 0.0)),
        "color_burst_us": np.asarray(rf.SysParams.get("colorBurstUS", (0.0, 0.0)),
                                     dtype=np.float64),
        "color_under_carrier_hz": float(
            rf.DecoderParams.get("color_under_carrier", 0.0)),
        # the interpolator's zero-shift leg and its half-sample leg (the
        # least flat one, the divisor the baseband stage uses)
        "downscale_taps": np.asarray(rf.downscale_sinc_lut[0], dtype=np.float64),
        "downscale_taps_half": np.asarray(
            rf.downscale_sinc_lut[len(rf.downscale_sinc_lut) // 2], dtype=np.float64
        ),
    }
    for name, key in (("FVideo", "fvideo"), ("FDeemp", "fdeemp"),
                      ("ChannelEQ", "channel_eq")):
        if name in filters:
            table = np.asarray(filters[name])
            declared[key] = (_half(table) if len(table) == n else table).astype(
                np.complex128
            )
    np.savez(path, **declared)
    return path


def time_base_track(rf):
    """b5's per-head time-base node state, where the node has run: the
    first entry of the per-field dimension (head parity)."""
    state = getattr(carrier_tbc, "time_base_state", None)
    if state is None:
        return {}
    try:
        per_head = state(rf) or {}
    except Exception:
        return {}
    out = {}
    for head, values in per_head.items():
        tag = "head_first" if head else "head_second"
        for key, value in (values or {}).items():
            try:
                out["time_base_%s_%s" % (tag, key)] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
    return out


def _output_shape(field):
    return int(field.outlinecount), int(field.outlinelen)


def resample_timing_only(field, buf):
    """`buf` (capture rate, the field's video span) on the output grid of
    the final linelocs, with the wow level adjust held at unity."""
    locs, wow = field.computewow_scaled()
    lines, width = _output_shape(field)
    dsout = np.zeros(lines * width, dtype=np.float32)
    scale_field(
        np.ascontiguousarray(buf, dtype=np.float32),
        dsout,
        locs,
        np.ones_like(wow),
        field.rf.downscale_sinc_lut,
        field.lineoffset,
        width,
    )
    return dsout.reshape(lines, width)


def resample_full(field, channel):
    """A channel through the decoder's own downscale: timing and the wow
    level adjust, exactly as the picture."""
    out, _, _ = ldd.Field.downscale(field, channel=channel)
    lines, width = _output_shape(field)
    return np.asarray(out, dtype=np.float32).reshape(lines, width)


def time_base_residual(field):
    """The final time base's residual: per output pixel, the wow factor
    minus one (the local speed deviation the resample map carries); per
    line, the line period's deviation from nominal."""
    locs, wow = field.computewow_scaled()
    lines, width = _output_shape(field)
    start = width * (int(field.lineoffset) + 1)
    per_pixel = (wow[start:start + lines * width] - 1.0).astype(np.float32)
    per_pixel = per_pixel.reshape(lines, width)
    linelocs = np.asarray(field.linelocs, dtype=np.float64)
    per_line = np.diff(linelocs) / float(field.inlinelen) - 1.0
    return per_pixel, per_line


def _period_deviation(linelocs, inlinelen):
    return np.diff(np.asarray(linelocs, dtype=np.float64)) / float(inlinelen)


def complex_time_residual(field):
    """The time component as one complex residual per line, in fractions of
    the nominal line period: the hsync's measured period deviation as the
    real part, what the color burst's phase lock added as the imaginary
    part - the two timing pilots - and, separately, what the carrier
    refinement added on top where it ran. Empty where the field does not
    keep its intermediate line locations."""
    hsync = getattr(field, "linelocs2", None)
    if hsync is None:
        return {}
    final = np.asarray(field.linelocs, dtype=np.float64)
    burst = getattr(field, "_linelocs_hsync", None)
    burst = final if burst is None else np.asarray(burst, dtype=np.float64)
    hsync = np.asarray(hsync, dtype=np.float64)
    if len(hsync) != len(final) or len(burst) != len(final):
        return {}
    inlinelen = field.inlinelen
    real = _period_deviation(hsync, inlinelen) - 1.0
    imaginary = _period_deviation(burst, inlinelen) - _period_deviation(
        hsync, inlinelen
    )
    out = {"time_per_line_complex": real + 1j * imaginary}
    if burst is not final:
        out["time_per_line_carrier"] = _period_deviation(
            final, inlinelen
        ) - _period_deviation(burst, inlinelen)
    return out


def chroma_framing_state(field):
    """The chroma framing this field carries, exported for the TIME BASE.

    The burst lock is part of the time base: it moves line positions to null
    the burst's phase. That makes the framing the lock's business and not
    the picture's - by the time a field reaches the output the lock has
    collapsed the standard's four-field sequence into the two-state
    alternation, and the absolute framing is no longer recoverable there.

    So what the lock itself saw is recorded here, per field, alongside the
    residual channels: the burst phase BEFORE the lock's own correction
    where the field kept it, and the field's parity. Downstream that is
    what a rotator's starting phase and a framing check can be built from,
    without either having to re-derive it from an output that no longer
    holds it."""
    out = {}
    hsync = getattr(field, "linelocs2", None)
    final = getattr(field, "linelocs", None)
    if hsync is None or final is None:
        return out
    hsync = np.asarray(hsync, dtype=np.float64)
    final = np.asarray(final, dtype=np.float64)
    if len(hsync) != len(final) or len(final) < 8:
        return out
    inlinelen = float(field.inlinelen)
    fsc = float(field.rf.SysParams.get("fsc_mhz", 0.0)) * 1e6
    if fsc <= 0 or inlinelen <= 0:
        return out
    line_seconds = inlinelen / float(field.rf.freq_hz)
    # what the lock MOVED, in cycles of subcarrier: the framing information
    # the output no longer carries
    moved = (final - hsync) / inlinelen * line_seconds * fsc
    out["chroma_lock_cycles"] = moved.astype(np.float64)
    out["chroma_lock_mean_cycles"] = np.array(float(np.nanmean(moved)))
    return out


def head_switch_location(field):
    """Where the head switch fell in this field, from the head-switch lane's
    own locator.

    Called by IMPORT, never reimplemented: `head_switch.locate` reads the
    luma amplitude stage's per-field deviation binned on the field's own
    line locations, which is the measurement that actually resolves the
    switch. Differencing picture lines does NOT work - on real content the
    line-to-line difference is the picture, and located points scatter over
    tens of microseconds.

    The switch is held by the PLAYBACK machine's servo, so its position is
    nearly fixed field to field and what little it moves is a clean
    differential for the time base. It is recorded per field so that drift
    can be measured rather than assumed."""
    try:
        from vhsdecode import head_switch
    except Exception:
        return {}
    try:
        regions = head_switch.locate(field)
    except Exception:
        return {}
    if not regions:
        return {}
    rows = np.array([[float(v) for v in region[:4]] for region in regions],
                    dtype=np.float64)
    return {"head_switch_regions": rows}


def channels(field):
    """Every residual channel this field can supply, on the output grid."""
    video = field.data["video"]
    out = {}
    if "envelope" in video:
        out["amplitude"] = resample_timing_only(field, video["envelope"])
    if "demod_noeq" in video:
        out["frequency"] = resample_full(field, "demod") - resample_full(
            field, "demod_noeq"
        )
    if "demod_burst" in video:
        # The COLOR-UNDER itself, on the final time base - the same
        # physical burst the up-converted chroma carries, before the
        # heterodyne. Comparing the two isolates the up-conversion's own
        # group delay and response, because they are the same event
        # measured at two carriers. Downscaled with the chroma path's own
        # shift so the measurement sees what the up-conversion sees.
        out["color_under"] = resample_full(field, "demod_burst")
    if "demod_raw" in video:
        # the demodulated carrier frequency before de-emphasis, in Hz: the
        # amplitude channel's abscissa - the envelope against it is the
        # path magnitude at the instantaneous carrier
        out["carrier"] = resample_full(field, "demod_raw")
    per_pixel, per_line = time_base_residual(field)
    out["time"] = per_pixel
    out["time_per_line"] = per_line.astype(np.float64)
    out.update(complex_time_residual(field))
    correction = getattr(field, "chroma_envelope_correction", None)
    if correction is not None:
        out["chroma_amplitude"] = resample_timing_only(field, correction)
    for name in ("head_switch", "baseband_eq"):
        if name in video:
            out[name] = resample_full(field, name)
    out.update(head_switch_location(field))
    out.update(chroma_framing_state(field))
    return out


def export_field(field):
    """Write this field's residual channels; returns the path, or None
    where no directory was named."""
    directory = getattr(field.rf, "_residual_channels_dir", None)
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    store = field.rf.__dict__
    count = int(store.get("_residual_channels_count", 0))
    store["_residual_channels_count"] = count + 1
    if count == 0 and hasattr(field.rf, "Filters"):
        export_decoder_filters(field.rf, directory)
    path = os.path.join(directory, "field_%06d.npz" % count)
    lines, width = _output_shape(field)
    np.savez(
        path,
        site=SITE,
        is_first_field=bool(field.isFirstField),
        readloc=int(getattr(field, "readloc", 0)),
        linelocs=np.asarray(field.linelocs, dtype=np.float64),
        freq_hz=float(field.rf.freq_hz),
        inlinelen=float(field.inlinelen),
        outlinelen=width,
        outlinecount=lines,
        lineoffset=int(field.lineoffset),
        **channels(field),
        **time_base_track(field.rf),
    )
    return path
