"""Complex RF channel equalizer from an offline-identified response.

The runtime half of the multidimensional information extrapolation: the
converged anti-residual set - one complex RF response per capture, identified
offline by the loop in tools/ringing_measure/ - is applied at the RF site,
immediately before the FM demodulator, where the whole chain before the
demodulator is linear and one filter fixes all levels at once.

The site is the one the luma path equalizer uses: `indata_fft` in
`demodblock`, AFTER the envelope has been taken from the uncorrected signal
(the envelope is a measurement of the tape - the color-under correction and
dropout detection read it - so a correction to the luma must not reach it;
folding one in with `RFVideo` instead was measured to cost 17% of the chroma
correction), and the analytic signal is rebuilt after the multiply.

Laws:
  - built ONCE at decoder init from the declared file and never mutated -
    the demodulation thread runs ahead of field assembly, so a filter that
    changes mid-decode makes decodes non-reproducible;
  - byte-inert when absent: with no table the site's operations are
    textually today's; a UNITY table is exact too ((a+bj)*(1+0j) is exact);
  - unity outside the file's declared band and at DC;
  - delay-normalized at the blanking carrier: the sync edges must not move,
    or the linelocs re-land and every A/B compares different grids;
  - conjugate-symmetric over the full block grid, real at DC and Nyquist,
    so the debug plots' real inverse transforms stay real;
  - this module never imports the offline gauges (they carry matplotlib
    and format geometry at import time).

Export contract read here (the rest of the file is the offline tool's
record): `freqs_hz` (RF axis, positive, ascending), `log_H` (complex:
ln|H| + j phase, the channel as identified - the table applies its inverse),
`belief` (0..1 per bin), scalars `site` (must name the pre-demod RF site),
`freq_hz`, and, when present, `system`, `tape_format`, `tape_speed`.
"""

import logging
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

import lddecode.core as ldd


REQUIRED_KEYS = ("freqs_hz", "log_H", "belief")
SITE_TOKEN = "rf_pre_demod"

# The most this stage will ever apply, per bin, in nepers of magnitude:
# 1.0 is a gain of 2.7, five times the largest channel tilt measured on
# any of these tapes (0.19 nepers across the whole deviation band). A
# table asking for more than this is not describing a playback channel,
# and applying it verbatim can destroy the RF so completely that the
# decoder finds no fields at all - which is how this bound was found. The
# excess is clamped, with the bins named, rather than refused: a
# correction that is merely too strong should still decode.
MAXIMUM_LOG_MAGNITUDE = 1.0


def _log():
    """The decoder's logger while a decode runs, the module's own otherwise."""
    return ldd.logger if getattr(ldd, "logger", None) is not None \
        else logging.getLogger(__name__)


def _scalar(z, key, default=None):
    if key not in z.files:
        return default
    value = z[key]
    return value.item() if np.ndim(value) == 0 else value


def _check_declarations(rf, z, path):
    site = str(_scalar(z, "site", "") or "")
    if SITE_TOKEN not in site.lower():
        raise ValueError(f"{path}: site {site!r} is not the pre-demodulator "
                         f"RF site this stage applies at")
    for key, attribute in (("system", "system"),
                           ("tape_format", "tape_format"),
                           ("tape_speed", "tape_speed")):
        declared = _scalar(z, key)
        actual = getattr(rf, attribute, None)
        if declared is not None and actual is not None \
                and str(declared).lower() != str(actual).lower():
            raise ValueError(f"{path}: declared {key}={declared!r} but the "
                             f"decoder is {actual!r}")
    declared_rate = _scalar(z, "freq_hz")
    if declared_rate is not None and abs(float(declared_rate) - rf.freq_hz) \
            > 1.0:
        _log().warning("channel_eq: %s was identified at %.3f MSps, decoding "
                       "at %.3f MSps - interpolating on the Hz axis",
                       path, float(declared_rate) / 1e6, rf.freq_hz / 1e6)


def _onto_half_grid(freqs_hz, values, grid_hz, fill):
    """Shape-preserving interpolation onto the block's positive half-grid,
    `fill` outside the declared span. A bin the export left non-finite is
    dropped rather than carried into the interpolator, which refuses one
    (and would otherwise abort the decode over a single bad bin)."""
    order = np.argsort(freqs_hz)
    x = np.asarray(freqs_hz, dtype=np.float64)[order]
    y = np.asarray(values)[order]
    finite = np.isfinite(x) & np.all(np.isfinite(np.atleast_2d(
        [y.real, y.imag] if np.iscomplexobj(y) else [y])), axis=0)
    if not finite.all():
        _log().warning("channel_eq: %d of %d bins are non-finite and are "
                       "dropped from the interpolation",
                       int((~finite).sum()), len(finite))
        x, y = x[finite], y[finite]
    out = np.full(len(grid_hz), fill, dtype=y.dtype)
    inside = (grid_hz >= x[0]) & (grid_hz <= x[-1])
    if inside.any() and len(x) >= 2:
        if np.iscomplexobj(y):
            out[inside] = (PchipInterpolator(x, y.real)(grid_hz[inside])
                           + 1j * PchipInterpolator(x, y.imag)(grid_hz[inside]))
        else:
            out[inside] = PchipInterpolator(x, y)(grid_hz[inside])
    return out


def _delay_normalize(table, grid_hz, carrier_hz):
    """Remove the linear phase so the group delay is zero at the carrier."""
    phase = np.unwrap(np.angle(table))
    step = grid_hz[1] - grid_hz[0]
    centre = int(np.clip(round(carrier_hz / step), 2, len(grid_hz) - 3))
    window = slice(centre - 2, centre + 3)
    slope = float(np.polyfit(grid_hz[window], phase[window], 1)[0])  # rad/Hz
    return table * np.exp(-1j * slope * grid_hz)


def _full_table(half):
    """The conjugate-symmetric full-length table from the positive half
    (length n//2 + 1): real at DC and at Nyquist."""
    bins = len(half)
    n = 2 * (bins - 1)
    full = np.empty(n, dtype=np.complex128)
    full[:bins] = half
    full[bins:] = np.conj(half[1:bins - 1][::-1])
    full[0] = full[0].real
    full[bins - 1] = full[bins - 1].real
    return full


def _audit(rf, grid_hz, table, belief):
    edges = [0.0, 0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, grid_hz[-1] + 1]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (grid_hz >= lo) & (grid_hz < hi) & (belief > 0)
        if not band.any():
            continue
        rows.append("%.1f-%.1f MHz: belief %.2f |E| %+.2f dB phase %+.1f deg"
                    % (lo / 1e6, hi / 1e6, float(np.mean(belief[band])),
                       float(20 * np.log10(np.mean(np.abs(table[band])))),
                       float(np.degrees(np.mean(np.angle(table[band]))))))
    full = _full_table(table)
    impulse = np.fft.ifft(full - 1.0).real
    total = float(np.sum(impulse ** 2)) or 1.0
    cut = int(getattr(rf, "blockcut", 0))
    cut_end = int(getattr(rf, "blockcut_end", 0))
    beyond = float(np.sum(impulse[cut:len(impulse) // 2] ** 2)) / total \
        if cut else 0.0
    wrap = float(np.sum(impulse[len(impulse) - cut_end:] ** 2)) / total \
        if cut_end else 0.0
    _log().info("channel_eq: %s; kernel energy beyond the front cut %.1e, in "
                "the end cut %.1e", "; ".join(rows) if rows else "unity",
                beyond, wrap)


def load(rf, path, amount):
    """Build `rf.Filters["ChannelEQ"]` from the export at `path`; once."""
    z = np.load(path, allow_pickle=False)
    for key in REQUIRED_KEYS:
        if key not in z.files:
            raise KeyError(f"{path}: missing {key!r}; keys are "
                           f"{sorted(z.files)}")
    _check_declarations(rf, z, path)
    n = int(rf.blocklen)
    grid_hz = np.arange(n // 2 + 1) * (rf.freq_hz / n)
    log_h = _onto_half_grid(z["freqs_hz"], np.asarray(z["log_H"],
                                                        dtype=np.complex128),
                            grid_hz, fill=0.0 + 0.0j)
    belief = np.clip(_onto_half_grid(z["freqs_hz"],
                                     np.asarray(z["belief"],
                                                dtype=np.float64),
                                     grid_hz, fill=0.0), 0.0, 1.0)
    belief[0] = 0.0
    # the inverse of the identified channel, as far as it is believed
    applied = float(amount) * belief * log_h
    if not np.all(np.isfinite(applied)):
        count = int((~np.isfinite(applied)).sum())
        _log().warning("channel_eq: %s has %d non-finite bins; they are "
                       "left uncorrected", path, count)
        applied = np.where(np.isfinite(applied), applied, 0.0)
    excess = np.abs(applied.real) > MAXIMUM_LOG_MAGNITUDE
    if excess.any():
        worst = float(np.max(np.abs(applied.real)))
        _log().warning("channel_eq: %s asks for up to %.1f dB at %d of %d "
                       "bins (%.2f-%.2f MHz); clamped to %.1f dB - a "
                       "playback channel does not have that much structure",
                       path,
                       20 * worst / math.log(10), int(excess.sum()),
                       len(applied), float(grid_hz[excess].min()) / 1e6,
                       float(grid_hz[excess].max()) / 1e6,
                       20 * MAXIMUM_LOG_MAGNITUDE / math.log(10))
        applied = (np.clip(applied.real, -MAXIMUM_LOG_MAGNITUDE,
                           MAXIMUM_LOG_MAGNITUDE) + 1j * applied.imag)
    half = np.exp(-applied)
    half[0] = 1.0 + 0.0j
    carrier_hz = float(rf.iretohz(0, spec=True)) \
        if hasattr(rf, "iretohz") else float(rf.SysParams["ire0"])
    half = _delay_normalize(half, grid_hz, carrier_hz)
    half[0] = 1.0 + 0.0j
    table = _full_table(half)
    rf.Filters["ChannelEQ"] = table
    rf.__dict__["_channel_eq"] = {"path": path, "amount": float(amount),
                                  "blocklen": n}
    # the ringing stage mutes its RF-derived synthesized kernels while this
    # stage carries that physics
    rf.__dict__.setdefault("_ringing_state", {})["channel_eq_active"] = True
    options = getattr(rf, "options", None)
    if options is not None and getattr(options, "luma_eq", 0) != 0:
        # The identified response carries the path magnitude the luma
        # equalizer would otherwise fold in; at the RF site this table
        # takes the equalizer's place. The equalizer's MEASUREMENT keeps
        # running - it is chain-independent and the offline loop reads it.
        _log().info("channel_eq: --luma_eq %s is superseded at the RF site "
                    "while the identified channel response is applied",
                    options.luma_eq)
    _audit(rf, grid_hz, half, belief)
    return table


def is_active(rf):
    return "ChannelEQ" in getattr(rf, "Filters", {})


def effective_rf_response(rf):
    """The RF response the DEMODULATOR actually sees, on the block's full
    FFT grid: `RFVideo` times this stage's table where the stage is active,
    `RFVideo` alone otherwise.

    Anything that models the pre-demodulator path from the stored filter -
    the luma-beat stage reads the sideband imbalance of `RFVideo` across the
    color-under band - must read this instead of `Filters["RFVideo"]`, or it
    models a path that no longer exists once the table is applied. The
    ENVELOPE, by contrast, is still taken from `RFVideo` alone: the table is
    applied after it, deliberately (see the module docstring)."""
    rf_video = rf.Filters["RFVideo"]
    table = rf.Filters.get("ChannelEQ")
    return rf_video if table is None else rf_video * table
