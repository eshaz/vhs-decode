#!/usr/bin/env python3
"""Part A - the channel identifier: the BASELINE the frequency axis is
differentiated against.

Reads a residual-channel directory written by `--residual_channels <dir>`
(one npz per field plus `decoder_filters.npz`) and identifies the playback
channel's MAGNITUDE at the RF site from the constant-envelope reference:
the luma carrier is recorded at constant amplitude, so the envelope against
the instantaneous carrier - the picture sweeps the carrier across the band
for free - is |H| at that carrier frequency, with the decoder's own RF
filter divided out analytically (a known filter is never estimated).
Measured per head, pooled in the log domain with the between-head
disagreement in the error budget (DerSimonian-Laird, the head-switch lane's
frozen `_pool`), with a belief per bin.

The envelope is taken BEFORE any equalizer in the decoder, so this
magnitude is chain-independent: it is the reference, not the loop's
convergence gauge. The phase half comes from the sync edges in the loop
(`multidimensional_information_extrapolation.py`); the baseline exports
phase zero with the magnitude's belief, and the loop's residuals refine
both from there.

Units: log_H is the natural log of the channel's magnitude relative to its
mean over the believed band (an overall gain is nothing to an FM limiter).

    channel_identify.py <residual_dir> <out.npz> [--bin-ire N]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from vhsdecode.baseband_eq import _pool  # noqa: E402  (frozen contract)

SITE = "rf_pre_demod_post_envelope"


def load_directory(directory):
    """The per-field exports and the decoder's filters."""
    filters_path = os.path.join(directory, "decoder_filters.npz")
    if not os.path.exists(filters_path):
        raise FileNotFoundError(f"{directory}: no decoder_filters.npz - was "
                                f"the decode run with --residual_channels?")
    filters = {k: v for k, v in np.load(filters_path, allow_pickle=False).items()}
    fields = sorted(glob.glob(os.path.join(directory, "field_*.npz")))
    if not fields:
        raise FileNotFoundError(f"{directory}: no field_*.npz")
    return filters, fields


def rf_video_log_magnitude(filters, freqs_hz):
    """ln|RFVideo| at the given frequencies, from the decoder's own table."""
    grid = np.asarray(filters["rf_grid_hz"], dtype=np.float64)
    magnitude = np.abs(np.asarray(filters["rf_video"], dtype=np.complex128))
    with np.errstate(divide="ignore"):
        log_mag = np.log(np.maximum(magnitude, 1e-12))
    return np.interp(freqs_hz, grid, log_mag)


def head_magnitude(field_files, head, bin_hz, filters):
    """Binned ln(envelope) against the carrier for one head: (centres,
    mean, variance-of-the-mean, population), decoder RF filter divided
    out. Only samples whose carrier moved less than one bin since the
    previous sample are believed: an amplitude taken while the carrier
    sweeps faster than that reports the path's transient, not the tape."""
    sums = {}
    for path in field_files:
        with np.load(path, allow_pickle=False) as z:
            if bool(z["is_first_field"]) != head:
                continue
            if "amplitude" not in z.files or "carrier" not in z.files:
                continue
            amplitude = np.asarray(z["amplitude"], dtype=np.float64)
            carrier = np.asarray(z["carrier"], dtype=np.float64)
        ok = np.isfinite(amplitude) & (amplitude > 0) & np.isfinite(carrier) \
            & (carrier > 0)
        slew = np.abs(np.diff(carrier, axis=1, prepend=carrier[:, :1]))
        ok &= slew < bin_hz
        if not ok.any():
            continue
        log_amplitude = np.log(amplitude[ok])
        bins = np.floor(carrier[ok] / bin_hz).astype(np.int64)
        for index in np.unique(bins):
            select = bins == index
            values = log_amplitude[select]
            entry = sums.setdefault(int(index), [0.0, 0.0, 0])
            entry[0] += float(values.sum())
            entry[1] += float((values ** 2).sum())
            entry[2] += int(values.size)
    if not sums:
        return None
    indices = np.array(sorted(sums), dtype=np.int64)
    centres = (indices + 0.5) * bin_hz
    total = np.array([sums[i][0] for i in indices])
    total2 = np.array([sums[i][1] for i in indices])
    population = np.array([sums[i][2] for i in indices], dtype=np.float64)
    mean = total / population
    variance = np.maximum(total2 / population - mean ** 2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance_of_mean = np.where(population > 1,
                                    variance / np.maximum(population - 1, 1),
                                    np.inf)
    mean = mean - rf_video_log_magnitude(filters, centres)
    return centres, mean, variance_of_mean, population


def identify(directory, bin_ire=1.0, minimum_population=64):
    filters, fields = load_directory(directory)
    hz_ire = float(filters["hz_ire"])
    bin_hz = float(bin_ire) * hz_ire
    per_head = {}
    for head, name in ((True, "head_first"), (False, "head_second")):
        measured = head_magnitude(fields, head, bin_hz, filters)
        if measured is not None:
            per_head[name] = measured
    if not per_head:
        raise RuntimeError(f"{directory}: no head supplied amplitude and "
                           f"carrier channels")
    # one grid for both heads
    low = min(m[0][0] for m in per_head.values())
    high = max(m[0][-1] for m in per_head.values())
    centres = np.arange(np.floor(low / bin_hz), np.floor(high / bin_hz) + 1) \
        * bin_hz + 0.5 * bin_hz
    L = np.full((len(per_head), len(centres)), np.nan + 0j, dtype=np.complex128)
    var = np.full((len(per_head), len(centres)), np.inf)
    population = np.zeros((len(per_head), len(centres)))
    names = sorted(per_head)
    for row, name in enumerate(names):
        c, mean, vom, pop = per_head[name]
        index = np.round((c - centres[0]) / bin_hz).astype(np.int64)
        good = (pop >= minimum_population) & np.isfinite(vom) & (vom > 0)
        L[row, index[good]] = mean[good]
        var[row, index[good]] = vom[good]
        population[row, index[good]] = pop[good]
    # relative to each head's own mean over its populated band (a gain is
    # nothing to the limiter; only the shape is the channel)
    for row in range(len(names)):
        ok = np.isfinite(L[row])
        if ok.any():
            weights = 1.0 / var[row, ok]
            L[row, ok] -= np.sum(weights * L[row, ok].real) / np.sum(weights)
    pooled, v_pool, tau2 = _pool(L, var)
    ok = np.isfinite(pooled) & np.isfinite(v_pool)
    error2 = np.where(ok, v_pool + tau2, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # the garrote: a bin whose level is within its own error is not
        # believed, and belief rises with the level's significance
        belief = np.where(ok, 1.0 - error2 / np.maximum(np.abs(pooled) ** 2,
                                                          1e-30), 0.0)
    belief = np.clip(np.nan_to_num(belief, nan=0.0), 0.0, 1.0)
    log_H = np.where(ok, pooled.real, 0.0) + 0.0j
    export = {
        "freqs_hz": centres,
        "log_H": log_H.astype(np.complex128),
        "belief": belief,
        "noise_psd": np.where(ok, error2, np.inf),
        "population": population.sum(axis=0),
        "site": SITE,
        "freq_hz": float(filters["freq_hz"]),
        "blocklen": int(filters["blocklen"]),
        "system": str(filters["system"]),
        "tape_format": str(filters["tape_format"]),
        "tape_speed": str(filters["tape_speed"]),
        "passes": 0,
        "exhaustion": "baseline: magnitude from the constant envelope; "
                      "phase zero (the loop's sync edges supply it)",
        "bin_hz": bin_hz,
        "rf_video_divided": True,
        "fields": len(fields),
        "heads": np.array(names),
        "tau2": tau2,
    }
    for row, name in enumerate(names):
        export[f"log_H_{name}"] = np.where(np.isfinite(L[row]), L[row], 0.0)
        export[f"se_{name}"] = np.sqrt(np.where(np.isfinite(var[row]), var[row],
                                                np.inf))
        export[f"population_{name}"] = population[row]
    return export


def summarize(export):
    ok = export["belief"] > 0
    f = export["freqs_hz"][ok] / 1e6
    g = export["log_H"][ok].real
    lines = [f"baseline over {f.min():.2f}-{f.max():.2f} MHz, "
             f"{int(ok.sum())} believed bins of {len(ok)}, "
             f"{export['fields']} fields, heads {list(export['heads'])}"]
    edges = np.linspace(f.min(), f.max(), 7)
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (f >= lo) & (f < hi)
        if band.any():
            lines.append("  %.2f-%.2f MHz: ln|H| %+.3f (%+.2f dB), belief %.2f"
                         % (lo, hi, g[band].mean(), 20 / np.log(10) * g[band].mean(),
                            export["belief"][ok][band].mean()))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("output")
    parser.add_argument("--bin-ire", type=float, default=1.0,
                        help="carrier bin width in IRE of deviation (default 1)")
    parser.add_argument("--minimum-population", type=int, default=64)
    args = parser.parse_args(argv)
    export = identify(args.directory, args.bin_ire, args.minimum_population)
    np.savez(args.output, **export)
    print(summarize(export))
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump({"fields": export["fields"], "bin_hz": export["bin_hz"],
                   "believed_bins": int((export["belief"] > 0).sum()),
                   "site": SITE}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
