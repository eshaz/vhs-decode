#!/usr/bin/env python3
"""Closure test: can ANY channel response explain the measured residual?

Borrowed from radio interferometry, where the per-antenna gains are
unknown and the observables that matter are the ones immune to them. Its
deeper use there is diagnostic rather than calibrating: because every
measurement involves a PAIR of antennas and every antenna appears in many
measurements, the measurements are over-determined, and combinations
exist that must vanish for ANY set of antenna gains. When they do not
vanish, the errors are not antenna-based and self-calibration can never
converge, however many rounds are run.

Our measurements have the same structure. A demodulated residual at
baseband offset x from a landing carrier c is not a measurement of one
radio frequency but of three:

    magnitude   [e(c+x) + e(c-x)]/2 - e(c)
    phase       [e(c+x) - e(c-x)]/2

for an applied log-response change e. Every radio frequency appears in
many such rows, from both landings, so the system is over-determined and
has a left null space: combinations of the measured residuals that are
zero for every possible e. Their size against the gauge's own error is
the answer to the question the loop keeps failing at - is the residual a
CHANNEL, or is it something no pre-demodulator response can produce?

  chi-square per degree of freedom near one   a channel explains the
                                              measurements; the loop's
                                              failure is estimation, and
                                              more or better passes help
  chi-square per degree of freedom far above  NO channel explains them;
                                              the residual contains
                                              something not of this form
                                              and inverting it at this
                                              site cannot work

The same design matrix also gives the RESOLUTION - radio astronomy's
sampling function, or which parts of the aperture were actually
measured. `diag(A+ A)` per radio-frequency bin says how much of each bin
the data determine as against the regularizer: a bin near one is
measured, a bin near zero is being supplied by the prior and must not be
reported as a finding.

    closure_test.py --work DIR --decode NAME [--decoder-response npz]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import multidimensional_information_extrapolation as loop  # noqa: E402


def design(views, filters, grid_hz):
    """The rows relating each measurement to the radio-frequency bins it
    involves, with its weight and its value: A, b, w for the magnitude
    (a second difference about the carrier) and the phase (a first
    difference). Returns them with the bin span the rows touch."""
    bin_hz = float(grid_hz[1] - grid_hz[0])
    carriers = loop.landing_carriers(filters)
    rows_m, b_m, w_m, rows_p, b_p, w_p = [], [], [], [], [], []
    touched = np.zeros(len(grid_hz), dtype=bool)
    for view, entry in views.items():
        freqs, L, var, valid = entry[:4]
        centre = carriers[view]
        index_c = int(np.round((centre - grid_hz[0]) / bin_hz))
        for k in np.nonzero(valid)[0]:
            up = int(np.round((centre + freqs[k] - grid_hz[0]) / bin_hz))
            down = int(np.round((centre - freqs[k] - grid_hz[0]) / bin_hz))
            if not (0 <= up < len(grid_hz) and 0 <= down < len(grid_hz)
                    and 0 <= index_c < len(grid_hz)):
                continue
            weight = 1.0 / var[k]
            if not np.isfinite(weight) or weight <= 0:
                continue
            touched[[up, down, index_c]] = True
            rows_m.append((up, down, index_c))
            b_m.append(float(L[k].real))
            w_m.append(weight)
            if abs(L[k].imag) <= np.pi:
                rows_p.append((up, down))
                b_p.append(float(L[k].imag))
                w_p.append(weight)
    span = np.nonzero(touched)[0]
    return (rows_m, np.array(b_m), np.array(w_m),
            rows_p, np.array(b_p), np.array(w_p),
            int(span.min()), int(span.max()) + 1)


def _matrix(rows, low, width, odd):
    A = np.zeros((len(rows), width))
    for index, entry in enumerate(rows):
        if odd:
            up, down = entry
            A[index, up - low] += 0.5
            A[index, down - low] -= 0.5
        else:
            up, down, centre = entry
            A[index, up - low] += 0.5
            A[index, down - low] += 0.5
            A[index, centre - low] -= 1.0
    return A


def closure(rows, b, w, low, width, odd, name):
    """The measurements' departure from ANYTHING this design can produce.

    The best fit's weighted residual IS the projection onto the left null
    space - the closure quantities - so its chi-square per degree of
    freedom answers the question directly, with the degrees of freedom
    counted as rows minus the rank the data actually constrain."""
    if not len(rows):
        return None
    A = _matrix(rows, low, width, odd)
    root = np.sqrt(w)[:, None]
    solution, _, rank, singular = np.linalg.lstsq(A * root, b * np.sqrt(w),
                                                  rcond=None)
    residual = b - A @ solution
    chi2 = float(np.sum(w * residual ** 2))
    dof = max(len(rows) - int(rank), 1)
    # how much of each bin the data determine, against the regularizer:
    # the sampling function, in radio-astronomy terms
    pseudo = np.linalg.pinv(A * root)
    resolution = np.diag(pseudo @ (A * root))
    return {"name": name, "rows": len(rows), "rank": int(rank), "dof": dof,
            "chi2": chi2, "reduced": chi2 / dof,
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "measured_rms": float(np.sqrt(np.mean(b ** 2))),
            "resolution": resolution,
            "condition": float(singular.max() / max(singular.min(), 1e-30))
            if len(singular) else float("nan")}


def report(result, grid_hz, low):
    if result is None:
        print("  no rows")
        return
    print("  %s: %d rows, rank %d, %d degrees of freedom" %
          (result["name"], result["rows"], result["rank"], result["dof"]))
    print("     measured rms %.4f, closure residual rms %.4f (%.0f%% of it)"
          % (result["measured_rms"], result["residual_rms"],
             100 * result["residual_rms"] / max(result["measured_rms"], 1e-30)))
    print("     chi-square per degree of freedom %.1f" % result["reduced"])
    print("     %s" % ("a channel CAN explain these measurements"
                       if result["reduced"] < 3.0 else
                       "NO channel of this form explains them - the residual "
                       "is not a pre-demodulator response"))
    resolution = result["resolution"]
    determined = resolution > 0.5
    if determined.any():
        f = grid_hz[low:low + len(resolution)][determined]
        print("     determined bins: %d of %d, %.2f-%.2f MHz"
              % (determined.sum(), len(resolution), f.min() / 1e6,
                 f.max() / 1e6))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--decoder-response", default=None)
    parser.add_argument("--bin-hz", type=float, default=5e3)
    args = parser.parse_args(argv)

    decode = type("Loaded", (), {})()
    decode.name = args.decode
    decode.prefix = os.path.join(args.work, args.decode)
    decode.residual_dir = os.path.join(args.work, args.decode + "_rc")
    loop.load_products(decode)
    reference = loop.load_decoder_reference(args.decoder_response)
    views = loop.pooled_views([loop.frequency_residual(decode, parity, None,
                                                       reference)
                               for parity in (0, 1)])
    grid = np.arange(0.5e6, 8e6, args.bin_hz)
    rows_m, b_m, w_m, rows_p, b_p, w_p, low, high = design(views,
                                                           decode.filters,
                                                           grid)
    width = high - low
    print(f"closure test on {args.decode}: can any pre-demodulator channel "
          f"explain the measured residual?\n")
    for rows, b, w, odd, name in ((rows_m, b_m, w_m, False, "magnitude"),
                                  (rows_p, b_p, w_p, True, "phase")):
        report(closure(rows, b, w, low, width, odd, name), grid, low)
    return 0


if __name__ == "__main__":
    sys.exit(main())
