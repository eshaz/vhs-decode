#!/usr/bin/env python3
"""The record-to-playback transfer: the tape, heads and preamp, measured
directly from the two RF taps.

The test-pattern captures include RECORD-side RF of the same signals as
the playback ones (CN261 pin 1 against pin 2, same deck, same session).
The ratio of their averaged spectra is everything BETWEEN the taps - the
record head drive, the tape, the read head and the preamp - which is the
channel the constant-envelope identifier tries to estimate, here measured
directly, at EVERY frequency, with no carrier-coverage limit and no
demodulator in the path.

Use the LUMA-ONLY captures wherever the lower sideband matters: the
color-under otherwise sits on top of it below about 1.2 MHz.

What the ratio is and is not:
  - between-signal SPREAD is the thing to report with it: for a linear
    time-invariant channel the ratio is the response squared at every
    frequency whatever signal carries it, so any content dependence means
    the path is not time-invariant or the captures differ by more than
    the channel. Here the dip is consistent while the bands above 5 MHz
    and below 1.1 MHz are not (spreads of 2.5 to 3.6 dB), which is where
    the signals' own energy differs most. Two explanations for the spread have been tested and
    falsified: the playback noise floor lifting the quiet signals, and
    the deck's automatic gain control changing the deviation between
    passes (correcting for the measured 2-8% deviation difference makes
    the spread worse, 0.77 -> 1.66 dB);
  - measured with the correct estimator the transfer is FLAT above the
    carrier - +0.26, +0.03, -0.38 dB across 3.9-5.5 MHz with spreads of
    0.06 to 0.97 - which is what a preamp equalizing its head's
    differentiation is designed to produce. An earlier reading of a
    2.3 to 2.9 dB "rise" there, attributed to residual head
    differentiation, was an artefact of averaging decibels instead of
    power and is withdrawn;
  - what remains is LOCAL: a dip of about 3.6 dB at 1.1 to 1.7 MHz,
    consistent across signals (spread 0.61 to 0.69 dB in the two tighter
    bands). That is the part to read;
  - the two captures are separate passes with their own oscilloscope
    gain, so only the SHAPE is meaningful, and every figure here is
    referenced to a band the caller names;
  - it does NOT see the record-side electronics BEFORE the record tap
    (the separation and combining filters), which is exactly the region
    a record-side truncation hypothesis lives in.

    record_playback_transfer.py [--pattern ntc7composite-y-only ...]
        [--speed SP] [--reference 3.4,3.9] [--out transfer.npz]
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf

ROOT = "/testdata/test_patterns/vhs"
RATE_HZ = 50e6                      # the captures' nominal rate
DEFAULT_PATTERNS = ("ntc7composite-y-only", "multiburst-y-only",
                    "pulseandbar-y-only", "ramp-y-only", "sweep-y-only")
DEFAULT_BANDS = ((0.2, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.4), (1.4, 1.7),
                 (1.7, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.4), (3.4, 3.9),
                 (3.9, 4.4), (4.4, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5))
# the deck's lead-in is unstable, so start well into the capture
SKIP = 3_000_000
TAKE = 8_000_000
SEGMENT = 32768


def path_for(pattern, speed, side):
    tag = "rec" if side == "record" else "pb"
    name = f"zaroff-{pattern}-NTSC-{speed}-SLV-778HF-50msps-rf-{tag}.flac"
    return os.path.join(ROOT, side, name)


def spectrum(path):
    """Welch POWER spectrum (not dB), mean-removed, Hann windowed: the
    banding below takes the mean of power and then the log, which is the
    unbiased estimator."""
    with sf.SoundFile(path) as handle:
        handle.seek(min(SKIP, max(len(handle) - TAKE, 0)))
        samples = handle.read(TAKE, dtype="float32", always_2d=False)
    samples = np.asarray(samples).ravel()
    samples = samples - samples.mean()
    window = np.hanning(SEGMENT)
    total = np.zeros(SEGMENT // 2 + 1)
    count = 0
    for start in range(0, len(samples) - SEGMENT, SEGMENT // 2):
        total += np.abs(np.fft.rfft(samples[start:start + SEGMENT] * window)) ** 2
        count += 1
    if not count:
        raise RuntimeError(f"{path}: too few samples")
    return np.fft.rfftfreq(SEGMENT, 1.0 / RATE_HZ), total / count


def band_mean(freqs, power, low_mhz, high_mhz):
    """The band's response in dB from the mean POWER, not the mean of the
    per-bin dB. The unbiased estimate of the response over a band is the
    ratio of mean powers; averaging decibels weights the quiet bins as
    heavily as the loud ones and biases the answer wherever the spectrum
    has structure - measured, it inflated a sideband imbalance from about
    0.3 nepers to 0.57."""
    inside = (freqs >= low_mhz * 1e6) & (freqs < high_mhz * 1e6)
    if not inside.any():
        return float("nan")
    return float(10.0 * np.log10(power[inside].mean() + 1e-30))


def transfer(patterns, speed, bands, reference):
    rows, freqs = {}, None
    for pattern in patterns:
        try:
            freqs, record = spectrum(path_for(pattern, speed, "record"))
            _, playback = spectrum(path_for(pattern, speed, "playback"))
        except (RuntimeError, OSError) as error:
            print(f"  skipped {pattern}: {error}", file=sys.stderr)
            continue
        row = np.array([band_mean(freqs, playback, *b)
                        - band_mean(freqs, record, *b) for b in bands])
        rows[pattern] = row - row[bands.index(reference)]
    return rows, freqs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument("--speed", default="SP")
    parser.add_argument("--reference", default="3.4,3.9",
                        help="the band every figure is referenced to")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    patterns = tuple(args.pattern or DEFAULT_PATTERNS)
    reference = tuple(float(v) for v in args.reference.split(","))
    bands = DEFAULT_BANDS if reference in DEFAULT_BANDS \
        else DEFAULT_BANDS + (reference,)
    rows, freqs = transfer(patterns, args.speed, bands, reference)
    if not rows:
        print("no capture pairs found", file=sys.stderr)
        return 1
    values = np.array([rows[p] for p in rows])
    print(f"record-to-playback transfer, {args.speed}, {len(rows)} signals, "
          f"referenced to {reference[0]}-{reference[1]} MHz")
    print("(the smooth tilt is the read head's differentiation and the "
          "preamp's equalization; the LOCAL structure is the part to read)\n")
    print("%-12s %9s %9s" % ("band MHz", "mean dB", "spread"))
    for index, (low, high) in enumerate(bands):
        print("%-12s %+9.2f %9.2f"
              % ("%.1f-%.1f" % (low, high), values[:, index].mean(),
                 values[:, index].std()))
    if args.out:
        np.savez(args.out, bands=np.array(bands), signals=np.array(list(rows)),
                 transfer_db=values, reference=np.array(reference),
                 freqs_hz=freqs)
        print("\nwritten:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
