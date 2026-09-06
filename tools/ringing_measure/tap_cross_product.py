"""The Hilbert cross product of one video field: record tap against playback.

Ethan: *"hilbert cross product for one standard video field in our final
differential, and the same for a vcr playback, and vcr record cross
product. Use zaroff's recordings and playback samples to generate that. And
that will be a model of a bad playback and a good VCR."*

The record tap is the RF the deck wrote - the machine's own intention,
after emphasis, clipping and modulation, before the head ever touches the
tape. The playback tap is what came back through the tape, the head and the
head amplifier. Their cross product is therefore the degradation with the
tape in the middle: the good VCR against the bad playback.

Run:
    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/tap_cross_product.py [pattern ...]

The captures are 24,000,512 samples at 50 MSps (the FLAC header says 50 kHz
and is wrong) and 0.48 seconds long, so one NTSC field is 834,167 samples
and a capture holds about 28 of them. The patterns are static, so any field
of one capture matches any field of the other and the alignment is a lag,
not a search for the same frame.

MEASURED (2026-09-05, SP, one field from the middle of each capture, the
best whole-sample lag and then the best fractional delay removed):

    pattern          lag      peak/rms   transfer   quadrature share
                                                    before    after
    75bars          509546      109        0.078     0.087     0.003
    multiburst      466438       87        0.016     0.627     0.400
    ntc7composite   770376       78       -0.010     0.723     0.900
    control (record against itself)         1.000     0.000     0.000

On 75 bars the whole record-to-playback relation is a gain and a delay:
three parts in a thousand of the cross power survive in the quadrature
components once the delay is removed. On multiburst two fifths survive, and
a multiburst is precisely the pattern that carries many frequencies at once,
so a frequency-dependent phase is what a single delay cannot absorb.

AND THE LIMIT OF THIS PARTICULAR MEASUREMENT, which the transfer column
states plainly: the in-phase correlation between the two taps is only one
to eight per cent. The RF is a frequency-modulated carrier and its absolute
phase is not reproducible between the pass that wrote the tape and the pass
that read it, so a cross product taken on the SAMPLES is dominated by
carrier phase that carries no information about the channel. That is why
the NTC7 row moves the wrong way when its delay is fitted: with a transfer
of -0.010 the fit is maximising noise. The comparison that does not have
this limit is per frequency bin on the envelope or on the demodulated
signal, which is what `models/tap_transfer.py` builds; this tool is the
field-level view and its 75-bars row is the one to trust.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vhsdecode.models import hypercomplex, profiles  # noqa: E402

RECORD = "/testdata/test_patterns/vhs/record/zaroff-%s-NTSC-SP-SLV-778HF-50msps-rf-rec.flac"
PLAYBACK = "/testdata/test_patterns/vhs/playback/zaroff-%s-NTSC-SP-SLV-778HF-50msps-rf-pb.flac"
RATE_HZ = 50e6                      # the capture rate; the header's 50 kHz is wrong
PATTERNS = ("75bars", "multiburst", "ntc7composite")


def read_window(path, start, count):
    """One window of a capture, centred and scaled to unit variance."""
    with sf.SoundFile(path) as handle:
        handle.seek(start)
        block = handle.read(count, dtype="float64", always_2d=False)
    block = np.asarray(block).ravel()
    block = block - block.mean()
    scale = block.std()
    return block / (scale if scale > 0 else 1.0)


def align(a, b):
    """The lag of b against a, from the whitened cross-correlation."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    product = np.conj(A) * B
    whitened = product / (np.abs(product) + np.median(np.abs(product)) + 1e-300)
    correlation = np.fft.irfft(whitened, n)
    peak = int(np.argmax(np.abs(correlation)))
    lag = peak if peak < n // 2 else peak - n
    return lag, float(np.abs(correlation).max() / (np.abs(correlation).std() + 1e-300))


def main(argv):
    patterns = argv[1:] or list(PATTERNS)
    field_samples = int(round(RATE_HZ / (60.0 / 1.001)))
    print(__doc__.split("Run:")[0].rstrip())
    print()
    profile = profiles.for_capture("zaroff")
    print("profile: %s / %s / %s" % (profile["system"]["name"],
                                     profile["source"]["name"],
                                     profile["deck"]["name"]))
    print("one field is %d samples at %.0f MSps" % (field_samples, RATE_HZ / 1e6))
    print()
    print("%-16s %8s %10s %12s %12s %12s %12s" % (
        "pattern", "lag", "peak/rms", "transfer", "mixed after", "mixed before",
        "control"))
    for pattern in patterns:
        record_path, playback_path = RECORD % pattern, PLAYBACK % pattern
        if not (os.path.exists(record_path) and os.path.exists(playback_path)):
            print("%-16s capture missing, skipped" % pattern)
            continue
        # a field from the middle of each capture, away from either end
        start = 8 * field_samples
        record = read_window(record_path, start, field_samples)
        playback = read_window(playback_path, start, field_samples)
        lag, sharpness = align(record, playback)
        rolled = np.roll(playback, -lag)
        # THE FRACTIONAL PART TOO, or it is counted as degradation. A whole
        # sample roll leaves up to half a sample of delay, and a delay is
        # exactly what the quadrature components carry - so an uncorrected
        # fraction would report the alignment's own remainder as the
        # channel's phase. The best pure delay is found on a grid and
        # removed with the exact phase-ramp shift, and what remains is what
        # no single delay can explain.
        best, best_fraction = None, 0.0
        for fraction in np.linspace(-0.5, 0.5, 21):
            trial = hypercomplex.fractional_delay(rolled, fraction)
            value = hypercomplex.cross_product(trial, record, (0,))["transfer"]
            if best is None or value > best:
                best, best_fraction = value, fraction
        aligned = hypercomplex.fractional_delay(rolled, best_fraction)
        cross = hypercomplex.cross_product(aligned, record, (0,))
        raw = hypercomplex.cross_product(rolled, record, (0,))
        # the record tap against itself, as the control: a perfect chain
        control = hypercomplex.cross_product(record, record, (0,))
        print("%-16s %8d %10.1f %12.4f %12.4f %12.4f %12.4f" % (
            pattern, lag, sharpness, cross["transfer"], cross["mixed_share"],
            raw["mixed_share"], control["mixed_share"]))
    print()
    print("transfer is the matched (in-phase) part of the relation; mixed share is")
    print("the fraction of the cross power in the quadrature components, which is")
    print("where a delay or a reflection lives and which a magnitude comparison")
    print("cannot see. The record tap against itself is the control: a chain with")
    print("no degradation puts nothing in the quadrature terms.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
