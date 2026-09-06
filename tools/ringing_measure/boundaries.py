"""Run the stage-boundary survey and the distribution classifier.

Ethan: *"Let's make sure this is implemented in the code, since I want to test
it."*

This is the runnable form of two model modules:

    vhsdecode/models/stage_boundaries.py           where the unknowns live
    vhsdecode/models/interference_distributions.py what they are drawn from

With no arguments it prints the chain's joins, each join's prior, and what
each prior can and cannot resolve on a given band - all from the declared
chain, so it needs no data and answers "is this wired up".

Given a capture with `--capture`, it goes further and classifies the real
residual at the ONE join where a measurement is available without a stimulus:
the playback machine to the capture. Above the FM signal's own band the
recording carries nothing but noise, so the measured samples up there ARE the
unknown at that join, and its distribution can be read directly.

    python3 -m tools.ringing_measure.boundaries
    python3 -m tools.ringing_measure.boundaries --capture /testdata/countdown.flac
    python3 -m tools.ringing_measure.boundaries --capture ... --rate 40e6 --start 2e9
"""

import argparse
import sys

import numpy as np

from vhsdecode.models import interference_distributions as distributions
from vhsdecode.models import stage_boundaries as boundaries


def show_chain(band):
    print("=" * 74)
    print("THE CHAIN'S LINKS, from the declared component positions")
    print("=" * 74)
    for link in boundaries.links_of():
        if not link["occupied"]:
            continue
        members = ", ".join(link["members"][:4])
        more = f" (+{len(link['members']) - 4} more)" \
            if len(link["members"]) > 4 else ""
        print(f"  {link['link']:<24} {str(link['band']):<9} "
              f"{len(link['members']):2d} entries")
        print(f"      {members}{more}")

    print()
    print("=" * 74)
    print("THE JOINS, where the unknown components live")
    print("=" * 74)
    for join in boundaries.boundaries():
        prior = join["prior"]
        print(f"\n  {join['boundary']}")
        if not prior:
            print("      no prior declared")
            continue
        print(f"      family    {prior['family']}  "
              f"({prior['parameter']} over {prior['range']})")
        print(f"      why       {prior['why']}")
        reach = boundaries.resolvable(str(join["boundary"]), band)
        if reach.get("shortest_resolvable_s"):
            print(f"      on this band, the shortest resolvable "
                  f"{prior['parameter']} is "
                  f"{reach['shortest_resolvable_s'] * 1e9:.0f} ns, so "
                  f"{100 * reach['unresolvable_fraction']:.0f} per cent of "
                  f"the prior is unavailable")


def show_distributions():
    print()
    print("=" * 74)
    print("WHAT EACH INTERFERENCE TYPE IS DRAWN FROM")
    print("=" * 74)
    print(f"  {'component':<22}{'distribution':<20}{'excess kurtosis':>16}")
    for name, entry in distributions.DISTRIBUTIONS.items():
        moment = entry["excess_kurtosis"]
        shown = f"{moment:+.1f}" if moment is not None else "-"
        print(f"  {name:<22}{str(entry['distribution']):<20}{shown:>16}")
    print()
    print("  the discriminator, and it needs no model of the chain:")
    print("      arcsine -1.5  |  uniform -1.2  |  gaussian 0  |  dropouts >> 0")
    print()
    print("  how many samples it takes to separate the hardest pairs:")
    for pair in (("arcsine", "gaussian"), ("uniform", "gaussian"),
                 ("arcsine", "uniform")):
        control = distributions.separable_at(20000, pair)
        print(f"      {' vs '.join(pair):<22} gap {control['gap']:.1f}, "
              f"needs n >= {control['samples_needed']}")


def classify_capture(path, rate, start, length):
    """Read the capture's own noise floor and classify it.

    Above the FM band the recording is noise, so the samples there are the
    unknown at the playback-to-capture join with no stimulus needed.
    """
    try:
        import soundfile
    except ImportError:
        print("  soundfile is not available; cannot read the capture")
        return
    print()
    print("=" * 74)
    print("THE ONE JOIN WITH A MEASUREMENT: the playback machine -> the capture")
    print("=" * 74)
    samples = soundfile.read(path, start=int(start), frames=int(length),
                             dtype="int16")[0]
    samples = np.asarray(samples, dtype=np.float64).ravel()
    print(f"  {path}")
    print(f"  {samples.size} samples from {int(start)}, "
          f"rate {rate / 1e6:.3f} MSps, Nyquist {rate / 2e6:.3f} MHz")

    # the whole capture first, then the part above the signal band, which is
    # the part that is only noise
    spectrum = np.fft.rfft(samples * np.hanning(samples.size))
    frequency = np.fft.rfftfreq(samples.size, 1.0 / rate)
    above = frequency > 8e6
    floor = np.fft.irfft(np.where(above, spectrum, 0.0), n=samples.size)

    for name, block in (("the whole capture", samples),
                        ("above 8 MHz, which is only noise", floor)):
        verdict = distributions.classify(block)
        print(f"\n  {name}")
        print(f"      excess kurtosis {verdict['excess_kurtosis']:+.4f} "
              f"+- {verdict['standard_error']:.4f}")
        print(f"      sparsity        {verdict['sparsity']:.4f}")
        print(f"      -> {verdict['distribution']}"
              f"   identified={verdict['identified']}")
        if verdict.get("nearest") and not verdict["identified"]:
            print(f"      nearest candidate {verdict['nearest']} at a "
                  f"distance of {verdict['nearest_distance']:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="the stage-boundary survey and distribution classifier")
    parser.add_argument("--capture", help="an RF capture to classify")
    parser.add_argument("--rate", type=float, default=40e6,
                        help="the capture's sample rate (default 40 MSps)")
    parser.add_argument("--start", type=float, default=2e8,
                        help="sample offset into the capture")
    parser.add_argument("--length", type=float, default=1 << 21,
                        help="how many samples to read")
    parser.add_argument("--band", type=float, nargs=2, default=(1e6, 7e6),
                        metavar=("LOW", "HIGH"),
                        help="the band to report resolvability on")
    args = parser.parse_args(argv)

    band = np.linspace(args.band[0], args.band[1], 1024)
    show_chain(band)
    show_distributions()
    if args.capture:
        classify_capture(args.capture, args.rate, args.start, args.length)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
