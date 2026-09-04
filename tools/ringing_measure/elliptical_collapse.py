"""Elliptical curve estimation over the measured components.

Ethan's algorithm, made runnable. In his words, across the messages that
specified it:

    The differentiation is the differential of our synthetic components to
    the actual measured componnet, this has nested differentials in it as
    well according to our graph, so this expands out to a multi dimensional
    matrix that represents our known components.

    The shapes of the dimensions are asymetric, and we can extrapolate how
    symetric they are based on their fit to our 3 dimensional eliptical
    plane which represents the final result. The constant is the elipse.
    All I need is to fit these residuals to an elipse of how ever many
    dimensions I have, and I have fully collapsed the differentiation.

    Then differentiation using the target curve onto the source data. This
    loops until the information floor is reached (total possible
    information on the source signal) ... Down to the circular shape of the
    complex signal.

    I think the total eigenvalue is the total amount of information we
    could possible derive from our components. I.e. this is multi component
    radio tuning.

WHAT IT DOES, in the order it does it:

  1. reads the capture card's profile from the capture itself - word length
     from the code lattice, rate, occupancy - and computes which link binds,
     the tape's radio channel or the converter;
  2. decodes, harvesting per field the measured frequency response over
     carrier frequency and the model in force at the same bins;
  3. forms the differential of synthetic against measured per field, and the
     nested differentials the graph relates;
  4. fits the ellipse to that matrix and reports its shape - the information
     budget, how much of it stands in real directions, and how far the shape
     is from a sphere;
  5. runs the loop to the floor and reports where it stopped and why.

USAGE

    python3 -m tools.ringing_measure.elliptical_collapse CAPTURE OUT \\
        [-- decode flags ...]

Everything after `--` is handed to the decoder unchanged. With no flags the
decode runs the arc's standard set. Nothing is written to the decode's
output beyond what the decoder itself writes; this tool only reports.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import vhsdecode.luma_amplitude as la
from vhsdecode.residual_limit import cell_medians
from vhsdecode.models import capture_profile as cp
from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import interference as inf

# The decode flags this arc measures on. Stated here rather than left to the
# caller so two runs are comparable by default; anything after `--`
# overrides them entirely.
DEFAULT_FLAGS = ["-n", "-f", "50", "-l", "14", "-t", "4", "--dctp",
                 "--lti_gain", "0", "--ire0_adjust",
                 "--chroma_env_gain", "0.5", "--overwrite"]

# A bin needs this many samples behind it before its median is a
# measurement rather than a sample. The same gate the runtime uses.
MINIMUM_POPULATION = 64


class Harvest:
    """Per-field measured response and the model in force at the same bins.

    The measured side is the binned median of the flattened envelope over
    carrier frequency - the decoder's own path response already divided out,
    so what remains is the tape, the heads and the two machines' equalisation
    together. The synthetic side is the response curve the decoder fits and
    acts on, evaluated at the same bin centres.

    Their difference is Ethan's differential: the model the specification
    implies against what the tape actually produced.
    """

    def __init__(self):
        self.fields: List[Dict[str, object]] = []
        self.axis_hz: Optional[np.ndarray] = None
        self._seen = 0

    def take(self, field) -> None:
        rf = field.rf
        video = field.data["video"]
        envelope = np.asarray(video["envelope"], dtype=np.float32)
        raw = np.asarray(video["demod_raw"], dtype=np.float32)
        # the carrier at each sample, from the demodulated frequency
        carrier = np.empty_like(raw)
        np.add(raw[:-1], raw[1:], out=carrier[:-1])
        carrier[:-1] *= np.float32(0.5)
        carrier[-1] = raw[-1]

        response, step = la.rf_path_response(rf)
        sync_tip, _, bin_count = la.carrier_frequency_bins(rf.SysParams, rf)
        bin_hz = rf.SysParams["hz_ire"] * la.RESPONSE_CURVE_RESOLUTION_IRE
        stride = max(1, len(envelope) // (la.CURVE_INDEPENDENT_POINTS
                                          * la.CURVE_SAMPLES_PER_POINT))
        flat = la._flatten_by_response(
            np.asarray(envelope[::stride], dtype=np.float64),
            np.asarray(carrier[::stride], dtype=np.float64), response, step)
        bins = la._frequency_bin_of(
            np.asarray(carrier[::stride], dtype=np.float64),
            sync_tip, bin_hz, bin_count)
        level, population, centre = la._binned_median(
            flat, bins, sync_tip, bin_hz, bin_count)

        usable = (population >= MINIMUM_POPULATION) & (level > 0.0)
        if usable.sum() < 16:
            return
        measured = np.zeros(bin_count)
        measured[usable] = np.log(level[usable])

        # the model in force: the decoder's own fitted response curve
        curve = la.response_curve(level, population, centre)
        synthetic = np.zeros(bin_count)
        if curve is not None:
            centre_hz, level_ln, slope = curve
            axis = sync_tip + (np.arange(bin_count) + 0.5) * bin_hz
            synthetic = level_ln + slope * (axis - centre_hz)

        if self.axis_hz is None:
            self.axis_hz = sync_tip + (np.arange(bin_count) + 0.5) * bin_hz
        record = {
            "index": self._seen,
            "head": bool(field.isFirstField),
            "usable": usable,
            "measured": measured,
            "synthetic": synthetic,
            "population": population,
            "spread": self._spread(flat, bins, bin_count),
        }
        self._seen += 1
        self._take_amplitude(record, rf, carrier, flat, stride)
        self._take_time(record, field)
        self.fields.append(record)

    @staticmethod
    def _take_amplitude(record, rf, carrier, flat, stride) -> None:
        """The AMPLITUDE axis: the envelope over the sweep argument.

        The collapse's own variable - how fast the carrier is moving when
        the sample was taken - which is the axis the level-dependent part
        of the path lives on. The synthetic side is flat: the
        specification says the envelope does not depend on how fast the
        carrier is sweeping, and every departure from that is the
        measurement.
        """
        try:
            tables = la.sweep_collapse_tables(rf)
            edges = la.sweep_residue_edges()
            response, step = la.rf_path_response(rf)
            grid = np.arange(len(response)) * step
            track = np.empty_like(carrier)
            slope = np.empty_like(carrier)
            la._sweep_arrays(carrier, tables.span, tables.track_width,
                             track, slope)
            sampled = np.asarray(carrier[::stride], dtype=np.float64)
            theta = (np.interp(sampled, grid, tables.q)
                     * np.asarray(slope[::stride], dtype=np.float64) ** 2)
            place = np.clip(np.searchsorted(edges, theta, side="right") - 1,
                            0, len(edges) - 1)
            keep = flat > 0
            level, population = cell_medians(
                place[keep], np.log(flat[keep]), len(edges))
        except Exception:
            return
        usable = population >= MINIMUM_POPULATION
        if usable.sum() < 4:
            return
        record["amplitude_usable"] = usable
        record["amplitude_measured"] = np.where(usable, level, 0.0)
        record["amplitude_synthetic"] = np.zeros_like(level)

    @staticmethod
    def _take_time(record, field) -> None:
        """The TIME axis: the line period against the SPEC period.

        The synthetic side is a constant - "a perfect flat timebase and
        constant" - so the differential is the residual itself, which is
        what the design has always called the time-base residual.
        """
        try:
            measured = la.field_time_base(field)
        except Exception:
            return
        if measured is None:
            return
        values, live = measured
        values = np.asarray(values, dtype=np.float64)
        live = np.asarray(live, dtype=bool)
        if live.sum() < 16:
            return
        record["time_usable"] = live
        record["time_measured"] = np.where(live, values, 0.0)
        record["time_synthetic"] = np.zeros_like(values)

    @staticmethod
    def _spread(flat, bins, bin_count) -> float:
        """The within-field spread of the samples inside a bin, in dB.

        This is the tape's own carrier-to-noise seen directly: the FM
        carrier is nominally constant in amplitude, so what the envelope
        does between samples inside one frequency bin is the channel's
        noise rather than the picture.
        """
        spreads = []
        live = (bins >= 0) & (bins < bin_count)
        for index in np.unique(bins[live]):
            chosen = flat[bins == index]
            chosen = chosen[chosen > 0]
            if chosen.size >= MINIMUM_POPULATION:
                spreads.append(float(np.std(np.log(chosen))))
        return float(np.mean(spreads) * la._LOG_TO_DB) if spreads else float("nan")


def common_bins(fields: List[Dict[str, object]]) -> np.ndarray:
    """The bins every field in the ensemble describes.

    A component that is only defined where another is not cannot be
    differenced against it, so the ensemble is cut to what they share. The
    alternative - padding the gaps with zeros - would put the padding into
    the ellipse as though it were a measurement.
    """
    if not fields:
        return np.zeros(0, dtype=bool)
    shared = fields[0]["usable"].copy()
    for field in fields[1:]:
        shared &= field["usable"]
    return shared


# The four axes the model declares, and the keys each is harvested under.
# R2 of the design: the limit is taken in every measured dimension at once,
# so the collapse is run on each rather than on the frequency axis alone.
AXES = {
    "frequency": ("usable", "measured", "synthetic",
                  "the response over carrier frequency"),
    "amplitude": ("amplitude_usable", "amplitude_measured",
                  "amplitude_synthetic", "the envelope over the sweep"),
    "time": ("time_usable", "time_measured", "time_synthetic",
             "the line period against the spec period"),
}


def common_support(fields, key: str) -> np.ndarray:
    """The places every field in the ensemble describes.

    A component defined only where another is not cannot be differenced
    against it, so the ensemble is cut to what they share. Padding the gaps
    with zeros would put the padding into the ellipse as a measurement.
    """
    present = [f[key] for f in fields if key in f]
    if not present:
        return np.zeros(0, dtype=bool)
    width = min(int(np.asarray(p).size) for p in present)
    shared = np.ones(width, dtype=bool)
    for mask in present:
        shared &= np.asarray(mask, dtype=bool)[:width]
    return shared


def build_matrix(fields: List[Dict[str, object]], head: Optional[bool],
                 axis: str = "frequency"):
    """The nested differential matrix for one head's fields, on one axis.

    Fields of the same head are siblings in the graph: they do not derive
    from one another, but they share a parent - the head's own path - so a
    differential between any two of them is a relationship the graph
    states.
    """
    usable_key, measured_key, synthetic_key, _ = AXES[axis]
    chosen = [f for f in fields
              if (head is None or f["head"] == head) and measured_key in f]
    if len(chosen) < 3:
        return None, chosen, np.zeros(0, dtype=bool)
    shared = common_support(chosen, usable_key)
    if shared.sum() < 8:
        return None, chosen, shared
    width = shared.size
    synthetic = {f"field {f['index']}":
                 np.asarray(f[synthetic_key])[:width][shared] for f in chosen}
    measured = {f"field {f['index']}":
                np.asarray(f[measured_key])[:width][shared] for f in chosen}
    names = list(synthetic)
    relations = {name: [other for other in names if other != name]
                 for name in names}
    matrix = ie.component_differentials(synthetic, measured, axis, relations)
    return matrix, chosen, shared


def build_field_matrix(fields: List[Dict[str, object]],
                       head: Optional[bool]):
    """The FIELD axis: the transpose.

    Every other axis has the fields as its components and the measurement's
    own grid as its length. The per-field dimension is the other way round -
    each frequency bin is a component and its vector runs over the fields -
    so what the ellipse reads there is how the bins vary together from one
    field to the next, which is the non-reducible residual the design says
    is tracked field by field.
    """
    chosen = [f for f in fields if head is None or f["head"] == head]
    if len(chosen) < 4:
        return None, chosen, np.zeros(0, dtype=bool)
    shared = common_support(chosen, "usable")
    if shared.sum() < 8:
        return None, chosen, shared
    places = np.flatnonzero(shared)
    synthetic, measured = {}, {}
    for place in places:
        synthetic[f"bin {place}"] = np.array(
            [f["synthetic"][place] for f in chosen])
        measured[f"bin {place}"] = np.array(
            [f["measured"][place] for f in chosen])
    names = list(synthetic)
    relations = {name: [other for other in names if other != name]
                 for name in names}
    matrix = ie.component_differentials(synthetic, measured, "field",
                                        relations)
    return matrix, chosen, shared


def vertical_interval(output: str, guard_ire: float = 12.0):
    """The components carried in the VERTICAL INTERVAL, read from the TBC.

    Ethan: *"I have those components in the vertical blanking intervals of
    my recordings, use those."* They are the best components this algorithm
    can be given, for two reasons that no picture line shares.

    First, the synthetic side is KNOWN rather than fitted. A vertical
    interval test signal is specified - its bar amplitude, its multiburst
    frequencies, its pulse half-amplitude duration - so the differential is
    against the standard itself instead of against a curve fitted to the
    same data it is then judged on, which is what R5 exists to prevent.

    Second, the vertical interval is NOT active picture, so using it does
    not cross the standing sync-only constraint: the correction still
    derives from what the format reserves for measurement rather than from
    programme content.

    A line is taken as carrying a component when it departs from blanking
    by more than the interval's own scatter allows. A capture whose
    vertical interval is blank reports none, which is a finding rather than
    a failure - the 75-bars capture here is blank from line 9 to 19.
    """
    stem = output[:-4] if output.endswith(".tbc") else output
    try:
        with open(stem + ".tbc.json") as handle:
            meta = json.load(handle)
        parameters = meta["videoParameters"]
        width = int(parameters["fieldWidth"])
        height = int(parameters["fieldHeight"])
        black = float(parameters["black16bIre"])
        white = float(parameters["white16bIre"])
        frames = np.fromfile(stem + ".tbc", dtype=np.uint16)
    except Exception as error:
        return None, f"vertical interval unavailable ({error})"
    usable = (frames.size // (width * height)) * width * height
    if not usable:
        return None, "no field data"
    fields = frames[:usable].reshape(-1, height, width)
    scale = 100.0 / max(white - black, 1e-9)

    # After the vertical sync and equalising pulses, and STOPPING BEFORE
    # THE PICTURE. The window used to run to line 22, which is two lines
    # into active video, and the "perfect needle" it reported on a
    # full-field test pattern was the first two picture lines - identical
    # every field, as picture from a pattern generator is. On a programme
    # recording it fed two lines of content into the ellipse, which the
    # standing sync-only constraint forbids outright: the correction
    # derives from sync and reserved intervals, never from active picture.
    #
    # The bound is taken from the format's own line counts rather than a
    # constant. SMPTE 170M gives 525/60 nine lines of vertical blanking
    # after equalisation, with active video from line 20; the ratio is the
    # same in 625/50, so scaling by the field height carries to PAL without
    # a second number.
    first = max(int(round(height * 9.0 / 263.0)), 7)
    last = int(round(height * 20.0 / 263.0))
    if last <= first:
        return None, "no reserved interval between the pulses and the picture"
    lines, blank = [], []
    for line in range(first, last):
        rows = (fields[:, line, int(0.12 * width):int(0.95 * width)]
                .astype(np.float64) - black) * scale
        span = float(np.median(rows.max(axis=1) - rows.min(axis=1)))
        (lines if span > guard_ire else blank).append(
            {"line": line, "span": span, "rows": rows})
    if not lines:
        spans = [entry["span"] for entry in blank] or [0.0]
        return None, (f"blank from line {first} to {last - 1} "
                      f"(largest departure {max(spans):.1f} IRE of the "
                      f"{guard_ire:.0f} IRE guard) - no test signal inserted")
    return lines, (f"{len(lines)} line(s) carrying a component: "
                   + ", ".join(f"line {entry['line']} at {entry['span']:.0f} IRE"
                               for entry in lines))


def report_profile(path: str, sample_rate_hz: float, spread_db: float) -> Dict:
    """The capture chain: the fourth link, and the terminal bound."""
    try:
        import soundfile as sf
        info = sf.info(path)
        frames = min(1 << 20, int(info.frames))
        samples, _ = sf.read(path, frames=frames, start=min(int(2e6),
                                                            max(info.frames - frames, 0)),
                             dtype="int16")
    except Exception as error:                       # unreadable container
        print(f"  capture profile unavailable ({error})")
        return {}
    profile = cp.profile_of(samples, sample_rate_hz)
    print("CAPTURE PROFILE, read from the capture")
    print(f"  word length            {profile['bits']:.0f} bit "
          f"(step {profile['step']:.0f} in the container)")
    print(f"  codes present          {profile['codes']:.0f}, occupying "
          f"{100 * profile['occupancy']:.1f}% of full scale")
    print(f"  sample rate            {profile['sample_rate_hz'] / 1e6:.1f} MHz")
    print(f"  signal                 {profile['signal_rms_codes']:.1f} codes rms")
    print(f"  quantisation floor     {profile['signal_to_noise_db']:.1f} dB "
          f"full band, budget {profile['budget_bits_per_s'] / 1e6:.0f} Mbit/s")
    if np.isfinite(spread_db):
        limit = cp.binding_limit(spread_db, profile["signal_rms_codes"], profile)
        print()
        print("WHICH LINK BINDS  (Shannon over the band Carson's rule gives)")
        print(f"  band                   {limit['band_hz'] / 1e6:.1f} MHz")
        print(f"  the tape (radio)       {limit['tape_snr_db']:5.1f} dB C/N  ->"
              f" {limit['tape_capacity_bits_per_s'] / 1e6:7.1f} Mbit/s")
        print(f"  the capture card       {limit['capture_snr_db']:5.1f} dB C/N  ->"
              f" {limit['capture_capacity_bits_per_s'] / 1e6:7.1f} Mbit/s")
        print(f"  -> the {limit['binds'].upper()} binds; the converter holds "
              f"{limit['ratio']:.1f}x and spends "
              f"{limit['headroom_bits_per_s'] / 1e6:.0f} Mbit/s on nothing")
        profile["binding"] = limit
    return profile


def report_ellipse(matrix, label: str) -> Dict:
    axis = next(iter(next(iter(matrix.values()))))
    # `removed=` matters: a nested matrix is rank-deficient by construction,
    # and charging that deficiency to the shape reports the construction's
    # own bookkeeping as structure.
    provisional = ie.ellipsoid(matrix, axis)
    deficiency = max(len(provisional["names"]) - int(provisional["rank"]), 0)
    fit = ie.ellipsoid(matrix, axis, removed=deficiency)
    if not fit["names"]:
        print(f"  {label}: no projectable group")
        return {}
    print(f"  {label}")
    print(f"    entries {len(fit['names'])}, numerical rank {fit['rank']}, "
          f"directions above the noise edge {fit['significant']}")
    print(f"    information budget {fit['information']:.0f} units, "
          f"{fit['resolved']:.3f} resolved "
          f"({100 * fit['resolved_fraction']:.1f}%)")
    print(f"    asymmetry {fit['asymmetry']:.4f} against a sphere floor of "
          f"{fit['sphere_floor']:.4f}  ->  {fit['sigma']:+.1f} sigma")
    axes = np.asarray(fit["semi_axes"])[:min(6, len(fit["semi_axes"]))]
    print(f"    semi-axes {np.round(axes, 3)}")
    return fit


def report_loop(matrix, label: str) -> Dict:
    out = ie.differentiate_to_floor(matrix, "frequency", maximum_passes=8)
    print(f"  {label}")
    for index, step in enumerate(out["trace"]):
        print(f"    pass {index}: asymmetry {step['asymmetry']:.4f}  "
              f"floor {step['sphere_floor']:.4f}  {step['sigma']:+8.1f} sigma")
    print(f"    -> {out['reason']}; {len(out['targets'])} direction(s) "
          f"recovered in {out['passes']} pass(es)")
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    flags: List[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, flags = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(
        description="Elliptical curve estimation over the measured components")
    parser.add_argument("capture")
    parser.add_argument("output")
    parser.add_argument("--json", default=None,
                        help="write the numbers to this path as well")
    parser.add_argument("--rate", type=float, default=None,
                        help="capture sample rate in Hz "
                             "(default: read from the decode flags)")
    arguments = parser.parse_args(argv)
    flags = flags or list(DEFAULT_FLAGS)

    rate_hz = arguments.rate
    if rate_hz is None:
        rate_hz = 40e6
        if "-f" in flags:
            rate_hz = float(flags[flags.index("-f") + 1]) * 1e6
        elif "--frequency" in flags:
            rate_hz = float(flags[flags.index("--frequency") + 1]) * 1e6

    harvest = Harvest()
    original = la.measured_amplitude_deviation

    def wrapped(field):
        out = original(field)
        if out is not None and not getattr(field, "_collapse_seen", False):
            field._collapse_seen = True
            try:
                harvest.take(field)
            except Exception as error:               # never break a decode
                print(f"  (field skipped: {error})", file=sys.stderr)
        return out

    la.measured_amplitude_deviation = wrapped
    # The decoder reads `sys.argv` in places as well as the argv it is
    # handed, so this tool's own flags have to be off the list while it
    # runs or they arrive at its parser and the positional arguments land
    # in the wrong places.
    decode_argv = [arguments.capture, arguments.output] + flags
    saved_argv = list(sys.argv)
    sys.argv = ["vhs-decode"] + decode_argv
    try:
        from vhsdecode.main import main as decode
        decode(decode_argv)
    except SystemExit:
        pass
    finally:
        sys.argv = saved_argv
        la.measured_amplitude_deviation = original

    print()
    print("=" * 72)
    print("ELLIPTICAL CURVE ESTIMATION OVER THE MEASURED COMPONENTS")
    print("=" * 72)
    if not harvest.fields:
        print("no fields were harvested; nothing to fit")
        return 1
    spreads = [f["spread"] for f in harvest.fields
               if np.isfinite(f["spread"])]
    spread_db = float(np.mean(spreads)) if spreads else float("nan")
    print(f"fields harvested {len(harvest.fields)} "
          f"({sum(1 for f in harvest.fields if f['head'])} on one head, "
          f"{sum(1 for f in harvest.fields if not f['head'])} on the other)")
    print(f"within-field envelope spread {spread_db:.3f} dB per sample")
    print()
    results: Dict[str, object] = {
        "fields": len(harvest.fields),
        "spread_db": spread_db,
    }
    results["profile"] = report_profile(arguments.capture, rate_hz, spread_db)

    print()
    print("THE VERTICAL INTERVAL  (components whose synthetic side is KNOWN)")
    vbi, note = vertical_interval(arguments.output)
    print(f"  {note}")
    if vbi:
        matrix = {}
        for entry in vbi:
            rows = entry["rows"]
            # the standard's own synthetic side for a reserved line is
            # blanking; every departure from it is the inserted component
            # plus what the chain did to it
            for index, row in enumerate(rows):
                matrix[f"line {entry['line']} field {index}"] = {
                    "amplitude": -row}
        if len(matrix) >= 3:
            fit = ie.ellipsoid(matrix, "amplitude")
            print(f"    ellipse over {len(fit['names'])} vertical-interval "
                  f"components: rank {fit['rank']}, directions "
                  f"{fit['significant']}, asymmetry {fit['asymmetry']:.4f} "
                  f"vs floor {fit['sphere_floor']:.4f} ({fit['sigma']:+.1f} sigma)")
            out = ie.differentiate_to_floor(matrix, "amplitude",
                                            maximum_passes=8)
            print(f"    loop: {len(out['targets'])} direction(s) in "
                  f"{out['passes']} pass(es); {out['reason']}")
            results["vertical_interval"] = {
                "lines": [entry["line"] for entry in vbi],
                "directions": len(out["targets"]),
                "reason": out["reason"]}
    else:
        results["vertical_interval"] = {"note": note}

    print()
    print("THE NESTED DIFFERENTIAL MATRIX, IN EVERY MEASURED DIMENSION")
    print("(synthetic minus measured, and the differences the graph relates:")
    print(" fields of one head are siblings, sharing that head's own path)")

    def ensembles():
        """Every (axis, head) ensemble, and how to build its matrix."""
        for axis in AXES:
            for head, who in ((True, "head A"), (False, "head B"),
                              (None, "both heads")):
                yield axis, who, lambda a=axis, h=head: build_matrix(
                    harvest.fields, h, a)
        for head, who in ((True, "head A"), (False, "head B"),
                          (None, "both heads")):
            yield "field", who, lambda h=head: build_field_matrix(
                harvest.fields, h)

    built = {}
    for axis, who, make in ensembles():
        matrix, chosen, shared = make()
        label = f"{axis:>9} / {who}"
        if matrix is None:
            print(f"  {label}: not enough evidence "
                  f"({len(chosen)} fields, {int(np.sum(shared))} shared places)")
            continue
        built[(axis, who)] = matrix
        note = AXES[axis][3] if axis in AXES else "the per-field variation"
        print(f"  {label}: {len(chosen)} fields over {int(np.sum(shared))} "
              f"places -> {len(matrix)} entries   ({note})")

    print()
    print("IS THE KEY COMPLETE?  the share of the measured residual that the")
    print("modelled set of possible modifications spans")
    shared = common_support(harvest.fields, "usable")
    if shared.sum() >= 16:
        rf_first = harvest.fields[0]
        width = shared.size
        places = np.flatnonzero(shared)
        # THE CARRIER AXIS THE RESPONSE WAS ACTUALLY BINNED ON, taken from
        # the decoder rather than assumed. A guessed 1-7 MHz span was wrong
        # by a factor of 2.2 against the real 2.543-5.257 MHz, and the
        # modelled signatures are functions of frequency, so evaluating
        # them off-axis measures the wrong shapes entirely.
        grid = np.asarray(harvest.axis_hz)[:width][shared]
        measured = {}
        for entry in harvest.fields:
            value = (np.asarray(entry["synthetic"])[:width][shared]
                     - np.asarray(entry["measured"])[:width][shared])
            measured[f"field {entry['index']}"] = value
        cover = inf.span_completeness(measured, grid)
        print(f"  modelled basis {cover['basis']} directions over "
              f"{cover['length']} places")
        print(f"  inside the span {100 * cover['inside']:.1f}%   "
              f"a basis this size captures {100 * cover['null']:.1f}% of "
              f"anything   excess {100 * cover['excess']:+.1f}%")
        print(f"  OUTSIDE the key {100 * cover['outside']:.1f}% - the honest "
              f"upper bound on what no amount of further fitting recovers")
        results["completeness"] = {
            "inside": cover["inside"], "outside": cover["outside"],
            "excess": cover["excess"], "basis": cover["basis"]}
    else:
        print("  too few shared places to test")

    print()
    print("THE ELLIPSE, PER DIMENSION")
    ellipses = {}
    for (axis, who), matrix in built.items():
        fit = report_ellipse(matrix, f"{axis:>9} / {who}")
        if fit:
            ellipses[f"{axis}/{who}"] = {
                name: float(fit[name]) for name in
                ("rank", "significant", "asymmetry", "sphere_floor", "sigma",
                 "information", "resolved", "resolved_fraction")}
    results["ellipse"] = ellipses

    print()
    print("THE LOOP, TO THE FLOOR, PER DIMENSION")
    loops = {}
    for (axis, who), matrix in built.items():
        out = ie.differentiate_to_floor(matrix, axis, maximum_passes=8)
        first = out["trace"][0]["asymmetry"] if out["trace"] else float("nan")
        last = out["trace"][-1]["asymmetry"] if out["trace"] else float("nan")
        print(f"  {axis:>9} / {who}: {first:.4f} -> {last:.4f}   "
              f"{len(out['targets'])} direction(s) in {out['passes']} pass(es)"
              f"   {out['reason']}")
        loops[f"{axis}/{who}"] = {
            "passes": out["passes"], "reason": out["reason"],
            "directions": len(out["targets"]), "at_floor": out["at_floor"],
            "asymmetry_first": first, "asymmetry_last": last}
    results["loop"] = loops

    reached = sum(1 for v in loops.values() if v["at_floor"])
    print()
    print(f"REACHED THE FLOOR IN {reached} OF {len(loops)} ENSEMBLES; "
          f"{sum(v['directions'] for v in loops.values())} directions "
          f"recovered in total")

    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(results, handle, indent=2, default=float)
        print(f"\nnumbers written to {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
