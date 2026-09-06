"""Measure the tape's modulation noise on real captures, sync and blanking
only, per head, and print it beside the particulate prediction.

    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/modulation_noise_measure.py \\
        --capture bars_sp=/testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \\
        --capture bars_ep=/testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-EP-SLV-778HF-50msps-rf-pb.flac:ep \\
        --capture cd=/testdata/countdown.flac --capture home=/testdata/home.flac \\
        --seconds 1.0 --positions 60,600,1800,3600,6000 --json <out.json>

One readable tool, arguments on the command line, nothing in the
environment. A capture is `name=path[:speed]`; `--positions` are seconds
into the capture, and every capture is read as SHORT WINDOWS at those
positions, never decoded long (the two-hour captures carry no frame count,
so libsndfile refuses to seek them and `ffmpeg -ss` is used instead, which
counts the header's nominal rate as its seconds and lands sample-exact -
the convention `content_independence.py` verified). The FLAC header states
its rate in kilohertz for these captures (40000 means 40 MSps), which is
how the decoder's `inputfreq` is derived. A window shorter than the
capture is clipped to it.

What it prints, per capture and per head - and PER HEAD ONLY, because the
two heads differ in gain, and a level difference that is a gain is not a
lever arm: pooling them manufactured a 4 sigma slope out of nothing on the
first run:

  - THE FAST BAND, from the regression of cell variance on cell level
    squared over the sync-tip cells: the level-proportional (tape's
    own) power and index with the larger of two error bars, the
    level-independent power and the in-phase carrier-to-noise it
    implies, the fraction at the working level, the lever arm, the
    events excluded, and the bound to quote where the slope is within
    its error;
  - THE SLOW BAND, from the scatter of cell levels about the field-locked
    line curve: the relative rms of modulation slower than a cell, its
    line-to-line correlation and how much of it the tip and porch share;
  - the back porch against the white-additive prediction from the tip
    fit, which it fails by tens of per cent on every tape measured
    (the porch is inside the edge transient `position_split` describes);
  - the predicted modulation index from the particulate model beside the
    measured one;
  - the weight the envelope-derived |H| should carry at 3.4, 4.0 and
    4.4 MHz, from the model, with the additive, fast and slow parts split
    and the channel excess the raw variance holds beyond them. The bins'
    levels and counts come from the whole window, active picture
    included, as an OFFLINE check only; every model parameter comes from
    sync and blanking;
  - with `--spectra`, the fast residual's modulation and additive sideband
    spectra and how each matches the predicted images.

MEASURED 2026-09-05: transcribed into `vhsdecode/models/modulation_noise.py`
(MEASURED), beside the prediction that was written first.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import logging                                                  # noqa: E402
import lddecode.core as ldd                                     # noqa: E402
from vhsdecode.models import head_model                         # noqa: E402
from vhsdecode.models import modulation_noise as mn             # noqa: E402
from vhsdecode.process import VHSRFDecode                       # noqa: E402

ldd.logger = logging.getLogger("modulation_noise_measure")
ldd.logger.setLevel(logging.ERROR)

# The FLAC headers of these captures carry the rate in kilohertz; a header
# below this many hertz is read that way. A capture genuinely sampled below
# one megahertz is not an RF capture.
RATE_IS_KILOHERTZ_BELOW_HZ = 1e6


def capture_rate_hz(path):
    header = float(sf.info(path).samplerate)
    return header * 1000.0 if header < RATE_IS_KILOHERTZ_BELOW_HZ else header


def read_window(path, seconds, position_s):
    """A window of raw samples as float64 (int16 scaling, so an 8-bit
    capture's code is 256), and the capture's rate in hertz. soundfile
    where the header carries a frame count; ffmpeg where it does not."""
    info = sf.info(path)
    rate_hz = capture_rate_hz(path)
    frames = int(round(seconds * rate_hz))
    start = int(round(position_s * rate_hz))
    finite_length = info.frames < (1 << 62)
    if finite_length:
        if start >= info.frames:
            return None, rate_hz
        with sf.SoundFile(path) as handle:
            handle.seek(start)
            data = handle.read(min(frames, info.frames - start), dtype="int16",
                               always_2d=False)
        data = np.asarray(data).ravel()
    else:
        header_rate = float(info.samplerate)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-ss", repr(start / header_rate), "-i", path,
                   "-t", repr(frames / header_rate),
                   "-c:a", "pcm_s16le", "-f", "s16le", "-"]
        raw = subprocess.run(command, capture_output=True, check=True).stdout
        data = np.frombuffer(raw, "<i2")[:frames]
    if data.size < frames // 4:
        return None, rate_hz
    return data.astype(np.float64), rate_hz


def carrier_bins_from_window(samples, rf, bin_hz):
    """The offline check the weights need: the envelope's level, raw
    variance, independent-sample count and line count per carrier bin over
    the whole window, active picture included. NOT the modulation-noise
    estimate; it is the level the channel fit reads at each carrier and how
    many samples and lines it reads it from."""
    channels = mn.decoder_channels(samples, rf)
    envelope, frequency = channels["envelope"], channels["frequency_05"]
    grid, magnitude = mn.decoder_response_grid(rf)
    sp = rf.SysParams
    tip_hz = float(sp["ire0"] + sp["vsync_ire"] * sp["hz_ire"])
    peak_white_hz = float(sp["ire0"] + 100.0 * sp["hz_ire"])
    edges = np.arange(tip_hz - bin_hz, peak_white_hz + 2 * bin_hz, bin_hz)
    index = np.digitize(frequency, edges) - 1
    line_samples = float(sp["line_period"]) * 1e-6 * float(rf.freq_hz)
    result = []
    for b in range(edges.size - 1):
        members = np.flatnonzero(index == b)
        if members.size < 64:
            continue
        centre = 0.5 * (edges[b] + edges[b + 1])
        gain = float(np.interp(centre, grid, magnitude))
        if gain <= 0:
            continue
        values = envelope[members] / gain
        pooled = mn.variance_effective_count(values - values.mean())
        result.append({
            "carrier_hz": centre,
            "level": float(np.mean(values)),
            "mean_square": float(np.mean(values)) ** 2,
            "variance": float(np.var(values)),
            "count": int(values.size),
            "effective_count": float(pooled["mean_count"]),
            "line_count": int(np.unique((members // line_samples).astype(np.int64)).size),
        })
    return result


def describe(fitted, label):
    if not fitted.get("resolved"):
        print("  %-14s NOT RESOLVED: %s" % (label, fitted.get("why")))
        return
    print("  %-14s cells %5d  fields %3d  events out %3d  lever arm %.3f  err corr %+.2f  chi2/dof %.2f  variance count/cell %.1f  atten x%.3f"
          % (label, fitted["cells"], fitted["groups"], fitted.get("excluded_as_events", 0),
             fitted["lever_arm"], fitted["error_correlation"], fitted["reduced_chi_square"],
             fitted["effective_count_median"], fitted["attenuation_correction"]))
    print("      level-proportional power %.3e +- %.1e (fit %.1e, jackknife %.1e) = %+.1f sigma; index %.4f +- %.4f; bound %.3e (index <= %.4f)"
          % (fitted["modulation_power"], fitted["modulation_power_error"],
             fitted["modulation_power_fit_error"], fitted["modulation_power_jackknife_error"],
             fitted["modulation_power_sigma"], fitted["modulation_index"],
             fitted["modulation_index_error"], fitted["modulation_power_bound"],
             np.sqrt(fitted["modulation_power_bound"])))
    print("      level-independent power %.3e +- %.1e = %.1f sigma; relative %.3e; in-phase C/N %.1f dB; working level %.1f"
          % (fitted["additive_power"], fitted["additive_power_error"],
             fitted["additive_power_sigma"], fitted["relative_additive"],
             fitted["in_phase_carrier_to_noise_db"], fitted["working_level"]))
    print("      FAST-BAND MODULATION FRACTION at the working level %.3f +- %.3f"
          % (fitted["modulation_fraction"], fitted["modulation_fraction_error"]))


def scalar_items(mapping):
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in mapping.items()
            if not isinstance(v, (dict, list, np.ndarray, slice))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="append", required=True,
                        help="name=path[:speed], speed sp/lp/ep (default sp)")
    parser.add_argument("--seconds", type=float, default=1.0,
                        help="window length per position in seconds")
    parser.add_argument("--positions", default="0",
                        help="comma-separated seconds into each capture")
    parser.add_argument("--json", default=None, help="write every number here")
    parser.add_argument("--spectra", action="store_true",
                        help="also fit the per-bin sideband spectra (slower)")
    parser.add_argument("--bin_hz", type=float, default=100e3,
                        help="carrier bin width for the offline weight check")
    args = parser.parse_args(argv)

    positions = [float(p) for p in args.positions.split(",") if p.strip()]
    report = {}
    for spec in args.capture:
        name, _, rest = spec.partition("=")
        path, _, speed = rest.partition(":")
        speed = (speed or "sp").lower()
        print("=" * 110)
        print("CAPTURE %s  (%s, %s)" % (name, path, speed.upper()))
        rf = None
        all_cells = []
        active_bins = []
        field_offset = 0
        last = None
        for position in positions:
            samples, rate_hz = read_window(path, args.seconds, position)
            if samples is None:
                print("  position %.0f s: not readable (past the end)" % position)
                continue
            if rf is None:
                rf = VHSRFDecode(inputfreq=rate_hz / 1e6, system="NTSC",
                                 tape_format="VHS", rf_options={"tape_speed": speed})
            got = mn.cells_from_rf(samples, rf, first_field_index=field_offset,
                                   periodogram=args.spectra)
            last = got
            cells = got["cells"]
            fields = got["fields"]
            field_offset += int(fields["start"].size) + 1
            print("  position %6.0f s (%.2f s read): %d fields (parity confident %d, inferred %d), %d cells (tip %d, porch %d), noise runs %d"
                  % (position, samples.size / rate_hz, fields["start"].size,
                     int(np.count_nonzero(fields["confident"])), fields["inferred"],
                     len(cells), sum(1 for c in cells if c["kind"] == "tip"),
                     sum(1 for c in cells if c["kind"] == "porch"), got["pulses"]["noise_runs"]))
            for key in sorted(got["interiors"]):
                interior = got["interiors"][key]
                count = got["effective_counts"].get(key)
                split = got["position_splits"].get(key, {})
                inner_slope = (np.mean(split["slope"][interior]) if split.get("resolved")
                               and interior.stop > interior.start else float("nan"))
                print("      %-12s interior %3d of %3d samples; variance count %s, mean count %s; edge/interior variance %.1f; interior slope %.1e"
                      % (key, interior.stop - interior.start, got["templates"][key].size,
                         "%.1f" % count["effective_count"] if count else "-",
                         "%.1f" % count["mean_count"] if count else "-",
                         (float(np.max(split["profile_relative"])
                                / max(np.median(split["profile_relative"][interior]), 1e-300))
                          if split.get("resolved") and interior.stop > interior.start else float("nan")),
                         inner_slope))
            for c in cells:
                c["position_s"] = position
            all_cells.extend(cells)
            active_bins.extend(carrier_bins_from_window(samples, rf, args.bin_hz))
        if rf is None or not all_cells or last is None:
            print("  nothing measured")
            continue

        tip_hz = float(last["tip_hz"])
        grid, magnitude = mn.decoder_response_grid(rf)
        mechanics = (head_model.mechanics_for("VHS", "NTSC", speed.upper(), rf.DecoderParams)
                     or head_model.mechanics_for("VHS", "NTSC", "SP", rf.DecoderParams))
        speed_m_s = head_model.writing_speed(mechanics)
        track_width_m = float(rf.DecoderParams["video_track_width"]) * 1e-6
        bandwidth_m = float(last["modulation_bandwidth_hz"])
        bandwidth_n = float(last["additive_bandwidth_hz"])
        predicted = mn.predicted_modulation_index(tip_hz, speed_m_s, track_width_m, bandwidth_m)
        print("  band-pass at the tip: modulation bandwidth %.3f MHz, additive %.3f MHz; track width %.1f um, writing speed %.2f m/s"
              % (bandwidth_m / 1e6, bandwidth_n / 1e6, track_width_m * 1e6, speed_m_s))
        print("  PREDICTED (particulate, Poisson, written first): index %.4f (power %.2e) from %.0f particles in the envelope's read volume (%.2f um); spacing sensitivity %.4f per 10 nm; tape-only additive C/N %.1f dB"
              % (predicted["count_index"], predicted["modulation_power"],
                 predicted["particles_in_envelope_band"], predicted["read_length_m"] * 1e6,
                 predicted["spacing_index_per_10nm"], predicted["additive_tape_only_cn_db"]))

        entry = {"predicted": scalar_items(predicted), "heads": {},
                 "bandwidth_modulation_hz": bandwidth_m, "bandwidth_additive_hz": bandwidth_n}
        for head in (0, 1):
            label = "head %d (%s)" % (head, "first field" if head == 0 else "second field")
            tips = [c for c in all_cells if c["kind"] == "tip" and c["head"] == head]
            porch = [c for c in all_cells if c["kind"] == "porch" and c["head"] == head]
            print("  --- %s: %d tip cells, %d porch cells" % (label, len(tips), len(porch)))
            fitted = mn.estimate_with_gate(tips, group_key="field")
            describe(fitted, "fast band, tip")
            head_entry = {"fast": scalar_items(fitted)}
            if fitted.get("resolved") and porch:
                observed = float(np.mean([c["variance"] for c in porch]))
                level_square = float(np.mean([c["mean_square"] for c in porch]))
                b_n = float(np.mean([c["additive_bandwidth_ratio"] for c in porch]))
                b_m = float(np.mean([c["modulation_bandwidth_ratio"] for c in porch]))
                expected = (fitted["additive_power"] * b_n
                            + max(fitted["modulation_power"], 0.0) * b_m * level_square)
                print("      back porch (carrier %.3f MHz, b_n %.3f b_m %.3f): variance %.3e against %.3e from the tip fit through the white-additive image: %+.1f%%"
                      % (np.mean([c["carrier_hz"] for c in porch]) / 1e6, b_n, b_m,
                         observed, expected, 100.0 * (observed / expected - 1.0)))
                head_entry["porch_excess_over_white_additive"] = float(observed / expected - 1.0)
            slow = mn.slow_modulation(tips + porch)
            if slow.get("resolved"):
                print("      SLOW BAND (%s): index %.4f (power %.3e +- %.1e; raw scatter %.3e, fast leakage %.1e); lag-1 line correlation %+.3f; tip-porch correlation %+.3f"
                      % (slow["kind"], slow["slow_index"], slow["slow_power"], slow["slow_power_error"],
                         slow["raw_scatter_power"], slow["fast_leakage_power"],
                         slow["lag_one_line_correlation"], slow["tip_porch_correlation"]))
                head_entry["slow"] = scalar_items(slow)
            else:
                print("      slow band: %s" % slow.get("why"))
            if fitted.get("resolved"):
                total_index = float(np.sqrt(max(fitted["modulation_power"], 0.0)
                                            + (slow.get("slow_power", 0.0) if slow.get("resolved") else 0.0)))
                print("      TOTAL modulation index (fast + slow) %.4f against predicted count term %.4f" % (total_index, predicted["count_index"]))
                head_entry["total_index"] = total_index
            entry["heads"][head] = head_entry

            # the weights at the three carriers, per head, from the model
            if fitted.get("resolved") and active_bins:
                carriers = np.array([b["carrier_hz"] for b in active_bins])
                ratios = mn.bandwidth_ratios(carriers, tip_hz, grid, magnitude)
                weights = mn.weight_for_response(
                    fitted,
                    np.array([b["variance"] for b in active_bins]),
                    np.array([b["mean_square"] for b in active_bins]),
                    np.array([b["effective_count"] for b in active_bins]),
                    modulation_bandwidth_ratio=ratios["modulation"],
                    additive_bandwidth_ratio=ratios["additive"],
                    slow_power=slow.get("slow_power", 0.0) if slow.get("resolved") else 0.0,
                    line_count=np.array([b["line_count"] for b in active_bins]))
                print("      WEIGHTS for the envelope-derived |H| (offline check; variance of the bin's log mean, relative units):")
                head_entry["weights"] = {}
                for target in (3.4e6, 4.0e6, 4.4e6):
                    i = int(np.argmin(np.abs(carriers - target)))
                    print("        %.2f MHz: level %8.1f  samples %7d (indep. %6.0f, lines %5d)  weight %.3f (residual weight %.3f)  additive part %.2e  modulation part %.2e  (%s dominates)  modelled rel var %.2e  observed %.2e  channel excess %.2e"
                          % (carriers[i] / 1e6, active_bins[i]["level"], active_bins[i]["count"],
                             active_bins[i]["effective_count"], active_bins[i]["line_count"],
                             weights["weight"][i], weights["residual_weight"][i],
                             weights["additive_part"][i], weights["modulation_part"][i],
                             "additive" if weights["additive_dominates"][i] else "modulation",
                             weights["modelled_relative_variance"][i],
                             weights["observed_relative_variance"][i], weights["channel_excess"][i]))
                    head_entry["weights"]["%.1f" % (carriers[i] / 1e6)] = {
                        "weight": float(weights["weight"][i]),
                        "residual_weight": float(weights["residual_weight"][i]),
                        "additive_part": float(weights["additive_part"][i]),
                        "modulation_part": float(weights["modulation_part"][i]),
                        "channel_excess": float(weights["channel_excess"][i]),
                        "level": active_bins[i]["level"],
                        "effective_count": active_bins[i]["effective_count"],
                        "line_count": active_bins[i]["line_count"],
                    }

            if args.spectra and tips:
                with_spectra = [c for c in tips if "periodogram" in c]
                if with_spectra:
                    length = min(c["periodogram"].size for c in with_spectra)
                    power = np.array([c["periodogram"][:length] for c in with_spectra])
                    hz = with_spectra[0]["periodogram_hz"][:length]
                    spectra = mn.sideband_spectra(
                        power, np.array([c["mean_square"] for c in with_spectra]),
                        np.ones(len(with_spectra)), [c["field"] for c in with_spectra])
                    shapes = mn.predicted_sideband_shape(hz, tip_hz, grid, magnitude)
                    agree_add = mn.shape_agreement(spectra["additive_spectrum"], shapes["additive"],
                                                   spectra["additive_spectrum_error"])
                    agree_mod = mn.shape_agreement(spectra["modulation_spectrum"], shapes["count"],
                                                   spectra["modulation_spectrum_error"])
                    print("      SIDEBAND SPECTRA (tip): additive vs white band-pass image r=%.3f chi2/dof %.2f; modulation vs count image r=%.3f chi2/dof %.2f (bins of %.2f MHz)"
                          % (agree_add["correlation"], agree_add["reduced_chi_square"],
                             agree_mod["correlation"], agree_mod["reduced_chi_square"], (hz[1] - hz[0]) / 1e6))
                    print("        offset MHz : slope (x1e-4 rel)        intercept (rel)      white image  count image")
                    reference = np.nanmax(spectra["additive_spectrum"][1:4]) if length > 3 else 1.0
                    level_square = float(np.mean([c["mean_square"] for c in with_spectra]))
                    for i in range(0, min(length, 8)):
                        print("        %6.2f     : %+8.3f +- %6.3f   %9.3e +- %8.1e   %.3f   %.3f"
                              % (hz[i] / 1e6, spectra["modulation_spectrum"][i] * 1e4,
                                 spectra["modulation_spectrum_error"][i] * 1e4,
                                 spectra["additive_spectrum"][i] / level_square,
                                 spectra["additive_spectrum_error"][i] / level_square,
                                 shapes["additive"][i], shapes["count"][i]))
                    head_entry["spectra"] = {
                        "hz": hz.tolist(),
                        "modulation": spectra["modulation_spectrum"].tolist(),
                        "modulation_error": spectra["modulation_spectrum_error"].tolist(),
                        "additive": spectra["additive_spectrum"].tolist(),
                        "additive_error": spectra["additive_spectrum_error"].tolist(),
                        "agreement_additive": agree_add, "agreement_modulation": agree_mod,
                    }
        report[name] = entry

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1, default=float)
        print("written", args.json)


if __name__ == "__main__":
    main()
