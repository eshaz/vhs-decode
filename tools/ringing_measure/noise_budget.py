"""Print the noise budget for a capture: the three predicted floors, their
sum, the measured floor, and the gap between them.

    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/noise_budget.py \\
        --capture bars_sp=/testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \\
        --capture cd=/testdata/countdown.flac --capture home=/testdata/home.flac \\
        --reference bars_rec=/testdata/test_patterns/vhs/record/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-rec.flac \\
        --seconds 0.25 --positions 60,1800,3600,5400 --json /path/to/noise_budget.json

One readable tool, arguments on the command line, nothing in the
environment. A capture is `name=path`; `--positions` are seconds into the
capture (the two-hour captures are read as short windows at several
positions, never decoded long; a short capture uses the positions that fit
inside it, or two of its own if none do). The FLAC header of these
captures states its rate in kilohertz (40000 means 40 MSps), and the rate
is then MEASURED from the format's line period and reported beside it.

A `--reference` is a CONTROL: a capture measured with the same instrument
and printed beside the budgeted ones, never folded into a budget. Two
controls earn their place here.

  The RECORD SIDE of the same recording, at CN261 pin 1. It carries the
  modulator and the record amplifier but no tape and no head amplifier, so
  it separates the recording chain and the converter from everything the
  tape and the playback head add. Measured 2026-09-06 it reads 109.2 dB-Hz
  against 93.2 to 95.6 for the playback captures, and its floor is flat
  across the tone's offsets to 1.7 dB where a playback floor rises 4.7 to
  9.6 dB toward the tone.

  The SAME RECORDING AT ANOTHER TAPE SPEED, which tests the particulate
  term's own scaling law - the only test of it the collection allows. NTSC
  VHS EP keeps the drum speed and so the writing speed, the wavelength, the
  read depth, the along-track length and the carriers exactly, and changes
  only the recorded track width, 58 to 19.3 um, so the particle count
  predicts the EP floor worse by 10 log10(58 / 19.3) = 4.78 dB and by
  nothing else. Run the EP playback as a control against the SP playback:

    PYTHONPATH=/workspaces/vhs-decode python3 tools/ringing_measure/noise_budget.py \\
        --capture bars_sp=/testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \\
        --reference bars_ep=/testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-EP-SLV-778HF-50msps-rf-pb.flac \\
        --reference bars_rec_sp=/testdata/test_patterns/vhs/record/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-rec.flac \\
        --reference bars_rec_ep=/testdata/test_patterns/vhs/record/zaroff-75bars-NTSC-EP-SLV-778HF-50msps-rf-rec.flac

  It measured 93.81 (SP) against 80.30 (EP) dB-Hz, a difference of 13.52 dB
  where 4.78 was predicted; the record-side pair of the same two recordings
  differs by 2.85 dB, so net of the recording chain's own EP penalty the
  playback difference is 10.67 dB, still 5.9 dB more than the count allows.
  The BUDGET the tool prints for an EP capture would be wrong - the
  particulate term would use the SP track width, and
  `noise_budget.format_figures` refuses an EP speed because the writing
  speed it needs is not in the tree for EP - which is why the EP capture
  belongs here, where only its measured floor is used.

What it prints, per capture and per position: the measured rate and the
tip frequency against the format's tolerance, the pulses found, the
carrier amplitude in codes, the word length and the effective bits, and
the floor under the band with its uncertainty; then the budget table for
the capture - each term's C/N0 in dB-Hz and in Carson's band, the sum, the
measured floor, the gap and the verdict - and the same floor under every
definition this arc has used, with the labels that land near 45 dB.
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vhsdecode.models import noise_budget as nb                # noqa: E402
import rf_noise                                                 # noqa: E402  (this directory)


QUOTED_FLOOR_DB = 45.0        # the figure Ethan's directive quotes as measured
QUOTED_TOLERANCE_DB = 1.5     # how near a definition must land to be a candidate


def header_rate_hz(info):
    """The FLAC header states kilohertz for these captures."""
    rate = float(info.samplerate)
    return rate * 1000.0 if rate < 1e6 else rate


def duration_s(info):
    """The header's frame count, when it is real: the two-hour captures
    carry a placeholder (the largest 64-bit integer)."""
    if info.frames >= 2 ** 62:
        return None
    return info.frames / header_rate_hz(info)


def positions_for(path, requested, seconds):
    duration = duration_s(sf.info(path))
    if duration is None:
        return list(requested)
    fitting = [p for p in requested if p + seconds <= duration]
    if fitting:
        return fitting
    # a short capture: one window near its start and one at its middle
    return sorted({min(0.1 * duration, max(duration - seconds, 0.0)),
                   min(0.5 * duration, max(duration - seconds, 0.0))})


def flac_metadata(path):
    """The FLAC metadata blocks: the declared length and whether a seek
    table is present. Returns None for anything that is not a FLAC file,
    which is then left to libsndfile.

    A FLAC stream's STREAMINFO may declare total_samples = 0, meaning "not
    known", and a stream written without a SEEKTABLE gives a decoder
    nothing to seek by. Both together mean random access rests on an
    estimated byte offset, and `read_window` does not trust one."""
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"fLaC":
                return None
            found = {"seek_table": False, "total_samples": 0, "blocks": []}
            while True:
                header = handle.read(4)
                if len(header) < 4:
                    return found
                last, kind = header[0] >> 7, header[0] & 0x7F
                size = int.from_bytes(header[1:4], "big")
                data = handle.read(size)
                found["blocks"].append(kind)
                if kind == 0 and len(data) >= 18:      # STREAMINFO
                    found["total_samples"] = (
                        ((data[13] & 0x0F) << 32) | int.from_bytes(data[14:18], "big"))
                if kind == 3:                          # SEEKTABLE
                    found["seek_table"] = True
                if last:
                    return found
    except OSError:
        return None


def libsndfile_can_seek(path):
    """Whether libsndfile's own seek can be trusted on this file.

    MEASURED, not assumed (2026-09-05). The two short zaroff captures each
    carry a SEEKTABLE and a real total_samples (24000512), and libsndfile
    seeks them instantly. Neither two-hour capture carries either
    (STREAMINFO: rate 40000, 1 channel, 8 bits, total_samples 0, fixed
    block 40000; then VORBIS_COMMENT and PADDING, and no seek table), and
    libsndfile's behaviour on them is not merely slow but inconsistent: on
    /testdata/countdown.flac a seek to 1800 s returns in 0.2 s and lands
    correctly, while on /testdata/home.flac every seek scans forward and
    then fails with "Internal psf_fseek() failed" - 0.3 s at 1 s in, 31 s
    at 60 s, 192 s at 300 s, the time growing with the position and the
    seek never succeeding. One file working is luck in an estimated byte
    offset, not a guarantee, so the container's own declaration decides."""
    found = flac_metadata(path)
    if found is None:
        return True
    return bool(found["seek_table"]) or found["total_samples"] > 0


def read_window_with_ffmpeg(path, seconds, position_s, rate_hz, bits, step):
    """The same window through ffmpeg, for a container libsndfile cannot
    seek in (see `libsndfile_can_seek`). ffmpeg resynchronises on the FLAC
    frame headers, which carry the frame number, and seeks either two-hour
    capture in well under a second.

    IT IS THE SAME DATA, checked rather than assumed: on countdown.flac,
    where libsndfile's seek does work, the 400000 samples this path returns
    at 60 s and at 1800 s are bit-for-bit equal to the ones libsndfile
    returns (2026-09-05).

    ffmpeg's -ss is in the container's declared seconds, and the header
    declares 40000 where the capture runs at 40 MSps, so a position in
    seconds of tape is that many thousands of container seconds - the same
    factor `header_rate_hz` applies. The output is asked for in the
    capture's own word length so no requantisation happens on the way, and
    it is scaled back into the int16 container units the rest of the tool
    works in."""
    import subprocess

    declared_rate = sf.info(path).samplerate
    codec = {8: ("s8", np.int8), 16: ("s16le", np.int16)}.get(int(bits))
    if codec is None:
        return None
    name, dtype = codec
    command = ["ffmpeg", "-v", "error",
               "-ss", repr(float(position_s) * rate_hz / declared_rate),
               "-i", str(path),
               "-t", repr(float(seconds) * rate_hz / declared_rate),
               "-f", name, "-c:a", "pcm_" + name, "-"]
    try:
        raw = subprocess.run(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if not raw:
        return None
    return np.frombuffer(raw, dtype=dtype).astype(np.int16) * int(step)


def read_window(path, seconds, position_s):
    """A window of raw samples in their container units (int16, so an
    8-bit capture has a code step of 256), and the header's rate.

    libsndfile where the container declares something to seek by, ffmpeg
    where it does not, and ffmpeg again if libsndfile's seek fails anyway.
    The two return the same samples (see `read_window_with_ffmpeg`)."""
    info = sf.info(path)
    rate_hz = header_rate_hz(info)
    frames = int(round(seconds * rate_hz))
    # soundfile reports the container's word length and its int16 view
    # scales an 8-bit capture by 256; ffmpeg is asked for the same
    bits = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16}.get(info.subtype, 16)
    step = 2 ** (16 - bits)
    data = None
    if position_s <= 0 or libsndfile_can_seek(path):
        try:
            with sf.SoundFile(path) as handle:
                handle.seek(int(round(position_s * rate_hz)))
                data = handle.read(frames, dtype="int16", always_2d=False)
        except RuntimeError:
            data = None
    if data is None or data.size < frames // 2:
        data = read_window_with_ffmpeg(path, seconds, position_s, rate_hz,
                                       bits, step)
    if data is None or data.size < frames // 2:
        return None, rate_hz
    return data, rate_hz


def parse_capture(text):
    if "=" not in text:
        raise SystemExit("a capture is name=path, got %r" % text)
    name, path = text.split("=", 1)
    return name, path


def finite(value):
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


def fmt(value, width=6, decimals=1):
    if not finite(value):
        return "%*s" % (width, "-")
    return "%*.*f" % (width, decimals, value)


def measure_window(path, seconds, position, figures):
    samples, header_rate = read_window(path, seconds, position)
    if samples is None:
        return None
    rate = nb.sample_rate_from_line_period(samples, header_rate, figures)
    floor = nb.measure_floor(samples, rate["sample_rate_hz"], figures)
    floor["rate"] = rate
    floor["position_s"] = position
    floor["header_rate_hz"] = header_rate
    floor["seconds"] = seconds
    # the position and length were computed at the header's rate; at the
    # measured rate they are these
    floor["true_position_s"] = position * header_rate / rate["sample_rate_hz"]
    floor["true_seconds"] = seconds * header_rate / rate["sample_rate_hz"]
    floor["histogram"] = nb.code_histogram(samples, floor["step"]) if floor.get("step") else None
    return floor


def print_window(floor):
    rate = floor["rate"]
    print("  position %7.2f s (%.1f s of tape, %.2f s window at the measured rate): %.3f MSps "
          "measured (header %.0f), line %.2f samples, tip %.3f MHz %s" % (
              floor["position_s"], floor["true_position_s"], floor["true_seconds"],
              rate["sample_rate_hz"] / 1e6, floor["header_rate_hz"] / 1e6,
              rate["line_period_samples"], floor.get("tip_frequency_measured_hz", float("nan")) / 1e6,
              "on spec" if floor.get("tip_within_tolerance") else "OFF SPEC"))
    if "why" in floor:
        print("    %s" % floor["why"])
        return
    print("    broad pulses %d (used %d), line syncs %d; carrier %.1f codes at the tip; "
          "%d bits, %.2f effective (spectrum minimum %.2f MHz, flat from %.2f MHz, "
          "%.1f dB over arithmetic)" % (
              floor["broad_pulses"], floor["in_band_dwells"], floor["line_syncs"],
              floor["carrier_amplitude_tip_codes"], floor["bits"], floor["effective_bits"],
              floor["converter_at_hz"] / 1e6, floor["converter_flat_from_hz"] / 1e6,
              floor["converter_over_arithmetic_db"]))
    low, high = floor["in_band_region_hz"]
    print("    floor under the band (%.2f-%.2f MHz): C/N0 %s dB-Hz +/- %s "
          "(mean spectrum %s; per-pulse mean/median %s dB)" % (
              low / 1e6, high / 1e6, fmt(floor["c_over_n0_db_hz"]),
              fmt(floor["c_over_n0_uncertainty_db"], 4, 2),
              fmt(floor["c_over_n0_from_mean_spectrum_db_hz"]),
              fmt(floor["in_band_mean_over_median_db"], 4, 2)))
    histogram = floor.get("histogram") or {}
    print("    continuum (median over the band's %d bins, %.1f independent) C/N0 %s dB-Hz; total above "
          "continuum %s dB; electronics at the spectrum minimum: C/N0 >= %s dB-Hz; codes %d present "
          "of %d spanned (%d missing), DNL rms %.2f" % (
              floor["in_band_bins"], floor["in_band_independent_bins"],
              fmt(floor["c_over_n0_continuum_db_hz"]), fmt(floor["in_band_mean_over_median_db"], 4, 2),
              fmt(floor["top_of_band_excess_c_over_n0_db_hz"]),
              histogram.get("codes_present", 0), histogram.get("codes_spanned", 0),
              histogram.get("missing_codes_in_span", 0), histogram.get("dnl_rms", float("nan"))))
    rings = "  ".join("%s %s" % (name, fmt(value)) for name, value in floor["rings_c_over_n0_db_hz"].items())
    print("    about the tone: %s; upper sideband %s; converter %s; envelope %s; "
          "kurtosis full-band C/N %s dB (carrier share %.3f)" % (
              rings, fmt(floor["upper_sideband_c_over_n0_db_hz"]),
              fmt(floor["converter_c_over_n0_db_hz"]), fmt(floor["envelope_c_over_n0_db_hz"]),
              fmt(floor["kurtosis_c_over_n_full_band_db"]), floor["kurtosis_carrier_share"]))


def summarise(windows):
    """The median across positions, and the spread between them."""
    good = [w for w in windows if w is not None and "why" not in w and finite(w["c_over_n0_db_hz"])]
    if not good:
        return None
    values = np.array([w["c_over_n0_db_hz"] for w in good])
    statistical = np.nanmedian([w["c_over_n0_uncertainty_db"] for w in good])
    spread = float(values.max() - values.min()) if values.size > 1 else 0.0
    # the uncertainty is the larger of the statistical error and half the
    # spread between positions, which is the tape's own variation
    uncertainty = max(float(statistical) if finite(statistical) else 0.0, spread / 2.0)
    return {
        "windows": len(good),
        "c_over_n0_db_hz": float(np.median(values)),
        "c_over_n0_continuum_db_hz": float(np.nanmedian([w["c_over_n0_continuum_db_hz"] for w in good])),
        "top_of_band_excess_c_over_n0_db_hz": float(np.nanmedian(
            [w["top_of_band_excess_c_over_n0_db_hz"] for w in good
             if np.isfinite(w["top_of_band_excess_c_over_n0_db_hz"])] or [float("nan")])),
        "spread_db": spread,
        "uncertainty_db": uncertainty,
        "sample_rate_hz": float(np.median([w["sample_rate_hz"] for w in good])),
        "carrier_amplitude_codes": float(np.median([w["carrier_amplitude_tip_codes"] for w in good])),
        "carrier_power": float(np.median([w["carrier_power_tip"] for w in good])),
        "bits": float(good[0]["bits"]),
        "step": float(good[0]["step"]),
        "effective_bits": float(np.median([w["effective_bits"] for w in good])),
        "converter_c_over_n0_db_hz": float(np.median([w["converter_c_over_n0_db_hz"] for w in good])),
        "upper_sideband_c_over_n0_db_hz": float(np.nanmedian(
            [w["upper_sideband_c_over_n0_db_hz"] for w in good])),
        "envelope_c_over_n0_db_hz": float(np.nanmedian([w["envelope_c_over_n0_db_hz"] for w in good])),
        "kurtosis_c_over_n_full_band_db": float(np.nanmedian(
            [w["kurtosis_c_over_n_full_band_db"] for w in good])),
        "tip_frequency_hz": float(np.median([w["tip_frequency_measured_hz"] for w in good])),
        "rings_c_over_n0_db_hz": {name: float(np.nanmedian([w["rings_c_over_n0_db_hz"][name] for w in good]))
                                  for name in good[0]["rings_c_over_n0_db_hz"]},
    }


def print_budget(name, summary, figures):
    result = nb.budget(figures, summary["sample_rate_hz"], summary["carrier_power"],
                       summary["bits"], summary["step"], effective=summary["effective_bits"],
                       measured_c_over_n0_db_hz=summary["c_over_n0_db_hz"],
                       measured_uncertainty_db=summary["uncertainty_db"])
    band_db = 10 * np.log10(result["band_hz"])
    thermal, quant, part = result["thermal"], result["quantisation"], result["particulate"]
    print("  budget for %s (carrier %.1f codes, %.2f effective bits, %.2f MSps):" % (
        name, summary["carrier_amplitude_codes"], summary["effective_bits"],
        summary["sample_rate_hz"] / 1e6))
    print("    %-14s %10s %13s   %s" % ("term", "C/N0 dB-Hz", "Carson %.0f MHz" % (result["band_hz"] / 1e6), "basis"))
    print("    %-14s %10s %13s   %s" % (
        "thermal", fmt(thermal["c_over_n0_db_hz"]), fmt(thermal["c_over_n_band_db"]),
        "ASSUMED %.2f mV rms, %.0f ohm, NF %.0f dB, %.0f K; range %.1f to %.1f dB-Hz over the head "
        "output, %.1f to %.1f over every assumption" % (
            thermal["carrier_rms_v"] * 1e3, thermal["source_resistance_ohm"], thermal["noise_figure_db"],
            thermal["temperature_k"], *thermal["c_over_n0_range_db_hz"],
            *thermal["c_over_n0_full_range_db_hz"])))
    print("    %-14s %10s %13s   %s" % (
        "quantisation", fmt(quant["c_over_n0_db_hz"]), fmt(quant["c_over_n_band_db"]),
        "measured lattice %d bits, %.2f effective from the spectrum's minimum" % (
            quant["bits"], quant["effective_bits"])))
    print("    %-14s %10s %13s   %s" % (
        "particulate", fmt(part["c_over_n0_db_hz"]), fmt(part["c_over_n_band_db"]),
        "%.0f particles in %.1f um^3 (%.0f um x %.3f um x %.2f um at %.1f MHz), 10 log N = %.1f dB "
        "over %.1f MHz; density ASSUMED %.0f per um^3" % (
            part["particles"], part["volume_m3"] * 1e18, figures["track_width_m"] * 1e6,
            part["read_depth_m"] * 1e6, part["along_track_m"] * 1e6, 4.0,
            part["ten_log_n_db"], part["count_band_hz"] / 1e6,
            part["packing"] / part["particle_volume_m3"] * 1e-18)))
    print("    %-14s %10s %13s   shares thermal %.0f%%, quantisation %.0f%%, particulate %.0f%%; "
          "range %.1f to %.1f dB-Hz" % (
              "sum", fmt(result["predicted_c_over_n0_db_hz"]), fmt(result["predicted_carson_db"]),
              100 * result["shares"]["thermal"], 100 * result["shares"]["quantisation"],
              100 * result["shares"]["particulate"], *result["predicted_c_over_n0_range_db_hz"]))
    print("    %-14s %10s %13s   median of %d windows, +/- %.2f dB (spread between positions %.2f dB)" % (
        "MEASURED", fmt(result["measured_c_over_n0_db_hz"]), fmt(result["measured_carson_db"]),
        summary["windows"], summary["uncertainty_db"], summary["spread_db"]))
    print("    %-14s %10s %13s   %s" % (
        "gap", fmt(result["gap_db"], 6, 2), "",
        "sum minus measured; %.0f%% of the measured noise is outside the three terms; "
        "over the thermal range %.1f to %.1f dB" % (
            100 * result["unexplained_share"], *result["gap_range_db"])))
    continuum_gap = result["predicted_c_over_n0_db_hz"] - summary["c_over_n0_continuum_db_hz"]
    rings = "; ".join("%s from the tone %s" % (name, fmt(value))
                      for name, value in summary["rings_c_over_n0_db_hz"].items())
    print("    the gap against the in-band CONTINUUM (median over bins, blind to lines and to a skirt) "
          "is %+.2f dB; the floor by offset: %s dB-Hz - a floor that rises toward the tone rides the "
          "carrier and is not additive noise, so none of the three terms holds it" % (
              continuum_gap, rings))
    print("    the electronics at the spectrum's minimum stand at C/N0 >= %s dB-Hz (the converter's "
          "measured floor less its arithmetic); the thermal term assumes %s - a bound only, since "
          "the head amplifier's noise gain at the top of the band is unknown" % (
              fmt(summary["top_of_band_excess_c_over_n0_db_hz"]), fmt(thermal["c_over_n0_db_hz"])))
    print("    verdict: %s" % result["verdict"])
    print("    other readings of the same window: upper sideband %s, envelope (upper bound) %s, "
          "converter alone %s, kurtosis full band %s dB" % (
              fmt(summary["upper_sideband_c_over_n0_db_hz"]), fmt(summary["envelope_c_over_n0_db_hz"]),
              fmt(summary["converter_c_over_n0_db_hz"]), fmt(summary["kurtosis_c_over_n_full_band_db"])))
    return result


def print_definitions(summary, figures, bin_hz):
    defined = nb.definitions(summary["c_over_n0_db_hz"], figures, summary["sample_rate_hz"],
                             bin_hz=bin_hz, target_db=QUOTED_FLOOR_DB)
    converter = nb.definitions(summary["converter_c_over_n0_db_hz"], figures,
                               summary["sample_rate_hz"], bin_hz=bin_hz)
    print("  the measured floor under every definition (tape) and the converter's own floor:")
    for key in ("per_hz_db_hz", "carson_band_db", "deviation_band_db", "nyquist_band_db",
                "per_bin_db", "video_unweighted_no_deemphasis_db", "video_deemphasised_db",
                "video_deemphasised_weighted_db"):
        if key in defined:
            print("    %-38s tape %s   converter %s" % (key, fmt(defined[key]), fmt(converter[key])))
    print("    a per-bin reading of the tape floor shows %.0f dB at a bin width of %.0f kHz" % (
        QUOTED_FLOOR_DB, defined["bin_width_reading_target_hz"] / 1e3))
    near_tape = nb.labels_near(defined, QUOTED_FLOOR_DB, QUOTED_TOLERANCE_DB)
    near_converter = nb.labels_near(converter, QUOTED_FLOOR_DB, QUOTED_TOLERANCE_DB)
    print("    within %.1f dB of the quoted %.0f dB: tape %s; converter %s" % (
        QUOTED_TOLERANCE_DB, QUOTED_FLOOR_DB, near_tape or "none", near_converter or "none"))
    return {"tape": defined, "converter": converter,
            "near_quoted": {"tape": near_tape, "converter": near_converter}}


def strip_arrays(record):
    if isinstance(record, dict):
        return {k: strip_arrays(v) for k, v in record.items() if not isinstance(v, np.ndarray)}
    if isinstance(record, (list, tuple)):
        return [strip_arrays(v) for v in record]
    if isinstance(record, (np.floating, np.integer)):
        return record.item()
    if isinstance(record, np.bool_):
        return bool(record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="append", required=True,
                        help="name=path of a playback RF capture; repeatable")
    parser.add_argument("--reference", action="append", default=[],
                        help="name=path of a control: measured with the same instrument and "
                             "printed against the budgeted captures, never budgeted itself. "
                             "The record side of the same recording, or the same recording at "
                             "another tape speed; repeatable")
    parser.add_argument("--seconds", type=float, default=0.25,
                        help="window length in seconds at each position")
    parser.add_argument("--positions", default="60,1800,3600,5400",
                        help="seconds into the capture, comma separated")
    parser.add_argument("--json", default=None, help="write every number here")
    args = parser.parse_args(argv)
    positions = [float(p) for p in args.positions.split(",") if p.strip()]
    figures = nb.format_figures()
    everything = {"figures": figures, "captures": {}, "references": {}}

    print("noise budget - NTSC VHS SP: tip %.2f MHz, white %.2f MHz, Carson %.0f MHz, "
          "writing speed %.2f m/s, track %.0f um" % (
              figures["tip_hz"] / 1e6, figures["white_hz"] / 1e6, figures["carson_hz"] / 1e6,
              figures["writing_speed_m_s"], figures["track_width_m"] * 1e6))
    factors = nb.weighting_noise_factors()
    print("weighting network check against BT.1439: flat %.2f dB (published %.1f), "
          "triangular %.2f dB (published %.1f)" % (
              factors["flat_db"], factors["published_flat_db"],
              factors["triangular_db"], factors["published_triangular_db"]))
    print("statistics shared with rf_noise.py: chance bound at 90 averages %.4f (rf_noise %.4f), "
          "median over mean %.4f (rf_noise %.4f), Boltzmann %s" % (
              nb._chance_bound(90, 0.95), rf_noise.spectrum_confidence(90),
              nb.chi_square_median_factor(1), rf_noise.MEDIAN_OVER_MEAN,
              "same" if nb.BOLTZMANN_J_PER_K == rf_noise.BOLTZMANN else "DIFFERENT"))

    summaries = {}
    for text in args.capture:
        name, path = parse_capture(text)
        print("\ncapture %s: %s" % (name, path))
        windows = []
        for position in positions_for(path, positions, args.seconds):
            floor = measure_window(path, args.seconds, position, figures)
            if floor is None:
                print("  position %.0f s: not readable (past the end, or seek failed)" % position)
                continue
            print_window(floor)
            windows.append(floor)
        summary = summarise(windows)
        record = {"windows": [strip_arrays(w) for w in windows], "summary": summary}
        if summary is None:
            print("  no usable window")
        else:
            record["budget"] = strip_arrays(print_budget(name, summary, figures))
            record["definitions"] = print_definitions(summary, figures, windows[0]["spectrum_bin_hz"])
            summaries[name] = summary
        everything["captures"][name] = record

    for text in args.reference:
        name, path = parse_capture(text)
        print("\ncontrol (same instrument, no budget) %s: %s" % (name, path))
        windows = []
        for position in positions_for(path, positions, args.seconds):
            floor = measure_window(path, args.seconds, position, figures)
            if floor is None:
                print("  position %.0f s: not readable" % position)
                continue
            print_window(floor)
            windows.append(floor)
        summary = summarise(windows)
        if summary is not None:
            print("  this control's floor: C/N0 %s dB-Hz (%s in Carson's band); converter %s; "
                  "against the budgeted captures: %s" % (
                      fmt(summary["c_over_n0_db_hz"]),
                      fmt(summary["c_over_n0_db_hz"] - 10 * np.log10(figures["carson_hz"])),
                      fmt(summary["converter_c_over_n0_db_hz"]),
                      ", ".join("%s %+.1f dB" % (other, summary["c_over_n0_db_hz"] - s["c_over_n0_db_hz"])
                                for other, s in summaries.items())))
            rings = "  ".join("%s %s" % (key, fmt(value))
                              for key, value in summary["rings_c_over_n0_db_hz"].items())
            print("  its floor by offset from the tone: %s dB-Hz (spread %.1f dB) - a control that "
                  "is flat here has no carrier-riding noise of its own" % (
                      rings, float(np.nanmax(list(summary["rings_c_over_n0_db_hz"].values()))
                                   - np.nanmin(list(summary["rings_c_over_n0_db_hz"].values())))))
        everything["references"][name] = {"windows": [strip_arrays(w) for w in windows],
                                          "summary": summary}

    if summaries:
        print("\nsummary (C/N0 in dB-Hz; Carson's band %.0f MHz)" % (figures["carson_hz"] / 1e6))
        print("  %-10s %8s %6s %8s %8s %8s %8s %8s %7s %9s" % (
            "capture", "MSps", "ENOB", "carrier", "measured", "contin", "Carson", "sum", "gap", "weighted"))
        for name, summary in summaries.items():
            result = nb.budget(figures, summary["sample_rate_hz"], summary["carrier_power"],
                               summary["bits"], summary["step"], effective=summary["effective_bits"],
                               measured_c_over_n0_db_hz=summary["c_over_n0_db_hz"],
                               measured_uncertainty_db=summary["uncertainty_db"])
            weighted = nb.video_signal_to_noise(summary["c_over_n0_db_hz"], figures, weighting=True)
            print("  %-10s %8.2f %6.2f %8.1f %8.1f %8.1f %8.1f %8.1f %+7.2f %9.1f" % (
                name, summary["sample_rate_hz"] / 1e6, summary["effective_bits"],
                summary["carrier_amplitude_codes"], summary["c_over_n0_db_hz"],
                summary["c_over_n0_continuum_db_hz"],
                result["measured_carson_db"], result["predicted_c_over_n0_db_hz"],
                result["gap_db"], weighted))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(strip_arrays(everything), handle, indent=1)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
