#!/usr/bin/env python3
"""Can the video head see the linear audio or the control track?

    PYTHONPATH=/workspaces/vhs-decode python3 \\
        tools/ringing_measure/edge_tracks_measure.py [--speed SP] [--fields 90]

Ethan's question was whether the edge tracks overlap the video sweep enough
to be read by the video head, and whether the control track's frame pulse
could then tie a linear audio capture to the video. `vhsdecode.models.
edge_tracks` holds the geometry, the prediction and the estimator; this
script runs it over the captures and prints the four things that decide it:

  the PREDICTION      the guard between the sweep and each edge track, the
                      speed ratio, the reproduce frequency and the level a
                      fringe could most generously reach;
  the FREQUENCY sweep the matched filter across a ladder of frequencies at
                      the sweep end, which is what separates a line at the
                      predicted frequency from the tap's own smooth
                      low-frequency noise;
  the POSITION scan   the same filter against position along the sweep,
                      which is what separates an edge effect from anything
                      that lives in the whole field;
  the CONTROLS        the record tap, which carries the drive going TO the
                      head and can hold nothing read off the tape; the
                      mid-field anchor; the non-alternating average; and
                      the tape speed, which moves the predicted frequency
                      from 5183.9 Hz at SP to 15628.5 Hz at EP while
                      leaving every other rate in the deck where it was.

The two-hour consumer captures are read through ffmpeg: home.flac carries
no seek table and no total_samples, and libsndfile's seek on it scans
forward and then fails (see `noise_budget.libsndfile_can_seek`).
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy import signal

from vhsdecode.models import edge_tracks as et
from vhsdecode.models import tap_transfer as tt

PATTERNS = "/testdata/test_patterns/vhs"
CONSUMER = ("/testdata/countdown.flac", "/testdata/home.flac")

# The head switch, MEASURED rather than taken from the standard's 5-to-8
# line window: the FM envelope of `zaroff-75bars-NTSC-SP` playback, averaged
# over 27 fields and normalised to lines -30..-25, dips from line -6.3,
# bottoms at -5.29 at 88.3 per cent, and is back by -3.5.
SWITCH_ONSET_H = -6.3
SWITCH_END_H = -3.5
SPAN_H = 28.0


def read_capture(path, position_s=0.0, seconds=None, rate_hz=50e6):
    """A window of a capture as float in the ADC's own +-1 units."""
    if seconds is None:
        return tt.read_capture(path)
    declared = sf.info(path).samplerate
    command = ["ffmpeg", "-v", "error",
               "-ss", repr(float(position_s) * rate_hz / declared),
               "-i", str(path),
               "-t", repr(float(seconds) * rate_hz / declared),
               "-f", "s8", "-c:a", "pcm_s8", "-"]
    raw = subprocess.run(command, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int8).astype(np.float64) / 128.0


def carrier_rms(x, rate_hz):
    """The reference every level is quoted against: the rms of the luma FM
    band, so a level in dBc means the same thing across captures whose gains
    and ADC ranges differ."""
    sos = signal.butter(4, (2.5e6, 5.5e6), btype="band", fs=rate_hz,
                        output="sos")
    return float(np.std(signal.sosfiltfilt(sos, x)))


def prepared(path, rate_hz, speed, position_s=None, fields=None):
    """One window: the centred capture, its field starts and its carrier rms."""
    params = tt.capture_parameters("NTSC", "VHS", speed)
    if position_s is None:
        x = read_capture(path)
    else:
        x = read_capture(path, position_s, (fields or 90) * 1.06 * 1001 / 60000,
                         rate_hz)
    found = tt.locate_fields(x, rate_hz, params)
    if fields:
        found = found[:fields]
    centred = x - x.mean()
    return centred, [f.start for f in found], carrier_rms(centred, rate_hz)


def pooled(windows, rate_hz, speed, frequencies, track, anchor,
           alternate=True, span=SPAN_H):
    """The matched filter over several windows at once, pooling the
    per-field projections rather than the per-window means - the fields are
    the repeats and their own scatter is the error bar."""
    low, high = ((anchor, anchor + span) if track == "control"
                 else (anchor - span, anchor))
    params = tt.capture_parameters("NTSC", "VHS", speed)
    line = params["line_period_s"] * rate_hz
    values, scales, norms, lengths = {}, [], {}, {}
    for frequency in frequencies:
        reference = et.matched_reference(rate_hz, low, high, anchor,
                                         float(frequency), track, speed=speed)
        norms[float(frequency)] = float(np.linalg.norm(reference))
        lengths[float(frequency)] = len(reference)
        start = int(low * line)
        for capture, starts, rms in windows:
            scales.append(rms)
            for index, begin in enumerate(starts):
                at = int(begin) + start
                if at < 0 or at + len(reference) > len(capture):
                    continue
                sign = (-1.0) ** index if alternate else 1.0
                values.setdefault(float(frequency), []).append(
                    sign * np.vdot(reference, capture[at:at + len(reference)]))
    rms = float(np.mean(scales))
    out = {}
    for frequency, sequence in values.items():
        array = np.asarray(sequence)
        error = np.hypot(array.real.std(ddof=1),
                         array.imag.std(ddof=1)) / np.sqrt(len(array))
        found = et.Projection(amplitude=complex(array.mean()),
                              standard_error=float(error), fields=len(array),
                              reference_rms=rms,
                              window_samples=lengths[frequency],
                              template_norm=norms[frequency])
        out[frequency] = {"dbc": found.dbc, "error_dbc": found.error_dbc,
                          "bound_dbc": found.bound_dbc,
                          "ratio": found.ratio, "fields": found.fields}
    return out


def position_scan(windows, rate_hz, speed, frequency, centres,
                  probe_lines=4.0, alternate=True):
    """The same level against POSITION along the sweep, with a short probe
    of fixed length so the only thing changing is where it looks.

    This is the control that decided the question. A control-track fringe
    must be LARGEST at the tape edge and fall inward at the fringing rate;
    anything that peaks in the middle of the picture is not one, whatever
    frequency it sits at.
    """
    params = tt.capture_parameters("NTSC", "VHS", speed)
    line = params["line_period_s"] * rate_hz
    n = int(probe_lines * line)
    window = np.hanning(n)
    out = {}
    for centre in centres:
        start = int((centre - probe_lines / 2.0) * line)
        template = window * np.exp(2j * np.pi * frequency
                                   * (np.arange(n) + start) / rate_hz)
        # orthogonal to a constant, for the reason in
        # `edge_tracks.matched_reference`: a four-line Hanning probe at
        # 5183.9 Hz still passes DC at a tenth of its peak gain
        template = template - template.mean()
        values, scales = [], []
        for capture, starts, rms in windows:
            scales.append(rms)
            for index, begin in enumerate(starts):
                at = int(begin) + start
                if at < 0 or at + n > len(capture):
                    continue
                sign = (-1.0) ** index if alternate else 1.0
                values.append(sign * np.vdot(template, capture[at:at + n]))
        array = np.asarray(values)
        mean = array.mean()
        error = np.hypot(array.real.std(ddof=1),
                         array.imag.std(ddof=1)) / np.sqrt(len(array))
        found = et.Projection(amplitude=complex(mean), standard_error=float(error),
                              fields=len(array),
                              reference_rms=float(np.mean(scales)),
                              window_samples=n,
                              template_norm=float(np.linalg.norm(template)))
        out[centre] = {"dbc": found.dbc, "ratio": found.ratio,
                       "bound_dbc": found.bound_dbc, "fields": found.fields}
    return out


def show(title, rows, marks=()):
    print("  " + title)
    if not rows:
        print("    (nothing)")
        return
    base = float(np.median([r["dbc"] for r in rows.values()]))
    for frequency in sorted(rows):
        row = rows[frequency]
        tag = "".join("  <-- %s" % name for hz, name in marks
                      if abs(frequency - hz) < 1.0)
        bar = "#" * max(0, int(row["dbc"] - base + 25))
        print("    %9.1f Hz  %7.1f dBc  ratio %6.2f  bound %7.1f  n %5d  %s%s"
              % (frequency, row["dbc"], row["ratio"], row["bound_dbc"],
                 row["fields"], bar, tag))
    peak = max(rows, key=lambda f: rows[f]["dbc"])
    print("    baseline %.1f dBc; peak %.1f Hz at %.1f dBc (%+.1f dB over it)"
          % (base, peak, rows[peak]["dbc"], rows[peak]["dbc"] - base))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--speed", default="SP", choices=("SP", "EP"))
    parser.add_argument("--fields", type=int, default=90,
                        help="fields per window of a consumer capture")
    parser.add_argument("--positions", type=float, nargs="*",
                        default=[2.0, 400.0],
                        help="seconds into each consumer capture")
    parser.add_argument("--captures", default="all",
                        choices=("all", "patterns", "record", "consumer"),
                        help="which captures to run")
    options = parser.parse_args(argv)
    speed = options.speed

    place = et.layout(speed)
    bound = et.geometric_upper_bound_db(speed)
    print("PREDICTION, from SMPTE 32M table 2 and table 3 alone")
    print("  control track %.3f..%.3f mm, audio track %.3f..%.3f mm, "
          "sweep %.3f..%.3f mm"
          % (place["control_track_m"][0] * 1e3, place["control_track_m"][1] * 1e3,
             place["audio_track_m"][0] * 1e3, place["audio_track_m"][1] * 1e3,
             place["sweep_m"][0] * 1e3, place["sweep_m"][1] * 1e3))
    print("  guard to the control track %.0f um, to the audio track %.0f um; "
          "no overlap at nominal dimensions"
          % (place["guard_control_m"] * 1e6, place["guard_audio_m"] * 1e6))
    print("  speed ratio %.3f, control-track line %.1f Hz, decay length "
          "%.1f um = %.2f lines (%.3f dB/line)"
          % (et.speed_ratio(speed), bound["reproduce_hz"],
             et.fringing_decay_length_m(bound["reproduce_hz"], speed) * 1e6,
             et.fringing_decay_length_m(bound["reproduce_hz"], speed)
             / et.transverse_rate_m_per_line(),
             8.685889638 * et.transverse_rate_m_per_line()
             / et.fringing_decay_length_m(bound["reproduce_hz"], speed)))
    print("  generous upper bound %.1f dBc = %.1f (differentiation) %+.1f (guard)"
          % (bound["total_db"], bound["differentiation_db"], bound["guard_db"]))
    print("  the audio track admits only tape frequencies below %.0f Hz at "
          "-60 dB, arriving at %.1f kHz"
          % (et.admitted_tape_band_hz(speed=speed),
             et.reproduce_hz(et.admitted_tape_band_hz(speed=speed), speed) / 1e3))
    other = "EP" if speed == "SP" else "SP"
    print("  the same signal at %s would arrive at %.1f Hz - the speed "
          "control" % (other, et.control_track_reproduce_hz(other)))

    predicted = bound["reproduce_hz"]
    ladder = sorted(set(list(np.arange(predicted - 1800, predicted + 1801, 200.0))
                        + [predicted, et.control_track_reproduce_hz(other),
                           15734.264, 1000.0, 2000.0, 30000.0]))
    marks = [(predicted, "PREDICTED at this speed"),
             (et.control_track_reproduce_hz(other), "the other speed's line"),
             (15734.264, "line rate")]

    jobs = []
    want = options.captures
    if want in ("all", "patterns"):
        files = sorted(glob.glob(os.path.join(PATTERNS, "playback",
                                              "*-NTSC-%s-*.flac" % speed)))
        if files:
            jobs.append(("zaroff %s playback (%d captures)" % (speed, len(files)),
                         [prepared(p, 50e6, speed) for p in files], 50e6))
    if want in ("all", "record"):
        control = sorted(glob.glob(os.path.join(PATTERNS, "record",
                                                "*-NTSC-%s-*.flac" % speed)))
        if control:
            jobs.append(("zaroff %s RECORD TAP - the control (%d captures)"
                         % (speed, len(control)),
                         [prepared(p, 50e6, speed) for p in control], 50e6))
    if want in ("all", "consumer") and speed == "SP":
        for path in CONSUMER:
            windows = [prepared(path, 40e6, speed, at, options.fields)
                       for at in options.positions]
            jobs.append((os.path.basename(path), windows, 40e6))

    for title, windows, rate in jobs:
        print("\n########## %s ##########" % title)
        for track, anchor in (("audio", SWITCH_ONSET_H),
                              ("control", SWITCH_END_H)):
            show("%s end of the sweep, anchor line %+.1f, alternating"
                 % (track, anchor),
                 pooled(windows, rate, speed, ladder, track, anchor), marks)
        show("mid-field anchor (line 120) - four millimetres from any edge "
             "track", pooled(windows, rate, speed, ladder, "control", 120.0),
             marks)
        show("sweep end, NOT alternating - the parity control",
             pooled(windows, rate, speed, ladder, "audio", SWITCH_ONSET_H,
                    alternate=False), marks)
        print("  POSITION at %.1f Hz, 4-line probe, alternating - a fringe "
              "must be largest at the tape edge" % predicted)
        centres = list(np.arange(-38.0, 24.1, 2.0))
        scan = position_scan(windows, rate, speed, predicted, centres)
        base = float(np.median([r["dbc"] for r in scan.values()]))
        for centre in centres:
            row = scan[centre]
            edge = ("  <-- the switch" if SWITCH_ONSET_H - 1 <= centre
                    <= SWITCH_END_H + 1 else "")
            print("    line %+7.1f  %7.1f dBc  ratio %6.2f  bound %7.1f%s"
                  % (centre, row["dbc"], row["ratio"], row["bound_dbc"], edge))
        peak = max(scan, key=lambda c: scan[c]["dbc"])
        print("    peak at line %+.1f (%.1f dBc), median %.1f dBc"
              % (peak, scan[peak]["dbc"], base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
