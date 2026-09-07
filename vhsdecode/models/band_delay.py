"""THE LUMA AND CHROMA BANDS AS ONE CLOCK, read twice.

Ethan, 2026-09-06: *"The difference in time between the luma and chroma
bands are the measurement we can use to observe the delay on each video
head. This applies to playback and recording. Use this to relate on the
time axis for the head measurements."*

WHAT MAKES THIS A CLEAN MEASUREMENT. A VHS track carries two signals at
once: the luminance as frequency modulation between 3.4 and 4.4 MHz
(SMPTE 32M clause 3.9.1.1.4) and the chrominance heterodyned down to 40
times the line rate, 629.371 kHz (clause 3.9.2.1.4). One head wrote both
in one pass and one head reads both back in one pass, so everything that
moves the signal in time as a whole - the drum's phase, the capstan, the
tape's stretch, the capture clock - moves the two bands together and
cancels in their difference. What does not cancel is anything whose delay
depends on frequency, and over a span of eleven to one in frequency that
is the head's own dispersion.

THE TWO CLOCKS ARE ALREADY IN THE SIGNAL, and both are in reserved
intervals rather than in the picture. The luma band carries the sync
pulse, whose trailing edge is a specified event; the chroma band carries
the colour burst, whose start is specified at 5.3 microseconds after the
leading edge of sync (BTS-3 clause 2.4.9, tolerance 4.71 to 5.71). So the
interval between them is a specified constant, and any departure from it
is the differential delay of the two bands.

AND THE DECK WAS TAPPED ON BOTH SIDES, which is what separates recording
from playback. The capture set records the same test signal at CN261 pin
1 while recording and at CN261 pin 2 while playing back, on one Sony
SLV-778HF (`/testdata/test_patterns/vhs/readme.txt`). The record tap
carries the signal on its way TO the head and the playback tap carries it
on the way BACK, so the record-side electronics are common to both and
the difference between the taps is the head's write, the tape, the head's
read and the playback preamp - and nothing else.

MEASURED, on the Sony SLV-778HF capture set, half a second of each
capture at 50 MSps, NTSC SP, about 3600 lines a head:

    sync-to-burst interval, nanoseconds after the sync rise
    signal        tap         head A          head B      A minus B
    75bars        record     2140.49         2139.68     +0.81 +/- 0.63
    75bars        playback   2060.90         2058.40     +2.50 +/- 1.16
    pulseandbar   record     2144.66         2145.19     -0.53 +/- 0.68
    pulseandbar   playback   2067.79         2069.85     -2.06 +/- 1.05

THE RECORD TAP IS THE INSTRUMENT'S OWN NULL, and it passes. The record tap
carries the drive on its way to the head, and the same drive goes to
whichever head is switched in, so the two heads MUST agree there. They do,
at 1.3 and 0.8 standard errors. Any per-head difference appearing at the
record tap would have been the instrument's, and none does.

THE TAP DIFFERENCE IS LARGE AND REPRODUCIBLE. Playback less record:

    75bars       head A   -79.59 +/- 1.07 ns      head B   -81.28 +/- 0.78
    pulseandbar  head A   -76.87 +/- 0.98         head B   -75.34 +/- 0.78
    mean of the four     -78.27 ns, spread 5.94, each 75 to 104 sigma

So the chroma band arrives 78 nanoseconds EARLIER relative to the luma
after the round trip through the head and the tape than it did on the way
in. Two unrelated test signals agree to six nanoseconds.

AND IT IS NOT A SPACING LOSS, which the sign settles rather than the size.
A Wallace separation is minimum phase, so its delay is fixed by its
magnitude, and a positive separation delays the CHROMA more than the luma
and lengthens this interval. Shortening it by 78.27 ns requires a
separation of -0.3909 micron, which does not exist. The size would have
been a near-perfect match - 0.39 micron is an ordinary head clearance -
and quoting it without the sign would have manufactured a result.

THE PER-HEAD DIFFERENCE IS NOT RESOLVED, and the reason is a disagreement
rather than noise. The two playback captures give +2.50 +/- 1.16 and -2.06
+/- 1.05 nanoseconds: the same deck, the same tape, opposite signs. The
head labels are not the explanation - an independent per-head quantity,
the radio-frequency envelope level, was checked and agrees between the two
captures, reading -13.07 and -8.12 per cent with the same head stronger in
both, while the record tap gives +0.031 per cent in both as it should. So
the labelling is consistent and the timing genuinely disagrees. Two
readings that should agree and do not are the honest error bar, and it is
the spread rather than either standard error: the per-head luma-chroma
delay is BOUNDED at about 4.6 nanoseconds and is not measured. That bears
on `depth_split.common_mode_timing`, which reports -2.0 +/- 0.4 ns for the
same quantity from the decoded time-base file; the magnitude agrees but
this measurement cannot reproduce its sign, so the 0.4 is optimistic.

THE Y-ONLY CAPTURES ARE THE CONTROL FOR THE CHROMA READING. Recorded with
no chrominance, they carry 22.6 dB less power in the chroma band at the
record tap and 15.5 dB less at playback, the burst amplitude falls from
0.134 to 0.0013, and the interval's scatter rises from 0.10 to 0.41
microsecond - the estimator correctly fails to lock on a burst that is not
there.

EP DOES NOT LOCK ON EVERY PATTERN and is reported rather than hidden: the
EP 75bars captures return a standard error of 8.5 ns against 0.4 at SP,
while EP pulseandbar at the record tap is clean at 0.32. The long-play
recordings need work this module does not yet do.

WHAT THIS CANNOT SEPARATE, stated before the numbers. Write and read are
the same physical head in the same signal chain and they always co-occur,
so no measurement on one deck can split the write dispersion from the read
dispersion; what the tap difference gives is their sum. Splitting them
needs a tape recorded on one deck and played on another, which the capture
set does not contain.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import (burst_sync_lock, colour_under, hypercomplex,
                              sync_geometry, vhs_specification)

# The field rate is 60/1.001 by definition of the 525-line system, and one
# drum revolution is two fields, so the drum turns at half of it. Both are
# derived here rather than typed.
FIELD_RATE_HZ = 60.0 / 1.001
DRUM_RATE_HZ = FIELD_RATE_HZ / 2.0

# The carrier that separates a sync tip from blanking, taken at the
# midpoint IN IRE rather than in hertz, because SMPTE 32M 3.9.1.1.4 makes
# levels frequencies and the IRE scale is the one the standard states.
SYNC_TIP_IRE = -40.0
BLANKING_IRE = 0.0
THRESHOLD_IRE = 0.5 * (SYNC_TIP_IRE + BLANKING_IRE)


def bands(system: str = "NTSC") -> Dict[str, object]:
    """The two bands one head wrote in one pass, each from its own clause.

    Their ratio is what buys the measurement: a delay that is the same at
    both frequencies cancels, and only a delay that changes across eleven
    to one in frequency survives.
    """
    chroma = colour_under.carrier_hz(system)
    tip = vhs_specification.carrier_hz_for_ire(SYNC_TIP_IRE)
    white = vhs_specification.carrier_hz_for_ire(100.0)
    blanking = vhs_specification.carrier_hz_for_ire(BLANKING_IRE)
    centre = 0.5 * (tip + white)
    return {
        "chroma_hz": chroma,
        "luma_sync_tip_hz": tip,
        "luma_blanking_hz": blanking,
        "luma_peak_white_hz": white,
        "luma_centre_hz": centre,
        "ratio": centre / chroma,
        "threshold_hz": vhs_specification.carrier_hz_for_ire(THRESHOLD_IRE),
        "threshold_ire": THRESHOLD_IRE,
        "cite": ("SMPTE 32M-2004 clause 3.9.1.1.4 for the luma deviation "
                 "and clause 3.9.2.1.4 for the colour-under carrier"),
        "why": ("one head wrote both, so everything common to the two "
                "bands cancels in their difference and only a delay that "
                "varies with frequency survives"),
    }


def specified_interval(system: str = "NTSC") -> Dict[str, float]:
    """Where the burst sits relative to the SYNC PULSE'S TRAILING EDGE.

    The standard measures the burst from the leading edge of sync; the
    trailing edge is what the luma band gives cleanly, because it is the
    rise out of the sync tip and no picture content reaches that far down.
    So the specified interval is the burst's start less the sync pulse's
    own width, and the reading is taken at the burst's centre.
    """
    sync_us = sync_geometry.LINE_SYNC_US["525" if system == "NTSC" else "625"]
    start_us = burst_sync_lock.BURST_START_US - sync_us
    low, high = burst_sync_lock.BURST_START_TOLERANCE_US
    length_us = (burst_sync_lock.BURST_CYCLES
                 / burst_sync_lock.subcarrier_hz(system) * 1e6)
    return {
        "start_us": start_us,
        "length_us": length_us,
        "centre_us": start_us + 0.5 * length_us,
        "tolerance_us": (low - sync_us, high - sync_us),
        "sync_width_us": sync_us,
        "cite": ("BTS-3 clause 2.4.9 for the burst's position and length; "
                 "sync_geometry for the pulse width"),
    }


def luma_frequency(samples, sample_rate_hz: float,
                   band_hz: Tuple[float, float] = (2.6e6, 5.4e6)
                   ) -> np.ndarray:
    """The luma band's instantaneous frequency, in hertz.

    The band limit is not optional: without it the colour-under corrupts
    the reading, which is the same defect this arc found when a burst
    inside the window moved a measured blanking carrier from 3.7236 to
    3.4857 MHz. `hypercomplex.demodulate` supplies the exact derivative,
    verified here against a planted 3.900000 MHz carrier which it returns
    as 3.900000 MHz.
    """
    from scipy import signal as _signal
    values = np.asarray(samples, dtype=np.float64).ravel()
    taps = _signal.firwin(255, list(band_hz), fs=sample_rate_hz,
                          pass_zero=False)
    # `hypercomplex.zero_phase` is the same operator as `filtfilt` in one
    # convolution instead of two passes - an identity, agreeing to 1.3e-15
    # and running 3.9 times faster on a field. See its docstring.
    limited = hypercomplex.zero_phase(values, taps)
    frequency = hypercomplex.demodulate(limited, sample_rate_hz)
    smooth = _signal.firwin(63, 3.0e6, fs=sample_rate_hz)
    return hypercomplex.zero_phase(frequency, smooth)


def chroma_envelope(samples, sample_rate_hz: float,
                    system: str = "NTSC") -> np.ndarray:
    """The chroma band's COMPLEX envelope, at the specified carrier.

    Complex, not a modulus: the burst's phase is the finer of its two
    clocks and discarding it is the defect that made an earlier burst
    instrument rank two of four.
    """
    from scipy import signal as _signal
    values = np.asarray(samples, dtype=np.float64).ravel()
    carrier = colour_under.carrier_hz(system)
    taps = _signal.firwin(255, [0.30e6, 1.05e6], fs=sample_rate_hz,
                          pass_zero=False)
    limited = hypercomplex.zero_phase(values, taps)
    index = np.arange(limited.size)
    mixed = (_signal.hilbert(limited)
             * np.exp(-2j * np.pi * carrier * index / sample_rate_hz))
    low = _signal.firwin(255, 0.40e6, fs=sample_rate_hz)
    return (hypercomplex.zero_phase(mixed.real, low)
            + 1j * hypercomplex.zero_phase(mixed.imag, low))


def sync_rises(frequency_hz, sample_rate_hz: float,
               system: str = "NTSC") -> np.ndarray:
    """Every rise out of the sync tip, located to a fraction of a sample.

    The threshold is the carrier at the midpoint IN IRE between the sync
    tip and blanking, which no picture content reaches, so this finds one
    crossing a line and is not confused by the picture. An earlier version
    put the threshold midway in FREQUENCY between the extremes and found
    461 crossings where 315 lines existed, because content crosses that
    level constantly.
    """
    values = np.asarray(frequency_hz, dtype=np.float64).ravel()
    threshold = vhs_specification.carrier_hz_for_ire(THRESHOLD_IRE)
    period = (sync_geometry.LINE_PERIOD_US["525" if system == "NTSC"
                                           else "625"] * 1e-6 * sample_rate_hz)
    below = values < threshold
    candidates = np.flatnonzero(below[:-1] & (~below[1:]))
    # The band limit and the demodulator both have a transient at each end
    # of the record. Measured on a planted 30-line signal, the first rise
    # came back 68.09 microseconds from the second where every later
    # spacing was 63.54 to 63.58, so one line at each end is discarded.
    guard = int(round(period))
    kept, last = [], -period
    for index in candidates:
        if index < guard or index > values.size - guard:
            continue
        if index - last > 0.75 * period:
            kept.append(int(index))
            last = index
    out = []
    for index in kept:
        before, after = values[index], values[index + 1]
        out.append(index + (threshold - before) / (after - before)
                   if after != before else float(index))
    return np.asarray(out, dtype=np.float64)


def burst_readings(envelope, rises, sample_rate_hz: float,
                   system: str = "NTSC") -> Dict[str, np.ndarray]:
    """Where the burst sits after each sync rise, and how strong it is.

    Two readings, deliberately: the envelope's centroid, which is
    unambiguous but coarse, and the tone's own phase, which is fine but
    ambiguous by one carrier cycle - 1.5889 microseconds at 629.371 kHz.
    The amplitude comes back too, because it is what tells a burst from
    its absence, and the y-only captures use exactly that.
    """
    values = np.asarray(envelope, dtype=np.complex128).ravel()
    where = np.asarray(rises, dtype=np.float64).ravel()
    window = specified_interval(system)
    guard_us = 0.6
    low = int((window["start_us"] - guard_us) * 1e-6 * sample_rate_hz)
    high = int((window["start_us"] + window["length_us"] + guard_us)
               * 1e-6 * sample_rate_hz)
    carrier = colour_under.carrier_hz(system)
    lines, centroid, phase, amplitude = [], [], [], []
    for start in where:
        first, last = int(start) + low, int(start) + high
        if first < 0 or last >= values.size:
            continue
        segment = values[first:last]
        magnitude = np.abs(segment)
        peak = float(magnitude.max()) if magnitude.size else 0.0
        if peak <= 0.0:
            continue
        weight = np.where(magnitude > 0.5 * peak, magnitude, 0.0)
        total = float(weight.sum())
        if total <= 0.0:
            continue
        index = np.arange(segment.size, dtype=np.float64)
        lines.append(float(start))
        centroid.append(first + float((index * weight).sum() / total))
        # the tone's phase, referred to the sync rise so it is a time
        angle = float(np.angle(complex((segment * weight).sum())))
        reference = 2.0 * np.pi * carrier * start / sample_rate_hz
        phase.append(float(np.angle(np.exp(1j * (angle + reference)))))
        amplitude.append(peak)
    return {
        "rise": np.asarray(lines),
        "centroid": np.asarray(centroid),
        "phase_rad": np.asarray(phase),
        "amplitude": np.asarray(amplitude),
        "cycle_s": 1.0 / carrier,
        "window_us": (window["start_us"] - guard_us,
                      window["start_us"] + window["length_us"] + guard_us),
    }


def head_labels(rises, sample_rate_hz: float, system: str = "NTSC"
                ) -> Dict[str, object]:
    """Which head read each line, from the vertical interval's own marker.

    The line locator keeps only crossings more than three quarters of a
    line apart, so the equalizing and serration pulses - which come at
    half-line spacing - are rejected and the vertical interval leaves a
    GAP in the sequence. That gap is once per drum revolution, and one
    drum revolution is two fields and one head cycle, so the gap labels
    the heads without needing the envelope at all.

    An earlier attempt read the heads from the drum term in the radio-
    frequency envelope and failed for a reason worth recording: a filter
    resolving the 29.97 Hz drum at a 50 MHz sample rate needs about 67
    milliseconds of impulse response, which is three and a half million
    taps, and the thousand-tap filter that was written instead is
    effectively an all-pass at that frequency. It labelled one line as the
    first head and 7271 as the second.

    The labels are relative to that marker, so which physical head is
    called first is not determined here. The deck's own switching fixes
    the head at a given point in the frame (`sync_geometry` carries the
    switch's position ahead of the vertical sync), so the assignment is
    consistent between captures of one deck, and that is recorded as an
    assumption rather than a measurement.
    """
    where = np.asarray(rises, dtype=np.float64).ravel()
    if where.size < 8:
        raise ValueError("head labelling needs a run of lines")
    line_us = sync_geometry.LINE_PERIOD_US["525" if system == "NTSC" else "625"]
    spacing = np.diff(where) / sample_rate_hz * 1e6
    gaps = np.flatnonzero(spacing > 1.5 * line_us)
    if gaps.size < 2:
        return {"labels": None, "found": int(gaps.size),
                "why": "no repeated vertical-interval marker in this span"}
    period = int(round(float(np.median(np.diff(gaps)))))
    origin = int(gaps[0]) + 1
    index = np.arange(where.size) - origin
    field = np.floor_divide(index, max(period // 2, 1))
    return {
        "labels": (field % 2).astype(int),
        "marker_at": gaps,
        "rises_per_revolution": period,
        "rises_per_field": period // 2,
        "gap_lines": float(np.median(spacing[gaps]) / line_us),
        "assumption": ("the head at a given point in the frame is fixed by "
                       "the deck's own switching, so the labels are "
                       "consistent between captures of one deck; which "
                       "physical head is first is not measured here"),
    }


def interval(readings: Dict[str, np.ndarray], sample_rate_hz: float,
             system: str = "NTSC") -> Dict[str, object]:
    """The sync-to-burst interval per line, against its specified value.

    The reading kept is the envelope's, because it is unambiguous. The
    burst's phase is returned beside it and is NOT combined, for a reason
    the data settles and which is not the obvious one.

    MEASURED on 3578 lines of the bars playback capture, one colour-under
    cycle being 1588.89 nanoseconds and a uniform distribution giving
    458.67:

        the phase as read                         447.04 ns of scatter
        de-rotated by 90 degrees a line           451.68
        de-rotated by 180 degrees a line          446.38
        taken within each residue class modulo 4  445 to 456

    So it is uniform, and the record-side rotation of SMPTE 32M 3.9.2.1.5
    is not the explanation - removing it changes nothing. The cause is the
    reference: the carrier is anchored to absolute time, and one full cycle
    of accumulated drift over a quarter-second capture needs a line-rate
    error of only 6.4 parts per million, which a video tape recorder
    exceeds by orders of magnitude. The phase is therefore a fine
    instrument for the TIME BASE, read line against neighbouring line where
    the drift is negligible, and not for this interval, which needs an
    origin that survives the whole capture. They are different
    measurements and combining them would mix them.
    """
    rise = np.asarray(readings["rise"], dtype=np.float64)
    centroid = np.asarray(readings["centroid"], dtype=np.float64)
    measured = (centroid - rise) / sample_rate_hz * 1e6
    window = specified_interval(system)
    inside = np.abs(measured - window["centre_us"]) < 1.0
    amplitude = np.asarray(readings["amplitude"], dtype=np.float64)
    if amplitude.size:
        inside &= amplitude > 0.3 * float(np.median(amplitude))
    kept = measured[inside]
    return {
        "interval_us": measured,
        "kept": inside,
        "median_us": float(np.median(kept)) if kept.size else float("nan"),
        "sd_us": float(np.std(kept)) if kept.size else float("nan"),
        "standard_error_us": (float(np.std(kept) / math.sqrt(kept.size))
                              if kept.size else float("nan")),
        "lines": int(kept.size),
        "specified_us": window["centre_us"],
        "departure_ns": ((float(np.median(kept)) - window["centre_us"]) * 1e3
                         if kept.size else float("nan")),
        "phase_rad": np.asarray(readings["phase_rad"], dtype=np.float64),
        "phase_is_ambiguous_by_s": readings["cycle_s"],
        "why": ("the interval is specified, so its departure is the two "
                "bands' differential delay"),
    }


def tap_difference(record: Dict[str, object], playback: Dict[str, object]
                   ) -> Dict[str, object]:
    """Playback tap less record tap: the head, the tape and the preamp.

    The record tap carries the signal on its way to the head and the
    playback tap carries what comes back, so the record-side electronics
    are present in both and cancel. What is left is the head's write, the
    tape, the head's read and the playback preamp.

    IT DOES NOT SPLIT WRITE FROM READ. They are the same head in the same
    chain and always co-occur; the sum is what one deck can give. A tape
    recorded on one deck and played on another would split them, and the
    capture set does not contain one.
    """
    delta_us = playback["median_us"] - record["median_us"]
    error_us = math.hypot(playback["standard_error_us"],
                          record["standard_error_us"])
    return {
        "delay_ns": delta_us * 1e3,
        "error_ns": error_us * 1e3,
        "sigma": abs(delta_us) / error_us if error_us > 0 else float("inf"),
        "record_us": record["median_us"],
        "playback_us": playback["median_us"],
        "contains": ("the head's write, the tape, the head's read and the "
                     "playback preamp"),
        "cancels": ("the record-side electronics, the drum phase, the "
                    "capstan, the tape's stretch and the capture clock"),
        "cannot_split": "write from read, on one deck",
    }


def wallace_comparison(interval_change_ns: float, system: str = "NTSC",
                       writing_speed_m_s: float = 5.8) -> Dict[str, object]:
    """What head-to-tape separation would explain a measured delay, and the
    sign the separation predicts.

    A Wallace spacing loss is minimum phase, so its delay is fixed by its
    magnitude: log|H| = -(d/v)|w| has the partner phi = (2d/(pi v)) w ln|w|
    and hence tau = -(2d/(pi v))(ln w + 1), so the two bands differ by
    -(2d/(pi v)) ln(f_luma/f_chroma). With a POSITIVE separation and the
    luma band above the chroma band that difference is NEGATIVE - the luma
    arrives earlier - which LENGTHENS the sync-to-burst interval. A
    measurement that shortens it cannot be a separation loss whatever its
    size, and reporting the equivalent separation without the sign would
    hide exactly that.
    """
    twin = bands(system)
    ratio = twin["luma_centre_hz"] / twin["chroma_hz"]
    # seconds of (luma delay minus chroma delay) per metre of separation;
    # this agrees with `depth_split.wallace_delay_difference_s` to the digit,
    # both giving -10.01 ns at 0.05 micron
    per_metre = -(2.0 / (math.pi * float(writing_speed_m_s))) * math.log(ratio)
    # THE ARGUMENT IS THE MEASURED QUANTITY, not the derived one. A change
    # in the sync-to-burst interval is the chroma's delay less the luma's,
    # which is the negative of what the formula above returns, and taking
    # the argument the other way round is a sign trap in the caller.
    luma_minus_chroma_s = -float(interval_change_ns) * 1e-9
    separation = luma_minus_chroma_s / per_metre if per_metre != 0 else float("nan")
    return {
        "interval_change_ns": float(interval_change_ns),
        "luma_minus_chroma_ns": luma_minus_chroma_s * 1e9,
        "band_ratio": ratio,
        "seconds_per_metre": per_metre,
        "ns_per_micron": per_metre * 1e-6 * 1e9,
        "equivalent_separation_m": separation,
        "equivalent_separation_um": separation * 1e6,
        "separation_is_physical": bool(separation > 0.0),
        "sign_note": ("a positive separation shortens nothing: it delays "
                      "the chroma more than the luma and LENGTHENS the "
                      "sync-to-burst interval"),
        "why": ("the spacing loss is minimum phase, so its delay is not a "
                "free parameter - the magnitude fixes it"),
    }


def analytics(samples, sample_rate_hz: float, system: str = "NTSC"
              ) -> Dict[str, object]:
    """The two full-field transforms every raw-radio-frequency stage needs.

    Ethan, 2026-09-07: *"I believe the hypercomplex stages and the folding
    can be consolidated through some mathmetical simplification."*

    This is the consolidation, and it is a sharing rather than a
    simplification: the luma band's instantaneous frequency and the chroma
    band's complex envelope are the only two expensive things any of these
    stages computes, both are functions of the same samples and the same
    rate, and two stages were computing both independently.

    MEASURED on one field of 667334 samples at 40 MSps: the band split
    costs 35.3 ms and the analytic signal 108.1 ms, so each duplicate
    caller was paying 143.4 ms a field for an answer it could have been
    handed. Per-stage decode cost, measured by turning one stage on at a
    time against a run with none: `band_delay` +5.0 s and
    `head_switch_pair` +9.3 s over four fields.

    AND THE ANALYTIC SIGNAL IS SLOW FOR A REASON WORTH RECORDING. A field
    at 40 MSps is 667334 samples, which factors as 2 x 333667 with 333667
    PRIME, so every transform at that length falls back to Bluestein's
    algorithm. Padding to the next fast length, 668250, makes the analytic
    signal 6.7 times faster and the real transform 19.6 times. It is not
    done here: the transform is circular either way, and measured against
    the unpadded answer a zero pad costs 0.30 of the output floor in the
    interior but 1815 floors at the very edges, where the head switch and
    the vertical interval live. Sharing one exact transform is the honest
    saving; padding trades accuracy where this arc reads its most delicate
    events.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    return {
        "frequency": luma_frequency(values, sample_rate_hz),
        "envelope": chroma_envelope(values, sample_rate_hz, system),
        "sample_rate_hz": float(sample_rate_hz),
        "system": system,
        "samples": int(values.size),
    }


def measure_capture(samples, sample_rate_hz: float, system: str = "NTSC",
                    shared: Optional[Dict[str, object]] = None
                    ) -> Dict[str, object]:
    """The whole chain on one capture: bands, lines, burst, heads, interval.

    Returns the interval pooled and per head. The per-head split at the
    RECORD tap is the measurement's own null: the record tap carries the
    drive on its way to the head, and the same drive goes to whichever
    head is switched in, so the two heads must agree there. Any per-head
    difference that appears at the record tap is the instrument's, not the
    deck's.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    # THE TWO FULL-FIELD TRANSFORMS ARE THE COST, so a caller that already
    # has them may hand them in. `head_switch_pair.measure_capture` reads
    # the same samples of the same field and computed the same pair
    # independently; measured on one field of 667334 samples at 40 MSps the
    # band split costs 35.3 ms and the analytic signal 108.1 ms, so sharing
    # them removes 143.4 ms per field per duplicate caller. Nothing about
    # the answer changes - `shared` carries the SAME arrays, not an
    # approximation of them.
    if shared is None:
        shared = analytics(values, sample_rate_hz, system)
    frequency = shared["frequency"]
    envelope = shared["envelope"]
    rises = sync_rises(frequency, sample_rate_hz, system)
    readings = burst_readings(envelope, rises, sample_rate_hz, system)
    pooled = interval(readings, sample_rate_hz, system)
    heads = head_labels(readings["rise"], sample_rate_hz, system)
    per_head = {}
    labels = heads.get("labels")
    if labels is not None and labels.size == readings["rise"].size:
        for which in (0, 1):
            chosen = pooled["kept"] & (labels == which)
            values_us = pooled["interval_us"][chosen]
            per_head[which] = {
                "median_us": float(np.median(values_us)) if values_us.size
                else float("nan"),
                "standard_error_us": (
                    float(np.std(values_us) / math.sqrt(values_us.size))
                    if values_us.size else float("nan")),
                "lines": int(values_us.size),
            }
    return {
        "pooled": pooled,
        "per_head": per_head,
        "heads": heads,
        "bands": bands(system),
        "burst_amplitude": float(np.median(readings["amplitude"]))
        if readings["amplitude"].size else float("nan"),
        "why": ("one head wrote both bands in one pass, so their timing "
                "difference is that head's dispersion and nothing else"),
    }


def head_difference(measurement: Dict[str, object]) -> Dict[str, object]:
    """The two heads against each other, which needs no absolute reference.

    Whatever is common to the deck - its chroma filters, its preamp, its
    luma modulator - is shared by both heads, so it cancels here even
    though it does not cancel in the tap difference. This is the reading
    that belongs to the heads alone.
    """
    per_head = measurement.get("per_head") or {}
    if 0 not in per_head or 1 not in per_head:
        return {"difference_ns": float("nan"), "usable": False,
                "why": "the heads were not labelled on this capture"}
    first, second = per_head[0], per_head[1]
    delta = (first["median_us"] - second["median_us"]) * 1e3
    error = math.hypot(first["standard_error_us"],
                       second["standard_error_us"]) * 1e3
    return {
        "difference_ns": delta,
        "error_ns": error,
        "sigma": abs(delta) / error if error > 0 else float("inf"),
        "lines": (first["lines"], second["lines"]),
        "usable": True,
        "cancels": ("everything the deck shares between its two heads - the "
                    "chroma filters, the preamp and the luma modulator"),
        "why": ("a difference between the heads carries no deck-common "
                "term, so it belongs to the heads alone"),
    }


def relate_to_head_delays(record: Dict[str, object],
                          playback: Dict[str, object]) -> Dict[str, object]:
    """This measurement against the sync edge's, on the same time axis.

    Ethan, 2026-09-06: *"Use this to relate on the time axis for the head
    measurements."*

    `head_differential.phase_signature` already carries a per-head delay
    from a different instrument entirely - the sync edge's own complex
    response, which reads the luma band alone and takes the two heads
    against each other. Its four captures give +1.5, 0.0, -1.0 and -7.1
    nanoseconds, and its docstring states the question it cannot answer:
    *"Which machine it lives in is not determined ... the two-tape test
    that settled the amplitude's side has never been run on the delay."*

    THE TAP PAIR IS THAT TEST, on the record side at least. The record tap
    carries the drive on its way to the head and the playback tap what
    comes back, so a per-head difference present at the record tap lives in
    the drive electronics and one absent there does not. Measured, the
    record tap gives +0.81 +/- 0.63 and -0.53 +/- 0.68 nanoseconds on two
    signals - consistent with zero at 1.3 and 0.8 standard errors - while
    the playback tap gives 2.0 to 2.5. So the per-head delay is NOT in the
    record-side drive.

    IT STILL DOES NOT SAY WHICH OF THE THREE REMAINING PLACES, and the
    honest list is short: the head's write, the head's read, and the
    playback preamp. Write and read are one physical head in one chain and
    never occur apart, so no capture from a single deck can separate them.

    AND THE TWO INSTRUMENTS DISAGREE ABOUT SIZE ON ONE CAPTURE. Three of
    the sync edge's four delays - +1.5, 0.0 and -1.0 ns - sit inside this
    measurement's own 4.6 nanosecond bound, and the fourth, -7.1 ns from
    the long-play capture, does not. That is worth holding: the long-play
    recordings are also where this module's own estimator stops locking.
    """
    from vhsdecode.models import head_differential
    here = tap_difference(record, playback)
    record_heads = head_difference(record)
    playback_heads = head_difference(playback)
    there = np.asarray(head_differential.MEASURED_DELAYS_S) * 1e9
    bound = float(abs(playback_heads.get("difference_ns", float("nan"))))
    inside = [float(value) for value in there if abs(value) <= max(bound, 4.6)]
    return {
        "tap_difference_ns": here["delay_ns"],
        "record_tap_head_difference_ns": record_heads.get("difference_ns"),
        "record_tap_head_sigma": record_heads.get("sigma"),
        "playback_tap_head_difference_ns": playback_heads.get("difference_ns"),
        "sync_edge_delays_ns": there,
        "sync_edge_inside_this_bound": inside,
        "record_side_excluded": bool(
            record_heads.get("usable")
            and record_heads.get("sigma", float("inf")) < 3.0),
        "remaining_places": ("the head's write, the head's read and the "
                             "playback preamp"),
        "cannot_separate": ("write from read, because they are one head in "
                            "one chain and never occur apart"),
        "why": ("the sync edge reads the luma band alone and cannot say "
                "which machine its delay lives in; the two taps can say "
                "whether it is in the record-side drive, and it is not"),
    }
