"""The vertical interval's own geometry as the low-frequency probe.

Ethan: *"Also use the equalization pulses in the same way but for longer
frequencies and possibly ghosting. I think we can just use the same complex
3d differential again."* And: *"I already described this method earlier, but
it seems it was missed at the point in the process."*

HE IS RIGHT THAT IT WAS MISSED. `docs/VERTICAL_INTERVAL_COLLAPSE.md` names
three populations in the interval and separates them explicitly - the
interval's OWN GEOMETRY (equalizing pulses, serrations, the field-sync
block), the inserted test signals, and the data signals. Only the second was
built. `vertical_interval.py` models NTC-7 and the colour bars; nothing
modelled the geometry, which is the population the format itself specifies
and the only one present on every tape whatever was recorded.

WHY IT IS THE ANSWER TO "LONGER FREQUENCIES", IN ONE LINE OF ARITHMETIC. A
probe resolves frequencies no finer than the reciprocal of its own duration.

    a single line-sync pulse         4.70 us   ->  212.8 kHz
    one whole line                  63.56 us   ->   15.7 kHz
    the equalizing sequence alone  190.67 us   ->    5.2 kHz
    the whole 9-line interval      572.00 us   ->    1.7 kHz

So the sync pulse cannot see anything below about 200 kHz, and the interval
sees to 1.7 kHz - a factor of 122. That is not a refinement of the same
measurement; it is the difference between having a low-frequency probe and
not having one.

AND IT IS THE INSTRUMENT THE BACK-PORCH DISTORTION NEEDS. The ringing lane
measured the residual Ethan is asking to correct: a recovery tail running
-3.6 IRE at the sync rise to +0.3 IRE at active video, a single time constant
of 1.22 to 1.34 us, its energy peaking at 260 kHz, polarity-COMMON with the
two heads correlating at r = +1.000, and only PARTLY removed by the existing
correction - see below, where the surviving amplitude is measured per head.
260 kHz is barely above
the 213 kHz a single sync pulse can resolve at all, and a 1.2 us relaxation
has most of its energy BELOW that. The probe and the defect were mismatched,
which is why measuring it on the pulse alone found a tail and could not
correct it.

GHOSTING FALLS OUT OF THE SAME NUMBER, AND THIS IS THE STRONGER RESULT. An
echo at delay `tau` is a ripple of period `1/tau` in the frequency response,
so resolving it needs a frequency resolution finer than half that ripple:
`tau <= 1/(2 df)`. The reachable delay window is therefore

    from a single sync pulse    0.14 us to   2.35 us
    from the whole interval     0.14 us to 286.16 us

with the short end set by the specified 140 ns pulse edge in both cases. A
broadcast ghost is typically a fraction of a microsecond to a few tens of
microseconds. THE SINGLE SYNC PULSE CANNOT SEE A TYPICAL GHOST AT ALL - its
whole reachable window ends at 2.35 us - and the equalizing-pulse train
covers the entire multipath range with two decades to spare.

THE TIME SCALES ARE THE STANDARD'S OWN, NOT AN INVENTION HERE. ITU-R
BT.1439-1 splits linear waveform distortion by time scale into long-time,
field-time, line-time and short-time, and gives each its own method. Those
four are what the probes above reach, one for one, and naming them that way
is what keeps this a measurement in the standard's vocabulary rather than a
new axis with no reference.

WHERE IT SITS IN THE CHAIN. Ethan: *"The distortion likely happened before
the clipping, so somewhere on the input signal."* The recorder's clip is at
position 9, so this is below it. The ringing lane's independent evidence
agrees and is stronger than the localisation alone: a single relaxation
COMMON to both heads, at r = +1.000, cannot be the head or the head-to-tape
path, because those differ measurably between the two. Position 6 puts it
after the tuner's own clip at 5 and before the recorder at 9, which is where
a video clamp sits - and the placement relative to the TUNER's clip is an
assumption, stated as one, not a measurement.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# The format's own timing specification.
#
# ITU-R BT.1700 Table 3, as extracted in docs/BT1700_COLLAPSE.md section 1.2
# (the row letters are that table's). Values are the 525/60 column unless a
# 625 value is named beside it.
# --------------------------------------------------------------------------

LINE_PERIOD_US = {"525": 1e6 / (525 * 30000.0 / 1001.0),
                  "625": 1e6 / (625 * 25.0)}

# p: equalizing pulse width
EQUALIZING_PULSE_US = {"525": 2.30, "625": 2.35}
EQUALIZING_PULSE_TOLERANCE_US = {"525": 0.10, "625": 0.10}
# r: serration interval
SERRATION_US = {"525": 4.70, "625": 4.70}
# q: field-sync pulse width
FIELD_SYNC_PULSE_US = {"525": 27.1, "625": 27.3}
# f: line-sync pulse width, the single-pulse probe this one is measured
# against
LINE_SYNC_US = {"525": 4.70, "625": 4.70}
# s: sync / equalizing pulse edge, which sets the short end of every window
PULSE_EDGE_NS = {"525": 140.0, "625": 200.0}
# K: blanking to the first equalizing pulse
BLANKING_TO_FIRST_EQ_US = {"525": 1.50, "625": 3.00}
# l, m, n: the equalizing / field-sync / equalizing sequences, in lines
SEQUENCE_LINES = {"525": (3.0, 3.0, 3.0), "625": (2.5, 2.5, 2.5)}

# The pulses in each sequence sit at HALF-LINE spacing, which is what makes
# the interval a train rather than three long pulses, and is why it carries
# structure at the line rate and below rather than only at the line rate.
PULSES_PER_LINE = 2

CHAIN_PREFIX = "waveform distortion"
CHAIN_POSITION = 6

COMPONENT_POSITIONS: Dict[str, int] = {
    CHAIN_PREFIX: CHAIN_POSITION,
    # The four time scales ITU-R BT.1439-1 defines. They share a position
    # because they are one mechanism read at four durations, and a
    # measurement at one scale commutes with a measurement at another.
    "waveform distortion (short time)": CHAIN_POSITION,
    "waveform distortion (line time)": CHAIN_POSITION,
    "waveform distortion (field time)": CHAIN_POSITION,
    "waveform distortion (long time)": CHAIN_POSITION,
}

# The four scales, named as BT.1439-1 names them, each with the probe whose
# duration reaches it. The durations are derived in `probes`, never stated.
TIME_SCALES = ("short time", "line time", "field time", "long time")


def _system(system: str) -> str:
    name = str(system).upper()
    if name in ("NTSC", "525", "M", "525/60"):
        return "525"
    if name in ("PAL", "SECAM", "625", "B", "625/50"):
        return "625"
    raise ValueError(f"no vertical-interval geometry stated for {system!r}")


def interval_geometry(system: str = "NTSC") -> Dict[str, object]:
    """The interval's own geometry, from the format's timing table.

    The three sequences, the pulse each carries, and the total duration -
    every number from the specification and none from a waveform.
    """
    key = _system(system)
    line = LINE_PERIOD_US[key]
    first, middle, last = SEQUENCE_LINES[key]
    return {
        "system": key,
        "line_period_us": line,
        "sequences": (
            {"name": "equalizing", "lines": first,
             "pulse_us": EQUALIZING_PULSE_US[key],
             "pulses": int(round(first * PULSES_PER_LINE))},
            {"name": "field sync", "lines": middle,
             "pulse_us": FIELD_SYNC_PULSE_US[key],
             "serration_us": SERRATION_US[key],
             "pulses": int(round(middle * PULSES_PER_LINE))},
            {"name": "equalizing", "lines": last,
             "pulse_us": EQUALIZING_PULSE_US[key],
             "pulses": int(round(last * PULSES_PER_LINE))},
        ),
        "total_lines": first + middle + last,
        "total_us": (first + middle + last) * line,
        "edge_ns": PULSE_EDGE_NS[key],
        "blanking_to_first_eq_us": BLANKING_TO_FIRST_EQ_US[key],
        "source": ("ITU-R BT.1700 Table 3 rows K, l, m, n, p, q, r, s, "
                   "extracted at docs/BT1700_COLLAPSE.md section 1.2"),
    }


def resolution_hz(duration_us: float) -> float:
    """The finest frequency a probe of this duration can resolve: `1/T`."""
    return float(1e6 / max(float(duration_us), 1e-12))


def probes(system: str = "NTSC") -> Dict[str, Dict[str, float]]:
    """EVERY PROBE THE SYNC AREA OFFERS, WITH WHAT EACH ONE REACHES.

    This is the whole argument for using the equalizing pulses, as four
    numbers derived from the timing table and nothing else. Each probe is
    matched to the BT.1439-1 time scale its resolution puts it at.
    """
    key = _system(system)
    geometry = interval_geometry(key)
    line = LINE_PERIOD_US[key]
    first, middle, last = SEQUENCE_LINES[key]
    windows = (
        ("the line-sync pulse", LINE_SYNC_US[key], "short time"),
        ("one whole line", line, "line time"),
        ("the equalizing sequence", first * line, "field time"),
        ("the whole vertical interval", float(geometry["total_us"]),
         "field time"),
    )
    out = {}
    for name, duration, scale in windows:
        out[name] = {
            "duration_us": float(duration),
            "resolution_hz": resolution_hz(duration),
            "time_scale": scale,
        }
    return out


def ghost_reach(duration_us: float, system: str = "NTSC") -> Dict[str, float]:
    """THE ECHO DELAYS A PROBE OF THIS DURATION CAN RESOLVE.

    An echo at delay `tau` is a ripple of period `1/tau` in the frequency
    response, so a probe resolving `df` can see it only while `1/tau >= 2 df`,
    that is `tau <= 1/(2 df)`. The short end is set by the pulse edge the
    format specifies, because an echo closer than the impulse's own width
    cannot be separated from it.

    Measured against a broadcast ghost, which runs from a fraction of a
    microsecond to a few tens: a single line-sync pulse reaches 2.35 us and
    so cannot see a typical one at all, and the whole interval reaches
    286 us.
    """
    key = _system(system)
    df = resolution_hz(duration_us)
    return {
        "shortest_us": PULSE_EDGE_NS[key] / 1000.0,
        "longest_us": float(1e6 / (2.0 * df)),
        "resolution_hz": df,
        "why": ("an echo at tau is a ripple of period 1/tau, so tau <= "
                "1/(2 df); the short end is the specified pulse edge"),
    }


def pulse_train(system: str = "NTSC", sample_rate_hz: float = 40e6,
                depth: float = -40.0, level: float = 0.0) -> Dict[str, object]:
    """THE STIMULUS: the interval's geometry as a waveform, from the spec.

    A rectangle per pulse with edges of the specified rise time, at the
    specified widths and half-line spacing, across the three sequences. This
    is the synthetic side of the differential - what the format says was
    sent - and it is built from the timing table alone, so no part of it is
    read off a recording.

    `depth` is the sync level and `level` the blanking level, in IRE; the
    field-sync pulses are the inverse of the equalizing ones, being blanking
    interrupted by serrations rather than sync interrupted by blanking.
    """
    key = _system(system)
    geometry = interval_geometry(key)
    line = LINE_PERIOD_US[key]
    rate = float(sample_rate_hz)
    total = float(geometry["total_us"])
    n = int(round(total * 1e-6 * rate))
    t = np.arange(n, dtype=np.float64) / rate * 1e6      # microseconds
    wave = np.full(n, float(level), dtype=np.float64)
    edge = PULSE_EDGE_NS[key] / 1000.0

    marks: List[Tuple[float, float, str]] = []
    cursor = 0.0
    half = line / PULSES_PER_LINE
    for sequence in geometry["sequences"]:
        count = int(sequence["pulses"])
        if sequence["name"] == "field sync":
            # blanking interrupted by serrations: the pulse is the WIDE one
            # and the serration is the gap
            width = float(sequence["pulse_us"])
        else:
            width = float(sequence["pulse_us"])
        for index in range(count):
            start = cursor + index * half
            marks.append((start, start + width, sequence["name"]))
        cursor += float(sequence["lines"]) * line

    for start, stop, _name in marks:
        # a raised-cosine edge of the specified rise time, which is the shape
        # a band-limited system makes of an ideal step
        rising = np.clip((t - start) / max(edge, 1e-9), 0.0, 1.0)
        falling = np.clip((stop - t) / max(edge, 1e-9), 0.0, 1.0)
        gate = (0.5 - 0.5 * np.cos(np.pi * rising)) * \
               (0.5 - 0.5 * np.cos(np.pi * falling))
        wave = np.minimum(wave, float(level)
                          + (float(depth) - float(level)) * gate)
    return {
        "waveform": wave,
        "time_us": t,
        "sample_rate_hz": rate,
        "marks": marks,
        "duration_us": total,
        "resolution_hz": resolution_hz(total),
        "geometry": geometry,
    }


# SMPTE 32M-2004, catalogued at docs/SPECIFICATION_INVENTORY.md:1053: the
# head-switch point is specified as 5 to 8 H ahead of the V-sync leading
# edge. The interval begins only 3 H ahead of that edge, so ON SPEC the
# switch always clears the interval - see `head_switch_window` for what the
# measurement says instead.
HEAD_SWITCH_AHEAD_OF_VSYNC_H = (5.0, 8.0)

# How long the switch's transient runs after its onset. This is NOT a
# specified number and is labelled as an assumption wherever it is used; the
# head-switch lane owns the measurement and `head_switch_window` takes it as
# an argument so a measured value replaces it.
ASSUMED_SWITCH_TRANSIENT_H = 1.0


def head_switch_window(system: str = "NTSC",
                       onset_h_before_vsync: Optional[float] = None,
                       transient_h: float = ASSUMED_SWITCH_TRANSIENT_H
                       ) -> Dict[str, object]:
    """WHERE THE HEAD SWITCH FALLS RELATIVE TO THE INTERVAL, AND WHETHER IT
    LANDS INSIDE IT.

    Ethan: *"I do have outliers caused by head switching distortion that may
    be in the vertical sync area. We should account for those."*

    They may, and the specification says they should not, which is the
    interesting part. SMPTE 32M puts the switch 5 to 8 H ahead of the V-sync
    leading edge; the interval opens with its first equalizing sequence only
    3 H ahead of that edge. So on the specification the switch is 2 to 5 H
    clear of the interval and its transient never reaches the equalizing
    pulses.

    NEITHER MEASURED READING IS ON SPEC, AND THE TWO DISAGREE WITH EACH
    OTHER. This is reported as an open question rather than resolved,
    because resolving it by picking the convenient one would be a fit.

        the specification            5 to 8 H BEFORE V-sync, clears by 2-5 H
        one earlier reading          about 3.5 H before V-sync, landing half
                                     a line ahead of the interval's start
        the head-switch lane's own   linelocs index 267.0, spread 0.8 rows,
        measurement, 51 fields       38 of 51 fields at exactly 267

    The lane's figure is servo-tight and hard to dismiss - an earlier attempt
    of theirs using an envelope step scattered over 21 rows and was picking
    up dropouts, and this one does not. What is uncertain is the MAPPING, not
    the measurement. Their output buffer is 263 rows with the interval at
    rows 0 to 8 and the V-sync leading edge at row 3, while `linelocs` holds
    273 entries - the extra ten being padding into the next field. Read
    one-to-one that puts the switch at row 4 of the FOLLOWING field, one line
    AFTER its V-sync leading edge and on the serrations, which is the worst
    place in the interval for it to be and further from spec than the other
    reading rather than closer. They say plainly they do not believe their
    own chain of inference there, and the weak link they name is the
    assumption that a linelocs index maps one-to-one onto an output row.

    THE CONCLUSION SURVIVES THE DISAGREEMENT, which is why it is safe to act
    on now. Masking costs 5.6 per cent of the window on the first reading and
    11.1 per cent on the second, taking the resolution from 1748 Hz to 1851
    or 1967 Hz. Against a sync pulse's 212 765 Hz either is nothing, so the
    switch is masked out on both readings and the question of which is right
    does not have to be settled first.

    Pass `onset_h_before_vsync` to use a measured value - it is SIGNED, so a
    negative value places the switch after the V-sync leading edge. The
    default reports the specification's own range.

    Times are returned relative to the START OF THE INTERVAL, matching
    `pulse_train`, so a negative start means the switch precedes it.
    """
    key = _system(system)
    line = LINE_PERIOD_US[key]
    first, _middle, _last = SEQUENCE_LINES[key]
    # the interval begins `first` lines ahead of the V-sync leading edge
    if onset_h_before_vsync is None:
        low, high = HEAD_SWITCH_AHEAD_OF_VSYNC_H
        onsets = (high, low)          # earliest and latest, in time order
        measured = False
    else:
        onsets = (float(onset_h_before_vsync),) * 2
        measured = True
    earliest = (first - onsets[0]) * line
    latest = (first - onsets[1]) * line + float(transient_h) * line
    interval_us = float(interval_geometry(key)["total_us"])
    overlaps = latest > 0.0 and earliest < interval_us
    return {
        "start_us": earliest,
        "stop_us": latest,
        "interval_us": interval_us,
        "overlaps_the_interval": bool(overlaps),
        "onset_h_before_vsync": (onsets if not measured else onsets[0]),
        "from_measurement": measured,
        "specified_range_h": HEAD_SWITCH_AHEAD_OF_VSYNC_H,
        "transient_h": float(transient_h),
        "transient_is_assumed": transient_h == ASSUMED_SWITCH_TRANSIENT_H,
        "why": ("the specification clears the interval by 2 to 5 H; the "
                "measured onset near line 259-260 does not, and lands about "
                "half a line ahead of the interval's own start"),
    }


def interval_mask(system: str = "NTSC", sample_rate_hz: float = 40e6,
                  onset_h_before_vsync: Optional[float] = None,
                  transient_h: float = ASSUMED_SWITCH_TRANSIENT_H
                  ) -> Dict[str, object]:
    """The part of the interval the head switch has not touched.

    EXCLUDED, NOT REPAIRED. A switch transient inside the window is not a
    small perturbation of the response being measured, it is a different
    channel for its duration - the other head - so a differential taken
    across it measures the difference between two heads and calls it a
    frequency response. Masking is the only correct treatment, and losing
    the masked samples costs resolution, which `resolution_hz` on the
    surviving span reports honestly rather than quietly keeping the full
    window's figure.
    """
    key = _system(system)
    train = pulse_train(key, sample_rate_hz)
    t = np.asarray(train["time_us"], dtype=np.float64)
    window = head_switch_window(key, onset_h_before_vsync, transient_h)
    keep = ~((t >= float(window["start_us"]))
             & (t <= float(window["stop_us"])))
    kept_us = float(keep.sum()) / float(sample_rate_hz) * 1e6
    return {
        "mask": keep,
        "time_us": t,
        "removed_fraction": float(1.0 - keep.mean()),
        "kept_us": kept_us,
        "resolution_hz": resolution_hz(kept_us) if kept_us > 0 else np.inf,
        "full_resolution_hz": resolution_hz(float(train["duration_us"])),
        "head_switch": window,
    }


def excluding_clip(ideal, measured, limit: Optional[float] = None,
                   edge: str = "dark", format_name: str = "VHS"
                   ) -> Dict[str, object]:
    """THE IDEAL TRUNCATED WHERE A CLIP IS DETECTED, AND THE MASK THAT DROPS
    IT FROM THE DIFFERENTIAL.

    Ethan: *"I think it is the 3d complex differential using the shape of the
    sync pulse, excluding clipping if that is detected."*

    Two things, and both are needed. The ideal is truncated the way the
    recording was, which is `clipping.truncated_ideal` and stops the clip
    being attributed to everything downstream of it. And the samples the clip
    flattened are MASKED OUT of the differential, because at a clipped level
    the channel's gain is zero and the ratio the differential forms there is
    not a measurement of anything.

    `limit` defaults to the standard's own dark clip for the format, which is
    stated as a fraction of the sync-tip-to-peak-white span; pass a measured
    one where a clip has been detected at a different level.
    """
    from vhsdecode.models import clipping

    reference = np.asarray(ideal, dtype=np.float64)
    observed = np.asarray(measured, dtype=np.float64)
    if limit is None:
        span = float(np.max(reference) - np.min(reference))
        limit = float(np.min(reference)
                      + clipping.clip_levels(format_name)["dark"] * span)
    truncated = clipping.truncated_ideal(reference, limit, edge)
    clipped = truncated != reference
    error = clipping.truncation_error(reference, limit, edge)
    return {
        "ideal": truncated,
        "mask": ~clipped,
        "clipped_fraction": float(np.mean(clipped)),
        "limit": float(limit),
        "truncation_error": error,
        "measured": observed,
        "why": ("the ideal is truncated the way the recording was, and the "
                "flattened samples are dropped because a clipped level has "
                "no gain and the ratio there measures nothing"),
    }


def differential(measured, ideal, sample_rate_hz: float,
                 mask=None, floor: float = 1e-9) -> Dict[str, object]:
    """THE COMPLEX DIFFERENTIAL OF A MEASURED SHAPE AGAINST THE SPECIFIED ONE.

    The same transform the rest of the arc uses, applied to whatever window
    the caller passes - a single sync pulse, the back porch, or the whole
    equalizing train. The window's DURATION is what decides which frequencies
    come back, which is the entire point of using the train.

    A difference and never a ratio in the time domain: the spectra are
    divided only where the stimulus has mass, and the reported band is
    bounded below by the window's own resolution, so nothing is claimed at a
    frequency the window cannot resolve.
    """
    y, x = np.asarray(measured), np.asarray(ideal)
    if y.shape != x.shape:
        raise ValueError("the measured and specified shapes must be on the "
                         "same grid")
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        y, x = y * keep, x * keep
    rate = float(sample_rate_hz)
    n = y.size
    # THE TRANSFER IS COMPLEX AND SO IS THE TRANSFORM THAT RECOVERS IT.
    # Ethan: *"The transfer of all of these are in the complex domain."*
    # `rfft` folds the negative frequencies onto the positive ones, which is
    # exact for a real waveform and destroys a carrier's sideband asymmetry -
    # so a complex input takes the full transform. The real path is the same
    # as before.
    if np.iscomplexobj(y) or np.iscomplexobj(x):
        freqs = np.fft.fftfreq(n, 1.0 / rate)
        Y = np.fft.fft(y.astype(np.complex128) - y.mean())
        X = np.fft.fft(x.astype(np.complex128) - x.mean())
    else:
        y, x = y.astype(np.float64), x.astype(np.float64)
        freqs = np.fft.rfftfreq(n, 1.0 / rate)
        Y, X = np.fft.rfft(y - y.mean()), np.fft.rfft(x - x.mean())
    power = np.abs(X) ** 2
    live = power > float(floor) * max(float(power.max()), 1e-30)
    H = np.zeros_like(Y)
    H[live] = Y[live] / X[live]
    duration_us = n / rate * 1e6
    return {
        "frequency_hz": freqs,
        "H": H,
        "valid": live,
        "resolution_hz": resolution_hz(duration_us),
        "duration_us": duration_us,
        "lowest_honest_hz": resolution_hz(duration_us),
        "ghost_reach": ghost_reach(duration_us),
        "why": ("the window's duration sets the lowest frequency that can be "
                "claimed; nothing below the resolution is reported"),
    }


def recovery_tail(time_us, amplitude_ire: float, time_constant_us: float
                  ) -> np.ndarray:
    """The back-porch residual as a shape: one decaying exponential.

    The ringing lane measured it on the home tape - the tail rises from
    +2.92 IRE at the sync rise's 10-90 end to a settled +6.54 over 3.77 us,
    the two heads correlating at r = +1.000, and its energy peaking at
    260 kHz. A single time constant fits it, which is why it is a RELAXATION
    and not a resonance: 260 kHz is where a microsecond relaxation puts its
    energy rather than a resonant line.

    THE TIME CONSTANT IS DISPUTED BY A FACTOR OF THREE AND THE CONCLUSION
    SURVIVES EITHER VALUE, which is why it is recorded rather than resolved.

        the lane's first figures      1.22 / 1.29 us
        the same samples, refitted    1.071 / 1.057 us
        the lane's correction         2.94 / 3.18 us

    The lane withdrew its first figures with a mechanism that is real and
    general: an estimator that ASSUMES the level a tail decays to, taking it
    from the trailing samples of a window the tail has not settled in, biases
    every time constant short - invisibly on short constants and totally on
    long ones. Their fix is to fit the offset JOINTLY, which for fixed tau is
    linear in the amplitude and the offset and so is exact by least squares.

    THAT FIX DOES NOT REPRODUCE THEIR CORRECTION ON THE SAMPLES THEY
    EXPORTED. Running exactly that form here gives 1.071 and 1.057 us, and
    2.94 fits the same samples 36 per cent worse in rms. And the window is
    not the limitation: planting known tails in that same 3.77 us window at
    its own noise, 200 draws each, the scan recovers 0.50 as 0.50, 1.07 as
    1.06, 2.94 as 2.91 and only loses identifiability past about 0.75 time
    constants of window. A 2.94 us tail in that window would have come back
    as 2.94.

    So the difference is in the INPUT, not the fit - most likely a corrected
    against an uncorrected decode arm, which the export's metadata does not
    state - and it has been asked rather than guessed.

    EITHER VALUE SUPPORTS THE FINDING, AND THE LARGER ONE SUPPORTS IT MORE. A
    relaxation corners at `1/(2 pi tau)`: 123 to 151 kHz at the short values,
    50 to 54 kHz at the long ones. Every one of those is below the 213 kHz
    floor a 4.7 us sync pulse can resolve, so the probe could not see what it
    was being asked to remove on any reading of the number.

    A single exponential is a good but not exact description either way -
    0.21 IRE rms against a 3.9 IRE excursion, about five per cent - so the
    shape should be entered from exported samples rather than regenerated
    from two parameters.

    IT IS AN UNCORRECTED RESIDUAL, AND THE SIZE IS HEAD-DEPENDENT. This went
    round twice and the settled measurement is worth stating carefully.

    The lane first reported the relaxation ALREADY REMOVED by the ringing
    stage - head A fitting a pole at 0.976450 against a measurement of
    0.976526, four-decimal agreement. That was the same window error as the
    time constant: two fits of the same driving edge. They then reported it
    barely touched, one and seven per cent, from a ring-off against ring-on
    pair; THOSE ARMS ARE ALSO WITHDRAWN, because the flag that was supposed
    to enable the stage was silently ignored and the two arms were identical.

    The measurement that stands is on the corrected field, window 76 to 128:

        raw fold                head A -2.410 IRE   head B -2.476
        surviving the stage     head A -1.672       head B -0.580
        so the stage removes           31 per cent         77 per cent

    THE MECHANISM IS HEAD-COMMON AND THE RESIDUAL IS NOT, which is the part
    that matters for entering it. The relaxation itself correlates at
    r = +1.000 between the heads, so it is one object arriving before the
    head; but the ringing stage's sections DIFFER between heads, so what
    survives it does not. A component entered at position 6 from a
    head-pooled shape would therefore be right about the mechanism and wrong
    about the residual, and it has to be entered per head.

    The position is declared and still unfilled: the export in hand
    (porch_remainder_home.npz) was measured on the RAW fold, so its shapes
    are of the uncorrected signal rather than of the residual, and a
    corrected-field export on the same geometry has been asked for.

    Its size is established on ONE TAPE, over 8 fields. The chroma lane's
    independent figure - +4.5 to +0.3 IRE over about 4 us with a 1.5 us time
    constant peaking at 0.1 to 0.3 MHz - agrees on duration, time constant
    and band, which is what says it is not one tape's artefact. It does not
    establish the size anywhere else, and this arc has twice mistaken one
    tape's consistency for generality.

    Its size is established on ONE TAPE. The chroma lane's independent figure
    - +4.5 to +0.3 IRE over about 4 us with a 1.5 us time constant peaking at
    0.1 to 0.3 MHz - agrees on duration, time constant and band, which is
    what says it is not one tape's artefact. It does not establish the size
    anywhere else, and this arc has twice mistaken one tape's consistency for
    generality.
    """
    t = np.asarray(time_us, dtype=np.float64)
    return float(amplitude_ire) * np.exp(-np.maximum(t, 0.0)
                                         / max(float(time_constant_us), 1e-9))


def signatures(frequency_hz, system: str = "NTSC",
               time_constant_us: float = 1.22,
               scales: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """The interval's entries for the key, one per BT.1439-1 time scale.

    Each is the relaxation's response read at the scale that probe reaches:
    a single pole at the corner the time constant sets, evaluated over the
    band the window can resolve. They are separate entries because the
    standard measures them separately and because a chain can distort at one
    scale and not another - a clamp acts at line time, a thermal drift at
    long time - not because the pole differs.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    tau = max(float(time_constant_us), 1e-9) * 1e-6
    reach = probes(system)
    corners = {
        "short time": reach["the line-sync pulse"]["resolution_hz"],
        "line time": reach["one whole line"]["resolution_hz"],
        "field time": reach["the whole vertical interval"]["resolution_hz"],
        "long time": reach["the whole vertical interval"]["resolution_hz"] / 60.0,
    }
    wanted = list(scales) if scales is not None else list(TIME_SCALES)
    out: Dict[str, np.ndarray] = {}
    for scale in wanted:
        if scale not in corners:
            raise ValueError(f"{scale!r} is not one of {TIME_SCALES}")
        # the relaxation, high-passed at the resolution the probe reaches:
        # below its own resolution a probe says nothing, and a signature that
        # claimed something there would be manufacturing it
        pole = 1.0 / (1.0 + 2j * np.pi * grid * tau)
        corner = corners[scale]
        reachable = (1j * grid / corner) / (1.0 + 1j * grid / corner)
        out[f"waveform distortion ({scale})"] = (pole * reachable
                                                 ).astype(np.complex128)
    return out


def blanking_mask(json_path: Optional[str] = None,
                  field_width: Optional[int] = None,
                  active_start: Optional[int] = None,
                  active_end: Optional[int] = None) -> Dict[str, object]:
    """THE WHOLE SYNC REGION OF A LINE, with the active picture masked out.

    Ethan: *"I need to measure the entire sync region, and mask out the
    active area from this measurement."*

    Everything between the end of one line's active picture and the start
    of the next's belongs to the measurement: the front porch, the sync
    pulse, the breezeway, the colour burst and the back porch. The arc has
    until now measured the sync PULSE and treated the porches separately,
    which throws away the burst - the finest timing instrument there is -
    and the porches' own shape.

    The boundaries are READ FROM THE DECODE, never assumed: a `.tbc.json`
    carries `fieldWidth`, `activeVideoStart` and `activeVideoEnd`, and
    those are what the decoder actually used. Measured on this session's
    decodes: 910 samples a line with active video from 134 to 894, so the
    sync region is 150 samples - the 16 at the end of the line plus the 134
    at the start of the next - which at 4 fsc is 10.5 microseconds against
    the 10.9 +/- 0.2 that ITU-R BT.1700 specifies for line blanking.

    AND THE DECODE'S BOUNDS DISAGREE WITH THE STANDARD, by six samples.
    Measured on all three of this session's decodes: the mask they imply is
    10.476 microseconds where BT.1700 specifies 10.9 +/- 0.2, so the
    decoder's active window is 0.42 microseconds - six samples at 4 fsc -
    WIDER than blanking allows. The direction is the safe one for this
    purpose: the mask is conservative, containing only blanking and no
    active picture, at the cost of six samples of back porch treated as
    though they were picture. `specified` returns the standard's own extent
    instead, for a caller who would rather have the whole region and accept
    the risk at its edge; the two are reported side by side so the
    disagreement is visible rather than resolved silently.

    Returns the mask over one line's samples and the region's own extent,
    so a caller can report the resolution the surviving span affords rather
    than the whole line's.
    """
    if json_path is not None:
        import json
        with open(json_path) as handle:
            parameters = json.load(handle)["videoParameters"]
        field_width = int(parameters["fieldWidth"])
        active_start = int(parameters["activeVideoStart"])
        active_end = int(parameters["activeVideoEnd"])
    if field_width is None or active_start is None or active_end is None:
        raise ValueError("give a decode's JSON, or its width and active "
                         "bounds; these are not assumed")
    columns = np.arange(int(field_width))
    active = (columns >= int(active_start)) & (columns < int(active_end))
    mask = ~active
    rate = 4.0 * SUBCARRIER_HZ if "SUBCARRIER_HZ" in globals() else 4.0 * 315e6 / 88.0
    per_sample_us = 1e6 / rate
    # the standard's own extent, for comparison and as an alternative mask
    specified_samples = int(round(10.9e-6 * rate))
    specified = np.zeros(columns.size, dtype=bool)
    tail = min(specified_samples, columns.size)
    lead = max(int(active_start), 0)
    specified[:min(lead, tail)] = True
    remaining = tail - min(lead, tail)
    if remaining > 0:
        specified[columns.size - remaining:] = True
    return {
        "mask": mask, "active": active,
        "specified_mask": specified,
        "specified_samples": specified_samples,
        "shortfall_samples": specified_samples - int(mask.sum()),
        "shortfall_us": float((specified_samples - int(mask.sum())) * per_sample_us),
        "field_width": int(field_width),
        "active_start": int(active_start), "active_end": int(active_end),
        "sync_region_samples": int(mask.sum()),
        "active_samples": int(active.sum()),
        "sync_region_us": float(mask.sum() * per_sample_us),
        "specified_blanking_us": 10.9,
        "specified_blanking_tolerance_us": 0.2,
        "within_specification": bool(
            abs(mask.sum() * per_sample_us - 10.9) <= 0.2 + per_sample_us),
        "resolution_hz": float(1.0 / (mask.sum() * per_sample_us * 1e-6))
        if mask.sum() else float("inf"),
        "why": ("the whole region between two active pictures is the "
                "measurement - porch, pulse, breezeway, burst and porch - "
                "and its bounds come from the decode rather than from an "
                "assumption"),
    }


# The vertical interval's line numbers, ITU-R BT.1700 (525-line): three
# lines of pre-equalizing pulses, three of field sync with serrations,
# three of post-equalizing, then the VBI proper carrying test and data
# signals until active picture begins. Numbered from the field's first
# line, which is what a decoded field's row index counts.
VSYNC_LINES = 9                  # 3 pre-equalizing + 3 field sync + 3 post
VBI_LAST_LINE = 21               # active picture begins at line 22 (NTSC)


def frame_mask(json_path: Optional[str] = None,
               field_width: Optional[int] = None,
               field_height: Optional[int] = None,
               active_start: Optional[int] = None,
               active_end: Optional[int] = None,
               front_porch_keep_us: float = 0.2,
               vsync_lines: int = VSYNC_LINES,
               vbi_last_line: int = VBI_LAST_LINE) -> Dict[str, object]:
    """THE 2-D MASK OVER A WHOLE FIELD: keep the vertical sync and the line
    sync pulses, mask the vertical interval's test lines and all active
    picture.

    Ethan, 2026-09-06: *"Mask out the entire VBI and the active area, only
    the vsync and sync pulse area should remain. Stop the mask just after
    the tail end of the active area in the front porch."* And, on why:
    *"make sure to mask out the active area from carrying a residual that
    is not correlated with the timing information."*

    So three things go, and each for its own reason:

      * THE ACTIVE PICTURE, because a residual measured there is picture,
        not chain, and the arc's standing rule is that a correction derives
        from reserved intervals only.
      * THE VERTICAL INTERVAL'S TEST LINES, which are blanking by position
        but ACTIVE by content - a VITS line carries a drawn waveform, and
        including it puts a test signal's own shape into a timing
        measurement.
      * THE BURST, THE BREEZEWAY AND THE BACK PORCH, because Ethan's
        window ends at the sync pulse. What is kept is the front porch's
        tail and the pulse itself, which is the timing, and nothing that
        carries colour.

    The kept span per line therefore begins `front_porch_keep_us` after
    active picture ends - inside the front porch, as he specifies - and
    runs through the line sync pulse. The vertical sync block's own lines
    are kept whole, because there is no active picture on them at all.

    THE GUARD IS SMALL BECAUSE THE PORCH IS. Measured on this session's
    decodes the front porch is sixteen samples, columns 894 to 909, so a
    microsecond of guard would leave two of them and a fifth of a
    microsecond leaves thirteen. The default is the smaller, which is what
    "just after the tail end" asks for; a caller who wants the decoder's
    low-pass ringing cleared as well should raise it and accept the loss,
    and `front_porch_samples_kept` says what it costs.

    WHAT THE ACTIVE AREA IS FOR, since it is masked here. Ethan: *"The
    active area is the information that we are extracting from the luma."*
    It is the data, not the measurement: the chain is measured on the
    reserved intervals and the correction is applied to the picture. Its
    own residual has one further use, which is that it carries the residual
    colour carrier and therefore the colour lock's amplitude and phase.

    Bounds come from the decode's JSON where one is given, never assumed.
    """
    if json_path is not None:
        import json
        with open(json_path) as handle:
            parameters = json.load(handle)["videoParameters"]
        field_width = int(parameters["fieldWidth"])
        field_height = int(parameters["fieldHeight"])
        active_start = int(parameters["activeVideoStart"])
        active_end = int(parameters["activeVideoEnd"])
    if None in (field_width, field_height, active_start, active_end):
        raise ValueError("give a decode's JSON, or its geometry; these are "
                         "not assumed")
    rate = 4.0 * 315e6 / 88.0
    per_sample_us = 1e6 / rate
    keep_from = int(active_end) + int(round(front_porch_keep_us / per_sample_us))
    # THE SPAN ENDS AT THE SYNC PULSE, not at active video. A decoded line
    # starts at the sync datum, so columns 0 onward hold the pulse, then the
    # breezeway, the burst and the back porch before active picture at
    # `active_start`. Keeping as far as `active_start` would take in the
    # burst and both porches - colour and level, not timing - which is the
    # opposite of what this window is for. The pulse's own width is the
    # standard's own LINE_SYNC_US, BT.1700's 4.7 microseconds.
    pulse_end = int(round(LINE_SYNC_US["525"] / per_sample_us))
    columns = np.arange(int(field_width))
    # the kept span wraps the line end: from inside the front porch, over
    # the line boundary, and through the sync pulse
    line_keep = (columns >= keep_from) | (columns < pulse_end)
    rows = np.arange(int(field_height))
    is_vsync = rows < int(vsync_lines)
    is_vbi = (rows >= int(vsync_lines)) & (rows <= int(vbi_last_line))
    mask = np.zeros((int(field_height), int(field_width)), dtype=bool)
    mask[is_vsync, :] = True                      # the vertical sync, whole
    mask[~is_vsync & ~is_vbi, :] = line_keep      # picture lines: timing only
    # the VBI's test lines are dropped entirely
    return {
        "mask": mask,
        "line_keep": line_keep,
        "keep_from_column": keep_from,
        "sync_pulse_end_column": pulse_end,
        "front_porch_keep_us": float(front_porch_keep_us),
        "front_porch_samples_kept": int(field_width - keep_from),
        "vsync_rows": int(is_vsync.sum()),
        "vbi_rows_dropped": int(is_vbi.sum()),
        "picture_rows": int((~is_vsync & ~is_vbi).sum()),
        "kept_samples": int(mask.sum()),
        "field_samples": int(mask.size),
        "kept_fraction": float(mask.sum() / mask.size),
        "why": ("only the vertical sync and the line sync pulses carry "
                "timing without carrying content; the VBI's test lines are "
                "blanking by position and active by content, and the burst "
                "and back porch carry colour rather than timing"),
    }


def crossover_hz(short_probe_us: float = LINE_SYNC_US["525"],
                 long_probe_us: float = 572.0) -> Dict[str, float]:
    """Where the short probe stops and the long one must take over.

    Ethan: *"Use the eq pulses to get the longer value, and the hsync
    pulses to refine the higher frequency details."*

    A probe of duration T carries no information below 1/T, so the line
    sync pulse - 4.7 microseconds, ITU-R BT.1700 - says nothing under
    212.8 kHz however carefully it is measured, and the vertical interval's
    572 microseconds reaches down to 1.7 kHz. The crossover is therefore
    the SHORT probe's own resolution: below it only the long probe has
    anything to say, above it the short probe resolves detail the long one
    smears. Neither number is chosen; both are one over a duration the
    standard fixes.
    """
    short = 1.0 / (float(short_probe_us) * 1e-6)
    long_ = 1.0 / (float(long_probe_us) * 1e-6)
    return {
        "crossover_hz": short,
        "short_probe_us": float(short_probe_us),
        "short_resolution_hz": short,
        "long_probe_us": float(long_probe_us),
        "long_resolution_hz": long_,
        "decades_below_crossover": float(np.log10(short / long_)),
        "why": ("a probe of duration T carries nothing below 1/T, so the "
                "sync pulse's own width is the frequency beneath which only "
                "the vertical interval can speak"),
    }


def combined_response(low_hz, low_H, high_hz, high_H,
                      short_probe_us: float = 4.7,
                      long_probe_us: float = 572.0) -> Dict[str, object]:
    """One luma response from two probes, each used where it can resolve.

    `low_*` is measured on the equalizing and field-sync train, `high_*` on
    the line sync pulse. They are joined at `crossover_hz`, and the joint is
    made continuous by matching the LEVEL of the high-frequency measurement
    to the low one across the crossover rather than by fitting anything: a
    step at the joint would be an artefact of two instruments' gains, not a
    property of the channel, and the arc's rule is that a level belongs to
    `standard_levels` and never to a shape.

    The overlap band - from the crossover up to where the long probe's own
    resolution still holds - is where the two must AGREE, and their
    disagreement there is the honest error bar on the join.
    """
    edges = crossover_hz(short_probe_us, long_probe_us)
    crossover = edges["crossover_hz"]
    low_f = np.asarray(low_hz, dtype=np.float64).ravel()
    high_f = np.asarray(high_hz, dtype=np.float64).ravel()
    low = np.asarray(low_H).ravel()
    high = np.asarray(high_H).ravel()
    below = low_f < crossover
    above = high_f >= crossover
    # the overlap: where both probes are inside their own resolution
    overlap_low = low_f >= crossover
    overlap_high = high_f < crossover * 4.0
    offset = 0.0
    disagreement = float("nan")
    if overlap_low.any() and overlap_high.any():
        common = np.interp(low_f[overlap_low], high_f, np.abs(high))
        measured = np.abs(low[overlap_low])
        good = (common > 0) & (measured > 0)
        if good.any():
            ratio = np.log(measured[good] / common[good])
            offset = float(np.median(ratio))
            disagreement = float(np.std(ratio))
    joined_f = np.concatenate([low_f[below], high_f[above]])
    joined_H = np.concatenate([low[below], high[above] * np.exp(offset)])
    order = np.argsort(joined_f)
    return {
        "frequency_hz": joined_f[order], "H": joined_H[order],
        "crossover_hz": crossover,
        "from_long_probe": int(below.sum()),
        "from_short_probe": int(above.sum()),
        "level_offset_nepers": offset,
        "overlap_disagreement_nepers": disagreement,
        "edges": edges,
        "why": ("each probe is used only where its own duration lets it "
                "resolve; the joint carries a level match, never a fit, and "
                "the overlap's disagreement is the error bar on the join"),
    }


def probe_control(system: str = "NTSC",
                  time_constant_us: float = 1.22) -> Dict[str, object]:
    """THE CONTROL: A PROBE CANNOT RESOLVE BELOW ITS OWN RECIPROCAL DURATION,
    and the measured defect lies below the sync pulse's floor.

    The comparison is with the relaxation's own CORNER rather than with the
    band its residual peaks in, because the corner is where the information
    is and it is derived rather than read off a plot: a single relaxation of
    time constant `tau` has its corner at `1/(2 pi tau)`, so the 1.22 us tail
    the ringing lane measured corners at 130 kHz. The line-sync pulse
    resolves no finer than 213 kHz. THE WHOLE INFORMATIVE PART OF THE DEFECT
    IS BELOW THE PULSE'S FLOOR, which is why measuring it on the pulse found
    a tail and could not correct it.

    If this ever passes while a window is credited with reaching a feature
    finer than its own reciprocal duration, the resolution arithmetic has
    gone wrong and every low-frequency claim built on it is void.
    """
    reach = probes(system)
    pulse = reach["the line-sync pulse"]["resolution_hz"]
    interval = reach["the whole vertical interval"]["resolution_hz"]
    corner = 1.0 / (2.0 * np.pi * max(float(time_constant_us), 1e-9) * 1e-6)
    return {
        "sync_pulse_resolution_hz": pulse,
        "interval_resolution_hz": interval,
        "defect_corner_hz": corner,
        "sync_pulse_resolves_the_corner": bool(pulse <= corner),
        "interval_resolves_the_corner": bool(interval <= corner),
        "improvement": float(pulse / max(interval, 1e-12)),
        "passes": bool(interval < pulse and interval <= corner
                       and pulse > corner),
        "why": ("a 1.22 us relaxation corners at 1/(2 pi tau) = 130 kHz; the "
                "line-sync pulse resolves 213 kHz and cannot reach it, and "
                "the whole interval resolves 1.7 kHz and reaches it with two "
                "decades to spare"),
    }


# --------------------------------------------------------------------------
# The front porch, which is not a level and is not intrinsically unusable
# --------------------------------------------------------------------------
#
# ITU-R BT.1700 Table 3 row c: the front porch is 1.5 +- 0.1 us on 525/60,
# from blanking start to the sync 50 per cent fall. It is the SHORTEST
# settled region in the line and the only one preceded directly by active
# picture.
FRONT_PORCH_US = {"525": 1.50, "625": 1.50}
FRONT_PORCH_TOLERANCE_US = {"525": 0.10, "625": 0.10}


def front_porch_residual(samples, sample_rate_hz: float,
                         preceding_level_ire: float,
                         settled_ire: float,
                         time_constant_us: float = 1.07,
                         system: str = "NTSC") -> Dict[str, object]:
    """THE FRONT PORCH ISOLATED FROM WHAT RUNS INTO IT.

    Ethan: *"Regarding the front porch measurement, we still need to isolate
    that part to remove the residual ... I think the measurement component is
    not correct, or the measurement alignment or magnitude is not correct."*

    THAT IS A CORRECTION TO A STANDING ASSUMPTION IN THIS ARC AND IT IS
    RIGHT. The front porch was measured to move with picture content at 41 to
    66 sigma and was retired as "not a level", with the back porch after the
    burst used instead. Read as a property of the porch that is a dead end.
    Read as Ethan reads it - a wrong measurement COMPONENT - it is not, and
    the missing component is one this module already holds.

    The front porch is the settled blanking level PLUS THE RELAXATION LEFT
    OVER FROM THE ACTIVE LINE THAT PRECEDES IT. It is the same single
    relaxation the ringing lane measured decaying INTO the back porch after
    the sync rise, entering the front porch from the other side: same
    channel, same time constant, different stimulus. So the porch's
    dependence on content is not noise and not a disqualification - it is a
    MODELLED term, and subtracting it is what isolates the porch.

    THE ARITHMETIC SAYS WHY IT DOMINATES, and it does so on every disputed
    value of the time constant. The front porch is 1.5 us and the
    relaxation's time constant is somewhere between 1.07 and 3.18 us - see
    this module's docstring for why that is a range and not a number - so the
    porch spans between 1.40 and 0.47 time constants. A step of
    `preceding - settled` has therefore decayed only to between 25 and 63 per
    cent by the sync fall, and has never settled anywhere inside it on any
    reading. A 100 IRE swing in the preceding content leaves 25 to 63 IRE at
    the far end of the window, which is the scale of the 41-66 sigma
    dependence that retired the front porch as a level. The back porch is
    4.5 us, three times longer, which is the whole reason it settles and this
    one does not - and on the longer time constants it barely settles either.

    The default below is the SHORT value, which is the conservative choice
    here: it removes less, so what survives into the residual is honest
    rather than over-corrected.

    Returns the residual after the modelled relaxation is removed, and the
    share of the porch's variance that removal accounts for. If the share is
    high the porch is usable once corrected; if it is low then the alignment
    or the magnitude is wrong, which is the other half of what Ethan names.
    """
    values = np.asarray(samples, dtype=np.float64).ravel()
    if values.size < 3:
        return {"residual": values, "explained": 0.0,
                "why": "too few samples to isolate"}
    t = np.arange(values.size, dtype=np.float64) / float(sample_rate_hz) * 1e6
    step = float(preceding_level_ire) - float(settled_ire)
    modelled = float(settled_ire) + step * np.exp(
        -t / max(float(time_constant_us), 1e-9))
    residual = values - modelled
    before = float(np.var(values))
    after = float(np.var(residual))
    key = _system(system)
    length_us = values.size / float(sample_rate_hz) * 1e6
    return {
        "residual": residual,
        "modelled": modelled,
        "explained": float(1.0 - after / before) if before > 0 else 0.0,
        "variance_before": before,
        "variance_after": after,
        "porch_us": length_us,
        "specified_us": FRONT_PORCH_US[key],
        "time_constants_spanned": length_us
        / max(float(time_constant_us), 1e-9),
        "settled_fraction_remaining": float(np.exp(
            -length_us / max(float(time_constant_us), 1e-9))),
        "why": ("the front porch is blanking plus the relaxation left over "
                "from the preceding active line; it spans only about 1.4 "
                "time constants, so it never settles and its content "
                "dependence is a modelled term rather than a disqualification"),
    }


# --------------------------------------------------------------------------
# The edge and the tail are ONE channel, and a real pole cannot carry a ring
# --------------------------------------------------------------------------


def relaxation_or_ringing(tail_ire, time_us,
                          decays_us: Optional[Sequence[float]] = None,
                          frequencies_hz: Optional[Sequence[float]] = None
                          ) -> Dict[str, object]:
    """IS THE BACK-PORCH TAIL A RELAXATION OR A RING? Fit both and compare.

    Ethan: *"I see a missing dimension, or possibly a complex pair is not
    interconnected, in the picture, which is showing as a ringing and noise
    profile on the sync pulse's response. I am still seeing distortion in the
    back porch that should be able to be reversed."*

    THE PAIR THAT IS NOT INTERCONNECTED IS THE EDGE AND THE TAIL, and they
    are one object. The sync rise is a step; the back porch is where that step
    SETTLES. So the response measured from the edge and the response measured
    from the tail are the same `H(f)`, and anything true of one must be true
    of the other.

    THEY HAVE NOT BEEN FITTED WITH THE SAME MODEL. The sync edge is
    characterised with ring POLES - complex pairs, a frequency and a decay -
    which is what a ringing channel needs. The back-porch tail has been fitted
    with a single REAL exponential, `A exp(-t/tau) + C`. A real pole cannot
    represent a ring: it has no frequency, so an oscillatory component of the
    channel is invisible to it and comes back as whatever real decay best
    absorbs the residue. That is a missing dimension in the literal sense -
    the model has one parameter where the physics has two - and it would show
    exactly as Ethan describes: ringing visible on the edge, distortion left
    on the porch, and a tail fit that looks acceptable while removing the
    wrong thing.

    AND ON THE REAL TAPE IT IS A RING. Run on the ringing lane's corrected
    export, home tape, 52 samples over 3.56 us, on the REMAINDER left after
    the relaxation has been removed:

                    real pole rms   ring rms   improvement   ring frequency
        head A         0.1660        0.1042      +37.2%        2.290 MHz
        head B         0.2000        0.1410      +29.5%        2.213 MHz

    with fitted decays of 0.46 and 0.54 us and amplitudes of +0.717 and
    +0.708 IRE. FOUR INDEPENDENT REASONS TO BELIEVE IT, because a ring fitted
    to a small residual is exactly the kind of claim that is usually an
    artefact:

      - the noise null. Fitting the same ring model to noise of the same
        length and rms buys a median improvement of +7.2 per cent and at most
        +15.9 over sixty draws. The measurement is at +37.2 and +29.5.
      - on noise the fitted frequency is UNIFORM over the whole scan, 0.28 to
        7.00 MHz - there is no preferred value for chance to land on. The two
        heads land at 2.290 and 2.213 MHz, agreeing to 3.5 per cent.
      - the amplitudes agree to 1.3 per cent between heads.
      - THE TWO HEADS' REMAINDERS CORRELATE AT r = +0.9492. Two independently
        measured residuals agreeing that closely are one object.

    Head-common at r = +0.95 places it before the head, like the relaxation
    it sits inside. It also explains the lane's report that a decay scan on
    this remainder "rails at its bound with the sign reversed": a real pole
    has no frequency, so asked to fit an oscillation it runs to the end of
    its range and inverts trying to follow the first half-cycle.

    So this fits BOTH and reports which the data prefers:

        a real pole      A exp(-t/tau) + C          two free, tau scanned
        a complex pair   A exp(-t/tau) cos(wt + p) + C
                                                    three free, tau and w
                                                    scanned

    Both are linear in their amplitudes once the decay and frequency are
    fixed, so a scan with least squares at each step is exact - no optimiser,
    and no logarithm to reweight the quiet samples. The comparison is on
    residual rms with the parameter count stated, because a model with more
    freedom fits better by construction and the question is whether it fits
    ENOUGH better.
    """
    t = np.asarray(time_us, dtype=np.float64).ravel()
    y = np.asarray(tail_ire, dtype=np.float64).ravel()
    if t.size != y.size or t.size < 8:
        return {"decided": False, "why": "too few samples to fit either"}
    span = float(t.max() - t.min())
    if not span > 0:
        return {"decided": False, "why": "the window has no duration"}

    decays = (np.asarray(decays_us, dtype=np.float64) if decays_us is not None
              else np.geomspace(0.05 * span, 4.0 * span, 120))
    # frequencies the window can actually carry: at least one full cycle in
    # the window, and at most the Nyquist of the sample spacing
    step = float(np.median(np.diff(t))) if t.size > 1 else span
    frequencies = (np.asarray(frequencies_hz, dtype=np.float64)
                   if frequencies_hz is not None
                   else np.linspace(1.0 / span, 0.5 / max(step, 1e-9), 90) * 1e6)

    def best_fit(columns_for):
        best = None
        for decay in decays:
            for frequency in (frequencies if columns_for == "ring"
                              else [0.0]):
                envelope = np.exp(-t / max(decay, 1e-9))
                if columns_for == "ring":
                    phase = 2.0 * np.pi * frequency * t * 1e-6
                    design = np.column_stack([envelope * np.cos(phase),
                                              envelope * np.sin(phase),
                                              np.ones_like(t)])
                else:
                    design = np.column_stack([envelope, np.ones_like(t)])
                coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
                residual = float(np.sqrt(np.mean(
                    (y - design @ coefficients) ** 2)))
                if best is None or residual < best["rms"]:
                    best = {"rms": residual, "decay_us": float(decay),
                            "frequency_hz": float(frequency),
                            "coefficients": coefficients,
                            "parameters": design.shape[1] + 1}
        return best

    relaxation = best_fit("pole")
    ringing = best_fit("ring")
    improvement = (1.0 - ringing["rms"] / relaxation["rms"]
                   if relaxation["rms"] > 0 else 0.0)
    # a ring has two more free parameters, so it must beat the pole by more
    # than the fraction that alone buys: (n - p_pole) / (n - p_ring) - 1
    freedom = float(np.sqrt(max(t.size - relaxation["parameters"], 1)
                            / max(t.size - ringing["parameters"], 1))) - 1.0
    return {
        "decided": True,
        "relaxation": {"rms": relaxation["rms"],
                       "tau_us": relaxation["decay_us"],
                       "amplitude": float(relaxation["coefficients"][0]),
                       "settled": float(relaxation["coefficients"][-1])},
        "ringing": {"rms": ringing["rms"],
                    "tau_us": ringing["decay_us"],
                    "frequency_hz": ringing["frequency_hz"],
                    "amplitude": float(np.hypot(ringing["coefficients"][0],
                                                ringing["coefficients"][1])),
                    "settled": float(ringing["coefficients"][-1])},
        "improvement": improvement,
        "free_parameter_allowance": freedom,
        "prefers_ringing": bool(improvement > max(3.0 * freedom, 0.05)),
        "why": ("the edge and the tail are one channel, so a ring visible on "
                "the edge must be present in the tail; a single real pole has "
                "no frequency and cannot carry one, so it absorbs the ring "
                "into whatever real decay fits least badly"),
    }
