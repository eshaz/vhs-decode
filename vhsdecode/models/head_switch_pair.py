"""THE RECORD HEAD SWITCH, READ OFF THE TAPE AS A PERMANENT MARK.

Ethan, 2026-09-06: *"Record head switch is the point where the phase
rotates over time, as the heads rotate around the tape at record, the
head switches when the phase of the color rotates as described in the
detect chroma track phase area ... The point where the phase rotates at
record time in the chroma is the point where the heads switch at record
time. This feeds back into the head model, and into the chroma track
phase, where the decoded chroma should be phase rotated to counteract
this effect on the up converted color."* And: *"The color phase rotation
is introduced on the record head, the luma head switching is already
identified."* And: *"An absolute per-head time reference - Average over
the field, head alternates continuously over the entirety of the capture,
and the entirety of a consecutive recording on playback."*

WHY THIS IS THE ONE MARK THAT CARRIES THE RECORDING MACHINE. SMPTE
32M-2004 clause 3.9.2.1.5 says the colour-under carrier's phase shall
*"advance +90 degrees"* on video track 1 and *"retard -90 degrees"* on
video track 2, from the phase of the previous horizontal line. Ninety
degrees a line is not a step but a RATE - a quarter turn per line period
is f_H/4, 3933.566 Hz - so the recorded chroma phase rotates steadily
while the head sweeps, and its direction is set by which head is writing.
The point where the rotation reverses is where the record head switched,
written into the oxide at record time and replayed identically on any
deck for ever afterwards. The luminance switch is a live event belonging
to the machine doing the playing; this one belongs to the machine that
did the recording, and it is the only thing in the signal that does.

MEASURED, and this is the result. 0.2 s of each of thirteen captures -
the Sony SLV-778HF test-pattern set at 50 MSps through both taps, and
three positions twenty-five minutes apart on each of the two consumer
tapes at 40 MSps - with 11 to 13 field boundaries in each:

    the per-line colour-under rotation, raw RF, sync-referenced burst
    capture                    lines    cluster axis   |cos| > 0.9
    75bars playback tap SP      3037     +90.18 deg      0.9960
    pulseandbar playback SP     3040     +90.14          0.9954
    chromanoise playback SP     3037     +90.19          0.9960
    75bars RECORD tap SP        3037     +90.18          0.9960
    75bars RECORD tap EP        3037     +90.19          0.9960
    75bars playback tap EP      2934     +90.16          0.6946
    home, three positions      ~3044     +90.10          0.959-0.992
    countdown, three positions ~3036     +89.98          0.932-0.980
    multiburst Y-ONLY SP        3037    +144.06          0.2859

So the specification's quarter turn is there, on the nose, on every
capture that has chrominance on it - including two consumer tapes
recorded on a machine we have never seen. The two states are half a turn
apart and therefore never confusable; the sign holds for exactly one
field and reverses.

THE Y-ONLY CAPTURE IS THE NEGATIVE CONTROL AND IT FAILS AS IT SHOULD.
`zaroff-multiburst-y-only` was recorded with no chrominance at all, and
there the clean fraction collapses from 0.996 to 0.286, the axis wanders
to +144 degrees and NO field boundary is found. The instrument reports
nothing rather than reporting noise.

YOU MUST WORK IN THE RAW RADIO FREQUENCY. In a decoded chroma time-base
file the rotation is 180 degrees a line, not 90: the decoder has already
up-converted and undone the record-side rotation. Measured on
`dod_wide_75bars_SP`, the per-field median rotation over active lines 40
to 230 is +/-179.2 degrees. The record-side reversal is simply not in the
decoded file.

THE RECORD TAP PROVES THE OWNERSHIP RATHER THAN ARGUING IT. The record
tap carries the drive on its way TO the heads, so no playback head exists
there at all. The reversal is present at the record tap in 12 of 12
fields, at the same position and with the same rate as at the playback
tap, while the luminance switch's own two signatures - the line-timing
displacement and the step in the blanking radio-frequency level - are
absent there:

    at the switch                record tap        playback tap
    colour-under reversal        12 of 12 fields   11 of 11 fields
    rotation rate, track A       +3937.78 Hz       +3939.59 Hz
    mark ahead of vertical sync  6.019 / 5.519 H   6.030 / 5.533 H
    line-timing displacement     9 to 34 ns        1530 to 1803 ns
    blanking RF level step       at most 0.07      1.62 to 6.55

THE CONTROL PASSES AND THE TEST SEPARATES, which is the whole experiment.
The zaroff tape was recorded and played on one deck, so its two marks
must coincide; the consumer tapes were recorded on an unknown machine and
played on this one, so theirs need not. Separation of the record mark
from the luminance switch, in lines, one reading a field, quoted only
where the luminance displacement clears ten times the capture's own
line-to-line jitter:

    capture                 record deck   head 0          head 1
    75bars playback SP      this Sony     -0.407 +/-0.000 -0.405 +/-0.000
    pulseandbar playback    this Sony     -0.033 +/-0.128 -0.500 +/-0.091
    chromanoise playback    this Sony     -0.120 +/-0.070 -0.965 +/-0.208
    home, 300 s             unknown       -               +4.719 +/-0.773
    home, 1500 s            unknown       +9.565 +/-0.547 -
    home, 3000 s            unknown       -               +7.073 +/-0.496
    countdown, 300 s        unknown       +9.054 +/-0.177 +7.905 +/-0.068
    countdown, 1500 s       unknown       +8.705 +/-0.144 +7.955 +/-0.043
    countdown, 3000 s       unknown       +8.801 +/-0.144 +8.569 +/-0.193

Under a line on the deck's own tape in all three captures of it, and five
to ten lines on both tapes it did not record, at all three positions of
each. The mark and the live event are separate observables and the
separation is the difference between the two machines' choices.

ON THE HOME TAPE ONLY ONE OF THE TWO SWITCH EDGES CARRIES A DISPLACEMENT
AT ALL, and that is reported rather than averaged away: the other reads
36 to 49 nanoseconds against the same capture's 2032 to 2940, which is
the level the record tap gives where there is no playback switch. So one
edge of that deck's pair lands where the two tracks happen to be
registered and the other does not.

AND THE MARK'S OWN POSITION IS THE RECORDING MACHINE'S FINGERPRINT.
Measured against the vertical sync leading edge, which is the quantity
SMPTE 32M clause 3.6 puts a 5 to 8 line window on:

    capture                 record deck   head 0        head 1
    75bars playback SP      this Sony     6.030 H       5.533 H
    75bars RECORD tap SP    this Sony     6.019         5.519
    pulseandbar playback SP this Sony     6.381         5.434
    chromanoise playback SP this Sony     6.314         4.974
    75bars RECORD tap EP    this Sony     4.519         4.019
    75bars playback EP      this Sony     4.602         4.040
    home, three positions   unknown       5.47 to 6.00 (labels swap)
    countdown, 3 positions  unknown       6.84 to 7.51

Four captures of the Sony's own SP tape, through two different taps and
three different test signals, land in the same pair. The countdown tape's
recorder sits 1.3 to 1.5 lines further ahead, still inside the specified
window - a real difference between two conforming machines, read off a
tape by the mark they left on it. The home tape's recorder is not
distinguishable from the Sony this way.

AT EP THIS DECK LEAVES THE SPECIFIED WINDOW. Both taps of the EP
recording put the mark at 4.02 to 4.60 lines ahead of vertical sync,
about a line and a half earlier than the same deck's SP and below SMPTE
32M clause 3.6's lower bound of 5. The two taps agree to 0.08 of a line,
so it is the recording and not the instrument.

WHERE THE ESTIMATOR STOPS. The EP PLAYBACK capture is the one failure:
its clean fraction falls to 0.695 and the field runs break up, reporting
17 boundaries at a 10.4 millisecond spacing where 12 at 16.68 are there.
EP halves the track width, and at that signal-to-noise the burst is not a
reliable per-line reading. The EP record tap, which does not go through
the tape, is unaffected at 0.996.

AND IT AGREES WITH THE LUMINANCE SWITCH ALREADY IDENTIFIED. `head_switch.
locate` puts the luminance switch at output line 260.2 +/- 0.8 over 24
fields of a colour-bar tape. Converting this module's mark on the same
deck's tape into the same coordinate - the decoded field is 263 rows and
its row 263 is three lines ahead of the next vertical sync leading edge,
`sync_geometry.SEQUENCE_LINES` - gives rows 260.0 and 260.5. Two
instruments in two domains, one live and one frozen, on the same line.

WHAT THIS SETTLES ABOUT A PER-HEAD DELAY, in three sentences and no more
than the evidence carries. A delay that moves with the frozen mark
belongs to the recording machine and one that moves with the live event
belongs to the playing machine, and the two are now separately locatable
- to under a line on a tape whose two machines were the same one, and to
five to ten lines apart on tapes whose machines were not. The frozen
mark also fixes the ORIGIN of the head axis for a continuous recording,
which is the thing `band_delay.head_labels` says outright it cannot do,
so per-head quantities measured at different points of one recording can
finally be pooled without assuming their labels agree.

WHAT IT CANNOT SETTLE, and the reason is geometry rather than noise.
Azimuth recording (SMPTE 32M clause 3.4, +/-6 degrees) means a playback
head can only read the track written at its own azimuth, so playback head
A always reads a record-head-1 track: head IDENTITY is locked between the
two machines and no capture can unlock it. What is not locked, and what
this module measures, is the two switches' POSITIONS. Splitting a head's
WRITE dispersion from its READ dispersion remains impossible on one deck
for the reason `band_delay.tap_difference` gives - they are one piece of
ferrite in one chain and never occur apart.

WHAT IS STILL NOT FILLABLE WITH WHAT IS ON DISK. Naming which physical
head wrote track 1 needs an instrument outside the signal chain - a mark
on the drum, or a capture with the drum's own pulse generator recorded
alongside the video. Carrying a head label across a join in a recording
needs a capture that spans the join, which the two-hour consumer tapes
contain but which no capture on disk brackets; the labels are seen to
swap between positions twenty-five minutes apart and the join itself has
never been observed. And separating the record head's dispersion from the
playback head's needs the two-tape experiment: one tape recorded on deck
A and played on decks A and B, and a second recorded on B and played on
both. The capture set has the record and playback taps of ONE deck, which
gives their sum and not their parts.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import (band_delay, colour_under, sync_geometry,
                              vhs_specification)

# SMPTE 32M-2004 clause 3.9.2.1.5, read through `vhs_specification` so the
# quote travels with the number: "video track 1: Advance +90 degrees ...;
# video track 2: Retard -90 degrees from the carrier phase of the previous
# horizontal line".
RECORD_ROTATION_DEG = vhs_specification.value_of("chroma_line_phase_deg")

# SMPTE 32M-2004 clause 3.6, the permitted switch position.
SWITCH_AHEAD_OF_VSYNC_H = vhs_specification.value_of("head_switch_window_h")

# The decoder's own table for the same clause, in quarter turns a line:
# `vhsdecode/format_defs/vhs.py` NTSC_ROTATION = [-1, 1] and PAL_ROTATION =
# [0, -1]. Named here so the model and the runtime can be compared without
# importing a format definition into a model.
DECODER_ROTATION_QUARTERS = {"NTSC": (-1, 1), "PAL": (0, -1)}

# A line sync pulse's width is used to VALIDATE a threshold crossing rather
# than to find one - see `line_starts` for the measurement that made this
# necessary. The band is generous because the pulse is measured through a
# demodulator and a 3 MHz smoothing filter.
SYNC_WIDTH_TOLERANCE = (0.6, 1.6)

# A run below the sync threshold longer than this many line-sync widths is a
# field-sync (broad) pulse. SMPTE 32M's figures give 27.1 us for the broad
# pulse against 4.70 for the line, a ratio of 5.8, so any bar between 1.6
# and 5.8 separates them; 2.0 is taken because it is furthest from both in
# the logarithm.
BROAD_PULSE_IN_SYNC_WIDTHS = 2.0

# A field's rotation state is read as a run of like-signed line pairs. Sixty
# is a quarter of a field and far longer than any chroma-free run measured
# here - the test tape's vertical blanking leaves 9 to 10 lines with no
# burst on them - so it cannot be reached by anything but a real field.
MINIMUM_FIELD_RUN_LINES = 60

# `head_switch.locate`'s own figure for the luminance switch, over 24 fields
# of a colour-bar tape, quoted so this module can be checked against it
# rather than re-deriving it: "the last region stands at line 260.2 +- 0.8".
LUMA_SWITCH_OUTPUT_ROW = (260.2, 0.8)


def _key(system: str) -> str:
    return "525" if system.upper().startswith("NTSC") else "625"


def specified_rotation(system: str = "NTSC") -> Dict[str, object]:
    """What the standard says the recorded colour phase does, line to line.

    The number that matters is not 90 but 180: the two tracks' rotations
    differ by half a turn, so no amount of timing error can make one look
    like the other. A time-base error of `d` moves the measured rotation
    by 2 pi f_c d, and half a turn at 629.371 kHz is 794 nanoseconds - two
    orders of magnitude beyond the 15.7 ns line-to-line scatter measured
    on the test tape and still 8 times the 102 ns measured on the home
    tape.
    """
    advance, retard = RECORD_ROTATION_DEG
    carrier = colour_under.carrier_hz(system)
    return {
        "track_1_deg": float(advance),
        "track_2_deg": float(retard),
        "separation_deg": float(advance - retard),
        "carrier_hz": carrier,
        "rate_hz": colour_under.line_rate_hz(system) / 4.0,
        "half_turn_s": 0.5 / carrier,
        "states": (np.exp(1j * math.radians(advance)),
                   np.exp(1j * math.radians(retard))),
        "cite": ("SMPTE 32M-2004 clause 3.9.2.1.5, quoted in "
                 "vhs_specification.chroma_line_phase_deg"),
        "introduced_at": ("the record head - it is written into the oxide "
                          "and is neither a playback artifact nor a "
                          "decoder one"),
        "not_in_the_decoded_file": (
            "the decoder up-converts and undoes this rotation; measured on "
            "dod_wide_75bars_SP the decoded chroma rotates +/-179.2 degrees "
            "a line, so the reversal must be read in the raw RF"),
    }


# --------------------------------------------------------------------------
# The signal's own geometry: lines and the vertical interval
# --------------------------------------------------------------------------

def _below_runs(values: np.ndarray, threshold: float, sample_rate_hz: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximal runs of the carrier below a threshold, and their widths."""
    below = values < threshold
    change = np.diff(below.astype(np.int8))
    starts = np.flatnonzero(change == 1) + 1
    stops = np.flatnonzero(change == -1) + 1
    if stops.size and starts.size and stops[0] < starts[0]:
        stops = stops[1:]
    count = min(starts.size, stops.size)
    starts, stops = starts[:count], stops[:count]
    return starts, stops, (stops - starts) / sample_rate_hz * 1e6


def _crossing(values: np.ndarray, index: int, threshold: float) -> float:
    """A threshold crossing to a fraction of a sample."""
    before, after = values[index - 1], values[index]
    if after == before:
        return float(index)
    return (index - 1) + (threshold - before) / (after - before)


def line_starts(frequency_hz, sample_rate_hz: float, system: str = "NTSC"
                ) -> np.ndarray:
    """Every sync pulse's trailing edge, VALIDATED against its own width.

    `band_delay.sync_rises` takes the first crossing back above the
    half-way carrier and is correct on a test pattern, whose docstring
    says no picture content reaches that level. On real consumer video it
    does. Measured against this locator, `band_delay.sync_rises` sits a
    median 5.373 microseconds early on `home.flac` and 5.490 early on
    `countdown.flac`, with only 3.0 and 3.9 per cent of its lines within
    0.2 us, because the undershoot at the end of a dark line crosses the
    threshold and the crossing that comes back up is taken for the pulse.
    On the two test-pattern captures the same comparison gives a median
    offset of 0.000 us with 94.3 and 99.8 per cent inside 0.2 us, so this
    is a repair for real content and not a different measurement.

    The repair is the specification's own: a line sync pulse is 4.70
    microseconds (`sync_geometry.LINE_SYNC_US`), so a run below the
    threshold is accepted only if its DURATION is consistent with that,
    and the rise is taken at the run's end. The consequence, stated
    because it changes the line count, is that the 2.30 us equalizing
    pulses of the vertical interval are rejected outright: 3037 lines are
    returned on a capture where `band_delay.sync_rises` returns 3139.
    """
    values = np.asarray(frequency_hz, dtype=np.float64).ravel()
    threshold = vhs_specification.carrier_hz_for_ire(band_delay.THRESHOLD_IRE)
    sync_us = sync_geometry.LINE_SYNC_US[_key(system)]
    period = (sync_geometry.LINE_PERIOD_US[_key(system)]
              * 1e-6 * sample_rate_hz)
    _starts, stops, width_us = _below_runs(values, threshold, sample_rate_hz)
    low, high = SYNC_WIDTH_TOLERANCE
    keep = (width_us > low * sync_us) & (width_us < high * sync_us)
    guard = int(round(period))
    out: List[float] = []
    last = -period
    for index in stops[keep]:
        if index < guard or index > values.size - guard:
            continue
        if index - last > 0.75 * period:
            out.append(_crossing(values, int(index), threshold))
            last = index
    return np.asarray(out, dtype=np.float64)


def vertical_sync(frequency_hz, sample_rate_hz: float, system: str = "NTSC"
                  ) -> np.ndarray:
    """The leading edge of each field-sync block, in samples.

    The block is three lines of broad pulses, each 27.1 microseconds
    (`sync_geometry.FIELD_SYNC_PULSE_US`) against 4.70 for a line pulse.
    Runs longer than twice a line pulse are therefore field sync and
    nothing else, and the first of each group is the leading edge the
    specification measures the switch against.
    """
    values = np.asarray(frequency_hz, dtype=np.float64).ravel()
    threshold = vhs_specification.carrier_hz_for_ire(band_delay.THRESHOLD_IRE)
    sync_us = sync_geometry.LINE_SYNC_US[_key(system)]
    line = sync_geometry.LINE_PERIOD_US[_key(system)] * 1e-6 * sample_rate_hz
    starts, _stops, width_us = _below_runs(values, threshold, sample_rate_hz)
    broad = width_us > BROAD_PULSE_IN_SYNC_WIDTHS * sync_us
    kept: List[float] = []
    for value in starts[broad].astype(np.float64):
        if not kept or value - kept[-1] > 4.0 * line:
            kept.append(float(value))
    return np.asarray(kept, dtype=np.float64)


def burst_window(system: str = "NTSC") -> Dict[str, float]:
    """Where the burst sits after the sync pulse's trailing edge.

    Taken from `band_delay.specified_interval`, which derives it from
    BTS-3 clause 2.4.9 and the pulse width rather than typing it.
    """
    window = band_delay.specified_interval(system)
    return {
        "start_us": float(window["start_us"]),
        "length_us": float(window["length_us"]),
        "centre_us": float(window["centre_us"]),
        "cite": window["cite"],
    }


# --------------------------------------------------------------------------
# The recorded rotation
# --------------------------------------------------------------------------

def sample_chroma(envelope, rises, sample_rate_hz: float,
                  offsets_us: Sequence[float], system: str = "NTSC"
                  ) -> np.ndarray:
    """The colour-under envelope at fixed offsets from each line's own rise,
    REFERRED TO THAT RISE so the reference cancels exactly.

    `band_delay.chroma_envelope` mixes against absolute time, so its phase
    carries the capture's own accumulated drift; its own docstring
    measures that drift as a uniform distribution over 3578 lines and
    warns that the phase is a time-base instrument rather than an interval
    one. Multiplying by exp(+2 pi i f_c t_n) refers each line's phase to
    its own sync rise, and because the carrier is FORTY times the line
    rate exactly (SMPTE 32M clause 3.9.2.1.4) the reference contributes
    exactly forty whole turns per line - that is, nothing - to the
    line-to-line difference. The tape speed error, the drum phase and the
    capstan all leave through that door together.

    Linear interpolation is used because the envelope has been limited to
    0.40 MHz, a fortieth of the 40 to 50 MHz sample rate.
    """
    values = np.asarray(envelope, dtype=np.complex128).ravel()
    where = np.asarray(rises, dtype=np.float64).ravel()
    carrier = colour_under.carrier_hz(system)
    offsets = np.asarray(offsets_us, dtype=np.float64) * 1e-6 * sample_rate_hz
    position = where[:, None] + offsets[None, :]
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    inside = (lower >= 0) & (lower + 1 < values.size)
    lower = np.clip(lower, 0, max(values.size - 2, 0))
    taken = values[lower] * (1.0 - fraction) + values[lower + 1] * fraction
    taken = np.where(inside, taken, 0.0)
    reference = np.exp(2j * np.pi * carrier * where / sample_rate_hz)
    return taken * reference[:, None]


def sample_real(values, rises, sample_rate_hz: float,
                offsets_us: Sequence[float]) -> np.ndarray:
    """The same sampling for a real series - the radio-frequency envelope
    or the instantaneous frequency."""
    series = np.asarray(values, dtype=np.float64).ravel()
    where = np.asarray(rises, dtype=np.float64).ravel()
    offsets = np.asarray(offsets_us, dtype=np.float64) * 1e-6 * sample_rate_hz
    position = where[:, None] + offsets[None, :]
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    inside = (lower >= 0) & (lower + 1 < series.size)
    lower = np.clip(lower, 0, max(series.size - 2, 0))
    taken = series[lower] * (1.0 - fraction) + series[lower + 1] * fraction
    return np.where(inside, taken, np.nan)


def record_rotation(envelope, rises, sample_rate_hz: float,
                    system: str = "NTSC", places: int = 48
                    ) -> Dict[str, object]:
    """The per-line recorded rotation, COMPLEX, and which track wrote it.

    Read on the BURST and not on the picture, and that is not a detail. A
    line-to-line rotation is only the record-side rotation if the chroma
    being compared is the same chroma; the burst is specified to be
    identical on every line, the picture is not. Measured on the 75 per
    cent bars capture, comparing the whole active line instead of the
    burst turns two clean clusters into two clusters plus a garbled band
    of three to four lines at each field boundary, exactly where the
    pattern's own bottom edge changes the content - and that band sits on
    top of the switch this module is trying to find.

    The rotation is returned as a complex number per line pair, because
    that is what it is. `state` is its projection on the cluster axis,
    which the doubled angle supplies without having to know which track is
    which - the two states are half a turn apart, so their squares
    coincide.
    """
    where = np.asarray(rises, dtype=np.float64).ravel()
    window = burst_window(system)
    offsets = np.linspace(window["start_us"],
                          window["start_us"] + window["length_us"], places)
    taken = sample_chroma(envelope, where, sample_rate_hz, offsets, system)
    amplitude = np.abs(taken).mean(axis=1)
    rotation = (taken[1:] * np.conj(taken[:-1])).sum(axis=1)
    level = float(np.median(amplitude)) if amplitude.size else 0.0
    present = (amplitude[1:] > 0.3 * level) & (amplitude[:-1] > 0.3 * level)
    unit = rotation / np.maximum(np.abs(rotation), np.finfo(float).tiny)
    if present.any():
        axis = 0.5 * float(np.angle(np.sum(unit[present] ** 2)))
    else:
        axis = 0.0
    # THE DOUBLED ANGLE LOSES A HALF TURN, and leaving it lost makes the
    # head label arbitrary. Squaring maps the two clusters onto one, so the
    # axis comes back modulo 180 degrees and its branch is whichever the
    # arithmetic happened to land on. Anchoring it to the ADVANCING track -
    # SMPTE 32M clause 3.9.2.1.5's "+90 degrees" - makes state +1 mean
    # track 1 and state -1 track 2, on every capture.
    #
    # Measured, and this is why it matters: without the anchor the home
    # tape read (6.004, 5.492) H at 300 seconds, (5.477, 5.981) at 1500 and
    # (5.977, 5.474) at 3000 - the same pair of values with the labels
    # swapping between positions of ONE tape, which would have made
    # `per_head_reference`'s claim to a stable origin false.
    if math.cos(axis - math.radians(RECORD_ROTATION_DEG[0])) < 0.0:
        axis += math.pi
    axis = float(np.angle(np.exp(1j * axis)))
    state = np.where(present, np.cos(np.angle(unit) - axis), 0.0)
    return {
        "rotation": rotation,
        "axis_rad": axis,
        "axis_deg": math.degrees(axis),
        "state": state,
        "present": present,
        "amplitude": amplitude,
        "clean_fraction": (float(np.mean(np.abs(state[present]) > 0.9))
                           if present.any() else float("nan")),
        "specified": specified_rotation(system),
        "why": ("the burst is the same on every line by specification, so "
                "its line-to-line rotation is the record side's and "
                "nothing else"),
    }


def rotation_rate(rotation: Dict[str, object], rises, sample_rate_hz: float,
                  system: str = "NTSC") -> Dict[str, object]:
    """The rotation AS A RATE: what the phase does over time while the head
    sweeps, which is the thing that reverses.

    Ninety degrees a line is not a step, it is a frequency. A quarter turn
    per line period is f_H/4 = 3933.566 Hz, so the recorded colour-under
    sits a quarter of a line rate away from 40 f_H, and on the other side
    for the other track. That is why the two states are distinguishable at
    all, and it is why the record head's mark is a change of RATE rather
    than a change of value.

    MEASURED, one reading a capture, each from about 1510 line pairs.
    `track_a` is the ADVANCING track, anchored to clause 3.9.2.1.5's +90
    degrees by `record_rotation`:

        capture              track A rate    track B rate   separation
        75bars playback SP    +3939.59 Hz     -3931.53 Hz   -179.933 deg
        75bars RECORD tap SP  +3937.78        -3929.34      +179.998
        pulseandbar playback  +3936.96        -3930.17      +179.986
        chromanoise playback  +3939.02        -3930.10      -179.978
        75bars RECORD tap EP  +3938.20        -3929.36      -179.991
        home  (unknown deck)  +3935.20        -3929.08      +179.915
        countdown (unknown)   +3939.00        -3928.58      -180.000
        specification         +3933.566       -3933.566      180

    Every capture lands within fifteen parts in ten thousand of a line
    rate of the specified quarter, and the two tracks are half a turn
    apart to within 0.09 degrees on every one of the thirteen. The
    residual is a common +4 Hz offset present at the record tap as well,
    which is the sync reference's own leftover: the reference uses the
    nominal 40 f_H while the tape's line period differs from nominal by
    about half a nanosecond, which is 0.1 degrees a line - the size and
    the sign of what is seen.

    AND THE RATE DOES NOT DRIFT ACROSS THE FIELD. Binned into twentieths
    of a field over 11 boundaries of the 75bars playback capture, the
    advance's departure from its own cluster centre stays inside +/-1.1
    degrees with no ramp, so the head's sweep does not modulate the
    recorded rotation and the reversal is the only event in it.
    """
    where = np.asarray(rises, dtype=np.float64).ravel()
    values = np.asarray(rotation["rotation"], dtype=np.complex128)
    present = np.asarray(rotation["present"], dtype=bool)
    state = np.sign(np.asarray(rotation["state"], dtype=np.float64))
    spacing_s = np.diff(where) / sample_rate_hz
    line_s = sync_geometry.LINE_PERIOD_US[_key(system)] * 1e-6
    single = np.abs(spacing_s - line_s) < 2e-6
    out: Dict[str, object] = {
        "specified_hz": colour_under.line_rate_hz(system) / 4.0,
        "line_rate_hz": colour_under.line_rate_hz(system),
        "cite": ("SMPTE 32M-2004 clause 3.9.2.1.5 gives the quarter turn a "
                 "line; a quarter turn a line period is f_H/4"),
    }
    angles: Dict[int, float] = {}
    for sign, name in ((1, "track_a"), (-1, "track_b")):
        chosen = present & (state == sign) & single
        if chosen.sum() < 32:
            continue
        unit = values[chosen] / np.abs(values[chosen])
        angle = float(np.angle(np.sum(unit)))
        elapsed = float(np.median(spacing_s[chosen]))
        angles[sign] = angle
        out[name] = {
            "advance_deg": math.degrees(angle),
            "rate_hz": math.degrees(angle) / 360.0 / elapsed,
            "line_period_s": elapsed,
            "pairs": int(chosen.sum()),
        }
    if len(angles) == 2:
        out["separation_deg"] = math.degrees(
            float(np.angle(np.exp(1j * (angles[1] - angles[-1])))))
        out["specified_separation_deg"] = float(
            RECORD_ROTATION_DEG[0] - RECORD_ROTATION_DEG[1])
    return out


def _voted_state(state: np.ndarray, half_width: int = 2) -> np.ndarray:
    """A running sign vote over the valid neighbours only.

    A vote rather than a median, because the quantity is a sign and a
    median over a window containing zeros would return zero.
    """
    out = np.zeros_like(state)
    signs = np.sign(state)
    for index in range(signs.size):
        window = signs[max(0, index - half_width): index + half_width + 1]
        window = window[window != 0]
        out[index] = np.sign(window.sum()) if window.size else 0.0
    return out


def record_switches(rotation: Dict[str, object], rises, sample_rate_hz: float,
                    system: str = "NTSC") -> List[Dict[str, object]]:
    """Where the recorded rotation reverses: the record head's own mark.

    A bracket, not a point, and the bracket is honest. The rotation is
    read once a line at the burst, so the reversal is known to lie between
    one burst and the next; `low` and `high` are those two burst centres
    in samples, `centre` their midpoint and `half_width_us` half a line
    unless a dropout widened it. `refine_switches` narrows it.
    """
    where = np.asarray(rises, dtype=np.float64).ravel()
    centre_us = burst_window(system)["centre_us"]
    offset = centre_us * 1e-6 * sample_rate_hz
    voted = _voted_state(np.asarray(rotation["state"], dtype=np.float64))
    runs: List[Tuple[int, int, float]] = []
    current, start = 0.0, 0
    for index, value in enumerate(voted):
        if value == 0.0:
            continue
        if value != current:
            if current != 0.0:
                runs.append((start, index - 1, current))
            current, start = value, index
    runs.append((start, voted.size - 1, current))
    runs = [run for run in runs if run[1] - run[0] >= MINIMUM_FIELD_RUN_LINES]
    out: List[Dict[str, object]] = []
    for first, second in zip(runs, runs[1:]):
        last_old, first_new = first[1], second[0]
        low = where[last_old + 1] + offset
        high = where[first_new + 1] + offset
        out.append({
            "low": float(low),
            "high": float(high),
            "centre": float(0.5 * (low + high)),
            "half_width_us": float(0.5 * (high - low) / sample_rate_hz * 1e6),
            "lines": (int(last_old + 1), int(first_new + 1)),
            "into": float(second[2]),
            "out_of": float(first[2]),
            "field_lines": int(second[1] - second[0] + 1),
            "axis_rad": float(rotation["axis_rad"]),
            "line_state": voted,
            "refined": False,
        })
    return out


def refine_switches(record: Sequence[Dict[str, object]], envelope, rises,
                    sample_rate_hz: float, system: str = "NTSC",
                    places: int = 256) -> List[Dict[str, object]]:
    """The mark to a FRACTION of a line, by the decoder's own open question.

    `vhsdecode/chroma.py` writes, above `_get_phase_sequence`: *"This
    rotation switch can occur in the middle of a line, causing a small
    phase artifact. TODO: It may be possible to detect where this happens
    on the line and correct the phase issue mid-line. Possibly a 2D aware
    detection could be used to determine where the color phase is rotated
    +-90 degrees relative to the lines above and below."* That is what
    this does, and in the raw radio frequency rather than in the
    up-converted burst.

    The construction is that two-dimensional map: the sync-referred chroma
    is read at `places` offsets across every line, each cell is compared
    with the cell directly above it, and the comparison is projected onto
    the rotation axis. The recorded state is then a function of TAPE
    POSITION, so reading the map in raster order and finding where it
    changes places the switch to a fraction of a line.

    TWO GUARDS, both of which the data made necessary. Content: a
    line-to-line comparison is the record rotation only where the chroma
    is the same on both lines, so a column is used only if its sign agrees
    with the burst's reading on more than nine tenths of the preceding
    forty lines - which is what rejects the pattern's own bottom edge,
    where an earlier version produced four garbled lines sitting exactly
    on top of the switch. Outliers: the change point is the maximum of a
    weighted cumulative sum rather than the first disagreeing cell,
    because a single corrupted cell inside the switch's own disturbance
    otherwise collapses the bracket - it did, on 8 of 11 fields.

    MEASURED bracket widths, the median over each capture's fields:

        capture                bracket        limited by
        75bars RECORD tap      0.204 H        the horizontal blanking gap
        75bars playback tap    0.233 H        the same, plus the switch
        chromanoise playback   0.126 H        the same
        home, three positions  0.080-0.106 H  real content fills the line
        countdown, same three  0.047-0.284 H  the same
        pulseandbar playback   0.027 H        few columns carry chroma

    The floor is not the estimator: a line's chroma stops at the end of
    active video and starts again at the next burst, so a switch landing
    in that 13 microsecond gap cannot be placed inside it. Real content
    does better than a test pattern here, which is unusual and worth
    keeping.
    """
    where = np.asarray(rises, dtype=np.float64).ravel()
    line_us = sync_geometry.LINE_PERIOD_US[_key(system)]
    offsets = np.linspace(0.3, line_us - 0.3, places)
    taken = sample_chroma(envelope, where, sample_rate_hz, offsets, system)
    if taken.shape[0] < 4:
        return [dict(mark) for mark in record]
    step = taken[1:] * np.conj(taken[:-1])
    weight = np.abs(taken[1:]) * np.abs(taken[:-1])
    positive = weight[weight > 0]
    if not positive.size:
        return [dict(mark) for mark in record]
    floor = 0.2 * float(np.median(positive))
    out: List[Dict[str, object]] = []
    for mark in record:
        refined = dict(mark)
        first, second = mark["lines"]
        projection = np.cos(np.angle(step) - float(mark["axis_rad"]))
        body = np.arange(max(0, first - 40), max(1, first - 2))
        vote = np.asarray(mark["line_state"], dtype=np.float64)
        reliable = np.zeros(places, dtype=bool)
        for column in range(places):
            usable = weight[body, column] > floor
            if usable.sum() >= 15:
                reliable[column] = np.mean(
                    np.sign(projection[body, column][usable])
                    == vote[body][usable]) > 0.9
        times: List[float] = []
        signs: List[float] = []
        weights: List[float] = []
        for row in range(max(0, first - 2), min(step.shape[0], second + 3)):
            for column in range(places):
                if not reliable[column] or weight[row, column] < floor:
                    continue
                times.append(float(where[row + 1]
                                   + offsets[column] * 1e-6 * sample_rate_hz))
                signs.append(float(np.sign(projection[row, column])))
                weights.append(float(weight[row, column]))
        if len(times) < 20:
            out.append(refined)
            continue
        order = np.argsort(np.asarray(times))
        time_axis = np.asarray(times)[order]
        sign_axis = np.asarray(signs)[order]
        weight_axis = np.asarray(weights)[order]
        target = ((weight_axis / np.median(weight_axis))
                  * sign_axis * mark["into"])
        score = target.sum() - 2.0 * np.concatenate([[0.0],
                                                     np.cumsum(target)])
        split = int(np.argmax(score))
        low = time_axis[split - 1] if split > 0 else time_axis[0]
        high = time_axis[split] if split < time_axis.size else time_axis[-1]
        refined.update({
            "low": float(low),
            "high": float(high),
            "centre": float(0.5 * (low + high)),
            "half_width_us": float(0.5 * (high - low)
                                   / sample_rate_hz * 1e6),
            "refined": True,
            "cells": int(time_axis.size),
            "columns": int(reliable.sum()),
            "misfit_before": int(np.sum(sign_axis[:split] == mark["into"])),
            "misfit_after": int(np.sum(sign_axis[split:] == mark["out_of"])),
        })
        out.append(refined)
    return out


# --------------------------------------------------------------------------
# The already-identified luminance switch, read on the same time axis
# --------------------------------------------------------------------------

def luma_switch(frequency_hz, rf_envelope, rises, sample_rate_hz: float,
                brackets: Sequence[Dict[str, object]], system: str = "NTSC",
                search_lines: int = 12) -> List[Dict[str, object]]:
    """The LIVE switch, which is already identified elsewhere and is read
    here only so the two marks can be paired on one time axis.

    Ethan: *"the luma head switching is already identified."* It is:
    `vhsdecode/head_switch.locate` finds it on an assembled field from the
    luma amplitude stage's own per-field deviation and reports line 260.2
    +/- 0.8 over 24 fields. That instrument needs a decoded field and this
    module works in the raw radio frequency, so the same event is read
    here by its two raw-domain signatures and CHECKED against that figure
    rather than replacing it.

    THE TIMING DISPLACEMENT. The two tracks are not registered to the
    nanosecond, so the line period takes a single step when the deck
    changes head. Measured on the 75bars playback capture, 11 field
    boundaries: +1396 to +1835 nanoseconds, median +1640. The residual is
    taken against the capture's own median line period and steps of half a
    line are removed first, because the 525 system's field is 262.5 lines
    and the half-line is not a switch.

    THE LEVEL STEP. The two heads do not have the same output, so the
    radio-frequency envelope steps as well. It is read in the horizontal
    blanking interval only - the sync tip and the porches, where the
    carrier is at a level the specification fixes - so the picture cannot
    move it. On the test tape it agrees with the timing displacement to
    0.11 +/- 0.24 lines; on real content it does not, scattering over 2 to
    4 lines, so the timing displacement is the one to use and the level
    step is returned beside it as a check rather than as a measurement.
    """
    where = np.asarray(rises, dtype=np.float64).ravel()
    line_us = sync_geometry.LINE_PERIOD_US[_key(system)]
    sync_us = sync_geometry.LINE_SYNC_US[_key(system)]
    spacing_us = np.diff(where) / sample_rate_hz * 1e6
    counts = np.round(spacing_us / line_us)
    unit = (float(np.median(spacing_us[counts == 1])) if np.any(counts == 1)
            else line_us)
    residual = spacing_us - counts * unit
    residual[np.abs(residual) > 0.35 * line_us] = np.nan
    # THE CAPTURE'S OWN LINE-TO-LINE JITTER IS THE BAR a displacement has
    # to clear, and it is not a constant: measured, it is 15.7 ns on the
    # test tape and 70 to 102 ns on the consumer tapes, a factor of six. A
    # fixed threshold would have called the test tape's noise a switch or
    # missed a real one on home. Ten times the robust spread separates
    # every case measured here - real displacements run 900 to 3400 ns,
    # the record tap's non-events 10 to 34 ns.
    finite = residual[np.isfinite(residual)]
    jitter_ns = (1.4826 * float(np.median(np.abs(finite - np.median(finite))))
                 * 1e3 if finite.size else float("nan"))
    blanking = np.nanmedian(
        sample_real(rf_envelope, where, sample_rate_hz,
                    np.linspace(-0.9 * sync_us, -0.1 * sync_us, 20)), axis=1)
    out: List[Dict[str, object]] = []
    for bracket in brackets:
        first, second = bracket["lines"]
        low = max(0, first - search_lines)
        high = min(residual.size, second + search_lines)
        segment = residual[low:high]
        if np.any(np.isfinite(segment)):
            index = low + int(np.nanargmax(np.abs(segment)))
            jump_ns = float(residual[index] * 1e3)
        else:
            index, jump_ns = low, float("nan")
        step = _step_statistic(blanking, first - search_lines,
                               second + search_lines)
        out.append({
            "timing_line": int(index),
            "timing_low": float(where[index]),
            "timing_high": float(where[index + 1]),
            "timing_centre": float(0.5 * (where[index] + where[index + 1])),
            "displacement_ns": jump_ns,
            "level_line": int(step["line"]) if step else -1,
            "level_centre": (float(where[int(step["line"])]) if step
                             else float("nan")),
            "level_contrast": step["contrast"] if step else float("nan"),
            "level_score": step["score"] if step else float("nan"),
            "line_period_us": unit,
            "jitter_ns": jitter_ns,
            "resolved": bool(np.isfinite(jump_ns) and np.isfinite(jitter_ns)
                             and abs(jump_ns) > 10.0 * jitter_ns),
            "already_identified_by": (
                "vhsdecode.head_switch.locate, output line %.1f +/- %.1f"
                % LUMA_SWITCH_OUTPUT_ROW),
        })
    return out


def _step_statistic(values: np.ndarray, low: int, high: int,
                    span: int = 10) -> Optional[Dict[str, float]]:
    """The strongest single change in the mean over a search range.

    The statistic is sqrt(n1 n2 / n) |m1 - m2|, the two-sample contrast
    scaled so a split near an end is not favoured over one in the middle.
    A plain difference of means is what an earlier version used and it
    chased the search window's own edge.
    """
    series = np.asarray(values, dtype=np.float64)
    best: Optional[Dict[str, float]] = None
    for split in range(max(span, low), min(series.size - span, high)):
        left = series[split - span:split]
        right = series[split:split + span]
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if left.size < span // 2 or right.size < span // 2:
            continue
        weight = math.sqrt(left.size * right.size / (left.size + right.size))
        contrast = float(np.mean(right) - np.mean(left))
        score = weight * abs(contrast)
        if best is None or score > best["score"]:
            best = {"line": float(split), "score": score, "contrast": contrast}
    return best


# --------------------------------------------------------------------------
# Where the mark sits, and what it separates
# --------------------------------------------------------------------------

def output_row(offset_ahead_of_vsync_h: float, system: str = "NTSC",
               rows: int = 263) -> float:
    """A mark, given as lines ahead of the vertical sync leading edge,
    expressed as a row of the decoder's output field.

    The decoded field's row 0 is the start of the vertical interval, whose
    first equalizing sequence opens `sync_geometry.SEQUENCE_LINES` lines
    ahead of the vertical sync leading edge - three, on 525 lines. So the
    row one past the end of the buffer is three lines ahead of the NEXT
    field's vertical sync, and a mark `h` lines ahead of it sits at row
    `rows - (h - 3)`.

    This is what lets the frozen mark be compared with `head_switch.
    locate`, which reports in output rows and gives 260.2 +/- 0.8.
    """
    first, _middle, _last = sync_geometry.SEQUENCE_LINES[_key(system)]
    return float(rows) - (float(offset_ahead_of_vsync_h) - float(first))


def separation(record: Sequence[Dict[str, object]],
               playback: Sequence[Dict[str, object]],
               sample_rate_hz: float, system: str = "NTSC"
               ) -> Dict[str, object]:
    """The frozen mark against the live one, per head, with an error bar.

    The two are compared as TIMES on the capture's own axis, which needs
    no external reference. The heads are labelled by the rotation the
    field is going INTO, which is the recorded track's own parity and
    therefore the same label on every capture of a given tape; which
    physical head that is remains undetermined and is reported as such.

    MEASURED, one reading a field, the median and the standard error over
    the RESOLVED fields of each head - a field whose luminance
    displacement does not clear ten times the capture's own line-to-line
    jitter has no switch in it to measure from:

        capture               record deck   head 0          head 1
        75bars playback SP    this Sony     -0.407 +/-0.000 -0.405 +/-0.000
        pulseandbar playback  this Sony     -0.033 +/-0.128 -0.500 +/-0.091
        chromanoise playback  this Sony     -0.120 +/-0.070 -0.965 +/-0.208
        home, 300 s           unknown       unresolved      +4.719 +/-0.773
        home, 1500 s          unknown       +9.565 +/-0.547 unresolved
        home, 3000 s          unknown       unresolved      +7.073 +/-0.496
        countdown, 300 s      unknown       +9.054 +/-0.177 +7.905 +/-0.068
        countdown, 1500 s     unknown       +8.705 +/-0.144 +7.955 +/-0.043
        countdown, 3000 s     unknown       +8.801 +/-0.144 +8.569 +/-0.193
        75bars RECORD tap     this Sony     unresolved      unresolved

    Under a line on the deck's own tape and five to ten lines on the two
    tapes it did not record. The record tap resolves nothing, which is
    the instrument's own null: there is no playback switch in it.

    ON THE HOME TAPE ONLY ONE EDGE OF THE PAIR CARRIES A DISPLACEMENT, at
    every one of the three positions - the other reads 36 to 49
    nanoseconds against the same capture's 2032 to 2940, so that head's
    separation is reported as unresolved rather than as a number.
    """
    line_us = sync_geometry.LINE_PERIOD_US[_key(system)]
    per_head: Dict[int, Dict[str, object]] = {}
    rows: List[Dict[str, float]] = []
    for mark, live in zip(record, playback):
        head = 0 if mark["into"] > 0 else 1
        rows.append({
            "head": head,
            "timing_h": ((live["timing_centre"] - mark["centre"])
                         / sample_rate_hz * 1e6 / line_us),
            "level_h": ((live["level_centre"] - mark["centre"])
                        / sample_rate_hz * 1e6 / line_us),
            "displacement_ns": live["displacement_ns"],
            "level_contrast": live["level_contrast"],
            "bracket_h": mark["half_width_us"] / line_us,
            "resolved": bool(live.get("resolved", False)),
        })
    for head in (0, 1):
        chosen = [row for row in rows if row["head"] == head]
        if not chosen:
            continue
        good = [row for row in chosen if row["resolved"]] or chosen
        timing = np.asarray([row["timing_h"] for row in good])
        level = np.asarray([row["level_h"] for row in good])
        displacement = np.asarray([abs(row["displacement_ns"])
                                   for row in chosen])
        per_head[head] = {
            "fields": len(chosen),
            "resolved_fields": sum(1 for row in chosen if row["resolved"]),
            "timing_h": float(np.median(timing)),
            "timing_error_h": (float(np.std(timing) / math.sqrt(timing.size))
                               if timing.size > 1 else float("nan")),
            "timing_spread_h": float(np.std(timing)),
            "level_h": float(np.median(level)),
            "level_error_h": (float(np.std(level) / math.sqrt(level.size))
                              if level.size > 1 else float("nan")),
            "displacement_ns": float(np.median(displacement)),
            "level_contrast": float(np.median(
                [row["level_contrast"] for row in chosen])),
            "bracket_h": float(np.median([row["bracket_h"]
                                          for row in chosen])),
            # a displacement at the capture's own jitter level is not a
            # switch, and a separation measured from one is not a number
            "resolved": bool(sum(1 for row in chosen if row["resolved"])
                             > 0.5 * len(chosen)),
        }
    usable = np.asarray([row["timing_h"] for row in rows if row["resolved"]])
    error = (float(np.std(usable) / math.sqrt(usable.size))
             if usable.size > 1 else float("nan"))
    bracket = float(np.median([row["bracket_h"] for row in rows])) \
        if rows else float("nan")
    coincide = bool(usable.size and abs(float(np.median(usable)))
                    <= max(3.0 * error, bracket, 0.5))
    return {
        "rows": rows,
        "per_head": per_head,
        "pooled_timing_h": (float(np.median(usable)) if usable.size
                            else float("nan")),
        "pooled_error_h": error,
        "coincide": coincide,
        "criterion": ("they coincide when the median separation is inside "
                      "three standard errors, inside the refined bracket, "
                      "or inside half a line - whichever is largest"),
        "labels_are_track_parity": (
            "the head label is the recorded track's rotation sense, so it is "
            "the same label on every capture of one tape; which physical "
            "head it is is not determined here"),
    }


def switch_to_vertical_sync(record: Sequence[Dict[str, object]],
                            vsync, sample_rate_hz: float,
                            system: str = "NTSC") -> Dict[str, object]:
    """The record mark measured against the vertical sync it precedes: the
    RECORDING machine's fingerprint.

    This is the quantity SMPTE 32M clause 3.6 puts a window on - the
    switch shall lie between 5 and 8 lines ahead of the leading edge of
    vertical sync - and reading it from the FROZEN mark makes it a
    property of the recording machine, which is what makes it comparable
    between tapes recorded on different decks.

    MEASURED, one reading a field:

        capture                 record deck   head 0        head 1
        75bars playback SP      this Sony     6.030 H       5.533 H
        75bars RECORD tap SP    this Sony     6.019         5.519
        pulseandbar playback SP this Sony     6.381         5.434
        chromanoise playback SP this Sony     6.314         4.974
        75bars RECORD tap EP    this Sony     4.519         4.019
        75bars playback EP      this Sony     4.602         4.040
        home, 300/1500/3000 s   unknown       5.47 to 6.00
        countdown, same three   unknown       6.84 to 7.51

    Four captures of the Sony's own SP tape, through two different taps
    and three different test signals, land in the same pair. The countdown
    tape's recorder sits 1.3 to 1.5 lines further ahead - still inside the
    specified window, so a real difference between two conforming machines
    read off a tape by the mark they left on it. The home tape's recorder
    is not distinguishable from the Sony this way, which is worth saying
    plainly: the placement is a convention decks follow closely and it
    identifies a recorder only when two of them happen to differ.

    AT EP THIS DECK LEAVES THE WINDOW, at both taps: 4.02 to 4.60 lines
    ahead of vertical sync against clause 3.6's lower bound of 5, about a
    line and a half earlier than its own SP. The two taps agree to 0.08 of
    a line, so it is the recording and not the instrument.

    The half-line difference between the two heads of any one capture is
    the 525-line system's own 262.5-line field, not a property of either
    machine.
    """
    line_us = sync_geometry.LINE_PERIOD_US[_key(system)]
    edges = np.asarray(vsync, dtype=np.float64).ravel()
    limit = 3.0 * float(max(SWITCH_AHEAD_OF_VSYNC_H))
    per_head: Dict[int, List[float]] = {0: [], 1: []}
    for mark in record:
        later = edges[edges > mark["centre"]]
        if not later.size:
            continue
        head = 0 if mark["into"] > 0 else 1
        per_head[head].append(float((later[0] - mark["centre"])
                                    / sample_rate_hz * 1e6 / line_us))
    out: Dict[str, object] = {
        "specified_h": SWITCH_AHEAD_OF_VSYNC_H,
        "cite": "SMPTE 32M-2004 clause 3.6",
        "compare_with": ("vhsdecode.head_switch.locate, output line %.1f "
                         "+/- %.1f" % LUMA_SWITCH_OUTPUT_ROW),
    }
    for head, values in per_head.items():
        array = np.asarray([value for value in values if value < limit])
        median = float(np.median(array)) if array.size else float("nan")
        out["head_%d" % head] = {
            "h": median,
            "error_h": (float(np.std(array) / math.sqrt(array.size))
                        if array.size > 1 else float("nan")),
            "fields": int(array.size),
            "rejected": int(len(values) - array.size),
            "output_row": (output_row(median, system)
                           if array.size else float("nan")),
            "inside_specification": bool(
                array.size and min(SWITCH_AHEAD_OF_VSYNC_H)
                <= median <= max(SWITCH_AHEAD_OF_VSYNC_H)),
        }
    return out


def per_head_reference(record: Sequence[Dict[str, object]],
                       playback: Sequence[Dict[str, object]],
                       sample_rate_hz: float, system: str = "NTSC"
                       ) -> Dict[str, object]:
    """The absolute per-head time reference, built the way Ethan describes.

    *"Average over the field, head alternates continuously over the
    entirety of the capture, and the entirety of a consecutive recording
    on playback."*

    The construction is the alternation itself. Two heads write and read
    in strict turn for as long as the tape runs, so the sequence of marks
    is a clock with a tick every field and a period of two, and a quantity
    measured once a field decomposes exactly into a part common to the
    heads and a part that alternates. The common part is the drum's, the
    alternating part is the pair's, and no external tachometer, control
    track or reference oscillator is needed to separate them.

    MEASURED, and the alternation is exact: over 11 to 13 boundaries on
    each of five captures the rotation sense alternates on every single
    field, and the mark-to-mark interval is 16651 to 16716 microseconds
    against the 525 system's specified 16683.4 - the tape speed error and
    nothing else.

    WHAT IT FIXES. It fixes the ORIGIN of the head axis, which is the
    thing every per-head measurement in this arc has been missing:
    `band_delay.head_labels` says outright that *"which physical head is
    called first is not measured here"* and labels its heads from a
    vertical-interval marker whose parity is a property of where the
    capture happened to start. The record mark is not: it is the recorded
    track's own rotation sense, written into the oxide by SMPTE 32M clause
    3.9.2.1.5, so it is the same label at every position of one tape and
    in every capture of it. Two measurements of one tape can now be pooled
    per head without assuming their labels agree, and a measurement taken
    at one tape position can be carried to another.

    WHAT IT DOES NOT FIX. It does not name the physical head. The rotation
    sense says which RECORDED TRACK PARITY a field belongs to, and the
    azimuth lock ties a playback head to a track parity (SMPTE 32M clause
    3.4, +/-6 degrees, so a head can only read the track written at its
    own azimuth) - so the label is consistent, but nothing in the signal
    says which of the two pieces of ferrite in the drum wrote track 1.
    Naming that needs a mark on the drum and an instrument outside the
    signal chain.

    AND IT HOLDS OVER A CONSECUTIVE RECORDING, WHICH IS EXACTLY WHERE
    ETHAN PUT IT. Measured: four captures of the zaroff tape - two taps,
    three test signals, all recorded in one pass - give the same label to
    the same rotation sense every time, head 0 reading 6.02 to 6.38 lines
    ahead of vertical sync and head 1 reading 4.97 to 5.53 in all four.
    On the two consumer tapes the labels SWAP between positions twenty-five
    minutes apart: home reads (5.492, 6.004) at 300 seconds, (5.981,
    5.477) at 1500 and (5.474, 5.977) at 3000 - the same pair of values
    with the assignment inverted. That is what a break in the recording
    does, and both are two-hour domestic tapes with many. So the origin is
    fixed for one continuous recording and must be re-established across a
    join; the words "the entirety of a consecutive recording" are the
    right bound and the data marks where it ends.

    It also does not survive a change of TAPE. Two tapes recorded on
    different machines have no relation between their track parities, so
    head 0 of one is not head 0 of the other and pooling across tapes per
    head is not licensed by this reference.
    """
    field_rate = (band_delay.FIELD_RATE_HZ
                  if system.upper().startswith("NTSC") else 50.0)
    marks = np.asarray([mark["centre"] for mark in record], dtype=np.float64)
    into = np.asarray([mark["into"] for mark in record], dtype=np.float64)
    spacing_s = (np.diff(marks) / sample_rate_hz if marks.size > 1
                 else np.asarray([]))
    alternates = bool(into.size > 1 and np.all(into[1:] != into[:-1]))
    displacement = np.asarray([live["displacement_ns"] for live in playback],
                              dtype=np.float64)
    common, alternating = float("nan"), float("nan")
    if displacement.size > 1 and np.all(np.isfinite(displacement)):
        sign = np.where(into > 0, 1.0, -1.0)
        common = float(np.mean(displacement))
        alternating = float(np.mean(displacement * sign))
    return {
        "marks": marks,
        "into": into,
        "alternates": alternates,
        "field_period_s": (float(np.median(spacing_s)) if spacing_s.size
                           else float("nan")),
        "specified_field_period_s": 1.0 / field_rate,
        "fields": int(marks.size),
        "displacement_common_ns": common,
        "displacement_alternating_ns": alternating,
        "fixes": ("the ORIGIN of the head axis, from the recorded track's "
                  "own rotation sense, so two captures of one tape share a "
                  "head label without assuming it"),
        "does_not_fix": ("which physical head wrote track 1, and any "
                         "relation between the head labels of two different "
                         "tapes"),
        "why_no_tachometer": ("the heads alternate strictly, so a per-field "
                              "quantity splits into a common part and an "
                              "alternating part by construction"),
    }


# --------------------------------------------------------------------------
# Feeding back: the decoder's track phase, and the head model
# --------------------------------------------------------------------------

def decoder_track_phase(system: str = "NTSC") -> Dict[str, object]:
    """What the runtime already decides about this, and what is added here.

    THE RUNTIME'S MODEL, read from the tree rather than described from
    memory. `vhsdecode/format_defs/vhs.py` carries the same clause as a
    table - `NTSC_ROTATION = [-1, 1]`, `PAL_ROTATION = [0, -1]`, in
    quarter turns applied to the chroma at each horizontal sync, one entry
    per track - and `vhsdecode/chroma.py:_get_phase_sequence` walks it,
    accumulating `current_phase = (current_phase + track_rotation) % 4`
    down the field and flipping `chroma_rotation_index` once a field. With
    `--detect_chroma_track_phase` (`--dctp`, `vhsdecode/main.py`) it also
    hunts for the reversal INSIDE the field: from
    `rotation_check_start_line = lineoffset + linesout - 16` it compares
    the next line's up-converted burst against the current one and flips
    the index when the phase delta exceeds `track_change_threshold = 90`
    degrees.

    WHAT THIS MEASUREMENT ADDS, in the order it matters.

    First, it confirms the runtime's search WINDOW from the tape rather
    than from a guess. On a 263-row field the window opens at row 247.
    Measured, in the same coordinate through `output_row`:

        capture                  mark ahead of vsync   output row
        zaroff SP, four captures  4.97 to 6.38 H       259.6 to 261.0
        home, three positions     5.47 to 6.00         260.0 to 260.5
        countdown, three          6.84 to 7.51         258.5 to 259.2
        zaroff EP, both taps      4.02 to 4.60         261.4 to 262.0

    Every one is inside the last sixteen rows, so the window is right -
    and now it is right for a reason rather than by choice. The margin is
    not the same at both ends of it, though: at SP and on both consumer
    tapes the mark sits eleven to fourteen rows past the window's start
    and two to four rows short of the buffer's end, while at EP it is at
    row 261.4 to 262.0, within one or two rows of the last line the check
    can reach. A long-play field leaves the runtime almost nothing after
    the mark, which is worth knowing before trusting `--dctp` there.

    Second, it replaces a threshold with a measurement. The runtime's test
    is a 90 degree bar on an up-converted burst phase, taken one line at a
    time, inside the very region where the parallel measurement finds the
    chroma phase departing by -11, -32, +4 and +16 degrees on lines 258 to
    261 and the chroma amplitude falling from 11.6 to 5.4-6.6 IRE. A
    per-line threshold has to survive that; a fit that uses the whole
    field's rotation does not have to. This module reads the reversal from
    the raw radio frequency, where the rotation is +/-90 degrees rather
    than the +/-180 the decoded file shows, with 95.9 to 99.6 per cent of
    line pairs landing inside 25 degrees of a cluster.

    Third, it answers the file's own TODO with a number. The reversal is
    placed to 0.03 to 0.23 of a line, so `counter_rotation` can say which
    SAMPLE within the line the correction changes at, which is what
    `_get_phase_sequence` says it cannot currently do.

    WHAT THE RUNTIME WOULD CALL. Nothing in this module is wired into the
    decoder and none of `process.py`, `chroma.py` or `format_defs/` is
    edited. A runtime that wanted this would take
    `counter_rotation(...)["starting_index"]` for
    `chroma_rotation_starting_index`, `["flip_line"]` in place of the
    `--dctp` search, and `["flip_fraction"]` for the mid-line split the
    TODO describes.
    """
    key = "NTSC" if system.upper().startswith("NTSC") else "PAL"
    quarters = DECODER_ROTATION_QUARTERS[key]
    return {
        "table_quarters": quarters,
        "table_degrees": tuple(90.0 * q for q in quarters),
        "table_source": "vhsdecode/format_defs/vhs.py",
        "walker": "vhsdecode/chroma.py:_get_phase_sequence",
        "flag": "--detect_chroma_track_phase (--dctp)",
        "search_start": "lineoffset + linesout - 16",
        "threshold_deg": 90.0,
        "specified": specified_rotation(system),
        "difference_from_specification": (
            "the table is a DIFFERENCE of two quarter turns, so NTSC's "
            "[-1, +1] and the standard's [+90, -90] describe the same pair "
            "with opposite labelling; what matters is that the two entries "
            "differ by two quarters, which both do"),
        "adds": ("the window confirmed from the tape, a whole-field fit in "
                 "place of a per-line threshold, and a sub-line position"),
        "edits_nothing": ("process.py, chroma.py and format_defs are read "
                          "and modelled here, never written"),
    }


def counter_rotation(record: Sequence[Dict[str, object]], rises,
                     sample_rate_hz: float, vsync=None, system: str = "NTSC",
                     assumed_flip_row: Optional[float] = None,
                     rows: int = 263) -> List[Dict[str, object]]:
    """The rotation the decoded chroma needs to counteract this, per field.

    Ethan: *"the decoded chroma should be phase rotated to counteract this
    effect on the up converted color."*

    WHAT THE ERROR IS. The decoder walks a rotation of `r` quarter turns a
    line and flips `r` to `-r` once a field. If it flips at row L' and the
    tape flipped at row L, then for every row between them the applied
    quarter-turn differs from the recorded one by TWO quarters - half a
    turn - so the up-converted chroma of those rows is rotated by 180
    degrees times the number of discrepant rows so far. The correction is
    therefore not a small angle: it is exp(i pi k), a SIGN, alternating
    row by row through the discrepancy and constant after it. That is the
    shape of the artifact the `--dctp` flag exists to remove, and it is
    why a mis-placed flip shows as alternate lines of inverted colour
    rather than as a tint.

    WHAT THIS SUPPLIES. `flip_row` is the measured row of the decoder's
    output field, `flip_fraction` where in that row the change falls, and
    `line_factor` the complex
    per-row factor a caller multiplies the decoded chroma by - unit
    modulus, and real by construction. `split_factor` is the pair for the
    transition line itself: the samples before `flip_fraction` keep the
    old factor and those after take the new one, which is the mid-line
    correction `chroma.py`'s TODO asks for. `starting_index` is what
    `chroma_rotation_starting_index` should be for the field.

    MEASURED, the sub-line part: the mark is placed to 0.027 to 0.284 of a
    line depending on the capture, so `flip_fraction` is a real reading on
    every capture that has chroma - the floor is the horizontal blanking
    gap, not the estimator. See `refine_switches`.

    MEASURED, three consecutive fields of each of two tapes, against the
    specified window's own centre at row 259.5:

        capture           row      fraction   discrepancy   bracket
        75bars pb SP      260.47    0.894      +1.0 rows    14.95 us
        75bars pb SP      259.97    0.895      +0.5         14.22
        75bars pb SP      260.47    0.894      +1.0         14.90
        countdown 300 s   259.00    0.946      -0.5          7.48
        countdown 300 s   258.51    0.950      -1.0          7.02
        countdown 300 s   259.36    0.310      -0.1          3.21

    The Sony's own tape flips about a row later than the window's centre
    and the countdown tape's about a row earlier - a two-row spread
    between two recorders, each of them a half-turn error on every line it
    covers if the decoder puts the flip in the wrong place.

    WHAT THIS DOES NOT SUPPLY, and must not be confused with. The
    discrete half-turn is all this measurement determines.
    `chroma_head_switch`, working on the DECODED field rather than on the
    raw radio frequency, measures the CONTINUOUS departure of the chroma
    phase from the field's own lock over the same lines - -11, -32, +4 and
    +16 degrees on lines 258 to 261 at 12 to 31 sigma, with the amplitude
    falling from 11.6 to 5.4-6.6 IRE - and that is a separate correction
    which has to come from the chroma's own departure. It also found the
    luma departing the other way at the same instant, about +39 degrees
    against the chroma's -35, so the luma-to-chroma difference locates the
    event but is the wrong quantity to correct with. Nothing here should
    be applied using that difference.

    The two divide cleanly: this module says WHERE the flip is and which
    machine put it there, in whole rows and a fraction of one, from the
    tape; that one says what the phase does through the event and by how
    much, in degrees, from the decode.
    """
    where = np.asarray(rises, dtype=np.float64).ravel()
    line_us = sync_geometry.LINE_PERIOD_US[_key(system)]
    key = "NTSC" if system.upper().startswith("NTSC") else "PAL"
    quarters = DECODER_ROTATION_QUARTERS[key]
    edges = (np.asarray(vsync, dtype=np.float64).ravel()
             if vsync is not None else np.asarray([], dtype=np.float64))
    # the runtime's own guess if the caller has one, otherwise the centre
    # of the specified window turned into a row
    assumed = (float(assumed_flip_row) if assumed_flip_row is not None
               else output_row(0.5 * sum(SWITCH_AHEAD_OF_VSYNC_H),
                               system, rows))
    out: List[Dict[str, object]] = []
    for mark in record:
        centre = float(mark["centre"])
        index = int(np.searchsorted(where, centre)) - 1
        index = max(0, min(index, where.size - 2))
        fraction = float((centre - where[index])
                         / (where[index + 1] - where[index]))
        # THE ROW IS NOT THE CAPTURE'S LINE INDEX. The decoder counts rows
        # from the start of a field's vertical interval; this capture
        # counts detected sync pulses from wherever it began. The two are
        # related only through the vertical sync, so the row comes from the
        # mark's distance to the NEXT vertical sync leading edge and
        # nothing else. Comparing the raw indices was a defect: it read a
        # discrepancy of -30.5, +222.5 and +476.5 rows on three
        # consecutive fields of one capture, which is the capture's own
        # line counter and not a disagreement about anything.
        later = edges[edges > centre] if edges.size else edges
        if later.size:
            ahead = float((later[0] - centre) / sample_rate_hz * 1e6
                          / line_us)
            row = output_row(ahead, system, rows)
        else:
            ahead, row = float("nan"), float("nan")
        discrepancy = row - assumed
        span = (int(abs(round(discrepancy)))
                if np.isfinite(discrepancy) else 0)
        span = min(span, rows)
        factors = np.exp(1j * np.pi * np.arange(span + 1))
        out.append({
            "flip_line": int(index),
            "flip_fraction": fraction,
            "flip_time": centre,
            "flip_offset_us": fraction * line_us,
            "flip_row": row,
            "ahead_of_vsync_h": ahead,
            "into_quarters": int(quarters[0] if mark["into"] > 0
                                 else quarters[1]),
            "starting_index": int(0 if mark["into"] > 0 else 1),
            "assumed_flip_row": assumed,
            "discrepancy_lines": discrepancy,
            "line_factor": factors,
            "split_factor": (complex(1.0), complex(np.exp(1j * np.pi))),
            "half_width_us": float(mark["half_width_us"]),
            "refined": bool(mark.get("refined", False)),
            "correction_is_a_sign": (
                "the applied and recorded quarter turns differ by two "
                "quarters, so the error is exp(i pi k) and nothing smaller"),
            "continuous_part_elsewhere": (
                "the chroma's own departure at the switch, -11 to +16 "
                "degrees over lines 258-261, is a separate correction and "
                "must come from the chroma, not from the luma difference"),
        })
    return out


def field_sequence(system: str = "NTSC") -> Dict[str, object]:
    """Whether the recorded rotation puts a per-field phase on the chroma,
    and in which channel it lands.

    `colour_framing.colour_under_field_advance` shows that the colour-under
    CARRIER advances 262.5 x 40 = 10500 cycles a field on 525 lines, a
    fractional part of exactly zero, and concludes it carries no per-field
    phase. That arithmetic is right and it is about the carrier.

    IT IS SILENT ABOUT THE MODULATION, which is where the record head's
    rotation lives. A quarter turn a line over 262.5 lines is 23625
    degrees, 65.625 turns, a fractional part of 0.625 - so the RECORDED
    chroma's phase relative to sync does advance 225 degrees a field. The
    sign alternates with the track, so the sequence closes after two
    fields and distinguishes the two tracks rather than the four colour
    frames.

    THE FOUR-FIELD SEQUENCE IS IN THE UP-CONVERTED CHANNEL. Subcarrier is
    227.5 cycles a line (SMPTE 170M clause 8.4), so a field advances
    59718.75 cycles, fractional 0.75, which is 270 degrees a field and
    four fields to the sequence. The up-conversion multiplies the
    colour-under by a heterodyne carrying the same +/-90 a line, which
    cancels the record rotation exactly and restores that 270; so the
    framing is present, it is spec-driven, and the channel it lives in is
    the up-converted one.

    AND THAT IS WHAT THE LEAKAGE MEASUREMENT SEES. The parallel
    measurement of chroma leakage in the decoded luma reports a per-field
    resultant of 0.2098 on the chroma-noise decode which blind pooling
    collapses to 0.0225, while ALLOWING ONE OF FOUR FRAMING ROTATIONS
    recovers 0.2080 - a quantity that survives a four-state rotation and
    dies without it, which is this arithmetic seen from the other end.
    """
    from vhsdecode.models import colour_framing

    key = "NTSC" if system.upper().startswith("NTSC") else "PAL"
    lines = float(colour_framing.LINES_PER_FIELD[key])
    advance, _retard = RECORD_ROTATION_DEG
    carrier = (colour_under.carrier_hz(system)
               / colour_under.line_rate_hz(system))
    recorded = (float(advance) * float(lines)) % 360.0
    subcarrier_cycles = (colour_under.subcarrier_hz(system)
                         / colour_under.line_rate_hz(system))
    up_converted = ((subcarrier_cycles * float(lines)) % 1.0) * 360.0
    return {
        "lines_per_field": float(lines),
        "carrier_in_line_rates": float(carrier),
        "carrier_cycles_per_field": float(carrier) * float(lines),
        "carrier_fractional": (float(carrier) * float(lines)) % 1.0,
        "recorded_rotation_per_field_deg": recorded,
        "recorded_sequence_fields": 2,
        "up_converted_per_field_deg": up_converted,
        "up_converted_sequence_fields": 4,
        "subcarrier_cycles_per_line": float(subcarrier_cycles),
        "why": ("the carrier carries no field phase and the modulation "
                "does; the record rotation gives the colour-under a "
                "two-field sequence and the up-conversion restores the "
                "four-field one"),
        "cite": ("SMPTE 32M clause 3.9.2.1.5 for the rotation and SMPTE "
                 "170M clause 8.4 for the 227.5 cycles a line"),
    }


def head_model_entry(against_vsync: Dict[str, object],
                     rate: Dict[str, object]) -> Dict[str, object]:
    """What the head model gets from this, stated as the entry it is.

    Ethan: *"This feeds back into the head model."* What feeds back is a
    property of the RECORDING machine, which is a thing the head model has
    not had before: every per-head quantity this arc has measured came
    from the playback side, because the playback deck is the one in the
    room. The record mark's position ahead of the vertical sync is set by
    the recorder's drum servo and is frozen, so it is an entry on the
    record side of the chain.

    ITS RESOLVING POWER IS SMALL AND IS STATED AS SUCH. The specified
    window is three lines wide and both recorders measured here sit inside
    it, 1.3 to 1.5 lines apart, so the quantity separates two machines
    only when they happen to differ - it is a coarse label, not a
    calibration. What it does do without qualification is fix the head
    axis's ORIGIN for a given tape, which `per_head_reference` sets out.
    """
    heads = [against_vsync.get("head_%d" % head) for head in (0, 1)]
    values = [head["h"] for head in heads
              if head and np.isfinite(head.get("h", float("nan")))]
    low, high = SWITCH_AHEAD_OF_VSYNC_H
    return {
        "stage": "recording",
        "quantity": "the record head switch, lines ahead of vertical sync",
        "value_h": float(np.mean(values)) if values else float("nan"),
        "per_head_h": values,
        "specified_window_h": (float(low), float(high)),
        "window_width_h": float(high - low),
        "rotation_rate_hz": rate.get("specified_hz"),
        "belongs_to": "the machine that made the recording",
        "resolving_power": ("a three-line window that conforming machines "
                            "share, so it labels a recorder only when two "
                            "of them differ"),
        "also_fixes": ("the origin of the head axis for one tape, which is "
                       "the part with no qualification on it"),
    }


# --------------------------------------------------------------------------
# The whole chain
# --------------------------------------------------------------------------

def measure_capture(samples, sample_rate_hz: float, system: str = "NTSC",
                    refine: bool = True,
                    shared: Optional[Dict[str, object]] = None
                    ) -> Dict[str, object]:
    """Both marks on one capture, and everything that follows from them.

    The demodulator, the colour-under envelope and the specified intervals
    come from `band_delay`; the luminance switch is already identified by
    `head_switch.locate` and is read here only to pair the two on one time
    axis. What is added is the width-validated line locator, the
    vertical-sync locator, the recorded rotation and its reversal.
    """
    from scipy import signal as _signal

    values = np.asarray(samples, dtype=np.float64).ravel()
    # The same two full-field transforms `band_delay` needs, on the same
    # samples of the same field. A caller holding them may hand them in;
    # `band_delay.analytics` records what the duplication costs and why
    # padding the transform is not the answer.
    if shared is None:
        shared = band_delay.analytics(values, sample_rate_hz, system)
    frequency = shared["frequency"]
    envelope = shared["envelope"]
    rises = line_starts(frequency, sample_rate_hz, system)
    taps = _signal.firwin(255, [2.6e6, 5.4e6], fs=sample_rate_hz,
                          pass_zero=False)
    limited = _signal.filtfilt(taps, [1.0], values)
    magnitude = np.abs(_signal.hilbert(limited))
    smooth = _signal.firwin(255, 0.15e6, fs=sample_rate_hz)
    rf_envelope = _signal.filtfilt(smooth, [1.0], magnitude)
    rotation = record_rotation(envelope, rises, sample_rate_hz, system)
    record = record_switches(rotation, rises, sample_rate_hz, system)
    if refine and record:
        record = refine_switches(record, envelope, rises, sample_rate_hz,
                                 system)
    playback = luma_switch(frequency, rf_envelope, rises, sample_rate_hz,
                           record, system)
    vsync = vertical_sync(frequency, sample_rate_hz, system)
    against = switch_to_vertical_sync(record, vsync, sample_rate_hz, system)
    rate = rotation_rate(rotation, rises, sample_rate_hz, system)
    return {
        "rises": rises,
        "rotation": rotation,
        "rate": rate,
        "record": record,
        "luma": playback,
        "vertical_sync": vsync,
        "separation": separation(record, playback, sample_rate_hz, system),
        "against_vertical_sync": against,
        "reference": per_head_reference(record, playback, sample_rate_hz,
                                        system),
        "counter_rotation": counter_rotation(record, rises, sample_rate_hz,
                                             vsync, system),
        "head_model": head_model_entry(against, rate),
        "decoder": decoder_track_phase(system),
    }


def ownership(record_tap: Dict[str, object], playback_tap: Dict[str, object]
              ) -> Dict[str, object]:
    """Which machine each mark belongs to, decided by the two taps.

    The record tap carries the drive on its way to the heads. There is no
    playback head in it at all, so a mark present there was made by the
    recording machine and a mark absent there was not.

    MEASURED on the Sony SLV-778HF capture set, 12 field boundaries at the
    record tap and 11 at the playback tap of the same 75 per cent bars
    signal:

        mark                        record tap          playback tap
        colour-under reversal       12 of 12 fields     11 of 11 fields
        rotation rate, track A      +3937.78 Hz         +3939.59 Hz
        mark ahead of vertical sync 6.019 / 5.519 H     6.030 / 5.533 H
        line-timing displacement    9 to 20 ns          1530 to 1752 ns
        blanking level step         +0.00 to +0.07      -6.55 to +3.63

    The reversal is at both taps, at the same rate and in the same place
    to a hundredth of a line; the two luminance markers are at one tap
    only, and by a factor of about eighty. So the reversal is a
    record-side mark and the luminance switch a playback-side one, each
    shown by the presence or the absence of the thing itself rather than
    by argument.
    """
    def _displacements(measurement: Dict[str, object]) -> np.ndarray:
        return np.asarray([live["displacement_ns"]
                           for live in measurement.get("luma", ())],
                          dtype=np.float64)

    def _contrasts(measurement: Dict[str, object]) -> np.ndarray:
        return np.asarray([live["level_contrast"]
                           for live in measurement.get("luma", ())],
                          dtype=np.float64)

    at_record = _displacements(record_tap)
    at_playback = _displacements(playback_tap)
    level_record = _contrasts(record_tap)
    level_playback = _contrasts(playback_tap)
    rotation_both = (len(record_tap.get("record", ())) > 0
                     and len(playback_tap.get("record", ())) > 0)
    louder = (at_record.size and at_playback.size
              and np.median(np.abs(at_playback))
              > 10.0 * np.median(np.abs(at_record)))
    return {
        "rotation_at_both_taps": bool(rotation_both),
        "record_tap_fields": len(record_tap.get("record", ())),
        "playback_tap_fields": len(playback_tap.get("record", ())),
        "displacement_record_ns": (float(np.median(np.abs(at_record)))
                                   if at_record.size else float("nan")),
        "displacement_playback_ns": (float(np.median(np.abs(at_playback)))
                                     if at_playback.size else float("nan")),
        "level_record": (float(np.median(np.abs(level_record)))
                         if level_record.size else float("nan")),
        "level_playback": (float(np.median(np.abs(level_playback)))
                           if level_playback.size else float("nan")),
        "rotation_owner": ("the recording machine" if rotation_both
                           else "undetermined"),
        "displacement_owner": ("the playing machine" if louder
                               else "undetermined"),
        "why": ("the record tap carries the drive on its way to the heads, "
                "so it contains no playback head at all"),
        "cannot_split": ("the head's write from the head's read, on one "
                         "deck - they are one piece of ferrite in one chain "
                         "and never occur apart"),
    }
