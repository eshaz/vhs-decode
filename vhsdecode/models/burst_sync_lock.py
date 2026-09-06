"""The burst locked ABSOLUTELY to the sync pulse, within the field.

Ethan, having had to say it twice: "it is directly tied to frequency, i.e.
burst is a constant phase, and time is known across the period of the
burst, it should be locked absolutely to the sync pulse with in the field
by comparing the position relative to the position in the burst."

THE STATEMENT THIS LANE HAD BEEN MAKING WAS HALF OF IT. "The burst's phase
is absolute and its amplitude is not" is true and stops short. The whole
statement is that the burst's PHASE and the sync pulse's POSITION are one
measurement in two units, because within a gated subcarrier of known
frequency phase and elapsed time are interchangeable:

    phi = 2 pi f t

so phase, frequency and time are ONE relation and not three axes to be
measured separately. That is the "directly tied to frequency" clause.

WHAT THAT BUYS: A VERNIER. The two references fail in opposite ways and
compose exactly.

  THE SYNC EDGE is UNAMBIGUOUS and COARSE. It happens once a line, so
  there is no question which one you are looking at, but it is a
  band-limited edge in noise and localises to a fraction of a sample.

  THE BURST is PRECISE and WRAPPED. Its phase is fixed by specification -
  180 degrees from the +(B-Y) axis, BTS-3 section 2.4.10 - so a departure
  is the channel's, and one degree of it is 0.776 ns. But phase repeats
  every subcarrier cycle, so on its own it cannot say WHICH cycle.

  Together: the sync says which cycle, the burst says where in it. The
  result is absolute, unambiguous and 100 times finer than the sync alone,
  and it holds across the field rather than drifting per line, because
  both are read on the same line against the same clock.

THE CONDITION FOR IT TO WORK is a single inequality and it is checked
rather than assumed: the sync position's own error must be smaller than
half a subcarrier period, or the vernier picks the wrong cycle and the
answer is wrong by exactly one period - a failure that looks like a clean
measurement, which is why `resolve` reports the margin.
"""

import math
from typing import Dict, Optional

import numpy as np

__all__ = [
    "SUBCARRIER_HZ", "PAL_SUBCARRIER_HZ", "CYCLES_PER_LINE",
    "BURST_PHASE_DEG", "BURST_START_US", "BURST_START_TOLERANCE_US",
    "BURST_CYCLES", "subcarrier_hz", "ambiguity_interval_s",
    "phase_to_time_s", "time_to_phase_rad", "expected_burst_phase_rad",
    "resolve", "lock_from_lines", "seconds_per_degree",
]

SUBCARRIER_HZ = 315.0e6 / 88.0
"""3.579545 MHz. BTS-3 2.2.2 prints the value; 315/88 is the derivation
that makes it exact, and the two agree to the printed digits."""

PAL_SUBCARRIER_HZ = 4433618.75

CYCLES_PER_LINE = 455.0 / 2.0
"""227.5. BTS-3 2.2.3: "The horizontal scanning frequency shall be 2/455
times the chrominance subcarrier frequency." The HALF is the whole reason
the burst phase alternates line to line, and a lock that does not expect
that alternation reads it as a 180 degree error on every other line."""

BURST_PHASE_DEG = 180.0
"""BTS-3 2.4.10: "The phase of the chrominance subcarrier burst shall be
180 degrees relative to the (E'_B-E'_Y) axis". THIS IS THE CONSTANT PHASE
Ethan names - it is specified, not measured, so every departure from it
belongs to the channel."""

BURST_START_US = 5.3
BURST_START_TOLERANCE_US = (4.71, 5.71)
"""BTS-3 2.4.9: "The start of the subcarrier burst shall be 4.71 to 5.71
microseconds (5.3 microseconds nominal) after the front edge of horizontal
sync." The tolerance is a full microsecond wide - which is exactly why the
burst's START is not the time reference and its PHASE is."""

BURST_CYCLES = 9.0
"""9 +- 1 cycles, 2.23 to 3.11 us (BTS-3 2.4.9). Nine cycles is how much
averaging the phase estimate gets, and it is what sets its precision."""


def subcarrier_hz(system: str = "NTSC") -> float:
    return PAL_SUBCARRIER_HZ if str(system).upper().startswith("PAL") \
        else SUBCARRIER_HZ


def ambiguity_interval_s(system: str = "NTSC") -> float:
    """One subcarrier period - the span the burst phase CANNOT distinguish.

    279.365 ns for NTSC. At 4fsc sampling that is exactly four samples,
    which is the number the sync edge has to beat.
    """
    return 1.0 / subcarrier_hz(system)


def seconds_per_degree(system: str = "NTSC") -> float:
    """What one degree of burst phase is worth in time: 0.776 ns for NTSC.

    The reason the vernier is worth building. A sync edge localised to a
    tenth of a 4fsc sample is 7 ns; a burst phase good to one degree is
    0.78 ns, and nine cycles of averaging make that realistic.
    """
    return ambiguity_interval_s(system) / 360.0


def phase_to_time_s(phase_rad, system: str = "NTSC") -> np.ndarray:
    """Phase into elapsed time, wrapped into one period.

    This IS the "time is known across the period of the burst" clause: a
    subcarrier of known frequency converts phase to time with no free
    parameter.
    """
    period = ambiguity_interval_s(system)
    phase = np.asarray(phase_rad, dtype=np.float64)
    return np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi) * period


def time_to_phase_rad(time_s, system: str = "NTSC") -> np.ndarray:
    period = ambiguity_interval_s(system)
    return np.mod(np.asarray(time_s, dtype=np.float64) / period, 1.0) \
        * 2.0 * np.pi


def expected_burst_phase_rad(line_index, system: str = "NTSC",
                             cycles_per_line: Optional[float] = None
                             ) -> np.ndarray:
    """The phase the burst MUST have on a given line, from specification.

    The subcarrier is continuous and the line contains 227.5 of its cycles,
    so the phase at the same offset in successive lines advances by half a
    cycle - 180 degrees, alternating. Anything else is the channel.

    Returned relative to the specified burst phase, so an ideal signal
    gives zero on every line and the sign convention never has to be
    rediscovered.
    """
    cycles = CYCLES_PER_LINE if cycles_per_line is None else float(cycles_per_line)
    index = np.asarray(line_index, dtype=np.float64)
    return np.mod(2.0 * np.pi * cycles * index, 2.0 * np.pi)


def resolve(sync_position_s, burst_phase_rad, sync_uncertainty_s: float,
            system: str = "NTSC") -> Dict[str, object]:
    """THE VERNIER: an absolute position from a coarse one and a wrapped one.

    `sync_position_s` is where the sync edge says the line sits, in
    seconds, with `sync_uncertainty_s` its own error. `burst_phase_rad` is
    the measured burst phase, already referred to the specified phase so
    that an ideal line reads zero.

    The burst says the position is congruent to `phase_to_time_s(phase)`
    modulo one subcarrier period. The sync says which period. Choosing the
    period is a rounding, and rounding is where this can silently fail:
    if the sync's own error exceeds half a period the wrong cycle is
    chosen and the answer is off by exactly 279 ns, which is a plausible
    number that no downstream check would question.

    So the margin is reported and `resolved` is False when it is not met.
    A caller that ignores the flag gets a number; a caller that reads it
    gets a measurement.
    """
    period = ambiguity_interval_s(system)
    coarse = np.asarray(sync_position_s, dtype=np.float64)
    fine = phase_to_time_s(burst_phase_rad, system)
    cycles = np.round((coarse - fine) / period)
    absolute = cycles * period + fine
    margin = 0.5 * period - float(sync_uncertainty_s)
    return {
        "absolute_s": absolute,
        "cycle_index": cycles,
        "fine_s": fine,
        "displacement_s": absolute - coarse,
        "ambiguity_interval_s": period,
        "sync_uncertainty_s": float(sync_uncertainty_s),
        "margin_s": margin,
        "resolved": bool(margin > 0.0),
        "improvement": (float(sync_uncertainty_s) / max(
            seconds_per_degree(system), 1e-30)),
        "why": ("the sync edge chooses the subcarrier cycle and the burst "
                "phase places the line within it; unresolved means the sync "
                "error exceeds half a period, where the vernier picks the "
                "wrong cycle and is wrong by exactly one period"),
    }


def lock_from_lines(sync_positions_s, burst_phases_rad,
                    sync_uncertainty_s: float, line_indices=None,
                    system: str = "NTSC") -> Dict[str, object]:
    """The lock across a FIELD, which is the scope Ethan specifies.

    Per line the specified alternation is removed first - without that,
    every other line reads 180 degrees out and the field's spread is
    meaningless. What remains is the channel's own time error per line,
    and its statistics across the field are the lock:

      * its MEAN is the field's constant displacement between the two
        references, which is the absolute lock itself;
      * its SPREAD is what the time base failed to remove, in seconds
        rather than in degrees, because the conversion is exact.

    "Absolutely ... within the field" is the claim being tested: if the
    spread is small the two references agree across the whole field and the
    lock is absolute; if it grows with line index the burst is being read
    relative to a drifting line origin, which is the failure the per-line
    reading has.
    """
    positions = np.asarray(sync_positions_s, dtype=np.float64).ravel()
    phases = np.asarray(burst_phases_rad, dtype=np.float64).ravel()
    if len(positions) != len(phases):
        raise ValueError("one sync position and one burst phase per line")
    if line_indices is None:
        line_indices = np.arange(len(positions), dtype=np.float64)
    indices = np.asarray(line_indices, dtype=np.float64).ravel()

    # THE SPECIFIED ALTERNATION COMES OFF FIRST. 227.5 cycles a line means
    # the burst is 180 degrees out on alternate lines BY DESIGN.
    departure = np.angle(np.exp(1j * (
        phases - expected_burst_phase_rad(indices, system))))
    resolved = resolve(positions, departure, sync_uncertainty_s, system)
    error_s = phase_to_time_s(departure, system)
    period = ambiguity_interval_s(system)
    # centred, so a departure just under a full period reads as a small
    # negative one rather than a large positive one
    centred = np.where(error_s > 0.5 * period, error_s - period, error_s)

    if len(indices) > 2 and np.ptp(indices) > 0:
        slope = float(np.polyfit(indices, centred, 1)[0])
    else:
        slope = float("nan")
    return {
        "displacement_mean_s": float(np.mean(centred)),
        "displacement_rms_s": float(np.sqrt(np.mean(centred ** 2))),
        "spread_s": float(np.std(centred)),
        "drift_s_per_line": slope,
        "lines": int(len(centred)),
        "per_line_s": centred,
        "resolved": resolved["resolved"],
        "margin_s": resolved["margin_s"],
        "drift_across_field_s": abs(slope) * len(centred)
        if np.isfinite(slope) else float("nan"),
        # "ABSOLUTELY ... WITHIN THE FIELD" is the claim, so the test is
        # that the systematic drift across the whole field does not exceed
        # the random spread - with a PHYSICAL FLOOR under both.
        #
        # The floor is not decoration. On an ideal signal the drift and the
        # spread are both numerically zero, and comparing one to the other
        # then compares rounding noise to rounding noise and returns
        # whichever happened to be larger - which read False on a perfect
        # field. One degree of burst phase is the finest thing this
        # measurement can mean, so nothing below it is a drift.
        "absolute_within_field": bool(
            resolved["resolved"] and np.isfinite(slope)
            and abs(slope) * len(centred)
            <= max(float(np.std(centred)), seconds_per_degree(system))),
    }
