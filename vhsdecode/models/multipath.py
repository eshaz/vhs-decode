"""A TAPPED-DELAY CHANNEL FITTED IN ALL DIMENSIONS, not in phase alone.

Ethan, 2026-09-06, after the general propagation model was ruled out:
*"Let's continue onto the specific modeling, all dimensions not just
phase."*

WHAT WAS RULED OUT, AND WHAT WAS NOT. A general propagation model will not
remove the arc's all-pass residual: an echo weaker than the direct path is
minimum phase and contributes none of it, one stronger gives about seven
radians, and the measurement sits between the regimes at 0.87 to 2.07.
Two of the three tapes never passed a transmitter and carry most of it.
That closes the general question but not the specific one - countdown's
1.76 radian EXCESS over the two non-broadcast tapes is a real, measured
quantity that a tapped-delay channel could still explain, and the way to
find out is to fit it properly rather than to argue about it.

ALL DIMENSIONS MEANS THE COMPLEX FIT. A delayed copy changes the magnitude
and the phase together - it is one complex tap, `a exp(-j 2 pi f tau)`,
and fitting the phase alone throws away half the evidence and lets a tap
match a phase it has no magnitude for. So the residual is fitted as a
COMPLEX log departure with the real and imaginary parts stacked, which is
the arc's standing rule and the one that has caught five defects in this
work.

    magnitude   a delayed copy puts a comb on the response, of period
                1/tau, and its depth fixes the tap's amplitude
    phase       the same tap's phase, which is what the all-pass reading
                sees
    time        the taps must be the same on evidence they did not build,
                so the fit is judged on the half of the fields it never
                saw

THE DELAYS ARE BOUNDED BY THE PROBE, not chosen. A sync pulse of 4.7
microseconds cannot see a ghost beyond about 2.35 microseconds - the
arc's own reach law, `sync_geometry.ghost_reach` - and cannot resolve one
closer than a sample. So the search runs over that window and reports the
bound rather than pretending to a range it does not have.

THE RESULT, AND IT IS BACKWARDS FROM THE HYPOTHESIS. Fitted on the first
half of each record's fields and judged on the second:

    tape   head   in sample   held out   magnitude   phase
    home     a       52.5 %     52.6 %     -16.8 %    69.1 %
    home     b       51.9        52.0       -12.8      68.7
    bars     a       37.4        38.2        27.9      47.8
    bars     b       39.5        38.6        29.7      47.7
    countdown a      56.1        17.7        73.1     -57.7
    countdown b      55.7        14.6        72.5     -40.2

The two tapes that NEVER passed through a transmitter hold out almost
exactly - 52.6 against 52.5, 38.2 against 37.4 - and countdown, the one
tape where propagation multipath is possible at all, collapses from 56 per
cent to 17, with its PHASE going negative: the fit makes the held-out
phase worse than doing nothing. So whatever a tapped-delay channel is
describing here, it is not the air.

AND THE TAPS ARE THE SAME ON BOTH HEADS, which places them. Home's four
taps come out at 0.152, 0.341, 0.152 and 0.569 microseconds on head A and
at the same delays with the same amplitudes and phases to three decimals
on head B; the bars tape's at 0.152, 0.645, 0.379 and 0.834 likewise. Two
heads are two different physical paths through the tape, so a delayed copy
identical on both cannot be in that path - it is downstream of the head
switch, in the electronics the two share or in the decoder. On the bars
tape every amplitude is below one (0.65, 0.42, 0.35, 0.26), which is a
physically ordinary reflection structure; on home the first two exceed one
and are not.

Two cautions on reading the table. The 0.152 microsecond delay recurs in
every fit, which is close enough to the search grid's lower end to be a
property of the grid rather than of the signal, and it should be confirmed
on a finer grid before it is believed. And a held-out figure of 52 per
cent is not a model of anything until the entries are offered to the
arc's own ladder alongside everything else - `holdout.ladder` is where
that judgement belongs, not here.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def channel(frequency_hz, taps: Sequence[Tuple[float, complex]]
            ) -> np.ndarray:
    """The complex transfer of a direct path plus delayed copies.

    `taps` are (delay in seconds, complex amplitude). The direct path is
    the leading one and is not a tap.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    total = np.ones(f.size, dtype=np.complex128)
    for delay, amplitude in taps:
        total = total + complex(amplitude) * np.exp(-2j * np.pi * f * float(delay))
    return total


def _design(frequency_hz, delay: float) -> np.ndarray:
    """One tap's contribution to the complex log, to first order.

    log(1 + a e) is approximately a e for small a, so the fit is linear in
    the complex amplitude at each delay and a greedy search over delays is
    exact at each step rather than an approximation to one.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    return np.exp(-2j * np.pi * f * float(delay))


def fit(frequency_hz, departure, delays: Optional[Sequence[float]] = None,
        taps: int = 4, reach_s: float = 2.35e-6) -> Dict[str, object]:
    """Greedily fit complex taps to a complex log departure.

    At each step every candidate delay is offered, its complex amplitude
    solved in closed form by projection, and the one that removes most is
    kept. Real and imaginary parts are stacked so the projection cannot be
    blind to a quarter turn.
    """
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    residual = np.asarray(departure, dtype=np.complex128).ravel().copy()
    if delays is None:
        step = 1.0 / (2.0 * (f.max() - f.min()))
        delays = np.arange(step, float(reach_s), step / 4.0)
    found: List[Dict[str, object]] = []
    start = float(np.mean(np.abs(residual) ** 2))
    for _ in range(int(taps)):
        best = None
        for delay in delays:
            column = _design(f, delay)
            # least squares for a complex amplitude, real and imaginary
            # parts stacked so the inner product is not Hermitian
            basis = np.concatenate([np.column_stack([column.real, -column.imag]),
                                    np.column_stack([column.imag, column.real])])
            target = np.concatenate([residual.real, residual.imag])
            solution, *_ = np.linalg.lstsq(basis, target, rcond=None)
            amplitude = complex(solution[0], solution[1])
            left = residual - amplitude * column
            power = float(np.mean(np.abs(left) ** 2))
            if best is None or power < best[0]:
                best = (power, delay, amplitude, left)
        power, delay, amplitude, left = best
        found.append({"delay_s": float(delay), "amplitude": amplitude,
                      "magnitude": float(abs(amplitude)),
                      "phase_deg": float(np.degrees(np.angle(amplitude))),
                      "removed": 1.0 - power / max(start, 1e-30)})
        residual = left
    return {
        "taps": found, "residual": residual,
        "start_power": start,
        "end_power": float(np.mean(np.abs(residual) ** 2)),
        "removed_fraction": 1.0 - float(np.mean(np.abs(residual) ** 2)) / max(start, 1e-30),
        "delay_step_s": float(delays[1] - delays[0]) if len(delays) > 1 else 0.0,
        "reach_s": float(reach_s),
        "why": ("a delayed copy moves magnitude and phase together, so it "
                "is fitted as one complex tap; the delays are bounded by "
                "what the probe can reach, not chosen"),
    }


def apply_taps(frequency_hz, taps: Sequence[Dict[str, object]]) -> np.ndarray:
    """The complex log departure a fitted tap set predicts."""
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    total = np.zeros(f.size, dtype=np.complex128)
    for tap in taps:
        total = total + complex(tap["amplitude"]) * _design(f, tap["delay_s"])
    return total


def held_out(frequency_hz, fit_departure, judge_departure,
             taps: int = 4, reach_s: float = 2.35e-6) -> Dict[str, object]:
    """Fit on one half of the evidence and judge on the other.

    A tapped-delay channel is a property of the path, so its taps must be
    the same on fields the fit never saw. Reported in all three dimensions
    separately, because they can fail separately: a tap that matches the
    magnitude and not the phase is not a delayed copy of anything.
    """
    fitted = fit(frequency_hz, fit_departure, taps=taps, reach_s=reach_s)
    predicted = apply_taps(frequency_hz, fitted["taps"])
    judge = np.asarray(judge_departure, dtype=np.complex128).ravel()
    left = judge - predicted

    def share(before, after):
        b = float(np.mean(np.abs(before) ** 2))
        return 1.0 - float(np.mean(np.abs(after) ** 2)) / max(b, 1e-30)

    return {
        "taps": fitted["taps"],
        "in_sample_removed": fitted["removed_fraction"],
        "held_out_removed": share(judge, left),
        "magnitude_removed": share(judge.real, left.real),
        "phase_removed": share(judge.imag, left.imag),
        "residual": left,
        "why": ("the taps are a property of the path, so they must hold on "
                "fields they did not build; a fit that helps in sample and "
                "not out of it has found the noise"),
    }
