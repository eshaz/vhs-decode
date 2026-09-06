"""The burst locked absolutely to the sync pulse, within the field.

Ethan, twice: "burst is a constant phase, and time is known across the
period of the burst, it should be locked absolutely to the sync pulse with
in the field by comparing the position relative to the position in the
burst."

The vernier's one failure mode is silent - too coarse a sync pick the
wrong subcarrier cycle and the answer is wrong by exactly one period,
which is a plausible number - so half of these tests are about that.
"""

import numpy as np
import pytest

from vhsdecode.models import burst_sync_lock as lock

FS4 = 4 * 315e6 / 88


def test_the_ambiguity_is_exactly_four_samples_at_4fsc():
    """Why the vernier is buildable at all: the sync only has to beat two
    samples to choose the right cycle, and it localises far better."""
    assert lock.ambiguity_interval_s() * FS4 == pytest.approx(4.0)
    assert lock.ambiguity_interval_s() == pytest.approx(279.365e-9, rel=1e-5)


def test_one_degree_of_burst_is_sub_nanosecond():
    """The precision the burst brings: 0.776 ns a degree, against a sync
    edge that localises to a fraction of a 70 ns sample."""
    assert lock.seconds_per_degree() == pytest.approx(0.776e-9, rel=1e-3)


def test_phase_and_time_are_the_same_measurement():
    """The 'directly tied to frequency' clause: within a subcarrier of
    known frequency, phase and elapsed time convert with no free
    parameter."""
    times = np.linspace(0.0, lock.ambiguity_interval_s() * 0.999, 17)
    recovered = lock.phase_to_time_s(lock.time_to_phase_rad(times))
    assert np.allclose(recovered, times, atol=1e-15)


def test_the_specified_alternation_is_expected_not_measured():
    """227.5 cycles a line means the burst is 180 degrees out on alternate
    lines BY DESIGN. A lock that does not expect it reads a 180 degree
    error on every other line."""
    phases = lock.expected_burst_phase_rad(np.arange(6))
    assert np.allclose(phases[0::2], 0.0, atol=1e-9)
    assert np.allclose(phases[1::2], np.pi, atol=1e-9)


def test_the_vernier_recovers_a_planted_position():
    """The coarse reference chooses the cycle, the fine one places the
    line inside it."""
    period = lock.ambiguity_interval_s()
    truth = 37.0 * period + 0.31 * period
    generator = np.random.default_rng(20260906)
    for _ in range(50):
        coarse = truth + generator.normal(0.0, 0.15 * period)
        phase = lock.time_to_phase_rad(truth)
        got = lock.resolve(coarse, phase, 0.15 * period)
        assert got["resolved"]
        assert float(got["absolute_s"]) == pytest.approx(truth, abs=1e-15)


def test_it_refuses_when_the_sync_cannot_choose_the_cycle():
    """The inequality that makes the vernier valid, checked rather than
    assumed."""
    period = lock.ambiguity_interval_s()
    got = lock.resolve(10.0 * period, 1.0, 0.6 * period)
    assert not got["resolved"]
    assert got["margin_s"] < 0.0


def test_a_coarse_sync_lands_exactly_one_period_out():
    """THE SILENT FAILURE, demonstrated. When the sync error exceeds half
    a period the vernier picks the neighbouring cycle, and the answer is
    wrong by exactly one subcarrier period - smooth, plausible, and 279 ns
    from the truth."""
    period = lock.ambiguity_interval_s()
    truth = 12.0 * period + 0.1 * period
    phase = lock.time_to_phase_rad(truth)
    got = lock.resolve(truth + 0.7 * period, phase, 0.7 * period)
    error = float(got["absolute_s"]) - truth
    assert abs(error) == pytest.approx(period, rel=1e-9)
    assert not got["resolved"]                 # and it says so


def test_an_ideal_field_locks_at_zero_on_every_line():
    """A signal with no channel error reads zero displacement on every
    line, alternation and all - so any departure is the channel's."""
    lines = np.arange(240)
    positions = lines * (1.0 / 15734.265734)
    phases = lock.expected_burst_phase_rad(lines)
    got = lock.lock_from_lines(positions, phases, 0.1 * lock.ambiguity_interval_s(),
                               line_indices=lines)
    assert got["displacement_rms_s"] < 1e-15
    assert got["resolved"]


def test_a_constant_displacement_is_recovered_across_the_field():
    """'Absolutely within the field' - the whole field shares one offset
    and the lock reports it rather than integrating it up per line."""
    lines = np.arange(240)
    offset = 11.0e-9
    positions = lines * (1.0 / 15734.265734)
    phases = lock.expected_burst_phase_rad(lines) + lock.time_to_phase_rad(offset)
    got = lock.lock_from_lines(positions, phases,
                               0.1 * lock.ambiguity_interval_s(),
                               line_indices=lines)
    assert got["displacement_mean_s"] == pytest.approx(offset, rel=1e-6)
    assert got["spread_s"] < 1e-15
    assert got["absolute_within_field"]


def test_a_drifting_lock_is_not_absolute_within_the_field():
    """The failure the per-line reading has: a burst read against a
    drifting line origin accumulates, and the field test catches it."""
    lines = np.arange(240)
    drift = np.linspace(0.0, 40e-9, len(lines))
    positions = lines * (1.0 / 15734.265734)
    phases = lock.expected_burst_phase_rad(lines) + lock.time_to_phase_rad(drift)
    got = lock.lock_from_lines(positions, phases,
                               0.1 * lock.ambiguity_interval_s(),
                               line_indices=lines)
    assert got["drift_s_per_line"] == pytest.approx(40e-9 / 239, rel=0.05)
    assert not got["absolute_within_field"]


def test_the_specification_constants_match_bts3():
    """Cross-checked against Canada's BTS-3 (Issue 2, May 2023 revision),
    which is an independent printing of the same system:

      2.4.10  burst phase 180 degrees relative to (E'_B - E'_Y)
      2.4.9   9 +- 1 cycles, starting 4.71 to 5.71 us (5.3 nominal)
      2.2.2   subcarrier 3.579545 MHz
      2.2.3   line rate = 2/455 of the subcarrier
    """
    assert lock.BURST_PHASE_DEG == 180.0
    assert lock.BURST_CYCLES == 9.0
    low, high = lock.BURST_START_TOLERANCE_US
    assert low < lock.BURST_START_US < high
    assert lock.SUBCARRIER_HZ == pytest.approx(3.579545e6, abs=1.0)
    assert lock.SUBCARRIER_HZ * 2.0 / 455.0 == pytest.approx(15734.264, abs=0.01)
    # 19 cycles after the sync edge, the models' figure, is inside BTS-3's
    # stated window - so the two conventions in this tree are both legal
    nineteen_us = 19.0 / lock.SUBCARRIER_HZ * 1e6
    assert low < nineteen_us < high
