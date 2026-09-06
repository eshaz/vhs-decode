"""The differential as a file: bounded window, self-measured residual.

Pins the three things that make the loop honest. The window has two bounds
and they can cross, in which case it must refuse rather than pick one. The
residual is measured before the field joins the window, so it is out of
sample by construction and a planted model error must show in it. And the
file holds the contrasts, from which every vertex difference is rebuilt.
"""

import numpy as np
import pytest

from vhsdecode.models import running_differential as rd


def test_the_window_is_bounded_at_both_ends_and_refuses_when_they_cross():
    rate = 60.0 / 1.001
    loose = rd.window_bounds(rate, drift_hz=0.293, floor_variance=1.0,
                             target_variance=0.02)
    assert loose["feasible"]
    assert loose["minimum_fields"] == pytest.approx(50.0)
    # a quarter of the drift's period, in fields
    assert loose["maximum_fields"] == pytest.approx(0.25 * rate / 0.293, rel=1e-6)
    assert 23 <= rd.choose_window(loose) <= 51
    tight = rd.window_bounds(rate, drift_hz=0.293, floor_variance=1.0,
                             target_variance=0.005)
    assert not tight["feasible"]
    with pytest.raises(ValueError, match="no window satisfies both bounds"):
        rd.choose_window(tight)
    # a faster drift shrinks the ceiling
    faster = rd.window_bounds(rate, drift_hz=2.0, floor_variance=1.0,
                              target_variance=0.02)
    assert faster["maximum_fields"] < loose["maximum_fields"]


def _series(fields, bins, rng, model=None, noise=0.05):
    shape = (bins,)
    base = np.exp(-2 * np.linspace(0, 1, bins)) if model is None else model
    out = {}
    for n in range(fields):
        out[n] = base + noise * (rng.standard_normal(bins)
                                 + 1j * rng.standard_normal(bins)) / np.sqrt(2)
    return out, np.full(shape, noise ** 2)


def test_a_stationary_signal_settles_at_its_floor():
    rng = np.random.default_rng(0)
    values, variance = _series(400, 32, rng)
    run = rd.RunningDifferential((), np.arange(32) * 1e5, window=40)
    for n, v in values.items():
        run.update(n, v, variance)
    out = run.verdict()
    assert out["judged"] == 399
    # nothing but noise: the out-of-sample residual sits at the floor
    assert 0.8 < out["residual_over_floor"] < 1.25
    assert out["at_floor"]
    assert out["gain"] < 0.25


def test_a_drift_the_window_cannot_track_shows_in_the_residual():
    rng = np.random.default_rng(1)
    bins = 32
    base = np.exp(-2 * np.linspace(0, 1, bins))
    run = rd.RunningDifferential((), np.arange(bins) * 1e5, window=40)
    variance = np.full(bins, 0.05 ** 2)
    for n in range(400):
        drift = 0.5 * np.sin(2 * np.pi * n / 60.0)      # far faster than the window
        value = base * (1 + drift) + 0.05 * (
            rng.standard_normal(bins) + 1j * rng.standard_normal(bins)) / np.sqrt(2)
        run.update(n, value, variance)
    out = run.verdict()
    assert out["residual_over_floor"] > 5.0
    assert not out["at_floor"]
    assert out["gain"] > 0.8            # a large excess earns a large gain


def test_the_judgement_is_out_of_sample_by_construction():
    """The field being judged has not yet entered the window that judges it,
    so a single wild field cannot hide inside its own model."""
    rng = np.random.default_rng(2)
    bins = 16
    run = rd.RunningDifferential((), np.arange(bins) * 1e5, window=20)
    variance = np.full(bins, 0.01 ** 2)
    base = np.ones(bins, dtype=complex)
    for n in range(60):
        run.update(n, base + 0.01 * rng.standard_normal(bins), variance)
    report = run.update(60, base + 5.0, variance)          # one wild field
    assert report["over_floor"] > 100.0


def test_the_file_holds_the_contrasts_and_rebuilds_a_vertex():
    rng = np.random.default_rng(3)
    bins = 24
    head = 0.3 * np.exp(-np.linspace(0, 1, bins))
    run = rd.RunningDifferential(("head",), np.arange(bins) * 1e5, window=30)
    variance = np.full((2, bins), 0.01 ** 2)
    for n in range(60):
        values = np.stack([head, -head]) + 0.01 * rng.standard_normal((2, bins))
        run.update(n, values, variance)
    path = "/tmp/claude-1000/-workspaces-vhs-decode/61dd5ae6-c67f-4bd1-bbaa-8c55cd55ffc0/scratchpad/test_diff.npz"
    run.write(path, metadata="planted")
    stored = rd.read(path)
    assert [str(n) for n in stored["contrast_names"]] == ["mean", "head"]
    # applying the differential at vertex (0,) removes the head term
    corrected = rd.apply_differential(head, stored, (0,), gain=1.0)
    assert np.abs(corrected).max() < 0.05
    # and at the other vertex it removes it with the opposite sign
    other = rd.apply_differential(-head, stored, (1,), gain=1.0)
    assert np.abs(other).max() < 0.05
