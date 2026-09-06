"""The back porch's relaxation measurement.

The estimator's one real trap is the level the tail decays TO. The back
porch does not settle before active video starts - that is where this
round began - so taking the asymptote as the median of the window's own
last samples reads a level that is still up the curve, and biases every
time constant short. Measured: a planted tau of 33 samples in a 59-sample
window came back 29 per cent low that way, while a planted 9, which does
settle inside it, came back to half a per cent. The bias is exactly where
the assumption fails, which is why it survived a first look.

Fitting the offset jointly removes it, and these tests hold that: the long
time constants are the ones that matter, and they are the ones the naive
estimator got wrong.
"""

import numpy as np
import pytest

from vhsdecode.addons import ringing_cancellation as rc

SYS_PARAMS = {
    "activeVideoUS": (9.45, 62.5), "hsyncPulseUS": 4.7,
    "syncTransitionUS": 0.140, "frontPorchUS": 1.5,
    "line_period": 63.5556, "outfreq": 4 * 315.0 / 88,
    "vsync_ire": -40.0, "hz_ire": 1e6 / 140,
    "numPulses": 6, "colorBurstUS": (5.3, 7.8), "activeVideoLines": 262,
}
# the decoder's real VHS NTSC luma low-pass: a supergauss applied as a
# MAGNITUDE, so zero phase and symmetric - which is what puts a precursor
# ahead of the active transition and inside the back porch
DECODER_PARAMS = {"video_lpf_freq": 6.6e6, "video_lpf_order": 9,
                  "video_lpf_supergauss": True}
WIDTH = 910


@pytest.fixture(scope="module")
def frame():
    geometry = rc.build_geometry(SYS_PARAMS, DECODER_PARAMS, WIDTH,
                                 line_offset=1, line_count=262)
    return geometry, rc.plan_measurement_window(geometry)


def plant(frame, tau, amplitude, settled=0.0):
    """A profile carrying one known relaxation on the read window.

    Planted where the measurement READS, not on `plan.back_porch` - the
    two differ deliberately, because the read window excludes the sync
    rise, and a test that planted on the other one would be asserting
    against a window the estimator never sees.
    """
    geometry, plan = frame
    start, stop = rc.porch_relaxation_window(geometry, plan)
    # THE PEDESTAL GOES ON THE WHOLE LINE, not just the window. Confining
    # it to the porch would put a step of that size at active video, and
    # the measurement now removes that transition's computed precursor -
    # so the planted signal would carry a pedestal-scaled artefact that
    # nothing in the physical picture puts there.
    profile = np.full(plan.total_samples, settled, dtype=np.float64)
    lags = np.arange(stop - start, dtype=np.float64)
    profile[start:stop] = settled + amplitude * np.exp(-lags / tau)
    return {"rise_interval_mean": profile}


@pytest.mark.parametrize("tau,amplitude", [
    (9.0, -0.90),      # settles well inside the window
    (17.0, -3.60),     # comparable to the window
    (33.0, +2.10),     # LONGER than the window - the naive estimator's bug
    (60.0, -4.00),     # longer still, where the tail is nearly a ramp
])
def test_planted_relaxation_recovers(frame, tau, amplitude):
    geometry, plan = frame
    got = rc.measure_porch_relaxation(plant(frame, tau, amplitude), geometry,
                                      plan)
    assert got is not None
    assert got["tau_samples"] == pytest.approx(tau, rel=0.06)
    assert got["amplitude_ire"] == pytest.approx(amplitude, rel=0.06)


def test_offset_is_fitted_not_assumed(frame):
    """A relaxation sitting on a DC pedestal must recover unchanged.

    The naive estimator subtracted the window's own trailing median, so a
    pedestal it could not separate from the tail's own unsettled value
    went straight into the amplitude. Fitting the offset makes the answer
    independent of the pedestal, which is the property being asserted.
    """
    geometry, plan = frame
    reference = rc.measure_porch_relaxation(plant(frame, 40.0, -3.0),
                                            geometry, plan)
    for pedestal in (-5.0, 0.0, +12.0):
        got = rc.measure_porch_relaxation(
            plant(frame, 40.0, -3.0, settled=pedestal), geometry, plan)
        assert got["tau_samples"] == pytest.approx(
            reference["tau_samples"], rel=1e-6)
        assert got["amplitude_ire"] == pytest.approx(
            reference["amplitude_ire"], rel=1e-6)
        # to the tau grid's own resolution: the scan is 96 geometric
        # steps, so the fitted tau is within about 4 per cent of the
        # truth and the offset absorbs that small mismatch. The
        # INVARIANCE above is the property under test; this only holds
        # the offset to the level it is measuring.
        assert got["settled_ire"] == pytest.approx(pedestal, abs=0.05)


def test_flat_porch_reports_nothing(frame):
    """No relaxation means None, not a fit to the noise floor."""
    geometry, plan = frame
    assert rc.measure_porch_relaxation(
        plant(frame, 40.0, 0.0, settled=2.5), geometry, plan) is None


def test_amplitude_below_the_floor_is_refused(frame):
    geometry, plan = frame
    small = rc.PORCH_RELAXATION_MIN_AMPLITUDE_IRE / 2.0
    assert rc.measure_porch_relaxation(plant(frame, 30.0, -small), geometry,
                                       plan) is None


def test_missing_fold_reports_nothing(frame):
    geometry, plan = frame
    assert rc.measure_porch_relaxation({}, geometry, plan) is None


def test_pole_and_tau_agree(frame):
    """The pole is the time constant's own exponential, not a second
    number that could drift from it."""
    geometry, plan = frame
    got = rc.measure_porch_relaxation(plant(frame, 25.0, -2.0), geometry, plan)
    assert got["pole"] == pytest.approx(np.exp(-1.0 / got["tau_samples"]))


def test_reported_in_window_coordinates(frame):
    """The fold's frame, not the line buffer's - they differ by the
    window's own lead on the sync fall, and conflating them puts the
    correction `sync_fall_index` samples from where it was measured."""
    geometry, plan = frame
    got = rc.measure_porch_relaxation(plant(frame, 30.0, -2.0), geometry, plan)
    assert (got["window_start"], got["window_stop"]) == \
        rc.porch_relaxation_window(geometry, plan)
    # and it is NOT the stage's back-porch region: that one begins on the
    # sync rise, and fitting a decay across the edge that drives it reads
    # the edge (2.94 us) instead of the tail (1.25 us)
    assert got["window_start"] > plan.back_porch[0]


def test_a_ramp_rails_the_scan(frame):
    """What a RAILED fit looks like, held so a reader can recognise one.

    A straight ramp has no time constant, but it is well approximated by
    a very long exponential plus an offset, so the scan runs to its upper
    bound rather than refusing. That railed answer is exactly the
    signature the corrected back porch shows on both heads - tau at the
    bound with the amplitude's sign reversed - and reading it as a
    measurement rather than as a rail is how a double correction would
    get shipped.
    """
    geometry, plan = frame
    start, stop = rc.porch_relaxation_window(geometry, plan)
    profile = np.zeros(plan.total_samples)
    profile[start:stop] = np.linspace(0.0, -3.0, stop - start)
    got = rc.measure_porch_relaxation({"rise_interval_mean": profile},
                                      geometry, plan)
    ceiling = rc.PORCH_RELAXATION_TAU_SEARCH_SPAN * (stop - start)
    assert got is not None
    assert got["tau_samples"] == pytest.approx(ceiling, rel=1e-9)


def test_precursor_is_derived_not_fitted(frame):
    """The active transition's precursor comes from the decoder's own two
    low-pass parameters, and nothing else.

    Verified against the values e4 derived independently: against a 100 IRE
    transition the SIGNED precursor is -4.22, +0.94, +0.25, -0.70 IRE at 2,
    4, 6 and 8 samples ahead. It rings rather than decaying because order 9
    is nearly a brick wall, which is why no affordable guard clears it.
    """
    geometry, _ = frame
    precursor = rc.active_transition_precursor(geometry, 40)
    assert precursor is not None
    for lag, expected in ((2, -4.22), (4, +0.94), (6, +0.25), (8, -0.70)):
        assert 100.0 * precursor[lag - 1] == pytest.approx(expected, abs=0.02)


def test_causal_lowpass_has_no_precursor():
    """A Butterworth low-pass is causal, so there is nothing ahead of the
    transition to remove and the term vanishes by construction."""
    causal = dict(DECODER_PARAMS)
    causal["video_lpf_supergauss"] = False
    geometry = rc.build_geometry(SYS_PARAMS, causal, WIDTH, line_offset=1,
                                 line_count=262)
    assert rc.active_transition_precursor(geometry, 40) is None


def test_planted_precursor_is_removed(frame):
    """A window carrying ONLY the precursor of a known step must come back
    empty.

    THE ALIGNMENT IS WHAT THIS PINS. Taking the wrong end of the reversed
    precursor does not fail loudly - it INJECTS the supergauss's own
    3.3 MHz ringing into the porch, which then reads as a clean narrowband
    component with a stable frequency that was never there. It cost a whole
    measurement pass before the growing amplitude gave it away.
    """
    geometry, plan = frame
    transition = int(round(geometry.active_video_start_sample))
    start, stop = 76, 128
    precursor = rc.active_transition_precursor(geometry, transition - start)
    line = np.zeros(geometry.samples_per_line)
    line[transition:] = 50.0
    line[start:transition] = 50.0 * precursor[::-1]
    before = float(np.sqrt(np.mean(line[start:stop] ** 2)))
    cleaned, removed = rc.remove_active_precursor(line, geometry, start, stop,
                                                  transition)
    after = float(np.sqrt(np.mean(cleaned[start:stop] ** 2)))
    assert before > 0.15
    assert after < before / 100.0
    assert removed == pytest.approx(before, rel=0.02)


def test_precursor_removal_leaves_a_flat_porch_alone(frame):
    """No transition means nothing removed - the correction must not
    invent a shape where the next line is at blanking."""
    geometry, plan = frame
    transition = int(round(geometry.active_video_start_sample))
    line = np.zeros(geometry.samples_per_line)
    cleaned, removed = rc.remove_active_precursor(line, geometry, 76, 128,
                                                  transition)
    assert removed == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(cleaned, line)
