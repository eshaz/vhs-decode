"""The VCR's automatic gain control: its level response, its clipped
reference, and the field-length probe that gives its time constant a
measurement instead of an assumption.

The module had no test file at all before this one, which is part of why
`ASSUMED_TIME_CONSTANTS_US` went five decades wide and unchallenged for so
long. The numbers checked here against real decodes come from
/output/decodes_2026-09-06/, read over the 236 lines of
`sync_depth.field_slope`'s window.
"""

import numpy as np
import pytest

from vhsdecode.models import sync_geometry, vcr_agc


# ---------------------------------------------------------------- the loop

def test_corner_is_one_over_two_pi_tau():
    assert vcr_agc.corner_hz(1000.0) == pytest.approx(159.1549, rel=1e-5)
    # a decade slower is a decade lower, exactly
    assert (vcr_agc.corner_hz(100.0) / vcr_agc.corner_hz(1000.0)
            == pytest.approx(10.0, rel=1e-12))


def test_level_response_is_a_high_pass_at_the_corner():
    """The loop passes what it cannot track and removes what it can."""
    tau = 1000.0
    corner = vcr_agc.corner_hz(tau)
    response = vcr_agc.level_response(
        np.array([corner * 1e-3, corner, corner * 1e3]), tau)
    assert abs(response[0]) < 1e-2                      # removed
    assert abs(response[1]) == pytest.approx(1 / np.sqrt(2), rel=1e-9)
    assert abs(response[2]) == pytest.approx(1.0, abs=1e-5)  # passed
    # and it is complex, so it carries phase - +90 degrees well below
    assert np.angle(response[0]) == pytest.approx(np.pi / 2, abs=1e-2)


def test_reference_error_is_the_reciprocal_of_what_the_clip_removed():
    assert vcr_agc.reference_error(0.4)["gain_error"] == pytest.approx(
        1.0 / 0.6, rel=1e-12)
    assert vcr_agc.reference_error(0.0)["gain_error"] == 1.0
    assert vcr_agc.reference_error(1.0)["gain_error"] == float("inf")


def test_line_rate_modulation_is_not_a_frequency_response():
    result = vcr_agc.line_rate_sidebands(1.0)
    assert result["is_a_frequency_response"] is False
    assert result["replication_weights"][0] == 1.0
    assert set(result["offsets_hz"]) == set(result["replication_weights"])


def test_signatures_decline_an_unspecified_time_constant():
    grid = np.linspace(1.0, 1e3, 16)
    assert vcr_agc.signatures(grid) == {}
    assert "vcr agc level response" in vcr_agc.signatures(grid, 1000.0)


# --------------------------------------------------------- the field probe

def test_field_probe_is_on_the_same_ruler_as_the_sync_probes():
    """It must be commensurable with `sync_geometry.probes`, not a separate
    claim on a separate clock."""
    one_line = vcr_agc.field_probe(1000)["duration_us"] / 1000.0
    assert one_line == pytest.approx(
        sync_geometry.LINE_PERIOD_US["525"], rel=1e-12)
    assert one_line == pytest.approx(
        sync_geometry.probes("NTSC")["one whole line"]["duration_us"],
        rel=1e-12)


def test_field_probe_reaches_far_past_the_vertical_interval():
    """The measured extension: 236 lines is 15.0 ms and 66.7 Hz, against the
    vertical interval's 572 us and 1748 Hz."""
    probe = vcr_agc.field_probe(236)
    interval = sync_geometry.probes("NTSC")["the whole vertical interval"]
    assert probe["duration_us"] == pytest.approx(14999.111, rel=1e-6)
    assert probe["resolution_hz"] == pytest.approx(66.6706, rel=1e-5)
    assert probe["reaches_time_constant_us"] == pytest.approx(2387.18, rel=1e-4)
    assert probe["resolution_hz"] < interval["resolution_hz"] / 26.0


def test_field_probe_refuses_a_window_too_short_to_hold_a_slope():
    with pytest.raises(ValueError):
        vcr_agc.field_probe(2)


def test_the_field_probe_reaches_rows_that_used_to_read_nothing():
    """100 us and 1 ms were reachable by nothing while only contiguous runs
    of sync counted as probes. 50 ms is still out of reach."""
    without = {row["time_constant_us"]: row["reachable_by"]
               for row in vcr_agc.where_the_corner_lands()["rows"]}
    with_field = {row["time_constant_us"]: row["reachable_by"] for row
                  in vcr_agc.where_the_corner_lands(lines_measured=236)["rows"]}
    for tau in (100.0, 1000.0):
        assert without[tau] == ["nothing here"]
        assert "236 lines of a field" in with_field[tau]
    assert with_field[50000.0] == ["nothing here"]
    # the default is unchanged, so nothing downstream shifts
    assert vcr_agc.where_the_corner_lands()["field_probe_included"] is False


# ------------------------------------------------------- the curvature law

def test_curvature_has_the_analytic_small_window_limit():
    """rms(bend)/drift -> x / (2 sqrt(180)) as the window shrinks against
    the time constant."""
    for x in (0.002, 0.01, 0.05):
        assert (float(vcr_agc.relaxation_curvature(x)[0])
                == pytest.approx(x / (2.0 * np.sqrt(180.0)), rel=2e-3))


def test_curvature_is_monotone_per_fitted_slope():
    """Normalised against the FITTED slope - the quantity actually measured
    - the law is monotone, so a resolvable ratio inverts to one `x`. It is
    normalising against the total excursion instead that makes it turn
    over, and that normalisation is not comparable with the measurement."""
    rising = vcr_agc.relaxation_curvature(
        [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 236.0])
    assert np.all(np.diff(rising) > 0)


def test_the_template_is_the_vector_form_of_the_scalar_law():
    for x in (0.1, 1.0, 5.0, 50.0):
        assert float(np.sqrt(np.mean(
            vcr_agc._bend_template(x, 2001) ** 2))) == pytest.approx(
                float(vcr_agc.relaxation_curvature(x, 2001)[0]), rel=1e-12)


def test_the_template_has_the_sign_a_falling_relaxation_leaves():
    """A relaxation that FALLS leaves a rising exponential's residual. The
    first version of this divided by a positive excursion and predicted the
    bend upside down; a planted relaxation read -2.04 where +1 was due."""
    points, x, drift = 401, 6.0, -0.26
    u = np.linspace(0.0, 1.0, points)
    shape = -np.expm1(-x * u) / (1.0 - np.exp(-x))
    series = 36.5 + drift * shape
    design = np.vstack([np.ones(points), u]).T
    coefficients = np.linalg.lstsq(design, series, rcond=None)[0]
    residual = series - design @ coefficients
    predicted = coefficients[1] * vcr_agc._bend_template(x, points)
    assert np.allclose(residual, predicted, atol=1e-12)


def test_the_curvature_limit_is_one_line_of_relaxation():
    limit = vcr_agc.curvature_limit(236)
    assert limit["window_over_time_constant"] == 236.0
    assert limit["curvature_over_drift"] == pytest.approx(
        float(vcr_agc.relaxation_curvature(236.0, 236)[0]), rel=1e-12)


def test_curvature_is_zero_for_a_window_with_no_relaxation_in_it():
    assert float(vcr_agc.relaxation_curvature(0.0)[0]) == 0.0


# ------------------------------------------- the pair, on planted evidence

def _series(mean, drift, scatter, points=236, seed=0):
    """A level series whose least-squares fit has EXACTLY the given drift
    across the window and exactly the given residual scatter, so the tests
    check the module's arithmetic rather than a random draw's."""
    index = np.arange(points, dtype=np.float64)
    unit = index / (points - 1.0) - 0.5           # mean zero, span one
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(points)
    design = np.vstack([np.ones(points), index]).T
    noise = noise - design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    if scatter > 0.0:
        noise *= scatter / np.sqrt(noise @ noise / (points - 2.0))
    else:
        noise *= 0.0
    return mean + drift * unit + noise


def test_the_planted_series_is_exact():
    """The fixture must be trustworthy before anything is checked with it."""
    result = vcr_agc.time_constant_from_drift(
        _series(36.5, -0.2577, 0.0656), _series(0.0, 0.0, 0.0656, seed=1))
    assert result["differential"]["drift_ire"] == pytest.approx(-0.2577,
                                                                rel=1e-9)
    assert result["differential"]["scatter_ire"] == pytest.approx(0.0656,
                                                                  rel=1e-9)
    assert result["gain_drift_fraction"] == pytest.approx(-0.2577 / 36.5,
                                                          rel=1e-9)


def test_the_pair_separates_a_gain_drift_from_a_level_shift():
    """A level shift moves both levels together and leaves the spacing
    alone; a gain drift moves the spacing. That is the whole separation,
    and the two cases must come out different."""
    quiet = _series(36.5, 0.0, 0.05, seed=2)
    level_only = vcr_agc.time_constant_from_drift(
        quiet, _series(0.0, 1.0, 0.05, seed=3))
    assert level_only["level_moves"] is True
    assert level_only["gain_moves"] is False
    assert level_only["which_moves"] == "the level"
    assert level_only["the_pair_disagrees"] is True

    gain_only = vcr_agc.time_constant_from_drift(
        _series(36.5, 1.0, 0.05, seed=4), _series(0.0, 0.0, 0.05, seed=5))
    assert gain_only["gain_moves"] is True
    assert gain_only["level_moves"] is False
    assert gain_only["which_moves"] == "the gain"
    assert gain_only["the_pair_disagrees"] is True

    both = vcr_agc.time_constant_from_drift(
        _series(36.5, 1.0, 0.05, seed=6), _series(0.0, 1.0, 0.05, seed=7))
    assert both["which_moves"] == "both the gain and the level"
    assert both["the_pair_disagrees"] is False


def test_the_bound_is_conservative_when_a_bend_is_present():
    """A real bend inflates the scatter it is measured against, so it can
    only WEAKEN the bound. That is the property the estimator is chosen
    for - a lower bound that can only be too generous - and it must hold:
    a bent series must never bound harder than the straight series it came
    from."""
    points = 236
    u = np.linspace(0.0, 1.0, points)
    straight = _series(36.5, -0.26, 0.01, points, seed=8)
    bent = straight - 0.26 * (-np.expm1(-4.0 * u) / (1.0 - np.exp(-4.0)) - u)
    flat = _series(0.0, 0.0, 0.01, points, seed=9)
    loose = vcr_agc.time_constant_from_drift(bent, flat)
    tight = vcr_agc.time_constant_from_drift(straight, flat)
    assert (loose["time_constant_us_at_least"]
            < tight["time_constant_us_at_least"])


def test_a_straight_series_bounds_the_time_constant_below():
    """No bend, so only slow loops survive - and the bound scales with the
    drift-to-scatter ratio the data affords."""
    flat = _series(0.0, 0.0, 0.02, seed=13)
    quiet = vcr_agc.time_constant_from_drift(
        _series(36.5, -0.26, 0.005, seed=14), flat)
    noisy = vcr_agc.time_constant_from_drift(
        _series(36.5, -0.26, 0.060, seed=15), flat)
    assert (quiet["time_constant_us_at_least"]
            > noisy["time_constant_us_at_least"])


def test_the_test_declines_rather_than_inventing_a_bound():
    """When the resolvable ratio exceeds what any relaxation can bend, no
    bound is possible and the module must say so instead of returning one."""
    result = vcr_agc.time_constant_from_drift(
        _series(36.5, 0.003, 0.06, seed=16), _series(0.0, 0.0, 0.06, seed=17))
    assert result["curvature_detectable_at"] > result["curvature_limit"][
        "curvature_over_drift"]
    assert result["window_over_time_constant_at_most"] is None
    assert result["time_constant_us_at_least"] is None
    # and then nothing is excluded, rather than everything
    assert (result["assumed_constants_surviving_us"]
            == [float(t) for t in vcr_agc.ASSUMED_TIME_CONSTANTS_US])


def test_the_two_levels_must_be_read_on_the_same_lines():
    with pytest.raises(ValueError):
        vcr_agc.time_constant_from_drift(np.zeros(236), np.zeros(100))


# ------------------------------------------------- the measured decodes

# Drift and scatter measured on /output/decodes_2026-09-06/ over the 236
# lines of `sync_depth.field_slope`'s window, quoted here so the reading is
# reproducible without the 95 MB captures. Columns: spacing drift, spacing
# scatter, blanking drift (IRE across the window).
MEASURED = {
    "countdown": (-0.2577, 0.0656, -0.5386),
    "home": (-0.0420, 0.0508, +0.3067),
    "pulse-and-bar": (+0.0336, 0.0590, +0.3344),
}


MEAN_SPACING_IRE = 36.4892      # countdown's, over the same window


def test_every_tape_bounds_its_time_constant_past_the_one_millisecond_row():
    """The measured reading: none of the three shows a bend, so all three
    bound their time constant below, and every bound clears 1 ms. The
    bounds differ by a factor of nine because the drifts do."""
    bounds = {}
    for tape, (drift, scatter, blanking) in MEASURED.items():
        result = vcr_agc.time_constant_from_drift(
            _series(MEAN_SPACING_IRE, drift, scatter),
            _series(0.0, blanking, scatter, seed=1))
        bounds[tape] = result["time_constant_us_at_least"]

    assert bounds["countdown"] == pytest.approx(11273.0, rel=0.02)
    assert bounds["home"] == pytest.approx(2076.0, rel=0.02)
    assert bounds["pulse-and-bar"] == pytest.approx(1255.0, rel=0.02)
    # more drift against the same scatter is a better probe of the loop
    assert bounds["countdown"] > bounds["home"] > bounds["pulse-and-bar"]


def test_all_three_leave_only_the_slowest_assumed_constant():
    """The headline: five candidates spanning five decades, narrowed to one
    by the same pair on three tapes independently."""
    for tape, (drift, scatter, blanking) in MEASURED.items():
        result = vcr_agc.time_constant_from_drift(
            _series(MEAN_SPACING_IRE, drift, scatter),
            _series(0.0, blanking, scatter, seed=1))
        assert result["assumed_constants_surviving_us"] == [50000.0], tape

    countdown = vcr_agc.time_constant_from_drift(
        _series(MEAN_SPACING_IRE, *MEASURED["countdown"][:2]),
        _series(0.0, MEASURED["countdown"][2], MEASURED["countdown"][1],
                seed=1))
    # and the drift itself is the 0.71 per cent gain drift reported
    assert countdown["gain_drift_fraction"] == pytest.approx(-0.00706,
                                                             rel=0.01)


def test_pulse_and_bar_is_the_sharpest_disagreement():
    """Its blanking is resolved and its spacing is not - one probe of the
    pair shouting, the other silent."""
    drift, scatter, blanking = MEASURED["pulse-and-bar"]
    result = vcr_agc.time_constant_from_drift(
        _series(MEAN_SPACING_IRE, drift, scatter),
        _series(0.0, blanking, scatter, seed=1))
    assert result["level_moves"] is True
    assert result["gain_moves"] is False
    assert result["the_pair_disagrees"] is True
