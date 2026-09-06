"""THE INSTRUMENT'S ERROR BAR, AUDITED.

Every "times the floor" figure in this arc divides by the sync export's
per-bin standard error, and until now that error bar had never been measured
against anything. `residual_floor.error_bar_audit` measures it against the
export's own independent-half estimates, which are a second reading of the
same noise and can disagree with the first.

The planted cases below are the controls: an error bar that is right, one
that is wrong by a known factor, and one that is wrong in SHAPE. Only the
third is the case the real data turns out to be in, so the discrimination
between the second and the third is what these tests exist to protect.
"""

import os

import numpy as np
import pytest

from vhsdecode.models import residual_floor as rf


EXPORTS = "/output/decodes"


def _planted(rng, cells=40, per_cell=25, se_scale=None, resolution_hz=1e5):
    """Two independent halves of a pooled estimate whose noise is known.

    Each half's per-component variance is twice the pool's, so the complex
    difference carries `HALF_DIFFERENCE_VARIANCE * se^2` when the quoted `se`
    is right. `se_scale` multiplies the QUOTED error bar without touching the
    planted noise, which is exactly how an instrument's error bar goes wrong.
    """
    bins = cells * per_cell
    f = np.arange(bins) * (resolution_hz / per_cell)
    truth = (1.0 + 0.3 * np.cos(2e-6 * f)) * np.exp(-0.5j * 1e-6 * f)
    se = np.full(bins, 0.02)
    # a half's complex variance is 2 se^2, split evenly between the parts
    sigma = se * np.sqrt(rf.HALF_DIFFERENCE_VARIANCE / 2.0 / 2.0)
    draw = lambda: sigma * (rng.standard_normal(bins)
                            + 1j * rng.standard_normal(bins))
    first, second = truth + draw(), truth + draw()
    quoted = se if se_scale is None else se * np.asarray(se_scale)
    return f, truth, quoted, first, second, resolution_hz


def test_the_cells_are_the_instruments_own_points_not_its_bins():
    f = np.arange(1000) * 1e3
    cells = rf.resolution_cells(f, 5e4)
    assert cells["bins"] == 1000
    assert cells["count"] == 20
    assert cells["oversampling"] == pytest.approx(50.0)
    # the oversampling is what a bin-counting error bar gets wrong
    assert np.sqrt(cells["oversampling"]) == pytest.approx(7.071, rel=1e-3)
    with pytest.raises(ValueError):
        rf.resolution_cells(f, 0.0)


def test_a_correct_error_bar_reads_one_and_is_not_called_frequency_dependent():
    f, _t, se, a, b, res = _planted(np.random.default_rng(0))
    out = rf.error_bar_audit(f, np.ones_like(f, dtype=complex), se, a, b, res)
    assert out["pooled_ratio"] == pytest.approx(1.0, rel=0.06)
    assert not out["frequency_dependent"]
    assert out["reading"] == "the error bar agrees with the second measurement"
    assert abs(out["trend_z"]) < rf.SIGMA_THRESHOLD


def test_an_error_bar_wrong_by_a_flat_factor_is_recovered_as_a_factor():
    # quote the error bar twice too large in amplitude: four times in variance
    f, _t, se, a, b, res = _planted(np.random.default_rng(1), se_scale=2.0)
    out = rf.error_bar_audit(f, np.ones_like(f, dtype=complex), se, a, b, res)
    assert out["pooled_ratio"] == pytest.approx(0.25, rel=0.08)
    assert out["se_should_scale_by"] == pytest.approx(0.5, rel=0.05)
    # a flat error is a factor, and must NOT be reported as a shape
    assert not out["frequency_dependent"]
    assert "single rescaling" in out["reading"]


def test_an_error_bar_wrong_in_shape_is_told_apart_from_one_wrong_by_a_factor():
    """THE DISCRIMINATION THE AUDIT EXISTS FOR.

    A single pooled ratio cannot tell a factor from a shape: it reports the
    band-weighted average of either. The per-cell trend can, and this is the
    case the real exports turn out to be in.
    """
    rng = np.random.default_rng(2)
    f, _t, se, a, b, res = _planted(rng)
    # the quoted bar is right at the bottom of the band and sixteen times too
    # large in variance at the top - a shape, not a factor
    tilt = np.sqrt(1.0 + 15.0 * (f - f.min()) / np.ptp(f))
    out = rf.error_bar_audit(f, np.ones_like(f, dtype=complex), se * tilt,
                             a, b, res)
    assert out["frequency_dependent"]
    assert out["trend_z"] < -rf.SIGMA_THRESHOLD
    assert out["low_band_ratio"] > 3.0 * out["high_band_ratio"]
    assert "SHAPE" in out["reading"]
    # and no single number describes it: the pooled ratio a factor-reading
    # would quote lies strictly between the two ends and is right at neither
    assert out["high_band_ratio"] < out["pooled_ratio"] < out["low_band_ratio"]


def test_a_gain_and_delay_drift_between_the_halves_is_attributed_to_drift():
    """A slow term is the competing explanation for an excess, so the audit
    must be able to charge one when it is there."""
    rng = np.random.default_rng(3)
    f, truth, se, a, b, res = _planted(rng)
    centred = (f - f.mean()) / np.abs(f - f.mean()).max()
    drifted = b + truth * (0.20 + 0.30j * centred)
    out = rf.error_bar_audit(f, truth, se, a, drifted, res)
    assert out["pooled_ratio"] > 5.0
    assert out["drift_share"] > 0.9
    # with the drift removed the error bar reads correct again
    clean = rf.error_bar_audit(f, truth, se, a, b, res)
    assert clean["drift_share"] < 0.2


def test_the_phase_dimension_the_single_real_error_bar_cannot_carry():
    """One real `se` per bin asserts the complex noise is circular. Rotating
    the difference by the transfer's own phase splits it into an amplitude
    error and a timing one, which is the axis `se` is silent on."""
    rng = np.random.default_rng(4)
    f, truth, se, a, _b, res = _planted(rng)
    unit = truth / np.abs(truth)
    sigma = se * np.sqrt(rf.HALF_DIFFERENCE_VARIANCE / 4.0)
    # noise three times stronger along the transfer than across it
    anisotropic = truth + unit * sigma * (3.0 * rng.standard_normal(f.size)
                                          + 1j * rng.standard_normal(f.size))
    out = rf.error_bar_audit(f, truth, se, a, anisotropic, res)
    assert out["anisotropy"] > 3.0
    assert out["anisotropy_z"] > rf.SIGMA_THRESHOLD
    assert not out["circular"]
    round_ = rf.error_bar_audit(f, truth, se, a, _b, res)
    assert round_["circular"]


def test_the_two_se_conventions_are_reported_and_are_exactly_two_apart():
    """The tree carries two readings of `se` differing by exactly two in
    variance, which is the size of several of the discrepancies being argued
    about, so the audit reports both rather than choosing."""
    f, _t, se, a, b, res = _planted(np.random.default_rng(5))
    out = rf.error_bar_audit(f, np.ones_like(f, dtype=complex), se, a, b, res)
    assert out["variance_factor"] == rf.HALF_DIFFERENCE_VARIANCE
    assert out["pooled_ratio_other_convention"] == pytest.approx(
        out["pooled_ratio"] / 2.0, rel=1e-12)
    other = rf.error_bar_audit(
        f, np.ones_like(f, dtype=complex), se, a, b, res,
        variance_factor=rf.HALF_DIFFERENCE_VARIANCE_PER_COMPONENT)
    assert other["pooled_ratio"] == pytest.approx(out["pooled_ratio"] / 2.0,
                                                  rel=1e-12)
    # and the shape verdict survives the choice, which is the point
    assert other["trend_rho"] == pytest.approx(out["trend_rho"], rel=1e-12)


def test_the_honest_degrees_of_freedom_are_the_cells_and_not_the_bins():
    f, _t, se, a, b, res = _planted(np.random.default_rng(6))
    out = rf.error_bar_audit(f, np.ones_like(f, dtype=complex), se, a, b, res)
    assert out["bins"] == 1000 and out["independent_points"] == 40
    assert out["quoted_dof"] == rf.REAL_PARTS * 1000
    assert out["honest_dof"] == rf.REAL_PARTS * 40
    assert out["error_bar_understated_by"] == pytest.approx(5.0, rel=1e-9)


def test_the_verdicts_distribution_field_names_a_distribution():
    """A regression: `verdict` asked `classify` for keys it does not return
    ("verdict", "best"), so this field was None on every call ever made and
    the held-out pair had nothing in it to disagree about."""
    rng = np.random.default_rng(7)
    floor = np.full(512, 1e-4)
    noise = np.sqrt(floor / rf.REAL_PARTS) * (rng.standard_normal(512)
                                              + 1j * rng.standard_normal(512))
    out = rf.verdict(noise, floor)
    assert out["distribution"] is not None
    assert out["distribution"] == out["distribution_detail"]["distribution"]
    assert out["distribution"] == "gaussian"


@pytest.mark.parametrize("tape", ["cd", "home", "pnb"])
def test_the_measured_error_bar_runs_with_frequency_on_every_export(tape):
    """MEASURED (2026-09-06). On all twelve tape/head/view combinations the
    per-cell ratio falls with frequency: twelve negative rank correlations
    out of twelve is a sign test at p = 2^-12. The error bar is wrong in
    shape, and the arc's competing single factors (3.7 to 4.0 one way on
    home, about 2 the other on countdown) are that shape read through
    different band weightings."""
    path = os.path.join(EXPORTS, f"ss_{tape}_off_sync_step_response.npz")
    if not os.path.exists(path):
        pytest.skip("the sync exports are not present in this checkout")
    for head in ("a", "b"):
        for view in ("fall", "rise"):
            export = rf.load_export(path, head, view=view)
            audit = rf.error_bar_audit(
                export["frequency_hz"], export["H"], export["se"],
                export["held_out"]["H_fit"], export["held_out"]["H_judge"],
                export["resolution_hz"])
            assert audit["trend_rho"] < 0.0, (tape, head, view)
            assert audit["independent_points"] < audit["bins"] / 25
            # and it is not a gain-and-delay drift between the halves
            assert audit["drift_share"] < 0.35, (tape, head, view)


class TestDeclaredFloor:
    """R5: "the floor is DECLARED rather than assumed".

    It is the stopping rule of the whole method, and until now it was
    implemented twice - once in a debug plot, once in the noise budget -
    with no test holding either to the relation.
    """

    def test_the_constants_are_derived_not_quoted(self):
        """Both are properties of the normal distribution, so they are
        checked against their closed forms rather than against the digits
        someone typed."""
        from scipy.stats import norm

        assert rf.MAD_TO_SIGMA == pytest.approx(1.0 / norm.ppf(0.75), rel=1e-12)
        assert rf.MAD_TO_SIGMA == pytest.approx(1.4826, abs=1e-4)
        assert rf.MEDIAN_STANDARD_ERROR_FACTOR == pytest.approx(
            np.sqrt(np.pi / 2.0), rel=1e-12)
        assert rf.MEDIAN_STANDARD_ERROR_FACTOR == pytest.approx(1.2533, abs=1e-4)
        assert 1.0 / rf.IQR_TO_SIGMA == pytest.approx(1.349, abs=1e-3)

    def test_robust_sigma_recovers_a_planted_scale(self):
        generator = np.random.default_rng(20260906)
        sample = generator.normal(3.0, 2.5, 20000)
        assert rf.robust_sigma(sample) == pytest.approx(2.5, rel=0.03)
        assert rf.robust_sigma(sample, "iqr") == pytest.approx(2.5, rel=0.03)

    def test_robust_sigma_survives_the_low_tail_a_dropout_makes(self):
        """The reason it is robust at all: dropouts sit in the low tail and
        a plain standard deviation lets them set the noise scale."""
        generator = np.random.default_rng(7)
        sample = generator.normal(0.0, 1.0, 4000)
        sample[:200] = -60.0                       # five per cent dropouts
        # IT IS ROBUST, NOT IMMUNE. Five per cent contamination in ONE tail
        # pulls the median a little and inflates the MAD with it - measured
        # here at 7.4 per cent high. The MAD's breakdown point is 50 per
        # cent, so it survives; the bias is stated rather than hidden
        # behind a loose tolerance, because a floor 7 per cent high is a
        # stopping rule 7 per cent early.
        assert rf.robust_sigma(sample) == pytest.approx(1.074, rel=0.02)
        # against a plain standard deviation, which the dropouts own
        assert np.std(sample) > 5.0

    def test_the_median_floor_is_25_per_cent_above_the_means(self):
        """The price of a robust centre, and the trap: a floor quoting the
        mean's error while the estimator takes a median is 25 per cent too
        low, so a residual between the two reads as converged."""
        generator = np.random.default_rng(11)
        got = rf.declared_floor(generator.normal(0.0, 1.0, 1000))
        assert got["floor"] / got["mean_floor"] == pytest.approx(
            np.sqrt(np.pi / 2.0), rel=1e-12)

    def test_the_floor_falls_as_root_n(self):
        """Root-N is the whole reason for accumulating, and a floor that
        does not carry it throws that away."""
        generator = np.random.default_rng(3)
        one = rf.declared_floor(generator.normal(0.0, 1.0, 250))["floor"]
        four = rf.declared_floor(generator.normal(0.0, 1.0, 1000))["floor"]
        assert one / four == pytest.approx(2.0, rel=0.12)

    def test_an_accumulated_profile_takes_the_count_it_was_built_from(self):
        """THE 45x ERROR, held. A profile accumulated over N lines has a
        floor root-N below one line's; using the per-line scatter for it
        overstates the floor by exactly root N and stops the derivation
        early."""
        generator = np.random.default_rng(5)
        per_line = generator.normal(0.0, 1.0, 200)
        naive = rf.declared_floor(per_line)["floor"]
        accumulated = rf.declared_floor(per_line, count=200 * 2016)["floor"]
        assert naive / accumulated == pytest.approx(np.sqrt(2016), rel=0.02)

    def test_noise_reaches_the_floor_and_structure_does_not(self):
        """The verdict R5 exists to deliver."""
        generator = np.random.default_rng(13)
        reference = generator.normal(0.0, 1.0, 4000)
        noise = generator.normal(0.0, 1.0, 4000) / np.sqrt(4000)
        assert rf.at_declared_floor(noise, reference)["at_floor"]
        structure = 0.05 * np.sin(np.linspace(0.0, 20.0, 4000))
        verdict = rf.at_declared_floor(structure, reference)
        assert not verdict["at_floor"]
        assert verdict["ratio"] > 1.0

    def test_the_ratio_is_reported_not_just_the_verdict(self):
        """"Eight to twelve times the floor" is the statement this arc has
        needed repeatedly; a bare boolean throws it away."""
        generator = np.random.default_rng(17)
        reference = generator.normal(0.0, 1.0, 1000)
        floor = rf.declared_floor(reference)["floor"]
        verdict = rf.at_declared_floor(np.full(1000, 9.0 * floor), reference)
        assert verdict["ratio"] == pytest.approx(9.0, rel=0.01)
