"""Loss-mechanism separation as an identifiability test.

Two clusters cannot split three mechanisms: the test refuses and names the
collinear pair. Three clusters that actually carry the shapes can, and the
planted parameters come back within their error bars. The composite is
returned in both cases, because the composite is what the data determines.
"""

import os

import numpy as np
import pytest

from vhsdecode.models import head_model, output_limit
from vhsdecode.models import loss_separation as ls

SPEED = ls.writing_speed()
SHARED = "/tmp/claude-1000/-workspaces-vhs-decode/shared/"
CD_EXPORT = SHARED + "cd_tree_noeq_sync_step_response.npz"


def planted(clusters, spacing_m, gap_m, thickness_m, levels, noise=0.0, seed=0):
    """The composite each cluster would measure for these parameters, with
    Gaussian noise at each cluster's SE scaled by `noise`."""
    rng = np.random.default_rng(seed)
    out = []
    for c, level in zip(clusters, levels):
        logged = ls.composite_log(
            c["wavelength_m"], spacing_m + c.get("spacing_offset_m", 0.0),
            gap_m, thickness_m, level)
        if noise:
            logged = logged + rng.normal(0.0, 1.0, logged.shape) * c["se_nepers"] * noise
        out.append(logged)
    return out


class TestForms:
    """The wavelength forms are the head model's forms, and the Jacobian is
    the derivative of the forms."""

    def test_wavelength_forms_match_the_head_model(self):
        f = np.linspace(0.5e6, 8e6, 64)
        lam = ls.wavelength_m(f, SPEED)
        width = head_model.mechanics_for("VHS", "NTSC", "SP")["track_width_m"]
        # log_response carries the differentiation term log f; the loss
        # forms are what is left once it is removed
        for mine, kw in (
                (ls.spacing_log(lam, 0.4e-6), {"spacing_m": 0.4e-6}),
                (ls.gap_log(lam, 0.3e-6), {"gap_m": 0.3e-6}),
                (ls.thickness_log(lam, 0.3e-6), {"thickness_m": 0.3e-6})):
            expected = head_model.log_response(f, SPEED, width, **kw) - np.log(f)
            assert np.allclose(mine, expected, atol=1e-9)

    def test_jacobian_matches_finite_differences(self):
        lam = np.array([1.3e-6, 1.5e-6, 1.7e-6, 4e-6, 9.2e-6, 50e-6])
        values = {"spacing": 0.48e-6, "gap": 0.30e-6, "thickness": 0.30e-6}
        analytic = ls.jacobian(lam, values["spacing"], values["gap"],
                               values["thickness"])
        step = 1e-6
        for k, name in enumerate(ls.MECHANISMS):
            up, down = dict(values), dict(values)
            up[name] = values[name] * np.exp(step)
            down[name] = values[name] * np.exp(-step)
            numeric = (ls.composite_log(lam, up["spacing"], up["gap"], up["thickness"])
                       - ls.composite_log(lam, down["spacing"], down["gap"],
                                          down["thickness"])) / (2 * step)
            assert np.max(np.abs(analytic[:, k] - numeric)) < 1e-7

    def test_thickness_sensitivity_has_two_regimes(self):
        """Thin: a slope, -x/2, proportional to 1/lambda like spacing.
        Thick: a level, -1, at every wavelength."""
        lam = np.array([1.5e-6, 9.2e-6])
        thin = ls.fractional_sensitivities(lam, 0.48e-6, 0.3e-6, 0.01e-6)["thickness"]
        x = 2 * np.pi * 0.01e-6 / lam
        assert np.allclose(thin, -x / 2, rtol=0.05)
        thick = ls.fractional_sensitivities(lam, 0.48e-6, 0.3e-6, 4.5e-6)["thickness"]
        assert np.allclose(thick[0], -1.0, atol=1e-6)

    def test_gap_sign_is_positive_over_every_cluster(self):
        lam = np.linspace(1.3e-6, 100e-6, 200)
        assert np.all(ls.gap_sign(lam, ls.SPEC_GAP_M) > 0)
        assert np.all(ls.gap_phase_rad(lam, ls.SPEC_GAP_M) == 0.0)
        # and it does flip past the first null, at lambda = g_eff
        null = ls.EFFECTIVE_GAP_FACTOR * ls.SPEC_GAP_M
        assert ls.gap_sign(np.array([0.9 * null]), ls.SPEC_GAP_M)[0] < 0
        assert ls.gap_phase_rad(np.array([0.9 * null]), ls.SPEC_GAP_M)[0] == pytest.approx(np.pi)


class TestWavelengthTable:
    """Ethan's reference numbers, from specification."""

    def test_reference_numbers(self):
        table = ls.wavelength_table()
        assert table["writing_speed_m_s"] == pytest.approx(5.80)
        assert table["four_fsc_hz"] == pytest.approx(14.318181e6, rel=1e-6)
        assert table["sample_period_s"] * 1e9 == pytest.approx(69.84, abs=0.01)
        assert table["tape_per_sample_m"] * 1e6 == pytest.approx(0.405, abs=0.001)
        rows = table["rows"]
        assert rows["sync_tip"]["wavelength_m"] * 1e6 == pytest.approx(1.71, abs=0.005)
        assert rows["peak_white"]["wavelength_m"] * 1e6 == pytest.approx(1.32, abs=0.005)
        assert rows["colour_under"]["wavelength_m"] * 1e6 == pytest.approx(9.22, abs=0.005)
        assert rows["hifi_ch1"]["wavelength_m"] * 1e6 == pytest.approx(4.46, abs=0.005)
        assert rows["hifi_ch2"]["wavelength_m"] * 1e6 == pytest.approx(3.41, abs=0.005)
        # read depth lambda / 2 pi: ~0.2 um luma, ~1.5 um chroma
        assert rows["mid_band"]["read_depth_m"] * 1e6 == pytest.approx(0.237, abs=0.002)
        assert rows["colour_under"]["read_depth_m"] * 1e6 == pytest.approx(1.47, abs=0.01)

    def test_thickness_loss_at_the_assumed_coating_and_the_spec_depth(self):
        rows = ls.wavelength_table()["rows"]
        # Ethan: about -26 dB luma, -10 dB chroma at 4-5 um
        assert rows["mid_band"]["thickness_loss_db"]["5.0 um"] == pytest.approx(-26.5, abs=0.1)
        assert rows["peak_white"]["thickness_loss_db"]["4.0 um"] == pytest.approx(-25.6, abs=0.1)
        assert rows["colour_under"]["thickness_loss_db"]["4.0 um"] == pytest.approx(-9.3, abs=0.1)
        assert rows["colour_under"]["thickness_loss_db"]["5.0 um"] == pytest.approx(-10.9, abs=0.1)
        # and the JVC recorded depth, which is a different answer by 21 dB
        assert rows["mid_band"]["thickness_loss_db"]["0.3 um"] == pytest.approx(-4.9, abs=0.1)


class TestRefusal:
    """Two clusters, three mechanisms: refused, with the pair named."""

    def clusters(self, se=4e-3):
        return [ls.luma_cluster(se, points=20), ls.chroma_cluster(se)]

    def test_two_clusters_are_refused_with_the_collinear_pair_named(self):
        clusters = self.clusters()
        measured = planted(clusters, 0.48e-6, 0.30e-6, 0.30e-6, [0.1, -0.2])
        out = ls.separate(clusters, measured)
        assert out["refused"]
        assert "fitted" not in out
        verdict = out["verdict"]
        assert verdict["shared"]["rank_at_noise"] < 3
        assert verdict["shared"]["structural_rank"] == 3
        assert set(verdict["proof"]["collinear_pair"]) == {"spacing", "thickness"}
        assert verdict["proof"]["coherence"] > 0.99
        assert verdict["proof"]["correlation"] > 0.99
        assert "spacing" in verdict["why"] and "thickness" in verdict["why"]

    def test_the_chroma_carrier_with_its_own_level_adds_nothing(self):
        luma, chroma = self.clusters()
        alone = ls.identifiable([luma])["shared"]
        both = ls.identifiable([luma, chroma])["shared"]
        assert np.allclose(alone["singular_values"], both["singular_values"])

    def test_thick_coating_makes_thickness_a_level(self):
        """In the coating regime the level absorbs thickness and the worst
        pair becomes spacing-gap."""
        verdict = ls.identifiable(self.clusters(), {"thickness_m": 4.5e-6})
        assert verdict["refused"]
        assert verdict["shared"]["column_norm_relative"]["thickness"] < 1e-4
        assert set(verdict["shared"]["worst_pair"]) == {"spacing", "gap"}

    def test_the_composite_is_recovered_under_refusal(self):
        clusters = self.clusters()
        measured = planted(clusters, 0.48e-6, 0.30e-6, 0.30e-6, [0.1, -0.2])
        verdict = ls.identifiable(clusters)
        composite = ls.composite_only(clusters[0]["wavelength_m"], measured[0],
                                      clusters[0]["se_nepers"],
                                      identifiability=verdict)
        assert np.allclose(composite["log_nepers"], measured[0])
        assert np.allclose(composite["db"], measured[0] * ls.DB_PER_NEPER)
        for mechanism in ls.MECHANISMS:
            assert composite["components"][mechanism]["status"] == "unidentified"
            assert composite["components"][mechanism]["proof"]["collinear_pair"]

    def test_hifi_with_its_own_head_leaves_the_split_where_it_was(self):
        luma, chroma = self.clusters()
        before = ls.identifiable([luma, chroma])["shared"]["fractional_se"]
        after = ls.identifiable([luma, chroma, ls.hifi_cluster(4e-3, points=20,
                                                               own_head=True)]
                                )["shared"]["fractional_se"]
        for name in ls.MECHANISMS:
            assert after[name] == pytest.approx(before[name], rel=0.02)


class TestRecovery:
    """Three clusters that carry the shapes: granted, and recovered."""

    def test_three_clusters_recover_within_error(self):
        se = 1e-3
        clusters = [ls.luma_cluster(se, points=24),
                    ls.chroma_cluster(se, points=24, sidebands=True),
                    ls.hifi_cluster(se, points=24, own_head=False)]
        truth = {"spacing": 0.42e-6, "gap": 0.34e-6, "thickness": 4.0e-6}
        levels = [0.15, -0.30, 0.05]
        measured = planted(clusters, truth["spacing"], truth["gap"],
                           truth["thickness"], levels, noise=1.0, seed=3)
        out = ls.separate(clusters, measured, {"thickness_m": 4.5e-6})
        assert not out["refused"], out.get("why")
        assert out["verdict"]["shared"]["rank_at_noise"] == 3
        for name, value in truth.items():
            fitted = out["fitted"][name]
            assert abs(fitted["value"] - value) < 3 * fitted["se_m"] + 1e-12, (name, fitted)
            assert fitted["se_fractional"] < 0.1
        for c, level in zip(clusters, levels):
            fitted = out["fitted"]["%s:level" % c["name"]]
            assert abs(fitted["value"] - level) < 3 * fitted["se"] + 1e-9
        # and the composite each cluster measured is reproduced by the fit
        for c, logged in zip(clusters, measured):
            model = ls.composite_log(
                c["wavelength_m"], out["fitted"]["spacing"]["value"],
                out["fitted"]["gap"]["value"], out["fitted"]["thickness"]["value"],
                out["fitted"]["%s:level" % c["name"]]["value"])
            assert np.sqrt(np.mean((model - logged) ** 2)) < 3 * se

    def test_the_composite_is_recovered_when_granted(self):
        se = 1e-3
        clusters = [ls.luma_cluster(se, points=24),
                    ls.chroma_cluster(se, points=24, sidebands=True),
                    ls.hifi_cluster(se, points=24, own_head=False)]
        measured = planted(clusters, 0.42e-6, 0.34e-6, 4.0e-6, [0.0, 0.0, 0.0])
        verdict = ls.identifiable(clusters, {"thickness_m": 4.5e-6})
        composite = ls.composite_only(clusters[1]["wavelength_m"], measured[1],
                                      clusters[1]["se_nepers"],
                                      identifiability=verdict)
        assert np.allclose(composite["log_nepers"], measured[1])
        assert composite["components"]["thickness"]["status"].startswith("resolved")


class TestLeverage:
    def test_luma_only_is_never_determined(self):
        luma = [ls.luma_cluster(4e-3, points=20)]
        for thicknesses in ([0.30e-6, 0.35e-6, 0.40e-6], [4e-6, 5e-6, 4.5e-6]):
            out = ls.formulation_leverage(thicknesses, luma, se_nepers=4e-3)
            assert out["determined_at_k"] is None
            assert out["per_k"][0]["unknowns_mechanisms"] == 3
            assert out["per_k"][-1]["unknowns_mechanisms"] == 5

    def test_sidebands_in_the_coating_regime_are_determined_at_two_stocks(self):
        base = [ls.luma_cluster(4e-3, points=20),
                ls.chroma_cluster(4e-3, points=20, sidebands=True)]
        out = ls.formulation_leverage([4e-6, 5e-6, 4.5e-6], base, se_nepers=4e-3)
        assert out["determined_at_k"] == 2
        assert not out["per_k"][0]["determined"]
        assert out["per_k"][1]["determined"]

    def test_per_stock_spacing_counts_one_plus_two_k(self):
        base = [ls.luma_cluster(4e-3, points=20)]
        out = ls.formulation_leverage([4e-6, 5e-6], base, se_nepers=4e-3,
                                      shared_spacing=False)
        assert out["per_k"][1]["unknowns_mechanisms"] == 5
        assert "spacing@stock1" in out["per_k"][1]["fractional_se"]


class TestDemodulated:
    """The exports are demodulated: the carrier law annihilates spacing."""

    def synthetic_export(self, se=4e-3):
        f = np.linspace(0.05e6, 3.5e6, 20)
        return {"frequency_hz": f, "se_nepers": np.full(20, se),
                "independent": np.arange(20),
                "carriers_hz": {"fall": 3.4e6, "rise": 3.69e6},
                "median_se_nepers": se}

    def test_carrier_law_annihilates_spacing(self):
        out = ls.demodulated_identifiable(self.synthetic_export())
        assert out["refused"]
        assert "spacing" in out["annihilated"]
        assert out["column_norm_relative"]["spacing"] < 1e-12
        assert out["fractional_se"]["spacing"] == np.inf
        assert out["structural_rank"] == 2

    @pytest.mark.skipif(not os.path.exists(CD_EXPORT),
                        reason="the shared export is not on this machine")
    def test_the_real_export_is_refused(self):
        export = ls.load_sync_step_export(CD_EXPORT, "a")
        assert len(export["independent"]) == 20
        assert 0.003 < export["median_se_nepers"] < 0.006
        out = ls.demodulated_identifiable(export)
        assert out["refused"] and "spacing" in out["annihilated"]
        assert out["rank_at_noise"] == 1


HOME_EXPORT = SHARED + "home_tree_noeq_sync_step_response.npz"
PNB_EXPORT = SHARED + "pnb_tree_noeq_sync_step_response.npz"


def write_export(path, drift=None, level=0.0, noise=0.0, seed=0, bins=2049,
                 rate_hz=output_limit.FOUR_FSC_HZ):
    """A minimal `sync_step_response` export with a PLANTED half difference.

    `drift` is a function of frequency in hertz returning the log-magnitude
    the first half carries above the second, so a test can put in a known
    departure and read it back out. Only the keys the half loader and the
    output-visibility conversion touch are written; the grid is the arc's
    own 4 f_sc output rate, from `output_limit`, and the standard error is
    a round fixture value chosen so a planted departure of ten times it is
    unmistakable."""
    rng = np.random.default_rng(seed)
    freq_mhz = np.linspace(0.0, rate_hz / 2e6, bins)
    f = freq_mhz * 1e6
    shape = np.zeros_like(f) if drift is None else np.asarray(drift(f), float)
    data = {"frequency_mhz": freq_mhz,
            "rate_mhz": np.array(rate_hz / 1e6),
            "metadata": np.array("planted export for the half-drift test")}
    # a smooth roll-off so the magnitude is bounded away from zero
    base = np.exp(-f / 4e6)
    se = np.full(bins, 0.004) * base
    valid = (f >= 0.04e6) & (f <= 3.5e6)
    for head in ("a", "b"):
        for view, carrier in (("fall", 3.4e6), ("rise", 3.69e6)):
            key = "head_%s_%s" % (head, view)
            wobble = rng.normal(0.0, noise, bins) if noise else 0.0
            data[key + "_H"] = base.astype(np.complex128)
            data[key + "_H_first"] = (base * np.exp(0.5 * shape + 0.5 * level
                                                    + wobble)).astype(np.complex128)
            data[key + "_H_second"] = (base * np.exp(-0.5 * shape - 0.5 * level
                                                     - wobble)).astype(np.complex128)
            data[key + "_se"] = se
            data[key + "_valid"] = valid
            data[key + "_resolution_mhz"] = np.array(0.1812)
            data[key + "_carrier_hz"] = np.array(carrier)
        data["head_%s_fall_mean" % head] = np.concatenate(
            [np.zeros(32), np.full(47, -40.0)])
    np.savez(path, **data)
    return str(path)


class TestTheTimeAxis:
    """The record's two halves are a measurement pair, and the module had no
    axis to put them on. These check the arithmetic of the pair, that a
    planted drift comes back, and that a level cannot masquerade as one."""

    def test_the_half_error_is_root_two_and_the_difference_is_twice(self, tmp_path):
        path = write_export(tmp_path / "flat.npz")
        loaded = ls.load_sync_step_halves(path, "a")
        idx = loaded["independent"]
        full = ls.load_sync_step_halves(path, "a")["first"]
        # each half pools half the fields: root-two the full accumulation
        ratio = full["se_nepers"][idx] / (loaded["se_difference_nepers"][idx] / 2.0)
        assert np.allclose(ratio, np.sqrt(2.0))

    def test_the_even_part_is_the_half_sum_of_the_polarities(self, tmp_path):
        path = write_export(tmp_path / "flat.npz",
                            drift=lambda f: 0.05 * np.sin(f / 4e5))
        z = np.load(path)
        even = ls.load_sync_step_halves(path, "a", "even")
        fall = ls.load_sync_step_halves(path, "a", "fall")
        rise = ls.load_sync_step_halves(path, "a", "rise")
        idx = even["independent"]
        direct = np.log(np.abs(
            0.5 * (z["head_a_fall_H_first"] + z["head_a_rise_H_first"])))
        assert np.allclose(even["first"]["log_nepers"][idx], direct[idx])
        # and the two polarities are separately available, which is what
        # tells a drift from a polarity contrast
        assert fall["view"] == "fall" and rise["view"] == "rise"

    def test_identical_halves_do_not_drift(self, tmp_path):
        path = write_export(tmp_path / "null.npz")
        out = ls.half_drift(path, "a")
        assert out["usable"]
        assert out["chi2"] < 1e-12
        assert not out["drifts"]
        assert out["p_value"] > 0.99

    def test_a_planted_shape_comes_back_and_is_flagged(self, tmp_path):
        # a departure of ten times the export's own error, so it must be seen
        def planted_drift(f):
            return 0.04 * np.cos(2.0 * np.pi * f / 2.0e6)
        path = write_export(tmp_path / "drift.npz", drift=planted_drift)
        out = ls.half_drift(path, "a")
        assert out["drifts"] and out["p_value"] < 1e-6
        expected = planted_drift(out["frequency_hz"])
        assert np.allclose(out["difference_nepers"], expected, atol=1e-9)

    def test_a_level_is_projected_out_and_does_not_read_as_a_drift(self, tmp_path):
        path = write_export(tmp_path / "level.npz", level=0.05)
        out = ls.half_drift(path, "a")
        # the level is measured
        assert out["level_nepers"] == pytest.approx(0.05, abs=1e-9)
        # and it is not a drift: every fit here leaves a level free
        assert out["chi2"] < 1e-12
        assert not out["drifts"]

    def test_the_halves_feed_the_existing_identifiability_unchanged(self, tmp_path):
        path = write_export(tmp_path / "flat.npz")
        loaded = ls.load_sync_step_halves(path, "a")
        for label in ("first", "second"):
            out = ls.demodulated_identifiable(loaded[label])
            assert out["refused"]
            assert "spacing" in out["annihilated"]

    def test_output_visibility_uses_the_files_own_floor(self, tmp_path):
        path = write_export(tmp_path / "big.npz", drift=lambda f: 0.2 + 0.0 * f)
        seen = ls.output_visibility(path, "a")
        floor = output_limit.output_profile()["quantisation_noise_ire"]
        assert seen["floor_ire"] == pytest.approx(floor)
        assert seen["meaningful"] and seen["over_floor"] > 1.0
        # and a departure far below the floor is invisible, not small
        tiny = ls.output_visibility(
            write_export(tmp_path / "tiny.npz",
                         drift=lambda f: 1e-5 + 0.0 * f), "a")
        assert not tiny["meaningful"]

    def test_composite_over_time_returns_two_composites(self, tmp_path):
        path = write_export(tmp_path / "drift.npz",
                            drift=lambda f: 0.04 * np.cos(f / 3e5))
        out = ls.composite_over_time(path, "a")
        assert not out["one_measurement"]
        for label in ("first", "second"):
            assert out[label]["log_nepers"].shape == out["wavelength_m"].shape
            # the module still refuses to name the mechanisms
            assert all(entry["status"] == "unidentified"
                       for entry in out[label]["components"].values())

    @pytest.mark.skipif(not os.path.exists(HOME_EXPORT),
                        reason="the shared export is not on this machine")
    def test_home_drifts_and_pnb_does_not(self):
        """The recorded verdict, on the real records: home's two halves are
        not the same measurement and pnb's are."""
        for head, ceiling in (("a", 0.05), ("b", 0.01)):
            out = ls.half_drift(HOME_EXPORT, head)
            assert out["drifts"], head
            assert out["p_value"] < ceiling
            assert out["chi2_per_dof"] > 1.5
        for head in ("a", "b"):
            out = ls.half_drift(PNB_EXPORT, head)
            assert not out["drifts"], head
            assert out["chi2_per_dof"] < 1.0
            # and on pnb the difference is under the file's own floor
            assert not out["output"]["meaningful"]

    @pytest.mark.skipif(not os.path.exists(CD_EXPORT),
                        reason="the shared export is not on this machine")
    def test_cds_disagreement_is_polarity_owned_not_a_drift(self):
        """It stands on the fall view and cancels in the even part, which is
        the part the LTI mechanisms live in."""
        for head in ("a", "b"):
            assert ls.half_drift(CD_EXPORT, head, "fall")["drifts"]
            assert not ls.half_drift(CD_EXPORT, head, "rise")["drifts"]
            assert not ls.half_drift(CD_EXPORT, head, "even")["drifts"]
