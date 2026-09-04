"""The physical head model, and the gain/spacing separation in particular.

The module carries the format's mechanical specification and the fitting
that turns a measured response into physical parameters, so what is tested
is that the specification figures are the specification's, and that the
fitted parameters mean what they are named.
"""

import numpy as np
import pytest

from vhsdecode.models import head_model as hm


def mechanics():
    return hm.mechanics_for("VHS", "NTSC", "SP", {"video_track_width": 58.0})


def band(low=3.0e6, high=5.2e6, count=32):
    return np.linspace(low, high, count)


def planted(gain_db=0.0, spacing_m=None):
    """One head differing from a reference head by the given parameters."""
    head = dict(hm.TYPICAL)
    head["gain_db"] = gain_db
    if spacing_m is not None:
        head["spacing_m"] = spacing_m
    mech = mechanics()
    return hm.head_difference_log(band(), hm.writing_speed(mech),
                                  mech["track_width_m"], head,
                                  dict(hm.TYPICAL))


class TestSpecification:
    """The mechanical figures are the format's, not tuning."""

    def test_writing_speed_is_the_stated_figure(self):
        # the JVC guide states it directly; deriving it from the drum's
        # circumference gives 5.877, which is 1.3% high and would carry
        # into every fitted spacing
        assert hm.writing_speed(mechanics()) == pytest.approx(5.80)

    def test_pal_differs_from_ntsc(self):
        pal = hm.mechanics_for("VHS", "PAL", "SP")
        assert hm.writing_speed(pal) == pytest.approx(4.85)
        assert pal["drum_revolutions_per_second"] == pytest.approx(25.0)

    def test_track_width_prefers_the_repo_parameter(self):
        supplied = hm.mechanics_for("VHS", "NTSC", "SP",
                                    {"video_track_width": 49.0})
        assert supplied["track_width_m"] == pytest.approx(49e-6)


class TestLogResponse:
    def test_no_response_at_or_below_zero_frequency(self):
        mech = mechanics()
        out = hm.log_response(np.array([-1.0e6, 0.0, 1.0e6]),
                              hm.writing_speed(mech), mech["track_width_m"],
                              **hm.TYPICAL)
        assert not np.isfinite(out[0]) and not np.isfinite(out[1])
        assert np.isfinite(out[2])

    def test_gain_absent_leaves_the_response_untouched(self):
        """The new term must not perturb any existing caller."""
        mech = mechanics()
        without = dict(hm.TYPICAL)
        without.pop("gain_db")
        a = hm.log_response(band(), hm.writing_speed(mech),
                            mech["track_width_m"], **without)
        b = hm.log_response(band(), hm.writing_speed(mech),
                            mech["track_width_m"], gain_db=0.0, **without)
        assert np.array_equal(a, b)

    def test_gain_is_flat_and_in_decibels(self):
        mech = mechanics()
        base = dict(hm.TYPICAL)
        raised = dict(hm.TYPICAL, gain_db=2.0)
        step = (hm.log_response(band(), hm.writing_speed(mech),
                                mech["track_width_m"], **raised)
                - hm.log_response(band(), hm.writing_speed(mech),
                                  mech["track_width_m"], **base))
        assert np.allclose(step, 2.0 / hm.DB_PER_NEPER)   # flat...
        assert np.ptp(step) == pytest.approx(0.0, abs=1e-12)  # ...exactly

    def test_spacing_loss_grows_with_frequency(self):
        """Wallace's law: the loss is linear in frequency, so a spacing
        difference is a slope and cannot look like a level."""
        difference = planted(spacing_m=0.30e-6)
        assert difference[0] > difference[-1]


class TestGainSpacingSeparation:
    def test_recovers_a_pure_gain(self):
        fitted = hm.fit_difference_response(band(), planted(gain_db=1.5),
                                            mechanics())
        assert fitted["gain_db"] == pytest.approx(1.5, abs=0.01)
        assert fitted["spacing_m"] == pytest.approx(hm.TYPICAL["spacing_m"],
                                                    abs=5e-9)

    def test_recovers_a_pure_spacing(self):
        fitted = hm.fit_difference_response(band(),
                                            planted(spacing_m=0.25e-6),
                                            mechanics())
        assert fitted["gain_db"] == pytest.approx(0.0, abs=0.01)
        assert fitted["spacing_m"] == pytest.approx(0.25e-6, rel=0.02)

    def test_recovers_both_together(self):
        fitted = hm.fit_difference_response(
            band(), planted(gain_db=1.2, spacing_m=0.10e-6), mechanics())
        assert fitted["gain_db"] == pytest.approx(1.2, abs=0.01)
        assert fitted["spacing_m"] == pytest.approx(0.10e-6, rel=0.02)

    def test_mean_centring_is_a_free_constant(self):
        """THE correction's own premise, kept as a test.

        Projecting out the mean is not a weaker fit that lets spacing
        absorb a level - it is exactly a free constant that is then
        discarded. So naming the gain must leave the spacing untouched,
        and the round's earlier claim that spacing had absorbed it was
        wrong about the mechanism."""
        from scipy.optimize import least_squares

        mech = mechanics()
        speed, width = hm.writing_speed(mech), mech["track_width_m"]
        measured = planted(gain_db=1.2, spacing_m=0.10e-6)

        def centred(values):
            head = dict(hm.TYPICAL, spacing_m=values[0])
            model = hm.head_difference_log(band(), speed, width, head,
                                           dict(hm.TYPICAL))
            return (model - model.mean()) - (measured - measured.mean())

        discarded = least_squares(centred, [hm.TYPICAL["spacing_m"]],
                                  bounds=([0.0], [1e-6]), x_scale=[1e-7])
        named = hm.fit_difference_response(band(), measured, mech)
        assert discarded.x[0] == pytest.approx(named["spacing_m"], abs=1e-12)

    def test_too_few_points_returns_nothing(self):
        assert hm.fit_difference_response(np.array([1e6, 2e6]),
                                          np.array([0.1, 0.2]),
                                          mechanics()) == {}

    def test_explained_uses_the_same_denominator_as_fit(self):
        """A comparison is only a comparison when both sides are computed
        the same way. Against the mean square the constant inflates the
        score, so that figure is reported under its own name."""
        fitted = hm.fit_difference_response(band(), planted(gain_db=1.5),
                                            mechanics())
        assert "explained" in fitted and "explained_with_constant" in fitted
        # a large constant and a tiny residual: the two must not agree
        assert fitted["explained_with_constant"] > fitted["explained"]


class TestCarrierLawBlindness:
    def test_the_video_domain_cannot_see_a_flat_gain(self):
        """Why the gain is fitted on the RF envelope and nowhere else.

        The demodulated magnitude follows half the sum of the two sidebands
        less the carrier, and for a constant that is identically zero. So a
        flat gain difference is not merely hard to measure in video - it is
        absent from it."""
        mech = mechanics()
        speed, width = hm.writing_speed(mech), mech["track_width_m"]
        baseband = np.linspace(0.1e6, 3.0e6, 24)
        carrier = 4.0e6
        raised, flat = dict(hm.TYPICAL, gain_db=1.5), dict(hm.TYPICAL)
        effect = hm.predicted_video_effect(
            baseband,
            lambda f: hm.head_difference_log(f, speed, width, raised, flat),
            carrier, speed, width, raised, flat)
        assert np.allclose(effect[np.isfinite(effect)], 0.0, atol=1e-12)

    def test_spacing_is_annihilated_too(self):
        """Wallace's loss is -2*pi*d*f/v: LINEAR in frequency, and the
        carrier law annihilates a linear tilt as exactly as a constant.

        So the video domain is blind to spacing as well, and a fit that
        leaves `spacing_m` free against a video measurement is fitting a
        null direction - unconstrained, and driven by noise alone."""
        assert self.reach(spacing_m=0.40e-6) < 1e-12

    @pytest.mark.parametrize("name,value", [("gap_m", 0.60e-6),
                                            ("thickness_m", 0.80e-6),
                                            ("azimuth_error_degrees", 0.5)])
    def test_the_curved_mechanisms_do_reach_video(self, name, value):
        """The control: the path is not blind to everything. What survives
        is exactly what is neither constant nor linear in frequency."""
        assert self.reach(**{name: value}) > 1e-3

    def test_the_null_space_is_declared_and_refused(self):
        with pytest.raises(ValueError, match="carrier law"):
            hm.fit_head_difference(np.linspace(0.1e6, 3.0e6, 24),
                                   np.zeros(24), 4.0e6, mechanics(),
                                   free=["spacing_m"])

    @staticmethod
    def reach(**parameters):
        """How far one mechanism's difference reaches demodulated video."""
        mech = mechanics()
        speed, width = hm.writing_speed(mech), mech["track_width_m"]
        baseband = np.linspace(0.1e6, 3.0e6, 24)
        head, flat = dict(hm.TYPICAL, **parameters), dict(hm.TYPICAL)
        effect = hm.predicted_video_effect(
            baseband,
            lambda f: hm.head_difference_log(f, speed, width, head, flat),
            4.0e6, speed, width, head, flat)
        return float(np.nanmax(np.abs(effect)))
