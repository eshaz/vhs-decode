"""The reference NTSC vectorscope, computed from SMPTE 170M's own equations.

The point of the module is that nothing in it is quoted: the graticule
angles follow from the luminance equation of clause 6 and the encoding of
clause 10, so they can be checked rather than trusted.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import vectorscope as vs


def test_the_bar_targets_come_out_at_the_standard_graticule_angles():
    targets = vs.targets(75.0)
    expected = {"Yl": 167.08, "R": 103.46, "Mg": 60.71,
                "B": 347.08, "Cy": 283.46, "G": 240.71}
    for name, angle in expected.items():
        assert targets[name]["phase_deg"] == pytest.approx(angle, abs=0.01)
    # three amplitudes, one per complementary pair
    assert targets["Yl"]["chroma"] == pytest.approx(31.034, abs=0.001)
    assert targets["G"]["chroma"] == pytest.approx(40.963, abs=0.001)
    assert targets["R"]["chroma"] == pytest.approx(43.869, abs=0.001)


def test_the_angles_do_not_depend_on_the_bar_level_but_the_amplitudes_do():
    seventy_five, hundred = vs.targets(75.0), vs.targets(100.0)
    for name in vs.BARS:
        assert seventy_five[name]["phase_deg"] == pytest.approx(
            hundred[name]["phase_deg"], abs=1e-9)
        assert hundred[name]["chroma"] / seventy_five[name]["chroma"] == \
            pytest.approx(4.0 / 3.0, rel=1e-9)


def test_every_complementary_pair_is_exactly_opposite_and_equal():
    targets = vs.targets(75.0)
    for first, second in vs.COMPLEMENTS:
        gap = abs(targets[first]["phase_deg"] - targets[second]["phase_deg"])
        assert gap == pytest.approx(180.0, abs=1e-6)
        assert targets[first]["chroma"] == pytest.approx(
            targets[second]["chroma"], rel=1e-12)


def test_nothing_in_the_signal_sits_on_an_i_or_q_axis():
    """The question Ethan asked. They are the modulation axes, a coordinate
    system rather than a signal: the closest any bar comes is 19.54
    degrees, and the only per-line reference is the burst."""
    targets = vs.targets(75.0)
    for name, target in targets.items():
        near = vs.nearest_axis(target["phase_deg"])
        assert not near["on_axis"]
        assert near["degrees_away"] > 12.0
    assert vs.axes()["I"] == 123.0 and vs.axes()["Q"] == 33.0
    assert min(vs.nearest_axis(t["phase_deg"])["degrees_away"]
               for t in targets.values()) == pytest.approx(12.92, abs=0.01)


def test_the_burst_convention_is_a_clean_factor_of_two():
    amplitude = vs.burst_reference(17.97)
    assert amplitude["specified_ire"] == 20.0
    assert amplitude["gain_percent"] == pytest.approx(89.85, abs=0.05)
    peak = vs.burst_reference(17.97, peak_to_peak=True)
    assert peak["specified_ire"] == 40.0
    assert peak["gain_percent"] == pytest.approx(amplitude["gain_percent"] / 2.0)


def test_circularity_separates_a_common_gain_from_differential_errors():
    targets = vs.targets(75.0)
    # a pure common gain and rotation: no differential error at all
    measured = {n: (t["chroma"] * 0.9, t["phase_deg"] + 4.0)
                for n, t in targets.items()}
    out = vs.circularity(measured)
    assert out["common_gain"] == pytest.approx(0.9, rel=1e-9)
    assert out["common_rotation_deg"] == pytest.approx(4.0, abs=1e-9)
    assert out["differential_gain_percent"] == pytest.approx(0.0, abs=1e-9)
    assert out["differential_phase_deg"] == pytest.approx(0.0, abs=1e-9)
    # now a hue-dependent gain, which is differential gain and nothing else
    measured["R"] = (targets["R"]["chroma"] * 0.99, targets["R"]["phase_deg"] + 4.0)
    out = vs.circularity(measured)
    assert out["differential_gain_percent"] > 5.0
    assert out["differential_phase_deg"] == pytest.approx(0.0, abs=1e-9)


class TestTheSamplingFrame:
    """Ethan, relaying the point: "4fsc SMPTE ST.0244 NTSC is sampled on the
    +I, +Q, -I, -Q phases which are 33 degrees rotated relative to U and V".

    The graticule's I/Q lines and the SAMPLE INSTANTS are the same four
    angles, which is why a phase read straight off 4 fsc samples is in the
    I/Q frame and not the U/V one everything else is quoted in.
    """

    def test_the_sample_instants_are_the_iq_axes(self):
        assert vs.SAMPLE_PHASES_DEG == (33.0, 123.0, 213.0, 303.0)
        graticule = vs.axes()
        assert vs.sample_phase_deg(0) == graticule["Q"]
        assert vs.sample_phase_deg(1) == graticule["I"]
        assert vs.sample_phase_deg(2) == graticule["-Q"]
        assert vs.sample_phase_deg(3) == graticule["-I"]

    def test_the_axes_are_named_in_sampling_order(self):
        assert [vs.sample_axis_name(n) for n in range(4)] == \
            ["+Q", "+I", "-Q", "-I"]

    def test_consecutive_samples_are_ninety_degrees_apart(self):
        """Which is why a MAGNITUDE needs no rotation - quadrature holds
        whatever the phase origin."""
        for n in range(4):
            step = (vs.sample_phase_deg(n + 1) - vs.sample_phase_deg(n)) % 360
            assert step == pytest.approx(90.0)

    def test_the_rotation_is_the_thirty_three_degrees(self):
        assert vs.IQ_ROTATION_DEG == 33.0
        assert vs.to_uv_frame(0.0) == pytest.approx(33.0)

    def test_the_burst_sits_at_147_against_the_sampling_axes(self):
        """AND THAT IS A REAL CROSS-CHECK. The burst is 180 degrees in the
        U/V frame; expressed against the sampling axes it is 147 - and the
        decoder's own measured framing constants are -32.4 and +147.6."""
        assert float(vs.from_uv_frame(180.0)) == pytest.approx(147.0)
        assert float(vs.from_uv_frame(180.0)) - 147.6 == pytest.approx(
            -0.6, abs=0.1)

    def test_the_two_frames_invert_each_other(self):
        for angle in (0.0, 33.0, 123.0, 180.0, 270.0, 359.0):
            assert float(vs.to_uv_frame(vs.from_uv_frame(angle))) == \
                pytest.approx(angle % 360.0)

    def test_the_contract_says_which_quantity_is_affected(self):
        """Getting it half right is the likely error: a magnitude is
        frame-free and an angle is not."""
        note = vs.quadrature_note()
        assert "UNAFFECTED" in note["magnitude"]
        assert "ROTATED" in note["angle"]
        assert note["rotation_deg"] == 33.0

    def test_it_names_where_the_rotation_is_already_applied(self):
        """So the 33 in chroma.py stops being a magic number."""
        assert "chroma.py" in vs.quadrature_note()["already_applied_at"]


class TestTheCoordinateSystemFromTheStreaks:
    """Ethan, 2026-09-06, with a composite vectorscope in front of him: 'I
    believe these lines pointing to I.Q.-I,-Q need to be corrected and
    represent a measureable shape that we can use for correcting the
    color's coordinate system.'

    The six bar TARGETS are not on those lines and that finding stands.
    The transitions between them are, and a direction is measurable.
    """

    @staticmethod
    def _streaks(first_deg, second_deg, count=4000, seed=0, burst=180.0):
        """Transitions drawn along two axes, at the burst-relative angles
        given, with random lengths and random directions along each - which
        is what a streak is: an orientation with no arrow."""
        rng = np.random.default_rng(seed)
        pick = rng.random(count) < 0.5
        angle = np.where(pick, np.radians(burst + first_deg),
                         np.radians(burst + second_deg))
        return rng.normal(0.0, 1.0, count) * np.exp(1j * angle)

    def test_an_orthogonal_pair_is_found_and_called_orthogonal(self):
        out = vs.transition_axes(self._streaks(33.0, 123.0))
        assert out["clustered"]
        # A perfect pair reaches 1 exactly, and the floor at this count is
        # 0.061 - so the margin is the whole of the interval, not a factor
        # that could be written arbitrarily large.
        assert out["clustering"] > 0.95
        assert out["clustering_floor"] < 0.1
        assert out["orthogonal"]
        assert out["versus_burst_deg"] == pytest.approx(33.0, abs=1.0)

    def test_scatter_is_refused_rather_than_reported(self):
        """A stage that cannot measure its subject must decline."""
        rng = np.random.default_rng(3)
        steps = rng.normal(0.0, 1.0, 2000) * np.exp(
            2j * np.pi * rng.random(2000))
        out = vs.transition_axes(steps)
        assert not out["clustered"]
        assert out["clustering"] < out["clustering_floor"]

    def test_a_shear_is_measured_and_not_mistaken_for_a_rotation(self):
        """If the two axes are not ninety apart no rotation corrects it."""
        out = vs.transition_axes(self._streaks(33.0, 103.0))
        assert not out["orthogonal"]
        assert out["separation_deg"] == pytest.approx(70.0, abs=2.0)
        assert out["axes_separately_resolved"]

    def test_the_separation_carries_its_own_resolution(self):
        """The estimator collapses at exactly the answer the specification
        predicts, so the reading has to say what it can resolve."""
        few = vs.transition_axes(self._streaks(33.0, 123.0, count=40))
        many = vs.transition_axes(self._streaks(33.0, 123.0, count=8000))
        assert few["separation_resolution_deg"] > \
            many["separation_resolution_deg"]
        assert few["orthogonal"] and many["orthogonal"]

    def test_the_sign_of_the_rotation_is_reported_both_ways(self):
        """33 and -33 are 66 degrees apart on a quantity defined modulo
        ninety, so a reading that agrees in magnitude and disagrees in sign
        looks like a 24 degree error unless both are given."""
        forward = vs.transition_axes(self._streaks(33.0, 123.0))
        assert forward["rotation_sense"] == "as specified"
        assert forward["departure_as_specified_deg"] == pytest.approx(
            0.0, abs=1.0)
        backward = vs.transition_axes(self._streaks(-33.0, 57.0))
        assert backward["rotation_sense"] == "reversed"
        assert backward["departure_reversed_deg"] == pytest.approx(
            0.0, abs=1.0)

    def test_the_statistic_cannot_tell_a_vector_from_its_reverse(self):
        """Which is what lets it read an envelope referenced to each line's
        own origin, where consecutive lines differ by 180 degrees."""
        steps = self._streaks(33.0, 123.0)
        flipped = steps * np.where(np.arange(steps.size) % 2, -1.0, 1.0)
        first = vs.transition_axes(steps)
        second = vs.transition_axes(flipped)
        assert second["pair_angle_deg"] == pytest.approx(
            first["pair_angle_deg"], abs=1e-9)
        assert second["separation_deg"] == pytest.approx(
            first["separation_deg"], abs=1e-9)

    def test_a_pure_rotation_has_no_shear_and_a_stretch_does(self):
        rng = np.random.default_rng(7)
        isotropic = (rng.normal(0.0, 1.0, 8000)
                     + 1j * rng.normal(0.0, 1.0, 8000))
        turned = isotropic * np.exp(1j * np.radians(17.0))
        assert vs.coordinate_map(turned)["shear_ratio"] < 0.05
        stretched = 2.0 * isotropic.real + 1j * isotropic.imag
        sheared = vs.coordinate_map(stretched)
        assert sheared["stretch"] == pytest.approx(2.0, abs=0.1)
        assert sheared["stretch_axis_deg"] == pytest.approx(0.0, abs=3.0)

    def test_the_isotropy_assumption_is_returned_as_one(self):
        rng = np.random.default_rng(11)
        out = vs.coordinate_map(rng.normal(0.0, 1.0, 500)
                                + 1j * rng.normal(0.0, 1.0, 500))
        assert "isotropic" in out["assumption"]
        assert "bar" in out["assumption"]
