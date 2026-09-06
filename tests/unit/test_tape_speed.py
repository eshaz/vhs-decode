"""Tape speed against its expected, as a component.

Ethan: "Let's use the tape speed difference from its expected as another
component, this is what ties in the tapes mechanical model we built
earlier."

The tie-in is the point: speed is invisible on the frequency axis and
directly measurable on the time axis, so the transport supplies what the
magnetics cannot.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import tape_speed as ts
from vhsdecode.models import transport_model


class TestTheExpectedSpeeds:
    def test_the_drum_turns_once_per_two_fields(self):
        """Not a free parameter - the format fixes it and the servo holds
        it, which is why the writing speed is the trustworthy one."""
        got = ts.expected_speeds()
        assert got["drum_rate_hz"] == pytest.approx(30000.0 / 1001.0)
        assert got["drum_rate_hz"] == pytest.approx(
            transport_model.drum_rate_hz(60000.0 / 1001.0))

    def test_the_geometry_and_the_datasheet_disagree_by_one_per_cent(self):
        got = ts.expected_speeds()
        assert got["writing_derived_m_s"] == pytest.approx(5.8709, abs=1e-3)
        assert got["writing_stated_m_s"] == 5.80
        assert got["spec_disagreement"] == pytest.approx(0.0122, abs=2e-3)

    def test_the_stated_figure_is_the_one_used(self):
        assert ts.expected_speeds()["writing_m_s"] == ts.STATED_WRITING_M_S

    def test_the_longitudinal_speed_is_half_a_per_cent_of_the_writing(self):
        """Which is why a capstan error is a tracking problem and barely a
        wavelength one."""
        assert ts.expected_speeds()["longitudinal_share"] == pytest.approx(
            0.0057, abs=5e-4)

    def test_the_slower_speeds_are_the_spec_divisions(self):
        assert ts.NOMINAL_LONGITUDINAL_M_S["LP"] == pytest.approx(
            ts.NOMINAL_LONGITUDINAL_M_S["SP"] / 2, rel=1e-3)
        assert ts.NOMINAL_LONGITUDINAL_M_S["EP"] == pytest.approx(
            ts.NOMINAL_LONGITUDINAL_M_S["SP"] / 3, rel=1e-3)


class TestTheTimeWitness:
    def test_a_fast_tape_shortens_the_line_period(self):
        expected = 1.0 / 15734.265734
        for planted in (-0.01, 0.0, 0.004):
            measured = np.full(200, expected / (1.0 + planted))
            got = ts.from_line_period(measured, expected)
            assert got["epsilon"] == pytest.approx(planted, abs=1e-9)

    def test_it_is_robust_to_a_few_bad_lines(self):
        expected = 1.0 / 15734.265734
        measured = np.full(300, expected / 1.002)
        measured[:20] = expected * 4.0             # dropouts, missed pulses
        assert ts.from_line_period(measured, expected)["epsilon"] == \
            pytest.approx(0.002, abs=1e-6)

    def test_it_reports_what_it_actually_measures(self):
        """Playback against RECORD, not against spec - a source whose sync
        generator ran fast reads the same as a slow deck."""
        got = ts.from_line_period(np.full(10, 1.0 / 15734.0), 1.0 / 15734.0)
        assert "RECORD" in got["measures"]
        assert got["read_before_the_tbc"] is True

    def test_no_usable_lines_reports_nothing(self):
        got = ts.from_line_period(np.array([0.0, -1.0, np.nan]), 6.35e-5)
        assert not np.isfinite(got["epsilon"])


class TestTheDrumWitness:
    def test_the_drum_line_reads_the_writing_speed(self):
        expected = transport_model.drum_rate_hz(60000.0 / 1001.0)
        got = ts.from_drum_line(expected * 1.003)
        assert got["epsilon"] == pytest.approx(0.003, rel=1e-6)

    def test_it_lives_on_the_amplitude_axis(self):
        """An envelope line, so the time base cannot null it - unlike the
        line period, which the corrector zeroes by construction."""
        assert "amplitude" in ts.from_drum_line(29.97)["axis"]


class TestTheDegeneracy:
    def test_speed_is_exactly_a_common_scaling_of_every_length(self):
        """AN IDENTITY, NOT AN APPROXIMATION. Every loss depends on a
        length over the speed, so scaling one and scaling the others are
        the same operation written twice."""
        got = ts.degeneracy_with_lengths(np.linspace(0.5e6, 6e6, 400))
        assert got["against_common_scaling"] == pytest.approx(1.0, abs=1e-9)
        assert got["worst_difference_nepers"] < 1e-12

    def test_it_is_not_separable_from_spacing_alone_either(self):
        """0.999 in magnitude - the sign is negative because more speed is
        less effective spacing."""
        got = ts.degeneracy_with_lengths(np.linspace(0.5e6, 6e6, 400))
        assert abs(got["against_spacing_alone"]) > 0.99

    def test_the_verdict_is_frequency_blind_and_time_sighted(self):
        got = ts.degeneracy_with_lengths(np.linspace(0.5e6, 6e6, 200))
        assert got["separable_on_frequency"] is False
        assert got["separable_on_time"] is True

    def test_a_speed_error_rescales_wavelength_one_for_one(self):
        assert ts.wavelength_scaling(0.02) == pytest.approx(1.02)

    def test_the_departure_grows_with_frequency(self):
        """Because the losses do: at long wavelengths every ratio is small
        and a rescaling of them changes little."""
        departure = ts.loss_departure(np.array([0.5e6, 5.0e6]), 0.02)
        assert abs(departure[1]) > abs(departure[0])

    def test_no_speed_error_is_no_departure(self):
        assert np.allclose(ts.loss_departure(np.linspace(1e6, 5e6, 50), 0.0),
                           0.0, atol=1e-15)


class TestTheMechanicalTieIn:
    def test_only_the_capstan_and_pinch_roller_act_on_speed(self):
        """The tie-in: transport_model already declares which elements can
        produce this, so the candidate list is not invented here."""
        got = ts.attribute(0.001)
        assert "capstan" in got["candidates"]
        assert "pinch roller" in got["candidates"]
        assert "supply reel" not in got["candidates"]

    def test_the_drum_is_the_only_certain_geometry(self):
        assert ts.attribute(0.0)["drum_is_certain"] is True

    def test_the_two_witnesses_separate_the_causes(self):
        assert "capstan" in ts.attribute(0.002)["cause"]
        assert "drum" in ts.attribute(0.0, drum_epsilon=0.002)["cause"]
        assert "both" in ts.attribute(0.002, drum_epsilon=0.002)["cause"]
        assert ts.attribute(0.0, drum_epsilon=0.0)["cause"] == "no departure"


class TestTheComponent:
    def test_it_declares_both_axes(self):
        assert set(ts.component()["axes"]) == {"time", "frequency"}

    def test_its_criterion_names_the_time_axis_as_the_identifier(self):
        criterion = ts.component()["criterion"]
        assert "time axis is the one that identifies it" in criterion
        assert "degenerate" in criterion

    def test_it_declares_its_own_null_space(self):
        """The time base corrector nulls it by construction - the same
        trap the burst lock has."""
        assert "time base corrector" in ts.component()["null_space"]


class TestTheFieldOnTheTape:
    """Ethan: "if we can derive amplitude of the luma we also know the time
    base, which is expected to be the line rate frequency, and the total
    overall size of the field on the VHS tape itself"."""

    def test_a_field_is_about_98_millimetres_of_track(self):
        got = ts.field_geometry()
        assert got["track_length_m"] * 1e3 == pytest.approx(97.95, abs=0.01)

    def test_a_line_is_a_third_of_a_millimetre(self):
        got = ts.field_geometry()
        assert got["track_per_line_m"] * 1e3 == pytest.approx(0.373, abs=1e-3)

    def test_the_half_revolution_closure_is_algebraic_not_evidence(self):
        """It must agree, because the derived speed IS pi D f + v_tape and
        a field IS half a drum period. Asserted as an identity so nobody
        later reads it as a confirmation of the geometry."""
        got = ts.field_geometry()
        assert got["half_revolution_m"] == pytest.approx(
            got["track_length_m"], rel=1e-12)

    def test_the_track_angle_is_not_the_mechanical_wrap(self):
        """The head sweeps exactly 180 degrees of drum; the track is
        longer than 180 degrees of circumference because the tape moved
        under it. Reading the first number as a wrap angle is the error
        this test exists to prevent."""
        got = ts.field_geometry()
        assert got["mechanical_sweep_degrees"] == 180.0
        assert got["track_in_circumference_degrees"] == pytest.approx(
            181.03, abs=0.05)

    def test_the_stated_speed_is_the_head_only_figure_rounded(self):
        """Which resolves the 1.2 per cent disagreement: 5.80 is the
        head's own 5.8375 to two significant figures, not the relative
        5.8709 that actually sets wavelength."""
        got = ts.field_geometry()
        assert got["stated_is_head_only_rounded"] is True
        assert got["head_only_m_s"] == pytest.approx(5.8375, abs=1e-3)
        assert got["relative_speed_m_s"] == pytest.approx(5.8709, abs=1e-3)
        assert abs(got["head_only_m_s"] / 5.80 - 1) < 0.01
        assert abs(got["relative_speed_m_s"] / 5.80 - 1) > 0.01

    def test_the_line_rate_falls_out_of_the_field_rate(self):
        got = ts.field_geometry()
        assert got["line_rate_hz"] == pytest.approx(15734.2657, abs=1e-3)


class TestAmplitudeToTimeBase:
    def test_the_field_rate_gives_the_line_rate_and_the_size(self):
        got = ts.time_base_from_amplitude(60000.0 / 1001.0)
        assert got["line_rate_hz"] == pytest.approx(15734.2657, abs=1e-3)
        assert got["epsilon"] == pytest.approx(0.0, abs=1e-12)
        assert got["track_length_m"] * 1e3 == pytest.approx(97.95, abs=0.01)

    def test_a_fast_transport_reads_as_a_positive_epsilon(self):
        got = ts.time_base_from_amplitude(60000.0 / 1001.0 * 1.002)
        assert got["epsilon"] == pytest.approx(0.002, rel=1e-6)

    def test_it_names_two_amplitude_witnesses(self):
        got = ts.time_base_from_amplitude(59.94)
        assert len(got["witnesses"]) == 2
        assert any("head switch" in w for w in got["witnesses"])
        assert any("drum-rate line" in w for w in got["witnesses"])

    def test_it_survives_the_time_base_corrector(self):
        """The reason it is worth having when sync already gives the line
        rate: the corrector locks sync to sync and nulls a speed error by
        construction, while an envelope modulation is on the amplitude
        axis and survives."""
        assert ts.time_base_from_amplitude(59.94)["survives_the_tbc"] is True

    def test_it_sees_only_the_playback_transport(self):
        assert "PLAYBACK" in ts.time_base_from_amplitude(59.94)["measures"]


class TestTheReels:
    """Ethan: "Reel speed can be derived using the VHS specs against the
    different types of possible VHS tapes."

    It can, and per tape type it is sharper than a fixed radius range.
    """

    def test_the_pack_radius_is_the_archimedean_spiral(self):
        """Area pi(r^2 - r0^2) = L t, nothing fitted."""
        radius = ts.reel_pack_radius_m(266.0, 19e-6)
        assert radius == pytest.approx(
            np.sqrt(ts.HUB_RADIUS_M ** 2 + 266.0 * 19e-6 / np.pi))
        assert radius * 1e3 == pytest.approx(41.87, abs=0.01)

    def test_an_empty_reel_is_the_hub(self):
        assert ts.reel_pack_radius_m(0.0) == pytest.approx(ts.HUB_RADIUS_M)

    def test_the_length_law_is_the_specifications(self):
        """JVC s.1.1.25-1: L = 2.2t + 2 metres for NTSC."""
        assert ts.tape_length_m(120) == pytest.approx(266.0)
        assert ts.tape_length_m(30) == pytest.approx(68.0)
        assert ts.tape_length_m(120, "PAL") == pytest.approx(172.4)

    @pytest.mark.parametrize("tape,slowest", [
        ("T-30", 0.2253), ("T-60", 0.1718), ("T-90", 0.1443),
        ("T-120", 0.1268), ("T-160", 0.1217),
    ])
    def test_each_tape_has_its_own_lower_edge(self, tape, slowest):
        got = ts.reel_band_hz(tape)
        assert got["slowest_hz"] == pytest.approx(slowest, abs=1e-3)

    def test_the_upper_edge_is_common_because_it_is_the_bare_hub(self):
        edges = {ts.reel_band_hz(t)["fastest_hz"]
                 for t in ("T-30", "T-90", "T-160")}
        assert len(edges) == 1
        assert edges.pop() == pytest.approx(0.4423, abs=1e-3)

    def test_the_long_tape_is_thinner(self):
        """Which is how T-160 fits 354 m where T-120 fits 266."""
        assert ts.TAPE_THICKNESS_M["T-160"] < ts.TAPE_THICKNESS_M["T-120"]
        assert ts.reel_band_hz("T-160")["length_m"] > \
            ts.reel_band_hz("T-120")["length_m"]

    def test_the_fixed_range_admits_a_region_no_cassette_reaches(self):
        """The tree's 12-45 mm gives 0.1180 Hz, below every real tape's
        lower edge."""
        fixed = 33.35e-3 / (2 * np.pi * 0.045)
        assert fixed < min(ts.reel_band_hz(t)["slowest_hz"]
                           for t in ts.TAPE_THICKNESS_M)

    def test_the_reel_rate_is_a_position_measurement(self):
        """The reason it is worth deriving rather than bounding: the rate
        gives the radius, the radius gives the length wound."""
        got = ts.position_from_reel_rate(0.293)
        assert got["radius_m"] * 1e3 == pytest.approx(18.12, abs=0.05)
        assert got["length_wound_m"] == pytest.approx(30.5, abs=0.5)
        assert got["within_band"]

    def test_position_inverts_the_pack_radius(self):
        for wound in (10.0, 100.0, 250.0):
            radius = ts.reel_pack_radius_m(wound, 19e-6)
            back = ts.position_from_reel_rate(ts.reel_rate_hz(radius))
            assert back["length_wound_m"] == pytest.approx(wound, rel=1e-6)


class TestTrackingMismatch:
    def test_there_is_no_guard_band_so_an_offset_costs_twice(self):
        got = ts.tracking_mismatch(10e-6)
        assert got["no_guard_band"] is True
        assert got["loss_db"] > 0.0
        assert np.all(np.isfinite(got["crosstalk_db"]))

    def test_perfect_tracking_loses_nothing(self):
        got = ts.tracking_mismatch(0.0)
        assert got["overlap_fraction"] == pytest.approx(1.0)
        assert got["loss_db"] == pytest.approx(0.0, abs=1e-9)

    def test_the_azimuth_protects_the_luma_and_not_the_chroma(self):
        """THE TWO-BAND CONSEQUENCE, from the specification's own +-6
        degrees and 58 um: 34.5 dB at the luma sync tip and 13.8 at the
        colour-under - twenty decibels less for the chroma."""
        got = ts.tracking_mismatch(5e-6,
                                   frequency_hz=(0.629e6, 3.4e6))
        chroma, luma = got["azimuth_suppression_db"]
        assert chroma == pytest.approx(13.8, abs=0.3)
        assert luma == pytest.approx(34.5, abs=0.5)
        assert luma - chroma > 18.0

    def test_the_gap_sweeps_twice_the_azimuth_across_the_track(self):
        """The two heads differ by 2 alpha, not alpha."""
        got = ts.tracking_mismatch(0.0)
        assert got["across_track_m"] * 1e6 == pytest.approx(12.33, abs=0.05)

    def test_a_larger_offset_leaks_more_of_the_neighbour(self):
        small = ts.tracking_mismatch(2e-6, frequency_hz=(0.629e6,))
        large = ts.tracking_mismatch(20e-6, frequency_hz=(0.629e6,))
        assert large["crosstalk_db"][0] > small["crosstalk_db"][0]
        assert large["loss_db"] > small["loss_db"]
