"""What the VHS standard actually fixes - and what it does not.

Ethan: "Let's use the VHS specs to get the needed values for the heads."
The answer is largely negative, and these tests hold that answer so it
cannot quietly drift into an assumption that the head is specified.
"""

import pytest

from vhsdecode.models import vhs_specification as spec


class TestWhatIsPresent:
    def test_the_transport_is_fully_specified(self):
        assert spec.value_of("writing_speed_m_s") == 5.80
        assert spec.value_of("drum_diameter_m") == 0.0620
        assert spec.value_of("tape_speed_sp_m_s") == pytest.approx(33.35e-3)

    def test_the_slower_speeds_are_stated_not_derived(self):
        """The standard prints all three rather than defining LP and EP as
        divisions of SP."""
        assert spec.value_of("tape_speed_lp_m_s") == pytest.approx(16.67e-3)
        assert spec.value_of("tape_speed_ep_m_s") == pytest.approx(11.12e-3)

    def test_the_fm_carriers_and_deviation(self):
        assert spec.value_of("fm_sync_tip_hz") == 3.4e6
        assert spec.value_of("fm_peak_white_hz") == 4.4e6
        assert spec.value_of("fm_deviation_hz") == 1.0e6

    def test_the_interpolated_blanking_carrier_matches_the_decoder(self):
        """The standard states only sync tip and peak white, so blanking is
        interpolated - and it lands exactly on the ire0 and hz_ire the
        decoder already carries, which is a real cross-check."""
        assert spec.carrier_hz_for_ire(0.0) == pytest.approx(3685714.2857,
                                                             abs=1e-3)
        per_ire = spec.carrier_hz_for_ire(1.0) - spec.carrier_hz_for_ire(0.0)
        assert per_ire == pytest.approx(1e6 / 140.0, rel=1e-12)

    def test_the_colour_under_is_forty_times_the_line_rate(self):
        assert spec.value_of("colour_under_hz") == pytest.approx(629.371e3)

    def test_the_burst_doubler_is_six_decibels(self):
        assert spec.value_of("burst_doubler_db") == 6.0


class TestWhatIsAbsent:
    """The finding. A model built on these is built on the JVC guide or on
    a fit, and must say so."""

    def test_the_standard_does_not_contain_the_head(self):
        present = spec.head_values_present()
        for name in ("gap_length_m", "head_to_tape_spacing_m",
                     "coating_thickness_m", "remanence_t", "saturation_t"):
            assert present[name] is False, name

    def test_the_entire_electrical_side_is_absent(self):
        present = spec.head_values_present()
        for name in ("coil_turns", "core_permeability_r",
                     "core_saturation_t"):
            assert present[name] is False, name

    def test_only_the_track_geometry_and_coercivity_survive(self):
        present = spec.head_values_present()
        assert present["track_width_sp_m"] is True
        assert present["azimuth_deg"] is True
        assert present["coercivity_a_m"] is True

    def test_the_drum_rate_is_ours_not_the_standards(self):
        """No rpm, no r/s, no 29.97 anywhere - so half the field rate is a
        derivation of ours."""
        assert spec.value_of("drum_rate_hz") is None

    def test_the_record_current_is_operational_only(self):
        """Which is exactly why record level is treated as an AXIS rather
        than a fixed shape."""
        assert spec.value_of("record_current_a") is None
        assert "optimum" in spec.SMPTE_32M["record_current_a"]["quote"]

    def test_no_playback_response_is_specified_at_all(self):
        """The formal justification for treating the reproduce chain as
        machine-specific."""
        assert spec.value_of("playback_response") is None
        assert "record-side" in spec.SMPTE_32M["playback_response"]["quote"]

    def test_no_head_matching_tolerance_exists(self):
        """So a measured head-pair asymmetry cannot be checked against a
        conformance limit - the standard neither permits nor forbids it."""
        assert spec.value_of("head_matching_tolerance") is None

    def test_the_absences_are_queryable_as_a_set(self):
        absent = spec.absent_from_the_standard()
        assert len(absent) > 10
        assert "gap_length_m" in absent
        assert all(spec.value_of(name) is None for name in absent)


class TestTheTrapsWorthHolding:
    def test_the_azimuth_is_not_an_azimuth_error(self):
        """+-6 degrees opposes the two heads to suppress crosstalk between
        abutting tracks; a head reading its own track sees no azimuth loss,
        so head_model's default of zero error is right."""
        assert spec.value_of("azimuth_deg") == (6.0, -6.0)
        assert "NOT AN AZIMUTH ERROR" in spec.SMPTE_32M["azimuth_deg"]["note"]

    def test_the_video_tracks_abut_with_no_guard_band(self):
        assert spec.value_of("track_pitch_sp_m") == \
            spec.value_of("track_width_sp_m")
        assert spec.value_of("video_guard_band_m") == 0.0

    def test_the_two_coercivities_disagree_by_five_per_cent(self):
        """SMPTE says approximately 50 kA/m; the JVC guide's 600 oersted
        class is 47.7. Both are hedged, so both are carried."""
        smpte = spec.value_of("coercivity_a_m")
        jvc = 600.0 * 1000.0 / (4.0 * 3.141592653589793)
        assert smpte / jvc - 1.0 == pytest.approx(0.047, abs=0.005)

    def test_the_carrier_interleave_is_lp_and_ep_only(self):
        """A real per-head difference, but not on the SP tapes this arc
        measures."""
        note = spec.SMPTE_32M["carrier_interleave_hz"]["note"]
        assert "LP AND EP ONLY" in note

    def test_the_head_switch_position_is_a_window_not_a_point(self):
        """Five to eight lines - so a machine-specific onset is conformant
        rather than anomalous."""
        assert spec.value_of("head_switch_window_h") == (5.0, 8.0)

    def test_every_entry_carries_its_clause_or_says_it_is_absent(self):
        for name, entry in spec.SMPTE_32M.items():
            assert entry["quote"], name
            if entry["value"] is not None:
                assert entry["clause"], name
