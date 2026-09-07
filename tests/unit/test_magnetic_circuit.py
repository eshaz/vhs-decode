"""The magnetic circuit: turns and current in, flux and voltage out.

Ethan specified the SI set and the relations; these hold the code to them,
and to the physics that decides which way round the two directions run.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import magnetic_circuit as mc


class TestTheUnitSet:
    def test_every_quantity_ethan_listed_is_carried(self):
        names = {q[0] for q in mc.QUANTITIES}
        assert names == {"field force", "field flux", "field intensity",
                         "flux density", "reluctance", "permeability"}

    def test_the_units_are_the_ones_he_gave(self):
        units = {q[0]: q[2] for q in mc.QUANTITIES}
        assert units["field force"] == "ampere-turn"
        assert units["field flux"] == "weber"
        assert units["field intensity"] == "ampere-turn per metre"
        assert units["flux density"] == "tesla"
        assert units["reluctance"] == "ampere-turn per weber"
        assert units["permeability"] == "tesla-metre per ampere-turn"

    def test_permeability_in_his_units_is_the_henry_per_metre(self):
        """T*m/(A*turn) and H/m are the same unit, so a datasheet figure
        needs no conversion."""
        # B = mu H: tesla = [mu] * ampere-turn/metre, so [mu] = T*m/(A*turn)
        h = 1000.0
        b = mc.flux_density(h, mc.VACUUM_PERMEABILITY)
        assert b / h == pytest.approx(mc.VACUUM_PERMEABILITY)


class TestTheCircuit:
    def test_the_field_force_is_turns_times_current(self):
        assert mc.magnetomotive_force(120, 0.05) == pytest.approx(6.0)

    def test_ethans_ratio_is_turns_over_coil_length(self):
        """"Head current to magnetic ratio is the number of turns divided
        by the length of the coil" - so H/I = N/l exactly."""
        turns, length, current = 150.0, 3e-3, 0.02
        ratio = mc.turns_per_metre(turns, length)
        assert ratio == pytest.approx(turns / length)
        assert mc.field_intensity(turns, current, length) == pytest.approx(
            ratio * current)

    def test_ohms_law_for_the_magnetic_circuit_closes(self):
        """Phi = F / R, so Phi * R must return F."""
        force = mc.magnetomotive_force(200, 0.01)
        r = mc.reluctance(0.3e-6, 58e-6 * 20e-6)
        phi = mc.flux(force, r)
        assert float(phi) * r == pytest.approx(float(force), rel=1e-12)

    def test_reluctance_falls_with_permeability_and_area(self):
        base = mc.reluctance(1e-6, 1e-9)
        assert mc.reluctance(1e-6, 2e-9) == pytest.approx(base / 2)
        assert mc.reluctance(2e-6, 1e-9) == pytest.approx(base * 2)
        assert mc.reluctance(1e-6, 1e-9, 1000 * mc.VACUUM_PERMEABILITY) \
            == pytest.approx(base / 1000)

    def test_whether_the_gap_dominates_turns_on_the_core_permeability(self):
        """NOT AUTOMATIC, and the textbook version is wrong here.

        Reluctance is l/(mu A), so gap and core compete on l/mu and the
        gap wins only where mu_r(core) > l(core)/g. Against the stated
        0.30 um gap a 2 mm core needs mu_r above 6667; at a typical
        ferrite 3000 the core carries twice the gap's reluctance. Which
        element sets the circuit therefore depends on the one property
        `reference_properties` marks missing.
        """
        area = 58e-6 * 20e-6
        gap = mc.reluctance(0.30e-6, area)
        modest = mc.reluctance(2e-3, area, 3000 * mc.VACUUM_PERMEABILITY)
        high = mc.reluctance(2e-3, area, 10000 * mc.VACUUM_PERMEABILITY)
        assert modest > gap                       # the core dominates
        assert high < gap                         # now the gap does
        crossover = 2e-3 / 0.30e-6
        assert crossover == pytest.approx(6667, rel=0.01)

    def test_faraday_differentiates_so_the_head_rises_with_frequency(self):
        times = np.linspace(0.0, 1e-3, 20001)
        for frequency in (1e3, 2e3):
            phi = 1e-9 * np.sin(2 * np.pi * frequency * times)
            volts = mc.induced_voltage(100, phi, times)
            peak = float(np.max(np.abs(volts[100:-100])))
            expected = 100 * 2 * np.pi * frequency * 1e-9
            assert peak == pytest.approx(expected, rel=0.02)


class TestTheBHCurve:
    def test_it_saturates_at_the_saturation_flux_density(self):
        b = mc.anhysteretic_b(np.array([1e9]), 0.35, 47.7e3)
        assert float(b[0]) == pytest.approx(0.35, rel=1e-9)

    def test_its_slope_at_the_origin_is_bs_over_hc(self):
        saturation, coercivity = 0.35, 47.7e3
        slope = mc.differential_permeability(0.0, saturation, coercivity)
        assert float(slope) == pytest.approx(saturation / coercivity)

    def test_the_differential_permeability_is_the_curves_derivative(self):
        saturation, coercivity = 0.35, 47.7e3
        h = np.linspace(-3 * coercivity, 3 * coercivity, 20001)
        numeric = np.gradient(mc.anhysteretic_b(h, saturation, coercivity), h)
        exact = mc.differential_permeability(h, saturation, coercivity)
        assert np.max(np.abs(numeric - exact)) < 1e-9

    def test_permeability_collapses_in_saturation(self):
        """A head driven hard has less incremental response - which is why
        the record level is an axis and not a scale factor."""
        saturation, coercivity = 0.35, 47.7e3
        quiet = mc.differential_permeability(0.0, saturation, coercivity)
        driven = mc.differential_permeability(3 * coercivity, saturation,
                                              coercivity)
        assert float(driven) < 0.02 * float(quiet)

    def test_the_hysteresis_branches_are_displaced_by_the_coercivity(self):
        saturation, coercivity = 0.35, 47.7e3
        up = mc.hysteresis_branch(0.0, saturation, coercivity, ascending=True)
        down = mc.hysteresis_branch(0.0, saturation, coercivity,
                                    ascending=False)
        assert float(up) < 0.0 < float(down)      # the loop is open at H = 0
        assert float(down) == pytest.approx(-float(up))


class TestRotationAndResponse:
    def test_the_writing_speed_comes_from_the_rotation(self):
        got = mc.writing_speed_m_s()
        assert got["circumference_m"] == pytest.approx(math.pi * 0.0620)
        assert got["head_speed_m_s"] == pytest.approx(5.8375, abs=1e-3)

    def test_the_geometry_and_the_datasheet_disagree_by_one_per_cent(self):
        """Kept visible rather than resolved: a 1.2 per cent error in v is
        1.2 per cent in every wavelength and so in every loss exponent."""
        got = mc.writing_speed_m_s()
        assert got["departure_from_stated"] == pytest.approx(0.0122, abs=2e-3)

    def test_the_wavelength_is_speed_over_frequency(self):
        assert float(mc.wavelength_m(3.9e6, 5.8)) == pytest.approx(
            5.8 / 3.9e6)

    def test_the_bare_head_rises_six_decibels_an_octave(self):
        """Faraday, before any loss - the term the rotation sets."""
        response = mc.rotation_response(np.array([1e6, 2e6]), 100, 0.0, 5.8,
                                        gap_m=1e-12)
        ratio = response["faraday"][1] / response["faraday"][0]
        assert 20 * np.log10(ratio) == pytest.approx(6.02, abs=0.01)

    def test_the_gap_null_is_the_speed_over_the_gap(self):
        response = mc.rotation_response(np.array([1e6]), 100, 0.0, 5.8,
                                        gap_m=0.30e-6)
        assert response["gap_null_hz"] == pytest.approx(5.8 / 0.30e-6)
        # and it is far outside the recorded band, which is why the gap
        # cannot be separated from azimuth by its own null
        assert response["gap_null_hz"] > 15e6

    def test_spacing_loss_costs_more_at_short_wavelengths(self):
        response = mc.rotation_response(np.array([1e6, 5e6]), 100, 0.05e-6,
                                        5.8)
        assert response["spacing_loss"][1] < response["spacing_loss"][0]


class TestSpacingAndDropouts:
    def test_the_wallace_constant_is_derived(self):
        """54.6 dB per wavelength of spacing = 20 log10(e) * 2 pi."""
        got = float(mc.spacing_loss_db(1.0, 1.0))
        assert got == pytest.approx(20 * np.log10(np.e) * 2 * np.pi)
        assert got == pytest.approx(54.575, abs=1e-3)

    def test_a_smoke_particle_costs_five_decibels(self):
        """The sharpest lever in the chain: at the luma carrier's 1.5 um
        wavelength, a tenth of that is 0.15 um."""
        assert float(mc.spacing_loss_db(0.15e-6, 1.5e-6)) == pytest.approx(
            5.46, abs=0.01)

    def test_dropouts_are_derived_from_the_distance_over_time(self):
        """Ethan's construction: not a category of event, but the spacing
        loss exceeding what the demodulator carries."""
        times = np.linspace(0.0, 1e-3, 5001)
        spacing = mc.spacing_over_time(
            times, 0.05e-6,
            [{"centre_s": 3e-4, "width_s": 1e-5, "height_m": 0.6e-6},
             {"centre_s": 7e-4, "width_s": 5e-6, "height_m": 0.5e-6}])
        got = mc.dropouts_from_spacing(times, spacing, 1.5e-6,
                                       threshold_db=6.0)
        assert got["count"] == 2
        assert 0.0 < got["fraction"] < 0.2
        assert got["worst_db"] > 6.0

    def test_a_steady_spacing_makes_no_dropout(self):
        times = np.linspace(0.0, 1e-3, 1001)
        spacing = mc.spacing_over_time(times, 0.05e-6)
        got = mc.dropouts_from_spacing(times, spacing, 1.5e-6)
        assert got["count"] == 0
        assert got["fraction"] == 0.0


class TestAir:
    def test_air_is_vacuum_to_four_parts_in_ten_million(self):
        assert mc.air_permeability() / mc.VACUUM_PERMEABILITY - 1.0 \
            == pytest.approx(mc.AIR_SUSCEPTIBILITY, rel=1e-6)

    def test_air_is_paramagnetic_not_diamagnetic(self):
        """The one correction to Ethan's statement, and it changes nothing:
        the magnitude he gave (1e-7) is right, the sign is the other way
        because molecular oxygen is paramagnetic and is a fifth of air."""
        assert mc.AIR_SUSCEPTIBILITY > 0
        assert 1e-7 < mc.AIR_SUSCEPTIBILITY < 1e-6

    def test_mu_zero_is_the_measured_value_not_four_pi(self):
        """Since the 2019 SI redefinition mu_0 is measured, and 4 pi e-7 is
        an approximation good to 5.4e-10."""
        assert mc.VACUUM_PERMEABILITY == pytest.approx(1.25663706212e-6)
        assert mc.VACUUM_PERMEABILITY != 4 * math.pi * 1e-7


class TestTheReferenceList:
    def test_completeness_is_the_conjunction_of_three_conditions(self):
        for name, entry in mc.reference_properties().items():
            assert entry["complete"] == (entry["specified"]
                                         and entry["witnessed"]
                                         and entry["separable"]), name

    def test_every_entry_names_its_si_quantity_and_unit(self):
        # the six magnetic units, plus the geometric and kinematic ones a
        # magnetic property can legitimately be expressed in
        units = {q[2] for q in mc.QUANTITIES} | {"degree", "metre",
                                                 "turn (dimensionless)",
                                                 "metre per second"}
        for name, entry in mc.reference_properties().items():
            assert entry["unit"] in units, (name, entry["unit"])

    def test_the_separable_ones_are_the_short_list(self):
        """Most magnetic properties cannot form a reference, and the reason
        is separability rather than ignorance: the six mechanisms collapse
        to 1.58 distinguishable of 6."""
        table = mc.reference_properties()
        complete = {k for k, v in table.items() if v["complete"]}
        assert complete == {"coercivity", "gap reluctance",
                            "air permeability", "writing speed",
                            "drum diameter"}

    def test_the_electrical_side_is_declared_missing_not_guessed(self):
        """N, l and mu_core have no datasheet in this tree, and a guessed
        reference would be comparing a guess to a guess."""
        table = mc.reference_properties()
        for name in ("turns", "coil length", "core permeability"):
            assert table[name]["expected"] is None
            assert not table[name]["specified"]

    def test_gap_and_azimuth_are_both_marked_inseparable(self):
        table = mc.reference_properties()
        assert not table["gap length"]["separable"]
        assert not table["azimuth error"]["separable"]


class TestTheRecordSide:
    def test_the_record_head_s_amplitude_is_observable(self):
        assert "observable" in mc.record_side_observability()["amplitude"]

    def test_the_phase_claim_now_turns_on_whether_a_record_tap_exists(self):
        """THIS TEST USED TO ASSERT THE OPPOSITE, and the premise changed
        rather than the physics. Ethan's reason was that "we don't have the
        original signal being recorded to tape"; this project's test-pattern
        captures are MATCHED taps, so it does. Measured through
        `tap_transfer` on 75bars SP, the record tap is coherent with itself
        to 0.99799 and the transfer's phase is determined across the luma
        band to 0.209 and 0.184 degrees rms for the two heads. What is still
        unavailable is the SPLIT between the record head's share and the
        playback head's, because on one deck they are the same head."""
        got = mc.record_side_observability()
        assert "MATCHED RECORD TAP" in got["phase"]
        assert "NOT separable" in got["phase"]
        assert "0.99799" in got["reference_when_tapped"]

    def test_the_split_still_needs_one_of_the_named_experiments(self):
        """The correction above must not be read as saying the record head
        is now measurable on its own. `record_head_derivable` names the
        three experiments that would part the two, and all three are still
        required."""
        got = mc.record_head_derivable()
        assert len(got["separators"]) == 3
        assert "PRODUCT" in got["recovered_as"]

    def test_it_names_the_head_switch_as_the_witness(self):
        assert "head-switch" in mc.record_side_observability()["witness"]


class TestCoercivityInTheModel:
    """Ethan: "we can use the magnetic coercivity from the tape to further
    define our model, if we aren't already". We were not - `magnetic.py`'s
    self-demagnetisation cap is purely geometric and knows nothing about
    the material.
    """

    def test_remanence_converts_to_a_magnetisation_through_mu_zero(self):
        """The datasheet quotes tesla and the transition length wants A/m;
        they differ by mu_0, which is five orders of magnitude."""
        assert mc.remanent_magnetisation(0.15) == pytest.approx(
            0.15 / mc.VACUUM_PERMEABILITY)
        assert mc.remanent_magnetisation(0.15) == pytest.approx(119366,
                                                                rel=1e-3)

    def test_the_transition_length_is_three_times_the_spacing(self):
        """Which is why leaving it out mattered: it is the dominant
        short-wavelength loss, not a correction to one."""
        a = mc.transition_length()
        assert a == pytest.approx(0.1545e-6, rel=0.01)
        assert a > 3 * 0.05e-6

    def test_it_costs_more_at_the_carrier_than_the_spacing_does(self):
        carrier = np.array([3.9e6])
        transition = float(mc.transition_loss_db(carrier)[0])
        spacing = float(mc.spacing_loss_db(
            0.05e-6, mc.wavelength_m(carrier, 5.8709))[0])
        assert transition == pytest.approx(5.60, abs=0.05)
        assert transition > 3 * spacing

    def test_a_harder_tape_holds_a_shorter_transition(self):
        """The physics: the recorded magnetisation's own demagnetising
        field erases the transition and the coercivity resists it, so
        a goes as one over the square root of Hc."""
        soft = mc.transition_length(coercivity_a_m=47746 / 4.0)
        hard = mc.transition_length(coercivity_a_m=47746)
        assert hard == pytest.approx(soft / 2.0, rel=1e-9)

    def test_the_material_enters_only_as_the_ratio(self):
        """Mr/Hc is the lever, which is what makes the O(1) constant in
        the formula arguable without the model changing."""
        base = mc.transition_length()
        doubled = mc.transition_length(coercivity_a_m=47746 * 2,
                                       remanence_t=0.30)
        assert doubled == pytest.approx(base, rel=1e-9)

    def test_the_level_is_what_separates_it_from_spacing(self):
        """Same exp(-2 pi a / lambda) form as spacing, so degenerate on the
        frequency axis - but a goes as sqrt(Mr) and Mr is what the record
        current laid down, while the head-to-tape spacing does not care."""
        quiet = mc.transition_length(level=0.25)
        full = mc.transition_length(level=1.0)
        assert full == pytest.approx(2.0 * quiet, rel=1e-9)
        assert "record level" in mc.coercivity_constrains()["separated_by"]

    def test_the_coercivity_is_a_loss_and_not_a_bandwidth_limit(self):
        """The correction I had to make: the material allows 0.97 um while
        the coating's geometry caps at 1.26 um, so the geometry binds. A
        harder tape records the same frequencies more strongly, not higher
        frequencies."""
        got = mc.coercivity_constrains()
        assert got["material_limit_binds_first"] is False
        assert got["shortest_wavelength_m"] < got["geometric_limit_m"]
        assert got["loss_at_carrier_db"] > 5.0

    def test_the_record_field_must_reach_the_coercivity(self):
        """The first of the three things coercivity fixes, and with N/l
        known it gives the record current itself."""
        got = mc.coercivity_constrains()
        needed = got["record_field_needed_a_m"]
        ratio = mc.turns_per_metre(100, 2e-3)
        assert mc.field_intensity(100, needed / ratio, 2e-3) == pytest.approx(
            needed)

    def test_coercivity_is_the_one_complete_magnetic_reference(self):
        """It was already marked complete, and now it earns it by being
        used: specified by the datasheet, witnessed by the record-level
        axis, and separable because it acts at record."""
        table = mc.reference_properties()
        assert table["coercivity"]["complete"] is True


class TestTheLoopArea:
    """Ethan: "Add up the area of the BH curve to form how lossy the
    magnetic BH curve is, which itself may be derived or explained by a
    constant by transforming that over the specification's area."

    Both halves hold, and the constant is exact.
    """

    @staticmethod
    def _material():
        coercivity = 600.0 * 1000.0 / (4.0 * math.pi)
        return 0.15 / math.tanh(1.0), coercivity

    def test_the_area_matches_numerical_integration(self):
        """The closed form is 4 Bs Hc, from the bracket
        [ln cosh(u+1) - ln cosh(u-1)] tending to +2 and -2."""
        saturation, coercivity = self._material()
        grid = np.linspace(-30 * coercivity, 30 * coercivity, 200001)
        numeric = float(np.trapezoid(
            mc.hysteresis_branch(grid, saturation, coercivity, False)
            - mc.hysteresis_branch(grid, saturation, coercivity, True), grid))
        got = mc.hysteresis_loss(saturation, coercivity)
        assert got["major_loop_area"] == pytest.approx(numeric, rel=1e-9)
        assert got["major_loop_area"] == pytest.approx(
            4.0 * saturation * coercivity, rel=1e-12)

    def test_the_loss_constant_is_one_over_tanh_one(self):
        """THE CONSTANT ETHAN PREDICTED. The loop area over the
        specification's own rectangle 4 Hc Br is Bs/Br - the reciprocal
        squareness - and for this loop shape that is exactly 1/tanh(1).
        A pure number, set by the shape and not by the tape's figures."""
        saturation, coercivity = self._material()
        got = mc.hysteresis_loss(saturation, coercivity)
        assert got["loss_constant"] == pytest.approx(1.0 / math.tanh(1.0),
                                                     rel=1e-12)
        assert got["loss_constant"] == pytest.approx(1.313035, abs=1e-6)

    def test_the_constant_does_not_depend_on_the_tapes_figures(self):
        """Which is what makes it a constant rather than a measurement."""
        first = mc.hysteresis_loss(0.45, 47746.0)["loss_constant"]
        second = mc.hysteresis_loss(1.20, 200000.0)["loss_constant"]
        assert first == pytest.approx(second, rel=1e-12)

    def test_the_squareness_is_the_constants_reciprocal(self):
        saturation, coercivity = self._material()
        got = mc.hysteresis_loss(saturation, coercivity)
        assert got["squareness"] == pytest.approx(1.0 / got["loss_constant"],
                                                  rel=1e-12)
        assert got["squareness"] == pytest.approx(math.tanh(1.0), rel=1e-12)

    def test_the_area_grows_with_drive_and_then_saturates(self):
        """Past about three times the coercivity the branches have
        converged and driving harder adds nothing - that plateau is the
        major loop, and an FM record head sits well inside it."""
        saturation, coercivity = self._material()
        areas = [mc.hysteresis_loss(saturation, coercivity,
                                    k * coercivity)["area_j_per_m3_per_cycle"]
                 for k in (0.5, 1.0, 2.0, 3.0, 6.0)]
        assert areas == sorted(areas)                    # monotone
        assert areas[-1] == pytest.approx(areas[-2], rel=0.01)   # plateau
        major = mc.hysteresis_loss(saturation, coercivity)["major_loop_area"]
        assert areas[-1] == pytest.approx(major, rel=1e-3)

    def test_a_small_drive_encloses_less_than_the_specification(self):
        saturation, coercivity = self._material()
        got = mc.hysteresis_loss(saturation, coercivity, 0.5 * coercivity)
        assert got["loss_constant"] < 0.6
        assert got["at_the_major_loop"] is False

    def test_the_power_is_the_area_times_the_rate(self):
        """Steinmetz at fixed amplitude: the area supplies k B^n and the
        frequency supplies the f. It matters because the carrier traverses
        the loop millions of times a second."""
        saturation, coercivity = self._material()
        area = mc.hysteresis_loss(saturation, coercivity)["major_loop_area"]
        power = mc.loss_power_w_per_m3(saturation, coercivity,
                                       np.array([1.0, 3.4e6]))
        assert float(power[0]) == pytest.approx(area)
        assert float(power[1]) == pytest.approx(area * 3.4e6)


class TestTheRecordHead:
    def test_the_static_geometry_is_in_the_signal(self):
        """Ethan: "we should also be able to derive that set of video head
        measurements, since we are reading the signal it wrote" - true of
        the STATIC geometry, which shaped magnetisation that is still
        there."""
        got = mc.record_head_derivable()
        assert "IN THE SIGNAL" in got["static_geometry"]
        assert "NOT recoverable" in got["time_varying_profile"]

    def test_it_arrives_as_a_product_not_a_term(self):
        """Which is why head_model's constants are a property of the
        (recording, playback) PAIR rather than of a machine."""
        got = mc.record_head_derivable()
        assert "PRODUCT" in got["recovered_as"]
        assert "PAIR" in got["recovered_as"]

    def test_it_names_what_would_separate_the_two_heads(self):
        got = mc.record_head_derivable()
        assert len(got["separators"]) == 3
        assert any("head switch" in s for s in got["separators"])

    def test_the_head_switch_gives_a_difference_of_differences(self):
        """Because both sides alternate - which is the four-way contrast
        the tesseract fold already measures."""
        got = mc.record_head_derivable()
        assert any("difference of" in s for s in got["separators"])
        assert "tesseract" in got["already_measured_by"]


class TestTheTwoBandPair:
    """Ethan: "Since colour under and luma occupy different bands on the
    tape, but are sourced from the same signal, this pair of signals on
    the tape are used to form the two on tape measurements needed for the
    flux density and the other per head magnetic properties."

    It works, for exactly two unknowns, and the limit is exact rather than
    statistical.
    """

    def test_the_pair_beats_either_band_alone(self):
        got = mc.two_band_pair()
        assert got["both"]["spacing_sigma_m"] < got["luma"]["spacing_sigma_m"]
        assert got["both"]["spacing_sigma_m"] < \
            got["colour_under"]["spacing_sigma_m"]
        assert got["improvement"] == pytest.approx(8.0, abs=0.5)

    def test_the_two_unknowns_become_separable(self):
        """0.997 is two names for one quantity; 0.811 is two quantities."""
        got = mc.two_band_pair()
        assert got["luma"]["coherence"] > 0.99
        assert got["both"]["coherence"] < 0.85

    def test_the_lever_is_the_wavelength_ratio(self):
        """629 kHz against 4 MHz on the same head in the same pass."""
        assert mc.two_band_pair()["wavelength_lever"] == pytest.approx(
            11.0, abs=0.5)

    def test_it_separates_a_flat_scale_from_a_sloped_loss(self):
        """Which is why it works: N*B*A is frequency-independent and the
        loss is not, so one band cannot tell the Faraday rise from the
        roll-off and two bands can."""
        got = mc.two_band_pair()
        assert "flat in frequency" in got["separates"][0]

    def test_a_narrower_lever_buys_less(self):
        wide = mc.two_band_pair()
        narrow = mc.two_band_pair(colour_under_hz=(2.0e6, 2.4e6))
        assert narrow["improvement"] < wide["improvement"]

    def test_spacing_and_transition_are_one_parameter_at_any_bandwidth(self):
        """THE EXACT LIMIT, and it is not statistical: both enter as
        exp(-2 pi x / lambda), so they are the same function and no lever
        separates them. A fit offered four names still measures one."""
        assert "SAME function" in mc.two_band_pair()["does_not_separate"]
        v = 5.8709
        f = np.linspace(0.4e6, 4.4e6, 500)
        lam = v / f
        spacing = -2 * np.pi * 1e-9 / lam
        transition = -2 * np.pi * 1e-9 / lam
        assert np.allclose(spacing, transition)


class TestTheComplexRotationTransfer:
    """The head's response is a complex quantity and `rotation_response`
    returned magnitudes. These hold the complex form to the three phases the
    magnitude threw away: Faraday's quarter turn, the separation loss's
    minimum-phase companion, and the gap sinc's sign."""

    def _band(self):
        return mc.spec_band_centres()

    def test_the_magnitude_is_unchanged(self):
        """The complex form must not move the level: it adds phase and
        nothing else, or every magnitude result already recorded would have
        to be re-measured."""
        speed = self._band()["writing_speed_m_s"]
        f = np.linspace(0.4e6, 4.4e6, 501)
        old = mc.rotation_response(f, 100, 0.2045e-6, speed)
        new = mc.rotation_transfer(f, 100, 0.2045e-6, speed)
        assert np.allclose(np.abs(new["volts_per_tesla_metre"]),
                           old["volts_per_tesla_metre"], rtol=1e-12)

    def test_faraday_carries_exactly_a_quarter_turn(self):
        """`e = -N dPhi/dt` is `j omega`. Written as `2 pi f N` the quarter
        turn is gone, and it is the same at every frequency - so it cancels
        out of a band difference, which is why it was possible to lose it
        without anything downstream complaining."""
        speed = self._band()["writing_speed_m_s"]
        f = np.linspace(0.4e6, 4.4e6, 101)
        got = mc.rotation_transfer(f, 100, 0.2045e-6, speed)
        assert np.allclose(np.degrees(np.angle(got["faraday"])), 90.0)

    def test_the_gap_sinc_never_changes_sign_in_band(self):
        """A 0.30 um gap at 5.80 m/s nulls at 19.33 MHz, far above the
        format's own band, so the sign correction is inert here. It is
        carried anyway because a caller may pass a wider gap or a slower
        speed, and a sign change is not something a magnitude can express."""
        speed = self._band()["writing_speed_m_s"]
        f = np.linspace(0.4e6, 4.4e6, 1001)
        got = mc.rotation_transfer(f, 100, 0.2045e-6, speed, gap_m=0.30e-6)
        assert got["gap_sign_flips"] == 0
        assert got["gap_null_hz"] == pytest.approx(speed / 0.30e-6)

    def test_a_wide_enough_gap_does_flip_and_the_flip_is_carried(self):
        """The control: put the null inside the band and the sign must
        appear, or the correction is untested rather than inert."""
        speed = self._band()["writing_speed_m_s"]
        gap = speed / 2.0e6                      # null placed at 2 MHz
        f = np.linspace(0.4e6, 4.4e6, 2001)
        got = mc.rotation_transfer(f, 100, 0.0, speed, gap_m=gap)
        assert got["gap_sign_flips"] >= 1
        below = got["gap_loss"][f < 1.9e6]
        above = got["gap_loss"][(f > 2.1e6) & (f < 3.9e6)]
        assert np.allclose(np.angle(below), 0.0, atol=1e-9)
        assert np.allclose(np.abs(np.angle(above)), np.pi, atol=1e-9)

    def test_the_transfer_is_complex_and_the_old_one_is_not(self):
        speed = self._band()["writing_speed_m_s"]
        f = np.linspace(0.4e6, 4.4e6, 51)
        assert np.iscomplexobj(mc.rotation_transfer(f, 1.0, 0.1e-6, speed)["response"])
        assert not np.iscomplexobj(mc.rotation_response(f, 1.0, 0.1e-6, speed)["response"])

    def test_the_group_delay_does_not_depend_on_the_grid(self):
        """THE PROPERTY THE NUMERICAL ROUTE LACKS. Two grids of different
        extent and different point count must give the same band-to-band
        delay for the same head, because a head does not know what grid it
        was evaluated on. Measured: -40.771 ns over 0.4-4.4 MHz in 4001
        points and -40.770 ns over 0.2-8.0 MHz in 8001, against a closed
        form of -40.756 ns."""
        band = self._band()
        speed = band["writing_speed_m_s"]
        answers = []
        for low, high, points in ((0.4e6, 4.4e6, 4001), (0.2e6, 8.0e6, 8001)):
            f = np.linspace(low, high, points)
            got = mc.rotation_transfer(
                f, 100, 0.2045e-6, speed,
                phase_reference_hz=band["colour_under_hz"])
            assert got["phase_reference_hz"] == band["colour_under_hz"]
            phase = np.unwrap(np.angle(got["volts_per_tesla_metre"]))
            delay = -np.gradient(phase, 2.0 * np.pi * f)
            answers.append(
                delay[int(np.argmin(np.abs(
                    f - band["luma_geometric_centre_hz"])))]
                - delay[int(np.argmin(np.abs(f - band["colour_under_hz"])))])
        assert answers[0] == pytest.approx(answers[1], rel=1e-3)
        assert answers[0] == pytest.approx(
            mc.band_delay_difference(0.2045e-6), rel=1e-3)

    def test_a_phase_reference_at_or_below_zero_is_refused(self):
        speed = self._band()["writing_speed_m_s"]
        with pytest.raises(ValueError):
            mc.rotation_transfer(np.linspace(1e6, 4e6, 32), 1.0, 0.1e-6,
                                 speed, phase_reference_hz=0.0)

    def test_its_group_delay_is_the_closed_form(self):
        """The phase carried on the separation loss must differentiate to
        the same band-to-band delay the closed form states, or the two
        halves of this module disagree with each other."""
        band = self._band()
        speed = band["writing_speed_m_s"]
        f = np.linspace(0.4e6, 4.4e6, 4001)
        got = mc.rotation_transfer(f, 100, 0.2045e-6, speed)
        phase = np.unwrap(np.angle(got["volts_per_tesla_metre"]))
        delay = -np.gradient(phase, 2.0 * np.pi * f)
        low = int(np.argmin(np.abs(f - band["colour_under_hz"])))
        high = int(np.argmin(np.abs(f - band["luma_geometric_centre_hz"])))
        assert (delay[high] - delay[low]) == pytest.approx(
            mc.band_delay_difference(0.2045e-6), rel=1e-3)


class TestTheTwoBandsOnTheTimeAxis:
    """Ethan: "the difference in time between the luma and chroma bands are
    the measurement we can use to observe the delay on each video head"."""

    def test_the_bands_come_from_the_standard_and_not_from_here(self):
        from vhsdecode.models import vhs_specification as spec
        band = mc.spec_band_centres()
        assert band["luma_low_hz"] == spec.value_of("fm_sync_tip_hz")
        assert band["luma_high_hz"] == spec.value_of("fm_peak_white_hz")
        assert band["colour_under_hz"] == spec.value_of("colour_under_hz")
        assert band["writing_speed_m_s"] == spec.value_of("writing_speed_m_s")

    def test_the_two_centres_differ_because_the_two_laws_differ(self):
        """Wallace's magnitude is linear in frequency and its group delay is
        linear in the logarithm, so a band average lands in two different
        places and one centre cannot serve both."""
        band = mc.spec_band_centres()
        assert band["luma_geometric_centre_hz"] < band["luma_linear_centre_hz"]
        assert band["luma_geometric_centre_hz"] == pytest.approx(
            math.sqrt(band["luma_low_hz"] * band["luma_high_hz"]))

    def test_the_luma_band_arrives_first(self):
        """A separation loss is a low pass and a minimum-phase low pass
        delays its low end most, so the high band LEADS. The sign is the
        part an instrument can get backwards without noticing."""
        assert mc.band_delay_difference(0.2045e-6) < 0.0

    def test_the_delay_is_linear_in_the_spacing(self):
        one = mc.band_delay_difference(0.1e-6)
        two = mc.band_delay_difference(0.2e-6)
        assert two == pytest.approx(2.0 * one)
        assert mc.band_delay_difference(0.0) == 0.0

    def test_the_delay_is_logarithmic_in_the_band_ratio(self):
        """Doubling the ratio twice must add the same amount each time -
        the property that distinguishes this law from the amplitude's,
        which is linear in the frequency DIFFERENCE."""
        first = mc.band_delay_difference(0.2e-6, low_hz=1e6, high_hz=2e6)
        second = mc.band_delay_difference(0.2e-6, low_hz=2e6, high_hz=4e6)
        assert first == pytest.approx(second)

    def test_a_band_centre_at_zero_is_refused(self):
        with pytest.raises(ValueError):
            mc.band_delay_difference(0.2e-6, low_hz=0.0)

    def test_the_closed_form_matches_an_independent_minimum_phase_route(self):
        """THE CONTROL THAT CAN FAIL IN BOTH DIRECTIONS.

        A first-order minimum-phase filter `1 - r z^-1` has a phase known in
        closed form, so the cepstral reconstruction used here is checked
        against it first - it agrees to about 1e-15 radians. Only then is
        the separation loss put through the same machinery, where it must
        reproduce `-(2 d / (pi v)) ln(f2 / f1)`. A construction that had
        collapsed to zero would pass neither.
        """
        n = 1 << 16

        def minimum_phase(log_magnitude):
            full = np.concatenate([log_magnitude, log_magnitude[-2:0:-1]])
            cepstrum = np.fft.ifft(full).real
            window = np.zeros(n)
            window[0] = 1.0
            window[1:n // 2] = 2.0
            window[n // 2] = 1.0
            return np.angle(np.exp(np.fft.fft(cepstrum * window)))[:n // 2 + 1]

        theta = 2.0 * np.pi * np.fft.rfftfreq(n, 1.0)
        reference = 1.0 - 0.95 * np.exp(-1j * theta)
        assert np.max(np.abs(np.angle(reference)
                             - minimum_phase(np.log(np.abs(reference))))) < 1e-12

        # the separation loss on the same machinery. The digital radian
        # frequency stands in for `omega`, so `a` stands in for `d / v` and
        # the law must come back with that scale exactly.
        for a in (0.5, 1.0, 2.0):
            delay = -np.gradient(minimum_phase(-a * theta), theta)
            low = int(np.argmin(np.abs(theta - 0.05)))
            high = int(np.argmin(np.abs(theta - 0.31)))
            assert (delay[high] - delay[low]) == pytest.approx(
                -(2.0 * a / np.pi) * np.log(0.31 / 0.05), rel=0.01)

    def test_the_estimator_inverts_the_law(self):
        for spacing in (0.05e-6, 0.2045e-6, 0.6e-6):
            got = mc.spacing_from_band_delay(
                mc.band_delay_difference(spacing))
            assert got["spacing_m"] == pytest.approx(spacing)

    def test_the_amplitude_estimator_inverts_wallace(self):
        band = mc.spec_band_centres()
        for spacing in (0.05e-6, 0.2045e-6, 0.6e-6):
            tilt = -(2.0 * np.pi * spacing / band["writing_speed_m_s"]) \
                * band["frequency_difference_hz"]
            assert mc.spacing_from_band_tilt(tilt)["spacing_m"] == \
                pytest.approx(spacing)

    def test_the_error_bar_carries_through_both_estimators(self):
        delay = mc.spacing_from_band_delay(-40e-9, 1e-9)
        assert delay["spacing_error_m"] == pytest.approx(
            delay["metres_per_second_of_delay"] * 1e-9)
        tilt = mc.spacing_from_band_tilt(-0.7, 0.01)
        assert tilt["spacing_error_m"] == pytest.approx(
            tilt["metres_per_neper"] * 0.01)

    def test_a_consistent_pair_agrees_with_itself(self):
        """Fed a delay and a tilt that the same separation produced, the
        two routes must return that separation and no discrepancy. This is
        the null case, and a pair that cannot pass it cannot be read when it
        fails."""
        band = mc.spec_band_centres()
        spacing = 0.2045e-6
        tilt = -(2.0 * np.pi * spacing / band["writing_speed_m_s"]) \
            * band["frequency_difference_hz"]
        got = mc.two_band_delay_pair(mc.band_delay_difference(spacing), tilt,
                                     1e-12, 1e-6)
        assert got["time_spacing_m"] == pytest.approx(spacing)
        assert got["amplitude_spacing_m"] == pytest.approx(spacing)
        assert got["ratio"] == pytest.approx(1.0)
        assert abs(got["difference_m"]) < 1e-15

    def test_the_measured_pair_disagrees_and_says_so(self):
        """THE REAL MEASUREMENT, kept as a test so the result cannot drift
        silently. `tap_transfer`'s record-tap to playback-tap transfer of one
        Sony SLV-778HF, 75bars SP, head 0 minus head 1: a delay difference of
        +11.533 +/- 0.388 ns and a log-magnitude tilt of +0.13625 +/- 0.00046
        nepers. The two routes agree in sign and disagree by ten standard
        deviations, which is the finding rather than the failure."""
        got = mc.two_band_delay_pair(
            11.533e-9, 0.13625, 0.388e-9, 0.00046,
            delay_low_hz=0.62996e6, delay_high_hz=3.88880e6,
            tilt_low_hz=0.62729e6, tilt_high_hz=4.00623e6)
        assert got["time_spacing_m"] * 1e9 == pytest.approx(-57.7, abs=1.0)
        assert got["amplitude_spacing_m"] * 1e9 == pytest.approx(-37.2, abs=1.0)
        assert got["time_spacing_m"] < 0 and got["amplitude_spacing_m"] < 0
        assert got["sigma"] > 3.0
        assert got["usable"]

    def test_the_coherence_floor_is_derived_and_not_chosen(self):
        """It is the coherence at which one average's phase error reaches a
        radian, `sqrt((1 - g^2) / (2 g^2)) = 1`, so `g^2 = 1/3`."""
        g = mc.PHASE_COHERENCE_FLOOR
        assert math.sqrt((1.0 - g * g) / (2.0 * g * g)) == pytest.approx(1.0)

    def test_a_low_coherence_reading_is_refused_however_small_its_error_bar(self):
        """MEASURED, and this is why the gate is on the coherence and not on
        the error bar. The luma-only capture `pulseandbar-y-only` has no
        colour-under signal at all - coherence 0.127 - and reports a head
        delay difference of +2229.70 +/- 104.29 ns, twenty-one standard
        deviations of nonsense. An instrument reading the error bar alone
        would have taken it."""
        got = mc.two_band_delay_pair(2229.70e-9, 0.0, 104.29e-9, 0.01,
                                     coherence=0.127)
        assert not got["usable"]
        assert "coherence" in got["why"]
        good = mc.two_band_delay_pair(11.533e-9, 0.13625, 0.388e-9, 0.00046,
                                      coherence=0.956)
        assert good["usable"]

    def test_the_sensitivity_is_stated_in_both_currencies(self):
        """The number an instrument builder needs first: what a nanosecond
        is worth against what a decibel is worth."""
        delay = mc.spacing_from_band_delay(0.0)
        tilt = mc.spacing_from_band_tilt(0.0)
        assert delay["metres_per_second_of_delay"] * 1e-9 == \
            pytest.approx(5.018e-9, rel=1e-3)
        assert tilt["metres_per_neper"] == pytest.approx(282.24e-9, rel=1e-3)


class TestTheRecordSideWriteField:
    """Ethan asked twice for the head modelled "in terms of voltage and
    current that is being used magnetize the head", and the module carried
    the field inside the CORE and never the field at the TAPE. These hold
    the step that was missing."""

    def test_the_efficiency_is_the_gap_share_the_transformer_already_had(self):
        """It was computed all along under another name and never used to
        reach the tape. Holding the two together stops them drifting."""
        assert mc.head_efficiency()["efficiency"] == pytest.approx(
            mc.head_transformer()["gap_share"])

    def test_a_longer_gap_takes_a_larger_share_of_the_drive(self):
        """Two reluctances in series divide the magnetomotive force, and the
        gap's reluctance rises with its length."""
        assert mc.head_efficiency(gap_m=0.6e-6)["efficiency"] > \
            mc.head_efficiency(gap_m=0.3e-6)["efficiency"]

    def test_the_deep_gap_field_is_the_efficiency_times_the_ampere_turns(self):
        gap, turns, current = 0.30e-6, 100.0, 1e-3
        expected = (mc.head_efficiency(gap_m=gap)["efficiency"]
                    * turns * current / gap)
        assert mc.deep_gap_field_a_m(current, turns, gap) == \
            pytest.approx(expected)

    def test_the_karlqvist_field_has_no_sources_and_no_currents(self):
        """THE CONTROL, AND IT SETTLES THE SIGN. A field in a region with no
        sources and no currents has zero divergence and zero curl, which is
        the same statement as `Hx - j Hy` being analytic. Sources write the
        perpendicular component's sign both ways; only one of them passes.
        Measured: with the sign this module uses, both read about 1e-01
        against a derivative scale of 5.5e+05; reversed, both read 1e+06."""
        gap, step = 0.30e-6, 1e-10
        x, y = 0.07e-6, 0.22e-6

        def at(dx, dy):
            got = mc.karlqvist_field(np.array([x + dx]), np.array([y + dy]),
                                     gap, 1.0)
            return (float(got["along_track_a_m"][0]),
                    float(got["perpendicular_a_m"][0]))

        dhx_dx = (at(step, 0)[0] - at(-step, 0)[0]) / (2 * step)
        dhx_dy = (at(0, step)[0] - at(0, -step)[0]) / (2 * step)
        dhy_dx = (at(step, 0)[1] - at(-step, 0)[1]) / (2 * step)
        dhy_dy = (at(0, step)[1] - at(0, -step)[1]) / (2 * step)
        scale = abs(dhx_dx)
        assert abs(dhx_dx + dhy_dy) < 1e-5 * scale          # divergence
        assert abs(dhy_dx - dhx_dy) < 1e-5 * scale          # curl

    def test_the_field_is_returned_complex(self):
        got = mc.karlqvist_field(np.array([0.0]), np.array([0.2e-6]))
        assert np.iscomplexobj(got["field"])
        assert got["field"].real == pytest.approx(got["along_track_a_m"])
        assert got["field"].imag == pytest.approx(-got["perpendicular_a_m"])

    def test_the_perpendicular_component_vanishes_under_the_gap_centre(self):
        """Which is why the write depth has a closed form: at x = 0 the
        symmetry kills one component and the other is a single arctangent."""
        got = mc.karlqvist_field(np.array([0.0]), np.array([0.1e-6, 0.3e-6]))
        assert np.allclose(got["perpendicular_a_m"], 0.0)

    def test_the_write_depth_matches_the_field_it_came_from(self):
        """The closed form must agree with the field it was derived from,
        or one of the two is wrong. Measured at a drive of 2.5: the closed
        form gives 0.2064 um and the field's own contour 0.2063 um."""
        gap, ratio = 0.30e-6, 2.5
        coercivity = 1.0
        depth = mc.write_depth_m(ratio, gap)
        got = mc.karlqvist_field(np.array([0.0]), np.array([depth]), gap,
                                 ratio * coercivity)
        assert float(got["magnitude_a_m"][0]) == pytest.approx(coercivity)

    def test_the_deepest_contour_really_is_under_the_gap_centre(self):
        """The closed form assumes it, so it is measured rather than
        assumed: on a grid across the track the deepest point at which the
        field still reaches the coercivity sits at x = 0."""
        gap, ratio = 0.30e-6, 2.5
        x = np.linspace(-3e-6, 3e-6, 601)
        y = np.linspace(1e-9, 1.0e-6, 501)
        grid_x, grid_y = np.meshgrid(x, y)
        got = mc.karlqvist_field(grid_x, grid_y, gap, ratio)
        reach = got["magnitude_a_m"] >= 1.0
        deepest = int(np.max(np.flatnonzero(reach.any(axis=1))))
        assert abs(float(np.mean(x[np.flatnonzero(reach[deepest])]))) < 1e-9
        assert y[deepest] == pytest.approx(mc.write_depth_m(ratio, gap),
                                           rel=0.01)

    def test_a_drive_below_the_coercivity_records_nothing(self):
        assert mc.write_depth_m(1.0) == 0.0
        assert mc.write_depth_m(0.5) == 0.0
        assert mc.write_depth_m(1.001) > 0.0

    def test_the_depth_and_the_drive_invert_each_other(self):
        for ratio in (1.5, 2.0, 2.441, 3.0, 5.0):
            assert mc.drive_ratio_for_depth(mc.write_depth_m(ratio)) == \
                pytest.approx(ratio)

    def test_the_two_assumed_depths_are_one_quantity_at_two_currents(self):
        """`magnetic.NOMINAL_DEPTH_M` calls its 0.15 um "a SCALE, not a
        constant of the format" and `tape_model` carries 0.20 um as the
        recording depth. Read back through the write field, with the
        separation added because the coating begins there, they are drives
        of 2.441 and 2.907 times the coercivity - the same quantity at two
        record currents, which is the axis `magnetic.level_axis` measures."""
        from vhsdecode.models import magnetic, tape_model
        spacing = 0.05e-6
        assert mc.drive_ratio_for_depth(
            magnetic.NOMINAL_DEPTH_M + spacing) == pytest.approx(2.441, abs=0.01)
        depth = [entry["typical"] for entry in tape_model.TAPE_VARIATIONS
                 if entry["name"] == "coating thickness"][0]
        assert mc.drive_ratio_for_depth(depth + spacing) == \
            pytest.approx(2.907, abs=0.01)

    def test_the_optimum_is_the_standard_s_own_definition(self):
        """SMPTE 32M 3.9.1.1.6 defines the record current only by what it
        achieves - the maximum playback output - and gives no value.
        Measured through that definition: 3.598 at sync tip, 3.278 at the
        luma centre, 3.000 at peak white."""
        from vhsdecode.models import vhs_specification as spec
        assert spec.value_of("record_current_a") is None
        band = mc.spec_band_centres()
        assert mc.optimum_record_drive(band["luma_low_hz"])["drive_ratio"] == \
            pytest.approx(3.598, abs=0.01)
        assert mc.optimum_record_drive()["drive_ratio"] == \
            pytest.approx(3.278, abs=0.01)
        assert mc.optimum_record_drive(band["luma_high_hz"])["drive_ratio"] == \
            pytest.approx(3.000, abs=0.01)
        assert mc.optimum_record_drive()["clause"] == "SMPTE 32M 3.9.1.1.6"

    def test_the_optimum_writes_exactly_to_the_read_depth(self):
        """That is the whole argument, so it is held rather than described:
        at the optimum the layer written reaches the deepest point the
        reproduce head can see and no further."""
        got = mc.optimum_record_drive()
        assert got["write_depth_m"] == pytest.approx(
            got["read_depth_m"] + 0.05e-6)

    def test_one_current_serves_the_whole_fm_carrier_and_not_the_chroma(self):
        """WHY THE CLAUSE IS WORDED AS IT IS. Across the FM carrier's own
        band the optimum moves from 3.000 to 3.598, so one current serves
        all of it. The colour-under carrier asks for 15.934, which is 4.9
        times the luma's and above the 7.50 of headroom `head_transformer`
        measures before the core saturates - so the chroma's optimum is
        unreachable, not merely unchosen."""
        band = mc.spec_band_centres()
        low = mc.optimum_record_drive(band["luma_high_hz"])["drive_ratio"]
        high = mc.optimum_record_drive(band["luma_low_hz"])["drive_ratio"]
        assert high / low < 1.25
        chroma = mc.optimum_record_drive(band["colour_under_hz"])["drive_ratio"]
        assert chroma == pytest.approx(15.934, abs=0.05)
        assert chroma > mc.head_transformer()["headroom"]

    def test_the_write_depth_does_not_depend_on_frequency(self):
        """The head's field knows the drive and the geometry and nothing
        about the signal, so every band written by one head at one current
        goes to the same depth. It is the READ depth that is a wavelength."""
        band = mc.spec_band_centres()
        depths = {mc.recorded_layer(3.0, frequency_hz=f)["write_depth_m"]
                  for f in (band["luma_low_hz"], band["luma_high_hz"],
                            band["colour_under_hz"])}
        assert len(depths) == 1

    def test_the_colour_under_layer_is_bounded_by_the_record_current(self):
        """THE FINDING. At a drive of 3.000 the colour-under band can read
        1.4667 um down and nothing is recorded below 0.2598 um, so its
        effective thickness is the record current's and not its
        wavelength's. The margin is +1.2569 um of coating the head never
        magnetised."""
        band = mc.spec_band_centres()
        got = mc.recorded_layer(3.0, frequency_hz=band["colour_under_hz"])
        assert got["bound_by"] == "write"
        assert got["margin_m"] * 1e6 == pytest.approx(1.2569, abs=0.01)
        assert got["thickness_m"] * 1e6 == pytest.approx(0.2098, abs=0.01)

    def test_the_margin_changes_sign_at_the_optimum(self):
        """The margin is the quantity and the label is the convenience, so
        the margin is what is held: negative below the optimum drive, zero
        at it, positive above."""
        band = mc.spec_band_centres()
        optimum = mc.optimum_record_drive()["drive_ratio"]
        frequency = band["luma_geometric_centre_hz"]
        assert mc.recorded_layer(optimum * 1.2,
                                 frequency_hz=frequency)["margin_m"] < 0.0
        assert mc.recorded_layer(optimum,
                                 frequency_hz=frequency)["margin_m"] == \
            pytest.approx(0.0, abs=1e-12)
        assert mc.recorded_layer(optimum * 0.9,
                                 frequency_hz=frequency)["margin_m"] > 0.0


class TestTheTapeSOwnBHCurveAndTheOptimumCurrent:
    """Ethan: "use the BH curve to model this part". The curve in this
    module was only ever applied to the head's ferrite; these hold it to the
    tape, and hold the two independent routes to the record current the
    standard defines only operationally."""

    def test_the_magnetisation_profile_stops_at_the_write_depth(self):
        """Below the depth where the field falls under the coercivity
        nothing is recorded, so the profile has no business existing there."""
        got = mc.recorded_magnetisation(3.0)
        assert got["depth_m"][-1] == pytest.approx(got["write_depth_m"])
        assert got["depth_m"][0] == pytest.approx(0.05e-6)
        assert np.all(np.diff(got["depth_m"]) > 0)

    def test_the_field_falls_to_the_coercivity_at_the_bottom(self):
        coercivity = 600.0 * 1000.0 / (4.0 * math.pi)
        got = mc.recorded_magnetisation(3.0, coercivity_a_m=coercivity)
        assert float(got["field_a_m"][-1]) == pytest.approx(coercivity,
                                                            rel=1e-6)
        assert float(got["field_a_m"][0]) > coercivity

    def test_the_magnetisation_falls_with_depth_and_never_exceeds_remanence(self):
        got = mc.recorded_magnetisation(4.0, remanence_t=0.15)
        assert np.all(np.diff(got["magnetisation_t"]) < 0)
        assert float(np.max(got["magnetisation_t"])) < 0.15

    def test_a_drive_that_records_nothing_returns_an_empty_profile(self):
        got = mc.recorded_magnetisation(1.0)
        assert got["depth_m"].size == 0
        assert mc.linked_flux(1.0, 3.9e6) == 0.0

    def test_the_depth_integral_alone_has_no_maximum(self):
        """THE CONTROL FOR THE OPTIMUM'S CAUSE. Driving harder always writes
        more magnetisation, so without the transition length the linked flux
        rises to the end of any grid. Measured over 1.02 to 20 times the
        coercivity it is monotone at every frequency in the band."""
        band = mc.spec_band_centres()
        for frequency in (band["luma_low_hz"], band["luma_geometric_centre_hz"],
                          band["luma_high_hz"], band["colour_under_hz"]):
            got = mc.record_current_curve(frequency)
            assert got["without_transition_is_monotone"]
            plain = got["linked_flux_without_transition"]
            # zero until the field reaches past the head-to-tape separation,
            # which is the honest answer rather than a defect, and strictly
            # rising from there
            live = plain[plain > 0]
            assert np.all(np.diff(plain) >= 0)
            assert np.all(np.diff(live) > 0)
            assert live.size > 0.5 * plain.size

    def test_the_transition_length_is_what_turns_the_curve_over(self):
        """With it there is a peak, and it is interior to the grid rather
        than at either end - which is what a maximum means."""
        got = mc.record_current_curve()
        flux = got["linked_flux"]
        peak = int(np.argmax(flux))
        assert 0 < peak < len(flux) - 1

    def test_the_two_routes_to_the_optimum_agree(self):
        """THE PAIR, AND NOTHING WAS TUNED TO MAKE IT AGREE. One route knows
        only the depth the reproduce head can see against the depth the
        field reaches; the other knows only the tape's remanence curve and
        the length of a reversal. Measured: 3.616 against 3.598 at sync tip,
        3.315 against 3.278 at the luma centre, 3.046 against 3.000 at peak
        white, and 15.520 against 15.934 for the colour under."""
        band = mc.spec_band_centres()
        expected = {"luma_low_hz": 3.616, "luma_geometric_centre_hz": 3.315,
                    "luma_high_hz": 3.046, "colour_under_hz": 15.520}
        for name, peak in expected.items():
            got = mc.record_current_curve(band[name])
            assert got["peak_drive_ratio"] == pytest.approx(peak, abs=0.02)
            assert got["agreement"] == pytest.approx(1.0, abs=0.03)

    def test_the_curve_carries_the_clause_it_implements(self):
        assert mc.record_current_curve()["clause"] == "SMPTE 32M 3.9.1.1.6"
