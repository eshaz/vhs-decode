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
    def test_the_record_head_reaches_amplitude_and_never_phase(self):
        got = mc.record_side_observability()
        assert "observable" in got["amplitude"]
        assert "NOT observable" in got["phase"]

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
