"""Tape bias: the luminance FM as the chrominance's AC bias.

Ethan: "look through the VHS specs and magnetic tape recording principles to
check if tape bias is a component to this measurement."

The answer is that it is stated in both normative documents, so most of what
these tests assert is that the model carries the clause rather than a
paraphrase of it - and that the two quantitative consequences, the operating
point the standard chooses and the third-order product it leaves behind, come
out where the specification says they should.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import colour_under, magnetic_circuit, tape_bias
from vhsdecode.models import vhs_specification


# --------------------------------------------------------------------------
# The specification is carried, not paraphrased
# --------------------------------------------------------------------------

def test_the_bias_clause_is_quoted_with_its_scope():
    """SMPTE 32M says it once, in the S-VHS clause, and the scope travels."""
    entry = tape_bias.SPECIFICATION["bias_stated"]
    assert entry["clause"] == "7.5.1.2.2"
    assert "acting as bias" in entry["quote"]
    assert entry["scope"] == "S-VHS"
    # and the baseline-VHS statement is the guide's, not the standard's
    guide = tape_bias.SPECIFICATION["bias_stated_for_vhs"]
    assert guide["source"].startswith("JVC")
    assert "AC bias" in guide["quote"]
    assert guide["scope"] == "VHS"


def test_every_clause_carries_a_source_and_a_quote():
    for name, entry in tape_bias.SPECIFICATION.items():
        assert entry["source"], name
        assert entry["clause"], name
        assert len(entry["quote"]) > 20, name


def test_the_absences_include_the_ones_that_carry_the_argument():
    """The chroma has no record-current clause and the standard states no
    carrier amplitude; both are load-bearing and both are absences."""
    absent = tape_bias.absent_from_the_standard()
    assert "chroma_record_current" in absent
    assert "fm_carrier_amplitude" in absent
    assert "saturation_referent" in absent
    assert "video_bias_in_clause_3" in absent
    for name, reason in absent.items():
        assert len(reason) > 20, name


def test_the_standard_really_does_not_state_the_things_relied_on():
    """Cross-checked against `vhs_specification`, which was written from the
    same document by a different pass: the head values this bias model needs
    are not in the standard, so anything numeric here is the guide's."""
    present = vhs_specification.head_values_present()
    assert present["coercivity_a_m"] is True
    assert present["remanence_t"] is False
    assert present["coating_thickness_m"] is False
    assert present["gap_length_m"] is False


# --------------------------------------------------------------------------
# Is the FM a valid bias by the format's own rule
# --------------------------------------------------------------------------

def test_the_bias_ratio_meets_the_guides_own_criterion_everywhere():
    """The guide states the rule for the linear audio track and states no
    other: at least twice, typically three to five times."""
    out = tape_bias.bias_ratio_over_deviation()
    minimum = tape_bias.AUDIO_BIAS_RULE[0]
    assert out["lowest_ratio"] >= minimum
    assert out["meets_the_minimum"]
    # the FM is above the typical band across the whole deviation
    assert out["above_the_typical_band"]
    assert out["points"]["sync_tip"]["ratio"] == pytest.approx(5.402, abs=1e-3)
    assert out["points"]["peak_white"]["ratio"] == pytest.approx(6.991,
                                                                abs=1e-3)


def test_the_bias_ratio_swings_across_the_deviation():
    """The prediction a two-independent-channels account cannot make: the
    bias is a function of the picture, so the sensitivity is too."""
    out = tape_bias.bias_ratio_over_deviation()
    assert out["swing"] == pytest.approx(4.4 / 3.4, rel=1e-6)
    assert out["highest_ratio"] > out["lowest_ratio"]


def test_the_signal_wavelength_is_the_colour_unders():
    out = tape_bias.bias_ratio_over_deviation()
    speed = float(magnetic_circuit.spec_band_centres()["writing_speed_m_s"])
    assert out["signal_wavelength_m"] == pytest.approx(
        speed / colour_under.carrier_hz("NTSC"))
    # and every bias wavelength is shorter than the signal's, which is what
    # makes it a bias at all
    for row in out["points"].values():
        assert row["bias_wavelength_m"] < out["signal_wavelength_m"]


# --------------------------------------------------------------------------
# It is a FEW-cycle bias, and the model says so rather than glossing it
# --------------------------------------------------------------------------

def test_the_ideal_anhysteretic_limit_is_not_reached():
    """The honest limit of the mechanism, from the arc's own Karlqvist field.

    Anhysteretic magnetisation needs the field's envelope to decay by a small
    fraction per bias cycle. It does not: measured 5.045 at the coercivity
    crossing, where the ideal wants far below one.
    """
    out = tape_bias.bias_cycles()
    assert out["reaches_the_ideal"] is False
    assert out["fractional_change_per_cycle"] > 1.0
    assert out["cycles"]["2_to_0.5_Hc"] < 1.0
    # and the far tail is where the ten cycles come from, not the switching
    # band, so the two must not be confused
    assert out["cycles"]["2_to_0.02_Hc"] > out["cycles"]["2_to_0.5_Hc"] * 5


def test_the_cycle_count_rises_with_the_bias_frequency():
    """A shorter bias wavelength buys more cycles over the same geometry,
    which is the axis the differential gain could be riding."""
    low = tape_bias.bias_cycles(bias_hz=3.4e6)
    high = tape_bias.bias_cycles(bias_hz=4.4e6)
    assert high["cycles"]["1_to_0.1_Hc"] > low["cycles"]["1_to_0.1_Hc"]
    assert high["bias_wavelength_m"] < low["bias_wavelength_m"]


def test_the_field_at_the_top_of_the_coating_is_below_the_deep_gap_field():
    """A sanity check on the geometry rather than on the physics: the
    Karlqvist field at depth is less than the field in the gap."""
    out = tape_bias.bias_cycles()
    assert out["peak_field_over_coercivity"] < out["drive_ratio"]
    assert out["peak_field_over_coercivity"] > 1.0


# --------------------------------------------------------------------------
# The transfer, and the prediction that separates bias from no bias
# --------------------------------------------------------------------------

def test_biased_distortion_falls_as_the_level_falls():
    out = tape_bias.anhysteretic_distortion([3.0, 1.0, 0.5, 0.25])
    fraction = out["third_harmonic_fraction"]
    assert np.all(np.diff(fraction) < 0.0)


def test_unbiased_distortion_rises_as_the_level_falls():
    """The discriminating prediction, and it goes the other way."""
    out = tape_bias.unbiased_distortion([3.0, 2.0, 1.0])
    fraction = out["third_harmonic_fraction"]
    assert np.all(np.diff(fraction) > 0.0)
    assert fraction[0] == pytest.approx(0.113, abs=0.005)
    assert fraction[-1] > 1.0


def test_the_unbiased_model_refuses_below_the_coercivity():
    """Below Hc the parametrisation stops meaning anything, and the module
    says which rows are usable rather than reporting a number there."""
    out = tape_bias.unbiased_distortion([0.5, 1.0, 2.0])
    assert list(out["usable"]) == [False, True, True]
    assert out["why_cut_at_one_coercivity"]


def test_the_standards_operating_point_is_below_the_coercivity():
    """The whole argument in one assertion: 3.9.2.1.2 puts the chroma at a
    field that, unbiased, would record nothing."""
    out = tape_bias.operating_point()
    for name, row in out["window"].items():
        assert row["below_the_coercivity"] is True, name
        assert row["drive_ratio"] < 1.0, name


def test_the_back_off_buys_an_order_of_magnitude_in_distortion():
    out = tape_bias.operating_point()
    luma = out["luma_third_harmonic_fraction"]
    seven = out["window"]["7_db_below_saturation"]["third_harmonic_fraction"]
    ten = out["window"]["10_db_below_saturation"]["third_harmonic_fraction"]
    assert luma / seven > 8.0
    assert luma / ten > 17.0
    assert seven == pytest.approx(0.0294, abs=5e-4)
    assert ten == pytest.approx(0.0141, abs=5e-4)


def test_saturation_is_the_square_wave_limit():
    """3.9.2.1.2 names a saturation and defines none, so the model has to
    say which one it means: the fundamental of a fully limited sinusoid."""
    out = tape_bias.operating_point()
    assert out["saturated_fundamental"] == pytest.approx(4.0 / math.pi)
    hard = tape_bias.anhysteretic_distortion([1e4])
    assert float(hard["fundamental"][0]) == pytest.approx(4.0 / math.pi,
                                                          rel=1e-3)


# --------------------------------------------------------------------------
# What bias says about the record current this tree already computed
# --------------------------------------------------------------------------

def test_the_colour_unders_own_optimum_is_unreachable():
    """`magnetic_circuit`'s 15.934 reproduced, and placed against the core."""
    out = tape_bias.record_current_scope()
    assert out["colour_under_optimum_drive"] == pytest.approx(15.93, abs=0.02)
    assert out["ratio"] == pytest.approx(4.86, abs=0.02)
    assert out["exceeds_the_core"] is True
    assert out["over_the_core_headroom"] > 2.0


def test_the_optimum_clause_is_scoped_to_the_fm_carrier():
    out = tape_bias.record_current_scope()
    assert "entire bandwidth of the FM carrier" in out["clause_scope"]
    assert out["chroma_has_no_current_clause"] is True


def test_the_chroma_occupies_the_layer_the_luma_wrote():
    """Bias supplies the justification `recorded_layer` was missing: the
    colour under's depth is the luma's write depth and nothing of its own."""
    out = tape_bias.chroma_recorded_layer()
    assert out["same_bottom"] is True
    assert out["chroma_is_write_bound"] is True
    # its own reading limit is six times deeper than anything that was written
    assert (out["chroma_read_depth_m"]
            > 5.0 * out["chroma_layer"]["layer_bottom_m"])


def test_the_head_and_the_tape_each_own_their_half():
    out = tape_bias.head_to_tape_link()
    assert out["anhysteretic_susceptibility_t_per_a_m"] == pytest.approx(
        out["remanence_t"] / out["coercivity_a_m"])
    assert out["write_depth_m"] > 0.0
    assert "remanence" in out["assumption"]


# --------------------------------------------------------------------------
# The residual: which product, and where the format put it
# --------------------------------------------------------------------------

def test_the_product_demodulates_to_twice_the_colour_under():
    assert tape_bias.beat_hz("NTSC") == pytest.approx(
        2.0 * colour_under.carrier_hz("NTSC"))
    out = tape_bias.intermodulation_products()
    for name in ("f_y - 2 f_c", "f_y + 2 f_c"):
        assert out["products"][name]["order"] == 3
        assert out["products"][name]["demodulates_to_hz"] == pytest.approx(
            tape_bias.beat_hz("NTSC"))
    assert out["the_one_that_matters"] == "f_y - 2 f_c"


def test_the_interleave_is_the_formats_own_arithmetic():
    """40 f_H, a quarter turn a line, and therefore 80 f_H +- half a line
    rate - the guide's stated reason for choosing 629.371 kHz."""
    out = tape_bias.interleave()
    assert out["harmonic_of_line_rate"] == pytest.approx(40.0)
    assert out["product_harmonic_of_line_rate"] == pytest.approx(80.0)
    assert out["product_offset_lines"] == 0.5
    assert out["alternates_line_to_line"] is True
    assert out["product_hz"] == pytest.approx(1258741.26, abs=0.1)


# --------------------------------------------------------------------------
# The estimator, on planted signals
# --------------------------------------------------------------------------

_LINE_RATE = colour_under.line_rate_hz("NTSC")
_SAMPLE_RATE = 910.0 * _LINE_RATE
_SPAN = 728            # eight groups of 91, so 80 f_H lands on a bin


def _planted(lines, still=0.0, alternating=0.0, noise=0.0, seed=0):
    """A picture with a known still component and a known alternating one at
    exactly the product's frequency."""
    t = np.arange(_SPAN)
    tone = np.sin(2.0 * np.pi * tape_bias.beat_hz("NTSC") * t / _SAMPLE_RATE)
    sign = ((-1.0) ** np.arange(lines))[:, None]
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, noise, (lines, _SPAN))
            + still * tone + alternating * sign * tone)


def test_the_product_lands_on_an_exact_bin():
    out = tape_bias.product_in_the_luma(_planted(64), _SAMPLE_RATE, 0, _SPAN)
    assert out["lands_on_a_bin"] is True
    assert out["cycles_in_the_span"] == pytest.approx(64.0)
    assert out["samples_per_line"] == 910


def test_a_planted_alternating_product_is_recovered_and_the_still_is_not():
    out = tape_bias.product_in_the_luma(
        _planted(120, still=50.0, alternating=7.0, noise=10.0),
        _SAMPLE_RATE, 0, _SPAN)
    assert out["alternating_peak"] == pytest.approx(7.0, rel=0.02)
    assert 2.0 * out["still_magnitude"] == pytest.approx(50.0, rel=0.02)
    # the two projections are orthogonal, so neither leaks into the other
    assert out["control_alternating_peak"] < 0.2
    assert out["ratio"] > 20.0


def test_the_measured_offset_finds_the_half_line_rate():
    """The interleave read back off the material rather than assumed."""
    out = tape_bias.product_in_the_luma(
        _planted(128, alternating=5.0, noise=1.0), _SAMPLE_RATE, 0, _SPAN)
    assert abs(out["offset_cycles_per_line"]) == pytest.approx(0.5, abs=1e-9)
    assert out["interleaved"] is True


def test_a_picture_with_no_product_reads_its_own_control():
    out = tape_bias.product_in_the_luma(
        _planted(120, still=50.0, noise=10.0), _SAMPLE_RATE, 0, _SPAN)
    assert out["alternating_peak"] < 3.0 * out["control_alternating_peak"]
    assert out["interleaved"] is False


def test_the_estimator_is_complex():
    """Outputs complex wherever a phase exists, and the phase is the axis the
    whole measurement stands on."""
    out = tape_bias.product_in_the_luma(
        _planted(64, alternating=3.0), _SAMPLE_RATE, 0, _SPAN)
    assert out["is_complex"] is True
    assert isinstance(out["alternating"], complex)
    assert isinstance(out["still"], complex)
    split = tape_bias.alternating_split(
        tape_bias.line_amplitudes(_planted(64, alternating=3.0),
                                  tape_bias.beat_hz("NTSC"), _SAMPLE_RATE,
                                  0, _SPAN))
    assert np.iscomplexobj(split["still"])
    assert np.iscomplexobj(split["alternating"])


def test_a_constant_line_contributes_nothing():
    """Each line's mean is removed over the same span the sum runs on, so the
    window's response to a constant multiplies a constant that is zero."""
    lines = np.full((32, _SPAN), 12345.0)
    out = tape_bias.product_in_the_luma(lines, _SAMPLE_RATE, 0, _SPAN)
    assert out["alternating_magnitude"] < 1e-9
    assert out["still_magnitude"] < 1e-9


def test_the_span_must_be_positive():
    with pytest.raises(ValueError):
        tape_bias.product_in_the_luma(_planted(8), _SAMPLE_RATE, 10, 10)


# --------------------------------------------------------------------------
# The refusal, and the measurements it rests on
# --------------------------------------------------------------------------

def test_the_differential_gain_is_refused_with_its_reason():
    out = tape_bias.differential_gain_refused()
    assert "one tap" in out["because"]
    assert out["would_settle_it"]
    assert "collinear" in out["and"]


def test_the_measured_table_carries_its_controls():
    """A measurement without its control is not reportable, so the table has
    to hold both."""
    measured = tape_bias.MEASURED
    slope = measured["chroma_gain_slope_db_per_mhz"]["SP"]
    control = measured["fm_envelope_slope_db_per_mhz"]["SP"]
    assert slope["tape"] / slope["tape_se"] > 20.0
    assert abs(control["tape"]) < 2.0 * control["tape_se"]
    # the product against the tap that cannot have made it
    assert measured["product_playback_tap_db"] > 5.0
    assert measured["product_record_tap_db"] < 1.5
    assert measured["product_y_only_playback_db"] < 1.0
    assert max(measured["product_control_db"]) < 1.0


def test_the_third_order_term_is_the_one_that_separates():
    """At BOTH speeds the third-order product is at the floor at the record
    tap and well above it at playback. The second order is not that clean -
    it is electrical at SP and made by the tape at EP - and the table has to
    say so rather than report the tidier SP result alone."""
    measured = tape_bias.MEASURED
    for playback, record in ((measured["product_playback_tap_db"],
                              measured["product_record_tap_db"]),
                             (measured["product_playback_tap_db_ep"],
                              measured["product_record_tap_db_ep"])):
        assert record < 1.5
        assert playback > 5.0
    second = measured["second_order_db"]
    # SP: at the record tap already and not grown. EP: the other way round.
    assert second["SP"]["record"] > second["SP"]["playback"]
    assert second["EP"]["record"] < second["EP"]["playback"]
    assert "NOT a clean" in second["what"]


def test_the_product_reproduces_at_both_tape_speeds():
    measured = tape_bias.MEASURED
    assert measured["product_y_only_playback_db"] < 1.0
    assert measured["product_y_only_playback_db_ep"] < 1.0
    assert measured["chroma_gain_slope_db_per_mhz"]["EP"]["tape"] > 1.0
    assert measured["chroma_gain_slope_db_per_mhz"]["SP"]["tape"] > 1.0


def test_the_delivered_picture_figures_beat_their_controls_where_chroma_is():
    delivered = tape_bias.MEASURED["delivered_picture_ire"]
    strong = delivered["chromanoise_SP"]
    assert strong["alternating"] / strong["control_alternating"] > 50.0
    # and the still part equals its own control, which is the in-band check
    assert strong["still"] == pytest.approx(strong["control_still"], rel=0.05)
    quiet = delivered["home"]
    assert quiet["alternating"] / quiet["control_alternating"] < 2.0


def test_the_products_order_is_refused_with_both_attempts_recorded():
    """The measurement that would settle the mechanism without appealing to
    the guide, refused with the reason and the datum that would allow it."""
    out = tape_bias.product_order_refused()
    assert "chroma amplitude" in out["refused"]
    assert out["spectral_attempt"]
    assert out["alternation_attempt"]
    assert "staircase" in out["would_settle_it"]


def test_the_interleave_offset_is_derived_and_is_not_a_half_everywhere():
    """525-line VHS alternates; 625-line does not, and the difference falls
    out of the carriers rather than being typed in.

    The 625-line colour under is 40.125 f_H, so the product at twice it is
    80.25 f_H - a QUARTER of a line rate off, a four-line cycle. Reporting a
    half there would make the estimator's alternating projection look like a
    universal law when it is one system's arithmetic.
    """
    ntsc = tape_bias.interleave("NTSC")
    assert ntsc["product_offset_lines"] == pytest.approx(0.5, abs=1e-6)
    assert ntsc["alternates_line_to_line"] is True
    assert ntsc["lines_in_the_cycle"] == 2
    pal = tape_bias.interleave("PAL")
    assert abs(pal["product_offset_lines"]) == pytest.approx(0.25, abs=1e-3)
    assert pal["alternates_line_to_line"] is False
    assert pal["lines_in_the_cycle"] == 4
    assert "NOT verified" in pal["verified_for"]


def test_the_rotation_is_a_parameter_not_a_constant():
    """With no rotation at all the 525-line product would sit ON the
    luminance's own harmonic, which is the case the format avoids."""
    plain = tape_bias.interleave("NTSC", rotation_per_line_degrees=0.0)
    assert plain["product_offset_lines"] == pytest.approx(0.0, abs=1e-6)
    assert plain["alternates_line_to_line"] is False


def test_a_compressive_playback_amplifier_is_excluded():
    """The one alternative to a record-time coupling, and the factor of two
    that rules it out.

    A memoryless compression moves a small signal's gain exactly twice as
    far as it moves the large signal's own, so the FM's own envelope would
    have to have moved half as far as the chroma did, in the same direction.
    It moved the other way, by a sixteenth.
    """
    out = tape_bias.playback_compression_bounded()
    assert out["small_signal_over_large_signal"] == 2.0
    assert out["required_fm_slope_db_per_mhz"] == pytest.approx(1.042,
                                                                abs=1e-3)
    assert out["wrong_sign"] is True
    assert out["sigma_away"] > 10.0
    assert out["excluded"] is True
    # and it is honest about what the argument does not reach
    assert "MEMORY" in out["not_bounded"]


def test_the_exclusion_uses_the_measurements_it_claims_to():
    """The bound must be computed from the same table the finding rests on,
    not from numbers typed beside it."""
    measured = tape_bias.MEASURED
    out = tape_bias.playback_compression_bounded(
        measured["chroma_gain_slope_db_per_mhz"]["SP"]["tape"],
        measured["fm_envelope_slope_db_per_mhz"]["SP"]["tape"],
        measured["fm_envelope_slope_db_per_mhz"]["SP"]["tape_se"])
    assert out["excluded"] is True
    ep = tape_bias.playback_compression_bounded(
        measured["chroma_gain_slope_db_per_mhz"]["EP"]["tape"],
        measured["fm_envelope_slope_db_per_mhz"]["EP"]["tape"],
        measured["fm_envelope_slope_db_per_mhz"]["EP"]["tape_se"])
    assert ep["excluded"] is True
