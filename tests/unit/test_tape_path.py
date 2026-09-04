"""The tape path as a mechanism, and the collapse it produces.

Every mechanical parameter of the path reaches the response through the
head-to-tape separation, and the separation reaches it only through Wallace's
law, so the seven mechanisms are seven values of one number as far as a
frequency response is concerned. These tests hold that negative result in
place, hold the two things that DO separate - a strain is a delay where a
clearance is a gain, and a rate is a rate - and hold the controls that would
catch the construction going wrong.
"""

import numpy as np
import pytest

from vhsdecode.models import interference as inf
from vhsdecode.models import tape_path as tp


BAND = np.linspace(1e6, 7e6, 1024)


# --------------------------------------------------------------------------
# The controls come first, because nothing below means anything without them
# --------------------------------------------------------------------------


def test_the_separability_control_passes_in_both_directions():
    """A control that cannot fail is not a control. This one runs the same
    participation-ratio code over a family that is one direction by algebra
    and a family that is a Fourier basis by construction, and their answers
    are known independently and are different from one another - so a
    measure that always collapses fails the second and one that manufactures
    directions out of rounding fails the first."""
    result = tp.separability_control()
    assert result["passes"]
    assert result["collinear_effective"] == pytest.approx(1.0, abs=0.01)
    assert result["orthogonal_effective"] > 4.5


def test_the_control_would_catch_a_manufactured_direction():
    """The failure `magnetic.py` records, reproduced exactly rather than
    described: write the recording depth as a FRACTION of the
    self-demagnetisation cap and `2 pi d / lambda` becomes independent of
    frequency, so the response is a constant, its log shape is nothing but
    rounding, and normalising that near-zero vector manufactures directions
    out of the last bits. A control that should have read 1.00 read 4.97.

    The check is that the reproduction still manufactures them - so the
    control is known to be capable of catching something - and that the
    collinear half of the control, run on the correct physical construction,
    reads 1.00 instead."""
    grid = np.linspace(1e6, 7e6, 256)
    wavelength = 5.80 / grid
    rows = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        # the error itself: a depth expressed as a fraction of the cap
        depth = wavelength / (2.0 * np.pi) * fraction
        x = 2.0 * np.pi * depth / wavelength          # constant, to rounding
        response = (1.0 - np.exp(-x)) / x
        logged = np.log(response)
        logged = logged - logged.mean()
        rows.append(logged / max(float(np.linalg.norm(logged)), 1e-300))
    singular = np.linalg.svd(np.array(rows), compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-300)
    manufactured = float(1.0 / np.sum(share ** 2))
    assert manufactured > 2.0        # rounding read as several directions

    # the same measure on the correct construction reads exactly one
    honest = tp.separability_control()
    assert honest["collinear_effective"] == pytest.approx(1.0, abs=0.01)


def test_the_entrainment_exponent_is_two_thirds():
    """The foil bearing's nominal clearance is 0.643 R (6 eta v / T')^(2/3) -
    Baumeister 1963, Eshel and Elrod 1965 - so the film goes as the
    two-thirds power of velocity and the minus two-thirds power of tension
    per unit width. Recovered numerically so a mistyped power is caught."""
    result = tp.entrainment_exponent_control()
    assert result["passes"]
    assert result["velocity_exponent"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert result["tension_exponent"] == pytest.approx(-2.0 / 3.0, abs=1e-9)


def test_the_contact_pressure_identity_holds():
    """`p R w = T` is statics, not an approximation. It catches a width or a
    radius entering the wrong way round, which would rescale every clearance
    in the budget without changing any shape and so would never show in a
    direction count."""
    result = tp.pressure_control()
    assert result["passes"]
    assert result["worst_absolute_error_n"] < 1e-12


# --------------------------------------------------------------------------
# The collapse: the result this component exists to establish
# --------------------------------------------------------------------------


def test_the_amplitude_side_is_exactly_one_direction():
    """Every mechanism acts through the separation and the separation acts
    through `-2 pi d f / v`, which is one vector at seven scales. Not
    approximately one direction: exactly one, and every pair of members is
    unit coherent."""
    result = tp.frequency_axis(BAND, phase=False)
    assert result["count"] == 7
    assert result["rank"] == 1
    assert result["effective"] == pytest.approx(1.0, abs=1e-9)
    assert result["worst_pair_coherence"] == pytest.approx(1.0, abs=1e-9)


def test_the_tape_path_is_worse_than_the_magnetics_it_joins():
    """The standard to measure against is the tape's six magnetic loss
    mechanisms at 1.58 effectively distinguishable of 6, and the comparison
    has to be like for like: that figure is amplitude only, so the path's
    amplitude-only figure is what stands beside it. The magnetics are
    recomputed here through the same function rather than quoted, so the
    comparison is a measurement and not a citation.

    The path loses, and by the whole of the margin. The magnetics at least
    contain sinc and exponential-integral forms; the path contains one
    exponential."""
    x = np.geomspace(0.01, 2.0, 512)
    magnetics = [
        np.exp(-2 * np.pi * x),                                # spacing
        np.abs(np.sinc(x)),                                    # gap
        np.where(x > 1e-9, (1 - np.exp(-2 * np.pi * x))
                 / np.maximum(2 * np.pi * x, 1e-12), 1.0),     # thickness
        np.abs(np.sinc(1.3 * x)),                              # azimuth
        1 + 0.02 * np.cos(2 * np.pi * x) * np.exp(-0.15 * x),  # contour
    ]
    losses = tp.participation([np.log(np.maximum(value, 1e-12))
                               - np.log(np.maximum(value, 1e-12)).mean()
                               for value in magnetics])
    path = tp.frequency_axis(BAND, phase=False)
    assert losses["effective"] > 1.5          # the 1.58 of 6 standard
    assert path["effective"] < losses["effective"]
    assert path["effective"] == pytest.approx(1.0, abs=1e-9)


def test_carrying_the_phase_finds_the_second_kind():
    """Four of the seven strain the tape and three do not, and a strain is a
    delay where a clearance is a gain. Under the Hermitian inner product a
    gain and a delay of the same shape are one direction; with real and
    imaginary parts stacked they are two, which is Proposition 9.3 and the
    reason `real_parameters` exists."""
    amplitude = tp.frequency_axis(BAND, phase=False)
    both = tp.frequency_axis(BAND, phase=True)
    assert amplitude["rank"] == 1
    assert both["rank"] == 2
    assert both["effective"] > amplitude["effective"] + 0.5


def test_the_second_direction_survives_every_plausible_modulus():
    """The base film's modulus is assumed, and it sets how large the phase
    part is beside the amplitude part, so it is the one assumption that could
    move a direction count. Over a factor of sixteen the count moves and the
    RANK does not, which is the honest form of the claim."""
    result = tp.modulus_sensitivity(BAND)
    assert result["always_two_ranked"]
    assert result["lowest"] > 1.0
    assert result["highest"] < 2.0


# --------------------------------------------------------------------------
# What does separate, and it is the time axis
# --------------------------------------------------------------------------


def test_the_time_axis_separates_what_the_frequency_axis_cannot():
    """A parameter that varies at a known rate is separable from one that
    does not even when their frequency shapes are identical. The rates are
    stated by `transport_model` from the parts' diameters before anything is
    measured, which is what makes this a prediction rather than a search."""
    frequency = tp.frequency_axis(BAND, phase=True)
    time = tp.time_axis()
    assert time["effective"] > 2.5 * frequency["effective"]
    assert time["rank"] > frequency["rank"]


def test_the_static_mechanisms_are_inert_on_the_time_axis():
    """Nothing that does not move can be told from anything else that does
    not move. The wrap and the penetration are set when the deck is threaded,
    and the entrained film is fixed by a drum speed the format fixes at
    5.80 m/s whatever the tape speed, so both rows are exactly zero and the
    honest count is over five live rows of seven."""
    time = tp.time_axis()
    assert set(time["inert"]) == {"wrap angle and penetration",
                                  "air entrainment"}
    assert time["alive"] == 5
    series = tp.time_signatures()
    for name in time["inert"]:
        assert np.all(series[name] == 0.0)


def test_the_two_monotone_mechanisms_split_only_by_kind():
    """The pack radius and head wear are both monotone trends through a pass,
    so their shapes are identical and nothing on the time axis alone can tell
    them apart. They separate at all only because the pack strains the tape
    and wear does not - the same distinction of kind the frequency axis
    found - and that is the whole of the difference between 4.60 and 3.56.

    3.56 and not 4.00, because the participation ratio is not the rank: with
    the strain suppressed there are four directions but one of them carries
    two of the five live rows, so its share is doubled and the ratio reads
    `1 / (0.4^2 + 3 x 0.2^2) = 3.57`. The rank is what says how many
    directions exist; the participation ratio says how evenly they are
    held."""
    series = tp.time_signatures()
    trend = series["pack radius"]
    assert np.allclose(series["head wear"], trend)
    with_kind = tp.time_axis()
    # with the strain suppressed the two trends become one direction
    without_kind = tp.time_axis(tension_amplitude_n=0.0)
    assert without_kind["rank"] == 4
    assert without_kind["effective"] == pytest.approx(1.0 / (0.16 + 3 * 0.04),
                                                      abs=0.05)
    assert with_kind["rank"] == 5
    assert with_kind["effective"] > without_kind["effective"] + 0.4


def test_the_joint_grid_buys_nothing_over_time_alone():
    """The frequency factor is proportional to `f` for both the clearance and
    the delay, so it is common to all seven, and a shared factor separates
    nothing. The joint count must therefore equal the time count exactly; a
    joint count that exceeded it would mean a direction had been
    manufactured from the grid rather than found in the physics."""
    time = tp.time_axis()
    joint = tp.joint_axis(np.linspace(1e6, 7e6, 64))
    assert joint["effective"] == pytest.approx(time["effective"], abs=1e-9)
    assert joint["rank"] == time["rank"]


# --------------------------------------------------------------------------
# The contract on every key entry
# --------------------------------------------------------------------------


def test_every_signature_is_subtractable():
    """The transform works in the log domain, so an entry whose magnitude
    reaches zero cannot be subtracted. A separation loss is an exponential
    and is strictly positive at every finite clearance, so the bounded form
    here is the exponential itself - no mask and no bare power term."""
    for name, value in tp.signatures(BAND).items():
        assert inf.subtractable(value), name
    for name, value in tp.all_member_signatures(BAND).items():
        assert inf.subtractable(value), name


def test_every_signature_is_complex_with_both_parts_meant():
    """The real part is what the mechanism does to amplitude and the
    imaginary part what it does to phase. A clearance has no phase and a
    strain no amplitude, and both statements have to be true in the returned
    arrays rather than only in the prose."""
    entries = tp.signatures(BAND)
    clearance = entries["tape path clearance (playback)"]
    strain = entries["tape path strain (record)"]
    assert np.iscomplexobj(clearance) and np.iscomplexobj(strain)
    assert np.allclose(np.angle(clearance), 0.0)
    assert not np.allclose(np.abs(clearance), np.abs(clearance)[0])
    assert np.allclose(np.abs(strain), 1.0)
    assert not np.allclose(np.angle(strain), 0.0)


def test_every_entry_has_a_declared_position_and_the_stages_differ():
    """A chain can only be inverted in the reverse of the order it was
    applied. The path acts at two stages - once when the signal is written
    and again when it is read - and a record-side parameter and a
    playback-side one are not the same entry."""
    positions = tp.COMPONENT_POSITIONS
    for name in tp.signatures(BAND):
        assert name in positions, name
    assert positions["tape path clearance (record)"] < \
        positions["tape path clearance (playback)"]
    # the recorder's own stages come before, the head's losses after
    assert (max(positions.values())
            < inf.COMPONENT_ORDER["head contact tilt"])
    assert (min(positions.values())
            > inf.COMPONENT_ORDER["record level dependence"])


def test_the_entries_would_be_ordered_by_the_key_once_declared():
    """`interference.ordered_key` refuses an entry with no declared
    position, and it is the coordinator who adds these to
    `COMPONENT_ORDER`. This checks the pair is orderable once they are
    there, without editing that file."""
    order = dict(inf.COMPONENT_ORDER)
    order.update(tp.COMPONENT_POSITIONS)
    entries = tp.signatures(BAND)
    for name in entries:
        assert order.get(name) is not None
    ranked = sorted(entries, key=lambda name: -order[name])
    assert ranked[0] == "tape path clearance (playback)"


# --------------------------------------------------------------------------
# The key's own verdict on these entries, which is negative
# --------------------------------------------------------------------------


def test_neither_entry_earns_a_place_in_the_frequency_key():
    """An entry earns its place by the direction it adds, not by being a
    mechanism. Every arrangement of these lowers the modelled key's effective
    count, because the clearance is a straight line in frequency and the
    strain a linear phase ramp and the key already spans both. The honest
    recommendation is that the key take neither."""
    result = tp.key_impact(BAND)
    baseline = result["baseline"]["effective"]
    assert baseline > 7.5
    for name, verdict in result["earns_a_place"].items():
        assert verdict is False, name
    assert result["with all seven members"]["effective"] < baseline - 2.0


def test_the_clearance_is_a_second_copy_of_something_already_modelled():
    """Naming which direction it duplicates is what makes the negative result
    actionable: the clearance's log magnitude is a straight line in frequency
    and the particle-noise entry's own law is a power of frequency, so the
    two are very nearly the same shape over this band."""
    def shape(value):
        value = np.asarray(value)
        magnitude = np.log(np.maximum(np.abs(value), 1e-12))
        phase = np.unwrap(np.angle(value))
        vector = np.concatenate([magnitude - magnitude.mean(),
                                 phase - phase.mean()])
        return vector / np.linalg.norm(vector)

    modelled = inf.signatures(BAND)
    mine = tp.signatures(BAND)
    clearance = shape(mine["tape path clearance (playback)"])
    strain = shape(mine["tape path strain (record)"])
    assert abs(float(clearance @ shape(modelled["particle noise"]))) > 0.9
    assert abs(float(strain @ shape(modelled["group delay"]))) > 0.9


# --------------------------------------------------------------------------
# The physics, checked against the specification it came from
# --------------------------------------------------------------------------


def test_the_specified_tension_range_sets_the_clearance_budget():
    """JVC VTG82063 gives the tension as 0.30 to 0.45 newtons. Across that
    range the clearance falls, because more tension is more contact pressure
    and a thinner air film at once, and both of the two largest terms move
    the same way."""
    slack = tp.spacing_budget(tp.TENSION_RANGE_N[0])
    tight = tp.spacing_budget(tp.TENSION_RANGE_N[1])
    assert tight["total_m"] < slack["total_m"]
    assert tight["air entrainment"] < slack["air entrainment"]
    assert tight["tape tension"] < slack["tape tension"]
    # and the whole budget is a fraction of a micrometre, as a head requires
    assert 0.05e-6 < slack["total_m"] < 0.5e-6


def test_the_head_must_protrude_to_reach_the_tape_at_all():
    """The air film over the DRUM is fifteen micrometres at the specified
    tension and the format's writing speed - a thousand times more separation
    than a video head can tolerate. That is why a head protrudes from the
    drum surface: it exists to pierce the film, and the clearance that
    matters is the one over the protruding tip."""
    over_drum = tp.entrainment_spacing_m(5.80, 0.375,
                                         radius_m=tp.DRUM_RADIUS_M)
    over_tip = tp.entrainment_spacing_m(5.80, 0.375)
    assert over_drum > 10e-6
    assert over_tip < 0.2e-6
    assert over_drum / over_tip == pytest.approx(
        tp.DRUM_RADIUS_M / tp.HEAD_TIP_RADIUS_M, rel=1e-9)


def test_a_constant_torque_brake_cannot_meet_the_specification():
    """Set so the middle of the pack sits in the middle of the specified
    range, a constant-torque holdback delivers 1.03 newtons at the hub -
    two and a half times the specified maximum. A transport meeting the
    specification at both ends of a tape therefore regulates the tension
    rather than leaving it to the brake, which is what the tension arm is
    for, and the pack radius enters as a residual trend inside the regulated
    range rather than as the full reciprocal law."""
    unregulated = tp.tension_from_pack([0.0, 0.5, 1.0], regulated=False)
    assert unregulated[-1] > 2.0 * tp.TENSION_RANGE_N[1]
    regulated = tp.tension_from_pack([0.0, 0.5, 1.0])
    assert np.all(regulated >= tp.TENSION_RANGE_N[0])
    assert np.all(regulated <= tp.TENSION_RANGE_N[1])


def test_the_pack_empties_by_constant_volume():
    """The wound cross-section is proportional to `r^2 - r_hub^2`, so a pack
    emptying linearly in time has `r(x) = sqrt(r_hub^2 + (r_out^2 -
    r_hub^2)(1 - x))`. It starts at the outer radius and ends at the hub."""
    radius = tp.pack_radius_m([0.0, 0.5, 1.0])
    assert radius[0] == pytest.approx(tp.REEL_OUTER_RADIUS_M)
    assert radius[-1] == pytest.approx(tp.REEL_HUB_RADIUS_M)
    assert radius[0] > radius[1] > radius[-1]


def test_the_strain_is_a_time_base_error_of_the_right_order():
    """Hooke's law on the tape's cross-section. The full width of the
    specified tension range is ten nanoseconds over one NTSC line - well
    inside what a time base corrector handles, and well outside what can be
    assumed to be zero."""
    span = tp.TENSION_RANGE_N[1] - tp.TENSION_RANGE_N[0]
    delay = tp.strain_delay_s(span)
    assert 1e-9 < delay < 100e-9
    # linear in the tension difference, because Hooke's law is
    assert tp.strain_delay_s(2 * span) == pytest.approx(2 * delay)


def test_the_bending_length_is_the_scale_that_lets_a_guide_reach_the_head():
    """A tensioned plate carries a bend over `sqrt(D w / T)`, which at the
    specified tension is under half a millimetre. A guide nearer than that
    still has its deviation standing when the tape reaches the gap; one a
    centimetre away does not."""
    length = tp.bending_length_m(0.375)
    assert 0.1e-3 < length < 1e-3
    near = tp.stiffness_spacing_m(0.5e-6, 0.2e-3, 0.375)
    far = tp.stiffness_spacing_m(0.5e-6, 10e-3, 0.375)
    assert near > 10.0 * far
    # a slacker tape carries the bend further
    assert tp.bending_length_m(0.30) > tp.bending_length_m(0.45)


def test_every_mechanism_declares_its_rate_and_its_confounds():
    """A parameter that changes nothing observable cannot be fitted, and a
    parameter two measurements see identically cannot be separated from its
    twin. Both cases are declared up front here, as `tape_model` does, so a
    fitted number cannot be reported as physics."""
    names = {str(entry["name"]) for entry in tp.mechanisms()}
    assert len(names) == 7
    for entry in tp.mechanisms():
        assert "law" in entry and "spec" in entry
        assert entry["rate"] in (None, "trend", "parity", "supply reel",
                                 "entry guide")
        for other in entry["confounded_with"]:
            assert other in names, other
