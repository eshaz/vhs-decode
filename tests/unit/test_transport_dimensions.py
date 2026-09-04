"""The transport's rates as time-axis dimensions.

The measurement that motivates the module: the tape's six magnetic loss
mechanisms are 1.58 effectively distinguishable of 6 because every one is a
function of one dimensionless group, so they differ only in a rate of decay.
The transport's eight parts differ in KIND - in rate on an axis that resolves
rates, and in the quantity they disturb - and read 6.39 of 8 at a condition
of 1.45 over the directions they span.

The established pairwise result is reproduced exactly and then extended: the
axis a part acts on separates pairs no rate difference reaches, and the
saturation at 26 of 28 belongs to holding the reel rate fixed rather than to
the transport.
"""

from unittest import mock

import numpy as np
import pytest

from vhsdecode.models import interference as inf
from vhsdecode.models import transport_dimensions as td
from vhsdecode.models import transport_model


# --------------------------------------------------------------------------
# the hard contracts
# --------------------------------------------------------------------------


def test_every_signature_is_complex_and_subtractable():
    """The transform works in the log domain, so an entry whose magnitude
    reaches zero cannot be subtracted off. A bare modulation `d*cos(wt)`
    crosses zero twice a cycle; the bounded exponential this module returns
    does not, and its magnitude stays within `exp(+/- depth)`."""
    entries = td.signatures(5.0, 0.0)
    assert len(entries) == 8
    for name, value in entries.items():
        assert np.iscomplexobj(value), name
        assert inf.subtractable(value), name
    magnitudes = np.concatenate([np.abs(v) for v in entries.values()])
    assert magnitudes.min() == pytest.approx(np.exp(-td.MODULATION_DEPTH),
                                             rel=1e-3)
    assert magnitudes.max() == pytest.approx(np.exp(td.MODULATION_DEPTH),
                                             rel=1e-3)


def test_the_real_part_is_the_envelope_and_the_imaginary_part_the_time_base():
    """A tension disturbance moves the RF envelope through the head-to-tape
    spacing and leaves the time base alone; a speed disturbance moves the
    time base and barely touches the envelope. The two land on the two parts
    of the complex time-axis residual, which is what makes them orthogonal
    before any rate is considered."""
    t, logs = td.log_signatures(5.0, 0.0)
    by_name = {str(p["name"]): str(p["acts_on"]) for p in td.parts(0.0)}
    for name, value in logs.items():
        acts = by_name[name.split(") ")[-1]]
        if acts == "tension":
            assert np.allclose(value.imag, 0.0)
            assert np.any(np.abs(value.real) > 0)
        elif acts == "speed":
            assert np.allclose(value.real, 0.0)
            assert np.any(np.abs(value.imag) > 0)
        else:                                    # the drum, which does both
            assert np.any(np.abs(value.real) > 0)
            assert np.any(np.abs(value.imag) > 0)


def test_every_entry_has_a_declared_position_in_the_chain():
    """A chain can only be inverted in the reverse of the order it was
    applied. The eight parts exist on both machines, and the two sides are
    at different positions: the recording deck's transport disturbed the tape
    before the medium held it, the playback deck's after."""
    entries = dict(td.signatures(5.0, machine="playback"))
    entries.update(td.signatures(5.0, machine="record"))
    with pytest.raises(ValueError):
        inf.ordered_key(entries)                 # undeclared, as it stands
    with mock.patch.dict(inf.COMPONENT_ORDER, td.CHAIN_POSITIONS):
        order = inf.ordered_key(entries)
        assert len(order) == 16
        # last applied, first removed: playback before record
        assert order[0][0] == 24 and order[-1][0] == 12
        assert all(name.startswith("transport (playback)")
                   for place, name in order if place == 24)


def test_the_signature_logarithm_round_trips():
    """`distinguishable` works on the logarithm rather than recovering it
    through `log|s| + j arg s`, because a timing modulation beyond pi radians
    would wrap. At the shipped depth it does not, and the two agree."""
    entries = td.signatures(5.0, 0.0)
    _, logs = td.log_signatures(5.0, 0.0)
    for name, value in entries.items():
        recovered = np.log(np.abs(value)) + 1j * np.angle(value)
        assert np.allclose(recovered, logs[name], atol=1e-12)


# --------------------------------------------------------------------------
# the controls
# --------------------------------------------------------------------------


def test_the_resolution_control_reads_what_fourier_resolution_says():
    """THE CONTROL. Two parts on the same quantity whose rates differ by far
    less than the window resolves are ONE direction, and two well beyond it
    are TWO. Neither number comes from this module - both are Rayleigh's
    criterion - and a construction that manufactures separation out of
    leakage, a mean left in, or a per-part phase fails the first."""
    control = td.resolution_control()
    assert control["passes"]
    assert control["unresolved"] == pytest.approx(1.0, abs=0.01)
    assert control["resolved"] == pytest.approx(2.0, abs=0.01)


def test_the_resolution_control_can_fail():
    """A control that cannot fail is not a control. Give the two parts
    independent phases - which is what an unthinking construction does, since
    every real disturbance has an arbitrary phase - and two parts at
    effectively one rate read as two directions, when two sinusoids at one
    rate sum to a single sinusoid and no measurement could attribute them."""
    control = td.resolution_control()
    assert control["phase_convention_matters"]
    assert control["unresolved_with_independent_phases"] > 1.9
    # and the same defect on the whole set, where it inflates the answer
    rng = np.random.default_rng(7)
    phases = {name: float(2 * np.pi * rng.random())
              for name in td.part_names()}
    honest = td.distinguishable(5.0, 0.0, fixed_rate=True)
    inflated = td.distinguishable(5.0, 0.0, fixed_rate=True, phases=phases)
    assert honest["rank"] == 6 and inflated["rank"] == 8
    assert inflated["effective"] > honest["effective"] + 0.9


def test_the_winding_control_checks_the_pack_law_against_the_specification():
    """THE SECOND CONTROL, on the geometry rather than on the signatures.
    One revolution adds one tape thickness to the pack radius, which is known
    without the integral that produced the closed form; the pack the hub,
    thickness and length wind must fit inside the specified flange; and two
    hours at the specified speed must be about as long as the specified
    tape."""
    control = td.winding_control()
    assert control["passes"]
    assert control["turns_departure"] < 1e-9
    assert control["full_pack_radius_m"] == pytest.approx(0.04070, abs=1e-4)
    assert control["full_pack_radius_m"] < td.FLANGE_RADIUS_M
    assert control["length_departure"] < 0.03


def test_the_revolutions_are_one_per_tape_thickness():
    """The identity the winding control rests on, taken directly: winding a
    pack from one radius to another takes exactly as many turns as the radius
    change divided by the tape thickness."""
    for position in (100.0, 1000.0, 5000.0):
        turns = td.reel_revolutions(position, "take-up")
        growth = (td.pack_radius_m(position, "take-up")
                  - td.pack_radius_m(0.0, "take-up"))
        assert turns == pytest.approx(growth / td.TAPE_THICKNESS_M,
                                      rel=1e-12)


# --------------------------------------------------------------------------
# the established result, confirmed
# --------------------------------------------------------------------------


def test_the_published_pair_table_is_reproduced_exactly():
    """`docs/ELLIPTICAL_COLLAPSE.md` section 7.3 and
    `docs/BROADCAST_VTR_COLLAPSE.md` section 5: 9 of 28 pairs at 0.23 s,
    13 at 0.5 s, 17 at 1.0 s, 25 at 2.0 s, 26 at 5.0 s, and 26 still at
    120 s. Confirmed, under the criterion those tables used - Rayleigh's on
    the nominal rates with both reels at the middle of their band."""
    expected = {0.23: 9, 0.5: 13, 1.0: 17, 2.0: 25, 5.0: 26, 120.0: 26}
    for duration, count in expected.items():
        result = td.rayleigh_pairs(duration)
        assert result["total"] == 28
        assert result["separable"] == count, duration


def test_the_two_unresolved_pairs_are_the_guides_and_the_reels():
    """And they are unresolved for different reasons, which is the whole of
    what follows: the guides share a diameter, the reels share only the
    middle of a band."""
    left = td.rayleigh_pairs(5.0)["unresolved"]
    assert {tuple(sorted((a, b))) for a, b, _ in left} == {
        ("entry guide", "exit guide"),
        ("supply reel", "take-up reel"),
    }


def test_the_rates_are_transport_model_s_own():
    """Nothing here re-derives a rate that model already predicts. The seven
    non-reel parts are carried through unchanged, so any correction to those
    diameters propagates rather than being duplicated."""
    theirs = {str(e["name"]): float(e["rate_hz"])
              for e in transport_model.rotation_rates(
                  td.LINEAR_TAPE_SPEED_M_S)}
    for entry in td.parts(0.0):
        if entry["kind"] != "reel":
            assert float(entry["rate_hz"]) == theirs[str(entry["name"])]
    assert theirs["head drum"] == pytest.approx(29.97, abs=1e-6)


# --------------------------------------------------------------------------
# and extended
# --------------------------------------------------------------------------


def test_the_axis_separates_pairs_that_no_rate_difference_reaches():
    """Ten of the twenty-eight pairs put a tension part against a speed part,
    and those are orthogonal in the shortest window there is because one
    modulates the envelope and the other the time base. At a quarter of a
    second: 9 separable by rate, 10 by axis, 2 counted in both, and 17
    measured - the decomposition is exact."""
    acts = {str(p["name"]): str(p["acts_on"]) for p in td.parts(0.0)}
    rates = {str(p["name"]): float(p["rate_hz"]) for p in td.parts(0.0)}
    names = list(acts)
    by_rate = by_axis = both = 0
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            rate = abs(rates[first] - rates[second]) > 1.0 / 0.23
            axis = {acts[first], acts[second]} == {"tension", "speed"}
            by_rate += rate
            by_axis += axis
            both += rate and axis
    assert (by_rate, by_axis, both) == (9, 10, 2)
    measured = td.separable_pairs(0.23, 0.0)["separable"]
    assert measured == by_rate + by_axis - both
    assert measured == 17


def test_the_transport_separates_where_the_magnetics_do_not():
    """The comparison the whole arc is measured on. Tape magnetics: 1.58
    effectively distinguishable of 6 at condition 1.06e4, because every loss
    is a monotone decay in one dimensionless group. The transport: 6.39 of 8
    at condition 1.45 over the directions it spans, because its parts differ
    in kind."""
    fit = td.distinguishable(5.0, 0.0)
    assert fit["count"] == 8
    assert fit["effective"] == pytest.approx(6.391, abs=0.01)
    assert fit["rank"] == 7
    assert fit["condition_of_the_rank"] == pytest.approx(1.453, abs=0.01)
    # six of the eight singular values within five per cent of one another,
    # which is what a nearly orthogonal family looks like
    middle = np.sort(fit["singular_values"])[::-1][1:7]
    assert middle.max() / middle.min() < 1.05
    # far better than the magnetics on both measures
    assert fit["effective"] / fit["count"] > 1.58 / 6.0
    assert fit["condition_of_the_rank"] < 1.06e4


def test_the_guides_are_degenerate_and_no_amount_of_tape_fixes_it():
    """They are declared with the same diameter and both act on tension, so
    their signatures are identical at every duration. This is the pair that
    needs a different witness - the `after_capstan` and `order` distinctions
    the model already carries - rather than more tape."""
    for duration in (0.23, 5.0, 120.0):
        fit = td.distinguishable(duration, 0.0)
        assert fit["worst_coherence"] == pytest.approx(1.0, abs=1e-6)
        assert set(name.split(") ")[-1] for name in fit["worst_pair"]) == {
            "entry guide", "exit guide"}
        assert fit["rank"] <= 7


def test_the_reels_are_degenerate_only_at_the_tape_midpoint():
    """They share a nominal BAND, not a rate. The supply pack unwinds and its
    rate rises; the take-up pack winds on and its rate falls; so the two are
    equal at exactly one place on the tape and far apart everywhere else."""
    middle = td.pack_midpoint_s()
    at_middle = td.reel_pair(middle, 5.0)
    assert at_middle["difference_hz"] == pytest.approx(0.0, abs=1e-9)
    assert at_middle["coherence"] == pytest.approx(1.0, abs=1e-6)
    assert not at_middle["separable"]

    at_head = td.reel_pair(0.0, 5.0)
    assert at_head["supply_hz"] == pytest.approx(0.1304, abs=1e-3)
    assert at_head["take_up_hz"] == pytest.approx(0.4083, abs=1e-3)
    assert at_head["separable"]

    outside = [row for row in td.across_the_tape(duration_s=5.0)
               if abs(row["position_s"] - middle) > 1500.0]
    assert all(row["separable"] for row in outside)
    assert all(not row["beyond_the_tape"] for row in outside)


def test_a_rate_that_moves_separates_where_a_fixed_one_never_does():
    """The useful part. A pair at a fixed common rate never separates however
    long the observation - the signatures are identical and their coherence
    stays at one. A pair whose rates drift apart separates after a time the
    closed form gives, `T > sqrt(1 / (2 |df/dt|))`, which at the T-120's
    midpoint is 160.5 s. So a moving rate is an EASIER identification problem
    than a fixed one, not a harder one."""
    middle = td.pack_midpoint_s()
    predicted = td.reel_separation_s(middle)
    assert predicted["seconds"] == pytest.approx(160.5, abs=1.0)

    # fixed: no window ever separates them
    for duration in (5.0, 200.0, 800.0):
        fixed = td.distinguishable(duration, middle, fixed_rate=True,
                                   sample_rate_hz=240.0)
        assert fixed["rank"] == 6
    # drifting: separated by the time the closed form names, not before
    assert td.reel_pair(middle, 60.0)["coherence"] > 0.9
    assert td.reel_pair(middle, 2.0 * predicted["seconds"])["coherence"] < 0.5
    late = td.distinguishable(400.0, middle, sample_rate_hz=240.0)
    early = td.distinguishable(5.0, middle, sample_rate_hz=240.0)
    assert early["effective"] == pytest.approx(5.32, abs=0.05)
    assert late["effective"] == pytest.approx(6.37, abs=0.05)


def test_saturation_at_26_belongs_to_the_fixed_rate_approximation():
    """The correction to the established result. Holding both reels at the
    middle of their band saturates at 26 of 28 and never improves; with the
    pack law the reels separate too and the count reaches 27 of 28, which is
    where it really saturates. The pair still left is the guides."""
    for duration in (5.0, 120.0):
        assert td.separable_pairs(duration, 0.0,
                                  fixed_rate=True)["separable"] == 26
        assert td.separable_pairs(duration, 0.0)["separable"] == 27
    fixed = td.distinguishable(5.0, 0.0, fixed_rate=True)
    moving = td.distinguishable(5.0, 0.0)
    assert fixed["effective"] == pytest.approx(5.323, abs=0.01)
    assert moving["effective"] == pytest.approx(6.391, abs=0.01)


def test_how_much_tape_a_decode_needs():
    """The practical table. Every decode this arc has examined is about
    0.22 s, where nothing mechanical is resolved by rate at all; two seconds
    reaches 27 of 28 pairs and 6.04 of 8; five seconds is where the effective
    count settles, and beyond it nothing at the head of the tape changes."""
    table = {row["duration_s"]: row
             for row in td.identifiability_vs_duration()}
    assert table[0.23]["separable_pairs"] == 17
    assert table[0.23]["effective"] == pytest.approx(2.235, abs=0.02)
    assert table[2.0]["separable_pairs"] == 27
    assert table[2.0]["effective"] == pytest.approx(6.041, abs=0.02)
    assert table[5.0]["effective"] == pytest.approx(6.391, abs=0.02)
    # monotone in the tape observed, and saturating
    counts = [table[d]["effective"] for d in sorted(table)]
    assert counts == sorted(counts)
    assert table[120.0]["effective"] - table[5.0]["effective"] < 0.02


# --------------------------------------------------------------------------
# nothing reported rests on an assumed constant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keyword, values", [
    ("depth", (0.005, 0.05, 0.5, 2.0)),
    ("drum_timing_ratio", (0.01, 1.0, 100.0)),
])
def test_no_reported_figure_depends_on_the_assumed_amplitudes(keyword, values):
    """Both are labelled ASSUMED at their constants because no specification
    states them. Neither moves an answer: every row is unit-normalised, so a
    depth is absorbed entirely, and the drum's internal balance never enters
    an overlap because its 29.97 Hz is resolved from every other part."""
    got = [td.distinguishable(5.0, 0.0, **{keyword: value})["effective"]
           for value in values]
    assert max(got) - min(got) < 1e-4


def test_the_sampling_of_the_time_axis_is_arithmetic_and_not_information():
    """The rates modelled here top out at 29.97 Hz while the decoder's own
    time-base residual is sampled once per line at 15.734 kHz, so the grid is
    an economy and not a limit."""
    got = [td.distinguishable(5.0, 0.0, sample_rate_hz=rate)["effective"]
           for rate in (4 * 29.97, 8 * 29.97, 32 * 29.97, 128 * 29.97)]
    assert max(got) - min(got) < 1e-3


def test_the_separability_threshold_does_not_choose_the_answer():
    """`coherence_limit` is the largest overlap two signatures may have and
    still be called separable. The count is flat across any defensible
    choice of it."""
    counts = {c: td.separable_pairs(5.0, 0.0, coherence_limit=c)["separable"]
              for c in (0.3, 0.4, 0.5, 0.6, 0.7)}
    assert set(counts.values()) == {27}
    short = {c: td.separable_pairs(0.23, 0.0, coherence_limit=c)["separable"]
             for c in (0.3, 0.4, 0.5, 0.6)}
    assert set(short.values()) == {17}


def test_the_chirp_reduces_to_the_exact_pack_law_over_a_short_window():
    """The reels' accumulated turns are taken from the closed-form pack law
    rather than as a linear chirp. Over a window short against the pack's own
    timescale the two must agree, and over a long one they must not - which
    is the check that the exact law is doing something."""
    grid = td.time_grid(5.0, 32.0)
    exact = td.cycles(grid, 0.1757, winding="take-up", position_s=3688.0)
    chirp = td.cycles(grid, td.reel_rate_hz(3688.0, "take-up"),
                      drift_hz_per_s=td.reel_drift_hz_per_s(3688.0,
                                                            "take-up"),
                      winding="take-up", position_s=3688.0, exact_pack=False)
    assert np.allclose(exact, chirp, atol=1e-9)
    long_grid = td.time_grid(3000.0, 4.0)
    exact_long = td.cycles(long_grid, 0.1757, winding="take-up",
                           position_s=3688.0)
    chirp_long = td.cycles(long_grid, td.reel_rate_hz(3688.0, "take-up"),
                           drift_hz_per_s=td.reel_drift_hz_per_s(3688.0,
                                                                 "take-up"),
                           winding="take-up", position_s=3688.0,
                           exact_pack=False)
    assert not np.allclose(exact_long, chirp_long, atol=1e-3)
