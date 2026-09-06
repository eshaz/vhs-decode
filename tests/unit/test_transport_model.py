"""The transport's rates as a SHAPE, and what the sampling does to it.

`transport_model` predicted eight mechanical rates and returned each as a
bare number. Its own `rotation_rates` docstring names three kinds of fault
and the orders each puts on a rate - once for an eccentricity, twice for an
out-of-round or two-lobe bearing, a train for a flat spot - and none of
them was ever built, so a search could only ever look at the fundamentals.
These tests cover the shape that was added and, with it, the one thing that
makes a shape honest: where the MEASUREMENT puts each order.

The case that forces the issue is the drum. The format fixes one revolution
per two fields, so in a series sampled once per FIELD the drum's rate is
exactly the Nyquist frequency, and `observable` used to accept it because
its comparison at the top end was `rate <= sample_rate/2`. It is strict now,
and `test_the_drum_is_refused_at_its_own_nyquist` plants the modulation and
shows why: sampled per field it is bit-identical to the head A/B
alternation.
"""

import numpy as np
import pytest

from vhsdecode.models import transport_model as tm
from vhsdecode.models import transport_dimensions as td


# Derived, never retyped: the field rate from the standard's own definition,
# and every mechanical rate from `rotation_rates`.
FIELD_RATE_HZ = 2.0 * 30000.0 / 1001.0
LINES_PER_FIELD = 262.5
LINE_RATE_HZ = FIELD_RATE_HZ * LINES_PER_FIELD


def rates():
    return tm.rotation_rates(td.LINEAR_TAPE_SPEED_M_S, FIELD_RATE_HZ)


def named(entries, name):
    return next(entry for entry in entries if entry["name"] == name)


# --------------------------------------------------------------------------
# the shape itself
# --------------------------------------------------------------------------


def test_the_fault_orders_are_the_three_the_docstring_names():
    """An eccentricity once per revolution, a two-lobe bearing twice, a flat
    spot a train. Nothing else is claimed, so nothing else may appear."""
    assert tm.FAULT_HARMONICS["eccentricity"] == (1,)
    assert tm.FAULT_HARMONICS["out of round"] == (2,)
    assert tm.FAULT_HARMONICS["two-lobe bearing"] == (2,)
    # a train has no end of its own, which is what None records
    assert tm.FAULT_HARMONICS["flat spot"] is None


def test_an_eccentricity_is_one_order_and_a_two_lobe_bearing_is_the_second():
    drum = named(rates(), "head drum")["rate_hz"]
    once = tm.harmonics(drum, "eccentricity")
    assert [entry["order"] for entry in once] == [1]
    assert once[0]["at_hz"] == pytest.approx(drum, rel=1e-12)

    twice = tm.harmonics(drum, "out of round")
    assert [entry["order"] for entry in twice] == [2]
    assert twice[0]["at_hz"] == pytest.approx(2.0 * drum, rel=1e-12)


def test_a_flat_spots_train_is_not_invented_out_of_nothing():
    """The train runs on until the measurement stops carrying it, so a
    length must come from the caller or from the sampling. Inventing one
    here would be a constant typed into the model."""
    with pytest.raises(ValueError, match="no end of its own"):
        tm.harmonics(named(rates(), "capstan")["rate_hz"], "flat spot")


def test_the_series_own_nyquist_ends_the_flat_spots_train():
    capstan = named(rates(), "capstan")["rate_hz"]
    train = tm.harmonics(capstan, "flat spot", sample_rate_hz=FIELD_RATE_HZ)
    top = max(entry["order"] for entry in train)
    assert top == int(np.floor(FIELD_RATE_HZ / 2.0 / capstan))
    # every order it kept is below the Nyquist, and one more would not be
    assert max(entry["at_hz"] for entry in train) < FIELD_RATE_HZ / 2.0
    assert (top + 1) * capstan >= FIELD_RATE_HZ / 2.0
    # a faster series carries more of the same train
    faster = tm.harmonics(capstan, "flat spot", sample_rate_hz=LINE_RATE_HZ)
    assert max(entry["order"] for entry in faster) > top


def test_an_unknown_fault_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        tm.harmonics(1.0, "sticky bushing")


# --------------------------------------------------------------------------
# where the sampling puts it
# --------------------------------------------------------------------------


def test_a_frequency_below_the_nyquist_is_where_it_is():
    assert tm.alias_hz(3.0, 100.0) == pytest.approx(3.0)
    assert tm.alias_hz(49.0, 100.0) == pytest.approx(49.0)


def test_an_exact_multiple_of_the_sampling_rate_folds_to_dc():
    for multiple in (1, 2, 3):
        assert tm.alias_hz(multiple * FIELD_RATE_HZ, FIELD_RATE_HZ) == \
            pytest.approx(0.0, abs=1e-9)


def test_alias_hz_takes_an_array():
    folded = tm.alias_hz([10.0, 60.0, 110.0], 100.0)
    assert isinstance(folded, np.ndarray)
    assert folded == pytest.approx([10.0, 40.0, 10.0])


def test_the_drums_second_order_lands_on_dc_in_a_per_field_series():
    """A drum with a two-lobe bearing puts its whole second order into the
    MEAN of a per-field series, where it is not a line at all and where any
    estimator that removes a mean removes it with the mean."""
    drum = named(rates(), "head drum")["rate_hz"]
    second = tm.harmonics(drum, "out of round",
                          sample_rate_hz=FIELD_RATE_HZ)[0]
    assert second["at_hz"] == pytest.approx(FIELD_RATE_HZ, rel=1e-12)
    assert second["alias_hz"] == pytest.approx(0.0, abs=1e-9)
    assert second["at_dc"] is True
    assert second["folded"] is True

    # per line the same order is an ordinary component of the series
    per_line = tm.harmonics(drum, "out of round",
                            sample_rate_hz=LINE_RATE_HZ)[0]
    assert per_line["at_dc"] is False
    assert per_line["folded"] is False
    assert per_line["alias_hz"] == pytest.approx(2.0 * drum, rel=1e-12)


# --------------------------------------------------------------------------
# the drum at the Nyquist, which is the trap this file exists for
# --------------------------------------------------------------------------


def test_the_drum_rate_is_exactly_the_per_field_nyquist():
    """Not approximately: the format fixes one revolution per two fields,
    so the equality is a definition and not a coincidence of numbers."""
    assert named(rates(), "head drum")["rate_hz"] == \
        pytest.approx(FIELD_RATE_HZ / tm.FIELDS_PER_DRUM_REVOLUTION, rel=1e-15)
    assert tm.FIELDS_PER_DRUM_REVOLUTION == 2


def test_the_drum_is_refused_at_its_own_nyquist_and_admitted_per_line():
    duration_s = 1024.0 / FIELD_RATE_HZ          # the arc's 1024-field record

    per_field = named(tm.observable(rates(), duration_s, FIELD_RATE_HZ),
                      "head drum")
    assert per_field["resolvable"] is False
    assert per_field["at_nyquist"] is True
    assert "half the sampling rate" in per_field["why_not"]

    per_line = named(tm.observable(rates(), duration_s, LINE_RATE_HZ),
                     "head drum")
    assert per_line["resolvable"] is True
    assert per_line["at_nyquist"] is False


def test_why_the_drum_is_refused_it_is_the_head_alternation():
    """The reason, planted rather than asserted. A drum modulation sampled
    once per field is bit-identical to the head A/B alternation, so no
    search can attribute a line there to one rather than the other; sampled
    once per line the two are all but uncorrelated."""
    drum = named(rates(), "head drum")["rate_hz"]
    fields = np.arange(1024)

    per_field = np.cos(2.0 * np.pi * drum * fields / FIELD_RATE_HZ)
    alternation = np.where(fields % 2 == 0, 1.0, -1.0)
    assert np.max(np.abs(per_field - alternation)) < 1e-9
    assert abs(np.corrcoef(per_field, alternation)[0, 1]) == \
        pytest.approx(1.0, abs=1e-9)

    lines = np.arange(int(1024 * LINES_PER_FIELD))
    per_line = np.cos(2.0 * np.pi * drum * lines / LINE_RATE_HZ)
    head_per_line = np.where((lines // int(LINES_PER_FIELD)) % 2 == 0,
                             1.0, -1.0)
    assert abs(np.corrcoef(per_line, head_per_line)[0, 1]) < 0.05


def test_the_other_seven_parts_are_unaffected_by_the_strict_comparison():
    duration_s = 1024.0 / FIELD_RATE_HZ
    for entry in tm.observable(rates(), duration_s, FIELD_RATE_HZ):
        if entry["name"] == "head drum":
            continue
        assert entry["resolvable"] is True, entry["name"]
        assert entry["why_not"] == ""


def test_a_rate_slower_than_one_cycle_is_still_refused_for_that_reason():
    short = named(tm.observable(rates(), 0.23, LINE_RATE_HZ), "supply reel")
    assert short["resolvable"] is False
    assert short["why_not"] == "slower than one cycle in the record"


def test_observable_names_the_two_permanent_degeneracies():
    """The guides share a declared diameter and the reels share a band, so
    each pair is one direction and not two. `aliases_onto` says so before a
    search credits either with a line of its own."""
    seen = {entry["name"]: entry["aliases_onto"]
            for entry in tm.observable(rates(), 17.0, LINE_RATE_HZ)}
    assert seen["entry guide"] == ("exit guide",)
    assert seen["exit guide"] == ("entry guide",)
    assert seen["supply reel"] == ("take-up reel",)
    assert seen["take-up reel"] == ("supply reel",)
    assert seen["capstan"] == ()


# --------------------------------------------------------------------------
# what the shape costs
# --------------------------------------------------------------------------


def test_the_guide_and_the_pinch_roller_collide_exactly_by_their_diameters():
    """3 x guide = 5 x pinch roller, and it is exact because the declared
    diameters are 6.0 and 10.0 mm. Checked against the diameters themselves
    so the test fails if either declaration moves."""
    guide = next(part for part in tm.TRANSPORT if part["name"] == "entry guide")
    pinch = next(part for part in tm.TRANSPORT if part["name"] == "pinch roller")
    # the rate goes as one over the diameter, so the orders go as the diameters
    assert guide["diameter_m"] / pinch["diameter_m"] == pytest.approx(3.0 / 5.0)

    hit = next(entry for entry in tm.harmonic_collisions(rates(), max_order=5)
               if entry["first"] == "entry guide" and entry["first_order"] == 3
               and entry["second"] == "pinch roller" and entry["second_order"] == 5)
    assert hit["exact"] is True
    assert hit["by_diameter"] is True
    assert hit["at_hz"] == pytest.approx(hit["against_hz"], rel=1e-12)
    # the escape is the axis, not the rate: a guide acts on tension and the
    # pinch roller on speed, so the complex residual still separates them
    assert hit["same_axis"] is False
    assert {hit["first_acts_on"], hit["second_acts_on"]} == {"tension", "speed"}


def test_a_reels_second_order_covers_the_impedance_roller_on_the_same_axis():
    """The case with no escape: a reel sweeps a band rather than a rate, its
    second order spans that band doubled, and the impedance roller's
    fundamental falls inside it. Both act on tension."""
    band = named(rates(), "take-up reel")["rate_band_hz"]
    roller = named(rates(), "impedance roller")["rate_hz"]
    assert 2.0 * band[0] < roller < 2.0 * band[1]

    hit = next(entry for entry in tm.harmonic_collisions(rates(), max_order=5)
               if entry["first"] == "impedance roller"
               and entry["second"] == "take-up reel"
               and entry["second_order"] == 2 and entry["first_order"] == 1)
    assert hit["by_band"] is True
    assert hit["same_axis"] is True
    assert hit["first_acts_on"] == "tension"


def test_the_drum_is_never_credited_with_a_diameter_collision():
    """Its rate comes from the format and not from its 62 mm, so a small
    integer ratio against another part's diameter has no cause behind it."""
    for entry in tm.harmonic_collisions(rates(), max_order=6):
        if "head drum" in (entry["first"], entry["second"]):
            assert entry["by_diameter"] is False


def test_the_fundamentals_alone_hide_the_collisions():
    """The point of the shape, stated as a count: at order one the parts
    that collide are only the two pairs already known to be degenerate; the
    orders add more."""
    first_order = [entry for entry in tm.harmonic_collisions(rates(), max_order=1)]
    assert {(entry["first"], entry["second"]) for entry in first_order} == {
        ("entry guide", "exit guide"), ("supply reel", "take-up reel")}
    assert len(tm.harmonic_collisions(rates(), max_order=5)) > len(first_order)


def test_collisions_can_be_compared_where_the_measurement_puts_them():
    found = tm.harmonic_collisions(rates(), max_order=3,
                                   sample_rate_hz=FIELD_RATE_HZ)
    assert all("at_alias_hz" in entry for entry in found)
    for entry in found:
        assert entry["at_alias_hz"] <= FIELD_RATE_HZ / 2.0 + 1e-9


# --------------------------------------------------------------------------
# nothing already relied on has moved
# --------------------------------------------------------------------------


def test_the_search_band_is_one_number_shared_with_find_lines():
    """`hifi_carriers` reads this default off `find_lines`' signature so the
    two cannot drift apart; every statement about what a measurement can
    separate must use the same band the search does."""
    import inspect
    assert inspect.signature(tm.find_lines).parameters["tolerance"].default \
        == tm.SEARCH_TOLERANCE


def test_rotation_rates_still_carries_every_key_its_callers_read():
    for entry in rates():
        assert set(entry) >= {"name", "order", "kind", "acts_on",
                              "after_capstan", "certain", "rate_hz",
                              "rate_note"}
    assert "rate_band_hz" in named(rates(), "supply reel")
    assert "diameter_m" in named(rates(), "capstan")


def test_observable_still_answers_the_question_it_always_did():
    """The added keys are additions. A caller reading only `resolvable` and
    `why_not` - which is what `hifi_carriers.searchable_rates` does - sees
    the same verdict on every part the change does not concern."""
    for entry in tm.observable(rates(), 5.0, LINE_RATE_HZ):
        assert isinstance(entry["resolvable"], bool)
        assert isinstance(entry["why_not"], str)
        assert entry["resolvable"] == (entry["why_not"] == "")


def test_no_rate_in_this_file_is_retyped():
    """Every figure here comes from `rotation_rates` or from the standard's
    own definition of the field rate, so a change to a declared diameter
    moves the tests with the model."""
    drum = named(rates(), "head drum")["rate_hz"]
    assert drum == pytest.approx(tm.drum_rate_hz(FIELD_RATE_HZ), rel=1e-15)
    for part in tm.TRANSPORT:
        if "diameter_m" in part and part["kind"] != "drum":
            expected = td.LINEAR_TAPE_SPEED_M_S / (np.pi * part["diameter_m"])
            assert named(rates(), part["name"])["rate_hz"] == \
                pytest.approx(expected, rel=1e-12)
