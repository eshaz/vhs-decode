"""Deck, source and system profiles: what applies before anything is fitted.

The three choices fail differently and the tests keep them apart. A system
is a specification and must be exact. A source decides which chain entries
EXIST, and an absent entry must be refused rather than fitted small. A deck
is the only measured one and must fall back to the specification when it
has not been measured.
"""

import pytest

from vhsdecode.models import profiles


def test_the_systems_carry_their_specified_ratios_exactly():
    ntsc = profiles.SYSTEMS["NTSC-M"]
    assert ntsc["subcarrier_cycles_per_line"] == (455, 2)
    assert ntsc["colour_frame_fields"] == 4
    assert ntsc["lines"] == 525
    pal = profiles.SYSTEMS["PAL-B/G/I"]
    assert pal["subcarrier_cycles_per_line"] == (1135, 4)
    assert pal["colour_frame_fields"] == 8      # the quarter line and the switch
    assert pal["burst_phase_deg"] == (135.0, -135.0)
    assert pal["line_rate_hz"] == 15625.0


def test_a_camera_has_no_tuner_and_a_television_source_does():
    television = profiles.profile(source="television")
    assert "tuner clipping" in television["chain_entries"]
    assert "vestigial sideband" in television["chain_entries"]
    camera = profiles.profile(source="camera")
    assert "tuner clipping" not in camera["chain_entries"]
    assert "echo" not in camera["chain_entries"]
    # and the absent ones are named, so they can be refused rather than fitted
    absent = profiles.absent_entries("camera")
    assert "tuner clipping" in absent and "vestigial sideband" in absent
    assert "tuner clipping" not in profiles.absent_entries("television")


def test_an_unmeasured_deck_falls_back_to_the_specification():
    ideal = profiles.profile(deck="ideal")["deck"]
    assert not ideal["measured"]
    nominal, tolerance, unit, cite = ideal["parameters"]["carrier_sync_tip_hz"]
    assert nominal == 3.4e6 and tolerance == 0.1e6 and unit == "Hz"
    assert "SMPTE" in cite
    measured = profiles.profile(deck="SLV-778HF")["deck"]
    assert measured["measured"]
    assert "head_to_tape_separation_m" in measured["measurements"]
    value, error, _unit, source = measured["measurements"]["head_to_tape_separation_m"]
    assert 0 < error < value                     # an error bar, not a guess
    assert "2026-09-05" in source


def test_every_capture_this_arc_uses_has_a_recorded_history():
    assert profiles.for_capture("countdown")["source"]["name"] == "television"
    assert profiles.for_capture("home")["source"]["name"] == "home vcr"
    pnb = profiles.for_capture("pnb")
    assert pnb["source"]["name"] == "composite"
    assert pnb["deck"]["measured"]
    with pytest.raises(ValueError, match="no recorded history"):
        profiles.for_capture("some_tape_nobody_wrote_down")


def test_the_known_constants_reach_the_profile_and_are_refused_as_components():
    from vhsdecode.models import running_differential as rd

    constants = profiles.profile(source="composite")["constants"]
    assert "which head reads which field" in constants
    assert "capture clock" in constants
    assert "source time base" in constants        # the composite source adds it
    with pytest.raises(ValueError, match="known constant"):
        rd.refuse_constant_component("capture clock")


def test_an_unknown_system_source_or_deck_is_refused_not_guessed():
    for kwargs in ({"system": "SECAM-L"}, {"source": "telepathy"},
                   {"deck": "a deck nobody owns"}):
        with pytest.raises(ValueError, match="unknown"):
            profiles.profile(**kwargs)


def test_the_television_source_carries_multipath_and_a_source_gain_control():
    """Ethan: 'The television stage also needs to contain a model for multi
    path reflection of television signals.' It does, with its measured
    status declared rather than assumed."""
    television = profiles.SOURCES["television"]
    assert "multipath" in television["chain"]
    assert "source agc band" in television["chain"]
    assert "tapped delay" in television["multipath"]
    # and the measurement that did not confirm it is recorded beside it
    assert "not confirmed" in television["multipath"].lower()
    # neither belongs to a source that never transmitted
    for name in ("composite", "camera"):
        assert "multipath" not in profiles.SOURCES[name]["chain"]
