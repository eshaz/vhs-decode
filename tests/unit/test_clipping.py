"""Clipping at the recorder's input.

Ethan: "the sync pulse of the video side may be clipped ... Clipping will
happen from the source -> recording VCR, if it exists." And: "If clipping is
detected, the ideal shape will be truncated at the bottom."

The second is the one that matters. A synthetic reference that was never
clipped, compared against a recording that was, reports the clip as a defect
of everything downstream of the recorder's input.
"""

import numpy as np
import pytest

from vhsdecode.models import clipping as clip
from vhsdecode.models import pair_dimension as pd

BAND = np.linspace(1e6, 7e6, 512)


def test_the_stated_clip_levels_and_the_sign_change():
    """SMPTE 32M gives VHS 160% white and 40% dark; IEC 60774-3 gives S-VHS
    210% and MINUS 70, so the dark clip changes sign between the formats
    rather than merely moving."""
    vhs = clip.clip_levels("VHS")
    svhs = clip.clip_levels("S-VHS")
    assert vhs["white"] == pytest.approx(1.60)
    assert vhs["dark"] == pytest.approx(0.40)
    assert svhs["white"] == pytest.approx(2.10)
    assert svhs["dark"] == pytest.approx(-0.70)
    assert np.sign(vhs["dark"]) != np.sign(svhs["dark"])
    with pytest.raises(ValueError):
        clip.clip_levels("Betamax")


def test_the_describing_function_is_the_identity_below_the_limit():
    assert float(clip.describing_function(0.5)) == pytest.approx(1.0)
    assert float(clip.describing_function(1.0)) == pytest.approx(1.0)
    # and falls monotonically above it
    above = [float(clip.describing_function(r)) for r in (1.5, 2.0, 3.0, 6.0)]
    assert all(a > b for a, b in zip(above, above[1:]))
    assert above[0] == pytest.approx(0.781, abs=0.01)


def test_the_control_shows_a_clipper_has_no_frequency_response():
    """THE CONTROL, and it must be able to fail. An earlier version asserted
    that a hard clip fails the subtractability rule; running it refuted
    that, because a clipped signal is bounded away from zero as often as an
    unclipped one. The real objection is that a memoryless non-linearity has
    no frequency response - its apparent gain depends on the drive."""
    rng = np.random.default_rng(7)
    result = clip.hard_limit_control(2.0 * rng.standard_normal(2048), 0.4)
    assert result["response_depends_on_drive"]
    responses = result["responses_by_drive"]
    assert all(a > b for a, b in zip(responses, responses[1:]))
    # a signal that never reaches the limit sees no such dependence
    quiet = clip.hard_limit_control(0.01 * rng.standard_normal(2048), 0.4)
    assert not quiet["response_depends_on_drive"]


def test_the_level_axis_carries_barely_a_direction():
    """Measured rather than assumed: 1.22 of 5 where the crossover falls in
    band, against the control's 1.00 and the record level's 2.90. A
    memoryless non-linearity has no frequency shape, and the only frequency
    dependence comes through the emphasis lift, which spans about two to one
    across the band."""
    spanning = clip.level_axis(BAND, levels=(0.05, 0.08, 0.12, 0.18, 0.25))
    control = clip.level_axis(BAND, levels=(0.05, 0.08, 0.12, 0.18, 0.25),
                              limited=False)
    assert control["effective"] == pytest.approx(1.0, abs=0.05)
    assert spanning["effective"] > control["effective"]
    assert spanning["effective"] < 2.0          # far below the record level


def test_a_detected_clip_truncates_the_ideal():
    """The operational consequence. The sync pulse is the largest excursion
    and sits at the dark end, so a dark clip takes it first - and the
    specified pulse is the synthetic side of the final dimension."""
    ideal = pd.spec_sync(50e6, 4096)
    clipped = clip.truncated_ideal(ideal, -24.0, "dark")
    assert clipped.min() == pytest.approx(-24.0)
    assert clipped.max() == pytest.approx(ideal.max())
    error = clip.truncation_error(ideal, -24.0, "dark")
    assert error["peak_error"] == pytest.approx(16.0, abs=0.1)
    assert 0.0 < error["truncated_fraction"] < 0.5
    # a clip below the pulse takes nothing
    assert clip.truncation_error(ideal, -40.0, "dark")["peak_error"] == \
        pytest.approx(0.0, abs=1e-9)


def test_the_truncation_error_is_reproducible_not_noise():
    """It is the same on every field, so it survives every average and reads
    as a real defect of whatever is being measured."""
    ideal = pd.spec_sync(50e6, 2048)
    first = clip.truncation_error(ideal, -30.0, "dark")
    second = clip.truncation_error(ideal, -30.0, "dark")
    assert first["error_share"] == pytest.approx(second["error_share"])
    assert first["error_share"] > 0.0


def test_the_entries_carry_a_declared_position_at_the_recorder_input():
    """Position 9: after the transmission path at 1 to 4, below the sub
    emphasis at 10 - the recording machine's input, which is where Ethan
    places it."""
    from vhsdecode.models import interference as inf
    assert clip.CHAIN_POSITION == 9
    for name in clip.signatures(BAND):
        assert name in clip.COMPONENT_POSITIONS
    assert clip.CHAIN_POSITION > max(
        inf.COMPONENT_ORDER[n] for n in ("vestigial sideband", "echo"))
    assert clip.CHAIN_POSITION < inf.COMPONENT_ORDER[
        "sub emphasis level dependence"]


def test_every_signature_is_subtractable():
    from vhsdecode.models import interference as inf
    for name, value in clip.signatures(BAND).items():
        assert inf.subtractable(value), name


def test_the_tuner_is_a_second_site_ahead_of_the_recorder():
    """Ethan: *"It might be at the television tuner stage, the cd sample was
    an over the air recording."* The chain has to be able to say so, and a
    site four positions ahead of the recorder's input is how it says it."""
    from vhsdecode.models import interference as inf
    assert clip.TUNER_CHAIN_POSITION < clip.CHAIN_POSITION
    assert clip.TUNER_CHAIN_POSITION > inf.COMPONENT_ORDER["echo"]
    places = clip.sites()
    assert places[clip.TUNER_PREFIX]["reaches_the_sync_tip"]
    # the recorder's dark clip is stated 40 per cent ABOVE the tip, so it
    # cannot be what takes the tip's own settled level
    assert not places[clip.CHAIN_PREFIX]["reaches_the_sync_tip"]


def test_provenance_decides_whether_the_tuner_entries_exist():
    """A tuner is in the chain of material that came over the air and in no
    other, so the entries appear for a broadcast source and not otherwise."""
    from vhsdecode.models import interference as inf
    plain = clip.signatures(BAND)
    air = clip.signatures(BAND, source="broadcast")
    assert not any(name.startswith(clip.TUNER_PREFIX) for name in plain)
    assert any(name.startswith(clip.TUNER_PREFIX) for name in air)
    for name, value in air.items():
        assert name in clip.COMPONENT_POSITIONS, name
        assert inf.subtractable(value), name


def test_the_two_sites_are_separable_only_because_the_levels_differ():
    """The separability law, applied: two limiters differing in threshold
    alone are collinear, and these two are separable because the sync tip
    lies in the window between their levels."""
    apart = clip.site_separation()
    assert apart["separable"]
    assert apart["the_tip_is_in_the_window"]
    assert apart["window"] == pytest.approx(clip.clip_levels()["dark"])
    # and it closes exactly when the receiver's overload reaches the
    # recorder's own dark clip, which would remove the whole sync pulse
    closed = clip.site_separation(overload_depth=clip.clip_levels()["dark"])
    assert not closed["separable"]


def test_the_composition_control_refuses_a_second_clip_at_the_same_level():
    """THE CONTROL THAT CAN REFUSE THE DESIGN: two hard limiters in series
    are one limiter at the tighter threshold, so a claim to have located a
    clip cannot rest on fitting a second one."""
    rng = np.random.default_rng(20260904)
    signal = rng.normal(0.0, 1.0, 4096)
    control = clip.composition_control(signal, 0.8, 0.5)
    assert control["composes_to_the_tighter"]
    assert control["tighter"] == pytest.approx(0.5)
    assert control["worst_difference"] == pytest.approx(0.0, abs=1e-12)
    # order does not matter either, which is what makes them one component
    assert clip.composition_control(signal, 0.5, 0.8)["tighter"] == 0.5


def test_the_broadcast_level_map_is_cited_not_assumed():
    levels = clip.broadcast_levels()
    assert levels["sync_percent_of_peak_carrier"] == 100.0
    low, high = levels["blanking_percent_of_peak_carrier"]
    assert low < high < levels["sync_percent_of_peak_carrier"]
    assert "BT.470-6" in levels["source"]


# --------------------------------------------------------------------------
# Detecting a clip, measuring its depth, and handing that to the levels
# --------------------------------------------------------------------------


def _planted(depth_ire, seed=4, noise=0.25):
    """a spec sync pulse with noise, hard-clipped `depth_ire` above its own
    tip"""
    from vhsdecode.models import pair_dimension as pd
    rng = np.random.default_rng(seed)
    ideal = pd.spec_sync(40e6, 400)
    noisy = ideal + rng.normal(0.0, noise, ideal.size)
    limit = float(ideal.min()) + float(depth_ire)
    return ideal, noisy, np.maximum(noisy, limit), limit


def test_flatness_is_not_the_witness_but_suppressed_noise_is():
    """AN UNCLIPPED SYNC TIP IS ALSO FLAT, so a detector built on flatness
    would report every pulse ever recorded. What a clip does that nothing
    else does is remove the gain at that level."""
    _ideal, noisy, clipped, _limit = _planted(6.0)
    tip, porch = slice(60, 240), slice(300, 390)
    hit = clip.detect_tip_clip(clipped[tip], clipped[porch])
    miss = clip.detect_tip_clip(noisy[tip], noisy[porch])
    assert hit["clipped"] and not miss["clipped"]
    # both have a long flat run - that is the point
    assert hit["run"] > 100 and miss["run"] > 100
    assert hit["noise_ratio"] < 0.1 < 0.9 < miss["noise_ratio"]


def test_the_depth_is_recovered_by_continuing_the_model_through_the_run():
    """Ethan: *"We may be able to determine how much was clipped by
    extrapolating the model down to where the waves were clipped."*"""
    ideal, _noisy, clipped, limit = _planted(6.0)
    flat = clipped <= limit + 1e-9
    got = clip.extrapolate_through_clip(clipped, ideal, flat)
    planted = limit - float(ideal.min())
    assert got["depth"] == pytest.approx(planted, rel=0.15)
    # and it errs LOW, which is the safe direction for a correction
    assert got["depth"] <= planted
    assert got["fitted_on"] == int((~flat).sum())


def test_fitting_on_the_clipped_samples_would_shrink_the_depth():
    """The failure the mask exists to avoid: including the flattened samples
    pulls the fit toward them and makes every clip look shallower the deeper
    it actually is."""
    ideal, _noisy, clipped, limit = _planted(8.0)
    flat = clipped <= limit + 1e-9
    honest = clip.extrapolate_through_clip(clipped, ideal, flat)
    cheating = clip.extrapolate_through_clip(clipped, ideal,
                                             np.zeros_like(flat, dtype=bool))
    assert cheating["depth"] < honest["depth"]


def test_a_clipped_tip_stretches_the_levels_and_the_factor_puts_it_back():
    """`hz_ire = (porch - tip) / -vsync_ire` at vhsdecode/field.py:1170. A
    clip moves the tip towards the porch, so the separation is short, hz_ire
    reads small, and every level is stretched."""
    got = clip.ire0_correction(porch_level=0.0, tip_level=-34.0, depth=6.0)
    assert got["hz_ire_measured"] == pytest.approx(34.0 / 40.0)
    assert got["hz_ire_corrected"] == pytest.approx(40.0 / 40.0)
    assert got["factor"] > 1.0
    # exactly 1.0 when nothing was clipped, so the hook is inert by default
    assert clip.ire0_correction(0.0, -40.0, 0.0)["factor"] == 1.0


def test_the_field_witness_pools_per_line_and_separates_the_two_arms():
    rng = np.random.default_rng(7)
    porch = rng.normal(0.0, 0.8, (200, 60))
    tip = rng.normal(-40.0, 0.8, (200, 60))
    unclipped = clip.tip_clip_from_lines(tip, porch)
    deep = clip.tip_clip_from_lines(np.maximum(tip, -40.0 + 1.6), porch)
    assert not unclipped["clipped"]
    assert unclipped["noise_ratio"] == pytest.approx(1.0, abs=0.05)
    assert deep["clipped"] and deep["noise_ratio"] < 0.2


def test_the_porch_anchored_rule_separates_a_clip_from_a_gain_change():
    """Ethan: *"the important thing is that the back porch is centered at
    0 ire ... if flatness is detected that does not correlate with the back
    porch, this is the amount of offset needed."*"""
    rng = np.random.default_rng(17)
    n = 200
    wander = rng.normal(0.0, 1.5, n)
    porch = wander + rng.normal(0.0, 0.05, n)

    follows = clip.clip_offset_from_porch(
        -40.0 + wander + rng.normal(0.0, 0.05, n), porch)
    pinned = clip.clip_offset_from_porch(
        -40.0 + rng.normal(0.0, 0.05, n), porch)
    assert not follows["clipped"] and follows["slope"] == pytest.approx(1.0, abs=0.05)
    assert pinned["clipped"] and pinned["offset_ire"] > 1.0


def test_the_slope_catches_a_partial_clip_that_correlation_misses():
    """THE CASE THAT DECIDES THE DESIGN: a tip following half the porch's
    movement is still 0.997 correlated with it."""
    rng = np.random.default_rng(17)
    n = 200
    wander = rng.normal(0.0, 1.5, n)
    porch = wander + rng.normal(0.0, 0.05, n)
    half = clip.clip_offset_from_porch(
        -40.0 + 0.5 * wander + rng.normal(0.0, 0.05, n), porch)
    assert abs(half["correlation"]) > 0.99, "correlation alone would pass it"
    assert half["slope"] == pytest.approx(0.5, abs=0.05)
    assert half["clipped"]


def test_a_gain_change_is_not_reported_as_a_clip():
    """A limiter can only REDUCE how much the tip follows. Anything moving
    the tip more than the porch is the deck's gain, not a clip."""
    rng = np.random.default_rng(17)
    n = 200
    wander = rng.normal(0.0, 1.5, n)
    porch = wander + rng.normal(0.0, 0.05, n)
    gain = clip.clip_offset_from_porch(
        -40.0 + 1.3 * wander + rng.normal(0.0, 0.05, n), porch)
    assert gain["slope"] > 1.0
    assert not gain["clipped"]


def test_a_porch_that_does_not_move_cannot_witness_anything():
    flat = clip.clip_offset_from_porch(np.full(50, -40.0), np.zeros(50))
    assert not flat["clipped"]
    assert "does not move" in flat["why"]
