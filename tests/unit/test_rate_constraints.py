"""Rate constraints: no fitted parameter may vary faster than the mechanism
that owns it.

Ethan: *"Constrain parameter rates to physical time constants. Drum
29.97 Hz, head switch 59.94 Hz, capstan servo tens of Hz. Nothing
mechanical varies above a few hundred Hz; a parameter floating per-line
will absorb things it shouldn't."*

What is tested: that every rate is the transport model's own and not a
retyped figure, that the deck's figures carry their schematic source and
the servo smoothing is computed from its parts, that the ceiling is a few
hundred hertz by derivation, that the registry refuses what it does not
know, and - on planted series - that a drum-rate wobble passes, a per-line
random walk and a static line-locked pattern are flagged, and `constrain`
keeps the wobble while removing the walk and hands the walk back as
evidence.
"""

import numpy as np
import pytest

from vhsdecode.models import head_model as hm
from vhsdecode.models import rate_constraints as rc
from vhsdecode.models import transport_dimensions as td
from vhsdecode.models import transport_model as tm

LINE_RATE = rc.line_rate_hz("NTSC")
FIELD_RATE = rc.field_rate_hz("NTSC")


@pytest.fixture(scope="module")
def rates():
    return rc.mechanism_rates("VHS", "NTSC", "SP", "SLV-778HF")


@pytest.fixture(scope="module")
def ceiling(rates):
    return float(rc.mechanical_ceiling_hz(rates)["ceiling_hz"])


def planted(fields=16, lines=244, seed=0, wobble=3.0, walk=0.0,
            pattern=0.0, two_heads=True):
    """A per-line series on the true time axis: fields at the field rate,
    lines at the line rate from line 10, white noise of unit variance, a
    drum-rate wobble, optionally a random walk and a line-locked pattern."""
    rng = np.random.default_rng(seed)
    field_index = np.repeat(np.arange(fields), lines)
    line_index = np.tile(np.arange(10, 10 + lines), fields)
    times = field_index / FIELD_RATE + line_index / LINE_RATE
    drum = tm.drum_rate_hz(FIELD_RATE)
    clean = wobble * np.cos(2.0 * np.pi * drum * times + 0.3)
    series = clean + rng.standard_normal(fields * lines)
    parts = {"wobble": clean}
    if walk:
        parts["walk"] = np.cumsum(walk * rng.standard_normal(fields * lines))
        series = series + parts["walk"]
    if pattern:
        parts["pattern"] = np.where(line_index % 7 < 3, pattern, -pattern)
        series = series + parts["pattern"]
    if not two_heads:
        keep = field_index % 2 == 0
        series, field_index, line_index = series[keep], field_index[keep], line_index[keep]
        parts = {k: v[keep] for k, v in parts.items()}
    return series, field_index, line_index, parts


class TestRates:
    """Every rate is the transport model's own figure, with its source."""

    def test_the_rotating_parts_are_the_transport_models_own(self, rates):
        mechanics = hm.mechanics_for("VHS", "NTSC", "SP")
        for part in tm.rotation_rates(mechanics["linear_tape_speed_m_s"], FIELD_RATE):
            entry = rates[part["name"]]
            assert entry["rate_hz"] == pytest.approx(part["rate_hz"], rel=1e-12)
            assert entry["kind"] == "mechanical"
            assert "VTG82063" in entry["source"]
        assert rates["head drum"]["rate_hz"] == pytest.approx(
            tm.drum_rate_hz(FIELD_RATE), rel=1e-12)
        assert rates["head drum"]["certain"]

    def test_the_head_switch_is_one_per_field(self, rates):
        assert rates["head switch"]["rate_hz"] == pytest.approx(
            rates["head drum"]["rate_hz"] * tm.FIELDS_PER_DRUM_REVOLUTION)
        assert rates["head switch"]["rate_hz"] == pytest.approx(FIELD_RATE)

    def test_the_field_rate_is_the_standards_not_the_guides_round_figure(self):
        # 525 lines at 15734.264 Hz is 59.94006 Hz; the JVC guide's 30 Hz
        # drum figure would make it 60 and put every predicted line 0.1
        # per cent off
        assert FIELD_RATE == pytest.approx(59.94006, rel=1e-6)
        assert LINE_RATE == pytest.approx(15734.264, rel=1e-6)

    def test_the_decks_figures_carry_their_schematic_source(self, rates):
        deck = rates["drum FG (deck)"]
        assert deck["rate_hz"] == 360.0
        assert "waveform 26" in deck["source"] and "pin 87" in deck["source"]
        assert "12 poles" in deck["note"]
        assert rates["drum FG (format)"]["rate_hz"] == pytest.approx(
            rates["head drum"]["rate_hz"] * td.DRUM_FG_POLES)
        # the two capstan FG sources disagree and both are carried
        assert rates["capstan FG (deck)"]["rate_hz"] == 1078.0
        assert rates["capstan FG (format)"]["rate_hz"] == 1440.0
        assert "disagree" in rates["capstan FG (deck)"]["note"]

    def test_the_servo_smoothing_is_computed_from_its_parts(self, rates):
        r_ohm, c_f = 47e3, 0.01e-6
        single = 1.0 / (2.0 * np.pi * r_ohm * c_f)
        assert rc.rc_ladder_corner_hz(r_ohm, c_f, 1) == pytest.approx(single, rel=1e-4)
        # two sections into an open load: 1/(1 + 3 s R C + (s R C)^2), whose
        # -3 dB point solves w^4 + 7 w^2 - 1 = 0 in units of 1/(R C)
        w = np.sqrt((-7.0 + np.sqrt(49.0 + 4.0)) / 2.0)
        assert rc.rc_ladder_corner_hz(r_ohm, c_f, 2) == pytest.approx(w * single, rel=1e-3)
        entry = rates["servo error smoothing (deck)"]
        assert entry["rate_hz"] == pytest.approx(w * single, rel=1e-3)
        assert entry["section_corner_hz"] == pytest.approx(single, rel=1e-4)
        assert entry["pwm_hz"] == 62.5e3
        assert "R116/C115" in entry["source"] and "R109/C111" in entry["source"]
        assert entry["kind"] == "servo" and not entry["certain"]

    def test_the_loop_bandwidths_are_labelled_assumptions(self, rates):
        for loop in ("drum servo loop", "capstan servo loop", "record AGC"):
            assert rates[loop]["assumed"] is True
            assert "ASSUMPTION" in rates[loop]["source"]
            assert not rates[loop]["certain"]

    def test_an_unknown_deck_or_format_is_refused(self):
        with pytest.raises(KeyError):
            rc.mechanism_rates(deck="no such deck")
        with pytest.raises(KeyError):
            rc.mechanism_rates("Betamax", "NTSC", "SP", None)


class TestCeiling:
    """'Nothing mechanical varies above a few hundred Hz', derived."""

    def test_the_sweep_shapes_converge_at_the_third_field_harmonic(self):
        bow = rc.sweep_ceiling_hz(FIELD_RATE)
        assert bow["harmonic"] == 3
        assert bow["ceiling_hz"] == pytest.approx(3 * FIELD_RATE)
        assert bow["share_captured"] >= rc.SWEEP_CAPTURE_SHARE

    def test_the_ceiling_is_a_few_hundred_hertz(self, rates, ceiling):
        assert 100.0 < ceiling < 1000.0
        assert rc.mechanical_ceiling_hz(rates)["set_by"] == "drum FG (deck)"
        assert ceiling == 360.0
        generic = rc.mechanical_ceiling_hz(rc.mechanism_rates(deck=None))
        assert generic["set_by"] == "drum FG (format)"
        assert generic["ceiling_hz"] == pytest.approx(
            td.DRUM_FG_POLES * tm.drum_rate_hz(FIELD_RATE))

    def test_the_capstan_fg_does_not_set_it(self, rates):
        candidates = rc.mechanical_ceiling_hz(rates)["candidates"]
        assert not any("capstan" in name for name in candidates)


class TestRegistry:
    def test_every_parameter_resolves_to_a_mechanism_and_a_rate(self, ceiling):
        for name in rc.PARAMETERS:
            entry = rc.allowed_rate(name)
            assert entry["parameter"] == name
            assert entry["allowed_hz"] >= 0.0
            assert entry["set_by"] and entry["source"]
            if entry["rule"] == "ceiling":
                assert entry["allowed_hz"] == ceiling
                assert entry["assumed"] is False

    def test_an_unknown_parameter_is_refused_with_the_known_ones_listed(self):
        with pytest.raises(KeyError) as refused:
            rc.allowed_rate("picture_brightness")
        assert "sync_tip_level" in str(refused.value)

    def test_every_per_line_parameter_is_allowed_far_less_than_it_floats(self):
        for name, entry in rc.PARAMETERS.items():
            if "per line" in entry["floats_at"]:
                assert rc.allowed_rate(name)["allowed_hz"] < LINE_RATE / 20.0

    def test_the_agc_rule_is_an_assumption_and_the_network_a_constant(self):
        assert rc.allowed_rate("clip_depth")["assumed"] is True
        assert rc.allowed_rate("front_porch_relaxation_tau")["allowed_hz"] == 0.0


class TestRateTest:
    """Planted series, the contiguous form and the structured form."""

    def test_a_drum_rate_wobble_passes(self, ceiling):
        series, *_ = planted(wobble=3.0)
        verdict = rc.rate_test(series, LINE_RATE, ceiling, field_rate_hz=FIELD_RATE)
        assert verdict["usable"] and not verdict["floats_too_fast"]
        assert verdict["slow_component_present"]
        assert verdict["raw"]["excess_ratio"] < verdict["null"]["excess_ratio_bound"]
        assert "white above the rate" in verdict["signature"]

    def test_a_per_line_random_walk_is_flagged_and_located(self, ceiling):
        series, *_ = planted(wobble=3.0, walk=1.0)
        verdict = rc.rate_test(series, LINE_RATE, ceiling, field_rate_hz=FIELD_RATE)
        assert verdict["floats_too_fast"]
        assert verdict["raw"]["excess_ratio"] > verdict["null"]["excess_ratio_bound"]
        assert "broadband" in verdict["signature"]
        # one over frequency squared: the excess lives in the first octave
        # above the rate
        low, high = verdict["excess_lives_hz"]
        assert low == pytest.approx(ceiling) and high == pytest.approx(2 * ceiling)

    def test_the_structured_form_passes_a_wobble_on_alternating_heads(self, ceiling, rates):
        series, field_index, line_index, _ = planted(fields=28, wobble=3.0)
        mechanisms = {name: entry["rate_hz"] for name, entry in rates.items()
                      if entry["kind"] == "mechanical" and entry["rate_hz"] > 0}
        verdict = rc.rate_test(series, LINE_RATE, ceiling, field_rate_hz=FIELD_RATE,
                               field_index=field_index, line_index=line_index,
                               mechanism_hz=mechanisms, draws=100)
        assert verdict["usable"] and verdict["form"] == "structured"
        assert not verdict["floats_too_fast"]
        assert not verdict["line_locked_above_rate"]
        # the wobble's zero crossing sits mid-field, so it is an alternating
        # TILT and the slow end finds the drum there
        strongest = verdict["slow"]["strongest_line"]
        assert strongest["name"] == "head drum"
        assert strongest["coefficient"] == "guide_tilt_m"
        assert strongest["sigma"] > 10.0

    def test_the_structured_form_flags_a_walk_field_to_field(self, ceiling):
        series, field_index, line_index, _ = planted(fields=28, wobble=3.0, walk=1.0)
        verdict = rc.rate_test(series, LINE_RATE, ceiling, field_rate_hz=FIELD_RATE,
                               field_index=field_index, line_index=line_index,
                               draws=100)
        assert verdict["floats_too_fast"]
        # a walk averaged over the fields still leaves one part in the field
        # count line-locked, so only the field-varying flag is asserted
        assert "field" in verdict["signature"]

    def test_the_structured_form_flags_a_static_pattern_as_line_locked(self, ceiling):
        series, field_index, line_index, _ = planted(fields=28, wobble=3.0, pattern=0.8)
        verdict = rc.rate_test(series, LINE_RATE, ceiling, field_rate_hz=FIELD_RATE,
                               field_index=field_index, line_index=line_index,
                               draws=100)
        assert verdict["line_locked_above_rate"]
        assert not verdict["floats_too_fast"]
        assert "line-locked" in verdict["signature"]
        first, last = verdict["line_locked"]["lives_in_lines"]
        assert first < last

    def test_a_single_heads_series_cannot_resolve_the_drum(self, ceiling, rates):
        series, field_index, line_index, _ = planted(fields=28, wobble=3.0,
                                                     two_heads=False)
        mechanisms = {"head drum": rates["head drum"]["rate_hz"]}
        verdict = rc.rate_test(series, LINE_RATE, ceiling, field_rate_hz=FIELD_RATE,
                               field_index=field_index, line_index=line_index,
                               mechanism_hz=mechanisms, draws=50)
        assert not verdict["floats_too_fast"]
        found = verdict["slow"]["mechanism_lines"]["guide_tilt_m"][0]
        assert found["name"] == "head drum" and not found["resolvable"]
        assert found["why_not"] == "above half the sampling rate"

    def test_a_series_sampled_too_slowly_is_declared_unusable(self, ceiling):
        per_field = np.random.default_rng(0).standard_normal(64)
        verdict = rc.rate_test(per_field, FIELD_RATE, ceiling)
        assert not verdict["usable"]
        assert verdict["nyquist_hz"] < ceiling


class TestConstrain:
    def test_the_lowpass_keeps_the_wobble_and_removes_only_white_noise(self, ceiling, rates):
        series, _, _, parts = planted(wobble=3.0)
        out = rc.constrain(series, LINE_RATE, ceiling, rates=rates,
                           field_rate_hz=FIELD_RATE)
        assert out["group_delay_s"] == 0.0
        # what remains beside the wobble is the noise below the cut: a
        # share ceiling / nyquist of unit variance, 0.21 rms
        kept_noise = np.sqrt(ceiling / (LINE_RATE / 2.0))
        assert np.std(out["constrained"] - parts["wobble"]) < 1.5 * kept_noise
        evidence = out["evidence"]
        assert abs(evidence["removed_lag_one"]) < 0.1
        assert evidence["mechanism_lines_in_removed"] == []
        # a white remainder puts on the field-harmonic picket exactly the
        # picket's share of the bins, and no more
        assert evidence["removed_share_on_field_harmonics"] < \
            1.5 * evidence["field_harmonic_bins_share"]

    def test_the_harmonic_form_keeps_the_wobble_and_hands_back_the_walk(self, ceiling, rates):
        series, _, _, parts = planted(wobble=3.0, walk=1.0)
        drum = rates["head drum"]["rate_hz"]
        out = rc.constrain(series, LINE_RATE, ceiling, method="harmonic",
                           fundamental_hz=drum, rates=rates)
        assert out["harmonics"] == int(ceiling // drum)
        # a walk has power at every frequency, the drum's included, and the
        # fit cannot tell that share from the wobble: the amplitude error
        # is bounded by the walk's OWN amplitude at the drum bin, measured
        # on the planted walk alone, and by nothing looser
        contamination = rc.constrain(parts["walk"], LINE_RATE, ceiling,
                                     method="harmonic",
                                     fundamental_hz=drum)["harmonic_amplitudes"][1]
        assert abs(out["harmonic_amplitudes"][1] - 3.0) <= contamination + 0.1
        # the removed part IS the walk, and says so: strongly serially
        # correlated where white noise would sit near zero
        assert np.corrcoef(out["removed"], parts["walk"])[0, 1] > 0.95
        assert out["evidence"]["removed_lag_one"] > 0.9

    def test_the_harmonic_form_needs_a_fundamental(self, ceiling):
        with pytest.raises(ValueError):
            rc.constrain(np.ones(64), LINE_RATE, ceiling, method="harmonic")
        with pytest.raises(ValueError):
            rc.constrain(np.ones(64), LINE_RATE, ceiling, method="magic")


class TestHalfContrast:
    def test_a_slow_step_is_identified_and_placed_at_the_records_fundamental(self, rates):
        rng = np.random.default_rng(3)
        times = np.arange(200) / FIELD_RATE
        values = rng.standard_normal(200) * 0.1 + np.where(times > times.mean(), 1.0, 0.0)
        mechanisms = {name: entry["rate_hz"] for name, entry in rates.items()
                      if entry["kind"] == "mechanical" and entry["rate_hz"] > 0}
        out = rc.half_contrast(values, times, mechanisms)
        assert out["identified"] and out["sigma"] > 3.0
        assert out["difference"] == pytest.approx(1.0, abs=0.1)
        assert out["lives_at_hz"] == pytest.approx(1.0 / out["record_s"])
        assert out["share_at_fundamental"] == pytest.approx(8.0 / np.pi ** 2)
        assert "head drum" in out["mechanisms_averaged_out"]
        assert "capstan" in out["mechanisms_averaged_out"]
        assert "supply reel" in out["mechanisms_that_can_live_there"]

    def test_white_noise_shows_no_contrast(self):
        rng = np.random.default_rng(4)
        times = np.arange(400) / FIELD_RATE
        out = rc.half_contrast(rng.standard_normal(400), times)
        assert not out["identified"] and abs(out["sigma"]) < 3.0
