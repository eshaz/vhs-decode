"""The content-independence check: planted bleed-through is detected and
sized, a clean series passes, and the things that are NOT content - field
drift, a fixed profile along the field, the two heads' differing profiles,
within-field drift - are not called content.

Ethan: *"Content-independence check: compute variance on blanking lines and
test-pattern segments, compare against full-field. Mismatch quantifies
picture-statistics bleed-through."*
"""

import numpy as np
import pytest

from vhsdecode.models import content_independence as ci

POPULATIONS = ci.line_populations("NTSC", 263)
FIELDS = 40


def run(planted, group=False, pattern=None):
    return ci.check_series(
        planted["values"], planted["field_index"], planted["line_index"],
        POPULATIONS, preceding_content=planted["content"],
        pattern_mask=pattern,
        group_index=planted["parity"] if group else None)


class TestThePopulationsComeFromTheSpecification:
    def test_the_525_line_sets(self):
        assert POPULATIONS["geometry"] == list(range(1, 10))
        assert POPULATIONS["reference"] == list(range(10, 21))
        assert POPULATIONS["first_unblanked"] == [21]
        assert POPULATIONS["picture"][0] == 22
        assert POPULATIONS["picture"][-1] == 258
        assert POPULATIONS["head_switch"] == [259, 260, 261, 262, 263]

    def test_the_625_line_sets(self):
        pal = ci.line_populations("PAL", 313)
        # 7.5 interval lines: line 8 still opens with an equalizing pulse
        assert pal["geometry"] == list(range(1, 9))
        assert pal["reference"] == list(range(9, 26))
        assert pal["picture"][0] == 27

    def test_the_count_law(self):
        # n reference lines: (1 + 1/n) / (1 - 1/n)
        assert ci.predicted_ratio([11] * 5) == pytest.approx(
            (1 + 1 / 11) / (1 - 1 / 11))
        assert 10 * np.log10(ci.predicted_ratio([11])) == pytest.approx(0.79, abs=0.01)
        assert 10 * np.log10(ci.predicted_ratio([5])) == pytest.approx(1.76, abs=0.01)


class TestPlantedBleedThrough:
    def test_a_clean_series_passes(self):
        rng = np.random.default_rng(1)
        planted = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=1.0)
        result = run(planted)
        for population in ("full_field", "picture"):
            channel = result[population]
            assert channel["verdict"] == "content-independent"
            assert abs(channel["excess_db"]) < 3 * channel["error_db"]
        assert result["content_slope"]["verdict"] == "no content slope"
        assert result["level"]["verdict"] == "no level shift"

    def test_planted_content_is_detected_and_sized(self):
        rng = np.random.default_rng(2)
        planted = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=1.0,
                           content_sigma=0.7)
        result = run(planted)
        channel = result["picture"]
        assert channel["verdict"] == "bleed-through"
        assert channel["z"] > ci.DETECTION_SIGMA
        # the size: what the picture added, in the series' units squared
        assert channel["added_variance"] == pytest.approx(0.49, rel=0.25)
        # and the slope on the planted covariate recovers the planted gain
        slope = result["content_slope"]
        assert slope["verdict"] == "content slope"
        assert slope["slope"] == pytest.approx(0.7, abs=3 * slope["error"])

    def test_a_static_shift_is_seen_by_the_level_channel_only(self):
        rng = np.random.default_rng(3)
        planted = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=0.1,
                           static_shift=0.3)
        result = run(planted)
        assert result["picture"]["verdict"] == "content-independent"
        level = result["level"]
        assert level["verdict"] == "level shift"
        assert level["shift"] == pytest.approx(0.3, abs=3 * level["error"])

    def test_the_static_pattern_population_reads_zero_on_static_content(self):
        """A static pattern cannot add line-to-line variance: the pattern
        column is the instrument's null control."""
        rng = np.random.default_rng(4)
        planted = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=1.0,
                           static_shift=2.0)
        pattern = np.ones(len(planted["values"]), dtype=bool)
        result = run(planted, pattern=pattern)
        assert result["pattern"]["verdict"] == "content-independent"


class TestWhatIsNotContent:
    def test_field_drift_is_absorbed_by_the_per_field_term(self):
        rng = np.random.default_rng(5)
        planted = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=1.0,
                           field_drift=5.0)
        assert run(planted)["picture"]["verdict"] == "content-independent"

    def test_a_fixed_profile_along_the_field_is_absorbed_by_the_line_term(self):
        rng = np.random.default_rng(6)
        profile = 3.0 * np.exp(-np.arange(263) / 20.0)     # the recovery
        planted = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=1.0,
                           line_profile=profile)
        assert run(planted)["picture"]["verdict"] == "content-independent"

    def test_the_per_head_pooling_trap(self):
        """Two heads with opposite tilts along the field: pooled, the
        residual carries half their difference and reads as content;
        decomposed within each head it does not."""
        rng = np.random.default_rng(7)
        planted = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=0.2,
                           head_profile=1.0)
        pooled = run(planted, group=False)["picture"]
        grouped = run(planted, group=True)["picture"]
        assert pooled["verdict"] == "bleed-through"
        assert grouped["verdict"] == "content-independent"

    def test_within_field_drift_climbs_along_the_field_and_content_does_not(self):
        rng = np.random.default_rng(8)
        drifting = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=0.5,
                            within_field_drift=2.0)
        content = ci.plant(rng, FIELDS, POPULATIONS, noise_sigma=0.5,
                           content_sigma=0.7)
        assert run(drifting)["drift"]["verdict"].startswith("climbs")
        assert run(content)["drift"]["verdict"] == "flat along the field"


class TestTheMeasuredGates:
    def test_occupied_reference_lines_are_measured(self):
        rng = np.random.default_rng(9)
        fields = 20
        lines = np.tile(np.array(POPULATIONS["reference"]), fields)
        rms = rng.normal(0.7, 0.02, size=len(lines))
        rms[np.isin(lines, [17, 18])] = 30.0            # an inserted signal
        found = ci.occupied_lines(rms, lines, POPULATIONS)
        assert found["occupied"] == [17, 18]
        assert found["floor"] == pytest.approx(0.7, abs=0.05)

    def test_plateau_fields_split_the_bounce_and_not_the_bars(self):
        rng = np.random.default_rng(10)
        bounce = {f: (4.5 if f < 13 else 103.5) + rng.normal(0, 0.2)
                  for f in range(27)}
        bounce[27] = 50.0                                 # a field in transition
        found = ci.plateau_fields(bounce, level_noise=0.1)
        assert not found["static_everywhere"]
        assert len(found["plateaus"]) == 2
        assert 27 in found["in_transition"]
        assert sorted(found["fields"]) == list(range(27))
        bars = {f: 47.0 + rng.normal(0, 0.05) for f in range(27)}
        assert ci.plateau_fields(bars, level_noise=0.1)["static_everywhere"]

    def test_the_control_attribution(self):
        mine = {"full_field": {"excess_db": 4.0, "error_db": 0.4}}
        control = {"full_field": {"excess_db": 1.0, "error_db": 0.3}}
        found = ci.control_attribution(mine, control)
        assert found["above_control_db"] == pytest.approx(3.0)
        assert found["error_db"] == pytest.approx(0.5)
        assert found["verdict"] == "picture statistics"
        same = ci.control_attribution(control, control)
        assert same["verdict"] == "within the static control"
