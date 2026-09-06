"""The deck log: validation refuses what cannot be used, natural
experiments are the swap-spanning pairs and say what they separate, and
the drift regressor recovers a planted drift and bounds it by the crystal.

Ethan: *"Log head swaps and maintenance dates. Gap geometry resets at a swap
while preamp and servo don't - that's a natural experiment for separating
head terms from deck terms."* / *"GPS/NTP-disciplined timestamps on
captures. Over a long acquisition, capture-chain drift becomes a hidden
variable correlated with everything; timestamps let you regress it out."*
"""

import copy
import datetime
import os

import numpy as np
import pytest

from vhsdecode.models import deck_log as dl

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tools", "ringing_measure", "deck_log.toml")
UTC = datetime.timezone.utc
LINE_HZ = 15734.2657


def capture(name, when, tape, source="bars", discipline="gps"):
    return {"file": name, "deck": "deck-a", "tape": tape, "source": source,
            "speed": "SP", "device": "card",
            "sample_rate_hz": 40_000_000, "bit_depth": 8,
            "timestamp": when, "timestamp_source": "GPS-disciplined host clock",
            "timestamp_discipline": discipline}


def base_log():
    return {
        "schema": 1,
        "clock": {"aging_ppm_per_year": 5.0, "thermal_ppm_per_kelvin": 0.5,
                  "temperature_excursion_kelvin": 10.0,
                  "status": "typical figures, not measured"},
        "capture_devices": [{"id": "card", "kind": "cxadc",
                             "crystal_hz": 28636363}],
        "tapes": [{"id": "tape-1"}, {"id": "tape-2"}],
        "decks": [{
            "id": "deck-a", "make": "Sony", "model": "SLV-778HF",
            "serial": "123",
            "head_swaps": [{"date": datetime.date(2026, 1, 15),
                            "heads": "video pair"}],
            "maintenance": [{"date": datetime.date(2025, 10, 1),
                             "kind": "head cleaning"}],
        }],
        "captures": [
            capture("a.flac", datetime.datetime(2025, 9, 1, tzinfo=UTC), "tape-1"),
            capture("b.flac", datetime.datetime(2026, 2, 1, tzinfo=UTC), "tape-1"),
            capture("c.flac", datetime.datetime(2026, 3, 1, tzinfo=UTC), "tape-2"),
        ],
    }


class TestValidation:
    def test_the_base_log_validates(self):
        report = dl.validate(base_log())
        assert report["ok"], report["errors"]
        assert report["warnings"] == []

    def test_a_capture_without_a_deck_is_refused(self):
        log = base_log()
        del log["captures"][0]["deck"]
        report = dl.validate(log)
        assert not report["ok"]
        assert any("no deck" in e for e in report["errors"])

    def test_a_capture_without_a_timestamp_source_is_refused(self):
        log = base_log()
        log["captures"][1]["timestamp_source"] = ""
        report = dl.validate(log)
        assert not report["ok"]
        assert any("timestamp_source" in e for e in report["errors"])
        log = base_log()
        del log["captures"][1]["timestamp_discipline"]
        assert not dl.validate(log)["ok"]

    def test_an_unstated_clock_bound_is_refused(self):
        log = base_log()
        del log["clock"]["status"]
        assert any("[clock]" in e for e in dl.validate(log)["errors"])
        del log["clock"]
        assert not dl.validate(log)["ok"]

    def test_gaps_are_warnings_not_errors(self):
        log = base_log()
        log["captures"][0]["timestamp_discipline"] = "none"
        log["captures"][0]["timestamp_source"] = "file mtime"
        log["decks"][0]["make"] = "unknown"
        report = dl.validate(log)
        assert report["ok"]
        assert any("undisciplined" in w for w in report["warnings"])
        assert any("make/model unrecorded" in w for w in report["warnings"])

    def test_a_maintenance_entry_needs_a_known_kind(self):
        log = base_log()
        log["decks"][0]["maintenance"].append({"date": datetime.date(2025, 11, 1)})
        assert not dl.validate(log)["ok"]


class TestNaturalExperiments:
    def test_swap_spanning_pairs_are_returned_and_ranked(self):
        found = dl.natural_experiments(base_log())
        swaps = [e for e in found if e["event"] == "head swap"]
        assert {(e["before"], e["after"]) for e in swaps} == {
            ("a.flac", "b.flac"), ("a.flac", "c.flac")}
        same_tape = next(e for e in swaps if e["after"] == "b.flac")
        other_tape = next(e for e in swaps if e["after"] == "c.flac")
        assert same_tape["rank"] == 0 and same_tape["same_tape"]
        assert other_tape["rank"] == 1 and other_tape["same_source"]
        # the list is ranked: every same-tape pair (rank 0) comes first
        assert found[0]["rank"] == 0 and found[0]["same_tape"]
        assert [e["rank"] for e in found] == sorted(e["rank"] for e in found)
        assert not same_tape["dating_valid"]

    def test_a_swap_separates_head_terms_from_deck_and_tape_terms(self):
        found = dl.natural_experiments(base_log())
        swap = next(e for e in found if e["event"] == "head swap")
        split = {(s["parameter"], s["with"]): s["moved"] for s in swap["separates"]}
        assert split[("head_spacing_m", "spacing_m")] == "head_spacing_m"
        assert split[("head_efficiency", "preamp_gain")] == "head_efficiency"
        assert split[("gain_db", "preamp_gain")] == "gain_db"
        # gap and azimuth both live on the head: a swap moves both and
        # leaves the identifiability collinearity in place
        still = {(s["parameter"], s["with"]) for s in swap["still_confounded"]}
        assert ("gap_m", "azimuth_error_degrees") in still
        assert all(t in " ".join(swap["holds"]) for t in ("preamp_gain", "servo_reference"))

    def test_a_cleaning_separates_the_debris_share_only(self):
        found = dl.natural_experiments(base_log())
        cleaning = next(e for e in found if e["event"] == "head cleaning")
        assert cleaning["moved"] == ["debris_spacing_m"]
        moved = {s["moved"] for s in cleaning["separates"]}
        assert moved == {"debris_spacing_m"}
        assert cleaning["dating_valid"]

    def test_tape_model_pairs_carry_the_instrument_verdict(self):
        pairs = dl.separated_by(dl.EVENT_RESETS["head swap"]["resets"])
        everything = pairs["split"] + pairs["still_confounded"]
        from_tape_model = [p for p in everything if p["source"] == "tape_model"]
        assert from_tape_model
        assert all("instruments_separate" in p for p in from_tape_model)

    def test_dating_is_refused_across_a_swap(self):
        log = base_log()
        a, b, c = log["captures"]
        assert not dl.dating_valid(log, a, b)["valid"]
        assert dl.dating_valid(log, b, c)["valid"]
        spans = dl.aging_spans(log, "deck-a")
        assert len(spans) == 2
        assert spans[0]["to"] == "2026-01-15" and spans[1]["from"] == "2026-01-15"


class TestTheDriftRegressor:
    def planted(self, log, ppm_per_day, seed=0, days=2.0):
        rng = np.random.default_rng(seed)
        series = {}
        t0 = dl._as_datetime(log["captures"][0]["timestamp"])
        for name in ("a.flac", "b.flac"):
            stamp = dl._as_datetime(next(c for c in log["captures"]
                                         if c["file"] == name)["timestamp"])
            offset = (stamp - t0).total_seconds()
            within = np.linspace(0, 600, 200)
            drift = LINE_HZ * ppm_per_day * 1e-6 / 86400.0 * (offset + within)
            series[name] = {"time_s": within,
                            "values": LINE_HZ + drift + rng.normal(0, 0.01, 200)}
        return series

    def two_day_log(self):
        log = base_log()
        log["captures"][1]["timestamp"] = datetime.datetime(2025, 9, 3, tzinfo=UTC)
        return log

    def test_a_planted_drift_is_recovered(self):
        log = self.two_day_log()
        planted_ppm_per_day = 1.0
        found = dl.drift_regressor(log, self.planted(log, planted_ppm_per_day),
                                   nominal=LINE_HZ)
        assert found["usable"]
        per_day = found["slope_per_second"] * 86400.0 / LINE_HZ * 1e6
        error = found["slope_error"] * 86400.0 / LINE_HZ * 1e6
        assert per_day == pytest.approx(planted_ppm_per_day, abs=3 * error)
        assert found["t"] > 3
        # two days of a 1 ppm/day drift sit inside the crystal's thermal
        # allowance: consistent with the capture chain
        assert found["within_crystal_bound"]
        assert found["bound"]["total_ppm"] == pytest.approx(
            5.0 * found["span_seconds"] / dl.SECONDS_PER_YEAR + 5.0)

    def test_a_drift_beyond_the_crystal_is_named_as_not_the_chain(self):
        log = self.two_day_log()
        found = dl.drift_regressor(log, self.planted(log, 50.0), nominal=LINE_HZ)
        assert not found["within_crystal_bound"]
        assert "not the capture chain" in found["verdict"]

    def test_an_undisciplined_timestamp_refuses_the_joint_axis(self):
        log = self.two_day_log()
        log["captures"][1]["timestamp_discipline"] = "none"
        found = dl.drift_regressor(log, self.planted(log, 1.0), nominal=LINE_HZ)
        assert not found["usable"]
        assert set(found["per_capture"]) == {"a.flac", "b.flac"}
        assert all(v["usable"] for v in found["per_capture"].values())

    def test_the_slow_term_can_be_regressed_out_of_another_series(self):
        log = self.two_day_log()
        found = dl.drift_regressor(log, self.planted(log, 1.0), nominal=LINE_HZ)
        rng = np.random.default_rng(3)
        target = 2.5 * (found["slow_term"] - found["slow_term"].mean()) \
            + rng.normal(0, 1e-4, found["count"])
        removed = dl.regress_out(target, found)
        assert removed["usable"]
        assert removed["coefficient"] == pytest.approx(2.5, rel=0.05)
        assert removed["explained"] > 0.95

    def test_the_bound_is_the_logs_and_not_the_modules(self):
        log = self.two_day_log()
        del log["clock"]
        with pytest.raises(ValueError):
            dl.drift_regressor(log, self.planted(log, 1.0), nominal=LINE_HZ)


class TestTheTemplate:
    def test_the_template_loads_and_validates(self):
        log = dl.load(TEMPLATE)
        report = dl.validate(log)
        assert report["ok"], report["errors"]
        # every timestamp in it is a file modification time, and it says so
        assert all(c["timestamp_discipline"] == "none" for c in log["captures"])
        assert sum("undisciplined" in w for w in report["warnings"]) == len(log["captures"])
        assert any("make/model unrecorded" in w for w in report["warnings"])

    def test_the_template_records_no_events_yet(self):
        log = dl.load(TEMPLATE)
        assert dl.natural_experiments(log) == []
        assert dl.aging_spans(log, "slv-778hf")[0]["to"] == "now"
        assert dl.describe(log)
