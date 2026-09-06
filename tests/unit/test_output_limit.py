"""The output file as the cap: 4 fsc, ten bits, and no further.

Ethan's stopping rule, and the point of it is that this floor is
arithmetic. It is fixed by the container before any measurement, so unlike
every instrument floor this arc has used it cannot be wrong by a factor of
four and cannot move when the instrument is re-calibrated.
"""

import math
import os

import numpy as np
import pytest

from vhsdecode.models import output_limit as ol


def test_the_floor_comes_from_the_container_and_not_from_a_constant():
    profile = ol.output_profile(bits=10)
    # 4 fsc derived from the line rate, not typed
    assert profile["sample_rate_hz"] == pytest.approx(14318181.818181818)
    assert profile["nyquist_hz"] == pytest.approx(7159090.909090909)
    # ten bits over this container: 64 of 65536 units, 400.768 units an IRE
    assert profile["step_ire"] == pytest.approx(64.0 / 400.768, rel=1e-9)
    assert profile["quantisation_noise_ire"] == pytest.approx(
        profile["step_ire"] / math.sqrt(12.0), rel=1e-12)
    assert profile["quantisation_noise_ire"] == pytest.approx(0.046100, abs=1e-6)
    assert profile["below_100_ire_db"] == pytest.approx(66.7, abs=0.1)


def test_more_bits_lower_the_floor_by_six_decibels_each():
    ten = ol.output_profile(bits=10)["quantisation_noise_ire"]
    eleven = ol.output_profile(bits=11)["quantisation_noise_ire"]
    assert ten / eleven == pytest.approx(2.0, rel=1e-12)
    # and a different decode's scale changes it, so the scale is not assumed
    wider = ol.output_profile(bits=10, black_16b=0.0, white_16b=65535.0)
    assert wider["quantisation_noise_ire"] < ten


def test_a_component_under_the_quantisation_noise_is_invisible_not_small():
    profile = ol.output_profile(bits=10)
    floor = profile["quantisation_noise_ire"]
    assert ol.is_meaningful(2.0 * floor, profile)["meaningful"]
    assert not ol.is_meaningful(0.5 * floor, profile)["meaningful"]
    edge = ol.is_meaningful(floor, profile)
    assert not edge["meaningful"] and edge["margin_db"] == pytest.approx(0.0)
    # the effect is an rms, so a vector is reduced the same way
    assert ol.is_meaningful(np.full(16, 2.0 * floor), profile)["over_floor"] \
        == pytest.approx(2.0)


def test_the_pooling_needed_to_reach_the_floor_is_root_n():
    profile = ol.output_profile(bits=10)
    out = ol.lines_to_reach_floor(0.9218, profile)
    assert out["lines_needed"] == pytest.approx(400.0, rel=0.02)
    # quarter the noise, a sixteenth of the lines
    quieter = ol.lines_to_reach_floor(0.9218 / 4.0, profile)
    assert quieter["lines_needed"] == pytest.approx(out["lines_needed"] / 16.0,
                                                    rel=1e-9)


def test_the_ranking_cuts_at_the_floor_and_orders_by_effect():
    ranked = ol.rank({"large": 5.0, "middling": 0.5, "tiny": 0.001})
    assert [row["component"] for row in ranked["rows"]] == [
        "large", "middling", "tiny"]
    assert ranked["meaningful"] == ["large", "middling"]
    assert ranked["invisible"] == ["tiny"]


def test_the_capped_budget_removes_what_the_file_cannot_hold():
    from vhsdecode.models import tesseract as T

    rng = np.random.default_rng(0)
    bins = 64
    frequency = np.linspace(0.2e6, 4.0e6, bins)
    values = 0.01 * (rng.standard_normal((2, 2, bins))
                     + 1j * rng.standard_normal((2, 2, bins)))
    cube = T.Cube(("head", "tape"), values, np.full(values.shape, 1e-8),
                  bins=frequency)
    budget = T.budget(cube)
    capped = ol.capped_budget(budget, frequency, signal_ire=100.0)
    assert capped["bins_out_of_band"] == 0          # 4 MHz is inside 7.16
    assert 0.0 <= capped["representable_fraction"] <= 1.0
    # a band reaching past Nyquist loses those bins
    wide = np.linspace(0.2e6, 20e6, bins)
    assert ol.capped_budget(budget, wide, 100.0)["bins_out_of_band"] > 0


def test_the_output_format_is_read_from_a_decode_and_the_cap_is_a_decision():
    fmt = ol.output_format()
    assert fmt["field_width"] == 910 and fmt["field_height"] == 263
    assert fmt["sample_rate_hz"] == pytest.approx(ol.FOUR_FSC_HZ)
    assert fmt["samples_per_field"] == 910 * 263
    # the container holds sixteen bits; the cap keeps ten, and the gap is
    # a stated decision rather than a property of the file
    assert fmt["container_bits"] == 16
    assert fmt["profile"]["bits"] == 10
    assert fmt["capped_bits_per_field"] < fmt["container_bits_per_field"]
    # and a real decode's own JSON overrides the defaults
    path = ("/tmp/claude-1000/-workspaces-vhs-decode/"
            "61dd5ae6-c67f-4bd1-bbaa-8c55cd55ffc0/scratchpad/time/home40_512.tbc.json")
    if os.path.exists(path):
        from_file = ol.output_format(path)
        assert from_file["source"] == path
        assert from_file["field_width"] == 910
