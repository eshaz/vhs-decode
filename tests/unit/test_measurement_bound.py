"""What a measured region can hold, before any estimator is chosen.

Ethan: 'The amount of possible memory is the limit of information
contained in the measured area ... Each measurement is bound in accuracy
by the amount of data it has to use for measurement.'
"""

import math

import pytest

from vhsdecode.models import measurement_bound as mb


def test_a_region_holds_two_bt_real_numbers_and_bt_complex_ones():
    out = mb.dimensions(4.7e-6, 3.0e6)
    assert out["time_bandwidth_product"] == pytest.approx(14.1)
    assert out["complex_dimensions"] == pytest.approx(14.1)
    assert out["real_dimensions"] == pytest.approx(28.2)
    # and the resolution is the 1/T law, the same one the probes obey
    assert out["resolution_hz"] == pytest.approx(1.0 / 4.7e-6)


def test_the_accuracy_bound_follows_root_n_and_root_parameters():
    one = mb.best_accuracy(4.7e-6, 3.0e6, signal_to_noise=100.0)
    # doubling the region buys a factor of root two and no more
    twice = mb.best_accuracy(9.4e-6, 3.0e6, signal_to_noise=100.0)
    assert one["relative_error"] / twice["relative_error"] == pytest.approx(
        math.sqrt(2.0), rel=1e-9)
    # and every extra parameter costs the same root
    four = mb.best_accuracy(4.7e-6, 3.0e6, 100.0, parameters=4)
    assert four["relative_error"] == pytest.approx(2.0 * one["relative_error"])


def test_a_measurement_finer_than_the_file_is_reported_as_finished():
    from vhsdecode.models import output_limit as ol

    floor = ol.output_profile(bits=10)["quantisation_noise_ire"]
    # the measured case: pooled over a field at the tapes' own SNR
    pooled = mb.best_accuracy(4.7e-6 * 241, 3.0e6, signal_to_noise=2633.0)
    verdict = mb.against_output(pooled["relative_error"], 40.0, floor)
    assert verdict["already_enough"]
    assert verdict["over_floor"] < 1.0
    # one pulse alone is not
    single = mb.best_accuracy(4.7e-6, 3.0e6, signal_to_noise=2633.0)
    assert not mb.against_output(single["relative_error"], 40.0, floor)["already_enough"]


def test_the_survey_orders_the_regions_by_what_they_can_hold():
    rows = mb.survey()["rows"]
    assert [r["region"] for r in rows][0] == "front porch"
    assert [r["region"] for r in rows][-1] == "whole vertical interval"
    # the vertical interval holds two orders of magnitude more than a pulse
    pulse = next(r for r in rows if r["region"] == "line sync pulse")
    interval = next(r for r in rows if r["region"] == "whole vertical interval")
    assert interval["complex_dimensions"] / pulse["complex_dimensions"] > 100
    # and every region cites where its duration comes from
    assert all(r["cite"] for r in rows)
