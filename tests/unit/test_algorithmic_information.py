"""Kolmogorov's three approaches, and the third applied to the remainder.

Ethan: 'I believe we are exhausting the combinatorial approach here by
modeling our functions against their expected data, along with some
aspects of the probalistic approach. I want to focus on identifying the
remainder of the components after we have excausted all other methods to
use the algorithmic approach ... I believe this can be applied to my
existing approach in hyper complex hilbert space.'
"""

import math

import numpy as np
import pytest

from vhsdecode.models import algorithmic_information as ai


def test_the_combinatorial_count_needs_no_distribution():
    """Kolmogorov's first approach: the logarithm of the state count."""
    out = ai.combinatorial(1e-3, 4.0e6, 1024)
    assert math.isclose(out["dimensions"], 4000.0, rel_tol=1e-12)
    assert math.isclose(out["bits_per_dimension"], 10.0, rel_tol=1e-12)
    # complex, so two real numbers a dimension
    assert math.isclose(out["bits"], 2.0 * 4000.0 * 10.0, rel_tol=1e-12)


def test_the_entropy_is_measured_and_its_bias_reported():
    rng = np.random.default_rng(0)
    out = ai.probabilistic(rng.normal(size=20000), levels=256)
    assert 0.0 < out["bits_per_sample"] < 8.0
    assert out["bias_bits"] > 0.0
    assert out["occupied_bins"] <= 256


def test_structure_compresses_and_noise_does_not():
    """The third approach, on cases whose answer is known."""
    rng = np.random.default_rng(0)
    count = 8192
    ramp = ai.three_approaches(np.linspace(0.0, 1.0, count), 1e-3, 4e6)
    noise = ai.three_approaches(rng.normal(size=count), 1e-3, 4e6)
    assert ramp["algorithmic_below_entropy"]
    assert not noise["algorithmic_below_entropy"]
    assert ramp["compression_against_entropy"] < 0.6
    assert noise["compression_against_entropy"] > 1.0


def test_a_large_bound_is_evidence_of_nothing():
    """The asymmetry is inherent and the module must say so."""
    out = ai.algorithmic(np.random.default_rng(1).normal(size=1024))
    assert out["is_an_upper_bound"]
    assert "upper bound" in out["why"]


def test_a_delay_is_one_number_complex_and_a_waveform_as_a_magnitude():
    """Why the third approach has to be taken in Hilbert space.

    A delay is the structure this arc says is left in the remainder. Its
    description length depends on the representation, and the difference
    is measured rather than asserted.
    """
    out = ai.delay_is_cheap_only_in_the_complex_form()
    assert out["complex_residual_rms"] < 1e-6
    assert out["magnitude_residual_rms"] > 1e-4
    assert out["ratio"] > 20.0
    assert out["parameters_the_complex_form_needed"] == 1


def test_the_range_relative_length_cannot_compare_two_amplitudes():
    """The defect the fixed step exists to remove.

    Scaling each record to its own range makes the length scale-invariant,
    so a model that shrinks the residual earns nothing and the rule it
    feeds goes blind to the improvement it exists to reward.
    """
    rng = np.random.default_rng(0)
    big = rng.normal(0.0, 1.0, 4096)
    small = rng.normal(0.0, 0.1, 4096)
    blind = (ai.algorithmic(big)["bits_per_sample"],
             ai.algorithmic(small)["bits_per_sample"])
    assert abs(blind[0] - blind[1]) < 0.5          # indistinguishable
    seeing = (ai.algorithmic(big, step=0.01)["bits_per_sample"],
              ai.algorithmic(small, step=0.01)["bits_per_sample"])
    assert seeing[0] - seeing[1] > 2.0             # now it costs what it should


def test_a_component_must_pay_for_itself_not_merely_lower_the_residual():
    rng = np.random.default_rng(2)
    before = rng.normal(0.0, 1.0, 4096)
    after = rng.normal(0.0, 0.1, 4096)
    cheap = ai.admits(before, after, parameters=2, step=0.01)
    assert cheap["admitted"]
    assert cheap["bits_saved"] > 0.0
    dear = ai.admits(before, after, parameters=100000, step=0.01)
    assert not dear["admitted"]
    assert "costs more" in dear["verdict"]


def test_a_component_that_changes_nothing_is_refused():
    rng = np.random.default_rng(3)
    values = rng.normal(size=2048)
    out = ai.admits(values, values, parameters=1, step=0.01)
    assert not out["admitted"]
    assert out["bits_saved"] < 0.0


def test_the_parameter_cost_grows_with_the_record_it_was_fitted_on():
    """Half the logarithm of the sample count, the standard code length."""
    short = ai.description_length(np.zeros(64), 1, step=1.0)
    long = ai.description_length(np.zeros(65536), 1, step=1.0)
    assert long["bits_per_parameter"] > short["bits_per_parameter"]
    assert math.isclose(short["bits_per_parameter"], 0.5 * math.log2(64),
                        rel_tol=1e-12)


def test_an_empty_record_is_handled_rather_than_dividing_by_nothing():
    out = ai.algorithmic(np.array([]))
    assert out["samples"] == 0
    assert out["bits"] == 0.0
    assert math.isnan(out["bits_per_sample"])


def test_a_planted_delay_is_recovered_exactly_by_one_number():
    grid = np.linspace(0.2e6, 4.0e6, 1024)
    planted = 3.7e-7
    response = np.exp(-grid / 6e6) * np.exp(-2j * np.pi * grid * planted)
    out = ai.remove_delay(response, grid)
    assert math.isclose(out["delay_s"], planted, rel_tol=1e-6)
    assert out["parameters"] == 1
    # and the corrected response has no ramp left
    residual = np.unwrap(np.angle(out["corrected"]))
    axis = grid - grid.mean()
    slope = float((axis * (residual - residual.mean())).sum()
                  / (axis * axis).sum())
    assert abs(slope) < 1e-12


def test_the_minimum_phase_share_costs_no_parameters():
    """Bode fixes the phase from the magnitude, so it is already paid for."""
    grid = np.linspace(0.2e6, 4.0e6, 512)
    magnitude = np.exp(-grid / 6e6)
    from vhsdecode.models import hypercomplex
    phase = hypercomplex.minimum_phase(np.log(magnitude))
    out = ai.minimum_phase_is_free(magnitude * np.exp(1j * phase))
    assert out["parameters_it_cost"] == 0
    assert out["implied_share"] > 0.9
    assert out["excess_rms_rad"] < 0.2


def test_an_all_pass_is_excess_and_the_magnitude_cannot_imply_it():
    grid = np.linspace(0.2e6, 4.0e6, 512)
    # unit magnitude, so the magnitude implies a phase of zero
    response = np.exp(1j * np.sin(grid / 5e5))
    out = ai.minimum_phase_is_free(response)
    assert out["excess_share"] > 0.8
    assert out["excess_rms_rad"] > 0.3


def test_the_component_language_beats_the_compressor_on_a_delay():
    grid = np.linspace(0.2e6, 4.0e6, 1024)
    response = np.exp(-grid / 6e6) * np.exp(-2j * np.pi * grid * 3.7e-7)
    out = ai.describe(response, grid, step=1e-4)
    assert out["beats_the_compressor"]
    assert out["saved_bits"] > 0.0
    assert [name for name, _, _ in out["kept"]] == ["delay"]


def test_the_language_declines_a_form_that_does_not_pay():
    """A response with no delay in it must not be charged for one."""
    rng = np.random.default_rng(4)
    grid = np.linspace(0.2e6, 4.0e6, 512)
    response = (rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size))
    out = ai.describe(response, grid, step=0.5)
    assert out["saved_bits"] >= 0.0 or not out["kept"]


def test_the_forms_declare_which_ones_are_free():
    forms = ai.component_forms()
    assert forms["minimum phase"]["parameters"] == 0
    assert forms["delay"]["parameters"] == 1
    assert forms["all-pass"]["parameters"] == 2
    assert forms["smooth shape"]["parameters"] is None
