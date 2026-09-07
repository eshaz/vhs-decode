"""What remains when every modelled component is exhausted.

Ethan's framework: the state space splits into a modelable subspace
governed by deterministic electrodynamics and its orthogonal complement,
the modelable part is taken by minimum-norm projection, and what is left
is bound by the micromagnetics rather than left as arbitrary noise.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import modelable_subspace as ms
from vhsdecode.models import vhs_specification

COERCIVITY = vhs_specification.SMPTE_32M["coercivity_a_m"]["value"]
TRACK_WIDTH = vhs_specification.SMPTE_32M["track_width_sp_m"]["value"]
WRITING_SPEED = 5.8


def test_the_formula_and_its_generalisation_agree_where_both_exist():
    """His section 3 is the pseudo-inverse whenever A A* can be inverted."""
    rng = np.random.default_rng(0)
    operator = rng.normal(size=(6, 20))
    data = rng.normal(size=6)
    out = ms.minimum_norm(operator, data)
    assert out["regime"]["underdetermined"]
    assert out["by_formula"] is not None
    assert out["agree"]
    # and it does solve the constraint exactly, which is what makes it
    # the minimum-norm solution rather than a least-squares one
    assert np.allclose(operator @ out["by_pseudo_inverse"], data, atol=1e-10)


def test_the_formula_is_withheld_when_it_does_not_apply():
    """Overdetermined: A A* is singular and the formula has no meaning."""
    rng = np.random.default_rng(1)
    out = ms.minimum_norm(rng.normal(size=(50, 6)), rng.normal(size=50))
    assert not out["regime"]["underdetermined"]
    assert out["by_formula"] is None
    assert out["by_pseudo_inverse"].shape == (6,)


def test_a_state_inside_the_subspace_leaves_nothing():
    rng = np.random.default_rng(2)
    operator = rng.normal(size=(80, 5))
    inside = operator @ np.array([1.3, -0.4, 0.0, 2.1, 0.7])
    out = ms.decompose(operator, inside)
    assert out["remainder_share"] < 1e-20
    assert math.isclose(out["modelled_share"], 1.0, abs_tol=1e-12)


def test_a_state_outside_the_subspace_survives_intact():
    """The complement is what the operator cannot reach, by construction."""
    rng = np.random.default_rng(3)
    operator = rng.normal(size=(80, 5))
    left, _, _ = np.linalg.svd(operator, full_matrices=True)
    outside = left[:, 7]
    out = ms.decompose(operator, outside)
    assert out["modelled_share"] < 1e-20
    assert math.isclose(out["remainder_share"], 1.0, abs_tol=1e-10)


def test_the_rank_cut_is_reported_rather_than_hidden():
    """A duplicated column is a null direction, and it must be named.

    This arc measured the magnetic mechanisms collapsing to 1.58
    distinguishable of six, so rank deficiency is the normal case and a
    projection that silently inverts through it reports a small residual
    made of amplified noise.
    """
    rng = np.random.default_rng(4)
    operator = rng.normal(size=(40, 6))
    doubled = np.column_stack([operator, operator[:, 0]])
    out = ms.decompose(doubled, rng.normal(size=40))
    assert out["rank_deficient"]
    assert out["rank"] == 6
    assert out["components"] == 7
    assert math.isfinite(out["condition"])


def test_the_operator_keeps_amplitude_and_phase_apart():
    """Stacked real, a shape and the same shape rotated are two directions."""
    grid = np.linspace(0.2e6, 4.0e6, 64)
    shape = np.exp(-grid / 2e6).astype(np.complex128)
    built = ms.build_operator({"amplitude": shape, "phase": 1j * shape})
    operator = built["operator"]
    assert operator.shape == (128, 2)
    assert np.linalg.matrix_rank(operator) == 2
    # under a Hermitian inner product they would have been one direction
    hermitian = abs(np.vdot(shape, 1j * shape)) / (np.vdot(shape, shape).real)
    assert math.isclose(hermitian, 1.0, rel_tol=1e-12)


def test_the_operator_refuses_an_empty_key():
    with pytest.raises(ValueError):
        ms.build_operator({})


def test_the_anisotropy_is_derived_from_the_specified_coercivity():
    out = ms.anisotropy_j_m3(COERCIVITY)
    assert math.isclose(
        out["anisotropy_j_m3"],
        0.5 * ms.VACUUM_PERMEABILITY * ms.SATURATION_MAGNETISATION_A_M
        * COERCIVITY, rel_tol=1e-12)
    # and it scales with the coercivity, which is the specified quantity
    doubled = ms.anisotropy_j_m3(2.0 * COERCIVITY)["anisotropy_j_m3"]
    assert math.isclose(doubled, 2.0 * out["anisotropy_j_m3"], rel_tol=1e-12)


def test_the_domain_wall_is_pi_times_the_exchange_length():
    out = ms.micromagnetic_lengths(COERCIVITY)
    assert math.isclose(out["domain_wall_width_m"],
                        math.pi * out["exchange_length_m"], rel_tol=1e-12)
    assert 0.0 < out["quality_factor"] < 1.0     # shape-dominated, acicular


def test_no_micromagnetic_length_lands_inside_the_vhs_band():
    """The finding, and it is a negative one.

    Every length of the medium sits above the band the format records, so
    on this format the micromagnetics constrain nothing that was recorded.
    The tightest is the particle, and it is more than three times above.
    """
    out = ms.physical_bound(COERCIVITY, WRITING_SPEED, (0.4e6, 4.4e6),
                            gap_m=0.3e-6)
    assert not out["any_inside"]
    assert out["inside_the_band"] == {}
    assert out["binding"] == "particle length"
    assert out["headroom"] > 3.0
    # the ordering is the physics: exchange above wall above particle
    order = sorted(out["frequencies_hz"].items(), key=lambda item: item[1])
    assert [name for name, _ in order][-1] == "exchange length"


def test_a_slower_writing_speed_brings_the_lengths_into_reach():
    """The bound is a real constraint, not one that can never bind."""
    fast = ms.physical_bound(COERCIVITY, 5.8, (0.4e6, 4.4e6))
    slow = ms.physical_bound(COERCIVITY, 0.58, (0.4e6, 4.4e6))
    assert fast["binding_hz"] > slow["binding_hz"]
    assert slow["any_inside"]


def test_the_particulate_floor_is_one_number_not_two():
    """The amplitude ratio is the root of the count, so the decibels agree."""
    out = ms.particulate_floor(TRACK_WIDTH, 0.2064e-6, WRITING_SPEED / 3.9e6)
    assert math.isclose(out["snr_db"],
                        20.0 * math.log10(out["amplitude_snr"]), rel_tol=1e-12)
    assert 30.0 < out["snr_db"] < 45.0


def test_a_longer_wavelength_holds_more_particles():
    short = ms.particulate_floor(TRACK_WIDTH, 0.2064e-6, WRITING_SPEED / 4.4e6)
    long = ms.particulate_floor(TRACK_WIDTH, 0.2064e-6, WRITING_SPEED / 0.629e6)
    assert long["particles_per_cell"] > short["particles_per_cell"]
    assert long["snr_db"] > short["snr_db"] + 5.0


def test_a_remainder_at_the_floor_is_declared_finished():
    floor = ms.particulate_floor(TRACK_WIDTH, 0.2064e-6, WRITING_SPEED / 3.9e6)
    at = ms.against_the_floor(1.0 / floor["particles_per_cell"], 1.0, floor)
    assert at["at_the_floor"]
    assert abs(at["ratio_db"]) < 0.01
    above = ms.against_the_floor(0.02, 1.0, floor)
    assert not above["at_the_floor"]
    assert above["ratio_db"] > 10.0


def test_exhausted_runs_the_whole_framework_on_one_response():
    """A response built only from key entries leaves nothing to survive."""
    from vhsdecode.models import residual_floor
    grid = np.linspace(0.2e6, 4.0e6, 400)
    key = residual_floor.keys_for(grid)["base"]
    names = list(key)
    logged = 0.9 * np.asarray(key[names[0]]) - 0.4 * np.asarray(key[names[5]])
    response = np.exp(logged)
    out = ms.exhausted(response, grid, key)
    assert out["bins_used"] == grid.size
    assert out["decomposition"]["remainder_share"] < 1e-12
    assert out["verdict"]["at_the_floor"]


def test_exhausted_leaves_what_the_key_cannot_reach():
    from vhsdecode.models import residual_floor
    grid = np.linspace(0.2e6, 4.0e6, 400)
    key = residual_floor.keys_for(grid)["base"]
    logged = (0.9 * np.asarray(key[list(key)[0]])
              + 0.3 * np.cos(np.linspace(0.0, 41.0 * np.pi, grid.size)))
    out = ms.exhausted(np.exp(logged), grid, key)
    assert out["decomposition"]["remainder_share"] > 1e-3
    assert not out["verdict"]["at_the_floor"]
    assert out["verdict"]["ratio_db"] > 0.0


def test_the_rank_cut_follows_the_datas_own_noise():
    """A direction the measurement cannot see is not a direction it has."""
    from vhsdecode.models import residual_floor
    grid = np.linspace(0.2e6, 4.0e6, 400)
    key = residual_floor.keys_for(grid)["base"]
    response = np.exp(0.9 * np.asarray(key[list(key)[0]]))
    loose = ms.exhausted(response, grid, key, relative_error=1e-9)
    tight = ms.exhausted(response, grid, key, relative_error=1e-1)
    assert loose["decomposition"]["rank"] > tight["decomposition"]["rank"]
    assert (loose["decomposition"]["unreachable_share"]
            <= tight["decomposition"]["unreachable_share"])


def test_exhausted_refuses_a_response_with_nothing_in_it():
    grid = np.linspace(0.2e6, 4.0e6, 4)
    with pytest.raises(ValueError):
        ms.exhausted(np.zeros(4, dtype=complex), grid, {"a": np.ones(4)})
