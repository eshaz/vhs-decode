"""The tape tensor: Kruskal uniqueness with the tape as a third axis.

Pins three things. `k_rank` and `kruskal` are arithmetic and must not
drift. `cp_decompose` must reproduce a NOISELESS planted tensor to machine
precision in a handful of rounds, real and complex alike - the history of
this solver is three defects that each left the fit error stuck (1.0 with
the Khatri-Rao order backwards, 0.9 with the held factors normalised,
3.2e-3 with alternating least squares crawling from a singular-vector
start on a two-head tensor), and each looked like slow convergence rather
than a bug. And `rotational_ambiguity` must show the claim: a matrix fit
gets the subspace and not the components, a tensor fit with three tapes
gets the components, and the control with two tapes does not.
"""

import numpy as np
import pytest

from vhsdecode.models import tape_tensor as tt


def test_k_rank_is_the_largest_all_independent_column_count():
    assert tt.k_rank(np.eye(3)) == 3
    dup = np.column_stack([np.eye(3)[:, 0], np.eye(3)[:, 0], np.eye(3)[:, 1]])
    assert tt.k_rank(dup) == 1


def test_kruskal_condition_two_heads_three_tapes_rank_three():
    rng = np.random.default_rng(0)
    A, B, C = (rng.standard_normal((16, 3)), rng.standard_normal((2, 3)),
               rng.standard_normal((3, 3)))
    verdict = tt.kruskal((A, B, C))
    assert verdict["sum"] == 8 and verdict["needed"] == 8
    assert verdict["unique"]
    assert not tt.kruskal((A, B, C[:2]))["unique"]


@pytest.mark.parametrize("complex_valued", [False, True])
def test_noiseless_planted_tensor_is_reproduced_to_machine_precision(
        complex_valued):
    rng = np.random.default_rng(3)

    def factor(shape):
        value = rng.standard_normal(shape)
        if complex_valued:
            value = value + 1j * rng.standard_normal(shape)
        return value

    A0, B0, C0 = factor((64, 3)), factor((2, 3)), factor((3, 3))
    X = np.einsum("ir,jr,kr->ijk", A0, B0, C0)
    fit = tt.cp_decompose(X, 3)
    assert fit["relative_error"] < 1e-8
    assert fit["iterations"] <= 10
    for recovered, planted in zip(fit["factors"], (A0, B0, C0)):
        assert tt.match_components(recovered, planted)["worst_component"] > 0.999999


def test_gevd_start_is_exact_and_falls_back_when_shape_forbids_it():
    rng = np.random.default_rng(5)
    A0, B0, C0 = (rng.standard_normal((32, 3)), rng.standard_normal((2, 3)),
                  rng.standard_normal((3, 3)))
    X = np.einsum("ir,jr,kr->ijk", A0, B0, C0)
    start = tt.gevd_initial(X, 3)
    assert start is not None
    assert tt.match_components(start[0], A0)["worst_component"] > 0.999999
    # two tapes cannot compress the tape axis to rank three: no start
    assert tt.gevd_initial(X[:, :, :2], 3) is None


def test_the_demonstration_holds_and_its_control_fails():
    result = tt.rotational_ambiguity(noise=1e-3)
    assert result["matrix"]["subspace"] > 0.99
    assert result["matrix"]["components"] < 0.95
    assert result["tensor"]["components"] > 0.99
    assert result["tensor"]["kruskal"]["unique"]
    assert result["control"]["components"] < 0.95
    assert not result["control"]["kruskal"]["unique"]
    assert result["claim_holds"]


def test_identification_degrades_with_noise_and_the_ceiling_is_recorded():
    # at one per cent of the peak the tensor still identifies; at three it
    # no longer clears the gate. This is the synthetic's ceiling for 256
    # frequencies x 2 heads x 3 tapes, recorded so a change is noticed.
    assert tt.rotational_ambiguity(noise=1e-2)["tensor"]["components"] > 0.99
    assert tt.rotational_ambiguity(noise=3e-2)["tensor"]["components"] < 0.99
