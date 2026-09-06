"""The unknown components live at the joins, and each join has its own prior.

Ethan: *"I think this method goes at the end of each component chain stage,
where the unknown components are most likely to be ... Gausian noise at the rf
stage, multi path reflection at the source stage, etc."*
"""

import numpy as np
import pytest

from vhsdecode.models import stage_boundaries as sb

BAND = np.linspace(1e6, 7e6, 1024)


def _planted(boundary, value, grid=BAND):
    family = sb.candidates(boundary, grid)
    curve = family[min(family, key=lambda v: abs(v - value))]
    logged = (np.log(np.maximum(np.abs(curve), 1e-30))
              + 1j * np.unwrap(np.angle(curve)))
    return logged - logged.mean()


def test_the_joins_are_derived_from_the_declared_chain():
    """A list would drift the moment a position was added; reading the chain
    means a component added anywhere moves the joins with it."""
    joins = sb.boundaries()
    assert joins, "the declared chain must produce joins"
    assert all(j["has_prior"] for j in joins), "every join needs a prior"
    # they are ordered, and each one's bands touch
    for below, above in zip(joins, joins[1:]):
        assert below["above"] == above["below"]
    names = [j["boundary"] for j in joins]
    assert names[0].startswith("the source")
    assert names[-1].endswith("the picture stage")


def test_a_planted_echo_at_a_resolvable_delay_is_identified():
    boundary = "the source -> the transmission path"
    got = sb.estimate(_planted(boundary, 3.0e-6), BAND, boundary)
    assert got["identified"]
    assert got["value"] == pytest.approx(3.0e-6, rel=0.15)
    assert got["explained"] > 0.9


def test_unstructured_noise_is_refused_at_every_join():
    """THE CONTROL THAT THE FIRST GATE FAILED. Gating on `best > 2x median`
    declared noise identified, because thirty-two candidates were searched
    and the luckiest kept."""
    rng = np.random.default_rng(4)
    noise = (rng.normal(size=BAND.size) + 1j * rng.normal(size=BAND.size))
    for join in sb.boundaries():
        got = sb.estimate(noise, BAND, str(join["boundary"]))
        if got.get("identified") is None:
            continue
        assert not got["identified"], join["boundary"]
        assert got["explained"] < got["null_threshold"], join["boundary"]


def test_a_rate_family_identifies_the_family_and_never_the_parameter():
    """`exp(-2 pi L / lambda)` has log magnitude `-2 pi L / lambda`, so every
    member is one shape times a constant. This is the 1.58-of-6 collapse
    arriving at the joins."""
    boundary = "the recording machine -> the tape"
    got = sb.estimate(_planted(boundary, 0.15e-6), BAND, boundary)
    assert got["explained"] == pytest.approx(1.0, abs=1e-6)
    assert got["margin"] == pytest.approx(0.0, abs=1e-9)
    assert not got["identified"], "the parameter is not identifiable"


def test_a_sub_resolution_echo_is_measurable_as_a_delay_not_a_ripple():
    """The band limit is true about RIPPLES and was never true about
    identifiability, which took a wrong claim to establish.

    An echo at tau is a ripple of period 1/tau, so a 6 MHz span cannot
    resolve anything shorter than 167 ns as a ripple - and 81 per cent of
    this join's prior sits below that. But a short echo has a nearly LINEAR
    PHASE, and a linear phase is measurable on a band that cannot resolve the
    ripple which produced it."""
    boundary = "the transmission path -> the recording machine"
    reach = sb.resolvable(boundary, BAND)
    assert reach["shortest_resolvable_s"] == pytest.approx(1.0 / 6e6, rel=1e-6)
    assert reach["unresolvable_fraction"] > 0.75

    got = sb.estimate(_planted(boundary, 8e-8), BAND, boundary)
    assert got["explained"] == pytest.approx(1.0, abs=1e-6)
    assert got["identified"], "a sub-resolution echo is still a delay"
    assert got["median_of_family"] < 0.2, (
        "the family is NOT collinear once the projection carries the phase "
        "separately from the magnitude")


def test_a_join_with_no_residual_is_reported_unmeasured_not_skipped():
    """The joins nobody has looked at are exactly what this structure exists
    to make visible."""
    survey = sb.survey({}, BAND)
    assert survey["count"] == len(sb.boundaries())
    assert survey["measured"] == 0
    assert all(not j["measured"] for j in survey["joins"])


def test_the_projection_is_stacked_not_hermitian():
    """THE BUG THIS CATCHES. `|<u,v>|^2` treats a ninety-degree phase
    difference as full alignment, so a planted MAGNETIC residual - whose log
    is pure real - was explained 1.0000 by a 1 ns ECHO, whose log is pure
    imaginary. They are orthogonal."""
    magnetic = _planted("the recording machine -> the tape", 0.15e-6)
    # the specific pure-delay member, which is what the bug credited with a
    # perfect fit to a pure-magnitude residual
    tiny = sb._echo(BAND, 1e-9)
    shape = (np.log(np.maximum(np.abs(tiny), 1e-30))
             + 1j * np.unwrap(np.angle(tiny)))
    shape = shape - shape.mean()
    pair = np.concatenate([shape.real, shape.imag])
    pair = pair / np.linalg.norm(pair)
    stacked = np.concatenate([magnetic.real, magnetic.imag])
    energy = float(np.vdot(magnetic, magnetic).real)
    explained = float(abs(float(np.dot(pair, stacked))) ** 2 / energy)
    assert explained < 0.01, (
        "a pure-magnitude residual must not be explained by a pure-delay "
        "shape; the Hermitian form gave 1.0000 here")

    # and the longer members of that family legitimately DO share structure
    # with a magnitude roll-off, which is physics rather than the bug
    family = sb.estimate(magnetic, BAND,
                         "the transmission path -> the recording machine")
    assert family["explained"] > 0.5


def test_a_planted_ingress_line_is_identified_and_not_confused_with_an_echo():
    """Ethan: *"Primarily unknown other RF interference components (especially
    on the interconnects between devices)."* A cable collects somebody else's
    carrier - a LINE at an unknown frequency, not a ripple at an unknown
    delay."""
    boundary = "the transmission path -> the recording machine (ingress)"
    found = sb.search_families(_planted(boundary, 4.2e6), BAND, boundary)
    assert found["winner"] == "ingress"
    assert found["agrees_with_the_prior"]
    assert found["over_runner_up"] > 0.5


def test_the_magnitude_families_are_not_separable_from_each_other():
    """Honest limit: over one band a magnetic loss, a pole cascade and the
    skirt of a narrow line are all monotone magnitude roll-offs."""
    found = sb.search_families(
        _planted("the recording machine -> the tape", 0.15e-6), BAND)
    others = [v["explained"] for k, v in found["joins"].items()
              if v["family"] in ("ingress", "noise_and_pole")]
    assert max(others) > 0.9, "they really are that close"
    assert found["winner"] == "magnetic", "but the right one still wins"


def test_the_informed_limit_takes_its_stopping_rule_from_the_noise():
    """The limit needs somewhere to stop, and what the floor IS depends on
    what the noise is."""
    rng = np.random.default_rng(5)
    residual = _planted("the source -> the transmission path", 3e-6)
    boundary = "the source -> the transmission path"
    cases = {
        "gaussian": (rng.normal(0, 1, 20000), True),
        "uniform": (rng.uniform(-0.5, 0.5, 20000), False),
        "arcsine": (np.sin(2 * np.pi * rng.uniform(0, 1, 20000)), False),
    }
    for expected, (samples, reducible) in cases.items():
        got = sb.informed_limit(residual, BAND, boundary, samples=samples)
        assert got["distribution"] == expected
        assert got["floor_is_reducible"] is reducible, expected
