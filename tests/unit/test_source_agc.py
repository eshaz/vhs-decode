"""The source's gain control, derived by asserting the levels.

Ethan: 'AGC can be derived by asserting the levels should be constant, and
back porch should be ire 0.' And on where it sits: 'between the
propagation model and the VCR input model.'
"""

import numpy as np
import pytest

from vhsdecode.models import source_agc as ag


def test_it_sits_between_the_propagation_terms_and_the_recorders_own_agc():
    from vhsdecode.models import interference

    chain = interference.full_chain(strict=False)
    assert chain["echo"] < ag.CHAIN_POSITION < chain["vcr agc"]
    assert chain["tuner clipping"] < ag.CHAIN_POSITION


def test_the_two_observables_separate_a_gain_from_an_offset():
    """A gain cannot move a zero and an offset cannot change a spacing."""
    out = ag.assert_levels([0.0, 0.0], [40.0, 40.0])
    assert out["offset_mean"] == 0.0 and out["gain_mean"] == 1.0
    shifted = ag.assert_levels([-2.0, -2.0], [40.0, 40.0])
    assert shifted["offset_mean"] == -2.0
    assert shifted["gain_mean"] == 1.0          # a shift is not a gain
    squeezed = ag.assert_levels([0.0, 0.0], [34.0, 34.0])
    assert squeezed["offset_mean"] == 0.0       # a gain is not a shift
    assert squeezed["gain_mean"] == pytest.approx(0.85)


def test_the_control_falls_fast_and_recovers_slowly():
    line = 1.0 / (525 * 30 / 1.001)
    drive = np.ones(4000)
    drive[1000:] = 0.8
    out = ag.response(drive, line, attack_s=2e-3, decay_s=200e-3)
    # the fall is quick: within three attack constants
    fallen = np.argmax(out[1000:] < 0.81)
    assert 0 < fallen < 4.0 * 2e-3 / line
    # and a rise from the same state is far slower
    back = np.concatenate([np.full(1000, 0.8), np.ones(3000)])
    up = ag.response(back, line, attack_s=2e-3, decay_s=200e-3)
    risen = np.argmax(up[1000:] > 0.99)
    assert risen == 0 or risen > 10 * fallen


def test_with_the_drive_known_both_constants_are_recovered():
    line = 1.0 / (525 * 30 / 1.001)
    drive = np.ones(6000)
    drive[1500:3000] = 0.8
    observed = ag.response(drive, line, attack_s=2e-3, decay_s=200e-3)
    out = ag.solve_with_drive(observed, drive, line)
    assert out["attack_s"] == pytest.approx(2e-3, rel=0.2)
    assert out["decay_s"] == pytest.approx(200e-3, rel=0.3)


def test_without_the_drive_only_the_ordering_survives():
    """Verified against planted controls: a ratio of 10 reads 1.0 and one
    of 100 reads 4.3, so the statistic orders and does not measure."""
    line = 1.0 / (525 * 30 / 1.001)
    drive = np.ones(6000)
    drive[1500:3000] = 0.8
    gentle = ag.fit(ag.response(drive, line, 2e-3, 20e-3), line)
    sharp = ag.fit(ag.response(drive, line, 2e-3, 200e-3), line)
    assert sharp["asymmetry"] > gentle["asymmetry"]
    assert sharp["falls_faster"]
    assert sharp["asymmetry"] < 100.0            # it is not the ratio


def test_a_level_gain_is_a_deviation_error_once_the_signal_is_modulated():
    out = ag.levels_as_frequencies(0.8402)
    assert out["deviation_hz"] == pytest.approx(840.2e3)
    assert out["deviation_error_hz"] == pytest.approx(-159.8e3)
    assert out["peak_white_hz"] == pytest.approx(4.2402e6)
    assert out["peak_white_error_hz"] == pytest.approx(-159.8e3)
    assert "3.9.1.1.4" in out["cite"]
    assert ag.levels_as_frequencies(1.0)["deviation_error_hz"] == pytest.approx(0.0)
