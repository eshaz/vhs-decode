"""Why the amplitudes are solved as a whole.

Ethan's rule, and the reason the picture stage re-solves after its peel:

    Since each correction will affect all the other frequency responses, it
    is critical that this residual is measured and corrected as a whole.

The greedy alternative fits one component, subtracts it, then fits the next
against what is left. That is correct only when the components are
orthogonal over the window. Where two lie close enough to overlap within
the window's resolution it is not: the first fitted takes energy belonging
to the second, and the second is then fitted to a residual that error has
already distorted.

These tests plant known components at known amplitudes and check that the
joint solve recovers them where the sequential one does not. They test the
principle rather than the picture stage's own machinery, because the
principle is what the change rests on.
"""

import numpy as np
import pytest


SAMPLES = 96


def _component(frequency, decay, length=SAMPLES):
    """One damped resonance, of the kind the picture stage fits."""
    n = np.arange(length, dtype=np.float64)
    return np.exp(-n / decay) * np.cos(2.0 * np.pi * frequency * n)


def _solve_jointly(basis, target):
    """All amplitudes at once, over the whole support."""
    amplitudes, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return amplitudes


def _solve_greedily(basis, target):
    """One at a time, deflating after each - what the peel used to leave."""
    residual = target.copy()
    amplitudes = np.zeros(basis.shape[1])
    for column in range(basis.shape[1]):
        vector = basis[:, column]
        amplitudes[column] = float(vector @ residual) / float(vector @ vector)
        residual = residual - amplitudes[column] * vector
    return amplitudes


class TestCloseComponents:
    """The case the rule exists for."""

    @staticmethod
    def _planted():
        # two resonances a quarter of a resolution cell apart - close
        # enough that their columns are strongly correlated over the window
        first = _component(0.070, 30.0)
        second = _component(0.077, 30.0)
        basis = np.column_stack([first, second])
        truth = np.array([1.0, -0.6])
        return basis, truth, basis @ truth

    def test_the_columns_really_do_overlap(self):
        """Without this the test would be proving nothing: a greedy solve
        is exact on an orthogonal basis."""
        basis, _, _ = self._planted()
        a, b = basis[:, 0], basis[:, 1]
        coherence = abs(a @ b) / np.sqrt((a @ a) * (b @ b))
        assert coherence > 0.5, coherence

    def test_the_joint_solve_recovers_both(self):
        basis, truth, target = self._planted()
        recovered = _solve_jointly(basis, target)
        assert recovered == pytest.approx(truth, abs=1e-6)

    def test_the_greedy_solve_mis_splits_them(self):
        """The first fitted takes energy belonging to the second."""
        basis, truth, target = self._planted()
        recovered = _solve_greedily(basis, target)
        assert not np.allclose(recovered, truth, atol=0.05)
        # and specifically, the first one is inflated by the second's energy
        assert abs(recovered[0] - truth[0]) > abs(recovered[1] - truth[1]) / 4

    def test_the_joint_residual_is_smaller(self):
        basis, _, target = self._planted()
        joint = target - basis @ _solve_jointly(basis, target)
        greedy = target - basis @ _solve_greedily(basis, target)
        assert np.linalg.norm(joint) < np.linalg.norm(greedy)


class TestWellSeparatedComponents:
    """The control: where the components do not overlap, the two agree, so
    the joint solve is not simply a different answer - it is the same
    answer wherever the greedy one was entitled to be right."""

    @staticmethod
    def _planted():
        basis = np.column_stack([_component(0.03, 40.0),
                                 _component(0.31, 12.0)])
        truth = np.array([0.8, 0.5])
        return basis, truth, basis @ truth

    def test_both_methods_agree_when_the_basis_is_near_orthogonal(self):
        basis, truth, target = self._planted()
        a, b = basis[:, 0], basis[:, 1]
        assert abs(a @ b) / np.sqrt((a @ a) * (b @ b)) < 0.2
        assert _solve_jointly(basis, target) == pytest.approx(truth, abs=1e-6)
        assert _solve_greedily(basis, target) == pytest.approx(truth, abs=0.05)


class TestOrderIndependence:
    """A joint solve has no ordering. A greedy one does, and that is a
    property of the method rather than of the signal - which is why the
    order components happen to be found in must not set their amplitudes."""

    @staticmethod
    def _planted():
        basis = np.column_stack([_component(0.070, 30.0),
                                 _component(0.077, 30.0),
                                 _component(0.140, 22.0)])
        truth = np.array([1.0, -0.6, 0.4])
        return basis, truth, basis @ truth

    def test_the_joint_answer_does_not_depend_on_the_order(self):
        basis, truth, target = self._planted()
        order = [2, 0, 1]
        forward = _solve_jointly(basis, target)
        shuffled = _solve_jointly(basis[:, order], target)
        assert forward[order] == pytest.approx(shuffled, abs=1e-6)

    def test_the_greedy_answer_does(self):
        """Same components, same data, different order of discovery -
        different amplitudes. On this planted set the disagreement is
        about 0.04 on an amplitude of 0.4, which is a tenth of the
        quantity, and it is attributable entirely to the order."""
        basis, _, target = self._planted()
        order = [2, 0, 1]
        forward = _solve_greedily(basis, target)
        shuffled = _solve_greedily(basis[:, order], target)
        spread = float(np.max(np.abs(forward[order] - shuffled)))
        assert spread > 1e-3, spread


# THE WIRING TESTS THAT STOOD HERE ARE GONE WITH THEIR SUBJECT. They read
# the source of `addons/ringing_cancellation.fit_artifact_model` to check
# that the picture stage solved its component amplitudes jointly rather
# than greedily. That module was deleted on 2026-09-06 and its correction
# replaced by `vhsdecode/models/ringing_tesseract`, which fits no component
# amplitudes at all: the kernel IS the measured departure, folded on the
# tesseract's orthogonal contrasts, so there is no set of amplitudes for a
# greedy pass to distort. The principle the rest of this file tests still
# stands and is why the replacement does not fit components one at a time.
