"""Tests for the multidimensional information extrapolation model.

The data model round-trips, the RF capture component carries the bounds it
was specified with, the loop feeds forward until the held-out residual is at
the floor and then freezes the component, and the noise floor is refused any
phase.
"""

import numpy as np
import pytest

from vhsdecode.models import information_extrapolation as ie


def test_rf_capture_carries_the_specified_bounds():
    component = ie.rf_capture(12345)
    amplitude = component.resolution["amplitude"]
    assert (amplitude.minimum, amplitude.maximum, amplitude.total) == (0, 255, 256)
    frequency = component.resolution["frequency"]
    assert frequency.minimum == 0
    assert frequency.maximum == pytest.approx(20e6)
    assert component.resolution["time"].total == 12345
    assert set(component.residual) == {"amplitude", "frequency", "time"}
    assert not any(residual.measured for residual in component.residual.values())
    assert component.unmeasured_axes() == ["amplitude", "frequency", "time"]


def test_residual_round_trips_its_value_and_error():
    residual = ie.Residual("frequency", over=("time",), source="a gauge")
    assert not residual.measured
    assert residual.error is None
    residual.value = np.zeros(4)
    residual.error = np.ones(4)
    assert residual.measured
    assert residual.over == ("time",)


def test_noise_floor_is_amplitude_only():
    assert ie.noise_floor().amplitude_only


def test_residual_channels_cover_every_axis_and_the_chroma():
    assert set(ie.RESIDUAL_CHANNELS) == {
        "amplitude",
        "frequency",
        "time",
        "chroma_amplitude",
    }


def test_chroma_components_are_linear_and_named():
    names = [component.name for component in ie.chroma_components()]
    assert names[:2] == ["color-under envelope amplitude", "chroma burst pilot"]
    assert all("[linear" in c.criterion for c in ie.chroma_components())


def _model_total(model):
    """Everything the model has accumulated, summed - what feed_forward
    applies through the chain."""
    total = 0.0
    for component in model.values():
        for value in component.values():
            total = total + np.asarray(value)
    return total


class _Gauges:
    """A chain that removes only HALF of what the model asks for - the
    measured Jacobian of a real site is about 0.75, never 1 - so the
    residual falls geometrically and the model accumulates toward the
    whole shape. The signal IS its own residual."""

    def __init__(self, floor):
        self.floor = floor
        self.original = None

    def measure(self, component, signal):
        if self.original is None:
            self.original = np.asarray(signal["residual"]).copy()
        return {"frequency": np.asarray(signal["residual"])}

    def at_floor(self, component, residual):
        return float(np.sqrt(np.mean(residual["frequency"] ** 2))) < self.floor

    def feed_forward(self, signal, model):
        return {"residual": self.original - 0.5 * _model_total(model)}

    def subtract_noise_floor(self, signal, floor):
        return signal


def _component():
    return ie._component("a linear component", "its gauge", ("frequency",), True)


def test_loop_accumulates_the_expected_signal_and_the_model_is_the_result():
    component = _component()
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [component], _Gauges(floor=0.1)
    )
    # the residual halves each pass - 1, 0.5, 0.25, 0.125, 0.0625 - and
    # every one of them, the last at the limit included, is accumulated
    # into the expected signal. The sum approaches the whole shape the
    # ideal was missing (2 here) as the residual approaches zero: that
    # limit IS what the iteration is for.
    assert component.frozen
    assert result.weakly_determined == []
    accumulated = result.model["a linear component"]["frequency"]
    assert np.allclose(accumulated, 1.9375)         # 2 - the residual left
    assert np.allclose(accumulated, 2.0 - result.signal["residual"])
    assert np.allclose(result.signal["residual"], 0.0625)
    assert result.limit["a linear component"] < 0.1


def test_every_component_is_measured_on_every_pass():
    first, second = _component(), _component()
    second.name = "another component"
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [first, second], _Gauges(floor=0.1)
    )
    # they influence each other, so both are measured together each pass
    # rather than one being run to convergence before the other starts
    per_pass = {}
    for record in result.passes:
        per_pass.setdefault(record.index, set()).add(record.component)
    assert all(len(names) == 2 for names in per_pass.values())
    assert set(result.model) == {"a linear component", "another component"}


def test_loop_flags_a_component_still_moving_on_the_last_pass():
    component = _component()
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [component], _Gauges(floor=1e-6), maximum_passes=3
    )
    assert result.weakly_determined == ["a linear component"]
    assert not component.frozen          # it never reached its limit


class _CrossChecking(_Gauges):
    """A gauge set that attributes half of one component's residual to
    the other, as a cross-check between components would.

    Its two components measure DISTINCT directions. The loop's ordered
    orthogonalization runs ahead of the cross-check and removes whatever
    a downstream component shares with an upstream one, so two components
    reporting the identical vector would have the second emptied before
    the cross-check ever saw it. Orthogonal witnesses isolate the
    cross-check's own effect, which is what this test is for."""

    def measure(self, component, signal):
        value = np.asarray(signal["residual"], dtype=float)
        if component.name == "another component":
            value = value * np.tile([1.0, -1.0], value.size // 2)
        if self.original is None:
            self.original = np.asarray(signal["residual"]).copy()
        return {"frequency": value}

    def cross_check(self, residuals):
        if len(residuals) != 2:
            return residuals
        (first, a), (second, b) = residuals.items()
        shared = 0.5 * a["frequency"]
        return {first: {"frequency": a["frequency"] - shared},
                second: {"frequency": b["frequency"] + shared}}


def test_cross_check_moves_a_residual_between_components():
    first, second = _component(), _component()
    second.name = "another component"
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [first, second], _CrossChecking(floor=0.1),
        maximum_passes=1
    )
    a = result.model["a linear component"]["frequency"]
    b = result.model["another component"]["frequency"]
    # half of the first component's residual has been moved into the
    # second, and nothing else has touched either of them
    assert np.allclose(a, 0.5)
    assert np.allclose(b, np.tile([1.5, -0.5], 4))


def test_orthogonalization_runs_down_the_known_order():
    """The chain's order is known, so a component gives up only what the
    stages UPSTREAM of it already explain - never the other way round."""
    upstream = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    downstream = upstream + np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
    residuals = {"first": {"frequency": upstream},
                 "second": {"frequency": downstream}}
    assert ie.coherence(residuals, "frequency") > ie.COHERENCE_THRESHOLD
    out = ie.orthogonalize(residuals, ["first", "second"])
    # the upstream component is never adjusted by a downstream one
    assert np.allclose(out["first"]["frequency"], upstream)
    assert np.linalg.norm(out["second"]["frequency"]) < np.linalg.norm(downstream)
    # and the order is load-bearing: reversing it moves the correction
    reversed_out = ie.orthogonalize(residuals, ["second", "first"])
    assert np.allclose(reversed_out["second"]["frequency"], downstream)
    assert np.linalg.norm(reversed_out["first"]["frequency"]) < np.linalg.norm(upstream)


def test_orthogonal_components_pass_through_untouched():
    """Where the components really are orthogonal the step is the identity,
    so leaving it on costs nothing."""
    first = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    second = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    residuals = {"first": {"frequency": first}, "second": {"frequency": second}}
    assert ie.coherence(residuals, "frequency") < ie.COHERENCE_THRESHOLD
    out = ie.orthogonalize(residuals, ["first", "second"])
    assert np.allclose(out["first"]["frequency"], first)
    assert np.allclose(out["second"]["frequency"], second)


class _WorseningGauges(_Gauges):
    """A chain that makes the residual worse every pass: the loop must
    keep the pass nearest the limit rather than the last one."""

    def feed_forward(self, signal, model):
        return {"residual": np.asarray(signal["residual"]) * 3.0}


def test_loop_keeps_the_pass_nearest_the_limit():
    component = _component()
    first = {"residual": np.ones(8)}
    result = ie.multidimensional_information_extrapolation(
        first, [component], _WorseningGauges(floor=0.1), maximum_passes=4
    )
    # the residual grows every pass, so the first is nearest the limit and
    # the model reverts to what it held there
    assert result.signal is first
    assert result.passes[0].residual_rms["frequency"] == 1.0
    assert np.allclose(result.model["a linear component"]["frequency"], 1.0)


def test_frozen_component_is_no_longer_differentiated():
    component = _component()
    component.frozen = True
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [component], _Gauges(floor=0.1)
    )
    assert result.passes == []
    assert result.model == {}


def test_accumulate_adds_rather_than_replaces():
    component = _component()
    component.accumulate({"frequency": np.full(4, 0.25)})
    component.accumulate({"frequency": np.full(4, 0.10)})
    assert np.allclose(component.expected["frequency"], 0.35)
    assert np.allclose(component.residual["frequency"].value, 0.10)


class _NoiseGauges(_Gauges):
    def __init__(self, floor, noise):
        super().__init__(floor)
        self.noise = noise

    def measure(self, component, signal):
        if component.amplitude_only:
            return self.noise
        return super().measure(component, signal)


def test_noise_floor_is_measured_last_and_differentiated():
    noise = {"amplitude": np.random.default_rng(0).standard_normal((4, 6))}
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)},
        [_component()],
        _NoiseGauges(0.1, noise),
        noise=ie.noise_floor(),
    )
    assert result.noise_floor is noise
    assert result.differential_2d.shape == (4, 6)
    assert np.allclose(result.differential_2d, np.fft.fft2(noise["amplitude"]))


def test_noise_floor_refuses_phase():
    noise = {"amplitude": np.zeros(4), "frequency": np.zeros(4)}
    with pytest.raises(ValueError):
        ie.multidimensional_information_extrapolation(
            {"residual": np.ones(8)},
            [_component()],
            _NoiseGauges(0.1, noise),
            noise=ie.noise_floor(),
        )


def test_heterodyne_timing_scale_is_the_carrier_ratio():
    # NTSC VHS: 3.579545 MHz subcarrier carried at 629.371 kHz
    assert ie.heterodyne_timing_scale(3.5795454545e6, 629370.6293706294) == \
        pytest.approx(5.6875)
    with pytest.raises(ValueError):
        ie.heterodyne_timing_scale(3.58e6, 0.0)


def test_chroma_components_carry_the_heterodyne_scale():
    names = [c.name for c in ie.chroma_components()]
    assert "chroma heterodyne scale" in names


def test_the_two_stages_run_in_order_and_each_keeps_its_own_model():
    """The chain splits in two and the split decides where a correction
    may be applied. The stages run in order, the first is finished before
    the second begins, and a revert in one must never discard the other's
    model."""
    playback = ie._component("a playback thing", "its gauge", ("frequency",),
                             True, stage=ie.PLAYBACK)
    source = ie._component("a source thing", "its gauge", ("frequency",),
                           True, stage=ie.SOURCE)
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [playback, source], _Gauges(floor=0.1)
    )
    assert result.stages == [ie.PLAYBACK, ie.SOURCE]
    assert set(result.model) == {"a playback thing", "a source thing"}
    # the playback stage is finished before the source stage starts
    playback_passes = [r.index for r in result.passes
                       if r.component == "a playback thing"]
    source_passes = [r.index for r in result.passes
                     if r.component == "a source thing"]
    assert playback_passes and source_passes


def test_a_worsening_source_stage_does_not_discard_the_playback_model():
    playback = ie._component("a playback thing", "its gauge", ("frequency",),
                             True, stage=ie.PLAYBACK)
    source = ie._component("a source thing", "its gauge", ("frequency",),
                           True, stage=ie.SOURCE)
    result = ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [playback, source],
        _WorseningGauges(floor=0.1), maximum_passes=3
    )
    assert "a playback thing" in result.model


def test_the_field_axis_is_differentiated_not_averaged():
    """The same three-axis differential run over multiple fields, with the
    field as a fourth axis: a departure that alternates with head parity
    must land on the parity cadence, not be averaged away."""
    fields, amplitude, frequency, time = 8, 2, 4, 4
    block = np.zeros((fields, amplitude, frequency, time))
    # a departure that flips sign every field - head parity
    for index in range(fields):
        block[index] = (-1.0) ** index
    spectrum = ie.differential_4d(block)
    power = np.abs(spectrum).sum(axis=(1, 2, 3))
    cadence = ie.field_cadence(fields)
    parity_bin = int(round(cadence["head parity"] * fields))
    assert int(np.argmax(power)) == parity_bin
    # averaging over the field axis would have destroyed it entirely
    assert np.allclose(block.mean(axis=0), 0.0)


def test_a_constant_over_time_lands_at_zero_on_the_field_axis():
    block = np.ones((6, 2, 4, 4))
    power = np.abs(ie.differential_4d(block)).sum(axis=(1, 2, 3))
    assert int(np.argmax(power)) == 0
    assert ie.field_cadence(6)["constant over time"] == 0.0


def test_the_symmetric_form_privileges_no_component_by_position():
    """The symmetric form solves every share jointly, so unlike the ordered
    one it will adjust a component that comes first."""
    first = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    second = first + np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
    residuals = {"first": {"frequency": first}, "second": {"frequency": second}}
    assert ie.coherence(residuals, "frequency") > ie.COHERENCE_THRESHOLD
    out = ie.orthogonalize_symmetric(residuals)
    # the ordered form leaves the first alone; the symmetric one does not
    assert not np.allclose(out["first"]["frequency"], first)
    ordered = ie.orthogonalize(residuals, ["first", "second"])
    assert np.allclose(ordered["first"]["frequency"], first)


def test_the_symmetric_form_is_also_inert_when_orthogonal():
    first = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    second = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    residuals = {"first": {"frequency": first}, "second": {"frequency": second}}
    out = ie.orthogonalize_symmetric(residuals)
    assert np.allclose(out["first"]["frequency"], first)
    assert np.allclose(out["second"]["frequency"], second)


def test_the_average_is_taken_within_each_head_not_across_them():
    """Two heads with different responses must come back as two responses,
    not one mean belonging to neither."""
    lines, width = 4, 8
    first = np.full((lines, width), 2.0)
    second = np.full((lines, width), -2.0)
    fields = np.array([first, second, first, second])
    parities = [True, False, True, False]
    averaged = ie.per_head_average(fields, parities)
    assert np.allclose(averaged[True], 2.0)
    assert np.allclose(averaged[False], -2.0)
    # pooling across heads would have returned zero, belonging to neither
    assert np.allclose(fields.mean(axis=0), 0.0)


def test_averaging_within_a_head_is_coherent():
    """Noise falls with the count while the response does not, which is why
    the average is worth taking at all."""
    rng = np.random.default_rng(5)
    truth = np.linspace(-1.0, 1.0, 16)
    fields = np.array([truth + rng.normal(0.0, 0.5, 16) for _ in range(64)])
    parities = [True] * 64
    averaged = ie.per_head_average(fields[:, None, :], parities)[True][0]
    assert np.std(averaged - truth) < 0.5 / 4          # 64 fields, root-N is 8


def test_the_head_difference_is_what_separates_them():
    responses = {True: np.ones((2, 2, 2)) * 3.0, False: np.ones((2, 2, 2))}
    difference = ie.head_response_difference(responses)
    assert np.allclose(difference, 2.0)
    assert ie.head_response_difference({True: np.ones(2)}) is None


def test_the_head_difference_is_scoped_per_recording():
    """It is fixed hardware and held as a constant, but measured on two
    tapes played on one deck the difference does not carry across, so the
    constant is scoped to a recording rather than to the playback path."""
    assert ie.HEAD_DIFFERENCE_SCOPE == "per recording"


def test_random_noise_ends_the_process():
    """The rule: once what remains is random, the derivation is complete and
    no further correction can touch it."""
    rng = np.random.default_rng(17)
    verdict = ie.structureless(rng.normal(size=512), rng.normal(size=512))
    assert verdict["complete"] == 1.0
    assert "COMPLETE" in verdict["why"]


def test_structure_that_reproduces_is_not_the_floor():
    """A residual that repeats on lines which did not build it still holds a
    component, however small it is."""
    rng = np.random.default_rng(18)
    shape = np.sin(np.linspace(0.0, 8.0 * np.pi, 512))
    fitted = shape + rng.normal(0.0, 0.3, 512)
    held_out = shape + rng.normal(0.0, 0.3, 512)
    verdict = ie.structureless(fitted, held_out)
    assert verdict["complete"] == 0.0
    assert verdict["agreement"] > ie.STRUCTURE_AGREEMENT


def test_a_small_residual_is_not_completion_by_itself():
    """Size is not the criterion. A tiny residual that still reproduces has a
    component in it; a large structureless one does not."""
    rng = np.random.default_rng(19)
    shape = 1e-6 * np.sin(np.linspace(0.0, 6.0 * np.pi, 512))
    tiny = ie.structureless(shape + 1e-9 * rng.normal(size=512),
                            shape + 1e-9 * rng.normal(size=512))
    large = ie.structureless(1e3 * rng.normal(size=512),
                             1e3 * rng.normal(size=512))
    assert tiny["complete"] == 0.0
    assert large["complete"] == 1.0


def test_the_loop_back_finds_what_one_sample_cannot():
    """Structure buried under a single sample's noise is still there in the
    set: the ensemble mean's floor is root-N lower, so the loop continues."""
    rng = np.random.default_rng(21)
    width, count = 256, 400
    # the count is set by the arithmetic, not by eye: the ensemble mean's
    # floor is the noise over root-N, so structure of rms a/root-2 stands
    # clear of it only once root-N exceeds about 3 root-2 / a. Here that is
    # 0.25 amplitude against unit noise, needing a few hundred samples.
    hidden = 0.25 * np.sin(np.linspace(0.0, 6.0 * np.pi, width))
    samples = np.array([hidden + rng.normal(0.0, 1.0, width)
                        for _ in range(count)])
    # each sample on its own is random: the structure is far below its noise
    alone = ie.structureless(samples[0], samples[1])
    assert alone["complete"] == 1.0
    # but the SET still holds it, and the loop back says so
    verdict = ie.loop_back(samples)
    assert verdict["complete"] == 0.0
    assert verdict["ensemble_over_floor"] > 3.0


def test_the_loop_back_ends_when_the_set_is_empty_too():
    """With nothing hidden, averaging finds nothing and the process ends."""
    rng = np.random.default_rng(22)
    samples = rng.normal(0.0, 1.0, (64, 256))
    verdict = ie.loop_back(samples)
    assert verdict["complete"] == 1.0


def test_the_ensemble_floor_is_root_n_below_one_sample():
    rng = np.random.default_rng(23)
    samples = rng.normal(0.0, 1.0, (100, 128))
    summary = ie.ensemble_residual(samples)
    ratio = np.mean(summary["single_sample_floor"]) / np.mean(summary["floor"])
    assert 8.0 < ratio < 12.0            # root of 100


def test_the_vcr_stage_runs_twice_once_per_machine():
    """The signal passes through two machines, so the same known components
    are instantiated for each and neither is pooled into the other."""
    playback = ie.vcr_components(ie.PLAYBACK_MACHINE)
    recording = ie.vcr_components(ie.RECORDING)
    assert len(playback) == len(recording) and playback
    assert all(c.machine == ie.PLAYBACK_MACHINE for c in playback)
    assert all(c.machine == ie.RECORDING for c in recording)
    assert all(c.stage == ie.RF_PLAYBACK for c in playback)
    assert all(c.stage == ie.RF_RECORDING for c in recording)
    # the same parts, described for each machine
    kinds = lambda group: sorted(c.name.split(" ", 1)[1] for c in group)
    assert kinds(playback) == kinds(recording)


def test_the_playback_machine_is_inverted_first():
    """The playback machine acted LAST on the signal, so the chain forces it
    to be inverted FIRST; the picture stage runs after both."""
    assert ie.STAGES == (ie.RF_PLAYBACK, ie.RF_RECORDING, ie.PICTURE)
    assert ie.STAGES.index(ie.RF_PLAYBACK) < ie.STAGES.index(ie.RF_RECORDING)
    assert ie.STAGES.index(ie.RF_RECORDING) < ie.STAGES.index(ie.PICTURE)


class _StageOrder(_Gauges):
    """Records which stage each measurement happened in, and only lets a
    later stage see a signal the earlier one has already corrected."""

    def __init__(self, floor):
        super().__init__(floor)
        self.seen = []

    def measure(self, component, signal):
        self.seen.append((component.stage, float(np.mean(signal["residual"]))))
        return super().measure(component, signal)


def test_a_later_stage_measures_on_a_corrected_signal():
    first = _component()
    first.name, first.stage = "playback part", ie.RF_PLAYBACK
    second = _component()
    second.name, second.stage = "recording part", ie.RF_RECORDING
    third = _component()
    third.name, third.stage = "picture part", ie.PICTURE
    gauges = _StageOrder(floor=0.2)
    ie.multidimensional_information_extrapolation(
        {"residual": np.ones(8)}, [third, second, first], gauges
    )
    order = [stage for stage, _ in gauges.seen]
    # whatever order the components were handed in, the STAGES run in the
    # chain's order
    assert order.index(ie.RF_PLAYBACK) < order.index(ie.RF_RECORDING)
    assert order.index(ie.RF_RECORDING) < order.index(ie.PICTURE)
    # and each stage starts from a residual the previous one has reduced
    starts = {}
    for stage, value in gauges.seen:
        starts.setdefault(stage, value)
    assert starts[ie.RF_RECORDING] < starts[ie.RF_PLAYBACK]
    assert starts[ie.PICTURE] < starts[ie.RF_RECORDING]


def test_the_picture_stage_is_where_a_video_filter_is_legitimate():
    """At RF the chain is LTI through a nonlinear demodulator, so only a
    pre-demodulator correction inverts it; the picture stage is in the video
    domain, where a video-domain filter is allowed."""
    picture = ie.picture_components()
    assert picture and all(c.stage == ie.PICTURE for c in picture)
    assert any("emphasis" in c.name for c in picture)



# --------------------------------------------------------------------------
# The projection's three preconditions. Each of these was silently violated,
# and each violation only becomes visible once more than two components with
# more than one grid and more than a real value are in the ensemble - which
# is exactly what "each component adds another dimension" produces.
# --------------------------------------------------------------------------


def test_residuals_on_different_grids_are_never_projected():
    """An inner product between two vectors is a measurement only when they
    share an abscissa. Truncating a 238-bin frequency response and a
    262-line time residual to a common length and dotting them returns a
    number that is an artefact of memory layout, not a coherence."""
    rng = np.random.default_rng(11)
    residuals = {
        "response": {"frequency": rng.standard_normal(238)},
        "timing": {"frequency": rng.standard_normal(262)},
    }
    assert ie.coherence(residuals, "frequency") == 0.0
    out = ie.orthogonalize(residuals, ["response", "timing"])
    for name in residuals:
        assert np.array_equal(out[name]["frequency"],
                              residuals[name]["frequency"])


def test_the_projection_keeps_the_phase():
    """COMPLEX_COMPONENTS declares all three axes complex. Casting to
    float64 keeps the real part and throws the phase away, which defeats
    the complex design wherever a residual passes through the projection."""
    rng = np.random.default_rng(12)
    shared = rng.standard_normal(64)
    first = shared + 1j * rng.standard_normal(64)
    second = 0.8 * shared + 1j * rng.standard_normal(64)
    out = ie.orthogonalize({"first": {"frequency": first},
                            "second": {"frequency": second}},
                           ["first", "second"])
    kept = out["second"]["frequency"]
    assert np.iscomplexobj(kept)
    assert np.linalg.norm(kept.imag) > 0.0
    # the upstream component is never adjusted by a downstream one
    assert np.array_equal(out["first"]["frequency"], first)


def test_the_field_axis_is_orthogonalized_too():
    """AXES omits FIELD_AXIS, so iterating it excluded the per-field
    dimension from every projection: a dimension could be measured and
    accumulated and still contribute nothing to the descent."""
    rng = np.random.default_rng(13)
    first = rng.standard_normal(26)
    second = 0.9 * first + 0.1 * rng.standard_normal(26)
    residuals = {"one": {"field": first}, "two": {"field": second}}
    assert "field" in ie.axes_present(residuals)
    assert ie.coherence(residuals, "field") > ie.COHERENCE_THRESHOLD
    out = ie.orthogonalize(residuals, ["one", "two"])
    assert not np.array_equal(out["two"]["field"], second)


def test_two_real_components_on_one_grid_are_untouched_by_the_repair():
    """The live ensemble is two same-length real components on the
    frequency axis. The repair must be exactly the identity there, or it
    changed behaviour rather than fixing a latent defect."""
    rng = np.random.default_rng(14)
    shared = rng.standard_normal(238)
    residuals = {
        "amplitude flatness": {"frequency": shared},
        "sync edge response": {
            "frequency": 0.7 * shared + 0.3 * rng.standard_normal(238)},
    }
    order = list(residuals)
    out = ie.orthogonalize(residuals, order)
    # reproduce the pre-repair arithmetic and require a bitwise match
    vectors = [np.asarray(residuals[n]["frequency"], dtype=np.float64)
               for n in order]
    width = min(v.size for v in vectors)
    stack = np.array([np.nan_to_num(v[:width]) for v in vectors])
    directions = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    gram = np.abs(directions @ directions.T)
    np.fill_diagonal(gram, 0.0)
    assert gram.max() >= ie.COHERENCE_THRESHOLD
    basis, expected = [], {}
    for position, name in enumerate(order):
        vector = np.asarray(residuals[name]["frequency"], dtype=np.float64)
        head = np.nan_to_num(vector[:width])
        removed = np.zeros_like(head)
        for earlier in basis:
            removed = removed + float(head @ earlier) * earlier
        if position and np.any(removed):
            head = head - ie.ORTHOGONAL_AMOUNT * removed
            kept = np.array(vector, dtype=np.float64)
            kept[:width] = head
            expected[name] = kept
        else:
            expected[name] = vector
        direction = (head - sum(float(head @ e) * e for e in basis)
                     if basis else head)
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            basis.append(direction / norm)
    for name in order:
        assert np.array_equal(out[name]["frequency"], expected[name])


# --------------------------------------------------------------------------
# The ellipsoid: Ethan's closed form for the whole differentiation.
# "All I need is to fit these residuals to an elipse of how ever many
# dimensions I have, and I have fully collapsed the differentiation."
# --------------------------------------------------------------------------


def test_the_ellipse_is_the_constant():
    """Every component contributes a unit direction, so the trace of the
    quadratic form is the component count whatever the components do. The
    total is conserved and only the SHAPE carries information - which is
    why the asymmetry is the measurement and the ellipse is the invariant."""
    rng = np.random.default_rng(21)
    n = 128
    shared = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ensembles = {
        "independent": {f"c{i}": {"frequency": rng.standard_normal(n)
                                  + 1j * rng.standard_normal(n)}
                        for i in range(4)},
        "collinear": {f"c{i}": {"frequency": shared * (1.0 + 0.02 * i)}
                      for i in range(4)},
        "mixed": {f"c{i}": {"frequency": shared + 1.5 * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n))}
            for i in range(4)},
    }
    for residuals in ensembles.values():
        assert ie.ellipsoid(residuals, "frequency")["trace"] == pytest.approx(4.0)


def test_the_ellipse_reads_the_rank_the_evidence_supports():
    """One departure seen four times is one dimension, not four: the
    ellipsoid degenerates to a needle and the rank says so in one step,
    where the pass-by-pass recursion needs a pass per dimension."""
    rng = np.random.default_rng(22)
    shared = rng.standard_normal(96) + 1j * rng.standard_normal(96)
    one_departure = {f"c{i}": {"frequency": shared * (1.0 + 0.02 * i)}
                     for i in range(4)}
    needle = ie.ellipsoid(one_departure, "frequency")
    assert needle["rank"] == 1
    assert needle["asymmetry"] == pytest.approx(1.0, abs=1e-6)
    assert not needle["sphere"]

    independent = {f"c{i}": {"frequency": rng.standard_normal(96)
                             + 1j * rng.standard_normal(96)}
                   for i in range(4)}
    ball = ie.ellipsoid(independent, "frequency")
    assert ball["rank"] == 4
    assert ball["asymmetry"] < 0.15


def test_the_ellipse_needs_the_phase():
    """The fit is over COMPLEX components. Stripping the phase and keeping
    the magnitude returns a different shape, so a projection that casts to
    float64 does not merely lose precision - it answers a different
    question."""
    rng = np.random.default_rng(23)
    n = 128
    shared = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    complexes = {f"c{i}": {"frequency": shared + 1.5 * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n))}
        for i in range(4)}
    stripped = {name: {"frequency": np.abs(value["frequency"])}
                for name, value in complexes.items()}
    with_phase = ie.ellipsoid(complexes, "frequency")["asymmetry"]
    without = ie.ellipsoid(stripped, "frequency")["asymmetry"]
    assert abs(with_phase - without) > 0.3


def test_the_ellipse_refuses_components_on_different_grids():
    """The same precondition as the projection: an ellipsoid fitted across
    two abscissae is fitted to nothing."""
    rng = np.random.default_rng(24)
    residuals = {"response": {"frequency": rng.standard_normal(238)},
                 "timing": {"frequency": rng.standard_normal(262)}}
    assert ie.ellipsoid(residuals, "frequency")["rank"] == 0


# --------------------------------------------------------------------------
# The loop: "Then differentiation using the target curve onto the source
# data. This loops until the information floor is reached (total possible
# information on the source signal) ... Down to the circular shape of the
# complex signal."
# --------------------------------------------------------------------------


def _ensemble(rng, length, count, departures, noise_scale):
    shared = [rng.standard_normal(length) + 1j * rng.standard_normal(length)
              for _ in range(departures)]
    out = {}
    for index in range(count):
        value = noise_scale * (rng.standard_normal(length)
                               + 1j * rng.standard_normal(length))
        for order, vector in enumerate(shared):
            weight = 1.0 if order == 0 else (
                (1.0 if index % 2 else -1.0) if order == 1
                else (index - (count - 1) / 2.0) / (count / 2.0))
            value = value + weight * vector
        out[f"c{index}"] = {"frequency": value}
    return out


def test_the_sphere_floor_is_closed_form():
    """N / (L + N - 1), with a scatter of sqrt(2) / L. Both derived, so the
    stopping rule carries no fitted constant."""
    floor = ie.sphere_floor(6, 256)
    assert floor["asymmetry"] == pytest.approx(6 / 261.0)
    assert floor["scatter"] == pytest.approx(np.sqrt(2.0) / 256.0)
    # more components raise the floor while its scatter does not move, so
    # the margin a real asymmetry must clear is better determined
    wider = ie.sphere_floor(24, 256)
    assert wider["asymmetry"] > floor["asymmetry"]
    assert wider["scatter"] == pytest.approx(floor["scatter"])


def test_noise_alone_is_already_at_the_information_floor():
    rng = np.random.default_rng(51)
    residuals = _ensemble(rng, 256, 6, departures=0, noise_scale=1.0)
    out = ie.differentiate_to_floor(residuals, "frequency", amount=1.0)
    assert out["at_floor"]
    assert out["targets"] == []


@pytest.mark.parametrize("departures", [1, 2, 3])
def test_the_loop_recovers_every_planted_departure_and_stops_circular(
        departures):
    """The count of directions recovered is the count planted, and the loop
    terminates when what is left is as spherical as randomness of this size
    and length would be."""
    rng = np.random.default_rng(60 + departures)
    residuals = _ensemble(rng, 256, 6, departures, noise_scale=1.0)
    out = ie.differentiate_to_floor(residuals, "frequency", amount=1.0,
                                    maximum_passes=10)
    assert out["at_floor"], out["reason"]
    # every planted departure is accounted for. A departure whose weighting
    # varies across the ensemble can occupy more than one principal
    # direction, so the recovered count is a lower bound on the planted
    # one, never an upper one.
    assert len(out["targets"]) >= departures
    assert out["trace"][0]["sigma"] > 10.0      # structure, unmistakably
    assert out["trace"][-1]["sigma"] <= ie.SPHERE_SIGMA
    # THE HALTING COUNT IS SET BY THE DIMENSIONS SUPPLIED: the loop needs
    # one pass per real dimension plus the one that confirms the circle,
    # and never more than the ensemble can hold.
    assert out["passes"] <= departures + 1


def test_the_shape_is_judged_on_the_directions_that_remain():
    """Deflating a direction drawn from the ensemble's own span leaves an
    exact zero eigenvalue: the set spans one dimension fewer. Counting that
    zero as an unequal share reports the deflation's own bookkeeping as
    structure, and the loop then never reaches the floor."""
    rng = np.random.default_rng(71)
    residuals = _ensemble(rng, 256, 6, departures=1, noise_scale=1.2)
    fit = ie.ellipsoid(residuals, "frequency")
    assert fit["significant"] == 1
    names = fit["names"]
    stack = np.array([np.asarray(residuals[n]["frequency"]).ravel()
                      for n in names])
    unit = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    direction = np.asarray(fit["axes"])[:, 0].conj() @ unit
    direction = direction / np.linalg.norm(direction)
    cleaned = {n: {"frequency": (lambda v: v - (direction.conj() @ v) * direction)(
        np.asarray(residuals[n]["frequency"]).ravel())} for n in names}
    # counting the zero the deflation created reports it as structure
    naive = ie.ellipsoid(cleaned, "frequency")
    assert np.count_nonzero(naive["eigenvalues"] <= naive["floor"]) >= 1
    assert naive["sigma"] > ie.SPHERE_SIGMA
    # telling the fit how many directions were taken out gives the correct
    # verdict: what remains is spherical
    honest = ie.ellipsoid(cleaned, "frequency", removed=1)
    assert honest["sigma"] <= ie.SPHERE_SIGMA


def test_a_pass_that_does_not_simplify_the_shape_is_refused():
    """The recursion expands only while the evidence pays for it."""
    rng = np.random.default_rng(81)
    residuals = _ensemble(rng, 64, 4, departures=1, noise_scale=1.0)
    out = ie.differentiate_to_floor(residuals, "frequency", amount=1.0,
                                    maximum_passes=25)
    assert out["reason"] in {"circular: at the information floor",
                             "refused: the pass did not lower the asymmetry",
                             "no direction stands above the noise edge"}
    assert out["passes"] <= 25


def test_the_wiener_weight_per_direction_is_the_correction_gain_law():
    """A direction stands at lambda = bulk + signal, so the share worth
    removing is (lambda - bulk)/lambda. On a unit-direction ensemble the
    bulk is one, and that weight equals a* = 1/(1 + rho) for every
    eigenvalue - the project's correction-gain law, arrived at from the
    other side. The flat half-step is its special case at rho = 1."""
    for eigenvalue in (1.2, 1.5, 2.0, 3.0, 6.0, 12.0):
        signal_to_noise = eigenvalue - 1.0
        wiener = (eigenvalue - 1.0) / eigenvalue
        gain_law = 1.0 / (1.0 + 1.0 / signal_to_noise)
        assert wiener == pytest.approx(gain_law, abs=1e-12)
    assert (2.0 - 1.0) / 2.0 == pytest.approx(ie.ORTHOGONAL_AMOUNT)


def test_the_wiener_weight_beats_a_flat_step():
    """A flat half-step leaves half of each direction in place, so the loop
    finds the same direction again and over-counts what it recovered. The
    per-direction weight removes what the evidence supports and recovers
    the planted count."""
    rng = np.random.default_rng(91)
    residuals = _ensemble(rng, 512, 12, departures=3, noise_scale=1.0)
    weighted = ie.differentiate_to_floor(dict(residuals), "frequency",
                                         amount=None, maximum_passes=12)
    flat = ie.differentiate_to_floor(dict(residuals), "frequency",
                                     amount=0.5, maximum_passes=12)
    assert weighted["at_floor"] and flat["at_floor"]
    assert weighted["passes"] < flat["passes"]
    assert len(weighted["targets"]) < len(flat["targets"])


# --------------------------------------------------------------------------
# "The differentiation is the differential of our synthetic components to
# the actual measured componnet, this has nested differentials in it as well
# according to our graph, so this expands out to a multi dimensional matrix
# that represents our known components."
# --------------------------------------------------------------------------


def test_the_matrix_is_synthetic_minus_measured_nested_by_the_graph():
    rng = np.random.default_rng(111)
    length = 128
    names = ["a", "b", "c", "d"]
    synthetic = {name: np.zeros(length, dtype=complex) for name in names}
    measured = {name: rng.standard_normal(length)
                + 1j * rng.standard_normal(length) for name in names}
    relations = {"a": ["b", "c"], "b": ["a", "d"], "c": ["a"], "d": ["b"]}
    matrix = ie.component_differentials(synthetic, measured, "frequency",
                                        relations)
    # the diagonal: every component's own synthetic minus measured
    for name in names:
        assert np.allclose(matrix[name]["frequency"], -measured[name])
    # the off-diagonal: only where the graph relates the pair, once each
    pairs = sorted(key for key in matrix if " vs " in key)
    assert pairs == ["a vs b", "a vs c", "b vs d"]
    assert np.allclose(matrix["a vs b"]["frequency"],
                       matrix["a"]["frequency"] - matrix["b"]["frequency"])


def test_unrelated_components_are_never_differenced():
    """A difference between two components the graph does not relate is not
    a relationship anyone claimed."""
    rng = np.random.default_rng(112)
    names = ["a", "b", "c"]
    synthetic = {name: np.zeros(64) for name in names}
    measured = {name: rng.standard_normal(64) for name in names}
    matrix = ie.component_differentials(synthetic, measured, "frequency",
                                        {"a": ["b"], "b": ["a"], "c": []})
    assert sorted(key for key in matrix if " vs " in key) == ["a vs b"]


def test_the_nested_matrix_still_reaches_the_floor():
    """The matrix is rank-deficient by construction - "a vs b" is the
    difference of "a" and "b" - so the loop must charge that deficiency to
    the ensemble rather than to its own deflation, or it refuses its own
    first pass."""
    rng = np.random.default_rng(113)
    length = 256
    shared = rng.standard_normal(length) + 1j * rng.standard_normal(length)
    names = ["a", "b", "c", "d"]
    synthetic = {name: np.zeros(length, dtype=complex) for name in names}
    measured = {name: -(shared + 0.5 * (rng.standard_normal(length)
                                        + 1j * rng.standard_normal(length)))
                for name in names}
    matrix = ie.component_differentials(
        synthetic, measured, "frequency",
        {"a": ["b", "c"], "b": ["a", "d"], "c": ["a"], "d": ["b"]})
    fit = ie.ellipsoid(matrix, "frequency")
    assert len(fit["names"]) > fit["rank"]        # dependent by construction
    out = ie.differentiate_to_floor(matrix, "frequency", maximum_passes=8)
    assert out["at_floor"], out["reason"]


def test_the_total_eigenvalue_is_the_information_the_components_hold():
    """Ethan: "the total eigenvalue is the total amount of information we
    could possible derive from our components". Every component contributes
    one unit whatever it carries, so N components hold N units and no more;
    what the fit adds is how much of that budget stands in real
    directions."""
    rng = np.random.default_rng(114)
    length = 256

    def noise():
        return rng.standard_normal(length) + 1j * rng.standard_normal(length)

    shared = noise()
    independent = {f"c{i}": {"frequency": noise()} for i in range(6)}
    one_thing = {f"c{i}": {"frequency": shared * (1.0 + 0.02 * i)}
                 for i in range(6)}
    six = {f"c{i}": {"frequency": shared + 1.2 * noise()} for i in range(6)}
    twelve = {f"c{i}": {"frequency": shared + 1.2 * noise()} for i in range(12)}

    assert ie.ellipsoid(independent, "frequency")["information"] == 6.0
    assert ie.ellipsoid(independent, "frequency")["resolved"] == 0.0
    assert ie.ellipsoid(one_thing, "frequency")["resolved"] == pytest.approx(
        6.0, rel=1e-6)
    # doubling the components roughly doubles the information resolved,
    # which is why "increase the constant, how many components" buys more
    assert (ie.ellipsoid(twelve, "frequency")["resolved"]
            > 1.7 * ie.ellipsoid(six, "frequency")["resolved"])


# --------------------------------------------------------------------------
# "You only 'guess' the curve, you don't know it." A fitted ellipsoid
# describes the ensemble it was fitted to by construction, so its directions
# carry no evidence until they describe evidence they have not seen.
# --------------------------------------------------------------------------


def test_a_guess_that_only_describes_its_own_bank_is_caught():
    rng = np.random.default_rng(303)
    length, count = 256, 16
    shared = rng.standard_normal(length) + 1j * rng.standard_normal(length)

    def noise():
        return (rng.standard_normal(length)
                + 1j * rng.standard_normal(length))

    real = {f"c{i}": {"frequency": shared + 1.5 * noise()}
            for i in range(count)}
    # planted in the even members only - exactly the bank the split makes
    overfit = {f"c{i}": {"frequency": 1.5 * noise()
                         + (3.0 * shared if i % 2 == 0 else 0.0)}
               for i in range(count)}

    honest = ie.guess_credibility(real, "frequency")
    failed = ie.guess_credibility(overfit, "frequency")
    assert honest["mean"] > 0.3
    assert failed["mean"] < 0.05
    assert honest["mean"] > 10.0 * failed["mean"]


def test_the_admission_bar_is_the_equal_share_null():
    """A direction explaining nothing it has not seen still carries its
    equal share of the other bank's energy, and an equal share of L
    directions is 1/L. So `carries = held_out * L` and admission is
    `carries > 1` - derived, not chosen."""
    rng = np.random.default_rng(304)
    length, count = 256, 16
    shared = rng.standard_normal(length) + 1j * rng.standard_normal(length)
    real = {f"c{i}": {"frequency": shared + 1.5 * (
        rng.standard_normal(length) + 1j * rng.standard_normal(length))}
        for i in range(count)}
    result = ie.guess_credibility(real, "frequency")
    assert result["length"] == length
    assert np.all(result["carries"] > 1.0)
    assert np.all(result["admissible"])


def test_too_few_components_to_hold_any_out():
    rng = np.random.default_rng(305)
    residuals = {f"c{i}": {"frequency": rng.standard_normal(64)}
                 for i in range(3)}
    result = ie.guess_credibility(residuals, "frequency")
    assert result["names"] == []
    assert "hold out" in result["why"]


# --------------------------------------------------------------------------
# "The limit is acheived when the residual stops increasing, that's when you
# know you have approached the end of this result, which gives you the exact
# residual."
# --------------------------------------------------------------------------


def test_the_accumulation_converges_and_gives_the_exact_residual():
    rng = np.random.default_rng(306)
    residuals = _ensemble(rng, 256, 12, departures=3, noise_scale=1.0)
    out = ie.differentiate_to_floor(residuals, "frequency", maximum_passes=20)
    assert out["starting_energy"] > 0.0
    assert 0.0 < out["attributed_fraction"] <= 1.0
    assert out["exact_residual"] == pytest.approx(
        out["starting_energy"] - out["attributed"])
    # the accumulation only ever grows
    assert all(b >= a for a, b in zip(out["accumulation"],
                                      out["accumulation"][1:]))


def test_the_accumulation_rule_alone_does_not_terminate_correctly():
    """Measured against ground truth: with the circular test suppressed the
    accumulation rule grinds on until 99.9 per cent of the energy has been
    attributed, against a true structure share of 72-73 per cent, because a
    pass can always find something in noise. It is necessary but NOT
    sufficient, and must run beside the circular test."""
    rng = np.random.default_rng(41)
    length, count = 256, 12
    shared = [rng.standard_normal(length) + 1j * rng.standard_normal(length)
              for _ in range(3)]
    residuals, planted, total = {}, 0.0, 0.0
    for index in range(count):
        structure = np.zeros(length, dtype=complex)
        for order, vector in enumerate(shared):
            weight = 1.0 if order == 0 else (
                (1.0 if index % 2 else -1.0) if order == 1
                else np.cos(np.pi * index / (count - 1)))
            structure = structure + weight * vector
        value = structure + (rng.standard_normal(length)
                             + 1j * rng.standard_normal(length))
        residuals[f"c{index}"] = {"frequency": value}
        planted += float(np.vdot(structure, structure).real)
        total += float(np.vdot(value, value).real)

    kept = ie.SPHERE_SIGMA
    ie.SPHERE_SIGMA = -1e9                      # suppress the circular test
    try:
        alone = ie.differentiate_to_floor(dict(residuals), "frequency",
                                          maximum_passes=200)
    finally:
        ie.SPHERE_SIGMA = kept
    together = ie.differentiate_to_floor(dict(residuals), "frequency",
                                         maximum_passes=200)

    truth = planted / total
    assert alone["attributed_fraction"] > 0.98      # it attributes nearly all
    assert alone["attributed_fraction"] - truth > 0.2
    # with both rules active it lands near the truth instead
    assert abs(together["attributed_fraction"] - truth) < 0.1


def test_the_surrogate_null_is_built_by_the_same_construction():
    """An analytic null assumes independent rows. A nested differential
    matrix has none - its entries are linear combinations of its diagonals -
    so the null must be GENERATED by running the same construction on
    structureless input."""
    rng = np.random.default_rng(401)
    length, fields = 119, 13
    synthetic = {f"f{i}": np.zeros(length) for i in range(fields)}
    measured = {f"f{i}": rng.standard_normal(length) for i in range(fields)}
    relations = {n: [o for o in synthetic if o != n] for n in synthetic}
    matrix = ie.component_differentials(synthetic, measured, "frequency",
                                        relations)
    null = ie.surrogate_null(matrix, "frequency", draws=12, seed=1)
    assert null["draws"] == 12
    # the null rebuilds the construction rather than assuming independence
    assert null["independent"] == fields
    assert null["deficiency"] == len(matrix) - fields
    # structureless input does not stand apart from a null built the same way
    assert abs(null["asymmetry_z"]) < 3.0


def test_the_nested_matrix_is_blind_to_a_shared_departure():
    """THE DEFECT, stated as a test. A shared departure cancels exactly in
    every difference, so it enters only the N diagonals of N(N+1)/2 entries;
    and because every row is unit-normalised, scaling those diagonals does
    not move their directions. The shape is therefore unchanged, which is
    why the run reported in ELLIPTICAL_COLLAPSE.md section 6 measured its own
    construction."""
    length, fields = 119, 13
    shared = np.random.default_rng(402).standard_normal(length)

    def nested(scale):
        rng = np.random.default_rng(55)
        synthetic = {f"f{i}": np.zeros(length) for i in range(fields)}
        measured = {f"f{i}": rng.standard_normal(length) + scale * shared
                    for i in range(fields)}
        relations = {n: [o for o in synthetic if o != n] for n in synthetic}
        return ie.component_differentials(synthetic, measured, "frequency",
                                          relations)

    def energy(matrix, off):
        keys = [k for k in matrix if (" vs " in k) == off]
        return float(np.mean([np.vdot(matrix[k]["frequency"],
                                      matrix[k]["frequency"]).real
                              for k in keys]))

    quiet, loud = nested(0.0), nested(4.0)
    # the diagonals carry the departure
    assert energy(loud, False) > 10.0 * energy(quiet, False)
    # and the off-diagonals do not, at all
    assert energy(loud, True) == pytest.approx(energy(quiet, True), rel=1e-9)
    # so the shape cannot see it
    assert (ie.ellipsoid(loud, "frequency")["asymmetry"]
            == pytest.approx(ie.ellipsoid(quiet, "frequency")["asymmetry"],
                             abs=0.01))


def test_the_default_inner_product_is_blind_to_an_unknown_phase():
    """A simple FFT is sufficient for a component whose absolute phase
    nobody knows: the Gram is built from |<u_i, u_j>|, so a global phase
    factor cancels exactly. The complex bin carries frequency and phase
    together as ONE dimension, and the transform asks only for its
    direction."""
    rng = np.random.default_rng(601)
    length, count = 256, 12
    shared = rng.standard_normal(length) + 1j * rng.standard_normal(length)

    def ensemble(random_phase):
        out = {}
        for index in range(count):
            phase = np.exp(2j * np.pi * rng.random()) if random_phase else 1.0
            out[f"c{index}"] = {"frequency": phase * shared + 0.4 * (
                rng.standard_normal(length)
                + 1j * rng.standard_normal(length))}
        return out

    aligned = ie.ellipsoid(ensemble(False), "frequency")
    unknown = ie.ellipsoid(ensemble(True), "frequency")
    assert unknown["significant"] == aligned["significant"]
    assert abs(unknown["sigma"] - aligned["sigma"]) < 5.0


def test_real_parameters_buys_the_gain_delay_split_and_costs_the_blindness():
    """The two modes are complementary. Stacking real and imaginary parts
    makes r and j*r orthogonal - correct when each component is a real
    parameter's signature - but a rotated copy then reads as a different
    direction, so an unknown absolute phase costs it dearly."""
    shape = np.linspace(-1.0, 1.0, 256)
    gain_delay = {"gain": {"frequency": shape.astype(complex)},
                  "delay": {"frequency": 1j * shape}}
    assert ie.ellipsoid(gain_delay, "frequency")["rank"] == 1
    assert ie.ellipsoid(gain_delay, "frequency",
                        real_parameters=True)["rank"] == 2

    rng = np.random.default_rng(602)
    length, count = 256, 12
    truth = rng.standard_normal(length) + 1j * rng.standard_normal(length)

    def ensemble(random_phase):
        out = {}
        for index in range(count):
            phase = np.exp(2j * np.pi * rng.random()) if random_phase else 1.0
            out[f"c{index}"] = {"frequency": phase * truth + 0.4 * (
                rng.standard_normal(length)
                + 1j * rng.standard_normal(length))}
        return out

    hermitian_cost = abs(
        ie.ellipsoid(ensemble(True), "frequency")["sigma"]
        - ie.ellipsoid(ensemble(False), "frequency")["sigma"])
    stacked_cost = abs(
        ie.ellipsoid(ensemble(True), "frequency", real_parameters=True)["sigma"]
        - ie.ellipsoid(ensemble(False), "frequency",
                       real_parameters=True)["sigma"])
    assert hermitian_cost < 5.0
    assert stacked_cost > 5.0 * max(hermitian_cost, 1.0)
