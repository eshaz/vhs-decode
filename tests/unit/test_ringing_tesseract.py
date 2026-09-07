"""The ringing correction on the tesseract graph.

Every test here holds one of the properties the design rests on rather
than a number the module happens to produce: the fold is exactly
invertible, the correction is a subtraction and not a filter, it is
strictly causal, it is level neutral, it derives from sync alone, it uses
nothing from any other field, and it removes a planted artifact it was
never told about.
"""

import logging
import math

import numpy as np
import pytest

from vhsdecode.formats import get_format_params, parse_tape_speed
from vhsdecode.models import ringing_tesseract as rt
from vhsdecode.models import tesseract


@pytest.fixture(scope="module")
def ntsc():
    sys_params, decoder_params = get_format_params(
        "NTSC", "VHS", parse_tape_speed("sp"), logging.getLogger(__name__))
    return sys_params, decoder_params


@pytest.fixture(scope="module")
def geometry(ntsc):
    sys_params, decoder_params = ntsc
    return rt.build_geometry(sys_params, decoder_params,
                             sys_params["outlinelen"], 0, 263)


def synthetic_field(geometry, artifact=None, lines=263, noise=0.0,
                    seed=3, picture=True):
    """A field of specification-shaped sync pulses, with an artifact.

    The blanking is at zero IRE and the tip at minus the stated depth, and
    every transition is the specification's own erf. `artifact` is a
    kernel in IRE per unit signed step, laid down one sample past each
    transition's crossing, which is exactly where the module looks for it.
    """
    columns = geometry.samples_per_line
    rng = np.random.default_rng(seed)
    field = np.zeros((lines, columns))
    edge = geometry.sync_transition_samples / rt.ERF_TEN_NINETY_FACTOR
    positions = np.arange(columns, dtype=np.float64)
    fall = 3.0
    rise = fall + geometry.sync_pulse_samples
    active = geometry.active_video_start_sample
    for row in range(lines):
        line = np.zeros(columns)
        line += -geometry.sync_depth_ire * 0.5 * (
            1.0 + rt._erf((positions - fall) / (math.sqrt(2.0) * edge)))
        line += geometry.sync_depth_ire * 0.5 * (
            1.0 + rt._erf((positions - rise) / (math.sqrt(2.0) * edge)))
        if picture:
            # a bar in the active area, so the gate has content to find
            level = 20.0 + 40.0 * ((row % 5) / 4.0)
            start = int(active) + 40
            stop = columns - 60
            line[start:stop] += level
        field[row] = line
    if artifact is not None:
        length = len(artifact)
        for row in range(lines):
            line = field[row]
            padded = np.zeros(columns + length)
            for origin, step in ((int(fall) + 1, -geometry.sync_depth_ire),
                                 (int(rise) + 1, +geometry.sync_depth_ire)):
                padded[origin:origin + length] += step * artifact
            if picture:
                start = int(active) + 40
                stop = columns - 60
                level = 20.0 + 40.0 * ((row % 5) / 4.0)
                padded[start:start + length] += level * artifact
                padded[stop:stop + length] += -level * artifact
            field[row] = line + padded[:columns]
    if noise:
        field = field + rng.normal(0.0, noise, field.shape)
    return field


def decaying_ring(geometry, frequency_mhz=1.4, decay_us=0.7, amplitude=0.06):
    """One damped oscillation, per unit signed step."""
    lags = np.arange(geometry.aftermath_samples, dtype=np.float64)
    time_us = lags / geometry.sample_rate_mhz
    return amplitude * np.exp(-time_us / decay_us) * np.cos(
        2.0 * np.pi * frequency_mhz * time_us)


class TestGeometry:
    """Every number from the format tables, and nothing chosen."""

    def test_the_truncation_is_the_shorter_witnessed_flat(self, geometry, ntsc):
        """Each flat guarded by ONE settling time, because the decoder's
        zero-phase luma low-pass puts a precursor that far ahead of the
        next transition."""
        sys_params, _ = ntsc
        rate = sys_params["outfreq"]
        settle = geometry.transition_settle_samples
        tip = sys_params["hsyncPulseUS"] * rate - settle
        back = ((sys_params["activeVideoUS"][0] - sys_params["hsyncPulseUS"])
                * rate - settle)
        assert geometry.aftermath_samples == int(math.floor(min(tip, back)))

    def test_the_vertical_interval_is_excluded_by_the_specification(
            self, geometry, ntsc):
        sys_params, _ = ntsc
        expected = int(math.ceil(rt.VSYNC_SECTIONS * sys_params["numPulses"] / 2))
        assert geometry.first_measurable_line == expected

    def test_the_burst_window_comes_from_the_format(self, geometry):
        low, high = geometry.burst_lags
        assert 0 < low < high <= geometry.aftermath_samples

    def test_pal_derives_its_own_geometry(self):
        sys_params, decoder_params = get_format_params(
            "PAL", "VHS", parse_tape_speed("sp"), logging.getLogger(__name__))
        pal = rt.build_geometry(sys_params, decoder_params,
                                sys_params["outlinelen"], 2, 312)
        assert pal.aftermath_samples > 0
        assert pal.sync_depth_ire == pytest.approx(42.857, abs=1e-2)
        assert "PAL" not in rt.describe_geometry(pal)   # no format names inside


class TestTheFold:
    """The tesseract's own operator, used unchanged."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_unfold_matches_the_tesseracts_reconstruct(self, order):
        rng = np.random.default_rng(11)
        shape = (2,) * order + (13,)
        values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        variance = np.abs(rng.normal(size=shape)) + 0.1
        cube = tesseract.Cube(tuple(f"a{i}" for i in range(order)),
                              values, variance)
        contrasts = tesseract.walsh(cube)
        assert np.abs(rt.unfold(cube, contrasts)
                      - tesseract.reconstruct(cube, contrasts)).max() < 1e-12

    def test_keeping_every_contrast_returns_the_cube_exactly(self):
        rng = np.random.default_rng(5)
        shape = (2, 2, 2, 9)
        values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        cube = tesseract.Cube(("a", "b", "c"), values, np.ones(shape))
        contrasts = tesseract.walsh(cube)
        assert np.abs(rt.unfold(cube, contrasts) - cube.values).max() < 1e-12

    def test_the_line_axes_put_parity_first_then_the_slow_scales(self):
        names, bits, holdout = rt.line_axes(3, 128)
        assert names[0] == "line parity" and bits[0] == 0
        assert bits[1] == 6 and bits[2] == 5      # the halves, then quarters
        assert holdout == 4
        assert holdout not in bits


class TestTheMeasurement:
    """Sync only, on one lag grid, against the specification's own edge."""

    def test_it_reads_two_events_a_line(self, geometry):
        events = rt.measure_events(synthetic_field(geometry), geometry)
        assert events is not None
        assert events.departure.shape[0] == 2
        assert events.count > 200
        assert events.lags == geometry.aftermath_samples

    def test_the_steps_are_the_specified_depth_and_signed(self, geometry):
        events = rt.measure_events(synthetic_field(geometry), geometry)
        assert events.step_ire[0].mean() == pytest.approx(
            -geometry.sync_depth_ire, abs=0.2)
        assert events.step_ire[1].mean() == pytest.approx(
            +geometry.sync_depth_ire, abs=0.2)

    def test_a_clean_field_departs_from_the_specification_by_nothing(
            self, geometry):
        """The fall reads exactly zero. The rise reads 0.003 per unit step
        at its first lag and nothing after, which is the crossing
        estimator interpolating linearly across an erf only two samples
        wide - the synthetic edge is the SPECIFIED width, and a real one
        is five to six samples, where the same error is an order of
        magnitude smaller."""
        events = rt.measure_events(synthetic_field(geometry), geometry)
        assert np.abs(events.departure[0]).max() < 1e-9
        assert np.abs(events.departure[1]).max() < 5e-3

    def test_it_recovers_a_planted_kernel(self, geometry):
        """The artifact is planted as `step * kernel`, so the departure
        per unit SIGNED step is the kernel itself on both polarities -
        which is the whole point of dividing by the signed step."""
        planted = decaying_ring(geometry)
        events = rt.measure_events(
            synthetic_field(geometry, planted), geometry)
        # the fall's crossing lands on a whole sample here, the rise's a
        # quarter of one off it, and the crossing estimator interpolates
        # linearly across an erf only two samples wide - so the rise
        # carries a little of that error on its steepest lags
        assert np.abs(events.departure[0].mean(axis=0) - planted).max() < 5e-3
        assert np.abs(events.departure[1].mean(axis=0) - planted).max() < 2e-2

    def test_the_picture_never_enters_the_measurement(self, geometry):
        """Two fields differing ONLY in their active picture must give the
        same model: rule 2, checked rather than asserted."""
        planted = decaying_ring(geometry)
        with_picture = rt.measure_events(
            synthetic_field(geometry, planted, picture=True), geometry)
        without = rt.measure_events(
            synthetic_field(geometry, planted, picture=False), geometry)
        assert np.abs(with_picture.departure
                      - without.departure).max() < 1e-4

    def test_a_field_with_no_pulses_is_refused(self, geometry):
        flat = np.zeros((263, geometry.samples_per_line))
        assert rt.measure_events(flat, geometry) is None


class TestTheChromaGate:
    """A contrast living in the burst lags is the chroma, not the channel."""

    def test_a_burst_confined_alternation_is_refused(self, geometry):
        planted = decaying_ring(geometry)
        field = synthetic_field(geometry, planted)
        low, high = geometry.burst_lags
        rise = int(3.0 + geometry.sync_pulse_samples) + 1
        # a burst at the frequency the real one is measured at in the
        # luma back porch, about 1.3 MHz - a Nyquist-rate plant would be
        # destroyed by the sub-sample alignment and prove nothing
        burst = np.zeros(geometry.aftermath_samples)
        burst[low:high] = 2.0 * np.cos(
            2.0 * np.pi * (1.3 / geometry.sample_rate_mhz)
            * np.arange(high - low))
        for row in range(field.shape[0]):
            sign = 1.0 if row % 2 == 0 else -1.0
            field[row, rise:rise + geometry.aftermath_samples] += sign * burst
        events = rt.measure_events(field, geometry)
        cube, _counts, _bits = rt.cube_from_events(events, geometry, 2)
        gate = rt.chroma_contaminated(cube, geometry)
        refused = {":".join(s) for s in gate["refused"]}
        assert "line parity" in refused
        assert "polarity" not in refused
        assert "level" not in gate["refused"]

    def test_the_channel_itself_is_not_refused(self, geometry):
        events = rt.measure_events(
            synthetic_field(geometry, decaying_ring(geometry)), geometry)
        cube, _counts, _bits = rt.cube_from_events(events, geometry, 2)
        gate = rt.chroma_contaminated(cube, geometry)
        assert gate["per_contrast"]["polarity"]["refused"] is False


class TestTheCollapse:
    """One real signal per vertex, and it is the planted artifact."""

    def test_the_kernel_is_real_and_is_what_was_planted(self, geometry):
        planted = decaying_ring(geometry)
        events = rt.measure_events(
            synthetic_field(geometry, planted, noise=0.05), geometry)
        depth = rt.choose_depth(events, geometry)
        cube, _counts, bits = rt.cube_from_events(events, geometry,
                                                  depth["depth"])
        collapse = rt.collapse_to_one_signal(cube, geometry)
        assert collapse["kernel_is_real"]
        bank = rt.kernel_bank(collapse, events, bits)
        # The FALL's crossing lands on a whole sample in this fixture and
        # its kernel comes back essentially exactly. The RISE's lands a
        # quarter of a sample off one, and this fixture's edge is the
        # SPECIFIED two-sample transition - a 50 per cent crossing
        # interpolated linearly across an edge that steep is imprecise, so
        # its first lags carry a little of that. A real edge is five to
        # six samples wide, where the same error is far smaller.
        assert np.corrcoef(bank.mean[0], planted)[0, 1] > 0.999
        assert np.corrcoef(bank.mean[1], planted)[0, 1] > 0.98

    def test_the_kernel_ends_at_zero_so_levels_do_not_move(self, geometry):
        planted = decaying_ring(geometry)
        events = rt.measure_events(
            synthetic_field(geometry, planted), geometry)
        cube, _counts, bits = rt.cube_from_events(events, geometry, 1)
        collapse = rt.collapse_to_one_signal(cube, geometry)
        bank = rt.kernel_bank(collapse, events, bits)
        assert np.abs(bank.mean[:, -1]).max() < 1e-9

    def test_a_deeper_cube_is_refused_on_lines_it_did_not_fit(self, geometry):
        """The held-out scan is the guard against the template that
        lowers the in-sample residual and raises the held-out one."""
        events = rt.measure_events(
            synthetic_field(geometry, decaying_ring(geometry), noise=0.4),
            geometry)
        depth = rt.choose_depth(events, geometry)
        assert depth["depth"] == 1
        residuals = [row["held_out_residual"] for row in depth["scan"]]
        assert residuals[0] == min(residuals)


class TestTheCorrection:
    """A subtraction on a gated drive: causal, level neutral, sign aware."""

    def test_the_gate_places_the_kernel_at_the_crossing(self, geometry):
        line = np.zeros(400)
        line[200:] = -40.0
        line[196:200] = [-8.0, -18.0, -28.0, -36.0]
        origins, steps = rt.gated_events(line, 0.2, 3.0, 32)
        assert len(origins) == 1
        assert steps[0] == pytest.approx(-40.0)
        # the 50 per cent crossing sits between 197 and 198; the kernel
        # starts one sample past it
        assert origins[0] in (198, 199)

    def test_a_run_slower_than_the_gate_is_refused(self, geometry):
        line = np.concatenate([np.zeros(50), np.linspace(0.0, -40.0, 200),
                               np.full(150, -40.0)])
        assert len(rt.gated_events(line, 0.2, 3.0, 20)[0]) == 0
        assert len(rt.gated_events(line, 0.2, 3.0, None)[0]) == 1

    def test_a_change_under_the_noise_is_refused(self):
        """Pure noise of the declared size passes almost nothing: the
        threshold is three times the noise of a two-sample difference, so
        what gets through is the tail of that distribution and not a
        transient."""
        rng = np.random.default_rng(2)
        line = rng.normal(0.0, 1.0, 2000)
        passed = len(rt.gated_events(line, 1.0, 3.0, 20)[0])
        assert passed < 0.02 * (line.size / 2)

    def test_the_correction_is_strictly_causal(self, geometry):
        """Nothing may change before the transient that drives it."""
        planted = decaying_ring(geometry)
        field = synthetic_field(geometry, planted)
        events = rt.measure_events(field, geometry)
        cube, _counts, bits = rt.cube_from_events(events, geometry, 1)
        collapse = rt.collapse_to_one_signal(cube, geometry)
        bank = rt.kernel_bank(collapse, events, bits)
        out = rt.apply_kernels(field, bank, geometry, 0.05, 1.0,
                               max_duration=events.event_span)
        difference = out["field"] - field
        row = geometry.first_measurable_line + 20
        assert np.abs(difference[row, :3]).max() < 1e-9

    def test_the_correction_removes_the_planted_artifact(self, geometry):
        planted = decaying_ring(geometry)
        field = synthetic_field(geometry, planted, noise=0.05)
        events = rt.measure_events(field, geometry)
        depth = rt.choose_depth(events, geometry)
        cube, _counts, bits = rt.cube_from_events(events, geometry,
                                                  depth["depth"])
        collapse = rt.collapse_to_one_signal(cube, geometry)
        bank = rt.kernel_bank(collapse, events, bits)
        out = rt.apply_kernels(field, bank, geometry, 0.05, 1.0,
                               max_duration=events.event_span)
        after = rt.measure_events(out["field"], geometry)
        before_rms = float(np.sqrt(np.mean(events.departure ** 2)))
        after_rms = float(np.sqrt(np.mean(after.departure ** 2)))
        assert after_rms < 0.3 * before_rms

    def test_the_sign_of_the_step_decides_the_subtraction(self, geometry):
        """Using the magnitude instead inverts every falling edge - the
        defect this test exists to keep out."""
        planted = decaying_ring(geometry)
        field = synthetic_field(geometry, planted)
        events = rt.measure_events(field, geometry)
        cube, _counts, bits = rt.cube_from_events(events, geometry, 1)
        bank = rt.kernel_bank(rt.collapse_to_one_signal(cube, geometry),
                              events, bits)
        out = rt.apply_kernels(field, bank, geometry, 0.05, 1.0,
                               max_duration=events.event_span)
        difference = (out["field"] - field)[geometry.first_measurable_line + 5]
        fall = int(3.0) + 1
        # what was added at the fall is `step * kernel` with a NEGATIVE
        # step, so what the correction takes away is negative and the
        # field moves the other way: the difference must align with the
        # planted ring times plus the sync depth
        assert difference[fall:fall + 8].dot(
            planted[:8] * geometry.sync_depth_ire) > 0.0

    def test_it_leaves_a_flat_field_alone(self, geometry):
        field = synthetic_field(geometry, artifact=None, picture=False)
        events = rt.measure_events(field, geometry)
        cube, _counts, bits = rt.cube_from_events(events, geometry, 1)
        bank = rt.kernel_bank(rt.collapse_to_one_signal(cube, geometry),
                              events, bits)
        out = rt.apply_kernels(field, bank, geometry, 0.05, 1.0,
                               max_duration=events.event_span)
        # what is left is the crossing estimator's own error on this
        # fixture's two-sample edge, 0.003 per unit step at one lag,
        # applied at a 40 IRE step
        assert np.abs(out["field"] - field).max() < 0.15

    def test_the_gain_law_applies_half_the_believed_optimum(self):
        assert rt.correction_strength(1.0) == pytest.approx(0.5)
        assert rt.correction_strength(0.8) == pytest.approx(0.4)
        assert rt.correction_strength(-1.0) == 0.0
        assert rt.correction_strength(4.0) == 1.0


class TestTheRuntimeEntryPoint:
    """The signature the decoder calls, and the laws it must keep."""

    def _buffer(self, geometry, artifact=None, noise=0.05):
        field = synthetic_field(geometry, artifact, noise=noise)
        units = 100.0                     # arbitrary signal units per IRE
        blanking = 20000.0
        tip = blanking - geometry.sync_depth_ire * units
        return (field * units + blanking).reshape(-1).copy(), tip, blanking

    def test_it_returns_the_buffer_and_transient_parameters(self, geometry):
        buffer, tip, blanking = self._buffer(geometry, decaying_ring(geometry))
        state = {}
        out, lti = rt.process_field(buffer, geometry, tip, blanking, state,
                                    average_fields=4, head_parity=True)
        assert out is buffer
        assert set(lti) >= {"gain", "threshold", "blur_radius"}
        # read off the CORRECTOR's impulse, so it is a small number: what
        # the chain still leaves off the edge after this stage has run
        assert 0.0 <= lti["gain"] < 0.2
        assert state[rt.STATE_KEY]["status"] == "applied"

    def test_average_fields_changes_nothing(self, geometry):
        """Design law 3, checked: the argument is accepted and ignored."""
        artifact = decaying_ring(geometry)
        results = []
        for horizon in (0, 1, 4, 32):
            buffer, tip, blanking = self._buffer(geometry, artifact)
            state = {}
            out, _lti = rt.process_field(buffer, geometry, tip, blanking,
                                         state, average_fields=horizon)
            results.append(np.array(out))
        for other in results[1:]:
            assert np.array_equal(results[0], other)

    def test_nothing_is_carried_between_fields(self, geometry):
        """The same field corrected twice through one shared state gives
        the same answer as through two - there is no accumulation."""
        artifact = decaying_ring(geometry)
        shared = {}
        first, tip, blanking = self._buffer(geometry, artifact)
        rt.process_field(first, geometry, tip, blanking, shared,
                         average_fields=4)
        second, _tip, _blanking = self._buffer(geometry, artifact)
        rt.process_field(second, geometry, tip, blanking, shared,
                         average_fields=4)
        fresh, _tip, _blanking = self._buffer(geometry, artifact)
        rt.process_field(fresh, geometry, tip, blanking, {},
                         average_fields=4)
        assert np.array_equal(second, fresh)

    def test_measure_only_leaves_the_buffer_untouched(self, geometry):
        buffer, tip, blanking = self._buffer(geometry, decaying_ring(geometry))
        before = buffer.copy()
        state = {}
        rt.process_field(buffer, geometry, tip, blanking, state,
                         average_fields=4, apply_correction=False)
        assert np.array_equal(buffer, before)
        assert state[rt.STATE_KEY]["status"] == "measured"

    def test_the_rows_the_model_does_not_own_are_left_alone(self, geometry):
        buffer, tip, blanking = self._buffer(geometry, decaying_ring(geometry))
        before = buffer.copy()
        rt.process_field(buffer, geometry, tip, blanking, {}, average_fields=4)
        columns = geometry.samples_per_line
        head = geometry.first_measurable_line * columns
        assert np.array_equal(buffer[:head], before[:head])

    def test_a_degenerate_level_scale_degrades_quietly(self, geometry):
        buffer, _tip, blanking = self._buffer(geometry, None)
        before = buffer.copy()
        out, lti = rt.process_field(buffer, geometry, blanking, blanking, {},
                                    average_fields=4)
        assert np.array_equal(out, before)
        assert lti["gain"] == 0.0

    def test_a_field_of_noise_does_not_crash_or_correct(self, geometry):
        rng = np.random.default_rng(9)
        buffer = rng.normal(20000.0, 50.0,
                            263 * geometry.samples_per_line)
        before = buffer.copy()
        out, lti = rt.process_field(buffer, geometry, 16000.0, 20000.0, {},
                                    average_fields=4)
        assert np.array_equal(out, before)
        assert lti["gain"] == 0.0

    def test_the_chroma_gate_can_be_switched_off_by_the_stage_selection(
            self, geometry):
        buffer, tip, blanking = self._buffer(geometry, decaying_ring(geometry))
        shared = {"stage_selection":
                  {"components": {("ringing", "chroma_gate"): False}}}
        rt.process_field(buffer, geometry, tip, blanking, shared,
                         average_fields=4)
        assert shared[rt.STATE_KEY]["collapse"]["refused"] == []


class TestTheShapeReading:
    """The aftermath read in Hilbert space, with no order to choose."""

    def test_it_reports_three_axes_and_an_excess_phase(self, geometry):
        reading = rt.kernel_shape(decaying_ring(geometry), geometry)
        shape = reading["shape"]
        assert shape["amplitude"].shape == shape["frequency_hz"].shape
        assert shape["group_delay_s"].shape == shape["frequency_hz"].shape
        assert np.isfinite(reading["excess_phase_rms"])
        assert 0.0 <= reading["in_band_share"] <= 1.0

    def test_a_stronger_ring_reads_as_more_shape_and_no_more_noise(
            self, geometry):
        quiet = rt.kernel_shape(decaying_ring(geometry, amplitude=0.02),
                                geometry)["shape"]
        loud = rt.kernel_shape(decaying_ring(geometry, amplitude=0.20),
                               geometry)["shape"]
        assert loud["amplitude_rms"] > 5 * quiet["amplitude_rms"]

    def test_the_pencil_is_gone(self):
        """Ethan's ruling, held as a test so it cannot creep back."""
        assert not hasattr(rt, "estimate_decay_modes")
        assert not hasattr(rt, "decay_modes")
        assert not hasattr(rt, "pencil_capacity")


class TestTheTransientStageParameters:
    """Derived from the CORRECTOR's impulse response, with no coefficient."""

    def test_the_corrector_impulse_has_a_direct_tap(self, geometry):
        """The quantity the transient stage must read is what the whole
        stage does to a unit step, not the artifact on its own: the
        artifact has no direct tap, so a dispersion measured on it comes
        out near one and the stage then runs at almost full gain."""
        kernel = decaying_ring(geometry)
        impulse = rt.corrector_impulse(kernel, 0.5)
        assert impulse[0] == 1.0
        assert np.allclose(impulse[1:], -0.5 * kernel)
        bare = rt.derive_lti_parameters(kernel, 0.01)["gain"]
        composed = rt.derive_lti_parameters(impulse, 0.01)["gain"]
        assert composed < 0.1 < bare

    def test_a_kernel_with_all_its_energy_at_the_edge_asks_for_no_gain(self):
        kernel = np.zeros(32)
        kernel[0] = 1.0
        assert rt.derive_lti_parameters(kernel, 0.01)["gain"] == 0.0

    def test_a_dispersed_kernel_asks_for_more(self):
        spread = np.exp(-np.arange(32) / 8.0)
        tight = np.exp(-np.arange(32) / 1.0)
        assert (rt.derive_lti_parameters(spread, 0.01)["gain"]
                > rt.derive_lti_parameters(tight, 0.01)["gain"])
        assert (rt.derive_lti_parameters(spread, 0.01)["blur_radius"]
                > rt.derive_lti_parameters(tight, 0.01)["blur_radius"])

    def test_an_empty_kernel_is_neutral(self):
        assert rt.derive_lti_parameters(np.zeros(8), 0.01)["gain"] == 0.0


class TestTheOfflineWindowView:
    """The pooled view the export instrument reads, and nothing else."""

    def test_it_pools_across_calls_and_the_runtime_never_calls_it(
            self, geometry):
        plan = rt.plan_measurement_window(geometry)
        state = {}
        for seed in range(3):
            rt.accumulate_field_lines(
                synthetic_field(geometry, decaying_ring(geometry),
                                noise=0.2, seed=seed),
                geometry, plan, state)
        assert state["slow_fields_accumulated"] == 3
        assert state["slow_interval_mean"].shape == (plan.total_samples,)
        assert np.all(state["slow_interval_variance"] >= 0.0)
        assert np.isfinite(state["fall_edge_width_samples"])

    def test_pooling_lowers_the_scatter_of_the_mean(self, geometry):
        plan = rt.plan_measurement_window(geometry)
        one, many = {}, {}
        rt.accumulate_field_lines(
            synthetic_field(geometry, None, noise=0.5, seed=1),
            geometry, plan, one)
        for seed in range(8):
            rt.accumulate_field_lines(
                synthetic_field(geometry, None, noise=0.5, seed=seed),
                geometry, plan, many)
        # read on the tip's settled interior, past the transition's own
        # shape, so what is compared is scatter and not the edge
        window = slice(plan.sync_tip[0] + geometry.transition_settle_samples,
                       plan.sync_tip[1])
        assert np.std(many["slow_interval_mean"][window]) < 0.8 * np.std(
            one["slow_interval_mean"][window])

    def test_the_regions_are_ordered_and_non_empty(self, geometry):
        plan = rt.plan_measurement_window(geometry)
        for region in (plan.front_porch, plan.sync_tip, plan.back_porch):
            assert region[1] > region[0]
        assert plan.front_porch[1] <= plan.sync_tip[0]
        assert plan.sync_tip[1] <= plan.back_porch[0]
