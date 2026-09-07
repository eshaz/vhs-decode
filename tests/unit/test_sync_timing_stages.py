"""The sync and timing stages, wired and gated.

Ethan: 'Every stage that we have designed must be called and used ... All of
them need to be used and all of them need to model all dimensions.'

These tests hold three properties of the six stages this file covers:
their geometry is derived rather than declared, their corrections do what
they say on a signal whose answer is known by construction, and every one of
them is reachable from a decode by name and off unless named.
"""

import logging
import math
import types

import numpy as np
import pytest

import lddecode.core as ldd

from vhsdecode import model_stages, pipeline_graph
from vhsdecode.models import pair_dimension, precursor, sync_depth, sync_shape


GROUP = ("sync_shape", "precursor", "sync_depth", "burst_sync_lock",
         "vertical_interval", "tape_speed")


@pytest.fixture(autouse=True)
def _logger():
    """`ldd.logger` is None until a decode initialises it, and every stage
    below reports what it found. A test that constructs no decoder has to
    supply one or the reporting is what fails."""
    previous = getattr(ldd, "logger", None)
    if previous is None:
        ldd.logger = logging.getLogger("test_sync_timing_stages")
    yield
    ldd.logger = previous

# The output geometry of an NTSC four-times-subcarrier decode, which is what
# every figure below is quoted on.
SUBCARRIER_HZ = 315.0e6 / 88.0
RATE_HZ = 4.0 * SUBCARRIER_HZ
WIDTH = 910
HEIGHT = 263
LINE_US = 1e6 / (RATE_HZ / WIDTH)
BLANKING = 15360.0
UNITS_PER_IRE = 358.4


def _stub_field(spacing_ire=40.0, rows_with_sync=None, seed=1):
    """A field with the specified sync pulse on the specified lines.

    Built rather than decoded so the answer is known: the spacing below is
    what `sync_depth` must report, and nothing else in the field carries a
    transition the shape fit could confuse for a pulse.
    """
    rng = np.random.default_rng(seed)
    rf = types.SimpleNamespace(
        color_system="NTSC",
        SysParams={
            "outfreq": RATE_HZ / 1e6,
            "hsyncPulseUS": 4.7,
            "colorBurstUS": (5.3, 7.8),
            "activeVideoUS": (9.45, 62.5555555),
            "ire0": 3685714.2857142854,
            "hz_ire": 1e6 / 140.0,
        },
        DecoderParams={"video_lpf_freq": 6.6e6, "video_lpf_order": 9,
                       "video_lpf_supergauss": True},
        freq_hz=40e6, freq_hz_half=20e6, blocklen=4096,
        options=types.SimpleNamespace(
            tape_format="VHS", export_raw_tbc=False, ire0_adjust=(),
            sync_shape=0, precursor=0, sync_depth=0, burst_sync_lock=0,
            vertical_interval=0, tape_speed_stage=0),
    )
    rf.iretohz = lambda ire: (rf.SysParams["ire0"]
                              + ire * rf.SysParams["hz_ire"])
    # THE DECODER'S OWN FILTER TABLE, built the way the decoder builds it, so
    # `precursor` recovers a real zero-phase response rather than a stand-in:
    # `FVideo = FDeemp * lowpass`, and the module divides the causal
    # deemphasis back out to be left with the zero-phase factor alone.
    from vhsdecode.compute_video_filters import gen_video_lpf_params
    _, _lowpass = gen_video_lpf_params(rf.DecoderParams, rf.freq_hz_half,
                                       rf.blocklen)
    _grid = np.linspace(0.0, rf.freq_hz_half, rf.blocklen // 2 + 1)
    _deemp = 1.0 / (1.0 + 2j * np.pi * _grid * 1.0e-6)   # a causal one-pole
    rf.Filters = {"FVideo": _lowpass * _deemp, "FDeemp": _deemp}

    pulse = pair_dimension.spec_sync(RATE_HZ, WIDTH, depth_ire=-spacing_ire)
    # `spec_sync` centres the pulse; a decoded line begins at its leading
    # edge, so it is rolled back by half a line less half a pulse.
    pulse = np.roll(pulse, -(WIDTH // 2) + int(round(
        0.5 * pair_dimension.SYNC_WIDTH_S * RATE_HZ)))
    lines = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    rows = (precursor.sync_rows(HEIGHT, "NTSC") if rows_with_sync is None
            else np.asarray(rows_with_sync))
    for row in range(HEIGHT):
        if row in set(int(r) for r in rows) or row in set(
                int(r) - 1 for r in rows):
            lines[row] += pulse
    lines = lines * UNITS_PER_IRE + BLANKING
    lines += rng.normal(0.0, 0.5, lines.shape)
    picture = np.clip(lines, 0, 65535).reshape(-1).astype(np.uint16)

    field = types.SimpleNamespace(
        rf=rf, outlinelen=WIDTH, outlinecount=HEIGHT, lineoffset=1,
        isFirstField=True, dspicture=picture)
    field.usectooutpx = lambda us: us * RATE_HZ * 1e-6
    field.hz_to_output = lambda hz: np.uint16(
        round(BLANKING + (hz - rf.SysParams["ire0"])
              / rf.SysParams["hz_ire"] * UNITS_PER_IRE))
    field.level_windows = lambda: (
        (int(math.ceil(0.30 * 4.7 * RATE_HZ * 1e-6)),
         int(math.floor(0.70 * 4.7 * RATE_HZ * 1e-6))),
        (int(math.ceil(7.8 * RATE_HZ * 1e-6)) + 2,
         int(math.floor(9.45 * RATE_HZ * 1e-6)) - 4))
    return field, picture


class TestTheDeclaration:
    """Every one of the six is a node, off by default, with no flag."""

    @pytest.fixture(scope="class")
    @classmethod
    def declared(cls):
        return pipeline_graph.load()

    def test_all_six_are_declared_nodes(self, declared):
        for name in GROUP:
            assert name in declared["by_name"], f"{name} is not in the graph"

    def test_each_carries_all_three_axes(self, declared):
        for name in GROUP:
            axes = declared["by_name"][name]["axis"]
            assert set(axes) == {"amplitude", "frequency", "time"}, (
                f"{name} declares {axes}; Ethan's requirement is all three")

    def test_the_declaration_owns_the_default_and_it_is_on(self, declared):
        """Ethan, 2026-09-07: "I want all the stages defaulted to on, remove
        any flags that turn off stages."

        This test asserted the opposite until he said that, and the reason
        it did was sound at the time: a stage that defaults off cannot
        change a decode nobody asked it to change, and byte identity
        against an old baseline was the safety property being protected.
        He has overridden it deliberately, and the reason is his to give -
        he cannot see a correction that never runs. What survives of the
        old rule is that the DECLARATION still owns the default, so the
        value lives in one file and not in thirty-six argument parsers.
        """
        for name in GROUP:
            gate = declared["by_name"][name]["gate"]
            verdict = pipeline_graph.declared_default(gate)
            assert verdict["seed"], f"{name}'s default is not the graph's"
            assert verdict["value"], f"{name} defaults off"

    def test_no_gate_shares_a_name_with_an_existing_option(self, declared):
        """`--tape_speed` already carries the SP / LP / EP string, which is
        truthy - a gate named after the module ran the stage on every decode
        until this was caught."""
        assert declared["by_name"]["tape_speed"]["gate"]["option"] \
            == "tape_speed_stage"

    def test_they_can_be_switched_on_by_name(self, declared):
        options = {}
        selection = pipeline_graph.parse_selection(
            ",".join("+" + name for name in GROUP), declared)
        pipeline_graph.apply_selection(options, selection, declared)
        for name in GROUP:
            option = declared["by_name"][name]["gate"]["option"]
            assert options[option], f"--stages +{name} did not turn it on"

    def test_every_site_resolves_to_a_real_line(self, declared):
        for name in GROUP:
            site = pipeline_graph.resolve_site(declared["by_name"][name])
            assert site and ":" in site, f"{name} does not cite a live site"


class TestTheOutputScale:
    """`--level_adjust` spreads the black and white levels the decoder
    writes, and every offline IRE reading in this arc was short by that
    factor until it was inverted."""

    def _parameters(self, adjust):
        return {"blanking16bIre": BLANKING,
                "black16bIre": (BLANKING + 7.5 * UNITS_PER_IRE) * (1 - adjust),
                "white16bIre": (BLANKING + 100.0 * UNITS_PER_IRE) * (1 + adjust),
                "system": "NTSC"}

    @pytest.mark.parametrize("adjust", [0.0, 0.05, 0.1, 0.25])
    def test_the_adjustment_is_inverted_exactly(self, adjust):
        got = precursor.output_scale(self._parameters(adjust))
        assert got["inverted"]
        assert got["units_per_ire"] == pytest.approx(UNITS_PER_IRE, rel=1e-9)
        assert got["level_adjust"] == pytest.approx(adjust, abs=1e-9)
        assert got["black_ire"] == pytest.approx(7.5, abs=1e-6)

    def test_the_naive_reading_is_the_one_that_was_wrong(self):
        """(white16bIre - blanking16bIre) / 100 is what every offline reader
        in this tree used, and on the shipped default it is 14.3 per cent
        too large - which made every IRE it reported short by that much."""
        parameters = self._parameters(0.1)
        naive = (parameters["white16bIre"]
                 - parameters["blanking16bIre"]) / 100.0
        assert naive > UNITS_PER_IRE * 1.14
        assert precursor.output_scale(parameters)["units_per_ire"] \
            == pytest.approx(UNITS_PER_IRE, rel=1e-9)

    def test_a_caller_that_knows_the_value_may_supply_it(self):
        got = precursor.output_scale(self._parameters(0.1), level_adjust=0.1)
        assert got["units_per_ire"] == pytest.approx(UNITS_PER_IRE, rel=1e-9)


class TestTheFrequencyAxis:
    """`sync_depth` read two axes and claimed a stage; `depth_spectrum` is
    the third, and it has to recover a gain and a tilt that were planted."""

    def _pulses(self, gain=1.0, count=64, seed=3):
        rng = np.random.default_rng(seed)
        ideal = pair_dimension.spec_sync(RATE_HZ, 128, depth_ire=-40.0)
        return np.asarray([gain * ideal + rng.normal(0, 0.02, ideal.size)
                           for _ in range(count)])

    def test_a_flat_gain_comes_back_at_dc(self):
        got = sync_depth.depth_spectrum(self._pulses(0.85), RATE_HZ)
        assert got["dc_gain"] == pytest.approx(0.85, rel=0.02)
        assert abs(got["tilt_db_per_mhz"]) < 0.5

    def test_the_transfer_is_complex(self):
        got = sync_depth.depth_spectrum(self._pulses(0.9), RATE_HZ)
        assert np.iscomplexobj(got["transfer"])
        assert np.any(np.abs(np.angle(got["transfer"])) > 0)

    def test_a_single_pulse_is_refused(self):
        with pytest.raises(ValueError):
            sync_depth.depth_spectrum(
                pair_dimension.spec_sync(RATE_HZ, 128), RATE_HZ)


class TestTheGeometryIsDerived:
    def test_the_population_is_the_specification_s(self):
        field, _ = _stub_field()
        spans = model_stages.sync_spans(field)
        assert spans is not None
        assert spans["rows"][0] == 9      # the interval's own nine lines
        # SMPTE 32M puts the head switch 5 to 8 H ahead of vertical sync with
        # a one-line transient, so the last nine rows are dropped
        assert spans["rows"][-1] == HEIGHT - 10
        assert spans["units_per_ire"] == pytest.approx(UNITS_PER_IRE, rel=1e-6)

    def test_the_segment_holds_the_porch_and_stops_before_the_burst(self):
        field, _ = _stub_field()
        spans = model_stages.sync_spans(field)
        assert spans["front_porch"] == int(math.ceil(1.5e-6 * RATE_HZ))
        assert spans["segment"] <= spans["front_porch"] + int(
            5.3e-6 * RATE_HZ) + 1

    def test_an_unsupported_format_declines_rather_than_borrowing(self):
        field, _ = _stub_field()
        field.rf.options.tape_format = "BETAMAX"
        field.__dict__.pop("_sync_spans", None)
        assert model_stages.sync_spans(field)["band_limit_hz"] is None
        assert model_stages.sync_pulses(field, field.dspicture) is None


class TestSyncShape:
    def test_it_returns_three_axes_and_a_complex_spectrum(self):
        field, picture = _stub_field()
        got = model_stages.measure_sync_shape(field, picture)
        assert got is not None
        assert got["amplitude"].size == got["frequency_hz"].size
        assert np.iscomplexobj(got["spectrum"])
        assert got["band_limit_hz"] == sync_shape.VHS_LUMA_BAND_HZ
        # the specified pulse is 40 IRE deep, so the shape carries real
        # amplitude and the out-of-band remainder is the planted noise
        assert got["amplitude_rms_ire"] > 5.0
        assert 0.0 < got["noise_rms_ire"] < 1.0

    def test_it_applies_nothing(self):
        field, picture = _stub_field()
        before = picture.copy()
        model_stages.measure_sync_shape(field, picture)
        assert np.array_equal(before, picture)


class TestSyncDepth:
    def test_a_planted_deficit_is_measured_and_removed(self):
        field, picture = _stub_field(spacing_ire=34.0)
        got = model_stages.correct_sync_depth(field, picture)
        assert got is not None
        assert got["spacing_ire"] == pytest.approx(34.0, abs=0.3)
        assert got["gain"] == pytest.approx(34.0 / 40.0, abs=0.01)
        # and reading the corrected picture back gives the specified depth
        field.__dict__.pop("_sync_pulses", None)
        field.__dict__.pop("_sync_spans", None)
        field.rf.__dict__.pop("_model_stages_logged", None)
        again = model_stages.correct_sync_depth(field, picture)
        assert again["spacing_ire"] == pytest.approx(40.0, abs=0.3)

    def test_it_does_not_move_the_zero(self):
        field, picture = _stub_field(spacing_ire=34.0)
        spans = model_stages.sync_spans(field)
        before = np.asarray(picture, dtype=np.float64).reshape(
            HEIGHT, WIDTH)[spans["rows"], spans["porch"][0]:spans["porch"][1]]
        model_stages.correct_sync_depth(field, picture)
        after = np.asarray(picture, dtype=np.float64).reshape(
            HEIGHT, WIDTH)[spans["rows"], spans["porch"][0]:spans["porch"][1]]
        moved = abs(float(after.mean()) - float(before.mean())) / UNITS_PER_IRE
        assert moved < 0.05, "a gain about blanking must leave blanking put"

    def test_it_declines_beside_ire0_adjust_hsync(self):
        field, picture = _stub_field(spacing_ire=34.0)
        field.rf.options.ire0_adjust = ("backporch", "hsync")
        before = picture.copy()
        assert model_stages.correct_sync_depth(field, picture) is None
        assert np.array_equal(before, picture)

    def test_a_tip_on_the_container_floor_is_refused(self):
        """The censored fit still returns a number when almost nothing
        survives the floor, and on the countdown decode it returned a
        spacing of 107.7 IRE from a window 78.9 per cent pinned - a gain of
        2.69 that passed the module's own plausibility guard. A fit whose
        mean lies further below the floor than its own sigma reports a level
        that was never observed, and the stage must decline."""
        field, picture = _stub_field(spacing_ire=40.0)
        lines = np.asarray(picture, dtype=np.float64).reshape(HEIGHT, WIDTH)
        spans = model_stages.sync_spans(field)
        # A pulse interior whose true level sits below the container floor
        # with real scatter about it, so what survives is a genuine censored
        # sample rather than an empty one: 79 per cent pinned, which is what
        # the countdown decode showed.
        rng = np.random.default_rng(11)
        window = lines[:, spans["tip"][0]:spans["tip"][1]]
        lines[:, spans["tip"][0]:spans["tip"][1]] = np.clip(
            rng.normal(-0.55 * 300.0, 300.0, window.shape), 0.0, 65535.0)
        picture[:] = np.clip(lines, 0, 65535).reshape(-1).astype(np.uint16)
        field.__dict__.pop("_sync_pulses", None)
        pinned = (picture.reshape(HEIGHT, WIDTH)[
            :, spans["tip"][0]:spans["tip"][1]] <= 0).mean()
        assert 0.7 < pinned < 0.9, "the plant did not censor what it meant to"
        before = picture.copy()
        assert model_stages.correct_sync_depth(field, picture) is None
        assert np.array_equal(before, picture)

    def test_the_frequency_axis_travels_with_the_correction(self):
        field, picture = _stub_field(spacing_ire=34.0)
        got = model_stages.correct_sync_depth(field, picture)
        assert got["spectrum"] is not None
        assert np.iscomplexobj(got["spectrum"]["transfer"])


class TestPrecursor:
    def test_it_removes_a_ripple_ahead_of_the_edge_and_nothing_behind(self):
        field, picture = _stub_field()
        spans = model_stages.sync_spans(field)
        before = np.asarray(picture, dtype=np.float64).reshape(HEIGHT, WIDTH)
        kept = before[spans["rows"], spans["porch"][0]:spans["porch"][1]].copy()
        got = model_stages.correct_precursor(field, picture)
        assert got is not None
        assert got["kernel"]["split_point_verified"]
        after = np.asarray(picture, dtype=np.float64).reshape(HEIGHT, WIDTH)
        # the back porch is behind the edge and must be untouched
        assert np.array_equal(
            kept, after[spans["rows"], spans["porch"][0]:spans["porch"][1]])
        # and the front porch is not
        changed = np.abs(after - before).max()
        assert changed > 0

    def test_the_support_is_measured_against_the_field_s_own_noise(self):
        field, picture = _stub_field()
        got = model_stages.correct_precursor(field, picture)
        assert got["support_samples"] > 0
        assert got["support_us"] > 0


class TestVerticalIntervalDeclines:
    def test_a_blank_reserved_interval_is_refused(self):
        field, picture = _stub_field()
        got = model_stages.measure_vertical_interval(field, picture)
        assert got is not None
        assert got["present"] is False
        assert got["floor"] > 0

    def test_the_floor_is_derived_from_the_band_and_the_window(self):
        """2 B T independent samples, so the null correlation's spread is
        1/sqrt(2 B T) and the threshold is the detector's own significance
        of that - no number chosen anywhere."""
        field, picture = _stub_field()
        got = model_stages.measure_vertical_interval(field, picture)
        duration = (62.5555555 - 9.45) * 1e-6
        expected = 4.0 / math.sqrt(2.0 * sync_shape.VHS_LUMA_BAND_HZ * duration)
        assert got["floor"] == pytest.approx(expected, rel=0.02)

    def test_it_applies_nothing(self):
        field, picture = _stub_field()
        before = picture.copy()
        model_stages.measure_vertical_interval(field, picture)
        assert np.array_equal(before, picture)



    def test_picture_spilled_into_the_reserved_interval_is_excluded(self):
        """A source that starts its picture inside the interval the standard
        reserves offered the stage a perfect match to the specified colour
        bars on row 20 - a correct match to the wrong object. A test line is
        an ISLAND in a blank interval; spilled picture is a RUN that keeps
        going, and position is the only thing that separates them."""
        field, picture = _stub_field()
        spans = model_stages.sync_spans(field)
        lines = np.asarray(picture, dtype=np.float64).reshape(HEIGHT, WIDTH)
        rng = np.random.default_rng(7)
        # picture from the last reserved row downwards, as that tape has it
        for row in range(20, HEIGHT):
            lines[row, 200:880] += rng.uniform(0, 90, 680) * UNITS_PER_IRE
        picture[:] = np.clip(lines, 0, 65535).reshape(-1).astype(np.uint16)
        field.__dict__.pop("_sync_pulses", None)
        got = model_stages.measure_vertical_interval(field, picture)
        assert got is not None and got["present"] is False


class TestTapeSpeed:
    def test_a_planted_speed_error_comes_back_on_the_time_axis(self):
        field, picture = _stub_field()
        spans = model_stages.sync_spans(field)
        expected = spans["line_samples"] / spans["sample_rate_hz"]
        epsilon = 0.004
        period = expected / (1.0 + epsilon) * field.rf.freq_hz
        field.linelocs2 = np.arange(HEIGHT + 8, dtype=np.float64) * period
        got = model_stages.measure_tape_speed(field)
        assert got is not None
        assert got["epsilon"] == pytest.approx(epsilon, rel=1e-6)
        assert got["wavelength_scaling"] == pytest.approx(1.0 + epsilon,
                                                          rel=1e-9)
        # the frequency axis exists and is derived, not measured
        assert got["loss_departure_nepers"].size == got["frequency_hz"].size
        assert got["peak_departure_db"] > 0

    def test_it_declines_without_pre_correction_positions(self):
        field, picture = _stub_field()
        assert model_stages.measure_tape_speed(field) is None