"""The single transform's wiring into the decode.

The plan puts one picture stage and one RF stage in place of the four gated
groups `downscale` ran and the field-level chroma list `decode_chroma` ran.
This file holds the WIRING to that contract and nothing of the transforms
themselves, which `model_stages` owns:

  * `decode_chroma` hands the field and its chroma to
    `model_stages.transform_picture` when `picture_transform` is on, runs the
    legacy list when it is off, hands None for the chroma of a format that
    writes none, and runs `cti` last in both cases;
  * `downscale` defers the three luma groups to that transform and stands
    `model_stages.transform_rf` in for the radio-frequency group;
  * `lock_to_burst` stops rebuilding `LumaPathEQ` while the transform's
    table is live;
  * `demodblock` multiplies by the transform's per-block table at the
    equalizer site - after the envelope is taken, below a user-supplied
    `ChannelEQ` and above `LumaPathEQ`;
  * the cache's worker tells `demodblock` where each block starts;
  * `readfield` repays the blocks demodulated before the table latched,
    once, exactly as the head-switch one-shot does;
  * `burst_sparing_mask` reproduces the taper it was extracted from.

Every transform function is replaced by a spy here, so these tests hold
whether or not the transforms have bodies yet - and they test the wiring
only, never what a transform does with what it is handed.
"""

import inspect
import logging
import queue
import re
import types

import numpy as np
import numpy.fft as npfft
import pytest

import lddecode.core as ldd

from vhsdecode import chroma as chroma_module
from vhsdecode import field as field_module
from vhsdecode import model_stages
from vhsdecode import process as process_module
from vhsdecode.demodcache import DemodCacheTape


@pytest.fixture(autouse=True)
def _logger():
    """`ldd.logger` is None until a decode initialises it; the demodulator
    warns through it when an envelope touches zero."""
    previous = getattr(ldd, "logger", None)
    if previous is None:
        ldd.logger = logging.getLogger("test_single_transform_wiring")
    yield
    ldd.logger = previous


class _Spy:
    """Records every call, in a log shared between spies so that the ORDER
    of calls across them is kept as well as the count."""

    def __init__(self, log, name, result=None):
        self.log = log
        self.name = name
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.log.append(self.name)
        return self.result


# ---------------------------------------------------------------------------
# decode_chroma


def _chroma_field(picture_transform, write_chroma=True, cti_mix=1,
                  selection=None):
    options = types.SimpleNamespace(
        write_chroma=write_chroma, picture_transform=picture_transform,
        disable_comb=False, cti_mix=cti_mix, stage_selection=selection)
    return types.SimpleNamespace(rf=types.SimpleNamespace(options=options))


@pytest.fixture
def chroma_spies(monkeypatch):
    """`process_chroma`, the transform, one legacy stage and `cti`, all
    replaced. The legacy list carries `cti` as its last entry exactly as the
    declared list does, and the transform path reaches the same spy through
    the module attribute, so one spy answers for the sharpener on both."""
    log = []
    uphet = np.zeros(64, dtype=np.float64)
    spies = types.SimpleNamespace(
        log=log,
        uphet=uphet,
        process=_Spy(log, "process_chroma", result=uphet),
        transform=_Spy(log, "transform_picture"),
        legacy=_Spy(log, "legacy_stage"),
        cti=_Spy(log, "cti"),
    )
    monkeypatch.setattr(chroma_module, "process_chroma", spies.process)
    monkeypatch.setattr(model_stages, "transform_picture", spies.transform,
                        raising=False)
    monkeypatch.setattr(chroma_module, "FIELD_CHROMA_STAGES", (
        chroma_module._FieldChromaStage("burst_instrument", spies.legacy),
        chroma_module._FieldChromaStage("cti", spies.cti),
    ))
    monkeypatch.setattr(chroma_module, "apply_chroma_transient_improvement",
                        spies.cti)
    return spies


def test_the_transform_stands_in_for_the_legacy_chroma_list(chroma_spies):
    field = _chroma_field(picture_transform=1)
    out = chroma_module.decode_chroma(field)
    assert len(chroma_spies.transform.calls) == 1
    (called_field, called_uphet), _ = chroma_spies.transform.calls[0]
    assert called_field is field
    assert called_uphet is chroma_spies.uphet
    assert chroma_spies.legacy.calls == []
    assert field.uphet_temp is chroma_spies.uphet
    assert out.dtype == np.uint16 and len(out) == len(chroma_spies.uphet)


def test_the_legacy_list_runs_when_the_transform_is_off(chroma_spies):
    field = _chroma_field(picture_transform=0)
    out = chroma_module.decode_chroma(field)
    assert chroma_spies.transform.calls == []
    assert len(chroma_spies.legacy.calls) == 1
    assert chroma_spies.log == ["process_chroma", "legacy_stage", "cti"]
    assert out.dtype == np.uint16


def test_a_format_that_writes_no_chroma_still_hands_the_luma_over(
        chroma_spies):
    field = _chroma_field(picture_transform=1, write_chroma=False)
    assert chroma_module.decode_chroma(field) is None
    assert chroma_spies.process.calls == []
    assert len(chroma_spies.transform.calls) == 1
    (called_field, called_uphet), _ = chroma_spies.transform.calls[0]
    assert called_field is field
    assert called_uphet is None
    # there is no chroma for the sharpener to run on
    assert chroma_spies.cti.calls == []
    assert not hasattr(field, "uphet_temp")


def test_without_the_transform_a_chroma_less_format_is_untouched(
        chroma_spies):
    field = _chroma_field(picture_transform=0, write_chroma=False)
    assert chroma_module.decode_chroma(field) is None
    assert chroma_spies.log == []


def test_cti_runs_last_after_the_transform(chroma_spies):
    field = _chroma_field(picture_transform=1, cti_mix=1)
    chroma_module.decode_chroma(field)
    assert chroma_spies.log == ["process_chroma", "transform_picture", "cti"]
    (called_field, called_uphet), _ = chroma_spies.cti.calls[0]
    assert called_field is field
    assert called_uphet is chroma_spies.uphet


def test_cti_still_answers_to_its_name_after_the_transform(chroma_spies):
    field = _chroma_field(picture_transform=1,
                          selection={"stages": {"cti": False}})
    chroma_module.decode_chroma(field)
    assert chroma_spies.log == ["process_chroma", "transform_picture"]


def test_cti_mix_zero_declines_after_the_transform(monkeypatch):
    """The sharpener reads `cti_mix` itself: the wiring hands it the chroma
    and it declines, leaving the chroma exactly as the transform left it.
    The real sharpener runs here, not a spy."""
    log = []
    uphet = np.linspace(-100.0, 100.0, 64)
    before = uphet.copy()
    monkeypatch.setattr(chroma_module, "process_chroma",
                        _Spy(log, "process_chroma", result=uphet))
    monkeypatch.setattr(model_stages, "transform_picture",
                        _Spy(log, "transform_picture"), raising=False)
    field = _chroma_field(picture_transform=1, cti_mix=0)
    chroma_module.decode_chroma(field)
    assert log == ["process_chroma", "transform_picture"]
    assert np.array_equal(uphet, before)


# ---------------------------------------------------------------------------
# downscale


LUMA_GROUP = (
    "measure_sync_shape", "correct_precursor", "correct_sync_depth",
    "measure_vertical_interval", "measure_tape_speed",
    "measure_head_model", "measure_head_differential", "measure_magnetic",
    "measure_magnetic_circuit", "measure_tape_path",
    "measure_level_from_frequency", "measure_source_agc", "measure_vcr_agc",
    "measure_composite_channel", "measure_multipath",
    "correct_standard_levels", "correct_source_response",
)
RF_GROUP = (
    "measure_capture_profile", "measure_capture_filter", "measure_rf_stages",
    "measure_filter_model", "measure_transport", "measure_band_delay",
    "measure_head_switch_pair", "measure_interference",
)
GROUP_OPTIONS = (
    "sync_shape", "precursor", "sync_depth", "vertical_interval",
    "tape_speed_stage",
    "capture_profile", "capture_filter", "rf_stages", "filter_model",
    "transport_model", "band_delay", "head_switch_pair", "interference",
    "head_model", "head_differential", "magnetic", "magnetic_circuit",
    "tape_path",
    "level_from_frequency", "source_agc", "vcr_agc", "composite_channel",
    "multipath", "standard_levels", "source_correction",
)


class _Resampled:
    """Stands where lddecode's Field stands in the MRO: the parent
    `downscale`, which hands back the resampled picture in Hz."""

    def downscale(self, final=False, *args, **kwargs):
        return self._resampled.copy(), None, None


class _Field(field_module.FieldShared, _Resampled):
    def __init__(self, rf, resampled):
        self.rf = rf
        self._resampled = resampled

    def hz_to_output(self, input):
        """The decoder's own mapping reads the track phase, the level
        windows and the ire0 modes; none of that is what these tests are
        about, so the plain mapping stands in for it here."""
        params = self.rf.DecoderParams
        ire = (input - params["ire0"]) / params["hz_ire"] - params["vsync_ire"]
        return np.uint16(np.clip(ire + self.rf.SysParams["outputZero"],
                                 0, 65535) + 0.5)


def _downscale_field(picture_transform, rf_transform):
    options = dict((name, 1) for name in GROUP_OPTIONS)
    options.update(picture_transform=picture_transform,
                   rf_transform=rf_transform, carrier_tbc=0, y_comb=0,
                   inverse_eq=-1, lti_gain=None, export_raw_tbc=False,
                   ire0_adjust=())
    rf = types.SimpleNamespace(
        options=types.SimpleNamespace(**options), debug_plot=None,
        DecoderParams={"ire0": 3.7e6, "hz_ire": 1e6 / 140.0,
                       "vsync_ire": -40.0},
        SysParams={"outputZero": 1024.0})
    return _Field(rf, np.full(2048, 3.7e6))


@pytest.fixture
def group_spies(monkeypatch):
    log = []
    spies = {}
    for name in LUMA_GROUP + RF_GROUP:
        spies[name] = _Spy(log, name)
        monkeypatch.setattr(model_stages, name, spies[name])
    spies["transform_rf"] = _Spy(log, "transform_rf")
    monkeypatch.setattr(model_stages, "transform_rf", spies["transform_rf"],
                        raising=False)
    return spies


def _fired(spies):
    return sorted(name for name, spy in spies.items() if spy.calls)


def test_both_transforms_on_defer_every_legacy_group(group_spies):
    field = _downscale_field(picture_transform=1, rf_transform=1)
    dsout, _, _ = field.downscale(final=True)
    assert _fired(group_spies) == ["transform_rf"]
    (called,), _ = group_spies["transform_rf"].calls[0]
    assert called is field
    # the picture the transform reads is the one on the output scale
    assert field.dspicture is dsout and dsout.dtype == np.uint16


def test_both_transforms_off_run_every_legacy_group(group_spies):
    field = _downscale_field(picture_transform=0, rf_transform=0)
    field.downscale(final=True)
    assert _fired(group_spies) == sorted(LUMA_GROUP + RF_GROUP)


def test_the_two_transforms_are_independent(group_spies):
    field = _downscale_field(picture_transform=1, rf_transform=0)
    field.downscale(final=True)
    assert _fired(group_spies) == sorted(RF_GROUP)
    for spy in group_spies.values():
        spy.calls.clear()
    field = _downscale_field(picture_transform=0, rf_transform=1)
    field.downscale(final=True)
    assert _fired(group_spies) == sorted(LUMA_GROUP + ("transform_rf",))


def test_a_non_final_downscale_runs_no_stage_either_way(group_spies):
    for picture_transform, rf_transform in ((1, 1), (0, 0)):
        field = _downscale_field(picture_transform, rf_transform)
        field.downscale(final=False)
        assert _fired(group_spies) == []


# ---------------------------------------------------------------------------
# process


class _Processed:
    """Stands where lddecode's Field stands in the MRO: the parent
    `process`, which here has nothing to do."""

    def process(self):
        pass


class _ProcessField(field_module.FieldShared, _Processed):
    def __init__(self, rf):
        self.rf = rf
        self.prevfield = None
        self.valid = True


def _process_field(**options):
    gates = dict(luma_eq=0, luma_transient=0, head_switch=0, rf_transform=0)
    gates.update(options)
    return _ProcessField(types.SimpleNamespace(
        color_system="NTSC", options=types.SimpleNamespace(**gates)))


def test_the_rf_transform_feeds_the_amplitude_measurement(monkeypatch):
    measured = _Spy([], "measured_amplitude_deviation")
    monkeypatch.setattr(field_module.luma_amplitude,
                        "measured_amplitude_deviation", measured)
    field = _process_field(rf_transform=1)
    field.process()
    assert len(measured.calls) == 1 and measured.calls[0][0][0] is field
    measured.calls.clear()
    _process_field().process()
    assert measured.calls == []


# ---------------------------------------------------------------------------
# lock_to_burst


def _burst_lock_field(luma_eq=1):
    rf = types.SimpleNamespace(
        options=types.SimpleNamespace(
            luma_eq=luma_eq, luma_transient=0, detect_chroma_track_phase=False),
        DecoderParams={})
    return types.SimpleNamespace(rf=rf)


@pytest.fixture
def burst_lock_spies(monkeypatch):
    log = []
    spies = types.SimpleNamespace(
        log=log,
        envelope=_Spy(log, "apply_chroma_envelope_gain", result=False),
        equalizer=_Spy(log, "update_luma_equalizer"),
        rotation=_Spy(log, "decode_chroma_phase_rotation",
                      result=(0, None, None, 0.0, 0.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(field_module, "apply_chroma_envelope_gain",
                        spies.envelope)
    monkeypatch.setattr(field_module.luma_amplitude, "update_luma_equalizer",
                        spies.equalizer)
    monkeypatch.setattr(field_module, "decode_chroma_phase_rotation",
                        spies.rotation)
    return spies


def test_the_luma_equalizer_is_not_rebuilt_while_the_table_is_live(
        burst_lock_spies, monkeypatch):
    live = _Spy(burst_lock_spies.log, "rf_transform_live", result=True)
    monkeypatch.setattr(model_stages, "rf_transform_live", live,
                        raising=False)
    field = _burst_lock_field()
    field_module.FieldShared.lock_to_burst(field)
    assert len(live.calls) == 1 and live.calls[0][0][0] is field.rf
    assert burst_lock_spies.equalizer.calls == []
    assert field.track_phase_set is True


def test_the_luma_equalizer_is_rebuilt_until_the_table_is_live(
        burst_lock_spies, monkeypatch):
    monkeypatch.setattr(
        model_stages, "rf_transform_live",
        _Spy(burst_lock_spies.log, "rf_transform_live", result=False),
        raising=False)
    field = _burst_lock_field()
    field_module.FieldShared.lock_to_burst(field)
    assert len(burst_lock_spies.equalizer.calls) == 1


def test_luma_eq_off_never_asks_whether_the_table_is_live(
        burst_lock_spies, monkeypatch):
    live = _Spy(burst_lock_spies.log, "rf_transform_live", result=False)
    monkeypatch.setattr(model_stages, "rf_transform_live", live,
                        raising=False)
    field_module.FieldShared.lock_to_burst(_burst_lock_field(luma_eq=0))
    assert live.calls == []
    assert burst_lock_spies.equalizer.calls == []


# ---------------------------------------------------------------------------
# demodblock


def _analytic_mask(n):
    """lddecode's hilbert mask: the one-sided spectrum, doubled."""
    mask = np.zeros(n)
    mask[0] = 1.0
    mask[1:n // 2] = 2.0
    mask[n // 2] = 1.0
    return mask


def _rf_stub(n=256, rf_transform=1, **filters):
    """A demodulator with the filters `demodblock` reads and a spy where the
    demodulation proper starts, which records what it was handed."""
    calls = []

    def _demodulate_to_video(hilbert, envelope=None, calibrate=True):
        calls.append((hilbert.copy(), envelope.copy()))
        video = np.zeros(n)
        return video, video, np.zeros(n // 2 + 1, dtype=complex), None, None

    options = types.SimpleNamespace(
        rf_transform=rf_transform, export_raw_tbc=False, color_under=True,
        chroma_offset=0, chroma_env_gain=0, luma_eq=0, luma_transient=0,
        head_switch=0, carrier_tbc=0, baseband_eq=0, channel_eq=0)
    return types.SimpleNamespace(
        blocklen=n, blockcut=8, blockcut_end=8, options=options,
        debug_plot=None, _notch=None, _high_boost=None,
        _residual_channels_dir=None, _baseband_eq_declared=False,
        _channel_eq_declared=False, _do_cafc=True,
        _demodulate_to_video=_demodulate_to_video, demod_calls=calls,
        Filters=dict({"RFVideo": np.ones(n), "hilbert": _analytic_mask(n),
                      "FVideo05": np.ones(n // 2 + 1), "F05_offset": 0},
                     **filters))


def _carrier(n):
    """A whole number of cycles, so the analytic envelope is exactly flat
    and the demodulator's weak-signal branch is never entered."""
    return np.cos(2.0 * np.pi * (n // 8) * np.arange(n) / n)


def _expected(rf, data, table=None):
    """What `demodblock` hands on: the analytic signal through the RF filter
    and the table, and the envelope taken BEFORE the table."""
    spectrum = npfft.fft(data) * rf.Filters["RFVideo"]
    envelope = np.abs(
        npfft.ifft(spectrum * rf.Filters["hilbert"])).astype(np.single)
    if table is not None:
        spectrum = spectrum * table
    return npfft.ifft(spectrum * rf.Filters["hilbert"]), envelope


def _demodblock(rf, data, block_start):
    return process_module.VHSRFDecode.demodblock(
        rf, data=data.copy(), block_start=block_start)


def test_a_supplied_channel_response_outranks_the_table(monkeypatch):
    n = 256
    channel = 1.0 + 0.25 * np.cos(2.0 * np.pi * np.arange(n) / n)
    rf = _rf_stub(n, ChannelEQ=channel, LumaPathEQ=np.full(n, 3.0))
    table = _Spy([], "rf_transform_table", result=np.full(n, 7.0))
    monkeypatch.setattr(model_stages, "rf_transform_table", table,
                        raising=False)
    data = _carrier(n)
    _demodblock(rf, data, block_start=4096)
    assert table.calls == []
    hilbert, envelope = rf.demod_calls[0]
    expected_hilbert, expected_envelope = _expected(rf, data, channel)
    assert np.array_equal(hilbert, expected_hilbert)
    assert np.array_equal(envelope, expected_envelope)


def test_the_table_multiplies_after_the_envelope_and_above_the_luma_eq(
        monkeypatch):
    n = 256
    rf = _rf_stub(n, LumaPathEQ=np.full(n, 3.0))
    published = 1.0 + 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
    table = _Spy([], "rf_transform_table", result=published)
    monkeypatch.setattr(model_stages, "rf_transform_table", table,
                        raising=False)
    data = _carrier(n)
    _demodblock(rf, data, block_start=4096)
    assert len(table.calls) == 1
    (called_rf, block_start, n_fft), _ = table.calls[0]
    assert called_rf is rf and block_start == 4096 and n_fft == n
    hilbert, envelope = rf.demod_calls[0]
    expected_hilbert, expected_envelope = _expected(rf, data, published)
    assert np.array_equal(hilbert, expected_hilbert)
    # the envelope is the tape's measurement and was taken before the table
    assert np.array_equal(envelope, expected_envelope)
    assert not np.array_equal(np.abs(hilbert).astype(np.single), envelope)


def test_no_table_yet_leaves_the_luma_equalizer_standing(monkeypatch):
    n = 256
    luma_eq = np.full(n, 3.0)
    rf = _rf_stub(n, LumaPathEQ=luma_eq)
    table = _Spy([], "rf_transform_table", result=None)
    monkeypatch.setattr(model_stages, "rf_transform_table", table,
                        raising=False)
    data = _carrier(n)
    _demodblock(rf, data, block_start=0)
    assert len(table.calls) == 1
    hilbert, _ = rf.demod_calls[0]
    assert np.array_equal(hilbert, _expected(rf, data, luma_eq)[0])


def test_no_table_and_no_equalizer_leaves_the_analytic_signal_alone(
        monkeypatch):
    n = 256
    rf = _rf_stub(n)
    monkeypatch.setattr(model_stages, "rf_transform_table",
                        _Spy([], "rf_transform_table", result=None),
                        raising=False)
    data = _carrier(n)
    out = _demodblock(rf, data, block_start=0)
    hilbert, envelope = rf.demod_calls[0]
    expected_hilbert, expected_envelope = _expected(rf, data)
    assert np.array_equal(hilbert, expected_hilbert)
    assert np.array_equal(envelope, expected_envelope)
    # and the transform's measurement has the raw demod to read
    assert "demod_raw" in out["video"]


def test_the_transform_off_never_consults_the_table(monkeypatch):
    n = 256
    rf = _rf_stub(n, rf_transform=0)
    table = _Spy([], "rf_transform_table", result=np.full(n, 7.0))
    monkeypatch.setattr(model_stages, "rf_transform_table", table,
                        raising=False)
    data = _carrier(n)
    out = _demodblock(rf, data, block_start=0)
    assert table.calls == []
    hilbert, _ = rf.demod_calls[0]
    assert np.array_equal(hilbert, _expected(rf, data)[0])
    assert "demod_raw" not in out["video"]


# ---------------------------------------------------------------------------
# the cache's worker


def test_the_worker_tells_the_demodulator_where_the_block_starts():
    cache = DemodCacheTape.__new__(DemodCacheTape)
    cache._gnrc_afe = False
    cache.blocksize = 1000
    cache.ended = True
    cache.q_in = queue.Queue()
    cache.q_out = queue.Queue()
    calls = []

    def demodblock(**kwargs):
        calls.append(kwargs)
        return {"video": {}}

    cache.rf = types.SimpleNamespace(demodblock=demodblock)
    block = {"rawinput": np.zeros(16), "fft": np.zeros(16, dtype=complex)}
    cache.q_in.put(("DEMOD", 7, block, 0, 3))
    cache.q_in.put(None)
    cache.worker()
    assert len(calls) == 1
    assert calls[0]["block_start"] == 7 * cache.blocksize
    assert calls[0]["fftdata"] is block["fft"]
    assert calls[0]["cut"] is True
    blocknum, output = cache.q_out.get_nowait()
    assert blocknum == 7 and output["request"] == 3


def test_the_block_start_is_where_the_cache_loads_the_block_from():
    """`blocknum * blocksize` is a claim about lddecode's cache, which this
    lane does not own, so the claim is held against its source: block b is
    loaded from `b * blocksize`, and blocksize is blocklen less the cuts."""
    source = open(inspect.getsourcefile(ldd)).read()
    assert "b * self.blocksize, self.rf.blocklen" in source
    assert ("self.blocksize       = self.rf.blocklen - "
            "(self.rf.blockcut + self.rf.blockcut_end)") in source


# ---------------------------------------------------------------------------
# readfield


def _decoder_stub(monkeypatch, rf_transform, head_switch=0, answers=(True,)):
    """`VHSDecode.readfield` on a decoder whose field decoding is a stub: the
    stub records every call and whether it was a redo, and the transform's
    request answers from `answers` once each, then False."""
    decoder = process_module.VHSDecode.__new__(process_module.VHSDecode)
    field = types.SimpleNamespace(
        valid=True, needrerun=False, isFirstField=True,
        downscale=lambda **kwargs: (None, None, None))
    calls = []

    def decodefield(start, mtf_level, prevfield=None, initphase=False,
                    redo=False, rv=None):
        calls.append({"start": start, "redo": redo})
        if rv is not None:
            rv["field"] = field
            rv["offset"] = 500
            return None
        return field, 500

    decoder.decodefield = decodefield
    decoder.computeMetrics = lambda f, metrics, verbose=False: {}
    decoder.fieldstack = [None]
    decoder.second_decode = None
    decoder.fields_written = 0
    decoder.decodethread = None
    decoder.threadreturn = {}
    decoder.fdoffset = 10_000
    decoder.mtf_level = 0
    decoder.numthreads = 0
    decoder.useAGC = False
    decoder.fname_out = None
    decoder.output_lines = 263
    decoder.analog_audio = False
    decoder.lastFieldWritten = None
    # `LDdecode.__del__` deletes this, so the stub has to carry it
    decoder.demodcache = None
    decoder.rf = types.SimpleNamespace(options=types.SimpleNamespace(
        rf_transform=rf_transform, head_switch=head_switch))
    pending = list(answers)
    wants = _Spy([], "transform_wants_redo")

    def transform_wants_redo(rf):
        wants(rf)
        return pending.pop(0) if pending else False

    monkeypatch.setattr(model_stages, "transform_wants_redo",
                        transform_wants_redo, raising=False)
    return decoder, calls, wants, field


def test_readfield_repays_the_blocks_once_when_the_table_latches(
        monkeypatch):
    decoder, calls, wants, field = _decoder_stub(monkeypatch, rf_transform=1)
    assert decoder.readfield() is field
    redos = [call for call in calls if call["redo"]]
    assert len(redos) == 1
    # The field is decoded again from its own start: `fdoffset` had advanced
    # by the field's offset before the redo was computed back from it.
    assert redos[0]["redo"] == 10_000 and redos[0]["start"] == 10_000
    # Asked once per assembled field - the first answered True and the
    # redone one False - and the redone field is the one kept.
    assert len(wants.calls) == 2
    assert decoder.fieldstack[0] is field


def test_readfield_does_not_ask_when_the_transform_is_off(monkeypatch):
    decoder, calls, wants, _ = _decoder_stub(monkeypatch, rf_transform=0)
    decoder.readfield()
    assert wants.calls == []
    assert [call for call in calls if call["redo"]] == []


def test_a_redo_already_owed_consumes_the_transforms_request(monkeypatch):
    """The head-switch one-shot owes a redo on the same field. The
    transform's request is consumed by that redo - it re-demodulates the
    same blocks against the table that is now live - rather than firing a
    second redo on the field after."""
    decoder, calls, wants, _ = _decoder_stub(
        monkeypatch, rf_transform=1, head_switch=1)
    monkeypatch.setattr(process_module.luma_amplitude, "block_model",
                        lambda rf: object())
    decoder.readfield()
    assert len([call for call in calls if call["redo"]]) == 1
    assert len(wants.calls) == 2
    assert decoder.rf.__dict__["_head_switch_redone"] is True


# ---------------------------------------------------------------------------
# the options


def test_the_two_gates_sit_together_in_both_option_lists():
    """The namedtuple is positional; the two names stand immediately after
    `tape_path` in the field list and in the value list alike."""
    source = open(inspect.getsourcefile(process_module)).read()
    start = source.index("self._options = namedtuple(")
    end = source.index("namedtuple(", source.index("namedtuple(", start) + 1)
    block = source[start:end]
    names = re.findall(r'^\s+"([a-z_0-9]+)",\s*$', block, re.MULTILINE)
    values = re.findall(r'rf_options\.get\("([a-z_0-9]+)"', block)
    for listing in (names, values):
        anchor = listing.index("tape_path")
        assert listing[anchor + 1:anchor + 3] == ["picture_transform",
                                                  "rf_transform"]


# ---------------------------------------------------------------------------
# burst_sparing_mask


def _inline_taper(field, phase):
    """The computation the mask was extracted from, verbatim, applied to
    `phase` in place."""
    rf = field.rf
    linelocs = getattr(field, "linelocs2", None)
    if linelocs is not None:
        fs_us = rf.freq_hz / 1e6
        burst_us = rf.SysParams["colorBurstUS"]
        out_us = 8.0 / (4.0 * rf.SysParams["fsc_mhz"])
        b_lo = int(round((burst_us[0] - 2.0 * out_us) * fs_us))
        b_hi = int(round((burst_us[1] + 2.0 * out_us) * fs_us))
        taper = int(round(0.25 * fs_us))
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
        n_ph = len(phase)
        for loc in np.asarray(linelocs, dtype=np.float64):
            a = int(round(loc))
            lo = a + b_lo - taper
            hi = a + b_hi + taper
            if lo < 0 or hi >= n_ph:
                continue
            phase[lo:lo + taper] *= ramp.astype(np.float32)
            phase[lo + taper:hi - taper] = 0.0
            phase[hi - taper:hi] *= ramp[::-1].astype(np.float32)
    return phase


def _taper_field(linelocs):
    rf = types.SimpleNamespace(
        freq_hz=40e6,
        SysParams={"colorBurstUS": (5.3, 7.8), "fsc_mhz": 315.0 / 88.0})
    field = types.SimpleNamespace(rf=rf)
    if linelocs is not None:
        field.linelocs2 = np.asarray(linelocs, dtype=np.float64)
    return field


def test_the_mask_reproduces_the_taper_it_was_extracted_from():
    rng = np.random.default_rng(3)
    n = 12000
    # three lines, and a fourth whose spared window would run past the end
    field = _taper_field([100.0, 2640.6, 5181.2, 11900.0])
    phase = rng.normal(0.0, 0.1, n).astype(np.float32)
    inline = _inline_taper(field, phase.copy())
    mask = chroma_module.burst_sparing_mask(field, n)
    assert mask.dtype == np.float32 and len(mask) == n
    assert np.array_equal(phase * mask, inline)
    # the spared span is really zero and the picture really unity
    assert np.any(mask == 0.0)
    assert mask[0] == 1.0 and mask[-1] == 1.0
    # and the line whose window runs past the end was skipped by both
    assert np.all(mask[11000:] == 1.0)


def test_the_mask_is_unity_without_line_locations():
    field = _taper_field(None)
    assert np.array_equal(chroma_module.burst_sparing_mask(field, 500),
                          np.ones(500, dtype=np.float32))


def test_the_luma_image_map_undoes_a_planted_image_on_the_analytic_signal():
    """Ethan: 'Apply the IQ imbalance to the luma channel as well ...
    correct it in the same existing correction pattern.' A planted pair
    on the carrier's complex baseband is undone on the analytic signal to
    machine precision, with the rotation's phase continuous from the
    block's absolute start."""
    import numpy as np
    from vhsdecode import model_stages

    rate = 40e6
    carrier = 3.9e6
    n = 4096
    block_start = 7 * 30720
    t = (np.arange(n) + block_start) / rate
    rng = np.random.default_rng(3)
    # a baseband with structure on both sides of the carrier
    baseband = (rng.normal(size=n) + 1j * rng.normal(size=n))
    baseband = np.fft.ifft(np.fft.fft(baseband) * (np.abs(np.fft.fftfreq(n, 1 / rate)) < 1e6))
    alpha, beta = 1.0 + 0.05j, 0.08 - 0.03j
    clean = baseband * np.exp(2j * np.pi * carrier * t)
    spoiled = (alpha * baseband + beta * np.conj(baseband)) * np.exp(2j * np.pi * carrier * t)
    image = {"alpha": alpha, "beta": beta, "carrier_hz": carrier, "amount": 1.0}
    restored = model_stages.apply_luma_image(spoiled, image, block_start, rate)
    assert np.max(np.abs(restored - clean)) < 1e-9 * np.max(np.abs(clean))
    # half the amount moves halfway
    half = model_stages.apply_luma_image(spoiled, dict(image, amount=0.5), block_start, rate)
    assert np.allclose(half, spoiled + 0.5 * (clean - spoiled))
    # no block position, no map
    assert model_stages.apply_luma_image(spoiled, image, None, rate) is spoiled


def test_the_worker_reads_the_image_from_the_published_snapshot():
    import numpy as np
    from vhsdecode import model_stages
    from vhsdecode.models import single_transform as st

    class Rf:
        pass
    rf = Rf()
    rf.__dict__["_rf_transform"] = st.publish(
        1, np.ones(8), None, None, extras={"image": {
            "alpha": 1.0, "beta": 0.1j, "carrier_hz": 3.9e6, "amount": 0.5}})
    image = model_stages.rf_transform_image(rf)
    assert image is not None and image["amount"] == 0.5
    rf.__dict__["_rf_transform"] = st.publish(1, np.ones(8), None, None)
    assert model_stages.rf_transform_image(rf) is None
    # a singular pair is refused
    rf.__dict__["_rf_transform"] = st.publish(
        1, np.ones(8), None, None, extras={"image": {
            "alpha": 1.0, "beta": 1.0, "carrier_hz": 3.9e6}})
    assert model_stages.rf_transform_image(rf) is None
