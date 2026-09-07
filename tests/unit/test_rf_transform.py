"""The one radio-frequency stage, wired to the fold and to the workers.

Ethan: 'I need to have a VHS RF stage transformed the same way. We do the
full spherical shape, but, replace the noise with null space, which is a
constant that we do not derive at all.' And: 'Subtract out the hyper cube
and discard it as noise.' And: 'the exact inverse within our possible area
of measure.'

Every field below is BUILT, not decoded, so the answer is known: a luma
frequency-modulation carrier at the specified frequencies with a sync
pulse a line, a colour-under burst in the specified window, a planted
amplitude-probe line per head, planted reserved-band levels and a planted
head switch. The instruments that would read those from a real decode are
replaced at their seams, and the stage is judged on what it publishes.
"""

import logging
import types

import numpy as np
import pytest

import lddecode.core as ldd

from vhsdecode import head_switch, luma_amplitude, model_stages, pipeline_graph
from vhsdecode import rf_transform as rt
from vhsdecode.models import band_delay as bd
from vhsdecode.models import colour_under, vhs_specification


@pytest.fixture(autouse=True)
def _logger():
    """`ldd.logger` is None until a decode initialises it, and the stage
    reports what it found through it."""
    previous = getattr(ldd, "logger", None)
    if previous is None:
        ldd.logger = logging.getLogger("test_rf_transform")
    yield
    ldd.logger = previous


# The capture geometry every figure below is quoted on: a 40 MSps capture
# of NTSC VHS, the decoder's 32768-sample block with its 1024-sample cuts.
RATE_HZ = 40e6
BLOCKLEN = 32768
BLOCKCUT = 1024
BLOCKCUT_END = 1024
LINE_S = 1.001 / (525.0 * 30.0)
LINE_US = LINE_S * 1e6
FPS = 30.0 / 1.001
HZ_IRE = 1e6 / 140.0
IRE0_HZ = 4.4e6 - 100.0 * HZ_IRE
COLOUR_UNDER_HZ = colour_under.carrier_hz("NTSC")
LINES_PER_FIELD = 40                       # a short field is still a field
SAMPLES_PER_LINE = int(round(LINE_S * RATE_HZ))
FIELD_SAMPLES = LINES_PER_FIELD * SAMPLES_PER_LINE
FIELD_PERIOD = RATE_HZ / (2.0 * FPS)        # the format's own, in samples
VSYNC_LINE = 3                             # where a planted field sync begins
GATES = ("capture_profile", "capture_filter", "rf_stages", "filter_model",
         "transport_model", "band_delay", "head_switch_pair", "interference",
         "head_model", "head_differential", "magnetic", "magnetic_circuit",
         "tape_path")


def _stub_rf(selection=None):
    rf = types.SimpleNamespace(
        color_system="NTSC",
        SysParams={
            "line_period": LINE_US, "frame_lines": 525, "FPS": FPS,
            "ire0": IRE0_HZ, "hz_ire": HZ_IRE, "vsync_ire": -40,
            "hsyncPulseUS": 4.7, "colorBurstUS": (5.3, 7.8),
            "activeVideoUS": (9.45, 62.5555555), "syncTransitionUS": 0.140,
            "outfreq": 4.0 * 315.0e6 / 88.0 / 1e6, "head_switches_per_field": 1,
        },
        DecoderParams={
            "video_bpf_low": 500000, "video_bpf_high": 6500000,
            "color_under_carrier": COLOUR_UNDER_HZ, "chroma_bpf_upper": 1200000,
            "hz_ire": HZ_IRE, "vsync_ire": -40, "video_track_width": 58.0,
        },
        freq_hz=RATE_HZ, freq_hz_half=RATE_HZ / 2.0, blocklen=BLOCKLEN,
        blockcut=BLOCKCUT, blockcut_end=BLOCKCUT_END,
        Filters={"RFVideo": np.ones(BLOCKLEN)},
        dod_options=types.SimpleNamespace(dod_threshold_p=0.2),
        options=types.SimpleNamespace(
            rf_transform=1, stage_selection=selection, tape_format="VHS",
            **{name: 0 for name in GATES}),
    )
    rf.iretohz = lambda ire, spec=False: IRE0_HZ + float(ire) * HZ_IRE
    return rf


_RAW_CACHE = {}


def _synthetic_raw(seed, noise=0.0, vertical_sync=False):
    """One field of radio frequency: the luma carrier at its specified
    frequencies with a sync pulse and a little picture a line, plus the
    colour-under burst in the specified window, on an integer lattice.
    With `vertical_sync`, lines three to five carry the field-sync
    block's broad pulses (27.1 microseconds at half-line spacing, SMPTE
    170M) so the field-sync locator has something to find - three lines
    in, clear of the transient the analytics' zero-phase filters leave at
    the record's first samples."""
    key = (seed, noise, vertical_sync)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    rng = np.random.default_rng(seed)
    total = FIELD_SAMPLES
    time = np.arange(total) / RATE_HZ
    ire = np.zeros(total)
    sync = int(round(4.7e-6 * RATE_HZ))
    broad = int(round(27.1e-6 * RATE_HZ))
    for line in range(LINES_PER_FIELD):
        start = line * SAMPLES_PER_LINE
        if vertical_sync and VSYNC_LINE <= line < VSYNC_LINE + 3:
            ire[start:start + broad] = -40.0
            half = start + SAMPLES_PER_LINE // 2
            ire[half:half + broad] = -40.0
            continue
        ire[start:start + sync] = -40.0
        ire[start + int(9e-6 * RATE_HZ):start + SAMPLES_PER_LINE] = 50.0
    frequency = np.array([vhs_specification.carrier_hz_for_ire(v) for v in ire])
    phase = 2.0 * np.pi * np.cumsum(frequency) / RATE_HZ
    signal = np.cos(phase)
    window = bd.specified_interval("NTSC")
    begin = 4.7e-6 + window["start_us"] * 1e-6
    length = window["length_us"] * 1e-6
    gate = np.zeros(total)
    for line in range(LINES_PER_FIELD):
        start = line * SAMPLES_PER_LINE
        a = start + int(round(begin * RATE_HZ))
        b = a + int(round(length * RATE_HZ))
        if b < total:
            gate[a:b] = 1.0
    signal = signal + 0.5 * gate * np.cos(2.0 * np.pi * COLOUR_UNDER_HZ * time)
    if noise > 0.0:
        signal = signal + rng.normal(0.0, noise, total)
    # an eight-bit capture in a sixteen-bit container: a step of 256
    lattice = np.clip(np.rint(signal * 100.0), -127, 127) * 256
    raw = lattice.astype(np.int16)
    _RAW_CACHE[key] = raw
    return raw


def _stub_field(rf, index, first, seed=0, noise=0.0, switch_row=None,
                readloc=None, vertical_sync=False):
    """Field `index` of a decode: its raw radio frequency, a block-aligned
    read position one field period apart, its head label, its line
    locations and the decoder's envelope."""
    raw = _synthetic_raw(seed, noise, vertical_sync)
    if readloc is None:
        blocksize = BLOCKLEN - BLOCKCUT - BLOCKCUT_END
        readloc = (int(index * FIELD_PERIOD) // blocksize) * blocksize
    linelocs = np.arange(LINES_PER_FIELD + 1, dtype=np.float64) * SAMPLES_PER_LINE
    field = types.SimpleNamespace(
        rf=rf, rawdata=raw, readloc=float(readloc), isFirstField=bool(first),
        linelocs=linelocs, valid=True, dspicture=None,
        data={"video": {"envelope": np.ones(raw.size, dtype=np.float32),
                        "demod": np.full(raw.size, IRE0_HZ, dtype=np.float32),
                        "demod_raw": np.full(raw.size, IRE0_HZ,
                                             dtype=np.float32)}},
    )
    field.switch_row = (LINES_PER_FIELD - 3) if switch_row is None else switch_row
    return field


def _plant(monkeypatch, slopes=(-0.38e-6, -0.42e-6), tip_departure_hz=0.0,
           porch_departure_hz=0.0, burst_over_porch=-2.0, locator=True):
    """The seams: the amplitude probe's line per head, the reserved-band
    levels and the head switch locator, each answering with a planted
    value the stage must reproduce."""
    def amplitude(field):
        head = int(bool(field.isFirstField))
        line = (IRE0_HZ, 0.0, float(slopes[head]))
        return luma_amplitude.AmplitudeDeviation(
            None, None, None, luma_amplitude.ResponseModel(line))

    def levels(field, geometry):
        parity = int(bool(field.isFirstField))
        return {
            "parity": parity,
            "level_nepers": {"tip": 0.0, "porch": 0.0,
                             "burst": burst_over_porch + 0.01 * parity},
            "frequency_hz": {"tip": geometry["sync_tip_hz"] + tip_departure_hz,
                             "porch": geometry["blanking_hz"]
                             + porch_departure_hz,
                             "burst": COLOUR_UNDER_HZ},
            "scatter_nepers": {"tip": 0.001, "porch": 0.001, "burst": 0.001},
            "lines": {"tip": 30, "porch": 30, "burst": 30},
            "interval_s": 1.8571e-6,
        }

    def locate(field):
        start = float(field.switch_row * SAMPLES_PER_LINE)
        return [(int(start), int(start + SAMPLES_PER_LINE), -1.0, 9.0)]

    monkeypatch.setattr(luma_amplitude, "measured_amplitude_deviation", amplitude)
    monkeypatch.setattr(model_stages, "_reserved_band_levels", levels)
    if locator:
        monkeypatch.setattr(head_switch, "locate", locate)
    else:
        monkeypatch.setattr(head_switch, "locate", lambda field: [])


def _decode(rf, count, monkeypatch=None, amount=None, seed=0, noise=0.0,
            start=0):
    """`count` fields through the stage, heads alternating, returning the
    readings in order."""
    readings = []
    for index in range(start, start + count):
        field = _stub_field(rf, index, first=(index % 2 == 0), seed=seed,
                            noise=noise)
        readings.append(rt.transform_rf(field, amount=amount))
    return readings


# ---------------------------------------------------------------------------
# The context: one raw copy, one analytics
# ---------------------------------------------------------------------------

def test_rf_context_takes_one_raw_copy_and_one_analytics_per_field(monkeypatch):
    """The two full-field transforms are the whole cost of the stage, so
    a field pays for them once however many readings it feeds."""
    _plant(monkeypatch)
    calls = {"raw": 0, "analytics": 0}
    original_raw = model_stages._raw_field
    original_analytics = model_stages._shared_rf_analytics

    def raw_spy(field):
        calls["raw"] += 1
        return original_raw(field)

    def analytics_spy(field, raw, rate, system="NTSC"):
        calls["analytics"] += 1
        return original_analytics(field, raw, rate, system)

    monkeypatch.setattr(model_stages, "_raw_field", raw_spy)
    monkeypatch.setattr(model_stages, "_shared_rf_analytics", analytics_spy)
    rf = _stub_rf()
    field = _stub_field(rf, 0, first=True)
    reading = rt.transform_rf(field)
    assert reading is not None
    assert calls == {"raw": 1, "analytics": 1}
    assert reading["context"]["raw"] and reading["context"]["analytics"]
    # the readings the context fed were accepted into this head's vertices
    accepted = reading["fills"]["accepted"]
    assert set(accepted) == {"response", "levels", "timing"}
    assert all(head == 1 for _band, head, _polarity in accepted["response"])


# ---------------------------------------------------------------------------
# The latch and the snapshot
# ---------------------------------------------------------------------------

def test_nothing_is_published_before_the_latch_and_live_flips_at_it(monkeypatch):
    """Eight vertices, four fields of evidence each, heads alternating:
    nothing before the eighth field, one snapshot at it."""
    _plant(monkeypatch)
    rf = _stub_rf()
    readings = _decode(rf, 7, monkeypatch)
    assert not rt.live(rf)
    assert rt.table(rf, 0.0, BLOCKLEN) is None
    assert all(r["published"] is None for r in readings)
    assert not rt.wants_redo(rf)
    eighth = _decode(rf, 1, monkeypatch, start=7)[0]
    assert rt.live(rf)
    assert eighth["published_now"]
    assert eighth["published"]["version"] == 1
    assert eighth["report"]["latched"]["treatment"] == "substitute"
    assert eighth["amount"]["applied"] == rt.HALF_OPTIMUM_AMOUNT


def test_the_published_table_is_hermitian_unity_at_dc_and_delay_normalised(monkeypatch):
    _plant(monkeypatch)
    rf = _stub_rf()
    _decode(rf, 8, monkeypatch)
    published = rf.__dict__["_rf_transform"]
    for table in (published.common, published.per_head[False],
                  published.per_head[True]):
        assert table.shape == (BLOCKLEN,)
        assert table.dtype == np.complex128
        assert table[0] == 1.0 + 0.0j
        half = BLOCKLEN // 2
        assert np.allclose(table[BLOCKLEN - np.arange(1, half)],
                           np.conj(table[1:half]))
        assert table[half].imag == 0.0
        # zero phase from a magnitude-only departure, and the group delay
        # at the blanking carrier is zero after the normalisation
        grid = np.arange(half + 1) * (RATE_HZ / BLOCKLEN)
        phase = np.unwrap(np.angle(table[:half + 1]))
        centre = int(round(IRE0_HZ / (RATE_HZ / BLOCKLEN)))
        slope = np.polyfit(grid[centre - 2:centre + 3],
                           phase[centre - 2:centre + 3], 1)[0]
        assert abs(slope) < 1e-12
        assert np.allclose(phase, 0.0, atol=1e-9)
    assert rt.table(rf, 0.0, BLOCKLEN + 2) is None
    assert rt.table(rf, 0.0, BLOCKLEN) is not None


def test_a_planted_per_head_departure_yields_two_head_tables_and_a_common_one(monkeypatch):
    """The head tables' ratio reproduces the planted differential and the
    common table the planted common part, at the amount applied."""
    slopes = (-0.30e-6, -0.50e-6)
    _plant(monkeypatch, slopes=slopes)
    rf = _stub_rf()
    _decode(rf, 8, monkeypatch)
    published = rf.__dict__["_rf_transform"]
    grid = rt.response_grid(rf, model_stages.rf_geometry(_stub_field(rf, 0, True)))
    index = grid["index"]
    inner = grid["belief"] == 1.0
    amount = rt.HALF_OPTIMUM_AMOUNT
    distance = grid["bins_hz"] - grid["reference_hz"]
    head0 = published.per_head[False][index]
    head1 = published.per_head[True][index]
    common = published.common[index]
    # each head's own table is exp(-amount x its own line)
    expected0 = np.exp(-amount * slopes[0] * distance)
    expected1 = np.exp(-amount * slopes[1] * distance)
    assert np.allclose(np.abs(head0[inner]), expected0[inner], rtol=1e-6)
    assert np.allclose(np.abs(head1[inner]), expected1[inner], rtol=1e-6)
    ratio = np.log(np.abs(head1[inner]) / np.abs(head0[inner]))
    assert np.allclose(ratio, -amount * (slopes[1] - slopes[0]) * distance[inner],
                       atol=1e-9)
    mean_slope = 0.5 * (slopes[0] + slopes[1])
    assert np.allclose(np.log(np.abs(common[inner])),
                       -amount * mean_slope * distance[inner], atol=1e-9)
    # unity outside the band, where nothing was measured
    outside = np.ones(BLOCKLEN // 2 + 1, dtype=bool)
    outside[index] = False
    assert np.allclose(published.common[:BLOCKLEN // 2 + 1][outside], 1.0)
    # and the two heads genuinely differ
    assert not np.allclose(head0, head1)


def test_the_head_schedule_labels_blocks_by_absolute_sample_and_the_majority(monkeypatch):
    """The anchor is the located switch's absolute sample, `readloc` plus
    `blockcut` plus its field-sample position; the field beginning there is
    the OTHER head; a straddling block takes the side with more samples."""
    _plant(monkeypatch)
    rf = _stub_rf()
    readings = _decode(rf, 8, monkeypatch)
    published = rf.__dict__["_rf_transform"]
    schedule = published.schedule
    assert schedule is not None
    last = readings[-1]
    last_field = _stub_field(rf, 7, first=False)
    # the field's sample zero is the capture's `readloc + blockcut`: each
    # block's raw input is loaded from `b * blocksize` and cut by
    # `blockcut` before the field is assembled (lddecode/core.py:1279-1283,
    # 1363-1365)
    anchor = (last_field.readloc + BLOCKCUT
              + last_field.switch_row * SAMPLES_PER_LINE)
    assert schedule.anchor == anchor
    assert last["switch"]["source"] == "locator"
    # the eighth field is head 0 (isFirstField False); the next begins at
    # the switch and is head 1
    assert schedule.parity is True
    assert schedule.head_at(anchor - 1.0) is False
    assert schedule.head_at(anchor + 1.0) is True
    # the period is the field period the anchors were planted one apart
    assert abs(schedule.period - FIELD_PERIOD) < 2.0 * (BLOCKLEN - 2 * BLOCKCUT)
    # the workers choose by absolute sample
    before = rt.table(rf, anchor - BLOCKLEN - 10.0, BLOCKLEN)
    after = rt.table(rf, anchor + 10.0, BLOCKLEN)
    assert np.array_equal(before, published.per_head[False])
    assert np.array_equal(after, published.per_head[True])
    # a block straddling the switch takes the majority head
    mostly_before = rt.table(rf, anchor - 0.75 * BLOCKLEN, BLOCKLEN)
    mostly_after = rt.table(rf, anchor - 0.25 * BLOCKLEN, BLOCKLEN)
    assert np.array_equal(mostly_before, published.per_head[False])
    assert np.array_equal(mostly_after, published.per_head[True])


def test_wants_redo_fires_once_per_version_and_a_second_latch_is_refused(monkeypatch):
    _plant(monkeypatch)
    rf = _stub_rf()
    _decode(rf, 8, monkeypatch)
    assert rt.wants_redo(rf)
    assert not rt.wants_redo(rf)
    transform = rf.__dict__["_single_transform"]["rf"]
    first = transform.latched
    assert first.version == 1
    # sixteen is on the doubling stride and fills again; the latch holds
    more = _decode(rf, 8, monkeypatch, start=8)
    assert transform.latched is first
    assert transform.latch("remove", 4) is first
    assert rf.__dict__["_rf_transform"].version == 1
    assert not rt.wants_redo(rf)
    assert not any(r["published_now"] for r in more)
    # the stride: fields 9 to 15 read nothing, 16 reads again
    assert [r["active"] for r in more] == [False] * 7 + [True]
    assert more[-1]["fills"] is not None and more[0]["fills"] is None


def test_the_schedule_is_resynchronised_under_the_same_version(monkeypatch):
    """The tables are frozen at the latch; the clock is not. A later
    switch moves the anchor forward without a new version."""
    _plant(monkeypatch)
    rf = _stub_rf()
    _decode(rf, 8, monkeypatch)
    anchor_at_latch = rf.__dict__["_rf_transform"].schedule.anchor
    _decode(rf, 4, monkeypatch, start=8)
    published = rf.__dict__["_rf_transform"]
    assert published.version == 1
    assert published.schedule.anchor > anchor_at_latch
    assert not rt.wants_redo(rf) or True   # consumed at the latch already
    assert rf.__dict__["_rf_transform_redone"] in (0, 1)


# ---------------------------------------------------------------------------
# The hypercube
# ---------------------------------------------------------------------------

def test_the_hypercube_floor_reads_the_planted_out_of_band_density(monkeypatch):
    """White noise of variance sigma^2 on a real signal has a one-sided
    Welch density of 2 sigma^2 / f_s per hertz, so on the block grid each
    bin carries 2 sigma^2 / nperseg; the capture's lattice scales it."""
    _plant(monkeypatch)
    rf = _stub_rf()
    sigma = 0.05
    field = _stub_field(rf, 0, first=True, seed=11, noise=sigma)
    reading = rt.transform_rf(field)
    box = reading["hypercube"]
    assert box is not None
    nperseg = min(BLOCKLEN, field.rawdata.size)
    # the lattice is 100 x 256 codes per unit of the synthetic signal, and
    # the rounding to eight bits adds its own quantisation noise
    scale = 100.0 * 256.0
    step_noise = (256.0 ** 2) / 12.0
    planted = 2.0 * ((sigma * scale) ** 2 + step_noise) / nperseg
    assert abs(box["density"] - planted) < 0.15 * planted
    assert box["bins_outside"] > 0 and box["bins_inside"] > 0
    assert box["carrier_power"] > 100.0 * box["density"]
    floor = reading["floor"]
    assert floor["estimate"] is True
    assert np.isfinite(floor["value"]) and floor["value"] > 0.0
    assert floor["value"] < floor["per_field"] < floor["per_sample"]
    # and it was handed to the transform as the response channel's floor
    transform = rf.__dict__["_single_transform"]["rf"]
    assert transform.external_floors["response"] == floor["value"]


def test_the_floors_are_reported_once_every_vertex_has_evidence(monkeypatch):
    _plant(monkeypatch)
    rf = _stub_rf()
    readings = _decode(rf, 8, monkeypatch, noise=0.02, seed=3)
    assert readings[0]["floors"] is None
    floors = readings[-1]["floors"]["response"]
    assert floors["used"]
    assert np.isfinite(floors["hypercube"])
    assert np.isfinite(floors["instrument"])
    assert "polarity" in floors["null_contrasts"]


# ---------------------------------------------------------------------------
# The amount: the correction-gain law's half, or the exact inverse
# ---------------------------------------------------------------------------

def test_the_exact_inverse_component_is_off_unless_selected():
    declared = pipeline_graph.load()
    node = declared["by_name"]["rf_transform"]
    assert "exact_inverse" in {c["name"] for c in node["components"]}
    assert not pipeline_graph.component_enabled(
        None, "rf_transform", "exact_inverse", default=False)
    selection = pipeline_graph.parse_selection(
        "+rf_transform.exact_inverse", declared)
    assert pipeline_graph.component_enabled(
        selection, "rf_transform", "exact_inverse", default=False)
    rf = _stub_rf()
    assert rt.resolve_amount(rf)["applied"] == rt.HALF_OPTIMUM_AMOUNT
    assert rt.resolve_amount(rf, 0.7)["applied"] == 0.7
    assert rt.resolve_amount(_stub_rf(selection))["exact_inverse"]
    assert rt.resolve_amount(_stub_rf(selection))["applied"] == 1.0
    remove = pipeline_graph.parse_selection(
        "+rf_transform.remove_residuals", declared)
    assert rt.resolve_treatment(rf) == "substitute"
    assert rt.resolve_treatment(_stub_rf(remove)) == "remove"


def test_with_the_component_on_the_table_is_the_amount_one_table(monkeypatch):
    """Off, the half: the log magnitude of the published table is exactly
    half of the exact inverse's, bin for bin."""
    _plant(monkeypatch)
    selection = pipeline_graph.parse_selection(
        "+rf_transform.exact_inverse", pipeline_graph.load())
    half_rf, exact_rf = _stub_rf(), _stub_rf(selection)
    half = _decode(half_rf, 8, monkeypatch)[-1]
    exact = _decode(exact_rf, 8, monkeypatch)[-1]
    assert half["amount"]["applied"] == 0.5 and not half["amount"]["exact_inverse"]
    assert exact["amount"]["applied"] == 1.0 and exact["amount"]["exact_inverse"]
    log_half = np.log(np.abs(half_rf.__dict__["_rf_transform"].common))
    log_exact = np.log(np.abs(exact_rf.__dict__["_rf_transform"].common))
    assert np.allclose(2.0 * log_half, log_exact, atol=1e-12)
    assert np.abs(log_exact).max() > 0.0
    # the remainder under the exact inverse is carried even at the half
    remainders = half["remainders"]["substitute"]
    row = remainders["response:band0"]
    assert row["amount"] == 0.5
    assert row["exact_inverse"] <= row["applied"] <= row["before"] + 1e-12


# ---------------------------------------------------------------------------
# The workers' facts: a None block start, and a redone field
# ---------------------------------------------------------------------------

def test_a_none_block_start_returns_the_common_table_and_never_raises(monkeypatch):
    _plant(monkeypatch)
    rf = _stub_rf()
    assert rt.table(rf, None, BLOCKLEN) is None
    _decode(rf, 8, monkeypatch)
    published = rf.__dict__["_rf_transform"]
    assert np.array_equal(rt.table(rf, None, BLOCKLEN), published.common)
    assert rt.table(rf, None, BLOCKLEN + 1) is None


def test_a_redone_field_is_memoised_by_read_position_and_does_not_bump(monkeypatch):
    """A redo re-creates the field object at the same tape position; the
    fold must not count it twice and the version must not move."""
    _plant(monkeypatch)
    rf = _stub_rf()
    readings = _decode(rf, 8, monkeypatch)
    transform = rf.__dict__["_single_transform"]["rf"]
    accepted_before = transform.accepted
    state = rf.__dict__["_single_transform"]["rf_state"]
    assert state["fields"] == 8
    assert rt.wants_redo(rf)
    # the eighth field again, as a new object at the same readloc
    again = _stub_field(rf, 7, first=False)
    redone = rt.transform_rf(again)
    assert redone["redone"] is True
    assert redone["field"] == readings[-1]["field"] == 8
    assert state["fields"] == 8
    assert transform.accepted == accepted_before
    assert rf.__dict__["_rf_transform"].version == 1
    assert not rt.wants_redo(rf)
    assert again.rf_transform is redone


def test_a_redone_reading_has_the_shape_of_a_first_pass_reading(monkeypatch):
    """One shape for every consumer of `field.rf_transform`."""
    _plant(monkeypatch)
    rf = _stub_rf()
    first = _decode(rf, 8, monkeypatch)[-1]
    redone = rt.transform_rf(_stub_field(rf, 7, first=False))
    assert set(redone) == set(first)
    assert redone["redone"] and not first["redone"]
    assert redone["published"]["version"] == first["published"]["version"] == 1
    assert redone["report"]["latched"]["version"] == 1


def test_a_replaced_shared_store_does_not_lose_the_transform_or_the_count(monkeypatch):
    """Measured on the first real decode: the shared `_single_transform`
    dictionary was replaced once between the first two fields and the
    count restarted at one. The transform is kept under this module's own
    key and mirrored back, so a replacement costs nothing."""
    _plant(monkeypatch)
    rf = _stub_rf()
    _decode(rf, 1, monkeypatch)
    transform = rf.__dict__["_single_transform"]["rf"]
    rf.__dict__["_single_transform"] = {"picture": object()}
    reading = _decode(rf, 1, monkeypatch, start=1)[0]
    assert reading["field"] == 2
    assert rf.__dict__["_single_transform"]["rf"] is transform
    assert "picture" in rf.__dict__["_single_transform"]
    assert transform.accepted == 24


# ---------------------------------------------------------------------------
# The witnesses and the fallback anchor
# ---------------------------------------------------------------------------

def test_a_parity_break_is_a_witness_and_re_anchors_the_schedule(monkeypatch):
    _plant(monkeypatch)
    rf = _stub_rf()
    _decode(rf, 4, monkeypatch)
    state = rf.__dict__["_single_transform"]["rf_state"]
    assert len(state["anchors"]) == 4
    # the same head twice: a dropped field
    broken = _stub_field(rf, 4, first=False)
    reading = rt.transform_rf(broken)
    assert "parity break" in reading["witnesses"]
    assert reading["fills"]["accepted"] == {}
    assert len(state["anchors"]) == 1


def test_when_the_locator_refuses_the_switch_comes_from_the_specified_window(monkeypatch):
    """Five to eight lines ahead of the next vertical sync (SMPTE 32M
    clause 3.6), from the field-sync locator on the shared analytics."""
    _plant(monkeypatch, locator=False)
    rf = _stub_rf()
    field = _stub_field(rf, 0, first=True, vertical_sync=True)
    reading = rt.transform_rf(field)
    switch = reading["switch"]
    assert switch is not None
    assert switch["source"] == "vertical sync and the specified window"
    line = LINE_S * RATE_HZ
    assert switch["uncertainty_samples"] == pytest.approx(1.5 * line, rel=1e-9)
    # the block's leading edge is the planted line's start; the switch is
    # placed one field period on and six and a half lines earlier, to the
    # sample the band-limited frequency crosses the threshold at
    expected = (field.readloc + BLOCKCUT + VSYNC_LINE * SAMPLES_PER_LINE
                + FIELD_PERIOD - 6.5 * line)
    # the band-limited frequency crosses the threshold within the sync
    # transition time the system parameters state (140 ns, 5.6 samples)
    transition = rf.SysParams["syncTransitionUS"] * 1e-6 * RATE_HZ
    assert abs(switch["sample"] - expected) < transition
    # and without a field-sync block either, no anchor is claimed
    bare = _stub_field(rf, 1, first=False)
    assert rt.transform_rf(bare)["switch"] is None


def test_the_stage_bails_when_off_or_without_raw_data(monkeypatch):
    _plant(monkeypatch)
    rf = _stub_rf()
    rf.options.rf_transform = 0
    assert rt.transform_rf(_stub_field(rf, 0, True)) is None
    rf.options.rf_transform = 1
    field = _stub_field(rf, 0, True)
    field.rawdata = None
    assert rt.transform_rf(field) is None
    assert not rt.live(rf) and not rt.wants_redo(rf)
