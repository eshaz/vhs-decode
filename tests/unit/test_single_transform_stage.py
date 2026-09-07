"""The one picture stage: measured once, filled, latched, realised together.

Ethan, 2026-09-07: 'I can make a singla picture stage that transforms luma
chroma all up to the composite functions. There doesn't need to be
sequencing, just all the dimensions execute at once in a single transform.'
And on the amount: 'Let's see what happens if we take this to hyperspace as
the null space model, the exact inverse within our possible area of measure.'

These tests hold the stage's contract on a field whose answer is known by
construction - the planted sync pulse of `test_sync_timing_stages` with a
planted gain and a planted zero - rather than on a decode: nothing is
applied before the latch and everything once after it; the closed-form
level map reproduces the legacy sequence; the luma alone is a transform of
its own; every absorbed node is measured once a field; a node that is off
leaves its channel unfilled and says so; the treatment and the amount answer
to their components; and a field folds in once however many times it is
handed over.
"""

import logging
import types

import numpy as np
import pytest

import lddecode.core as ldd

from vhsdecode import model_stages, pipeline_graph
from tests.unit.test_sync_timing_stages import (
    _stub_field, HEIGHT, SUBCARRIER_HZ, UNITS_PER_IRE, WIDTH)


DECISION = model_stages._LOCK_DECISION_FIELDS
# One decode reaches the latch when every vertex of every used channel holds
# the decision count; the heads alternate, so that is two fields per fill.
FIELDS_TO_LATCH = 2 * DECISION

# SysParams_NTSC in lddecode/core.py: six equalizing pulses a section, the
# subcarrier, the container's sync tip, and a line of 227.5 subcarrier
# cycles. The stub in `test_sync_timing_stages` carries only what the sync
# group reads; `geometry` and the level series read these as well.
NTSC = {
    "numPulses": 6,
    "fsc_mhz": SUBCARRIER_HZ / 1e6,
    "outputZero": 1024,
    "line_period": 1e6 / (SUBCARRIER_HZ / 227.5),
}
# vhsdecode/format_defs/vhs.py, NTSC VHS at standard play: the burst peak the
# chroma automatic gain drives to.
BURST_ABS_REF = 5730
# vhsdecode/format_defs/vhs.py: the NTSC colour-under carrier, 40 f_H.
COLOUR_UNDER_HZ = 40.0 * (SUBCARRIER_HZ / 227.5)


@pytest.fixture(autouse=True)
def _logger():
    """`ldd.logger` is None until a decode initialises it, and the stage
    reports what it found. A test that constructs no decoder has to supply
    one or the reporting is what fails."""
    previous = getattr(ldd, "logger", None)
    if previous is None:
        ldd.logger = logging.getLogger("test_single_transform_stage")
    yield
    ldd.logger = previous


def _options(**overrides):
    """Every absorbed node off, the transform on, nothing selected."""
    base = {name: 0 for name in model_stages._PICTURE_NODES}
    base.update(tape_format="VHS", export_raw_tbc=False, ire0_adjust=(),
                picture_transform=1, stage_selection=None, write_chroma=False)
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _Decode:
    """A run of planted fields sharing ONE decoder object, which is where
    the transform accumulates: a new decode is a new `rf`, and a field of
    this decode is a new object every time, as the decoder's own are.

    Every field carries the specified pulse with a planted gain (the
    spacing is `40 gain` IRE) and a planted zero (every sample sits
    `zero_ire` above the container's blanking), so the map the transform
    has to find is known: a gain of `gain` about blanking and a zero of
    `zero_ire`.
    """

    def __init__(self, gain=0.95, zero_ire=2.0, seed=1, **options):
        self.gain = float(gain)
        self.zero = float(zero_ire)
        self.seed = int(seed)
        self.options = _options(**options)
        self.rf = None
        self.count = 0

    def field(self, first=None, readloc=None, seed=None):
        field, picture = _stub_field(
            spacing_ire=40.0 * self.gain,
            seed=self.seed + self.count if seed is None else int(seed))
        shifted = picture.astype(np.float64) + self.zero * UNITS_PER_IRE
        field.dspicture = (np.clip(shifted, 0.0, 65535.0) + 0.5
                           ).astype(np.uint16)
        if self.rf is None:
            rf = field.rf
            rf.SysParams.update(NTSC)
            rf.SysParams["burst_abs_ref"] = BURST_ABS_REF
            rf.DecoderParams.update({"color_under_carrier": COLOUR_UNDER_HZ,
                                     "vsync_ire": -40.0})
            rf.options = self.options
            self.rf = rf
        field.rf = self.rf
        field.out_scale = UNITS_PER_IRE
        field.burst_detected_line = 0
        field.sync_confidence = 100
        field.isFirstField = (self.count % 2 == 0) if first is None else bool(first)
        field.readloc = self.count * 1_000_000 if readloc is None else readloc
        self.count += 1
        return field

    def transform(self):
        return self.rf.__dict__["_single_transform"]["picture"]


def _latched(decode, uphet=None, amount=None):
    """Run the decode up to its latch and hand it back latched."""
    for _ in range(FIELDS_TO_LATCH):
        model_stages.transform_picture(decode.field(), uphet, amount)
    assert decode.transform().latched is not None
    return decode


def _selection(text):
    return pipeline_graph.parse_selection(text, pipeline_graph.load())


# --------------------------------------------------------------------------
# Nothing before the latch, everything once after it
# --------------------------------------------------------------------------

def test_nothing_is_applied_before_the_latch_and_once_after():
    """Rule 4 of the ringing rules, at the stage's own boundary: the fields
    ahead of the latch are delivered as they arrived, the field that latches
    is corrected, and the reading carries the remainder under BOTH
    treatments so the decode can compare them."""
    decode = _Decode(sync_shape=1, precursor=1, sync_depth=1,
                     standard_levels=1, source_correction=1)
    for _ in range(FIELDS_TO_LATCH - 1):
        field = decode.field()
        before = field.dspicture.copy()
        reading = model_stages.transform_picture(field, None)
        assert reading is not None
        assert reading["applied"] is None
        assert reading["latched"] is False
        assert np.array_equal(field.dspicture, before)
        assert field.picture_transform is reading

    field = decode.field()
    before = field.dspicture.copy()
    reading = model_stages.transform_picture(field, None)
    assert reading["latched"] is True
    assert reading["applied"] is not None
    assert not np.array_equal(field.dspicture, before)
    assert reading["treatment"]["applied"] == "substitute"
    assert reading["treatment"]["computed"] == ["substitute", "remove"]
    for treatment in ("substitute", "remove"):
        remainder = reading["remainder"][treatment]
        assert remainder is not None
        assert "levels" in remainder and "response" in remainder
        for entry in remainder["levels"].values():
            assert entry["before"] >= 0.0
            assert entry["after_exact"] >= 0.0
            assert entry["after_applied"] >= 0.0
    # the latch is immutable: a later field neither re-latches nor re-fits
    version = decode.transform().latched.version
    model_stages.transform_picture(decode.field(), None)
    assert decode.transform().latched.version == version


def test_the_per_field_caches_are_dropped_and_the_reading_attached():
    decode = _Decode(sync_depth=1, standard_levels=1)
    field = decode.field()
    reading = model_stages.transform_picture(field, None)
    assert "_sync_pulses" not in field.__dict__
    assert "_level_series" not in field.__dict__
    assert reading["channels"]["levels"]["used"] is True
    assert reading["measured"]["sync_depth"]["spacing_ire"] == pytest.approx(
        38.0, abs=0.1)
    assert reading["measured"]["luma_levels"]["blanking_ire"] == pytest.approx(
        2.0, abs=0.05)


# --------------------------------------------------------------------------
# The closed-form level map against the legacy sequence
# --------------------------------------------------------------------------

def test_the_closed_form_level_map_matches_the_legacy_sequence():
    """THE 'EVERYTHING STILL MATCHES' GATE. The transform's one write, from
    the latched levels at amount one, against the legacy sequence
    `correct_precursor` -> `correct_sync_depth` -> `correct_standard_levels`
    on the same planted field.

    The two differ by the rounding the sequence performs between its writes
    and the closed form does not: each intermediate write moves a sample by
    at most half a least significant bit, the gain carries that through at
    one over 0.95, and the final rounding of each path can then land on
    neighbouring integers. So the bound is one bit on almost every sample
    and two bits on the rare sample whose accumulated perturbation crosses
    a rounding boundary twice; the root mean square of the difference is
    what the rounding budget predicts and nothing more.
    """
    decode = _latched(_Decode(precursor=1, sync_depth=1, standard_levels=1),
                      amount=1.0)
    field = decode.field(seed=99)
    legacy = decode.field(first=field.isFirstField, seed=99)
    assert np.array_equal(field.dspicture, legacy.dspicture)

    reading = model_stages.transform_picture(field, None, amount=1.0)
    assert reading["applied"] is not None
    assert reading["amount"] == {"value": 1.0, "reason": "caller"}
    levels = reading["applied"]["levels"]
    assert levels["gain_applied"] and levels["zero_applied"]
    assert levels["gain"] == pytest.approx(0.95, abs=2e-3)
    assert levels["zero_ire"] == pytest.approx(2.0, abs=0.05)

    model_stages.correct_precursor(legacy, legacy.dspicture)
    assert model_stages.correct_sync_depth(legacy, legacy.dspicture) is not None
    assert model_stages.correct_standard_levels(legacy, 1.0) is not None

    difference = (field.dspicture.astype(np.float64)
                  - legacy.dspicture.astype(np.float64))
    assert np.sqrt(np.mean(difference ** 2)) < 1.0
    assert np.percentile(np.abs(difference), 99.9) <= 1.0
    assert np.abs(difference).max() <= 2.0
    # and the map did what the plant asked: blanking at the container's
    # zero, the tip forty beneath it, on the corrected field itself
    spans = model_stages.sync_spans(field)
    lines = field.dspicture.astype(np.float64).reshape(HEIGHT, WIDTH)
    rows = spans["rows"]
    porch = lines[rows][:, spans["porch"][0]:spans["porch"][1]].mean()
    tip = lines[rows][:, spans["tip"][0]:spans["tip"][1]].mean()
    assert (porch - spans["blanking"]) / UNITS_PER_IRE == pytest.approx(
        0.0, abs=0.05)
    assert (porch - tip) / UNITS_PER_IRE == pytest.approx(40.0, abs=0.1)


# --------------------------------------------------------------------------
# The luma alone, the measurements once, and a node that is off
# --------------------------------------------------------------------------

def test_the_luma_alone_is_a_transform_without_the_colour_axis():
    """A decode that writes no chroma has the colour axis as a known
    constant, exactly as every decode has the field axis and the tap."""
    decode = _Decode(sync_depth=1, standard_levels=1)
    reading = model_stages.transform_picture(decode.field(), None)
    assert reading["axes"] == ("head", "polarity")
    assert "colour" in reading["unreached"]
    assert "field" in reading["unreached"]
    assert decode.transform().kept == {(), ("head",), ("polarity",)}


def test_each_absorbed_node_is_measured_once_per_field(monkeypatch):
    """The three shared measurements are COMPUTED once a field however many
    readers ask: the spies count only the calls that found the field's
    cache cold."""
    decode = _Decode(sync_shape=1, precursor=1, sync_depth=1,
                     standard_levels=1, source_correction=1,
                     chroma_head_switch=1, colour_free_luma=1)
    field = decode.field()
    rng = np.random.default_rng(7)
    uphet = rng.normal(0.0, 100.0, HEIGHT * WIDTH)

    cold = {"sync_pulses": 0, "_level_series": 0, "measure": 0}
    real = {name: getattr(model_stages, name) for name in cold}

    def spy(name, cache):
        def wrapper(field, *args, **kwargs):
            if field.__dict__.get(cache) is None:
                cold[name] += 1
            return real[name](field, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(model_stages, "sync_pulses",
                        spy("sync_pulses", "_sync_pulses"))
    monkeypatch.setattr(model_stages, "_level_series",
                        spy("_level_series", "_level_series"))
    monkeypatch.setattr(model_stages, "measure", spy("measure", "_colour_lock"))
    reading = model_stages.transform_picture(field, uphet)
    assert reading is not None
    assert cold == {"sync_pulses": 1, "_level_series": 1, "measure": 1}
    assert reading["axes"] == ("colour", "head", "polarity")


def test_a_node_that_is_off_leaves_its_channel_unfilled_and_says_so():
    """`--stages -sync_shape`: the response channel is never filled, the
    reading names it unused, and the transform still runs on the levels."""
    decode = _Decode(sync_depth=1, standard_levels=1)
    reading = model_stages.transform_picture(decode.field(), None)
    assert reading["on"]["sync_shape"] is False
    assert reading["channels"]["response"]["used"] is False
    assert "response" not in reading["fills"]
    assert reading["channels"]["levels"]["used"] is True
    assert "levels" in reading["fills"]
    assert "luma_response" not in reading["measured"]


# --------------------------------------------------------------------------
# The components: the treatment, the amount, and the fold-once key
# --------------------------------------------------------------------------

def test_remove_residuals_selects_the_treatment():
    """Off, the substitute treatment stands; on, every residual is removed
    by the amount it reproduces. Both are computed either way."""
    plain = _Decode(sync_depth=1)
    reading = model_stages.transform_picture(plain.field(), None)
    assert reading["treatment"]["applied"] == "substitute"
    assert reading["components"]["remove_residuals"] is False

    chosen = _Decode(sync_depth=1, stage_selection=_selection(
        "+picture_transform.remove_residuals"))
    reading = model_stages.transform_picture(chosen.field(), None)
    assert reading["treatment"]["applied"] == "remove"
    assert reading["components"]["remove_residuals"] is True
    assert reading["treatment"]["computed"] == ["substitute", "remove"]

    latched = _latched(_Decode(sync_depth=1, standard_levels=1,
                               stage_selection=_selection(
                                   "+picture_transform.remove_residuals")))
    assert latched.transform().latched.treatment == "remove"


def test_the_exact_inverse_component_sets_the_amount():
    """Ethan: 'the exact inverse within our possible area of measure.' On,
    the applied kernel and level map are at amount one; off, the
    correction-gain law's half; and the default is off unless selected."""
    declared = pipeline_graph.load()
    node = declared["by_name"]["picture_transform"]
    assert "exact_inverse" in {c["name"] for c in node["components"]}
    assert pipeline_graph.component_enabled(None, "picture_transform",
                                            "exact_inverse", False) is False

    exact = _latched(_Decode(sync_shape=1, sync_depth=1, standard_levels=1,
                             source_correction=1, stage_selection=_selection(
                                 "+picture_transform.exact_inverse")))
    half = _latched(_Decode(sync_shape=1, sync_depth=1, standard_levels=1,
                            source_correction=1))
    whole = model_stages.transform_picture(exact.field(), None)
    part = model_stages.transform_picture(half.field(), None)

    assert whole["components"]["exact_inverse"] is True
    assert whole["amount"] == {"value": 1.0, "reason": "exact_inverse component"}
    assert part["components"]["exact_inverse"] is False
    assert part["amount"] == {"value": 0.5, "reason": "correction-gain law"}
    assert whole["applied"]["response"]["fraction"] == pytest.approx(1.0)
    assert part["applied"]["response"]["fraction"] == pytest.approx(0.5)
    assert whole["applied"]["levels"]["amount"] == 1.0
    assert part["applied"]["levels"]["amount"] == 0.5
    # the same latched departure, scaled: the half's realised gain is the
    # square root of the exact one's in the log domain
    exact_map = model_stages._level_map(
        exact.transform(), model_stages.sync_spans(exact.field()), 0,
        "substitute", 1.0, True, True)
    half_map = model_stages._level_map(
        half.transform(), model_stages.sync_spans(half.field()), 0,
        "substitute", 0.5, True, True)
    assert np.allclose(np.log(half_map["gain_per_row"]),
                       0.5 * np.log(exact_map["gain_per_row"]), atol=1e-3)
    # and a caller's amount wins over both
    assert model_stages.transform_picture(half.field(), None, 0.25)["amount"] \
        == {"value": 0.25, "reason": "caller"}


def test_a_field_folds_in_once_however_often_it_is_handed_over():
    """A redone field is a NEW object with the same `readloc`; keying the
    fold on the object would count it twice."""
    decode = _Decode(sync_depth=1, standard_levels=1)
    first = decode.field(readloc=42)
    reading = model_stages.transform_picture(first, None)
    assert reading["folded"] is True
    accepted = decode.transform().accepted
    assert accepted > 0
    again = decode.field(first=first.isFirstField, readloc=42)
    reading = model_stages.transform_picture(again, None)
    assert reading["folded"] is False
    assert decode.transform().accepted == accepted
    # a genuinely new field of the other parity folds in
    reading = model_stages.transform_picture(
        decode.field(first=not first.isFirstField, readloc=43), None)
    assert reading["folded"] is True
    assert decode.transform().accepted > accepted


def test_the_witnesses_refuse_a_field_the_decoder_does_not_believe():
    """The decoder's own gate on sync confidence, and a broken parity
    alternation, mark a field transient: it is refused by witness and does
    not enter the vertices."""
    decode = _Decode(sync_depth=1, standard_levels=1)
    model_stages.transform_picture(decode.field(), None)
    accepted = decode.transform().accepted

    doubtful = decode.field()
    doubtful.sync_confidence = model_stages._SYNC_CONFIDENCE_FLOOR
    reading = model_stages.transform_picture(doubtful, None)
    assert reading["witness"]["sync_confidence"] is True
    assert reading["transient"] is True
    assert decode.transform().accepted == accepted
    assert decode.transform().rejected_by_witness > 0

    # two fields of one parity in a row: a skip or a repeat, not evidence
    same = decode.field(first=doubtful.isFirstField)
    reading = model_stages.transform_picture(same, None)
    assert reading["witness"]["parity"] is True
    assert reading["transient"] is True


def test_the_stage_is_off_when_its_option_is_off_and_needs_a_picture():
    decode = _Decode(sync_depth=1, picture_transform=0)
    assert model_stages.transform_picture(decode.field(), None) is None
    decode = _Decode(sync_depth=1)
    field = decode.field()
    field.dspicture = None
    assert model_stages.transform_picture(field, None) is None
    field = decode.field()
    field.dspicture = field.dspicture[:-1]
    assert model_stages.transform_picture(field, None) is None


def test_the_radio_frequency_shims_answer_from_the_snapshot_without_the_module():
    """Until `vhsdecode.rf_transform` exists the shims read the published
    snapshot, and report nothing where none is published."""
    rf = types.SimpleNamespace()
    assert model_stages.rf_transform_live(rf) is False
    assert model_stages.rf_transform_table(rf, 0, 4096) is None
    assert model_stages.transform_wants_redo(rf) is False


# --------------------------------------------------------------------------
# With chroma: the chroma faces, the image, and what the declaration does
# --------------------------------------------------------------------------

def _synthetic_chroma(field, seed, beta=0.05 * np.exp(0.7j), noise=20.0):
    """A burst on every line of the up-converted chroma at the decoder's own
    reference amplitude, turning the composite's half turn a line, and the
    colour-under it came from with a PLANTED quadrature image `beta`,
    turning the tape's quarter turn a line (SMPTE 32M clause 3.9.2.1.5)."""
    from tests.unit.test_sync_timing_stages import RATE_HZ

    rng = np.random.default_rng(seed)
    burst_us = field.rf.SysParams["colorBurstUS"]
    start = int(np.ceil(field.usectooutpx(burst_us[0])))
    stop = int(np.floor(field.usectooutpx(burst_us[1])))
    index = np.arange(WIDTH, dtype=np.float64)
    gate = np.zeros(WIDTH)
    gate[start:stop] = 1.0
    composite = np.zeros((HEIGHT, WIDTH))
    under = np.zeros((HEIGHT, WIDTH))
    for row in range(HEIGHT):
        composite[row] = BURST_ABS_REF * np.cos(
            2.0 * np.pi * SUBCARRIER_HZ * index / RATE_HZ + np.pi * row) * gate
        burst = np.exp(0.5j * np.pi * row)
        observed = burst + beta * np.conj(burst)
        under[row] = BURST_ABS_REF * np.real(
            observed * np.exp(2j * np.pi * COLOUR_UNDER_HZ * index / RATE_HZ)
        ) * gate
    composite += rng.normal(0.0, noise, composite.shape)
    under += rng.normal(0.0, noise, under.shape)
    field.chroma_under_tbc = under.reshape(-1)
    return composite.reshape(-1)


def test_with_chroma_every_face_fills_and_the_image_is_removed_by_its_agreement():
    """The three-axis transform: the chroma's response, level and image
    faces fill from the burst, the latch fires, the down-converted chroma
    is released, and the planted image is taken off the up-converted chroma
    at the gain law's half times the amount it reproduces across the banks -
    with the module's own before-and-after readings carried, including its
    verdict that the locked lines do not let it separate the image."""
    decode = _Decode(sync_shape=1, precursor=1, sync_depth=1,
                     standard_levels=1, source_correction=1,
                     burst_instrument=1, iq_imbalance=1)
    for k in range(FIELDS_TO_LATCH):
        field = decode.field()
        uphet = _synthetic_chroma(field, 100 + k)
        reading = model_stages.transform_picture(field, uphet)
        assert field.chroma_under_tbc is None
    assert reading["axes"] == ("colour", "head", "polarity")
    assert reading["latched"] is True
    for channel in ("response", "levels", "image"):
        assert reading["channels"][channel]["used"] is True
        accepted = [e["accepted"] for e in reading["fills"][channel].values()]
        assert len(accepted) == 4 and all(accepted)
    assert reading["channels"]["timing"]["used"] is False

    # THE INSTRUMENT'S OWN BOUND. On a gated colour-under burst - 1.6 cycles
    # of the 629 kHz carrier, which is not narrowband - the envelope's
    # per-line phasors match the plant to 1.2 per cent rms (measured on
    # this synthetic field, noise-free), and a five per cent image is read
    # to that fraction of the burst: 0.012 of it here. What the transform
    # applies is what it measured, and that is what is asserted exactly.
    planted = 0.05 * np.exp(0.7j)
    measured = reading["measured"]["image"]["beta"]
    assert abs(measured - planted) < 0.015
    applied = reading["applied"]["iq_imbalance"]
    assert applied["applied"] is True
    latched = decode.transform().departure(
        {"colour": 1, "head": int(bool(field.isFirstField)), "polarity": 0})
    assert applied["beta"] == complex(
        latched[decode.transform().channels["image"]][0])
    assert applied["agreement"] > 0.99
    assert applied["amount"] == pytest.approx(0.5 * applied["agreement"])
    assert applied["lines"] == HEIGHT
    assert applied["residual_image_rejection_db"] == pytest.approx(
        applied["image_rejection_db"] + 20.0 * np.log10(1.0 - applied["amount"]))
    assert applied["before"]["usable"] is False
    assert applied["after"]["usable"] is False

    chroma = reading["measured"]["chroma_response"]
    assert chroma["usable_bins"] >= 2
    assert abs(chroma["gain_fsc"]) == pytest.approx(BURST_ABS_REF, rel=0.05)


def test_the_declaration_realises_a_luma_only_polarity_quantity_in_full():
    """THE FINDING, AND ITS REMEDY. The chroma has no polarity state and is
    filled alike on both faces, so half of the luma's fall-rise difference
    is a colour-by-polarity interaction. With that contrast declared null
    the luma tip's departure was realised at HALF under SUBSTITUTE (a
    quarter of the difference left on each face, measured); the polarity
    mechanism is luma-only, so `("colour", "polarity")` now belongs to the
    kept set and both treatments realise the departure whole. Both
    remainders are in every reading."""
    decode = _Decode(sync_depth=1, standard_levels=1, burst_instrument=1)
    for k in range(FIELDS_TO_LATCH + 1):
        field = decode.field()
        reading = model_stages.transform_picture(
            field, _synthetic_chroma(field, 200 + k))
    head = int(bool(field.isFirstField))
    tip = "colour0:head%d:polarity0" % head
    blank = "colour0:head%d:polarity1" % head
    remove = reading["remainder"]["remove"]["levels"]
    substitute = reading["remainder"]["substitute"]["levels"]
    # the plant: a tip 4 IRE above the specified -40 (2 of zero, 2 of
    # deficit) and blanking 2 IRE above the specified zero
    assert remove[tip]["before"] == pytest.approx(4.0, abs=0.05)
    assert remove[blank]["before"] == pytest.approx(2.0, abs=0.05)
    assert remove[tip]["after_exact"] < 0.01
    assert remove[blank]["after_exact"] < 0.01
    # substitute now realises the whole departure too: the interaction
    # that carried the other half of the difference is a kept contrast
    assert substitute[tip]["after_exact"] < 0.01
    assert substitute[blank]["after_exact"] < 0.01
    # and the chroma's own level, a gain with no polarity state, reads the
    # same on both of its faces
    chroma = ["colour1:head%d:polarity%d" % (head, p) for p in (0, 1)]
    assert remove[chroma[0]]["before"] == pytest.approx(
        remove[chroma[1]]["before"])


def test_the_image_pair_is_exact_on_phasors_and_aligned_across_runs():
    """`_image_pair` forms the complex coefficient `iq_imbalance.correct`
    needs, in the frame of the field's first picture row: exact on an ideal
    series on either track, and with every run referred to that row - a run
    beginning an odd number of lines later reads the alternation with the
    opposite sign. An odd run length leaves the alternation uncancelled by
    one part in the run, which is the estimator's own bias and is bounded
    here rather than hidden."""
    planted = 0.05 * np.exp(0.7j)
    for count, bound in ((244, 1e-3), (200, 1e-3), (243, 0.01)):
        for turn in (0.5j * np.pi, -0.5j * np.pi):
            index = np.arange(count)
            burst = np.exp(turn * index)
            pair = model_stages._image_pair({
                "phasors": burst + planted * np.conj(burst),
                "rows": (10, 10 + count)})
            assert abs(pair["beta"] - planted) < bound
            assert pair["first_row"] == 10
    # a pure gain imbalance of a real coefficient is read as real
    index = np.arange(244)
    burst = np.exp(0.5j * np.pi * index)
    pair = model_stages._image_pair({"phasors": burst + 0.02 * np.conj(burst),
                                     "rows": (0, 244)})
    assert pair["beta"].real == pytest.approx(0.02, abs=1e-3)
    assert abs(pair["beta"].imag) < 1e-3


def test_the_picture_reset_leaves_the_radio_frequency_s_entry_alone():
    """Both transforms keep their state in one dictionary on the decoder.
    The picture side owns `picture`, `picture_folded` and `picture_state`
    and resets only those; a populated `rf` entry survives the first
    construction and a reconstruction on a changed key. A first form
    cleared the whole dictionary and, on a real decode, wiped the
    radio-frequency cube after the first field."""
    decode = _Decode(sync_depth=1, standard_levels=1)
    field = decode.field()
    sentinel = object()
    field.rf.__dict__["_single_transform"] = {"rf": sentinel, "rf_note": 1}
    model_stages.transform_picture(field, None)
    shared = field.rf.__dict__["_single_transform"]
    assert shared["rf"] is sentinel and shared["rf_note"] == 1
    assert shared["picture"] is decode.transform()
    assert shared["picture_folded"] == {field.readloc}
    first = shared["picture"]
    # a changed key - chroma arrives - reconstructs the picture's transform
    # and resets its own keys, and nothing else
    field = decode.field()
    model_stages.transform_picture(field, _synthetic_chroma(field, 3))
    assert shared["rf"] is sentinel and shared["rf_note"] == 1
    assert shared["picture"] is not first
    assert shared["picture"].axes == ("colour", "head", "polarity")
    assert shared["picture_folded"] == {field.readloc}
    assert set(shared) == {"rf", "rf_note", "picture", "picture_folded",
                           "picture_state"}
