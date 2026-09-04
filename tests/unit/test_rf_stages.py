"""The two RF stages: the machine that wrote the tape and the one that read it.

The component's own question is whether the two are separable at all, and
these tests are how it is answered rather than asserted. Four of them are the
measured null spaces, one is the effective count against the magnetics
baseline, one is the speed lever's confound, and one is THE CONTROL together
with the three realistic construction errors it catches.
"""

import numpy as np
import pytest

from vhsdecode.models import head_model, interference, rf_stages


RF = np.linspace(0.5e6, 8.0e6, 512)
BASEBAND = np.linspace(50e3, 3.0e6, 512)


# --------------------------------------------------------------------------
# The hard contracts
# --------------------------------------------------------------------------


def test_every_signature_is_subtractable():
    """Ethan: "the shape of the keys MUST be subtractable." The transform
    works in the log domain, so an entry whose magnitude reaches zero cannot
    be taken off by subtraction - and entered anyway, the clamp turns it into
    a large constant and the fit sees that instead of the shape."""
    entries = rf_stages.stage_signatures(BASEBAND, RF)
    assert len(entries) >= 10
    for name, value in entries.items():
        assert interference.subtractable(value), name
        assert np.iscomplexobj(np.asarray(value)), name


def test_every_signature_has_a_declared_chain_position():
    """A chain can only be inverted in the reverse of the order it was
    applied, so a key entry with no position cannot be inverted at all.
    `ordered_key` refuses one, and the merged map must leave none."""
    entries = rf_stages.stage_signatures(BASEBAND, RF)
    merged = dict(interference.COMPONENT_ORDER)
    merged.update(rf_stages.RF_STAGE_ORDER)
    undeclared = [name for name in entries
                  if name not in merged
                  and not any(name.startswith(key) for key in merged)]
    assert undeclared == []


def test_the_record_stage_is_undone_after_the_playback_stage():
    """THE WHOLE POINT OF THE COMPONENT. A record-side entry and a
    playback-side one are at DIFFERENT positions, and a correction undoes
    the chain last-applied-first - so every playback entry comes off before
    every record entry, with the tape in between."""
    entries = rf_stages.stage_signatures(BASEBAND, RF)
    order = rf_stages.RF_STAGE_ORDER
    record = [p for name, p in order.items() if name.startswith("record")]
    playback = [p for name, p in order.items() if name.startswith("playback")]
    assert max(record) < min(playback)
    # the tape's own positions sit between them, which is what makes the
    # split a chain rather than a label
    tape = [interference.COMPONENT_ORDER[k]
            for k in ("head contact tilt", "dropout", "modulation noise",
                      "particle noise")]
    assert max(record) < min(tape) <= max(tape) < min(playback)

    saved = dict(interference.COMPONENT_ORDER)
    try:
        interference.COMPONENT_ORDER.update(rf_stages.RF_STAGE_ORDER)
        sequence = interference.ordered_key(entries)
    finally:
        interference.COMPONENT_ORDER.clear()
        interference.COMPONENT_ORDER.update(saved)
    positions = [position for position, _ in sequence]
    assert positions == sorted(positions, reverse=True)
    assert sequence[0][1].startswith("playback")
    assert sequence[-1][1] in ("sub emphasis level dependence",
                               "record level dependence")


def test_the_two_stages_never_share_an_abscissa_by_accident():
    """An inner product across two abscissae is memory layout and not
    coherence, and this component is where that trap bites: the emphases act
    on the modulating video and the amplifiers and heads act on the RF."""
    entries = rf_stages.stage_signatures(BASEBAND, RF)
    for name in entries:
        assert name in rf_stages.ABSCISSA, name
    assert rf_stages.ABSCISSA["record main pre-emphasis"] == "baseband"
    assert rf_stages.ABSCISSA["playback head losses"] == "rf"


# --------------------------------------------------------------------------
# THE CONTROL, and proof that it can fail
# --------------------------------------------------------------------------


def test_the_control_passes_on_the_module_s_own_construction():
    """One mechanism entered once on each side is physically one shape, so
    the pair must span exactly ONE direction; and the format specifies the
    de-emphasis as the exact inverse of the pre-emphasis, so their product is
    unity to machine precision."""
    control = rf_stages.cascade_control(RF, BASEBAND)
    assert control["effective"] == pytest.approx(1.0, abs=0.01)
    assert control["round_trip_nepers"] < 1e-9
    assert control["passes"]


def test_the_control_catches_a_wavelength_from_the_wrong_speed():
    """The first of three realistic breaks. `head_model.writing_speed` exists
    because deriving the writing speed the wrong way is 1.3% out and carries
    into every wavelength; taking it from the LINEAR tape speed instead is
    the same mistake made much larger. One mechanism then reads as two."""
    def wrong_speed(rf_hz, gap_m, writing_speed_m_s=5.80):
        mechanics = rf_stages.mechanics_at_speed("SP")
        return rf_stages._reproduce_gap_loss(
            rf_hz, gap_m, mechanics["linear_tape_speed_m_s"])

    control = rf_stages.cascade_control(RF, BASEBAND, playback=wrong_speed)
    assert control["effective"] > 1.9          # measured 1.9996
    assert not control["one_mechanism_is_one_direction"]
    assert not control["passes"]


def test_the_control_catches_a_logarithm_where_a_response_belongs():
    """The second break, and it is present in the tree rather than invented:
    `interference.sub_emphasis` returns nepers where every other signature in
    that module returns a response, so a caller that mixes the two takes the
    logarithm of a logarithm. The emphasis round trip catches it."""
    def as_a_logarithm(baseband_hz, *args, **kwargs):
        return np.log(rf_stages.main_pre_emphasis(baseband_hz))

    control = rf_stages.cascade_control(RF, BASEBAND,
                                        post_emphasis=as_a_logarithm)
    assert control["round_trip_nepers"] > 1.0   # measured 2.06 nepers
    assert not control["emphasis_inverts_exactly"]
    assert not control["passes"]


def test_the_control_catches_decibels_where_nepers_belong():
    """The third break. Every loss in this arc is an exponential and works in
    nepers; the standards print emphasis in decibels. A stage returned in the
    wrong unit is a factor of 8.686 in the exponent and the round trip stops
    being unity."""
    def in_decibels(baseband_hz, *args, **kwargs):
        return (1.0 / rf_stages.main_pre_emphasis(baseband_hz)
                ) ** (1.0 / rf_stages.DB_PER_NEPER)

    control = rf_stages.cascade_control(RF, BASEBAND, post_emphasis=in_decibels)
    assert control["round_trip_nepers"] > 1.0   # measured 1.41 nepers
    assert not control["passes"]


# --------------------------------------------------------------------------
# The null spaces: what can never be attributed to one machine
# --------------------------------------------------------------------------


def test_the_emphasis_pair_is_exactly_one_direction():
    """The format specifies de-emphasis as the inverse of pre-emphasis, so
    their log magnitudes are exact negatives. A shape and its negation are
    ONE direction, and a departure in the emphasis therefore fits the
    recording machine and the playback machine equally well - the emphasis's
    analogue of the carrier law annihilating gain and spacing."""
    null = rf_stages.null_space(RF, BASEBAND)
    assert null["emphasis_pair_residual_nepers"] < 1e-12
    assert null["emphasis_pair_coherence"] == pytest.approx(1.0, abs=1e-9)
    baseband = rf_stages.baseband_distinguishable(BASEBAND)
    assert baseband["emphasis_pair_coherence"] == pytest.approx(1.0, abs=1e-6)
    # so the four-entry baseband family is rank deficient by exactly one
    assert np.linalg.matrix_rank(
        np.array([np.asarray(r) for r in baseband["coherence"]])) < 4


def test_the_wavelength_losses_compose_in_the_exponent():
    """The record head's transition-length loss and the playback head's
    spacing loss are both `exp(-2 pi length / lambda)`, so their product
    carries only the SUM of the two lengths. The difference is annihilated
    exactly, which is why no fit of a fused response can say how much of the
    clearance belonged to the writing and how much to the reading."""
    null = rf_stages.null_space(RF, BASEBAND)
    assert null["wavelength_loss_coherence"] == pytest.approx(1.0, abs=1e-9)
    # stated directly: two lengths, one product
    writing = rf_stages.DEFAULT_WRITING_SPEED_M_S
    both = (rf_stages.record_head_write(RF, 0.10e-6, writing)
            * rf_stages.record_head_write(RF, 0.05e-6, writing))
    one = rf_stages.record_head_write(RF, 0.15e-6, writing)
    assert np.allclose(np.abs(both), np.abs(one))
    # and demonstrated as a null space rather than asserted: two different
    # splits of the same total compose to the same response exactly
    assert null["wavelength_split_residual_nepers"] < 1e-12


def test_a_gain_and_a_delay_are_only_ever_seen_as_their_sum():
    """The two remaining nulls, demonstrated the same way. A null space is
    the statement that the split cannot be recovered from the composition,
    so the check is that two different splits of one total are identical -
    and both are, to machine precision. On this dataset a gain is doubly
    lost, because the captures were AC coupled and the scope's vertical scale
    was set per capture."""
    null = rf_stages.null_space(RF, BASEBAND)
    assert null["flat_gain_split_residual_nepers"] < 1e-12
    assert null["pure_delay_split_residual_radians"] < 1e-12


def test_a_band_limit_at_either_end_is_one_direction():
    """A single pole well above the band has a log-magnitude derivative
    proportional to `f^2` whatever its corner, so the record amplifier's
    upper limit and the playback preamplifier's are the same direction. This
    is the one null here that is a LIMIT rather than an identity: a band that
    reached either corner would break it."""
    null = rf_stages.null_space(RF, BASEBAND)
    assert null["band_limit_coherence"] > 0.99


def test_phase_adds_no_direction_to_a_minimum_phase_family():
    """Every mechanism in this chain except a transport delay is minimum
    phase, and the Hilbert transform is unitary on mean-removed signals - so
    stacking magnitude and phase doubles the Gram matrix and leaves the
    participation ratio where it was. Phase buys a direction only for a
    non-minimum-phase mechanism, and a record delay and a playback delay are
    themselves only ever seen as their sum."""
    rng = np.random.default_rng(11)
    magnitudes = rng.standard_normal((6, 512))
    magnitudes -= magnitudes.mean(axis=1, keepdims=True)
    phases = np.array([rf_stages.minimum_phase_of(row) for row in magnitudes])
    alone = rf_stages.effective_count(list(magnitudes), demean=True)
    stacked = rf_stages.effective_count(
        [np.concatenate([m, p]) for m, p in zip(magnitudes, phases)],
        demean=True)
    assert stacked["effective"] == pytest.approx(alone["effective"], rel=1e-3)
    # and an all-pass, which is NOT minimum phase, does add one
    phases[0] = np.linspace(0.0, -3.0, 512)
    with_all_pass = rf_stages.effective_count(
        [np.concatenate([m, p]) for m, p in zip(magnitudes, phases)],
        demean=True)
    assert with_all_pass["effective"] > stacked["effective"]


# --------------------------------------------------------------------------
# What data would separate the two machines
# --------------------------------------------------------------------------


def test_one_deck_and_one_tape_separate_nothing():
    """`y = R + T + P` is a single sum. Neither machine is reachable, and no
    number of tapes on that deck changes it: more tapes move `T` and leave
    the two machines exactly where they were."""
    one = rf_stages.separation_design([("t1", "A")], [(0, "A")])
    assert not one["estimable"]["record against playback, one deck"]
    assert not one["estimable"]["the record stage alone"]

    more = rf_stages.separation_design([("t1", "A"), ("t2", "A")],
                                       [(0, "A"), (1, "A")])
    assert more["estimable"]["tapes differ"]
    assert not more["estimable"]["record against playback, one deck"]


def test_a_second_deck_separates_the_machines_from_each_other_only():
    """A second deck playing the same tape reaches how the PLAYBACK machines
    differ; the same source recorded on two decks reaches how the RECORD
    machines differ. Neither reaches the record path against the playback
    path on one deck, because the two occur in every observation together."""
    playing = rf_stages.separation_design([("t1", "A")],
                                          [(0, "A"), (0, "B")])
    assert playing["estimable"]["playback machines differ"]
    assert not playing["estimable"]["record machines differ"]

    recording = rf_stages.separation_design([("t1", "A"), ("t1", "B")],
                                            [(0, "A"), (1, "A")])
    assert recording["estimable"]["record machines differ"]

    crossed = rf_stages.separation_design(
        [("t1", "A"), ("t1", "B")],
        [(0, "A"), (0, "B"), (1, "A"), (1, "B")])
    assert crossed["estimable"]["record machines differ"]
    assert crossed["estimable"]["playback machines differ"]
    assert not crossed["estimable"]["record against playback, one deck"]


def test_only_a_record_tap_and_a_reference_together_separate_the_pair():
    """The test directory holds the record tap - `/testdata/test_patterns/
    vhs/` has both a record and a playback capture of every zaroff pattern
    from one Sony SLV-778HF. On its own the tap reaches the record stage and
    stops, because what is left of the playback still has the tape in it. Add
    a known reference recorded on the tape and the three unknowns are three
    equations, which is the only pairing among the five separators that
    reaches the pair."""
    tap = rf_stages.separation_design([("t1", "A")], [(0, "A")],
                                      record_taps=("A",))
    assert tap["estimable"]["the record stage alone"]
    assert not tap["estimable"]["record against playback, one deck"]

    reference = rf_stages.separation_design([("t1", "A")], [(0, "A")],
                                            known_tapes=("t1",))
    assert not reference["estimable"]["record against playback, one deck"]

    both = rf_stages.separation_design([("t1", "A")], [(0, "A")],
                                       record_taps=("A",),
                                       known_tapes=("t1",))
    assert both["estimable"]["record against playback, one deck"]
    assert both["deficiency"] == 0


def test_the_measured_tap_pairing_reports_what_it_separates():
    """The measurement on the eight zaroff patterns. The RATIO of the two
    taps is a response, because the same content passed through both:
    subtracting the record tap reduces the playback tap's log-magnitude shape
    3.3x at SP and 1.65x at EP, and what remains repeats 0.80 and 0.84 across
    content, which is what says it belongs to the chain. The record tap's own
    spectrum is not a response - its 1.67 nepers rms is the modulated
    carrier's shape as much as the chain's, and its 0.91 agreement across
    patterns says only that an FM spectrum is dominated by its carrier.

    THE REGRESSION BELOW UNITY IS THE WARNING IN THE MEASUREMENT. If the
    playback tap were the record tap through a linear channel the regression
    would be one; measured it is 0.835 at SP and 0.652 at EP, because the
    playback tap carries an additive noise floor that fills the FM spectrum's
    valleys and biases a log-spectral ratio toward zero wherever the record
    spectrum is near it."""
    measured = rf_stages.tap_separation()
    assert measured["record_tap_repeatable_sp"] > 0.9
    assert measured["round_trip_repeatable_sp"] > 0.75
    assert measured["playback_on_record_regression_sp"] < 1.0
    assert (measured["round_trip_rms_nepers_sp"]
            < measured["playback_tap_rms_nepers_sp"] / 3.0)

    # and the same computation, run on arrays
    rng = np.random.default_rng(3)
    record = rng.standard_normal(512)
    channel = np.linspace(-0.4, 0.4, 512)
    recomputed = rf_stages.tap_separation(record, record + channel)
    assert recomputed["regression"] == pytest.approx(1.0, abs=0.05)
    assert recomputed["round_trip_rms_nepers"] == pytest.approx(
        channel.std(), rel=1e-6)


# --------------------------------------------------------------------------
# The speed lever, and its confound
# --------------------------------------------------------------------------


def test_a_vhs_speed_switch_barely_moves_the_tape_side():
    """The drum turns at one rate whatever the tape speed, so the writing
    speed changes only by the tape's own contribution - 0.384% from SP to
    EP - and what really changes is the track pitch, 58 to 19.3 um. Both are
    already measured as inert to identifiability."""
    fast = rf_stages.mechanics_at_speed("SP")
    slow = rf_stages.mechanics_at_speed("EP")
    assert head_model.writing_speed(fast) == pytest.approx(5.80, abs=1e-6)
    change = abs(head_model.writing_speed(slow)
                 - head_model.writing_speed(fast)) / head_model.writing_speed(fast)
    assert change < 0.005
    assert slow["track_width_m"] == pytest.approx(19.3e-6)
    assert fast["track_width_m"] == pytest.approx(58e-6)


def test_the_speed_lever_is_almost_entirely_record_processing():
    """So SP against EP is not a controlled experiment on the record
    machine's magnetics: the tape arm is 3.5% of the lever and the record
    machine's own emphasis switch is 96.5%. And the emphasis switch is a
    whole processing stage one arm has and the other does not - VHS applies
    the extra record-side emphasis at LP, EP and SLP and not at SP."""
    result = rf_stages.speed_confound(RF)
    assert result["tape_arm_rms_nepers"] < 0.02
    assert result["tape_share_of_the_lever"] < 0.05
    assert result["processing_share_of_the_lever"] > 0.95
    assert result["confounded"]
    assert result["extra_emphasis_fast"] is False
    assert result["extra_emphasis_slow"] is True
    # and LP against EP is worse, not better: both arms carry the emphasis
    # and the mechanics move less still
    slower = rf_stages.speed_confound(RF, fast="LP", slow="EP")
    assert slower["tape_arm_rms_nepers"] < result["tape_arm_rms_nepers"]
    assert not slower["confounded"]


def test_the_record_tap_s_own_speed_difference_barely_repeats():
    """The trap in using the tap to break the confound: the record tap's
    SP-minus-EP difference repeats only 0.268 across eight patterns against
    0.816 for the playback tap's, because it is a small difference between
    two large and nearly identical FM spectra. Three quarters of what looks
    like a record-side speed effect is the picture."""
    measured = rf_stages.tap_separation()
    assert measured["speed_lever_repeatable_record_tap"] < 0.4
    assert measured["speed_lever_repeatable_playback_tap"] > 0.7


# --------------------------------------------------------------------------
# The effective count against the magnetics baseline
# --------------------------------------------------------------------------


def test_the_magnetics_baseline_is_reproduced_here():
    """Quoted rather than recomputed, 1.58 of 6 would be a number from
    another run on another band. Recomputed on this module's own procedure it
    is 1.576, so any difference between it and the combined count is the
    stages' doing and not the measure's."""
    baseline = rf_stages.magnetics_baseline()
    assert baseline["count"] == 6
    assert baseline["effective"] == pytest.approx(1.576, abs=0.02)
    assert baseline["condition"] > 1e4


def test_the_two_stages_are_the_same_collinear_family_seen_twice():
    """THE ANSWER TO THE COUNT. Six more members buy 0.23 of a direction:
    1.576 of 6 becomes 1.803 of 12. Every entry but two is another monotone
    function of one length over the recorded wavelength, which is the law
    that a family is collinear when its members differ only in a rate."""
    result = rf_stages.distinguishable(RF)
    assert result["baseline"]["effective"] == pytest.approx(1.576, abs=0.02)
    assert result["with_both"]["count"] == 12
    assert result["with_both"]["effective"] == pytest.approx(1.80, abs=0.05)
    assert result["gain_over_the_baseline"] < 0.5
    # neither stage on its own does better
    assert result["with_record"]["effective"] < 1.8
    assert result["with_playback"]["effective"] < 1.9
    # and the ensemble is rank deficient, which is the two exact nulls
    # arriving as arithmetic
    assert result["condition"] > 1e10


def test_the_two_entries_that_earn_their_place():
    """A modification earns a place in the key by the direction it adds, not
    by being a modification. Only two of the six stage entries do: the record
    current's level dependence, which is the axis `magnetic` established, and
    the playback equaliser, whose corner is INSIDE the band where every other
    electronic corner is outside it."""
    result = rf_stages.distinguishable(RF)
    names = result["names"]
    coherence = result["coherence"]
    worst = {}
    for i, name in enumerate(names[6:], start=6):
        others = [coherence[i, j] for j in range(len(names)) if j != i]
        worst[name] = max(others)
    assert worst["record level dependence"] < 0.5
    assert worst["playback equalisation"] < 0.7
    # while the rest are collinear with the family that already collapsed
    assert worst["record head write response"] > 0.99
    assert worst["record amplifier"] > 0.99
    assert worst["playback preamplifier"] > 0.99


def test_the_two_level_dependent_record_stages_are_nearly_one_axis():
    """The S-VHS sub pre-emphasis and the non-linear emphasis are both a
    high-frequency boost whose amount falls as the drive rises, and they read
    0.986 coherent. The recording machine's two non-linear stages are one
    axis, not two - the same lesson the sub emphasis taught when entering it
    as two of its tabulated curves lowered the count from 6.92 to 6.00."""
    baseband = rf_stages.baseband_distinguishable(BASEBAND)
    assert baseband["level_dependent_pair_coherence"] > 0.95
    assert baseband["effective"] < 2.0


# --------------------------------------------------------------------------
# The closed forms themselves
# --------------------------------------------------------------------------


def test_the_main_pre_emphasis_matches_the_decoder_s_own_shelf():
    """IEC 774-1 (1994) page 67 gives an RC of 1.3 us and a 4:1 divider, so a
    gain factor of 5. The repository's own format definitions carry the same
    network as a shelf, `deemph_mid` 273755.82 Hz and `deemph_gain` 13.9794
    dB, and the two parametrizations must agree - which is a check on this
    module rather than a restatement of it."""
    gain = rf_stages.MAIN_EMPHASIS_GAIN_FACTOR
    tau = rf_stages.MAIN_EMPHASIS_TIME_CONSTANT_S
    mid = np.sqrt(gain) / (2.0 * np.pi * tau)
    # 273754.82 against the file's 273755.82: four parts in a million, which
    # is the file's own rounding of the same network and not a disagreement
    assert mid == pytest.approx(273755.82, rel=1e-5)
    assert 20.0 * np.log10(gain) == pytest.approx(13.9794, abs=1e-4)
    # the shelf itself: unity at DC, the gain factor well above the corner,
    # and the geometric mean of the two exactly at `mid`
    response = rf_stages.main_pre_emphasis(np.array([1.0, mid, 1e9]))
    assert abs(response[0]) == pytest.approx(1.0, abs=1e-6)
    assert abs(response[2]) == pytest.approx(gain, rel=1e-6)
    assert abs(response[1]) == pytest.approx(np.sqrt(gain), rel=1e-6)


def test_the_carriers_and_deviations_are_the_standards_figures():
    """IEC 60774-1 gives VHS as 3.4/4.4 MHz for 525 lines and 3.8/4.8 for
    625, both with 1.0 MHz of deviation; IEC 60774-3 gives S-VHS as 5.4/7.0
    with 1.6 MHz, identically for both line standards."""
    for system in ("525", "625"):
        assert rf_stages.deviation_hz(
            rf_stages.VHS_CARRIER_HZ[system]) == pytest.approx(1.0e6)
    assert rf_stages.deviation_hz(
        rf_stages.SVHS_CARRIER_HZ) == pytest.approx(1.6e6)
    # and the S-VHS pair is one entry because it does not vary with the line
    # standard, which is why it needs no per-system table
    assert isinstance(rf_stages.SVHS_CARRIER_HZ, tuple)


def test_the_limiter_describing_function_is_the_level_axis():
    """The non-linear emphasis is a level axis because its limit binds at a
    different frequency for every drive - the same structure the
    self-demagnetisation cap gives the record level, where removing the cap
    collapsed the axis to exactly one direction."""
    assert rf_stages.saturation_gain(np.array([0.0, 0.5, 1.0]))[2] == \
        pytest.approx(1.0)
    beyond = rf_stages.saturation_gain(np.array([2.0, 4.0, 16.0]))
    assert np.all(np.diff(beyond) < 0)          # the gain falls with drive
    assert beyond[-1] < 0.2
    # and the response therefore moves with the drive rather than scaling
    low = rf_stages.nonlinear_pre_emphasis(BASEBAND, 4000.0)
    high = rf_stages.nonlinear_pre_emphasis(BASEBAND, 40000.0)
    assert not np.allclose(np.abs(low), np.abs(high))
    ratio = rf_stages.nonlinear_emphasis_level_dependence(
        BASEBAND, 4000.0, 40000.0)
    assert interference.subtractable(ratio)
    assert np.abs(ratio).max() / np.abs(ratio).min() > 1.05


def test_the_playback_head_differentiates_and_the_record_head_does_not():
    """The reproduce head's output is the rate of change of the flux it
    links, so it rises at six decibels per octave with exactly a quarter turn
    of phase. The record head has no counterpart: the recorded magnetisation
    follows the record current, and a current is not a derivative of
    anything. This is the only mechanism either machine owns outright."""
    response = rf_stages.playback_head_differentiation(np.array([1e6, 2e6]))
    assert abs(response[1]) / abs(response[0]) == pytest.approx(2.0)
    assert np.angle(response[0]) == pytest.approx(np.pi / 2.0)
    # and it is weak on a narrow band, which is the actionable rule: fit over
    # the widest band the RF occupies
    def gap(band):
        shape = np.log(np.abs(
            rf_stages.playback_head_differentiation(band)))
        flat = np.ones_like(band)
        shape = shape - shape.mean()
        flat = flat - flat.mean() + 1e-12 * (band - band.mean())
        return abs(shape @ flat) / (np.linalg.norm(shape)
                                    * np.linalg.norm(flat))
    narrow = gap(np.linspace(3.0e6, 5.0e6, 512))
    wide = gap(np.linspace(0.1e6, 20e6, 512))
    assert narrow > wide


def test_the_sub_emphasis_is_read_from_the_standard_and_not_retranscribed():
    """IEC 60774-3 table 3 is already transcribed in `interference
    .SUB_EMPHASIS_DB`, separately for SP and for EP/LP, and it is read rather
    than copied. What this module changes is the CONVENTION: it returns a
    response where that function returns nepers, so the key has one kind of
    entry throughout."""
    assert set(interference.SUB_EMPHASIS_DB) >= {"SP", "EP", "LP"}
    level = rf_stages.sub_pre_emphasis_level_dependence(BASEBAND, "EP")
    assert interference.subtractable(level)
    assert np.all(np.abs(level) > 0)
    nepers = (interference.sub_emphasis(BASEBAND, -20.0, "EP").real
              - interference.sub_emphasis(BASEBAND, 0.0, "EP").real)
    assert np.allclose(np.log(np.abs(level)), nepers)
    # the two tape speeds have different tables, which is the confound in the
    # speed lever stated by the standard itself
    assert not np.allclose(
        np.abs(rf_stages.sub_pre_emphasis_level_dependence(BASEBAND, "SP")),
        np.abs(level))


def test_the_speed_switch_decides_whether_the_extra_emphasis_runs():
    """From the repository's own speed table: VHS applies the extra
    record-side emphasis at LP, EP and SLP and not at SP, and S-VHS applies
    it always. That difference is the confound, stated where it lives."""
    vhs = rf_stages.EXTRA_EMPHASIS_BY_SPEED["VHS"]
    assert vhs["SP"] is False
    assert vhs["LP"] is vhs["EP"] is vhs["SLP"] is True
    assert all(rf_stages.EXTRA_EMPHASIS_BY_SPEED["SVHS"].values())
    # so a VHS SP key carries neither extra emphasis, a VHS EP key carries
    # the non-linear one, and an S-VHS key carries both - and the difference
    # between the first two IS the confound the speed lever runs into
    at_sp = rf_stages.record_signatures(BASEBAND, RF, speed="SP",
                                        tape_format="VHS")
    at_ep = rf_stages.record_signatures(BASEBAND, RF, speed="EP",
                                        tape_format="VHS")
    svhs = rf_stages.record_signatures(BASEBAND, RF, speed="SP",
                                       tape_format="SVHS")
    nonlinear = "record non-linear emphasis level dependence"
    sub = "sub emphasis level dependence"
    assert nonlinear not in at_sp and sub not in at_sp
    assert nonlinear in at_ep and sub not in at_ep
    assert nonlinear in svhs and sub in svhs
    # the SP and EP record keys differ by a whole processing stage, which is
    # exactly what makes an SP-versus-EP comparison not a controlled one
    assert set(at_ep) - set(at_sp) == {nonlinear}
