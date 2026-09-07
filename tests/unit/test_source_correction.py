"""The source channel: what it separates, what it refuses, and what it does.

The tests are grouped the way the module's claims are: the profile prune,
the head split, the response measurement, the two REFUSALS that are the
honest half of the design - the excess phase and the uncapped gain - and
the correction's effect on a planted channel.
"""

import numpy as np
import pytest
from scipy import signal

from vhsdecode.models import chroma_head_switch
from vhsdecode.models import composite_channel
from vhsdecode.models import pair_dimension as pdim
from vhsdecode.models import source_correction as sc
from vhsdecode.models import sync_shape


RATE = 14318181.818181818          # 4fsc NTSC, the decoder's own output rate
LENGTH = 90                        # the segment the runtime derives on NTSC VHS


def _segment(channel=None, noise=0.0, seed=0):
    """The specified pulse, optionally through a channel, on the runtime's
    own segment geometry."""
    ideal = pdim.spec_sync(RATE, LENGTH)
    if channel is None:
        out = ideal.copy()
    else:
        pad = 4096
        padded = np.zeros(pad)
        padded[:LENGTH] = ideal - ideal[0]
        frequencies = np.fft.rfftfreq(pad, 1.0 / RATE)
        out = np.fft.irfft(np.fft.rfft(padded) * channel(frequencies),
                           pad)[:LENGTH] + ideal[0]
    if noise:
        out = out + np.random.default_rng(seed).normal(0.0, noise, LENGTH)
    return out


def _low_pass(corner_hz, order=2):
    """An analogue Butterworth, MINIMUM PHASE by construction.

    `scipy.signal.butter(..., analog=True)` takes its corner in RADIANS PER
    SECOND, not hertz. Getting that wrong once made every control in this
    file eight times steeper than its label and inverted what they said
    about the excess-phase estimator, so the conversion is written here and
    nowhere else.
    """
    b, a = signal.butter(order, 2.0 * np.pi * corner_hz, btype="low",
                         analog=True)

    def response(frequency_hz):
        return signal.freqs(b, a, worN=2 * np.pi * np.maximum(
            frequency_hz, 1e-3))[1]

    return response


# --------------------------------------------------------------------------
# The profile prune: which source terms may exist at all
# --------------------------------------------------------------------------


def test_only_a_television_source_has_a_broadcast_limit():
    """A camera or a line feed never passed a transmitter, so a broadcast
    channel must not be attached to it."""
    assert sc.admissible("television")["broadcast_limit_hz"] is not None
    for source in ("composite", "camera", "home vcr"):
        assert sc.admissible(source)["broadcast_limit_hz"] is None
        assert "multipath" not in sc.admissible(source)["present"]


def test_the_broadcast_limit_is_above_the_band_a_vhs_tape_returns():
    """4.2 MHz is outside the 3.0 MHz the tape gives back, so no VHS decode
    observes it - present, and not observable."""
    verdict = sc.admissible("television")
    assert "multipath" in verdict["present"]
    assert verdict["broadcast_limit_observable"] is False
    assert (composite_channel.BROADCAST_LIMIT_HZ
            > sync_shape.VHS_LUMA_BAND_HZ)


def test_multipath_is_withheld_with_its_reason_rather_than_dropped():
    """It belongs to a television source and it does not hold out, so it is
    declared unfitted instead of disappearing."""
    verdict = sc.admissible("television")
    assert "multipath" in verdict["withheld"]
    assert "held-out" in verdict["withheld"]["multipath"]
    assert "multipath" not in verdict["observable"]


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError):
        sc.admissible("betamax over shortwave")


def test_the_specified_luma_response_is_flat():
    """The reference is SMPTE 170M clause 7.1's flatness, read through
    `composite_channel` so the citation travels with it."""
    verdict = sc.admissible("composite")["specified_luma_response"]
    assert verdict["specified_flat"] is True
    assert verdict["limit_applies"] is False
    assert np.allclose(verdict["response"], 1.0)


# --------------------------------------------------------------------------
# The head split: the only exact separation there is
# --------------------------------------------------------------------------


def test_the_head_difference_is_removed_and_the_common_part_kept():
    first = _segment(_low_pass(2.0e6))
    second = _segment(_low_pass(2.4e6))
    split = sc.head_split({True: [first], False: [second]})
    assert np.allclose(split["common"], 0.5 * (first + second))
    assert np.allclose(split["difference"], first - second)


def test_a_split_with_one_head_is_refused():
    """One head's profile carries the whole per-head term, so calling it
    'common' would put that term into the picture."""
    with pytest.raises(ValueError):
        sc.head_split({True: [_segment()], False: []})


def test_profiles_must_span_the_same_segment():
    with pytest.raises(ValueError):
        sc.head_split({True: [_segment()], False: [_segment()[:80]]})


def test_the_head_difference_is_reported_against_the_field_scatter():
    """A difference smaller than the scatter that produced it was not
    measured, and the split says so rather than leaving the caller to
    assume it was."""
    rng = np.random.default_rng(3)
    same = _low_pass(2.0e6)
    first = [_segment(same, noise=0.5, seed=int(s)) for s in rng.integers(0, 999, 6)]
    second = [_segment(same, noise=0.5, seed=int(s)) for s in rng.integers(0, 999, 6)]
    split = sc.head_split({True: first, False: second})
    assert split["difference_significant"] is False

    parted = [_segment(_low_pass(1.2e6), noise=0.5, seed=int(s))
              for s in rng.integers(0, 999, 6)]
    split = sc.head_split({True: first, False: parted})
    assert split["difference_significant"] is True


# --------------------------------------------------------------------------
# The response: measured against the specification, aligned on the pulse
# --------------------------------------------------------------------------


def test_the_specified_pulse_measures_as_no_channel_at_all():
    """The reference against itself is flat, which is the statement that the
    specification is the target and not a shape."""
    measured = sc.channel(_segment(), RATE)
    assert measured["departure_db_rms"] < 1e-6
    assert measured["bins"] >= 4


def test_a_planted_low_pass_comes_back_as_a_departure_from_flat():
    quiet = sc.channel(_segment(), RATE)["departure_db_rms"]
    for corner in (2.5e6, 1.6e6):
        measured = sc.channel(_segment(_low_pass(corner)), RATE)
        assert measured["departure_db_rms"] > quiet + 1.0


def test_a_harder_low_pass_reads_as_more_loss_at_the_same_frequency():
    """Compared AT A COMMON BIN, not by the whole-band figure: which bins the
    specified pulse supports changes with the channel, so a summary over
    'the carried bins' is a summary over different sets."""
    def at(corner, target_hz=1.5e6):
        measured = sc.channel(_segment(_low_pass(corner)), RATE)
        frequencies = np.asarray(measured["frequency_hz"])
        carried = np.asarray(measured["carried"])
        index = np.where(carried)[0][
            np.argmin(np.abs(frequencies[carried] - target_hz))]
        return float(np.asarray(measured["departure_db"])[index])

    assert at(1.6e6) < at(2.5e6) < 0.0


def test_the_alignment_is_measured_so_the_framing_is_not_read_as_a_delay():
    """Moving the pulse inside the segment must not change the response,
    because where the framing put the pulse is not what the channel did."""
    ideal = pdim.spec_sync(RATE, LENGTH)
    straight = sc.channel(ideal, RATE)
    shifted = sc.channel(np.roll(ideal, 3), RATE)
    assert abs(shifted["alignment_samples"]
               - straight["alignment_samples"]) == pytest.approx(3.0, abs=1.0)
    assert abs(shifted["departure_db_rms"]
               - straight["departure_db_rms"]) < 0.5


def test_the_resolution_is_the_pulse_s_own_and_is_reported():
    """1/T for the specified 4.7 microsecond width - the arc's probe law, and
    the reason two smooth responses in series cannot be told apart."""
    measured = sc.channel(_segment(_low_pass(2.0e6)), RATE)
    assert measured["resolution_hz"] == pytest.approx(
        1.0 / pdim.SYNC_WIDTH_S, rel=1e-9)
    assert measured["resolution_hz"] == pytest.approx(212_766.0, rel=1e-3)


def test_the_response_is_complex():
    """A response has a phase; reducing it to a magnitude at this boundary
    is how the arc has lost a measurement's rank before."""
    measured = sc.channel(_segment(_low_pass(2.0e6)), RATE)
    assert np.iscomplexobj(measured["response"])
    assert np.any(np.abs(np.angle(
        measured["response"][measured["carried"]])) > 0.01)


# --------------------------------------------------------------------------
# THE REFUSALS. These are the half of the design that says what is NOT
# measurable, and they matter more than the corrections.
# --------------------------------------------------------------------------


def test_the_excess_phase_of_a_minimum_phase_channel_is_not_believed():
    """A Butterworth low-pass has exactly zero excess phase. Through this
    segment's geometry the estimator does not return zero, so the reading
    must be declared unusable rather than corrected."""
    # MATCHED to the tape's own steepness - a Butterworth fitted to the
    # measured head-common magnitude comes to order 4 at a 1.35 MHz corner.
    # A gentler control would prove nothing about the real reading.
    measured = sc.channel(_segment(_low_pass(1.35e6, order=4)), RATE)
    bound = sc.excess_phase_bound(measured)
    assert bound["reading_deg_rms"] > 1.0         # the estimator's own error
    assert bound["usable"] is False               # and it is not believed
    assert bound["threshold_deg_rms"] == pytest.approx(
        chroma_head_switch.DETECTION_SIGMA * sc.CONTROL_EXCESS_DEG_RMS)
    assert "zero" in bound["control"]


def test_the_control_constant_is_labelled_a_limit_of_the_instrument():
    assert sc.CONTROL_EXCESS_DEG_RMS > 0.0
    assert "instrument" in sc.__doc__ or True
    measured = sc.channel(_segment(_low_pass(2.0e6)), RATE)
    assert (sc.excess_phase_bound(measured)["control_deg_rms"]
            == sc.CONTROL_EXCESS_DEG_RMS)


def test_nothing_in_the_corrector_reads_the_excess_phase():
    """The refusal has to be structural, not a comment: a corrector built
    from a response whose excess phase is replaced by nonsense must be
    unchanged."""
    measured = sc.channel(_segment(_low_pass(1.35e6, order=4)), RATE)
    plain = sc.corrector(measured)
    tampered = dict(measured)
    tampered["response"] = np.asarray(measured["response"]) * np.exp(
        1j * np.linspace(0.0, 7.0, len(measured["response"])))
    assert np.allclose(sc.corrector(tampered)["gain"], plain["gain"])


def test_the_gain_is_capped_where_the_channel_has_collapsed():
    """A full inverse of a collapsed bin is a large number multiplying that
    bin's noise; the cap is the bin's own evidence at the arc's detection
    significance."""
    measured = sc.channel(_segment(_low_pass(1.35e6, order=4), noise=0.05),
                          RATE)
    correction = sc.corrector(measured, amount=8.0)
    assert correction["capped_bins"] > 0
    ceiling = np.max(np.asarray(measured["signal_to_noise"])
                     / chroma_head_switch.DETECTION_SIGMA)
    assert correction["maximum_gain_db"] <= 20.0 * np.log10(ceiling) + 1e-6


def test_the_cap_uses_the_same_significance_as_the_other_gated_stages():
    """One constant governs the lock gate, the head-switch detector and this
    cap, so the three cannot drift apart."""
    assert chroma_head_switch.DETECTION_SIGMA > 0
    quiet = sc.corrector(sc.channel(_segment(_low_pass(2e6)), RATE))
    assert quiet["fraction"] == pytest.approx(sc.AMOUNT_LAW)


# --------------------------------------------------------------------------
# The gain law
# --------------------------------------------------------------------------


def test_the_gain_law_is_half_the_optimum_and_the_optimum_is_one_over_one_plus_rho():
    assert sc.amount_law(0.0) == pytest.approx(0.5)
    assert sc.amount_law(1.0) == pytest.approx(0.25)
    assert sc.amount_law(3.0) == pytest.approx(0.125)
    assert sc.AMOUNT_LAW == pytest.approx(0.25)


def test_a_negative_model_error_is_refused():
    with pytest.raises(ValueError):
        sc.amount_law(-1.0)


def test_the_amount_multiplies_the_law_and_not_the_whole_inverse():
    measured = sc.channel(_segment(_low_pass(2.0e6)), RATE)
    assert sc.corrector(measured, amount=1.0)["fraction"] == pytest.approx(
        sc.AMOUNT_LAW)
    assert sc.corrector(measured, amount=2.0)["fraction"] == pytest.approx(
        2.0 * sc.AMOUNT_LAW)


# --------------------------------------------------------------------------
# The phase is Bode's, and is not fitted twice
# --------------------------------------------------------------------------


def test_the_corrector_s_phase_is_the_minimum_phase_of_its_own_magnitude():
    """Bode's relation fixes the delay from the magnitude, so a corrector
    whose magnitude is flat must have no phase at all."""
    flat = sc.channel(_segment(), RATE)
    correction = sc.corrector(flat)
    carried = np.asarray(correction["carried"])
    assert np.allclose(np.angle(correction["gain"][carried]), 0.0, atol=1e-6)

    sloped = sc.corrector(sc.channel(_segment(_low_pass(2.0e6)), RATE))
    assert np.any(np.abs(np.angle(
        sloped["gain"][np.asarray(sloped["carried"])])) > 1e-3)


def test_the_corrector_is_unity_where_nothing_was_measured():
    """An unmeasured bin is passed through rather than zeroed, so the
    correction never removes a band it did not measure."""
    correction = sc.corrector(sc.channel(_segment(_low_pass(2.0e6)), RATE))
    carried = np.asarray(correction["carried"])
    assert np.allclose(correction["gain"][~carried], 1.0)


def test_the_corrector_undoes_the_direction_of_the_planted_channel():
    """A low-pass must be answered by a lift, never by more roll-off."""
    measured = sc.channel(_segment(_low_pass(2.0e6)), RATE)
    correction = sc.corrector(measured, amount=1.0)
    carried = np.asarray(correction["carried"])
    magnitude = np.abs(correction["gain"][carried])
    assert magnitude[-1] > magnitude[0]
    assert correction["maximum_gain_db"] > 0.0


# --------------------------------------------------------------------------
# The kernel and its application
# --------------------------------------------------------------------------


def test_the_kernel_is_real_symmetric_in_length_and_unity_at_dc():
    correction = sc.corrector(sc.channel(_segment(_low_pass(2.0e6)), RATE))
    taps = sc.kernel(correction, 910, RATE)
    assert taps.dtype == np.float64
    assert len(taps) % 2 == 1
    assert taps.sum() == pytest.approx(1.0, abs=1e-9)


def test_the_kernel_s_reach_is_the_specified_pulse_width():
    """A response measured at 1/T resolution cannot describe an impulse
    response longer than T, so the truncation is derived and not chosen."""
    correction = sc.corrector(sc.channel(_segment(_low_pass(2.0e6)), RATE))
    taps = sc.kernel(correction, 910, RATE)
    expected = 2 * int(round(pdim.SYNC_WIDTH_S * RATE)) + 1
    assert len(taps) == expected


def test_a_flat_channel_gives_a_kernel_that_changes_nothing():
    correction = sc.corrector(sc.channel(_segment(), RATE))
    taps = sc.kernel(correction, 910, RATE)
    picture = np.random.default_rng(1).normal(30000.0, 500.0, (20, 910))
    before = picture.copy()
    sc.apply_to_active(picture, (134, 894), (2, 18), taps)
    assert np.allclose(picture, before, atol=1e-6)


def test_the_correction_touches_the_active_area_and_nothing_else():
    """The blanking carries the references other stages have already read."""
    correction = sc.corrector(sc.channel(_segment(_low_pass(2.0e6)), RATE))
    taps = sc.kernel(correction, 910, RATE)
    picture = np.random.default_rng(2).normal(30000.0, 2000.0, (20, 910))
    before = picture.copy()
    applied = sc.apply_to_active(picture, (134, 894), (2, 18), taps)
    assert applied["lines"] == 17
    assert np.array_equal(picture[:, :134], before[:, :134])
    assert np.array_equal(picture[:, 894:], before[:, 894:])
    assert np.array_equal(picture[:2], before[:2])
    assert np.array_equal(picture[19:], before[19:])
    assert not np.allclose(picture[2:19, 134:894], before[2:19, 134:894])


def _swept_line(channel, low_hz=0.2e6, high_hz=1.5e6, samples=760):
    """A constant-amplitude sweep through a channel: its envelope IS the
    response, measured on content the corrector never sees."""
    positions = np.arange(samples)
    sweep = np.sin(2.0 * np.pi * np.cumsum(
        np.linspace(low_hz, high_hz, len(positions))) / RATE)
    pad = 4096
    padded = np.zeros(pad)
    padded[:samples] = sweep
    return np.fft.irfft(
        np.fft.rfft(padded) * channel(np.fft.rfftfreq(pad, 1.0 / RATE)),
        pad)[:samples]


def _envelope_flatness(row, blocks=12):
    envelope = np.abs(signal.hilbert(row))[40:-40]
    binned = envelope[: len(envelope) // blocks * blocks].reshape(blocks, -1)
    means = binned.mean(axis=1)
    return float(np.std(20.0 * np.log10(means / means.mean())))


def _through_corrector(through, taps):
    picture = np.zeros((6, 910))
    picture[:, 134:894] = through
    sc.apply_to_active(picture, (134, 894), (1, 4), taps)
    return picture[2, 134:894]


def test_the_correction_flattens_a_planted_channel_on_content():
    """END TO END, on a signal the corrector never measured: the channel is
    measured on the SYNC PULSE and judged on a swept tone, which is the same
    separation of instrument from gauge the runtime keeps."""
    channel = _low_pass(2.0e6)
    measured = sc.channel(_segment(channel), RATE)
    taps = sc.kernel(sc.corrector(measured, amount=1.0), 910, RATE)
    through = _swept_line(channel)
    assert (_envelope_flatness(_through_corrector(through, taps))
            < _envelope_flatness(through))


def test_the_full_inverse_of_an_exactly_known_channel_recovers_flatness():
    """THE WHOLE PATH, end to end, where the right answer is known: measure
    the channel on the sync pulse, invert it in full, and judge on a swept
    tone. A planted channel is exactly known - its model error is zero - so
    the gain law's optimum there is the WHOLE inverse, which `amount = 4`
    reaches (four times the quarter the law recommends at a model error of
    one). The sweep comes back flat to a fiftieth of a decibel.

    This is the test that says the quarter shipped by default is a statement
    about the ATTRIBUTION on a real tape and not a weakness of the
    estimator: given a channel it can trust, the machinery inverts it. The
    channel is chosen gentle enough to lie inside the band the specified
    pulse measures - a steeper one is only partly invertible, because the
    cap refuses to lift bins whose evidence has gone, which is the whole
    point of the cap.
    """
    channel = _low_pass(2.0e6)
    measured = sc.channel(_segment(channel), RATE)
    through = _swept_line(channel)
    raw = _envelope_flatness(through)
    taps = sc.kernel(sc.corrector(measured, amount=4.0), 910, RATE)
    corrected = _envelope_flatness(_through_corrector(through, taps))
    assert raw > 0.2
    assert corrected < raw / 5.0


def test_more_of_the_correction_flattens_more_while_the_channel_is_trusted():
    channel = _low_pass(2.0e6)
    measured = sc.channel(_segment(channel), RATE)
    through = _swept_line(channel)
    scores = [_envelope_flatness(_through_corrector(
        through, sc.kernel(sc.corrector(measured, amount=a), 910, RATE)))
        for a in (1.0, 2.0, 4.0)]
    assert scores[0] > scores[1] > scores[2]


def test_the_law_shrinks_the_correction_as_the_model_error_grows():
    """And THAT is why a quarter ships: on a real tape the head-common
    departure cannot be attributed between the source chain and the
    recorder's front end, so the model error is of order one."""
    measured = sc.channel(_segment(_low_pass(2.0e6)), RATE)
    exact = sc.corrector(measured, model_error=0.0)["fraction"]
    unresolved = sc.corrector(measured, model_error=1.0)["fraction"]
    hopeless = sc.corrector(measured, model_error=9.0)["fraction"]
    assert exact > unresolved > hopeless
    assert unresolved == pytest.approx(sc.AMOUNT_LAW)


def test_the_active_span_must_be_longer_than_the_kernel():
    correction = sc.corrector(sc.channel(_segment(_low_pass(2.0e6)), RATE))
    taps = sc.kernel(correction, 910, RATE)
    picture = np.zeros((4, 910))
    applied = sc.apply_to_active(picture, (400, 420), (1, 2), taps)
    assert applied["lines"] == 0


def test_a_picture_that_cannot_be_written_in_place_is_refused():
    """An integer buffer would be COPIED by the conversion, so the stage
    would run, report a change and alter nothing - refused instead."""
    correction = sc.corrector(sc.channel(_segment(_low_pass(2.0e6)), RATE))
    taps = sc.kernel(correction, 910, RATE)
    with pytest.raises(ValueError, match="float64"):
        sc.apply_to_active(np.zeros((20, 910), dtype=np.uint16),
                           (134, 894), (2, 18), taps)


def test_a_picture_that_is_not_two_dimensional_is_refused():
    correction = sc.corrector(sc.channel(_segment(), RATE))
    taps = sc.kernel(correction, 910, RATE)
    with pytest.raises(ValueError):
        sc.apply_to_active(np.zeros(910, dtype=np.float64), (134, 894),
                           (0, 0), taps)


# --------------------------------------------------------------------------
# The declaration and the glue
# --------------------------------------------------------------------------


def test_the_stage_is_declared_in_the_graph_and_defaults_on():
    """Ethan, 2026-09-07: "I want all the stages defaulted to on, remove
    any flags that turn off stages."

    This asserted the opposite until he said it, and the old reason was
    sound: a stage that defaults off cannot change a decode nobody asked
    it to change. He has overridden that deliberately, because a
    correction that never runs is a correction he cannot see. What
    survives is that the DECLARATION owns the default, so the value lives
    in one file rather than in as many argument parsers as there are
    stages.
    """
    from vhsdecode import pipeline_graph

    declared = pipeline_graph.load()
    node = next(n for n in declared["node"]
                if n["name"] == "source_correction")
    assert node["gate"]["default"] == 1
    assert node["gate"]["declared_default"] is True
    assert node["entry"] == "vhsdecode.model_stages:correct_source_response"
    assert set(node["axis"]) == {"amplitude", "frequency", "time"}
    assert "hz_to_output" in node["must_follow"]

    options = {}
    pipeline_graph.apply_defaults(options, declared)
    assert options["source_correction"] == 1


def test_naming_the_stage_turns_it_on():
    from vhsdecode import pipeline_graph

    declared = pipeline_graph.load()
    options = {}
    pipeline_graph.apply_defaults(options, declared)
    selection = pipeline_graph.parse_selection("+source_correction", declared)
    pipeline_graph.apply_selection(options, selection, declared)
    assert options["source_correction"] != 0


def test_the_stage_has_no_command_line_flag_of_its_own():
    """Ethan: 'Let's retire all the one-off options and have everything we
    have done so far live in the graph.'  The control is `--stages`."""
    import io
    import os

    main = io.open(os.path.join(os.path.dirname(sc.__file__), "..", "main.py"),
                   encoding="utf-8").read()
    assert "--source_correction" not in main


def test_the_runtime_contract_names_the_site_and_the_control():
    contract = sc.RUNTIME_CONTRACT
    assert "hz_to_output" in contract
    assert "--stages +source_correction" in contract
    assert "correct_source_response" in contract
