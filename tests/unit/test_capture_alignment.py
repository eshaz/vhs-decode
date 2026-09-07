"""Aligning two radio-frequency captures of one tape pass.

Ethan: *"I can use the residual carriers in the video track to synchronize
a separate hifi rf capture with the video rf capture."* The route he named
is bounded closed, so what is tested here is the answer to the question it
was for: the drum seen twice, the tape crossed twice, and the arithmetic
that turns a timing figure into a physical one.

THERE IS NO AUDIO CAPTURE ON THIS MACHINE. Every test below that involves
two captures builds the second one, and each such test says so in its own
name or docstring. None of them is evidence that the alignment works on
real material; they establish that the estimator recovers what is put into
it and that the specification arithmetic is right.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import capture_alignment as ca
from vhsdecode.models import hifi_carriers, interference


# --------------------------------------------------------------------------
# the specification
# --------------------------------------------------------------------------

def test_every_specification_entry_carries_its_clause_or_says_it_is_absent():
    for name, entry in ca.SPECIFICATION.items():
        assert entry["quote"], f"{name} has no quotation"
        if entry["clause"] is None:
            assert "NOT STATED" in entry["quote"], (
                f"{name} has no clause and does not say the standard is "
                "silent")
        else:
            assert entry["clause"][0].isdigit() or entry["clause"].startswith("table")


def test_the_permitted_lead_is_exactly_one_drum_revolution_at_every_speed():
    """The central result: the standard refuses to fix the offset, and it
    refuses by exactly the whole circle."""
    out = ca.offset_is_not_specified()
    for speed, revolutions in out["width_in_drum_revolutions"].items():
        assert revolutions == pytest.approx(1.0, abs=1e-12), speed
    assert out["permitted_angle_deg"] == pytest.approx(360.0)


def test_the_lead_window_has_the_sign_the_standard_gives_it():
    """Clause 5.3: the audio is recorded PRIOR to the video at the same
    place on the tape, so the lead is non-negative."""
    for speed in ("SP", "LP", "EP"):
        low, high = ca.audio_lead_window(speed)["seconds"]
        assert low >= 0.0 and high > low


def test_an_unknown_tape_speed_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        ca.audio_lead_window("SLP")


def test_the_head_pair_placement_is_derived_and_self_consistent():
    """LP and EP both land 60 degrees from the SP pair modulo 180, which is
    one extra pair mounted at 60 degrees seen from its two ends."""
    out = ca.head_pair_placement()
    assert out["status"].startswith("DERIVED")
    places = out["placements"]
    assert places["SP"]["degrees_from_sp"] == pytest.approx(0.0)
    assert places["LP"]["degrees_from_sp_mod_180"] == pytest.approx(60.0)
    assert places["EP"]["degrees_from_sp_mod_180"] == pytest.approx(60.0)
    # and they are genuinely different placements, not the same one twice
    assert places["LP"]["degrees_from_sp"] != places["EP"]["degrees_from_sp"]


def test_the_drum_rate_is_taken_from_the_module_that_owns_it():
    r = ca.rates()
    assert r["drum_rate_hz"] == pytest.approx(r["field_rate_hz"] / 2.0)
    assert r["drum_period_s"] == pytest.approx(2.0 * r["field_period_s"])


# --------------------------------------------------------------------------
# the drum-phase estimator, on planted events
# --------------------------------------------------------------------------

def _planted_switches(count, period_s, epoch_s=0.0, jitter_s=0.0, seed=0):
    """A run of head-switch times. SYNTHETIC - this is a test fixture and
    is never offered as a measurement."""
    rng = np.random.default_rng(seed)
    k = np.arange(count, dtype=np.float64)
    return epoch_s + k * period_s + rng.normal(0.0, jitter_s, count)


def test_the_estimator_recovers_a_planted_rate_and_phase():
    r = ca.rates()
    times = _planted_switches(60, r["field_period_s"], epoch_s=0.0012)
    fit = ca.switch_phase(times)
    assert fit["period_s"] == pytest.approx(r["field_period_s"], rel=1e-12)
    assert fit["residual_rms_s"] < 1e-15
    assert fit["count"] == 60


def test_a_missing_event_must_be_left_out_by_index_not_renumbered():
    """Renumbering turns a gap into a rate error, which is the whole reason
    `index` exists."""
    r = ca.rates()
    k = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.float64)
    times = 0.005 + k * r["field_period_s"]
    honest = ca.switch_phase(times, index=k)
    assert honest["period_s"] == pytest.approx(r["field_period_s"], rel=1e-12)
    renumbered = ca.switch_phase(times)
    assert renumbered["period_s"] != pytest.approx(r["field_period_s"], rel=1e-6)


def test_the_estimator_refuses_a_run_too_short_to_carry_a_rate():
    with pytest.raises(ValueError):
        ca.switch_phase([0.0, 0.0167])


def test_the_phase_error_falls_as_one_over_the_root_of_the_count():
    r = ca.rates()
    scatter = 50e-9
    small = ca.switch_phase(_planted_switches(25, r["field_period_s"],
                                              jitter_s=scatter, seed=1))
    large = ca.switch_phase(_planted_switches(400, r["field_period_s"],
                                              jitter_s=scatter, seed=1))
    # four times the count, half the error, to within the estimate's own noise
    assert large["epoch_se_s"] < small["epoch_se_s"]
    assert large["epoch_se_s"] == pytest.approx(small["epoch_se_s"] / 4.0,
                                                rel=0.5)


def test_a_planted_offset_between_two_captures_comes_back(monkeypatch):
    """SYNTHETIC PAIR. There is no audio capture on this machine, so the
    second run is built here; this tests the estimator, not the deck."""
    r = ca.rates()
    planted = 3.7e-3
    video = ca.switch_phase(_planted_switches(120, r["field_period_s"], 0.001))
    audio = ca.switch_phase(_planted_switches(120, r["field_period_s"],
                                              0.001 - planted))
    out = ca.align_by_switches(video, audio, r["drum_period_s"])
    assert out["offset_s"] == pytest.approx(planted, abs=1e-12)
    assert out["ambiguity_s"] == pytest.approx(r["drum_period_s"])


def test_the_offset_is_wrapped_into_one_cycle_because_it_has_to_be():
    r = ca.rates()
    beyond = r["drum_period_s"] * 1.25
    video = ca.switch_phase(_planted_switches(60, r["field_period_s"], 0.0))
    audio = ca.switch_phase(_planted_switches(60, r["field_period_s"], -beyond))
    out = ca.align_by_switches(video, audio, r["drum_period_s"])
    assert abs(out["offset_s"]) <= r["drum_period_s"] / 2 + 1e-12
    # and it differs from the planted value by a whole number of cycles
    turns = (beyond - out["offset_s"]) / r["drum_period_s"]
    assert turns == pytest.approx(round(turns), abs=1e-9)


def test_a_non_positive_cycle_is_refused():
    r = ca.rates()
    fit = ca.switch_phase(_planted_switches(10, r["field_period_s"]))
    with pytest.raises(ValueError):
        ca.align_by_switches(fit, fit, 0.0)


# --------------------------------------------------------------------------
# why the difference beats either capture alone
# --------------------------------------------------------------------------

def test_drum_jitter_cancels_at_a_whole_revolution_and_doubles_at_a_half():
    r = ca.rates()
    assert float(ca.jitter_survival(r["drum_rate_hz"],
                                    r["drum_period_s"])) < 1e-12
    assert float(ca.jitter_survival(r["drum_rate_hz"],
                                    r["drum_period_s"] / 2)) == pytest.approx(2.0)
    # and it goes to zero as the lead does, for any disturbance
    assert float(ca.jitter_survival(5.0, 0.0)) == pytest.approx(0.0)


def test_the_drum_sits_exactly_at_the_nyquist_of_once_a_field_sampling():
    """Which is why drum wow is a per-head constant and not scatter."""
    out = ca.field_sampling_alias([ca.rates()["drum_rate_hz"]])
    assert out["drum_is_at_nyquist"]


def test_the_locator_term_falls_with_the_count_and_the_transport_term_does_not():
    common = dict(locator_sigma_video_s=200e-9, locator_sigma_audio_s=200e-9,
                  transport_jitter_rms_s=2.1e-6,
                  transport_jitter_hz=ca.rates()["drum_rate_hz"],
                  lead_s=ca.rates()["drum_period_s"] / 2)
    few = ca.offset_precision(count=25, **common)
    many = ca.offset_precision(count=2500, **common)
    assert many["locator_term_s"] == pytest.approx(few["locator_term_s"] / 10.0)
    assert many["transport_term_s"] == pytest.approx(few["transport_term_s"])
    # and with the lead at a whole revolution the transport term vanishes
    cancelled = ca.offset_precision(count=25, **{**common,
                                                 "lead_s": ca.rates()["drum_period_s"]})
    assert cancelled["transport_term_s"] < 1e-15


def test_a_speed_error_becomes_a_displacement_by_dividing_by_the_frequency():
    """The time base is the integral of the speed error, so a slow
    disturbance costs far more displacement than a fast one."""
    slow = ca.timing_from_speed_error(4e-4, 1.0)
    fast = ca.timing_from_speed_error(4e-4, 100.0)
    assert slow == pytest.approx(fast * 100.0)
    with pytest.raises(ValueError):
        ca.timing_from_speed_error(4e-4, 0.0)


def test_the_rate_bound_falls_as_the_duration_to_the_three_halves():
    short = ca.rate_precision(100e-9, 1.0)
    long = ca.rate_precision(100e-9, 4.0)
    assert short["fractional_rate_sigma"] / long["fractional_rate_sigma"] == \
        pytest.approx(4.0 ** 1.5, rel=1e-9)


def test_events_for_precision_inverts_the_locator_term():
    want = 20e-9
    out = ca.events_for_precision(want, 200e-9, 200e-9)
    back = ca.offset_precision(200e-9, 200e-9, int(round(out["events"])))
    assert back["locator_term_s"] == pytest.approx(want, rel=1e-3)


# --------------------------------------------------------------------------
# the tape, crossed twice
# --------------------------------------------------------------------------

def test_a_dropout_is_between_two_and_three_and_a_half_times_weaker_in_audio():
    """Wallace's law at the two bands. This is the mechanism that makes
    Hi-Fi audio survive dropouts that wipe the picture, and it is a hard
    limit on the dropout method."""
    out = ca.spacing_loss_ratio()
    low, high = out["range"]
    assert low == pytest.approx(2.0, rel=1e-9)
    assert high == pytest.approx(4.4e6 / 1.3e6, rel=1e-9)
    # the ratio IS the wavelength ratio, not an independent fit
    lam = out["wavelengths_m"]
    assert out["video_over_audio_db"]["sync_tip / channel_1"] == pytest.approx(
        lam["hifi"]["channel_1"] / lam["luma"]["sync_tip"])


def test_the_same_separation_gives_both_depths_and_inverts_cleanly():
    out = ca.dropout_visibility([20.0])
    assert float(out["audio_depth_db"][0]) == pytest.approx(
        20.0 / (4.4615384615384614 / 1.7058823529411764), rel=1e-9)
    # a tenth of a micrometre of dirt is the scale, not a millimetre
    assert 0.1 < float(out["separation_um"][0]) < 10.0


def test_the_dropout_timing_law_is_the_edge_over_the_signal_to_noise():
    out = ca.dropout_timing_precision(200e-9, 20.0)
    assert out["timing_sigma_s"] == pytest.approx(10e-9)
    with pytest.raises(ValueError):
        ca.dropout_timing_precision(0.0, 20.0)


def test_two_unknowns_close_and_a_lead_outside_the_standard_is_rejected():
    r = ca.rates()
    good = ca.separate_skew_from_lead(switch_offset_s=0.012,
                                      dropout_lead_s=0.010)
    assert good["capture_skew_s"] == pytest.approx(0.002)
    assert good["lead_inside_the_permitted_window"]
    bad = ca.separate_skew_from_lead(switch_offset_s=0.012,
                                     dropout_lead_s=r["drum_period_s"] * 2)
    assert not bad["lead_inside_the_permitted_window"]
    assert "outside" in bad["verdict"]


# --------------------------------------------------------------------------
# what the alignment buys the fold
# --------------------------------------------------------------------------

def test_the_phase_a_misalignment_costs_is_linear_in_it():
    one = ca.resolution_from_alignment(10e-9)
    ten = ca.resolution_from_alignment(100e-9)
    for band, value in one["phase_rad"].items():
        assert ten["phase_rad"][band] == pytest.approx(value * 10.0)


def test_a_time_base_probe_needs_the_alignment_finer_than_the_error_it_probes():
    out = ca.resolution_from_alignment(100e-9, time_base_error=4e-4)
    assert out["fraction_of_the_time_base_error"] < 1.0
    coarse = ca.resolution_from_alignment(100e-6, time_base_error=4e-4)
    assert coarse["fraction_of_the_time_base_error"] > 1.0


def test_the_sixth_axis_widens_the_band_by_more_than_four_times():
    """This arc's identifiability ceiling says only fractional bandwidth
    moves it. This is how much the audio tap moves it."""
    out = ca.fold_axis()
    assert out["axis"] == "capture"
    assert out["frequency_ratio_video_only"] == pytest.approx(4.4 / 3.4)
    assert out["frequency_ratio_with_audio"] == pytest.approx(4.4 / 1.3)
    assert out["decades_gained"] > 4.0
    # four depth clusters where the video alone has two
    assert out["depth_cluster_count"] == 5
    assert out["proposed_axis_stage"] in interference.full_chain()
    assert "assumption" in out["axis_stage_status"]


def test_the_depths_are_taken_from_the_module_that_owns_them():
    out = ca.fold_axis()
    low, high = hifi_carriers.read_depth_m()
    assert out["depth_clusters_m"]["audio_channel_1"] == pytest.approx(low)
    assert out["depth_clusters_m"]["audio_channel_2"] == pytest.approx(high)


# --------------------------------------------------------------------------
# the carrier route, restated closed
# --------------------------------------------------------------------------

def test_the_records_that_cannot_carry_a_carrier_are_excluded_by_name():
    out = ca.carrier_route_bound()
    assert set(out["withdrawn"]) == {"countdown", "home"}
    assert set(out["subjects"]) == {"zaroff SP playback", "zaroff EP playback"}
    assert out["control"] == ["zaroff SP record (control)"]
    assert out["pairs"] == 4


def test_no_narrow_line_on_any_record_that_could_carry_one():
    out = ca.carrier_route_bound()
    low, high = out["line_sigma_range"]
    assert high < out["detection_threshold_sigma"]
    assert not out["detected"]


def test_the_restated_bound_is_weaker_than_the_withdrawn_one():
    """The correction must not be allowed to look like an improvement. The
    -50 dB came from records with no Hi-Fi audio; what survives is worse."""
    out = ca.carrier_route_bound()
    assert out["conservative_bound_db"][1] > -50.0
    assert out["conservative_bound_db"][2] > out["conservative_bound_db"][1]
    assert "-50 dB" in ca.WITHDRAWN["claim"]
    assert "no hifi" in ca.WITHDRAWN["withdrawn_because"].lower() or \
        "no Hi-Fi audio" in ca.WITHDRAWN["withdrawn_because"]


def test_the_withdrawal_stays_visible_with_its_reason():
    for field in ("claim", "withdrawn_because", "what_survives", "restated_by"):
        assert ca.WITHDRAWN[field]
    for name, entry in ca.CARRIER_BEARING_RECORDS.items():
        assert entry["why"], name
        if not entry["carries_hifi"]:
            assert entry["role"].startswith("WITHDRAWN")


def test_the_bound_is_read_from_the_measurement_and_not_copied():
    """If `hifi_carriers.MEASURED` changes, the restatement must follow it
    rather than drift from it."""
    out = ca.carrier_route_bound()
    rows = hifi_carriers.MEASURED["results"]["zaroff SP playback"]
    assert out["line_sigma_range"][0] <= rows[1][0] <= out["line_sigma_range"][1]


# --------------------------------------------------------------------------
# and if it were ever found, it is a component like any other
# --------------------------------------------------------------------------

def test_the_carrier_signature_is_complex_and_subtractable():
    f = np.linspace(1.0e6, 2.0e6, 256)
    sig = ca.afm_carrier_signature(f, channel=1)
    assert sig.dtype == np.complex128
    assert interference.subtractable(sig)
    # it peaks at the specified centre and nowhere else
    assert f[int(np.argmax(np.abs(sig)))] == pytest.approx(1.3e6, abs=5e3)


def test_a_silent_carrier_is_narrower_than_a_modulated_one():
    """And its width comes from the transport, because the standard states
    no width for a carrier nothing is modulating."""
    f = np.linspace(1.0e6, 1.6e6, 4096)
    narrow = ca.silent_carrier_width_hz(4e-4, 1)
    assert narrow == pytest.approx(4e-4 * 1.3e6)
    silent = np.abs(ca.afm_carrier_signature(f, 1, width_hz=narrow))
    loud = np.abs(ca.afm_carrier_signature(f, 1))
    assert (loud - 1.0).sum() > (silent - 1.0).sum()
    # the default is the only width the standard states
    assert ca.afm_carrier_signature(f, 1)[0] == pytest.approx(
        ca.afm_carrier_signature(
            f, 1, width_hz=hifi_carriers.reference_half_width_hz())[0])


def test_a_line_of_no_width_is_refused():
    with pytest.raises(ValueError):
        ca.afm_carrier_signature(np.linspace(1e6, 2e6, 8), 1, width_hz=0.0)


def test_only_the_two_specified_channels_exist():
    with pytest.raises(ValueError):
        ca.afm_carrier_signature(np.linspace(1e6, 2e6, 8), channel=3)


def test_the_entry_composes_into_the_chain_without_ambiguity():
    chain = interference.full_chain(strict=True)
    assert chain["afm carrier crosstalk"] == ca.COMPONENT_ORDER[
        "afm carrier crosstalk"]
    # it is entered as an additive term at the head, not at the capture
    assert chain["afm carrier crosstalk"] == chain["particle noise"]
    assert chain["afm carrier crosstalk"] < chain["beat / co-channel"]


def test_the_entry_is_refused_on_present_evidence():
    route = ca.admission_route()
    assert route["present_verdict"] == "refused for want of a detection"
    assert route["judge"].startswith("interference.admission")


# --------------------------------------------------------------------------
# what would lower the bound, and what is refused
# --------------------------------------------------------------------------

def test_correcting_the_time_base_is_worth_more_than_the_capture_is_long():
    out = ca.bound_lowering(capture_duration_s=0.480, time_base_error=4e-4)
    assert out["coherence_gain_db"] > 20.0
    # the smear is many bins wide, which is the whole argument
    assert out["smear_hz"] / out["transform_bin_hz"] > 100.0


def test_a_still_time_base_leaves_nothing_to_concentrate():
    out = ca.bound_lowering(time_base_error=0.0)
    assert out["coherence_gain_db"] == pytest.approx(0.0)


def test_every_way_of_lowering_the_bound_names_its_condition():
    for name, gain, why in ca.bound_lowering()["ranked"]:
        assert name and why
        if gain is None:
            assert "only if" in why.lower() or "not" in why.lower() or \
                "removes" in why.lower()


def test_each_refusal_names_what_would_lift_it():
    out = ca.refusals()
    assert len(out) >= 5
    for row in out:
        assert row["refused"] and row["because"] and row["lifted_by"]


def test_the_proving_capture_names_both_taps_and_the_shared_clock():
    out = ca.the_capture_that_would_prove_it()
    assert "CN261 pin 2" in out["channel_a"]
    assert "CN341 pin 3" in out["channel_b"]
    assert "clock" in out["clock"]
    assert out["duration_s"] > 1.0


# --------------------------------------------------------------------------
# the tape-locked recurrence lag, which is what makes method two work
# --------------------------------------------------------------------------

def test_a_tape_mark_and_a_deck_mark_recur_at_different_lags():
    """MEASURED: the pair-lag histogram is a single clean peak at the tape
    lag with nothing at the field period - home 159 coincidences against a
    uniform-random null of 0.20 +- 0.46."""
    out = ca.tape_recurrence_lag("SP")
    assert out["tape_locked_lag_s"] > out["deck_locked_lag_s"]
    assert out["discriminator_s"] == pytest.approx(96.4e-6, abs=1e-6)
    # and it is more than a line wide, so it is not a subtle separation
    assert out["discriminator_in_lines"] > 1.0


def test_the_discriminator_scales_with_the_tape_speed_and_the_angle_does_not():
    """The heads sweep the same path whatever the tape does, so a slower
    tape lays closer tracks at the SAME angle. An angle that moved with the
    speed would mean the pitch had not been scaled with it."""
    angles = {speed: ca.tape_recurrence_lag(speed)["track_angle_deg"]
              for speed in ("SP", "LP", "EP")}
    assert angles["SP"] == pytest.approx(angles["EP"], abs=1e-9)
    sp = ca.tape_recurrence_lag("SP")["discriminator_s"]
    ep = ca.tape_recurrence_lag("EP")["discriminator_s"]
    # EP tape runs at 11.12 mm/s against SP's 33.35, so the lag is a third
    assert ep / sp == pytest.approx(11.12 / 33.35, rel=1e-6)
    # and this is the EP prediction the measurement confirmed at +32.5 us
    assert ep == pytest.approx(32.16e-6, abs=0.2e-6)


def test_the_derived_track_angle_agrees_with_table_two():
    out = ca.tape_recurrence_lag("SP")
    assert out["track_angle_deg"] == pytest.approx(
        out["table_2_track_angle_deg"], abs=0.02)


def test_an_unspecified_tape_speed_is_refused():
    with pytest.raises(ValueError):
        ca.tape_recurrence_lag("SLP")


def test_the_measured_coincidence_beats_its_null_by_hundreds_of_sigma():
    """The load-bearing measurement: a defect IS a mark shared between two
    head passes. Video head against video head - the audio case is
    refused, and `refusals` says so."""
    co = ca.MEASURED["coincidence"]
    for tape in ("home", "countdown"):
        row = co[tape]
        assert row["observed"] > row["null_max_of_500_draws"] * 10
        assert row["z"] > 100.0
        assert 0.25 < row["recurrence_fraction"] < 0.45
    assert "not at" in co["at_the_field_period_instead"]


def test_the_mark_precision_is_physical_and_not_measurement_noise():
    mark = ca.MEASURED["mark_precision"]
    floor_low, floor_high = mark["noise_floor_s"]
    for tape, sigma in mark["robust_sigma_s"].items():
        assert sigma > floor_high * 10, tape
    # and the per-head term is a deck constant, agreeing between two tapes
    a, b = mark["per_head_systematic_s"].values()
    assert abs(a - b) < 0.1e-6


def test_the_deck_locked_population_is_recorded_as_the_trap_it_is():
    pop = ca.MEASURED["deck_locked_population"]
    assert pop["home"] > 0.8
    assert pop["sp75"] == 0.0
    assert "not a tape defect" in pop["what_it_is"]


def test_pooling_marks_reaches_the_sub_microsecond():
    """1.5 us a mark at 0.65 to 2.65 marks a second."""
    mark = ca.MEASURED["mark_precision"]
    per_mark = max(mark["robust_sigma_s"].values())
    rate = min(mark["yield_per_second"].values())
    pooled = per_mark / math.sqrt(rate * 10.0)
    assert pooled < 1e-6


# --------------------------------------------------------------------------
# the deck's switching topology, read from its schematic
# --------------------------------------------------------------------------

def test_both_switches_come_from_one_controller_on_one_tachometer():
    """This is what turns method one's central assumption into a reading.
    RF SWP and AF SWP are pins 18 and 19 of IC160; the machine holds one
    rotation sensor and it reaches only that controller."""
    top = ca.SWITCHING_TOPOLOGY
    assert "IC160" in top["video_switch"]["origin"]
    assert "IC160" in top["audio_switch"]["origin"]
    assert top["tachometer"]["count"] == 1
    assert top["tachometer"]["reaches_the_afm_section"] is False
    assert "16 MHz" in top["controller_time_base"]["X160"]


def test_the_afm_path_really_does_switch_heads_and_by_its_own_signal():
    top = ca.SWITCHING_TOPOLOGY
    assert "changeover switch" in top["afm_head_switch_is_real"]
    assert "AF SWP and NOT RF SWP" in top["driven_by"]


def test_the_firmware_constant_is_labelled_an_inference():
    """The schematic does not say the offset is constant; the topology
    implies it. The distinction has to survive in the record."""
    assert "inference" in ca.SWITCHING_TOPOLOGY["inference"].lower()
    assert "does not state it" in ca.SWITCHING_TOPOLOGY["inference"]


def test_the_polarity_caution_is_carried_rather_than_assumed_away():
    assert "overbar" in ca.SWITCHING_TOPOLOGY["polarity_caution"].lower()


def test_what_could_not_be_read_is_named():
    assert ca.SWITCHING_TOPOLOGY["not_legible"]


def test_the_bench_measurement_needs_no_capture_at_all():
    out = ca.switch_offset_on_the_bench()
    assert "CN261 pin 3" in out["channel_a"]
    assert "JL345" in out["channel_b"]
    assert "any radio frequency" in out["does_not_need"]
    assert "the capture skew" in out["does_not_measure"]
    assert out["caution"] == ca.SWITCHING_TOPOLOGY["polarity_caution"]


def test_the_proving_capture_now_names_the_free_bench_check_first():
    out = ca.the_capture_that_would_prove_it()
    assert "JL345" in out["also_capture"]
    assert "oscilloscope" in out["do_first_because_it_is_free"]


def test_the_entry_orders_alongside_the_real_key_and_not_beside_it():
    """The 'same door' claim, exercised on the module's own machinery
    rather than asserted: the entry must be orderable in a key containing
    real components, and it must sit where its position says."""
    f = np.linspace(2.0e6, 5.0e6, 256)
    entries = {
        "afm carrier crosstalk": ca.afm_carrier_signature(f, 1),
        "particle noise": interference.particle_noise(f, 5.8, 58e-6),
        "beat / co-channel": interference.beat(f, 3.0e6),
    }
    ordered = interference.ordered_key(entries, interference.full_chain())
    positions = dict((name, place) for place, name in ordered)
    assert positions["afm carrier crosstalk"] == positions["particle noise"]
    assert positions["beat / co-channel"] > positions["afm carrier crosstalk"]
    for name, signature in entries.items():
        assert interference.subtractable(signature), name


# --------------------------------------------------------------------------
# the audio switch is a phase step, and the guide corroborates the schematic
# --------------------------------------------------------------------------

def test_the_guide_corroborates_the_schematic_at_the_format_level():
    """One deck's schematic is one deck. The manufacturer's guide makes it
    the format: the audio channel switcher is drum-derived everywhere."""
    jvc = ca.JVC_CORROBORATION
    assert "drum flipflop" in jvc["channel_switcher"]
    assert "phase of the carrier changes" in jvc["the_marker"]
    assert "blanking is absent" in jvc["no_blanking"]


def test_both_normative_sources_decline_to_fix_the_offset():
    """Which is what makes `offset_is_not_specified` a conclusion about the
    format rather than about one document."""
    assert "NOTHING about its angular position" in \
        ca.JVC_CORROBORATION["silent_on_the_offset"]
    assert ca.offset_is_not_specified()["permitted_angle_deg"] == 360.0


def test_the_audio_switch_marker_is_a_phase_and_not_a_magnitude():
    out = ca.afm_switch_marker()
    assert "phase" in out["marker"]
    # 1.3 MHz sampled at 40 MSps is about 31 samples a cycle
    assert out["samples_per_carrier_cycle"] == pytest.approx(40e6 / 1.3e6)


def test_a_precision_is_refused_without_a_measured_ratio():
    """There is no audio capture here, so there is no ratio to use, and a
    figure quoted anyway would be invented."""
    out = ca.afm_switch_marker()
    assert out["timing_sigma_s"] is None
    assert "invented" in out["refused"]


def test_the_marker_is_floored_at_the_capture_grid():
    coarse = ca.afm_switch_marker(amplitude_snr=10.0, sample_rate_hz=40e6)
    fine = ca.afm_switch_marker(amplitude_snr=100.0, sample_rate_hz=40e6)
    assert coarse["limited_by"] == "the carrier and the noise"
    assert fine["limited_by"] == "the capture grid"
    assert fine["timing_sigma_s"] == pytest.approx(1.0 / 40e6)
    with pytest.raises(ValueError):
        ca.afm_switch_marker(amplitude_snr=0.0)


def test_channel_two_gives_the_sharper_marker():
    """Its carrier is higher, so its cycle is shorter."""
    one = ca.afm_switch_marker(channel=1, amplitude_snr=10.0)
    two = ca.afm_switch_marker(channel=2, amplitude_snr=10.0)
    assert two["timing_sigma_s"] < one["timing_sigma_s"]


# --------------------------------------------------------------------------
# the helpers that exist so that one number has one owner
# --------------------------------------------------------------------------

def test_the_subcarrier_and_the_depth_divisor_are_borrowed_not_typed():
    from vhsdecode.models import colour_under, magnetic
    assert ca.colour_subcarrier_hz("NTSC") == pytest.approx(
        colour_under.subcarrier_hz("NTSC"))
    assert ca.magnetic_circuit_divisor() == pytest.approx(
        magnetic.DEMAGNETISATION_DIVISOR)


def test_the_deck_named_is_the_one_that_made_the_captures():
    assert ca.DECK == "SLV-778HF"
    assert ca.DECK in ca.SWITCHING_TOPOLOGY["deck"]


def test_the_wide_sweep_records_what_the_narrow_line_statistic_does_wrong():
    """Ten playback pairs cross 3 sigma and no control pair does, yet the
    control shows larger bumps. The statistic reads surround smoothness."""
    w = ca.WIDE_SWEEP
    assert w["line_sigma_over_threshold_playback"] > 0
    assert w["line_sigma_over_threshold_record_control"] == 0
    assert "surround smoothness" in w["why_the_ten_are_not_detections"]
    assert "not the colour-under second harmonic" in \
        w["second_harmonic_verdict"].lower()


def test_the_wide_sweep_agrees_with_the_bound_read_from_the_measurement():
    """Two independent routes to the same conservative figure: one reads
    `hifi_carriers.MEASURED`, the other records the wider re-run."""
    assert ca.WIDE_SWEEP["conservative_bound_db"] == \
        ca.carrier_route_bound()["conservative_bound_db"]


def test_removing_the_colour_under_barely_moved_its_supposed_harmonic():
    w = ca.WIDE_SWEEP
    assert w["colour_under_fundamental_removed_mean_db"] < -15.0
    assert abs(w["second_harmonic_removed_mean_db"]) < 1.0


def test_the_new_axis_doubles_the_cube_and_the_claim_tracks_the_fold():
    """The docstring says 32 vertices become 64. That is only true while
    the fold has five axes, so the claim is checked against the fold
    itself rather than left as a number in prose."""
    from vhsdecode.models import tesseract
    existing = len(tesseract.AXIS_STAGE)
    assert 2 ** existing == 32
    assert 2 ** (existing + 1) == 64
    assert ca.fold_axis()["axis"] not in tesseract.AXIS_STAGE


# --------------------------------------------------------------------------
# the drum-phase marker, measured
# --------------------------------------------------------------------------

def test_the_envelope_drum_line_is_recorded_as_rejected():
    """It is found easily and is still not a clock, and the reason is
    structural: the line is a square wave fitted as a sinusoid."""
    line = ca.MEASURED["drum_phase"]["envelope_line_rejected"]
    lo, hi = line["split_half_disagreement_s"]
    analytic_hi = line["analytic_sigma_s"]["0.48 s"][1]
    # the honest reproducibility is far worse than the analytic error
    assert lo > analytic_hi / 2
    assert "not a competitive" in line["verdict"]
    assert "square wave" in line["why"]


def test_the_switch_beats_the_envelope_line_by_two_orders():
    drum = ca.MEASURED["drum_phase"]
    best_switch = min(drum["playback_switch_residual_s"].values())
    worst_line = min(drum["envelope_line_rejected"]["analytic_sigma_s"]["0.48 s"])
    assert best_switch * 100 < worst_line


def test_the_switch_locator_sits_near_its_own_instrument_floor():
    drum = ca.MEASURED["drum_phase"]
    floor_lo, floor_hi = drum["instrument_floor_s"]
    assert drum["playback_switch_residual_s"]["zaroff-75bars SP"] < floor_hi * 3


def test_the_two_consumer_tapes_are_worse_because_of_where_the_switch_lands():
    """SMPTE 32M 3.6 permits 5 to 8 lines ahead of vertical sync. On both
    consumer tapes the playback switch lands ON it, where there is nothing
    to see a step against."""
    drum = ca.MEASURED["drum_phase"]
    pos = drum["switch_position_lines_ahead_of_vsync"]
    res = drum["playback_switch_residual_s"]
    assert pos["zaroff-75bars SP"] > 5.0
    assert abs(pos["home"]) < 2.0 and abs(pos["countdown"]) < 2.0
    assert res["zaroff-75bars SP"] < res["home"] < res["countdown"]
    permitted = ca.SPECIFICATION["head_switch_ahead_of_vsync_lines"]["value"]
    assert permitted[0] <= pos["zaroff-75bars SP"] <= permitted[1]


def test_the_drum_is_steadier_than_the_tape_which_is_the_whole_premise():
    drum = ca.MEASURED["drum_phase"]
    assert "STEADIER IN ABSOLUTE TIME THAN THE VERTICAL SYNC" in \
        drum["premise_confirmed"]
    # and on the deck's own tape the two agree to the instrument floor
    floor_hi = drum["instrument_floor_s"][1]
    assert drum["switch_minus_vsync_rms_s"]["zaroff-75bars SP"] < floor_hi


def test_no_coherent_transport_term_was_resolved():
    """Which is why `offset_precision`'s second term carries no measured
    value: the scatter averages down with no floor in sight."""
    wander = ca.MEASURED["drum_phase"]["drum_wander"]
    lo, hi = wander["running_mean_over_white_ratio"]
    assert 0.8 < lo and hi < 1.5, "a ratio near one is a white residual"
    assert "white" in wander["verdict"]


def test_the_record_mark_is_better_located_and_is_the_wrong_marker():
    drum = ca.MEASURED["drum_phase"]
    assert drum["record_mark_residual_s"]["zaroff-75bars SP"] < \
        drum["playback_switch_residual_s"]["zaroff-75bars SP"]
    assert "only THROUGH THE TAPE" in drum["record_mark_is_not_the_drum"]


def test_thirty_seconds_of_the_decks_own_tape_reaches_the_ten_nanosecond_aim():
    """The figure section 6.1 asks for, from the measured per-event
    scatter. The audio side is a PREDICTION - there is no audio capture."""
    sigma = ca.MEASURED["drum_phase"]["playback_switch_residual_s"][
        "zaroff-75bars SP"]
    events = int(ca.rates()["field_rate_hz"] * 30.0)
    both = ca.offset_precision(sigma, sigma, events)["offset_sigma_s"]
    logic = ca.offset_precision(sigma, 0.0, events)["offset_sigma_s"]
    assert both < 13e-9 and logic < 9e-9
    assert ca.resolution_from_alignment(both)["phase_rad"][
        "luma_peak_white"] < 0.5
