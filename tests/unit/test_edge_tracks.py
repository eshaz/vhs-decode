"""The linear audio and control tracks, as seen by the video head.

Ethan asked whether the edge tracks overlap the video sweep enough to be
read. The arithmetic that answers him is short enough to be checked here
line by line, and it MUST be checked here, because the whole conclusion
rests on one number that the standard never prints: the 150 um between the
head's sweep and each edge track, which exists only by subtraction.

The estimator is checked the way this arc checks estimators - against a
planted signal, and against the same signal with one of its three
signatures broken.
"""

import math

import numpy as np
import pytest

from vhsdecode.models import edge_tracks as et


# --------------------------------------------------------------------------
# The geometry, which is the finding
# --------------------------------------------------------------------------


def test_the_transverse_layout_closes():
    """Control track, free span and audio track must fill the tape exactly."""
    place = et.layout("SP")
    control_top = place["control_track_m"][1]
    audio_bottom = place["audio_track_m"][0]
    assert place["audio_track_m"][1] == pytest.approx(place["tape_width_m"])
    assert control_top == pytest.approx(0.75e-3)
    assert audio_bottom == pytest.approx(11.65e-3)


def test_the_video_centre_is_the_midpoint_of_the_free_span():
    """The check that decides which edge carries which track.

    SMPTE 32M puts the video track centre at 6.20 mm and the free span
    between the edge tracks runs 0.75 to 11.65 mm, whose midpoint is 6.20
    exactly. Put the audio at the reference edge instead and the midpoint
    would be 6.45. So this equality is the standard's own arithmetic saying
    the control track is at the reference edge.
    """
    place = et.layout("SP")
    assert place["free_span_midpoint_m"] == pytest.approx(6.20e-3, abs=1e-9)
    centre = sum(place["sweep_m"]) / 2.0
    assert centre == pytest.approx(place["free_span_midpoint_m"], abs=1e-9)


def test_there_is_no_overlap_and_the_guard_is_the_same_at_both_ends():
    place = et.layout("SP")
    assert place["guard_control_m"] == pytest.approx(150e-6, abs=1e-9)
    assert place["guard_audio_m"] == pytest.approx(150e-6, abs=1e-9)
    assert place["sweep_m"][0] > place["control_track_m"][1]
    assert place["sweep_m"][1] < place["audio_track_m"][0]


def test_the_stated_tolerances_cannot_close_the_guard():
    """Worst case in every direction the guard shrinks to 100 um, not zero.

    The control track's width carries +- 0.10 mm and the audio reference
    line +- 0.05 mm; the video recording area, the track centre and the
    tape width carry either nothing or a hundredth of a millimetre. Stack
    them all against the guard and it survives, which is why the negative is
    a geometric statement and not a nominal-dimensions caveat.
    """
    control = et.TRACK_GEOMETRY["control_track_width_m"]
    audio = et.TRACK_GEOMETRY["audio_track_reference_line_m"]
    tape = et.TRACK_GEOMETRY["tape_width_m"]
    place = et.layout("SP")
    worst_control = place["guard_control_m"] - control["tolerance"]
    worst_audio = place["guard_audio_m"] - audio["tolerance"] - tape["tolerance"]
    assert worst_control > 0.0
    assert worst_audio > 0.0
    assert worst_control == pytest.approx(50e-6, abs=1e-9)


def test_one_field_of_scan_is_the_effective_area():
    """W over the field's lines is the transverse rate, and W at the track
    angle is the 96.8 mm track a 5.80 m/s head writes in 16.683 ms."""
    assert et.transverse_rate_m_per_line("NTSC") == pytest.approx(38.362e-6,
                                                                  rel=1e-4)
    effective = et.TRACK_GEOMETRY["video_effective_area_m"]["value"]
    angle = et.SPEED_GEOMETRY["SP"]["track_angle_rad"]
    length = effective / math.sin(angle)
    field_s = 262.5 * 1001.0 / (525 * 30000.0)
    assert length == pytest.approx(5.80 * field_s, rel=2e-3)


def test_the_overlap_is_the_difference_between_the_two_areas():
    place = et.layout("SP")
    assert place["overlap_m"] == pytest.approx(0.53e-3, abs=1e-9)
    assert et.visible_window()["overlap_lines"] == pytest.approx(13.82, rel=1e-3)


# --------------------------------------------------------------------------
# The speed ratio and what it does to a frequency
# --------------------------------------------------------------------------


def test_the_speed_ratio_uses_the_longitudinal_component():
    """172.970, not the 174 a reader gets by dropping the cosine."""
    ratio = et.speed_ratio("SP")
    assert ratio == pytest.approx(5.80 * math.cos(
        et.SPEED_GEOMETRY["SP"]["track_angle_rad"]) / 33.35e-3, rel=1e-12)
    assert ratio == pytest.approx(172.970, rel=1e-4)
    assert ratio < 5.80 / 33.35e-3          # the cosine can only reduce it


def test_the_control_track_returns_at_the_predicted_frequency():
    assert et.control_track_reproduce_hz("SP") == pytest.approx(5183.9, rel=1e-4)
    assert et.reproduce_hz(1000.0, "SP") == pytest.approx(172970.0, rel=1e-4)


def test_the_tape_speed_moves_the_line_and_nothing_else_does():
    """THE CONTROL THAT DECIDES THE MEASUREMENT. EP writes the same one
    pulse per frame at a third of the tape speed, so the head reads it three
    times higher - while the line rate, the field rate and every other rate
    in the deck stay exactly where they are."""
    sp = et.control_track_reproduce_hz("SP")
    ep = et.control_track_reproduce_hz("EP")
    assert ep / sp == pytest.approx(3.015, rel=1e-3)
    assert ep == pytest.approx(15628.5, rel=1e-4)


def test_lp_is_refused_rather_than_guessed():
    """No document in this project's sources states an NTSC LP writing
    speed, so the module must decline rather than reuse SP's."""
    with pytest.raises(KeyError):
        et.control_track_reproduce_hz("LP")
    with pytest.raises(ValueError):
        et.speed_ratio("XP")


# --------------------------------------------------------------------------
# Fringing
# --------------------------------------------------------------------------


def test_the_decay_length_is_the_wavelength_over_two_pi():
    """The only length in the problem, and it must be exactly that."""
    f = et.control_track_reproduce_hz("SP")
    lam = et.apparent_wavelength_m(f, "SP")
    assert et.fringing_decay_length_m(f, "SP") == pytest.approx(lam / (2 * math.pi))
    assert et.fringing_decay_length_m(f, "SP") == pytest.approx(178.1e-6, rel=1e-3)


def test_the_fringing_factor_is_an_exponential_in_the_decay_length():
    f = et.control_track_reproduce_hz("SP")
    decay = et.fringing_decay_length_m(f, "SP")
    assert et.fringing_factor(decay, f, "SP") == pytest.approx(math.exp(-1.0))
    assert et.fringing_factor(0.0, f, "SP") == pytest.approx(1.0)
    guard = et.layout("SP")["guard_control_m"]
    assert et.fringing_factor(guard, f, "SP") == pytest.approx(0.431, rel=1e-2)
    # and it must fall with distance and with frequency, both
    assert et.fringing_factor(2 * guard, f) < et.fringing_factor(guard, f)
    assert et.fringing_factor(guard, 10 * f) < et.fringing_factor(guard, f)


def test_the_guard_admits_only_the_bass_of_the_audio_track():
    """The sentence the audio half of the idea reduces to."""
    admitted = et.admitted_tape_band_hz(floor_db=-60.0)
    assert admitted == pytest.approx(245.8, rel=1e-2)
    guard = et.layout("SP")["guard_audio_m"]
    at_the_edge = et.fringing_factor(guard, et.reproduce_hz(admitted))
    assert 20 * math.log10(at_the_edge) == pytest.approx(-60.0, abs=0.2)
    # a kilohertz tone is hundreds of dB down, which is a way of saying gone
    assert 20 * math.log10(
        et.fringing_factor(guard, et.reproduce_hz(1000.0))) < -200.0


def test_azimuth_is_named_in_the_premise_and_is_not_the_limit():
    """At every frequency the guard admits, the azimuth loss is nothing."""
    f = et.control_track_reproduce_hz("SP")
    assert et.azimuth_loss(f, "SP") == pytest.approx(1.0, abs=1e-4)
    # the first null is where the track width times tan(alpha) equals the
    # wavelength, far above anything that survives the guard
    width = 58e-6
    null_wavelength = width * math.tan(math.radians(6.0))
    null_hz = 5.80 / null_wavelength
    assert et.azimuth_loss(null_hz, "SP") == pytest.approx(0.0, abs=1e-9)
    assert null_hz > 100.0 * f


def test_the_upper_bound_is_the_sum_of_the_two_terms_it_can_compute():
    bound = et.geometric_upper_bound_db("SP")
    assert bound["total_db"] == pytest.approx(bound["differentiation_db"]
                                              + bound["guard_db"])
    assert bound["differentiation_db"] < -50.0
    assert -8.0 < bound["guard_db"] < -7.0
    assert -66.0 < bound["total_db"] < -63.0


# --------------------------------------------------------------------------
# Where the head is
# --------------------------------------------------------------------------


def test_the_head_climbs_away_from_the_control_track_after_the_switch():
    switch = -3.5
    near = et.distance_to_track_m(switch, switch, "control")
    far = et.distance_to_track_m(switch + 10.0, switch, "control")
    assert near == pytest.approx(150e-6, abs=1e-9)
    assert far > near
    assert far - near == pytest.approx(10.0 * et.transverse_rate_m_per_line(),
                                       rel=1e-9)


def test_the_leaving_head_is_a_different_geometry_from_the_entering_one():
    """Before the switch the tap shows the head that is LEAVING, and asking
    the entering branch about those lines clamps the distance at the guard
    for the whole picture - wrong by millimetres."""
    switch = -6.3
    guard = et.layout("SP")["guard_audio_m"]
    near = et.distance_to_track_m(switch, switch, "audio", head="leaving")
    far = et.distance_to_track_m(switch - 12.0, switch, "audio", head="leaving")
    assert near == pytest.approx(guard, abs=1e-9)
    assert far - near == pytest.approx(12.0 * et.transverse_rate_m_per_line(),
                                       rel=1e-9)
    # the entering branch clamps instead, which is the bug this guards
    assert et.distance_to_track_m(switch - 12.0, switch, "control",
                                  head="entering") == 0.0
    with pytest.raises(ValueError):
        et.sweep_position_m(0.0, 0.0, head="hovering")


def test_the_mid_picture_hump_cannot_be_an_edge_track():
    """The arithmetic that refused the one thing that looked like a find.

    A field-alternating component near 5183.9 Hz peaks at line -18 to -22.
    Even on the most favourable anchoring the leaving head is 0.449 mm past
    its edge by line -18, so a fringe there is 21.9 dB below its own value
    at the edge - which would put the edge value 27 dB ABOVE the generous
    geometric bound. It cannot be one.
    """
    frequency = et.control_track_reproduce_hz("SP")
    guard = et.layout("SP")["guard_audio_m"]
    at_the_edge = et.fringing_factor(guard, frequency)
    there = et.fringing_factor(
        et.distance_to_track_m(-18.0, -6.3, "audio", head="leaving"), frequency)
    penalty = 20 * math.log10(there / at_the_edge)
    assert penalty == pytest.approx(-21.9, abs=0.5)
    measured_there = -59.3
    implied_at_the_edge = measured_there - penalty
    assert implied_at_the_edge > et.geometric_upper_bound_db("SP")["total_db"] + 20.0


def test_the_closest_approach_is_refused_because_the_switch_hides_it():
    window = et.visible_window()
    assert window["refused"]
    assert "overlap" in window["refused"]
    assert window["capture_that_would_settle_it"]
    low, high = window["specified_switch_ahead_h"]
    assert low <= -window["measured_switch_lines"][0] <= high


# --------------------------------------------------------------------------
# The estimator, against a planted signal
# --------------------------------------------------------------------------


def _planted(rate_hz, fields, switch_line, frequency, amplitude,
             alternate=True, track="audio", seed=3, noise=1.0):
    """A synthetic capture carrying a fringe with all three signatures."""
    rng = np.random.default_rng(seed)
    line = et.sync_geometry.LINE_PERIOD_US["525"] * 1e-6 * rate_hz
    field = 262.5 * line
    n = int((fields + 2) * field)
    x = rng.normal(0.0, float(noise), n)
    starts = [int((k + 1) * field) for k in range(fields)]
    decay = et.fringing_decay_length_m(frequency, "SP")
    per_line = et.transverse_rate_m_per_line()
    guard = et.layout("SP")["guard_audio_m" if track == "audio"
                            else "guard_control_m"]
    for k, start in enumerate(starts):
        a = int((switch_line - 30.0) * line)
        b = int((switch_line + 30.0) * line)
        index = np.arange(a, b)
        lines = index / line
        if track == "audio":
            distance = guard + per_line * (switch_line - lines)
        else:
            distance = guard + per_line * (lines - switch_line)
        profile = np.exp(-(np.maximum(distance, 0.0) - guard) / decay)
        sign = (-1.0) ** k if alternate else 1.0
        wave = amplitude * sign * profile * np.cos(
            2 * np.pi * frequency * index / rate_hz + 0.7)
        x[start + a:start + b] += wave
    return x, starts


def test_the_estimator_finds_a_planted_fringe():
    rate, fields = 4e6, 40
    frequency = et.control_track_reproduce_hz("SP")
    reference = et.matched_reference(rate, -6.3 - 28.0, -6.3, -6.3,
                                     frequency, "audio")
    x, starts = _planted(rate, fields, -6.3, frequency, 0.25)
    found = et.project(x, starts, rate, reference, -6.3 - 28.0, 1.0)
    assert found.fields == fields
    assert found.ratio > 8.0
    # and the LEVEL is recovered, not merely the presence: twice the plant
    # must read twice the amplitude, which a detector that only sees energy
    # would fail
    twice, _ = _planted(rate, fields, -6.3, frequency, 0.50)
    louder = et.project(twice, starts, rate, reference, -6.3 - 28.0, 1.0)
    assert abs(louder.amplitude) / abs(found.amplitude) == pytest.approx(2.0,
                                                                        rel=0.1)


def test_the_reading_is_linear_in_the_planted_level():
    """The phase error is systematic, so doubling the plant must double the
    reading exactly even where the absolute level carries its few dB."""
    rate, fields = 4e6, 60
    frequency = et.control_track_reproduce_hz("SP")
    reference = et.matched_reference(rate, -34.3, -6.3, -6.3, frequency,
                                     "audio")
    quiet, starts = _planted(rate, fields, -6.3, frequency, 0.2, seed=5)
    loud, _ = _planted(rate, fields, -6.3, frequency, 0.4, seed=5)
    one = et.project(quiet, starts, rate, reference, -34.3, 0.3)
    two = et.project(loud, starts, rate, reference, -34.3, 0.3)
    assert two.dbc - one.dbc == pytest.approx(6.02, abs=0.3)


def test_breaking_the_parity_kills_it():
    """The plant with no field-to-field sign reversal must not survive the
    alternating average - that is what the parity signature is for."""
    rate, fields = 4e6, 40
    frequency = et.control_track_reproduce_hz("SP")
    x, starts = _planted(rate, fields, -6.3, frequency, 0.25, alternate=False)
    reference = et.matched_reference(rate, -6.3 - 28.0, -6.3, -6.3,
                                     frequency, "audio")
    alternating = et.project(x, starts, rate, reference, -6.3 - 28.0, 1.0,
                             alternate=True)
    straight = et.project(x, starts, rate, reference, -6.3 - 28.0, 1.0,
                          alternate=False)
    assert straight.ratio > 8.0
    assert alternating.ratio < straight.ratio / 3.0


def test_breaking_the_position_kills_it():
    """A fringe planted at the sweep end must not be found in mid-field."""
    rate, fields = 4e6, 40
    frequency = et.control_track_reproduce_hz("SP")
    x, starts = _planted(rate, fields, -6.3, frequency, 0.25)
    edge = et.project(x, starts, rate,
                      et.matched_reference(rate, -34.3, -6.3, -6.3, frequency,
                                           "audio"), -34.3, 1.0)
    middle = et.project(x, starts, rate,
                        et.matched_reference(rate, 92.0, 120.0, 120.0,
                                             frequency, "audio"), 92.0, 1.0)
    assert edge.ratio > 8.0
    assert middle.ratio < 4.0


def test_a_projection_needs_repeats():
    rate = 4e6
    x, starts = _planted(rate, 4, -6.3,
                         et.control_track_reproduce_hz("SP"), 0.25)
    reference = et.matched_reference(rate, -34.3, -6.3, -6.3,
                                     et.control_track_reproduce_hz("SP"),
                                     "audio")
    with pytest.raises(ValueError):
        et.project(x, starts[:2], rate, reference, -34.3, 1.0)


def test_the_reference_peaks_at_one_and_decays_at_the_predicted_rate():
    """UNNORMALISED, peak one at the tape edge - the profile's own norm is
    what turns a projection into a level, so it must not be divided away -
    and blind to a constant, so it cannot read an offset as a fringe."""
    rate = 4e6
    frequency = et.control_track_reproduce_hz("SP")
    reference = et.matched_reference(rate, -34.3, -6.3, -6.3, frequency,
                                     "audio")
    assert np.abs(reference).max() == pytest.approx(1.0, rel=0.02)
    assert abs(np.vdot(reference, np.ones(len(reference)))) < 1e-9

    # the decay law itself is arithmetic and is asserted exactly
    expected = 8.685889638 * et.transverse_rate_m_per_line() \
        / et.fringing_decay_length_m(frequency, "SP")
    assert expected == pytest.approx(1.871, rel=1e-3)

    # and the template carries it, to within the 1.7 per cent the constant's
    # removal costs the envelope: |m| is 0.104 of the profile's sum over the
    # window, which is the same 19.6 dB of DC rejection the removal bought
    line = et.sync_geometry.LINE_PERIOD_US["525"] * 1e-6 * rate
    envelope = np.abs(reference)
    step = 20 * np.log10(envelope[-1] / envelope[-1 - int(round(line))])
    assert step == pytest.approx(expected, rel=0.1)


def test_the_reference_refuses_an_empty_window_and_an_unknown_track():
    rate = 4e6
    frequency = et.control_track_reproduce_hz("SP")
    with pytest.raises(ValueError):
        et.matched_reference(rate, 5.0, 5.0, 0.0, frequency, "audio")
    with pytest.raises(ValueError):
        et.matched_reference(rate, -34.3, -6.3, -6.3, frequency, "hifi")


# --------------------------------------------------------------------------
# The trend fit, which is what turned an apparent detection into a bound
# --------------------------------------------------------------------------


def _sweep_from(levels):
    return {f: et.Projection(amplitude=complex(10 ** (db / 20.0), 0.0),
                             standard_error=1e-6, fields=30,
                             reference_rms=1.0, window_samples=1)
            for f, db in levels.items()}


def test_a_smooth_sweep_shows_no_excess():
    """The playback tap's own low-frequency noise is a straight line in log
    frequency, and a straight line must read as nothing."""
    predicted = 5183.9
    frequencies = [1500.0, 2500.0, 3500.0, 4500.0, predicted, 6000.0, 7500.0,
                   10000.0, 15734.264, 25000.0]
    levels = {f: -40.0 - 5.44 * math.log2(f / 1500.0) for f in frequencies}
    found = et.excess_over_trend(_sweep_from(levels), predicted)
    assert abs(found["excess_db"]) < 0.2
    assert found["trend_slope_db_per_octave"] == pytest.approx(-5.44, rel=1e-3)


def test_a_planted_bump_shows_up_as_an_excess():
    predicted = 5183.9
    frequencies = [1500.0, 2500.0, 3500.0, 4500.0, predicted, 6000.0, 7500.0,
                   10000.0, 15734.264, 25000.0]
    levels = {f: -40.0 - 5.44 * math.log2(f / 1500.0) for f in frequencies}
    levels[predicted] += 15.0
    found = et.excess_over_trend(_sweep_from(levels), predicted)
    assert found["excess_db"] == pytest.approx(15.0, abs=0.5)
    assert found["sigma"] > 10.0


def test_the_trend_needs_neighbours():
    with pytest.raises(ValueError):
        et.excess_over_trend(_sweep_from({5183.9: -50.0, 6000.0: -51.0}),
                             5183.9)


def test_a_projection_reports_its_own_bound():
    found = et.Projection(amplitude=complex(1e-6, 0.0), standard_error=1e-3,
                          fields=30, reference_rms=0.2, window_samples=10000,
                          template_norm=50.0)
    assert found.ratio < 0.01
    assert found.bound_dbc > found.dbc
    assert found.bound_dbc == pytest.approx(20 * math.log10(
        (2.0 * 2e-3 / 2500.0) / (math.sqrt(2.0) * 0.2)))


def test_the_calibration_recovers_the_planted_peak_amplitude():
    """The reading must be a LEVEL, not merely a detection: a fringe planted
    with a known peak amplitude against a carrier of known rms must come
    back at the right number of dB below it, whatever the frequency - which
    is exactly what a unit-norm projection gets wrong, and by a
    frequency-dependent amount."""
    rate, fields = 4e6, 60
    carrier_rms = 0.30
    for frequency, planted in ((et.control_track_reproduce_hz("SP"), 0.02),
                               (2.0 * et.control_track_reproduce_hz("SP"), 0.02),
                               (et.control_track_reproduce_hz("SP"), 0.005)):
        # a tenth of the noise, so this test measures the CALIBRATION and
        # not the floor - the floor has its own tests above
        x, starts = _planted(rate, fields, -6.3, frequency, planted, seed=11,
                             noise=0.05)
        reference = et.matched_reference(rate, -6.3 - 28.0, -6.3, -6.3,
                                         frequency, "audio")
        found = et.project(x, starts, rate, reference, -6.3 - 28.0, carrier_rms)
        expected = 20 * math.log10(planted / (math.sqrt(2.0) * carrier_rms))
        # +-3 dB, and the reason is measured rather than fudged: the profile
        # holds 1.53 cycles at the control frequency, so the projection's
        # negative-frequency image does not vanish and the reading depends
        # on the planted phase - 2.91 dB at worst over twelve phases, 1.90
        # dB rms. See `Projection`.
        assert found.dbc == pytest.approx(expected, abs=3.0)


# --------------------------------------------------------------------------
# What he actually asked for
# --------------------------------------------------------------------------


def test_the_audio_displacement_is_the_synchronisation_answer():
    found = et.audio_displacement("SP")
    assert found["distance_m"] == pytest.approx(79.244e-3)
    assert found["seconds"] == pytest.approx(2.376132, rel=1e-6)
    assert found["frames"] == pytest.approx(71.213, rel=1e-4)
    # its precision is the tape speed's, which the standard does state
    assert found["seconds_tolerance"] == pytest.approx(0.011881, rel=1e-4)


def test_the_synchronisation_verdict_names_which_route_survives():
    found = et.synchronisation_precision()
    assert found["fringe_route"]["available"] is False
    assert found["fringe_route"]["because"]
    assert found["switch_route"]["available"] is True
    assert found["switch_route"]["specified_speed_tolerance_seconds"] > \
        found["switch_route"]["measured_vsync_scatter_seconds"]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_every_geometry_entry_is_cited_or_declared_absent():
    for name, entry in et.TRACK_GEOMETRY.items():
        assert entry["quote"], name
        if entry["value"] is None:
            assert "NOT STATED" in entry["quote"], name
        else:
            assert entry["clause"], name


def test_the_measured_record_names_its_captures_and_controls():
    record = et.MEASURED
    assert record["verdict"]
    assert record["tool"].endswith("edge_tracks_measure.py")
    assert set(record["controls"]) >= {"position", "parity", "record tap",
                                       "mid-field", "tape speed"}
    assert record["reservation"]
    assert record["open_observation"]
    assert record["estimator_defects_found"]
    assert len(record["captures"]) >= 5
    # the position scan is the evidence, so it must be in the record and it
    # must say what it says: weakest at the edge, strongest in mid-picture
    scan = record["position_scan_sp"]
    assert scan[-2][0] < scan[-18][0] - 20.0
    assert scan[-2][0] < record["geometry"]["upper_bound_dbc_sp"] - 15.0
