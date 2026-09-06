"""The external reference, and what has to hold whether or not it is present.

The generator is a separate C++ program, so most environments will not have
it. Everything that does not need it is pinned unconditionally - the project
file's determinism, the level and rate tables against the standards, the
forward chain's arithmetic, the residual's own algebra - and only the four
tests that must actually run videosynth are skipped when it is absent.

The chain tests use PLANTED signals, whose answer is arithmetic rather than
measured: a flat field at one level with the emphasis off must modulate to
exactly one carrier frequency, and a reference differenced with itself must
give exactly zero. Neither can pass by accident.
"""

import os

import numpy as np
import pytest

from vhsdecode.models import output_limit, reference_signal as rs


AVAILABLE = rs.available()
needs_generator = pytest.mark.skipif(
    not AVAILABLE["present"],
    reason=f"videosynth is not available: {AVAILABLE['why']}")


# --------------------------------------------------------------------------
# The interface, with or without the generator
# --------------------------------------------------------------------------


def test_available_answers_rather_than_raising():
    """A caller - and a test - must be able to ask whether the generator is
    there without handling an exception, because it is a separate program and
    its absence is ordinary rather than exceptional."""
    tool = rs.available()
    assert set(tool) >= {"present", "binary", "version", "encodings",
                         "systems", "patterns", "why"}
    assert isinstance(tool["present"], bool)
    if tool["present"]:
        assert tool["version"]
        assert os.access(str(tool["binary"]), os.X_OK)
    else:
        assert tool["binary"] is None
        assert "videosynth" in str(tool["why"])


def test_the_project_file_is_a_pure_function_of_the_request():
    """The cache key is a hash of this text, so anything that varies between
    two identical requests - a timestamp, a path, a dictionary order - would
    make the same reference generate twice and compare unequal."""
    first = rs.project_text("NTSC", "75bars", "CVBS_U10_4FSC", frames=2)
    second = rs.project_text("NTSC", "75bars", "CVBS_U10_4FSC", frames=2)
    assert first == second
    assert rs.project_text("NTSC", "75bars", "RAW_S16_40M", 2) != first
    assert rs.project_text("NTSC", "100bars", "CVBS_U10_4FSC", 2) != first
    assert rs.project_text("NTSC", "75bars", "CVBS_U10_4FSC", 3) != first


def test_the_project_file_is_the_five_blocks_the_reference_names():
    """videosynth's own reference states exactly five top-level blocks and
    rejects any other key, so a project written here that is missing one, or
    carries a sixth, does not parse."""
    text = rs.project_text("NTSC", "75bars", "CVBS_U10_4FSC", frames=1)
    top = [line.split(":")[0] for line in text.splitlines()
           if line and not line[0].isspace()]
    assert top == ["project", "cvbs_presets", "output", "sections"]
    assert "video_standard_preset: NTSC" in text
    assert "sample_encoding_preset: CVBS_U10_4FSC" in text
    assert "signal_state_preset: STANDARD_STABLE_LOCKED" in text
    assert "720x486" in text          # the NTSC raster, not PAL's
    assert "duration_frames: 1" in text


def test_the_setup_key_is_dropped_on_pal():
    """`ntsc_black_setup_ire` is a System M key and the generator rejects it
    on a PAL project, so writing one there produces a file that will not
    validate."""
    assert "ntsc_black_setup_ire" in rs.project_text("NTSC", "75bars")
    assert "ntsc_black_setup_ire" not in rs.project_text("PAL", "75bars")
    assert "720x576" in rs.project_text("PAL", "75bars")


def test_an_unknown_request_is_refused_rather_than_guessed():
    for bad in (lambda: rs.project_text("SECAM", "75bars"),
                lambda: rs.project_text("NTSC", "not a pattern"),
                lambda: rs.project_text("NTSC", "75bars", "MADE_UP"),
                lambda: rs.project_text("NTSC", "75bars", frames=0),
                lambda: rs.project_text("NTSC", "flat_black")):
        with pytest.raises(ValueError):
            bad()


def test_the_cache_key_carries_the_generator_build():
    """A reference is only reproducible against the build that made it, which
    is why videosynth writes its commit into every file. The same project on
    two builds must not share a key."""
    text = rs.project_text("NTSC", "75bars")
    assert rs._request_key(text, "aaaaaaa") != rs._request_key(text, "bbbbbbb")
    assert rs._request_key(text, "aaaaaaa") == rs._request_key(text, "aaaaaaa")


def test_the_cache_stays_out_of_the_working_tree():
    """Generated media is megabytes a frame and other sessions hold
    uncommitted work in this repository, so the default cache must not be
    inside it."""
    repository = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(rs.__file__))))
    assert not rs.cache_dir({}).startswith(repository)


# --------------------------------------------------------------------------
# The tables, against the standards they cite
# --------------------------------------------------------------------------


def test_the_ire_scale_lands_on_the_standard_levels():
    """System M puts 140 IRE on the volt, so the generator's own quantisation
    profile must place sync tip at -40 and peak white at +100 IRE. Measured
    on the generated file: codes 15, 240, 282 and 800."""
    scale = rs.ire_scale("NTSC", "CVBS_U10_4FSC")
    ire = lambda code: (code - scale["offset_code"]) * scale["ire_per_code"]
    assert ire(240) == pytest.approx(0.0, abs=1e-12)
    assert ire(800) == pytest.approx(100.0, abs=0.01)
    assert ire(282) == pytest.approx(7.5, abs=0.01)
    assert ire(15) == pytest.approx(-40.0, abs=0.2)
    raw = rs.ire_scale("NTSC", "RAW_S16_40M")
    assert raw["ire_per_code"] == pytest.approx(140.0 / 1000.0, rel=1e-12)


def test_the_rates_are_the_generators_own():
    assert rs.sample_rate_of("NTSC", "RAW_S16_40M") == 40e6
    assert rs.sample_rate_of("NTSC", "RAW_S16_28M") == 28e6
    assert rs.sample_rate_of("NTSC", "CVBS_U10_4FSC") == 14318180.0
    assert rs.sample_rate_of("PAL", "CVBS_U10_4FSC") == 17734475.0


def test_the_generators_4fsc_clock_differs_from_the_exact_one():
    """RECORDED RATHER THAN CORRECTED. videosynth carries SMPTE 244M's
    printed 14.31818 MHz; the exact value implied by the line rate is
    14 318 181.8 Hz. The gap is 0.127 parts per million - nothing for a
    residual aligned per line, everything for an absolute phase argument."""
    generator = rs.SYSTEMS["NTSC"]["sample_rate_4fsc_hz"]
    exact = output_limit.FOUR_FSC_HZ
    parts_per_million = abs(exact - generator) / exact * 1e6
    assert 0.12 < parts_per_million < 0.13


def test_the_reference_is_not_quieter_than_the_output_cap():
    """A residual cannot honestly be quoted below the reference's own
    quantisation. Both encodings sit within a factor of about one of the
    file's floor, so about one floor is the useful limit of this instrument
    and not a matter of how long it is averaged."""
    floor = output_limit.output_profile()["quantisation_noise_ire"]
    for encoding, expected in (("CVBS_U10_4FSC", 1.12), ("RAW_S16_40M", 0.88)):
        step = abs(rs.ire_scale("NTSC", encoding)["ire_per_code"])
        noise = step / np.sqrt(output_limit.QUANTISATION_DIVISOR)
        assert noise / floor == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------
# The forward chain, on planted signals
# --------------------------------------------------------------------------


def test_a_flat_field_modulates_to_exactly_one_frequency():
    """THE TEST THAT PINS THE FM LAW. With the emphasis off a constant
    picture level has nothing for the shelf to do and nothing for the clip to
    catch, so the instantaneous frequency must be the standard's stated value
    for that level and must not move at all."""
    law = rs.carrier_law()
    for level, expected in ((-40.0, 3.4e6), (0.0, 3685714.2857142854),
                            (100.0, 4.4e6)):
        planted = rs.planted_flat_field(level, 100e-6, 50e6)
        forward = rs.forward_to_rf(planted, emphasis=False, clip=False)
        frequency = forward["instantaneous_hz"]
        assert frequency.mean() == pytest.approx(expected, abs=1.0)
        assert frequency.std() < 1e-6
    assert law["hz_per_ire"] == pytest.approx(1e6 / 140.0, rel=1e-12)


def test_the_carrier_law_and_the_decoders_own_agree_exactly():
    """Two routes that share no arithmetic: the decoder's `hz_ire` and `ire0`
    are read from its system parameters, and `carrier_law` computes the same
    pair from SMPTE 32M's two stated ends through `profiles.IDEAL_DECK`."""
    params = rs.instrument_parameters("NTSC", "VHS", "SP")
    law = rs.carrier_law()
    assert law["hz_per_ire"] == pytest.approx(params["decoder_hz_per_ire"],
                                              rel=1e-12)
    blanking = law["sync_tip_hz"] - law["sync_tip_ire"] * law["hz_per_ire"]
    assert blanking == pytest.approx(params["decoder_blanking_hz"], rel=1e-12)


def test_the_planted_rf_demodulates_back_to_what_was_planted():
    """The forward chain and the instrument must be inverses on a signal
    whose answer is known, or a residual measures the pair rather than the
    deck."""
    params = rs.instrument_parameters("NTSC", "VHS", "SP")
    planted = rs.planted_flat_field(7.5, 2e-3, 50e6)
    forward = rs.forward_to_rf(planted, emphasis=False, clip=False)
    recovered = rs.demodulate(forward["rf"], 50e6, params)
    interior = recovered[20000:-20000]
    assert interior.mean() == pytest.approx(
        forward["instantaneous_hz"].mean(), abs=10.0)


def test_the_two_clip_readings_are_both_expressible_and_differ():
    """The clause's percentages are stated against a span, and which span is
    a reading. Both are kept so the one the measurement excluded stays
    demonstrable rather than merely asserted."""
    blanking = rs.clip_levels_ire(reading="blanking")
    tip = rs.clip_levels_ire(reading="sync_tip")
    assert blanking["dark_ire"] == pytest.approx(-40.0)
    assert blanking["white_ire"] == pytest.approx(160.0)
    assert tip["dark_ire"] == pytest.approx(16.0)
    assert tip["white_ire"] == pytest.approx(184.0)
    # the blanking reading's dark clip is the sync-tip level itself, which is
    # what makes it a dark clip rather than a picture clip
    assert blanking["dark_ire"] == pytest.approx(blanking["sync_tip_ire"])
    with pytest.raises(ValueError):
        rs.clip_levels_ire(reading="whatever")


def test_a_measured_clip_replaces_the_standards_and_says_so():
    """The SLV-778HF's own dark floor is at -125.8 IRE, which neither reading
    puts it at, so a caller must be able to state a measurement instead of
    the standard - and the result must record that it did."""
    planted = rs.planted_flat_field(-40.0, 100e-6, 50e6)
    forward = rs.forward_to_rf(planted, clip=True, clip_ire=(-125.8, 400.0))
    assert forward["clips_ire"]["reading"] == "measured"
    assert forward["clips_ire"]["dark_ire"] == pytest.approx(-125.8)


def test_the_stages_are_switchable_and_each_one_changes_the_answer():
    """Each stage has to be checkable on its own, so switching one off must
    make a difference - a stage that can be removed for nothing is not being
    applied."""
    planted = {"ire": np.sin(2 * np.pi * 1.0e6 * np.arange(20000) / 40e6) * 50.0,
               "sample_rate_hz": 40e6, "system": "NTSC"}
    plain = rs.forward_to_rf(planted, emphasis=False, clip=False)
    emphasised = rs.forward_to_rf(planted, emphasis=True, clip=False)
    assert not np.allclose(plain["instantaneous_hz"],
                           emphasised["instantaneous_hz"])
    # a 50 IRE sine lifted by the shelf reaches past the +160 IRE white clip
    clipped = rs.forward_to_rf(planted, emphasis=True, clip=True)
    assert not np.allclose(emphasised["instantaneous_hz"],
                           clipped["instantaneous_hz"])
    assert len(clipped["stages"]) > len(plain["stages"])


def test_the_rate_conversion_preserves_the_level():
    """Forty megasamples to fifty is five over four exactly, so the level a
    reference carries must survive it - a rate conversion that changes the
    carrier is changing the picture."""
    planted = rs.planted_flat_field(0.0, 1e-3, 40e6)
    same = rs.forward_to_rf(planted, emphasis=False, clip=False)
    converted = rs.forward_to_rf(planted, sample_rate_hz=50e6,
                                 emphasis=False, clip=False)
    assert converted["sample_rate_hz"] == 50e6
    interior = converted["instantaneous_hz"][1000:-1000]
    assert interior.mean() == pytest.approx(
        same["instantaneous_hz"].mean(), abs=1.0)
    assert len(converted["instantaneous_hz"]) == pytest.approx(
        len(same["instantaneous_hz"]) * 1.25, rel=0.01)


# --------------------------------------------------------------------------
# The residual's own algebra
# --------------------------------------------------------------------------


def test_a_reference_differenced_with_itself_is_exactly_zero():
    """THE CONTROL. Every alignment, every window and every conversion in the
    residual must leave nothing at all when the two sides are the same array.
    Anything that survives here is the instrument."""
    params = rs.instrument_parameters("NTSC", "VHS", "SP")
    law = rs.carrier_law()
    rate = 50e6
    line = int(round(params["line_period_s"] * rate))
    frequency = np.full(line * 12, law["sync_tip_hz"] + 40.0
                        * law["hz_per_ire"])
    for k in range(12):                       # a sync pulse a line
        start = k * line
        width = int(round(params["pulse_widths_us"]["horizontal"]
                          * 1e-6 * rate))
        frequency[start:start + width] = law["sync_tip_hz"]
    start = 5 * line
    one = rs.line_residual(frequency, start, frequency, start, params, rate)
    assert one is not None
    assert np.max(np.abs(one["residual_hz"])) == pytest.approx(0.0, abs=1e-6)
    assert one["delay_samples"] == pytest.approx(0.0, abs=1e-9)
    windows = rs.residual_by_window(one["residual_hz"],
                                    one["window_start_us"], params, rate, law)
    assert set(windows) == set(rs.windows_of(params))
    for name, values in windows.items():
        assert np.max(np.abs(values)) == pytest.approx(0.0, abs=1e-6), name


def test_a_planted_offset_comes_back_as_the_offset():
    """A residual instrument that cannot report a known departure is not
    measuring one. One IRE of carrier offset must read as one IRE."""
    params = rs.instrument_parameters("NTSC", "VHS", "SP")
    law = rs.carrier_law()
    rate = 50e6
    line = int(round(params["line_period_s"] * rate))
    reference = np.full(line * 12, law["sync_tip_hz"] + 40.0 * law["hz_per_ire"])
    width = int(round(params["pulse_widths_us"]["horizontal"] * 1e-6 * rate))
    for k in range(12):
        reference[k * line:k * line + width] = law["sync_tip_hz"]
    measured = reference + 1.0 * law["hz_per_ire"]
    start = 5 * line
    one = rs.line_residual(measured, start, reference, start, params, rate)
    residual = rs.hz_to_ire(one["residual_hz"], law)
    assert np.median(residual) == pytest.approx(1.0, abs=1e-6)
    verdict = rs.against_floor(residual)
    assert verdict["meaningful"] is True
    assert verdict["over_floor"] == pytest.approx(
        1.0 / output_limit.output_profile()["quantisation_noise_ire"], rel=0.01)


def test_a_departure_under_the_cap_is_reported_invisible():
    """Ethan's cap is the stopping rule: a residual under the file's own
    quantisation noise is not small, it is unrepresentable, and the instrument
    must say so rather than quote a small number."""
    floor = output_limit.output_profile()["quantisation_noise_ire"]
    verdict = rs.against_floor(np.full(1000, floor / 2.0))
    assert verdict["meaningful"] is False
    assert verdict["over_floor"] == pytest.approx(0.5, rel=1e-9)


def test_the_windows_never_overlap_where_they_must_not():
    """The reported intervals must partition the line where they are meant
    to, and the blanking windows must stop short of active video - the two
    generators do not start their pictures at the same place."""
    params = rs.instrument_parameters("NTSC", "VHS", "SP")
    windows = rs.windows_of(params)
    edge, tip, trailing = (windows["sync leading edge"], windows["sync tip"],
                           windows["sync trailing"])
    assert edge[1] == tip[0] and tip[1] == trailing[0]
    assert windows["back porch"][1] < params["active_video_us"][0]
    assert windows["blanking"][1] < params["active_video_us"][0]
    assert windows["blanking"][0] == -params["front_porch_us"]
    for name, (low, high) in windows.items():
        assert low < high, name


def test_the_sync_calibration_is_two_numbers_and_recovers_a_planted_pair():
    """The anchoring is the pair the sync-only law licenses and no more, and
    it must recover a planted tip and porch exactly."""
    params = rs.instrument_parameters("NTSC", "VHS", "SP")
    law = rs.carrier_law()
    rate = 50e6
    line = int(round(params["line_period_s"] * rate))
    blanking = law["sync_tip_hz"] + 40.0 * law["hz_per_ire"]
    frequency = np.full(line * 12, blanking)
    width = int(round(params["pulse_widths_us"]["horizontal"] * 1e-6 * rate))
    for k in range(12):
        frequency[k * line:k * line + width] = law["sync_tip_hz"]
    starts = [k * line for k in range(2, 10)]
    anchor = rs.sync_calibration(frequency, starts, params, rate)
    assert anchor["sync_tip_hz"] == pytest.approx(law["sync_tip_hz"], abs=1.0)
    assert anchor["back_porch_hz"] == pytest.approx(blanking, abs=1.0)
    assert anchor["span_hz"] == pytest.approx(40.0 * law["hz_per_ire"], abs=2.0)


# --------------------------------------------------------------------------
# The four that need the generator
# --------------------------------------------------------------------------


@needs_generator
def test_the_generated_frame_is_the_standards_raster():
    """One NTSC frame at 4fsc is 910 by 525 orthogonal samples. The stream is
    headerless, so if the geometry is wrong nothing else notices."""
    made = rs.reference("NTSC", "75bars", "CVBS_U10_4FSC", frames=1)
    assert made["samples_per_line"] == 910
    assert made["lines"] == 525
    assert len(made["ire"]) == 910 * 525
    assert made["frames"] == pytest.approx(1.0)


@needs_generator
def test_the_generated_levels_are_the_standards_levels():
    """Nothing is fitted here: the generator's file simply lands on sync tip,
    blanking, the NTSC setup and peak white."""
    made = rs.reference("NTSC", "75bars", "CVBS_U10_4FSC", frames=1)
    codes = np.asarray(made["codes"])
    common = np.bincount(codes)
    for expected in (15, 240, 282, 800):
        assert common[expected] > 1000, expected
    assert np.min(codes) == 15
    # the 75 per cent bars peak above 100 IRE because the yellow bar's chroma
    # rides on its luma; that is the pattern, not an error
    assert 100.0 < float(np.max(made["ire"])) < 110.0


@needs_generator
def test_the_cache_returns_the_identical_array():
    """Ethan asked for this to be usable while a residual is being formed, so
    a repeated request must not run the generator, must not read the disc,
    and must hand back the same object rather than a copy."""
    rs.clear_cache()
    first = rs.reference("NTSC", "75bars", "CVBS_U10_4FSC", frames=1)
    second = rs.reference("NTSC", "75bars", "CVBS_U10_4FSC", frames=1)
    assert second["ire"] is first["ire"]
    assert second["codes"] is first["codes"]
    assert second["request"]["generated"] is False
    assert second["request"]["key"] == first["request"]["key"]


@needs_generator
def test_the_generators_sync_pulses_are_one_transition_narrow():
    """A DEFECT IN THE REFERENCE, pinned so it cannot regress into silence.

    videosynth applies each pulse's specified duration from the start of the
    leading transition to the end of the trailing one, where every standard
    specifies it between the half-amplitude points, so all three pulse classes
    measure one transition narrow: 4.4011 us against 4.700, 2.0254 against
    2.300, 26.8190 against 27.100. The cause is `PulseWidthSamples` rounding
    to 67, 33 and 388 samples at 4fsc and `ShapedPulseLevel` placing both
    four-sample ramps inside that window.

    This is why the residual instrument aligns on the leading edge alone. If
    a later videosynth fixes it this test fails, which is the right way round:
    the alignment can then be widened.
    """
    made = rs.reference("NTSC", "75bars", "CVBS_U10_4FSC", frames=1)
    signal = np.asarray(made["ire"])
    rate = made["sample_rate_hz"]
    low = signal < -20.0
    down = np.flatnonzero(low[1:] & ~low[:-1])
    up = np.flatnonzero(~low[1:] & low[:-1])
    up = up[up > down[0]]
    count = min(len(down), len(up))
    widths = (up[:count] - down[:count]) / rate * 1e6
    horizontal = widths[(widths > 3.0) & (widths < 6.0)]
    assert len(horizontal) > 400
    assert float(np.median(horizontal)) == pytest.approx(4.40, abs=0.02)
    assert float(np.median(horizontal)) < 4.7 - 0.1     # outside the tolerance
