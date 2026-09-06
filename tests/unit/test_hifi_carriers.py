"""The Hi-Fi carriers as an independent timing probe.

What is tested: that the specification figures are the specification's;
that a planted time-base wander on an AFM carrier is recovered with an
honest error bar; that planted audio modulation does not leak into the
time-base error beyond the stated bound, and that what does leak in-band
is removed by the two-carrier estimator; that the video path's
sync-crossing readout recovers the same planted wander; that the
cross-check calls agreement agreement and a scale error a scale error;
and that the presence detector finds a planted carrier among line-harmonic
pickets and reports its absence as absence.
"""

from fractions import Fraction

import numpy as np
import pytest

from vhsdecode.models import colour_under, transport_model
from vhsdecode.models import hifi_carriers as hc


LINE_RATE = colour_under.line_rate_hz("NTSC")


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------

def wander(t):
    """A drum-rate wow of 0.04 percent - the measured size on this
    transport - and a smaller capstan-rate component."""
    return 4e-4 * np.sin(2 * np.pi * 29.97 * t) + 2e-4 * np.sin(2 * np.pi * 4.83 * t)


def afm_carrier(fs, seconds, f0, eps=None, audio_hz=None, noise=0.3, seed=0):
    """An FM audio carrier played back through a time base error `eps(t)`
    (fractional period error, positive = slow: every frequency plays back
    divided by 1 + eps) and modulated by `audio_hz(t)` hertz of deviation."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    e = eps(t) if eps else np.zeros(n)
    a = audio_hz(t) if audio_hz else np.zeros(n)
    f_inst = f0 / (1.0 + e) + a
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    rng = np.random.default_rng(seed)
    return np.cos(phase) + noise * rng.standard_normal(n), t


def fm_video(fs, seconds, eps, noise=0.05, seed=1):
    """A luma FM record: sync tip (3.4 MHz) for 4.7 us of every line,
    blanking (3.686 MHz) for the rest, with equalizing pulses (2.3 us at
    half-line spacing) on a few lines to exercise the width gate; played
    back through `eps(t)` so that both the line period stretches by
    (1 + eps) and every carrier frequency divides by it."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    e = eps(t)
    tape_time = np.cumsum(1.0 / (1.0 + e)) / fs
    period = 1.0 / LINE_RATE
    tip, white = 3.4e6, 4.4e6
    blank = tip + 40.0 / 140.0 * (white - tip)
    line = np.floor(tape_time / period)
    phase_in_line = tape_time - line * period
    sync = phase_in_line < 4.7e-6
    equalizing_lines = (line >= 40) & (line < 46)
    half = phase_in_line % (period / 2.0)
    eq = equalizing_lines & (half < 2.3e-6)
    f_video = np.where(np.where(equalizing_lines, eq, sync), tip, blank)
    f_played = f_video / (1.0 + e)
    phase = 2 * np.pi * np.cumsum(f_played) / fs
    rng = np.random.default_rng(seed)
    return np.cos(phase) + noise * rng.standard_normal(n), t


def compare(readout_time, readout, t, eps, trim=200):
    truth = np.interp(readout_time, t, eps(t))
    a = np.asarray(readout)[trim:-trim]
    b = truth[trim:-trim]
    return (float(np.corrcoef(a, b)[0, 1]), float(np.std(a) / np.std(b)),
            float(np.std(a - b)))


def video_tap_like(fs, seconds, seed=7):
    """What the AFM bands of a video-tap capture look like without a
    carrier: a sloping continuum plus a picket fence of line harmonics
    15.734 kHz apart (the static test pattern's line spectrum), the
    pickets 15 dB over the continuum."""
    n = int(fs * seconds)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    # a gentle slope across the band: first-order low-pass at 1 MHz
    from scipy import signal
    b, a = signal.butter(1, 1e6, fs=fs)
    x = signal.lfilter(b, a, x)
    t = np.arange(n) / fs
    harmonics = np.arange(int(1.0e6 / LINE_RATE), int(2.0e6 / LINE_RATE))
    density = np.mean(x ** 2) / (fs / 2)          # white-equivalent density
    picket = np.sqrt(2 * density * 15.0 * 10 ** 1.5)   # 15 dB over the continuum in a 15 Hz bin
    for k in harmonics:
        x = x + picket * np.cos(2 * np.pi * k * LINE_RATE * t + rng.uniform(0, 2 * np.pi))
    return x


# --------------------------------------------------------------------------
# the specification
# --------------------------------------------------------------------------

class TestSpecification:
    def test_carriers_are_the_standards(self):
        assert hc.carriers_hz("NTSC") == (1.3e6, 1.7e6)      # SMPTE 32M 5.8
        assert hc.carriers_hz("PAL") == (1.4e6, 1.8e6)       # JVC 7.2.3
        assert hc.CARRIER_TOLERANCE_HZ == 10e3
        assert hc.MAX_DEVIATION_HZ == 150e3                  # 5.9
        assert hc.REFERENCE_DEVIATION_HZ == 50e3

    def test_nominal_ratio_is_the_integer_pair(self):
        assert hc.nominal_ratio("NTSC") == Fraction(17, 13)
        assert hc.nominal_ratio("PAL") == Fraction(9, 7)

    def test_wavelengths_at_the_stated_writing_speed(self):
        low, high = hc.wavelength_m("NTSC")
        assert low == pytest.approx(5.80 / 1.3e6)
        assert high == pytest.approx(5.80 / 1.7e6)
        assert low * 1e6 == pytest.approx(4.4615, abs=1e-3)   # Ethan's 4.5
        assert high * 1e6 == pytest.approx(3.4118, abs=1e-3)  # Ethan's 3.4
        pal = hc.wavelength_m("PAL")
        assert pal[0] == pytest.approx(4.85 / 1.4e6)

    def test_read_depth_is_lambda_over_two_pi(self):
        low, high = hc.read_depth_m("NTSC")
        assert low == pytest.approx(5.80 / 1.3e6 / (2 * np.pi))
        assert high == pytest.approx(5.80 / 1.7e6 / (2 * np.pi))

    def test_hifi_cluster_sits_between_chroma_and_luma(self):
        c = hc.wavelength_clusters("NTSC")
        assert c["chroma"]["colour_under"] > max(c["hifi"].values())
        assert min(c["hifi"].values()) > max(c["luma"].values())

    def test_relative_azimuth_follows_table_5(self):
        assert hc.relative_azimuth_degrees("SP") == pytest.approx(36.5)
        assert hc.relative_azimuth_degrees("EP") == pytest.approx(24.5)

    def test_deck_does_not_fix_the_ratio(self):
        assert hc.DECK_CARRIER_GENERATION["ratio_fixed_by_design"] is False

    def test_crosstalk_expectation_is_a_loss(self):
        out = hc.azimuth_crosstalk_db("NTSC", "SP", 1)
        assert out["spacing_loss_db"] < 0
        assert out["total_db_29um"] < out["total_db_10um"] < 0

    def test_reach_follows_carson(self):
        assert hc.carrier_half_width_hz() == pytest.approx(150e3 + 20e3 + 10e3)
        assert hc.reference_half_width_hz() == pytest.approx(50e3 + 20e3)
        assert hc.luma_band_hz("NTSC") == pytest.approx((2.4e6, 5.4e6))


# --------------------------------------------------------------------------
# the carrier readout
# --------------------------------------------------------------------------

class TestCarrierReadout:
    def test_planted_wander_is_recovered_with_an_honest_error(self):
        fs = 8e6
        x, t = afm_carrier(fs, 0.3, 1.3e6, eps=wander)
        out = hc.time_base_error(x, fs, system="NTSC", channel=1, band_hz=100.0)
        r, ratio, rms = compare(out["time_s"], out["tbe_band"], t, wander)
        assert r > 0.99
        assert ratio == pytest.approx(1.0, abs=0.05)
        # the stated error must cover the actual error, and not by a mile
        assert rms <= 3.0 * out["standard_error"]
        assert rms >= 0.1 * out["standard_error"]

    def test_series_starts_after_the_filter_settles(self):
        fs = 8e6
        x, t = afm_carrier(fs, 0.05, 1.3e6, eps=wander)
        f, rate, start = hc.instantaneous_frequency(x, fs, 1.3e6)
        assert start > 0
        assert start < 1.0 / LINE_RATE           # under a line at this width
        out = hc.time_base_error(x, fs)
        assert out["time_s"][0] > 0

    def test_mean_reference_separates_the_electronic_offset(self):
        fs = 8e6
        off = 5e3                                  # within the +/- 10 kHz allowed
        x, t = afm_carrier(fs, 0.2, 1.3e6 + off, eps=wander)
        by_mean = hc.time_base_error(x, fs, reference="mean")
        by_nominal = hc.time_base_error(x, fs, reference="nominal")
        assert by_mean["electronic_offset"] == pytest.approx(off / 1.3e6, rel=0.05)
        assert abs(by_mean["tbe"].mean()) < 1e-5
        assert by_nominal["tbe"].mean() == pytest.approx(1.3e6 / (1.3e6 + off) - 1, rel=0.05)

    def test_convention_is_period_error_positive_slow(self):
        fs = 8e6
        slow = 2e-3
        x, t = afm_carrier(fs, 0.05, 1.3e6, eps=lambda t: np.full_like(t, slow))
        out = hc.time_base_error(x, fs, reference="nominal")
        assert out["tbe"].mean() == pytest.approx(slow, rel=0.02)


class TestAudioLeakage:
    def test_out_of_band_audio_stays_within_the_bound(self):
        """The reference modulation, +/- 50 kHz at 400 Hz (5.9): 3.8 percent
        of the carrier in the raw readout, and after the 100 Hz band no
        more than the stated bound."""
        fs = 8e6
        audio = lambda t: hc.REFERENCE_DEVIATION_HZ * np.cos(
            2 * np.pi * hc.REFERENCE_MODULATION_HZ * t)
        x, t = afm_carrier(fs, 0.3, 1.3e6, eps=wander, audio_hz=audio)
        out = hc.time_base_error(x, fs, band_hz=100.0)
        raw_leak = np.std(out["tbe"])
        assert raw_leak > 0.01                       # the raw readout is swamped
        bound = hc.audio_leakage_bound(hc.REFERENCE_MODULATION_HZ,
                                       hc.REFERENCE_DEVIATION_HZ, 1.3e6,
                                       out["rate_hz"], 100.0)
        r, ratio, rms = compare(out["time_s"], out["tbe_band"], t, wander)
        assert rms <= bound + 3.0 * out["standard_error"]
        assert r > 0.99

    def test_in_band_audio_is_the_honest_failure_of_one_carrier(self):
        """A 30 Hz bass note at -20 dB re the reference deviation sits INSIDE
        the flutter band and no single-carrier readout can remove it."""
        fs = 8e6
        audio = lambda t: 5e3 * np.cos(2 * np.pi * 30.0 * t)
        x, t = afm_carrier(fs, 0.3, 1.3e6, eps=wander, audio_hz=audio)
        out = hc.time_base_error(x, fs, band_hz=100.0)
        r, ratio, rms = compare(out["time_s"], out["tbe_band"], t, wander)
        bound = hc.audio_leakage_bound(30.0, 5e3, 1.3e6, out["rate_hz"], 100.0)
        assert bound > 3e-3                          # ten times the wander
        assert rms <= bound + 3.0 * out["standard_error"]
        assert rms > 1e-3                            # and it really leaks

    def test_two_carriers_remove_the_shared_audio(self):
        fs = 8e6
        audio = lambda t: 5e3 * np.cos(2 * np.pi * 30.0 * t)
        one, t = afm_carrier(fs, 0.3, 1.3e6, eps=wander, audio_hz=audio, seed=2)
        two, _ = afm_carrier(fs, 0.3, 1.7e6, eps=wander, audio_hz=audio, seed=3)
        x = one + two
        out = hc.time_base_error_pair(x, fs, system="NTSC", band_hz=100.0)
        r, ratio, rms = compare(out["time_s"], out["tbe_band"], t, wander)
        assert r > 0.99
        assert ratio == pytest.approx(1.0, abs=0.05)
        assert rms <= 3.0 * out["standard_error"] + 2e-5
        # and the common audio comes back for inspection, band-limited
        recovered = out["audio_hz_band"]
        truth = np.interp(out["time_s"], t, audio(t))
        assert np.corrcoef(recovered[200:-200], truth[200:-200])[0, 1] > 0.99
        assert np.std(recovered[200:-200] - truth[200:-200]) < 0.02 * np.std(truth)
        assert out["measured_ratio"] == pytest.approx(17.0 / 13.0, rel=1e-3)


# --------------------------------------------------------------------------
# band limiting and its error bar
# --------------------------------------------------------------------------

class TestBandLimit:
    def test_error_bar_matches_white_noise(self):
        rng = np.random.default_rng(5)
        rate, band = 15734.0, 100.0
        x = rng.standard_normal(60000)
        passed, error, _ = hc.band_limit(x, rate, band)
        assert error == pytest.approx(np.std(passed), rel=0.25)

    def test_error_bar_matches_blue_noise(self):
        """Differenced white noise - what a period readout carries - has a
        density rising as f squared; the white assumption would over-state
        the in-band error many times over, the law must not."""
        rng = np.random.default_rng(6)
        rate, band = 15734.0, 100.0
        x = np.diff(rng.standard_normal(120001))
        passed, error, _ = hc.band_limit(x, rate, band)
        law = hc.noise_law(x, rate, band)
        assert law["slope"] > 1.5
        assert error == pytest.approx(np.std(passed[1000:-1000]), rel=0.3)

    def test_block_average_runs_at_the_requested_rate(self):
        x = np.arange(1000, dtype=float)
        out, rate = hc.block_average(x, 1000.0, 100.0)
        assert rate == pytest.approx(100.0)
        assert len(out) == 100
        assert out[0] == pytest.approx(4.5)


# --------------------------------------------------------------------------
# the presence detector
# --------------------------------------------------------------------------

class TestPresence:
    fs = 8e6
    seconds = 0.25

    def test_absence_is_reported_as_absence(self):
        x = video_tap_like(self.fs, self.seconds)
        out = hc.carrier_presence(x, self.fs, 1.3e6)
        assert out["line_sigma"] < hc.DETECTION_SIGMA
        assert abs(out["hump_sigma"]) < hc.DETECTION_SIGMA

    def test_planted_carriers_are_found_and_bounded(self):
        x = video_tap_like(self.fs, self.seconds)
        bound = hc.detection_bound(x, self.fs, 1.3e6, levels_db=(-20.0, -30.0, -40.0, -50.0),
                                   reference_power=float(np.mean(x ** 2)))
        rows = {row["level_db"]: row for row in bound["levels"]}
        assert rows[-20.0]["line_detectable"] and rows[-20.0]["hump_detectable"]
        # the statistics fall monotonically with the planted level
        sigmas = [rows[level]["hump_sigma"] for level in (-20.0, -30.0, -40.0, -50.0)]
        assert sigmas == sorted(sigmas, reverse=True)
        assert bound["line_bound_db"] is not None and bound["line_bound_db"] <= -20.0
        assert bound["hump_bound_db"] is not None and bound["hump_bound_db"] <= -20.0

    def test_planted_deviation_is_the_deviation_asked_for(self):
        """The planted carrier's peak deviation must be the hertz asked for.
        Writing the modulation term as 2 pi (dev/mod) sin(...) instead of
        (dev/mod) sin(...) multiplies it by 2 pi - 314.2 kHz for a requested
        50 kHz, past the 150 kHz maximum of 5.9 - and throws 86 percent of
        the planted power outside the hump band the detector integrates."""
        fs = 8e6
        x = hc.plant_fm_carrier(int(fs * 0.02), fs, 1.3e6, 1.0)
        z = x * np.exp(-2j * np.pi * 1.3e6 * np.arange(len(x)) / fs)
        from scipy import signal
        h = signal.firwin(1001, 4 * hc.REFERENCE_DEVIATION_HZ, fs=fs)
        z = signal.fftconvolve(z, h, mode="same")[2000:-2000]
        deviation = np.diff(np.unwrap(np.angle(z))) * fs / (2 * np.pi)
        assert np.max(np.abs(deviation)) == pytest.approx(
            hc.REFERENCE_DEVIATION_HZ, rel=0.02)
        assert np.max(np.abs(deviation)) < hc.MAX_DEVIATION_HZ      # 5.9

    def test_planted_power_lands_inside_the_hump_band(self):
        """A reference-modulated carrier reaches +/- 50 kHz, so essentially
        all of its power must fall in the +/- 70 kHz band the hump statistic
        integrates - the property the 2 pi error destroyed."""
        fs = 8e6
        x = hc.plant_fm_carrier(int(fs * 0.25), fs, 1.3e6, 1.0)
        spectrum = hc.power_spectrum(x, fs, 2 * hc.CARRIER_TOLERANCE_HZ
                                     / hc.SPECTRUM_BINS_PER_TOLERANCE)
        hump = hc.reference_half_width_hz()
        inside = hc.band_power(*spectrum, 1.3e6 - hump, 1.3e6 + hump)
        assert inside / np.mean(x ** 2) > 0.95

    def test_narrow_line_is_placed_on_the_carrier_not_a_picket(self):
        x = video_tap_like(self.fs, self.seconds)
        amplitude = np.sqrt(2 * np.mean(x ** 2) * 10 ** (-25 / 10))
        y = x + hc.plant_fm_carrier(len(x), self.fs, 1.3e6 + 4e3, amplitude, 0.0)
        out = hc.carrier_presence(y, self.fs, 1.3e6)
        assert abs(out["peak_at_hz"] - (1.3e6 + 4e3)) <= out["bin_hz"]
        assert out["line_sigma"] > hc.DETECTION_SIGMA


class TestConfounds:
    """What else lives at 1.3 and 1.7 MHz in a video-tap capture, and
    whether the detector can be told to ignore it."""

    def test_a_line_picket_falls_inside_both_tolerance_windows(self):
        """The line-harmonic grid is 15.734 kHz (SMPTE 32M 4.3.1.2) and the
        centre tolerance is +/- 10 kHz (5.8), so a picket falls inside the
        window at BOTH nominal carriers: the 83rd at 1.305944 MHz, 5.94 kHz
        above 1.3 MHz, and the 108th at 1.699301 MHz, 699 Hz below 1.7 MHz.
        A narrow line found there is not by itself a carrier."""
        # the offset is the FREQUENCY minus the picket, so the carrier sits
        # 5944 Hz below the 83rd picket and 699 Hz above the 108th
        for centre, order, offset in ((1.3e6, 83, -5944.1), (1.7e6, 108, 699.3)):
            out = hc.line_harmonic_offset_hz(centre, "NTSC")
            assert out["harmonic"] == order
            assert out["offset_hz"] == pytest.approx(offset, abs=1.0)
            assert abs(out["offset_hz"]) < hc.CARRIER_TOLERANCE_HZ

    def test_the_colour_under_second_harmonic_sits_in_channel_1s_hump(self):
        """2 x 629.371 kHz = 1.258741 MHz, 41.26 kHz below the channel 1
        carrier: outside the +/- 10 kHz window but inside the +/- 70 kHz band
        the hump statistic integrates. Channel 2's nearest is the third
        harmonic, 188 kHz away - outside even the +/- 180 kHz reach."""
        one = hc.colour_under_harmonic_offset_hz(1.3e6, "NTSC")
        assert one["harmonic"] == 2
        assert one["offset_hz"] == pytest.approx(41258.7, abs=1.0)
        assert hc.CARRIER_TOLERANCE_HZ < abs(one["offset_hz"]) < hc.reference_half_width_hz()
        two = hc.colour_under_harmonic_offset_hz(1.7e6, "NTSC")
        assert abs(two["offset_hz"]) > hc.carrier_half_width_hz()

    def test_confounds_are_named_placed_and_ordered(self):
        found = hc.confounds(1.3e6, "NTSC")
        assert {c["what"] for c in found} == {"colour-under harmonic", "line harmonic"}
        offsets = [abs(c["offset_hz"]) for c in found]
        assert offsets == sorted(offsets)
        assert all(abs(c["offset_hz"]) <= hc.carrier_half_width_hz() for c in found)
        cu = [c for c in found if c["what"] == "colour-under harmonic"]
        assert [c["harmonic"] for c in cu] == [2]
        assert cu[0]["in_hump"] and not cu[0]["in_tolerance"]
        assert any(c["in_tolerance"] for c in found if c["what"] == "line harmonic")

    def test_excluding_a_named_neighbour_takes_it_out_of_the_hump(self):
        """A strong colour-under second harmonic raises the hump statistic
        at channel 1 although no AFM carrier is present. Naming it in
        `exclude` must remove it; nothing else may move."""
        fs, seconds = 8e6, 0.25
        x = video_tap_like(fs, seconds)
        harmonic = hc.colour_under_harmonic_offset_hz(1.3e6, "NTSC")["harmonic_hz"]
        amplitude = np.sqrt(2 * np.mean(x ** 2) * 10 ** (-15 / 10))
        y = x + amplitude * np.cos(2 * np.pi * harmonic * np.arange(len(x)) / fs)
        polluted = hc.carrier_presence(y, fs, 1.3e6)
        cleaned = hc.carrier_presence(y, fs, 1.3e6,
                                      exclude=[(harmonic, 3 * polluted["bin_hz"])])
        clean = hc.carrier_presence(x, fs, 1.3e6)
        assert polluted["hump_sigma"] > hc.DETECTION_SIGMA        # it fires
        assert abs(cleaned["hump_sigma"]) < hc.DETECTION_SIGMA    # and stops
        assert cleaned["hump_sigma"] == pytest.approx(clean["hump_sigma"], abs=1.0)
        assert cleaned["bins_excluded"] > 0 and clean["bins_excluded"] == 0
        # the narrow-line statistic never saw it: it is 41 kHz out of window
        # (not exactly: the planted harmonic is inside the +/- 180 kHz reach
        # and moves the floor's scatter in the seventh decimal place)
        assert polluted["line_sigma"] == pytest.approx(clean["line_sigma"], abs=1e-3)

    def test_the_peak_carries_its_distance_from_the_picket_grid(self):
        fs, seconds = 8e6, 0.25
        x = video_tap_like(fs, seconds)
        out = hc.carrier_presence(x, fs, 1.3e6)
        expected = hc.line_harmonic_offset_hz(out["peak_at_hz"], "NTSC")["offset_hz"]
        assert out["peak_line_harmonic_offset_hz"] == pytest.approx(expected)
        assert abs(out["peak_line_harmonic_offset_hz"]) <= 0.5 * LINE_RATE + 1.0


# --------------------------------------------------------------------------
# the video readout and the cross-check
# --------------------------------------------------------------------------

# Long enough that the cross-check can actually SEARCH for the drum line.
# The search's resolution is the record's own, and after cross_check trims
# half a 100 Hz kernel (50 ms) from each end of the overlap, the drum at
# 29.97 Hz needs 0.278 s of what is left before its local background holds
# the four bins transport_model.find_lines requires
# (hc.line_search_duration_s(29.97)). At 0.3 s the trimmed span was 0.249 s
# and the search declined; 0.5 s leaves 0.449 s, six background bins.
FIXTURE_SECONDS = 0.5


@pytest.fixture(scope="module")
def video_and_carrier():
    fs = 16e6
    seconds = FIXTURE_SECONDS
    rf, t = fm_video(fs, seconds, wander)
    afm, _ = afm_carrier(fs, seconds, 1.3e6, eps=wander, seed=4)
    return fs, t, rf, afm


class TestSyncCrossing:
    def test_planted_wander_is_recovered_from_sync_periods(self, video_and_carrier):
        fs, t, rf, _ = video_and_carrier
        out = hc.sync_crossing_time_base(rf, fs, system="NTSC")
        assert out["accepted"] > 0.9 * FIXTURE_SECONDS * LINE_RATE
        grid, values = hc.on_uniform_grid(out["time_s"], out["tbe"], LINE_RATE)
        passed, error, _ = hc.band_limit(values, LINE_RATE, 100.0)
        r, ratio, rms = compare(grid, passed, t, wander, trim=400)
        assert r > 0.98
        assert ratio == pytest.approx(1.0, abs=0.1)

    def test_equalizing_pulses_are_rejected(self, video_and_carrier):
        fs, t, rf, _ = video_and_carrier
        out = hc.sync_crossing_time_base(rf, fs, system="NTSC")
        # the six equalizing lines carry twelve short pulses that must not
        # be counted as line syncs: the accepted periods stay near nominal
        assert np.all(np.abs(out["tbe"]) < 0.01)
        assert out["rejected"] >= 12

    def test_record_edges_are_not_read(self, video_and_carrier):
        fs, t, rf, _ = video_and_carrier
        out = hc.sync_crossing_time_base(rf, fs, system="NTSC")
        assert out["time_s"][0] >= hc.ANALYTIC_OVERLAP / fs
        truth = np.interp(out["time_s"], t, wander(t))
        # no seven-sigma outlier at the start: the first periods are as
        # good as the rest
        assert np.max(np.abs(out["tbe"][:5] - truth[:5])) < 5 * out["noise"]


class TestCrossCheck:
    def test_same_wander_reads_consistent(self, video_and_carrier):
        fs, t, rf, afm = video_and_carrier
        video = hc.sync_crossing_time_base(rf, fs, system="NTSC")
        hifi = hc.time_base_error(afm, fs, system="NTSC", channel=1)
        out = hc.cross_check(video, hifi, band_hz=100.0)
        assert out["usable"]
        assert out["correlation"] > 0.98
        assert out["slope"] == pytest.approx(1.0, abs=0.1)
        assert out["consistent"], out
        assert out["verdict"] == "consistent"
        # the drum line is in both readouts, and NOT in their difference:
        # both must see the same transport, so a line that survives the
        # subtraction is one readout's systematic error
        assert "head drum" in out["lines_searched"]
        for tag in ("lines_video", "lines_hifi"):
            found = {line["name"]: line for line in out[tag]}
            assert "head drum" in found, (tag, out[tag], out["lines_not_searched"])
            assert abs(found["head drum"]["at_hz"] - 29.97) <= 0.08 * 29.97
            assert found["head drum"]["height_over_background"] > 100.0
        difference = {line["name"]: line for line in out["lines_difference"]}
        assert difference["head drum"]["height_over_background"] < 10.0

    def test_a_rate_the_record_cannot_reach_is_named_not_dropped(self, video_and_carrier):
        """The capstan turns at 4.83 Hz; the line search needs 1.73 s before
        its local background holds four bins, and this record is 0.45 s. A
        silent empty list would read as a clean transport, so the rate must
        come back named, with the length it would need."""
        fs, t, rf, afm = video_and_carrier
        out = hc.cross_check(hc.sync_crossing_time_base(rf, fs),
                             hc.time_base_error(afm, fs), band_hz=100.0)
        refused = {row["name"]: row for row in out["lines_not_searched"]}
        assert "capstan" in refused
        assert refused["capstan"]["needs_s"] > out["line_search_s"]
        assert refused["capstan"]["needs_s"] == pytest.approx(
            hc.line_search_duration_s(refused["capstan"]["rate_hz"])["duration_s"])
        # nothing is both searched and refused, and every rate is accounted for
        assert set(out["lines_searched"]).isdisjoint(refused)
        assert len(out["lines_searched"]) + len(refused) == 8    # the transport

    def test_the_grid_is_per_line_so_the_drum_is_not_aliased(self, video_and_carrier):
        """A series sampled once per FIELD has its Nyquist frequency at
        29.970 Hz - exactly the drum rate, one revolution per two fields -
        so the drum aliases onto the head alternation and cannot be told
        from it. The sync-crossing readout is per LINE, 15.734 kHz, which is
        why the drum is an ordinary low-frequency component of it."""
        fs, t, rf, afm = video_and_carrier
        video = hc.sync_crossing_time_base(rf, fs)
        assert video["line_rate_hz"] == pytest.approx(LINE_RATE)
        # one value per accepted line period, not per field
        span = video["time_s"][-1] - video["time_s"][0]
        assert len(video["tbe"]) == pytest.approx(span * LINE_RATE, rel=0.02)
        field_rate = 2 * transport_model.drum_rate_hz()
        assert field_rate / 2 == pytest.approx(29.97)            # the Nyquist trap
        assert video["line_rate_hz"] / 2 > 100 * 29.97

    def test_scale_error_is_named(self, video_and_carrier):
        fs, t, rf, afm = video_and_carrier
        video = hc.sync_crossing_time_base(rf, fs, system="NTSC")
        hifi = hc.time_base_error(afm, fs, system="NTSC", channel=1)
        hifi = dict(hifi, tbe=hifi["tbe"] * 1.3)
        out = hc.cross_check(video, hifi, band_hz=100.0)
        assert not out["consistent"]
        assert out["verdict"] == "scale error"
        assert out["slope"] == pytest.approx(1.3, abs=0.1)

    def test_too_short_an_overlap_is_refused(self, video_and_carrier):
        fs, t, rf, afm = video_and_carrier
        video = hc.sync_crossing_time_base(rf, fs, system="NTSC")
        hifi = hc.time_base_error(afm, fs, system="NTSC", channel=1)
        short = {k: (v[:300] if isinstance(v, np.ndarray) else v) for k, v in hifi.items()}
        out = hc.cross_check(video, short, band_hz=100.0)
        assert not out["usable"]
