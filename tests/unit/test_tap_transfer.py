"""The cross-instrument transfer, on a synthetic pair of taps.

A static pattern is modulated onto a VHS-like FM with sync structure; the
"playback" tap is the same generator's output for OTHER frames (so the
carrier phase differs, as it does between the real captures) passed through
a planted complex channel with delay, dispersion, noise and a wandering time
base. The instrument must recover the channel in magnitude AND phase, flag a
planted content dependence, and refuse a misaligned or unrelated pair.
"""

import numpy as np
import pytest

from vhsdecode.models import tap_transfer as tt


RATE = 50e6
PARAMS = tt.capture_parameters("NTSC", "VHS", "SP")


def _pattern_video(levels, line_samples, rng=None):
    """One line of luma in Hz: sync, blanking, then bars at `levels` IRE."""
    fs = RATE
    sync = int(round(PARAMS["pulse_widths_us"]["horizontal"] * 1e-6 * fs))
    porch = int(round(9.45e-6 * fs))          # the standard's active-video start
    video = np.full(line_samples, PARAMS["blanking_hz"])
    video[:sync] = PARAMS["sync_tip_hz"]
    hz_ire = (PARAMS["peak_white_hz"] - PARAMS["blanking_hz"]) / 100.0
    active = line_samples - porch
    edges = np.linspace(porch, line_samples, len(levels) + 1).astype(int)
    for level, a, b in zip(levels, edges[:-1], edges[1:]):
        video[a:b] = PARAMS["blanking_hz"] + level * hz_ire
    # a smooth transition so the FM has finite bandwidth
    k = int(round(PARAMS["sync_transition_us"] * 1e-6 * fs)) * 2 + 1
    return np.convolve(video, np.ones(k) / k, mode="same")


def synthetic_tap(levels, fields, chroma=True, seed=0, phase0=0.0):
    """A capture of `fields` fields of a static pattern: NTSC-like vertical
    interval (equalizing, broad, equalizing), then the pattern lines, FM
    modulated with a continuously advancing carrier phase; optional
    colour-under tone with the format's 90-degree per-line rotation."""
    fs = RATE
    line_samples = int(round(PARAMS["line_period_s"] * fs))
    half = line_samples // 2
    eq = int(round(PARAMS["pulse_widths_us"]["equalizing"] * 1e-6 * fs))
    broad = int(round(PARAMS["pulse_widths_us"]["field"] * 1e-6 * fs))
    line = _pattern_video(levels, line_samples)
    lines_per_field = PARAMS["field_lines"]
    frames = []
    for k in range(fields):
        n_lines = lines_per_field[k % 2]
        freq = np.tile(line, n_lines).astype(np.float64)
        # the second field's interval starts half a line in
        offset = 0 if k % 2 == 0 else half
        interval = np.full(9 * line_samples, PARAMS["blanking_hz"])
        for i in range(6):
            interval[i * half:i * half + eq] = PARAMS["sync_tip_hz"]
        for i in range(6, 12):
            interval[i * half:i * half + broad] = PARAMS["sync_tip_hz"]
        for i in range(12, 18):
            interval[i * half:i * half + eq] = PARAMS["sync_tip_hz"]
        freq[offset:offset + len(interval)] = interval
        frames.append(freq)
    freq = np.concatenate(frames)
    phase = 2 * np.pi * np.cumsum(freq) / fs + phase0
    x = np.cos(phase)
    if chroma:
        t = np.arange(len(freq)) / fs
        # colour-under carrier rotated 90 degrees per line, a chroma
        # sideband, GATED to the active line as a real colour-under is: the
        # composite's chroma is zero through the sync pulse and the porches,
        # so the per-line phase step lands where the amplitude is zero. A
        # continuous tone with a phase step at the boundary is a wideband
        # splatter that flips sign between same-parity fields and had cost
        # the record fixture its own line-to-line coherence (beta 0.26 at
        # 1.0-1.25 MHz with no noise in it).
        line_index = (np.arange(len(freq)) // line_samples)
        rotation = np.pi / 2 * line_index
        cu = PARAMS["colour_under_hz"]
        gate_line = np.zeros(line_samples)
        active_start = int(round(9.45e-6 * fs))   # the standard's active-video start
        gate_line[active_start:] = 1.0
        k = int(round(PARAMS["sync_transition_us"] * 1e-6 * fs)) * 4 + 1
        gate_line = np.convolve(gate_line, np.ones(k) / k, mode="same")
        gate = np.tile(gate_line, len(freq) // line_samples + 1)[:len(freq)]
        x = x + gate * (0.25 * np.cos(2 * np.pi * cu * t + rotation)
                        + 0.12 * np.cos(2 * np.pi * (cu + 0.3e6) * t + rotation))
    return x


def planted_channel(freqs):
    """A complex channel: a one-pole roll-off, a notch, and 60 ns of extra
    group delay across the band (dispersion), i.e. NOT minimum phase."""
    w = 1j * freqs / 5.0e6
    H = 1.0 / (1.0 + w)
    notch = 1.0 - 0.7 / (1.0 + ((freqs - 2.7e6) / 0.15e6) ** 2)
    dispersion = np.exp(-1j * 2 * np.pi * freqs ** 2 / (2 * 6.5e6) * 60e-9)
    return H * notch * dispersion


def apply_channel(x, channel, delay_samples, noise, rng, wobble_samples=0.0):
    """Filter by the channel, delay, add noise, and let the time base wander
    slowly (a sinusoidal delay modulation over the capture)."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n)))
    f = np.fft.rfftfreq(nfft, 1.0 / RATE)
    X = np.fft.rfft(x, nfft) * channel(f) * np.exp(-2j * np.pi * f * delay_samples / RATE)
    y = np.fft.irfft(X, nfft)[:n]
    if wobble_samples:
        t = np.arange(n)
        wander = wobble_samples * np.sin(2 * np.pi * t / n * 3.0)
        y = np.interp(t - wander, t, y)
    return (y + noise * rng.standard_normal(n)).astype(np.float32)


@pytest.fixture(scope="module")
def taps():
    rng = np.random.default_rng(3)
    levels = (0, 20, 45, 75, 100, 60, 30, 10)
    record = synthetic_tap(levels, fields=8, seed=1).astype(np.float32)
    # other frames: the same content with a different carrier phase history
    later = synthetic_tap(levels, fields=8, seed=2, phase0=1.3)
    playback = apply_channel(later, planted_channel, 37.4, 0.02, rng, wobble_samples=1.5)
    return record, playback


class TestRecovery:
    def test_magnitude_and_phase_recovered(self, taps):
        record, playback = taps
        transfers = tt.complex_transfer(record, playback, RATE, PARAMS)
        assert set(transfers) == {0, 1}
        for t in transfers.values():
            f = t.freqs
            planted = planted_channel(f)
            # the same convention on the planted channel: excess phase over the reference band
            planted_conv, _ = tt._phase_convention(planted, f, PARAMS["reference_band_hz"])
            ref = (f >= PARAMS["reference_band_hz"][0]) & (f < PARAMS["reference_band_hz"][1])
            good = (t.se_log_magnitude < 0.05) & (f > 1.3e6) & (f < 6.0e6)
            assert good.sum() > 100
            measured_db = tt.DB_PER_NEPER * (t.log_magnitude - t.log_magnitude[ref].mean())
            planted_db = tt.DB_PER_NEPER * (np.log(np.abs(planted)) - np.log(np.abs(planted))[ref].mean())
            error_db = measured_db[good] - planted_db[good]
            assert np.sqrt(np.mean(error_db ** 2)) < 0.5, np.sqrt(np.mean(error_db ** 2))
            phase_error = np.angle(np.exp(1j * (t.phase[good] - np.angle(planted_conv)[good])))
            assert np.degrees(np.sqrt(np.mean(phase_error ** 2))) < 6.0, np.degrees(np.sqrt(np.mean(phase_error ** 2)))
            # The notch is seen in magnitude. It is admitted on its OWN bar:
            # the planted channel takes 7 dB out of the playback there, so the
            # coherence falls to 0.02-0.5 and the estimator's honest error bar
            # in the notch is 0.4-0.9 dB. Two or three bins reach the 0.05 Np
            # bar the rest of this test uses; nine reach 0.10 Np. That is a
            # measured ceiling of the instrument, not a target.
            notch = (t.se_log_magnitude < 0.10) & (f > 2.6e6) & (f < 2.8e6)
            assert notch.sum() > 4, notch.sum()
            assert measured_db[notch].mean() < -4.0, measured_db[notch].mean()
            assert t.fields >= 2 and t.lines > 300

    def test_identity_is_unity(self, taps):
        record, _ = taps
        rng = np.random.default_rng(5)
        copy = (record + 0.02 * rng.standard_normal(len(record))).astype(np.float32)
        transfers = tt.complex_transfer(record, copy, RATE, PARAMS)
        for t in transfers.values():
            good = t.se_log_magnitude < 0.05
            # The mask admits bins whose stated bar is as wide as 0.05 Np =
            # 0.43 dB, so unity can sit 0.87 dB away at two sigma. Measured:
            # 0.739 dB, both heads. A measured ceiling, not a target - the
            # claim being tested is the one below it.
            assert np.abs(tt.DB_PER_NEPER * t.log_magnitude[good]).max() < 0.8
            # THE CLAIM: the error bar the estimator reports must bound the
            # error it makes, or a mask on that bar admits exactly the bins
            # where the estimate is worst. Measured: 1.21 and 1.17.
            z = np.abs(t.log_magnitude[good]) / t.se_log_magnitude[good]
            assert np.sqrt(np.mean(z ** 2)) < 1.5, np.sqrt(np.mean(z ** 2))
            # the phase on the phase's OWN bar: log H = log|H| + i arg H, so
            # 0.05 radians is the same tolerance as 0.05 nepers
            phase_good = t.se_phase < 0.05
            assert phase_good.sum() > 100
            assert np.degrees(np.abs(t.phase[phase_good])).max() < 5.0

    def test_non_minimum_phase_is_detected(self, taps):
        record, playback = taps
        transfers = tt.complex_transfer(record, playback, RATE, PARAMS)
        separation = tt.separate({"synthetic": transfers})
        attribution = tt.stage_attribution(separation[0], PARAMS)
        assert attribution["fitted"]
        # 60 ns of dispersion across the band is far beyond the minimum
        # phase of the roll-off and the notch
        assert attribution["non_minimum_phase_rms_deg"] > 8.0


class TestWitnesses:
    def test_planted_content_dependence_is_flagged(self, taps):
        record, playback = taps
        rng = np.random.default_rng(9)
        # a second "pattern" whose channel differs above 4 MHz by 3 dB
        def other(freqs):
            return planted_channel(freqs) * (1.0 - 0.3 / (1.0 + np.exp(-(freqs - 4.5e6) / 0.2e6)))
        levels = (0, 20, 45, 75, 100, 60, 30, 10)
        later = synthetic_tap(levels, fields=8, seed=4, phase0=0.4)
        playback_b = apply_channel(later, other, 21.0, 0.02, rng, wobble_samples=1.0)
        transfers = {"a": tt.complex_transfer(record, playback, RATE, PARAMS),
                     "b": tt.complex_transfer(record, playback_b, RATE, PARAMS)}
        separation = tt.separate(transfers)
        bands = [(2.0e6, 3.0e6), (5.0e6, 6.0e6)]
        rows = {r["low_hz"]: r for r in tt.band_separation(separation[0], bands)}
        assert rows[5.0e6]["content_excess_db"] > 1.5
        assert rows[5.0e6]["chi2_magnitude"] > 10.0
        assert rows[2.0e6]["content_excess_db"] < 0.5

    def test_misalignment_is_refused(self, taps):
        record, _ = taps
        rng = np.random.default_rng(11)
        unrelated = synthetic_tap((100, 0, 100, 0, 100, 0, 100, 0), fields=8, seed=7)
        unrelated = apply_channel(unrelated, planted_channel, 12.0, 0.02, rng)
        with pytest.raises(ValueError):
            tt.complex_transfer(record, unrelated, RATE, PARAMS)

    def test_line_pair_rejects_a_lag_beyond_the_guard(self, taps):
        record, playback = taps
        grid = tt.make_grid(RATE, PARAMS)
        fields = tt.locate_fields(record, RATE, PARAMS)
        number = 100
        start = fields[2].lines[number]
        assert tt.align_line_pair(record, start, record, start + grid.lag_guard * 3, grid) is None
        pair = tt.align_line_pair(record, start, record, start + 5, grid)
        # `delay_samples` is the CORRECTION, the offset to add to b's start to
        # bring b's window onto a's content, so a window read five samples
        # later - which holds content five samples earlier - reads -5
        assert pair is not None and abs(pair.delay_samples + 5.0) < 0.1


class TestStructure:
    def test_fields_and_parity_are_located(self, taps):
        record, _ = taps
        fields = tt.locate_fields(record, RATE, PARAMS)
        assert len(fields) >= 6
        assert [f.parity for f in fields[:4]] in ([0, 1, 0, 1], [1, 0, 1, 0])
        for f in fields:
            assert PARAMS["first_usable_line"] in f.lines
            assert PARAMS["last_usable_line"] in f.lines

    def test_parameters_are_the_decoders_own(self):
        assert PARAMS["sync_tip_hz"] == pytest.approx(3.4e6)
        assert PARAMS["peak_white_hz"] == pytest.approx(4.4e6)
        assert PARAMS["fm_band_hz"][0] == pytest.approx(1.2e6)
        assert PARAMS["colour_under_hz"] == pytest.approx(40 * 525 * 30 / 1.001)
        assert PARAMS["first_usable_line"] == 7
        assert PARAMS["last_usable_line"] == 252
        assert tt.sample_rate_from_name("zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-rec.flac") == 50e6
