"""The chroma transient improvement's measured sweep radius.

CTI's reach used to be a knob (`--cti_width`, fixed at 2 subcarrier
cycles). The right reach is a property of the tape and the machine - it is
half the span of the transition the chroma path actually produces - so it
is measured per field from the colour burst, which is a gated subcarrier at
a fixed place on every line and therefore the path's own step response.
"""

import types

import numpy as np
import pytest

from vhsdecode import chroma as c


SAMPLES_PER_LINE = 910
LINES = 263
OUTFREQ = 4 * 315e6 / 88 / 1e6            # 4fsc NTSC, samples per microsecond


def _field(cti_width="auto", lines=LINES):
    return types.SimpleNamespace(
        outlinelen=SAMPLES_PER_LINE,
        outlinecount=lines,
        lineoffset=0,
        usectooutpx=lambda x: x * OUTFREQ,
        rf=types.SimpleNamespace(
            SysParams={"colorBurstUS": (5.3, 7.8)},
            options=types.SimpleNamespace(cti_width=cti_width, cti_mix=1.0),
        ),
    )


def _burst_field(rise_samples, field=None, lines=LINES):
    """A field buffer whose burst envelope rises over `rise_samples`.

    Sized EXACTLY `outlinecount * outlinelen`, as the decoder's own buffer
    is - a fixture one line longer hides the overrun this module once had.
    """
    field = field or _field(lines=lines)
    burst_start, burst_end = c.get_burst_area(field)
    t = np.arange(SAMPLES_PER_LINE)
    k = 4.394 / rise_samples                       # logistic 10%-90% width
    envelope = 1.0 / (1.0 + np.exp(-k * (t - (burst_start + rise_samples))))
    envelope[burst_end:] = 0.0
    # a real subcarrier at 4fsc: the phase advances 90 degrees per sample
    line = (envelope * np.cos(np.pi / 2 * t)).astype(np.float32)
    return field, np.tile(line, lines)


class TestMeasuredRise:
    @pytest.mark.parametrize("planted", [8.0, 12.0, 16.0, 24.0])
    def test_recovers_a_planted_rise(self, planted):
        field, uphet = _burst_field(planted)
        measured = c.measured_chroma_rise(field, uphet)
        assert measured == pytest.approx(planted, abs=0.6)

    def test_the_buffer_is_not_overrun(self):
        """THE bug this test exists for. `uphet` is exactly
        `outlinecount * outlinelen` long and CTI starts a line into it, so
        the rows available are what remain AFTER that offset. Taking
        `outlinecount` rows from there overruns by exactly the offset, and
        the measurement then declined on every real field while a fixture
        built one line too long still passed."""
        field, uphet = _burst_field(16.0)
        assert len(uphet) == field.outlinecount * field.outlinelen
        assert c.measured_chroma_rise(field, uphet) is not None

    def test_the_lead_in_reaches_before_the_burst(self):
        """`get_burst_area` leaves only a few samples ahead of the burst -
        enough to gate it, not enough to see it start. A rise of order 17
        samples never reaches its own 10% point in that window."""
        field, uphet = _burst_field(17.0)
        assert c.measured_chroma_rise(field, uphet) is not None

    def test_a_flat_field_measures_nothing(self):
        field = _field()
        assert c.measured_chroma_rise(
            field, np.zeros(LINES * SAMPLES_PER_LINE, dtype=np.float32)) is None

    def test_a_short_buffer_measures_nothing(self):
        field = _field()
        assert c.measured_chroma_rise(
            field, np.zeros(4 * SAMPLES_PER_LINE, dtype=np.float32)) is None


class TestSweepRadius:
    @pytest.mark.parametrize("planted,expected", [(8.0, 8), (12.0, 12),
                                                  (16.0, 16)])
    def test_radius_follows_the_measured_rise(self, planted, expected):
        field, uphet = _burst_field(planted)
        assert c.chroma_sweep_radius(field, uphet) == expected

    def test_an_explicit_width_overrides_the_measurement(self):
        field, uphet = _burst_field(16.0, field=_field(cti_width=2))
        assert c.chroma_sweep_radius(field, uphet) == 8

    def test_falls_back_when_nothing_can_be_measured(self):
        field = _field()
        flat = np.zeros(LINES * SAMPLES_PER_LINE, dtype=np.float32)
        # the historical fixed default, so a field CTI cannot measure is
        # sharpened exactly as it always was rather than not at all
        assert c.chroma_sweep_radius(field, flat) == 8

    def test_the_radius_is_always_a_whole_subcarrier_cycle(self):
        """CTI reads `(x[s], x[s-1])` as a quadrature pair and the 4fsc
        frame rotates 90 degrees per sample, so `x[s +/- R]` is only in the
        same frame when R is a multiple of 4. A radius that is not is a
        rotated vector, not a neighbour."""
        for planted in (6.0, 9.0, 11.0, 14.0, 19.0, 26.0):
            field, uphet = _burst_field(planted)
            radius = c.chroma_sweep_radius(field, uphet)
            assert radius % c.SUBCARRIER_QUADRATURE == 0, planted

    def test_the_radius_is_bounded(self):
        for planted in (4.0, 40.0):
            field, uphet = _burst_field(planted)
            radius = c.chroma_sweep_radius(field, uphet)
            assert c.SUBCARRIER_QUADRATURE <= radius <= (
                c.MAX_SWEEP_CYCLES * c.SUBCARRIER_QUADRATURE), planted


class TestShapedSharpener:
    """The sharpener's SLOPE, derived from the whole burst envelope.

    A single radius says how far the operator reaches; it says nothing
    about the shape of the emphasis. The passes now have distinct radii -
    a small filter bank - with weights solved jointly against the roll-off
    measured on the burst.
    """

    @staticmethod
    def _rf_field(**overrides):
        from scipy.signal import butter

        sos = butter(4, [60e3 / 20e6, 1.2e6 / 20e6], btype="bandpass",
                     output="sos")
        base = _field(**overrides)
        base.rf.SysParams["outfreq"] = OUTFREQ
        base.rf.DecoderParams = {"color_under_carrier": 629370.63}
        base.rf.Filters = {"FVideoBurst": sos}
        base.rf.freq_hz = 40e6
        return base

    def test_the_decoders_own_filter_is_measurable_and_rolls_off(self):
        """It has to be divided out, not corrected for: inverting it would
        undo the separation filter that keeps the chroma apart."""
        field = self._rf_field()
        response = c.decoder_envelope_response(
            field.rf, np.array([0.0, 100e3, 571e3]))
        assert response is not None
        assert response[0] > 0.95                     # flat at DC
        assert response[2] < response[0]              # rolled off at its edge

    def test_the_bank_has_distinct_radii(self):
        """Two passes at the same radius are one pass with twice the
        weight, and they make the solve's basis rank-deficient."""
        field, uphet = _burst_field(16.0, field=self._rf_field())
        radii, weights = c.sharpener_passes(field, uphet, 16)
        assert len(set(int(r) for r in radii)) == len(radii)
        assert len(weights) == len(radii)

    def test_every_radius_is_a_whole_subcarrier_cycle(self):
        field, uphet = _burst_field(16.0, field=self._rf_field())
        radii, _ = c.sharpener_passes(field, uphet, 16)
        for radius in radii:
            assert int(radius) % c.SUBCARRIER_QUADRATURE == 0

    def test_the_sharpener_never_pulls_the_other_way(self):
        field, uphet = _burst_field(16.0, field=self._rf_field())
        _, weights = c.sharpener_passes(field, uphet, 16)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= c.SHARPENER_WEIGHT_LIMIT)

    def test_it_falls_back_to_the_historical_decay_when_unmeasurable(self):
        """A field whose burst cannot be read is sharpened exactly as it
        always was, rather than not at all."""
        field = self._rf_field()
        flat = np.zeros(LINES * SAMPLES_PER_LINE, dtype=np.float32)
        radii, weights = c.sharpener_passes(field, flat, 16)
        expected = [c.SHARPENER_DECAY ** p for p in range(len(radii))]
        assert np.allclose(weights, expected)

    def test_the_flat_top_comes_from_the_spec_not_a_fraction(self):
        """The measurement window reaches back into the breezeway so the
        rise can be seen, which puts the middle third on the rising EDGE.
        Normalising there divides by a point on the transition - a planted
        burst read 12.8 at its own flat top before this was fixed."""
        field, uphet = _burst_field(16.0, field=self._rf_field())
        profile = c.burst_envelope(field, uphet)
        assert profile is not None
        measured, first = profile
        span = c._spec_flat_span(field, first, len(measured))
        assert span is not None
        burst_start_us = field.rf.SysParams["colorBurstUS"][0]
        assert first + span.start > burst_start_us * OUTFREQ

    def test_the_response_reports_only_bins_with_evidence(self):
        """Bins where the expected spectrum carries no energy are the ones
        a ratio would report as a deep roll-off measured on nothing."""
        field, uphet = _burst_field(16.0, field=self._rf_field())
        measured = c.chroma_path_response(field, uphet)
        assert measured is not None
        _, log_response, usable = measured
        assert not usable[0]                       # DC never speaks
        assert np.count_nonzero(usable) < len(usable)
        assert np.all(log_response[~usable] == 0.0)
