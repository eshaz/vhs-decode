"""The sync pulse shape as ONE component on all three axes.

Ethan, twice: "the shape is a component with all parts, amplitude,
frequency, time", and "The luma and chroma are band limited, so the data
outside this bandlimited area is noise, that can use as the differential".
"""

import numpy as np
import pytest

from vhsdecode.models import sync_shape as ss

FS = 4 * 315e6 / 88


def _shape(samples=52):
    t = np.arange(samples)
    return (-2.4 * np.exp(-t / 18.0)
            + 0.5 * np.exp(-t / 7.0) * np.cos(2 * np.pi * 2.43e6 * t / FS))


def test_all_three_axes_are_reported():
    """The directive itself: amplitude, frequency AND time, from one fit."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    for axis in ("amplitude", "frequency_hz", "group_delay_s"):
        assert len(np.asarray(got[axis])) == got["modes"]
    assert np.isfinite(got["amplitude_rms"])
    assert np.isfinite(got["centroid_hz"])
    assert np.isfinite(got["position_samples"])


def test_the_axes_are_readings_of_one_object_not_three_fits():
    """They share a frame: one coefficient per mode, one frequency per
    mode, one delay per mode. The previous treatment fitted a relaxation
    for its amplitude and a ring for its frequency, and they disagreed
    because nothing made them share one."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    assert len(got["coefficients"]) == len(got["frequency_hz"])
    assert np.allclose(got["amplitude"], np.abs(got["coefficients"]))


def test_the_band_limit_is_a_format_constant_and_sets_the_size():
    narrow = ss.shape_components(_shape(), FS, 1.5e6)
    wide = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    assert wide["modes"] > narrow["modes"]
    assert wide["frequency_hz"][-1] <= ss.VHS_LUMA_BAND_HZ
    assert narrow["frequency_hz"][-1] <= 1.5e6


@pytest.mark.parametrize("tone_hz,belongs", [
    (0.5e6, "shape"), (2.5e6, "shape"), (5.5e6, "noise"), (6.9e6, "noise"),
])
def test_the_band_routes_content_to_shape_or_noise(tone_hz, belongs):
    """THE STATEMENT, made operational: what is inside the band is the
    shape and what is outside it is noise, BY CONSTRUCTION rather than by
    a chosen threshold."""
    base = _shape()
    tone = 0.3 * np.cos(2 * np.pi * tone_hz * np.arange(len(base)) / FS)
    plain = ss.shape_components(base, FS, ss.VHS_LUMA_BAND_HZ)
    got = ss.shape_components(base + tone, FS, ss.VHS_LUMA_BAND_HZ)
    moved_in_band = float(np.sqrt(np.mean(
        (got["in_band"] - plain["in_band"]) ** 2)))
    if belongs == "shape":
        assert moved_in_band > got["noise_rms"]
    else:
        assert got["noise_rms"] > 0.5 * moved_in_band


def test_a_band_limited_shape_leaves_no_noise():
    """Nothing outside the band means nothing in the noise channel - so a
    non-zero noise reading is a measurement, not a residue."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    # 1e-6 IRE, which is 25000x below the 0.025 IRE floor two independent
    # measurements of this porch differ by - the basis is finite, so the
    # residual is small rather than zero, and the bar is what the
    # measurement can distinguish
    assert got["noise_rms"] < 1e-5


def test_in_band_and_noise_reconstruct_the_input():
    """The split is exact: nothing is lost between the two channels."""
    values = _shape()
    got = ss.shape_components(values, FS, ss.VHS_LUMA_BAND_HZ)
    assert np.allclose(got["in_band"] + got["out_of_band"], values, atol=1e-9)


def test_the_effective_rank_is_far_below_the_mode_count():
    """The shape has real degrees of freedom - more than the two or three
    a pole enumeration allowed, fewer than the basis offers."""
    got = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    assert 2.0 < got["effective_rank"] < got["modes"]


def test_the_time_axis_moves_when_the_shape_moves():
    """A shape shifted along the segment must read a different position -
    otherwise the time axis is decoration."""
    values = _shape()
    # a REAL shift, not np.roll - rolling wraps the tail back to the start
    # and moves the centroid the other way, which is a property of the
    # test rather than of the shape
    shifted = np.concatenate([np.full(6, values[-1]), values[:-6]])
    here = ss.shape_components(values, FS, ss.VHS_LUMA_BAND_HZ)
    there = ss.shape_components(shifted, FS, ss.VHS_LUMA_BAND_HZ)
    moved = there["position_samples"] - here["position_samples"]
    assert moved == pytest.approx(6.0, abs=1.0)


def test_the_frequency_axis_moves_when_the_content_moves():
    t = np.arange(52)
    low = np.cos(2 * np.pi * 0.4e6 * t / FS)
    high = np.cos(2 * np.pi * 2.4e6 * t / FS)
    assert (ss.shape_components(high, FS, ss.VHS_LUMA_BAND_HZ)["centroid_hz"]
            > ss.shape_components(low, FS, ss.VHS_LUMA_BAND_HZ)["centroid_hz"])


def test_the_differential_is_subtractive_on_every_axis():
    """A DIFFERENCE, never a ratio - spectral division manufactured
    phantoms in this tree twice."""
    a = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    b = ss.shape_components(_shape() * 1.10, FS, ss.VHS_LUMA_BAND_HZ)
    got = ss.differential(b, a)
    assert np.allclose(got["amplitude"],
                       np.asarray(b["amplitude"]) - np.asarray(a["amplitude"]))
    assert np.allclose(got["residual"],
                       np.asarray(b["in_band"]) - np.asarray(a["in_band"]))


def test_the_differential_carries_the_noise_channel_too():
    """The second half of the statement: the noise is not discarded, it is
    itself a channel of the differential. Two shapes agreeing in band and
    differing out of it have a real difference."""
    values = _shape()
    t = np.arange(len(values))
    noisy = values + 0.4 * np.cos(2 * np.pi * 6.5e6 * t / FS)
    a = ss.shape_components(values, FS, ss.VHS_LUMA_BAND_HZ)
    b = ss.shape_components(noisy, FS, ss.VHS_LUMA_BAND_HZ)
    got = ss.differential(b, a)
    assert abs(got["noise_rms"]) > 0.05
    assert float(np.sqrt(np.mean(np.asarray(got["out_of_band"]) ** 2))) > 0.05


def test_the_differential_refuses_a_mismatched_basis():
    """A comparison is only a comparison when both sides are computed over
    the same set."""
    a = ss.shape_components(_shape(), FS, ss.VHS_LUMA_BAND_HZ)
    b = ss.shape_components(_shape(), FS, 1.5e6)
    with pytest.raises(ValueError, match="same basis"):
        ss.differential(a, b)


# --------------------------------------------------------------------------
# The time axis MEASURED rather than derived from the amplitude:
# `transit_delay` and the two-transient pair `edge_pair`.
# --------------------------------------------------------------------------

from vhsdecode.models import pair_dimension as pdim   # noqa: E402

SEGMENT = 130          # samples, the span one sync pulse needs at 4fsc
BLANKING = 0.0
TIP_IRE = pdim.SYNC_DEPTH_IRE


def _pulse(fall=19.0, rise=None, samples=SEGMENT):
    """A sync pulse of the SPECIFIED width and rise, in IRE.

    SMPTE 170M table 2 through `pair_dimension`: 4.7 us wide, -40 IRE, 140
    ns of 10 to 90 per cent rise. A hyperbolic tangent covers 10 to 90 per
    cent in 2.197 of its own scale, which is where the divisor comes from
    rather than from a fit.
    """
    if rise is None:
        rise = fall + pdim.SYNC_WIDTH_S * FS
    scale = pdim.SYNC_RISE_S * FS / (2.0 * np.arctanh(0.8))
    t = np.arange(samples, dtype=np.float64)
    gate = (0.5 * (1.0 + np.tanh((t - fall) / scale))
            * 0.5 * (1.0 + np.tanh((rise - t) / scale)))
    return BLANKING + TIP_IRE * gate


def _minimum_phase_filtered(values, pole=0.55):
    """One real pole inside the unit circle: minimum phase by construction,
    so whatever delay it imposes is the delay its magnitude alone fixes."""
    out = np.empty_like(values)
    previous = values[0]
    for index, sample in enumerate(values):
        previous = (1.0 - pole) * sample + pole * previous
        out[index] = previous
    return out


def test_a_planted_delay_comes_back():
    """The instrument on a known answer. Nothing else it says means
    anything until this does. Measured, the error over -20 to +50 ns is
    0.0000, 0.0050, 0.0095, 0.0055, 0.0210 and 0.0825 ns, so the bar below
    is loose by an order of magnitude and a scale error of the kind a
    window introduces would break it at every size."""
    reference = _pulse()
    for planted in (-20.0e-9, -2.0e-9, 0.0, 2.0e-9, 5.0e-9, 12.0e-9,
                    20.0e-9, 50.0e-9):
        moved = _pulse(fall=19.0 + planted * FS)
        got = ss.transit_delay(moved, reference, FS, ss.VHS_LUMA_BAND_HZ)
        assert got["delay_s"] == pytest.approx(planted, abs=0.2e-9)
        assert abs(got["delay_s"] - planted) < 0.01 * abs(planted) + 1e-12


def test_a_planted_delay_is_all_excess():
    """A pure delay is an ALL-PASS: its magnitude is flat, so the
    minimum-phase partner of that magnitude is no delay at all and the
    whole of the reading has to land in the excess term."""
    reference = _pulse()
    moved = _pulse(fall=19.0 + 20.0e-9 * FS)
    got = ss.transit_delay(moved, reference, FS, ss.VHS_LUMA_BAND_HZ)
    assert abs(got["minimum_phase_delay_s"]) < 0.15 * abs(got["delay_s"])
    assert got["excess_delay_s"] == pytest.approx(got["delay_s"], rel=0.2)


def test_a_minimum_phase_filter_leaves_almost_no_excess():
    """The other side of the same split, and the check that the two terms
    are not one term counted twice. A single real pole is minimum phase, so
    its delay is fixed by its magnitude and the excess must nearly vanish -
    where a pure delay of similar size lands entirely in the excess."""
    reference = _pulse()
    filtered = _minimum_phase_filtered(reference)
    got = ss.transit_delay(filtered, reference, FS, ss.VHS_LUMA_BAND_HZ)
    # measured: delay 63.386 ns, of which the magnitude alone fixes 54.043
    # and 9.342 is left over - 85 per cent accounted for, where the pure
    # delay above leaves 101 per cent
    assert abs(got["minimum_phase_delay_s"]) > 0.6 * abs(got["delay_s"])
    assert abs(got["excess_delay_s"]) < 0.4 * abs(got["delay_s"])


def test_the_response_is_complex():
    """Ethan's standing rule. A response given as a magnitude cannot carry
    a delay, and the delay is the whole subject here."""
    got = ss.transit_delay(_minimum_phase_filtered(_pulse()), _pulse(), FS,
                           ss.VHS_LUMA_BAND_HZ)
    response = np.asarray(got["response"])
    assert np.iscomplexobj(response)
    assert float(np.max(np.abs(response.imag))) > 1e-6


def test_a_dispersive_response_is_refused_as_a_delay():
    """`is_a_delay` is the guard that stops a scalar being quoted where the
    phase is not linear. An all-pass with a phase quadratic in frequency
    has no delay to speak of and a large residual."""
    reference = _pulse()
    spectrum = np.fft.rfft(reference - reference.mean())
    frequencies = np.fft.rfftfreq(len(reference), 1.0 / FS)
    curved = np.fft.irfft(spectrum * np.exp(
        -2j * np.pi * (frequencies / 1e6) ** 2 * 12e-9), n=len(reference))
    curved = curved + float(np.mean(reference))
    got = ss.transit_delay(curved, reference, FS, ss.VHS_LUMA_BAND_HZ)
    assert not got["is_a_delay"]
    assert got["equivalent_residual_delay_s"] > abs(got["delay_s"])
    plain = ss.transit_delay(_pulse(fall=19.0 + 20.0e-9 * FS), reference, FS,
                             ss.VHS_LUMA_BAND_HZ)
    assert plain["is_a_delay"]


def test_the_transients_are_found_and_a_mis_framed_segment_is_refused():
    where = ss.locate_transients(_pulse(), FS)
    assert where["fall"] == pytest.approx(19, abs=2)
    assert where["width_s"] == pytest.approx(pdim.SYNC_WIDTH_S, rel=0.05)
    assert where["fall"] < where["midpoint"] < where["rise"]
    half = _pulse()[:60]                       # no trailing edge in view
    with pytest.raises(ValueError):
        ss.locate_transients(half, FS)


def test_the_pair_calls_a_delay_a_delay():
    """A displacement of the whole pulse moves both transients the same
    way: the common term carries it and the width term is empty."""
    reference = _pulse()
    planted = 12.0e-9
    moved = _pulse(fall=19.0 + planted * FS)
    got = ss.edge_pair(moved, reference, FS, ss.VHS_LUMA_BAND_HZ)
    assert got["common_delay_s"] == pytest.approx(planted, abs=0.1e-9)
    # THE GUARD THAT CAUGHT THE WINDOW. A displacement moves the two
    # transients by the SAME amount, so a method that reads them unequally
    # is putting a width change where there is none. Measured, the two
    # differ by 0.16 ns on a 12 ns shift; through a Hann window they
    # differed by 1.7 ns, which is larger than several of the width
    # changes this pair is asked to report on real tape.
    assert abs(got["width_change_s"]) < 0.05 * planted


def test_the_pair_refuses_to_call_a_width_change_a_delay():
    """THE MEASUREMENT THE PAIR EXISTS FOR. A pulse widened at both ends
    moves its transients in OPPOSITE directions. One number from the whole
    pulse cannot tell that from a delay; two disjoint transients can, and
    the same confusion has already cost this lane a round once, when a fall
    time stood in for a rise nobody had measured on its own."""
    reference = _pulse()
    widen = 6.0e-9
    wider = _pulse(fall=19.0 - widen * FS,
                   rise=19.0 + pdim.SYNC_WIDTH_S * FS + widen * FS)
    got = ss.edge_pair(wider, reference, FS, ss.VHS_LUMA_BAND_HZ)
    assert got["width_change_s"] == pytest.approx(2.0 * widen, rel=0.3)
    assert abs(got["common_delay_s"]) < 0.25 * abs(got["width_change_s"])


def test_the_pair_separates_a_delay_and_a_width_carried_together():
    """Both at once, which is what the SP recording turned out to hold."""
    reference = _pulse()
    delay, widen = 10.0e-9, 5.0e-9
    both = _pulse(fall=19.0 + (delay - widen) * FS,
                  rise=19.0 + pdim.SYNC_WIDTH_S * FS + (delay + widen) * FS)
    got = ss.edge_pair(both, reference, FS, ss.VHS_LUMA_BAND_HZ)
    assert got["common_delay_s"] == pytest.approx(delay, abs=1.0e-9)
    assert got["width_change_s"] == pytest.approx(2.0 * widen, rel=0.3)


def test_the_two_transients_are_measured_on_disjoint_samples():
    """A pair that shares samples is not a pair."""
    reference = _pulse()
    got = ss.edge_pair(reference, reference, FS, ss.VHS_LUMA_BAND_HZ)
    cut = got["transients"]["midpoint"]
    assert len(got["leading"]["frequency_hz"]) == cut // 2 + 1
    assert len(got["trailing"]["frequency_hz"]) == (len(reference) - cut) // 2 + 1


def test_the_band_limit_decides_which_bins_carry_the_shape():
    """The same format constant the shape-and-noise split already uses: the
    power above the luma band is noise by construction, and a bin below it
    carries no shape and no usable phase."""
    reference = _pulse()
    got = ss.transit_delay(reference, reference, FS, ss.VHS_LUMA_BAND_HZ)
    frequencies = np.asarray(got["frequency_hz"])
    carried = np.asarray(got["carried"])
    assert carried.any()
    assert frequencies[carried].max() <= ss.VHS_LUMA_BAND_HZ
    assert frequencies[carried].min() > 0.0
    narrow = ss.transit_delay(reference, reference, FS, 1.5e6)
    assert int(narrow["bins"]) < int(got["bins"])


def test_a_segment_with_no_out_of_band_room_is_refused():
    """The noise floor comes from above the band limit, so a segment whose
    Nyquist sits inside the band has nothing to set it with."""
    reference = _pulse()
    with pytest.raises(ValueError, match="out-of-band"):
        ss.transit_delay(reference, reference, FS, FS)


def test_shapes_of_different_length_are_refused():
    with pytest.raises(ValueError, match="same segment"):
        ss.transit_delay(_pulse()[:100], _pulse(), FS, ss.VHS_LUMA_BAND_HZ)


def test_a_level_step_across_the_segment_does_not_move_the_delay():
    """The joining line is removed, so framing the segment on a back porch
    that has not settled costs nothing. On the real profiles that line is
    2.3 to 6.5 IRE against noise amplitudes of 0.06 to 0.24, so this is the
    ordinary case rather than a corner of it."""
    reference = _pulse()
    planted = 8.0e-9
    moved = _pulse(fall=19.0 + planted * FS)
    # the synthetic pulse's own out-of-band content sets the scale
    # `ends_match` compares against, and on this shape that is 1.35 IRE, so
    # the ramp is taken well past it to make the flag change
    ramp = np.linspace(0.0, 25.0, len(reference))
    clean = ss.transit_delay(moved, reference, FS, ss.VHS_LUMA_BAND_HZ)
    stepped = ss.transit_delay(moved + ramp, reference + ramp, FS,
                               ss.VHS_LUMA_BAND_HZ)
    assert stepped["delay_s"] == pytest.approx(clean["delay_s"], abs=0.05e-9)
    assert stepped["end_step"] == pytest.approx(25.0, abs=0.1)
    assert not stepped["ends_match"]
    assert clean["ends_match"]


def test_the_two_transients_agree_on_a_pure_delay():
    """The pair's own consistency check, and the one that would have caught
    the window: the leading and trailing readings of a shifted pulse must
    be the same number, because there is only one shift."""
    reference = _pulse()
    for planted in (2.0e-9, 12.0e-9, 20.0e-9):
        got = ss.edge_pair(_pulse(fall=19.0 + planted * FS), reference, FS,
                           ss.VHS_LUMA_BAND_HZ)
        assert got["leading_delay_s"] == pytest.approx(
            got["trailing_delay_s"], abs=0.3e-9)
