"""The colour-under channel as components of the key.

The measurement that motivates the module: of the four components built - the
chroma envelope's amplitude response, its phase, the heterodyne offset, and
the luma-to-colour-under transfer - only two are new KINDS. The phase and the
transfer's roll-off are orthogonal to machine precision and to everything the
key already holds; the amplitude response and the heterodyne offset are the
magnetic family that already collapsed to 1.58 of 6, read at a wavelength six
times longer, and entering them LOWERS the effective count.

Every figure quoted in `vhsdecode/models/colour_under.py` is regenerated here.
"""

import numpy as np
import pytest

from vhsdecode.models import colour_under as cu
from vhsdecode.models import interference as inf


def _colour_under_band(places=1024, system="NTSC"):
    low = cu.carrier_hz(system) - cu.modulation_half_width_hz(system)
    return np.linspace(low, cu.band_upper_for(system), places)


def test_the_carrier_is_forty_times_the_line_rate_and_ntsc_is_derived():
    """The standard states the carrier as a multiple of the horizontal
    scanning rate - forty times it for NTSC, forty times it plus 1.953 kHz for
    PAL. The PAL figure is PRINTED as 626.953 kHz and the derivation
    reproduces it exactly; the NTSC figure is not printed anywhere and 629.371
    kHz is a derivation, which the provenance must say."""
    ntsc = cu.carrier_provenance("NTSC")
    assert ntsc["carrier_hz"] == pytest.approx(629370.63, abs=0.01)
    assert ntsc["printed"] is False
    assert ntsc["relation"] == "40 f_H"

    pal = cu.carrier_provenance("PAL")
    assert pal["carrier_hz"] == pytest.approx(626953.0, abs=0.01)
    assert pal["printed"] is True
    # the derivation and the printed figure agree
    assert pal["agreement_hz"] == pytest.approx(0.0, abs=0.5)

    # and the derivation is the relation, not a stored number
    assert cu.carrier_hz("NTSC") == pytest.approx(
        40.0 * cu.line_rate_hz("NTSC"))


def test_the_band_pass_is_not_the_signal_and_ntsc_clears_zero():
    """The colour-under is a double-sideband carrier, so modulation faster
    than the carrier folds through zero and returns as its own conjugate.
    Against the decoder's own band-pass ceilings NTSC clears it by 58.74 kHz
    and PAL does not clear it at all - but PAL's ceiling is 1.3 MHz, a filter
    setting deliberately wider than the recorded chroma, so its notional
    lower edge lying 46 kHz below zero says something about the filter and
    nothing about the signal. Given a measured extent instead, PAL clears."""
    ntsc = cu.sideband_fold_headroom_hz("NTSC")
    assert ntsc["folds"] is False
    assert ntsc["headroom_hz"] == pytest.approx(58741.26, abs=1.0)
    assert ntsc["from_the_band_pass"] is True

    pal = cu.sideband_fold_headroom_hz("PAL")
    assert pal["band_upper_hz"] == pytest.approx(1.3e6)
    assert pal["folds"] is True
    # the qualification that matters
    assert pal["from_the_band_pass"] is True
    # with a measured modulation extent rather than the filter setting
    measured = cu.sideband_fold_headroom_hz(
        "PAL", cu.carrier_hz("PAL") + 500e3)
    assert measured["folds"] is False
    assert measured["from_the_band_pass"] is False


def test_the_up_conversion_mirrors_the_band():
    """The mix keeps the difference product against an oscillator at
    `f_sc + f_cu`, so a place at `f_cu + m` lands at `f_sc - m`. The
    colour-under's LOWER sideband carries the chroma's UPPER one, and a
    correction carried across the mix needs that mirror."""
    carrier, subcarrier = cu.carrier_hz(), cu.subcarrier_hz()
    offsets = np.array([-200e3, 0.0, 200e3])
    landed = cu.up_converted_hz(carrier + offsets)
    assert landed == pytest.approx(subcarrier - offsets)


def test_every_entry_is_subtractable_and_has_a_declared_position():
    """The two hard contracts on a key entry. Subtractable: the transform
    works in the log domain, so an entry whose magnitude reaches zero cannot
    be taken off. Ordered: a chain can only be inverted in the reverse of the
    order it was applied, so an entry with no declared position makes the key
    uninvertible."""
    band = _colour_under_band()
    for name, value in cu.candidates(band).items():
        assert inf.subtractable(value), name
        assert name in cu.COMPONENT_ORDER, name
    # the roll-off is the entry that could most easily have been a mask; it
    # is a bounded Lorentzian and never reaches zero
    assert np.min(np.abs(cu.transfer_rolloff(band))) > 0.06
    # every registered entry is one of the declared four
    assert set(cu.REGISTERED) <= set(cu.COMPONENT_ORDER)
    assert set(cu.signatures(band)) == set(cu.REGISTERED)


def test_the_key_order_accepts_the_colour_under_entries(monkeypatch):
    """Registered into `interference.COMPONENT_ORDER`, the entries order
    correctly: last applied, first removed, so the head-to-medium pair at
    position 20 comes off before the recorder's chroma path at 10."""
    merged = dict(inf.COMPONENT_ORDER)
    merged.update(cu.COMPONENT_ORDER)
    monkeypatch.setattr(inf, "COMPONENT_ORDER", merged)
    order = inf.ordered_key(cu.candidates(_colour_under_band(256)))
    positions = [place for place, _ in order]
    assert positions == sorted(positions, reverse=True)
    assert positions[0] == 20 and positions[-1] == 10


def test_the_registered_entries_are_new_kinds_and_are_near_orthogonal():
    """The standard to read this against: six tape magnetic loss mechanisms
    span 1.58 effective directions of 6 at condition 1.06e4, every one being a
    function of one dimensionless group.

    THREE KINDS HERE, AND THEY ARE ALMOST INDEPENDENT: 2.940 of 3 at
    condition 1.193. A delay lives on the axis conjugate to frequency, a
    Lorentzian roll-off lives where no monotone decay has any feature, and
    the residual chroma carrier is a LINE at 629 kHz, which is a third kind
    again - the first two were exactly orthogonal at 2.000 of 2 before it
    joined them, and the line costs 0.06 of a direction to add one."""
    band = _colour_under_band()
    result = cu.distinguishable(band)
    assert result["count"] == 3
    assert result["effective"] == pytest.approx(2.940, abs=0.01)
    assert result["condition"] == pytest.approx(1.193, abs=0.01)
    assert result["worst_coherence"] < 0.2

    # and against the six magnetic mechanisms, each is a direction they lack
    magnetics = cu.magnetic_mechanisms(band)
    base = cu.spread(magnetics)["effective"]
    assert base == pytest.approx(1.4978, abs=0.01)
    for name, value in cu.signatures(band).items():
        with_it = dict(magnetics)
        with_it[name] = value
        got = cu.spread(with_it)["effective"]
        assert got - base > 0.4, name


def test_the_amplitude_and_the_offset_are_the_collapsed_family():
    """The result that decided what gets registered. Once the deck's playback
    equalisation is off - which is what the decoder's envelope shows - the
    colour-under's amplitude response is Wallace's spacing loss read at a
    longer wavelength, and the heterodyne offset is that response
    differentiated. Neither is a mechanism the key is missing, and entering
    either LOWERS the effective count, exactly as a second copy of the S-VHS
    sub pre-emphasis did."""
    band = _colour_under_band()
    magnetics = cu.magnetic_mechanisms(band)
    amplitude = cu.envelope_amplitude(band)
    offset = cu.heterodyne_offset(band)

    assert cu.coherence(amplitude, magnetics["spacing loss"]) > 0.999
    assert cu.coherence(amplitude, magnetics["thickness loss"]) > 0.999
    assert cu.coherence(amplitude, offset) > 0.999

    base = cu.spread(magnetics)["effective"]
    for name, value in (("amplitude", amplitude), ("offset", offset)):
        with_it = dict(magnetics)
        with_it[name] = value
        # a second copy of a direction already in the span LOWERS the count
        assert cu.spread(with_it)["effective"] < base

    # the wavelength-sharing law is the same monotone decay and is likewise
    # not a direction
    sharing = cu.wavelength_sharing(band)
    assert cu.coherence(sharing, magnetics["head differentiation"]) > 0.99
    with_it = dict(magnetics)
    with_it["wavelength sharing"] = sharing
    assert cu.spread(with_it)["effective"] < base


def test_a_complex_gain_crosses_the_heterodyne_with_no_scale_factor():
    """THE CONTROL, computed rather than asserted: a real colour-under tone
    with a known amplitude and starting phase is multiplied by the decoder's
    own oscillator and measured at the subcarrier. The independently known
    answers are an amplitude slope of exactly 1 - the mix does not scale a
    gain - and a phase slope of exactly -1, unit magnitude being the ruling
    and the sign being the band's mirror.

    It can fail: -5.6875 would be `heterodyne_timing_scale` wrongly applied to
    a gain, +1 would be the mirror forgotten, and an amplitude slope away from
    1 would be a scale factor that does not exist."""
    result = cu.heterodyne_control()
    assert result["passes"] is True
    assert result["amplitude_slope"] == pytest.approx(1.0, abs=1e-6)
    assert result["phase_slope"] == pytest.approx(-1.0, abs=1e-6)
    assert result["unmirrored_image_share"] < 1e-6
    # the ratio the ruling exists to keep out of a gain
    assert result["timing_scale"] == pytest.approx(5.6875, abs=1e-4)
    assert abs(result["phase_slope"]
               - result["phase_slope_if_the_ratio_were_applied"]) > 4.0


def test_the_shared_loss_is_in_the_exact_frequency_ratio():
    """THE SECOND CONTROL. Wallace's loss in nepers is `2 pi d f / v`, linear
    in frequency with no constant term, so two carriers through one separation
    took losses in the exact ratio of their frequencies - a prediction with no
    free parameter. The answer is 1.000000 with no tolerance to negotiate, and
    writing the wavelength as `v * f`, dropping the `2 pi`, or letting the
    measured wavelength-independent share leak into the ratio all break it."""
    result = cu.wavelength_sharing_control()
    assert result["passes"] is True
    assert result["worst_departure"] < 1e-12
    # and the control notices the difference it exists to notice
    assert result["departure_if_the_share_leaked"] > 0.5


def test_an_offset_on_a_linear_channel_carries_no_direction():
    """THE THIRD CONTROL, and the one that disqualified the heterodyne offset.
    `H(f + d) / H(f)` has a log that is a difference of log responses, so on a
    channel whose log response is linear in frequency it is a CONSTANT and the
    mean-removed shape is identically zero. Wallace's spacing loss is exactly
    linear, so the answer is known before the code runs.

    It can fail: forming the ratio as a product, as a difference of responses
    rather than of their logarithms, or without removing the mean all give a
    non-zero norm where the answer is provably zero - which is how an inert
    entry comes to look like a direction."""
    result = cu.linear_channel_control()
    assert result["passes"] is True
    assert result["pure_wallace_shape_norm"] < 1e-12
    # the head's whole equalised loss stack is barely more than that
    assert result["equalised_loss_stack_shape_norm"] < 1e-2
    # and what does have a shape is the deck's equaliser, not the tape
    assert result["unequalised_response_shape_norm"] > 1.0


def test_the_effective_count_does_not_read_the_grid():
    """THE FOURTH CONTROL. A participation ratio over normalised shapes is a
    property of the shapes, so sampling the same band more finely must not
    move it. The module's headline reading is a comparison between two BANDS,
    and it would be worthless if the measure responded to the grid."""
    result = cu.grid_control()
    assert result["passes"] is True
    assert result["spread"] < 1e-3
    assert result["spread_all_four"] < 1e-3


def test_the_transfer_halves_by_about_150_kilohertz():
    """The established result reproduced: the luma-to-colour-under amplitude
    transfer is the whole of the modelled transfer at a few tens of kHz and
    about half of it by 150 kHz, on both tape speeds and both test patterns.
    The half-point is the measured parameter; the shape is a one-parameter
    Wiener weight, and the same measurement is explicit that no pole order
    fits the curvature."""
    carrier = cu.carrier_hz()
    grid = carrier + np.array([0.0, 20e3, cu.TRANSFER_HALF_HZ])
    weight = np.abs(cu.transfer_rolloff(grid))
    assert weight[0] == pytest.approx(1.0)
    assert weight[1] > 0.94            # the whole of it at a few tens of kHz
    assert weight[2] == pytest.approx(0.5, abs=1e-9)   # half by 150 kHz
    # and it is symmetric about the carrier, the band being a translation
    below = np.abs(cu.transfer_rolloff([carrier - cu.TRANSFER_HALF_HZ]))
    assert float(below[0]) == pytest.approx(0.5, abs=1e-9)


def test_no_per_head_amount_difference_is_a_direction():
    """The second established result, reproduced with its mechanism and then
    strengthened. An amount is a SCALE on one direction, and a family
    differing only in a scale is exactly one direction however far apart the
    scales are - so a per-head amount could not have survived pooling because
    it was never a direction. And even a per-head HALF-POINT, which is a shape
    rather than a scale, is worth about a fiftieth of a direction at the
    difference the runtime sees."""
    band = _colour_under_band()
    result = cu.per_head_amount(band)
    assert result["by_amount"]["effective"] == pytest.approx(1.0, abs=1e-9)
    assert result["by_shape"]["effective"] == pytest.approx(1.011, abs=0.01)
    wide = cu.per_head_amount(band, half_points_hz=(150e3, 1200e3))
    assert wide["by_shape"]["effective"] == pytest.approx(1.048, abs=0.01)
    # a shape is worth more than a scale, and both are worth very little
    assert (wide["by_shape"]["effective"]
            > result["by_shape"]["effective"]
            > result["by_amount"]["effective"])


def test_the_band_extension_helps_the_magnetics_and_dilutes_the_interference():
    """THE HEADLINE MEASUREMENT, and it does not have one answer.

    For the tape's six magnetic mechanisms, which are parameterised across the
    whole band, extending the fit down to the colour-under carrier buys 0.11
    of a direction and improves the conditioning by 2.3 times - reproducing
    the U-matic reading of COMPONENT_MAPPINGS section 3 on a different format
    - and the two registered entries then buy 0.75 more.

    For the modelled interference set it COSTS 0.49, because a beat, a
    dropout, a sound trap and a set of echoes are all featureless below the
    chroma ceiling, and entries that say nothing in the same place are more
    alike than they were. A band earns its place by the directions it
    distinguishes, exactly as a modification does."""
    result = cu.band_extension()
    magnetics = result["readings"]["tape magnetics"]
    interference = result["readings"]["modelled interference"]

    assert magnetics["luma band alone"]["effective"] == pytest.approx(
        1.442, abs=0.02)
    assert magnetics["gained_by_the_band"] > 0.05
    assert magnetics["gained_by_the_entries"] > 0.5
    # and the extension improves the magnetics' conditioning
    assert (magnetics["plus the colour-under band"]["condition"]
            < 0.6 * magnetics["luma band alone"]["condition"])

    # the localised set is diluted by the same move
    assert interference["gained_by_the_band"] < -0.2
    assert interference["gained_by_the_entries"] > 0.0

    # both readings are on the same grid step, so only the band differs
    luma_places, whole_places = result["places"]
    assert whole_places > luma_places
    assert result["whole_band_hz"][0] < 100e3


def test_the_entries_are_withheld_where_the_band_cannot_see_them():
    """The rule `interference.signatures` already applies to the record-level
    entry: an entry whose whole shape lies outside the grid is a featureless
    constant there, and entering it costs conditioning without adding a
    direction. Forced onto the luma band the pair's condition number is
    1e+30."""
    luma = np.linspace(1.2e6, 6.5e6, 512)
    assert cu.signatures(luma) == {}
    forced = {
        "phase": cu.envelope_phase(luma, 0.25 / cu.modulation_half_width_hz()),
        "roll-off": cu.transfer_rolloff(luma),
    }
    assert cu.spread(forced)["condition"] > 1e12
    # and they are present the moment the carrier is inside the grid
    assert set(cu.signatures(np.linspace(58.7e3, 6.5e6, 512))) == set(
        cu.REGISTERED)


def test_the_magnetic_baseline_reproduces_its_published_worst_pair():
    """The comparison is only worth making if the baseline is the same one.
    Taken from `head_model` itself, the six mechanisms' worst coherent pair is
    gap against azimuth at 0.99997 - the same two mechanisms and the same
    1.000 the magnetics collapse reported, because both are the same sinc of
    the same dimensionless group."""
    band = np.linspace(58.7e3, 6.5e6, 1024)
    first, second, coherence = cu.worst_pair(cu.magnetic_mechanisms(band))
    assert {first, second} == {"gap loss", "azimuth loss"}
    assert coherence > 0.999


def test_the_offset_scale_is_a_scale_and_not_a_shape():
    """`heterodyne_offset` is the response's derivative to first order, so the
    normalised signature does not depend on the offset. The default is one
    line rate because the oscillator is a line-rate multiple by specification
    and a counter landing on the neighbouring multiple is off by exactly that
    - a principled quantum rather than a chosen size, and in any case only a
    size."""
    band = _colour_under_band()
    rate = cu.line_rate_hz()
    reference = cu.heterodyne_offset(band, 0.1 * rate)
    for factor in (0.3, 1.0, 3.0, 10.0):
        got = cu.heterodyne_offset(band, factor * rate)
        assert 1.0 - cu.coherence(reference, got) < 1e-6
