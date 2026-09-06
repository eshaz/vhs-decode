"""The head pair as a differential: amplitude, phase and frequency.

The three quantities Ethan's architecture asks of a differential pair, built
as complex signatures and measured against the standard the rest of this arc
uses - the participation ratio of the singular values, against the tape
magnetics' 1.58 effectively distinguishable of 6.

The finding these tests pin down is that the set is ONE new kind and two
members of the collinear family: the delay is orthogonal to both magnitude
quantities, while the frequency scaling sits at 0.9976 with the separation
because a rescaling of the axis acting on a family of rates is another rate.
"""

import numpy as np
import pytest
from scipy.signal import hilbert

from vhsdecode.models import head_differential as hd
from vhsdecode.models import head_model
from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import interference as inf


@pytest.fixture(scope="module")
def band():
    """The VHS luma RF band, which is the band every count in this arc is
    quoted on."""
    return np.linspace(*hd.EVIDENCE_BAND_HZ, hd.EVIDENCE_PLACES)


@pytest.fixture(scope="module")
def sizes():
    return hd.default_parameters()


# --------------------------------------------------------------------------
# The hard contracts
# --------------------------------------------------------------------------


def test_every_signature_is_subtractable(band):
    """No masks and no bare power contributions: the transform works in the
    log domain and an entry whose magnitude reaches zero has no logarithm
    there. All three are bounded forms already."""
    for name, value in hd.signatures(band).items():
        assert inf.subtractable(value), name
        assert np.all(np.isfinite(value)), name
        assert np.iscomplexobj(value), name


def test_every_entry_has_a_declared_position(band):
    """A chain can only be inverted in the reverse of the order it was
    applied, so an entry without a position cannot be inverted at all."""
    names = set(hd.signatures(band))
    assert names == set(hd.COMPONENT_POSITIONS)
    # the two record-side quantities act before the medium's own losses at 20
    # and the playback delay after them
    assert hd.COMPONENT_POSITIONS["head differential frequency"] < \
        hd.COMPONENT_POSITIONS["head differential amplitude"] < 20
    assert hd.COMPONENT_POSITIONS["head differential phase"] > 23


def test_the_declared_positions_survive_the_key_s_own_ordering(band,
                                                              monkeypatch):
    """`ordered_key` refuses an entry with no position. Merged into the
    modelled set, these three must pass it - which is the check the
    coordinator's registration has to satisfy."""
    for name, place in hd.COMPONENT_POSITIONS.items():
        monkeypatch.setitem(inf.COMPONENT_ORDER, name, place)
    merged = dict(inf.signatures(band))
    merged.update(hd.signatures(band))
    ordered = inf.ordered_key(merged)
    places = {name: place for place, name in ordered}
    # last applied, first removed: the playback delay comes off before the
    # record-side separation, which comes off before the frequency scaling
    sequence = [name for _, name in ordered]
    assert sequence.index("head differential phase") < \
        sequence.index("head differential amplitude") < \
        sequence.index("head differential frequency")
    assert places["head differential amplitude"] == 13


def test_the_signatures_are_complex_with_amplitude_and_phase_apart(band, sizes):
    """The real part is what the mechanism does to amplitude and the
    imaginary part what it does to phase. The separation has no phase - which
    is the measurement, not a simplification - and the delay has no
    magnitude."""
    entries = hd.signatures(band)
    amplitude = entries["head differential amplitude"]
    phase = entries["head differential phase"]
    assert np.allclose(np.angle(amplitude), 0.0, atol=1e-12)
    assert not np.allclose(np.abs(amplitude), 1.0)
    assert np.allclose(np.abs(phase), 1.0, atol=1e-12)
    assert np.ptp(np.unwrap(np.angle(phase))) > 0.1


# --------------------------------------------------------------------------
# The closed forms
# --------------------------------------------------------------------------


def test_the_amplitude_is_wallace_s_law_in_one_dimensionless_group(sizes):
    """`exp(-2 pi dd / lambda)`: the whole dependence is on the separation
    measured in recorded wavelengths, so doubling the separation and halving
    the frequency leave the signature unchanged."""
    speed = sizes["writing_speed_m_s"]
    f = np.linspace(1e6, 7e6, 128)
    one = hd.amplitude_signature(f, 20e-9, speed)
    other = hd.amplitude_signature(0.5 * f, 40e-9, speed)
    assert np.allclose(one, other, rtol=1e-12)
    # and one wavelength of separation costs 2 pi nepers, which is 54.6 dB
    assert hd.DB_PER_WAVELENGTH_OF_SEPARATION == pytest.approx(54.6, abs=0.05)


def test_the_phase_is_a_delay_and_only_a_delay(band):
    """`exp(-2 pi j f tau)`. Removing the best-fitting delay must leave
    nothing, since there is nothing else there."""
    value = hd.phase_signature(band, 3.5e-9)
    read = hd.residual_phase_bound(np.unwrap(np.angle(value)), band)
    assert read["delay_s"] == pytest.approx(3.5e-9, rel=1e-9)
    assert read["excess_peak_degrees"] < 1e-9


def test_the_measured_excess_phase_stays_inside_its_published_bound(band):
    """The evidence the whole module rests on: once a delay of a few
    nanoseconds is removed the residual phase is within plus or minus 1.7
    degrees, which is what says the amplitude difference is a separation and
    not a gap or an azimuth difference. A gap difference would not pass this,
    and that is the point of the test."""
    delay = hd.phase_signature(band, -7.1e-9)
    read = hd.residual_phase_bound(np.unwrap(np.angle(delay)), band)
    assert read["excess_peak_degrees"] < hd.MEASURED_PHASE_BOUND_DEGREES

    # a gap-length difference between the heads, which the measurement would
    # have seen: its sinc turns the phase through the null
    speed = head_model.writing_speed(hd.mechanics())
    width = hd.mechanics()["track_width_m"]
    wide = dict(head_model.TYPICAL, gap_m=0.45e-6)
    narrow = dict(head_model.TYPICAL, gap_m=0.30e-6)
    difference = head_model.head_difference_log(band, speed, width, wide,
                                                narrow)
    # a minimum-phase loss carries the phase its own magnitude implies
    turned = -np.imag(hilbert(difference))
    assert np.degrees(np.ptp(turned)) > hd.MEASURED_PHASE_BOUND_DEGREES


def test_the_frequency_signature_is_a_shift_on_the_log_frequency_axis(sizes):
    """The log-magnitude domain makes a scaling a shift, and to first order
    the difference it makes is `ln(1+s)` times `dLbar/du`. Checked against
    the exact difference at a scaling small enough for the first order to
    hold."""
    f = np.linspace(1e6, 7e6, 512)
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    scaling = 1e-4
    exact = np.log(np.abs(hd.frequency_signature(f, scaling, speed, width)))
    linear = np.log1p(scaling) * hd.shift_direction(f, speed, width)
    # compared as vectors rather than place by place: the exact difference
    # crosses zero inside the band, and a relative tolerance at a zero
    # crossing measures the crossing's position rather than the agreement
    departure = (np.linalg.norm(exact - linear)
                 / np.linalg.norm(exact))
    assert departure < 1e-3


def test_the_frequency_signature_s_shape_is_scale_free(sizes):
    """Its size is bounded but its direction is not in question: the shape at
    a scaling of 1e-7 and at 1e-4 is the same vector, so the size decides only
    whether the signature is numerically alive."""
    f = np.linspace(1e6, 7e6, 512)
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]

    def shape(s):
        value = np.log(np.abs(hd.frequency_signature(f, s, speed, width)))
        return value - value.mean()

    small, large = shape(1e-7), shape(1e-4)
    cosine = float(small @ large / np.linalg.norm(small)
                   / np.linalg.norm(large))
    assert cosine == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# How many directions the three actually span
# --------------------------------------------------------------------------


def test_amplitude_and_phase_are_two_kinds_and_perfectly_conditioned(band):
    """A magnitude and a phase are two mechanisms, and only the real-stacked
    form says so. Measured: 2.0000 of 2 at condition 1.000."""
    shapes = hd._shapes(hd.key_signatures(band))
    read = hd._participation(shapes)
    assert read["count"] == 2
    assert read["effective"] == pytest.approx(2.0, abs=1e-6)
    assert read["condition"] == pytest.approx(1.0, abs=1e-6)
    assert read["worst_coherence"] < 1e-12


def test_the_hermitian_inner_product_would_have_lost_that_distinction(band):
    """Under `<u,v> = u^H v` a pure-amplitude and a pure-phase signature of
    the same shape are ONE direction, since `|<r, jr>| = ||r||^2`. This is
    why `real_parameters=True` is not optional for this set."""
    shapes = {name: {"frequency": value}
              for name, value in hd._shapes(hd.key_signatures(band)).items()}
    hermitian = ie.ellipsoid(shapes, "frequency", real_parameters=False)
    stacked = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    assert hermitian["rank"] < stacked["rank"]
    assert stacked["rank"] == 2


def test_the_frequency_scaling_is_a_kind_in_its_phase_and_a_rate_in_its_magnitude(band):
    """THE FINDING THIS MODULE EXISTS TO HAVE MEASURED, AND IT REVERSED.

    It first read 1.8035 of 3 at condition 28.84 with the amplitude-frequency
    pair at 0.9976, and the conclusion was that a rescaling of the axis acting
    on a response built of rate mechanisms is itself a rate. That was measured
    while `frequency_signature` built `exp(there - here)` from two log
    MAGNITUDES and carried no phase at all.

    A frequency scaling is the one case where that is not recoverable: the two
    sides are the same response at two different frequencies, so the entry
    carries `phi(s f) - phi(f)`, which the magnitude difference does not
    imply. At a fractional scaling of 1e-3 the magnitude difference peaks at
    0.00092 nepers and the phase difference at 0.00362 radians - four fifths
    of the effect was missing.

    With it carried the pair falls to 0.7054 and the scaling is another KIND
    after all. The magnitude halves are still one direction, which is what the
    separability argument always got right."""
    read = hd.distinguishable(band)
    assert read["count"] == 3
    assert read["effective"] == pytest.approx(2.2505, abs=0.01)
    assert read["condition"] == pytest.approx(2.413, rel=0.02)
    assert set(read["worst_pair"]) == {"head differential amplitude",
                                       "head differential frequency"}
    assert read["worst_coherence"] == pytest.approx(0.7054, abs=0.002)

    shapes = hd._shapes(hd.signatures(band))
    pair = hd._participation({name: shapes[name] for name in
                              ("head differential amplitude",
                               "head differential frequency")})
    assert pair["effective"] == pytest.approx(1.3355, abs=0.01)

    # and against the PHASE it is essentially orthogonal, which is the half
    # that was invisible
    against_phase = hd._participation({name: shapes[name] for name in
                                       ("head differential phase",
                                        "head differential frequency")})
    assert against_phase["effective"] == pytest.approx(1.996, abs=0.01)


def test_the_frequency_scaling_adds_a_direction_but_does_not_pay_for_it(band):
    """AN ENTRY EARNS ITS PLACE BY THE DIRECTION IT ADDS - and the reason this
    one is held has changed even though the verdict has not.

    It used to LOWER the count, 2.0000 of 2 becoming 1.8035 of 3, and cost
    more than twenty times the conditioning. That was measured without its
    phase. With the phase carried it RAISES the count to 2.2505 and costs
    2.41 times the conditioning.

    So it is still held, on a narrower argument that has to be stated rather
    than inherited: `interference.admission` asks for more than half a
    direction of gain at no more than 1.25 times the cost, and +0.25 for
    2.41x clears neither."""
    two = hd._participation(hd._shapes(hd.key_signatures(band)))
    three = hd._participation(hd._shapes(hd.signatures(band)))
    gain = three["effective"] - two["effective"]
    cost = three["condition"] / two["condition"]
    assert gain > 0.0, "it now adds a direction rather than removing one"
    assert gain == pytest.approx(0.2505, abs=0.01)
    assert cost == pytest.approx(2.413, rel=0.02)
    # the arc's own admission gate, which it fails on both counts
    assert not (gain > 0.5 and cost <= 1.25)


def test_the_set_stands_against_the_magnetics_baseline(band):
    """1.58 effectively distinguishable of 6 is what a family differing only
    in a rate gives. The head pair's two key entries give 2.00 of 2, a
    strictly better ratio, and the reason is that one of them is a phase."""
    magnetics_ratio = 1.58 / 6.0
    read = hd._participation(hd._shapes(hd.key_signatures(band)))
    assert read["effective"] / read["count"] > magnetics_ratio


# --------------------------------------------------------------------------
# THE CONTROLS. Each has an independently known answer and each can fail.
# --------------------------------------------------------------------------


def test_the_collinear_control_reads_exactly_one(band):
    """A family of separations differs only in a RATE, so five of them are
    five multiples of one vector and the participation ratio must be 1.000.
    An earlier component in this arc reported 4.65 of 5 and what exposed it
    was a control of this shape reading 4.97 where 1.00 was required."""
    read = hd.collinear_control(band)
    assert read["count"] == 5
    assert read["effective"] == pytest.approx(1.0, abs=0.01)
    assert read["passes"]


def test_the_collinear_control_can_fail(band):
    """A control that cannot fail is not a control. Given a family that
    differs in KIND rather than in rate, the same construction must read well
    above one - so a pass is evidence about the ensemble and not about the
    code always returning one."""
    speed = hd.default_parameters()["writing_speed_m_s"]
    mixed = {
        "separation": hd.amplitude_signature(band, 20e-9, speed),
        "delay": hd.phase_signature(band, 3.5e-9),
        "a tilt": inf.head_contact_tilt(band, band[0], band[-1]) + 2.0,
        "an echo": inf.echo(band, 4.0 / (band[-1] - band[0])),
    }
    read = hd._participation(hd._shapes(mixed))
    assert read["effective"] > 2.5


def test_the_azimuth_control_cancels_a_common_misalignment(band):
    """Azimuth loss is `sinc(W tan(dtheta)/lambda)`; `tan` is odd and `sinc`
    is even, so a single drum misalignment reaching one head as `+delta` and
    the other as `-delta` gives IDENTICAL magnitude responses and cancels
    exactly. This is why azimuth is not one of the three quantities."""
    read = hd.azimuth_cancels_control(band)
    assert read["common_misalignment_rms_nepers"] < 1e-12
    assert read["passes"]


def test_the_azimuth_control_can_fail(band):
    """It cannot pass by returning zero for everything: two heads with
    INDEPENDENT construction errors do differ, and the same control measures
    that difference at four hundredths of a neper."""
    read = hd.azimuth_cancels_control(band)
    assert read["independent_errors_rms_nepers"] > 1e-3
    assert read["independent_errors_rms_nepers"] > \
        1e9 * max(read["common_misalignment_rms_nepers"], 1e-300)


def test_the_wallace_round_trip_control_reproduces_the_published_range():
    """The only place in this module where the SIZE of the law is exercised
    rather than its shape, so a wrong neper-to-decibel conversion - the
    classic factor of two between a field and a power ratio - shows here and
    nowhere else. The independently known answers are the input level and
    section 4.2g's published 13.5 to 39.8 nm."""
    read = hd.wallace_round_trip_control()
    assert read["passes"]
    assert read["round_trip_error_db"] < 1e-9
    assert read["separation_m"] == pytest.approx(14.767e-9, rel=1e-3)
    low, high = read["published_range_m"]
    assert low == pytest.approx(13.5e-9, abs=0.1e-9)
    assert high == pytest.approx(39.8e-9, abs=0.1e-9)


def test_the_wallace_round_trip_control_can_fail():
    """Given a wrong conversion the round trip must not close. Checked by
    inverting the level with a decibel-per-wavelength figure that is out by
    the factor of two this control exists to catch."""
    speed = hd.default_parameters()["writing_speed_m_s"]
    wrong = hd.spacing_from_level(-0.5558, 4e6, speed) * 2.0
    back = hd.amplitude_signature(np.array([4e6]), wrong, speed)
    returned = head_model.DB_PER_NEPER * np.log(abs(complex(back[0])))
    assert abs(returned + 0.5558) > 0.5


def test_the_wallace_consistency_check_can_refute_a_pure_separation():
    """A pure separation is a straight line through the origin, so its shape
    over its level fixes the band it was measured on. The measured pair
    implies the VHS SP carrier band, which is where it was measured; a shape
    ten times larger would imply a band the instrument does not have."""
    read = hd.wallace_consistency(-0.5558, hd.MEASURED_SHAPE_DB_RMS[0], 4e6)
    low, high = read["implied_band_hz"]
    assert 2.5e6 < low < 3.2e6
    assert 4.8e6 < high < 5.6e6
    impossible = hd.wallace_consistency(-0.5558, 0.936, 4e6)
    assert impossible["implied_half_width_hz"] > 1e7


# --------------------------------------------------------------------------
# The estimator, and the two standing rules it keeps
# --------------------------------------------------------------------------


def _planted(band, sizes, scaling, separation_m, noise_db, seed):
    """A head difference in log magnitude with a known scaling and a known
    separation in it, plus noise at the measured per-pair level."""
    rng = np.random.default_rng(seed)
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    value = (np.log(np.abs(hd.frequency_signature(band, scaling, speed,
                                                  width)))
             + np.log(np.abs(hd.amplitude_signature(band, separation_m,
                                                    speed))))
    return value + rng.standard_normal(band.size) * noise_db \
        / head_model.DB_PER_NEPER


def test_the_estimator_recovers_a_planted_scaling_without_noise(band, sizes):
    """Noiseless, the only error left is the first-order linearisation:
    1.0e-03 planted comes back as 1.0015e-03, which is 0.15 per cent."""
    measured = _planted(band, sizes, 1e-3, 15e-9, 0.0, 3)
    read = hd.estimate_frequency_shift(band, measured,
                                       sizes["writing_speed_m_s"],
                                       sizes["track_width_m"])
    assert read["usable"]
    assert read["fractional_scaling"] == pytest.approx(1e-3, rel=0.01)
    assert read["spacing_difference_m"] == pytest.approx(15e-9, rel=0.01)
    assert read["explained"] > 0.999


def test_the_estimator_solves_the_spacing_jointly(band, sizes):
    """A separation difference and a frequency scaling are 0.9976 coherent,
    so an estimator that did not fit them together would read one as the
    other. With a large separation planted and no scaling, the fitted scaling
    must stay at zero."""
    measured = _planted(band, sizes, 0.0, 40e-9, 0.0, 4)
    read = hd.estimate_frequency_shift(band, measured,
                                       sizes["writing_speed_m_s"],
                                       sizes["track_width_m"])
    assert read["spacing_difference_m"] == pytest.approx(40e-9, rel=0.01)
    assert abs(read["fractional_scaling"]) < 1e-5
    assert read["coherence_with_spacing"] == pytest.approx(0.9976, abs=0.001)
    assert read["variance_inflation"] == pytest.approx(208.5, rel=0.05)


def test_a_difference_and_never_a_ratio(band, sizes):
    """The standing rule, tested where it bites: one place where the second
    head's response is small.

    In the DIFFERENCE that place contributes the logarithm of the shortfall,
    which grows by a factor of two as the null deepens by three decades. In
    the RATIO it contributes the shortfall itself, which grows by three
    decades. Least squares weighs a place by the square of its value, so the
    same bad place carries eighteen thousand times the leverage in the ratio
    form, and the estimate it produces is wrong by thirty-two orders of
    magnitude against the difference form's three.

    Both forms are disturbed - an unmodelled bad place disturbs any fit - and
    that is the honest statement. What the rule buys is that the logarithm
    BOUNDS the disturbance where the ratio does not."""
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    first = (np.abs(hd.frequency_signature(band, 1e-3, speed, width))
             * np.abs(hd.amplitude_signature(band, 15e-9, speed)))

    def forms(shortfall):
        second = np.ones_like(band)
        second[second.size // 2] = shortfall
        return (np.log(first) - np.log(second), first / second)

    shallow_difference, shallow_ratio = forms(1e-3)
    deep_difference, deep_ratio = forms(1e-6)
    assert np.ptp(deep_difference) / np.ptp(shallow_difference) == \
        pytest.approx(2.0, rel=0.05)
    assert np.ptp(deep_ratio) / np.ptp(shallow_ratio) > 500.0
    leverage = (np.ptp(shallow_ratio) / np.ptp(shallow_difference)) ** 2
    assert leverage > 1e4

    kept = hd.estimate_frequency_shift(band, shallow_difference, speed, width)
    lost = hd.estimate_frequency_shift(band, shallow_ratio, speed, width)
    assert abs(kept["fractional_scaling"]) < 10.0
    assert abs(lost["fractional_scaling"]) > 1e20


def test_the_estimator_needs_no_spectral_feature_to_seed_it(band, sizes):
    """The second standing rule: the shift is one projection onto a direction
    the physics fixes in advance, so the estimate does not depend on the
    measurement having any feature at all. A featureless straight-line
    difference is estimated just as well as a structured one."""
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    featureless = _planted(band, sizes, 0.0, 25e-9, 0.0, 6)
    assert np.max(np.abs(np.diff(featureless, 2))) < 1e-9   # a straight line
    read = hd.estimate_frequency_shift(band, featureless, speed, width)
    assert read["usable"]
    assert read["spacing_difference_m"] == pytest.approx(25e-9, rel=0.01)


def test_the_error_bar_survives_the_column_scaling(band, sizes):
    """A separation is in metres and a scaling is dimensionless, so the two
    design columns' norms differ by thirteen orders of magnitude. Solved on
    the raw columns the normal matrix is numerically singular and the error
    bar comes back at 1e-17 on a quantity that is not determined at all. On
    unit columns it comes back at 1.55e-02, which is what the noise and the
    variance inflation together predict."""
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    measured = _planted(band, sizes, 0.0, 15e-9,
                        hd.MEASURED_VARYING_DB_RMS[0], 7)
    read = hd.estimate_frequency_shift(band, measured, speed, width)
    assert read["log_shift_error"] == pytest.approx(1.55e-2, rel=0.1)
    # and the truth sits inside the bar it returns
    assert abs(read["log_shift"]) < 3.0 * read["log_shift_error"]


# --------------------------------------------------------------------------
# The frequency quantity is a BOUND
# --------------------------------------------------------------------------


def test_geometry_bounds_the_scaling_far_below_the_noise(band, sizes):
    """The decisive argument, and it is arithmetic. A head standing proud by
    `dr` sweeps at `(R+dr)/R` times the speed, and `dr` is the same protrusion
    the separation measures. 14.767 nm on a 31 mm radius gives 4.76e-07, whose
    log-magnitude shape is 1.6e-06 dB rms against a measured varying part of
    0.0841 dB rms."""
    bound = hd.geometric_bound(sizes["spacing_difference_m"],
                               sizes["drum_diameter_m"])
    assert bound["from_the_measured_separation"] == pytest.approx(4.764e-7,
                                                                  rel=0.01)
    assert bound["from_the_drum_off_axis"] == pytest.approx(4.839e-5,
                                                            rel=0.01)
    shape = np.log(np.abs(hd.frequency_signature(
        band, bound["from_the_measured_separation"],
        sizes["writing_speed_m_s"], sizes["track_width_m"])))
    shape = (shape - shape.mean()) * head_model.DB_PER_NEPER
    rms = float(np.sqrt(np.mean(shape ** 2)))
    assert rms == pytest.approx(1.642e-6, rel=0.05)
    assert hd.MEASURED_VARYING_DB_RMS[0] / rms > 1e4


def test_the_verdict_refuses_to_promote_without_an_out_of_sample_pass(band,
                                                                     sizes):
    """Report it as a bound rather than a correction unless it clears an
    out-of-sample test. With no such test supplied the verdict is a bound,
    and geometry is what binds."""
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    measured = _planted(band, sizes, 0.0, 15e-9,
                        hd.MEASURED_VARYING_DB_RMS[0], 8)
    read = hd.frequency_verdict(band, measured, speed, width,
                                drum_diameter_m=sizes["drum_diameter_m"],
                                spacing_difference_m=sizes["spacing_difference_m"])
    assert read["promoted"] is False
    assert read["report_as"] == "a bound"
    assert read["binds"] == "geometry"
    assert read["bound_from_geometry"] < read["bound_from_the_data"] / 1e4


def test_the_out_of_sample_test_refuses_a_shift_that_is_only_noise(band,
                                                                  sizes):
    """Two halves with no scaling in them at all: the coefficient fitted on
    one explains nothing of the other, and the verdict stays a bound."""
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    first = _planted(band, sizes, 0.0, 15e-9, hd.MEASURED_VARYING_DB_RMS[0], 11)
    second = _planted(band, sizes, 0.0, 15e-9, hd.MEASURED_VARYING_DB_RMS[0], 12)
    read = hd.out_of_sample_shift(band, first, second, speed, width)
    assert read["usable"]
    assert read["clears"] is False
    assert read["pooled_sigma"] < ie.SPHERE_SIGMA


def test_the_out_of_sample_test_can_clear(band, sizes):
    """A test that cannot pass is no test. A scaling large enough to stand
    above the same noise reproduces across the halves, explains a positive
    share of evidence it was not fitted on, and clears."""
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]
    first = _planted(band, sizes, 0.10, 15e-9, hd.MEASURED_VARYING_DB_RMS[0], 21)
    second = _planted(band, sizes, 0.10, 15e-9, hd.MEASURED_VARYING_DB_RMS[0], 22)
    read = hd.out_of_sample_shift(band, first, second, speed, width)
    assert read["clears"] is True
    assert min(read["held_out_gain"]) > 0.0
    assert read["pooled_sigma"] > ie.SPHERE_SIGMA

    verdict = hd.frequency_verdict(band, first, speed, width,
                                   out_of_sample=read,
                                   drum_diameter_m=sizes["drum_diameter_m"],
                                   spacing_difference_m=sizes["spacing_difference_m"])
    assert verdict["promoted"] is True
    assert verdict["report_as"] == "a correction"


def test_a_scaling_at_the_geometric_bound_is_invisible(band, sizes):
    """The sharpest statement this module can make, and the one that settles
    the question. Planted at the drum's own measured off-axis bound, under the
    measured per-pair noise, the estimator behaves IDENTICALLY to the case
    where nothing is planted at all - the same median significance to two
    decimals and the same one draw in sixty reaching three sigma, which is
    the rate three sigma is supposed to give. Planted a thousand times larger
    it is seen in two thirds of the draws. So the prior attempt's 0.40 to
    0.66 per cent was consistent with nothing being there, and more fields
    rather than a better estimator is what the question needs."""
    speed, width = sizes["writing_speed_m_s"], sizes["track_width_m"]

    def significance(scaling):
        found = []
        for seed in range(200, 260):
            measured = _planted(band, sizes, scaling, 15e-9,
                                hd.MEASURED_VARYING_DB_RMS[0], seed)
            found.append(hd.estimate_frequency_shift(band, measured, speed,
                                                     width)["sigma"])
        return np.asarray(found)

    nothing = significance(0.0)
    at_bound = significance(hd.MEASURED_DRUM_OFF_AXIS_M
                            / (0.5 * sizes["drum_diameter_m"]))
    well_above = significance(0.05)
    assert np.median(at_bound) == pytest.approx(float(np.median(nothing)),
                                                abs=0.01)
    assert np.count_nonzero(at_bound > ie.SPHERE_SIGMA) <= 3
    assert np.count_nonzero(well_above > ie.SPHERE_SIGMA) > 30
    assert max(hd.MEASURED_SHIFT_EXPLAINED) < 0.01


# --------------------------------------------------------------------------
# What the module states about the evidence it did not gather
# --------------------------------------------------------------------------


def test_the_scope_matches_the_repository_s_own_ruling():
    """Two tapes on one deck give head-difference vectors correlating at
    r = +0.11, so the difference travels with the tape and the constant is
    held per recording. This module must not contradict the ruling the
    transform's own module already carries."""
    assert hd.SCOPE == ie.HEAD_DIFFERENCE_SCOPE
    assert abs(hd.MEASURED_TWO_TAPE_R) < 0.3


def test_the_mechanics_come_from_the_head_model_and_are_not_restated():
    """A second copy of the drum diameter or the writing speed could drift
    away from the head model's silently, and the geometric bound depends on
    both."""
    theirs = head_model.mechanics_for("VHS", "NTSC", "SP")
    ours = hd.mechanics()
    assert ours["drum_diameter_m"] == theirs["drum_diameter_m"]
    assert head_model.writing_speed(ours) == \
        head_model.writing_speed(theirs) == 5.80


class TestTheTwoBandHeadDifference:
    """Ethan: "We can use the difference between the color under and luma
    components like we did before to measure the properties of the head.
    The luma and chroma components are band measurements of each head."

    Two bands, eleven wavelengths apart, written by the same head in the
    same pass - so everything but the head cancels and the second band is
    a second lever on what remains.
    """

    V = 5.8709

    def _bands(self):
        return (np.linspace(3.4e6, 4.4e6, 60), np.linspace(0.4e6, 0.9e6, 60))

    def _departure(self, grid, gain_db, clearance_m):
        gain = gain_db * np.log(10.0) / 20.0
        return gain - 2.0 * np.pi * clearance_m * grid / self.V

    def test_the_ratio_alone_names_the_mechanism(self):
        """No fitting at all: a clearance difference is linear in frequency
        and a gain difference is flat, so the ratio of the two bands'
        departures lands on 6.0 or on 1.0."""
        got = hd.band_ratio_discriminator()
        assert got["clearance_predicts"] == pytest.approx(6.0, abs=0.01)
        assert got["gain_predicts"] == 1.0

    @pytest.mark.parametrize("gain_db,clearance_m", [
        (0.98, 0.0), (0.0, 50e-9), (0.98, 50e-9), (-0.34, -20e-9),
    ])
    def test_it_recovers_planted_gain_and_clearance_together(
            self, gain_db, clearance_m):
        luma, cu = self._bands()
        got = hd.two_band_difference(
            luma, self._departure(luma, gain_db, clearance_m),
            cu, self._departure(cu, gain_db, clearance_m))
        assert got["gain_db"] == pytest.approx(gain_db, abs=1e-6)
        assert got["clearance_m"] == pytest.approx(clearance_m, abs=1e-12)

    def test_a_pure_gain_is_not_reported_as_a_clearance(self):
        """The discrimination that one band cannot make - and the reason
        the head difference was established as a flat gain."""
        luma, cu = self._bands()
        got = hd.two_band_difference(luma, self._departure(luma, 0.98, 0.0),
                                     cu, self._departure(cu, 0.98, 0.0))
        assert got["gain_significant"]
        assert not got["clearance_significant"]
        assert got["gain_share"] > 0.99

    def test_a_pure_clearance_is_not_reported_as_a_gain(self):
        luma, cu = self._bands()
        got = hd.two_band_difference(luma, self._departure(luma, 0.0, 50e-9),
                                     cu, self._departure(cu, 0.0, 50e-9))
        assert got["clearance_significant"]
        assert not got["gain_significant"]

    def test_the_significance_floor_stops_dust_reading_as_a_measurement(self):
        """On noiseless data the residual is zero, so three sigma is zero,
        and a planted pure gain reported a significant clearance of 1e-17
        metres until this floor existed."""
        luma, cu = self._bands()
        got = hd.two_band_difference(luma, self._departure(luma, 0.98, 0.0),
                                     cu, self._departure(cu, 0.98, 0.0))
        assert got["significance_floor"] > 0.0
        assert abs(got["clearance_m"]) < 1e-12

    def test_the_pair_beats_either_band_alone(self):
        """Measured: sigma on the gain falls from 0.78 dB to 0.070, and on
        the clearance from 186 nm to 23 - because over one band a constant
        and a term linear in frequency are nearly the same shape."""
        luma, cu = self._bands()
        generator = np.random.default_rng(20260906)
        noise_l = generator.normal(0.0, 0.01, len(luma))
        noise_c = generator.normal(0.0, 0.01, len(cu))
        both = hd.two_band_difference(
            luma, self._departure(luma, 0.98, 20e-9) + noise_l,
            cu, self._departure(cu, 0.98, 20e-9) + noise_c)
        alone = hd.two_band_difference(
            luma[:30], self._departure(luma[:30], 0.98, 20e-9) + noise_l[:30],
            luma[30:], self._departure(luma[30:], 0.98, 20e-9) + noise_l[30:])
        assert both["clearance_sigma_m"] < alone["clearance_sigma_m"] / 3.0
        assert both["gain_sigma_nepers"] < alone["gain_sigma_nepers"] / 3.0

    def test_it_states_what_the_difference_cancels(self):
        """The head difference is worth taking because everything shared
        goes with it."""
        luma, cu = self._bands()
        got = hd.two_band_difference(luma, self._departure(luma, 0.5, 0.0),
                                     cu, self._departure(cu, 0.5, 0.0))
        assert "tape" in got["cancels"]
        assert "de-emphasis" in got["cancels"]

    def test_mismatched_lengths_are_refused(self):
        luma, cu = self._bands()
        with pytest.raises(ValueError, match="one departure per frequency"):
            hd.two_band_difference(luma, np.zeros(5), cu, np.zeros(len(cu)))


class TestTheFixedResponse:
    """Ethan: "The model of the VCR is fixed throughout recording and
    playback depending on which circuit path is enabled on the VCR so a
    fixed response for each head confirms this though testing."

    The test, and its limit - which turns out to be the interesting part,
    because the time scale is what separates the circuit from the
    mechanics.
    """

    @staticmethod
    def _halves(level, noise, seed, n=140):
        generator = np.random.default_rng(seed)
        pulse = -40.0 * np.exp(-np.arange(n) / 8.0)
        return (pulse + level + generator.normal(0.0, noise, n),
                pulse + level + generator.normal(0.0, noise, n))

    def test_a_planted_per_head_level_is_resolved(self):
        got = hd.fixed_response_verdict(self._halves(0.0, 0.02, 1),
                                        self._halves(0.15, 0.02, 2))
        assert got["between_level"] == pytest.approx(-0.15, abs=0.02)
        assert got["level_resolved"]
        assert got["fixed_per_head"]

    def test_two_identical_heads_are_not_resolved(self):
        """The distinction must fail when there is nothing to distinguish -
        otherwise it is not a test."""
        got = hd.fixed_response_verdict(self._halves(0.0, 0.05, 3),
                                        self._halves(0.0, 0.05, 4))
        assert not got["level_resolved"]
        assert not got["fixed_per_head"]

    def test_noisier_heads_raise_the_bar(self):
        """The within-head variation is the yardstick, so a difference that
        clears it on quiet data need not on noisy."""
        quiet = hd.fixed_response_verdict(self._halves(0.0, 0.01, 5),
                                          self._halves(0.06, 0.01, 6))
        noisy = hd.fixed_response_verdict(self._halves(0.0, 0.20, 7),
                                          self._halves(0.06, 0.20, 8))
        assert quiet["level_ratio"] > noisy["level_ratio"]

    def test_level_and_shape_are_judged_separately(self):
        """Because they answer different questions, and on real tape the
        level resolves on both tapes while the shape resolves on one."""
        n = 140
        generator = np.random.default_rng(9)
        pulse = -40.0 * np.exp(-np.arange(n) / 8.0)
        ripple = 0.3 * np.sin(2 * np.pi * np.arange(n) / 17.0)
        a = (pulse + generator.normal(0, 0.01, n),
             pulse + generator.normal(0, 0.01, n))
        b = (pulse + ripple + generator.normal(0, 0.01, n),
             pulse + ripple + generator.normal(0, 0.01, n))
        got = hd.fixed_response_verdict(a, b)
        assert got["shape_resolved"]
        assert abs(got["between_level"]) < 0.05      # a pure shape difference

    def test_a_short_span_does_not_speak_for_stability(self):
        """THE LIMIT, and it is the finding. Sixteen fields is 0.27 s. The
        transport's drift lives from a quarter second upward, so a verdict
        from under a second speaks for the CIRCUIT only - which is exactly
        the part Ethan's claim is about."""
        short = hd.fixed_response_verdict(self._halves(0.0, 0.02, 10),
                                          self._halves(0.15, 0.02, 11),
                                          span_s=0.27)
        assert short["span_covers_drift"] is False
        long = hd.fixed_response_verdict(self._halves(0.0, 0.02, 12),
                                         self._halves(0.15, 0.02, 13),
                                         span_s=17.0)
        assert long["span_covers_drift"] is True

    def test_it_states_why_a_fixed_response_at_all_scales_would_refute(self):
        """A response fixed at every scale would contradict the transport
        model, not confirm it - the drum and the reels have to show."""
        got = hd.fixed_response_verdict(self._halves(0.0, 0.02, 14),
                                        self._halves(0.15, 0.02, 15))
        assert "CIRCUIT" in got["limit"]
        assert "stability" in got["limit"]

    def test_mismatched_profile_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same profile length"):
            hd.fixed_response_verdict((np.zeros(10), np.zeros(12)),
                                      (np.zeros(10), np.zeros(10)))
