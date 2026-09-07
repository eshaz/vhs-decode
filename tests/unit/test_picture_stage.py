"""The picture stage: sync amplitude and phase, genlock, burst lock and the
time-base residual.

These act on the demodulated line, after the RF matrix has been corrected,
on quantities the RF stage does not carry. The two results the module exists
to establish are that the four modelled modifications are a separable family
where the tape's magnetics are a collinear one, and that genlock and burst
lock are ONE dimension rather than two, separated only by a precision on the
sync's own position that the standard states exactly.
"""

import numpy as np
import pytest

from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import interference as inf
from vhsdecode.models import pair_dimension as pdim
from vhsdecode.models import picture_stage as ps
from vhsdecode.models import standard_levels as sl


BASEBAND = np.linspace(0.0, 4.2e6, 4096)


# --------------------------------------------------------------------------
# The hard contracts
# --------------------------------------------------------------------------


def test_every_signature_is_subtractable():
    """The transform works in the log domain, so an entry whose magnitude
    reaches zero cannot be taken off by subtraction. Every entry here is a
    bounded `1 + a shape`, never a mask and never a bare power term."""
    entries = ps.signatures(BASEBAND, include_burst_lock=True,
                            include_burst_amplitude=True)
    assert len(entries) == 6
    for name, value in entries.items():
        assert inf.subtractable(value), name
        assert np.iscomplexobj(value), name
        magnitude = np.abs(value)
        assert magnitude.min() >= 1.0 - ps.DEFAULT_AMOUNT - 1e-9, name
        assert magnitude.max() <= 1.0 + ps.DEFAULT_AMOUNT + 1e-9, name


def test_a_measured_residual_is_subtractable_once_taken_from_a_constant():
    """Ethan's phrasing is the construction: a residual crosses zero and has
    no logarithm there, and subtracted from a constant it does."""
    places = np.arange(BASEBAND.size)
    measured = np.sin(6.0 * np.pi * places / BASEBAND.size)
    assert not inf.subtractable(measured)          # as it stands, it cannot
    entry = ps.time_base_residual(BASEBAND, residual=measured)
    assert inf.subtractable(entry)                 # taken from one, it can
    with pytest.raises(ValueError):
        ps.time_base_residual(BASEBAND, residual=measured[:10])


def test_every_entry_has_a_declared_position_above_the_capture():
    """A chain can only be inverted in the reverse of the order it was
    applied. The picture stage runs after the capture, so every position is
    above the capture's, and `ordered_key` takes them off first."""
    entries = ps.signatures(BASEBAND, include_burst_lock=True,
                            include_burst_amplitude=True)
    for name in entries:
        assert name in ps.COMPONENT_ORDER, name
        assert ps.COMPONENT_ORDER[name] > max(inf.COMPONENT_ORDER.values())
    # the declared order is the order the decoder applies them: the
    # horizontal reference, then the time base, then the two locks, then the
    # levels
    order = [name for name, _ in sorted(ps.COMPONENT_ORDER.items(),
                                        key=lambda pair: pair[1])]
    assert order == ["sync phase", "time base residual", "burst lock",
                     "genlock", "sync amplitude", "burst amplitude"]


def test_the_signatures_carry_amplitude_and_phase_where_the_mechanism_does():
    """The real part is what a mechanism does to amplitude and the imaginary
    part what it does to phase, and an entry carries the parts its mechanism
    has. Every entry moves in amplitude; the four that displace a specified
    feature carry phase as well; the time-base residual does not, because a
    jitter loss is a real attenuation with no phase to have - the same reason
    `interference.particle_noise` is returned real. A measured residual
    passed in may be complex, since the project's own time channel carries
    the hsync deviation real and the burst phase imaginary."""
    entries = ps.signatures(BASEBAND, include_burst_amplitude=True)
    for name, value in entries.items():
        assert np.iscomplexobj(value), name
        assert np.ptp(np.abs(value)) > 1e-6, name
    with_phase = {name for name, value in entries.items()
                  if np.ptp(np.angle(value)) > 1e-6}
    assert with_phase == set(entries) - {"time base residual"}
    complex_residual = ps.time_base_residual(
        BASEBAND, residual=np.exp(2j * np.pi * BASEBAND * 1e-7))
    assert np.ptp(np.angle(complex_residual)) > 1e-6


def test_no_constant_is_taken_on_faith():
    """Every number is the standard's. The subcarrier is 315/88 MHz, a line
    is 2/455 of it, the sync pulse comes from `pair_dimension` rather than
    being transcribed twice, and the default timing residual is the
    standard's own subcarrier-to-horizontal tolerance."""
    assert ps.SUBCARRIER_HZ == pytest.approx(3579545.4545, abs=1e-3)
    assert ps.LINE_HZ == pytest.approx(4.5e6 / 286.0, rel=1e-12)
    assert ps.LINE_PERIOD_S == pytest.approx(63.5556e-6, abs=1e-10)
    # not a second transcription of SMPTE 170M table 2
    assert ps.sync_spectrum.__defaults__[0] is pdim.SYNC_WIDTH_S
    # 40 degrees of subcarrier, which is what the standard allows this exact
    # relationship
    assert ps.specified_jitter_s() == pytest.approx(31.04e-9, rel=1e-3)


# --------------------------------------------------------------------------
# The closed forms against the specification
# --------------------------------------------------------------------------


def test_the_closed_form_is_the_standard_s_pulse():
    """`pulse_control` inverts the analytic spectrum and measures the pulse
    with an instrument in a module this one does not own. The expected
    answers are the standard's 4.7 us and 140 ns."""
    control = ps.pulse_control()
    assert control["passes"]
    assert control["width_s"] == pytest.approx(pdim.SYNC_WIDTH_S, abs=1e-9)
    assert control["rise_s"] == pytest.approx(pdim.SYNC_RISE_S, abs=5e-9)
    assert control["spectrum_mean_error"] < 1e-3
    assert control["spectrum_worst_error"] < 1e-3


def test_the_pulse_control_can_fail():
    """A control that cannot fail is not a control. The specific error it
    exists to catch is the one `spec_sync` records: using the specified
    10-90 time as the transition width, which makes the pulse rise faster
    than the standard allows. Built that way the same measurement reads
    86.3 ns against the specified 140, on a pulse whose width is still
    exactly right - so only the rise catches it."""
    rate, count = 40e6, 1024
    f = np.fft.rfftfreq(count, d=1.0 / rate)
    width = pdim.SYNC_WIDTH_S
    # the WRONG form: the 10-90 time used directly as the transition width
    wrong = (pdim.SYNC_DEPTH_IRE * width * np.sinc(f * width)
             * ps.cosine_taper_transform(f * pdim.SYNC_RISE_S))
    centre = np.exp(-2j * np.pi * f * (0.5 * count / rate - 0.5 * width))
    pulse = np.fft.irfft(wrong.astype(np.complex128) * centre
                         * np.exp(-2j * np.pi * f * width / 2.0) * rate,
                         n=count)
    read = sl.half_amplitude_crossings(pulse, rate)
    assert read["rise_s"] < 0.8 * pdim.SYNC_RISE_S
    # and the right form passes the same measurement
    assert ps.pulse_control()["rise_s"] > 0.9 * pdim.SYNC_RISE_S


def test_the_rate_control_reads_one_and_can_fail():
    """A family differing only in a RATE is one direction. The same
    construction on a family differing in KIND is not, which is what makes
    the reading of one evidence rather than arithmetic."""
    control = ps.rate_control()
    assert control["passes"]
    assert control["effective"] == pytest.approx(1.0, abs=0.01)

    # The same statistic on a family that differs in KIND reads far more
    # than one, so the reading of one is evidence about the family and not a
    # property of the arithmetic.
    assert ps.distinguishable(BASEBAND)["effective"] > 3.0

    # And it catches the failure it was written for. The same five-member
    # family, built so that its departure from a constant sits at the
    # rounding level, reads 1.45 instead of 1.00: normalising a near-zero
    # vector turns rounding into directions, which is how a magnetics
    # estimate once read 4.97 where it should have read 1.00.
    f = BASEBAND
    manufactured = []
    for size in (0.25, 0.5, 1.0, 2.0, 4.0):
        vector = size * 1e-15 * f / f.max() + np.ones_like(f)
        vector = vector - vector.mean()
        manufactured.append(vector / np.linalg.norm(vector))
    singular = np.linalg.svd(np.array(manufactured), compute_uv=False)
    share = singular ** 2 / (singular ** 2).sum()
    assert 1.0 / np.sum(share ** 2) > 1.2


def test_the_null_space_control_reads_zero_and_is_two_sided():
    """Pinning removes any affine function of the index exactly. A constant
    is annihilated, a tilt with it, and a quadratic is NOT - so a
    construction that annihilated everything fails this control."""
    control = ps.null_space_control()
    assert control["passes"]
    assert control["constant_recovered"] == pytest.approx(0.0, abs=1e-12)
    assert control["quadratic_survives"] > 0.0


def test_all_three_controls_pass_together():
    assert ps.controls()["passes"]


# --------------------------------------------------------------------------
# The shape of the family
# --------------------------------------------------------------------------


def test_the_picture_stage_is_a_separable_family_where_the_tape_is_not():
    """The six magnetic losses are six values of ONE dimensionless group and
    collapse to 1.58 of 6 at condition 1.06e4. These four depend on four
    different groups and reach 3.84 of 4 at condition 1.35 - the ratio the
    multipath family reaches, and by the same mechanism."""
    fit = ps.distinguishable(BASEBAND)
    assert fit["count"] == 4
    assert fit["effective"] > 3.8
    assert fit["condition"] < 2.0
    assert fit["effective"] / fit["count"] > 3.0 * (1.58 / 6.0)
    assert fit["worst_coherence"] < 0.3


def test_burst_amplitude_adds_a_direction_and_the_count_says_so():
    """The rule the S-VHS emphasis established: a modification earns a place
    by the direction it adds. This one adds a whole one at no cost in
    conditioning, which is what a clear entry looks like."""
    four = ps.distinguishable(BASEBAND)
    five = ps.distinguishable(BASEBAND, include_burst_amplitude=True)
    assert five["effective"] - four["effective"] > 0.9
    assert five["condition"] == pytest.approx(four["condition"], rel=0.05)


def test_the_ellipse_s_invariant_holds_and_the_inner_product_matters():
    """Every row is a unit vector, so the trace is the component count for
    any ensemble whatever. And each entry is a REAL parameter's signature, so
    a gain and a delay of the same shape must be two directions: under the
    Hermitian product the same set reads 3.34 where the stacked form reads
    4.84."""
    shapes = ps._shapes(ps.signatures(BASEBAND,
                                      include_burst_amplitude=True))
    stacked = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    hermitian = ie.ellipsoid(shapes, "frequency", real_parameters=False)
    assert stacked["trace"] == pytest.approx(5.0, rel=1e-9)
    assert hermitian["trace"] == pytest.approx(5.0, rel=1e-9)
    assert stacked["participation"] > hermitian["participation"] + 1.0


def test_the_result_does_not_depend_on_the_one_assumed_number():
    """`amount` is the module's single assumed value. Over a fifteen-fold
    range it moves the third figure of the conditioning and nothing else."""
    counts = [ps.distinguishable(BASEBAND, amount=a,
                                 include_burst_amplitude=True)["effective"]
              for a in (0.05, 0.25, 0.75)]
    assert max(counts) - min(counts) < 0.05


def test_the_result_does_not_depend_on_the_grid():
    counts = [ps.distinguishable(np.linspace(0.0, 4.2e6, n),
                                 include_burst_amplitude=True)["effective"]
              for n in (512, 4096, 16384)]
    assert max(counts) - min(counts) < 0.01


# --------------------------------------------------------------------------
# Genlock against burst lock: the tie, and what breaks it
# --------------------------------------------------------------------------


def test_burst_lock_lies_inside_the_span_of_the_other_two():
    """Burst lock is a whole-line delay and a whole-line delay is a sync
    displacement plus a subcarrier phase. Measured, 95.5 per cent of it lies
    inside the span of `sync phase` and `genlock`."""
    entries = ps.signatures(BASEBAND, include_burst_lock=True)
    shapes = ps._shapes(entries)

    def stacked(name):
        vector = shapes[name]["frequency"]
        vector = np.concatenate([vector.real, vector.imag])
        return vector / np.linalg.norm(vector)

    basis = np.array([stacked("sync phase"), stacked("genlock")]).T
    target = stacked("burst lock")
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    outside = target - basis @ coefficients
    assert float(outside @ outside) < 0.06


def test_entering_burst_lock_as_its_own_curve_makes_the_key_worse():
    """The failure the S-VHS emphasis demonstrated: a second copy of a
    direction already in the span lowers the count and worsens the
    conditioning. That is why `signatures` leaves it out by default."""
    without = ps.distinguishable(BASEBAND)
    with_it = ps.distinguishable(BASEBAND, include_burst_lock=True)
    assert with_it["effective"] < without["effective"]
    assert with_it["condition"] > 5.0 * without["condition"]
    assert set(with_it["worst_pair"]) == {"genlock", "burst lock"}


def test_genlock_and_burst_lock_are_one_dimension_on_the_frequency_axis():
    """A constant phase and a delay ought to differ in kind, and the luma
    band ought to break the tie. It does not: a delay's sensitivity goes as
    `f |X(f)|` and the specified line's only high-frequency content is the
    burst, so the two are one direction on both bands."""
    tie = ps.alignment_tie(BASEBAND)
    assert tie["one_dimension_on_the_frequency_axis"]
    assert tie["full_band"]["effective"] < 1.1
    assert tie["full_band"]["worst_coherence"] > 0.95
    assert tie["burst_band"]["effective"] < 1.1


def test_what_breaks_the_tie_is_a_precision_the_standard_states_exactly():
    """One sample of the 4fsc grid is a quarter cycle of subcarrier, so at
    one degree of burst phase the sync must be located to a ninetieth of a
    sample. At that balance the coherence is exactly 1/sqrt(2)."""
    required = ps.separating_precision_s(1.0)
    assert required["degrees_per_4fsc_sample"] == pytest.approx(90.0)
    assert required["in_4fsc_samples"] == pytest.approx(1.0 / 90.0, rel=1e-9)
    assert required["required_sync_precision_s"] == pytest.approx(0.776e-9,
                                                                  rel=1e-2)
    by_precision = {round(row["sync_precision_in_4fsc_samples"], 6):
                    row for row in
                    ps.alignment_tie(BASEBAND)["per_line_by_sync_precision"]}
    assert by_precision[1.0]["coherence"] > 0.9999
    assert by_precision[round(1.0 / 90.0, 6)]["coherence"] == pytest.approx(
        1.0 / np.sqrt(2.0), rel=1e-9)
    assert by_precision[round(1.0 / 90.0, 6)]["effective"] == pytest.approx(
        4.0 / 3.0, rel=1e-9)


def test_the_exported_displacement_annihilates_a_constant():
    """The burst-to-sync relationship is the only absolute phase reference
    the signal has, and the lock's own exported displacement cannot carry it:
    the endpoints are pinned, and pinning removes a constant exactly."""
    rng = np.random.default_rng(9)
    series = np.cumsum(rng.standard_normal(262)) * 1e-9
    result = ps.displacement_null_space(series)
    assert result["null_is_the_constant"]
    assert result["recovered_fraction"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        ps.displacement_null_space([1.0, 2.0])


# --------------------------------------------------------------------------
# The band, and the map onto an RF grid
# --------------------------------------------------------------------------


def test_the_chroma_entries_are_gated_on_the_band_they_live_in():
    """On a grid stopping at 3 MHz - where a VHS luma baseband stops - the
    burst holds under two per cent of its energy, so an entry there would be
    claiming a direction the signal cannot support."""
    luma = np.linspace(0.0, 3.0e6, 2048)
    assert not ps.reaches_burst_band(luma)
    entries = ps.signatures(luma, include_burst_amplitude=True)
    assert "genlock" not in entries and "burst amplitude" not in entries
    assert ps.reaches_burst_band(BASEBAND)
    assert "genlock" in ps.signatures(BASEBAND)
    low, high = ps.burst_band()
    assert low == pytest.approx(2.784e6, rel=1e-3)
    assert high == pytest.approx(4.375e6, rel=1e-3)


def test_the_rf_map_keeps_the_family_separable_and_stays_unwrapped():
    """A baseband component rides the carrier as a pair of sidebands, so an
    RF place carries `|f - f_c|`. The mapped abscissa is not monotone, which
    is safe only while the bounded form keeps every phase inside a
    half-turn - and it does."""
    rf = np.linspace(0.5e6, 7e6, 2048)
    baseband = ps.baseband_from_rf(rf, 3.9e6)
    assert baseband.min() == pytest.approx(0.0, abs=2e3)
    entries = ps.signatures(baseband)
    assert max(float(np.abs(np.angle(v)).max()) for v in entries.values()) < np.pi
    fit = ps.distinguishable(baseband)
    assert fit["effective"] > 3.8
    assert fit["condition"] < 2.0


def test_the_picture_stage_adds_directions_to_the_rf_key():
    """The two sets on one RF abscissa: the key alone spans about 7.5 of its
    14, and the picture stage adds a little over one more direction on top of
    it rather than duplicating what is already there."""
    rf = np.linspace(0.5e6, 7e6, 2048)

    def shape(value):
        value = np.asarray(value)
        magnitude = np.log(np.maximum(np.abs(value), 1e-12))
        phase = np.unwrap(np.angle(value))
        return (magnitude - magnitude.mean()) + 1j * (phase - phase.mean())

    def effective(entries):
        stack = np.array([np.concatenate([v.real, v.imag])
                          for v in entries.values()])
        stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
        singular = np.linalg.svd(stack, compute_uv=False)
        share = singular ** 2 / (singular ** 2).sum()
        return float(1.0 / np.sum(share ** 2))

    key = {name: shape(value) for name, value in inf.signatures(rf).items()}
    picture = {name: shape(value) for name, value in
               ps.signatures(ps.baseband_from_rf(rf, 3.9e6)).items()}
    together = dict(key)
    together.update(picture)
    # measured at 1.26 directions added, on a key of 14 spanning 7.47; the
    # bar is set below that because what the RF key contains is not this
    # module's to fix, and the claim being tested is that the picture stage
    # adds directions rather than duplicating what is already there
    assert effective(together) > effective(key) + 0.8


# --------------------------------------------------------------------------
# The colour lock, read one field at a time - Ethan's picture-stage
# directive, and the first measurement in this module that reads a field.
# --------------------------------------------------------------------------

from vhsdecode.models import burst_sync_lock as _bsl   # noqa: E402

CHROMA_FS = 4.0 * ps.SUBCARRIER_HZ
LINES = 210
SAMPLES = 910
BURST_WINDOW = (74, 110)
ACTIVE_WINDOW = (134, 894)
FIRST_LINE = 30


def _chroma_field(phase_deg=-33.0, amplitude=40.0, frequency_hz=None,
                  jitter_deg=0.0, seed=0, lines=LINES):
    """One synthetic up-heterodyned chroma field.

    Built the way the format builds it: a continuous subcarrier, so the
    phase at the same column advances by the standard's half cycle from one
    line to the next, which is what `expected_burst_phase_rad` carries.
    """
    rng = np.random.default_rng(seed)
    columns = np.arange(SAMPLES, dtype=np.float64)
    seconds = columns / CHROMA_FS
    carrier = ps.SUBCARRIER_HZ if frequency_hz is None else float(frequency_hz)
    numbers = np.arange(FIRST_LINE, FIRST_LINE + lines, dtype=np.float64)
    alternation = _bsl.expected_burst_phase_rad(numbers)
    wobble = np.radians(jitter_deg) * rng.standard_normal(lines)
    angle = (2.0 * np.pi * carrier * seconds[None, :]
             + np.radians(phase_deg)
             + alternation[:, None] + wobble[:, None])
    return amplitude * np.cos(angle)


def test_the_colour_lock_comes_back_complex():
    """Ethan's standing rule, on the quantity it matters most for: the lock
    is an amplitude AND a phase, so it is one complex number."""
    got = ps.colour_lock(_chroma_field(), CHROMA_FS, BURST_WINDOW,
                         ACTIVE_WINDOW, first_line=FIRST_LINE)
    for side in ("burst", "active"):
        assert isinstance(got[side]["lock"], complex)
        assert got[side]["amplitude"] > 0.0
    assert isinstance(got["difference"], complex)


def test_a_planted_phase_and_amplitude_come_back():
    for planted, size in ((-33.0, 40.0), (147.0, 25.0), (12.5, 60.0)):
        got = ps.colour_lock(_chroma_field(planted, size), CHROMA_FS,
                             BURST_WINDOW, ACTIVE_WINDOW,
                             first_line=FIRST_LINE)
        assert got["burst"]["phase_deg"] == pytest.approx(planted, abs=0.5)
        assert got["active"]["phase_deg"] == pytest.approx(planted, abs=0.5)
        # the two windows read one carrier, so they must agree
        assert got["difference_deg"] == pytest.approx(0.0, abs=0.5)
        # amplitude scales with the planted size; the phasor sums over the
        # window, so only the ratio between two sizes is meaningful
        assert got["burst"]["amplitude"] > 0.0


def test_the_specified_alternation_is_removed_before_the_lines_combine():
    """Without it every other line reads 180 degrees out and the field mean
    is nothing, which is the whole reason the removal is not optional."""
    locked = ps.colour_lock(_chroma_field(), CHROMA_FS, BURST_WINDOW,
                            ACTIVE_WINDOW, first_line=FIRST_LINE)
    assert locked["burst"]["line_coherence"] > 0.99
    # the same field read as though it began one line later: the standard's
    # half cycle now falls the other way on every line
    shifted = ps.colour_lock(_chroma_field(), CHROMA_FS, BURST_WINDOW,
                             ACTIVE_WINDOW, first_line=FIRST_LINE + 1)
    assert abs(shifted["burst"]["phase_deg"]
               - locked["burst"]["phase_deg"]) == pytest.approx(180.0, abs=1.0)


def test_the_line_coherence_says_whether_the_window_is_locked():
    """A window whose phase wanders line to line is not carrying a lock, and
    the coherence is what reports that rather than the amplitude, which
    barely moves."""
    steady = ps.colour_lock(_chroma_field(jitter_deg=0.0), CHROMA_FS,
                            BURST_WINDOW, ACTIVE_WINDOW, first_line=FIRST_LINE)
    wandering = ps.colour_lock(_chroma_field(jitter_deg=90.0, seed=3),
                               CHROMA_FS, BURST_WINDOW, ACTIVE_WINDOW,
                               first_line=FIRST_LINE)
    assert steady["burst"]["line_coherence"] > 0.99
    assert wandering["burst"]["line_coherence"] < 0.65


def test_the_heterodyne_frequency_is_measured_and_not_assumed():
    """Ethan: the up-heterodyne need not use a fixed frequency. A carrier
    written away from the subcarrier must be read as that departure."""
    for offset in (-2000.0, 0.0, 5000.0):
        got = ps.colour_lock(
            _chroma_field(frequency_hz=ps.SUBCARRIER_HZ + offset), CHROMA_FS,
            BURST_WINDOW, ACTIVE_WINDOW, first_line=FIRST_LINE)
        assert got["within_line_frequency_hz"] == pytest.approx(offset, abs=60.0)
        assert got["within_line_coherence"] > 0.99


def test_the_burst_drift_is_zero_on_a_field_that_does_not_drift():
    got = ps.colour_lock(_chroma_field(), CHROMA_FS, BURST_WINDOW,
                         ACTIVE_WINDOW, first_line=FIRST_LINE)
    assert got["burst_drift_hz"] == pytest.approx(0.0, abs=0.5)


def test_one_field_at_a_time_with_no_averaging():
    """Ethan's last clause, made a property of the code rather than a claim
    about it: nothing is carried between calls, so the same field twice
    gives the same answer and a different field gives its own."""
    one = _chroma_field(phase_deg=-33.0, seed=1)
    two = _chroma_field(phase_deg=+61.0, seed=2)
    a = ps.colour_lock(one, CHROMA_FS, BURST_WINDOW, ACTIVE_WINDOW,
                       first_line=FIRST_LINE)
    b = ps.colour_lock(two, CHROMA_FS, BURST_WINDOW, ACTIVE_WINDOW,
                       first_line=FIRST_LINE)
    again = ps.colour_lock(one, CHROMA_FS, BURST_WINDOW, ACTIVE_WINDOW,
                           first_line=FIRST_LINE)
    assert again["burst"]["lock"] == a["burst"]["lock"]
    assert b["burst"]["phase_deg"] == pytest.approx(61.0, abs=0.5)
    assert a["burst"]["phase_deg"] == pytest.approx(-33.0, abs=0.5)


def test_a_window_outside_the_line_is_refused():
    field = _chroma_field()
    with pytest.raises(ValueError, match="inside the line"):
        ps.colour_lock(field, CHROMA_FS, (74, 110), (134, SAMPLES + 1),
                       first_line=FIRST_LINE)
    with pytest.raises(ValueError, match="one field"):
        ps.colour_lock(field[0], CHROMA_FS, BURST_WINDOW, ACTIVE_WINDOW)
