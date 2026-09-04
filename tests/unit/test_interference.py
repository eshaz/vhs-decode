"""The interference types as complex components.

The measurement that motivates the module: given a random phase per field
so that none can correlate trivially, four of five disturbances are still
not random to the transform. Only particle noise sits at the sphere floor.
"""

import numpy as np
import pytest

from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import interference as inf


def _across_fields(make, fields=16, seed=0):
    rng = np.random.default_rng(seed)
    return {f"field {i}": {"frequency": make(rng)} for i in range(fields)}


def test_only_particle_noise_sits_at_the_sphere_floor():
    """A beat has a frequency even when its phase is random; a tilt has a
    shape even when its size is not. Particle noise has neither."""
    f = np.linspace(1e6, 7e6, 512)

    def white(rng):
        return rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)

    def line(rng):
        return inf.beat(f, 3.9e6) * np.exp(2j * np.pi * rng.random())

    def tilt(rng):
        return inf.head_contact_tilt(f, f[0], f[-1]) * np.exp(
            2j * np.pi * rng.random())

    noise = ie.ellipsoid(_across_fields(white, seed=1), "frequency")
    assert noise["significant"] == 0
    assert noise["sigma"] <= ie.SPHERE_SIGMA

    for make, seed in ((line, 2), (tilt, 3)):
        fit = ie.ellipsoid(_across_fields(make, seed=seed), "frequency")
        assert fit["significant"] >= 1
        assert fit["sigma"] > 50.0        # unmistakably structured


def test_echoes_are_a_fourier_family_and_separate():
    """Every tape loss is a monotone decay and they collapse together. An
    echo family is exp(-2 pi j f tau), orthogonal by construction, so
    delays spaced beyond the band's resolution are nearly independent."""
    f = np.linspace(1e6, 7e6, 2048)
    resolution = 1.0 / (f[-1] - f[0])
    delays = [resolution * k for k in range(1, 9)]
    shapes = {}
    for delay in delays:
        value = inf.echo(f, delay, amplitude=0.2)
        vector = np.log(np.abs(value)) + 1j * np.unwrap(np.angle(value))
        shapes[f"{delay * 1e6:.2f} us"] = {"frequency": vector - vector.mean()}
    fit = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    # NOT `significant`: that counts directions standing ABOVE the bulk, and
    # a perfectly orthogonal family has none - it IS the bulk, uniformly, so
    # a sphere is the correct reading. How many independent directions a
    # family spans is the participation ratio instead.
    assert fit["rank"] == 8
    assert fit["asymmetry"] < 0.05        # very nearly a sphere
    stack = np.array([
        np.concatenate([shapes[n]["frequency"].real,
                        shapes[n]["frequency"].imag]) for n in fit["names"]])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    values = np.linalg.svd(stack, compute_uv=False)
    share = values ** 2 / (values ** 2).sum()
    assert 1.0 / np.sum(share ** 2) > 7.5      # 7.92 of 8, measured
    assert values[0] / values[-1] < 2.0        # condition 1.2, measured


def test_a_gain_and_a_delay_are_two_mechanisms_not_one():
    """Under the Hermitian inner product a pure-amplitude and a pure-phase
    signature of the same shape are ONE direction, because |<r, jr>| =
    |r|^2. When each component is the effect of a REAL physical parameter
    they are two, and only the stacked form says so."""
    shape = np.linspace(-1.0, 1.0, 128)
    residuals = {"gain": {"frequency": shape.astype(complex)},
                 "delay": {"frequency": 1j * shape}}
    hermitian = ie.ellipsoid(residuals, "frequency")
    stacked = ie.ellipsoid(residuals, "frequency", real_parameters=True)
    assert hermitian["rank"] == 1         # reads as one direction
    assert stacked["rank"] == 2           # correctly two


def test_the_modelled_set_beats_the_magnetics_alone():
    """Tape magnetics alone are 1.58 effectively distinguishable of 6,
    because every loss is a function of one dimensionless group. Adding the
    interference types, whose signatures are not all monotone decays,
    raises the count several-fold."""
    f = np.linspace(1e6, 7e6, 1024)
    result = inf.distinguishable(f)
    assert result["count"] >= 10
    assert result["effective"] > 4.0
    assert result["condition"] < 100.0    # far better conditioned than 1.06e4


def test_a_wider_band_tells_more_apart():
    """The actionable finding from the broadcast-VTR comparison: only
    fractional bandwidth moves identifiability, so a fit should use the
    widest band the RF occupies rather than the demodulator's band-pass."""
    narrow = inf.distinguishable(np.linspace(0.5e6, 1.5e6, 1024))
    wide = inf.distinguishable(np.linspace(0.5e6, 12e6, 1024))
    assert wide["effective"] > narrow["effective"]
    assert wide["condition"] < narrow["condition"]


def test_particle_noise_rises_with_frequency():
    """The read volume shrinks with the wavelength, so tape noise rises
    with frequency where a converter's does not."""
    f = np.array([1e6, 4e6, 7e6])
    level = np.abs(inf.particle_noise(f, 5.8, 58e-6, 0.35e-6))
    assert level[0] < level[1] < level[2]


def test_the_key_is_complete_only_where_it_spans():
    """Ethan: "We have the key which is all possible modifications to our
    video signal." Completeness is a measurement: what lies outside the span
    cannot be recovered by any fit, because nothing models it."""
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    rng = np.random.default_rng(501)
    modelled = inf.head_contact_tilt(frequencies, frequencies[0],
                                     frequencies[-1])
    unmodelled = np.sin(2 * np.pi * 17 * (frequencies - frequencies[0])
                        / (frequencies[-1] - frequencies[0])).astype(complex)

    inside = inf.span_completeness(
        {f"c{i}": modelled * (1.0 + 0.1 * rng.standard_normal())
         for i in range(8)}, frequencies)
    outside = inf.span_completeness(
        {f"c{i}": unmodelled * np.exp(1j * np.pi * rng.random())
         for i in range(8)}, frequencies)
    assert inside["inside"] > 0.9
    assert inside["excess"] > 0.8
    # a shape the key does not model reads at or below the null
    assert outside["excess"] < 0.05


def test_the_three_way_split_separates_unmodelled_from_noise():
    """The middle part - outside the key but still structured - is the only
    thing that is both unreached and reachable, and it is separable from
    noise only because the surrogate null exists."""
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    rng = np.random.default_rng(502)

    def noise():
        return (rng.standard_normal(380) + 1j * rng.standard_normal(380))

    unmodelled = np.sin(2 * np.pi * 17 * (frequencies - frequencies[0])
                        / (frequencies[-1] - frequencies[0])).astype(complex)
    pure = inf.split_residual({f"c{i}": noise() for i in range(13)},
                              frequencies, draws=12, seed=2)
    missing = inf.split_residual(
        {f"c{i}": 3 * unmodelled + noise() for i in range(13)},
        frequencies, draws=12, seed=2)
    assert pure["noise"] > 0.9
    assert pure["structured"] < 0.05
    assert missing["structured"] > 0.4
    assert missing["outside_stands_above_the_null"]


def test_self_demagnetisation_binds_the_depth_not_the_optimum():
    """The guide gives the optimum recording depth as a quarter wavelength;
    the shortest wavelength a coating of depth d can hold is 2*pi*d, so
    recording lambda requires d <= lambda/(2*pi). Since 1/(2*pi) < 1/4 the
    demagnetisation limit binds EVERYWHERE and the optimum is never
    reachable - the two have no fixed point."""
    frequencies = np.array([1e6, 3.9e6, 7e6])
    ideal = inf.ideal_tape_response(frequencies)
    assert np.all(ideal["depth_m"] < ideal["optimum_depth_m"])
    # and the cost is a constant, because both laws scale with lambda
    assert ideal["cost_of_the_limit_db"] == pytest.approx(1.96, abs=0.01)
    # so the slope is unchanged at 20 dB per decade
    slope = ((ideal["snr_db"][-1] - ideal["snr_db"][0])
             / np.log10(frequencies[-1] / frequencies[0]))
    assert slope == pytest.approx(-20.0, abs=0.5)


def test_the_closing_step_separates_the_absolute_floor():
    """Ethan's closing step: interpolate the final unknown residual over the
    tape's ideal response. What the tape's response could never have carried
    is not tape information; what it could, and that reconstructs, is
    recoverable; what remains is the absolute floor."""
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    rng = np.random.default_rng(1001)

    def noise():
        return rng.standard_normal(380) + 1j * rng.standard_normal(380)

    smooth = np.sin(2 * np.pi * 3 * (frequencies - frequencies[0])
                    / (frequencies[-1] - frequencies[0])).astype(complex)
    tilt = inf.head_contact_tilt(frequencies, frequencies[0], frequencies[-1])

    modelled = inf.interpolate_over_ideal(
        {f"c{i}": 3 * tilt + noise() for i in range(13)}, frequencies)
    with_missing = inf.interpolate_over_ideal(
        {f"c{i}": 3 * tilt + 3 * smooth + noise() for i in range(13)},
        frequencies)
    # the smooth unmodelled part is recoverable by interpolation
    assert with_missing["reconstructable"] > 10.0 * modelled["reconstructable"]
    # and recovering it lowers the floor
    assert with_missing["absolute_floor"] < modelled["absolute_floor"]
    # the shares account for everything
    for result in (modelled, with_missing):
        assert (result["inside_the_tape_band"]
                + result["outside_the_tape_band"]) == pytest.approx(1.0)


def test_the_whole_chain_is_declared_and_strictly_ordered():
    """Ethan: "The order of the components MUST be know though." Each
    component module declares its own positions and `full_chain` composes
    them, so the chain runs from the source's own inserted test signals
    through the transmission path, the recording machine, the tape, the
    playback machine, the capture, and the picture stage."""
    chain = inf.full_chain()
    bands = {
        "source": (0, 0),
        # 5 is the tuner's clip, the last stage of a broadcast source;
        # 6 is the clamp; the VCR's own AGC at 8 sits with the recording
        # machine, before its input clip at 9
        "transmission": (1, 6),
        # 9 is the recorder's INPUT stage, where a clip is applied to
        # the incoming video before the emphasis touches it
        "the recording machine": (8, 19),
        "the tape": (20, 23),
        "the playback machine": (24, 29),
        "the capture": (30, 30),
        "the picture stage": (40, 49),
    }
    for name, position in chain.items():
        assert any(low <= position <= high
                   for low, high in bands.values()), f"{name} at {position}"
    # the recording machine acts before the tape, the tape before the capture
    assert chain["sub emphasis level dependence"] < chain["particle noise"]
    assert chain["particle noise"] < chain["beat / co-channel"]
    assert chain["record level dependence"] < chain["head contact tilt"]
    assert chain["specified test signal"] < chain["vestigial sideband"]


def test_a_module_may_not_contradict_another_about_a_position():
    """A chain with an ambiguous order cannot be inverted, so a name claimed
    at two different positions is refused rather than silently resolved."""
    chain = inf.full_chain()
    contested = next((name for name in chain
                      if name not in inf.COMPONENT_ORDER), None)
    if contested is None:
        pytest.skip("no other component module is installed")
    original = dict(inf.COMPONENT_ORDER)
    try:
        inf.COMPONENT_ORDER[contested] = chain[contested] + 1
        with pytest.raises(ValueError, match="ambiguous order"):
            inf.full_chain()
        # and it can be composed anyway when the caller says so
        assert isinstance(inf.full_chain(strict=False), dict)
    finally:
        inf.COMPONENT_ORDER.clear()
        inf.COMPONENT_ORDER.update(original)


def test_ordered_key_still_resolves_every_shipped_entry():
    """A key entry with no declared position is refused outright, so the
    shipped signatures must all be covered by the chain above."""
    frequencies = np.linspace(1e6, 7e6, 512)
    entries = inf.signatures(frequencies)
    resolved = inf.ordered_key(entries)
    assert len(resolved) == len(entries)
    # and it comes back last-applied-first-removed
    positions = [position for position, _ in resolved]
    assert positions == sorted(positions, reverse=True)


# --------------------------------------------------------------------------
# The opt-in modules joining the key
# --------------------------------------------------------------------------

CARRIER = np.linspace(2.543e6, 5.257e6, 1024)
LUMA = np.linspace(1e6, 7e6, 1024)
FULL_RF = np.linspace(0.5e6, 12e6, 1024)


def _mechanics():
    from vhsdecode.models import head_model
    return head_model.mechanics_for("VHS", "NTSC", "SP")


def test_the_default_key_is_unmoved_by_the_opt_in_parameter():
    """Adding a module to the tree must not silently move a number a caller
    has already measured, which is why `include` defaults to none."""
    plain = inf.signatures(LUMA)
    same = inf.signatures(LUMA, include=None)
    assert set(plain) == set(same)
    for name in plain:
        assert np.array_equal(plain[name], same[name]), name


def test_the_admission_measurement_describes_the_key_it_builds():
    """These two read 15.18 of 23 and 14.72 of 23 on the same band before
    the builder was made to project against the same snapshot. A measurement
    that describes a different key than the one assembled is worthless."""
    verdict = inf.admission(LUMA, mechanics=_mechanics())
    built = inf.signatures(LUMA, mechanics=_mechanics(),
                           include=verdict["admitted"])
    effective, count, condition = inf._conditioning(built)
    assert count == verdict["end"]["count"]
    assert effective == pytest.approx(verdict["end"]["effective"], abs=1e-9)
    assert condition == pytest.approx(verdict["end"]["condition"], abs=1e-9)


def test_a_held_module_is_re_offered_and_can_turn_from_bad_to_good():
    """THE REASON THE ADMISSION RUNS TO A FIXED POINT. The same module, the
    same band and the same gate: a bad trade against the bare key and a good
    one against the enriched key, because orthogonalising against a richer
    key removes more of what the module duplicates."""
    verdict = inf.admission(LUMA, mechanics=_mechanics())
    offers = [step for step in verdict["trace"]
              if step["module"] == "head_differential"]
    assert len(offers) > 1, "a held module must be offered again"
    assert offers[0]["verdict"] == "held"
    assert offers[-1]["verdict"] == "admitted"
    # and it is the projection that changed, not the gate
    assert offers[-1]["delta_effective"] > offers[0]["delta_effective"]
    assert offers[-1]["delta_condition"] < offers[0]["delta_condition"]


def test_a_band_too_narrow_to_tell_the_shapes_apart_admits_nothing():
    """The carrier band is the measurement working, not failing: 2.5 to
    5.3 MHz cannot separate these shapes, so every candidate costs more
    conditioning than the direction it brings."""
    verdict = inf.admission(CARRIER, mechanics=_mechanics())
    assert verdict["admitted"] == []
    assert verdict["end"] == verdict["start"]


def test_the_test_signals_are_held_on_every_band():
    """`vertical_interval` emits specified test SIGNALS, not modifications
    of the chain, and 25 of them take the condition past 1500."""
    for band in (CARRIER, LUMA, FULL_RF):
        verdict = inf.admission(band, mechanics=_mechanics())
        assert "vertical_interval" in verdict["held"]


def test_every_admitted_entry_is_ordered_and_subtractable():
    chain = inf.full_chain()
    for band in (LUMA, FULL_RF):
        verdict = inf.admission(band, mechanics=_mechanics())
        assert verdict["admitted"], "these bands do admit modules"
        built = inf.signatures(band, mechanics=_mechanics(),
                               include=verdict["admitted"])
        for name, value in built.items():
            assert inf.subtractable(value), name
            assert inf.position_of(name, chain) is not None, name
        # and the whole key can be inverted, last applied first removed
        order = inf.ordered_key(built, chain=chain)
        assert [place for place, _ in order] == sorted(
            (place for place, _ in order), reverse=True)


def test_a_time_axis_module_is_not_a_candidate_for_a_frequency_key():
    """Matching is on the dimension and the period, never on the name."""
    assert "transport_dimensions" not in inf.CANDIDATE_MODULES
    with pytest.raises(ValueError):
        inf._candidate_signatures("transport_dimensions", LUMA)


# --------------------------------------------------------------------------
# Outliers, corrected at the residual step
# --------------------------------------------------------------------------


def test_an_outlier_the_key_spans_is_attributed_and_corrected():
    """Ethan: *"outliers that do not explain the model, but are more
    significant than the noise floor, should be corrected at the residual
    step. We need to measure the response of the outlier, which should be
    contained in one of the many dimensions we have."*"""
    grid = LUMA
    key = inf.signatures(grid, mechanics=_mechanics())
    # plant an outlier that IS one of the key's own shapes, at a size well
    # above the floor, on a fraction of the band
    carrier = np.asarray(key["sound trap"], dtype=np.complex128)
    shape = carrier - carrier.mean()
    shape = shape / np.linalg.norm(shape)
    residual = 40.0 * shape
    got = inf.outlier_response(residual, grid, noise_floor=1e-3,
                               mechanics=_mechanics())
    assert got["outliers"] > 0
    assert got["spanned"], "a key entry's own shape must lie in the key"
    assert got["inside"] > 0.9
    assert got["carried_by"] == "sound trap"
    # and the correction removes it
    assert np.linalg.norm(got["corrected"]) < 0.2 * np.linalg.norm(residual)


def test_nothing_below_the_noise_floor_is_called_an_outlier():
    """Below the floor there is nothing to explain, because the floor is the
    total information the sample holds."""
    grid = LUMA
    rng = np.random.default_rng(5)
    quiet = rng.normal(0, 1e-6, grid.size) + 1j * rng.normal(0, 1e-6, grid.size)
    got = inf.outlier_response(quiet, grid, noise_floor=1.0,
                               mechanics=_mechanics())
    assert got["outliers"] == 0
    assert np.array_equal(got["corrected"], quiet)


def test_an_outlier_outside_the_span_is_reported_not_fitted():
    """The honest outcome: what nothing models is not recoverable by any
    fit, and the answer is a new dimension rather than a larger coefficient
    on an old one."""
    grid = LUMA
    key = inf.signatures(grid, mechanics=_mechanics())
    basis = []
    for entry in key.values():
        vector = np.asarray(entry, dtype=np.complex128)
        vector = vector - vector.mean()
        for direction in basis:
            vector = vector - (direction.conj() @ vector) * direction
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            basis.append(vector / norm)
    rng = np.random.default_rng(9)
    stranger = rng.normal(0, 1, grid.size) + 1j * rng.normal(0, 1, grid.size)
    for direction in basis:                      # make it orthogonal to all
        stranger = stranger - (direction.conj() @ stranger) * direction
    stranger = 40.0 * stranger / np.linalg.norm(stranger)
    got = inf.outlier_response(stranger, grid, noise_floor=1e-3,
                               mechanics=_mechanics())
    assert got["inside"] < 1e-6, "nothing modelled may claim it"
    assert got["excess"] < 0.0, "and it is below the size-of-basis null"


def test_the_share_is_reported_against_the_null_a_large_key_captures():
    """A basis of k vectors captures k/L of ANY vector by chance, so the
    excess above that null is what says the key covers the outlier rather
    than merely being large."""
    grid = LUMA
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 1, grid.size) + 1j * rng.normal(0, 1, grid.size)
    got = inf.outlier_response(40.0 * noise, grid, noise_floor=1.0,
                               mechanics=_mechanics())
    assert got["null"] == pytest.approx(got["basis"] / grid.size)
    assert abs(got["excess"]) < 0.05, "unstructured noise sits at the null"


def test_offering_a_module_as_a_block_is_itself_a_limitation():
    """A module is held when its entries are collinear WITH EACH OTHER,
    whatever each would be worth alone. Offered entry by entry, the same
    entries are admitted."""
    block = inf.admission(LUMA, mechanics=_mechanics(), passes=6)
    entrywise = inf.admission(LUMA, mechanics=_mechanics(),
                              granularity="entry", passes=6)
    assert "vertical_interval" in block["held"]
    assert entrywise["end"]["effective"] > 2.5 * block["start"]["effective"]
    assert entrywise["end"]["count"] > block["end"]["count"]
    # and every admitted entry is still subtractable and orderable
    chain = inf.full_chain()
    built = inf.signatures(LUMA, mechanics=_mechanics(),
                           include=[label.split(":", 1)[0]
                                    for label in entrywise["admitted"]])
    for name, value in built.items():
        assert inf.subtractable(value), name
        assert inf.position_of(name, chain) is not None, name


def test_the_entry_wise_gate_is_bounded_in_total_not_only_per_offer():
    """Thirty-eight separate offers at 1.25 each would license a 4815x rise.
    The total ceiling is what makes the guarantee auditable rather than
    incidentally small."""
    got = inf.admission(LUMA, mechanics=_mechanics(),
                        granularity="entry", passes=6)
    assert got["condition_rise"] < 4.0
    assert got["condition_rise"] * 100 < got["licensed_by_per_offer_gate"]
    assert got["end"]["condition"] <= got["condition_ceiling"]
    # and a tight total ceiling actually binds, so it is not decoration
    tight = inf.admission(LUMA, mechanics=_mechanics(),
                          granularity="entry", passes=6, total_cost=1.05)
    assert len(tight["admitted"]) < len(got["admitted"])


def test_the_granularity_default_moves_nothing():
    assert inf.admission(LUMA, mechanics=_mechanics())["granularity"] \
        == "module"
    with pytest.raises(ValueError):
        inf.admission(LUMA, granularity="nonsense")
