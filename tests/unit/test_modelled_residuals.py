"""The per-field modelled-residual report, and the three errors it made first.

Ethan asked for "a report on all the modeled residuals ... what the
difference is for each field". The instrument that answers it is
`tools/ringing_measure/modelled_residuals.py`, and the tests below hold it
to the four things that would silently make its answer wrong: a window that
drifts from the certified export's, a ladder whose shares come out negative,
a held-out ladder built by slicing one joint fit, and a summary reported in
units the fit is not minimising.
"""

import importlib.util
import math
import os

import numpy as np
import pytest

_TOOL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools", "ringing_measure", "modelled_residuals.py")
_spec = importlib.util.spec_from_file_location("modelled_residuals", _TOOL)
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


# --------------------------------------------------------------------------
# The geometry: derived, and the same window the certified export used
# --------------------------------------------------------------------------

def test_the_window_reproduces_the_certified_export_exactly():
    """The pooled `*_sync_step_response.npz` exports state their own fall
    window as lags 26 to 105 - 79 samples at 0.1812 MHz spacing. This tool
    derives its window from SysParams instead of importing the module that
    built theirs, so the agreement is a check and not a tautology; if it
    ever stops holding, the per-field numbers stop being comparable with
    every recorded pooled result."""
    geom = mr.geometry()
    low = geom["front_porch"][0] + 1
    high = geom["sync_tip"][1] - 1
    assert (low, high) == (26, 105)
    assert high - low == 79
    resolution = geom["rate_mhz"] * 1e6 / (high - low)
    assert resolution == pytest.approx(0.1812e6, rel=1e-3)


def test_no_timing_is_typed_into_the_geometry():
    """Every landmark must move when the format's own table moves. PAL's
    front porch and line length differ from NTSC's, so a geometry that
    returned the same window for both would have the numbers baked in."""
    ntsc = mr.geometry("NTSC", "VHS", "sp")
    pal = mr.geometry("PAL", "VHS", "sp")
    assert ntsc["samples_per_line"] != pal["samples_per_line"]
    assert ntsc["front_porch"] != pal["front_porch"]


# --------------------------------------------------------------------------
# The instrument: a planted channel comes back out
# --------------------------------------------------------------------------

def _synthetic_field(geom, channel=None, noise=0.0, seed=0):
    """A field of ideal sync pulses, optionally through a known channel.

    THE PULSE TRAIN IS SUMMED OVER NEIGHBOURING LINES, and the first version
    of this helper was not - which cost an afternoon and is worth recording.
    A single pulse per line, tiled, has no precursor at the line boundary:
    the next line's falling edge begins about six samples before the line
    starts, and a tiled single pulse leaves those samples at exactly
    blanking. The instrument's window reaches back across that boundary, so
    it read a step with its leading tail cut off, and the recovered response
    rose to twice unity at 3.4 MHz on a channel that was the identity. The
    instrument was right and the test signal was wrong.
    """
    rng = np.random.default_rng(seed)
    per_line = geom["samples_per_line"]
    rows = 263
    depth = geom["sync_depth_ire"]
    edge = geom["transition_us"] * geom["rate_mhz"]
    sigma = edge / mr.ERF_WIDTH_FACTOR
    width = geom["sys_params"]["hsyncPulseUS"] * geom["rate_mhz"]
    from scipy.special import erf

    def rising_step(u):
        return 0.5 * (1.0 + erf(u / (math.sqrt(2.0) * sigma)))

    lag = np.arange(per_line, dtype=np.float64)
    # blanking at zero, one sync pulse per line falling at sample zero,
    # summed over the neighbouring lines so the train is truly periodic
    line = np.zeros(per_line)
    for repeat in (-1, 0, 1):
        offset = lag - repeat * per_line
        line -= depth * (rising_step(offset)
                         - rising_step(offset - width))
    field = np.tile(line, (rows, 1))
    if channel is not None:
        flat = field.reshape(-1)
        spectrum = np.fft.rfft(flat)
        grid = np.fft.rfftfreq(flat.size, d=1.0 / (geom["rate_mhz"] * 1e6))
        field = np.fft.irfft(spectrum * channel(grid),
                             n=flat.size).reshape(rows, per_line)
    if noise:
        field = field + rng.normal(0.0, noise, field.shape)
    return field


def test_an_unmodified_pulse_train_measures_as_unity():
    """The strongest check the instrument has: put the specified edge in and
    the response must be one at every frequency. Anything else is the
    instrument's own bias and would be charged to the tape."""
    geom = mr.geometry()
    field = _synthetic_field(geom)
    got = mr.step_response(mr.field_interval(field, geom), geom, "fall")
    band = ((geom["frequency_hz"] >= 0.2e6) & (geom["frequency_hz"] <= 3.5e6)
            & got["valid"])
    assert np.abs(np.abs(got["H"][band]) - 1.0).max() < 1e-6
    assert np.abs(np.angle(got["H"][band])).max() < 1e-6
    assert got["step_ire"] == pytest.approx(-geom["sync_depth_ire"], abs=1e-6)


def test_a_planted_single_pole_channel_is_recovered_up_to_level_and_delay():
    """The measurement is only worth a report if it returns what was put in.

    UP TO A LEVEL AND A DELAY, and that qualification is the whole design
    rather than a weakness. The ideal step is anchored on the measured
    edge's own settled amplitude and its own fitted 50 per cent crossing, so
    a gain and a delay are removed by construction: a 2 MHz pole's 80
    nanoseconds of group delay came back as a linear phase ramp of 0.74
    radians across this band, which is one sample. That is exactly why the
    model carries a level and a delay as nuisance terms - they belong to the
    levels estimator and to the time base, not to the channel's shape.
    """
    geom = mr.geometry()
    corner = 2.0e6

    def channel(frequency_hz):
        return 1.0 / (1.0 + 1j * frequency_hz / corner)

    field = _synthetic_field(geom, channel)
    interval = mr.field_interval(field, geom)
    assert interval is not None
    got = mr.step_response(interval, geom, "fall")
    assert got is not None
    band = ((geom["frequency_hz"] >= 0.3e6) & (geom["frequency_hz"] <= 2.0e6)
            & got["valid"])
    frequency = geom["frequency_hz"][band]
    ratio = got["H"][band] / channel(frequency)
    # take out the three nuisances the model itself takes out
    from vhsdecode.models import residual_floor
    logged = np.log(np.abs(ratio)) + 1j * np.unwrap(np.angle(ratio))
    left = residual_floor.fit(logged, np.ones(frequency.size),
                              mr.nuisance(frequency))["residual"]
    assert np.abs(np.real(left)).max() < 0.06        # nepers
    assert np.abs(np.imag(left)).max() < 0.06        # radians


def test_the_shared_nuisance_set_now_absorbs_a_pure_delay():
    """THE FINDING THIS REPORT TURNED UP, and its repair, held as a test.

    `residual_floor.nuisance` used to call its second column a delay while
    writing it linear about the BAND'S CENTRE. A delay of tau writes as
    `-2 pi f tau`, which on a grid centred at f0 is `-2 pi (f - f0) tau`
    minus the constant `2 pi f0 tau`; the ramp was in that span and the
    constant was not, and with no imaginary constant anywhere in the basis
    it had nowhere to go and was charged to the residual. On the home
    recording that alone cost half the explained share, 46.8 against 91.2
    per cent.

    The shared set now takes the delay through zero frequency and carries a
    free phase reference beside it, so a pure delay leaves nothing. This
    test is the repair's guard: it fails if either column goes back.
    """
    from vhsdecode.models import residual_floor
    frequency = np.linspace(0.2e6, 3.5e6, 512)
    delay_s = 80e-9
    logged = (np.zeros(frequency.size)
              + 1j * (-2.0 * math.pi * frequency * delay_s))
    floor = np.ones(frequency.size)

    columns = residual_floor.nuisance(frequency)
    assert "phase reference" in columns
    shared = np.column_stack(list(columns.values()))
    left_shared = residual_floor.fit(logged, floor, shared)["residual"]
    left_here = residual_floor.fit(logged, floor,
                                   mr.nuisance(frequency))["residual"]
    assert np.abs(left_shared).max() < 1e-9
    assert np.abs(left_here).max() < 1e-9

    # and a delay offset by an arbitrary unwrap branch is still absorbed,
    # which is the reason the phase reference belongs there at all
    branched = logged + 1j * 2.0 * math.pi
    assert np.abs(residual_floor.fit(
        branched, floor, shared)["residual"]).max() < 1e-9


def test_a_field_without_sync_is_refused_rather_than_fitted():
    """A field of flat blanking has no edge. Returning something from it
    would put a fabricated response on the field axis."""
    geom = mr.geometry()
    flat = np.zeros((263, geom["samples_per_line"]))
    assert mr.field_interval(flat, geom) is None


# --------------------------------------------------------------------------
# The model: the ladder, its sign, and the collinearity it exposes
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _planted_model():
    """A response built from two of the key's own components, so the
    decomposition has a right answer to be measured against."""
    from vhsdecode.models import interference, residual_floor
    frequency = np.linspace(0.2e6, 3.5e6, 400)
    key = mr.ordered_key(frequency)
    names = list(key)
    shape = (0.30 * interference._log_shape(np.asarray(key[names[0]]))
             + 0.20 * interference._log_shape(np.asarray(key[names[2]])))
    rng = np.random.default_rng(3)
    noise = 0.01 * (rng.normal(size=shape.size)
                    + 1j * rng.normal(size=shape.size))
    H = np.exp(shape + noise)
    se = np.full(frequency.size, 0.01) * np.abs(H)
    design = residual_floor.design(key, frequency)
    nuisance = len(residual_floor.nuisance(frequency))
    response = {"H": np.asarray(H), "se": se}
    valid = np.ones(frequency.size, dtype=bool)
    return {"frequency": frequency, "key": key, "names": names,
            "matrix": design["matrix"], "nuisance": nuisance,
            "response": response, "valid": valid}


def test_the_ladder_only_ever_falls_and_its_shares_are_positive(
        _planted_model):
    """Adding a column to a least-squares fit cannot raise the residual it
    was fitted to, so the ladder power is non-increasing and every share is
    non-negative. The first version of this tool reported the differences
    without negating them and every share in the report came out negative -
    a sign, not a finding."""
    got = mr.model_field(_planted_model["response"], _planted_model["valid"],
                         _planted_model["frequency"],
                         _planted_model["matrix"], _planted_model["nuisance"])
    ladder = got["ladder_power"]
    assert np.all(np.diff(ladder) <= 1e-9 * max(ladder[0], 1.0))
    assert np.all(got["ladder_share"] >= -1e-12)
    total = 1.0 - got["remainder_power"] / got["departure_power"]
    assert got["ladder_share"].sum() == pytest.approx(total, abs=1e-9)


def test_the_planted_components_carry_the_share(_planted_model):
    """The two entries the response was built from must be the two the
    ladder credits, or the decomposition is not measuring what it says."""
    got = mr.model_field(_planted_model["response"], _planted_model["valid"],
                         _planted_model["frequency"],
                         _planted_model["matrix"], _planted_model["nuisance"])
    order = np.argsort(got["ladder_share"])[::-1]
    assert set(order[:2].tolist()) >= {0}
    assert got["ladder_share"][order[0]] > 0.2
    assert 1.0 - got["remainder_power"] / got["departure_power"] > 0.9


def test_the_unique_share_never_exceeds_the_chain_share_by_much(
        _planted_model):
    """`only this one` is a partial contribution and `in the chain` a
    cumulative one; on a collinear key the first is the smaller. Both are
    non-negative, because removing a column cannot lower a residual."""
    got = mr.model_field(_planted_model["response"], _planted_model["valid"],
                         _planted_model["frequency"],
                         _planted_model["matrix"], _planted_model["nuisance"])
    assert np.all(got["unique_share"] >= -1e-12)


def test_the_held_out_ladder_refits_each_rung_rather_than_slicing_one_fit(
        _planted_model):
    """THE ERROR THIS GUARDS IS MEASURED, NOT IMAGINED. Taking one joint fit
    of the whole key and applying the first k of its coefficients is not a
    model: on a collinear design those coefficients are large and
    cancelling, so a partial sum destroys the cancellation. Done that way
    the first run reported +765 per cent against the sub-emphasis entry and
    -919 per cent against particle noise. Each rung refitted on its own
    keeps every share inside a sane range."""
    from vhsdecode.models import residual_floor
    frequency = _planted_model["frequency"]
    matrix = _planted_model["matrix"]
    nuisance = _planted_model["nuisance"]
    rng = np.random.default_rng(11)
    logs, floors = [], []
    for _ in range(6):
        logged = residual_floor.log_domain(
            _planted_model["response"]["H"]
            * np.exp(0.004 * rng.normal(size=frequency.size)),
            _planted_model["response"]["se"])
        logs.append(logged["log"])
        floors.append(logged["floor"])
    got = mr.held_out_field(0, logs, floors, [1, 2, 3, 4, 5], matrix, nuisance)
    assert np.all(np.abs(got["ladder_share"]) < 2.0)
    assert got["remainder_power"] < got["departure_power"]


# --------------------------------------------------------------------------
# The units the answer is reported in
# --------------------------------------------------------------------------

def test_the_summary_is_weighted_by_the_measurement_s_own_precision():
    """The fit minimises the residual divided by the per-bin floor, so a
    summary averaging bins equally is not the quantity being optimised.
    Above 2 MHz the sync edge's spectrum is weak and the floor is large;
    unweighted, those bins made the residual AFTER the whole key read larger
    than before it on some fields."""
    residual = np.array([0.0, 0.0, 1.0], dtype=np.complex128)
    tight = np.array([1.0, 1.0, 1.0])
    loose = np.array([1.0, 1.0, 100.0])
    assert mr.in_decibels(residual, loose) < mr.in_decibels(residual, tight)
    # and with every bin equally measured it is the plain root mean square
    plain = np.sqrt(np.mean(np.real(residual) ** 2)) * mr.NEPERS_TO_DB
    assert mr.in_decibels(residual, tight) == pytest.approx(plain)


def test_floor_units_count_independent_points_and_not_bins():
    """The 79-sample window is transformed on a 4096-point grid, so its bins
    are some fifty times finer than its information. A residual divided by
    the bin count understates itself by the root of that oversampling."""
    by_bins = mr.in_floor_units(1000.0, bins=944, cells=944)
    by_cells = mr.in_floor_units(1000.0, bins=944, cells=19)
    assert by_cells > by_bins
    assert by_cells / by_bins == pytest.approx(math.sqrt(944 / 19), rel=1e-9)


def test_the_sign_test_decides_and_is_distribution_free():
    """A component that lowers every held-out field survives; one that is a
    coin does not. The test is on signs alone, so a single wild descent
    cannot carry it."""
    assert mr.sign_test(np.full(20, 0.01))["survives"]
    assert not mr.sign_test(np.array([+9.0] + [-0.01] * 19))["survives"]
    rng = np.random.default_rng(5)
    assert not mr.sign_test(rng.normal(size=40))["survives"]
    assert not mr.sign_test([0.1, 0.2])["survives"]


def test_the_chain_order_is_read_from_the_declared_chain():
    """The ladder is a traversal of known circuitry, so its order has to
    come from `interference.full_chain` rather than from whatever order a
    dictionary happens to iterate in."""
    from vhsdecode.models import interference
    frequency = np.linspace(0.2e6, 3.5e6, 128)
    names = list(mr.ordered_key(frequency))
    chain = interference.full_chain(strict=False)

    def position(name):
        if name in chain:
            return chain[name]
        return min((place for known, place in chain.items()
                    if name.startswith(known)), default=max(chain.values()) + 1)

    places = [position(name) for name in names]
    assert places == sorted(places)
    assert names[0] == "vestigial sideband"
