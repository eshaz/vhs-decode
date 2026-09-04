"""The vertical interval test signals as components whose synthetic side is
exactly specified.

Every other component in this arc has a synthetic side that was fitted to
the data it is then judged on. These are printed in a standard to a tenth of
an IRE, so the differential is against the standard and nothing is
estimated. What these tests hold is that the module renders exactly what the
standard prescribes, that its entries meet the transform's two hard
requirements - subtractable and ordered - and that its control catches a
deliberately broken construction rather than merely existing.
"""

import numpy as np
import pytest

from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import interference as inf
from vhsdecode.models import vertical_interval as vi


# --------------------------------------------------------------------------
# The waveforms are the standard's
# --------------------------------------------------------------------------


def test_the_pulse_ladder_is_derived_from_the_cut_off_not_written_down():
    """ITU-R BT.1439-1 Annex 4 section 2: T = 1/(2 Fc). At the 525-line
    system's 4 MHz that is 125 ns, so 2T is 250 ns and 12.5T is 1.5625 us.
    The ladder is a consequence of the cut-off, not three constants."""
    unit = vi.nyquist_interval_s(vi.NOMINAL_CUTOFF_HZ)
    assert unit == pytest.approx(125e-9, rel=1e-12)
    assert vi.NARROW_PULSE_T * unit == pytest.approx(250e-9, rel=1e-12)
    assert vi.COMPOSITE_PULSE_T * unit == pytest.approx(1.5625e-6, rel=1e-12)
    # and it moves with the cut-off, which is what makes it a derivation
    assert vi.nyquist_interval_s(5e6) == pytest.approx(100e-9, rel=1e-12)


def test_the_line_grid_is_the_decoder_s_own_and_910_is_not_arbitrary():
    """SMPTE 170M-2004 clause 8.1 fixes the subcarrier at 315/88 MHz and
    the line frequency at 2/455 of it, so four times the subcarrier is
    exactly 910 samples on one line. That is the grid the time base
    correction writes, which is why a synthetic side built on it needs no
    interpolation to meet a decode."""
    assert vi.SAMPLES_PER_LINE / vi.SAMPLE_RATE_HZ == pytest.approx(
        vi.LINE_S, rel=1e-12)
    assert vi.LINE_RATE_HZ == pytest.approx(15734.266, abs=1e-3)
    line = vi.ntc7_composite()
    assert line["time_s"].size == vi.SAMPLES_PER_LINE
    # and every element ends inside the line, so nothing wraps into the
    # next one when the reading interpolates
    last = max(stop for entry in line["elements"].values()
               for _start, stop in [entry["window_s"]])
    assert last < vi.LINE_S


def test_every_rendered_line_reads_back_as_its_specification():
    """The first check any synthetic side must pass, and it is not
    circular: the line is built from amplitudes and durations and read back
    by an instrument that finds its edges and crossings without being told
    where they are."""
    for build in vi.SIGNALS.values():
        control = vi.readback_control(build())
        failed = [row for row in control["rows"] if not row["passes"]]
        assert not failed, [(row["reading"], row["specified"], row["read"])
                            for row in failed]
        assert control["passes"]


def test_the_colour_bar_table_comes_back_to_a_tenth_of_an_ire():
    """SMPTE EG 27-2004 table 2, restated to 1e-4 IRE as SMPTE 170M-2004
    Annex A table A.3. Eight luminance levels, six chrominance amplitudes
    and six subcarrier phases, and the phases are what nothing else in the
    vertical interval supplies."""
    line = vi.colour_bars()
    got = vi.readings(line, interpolate=8)
    for name, luminance, chroma_pp, phase, _start, _stop in \
            line["geometry"]["bars"]:
        assert got[f"bar {name} luminance"] == pytest.approx(luminance,
                                                             abs=0.1)
        if chroma_pp > 0.0:
            assert got[f"bar {name} chrominance"] == pytest.approx(
                chroma_pp, abs=0.1)
            read = got[f"bar {name} phase"]
            turned = (read - phase + 180.0) % 360.0 - 180.0
            assert turned == pytest.approx(0.0, abs=0.1)


def test_luminance_and_chrominance_are_rendered_on_separate_planes():
    """A modulated staircase is a luminance element that REPLACES the level
    laid over a chrominance element that ADDS across the same window.
    Rendered into one array in sequence the staircase erases the subcarrier
    and the modulated staircase carries no chrominance at all - which is the
    whole element."""
    line = vi.ntc7_composite()
    treads = line["geometry"]["treads"]
    time = line["time_s"]
    inside = ((time > treads[2][0] + 1e-6) & (time < treads[2][1] - 1e-6))
    # the luminance plane holds the tread's level and no subcarrier
    assert np.ptp(line["luma_ire"][inside]) < 0.5
    assert np.mean(line["luma_ire"][inside]) == pytest.approx(treads[2][2],
                                                              abs=0.5)
    # and the chrominance plane holds 40 IRE peak to peak on top of it
    assert np.ptp(line["chroma_ire"][inside]) == pytest.approx(40.0, abs=1.0)


# --------------------------------------------------------------------------
# The hard contracts
# --------------------------------------------------------------------------


def test_every_signature_is_subtractable():
    """Ethan: "the shape of the keys MUST be subtractable." The transform
    works in the log domain, so an entry whose magnitude reaches zero cannot
    be taken off at all. Bounded into [1-a, 1+a], every entry has a finite
    logarithm everywhere."""
    frequencies = np.linspace(1e6, 7e6, 512)
    entries = dict(vi.signatures(frequencies))
    entries.update(vi.amplitude_signatures())
    assert len(entries) >= 20
    for name, value in entries.items():
        assert inf.subtractable(value), name
        assert np.iscomplexobj(value), name
        magnitude = np.abs(value)
        assert magnitude.min() >= 1.0 - vi.SIGNATURE_DEPTH - 1e-9, name
        assert magnitude.max() <= 1.0 + vi.SIGNATURE_DEPTH + 1e-9, name


def test_the_entries_are_ordered_once_the_prefix_is_registered(monkeypatch):
    """A chain can only be inverted in the reverse of the order it was
    applied, so `ordered_key` refuses an entry with no declared position.
    These are SOURCE-side - inserted before the transmission path and long
    before the recorder saw anything - so they sit below every position
    already declared, and one prefix declaration covers all of them."""
    frequencies = np.linspace(1e6, 7e6, 256)
    entries = dict(vi.signatures(frequencies, ["NTC-7 composite"]))

    # undeclared, the key cannot be inverted, and it says so
    with pytest.raises(ValueError):
        inf.ordered_key(entries)

    monkeypatch.setitem(inf.COMPONENT_ORDER, vi.CHAIN_PREFIX,
                        vi.CHAIN_POSITION)
    order = inf.ordered_key(entries)
    assert len(order) == len(entries)
    assert {place for place, _name in order} == {vi.CHAIN_POSITION}
    # and they are the earliest thing in the chain: the transmission path
    # acts on them, not the other way round
    assert vi.CHAIN_POSITION < min(inf.COMPONENT_ORDER[name]
                                   for name in ("vestigial sideband",
                                                "sub emphasis level "
                                                "dependence",
                                                "particle noise"))
    # last applied, first removed - so these come off last of all
    with_tape = dict(entries)
    with_tape["particle noise"] = np.ones(256, dtype=np.complex128)
    assert inf.ordered_key(with_tape)[-1][1].startswith(vi.CHAIN_PREFIX)


def test_the_signatures_carry_amplitude_and_phase_separately():
    """Real part amplitude, imaginary part phase. The 12.5T pulse's
    chrominance half is specified at 60.8 degrees and its luminance half at
    none, so the two halves of one element are two mechanisms and not one -
    which is what `real_parameters` exists to say."""
    frequencies = np.linspace(1e6, 7e6, 512)
    entries = vi.element_signatures(frequencies, vi.ntc7_composite())
    luminance = next(value for name, value in entries.items()
                     if "12.5T pulse, luminance" in name)
    chrominance = next(value for name, value in entries.items()
                       if "12.5T pulse, chrominance" in name)
    # the chrominance half carries a phase the luminance half does not
    assert np.ptp(np.angle(chrominance)) > np.ptp(np.angle(luminance))
    shapes = {name: {"frequency": np.log(np.abs(value))
                     + 1j * np.unwrap(np.angle(value))}
              for name, value in (("luminance", luminance),
                                  ("chrominance", chrominance))}
    hermitian = ie.ellipsoid(shapes, "frequency")
    stacked = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    assert stacked["rank"] >= hermitian["rank"]


# --------------------------------------------------------------------------
# The control, and the proof that it can fail
# --------------------------------------------------------------------------


def test_the_control_reads_one_direction_on_a_correct_construction():
    """A flat insertion-gain change is ONE parameter, so every reading that
    responds to it must lie along ONE direction, however many readings the
    line has. And a correct reading is homogeneous in the gain: degree one
    for a level, degree zero for a duration read at the pulse's own half
    amplitude."""
    control = vi.gain_control()
    assert control["effective"] == pytest.approx(1.0, abs=0.01)
    assert not control["inhomogeneous"]
    assert control["worst_misfit"] < 1e-6
    assert control["passes"]
    # the durations, phases and ratios do not move at all, which is the
    # statement that they were measured against the pulse and not against a
    # number
    for name in ("sync width", "2T pulse HAD", "burst phase",
                 "differential gain", "differential phase"):
        assert name in control["invariant"], name


def test_the_control_catches_a_deliberately_broken_construction():
    """A control that cannot fail is not a control. `magnetic.py` records
    one that read 4.97 where the answer was 1.00 and thereby exposed a
    published 4.65-of-5; this is the same device for this component.

    The broken construction is the one this subject is prone to and that
    `docs/VERTICAL_INTERVAL_COLLAPSE.md` records twice: pulse widths read at
    a threshold fixed in absolute IRE, and crossings taken to the nearest
    sample of the 4fsc grid."""
    broken = vi.gain_control(reader=vi.naive_readings)
    assert not broken["passes"]
    # it names the two readings a fixed threshold cannot measure
    assert set(broken["inhomogeneous"]) == {"2T pulse HAD", "sync width"}
    assert broken["worst_misfit"] > 0.1
    # and the participation ratio moves off one as well
    assert broken["effective"] > 1.1

    # the defect is real and not an artefact of the control: read naively,
    # the ideal 2T pulse is more than 10 per cent wide against a standard
    # whose whole tolerance is 4 per cent
    line = vi.ntc7_composite()
    centre, had = line["geometry"]["narrow_pulse"]
    window = vi._pulse_window(centre, had)
    signal = line["composite_ire"]
    rate = line["sample_rate_hz"]
    correct = vi.read_pulse(signal, rate, window, interpolate=8)
    naive = vi.naive_read_pulse(signal, rate, window)
    assert correct["width_s"] == pytest.approx(had, rel=0.02)
    assert naive["width_s"] > had * 1.10


# --------------------------------------------------------------------------
# How many independent dimensions a line provides
# --------------------------------------------------------------------------


def test_the_composite_line_is_the_inefficient_one():
    """`docs/VERTICAL_INTERVAL_COLLAPSE.md` section 4.3 reports 4.64
    effective of 25 readings for the NTC-7 composite, 0.186 per reading, the
    worst of eleven insertion signals, because its bar, its 2T pulse, its
    12.5T luminance half and its staircase treads are four views of one
    low-frequency luminance channel.

    Reproduced here by a construction that shares no code with that one:
    5.61 of 27, 0.208 per reading, against the combination's 10.53 of 26 and
    0.405."""
    composite = vi.line_dimensions(vi.ntc7_composite())
    combination = vi.line_dimensions(vi.ntc7_combination())
    assert composite["responding"] == composite["readings"]
    assert combination["responding"] == combination["readings"]
    # the conventional reading is several times what the line determines
    assert composite["effective"] < 0.5 * composite["readings"]
    # and the multiburst line is the efficient one, by a wide margin
    assert combination["per_reading"] > 1.5 * composite["per_reading"]
    assert combination["condition"] < 0.1 * composite["condition"]
    # both stand well above their own sphere floor, so this is structure
    for result in (composite, combination):
        assert result["sigma"] > ie.SPHERE_SIGMA


def test_the_colour_bars_collapse_on_the_frequency_axis():
    """The correction to that document's table, which does not contain
    them: eight bar luminances are eight readings of one level, so on the
    frequency axis the bars collapse harder than the composite does. Their
    value is on the amplitude axis, and that is a different measurement."""
    bars = vi.line_dimensions(vi.colour_bars())
    composite = vi.line_dimensions(vi.ntc7_composite())
    assert bars["per_reading"] < composite["per_reading"]
    assert bars["span_90"] <= 6          # six directions carry 90 per cent


def test_a_line_beats_the_tape_magnetics_per_reading():
    """`ELLIPTICAL_COLLAPSE.md` section 7.2 gives the tape's six magnetic
    mechanisms as 1.58 effectively distinguishable, 0.263 per reading,
    because every tape loss is a monotone decay in one dimensionless group.
    A multiburst is parameterised along the axis that makes its parts
    different instead."""
    combination = vi.line_dimensions(vi.ntc7_combination())
    assert combination["per_reading"] > 1.58 / 6.0


# --------------------------------------------------------------------------
# What survives a recording
# --------------------------------------------------------------------------


def test_three_packets_survive_one_is_marginal_and_one_is_lost():
    """VHS luminance stops around 3 MHz. Measured, on the specified
    packets' own energy: 0.5, 1.0 and 2.0 MHz pass essentially untouched,
    3.0 MHz keeps about half, and 4.2 MHz is gone."""
    result = vi.survival(vi.ntc7_combination())
    elements = result["elements"]
    for frequency in ("0.50", "1.00", "2.00"):
        entry = elements[f"C2 packet {frequency} MHz"]
        assert entry["verdict"] == "survives"
        assert entry["path"] == "luminance"
        assert entry["in_luma_band"] > 0.99
    assert elements["C2 packet 3.00 MHz"]["verdict"] == "marginal"
    assert elements["C2 packet 4.20 MHz"]["verdict"] == "structurally lost"
    # and the readings agree with the energy accounting
    direct, recorded = result["direct"], result["recorded"]
    kept = (recorded["C2 packet 3.00 MHz amplitude"]
            / direct["C2 packet 3.00 MHz amplitude"])
    assert kept == pytest.approx(0.55, abs=0.1)
    gone = (recorded["C2 packet 4.20 MHz amplitude"]
            / direct["C2 packet 4.20 MHz amplitude"])
    assert gone < 0.05


def test_the_subcarrier_packet_survives_through_the_wrong_channel():
    """The row that catches people out. The 3.58 MHz packet lands inside the
    chrominance passband, so a colour-under recorder carries it on the
    chroma path and it comes back intact - measuring a channel that is not
    the luminance one. A flat 3.58 MHz packet off a VHS decode is not
    evidence of luminance bandwidth."""
    result = vi.survival(vi.ntc7_combination())
    packet = result["elements"]["C2 packet 3.58 MHz"]
    assert packet["path"] == "chrominance"
    assert packet["verdict"] == "survives"
    assert packet["in_luma_band"] < 0.05
    kept = (result["recorded"]["C2 packet 3.58 MHz amplitude"]
            / result["direct"]["C2 packet 3.58 MHz amplitude"])
    assert kept == pytest.approx(1.0, abs=0.05)


def test_the_2t_pulse_survives_as_a_measurement_of_the_tape():
    """Its name says 4 MHz, but 99 per cent of a 2T pulse's energy is below
    3 MHz, so the recorder passes nearly all of it and the reading keeps its
    sensitivity. What it now reports is the recorder's own filter, which is
    why it measures the source only against a direct feed."""
    result = vi.survival(vi.ntc7_composite())
    pulse = result["elements"]["B1 2T pulse"]
    assert pulse["in_luma_band"] > 0.95
    assert pulse["verdict"] == "survives"
    # it responds - the reading moves - and that movement is the tape's
    assert abs(result["change"]["2T pulse HAD"]) > 0.0


def test_the_specified_line_is_a_dimension_in_the_arc_s_own_sense():
    """`pair_dimension.sync_dimension` takes the specified sync pulse
    against the measured one and calls it the best-conditioned dimension
    available. A test line is the same object with energy across the whole
    band instead of at one edge, and it needs no new machinery: the
    specified line is the input, the line as it came back is the output,
    and their Wiener transfer is the dimension.

    The identity is the control: given itself as its measured side the
    transfer must be exactly one everywhere."""
    line = vi.ntc7_combination()
    identity = vi.line_dimension(line, line["composite_ire"])
    assert np.allclose(np.abs(identity["transfer"]), 1.0, atol=1e-9)

    recorded = vi.through_colour_under(line, line["composite_ire"])
    through = vi.line_dimension(line, recorded)
    frequency = through["frequency_hz"]
    magnitude = np.abs(through["transfer"])

    def at(want):
        return int(np.argmin(np.abs(frequency - want)))

    # flat and fully coherent through the luminance band
    assert magnitude[at(2.0e6)] == pytest.approx(1.0, abs=0.02)
    assert through["coherence"][at(2.0e6)] > 0.99
    # gone above it
    assert magnitude[at(4.0e6)] < 0.1
    assert through["coherence"][at(4.0e6)] < 0.5
    # and flat again AT the subcarrier, because the chroma path carries it
    assert magnitude[at(vi.SUBCARRIER_HZ)] == pytest.approx(1.0, abs=0.02)
    assert through["coherence"][at(vi.SUBCARRIER_HZ)] > 0.99

    with pytest.raises(ValueError):
        vi.line_dimension(line, np.zeros(17))


# --------------------------------------------------------------------------
# The amplitude axis
# --------------------------------------------------------------------------


def test_the_bars_relieve_the_amplitude_axis_with_phase_not_with_places():
    """`ELLIPTICAL_COLLAPSE.md` section 6a records the amplitude axis as
    saturated for want of places - 91 entries over 9 sweep bins. Eight more
    places do not rescue that, and eight specified levels are worth no more
    than nine fitted bins AS LEVELS. What no sweep bin has is a subcarrier
    phase specified at a specified level, and that is the relief."""
    axis = vi.amplitude_axis()
    # the count alone does not lift the floor
    assert axis["sweep_alone"]["asymmetry"] > 0.9
    assert axis["sweep_plus_bars"]["asymmetry"] > 0.8
    assert axis["floor_below_half_needs"]["places"] > 10 * 8
    # and as levels the bars are worth about what the bins are
    assert axis["bar_directions"]["effective"] == pytest.approx(
        axis["sweep_directions"]["effective"], rel=0.1)
    # but the phase column adds directions nothing else can supply
    assert (axis["bar_directions_with_phase"]["effective"]
            > axis["bar_directions"]["effective"] + 1.0)
    assert axis["levels_with_a_phase"] == 6


def test_the_amplitude_axis_entries_are_complex_and_specified():
    """The synthetic side of the amplitude axis, needing no fit at all: the
    level, the chrominance amplitude that rides on it, and that
    chrominance's angle, at eight levels printed in a standard."""
    entries = vi.amplitude_signatures()
    assert len(entries) == 3
    places = {value.size for value in entries.values()}
    assert places == {len(vi.COLOUR_BARS_75)}
    phase = next(value for name, value in entries.items()
                 if "subcarrier phase" in name)
    # the phase entry is the only one with an imaginary part, because it is
    # the only one carrying an angle
    assert np.max(np.abs(phase.imag)) > 0.1
    for name, value in entries.items():
        if "subcarrier phase" not in name:
            assert np.max(np.abs(value.imag)) == 0.0, name
