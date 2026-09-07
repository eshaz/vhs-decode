"""The two gauges that decide when a stage may be retired.

Ethan, this session: *"Remove the chroma transient removal step and the colour
filtering; the transform-based stage supersedes them."* and *"Remove the
averaging in the ringing path and work ONE FIELD AT A TIME."*

Neither change can be made in this lane's files, so what was built is the
number each retirement has to be judged on. These tests check that the two
gauges measure what they claim and refuse what they should refuse; the
decode-to-decode figures live in the modules' own docstrings.
"""

import numpy as np
import pytest

from vhsdecode.models import chroma_stage_supersession as css
from vhsdecode.models import colour_under
from vhsdecode.models import per_field_ringing as pfr

SAMPLE_RATE = 4.0 * colour_under.subcarrier_hz("NTSC")
WIDTH = 910
HEIGHT = 263
ACTIVE = (134, 894)
BURST = (78, 106)
LINES = (60, 230)
CENTRE = 32768.0


def _bar_field(scatter_deg, bars=7, seed=0, burst_amplitude=1000.0,
               bar_amplitude=3000.0, blank_bar=0):
    """A bar field with a known per-line phase scatter and a real burst."""
    rng = np.random.default_rng(seed)
    index = np.arange(WIDTH, dtype=np.float64)
    carrier = np.exp(2j * np.pi * 0.25 * index)
    field = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    hues = np.linspace(0.0, 2 * np.pi, bars, endpoint=False)
    edges = np.linspace(ACTIVE[0], ACTIVE[1], bars + 1).astype(int)
    for line in range(HEIGHT):
        # A line at 227.5 subcarrier cycles alternates by 180 degrees in a
        # frame reset each line; the burst carries the same alternation, which
        # is what lets the gauge divide it out.
        alternation = np.pi * (line % 2)
        wobble = np.radians(rng.normal(0.0, scatter_deg))
        row = np.zeros(WIDTH, dtype=np.complex128)
        row[BURST[0]:BURST[1]] = burst_amplitude * np.exp(
            1j * (alternation + np.pi))
        for bar, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
            if bar == blank_bar:
                continue
            row[first:last] = bar_amplitude * np.exp(
                1j * (hues[bar] + alternation + wobble))
        field[line] = np.real(row * carrier)
    return field + CENTRE


def test_the_gauge_reads_back_a_planted_scatter():
    fields = np.stack([_bar_field(3.0, seed=n) for n in range(4)])
    result = css.line_phase_scatter(fields, SAMPLE_RATE, ACTIVE, LINES, BURST)
    assert result["scatter_deg"] == pytest.approx(3.0, abs=0.6)
    assert result["fields"] == 4


def test_the_gauge_orders_two_decodes_the_right_way_round():
    quiet = css.line_phase_scatter(np.stack([_bar_field(1.0, seed=n)
                                             for n in range(4)]),
                                   SAMPLE_RATE, ACTIVE, LINES, BURST)
    noisy = css.line_phase_scatter(np.stack([_bar_field(4.0, seed=n)
                                             for n in range(4)]),
                                   SAMPLE_RATE, ACTIVE, LINES, BURST)
    assert noisy["scatter_deg"] > 3.0 * quiet["scatter_deg"]


def test_a_colourless_bar_is_not_counted():
    """The white bar carries no chroma; its phase is the noise's.

    Averaged in it would read about ninety degrees whatever the decoder did,
    so the gauge drops any window below the burst's own amplitude - which
    SMPTE 170M fixes - and the count says so.
    """
    fields = np.stack([_bar_field(2.0, seed=n, blank_bar=0)
                       for n in range(3)])
    result = css.line_phase_scatter(fields, SAMPLE_RATE, ACTIVE, LINES, BURST)
    assert len(result["per_bar_deg"]) == 6
    assert result["scatter_deg"] == pytest.approx(2.0, abs=0.6)


def test_the_bar_windows_never_straddle_a_transition():
    windows = css.bar_windows(ACTIVE, 7, keep=0.5)
    width = (ACTIVE[1] - ACTIVE[0]) / 7.0
    for index, (first, last) in enumerate(windows):
        assert first >= ACTIVE[0] + width * index
        assert last <= ACTIVE[0] + width * (index + 1)


def test_the_measured_tables_carry_their_own_ordering():
    """The recorded A/B says what it says: the comb is load bearing and the
    transient improvement is not."""
    measured = css.MEASURED
    assert measured["cti_off"]["scatter_deg"] < 1.1 * measured["shipped"][
        "scatter_deg"]
    assert measured["comb_off"]["scatter_deg"] > 2.0 * measured["shipped"][
        "scatter_deg"]


def _edge_field(ripple_ire, ire_per_unit=400.0, black=16243.2, seed=0,
                edge=533, cycles=9.0):
    """A picture with a bar edge and a known coherent ripple after it."""
    rng = np.random.default_rng(seed)
    index = np.arange(WIDTH, dtype=np.float64)
    step = np.where(index >= edge, 70.0, 0.0)
    phase = 2 * np.pi * cycles * (index - edge) / 108.0
    ring = np.where(index >= edge, ripple_ire * np.sqrt(2.0) * np.sin(phase),
                    0.0)
    lines = np.tile(step + ring, (HEIGHT, 1))
    lines = lines + rng.normal(0.0, 0.05, lines.shape)
    return lines * ire_per_unit + black


def test_the_ripple_gauge_reads_back_a_planted_ring():
    field = _edge_field(0.25)
    result = pfr.edge_ripple(field[np.newaxis, LINES[0]:LINES[1], :], ACTIVE,
                             16243.2, 400.0)
    assert result["edge"] == pytest.approx(533, abs=2)
    assert result["coherent_ire"] == pytest.approx(0.25, rel=0.2)


def test_the_ripple_gauge_ignores_a_level_shift_and_a_droop():
    """It has to, or a de-emphasis change would read as ringing."""
    plain = _edge_field(0.20, seed=3)
    index = np.arange(WIDTH, dtype=np.float64)
    tilted = plain + (400.0 * 0.01 * index)[np.newaxis, :] + 400.0 * 5.0
    a = pfr.edge_ripple(plain[np.newaxis, LINES[0]:LINES[1], :], ACTIVE,
                        16243.2, 400.0)
    b = pfr.edge_ripple(tilted[np.newaxis, LINES[0]:LINES[1], :], ACTIVE,
                        16243.2, 400.0, edge=a["edge"])
    assert b["coherent_ire"] == pytest.approx(a["coherent_ire"], rel=0.05)


def test_the_recorded_one_field_result_is_worse_than_no_correction():
    """THE FINDING, as an assertion, so a later round cannot lose it.

    It survived the engine changing under it, which is the reason to keep
    asserting it: the ringing path is now `ringing_tesseract` and the addon
    the original measurement was taken through has been deleted, and one
    field at a time is STILL worse than applying no correction at all.
    """
    measured = pfr.MEASURED
    assert (measured["one_field"]["coherent_ire"]
            > measured["correction_off"]["coherent_ire"])
    retired = measured["on_the_retired_engine"]
    assert (retired["one_field"]["coherent_ire"]
            > retired["correction_off"]["coherent_ire"])
    # AND ON THE RETIRED ENGINE THE HORIZON BOUGHT SOMETHING. A four-field
    # average was marginally better than no correction there, which is what
    # made the one-field result a cost rather than a wash.
    assert (retired["four_field"]["coherent_ire"]
            <= retired["correction_off"]["coherent_ire"])


def test_the_averaging_horizon_no_longer_reaches_the_ringing_path():
    """The directive is implemented, measured rather than asserted.

    `ringing_tesseract` accepts `average_fields` and does not use it, so
    `--inverse_eq 0` and `--inverse_eq 4` are the same decode. Both decodes
    were taken and their luma files carry one md5, which is the strongest
    form this claim can take.
    """
    measured = pfr.MEASURED
    assert measured["horizon_is_inert"] is True
    for figure in ("coherent_ire", "total_ire"):
        assert (measured["four_field"][figure]
                == measured["one_field"][figure])
    assert len(measured["identical_md5"]) == 32
    # and the contract must say the structural change is done rather than
    # still naming it as work outstanding. The factor's NAME may appear -
    # saying a thing is gone requires naming it - so what is asserted is
    # that the file it lived in is recorded as deleted.
    assert pfr.RUNTIME_CONTRACT.strip().startswith("DISCHARGED")
    assert "deleted" in pfr.RUNTIME_CONTRACT
    assert "ringing_tesseract" in pfr.RUNTIME_CONTRACT
