"""The tesseract: the fold on all dimensions, exact in and exact out."""

import numpy as np
import pytest

from vhsdecode.models import tesseract as T


def planted(seed=0, bins=40, noise=1e-3):
    rng = np.random.default_rng(seed)
    axes = ("head", "polarity", "half", "tape")
    grid = np.linspace(0.0, 1.0, bins)
    shape = np.exp(-2 * grid) + 1j * np.sin(3 * grid)
    plant = {("head",): 1.0, ("tape",): 0.5, ("head", "tape"): 0.3}
    values = np.zeros((2, 2, 2, 2, bins), dtype=complex)
    cube = T.Cube(axes, values, np.full(values.shape, noise ** 2))
    for vertex in cube.vertices:
        total = np.zeros(bins, dtype=complex)
        for subset, amount in plant.items():
            sign = np.prod([-1 if vertex[axes.index(a)] else 1 for a in subset])
            total += sign * amount * shape
        values[vertex] = total + noise * (rng.standard_normal(bins)
                                          + 1j * rng.standard_normal(bins)) / np.sqrt(2)
    return T.Cube(axes, values, np.full(values.shape, noise ** 2),
                  bins=grid * 1e6), plant


def test_all_contrasts_rebuild_the_cube_exactly():
    cube, _ = planted()
    contrasts = T.walsh(cube)
    assert len(contrasts) == 2 ** cube.order
    assert np.abs(T.reconstruct(cube, contrasts) - cube.values).max() < 1e-12


def test_the_fold_order_does_not_matter():
    cube, _ = planted(1)
    first = T.fold(T.fold(cube, "head")["differential"], "tape")["differential"]
    second = T.fold(T.fold(cube, "tape")["differential"], "head")["differential"]
    assert np.allclose(first.values, second.values)
    assert np.allclose(first.variance, second.variance)


def test_only_the_planted_contrasts_are_identified():
    cube, plant = planted(2)
    verdicts = T.identified(T.walsh(cube))
    found = {s for s, v in verdicts.items() if v["identified"]}
    assert found == set(plant)


def test_the_unreached_residual_is_at_the_floor_when_nothing_more_is_planted():
    cube, _ = planted(3)
    out = T.unreached(cube)
    assert not out["top_order_identified"]
    assert 0.5 < out["residual_over_floor"] < 1.5
    assert out["edge_fraction_by_count"] == pytest.approx(4 / 15)


def test_the_dense_graph_counts_edges_and_diagonals():
    cube, _ = planted(4)
    graph = T.dense_graph(cube)
    assert len(graph["vertices"]) == 16
    assert graph["edges"] == 4 * 2 ** 3          # n 2^(n-1)
    assert graph["diagonals"] == 16 * 15 // 2 - 32
    # an edge reads every contrast containing its axis, a diagonal on two
    # axes reads the contrasts with an odd overlap
    edge = next(p for p in graph["pairs"] if p["kind"] == "edge")
    assert all(edge["differ_on"][0] in s for s in edge["reads"])


def test_asymmetry_reads_the_planted_sides():
    cube, plant = planted(5)
    sides = T.asymmetry(cube)
    assert sides["head"]["side_in_floors"] > sides["tape"]["side_in_floors"] > 1.0
    assert sides["polarity"]["side_in_floors"] < 3.0


def test_one_real_signal_is_real_and_leaves_the_floor():
    cube, _ = planted(6)
    out = T.one_real_signal(cube, sample_rate_hz=4e6)
    assert out["kernel_is_real"]
    assert set(out["identified"]) == {"head", "tape", "head:tape"}
    assert 0.5 < out["residual_over_floor"] < 1.5


def test_the_video_cube_folds_the_luma_onto_the_chroma_axes():
    cube, _ = planted(7)
    luma = T.fold(cube, "tape")["parent"]                    # head x polarity x half
    chroma = T.Cube(("head", "half"), np.ones((2, 2, 1)), np.ones((2, 2, 1)))
    video = T.video_cube(luma, chroma)
    assert video.axes == ("head", "half")
    assert video.values.shape[-1] == luma.values.shape[-1] + 1
    assert video.channels["chroma"] == slice(luma.values.shape[-1], None)


def test_the_field_index_bits_fold_every_time_scale_at_once():
    rng = np.random.default_rng(8)
    fields, bins = 16, 12
    n = np.arange(fields)
    # a head term (bit 0), a drum term (bit 1) and an 8-field term (bit 3)
    head = np.where(n % 2, 1.0, -1.0)
    drum = np.where((n >> 1) & 1, 1.0, -1.0)
    slow = np.where((n >> 3) & 1, 1.0, -1.0)
    series = (0.5 * head[:, None] + 0.2 * drum[:, None] + 0.1 * slow[:, None]
              + 0.001 * rng.standard_normal((fields, bins)))
    built = T.from_field_series(series, np.full(series.shape, 1e-6), n)
    cube = built["cube"]
    assert cube.axes == tuple(f"field bit {b}" for b in range(4))
    assert built["fields_used"] == 16 and built["fields_dropped"] == 0
    verdicts = T.identified(T.walsh(cube))
    found = {s for s, v in verdicts.items() if v["identified"]}
    assert found == {("field bit 0",), ("field bit 1",), ("field bit 3",)}
    assert built["scales"]["field bit 1"]["fields"] == 2
    # a record of 20 fields keeps 16 and says so
    longer = T.from_field_series(np.vstack([series, series[:4]]),
                                 np.full((20, bins), 1e-6), np.arange(20))
    assert longer["fields_used"] == 16 and longer["fields_dropped"] == 4


def test_the_one_real_signal_kernel_undoes_the_departure_on_the_band():
    cube, _ = planted(9)
    # the planted bins run 0-1 MHz; a real kernel cannot carry a phase at
    # DC, so the band is moved off it as every measured band is
    cube = T.Cube(cube.axes, cube.values, cube.variance,
                  bins=cube.bins + 0.2e6)
    rate = 4.0e6 * 4
    out = T.one_real_signal(cube, sample_rate_hz=rate)
    for vertex in cube.vertices:
        kernel = out["kernels"][vertex]
        assert np.isrealobj(kernel)
        spectrum = np.fft.rfft(kernel)
        full, _grid, index = T.on_full_grid(out["departure"][vertex], cube.bins, rate)
        assert np.abs(spectrum[index] * np.exp(out["departure"][vertex]) - 1.0).max() < 1e-10


def test_missing_fields_are_filled_from_same_head_neighbours_and_reported():
    rng = np.random.default_rng(10)
    n = np.arange(16)
    series = np.where(n % 2, 1.0, -1.0)[:, None] * np.ones((16, 5)) + 0.01 * rng.standard_normal((16, 5))
    keep = np.array([i for i in range(16) if i not in (5, 9)])
    out = T.fill_field_gaps(series[keep], np.full((keep.size, 5), 1e-4), keep, 16)
    assert out["filled"] == [5, 9]
    assert out["field_index"].size == 16
    # a filled odd field reads like its odd neighbours, not the even ones
    assert np.allclose(out["responses"][5].real, 1.0, atol=0.05)
    assert np.allclose(out["variances"][5], 4e-4)
    built = T.from_field_series(out["responses"], out["variances"], out["field_index"])
    assert built["fields_used"] == 16
