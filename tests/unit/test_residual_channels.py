"""Tests for the residual channels' downscale to 4fsc on the final time base.

An amplitude channel is resampled on the timing alone - the decoder's wow
level adjust, a frequency-domain term, must not multiply it - while the time
channel is the final time base's own speed deviation; and the export writes
one file per field carrying the channels.
"""

import types
from importlib.resources import files

import numpy as np
from scipy import interpolate

import lddecode.core as ldd
from lddecode.utils import scale_field
from vhsdecode import residual_channels as rc

LINES = 8
WIDTH = 64
LINEOFFSET = 1
INLINELEN = 128.0


def _field(speed=1.0, directory=None):
    line_count = LINES + LINEOFFSET + 3
    linelocs = np.arange(line_count) * INLINELEN * speed + 10.0
    total = int(linelocs[-1] + 2 * INLINELEN) + 64
    lut = np.load(files("lddecode").joinpath("sinc_lut.npz"))["downscale_sinc_lut"]
    rf = types.SimpleNamespace(
        downscale_sinc_lut=lut, freq_hz=40e6, _residual_channels_dir=directory
    )
    field = types.SimpleNamespace(
        rf=rf,
        outlinecount=LINES,
        outlinelen=WIDTH,
        lineoffset=LINEOFFSET,
        inlinelen=INLINELEN,
        linelocs=linelocs,
        isFirstField=True,
        readloc=0,
        wow_interpolation_method="linear",
        data={"video": {"envelope": np.full(total, 3.0, dtype=np.float32)}},
    )
    field.computewow_scaled = lambda: ldd.Field.computewow_scaled(field)
    return field


def test_timing_only_resample_leaves_the_level_adjust_out():
    field = _field(speed=1.002)
    timing_only = rc.resample_timing_only(field, field.data["video"]["envelope"])
    locs, wow = field.computewow_scaled()
    full = np.zeros(LINES * WIDTH, dtype=np.float32)
    scale_field(
        field.data["video"]["envelope"],
        full,
        locs,
        wow,
        field.rf.downscale_sinc_lut,
        LINEOFFSET,
        WIDTH,
    )
    assert timing_only.shape == (LINES, WIDTH)
    assert np.allclose(timing_only, 3.0, rtol=2e-3)
    # the decoder's own path multiplies the uniform 0.2% speed error in
    assert np.allclose(full.reshape(LINES, WIDTH), timing_only * 1.002, rtol=1e-5)


def test_time_base_residual_is_the_speed_deviation():
    per_pixel, per_line = rc.time_base_residual(_field(speed=1.002))
    assert per_pixel.shape == (LINES, WIDTH)
    assert np.allclose(per_pixel, 0.002, atol=1e-6)
    assert np.allclose(per_line, 0.002, atol=1e-9)


def test_export_writes_one_file_per_field(tmp_path):
    field = _field(directory=str(tmp_path))
    first = rc.export_field(field)
    second = rc.export_field(field)
    assert first.endswith("field_000000.npz")
    assert second.endswith("field_000001.npz")
    with np.load(first) as z:
        assert set(z.files) >= {
            "amplitude",
            "time",
            "time_per_line",
            "site",
            "is_first_field",
            "linelocs",
        }
        assert z["amplitude"].shape == (LINES, WIDTH)
        assert z["is_first_field"]
        assert str(z["site"]) == rc.SITE


def test_export_is_inert_without_a_directory():
    assert rc.export_field(_field()) is None
