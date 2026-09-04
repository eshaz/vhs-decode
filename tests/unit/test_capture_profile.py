"""The capture chain: the fourth link, and the one that terminates the rest.

The profile is READ from the capture rather than asserted, because the three
constants it replaces - eight bits, forty megahertz, a 13.3 MHz roll-off -
were wrong on every capture this arc actually uses.
"""

import numpy as np
import pytest

from vhsdecode.models import capture_profile as cp


def test_the_word_length_comes_from_the_code_lattice():
    """An 8-bit capture carried in a 16-bit container is still 8-bit. The
    spacing of the codes present says so; the container does not."""
    rng = np.random.default_rng(3)
    codes = (rng.integers(-58, 59, size=4096) * 256).astype(np.int16)
    lattice = cp.measure_bit_depth(codes)
    assert lattice["bits"] == 8.0
    assert lattice["step"] == 256.0


def test_occupancy_is_not_full_scale():
    """A capture using 116 of 256 codes is still 8-bit. Taking the occupancy
    as the full scale would understate the quantiser's step by over a bit."""
    rng = np.random.default_rng(4)
    codes = (rng.integers(-58, 59, size=4096) * 256).astype(np.int16)
    profile = cp.profile_of(codes, 50e6)
    assert profile["bits"] == 8.0
    assert profile["occupancy"] < 0.6                   # under half used
    assert profile["full_scale"] == pytest.approx(65536.0)


def test_the_quantization_floor_is_the_word_length():
    floor = cp.quantization_floor(8, 50e6, 256.0)
    assert floor["signal_to_noise_db"] == pytest.approx(6.02 * 8 + 1.76)
    assert floor["rms"] == pytest.approx(1.0 / np.sqrt(12.0))
    assert floor["nyquist_hz"] == 25e6


def test_carson_bandwidth_from_the_format_figures():
    """2 x (deviation + baseband). The NTSC VHS SP luma carrier runs
    3.4 MHz at sync tip to 4.4 MHz at peak white over a 3 MHz baseband."""
    assert cp.carson_bandwidth(1.0e6, 3.0e6) == pytest.approx(8.0e6)


def test_the_radio_binds_not_the_converter():
    """Ethan: "since radio is where we got our vhs signal, that is the
    limit". On this capture the converter holds about twice the tape's
    capacity in the signal band, so the tape's own carrier-to-noise is the
    wall and better capture hardware buys nothing."""
    profile = {"bits": 8.0, "sample_rate_hz": 50e6, "full_scale": 256.0}
    limit = cp.binding_limit(tape_spread_db=0.669, signal_rms_codes=24.9,
                             profile=profile)
    assert limit["binds"] == "tape"
    assert limit["ratio"] > 1.5
    assert limit["tape_snr_db"] == pytest.approx(21.9, abs=0.5)
    assert limit["headroom_bits_per_s"] > 0.0


def test_a_quieter_tape_would_eventually_bind_on_the_converter():
    """The verdict is a measurement, not a foregone conclusion: a tape with
    a carrier-to-noise near the converter's own would reverse it."""
    profile = {"bits": 8.0, "sample_rate_hz": 50e6, "full_scale": 256.0}
    pristine = cp.binding_limit(tape_spread_db=0.005, signal_rms_codes=24.9,
                                profile=profile)
    assert pristine["binds"] == "capture"
