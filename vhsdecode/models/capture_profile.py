"""The capture chain: the fourth link, and the one that terminates the rest.

Ethan named the chain as four links, where the component model carried only
three:

    Picture chain (TV to VCR); VCR chain: VCR -> Tape; Tape -> VCR;
    Capture chain: VCR -> capture card; capture card profile -> complete
    limits of the sample rate i.e. 8bits, and 40mhz

The capture card's profile is the complete limit of everything downstream of
it, so it is measured here rather than asserted. Three constants previously
stood in `information_extrapolation` - eight bits, forty megahertz and a
13.3 MHz roll-off - and on the captures this arc actually uses, the rate is
fifty megahertz, so the criterion string those constants produced was wrong
on every tape. Reading the profile from the file is both the project's
standing rule (no hard-coded constants) and the only way the number is right
on the next capture.

WHY THE CODE LATTICE. The bit depth is read from the spacing of the codes
the file actually contains, not from the container's word length: an 8-bit
capture carried in a 16-bit container is still an 8-bit capture, and a flag
that says otherwise is wrong in a way nothing catches. The full scale is the
range the converter CAN produce, not the range it used - a capture occupying
116 of 256 codes is still 8-bit, and taking the occupancy as full scale
would understate the quantiser's step by more than a bit.

WHAT IT IS FOR. Two numbers come out of it and both terminate something:

  * the quantisation floor, which no component's residual may claim to sit
    below, because below it there is no information to have measured;
  * the capture's Shannon capacity in the signal band, which answers whether
    the converter or the tape is the binding limit. On the captures here the
    answer is the tape, by a factor of two - see `channel_capacity`.
"""

from typing import Dict, Optional

import numpy as np

# A uniform quantiser's noise variance is one twelfth of a step squared.
QUANTIZATION_NOISE_DIVISOR = 12.0

# Carson's rule needs the deviation and the baseband width; both are format
# figures taken from the manufacturer's guide by the caller, never assumed
# here. These are the NTSC VHS SP luma values (JVC VTG82063 section 3): the
# carrier runs 3.4 MHz at sync tip to 4.4 MHz at peak white, so the
# deviation is 1.0 MHz, over a 3.0 MHz luma baseband.
NTSC_VHS_SP_DEVIATION_HZ = 1.0e6
NTSC_VHS_SP_BASEBAND_HZ = 3.0e6


def measure_bit_depth(samples: np.ndarray) -> Dict[str, float]:
    """The converter's word length, from the lattice the codes sit on.

    The spacing between the distinct values present is the quantiser step,
    whatever container the samples arrived in. The greatest common divisor
    of the differences is that step: a run of 8-bit codes left-shifted into
    16 bits has a step of 256 and 8 bits of depth, and reading the container
    instead would report 16.
    """
    values = np.unique(np.asarray(samples).ravel())
    if values.size < 2:
        return {"bits": 0.0, "step": 0.0, "codes": float(values.size),
                "full_scale": 0.0}
    differences = np.diff(values)
    differences = differences[differences > 0]
    if not differences.size:
        return {"bits": 0.0, "step": 0.0, "codes": float(values.size),
                "full_scale": 0.0}
    # integer captures land on an exact lattice; a gcd finds it without
    # assuming every adjacent code is present
    rounded = np.rint(differences).astype(np.int64)
    if np.all(rounded > 0) and np.allclose(differences, rounded, atol=1e-9):
        step = float(np.gcd.reduce(rounded))
    else:
        step = float(np.min(differences))
    if not step > 0:
        return {"bits": 0.0, "step": 0.0, "codes": float(values.size),
                "full_scale": 0.0}
    # THE FULL SCALE IS THE CONTAINER'S RANGE, NOT THE DATA'S SPAN. A
    # capture that happens to sit quietly still passes through the same
    # converter, so deriving the scale from how much of it the signal used
    # reports a quiet capture as having fewer bits than it has - and this
    # capture uses 117 of 256 codes, so that error is over a bit here.
    dtype = np.asarray(samples).dtype
    if np.issubdtype(dtype, np.integer):
        span = float(np.iinfo(dtype).max) - float(np.iinfo(dtype).min) + 1.0
    else:
        # a float container carries no declared range; fall back to what the
        # data spans, rounded out to whole bits
        span = float(np.max(np.abs(values))) * 2.0 + step
    bits = float(np.round(np.log2(max(span / step, 2.0))))
    return {
        "bits": bits,
        "step": step,
        "codes": float(values.size),
        "full_scale": step * (2.0 ** bits),
    }


def quantization_floor(bits: float, sample_rate_hz: float,
                       full_scale: float) -> Dict[str, float]:
    """The capture's own arithmetic noise floor.

    The one part of the limit that needs no measurement once the profile is
    known: a uniform quantiser of this word length has a noise variance of
    one twelfth of a step squared, flat across the band, because
    quantisation is added at the converter after everything analogue.
    """
    step = float(full_scale) / (2.0 ** float(bits)) if bits > 0 else 0.0
    power = step * step / QUANTIZATION_NOISE_DIVISOR
    nyquist = float(sample_rate_hz) / 2.0
    return {
        "step": step,
        "power": power,
        "rms": float(np.sqrt(power)),
        "density_per_hz": power / max(nyquist, 1e-12),
        "nyquist_hz": nyquist,
        "signal_to_noise_db": 6.02 * float(bits) + 1.76,
    }


def in_band_noise_rms(floor: Dict[str, float], band_hz: float) -> float:
    """The quantisation noise inside one band.

    Quantisation is flat to Nyquist, so a band holds the fraction of its
    power that its width is of the Nyquist span. The envelope of a carrier
    takes the amplitude quadrature alone, which is half the power - hence
    the root two.
    """
    nyquist = max(float(floor["nyquist_hz"]), 1e-12)
    return float(floor["rms"]) * float(np.sqrt(
        max(float(band_hz), 0.0) / nyquist)) / float(np.sqrt(2.0))


def carson_bandwidth(deviation_hz: float, baseband_hz: float) -> float:
    """The band an FM carrier occupies, from the format's own figures."""
    return 2.0 * (float(deviation_hz) + float(baseband_hz))


def channel_capacity(band_hz: float, signal_to_noise: float) -> float:
    """Shannon, in bits per second."""
    return float(band_hz) * float(np.log2(1.0 + max(signal_to_noise, 0.0)))


def binding_limit(tape_spread_db: float, signal_rms_codes: float,
                  profile: Dict[str, float],
                  deviation_hz: float = NTSC_VHS_SP_DEVIATION_HZ,
                  baseband_hz: float = NTSC_VHS_SP_BASEBAND_HZ
                  ) -> Dict[str, float]:
    """Which link binds: the tape's radio channel, or the capture card?

    Ethan: *"since radio is where we got our vhs signal, that is the
    limit"*. Measured on the captures this arc uses, that is right and the
    margin is a factor of two.

    The tape's carrier-to-noise comes from the within-field spread of the
    envelope per sample - the FM carrier is nominally constant in amplitude,
    so what the envelope does between samples is the channel's own noise.
    The capture's comes from its quantisation inside the same band. Both are
    then Shannon over the band Carson's rule gives for the format.
    """
    band_hz = carson_bandwidth(deviation_hz, baseband_hz)
    floor = quantization_floor(profile["bits"], profile["sample_rate_hz"],
                               profile["full_scale"])
    relative = 10.0 ** (float(tape_spread_db) / 20.0) - 1.0
    tape_snr = 1.0 / max(relative, 1e-30) ** 2
    quantisation = in_band_noise_rms(floor, band_hz)
    # the signal is measured in the same codes the step is
    capture_snr = (float(signal_rms_codes) * floor["step"]
                   / max(quantisation, 1e-30)) ** 2
    tape = channel_capacity(band_hz, tape_snr)
    capture = channel_capacity(band_hz, capture_snr)
    return {
        "band_hz": band_hz,
        "tape_snr_db": 10.0 * float(np.log10(max(tape_snr, 1e-30))),
        "capture_snr_db": 10.0 * float(np.log10(max(capture_snr, 1e-30))),
        "tape_capacity_bits_per_s": tape,
        "capture_capacity_bits_per_s": capture,
        "headroom_bits_per_s": capture - tape,
        "ratio": capture / max(tape, 1e-30),
        "binds": "tape" if tape <= capture else "capture",
    }


def profile_of(samples: np.ndarray, sample_rate_hz: float,
               roll_off_hz: Optional[float] = None) -> Dict[str, float]:
    """The capture card's complete profile, read from the capture."""
    lattice = measure_bit_depth(samples)
    values = np.asarray(samples, dtype=np.float64).ravel()
    profile = {
        "sample_rate_hz": float(sample_rate_hz),
        "nyquist_hz": float(sample_rate_hz) / 2.0,
        "signal_rms": float(values.std()) if values.size else 0.0,
        "peak_to_peak": float(np.ptp(values)) if values.size else 0.0,
    }
    profile.update(lattice)
    if lattice["step"] > 0:
        profile["signal_rms_codes"] = profile["signal_rms"] / lattice["step"]
        profile["occupancy"] = (profile["peak_to_peak"] / lattice["step"] + 1.0
                                ) / max(2.0 ** lattice["bits"], 1.0)
    else:
        profile["signal_rms_codes"] = 0.0
        profile["occupancy"] = 0.0
    profile.update(quantization_floor(profile["bits"],
                                      profile["sample_rate_hz"],
                                      profile["full_scale"]))
    if roll_off_hz is not None:
        profile["roll_off_hz"] = float(roll_off_hz)
    # the information budget the converter can carry at all
    profile["budget_bits_per_s"] = profile["bits"] * profile["sample_rate_hz"]
    return profile
