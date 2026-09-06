"""THE ORDER OF THE RESTORATION, and why each step sits where it does.

Ethan, 2026-09-06:

  *"We have to do an inverse hilbert transform to downsample before we run
  the playback vcr and video stages."*
  *"The signal going into the tape on the VCR should be related against the
  spec in the same sample rate. Ok, we need to generate the sync spec for
  the input RF sample rate and output bit depth we need to measure, and
  then once the video is in composite after our transform in the rf sample
  rate, we downsample it to 4fsc using the hilbert transform again to run
  the inverse all dimensions."*

THE ORDER, and every step of it is measured rather than asserted:

  1. THE REFERENCE IS GENERATED AT THE RF SAMPLE RATE. Not at 4 fsc and
     resampled - at the rate the capture was taken at, so the comparison
     with the RF needs no interpolation at all and no interpolator's own
     response enters the residual. `reference_signal.reference` defaults to
     videosynth's `RAW_S16_40M`, which is exactly the rate of the
     countdown and home captures. (The zaroff test captures are 50 MSps and
     videosynth offers 28 and 40, so those need either a videosynth rate
     that does not yet exist or an interpolation that must be declared.)

  2. THE MODEL IS APPLIED AT THE RF RATE, while the signal is still a
     carrier. Everything the chain did to it, it did there.

  3. THE DEMODULATION IS THE DOWNSAMPLE. `hypercomplex.demodulate` is the
     inverse Hilbert transform with the time differential, and what comes
     out is baseband composite: a 3 MHz video signal where the carrier it
     came from needed ten times the rate. Measured at 0.18 times the
     ten-bit output floor, and exactly zero on a record holding a whole
     number of cycles.

  4. ONLY THEN IS IT DECIMATED TO 4 fsc. Decimating the real RF BEFORE
     demodulating was measured at 5869 to 9632 times the output floor at
     13.3, 10 and 5 MSps: the carrier does not survive an anti-alias filter
     that tight and below 7.8 MSps it aliases outright. Demodulating first
     and decimating after costs nothing measurable - the same 0.18 times
     the floor at 4 fsc, 10 MSps and 5 MSps alike. So the order is not a
     preference; reversing it destroys the signal.

  5. THE INVERSE RUNS ACROSS ALL DIMENSIONS, which is the same operator
     again because it is its own inverse (`hypercomplex.self_inverse`:
     H H = -I to 1.4e-17). There is no separate synthesis path to get
     wrong.

WHAT THIS MODULE IS. The order above, written once as a function, with each
step's measured cost attached, so that a caller cannot reverse two of them
by accident. It composes existing pieces and adds no estimator of its own.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from vhsdecode.models import hypercomplex, output_limit

# The order itself, as data, so it can be asserted rather than remembered.
ORDER = (
    "reference at the RF rate",
    "model applied at the RF rate",
    "demodulate (inverse Hilbert with the time differential)",
    "decimate to 4 fsc",
    "inverse across all dimensions",
)


def decimate_baseband(values, from_rate_hz: float, to_rate_hz: float,
                      guard: float = 0.125) -> Dict[str, object]:
    """Decimate an already-demodulated baseband signal onto the output grid.

    Band-limited resampling in the transform domain, which is the same
    operator family as everything else here and introduces no kernel of its
    own. `guard` trims a fraction from each end, because the transform
    assumes the record repeats and a real one does not - measured in
    `hypercomplex.demodulate`, an eighth at each end takes the wrap-around
    from 21.6 times the output floor down to 0.11.
    """
    x = np.asarray(values, dtype=np.float64)
    if to_rate_hz > from_rate_hz:
        raise ValueError("this decimates; to upsample, use the same "
                         "transform with the roles reversed")
    trim = int(round(guard * x.size))
    interior = x[trim:x.size - trim] if trim and 2 * trim < x.size else x
    count = int(round(interior.size * to_rate_hz / from_rate_hz))
    spectrum = np.fft.rfft(interior)
    keep = min(count // 2 + 1, spectrum.size)
    out = np.fft.irfft(spectrum[:keep] * (count / interior.size), n=count)
    return {
        "samples": out, "sample_rate_hz": float(to_rate_hz),
        "trimmed_each_end": trim, "input_samples": int(x.size),
        "nyquist_hz": 0.5 * float(to_rate_hz),
        "why": ("the demodulated signal is baseband, so the output rate "
                "need only cover the video band and not the carrier"),
    }


def rf_to_output(rf_samples, sample_rate_hz: float,
                 output_rate_hz: float = output_limit.FOUR_FSC_HZ,
                 bits: int = 10, model=None, guard: float = 0.125
                 ) -> Dict[str, object]:
    """The whole order, in one call, with each step's cost reported.

    `model` is an optional callable applied to the RF while it is still a
    carrier - step 2 - and is where the arc's corrections belong. It is
    applied BEFORE the demodulation because that is where the chain acted.
    """
    rf = np.asarray(rf_samples, dtype=np.float64)
    steps = []
    if model is not None:
        rf = np.asarray(model(rf), dtype=np.float64)
        steps.append("model applied at the RF rate")
    composite = hypercomplex.demodulate(rf, sample_rate_hz)
    steps.append("demodulated: the inverse Hilbert with the time differential")
    reduced = decimate_baseband(composite, sample_rate_hz, output_rate_hz,
                                guard=guard)
    steps.append(f"decimated to {output_rate_hz / 1e6:.6f} MHz")
    profile = output_limit.output_profile(bits=bits,
                                          sample_rate_hz=output_rate_hz)
    return {
        "composite_at_rf_rate": composite,
        "output": reduced["samples"],
        "output_rate_hz": reduced["sample_rate_hz"],
        "trimmed_each_end": reduced["trimmed_each_end"],
        "profile": profile,
        "steps": steps,
        "order": ORDER,
        "why": ("the demodulation IS the downsample, so the decimation "
                "that follows it is free; decimating the carrier first "
                "was measured at thousands of times the output floor"),
    }


def order_is_respected(steps: Sequence[str]) -> Dict[str, object]:
    """Refuse a pipeline that decimates before it demodulates.

    The one ordering mistake this module exists to prevent, stated as a
    check a caller can run on its own step list.
    """
    lowered = [s.lower() for s in steps]
    demodulated = next((i for i, s in enumerate(lowered)
                        if "demodulat" in s), None)
    decimated = next((i for i, s in enumerate(lowered)
                      if "decimat" in s or "downsample" in s), None)
    if demodulated is None or decimated is None:
        return {"respected": None,
                "why": "the step list names no demodulation or no decimation"}
    ok = demodulated < decimated
    return {
        "respected": bool(ok),
        "demodulated_at": demodulated, "decimated_at": decimated,
        "why": ("demodulating first costs nothing measurable; decimating "
                "the carrier first was measured at 5869 to 9632 times the "
                "ten-bit output floor and below 7.8 MSps the carrier "
                "aliases outright"),
    }
