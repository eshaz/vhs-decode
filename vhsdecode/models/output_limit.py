"""THE OUTPUT FILE IS THE CAP: 4 fsc, ten bits, and no further.

Ethan, 2026-09-06:

  *"I am interested in modeling enough things that will get us to the limit
  of our output composite data, 4fsc, 10bit and no further."*
  *"I just need to use the components that are meaningfully influential in
  this graph to generate a perfect output file."*
  *"Cap the constant to be the sample rate and bitdepth of the output file
  only. 4fsc 10bits."*

WHY THIS CHANGES THE PROBLEM. Every stopping rule this arc has used
compares the residual against a floor the INSTRUMENT sets - a standard
error, a cross-line variance, a chi-square. Those floors move with the
instrument and two of them have already been found wrong by a factor of
four. The output file's floor does not move: it is arithmetic, fixed by
the container before any measurement, and nothing below it can appear in
the file whatever the model does. So it is the honest cap, and a component
whose effect on the output is below it is not small, it is INVISIBLE.

THE TWO LIMITS, and both are exact.

  BANDWIDTH. Four times the colour subcarrier is 14.318 MHz, so the file
  represents nothing above 7.159 MHz. The VHS luma baseband is 3 MHz
  (SMPTE 32M clause 3.9.1.1.4 by way of the deviation), so the format is
  comfortably inside the container and bandwidth is not the binding limit
  here - but a model term above Nyquist is unrepresentable, not merely
  attenuated, and must not be fitted.

  AMPLITUDE. Ten bits over the container's full scale. Read from a real
  decode's own parameters rather than assumed - black at 16243.2 and white
  at 56320.0 of 65536 give 400.768 units per IRE, so the full scale spans
  163.5 IRE - a ten-bit step is 64 of those units, 0.1597 IRE, and a
  uniform quantiser's noise is the step over root twelve:

      0.0461 IRE rms,  which is 66.7 dB below 100 IRE

  Eight bits would give 0.1844 IRE and sixteen 0.0007 IRE; the arc's
  current output is sixteen-bit, so ten bits is a DELIBERATE cap Ethan is
  setting, not a limitation of the container.

WHAT IT DECIDES. Three things, and each one prunes rather than tunes.

  1. WHICH COMPONENTS ARE MEANINGFUL. A component earns its place if the
     change it makes to the output exceeds 0.0461 IRE rms. That is a
     physical criterion, not a statistical one: it does not care how
     confidently the component was measured, only whether the file can
     show it.

  2. HOW FAR TO AVERAGE. The per-line noise on this material is 0.9218 IRE
     (the arc's own measurement, `docs/CHANNEL_LOOP_PROPOSAL.md`), so
     reaching the output floor needs (0.9218/0.0461)^2 = 400 lines pooled,
     and pooling further buys nothing the file can show. Two thousand and
     sixteen lines - eight fields - reaches 0.0205 IRE, already
     comfortably past it, which is why the arc's accumulated measurements
     are limited by their model and not by their evidence.

  3. WHEN THE MODEL IS FINISHED. Not when the residual reaches the tape's
     noise, which is unreachable, but when it reaches the file's floor.

THE CAPPED BUDGET. `tesseract.budget` conserves a total, and Ethan's cap
says which part of that total the file can hold: the change inside the
band, above the quantisation floor. `capped_budget` intersects the two and
returns the fraction of the conserved constant that is REPRESENTABLE, which
is the only part worth modelling.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

# Four times the NTSC colour subcarrier, the arc's output grid. Derived
# from the line rate rather than typed: f_sc = 455/2 x f_H, f_H = 525 x 30
# / 1.001 (SMPTE 170M).
NTSC_LINE_RATE_HZ = 525.0 * 30.0 / 1.001
NTSC_SUBCARRIER_HZ = 455.0 / 2.0 * NTSC_LINE_RATE_HZ
FOUR_FSC_HZ = 4.0 * NTSC_SUBCARRIER_HZ

# A uniform quantiser's noise variance is one twelfth of a step squared.
QUANTISATION_DIVISOR = 12.0

# The per-line noise this arc measured on its own material, in IRE. Quoted
# with its source so a different material replaces it rather than inherits
# it.
MEASURED_LINE_NOISE_IRE = 0.9218
MEASURED_LINE_NOISE_SOURCE = (
    "docs/CHANNEL_LOOP_PROPOSAL.md, the cross-line noise of one line on "
    "the accumulated sync fold; the noise of the mean over 2016 lines is "
    "0.0205 IRE")


def output_profile(bits: int = 10, sample_rate_hz: float = FOUR_FSC_HZ,
                   black_16b: float = 16243.2, white_16b: float = 56320.0
                   ) -> Dict[str, float]:
    """The file's own floor and ceiling, from the container's parameters.

    `black_16b` and `white_16b` are the decode's own scale (its JSON's
    `black16bIre` and `white16bIre`); the defaults are what this arc's
    decodes carry, and a caller with a different decode passes its own
    rather than inheriting these.
    """
    units_per_ire = (float(white_16b) - float(black_16b)) / 100.0
    if units_per_ire <= 0:
        raise ValueError("white must sit above black on the container's scale")
    step_units = 2.0 ** (16 - int(bits))
    step_ire = step_units / units_per_ire
    noise_ire = step_ire / math.sqrt(QUANTISATION_DIVISOR)
    return {
        "bits": int(bits),
        "sample_rate_hz": float(sample_rate_hz),
        "nyquist_hz": 0.5 * float(sample_rate_hz),
        "units_per_ire": units_per_ire,
        "full_scale_ire": 65536.0 / units_per_ire,
        "step_ire": step_ire,
        "quantisation_noise_ire": noise_ire,
        "below_100_ire_db": 20.0 * math.log10(100.0 / noise_ire),
        "why": ("the container's floor is arithmetic and fixed before any "
                "measurement, so nothing below it can reach the file "
                "however well the model is fitted"),
    }


def output_format(json_path: Optional[str] = None) -> Dict[str, object]:
    """The output data format itself, read from a decode rather than assumed.

    Ethan named the cap as "the sample rate and bitdepth of the output file
    only", and the file is a `.tbc`: unsigned samples at four times the
    subcarrier, one field per raster, with its geometry and its IRE scale in
    the JSON beside it. The container is SIXTEEN bit and Ethan is capping
    the meaningful content at TEN, so the cap is a decision about what is
    worth modelling and not a limitation of the format - which is exactly
    why it must be recorded here rather than inferred from the file.

    Measured on this arc's decodes: 910 samples a line, 263 lines a field,
    14.318182 MHz, black at 16243.2 and white at 56320.0 of 65536.
    """
    geometry = {"field_width": 910, "field_height": 263,
                "sample_rate_hz": FOUR_FSC_HZ,
                "black_16b": 16243.2, "white_16b": 56320.0,
                "container_bits": 16, "source": "this arc's own decodes"}
    if json_path:
        import json
        with open(json_path) as handle:
            parameters = json.load(handle)["videoParameters"]
        geometry = {"field_width": int(parameters["fieldWidth"]),
                    "field_height": int(parameters["fieldHeight"]),
                    "sample_rate_hz": float(parameters["sampleRate"]),
                    "black_16b": float(parameters["black16bIre"]),
                    "white_16b": float(parameters["white16bIre"]),
                    "container_bits": 16, "source": json_path}
    samples = geometry["field_width"] * geometry["field_height"]
    meaningful = output_profile(bits=10,
                                sample_rate_hz=geometry["sample_rate_hz"],
                                black_16b=geometry["black_16b"],
                                white_16b=geometry["white_16b"])
    return {
        **geometry,
        "samples_per_field": samples,
        "container_bits_per_field": samples * geometry["container_bits"],
        "capped_bits_per_field": samples * meaningful["bits"],
        "profile": meaningful,
        "why": ("the container holds sixteen bits and the cap keeps ten, so "
                "the difference is a modelling decision that must be stated "
                "rather than a property of the file"),
    }


def is_meaningful(effect_ire, profile: Optional[Dict[str, float]] = None,
                  ) -> Dict[str, object]:
    """Is a component's effect on the output large enough for the file?

    `effect_ire` is the rms change the component makes to the output, in
    IRE. The comparison is against the quantisation noise, not the step:
    a departure smaller than the noise the quantiser already adds cannot
    be distinguished from it.
    """
    profile = profile or output_profile()
    effect = float(np.sqrt(np.mean(np.asarray(effect_ire, dtype=float) ** 2)))
    floor = profile["quantisation_noise_ire"]
    return {
        "effect_ire": effect,
        "floor_ire": floor,
        "over_floor": effect / max(floor, 1e-300),
        "meaningful": bool(effect > floor),
        "margin_db": 20.0 * math.log10(max(effect, 1e-300) / floor),
        "why": ("a component whose effect is under the file's own "
                "quantisation noise is not small, it is invisible"),
    }


def lines_to_reach_floor(line_noise_ire: float = MEASURED_LINE_NOISE_IRE,
                         profile: Optional[Dict[str, float]] = None
                         ) -> Dict[str, float]:
    """How many lines must be pooled before the file's floor is reached.

    Averaging N lines divides the noise power by N, so reaching the floor
    needs `(line_noise / floor)^2` lines and pooling further buys nothing
    the file can show. On this arc's material that is 400 lines, and the
    eight-field fold it already uses reaches 0.0205 IRE, which is past it.
    """
    profile = profile or output_profile()
    floor = profile["quantisation_noise_ire"]
    needed = (float(line_noise_ire) / max(floor, 1e-300)) ** 2
    return {
        "line_noise_ire": float(line_noise_ire),
        "floor_ire": floor,
        "lines_needed": needed,
        "fields_needed": needed / 252.0,      # usable lines a field supplies
        "source": MEASURED_LINE_NOISE_SOURCE,
        "why": ("root-N against a fixed floor: past this the evidence is no "
                "longer the limit, the model is"),
    }


def capped_budget(budget: Dict[str, object], frequency_hz,
                  signal_ire: float = 100.0,
                  profile: Optional[Dict[str, float]] = None
                  ) -> Dict[str, object]:
    """Ethan's cap applied to the conserved constant.

    `budget` is `tesseract.budget`'s result, whose total is in the log
    domain (nepers squared, summed over bins and vertices). A departure of
    `x` nepers on a signal of `signal_ire` IRE moves the output by about
    `x * signal_ire` IRE for small `x`, so the log-domain floor that
    corresponds to the file's quantisation noise is `floor_ire /
    signal_ire`, and the bins above Nyquist are outside the file entirely.

    Returns how much of the conserved change is REPRESENTABLE: inside the
    band and above the floor. That is the only part worth modelling, and
    it is a fraction of a constant fixed before any fitting.
    """
    profile = profile or output_profile()
    f = np.asarray(frequency_hz, dtype=float).ravel()
    in_band = f <= profile["nyquist_hz"]
    floor_nepers = profile["quantisation_noise_ire"] / max(signal_ire, 1e-300)
    change = float(budget["change"])
    # the floor's share of the same total: one floor-sized departure in
    # every representable bin of every vertex
    bins = int(in_band.sum())
    vertices = float(budget["total"]) and 2 ** len(budget["by_order"])
    floor_total = float(floor_nepers ** 2 * bins * vertices)
    representable = max(change - floor_total, 0.0)
    return {
        "bins_in_band": bins,
        "bins_out_of_band": int((~in_band).sum()),
        "nyquist_hz": profile["nyquist_hz"],
        "floor_nepers": floor_nepers,
        "floor_share_of_change": floor_total / max(change, 1e-300),
        "representable_change": representable,
        "representable_fraction": representable / max(change, 1e-300),
        "signal_ire": float(signal_ire),
        "why": ("the conserved constant is bounded above by what the file "
                "can hold: inside the band and over the quantiser's noise"),
    }


def rank(effects_ire: Dict[str, float],
         profile: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Every component ordered by its effect on the output, cut at the floor.

    This is the admission test Ethan asked for, and it replaces a
    statistical one with a physical one: not "is this component measured
    confidently" but "can the file show it".
    """
    profile = profile or output_profile()
    rows = []
    for name, effect in effects_ire.items():
        verdict = is_meaningful(effect, profile)
        rows.append({"component": name, **verdict})
    rows.sort(key=lambda r: -r["effect_ire"])
    keep = [r["component"] for r in rows if r["meaningful"]]
    return {
        "rows": rows, "meaningful": keep,
        "invisible": [r["component"] for r in rows if not r["meaningful"]],
        "floor_ire": profile["quantisation_noise_ire"],
        "why": ("the cut is the file's own quantisation noise, so it does "
                "not move with the instrument that measured the component"),
    }
