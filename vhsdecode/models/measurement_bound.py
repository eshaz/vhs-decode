"""WHAT A MEASURED REGION CAN POSSIBLY TELL YOU, before anything is fitted.

Ethan, 2026-09-06: *"The amount of possible memory is the limit of
information contained in the measured area. Therefore this is where the
hilbert space comes in. Each measurement is bound in accuracy by the
amount of data it has to use for measurement."*

THE BOUND IS ARITHMETIC AND COMES FIRST. A region of duration T carrying
signal over a bandwidth B holds 2BT real numbers and no more - the
sampling theorem, and the reason a probe of duration T resolves nothing
finer than 1/T. Whatever is measured on that region is a function of those
numbers, so its accuracy is bounded by them however the estimator is
built. That bound can be computed before any data is looked at, and a
measurement that claims to beat it has an error somewhere.

WHY IT IS THE HILBERT SPACE. The 2BT real numbers are BT COMPLEX ones -
the analytic signal's samples - and that is the natural count, because a
band-limited real signal is fully described by its analytic form at the
band's own rate. So the dimension of the space a region's measurements
live in is BT, and every quantity measured there is a projection onto one
of those dimensions. Two quantities measured on the same region are not
independent unless their projections are orthogonal, which is exactly what
`hypercomplex.relate` measures.

WHAT THE ARC'S OWN REGIONS HOLD, computed below rather than asserted: the
line sync pulse is 4.7 microseconds and the luma band 3 MHz, so one pulse
carries 14 complex dimensions; the whole vertical interval carries 1716;
one field's worth of sync pulses, 241 of them, carries 3400. Those are
ceilings on everything measured there, and they say directly why the
sync pulse can carry a level and a slope and a curvature and not a
fifty-parameter shape.

AND THE ACCURACY FOLLOWS. With `n` complex dimensions and a signal-to-
noise ratio `s` per dimension, a single parameter's relative standard
error cannot beat `1 / sqrt(n s)` - the Cramer-Rao bound for a linear
parameter in Gaussian noise. Pooling more regions raises `n` and nothing
else does; a better estimator cannot.

WHAT THE BOUND SAYS ABOUT THIS ARC, with the signal-to-noise MEASURED
rather than assumed. The per-sample scatter inside the sync pulse's flat
interior is 0.742, 0.895 and 0.780 IRE on the three decodes, which against
a 40 IRE depth is a per-dimension ratio of 2905, 1997 and 2633. So the
bound on a depth measurement is:

    tape        one pulse            pooled over a field
    countdown   0.198 IRE  4.3 floors   0.0127 IRE  0.28 floors
    home        0.238      5.2          0.0154      0.33
    bars        0.208      4.5          0.0134      0.29

Pooled over a single field the bound is already BELOW the ten-bit output
floor. And the arc's own held-out residual on the ringing work is 0.118
IRE, which is EIGHT TO NINE TIMES that bound - about 19 dB of headroom
that the data contains and the model is not using.

THAT IS THE USEFUL FORM OF ETHAN'S STATEMENT. The information in the
measured area is finite and computable, and knowing it separates two very
different situations: a residual sitting at the bound is finished, and one
sitting above it is a modelling shortfall with the evidence already in
hand. This arc's is the second. (An earlier version of this note used an
assumed signal-to-noise of 100 and found the residual sitting exactly at
the bound; that agreement was an artefact of the assumption and is
withdrawn.)
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np


def dimensions(duration_s: float, bandwidth_hz: float) -> Dict[str, float]:
    """How many independent numbers a region can hold.

    `2 B T` real, `B T` complex. The complex count is the one to use,
    because a band-limited real signal is fully described by its analytic
    form at the band's own rate, and because every quantity in this model
    is carried complex.
    """
    product = float(duration_s) * float(bandwidth_hz)
    return {
        "duration_s": float(duration_s),
        "bandwidth_hz": float(bandwidth_hz),
        "time_bandwidth_product": product,
        "real_dimensions": 2.0 * product,
        "complex_dimensions": product,
        "resolution_hz": 1.0 / float(duration_s) if duration_s > 0 else float("inf"),
        "why": ("a region of duration T over bandwidth B holds 2BT real "
                "numbers, so nothing measured on it can carry more"),
    }


def best_accuracy(duration_s: float, bandwidth_hz: float,
                  signal_to_noise: float, parameters: int = 1
                  ) -> Dict[str, float]:
    """The floor on a measurement's relative error, before any estimator.

    For `p` parameters shared over `n` complex dimensions at a per-
    dimension signal-to-noise ratio `s`, the Cramer-Rao bound on each is
    `sqrt(p / (n s))`. Two consequences worth stating: doubling the region
    buys a factor of root two and nothing more, and every extra parameter
    costs the same root, which is why a model's parameter count is not
    free even when the fit improves.
    """
    counted = dimensions(duration_s, bandwidth_hz)
    n = counted["complex_dimensions"]
    if n <= 0 or signal_to_noise <= 0:
        return {**counted, "relative_error": float("inf"), "parameters": parameters}
    relative = math.sqrt(float(parameters) / (n * float(signal_to_noise)))
    return {
        **counted,
        "signal_to_noise": float(signal_to_noise),
        "parameters": int(parameters),
        "relative_error": relative,
        "decibels": 20.0 * math.log10(relative),
        "why": ("the Cramer-Rao bound for a linear parameter in Gaussian "
                "noise; a better estimator cannot beat it and only more "
                "region or more signal can move it"),
    }


def against_output(relative_error: float, quantity: float,
                   floor_ire: float, units_are_ire: bool = True
                   ) -> Dict[str, object]:
    """Is the bound already finer than the output file can show?

    The arc's cap is a ten-bit 4 fsc file with a quantisation noise of
    0.0461 IRE (`output_limit`). A measurement whose own floor sits under
    that is as good as it needs to be, and effort spent improving it buys
    nothing the file can carry.
    """
    absolute = float(relative_error) * abs(float(quantity))
    return {
        "absolute_error": absolute,
        "floor": float(floor_ire),
        "over_floor": absolute / max(float(floor_ire), 1e-30),
        "already_enough": bool(absolute < float(floor_ire)),
        "why": ("a measurement finer than the file's own quantisation is "
                "finished, however much region is left"),
    }


REGIONS = {
    "line sync pulse": (4.7e-6, "ITU-R BT.1700, the pulse's own width"),
    "front porch": (1.5e-6, "ITU-R BT.1700"),
    "colour burst": (9.0 / 3579545.4545, "9 cycles, ITU-R BT.1700"),
    "back porch": (1.6e-6, "the interval from the burst to active picture"),
    "whole line blanking": (10.9e-6, "ITU-R BT.1700, line blanking"),
    "equalizing sequence": (190.667e-6, "3 lines of the vertical interval"),
    "whole vertical interval": (572.0e-6, "9 lines"),
}


def survey(bandwidth_hz: float = 3.0e6,
           signal_to_noise: float = 100.0,
           lines_per_field: int = 241) -> Dict[str, object]:
    """Every region the arc measures on, and what each can hold.

    `bandwidth_hz` defaults to the VHS luma baseband, 3 MHz, which is what
    limits every one of these regions; `lines_per_field` is how many
    picture lines a field contributes when the region repeats per line.
    """
    rows = []
    for name, (duration, cite) in REGIONS.items():
        one = best_accuracy(duration, bandwidth_hz, signal_to_noise)
        pooled = best_accuracy(duration * lines_per_field, bandwidth_hz,
                               signal_to_noise)
        rows.append({
            "region": name, "cite": cite,
            "duration_us": duration * 1e6,
            "complex_dimensions": one["complex_dimensions"],
            "resolution_khz": one["resolution_hz"] / 1e3,
            "relative_error": one["relative_error"],
            "pooled_over_a_field": pooled["relative_error"],
            "pooled_dimensions": pooled["complex_dimensions"],
        })
    rows.sort(key=lambda r: r["complex_dimensions"])
    return {"rows": rows, "bandwidth_hz": bandwidth_hz,
            "signal_to_noise": signal_to_noise,
            "lines_per_field": lines_per_field,
            "why": ("the ceiling on every measurement taken there, fixed "
                    "before any estimator is chosen")}
