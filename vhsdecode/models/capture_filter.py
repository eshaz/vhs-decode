"""The analogue low-pass between the RF tap and the converter.

Ethan: *"I also have a low pass filter just after the RF tap on my VCR. Model
that based on the noise profile of the capture, where it rolls off consistent
with an analog low pass filter. There may be many orders in the filter. The
comparison component is the nyquist roll off."*

THE NOISE IS THE PROBE, WHICH IS WHY THIS NEEDS NO STIMULUS. Above the FM
signal's own band the capture carries nothing but noise, and the noise
entering the filter is broadband. So the measured spectrum up there IS the
filter's `|H(f)|^2`, scaled - the one component in this whole arc that can be
identified without a synthetic reference, because the reference is flatness.

WHAT THE MEASUREMENT ACTUALLY SHOWS, and it is not a single roll-off. Pooled
over four tape positions of the off-air capture, 16384-point Welch, spurs
rejected against a running median, at a rate of 40 MSps established from the
data rather than assumed (the FM carrier lands at 3.78 MHz, mid-band for
VHS NTSC's 3.4 to 4.4 MHz; at 50 MSps it would fall above peak white):

     6 MHz  -42.0 dB        15 MHz  -50.5 dB
     8 MHz  -43.0           16 MHz  -59.6
    10 MHz  -45.5           17 MHz  -62.1
    12 MHz  -43.6           18 MHz  -62.8
    13 MHz  -45.3           19 MHz  -63.0
    14 MHz  -47.5        19.9 MHz  -62.9

A FILTER DOES NOT FLATTEN. The floor falls to -63 dB by 17 MHz and then sits
there to Nyquist, and no analogue low-pass does that - a pole cascade keeps
falling. A flat floor beneath a falling one is a SECOND, ADDITIVE term that
the filter cannot act on because it is added after it: the converter's own
noise, quantisation included. So the measured spectrum is

    measured(f) = |H(f)|^2 * S_in(f) * |aperture(f)|^2  +  converter floor

and the filter is identifiable only while the first term is above the second
- here, below about 16 MHz. Above that the measurement is reporting the
converter, not the filter, and a fit that ignores this reads the converter's
flatness as the filter running out of poles. `identify` therefore fits the
floor as a free term and reports the frequency above which the filter is no
longer observable.

THE COMPARISON COMPONENT, WHICH IS ETHAN'S AND IS THE RIGHT ONE TO ASK FOR.
The sampler itself imposes a roll-off that is not the filter's and is known
exactly: a sample-and-hold of aperture equal to the sample period has
response `sinc(f/fs)`, -3.92 dB at Nyquist, with no parameters at all. It is
the yardstick because it is FIXED.

THE TWO DIFFER IN KIND AND ARE SEPARABLE - BUT NOT BY THIS EVIDENCE, AND THE
DIFFERENCE BETWEEN THOSE TWO STATEMENTS IS THE WHOLE POINT.

This took three passes to get right and the wrong turns are kept because they
are instructive. The first version argued that a sinc and a pole cascade
differ in KIND - the sinc has nulls and linear phase, the cascade has neither
- and are therefore well separated. `separability` was written to demonstrate
that and appeared to REFUTE it, reporting the aperture 0.94 to 0.997 coherent
with Butterworth cascades, as collinear with them as they are with each
other. That refutation was recorded and the claim withdrawn.

THE CONTROL WAS ITSELF THE DEFECT. It compared LOG MAGNITUDES only, so it was
measuring the one half in which a sinc and a pole cascade genuinely do look
alike - both are smooth, monotone and gentle below Nyquist. Carrying the
phase each actually has, the same comparison over the same band reads

    2 poles 0.4038    4 poles 0.4916    6 poles 0.5154    8 poles 0.5239

against 0.934 to 0.999 for the cascades among themselves. The aperture is
plainly a different kind of object and the cascades are plainly one family
differing in a rate, which is the separability law reading exactly as it
should. The original claim was right; the instrument that appeared to refute
it was blind in the half that carries the answer.

The KIND difference is specific and derivable rather than asserted: a
sample-and-hold is a half-sample delay, so its phase is exactly LINEAR,
`-pi f / fs`. A pole cascade is minimum phase, so its phase follows its own
magnitude. Those cannot be confused once both are looked at.

AND THE ORDER BELOW IS STILL CONDITIONAL, FOR A DIFFERENT REASON THAN FIRST
RECORDED. Not because the two are collinear - they are not - but because THE
EVIDENCE HERE IS A NOISE POWER SPECTRUM, and a power spectrum has no phase at
all. The half that separates them is exactly the half a noise floor cannot
supply. So the fitted order remains conditional on the aperture being the
assumed `sinc(f/fs)`, and the way to break that is not a better fit but a
different measurement: a phase-bearing probe through the same path, or
evidence above Nyquist where the sinc's null lives.

THE FIT, on the floor above, over 5 to 19.9 MHz, 6104 points:

    order 15.5 poles     corner 14.23 MHz     rms 1.78 dB
    converter floor -62.96 dB     filtered noise -42.4 dB
    observable to 16.4 MHz, above which the filter is under the converter

The order is interior to the search grid rather than railed against it - it
holds at 15.5 whether the grid stops at 20 or at 30 - so "there may be many
orders" is confirmed rather than assumed. Fifteen poles is more than one
section, which is consistent with a multi-section anti-alias filter, and the
9 dB step between 15 and 16 MHz is steeper than any Butterworth corner and
hints at an elliptic transition or a decimation edge that this two-parameter
family cannot represent. That is the honest residual: 1.78 dB rms is a good
fit and not a perfect one, and the misfit is concentrated exactly there.

AND ALIASING IS WHY THE FILTER IS THERE. Everything above Nyquist folds back
onto `fs - f`, so the filter's job is to put the fold below the converter
floor before it can land on the signal. `fold_suppression` reports whether it
succeeds, which is the only question the filter's order actually has to
answer.
"""

from typing import Dict, Optional, Sequence

import numpy as np

# The sample-and-hold aperture is the FULL sample period unless a converter
# states otherwise. This is the only assumption in the comparison component,
# and it is the conventional one: a track-and-hold with a shorter aperture
# rolls off LESS, so assuming the full period is the conservative direction.
APERTURE_FRACTION = 1.0

CHAIN_PREFIX = "capture low-pass"
# The capture is position 30 in the chain - the last thing that happens to
# the signal, and so the FIRST thing an inversion undoes.
CHAIN_POSITION = 30

COMPONENT_POSITIONS: Dict[str, int] = {
    CHAIN_PREFIX: CHAIN_POSITION,
    "capture low-pass response": CHAIN_POSITION,
    "capture sampler aperture": CHAIN_POSITION,
}


def nyquist_hz(sample_rate_hz: float) -> float:
    return float(sample_rate_hz) / 2.0


def aperture_response(frequency_hz, sample_rate_hz: float,
                      fraction: float = APERTURE_FRACTION) -> np.ndarray:
    """THE COMPARISON COMPONENT: the sampler's own roll-off, `sinc(f/fs)`.

    Fixed, parameter-free, and -3.92 dB at Nyquist. It is the yardstick the
    analogue filter is measured against, and it is the reason the filter's
    order can be read at all: anything falling faster than this is the
    filter.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    x = float(fraction) * grid / max(float(sample_rate_hz), 1.0)
    return np.sinc(x)


def butterworth_magnitude(frequency_hz, corner_hz: float, order: float
                          ) -> np.ndarray:
    """`1 / sqrt(1 + (f/fc)^(2n))` - the maximally flat n-pole magnitude.

    Butterworth is the reference family because it is the one whose
    magnitude is fully determined by two numbers, so a fit that prefers a
    different family has to earn the difference. Its asymptote is exactly
    `6n` dB per octave, which is what makes the order readable off a slope.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    ratio = grid / max(float(corner_hz), 1.0)
    with np.errstate(over="ignore"):
        return 1.0 / np.sqrt(1.0 + ratio ** (2.0 * float(order)))


def order_from_slope(db_per_octave: float) -> float:
    """A pole cascade falls 6 dB per octave per pole, so the order is the
    slope over six. Stated as a function because it is the whole content of
    'there may be many orders'."""
    return float(db_per_octave) / 6.0


def model_floor(frequency_hz, sample_rate_hz: float, corner_hz: float,
                order: float, filtered_db: float, converter_db: float
                ) -> np.ndarray:
    """The measured floor as the model predicts it, in dB.

    Two ADDITIVE terms and not one. The filtered noise passes the analogue
    filter and the sampler's aperture; the converter's own noise is added
    afterwards and neither term touches it. Adding them in POWER, which is
    what makes the flat tail come out flat.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    shaped = (butterworth_magnitude(grid, corner_hz, order)
              * aperture_response(grid, sample_rate_hz)) ** 2
    return 10.0 * np.log10(
        shaped * 10.0 ** (float(filtered_db) / 10.0)
        + 10.0 ** (float(converter_db) / 10.0))


def identify(frequency_hz, measured_db, sample_rate_hz: float,
             band: Optional[Sequence[float]] = None,
             orders: Sequence[float] = tuple(np.arange(1.0, 13.5, 0.5)),
             ) -> Dict[str, object]:
    """FIT THE FILTER TO A MEASURED NOISE FLOOR.

    Four free parameters - corner, order, the filtered noise's level and the
    converter's floor - fitted over a band the caller chooses. The converter
    floor is FREE AND NOT ASSUMED, because it is the term that decides where
    the filter stops being observable, and fixing it would smuggle in the
    answer to the question the fit exists to ask.

    Returns the fitted parameters, the frequency above which the filter is
    below the converter and therefore unmeasurable, and how much of the total
    fall the sampler's aperture accounts for - which is the comparison Ethan
    asked for, stated as a number.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    observed = np.asarray(measured_db, dtype=np.float64).ravel()
    if grid.size != observed.size:
        raise ValueError("the grid and the measured floor must match")
    low, high = (band if band is not None
                 else (0.1 * nyquist_hz(sample_rate_hz),
                       0.999 * nyquist_hz(sample_rate_hz)))
    use = (grid >= float(low)) & (grid <= float(high)) & np.isfinite(observed)
    if use.sum() < 8:
        return {"fitted": False, "why": "too few points in the band"}

    f, y = grid[use], observed[use]
    best = None
    # the converter floor cannot be above the quietest thing measured, and a
    # decade below it can no longer matter
    tail = float(np.percentile(y, 5))
    for order in orders:
        for corner in np.linspace(0.10 * high, 1.20 * high, 60):
            for converter in np.linspace(tail - 12.0, tail + 3.0, 16):
                shaped = (butterworth_magnitude(f, corner, order)
                          * aperture_response(f, sample_rate_hz)) ** 2
                # the filtered level enters linearly in dB, so solve it
                # rather than searching it
                floor_lin = 10.0 ** (converter / 10.0)
                def residual(level_db):
                    return y - 10.0 * np.log10(
                        shaped * 10.0 ** (level_db / 10.0) + floor_lin)
                lo, hi = tail - 10.0, float(np.max(y)) + 30.0
                for _ in range(40):                      # bisect on the mean
                    mid = 0.5 * (lo + hi)
                    if residual(mid).mean() > 0:
                        lo = mid
                    else:
                        hi = mid
                level = 0.5 * (lo + hi)
                error = float(np.sqrt((residual(level) ** 2).mean()))
                if best is None or error < best["rms_db"]:
                    best = {"order": float(order), "corner_hz": float(corner),
                            "filtered_db": float(level),
                            "converter_db": float(converter),
                            "rms_db": error}
    if best is None:
        return {"fitted": False, "why": "no candidate fitted"}

    nyq = nyquist_hz(sample_rate_hz)
    shaped_db = 20.0 * np.log10(np.maximum(
        butterworth_magnitude(grid, best["corner_hz"], best["order"]), 1e-30))
    with_aperture = shaped_db + 20.0 * np.log10(np.maximum(
        np.abs(aperture_response(grid, sample_rate_hz)), 1e-30))
    # where the filtered term drops below the converter's own floor
    filtered_db = best["filtered_db"] + with_aperture
    below = grid[filtered_db < best["converter_db"]]
    return {
        "fitted": True,
        **best,
        "poles": best["order"],
        "db_per_octave": 6.0 * best["order"],
        "observable_to_hz": float(below.min()) if below.size else float(nyq),
        "aperture_db_at_nyquist": float(20.0 * np.log10(
            abs(float(aperture_response([nyq], sample_rate_hz)[0])))),
        "filter_db_at_nyquist": float(20.0 * np.log10(max(float(
            butterworth_magnitude([nyq], best["corner_hz"],
                                  best["order"])[0]), 1e-30))),
        "band": (float(low), float(high)),
        "points": int(use.sum()),
        "why": ("the converter's floor is fitted rather than assumed, "
                "because it decides where the filter stops being observable "
                "and fixing it would smuggle in the answer"),
    }


def fold_suppression(sample_rate_hz: float, corner_hz: float, order: float,
                     converter_db: float, filtered_db: float,
                     reach: float = 3.0) -> Dict[str, object]:
    """WHETHER THE FILTER DOES ITS JOB: is the aliased fold below the floor?

    Everything above Nyquist folds onto `fs - f`, so the filter's only real
    duty is to put that fold under the converter's own noise before it can
    land on the signal band. This integrates the folded power the fitted
    filter passes and compares it with the floor - which is the question the
    ORDER exists to answer, and a better use for the order than quoting it.
    """
    nyq = nyquist_hz(sample_rate_hz)
    above = np.linspace(nyq, float(reach) * nyq, 4096)
    passed = (butterworth_magnitude(above, corner_hz, order)
              * aperture_response(above, sample_rate_hz)) ** 2
    folded = float(np.trapezoid(passed, above)) if hasattr(np, "trapezoid") \
        else float(np.trapz(passed, above))
    folded_db = 10.0 * np.log10(max(folded / max(nyq, 1.0), 1e-30)) \
        + float(filtered_db)
    return {
        "folded_db": folded_db,
        "converter_db": float(converter_db),
        "margin_db": float(converter_db - folded_db),
        "below_the_floor": bool(folded_db < float(converter_db)),
        "integrated_to_hz": float(reach * nyq),
        "why": ("the fold lands on fs - f, so the filter's duty is to put it "
                "under the converter's own noise; the margin is what the "
                "order buys"),
    }


def separability(sample_rate_hz: float, corner_hz: float,
                 orders: Sequence[float] = (2.0, 4.0, 6.0, 8.0)
                 ) -> Dict[str, object]:
    """Can the analogue roll-off be told from the sampler's own?

    THE CONTROL, and it can fail. If the aperture's sinc were collinear with
    a pole cascade, no measurement could attribute the fall to either, and
    every order reported here would be an artefact of that degeneracy. The
    separability law says they differ in KIND - a sinc has nulls and linear
    phase, a cascade has neither - so they should be well separated, and this
    measures it instead of asserting it.

    The pole cascades among THEMSELVES differ only in a rate, so they are
    expected to be strongly coherent with each other; that contrast is what
    makes the aperture's low coherence meaningful rather than a scale.
    """
    from scipy.signal import hilbert

    grid = np.linspace(0.05 * nyquist_hz(sample_rate_hz),
                       0.99 * nyquist_hz(sample_rate_hz), 1024)

    def shape(log_magnitude, phase):
        """A unit shape carrying BOTH parts, stacked real-then-imaginary.

        The first version of this took only the log magnitude, which is why
        it reported the two as collinear - it was measuring the one half in
        which they genuinely do look alike.
        """
        vector = np.concatenate([log_magnitude - log_magnitude.mean(),
                                 phase - phase.mean()])
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    aperture_log = np.log(np.maximum(
        np.abs(aperture_response(grid, sample_rate_hz)), 1e-30))
    # A SAMPLE-AND-HOLD IS A HALF-SAMPLE DELAY, so its phase is exactly
    # LINEAR in frequency: -pi f / fs. That is the KIND difference, and it is
    # derived rather than Hilbert-transformed because a sinc is not minimum
    # phase and the minimum-phase relation would be the wrong construction
    # for it.
    aperture = shape(aperture_log, -np.pi * grid / float(sample_rate_hz))

    poles = {}
    for n in orders:
        logged = np.log(np.maximum(
            butterworth_magnitude(grid, corner_hz, n), 1e-30))
        # a pole cascade IS minimum phase, so its phase follows its magnitude
        poles[n] = shape(logged, -np.imag(hilbert(logged)))
    against = {f"{n} poles": float(abs(aperture @ v)) for n, v in poles.items()}
    names = list(poles)
    among = [float(abs(poles[a] @ poles[b]))
             for i, a in enumerate(names) for b in names[i + 1:]]
    return {
        "aperture_against_poles": against,
        "worst_aperture_coherence": max(against.values()),
        "poles_among_themselves": among,
        "worst_pole_coherence": max(among) if among else 0.0,
        "separable": bool(max(against.values()) < 0.9),
        "why": ("the pole cascades differ from each other in a RATE and are "
                "near-collinear; the aperture differs in KIND and is not, "
                "which is what lets the fit attribute the fall"),
    }


def signatures(frequency_hz, sample_rate_hz: float,
               corner_hz: Optional[float] = None,
               order: Optional[float] = None) -> Dict[str, np.ndarray]:
    """The capture filter's entries for the key.

    The aperture is emitted always, because it is parameter-free and known.
    The analogue filter is emitted only when a corner and an order have been
    IDENTIFIED from a measurement - there is no default, because a default
    would be a filter nobody measured.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    out = {
        "capture sampler aperture":
            aperture_response(grid, sample_rate_hz).astype(np.complex128),
    }
    if corner_hz is not None and order is not None:
        out["capture low-pass response"] = butterworth_magnitude(
            grid, corner_hz, order).astype(np.complex128)
    return out
