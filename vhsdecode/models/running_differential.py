"""THE DIFFERENTIAL AS A FILE, regenerated and averaged as the decode runs.

Ethan, after the involution was verified:

  *"Let's reframe the problem to that statement. We will need to keep the
  models, but the graph infrastructure will need to be simplified
  dramatically if this works. And I think we can render a differential of
  this model to a file that we only need to regenerate over the fields as
  we decode it in real time, averaged over time as the decode proceeds, it
  measures its own behaviour as we go and that is our residual that we
  average down 2."*

WHAT THE INVOLUTION REMOVES. `hypercomplex.self_inverse` measures that the
multi-axis Hilbert transform is its own inverse - `H H = -I`, `H^4 = I`,
`(-1)^n` on n axes, to 1.4e-17. Three pieces of machinery existed only
because analysis and synthesis were assumed to be different operators, and
all three go:

  * the traced path back up the tree. There is no "back up": the same
    operator applied a second time is the return. `tesseract.trace` stays
    only for the entries that genuinely do NOT commute (a clip, an AGC),
    which is a short ordered list, not a graph.
  * the separate synthesis kernel. `one_real_signal` built a kernel by
    inverting a spectrum; the differential IS the kernel, and applying it
    is the same fold.
  * the dense graph's diagonals as objects to store. Every vertex-to-vertex
    difference is a signed sum of the same 2^n - 1 contrasts, so the file
    holds the contrasts and nothing else: 2^n - 1 spectra, not 2^n(2^n-1)/2
    differences.

What CANNOT go, and is the whole of the remaining risk, is the condition
Ethan attached to it: the models must arrive as the full multi-axis
transform, expanded on every component, or the subtraction leaves the
quadrature partners standing - and the partners are where the delays and
the all-pass terms live (`hypercomplex.relate_to_spec`, `strict=True`).

THE FILE. One npz per decode, holding the contrasts of the current fold,
the field count each was averaged over, the running residual, and the
floor. It is a differential, not a correction: it is what the measurement
departs from the model by, on every component, and applying it is one
multiply in the transform domain.

THE AVERAGING WINDOW IS BOUNDED AT BOTH ENDS, AND BOTH BOUNDS ARE
MEASURED. Averaging N fields lowers the noise as root N, so the window
must be long enough to reach the floor; but the response DRIFTS - measured
on 1024 fields of home, 33 times the median at 0.293 Hz, inside the reels'
own 0.118-0.442 Hz band (`tesseract.from_field_series`) - so a window
longer than that drift's own period averages away the thing it is meant to
track. At 59.94 fields a second a 0.293 Hz line has a period of 3.4 s, so
the window may not exceed about 200 fields, and the noise sets how few it
may be. `window_bounds` computes both from the measured drift and the
measured floor and REFUSES a window outside them.

THE RESIDUAL MEASURES ITSELF. Each field, the prediction from the current
average is compared with that field's own measurement before the field
joins the average; that difference is the residual, and it is out-of-sample
by construction - the field being judged has not yet contributed to what
judges it. No separate held-out split is needed, and none can be forgotten.

AVERAGED DOWN BY TWO. What the model does not explain is, by construction,
indistinguishable from noise by any dimension the model carries. The
optimum linear treatment of a remainder of power S in noise of power N is
the Wiener gain S/(S+N); at S = N that is exactly one half, and applying
the whole remainder would double the error rather than halve it. The gain
is computed from the measured excess, never assumed
(`residual_floor.divided_by_two`).

MEASURED, AND THE FIRST READING OF IT WAS WRONG (2026-09-05, home, 1024
consecutive fields at the recorded flags, the fall-edge sync response per
field over 0.2-4 MHz, one accumulator per head because a field carries one
head and one fall edge).

The window version, which bounded the average below by root-N and above by
a supposed 0.293 Hz drift, chose 23 to 51 fields and left 3.6 to 3.9 times
the floor, refusing a 200-field window as too long for the drift. Then the
lag test above was run and the premise failed: the difference variance is
FLAT against the lag (0.185 at lag 1, 0.253 at lag 64, against the
sixty-four-fold rise a walk would give), so the state is a CONSTANT over
the span plus a small slow term, and the process variance is 1.1e-3 per
field against a measurement variance of 0.34. The correct treatment is
therefore to average the WHOLE capture with the transients rejected, not to
track a drift; the 0.293 Hz line in the field-time spectrum is real but is
far too small to justify a 51-field ceiling.

Two things follow, and both are Ethan's rule rather than an adjustment:
what is constant over the capture is not a component, and a filter whose
process variance is zero has gain zero and averages for ever.

THE TRANSIENTS ARE NOT DRIFT. Thirty-one per cent of consecutive same-head
field pairs jump by more than five times the median step, up to 35 times
it. Those are dropouts and must be rejected before any variance is taken;
with them in, the estimated process variance is 1.9 - a thousand times the
truth - and the filter would refuse to average at all.

AND THE ERROR BAR IS OVERSTATED, so the floor must be measured from the
data rather than quoted. Two readings of a constant differ by twice the
noise, so half the transient-rejected lag-one difference variance IS the
per-field noise, and it disagrees with the instrument's own standard error
in both directions: on home the quoted floor is 3.7 to 4.0 times too
LARGE, on countdown about 2 times too SMALL. Every "times the floor"
figure in this arc that uses the quoted standard error is wrong by that
factor until the instrument's error bar is audited.

THE ANSWER TO "ZERO, OR AN UNRECOVERABLE RESIDUAL", measured consistently:
the level removed, the transients rejected, and the floor taken from the
data rather than quoted.

    tape        head   change about the mean, over the measured floor
    countdown    A     0.94        AT THE FLOOR
    countdown    B     0.96        AT THE FLOOR
    home         A     2.71
    home         B     3.07

On countdown the answer is ZERO: after the level, the per-field sync
response carries nothing above its own noise, on both heads. On home it is
an unrecoverable residual of about three times the floor. The two tapes
differ and the model does not, so what is left on home belongs to that
recording rather than to the chain.

AND THE TRANSIENTS ARE MOST OF WHAT LOOKED LIKE STRUCTURE: they carry 84
to 86 per cent of the change on home and 54 to 67 per cent on countdown,
across the 93 to 112 fields of 512 that a five-times-median gate rejects.
A model fitted without rejecting them is fitting dropouts.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import residual_floor, tesseract

# The format's own version, so a file written by an older build is refused
# rather than half-read.
FORMAT_VERSION = 1


# WHAT IS CONSTANT OVER A CAPTURE IS NOT A COMPONENT (Ethan: *"Time over the
# span of the RF capture is constant so that is not on our component to
# extract out. That's how you treat known constants in this."*). A quantity
# that does not vary over the span carries no dimension: it is fixed, not
# fitted, and giving it a state would let the fit spend evidence on
# something there is nothing to learn about. In filter terms its process
# variance is zero, so its gain is zero and it is averaged for ever - which
# is the same statement.
KNOWN_CONSTANTS = {
    "any": {
        "capture clock": (
            "the capture card's sample rate over one file. The card is "
            "free-running crystal-referenced hardware and the file is one "
            "unbroken acquisition, so the rate is one number for the whole "
            "span; the time-base variation a decode sees is the medium's "
            "transport, not the card's. Fix it from the capture profile, "
            "never fit it."),
        "specified line and field rates": (
            "the standard's f_H and f_V are definitions, not measurements; "
            "the medium's departure from them is the measurement and the "
            "reference itself is exact."),
    },
    "VHS": {
        "which head reads which field": (
            "the drum carries two heads at opposite azimuth and lays down "
            "one field per half revolution, so the head alternates every "
            "field for the whole capture. Which head is reading is KNOWN "
            "from the field index, not inferred: the head axis of a cube is "
            "a known partition, and only the DIFFERENCE between the heads "
            "is a measurement. A model that spends a parameter deciding "
            "which head it is looking at is fitting something the format "
            "already fixed."),
        "subcarrier ratio": (
            "227.5 cycles a line (455/2) and the colour-under's 40 f_H are "
            "defined ratios; no component may carry either as a parameter."),
        "azimuth sign": (
            "plus and minus six degrees, SMPTE 32M-2004 clause 3, alternating "
            "with the head: a known sign, not a fitted one."),
    },
    "LaserDisc": {
        "rotation": (
            "constant to specification - 1800 rpm exactly in CAV, and in "
            "CLV a rate the standard fixes against radius - so the "
            "rotation is a known constant and only its DEPARTURE is a "
            "measurement. Ethan's example, and the same rule as the VHS "
            "head: what the format fixes is not a dimension."),
    },
}


def constants_for(fmt: str = "VHS") -> Dict[str, str]:
    """What this format fixes, and therefore what may not be a component.

    Ethan: *"Like which head is reading the field is constant over the rf
    capture, the rotation part of the laserdisc is constant to spec, etc."*
    and *"this is for VHS but may be of other interests to other formats"* -
    so the list is per format, with the format-independent entries always
    included. An unknown format returns only those, rather than pretending
    to know what it fixes.
    """
    out = dict(KNOWN_CONSTANTS["any"])
    out.update(KNOWN_CONSTANTS.get(fmt, {}))
    return out


def refuse_constant_component(name: str, fmt: str = "VHS") -> None:
    """Raise if a component would fit something the format already fixes."""
    fixed = constants_for(fmt)
    for constant, why in fixed.items():
        if constant.lower() in name.lower() or name.lower() in constant.lower():
            raise ValueError(
                f"{name!r} is a known constant of {fmt} and may not be a "
                f"component: {why}")


def is_constant_over_span(series, measurement_variance,
                          lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
                          reject_above: float = 5.0) -> Dict[str, object]:
    """Is the state a constant over this capture, or does it walk?

    THE TEST, and it needs no model. The variance of `x_{k+L} - x_k` is
    `2r` for a constant read through noise of variance `r`, and `2r + L q`
    for a random walk of step variance `q`: FLAT against the lag, or rising
    in proportion to it. Transients are rejected first (a dropout is not a
    step of the state), by the same `reject_above` multiple of the median
    the rest of this module uses.

    MEASURED (home, 512 same-head fields, the fall-edge response over
    0.2-4 MHz): the difference variance runs 0.185, 0.184, 0.201, 0.221,
    0.200, 0.213, 0.253 at lags 1 to 64 - a rise of a third across a
    sixty-four-fold change of lag, against the sixty-four-fold rise a walk
    would give. The response is a CONSTANT over the span plus a small slow
    term, exactly as Ethan said the time base is, and the right treatment
    is to average the whole capture rather than to track it.

    AND THE SAME NUMBERS CONVICT THE ERROR BAR. Two independent readings of
    a constant differ by `2r`, and the instrument's own per-bin standard
    error gives `2r = 0.688` against the 0.185 measured. The difference
    cannot be smaller than the noise unless the noise is smaller than
    claimed: the standard error is overstated by about 3.7 in variance,
    just under 2 in amplitude - or consecutive fields share part of their
    noise. Every "times the floor" figure computed from that standard
    error is therefore optimistic by up to that factor, this module's
    included, and the instrument's error bar needs its own audit.
    """
    values = np.asarray(series, dtype=np.complex128)
    r = float(np.mean(np.asarray(measurement_variance, dtype=np.float64)))
    rows = []
    for lag in lags:
        if lag >= values.shape[0]:
            continue
        difference = values[lag:] - values[:-lag]
        power = np.mean(np.abs(difference) ** 2, axis=tuple(
            range(1, difference.ndim)))
        keep = power < reject_above * np.median(power)
        rows.append({"lag": int(lag), "variance": float(power[keep].mean()),
                     "kept": int(keep.sum()), "of": int(keep.size)})
    if len(rows) < 2:
        raise ValueError("too few lags to tell a constant from a walk")
    first, last = rows[0]["variance"], rows[-1]["variance"]
    span = rows[-1]["lag"] / max(rows[0]["lag"], 1)
    growth = last / max(first, 1e-30)
    # a walk grows in proportion to the lag; a constant does not grow at all
    walking = growth > 1.0 + 0.25 * (span - 1.0)
    implied_q = max((last - first) / max(rows[-1]["lag"] - rows[0]["lag"], 1), 0.0)
    return {
        "rows": rows, "quoted_measurement_variance": r,
        "variance_at_shortest_lag": first, "variance_at_longest_lag": last,
        "growth": growth, "lag_span": span,
        "is_constant": not walking,
        "implied_process_variance": implied_q,
        "error_bar_overstated_by": float(2.0 * r / max(first, 1e-30)),
        "why": ("the difference of two readings of a constant is twice the "
                "noise whatever the lag; a walk's difference grows with it, "
                "and the shortest lag also bounds the noise itself"),
    }


def steady_state_gain(process_variance: float, measurement_variance: float
                      ) -> Dict[str, float]:
    """THE GAIN, DERIVED RATHER THAN CHOSEN: model-based filtering.

    Ethan: *"I think this is model based filtering on a complex signal."*
    It is, and naming it supplies the one number the window version still
    picked by hand.

    The state is the differential, which drifts (measured: the transport
    band, 0.293 Hz on home); the measurement is one field's response, which
    carries the instrument's noise. Write the drift as a random walk of
    variance `q` per field and the measurement as that state plus noise of
    variance `r`. The steady-state prediction variance solves

        P = P r / (P + r) + q   ->   P^2 - q P - q r = 0
        P = (q + sqrt(q^2 + 4 q r)) / 2,     K = P / (P + r)

    and the filter is `x <- x + K (z - x)`. No window, no tuning: `K` comes
    from two measured variances, and `1/K` is the number of fields the
    filter effectively averages, which is what `window_bounds` was
    approximating between its two bounds.

    AND THIS IS WHERE ETHAN'S DIVISION BY TWO COMES FROM. At `P = r` - the
    model's own uncertainty equal to the measurement's noise - the gain is
    exactly one half. "Average it down by two" is not a convention: it is
    the optimum at the crossover, and the filter simply computes what the
    optimum is when the two are not equal.

    ON A COMPLEX SIGNAL the noise is circularly symmetric, so the same real
    gain applies to both parts and no phase is preferred - which is why the
    gain may be a scalar here while every inner product in this arc must
    still stack the real and imaginary parts rather than use a Hermitian
    form.
    """
    q = float(max(process_variance, 0.0))
    r = float(max(measurement_variance, 1e-300))
    if q <= 0.0:
        return {"process_variance": 0.0, "measurement_variance": r,
                "prediction_variance": 0.0, "gain": 0.0,
                "effective_fields": float("inf"),
                "why": "a state that does not move is averaged for ever"}
    prediction = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
    gain = prediction / (prediction + r)
    return {
        "process_variance": q, "measurement_variance": r,
        "prediction_variance": prediction, "gain": float(gain),
        "effective_fields": float(1.0 / gain),
        "at_the_half": bool(abs(prediction - r) < 0.05 * r),
        "why": ("the steady-state Kalman gain for a drifting state read "
                "through a noisy measurement; one half exactly when the "
                "model's uncertainty equals the measurement's noise"),
    }


def measure_process_variance(series, measurement_variance) -> Dict[str, float]:
    """The drift's own variance per field, from consecutive differences.

    `z_k - z_{k-1}` carries the state's step (variance `q`) plus two
    independent measurement noises (variance `2r`), so `q` is the observed
    difference variance less twice the measurement variance. A negative
    result means the state does not move faster than the noise can see,
    and is reported as zero rather than as a negative variance.
    """
    values = np.asarray(series, dtype=np.complex128)
    if values.shape[0] < 3:
        raise ValueError("a drift needs at least three fields to be seen")
    steps = np.diff(values, axis=0)
    observed = float(np.mean(np.abs(steps) ** 2))
    r = float(np.mean(np.asarray(measurement_variance, dtype=np.float64)))
    q = observed - 2.0 * r
    return {"observed_step_variance": observed, "measurement_variance": r,
            "process_variance": float(max(q, 0.0)),
            "resolved": bool(q > 0.0),
            "why": ("consecutive measurements differ by the state's step "
                    "plus two independent noises, so the step's variance is "
                    "what is left after twice the measurement variance")}


def window_bounds(field_rate_hz: float, drift_hz: float,
                  floor_variance: float, target_variance: float
                  ) -> Dict[str, float]:
    """How many fields may be averaged: enough for the floor, few enough
    for the drift.

    The lower bound is root-N: to bring a per-field variance `floor_variance`
    down to `target_variance` needs `floor/target` fields. The upper bound
    is the drift: a window longer than a quarter of the drift's period
    averages across its own variation (a quarter period is where a sine
    departs from its mean by more than half its amplitude).
    """
    lower = float(max(floor_variance / max(target_variance, 1e-30), 1.0))
    period_fields = field_rate_hz / max(drift_hz, 1e-30)
    upper = float(0.25 * period_fields)
    return {
        "minimum_fields": lower, "maximum_fields": upper,
        "drift_hz": float(drift_hz), "drift_period_s": 1.0 / max(drift_hz, 1e-30),
        "feasible": bool(lower <= upper),
        "why": ("root-N sets how few fields reach the floor and the drift "
                "sets how many may be pooled before the average tracks "
                "something that is no longer there"),
    }


def choose_window(bounds: Dict[str, float]) -> int:
    """The geometric mean of the two bounds, refused when they cross."""
    if not bounds["feasible"]:
        raise ValueError(
            "no window satisfies both bounds: the floor needs at least "
            f"{bounds['minimum_fields']:.0f} fields and the drift allows at "
            f"most {bounds['maximum_fields']:.0f}; the measurement is not "
            "precise enough per field to track this drift")
    return int(round(np.sqrt(bounds["minimum_fields"] * bounds["maximum_fields"])))


class RunningDifferential:
    """The differential over a moving window of fields, measuring itself.

    `axes` are the binary measurement axes of the cube each field supplies
    (head, polarity, ...); `window` is the field count from `choose_window`.
    """

    def __init__(self, axes: Sequence[str], bins, window: Optional[int] = None,
                 field_rate_hz: float = 60.0 / 1.001,
                 reject_above: Optional[float] = 5.0):
        """`window` None means the state is a CONSTANT over the span and
        every field is pooled - which is what `is_constant_over_span`
        measures on this material, and what Ethan's rule says to do with a
        quantity the capture does not change. `reject_above` drops a field
        whose departure from the current model exceeds that multiple of the
        running median departure: a dropout is not a step of the state, and
        with the transients in, the pooled model chases them."""
        self.axes = tuple(axes)
        self.bins = np.asarray(bins, dtype=np.float64)
        self.window = None if window is None else int(window)
        self.field_rate_hz = float(field_rate_hz)
        self.reject_above = reject_above
        self.rejected: List[int] = []
        self._departures: List[float] = []
        self._fields: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self.residual_power = 0.0
        self.floor_power = 0.0
        self.judged = 0
        self.history: List[Dict[str, float]] = []

    # -- the model over the window ----------------------------------------
    def _cube(self) -> Optional[tesseract.Cube]:
        if not self._fields:
            return None
        values = np.mean([v for _i, v, _w in self._fields], axis=0)
        variance = np.mean([w for _i, _v, w in self._fields],
                           axis=0) / len(self._fields)
        return tesseract.Cube(self.axes, values, variance, bins=self.bins)

    def contrasts(self) -> Optional[Dict[Tuple[str, ...], Dict[str, np.ndarray]]]:
        cube = self._cube()
        return None if cube is None else tesseract.walsh(cube)

    def predict(self) -> Optional[np.ndarray]:
        """The current model's vertices, from its identified contrasts only."""
        cube = self._cube()
        if cube is None:
            return None
        contrasts = tesseract.walsh(cube)
        verdicts = tesseract.identified(contrasts)
        keep = [s for s, v in verdicts.items() if v["identified"] or not s]
        return tesseract.reconstruct(cube, contrasts, keep)

    # -- the loop ----------------------------------------------------------
    def update(self, index: int, values, variance) -> Dict[str, object]:
        """Judge one field against the current model, then admit it.

        THE ORDER IS THE POINT. The field is compared BEFORE it joins the
        window, so every residual this reports is out-of-sample and no
        separate hold-out can be forgotten.
        """
        values = np.asarray(values, dtype=np.complex128)
        variance = np.asarray(variance, dtype=np.float64)
        report: Dict[str, object] = {"field": int(index),
                                     "fields_in_window": len(self._fields)}
        prediction = self.predict()
        if prediction is not None:
            departure = values - prediction
            power = float(np.mean(np.abs(departure) ** 2))
            floor = float(np.mean(variance)) * (1.0 + 1.0 / len(self._fields))
            self.residual_power += power
            self.floor_power += floor
            self.judged += 1
            report.update({
                "residual_power": power, "floor_power": floor,
                "over_floor": power / max(floor, 1e-30),
                "running_over_floor": (self.residual_power
                                       / max(self.floor_power, 1e-30)),
            })
            self.history.append({"field": int(index),
                                 "over_floor": report["over_floor"]})
            # a transient is not a step of the state: judge it, report it,
            # and keep it out of the model
            if self.reject_above is not None and len(self._departures) >= 8:
                limit = float(self.reject_above * np.median(self._departures))
                if power > limit:
                    self.rejected.append(int(index))
                    report["rejected"] = True
                    self._departures.append(power)
                    return report
            self._departures.append(power)
        self._fields.append((int(index), values, variance))
        if self.window is not None and len(self._fields) > self.window:
            self._fields.pop(0)
        return report

    # -- what to do with what is left --------------------------------------
    def verdict(self) -> Dict[str, object]:
        """The self-measured residual, and the gain it has earned."""
        if not self.judged:
            return {"judged": 0, "why": "no field has been judged yet"}
        over = self.residual_power / max(self.floor_power, 1e-30)
        half = residual_floor.divided_by_two(max(over - 1.0, 0.0))
        return {
            "judged": self.judged,
            "residual_over_floor": over,
            "excess_db": float(10.0 * np.log10(max(over, 1e-30))),
            "at_floor": bool(abs(over - 1.0) <= 3.0 * np.sqrt(2.0 / self.judged)),
            "gain": half["wiener_gain"],
            "half": half,
            "rejected": len(self.rejected),
            "pooled": len(self._fields),
            "why": ("every field was judged before it entered the window, so "
                    "this is out of sample; the gain is the Wiener optimum "
                    "for what is left, which is one half when the "
                    "unexplained power equals the floor"),
        }

    # -- the file ----------------------------------------------------------
    def write(self, path: str, metadata: str = "") -> Dict[str, object]:
        """Render the differential to a file: the contrasts, and nothing else.

        The dense graph's differences are all signed sums of these, so
        storing the contrasts stores the graph.
        """
        contrasts = self.contrasts()
        if contrasts is None:
            raise ValueError("nothing has been accumulated to write")
        names = [":".join(s) if s else "mean" for s in contrasts]
        stack = np.array([contrasts[s]["value"] for s in contrasts])
        noise = np.array([contrasts[s]["variance"] for s in contrasts])
        out = {
            "format_version": np.int64(FORMAT_VERSION),
            "axes": np.array(self.axes),
            "frequency_hz": self.bins,
            "contrast_names": np.array(names),
            "contrast": stack,
            "contrast_variance": noise,
            "window_fields": np.int64(self.window),
            "fields_in_window": np.int64(len(self._fields)),
            "field_rate_hz": np.float64(self.field_rate_hz),
            "judged": np.int64(self.judged),
            "residual_power": np.float64(self.residual_power),
            "floor_power": np.float64(self.floor_power),
            "metadata": np.array(metadata),
        }
        np.savez(path, **out)
        return {"path": path, "contrasts": len(names),
                "bytes": os.path.getsize(path) if os.path.exists(path) else 0}


def read(path: str) -> Dict[str, object]:
    """Read a differential file, refusing a version this build cannot read."""
    data = np.load(path, allow_pickle=False)
    version = int(data["format_version"])
    if version != FORMAT_VERSION:
        raise ValueError(f"differential file is version {version}, this build "
                         f"reads version {FORMAT_VERSION}")
    return {key: data[key] for key in data.files}


def apply_differential(values, stored: Dict[str, object],
                       vertex: Sequence[int], gain: float = 0.5
                       ) -> np.ndarray:
    """Subtract the stored differential at one vertex, at the earned gain.

    The vertex's departure is the signed sum of the contrasts on the axes
    where its state is 1 - the dense graph read out of the stored
    contrasts - and it is applied in the log domain, at `gain`, which is
    the Wiener half by default and errs low as the arc's correction-gain
    law requires.
    """
    names = [str(n) for n in stored["contrast_names"]]
    axes = [str(a) for a in stored["axes"]]
    total = np.zeros(np.asarray(values).shape, dtype=np.complex128)
    for name, contrast in zip(names, stored["contrast"]):
        if name == "mean":
            continue
        sign = 1.0
        for axis in name.split(":"):
            if vertex[axes.index(axis)]:
                sign = -sign
        total = total + sign * contrast
    return np.asarray(values, dtype=np.complex128) - float(gain) * total
