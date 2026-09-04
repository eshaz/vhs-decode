"""Every claim in the elliptical-collapse documents, as runnable evidence.

    python3 -m tools.ringing_measure.proofs                 # all of them
    python3 -m tools.ringing_measure.proofs --list          # what there is
    python3 -m tools.ringing_measure.proofs --only 7 12     # a few
    python3 -m tools.ringing_measure.proofs --report out.md # write it down

Each proof regenerates a number that appears in `docs/THE_ALGORITHM.md`,
`docs/ELLIPTICAL_COLLAPSE.md` or `docs/COMPONENT_MAPPINGS.md`, states what it
expected, and says PASS or FAIL against it. Nothing here reads a capture, so
the whole suite runs in seconds and can be run by anyone; the proofs that
need a decode are named at the end and marked as such.

WHY THIS FILE EXISTS. Four numbers that were reported during this work turned
out to be artefacts of their own construction, and one of them - the
real-data run - was reproduced exactly by structureless noise. A claim that
cannot be regenerated on demand is not a result, so every claim that survived
is here with the code that produces it, including the proofs that
*established the failures*. Proofs 12 to 15 are those; they are expected to
demonstrate the defect, and they FAIL if the defect stops reproducing,
because that would mean the record no longer matches the code.
"""

import argparse
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from vhsdecode.models import capture_profile as cp
from vhsdecode.models import clipping as clp
from vhsdecode.models import head_model
from vhsdecode.models import information_extrapolation as ie
from vhsdecode.models import interference as inf
from vhsdecode.models import pair_dimension as pdim
from vhsdecode.models import magnetic as mag
from vhsdecode.models import standard_levels as sl

PROOFS: List[Tuple[str, str, Callable[[], Dict]]] = []


def proof(title: str, cites: str):
    def register(function):
        PROOFS.append((title, cites, function))
        return function
    return register


def _ok(condition: bool, claim: str, got, expected) -> Dict:
    return {"pass": bool(condition), "claim": claim,
            "got": got, "expected": expected}


def _noise(rng, length, complex_valued=True):
    value = rng.standard_normal(length)
    if complex_valued:
        value = value + 1j * rng.standard_normal(length)
    return value


# --------------------------------------------------------------------------
# The transform's own arithmetic
# --------------------------------------------------------------------------


@proof("The ellipse is the constant: trace(G) = N for any ensemble",
       "THE_ALGORITHM.md section 3; ELLIPTICAL_COLLAPSE.md section 2")
def trace_is_the_component_count():
    rng = np.random.default_rng(1)
    length, count = 128, 4
    shared = _noise(rng, length)
    cases = {
        "independent": {f"c{i}": {"frequency": _noise(rng, length)}
                        for i in range(count)},
        "collinear": {f"c{i}": {"frequency": shared * (1.0 + 0.02 * i)}
                      for i in range(count)},
        "wildly different norms": {
            f"c{i}": {"frequency": _noise(rng, length) * 10.0 ** (3 * i - 4)}
            for i in range(count)},
    }
    traces = {name: ie.ellipsoid(r, "frequency")["trace"]
              for name, r in cases.items()}
    worst = max(abs(v - count) for v in traces.values())
    return {"lines": [f"{name:>24}: trace {value:.10f}"
                      for name, value in traces.items()],
            "checks": [_ok(worst < 1e-9,
                           "trace equals the component count whatever the "
                           "components do",
                           f"worst departure {worst:.2e}", "< 1e-9")],
            "note": "It is an identity of unit-normalising each row, not a "
                    "discovery. What it licenses is the next sentence: only "
                    "the SHAPE carries information."}


@proof("The sphere floor N/(L+N-1), derived and measured",
       "THE_ALGORITHM.md section 5; ELLIPTICAL_COLLAPSE.md section 3")
def sphere_floor_holds():
    rng = np.random.default_rng(2)
    lines, ratios = [], []
    for count in (3, 4, 8, 16, 32):
        for length in (64, 256, 1024):
            # 300 trials, not 20: at N=3 the estimate's own standard error
            # is 3 per cent of the prediction, so a short run reads a
            # shortfall that is not there. Measured at 20 trials: 0.797.
            trials = [ie.ellipsoid(
                {f"c{i}": {"frequency": _noise(rng, length)}
                 for i in range(count)}, "frequency")["asymmetry"]
                for _ in range(300)]
            predicted = count / (length + count - 1.0)
            ratio = float(np.mean(trials)) / predicted
            ratios.append(ratio)
            if length == 256:
                lines.append(f"  N={count:>3} L={length:>5}: predicted "
                             f"{predicted:.5f}  measured "
                             f"{np.mean(trials):.5f}  ratio {ratio:.3f}")
    return {"lines": lines,
            "checks": [_ok(0.85 < min(ratios) and max(ratios) < 1.20,
                           "measured over predicted has no trend and stays "
                           "near one",
                           f"{min(ratios):.3f} to {max(ratios):.3f}",
                           "0.85 to 1.20")],
            "note": "The formula is a delta-method approximation that happens "
                    "to be accurate. Its SCATTER is a separate matter - see "
                    "proof 14, which shows the shipped scatter is the complex "
                    "large-N asymptote and wrong for a real ensemble."}


@proof("The Marchenko-Pastur edge is where noise stops",
       "THE_ALGORITHM.md section 4")
def marchenko_pastur_edge():
    rng = np.random.default_rng(3)
    lines, ratios = [], []
    for count in (4, 8, 16):
        for length in (64, 256, 1024):
            top = [ie.ellipsoid({f"c{i}": {"frequency": _noise(rng, length)}
                                 for i in range(count)},
                                "frequency")["eigenvalues"][0]
                   for _ in range(20)]
            edge = (1.0 + np.sqrt(count / length)) ** 2
            ratio = float(np.mean(top)) / edge
            ratios.append(ratio)
            lines.append(f"  N={count:>3} L={length:>5}: edge {edge:.4f}  "
                         f"largest eigenvalue {np.mean(top):.4f}  "
                         f"ratio {ratio:.3f}")
    return {"lines": lines,
            "checks": [_ok(max(ratios) < 1.0,
                           "on noise the largest eigenvalue stays below the "
                           "edge at these sizes",
                           f"worst {max(ratios):.3f}", "< 1.000")],
            "note": "The shortfall is the finite-size Tracy-Widom effect and "
                    "shrinks with L. At larger sizes the exceedance rate "
                    "settles at a few per cent, so 'never admits noise' is "
                    "true only at the sizes tested. The edge does NOT apply "
                    "at all to a nested matrix - see proof 12."}


@proof("The Wiener weight equals the correction-gain law",
       "THE_ALGORITHM.md section 6; ELLIPTICAL_COLLAPSE.md section 3.1")
def wiener_is_the_gain_law():
    lines, worst = [], 0.0
    for eigenvalue in (1.2, 1.5, 2.0, 3.0, 6.0, 12.0):
        snr = eigenvalue - 1.0
        wiener = (eigenvalue - 1.0) / eigenvalue
        law = 1.0 / (1.0 + 1.0 / snr)
        worst = max(worst, abs(wiener - law))
        lines.append(f"  lambda {eigenvalue:>5.2f}  SNR {snr:>5.2f}  "
                     f"(l-1)/l {wiener:.6f}  1/(1+rho) {law:.6f}")
    return {"lines": lines,
            "checks": [_ok(worst < 1e-12,
                           "the two expressions are identical for every "
                           "eigenvalue",
                           f"worst difference {worst:.2e}", "< 1e-12"),
                       _ok(abs((2.0 - 1.0) / 2.0 - ie.ORTHOGONAL_AMOUNT) < 1e-12,
                           "the flat half-step is the case SNR = 1",
                           f"{(2.0-1.0)/2.0}", f"{ie.ORTHOGONAL_AMOUNT}")],
            "note": "This is a rearrangement of one MMSE expression, not two "
                    "derivations meeting. And the two rho are different "
                    "quantities: this one is read from the same ensemble the "
                    "ellipse was fitted to, which the correction-gain law's "
                    "own record says returns 0.00 where the truth is 2.47."}


# --------------------------------------------------------------------------
# The projection's preconditions
# --------------------------------------------------------------------------


@proof("Residuals on different grids are never projected",
       "COMPONENT_MAPPINGS.md section 9")
def grids_are_not_mixed():
    rng = np.random.default_rng(4)
    residuals = {"response": {"frequency": rng.standard_normal(238)},
                 "timing": {"frequency": rng.standard_normal(262)}}
    coherence = ie.coherence(residuals, "frequency")
    out = ie.orthogonalize(residuals, ["response", "timing"])
    untouched = all(np.array_equal(out[n]["frequency"],
                                   residuals[n]["frequency"])
                    for n in residuals)
    return {"lines": [f"  a 238-bin response against a 262-line residual: "
                      f"coherence {coherence:.4f}",
                      f"  both left untouched by the projection: {untouched}"],
            "checks": [_ok(coherence == 0.0,
                           "different abscissae give no coherence at all",
                           coherence, 0.0),
                       _ok(untouched, "and neither is modified", untouched,
                           True)],
            "note": "Before the repair the two were truncated to a common "
                    "length and dotted together, which returns a number that "
                    "is an artefact of memory layout."}


@proof("The projection keeps the phase",
       "COMPONENT_MAPPINGS.md section 9; ELLIPTICAL_COLLAPSE.md section 8")
def phase_survives():
    rng = np.random.default_rng(5)
    shared = rng.standard_normal(64)
    first = shared + 1j * rng.standard_normal(64)
    second = 0.8 * shared + 1j * rng.standard_normal(64)
    out = ie.orthogonalize({"first": {"frequency": first},
                            "second": {"frequency": second}},
                           ["first", "second"])
    kept = out["second"]["frequency"]
    # what stripping the phase costs, on the same ensemble
    complexes = {f"c{i}": {"frequency": shared + 1.5 * (
        rng.standard_normal(64) + 1j * rng.standard_normal(64))}
        for i in range(4)}
    stripped = {n: {"frequency": np.abs(v["frequency"])}
                for n, v in complexes.items()}
    with_phase = ie.ellipsoid(complexes, "frequency")["asymmetry"]
    without = ie.ellipsoid(stripped, "frequency")["asymmetry"]
    return {"lines": [f"  output dtype {kept.dtype}, imaginary energy "
                      f"{np.linalg.norm(kept.imag):.3f}",
                      f"  asymmetry with the phase {with_phase:.4f}, "
                      f"with it stripped {without:.4f}"],
            "checks": [_ok(np.iscomplexobj(kept) and
                           np.linalg.norm(kept.imag) > 0,
                           "the projection returns a complex value",
                           str(kept.dtype), "complex"),
                       _ok(abs(with_phase - without) > 0.2,
                           "stripping the phase answers a different question",
                           f"{abs(with_phase-without):.3f}", "> 0.2")],
            "note": "np.asarray(complex, dtype=float64) keeps the real part "
                    "and drops the imaginary one. It is not a precision "
                    "loss."}


@proof("A gain and a delay of the same shape are two mechanisms",
       "THE_ALGORITHM.md section 3; COMPONENT_MAPPINGS.md section 9")
def real_parameters_separates_gain_from_delay():
    shape = np.linspace(-1.0, 1.0, 128)
    residuals = {"gain": {"frequency": shape.astype(complex)},
                 "delay": {"frequency": 1j * shape}}
    hermitian = ie.ellipsoid(residuals, "frequency")
    stacked = ie.ellipsoid(residuals, "frequency", real_parameters=True)
    return {"lines": [f"  Hermitian inner product : rank {hermitian['rank']}, "
                      f"asymmetry {hermitian['asymmetry']:.4f}",
                      f"  real-stacked            : rank {stacked['rank']}, "
                      f"asymmetry {stacked['asymmetry']:.4f}"],
            "checks": [_ok(hermitian["rank"] == 1,
                           "under the Hermitian product they are one "
                           "direction", hermitian["rank"], 1),
                       _ok(stacked["rank"] == 2,
                           "stacking real and imaginary parts makes them two",
                           stacked["rank"], 2)],
            "note": "Correct when the complex value IS the measurement; wrong "
                    "when each component is a real parameter's signature."}


# --------------------------------------------------------------------------
# The physics the domains carry
# --------------------------------------------------------------------------


@proof("Tape losses collapse: one dimensionless group, five lengths",
       "COMPONENT_MAPPINGS.md section 2; ELLIPTICAL_COLLAPSE.md section 7.4")
def tape_losses_are_one_variable():
    x = np.geomspace(0.01, 2.0, 512)
    forms = {
        "spacing (Wallace)": np.exp(-2 * np.pi * x),
        "gap (sinc)": np.abs(np.sinc(x)),
        "thickness": np.where(x > 1e-9,
                              (1 - np.exp(-2 * np.pi * x))
                              / np.maximum(2 * np.pi * x, 1e-12), 1.0),
        "azimuth (sinc)": np.abs(np.sinc(x)),
        "contour": 1 + 0.02 * np.cos(2 * np.pi * x) * np.exp(-0.15 * x),
    }
    stack = []
    for value in forms.values():
        logged = np.log(np.maximum(value, 1e-12))
        logged = logged - logged.mean()
        stack.append(logged / np.linalg.norm(logged))
    singular = np.linalg.svd(np.array(stack), compute_uv=False)
    share = singular ** 2 / (singular ** 2).sum()
    effective = 1.0 / np.sum(share ** 2)
    return {"lines": [f"  singular values {np.round(singular, 4)}",
                      f"  effective distinct shapes {effective:.2f} of "
                      f"{len(forms)}"],
            "checks": [_ok(effective < 2.5,
                           "five mechanisms span far fewer than five shapes",
                           f"{effective:.2f}", "< 2.5")],
            "note": "Every one is a function of ONE dimensionless group, a "
                    "length over the recorded wavelength. As far as a "
                    "frequency response is concerned they are five values of "
                    "the same variable."}


@proof("Echoes do not collapse: a Fourier basis over delay",
       "COMPONENT_MAPPINGS.md section 2")
def echoes_are_independent():
    frequencies = np.linspace(1e6, 7e6, 2048)
    resolution = 1.0 / (frequencies[-1] - frequencies[0])
    shapes = {}
    for k in range(1, 9):
        value = inf.echo(frequencies, resolution * k, amplitude=0.2)
        vector = (np.log(np.abs(value))
                  + 1j * np.unwrap(np.angle(value)))
        shapes[f"{resolution*k*1e6:.2f}us"] = {"frequency": vector
                                               - vector.mean()}
    fit = ie.ellipsoid(shapes, "frequency", real_parameters=True)
    stack = np.array([np.concatenate([shapes[n]["frequency"].real,
                                      shapes[n]["frequency"].imag])
                      for n in fit["names"]])
    stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
    singular = np.linalg.svd(stack, compute_uv=False)
    share = singular ** 2 / (singular ** 2).sum()
    effective = 1.0 / np.sum(share ** 2)
    condition = singular[0] / singular[-1]
    return {"lines": [f"  eight echoes spaced at 1/B: effective "
                      f"{effective:.2f} of 8, condition {condition:.2f}",
                      f"  asymmetry {fit['asymmetry']:.4f} - very nearly a "
                      f"sphere"],
            "checks": [_ok(effective > 7.5,
                           "an echo family is nearly fully independent",
                           f"{effective:.2f}", "> 7.5"),
                       _ok(condition < 2.0, "and superbly conditioned",
                           f"{condition:.2f}", "< 2.0")],
            "note": "exp(-2 pi j f tau) is a Fourier basis over delay, "
                    "orthogonal by construction. A family is collinear when "
                    "its members differ in a RATE and separable when they "
                    "differ in KIND."}


@proof("Only particle noise is random; four disturbances are not",
       "COMPONENT_MAPPINGS.md section 7; ELLIPTICAL_COLLAPSE.md section 7.7")
def four_of_five_are_not_random():
    frequencies = np.linspace(1e6, 7e6, 512)
    lines, verdicts = [], {}
    for label, make in (
        ("particle noise", lambda r: _noise(r, 512)),
        ("beat / co-channel", lambda r: inf.beat(frequencies, 3.9e6)
         * np.exp(2j * np.pi * r.random())),
        ("head contact tilt", lambda r: inf.head_contact_tilt(
            frequencies, frequencies[0], frequencies[-1])
         * np.exp(2j * np.pi * r.random())),
    ):
        rng = np.random.default_rng(abs(hash(label)) % 2 ** 31)
        ensemble = {f"field {i}": {"frequency": make(rng)} for i in range(16)}
        fit = ie.ellipsoid(ensemble, "frequency")
        verdicts[label] = (fit["significant"], fit["sigma"])
        lines.append(f"  {label:>20}: {fit['significant']} direction(s), "
                     f"asymmetry {fit['asymmetry']:.4f} vs floor "
                     f"{fit['sphere_floor']:.4f}, {fit['sigma']:+.1f} sigma")
    return {"lines": lines,
            "checks": [
                _ok(verdicts["particle noise"][0] == 0
                    and abs(verdicts["particle noise"][1]) <= ie.SPHERE_SIGMA,
                    "particle noise sits at the sphere floor",
                    f"{verdicts['particle noise'][1]:+.1f} sigma",
                    f"within {ie.SPHERE_SIGMA}"),
                _ok(verdicts["beat / co-channel"][1] > 50,
                    "a beat has a frequency even when its phase is random",
                    f"{verdicts['beat / co-channel'][1]:+.0f} sigma", "> 50")],
            "note": "Every disturbance here is given a RANDOM PHASE per "
                    "field, so none can correlate trivially."}


@proof("The specification's recording depth is lambda/4, not a constant",
       "ELLIPTICAL_COLLAPSE.md section 7.6; interference.particle_noise")
def recording_depth_is_a_quarter_wavelength():
    frequencies = np.array([1e6, 3.9e6, 7e6])
    law = 20 * np.log10(np.abs(inf.particle_noise(frequencies, 5.8, 58e-6)))
    fixed = 20 * np.log10(np.abs(inf.particle_noise(frequencies, 5.8, 58e-6,
                                                    0.35e-6)))
    law_slope = (law[-1] - law[0]) / np.log10(7.0)
    fixed_slope = (fixed[-1] - fixed[0]) / np.log10(7.0)
    # the guide's own worked range, at its stated 4.85 m/s
    worked = [4.85 / f / 4 * 1e6 for f in (3.8e6, 4.8e6)]
    return {"lines": [f"  the guide works d = lambda/4 to "
                      f"{worked[1]:.3f}-{worked[0]:.3f} um; it prints "
                      f"0.26-0.36 um",
                      f"  slope with the law   {law_slope:+.1f} dB/decade",
                      f"  slope at fixed depth {fixed_slope:+.1f} dB/decade"],
            "checks": [_ok(abs(law_slope - 20.0) < 0.5,
                           "the read volume goes as lambda squared",
                           f"{law_slope:+.1f} dB/decade", "+20.0"),
                       _ok(0.25 < worked[1] and worked[0] < 0.37,
                           "the transcription reproduces the guide's own "
                           "worked range",
                           f"{worked[1]:.3f}-{worked[0]:.3f} um",
                           "0.26-0.36 um")],
            "note": "JVC VTG82063 section 7.2: 'the optimum playback "
                    "sensitivity is obtained when recording is performed at a "
                    "depth equivalent to 1/4th the recording wavelength'."}


# --------------------------------------------------------------------------
# The failures. These prove the DEFECTS, and fail if a defect stops
# reproducing - because the record would then no longer match the code.
# --------------------------------------------------------------------------


@proof("WITHDRAWN: pure noise reproduces the nested-matrix run",
       "ELLIPTICAL_COLLAPSE.md section 6c")
def noise_reproduces_the_published_run():
    def nested(fields, places, seed):
        rng = np.random.default_rng(seed)
        synthetic = {f"f{i}": np.zeros(places) for i in range(fields)}
        measured = {f"f{i}": rng.standard_normal(places)
                    for i in range(fields)}
        relations = {n: [o for o in synthetic if o != n] for n in synthetic}
        return ie.component_differentials(synthetic, measured, "frequency",
                                          relations)
    lines, results = [], {}
    for label, fields, places, reported in (
            ("frequency / head A", 13, 119, (13, 11, 92.9, 39.3)),
            ("time / head A", 13, 264, (13, 13, 100.0, 118.0))):
        draws = []
        for seed in range(6):
            fit = ie.ellipsoid(nested(fields, places, 900 + seed), "frequency")
            draws.append((fit["rank"], fit["significant"],
                          100 * fit["resolved_fraction"], fit["sigma"]))
        got = np.array(draws).mean(axis=0)
        results[label] = (reported, got)
        lines.append(f"  {label}")
        lines.append(f"      reported : rank {reported[0]}, "
                     f"{reported[1]} directions, {reported[2]:.1f}% resolved, "
                     f"{reported[3]:+.1f} sigma")
        lines.append(f"      PURE NOISE: rank {got[0]:.0f}, "
                     f"{got[1]:.0f} directions, {got[2]:.1f}% resolved, "
                     f"{got[3]:+.1f} sigma")
    close = all(abs(rep[3] - got[3]) < 6.0 and abs(rep[2] - got[2]) < 10.0
                for rep, got in results.values())
    return {"lines": lines,
            "checks": [_ok(close,
                           "the defect still reproduces: noise gives the same "
                           "answer as the reported run",
                           "within 6 sigma and 10 points", "the defect")],
            "note": "component_differentials emits N(N+1)/2 entries spanning "
                    "only N dimensions, so the rank is forced, every "
                    "eigenvalue clears the noise edge, and the shape "
                    "statistics measure the construction. This proof FAILS if "
                    "the defect stops reproducing, which would mean the "
                    "withdrawal in section 6c no longer matches the code."}


@proof("The exact 0.000 asymmetry is a literal constant",
       "ELLIPTICAL_COLLAPSE.md section 6c")
def the_exact_zero_is_a_constant():
    rng = np.random.default_rng(7)
    residuals = {f"c{i}": {"frequency": _noise(rng, 64)} for i in range(6)}
    at_full = ie.ellipsoid(residuals, "frequency", removed=5)["asymmetry"]
    at_none = ie.ellipsoid(residuals, "frequency", removed=0)["asymmetry"]
    return {"lines": [f"  removed=0 : asymmetry {at_none:.6f}",
                      f"  removed=5 : asymmetry {at_full:.6f}  "
                      f"(effective clips to 1)"],
            "checks": [_ok(at_full == 0.0,
                           "once effective reaches one the expression returns "
                           "the else branch",
                           at_full, 0.0)],
            "note": "So 'the asymmetry falls to exactly zero' is that "
                    "constant, not a measurement, whenever significant "
                    "equals rank."}


@proof("The shipped scatter is the complex asymptote, wrong for real data",
       "ELLIPTICAL_COLLAPSE.md section 6c")
def the_scatter_is_wrong_for_real_ensembles():
    rng = np.random.default_rng(8)
    lines, rows = [], []
    for count, length in ((8, 1024), (32, 1024)):
        for kind in ("real", "complex"):
            trials = [ie.ellipsoid(
                {f"c{i}": {"frequency": _noise(rng, length,
                                               kind == "complex")}
                 for i in range(count)}, "frequency")["asymmetry"]
                for _ in range(60)]
            measured = float(np.std(trials, ddof=1)) * length
            rows.append((kind, measured))
            lines.append(f"  N={count:>3} L={length}, {kind:>8}: "
                         f"scatter x L = {measured:.3f}   "
                         f"(the code uses {np.sqrt(2):.3f})")
    real = [m for k, m in rows if k == "real"]
    complexes = [m for k, m in rows if k == "complex"]
    return {"lines": lines,
            "checks": [_ok(min(real) > 1.7,
                           "a real ensemble scatters about 2/L, not sqrt(2)/L",
                           f"{min(real):.2f} to {max(real):.2f}", "> 1.7"),
                       _ok(abs(np.mean(complexes) - np.sqrt(2)) < 0.25,
                           "sqrt(2)/L is the COMPLEX asymptote",
                           f"{np.mean(complexes):.3f}",
                           f"{np.sqrt(2):.3f}")],
            "note": "Every sigma reported on a real ensemble is inflated by "
                    "about 1.4x, and when N approaches L the true scatter "
                    "collapses further still."}


@proof("The surrogate null is the only valid one for a constructed ensemble",
       "ELLIPTICAL_COLLAPSE.md section 6c; information_extrapolation."
       "surrogate_null")
def the_surrogate_null_works():
    rng = np.random.default_rng(9)
    length, fields = 119, 13
    truth = rng.standard_normal(length)

    def nested(scale):
        r = np.random.default_rng(55)
        synthetic = {f"f{i}": np.zeros(length) for i in range(fields)}
        measured = {f"f{i}": r.standard_normal(length) + scale * truth
                    for i in range(fields)}
        relations = {n: [o for o in synthetic if o != n] for n in synthetic}
        return ie.component_differentials(synthetic, measured, "frequency",
                                          relations)
    lines, zs = [], {}
    for label, scale in (("pure noise", 0.0), ("a departure x1.5", 1.5),
                         ("a departure x4", 4.0)):
        null = ie.surrogate_null(nested(scale), "frequency", draws=24, seed=3)
        zs[label] = null["asymmetry_z"]
        lines.append(f"  {label:>18}: observed "
                     f"{null['asymmetry_observed']:.4f}, null "
                     f"{null['asymmetry_mean']:.4f} +- "
                     f"{null['asymmetry_sd']:.4f}, z "
                     f"{null['asymmetry_z']:+.1f}")
    lines.append(f"  the null rebuilds the construction: "
                 f"{null['independent']} independent draws, deficiency "
                 f"{null['deficiency']}")
    # WHY THE SIGN IS NEGATIVE, and it is the construction speaking
    lines.append("  and where the departure's energy goes:")
    for scale in (0.0, 1.5, 4.0):
        matrix = nested(scale)
        diagonal = [k for k in matrix if " vs " not in k]
        off = [k for k in matrix if " vs " in k]
        energy = lambda keys: float(np.mean(
            [np.vdot(matrix[k]["frequency"],
                     matrix[k]["frequency"]).real for k in keys]))
        lines.append(f"      x{scale:<4}: diagonal {energy(diagonal):8.1f}, "
                     f"off-diagonal {energy(off):8.1f}   "
                     f"({len(diagonal)} entries vs {len(off)})")
    return {"lines": lines,
            "checks": [_ok(abs(zs["pure noise"]) < 3.0,
                           "noise does not stand apart from a null built the "
                           "same way", f"{zs['pure noise']:+.1f}",
                           "within 3"),
                       _ok(all(abs(z) < 3.0 for z in zs.values()),
                           "AND NEITHER DOES A SHARED DEPARTURE, however "
                           "strong - the asymmetry on a nested matrix is "
                           "BLIND to it",
                           ", ".join(f"{k.split()[-1]} {v:+.1f}"
                                     for k, v in zs.items()),
                           "all within 3, i.e. blind")],
            "note": "THE DEFECT, IN ITS SHARPEST FORM. A shared departure "
                    "cancels exactly in every difference - off-diagonal "
                    "energy is unchanged at 249.6 whatever the departure "
                    "does - so it enters only through the 13 diagonals "
                    "against 78 off-diagonals. And because every row is "
                    "unit-normalised, growing the diagonals seventeenfold "
                    "does not move their DIRECTIONS, so the shape is "
                    "unchanged: at x4 the asymmetry reads 0.8862 against "
                    "pure noise's 0.8862. The statistic could not see the "
                    "thing it was supposed to measure, which is the whole "
                    "of why the run in section 6 reported the construction. "
                    "The remedy is not a better null on this matrix - it is "
                    "to fit the ellipse over TRANSFERS rather than "
                    "differences, which is what section 0a now does."}


# --------------------------------------------------------------------------
# What the key covers, and what it never will
# --------------------------------------------------------------------------


@proof("Representable is not knowable: a full basis spans all noise",
       "THE_ALGORITHM.md, 'Representable is not knowable'")
def a_complete_basis_spans_noise():
    length = 256
    lines, rows = [], []
    for count in (8, 64, 200, 256):
        rng = np.random.default_rng(count)
        basis = np.linalg.qr(rng.standard_normal((length, count)))[0]
        target = rng.standard_normal(length)
        inside = float(np.sum((basis.T @ target) ** 2)
                       / np.sum(target ** 2))
        rows.append((count, inside, count / length))
        lines.append(f"  a basis of {count:>3} sinusoids spans "
                     f"{100*inside:>5.1f}% of pure noise   "
                     f"(equal share {100*count/length:>5.1f}%)")
    worst = max(abs(inside - share) for _, inside, share in rows)
    return {"lines": lines,
            "checks": [_ok(worst < 0.05,
                           "a basis captures exactly its equal share of noise "
                           "and no more", f"worst departure {worst:.3f}",
                           "< 0.05"),
                       _ok(rows[-1][1] > 0.999,
                           "a complete basis represents noise perfectly",
                           f"{100*rows[-1][1]:.1f}%", "100%")],
            "note": "Spanning is not predicting. This is exactly how a "
                    "construction whose entries spanned the whole space came "
                    "to report noise as structure."}


@proof("The key must span, not merely be large",
       "COMPONENT_MAPPINGS.md section 8a")
def the_key_must_span():
    frequencies = np.linspace(1e6, 7e6, 1024)
    signature = inf.signatures(frequencies)

    def effective(names):
        shapes = {}
        for name in names:
            value = np.asarray(signature[name])
            magnitude = np.log(np.maximum(np.abs(value), 1e-12))
            phase = (np.unwrap(np.angle(value))
                     if np.iscomplexobj(value) else np.zeros_like(magnitude))
            vector = ((magnitude - magnitude.mean())
                      + 1j * (phase - phase.mean()))
            if np.linalg.norm(vector) > 0:
                shapes[name] = {"frequency": vector}
        fit = ie.ellipsoid(shapes, "frequency", real_parameters=True)
        stack = np.array([np.concatenate([shapes[n]["frequency"].real,
                                          shapes[n]["frequency"].imag])
                          for n in fit["names"]])
        stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
        singular = np.linalg.svd(stack, compute_uv=False)
        share = singular ** 2 / (singular ** 2).sum()
        return (1.0 / np.sum(share ** 2), singular[0] / singular[-1],
                len(fit["names"]))
    without = effective([n for n in signature if "emphasis" not in n])
    with_all = effective(list(signature))
    # what happens if a near-duplicate is added instead
    duplicated = dict(signature)
    duplicated["a second copy of the tilt"] = inf.head_contact_tilt(
        frequencies, frequencies[0], frequencies[-1]) * 1.01
    signature.update(duplicated)
    with_copy = effective(list(duplicated))
    return {"lines": [f"  without the specified emphasis : "
                      f"{without[0]:.2f} of {without[2]}, condition "
                      f"{without[1]:.1f}",
                      f"  with it, as its level dependence: "
                      f"{with_all[0]:.2f} of {with_all[2]}, condition "
                      f"{with_all[1]:.1f}",
                      f"  with a near-duplicate added    : "
                      f"{with_copy[0]:.2f} of {with_copy[2]}, condition "
                      f"{with_copy[1]:.1f}"],
            "checks": [_ok(with_copy[0] < with_all[0],
                           "a second copy of a direction already in the span "
                           "LOWERS the effective count",
                           f"{with_copy[0]:.2f} against {with_all[0]:.2f}",
                           "lower")],
            "note": "A modification earns a place in the key by the DIRECTION "
                    "it adds, not by being a modification. When a standard "
                    "hands over a family of curves, enter what VARIES between "
                    "them."}


@proof("The three-way split: inside, structured-but-outside, and noise",
       "interference.split_residual")
def the_three_way_split():
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    rng = np.random.default_rng(10)
    tilt = inf.head_contact_tilt(frequencies, frequencies[0], frequencies[-1])
    unmodelled = np.sin(2 * np.pi * 17 * (frequencies - frequencies[0])
                        / (frequencies[-1] - frequencies[0])).astype(complex)
    lines, results = [], {}
    for label, make in (
            ("pure noise", lambda: _noise(rng, 380)),
            ("a modelled tilt", lambda: 3 * tilt + _noise(rng, 380)),
            ("an UNMODELLED shape", lambda: 3 * unmodelled
             + _noise(rng, 380))):
        residual = {f"c{i}": make() for i in range(13)}
        split = inf.split_residual(residual, frequencies, draws=16, seed=5)
        results[label] = split
        lines.append(f"  {label:>22}: inside {100*split['inside']:>5.1f}%  "
                     f"structured {100*split['structured']:>5.1f}%  "
                     f"noise {100*split['noise']:>5.1f}%")
    return {"lines": lines,
            "checks": [
                _ok(results["pure noise"]["noise"] > 0.9,
                    "pure noise lands almost entirely in the third part",
                    f"{100*results['pure noise']['noise']:.1f}%", "> 90%"),
                _ok(results["a modelled tilt"]["inside"] > 0.4,
                    "a modelled mechanism lands inside the key",
                    f"{100*results['a modelled tilt']['inside']:.1f}%",
                    "> 40%"),
                _ok(results["an UNMODELLED shape"]["structured"] > 0.4,
                    "an unmodelled mechanism is found OUTSIDE the key and "
                    "still structured",
                    f"{100*results['an UNMODELLED shape']['structured']:.1f}%",
                    "> 40%")],
            "note": "The middle part is the only thing that is both unreached "
                    "and reachable, and it is separable from noise only "
                    "because the surrogate null exists."}


# --------------------------------------------------------------------------
# The capture, and the floors
# --------------------------------------------------------------------------


@proof("The capture profile is read, not assumed",
       "capture_profile.measure_bit_depth")
def the_capture_profile_is_measured():
    rng = np.random.default_rng(11)
    codes = (rng.integers(-58, 59, size=8192) * 256).astype(np.int16)
    profile = cp.profile_of(codes, 50e6)
    return {"lines": [f"  word length {profile['bits']:.0f} bit, step "
                      f"{profile['step']:.0f} in the container",
                      f"  {profile['codes']:.0f} codes present, occupying "
                      f"{100*profile['occupancy']:.1f}% of full scale",
                      f"  quantisation SNR "
                      f"{profile['signal_to_noise_db']:.1f} dB"],
            "checks": [_ok(profile["bits"] == 8.0,
                           "an 8-bit capture in a 16-bit container reads as "
                           "8-bit", profile["bits"], 8.0),
                       _ok(profile["occupancy"] < 0.6,
                           "and occupancy is not mistaken for full scale",
                           f"{profile['occupancy']:.3f}", "< 0.6")],
            "note": "The full scale is the container's range. Deriving it "
                    "from the data's span reports a quiet capture as having "
                    "fewer bits than it has."}


@proof("The radio binds, not the converter - and at what C/N that reverses",
       "ELLIPTICAL_COLLAPSE.md section 7.1; COMPONENT_MAPPINGS.md section 8")
def the_radio_binds():
    profile = {"bits": 8.0, "sample_rate_hz": 50e6, "full_scale": 256.0}
    measured = cp.binding_limit(0.697, 24.9, profile)
    pristine = cp.binding_limit(0.005, 24.9, profile)
    # where the crossover sits
    crossover = None
    for spread in np.geomspace(0.001, 1.0, 400):
        if cp.binding_limit(spread, 24.9, profile)["binds"] == "tape":
            crossover = cp.binding_limit(spread, 24.9, profile)["tape_snr_db"]
            break
    return {"lines": [f"  measured spread 0.697 dB: tape "
                      f"{measured['tape_snr_db']:.1f} dB -> "
                      f"{measured['tape_capacity_bits_per_s']/1e6:.1f} "
                      f"Mbit/s; capture "
                      f"{measured['capture_snr_db']:.1f} dB -> "
                      f"{measured['capture_capacity_bits_per_s']/1e6:.1f}",
                      f"  the {measured['binds'].upper()} binds, by "
                      f"{measured['ratio']:.2f}x",
                      f"  a pristine tape (0.005 dB) instead binds on the "
                      f"{pristine['binds'].upper()}",
                      f"  the crossover sits near "
                      f"{crossover:.1f} dB tape C/N"],
            "checks": [_ok(measured["binds"] == "tape",
                           "on this capture the tape is the limit",
                           measured["binds"], "tape"),
                       _ok(pristine["binds"] == "capture",
                           "the verdict is a measurement, not a foregone "
                           "conclusion", pristine["binds"], "capture")],
            "note": "The envelope-to-C/N conversion is 3.01 dB generous on "
                    "the tape side only, and Carson's deviation should be the "
                    "0.5 MHz peak rather than the 1.0 MHz swing. Both move "
                    "the numbers; neither moves the ordering."}


@proof("A dimension is the Wiener transfer between two channels",
       "THE_ALGORITHM.md section 0a; pair_dimension")
def the_pair_transfer_recovers_a_known_channel():
    rng = np.random.default_rng(12)
    x = rng.standard_normal(8192)
    y = np.zeros_like(x)
    previous = 0.0
    for index, value in enumerate(x):
        previous = 0.3 * value + 0.7 * previous
        y[index] = previous
    y = y + 0.05 * rng.standard_normal(8192)
    result = pdim.pair_transfer(x, y, segments=16)
    frequencies = np.fft.rfftfreq(8192 // 16)
    truth = 0.3 / (1.0 - 0.7 * np.exp(-2j * np.pi * frequencies))
    band = result["coherence"] > 0.5
    gain = float(np.mean(np.abs(np.abs(result["transfer"][band])
                                / np.abs(truth[band]) - 1.0)))
    phase = float(np.degrees(np.mean(np.abs(
        np.angle(result["transfer"][band] / truth[band])))))
    z = rng.standard_normal(8192)
    lines = [f"  a known one-pole channel: gain error {100*gain:.2f}%, "
             f"phase error {phase:.2f} degrees"]
    biases = []
    for segments in (4, 16, 64):
        independent = pdim.pair_transfer(x, z, segments=segments)
        biases.append((segments, float(independent["raw_coherence"].mean()),
                       float(independent["coherence"].mean())))
        lines.append(f"  {segments:>3} segments on INDEPENDENT channels: raw "
                     f"coherence {biases[-1][1]:.4f} (= 1/n = "
                     f"{1/segments:.4f}), debiased {biases[-1][2]:.4f}")
    return {"lines": lines,
            "checks": [_ok(gain < 0.10 and phase < 5.0,
                           "the transfer is recovered in gain and phase",
                           f"{100*gain:.1f}% and {phase:.1f} deg",
                           "< 10% and < 5 deg"),
                       _ok(all(abs(raw - 1.0 / n) / (1.0 / n) < 0.3
                               for n, raw, _ in biases),
                           "an uncorrelated pair returns exactly 1/n before "
                           "debiasing", "within 30% of 1/n", "1/n")],
            "note": "|T|, arg T and the frequency it lives at are the "
                    "amplitude, phase and frequency of the differential pair."}


@proof("The information a VHS field holds",
       "ELLIPTICAL_COLLAPSE.md section 7.6")
def how_much_information_vhs_holds():
    field_hz = 59.94
    band = cp.carson_bandwidth(0.5e6, 3.0e6)     # deviation as PEAK
    points = 50e6 / field_hz / 2
    lines = [f"  the plane's extent: {points:,.0f} independent complex "
             f"points per field"]
    rows = []
    for label, cn in (("raw envelope spread, corrected", 18.55),
                      ("after separating structure", 33.1),
                      ("the particle-noise floor", 41.1)):
        per_field = cp.channel_capacity(band, 10 ** (cn / 10)) / field_hz
        rows.append(per_field)
        lines.append(f"  {label:>32}: {cn:>5.1f} dB -> "
                     f"{per_field/1e3:>7.1f} kbit/field, "
                     f"{per_field/points:.2f} bits per point")
    return {"lines": lines,
            "checks": [_ok(band == 7.0e6,
                           "Carson's band uses the PEAK deviation",
                           f"{band/1e6:.1f} MHz", "7.0 MHz"),
                       _ok(rows[0] < rows[1] < rows[2],
                           "separating structure raises what the tape is "
                           "worth", "monotone", "monotone")],
            "note": "A VHS field holds between 722 and 1284 kbit depending on "
                    "how much of the envelope variation is structure - the "
                    "one quantity the transform exists to measure."}


@proof("Every key entry is subtractable, and the key has a known order",
       "MATHEMATICS.md section 3; interference.subtractable, ordered_key")
def the_key_is_ordered_and_subtractable():
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    entries = inf.signatures(frequencies)
    unsubtractable = [n for n, v in entries.items()
                      if not inf.subtractable(v)]
    order = inf.ordered_key(entries)
    refused = False
    try:
        inf.ordered_key({**entries, "undeclared": entries["dropout"]})
    except ValueError:
        refused = True
    return {"lines": [f"  {len(entries)} entries, "
                      f"{len(entries)-len(unsubtractable)} subtractable",
                      f"  order runs from position {order[0][0]} down to "
                      f"{order[-1][0]} - last applied, first removed",
                      f"  an undeclared entry is refused: {refused}"],
            "checks": [_ok(not unsubtractable,
                           "every key entry has a finite logarithm, so it "
                           "can be subtracted",
                           f"{len(unsubtractable)} failing", "0"),
                       _ok(refused,
                           "a key entry with no declared position is refused, "
                           "because a chain can only be inverted in the "
                           "reverse of its order", refused, True)],
            "note": "Two ways an entry used to fail: a MASK, zero outside "
                    "its support, and a POWER contribution, zero where the "
                    "interferer is not. Both now enter in bounded form - "
                    "1 - d*shape and 1 + a*shape - and enforcing that raised "
                    "the effective count from 6.92 to 7.66 and improved the "
                    "condition from 11.4 to 7.3."}


@proof("What no model reaches can be reconstructed if it is band-limited",
       "interference.interpolate_unmodelled")
def the_interpolation_substitute():
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    rng = np.random.default_rng(31)

    def noise():
        return _noise(rng, 380)
    smooth = np.sin(2 * np.pi * 3 * (frequencies - frequencies[0])
                    / (frequencies[-1] - frequencies[0])).astype(complex)
    rough = np.sin(2 * np.pi * 90 * (frequencies - frequencies[0])
                   / (frequencies[-1] - frequencies[0])).astype(complex)
    lines, gains = [], {}
    for label, build in (("a SMOOTH unmodelled shape",
                          lambda: 3 * smooth + noise()),
                         ("a ROUGH unmodelled shape",
                          lambda: 3 * rough + noise()),
                         ("pure noise", noise)):
        result = inf.interpolate_unmodelled(
            {f"c{i}": build() for i in range(13)}, frequencies)
        gains[label] = result["gain_over_equal_share"]
        lines.append(f"  {label:>26}: concentration "
                     f"{100*result['concentration']:>5.1f}% against an equal "
                     f"share of {100*result['equal_share']:.1f}%, gain "
                     f"{result['gain_over_equal_share']:.2f}")
    return {"lines": lines,
            "checks": [_ok(gains["a SMOOTH unmodelled shape"] > 3.0,
                           "a smooth departure the key cannot model is "
                           "recoverable by band-limited interpolation",
                           f"{gains['a SMOOTH unmodelled shape']:.2f}x",
                           "> 3x"),
                       _ok(gains["pure noise"] < 1.5 and
                           gains["a ROUGH unmodelled shape"] < 1.5,
                           "noise and a rough shape are not, and report "
                           "their own equal share",
                           f"{gains['pure noise']:.2f}x and "
                           f"{gains['a ROUGH unmodelled shape']:.2f}x",
                           "< 1.5x")],
            "note": "The band must be narrow or the test cannot "
                    "discriminate: at a quarter of the transform the equal "
                    "share is already fifty per cent and a rough shape "
                    "scores higher than a smooth one."}


@proof("A remainder is the floor only if it has the floor's SHAPE",
       "interference.noise_shape_differential")
def the_noise_has_a_shape():
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    rng = np.random.default_rng(32)
    particle = np.abs(inf.particle_noise(frequencies, 5.8, 58e-6))
    lines, verdicts = [], {}
    for label, build in (
            ("particle noise, correct shape",
             lambda: particle * _noise(rng, 380)),
            ("white noise, flat", lambda: _noise(rng, 380)),
            ("a falling shape",
             lambda: _noise(rng, 380)
             / np.sqrt(frequencies / frequencies[0]))):
        result = inf.noise_shape_differential(
            {f"c{i}": build() for i in range(13)}, frequencies)
        verdicts[label] = result
        lines.append(f"  {label:>30}: slope "
                     f"{result['measured_slope']:>6.1f} dB/decade against an "
                     f"expected {result['expected_slope']:.1f}, departure "
                     f"{result['departure']:+.1f} -> floor: "
                     f"{result['has_the_floor_shape']}")
    return {"lines": lines,
            "checks": [
                _ok(verdicts["particle noise, correct shape"]
                    ["has_the_floor_shape"],
                    "particle noise is recognised as the floor",
                    "yes", "yes"),
                _ok(not verdicts["white noise, flat"]["has_the_floor_shape"],
                    "a flat remainder is NOT the floor, however small",
                    "no", "no")],
            "note": "The read volume goes as the wavelength squared because "
                    "the standard fixes the recording depth at a quarter "
                    "wavelength, so particle noise rises at twenty decibels "
                    "per decade and nothing else in the modelled set does. "
                    "That gives the noise a synthetic side of its own."}


@proof("Have we used all the information? The composite verdict",
       "interference.information_used")
def the_exhaustion_verdict():
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    rng = np.random.default_rng(33)

    def noise():
        return _noise(rng, 380)
    tilt = inf.head_contact_tilt(frequencies, frequencies[0],
                                 frequencies[-1])
    missing = np.sin(2 * np.pi * 17 * (frequencies - frequencies[0])
                     / (frequencies[-1] - frequencies[0])).astype(complex)
    lines, verdicts = [], {}
    for label, build in (("only what the key models",
                          lambda: 3 * tilt + noise()),
                         ("plus something it does NOT model",
                          lambda: 3 * tilt + 3 * missing + noise())):
        result = inf.information_used(
            {f"c{i}": build() for i in range(13)}, frequencies,
            draws=16, seed=4)
        verdicts[label] = result
        lines.append(f"  {label:>34}: exhausted {result['exhausted']}, "
                     f"inside {100*result['inside']:.1f}%, structured "
                     f"{100*result['structured']:.1f}%")
        lines.append(f"  {'':>34}  -> {result['remaining']}")
    return {"lines": lines,
            "checks": [
                _ok(verdicts["only what the key models"]["exhausted"],
                    "with only modelled mechanisms present the key has "
                    "reached everything it can", True, True),
                _ok(not verdicts["plus something it does NOT model"]
                    ["exhausted"],
                    "and it says so when something is outside it",
                    False, False)],
            "note": "Four conditions, all required: the key is ordered, "
                    "every entry is subtractable, nothing structured remains "
                    "outside it, and what does remain sits at the null. The "
                    "first two are preconditions - an unordered or "
                    "unsubtractable key can appear to leave nothing behind "
                    "simply by being unable to remove anything properly."}


@proof("The final dimension: the sync as specified against the sync as it came back",
       "pair_dimension.spec_sync, sync_dimension")
def the_sync_dimension():
    rate, length = 50e6, 4096
    ideal = pdim.spec_sync(rate, length)
    below = np.flatnonzero(ideal <= ideal.min() / 2)
    width = (below[-1] - below[0] + 1) / rate
    leading = ideal[:length // 2]
    ten = np.flatnonzero(leading <= 0.1 * ideal.min())[0]
    ninety = np.flatnonzero(leading <= 0.9 * ideal.min())[0]
    rise = (ninety - ten) / rate
    # a known channel applied to the pulse, then recovered
    measured = np.zeros_like(ideal)
    previous = 0.0
    for index, value in enumerate(ideal):
        previous = 0.25 * value + 0.75 * previous
        measured[index] = previous
    rng = np.random.default_rng(41)
    measured = measured + 0.02 * np.abs(ideal).max() * rng.standard_normal(
        length)
    result = pdim.sync_dimension(measured, rate, segments=16)
    frequencies = np.fft.rfftfreq(length // 16, 1.0 / rate)
    truth = 0.25 / (1.0 - 0.75 * np.exp(-2j * np.pi * frequencies / rate))
    band = result["coherence"] > 0.5
    gain = float(np.mean(np.abs(np.abs(result["transfer"][band])
                                / np.abs(truth[band]) - 1.0)))
    phase = float(np.degrees(np.mean(np.abs(
        np.angle(result["transfer"][band] / truth[band])))))
    return {"lines": [f"  the specified pulse: width {1e6*width:.3f} us "
                      f"(4.700), 10-90 fall {1e9*rise:.0f} ns (140), depth "
                      f"{ideal.min():.1f} IRE (-40)",
                      f"  a known channel recovered from it: gain error "
                      f"{100*gain:.1f}%, phase error {phase:.1f} degrees, "
                      f"coherent over {int(band.sum())} bins"],
            "checks": [_ok(abs(width - pdim.SYNC_WIDTH_S) < 2.0 / rate,
                           "the pulse is the specified width",
                           f"{1e6*width:.3f} us", "4.700 us"),
                       _ok(abs(rise - pdim.SYNC_RISE_S) < 2.0 / rate,
                           "and the specified 10-90 rise",
                           f"{1e9*rise:.0f} ns", "140 ns"),
                       _ok(gain < 0.15 and phase < 8.0,
                           "the channel applied to it is recovered",
                           f"{100*gain:.1f}% and {phase:.1f} deg",
                           "< 15% and < 8 deg")],
            "note": "The best-conditioned dimension available, for three "
                    "reasons no other pair has together: its synthetic side "
                    "is SPECIFIED rather than fitted, it is CONTENT-FREE so "
                    "the sync-only constraint is satisfied, and the pulse "
                    "exists on BOTH sides of the demodulator so a residual "
                    "on one is a prediction about the other. The standard "
                    "states a 10-90 time, and a raised cosine's 10-90 is "
                    "0.5904 of its transition - widening by that reciprocal "
                    "is what takes the pulse from 100 ns to 140."}


@proof("A constant is unknowable, and the standard is what supplies it",
       "standard_levels; MATHEMATICS.md section 9")
def the_standard_supplies_the_constant():
    rng = np.random.default_rng(55)
    length, count = 380, 13
    shape = rng.standard_normal(length)

    def ensemble(gain, offset):
        return {f"c{i}": {"frequency":
                          gain * (shape + 0.4 * rng.standard_normal(length))
                          + offset} for i in range(count)}
    plain = ie.ellipsoid(ensemble(1.0, 0.0), "frequency")
    scaled = ie.ellipsoid(ensemble(1.7, 0.0), "frequency")
    shifted = ie.ellipsoid(ensemble(1.0, 12.0), "frequency")
    invisible = (scaled["significant"] == plain["significant"]
                 and shifted["significant"] == plain["significant"])

    rate = 50e6
    line = pdim.spec_sync(rate, 4096)
    lines = [f"  a constant gain or offset moves no direction: "
             f"{plain['significant']}, {scaled['significant']}, "
             f"{shifted['significant']} directions"]
    landed = []
    for gain, offset in ((1.0, 0.0), (1.7, 12.0), (0.6, -5.0)):
        seen = gain * line + offset
        edges = sl.half_amplitude_crossings(seen, rate)
        result = sl.compliance(seen, edges["bottom"], edges["top"],
                               measured_white=edges["top"] + 100.0 * gain)
        landed.append(result["white_lands_at_ire"])
        lines.append(f"  source gain {gain:.1f}, offset {offset:+.0f}: "
                     f"anchors {edges['bottom']:>7.2f} / {edges['top']:>6.2f} "
                     f"-> white lands at {result['white_lands_at_ire']:.1f} IRE")
    return {"lines": lines,
            "checks": [_ok(invisible,
                           "source equipment levels are in the null space of "
                           "any differential", "no direction moved",
                           "no direction moved"),
                       _ok(all(abs(v - 100.0) < 0.5 for v in landed),
                           "and the standard's own levels supply what the "
                           "measurement cannot see",
                           f"{np.round(landed, 1)}", "100.0 IRE")],
            "note": "A gain rescales every component alike and an offset "
                    "shifts them alike, so neither moves a DIRECTION. That "
                    "layer is the source equipment's, it is constant over "
                    "the signal, and no differential can reach it - which is "
                    "exactly what a specification is for."}


@proof("The closing step: the residual over the tape's ideal response",
       "interference.ideal_tape_response, interpolate_over_ideal")
def the_closing_step():
    frequencies = np.linspace(2.543e6, 5.257e6, 380)
    ideal = inf.ideal_tape_response(frequencies)
    rng = np.random.default_rng(51)

    def noise():
        return _noise(rng, 380)
    smooth = np.sin(2 * np.pi * 3 * (frequencies - frequencies[0])
                    / (frequencies[-1] - frequencies[0])).astype(complex)
    tilt = inf.head_contact_tilt(frequencies, frequencies[0],
                                 frequencies[-1])
    lines = [f"  self-demagnetisation caps the depth at lambda/(2 pi), not "
             f"the lambda/4 optimum: {ideal['cost_of_the_limit_db']:.2f} dB, "
             f"everywhere"]
    results = {}
    for label, build in (("only what the key models",
                          lambda: 3 * tilt + noise()),
                         ("plus a smooth unmodelled shape",
                          lambda: 3 * tilt + 3 * smooth + noise()),
                         ("pure noise", noise)):
        result = inf.interpolate_over_ideal(
            {f"c{i}": build() for i in range(13)}, frequencies)
        results[label] = result
        lines.append(f"  {label:>32}: in band "
                     f"{100*result['inside_the_tape_band']:>5.1f}%, "
                     f"recoverable {100*result['reconstructable']:>5.1f}%, "
                     f"ABSOLUTE FLOOR "
                     f"{100*result['absolute_floor']:>5.1f}%")
    slope = float((ideal["snr_db"][-1] - ideal["snr_db"][0])
                  / np.log10(frequencies[-1] / frequencies[0]))
    return {"lines": lines,
            "checks": [
                _ok(abs(ideal["cost_of_the_limit_db"] - 1.96) < 0.01,
                    "the optimum depth is never reachable, and the cost is "
                    "exactly 10 log10((1/4)/(1/2pi))",
                    f"{ideal['cost_of_the_limit_db']:.2f} dB", "1.96 dB"),
                _ok(abs(slope + 20.0) < 0.5,
                    "and the noise slope is unchanged, because both laws "
                    "scale with the wavelength",
                    f"{slope:.1f} dB/decade", "-20.0"),
                _ok(results["plus a smooth unmodelled shape"]["absolute_floor"]
                    < results["only what the key models"]["absolute_floor"],
                    "recovering an unmodelled but band-limited shape lowers "
                    "the floor",
                    f"{100*results['plus a smooth unmodelled shape']['absolute_floor']:.1f}%",
                    "lower")],
            "note": "The capture holds five to nine times what the tape has "
                    "to say, so the capture's band is WIDER than the tape's "
                    "and anything the tape's response could never have "
                    "carried is not tape information. Weighting by the ideal "
                    "response is therefore not a smoothing choice - it is "
                    "discarding what the medium could not have recorded. "
                    "What is left is inside the band, unmodelled, and does "
                    "not reconstruct, so it has nowhere else to be."}


@proof("Record level is not a new shape; its LEVEL DEPENDENCE is",
       "MATHEMATICS.md section 8b; models/magnetic.py")
def record_level_is_an_axis_not_a_shape():
    speed = head_model.mechanics_for("VHS", "NTSC", "SP")["writing_speed_m_s"]
    frequencies = np.linspace(0.5e6, 8e6, 512)
    control = mag.linear_control(frequencies, speed)
    axis = mag.level_axis(frequencies, speed)
    saturating = mag.level_axis(frequencies, speed,
                                saturation=lambda L: 3.3 * (1.0 - np.exp(-L)))

    # without the cap the axis must collapse to one direction
    def uncapped():
        shapes = []
        wavelength = speed / np.maximum(frequencies, 1.0)
        for level in (0.5, 1.0, 1.5, 2.0, 3.0):
            values = []
            for offset in (+0.05, -0.05):
                depth = mag.NOMINAL_DEPTH_M * (level + offset)
                x = 2.0 * np.pi * depth / wavelength
                values.append((1.0 - np.exp(-x)) / np.maximum(x, 1e-12))
            logged = np.log(np.maximum(values[0], 1e-12)
                            / np.maximum(values[1], 1e-12))
            logged = logged - logged.mean()
            norm = float(np.linalg.norm(logged))
            shapes.append(logged / norm if norm > 1e-9
                          else np.zeros_like(logged))
        singular = np.linalg.svd(np.array(shapes), compute_uv=False)
        share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
        return float(1.0 / np.sum(share ** 2))
    without = uncapped()
    return {"lines": [f"  the control (linear depth, no cap): "
                      f"{control['effective']:.2f} of 5, expected "
                      f"{control['expected']:.2f}",
                      f"  the level axis WITH the cap:        "
                      f"{axis['effective']:.2f} of {axis['count']}, condition "
                      f"{axis['condition']:.1f}",
                      f"  the same WITHOUT the cap:           "
                      f"{without:.2f} of 5",
                      f"  with magnetic saturation:           "
                      f"{saturating['effective']:.2f} of 5",
                      f"  crossovers "
                      f"{[f'{x/1e6:.2f}' for x in axis['crossovers_hz']]} MHz "
                      f"- the cap takes over lower as the level rises"],
            "checks": [
                _ok(control["passes"],
                    "the control reads one direction, as a linear depth law "
                    "with no cap must", f"{control['effective']:.2f}", "1.00"),
                _ok(axis["effective"] > 2.0,
                    "the level axis carries more than one direction",
                    f"{axis['effective']:.2f}", "> 2.0"),
                _ok(without < 1.2,
                    "and REMOVING THE CAP collapses it, so the cap is the "
                    "entire source of the axis", f"{without:.2f}", "< 1.2"),
                _ok(saturating["effective"] < axis["effective"],
                    "magnetic saturation REDUCES the axis, by compressing "
                    "the depth range explored",
                    f"{saturating['effective']:.2f} against "
                    f"{axis['effective']:.2f}", "lower")],
            "note": "Record level's frequency shape is 0.9995 coherent with "
                    "thickness loss and 1.0000 with spacing loss, so as a "
                    "shape it is another member of the family that collapsed "
                    "to 1.58 of 6. Its movement WITH level is a direction "
                    "nothing else supplies, and that direction is the "
                    "crossover between level-limited and cap-limited "
                    "recording moving along the band. The control exists "
                    "because a first estimate read 4.65 of 5 with the depth "
                    "written as a fraction of the cap, which makes the "
                    "response constant and measures numerical noise."}


@proof("A new entry must be orthogonalised against the family it joins",
       "interference.signatures; magnetic.orthogonal_level_signature")
def the_entry_must_be_orthogonalised():
    speed = head_model.mechanics_for("VHS", "NTSC", "SP")["writing_speed_m_s"]
    lines, rows = [f"  the crossover at level 1.0 sits at "
                   f"{mag.crossover_hz(1.0, speed)/1e6:.2f} MHz"], {}
    for low, high, label in ((2.543e6, 5.257e6, "carrier only"),
                             (1e6, 7e6, "VHS luma band"),
                             (0.5e6, 12e6, "full RF")):
        grid = np.linspace(low, high, 1024)
        entries = inf.signatures(grid)
        shipped = inf.distinguishable(grid)

        # the same entry RAW, to show what orthogonalising buys
        raw = dict(entries)
        raw["record level dependence"] = mag.level_signature(grid, 1.0, speed)
        shapes = {}
        for name, value in raw.items():
            value = np.asarray(value)
            magnitude = np.log(np.maximum(np.abs(value), 1e-12))
            phase = (np.unwrap(np.angle(value)) if np.iscomplexobj(value)
                     else np.zeros_like(magnitude))
            vector = ((magnitude - magnitude.mean())
                      + 1j * (phase - phase.mean()))
            if np.linalg.norm(vector) > 0:
                shapes[name] = {"frequency": vector}
        fit = ie.ellipsoid(shapes, "frequency", real_parameters=True)
        stack = np.array([np.concatenate([shapes[n]["frequency"].real,
                                          shapes[n]["frequency"].imag])
                          for n in fit["names"]])
        stack = stack / np.linalg.norm(stack, axis=1, keepdims=True)
        singular = np.linalg.svd(stack, compute_uv=False)
        share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
        raw_effective = float(1.0 / np.sum(share ** 2))
        raw_condition = float(singular[0] / max(singular[-1], 1e-30))

        rows[label] = (shipped["effective"], shipped["condition"],
                       raw_effective, raw_condition)
        lines.append(f"  {label:>16}: orthogonalised "
                     f"{shipped['effective']:.2f} of {shipped['count']} at "
                     f"condition {shipped['condition']:>6.1f}   |   raw "
                     f"{raw_effective:.2f} at {raw_condition:>7.1f}")
    unsubtractable = [n for n, v in inf.signatures(
        np.linspace(1e6, 7e6, 1024)).items() if not inf.subtractable(v)]
    return {"lines": lines,
            "checks": [
                _ok(all(row[1] < row[3] for row in rows.values()),
                    "orthogonalising improves the conditioning on every band",
                    ", ".join(f"{k} {v[1]:.1f} against {v[3]:.1f}"
                              for k, v in rows.items()),
                    "better everywhere"),
                _ok(rows["VHS luma band"][0] > rows["VHS luma band"][2],
                    "and the effective count with it",
                    f"{rows['VHS luma band'][0]:.2f} against "
                    f"{rows['VHS luma band'][2]:.2f}", "higher"),
                _ok(not unsubtractable,
                    "the orthogonalised entry is still subtractable",
                    f"{len(unsubtractable)} failing", "0")],
            "note": "Three independent derivations reached this. Entered RAW "
                    "most of the entry's shape duplicates the losses it sits "
                    "beside, so it costs conditioning without adding a "
                    "direction - on the carrier axis 6.37 effective at "
                    "condition 159 against 7.00 at 11.7 without it. Projected "
                    "orthogonal to them first it can be included on every "
                    "band and improves them. This supersedes an earlier fix "
                    "that simply excluded the entry where its crossover fell "
                    "outside the band: that worked, but only by declining to "
                    "use it."}


@proof("Two clip sites, and why only different LEVELS make them separable",
       "clipping.sites / site_separation / composition_control; "
       "PROPAGATION_COLLAPSE.md section 1")
def the_two_clip_sites():
    """Ethan: *"It might be at the television tuner stage, the cd sample was
    an over the air recording."*"""
    rng = np.random.default_rng(20260904)
    signal = rng.normal(0.0, 1.0, 8192)

    # 1. two limiters in series ARE one limiter, exactly
    worst = 0.0
    for first, second in ((0.8, 0.5), (0.5, 0.8), (1.3, 1.3), (0.2, 2.0)):
        control = clp.composition_control(signal, first, second)
        worst = max(worst, control["worst_difference"])

    # 2. so two clips differing only in threshold are ONE direction: their
    #    describing functions are the same curve read at different scales
    grid = np.linspace(1e6, 7e6, 1024)
    same_excursion = {}
    for index, limit in enumerate((0.30, 0.40, 0.55, 0.70)):
        drive = np.linspace(0.5, 3.0, grid.size)
        value = clp.describing_function(drive / limit).astype(np.complex128)
        logged = np.log(np.maximum(np.abs(value), 1e-12))
        same_excursion[f"limit {limit}"] = {
            "amplitude": (logged - logged.mean()).astype(np.complex128)}
    collinear = ie.ellipsoid(same_excursion, "amplitude",
                             real_parameters=True)

    # 3. the two DECLARED sites act 0.40 of the span apart, and the sync
    #    tip - which negative modulation puts at peak carrier - is between
    apart = clp.site_separation()
    closed = clp.site_separation(overload_depth=clp.clip_levels()["dark"])

    # 4. and the chain can invert them: last applied is first removed, so
    #    the recorder's clip comes off before the tuner's
    order = inf.ordered_key({
        "input clipping (dark)": np.ones(4, dtype=np.complex128),
        "tuner clipping (sync tip)": np.ones(4, dtype=np.complex128)},
        chain=inf.full_chain())
    lines = [
        f"  two limiters in series against one at the tighter threshold: "
        f"worst difference {worst:.1e}",
        f"  four limiters differing only in threshold: "
        f"{collinear['participation']:.4f} effective of "
        f"{len(same_excursion)} - a RATE, so collinear",
        f"  the declared sites: tuner at span "
        f"{apart['tuner_limit_in_span']:.2f}, recorder at "
        f"{apart['recorder_limit_in_span']:.2f}, window "
        f"{apart['window']:.2f}",
        f"  inverted in the order {' then '.join(n for _, n in order)}",
    ]
    return {"lines": lines,
            "checks": [
                _ok(worst <= 1e-12,
                    "two hard limits in series are one at the tighter",
                    f"{worst:.1e}", "0"),
                _ok(collinear["participation"] < 1.5,
                    "limiters differing only in threshold are collinear",
                    f"{collinear['participation']:.4f}", "< 1.5 of 4"),
                _ok(apart["separable"] and apart["the_tip_is_in_the_window"],
                    "the two sites act at different levels and the sync tip "
                    "is between them",
                    f"window {apart['window']:.2f}", "> 0, tip inside"),
                _ok(not closed["separable"],
                    "and the separation closes at an overload that removes "
                    "the whole sync pulse",
                    f"window {closed['window']:.2f}", "0"),
                _ok([n for _, n in order] == ["input clipping (dark)",
                                              "tuner clipping (sync tip)"],
                    "the recorder's clip is removed before the tuner's",
                    " then ".join(n for _, n in order),
                    "recorder then tuner")],
            "note": "This is the separability law applied to a component "
                    "that could have failed it. Two clips on the SAME "
                    "excursion differ in a rate and compose into one; no "
                    "amount of data splits them. These two are separable "
                    "only because negative modulation puts the sync tip at "
                    "peak carrier - so the receiver clips the tip - while "
                    "the recorder's dark clip is stated 40 per cent above "
                    "it and takes the emphasized undershoot instead. The "
                    "tip's own settled level is the tuner's witness and "
                    "nothing else's. On the data in hand that witness does "
                    "NOT separate the over-the-air sample from the home "
                    "control; see adversary/c10_tuner_site.py."}


@proof("The key doubles when the measured modules join it, and only "
       "orthogonalised and only to a fixed point",
       "interference.admission / orthogonalised; THE_ALGORITHM.md section 5")
def the_modules_join_the_key():
    mechanics = head_model.mechanics_for("VHS", "NTSC", "SP")
    bands = {"carrier only": (2.543e6, 5.257e6),
             "VHS luma band": (1e6, 7e6),
             "full RF": (0.5e6, 12e6)}
    lines, results = [], {}
    for label, (low, high) in bands.items():
        grid = np.linspace(low, high, 1024)
        verdict = inf.admission(grid, mechanics=mechanics)
        built = inf.signatures(grid, mechanics=mechanics,
                               include=verdict["admitted"])
        effective, count, condition = inf._conditioning(built)
        results[label] = (verdict, effective, count, condition)
        lines.append(
            f"  {label:>14}: {verdict['start']['effective']:6.2f} of "
            f"{verdict['start']['count']:2d} at "
            f"{verdict['start']['condition']:6.1f}  ->  {effective:6.2f} of "
            f"{count:2d} at {condition:6.1f}   admitted "
            f"{', '.join(verdict['admitted']) or 'nothing'}")

    luma = results["VHS luma band"][0]
    offers = [s for s in luma["trace"] if s["module"] == "head_differential"]
    lines.append("  head_differential, the same module offered three times "
                 "on the VHS luma band:")
    for offer in offers:
        lines.append(f"      d_eff {offer['delta_effective']:+5.2f}  "
                     f"d_cond {offer['delta_condition']:+7.1f}   "
                     f"{offer['verdict']}")

    # what the SAME modules do entered RAW, which is the trade being avoided
    grid = np.linspace(1e6, 7e6, 1024)
    base = inf.signatures(grid, mechanics=mechanics)
    raw = dict(base)
    for module_name in luma["admitted"]:
        raw.update(inf._candidate_signatures(module_name, grid, mechanics))
    raw_effective, raw_count, raw_condition = inf._conditioning(raw)
    orth_effective, _, orth_condition = results["VHS luma band"][1], 0, \
        results["VHS luma band"][3]
    lines.append(f"  the same modules entered RAW: {raw_effective:.2f} of "
                 f"{raw_count} at condition {raw_condition:.1f}, against "
                 f"{orth_effective:.2f} at {orth_condition:.1f} "
                 f"orthogonalised")

    unsubtractable = [n for n, v in inf.signatures(
        grid, mechanics=mechanics,
        include=luma["admitted"]).items() if not inf.subtractable(v)]
    return {"lines": lines,
            "checks": [
                _ok(results["VHS luma band"][1] > 1.6 *
                    luma["start"]["effective"],
                    "the key's effective directions roughly double",
                    f"{results['VHS luma band'][1]:.2f} from "
                    f"{luma['start']['effective']:.2f}", "> 1.6x"),
                _ok(raw_effective < results["VHS luma band"][1]
                    and raw_condition > results["VHS luma band"][3],
                    "and only orthogonalised - raw is worse on both counts",
                    f"raw {raw_effective:.2f} at {raw_condition:.1f}",
                    f"beaten by {results['VHS luma band'][1]:.2f} at "
                    f"{results['VHS luma band'][3]:.1f}"),
                _ok(len(offers) > 1 and offers[0]["verdict"] == "held"
                    and offers[-1]["verdict"] == "admitted",
                    "and only to a fixed point - a held module turns from a "
                    "bad trade to a good one against a richer key",
                    f"{len(offers)} offers, "
                    f"{offers[0]['verdict']} then {offers[-1]['verdict']}",
                    "held then admitted"),
                _ok(not results["carrier only"][0]["admitted"],
                    "a band too narrow to separate the shapes admits "
                    "nothing", "nothing admitted on 2.5-5.3 MHz",
                    "nothing"),
                _ok(not unsubtractable,
                    "every admitted entry is still subtractable",
                    f"{len(unsubtractable)} failing", "0")],
            "note": "The 27 components the parallel build declared were "
                    "outside the key entirely: the chain could ORDER them "
                    "and the key could not USE them. Entered raw the key is "
                    "not merely worse but SINGULAR - condition 2e15, "
                    "numerically rank deficient, so it cannot be "
                    "inverted at all. That is proof 31's rule and not a "
                    "new one. "
                    "What is new is that a single orthogonalising pass is "
                    "not enough either - head_differential is a bad trade "
                    "against the bare key and a good one against the key "
                    "that tape_path and picture_stage have already "
                    "enriched, so the offer has to be repeated until a pass "
                    "admits nothing. vertical_interval is held everywhere: "
                    "its entries are specified test SIGNALS rather than "
                    "modifications of the chain."}


# --------------------------------------------------------------------------


def run(selected: Optional[List[int]] = None,
        report: Optional[str] = None) -> int:
    chosen = [(index, item) for index, item in enumerate(PROOFS, start=1)
              if selected is None or index in selected]
    out: List[str] = []
    passed = failed = 0
    for index, (title, cites, function) in chosen:
        out.append(f"\n{'=' * 74}")
        out.append(f"PROOF {index}. {title}")
        out.append(f"    cited by: {cites}")
        out.append("=" * 74)
        try:
            result = function()
        except Exception as error:                # a proof that cannot run
            out.append(f"  ERROR: {error}")
            failed += 1
            continue
        out.extend(result.get("lines", []))
        for check in result.get("checks", []):
            mark = "PASS" if check["pass"] else "FAIL"
            passed += check["pass"]
            failed += not check["pass"]
            out.append(f"  [{mark}] {check['claim']}")
            out.append(f"         got {check['got']}, expected "
                       f"{check['expected']}")
        if result.get("note"):
            out.append(f"  -- {result['note']}")
    out.append(f"\n{'=' * 74}")
    out.append(f"{passed} checks passed, {failed} failed, "
               f"{len(chosen)} proofs run")
    out.append("=" * 74)
    text = "\n".join(out)
    print(text)
    if report:
        with open(report, "w") as handle:
            handle.write("# The elliptical collapse, proved\n\n"
                         "Regenerated by "
                         "`python3 -m tools.ringing_measure.proofs`.\n\n"
                         "```\n" + text + "\n```\n")
        print(f"\nwritten to {report}")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--report", default=None)
    arguments = parser.parse_args(argv)
    if arguments.list:
        for index, (title, cites, _) in enumerate(PROOFS, start=1):
            print(f"{index:>3}. {title}\n     {cites}")
        return 0
    return run(arguments.only, arguments.report)


if __name__ == "__main__":
    raise SystemExit(main())
