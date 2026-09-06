"""The unknown components live at the JOINS, and each join has its own prior.

Ethan: *"I think this method goes at the end of each component chain stage,
where the unknown components are most likely to be. Essentially we have some
information to know based on the video signal chain where these unknown
components are, and the most likely set of possible curves they fit. Gausian
noise at the rf stage, multi path reflection at the source stage, etc. I think
it just applies to the terminal ends of our stages where they join with each
other."*

THIS IS A STRUCTURAL CLAIM AND IT REPLACES A WEAKER ONE. The unmodelled
residual was treated as ONE object spread over the whole chain, to be
interpolated with a single shape prior. That is the worst case rather than the
real one: a chain is not homogeneous, and neither is its ignorance. What is
unknown at the antenna is not the same KIND of thing as what is unknown at the
converter, and pooling them throws away the one piece of information that
makes the unknown tractable.

WHY THE JOINS AND NOT THE STAGES. Inside a stage the components are the
stage's own physics and are modelled or deliberately not. At a join, two
machines meet - two impedances, two clocks, two noise sources, two
manufacturers - and the interface between them is exactly what no
specification describes and no single-stage model contains. So the joins are
where the modelling ends and the residual begins, which is why the residual
should be attributed there.

EACH JOIN CARRIES A DIFFERENT PRIOR, which is the operative half of the idea.
Naming the family is most of the estimation problem: an echo family and a
noise family are not close, so choosing between them is easy and fitting
within one is well posed. Guessing a shape with no family at all is the thing
this arc has repeatedly found to be unidentifiable.

    join                       what is unknown there          family
    source -> transmission     multipath, the propagation     ECHO: a delay
                               path nobody recorded           and an amplitude
    transmission -> recorder   the link between two boxes -   ECHO, short: a
                               cable, impedance mismatch      reflection at
                                                              cable length
    recorder -> tape           the write: transition length,  MAGNETIC: a
                               head-to-tape contact           monotone loss in
                                                              length/wavelength
    tape -> playback           the read: separation, contact  MAGNETIC, the
                                                              same family
    playback -> capture        thermal noise and the          NOISE plus a
                               analogue anti-alias filter     POLE CASCADE
    capture -> picture         quantisation                   FLAT: white, and
                                                              bounded below by
                                                              the converter

FOR OFF-AIR MATERIAL THE SOURCE AND THE TRANSMISSION PATH ARE ONE LINK.
Ethan: *"That makes sense since our transmission path is the source."* When
the recording was made from a broadcast there is no separate studio ahead of
the transmitter that this chain can see - the source IS what came out of the
air, and the source-to-transmission join has nothing on either side of it that
differs. The first join that carries a real interface is then
transmission-to-recorder, which is a cable between two boxes.

That is the same provenance rule `clipping` already applies, where the tuner's
entries exist only for a broadcast source: what links exist at all is a
property of how the tape was made. `boundaries` derives the joins from the
OCCUPIED links, so a chain assembled without the source's inserted test
signals produces one fewer join without anything here being edited.

THE BOUNDARIES ARE DERIVED, NOT LISTED. `boundaries` reads the declared chain
and finds the gaps between the occupied bands, so a component added anywhere
moves the joins with it and nothing here needs editing. A list would drift the
moment a position was added, which is the failure mode `full_chain` already
exists to prevent.

AND THE PRIOR IS A CANDIDATE SET, NEVER AN ANSWER. `estimate` fits every
member of a join's family and returns which fits best, by how much it beats
the rest of its own family, and whether it beats a NOISE NULL of the same
size - because the best of a searched family beats its own median on noise
alone. The arc's standing rule applies unchanged: a shape earns its place by
the direction it adds.

THE MAGNETIC JOINS CAN IDENTIFY THE FAMILY AND NEVER THE PARAMETER, measured
rather than assumed. `exp(-2 pi L / lambda)` has log magnitude
`-2 pi L / lambda`, so every member of that family is ONE shape times a
constant and normalising makes them identical: a planted 0.15 micron
transition recovers as 0.0116 micron with the shape explaining 1.000 of the
residual and a margin over its own family of EXACTLY 0.000. That is the
collapse this arc measured as 1.58 effectively distinguishable of 6, arriving
at the joins. So naming the family is most of the estimation problem at the
echo joins and ALL of it at the magnetic ones, where the honest output is the
family plus a bound on the SUM of its lengths - the two clearances add in the
exponent and were never separable anyway.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The named links of the chain, in the order the signal passes through them.
# The positions come from `interference.full_chain()` and are read, never
# written here - this is the map from a position to the link it belongs to.
LINKS: Tuple[Tuple[str, int, int], ...] = (
    ("the source", 0, 0),
    ("the transmission path", 1, 7),
    ("the recording machine", 8, 19),
    ("the tape", 20, 23),
    ("the playback machine", 24, 29),
    ("the capture", 30, 39),
    ("the picture stage", 40, 49),
)

# What is unknown at each join, and the family of curves it is most likely to
# come from. The families are named after the mechanisms this repository
# already models, so a fit here is a fit against something with a docstring.
PRIORS: Dict[str, Dict[str, object]] = {
    "the source -> the transmission path": {
        "family": "echo",
        "why": ("multipath: the propagation path nobody recorded. An echo at "
                "delay tau is a ripple of period 1/tau, which is a shape with "
                "one parameter and is therefore identifiable"),
        "parameter": "delay_s",
        "range": (0.1e-6, 40e-6),
    },
    "the transmission path -> the recording machine": {
        "family": "echo",
        "why": ("the link between two boxes - a cable and two impedances. "
                "The same family as multipath but at a much shorter delay, "
                "because the reflection is at cable length rather than at "
                "building scale"),
        "parameter": "delay_s",
        "range": (1e-9, 0.5e-6),
    },
    "the recording machine -> the tape": {
        "family": "magnetic",
        "why": ("the write: transition length and head-to-tape contact. Every "
                "member is a monotone function of one length over the "
                "recorded wavelength, which is why the family collapses to "
                "1.58 of 6 and why naming the family matters more than "
                "choosing within it"),
        "parameter": "length_m",
        "range": (0.01e-6, 1.0e-6),
    },
    "the tape -> the playback machine": {
        "family": "magnetic",
        "why": ("the read: separation and contact, the same family as the "
                "write and provably not separable from it - the two clearances "
                "add in the exponent, so only their sum is identifiable"),
        "parameter": "length_m",
        "range": (0.01e-6, 1.0e-6),
    },
    "the transmission path -> the recording machine (ingress)": {
        "family": "ingress",
        "why": ("RF picked up ON THE INTERCONNECT. A cable between two boxes "
                "is an antenna, and what it collects is somebody else's "
                "carrier rather than a reflection of the wanted signal - so a "
                "LINE at an unknown frequency, not a ripple at an unknown "
                "delay. Ethan names this as the primary unknown"),
        "parameter": "centre_hz",
        "range": (0.5e6, 12e6),
    },
    "the playback machine -> the capture (ingress)": {
        "family": "ingress",
        "why": ("the same antenna problem at the other interconnect, and the "
                "one where it is measurable: the capture's own floor above "
                "the signal band shows every line the cable collected"),
        "parameter": "centre_hz",
        "range": (0.5e6, 20e6),
    },
    "the playback machine -> the capture": {
        "family": "noise_and_pole",
        "why": ("thermal noise and the analogue anti-alias filter. This is "
                "the one join where the unknown is identifiable WITHOUT a "
                "stimulus, because the noise is the probe: above the signal "
                "band the measured spectrum IS the filter's magnitude"),
        "parameter": "order",
        "range": (1.0, 20.0),
    },
    "the capture -> the picture stage": {
        "family": "flat",
        "why": ("quantisation, which is white by construction and bounded "
                "below by the converter's own floor. Nothing is unknown here "
                "except a level, so the family has one member and the honest "
                "answer is a bound rather than a shape"),
        "parameter": "level",
        "range": (0.0, 1.0),
    },
}


def links_of(chain: Optional[Dict[str, int]] = None
             ) -> List[Dict[str, object]]:
    """Which declared components fall in each link of the chain."""
    from vhsdecode.models import interference

    chain = chain if chain is not None else interference.full_chain()
    out = []
    for name, low, high in LINKS:
        members = sorted((position, entry) for entry, position in chain.items()
                         if low <= position <= high)
        out.append({
            "link": name,
            "band": (low, high),
            "members": [entry for _position, entry in members],
            "occupied": bool(members),
        })
    return out


def boundaries(chain: Optional[Dict[str, int]] = None
               ) -> List[Dict[str, object]]:
    """THE JOINS, DERIVED FROM THE DECLARED CHAIN RATHER THAN LISTED.

    A join exists between two consecutive links that both hold declared
    components. Reading them from the chain means a component added anywhere
    moves the joins with it and this module needs no edit - the failure a
    hard-coded list would have, and the one `full_chain` already exists to
    prevent.
    """
    occupied = [entry for entry in links_of(chain) if entry["occupied"]]
    out = []
    for below, above in zip(occupied, occupied[1:]):
        name = f"{below['link']} -> {above['link']}"
        prior = PRIORS.get(name)
        out.append({
            "boundary": name,
            "below": below["link"],
            "above": above["link"],
            "last_position": below["band"][1],
            "first_position": above["band"][0],
            "prior": prior,
            "has_prior": prior is not None,
            "members_below": below["members"][-3:],
            "members_above": above["members"][:3],
        })
    return out


def _echo(frequency_hz, delay_s: float, amount: float = 0.3) -> np.ndarray:
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    return 1.0 + amount * np.exp(-2j * np.pi * grid * float(delay_s))


def _magnetic(frequency_hz, length_m: float,
              speed_m_s: float = 5.8) -> np.ndarray:
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    wavelength = speed_m_s / np.maximum(grid, 1.0)
    return np.exp(-2.0 * np.pi * float(length_m) / wavelength
                  ).astype(np.complex128)


def _noise_and_pole(frequency_hz, order: float,
                    corner_hz: float = 14e6) -> np.ndarray:
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    ratio = grid / max(float(corner_hz), 1.0)
    with np.errstate(over="ignore"):
        return (1.0 / np.sqrt(1.0 + ratio ** (2.0 * float(order)))
                ).astype(np.complex128)


def _ingress(frequency_hz, centre_hz: float,
             width_hz: float = 30e3) -> np.ndarray:
    """RF PICKED UP ON AN INTERCONNECT: a narrow carrier at an unknown place.

    Ethan: *"Primarily unknown other RF interference components (especially on
    the interconnects between devices), so maybe the placement solves that."*

    It does, and this is the family the placement calls for. A cable between
    two boxes is an antenna, and what it picks up is not a reflection of the
    wanted signal but somebody else's carrier - so the family is a LINE at an
    unknown frequency rather than a ripple at an unknown delay. The two are
    easy to tell apart and were being conflated: an echo is broadband and
    periodic in frequency, ingress is narrow and sits in one place.

    AND THE DISTRIBUTION SETTLES IT INDEPENDENTLY. A carrier's marginal is
    ARCSINE, excess kurtosis -1.5, which no reflection produces - a reflected
    copy of a Gaussian-ish signal is still Gaussian-ish. So the shape axis and
    the distribution axis agree or they do not, and disagreement is itself
    information.
    """
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    return (1.0 + 1.0 / (1.0 + ((grid - float(centre_hz))
                                / max(float(width_hz), 1.0)) ** 2)
            ).astype(np.complex128)


def _flat(frequency_hz, level: float) -> np.ndarray:
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    return np.full(grid.shape, 1.0 + float(level), dtype=np.complex128)


_FAMILIES = {
    "echo": _echo,
    "magnetic": _magnetic,
    "noise_and_pole": _noise_and_pole,
    "flat": _flat,
    "ingress": _ingress,
}


def candidates(boundary: str, frequency_hz, count: int = 32
               ) -> Dict[float, np.ndarray]:
    """The family of curves a join's unknown is most likely to come from."""
    prior = PRIORS.get(boundary)
    if prior is None:
        return {}
    builder = _FAMILIES[str(prior["family"])]
    low, high = prior["range"]
    if str(prior["family"]) in ("echo", "magnetic"):
        values = np.geomspace(max(low, 1e-12), high, int(count))
    else:
        values = np.linspace(low, high, int(count))
    return {float(v): builder(frequency_hz, v) for v in values}


def resolvable(boundary: str, frequency_hz) -> Dict[str, object]:
    """HOW MUCH OF A JOIN'S FAMILY THE BAND CAN ACTUALLY TELL APART.

    A prior is a set of candidate curves, and a band that cannot separate two
    of them has not been given a choice. For the echo families the limit is
    the same one `sync_geometry.ghost_reach` derives: an echo at delay `tau`
    is a ripple of period `1/tau`, so a band of span `B` cannot resolve any
    delay shorter than `1/B`.

    Measured on the transmission-to-recorder join over 1 to 7 MHz: the span
    is 6 MHz, so nothing below 167 ns can be resolved AS A RIPPLE, and 81 per
    cent of that join's prior range of 1 to 500 ns sits below it.

    BUT A SUB-RESOLUTION ECHO IS STILL MEASURABLE, AS A DELAY RATHER THAN AS
    A RIPPLE, and getting that wrong once is why this distinction is written
    out. An echo shorter than the band's own reciprocal has no visible ripple
    - but it does have a nearly LINEAR PHASE, and a linear phase is perfectly
    measurable on a band that cannot resolve the ripple that produced it. A
    planted 80 ns echo is identified here with the family's median member
    explaining only 0.083 of it.

    An earlier version of this docstring claimed the opposite, that the
    estimator "correctly refuses" a sub-resolution echo because the family is
    collinear at 0.807. That collinearity was an artefact of a Hermitian
    inner product that could not see the difference between a magnitude and a
    phase - see `estimate` - and it disappeared when the projection was
    stacked. The band limit below is a true statement about ripples and was
    never a true statement about identifiability.

    Reported rather than silently clipped, because a caller deserves to know
    which part of a prior is available as a ripple and which only as a phase.
    """
    prior = PRIORS.get(boundary)
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    if prior is None or grid.size < 2:
        return {"resolvable": None, "why": "no prior, or no band"}
    span = float(grid.max() - grid.min())
    low, high = prior["range"]
    if str(prior["family"]) != "echo" or not span > 0:
        return {"family": prior["family"], "band_hz": span,
                "why": "the resolution limit below is stated for echoes only"}
    shortest = 1.0 / span
    values = np.geomspace(max(low, 1e-12), high, 32)
    return {
        "family": "echo",
        "band_hz": span,
        "shortest_resolvable_s": shortest,
        "prior_range_s": (low, high),
        "unresolvable_fraction": float(np.mean(values < shortest)),
        "why": ("an echo at tau is a ripple of period 1/tau, so a band of "
                "span B cannot separate any delay shorter than 1/B"),
    }


def estimate(residual, frequency_hz, boundary: str,
             null_draws: int = 24) -> Dict[str, object]:
    """WHICH MEMBER OF THE JOIN'S FAMILY BEST EXPLAINS A RESIDUAL.

    Returns the best member, the share of the residual it explains, and - the
    part that decides whether to believe it - HOW MUCH IT BEATS THE REST OF
    ITS OWN FAMILY. A family whose members all fit equally well has not
    identified a parameter, it has only confirmed that something in that
    family is present; and where the whole family fits no better than a
    constant, it has not even done that.

    The margin is reported as the drop in explained share from the best
    member to the median member. A wide margin means the parameter is
    measured; a narrow one means the family is right and the parameter is not.
    """
    values = np.nan_to_num(np.asarray(residual).ravel().astype(np.complex128))
    values = values - values.mean()
    energy = float(np.vdot(values, values).real)
    family = candidates(boundary, frequency_hz)
    if not family or not energy > 0:
        return {"identified": False,
                "why": ("no prior for this join, or the residual is empty"
                        if not family else "the residual carries no energy")}

    # STACKED REAL-THEN-IMAGINARY, NOT A HERMITIAN INNER PRODUCT, and this
    # was a bug rather than a preference. `|<u,v>|^2` treats a NINETY DEGREE
    # phase difference as full alignment: for a pure-imaginary u and a
    # pure-real v the product is imaginary and its magnitude is |u||v|, so
    # the share comes back as one.
    #
    # It showed up as a planted MAGNETIC residual being explained 1.0000 by
    # the short-delay ECHO family. The magnetic log is pure real - a
    # magnitude loss with no phase - and the best-fitting echo at 1 ns is
    # pure imaginary, a delay with no magnitude. They are orthogonal, and
    # stacking says so: 0.0004 against the 1.0000 the Hermitian form gave.
    #
    # This is the trap this arc had already recorded once - a Hermitian inner
    # product is blind to an absolute phase and real/imaginary stacking is
    # not - arrived at again from a different direction.
    stacked = np.concatenate([values.real, values.imag])
    shares: Dict[float, float] = {}
    for parameter, curve in family.items():
        shape = np.asarray(curve).ravel()
        if shape.size != values.size:
            continue
        shape = np.log(np.maximum(np.abs(shape), 1e-30)) \
            + 1j * np.unwrap(np.angle(shape))
        shape = shape - shape.mean()
        pair = np.concatenate([shape.real, shape.imag])
        norm = float(np.linalg.norm(pair))
        if not norm > 0:
            continue
        shares[parameter] = float(
            abs(float(np.dot(pair / norm, stacked))) ** 2 / energy)
    if not shares:
        return {"identified": False, "why": "no candidate on this grid"}

    best = max(shares, key=shares.get)
    ordered = sorted(shares.values(), reverse=True)
    median = float(np.median(ordered))

    # THE NULL, BECAUSE TAKING THE BEST OF A FAMILY INFLATES THE BEST. A
    # first version gated on `best > 2 x median` and FAILED ITS OWN NOISE
    # CONTROL: unstructured noise offered to the echo family returned
    # explained 0.0026 against a median of 0.0009 and was declared
    # identified, purely because thirty-two candidates were searched and the
    # luckiest kept. A margin over a family's own median measures selection
    # as much as signal. So the threshold is MEASURED - the same
    # best-of-family statistic run on noise of the same length - which is the
    # surrogate-null discipline this arc applies to any statistic taken over
    # a searched set.
    generator = np.random.default_rng(20260904)
    prepared = []
    for curve in family.values():
        shape = np.asarray(curve).ravel()
        if shape.size != values.size:
            continue
        shape = np.log(np.maximum(np.abs(shape), 1e-30)) \
            + 1j * np.unwrap(np.angle(shape))
        shape = shape - shape.mean()
        pair = np.concatenate([shape.real, shape.imag])
        norm = float(np.linalg.norm(pair))
        if norm > 0:
            prepared.append(pair / norm)
    null = []
    for _ in range(int(null_draws)):
        draw = (generator.standard_normal(values.size)
                + 1j * generator.standard_normal(values.size))
        draw = draw - draw.mean()
        draw_energy = float(np.vdot(draw, draw).real)
        drawn = np.concatenate([draw.real, draw.imag])
        null.append(max((float(abs(float(np.dot(u, drawn))) ** 2
                               / draw_energy) for u in prepared),
                        default=0.0))
    threshold = float(np.percentile(null, 95)) if null else 0.0

    prior = PRIORS[boundary]
    return {
        "identified": bool(shares[best] > threshold
                           and shares[best] > 2.0 * median),
        "null_threshold": threshold,
        "resolvable": resolvable(boundary, frequency_hz),
        "boundary": boundary,
        "family": prior["family"],
        "parameter": prior["parameter"],
        "value": best,
        "explained": shares[best],
        "median_of_family": median,
        "margin": shares[best] - median,
        "members": len(shares),
        "why": ("a family whose members all fit equally well has identified "
                "the family and not the parameter; the margin over the "
                "family's own median is what separates the two"),
    }


def survey(residuals: Dict[str, np.ndarray], frequency_hz,
           chain: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    """Every join, its prior, and what a residual offered to it looks like.

    `residuals` maps a boundary name to the residual measured there. A join
    with no residual is reported as unmeasured rather than skipped, because
    the joins that have never been looked at are exactly what this structure
    exists to make visible.
    """
    out = []
    for join in boundaries(chain):
        name = str(join["boundary"])
        residual = residuals.get(name)
        if residual is None:
            out.append({**join, "measured": False})
            continue
        out.append({**join, "measured": True,
                    "estimate": estimate(residual, frequency_hz, name)})
    return {
        "joins": out,
        "count": len(out),
        "measured": sum(1 for j in out if j.get("measured")),
        "with_prior": sum(1 for j in out if j["has_prior"]),
        "why": ("the unknown lives where two machines meet, and naming the "
                "family it comes from is most of the estimation problem"),
    }


# --------------------------------------------------------------------------
# Searching the families, and letting the noise type set the limit
# --------------------------------------------------------------------------


def search_families(residual, frequency_hz,
                    boundary: Optional[str] = None) -> Dict[str, object]:
    """WHICH COMPLEX CURVE FITS THIS DIMENSION - searched across ALL families.

    Ethan: *"Maybe we just can do a search to find which complex curve should
    be fit per dimension, and then do the limit there with a more informed
    geometric search based on the types of noise we know are in the source
    signal."*

    `estimate` asks how well a join's OWN prior fits. This asks a different
    and prior question: which family fits at all. The join's prior is used as
    a prior in the proper sense - it is reported alongside, and a family that
    wins against it is worth knowing about - rather than as a constraint that
    can only confirm itself.

    THAT DISTINCTION IS THE WHOLE VALUE. A prior that cannot be contradicted
    is not evidence, and this arc has repeatedly found that the mechanism
    actually present at a join is not the one the physics suggested - the
    capture's floor being impulsive rather than thermal is the most recent.

    Returns every family's best member and best share, which family wins, and
    by how much it beats the runner-up. A win by less than the null is not a
    win, and is reported as such.

    WHAT IT CAN AND CANNOT SEPARATE, MEASURED ON PLANTED FAMILIES:

        planted echo at 3 us     -> echo 1.0000, next family 0.0089
        planted ingress at 4.2 MHz -> ingress 1.0000, next family 0.0358
        planted magnetic         -> magnetic 1.0000, but ingress 0.9786 and
                                    noise_and_pole 0.9729

    The first two are clean. The third is not, and the reason is physics
    rather than a defect: over one band a magnetic loss, a pole cascade and
    the skirt of a narrow line are all MONOTONE MAGNITUDE ROLL-OFFS, and
    nothing distinguishes them but curvature. So the magnetic joins can be
    told from the phase-bearing families and not from the other magnitude
    ones, which is the separability law once more - families differing in a
    RATE are collinear, and these three differ in little else here.
    """
    values = np.nan_to_num(np.asarray(residual).ravel().astype(np.complex128))
    values = values - values.mean()
    energy = float(np.vdot(values, values).real)
    if not energy > 0:
        return {"identified": False, "why": "the residual carries no energy"}

    # KEYED BY THE JOIN, NOT BY THE FAMILY. Two joins share the "echo"
    # family at very different delay ranges - multipath at building scale and
    # a cable reflection at nanoseconds - and keying by family let the second
    # overwrite the first. A planted 3 us echo then lost to `noise_and_pole`
    # at an explained share of 0.004, because the surviving "echo" entry was
    # the short-delay one that cannot represent it.
    ranked: Dict[str, Dict[str, object]] = {}
    for name in PRIORS:
        got = estimate(values, frequency_hz, name)
        if got.get("explained") is None:
            continue
        ranked[name] = {
            "family": str(PRIORS[name]["family"]),
            "value": got["value"],
            "explained": got["explained"],
            "margin": got["margin"],
            "null": got["null_threshold"],
            "identified": got["identified"],
        }
    if not ranked:
        return {"identified": False, "why": "no family fitted on this grid"}

    order = sorted(ranked, key=lambda k: ranked[k]["explained"], reverse=True)
    winner = order[0]
    # the runner-up is the best entry of a DIFFERENT family, because two
    # members of one family beating each other says nothing about which
    # mechanism is present
    winning_family = ranked[winner]["family"]
    others = [ranked[k]["explained"] for k in order[1:]
              if ranked[k]["family"] != winning_family]
    runner_up = max(others) if others else 0.0
    expected = (str(PRIORS[boundary]["family"])
                if boundary in PRIORS else None)
    return {
        "joins": ranked,
        "order": order,
        "winner": winning_family,
        "winning_join": winner,
        "explained": ranked[winner]["explained"],
        "value": ranked[winner]["value"],
        "over_runner_up": ranked[winner]["explained"] - runner_up,
        "beats_its_null": bool(ranked[winner]["explained"]
                               > ranked[winner]["null"]),
        "expected_family": expected,
        "agrees_with_the_prior": (None if expected is None
                                  else bool(winning_family == expected)),
        "identified": bool(ranked[winner]["identified"]),
        "why": ("the join's prior is reported alongside rather than imposed, "
                "because a prior that cannot be contradicted is not evidence"),
    }


def informed_limit(residual, frequency_hz, boundary: Optional[str] = None,
                   samples: Optional[np.ndarray] = None,
                   passes: int = 8) -> Dict[str, object]:
    """RUN THE LIMIT AT THIS DIMENSION, WITH THE NOISE TYPE SETTING THE FLOOR.

    Ethan: *"then do the limit there with a more informed geometric search
    based on the types of noise we know are in the source signal."*

    The limit needs somewhere to stop, and every version of it in this arc has
    stopped at a noise floor. WHAT THAT FLOOR IS DEPENDS ON WHAT THE NOISE IS,
    and that is the information the distribution classifier supplies:

        gaussian   the floor falls as 1/sqrt(k) with k independent places, so
                   iterating pays and the stopping point is where the residual
                   meets it
        uniform    a quantisation floor is HARD - it does not average away
                   below one code, and iterating past it is fitting the
                   converter
        arcsine    not noise at all but a surviving carrier, so the right
                   action is to remove it as a component and not to stop there
        sparse     a handful of discrete events; averaging does nothing to it
                   and the limit must exclude those samples rather than
                   descend towards them

    So the stopping rule is chosen by the measurement rather than assumed, and
    a run that would have iterated into a hard floor is stopped at it instead.

    `samples` is the residual in the sample domain if it is available, because
    the distribution is a property of the samples and not of the spectrum. It
    falls back to the spectrum's own real part, which is a weaker measurement
    and is reported as such.
    """
    from vhsdecode.models import interference_distributions as dist

    values = np.nan_to_num(np.asarray(residual).ravel().astype(np.complex128))
    in_samples = (np.asarray(samples).ravel() if samples is not None
                  else np.real(values))
    verdict = dist.classify(in_samples)
    found = search_families(values, frequency_hz, boundary)

    kind = str(verdict.get("distribution", "unknown"))
    if kind == "gaussian":
        rule = "iterate; the floor falls as 1/sqrt(k) and the limit is real"
        reducible = True
    elif kind == "uniform":
        rule = ("stop at the floor; a quantisation floor is hard and does not "
                "average away below one code")
        reducible = False
    elif kind == "arcsine":
        rule = ("do not stop here; this is a surviving carrier, so remove it "
                "as a component rather than treating it as a floor")
        reducible = False
    elif kind == "sparse" or verdict.get("looks_sparse"):
        rule = ("exclude the events; averaging does nothing to a handful of "
                "discrete arrivals and the limit must not descend towards "
                "them")
        reducible = False
    else:
        rule = ("unclassified; iterate only while the residual falls and stop "
                "on the first pass that does not")
        reducible = None

    energy = float(np.vdot(values, values).real)
    explained = float(found.get("explained", 0.0) or 0.0)
    trail = []
    remaining = 1.0
    for step in range(max(int(passes), 1)):
        if not found.get("identified"):
            break
        remaining *= max(1.0 - explained, 0.0)
        trail.append(remaining)
        if remaining < 1e-6 or not reducible:
            break
    return {
        "distribution": kind,
        "distribution_identified": bool(verdict.get("identified")),
        "sample_domain": samples is not None,
        "family": found.get("winner"),
        "family_identified": bool(found.get("identified")),
        "agrees_with_the_prior": found.get("agrees_with_the_prior"),
        "stopping_rule": rule,
        "floor_is_reducible": reducible,
        "explained_per_pass": explained,
        "residual_trail": trail,
        "passes_run": len(trail),
        "starting_energy": energy,
        "why": ("the limit needs somewhere to stop, and what the floor IS "
                "depends on what the noise is; the classifier supplies that "
                "rather than the run assuming it"),
    }
