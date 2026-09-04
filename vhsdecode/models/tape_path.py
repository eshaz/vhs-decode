"""The tape path as a mechanism: seven ways the transport sets the spacing.

The head model describes the magnetics, the transport model the rates, the
filter model the electronics, the tape model the medium. This is the
mechanical interface between the last two - the geometry and the forces that
decide how far the coating sits from the gap while the head is reading it.

Seven parameters of the path are modelled, and every one of them was named
because it moves the head-to-tape separation: the tension the reels hold and
the contact pressure it produces, the wrap angle and the head's penetration
into the tape path, the guide geometry working against the tape's own bending
stiffness, the air the moving tape drags into the interface, the pack radius
changing through a pass, and the head's protrusion and its wear.

THE CENTRAL QUESTION, AND THE ANSWER IS THE UNCOMFORTABLE ONE. Since all
seven act through the separation, and the separation reaches the response
only through Wallace's law `exp(-2 pi d / lambda)`, their log magnitudes are
all `-2 pi d f / v` - the SAME straight line through the origin, differing
only in slope. As a family of frequency responses they are not seven
mechanisms. They are seven values of one number.

    frequency axis, amplitude only      1.000000 of 7, rank 1
    worst coherent pair                 1.000000, and EVERY pair is 1.000000
    raw condition                       2.6e+30, which is the reciprocal of
                                        the machine epsilon and is not a
                                        number about tape at all

That is far worse than the tape's own magnetics, which reach 1.58 of 6
because gap, thickness, azimuth and contour are sinc and exponential-integral
forms rather than one exponential. The separability law in
`COMPONENT_MAPPINGS.md` section 2 says a family is collinear when its members
differ only in a RATE and separable when they differ in KIND, and on the
amplitude axis this family differs in nothing else whatsoever. THESE ARE NOT
NEW KINDS. They are seven more members of a collinear family, and the family
is tighter than the magnetics it joins.

WHAT IS SEPARABLE, AND IT IS TWO THINGS.

**In frequency, a second KIND appears once the phase is carried.** Four of
the seven act on the tape's TENSION, and tension strains the tape, and a
strained tape plays back at a slightly different rate - a timing signature,
not an amplitude one. Three of the seven (entrainment at the format's fixed
drum speed, head protrusion, head wear) change the clearance without touching
the tension, so they have no timing part at all. A clearance and a strain are
a gain and a delay of the same shape, which Proposition 9.3 says are one
direction under the Hermitian inner product and two under the stacked real
one. Measured with real and imaginary parts stacked, on 1 to 7 MHz:

    frequency axis, amplitude and phase   1.734 of 7, rank 2, condition 1.51

Two ranked directions of seven, and the second exists only because the phase
was kept. That 1.734 is NOT comparable with the magnetics' 1.58: the
magnetics figure is amplitude only, and the like-for-like comparison is the
1.000000 above. What is comparable is that the path has two KINDS where the
magnetics have one, and six fewer members holding them. Its resolution depends on the one number in this module that no
specification supplies - the base film's modulus, which sets how large the
phase part is beside the amplitude part - and `modulus_sensitivity` measures
that dependence rather than assuming it away: over a factor of sixteen in
modulus the count runs 1.94 down to 1.29 and THE RANK STAYS AT TWO
throughout. So the second direction is real at any plausible modulus; only
how well it is resolved is uncertain.

**In time, they separate, and the rate is the whole reason.** A parameter
that varies at a known rate is distinguishable from one that does not even
when their frequency shapes are identical, and
`transport_model.rotation_rates` states every rate from the parts' diameters
before anything is measured. Over 512 fields at the NTSC field rate:

    time axis                             4.597 of 7, rank 5, condition 1.66

Not seven, and the shortfall is the honest part. Two mechanisms are STATIC on
the scale of a recording - the wrap and penetration, set when the deck is
threaded, and the entrained air film at a drum speed the format fixes at
5.80 m/s - so they have no time signature to have and their rows are exactly
zero. Two more, the pack radius and head wear, are both monotone trends and
would be one direction were it not that the pack strains the tape and wear
does not, which splits them by kind and is why the count is 4.60 rather than
the 3.56 the shapes alone give. The rest land on a reel's rotation, a guide's
rotation and the head parity, and separate cleanly.

**And the two axes together buy nothing over time alone.** On the outer
product grid the count is 4.597196, equal to the time axis's 4.597196 to
eleven decimal places, because the frequency factor is proportional to `f`
for both the clearance and the delay and is therefore COMMON to all seven -
and a shared factor cannot separate anything. That equality is a check on the
construction rather than a coincidence: a rank-one product must produce it,
and a joint count exceeding the time count would mean a direction had been
manufactured from the grid.

WHAT THIS MEANS FOR THE KEY, AND THE ANSWER IS A NEGATIVE RESULT. The rule is
the one the sub pre-emphasis and the record level both taught: an entry earns
its place by the direction it adds, not by being a mechanism. Measured by
`key_impact` against the 14-entry VHS luma-band key, which stands at 7.908
effective at condition 7.76:

    adding the clearance alone            7.584 of 15, condition 110
    adding the strain alone               7.762 of 15, condition 40
    adding both                           7.558 of 16, condition 110
    adding all seven members              5.587 of 21, condition 122

EVERY ARRANGEMENT MAKES THE KEY WORSE, and the reason is nameable: the
clearance's log magnitude is a straight line in frequency and reads 0.972
coherent with the particle-noise entry, whose own law is a power of
frequency, while the strain is a linear phase ramp and reads 0.982 coherent
with the group-delay entry, whose quadratic phase over a band of this
fractional width is very nearly the same curve. So this component recommends
that NOTHING be added to `interference.signatures()`. Its two entries are
defined, named, ordered and subtractable, and its contribution is on the time
axis, where the key has no entries at all.

THE CONTROL. `separability_control` runs the same participation-ratio code
over two families whose answers are known independently and are DIFFERENT
from one another: a set of pure Wallace separation losses at five clearances,
which must read 1.00 because `-2 pi d f / v` at five values of `d` is one
vector at five scales, and a set of five echoes at delays spaced beyond the
band's resolution, which must read above 4.5 of 5 because
`exp(-2 pi j f tau)` is a Fourier basis over delay. A construction that has
collapsed everything by accident passes the first and fails the second; one
that manufactures directions out of rounding - the failure `magnetic.py`
records, where a control that should have read 1.00 read 4.97 - fails the
first. Measured: 1.000000 and 4.92. `entrainment_exponent_control` recovers
0.666667 and -0.666667 from the foil-bearing closed form, and
`pressure_control` checks the identity `p R w = T` to 5.6e-17 newtons.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from vhsdecode.models import head_model, transport_model

# --------------------------------------------------------------------------
# The specification, and what is assumed instead
# --------------------------------------------------------------------------

# JVC Video Technical Guide VTG82063: the tape tension the transport is
# required to hold. Everything below that depends on tension is evaluated at
# the middle of this range unless the caller supplies a measurement.
TENSION_RANGE_N = (0.30, 0.45)

# The format's tape width. VHS is the nominal half inch, 12.65 mm, and it is
# the width the tension is spread across, so tension per unit width - the
# quantity every foil-bearing result is written in - is the tension divided
# by this.
TAPE_WIDTH_M = 12.65e-3

# JVC VTG82063 section 1, quoted already in `tape_model`: the tape is
# 19 +/- 2 um overall. The polyester base carries essentially all of the
# tensile load and is thinner than the whole tape, so using the total here
# UNDERSTATES the strain by the base-to-total ratio, about 23 per cent. It is
# used because the total is what the specification states.
TAPE_THICKNESS_M = 19.0e-6

# The drum's diameter is a specification certainty for this format - a
# machine must have it, because it is what lays down the track geometry - and
# it is already declared once in `transport_model`. Read it from there rather
# than keeping a second copy that can drift.
_DRUM = next(part for part in transport_model.TRANSPORT
             if part["kind"] == "drum")
DRUM_RADIUS_M = 0.5 * float(_DRUM["diameter_m"])

# The reel hub and outer radii, likewise from the one declaration in
# `transport_model`, where they are marked TYPICAL rather than specified.
_SUPPLY = next(part for part in transport_model.TRANSPORT
               if part["name"] == "supply reel")
REEL_HUB_RADIUS_M, REEL_OUTER_RADIUS_M = _SUPPLY["radius_range_m"]

# The helical wrap. VHS's M-load carries the tape a little beyond half the
# drum so that one head is always in contact; the figure usually stated is
# 186 degrees. ASSUMED to that precision: it enters only through the arc
# length and the Euler capstan factor below, so a degree either way moves the
# tension by well under one per cent and moves no direction count at all.
WRAP_ANGLE_DEG = 186.0

# Coefficient of friction between the tape's back surface and the drum.
# ASSUMED. It appears only in the Euler capstan relation, where it multiplies
# the wrap angle; a factor of two error there changes the tension ratio
# across the wrap and therefore the amplitude of the wrap mechanism's
# signature, and nothing about its shape or its direction.
DRUM_FRICTION = 0.2

# Dynamic viscosity of air at 20 degrees Celsius and one atmosphere, CRC
# Handbook of Chemistry and Physics. The entrained film goes as this to the
# two-thirds power, so the ordinary spread of room temperature moves it by
# about one per cent.
AIR_VISCOSITY_PA_S = 1.81e-5

# THE FOIL BEARING. A flexible tape drawn over a cylinder at speed v under
# tension per unit width T' floats on a self-acting air film of nominal
# clearance
#
#     h = 0.643 R (6 eta v / T')^(2/3)
#
# The two-thirds exponent and the 0.643 coefficient are the classical
# foil-bearing result: Blok and van Rossum (1953), Baumeister, "Nominal
# clearance of the foil bearing", IBM Journal of Research and Development 7
# (1963) 153, and Eshel and Elrod, Journal of Basic Engineering 87 (1965) 831;
# reproduced in Mee and Daniel, Magnetic Recording Technology, second
# edition, chapter 7, and in Bhushan, Tribology and Mechanics of Magnetic
# Storage Devices. THE EXPONENT IS THE POINT: the film scales with the
# two-thirds power of velocity and the minus two-thirds power of tension per
# unit width.
FOIL_BEARING_COEFFICIENT = 0.643
ENTRAINMENT_VELOCITY_EXPONENT = 2.0 / 3.0

# The head is not the drum. The film above is the clearance over a cylinder
# of the drum's radius, which at this speed and tension comes to about
# 15 micrometres - a thousand times more separation than a video head can
# tolerate - and that is precisely why a video head PROTRUDES from the drum
# surface: it has to pierce the film. The clearance that matters is the one
# over the protruding tip, whose radius of curvature is far smaller than the
# drum's. ASSUMED at 0.2 mm, which puts the entrained film at the tenth of a
# micrometre that head fits on this project actually return. What it changes:
# the entrained spacing scales linearly with it, so it sets that mechanism's
# share of the total clearance and nothing else - no shape, no direction.
HEAD_TIP_RADIUS_M = 0.2e-3

# Biaxially oriented polyester base film: Young's modulus and Poisson's
# ratio. ASSUMED at values typical for the material, since no tape
# specification in this project's sources states them. What they change: the
# modulus divides the strain, so it sets the SIZE of every phase signature
# relative to its amplitude signature, and the ratio between the two parts is
# what makes the strain a second direction. A factor of two error in the
# modulus therefore does move the frequency-axis count - it is reported below
# as the sensitivity of that count to the modulus, which is measured rather
# than assumed away.
BASE_MODULUS_PA = 4.0e9
BASE_POISSON = 0.3

# RMS asperity height of the coating's surface. ASSUMED at 15 nanometres,
# which is the ordinary order for a calendered ferric coating. It scales the
# pressure-dependent part of the clearance; the reference pressure below is
# then calibrated so that the middle of the specified tension range returns
# the repository's own typical spacing, rather than a second free constant
# being introduced.
ASPERITY_RMS_M = 15.0e-9


def _grid(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).ravel()


def _tension_mid() -> float:
    return float(np.mean(TENSION_RANGE_N))


# --------------------------------------------------------------------------
# The closed forms, one per mechanism
# --------------------------------------------------------------------------


def tension_per_width(tension_n: float,
                      tape_width_m: float = TAPE_WIDTH_M) -> float:
    """Tension per unit width, in newtons per metre.

    Every foil-bearing and membrane result is written in this quantity rather
    than in the tension itself, because a wider tape under the same tension
    is a slacker tape.
    """
    return float(tension_n) / float(tape_width_m)


def contact_pressure_pa(tension_n: float, radius_m: float = DRUM_RADIUS_M,
                        tape_width_m: float = TAPE_WIDTH_M) -> float:
    """The pressure a tensioned tape presses onto a cylinder with.

    Elementary belt-over-pulley statics: an element of tape subtending
    `d theta` on a cylinder of radius `R` is pulled inward by `T d theta`
    over an area `R w d theta`, so `p = T / (R w)`. The wrap angle does not
    appear - it sets how MUCH tape is in contact, not how hard each part of
    it presses.
    """
    return float(tension_n) / (float(radius_m) * float(tape_width_m))


def euler_tension_ratio(wrap_angle_deg: float = WRAP_ANGLE_DEG,
                        friction: float = DRUM_FRICTION) -> float:
    """How much the wrap multiplies the tension between entry and exit.

    The Euler-Eytelwein capstan relation, `T_out = T_in exp(mu theta)`, which
    is why the wrap angle is a mechanical parameter at all rather than just a
    statement of how much tape is touching. At the format's 186 degrees and
    an assumed friction of 0.2 the ratio is 1.91, so the tape leaves the drum
    under nearly twice the tension it entered under and the mean tension over
    the wrap is not the tension either reel is holding.
    """
    return float(np.exp(float(friction) * np.radians(float(wrap_angle_deg))))


def entrainment_spacing_m(velocity_m_s: float, tension_n: float,
                          radius_m: float = HEAD_TIP_RADIUS_M,
                          tape_width_m: float = TAPE_WIDTH_M,
                          viscosity_pa_s: float = AIR_VISCOSITY_PA_S
                          ) -> float:
    """The self-acting air film the moving tape drags into the interface.

    The foil bearing, `h = 0.643 R (6 eta v / T')^(2/3)`, cited at
    `FOIL_BEARING_COEFFICIENT` above. THE VELOCITY EXPONENT IS TWO THIRDS and
    the tension-per-width exponent is minus two thirds.

    AND ON THIS FORMAT THE EXPONENT CANNOT BE MEASURED. The velocity that
    enters is the head-to-tape writing speed, which the format fixes at
    5.80 m/s for NTSC VHS at every tape speed - SP, LP and EP change the
    LINEAR speed and leave the drum alone. So the entrained film is a
    constant on any one machine, indistinguishable from the head's own
    protrusion and from the coating's roughness, and the exponent is a
    property of the model rather than of anything this project can weigh. It
    is carried because it is what would separate the film if a machine with a
    different drum speed were ever compared, and because getting it wrong
    would misattribute the whole clearance budget.
    """
    reduced = (6.0 * float(viscosity_pa_s) * float(velocity_m_s)
               / max(tension_per_width(tension_n, tape_width_m), 1e-30))
    return (FOIL_BEARING_COEFFICIENT * float(radius_m)
            * reduced ** ENTRAINMENT_VELOCITY_EXPONENT)


def asperity_reference_pressure_pa(nominal_spacing_m: Optional[float] = None,
                                   roughness_m: float = ASPERITY_RMS_M,
                                   tension_n: Optional[float] = None
                                   ) -> float:
    """The reference pressure the contact law is anchored at.

    Greenwood and Williamson, Proceedings of the Royal Society A 295 (1966)
    300: for an exponential distribution of asperity heights the real contact
    area, and with it the load, grows as `exp(-d / sigma)`, so the mean
    separation falls only LOGARITHMICALLY with pressure,

        d(p) = sigma ln(p_ref / p).

    That leaves one constant, and rather than inventing it this anchors it so
    that the middle of the specified tension range returns the separation the
    head model already carries as typical. The law's SHAPE - logarithmic in
    pressure, and therefore in tension - is the published part; the anchor is
    a calibration to this repository's own figure and is stated as such.
    """
    if nominal_spacing_m is None:
        nominal_spacing_m = head_model.TYPICAL["spacing_m"]
    if tension_n is None:
        tension_n = _tension_mid()
    pressure = contact_pressure_pa(tension_n)
    return float(pressure * np.exp(float(nominal_spacing_m)
                                   / max(float(roughness_m), 1e-30)))


def contact_spacing_m(tension_n: float, radius_m: float = DRUM_RADIUS_M,
                      roughness_m: float = ASPERITY_RMS_M,
                      reference_pressure_pa: Optional[float] = None) -> float:
    """The separation the asperities hold open at this contact pressure.

    `d = sigma ln(p_ref / p)`, the Greenwood-Williamson exponential-height
    result cited above, clipped at zero because a negative separation is a
    statement that the surfaces have gone into full contact and the model has
    left its range rather than a physical quantity.
    """
    if reference_pressure_pa is None:
        reference_pressure_pa = asperity_reference_pressure_pa(
            roughness_m=roughness_m)
    pressure = contact_pressure_pa(tension_n, radius_m)
    return float(max(float(roughness_m)
                     * np.log(max(reference_pressure_pa, 1e-30)
                              / max(pressure, 1e-30)), 0.0))


def flexural_rigidity(modulus_pa: float = BASE_MODULUS_PA,
                      thickness_m: float = TAPE_THICKNESS_M,
                      poisson: float = BASE_POISSON) -> float:
    """The tape's bending stiffness per unit width, `E t^3 / 12 (1 - nu^2)`.

    Classical plate theory. It is a strong function of thickness - the cube -
    which is why a thicker long-play tape conforms differently to the same
    guides.
    """
    return float(modulus_pa) * float(thickness_m) ** 3 / (
        12.0 * (1.0 - float(poisson) ** 2))


def bending_length_m(tension_n: float, tape_width_m: float = TAPE_WIDTH_M,
                     **rigidity) -> float:
    """How far a tensioned tape carries a bend before the tension flattens it.

    A tensioned plate obeys `D w y'''' - T y'' = 0`, whose decaying solution
    is `exp(-s / l)` with `l = sqrt(D w / T)`. That length is the whole of
    what tape stiffness contributes to the interface: a guide-induced
    deviation nearer the head than `l` is still standing when the tape
    reaches the gap, and one further away has been pulled flat.

    At the middle of the specified tension range and an assumed polyester
    base the length is 0.29 mm, which is the same order as the head's own
    contact arc - the reason guide geometry reaches the head at all, and the
    reason a guide a centimetre away does not.
    """
    return float(np.sqrt(flexural_rigidity(**rigidity) * float(tape_width_m)
                         / max(float(tension_n), 1e-30)))


def stiffness_spacing_m(deviation_m: float, distance_m: float,
                        tension_n: float, **kwargs) -> float:
    """The standoff a guide-induced deviation still holds at the head.

    `a exp(-s / l)` with `l` from `bending_length_m`: the deviation the guide
    imposes, decayed over the free span between the guide and the head. A
    stiffer tape or a slacker path carries it further.
    """
    length = bending_length_m(tension_n, **kwargs)
    return float(abs(float(deviation_m))
                 * np.exp(-abs(float(distance_m)) / max(length, 1e-30)))


def pack_radius_m(fraction_through: float,
                  hub_radius_m: float = REEL_HUB_RADIUS_M,
                  outer_radius_m: float = REEL_OUTER_RADIUS_M) -> np.ndarray:
    """The supply pack's radius, a given fraction of the way through a pass.

    Constant-volume winding: the wound cross-section is proportional to
    `r^2 - r_hub^2`, so a pack that starts full and empties linearly in time
    has `r(x) = sqrt(r_hub^2 + (r_out^2 - r_hub^2)(1 - x))`. Geometry, not a
    specification, but the hub and outer radii are.
    """
    x = np.clip(np.asarray(fraction_through, dtype=np.float64), 0.0, 1.0)
    hub, outer = float(hub_radius_m), float(outer_radius_m)
    return np.sqrt(hub ** 2 + (outer ** 2 - hub ** 2) * (1.0 - x))


def tension_from_pack(fraction_through, holdback_torque_n_m: Optional[float] = None,
                      regulated: bool = True, **kwargs) -> np.ndarray:
    """The tension the supply pack produces through a pass.

    A constant-torque holdback gives `T = Gamma / r`, so the tension RISES as
    the pack empties and its radius falls.

    AND THAT LAW IS REFUTED BY THE SPECIFICATION ITSELF, which is worth
    stating rather than hiding. Set so the middle of the pack sits in the
    middle of the specified 0.30 to 0.45 newtons, a constant-torque brake
    delivers 0.274 newtons at the full end and 1.029 at the hub - two and a
    half times the specified maximum. A transport that met the specification
    at both ends of a tape therefore cannot be leaving the tension to a
    constant-torque brake; it REGULATES, which is what the tension arm on
    every deck of this format is for. So the pack radius does not sweep the
    tension over a factor of four through a pass. It leaves a RESIDUAL trend
    inside the regulated range, and that residual is what the time axis sees.

    `regulated` therefore defaults to true and holds the result inside the
    specified range. The unregulated law is kept because it is the physics
    the regulator is working against, and because its refutation is the
    evidence that the regulator exists.
    """
    radius = pack_radius_m(fraction_through, **kwargs)
    if holdback_torque_n_m is None:
        holdback_torque_n_m = _tension_mid() * float(
            np.atleast_1d(pack_radius_m(0.5, **kwargs))[0])
    tension = float(holdback_torque_n_m) / np.maximum(radius, 1e-9)
    if regulated:
        tension = np.clip(tension, *TENSION_RANGE_N)
    return tension


def strain(tension_n: float, tape_width_m: float = TAPE_WIDTH_M,
           thickness_m: float = TAPE_THICKNESS_M,
           modulus_pa: float = BASE_MODULUS_PA) -> float:
    """The tape's elastic strain under tension: `T / (E w t)`.

    Hooke's law on the tape's cross-section. THIS IS THE ONLY WAY A PATH
    PARAMETER REACHES THE PHASE. A strained tape has all its recorded
    wavelengths stretched, so a tape recorded at one tension and replayed at
    another comes back at a slightly wrong rate; the amplitude side of the
    path is a clearance and this is a time base, and they are the two kinds
    the family contains.

    At the middle of the specified tension range the strain is 3.9e-4, which
    over one NTSC line is 24 nanoseconds - well inside what the time base
    correction handles, and well outside what it can be assumed to be zero.
    """
    area = float(tape_width_m) * float(thickness_m)
    return float(tension_n) / (float(modulus_pa) * max(area, 1e-30))


# --------------------------------------------------------------------------
# The seven mechanisms, declared
# --------------------------------------------------------------------------

# NTSC line period, from the line rate the standard fixes. A strain is a
# DILATION and not a delay until an interval is named, so this is the
# interval every phase signature below is referred to: the displacement a
# fractional rate error accumulates across one line.
LINE_PERIOD_S = 1.0 / 15734.264

# Each mechanism carries what it does to the clearance, what it does to the
# tension (and so to the strain and the phase), the rate at which it varies,
# and what is confounded with it. `rate` is a key into
# `transport_model.rotation_rates`, or one of the two shapes that are not a
# rotation: "trend" for something that moves monotonically through a pass,
# "parity" for something that alternates with the head, and None for
# something that does not vary at all on the scale of a recording.
MECHANISMS: List[Dict[str, object]] = [
    {
        "name": "tape tension",
        "acts_on": "clearance and tension",
        "law": "d = sigma ln(p_ref / p) with p = T / (R w); "
               "Greenwood-Williamson exponential asperity heights",
        "spec": "tension 0.30-0.45 N (JVC VTG82063)",
        "rate": "supply reel",
        "tension_bearing": True,
        "confounded_with": ("pack radius",),
        "note": "the pack's radius is what MOVES the tension, so the two "
                "share a mechanism and differ only in the rate they move at",
    },
    {
        "name": "wrap angle and penetration",
        "acts_on": "clearance and tension",
        "law": "the Euler-Eytelwein capstan relation exp(mu theta) sets the "
               "tension across the wrap; the pressure that follows sets the "
               "clearance by the same contact law",
        "spec": "the format's helical wrap, about 186 degrees",
        "rate": None,
        "tension_bearing": True,
        "confounded_with": ("tape tension",),
        "note": "static geometry: it is set when the deck is threaded and "
                "does not move again, so it has no time signature at all",
    },
    {
        "name": "guide geometry and tape stiffness",
        "acts_on": "clearance",
        "law": "a exp(-s / l) with l = sqrt(D w / T), the decay length of a "
               "bend on a tensioned plate",
        "spec": "guide diameters from transport_model; the base film's "
                "modulus is ASSUMED",
        "rate": "entry guide",
        "tension_bearing": False,
        "confounded_with": (),
        "note": "the one mechanism whose rate is a guide's own rotation, "
                "which is why it separates from everything else in time",
    },
    {
        "name": "air entrainment",
        "acts_on": "clearance",
        "law": "h = 0.643 R (6 eta v / T')^(2/3), the foil bearing",
        "spec": "Baumeister 1963; Eshel and Elrod 1965",
        "rate": None,
        "tension_bearing": False,
        "confounded_with": ("head protrusion", "head wear"),
        "note": "the velocity is the writing speed, which the format fixes "
                "at 5.80 m/s, so on one machine the film is a CONSTANT and "
                "is indistinguishable from any other constant clearance",
    },
    {
        "name": "pack radius",
        "acts_on": "clearance and tension",
        "law": "r(x) = sqrt(r_hub^2 + (r_out^2 - r_hub^2)(1 - x)) and "
               "T = Gamma / r",
        "spec": "reel hub and outer radii (JVC VTG82063; carried in "
                "transport_model)",
        "rate": "trend",
        "tension_bearing": True,
        "confounded_with": ("head wear",),
        "note": "a monotone trend through a pass, so it is confounded with "
                "anything else monotone - and head wear is monotone",
    },
    {
        "name": "head protrusion",
        "acts_on": "clearance",
        "law": "spacing_from_protrusion: more protrusion, less separation",
        "spec": "head_model.PROTRUSION_TO_SPACING",
        "rate": "parity",
        "tension_bearing": False,
        "confounded_with": ("air entrainment", "head wear"),
        "note": "the two heads protrude differently, so the DIFFERENCE "
                "alternates with the head and is separable; the common part "
                "is a constant and is not",
    },
    {
        "name": "head wear",
        "acts_on": "clearance",
        "law": "protrusion falling with head hours, then the same "
               "protrusion-to-spacing relation",
        "spec": "no specification states a wear rate; ASSUMED monotone",
        "rate": "trend",
        "tension_bearing": False,
        "confounded_with": ("pack radius", "air entrainment"),
        "note": "monotone over thousands of hours, which is a CONSTANT over "
                "any one recording; it is separable only across a tape's "
                "lifetime and this project has no such series",
    },
]


def mechanisms() -> List[Dict[str, object]]:
    return [dict(entry) for entry in MECHANISMS]


def spacing_budget(tension_n: Optional[float] = None,
                   writing_speed_m_s: float = 5.80,
                   protrusion_m: float = 0.0,
                   guide_deviation_m: float = 0.5e-6,
                   guide_distance_m: float = 1.0e-3) -> Dict[str, float]:
    """Every mechanism's contribution to the clearance, in metres.

    The budget is what says which mechanism is worth chasing. Measured at the
    middle of the specified tension range, 0.375 newtons, and the format's
    5.80 m/s writing speed, in micrometres:

        air entrainment                    0.0987
        tape tension, the asperity contact  0.0500
        guide geometry and stiffness        0.0161
        wrap angle and penetration         -0.0097
        head protrusion                     0        (the reference head)
        total                               0.1550

    So the clearance is an aerodynamic problem and a contact problem in
    roughly equal measure, at a ratio of 1.97 to one, and the mechanical
    terms are a tenth of it. The wrap's contribution is NEGATIVE because more
    tension across the wrap is more pressure and less separation, which is
    the sign the contact law gives and not an error.
    """
    if tension_n is None:
        tension_n = _tension_mid()
    entrained = entrainment_spacing_m(writing_speed_m_s, tension_n)
    contact = contact_spacing_m(tension_n)
    stiffness = stiffness_spacing_m(guide_deviation_m, guide_distance_m,
                                    tension_n)
    protrusion = head_model.spacing_from_protrusion(protrusion_m)
    wrapped = contact_spacing_m(tension_n * euler_tension_ratio()) - contact
    total = entrained + contact + stiffness + protrusion + wrapped
    return {
        "air entrainment": entrained,
        "tape tension": contact,
        "wrap angle and penetration": wrapped,
        "guide geometry and tape stiffness": stiffness,
        "head protrusion": protrusion,
        "total_m": total,
        "tension_n": float(tension_n),
        "pressure_pa": contact_pressure_pa(tension_n),
        "strain": strain(tension_n),
        "bending_length_m": bending_length_m(tension_n),
    }


# --------------------------------------------------------------------------
# The signatures
# --------------------------------------------------------------------------


def wallace_log(frequency_hz, spacing_m: float,
                writing_speed_m_s: float = 5.80) -> np.ndarray:
    """The separation loss in nepers, `-2 pi d f / v`.

    Wallace, "The reproduction of magnetically recorded signals", Bell System
    Technical Journal 30 (1951) 1145: the reproduced flux falls as
    `exp(-2 pi d / lambda)`. THIS IS THE WHOLE OF THE FREQUENCY AXIS FOR THIS
    COMPONENT, and it is a straight line through the origin whose only free
    quantity is the slope - which is why seven mechanisms give one direction.
    """
    return (-2.0 * np.pi * float(spacing_m) * _grid(frequency_hz)
            / float(writing_speed_m_s))


def strain_delay_s(tension_change_n: float,
                   interval_s: float = LINE_PERIOD_S,
                   **kwargs) -> float:
    """The timing displacement a tension change produces over one interval.

    A strain dilates the tape, so a tension differing by `dT` between record
    and replay returns the signal at a fractional rate error equal to the
    strain difference. A rate error is not a delay until an interval is
    named; this names one line, which is the interval the time base
    correction works over.
    """
    return float(strain(tension_change_n, **kwargs)) * float(interval_s)


def path_signature(frequency_hz, spacing_m: float = 0.0,
                   delay_s: float = 0.0,
                   writing_speed_m_s: float = 5.80) -> np.ndarray:
    """One mechanism's complex signature on the frequency axis.

    The real part is what it does to amplitude - Wallace's separation loss
    from the clearance it sets - and the imaginary part what it does to
    phase, which is the delay its tension change produces through the strain
    and is exactly zero for a mechanism that does not touch the tension.

    The delay is passed in rather than derived here, because it comes from
    the base film's modulus, which no specification in this project's sources
    states and which is therefore assumed; keeping it an argument is what
    lets `modulus_sensitivity` measure how much of the answer rests on it.

    SUBTRACTABLE BY CONSTRUCTION. The magnitude is `exp(-2 pi d f / v)`,
    strictly positive at every frequency for every finite clearance, so its
    logarithm is finite everywhere. No mask, no bare power term, and nothing
    that reaches zero: the bounded form is the exponential itself.
    """
    f = _grid(frequency_hz)
    magnitude = np.exp(wallace_log(f, spacing_m, writing_speed_m_s))
    return (magnitude * np.exp(-2j * np.pi * f * float(delay_s))
            ).astype(np.complex128)


def clearance_signature(frequency_hz, spacing_m: float = 0.155e-6,
                        writing_speed_m_s: float = 5.80) -> np.ndarray:
    """THE FIRST OF THE TWO ENTRIES: what the path does to amplitude.

    The whole seven-member family's amplitude side, which is one direction,
    entered once. The default clearance is what `spacing_budget` returns at
    the middle of the specified tension range, 0.155 micrometres; only the
    shape matters to the fit, since a fitted size is what the collapse is
    for.

    THE RECORD-SIDE AND PLAYBACK-SIDE CLEARANCES CANNOT BE SEPARATED and
    this signature is their sum. A separation loss applies once when the
    signal is written and again when it is read, and both are
    `exp(-2 pi d f / v)`, so the exponents add and no measurement on one tape
    can say which machine contributed which share. That is the same
    confound `tape_model` records between the coating's roughness and the
    head's own clearance, one stage further out.
    """
    return path_signature(frequency_hz, spacing_m=spacing_m,
                          writing_speed_m_s=writing_speed_m_s)


def strain_signature(frequency_hz, tension_change_n: Optional[float] = None,
                     interval_s: float = LINE_PERIOD_S) -> np.ndarray:
    """THE SECOND: what the path does to phase.

    The tension-bearing mechanisms strain the tape and a strained tape plays
    back at the wrong rate. Entered as a pure delay, because that is what it
    is - the amplitude part of the same mechanisms is already carried by
    `clearance_signature` and entering it twice would be the second copy of a
    direction that the emphasis curves and the record level both showed
    lowers the count.

    THIS IS A DIFFERENCE BETWEEN TWO STAGES, not a stage. The wavelengths are
    fixed in the tape at the tension the recorder held, and read back at the
    tension the player holds, so only `T_play - T_rec` is observable and
    neither tension alone is. It is declared at the record position because
    that is where the quantity is set and where an inversion has to put it
    back; the player's share is the same signature with the opposite sign.

    The default tension change is the full width of the specified range,
    0.30 to 0.45 newtons - the largest difference the specification permits
    between the machine that recorded the tape and the one replaying it -
    which is 9.9 nanoseconds over one NTSC line.
    """
    if tension_change_n is None:
        tension_change_n = float(TENSION_RANGE_N[1] - TENSION_RANGE_N[0])
    return path_signature(frequency_hz, spacing_m=0.0,
                          delay_s=strain_delay_s(tension_change_n,
                                                 interval_s))


# The two entries this component defines, and where each acts. Both are
# needed by `interference.ordered_key`, which refuses an entry with no
# declared position, so the pair is stated here in the form the coordinator
# adds to `interference.COMPONENT_ORDER`.
#
# The chain there runs: transmission 1-4, the recorder 10-11, the tape and
# head 20-23, the capture 30. The tape path sits in two places, and a
# record-side parameter and a playback-side one are not the same entry:
#
#   12  the RECORD-side head-to-tape interface. The record current is set at
#       11, and the clearance and the tape's strain act on it before the
#       medium holds anything.
#   19  the PLAYBACK-side interface, immediately before the head's own
#       losses at 20-23, because the mechanics decide the separation those
#       losses are then evaluated at.
#
# Only sums and differences across the pair are observable - the clearances
# add and the strains subtract - which is why `signatures` emits one entry
# per KIND rather than one per stage.
COMPONENT_POSITIONS: Dict[str, int] = {
    "tape path clearance (record)": 12,
    "tape path strain (record)": 12,
    "tape path clearance (playback)": 19,
    "tape path strain (playback)": 19,
}


def signatures(frequency_hz, writing_speed_m_s: float = 5.80,
               tension_n: Optional[float] = None,
               **kwargs) -> Dict[str, np.ndarray]:
    """THE TWO ENTRIES THIS COMPONENT DEFINES - and it does NOT recommend
    that either be added to the key.

    Not seven, because the seven-member family spans one clearance direction
    and one strain direction, and an entry earns its place by the direction
    it adds rather than by being a mechanism. Entering all seven lowers the
    modelled key's effective count from 7.91 to 5.59, which is the S-VHS
    emphasis failure arriving again.

    AND NOT TWO EITHER. `key_impact` measures what happens when these are
    added to `interference.signatures()` and every arrangement makes the key
    worse: the clearance takes it to 7.58 at condition 110, the strain to
    7.76 at condition 40, both together to 7.56. The clearance is 0.972
    coherent with the particle-noise entry and the strain 0.982 coherent with
    the group-delay entry, so both are second copies of directions already in
    the span. The honest recommendation is that the key take neither, and
    that this component be used on the TIME axis, where it does span four
    directions and where the key holds nothing.

    The pair is defined and named all the same, because a component whose
    entries have no declared position and no subtractable form cannot be
    checked against the contract at all, and because a caller with an
    independent measurement of the tension - a service manual figure, a
    tension gauge - can subtract this directly rather than fitting it.
    """
    budget = spacing_budget(tension_n, writing_speed_m_s, **kwargs)
    return {
        "tape path clearance (playback)": clearance_signature(
            frequency_hz, max(budget["total_m"], 1e-12), writing_speed_m_s),
        "tape path strain (record)": strain_signature(frequency_hz),
    }


def all_member_signatures(frequency_hz, tension_n: Optional[float] = None,
                          writing_speed_m_s: float = 5.80,
                          modulus_pa: float = BASE_MODULUS_PA,
                          **kwargs) -> Dict[str, np.ndarray]:
    """All seven members separately - for MEASURING the collapse, not for
    the key.

    This is the ensemble the headline numbers are computed over. It is kept
    as a function because the claim "they are collinear" has to be checkable
    rather than asserted, and because a later format with a different drum
    speed would make the entrainment member move where here it cannot.
    """
    if tension_n is None:
        tension_n = _tension_mid()
    budget = spacing_budget(tension_n, writing_speed_m_s, **kwargs)
    # each mechanism perturbed by its own share of the clearance, and by the
    # tension change it carries if it is tension-bearing
    span = float(TENSION_RANGE_N[1] - TENSION_RANGE_N[0])
    delay = strain_delay_s(span, modulus_pa=modulus_pa)
    out: Dict[str, np.ndarray] = {}
    for entry in MECHANISMS:
        name = str(entry["name"])
        spacing = float(budget.get(name, 0.0))
        if not spacing:
            # head protrusion is zero at the reference head and head wear
            # acts by removing protrusion, so neither has a budget line of
            # its own; give them a tenth of the total so the member is a
            # shape to be judged rather than a null vector. Only the shape is
            # measured here, and the shape does not depend on the size.
            spacing = float(budget["total_m"]) * 0.1
        out[name] = path_signature(
            frequency_hz, spacing_m=abs(spacing),
            delay_s=delay if entry["tension_bearing"] else 0.0,
            writing_speed_m_s=writing_speed_m_s)
    return out


# --------------------------------------------------------------------------
# The measure: how many directions, on each axis
# --------------------------------------------------------------------------


def _shape_of(value: np.ndarray) -> np.ndarray:
    """A signature reduced to the shape the participation ratio judges.

    The log magnitude with its mean removed, because a constant is a level
    and not a shape, and the unwrapped phase likewise. The same reduction
    `interference.distinguishable` uses, so the counts are comparable.
    """
    value = np.asarray(value)
    magnitude = np.log(np.maximum(np.abs(value), 1e-12))
    phase = (np.unwrap(np.angle(value)) if np.iscomplexobj(value)
             else np.zeros_like(magnitude))
    return (magnitude - magnitude.mean()) + 1j * (phase - phase.mean())


def participation(shapes: Sequence[np.ndarray]) -> Dict[str, object]:
    """The participation ratio of the singular values, and its conditioning.

    Real and imaginary parts stacked into one real vector before
    normalisation, which is `real_parameters=True` written out: each of these
    is a real physical parameter's signature, so a gain and a delay of the
    same shape must count as two mechanisms and not one.

    A ROW WHOSE NORM IS ZERO IS LEFT AT ZERO rather than normalised. This is
    the error `magnetic.py` records - normalising a near-zero vector measures
    rounding and manufactures directions out of it - and here it is not
    hypothetical, because two of the seven mechanisms have no time signature
    at all and their rows are exactly zero on that axis.
    """
    rows, alive = [], 0
    for value in shapes:
        vector = np.concatenate([np.real(value).ravel(),
                                 np.imag(value).ravel()])
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            rows.append(vector / norm)
            alive += 1
        else:
            rows.append(np.zeros_like(vector))
    if not rows or not alive:
        return {"effective": 0.0, "count": len(rows), "alive": 0,
                "singular_values": np.zeros(0), "condition": 0.0,
                "why": "no signature has any shape on this axis"}
    stack = np.asarray(rows, dtype=np.float64)
    singular = np.linalg.svd(stack, compute_uv=False)
    share = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    gram = stack @ stack.T
    off = np.abs(gram - np.diag(np.diag(gram)))
    worst = (int(np.argmax(off)) if off.size else 0)
    # A CONDITION NUMBER OVER A SINGULAR VALUE THAT IS ZERO IS NOT A NUMBER
    # ABOUT TAPE. An exactly collinear family has structural zeros, and
    # dividing by one returns the reciprocal of the machine epsilon - which
    # would be reported as though it were a conditioning measurement. So the
    # numerical rank is taken first, at the standard threshold, and the
    # conditioning reported is that of the directions that actually exist;
    # the raw ratio is kept beside it and is an ARTEFACT wherever the rank is
    # short of the count.
    floor = float(singular[0]) * max(stack.shape) * float(np.finfo(np.float64).eps)
    rank = int(np.count_nonzero(singular > floor))
    resolved = float(singular[max(rank - 1, 0)])
    return {
        "effective": float(1.0 / np.sum(share ** 2)),
        "count": len(rows),
        "alive": alive,
        "rank": rank,
        "singular_values": singular,
        "condition": float(singular[0] / max(resolved, 1e-30)),
        "condition_raw": float(singular[0] / max(singular[-1], 1e-30)),
        "worst_pair_coherence": float(off.max()) if off.size else 0.0,
        "worst_pair": (worst // len(rows), worst % len(rows)),
    }


def frequency_axis(frequency_hz, phase: bool = True,
                   **kwargs) -> Dict[str, object]:
    """How many directions the seven members span on the frequency axis.

    With `phase` false this is the amplitude-only question and the answer is
    exactly one, by algebra: every member's log magnitude is `-2 pi d f / v`
    and they differ only in `d`. With it true the tension-bearing members
    carry a delay as well and a second direction appears, because a gain and
    a delay of the same shape are two mechanisms under the stacked real inner
    product and one under the Hermitian one.

    Measured on the VHS luma band, 1 to 7 MHz over 1024 points: 1.000000 of 7
    at rank 1 without the phase, 1.734 of 7 at rank 2 with it.
    """
    entries = all_member_signatures(frequency_hz, **kwargs)
    names = list(entries)
    shapes = []
    for name in names:
        shape = _shape_of(entries[name])
        shapes.append(shape if phase else np.real(shape).astype(np.complex128))
    out = participation(shapes)
    out["names"] = names
    first, second = out.get("worst_pair", (0, 0))
    out["worst_named_pair"] = (names[first], names[second])
    out["why"] = ("every mechanism reaches the response through the same "
                  "separation loss, so the amplitude side is one direction; "
                  "the tension-bearing members also strain the tape, and a "
                  "strain is a delay rather than a gain")
    return out


def time_signatures(fields: int = 512, field_rate_hz: float = 59.94,
                    linear_speed_m_s: float = 0.03335
                    ) -> Dict[str, np.ndarray]:
    """Each mechanism's variation over a run of fields, at its own rate.

    The rates are not fitted and not searched for: `transport_model` states
    every one of them from the part's diameter and the tape speed before
    anything is measured, and a mechanism is attached to the part that drives
    it. A parameter that varies at a known rate is separable from one that
    does not even when their frequency shapes are identical, which is the
    only reason this family separates at all.

    Two mechanisms return an exactly zero series and that is the honest
    answer rather than a defect: the wrap and the penetration are set when the
    deck is threaded, and the entrained film is fixed by a drum speed the
    format fixes at 5.80 m/s whatever the tape speed. Nothing that does not
    move can be told from anything else that does not move.

    THE HEAD'S PROTRUSION IS THE INTERESTING CASE. Its COMMON part is a
    constant and would be as inert as those two; what is modelled is the
    DIFFERENCE between the two heads, which alternates field by field because
    the heads alternate, and that alternation is a shape no other mechanism in
    the transport has. It is the one member of this family that a single field
    pair can see.
    """
    rates = {str(entry["name"]): entry
             for entry in transport_model.rotation_rates(linear_speed_m_s,
                                                         field_rate_hz)}
    n = int(fields)
    index = np.arange(n, dtype=np.float64)
    seconds = index / float(field_rate_hz)
    out: Dict[str, np.ndarray] = {}
    for entry in MECHANISMS:
        name = str(entry["name"])
        rate = entry["rate"]
        if rate is None:
            series = np.zeros(n)
        elif rate == "trend":
            # a monotone pass through the tape, centred so it is a shape and
            # not a level
            series = np.linspace(-1.0, 1.0, n)
        elif rate == "parity":
            # the two heads alternate field by field: the one signature in
            # this set that is not a sinusoid and not a trend
            series = (-1.0) ** index
        else:
            part = rates.get(str(rate))
            hz = float(part["rate_hz"]) if part else 0.0
            series = np.cos(2.0 * np.pi * hz * seconds)
            series = series - series.mean()
        out[name] = series
    return out


def time_axis(fields: int = 512, field_rate_hz: float = 59.94,
              linear_speed_m_s: float = 0.03335,
              reference_hz: float = 3.9e6,
              writing_speed_m_s: float = 5.80,
              amplitude_m: float = 0.01e-6,
              tension_amplitude_n: float = 0.05,
              modulus_pa: float = BASE_MODULUS_PA) -> Dict[str, object]:
    """How many directions the same seven span on the TIME axis.

    Each mechanism's clearance and tension modulation are carried onto a
    reference frequency - the carrier - so that the series is what the RF
    envelope and the time base would actually show: the real part a level
    modulation through Wallace's law, the imaginary part a timing modulation
    through the strain.

    Measured over 512 fields at the NTSC field rate: 4.597 of 7 at rank 5 and
    condition 1.66. Five rows are alive, because two mechanisms do not move
    at all; the pack radius and head wear share a monotone shape and are
    split only by the pack straining the tape where wear does not, which is
    the difference between 4.60 and the 3.56 the shapes alone give.
    """
    series = time_signatures(fields, field_rate_hz, linear_speed_m_s)
    scale = -2.0 * np.pi * float(reference_hz) / float(writing_speed_m_s)
    shapes, names = [], []
    for entry in MECHANISMS:
        name = str(entry["name"])
        deviation = np.asarray(series[name], dtype=np.float64)
        real = scale * float(amplitude_m) * deviation
        if entry["tension_bearing"]:
            delay = strain_delay_s(float(tension_amplitude_n),
                                   modulus_pa=modulus_pa)
            imaginary = -2.0 * np.pi * float(reference_hz) * delay * deviation
        else:
            imaginary = np.zeros_like(deviation)
        shapes.append(real + 1j * imaginary)
        names.append(name)
    out = participation(shapes)
    out["names"] = names
    out["inert"] = [str(e["name"]) for e in MECHANISMS if e["rate"] is None]
    first, second = out.get("worst_pair", (0, 0))
    out["worst_named_pair"] = (names[first], names[second])
    out["why"] = ("a parameter that varies at a known rate is separable from "
                  "one that does not, even when their frequency shapes are "
                  "identical; what does not vary at all is separable from "
                  "nothing")
    return out


def joint_axis(frequency_hz, fields: int = 512, field_rate_hz: float = 59.94,
               linear_speed_m_s: float = 0.03335,
               writing_speed_m_s: float = 5.80,
               amplitude_m: float = 0.01e-6,
               tension_amplitude_n: float = 0.05,
               modulus_pa: float = BASE_MODULUS_PA) -> Dict[str, object]:
    """The two axes together, on the outer product grid.

    AND IT BUYS NOTHING OVER TIME ALONE, which is the point rather than a
    disappointment. Each mechanism's joint signature is the outer product of
    its own time shape with a frequency shape that is `proportional to f` for
    both the clearance and the delay, so it is the SAME frequency factor for
    all seven, and a factor common to every member of a family cannot
    separate any of them. The Gram matrix of the joint set is therefore the
    Gram matrix of the time set exactly, and the two counts must agree to
    numerical precision.

    THAT AGREEMENT IS ITSELF A CHECK. If the joint count ever exceeded the
    time count, the construction would have manufactured a direction from the
    grid rather than from the physics. Measured: 4.597196 against the time
    axis's 4.597196, agreeing to eleven decimal places.
    """
    f = _grid(frequency_hz)
    series = time_signatures(fields, field_rate_hz, linear_speed_m_s)
    shapes, names = [], []
    for entry in MECHANISMS:
        name = str(entry["name"])
        deviation = np.asarray(series[name], dtype=np.float64)
        # the clearance modulation, carried across the whole band
        real = np.outer(deviation, -2.0 * np.pi * float(amplitude_m) * f
                        / float(writing_speed_m_s)).ravel()
        if entry["tension_bearing"]:
            delay = strain_delay_s(float(tension_amplitude_n),
                                   modulus_pa=modulus_pa)
            imaginary = np.outer(deviation,
                                 -2.0 * np.pi * f * delay).ravel()
        else:
            imaginary = np.zeros_like(real)
        shapes.append(real + 1j * imaginary)
        names.append(name)
    out = participation(shapes)
    out["names"] = names
    out["why"] = ("the frequency factor is common to all seven, and a shared "
                  "factor separates nothing; the joint count must equal the "
                  "time count")
    return out


# --------------------------------------------------------------------------
# THE CONTROLS
# --------------------------------------------------------------------------


def separability_control(length: int = 1024) -> Dict[str, object]:
    """THE CONTROL, and it can fail in both directions.

    Two families go through `participation`, the same function every headline
    number above comes out of, and their answers are known independently and
    are DIFFERENT from one another:

    A family of pure Wallace separation losses at five different clearances
    must read exactly 1.00, because `-2 pi d f / v` at five values of `d` is
    one vector at five scales and normalisation removes the scale. That is
    algebra, not a measurement.

    A family of five echoes at delays spaced beyond the band's own resolution
    must read above 4.5 of 5, because `exp(-2 pi j f tau)` is a Fourier basis
    over delay - the repository's own measurement of eight such echoes is
    7.92 of 8 at condition 1.2.

    A construction that has collapsed everything by accident passes the first
    and fails the second. One that manufactures directions out of rounding -
    the failure `magnetic.py` records, where a control that should have read
    1.00 read 4.97 - fails the first. Only a correct one passes both.

    Measured: 1.000000 and 4.92.
    """
    f = np.linspace(1e6, 7e6, int(length))
    collinear = [_shape_of(np.exp(wallace_log(f, d)))
                 for d in (0.02e-6, 0.05e-6, 0.10e-6, 0.20e-6, 0.40e-6)]
    resolution = 1.0 / (f[-1] - f[0])
    orthogonal = [_shape_of(1.0 + 0.2 * np.exp(-2j * np.pi * f
                                               * resolution * k))
                  for k in (1, 2, 4, 8, 16)]
    one = participation(collinear)
    many = participation(orthogonal)
    passes = (abs(one["effective"] - 1.0) < 0.05
              and many["effective"] > 4.5)
    return {
        "collinear_effective": one["effective"],
        "collinear_expected": 1.0,
        "orthogonal_effective": many["effective"],
        "orthogonal_expected": "> 4.5 of 5",
        "passes": bool(passes),
        "why": ("one family is one direction by algebra and the other is a "
                "Fourier basis by construction; a measure that cannot tell "
                "them apart is not measuring anything"),
    }


def entrainment_exponent_control() -> Dict[str, object]:
    """THE SECOND CONTROL: the air film's power law, recovered numerically.

    The foil bearing's velocity exponent is two thirds and its
    tension-per-width exponent is minus two thirds. Both are recovered here
    by regressing the closed form over two decades of each variable, so a
    wrong power - the easiest thing in this module to mistype, and the one
    number the task asked for by name - is caught rather than propagated.

    Measured: 0.66666666666667 and -0.66666666666667, both within 4e-15 of
    two thirds.
    """
    speeds = np.geomspace(0.5, 50.0, 9)
    films = np.array([entrainment_spacing_m(v, _tension_mid()) for v in speeds])
    velocity_exponent = float(np.polyfit(np.log(speeds), np.log(films), 1)[0])
    tensions = np.geomspace(0.05, 5.0, 9)
    films = np.array([entrainment_spacing_m(5.80, t) for t in tensions])
    tension_exponent = float(np.polyfit(np.log(tensions), np.log(films), 1)[0])
    passes = (abs(velocity_exponent - 2.0 / 3.0) < 1e-9
              and abs(tension_exponent + 2.0 / 3.0) < 1e-9)
    return {
        "velocity_exponent": velocity_exponent,
        "velocity_expected": 2.0 / 3.0,
        "tension_exponent": tension_exponent,
        "tension_expected": -2.0 / 3.0,
        "passes": bool(passes),
        "why": ("Baumeister 1963 and Eshel and Elrod 1965 give the foil "
                "bearing's nominal clearance as 0.643 R (6 eta v / T')^(2/3)"),
    }


def modulus_sensitivity(frequency_hz,
                        moduli_pa: Sequence[float] = (1e9, 2e9, 4e9, 8e9,
                                                      16e9)) -> Dict[str, object]:
    """HOW MUCH OF THE ANSWER RESTS ON THE ONE ASSUMED NUMBER.

    The base film's modulus is assumed, and it divides the strain, so it sets
    how large every phase signature is beside its own amplitude signature -
    and the second frequency-axis direction exists precisely because that
    ratio is not zero. So the modulus is not a harmless scale: it is the one
    assumption that could move a direction count, and pretending otherwise
    would be exactly the failure this project keeps recording.

    Measured over a factor of sixteen in modulus, 1 to 16 GPa, on the VHS
    luma band: 1.936, 1.877, 1.734, 1.524, 1.294 - the count moves by half a
    direction and never reaches either 1 or 2, and the RANK is 2 at every one
    of them. So the SECOND DIRECTION IS REAL across every plausible modulus;
    a wrong modulus changes how well resolved it is, not whether it is there.
    The count should therefore be quoted as "between 1.29 and 1.94 of 7,
    1.734 at the assumed 4 GPa" rather than as a single figure.
    """
    counts = []
    for modulus in moduli_pa:
        fit = frequency_axis(frequency_hz, modulus_pa=float(modulus))
        counts.append({"modulus_pa": float(modulus),
                       "strain": strain(_tension_mid(),
                                        modulus_pa=float(modulus)),
                       "effective": float(fit["effective"]),
                       "rank": int(fit["rank"])})
    values = [entry["effective"] for entry in counts]
    return {
        "rows": counts,
        "lowest": float(min(values)),
        "highest": float(max(values)),
        "always_two_ranked": all(entry["rank"] == 2 for entry in counts),
        "why": ("the assumed modulus sets the size of the phase part beside "
                "the amplitude part, so it moves how well the second "
                "direction is resolved but not whether it exists"),
    }


def key_impact(frequency_hz, **kwargs) -> Dict[str, object]:
    """DOES EITHER ENTRY EARN A PLACE IN THE KEY? MEASURED, AND THE ANSWER
    IS NO.

    The standing rule is that an entry earns its place by the direction it
    adds rather than by being a mechanism, and the way to find out is to add
    it and look. On the VHS luma band the modelled interference key stands at
    7.91 effective of 14 at condition 7.76. Adding these entries:

        the clearance alone      7.58 of 15 at condition 110
        the strain alone         7.76 of 15 at condition 40
        both                     7.56 of 16 at condition 110
        all seven members        5.59 of 21 at condition 122

    EVERY ONE OF THEM LOWERS THE COUNT, and the reason is nameable rather
    than mysterious. The clearance's log magnitude is a straight line in
    frequency and reads 0.972 coherent with the particle-noise entry, whose
    own law is a power of frequency; the strain is a linear phase ramp and
    reads 0.982 coherent with the group-delay entry, whose phase is quadratic
    and over a band of this fractional width is very nearly the same curve.
    Both are second copies of directions the key already holds.

    So this component's honest recommendation is that NOTHING BE ADDED TO
    `interference.signatures()`. Its contribution is on the time axis, where
    the key has no entries at all, and its value on the frequency axis is the
    negative result: the tape path's whole mechanical parameter set is
    already inside the span of what is modelled, so no amount of further
    fitting on a frequency response can apportion it among its seven members.
    """
    from vhsdecode.models import interference as inf

    base = inf.signatures(frequency_hz, **kwargs)
    mine = signatures(frequency_hz)
    members = {"tape path: " + name: value
               for name, value in all_member_signatures(frequency_hz).items()}

    def count(entries):
        result = participation([_shape_of(value) for value in entries.values()])
        return {"count": result["count"], "effective": result["effective"],
                "condition": result["condition"]}

    cases = {
        "baseline": base,
        "with the clearance": {**base, **{k: v for k, v in mine.items()
                                          if "clearance" in k}},
        "with the strain": {**base, **{k: v for k, v in mine.items()
                                       if "strain" in k}},
        "with both": {**base, **mine},
        "with all seven members": {**base, **members},
    }
    out = {name: count(entries) for name, entries in cases.items()}
    baseline = out["baseline"]["effective"]
    out["earns_a_place"] = {
        name: bool(value["effective"] > baseline)
        for name, value in out.items() if name != "baseline"}
    out["why"] = ("an entry earns its place by the direction it adds; none "
                  "of these adds one, because the clearance is a straight "
                  "line in frequency and the strain is a linear phase ramp, "
                  "and the key already spans both")
    return out


def pressure_control() -> Dict[str, object]:
    """THE THIRD: the contact pressure's own identity.

    `p R w = T` exactly, by the statics it was derived from, at every tension
    in the specified range. It catches a width or a radius entering the wrong
    way round, which would rescale every clearance in the budget without
    changing any shape and so would never show in a direction count.
    """
    errors = []
    for tension in np.linspace(*TENSION_RANGE_N, 5):
        pressure = contact_pressure_pa(tension)
        errors.append(abs(pressure * DRUM_RADIUS_M * TAPE_WIDTH_M - tension))
    worst = float(max(errors))
    return {"worst_absolute_error_n": worst,
            "passes": bool(worst < 1e-12),
            "why": "p = T / (R w) is an identity, not an approximation"}
