"""The magnetic circuit: the head as turns and current, the tape as a B-H
curve, and the rotation that carries one past the other.

Ethan's directive, in his own terms and in the order he gave them:

  * "use magnetic flux to derive the rotation of the head itself to get the
    video head frequency response against the tape"
  * "magnetic poles and magnetic distance from the tape to the tape head
    over time, where time will derive the dropouts"
  * "model the head in terms of voltage and current that is being used
    magnetize the head"
  * "Head current to magnetic ratio is the number of turns divided by the
    length of the coil, voltage is what changes at the luma and this is
    modulating the head amp to amplitude. Use the expected properties of
    this and use the BH curve to model this part"
  * the same electromagnetism "to the RF cxadc capture device as a general
    profile"

NOTHING OF THIS EXISTED. The tree carried the head's LOSS terms - Wallace
spacing, gap, thickness, azimuth, all in `head_model` - and the tape's
coercivity and remanence in `tape_model`, but no magnetic circuit at all:
no magnetomotive force, no reluctance, no permeability, no B-H curve, and
no electrical side to the head whatever. The losses are what the circuit
does to a wavelength; this is the circuit.

THE TWO DIRECTIONS ARE NOT SYMMETRIC, and that asymmetry is the whole
reason the record side is a different problem from the playback side.

    RECORD    current in, magnetisation out.  H = N I / l, and the tape's
              B-H curve turns H into the remanent B it keeps. CURRENT is
              the drive.
    PLAYBACK  flux in, voltage out.  e = -N dPhi/dt, so the reproduce head
              differentiates and its output rises with frequency. VOLTAGE
              is the read-out.

Ethan: "voltage is what changes at the luma and this is modulating the
head amp to amplitude" - the FM carrier is a frequency, and what the head
delivers for it is a voltage that depends on that frequency through the
derivative above and on the wavelength through the losses.

AND THE RECORD SIDE IS AMPLITUDE-ONLY. Ethan again: the head correction
"needs to be extended to the recording VCR as well ... but does not
influence phase, only amplitude, since we don't have the original signal
being recorded to tape, we don't have that varying head profile over time,
I.e. also why dropouts are baked in." That is exactly right and
`record_side_observability` states why in the circuit's own terms.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "QUANTITIES", "VACUUM_PERMEABILITY", "AIR_SUSCEPTIBILITY",
    "air_permeability", "magnetomotive_force", "turns_per_metre",
    "field_intensity", "flux_density", "reluctance", "flux",
    "induced_voltage", "anhysteretic_b", "differential_permeability",
    "hysteresis_branch", "writing_speed_m_s", "wavelength_m",
    "rotation_response", "rotation_transfer", "spacing_loss_db",
    "spacing_over_time",
    "dropouts_from_spacing", "record_side_observability",
    "capture_input_profile", "reference_properties",
    "normative_status", "NORMATIVE_SOURCE", "NOT_IN_ANY_STANDARD",
    "PHASE_COHERENCE_FLOOR",
    "spec_band_centres", "band_delay_difference", "spacing_from_band_delay",
    "spacing_from_band_tilt", "two_band_delay_pair",
    "head_efficiency", "deep_gap_field_a_m", "karlqvist_field",
    "write_depth_m", "drive_ratio_for_depth", "optimum_record_drive",
    "recorded_layer", "recorded_magnetisation", "linked_flux",
    "record_current_curve",
]


# --------------------------------------------------------------------------
# The SI set, exactly as Ethan listed it
# --------------------------------------------------------------------------

QUANTITIES = (
    # name              symbol  SI unit                 defining relation
    ("field force",     "F",    "ampere-turn",          "F = N I"),
    ("field flux",      "Phi",  "weber",                "Phi = F / R"),
    ("field intensity", "H",    "ampere-turn per metre", "H = N I / l"),
    ("flux density",    "B",    "tesla",                "B = mu H"),
    ("reluctance",      "R",    "ampere-turn per weber", "R = l / (mu A)"),
    ("permeability",    "mu",   "tesla-metre per ampere-turn", "mu = B / H"),
)
"""Ethan's list, checked and kept in his units.

All six are correct and they are the standard magnetic-circuit set. Two
notes worth keeping beside them:

  * "tesla-metre per ampere-turn" is the same unit as the henry per metre
    the datasheets use - T*m/(A*turn) = Wb/(A*m) = H/m - so a permeability
    quoted either way needs no conversion.
  * the circuit is the exact analogue of an electrical one: F is the
    electromotive force, Phi the current, R the resistance, and
    Phi = F / R is Ohm's law. It is only an analogue - reluctance stores
    no energy and mu is not constant, which is what the B-H curve below
    is for - but every series and parallel rule carries over.
"""

VACUUM_PERMEABILITY = 1.25663706212e-6
"""mu_0 in H/m (= T*m/A), CODATA 2018.

NOT 4 pi x 10^-7 any more. The 2019 SI redefinition fixed the elementary
charge instead, so mu_0 became a MEASURED quantity; 4 pi x 10^-7 is now an
approximation good to 5.4e-10 relative. That is far below anything this
arc measures, and it is used exactly rather than rounded for the same
reason `rf_noise` carries Boltzmann's exact value: a constant that is known
should not be the thing that limits a residual.
"""

AIR_SUSCEPTIBILITY = 3.73e-7
"""chi for air at 20 C and one atmosphere, and it is POSITIVE.

ETHAN HAS THE MAGNITUDE RIGHT AND THE SIGN THE OTHER WAY: "Air is
diamagnetic so it follows the same magnetic conductivity curve, i.e. mu,
permiability which is -10e7 or so of air." The 10^-7 scale is exactly
right and is the number that matters here. But air is weakly
PARAMAGNETIC, not diamagnetic, because molecular oxygen carries two
unpaired electrons - O2 is one of the few paramagnetic gases, and it is
21 per cent of the air. Pure nitrogen IS diamagnetic (chi about -5e-9), so
the intuition holds for most gases and fails for the one mixture we care
about.

THE SIGN CHANGES NOTHING IN THIS MODEL and the magnitude is why: mu_r for
air is 1.00000037, so the head-to-tape gap is vacuum to within four parts
in ten million. Wallace's spacing loss treats it as vacuum and is right to
do so. What the number IS good for is stating that the gap contributes NO
reluctance modulation of its own - all of the gap's effect is geometric,
through the spacing, and none of it is magnetic.
"""


def air_permeability(temperature_k: float = 293.15,
                     pressure_pa: float = 101325.0) -> float:
    """mu of air, scaling the susceptibility by density (ideal gas).

    Included because it is the honest way to say "this is vacuum": the
    departure is 3.7e-7 at room conditions and falls with density, so no
    condition inside a VCR makes the gap magnetically anything but empty.
    """
    density_ratio = ((pressure_pa / 101325.0)
                     * (293.15 / max(temperature_k, 1e-9)))
    return VACUUM_PERMEABILITY * (1.0 + AIR_SUSCEPTIBILITY * density_ratio)


# --------------------------------------------------------------------------
# The circuit
# --------------------------------------------------------------------------

def magnetomotive_force(turns: float, current_a) -> np.ndarray:
    """F = N I, in ampere-turns. The circuit's driving force."""
    return float(turns) * np.asarray(current_a, dtype=np.float64)


def turns_per_metre(turns: float, coil_length_m: float) -> float:
    """N / l - ETHAN'S RATIO, stated exactly as he stated it: "Head current
    to magnetic ratio is the number of turns divided by the length of the
    coil".

    It is the constant that converts drive current to field intensity,
    H = (N/l) I, so it is the head's electrical-to-magnetic gain and it is
    fixed by construction. Units: ampere-turn per metre, per ampere.
    """
    return float(turns) / max(float(coil_length_m), 1e-30)


def field_intensity(turns: float, current_a, coil_length_m: float
                    ) -> np.ndarray:
    """H = N I / l, ampere-turns per metre."""
    return magnetomotive_force(turns, current_a) / max(float(coil_length_m),
                                                       1e-30)


def flux_density(field_intensity_a_m, permeability: Optional[float] = None
                 ) -> np.ndarray:
    """B = mu H, tesla. Linear - use `anhysteretic_b` where it is not."""
    mu = VACUUM_PERMEABILITY if permeability is None else float(permeability)
    return mu * np.asarray(field_intensity_a_m, dtype=np.float64)


def reluctance(length_m: float, area_m2: float,
               permeability: Optional[float] = None) -> float:
    """R = l / (mu A), ampere-turns per weber.

    WHETHER THE GAP DOMINATES IS NOT AUTOMATIC, and checking rather than
    repeating the textbook changed this module's own conclusion. The gap
    is often said to dominate a head's circuit because mu there is mu_0
    while the core's is thousands of times higher - but reluctance is
    l/(mu A), so what competes is the RATIO l/mu, and the gap wins only
    when

        mu_r(core) > l(core) / g

    Measured against the head's stated 0.30 um gap: a 2 mm core needs
    mu_r above 6667, a 5 mm core above 16667, a 20 mm core above 66667.
    At a typical ferrite mu_r of 3000 a 2 mm core carries TWICE the gap's
    reluctance, not a thousandth of it.

    So the core is not negligible, and `core permeability` is not the
    inert parameter it first appears in `reference_properties` - it
    decides which element sets the circuit. It is also the one we have no
    datasheet for.
    """
    mu = VACUUM_PERMEABILITY if permeability is None else float(permeability)
    return float(length_m) / max(mu * float(area_m2), 1e-300)


def flux(mmf_a_turn, reluctance_a_turn_per_wb) -> np.ndarray:
    """Phi = F / R, weber. The circuit's Ohm's law."""
    return (np.asarray(mmf_a_turn, dtype=np.float64)
            / max(float(reluctance_a_turn_per_wb), 1e-300))


def induced_voltage(turns: float, flux_wb, times_s) -> np.ndarray:
    """e = -N dPhi/dt, volts - Faraday, and the whole playback side.

    THIS IS WHY THE REPRODUCE HEAD DIFFERENTIATES, and therefore why its
    output rises 6 dB per octave before any loss is applied: a constant
    recorded flux amplitude at twice the frequency changes twice as fast.
    It is also the reason the colour-under sits 15 dB below the luma
    carrier for free (`colour_under.carrier_reduction_db`) - same flux,
    a fifth of the frequency, a fifth of the voltage.
    """
    phi = np.asarray(flux_wb, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    return -float(turns) * np.gradient(phi, t)


# --------------------------------------------------------------------------
# The B-H curve
# --------------------------------------------------------------------------

def anhysteretic_b(field_intensity_a_m, saturation_t: float,
                   coercivity_a_m: float) -> np.ndarray:
    """The anhysteretic B-H curve: B = Bs tanh(H / Hc).

    A LANGEVIN-CLASS CURVE, not a fit. The tanh form is the two-state
    limit of the Langevin function and carries the two things that matter
    - it saturates at Bs and its slope at the origin is Bs/Hc - with no
    free parameter beyond the two the datasheet already states. VHS tape
    is 600 oersted class (JVC VTG82063 s.1) with a remanence near 0.15 T,
    both already in `tape_model`.

    The anhysteretic curve is the right one for RECORDING because the AC
    bias is what puts the medium on it: bias plus signal walks the
    material through many decreasing loops, which is exactly the process
    that replaces the hysteretic path with the anhysteretic one and makes
    the record transfer close to linear. Without bias the same drive would
    ride `hysteresis_branch` and be badly nonlinear.

    FM CHANGES WHAT THAT LINEARITY IS FOR. The luma is recorded as a
    constant-amplitude carrier whose FREQUENCY carries the picture, so the
    record side is driven into saturation deliberately and the curve's
    linearity is not required of it - what is required is that the
    saturated level be STABLE, because it sets the flux the playback head
    recovers at every wavelength alike.

    BUT THE CHROMA IS THE BIASED CASE THE FIRST PARAGRAPH DESCRIBES, and
    this module said "without bias" as though VHS had none. It has one:
    SMPTE 32M 7.5.1.2.2 - *"The chrominance signal shall be recorded with
    the luminance FM signal acting as bias"* - and the JVC guide says the
    same for baseline VHS at 1.1.6. The colour under is an AM carrier
    recorded on the luma FM's own bias, so THIS curve, not
    `hysteresis_branch`, is the one it rides, and 3.9.2.1.2's order to
    record it 7 to 10 dB below saturation is the linearity budget that
    choice buys. `tape_bias` carries the clauses, the measurement and the
    limit - the bias is a FEW-CYCLE bias, three tenths of a cycle across
    the switching band, so the linearisation is partial and a third-order
    product survives.
    """
    h = np.asarray(field_intensity_a_m, dtype=np.float64)
    return float(saturation_t) * np.tanh(h / max(float(coercivity_a_m), 1e-30))


def differential_permeability(field_intensity_a_m, saturation_t: float,
                              coercivity_a_m: float) -> np.ndarray:
    """mu_d = dB/dH on the anhysteretic curve, in T*m/(A*turn).

    The small-signal permeability, which is what a modulation rides on.
    It is largest at H = 0 and collapses in saturation - so a head driven
    hard has less incremental response, and the same modulation depth
    produces less flux. That is the mechanism behind a record-level axis
    being an axis at all rather than a scale factor.
    """
    h = np.asarray(field_intensity_a_m, dtype=np.float64)
    hc = max(float(coercivity_a_m), 1e-30)
    return float(saturation_t) / hc / np.cosh(h / hc) ** 2


def hysteresis_branch(field_intensity_a_m, saturation_t: float,
                      coercivity_a_m: float, ascending: bool = True
                      ) -> np.ndarray:
    """One branch of the hysteresis loop, shifted by the coercivity.

    The loop is the anhysteretic curve displaced along H by +-Hc: going up
    the material lags, so it needs +Hc more field to reach a given B, and
    coming down it needs Hc less. Crude beside a Jiles-Atherton or
    Preisach model and sufficient for what is asked of it here - the
    remanence at H = 0 and the field needed to erase it.
    """
    h = np.asarray(field_intensity_a_m, dtype=np.float64)
    offset = float(coercivity_a_m) * (1.0 if ascending else -1.0)
    return anhysteretic_b(h - offset, saturation_t, coercivity_a_m)


# --------------------------------------------------------------------------
# The rotation, and what it does to the response
# --------------------------------------------------------------------------

def writing_speed_m_s(drum_diameter_m: float = 0.0620,
                      revolutions_per_s: float = 30000.0 / 1001.0,
                      tape_speed_m_s: float = 33.35e-3,
                      helix_degrees: float = 0.0) -> Dict[str, float]:
    """THE ROTATION, which is what Ethan asks the flux to be derived from.

    The head is on a drum; the tape moves too; the writing speed is their
    relative speed along the track. For VHS the drum is 62 mm and turns
    once per FRAME - two fields, one from each head - so 29.97 rev/s.

    MEASURED AGAINST THE SPEC AND THEY DISAGREE BY 1.3 PER CENT: the
    geometry gives pi x 0.0620 x 29.97 = 5.8375 m/s while JVC VTG82063
    STATES 5.80 m/s. The stated figure is the one this project uses (see
    the mechanical-spec note), and the discrepancy is kept visible here
    rather than resolved silently, because it propagates straight into
    every wavelength and therefore into every loss.
    """
    circumference = math.pi * float(drum_diameter_m)
    head = circumference * float(revolutions_per_s)
    along = float(tape_speed_m_s) * math.cos(math.radians(helix_degrees))
    return {
        "circumference_m": circumference,
        "head_speed_m_s": head,
        "tape_contribution_m_s": along,
        "writing_speed_m_s": head + along,
        "stated_m_s": 5.80,
        "departure_from_stated": (head + along) / 5.80 - 1.0,
    }


def wavelength_m(frequency_hz, writing_speed: float) -> np.ndarray:
    """lambda = v / f. The bridge between the electrical and the magnetic:
    every loss in `head_model` is a function of wavelength, and this is
    the only place the rotation enters them."""
    f = np.asarray(frequency_hz, dtype=np.float64)
    return float(writing_speed) / np.maximum(f, 1e-30)


def rotation_response(frequency_hz, turns: float, spacing_m: float,
                      writing_speed: float, gap_m: float = 0.30e-6,
                      remanence_t: float = 0.15,
                      track_width_m: float = 58e-6) -> Dict[str, np.ndarray]:
    """The head's frequency response against the tape, built from the flux.

    Assembled in the order the physics runs, so each term is separable:

      1. FARADAY, `e = -N dPhi/dt`: a sinusoidal flux of fixed amplitude
         gives a voltage proportional to `omega`, so the bare head rises
         6 dB per octave. This is the term the rotation sets, because
         omega for a given wavelength is v/lambda.
      2. SPACING, `exp(-2 pi d / lambda)`: Wallace, and the steepest term.
      3. GAP, `sinc(g / lambda)`: the head averages over its own gap and
         nulls where the gap spans a whole wavelength.

    Returned in volts per unit flux so a caller can scale by whatever the
    record side actually left, rather than being handed a normalised curve
    whose level means nothing.
    """
    f = np.asarray(frequency_hz, dtype=np.float64)
    lam = wavelength_m(f, writing_speed)
    faraday = 2.0 * np.pi * f * float(turns)
    spacing = np.exp(-2.0 * np.pi * float(spacing_m) / lam)
    ratio = float(gap_m) / lam
    gap = np.abs(np.sinc(ratio))          # numpy's sinc is sin(pi x)/(pi x)
    flux_amplitude = float(remanence_t) * float(track_width_m)
    return {
        "wavelength_m": lam,
        "faraday": faraday,
        "spacing_loss": spacing,
        "gap_loss": gap,
        "volts_per_tesla_metre": faraday * spacing * gap,
        "response": faraday * spacing * gap * flux_amplitude,
        "gap_null_hz": float(writing_speed) / max(float(gap_m), 1e-30),
    }


def rotation_transfer(frequency_hz, turns: float, spacing_m: float,
                      writing_speed: float, gap_m: float = 0.30e-6,
                      remanence_t: float = 0.15,
                      track_width_m: float = 58e-6,
                      phase_reference_hz: Optional[float] = None
                      ) -> Dict[str, np.ndarray]:
    """THE SAME RESPONSE AS A COMPLEX QUANTITY, which is what it always was.

    `rotation_response` returns magnitudes, and every one of its three terms
    has a phase that it throws away:

      FARADAY is a differentiator. `e = -N dPhi/dt` is `j omega` and not
              `omega`, so the term carries exactly a quarter turn at every
              frequency. Written as `2 pi f N` that quarter turn is gone.
      SPACING is minimum phase, and its companion is not a constant. It is
              the `ln f` law `band_delay_difference` derives below, which is
              the entire subject of this module's time axis.
      GAP     is `sinc`, and a sinc that has gone negative is a `pi` flip.
              `np.abs` around it removes a sign change rather than a level.

    So this returns `H(f)` and the caller may take a magnitude if a magnitude
    is what is wanted. The sign of the gap sinc is taken from
    `head_model.sinc_sign_phase` rather than reimplemented, so there is one
    account of it in the tree - but that function applies the standard
    fringing correction to the gap before taking the sinc, and
    `rotation_response` has always used the PHYSICAL gap, so the gap is
    scaled by that factor on the way in. A sign taken from a sinc that nulls
    at a different frequency from the magnitude's would be worse than no
    sign at all.

    MEASURED, at the standard's own writing speed and this project's typical
    head, over the standard's own luma band and colour-under carrier:

        gap 0.30 um, writing speed 5.80 m/s -> first sinc null 19.33 MHz
        sign flips inside 0.4 to 4.4 MHz     0
        Faraday phase                        +90.000 deg at every frequency
        spacing-loss phase, 0.2045 um        -40.76 ns of group delay
                                             between the two bands

    - so in this band the gap's sign contributes nothing, the Faraday
    quarter turn is a constant that cancels out of any band DIFFERENCE, and
    the whole of the recoverable time information is the spacing term.

    `phase_reference_hz` PINS THE ONE ARBITRARY QUANTITY. The minimum-phase
    companion of an unbounded log magnitude is fixed only up to a linear
    term, which is a constant group delay and therefore an origin of time.
    Left to itself this function anchors it at the grid's lowest positive
    frequency, so two calls on different grids return phases that differ by
    a delay - the same grid dependence recorded against
    `head_model.group_delay_s` above. Pass this to pin it, and the returned
    `phase_reference_hz` says which was used either way. What the anchor
    cannot move is any difference of group delays between two frequencies,
    which is exactly the quantity this module measures.
    """
    from vhsdecode.models import head_model

    f = np.asarray(frequency_hz, dtype=np.float64)
    lam = wavelength_m(f, writing_speed)
    # Faraday: j omega, the quarter turn written out
    faraday = 1j * 2.0 * np.pi * f * float(turns)
    # the separation loss and its minimum-phase companion. The companion is
    # `phi = (2 d / v) omega ln(omega / omega_ref)`, whose derivative is the
    # `ln f` group delay; the reference is arbitrary and cancels out of every
    # difference, so the grid's own lowest positive frequency is used.
    exponent = 2.0 * np.pi * float(spacing_m) / lam
    positive = f[f > 0]
    if phase_reference_hz is not None:
        reference = float(phase_reference_hz)
    else:
        reference = float(positive.min()) if positive.size else 1.0
    if reference <= 0:
        raise ValueError("the phase reference must be a positive frequency")
    with np.errstate(divide="ignore", invalid="ignore"):
        companion = np.where(
            f > 0,
            (2.0 * float(spacing_m) / float(writing_speed))
            * (2.0 * np.pi * f) * np.log(np.maximum(f, 1e-30) / reference)
            / np.pi,
            0.0)
    spacing = np.exp(-exponent + 1j * companion)
    ratio = float(gap_m) / lam
    sign = head_model.sinc_sign_phase(
        f, float(writing_speed), float(track_width_m),
        gap_m=float(gap_m) / head_model.EFFECTIVE_GAP_FACTOR)
    gap = np.abs(np.sinc(ratio)) * np.exp(1j * sign)
    flux_amplitude = float(remanence_t) * float(track_width_m)
    transfer = faraday * spacing * gap
    return {
        "wavelength_m": lam,
        "faraday": faraday,
        "spacing_loss": spacing,
        "gap_loss": gap,
        "volts_per_tesla_metre": transfer,
        "response": transfer * flux_amplitude,
        "gap_null_hz": float(writing_speed) / max(float(gap_m), 1e-30),
        "gap_sign_flips": int(np.count_nonzero(np.diff(sign) != 0.0)),
        "phase_reference_hz": reference,
    }


# --------------------------------------------------------------------------
# Distance over time, and the dropouts it derives
# --------------------------------------------------------------------------

def spacing_loss_db(spacing_m, wavelength) -> np.ndarray:
    """Wallace: 54.6 d / lambda decibels.

    The constant is 20 log10(e) x 2 pi = 54.575, derived rather than
    quoted. It is the sharpest lever in the whole chain - a tenth of a
    wavelength of dirt is 5.5 dB - and at the luma carrier's 1.5 um
    wavelength that is 0.15 um, which is a smoke particle.
    """
    d = np.asarray(spacing_m, dtype=np.float64)
    lam = np.asarray(wavelength, dtype=np.float64)
    return (20.0 * np.log10(np.e) * 2.0 * np.pi) * d / np.maximum(lam, 1e-30)


def spacing_over_time(times_s, base_spacing_m: float = 0.05e-6,
                      events: Sequence[Dict[str, float]] = ()
                      ) -> np.ndarray:
    """d(t) - "magnetic distance from the tape to the tape head over time".

    Ethan's own construction: the distance is the time-varying quantity,
    and the dropouts are DERIVED from it rather than detected as a
    separate phenomenon. Each event is a Gaussian excursion in spacing
    with a centre, a width and a height.
    """
    t = np.asarray(times_s, dtype=np.float64)
    d = np.full_like(t, float(base_spacing_m))
    for event in events:
        centre = float(event.get("centre_s", 0.0))
        width = max(float(event.get("width_s", 1e-6)), 1e-12)
        height = float(event.get("height_m", 0.0))
        d = d + height * np.exp(-0.5 * ((t - centre) / width) ** 2)
    return d


def dropouts_from_spacing(times_s, spacing_m, wavelength: float,
                          threshold_db: float = 6.0) -> Dict[str, object]:
    """THE DROPOUTS, derived from the distance rather than declared.

    "magnetic distance from the tape to the tape head over time, where
    time will derive the dropouts" - so a dropout is not a category of
    event, it is the spacing loss exceeding what the demodulator can
    carry. That makes it continuous with ordinary level variation, which
    is the right model: the same mechanism at a smaller amplitude is the
    envelope ripple the AGC chases.

    The threshold is the caller's, because it belongs to the demodulator
    and not to the magnetics - the FM threshold is where the carrier-to-
    noise stops supporting the deviation.
    """
    t = np.asarray(times_s, dtype=np.float64)
    loss = spacing_loss_db(spacing_m, wavelength)
    over = loss > float(threshold_db)
    edges = np.diff(over.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if over.size and over[0]:
        starts.insert(0, 0)
    if over.size and over[-1]:
        stops.append(len(over))
    spans = [(float(t[a]), float(t[min(b, len(t) - 1)]))
             for a, b in zip(starts, stops)]
    return {
        "loss_db": loss,
        "in_dropout": over,
        "spans_s": spans,
        "count": len(spans),
        "worst_db": float(np.max(loss)) if loss.size else 0.0,
        "fraction": float(np.mean(over)) if over.size else 0.0,
        "threshold_db": float(threshold_db),
    }


# --------------------------------------------------------------------------
# The record side
# --------------------------------------------------------------------------

def record_side_observability() -> Dict[str, object]:
    """Why the RECORDING head reaches amplitude and never phase.

    Ethan: the head correction "needs to be extended to the recording VCR
    as well ... but does not influence phase, only amplitude, since we
    don't have the original signal being recorded to tape, we don't have
    that varying head profile over time, I.e. also why dropouts are baked
    in."

    Stated in the circuit's terms, there are three separate reasons and
    they are not the same reason twice:

      1. NO REFERENCE. Phase is measured against something. The playback
         head has one - the specification's own sync and burst, which
         arrive with known shape - so its phase departure is measurable.
         The record head's input was the source signal, which was never
         captured, so there is nothing its phase can be referred to.

         THIS ONE IS NO LONGER TRUE OF THIS PROJECT'S CAPTURES, and the
         change is worth stating rather than leaving the reasoning to
         stand on a premise that has since altered. The test-pattern set
         holds MATCHED taps - the drive going to the record head and the
         signal coming back from the playback head, of the same signals
         on the same machine - so the original being recorded to tape IS
         captured, and it is the reference. Measured through
         `tap_transfer` on 75bars SP, the record tap is coherent with
         itself to 0.99799 and the record-to-playback transfer's phase is
         determined across the luma band to 0.209 and 0.184 degrees rms
         for the two heads over 13 and 14 fields. That is a phase
         measurement of the write and read path together, referred to the
         drive.

         WHAT REMAINS TRUE, and it is the operative limit, is reason 2
         below plus one this list did not have: on a single deck the head
         that wrote a track is the same physical head that reads it, so
         the record and playback contributions are not two terms to be
         separated but one head measured twice. `record_head_derivable`
         names the three experiments that would part them.
      2. THE PROFILE VARIED IN TIME AND THAT TIME IS GONE. The record
         head's spacing wandered while it wrote, exactly as the playback
         head's does while it reads, but that wander is now written into
         the medium. It is a property of the tape, not of the reading, so
         no amount of re-reading separates it.
      3. WHICH IS WHY DROPOUTS ARE BAKED IN. A record-side spacing
         excursion wrote less magnetisation; playing it back perfectly
         recovers exactly the magnetisation that is there. A playback-side
         dropout is a reading failure and can in principle be re-read; a
         record-side one is an absence.

    AMPLITUDE SURVIVES because it does not need a reference in the same
    way: the recorded level is a scalar on a known carrier, and a known
    carrier is what an FM system has by construction. Ethan's witness -
    "the color phase rotation change near the bottom of the screen" - is
    the head-switch region, where the two heads' records meet and their
    amplitude profiles differ; that difference is measurable without the
    original because it is a difference between two things we do have.
    """
    return {
        "amplitude": "observable - a scalar on a carrier of known shape, "
                     "and the two heads' records can be differenced against "
                     "each other at the head switch",
        "phase": "observable where a MATCHED RECORD TAP exists, which this "
                 "project's test-pattern captures have - the drive is the "
                 "reference, and the transfer's phase is determined to "
                 "about 0.2 degrees rms across the luma band; NOT separable "
                 "into a record share and a playback share on one deck, "
                 "because the head that wrote the track is the head that "
                 "reads it",
        "witness": "the colour phase rotation change near the bottom of the "
                   "screen, i.e. the head-switch region",
        "why_dropouts_are_baked_in":
            "a record-side spacing excursion wrote less magnetisation; "
            "perfect playback recovers exactly the magnetisation present, "
            "so the loss is an absence in the medium rather than a failure "
            "of the reading",
        "consequence": "without a record tap the record head enters the key "
                       "as an AMPLITUDE axis only, since offering it a phase "
                       "direction would let the playback head's phase leak "
                       "into it; with a matched record tap the phase of the "
                       "write and read path together is measurable, and what "
                       "is still not available is the split between them",
        "reference_when_tapped": "the record tap itself, coherent with "
                                 "itself to 0.99799 on 75bars SP",
    }


def capture_input_profile(turns: float = 1.0,
                          inductance_h: float = 1e-6,
                          resistance_ohm: float = 75.0,
                          capacitance_f: float = 20e-12) -> Dict[str, object]:
    """The same electromagnetism at the capture device - "apply to
    electromagnetitisim of the RF cxadc capture device as a general
    profile".

    The card is not a magnetic circuit but it is the same physics one
    stage further on: the head's inductance and the cable and input
    capacitance form a loaded resonant circuit, and its corner is where
    the flat assumption stops holding.

      f_resonance = 1 / (2 pi sqrt(L C))
      Q           = (1/R) sqrt(L / C)

    Returned as a profile rather than a correction, because the values
    are the caller's to supply per card - this states the SHAPE and its
    corner, which is what a general profile is.
    """
    l, c, r = float(inductance_h), float(capacitance_f), float(resistance_ohm)
    if l <= 0 or c <= 0:
        return {"resonance_hz": float("nan"), "q": float("nan")}
    resonance = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
    q = math.sqrt(l / c) / max(r, 1e-30)
    return {
        "resonance_hz": resonance,
        "q": q,
        "damping": "overdamped" if q < 0.5 else
                   ("critical" if abs(q - 0.5) < 1e-9 else "underdamped"),
        "turns": float(turns),
        "note": "a general profile: the shape is a second-order low pass "
                "with this corner and Q, and the values belong to the card",
    }


# --------------------------------------------------------------------------
# The list: which properties can form the reference
# --------------------------------------------------------------------------

def reference_properties() -> Dict[str, Dict[str, object]]:
    """EVERY MAGNETIC PROPERTY, and whether its measurement set is complete.

    Ethan asked for "the list of magnetic properties that have the complete
    set of measurements needed to form the reference used for comparison to
    get the one used for transform". The reference is the SYNTHETIC side of
    a component - what the specification says the quantity is - and a
    property can serve as one only when three things hold at once:

      SPECIFIED    its expected value comes from a published figure, not a
                   fit, or the transform is comparing a guess to a guess;
      WITNESSED    some measurement on this signal responds to it;
      SEPARABLE    that response is distinguishable from the other
                   properties', or the comparison attributes the residual
                   to whichever name was asked first.

    THE THIRD IS WHERE MOST OF THEM FAIL, and it is measured, not
    supposed: the six magnetic mechanisms collapse to 1.58 distinguishable
    of 6, with gap and azimuth coherent at 1.000 over the luma band. So the
    honest list is short, and the value of saying so is that a transform
    built on a property marked `separable=False` will move the residual
    without being able to say what it removed.

    `complete` is the conjunction. Each entry carries `si`, the unit from
    Ethan's own table, so the reference is stated in the units the
    transform will use.
    """
    # NORMATIVE is not the same as SPECIFIED, and conflating them was a
    # real weakness here. SMPTE 32M-1998 does NOT contain the head: not the
    # gap, not the spacing, not the coating thickness, not the remanence,
    # not the saturation, and none of the electrical side. It fixes the
    # track width, the azimuth and the coercivity ("approximately"), and
    # that is all - see `vhs_specification.head_values_present`. So a value
    # taken from `head_model.TYPICAL` is a TYPICAL value, not a
    # specification, and a reference built on it is a reference built on a
    # convention.
    def entry(si, unit, value, source, witness, specified, witnessed,
              separable, note):
        return {"si": si, "unit": unit, "expected": value, "source": source,
                "witness": witness, "specified": specified,
                "witnessed": witnessed, "separable": separable,
                "complete": bool(specified and witnessed and separable),
                "note": note}

    return {
        # ---- the medium ----
        "coercivity": entry(
            "H", "ampere-turn per metre", 600.0 * 1000.0 / (4.0 * math.pi),
            "JVC VTG82063 s.1, '600 oersted class (nominal)'",
            "the record-level axis: coercivity sets what magnetisation a "
            "given head current leaves, so it shows as the level at which "
            "the recording saturates",
            True, True, True,
            "COMPLETE. A published figure, a witness that responds to it "
            "alone, and it acts at RECORD where the others act at "
            "playback - which is what separates it."),
        "remanence": entry(
            "B", "tesla", 0.15,
            "typical for gamma-ferric VHS coating (tape_model)",
            "the envelope level",
            True, True, False,
            "NOT SEPARABLE. It scales the recovered flux at every "
            "wavelength alike, so it is collinear with every other gain "
            "in the chain - head sensitivity, preamp, AGC. Usable as a "
            "reference for the ABSOLUTE level only if one of those is "
            "independently pinned, and none is."),
        "coating thickness": entry(
            "l", "metre", 0.20e-6, "tape_model / head_model TYPICAL",
            "thickness loss, (1 - exp(-2 pi t / lambda)) / (2 pi t / lambda)",
            True, True, False,
            "NOT SEPARABLE from spacing over the luma band - both are "
            "smooth monotone roll-offs in d/lambda. The measured collapse "
            "puts the whole family at 1.58 of 6."),
        "saturation flux density": entry(
            "B", "tesla", None, "NOT IN THE TREE - no datasheet figure",
            "the anhysteretic curve's asymptote; would need a level sweep "
            "into saturation to witness",
            False, False, True,
            "MISSING. `anhysteretic_b` needs it and currently takes it "
            "from the caller. Without it the B-H curve has a shape and no "
            "scale."),

        # ---- the head, geometric ----
        "gap length": entry(
            "l", "metre", 0.30e-6, "head_model TYPICAL",
            "the gap null, at v/g - a ZERO in the response, which is the "
            "one unambiguous feature in the whole band",
            True, True, False,
            "SPECIFIED AND WITNESSED, NOT SEPARABLE: coherent with azimuth "
            "at 1.000 over the luma band. The NULL would separate it, but "
            "at 5.8 m/s and 0.30 um it sits at 19.3 MHz, far outside the "
            "recorded band - so the feature that would make it "
            "identifiable is not in the signal."),
        "head-to-tape spacing": entry(
            "l", "metre", 0.05e-6, "head_model TYPICAL",
            "Wallace loss, 54.6 d/lambda dB - and, over time, the dropouts",
            True, True, False,
            "The steepest lever and the least separable: exp(-2 pi d / "
            "lambda) is smooth and monotone, so it trades against gap, "
            "thickness and any other roll-off. Its TIME variation is a "
            "different matter and IS separable - see "
            "`dropouts_from_spacing` - because a transient is not "
            "collinear with a static loss."),
        "azimuth error": entry(
            "angle", "degree", 0.0,
            "JVC VTG82063 tolerance +-10 arc-min",
            "the azimuth loss across the track width",
            True, True, False,
            "Coherent with gap at 1.000. Note the fitted value in this "
            "arc EXCEEDS the +-10 arc-min the spec allows, so it is an "
            "upper bound absorbing other losses rather than a measurement "
            "of azimuth."),
        "track width": entry(
            "l", "metre", 58e-6,
            "the decoder's own rf parameters (video_track_width)",
            "sets the flux a given remanence delivers",
            True, True, False,
            "Collinear with remanence - both are a scale on the flux."),

        # ---- the head, electrical: NONE of this is known ----
        "turns": entry(
            "N", "turn (dimensionless)", None,
            "NOT IN THE TREE - no head datasheet",
            "would be witnessed by the head's inductance and its "
            "sensitivity in volts per tesla",
            False, False, True,
            "MISSING, and it is the first term of Ethan's own ratio N/l. "
            "Without it the circuit gives SHAPES - the 6 dB/octave "
            "Faraday rise, the gap null, the losses - but no absolute "
            "volts. That is survivable because every consumer here works "
            "in ratios, and it must be stated rather than assumed."),
        "coil length": entry(
            "l", "metre", None, "NOT IN THE TREE",
            "the second term of N/l", False, False, True,
            "MISSING. N and l enter only as their RATIO, so one number "
            "would complete both - which is worth knowing if a datasheet "
            "ever turns up."),
        "core permeability": entry(
            "mu", "tesla-metre per ampere-turn", None,
            "NOT IN THE TREE - ferrite or sendust, thousands x mu_0",
            "sets how completely the gap dominates the circuit's "
            "reluctance",
            False, False, True,
            "MISSING AND NOT INERT - this entry said the opposite until "
            "the arithmetic was run. Reluctance is l/(mu A), so gap and "
            "core compete on l/mu, and the gap dominates only where "
            "mu_r(core) > l(core)/g: above 6667 for a 2 mm core against "
            "the stated 0.30 um gap. At a typical ferrite 3000 the core "
            "carries twice the gap's reluctance. Which element sets the "
            "circuit therefore turns on the one number we do not have."),
        "gap reluctance": entry(
            "R", "ampere-turn per weber", None,
            "DERIVABLE from the gap length, the track width and mu_0",
            "`reluctance(gap_m, track_width x depth, mu_0)`",
            True, True, True,
            "COMPLETE ONCE A GAP DEPTH IS SUPPLIED - the only unmeasured "
            "term is the depth into the head, and every other quantity is "
            "already stated. This is the one electrical-side property "
            "that can be built from what the tree holds."),

        # ---- the gap medium ----
        "air permeability": entry(
            "mu", "tesla-metre per ampere-turn",
            VACUUM_PERMEABILITY * (1.0 + AIR_SUSCEPTIBILITY),
            "CODATA 2018 mu_0, with air's susceptibility at STP",
            "none needed - it is known to four parts in ten million",
            True, True, True,
            "COMPLETE, and it is the only one that needs no measurement at "
            "all. Its use is NEGATIVE: it establishes that the gap "
            "contributes no reluctance modulation, so every effect of the "
            "spacing is geometric. Note air is weakly PARAMAGNETIC (O2), "
            "not diamagnetic - the magnitude is what matters and it is "
            "3.7e-7 either way."),

        # ---- the transport ----
        "writing speed": entry(
            "v", "metre per second", 5.80,
            "JVC VTG82063 states 5.80 m/s",
            "the drum rate, and every wavelength in the model",
            True, True, True,
            "COMPLETE BUT CONTESTED: the geometry gives 5.8709 m/s from a "
            "62 mm drum at 29.97 rev/s plus the tape's 33.35 mm/s, which "
            "is 1.2 per cent above the stated figure. The stated one is "
            "used. A 1.2 per cent error in v is a 1.2 per cent error in "
            "every wavelength and so in every loss exponent."),
        "drum diameter": entry(
            "l", "metre", 0.0620, "the format requires it (transport_model)",
            "the drum-rate line in the envelope's spectrum",
            True, True, True,
            "COMPLETE. Fixed by the format, and directly witnessed - the "
            "drum rate is found in the envelope on both tapes."),
    }


# Which entries a PUBLISHED standard actually fixes, and which document.
# NORMATIVE IS NOT THE SAME AS PRESENT IN THE TREE, and conflating the two
# was a real weakness in `reference_properties`: a value taken from
# `head_model.TYPICAL` is a typical value, not a specification.
#
# Checked against both documents. SMPTE 32M-1998 does not contain the head
# at all - not the gap, not the spacing, not the coating, not the
# remanence, not the saturation - it fixes the track width, the azimuth and
# the coercivity ("approximately 50 x 10^3 A/m"), and that is all. The JVC
# guide VTG82063 adds the GAP (0.3 um, s.7.2.1) and the azimuth's tolerance
# (6 degrees +- 10 arc-minutes, table 1-1-1) and states Wallace's 54.6 d/l
# law outright with a worked example - but its electrical side is equally
# empty. It NAMES effective permeability, maximum flux density and residual
# flux density as the head's three design axes (s.1.2.2-6) and quantifies
# none of them.
NORMATIVE_SOURCE = {
    "coercivity": "SMPTE 32M-1998 s.3.2.1.4 (~50 kA/m) and JVC VTG82063 "
                  "s.1.1.25 (600 oersted class = 47.7 kA/m) - they differ "
                  "by 4.6 per cent and both are hedged",
    "gap length": "JVC VTG82063 s.7.2.1, '0.3 micron gap of the video "
                  "head' - NOT in SMPTE 32M, which mentions head gaps only "
                  "qualitatively",
    "azimuth error": "JVC VTG82063 table 1-1-1 item 18, '6 degrees +- 10 "
                     "arc-minutes' - the only per-head tolerance either "
                     "document states",
    "track width": "SMPTE 32M-1998 table 2 and JVC table 1-1-1, 0.058 mm, "
                   "nominal with no tolerance",
    "writing speed": "SMPTE 32M-1998 s.3.1.4 and JVC table 1-1-1, both "
                     "5.80 m/s STATED and never derived - JVC's own drum "
                     "figures give 5.843",
    "drum diameter": "SMPTE 32M-1998 s.3.1.5 and JVC table 1-1-1, 62.00 mm "
                     "+- 0.01",
}

NOT_IN_ANY_STANDARD = {
    "head-to-tape spacing": "neither document states a flying height, a "
                            "penetration or a contact spacing - though JVC "
                            "s.7.2.4 gives the 54.6 d/lambda law that "
                            "CONSUMES one, with a worked example",
    "coating thickness": "only TOTAL tape thickness (19 +1/-2 um) is "
                         "stated; the magnetic layer's own depth is not, "
                         "in either document. JVC gives a recording DEPTH "
                         "of about 0.3 um and the lambda/4 optimum rule "
                         "(s.7.2.4), which are depths of magnetisation "
                         "rather than the coating",
    "remanence": "not stated anywhere - 'remanen' does not occur in SMPTE "
                 "32M, and JVC names residual flux density as a design "
                 "axis without quantifying it",
    "saturation flux density": "not stated; SMPTE's one 'saturation' is a "
                               "signal level, and JVC's is qualitative",
    "turns": "absent from both",
    "coil length": "absent from both",
    "core permeability": "absent from both - JVC s.1.2.2-6 names 'high "
                         "effective permeability' as the first design axis "
                         "of a video head and gives no number",
}


def normative_status() -> Dict[str, Dict[str, object]]:
    """Which magnetic properties a PUBLISHED standard fixes, and which not.

    Ethan: "Let's use the VHS specs to get the needed values for the
    heads." The specs were read, and the answer is mostly a negative:

      SMPTE 32M-1998   the track width, the azimuth, the coercivity
                       ("approximately"), the transport, the carriers.
                       NOT the gap, the spacing, the coating, the
                       remanence, the saturation, or anything electrical.
      JVC VTG82063     adds the GAP at 0.3 um, the azimuth's +-10' , the
                       drum at 30 rps, and Wallace's 54.6 d/lambda law
                       with a worked example. Its electrical side is
                       equally empty - it names the three head design axes
                       and quantifies none.

    So five of the head model's parameters rest on a published figure and
    the rest are typical values or fits. That is not a criticism of the
    model; it is the honest provenance, and it is why the electrical side
    was marked missing rather than guessed.
    """
    table = reference_properties()
    out = {}
    for name, value in table.items():
        source = NORMATIVE_SOURCE.get(name)
        out[name] = {
            **value,
            "normative": source is not None,
            "normative_source": source,
            "absence": NOT_IN_ANY_STANDARD.get(name),
        }
    return out


# --------------------------------------------------------------------------
# What the coercivity actually buys: the transition length
# --------------------------------------------------------------------------

WILLIAMS_COMSTOCK_FACTOR = 1.0 / math.pi
"""The O(1) constant in the transition-length formula, isolated so it can
be argued with.

Derivations of the arctangent transition parameter differ among themselves
by factors of order one - Williams-Comstock, the slope model and Bertram's
self-consistent treatment do not agree on the constant, and the difference
between them is a few tens of per cent in `a`. What every one of them
agrees on, and what this module relies on, is the SCALING:

    a  proportional to  sqrt( Mr * delta * (d + delta/2) / Hc )

so the material enters only through the ratio Mr/Hc. That ratio is the
lever, it is 2.500 for this tape, and it is what makes coercivity a
modelling parameter rather than a datasheet decoration.
"""


def remanent_magnetisation(remanence_t: float = 0.15) -> float:
    """Mr in A/m from Br in tesla: Mr = Br / mu_0.

    The datasheet quotes a flux density and the transition length wants a
    magnetisation, and the two differ by mu_0 - about five orders of
    magnitude, so getting it wrong is not subtle.
    """
    return float(remanence_t) / VACUUM_PERMEABILITY


def transition_length(coercivity_a_m: float = 600.0 * 1000.0 / (4.0 * math.pi),
                      remanence_t: float = 0.15,
                      thickness_m: float = 0.20e-6,
                      spacing_m: float = 0.05e-6,
                      level: float = 1.0,
                      factor: float = WILLIAMS_COMSTOCK_FACTOR) -> float:
    """`a`, the length over which a recorded reversal actually turns over.

    THIS IS WHAT THE COERCIVITY IS FOR, and until now nothing used it.
    `magnetic.py` caps the recording depth at `lambda / (2 pi)`, which is a
    purely GEOMETRIC self-demagnetisation limit - it knows the wavelength
    and the coating and nothing about the material. The physical limit is
    a competition: the recorded magnetisation's own demagnetising field
    tries to erase the transition, and the coercivity resists it. A tape
    with twice the coercivity holds a transition half as long.

        a = sqrt( Mr * delta * (d + delta/2) / (pi * Hc) )

    MEASURED ON THIS TAPE and it is not a small term. With Mr = 119366 A/m
    (0.15 T over mu_0), Hc = 47746 A/m (600 oersted), delta = 0.20 um and
    d = 0.05 um, `a` is 0.155 um - THREE TIMES the head-to-tape spacing -
    and the loss it causes at the luma carrier is 5.60 dB against the
    spacing loss's 1.81. It is the dominant short-wavelength loss in the
    chain and the model did not have it.

    The geometric cap by comparison binds only above 4.67 MHz, where
    `lambda / (2 pi)` finally drops below the coating thickness. So over
    the whole recorded band it is the material, not the geometry, that
    sets the limit.

    `level` scales the recorded magnetisation, which is the ONLY thing
    that separates this from a spacing error - see `transition_loss_db`.
    """
    mr = remanent_magnetisation(remanence_t) * max(float(level), 0.0)
    hc = max(float(coercivity_a_m), 1e-30)
    delta = float(thickness_m)
    return math.sqrt(max(float(factor), 0.0) * mr * delta
                     * (float(spacing_m) + 0.5 * delta) / hc)


def transition_loss_db(frequency_hz, writing_speed_m_s: float = 5.8709,
                       **material) -> np.ndarray:
    """The loss the transition length costs, in decibels.

    SAME FUNCTIONAL FORM AS SPACING - `exp(-2 pi a / lambda)`, so 54.575
    dB per wavelength of `a` exactly as Wallace's law gives for `d`. That
    means on the frequency axis alone it is DEGENERATE with spacing, and
    a fit offered both will split the total between them arbitrarily.

    WHAT SEPARATES THEM IS THE RECORD LEVEL. `a` goes as sqrt(Mr), and Mr
    is what the record current actually laid down, so `a` grows with level
    while the head-to-tape spacing does not care about it at all. That is
    the axis `magnetic.level_axis` already exists to explore, and it is
    what turns coercivity from an unusable parameter into a measurable
    one.

    A SECOND SEPARATOR, weaker but real: `a` is a property of the TAPE and
    `d` of the transport, so a second recording on the same deck holds `a`
    and varies `d`, while a second playback of one recording holds `a` and
    varies `d` again. Only the level varies `a` alone.
    """
    a = transition_length(**material)
    lam = wavelength_m(frequency_hz, writing_speed_m_s)
    return (20.0 * np.log10(np.e) * 2.0 * np.pi) * a / np.maximum(lam, 1e-30)


def coercivity_constrains(coercivity_a_m: float = 600.0 * 1000.0
                          / (4.0 * math.pi),
                          remanence_t: float = 0.15,
                          thickness_m: float = 0.20e-6,
                          spacing_m: float = 0.05e-6,
                          writing_speed_m_s: float = 5.8709,
                          carrier_hz: float = 3.9e6) -> Dict[str, object]:
    """Everything the coercivity fixes, in one place.

    Ethan asked whether the tape's coercivity can further define the
    model. It can, in three separate ways, and only the first was in the
    tree:

      1. THE RECORD DRIVE. H must reach Hc for the medium to switch at
         all, so `field_intensity(N, I, l) >= Hc` is the condition on the
         record current - and with N/l known it gives the current itself.
      2. THE TRANSITION LENGTH, above. The dominant short-wavelength loss,
         5.60 dB at the luma carrier, previously absent.
      3. THE SHORTEST RECORDABLE WAVELENGTH - and here the answer is that
         the material does NOT bind. Where `a` approaches `lambda/(2 pi)`
         the transition fills a half cycle and nothing survives, which for
         this tape is 0.97 um; the geometric cap `lambda/(2 pi) >= delta`
         bites first, at 1.26 um. So the coating's thickness limits the
         bandwidth and the coercivity does not.

    THE DISTINCTION MATTERS AND I HAD IT BACKWARDS AT FIRST. Coercivity's
    contribution is a LOSS, not a limit: 5.60 dB at the carrier, present
    at every wavelength in the band, rather than a wall at the end of it.
    A tape with twice the coercivity would not record higher frequencies -
    the coating still stops it there - it would record the SAME
    frequencies more strongly.
    """
    a = transition_length(coercivity_a_m, remanence_t, thickness_m,
                          spacing_m)
    mr = remanent_magnetisation(remanence_t)
    lam_carrier = writing_speed_m_s / max(float(carrier_hz), 1.0)
    # where the transition fills a half cycle: a = lambda / (2 pi)
    lam_limit = 2.0 * math.pi * a
    geometric_limit = 2.0 * math.pi * float(thickness_m)
    return {
        "transition_length_m": a,
        "material_lever": mr / max(float(coercivity_a_m), 1e-30),
        "record_field_needed_a_m": float(coercivity_a_m),
        "loss_at_carrier_db": float(
            (20.0 * np.log10(np.e) * 2.0 * np.pi) * a / lam_carrier),
        "shortest_wavelength_m": lam_limit,
        "shortest_wavelength_hz": writing_speed_m_s / max(lam_limit, 1e-30),
        "geometric_limit_m": geometric_limit,
        "material_limit_binds_first": bool(lam_limit > geometric_limit),
        "degenerate_with": "spacing - identical exp(-2 pi a / lambda) form",
        "separated_by": "the record level, which scales Mr and so scales a "
                        "while leaving the head-to-tape spacing untouched",
    }


# --------------------------------------------------------------------------
# The head as a transformer, and what the core's shape decides
# --------------------------------------------------------------------------

def head_transformer(turns: float = 100.0, core_length_m: float = 2e-3,
                     core_area_m2: float = 58e-6 * 20e-6,
                     core_permeability_r: float = 3000.0,
                     core_saturation_t: float = 0.45,
                     gap_m: float = 0.30e-6,
                     winding_resistance_ohm: float = 5.0,
                     tape_coercivity_a_m: float = 600.0 * 1000.0
                     / (4.0 * math.pi)) -> Dict[str, object]:
    """THE HEAD IS A TRANSFORMER, and modelling it as one is what makes the
    core's shape a parameter rather than an omission.

    Ethan: "Ferromagnetic core shape determines saturation as the frequency
    and amplitude response of the correct magnetic property... Kind of like
    a transformer."

    It is exactly a transformer with an unusual secondary. On RECORD the
    coil is the primary and the TAPE is the secondary - the gap is where
    the flux is delivered, and the tape carries it away as magnetisation.
    On PLAYBACK they swap: the tape's remanent flux is the primary and the
    coil is the secondary, which is Faraday and is why the output
    differentiates.

    WHAT THE CORE'S SHAPE DECIDES, and it decides all of it:

      * `A` sets the flux density for a given flux, `B = Phi / A`, so a
        narrower core saturates at less drive;
      * `l / mu` sets the core's share of the reluctance, so the shape
        decides whether the GAP or the CORE dominates - measured earlier
        in `reluctance`, the gap wins only where mu_r > l/g, which is
        6667 for a 2 mm core against a 0.30 um gap;
      * and that share decides whether core saturation reaches the output
        at all. A gap-dominated circuit is linearised by its gap: the
        gap's reluctance does not saturate, so it swamps the core's
        collapse. A core-dominated one passes the saturation straight
        through.

    THE INDUCTANCE IS THE COUPLING: `L = N^2 / R`. It is the same
    reluctance the magnetic side uses, so the electrical and magnetic
    descriptions are one model and not two - which is the whole value of
    the transformer view.

    THE HEADROOM IS THE DESIGN INTENT. With the values defaulted here the
    core saturates at 3.46 mA while the tape reaches its coercivity at
    0.46 mA - 7.5 times the headroom - so the tape saturates and the core
    stays linear, which is what a record head is for. That ratio is
    entirely a property of the core's shape, and it is the number a
    datasheet would settle.
    """
    permeability = float(core_permeability_r) * VACUUM_PERMEABILITY
    core_r = reluctance(core_length_m, core_area_m2, permeability)
    gap_r = reluctance(gap_m, core_area_m2)
    total = core_r + gap_r
    inductance = float(turns) ** 2 / max(total, 1e-300)
    saturation_flux = float(core_saturation_t) * float(core_area_m2)
    current_to_saturate = saturation_flux * total / max(float(turns), 1e-30)
    # what the tape needs: the gap field must reach the tape's coercivity
    flux_for_tape = (float(tape_coercivity_a_m) * VACUUM_PERMEABILITY
                     * float(core_area_m2))
    current_for_tape = flux_for_tape * total / max(float(turns), 1e-30)
    return {
        "inductance_h": inductance,
        "core_reluctance": core_r,
        "gap_reluctance": gap_r,
        "total_reluctance": total,
        "gap_share": gap_r / max(total, 1e-300),
        "gap_dominates": gap_r > core_r,
        "current_to_saturate_core_a": current_to_saturate,
        "current_for_tape_coercivity_a": current_for_tape,
        "headroom": current_to_saturate / max(current_for_tape, 1e-30),
        "electrical_corner_hz": float(winding_resistance_ohm)
        / (2.0 * math.pi * max(inductance, 1e-30)),
        "saturation_reaches_the_output": bool(core_r > gap_r),
        "why": "the gap's reluctance cannot saturate, so a gap-dominated "
               "circuit is linearised by its own gap and a core-dominated "
               "one passes the core's collapse straight to the output",
    }


def saturation_response(drive_a, frequency_hz, turns: float = 100.0,
                        core_length_m: float = 2e-3,
                        core_area_m2: float = 58e-6 * 20e-6,
                        core_permeability_r: float = 3000.0,
                        core_saturation_t: float = 0.45,
                        gap_m: float = 0.30e-6,
                        winding_resistance_ohm: float = 5.0
                        ) -> Dict[str, np.ndarray]:
    """THE AMPLITUDE AND FREQUENCY RESPONSE, COUPLED - Ethan's actual claim.

    "Ferromagnetic core shape determines saturation AS THE FREQUENCY AND
    AMPLITUDE RESPONSE". They are one surface, not two curves, and the
    mechanism is short:

        drive rises -> the core's B rises -> its DIFFERENTIAL permeability
        falls (`differential_permeability`, the sech^2 collapse) -> the
        core's reluctance rises -> L = N^2/R falls -> the winding's
        electrical corner R_e / (2 pi L) MOVES UP.

    So the frequency response is a function of the amplitude, through the
    core alone. A head driven into its core has a different bandwidth from
    the same head driven gently - and nothing about the tape, the spacing
    or the gap changes.

    Returned as a (drive x frequency) surface, because that is what the
    quantity is; collapsing it to one curve at one drive is what hides the
    coupling.
    """
    drives = np.atleast_1d(np.asarray(drive_a, dtype=np.float64))
    f = np.atleast_1d(np.asarray(frequency_hz, dtype=np.float64))
    gap_r = reluctance(gap_m, core_area_m2)
    initial = float(core_permeability_r) * VACUUM_PERMEABILITY
    # the core's operating field, from the flux the drive produces at the
    # SMALL-SIGNAL reluctance - one step rather than a solve, which is
    # honest at these headrooms and stated so
    corners, inductances, permeabilities = [], [], []
    for drive in drives:
        core_r = reluctance(core_length_m, core_area_m2, initial)
        flux = float(drive) * float(turns) / max(core_r + gap_r, 1e-300)
        b = flux / max(float(core_area_m2), 1e-30)
        h = b / max(initial, 1e-30)
        mu_d = float(differential_permeability(
            h, float(core_saturation_t),
            float(core_saturation_t) / max(initial, 1e-30)))
        mu_d = max(mu_d, initial * 1e-6)
        saturated_r = reluctance(core_length_m, core_area_m2, mu_d) + gap_r
        inductance = float(turns) ** 2 / max(saturated_r, 1e-300)
        inductances.append(inductance)
        permeabilities.append(mu_d / VACUUM_PERMEABILITY)
        corners.append(float(winding_resistance_ohm)
                       / (2.0 * math.pi * max(inductance, 1e-30)))
    corners = np.asarray(corners)
    # a single electrical pole per drive: the surface itself
    surface = 1.0 / np.sqrt(1.0 + (f[None, :] / corners[:, None]) ** 2)
    return {
        "drive_a": drives,
        "frequency_hz": f,
        "response": surface,
        "corner_hz": corners,
        "inductance_h": np.asarray(inductances),
        "core_permeability_r": np.asarray(permeabilities),
        "corner_moved": float(corners[-1] / corners[0]) if len(corners) > 1
        else 1.0,
    }


def core_shape_sets_saturation(core_area_m2=(1e-9, 1.16e-9, 4e-9),
                               core_length_m: float = 2e-3,
                               turns: float = 100.0,
                               core_saturation_t: float = 0.45,
                               gap_m: float = 0.30e-6) -> Dict[str, object]:
    """The shape, swept - which is the property Ethan names as the one to
    model.

    Two levers and they pull against each other. A LARGER AREA raises the
    flux the core carries before saturating, so it takes more drive; it
    also LOWERS both reluctances, which raises the inductance and lowers
    the electrical corner. A LONGER PATH raises the core's reluctance and
    so its share, which pushes the circuit from gap-dominated toward
    core-dominated and makes the saturation visible at the output.

    So "core shape determines saturation as the frequency and amplitude
    response" is two statements: the area sets WHERE saturation happens
    and the length sets WHETHER it reaches the output.
    """
    rows = []
    for area in np.atleast_1d(np.asarray(core_area_m2, dtype=np.float64)):
        head = head_transformer(turns=turns, core_length_m=core_length_m,
                                core_area_m2=float(area),
                                core_saturation_t=core_saturation_t,
                                gap_m=gap_m)
        rows.append({
            "core_area_m2": float(area),
            "inductance_h": head["inductance_h"],
            "current_to_saturate_core_a": head["current_to_saturate_core_a"],
            "headroom": head["headroom"],
            "gap_share": head["gap_share"],
            "saturation_reaches_the_output":
                head["saturation_reaches_the_output"],
        })
    return {
        "rows": rows,
        "area_sets": "where saturation happens - the flux density is "
                     "Phi/A, so a narrower core saturates at less drive",
        "length_sets": "whether it reaches the output - the core's share "
                       "of the reluctance, and the gap's share cannot "
                       "saturate",
    }


# --------------------------------------------------------------------------
# The loop's area: how lossy the magnetics are
# --------------------------------------------------------------------------

def hysteresis_loss(saturation_t: float, coercivity_a_m: float,
                    drive_a_m: Optional[float] = None) -> Dict[str, float]:
    """THE AREA OF THE B-H LOOP, which IS the energy lost per cycle.

    Ethan: "Add up the area of the BH curve to form how lossy the magnetic
    BH curve is, which itself may be derived or explained by a constant by
    transforming that over the specification's area."

    Both halves check out, and the second is exact.

    THE AREA IS THE LOSS, with no modelling in between - `area = closed
    integral of H dB` is joules per cubic metre per cycle, a
    thermodynamic identity rather than an analogy. Every traversal of the
    loop dissipates it.

    AND THE CONSTANT IS DERIVABLE. For the displaced-tanh loop the major
    loop's area has a closed form. With the branches at plus and minus Hc,

        area = Bs Hc INT[ tanh(u+1) - tanh(u-1) ] du
             = Bs Hc [ ln cosh(u+1) - ln cosh(u-1) ] from -inf to +inf
             = 4 Bs Hc

    because that bracket tends to +2 and -2 at the two ends. The
    SPECIFICATION'S OWN AREA is the rectangle its two published numbers
    define, `4 Hc Br`, so

        area / specification area = Bs / Br = 1 / tanh(1) = 1.313035

    a pure number and nothing else - the RECIPROCAL SQUARENESS. Verified
    against numerical integration to 2.2e-16 relative. That is exactly the
    "constant by transforming that over the specification's area" Ethan
    predicted, and its value is set by the loop's shape alone, not by this
    tape's figures.

    THE AREA SATURATES WITH DRIVE, which is why `drive_a_m` exists: at half
    the coercivity the loop encloses 0.48 of the specification rectangle,
    at Hc 0.87, at 2 Hc 1.23, and past about 3 Hc the branches have
    converged and driving harder adds nothing. That plateau IS the major
    loop, and an FM record head sits well inside it.
    """
    bs = float(saturation_t)
    hc = max(float(coercivity_a_m), 1e-30)
    remanence = bs * math.tanh(1.0)
    major = 4.0 * bs * hc
    specification = 4.0 * hc * remanence
    if drive_a_m is None:
        area = major
    else:
        grid = np.linspace(-abs(float(drive_a_m)), abs(float(drive_a_m)),
                           20001)
        area = float(np.trapezoid(
            hysteresis_branch(grid, bs, hc, ascending=False)
            - hysteresis_branch(grid, bs, hc, ascending=True), grid))
    return {
        "area_j_per_m3_per_cycle": area,
        "major_loop_area": major,
        "specification_area": specification,
        "loss_constant": area / max(specification, 1e-300),
        "major_loss_constant": major / max(specification, 1e-300),
        "squareness": remanence / bs,
        "remanence_t": remanence,
        "at_the_major_loop": drive_a_m is None
        or abs(float(drive_a_m)) >= 3.0 * hc,
    }


def loss_power_w_per_m3(saturation_t: float, coercivity_a_m: float,
                        frequency_hz, drive_a_m: Optional[float] = None
                        ) -> np.ndarray:
    """The loop area is per CYCLE, so the power is the area times the rate.

    Which is the whole of Steinmetz's law at fixed amplitude: `P = k f
    B^n`. The area supplies `k B^n` and this supplies the `f`. It matters
    here because the luma carrier runs at 3.4 to 4.4 MHz - the medium
    traverses its loop millions of times a second - so a loss that is
    negligible per cycle is not negligible per second.
    """
    loop = hysteresis_loss(saturation_t, coercivity_a_m, drive_a_m)
    return (loop["area_j_per_m3_per_cycle"]
            * np.asarray(frequency_hz, dtype=np.float64))


def record_head_derivable() -> Dict[str, object]:
    """WHAT OF THE RECORDING HEAD IS IN THE SIGNAL, since we read what it
    wrote.

    Ethan: "on the record side, we should also be able to derive that set
    of video heads measurements, since we are reading the signal it wrote."

    HE IS RIGHT AND IT REFINES WHAT `record_side_observability` SAYS. That
    function's claim - amplitude yes, phase no - is about the head's
    TIME-VARYING profile. The STATIC geometry is a different matter: the
    record head's gap, its spacing and its azimuth all shaped the
    magnetisation pattern that is still on the tape, so their signature is
    in every playback.

    BUT IT ARRIVES AS A PRODUCT, NOT AS A TERM. What reaches the
    demodulator is the record head's transfer times the tape's times the
    playback head's, and a single playback measures only that product. So
    every geometric quantity is recovered for the PAIR and never for
    either head - which is exactly why `head_model`'s fitted constants are
    documented as a property of the (recording, playback) pair rather than
    of a machine.

    WHAT WOULD SEPARATE THEM, in ascending order of how easy it is:

      1. A SECOND PLAYBACK DECK on the same tape. The record term is held
         and the playback term varies, so the difference is pure playback.
      2. A SECOND TAPE recorded on the same deck. The record term is held
         across both and the tape term varies.
      3. THE HEAD SWITCH, which is free and already in every decode: the
         two record heads wrote alternate fields with their own
         geometries, so the field-to-field difference is a pure
         RECORD-SIDE contrast with the playback path common to both -
         except that the two playback heads also alternate, so what the
         switch actually gives is the difference of differences. That is
         the four-way contrast the tesseract fold already measures.

    MEASURED, so that "a PRODUCT" is a number rather than a caution. On the
    matched taps of one Sony SLV-778HF, 75bars SP, the difference between
    the two heads comes out as 37.22 +/- 0.13 nm of separation by the
    amplitude route and 57.73 +/- 1.94 nm by the timing route - see
    `two_band_delay_pair`. That is a real and well determined difference
    between the drum's two heads, and it is a difference of PAIRS: head A's
    write and read against head B's write and read. On one deck the head
    that wrote a track is the head that reads it, so no arrangement of this
    capture set parts the two, and the separators above are not optional
    refinements but the only route.
    """
    return {
        "static_geometry": "IN THE SIGNAL - gap, spacing and azimuth "
                           "shaped the magnetisation that is still there",
        "time_varying_profile": "NOT recoverable - it is baked into the "
                                "medium as an absence, which is why "
                                "record-side dropouts cannot be re-read",
        "recovered_as": "a PRODUCT with the tape's and the playback head's "
                        "transfer, never as a separate term - hence "
                        "head_model's constants being a property of the "
                        "(recording, playback) PAIR",
        "separators": ("a second playback deck on the same tape",
                       "a second tape recorded on the same deck",
                       "the head switch, which gives a difference of "
                       "differences because both sides alternate"),
        "already_measured_by": "tesseract's four-way contrast, head x "
                               "polarity x half x tape",
    }


# --------------------------------------------------------------------------
# The two on-tape bands as a pair of measurements
# --------------------------------------------------------------------------

def two_band_pair(luma_hz=(3.4e6, 4.4e6), colour_under_hz=(0.4e6, 0.9e6),
                  writing_speed_m_s: float = 5.8709,
                  effective_spacing_m: float = 0.2045e-6,
                  points: int = 300) -> Dict[str, object]:
    """THE PAIR OF ON-TAPE SIGNALS AS TWO MEASUREMENTS - Ethan's construction.

    "Since colour under and luma occupy different bands on the tape, but
    are sourced from the same signal, this pair of signals on the tape are
    used to form the two on tape measurements needed for the flux density
    and the other per head magnetic properties."

    IT WORKS, AND FOR EXACTLY TWO UNKNOWNS. The playback voltage is

        e(f) = 2 pi f * (N B A) * L(f)

    - the Faraday rise, a flux amplitude `N B A` that is FLAT in frequency,
    and a loss `L(f)` that is not. Over one band the rise and the roll-off
    trade against each other and the two cannot be told apart; over the
    eleven-fold wavelength lever between 629 kHz and 4 MHz they can.

    MEASURED, fitting the flux scale and the effective spacing together:

        luma band alone     cond 27.0   coherence 0.997   sigma 186 nm
        colour-under alone  cond  9.1   coherence 0.976   sigma 373 nm
        BOTH BANDS          cond  3.1   coherence 0.811   sigma  23 nm

    An EIGHT-FOLD better error bar on the spacing, and the coherence falls
    from 0.997 to 0.811 - which is the difference between two names for one
    quantity and two quantities.

    AND IT DOES NOT EXTEND TO MORE THAN TWO. Offered spacing, gap,
    thickness and the transition length, the second band changes almost
    nothing: the effective rank stays at 1.01 of 4 and the conditioning
    improves eight-fold without resolving a single extra direction. The
    reason is exact rather than statistical - the head-to-tape spacing and
    the recorded transition length enter as the SAME FUNCTION,
    `exp(-2 pi x / lambda)`, so they are one parameter and no bandwidth
    whatever separates them. Gap against thickness sits at 0.983 even with
    both bands.

    So the pair buys a scale and a loss, well determined. It does not buy
    the head's geometry term by term, and a fit offered four names will
    still split one measurement between them.

    AND ALL OF THAT IS AMPLITUDE. Everything above is a Jacobian of a log
    MAGNITUDE, so the numbers describe how well two levels determine a
    spacing and say nothing about time. The two bands carry a second and
    independent reading of the same spacing in their relative TIMING, which
    `band_delay_difference` derives and `two_band_delay_pair` puts beside
    this one. The timing route has a property this one lacks: the flux scale
    `N B A` is flat in frequency and therefore contributes exactly nothing
    to a group delay, so the delay reads the spacing with no nuisance
    parameter to marginalise over at all. Measured against the 23.2 nm this
    function reaches with both bands, a delay known to 4.6 ns would match it
    and one known to a nanosecond would beat it four-fold.
    """
    def band(edges):
        return np.linspace(float(edges[0]), float(edges[1]), int(points))

    def design(grid, scale, spacing):
        lam = float(writing_speed_m_s) / grid
        step_scale, step_spacing = scale * 1e-5, spacing * 1e-5

        def logged(sc, sp):
            return (np.log(sc) + np.log(2.0 * np.pi * grid)
                    - 2.0 * np.pi * sp / lam)

        return np.column_stack([
            (logged(scale + step_scale, spacing)
             - logged(scale - step_scale, spacing)) / (2.0 * step_scale),
            (logged(scale, spacing + step_spacing)
             - logged(scale, spacing - step_spacing)) / (2.0 * step_spacing)])

    luma, cu = band(luma_hz), band(colour_under_hz)
    out = {}
    for name, grid in (("luma", luma), ("colour_under", cu),
                       ("both", np.concatenate([cu, luma]))):
        jacobian = design(grid, 1e-9, float(effective_spacing_m))
        normalised = jacobian / np.linalg.norm(jacobian, axis=0, keepdims=True)
        values = np.linalg.svd(normalised, compute_uv=False)
        covariance = np.linalg.inv(jacobian.T @ jacobian)
        out[name] = {
            "condition": float(values[0] / values[-1]),
            "coherence": abs(float(normalised[:, 0] @ normalised[:, 1])),
            "spacing_sigma_m": float(np.sqrt(covariance[1, 1])),
        }
    lever = ((writing_speed_m_s / colour_under_hz[0])
             / (writing_speed_m_s / luma_hz[1]))
    out["wavelength_lever"] = float(lever)
    out["improvement"] = (out["luma"]["spacing_sigma_m"]
                          / max(out["both"]["spacing_sigma_m"], 1e-30))
    out["separates"] = ("the flux amplitude N*B*A, which is flat in "
                        "frequency", "the loss, which is not")
    out["does_not_separate"] = (
        "spacing from the transition length - they are the SAME function "
        "exp(-2 pi x / lambda), so they are one parameter at any bandwidth")
    return out


# --------------------------------------------------------------------------
# The two bands on the TIME axis: the head's delay between luma and chroma
# --------------------------------------------------------------------------

# The coherence below which a band's PHASE is not measured at all, derived
# rather than chosen. The random phase error of a transfer estimated from n
# averages at coherence gamma has standard deviation
# `sqrt((1 - gamma^2) / (2 n gamma^2))` radians, so at n = 1 the error
# reaches one radian - the point past which a phase is no longer a
# measurement of anything - when `gamma^2 = 1/3`. Nothing below this can
# contribute to a delay however many fields are averaged, because the
# averaging is already inside the estimate the instrument reports.
PHASE_COHERENCE_FLOOR = 1.0 / math.sqrt(3.0)


def spec_band_centres() -> Dict[str, object]:
    """The two on-tape bands, taken from the standard rather than declared.

    SMPTE 32M gives the luma FM carrier by its two reference points - sync
    tip at 3.4 MHz and peak white at 4.4 MHz, clause 3.9.1.1.4 - and the
    colour-under carrier as forty times the line rate, 629.371 kHz, clause
    3.9.2.1.4. Those three figures and the standard's own 5.80 m/s writing
    speed are the whole of this module's time axis; nothing here is fitted
    and nothing is chosen.

    TWO CENTRES, BECAUSE THE TWO LAWS WEIGHT THE BAND DIFFERENTLY. Wallace's
    log magnitude is linear in frequency, so a band average of it lands on
    the band's ARITHMETIC centre; the group delay the same loss implies is
    linear in the LOGARITHM of frequency, so a band average of that lands on
    the geometric one. Using one centre for both is a small error and an
    avoidable one.

    Measured from the standard's figures: the luma band's arithmetic centre
    is 3.900 MHz and its geometric centre 3.868 MHz, against a colour-under
    carrier of 0.629 MHz - a frequency ratio of 6.146 and a wavelength lever
    of the same, which is what makes two bands worth more than one.
    """
    from vhsdecode.models import vhs_specification as spec

    low = float(spec.value_of("fm_sync_tip_hz"))
    high = float(spec.value_of("fm_peak_white_hz"))
    colour = float(spec.value_of("colour_under_hz"))
    speed = float(spec.value_of("writing_speed_m_s"))
    linear = 0.5 * (low + high)
    geometric = math.sqrt(low * high)
    return {
        "luma_low_hz": low,
        "luma_high_hz": high,
        "luma_linear_centre_hz": linear,
        "luma_geometric_centre_hz": geometric,
        "colour_under_hz": colour,
        "writing_speed_m_s": speed,
        "frequency_ratio": geometric / colour,
        "frequency_difference_hz": linear - colour,
        "clauses": "3.9.1.1.4 (luma), 3.9.2.1.4 (colour under), "
                   "3.1.4 (writing speed)",
    }


def band_delay_difference(spacing_m: float, low_hz: Optional[float] = None,
                          high_hz: Optional[float] = None,
                          writing_speed_m_s: Optional[float] = None) -> float:
    """THE DELAY BETWEEN THE TWO BANDS THAT A HEAD SEPARATION PRODUCES.

    Ethan: *"The difference in time between the luma and chroma bands are
    the measurement we can use to observe the delay on each video head."*
    This is that quantity in closed form, and it is exact:

        tau(f2) - tau(f1) = -(2 d / (pi v)) ln(f2 / f1)

    WHERE IT COMES FROM. Wallace's separation loss is `exp(-2 pi d / lambda)`
    and therefore `log|H| = -(d / v) omega` - linear in frequency. A loss is
    minimum phase, so its phase is the Hilbert transform of its log
    magnitude, and the Hilbert transform of `|omega|` is `omega ln|omega|` up
    to a linear term. Differentiating gives a group delay that is
    LOGARITHMIC in frequency, `tau = -(2 d / (pi v)) ln omega + constant`,
    and the unknown constant - which is the arbitrary reference the Hilbert
    transform of an unbounded function leaves behind - cancels the moment two
    frequencies are subtracted. That cancellation is the whole reason the
    measurement works.

    AND IT IS WHY THIS IS MEASURABLE AT ALL. Two captures with no common
    clock cannot measure an absolute delay: any instrument comparing them
    must remove a linear phase, which removes a constant from the group
    delay at every frequency alike. A DIFFERENCE BETWEEN TWO BANDS is
    invariant under exactly that removal. So the one time quantity the
    apparatus can see is the one the physics puts information into.

    THE SIGN IS AN ADVANCE, not a delay. A separation loss is a low pass and
    a minimum-phase low pass has its largest group delay at its low end, so
    the LUMA BAND ARRIVES EARLIER than the colour-under band. Measured
    through this function at the standard's bands and this project's typical
    effective spacing of 0.2045 um, the luma band leads by 40.757 ns - which
    is 0.583 of a sample at 4fsc, so an instrument that cannot interpolate
    below a sample cannot see it.

    THE TREE'S OTHER ROUTE TO THIS NUMBER DOES NOT WORK, and that is why
    this function exists. `head_model.group_delay_s` takes the Hilbert
    transform of the log magnitude on whatever grid the caller passes, and a
    discrete Hilbert transform assumes its input is periodic. A separation
    loss is a straight line, so the wrap-around at the grid's ends dominates
    the answer. Measured on ONE head - 0.2045 um at 5.80 m/s - asking that
    function for the same 3.9 MHz against 0.629 MHz difference:

        grid                       answer
        0.2-7 MHz, 1024 points     +165.5 ns
        0.2-7 MHz, 4096 points     +166.9 ns
        0.5-7 MHz, 1024 points     +539.7 ns
        0.2-12 MHz, 1024 points    +114.8 ns
        0.05-20 MHz, 4096 points    -34.3 ns
        this closed form            -40.8 ns

    The sign itself depends on the window. Only the widest grid comes near
    the truth, and no caller knows to ask for one.

    THE CLOSED FORM IS VERIFIED, on a control that can fail. Reconstructing
    the minimum phase of a first-order minimum-phase filter `1 - r z^-1`,
    whose phase is known exactly, the cepstral method used for the check
    agrees to 2.2e-15 radians at r = 0.95. Put the separation loss through
    that same machinery and the band-to-band group delay difference comes
    back as -(2a/pi) ln(w2/w1) to within 0.372 per cent, and the ratio is
    1.00372 for a = 0.5, 1.0 and 2.0 alike - so the residual is the grid's
    and not the law's, which is what a scaling check is for.
    """
    bands = spec_band_centres()
    low = bands["colour_under_hz"] if low_hz is None else float(low_hz)
    high = (bands["luma_geometric_centre_hz"] if high_hz is None
            else float(high_hz))
    speed = (bands["writing_speed_m_s"] if writing_speed_m_s is None
             else float(writing_speed_m_s))
    if low <= 0 or high <= 0:
        raise ValueError("a band centre must be a positive frequency")
    return -(2.0 * float(spacing_m) / (math.pi * speed)) * math.log(high / low)


def spacing_from_band_delay(delay_difference_s: float,
                            delay_error_s: float = 0.0,
                            low_hz: Optional[float] = None,
                            high_hz: Optional[float] = None,
                            writing_speed_m_s: Optional[float] = None
                            ) -> Dict[str, object]:
    """THE ESTIMATOR: a measured two-band delay, read back as a separation.

    The inverse of `band_delay_difference`,

        d = -(pi v / 2) * (tau(f2) - tau(f1)) / ln(f2 / f1)

    and its sensitivity, which is the number an instrument builder needs
    before building anything. At the standard's own bands and writing speed
    one nanosecond of delay is 5.018 nanometres of separation, so the whole
    of this project's typical 0.2045 um effective spacing is 40.8 ns and the
    difference between two heads of one drum - measured below at about 37 nm
    - is 7.4 ns, which is 0.106 of a sample at 4fsc.

    WHAT IT MEASURES IS A SUM AND NOT A HEAD. The separation enters once
    when the signal is written and again when it is read, and both are the
    same exponential, so what comes back is the record-side and playback-side
    separations added - the confound `tape_path.clearance_signature` records
    and `record_head_derivable` explains. The quantity that IS a head is the
    difference between the two heads of one drum, where everything shared -
    the tape, the record electronics, the preamplifier, the cable - cancels.
    """
    bands = spec_band_centres()
    low = bands["colour_under_hz"] if low_hz is None else float(low_hz)
    high = (bands["luma_geometric_centre_hz"] if high_hz is None
            else float(high_hz))
    speed = (bands["writing_speed_m_s"] if writing_speed_m_s is None
             else float(writing_speed_m_s))
    lever = math.log(high / low)
    metres_per_second = math.pi * speed / (2.0 * lever)
    return {
        "spacing_m": -metres_per_second * float(delay_difference_s),
        "spacing_error_m": metres_per_second * abs(float(delay_error_s)),
        "metres_per_second_of_delay": metres_per_second,
        "log_lever": lever,
        "low_hz": low,
        "high_hz": high,
        "writing_speed_m_s": speed,
        "measures": "the record-side and playback-side separations ADDED; "
                    "only a difference between the two heads of one drum "
                    "isolates a head",
    }


def spacing_from_band_tilt(log_tilt_nepers: float,
                           tilt_error_nepers: float = 0.0,
                           low_hz: Optional[float] = None,
                           high_hz: Optional[float] = None,
                           writing_speed_m_s: Optional[float] = None
                           ) -> Dict[str, object]:
    """THE OTHER READING OF THE SAME NUMBER: the same separation, from the
    two bands' LEVELS instead of their timing.

    Wallace's law in the form the amplitude axis uses it,

        log|H(f2)| - log|H(f1)| = -(2 pi d / v) (f2 - f1)

    so a measured tilt between the two bands gives the same `d` that the
    delay does. This is the estimator the amplitude side of `two_band_pair`
    already implies; it is written out here so that the two routes can be
    put beside one another, which is the whole point of a pair.

    Its sensitivity at the standard's bands and writing speed is 282.2
    nanometres per neper, so this project's typical 0.2045 um effective
    spacing is a tilt of 0.7245 nepers, or 6.294 dB, between the two bands.
    Matching a one-nanosecond delay measurement therefore takes a level
    measurement good to 0.0178 nepers, which is 0.154 dB.

    THE TWO ROUTES ARE NOT THE SAME MEASUREMENT WEARING TWO HATS. One is a
    difference of levels and the other a difference of slopes of phase; one
    responds to every loss in the path and the other only to the part that
    is minimum phase; and the frequency dependence differs - `f2 - f1`
    against `ln(f2 / f1)` - so they weight the band differently even when
    they agree.
    """
    bands = spec_band_centres()
    low = bands["colour_under_hz"] if low_hz is None else float(low_hz)
    high = (bands["luma_linear_centre_hz"] if high_hz is None
            else float(high_hz))
    speed = (bands["writing_speed_m_s"] if writing_speed_m_s is None
             else float(writing_speed_m_s))
    span = high - low
    if span == 0.0:
        raise ValueError("the two bands must differ in frequency")
    metres_per_neper = speed / (2.0 * math.pi * span)
    return {
        "spacing_m": -metres_per_neper * float(log_tilt_nepers),
        "spacing_error_m": metres_per_neper * abs(float(tilt_error_nepers)),
        "metres_per_neper": metres_per_neper,
        "frequency_span_hz": span,
        "low_hz": low,
        "high_hz": high,
        "writing_speed_m_s": speed,
    }


def two_band_delay_pair(delay_difference_s: float, log_tilt_nepers: float,
                        delay_error_s: float = 0.0,
                        tilt_error_nepers: float = 0.0,
                        coherence: Optional[float] = None,
                        delay_low_hz: Optional[float] = None,
                        delay_high_hz: Optional[float] = None,
                        tilt_low_hz: Optional[float] = None,
                        tilt_high_hz: Optional[float] = None,
                        writing_speed_m_s: Optional[float] = None
                        ) -> Dict[str, object]:
    """THE PAIR: the same separation read twice, once in time and once in
    amplitude, so that the two can disagree.

    A pair is not two views of one arithmetic. These two share a parameter
    and nothing else: the timing route reads the slope of the phase and the
    amplitude route the difference of the levels, and a discrepancy between
    them says the path is not the pure minimum-phase separation loss the
    model assumes. That is a finding rather than a failure, which is what a
    pair is for.

    MEASURED, ON REAL DATA, and this is the result. The instrument is
    `tap_transfer`'s record-tap to playback-tap complex transfer of one Sony
    SLV-778HF, per head, 75bars SP, 13 and 14 fields, colour-under band
    0.4-0.9 MHz against luma band 3.4-4.4 MHz:

        head 0   tau(luma) - tau(colour under)  +84.33 +/- 0.30 ns
                 log-magnitude tilt             +0.4752 +/- 0.0004 nepers
        head 1   tau(luma) - tau(colour under)  +72.80 +/- 0.25 ns
                 log-magnitude tilt             +0.3389 +/- 0.0003 nepers

    Neither head's own figure is the head: the transfer also carries the
    preamplifier's equalisation and the head's electrical resonance, both of
    which are large and positive where the separation loss is small and
    negative. The head is the DIFFERENCE, where everything the two heads
    share cancels:

        head 0 minus head 1, delay         +11.533 +/- 0.388 ns
        head 0 minus head 1, tilt          +0.13625 +/- 0.00046 nepers
        separation by the TIME route       -57.74 +/- 1.94 nm
        separation by the AMPLITUDE route  -37.22 +/- 0.13 nm

    Same sign, same order, ratio 1.551, and 10.5 standard deviations apart.
    So the two heads of this drum genuinely differ by something close to
    forty nanometres of separation - head 0 running the closer of the two -
    and that something is not ONLY separation, because a pure separation
    difference would have to give both numbers alike.

    AND ACROSS PATTERNS THE TWO ROUTES ARE NOT EQUALLY GOOD. Over the ten SP
    captures whose colour-under coherence clears the floor below:

        amplitude route   -39.97 nm, scatter   6.10 nm, range -52.5 to -31.3
        time route       -107.32 nm, scatter 203.60 nm, range -630.7 to +86.1

    - a scatter thirty-three times larger, and a route that changes sign
    from pattern to pattern. The reason is visible one level down: over the
    seven SP captures whose delay error bar is under five nanoseconds, head
    0's own colour-under group delay runs from -135.6 ns to -10.3 ns - a
    125 ns swing on one head from content alone, against the 11.5 ns that
    separates the two heads. These are luma-dominated test charts and not
    one of them holds a steady full-amplitude colour-under carrier. The time
    route is sound and the captures are not yet good enough for it.

    THE COHERENCE GATE IS LOAD-BEARING AND THE ERROR BAR IS NOT ENOUGH. The
    luma-only captures have no colour-under signal at all and their
    coherence there falls to 0.002-0.35, so they must be refused; but
    `pulseandbar-y-only`, at coherence 0.127, reports a head delay difference
    of +2229.70 +/- 104.29 ns - twenty-one standard deviations of pure
    nonsense. An instrument that trusted the reported error bar alone would
    have accepted it. `PHASE_COHERENCE_FLOOR` refuses it on the coherence,
    and the honest-but-noisy captures are then handled by their own error
    bars, which is what those are for.

    THE TWO ROUTES TAKE THEIR OWN BAND CENTRES and the arguments are named
    apart for that reason. A delay averaged over a band lands on its
    geometric centre and a level on its arithmetic one, because one law is
    linear in the logarithm of frequency and the other in frequency itself.
    Handing one pair of centres to both is a small, silent error - on the
    measurement above it moves the amplitude route from -37.22 to -38.59 nm,
    which is ten times its own error bar - and a single catch-all argument
    would have invited it.
    """
    delay = spacing_from_band_delay(delay_difference_s, delay_error_s,
                                    low_hz=delay_low_hz,
                                    high_hz=delay_high_hz,
                                    writing_speed_m_s=writing_speed_m_s)
    tilt = spacing_from_band_tilt(log_tilt_nepers, tilt_error_nepers,
                                  low_hz=tilt_low_hz, high_hz=tilt_high_hz,
                                  writing_speed_m_s=writing_speed_m_s)
    combined = math.hypot(delay["spacing_error_m"], tilt["spacing_error_m"])
    difference = delay["spacing_m"] - tilt["spacing_m"]
    out: Dict[str, object] = {
        "time_spacing_m": delay["spacing_m"],
        "time_spacing_error_m": delay["spacing_error_m"],
        "amplitude_spacing_m": tilt["spacing_m"],
        "amplitude_spacing_error_m": tilt["spacing_error_m"],
        "difference_m": difference,
        "difference_error_m": combined,
        "sigma": abs(difference) / combined if combined > 0 else float("inf"),
        "ratio": (delay["spacing_m"] / tilt["spacing_m"]
                  if tilt["spacing_m"] != 0.0 else float("nan")),
        "metres_per_second_of_delay": delay["metres_per_second_of_delay"],
        "metres_per_neper": tilt["metres_per_neper"],
        "usable": True,
        "why": "a discrepancy means the path is not the pure minimum-phase "
               "separation loss the model assumes, which is a finding",
    }
    if coherence is not None:
        out["coherence"] = float(coherence)
        out["coherence_floor"] = PHASE_COHERENCE_FLOOR
        if float(coherence) < PHASE_COHERENCE_FLOOR:
            out["usable"] = False
            out["why"] = (
                "the colour-under band's coherence is %.3f, below the %.3f at "
                "which a single average's phase error reaches one radian; the "
                "delay is not measured here whatever error bar is reported "
                "beside it" % (float(coherence), PHASE_COHERENCE_FLOOR))
    return out


# --------------------------------------------------------------------------
# THE RECORD SIDE: the field that magnetises the tape, and the layer it writes
# --------------------------------------------------------------------------
#
# Ethan asked for this twice and the module had only half of it:
#
#   "model the head in terms of voltage and current that is being used
#    magnetize the head"
#   "Head current to magnetic ratio is the number of turns divided by the
#    length of the coil ... Use the expected properties of this and use the
#    BH curve to model this part"
#
# What existed was the field inside the CORE - `field_intensity` gives
# `H = N I / l`, which is the coil's own field - and the core's B-H curve in
# `anhysteretic_b`. What did not exist was the step from there to the tape:
# the fraction of the magnetomotive force that appears across the gap, the
# field that fraction produces in the coating, and the depth to which that
# field exceeds the tape's coercivity and therefore records anything at all.
#
# That step is the whole of the record side. Without it the recording depth
# is a free scale - `magnetic.NOMINAL_DEPTH_M` says so in its own comment -
# and the record current, which SMPTE 32M defines only by what it achieves,
# has no model to achieve it in.


def head_efficiency(gap_m: float = 0.30e-6, core_length_m: float = 2e-3,
                    core_area_m2: float = 58e-6 * 20e-6,
                    core_permeability_r: float = 3000.0) -> Dict[str, float]:
    """THE FRACTION OF THE DRIVE THAT REACHES THE TAPE.

    A record head is a magnetic circuit in series: the core carries the flux
    round to the gap, and the two reluctances divide the magnetomotive force
    between them exactly as two resistors divide a voltage. The share that
    falls across the GAP is the head's efficiency,

        eta = R_gap / (R_gap + R_core)

    and the field in the gap - the deep-gap field, which is what the
    Karlqvist expression below is written in terms of - is `eta N I / g`.

    `head_transformer` has computed this number all along and called it
    `gap_share`, without ever naming it as the efficiency or using it to
    reach the tape. With the values that function defaults to it is 0.3103,
    so two thirds of the drive is spent turning the core and one third does
    the recording.

    THE EFFICIENCY IS NOT A LOSS TO BE CORRECTED. It is a design quantity:
    an efficiency near one would mean a core contributing no reluctance,
    which also means a core that saturates at once. `head_transformer`
    measures the consequence - with these values the core saturates at
    3.462 mA while the tape reaches its coercivity at 0.4615 mA, a headroom
    of 7.50 - and that headroom is the ceiling on everything below.
    """
    permeability = float(core_permeability_r) * VACUUM_PERMEABILITY
    core = reluctance(core_length_m, core_area_m2, permeability)
    gap = reluctance(gap_m, core_area_m2)
    total = core + gap
    return {
        "efficiency": gap / max(total, 1e-300),
        "core_reluctance": core,
        "gap_reluctance": gap,
        "why": "the core and the gap divide the magnetomotive force in "
               "proportion to their reluctances, and only the gap's share "
               "reaches the tape",
    }


def deep_gap_field_a_m(current_a: float, turns: float = 100.0,
                       gap_m: float = 0.30e-6, **head) -> float:
    """The field in the gap, `eta N I / g`, in amperes per metre. This is
    the one number the whole record side is written in terms of, and it is
    where the coil's current finally becomes a field the tape can feel."""
    eta = head_efficiency(gap_m=gap_m, **head)["efficiency"]
    return eta * float(turns) * float(current_a) / max(float(gap_m), 1e-30)


def karlqvist_field(x_m, y_m, gap_m: float = 0.30e-6,
                    deep_gap_field_a_m: float = 1.0) -> Dict[str, np.ndarray]:
    """THE FIELD THE RECORD HEAD PUTS INTO THE COATING, as a complex number.

    Karlqvist's approximation (O. Karlqvist, "Calculation of the magnetic
    field in the ferromagnetic layer of a magnetic drum", Transactions of
    the Royal Institute of Technology, Stockholm, number 86, 1954), the
    standard closed form for a gap of length `g` in an infinitely permeable
    head carrying a deep-gap field `H_g`, at a point `x` along the track and
    `y` below the head's surface:

        Hx = (H_g / pi) [ arctan((g/2 + x)/y) + arctan((g/2 - x)/y) ]
        Hy = -(H_g / 2 pi) ln[ ((x + g/2)^2 + y^2) / ((x - g/2)^2 + y^2) ]

    RETURNED COMPLEX, AND NOT AS A CONVENTION. A field with no sources and
    no currents in the region it occupies is both divergence-free and
    curl-free there, which is precisely the statement that `Hx - j Hy` is an
    ANALYTIC function of `x + j y`. The complex form is what the field is;
    the pair of real components is the shadow of it.

    THAT IS ALSO THE CONTROL, and it settles the sign of `Hy`, which
    different sources write differently. Measured at 0.07 um along the track
    and 0.22 um deep, with the sign above: `div H` is 1.3e-01 and `curl H`
    is 1.9e-01 against a derivative scale of 5.5e+05, a relative 2.4e-07
    which is the finite difference's own error. With the sign reversed both
    read 1.1e+06 and 2.4e+06 - the size of the terms themselves. A sign
    error here is not subtle and the control catches it at once.

    THE APPROXIMATION'S RANGE. Karlqvist assumes the gap's faces are
    equipotentials and is accurate for `y` greater than about half the gap.
    At the depths the format records - a fifth of a micrometre against a
    gap of three tenths - `y/g` is near 0.7, which is inside that range but
    not far inside, so the depths below are good to a few per cent and not
    to a part in a thousand.
    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    half = 0.5 * float(gap_m)
    field = float(deep_gap_field_a_m)
    along = (field / np.pi) * (np.arctan((half + x) / y)
                               + np.arctan((half - x) / y))
    across = -(field / (2.0 * np.pi)) * np.log(
        ((x + half) ** 2 + y ** 2) / ((x - half) ** 2 + y ** 2))
    return {
        "along_track_a_m": along,
        "perpendicular_a_m": across,
        "field": (along - 1j * across).astype(np.complex128),
        "magnitude_a_m": np.hypot(along, across),
    }


def write_depth_m(drive_ratio: float, gap_m: float = 0.30e-6) -> float:
    """HOW DEEP THE RECORDING GOES, in closed form.

    A coating records where the head's field exceeds its coercivity and
    nowhere else, so the recorded layer's lower boundary is the deepest
    point of the contour `|H| = Hc`. That contour's deepest point lies
    directly under the gap's centre - measured on a 3001 by 2001 grid it is
    at `x` = 0.0000 um - and there the perpendicular component vanishes by
    symmetry and the field is the along-track term alone:

        Hx(0, y) = (2 H_g / pi) arctan( (g/2) / y )

    Setting that equal to the coercivity and solving gives the depth in one
    line, with `r = H_g / Hc` the drive in units of the coercivity:

        y = (g / 2) / tan( pi / (2 r) )

    and it inverts as cleanly, which `drive_ratio_for_depth` does.

    MEASURED, at the format's 0.30 um gap, against the numerical contour on
    the full two-dimensional field: the closed form gives 0.2064 um at
    `r = 2.5` and the grid search 0.2063 um. The two agree because the
    argument above is exact rather than approximate.

    A drive at or below the coercivity records nothing at all, and the
    formula says so: at `r = 1` the tangent's argument is `pi/2` and the
    depth is zero.
    """
    ratio = float(drive_ratio)
    if ratio <= 1.0:
        return 0.0
    return 0.5 * float(gap_m) / math.tan(math.pi / (2.0 * ratio))


def drive_ratio_for_depth(depth_m: float, gap_m: float = 0.30e-6) -> float:
    """The drive, in units of the tape's coercivity, that records to a given
    depth. The inverse of `write_depth_m`,

        H_g / Hc = pi / (2 arctan( (g/2) / y ))

    and it is the function that turns the recording depth from a free scale
    into a record current. `magnetic.NOMINAL_DEPTH_M` calls its own 0.15 um
    "a SCALE, not a constant of the format"; measured through here, with the
    format's gap and this project's 0.05 um head-to-tape spacing, that
    layer corresponds to a drive of 2.441 times the coercivity, and
    `tape_model`'s 0.20 um recording depth to 2.907. The two numbers the
    tree assumed independently turn out to be one quantity at two record
    currents, which is the axis `magnetic.level_axis` measures the shape of.
    """
    depth = float(depth_m)
    if depth <= 0.0:
        return float("inf")
    return math.pi / (2.0 * math.atan(0.5 * float(gap_m) / depth))


def optimum_record_drive(frequency_hz: Optional[float] = None,
                         gap_m: float = 0.30e-6,
                         spacing_m: float = 0.05e-6,
                         writing_speed_m_s: Optional[float] = None
                         ) -> Dict[str, float]:
    """SMPTE 32M's OPERATIONAL DEFINITION OF THE RECORD CURRENT, IMPLEMENTED.

    Clause 3.9.1.1.6 states it and states nothing else: *"The record current
    shall be set to the optimum value over the entire bandwidth of the FM
    carrier. Optimum record current shall be that which returns the maximum
    output signal level during playback."* `vhs_specification` records that
    as a value of None - the standard gives no number, no units and no
    tolerance - and nothing in this tree has ever computed what the clause
    describes.

    It is computable, because the clause names a maximum and the maximum has
    a cause. Driving harder writes deeper, and depth stops helping at the
    point where the layer written is as deep as the reproduce head can see:
    magnetisation below `lambda / (2 pi)` returns no flux to the gap, and it
    does add to the transition's own demagnetising field. So the output
    peaks where the write depth meets the read depth, and the optimum drive
    follows from `drive_ratio_for_depth` at that depth plus the head-to-tape
    spacing, since the coating begins there and not at the head's surface.

    MEASURED, at the format's own figures - a 0.30 um gap, 0.05 um of
    separation, the standard's 5.80 m/s:

        band              wavelength   read depth   optimum H_gap / Hc
        sync tip 3.4 MHz    1.7059 um    0.2715 um        3.598
        luma centre         1.4996 um    0.2387 um        3.278
        peak white 4.4 MHz  1.3182 um    0.2098 um        3.000
        colour under        9.2156 um    1.4667 um       15.934

    AND THE LAST ROW EXPLAINS THE CLAUSE'S OWN WORDING. Across the FM
    carrier's band the optimum moves only from 3.000 to 3.598, so one
    current serves the whole of it and the standard can sensibly ask for
    one. The colour-under carrier's optimum is 15.9 - 4.9 times the luma's,
    and above the 7.50 of headroom `head_transformer` measures before the
    core itself saturates. The chroma's optimum is therefore not merely
    unchosen but unreachable, and the standard is right to define the
    optimum over the FM carrier alone and to say nothing of the chroma.

    AND THE REASON IT SAYS NOTHING IS THAT THE CHROMA IS NOT RECORDED THIS
    WAY AT ALL. The colour under is recorded on the luma FM's own AC bias
    (SMPTE 32M 7.5.1.2.2; JVC VTG82063 1.1.6, which also states the chroma
    record current is a CONSTANT current rather than an optimised one), so
    it is a small signal on the anhysteretic curve and never needs a field
    of its own that reaches the coercivity - 3.9.2.1.2 deliberately keeps it
    7 to 10 dB below saturation. The 15.9 is therefore not a requirement
    that fails: it is the drive a 9.2156 um wavelength WOULD need if it were
    recorded the way the luma is, 2.12 times the whole of the core's
    headroom, which is the quantitative reason the format had to bias it
    instead. See `tape_bias.record_current_scope`.
    """
    band = spec_band_centres()
    speed = (float(band["writing_speed_m_s"]) if writing_speed_m_s is None
             else float(writing_speed_m_s))
    frequency = (float(band["luma_geometric_centre_hz"])
                 if frequency_hz is None else float(frequency_hz))
    wavelength = speed / max(frequency, 1e-30)
    read = wavelength / (2.0 * math.pi)
    ratio = drive_ratio_for_depth(read + float(spacing_m), gap_m)
    return {
        "frequency_hz": frequency,
        "wavelength_m": wavelength,
        "read_depth_m": read,
        "drive_ratio": ratio,
        "write_depth_m": write_depth_m(ratio, gap_m),
        "clause": "SMPTE 32M 3.9.1.1.6",
        "why": "the output stops rising once the layer written is as deep "
               "as the reproduce head can see, which is lambda / (2 pi)",
    }


def recorded_layer(drive_ratio: float, gap_m: float = 0.30e-6,
                   spacing_m: float = 0.05e-6,
                   frequency_hz: Optional[float] = None,
                   writing_speed_m_s: Optional[float] = None
                   ) -> Dict[str, float]:
    """THE LAYER THAT ACTUALLY CARRIES THE SIGNAL, and it has two lids.

    The record side sets the lower boundary and the playback side sets what
    of it is visible, and the two obey different laws:

      THE WRITE DEPTH DOES NOT DEPEND ON FREQUENCY. The head's field knows
      the drive and the geometry and nothing about the signal, so every
      band recorded by one head at one current is written to the same depth.
      THE READ DEPTH DOES. `lambda / (2 pi)` is a wavelength, so the
      colour-under band reads six times deeper into the coating than the
      luma band does.

    So the layer carrying a band is the INTERSECTION, and which lid binds is
    a per-band question rather than a property of the tape. Measured at a
    drive of 3.000 times the coercivity - the value `optimum_record_drive`
    returns for peak white - with the format's 0.30 um gap and this
    project's 0.05 um separation, every depth in micrometres:

        band            write   read   layer   bound by
        sync tip       0.2598 0.2715  0.2098   write
        luma centre    0.2598 0.2387  0.2098   write
        peak white     0.2598 0.2098  0.2098   read
        colour under   0.2598 1.4667  0.2098   write

    Peak white is read-bound because this is exactly its own optimum, which
    is what an optimum means; everything at a longer wavelength is
    write-bound, and the last row is the finding. The colour-under band's
    own reading limit is 1.4667 um and there is nothing recorded below
    0.2598 um for it to read, so the chroma's effective thickness is set by
    the RECORD CURRENT and not by its wavelength at all. Any account that
    gives the colour under a 1.5 um recorded thickness is describing coating
    the head never magnetised.

    WHICH RECORD CURRENT, THOUGH - AND UNTIL NOW THIS FUNCTION COULD NOT
    SAY. One `drive_ratio` is applied to every row, which is right for the
    luma because the luma's current is what the argument names, and was
    unjustified for the colour under because the chroma has a current of its
    own. Bias supplies the justification: the anhysteretic process runs
    exactly where the BIAS field exceeds the coercivity, and the bias is the
    luma, so the layer carrying the chroma is the layer the LUMA wrote and
    the chroma's own current does not enter the depth at all. See
    `tape_bias.chroma_recorded_layer`.

    `margin_m` is the signed distance between the two lids, `read + spacing
    - write`, and it is the quantity rather than the label: negative means
    the reproduce head is the limit, positive means the record current is,
    and zero is the optimum. At the tie the label falls one way or the other
    on rounding and the margin does not.
    """
    band = spec_band_centres()
    speed = (float(band["writing_speed_m_s"]) if writing_speed_m_s is None
             else float(writing_speed_m_s))
    frequency = (float(band["luma_geometric_centre_hz"])
                 if frequency_hz is None else float(frequency_hz))
    write = write_depth_m(drive_ratio, gap_m)
    read = speed / max(frequency, 1e-30) / (2.0 * math.pi)
    top = float(spacing_m)
    bottom = min(write, read + top)
    return {
        "frequency_hz": frequency,
        "write_depth_m": write,
        "read_depth_m": read,
        "layer_top_m": top,
        "layer_bottom_m": bottom,
        "thickness_m": max(bottom - top, 0.0),
        "margin_m": (read + top) - write,
        "bound_by": "read" if (read + top) <= write else "write",
        "why": "the write depth is set by the current and the geometry and "
               "is the same for every band; the read depth is a wavelength "
               "and is not, so which lid binds is a per-band question",
    }


def recorded_magnetisation(drive_ratio: float,
                           coercivity_a_m: float = 600.0 * 1000.0
                           / (4.0 * math.pi),
                           remanence_t: float = 0.15,
                           gap_m: float = 0.30e-6,
                           spacing_m: float = 0.05e-6,
                           points: int = 4000
                           ) -> Dict[str, np.ndarray]:
    """WHAT THE TAPE KEEPS, depth by depth - Ethan's B-H curve, at last used
    on the tape rather than on the core.

    His directive was to "use the BH curve to model this part", and until
    now the curve in this module was only ever applied to the head's own
    ferrite: `head_transformer` and `saturation_response` both put the CORE
    on it. The tape has a B-H curve too, and it is the one that decides what
    survives, because what a coating keeps after the field has passed is its
    REMANENCE curve - the same sigmoid, saturating at the remanent
    magnetisation rather than at the material's full saturation.

    So the chain finally runs end to end: a current makes a deep-gap field
    through `head_efficiency`, that field reaches into the coating through
    `karlqvist_field`, and the coating's own curve turns it into the
    magnetisation left behind. Below the depth where the field falls under
    the coercivity nothing is written, which is where `write_depth_m` comes
    from and where this profile stops.
    """
    depth = write_depth_m(drive_ratio, gap_m)
    top = float(spacing_m)
    if depth <= top:
        empty = np.zeros(0)
        return {"depth_m": empty, "field_a_m": empty,
                "magnetisation_t": empty, "write_depth_m": depth}
    y = np.linspace(top, depth, int(points))
    field = karlqvist_field(np.zeros_like(y), y, gap_m,
                            float(drive_ratio) * float(coercivity_a_m)
                            )["magnitude_a_m"]
    return {
        "depth_m": y,
        "field_a_m": field,
        "magnetisation_t": anhysteretic_b(field, float(remanence_t),
                                          float(coercivity_a_m)),
        "write_depth_m": depth,
    }


def linked_flux(drive_ratio: float, frequency_hz: float,
                writing_speed_m_s: Optional[float] = None,
                gap_m: float = 0.30e-6, spacing_m: float = 0.05e-6,
                coercivity_a_m: float = 600.0 * 1000.0 / (4.0 * math.pi),
                remanence_t: float = 0.15,
                transition: bool = True, **kwargs) -> float:
    """The flux the reproduce head actually links from that profile.

    Two things stand between the magnetisation and the output and both are
    already in this module. Wallace's `exp(-2 pi y / lambda)` weights each
    depth by how much of it the head can see, and the transition length -
    the distance over which a reversal actually turns over, which
    `transition_length` computes from Williams-Comstock - blurs the
    reversal itself and costs `exp(-2 pi a / lambda)`.

    THE SECOND TERM IS WHAT MAKES AN OPTIMUM EXIST. Driving harder always
    writes more magnetisation, so the depth integral alone rises without
    limit - measured, it is monotone in the drive over 1.02 to 20 times the
    coercivity at every frequency in the band. What turns it over is that a
    deeper recording carries a longer transition, and a longer transition
    costs short wavelengths. `transition=False` removes the term and is the
    control for that statement rather than an option worth using.
    """
    band = spec_band_centres()
    speed = (float(band["writing_speed_m_s"]) if writing_speed_m_s is None
             else float(writing_speed_m_s))
    wavelength = speed / max(float(frequency_hz), 1e-30)
    profile = recorded_magnetisation(drive_ratio, coercivity_a_m, remanence_t,
                                     gap_m, spacing_m, **kwargs)
    y = profile["depth_m"]
    if y.size < 2:
        return 0.0
    linked = float(np.trapezoid(
        profile["magnetisation_t"] * np.exp(-2.0 * np.pi * y / wavelength), y))
    if not transition:
        return linked
    length = transition_length(coercivity_a_m=coercivity_a_m,
                               remanence_t=remanence_t,
                               thickness_m=profile["write_depth_m"]
                               - float(spacing_m),
                               spacing_m=float(spacing_m))
    return linked * math.exp(-2.0 * np.pi * length / wavelength)


def record_current_curve(frequency_hz: Optional[float] = None,
                         drives=None, **kwargs) -> Dict[str, object]:
    """THE STANDARD'S OWN CURVE, COMPUTED, AND THE SECOND ROUTE TO ITS PEAK.

    SMPTE 32M 3.9.1.1.6 defines the optimum record current as the one
    "which returns the maximum output signal level during playback".
    `optimum_record_drive` answers that with a geometric argument - write
    until the layer is as deep as the reproduce head can see. This answers
    the same question by actually building the curve the clause describes:
    the flux linked at each drive, through the tape's own remanence curve,
    Wallace's depth weighting and the transition length.

    THE TWO ROUTES ARE INDEPENDENT AND THEY AGREE. Measured over 1200
    drives from 1.02 to 20 times the coercivity, at the format's 0.30 um
    gap, 0.05 um separation and the standard's 5.80 m/s:

        band            curve peak   geometric   ratio
        sync tip           3.616       3.598     1.005
        luma centre        3.315       3.278     1.011
        peak white         3.046       3.000     1.015
        colour under      15.520      15.934     0.974

    Nothing was tuned to make those agree. One route knows only the depth
    the head can see and the depth the field reaches; the other knows only
    the tape's remanence curve and the length of a reversal. That they land
    within one and a half per cent of each other across a factor of five in
    drive is the check on both.

    AND THE CONTROL FAILS INFORMATIVELY. With the transition term removed
    the curve has no peak at all - it is monotone to the end of the grid at
    every frequency - so the maximum the standard names is created by the
    transition length and by nothing else in this model. A curve that peaked
    without it would mean the depth integral was wrong.
    """
    band = spec_band_centres()
    frequency = (float(band["luma_geometric_centre_hz"])
                 if frequency_hz is None else float(frequency_hz))
    if drives is None:
        # from just above the coercivity, below which nothing is recorded at
        # all, to well past the core's own saturation headroom
        drives = np.linspace(1.02, 20.0, 1200)
    drives = np.asarray(drives, dtype=np.float64)
    flux = np.array([linked_flux(float(drive), frequency, **kwargs)
                     for drive in drives])
    plain = np.array([linked_flux(float(drive), frequency, transition=False,
                                  **kwargs) for drive in drives])
    peak = int(np.argmax(flux))
    geometric = optimum_record_drive(frequency, **{
        key: value for key, value in kwargs.items()
        if key in ("gap_m", "spacing_m", "writing_speed_m_s")})
    return {
        "frequency_hz": frequency,
        "drive_ratio": drives,
        "linked_flux": flux,
        "linked_flux_without_transition": plain,
        "peak_drive_ratio": float(drives[peak]),
        "geometric_drive_ratio": geometric["drive_ratio"],
        "agreement": float(drives[peak]) / geometric["drive_ratio"],
        "without_transition_is_monotone": bool(int(np.argmax(plain))
                                               == len(plain) - 1),
        "clause": "SMPTE 32M 3.9.1.1.6",
    }
