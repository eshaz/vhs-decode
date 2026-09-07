"""WHAT REMAINS WHEN EVERY MODELLED COMPONENT IS EXHAUSTED.

Ethan, 2026-09-06, with a written framework attached: *"The next test I
want is to see what remains after all the possible modeled components are
exhausted ... I believe I can use fundamental physical properties to
complete the estimation to its fullest extent."*

His framework, in his own terms. The state space is a Hilbert space split
into two closed orthogonal subspaces: a MODELABLE one spanned by the
transformations that linear electrodynamics and the deterministic head
geometry govern, and its ORTHOGONAL COMPLEMENT holding the microscopic
magnetic behaviour, the thermal fluctuation, the sub-grain interaction and
the local hysteresis that no macroscopic model reaches. Every state splits
uniquely between them. The modelable part is taken by minimum-norm
projection, so that all modelable energy is exhausted; the remainder is
then bound by the micromagnetics rather than left as arbitrary noise.

THE FORMULA AND ITS REGIME, because the two differ and the difference
matters. His section 3 gives the minimum-norm solution as

    x = A* (A A*)^-1 y

which is exactly right when the problem is UNDERDETERMINED - more unknowns
than data, so `A A*` is invertible and the equation `A x = y` has a whole
family of solutions from which this picks the shortest. This arc's data is
usually the other way round: 944 usable frequency bins against 14 to 52
key entries. There `A A*` is singular, the formula has no meaning as
written, and the object that carries his intent into both regimes is the
Moore-Penrose pseudo-inverse, which REDUCES to his formula exactly whenever
`A A*` is invertible and otherwise returns the minimum-norm least-squares
solution. `regime` reports which case the caller is in rather than leaving
it to be assumed.

MEASURED, WHICH IS THE TEST HE ASKED FOR. The full fourteen-entry
component key against the measured sync-edge response of three tapes, both
heads, with the ringing correction off and on, over 0.2 to 4.0 MHz, in the
log domain with the level and the delay removed as nuisances:

    export         head   modelled   remains   above the medium's floor
    cd off          a       0.6396    0.3604          +32.13 dB
    cd off          b       0.8009    0.1991          +29.56
    cd on           a       0.8250    0.1750          +28.99
    cd on           b       0.8334    0.1666          +28.78
    home off        a       0.8804    0.1196          +27.34
    home off        b       0.8642    0.1358          +27.89
    home on         a       0.8752    0.1248          +27.53
    home on         b       0.8615    0.1385          +27.98
    pnb off         a       0.8695    0.1305          +27.72
    pnb off         b       0.8782    0.1218          +27.42
    pnb on          a       0.8463    0.1537          +28.43
    pnb on          b       0.8607    0.1393          +28.00

So every modelled component together accounts for about 84 per cent of the
measured departure and 16 per cent survives, and that survivor sits 27 to
32 decibels above the floor the medium's own physics fixes. The modelling
is not near the medium's limit; a great deal of structure is still
unclaimed.

THE CONDITIONING IS SEVERE AND MOSTLY HARMLESS, which had to be checked
rather than assumed. The key's singular values span 89.8 down to 4.1e-05,
a condition number of 2.19e+06. Raising the rank cut to the export's own
measured relative standard error of 0.01093 drops the rank from fourteen
to ten and the modelled share only from 0.8250 to 0.8122, so the
directions the conditioning destroys carry almost nothing. Ten times that
cut collapses the rank to four and the modelled share to 0.3945, which is
the scale on which this answer is sensitive to the choice.

AND "ALL MODELABLE ENERGY IS EXHAUSTED" NEEDS ONE QUALIFICATION, which
this arc has measured rather than supposed. The components are not
independent: the identifiability work found the magnetic mechanisms
collapsing to 1.58 distinguishable directions of six, with gap and azimuth
collinear at coherence 1.000 and spacing and thickness at 0.994. So `A`
is rank-deficient in practice, and energy lying in its null space is
modelable in principle and unreachable from this data. Exhausted means
exhausted TO THE RANK, and `decompose` returns that rank, the condition
number and the energy it had to leave behind, because a projection that
quietly inverts a singular operator will report a beautifully small
residual made entirely of amplified noise.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np

# The vacuum permeability, exact by the 2019 redefinition to within the
# fine-structure constant's uncertainty; CODATA 2018.
VACUUM_PERMEABILITY = 4.0e-7 * math.pi


def build_operator(signatures: Dict[str, np.ndarray],
                   order: Optional[Sequence[str]] = None
                   ) -> Dict[str, object]:
    """Assemble the forward observation operator from the component key.

    Each modelled component contributes one column: its signature over the
    frequency grid, stacked as real and imaginary parts so that a component
    which acts in amplitude and one which acts in phase are ORTHOGONAL
    rather than the same direction. Under the Hermitian inner product a
    pure-amplitude signature and a pure-phase signature of the same shape
    satisfy `|<r, jr>| = ||r||^2` and are indistinguishable; stacked as
    real they are not. `head_differential.phase_signature` records the same
    argument for the same reason.
    """
    names = list(order) if order is not None else list(signatures)
    columns = []
    for name in names:
        values = np.asarray(signatures[name]).ravel()
        columns.append(np.concatenate([values.real, values.imag]))
    if not columns:
        raise ValueError("the modelable subspace needs at least one component")
    matrix = np.column_stack(columns)
    return {
        "operator": matrix,
        "names": names,
        "samples": matrix.shape[0],
        "components": matrix.shape[1],
        "why": ("real and imaginary stacked, so an amplitude mechanism and "
                "a phase mechanism of one shape are two directions and not "
                "one"),
    }


def regime(operator) -> Dict[str, object]:
    """Which of Ethan's two cases the data is in, and what follows.

    Underdetermined - fewer samples than components - is the case his
    formula is written for and `A A*` is invertible. Overdetermined is the
    case this arc's data is usually in, where it is not.
    """
    matrix = np.asarray(operator, dtype=np.float64)
    samples, components = matrix.shape
    under = samples < components
    return {
        "samples": int(samples),
        "components": int(components),
        "underdetermined": bool(under),
        "formula_applies": bool(under),
        "note": ("A A* is invertible only when the samples are the fewer; "
                 "otherwise the pseudo-inverse is the object that carries "
                 "the same intent"),
    }


def decompose(operator, data, tolerance: Optional[float] = None
              ) -> Dict[str, object]:
    """Split the measurement between the modelable subspace and its complement.

    The projection is taken through the singular value decomposition rather
    than by forming any inverse, so a rank-deficient operator is handled by
    reporting its rank instead of by amplifying the directions it cannot
    see. `tolerance` is the relative singular-value cut; the default is the
    conventional one, the largest singular value times the larger dimension
    times the double's epsilon.
    """
    matrix = np.asarray(operator, dtype=np.float64)
    values = np.asarray(data, dtype=np.float64).ravel()
    if values.size != matrix.shape[0]:
        raise ValueError("the data must have one entry per row of the operator")
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    cut = (tolerance if tolerance is not None
           else singular.max() * max(matrix.shape) * np.finfo(np.float64).eps)
    keep = singular > cut
    rank = int(keep.sum())
    basis = left[:, keep]
    modelled = basis @ (basis.T @ values)
    remainder = values - modelled
    total = float(values @ values)
    coefficients = np.zeros(matrix.shape[1])
    if rank:
        coefficients = (right[keep].T
                        @ ((left[:, keep].T @ values) / singular[keep]))
    # what the rank cut left in the modelable subspace but out of reach
    dropped = left[:, ~keep]
    unreachable = float(np.sum((dropped.T @ values) ** 2)) if dropped.size else 0.0
    return {
        "modelled": modelled,
        "remainder": remainder,
        "coefficients": coefficients,
        "rank": rank,
        "components": int(matrix.shape[1]),
        "singular_values": singular,
        "condition": float(singular.max() / singular[keep].min())
        if rank else float("inf"),
        "modelled_share": float(modelled @ modelled) / total if total else float("nan"),
        "remainder_share": float(remainder @ remainder) / total if total else float("nan"),
        "unreachable_share": unreachable / total if total else float("nan"),
        "rank_deficient": bool(rank < matrix.shape[1]),
        "why": ("exhausted means exhausted to the rank; energy in the "
                "operator's null space is modelable in principle and "
                "unreachable from this data"),
    }


def minimum_norm(operator, data) -> Dict[str, object]:
    """Ethan's section 3 verbatim where it applies, and its generalisation.

    Returns both: `by_formula` is `A* (A A*)^-1 y` and is computed only
    when `A A*` is invertible; `by_pseudo_inverse` is always computed. The
    test that they agree where both exist is in the suite, and it is the
    check that the generalisation is faithful rather than merely
    convenient.
    """
    matrix = np.asarray(operator, dtype=np.float64)
    values = np.asarray(data, dtype=np.float64).ravel()
    case = regime(matrix)
    pseudo = np.linalg.pinv(matrix) @ values
    formula = None
    if case["formula_applies"]:
        gram = matrix @ matrix.T
        if np.linalg.matrix_rank(gram) == gram.shape[0]:
            formula = matrix.T @ np.linalg.solve(gram, values)
    return {
        "by_pseudo_inverse": pseudo,
        "by_formula": formula,
        "agree": bool(formula is not None
                      and np.allclose(formula, pseudo, rtol=1e-8, atol=1e-12)),
        "regime": case,
        "norm": float(np.linalg.norm(pseudo)),
        "why": ("the pseudo-inverse reduces to his formula exactly when "
                "A A* is invertible and carries the same intent when it "
                "is not"),
    }


# ---------------------------------------------------------------------------
# Section 4: the physical bound on what remains
# ---------------------------------------------------------------------------

# The saturation magnetisation of gamma-Fe2O3, the oxide of a ferric VHS
# tape. ASSUMPTION, not a specification: SMPTE 32M states the coercivity
# and nothing else about the medium, and `vhs_specification` records
# remanence and coating thickness as absent from the standard. 370 kA/m is
# the bulk figure for maghemite; a real coating is a dispersion and its
# volume-averaged value is lower.
SATURATION_MAGNETISATION_A_M = 370e3
SATURATION_MAGNETISATION_STATUS = "assumed: bulk gamma-Fe2O3, not measured"

# The exchange stiffness of the same oxide. ASSUMPTION. Ferrimagnetic
# oxides sit one to two orders below the transition metals, and 1e-11 J/m
# is the usual figure quoted for maghemite. Every length below scales as
# its square root, so a factor of four error in it is a factor of two in
# them, and the conclusion survives that comfortably.
EXCHANGE_STIFFNESS_J_M = 1.0e-11
EXCHANGE_STIFFNESS_STATUS = "assumed: typical for a ferrimagnetic oxide"

# The long axis of an acicular particle in a ferric video tape.
# ASSUMPTION, from the coating literature rather than from any standard.
PARTICLE_LENGTH_M = 0.4e-6
PARTICLE_LENGTH_STATUS = "assumed: acicular gamma-Fe2O3, 0.3 to 0.5 micron"


def anisotropy_j_m3(coercivity_a_m: float,
                    saturation_a_m: float = SATURATION_MAGNETISATION_A_M
                    ) -> Dict[str, object]:
    """The anisotropy DERIVED from the specified coercivity, not assumed.

    For coherent rotation of a uniaxial particle the switching field is
    `H_c = 2 K / (mu_0 M_s)`, so the coercivity the standard states fixes
    the anisotropy once the saturation magnetisation is chosen. That is
    worth doing rather than quoting an anisotropy directly, because the
    coercivity IS specified - SMPTE 32M gives 50 kA/m for a ferric tape -
    and so the only assumption left in the number is the magnetisation.
    """
    energy = 0.5 * VACUUM_PERMEABILITY * float(saturation_a_m) * float(coercivity_a_m)
    return {
        "anisotropy_j_m3": energy,
        "coercivity_a_m": float(coercivity_a_m),
        "saturation_a_m": float(saturation_a_m),
        "cite": ("the coercivity is SMPTE 32M-2004 via vhs_specification; "
                 "the magnetisation is an assumption"),
        "relation": "H_c = 2 K / (mu_0 M_s), coherent rotation",
    }


def micromagnetic_lengths(coercivity_a_m: float,
                          saturation_a_m: float = SATURATION_MAGNETISATION_A_M,
                          exchange_j_m: float = EXCHANGE_STIFFNESS_J_M,
                          particle_m: float = PARTICLE_LENGTH_M
                          ) -> Dict[str, object]:
    """The fundamental lengths of the medium, each with what fixes it.

    These are the boundaries Ethan's section 4 asks the interpolation to
    respect. The exchange stiffness forbids the magnetisation turning over
    a distance shorter than the exchange length; a reversal cannot be
    narrower than a domain wall; and no structure can be finer than the
    particle that carries it.
    """
    anisotropy = anisotropy_j_m3(coercivity_a_m, saturation_a_m)["anisotropy_j_m3"]
    exchange_length = math.sqrt(float(exchange_j_m) / anisotropy)
    magnetostatic = math.sqrt(
        2.0 * float(exchange_j_m)
        / (VACUUM_PERMEABILITY * float(saturation_a_m) ** 2))
    return {
        "anisotropy_j_m3": anisotropy,
        "exchange_length_m": exchange_length,
        "domain_wall_width_m": math.pi * exchange_length,
        "magnetostatic_exchange_length_m": magnetostatic,
        "particle_length_m": float(particle_m),
        "quality_factor": anisotropy / (
            0.5 * VACUUM_PERMEABILITY * float(saturation_a_m) ** 2),
        "assumptions": {
            "saturation": SATURATION_MAGNETISATION_STATUS,
            "exchange": EXCHANGE_STIFFNESS_STATUS,
            "particle": PARTICLE_LENGTH_STATUS,
        },
        "why": ("the exchange stiffness sets the shortest distance over "
                "which the magnetisation can turn, and no structure is "
                "finer than the particle carrying it"),
    }


def physical_bound(coercivity_a_m: float, writing_speed_m_s: float,
                   band_hz, gap_m: Optional[float] = None,
                   **kwargs) -> Dict[str, object]:
    """Which physical length binds inside the band, and which do not.

    Every length becomes a frequency through the writing speed: a feature
    of extent L on the tape passes the head in L over v seconds, so it
    carries no structure above v over L. The question this answers is
    whether any of them lands INSIDE the recorded band, because a
    constraint that sits above the band constrains nothing that was
    recorded, and saying so is more useful than quoting it as though it did.
    """
    lengths = micromagnetic_lengths(coercivity_a_m, **kwargs)
    speed = float(writing_speed_m_s)
    low, high = float(band_hz[0]), float(band_hz[1])
    entries = {
        "exchange length": lengths["exchange_length_m"],
        "domain wall width": lengths["domain_wall_width_m"],
        "particle length": lengths["particle_length_m"],
    }
    if gap_m:
        entries["head gap"] = float(gap_m)
    frequencies = {name: speed / value for name, value in entries.items()}
    inside = {name: value for name, value in frequencies.items()
              if low <= value <= high}
    binding = min(frequencies.items(), key=lambda item: item[1])
    return {
        "lengths_m": entries,
        "frequencies_hz": frequencies,
        "band_hz": (low, high),
        "wavelength_at_band_top_m": speed / high,
        "inside_the_band": inside,
        "any_inside": bool(inside),
        "binding": binding[0],
        "binding_hz": binding[1],
        "headroom": binding[1] / high,
        "writing_speed_m_s": speed,
        "assumptions": lengths["assumptions"],
        "why": ("a length becomes a frequency through the writing speed; "
                "a constraint above the band did not shape anything that "
                "was recorded"),
    }


# The acicular particle's diameter, the second axis of the needle.
# ASSUMPTION from the coating literature; nothing in SMPTE 32M describes
# the medium beyond its coercivity.
PARTICLE_DIAMETER_M = 0.05e-6
PARTICLE_DIAMETER_STATUS = "assumed: acicular gamma-Fe2O3, aspect about 8 to 1"


def particulate_floor(track_width_m: float, recorded_depth_m: float,
                      wavelength_m: float,
                      particle_m: float = PARTICLE_LENGTH_M,
                      diameter_m: float = PARTICLE_DIAMETER_M,
                      packing: float = 0.4) -> Dict[str, object]:
    """THE PHYSICAL FLOOR THAT DOES BIND, since none of the lengths do.

    The medium is a dispersion of finite particles, and a resolution cell
    holds a countable number of them. Their orientations are independent,
    so the magnetisation a cell carries fluctuates as the square root of
    that count however perfectly everything else is modelled or corrected.
    This is the fundamental physical property that completes the estimate
    where the micromagnetic lengths cannot: they sit above the band, and
    this does not.

    The cell is the track's width, the depth the record current actually
    magnetised - not the coating's full thickness, which is deeper than
    the write reaches - and half a recorded wavelength along the track.
    `packing` is the volume fraction of oxide in the binder.
    """
    particle_volume = math.pi * (0.5 * float(diameter_m)) ** 2 * float(particle_m)
    cell_volume = (float(track_width_m) * float(recorded_depth_m)
                   * 0.5 * float(wavelength_m))
    count = float(packing) * cell_volume / particle_volume
    amplitude_ratio = math.sqrt(count)
    return {
        "particles_per_cell": count,
        "particle_volume_m3": particle_volume,
        "cell_volume_m3": cell_volume,
        "packing_fraction": float(packing),
        "amplitude_snr": amplitude_ratio,
        # ONE number, not two. The amplitude ratio is the root of the
        # count, so twenty times its logarithm and ten times the count's
        # are the same quantity; an earlier version returned both under
        # different names, which invites a reader to compare them and
        # conclude something from an identity.
        "snr_db": 10.0 * math.log10(count),
        "wavelength_m": float(wavelength_m),
        "assumptions": {
            "particle": PARTICLE_LENGTH_STATUS,
            "diameter": PARTICLE_DIAMETER_STATUS,
            "packing": "assumed: 0.4 by volume, typical for a ferric coating",
        },
        "why": ("the count is finite and the orientations independent, so "
                "the square root of the count is a floor no processing "
                "reaches past"),
    }


def against_the_floor(remainder_share: float, signal_power: float,
                      floor: Dict[str, object]) -> Dict[str, object]:
    """What remains, measured against the physical floor rather than zero.

    Ethan's framework asks what is left when the modelable subspace is
    exhausted, and the answer is only meaningful against something. The
    particulate floor is the right something: a remainder AT it is the
    medium and there is nothing further to model, while a remainder ABOVE
    it is still structure and the modelling is not finished.
    """
    remaining = float(remainder_share) * float(signal_power)
    floor_power = float(signal_power) / max(float(floor["particles_per_cell"]), 1e-300)
    ratio = remaining / floor_power if floor_power > 0 else float("inf")
    return {
        "remaining_power": remaining,
        "floor_power": floor_power,
        "ratio": ratio,
        "ratio_db": 10.0 * math.log10(ratio) if ratio > 0 else float("-inf"),
        "at_the_floor": bool(ratio <= 2.0),
        "verdict": ("at the medium's own floor, so nothing further is "
                    "modelable" if ratio <= 2.0 else
                    "above the medium's floor, so structure remains"),
        "why": ("a remainder is only meaningful against a floor, and the "
                "particle count is the one the physics fixes"),
    }


def exhausted(response, frequency_hz, signatures,
              relative_error: Optional[float] = None,
              track_width_m: Optional[float] = None,
              recorded_depth_m: float = 0.2064e-6,
              writing_speed_m_s: float = 5.8,
              carrier_hz: float = 3.9e6) -> Dict[str, object]:
    """The whole framework on one measured response: the test Ethan asked for.

    Takes a measured complex response, removes the level and the delay as
    nuisances, projects what is left onto the modelable subspace, and
    reports what survives against the medium's own particulate floor.

    `relative_error` sets the rank cut in the data's own units, which is
    the honest way to choose it: a direction the measurement cannot see
    above its own noise is not a direction the measurement has.
    """
    from vhsdecode.models import vhs_specification as _spec
    values = np.asarray(response, dtype=np.complex128).ravel()
    grid = np.asarray(frequency_hz, dtype=np.float64).ravel()
    good = np.isfinite(values) & (np.abs(values) > 0) & np.isfinite(grid)
    if good.sum() < 8:
        raise ValueError("too few usable bins to exhaust anything")
    logged = (np.log(np.abs(values[good]))
              + 1j * np.unwrap(np.angle(values[good])))
    logged = logged - logged.mean()
    built = build_operator({name: np.asarray(shape).ravel()[good]
                            for name, shape in signatures.items()})
    stacked = np.concatenate([logged.real, logged.imag])
    largest = np.linalg.svd(built["operator"], compute_uv=False).max()
    tolerance = (relative_error * largest if relative_error else None)
    split = decompose(built["operator"], stacked, tolerance=tolerance)
    width = (track_width_m if track_width_m is not None
             else _spec.SMPTE_32M["track_width_sp_m"]["value"])
    floor = particulate_floor(width, recorded_depth_m,
                              writing_speed_m_s / carrier_hz)
    verdict = against_the_floor(split["remainder_share"],
                               float(stacked @ stacked), floor)
    return {
        "decomposition": split,
        "floor": floor,
        "verdict": verdict,
        "bins_used": int(good.sum()),
        "names": built["names"],
        "why": ("the modelable subspace is exhausted to its rank and what "
                "survives is judged against the only floor the physics "
                "fixes, not against zero"),
    }
