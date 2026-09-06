"""Loss-mechanism separation as an identifiability test that REFUSES what one
tape cannot resolve.

Ethan: *"Loss mechanism separation. Three mechanisms (spacing, gap,
thickness) against two wavelength clusters is underdetermined on one tape.
Don't attempt the three-way split from a single capture. Different
formulations on the same deck give the leverage: thickness varies with
stock, gap and spacing don't. HiFi as a third cluster helps here too."*

*"Fitted coefficients aren't measurements on a single tape - the loss
mechanisms are near-collinear over 3.4-4.4 MHz. The composite H is well
determined; the named components aren't."*

This module is that ruling made executable. It builds the three mechanisms
as log responses IN WAVELENGTH, takes their Jacobian at the wavelengths a
format actually offers, whitens it by the measured noise, and reports how
many independent parameter directions the data pins down. When the requested
split exceeds that count it refuses, and the refusal carries its proof: which
pair is collinear, at what correlation, at what noise. What it will hand over
without argument is the composite - `composite_only` - because that is what
the data determines.

THE THREE MECHANISMS (Wallace 1951 for the spacing law; Bertram, "Theory of
Magnetic Recording", 1994, chapters 4-5 for the gap and thickness terms in
this form; the same forms `head_model.log_response` carries in frequency):

    spacing     exp(-2 pi d / lambda)                -54.6 d/lambda decibels
    gap         sinc(g_eff / lambda) = sin(pi g_eff/lambda)/(pi g_eff/lambda)
                with its sign carried separately, because a sinc past its
                first null is a pi phase flip that no log magnitude holds
    thickness   (1 - exp(-x)) / x,  x = 2 pi delta / lambda

The forms are functions of ONE dimensionless group each - a length over the
recorded wavelength - which is the reason they are collinear
(`docs/COMPONENT_MAPPINGS.md` section 2: a family differing in RATE is
collinear, one differing in KIND is separable). Their fractional
sensitivities, `p d(log|H|)/dp`, show it directly:

    spacing      -2 pi d / lambda                   proportional to 1/lambda
    gap          x cot x - 1,  x = pi g_eff/lambda   about -x^2/3: 1/lambda^2
    thickness    x e^-x/(1 - e^-x) - 1              -> -x/2 (thin), -> -1 (thick)

Over the luma cluster, 1.32 to 1.71 micrometres, that is a constant, a line
and a parabola in frequency over a 1.3:1 span. And the thickness term has
TWO regimes with opposite failure modes: a thin (depth-limited) coating is a
slope, collinear with spacing; a thick one is a LEVEL, collinear with the
gain that every fit leaves free. Which regime a VHS video track is in is
itself a question this module refuses to settle - see `wavelength_table`.

WAVELENGTH REFERENCE TABLE, reproduced from specification by
`wavelength_table` (writing speed 5.80 m/s, JVC VTG82063 section 1 as stated,
not the 5.877 derived from the drum; carriers IEC 60774-1 as carried by
`rf_stages.VHS_CARRIER_HZ`; colour-under 40 f_H, `colour_under.carrier_hz`;
Hi-Fi SMPTE 32M clause 5; 4 f_sc from SMPTE 170M's subcarrier):

    4 f_sc                14.3182 MHz  ->  69.84 ns per sample
    tape per sample       0.4051 um at 5.80 m/s
    lambda, sync tip      1.7059 um at 3.4 MHz
    lambda, peak white    1.3182 um at 4.4 MHz
    lambda, colour-under  9.2155 um at 629.37 kHz
    lambda, Hi-Fi         4.4615 um at 1.3 MHz, 3.4118 um at 1.7 MHz
    read depth lambda/2pi 0.21-0.27 um luma, 1.47 um chroma
    thickness loss, delta = 4 um   -23.4 to -25.6 dB luma, -9.3 dB chroma
    thickness loss, delta = 5 um   -25.3 to -27.5 dB luma, -10.9 dB chroma

Every figure of Ethan's reproduces at his stated precision. WHERE THE
NUMBERS DIFFER, AND WHY: his "about -26 dB luma" is the 5 um coating at
mid-band (-26.5 dB at 3.9 MHz) or either coating at peak white; at 4 um and
sync tip it is -23.4 dB. The larger difference is not arithmetic. His delta
of 4-5 um is the COATING, and no specification in this repository's
collection states a coating thickness (`docs/SPECIFICATION_INVENTORY.md`,
"no remanence, retentivity, squareness or coating thickness figure"), so it
is carried here as a LABELLED ASSUMPTION. JVC VTG82063 section 7.2.1 states
that the video *"is recorded to a depth of about 0.3 um ... by the 0.3 um gap
of the video head"*, and `docs/MATHEMATICS.md` 8a bounds the recorded depth
by self-demagnetisation at lambda/(2 pi) - the same lambda/2pi as his read
depth. If the delta that enters the thickness law is the RECORDED depth
rather than the coating, the luma thickness loss is -4.6 to -5.6 dB, not -26,
and the term has a SHAPE across the cluster instead of being a level. The
two readings differ by 21 dB and, more importantly for this module, differ in
KIND of unidentifiability. Both operating points are therefore evaluated and
neither is chosen.

WHAT WAS MEASURED, in one paragraph (the detail is in `identifiable`,
`formulation_leverage`, `demodulated_identifiable` and `composite_only`).
On the real sync-step exports of cd, home and pnb, both heads, at their own
noise - 0.0040-0.0120 nepers per independent value, 20 values per head -
Ethan's two clusters resolve ONE of the three mechanisms at 10% precision,
on every tape and head, in both thickness regimes: spacing-thickness at
coherence 0.9998 with the recorded depth, spacing-gap at 0.9992 with the
coating, where thickness is a level to one part in 10^6. The chroma
carrier with its own level adds nothing to the luma cluster alone. A Hi-Fi
cluster read by its own heads - which is what the format builds - leaves the
video's split exactly where it was. Only a measured colour-under SIDEBAND
response, 4.8 to 98.8 um, reaches a 4-5 um coating's turnover, and with it
two stocks on one deck determine all three; that measurement does not yet
exist. The exports themselves, being demodulated, annihilate spacing
exactly (7e-15 of the largest column) and determine one curvature.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import (colour_under, head_model, magnetic,
                             output_limit, rf_stages)

# --------------------------------------------------------------------------
# The specification, with provenance. Nothing below is a tuning constant.
# --------------------------------------------------------------------------

MECHANISMS: Tuple[str, str, str] = ("spacing", "gap", "thickness")

# JVC VTG82063 section 7.2.1: the video head gap is 0.3 um and the video is
# recorded to a depth of about 0.3 um. The first is the same figure
# `head_model.TYPICAL["gap_m"]` carries; the second is the recorded depth,
# which is NOT the coating.
SPEC_GAP_M = 0.30e-6
SPEC_RECORDED_DEPTH_M = 0.30e-6

# ASSUMPTION, labelled: the coating thickness Ethan's table uses, 4-5 um. No
# specification in the collection states it (SPECIFICATION_INVENTORY: "no
# ... coating thickness figure anywhere"). The tape's OVERALL thickness is
# specified - 19.0 +1/-2 um for T-120 (SMPTE 32M table 1) - and is not this.
ASSUMED_COATING_M: Tuple[float, float] = (4.0e-6, 5.0e-6)

# MEASURED in this repository, not assumed: the effective whole-path spacing
# fitted on the RF envelope, 0.452 um (home) to 0.481 um (bars, rescaled to
# the stated 5.80 m/s). An EFFECTIVE figure that absorbs every short-
# wavelength loss of record and playback together and, as the colour side
# later showed, part of a head GAIN difference. It is used here only as the
# operating point that sets the scale of a fractional sensitivity.
MEASURED_EFFECTIVE_SPACING_M = 0.48e-6

# Hi-Fi audio carriers, SMPTE 32M clause 5: 1.3 MHz +/- 10 kHz (channel 1)
# and 1.7 MHz +/- 10 kHz (channel 2) for NTSC, maximum deviation +/-150 kHz.
# The repository's own `format_defs/vhs.py` carries the same two figures as
# `fm_audio_channel_0_freq` and `fm_audio_channel_1_freq`.
HIFI_CARRIER_HZ: Tuple[float, float] = (1.3e6, 1.7e6)
HIFI_MAX_DEVIATION_HZ = 150e3

# The colour-under's modulation reach either side of its carrier is the
# decoder's band-pass setting, not a figure of the format
# (`colour_under.CHROMA_BAND_UPPER_HZ`); it is read from there.

# The gap's effective-width correction the head model uses, so the two cannot
# drift apart.
EFFECTIVE_GAP_FACTOR = head_model.EFFECTIVE_GAP_FACTOR
DB_PER_NEPER = head_model.DB_PER_NEPER

# The precision a parameter must reach to count as MEASURED here: its one-
# sigma fractional error at or below this. Ten percent is the bar at which a
# fitted spacing could be quoted to two figures. It is an argument everywhere
# it is used; this is only the default.
DEFAULT_PRECISION = 0.10

# A numerical-rank tolerance, relative to the largest singular value. This is
# the STRUCTURAL rank - what the design could resolve at zero noise - and it
# is reported beside the rank at the measured noise, never instead of it.
STRUCTURAL_TOLERANCE = 1e-9


def default_mechanics() -> Dict[str, float]:
    return head_model.mechanics_for("VHS", "NTSC", "SP") or {}


def writing_speed(mechanics: Optional[Dict[str, float]] = None) -> float:
    return head_model.writing_speed(mechanics or default_mechanics())


def wavelength_m(frequency_hz, speed_m_s: float) -> np.ndarray:
    """lambda = v / f. Not-a-number at or below zero frequency, where no
    wavelength exists."""
    f = np.asarray(frequency_hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, float(speed_m_s) / np.where(f > 0, f, 1.0),
                        np.nan)


# --------------------------------------------------------------------------
# The three mechanisms, in wavelength, with their fractional sensitivities
# --------------------------------------------------------------------------


def spacing_log(wavelength, spacing_m: float) -> np.ndarray:
    """Wallace's spacing loss, in nepers: -2 pi d / lambda. In decibels that
    is -54.6 d/lambda, the figure every recording text quotes."""
    lam = np.asarray(wavelength, dtype=np.float64)
    return -2.0 * np.pi * float(spacing_m) / lam


def gap_log(wavelength, gap_m: float,
            effective_factor: float = EFFECTIVE_GAP_FACTOR) -> np.ndarray:
    """The gap loss's log MAGNITUDE, log|sinc(g_eff / lambda)|, with the
    standard effective-gap widening. The sign is `gap_sign`."""
    lam = np.asarray(wavelength, dtype=np.float64)
    ratio = float(effective_factor) * float(gap_m) / lam
    return np.log(np.maximum(np.abs(np.sinc(ratio)), 1e-300))


def gap_sign(wavelength, gap_m: float,
             effective_factor: float = EFFECTIVE_GAP_FACTOR) -> np.ndarray:
    """+1 or -1: the sign of the gap sinc, which is a pi phase flip past each
    null at lambda = g_eff / n. With the specified 0.3 um gap the first null
    is at lambda = 0.333 um - 17.4 MHz at 5.80 m/s, above every carrier the
    format uses - so the sign is +1 over every cluster here. It is carried
    because a caller at another gap or speed may not be so placed."""
    lam = np.asarray(wavelength, dtype=np.float64)
    ratio = float(effective_factor) * float(gap_m) / lam
    return np.where(np.sinc(ratio) < 0, -1.0, 1.0)


def gap_phase_rad(wavelength, gap_m: float,
                  effective_factor: float = EFFECTIVE_GAP_FACTOR) -> np.ndarray:
    """The phase the gap sinc's sign contributes: pi where it is negative."""
    return np.where(gap_sign(wavelength, gap_m, effective_factor) < 0,
                    np.pi, 0.0)


def thickness_log(wavelength, thickness_m: float) -> np.ndarray:
    """The thickness (coating) loss, log[(1 - e^-x)/x] with
    x = 2 pi delta / lambda. Unity - zero nepers - as delta -> 0."""
    lam = np.asarray(wavelength, dtype=np.float64)
    x = 2.0 * np.pi * float(thickness_m) / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(x > 1e-9, (1.0 - np.exp(-x)) / np.maximum(x, 1e-300),
                        1.0)
    return np.log(np.maximum(term, 1e-300))


def composite_log(wavelength, spacing_m: float, gap_m: float,
                  thickness_m: float, level_nepers: float = 0.0) -> np.ndarray:
    """The composite log magnitude, in nepers: the three losses and a level.
    This is what one tape determines; the three are what it does not."""
    return (spacing_log(wavelength, spacing_m) + gap_log(wavelength, gap_m)
            + thickness_log(wavelength, thickness_m) + float(level_nepers))


def fractional_sensitivities(wavelength, spacing_m: float, gap_m: float,
                             thickness_m: float,
                             effective_factor: float = EFFECTIVE_GAP_FACTOR
                             ) -> Dict[str, np.ndarray]:
    """p d(log|H|)/dp for each mechanism: nepers per e-fold of the parameter.
    Analytic, and checked against finite differences in the unit tests."""
    lam = np.asarray(wavelength, dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    out["spacing"] = -2.0 * np.pi * float(spacing_m) / lam
    x = np.pi * float(effective_factor) * float(gap_m) / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        cot = np.where(np.abs(x) > 1e-9, x / np.tan(x), 1.0)
    out["gap"] = cot - 1.0
    x = 2.0 * np.pi * float(thickness_m) / lam
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # x e^-x / (1 - e^-x) = x / (e^x - 1), which is finite everywhere
        ratio = np.where(x > 1e-9, x / np.expm1(np.minimum(x, 700.0)), 1.0)
    out["thickness"] = ratio - 1.0
    return out


def jacobian(wavelength, spacing_m: float, gap_m: float, thickness_m: float,
             mechanisms: Sequence[str] = MECHANISMS,
             fractional: bool = True) -> np.ndarray:
    """The Jacobian of log|H| with respect to the mechanism parameters,
    evaluated at these wavelengths: one row per wavelength, one column per
    mechanism, in the order given.

    `fractional=True` gives d(log|H|)/d(ln p) - nepers per e-fold - so the
    columns are comparable and a singular value reads as "how many sigmas
    a 100% change moves the data". `fractional=False` gives d(log|H|)/dp in
    nepers per metre."""
    sens = fractional_sensitivities(wavelength, spacing_m, gap_m, thickness_m)
    values = {"spacing": spacing_m, "gap": gap_m, "thickness": thickness_m}
    columns = []
    for name in mechanisms:
        column = sens[name]
        if not fractional:
            column = column / max(float(values[name]), 1e-300)
        columns.append(column)
    return np.column_stack(columns)


def design_matrix(wavelengths, mechanisms: Sequence[str] = MECHANISMS,
                  operating_point: Optional[Dict[str, float]] = None,
                  fractional: bool = True) -> Dict[str, object]:
    """THE DESIGN MATRIX: the Jacobian at the cluster wavelengths, with its
    column names and the operating point it was taken at.

    The operating point defaults to the specification where one exists and
    the repository's measurement where one does not: gap 0.30 um (JVC 7.2.1),
    thickness 0.30 um (the JVC recorded depth; pass the coating to evaluate
    Ethan's regime), spacing 0.48 um (measured effective, RF envelope)."""
    point = operating_point_for(operating_point)
    lam = np.asarray(wavelengths, dtype=np.float64).ravel()
    matrix = jacobian(lam, point["spacing_m"], point["gap_m"],
                      point["thickness_m"], mechanisms, fractional)
    return {"matrix": matrix, "columns": list(mechanisms),
            "wavelength_m": lam, "operating_point": point,
            "fractional": bool(fractional)}


def operating_point_for(given: Optional[Dict[str, float]] = None
                        ) -> Dict[str, float]:
    point = {"spacing_m": MEASURED_EFFECTIVE_SPACING_M,
             "gap_m": SPEC_GAP_M,
             "thickness_m": SPEC_RECORDED_DEPTH_M}
    point.update({k: float(v) for k, v in (given or {}).items()
                  if k in point})
    return point


# --------------------------------------------------------------------------
# Clusters: the wavelengths a signal offers, the noise it was read at, and
# WHICH parameters it shares with the video luma track
# --------------------------------------------------------------------------


def cluster(name: str, wavelengths, se_nepers, shares: Iterable[str] = MECHANISMS,
            spacing_offset_m: float = 0.0, se_source: str = "given",
            level_free: bool = True) -> Dict[str, object]:
    """One cluster of wavelengths read at one noise level.

    `shares` names the mechanism parameters this cluster reads IN COMMON with
    the reference (video luma) track. A mechanism not shared becomes this
    cluster's own unknown column - which is the honest description of a
    Hi-Fi track, written and read by SEPARATE heads on the same drum with
    their own gap and their own contact, and of a second tape stock whose
    coating is its own. `spacing_offset_m` is a KNOWN addition to the
    separation for this cluster alone: the depth-multiplexed Hi-Fi audio
    lies under the video's recorded layer, so the video's 0.3 um (JVC 7.2.1)
    is spacing to the audio head. `level_free` gives the cluster its own
    unknown level, which is the default because the record level of the
    colour-under, the Hi-Fi and the FM luma are set separately by the
    recorder and are not calibrated against one another in any measurement
    this project has."""
    lam = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64)).ravel()
    se = np.broadcast_to(np.asarray(se_nepers, dtype=np.float64), lam.shape)
    return {"name": str(name), "wavelength_m": lam, "se_nepers": np.array(se),
            "shares": tuple(shares), "spacing_offset_m": float(spacing_offset_m),
            "se_source": str(se_source), "level_free": bool(level_free)}


def luma_cluster(se_nepers, points: int = 24,
                 mechanics: Optional[Dict[str, float]] = None,
                 carriers_hz: Tuple[float, float] = rf_stages.VHS_CARRIER_HZ["525"],
                 se_source: str = "given", **kwargs) -> Dict[str, object]:
    """The video luma cluster: the FM carrier's own range, sync tip to peak
    white, 3.4 to 4.4 MHz (IEC 60774-1), 1.32 to 1.71 um at 5.80 m/s."""
    speed = writing_speed(mechanics)
    f = np.linspace(min(carriers_hz), max(carriers_hz), int(points))
    return cluster("luma", wavelength_m(f, speed), se_nepers,
                   se_source=se_source, **kwargs)


def chroma_cluster(se_nepers, points: int = 1, sidebands: bool = False,
                   mechanics: Optional[Dict[str, float]] = None,
                   system: str = "NTSC", se_source: str = "given",
                   **kwargs) -> Dict[str, object]:
    """The colour-under cluster. By default the carrier alone, 9.2155 um,
    which is how Ethan's ruling counts it. With `sidebands=True` the
    decoder's own band-pass reach, 58.7 kHz to 1.2 MHz on NTSC
    (`colour_under.CHROMA_BAND_UPPER_HZ`), 4.8 to 98.8 um - a 20:1 span that
    is a different instrument altogether, and the only one on the format
    that reaches the thickness turnover of a 4-5 um coating."""
    speed = writing_speed(mechanics)
    carrier = colour_under.carrier_hz(system)
    if sidebands:
        upper = colour_under.band_upper_for(system)
        half = upper - carrier
        f = np.linspace(carrier - half, upper, max(int(points), 2))
        f = f[f > 0]
    else:
        f = np.array([carrier] if int(points) <= 1
                     else np.linspace(carrier, carrier, int(points)))
    return cluster("chroma", wavelength_m(f, speed), se_nepers,
                   se_source=se_source, **kwargs)


def hifi_cluster(se_nepers, points: int = 24, deviation: bool = False,
                 mechanics: Optional[Dict[str, float]] = None,
                 own_head: bool = True,
                 buried_by_m: float = SPEC_RECORDED_DEPTH_M,
                 se_source: str = "given", **kwargs) -> Dict[str, object]:
    """The Hi-Fi audio cluster, 1.3 to 1.7 MHz (SMPTE 32M clause 5), 3.41 to
    4.46 um; with `deviation=True` the +/-150 kHz reach, 3.14 to 5.04 um.

    `own_head=True` is the physical description: the Hi-Fi tracks are
    written and read by separate heads with their own gap and their own
    contact, so this cluster shares only the TAPE's parameter - thickness -
    with the video track, and `buried_by_m` (the video's recorded depth,
    JVC 7.2.1) is known extra spacing to the audio head. `own_head=False` is
    the idealisation in which the cluster reads the very same spacing and
    gap as the video head - the reading under which "HiFi as a third
    cluster helps" is evaluated, and which the report states as such."""
    speed = writing_speed(mechanics)
    low, high = HIFI_CARRIER_HZ
    if deviation:
        low, high = low - HIFI_MAX_DEVIATION_HZ, high + HIFI_MAX_DEVIATION_HZ
    f = np.linspace(low, high, int(points))
    shares = ("thickness",) if own_head else MECHANISMS
    return cluster("hifi", wavelength_m(f, speed), se_nepers, shares=shares,
                   spacing_offset_m=float(buried_by_m) if own_head else 0.0,
                   se_source=se_source, **kwargs)


# --------------------------------------------------------------------------
# Assembly: one whitened Jacobian over every cluster, columns named
# --------------------------------------------------------------------------

# Which mechanisms vary with the tape STOCK under Ethan's premise: the
# coating is the stock's, the gap and the contact are the deck's. Passing
# `("thickness", "spacing")` instead states the other case tape_model
# records - a stock's surface adds its own separation.
PER_STOCK: Tuple[str, ...] = ("thickness",)


def assemble(clusters: Sequence[Dict[str, object]],
             operating_point: Optional[Dict[str, float]] = None,
             mechanisms: Sequence[str] = MECHANISMS,
             stock_of: Optional[Dict[str, str]] = None,
             per_stock: Sequence[str] = ("thickness",)) -> Dict[str, object]:
    """Stack the clusters into one design, whitened by each row's SE.

    Columns: the shared mechanisms; then, for every cluster, its unshared
    mechanisms as `<cluster>:<mechanism>`; then one `<cluster>:level` per
    cluster with a free level. `stock_of` maps a cluster name to a stock
    name, and a mechanism listed in `PER_STOCK` gets one column per stock
    instead of one shared column - the formulation lever."""
    point = operating_point_for(operating_point)
    stock_of = dict(stock_of or {})
    rows: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    row_cluster: List[str] = []
    wavelengths: List[np.ndarray] = []
    columns: List[str] = []

    def column_index(name: str) -> int:
        if name not in columns:
            columns.append(name)
        return columns.index(name)

    blocks = []
    for c in clusters:
        lam = c["wavelength_m"]
        # a cluster may carry its own operating point for a parameter it
        # owns - a stock's coating, a separate head's gap - else the shared
        sens = fractional_sensitivities(
            lam, float(c.get("spacing_m", point["spacing_m"]))
            + float(c.get("spacing_offset_m", 0.0)),
            float(c.get("gap_m", point["gap_m"])),
            float(c.get("thickness_m", point["thickness_m"])))
        block: Dict[int, np.ndarray] = {}
        for name in mechanisms:
            if name in c["shares"]:
                stock = stock_of.get(c["name"])
                key = (name if stock is None or name not in per_stock
                       else "%s@%s" % (name, stock))
            else:
                key = "%s:%s" % (c["name"], name)
            block[column_index(key)] = sens[name]
        if c.get("level_free", True):
            block[column_index("%s:level" % c["name"])] = np.ones_like(lam)
        blocks.append(block)
        weights.append(1.0 / np.maximum(c["se_nepers"], 1e-300))
        row_cluster.extend([c["name"]] * len(lam))
        wavelengths.append(lam)

    total = sum(len(c["wavelength_m"]) for c in clusters)
    matrix = np.zeros((total, len(columns)))
    start = 0
    for c, block, w in zip(clusters, blocks, weights):
        n = len(c["wavelength_m"])
        for j, column in block.items():
            matrix[start:start + n, j] = column * w
        start += n
    return {"matrix": matrix, "columns": columns,
            "row_cluster": np.array(row_cluster),
            "wavelength_m": np.concatenate(wavelengths),
            "operating_point": point,
            "mechanism_columns": [i for i, name in enumerate(columns)
                                  if not name.endswith(":level")],
            "level_columns": [i for i, name in enumerate(columns)
                              if name.endswith(":level")]}


def _profile_out(matrix: np.ndarray, keep: Sequence[int],
                 nuisance: Sequence[int]) -> np.ndarray:
    """The kept columns projected orthogonal to the nuisance columns: the
    Fisher information for the kept parameters with the nuisances free."""
    kept = matrix[:, list(keep)]
    if not nuisance:
        return kept
    nuis = matrix[:, list(nuisance)]
    q, _ = np.linalg.qr(nuis)
    return kept - q @ (q.T @ kept)


def _analyse(matrix: np.ndarray, names: Sequence[str], precision: float
             ) -> Dict[str, object]:
    """Singular values, fractional errors, correlations and the worst pair of
    an already-whitened design."""
    names = list(names)
    if matrix.size == 0 or matrix.shape[1] == 0:
        return {"singular_values": np.array([]), "structural_rank": 0,
                "rank_at_noise": 0, "condition": float("inf")}
    norms = np.linalg.norm(matrix, axis=0)
    dead = norms <= STRUCTURAL_TOLERANCE * max(float(norms.max()), 1e-300)
    annihilated = [n for n, flag in zip(names, dead) if flag]
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    largest = float(s[0]) if len(s) else 0.0
    structural = int(np.sum(s > STRUCTURAL_TOLERANCE * max(largest, 1e-300)))
    with np.errstate(divide="ignore"):
        errors = np.where(s > 0, 1.0 / np.maximum(s, 1e-300), np.inf)
    rank_at_noise = int(np.sum(errors <= float(precision)))
    # the covariance through the pseudo-inverse, so an annihilated column
    # reads as infinite rather than as a division error
    inverse = np.where(s > STRUCTURAL_TOLERANCE * max(largest, 1e-300),
                       1.0 / np.maximum(s, 1e-300) ** 2, np.inf)
    finite = np.isfinite(inverse)
    cov = (vt.T[:, finite] * inverse[finite]) @ vt[finite, :]
    diag = np.sqrt(np.maximum(np.diag(cov), 0.0))
    # a column the instrument annihilates has NO error bar, not a small one:
    # the pseudo-inverse would report the minimum-norm solution's, which is
    # the wrong answer to the question being asked
    diag = np.where(dead, np.inf, diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(diag, diag)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    unit = matrix / np.where(norms > 0, norms, 1.0)
    coherence = np.abs(unit.T @ unit)
    relative = dict(zip(names, map(float, norms / max(float(norms.max()), 1e-300))))
    worst, worst_value = None, 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(coherence[i, j])
            if value > worst_value:
                worst, worst_value = (names[i], names[j]), value
    return {
        "names": names,
        "singular_values": s,
        "fractional_error_per_direction": errors,
        "structural_rank": structural,
        "rank_at_noise": rank_at_noise,
        "condition": float(largest / max(float(s[-1]), 1e-300)) if len(s) else float("inf"),
        "fractional_se": dict(zip(names, map(float, diag))),
        "estimate_correlation": corr,
        "column_coherence": coherence,
        "worst_pair": worst,
        "worst_pair_coherence": float(worst_value),
        "worst_pair_correlation": (float(abs(corr[names.index(worst[0]),
                                                names.index(worst[1])]))
                                   if worst else 0.0),
        "annihilated": annihilated,
        "column_norm_relative": relative,
    }


def identifiable(clusters: Sequence[Dict[str, object]],
                 operating_point: Optional[Dict[str, float]] = None,
                 mechanisms: Sequence[str] = MECHANISMS,
                 precision: float = DEFAULT_PRECISION,
                 stock_of: Optional[Dict[str, str]] = None,
                 per_stock: Sequence[str] = PER_STOCK) -> Dict[str, object]:
    """HOW MANY OF THE NAMED MECHANISMS THESE CLUSTERS CAN RESOLVE, at the
    noise they were read at - and a REFUSAL when the split asked for exceeds
    that number.

    The design is whitened row by row with each cluster's SE, the free
    levels are profiled out, and the mechanism block's singular values are
    read as fractional errors (nepers per e-fold in, so 1/s is the one-sigma
    fractional error along that direction). A direction is RESOLVED when
    that error is at or below `precision`. `structural_rank` is what the
    design could resolve at zero noise; `rank_at_noise` is what it does
    resolve; the refusal is against the second.

    The proof of a refusal names the worst pair by column coherence - the
    cosine between the two sensitivities after the levels are removed, the
    same measure `rf_stages` and the 1.58-of-6 result use - and gives the
    correlation of the two estimates, the singular values, and the noise.

    MEASURED on the real exports, `<tape>_tree_noeq_sync_step_response.npz`
    for cd, home and pnb, both heads. The luma noise is each export's
    median per-bin SE at its information spacing - 0.0043 (cd), 0.0040 to
    0.0044 (home) and 0.0105 to 0.0120 nepers (pnb), on 20 independent
    values per head - transplanted onto the cluster's wavelengths; every
    other cluster's noise is ASSUMED equal to it and labelled so. Precision
    10%. Every tape and both heads give the same ranks and the same pairs;
    the noise moves the fractional errors 2.5x between cd and pnb and
    nothing else.

    ETHAN'S TWO CLUSTERS, luma 1.32-1.71 um and the chroma carrier at
    9.22 um, each with its own level:
      delta = 0.30 um (JVC recorded depth): structural rank 3, rank at
        noise 1 of 3, condition 1.9e4; worst pair SPACING-THICKNESS at
        coherence 0.9998, estimate correlation 1.0000; fractional errors
        spacing 28x, gap 55x, thickness 97x on cd, 67x/133x/234x on pnb.
        REFUSED. The chroma carrier with its own level adds NOTHING: the
        numbers are identical to the luma cluster alone, whose singular
        values 167.6, 1.29, 0.009 say the three directions are known to
        0.6%, 78% and 11400%.
      delta = 4.5 um (the assumed coating): rank at noise 1 of 3, condition
        3.0e6; thickness's level-projected column is 1.8e-6 of spacing's -
        it is a LEVEL, and the free level absorbs it - so the worst pair
        becomes SPACING-GAP at coherence 0.9992, correlation 0.9996.
        REFUSED.
      The arc's measure, the participation ratio of the three columns:
        1.001 and 1.134 of 3, beside 1.58 of 6 for the six-mechanism family
        over 0.5-8 MHz.

    WHAT A THIRD CLUSTER BUYS, at cd/a's noise:
      Hi-Fi as its OWN head (the physical case: shares the coating only):
        the video's split is unchanged, 27.6x/54.6x/96.5x at 0.30 um; the
        cluster's whole shape is spent on its own gap and contact.
      Hi-Fi as the SAME head (the idealisation): rank stays 1 of 3 in both
        regimes. Spacing-thickness coherence falls 0.9998 -> 0.9914 and the
        second direction's error 78% -> 12%; with the +/-150 kHz deviation
        included, rank 2. Rank 3 would need the Hi-Fi read at 9.8e-8 Np in
        the recorded-depth regime - 4.5e4x below the luma noise, which is a
        structural verdict - or 2.0e-4 Np in the coating regime, 22x below.
      The colour-under SIDEBANDS, 4.8 to 98.8 um, the only cluster on the
        format that reaches a 4-5 um coating's turnover at lambda = 2 pi
        delta, about 28 um: coating regime rank 2 of 3 at condition 38,
        spacing 1.8%, thickness 1.1%, gap 11.4%, rank 3 at 3.6 mNp which is
        1.2x below the luma noise; recorded-depth regime rank 2, spacing-
        thickness 0.979, and rank 3 only below 9.7e-7 Np. NO SUCH
        MEASUREMENT EXISTS in the shared exports or the models: the runtime
        reads the burst as one point at 629 kHz, and the luma-to-chroma
        transfer is a modulation-rate transfer, not a wavelength response.
      Calibrating the chroma carrier's level against the luma's buys one
        direction, rank 2, and no more.
    """
    design = assemble(clusters, operating_point, mechanisms, stock_of, per_stock)
    mech = design["mechanism_columns"]
    names = [design["columns"][i] for i in mech]
    projected = _profile_out(design["matrix"], mech, design["level_columns"])
    result = _analyse(projected, names, precision)
    requested = len([n for n in names if n in mechanisms])
    shared_cols = [i for i in mech if design["columns"][i] in mechanisms]
    shared_rows = [names.index(design["columns"][i]) for i in shared_cols]
    # the rank of the SHARED block answers "can the named split be made",
    # and it is marginalised over EVERYTHING else the data has to pay for:
    # the levels AND every cluster-own or per-stock column. Profiling
    # against the levels alone read a Hi-Fi cluster with its own head as
    # lifting the video's split, when its whole shape was being spent on
    # its own gap and contact.
    nuisance = design["level_columns"] + [i for i in mech if i not in shared_cols]
    shared = (_analyse(_profile_out(design["matrix"], shared_cols, nuisance),
                       [names[i] for i in shared_rows], precision)
              if shared_rows else result)
    refused = shared["rank_at_noise"] < requested
    noise = {c["name"]: (float(np.median(c["se_nepers"])), c["se_source"])
             for c in clusters}
    result.update({
        "requested": requested,
        "shared": shared,
        "refused": bool(refused),
        "precision": float(precision),
        "noise_nepers": noise,
        "operating_point": design["operating_point"],
        "clusters": [c["name"] for c in clusters],
        "columns": design["columns"],
        "arc_measure": rf_stages.effective_count(
            [_profile_out(design["matrix"], shared_cols, nuisance)[:, k]
             for k in range(len(shared_cols))]) if len(shared_cols) > 1 else None,
    })
    if refused:
        pair = shared["worst_pair"]
        result["why"] = (
            "%d mechanisms requested, %d resolved at %.0f%% precision from "
            "clusters %s (structural rank %d): %s and %s are collinear at "
            "coherence %.4f, estimate correlation %.4f; the composite is "
            "determined and the named components are not"
            % (requested, shared["rank_at_noise"], 100 * precision,
               ", ".join(result["clusters"]), shared["structural_rank"],
               pair[0] if pair else "-", pair[1] if pair else "-",
               shared["worst_pair_coherence"], shared["worst_pair_correlation"]))
        result["proof"] = {
            "collinear_pair": pair,
            "coherence": shared["worst_pair_coherence"],
            "correlation": shared["worst_pair_correlation"],
            "singular_values": shared["singular_values"],
            "fractional_error_per_direction": shared["fractional_error_per_direction"],
            "annihilated": shared["annihilated"],
            "noise_nepers": noise,
        }
    else:
        result["why"] = ("%d of %d mechanisms resolved to %.0f%% from clusters %s"
                         % (shared["rank_at_noise"], requested, 100 * precision,
                            ", ".join(result["clusters"])))
    return result


def noise_for_rank(clusters: Sequence[Dict[str, object]], rank: int,
                   scale_cluster: Optional[str] = None,
                   operating_point: Optional[Dict[str, float]] = None,
                   mechanisms: Sequence[str] = MECHANISMS,
                   precision: float = DEFAULT_PRECISION,
                   floor: float = 1e-8) -> Dict[str, object]:
    """AT WHAT NOISE the clusters would resolve `rank` mechanisms.

    Scales the SE of one cluster (or of all, when none is named) and finds
    by bisection the largest SE at which `rank_at_noise` reaches the target.
    Returns that SE in nepers and the factor against the present noise, or
    `reachable=False` when even `floor` nepers does not get there - which is
    the structural verdict, and no instrument improvement changes it."""
    def at(factor: float) -> int:
        scaled = []
        for c in clusters:
            d = dict(c)
            if scale_cluster is None or c["name"] == scale_cluster:
                d["se_nepers"] = c["se_nepers"] * factor
            scaled.append(d)
        return identifiable(scaled, operating_point, mechanisms, precision
                            )["shared"]["rank_at_noise"]

    present = {c["name"]: float(np.median(c["se_nepers"])) for c in clusters}
    reference = (present[scale_cluster] if scale_cluster else
                 float(np.median(list(present.values()))))
    lowest = floor / max(reference, 1e-300)
    if at(1.0) >= rank:
        return {"reachable": True, "already": True, "factor": 1.0,
                "se_nepers": reference}
    if at(lowest) < rank:
        return {"reachable": False, "factor": lowest,
                "se_nepers": reference * lowest,
                "why": "not reached at %.0e nepers: structural, not noise" % floor}
    high, low = 1.0, lowest          # high fails, low succeeds
    for _ in range(60):
        mid = np.sqrt(high * low)
        if at(mid) >= rank:
            low = mid
        else:
            high = mid
    return {"reachable": True, "already": False, "factor": float(low),
            "se_nepers": float(reference * low),
            "improvement_needed": float(1.0 / low)}


# --------------------------------------------------------------------------
# The formulation lever: K stocks on one deck
# --------------------------------------------------------------------------


def formulation_leverage(stocks: Sequence[object],
                         clusters_per_stock: Optional[Sequence[Dict[str, object]]] = None,
                         se_nepers: float = 3e-3,
                         operating_point: Optional[Dict[str, float]] = None,
                         precision: float = DEFAULT_PRECISION,
                         level_free: bool = True,
                         shared_spacing: bool = True) -> Dict[str, object]:
    """WHAT K STOCKS ON THE SAME DECK BUY, under Ethan's premise that
    thickness varies with stock and gap and spacing do not.

    Unknowns: spacing and gap shared across the deck, one thickness per
    stock, and - with `level_free` - one level per (stock, cluster), because
    two stocks are not recorded at one calibrated level. That is 2 + K
    mechanism unknowns (Ethan's count) plus the levels; the equations are
    K times the directions the clusters resolve. The design is BUILT and its
    rank read rather than the count trusted, because the count is an upper
    bound: two rows that are the same shape are one equation.

    Returned per K from 1 to len(stocks): the rank at noise, the condition,
    the fractional errors of spacing, gap and every thickness, and
    `determined` - whether spacing and gap and every stock's thickness all
    reach `precision`. Also the counting statement, and `determined_at_k`,
    the first K at which the built design is determined, or None.

    THE CAVEAT THAT TRAVELS WITH IT. `tape_model.TAPE_VARIATIONS` lists
    surface roughness and debris as a STOCK property that enters Wallace's
    law exactly as head spacing does, and the record side of the chain is
    a different machine for each of this repository's three tapes. The
    premise that spacing is shared is therefore an assumption about the
    stocks' surfaces and about the recorders; `shared_spacing=False` states
    the other case, one spacing per stock and 1 + 2K unknowns. `stocks` may
    be names or thicknesses in metres; a thickness per stock sets that
    stock's operating point.

    MEASURED, cd/a's noise (0.0043 Np, 20 luma points per stock, 10%):
      luma only, recorded depth 0.30-0.40 um: K = 3 resolves 3 of 5
        directions, spacing 149%, gap 339%, thicknesses 446-482%. Never.
      luma only, coating 4-5 um: thickness errors 1e3 to 1e5 at every K.
        A coating is a level, and a level per stock is exactly what the
        free levels absorb. Never.
      luma + colour-under sidebands, coating: DETERMINED AT K = 2 -
        spacing 1.3%, gap 8.0%, thicknesses 0.83%; K = 1 misses on the gap
        at 12.8%; with the levels calibrated K = 1 suffices (0.9/8.1/0.4%).
        With one spacing per stock as well, still K = 2 (gap 8.1%).
      luma + sidebands, recorded depth: K = 3 gives spacing 9.6%, gap 25%,
        thicknesses 24-30% - not at 10%, with or without shared spacing.
    So Ethan's lever is real and it opens in exactly one configuration:
    the coating regime, read through a measured colour-under sideband
    response, with two stocks. In the recorded-depth regime the coating is
    a slope collinear with spacing at every wavelength the format offers.
    """
    per_stock = PER_STOCK if shared_spacing else PER_STOCK + ("spacing",)
    names, thicknesses = [], []
    for k, s in enumerate(stocks):
        if isinstance(s, (int, float)):
            names.append("stock%d" % (k + 1))
            thicknesses.append(float(s))
        else:
            names.append(str(s))
            thicknesses.append(operating_point_for(operating_point)["thickness_m"])
    base = clusters_per_stock or [luma_cluster(se_nepers, se_source="assumed")]
    per_k = []
    determined_at = None
    for count in range(1, len(names) + 1):
        clusters, stock_of, points = [], {}, {}
        for name, thickness in zip(names[:count], thicknesses[:count]):
            for c in base:
                d = dict(c)
                d["name"] = "%s/%s" % (name, c["name"])
                d["level_free"] = bool(level_free)
                d["thickness_m"] = thickness
                clusters.append(d)
                stock_of[d["name"]] = name
        # every stock shares the operating point except its own thickness,
        # which enters through the per-stock column at that stock's value
        result = identifiable(clusters, operating_point, MECHANISMS, precision,
                              stock_of=stock_of, per_stock=per_stock)
        # the per-stock columns are in the full result, not the shared
        # block; gather every wanted parameter's marginal error
        full = result["fractional_se"]
        wanted = {n: e for n, e in full.items()
                  if n in ("spacing", "gap") or "@" in n}
        determined = bool(wanted) and all(e <= precision for e in wanted.values())
        if determined and determined_at is None:
            determined_at = count
        per_k.append({
            "stocks": count,
            "unknowns_mechanisms": (2 + count if shared_spacing else 1 + 2 * count),
            "unknowns_levels": count * len(base) if level_free else 0,
            "rows": sum(len(c["wavelength_m"]) for c in clusters),
            "rank_at_noise": result["rank_at_noise"],
            "structural_rank": result["structural_rank"],
            "condition": result["condition"],
            "fractional_se": wanted,
            "determined": determined,
            "worst_pair": result["worst_pair"],
            "worst_pair_coherence": result["worst_pair_coherence"],
        })
    return {
        "stocks": names,
        "per_k": per_k,
        "determined_at_k": determined_at,
        "counting": ("unknowns %s mechanisms%s; equations K x (directions "
                     "the clusters resolve); the built design's rank decides"
                     % ("2 + K" if shared_spacing else "1 + 2K",
                        " + K levels per cluster" if level_free else "")),
        "shared_spacing": bool(shared_spacing),
        "caveat": ("shared spacing assumes the stocks' surfaces add the same "
                   "separation (tape_model: surface roughness is a stock "
                   "property in Wallace's law)"),
    }


# --------------------------------------------------------------------------
# What IS determined: the composite, with its error bar
# --------------------------------------------------------------------------


def composite_only(wavelength, measured_log_nepers, se_nepers,
                   name: str = "composite",
                   identifiability: Optional[Dict[str, object]] = None
                   ) -> Dict[str, object]:
    """The composite H(lambda) one tape determines, with its error bar, and
    the named components marked UNIDENTIFIED with the reason.

    The composite is returned as it was measured - the log magnitude in
    nepers, its SE, and the equivalent in decibels - and NOT as a sum of
    fitted mechanisms, because a sum of fitted mechanisms is the very
    apportionment the identifiability test refuses. Where `identifiability`
    is the result of `identifiable` on the same clusters, each mechanism's
    entry carries the proof; otherwise it is marked unidentified without one.

    MEASURED, the composite the real exports determine - the decoded luma
    output as a ratio to the spec sync pulse, in decibels with its SE at
    the information spacing, head a of each tape:

        MHz        cd            home          pnb
        0.59   -1.72 +- 0.01  -0.41 +- 0.01  +0.45 +- 0.03
        0.95   -2.99 +- 0.01  -1.24 +- 0.01  -0.45 +- 0.02
        2.04   -7.40 +- 0.03  -6.17 +- 0.03  -6.54 +- 0.07
        2.95  -28.59 +- 0.70 -16.75 +- 0.31 -13.25 +- 0.36

    The two heads agree to 0.1-0.3 dB below 2 MHz (home's +14 dB at 1.5 MHz
    is its known ring, not a loss). That composite is the whole chain's -
    emphasis mismatch, the carrier law, the deck's equalisation and the
    decoder's own 3.3 MHz low-pass included - and as a function of
    wavelength it is determined only modulo exp(a + b/lambda), a gain and a
    Wallace spacing: the carrier law's null space.
    """
    lam = np.asarray(wavelength, dtype=np.float64).ravel()
    logged = np.asarray(measured_log_nepers, dtype=np.float64).ravel()
    se = np.broadcast_to(np.asarray(se_nepers, dtype=np.float64), lam.shape)
    components: Dict[str, object] = {}
    for mechanism in MECHANISMS:
        entry = {"status": "unidentified"}
        if identifiability is not None:
            entry["status"] = ("unidentified" if identifiability["refused"]
                               else "resolved to %.0f%%"
                               % (100 * identifiability["precision"]))
            entry["fractional_se"] = identifiability["shared"]["fractional_se"].get(mechanism)
            if identifiability.get("proof"):
                entry["proof"] = identifiability["proof"]
        components[mechanism] = entry
    return {
        "name": str(name),
        "wavelength_m": lam,
        "log_nepers": logged,
        "se_nepers": np.array(se),
        "db": logged * DB_PER_NEPER,
        "se_db": np.array(se) * DB_PER_NEPER,
        "components": components,
        "why": ("the composite is the measurement; the named components are "
                "an apportionment the clusters do not determine"),
    }


def separate(clusters: Sequence[Dict[str, object]], measured: Sequence[np.ndarray],
             operating_point: Optional[Dict[str, float]] = None,
             precision: float = DEFAULT_PRECISION) -> Dict[str, object]:
    """THE GATED FIT. Refuses - returning the identifiability verdict and no
    numbers - when `identifiable` says the three-way split is not resolved;
    otherwise fits spacing, gap, thickness and the clusters' levels by
    weighted least squares and reports each with its standard error.

    `measured` is one log-magnitude array per cluster, in nepers, on that
    cluster's wavelengths. The clusters' `shares` and `spacing_offset_m` are
    honoured exactly as in the design."""
    verdict = identifiable(clusters, operating_point, MECHANISMS, precision)
    if verdict["refused"]:
        return {"refused": True, "verdict": verdict, "why": verdict["why"]}
    from scipy.optimize import least_squares

    design = assemble(clusters, operating_point)
    columns = design["columns"]
    point = design["operating_point"]
    starts = {"spacing": point["spacing_m"], "gap": point["gap_m"],
              "thickness": point["thickness_m"]}

    def start_for(name: str) -> float:
        base = name.split(":")[-1].split("@")[0]
        return 0.0 if base == "level" else starts[base]

    x0 = np.array([np.log(start_for(n)) if not n.endswith(":level") else 0.0
                   for n in columns])
    target = np.concatenate([np.asarray(m, dtype=np.float64).ravel()
                             for m in measured])
    weight = np.concatenate([1.0 / np.maximum(c["se_nepers"], 1e-300)
                             for c in clusters])

    def model(x: np.ndarray) -> np.ndarray:
        values = {n: (np.exp(v) if not n.endswith(":level") else v)
                  for n, v in zip(columns, x)}
        out = []
        for c in clusters:
            lam = c["wavelength_m"]
            def get(mech):
                key = mech if mech in c["shares"] else "%s:%s" % (c["name"], mech)
                return values[key]
            out.append(composite_log(
                lam, get("spacing") + c.get("spacing_offset_m", 0.0),
                get("gap"), get("thickness"),
                values.get("%s:level" % c["name"], 0.0)))
        return np.concatenate(out)

    solution = least_squares(lambda x: weight * (model(x) - target), x0,
                             max_nfev=4000)
    jac = solution.jac
    cov = np.linalg.pinv(jac.T @ jac)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    fitted = {}
    for n, v, e in zip(columns, solution.x, se):
        if n.endswith(":level"):
            fitted[n] = {"value": float(v), "se": float(e), "units": "nepers"}
        else:
            fitted[n] = {"value": float(np.exp(v)), "se_fractional": float(e),
                         "se_m": float(np.exp(v) * e), "units": "m"}
    return {"refused": False, "verdict": verdict, "fitted": fitted,
            "residual_rms_nepers": float(np.sqrt(np.mean(
                ((model(solution.x) - target)) ** 2)))}


# --------------------------------------------------------------------------
# The real exports: the demodulated instrument, and what it can see
# --------------------------------------------------------------------------


def load_sync_step_export(path: str, head: str = "a") -> Dict[str, object]:
    """One head of a `*_tree_noeq_sync_step_response.npz` export: the whole-
    pulse response `pulse_H` at the decoded luma output, its per-bin SE, the
    valid mask, and the information spacing.

    The response is a RATIO to the spec pulse, so its log is in nepers and
    the SE in nepers is `se / |H|`. THE BINS ARE FINER THAN THE INFORMATION:
    the export's own `resolution_mhz` says one independent value per ~0.18
    MHz against bins of 3.5 kHz, so the valid band is decimated to that
    spacing before it enters any design - keeping every bin would overstate
    the evidence fifty-fold."""
    z = np.load(path, allow_pickle=True)
    f = np.asarray(z["frequency_mhz"], dtype=np.float64) * 1e6
    H = np.asarray(z["head_%s_pulse_H" % head])
    se = np.asarray(z["head_%s_pulse_se" % head], dtype=np.float64)
    valid = np.asarray(z["head_%s_pulse_valid" % head], dtype=bool)
    resolution = max(float(z["head_%s_fall_resolution_mhz" % head]),
                     float(z["head_%s_rise_resolution_mhz" % head])) * 1e6
    carriers = {view: float(z["head_%s_%s_carrier_hz" % (head, view)])
                for view in ("fall", "rise")}
    magnitude = np.abs(H)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_h = np.where(magnitude > 0, np.log(np.maximum(magnitude, 1e-300)), np.nan)
        se_nepers = np.where(magnitude > 0, se / np.maximum(magnitude, 1e-300), np.nan)
    good = valid & np.isfinite(log_h) & np.isfinite(se_nepers) & (f > 0)
    # decimate to the information spacing
    picked = []
    last = -np.inf
    for i in np.flatnonzero(good):
        if f[i] - last >= resolution:
            picked.append(i)
            last = f[i]
    picked = np.array(picked, dtype=int)
    return {
        "path": str(path), "head": str(head),
        "frequency_hz": f, "log_nepers": log_h, "se_nepers": se_nepers,
        "valid": good, "resolution_hz": resolution, "carriers_hz": carriers,
        "independent": picked,
        "median_se_nepers": float(np.median(se_nepers[picked])) if len(picked) else float("nan"),
        "site": str(z["metadata"])[:120],
    }


def demodulated_design(export: Dict[str, object],
                       operating_point: Optional[Dict[str, float]] = None,
                       mechanics: Optional[Dict[str, float]] = None,
                       mechanisms: Sequence[str] = MECHANISMS
                       ) -> Dict[str, object]:
    """THE DESIGN THE EXPORTS ACTUALLY SUPPORT: the mechanisms' sensitivities
    seen through the carrier-normalisation law, at the export's own
    baseband bins and noise.

    THE THREE TAPES, and what is and is not the same across them. cd is
    `countdown.flac`, an off-air recording through a tuner (Ethan: "the cd
    sample was an over the air recording"), and home is `home.flac`, a 1994
    home recording; both were captured at 40 MSps on recorders the
    repository does not name. pnb is the zaroff 75-bars pattern, captured
    at 50 MSps as `zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac`, the
    Sony SLV-778HF being named on the PLAYBACK capture; which machine
    recorded it, and whether cd and home were played on that same deck,
    is not recorded here. So across the three the PLAYBACK head's gap and
    contact are shared at most, and only if the player was the same; the
    stock is three stocks (three coatings, three surfaces); the recorder is
    three recorders, each with its own record-side loss. Ethan's "same
    deck" lever therefore holds, if at all, for the playback half of the
    chain only, and `formulation_leverage(shared_spacing=False)` is the
    honest configuration for these three.

    A demodulated response is not the head's response. The measurement at
    baseband f_b reads the RF response either side of the landing carrier
    by `head_model.predicted_video_effect`'s law, half the sum of the two
    sidebands less the carrier's own value:

        J_video(f_b) = [J(lambda(f_c + f_b)) + J(lambda(f_c - f_b))]/2
                       - J(lambda(f_c))

    which annihilates a constant and any function linear in frequency
    EXACTLY. A level is a constant and Wallace's spacing is -2 pi d f / v -
    linear - so both leave this instrument as nothing; gap and thickness
    survive by their curvature. The rows are the mean over the two edges'
    landing carriers (sync tip for the fall, blanking for the rise, from
    the export), whitened by the per-bin SE at the information spacing."""
    point = operating_point_for(operating_point)
    speed = writing_speed(mechanics)
    f_b = export["frequency_hz"][export["independent"]]
    se = export["se_nepers"][export["independent"]]
    rows = []
    for fb, s in zip(f_b, se):
        views = []
        for carrier in export["carriers_hz"].values():
            if fb >= carrier:
                continue
            upper = jacobian(wavelength_m(np.array([carrier + fb]), speed),
                             point["spacing_m"], point["gap_m"],
                             point["thickness_m"], mechanisms)[0]
            lower = jacobian(wavelength_m(np.array([carrier - fb]), speed),
                             point["spacing_m"], point["gap_m"],
                             point["thickness_m"], mechanisms)[0]
            at = jacobian(wavelength_m(np.array([carrier]), speed),
                          point["spacing_m"], point["gap_m"],
                          point["thickness_m"], mechanisms)[0]
            views.append(0.5 * (upper + lower) - at)
        if views:
            rows.append(np.mean(views, axis=0) / max(float(s), 1e-300))
    matrix = np.array(rows) if rows else np.zeros((0, len(mechanisms)))
    return {"matrix": matrix, "columns": list(mechanisms),
            "baseband_hz": f_b, "se_nepers": se, "operating_point": point,
            "carriers_hz": export["carriers_hz"]}


def demodulated_identifiable(export: Dict[str, object],
                             operating_point: Optional[Dict[str, float]] = None,
                             precision: float = DEFAULT_PRECISION,
                             mechanics: Optional[Dict[str, float]] = None
                             ) -> Dict[str, object]:
    """`identifiable` for the demodulated instrument the exports are.
    Spacing's column is annihilated by the carrier law and is reported so,
    with its residual norm relative to the largest column.

    MEASURED on all six export heads (cd, home, pnb; a and b; 20 independent
    bins each): spacing's column is 6e-15 to 8e-15 of the largest -
    annihilated, error bar infinite - structural rank 2, rank at noise 1 of
    3, condition 1.5e14 to 2.2e14. REFUSED on every head. At delta 0.30 um
    gap and thickness are collinear at coherence 0.9999-1.0000 with errors
    of 320-580% and 540-980%: the export determines ONE thing about the
    losses, the curvature of log|H| about the carrier. At delta 4.5 um the
    pair separates (coherence 0.46-0.51) and the gap alone would read to
    2.8-7.9% with thickness at 14-21% - still one direction at 10%, and
    still no spacing."""
    design = demodulated_design(export, operating_point, mechanics)
    result = _analyse(design["matrix"], design["columns"], precision)
    norms = np.linalg.norm(design["matrix"], axis=0)
    result["column_norm_relative"] = dict(zip(
        design["columns"], map(float, norms / max(norms.max(), 1e-300))))
    result["requested"] = len(MECHANISMS)
    result["refused"] = result["rank_at_noise"] < len(MECHANISMS)
    result["instrument"] = "demodulated luma output through the carrier law"
    result["independent_bins"] = int(design["matrix"].shape[0])
    result["median_se_nepers"] = export["median_se_nepers"]
    result["why"] = (
        "the carrier law annihilates a constant and a linear tilt: spacing's "
        "column is %.1e of the largest; %d of 3 resolved at %.0f%%"
        % (result["column_norm_relative"]["spacing"], result["rank_at_noise"],
           100 * precision))
    return result


# --------------------------------------------------------------------------
# The reference table, from specification
# --------------------------------------------------------------------------


def wavelength_table(mechanics: Optional[Dict[str, float]] = None,
                     coatings_m: Sequence[float] = ASSUMED_COATING_M,
                     recorded_depth_m: float = SPEC_RECORDED_DEPTH_M,
                     system: str = "NTSC") -> Dict[str, object]:
    """Ethan's wavelength reference numbers, reproduced from specification.

    4 f_sc from SMPTE 170M's subcarrier (`colour_under.subcarrier_hz`); the
    writing speed from JVC VTG82063 section 1 as stated
    (`head_model.mechanics_for`); carriers from IEC 60774-1
    (`rf_stages.VHS_CARRIER_HZ`); the colour-under from 40 f_H
    (`colour_under.carrier_hz`); Hi-Fi from SMPTE 32M clause 5; the read
    depth lambda/2pi from the self-demagnetisation bound
    (`magnetic.DEMAGNETISATION_DIVISOR`, docs/MATHEMATICS.md 8a); the
    thickness loss at the assumed coatings AND at the JVC recorded depth,
    side by side, because which of the two enters the law is the question
    this module does not settle."""
    mechanics = mechanics or default_mechanics()
    speed = head_model.writing_speed(mechanics)
    four_fsc = 4.0 * colour_under.subcarrier_hz(system)
    period = 1.0 / four_fsc
    tip, white = rf_stages.VHS_CARRIER_HZ["525" if system.upper() == "NTSC" else "625"]
    mid = 0.5 * (tip + white)
    chroma = colour_under.carrier_hz(system)
    points = {
        "sync_tip": tip, "mid_band": mid, "peak_white": white,
        "colour_under": chroma,
        "hifi_ch1": HIFI_CARRIER_HZ[0], "hifi_ch2": HIFI_CARRIER_HZ[1],
    }
    rows = {}
    for name, f in points.items():
        lam = float(speed / f)
        row = {"frequency_hz": float(f), "wavelength_m": lam,
               "read_depth_m": lam / magnetic.DEMAGNETISATION_DIVISOR,
               "thickness_loss_db": {}}
        for delta in list(coatings_m) + [recorded_depth_m]:
            loss = float(thickness_log(np.array([lam]), delta)[0]) * DB_PER_NEPER
            row["thickness_loss_db"]["%.1f um" % (delta * 1e6)] = loss
        rows[name] = row
    return {
        "writing_speed_m_s": speed,
        "four_fsc_hz": four_fsc,
        "sample_period_s": period,
        "tape_per_sample_m": speed * period,
        "rows": rows,
        "sources": {
            "writing_speed": "JVC VTG82063 s.1, stated 5.80 m/s (not 5.877 derived)",
            "carriers": "IEC 60774-1 via rf_stages.VHS_CARRIER_HZ",
            "colour_under": "40 f_H, derived (colour_under.carrier_provenance)",
            "hifi": "SMPTE 32M clause 5",
            "four_fsc": "SMPTE 170M subcarrier x 4",
            "read_depth": "lambda/(2 pi), self-demagnetisation (MATHEMATICS 8a)",
            "coating": "ASSUMED 4-5 um (Ethan); no specification states it",
            "recorded_depth": "JVC VTG82063 s.7.2.1, about 0.3 um",
        },
    }


def format_table(table: Dict[str, object]) -> str:
    lines = ["writing speed %.2f m/s; 4 fsc %.4f MHz; %.2f ns/sample; "
             "%.4f um of tape per sample"
             % (table["writing_speed_m_s"], table["four_fsc_hz"] / 1e6,
                table["sample_period_s"] * 1e9, table["tape_per_sample_m"] * 1e6)]
    deltas = None
    for name, row in table["rows"].items():
        if deltas is None:
            deltas = list(row["thickness_loss_db"])
            lines.append("%-13s %10s %10s %10s  " % ("point", "f (MHz)", "lambda um", "read um")
                         + "  ".join("%12s" % ("loss@" + d) for d in deltas))
        lines.append("%-13s %10.4f %10.4f %10.4f  " % (
            name, row["frequency_hz"] / 1e6, row["wavelength_m"] * 1e6,
            row["read_depth_m"] * 1e6)
            + "  ".join("%9.2f dB" % row["thickness_loss_db"][d] for d in deltas))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# THE TIME AXIS: the record's own two halves
# --------------------------------------------------------------------------

# Everything above this line is a function of WAVELENGTH alone. The three
# mechanisms are geometric laws with no level in them, which is why this
# module carries no amplitude axis and why that absence is the physics
# rather than an omission: a spacing, a gap and a magnetised depth do not
# depend on how large the signal being written is, except through the
# recorded depth, whose level dependence `magnetic.level_signature` owns and
# `rf_stages.record_current_level_dependence` enters.
#
# TIME IS DIFFERENT, AND ITS ABSENCE WAS AN OMISSION. Every quantity this
# module inverts is read from `load_sync_step_export`, which takes the
# export's accumulation over the WHOLE record and returns one response with
# one standard error. That error is the sampling error of a mean, not the
# reproducibility of the thing being measured, and the two are different
# whenever the transport moves the geometry during the pass - which
# `tape_path` says it must, the pack radius changing through a pass being
# one of its seven separation mechanisms, and which the arc has already
# measured directly (the per-field fold of `tesseract.from_field_series`
# finds structure at every time scale from the head to two seconds).
#
# THE PAIR IS ALREADY IN THE EXPORTS. `sync_step_response` accumulates each
# polarity twice more, over the first half of that head's fields and over
# the second, and writes `head_<h>_<view>_H_first` and `_H_second`. Those
# are two independent measurements of one quantity, separated in TIME along
# the record, and they can disagree. That is the whole of the discipline
# this module already applies on the wavelength axis, applied to the axis it
# did not have.
#
# TWO RULES FROM THE EXPORT ITSELF, both followed here rather than invented:
#
#   WHICH PART MAY BE FOLDED. The export's metadata states that
#   `H_even = (H_fall + H_rise)/2 is the only part an LTI consumer may
#   fold; H_odd = (H_fall - H_rise)/2 is a validity bound, never
#   corrected`. The loss mechanisms are LTI, so the halves are compared on
#   the even part and never on a single polarity. The per-view halves are
#   returned as well, because a disagreement that appears on one polarity
#   and not the other is a polarity contrast and not a drift, and telling
#   the two apart is the reason to keep both.
#
#   WHAT A HALF'S ERROR IS. Each half pools half the fields, so its
#   variance is twice the full accumulation's - the root-two convention
#   `residual_floor.load_export` already states - and the difference of the
#   two halves therefore has variance 2v + 2v = 4v, standard error exactly
#   TWICE the export's own. No new noise model is introduced; the factor is
#   arithmetic from a convention the repository already holds.
#
# MEASURED, on the three exports at the information spacing (20 independent
# bins over 0.04-3.50 MHz, the level projected out because a level is free
# in every fit this module performs), chi-square per degree of freedom of
# the halves' shape disagreement against that doubled standard error:
#
#     tape   head a          head b          verdict
#     cd     0.88 (p 0.61)   0.99 (p 0.47)   at noise on the even part
#     home   1.87 (p 0.012)  2.52 (p 2.6e-4) DRIFTS, both heads
#     pnb    0.80 (p 0.70)   0.44 (p 0.98)   at noise
#
# The single-polarity views separate one further thing. On cd the FALL view
# reads 2.57 and 3.44 (p 1.9e-4 and 5.4e-7) while the RISE view reads 0.65
# and 0.63 (p 0.87 and 0.88), so cd's half-to-half difference is
# polarity-owned and CANCELS in the even part - a half-by-polarity cross
# term, not a drift of the losses, and it is the even part that the loss
# mechanisms live in. home drifts on every view - fall 2.33 and 2.57, rise
# 2.07 and 2.40, even 1.87 and 2.52 - which is what a change in the LTI
# channel looks like, and pnb drifts on none of them.
#
# AND WHETHER THE FILE CAN SHOW IT, by Ethan's cap. Filtering each export's
# own measured mean sync edge by the even-part half difference and reading
# the change in IRE - the file's own units - gives 0.1505 and 0.1868 IRE on
# cd, 0.1394 and 0.1027 on home, 0.0443 and 0.0418 on pnb, against the
# ten-bit 4 f_sc quantisation floor of 0.0461 IRE rms
# (`output_limit.output_profile`). On cd and home the record's two halves
# put a DIFFERENT edge in the file, by 2.2 to 4.1 times the smallest
# difference the file can hold; on pnb the difference is 0.96 and 0.91 of
# the floor and is invisible rather than small. cd is the instructive case:
# visible without being significant, because its two halves differ by more
# than the file can show while the export's own noise up the band is that
# large as well - so the difference is real in the file and not
# attributable to a drift. The two verdicts answer different questions and
# neither implies the other.


def load_sync_step_halves(path: str, head: str = "a",
                          view: str = "even") -> Dict[str, object]:
    """The two halves of one head's record, on the information spacing.

    `view` is `"even"` - the LTI-foldable half sum of the two polarities,
    which is the only part the export permits a linear consumer to use -
    or `"fall"` or `"rise"` for one polarity alone, which exists so that a
    disagreement can be attributed to a drift or to polarity.

    Returns `first` and `second` shaped exactly as `load_sync_step_export`
    returns one accumulation, so `demodulated_design` and
    `demodulated_identifiable` run on either half unchanged, together with
    the shared frequency grid, the independent-bin index and
    `se_difference_nepers`, the standard error of `first - second`: twice
    the full accumulation's, because each half pools half the fields.
    """
    z = np.load(path, allow_pickle=True)
    f = np.asarray(z["frequency_mhz"], dtype=np.float64) * 1e6
    if view == "even":
        parts = ("fall", "rise")
    elif view in ("fall", "rise"):
        parts = (view,)
    else:
        raise ValueError("view must be 'even', 'fall' or 'rise', not %r" % view)

    def stack(suffix: str) -> np.ndarray:
        return np.mean([np.asarray(z["head_%s_%s%s" % (head, p, suffix)])
                        for p in parts], axis=0)

    full = stack("_H")
    first, second = stack("_H_first"), stack("_H_second")
    # the even part's SE is the half sum of two independent polarities
    se = np.sqrt(np.mean([np.asarray(
        z["head_%s_%s_se" % (head, p)], dtype=np.float64) ** 2
        for p in parts], axis=0)) / np.sqrt(len(parts))
    valid = np.logical_and.reduce([np.asarray(
        z["head_%s_%s_valid" % (head, p)], dtype=bool) for p in parts])
    resolution = max(float(z["head_%s_%s_resolution_mhz" % (head, p)])
                     for p in ("fall", "rise")) * 1e6
    carriers = {v: float(z["head_%s_%s_carrier_hz" % (head, v)])
                for v in ("fall", "rise")}

    magnitude = np.abs(full)
    good = (valid & np.isfinite(first) & np.isfinite(second)
            & np.isfinite(se) & (magnitude > 0) & (se > 0) & (f > 0))
    picked, last = [], -np.inf
    for i in np.flatnonzero(good):
        if f[i] - last >= resolution:
            picked.append(i)
            last = f[i]
    picked = np.array(picked, dtype=int)

    with np.errstate(divide="ignore", invalid="ignore"):
        se_nepers = se / np.maximum(magnitude, 1e-300)

    def as_export(values: np.ndarray, label: str) -> Dict[str, object]:
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.where(np.abs(values) > 0,
                              np.log(np.maximum(np.abs(values), 1e-300)),
                              np.nan)
        # a half pools half the fields: root-two the full error
        half_se = se_nepers * np.sqrt(2.0)
        return {
            "path": str(path), "head": str(head), "half": label,
            "frequency_hz": f, "log_nepers": logged, "se_nepers": half_se,
            "valid": good, "resolution_hz": resolution,
            "carriers_hz": carriers, "independent": picked,
            "median_se_nepers": (float(np.median(half_se[picked]))
                                 if picked.size else float("nan")),
            "site": str(z["metadata"])[:120],
        }

    return {
        "path": str(path), "head": str(head), "view": str(view),
        "frequency_hz": f, "independent": picked, "resolution_hz": resolution,
        "carriers_hz": carriers,
        "first": as_export(first, "first"),
        "second": as_export(second, "second"),
        # var(first - second) = 2v + 2v = 4v, so the SE is twice the full one
        "se_difference_nepers": 2.0 * se_nepers,
        "full_log_nepers": np.log(np.maximum(magnitude, 1e-300)),
        "why": ("H_first and H_second are the export's own first and second "
                "halves of that head's fields: one quantity, two "
                "measurements, separated along the record"),
    }


def half_drift(path: str, head: str = "a", view: str = "even"
               ) -> Dict[str, object]:
    """DOES THE RECORD'S SECOND HALF MEASURE THE SAME LOSSES AS ITS FIRST?

    The halves' log-magnitude difference at the information spacing, split
    into a free LEVEL and the SHAPE, tested against the doubled standard
    error. The level is projected out with inverse-variance weights and is
    not tested, because every fit in this module leaves a level free and a
    level that moved would change no mechanism.

    Reported: the level, the shape's chi-square per degree of freedom and
    its p-value, the per-bin z, and `drifts` - the verdict that the two
    halves are not the same measurement. Also `output` from
    `output_limit.is_meaningful`, which asks the different question of
    whether the difference is large enough for the file to hold at all;
    the two verdicts are independent and both are reported, because a
    difference can be significant and invisible, or visible and not
    attributable.
    """
    from scipy import stats

    loaded = load_sync_step_halves(path, head, view)
    idx = loaded["independent"]
    if idx.size < 3:
        return {"usable": False, "why": "fewer than three independent bins"}
    first = loaded["first"]["log_nepers"][idx]
    second = loaded["second"]["log_nepers"][idx]
    difference = first - second
    error = loaded["se_difference_nepers"][idx]
    weight = 1.0 / np.maximum(error, 1e-300) ** 2
    level = float(np.sum(weight * difference) / np.sum(weight))
    shape = difference - level
    z = shape / np.maximum(error, 1e-300)
    chi2 = float(np.sum(z ** 2))
    dof = int(idx.size - 1)                     # the level costs one
    return {
        "usable": True,
        "path": str(path), "head": str(head), "view": str(view),
        "frequency_hz": loaded["frequency_hz"][idx],
        "difference_nepers": difference,
        "se_difference_nepers": error,
        "level_nepers": level,
        "shape_nepers": shape,
        "shape_rms_nepers": float(np.sqrt(np.mean(shape ** 2))),
        "z": z,
        "max_abs_z": float(np.max(np.abs(z))),
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / max(dof, 1),
        "p_value": float(stats.chi2.sf(chi2, dof)),
        "drifts": bool(stats.chi2.sf(chi2, dof) < 0.05),
        "output": output_visibility(path, head, view),
        "why": ("the level is projected out because every fit here leaves "
                "one free; what is tested is the SHAPE, which is what "
                "reaches the mechanisms"),
    }


def output_visibility(path: str, head: str = "a", view: str = "even"
                      ) -> Dict[str, object]:
    """Can the output file show the difference between the record's halves?

    The half difference is a log-magnitude departure, and the file holds
    IRE. The conversion uses no assumption: the export carries that head's
    own measured mean sync edge in IRE (`head_<h>_fall_mean`, referenced to
    the tbc.json black and white levels), so the departure is applied to it
    as a filter - half of it each way, so the two halves sit either side of
    the accumulation - and what is read is the rms change in IRE. That is
    compared against the ten-bit 4 f_sc quantisation floor by
    `output_limit.is_meaningful`.
    """
    z = np.load(path, allow_pickle=True)
    loaded = load_sync_step_halves(path, head, view)
    idx = loaded["independent"]
    if idx.size < 3:
        return {"usable": False, "why": "fewer than three independent bins"}
    f = loaded["frequency_hz"]
    difference = (loaded["first"]["log_nepers"][idx]
                  - loaded["second"]["log_nepers"][idx])
    rate = float(z["rate_mhz"]) * 1e6
    edge = np.asarray(z["head_%s_fall_mean" % head], dtype=np.float64)
    length = int(2 ** int(np.ceil(np.log2(max(edge.size, 2)))) * 8)
    spectrum = np.fft.rfft(edge, n=length)
    grid = np.fft.rfftfreq(length, 1.0 / rate)
    # the departure only where it was measured; zero outside, so nothing is
    # extrapolated into bins the export declares undetermined
    inside = (grid >= f[idx].min()) & (grid <= f[idx].max())
    gain = np.zeros_like(grid)
    gain[inside] = np.interp(grid[inside], f[idx], difference)
    upper = np.fft.irfft(spectrum * np.exp(0.5 * gain), n=length)[:edge.size]
    lower = np.fft.irfft(spectrum * np.exp(-0.5 * gain), n=length)[:edge.size]
    verdict = output_limit.is_meaningful(upper - lower)
    verdict["usable"] = True
    verdict["edge_samples"] = int(edge.size)
    verdict["band_hz"] = (float(f[idx].min()), float(f[idx].max()))
    verdict["why"] = ("the halves' difference applied to that head's own "
                      "measured sync edge, in the file's own units, against "
                      "the file's own floor")
    return verdict


def composite_over_time(path: str, head: str = "a", view: str = "even"
                        ) -> Dict[str, object]:
    """THE COMPOSITE, WITH A TIME AXIS: one per half of the record.

    `composite_only` is what this module hands over, because it is what one
    tape determines. This returns it TWICE - once for each half - together
    with the drift verdict, so a caller reads not one composite but the two
    the record actually produced and the statement of whether they are the
    same measurement.

    When `drifts` is false the two are one and the accumulation may be used
    as it always was. When it is true the accumulation is a mean over a
    quantity that moved, and its standard error understates the spread of
    the thing being inverted; the honest reading is then the pair, and the
    module refuses to collapse it for the same reason it refuses to split
    the three mechanisms.
    """
    loaded = load_sync_step_halves(path, head, view)
    idx = loaded["independent"]
    speed = writing_speed()
    lam = wavelength_m(loaded["frequency_hz"][idx], speed)
    drift = half_drift(path, head, view)
    out = {"path": str(path), "head": str(head), "view": str(view),
           "wavelength_m": lam, "drift": drift}
    for label in ("first", "second"):
        part = loaded[label]
        out[label] = composite_only(
            lam, part["log_nepers"][idx], part["se_nepers"][idx],
            name="%s %s half" % (head, label))
    out["one_measurement"] = bool(drift.get("usable") and not drift["drifts"])
    out["why"] = ("the composite is what one tape determines; this says "
                  "whether one RECORD determines one composite")
    return out
