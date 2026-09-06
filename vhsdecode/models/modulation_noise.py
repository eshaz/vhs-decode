"""Modulation noise: the coating's own amplitude modulation of the carrier,
and what the constant-envelope probe must do about it.

Ethan: *"Modulation noise - breaks the constant-envelope probe. Particulate
media put AM on the carrier, roughly proportional to signal. The envelope
probe assumes all envelope variation is channel. This term isn't. Frozen in
the coating: reproduces across replays of one tape, doesn't average out with
re-capture. Weight envelope-derived |H| by an estimate of this rather than
treating all envelope variance as channel."*

THE MODEL. The luma carrier is recorded at constant amplitude, so the
envelope that comes back is read as the channel's action on a constant. What
actually comes back, in every cell where the channel's action A is constant,
is

    e(t) = A * (1 + m(t)) + n(t)

with m the MULTIPLICATIVE term - the coating's own variation, which scales
with whatever was written - and n the ADDITIVE term - head and preamplifier
thermal noise, the converter, and the tape's bulk (erased-state) particle
noise, none of which knows what was written. Within a cell, to first order,

    mean(e)     = A
    variance(e) = A^2 sigma_m^2 + sigma_n^2

so across cells at different levels the variance is a STRAIGHT LINE in the
squared mean: its slope is the modulation-noise power sigma_m^2 and its
intercept the additive power sigma_n^2. That is the whole discriminant -
signal-proportional variance against level-independent variance - and it
needs no model of either mechanism's spectrum. `estimate` is that line,
fitted with the weights the variance of a sample variance demands, with the
errors-in-variables attenuation corrected, and with its error bar taken
twice: from the fit's own covariance and from a jackknife over fields.

sigma_n^2 here is the additive noise's IN-PHASE quadrature only. Around a
carrier well above the noise the quadrature component moves the phase, not
the envelope (`tools/ringing_measure/rf_noise.py`), so the intercept is half
the complex additive power, and a carrier-to-noise ratio quoted from it must
say so.

THE TWO SIDEBAND IMAGES, which is what makes cells at different carriers
comparable. A multiplicative term is amplitude modulation: its two sidebands
are COHERENT, and through the decoder's zero-phase band-pass R the envelope
sees them summed,

    G_m(f) = ( [R(fc + f) + R(fc - f)] / 2 R(fc) )^2

while additive noise's two sidebands are independent and add in power,

    G_n(f) = [R(fc + f)^2 + R(fc - f)^2] / 2 R(fc)^2.

Where the band-pass is symmetric about the carrier the two are equal; where
it is not the two images part. At the 3.4 MHz sync tip of this format it is
not: measured on the decoder's built NTSC VHS filter at 40 MSps, |R| stands
at -3.0 dB at the tip, -1.2 dB at blanking, -0.1 dB at 4.4 MHz, but
-11.5 dB at 2 MHz, -55 dB at 1 MHz and -22 dB at 6.5 MHz, so from the tip
the lower sideband is cut at 1.4 MHz of offset where the upper still runs
to 2.6. Two consequences. The two equivalent bandwidths (the integrals of
G_m and of G_n, `envelope_bandwidth_hz`) differ from each other and between
the tip and blanking - the values on the built filter are printed by the
measurement tool and recorded in MEASURED - so a cell at one carrier does
not carry the same noise as a cell at another even for identical tape and
electronics, and the regression carries each cell's two bandwidth ratios
from the decoder's own filter (`bandwidth_ratios`) so that the line is
fitted AT THE TIP and read elsewhere through them. And the sideband SPECTRA
of the two terms differ in shape even for a white coating, which
`sideband_spectra` measures.

THE PREDICTION, WRITTEN BEFORE THE MEASUREMENT (2026-09-05). The particulate
model gives the multiplicative term a size with no free parameter once the
particle count is fixed. A head reads the particles inside an aperture that
slides along the track; the count in it fluctuates; a density fluctuation
delta N / N multiplies the recorded magnetisation directly, so the
modulation index is the relative count fluctuation of the read volume:

    sigma_m = sqrt(F / N)

with N the particles in the read volume and F the Fano factor - one for
Poisson packing, larger where particles cluster. The read volume that
matters to the ENVELOPE is the one its own modulation bandwidth resolves: a
box of length L = v / (2 B_m) along the track, times the track width, times
the recording depth, which the format fixes at a quarter wavelength (JVC
VTG82063 section 7.2, as `interference.particle_noise` already cites). With
the repository's own assumptions - particle volume 1e-21 m^3 and packing
0.4, both from `tape_model` and both labelled there as typical rather than
measured - the format's writing speed of 5.80 m/s and SP track width of
58 um, and the decoder's band-pass integrated as built, the read length at
the 3.4 MHz sync tip is 0.84 um and holds 8,300 particles; the noise-budget
lane's rounder figure is about 7,000. So

    SP, Poisson (F = 1), sync tip:      sigma_m = 1.1 to 1.2 per cent
                                        (power 1.2e-4 to 1.4e-4; -39 dB)
    SP at 4.0 to 4.4 MHz:               0.9 to 1.0 per cent (the band-pass
                                        is symmetric there, the read length
                                        1.4 um, 10,700 to 12,200 particles)
    EP, track width 19.3 um, sync tip:  1.9 per cent (2,900 particles);
                                        1.5 to 1.6 per cent at 4.0 to 4.4
    clustered (F > 1):                  all rise as the root of F.

Two further terms are stated so that an excess over the count term has a
name rather than being called error. A fluctuating head-to-tape spacing
modulates the carrier through Wallace's law as m = -2 pi delta d / lambda:
3.7 per cent of modulation index for every 10 nm of rms spacing fluctuation
at the 3.4 MHz carrier, 4.8 per cent at 4.4 MHz. Its size depends on a
surface roughness this module does NOT assume a value for: the sensitivity
is derived, the roughness is left as the unknown, and the measured excess
over the count term is what bounds it. Both this term and the count term
scale with the carrier as f_c^2 in power (the count because the read
volume goes as lambda^2, the spacing because it goes as 1/lambda^2), and
both go as one over the track width, so neither the carrier nor the tape
speed separates them. Their spectra do: the count term is white in
wavenumber down to the particle length and takes the shape G_m; the spacing
term sits close in, below v / (roughness correlation length).

What the FRACTION will be is not predictable from the coating alone,
because it is a ratio against an additive term the electronics contribute
to. The tape's own bulk particle noise puts the additive carrier-to-noise
near 10 log10(N per wavelength) = 40 to 42 dB across the deviation range
(`tape_model.magnetic_limits`), which alone would leave the modulation term
about 0.7 to 0.9 of the constant-carrier envelope variance; the arc's
measured "true random" carrier-to-noise of 33.1 dB would leave it near 0.3;
if the raw 21.6 dB envelope spread were all additive it would be near 0.03.
The measurement decides between these; the numbers are kept beside it in
`PREDICTED`.

WHAT THE ENVELOPE-DERIVED |H| SHOULD DO WITH IT. The channel fit reads |H|
at a carrier bin as the mean envelope there. That mean's variance is

    (sigma_m^2 b_m + sigma_n^2 b_n / A^2) / n_eff

in relative (neper^2) units, with b_m and b_n the bin's bandwidth ratios.
The additive part falls as the level rises and is what the fit should
down-weight the weak bins against; the modulation part is the same at every
level, carries no information about |H|, and sets a floor on any bin's
precision that no amount of level buys back. What the raw envelope variance
contains BEYOND those two is the channel actually varying inside the bin -
the part to model, never to weight against. `weight_for_response` returns
all three.

THE FROZEN TEST. A term that scales with the signal is not thereby the
coating's: a playback-time gain flutter - head contact, an automatic gain
stage - scales with the signal too. And a term that does not scale with the
signal is not thereby the electronics': the bulk particle pattern is on the
tape and replays identically. So two replays of one tape, aligned, give a
second axis, and the two axes together make a two-by-two table:

                        reproduces across replays    does not
    scales with signal  MODULATION NOISE (coating)   gain flutter (playback)
    does not scale      bulk particle pattern (tape) electronics, converter

`frozen_test` fills that table from two aligned passes by regressing the
per-cell cross-covariance and the per-cell non-reproducible variance on the
squared level. The multi-pass lane owns the alignment; this module specifies
the test on already-aligned envelopes and proves it on planted data.

THE BIAS, STATED. (1) Any variation of A INSIDE a cell - the porch
relaxation, a sweep, contact flutter faster than the cell, edge jitter
leaking through the template's slope - inflates the variance in proportion
to A^2 and is read as modulation noise. The cells are therefore taken where
the carrier is constant (sync tip, back porch), the deterministic
line-locked shape is removed by a coherent template first, the interior is
where jitter leakage is below one per cent of the residual variance, and
the regression's slope is an UPPER BOUND on the coating's own term until
the frozen test has said how much of it reproduces. (2) The Rice bias,
mean(|A + n|) = A + sigma_n^2 / 2A, shifts the regressor by about sigma_n^2,
below one per cent of A^2 at any carrier-to-noise above 20 dB. (3) The
regressor is itself measured, and its error attenuates the slope by the
reliability ratio var(x) / (var(x) + mean error variance); with a small
lever arm this reaches several per cent and is corrected from the cells'
own effective counts. (4) The error bar rests on the effective sample count
of a band-limited envelope, never the raw count, following
`head_model.effective_sample_size`; that AR(1) count is conservative on a
band-limited residual, so the count the scatter of variances implies is
reported beside it. (5) Where the levels hardly differ between cells the
slope and intercept are collinear; the estimator reports the lever arm and
the correlation of the two errors, and a bound on the slope, rather than a
number alone. (6) The bandwidth ratios assume the additive noise and the
density fluctuation are white across the band-pass; the head's own tilt
across the sideband image is not included in them, which is why the fit's
primary form uses the tip cells alone, where every ratio is one.

MEASURED. See the module's MEASURED table, appended after the measurement
by `tools/ringing_measure/modulation_noise_measure.py`.

Nothing here is a tuned constant. Levels and pulse widths come from the
decoder's own `SysParams`, the band-pass from its own `Filters["RFVideo"]`,
the track width from its `DecoderParams`, the mechanics from `head_model`,
and every remaining structural choice carries its derivation at the point of
definition.
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from vhsdecode.models import head_model
from vhsdecode.models import tape_model


# --------------------------------------------------------------------------
# The prediction, kept beside the measurement
# --------------------------------------------------------------------------

# Recorded before the measurement was made (2026-09-05), in the units the
# estimator reports: modulation index sigma_m (dimensionless, relative to
# the carrier), at the sync-tip carrier, Poisson packing. Never edited to
# fit the measurement; the MEASURED table sits beside it.
PREDICTED = {
    "modulation_index_sp_tip": (0.011, 0.012),
    "modulation_index_sp_4mhz": (0.009, 0.010),
    "modulation_index_ep_tip": (0.019, 0.021),
    "modulation_index_ep_4mhz": (0.015, 0.016),
    "spacing_index_per_10nm_at_3p4_mhz": 0.037,
    "spacing_index_per_10nm_at_4p4_mhz": 0.048,
    "fraction_if_additive_is_tape_only": (0.7, 0.9),
    "fraction_if_additive_is_true_random_33db": 0.29,
    "fraction_if_additive_is_raw_spread_22db": 0.028,
    "why": ("sigma_m = sqrt(F/N) with N the particles in the volume the "
            "envelope's own modulation bandwidth resolves; the fraction "
            "depends on the additive term, which the electronics set and "
            "the tape only bounds"),
}

# MEASURED 2026-09-05, sync-tip cells only, per head, the larger of the fit
# and jackknife errors; filled in by the measurement tool's run and
# transcribed here. See the tool's docstring for the exact command.
MEASURED: Dict[str, Dict[str, float]] = {}

# The recording depth as a fraction of the wavelength, JVC VTG82063 section
# 7.2 ("a depth equivalent to 1/4th the recording wavelength"), the same
# citation `interference.particle_noise` rests on.
RECORDING_DEPTH_WAVELENGTHS = 0.25


def sideband_image(offset_hz, carrier_hz: float, response_hz, response,
                   coherent: bool, weight: Optional[Callable] = None
                   ) -> np.ndarray:
    """The envelope's view of the two sidebands about one carrier, through
    a real zero-phase response, normalised to one at zero offset.

    coherent=True is the multiplicative term's image G_m (sidebands summed
    in amplitude); coherent=False the additive term's image G_n (summed in
    power). `weight` multiplies the response by a further real gain of
    frequency - the head's own, where a caller has it.
    """
    offsets = np.abs(np.asarray(offset_hz, dtype=np.float64))
    grid = np.asarray(response_hz, dtype=np.float64)
    magnitude = np.abs(np.asarray(response, dtype=np.float64))
    gain = weight if weight is not None else (lambda f: np.ones_like(f))
    upper = np.interp(carrier_hz + offsets, grid, magnitude, right=0.0) \
        * gain(carrier_hz + offsets)
    lower = np.interp(np.abs(carrier_hz - offsets), grid, magnitude, right=0.0) \
        * gain(np.abs(carrier_hz - offsets))
    at_carrier = float(np.interp(carrier_hz, grid, magnitude)
                       * gain(np.array([float(carrier_hz)]))[0])
    if at_carrier <= 0:
        return np.full_like(offsets, np.nan)
    if coherent:
        return (0.5 * (upper + lower) / at_carrier) ** 2
    return 0.5 * (upper ** 2 + lower ** 2) / at_carrier ** 2


def envelope_bandwidth_hz(carrier_hz: float, response_hz, response,
                          coherent: bool = False, **kwargs) -> float:
    """The equivalent noise bandwidth of the envelope about one carrier: the
    integral over positive offsets of the sideband image. Nothing is assumed
    about the filter; it is integrated as built."""
    grid = np.asarray(response_hz, dtype=np.float64)
    offsets = np.linspace(0.0, float(grid.max()), 4096)
    image = sideband_image(offsets, carrier_hz, grid, response, coherent, **kwargs)
    if not np.all(np.isfinite(image)):
        return float("nan")
    return float(np.trapezoid(image, offsets))


def bandwidth_ratios(carrier_hz, reference_hz: float, response_hz, response,
                     **kwargs) -> Dict[str, np.ndarray]:
    """How much more (or less) of each term a cell at `carrier_hz` carries
    than one at `reference_hz`, for white sources: the ratio of the two
    equivalent bandwidths. Both are one at the reference."""
    carriers = np.atleast_1d(np.asarray(carrier_hz, dtype=np.float64))
    reference_m = envelope_bandwidth_hz(reference_hz, response_hz, response, True, **kwargs)
    reference_n = envelope_bandwidth_hz(reference_hz, response_hz, response, False, **kwargs)
    modulation = np.array([envelope_bandwidth_hz(c, response_hz, response, True, **kwargs)
                           / reference_m for c in carriers])
    additive = np.array([envelope_bandwidth_hz(c, response_hz, response, False, **kwargs)
                         / reference_n for c in carriers])
    return {"modulation": modulation, "additive": additive,
            "reference_modulation_bandwidth_hz": reference_m,
            "reference_additive_bandwidth_hz": reference_n}


def read_volume_particles(carrier_hz: float, writing_speed_m_s: float,
                          track_width_m: float,
                          modulation_bandwidth_hz: Optional[float] = None,
                          particle_volume_m3: float = tape_model.PARTICLE_VOLUME_M3,
                          packing: float = tape_model.PACKING_FRACTION,
                          depth_wavelengths: float = RECORDING_DEPTH_WAVELENGTHS
                          ) -> Dict[str, float]:
    """How many particles the envelope's read volume holds at one carrier.

    Two conventions, both returned. The arc's `tape_model.magnetic_limits`
    counts the particles under ONE WAVELENGTH of track; the envelope's
    multiplicative term resolves a box of length v / (2 B_m) along the
    track, which is what its variance actually averages over. The depth is
    the format's quarter wavelength.
    """
    wavelength = float(writing_speed_m_s) / max(float(carrier_hz), 1.0)
    depth = float(depth_wavelengths) * wavelength
    density = float(packing) / max(float(particle_volume_m3), 1e-30)
    per_wavelength = wavelength * float(track_width_m) * depth * density
    if modulation_bandwidth_hz is None or not np.isfinite(modulation_bandwidth_hz):
        read_length = wavelength
    else:
        read_length = float(writing_speed_m_s) / (2.0 * float(modulation_bandwidth_hz))
    in_envelope_band = read_length * float(track_width_m) * depth * density
    return {
        "wavelength_m": wavelength,
        "depth_m": depth,
        "read_length_m": read_length,
        "per_wavelength": per_wavelength,
        "in_envelope_band": in_envelope_band,
    }


def predicted_modulation_index(carrier_hz: float, writing_speed_m_s: float,
                               track_width_m: float,
                               modulation_bandwidth_hz: Optional[float] = None,
                               fano_factor: float = 1.0,
                               spacing_rms_m: Optional[float] = None,
                               **volume_assumptions) -> Dict[str, object]:
    """The modulation index the particulate model predicts, term by term.

    The count term is sqrt(F / N) with N from `read_volume_particles`. The
    spacing term is 2 pi sigma_d / lambda, Wallace's law differentiated, and
    is reported as a SENSITIVITY per 10 nm unless an rms spacing fluctuation
    is supplied - no roughness figure is assumed here. `fano_factor` is a
    labelled assumption defaulting to Poisson.
    """
    volume = read_volume_particles(carrier_hz, writing_speed_m_s,
                                   track_width_m, modulation_bandwidth_hz,
                                   **volume_assumptions)
    count = float(np.sqrt(float(fano_factor)
                          / max(volume["in_envelope_band"], 1.0)))
    per_10nm = float(2.0 * np.pi * 10e-9 / volume["wavelength_m"])
    spacing = (float(2.0 * np.pi * float(spacing_rms_m) / volume["wavelength_m"])
               if spacing_rms_m is not None else 0.0)
    total = float(np.hypot(count, spacing))
    additive_tape_cn_db = float(10.0 * np.log10(max(volume["per_wavelength"],
                                                     1.0)))
    return {
        "carrier_hz": float(carrier_hz),
        "read_length_m": volume["read_length_m"],
        "particles_in_envelope_band": volume["in_envelope_band"],
        "particles_per_wavelength": volume["per_wavelength"],
        "count_index": count,
        "spacing_index_per_10nm": per_10nm,
        "spacing_index": spacing,
        "modulation_index": total,
        "modulation_power": total ** 2,
        "additive_tape_only_cn_db": additive_tape_cn_db,
        "fraction_if_additive_is_tape_only": float(
            total ** 2 / (total ** 2 + 0.5 * 10 ** (-additive_tape_cn_db / 10.0))),
        "assumptions": ("particle volume and packing from tape_model; depth "
                        "lambda/4 from JVC VTG82063 7.2; Fano factor %.2f; "
                        "spacing rms %s" % (
                            float(fano_factor),
                            "not assumed" if spacing_rms_m is None
                            else "%.1f nm" % (float(spacing_rms_m) * 1e9))),
    }


def predicted_sideband_shape(offset_hz, carrier_hz: float,
                             response_hz, response,
                             head_log_response: Optional[Callable] = None,
                             roughness_corner_hz: Optional[float] = None
                             ) -> Dict[str, np.ndarray]:
    """The envelope-spectrum shapes each term should have, normalised to
    their own value at zero offset.

    additive   white at the input: the power image G_n
    count      white in wavenumber: the coherent image G_m, with the head's
               own response on the sidebands where a caller supplies it
    roughness  the count image times a first-order low-pass at the
               roughness correlation length, v / corner; stated only if a
               corner is supplied, since none is assumed here
    """
    offsets = np.abs(np.asarray(offset_hz, dtype=np.float64))
    weight = ((lambda f: np.exp(head_log_response(f)))
              if head_log_response is not None else None)
    shapes = {
        "additive": sideband_image(offsets, carrier_hz, response_hz, response, False),
        "count": sideband_image(offsets, carrier_hz, response_hz, response, True,
                                weight=weight),
    }
    if roughness_corner_hz is not None and roughness_corner_hz > 0:
        shapes["roughness"] = (shapes["count"]
                               / (1.0 + (offsets / float(roughness_corner_hz)) ** 2))
    return shapes


# --------------------------------------------------------------------------
# Cells and the straight line through them
# --------------------------------------------------------------------------


def variance_effective_count(centred_rows) -> Dict[str, float]:
    """How many independent samples a residual carries FOR A VARIANCE.

    The variance of the sample variance of n stationary Gaussian samples is
    (2 sigma^4 / n) * sum_k (1 - |k|/n) r(k)^2, so the count that governs
    it is n over the sum of SQUARED correlations. That sum is smaller than
    the sum of correlations that governs a mean, which is what the AR(1)
    count in `head_model.effective_sample_size` is built for: on the bars
    tape's tip residual (50 MSps through the decoder's 0.5-6.5 MHz
    band-pass) the AR(1) count reads 1 to 2 of 65 samples while this one
    reads several. The autocorrelation is POOLED across the rows of one
    kind and head, since one cell's few dozen samples cannot estimate it.
    Returns the count for a row of the given length and the pooled
    autocorrelation itself.
    """
    rows = np.asarray(centred_rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    length = rows.shape[1]
    if length < 2 or rows.shape[0] == 0:
        return {"effective_count": float(max(length, 1)),
                "autocorrelation": np.ones(max(length, 1))}
    spectrum = np.abs(np.fft.rfft(rows, n=2 * length, axis=1)) ** 2
    pooled = np.fft.irfft(spectrum.sum(axis=0))[:length]
    if pooled[0] <= 0:
        return {"effective_count": float(length), "autocorrelation": np.ones(length)}
    autocorrelation = pooled / pooled[0]
    lags = np.arange(length)
    squared_sum = 1.0 + 2.0 * float(np.sum((1.0 - lags[1:] / length)
                                           * autocorrelation[1:] ** 2))
    # the count for a MEAN is governed by the plain sum of correlations
    plain_sum = 1.0 + 2.0 * float(np.sum((1.0 - lags[1:] / length)
                                         * autocorrelation[1:]))
    return {"effective_count": float(np.clip(length / squared_sum, 1.0, length)),
            "mean_count": float(np.clip(length / max(plain_sum, 1e-12), 1.0, length)),
            "autocorrelation": autocorrelation}


def cell_statistics(residual, level: float, effective_count: Optional[float] = None,
                    **labels) -> Dict[str, object]:
    """One cell's contribution to the regression.

    `residual` is the cell's envelope after the coherent template is removed
    and `level` its mean envelope. The variance is of the residual about its
    own mean, so a level offset of the whole cell - which belongs to the
    slow axis, not to this one - does not count. The effective count is the
    variance count from `variance_effective_count` where the caller has
    pooled one (the instrument does); otherwise the residual's own AR(1)
    count, which is the conservative one. Both are kept.
    """
    values = np.asarray(residual, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count < 4:
        return {"usable": False, "count": count, **labels}
    centred = values - values.mean()
    variance = float(centred @ centred / (count - 1))
    ar1 = float(head_model.effective_sample_size(values))
    return {
        "usable": True,
        "level": float(level),
        "mean_square": float(level) ** 2,
        "variance": variance,
        "count": count,
        "effective_count": (float(min(max(effective_count, 1.0), count))
                            if effective_count is not None else ar1),
        "effective_count_ar1": ar1,
        **labels,
    }


def _weighted_columns(columns: np.ndarray, y: np.ndarray, weight: np.ndarray):
    """Weighted least squares of y on the given columns, solved in scaled
    coordinates so that columns spanning many orders of magnitude do not
    defeat the covariance. Returns coefficients, covariance and the
    weighted residual sum of squares, all in the ORIGINAL units."""
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    scale = np.array([max(float(np.sqrt(np.average(c ** 2, weights=weight))), 1e-300)
                      for c in columns.T])
    y_scale = max(float(np.sqrt(np.average(y ** 2, weights=weight))), 1e-300)
    design = columns / scale
    root = np.sqrt(weight)
    solution, _, rank, _ = np.linalg.lstsq(design * root[:, None],
                                           (y / y_scale) * root, rcond=None)
    # inv(X' W X) for the original columns X = design * scale: the column
    # scaling comes back out on both sides and the y scaling, which only
    # conditioned the solve, cancels. (A first version carried y_scale into
    # the covariance and overstated the fit errors by y_scale^2 - 1e9 on
    # real envelope units - which the jackknife error masked on planted data.)
    normal = (design * weight[:, None]).T @ design
    covariance = np.full((columns.shape[1],) * 2, np.inf)
    if rank == columns.shape[1] and np.all(np.isfinite(normal)):
        try:
            unscale = np.diag(1.0 / scale)
            covariance = unscale @ np.linalg.inv(normal) @ unscale
        except np.linalg.LinAlgError:
            pass
    coefficients = solution * y_scale / scale
    residual = y - columns @ coefficients
    chi_square = float(np.sum(weight * residual ** 2))
    return coefficients, covariance, chi_square


def estimate(cells: Iterable[Dict[str, object]], iterations: int = 4,
             group_key: str = "field") -> Dict[str, object]:
    """THE DISCRIMINANT: variance against squared mean across cells.

    Fits, by iteratively reweighted least squares,

        variance_i = sigma_n^2 * b_n,i + sigma_m^2 * b_m,i * mean_i^2

    where b_n and b_m are the cell's bandwidth ratios to the reference
    carrier (one where absent, so a set of tip cells fits the plain line).
    The weight of a cell is the inverse variance of its sample variance under
    the current model, 2 v_i^2 / (n_eff_i - 1), which is what a Gaussian
    residual gives; the reduced chi-square is reported and, where it exceeds
    one, inflates the covariance so that a model that does not describe the
    scatter cannot hide it. Where it is below one the effective count the
    scatter implies is reported as `effective_count_empirical`, because the
    AR(1) count is conservative on a band-limited residual.

    The errors-in-variables attenuation is corrected with the reliability
    ratio computed from the cells' own effective counts, and the correction
    factor is returned so it can be seen.

    Two error bars are returned and the LARGER is the one to quote: the
    fit's covariance, and a leave-one-group-out jackknife over `group_key`
    (fields by default), which sees whatever correlates cells within a
    group - the slow modulation the cell variance does not contain.

    The lever arm - the coefficient of variation of the modulation column
    over the additive column across cells - and the correlation between the
    slope's and the intercept's errors are returned because without a lever
    arm the two are the same number split arbitrarily. `modulation_power_
    bound` is the largest modulation power the data admit at two sigma,
    which is what to quote when the slope is consistent with zero.
    """
    usable = [c for c in cells if c.get("usable", True)]
    if len(usable) < 3:
        return {"resolved": False, "cells": len(usable),
                "why": "fewer than three usable cells"}
    x = np.array([float(c["mean_square"]) for c in usable])
    y = np.array([float(c["variance"]) for c in usable])
    b_n = np.array([float(c.get("additive_bandwidth_ratio", 1.0)) for c in usable])
    b_m = np.array([float(c.get("modulation_bandwidth_ratio", 1.0)) for c in usable])
    n_eff = np.array([max(float(c.get("effective_count", c["count"])), 2.0)
                      for c in usable])
    groups = np.array([c.get(group_key, index)
                       for index, c in enumerate(usable)], dtype=object)
    columns = np.vstack([b_n, b_m * x]).T

    def fit(cols, y_in, n_in):
        weight = np.ones(len(y_in))
        coefficients = np.zeros(2)
        covariance = np.full((2, 2), np.inf)
        chi_square = np.inf
        # a modelled variance is floored at a millionth of the typical
        # observed one so that an empty bin (a periodogram bin at an
        # offset the band-pass has removed) cannot give an infinite weight
        floor = 1e-6 * max(float(np.median(np.abs(y_in))), 1e-300)
        for _ in range(max(int(iterations), 1)):
            coefficients, covariance, chi_square = _weighted_columns(cols, y_in, weight)
            modelled = np.maximum(cols @ coefficients, floor)
            weight = (n_in - 1.0) / (2.0 * modelled ** 2)
        return coefficients, covariance, chi_square, weight

    coefficients, covariance, chi_square, weight = fit(columns, y, n_eff)
    dof = max(len(x) - 2, 1)
    reduced = chi_square / dof
    if np.isfinite(reduced) and reduced > 1.0:
        covariance = covariance * reduced

    # errors in variables: the modulation column b_m * mean^2 carries an
    # error of about 2 * b_m * mean * sqrt(variance / n_eff). The slope is
    # read against that column over the additive column, so the error is
    # taken in the same ratio units (divided by b_n), and its variance
    # across cells dilutes the slope by the reliability ratio.
    modelled = np.maximum(columns @ coefficients, 0.0)
    column_error_variance = float(np.average((b_m / b_n) ** 2 * 4.0 * x * modelled / n_eff,
                                             weights=weight))
    ratio_column = columns[:, 1] / np.maximum(columns[:, 0], 1e-300)
    spread = float(np.average((ratio_column - np.average(ratio_column, weights=weight)) ** 2,
                              weights=weight))
    reliability = (spread / (spread + column_error_variance)
                   if spread > 0 else 0.0)
    attenuation = 1.0 / reliability if reliability > 0 else np.inf
    slope = float(coefficients[1] * attenuation) if np.isfinite(attenuation) else float("nan")
    # the intercept moves with the slope: the line still passes through the
    # weighted centroid of (additive column, variance)
    centre_n = float(np.average(columns[:, 0], weights=weight))
    centre_mx = float(np.average(columns[:, 1], weights=weight))
    y_centre = float(np.average(y, weights=weight))
    intercept = (float((y_centre - slope * centre_mx) / centre_n)
                 if np.isfinite(slope) and centre_n > 0 else float("nan"))

    slope_error = float(np.sqrt(max(covariance[1, 1], 0.0)) * attenuation)
    intercept_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    error_correlation = (float(covariance[0, 1]
                               / np.sqrt(covariance[0, 0] * covariance[1, 1]))
                         if covariance[0, 0] > 0 and covariance[1, 1] > 0
                         and np.isfinite(covariance[0, 0]) else float("nan"))

    # jackknife over groups
    unique = list(dict.fromkeys(groups.tolist()))
    jack_slope, jack_intercept = [], []
    if len(unique) >= 3:
        for left_out in unique:
            keep = groups != left_out
            if np.count_nonzero(keep) < 3:
                continue
            sub, _, _, sub_weight = fit(columns[keep], y[keep], n_eff[keep])
            sub_slope = sub[1] * attenuation
            sub_centre_n = float(np.average(columns[keep, 0], weights=sub_weight))
            sub_centre_mx = float(np.average(columns[keep, 1], weights=sub_weight))
            jack_slope.append(sub_slope)
            jack_intercept.append((float(np.average(y[keep], weights=sub_weight))
                                   - sub_slope * sub_centre_mx) / max(sub_centre_n, 1e-300))
    if len(jack_slope) >= 3:
        count = len(jack_slope)
        jack_slope_error = float(np.sqrt((count - 1.0) / count
                                         * np.sum((np.array(jack_slope)
                                                   - np.mean(jack_slope)) ** 2)))
        jack_intercept_error = float(np.sqrt((count - 1.0) / count
                                             * np.sum((np.array(jack_intercept)
                                                       - np.mean(jack_intercept)) ** 2)))
    else:
        jack_slope_error = float("nan")
        jack_intercept_error = float("nan")

    quoted_slope_error = float(np.nanmax([slope_error, jack_slope_error]))
    quoted_intercept_error = float(np.nanmax([intercept_error,
                                              jack_intercept_error]))

    x_centre = float(np.average(x, weights=weight))
    lever_arm = (float(np.sqrt(spread) / np.average(ratio_column, weights=weight))
                 if spread > 0 else 0.0)
    modulation_power = slope
    modulation_index = float(np.sqrt(max(slope, 0.0))) if np.isfinite(slope) else float("nan")
    index_error = (float(quoted_slope_error / (2.0 * modulation_index))
                   if modulation_index > 0 else float(np.sqrt(max(quoted_slope_error, 0.0))))
    additive_power = intercept

    # the fraction at the working level, with a delta-method error
    at_level = modulation_power * x_centre
    total = at_level + additive_power
    fraction = float(at_level / total) if total > 0 else float("nan")
    if total > 0:
        d_slope = x_centre * additive_power / total ** 2
        d_intercept = -at_level / total ** 2
        rho = error_correlation if np.isfinite(error_correlation) else 0.0
        fraction_error = float(np.sqrt(max(
            (d_slope * quoted_slope_error) ** 2
            + (d_intercept * quoted_intercept_error) ** 2
            + 2.0 * d_slope * d_intercept * rho
            * quoted_slope_error * quoted_intercept_error, 0.0)))
    else:
        fraction_error = float("nan")

    slope_sigma = (float(slope / quoted_slope_error)
                   if quoted_slope_error > 0 else float("inf"))
    intercept_sigma = (float(intercept / quoted_intercept_error)
                       if quoted_intercept_error > 0 else float("inf"))
    n_eff_median = float(np.median(n_eff))
    return {
        "resolved": bool(np.isfinite(slope) and lever_arm > 0),
        "cells": int(len(x)),
        "groups": int(len(unique)),
        "modulation_power": modulation_power,
        "modulation_power_error": quoted_slope_error,
        "modulation_power_fit_error": slope_error,
        "modulation_power_jackknife_error": jack_slope_error,
        "modulation_power_sigma": slope_sigma,
        "modulation_power_bound": float(max(slope, 0.0) + 2.0 * quoted_slope_error),
        "modulation_index": modulation_index,
        "modulation_index_error": index_error,
        "additive_power": additive_power,
        "additive_power_error": quoted_intercept_error,
        "additive_power_fit_error": intercept_error,
        "additive_power_jackknife_error": jack_intercept_error,
        "additive_power_sigma": intercept_sigma,
        "working_mean_square": x_centre,
        "working_level": float(np.sqrt(x_centre)),
        "modulation_fraction": fraction,
        "modulation_fraction_error": fraction_error,
        "relative_additive": (float(additive_power / x_centre)
                              if x_centre > 0 else float("nan")),
        "in_phase_carrier_to_noise_db": (
            float(10.0 * np.log10(x_centre / additive_power))
            if additive_power > 0 and x_centre > 0 else float("nan")),
        "lever_arm": lever_arm,
        "error_correlation": error_correlation,
        "attenuation_correction": float(attenuation),
        "reduced_chi_square": float(reduced),
        "effective_count_median": n_eff_median,
        "effective_count_empirical": (float(1.0 + (n_eff_median - 1.0) / reduced)
                                      if np.isfinite(reduced) and reduced > 0 else float("nan")),
        "why": ("variance = additive * b_n + modulation * b_m * mean^2 across "
                "cells; the slope is signal-proportional power and the "
                "intercept is level-independent power, both at the reference "
                "carrier; quote the larger of the two error bars and the "
                "bound where the slope is within it"),
    }


def slow_modulation(cells: Sequence[Dict[str, object]],
                    minimum_fields: int = 3) -> Dict[str, object]:
    """THE SLOW BAND: the scatter of the cells' LEVELS, which the cell
    split leaves out of the regression.

    A cell's variance holds the modulation above about the reciprocal of
    the interior's length; whatever modulates the envelope more slowly
    than that moves a cell's mean level instead and shows up as scatter
    between cells. MEASURED on the bars tape (SP) that scatter is 4.0 to
    4.8 per cent rms per head at the sync tip after the field-locked line
    curve (the head's contact pattern along the track) is removed, white
    from line to line, and only 25 to 41 per cent shared between the tip
    and the back porch of the same line, 5 us apart - so it lives between
    about 0.2 and 0.8 MHz of the carrier, where a spacing fluctuation
    from surface roughness with a correlation length of 7 to 30 um would
    put it, and it is 16 dB per megahertz above the fast floor.

    The level's own sampling error - the fast residual leaking into a
    mean over `mean_count` independent samples - is subtracted. The
    field-locked curve is the median over fields per line index where at
    least `minimum_fields` fields hold that line; where fewer do, the
    per-field mean level stands in. The remainder is reported as a
    relative power and index, with the lag-one autocorrelation along the
    line sequence (near zero: faster than a line; near one: a slow
    wander) and, where both kinds are present, the correlation between
    the tip's and the porch's deviations on the same line.
    """
    usable = [c for c in cells if c.get("usable", True)]
    if len(usable) < 8:
        return {"resolved": False, "why": "fewer than eight usable cells"}
    by_kind: Dict[str, Dict[tuple, float]] = {}
    deviations_by_kind: Dict[str, np.ndarray] = {}
    leakage_by_kind: Dict[str, float] = {}
    for kind in sorted(set(str(c.get("kind", "")) for c in usable)):
        members = [c for c in usable if str(c.get("kind", "")) == kind]
        fields = np.array([int(c.get("field", 0)) for c in members])
        lines = np.array([int(c.get("line", 0)) for c in members])
        levels = np.array([float(c["level"]) for c in members])
        curve = np.full(levels.size, np.nan)
        for line in np.unique(lines):
            at_line = lines == line
            if np.unique(fields[at_line]).size >= minimum_fields:
                curve[at_line] = np.median(levels[at_line])
        for field in np.unique(fields):
            at_field = fields == field
            missing = at_field & ~np.isfinite(curve)
            if missing.any():
                curve[missing] = np.mean(levels[at_field])
        relative = levels / np.maximum(curve, 1e-300) - 1.0
        for field in np.unique(fields):
            at_field = fields == field
            relative[at_field] -= relative[at_field].mean()
        leakage = float(np.mean([
            float(c.get("variance_raw", c["variance"]))
            / (max(float(c.get("mean_count", c.get("effective_count", c["count"]))), 1.0)
               * max(float(c["level"]) ** 2, 1e-300))
            for c in members]))
        deviations_by_kind[kind] = relative
        leakage_by_kind[kind] = leakage
        by_kind[kind] = {(int(f), int(l)): float(r)
                         for f, l, r in zip(fields, lines, relative)}
    # the primary figure is the tip's where present
    kind = "tip" if "tip" in deviations_by_kind else sorted(deviations_by_kind)[0]
    relative = deviations_by_kind[kind]
    raw_power = float(np.mean(relative ** 2))
    power = max(raw_power - leakage_by_kind[kind], 0.0)
    # lag-one autocorrelation along the line sequence within fields
    members = [c for c in usable if str(c.get("kind", "")) == kind]
    keyed = by_kind[kind]
    pairs = [(keyed[(f, l)], keyed[(f, l + 1)]) for (f, l) in keyed if (f, l + 1) in keyed]
    if len(pairs) > 10:
        pair_array = np.array(pairs)
        lag_one = float(np.corrcoef(pair_array[:, 0], pair_array[:, 1])[0, 1])
    else:
        lag_one = float("nan")
    shared = float("nan")
    if "tip" in by_kind and "porch" in by_kind:
        common = [k for k in by_kind["tip"] if k in by_kind["porch"]]
        if len(common) > 10:
            a = np.array([by_kind["tip"][k] for k in common])
            b = np.array([by_kind["porch"][k] for k in common])
            shared = float(np.corrcoef(a, b)[0, 1])
    # the error bar of a mean square over n roughly independent cells
    error = float(np.sqrt(2.0 / max(relative.size, 2)) * raw_power)
    return {
        "resolved": True,
        "kind": kind,
        "cells": int(relative.size),
        "slow_power": power,
        "slow_power_error": error,
        "slow_index": float(np.sqrt(power)),
        "raw_scatter_power": raw_power,
        "fast_leakage_power": leakage_by_kind[kind],
        "lag_one_line_correlation": lag_one,
        "tip_porch_correlation": shared,
        "by_kind_power": {k: float(max(np.mean(v ** 2) - leakage_by_kind[k], 0.0))
                          for k, v in deviations_by_kind.items()},
        "why": ("the scatter of cell levels about the field-locked line "
                "curve, less the fast residual's leakage into the cell "
                "mean: modulation slower than the cell, which the "
                "regression cannot see and a single pass cannot split "
                "from playback flutter"),
    }


def event_gate(cells: Sequence[Dict[str, object]], fitted: Dict[str, object],
               confidence: float = 0.999) -> List[bool]:
    """Which cells the noise model does not describe - dropouts, a head
    switch, a partial collapse - so that they are EXCLUDED rather than fitted,
    as `interference_distributions` prescribes for events.

    A sample variance of n_eff Gaussian residuals is chi-square with
    n_eff - 1 degrees of freedom over that many; a cell above the model's
    prediction by more than that distribution's `confidence` quantile is an
    event, and so is one whose level is below the model's own additive
    noise (a carrier that has gone). The quantile is the only number here
    and it is a probability, stated.
    """
    from scipy import stats

    keep = []
    for cell in cells:
        if not cell.get("usable", True):
            keep.append(False)
            continue
        modelled = (fitted["additive_power"] * float(cell.get("additive_bandwidth_ratio", 1.0))
                    + fitted["modulation_power"] * float(cell.get("modulation_bandwidth_ratio", 1.0))
                    * float(cell["mean_square"]))
        dof = max(float(cell.get("effective_count", cell["count"])) - 1.0, 1.0)
        bound = stats.chi2.ppf(confidence, dof) / dof
        collapsed = float(cell["mean_square"]) < max(fitted["additive_power"], 0.0)
        keep.append(bool(modelled > 0 and not collapsed
                         and float(cell["variance"]) <= bound * modelled))
    return keep


def estimate_with_gate(cells: Sequence[Dict[str, object]], passes: int = 2,
                       **kwargs) -> Dict[str, object]:
    """Fit, gate the events out, fit again. Two passes suffice because the
    gate's threshold moves only as far as the events had pulled the line."""
    cells = list(cells)
    fitted = estimate(cells, **kwargs)
    kept = [c.get("usable", True) for c in cells]
    for _ in range(max(int(passes), 1)):
        if not fitted.get("resolved"):
            break
        kept = event_gate(cells, fitted)
        fitted = estimate([c for c, k in zip(cells, kept) if k], **kwargs)
    fitted["excluded_as_events"] = int(sum(1 for c, k in zip(cells, kept)
                                           if c.get("usable", True) and not k))
    return fitted


# --------------------------------------------------------------------------
# What the channel fit should do with it
# --------------------------------------------------------------------------


def weight_for_response(fitted: Dict[str, object], envelope_variance,
                        envelope_mean_square, effective_count,
                        modulation_bandwidth_ratio=1.0,
                        additive_bandwidth_ratio=1.0,
                        slow_power: float = 0.0,
                        line_count=None) -> Dict[str, np.ndarray]:
    """The weight an envelope-derived |H| bin should carry, from the model
    rather than from the raw variance.

    Per bin, in relative (neper^2) units:

        additive_part    sigma_n^2 b_n / (A^2 n_eff) - falls with level; the
                         part the fit down-weights weak bins against
        modulation_part  sigma_m^2 b_m / n_eff + sigma_slow^2 / n_lines -
                         the same at every level; no information about
                         |H|, and a floor on precision. The fast term
                         averages down with the bin's independent samples
                         (`effective_count`, a MEAN's count), the slow
                         term (`slow_power`, from `slow_modulation`) only
                         with the independent LINES the bin draws on
                         (`line_count`; the sample count where absent)
        channel_excess   observed / A^2 - the two above - what the channel
                         itself is doing inside the bin: to MODEL, never to
                         weight against
        weight           1 / (additive_part + modulation_part), normalised
                         to the best bin - an inverse VARIANCE, for a
                         weighted mean or a chi-square
        residual_weight  its square root - the form `head_model.fit` wants,
                         since it multiplies each RESIDUAL by its weight

    `envelope_variance` is the raw per-bin envelope variance the fit
    currently sees, `envelope_mean_square` the bin's squared mean envelope,
    `effective_count` the independent samples behind the bin, and the two
    ratios carry the bin's carrier from the reference the estimate was made
    at (`bandwidth_ratios`).
    """
    observed = np.asarray(envelope_variance, dtype=np.float64)
    level_square = np.asarray(envelope_mean_square, dtype=np.float64)
    count = np.maximum(np.asarray(effective_count, dtype=np.float64), 1.0)
    b_m = np.broadcast_to(np.asarray(modulation_bandwidth_ratio, dtype=np.float64),
                          level_square.shape)
    b_n = np.broadcast_to(np.asarray(additive_bandwidth_ratio, dtype=np.float64),
                          level_square.shape)
    modulation = max(float(fitted["modulation_power"]), 0.0)
    additive = max(float(fitted["additive_power"]), 0.0)
    slow = max(float(slow_power), 0.0)
    lines = (np.maximum(np.asarray(line_count, dtype=np.float64), 1.0)
             if line_count is not None else count)
    lines = np.broadcast_to(lines, level_square.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        additive_part = np.where(level_square > 0,
                                 additive * b_n / (level_square * count), np.inf)
        modulation_part = modulation * b_m / count + slow / lines
        relative_observed = np.where(level_square > 0,
                                     observed / level_square, np.inf)
        modelled_relative = np.where(level_square > 0,
                                     additive * b_n / np.maximum(level_square, 1e-300)
                                     + modulation * b_m + slow, np.inf)
    channel_excess = np.maximum(relative_observed - modelled_relative, 0.0)
    modelled_variance = additive_part + modulation_part
    with np.errstate(divide="ignore"):
        weight = np.where(modelled_variance > 0, 1.0 / modelled_variance, 0.0)
    finite = np.isfinite(weight)
    best = float(np.max(weight[finite])) if np.any(finite) else 0.0
    normalised = weight / best if best > 0 else weight
    return {
        "weight": normalised,
        "residual_weight": np.sqrt(np.maximum(normalised, 0.0)),
        "additive_part": additive_part,
        "modulation_part": modulation_part,
        "modelled_relative_variance": modelled_relative,
        "observed_relative_variance": relative_observed,
        "channel_excess": channel_excess,
        "additive_dominates": additive_part > modulation_part,
        "why": ("inverse of the modelled variance of the bin's log mean; the "
                "raw envelope variance beyond the model is the channel "
                "varying and is returned as channel_excess to be modelled"),
    }


# --------------------------------------------------------------------------
# The frozen test
# --------------------------------------------------------------------------


def frozen_test(pass_a: Sequence[Dict[str, object]],
                pass_b: Sequence[Dict[str, object]],
                group_key: str = "field") -> Dict[str, object]:
    """Two aligned replays of one tape fill the two-by-two table.

    Each pass is a sequence of cells, aligned so that cell i of one pass is
    the same tape position as cell i of the other, each carrying the
    template-subtracted residual samples as `residual` and the mean level as
    `level`. Per cell, with the residuals centred:

        reproducible_i     = mean(r_a r_b)            - what the tape holds
        non_reproducible_i = mean(r_a^2 + r_b^2) / 2 - reproducible_i

    Regressing each on the product of the two passes' levels gives four
    numbers:

        reproducible slope          modulation noise, frozen in the coating
        reproducible intercept      the bulk particle pattern, also on tape
        non-reproducible slope      gain flutter at playback (contact, AGC)
        non-reproducible intercept  electronics and converter

    The share of the multiplicative term that reproduces is the quantity
    Ethan's directive turns on: it is what licenses treating the slope from
    a single pass as the coating's, and it is what no single pass can give.
    """
    pairs = [(a, b) for a, b in zip(pass_a, pass_b)
             if a.get("usable", True) and b.get("usable", True)]
    if len(pairs) < 3:
        return {"resolved": False, "why": "fewer than three aligned cells"}
    reproducible_cells, non_reproducible_cells = [], []
    cross_sum, total_sum = 0.0, 0.0
    for a, b in pairs:
        ra = np.asarray(a["residual"], dtype=np.float64)
        rb = np.asarray(b["residual"], dtype=np.float64)
        length = min(ra.size, rb.size)
        if length < 4:
            continue
        ra = ra[:length] - ra[:length].mean()
        rb = rb[:length] - rb[:length].mean()
        level_square = float(a["level"]) * float(b["level"])
        cross = float(ra @ rb / (length - 1))
        total = 0.5 * float(ra @ ra + rb @ rb) / (length - 1)
        labels = {group_key: a.get(group_key, 0)}
        n_eff = float(head_model.effective_sample_size(ra))
        reproducible_cells.append({
            "mean_square": level_square, "variance": cross,
            "count": length, "effective_count": n_eff, **labels})
        non_reproducible_cells.append({
            "mean_square": level_square, "variance": total - cross,
            "count": length, "effective_count": n_eff, **labels})
        cross_sum += cross
        total_sum += total
    frozen = estimate(reproducible_cells, group_key=group_key)
    volatile = estimate(non_reproducible_cells, group_key=group_key)
    if not (frozen.get("resolved") and volatile.get("resolved")):
        return {"resolved": False, "cells": len(reproducible_cells),
                "frozen": frozen, "volatile": volatile,
                "why": "one of the two regressions did not resolve"}
    total_multiplicative = (max(frozen["modulation_power"], 0.0)
                            + max(volatile["modulation_power"], 0.0))
    return {
        "resolved": True,
        "cells": len(reproducible_cells),
        # a ratio of summed powers, never a mean of per-cell ratios
        # (RINGING_RULES 31 and 33)
        "reproducible_share": (float(cross_sum / total_sum) if total_sum > 0
                               else float("nan")),
        "coating_modulation_power": frozen["modulation_power"],
        "coating_modulation_power_error": frozen["modulation_power_error"],
        "tape_particle_power": frozen["additive_power"],
        "tape_particle_power_error": frozen["additive_power_error"],
        "playback_flutter_power": volatile["modulation_power"],
        "playback_flutter_power_error": volatile["modulation_power_error"],
        "electronics_power": volatile["additive_power"],
        "electronics_power_error": volatile["additive_power_error"],
        "frozen_share_of_multiplicative": (
            float(max(frozen["modulation_power"], 0.0) / total_multiplicative)
            if total_multiplicative > 0 else float("nan")),
        "frozen": frozen,
        "volatile": volatile,
        "why": ("the reproducible cross-covariance regressed on level^2 "
                "splits the tape's own terms; the non-reproducible remainder "
                "regressed the same way splits the playback's"),
    }


# --------------------------------------------------------------------------
# The spectrum, by the same line in every bin
# --------------------------------------------------------------------------


def sideband_spectra(periodograms, mean_squares, effective_averages,
                     group_key_values=None) -> Dict[str, np.ndarray]:
    """The modulation and additive SPECTRA, separated bin by bin.

    `periodograms` is (cells, bins), each row a cell's residual power
    spectrum in consistent units; `mean_squares` the cells' squared levels;
    `effective_averages` the number of independent periodogram averages
    behind each row (its chi-square degrees of freedom over two). The same
    straight line as `estimate` is fitted in every bin, so the slope
    spectrum is the modulation sideband shape and the intercept spectrum the
    additive image of the band-pass, each with an error per bin.
    """
    power = np.asarray(periodograms, dtype=np.float64)
    level_square = np.asarray(mean_squares, dtype=np.float64)
    averages = np.maximum(np.asarray(effective_averages, dtype=np.float64), 1.0)
    if power.ndim != 2 or power.shape[0] != level_square.size:
        raise ValueError("periodograms must be (cells, bins) matching mean_squares")
    bins = power.shape[1]
    slope = np.full(bins, np.nan)
    intercept = np.full(bins, np.nan)
    slope_error = np.full(bins, np.nan)
    intercept_error = np.full(bins, np.nan)
    for index in range(bins):
        cells = [{"mean_square": float(level_square[i]),
                  "variance": float(power[i, index]),
                  "count": int(round(averages[i])) + 1,
                  "effective_count": float(averages[i]) + 1.0,
                  "field": (group_key_values[i] if group_key_values is not None
                            else i)}
                 for i in range(level_square.size)]
        fitted = estimate(cells, iterations=3)
        if fitted.get("resolved"):
            slope[index] = fitted["modulation_power"]
            intercept[index] = fitted["additive_power"]
            slope_error[index] = fitted["modulation_power_error"]
            intercept_error[index] = fitted["additive_power_error"]
    return {
        "modulation_spectrum": slope,
        "modulation_spectrum_error": slope_error,
        "additive_spectrum": intercept,
        "additive_spectrum_error": intercept_error,
        "why": ("the variance-against-level line fitted in every bin; the "
                "slope spectrum is the sideband shape the coating imposes, "
                "the intercept spectrum the band-pass image the additive "
                "noise takes"),
    }


def shape_agreement(measured, predicted, error=None) -> Dict[str, float]:
    """How well a measured spectrum shape matches a predicted one, after the
    one free scale each is allowed: the correlation of the two and the
    chi-square of the scaled prediction against the measurement."""
    got = np.asarray(measured, dtype=np.float64)
    want = np.asarray(predicted, dtype=np.float64)
    good = np.isfinite(got) & np.isfinite(want)
    sigma = None
    if error is not None:
        sigma = np.asarray(error, dtype=np.float64)
        good &= np.isfinite(sigma) & (sigma > 0)
    if good.sum() < 3:
        return {"correlation": float("nan"), "reduced_chi_square": float("nan"),
                "scale": float("nan"), "bins": int(good.sum())}
    got, want = got[good], want[good]
    weight = (1.0 / sigma[good] ** 2) if sigma is not None else np.ones_like(got)
    scale = float(np.sum(weight * got * want) / max(np.sum(weight * want ** 2), 1e-300))
    residual = got - scale * want
    chi = float(np.sum(weight * residual ** 2) / max(good.sum() - 1, 1))
    correlation = (float(np.corrcoef(got, want)[0, 1])
                   if got.std() > 0 and want.std() > 0 else float("nan"))
    return {"correlation": correlation, "scale": scale,
            "reduced_chi_square": chi, "bins": int(good.sum())}


# --------------------------------------------------------------------------
# The instrument: cells from raw RF, sync and blanking only
# --------------------------------------------------------------------------


def decoder_response_grid(rf):
    """The decoder's own RF band-pass as a magnitude on a positive grid."""
    magnitude = np.abs(np.asarray(rf.Filters["RFVideo"], dtype=np.float64))
    grid = np.fft.fftfreq(magnitude.size, d=1.0 / float(rf.freq_hz))
    order = np.argsort(grid)
    positive = grid[order] >= 0
    return grid[order][positive], magnitude[order][positive]


def decoder_channels(samples, rf) -> Dict[str, object]:
    """The decoder's OWN front end run over a window, block by block with
    its own overlap, and nothing re-implemented.

    Three channels come back on one sample grid, output sample k being
    input sample k + `offset`:

        envelope       |analytic signal| of the block through
                       `Filters["RFVideo"]` - the same array `demodblock`
                       hands to dropout detection and the colour-under
                       amplitude correction, i.e. the envelope the channel
                       fit reads |H| from
        frequency_05   the raw demodulated frequency through the decoder's
                       0.5 MHz low-pass (`demod_05`), which is what the
                       decoder itself finds its sync pulses on
        frequency      the demodulated video, in hertz

    Using `demodblock` rather than a copy of it is what guarantees the
    envelope measured here is the one the response is derived from.
    """
    data = np.asarray(samples, dtype=np.float64)
    blocklen = int(rf.blocklen)
    cut_start = int(rf.blockcut)
    cut_end = int(rf.blockcut_end)
    stride = blocklen - cut_start - cut_end
    envelope, frequency_05, frequency = [], [], []
    for start in range(0, data.size - blocklen + 1, stride):
        video = rf.demodblock(data=data[start:start + blocklen], cut=True)["video"]
        envelope.append(np.asarray(video["envelope"], dtype=np.float64))
        frequency_05.append(np.asarray(video["demod_05"], dtype=np.float64))
        frequency.append(np.asarray(video["demod"], dtype=np.float64))
    if not envelope:
        empty = np.array([], dtype=np.float64)
        return {"envelope": empty, "frequency_05": empty, "frequency": empty,
                "offset": cut_start}
    return {"envelope": np.concatenate(envelope),
            "frequency_05": np.concatenate(frequency_05),
            "frequency": np.concatenate(frequency),
            "offset": cut_start}


def _runs(mask: np.ndarray):
    """Start and end (exclusive) of every run of True in `mask`."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def sync_pulses(frequency_05, rf) -> Dict[str, object]:
    """Every pulse at sync-tip level in the decoder's own sync channel,
    classified by the format's specified widths.

    The threshold is midway between the sync tip and blanking in hertz,
    both from `SysParams` (ire0, vsync_ire, hz_ire) - the decoder's own
    choice for its pulse search. No smoothing is added: `demod_05` is
    already the decoder's 0.5 MHz low-pass, whose zero-phase step response
    leaves each threshold crossing at the edge's own midpoint, so widths
    are preserved. A run narrower than half an equalizing pulse is a noise
    crossing and is dropped, counted in `noise_runs`; the rest are classed
    against the midpoints between the specified equalizing, horizontal and
    vertical pulse widths.
    """
    sp = rf.SysParams
    rate = float(rf.freq_hz)
    tip_hz = float(sp["ire0"] + sp["vsync_ire"] * sp["hz_ire"])
    blanking_hz = float(sp["ire0"])
    threshold = 0.5 * (tip_hz + blanking_hz)
    starts, ends = _runs(np.asarray(frequency_05, dtype=np.float64) < threshold)
    widths_us = (ends - starts) / rate * 1e6
    genuine = widths_us >= 0.5 * float(sp["eqPulseUS"])
    noise_runs = int(np.count_nonzero(~genuine))
    starts, ends, widths_us = starts[genuine], ends[genuine], widths_us[genuine]
    eq_edge = 0.5 * (sp["eqPulseUS"] + sp["hsyncPulseUS"])
    v_edge = 0.5 * (sp["hsyncPulseUS"] + sp["vsyncPulseUS"])
    kinds = np.where(widths_us < eq_edge, "equalizing",
                     np.where(widths_us < v_edge, "horizontal", "broad"))
    return {"start": starts, "end": ends, "width_us": widths_us, "kind": kinds,
            "threshold_hz": threshold, "tip_hz": tip_hz,
            "blanking_hz": blanking_hz, "noise_runs": noise_runs}


def _half_lines(spacing_samples: float, line_samples: float) -> float:
    """A pulse spacing rounded to the nearest half line, as the decoder's
    own field-order test rounds it."""
    return round(2.0 * spacing_samples / line_samples) / 2.0


def field_boundaries(pulses, rf) -> Dict[str, np.ndarray]:
    """Where each field begins - its first broad (vertical sync) pulse - and
    which head read it.

    THE HEAD IS THE FIELD PARITY (rule 38: `is_first_field` IS the head
    identity and stays consistent for a whole decode), so the parity is
    read by the decoder's own rule (`vhsdecode/sync.pyx`, the 525-line
    table): a FIRST field's last horizontal sync stands a full line before
    its first equalizing pulse and its last equalizing pulse half a line
    before the first horizontal sync after the interval; a second field
    has the two the other way round. Both spacings vote; an interval whose
    votes disagree, or that lacks the pulses to vote, takes the parity the
    alternation from the nearest confident interval implies and is counted
    in `inferred`. `head` is 0 for a first field and 1 for a second, at
    every tape position alike, so cells pooled by head across positions
    are pooled by the same physical head.
    """
    rate = float(rf.freq_hz)
    line = float(rf.SysParams["line_period"]) * 1e-6 * rate
    kinds = pulses["kind"]
    starts = pulses["start"]
    broad_index = np.flatnonzero(kinds == "broad")
    if broad_index.size == 0:
        return {"start": np.array([], dtype=np.int64), "head": np.array([], dtype=np.int64),
                "confident": np.array([], dtype=bool), "inferred": 0}
    # the broad pulses of one vertical interval lie within a few lines of
    # each other; the next interval is a field away, so a gap longer than
    # the serrated interval itself (nine lines) separates intervals
    gaps = np.diff(starts[broad_index])
    breaks = np.flatnonzero(gaps > line * 9)
    first_of_interval = np.concatenate([[0], breaks + 1])
    last_of_interval = np.concatenate([breaks, [broad_index.size - 1]])
    field_starts, votes = [], []
    for first, last in zip(first_of_interval, last_of_interval):
        head_pulse = int(broad_index[first])
        tail_pulse = int(broad_index[last])
        field_starts.append(int(starts[head_pulse]))
        vote = 0.0
        counted = 0
        # backwards through the leading equalizing pulses to the last
        # horizontal sync before the interval
        i = head_pulse - 1
        while i >= 0 and kinds[i] == "equalizing":
            i -= 1
        if i >= 0 and kinds[i] == "horizontal" and i + 1 < head_pulse:
            spacing = _half_lines(starts[i + 1] - starts[i], line)
            if spacing == 1.0:
                vote += 1.0
                counted += 1
            elif spacing == 0.5:
                vote -= 1.0
                counted += 1
        # forwards through the trailing equalizing pulses to the first
        # horizontal sync after the interval
        j = tail_pulse + 1
        while j < starts.size and kinds[j] == "equalizing":
            j += 1
        if j < starts.size and kinds[j] == "horizontal" and j - 1 > tail_pulse:
            spacing = _half_lines(starts[j] - starts[j - 1], line)
            if spacing == 0.5:
                vote += 1.0
                counted += 1
            elif spacing == 1.0:
                vote -= 1.0
                counted += 1
        votes.append(vote if counted else 0.0)
    field_starts = np.array(field_starts, dtype=np.int64)
    votes = np.array(votes)
    confident = votes != 0
    heads = np.full(field_starts.size, -1, dtype=np.int64)
    heads[confident] = np.where(votes[confident] > 0, 0, 1)
    inferred = 0
    if confident.any():
        anchor = int(np.flatnonzero(confident)[0])
        for index in range(field_starts.size):
            if not confident[index]:
                heads[index] = (heads[anchor] + (index - anchor)) % 2
                inferred += 1
            else:
                anchor = index
    else:
        heads = np.arange(field_starts.size) % 2
        inferred = int(field_starts.size)
    return {"start": field_starts, "head": heads, "confident": confident,
            "inferred": inferred}


# The head switch precedes vertical sync by 6.014 H (JVC VTG82063, section
# 4), and the head amplifiers' level step after it was measured to run
# through about ten lines (head_switch lane, 2026-09-01). The vertical
# interval lane found the first sixteen or so lines after vertical sync
# unusable for level work. The exclusion therefore runs from two lines
# before the switch to the later of the amplifier settle and those lines.
# Cited figures, not tuning.
SWITCH_LEAD_LINES = 6.014
SWITCH_SETTLE_LINES = 10.0
VERTICAL_INTERVAL_LEVEL_LINES = 16.0


def head_of_position(position: int, boundaries, rf):
    """The head reading a sample position, and whether that position lies
    inside the excluded switch / vertical-interval window."""
    rate = float(rf.freq_hz)
    line = float(rf.SysParams["line_period"]) * 1e-6 * rate
    starts = boundaries["start"]
    if starts.size == 0:
        return -1, True
    index = int(np.searchsorted(starts, position, side="right")) - 1
    before = starts[index] if index >= 0 else None
    after = starts[index + 1] if index + 1 < starts.size else None
    head = (int(boundaries["head"][index]) if index >= 0
            else int((boundaries["head"][0] + 1) % 2))
    excluded = False
    if before is not None:
        excluded |= (position - before) < line * max(VERTICAL_INTERVAL_LEVEL_LINES,
                                                     SWITCH_SETTLE_LINES - SWITCH_LEAD_LINES)
    if after is not None:
        excluded |= (after - position) < line * (SWITCH_LEAD_LINES + 2.0)
    # a partial field at either end of the window keeps its cells only if
    # its neighbouring boundary is known on at least one side
    excluded |= before is None and after is None
    return head, bool(excluded)


def coherent_template(stack: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """The line-locked mean shape of a cell kind, per unit level.

    Each row is scaled to unit level before the median so that cells at
    different levels contribute the same SHAPE, and the template is
    subtracted scaled back to each cell's own level. This is the coherent
    line-locked mean the amplitude lane subtracts before reading its noise
    floor (`luma_noise_response_2026-09-01.txt`), taken per unit level.
    """
    scaled = stack / np.maximum(levels[:, None], 1e-300)
    return np.median(scaled, axis=0)


# Integer alignment on the sync edge leaves up to half a sample of edge
# jitter. Where the template has slope s per sample (per unit level), that
# jitter leaks (s * level * 0.5)^2 of variance into the residual, in
# proportion to level^2 - exactly the signature of modulation noise. The
# interior is where that leakage is below one per cent of the residual
# variance, i.e. the leakage standard deviation below a tenth of the
# residual's.
JITTER_SAMPLES = 0.5
LEAKAGE_SHARE = 0.01


def position_split(residual: np.ndarray, levels: np.ndarray) -> Dict[str, object]:
    """The residual variance against position in the window, SPLIT into
    its level-independent and level-proportional parts by regressing each
    position's squared residual on the cells' squared levels.

    MEASURED on the bars tape (SP, 50 MSps, about 2,100 lines per head, and
    the same on both heads) the variance profile is a bowl: 3 to 7 times
    the central value within the first microsecond after the sync fall
    and the last half microsecond before the rise, decaying over about two
    microseconds to a broad minimum, with the template flat to 0.5 per
    cent over most of that. Three things were established about it before
    it was left alone. It is not the decoder's: an ideal constant-envelope
    FM sync train through `demodblock` gives a variance profile of exactly
    zero, and with white noise added a flat one at the level the real
    interior sits at. It is not coherent with the carrier: the phase of
    the carrier at the edge and the sub-sample edge offset together
    explain one per cent of it, so it is not a beat of the channel's
    ringing against the carrier. And it is LEVEL-INDEPENDENT: this split
    puts it in the intercept (50e-4 of the level squared at the edges
    against 3 to 6e-4 in the middle) with a slope that is NEGATIVE at the
    edges (-30e-4 +- 7e-4: lines at lower level ring more) and 0 to 3e-4
    +- 2e-4 across the interior; it is a single coherent low-frequency
    mode (62 to 74 per cent of the edge variance in one eigenvector, its
    spectrum falling to 0.4 by 0.8 MHz of offset) whose amplitude varies
    line to line, i.e. the channel's response to the carrier step, with a
    line-to-line amplitude that no measured quantity here predicts.

    So the edge transient does NOT masquerade as modulation noise: where a
    cell includes it, it inflates the additive term and pushes the slope
    negative - which is what the back porch shows, its variance sitting
    38 per cent above the white-additive prediction at its quietest
    position and the tip-plus-porch slope going to -5 sigma. The interior
    rule excludes it by the variance profile alone; this split is reported
    so that the slope's flatness across the interior can be seen.
    """
    values = np.asarray(residual, dtype=np.float64)
    level_square = np.asarray(levels, dtype=np.float64) ** 2
    squared = values ** 2
    centred = level_square - level_square.mean()
    spread = float(np.sum(centred ** 2))
    if values.shape[0] < 3 or spread <= 0:
        return {"resolved": False, "why": "no level spread across cells"}
    slope = (centred[:, None] * (squared - squared.mean(axis=0))).sum(axis=0) / spread
    intercept = squared.mean(axis=0) - slope * level_square.mean()
    fitted = intercept[None, :] + slope[None, :] * level_square[:, None]
    slope_error = np.sqrt(np.var(squared - fitted, axis=0) / spread)
    reference = float(np.median(level_square))
    return {
        "resolved": True,
        "slope": slope,
        "slope_error": slope_error,
        "intercept_relative": intercept / max(reference, 1e-300),
        "profile_relative": squared.mean(axis=0) / max(reference, 1e-300),
        "why": ("per position, squared residual = intercept + slope * level^2 "
                "across cells: the intercept is the level-independent part "
                "and the slope the level-proportional part"),
    }


def interior_of_template(template: np.ndarray, residual_variance: np.ndarray,
                         level: float, cells: int) -> slice:
    """Where the template is flat enough: the widest run through the middle
    over which edge jitter leaking through the template's slope stays below
    `LEAKAGE_SHARE` of the residual variance, and the residual variance does
    not exceed its central median by more than the chance bound for `cells`
    averages. The edge and its settle - the transient `position_split`
    describes - are excluded by this rule rather than by a fixed guard, so
    a longer settle on one tape simply costs it samples. Measured on the
    bars tape it keeps 65 to 70 of the tip's 235 samples and the same of
    the porch's 237.
    """
    from scipy import stats

    length = template.size
    centre = slice(length // 4, 3 * length // 4)
    variance_plateau = float(np.median(residual_variance[centre]))
    slope = np.abs(np.gradient(template)) * float(level)
    leakage_std = slope * JITTER_SAMPLES
    bound = stats.chi2.ppf(0.999, max(cells - 1, 1)) / max(cells - 1, 1)
    flat = (leakage_std <= np.sqrt(LEAKAGE_SHARE * np.maximum(residual_variance, 0.0))) \
        & (residual_variance <= bound * variance_plateau)
    middle = length // 2
    if not flat[middle]:
        good = np.flatnonzero(flat)
        if good.size == 0:
            return slice(middle, middle)
        middle = int(good[np.argmin(np.abs(good - middle))])
    left = middle
    while left > 0 and flat[left - 1]:
        left -= 1
    right = middle
    while right + 1 < length and flat[right + 1]:
        right += 1
    return slice(left, right + 1)


def cells_from_rf(samples, rf, kinds: Sequence[str] = ("tip", "porch"),
                  first_field_index: int = 0, keep_residuals: bool = False,
                  periodogram: bool = False, return_signals: bool = False
                  ) -> Dict[str, object]:
    """THE INSTRUMENT. Raw RF in, cells out - sync tip and back porch only.

    Steps, each from the decoder's own machinery or parameters:
      1. envelope, sync channel and video through `demodblock` itself
         (`decoder_channels`);
      2. sync pulses by the format's levels and widths on the decoder's own
         sync channel; fields and their parity (the head) by the decoder's
         own field-order rule; the switch and vertical interval excluded;
      3. per head and kind, the envelope stacked on the sync edge, its
         coherent template found and removed at each cell's own level;
      4. the interior chosen where jitter leakage is below the noise;
      5. per cell: level, variance, effective count, head, field, line,
         position (in INPUT samples), carrier, and the two bandwidth ratios
         to the sync tip; the level and residual are divided by the
         decoder's own |RFVideo| at the cell's carrier so cells at
         different carriers compare on the channel alone.

    The back porch runs from the sync's trailing edge to the start of active
    video (`activeVideoUS[0]`) less the sync width. The colour-under carrier
    sits at 629 kHz on this format and the decoder's band-pass removes it
    entirely (measured on the built NTSC filter: -55 dB at 1 MHz, below
    -200 dB at 0.5 MHz), so the porch carries no burst here. The front porch
    is not used: it is not a level (porch anchor lane, 41-66 sigma of
    content dependence).
    """
    sp = rf.SysParams
    rate = float(rf.freq_hz)
    channels = decoder_channels(samples, rf)
    envelope = channels["envelope"]
    frequency_05 = channels["frequency_05"]
    offset = int(channels["offset"])
    pulses = sync_pulses(frequency_05, rf)
    boundaries = field_boundaries(pulses, rf)
    grid, magnitude = decoder_response_grid(rf)
    line_samples = float(sp["line_period"]) * 1e-6 * rate
    tip_width = int(round(sp["hsyncPulseUS"] * 1e-6 * rate))
    porch_width = int(round((sp["activeVideoUS"][0] - sp["hsyncPulseUS"]) * 1e-6 * rate))
    # a horizontal pulse whose width is off the specified one by more than
    # the specified transition time at each edge, plus the sync channel's
    # own resolution at each edge (its low-pass corner is the decoder's
    # 0.5 MHz, one microsecond of rise), is not a clean pulse
    width_tolerance = 2.0 * sp["syncTransitionUS"] + 2.0 * 1.0
    horizontal = np.flatnonzero(
        (pulses["kind"] == "horizontal")
        & (np.abs(pulses["width_us"] - sp["hsyncPulseUS"]) < width_tolerance))

    windows = {"tip": [], "porch": []}
    labels = {"tip": [], "porch": []}
    for index in horizontal:
        start = int(pulses["start"][index])
        end = int(pulses["end"][index])
        head, excluded = head_of_position(start, boundaries, rf)
        if excluded or head < 0:
            continue
        field_index = int(np.searchsorted(boundaries["start"], start, side="right")) - 1
        line_index = (int((start - boundaries["start"][field_index]) / line_samples)
                      if field_index >= 0 else -1)
        if "tip" in kinds and start + tip_width <= envelope.size:
            inner = slice(start + tip_width // 4, start + 3 * tip_width // 4)
            windows["tip"].append(envelope[start:start + tip_width])
            labels["tip"].append((head, field_index + first_field_index, line_index,
                                  start + offset, float(np.mean(frequency_05[inner]))))
        if "porch" in kinds and end + porch_width <= envelope.size:
            inner = slice(end + porch_width // 4, end + 3 * porch_width // 4)
            windows["porch"].append(envelope[end:end + porch_width])
            labels["porch"].append((head, field_index + first_field_index, line_index,
                                    end + offset, float(np.mean(frequency_05[inner]))))

    ratio_cache: Dict[float, Dict[str, float]] = {}

    def ratios_at(carrier: float) -> Dict[str, float]:
        # ratios change slowly with the carrier; a 10 kHz grain is far below
        # the band-pass's own structure and keeps this affordable
        key = round(carrier / 1e4) * 1e4
        if key not in ratio_cache:
            got = bandwidth_ratios(key, pulses["tip_hz"], grid, magnitude)
            ratio_cache[key] = {"modulation": float(got["modulation"][0]),
                                "additive": float(got["additive"][0])}
        return ratio_cache[key]

    cells: List[Dict[str, object]] = []
    templates: Dict[str, np.ndarray] = {}
    interiors: Dict[str, slice] = {}
    counts: Dict[str, Dict[str, object]] = {}
    profiles: Dict[str, Dict[str, object]] = {}
    for kind in kinds:
        if not windows[kind]:
            continue
        stack = np.array(windows[kind])
        info = labels[kind]
        heads = np.array([i[0] for i in info])
        for head in np.unique(heads):
            rows = np.flatnonzero(heads == head)
            block = stack[rows]
            centre = slice(block.shape[1] // 4, 3 * block.shape[1] // 4)
            levels = np.median(block[:, centre], axis=1)
            template = coherent_template(block, levels)
            residual = block - template[None, :] * levels[:, None]
            residual_variance = np.var(residual, axis=0)
            interior = interior_of_template(template, residual_variance,
                                            float(np.median(levels)), block.shape[0])
            key = "%s_head%d" % (kind, head)
            templates[key] = template
            interiors[key] = interior
            profiles[key] = position_split(residual, levels)
            profiles[key]["interior"] = interior
            if interior.stop - interior.start < 8:
                continue
            inner_rows = residual[:, interior]
            pooled = variance_effective_count(
                inner_rows - inner_rows.mean(axis=1, keepdims=True))
            counts[key] = pooled
            for row, cell_index in enumerate(rows):
                head_label, field_label, line_label, position, carrier = info[cell_index]
                inner = block[row, interior]
                level = float(np.mean(inner))
                gain = float(np.interp(carrier, grid, magnitude))
                if gain <= 0:
                    continue
                residual_inner = residual[row, interior] / gain
                ratios = ratios_at(carrier)
                cell = cell_statistics(residual_inner, level / gain,
                                       effective_count=pooled["effective_count"],
                                       kind=kind, head=int(head_label),
                                       field=int(field_label), line=int(line_label),
                                       position=int(position), carrier_hz=carrier,
                                       decoder_gain=gain,
                                       mean_count=pooled["mean_count"],
                                       modulation_bandwidth_ratio=ratios["modulation"],
                                       additive_bandwidth_ratio=ratios["additive"])
                if keep_residuals:
                    cell["residual"] = residual_inner
                if periodogram and cell["usable"]:
                    window = np.hanning(residual_inner.size)
                    spectrum = np.fft.rfft((residual_inner - residual_inner.mean()) * window)
                    cell["periodogram"] = (np.abs(spectrum) ** 2
                                           / max(float(np.sum(window ** 2)), 1e-300))
                    cell["periodogram_hz"] = np.fft.rfftfreq(residual_inner.size, d=1.0 / rate)
                cells.append(cell)
    result = {
        "cells": cells,
        "templates": templates,
        "interiors": interiors,
        "effective_counts": counts,
        "position_splits": profiles,
        "pulses": pulses,
        "fields": boundaries,
        "modulation_bandwidth_hz": envelope_bandwidth_hz(pulses["tip_hz"], grid, magnitude, True),
        "additive_bandwidth_hz": envelope_bandwidth_hz(pulses["tip_hz"], grid, magnitude, False),
        "tip_hz": pulses["tip_hz"],
        "blanking_hz": pulses["blanking_hz"],
        "offset": offset,
    }
    if return_signals:
        result["envelope"] = envelope
        result["frequency_05"] = frequency_05
        result["frequency"] = channels["frequency"]
    return result
