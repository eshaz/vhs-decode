"""Baseband complex equalizer for the demodulated luma, from a measured response.

The horizontal sync pulse is ideal by the video specification, and on a flat
field so is the whole line. What the decoded waveform does that the specified
waveform does not is therefore the playback channel's own response - the
tape, the heads and the VCR's electronics, then the decoder's RF chain - to a
KNOWN input. Measured as a complex ratio against the ideal, that response is
the low-frequency amplitude AND phase behaviour that no content measurement
isolates, and this stage inverts it: `demod <- irfft(E(f) * rfft(demod))`,
`E = exp(-amount * s * log H)`, the same fractional inverse in the log domain
that the luma path equalizer applies to its own measured magnitude, here with
the phase carried as well. It runs on the raw instantaneous frequency - after
the AM-induced phase-noise cancellation, before every stage that reshapes the
signal (spike replacement, video equalizer, de-emphasis) and before the
nonlinear ones (the sub-de-emphasis on the slower tape speeds), because the
electronics act on the signal AFTER the tape is read and are undone first.

The measurements arrive from offline instruments as npz files declaring their
own reference point:
  - `--baseband_lf_response`: the flat-field line-harmonic identification
    (absolute complex H at k * line rate, per field with standard errors);
    load-bearing from the line rate to a few tens of kHz by its own SE.
  - `--sync_step_response`: the sync-step ratio FFT(step)/FFT(ideal) per head
    and polarity with its even/odd decomposition (rise/fall common and
    asymmetry), roughly 70 kHz to 3 MHz.
Every file declares the site it was measured at. At the decoded output the
recorded pre-emphasis has been undone by the decoder's de-emphasis, which is
by construction the inverse of the specified pre-emphasis, so the two cancel
and re-referencing to this site divides out only what remains between here
and there: the video low-pass, the custom video filter and the time base
corrector's interpolator magnitude - each taken from the decoder's own filter
tables, never assumed flat. Dividing out the whole de-emphasis instead would
attribute its shelf to the VCR and re-pre-emphasize the picture. The
converse trap is measured (2026-09-02): a product taken at the raw
demodulated frequency against an UN-emphasized ideal and "re-referenced" by
multiplying in the decoder's de-emphasis model carried that model's phase
as if it were the channel's - the deck's pre-emphasis is not the model's
exact inverse - and read a 0.6 us low-frequency lag the output-referenced
identification of the same chain (-1.4 to -5 degrees to 63 kHz) did not
show. Output-referenced products are therefore preferred: the ideal is
specified at the output, and whatever the decoder's emphasis model gets
wrong is then part of the measured departure and corrected with it.

Laws the fold obeys (agreed with the artifact-model and luma-measurement
lanes): only the EVEN part of a two-polarity measurement may be folded by a
linear stage - the odd part is level-dependent nonlinearity, and enters here
as ERROR in the same units, so the fold is withheld where the two are
comparable; E is exactly unity at DC and unity wherever nothing was measured
(the output stage re-anchors black per field to the porch, so a stage that
reshaped the porch's DC would move picture levels); the fold never reaches
the ring band, where the channel's kernel is operating-point indexed and no
single linear inverse exists - the measurements' own standard errors carry
the fold to zero well below it, and a declared ring pole table hard-limits
it. The error budget is the pooled variance plus the between-head (and
between-instrument) disagreement beyond what the variances explain, and the
garrote `s = max(0, 1 - err2/|log H|^2)` is the one head_switch uses:
withhold what the estimate's own error explains.

THE ERROR BUDGET BOUNDS NOISE AND CONFOUNDING ONLY WHERE IT IS TOLD TO.
The variance term comes from an instrument's own scatter, so it withholds
what does not reproduce; a departure that reproduces because its cause
reproduces - the same content measured again - passes it at full strength.
Two of the three terms are there because of that: the between-head
disagreement catches a departure that is really a head property, and the
odd share catches one that is really level-dependent. Neither catches
content, and nothing here does. A band this stage believes is one that
reproduced and was not explained by those two confounds; establishing that
it is a property of the channel needs the measurement repeated on
different material, which is a question for whoever supplies the file.

Bulk delay is removed, and the reason is the instrument's own null space
rather than a preference: every measurement here centres its ideal on the
fitted 50% sync crossing, so a real bulk delay is absorbed by that
alignment and cannot reach the ratio. Any linear phase that survives is
therefore the alignment's residue, and folding it would shift the whole
picture in time - a timing action by a stage that owns no timing, landing
in the same band the time base corrector works in. The best fit through
the origin, weighted by belief, is subtracted; the dispersion about it is
kept, being the part that is a property of the channel.

Verdict on the first product (flat-field identification, bounce1 SP,
2026-09-02), by the source tape's own self-consistency: the fold does NOT
flatten the flat field it was measured from. Uncorrected, the decoded flat
field is already flat (3.5 IRE peak to peak, 0.36 rms); the full fold
raises that to 13.8 (phase rotated by the measured angle in EITHER sign
makes it worse, 9-14), and the magnitude alone to 4.9 while deepening the
sync tip from -33.7 toward -40 IRE. The instrument's droop at the first
harmonics is therefore the sync pulse's DEPTH deficit against the spec
ideal (the tip sits 6 IRE shallow on both test tapes while black and white
land on spec) projected onto k = 1..4 - a level property of the recording,
not a linear response of the electronics - and four harmonics of inverse
buy tip depth at the price of ripple across the picture. The stage stays
default-off with that product; it awaits a measurement made around the
sync edges themselves (the sync-step export), which the same test judges.
"""

import numpy as np
import numpy.fft as npfft
from scipy.interpolate import PchipInterpolator

import logging

import lddecode.core as ldd


def _log():
    """The decoder's logger while a decode runs, the module's own otherwise."""
    return ldd.logger if getattr(ldd, "logger", None) is not None else logging.getLogger(__name__)


def _smooth(values, width_bins):
    """Hann-weighted mean over `width_bins`, NaN-aware, edge-normalized.

    THIS AVERAGES THE LOGARITHM, DELIBERATELY, AND THAT IS NOT THE
    AVERAGING ERROR RECORDED IN THE RINGING RULES. Estimating a power
    ratio from noisy spectra must average POWER and take the log after,
    because averaging decibels is dragged down by every low-energy bin.
    Here the quantity is a transfer being INVERTED multiplicatively, its
    bins are well determined, and the average is over adjacent resolution
    cells rather than a band: the geometric mean is the right centre for a
    multiplicative quantity, and it errs LOW against the arithmetic one,
    which is the direction the gain law wants an equalizer to err. Changing
    this to a power average would make the fold over-correct at every peak
    inside a resolution cell.
    """
    width = int(round(width_bins))
    if width < 3 or len(values) < width:
        return values
    kernel = np.hanning(width + 2)[1:-1]
    kernel = kernel / kernel.sum()
    good = np.isfinite(values)
    filled = np.where(good, values, 0.0)
    weight = np.convolve(good.astype(np.float64), kernel, mode="same")
    total = np.convolve(filled, kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), np.nan)
    return out


class _Product:
    """One instrument's measurement, re-referenced to this stage's site.

    `freqs` in Hz (ascending, positive), `L = log H_site` (nepers + unwrapped
    radians), `var` its variance in the same units (delta method from the SE
    of |H|), `odd2` the odd-to-even power ratio where the instrument measures
    two polarities (None otherwise), `ring_floor_hz` the lowest certified
    ring pole where the instrument certifies one (None otherwise).
    """

    def __init__(self, name, freqs, H, se, odd2=None, ring_floor_hz=None, resolution_hz=None):
        order = np.argsort(freqs)
        freqs = np.asarray(freqs, dtype=np.float64)[order]
        H = np.asarray(H, dtype=np.complex128)[order]
        se = np.asarray(se, dtype=np.float64)[order]
        self.name = name
        self.freqs = freqs
        self.L, self.var = _log_transfer(H, se)
        self.odd2 = None if odd2 is None else np.asarray(odd2, dtype=np.float64)[order]
        self.ring_floor_hz = ring_floor_hz
        # The instrument's declared resolution (its window's reciprocal
        # length) where it declares one, else the spacing of its points.
        self.resolution_hz = float(resolution_hz) if resolution_hz else float(np.median(np.diff(self.freqs)))
        # Bins finer than the information carry ripple between resolution
        # cells - spectral leakage of the measurement window, not the
        # channel - and an interpolant reproduces that ripple faithfully as
        # a low-frequency oscillation in the fold's kernel (seen as a slow
        # wave across the folded line). So the log-transfer, its variance
        # and the odd share are smoothed over one resolution before use.
        spacing = float(np.median(np.diff(self.freqs))) if len(self.freqs) > 1 else 0.0
        if spacing > 0 and self.resolution_hz > spacing:
            width = self.resolution_hz / spacing
            self.L = _smooth(self.L.real, width) + 1j * _smooth(self.L.imag, width)
            self.var = _smooth(self.var, width)
            if self.odd2 is not None:
                self.odd2 = _smooth(self.odd2, width)


def _site_divisor(rf, freqs_hz):
    """|D(f)| between this site and the decoded output, de-emphasis excluded.

    D = (FVideo / FDeemp) * |H_tbc|: the video low-pass and custom filter as
    the decoder holds them on the block's own rfft grid, times one polyphase
    leg of the time base corrector's interpolator evaluated at the input rate
    (the interpolator runs at the input rate, so its response lives on the
    input-rate axis; the half-sample leg is the least flat one).
    """
    fvideo = np.asarray(rf.Filters["FVideo"])
    fdeemp = np.asarray(rf.Filters["FDeemp"])
    grid = np.arange(len(fvideo)) * (rf.freq_hz / (2 * (len(fvideo) - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        path = np.where(np.abs(fdeemp) > 0, fvideo / fdeemp, 1.0)
    path = np.nan_to_num(np.abs(path), nan=1.0)
    divisor = np.interp(freqs_hz, grid, path)
    lut = getattr(rf, "downscale_sinc_lut", None)
    if lut is not None and np.ndim(lut) == 2:
        taps = np.asarray(lut[lut.shape[0] // 2], dtype=np.float64)
        t = np.arange(len(taps)) - (len(taps) - 1) / 2.0
        phase = np.exp(-2j * np.pi * np.outer(freqs_hz, t) / rf.freq_hz)
        divisor = divisor * np.abs(phase @ taps)
    return divisor


def _is_output_site(site):
    """The decoded output (post-de-emphasis, post-TBC) however the instrument names it."""
    site = str(site).lower()
    return site in ("output", "tbc") or site.startswith("decoded") or "output" in site


def _declared(z, key, default=None):
    if key not in z.files:
        return default
    value = z[key]
    return value.item() if np.ndim(value) == 0 else value


def _within_declared_band(z, path, freqs):
    """The instrument's own model-validity band, `valid_hz`.

    A ratio against an ideal is limited by the ideal's MODEL where the drive
    is weak, not only by noise: near a null of the specified waveform's
    spectrum the measured-to-ideal ratio is consistent from field to field
    and therefore carries a small standard error, yet means nothing. The
    standard error cannot reject it; only the instrument's declaration can,
    so the file must carry one, and a file without it is used in full with a
    warning on the record.
    """
    valid = _declared(z, "valid_hz")
    if valid is None:
        _log().warning(
            "baseband_eq: %s declares no valid_hz - every admitted bin is believed", path
        )
        return np.ones(len(freqs), dtype=bool)
    valid = np.atleast_1d(np.asarray(valid, dtype=np.float64))
    if valid.size >= 2:
        return (freqs >= valid[0]) & (freqs <= valid[1])
    return freqs <= float(valid[0])


def _load_flat_field(rf, path):
    """The flat-field line-harmonic identification (`--baseband_lf_response`)."""
    z = np.load(path, allow_pickle=False)
    site = _declared(z, "site")
    if site is None:
        raise ValueError(f"{path}: the file does not declare its measurement site")
    freqs = np.asarray(z["freqs"], dtype=np.float64)
    se = np.asarray(z["se"], dtype=np.float64)
    mask = np.asarray(z["mask"], dtype=bool) if "mask" in z.files else np.ones(len(freqs), bool)
    if site == "demod_raw":
        if "H_channel" in z.files:
            key = "H_channel"
        elif bool(_declared(z, "ideal_preemphasized", False)):
            key = "H_measured" if "H_measured" in z.files else "H"
        else:
            raise ValueError(
                f"{path}: measured at demod_raw against an un-emphasized ideal "
                "without an H_channel array - the ratio still contains the pre-emphasis"
            )
        divisor = 1.0
    elif _is_output_site(site):
        key = "H"
        divisor = _site_divisor(rf, freqs)
    else:
        raise ValueError(f"{path}: unknown measurement site {site!r}")
    ok = mask & (freqs > 0) & _within_declared_band(z, path, freqs)
    # Per-head arrays, where the instrument ships them, are pooled here with
    # their disagreement in the error budget; the pooled set is the fallback.
    heads = sorted(k[len(key) + 1 :] for k in z.files if k.startswith(key + "_") and f"se_{k[len(key) + 1:]}" in z.files)
    if heads:
        Hs = [np.asarray(z[f"{key}_{h}"], dtype=np.complex128) / divisor for h in heads]
        ses = [np.asarray(z[f"se_{h}"], dtype=np.float64) for h in heads]
        H, se, ok_heads = _pool_heads(Hs, ses)
        ok &= ok_heads
    else:
        H = np.asarray(z[key], dtype=np.complex128) / divisor
        se = np.asarray(z["se"], dtype=np.float64)
    ok &= np.isfinite(H) & (se > 0) & (np.abs(H) > 0)
    return _Product("flat_field", freqs[ok], H[ok], se[ok])


def _log_transfer(H, se):
    """log H (nepers + unwrapped radians) and its variance, over the bins
    where the estimate exists; NaN elsewhere. The unwrap runs over the
    measured bins only - a NaN inside a cumulative unwrap poisons every
    bin after it, which is how an export storing NaN outside its band
    once emptied the whole product."""
    magnitude = np.abs(H)
    good = np.isfinite(H) & np.isfinite(se) & (se > 0) & (magnitude > 0)
    L = np.full(len(H), np.nan + 0j, dtype=np.complex128)
    var = np.full(len(H), np.inf)
    if np.any(good):
        L[good] = np.log(magnitude[good]) + 1j * np.unwrap(np.angle(H[good]))
        var[good] = (se[good] / magnitude[good]) ** 2
    return L, var


def _pool_heads(Hs, ses):
    """Pool per-head complex transfers in the log domain.

    Returns the pooled transfer, an equivalent standard error of |H| that
    carries the pooled variance PLUS the between-head disagreement beyond
    what the variances explain, and the bins where at least one head spoke.
    """
    Ls, vs = [], []
    for H, se in zip(Hs, ses):
        L, v = _log_transfer(np.asarray(H, dtype=np.complex128), np.asarray(se, dtype=np.float64))
        Ls.append(L)
        vs.append(v)
    L, v, tau2 = _pool(np.array(Ls), np.array(vs))
    ok = np.isfinite(L) & np.isfinite(v)
    H = np.where(ok, np.exp(np.where(ok, L, 0.0)), np.nan)
    se = np.sqrt(np.where(ok, v + tau2, np.inf)) * np.abs(np.where(ok, H, 1.0))
    return H, se, ok


def _load_sync_step(rf, path):
    """The sync-step ratio export (`--sync_step_response`), even part only.

    Per head `<head>`: `H_even_<head>`, `H_odd_<head>`, `se_even_<head>` on
    `freqs`; the file declares `site` and `ideal_preemphasized`; an optional
    `ring_poles_hz` table hard-limits the fold below the ring band. The heads
    are pooled here in the log domain with their disagreement entering the
    error budget - blocks carry no field parity, so one filter serves both.
    """
    z = np.load(path, allow_pickle=False)
    site = _declared(z, "site")
    if site is None:
        raise ValueError(f"{path}: the file does not declare its measurement site")
    freqs = np.asarray(z["freqs"], dtype=np.float64)
    heads = sorted(k[len("H_even_"):] for k in z.files if k.startswith("H_even_"))
    if not heads:
        raise KeyError(f"{path}: no H_even_<head> arrays; keys are {sorted(z.files)}")
    divisor = 1.0
    if _is_output_site(site):
        divisor = _site_divisor(rf, freqs)
    elif site != "demod_raw":
        raise ValueError(f"{path}: unknown measurement site {site!r}")
    if site == "demod_raw" and not bool(_declared(z, "ideal_preemphasized", False)):
        divisor = divisor / np.interp(
            freqs,
            np.arange(len(rf.Filters["FDeemp"])) * (rf.freq_hz / (2 * (len(rf.Filters["FDeemp"]) - 1))),
            np.abs(rf.Filters["FDeemp"]),
        )
    Ls, vs, odds = [], [], []
    for head in heads:
        H = np.asarray(z[f"H_even_{head}"], dtype=np.complex128) / divisor
        se = np.asarray(z[f"se_even_{head}"], dtype=np.float64)
        odd = np.asarray(z[f"H_odd_{head}"], dtype=np.complex128)
        Ls.append(_log_transfer(H, se)[0])
        vs.append(_log_transfer(H, se)[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            odds.append(np.where(np.abs(H) > 0, np.abs(odd) ** 2 / np.where(np.abs(H) > 0, np.abs(H) ** 2, 1.0), 0.0))
    L, v, tau2 = _pool(np.array(Ls), np.array(vs))
    odd2 = np.max(np.array(odds), axis=0)
    poles = [np.atleast_1d(z[k]).astype(np.float64) for k in z.files if k.startswith("ring_poles_hz")]
    poles = [p[np.isfinite(p) & (p > 0)] for p in poles]
    poles = [p for p in poles if p.size]
    ring_floor = float(min(np.min(p) for p in poles)) if poles else None
    ok = np.isfinite(L) & np.isfinite(v) & (v > 0) & (freqs > 0)
    ok &= _within_declared_band(z, path, freqs)
    resolutions = [float(z[k]) * 1e6 for k in z.files if k.endswith("_resolution_mhz")]
    product = _Product(
        "sync_step", freqs[ok], np.exp(L[ok]), np.sqrt(v[ok]) * np.abs(np.exp(L[ok])), odd2[ok], ring_floor,
        resolution_hz=max(resolutions) if resolutions else None,
    )
    # The between-head disagreement is part of this product's own error.
    product.var = product.var + tau2[ok]
    return product


def _pool(L, var):
    """Inverse-variance pool of log-transfers with a heterogeneity term.

    `L`, `var` shaped (products, bins); infinite variance means "not
    measured here". Returns the pooled L, its variance, and `tau2` - the
    between-product variance beyond what the individual variances explain
    (DerSimonian-Laird: Q = sum w |L - Lbar|^2 has expectation P - 1 under
    agreement; the excess, scaled by the weights, is the disagreement).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.isfinite(var) & (var > 0), 1.0 / var, 0.0)
    sw = np.sum(w, axis=0)
    count = np.sum(w > 0, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Lbar = np.sum(w * np.where(w > 0, L, 0.0), axis=0) / sw
        v_pool = 1.0 / sw
        Q = np.sum(w * np.abs(np.where(w > 0, L, 0.0) - Lbar) ** 2, axis=0)
        denominator = sw - np.sum(w**2, axis=0) / sw
        tau2 = np.where(
            (count > 1) & (denominator > 0),
            np.maximum(Q - (count - 1), 0.0) / np.where(denominator > 0, denominator, 1.0),
            0.0,
        )
    Lbar = np.where(count > 0, Lbar, np.nan)
    v_pool = np.where(count > 0, v_pool, np.inf)
    return Lbar, v_pool, tau2


def _products(rf):
    """Load every declared measurement once; cached on the decoder."""
    cache = rf.__dict__.get("_baseband_eq_products")
    if cache is not None:
        return cache
    products = []
    flat = getattr(rf, "_baseband_lf_response", None)
    if flat:
        products.append(_load_flat_field(rf, flat))
    sync = getattr(rf, "_sync_step_response", None)
    if sync:
        # A comma-separated list is a CHAIN: each later export was measured
        # on a decode already corrected by the earlier ones (the residual
        # against the ideal after the fold), so its fold is applied in
        # addition - the closed loop that calibrates the response and the
        # amount from the corrected sync pulse itself.
        for stage, path in enumerate(str(sync).split(",")):
            path = path.strip()
            if not path:
                continue
            product = _load_sync_step(rf, path)
            product.chained = stage > 0
            products.append(product)
    rf.__dict__["_baseband_eq_products"] = products
    return products


def _onto_grid(product, freqs):
    """A product's L, variance and odd share on the block grid, PCHIP over
    its own span (shape preserving - no overshoot between coarse bins),
    unmeasured (infinite variance) outside it."""
    n = len(freqs)
    L = np.full(n, np.nan + 0j, dtype=np.complex128)
    var = np.full(n, np.inf)
    odd2 = np.zeros(n)
    resolved = np.zeros(n)
    if len(product.freqs) < 2:
        return L, var, odd2, resolved
    # The band's edges must not be steps: a step in E is a kernel tail that
    # decays as 1/t and overruns the block's cut margins. At the low edge the
    # channel is unity at DC by construction (the instruments match their
    # plateaus), so the interpolation is anchored there; and below the
    # instrument's own resolution - the reciprocal of its window length -
    # neither magnitude nor phase is independently resolved, so the belief
    # rises from zero at DC to full at the resolution. This is also the
    # porch guard: a residual phase tilt at unity gain near DC is a slope
    # across the porch, and the per-field black anchor follows the porch
    # (measured: a 2-4 degree low-frequency phase fold moved the anchor by
    # up to 0.4 IRE per field). At the high edge the belief tapers to zero
    # over the same resolution with the last measured value held. Both by
    # inflating the variance, so the pooling and the garrote see one law.
    spacing = product.resolution_hz
    knots = np.concatenate(([0.0], product.freqs))
    L_knots = np.concatenate(([0.0 + 0.0j], product.L))
    var_knots = np.concatenate(([product.var[0]], product.var))
    top = product.freqs[-1] + spacing
    inside = (freqs >= 0.0) & (freqs <= top)
    f = np.minimum(freqs[inside], product.freqs[-1])
    L[inside] = PchipInterpolator(knots, L_knots.real)(f) + 1j * PchipInterpolator(knots, L_knots.imag)(f)
    taper = np.clip((freqs[inside] - product.freqs[-1]) / spacing, 0.0, 1.0)
    rise = np.clip(freqs[inside] / spacing, 0.0, 1.0)
    weight = (0.5 + 0.5 * np.cos(np.pi * taper)) * (0.5 - 0.5 * np.cos(np.pi * rise))
    # Below the resolution the estimate is not resolved however small its
    # standard error, so the belief itself is bounded there - the variance
    # inflation alone cannot bound it when the error is tiny.
    resolved[inside] = weight
    with np.errstate(divide="ignore"):
        var[inside] = np.maximum(PchipInterpolator(knots, var_knots)(f), 0.0) / np.where(weight > 0, weight**2, np.inf)
    if product.odd2 is not None:
        odd_knots = np.concatenate(([product.odd2[0]], product.odd2))
        odd2[inside] = np.maximum(PchipInterpolator(knots, odd_knots)(f), 0.0)
    return L, var, odd2, resolved


def site_transfer(rf, n):
    """The pooled measured log-transfer and its garrote on the block grid.

    Returns `(L, s)` - `L = log H_site` (nan where unmeasured) and the
    evidence weight `s` in [0, 1] (zero where unmeasured) - or None when no
    measurement is declared. `exp(s * L)` is the channel as far as it is
    believed, which is what a consumer regularizing its own estimate toward
    the measured channel wants. Cached per block length.
    """
    cache = rf.__dict__.get("_baseband_eq_site")
    if cache is not None and cache["n"] == n:
        return cache["L"], cache["s"]
    products = _products(rf)
    if not products:
        return None
    freqs = np.arange(n // 2 + 1) * (rf.freq_hz / n)
    chained = [p for p in products if getattr(p, "chained", False)]
    products = [p for p in products if not getattr(p, "chained", False)]
    stacks = [_onto_grid(p, freqs) for p in products]
    L, v_pool, tau2 = _pool(
        np.array([s[0] for s in stacks]), np.array([s[1] for s in stacks])
    )
    odd2 = np.max(np.array([s[2] for s in stacks]), axis=0)
    resolved = np.max(np.array([s[3] for s in stacks]), axis=0)
    err2 = v_pool + tau2 + odd2
    measured = np.isfinite(L) & np.isfinite(err2)
    L = np.where(measured, L, 0.0)
    power = np.abs(L) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(measured & (power > 0), 1.0 - err2 / np.where(power > 0, power, 1.0), 0.0)
    s = np.clip(np.nan_to_num(s), 0.0, 1.0) * resolved
    # The fold's ceiling is the channel's own bandwidth: the frequency where
    # the smoothed even response has fallen 3 dB. Below it a departure is
    # equalization; above it the roll-off is the recorded bandwidth (it does
    # not move under the RF equalizer, and an ideal edge sharper than the
    # channel passes would only ring if inverted). A raised-cosine taper
    # over the octave below. The instruments' ring poles no longer bound
    # the fold: smoothing over the resolution averages them out of the
    # transfer, so a fold from the smoothed transfer cannot reproduce them
    # - they stay for the audit.
    ceiling = _bandwidth_hz(freqs, L, s)
    if ceiling is not None:
        taper = np.clip((freqs - ceiling / 2.0) / (ceiling / 2.0), 0.0, 1.0)
        s = s * (0.5 + 0.5 * np.cos(np.pi * taper))
    s[0] = 0.0
    # The instrument's own null space, removed rather than folded. Every
    # measurement here centres its ideal on the fitted 50% sync crossing,
    # so a real bulk delay in the channel is absorbed by that alignment and
    # cannot appear in the ratio: any linear phase left is the alignment's
    # own residue. Folding it would shift the whole picture in time, which
    # is a timing action by a stage that owns no timing and lands in the
    # band the time base corrector is working in (measured: 0.14 output
    # samples on one tape at half amount, against a timing residual of 0.15
    # that another stage spent a day reducing). The best fit through the
    # origin, weighted by belief, is subtracted; dispersion about it - the
    # part that is a channel property - is kept.
    believed = s > 0
    if np.count_nonzero(believed) > 2:
        weight = s[believed]
        f_believed = freqs[believed]
        phase = np.unwrap(np.imag(L[believed]))
        denominator = float(np.sum(weight * f_believed * f_believed))
        if denominator > 0:
            slope = float(np.sum(weight * f_believed * phase)) / denominator
            L = L - 1j * slope * freqs
            rf.__dict__["_baseband_eq_bulk_delay_us"] = -slope / (2.0 * np.pi) * 1e6
    if chained:
        # Later stages fold what remained after the earlier ones: their
        # believed departures add in the log domain. The pair returned is
        # the total believed departure with unit belief, so consumers of
        # exp(s * L) see the whole chain. Each stage keeps the first
        # stage's ceiling: a residual measured above the bandwidth is the
        # bandwidth again.
        total = s * L
        # The band is frozen with the ceiling, and for the same reason. Each
        # later export is measured on a decode the earlier ones corrected,
        # so its instrument may honestly declare a WIDER validity than the
        # first - the measurement got better because the signal did. Folding
        # that extra band would mean the total correction reaches into
        # territory the first pass never validated, and each pass's total
        # would then cover a different span, which is the one thing that
        # makes two passes incomparable. Later stages refine within the
        # first stage's band and are told when they wanted more.
        believed = freqs[s > 0]
        first_band = (believed.min(), believed.max()) if len(believed) else None
        for product in chained:
            Lc, vc, oc, rc = _onto_grid(product, freqs)
            if first_band is not None:
                outside = (freqs < first_band[0]) | (freqs > first_band[1])
                wanted = np.count_nonzero(outside & np.isfinite(Lc) & np.isfinite(vc))
                if wanted:
                    _log().info(
                        "baseband_eq: a later measurement covers %d bins outside the first "
                        "pass's %.0f-%.0f kHz band; held to that band so every pass's total "
                        "spans the same frequencies",
                        wanted, first_band[0] / 1e3, first_band[1] / 1e3)
                rc = np.where(outside, 0.0, rc)
            measured_c = np.isfinite(Lc) & np.isfinite(vc)
            Lc = np.where(measured_c, Lc, 0.0)
            power_c = np.abs(Lc) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                sc = np.where(measured_c & (power_c > 0), 1.0 - (vc + oc) / np.where(power_c > 0, power_c, 1.0), 0.0)
            sc = np.clip(np.nan_to_num(sc), 0.0, 1.0) * rc
            if ceiling is not None:
                taper = np.clip((freqs - ceiling / 2.0) / (ceiling / 2.0), 0.0, 1.0)
                sc = sc * (0.5 + 0.5 * np.cos(np.pi * taper))
            total = total + sc * Lc
        L = total
        s = (np.abs(total) > 0).astype(np.float64)
        s[0] = 0.0
    rf.__dict__["_baseband_eq_site"] = {"n": n, "L": L, "s": s, "err2": err2, "tau2": tau2}
    _audit(rf, freqs, L, s, err2, tau2)
    return L, s


def _bandwidth_hz(freqs, L, s):
    """The -3 dB point of the believed even response, or None if it never falls that far."""
    magnitude_db = 20.0 * L.real / np.log(10.0)
    below = np.where((s > 0) & (magnitude_db < -3.0))[0]
    if len(below) == 0:
        return None
    return float(freqs[below[0]])


def _audit(rf, freqs, L, s, err2, tau2):
    """Log what the fold believes, per band, and what its kernel spans."""
    edges = [0.0, 0.05e6, 0.1e6, 0.3e6, 1e6, 3e6, freqs[-1] + 1]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi) & (s > 0)
        if not np.any(band):
            continue
        rows.append(
            "%.2f-%.2f MHz: s=%.2f |H|=%+.2f dB phase=%+.1f deg z(heads)=%.1f"
            % (
                lo / 1e6,
                hi / 1e6,
                float(np.mean(s[band])),
                float(np.mean(L.real[band]) * 20 / np.log(10)),
                float(np.degrees(np.mean(L.imag[band]))),
                float(np.sqrt(np.mean(tau2[band] / np.maximum(err2[band], np.finfo(np.float64).tiny)))),
            )
        )
    E = np.exp(-s * L)
    E[0] = 1.0
    ir = npfft.irfft(E - 1.0, n=2 * (len(freqs) - 1))
    total = float(np.sum(ir**2)) or 1.0
    cut = int(getattr(rf, "blockcut", 0))
    cut_end = int(getattr(rf, "blockcut_end", 0))
    causal_tail = float(np.sum(ir[cut : len(ir) // 2] ** 2)) / total if cut else 0.0
    anticausal = float(np.sum(ir[len(ir) - cut_end :] ** 2)) / total if cut_end else 0.0
    measured = s > 0
    delay_us = 0.0
    if np.sum(measured) > 1:
        f, phi = freqs[measured], L.imag[measured]
        delay_us = -float(np.sum(f * phi) / np.sum(f * f)) / (2 * np.pi) * 1e6
    _log().info(
        "baseband_eq: %s; kernel energy beyond the front cut %.1e, in the end cut %.1e; "
        "residual linear phase %.4f us after removing the alignment's own %.4f us",
        "; ".join(rows) if rows else "nothing measured",
        causal_tail,
        anticausal,
        delay_us,
        float(rf.__dict__.get("_baseband_eq_bulk_delay_us", 0.0)),
    )


def equalizer(rf, n, amount):
    """E(f) on the block grid for this amount, cached; None without a measurement."""
    cache = rf.__dict__.get("_baseband_eq")
    if cache is not None and cache["n"] == n and cache["amount"] == amount:
        return cache["E"]
    site = site_transfer(rf, n)
    if site is None:
        return None
    L, s = site
    E = np.exp(-amount * s * L)
    E[0] = 1.0
    if n % 2 == 0:
        E[-1] = E[-1].real
    rf.__dict__["_baseband_eq"] = {"n": n, "amount": amount, "E": E}
    return E


def correct(rf, demod, amount):
    """Apply the measured baseband inverse to one block, in place.

    Full-block circular multiply, the pattern every filter in this chain
    uses; the block's cut margins absorb the kernel's wrap (audited at build).
    Stateless, so the plot's second demodulation pass sees the same filter.
    """
    E = equalizer(rf, len(demod), amount)
    if E is None:
        return
    demod[:] = npfft.irfft(E * npfft.rfft(demod), n=len(demod))
