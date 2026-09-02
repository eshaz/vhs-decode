"""Cancel AM-induced phase noise in the demodulated luma, self-calibrated.

The luma FM carrier is recorded at constant amplitude, so the amplitude that
comes back is a measurement channel: everything on it was put there by the
playback path. The luma amplitude stage fits the predictable part of that
channel, and what it cannot predict is the residual - the path misbehaving:
the DC step the head amplifiers put on the carrier at every head switch,
the collapses at dropout edges, and the tape's own modulation noise.
Emphasis and picture are encoded before those arise, so they shape the
fitted expectation, never the residual.

The factor model of the measured envelope, each factor with one owner:

    A(t) = S(f) . L(f) . C(t) . n(t)

    S - the decoder's own RF path response          [luma_amplitude]
    L - the fitted tape/head frequency response,
        pooled and averaged across the decode       [luma_amplitude]
    C - the sweep collapse: the predictable,
        content-correlated amplitude loss where
        the carrier sweeps                          [luma_amplitude]
    n - the residual. This module's subject.

The measured dimension is real: d = ln n. Its relationship to the
demodulated luma is complex - the phase partner of the measured amplitude
disturbance - and the mapping is a property of the decoder's own band-pass,
not a universal operator. Small-signal derivation: a real gain event
gamma(t) on a carrier at f_c, seen through the channel H, has the complex
log-envelope perturbation v with

    D(f)   = spectrum of Re v  (the measured d)
    Psi(f) = spectrum of Im v  (the phase artifact)
    Psi(f) = T(f) . D(f)

    T(f) = -j (H(fc+f) H*(fc) - H*(fc-f) H(fc))
              / (H(fc+f) H*(fc) + H*(fc-f) H(fc))

so the phase is carried entirely by the channel's ASYMMETRY about the
carrier: a channel symmetric about f_c converts amplitude to no phase at
all. The demodulator reports the phase's rate, psi'/2pi, which fuses with T
into a single spectral multiply by K(f) = j f T(f) over the whole RF block
(the block overlap exists precisely so a full-block FFT carries no edge
effects into the kept span).

Derivation record (session tools, 2026-09-01): the naive form Psi = H{d}
(spectral |f| alone) was tried first and FALSIFIED on a synthetic causal
chain - correlation +0.09 against the observed artifact. The kernel above,
with the H(fc) phase reference the naive form lacks, reproduces the
observed artifact from the measured d alone at correlation +0.94..+0.97
with best-fit gain 0.88..0.94 across carrier positions, causally.

THE CARRIER TERM. T depends on where the carrier sits, and the carrier
sweeps with the picture. One kernel per block, evaluated at the block's
median carrier, was round 1's stance, and its measured cost was porch noise
(the sync and blanking samples sit ~1 MHz from a bars-content median). The
correction is therefore first-order in the carrier: K0 is evaluated at the
0 IRE anchor and K1 is the secant of K between the format's deviation
extremes (sync tip to peak white), so the prediction is

    p = irfft(K0 . D) + dfc(t) . irfft(K1 . D)

with dfc the local carrier offset from the anchor, box-averaged over the
model's own track_width (the same support the envelope sample formed over).
The anchors are the MEASURED levels the amplitude stage exports
(porch_hz, tip_hz, hz_ire_measured - on the first measured tape the deck
recorded 5.8% shallow, so the measured axis differs from spec materially),
with the format's spec values as the cold-start fallback. Offline, on real
captures, the carrier term with measured anchors turned a +183% porch-noise
injection under plain full subtraction into a -22% porch-noise REMOVAL.

THE SELF-CALIBRATING GAIN. What fraction of the prediction truly appears
in the demodulated luma is not assumed; it is measured, per frequency,
against the decode itself. Physics: multiplicative disturbances (tape
modulation noise, head events) produce CORRELATED amplitude and phase
through the band's asymmetry - cancelable; additive channel noise has
independent quadratures to first order - its phase share is invisible to
the envelope and must not be chased. The regression

    W(f) = <Re P* U> / (<|P|^2> + ridge)

of the observed demod U on the prediction P measures exactly the cancelable
share: the additive part contributes prediction power but no cross-power,
so W settles at the Wiener-optimal shrinkage, and picture content inflates
only the estimate's variance (averaged away over blocks), not its
expectation - with one systematic exception, handled next.

Estimation hygiene, each element measured necessary on real captures:
 - The estimation spectra are Hann-windowed (the correction path is not).
   Without the window, line-locked content - picture and any expectation
   error alike, deterministic on a static pattern - leaks across the whole
   low band and the regression's numerator grows without bound (measured:
   W(0.1-1 MHz) climbing past 4 and rising with block count).
 - The line-harmonic bins (+-2 bins, the Hann main lobe) and the DC-side
   bins are EXCLUDED from the regression, and the smoothing that steadies
   the estimate averages over unmasked bins only - notch after smoothing
   was measured useless, the box smears the harmonics everywhere first.
   W is interpolated across the excluded bins; deterministic line-locked
   structure can then no longer vote, while the between-harmonic noise -
   the honest evidence - decides. With this, the runaway flattens to a
   stable plateau and a flat-field decode shows W>1 in 0.5% of bins.
 - The smoothing box spans the bins one line period occupies (the widest
   an event's spectral structure can be for anything shorter than a line).
 - The ridge is the posterior of a zero-centred unit prior - the weakest
   prior consistent with the physical bound |W| <= 1 - so W starts near
   zero (err low), rises as evidence accumulates, and is effectively
   converged within a field. No thresholds anywhere; W replaces round 1's
   spectral subtraction and garrote outright.
 - W is floored at 0 and capped at f/|K0| - the bound belongs to the TOTAL
   applied transfer (the single-sideband limit f), not to the model's
   share: the modeled T omits the record-side lower-sideband truncation,
   and the measured per-band transfer demands W = 1.4 in the 0.3-2 MHz
   bands that carry the picture (a cap at 1 was measured to withhold a
   third of the correction exactly there - the "not enough low frequency
   correction" symptom). An evidence garrote then withholds whatever the
   estimate's own standard error explains, so bands with thin evidence -
   beyond the band, where the f-weighted prediction power is enormous -
   apply nothing instead of injecting Var(W)-scaled noise.

Measured on the captures this design was built against (75bars SP /
bounce SP, 160 blocks each): flat active picture high-frequency noise
-11%, porch noise -22%, flat-field content near-neutral, versus full
subtraction's -16% flats bought with +183% porch. A line-comb on d was
tried at the correct period and REJECTED: it discards real cancelable
line-repeating content (-11% falls to -7%) without fixing the bias.

What this cannot remove: the phase share of additive noise (independent
quadratures), and a pure inter-head phase jump with no amplitude signature
(an all-pass event leaves no residual).

Interface: the expectation and the measured level axis are owned by the
luma amplitude stage through the pinned contract - block_model(rf) (None
until the first measured field; porch_hz/tip_hz/hz_ire_measured are 0.0
until measured), block_log_residual(...), block_sweep_phase(...) - and
this module never reads that stage's internals. The measured axis is an
EXPORT: the amplitude stage's own binning deliberately keeps its spec
anchor, and nothing here changes that. The deterministic sweep-imposed
phase is the other half of the complex relationship; its composition point
below stays parked pending arbitration.

Where: per RF block, in place on the raw demodulated luma, immediately
after demodulation and before every equalization stage - the artifact is
introduced after the tape is read, so it is removed first, in the reverse
order of how it arose. The calibration accumulates as blocks are
demodulated; under threading a block may read sums a few blocks stale and
increments may race (lost-update class, the same accepted caveat as the
equalizer's shared state) - single-threaded decodes are exactly
reproducible.
"""

import numpy as np
import scipy.ndimage as ndi
from numpy import fft as npfft

from vhsdecode import luma_amplitude


def _kernel(rf, n, carrier_bin):
    """K(f) = j f T(f) on the rfft grid of an n-sample block.

    T is built by indexing the decoder's own complex RF response about the
    carrier bin; modular indexing is the honest discrete-time evaluation
    (the sampled filter's response is periodic, and a negative-frequency
    lookup lands on the conjugate bin by the table's own symmetry). Where
    both sidebands of the response have died there is no measurement to map,
    and T is set to zero there rather than divided to noise.
    """
    h_full = np.asarray(rf.Filters["RFVideo"])
    grid = len(h_full)
    k = np.arange(n // 2 + 1)
    # rfft bin k of the block maps to full-grid bin k * grid / n; the block
    # length and the filter grid are both the decoder's blocklen, so the
    # ratio is exact for the only caller.
    step = grid // n if n and grid % n == 0 else None
    if step is None:
        return None
    h0 = h_full[carrier_bin]
    h_pos = h_full[(carrier_bin + k * step) % grid]
    h_neg = h_full[(carrier_bin - k * step) % grid]
    num = h_pos * np.conj(h0) - np.conj(h_neg) * h0
    den = h_pos * np.conj(h0) + np.conj(h_neg) * h0
    dead = np.abs(den) == 0.0
    t_map = np.where(dead, 0.0, -1j * num / np.where(dead, 1.0, den))
    # |T| = 1 is the single-sideband limit - one sideband carrying
    # everything. Beyond it the two sidebands are cancelling in the
    # denominator: a measurement-dead zone, not information, so the map is
    # capped there rather than allowed to amplify emptiness.
    t_mag = np.abs(t_map)
    over = t_mag > 1.0
    t_map = np.where(over, t_map / np.where(over, t_mag, 1.0), t_map)
    freqs = k * (rf.freq_hz / n)
    return 1j * freqs * t_map


def _anchors(rf, model):
    """The level axis the kernels stand on: measured when the amplitude
    stage has exported it, the format's specified values until then."""
    ire0 = float(getattr(model, "porch_hz", 0.0) or 0.0)
    tip = float(getattr(model, "tip_hz", 0.0) or 0.0)
    hz_ire = float(getattr(model, "hz_ire_measured", 0.0) or 0.0)
    if ire0 <= 0.0 or tip <= 0.0 or hz_ire <= 0.0:
        ire0 = float(rf.SysParams["ire0"])
        hz_ire = float(rf.SysParams["hz_ire"])
        tip = ire0 + float(rf.SysParams["vsync_ire"]) * hz_ire
    white = ire0 + 100.0 * hz_ire
    return ire0, tip, white


def _kernels(rf, n, model):
    """K0 at the 0 IRE anchor and the carrier-slope secant K1, cached.

    Rebuilt when an anchor moves by at least one bin of the filter table -
    the kernel's own resolution; finer motion cannot change it.
    """
    ire0, tip, white = _anchors(rf, model)
    grid = len(rf.Filters["RFVideo"])
    quant = (int(round(ire0 * grid / rf.freq_hz)),
             int(round(tip * grid / rf.freq_hz)),
             int(round(white * grid / rf.freq_hz)), n)
    cached = rf.__dict__.get("_head_switch_kernels")
    if cached is not None and cached[0] == quant:
        return cached[1], cached[2], cached[3], ire0
    k0 = _kernel(rf, n, quant[0] % grid)
    k_tip = _kernel(rf, n, quant[1] % grid)
    k_white = _kernel(rf, n, quant[2] % grid)
    if k0 is None or k_tip is None or k_white is None:
        return None, None, None, ire0
    k1 = (k_white - k_tip) / max(white - tip, 1.0)
    # The gain cap per band. The physical bound is on the TOTAL applied
    # transfer: |W . K0| may not exceed the single-sideband limit f (one
    # sideband carrying everything), so W itself is bounded by f/|K0| =
    # 1/|T_model| - NOT by 1. The model's T omits the record-side lower-
    # sideband truncation (the software band-pass is nearly symmetric at
    # small offsets where the recorded signal is not), and the measured
    # per-band transfer demands W = 1.4 in the 0.3-2 MHz bands that carry
    # the picture; a cap at 1 was measured to withhold a third of the
    # correction exactly there. Where the model carries no direction at
    # all the cap is zero.
    freqs = np.arange(n // 2 + 1) * (rf.freq_hz / n)
    mag = np.abs(k0)
    w_cap = np.where(mag > 0.0, freqs / np.where(mag > 0.0, mag, 1.0), 0.0)
    rf.__dict__["_head_switch_kernels"] = (quant, k0, k1, w_cap)
    return k0, k1, w_cap, ire0


def _calibration(rf, n):
    """The estimator's state and fixed furniture for n-sample blocks.

    The notch: with a Hann window a line-locked tone occupies its harmonic
    bin +-2 (the window's main lobe), so those bins - and the DC-side bins
    below the first harmonic, where field- and frame-locked structure
    lives unresolved - carry the deterministic content and are excluded
    from the regression. The smoothing box spans the bins one line period
    occupies: n // linelen, the spectral scale of sub-line structure.
    """
    cal = rf.__dict__.get("_head_switch_cal")
    if cal is not None and cal["n"] == n:
        return cal
    bins = n // 2 + 1
    line_period_us = float(rf.SysParams.get("line_period", 0.0))
    line_hz = 1e6 / line_period_us if line_period_us > 0 else rf.freq_hz / rf.linelen
    df = rf.freq_hz / n
    ok = np.ones(bins, dtype=np.float64)
    for harmonic in np.arange(line_hz, df * (bins - 1), line_hz):
        centre = int(round(harmonic / df))
        ok[max(centre - 2, 0) : centre + 3] = 0.0
    ok[:3] = 0.0
    box = max(1, n // int(rf.linelen))
    cal = {
        "n": n,
        "cross": np.zeros(bins, dtype=np.complex128),
        "power": np.zeros(bins),
        "seen": np.zeros(bins),
        "blocks": 0,
        "ok": ok,
        "kern": np.ones(2 * box + 1),
        "window": np.hanning(n),
        "j": 2 * box + 1,
    }
    rf.__dict__["_head_switch_cal"] = cal
    return cal


# Phase bins for the applied-waveform line fold: about four samples per bin
# at the decoder's rate - fine enough that a folded profile carries the
# line-locked structure, coarse enough that each bin averages every line in
# the block.
_COMB_BINS = 794


def _line_period(rf):
    """The line period in samples, from the format's own timing."""
    line_period_us = float(rf.SysParams.get("line_period", 0.0))
    if line_period_us > 0:
        return rf.freq_hz * line_period_us * 1e-6
    return float(getattr(rf, "linelen", 0.0))


def _boxmean(x, width):
    """Centred running mean over `width` samples, edges clamped."""
    w = max(int(width), 1)
    c = np.cumsum(np.concatenate([[0.0], x]))
    idx = np.arange(len(x))
    lo = np.clip(idx - w // 2, 0, len(x))
    hi = np.clip(idx + w - w // 2, 0, len(x))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def locate(field):
    """Locate the head switch region(s) on an assembled field.

    Identification, not correction: the correction is samplewise and needs
    no location, but the switch's own robust signature - the sustained
    offset the head amplifiers put on the carrier, decaying with the
    capture chain's AC coupling - is a DC feature the phase kernel nulls by
    design, so it never shows in the applied trace. It shows, at tens of
    sigma, in the luma amplitude stage's own per-field deviation: exactly
    the measurement this feature was asked to identify the switch from.
    The debug plot marks what this returns.

    The deviation's log is tracked as one median per LINE, binned on the
    field's own line locations - not on a fixed stride, which was tried
    and failed: fixed bins beat against the line structure, so the
    line-locked deviation profile (largest at the sync phases, where the
    response's low end is extrapolated) turned into a 50-70 mNp structured
    wander that buried the 130 mNp switch. Line-aligned bins each cover an
    identical phase mix, the line-locked profile contributes the same
    constant to every bin, and the track carries only line-to-line change:
    the switch offset, dropouts, drift. A median per line, so picture
    edges and in-line dropouts do not vote.

    A region is a run of lines standing off the field body further than
    the largest excursion that many lines of the track's own scatter would
    reach - the extreme-value bound, so a clean field reports nothing.
    Runs are scored by integrated offset, which lets the lines-long switch
    outrank a deep one-line dropout. At most the format's expected count
    of regions is returned as (start, end, offset_db, sigma) in RF sample
    coordinates; empty when nothing stands out or nothing was measured.
    """
    measured = luma_amplitude.measured_amplitude_deviation(field)
    if measured is None:
        return []
    deviation = np.asarray(measured.deviation, dtype=np.float64)
    linelocs = np.asarray(getattr(field, "linelocs", ()), dtype=np.float64)
    if not len(deviation) or len(linelocs) < 16:
        return []

    bounds = np.clip(linelocs.astype(np.int64), 0, len(deviation))
    with np.errstate(divide="ignore"):
        x = np.log(deviation)
    track = np.full(len(bounds) - 1, np.nan)
    for i in range(len(track)):
        seg = x[bounds[i] : bounds[i + 1]]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            track[i] = np.median(seg)
    valid = np.isfinite(track)
    if np.count_nonzero(valid) < 16:
        return []

    body = np.median(track[valid])
    spread = 1.4826 * np.median(np.abs(track[valid] - body))
    if spread <= 0.0:
        return []
    threshold = spread * np.sqrt(2.0 * np.log(np.count_nonzero(valid)))
    off = np.where(valid, track - body, 0.0)
    hot = np.abs(off) > threshold
    if not np.any(hot):
        return []

    # Two-threshold region growing: a region must be SEEDED by a line past
    # the extreme-value bound, but once seeded it extends over neighbours
    # past half that bar - a sustained offset flutters across a single
    # threshold, and the flutter is not two events.
    hold = np.abs(off) > 0.5 * threshold
    idx = np.flatnonzero(np.diff(np.concatenate([[0], hold.astype(np.int8), [0]])))
    runs = [(a, b) for a, b in zip(idx[::2], idx[1::2]) if np.any(hot[a:b])]
    # merge runs separated by a single line
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < 2:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    scored = sorted(
        merged, key=lambda r: float(np.sum(np.abs(off[r[0] : r[1]]))), reverse=True
    )
    # The field's data overhangs into the next field by the extra process
    # lines, so the array can hold one more switch region than the format
    # counts per field - typically the field's own, at its first lines,
    # AND the next field's, at the tail.
    keep = int(field.rf.SysParams.get("head_switches_per_field", 1)) + 1
    regions = []
    for a, b in sorted(scored[:keep]):
        seg = off[a:b]
        regions.append(
            (
                int(bounds[a]),
                int(bounds[b]),
                # nepers to dB
                float(np.mean(seg) * (20.0 / np.log(10.0))),
                float(np.max(np.abs(seg)) / spread),
            )
        )
    return regions


def correct(rf, demod, envelope, amount, sweep_amount=0.0, calibrate=True):
    """Cancel the residual's phase artifact from the demodulated luma.

    In place on demod (raw instantaneous frequency, Hz, one RF block).
    `amount` scales the measured, clipped per-band gain. `calibrate=False`
    applies without feeding the estimator (the debug plot's re-demodulation
    of a differently equalized channel must not vote). Returns the
    subtracted trace in IRE when the luma_noise plot asked for it, else
    None. Inert until the amplitude stage's block model exists, and nearly
    inert while calibration evidence is thin.
    """
    block_model = getattr(luma_amplitude, "block_model", None)
    if block_model is None or envelope is None:
        return None
    model = block_model(rf)
    if model is None or len(envelope) != len(demod):
        return None

    # The residual, C included in the expectation - the content-blind
    # quantity.
    d = np.asarray(
        luma_amplitude.block_log_residual(model, envelope, demod), dtype=np.float64
    )

    # The decoder's own definition of a dropout bounds how far the residual
    # is believed. Beyond it lies the dropout machinery's territory - and
    # the residual helper hands back an out-of-range marker (the constant
    # -100.0) for samples it could not measure at all - so past the bound
    # there is NO evidence, not extreme evidence: excluded, never clipped.
    # (Measured: clipping planted 1.7-neper spikes that dwarfed every event.)
    limit = float(-np.log(rf.dod_options.dod_threshold_p))
    d = np.where(np.abs(d) <= limit, d, 0.0)
    if not np.any(d):
        return None

    n = len(d)
    k0, k1, w_cap, ire0_hz = _kernels(rf, n, model)
    if k0 is None:
        return None

    track_width = max(int(getattr(model, "track_width", 1)), 1)
    level = _boxmean(demod, track_width)
    dfc = level - ire0_hz

    # Sweep-gated deflation. Where the carrier sweeps hard - every picture
    # and sync transition - the expectation is at its least trustworthy:
    # the model's own transition error (the convolution-memory residual the
    # closed form cannot carry) is content-locked there, and applying its
    # kernel image painted a coherent ring at every bar edge (measured:
    # up to 1.2 IRE peak-to-peak, identical each line). The residual is
    # therefore deflated by the same steadiness law the amplitude stage
    # uses for the same reason - 1/(1+(slew/knee)^2), knee and settling
    # span from the decoder's own filter (path_transient_scale) - with the
    # slew held over three settling spans so the gate covers the ring that
    # persists after the slope returns to zero. The discrimination is the
    # physics this module rests on: the head-switch step and the tape's
    # modulation noise ride a carrier that is NOT sweeping, so the flat-
    # field noise win and the switch correction pass untouched.
    knee, span = luma_amplitude.path_transient_scale(rf)
    slew = np.abs(np.gradient(level))
    hold = max(3 * int(span), 1)
    slew = ndi.maximum_filter1d(slew, size=hold, mode="nearest")
    d = d / (1.0 + (slew / knee) ** 2)

    spectrum = npfft.rfft(d)

    cal = _calibration(rf, n)
    if calibrate:
        window = cal["window"]
        d_spec = npfft.rfft(d * window)
        u_spec = npfft.rfft((demod - np.mean(demod)) * window)
        cal["cross"] += u_spec * np.conj(d_spec)
        cal["power"] += d_spec.real**2 + d_spec.imag**2
        cal["seen"] += u_spec.real**2 + u_spec.imag**2
        cal["blocks"] += 1
    if cal["blocks"] == 0:
        return None

    # The applied transfer is the MEASURED complex per-band transfer G(f)
    # from the residual to the demodulated luma - not the model kernel
    # scaled by a scalar gain. The model's T was measured wrong in shape at
    # both ends: it omits the record-side lower-sideband truncation, so the
    # true coupling in the 0.3-2 MHz picture bands is 1.4x what the model
    # carries (a scalar gain capped at 1 withheld exactly that - the "not
    # enough low frequency correction" symptom), and above ~3 MHz the model
    # saturates at the single-sideband limit where the real coupling has
    # collapsed, so any scalar gain on the model's shape injects there (the
    # "out of band noise" symptom). The measurement has the right shape at
    # every band by construction; the model kernel remains as the carrier-
    # slope term and as the phase prior its derivation record validates
    # (measured phase agreement within ~4-16 degrees in band).
    #
    # Estimation: smooth over unmasked bins only (the harmonic notch), a
    # ridge whose zero-centred prior width is the single-sideband limit f
    # itself (the largest transfer physics permits), the same limit as a
    # hard cap on the estimate's magnitude, and an evidence garrote that
    # withholds whatever the estimate's own standard error explains - so
    # bands with thin evidence apply nothing rather than Var(G)-scaled
    # noise.
    ok = cal["ok"]
    kern = cal["kern"]
    num = (
        np.convolve(cal["cross"].real * ok, kern, mode="same")
        + 1j * np.convolve(cal["cross"].imag * ok, kern, mode="same")
    )
    den_d = np.convolve(cal["power"] * ok, kern, mode="same")
    den_u = np.convolve(cal["seen"] * ok, kern, mode="same")
    freqs = np.arange(n // 2 + 1) * (rf.freq_hz / n)
    f_safe = np.maximum(freqs, freqs[1] if n > 1 else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = num / (
            den_d + den_u / (2.0 * cal["j"] * cal["blocks"] * f_safe**2)
        )
    g = np.nan_to_num(g)
    g_mag = np.abs(g)
    over = g_mag > freqs
    g = np.where(over, g * np.where(over, freqs / np.where(over, g_mag, 1.0), 1.0), g)

    residual = np.maximum(den_u - (np.abs(g) ** 2) * den_d, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se2 = residual / (
            2.0 * cal["j"] * cal["blocks"] * np.maximum(den_d, np.finfo(np.float64).tiny)
        )
        shrink = 1.0 - se2 / np.maximum(np.abs(g) ** 2, np.finfo(np.float64).tiny)
    g = g * np.clip(np.nan_to_num(shrink), 0.0, 1.0)

    # The carrier-slope term rides the same measured calibration: the
    # model's carrier DEPENDENCE (how the kernel varies across the
    # deviation range) is trusted, but its per-band magnitude is corrected
    # by the measured-to-model ratio g/K0 - applying the model-shaped K1
    # raw was measured to inject +7 dB at 3-4.5 MHz, exactly where the
    # model's shape is wrong.
    k0_mag = np.abs(k0)
    k0_ok = k0_mag > 0.0
    ratio = np.where(k0_ok, g / np.where(k0_ok, k0, 1.0), 0.0)
    applied = amount * (
        npfft.irfft(g * spectrum, n=n)
        + dfc * npfft.irfft(k1 * ratio * spectrum, n=n)
    )

    # The correction must not apply a field-coherent, line-locked pattern.
    # The line-locked share of the residual arrives through the kernel as a
    # convolution spread across the whole line, and was measured painting a
    # coherent ring at every bar edge (up to 1.2 IRE peak-to-peak,
    # identical each line) - content-locked structure a per-band gain
    # cannot separate and a time-domain gate on d only relocates (measured:
    # sweep-gating d left the rings standing). So the applied waveform's
    # own line-folded mean is removed per block: everything the correction
    # would do identically on every line - painted junk and any small real
    # line-repeating share alike - is withheld, while the incoherent noise
    # cancellation and once-per-field events pass whole.
    line_period = _line_period(rf)
    if line_period > 0:
        bins = np.minimum(
            ((np.arange(n) / line_period) % 1.0 * _COMB_BINS).astype(np.int64),
            _COMB_BINS - 1,
        )
        counts = np.bincount(bins, minlength=_COMB_BINS).astype(np.float64)
        counts[counts == 0] = 1.0
        profile = np.bincount(bins, weights=applied, minlength=_COMB_BINS) / counts
        applied = applied - (profile - np.mean(profile))[bins]

    if sweep_amount:
        # Composition point for the deterministic sweep-imposed phase - the
        # other half of the complex relationship, tabulated by the luma
        # amplitude stage. Parked: nothing wires a nonzero sweep_amount.
        block_sweep_phase = getattr(luma_amplitude, "block_sweep_phase", None)
        if block_sweep_phase is not None:
            psi = np.asarray(block_sweep_phase(model, demod), dtype=np.float64)
            applied = applied + sweep_amount * (
                np.gradient(psi) * (rf.freq_hz / (2.0 * np.pi))
            )

    demod -= applied

    debug_plot = getattr(rf, "debug_plot", None)
    if debug_plot and debug_plot.is_plot_requested("luma_noise"):
        return (applied / rf.SysParams["hz_ire"]).astype(np.float32)
    return None
