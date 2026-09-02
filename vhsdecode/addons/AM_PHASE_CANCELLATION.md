# AM-induced phase noise cancellation in FM video demodulation (`--head_switch`)

*Design record, 2026-09-01. Implementation: `vhsdecode/head_switch.py`; applied in
`process.py::_demodulate_to_video` immediately after `unwrap_hilbert`, before every
equalization stage. Measurement contract owned by `vhsdecode/luma_amplitude.py`.*

## The method

The luma FM carrier is recorded at constant amplitude. Whatever amplitude comes back
that the fitted playback model cannot explain is therefore the path misbehaving —
tape modulation noise, the per-head amplifier steps at head switching, dropout
collapses — measured as the log residual `d = ln A − ln(S·L·C)` against the
amplitude stage's expectation (decoder response `S`, fitted tape/head response `L`,
closed-form sweep collapse `C`).

A real amplitude disturbance on a carrier at `f_c`, seen through a channel `H`,
carries a phase partner determined by the channel's **asymmetry about the carrier**:

    Psi(f) = T(f) · D(f),
    T(f) = −j (H(fc+f)H*(fc) − H*(fc−f)H(fc)) / (H(fc+f)H*(fc) + H*(fc−f)H(fc))

A channel symmetric about the carrier converts amplitude to no phase at all; VHS
luma FM is quasi-single-sideband from record time, so the asymmetry — and the
coupling — is structural. The demodulator reports the phase's rate, fusing into one
spectral multiply `K(f) = j·f·T(f)` per RF block. The naive envelope-Hilbert form
(`Psi = H{d}`) was tried first and falsified (correlation +0.09 on a synthetic
causal chain); the kernel above, with the carrier phase reference, reaches +0.94..97
with near-unit gain, causally. `|T|` is capped at 1, the single-sideband limit.

Two calibrations make the kernel exact in practice:

1. **Absolute level.** `T` depends on where the carrier sits. The kernel is anchored
   on the **measured** sync reference levels the amplitude stage exports — the
   fitted 0 IRE and −40 IRE carrier frequencies and the measured Hz/IRE scale
   (the first measured deck records 5.8% shallower than the format's nominal
   deviation) — and made first-order carrier-dependent through a secant term:
   `p = irfft(K0·D) + Δfc(t)·irfft(K1·D)`, `K1` spanning sync tip to peak white,
   `Δfc` box-averaged over the model's own envelope support. Measured effect: plain
   full subtraction injected +183% porch noise; the anchored carrier term turned
   that into a −22% porch-noise removal (offline captures).

2. **Self-calibrated per-band gain.** How much of the prediction truly appears in
   the demodulated signal is measured against the decode itself:
   `W(f) = ⟨Re P*U⟩ / (⟨|P|²⟩ + ridge)`, accumulated per block. Physics: only
   *multiplicative* disturbances couple AM and PM coherently; additive channel
   noise has independent quadratures, contributes prediction power but no
   cross-power, and is automatically excluded — `W` settles at the Wiener-optimal
   shrinkage per band, with no assumed thresholds anywhere. Estimation hygiene,
   each element measured necessary: Hann-windowed estimation spectra; the
   line-harmonic bins (±2, the window's main lobe) and DC-side bins excluded from
   the regression and smoothing (deterministic line-locked content — picture and
   expectation error alike — must not vote; without this the low-band estimate grew
   without bound on a static pattern); smoothing over one line period's bin span;
   a zero-centred unit-prior ridge so `W` starts near zero and converges within a
   field; a final clip to [0, 1].

## What it removes, and what it cannot

Removed: the phase artifact of every amplitude disturbance the envelope witnesses,
to the measured coherent fraction — tape modulation noise (the dominant tape noise
mechanism in the literature), head-switch transients, dropout-edge glitches.

Not removable, by physics: the phase share of additive noise (independent
quadratures — invisible to the envelope), and pure inter-head phase jumps with no
amplitude signature (all-pass events leave no residual). The sync-witnessed floor
measurements put the honest per-sample envelope noise at 36–45 mNp (SP), strongly
low-pass (≈20–29 mNp below 1 MHz, <1 mNp above 5 MHz) — the calibrated `W` is
largest exactly where that spectrum lives.

## Measured results (zaroff test captures, SLV-778HF, 10-frame decodes, `-t 0`)

| Tape | Flat-field HF luma noise | Back-porch noise | Switch-line peak action |
|---|---|---|---|
| 75 bars SP | **−20.5%** (0.319 → 0.254 IRE) | **−1.8%** | 23 IRE |
| 75 bars EP | **−17.1%** (0.687 → 0.570 IRE) | +0.9% | 73 IRE |
| Chroma noise SP | (no flat columns for the metric) | **−2.7%** | 16 IRE |
| Bounce (flat field) SP | −3.1% | +0.6% | 50 IRE |

First fields are corrected gently (ridge ramp; first-2-fields A/B ≤ 0.2 IRE rms).
Rejected designs, by measurement: full ungated subtraction (−15.7% flats bought
with +183% raw porch injection); a line-comb on the residual (discards real
cancelable line-repeating content, −11% → −7%, and does not fix the estimator
bias); a block-local MAD gate (reads path structure as background and gates the
head-switch events away). Head-switch identification (`locate`) is separate from
the correction: the switch's sustained per-head offset is a DC feature the kernel
deliberately nulls, so it is located on the per-field deviation (median per line on
the field's own line locations; onset measured at line ~259–260, parity-alternating
±0.7–1.3 dB, 4–7σ per field) and drawn on the `luma_noise` debug plot.

## Prior art and the novelty claim

Feedforward cancellation of PM noise from *correlated* AM noise is established in
frequency metrology and photonics: NIST demonstrated ~20 dB oscillator PM-noise
reduction from correlated AM at >90% correlation ("Oscillator PM Noise Reduction
From Correlated AM Noise", tf.boulder.nist.gov/general/pdf/2789.pdf); balanced
optical-microwave phase detectors reach tens of dB of AM-to-PM suppression
(arxiv.org/pdf/1309.1116); patent families cover demodulate–process–remodulate
noise cancellation topologies (e.g. US 10,135,477). The magnetic-recording
literature identifies modulation noise — spurious AM from tape property variation —
as the dominant dynamic-SNR limit, but the compensation art there is limiters,
emphasis, and dropout gating; we did not locate any application of envelope-derived
phase-noise cancellation to FM video or magnetic-tape demodulation, nor any
model-based formulation in which the AM→PM transfer is *derived from the receiver's
own band-pass asymmetry about the carrier* (with the carrier phase reference) and
then per-band calibrated against the decode, rather than fitted as a black-box
correlation. The claim is therefore scoped: a novel application and formulation of
an established physical principle, to the search performed on 2026-09-01.

Related in-tree negative results: `luma_beat.py` records that the color-under's
AM crosstalk arrives in the demodulated luma as *frequency*, not phase, and that
naive envelope weighting/division did not help that correction — consistent with
this module's `j·f` (rate) kernel. That observation is also the flagged template
for removing the hifi audio carriers from the luma in a later stage.

## Limitations and accepted behaviors

- The kernel reads `RFVideo` only; an enabled FM-audio notch is upstream of the
  envelope but absent from `T` (second-order; out of scope this round).
- PAL is untested — no PAL captures exist in the test set; every scale in the
  implementation is table-driven (SysParams anchors, line period, filter tables),
  nothing NTSC-specific is encoded.
- Sync stays coupled by ruling: line detection benefits from the cleaned
  head-switch transient, so chroma A/B output differs when the flag is on
  (measured: rms ≈ 19 codes on 75 bars SP) — expected, documented behavior.
- The measured level axis is an *export* of the amplitude stage; that stage's own
  binning deliberately keeps its spec anchor. Nothing here changes that.
- Decode cost: ≈ +22% wall time at `-t 0` (six FFT-length transforms per block in
  the correction path). Single-threaded decodes are exactly reproducible; threaded
  decodes share the calibration accumulator (lost-update class, as the equalizer).

## Reproduction

    PYTHONPATH=/workspaces/vhs-decode python3 vhs-decode <capture> <out> \
        -n -f 50 -t 0 -l 10 --head_switch [--debug_plot luma_noise]

Offline instruments (session scratchpad): `hs_w_prototype.py` (estimator arms,
bias fingerprint, convergence), `hs_linefold.py` (line-locked discrimination),
`hs_noise_metric.py` (tbc-domain A/B), `hs_capture.py` (live-pipeline block
capture).
