# The spectral / harmonic arc — full design record

Opened Aug 12 2026. This is the active design thread and is expected to
run for several rounds. It supersedes the amplitude-law arc (steps 42-44,
all falsified — see §6). Companion documents: `RINGING_RULES.md` (standing
rules, read first), `RINGING_CONTEXT.md` (consolidated project state),
`/output/claude_validation/versions/NOTES.txt` (chronology).

---

## 1. The user's model, in their own terms

Stated across the session that opened this arc. Every clause is
load-bearing and each one changed the design:

1. **"The active level just before the front porch cutoff can be measured
   against the expected porch level. The expected porch level is 0 IRE."**
   The front porch is a *specified* level, so it is DC by definition. The
   content level immediately before the cutoff is known per line. The pair
   gives a transition whose start and end are both known without any model.
2. **"Any additional trailing frequencies away from that cutoff will be
   related to the horizontal smear."** Whatever non-DC content the porch
   carries is the channel's trailing response to the picture that preceded
   it. That is the definition of horizontal smear, measured against a
   reference level, inside blanking.
3. **"This may require a different averaging structure than the existing
   logic, for example if it's less noisy to just take the frequencies
   themselves."** Work in the frequency domain. Magnitude spectra are
   **alignment-invariant**, so no sub-sample crossing estimate is needed.
4. **"The slope will be derivable from the instantaneous frequency of the
   trailing active area."** The legitimate cutoff slope is not assumed and
   not taken from a filter spec — it is derived per line from how fast the
   signal is actually moving just before the transition.
5. **"Interleaving slopes will create the harmonic oscillations I am
   seeing as ringing."** The visible ringing is harmonic distortion
   produced by slew-limited (interleaving) slopes, not only a resonance
   excited by an edge.
6. **"I can visibly see the ringing and the settling, so it should be
   possible to model this much better than my eyes can."** The acceptance
   standard for the arc.

## 2. Why this instrument is better than everything before it

- **Alignment invariance.** Every time-domain estimator in this project
  averaged windows aligned on a sub-sample crossing. Crossing jitter
  scales as noise/slope, so it smears low-amplitude and slow transitions
  more than large fast ones. That bias *manufactures* amplitude trends.
  It is almost certainly why step 44's time-domain law came out
  super-linear while this instrument reads sub-linear on the same data.
- **No assumed ideal.** The porch's ideal is DC at the measured blanking
  level (rule 17: measured, never nominal).
- **No assumed slope.** Derived per line from the trailing content's
  instantaneous frequency (Hilbert analytic signal → d(phase)/dt).
- **Enormous statistics.** 10⁴–10⁵ line-ends per decode, power-averaged.

## 3. What has been measured so far (uncorrected decodes, 100 fields)

Geometry: NTSC 4fsc, porch = 21 samples → 0.68 MHz per bin, guard 3
samples each side, Hann window, spectra power-averaged.

**Trailing-content instantaneous frequency** (the legitimate slew):
- cd: median 0.99 MHz (p10 0.39, p90 1.56) → legitimate edge 7–18 samples
- home: median 0.64 MHz (p10 0.38, p90 1.11)

This alone killed my first attempt at the "legitimate" reference, which
used the decoder's 6.6 MHz luminance low-pass and was therefore about an
order of magnitude too fast.

**Porch residual spectra** (IRE per bin at 1.0 / 1.9 / 2.9 / 3.8 MHz):

| tape | transition | n | porch | legitimate slew |
|---|---|---|---|---|
| cd | 15.5 IRE | 18322 | 0.342 / 0.366 / 0.237 / 0.044 | 0.038 / 0.035 / 0.031 / 0.026 |
| cd | 26.0 | 5045 | 0.359 / 0.398 / 0.270 / 0.056 | 0.032 / 0.030 / 0.026 / 0.022 |
| cd | 47.6 | 255 | 0.423 / 0.469 / 0.287 / 0.066 | 0.020 / 0.019 / 0.017 / 0.014 |
| cd | 70.3 | 245 | 0.480 / 0.585 / 0.345 / 0.068 | 0.027 / 0.025 / 0.022 / 0.018 |
| home | 54.1 | 10220 | 0.467 / 0.442 / 0.312 / 0.090 | 0.066 / 0.062 / 0.054 / 0.045 |
| home | 65.0 | 13708 | 0.494 / 0.452 / 0.294 / 0.090 | 0.079 / 0.074 / 0.066 / 0.054 |

Findings:
- The porch carries **15–20× more energy at 1–3 MHz than the channel's own
  slew rate can justify**, and **nothing above ~4.8 MHz**.
- Excess over the low-amplitude floor grows with transition amplitude
  (cd at 1.0 MHz: 0.109 → 0.248 → 0.337 IRE for 26 → 48 → 70 IRE).
- The amplitude law is **sub-linear** here (0.34 → 0.60 IRE at 1.9 MHz
  while amplitude goes 15 → 70), the opposite of the time-domain result.
- **Supports the harmonic hypothesis:** trailing content runs at
  0.4–1.1 MHz while the porch residual peaks at ~1.9 MHz — roughly the
  2nd–5th harmonic of the driving slope.

## 4. THE EXPERIMENT HAS RUN — result and correction of §3

### 4.1 Does the porch's frequency track the drive? NO.

Stratified by the trailing content's instantaneous frequency
(n = 3000-7400 per band, five bands, both tapes):

| drive f_inst | cd centroid | home centroid |
|---|---|---|
| 0.41 / 0.42 MHz | 1.75 | 1.68 |
| 0.60 / 0.59 | 1.76 | 1.69 |
| 0.83 / 0.81 | 1.76 | 1.69 |
| 1.13 / 1.09 | 1.74 | 1.68 |
| 1.52 / 1.46 | 1.74 | 1.73 |

Slope: **-0.02 (cd) / +0.02 (home) MHz per MHz of drive.** Harmonic
distortion of the preceding content predicts +2 to +4. Over a 3.7x range
of drive frequency the porch does not move.

### 4.2 Is it a fixed SLEW RATE instead? NO.

A fixed slew rate makes the edge's duration (and so its harmonic content)
depend on amplitude. Stratified by transition amplitude, the centroid is
again fixed (cd 1.72->1.79 MHz over 14->36 IRE; home 1.72->1.65 over
54->79), and the live-band energy scales as **amplitude^0.17 (cd) and
^0.05 (home)** — i.e. essentially INDEPENDENT of transition amplitude. A
transition-excited component must scale with amplitude; most of what is
in the porch does not.

### 4.3 The control that corrects §3 (run late — it should have been first)

The same spectrum measured in the BACK porch, which follows the sync rise
rather than picture content:

| tape | after a >30 IRE picture transition | back-porch control |
|---|---|---|
| cd | 0.406 / 0.468 / 0.301, live rms 0.692 | 0.336 / 0.399 / 0.286, live rms 0.601 |
| home | 0.482 / 0.447 / 0.302, live rms 0.729 | 0.329 / 0.414 / 0.328, live rms 0.629 |

**So most of the porch's live-band energy is a floor present after any
edge.** Subtracting in power, the genuinely transition-excited part is
**~0.34 IRE rms (cd) and ~0.37 IRE rms (home)**, not the "15-20x more
than justified" figure quoted in §3 — that figure compared against a
legitimate-slew reference without subtracting the floor and is
**WITHDRAWN**. §3's table stands as raw measurement; its interpretation
does not.

### 4.4 What the arc actually established

The porch's trailing content sits at a **fixed ~1.7 MHz on both decks,
independent of drive frequency and of transition amplitude**, and the
same content follows the sync rise. That is the signature of a fixed
channel resonance — and 1.7 MHz matches each deck's own measured pole
(cd 1.54, home 1.87 MHz). The existing pole model's BASIS is therefore
right; the harmonic/slew mechanism is not what the porch sees.

The actionable gap is unchanged and now doubly confirmed: the model has
the right resonance but **does not apply it at picture transitions**
(§7: 2.23 IRE rms error vs 0.84 IRE applied, correlation -0.11, because
the level gate switches the settle/lring branches off there).

## 5. Superseded plan (kept for the record)

**Does the porch residual's peak frequency track the trailing content's
instantaneous frequency?**

- If the peak **moves with f_inst** → harmonic distortion from
  slew-limited slopes (the user's model). The correction is then a
  *predictable* function of the drive: harmonic amplitude and phase as a
  function of (f_inst, transition amplitude).
- If the peak **stays put** while f_inst varies → a fixed channel
  resonance, and the harmonic reading is coincidence.

Stratify lines by f_inst (not amplitude) and compare porch spectra. This
is the cleanest discriminator available and it needs no correction change.

## 5b. Where this connects to established physics

Slew-rate limiting is the time-domain face of the **record-side emphasis
clip** already established in this project (§3.2 of `RINGING_CONTEXT.md`:
dark clip pinned just below the sync-tip carrier). A slew-limited
transition generates harmonics of its own drive frequency, so the user's
"interleaving slopes" and the FM clip are the same mechanism seen from two
sides. This also connects to the **phase-locked 2× harmonic pairs**
catalogued on all three tapes (step 29, pnb 0.78→1.56 MHz, cd
0.62→1.25-1.32, home 0.79→1.58) which were measured but never shippable
because a single joint frequency mis-corrected content. A law keyed on the
*measured local f_inst* is exactly what that finding said was needed.

## 6. What is already falsified (do not re-attempt)

- **Per-line porch templates** (steps 36-42): imposed class-dependent
  level offsets → visible sporadic porch level shifting. Any realization
  must be strictly sum-zero and continuous in amplitude.
- **Ring amplitude law from the porch** (step 43): the ring and settling
  families cannot be separated over the porch's ~15 usable lags. cd's null
  control came out +3.25 ± 2.29 where it had to be ~0; home's regressor
  was a null column (its ring branch is empty — see below).
- **Landing transfer** (step 44, measured at the sites): applying
  porch-measured waveforms at picture landings left the mic-stand rod
  unchanged (train rms 2.27 → 2.24) and made the stage light worse
  (5.27 → 5.35). The settling polarity inverts at +15 IRE landings.
- **Two-anchor landing extrapolation** (step 44): sync tip (−40) and porch
  (0) predict a *positive* peak at +15 IRE where the stage light measures
  −15.7. Not linear, not learnable from two anchors.
- **Picture-amplitude ring extension** (step 42): ring_eye +0.12 med.
- **Continuous ring/settle blending** (step 43): the model EMA already
  smooths the family flip, and the blend cost the visible cd sites.

## 7. Structural facts to keep in mind

- **The two decks use different branches.** home is settle-dominated
  (ring branch rms 0.0002 IRE vs settle 0.2783) — its channel error is a
  slow recovery. cd is ring-dominated (0.30 vs 0.03).
- **The model does not treat picture transitions at all.** Measured at the
  porch: home 75 IRE falls carry a 2.23 IRE rms error while the runtime
  applies 0.84 IRE rms with correlation **−0.11** (subtracting it makes
  the residual worse). The level gate switches the settle/lring branches
  off at picture levels. This is the gap the user sees.
- **home's dominant 1.87 MHz pole sits 1% from the ring/settle threshold**
  and flips families in 58%/42% of fields. The blend fix was falsified,
  but the fragility is real and worth remembering when interpreting
  home's model.
- Only **~1% of picture falls land within 10 IRE of black** (cd median
  landing +20 IRE, home +51). The porch's landing regime is not the
  picture's.
- **pnb has no active-to-porch level difference**, so it populates no
  strata and every porch-derived mechanism is inert there — the
  graceful-degradation regression check.

## 8. Instruments

- `scratchpad/smear_spectrum.py` — the spectral porch instrument (§3).
- `scratchpad/stage0_law.py` — module-state replay with disk caching
  (`state_<tape>_<parity>_<fields>.pkl`); iterate estimators in seconds.
- `scratchpad/stage0b_joint.py` — joint settling+ringing gain estimator.
- `scratchpad/ex_landing.py` — offline application experiment (landing
  transfer test).
- Standing gauges: `tools/ringing_measure/` — pulse_trace (sync
  assertions), ring_eye (**the veto**; caught two over-reaches),
  micstand_gauge and stagelight_gauge (the two user-identified visible
  sites on cd), blurcheck (texture/weak-edge).
- Shipped build: `*_pc9`. Snapshots under
  `/output/claude_validation/versions/step4x-*`.

---

## 9. THE SWEEP DIMENSION — cross-spectral identification (Aug 12)

**The user's framing:** "This is another dimension of data... It will give
more of a sweep component to the impulse response that we are doing on the
fixed sync pulse frequency. Since we don't have a sweep in the sync
signal, i.e. it is only ever one transient width, we cannot reproduce the
component that a frequency sweep captures in the impulse response."

Exactly right, and it is a statement about identifiability: the sync edge
has ONE width, so its drive spectrum |D(f)|² is fixed and the channel is
identified only where that spectrum has energy. The picture's line-end
transitions span 0.4-1.6 MHz of instantaneous frequency (measured), so
their pooled drive spectrum covers the band - the sweep.

### 9.1 Why the earlier spectral attempts failed (the mathematics)

Power averaging does not remove noise. For measurement r = s + v with
noise v uncorrelated with signal s,

    E[|R|²] = |S|² + E[|V|²]

so averaging |R|² converges to signal power PLUS noise power - the floor
never averages away. That is why every stratification came out flat: the
porch's ~0.6-0.7 IRE live-band "energy" was dominated by the per-line
noise floor (0.75-0.85 IRE/line, measured by pulse_trace). Time-domain
aligned averaging does suppress noise (1/sqrt(n)) but reintroduces the
crossing-jitter bias. The estimator that has both properties is the
CROSS-spectrum, because

    E[R conj(D)] = H·E[|D|²] + E[V conj(D)] = H·E[|D|²]

the noise term vanishing as 1/sqrt(n) since v is uncorrelated with the
drive d. Drive and response come from the SAME line, so their relative
timing is exact and no cross-line alignment is needed.

### 9.2 The estimator, its bias and its variance

    H1(f) = S_rd / S_dd            (used here)
    H2(f) = S_rr / S_dr            (upper bound; input-noise-immune)
    gamma²(f) = |S_rd|² / (S_dd·S_rr)      coherence, 0..1

- H1 is biased LOW by noise on the DRIVE (errors-in-variables); H2 is
  biased high by noise on the response. The unbiased estimate lies
  between them (total-least-squares / Hv). Reporting both bounds the bias
  without any assumption.
- Coherence is the fraction of the response the drive explains, and it is
  the SNR in disguise: SNR = gamma²/(1-gamma²).
- Variance of the estimate: Var(|H|)/|H|² ≈ (1-gamma²)/(2·n·gamma²).
  With n = 24184 and gamma² ≈ 0.46 (home), that is ≈ 2.4e-5 → |H| is
  measured to ~0.5%. This is the first estimator in the arc with a
  quantified error bar that small.

### 9.3 First measurement (uncorrected decodes, 100 fields, N=64 window)

home (n=24184 line-ends):

| f MHz | 0.22 | 0.45 | 0.67 | 0.89 | 1.12 | 1.34 | 1.57 | 1.79 | 2.01 | 2.24 | 2.46 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| \|H\| | .035 | .044 | .055 | .068 | .077 | .081 | .086 | .100 | .124 | .138 | .150 |
| phase° | -14 | -41 | -79 | -120 | -160 | 159 | 117 | 77 | 38 | 2 | -33 |
| gamma² | .30 | .40 | .46 | .49 | .49 | .47 | .46 | .46 | .49 | .46 | .44 |

cd (n=24199): |H| 0.057 → 0.026 → 0.053 (dish-shaped), gamma² 0.20 → 0.03
→ 0.05 — an order of magnitude less coherent, because cd's line ends are
dark and low-contrast, so its drive spectrum is weak.

Two immediate readings:
- **home's response RISES with frequency** (0.035 → 0.150, a factor 4.3
  across 0.2-2.5 MHz): the trailing smear is a high-frequency-favouring
  transfer, not a resonance peak.
- **The phase rotates almost linearly** (≈ -169°/MHz ⇒ group delay
  ≈ 0.47 µs ≈ 6.7 samples), which is mostly the fixed offset between the
  drive window and the response window in this construction. The
  physically meaningful quantity is the DEVIATION from that linear phase
  — the dispersion — which must be extracted before any realization.

### 9.4 The mathematics of turning H into a correction

1. **Separate delay from dispersion.** Fit and remove the linear-phase
   term (group delay tau from the slope of unwrapped phase). What remains
   is the dispersive part; a pure delay must never be "corrected" (it is
   just window geometry here, and horizontal position in general).
2. **Minimum-phase / all-pass factorisation.** H = H_min · H_ap, where
   H_min is recovered from the magnitude alone by the Hilbert transform of
   the log-spectrum (cepstral method): arg H_min(f) = -Hilbert{ln|H(f)|}.
   The residual H_ap = H / H_min is all-pass and carries the true
   group-delay distortion. This split matters because RINGING_RULES #3
   allows phase correction and narrow measured-resonance damping but
   forbids broadband magnitude damping (which is blur). Correcting H_ap
   only is provably blur-free: |H_ap| = 1 at every frequency.
3. **Regularised inversion (Wiener).** G = conj(H)/(|H|² + lambda), with
   lambda set from the measured coherence, lambda(f) = (1-gamma²)/gamma².
   This is the Wiener-optimal inverse and it automatically stops
   correcting where the measurement is not trustworthy - no hand-tuned
   band limits. (Note: the project's earlier "wiener era" used a
   SYNC-derived spectrum; the novelty here is that H comes from picture
   transitions and therefore has the sweep dimension.)
4. **Parametric realisation.** Fit a low-order pole-zero (ARMA) model to
   the de-delayed H by Steiglitz-McBride or a Prony/matrix-pencil fit on
   its impulse response, then realise the inverse as a recursive filter.
   This matches the module's existing pole machinery and yields a compact,
   stable, causal correction rather than a long FIR.
5. **Causality.** Any realisation must be causal (rule 2). A minimum-phase
   inverse is causal and stable by construction; an all-pass inverse needs
   a delay, which is acceptable only if applied uniformly to the whole
   line (a constant shift is invisible), never per-transition.
6. **The nonlinear caveat.** The channel is level-dependent (record clip),
   so a single LTI H is a linearisation about the operating point. The
   principled extension is the Hammerstein form the module already uses
   (static nonlinearity then LTI) - identify H separately per landing-level
   or per-amplitude class and let the class index carry the nonlinearity,
   rather than pretending one H covers all.

### 9.5 Implementation candidates, in order of risk

- **(A) Coherence-weighted all-pass (phase-only) equaliser.** Correct only
  H_ap, weighted by gamma². Blur-free by construction, self-limiting where
  the data is poor (cd would receive almost nothing, home a real
  correction). Lowest risk, directly permitted by rule 3.
- **(B) Regularised Wiener inverse** with lambda from coherence. Corrects
  magnitude too; must be checked against blurcheck's weak-edge column.
- **(C) ARMA fit + recursive realisation**, folded into the existing
  train machinery as an additional measured branch.
- **(D) Per-class H** (amplitude and/or landing-stratified) to carry the
  nonlinearity - the Hammerstein extension, once (A) or (B) is validated.

### 9.6 Immediate next steps

1. Rebuild the cross-spectrum with a clean window geometry: separate
   drive and response segments (no shared window), Welch-style averaging,
   and both H1 and H2 so the bias is bracketed.
2. Remove the linear-phase term; report the dispersion and the
   minimum-phase/all-pass split.
3. Realise (A), decode cd and home, and put it in front of the eye with
   the standard battery (ring_eye as veto, blurcheck weak-edge as the
   null test, pulse_trace, micstand/stagelight).

## 10. Deck stability, and the VCR / TV source distinction (Aug 12)

**User:** the resonance "is possible that the frequency changed a bit over
time. The home camera was outside in cold weather and battery powered.
Also, cd is recorded from television and may have different distortion
models... I want to primarily focus on VCR based distortion models, but if
TV models do make sense, do implement them."

Measured: dominant residual component per 20-field group, from each tape's
own sync edges (matrix pencil on the aligned mean, no correction in the
loop), 4500 pulses per group, 200 fields:

| fields | cd dominant | cd tau | home dominant | home tau |
|---|---|---|---|---|
| 16-35 | 1.577 MHz | 9.7 | 0.000 (pure decay) | 13.8 |
| 36-55 | 1.578 | 9.8 | 0.000 | 10.4 |
| 56-75 | 1.587 | 10.4 | 0.000 | 11.8 |
| 76-95 | 1.604 | 10.7 | 0.000 | 14.2 |
| 96-115 | 1.597 | 10.0 | 0.000 | 14.4 |
| 116-135 | 1.584 | 10.0 | 0.000 | 12.7 |
| 136-155 | 1.592 | 10.0 | 2.294 | 5.8 |
| 176-195 | 1.602 | 10.1 | 0.063 | 10.0 |

Readings:

- **cd's resonance is stable to ~1%** (1.577-1.604 MHz, tau ~10 samples
  throughout). No meaningful drift over 200 fields. So for cd a *fixed*
  identified resonance is a sound model, and the existing EMA horizon is
  more than adequate.
- **home has no stable resonance to drift.** Its dominant component is a
  PURE DECAY (0.000 MHz, tau 10-14 samples ≈ 0.7-1.0 us) in seven of nine
  groups - a settling/recovery term, not an oscillation. This is the same
  conclusion reached three other ways (§7: ring branch rms 0.0002 vs
  settle 0.2783; the ring/settle family flip at rotation 3.17 vs pi; the
  null regressor in step 43).
- So the two tapes are **different mechanisms, not different amounts of
  the same one**: cd = a genuine damped resonance; home = a relaxation.
  Any correction must therefore be per-deck by construction (the module
  already identifies per tape), and a single "ringing model" will never
  serve both.

**On sources.** cd is off-air, so its chain is broadcast + VCR; home is
camera-original, so its chain is VCR only. The measurement above is
consistent with that split: the stable narrow 1.58 MHz resonance on cd
looks like fixed electronics (either the broadcast chain or the deck's
own video amplifier), while home's relaxation is the emphasis/clip
recovery this project has characterised as the record-side mechanism.
**Design decision (per the user): model the VCR mechanisms primarily.**
The home relaxation is therefore the priority target, and cd's narrow
resonance is treated as a fixed identified resonance which the existing
pole machinery already handles correctly - it needs no new physics, only
correct *application* (§4.4 / §7: the gate problem).

Practical consequence for the sweep work in §9: cd's low cross-spectral
coherence (0.02-0.20) is now explained - its line ends are dark and
low-contrast so the drive is weak, AND its dominant defect is a narrow
resonance that a broadband transfer estimate spreads thin. home, whose
coherence is 0.30-0.49, is the tape where the sweep transfer function is
worth realising first, and its defect (a relaxation with a rising
high-frequency transfer, |H| 0.035 -> 0.150) is exactly what an all-pass
/ dispersion correction addresses.

## 11. THE PORCH DECONVOLUTION — realised, and what it corrected (Aug 12)

The sweep dimension was realised, but not as §9.5 predicted. Three of
that section's expectations turned out to be wrong, and the reasons are
recorded here because they are structural, not incidental.

### 11.1 The estimator that finally works

On the front porch the ideal signal is a CONSTANT (flat at blanking, by
specification). So on the porch

    y[m] = ideal + sum_k h[k] x[m-k]

and the drive x is the ACTIVE PICTURE alone: wherever a lag lands on the
porch the ideal is that same constant, which a per-line intercept
absorbs, so those terms vanish identically. Consequences:

- No porch sample is ever a regressor, so the fit cannot degenerate into
  whitening the decoder's own noise correlation (which would inverse
  filter the luminance low-pass and amplify noise). This is the hazard
  that makes naive "flatten the porch" deconvolution fail.
- Drive and response come from the SAME line: relative timing is exact,
  so no crossing-jitter bias, and no power-averaging noise floor.
- Using observed active samples for the unobservable ideal is
  errors-in-variables, which biases |h| LOW. The correction therefore
  UNDER-corrects; it cannot run away.

The correction follows to first order: h is small, so c = delta - h,
with sum(h) = 0 enforced (unit DC gain, rule 6) and h starting at a
positive lag (causal, rule 2).

Measured (pc9 decodes, 120 fields, ~29k line ends, lags 1..36):
split-half tail agreement +0.98 (cd) / +0.96 (home); held-out porch
flatness 0.875 -> 0.772 IRE (cd) and 0.960 -> 0.864 (home).

### 11.2 FALSIFIED: the minimum-phase / all-pass split (§9.4-9.5)

Candidate (A), the coherence-weighted all-pass equaliser, is not merely
unhelpful - it is VACUOUS. Measured: the all-pass factor changes porch
flatness by -0.0%, exactly. The reason is a theorem, not a data
accident: for c = delta - h with ||h||_1 << 1 (measured 0.08), Rouche's
theorem puts every zero of C(z) strictly inside the unit circle, so c is
minimum-phase BY CONSTRUCTION and its all-pass factor is identically 1.
Group delay confirms it: <=1.3 samples on cd, <=0.4 on home.

**A weakly perturbing channel has no dispersion to correct.** The
magnitude and phase are Hilbert-linked and the split carries no
information. Do not re-attempt the all-pass realisation at this
perturbation level.

The same bound ||h||_1 < 1 is now used as the shipped VALIDITY GATE: it
is the condition under which c = delta - h inverts g = delta + h at all,
and simultaneously the condition making the realisation stable.

### 11.3 The resonance / relaxation split, and why it is mandatory

The identified tail mixes two mechanisms:

- a narrow damped RESONANCE - the channel's linear part, level
  independent; rule 10 explicitly permits linear resonances to
  extrapolate;
- a broadband RELAXATION - the record-side clip recovery, level
  dependent; rule 10 forbids extrapolating it beyond the witnessed
  regime.

The porch witnesses ONE landing level: blanking. Shipping the whole tail
therefore extrapolates exactly the component that must not be. The
damage was measured before the split was applied: mic-stand foot +9.0 ->
+10.2 (its landing is ~+15 IRE and its foot has the OPPOSITE SIGN to the
porch's undershoot - the level dependence, visible directly), stage light
5.27 -> 5.49 rms, and ring_eye GHOST +0.34 med / +0.89 p90. Restricting
to the resonant modes reversed all three.

Mode decomposition uses the module's own matrix pencil and its own
family criterion (rotation >= pi within the decay time).

| tape | modes found | verdict |
|---|---|---|
| cd | 1.87 MHz tau 6.8 (Q 2.8) RESONANCE; 3.14 MHz tau 1.6; 0.40 MHz tau 18.2 | resonance kept |
| home | 0.000 MHz (pure decay), 1.056 MHz rot 2.42, 0.152 MHz rot 1.15 | ALL relaxations -> identity |
| pnb | 2.69 MHz tau 23.8 (Q 14) | fails ||h||_1 < 1 (1.745) -> identity |

**home gets no equaliser, and that is the finding, not a failure.** Its
defect is entirely the level-dependent relaxation, which blanking-only
landings do not license applying to the picture. The VCR mechanism needs
the Hammerstein (per-landing-level) treatment - §9.4 step 6, candidate
(D) - not a linear filter.

cd's resonance is now confirmed by THREE independent instruments:
matrix pencil on sync edges 1.577-1.604 MHz (§10), the porch inverse's
notch at 1.65 MHz, and the two field-parity halves independently at
1.87 and 1.88 MHz.

### 11.4 The shipped acceptance rule (no tunable thresholds)

1. a mode must be found INDEPENDENTLY BY BOTH field-parity halves, at
   frequencies agreeing within its own half-bandwidth 1/(2 pi tau) - a
   property of the resonance under test, not a chosen tolerance;
2. the kernel fitted on either half must reduce the other half's error,
   both directions;
3. ||h||_1 < 1 (§11.2).

Two construction errors found and fixed along the way:
- projecting to zero-sum AFTER the constrained solve leaves the answer
  off the optimum - at one ridge it fitted WORSE than doing nothing
  (0.9319 against a 0.8884 baseline). The constraint must live in the
  BASIS (each mode column has its own mean removed).
- pencil modes are NOT orthogonal, so keeping a subset while inheriting
  amplitudes fitted alongside the discarded modes is not a projection:
  the kept part came out LARGER than the whole tail (L1 0.359 vs 0.273).
  Amplitudes must be re-fitted against the original normal equations.

### 11.5 Gauge audit — two measurement errors caught (rule 17)

- ring_eye's "sharp" is max|diff| per row THEN averaged. A maximum of
  noisy differences is biased upward by noise, so any filter that
  removes noise lowers it without touching the edge. Checked: the
  noise-robust averaged-profile version agrees (x0.931 vs x0.934), so
  here the reading was real - but the statistic must not be trusted
  unchecked.
- a 10-90 rise width computed on integer sample indices quantises to
  whole samples and turned a small shift into an apparent 3.00 -> 4.00
  jump (x1.333). Interpolated, the real change is 3.75 -> 3.97 (+6%).

### 11.6 What the correction does, measured against an ideal step

The channel BOOSTS narrow features by 14-21% (bar widths 2-40 samples);
channel+correction returns them to x1.002-1.017. On an ideal
band-limited step the correction alone costs 0.3% of peak slope and
NARROWS the 10-90 width by 0.5%. So the contrast reduction the gauges
report is removal of the channel's own resonant peaking, not blur - and
being errors-in-variables biased low, it is conservative.

The residual honest cost: on the tape's REAL edges (10-90 width 3.75
samples, i.e. transition energy sitting right at the 1.87 MHz notch) the
rise width grows 6% and the averaged peak step falls 4%.

### 11.7 cd scorecard (offline, applied to pc9)

| | RING med | RING p90 | GHOST med | GHOST p90 | noise | rise width |
|---|---|---|---|---|---|---|
| pc9 | 1.95 | 5.99 | 5.02 | 12.33 | 0.180 | 3.75 |
| full tail | 1.69 | 5.80 | 5.34 | 13.09 | 0.174 | 3.91 |
| resonances | 1.58 | 5.49 | 4.87 | 11.92 | 0.168 | 3.88 |
| SHIPPED (confirmed mode, re-fitted) | 1.73 | **4.93** | 4.96 | 12.35 | **0.147** | 3.97 |

Ring p90 -18%, ghost neutral, blanking noise -18.5%, rise width +6%.

### 11.8 IN-DECODER REALISATION: REGRESSED, AND WHY (application is OFF)

The identification of §11.1-11.7 is validated. The in-decoder
APPLICATION is not, and is currently disabled (`PEQ_APPLY = False`).
Decoded, full battery:

| | RING med | RING p90 | GHOST med | GHOST p90 | sharp | pulse_trace |
|---|---|---|---|---|---|---|
| cd_pc9 | 1.95 | 5.99 | 5.02 | 12.33 | - | all PASS |
| cd_eqrule (offline, on the .tbc) | 1.73 | **4.93** | 4.96 | 12.35 | 0.951 | **all PASS**, undershoot -0.94 -> -0.35, noise 0.76 -> 0.70 |
| cd_pc11 (in-decoder) | 1.94 | 5.94 | 5.47 | 13.22 | 1.054 | **back porch FAIL, rise overshoot FAIL** |
| home_pc9 | 1.31 | 2.89 | 2.33 | 7.57 | - | all PASS |
| home_pc11 (in-decoder) | 1.18 | 2.93 | 2.10 | 7.89 | 1.029 | **front porch FAIL, sync tip FAIL** |

**Two mechanisms, both specific and both fixable.**

1. **The application point interacts with `--ire0_adjust`.** Part 5 runs
   before `hz_to_output`, which re-anchors black per field from the
   porch. The equalizer disturbs the back porch (max|dev| 0.55 -> 1.48),
   the anchor moves with it, and the whole field's levels shift: front
   porch -1.26 -> -3.58 IRE on cd, -0.51 -> -2.92 on home. **Unit DC
   gain is not sufficient for level neutrality when a downstream stage
   re-anchors from a region the correction touches** - rule 6 has to be
   read as a property of the whole chain, not of the kernel alone.
   NOTE: a kernel-length explanation ("36 taps is longer than the
   21-sample front porch") was proposed and is FALSE - the same kernel
   length applied to a finished .tbc leaves the front porch at -1.27 and
   passes every assertion.
2. **The identification domain changes the mode set.** In-decoder
   (normalized buffer, pre-output) both halves confirm TWO modes,
   1.81 MHz AND 2.87 MHz, L1 0.62. On the finished .tbc only ONE mode
   is confirmed, 1.87 MHz, L1 0.37. The extra mode sits in the detail
   band, which is why in-decoder sharpness went UP (x1.054) while ring
   did not improve. Whatever removes/alters 2.87 MHz between the two
   domains has to be identified before the pre-output domain can be
   trusted for this measurement.

**Before switching `PEQ_APPLY` back on:** apply after the level
anchoring (or exclude the anchor window from the correction, the way
BOUNDARY_RAMP already confines the blanking-sourced trains), and
resolve which domain the identification belongs in.

**Also fixed this round** (real bugs, independent of the above):
alternating fields report one line more than the buffer carries, so a
strict bounds check skipped every other field - which additionally
pinned the parity split to a single half so the two-halves gate could
never pass. Parity now comes from the accumulator's own counter.

## 12. THE AMPLITUDE LAW — MEASURED, AND FALSIFIED (Aug 12)

**User's proposal:** "use the amplitude of the undershoot on the front
porch to determine the magnitude of correction. This should follow some
curve, but perhaps the curve needs to be polyfit" / "or interpolated
using a spline". Later, crucially: "the difference in width caused by
amplitude is a key part of information to use when modeling this."

Measured by four independent estimators plus four adversarial controls
(13 agents, 2.06M tokens). **RESULT: g(A) = 1.000 for all A on all
tapes. The scalar amplitude gain law does not exist. Do not ship it.**

### 12.1 How it dies

- **cd fails out of sample, 4 of 4.** Train on half the recording,
  predict the other half: linear-only 0.3208 -> linear+law 0.3353 IRE.
  The law makes prediction 2.5-5.3% WORSE in every direction tested.
- **cd's "knee" sits in an empty gap.** Requiring a settled plateau
  (active IQR < 2 IRE) leaves NO DATA between A = 49.5 and 78.9 IRE.
  The claimed sign change at A ~ 59-65 is inside it. Only 7% of lines
  exceed A=45 and they are essentially one scene (1692 of them in a
  single 23-field chunk).
- **home fails an ACAUSAL control.** Regress the porch on a FUTURE
  line's amplitude (L+4, 254 us ahead): b_causal = +0.02032 +- 0.00269,
  b_acausal = +0.02196 +- 0.00265. **The acausal coefficient is
  LARGER.** home's pedestal-immune causal gain is -0.0024 +- 0.0031,
  i.e. zero. 79% of its apparent slope is a reference-frame choice
  (per-line back-porch pedestal vs global).
- **Extremum estimators are 92% bias.** Porch noise sigma rises
  0.83 -> 1.72 IRE (2.07x) with amplitude, so E[-min] over ZERO-MEAN
  residuals rises 1.173 -> 2.841 IRE with no signal present at all.
- **The quadratic form was collinearity, not physics.** corr(A, A^2) =
  0.97-0.99 natively, 0.998 in the fitted window; |c|A/|b| lands on
  0.418-0.437 across four datasets with different physics. That is the
  design matrix.
- **Polarity is untested.** All 145200 line ends are bright->dark
  (cd min A = +2.55 IRE). The one direct even-order measurement (pnb's
  pillarbox rise/fall pair at matched |S|) gives c/|b| = 0.00305/IRE
  against the 0.0116-0.0171 the quadratic laws require - 4-6x too small.

### 12.2 THREE METHOD ERRORS THAT INVALIDATE EARLIER NUMBERS

1. **FIELD-PARITY SPLIT-HALF IS THE TWO HEADS, NOT NOISE.** Alternate
   fields come from different helical-scan heads. Amplitude-matched
   even-odd porch shape has rms 0.086-0.097 IRE against a 0.004 IRE
   noise expectation at n~44000. **Region and time splits are 10-30x
   larger than parity splits** (cd U_tail at A=78.3: parity 0.78,
   region 3.63, time 3.71 IRE). Every split-half figure in this
   document from before section 12 is understated by ~10x, including
   the +0.98/+0.96 tail agreements in section 11.1 and the module's own
   two-halves confirmation gate. A parity split remains useful as a
   HEAD-INDEPENDENCE test; it is NOT a reproducibility or noise test.
2. **COLS 889-893 ARE THE FALLING EDGE, NOT PORCH.** home sits at
   +18.3 IRE at col 889 at A=68, landing at col 891 (low A) to 894
   (high A). Every "settling tail super-linearity" measured in that
   window is edge slew. The settled porch is cols 894-906.
3. **THE LAW DRIFTS 60-100% OF ITS OWN SIZE ACROSS A RECORDING.**
   Nothing measured here may be fitted once and frozen.

### 12.3 WHAT SURVIVES: the user's width insight

The one adversarial check that came back CLEAN was the spectral
confound - the amplitude effect is NOT explained by the drive spectrum
(A survives at 93-102% under 8 different drive descriptors). And what
every instrument agrees on is an amplitude-dependent **DELAY /
BANDWIDTH change, not a gain change**:

- cd's fall widens 0.348 -> 0.588 us (1.7x) as A goes 12.8 -> 78.2 IRE;
  width ~ A^+0.171, peak slope ~ A^+0.835 (bandwidth-limited predicts
  A^0.00 / A^+1.00; slew-limited predicts A^+1.00 / A^0.00). So it is
  NEITHER pure case: a real, reproducible partial compression.
- home's relaxation hump holds CONSTANT amplitude (~1.7-1.9 IRE) while
  shifting LATER with amplitude (peak col 895 -> 896 -> 896.5 across
  A = 52 -> 68). Its matched-filter ladder crosses zero at A~59 - a
  fixed template under-reading a SHIFTED waveform, which is exactly a
  delay misread as gain loss.

**This also explains why U/A must fall with amplitude even for a
perfectly linear channel:** a wider edge carries far less energy at the
1.87 MHz resonance, and a convolution already accounts for that. Any
g(A) fitted on top of it double-counts the same physics.

### 12.4 The porch's structure budget (corrected geometry, cols 894-906)

| tape | position-locked | amplitude-driven | held-out gain: linear A | A^2 |
|---|---|---|---|---|
| cd | 0.734 IRE (77%) | 0.223 IRE (23%) | 1.7% | 2.6% |
| home | 1.274 IRE (53%) | 1.119 IRE (47%) | 1.2% | 0.2% |

Per-line noise is 0.75-0.85 IRE, i.e. comparable to the whole structure.

**IDENTIFIABILITY LIMIT (why the obvious fix is not available).** A
per-column intercept cannot distinguish a position-locked artefact from
the response to the MEAN transition, because nearly every line ends with
a similar-sized fall. So the 77% cannot be attributed, and correcting it
with a fixed template would re-introduce the same over/under-correction
in mirror image. **The front porch alone cannot separate them.**
Separating them needs a second landing level or a second transition
geometry - see the rule-10 landing-level test.

### 12.5 Standing conclusion

The step-48 porch equaliser earned its 12% porch-flatness gain by using
a CONVOLUTION (which scales with the drive) to remove structure that is
largely NOT drive-scaled. That is a structural mismatch between the
correction's form and the defect's form, and it is the mechanism behind
the user's observation that ring lobes read darker at bright edges. The
fix is not a gain law. `PEQ_APPLY` stays False.

## 13. SHIPPED: the sync-confirmed porch equalizer (Aug 12)

The porch-identified channel equalizer is now ACTIVE (`PEQ_APPLY = True`)
after two fixes, each derived from a measured failure.

### 13.1 Fix 1 - active-picture confinement

The kernel has unit DC gain, so it cannot move a level by itself. But
the decoder re-anchors black per field from the back porch
(`--ire0_adjust`), so ANY structure the equalizer leaves there drags the
whole field with it - measured in step 48: back porch max|dev| 0.55 ->
1.48 and front porch -1.26 -> -3.58 IRE. **Level neutrality (rule 6) is
a property of the whole chain, not of the kernel alone.**

The equalizer's change is therefore confined to the ACTIVE PICTURE and
faded across both blanking boundaries with the same BOUNDARY_RAMP the
blanking-sourced trains use. The blanking interval keeps its own
sync-derived correction (PARTS 3-4). The ramp must reach zero BEFORE the
front porch, not inside it: a gated correction is no longer sum-zero
across a window it is cut through, and cutting through the porch moved
its level -1.26 -> -1.66 IRE while improving its flatness.

### 13.2 Fix 2 - CROSS-INSTRUMENT CONFIRMATION (the decisive one)

Confinement alone did not help: the picture still regressed (ghost
+0.47/+1.25, sharp x1.051, stage light 5.27 -> 5.93). Cause: the
pre-output domain reproducibly yields a **2.87 MHz mode** that the
finished-output domain rejects. It sits in the detail band, so damping
it boosts detail (sharp > 1) and worsens ghosting. A time split instead
of a parity split did not reject it either - it is genuinely
reproducible, just domain-specific.

**A porch-identified mode is now accepted only if the SYNC EDGES' own
matrix pencil also found it**, within the mode's own half-bandwidth.
Two instruments, different regions of the line, different transients,
same channel: a resonance both see is a property of the channel; one
only the porch sees is a property of the porch measurement. cd's
resonance is confirmed at 1.577-1.604 MHz by the sync pencil; the
2.87 MHz mode is not, and is now rejected.

This is also what makes the correction sync-derived end to end: the
sync pulse supplies the confirmation, the porch supplies the broadband
drive (the sweep) that the single-width sync edge cannot.

### 13.3 Measured result

cd (200 fields, `-s 500 -l 100 -n -f 40 -t 3 --dctp --ire0_adjust
--lti_gain 0`):

| | RING med | RING p90 | GHOST med | GHOST p90 | sharp | pulse_trace |
|---|---|---|---|---|---|---|
| pc9 baseline | 1.95 | 5.99 | 5.02 | 12.33 | - | all PASS |
| pc12 confined only | 1.94 | 5.94 | 5.47 | 13.22 | 1.053 | all PASS |
| pc13 + time split | 1.89 | 5.98 | 5.48 | 13.16 | 1.051 | all PASS |
| **pc14 + sync confirmation** | **1.80** | **4.85** | 5.01 | 12.73 | 0.960 | **all PASS** |

- **RING p90 5.99 -> 4.85, -19%**, ghost median neutral (5.02 -> 5.01).
- Every blanking measure BIT-IDENTICAL to pc9 (front porch -1.26 /
  1.04 / 0.54, tip -38.35 / 0.93 / 0.52, back porch -2.26 / 0.55 / 0.27,
  amp 36.08) - the confinement is exact.
- Matches the offline-validated kernel (1.73 / 4.93) closely, so the
  in-decoder path now agrees with the bench result.
- Sharpness: averaged peak step x0.970, 10-90 rise x1.054. Per section
  11.6 this is removal of the channel's own resonant peaking, not blur:
  the channel boosts narrow features 14-21% and the correction returns
  them to x1.002-1.017, while costing only 0.3% of peak slope on an
  ideal band-limited step. Blanking noise x1.000.
- Sites are mixed: mic stand rms 2.27 -> 2.31 with the ring lobes
  reduced (+2.0 peak -> +1.4) but a new -0.8 region at lags 22-27;
  stage light rms 5.27 -> 5.54 (its rms is dominated by the settling
  foot, not the ring band, and its event population changed 664 -> 742).
  THE EYE IS THE AUTHORITY on whether this trade is right.

home: **BIT-IDENTICAL to home_pc9** (every ring_eye delta +0.00, sharp
x1.000, pulse_trace identical). It has no sync-confirmed porch
resonance, so it correctly receives nothing. Graceful degradation is
structural, not tuned.

## 14. RULE 10 TESTED: UPHELD, BUT RESTATED (Aug 12)

**User:** "I'm not sure how solid rule 10 is, if it can be disproven that's
fine" / "I think it can be modeled".

Tested with 7 agents on the only design blanking allows: sync fall (lands
-40 IRE), sync rise (lands 0, rising), active->porch fall (lands 0,
falling), with the porch falls restricted to the measured sync depth
+-6 IRE so amplitude and polarity are matched.

### 14.1 A NEW INSTRUMENT: the vertical blanking interval

Lines 1,2,6,7,8 (equalizing pulses), 3,4,5 (broad/vertical-sync) and
21-262 (normal) put THE SAME TRANSITION AT UP TO FIVE HORIZONTAL
POSITIONS with landing, polarity and amplitude fixed: falls-to-sync at
cols 1 and 455; rises-to-blanking at cols 34, 67, 388, 489, 841.

**This is the first time in the project that POSITION has been free of
LANDING.** Note these pulses are excluded from CALIBRATION by rule 1;
using them to measure a confound is diagnostic, and must stay diagnostic.

### 14.2 Three controls, three independent kills

1. **MEASURING SYNC-REGION RESPONSE ON A CORRECTED DECODE IS CIRCULAR.**
   pc9 has the module's completion template subtracted over
   COMPLETION_LAG_LO=2 .. COMPLETION_LAG_HI=48 after the sync crossing -
   EXACTLY the lags the response is read at. Correction footprint
   (pc9 minus uncorrected, field-aligned): **1.04 IRE mean / 1.90 max at
   the sync tip, 3.74 IRE at the sync fall, against 0.23 IRE on the front
   porch.** The footprint at the tip is 2-3x LARGER than the claimed
   0.45-0.56 IRE landing effect. Re-run on UNCORRECTED decodes the ring
   ratio T3/T1 collapses **5.08 -> 1.16**.
   => ALL FUTURE SYNC-REGION MEASUREMENT MUST USE UNCORRECTED DECODES.
   (The front porch's 0.23 IRE footprint is why the section 13 equalizer
   identification is not affected; its sync cross-check is taken in PART 1
   from the uncorrected buffer.)
2. **LANDING IS ALGEBRAICALLY INSEPARABLE FROM OPERATING POINT.**
   start = landing + signed amplitude, so matching amplitude forces
   corr(start, landing) = +1 EXACTLY. **Matching amplitude does not
   remove the confound - it CREATES it.** Polarity = sign(start-landing)
   is not independent either. The design over {T1,T2,T3} is saturated:
   rank 3, residual dof 0, so any affine f(start,landing) fits all three
   with zero residual and zero predictive content.
3. **IN BLANKING, LANDING IS A DETERMINISTIC FUNCTION OF POSITION.**
   Landing -40 occurs only in the sync tip (cols 7-62), landing 0 only in
   the porches. Singular, not merely confounded.

### 14.3 The composition test - the decisive one

cd, matched amplitude, detrended settled response, IRE rms:

| cell | value |
|---|---|
| landing changed only | 0.670 |
| polarity changed only | 0.596 |
| **BOTH changed** | **0.346** |

Two independent axes require sqrt(0.670^2 + 0.596^2) = 0.897. Perfect
cancellation requires |0.670 - 0.596| = 0.074. The observed 0.346 is
inconsistent with BOTH. home reproduces it (sync-vs-sync 0.498 while
every site-vs-porch is 1.01-1.63).

**There is no landing axis and no polarity axis. There is ONE OUTLIER
SITE - the active->front-porch fall, the only transition with picture on
one end - and everything else.** The two sync edges agree with each other
2x better than either agrees with the porch fall.

The proper null is NOT the noise floor. Live-transient, same-landing,
same-polarity, same-amplitude, DIFFERENT-POSITION pairs differ by
0.25-0.40 IRE rms. So the landing contrast is 0.67 against a null of
0.35 = **1.9x**, attributable term 0.57 +- 0.20 IRE rms - not the 3.8-5.1x
that using the noise floor implied.

### 14.4 Polarity - tested for the first time in this project

All 145,200 active->porch transitions are bright->dark, so polarity had
never been examined. Measured (both landing at blanking): rise vs fall
0.596-0.795 IRE rms, 1.6-2.4x null - **the same size as the landing
effect**, and by section 14.3 not separable from it.

### 14.5 VERDICT: (a) UPHELD, and restated

Not vacuous (a real 0.57 IRE rms difference exists); not disproven
(1.9x over null, zero degrees of freedom, separability falsified, and
the +15 IRE target is 49% beyond the anchor span). Two-anchor landing
extrapolation stays on the falsified list, now with a mechanism.

**RESTATEMENT.** Rule 10 should bind on the LANDING / LEVEL-BAND axis
specifically, and should explicitly NOT bind on the DRIVE-AMPLITUDE axis
at landing ~ blanking, where each tape's own blanking witnesses its own
picture regime:

- cd front-porch falls span A = 13-58 IRE (plus a peak-white 79.9 bin),
  half-to-half 0.13-0.89 IRE rms.
- home front-porch falls span A = 49-77 IRE, half-to-half 0.14-0.27 IRE
  rms - and **home's picture line-end regime is 49-74 IRE (1-99 pct),
  i.e. FULLY WITNESSED by home's own blanking.**

**home receiving nothing is therefore NOT required by rule 10.** It is
required only by a model that insists on a landing term. The amendment
unblocks home along the amplitude axis with no extrapolation at all.
Beyond A ~ 58 (cd) / 72 (home) the edge is not complete by lag 4 and the
kernel inverts sign - that is the real upper bound.

## 15. THE AGGREGATE BLANKING MEASUREMENT (Aug 12)

**User:** cd_pc14 "field 1 looks good, but then the ringing regresses from
field 2 onward, especially visible around the mic stand... the first ring
is corrected on frames 2 onward, but it shifts to reveal another harmonic
of ringing. I want to shift to use the entire sync pulse area:
(transition from active -> front porch, sync tip, back porch) as an
aggregate measurement."

### 15.1 The report is correct, and the cause is window length

The front porch carries TWO components, not one. Measured on the
uncorrected decode over the aggregate window:

| window | modes recovered |
|---|---|
| front porch, 13 lags | 2.54 MHz tau 8.4, 1.45 MHz tau 5.9 |
| back porch, 19 lags | 2.42 MHz tau 4.9, 1.28 MHz tau inf |
| **sync tip, 55 lags** | **2.02 MHz tau 8.3 (amp 0.365), 1.27 MHz tau 16.6 (amp 0.194)** |

Separating 1.6 from 2.0 MHz needs ~32 lags. The 13-lag porch cannot, so a
porch-only identification lands on ONE smeared mode, damps it, and leaves
the other standing - exactly "the first ring is corrected but another
appears". The user's aggregate proposal is the correct fix, because the
long sync windows supply the resolution the porch lacks.

Also measured, and worth acting on separately: **the existing blanking
correction halves the sync-tip ring (0.225 -> 0.116 IRE rms) but makes
the FRONT PORCH worse (0.163 -> 0.240)** - and the porch is where the eye
sees it.

### 15.2 A REAL CONDITIONING BUG IN THE PENCIL (fixed)

`_joint_pencil_poles` inverted singular values below numerical rank.
On a noise-free two-mode signal (1.27 + 2.02 MHz, rank 4) at the shipped
PENCIL_ORDER = 6, it returned a single 1.78 MHz pole with **neither true
frequency anywhere in the eigenvalues** - the garbage from inverting
3.6e-16 singular values does not merely add spurious poles, it DISPLACES
the real ones. Truncating at the numerical rank (max(shape)*eps*s0, the
standard tolerance - machine precision, not a tuned threshold) returns
1.27 and 2.02 exactly, and is robust with 0.05 IRE noise added. On real
noisy data every singular value clears the tolerance, so shipped
behaviour is unchanged - but the bug would bite any low-rank residual.

### 15.3 What the aggregate actually means (a correction to a first attempt)

Stacking a 13-lag porch beside a 59-lag sync window with a COMMON Hankel
width taken from the SHORTEST residual destroys the fit: a synthetic
2.02 MHz mode came back as 1.30 MHz. Width IS resolution. Taking it from
the LONGEST preserves the fit, but then the porch supplies no rows and is
structurally excluded - measured: cd_pc16 came out bit-identical to
cd_pc14, i.e. that version of "aggregate" was a no-op.

**The correct division of labour:** the sync tip and back porch (~59 lags)
fix the POLE FREQUENCIES; the front porch fixes the EXCITATION, because
it is the only leg driven by picture content and the only one where the
ring is visible to the eye. Both instruments must agree, each supplying
what it is good for.

### 15.4 Measured, cd (200 fields, blanking bit-identical in every row)

| round | pole source | RING med | RING p90 | GHOST med | GHOST p90 | sharp | mic lobes |
|---|---|---|---|---|---|---|---|
| pc9 | (baseline) | 1.95 | 5.99 | 5.02 | 12.33 | - | 3.5 |
| pc14 | porch, sync-confirmed | 1.80 | **4.85** | **5.01** | **12.73** | 0.960 | 3.4 |
| pc17 | all sync poles | **1.66** | 5.48 | 5.49 | 13.08 | 0.981 | **2.7** |
| pc18 | sync poles the porch supports | 1.68 | 5.10 | 5.23 | 12.94 | 0.972 | 3.1 |

The mic stand - the user's reported site - responds exactly as predicted.
pc9's lobe train `+0.7 +1.5 +2.0 +2.0 +1.8 +1.6 +1.3 +0.9` flattens to
pc17's `-0.5 -0.1 +0.2 +0.3 +0.4 +0.5 +0.4 +0.1`: the exposed second
component is now damped as well.

**No variant dominates.** pc14 holds the best tail metrics (ring p90,
ghost); pc17 the best ring median and lobe suppression; pc18 sits between
by construction. Ghost tracks how many sync poles are admitted, which is
the detail-band risk. THE EYE DECIDES - and the user's complaint was
specifically the mic stand, where pc17 > pc18 > pc14.

Tree state: pc18 (mutual confirmation - the most principled of the three:
frequency from the instrument that can resolve it, admission from the
instrument that sees picture coupling).

## 16. THE JOINT IMPULSE TEST — cd's ring is NOT linear (Aug 12)

**User:** "the correction happening in pc17 and 18 is causing another
ringing harmonic to appear. Is it possible to use the aggregate of the
entire sync area, where each known transition is measured, and additional
harmonics may be derived from the active -> sync area transition."

### 16.1 The test

Across blanking the ideal is known by specification, and three
transitions excite the channel: the active->front-porch fall (amplitude
varies per line with picture content), the sync fall, and the sync rise.
If ONE linear impulse response describes the channel then

    observed(t) - ideal(t) = sum_i s_i * h(t - t_i)

which is linear in h. Three excitations at three positions with different
amplitudes OVERCONSTRAIN h - a pattern locked to one position cannot
appear at all three offsets at once, which is precisely the
identifiability limit the porch alone could not escape (section 12.4).

Evaluated against the COHERENT MEAN profile (48400 lines, noise
suppressed ~220x), because the per-line residual is noise dominated
(0.76 IRE noise against a 1.4 IRE residual) and its R^2 measures the
noise floor rather than the model.

### 16.2 THE RESULT: the two tapes are opposite

| region | cd before -> after | home before -> after |
|---|---|---|
| front porch | 0.356 -> **0.370** WORSE | 0.548 -> **0.062** (-89%) |
| sync tip | 0.246 -> **0.254** WORSE | 0.847 -> 0.269 (-68%) |
| back porch | 0.193 -> **0.238** WORSE | 0.360 -> 0.342 (-5%) |
| whole window | 0.668 -> 0.456 | 1.167 -> 0.291 |

**home: one linear impulse response explains its blanking almost
completely.** The front porch falls from 0.548 to 0.062 IRE rms. The
model is right for that tape.

**cd: it fails.** The whole-window figure improves only by absorbing a
broad low-frequency term; EVERY individual region gets worse. A single
linear h driven by the three known transitions cannot explain cd's ring.

### 16.3 WHY THE EYE SAW NEW HARMONICS APPEAR

If the ring is not a linear response to the transitions, then ANY LTI
kernel fitted to it removes one component and leaves a MISMATCHED
residual - which reads as a new oscillation. That is the mechanism behind
the reported "the first ring is corrected but another appears", and it
gets worse as more modes are admitted (pc14 -> pc17/pc18). The eye was
reading a real property of the signal, not a tuning error.

This also inverts the tape roles the project had assumed: home was
labelled "no resonance, receives nothing" while cd received the
correction. On modellability it is the other way round.

### 16.4 Decision (user's call): APPLICATION OFF, MEASUREMENT RETAINED

`PEQ_APPLY = False`. cd returns to pc9 behaviour - verified: 0.075% of
samples differ by at most 1 LSB (0.0025 IRE), inside the known
non-determinism floor, with every ring_eye delta +0.00 and pulse_trace
identical.

**The front-porch measurement machinery is explicitly KEPT** (user: "I
need to continue to use the front porch as a measurement area. Do not
abandon that work"). It is now the project's primary instrument: it is
what produced home's validated model and what falsified cd's.

### 16.5 The two forward paths, now clearly separated

- **home** has a VALIDATED linear model from the porch, and rule 10's
  restatement (section 14.5) already unblocks it along the drive-amplitude
  axis - its picture line-end regime (49-74 IRE) is fully witnessed by its
  own blanking. This is the tape to correct next, and the correction is
  ordinary LTI.
- **cd** needs a LEVEL-DEPENDENT excitation model, not a better linear
  kernel. This is the FM-sideband direction, and it is consistent with the
  module's own recorded observation that cd's harmonic components sweep
  ~0.3 MHz across the witnessed level band. Do not ship another LTI
  kernel on cd.

## 17. SOLVED — THE MODEL CLASS WAS NEVER WRONG, THE SCORING WINDOW WAS

**User:** "using the entire sync area is the way to go... All of these need to
be considered as a single data set, where the active -> front porch gets
partial results depending on the active area, and the two sync pulses get a
consistent impulse response... Ideally a mathematical approach without monkey
patching and conditionals."

Nine agents, 1.56M tokens. The framing phase was run BEFORE any fitting, and
it found the error that invalidates most of sections 11-16.

### 17.1 THE ERROR: a guard that hid the response

Every measurement in this project - sections 11 through 16, and the two
adversarial audits - scored the blanking interval from **delay >= 7 samples
after each transition** (`guard = 3`). The residual energy after the sync fall,
coherent mean profile, uncorrected:

| | lags 1-7 (GUARDED OUT) | lags 8-40 (all that was ever scored) | hidden |
|---|---|---|---|
| home | 3.840 IRE rms | 1.053 IRE rms | **74%** |
| cd | 3.782 IRE rms | 0.256 IRE rms | **98%** |

**On cd we were scoring the 2% tail and concluding the model had failed.**

### 17.2 With the correct window, ONE shared response fits BOTH tapes

- Zero parameters, no estimator: the normalised step-response errors of the
  porch fall and the sync fall correlate **+0.940 (home) / +0.866 (cd)** over
  k = 1-7, reproducible across the time split to +-0.005 / +-0.04. Over the old
  window k = 8-40 they correlate only +0.292 / +0.135.
- Identify on the SYNC FALL ONLY (9 taps), then predict the picture-amplitude
  porch edge that was never fitted. LTI demands transfer coefficient gamma = 1:
  **gamma = 0.985 +- 0.018 (cd), 1.261 +- 0.009 (home)**.
  **cd is the CLEANER tape**, the exact opposite of section 16's conclusion.

### 17.3 cd is NOT nonlinear, NOT under-excited, NOT missing an input

Synthetic control: plant a known, exactly linear, exactly SHARED kernel into
cd's own record - cd's real per-line noise, real drive distribution and
autocorrelation, real geometry, real line count.

| cd's own noise + a known shared linear kernel | evidence AGAINST sharing |
|---|---|
| at 0.10 gain (3.5x smaller than cd's real ring) | +0.4 +- 0.3 nats |
| at cd's actual amplitude | +0.2 +- 0.2 nats |
| at actual amplitude, noise inflated x1.9 | +0.0 +- 0.0 nats |
| **REAL cd** | **+91.1 nats** |

cd's data would have supported the shared hypothesis overwhelmingly at a third
of its real amplitude with inflated noise. The rejection was real, but it was a
rejection OF THE TAIL, not of the model.

**What the tail actually is:** a POSITION-LOCKED DECODER ARTEFACT. It does not
decay (tail/head rms ratio 0.97 vs home's 0.42), does not scale with drive
(elasticity 0.005), and sits at 0.256 IRE - the bottom of the 0.25-0.40 IRE
position null. Corroborated independently: on VBI lines with NO PICTURE AT ALL,
76% of cd's and >50% of home's sync-tip structure is still present.
**It is not a response to anything, which is why correcting it produced new
harmonics.** That is the mechanism behind every visible regression this session.

Also retired: **Hammerstein is dead by algebra** - the input is piecewise
constant, so a static g is only ever evaluated at four levels and the transient
is exactly A_measured * h/H(0) for ANY g. It is a reparameterisation of LTI,
not a separate class. The earlier "g(A) = 1.000 on all tapes" was an IDENTITY,
not a measurement.

### 17.4 The model, with no conditionals anywhere

Input known by specification, three steps of size Delta_i at t_i. With
h = the step response of (G - 1),

    r_l(t) = y_l(t) - u_l(t) = SUM_i Delta_i,l * h(t - t_i)      linear in h

- **DC unity is a property of the parameterisation**: h(+-inf) = 0 so
  SUM_k dh_k = 0 so G(1) = 1 identically (measured 4.5e-17 / -1.4e-17).
  The correction C = 1/G then has SUM_m (c_m - delta_m) = 0 exactly - levels
  cannot move, with nothing bolted on.
- The 48400 lines collapse EXACTLY into three orthogonal profiles (rbar, c1,
  c2). That is the whole dataset; everything else is orthogonal to the class.
- The two open questions become two continuous hyperparameters, not branches:
  **kappa_e** (do the legs share one response) and **kappa_2** (level
  dependence), estimated by marginal likelihood alongside a stable-spline
  kernel. Complexity is set by EB effective dof, not by tap count.
- **The gate is a Bayesian model average**, w = P(model | porch-fall rows the
  filter never saw). No threshold: on the tail-derived kernel it returns
  w = 9.9e-305 and the correction collapses to identity by itself -
  |C| flat to 0.05 dB, no tap above 5e-4.
- Stability: ||dh||_1 = 0.66-0.85 < 1, so G is bounded away from zero on the
  unit circle and 1/G converges geometrically; the causal support [0,+8] is
  minimum-phase with ZERO look-ahead.

### 17.5 THE HONEST PROBLEM: the full inverse is a sharpening filter

|C| for the full inverse runs 0 -> **+8.67 dB at 2.31 MHz (home)** and
**+6.22 dB at 2.12 MHz (cd)**. That is broadband edge sharpening, which rule 3
excludes. The reason is structural: over the transition core the channel is
dominated by BAND-LIMITING and group delay, not by a resonance - the identified
pole has **Q ~ 0.7-1.2**, and a Q < 1.2 pole is a rolloff, not a ring.

Projected onto the admissible class (a second-order section with zeros on the
identified pole pair and poles at the same frequency with radius b < a,
normalised C(1) = 1):

| | resonance | a -> b | fraction of 1/G explained | |C| range |
|---|---|---|---|---|
| home | 0.980 MHz | 0.936 -> 0.891 | 14% | -4.47 .. +0.38 dB |
| cd | 0.812 MHz | 0.953 -> 0.928 | 12% | -3.66 .. +0.21 dB |

**DO NOT SHIP THE FULL INVERSE.** Ship only the admissible projection: legitimate,
level-neutral, sum-zero, stable, causal - and a sub-1 dB filter, because only
12-14% of what the blanking interval identifies is resonance damping at all.
Cost of strict causality: home -76% -> -64% held-out; **cd loses nothing**
(-57% -> -58%).

### 17.6 Method corrections that invalidate earlier numbers

- **Section 16's headline is withdrawn.** home's "porch 0.548 -> 0.062 (-89%)"
  does NOT survive the position null: displacing all three events by +7 samples
  fits the front porch BETTER (0.0471) than the true positions (0.0614), and a
  phase-randomised surrogate is "explained" +42% +- 13% by the same design.
- **The time split has no power.** corr(rbar_early, rbar_late) = 0.9967 (cd) /
  0.9983 (home) - the target barely moves, so fit-early/score-late is
  arithmetically almost in-sample. Every "holds out of sample" claim resting on
  it, including mine, is weaker than stated. Use leave-one-TRANSITION-out.
- **N_eff is 1288 (cd) / 2301 (home), not 48400** (adjacent lines correlate
  0.95 / 0.87).
- **90.7% (cd) / 99.0% (home) of the Fisher information is the FIXED shape.**
  The picture-varying leg carries 1.3% / 0.6% of the objective. The premise that
  three excitations "overconstrain" h is false.
- **Taps 0-2 were constrained by no event at all** under guard = 3 - exactly
  where a step-response error is largest.

## 18. THE DRIVE-SCALED PORCH RING — the component that IS a ring

**User:** "I'm looking at the area on the front porch, and I visibly see a
ringing like distortion right after the active area on the countdown sample.
This seems to match the frequency of the ringing after the mic stand. Let's
look at modeling the ringing against the falling edge of the active area."

Correct on both counts, and it identifies the one component in blanking that
is a genuine resonance AND provably free of the position-locked artefact.

### 18.1 The exact separation

Write each line's porch residual as r_l with drive A_l = Abar + a_l:

    rbar = mean_l r_l                  POSITION-LOCKED (identical every line)
    c1   = SUM a_l r_l / SUM a_l^2     DRIVE-SCALED (what the picture caused)

Anything locked to horizontal position is identical on every line and is
therefore ORTHOGONAL to a_l: it cancels EXACTLY in c1. No subtraction, no
model - the algebra removes it. That matters because the position-locked part
is the LARGER of the two on cd (0.356 vs 0.231 IRE rms) and is present on VBI
lines carrying no picture at all.

### 18.2 Measured (uncorrected decodes, 48300 line ends)

| tape | position-locked (rbar) | DRIVE-SCALED (c1) | omega*tau |
|---|---|---|---|
| cd | 2.30 MHz tau 19.4, rms 0.356 | **2.04 MHz tau 8.5, rms 0.231** | 7.6 rad |
| home | 2.47 MHz tau 2.8, rms 0.549 | **2.11 MHz tau 6.8, rms 1.029** | 6.3 rad |

**These are genuine resonances** - 6-8 radians of rotation within their own
decay time, Q ~ 3-4 - unlike the sync response window, which is an overdamped
rolloff (Q 0.7-1.2, section 17.5). The mic-stand rod carries a matching
1.83 MHz component (rms 0.40) beneath its settling foot, which is the match
the user saw by eye. home's drive-scaled ring is LARGER than its
position-locked part and plainly oscillatory in the profile:
+1.49 +2.45 +2.46 +1.73 +0.99 +0.80 +1.15 +1.63 IRE at mean drive.

### 18.3 Shipped identification (application currently off)

`_accumulate_porch_drive_response` keeps five running sums under the standard
forgetting factor (n, sum a, sum a^2, sum r, sum a*r) - a few floats per field
- and `_porch_drive_ring` forms c1, takes the most energetic pole that passes
the rotation-against-pi resonance test, and returns a second-order section with
zeros on that pole and poles at the same angle with a smaller radius,
normalised to unit DC gain. Levels cannot move; the passband away from the
resonance is untouched because zero and pole nearly coincide there.

The damping radius is fitted, not chosen: it is the value that leaves least
ring, stopping early if the ring reaches its own uncertainty. NOTE the floor
must be the uncertainty OF c1 - sigma/sqrt(SUM a^2), about 150x below the
per-line noise - not the per-line noise itself; comparing an averaged quantity
against a per-line floor stopped the fit before it acted (cd_pc28/29 were
effectively no-ops).

### 18.4 Decoded (cd_pc30 / home_pc30)

| | front porch max\|dev\| | front porch rms | undershoot | RING med | ghost | sharp |
|---|---|---|---|---|---|---|
| cd_pc9 | 1.04 | 0.54 | -0.94 | 1.95 | 5.02 | - |
| cd_pc30 | **0.92** | **0.49** | **-0.89** | 1.99 | 5.02 | 1.001 |
| home_pc30 | 1.14 (from 1.16) | 0.65 (0.66) | -1.00 | 1.34 (1.31) | 2.33 | 0.999 |

All pulse_trace assertions PASS on both. **The front porch - the exact area
the user pointed at - measurably improves on cd (max deviation -12%, rms -9%)
and nothing regresses.** But the picture ring gauges do not move: ring_eye
RING/GHOST are flat to +-0.05 and the mic-stand train is unchanged.

So the porch ring is real, resonant, drive-scaled, and now removable - but at
picture amplitudes its removal is below what the site gauges can resolve. The
open question is whether the eye sees what the gauges cannot; that is the one
thing the instruments cannot settle.

## 19. The front-porch ringing, modelled — and the limit of where it may be applied

**The symptom was real and the instruments had been reading the wrong
quantity.** The shipped correction was doing nothing at all to the
picture-caused porch ring: pc37 against uncorrected gave cd 0.4410 →
0.4457 and home 0.2434 → 0.2429 IRE rms, profiles identical sample for
sample. Three independent causes, each sufficient on its own: the applied
gate faded to zero *before* the front porch, so the transition the model
came from was the one transition never corrected; the identification
window was guarded at both ends and held **1.5%** of the response energy;
and the channel was reconstructed with the wrong sign (`g = δ − Δh`, the
code built `δ + Δh`), so the "inverse" was the channel — on a synthetic
channel built from the measured response the shipped kernel *adds* 80%
(cd) / 77% (home) where the correct-sign kernel removes 99%.

**The model.** A subtracted train, not an inverse filter:

    corrected = signal − conv(change, train)

Causality, exact level neutrality (`C(1) = 1` because the difference
operator annihilates DC) and no-blur are properties of that form rather
than constraints imposed on it. Inverting is falsified for this problem:
the porch response is dominated by the *edge completing*, and inverting
that is broadband sharpening.

**The estimator** fits `y_l(t) = p(t) + Σ_k change_l(end + t − k)·train(k)`
with `p(t)` a **free** per-lag constant. `p(t)` is the position-locked
decoder artefact — present on VBI lines carrying no picture — and leaving
it free is what stops the train absorbing it and carrying it into the
picture. A *scalar* line-end drive cannot be used: cd's apparent response
grows and widens with the number of fields averaged (+1.84 → +4.46 IRE at
the crossing between 30 and 200 fields) purely because its content stops
being flat at the line end, while home's is stationary to three figures.

Out-of-sample before any decode: split-half −1.4%/−1.0% (cd) and
−4.3%/−3.6% (home) against in-sample −1.4% / −4.3%; half-to-half train
agreement r = +0.896 / +0.950; each tape's train applied to the other
*fails* (+4.9% / +3.6%).

**§19a — carrying it into the active picture is falsified, for the second
time, and now the reason is measured.** Applied picture-wide it is a large
cd win (ring_eye abs med 1.89 → 1.59, p90 5.97 → 4.73) and a home
regression (1.01 → 1.42), reproducing the standing result recorded in
`_correction_for`. Two new instruments say why:

* `landing_transfer.py` projects each isolated picture edge's own
  post-edge residual onto the porch train (aggregate least squares,
  against a time-reversed null). Inside each tape's witnessed swing band:
  cd **+0.494 ± 0.095** (null +0.231), home **+0.016 ± 0.069** (null
  −0.058). Home does not transfer at all — **at matched amplitude and
  matched landing**, since both tapes' picture edges land at +17/+18 IRE.
  The landing hypothesis of rule 22 is *refuted* as the discriminator here.
* `porch_vs_sync.py` is the blanking-only test that decides it. The sync
  fall is the other transition into a flat-by-specification region; if the
  porch response described it too, the response would be a channel
  property and could be carried anywhere. It does not, on either tape:
  correlation **−0.249** (cd) / **+0.143** (home), projection gain −0.100
  / +0.067. The porch response belongs to the line-end transition that
  produced it.

So the correction ships confined to the front porch — the regime it was
witnessed in (rule 10/22). Decoded, that improves the porch on both tapes
past uncorrected (per-line ring cd 0.672 → 0.596, home 1.071 → 0.819)
with pulse_trace all-PASS, levels unchanged, and vsync/chroma untouched.

## 20. Two families, two places — the settling arc travels, the resonance stays

The eye reported the trade precisely: carrying the whole porch train across
the picture fixed the **overshoot** behind the mic stand and brought back the
**trailing ring**. Both come from one change, and the mic-stand rod shows
them side by side (lags from the fall):

```
            +4   +5   +6   +7  |  +9  +10  +11  +12   lobes  rms
uncorrected 6.1  6.0  5.5  4.4  | 1.8  0.9  0.9  1.5   +3.1  2.36
pc38        4.7  3.9  3.3  3.5  | 3.4  2.5  1.7  1.2   +3.8  2.03
pc39        5.3  5.7  5.7  4.9  | 1.8  0.5  0.2  0.7   +3.5  2.26
pc41        5.2  4.5  3.7  3.4  | 2.2  1.2  0.7  0.9   +3.0  2.07
```

**Orthogonality is the wrong boundary.** Projecting the carried train off the
six sync-derived trains (pc40) keeps the trough but recovers almost none of
the overshoot (+4…+7: 5.1 5.2 5.1 4.6). Both effects live in the component
that *overlaps* those trains — pc38 is applying more of that shape, which
helps early and over-corrects late.

**The family boundary is the right one**, and it is the module's own
(`ω·τ ≥ π`, its own rank-truncated pencil). cd's porch train splits into
3.08 MHz τ 3.1 (resonance) and 0.70 MHz τ 7.6 (relaxation):

* the **relaxation** arc travels across the active picture — no sync edge
  witnesses a settling arc at picture amplitude, so nothing else corrects it
  and it cannot collide with Parts 1–4;
* the **resonance** stays on the front porch — Parts 1–4 already correct that
  family at every transition from the sync edges, so carrying it again
  over-corrects; on the porch it is not a duplicate (no sync edge drives the
  porch) and it is the regime it was witnessed in.

Decoded, pc41 keeps most of pc38's picture gain (ring_eye 1.69/4.99 against
pc38 1.59/4.73 and pc39 1.91/5.84), holds the trailing trough near pc39,
gives the lowest post-foot lobe of any round including uncorrected, and
preserves high-frequency detail like pc38 (2–3 MHz ×0.92 against pc39's
×0.78) — so the confined version's HF loss is recovered too.

**The split is self-limiting.** `home_pc41` is byte-identical to `home_pc39`:
the pencil resolves no relaxation family in home's train, so nothing travels
and home keeps the porch-confined behaviour. A tape whose porch shows no
witnessed settling arc gets no picture-wide correction — no threshold to
tune, no per-tape configuration.
