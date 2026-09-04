# What the domains say to each other

The algorithm is in `THE_ALGORITHM.md`. This document is the other half: what
mapping different domains onto it revealed, and the characteristics that only
became visible once several domains could be compared on one measure.

Every number here is measured, by the same procedure, on the same
implementation. The measure throughout is the **participation ratio of the
singular values** — how many independent directions a family of signatures
actually spans, against how many members it has.

---

## 1. The comparison

| domain | effective / count | condition | worst pair |
|---|---|---|---|
| tape magnetics (VHS) | **1.58 / 6** | 1.06 × 10⁴ | gap vs azimuth, **1.000** |
| tape magnetics (U-matic low band) | 1.51 / 6 | — | same |
| tape magnetics (U-matic SP) | 1.55 / 6 | — | same |
| the same, from dimensional analysis | 1.90 / 5 | — | — |
| propagation | **3.79 / 6** | **6.4** | SAW delay vs sound trap, 0.764 |
| propagation, filters only | 2.95 / 5 | — | — |
| multipath echoes, spaced ≥ 1/B | **7.92 / 8** | **1.2** | — |
| modelled interference, VHS luma band | **6.89 / 12** | 10.1 | — |
| the same, U-matic band | 7.14 / 12 | 9.0 | — |
| the same, demodulator band only | 5.76 / 12 | 32.6 | — |
| BT.1700 timing (625 PAL) | **5.59 / 26** | — | nine symbols, one measurement |

---

## 2. The law that explains the spread

**A family is collinear when its members differ only in a rate, and
separable when they differ in kind.**

Every tape loss is `exp(−kf)` or `sinc(kf)`: a monotone decay, and the family
of monotone decays is nearly one-dimensional over a finite band. Reduced to
exact dimensions the reason is starker still — every one of them is a
function of a single dimensionless group, a length over the recorded
wavelength:

| mechanism | form |
|---|---|
| spacing | `exp(−2π d/λ)` |
| gap | `sinc(g/λ)` |
| thickness | `(1 − exp(−2π t/λ)) / (2π t/λ)` |
| azimuth | `sinc(w·tanθ / λ)` |
| contour | `cos(2π ℓ/λ)·e^(−0.15 ℓ/λ)` |

Five mechanisms, five different lengths, **one** group. As far as a frequency
response is concerned they are not five physical effects; they are five
values of the same variable.

An echo family is `exp(−2πjfτ)` — a Fourier basis over delay, orthogonal by
construction. Delays separated by more than the band's resolution are
independent, and the result is scale-free: identical in units of `1/B` at any
bandwidth. **The tape's mechanisms are parameterised along the axis that
makes them alike; multipath's along the axis that makes them independent.**

---

## 3. What moves identifiability, and what does not

Tested by moving one variable at a time across formats:

| variable | effect |
|---|---|
| writing speed (4.85 → 41.3 m/s) | **inert**, and very slightly adverse |
| track width (58 → 137 µm) | **inert** to three decimals |
| fractional bandwidth (1.3:1 → 60:1) | 1.395 → 1.589 — **the only thing that moves it** |

Writing speed is a similarity transformation on the frequency axis: it
rescales every signature at once and therefore cannot separate them. Track
width enters `log_response` only as the product `track_width · tan(azimuth)`,
algebraically degenerate with the angle.

**The actionable consequence:** fit the head model over the widest band the
RF occupies, not over the demodulator's band-pass. Extending a U-matic fit
down to its 688 kHz colour-under carrier moves 1.514 → 1.551 at no cost, and
on the interference set the demodulator band gives 5.76 where the full band
gives 6.89.

---

## 4. Where each domain's instrument is strong, and where it is weak

**A better format can be a worse instrument.** U-matic is a guard-band
format: its two heads are nominally at the same azimuth, so a systematic
azimuth error affects both alike and *cancels* in the head difference — the
one mechanism that separates two VHS heads. What remains in a U-matic head
difference is gain, spacing, gap and contour, and the first two lie in the
carrier law's null space. The higher-quality format has the weaker
head-difference instrument.

**The timing specification is nine symbols for one measurement.** BT.1700
lists 24–26 measurable timing quantities per system and has 9–10 independent
dimensions. Line frequency, subcarrier, field frequency, line period, the
vertical interval sequence lengths and burst duration are all one quantity
under different names. From timing alone, M/NTSC and 525 PAL cannot be
separated at all, nor 625 PAL from Argentina 625 PAL; reconciling the
standard's own duplicate specifications, all three 625-line variants have
identical timing vectors and line count is the only timing dimension that
survives. **The chrominance subcarrier is the sole quantity separating any
same-line-count pair.**

**A transmission defect admits a video-domain inverse where a tape defect
does not.** The VSB Nyquist property makes `R(f) + R(−f) = 1` exactly, so a
synchronously detected signal is recovered by a *linear* operation. This
project's standing "no video-domain inverse" law has a nonlinear demodulator
as its premise, and the transmission chain does not have one.

---

## 5. Two rules that need a stage qualifier

Both were written for the tape and are wrong for the transmission chain.

**Pre-echo.** `RINGING_RULES.md` rule 2 forbids response before the driving
edge absolutely, and that is right for a physical tape channel. ATSC A/49
admits echoes from −3 µs, so for a transmission component a non-causal echo
is **evidence of transmission** rather than grounds for rejection.

**Video-domain filtering.** Forbidden at RF for the tape, correct for a
synchronously detected transmission defect, per section 4.

Neither rule should be relaxed generally. Both need to say which stage they
apply to. **This is Ethan's call, not a change to make unilaterally.**

---

## 6. What separates a tape defect from a transmission defect

Ranked by how decisive each is:

| discriminator | why it works |
|---|---|
| **level dependence** | a tape echo varies with picture level; a transmission echo, being LTI in the video, does not |
| head parity | a tape defect alternates with the head; a transmission defect does not |
| absolute frequency landmarks | 4.5 MHz intercarrier is unreachable through VHS's 3 MHz luma baseband, so a sound trap seen in a tape decode was applied before recording |
| delay sign | a pre-echo cannot be the tape |
| wow coupling | a tape defect moves with the time base |
| transport rates | a tape defect lands on a drum or capstan rate |
| the GCR's nested differential | cancels the shared tape path by exactly the construction step 1 already implements |

---

## 7. The interference types, and which one is irreducible

Given a random phase per field so that none could correlate trivially:

| disturbance | directions | margin |
|---|---|---|
| **particle noise** | **0** | **−0.1 σ** — at the floor |
| beat / co-channel | 1 | +351 σ |
| head contact tilt | 1 | +351 σ |
| dropout | 4 | +133 σ |
| modulation noise | 4 | +89 σ |

Four of five are not random to the transform at all. **A beat has a frequency
even when its phase is random; a dropout has a shape even when its position
is not.** Only particle noise — the finite number of particles in the volume
the head reads — sits at the sphere floor, and it sits there to within a
tenth of a sigma.

Adding these to the magnetics raises the VHS luma band from 1.58 of 6 to
**6.89 of 12**, and the condition number from 1.06 × 10⁴ to 10.1.

---

## 8. Where the capture stops mattering

| chain | tape C/N | verdict |
|---|---|---|
| VHS, raw envelope spread | 21.6 dB | tape binds, converter holds 2.2× |
| VHS, after separating structure | 33.1 dB | tape binds, converter holds **1.41×** |
| U-matic (projected) | 23–25 dB | tape binds, 1.8–1.9× |
| off-air VHS recording | — | tape binds; worst broadcast contour 68.6 Mbit/s against the tape's 57.4 |
| direct channel capture | — | propagation binds; **10 bits at 20 MSps clears every FCC contour with 10 dB to spare** |

The converter takes over only above about **44–46 dB** tape carrier-to-noise
at 8 bits and 40–50 MSps, which is above the medium's own particle-noise
ceiling for every tape format examined. **Broadcast formats do not need
better capture hardware than VHS does.** For echoes the dynamic range that
matters comes from the ghost-cancelling reference's processing gain — 23.4 dB
on one line, 52.9 dB over 900 frames — not from more bits.

---

## 8a. The key must span, not merely be large

Ethan: *"it can be solved by finding as many dimensions as exist and being
able to model them exactly, which you cannot since they encode information.
That is the part that you cannot know unless you have the key. We have the
key which is all possible modifications to our video signal, that's the
key."*

This is the completeness question, and it is measurable rather than
assumable. `span_completeness` orthonormalises the modelled signatures,
projects a residual onto that basis, and reports the share that lands
inside. What is outside cannot be recovered by any fit, because nothing in
the model can represent it.

| residual made of | inside | a basis this size captures | excess |
|---|---|---|---|
| a head tilt the key models | 100.0% | 2.1% | **+97.9%** |
| an echo the key models | 86.3% | 2.1% | +84.2% |
| pure noise | 2.3% | 2.1% | +0.1% |
| a shape the key does NOT model | 1.6% | 2.1% | **−0.5%** |

The measure is charitable to the key by construction - a basis of `k`
vectors captures `k/L` of anything at all - so the excess above that null is
what says the key covers the residual rather than merely being large.

**And "merely being large" is a real failure mode.** Adding the S-VHS sub
pre-emphasis to the modelled set as two of its tabulated curves - the 0 dB
and −20 dB rows of IEC 60774-3 table 3 - **lowered** the effective count from
6.92 to 6.00 and worsened the condition from 10.5 to 28.5. Two curves of
nearly the same shape are one direction, and a second copy of a direction
already in the span makes the ensemble worse conditioned without adding any
coverage.

Entered instead as its **level dependence** - the difference between two
tabulated levels, which is the axis no loss mechanism varies along - the same
modification restores the count and improves it: **6.92 of 13 on the VHS luma
band, and 7.97 of 13 at condition 10.0 over the full RF band.**

The rule this gives: *a modification earns a place in the key by the
direction it adds, not by being a modification.* When a standard hands over a
family of curves, enter what varies between them.

## 9. The measurement traps this exercise exposed

Each cost a wrong number before it was found.

**A shape measure must be told the dimensionality it is judging.** Deflating
a direction drawn from the ensemble's own span leaves an exact zero
eigenvalue; counting it reports the deflation's own bookkeeping as structure.
Not fixable by dropping zeros — a genuine needle has `N−1` of them and that
is real asymmetry.

**A nested differential matrix is rank-deficient by construction**, since
`D_ij = d_i − d_j`. Charge that deficiency to the ensemble, once, not to the
deflation.

**Inner products across different abscissae are memory layout, not
coherence.** A 238-bin response and a 262-line residual truncated to a common
length return a number that means nothing.

**Casting complex to float keeps the real part and drops the phase.** Not a
precision loss — the same ensemble's asymmetry moves from 0.248 to 0.876, a
different answer to a different question.

**A gain and a delay of the same shape are one direction under the Hermitian
inner product.** Correct when the complex value is one measurement; wrong
when each component is a real parameter's signature. Moves a propagation
result from 2.20 to 3.79.

**`significant` and the participation ratio ask different questions.** The
first counts directions standing above the bulk; an orthogonal family has
none, because it *is* the bulk. Use the participation ratio to count how many
independent directions a family spans.

**Saturation is not emptiness.** With 91 entries over 9 places the sphere
floor is 0.92 — a random ensemble is already almost maximally asymmetric and
nothing can stand above it. The fix is more places, not more fields.
