# The mathematics of the complex multi-dimensional elliptical curve transform

Ethan supplied the design; this document derives it. Every closed form is
worked from its assumptions, every claim is marked as **design** (his), as
**derivation** (follows from the design), or as **measurement** (established
by running it, with the proof number that regenerates it).

Run `python3 -m tools.ringing_measure.proofs` to regenerate every number
here. `docs/PROOFS.md` is that run, written down.

---

## 1. Objects and notation

Let a **channel** be one measured quantity sampled on an abscissa: the
carrier envelope, the demodulated frequency, the per-line time-base residual.
Write `x[n]`, `n = 0 … L−1`.

Let a **component** `i` carry two vectors on a shared abscissa:

| | |
|---|---|
| `s_i` | its **synthetic** value — what the standard or the physics says |
| `m_i` | its **measured** value — what the chain produced |

and a position in a directed graph `G` whose edges are declared, not fitted.

Everything below is complex. Where a complex value *is* one measurement —
amplitude and phase as two parts of one quantity — the Hermitian inner
product `⟨u, v⟩ = uᴴv` is the physical one. Where each component is instead
the signature of a **real** physical parameter, the correct space is the
real vector `[Re u; Im u] ∈ ℝ^{2L}` (§9.3).

---

## 2. The differential

**Design.** *"The differentiation is the differential of our synthetic
components to the actual measured componnet, this has nested differentials in
it as well according to our graph."*

**Definition 2.1 (diagonal).** `d_i = s_i − m_i`.

**Definition 2.2 (nested).** For `(i,j)` related in `G`, the graph asserts an
expected relationship `s_i − s_j`; the measurements assert an actual one
`m_i − m_j`; the entry is their difference:

```
D_ij = (s_i − s_j) − (m_i − m_j) = d_i − d_j
```

**Lemma 2.3.** The nested matrix is rank-deficient by construction.

*Proof.* `D_ij = d_i − d_j` lies in `span{d_1 … d_N}`. With all pairs formed,
the ensemble has `N(N+1)/2` entries in a space of dimension at most `N`, so
the deficiency is at least `N(N+1)/2 − N = N(N−1)/2`. ∎

This lemma is the source of every failure in §10. It must be carried, not
discovered later.

---

## 3. A dimension

**Design.** *"The 'dimensions' are the Weiner transform of each complex pair
(i.e. luma frequency vs. luma amplitude)."*

An axis is **not** a dimension. An axis is one channel. A dimension is
manufactured from two channels co-registered on one abscissa.

**Definition 3.1.** For channels `x, y` cut into `n` segments and
Welch-averaged,

```
T(f)  = S_yx(f) / S_xx(f)                       the transfer  (H1 estimator)
γ²(f) = |S_yx(f)|² / (S_xx(f) S_yy(f))          the coherence
se(f) = √( (S_yy/S_xx)(1 − γ²) / n )            its standard error
```

`|T|`, `arg T`, and the frequency it lives at are the *"amplitude, phase, and
frequency of this differential pair"*. Three channels give three pairs.

**Lemma 3.2 (the coherence is biased).** For independent `x, y` and `n`
averaged segments, `E[γ̂²] = 1/n`, and at `n = 1`, `γ̂² ≡ 1`.

*Proof.* At `n = 1`, `|S_yx|² = |Y|²|X|²= S_xx S_yy` identically, so the ratio
is one whatever the data. For `n > 1` the numerator averages `n` independent
unit-modulus phasors, whose squared mean modulus has expectation `1/n`. ∎

Hence the debiased form actually used:

```
γ² ← (γ̂² − 1/n) / (1 − 1/n)
```

**Measurement (proof 21).** A known one-pole channel is recovered to **2.75%
in gain and 1.61° in phase** over the coherent band, and independent channels
return raw coherence 0.2476, 0.0575, 0.0153 at `n` = 4, 16, 64 — matching
`1/n` = 0.2500, 0.0625, 0.0156.

**Why the standing refutation of spectral division does not apply.** That
refutation is of dividing two spectra *of the same quantity* to recover a
response, which manufactures phantoms wherever the denominator is small. Here
the denominator is the input's own power, the numerator is a *cross*
spectrum, and `γ²` states where the estimate is trustworthy — the thing the
refuted construction lacked.

---

## 4. The quadratic form, and the invariant

**Design.** *"All I need is to fit these residuals to an elipse of how ever
many dimensions I have, and I have fully collapsed the differentiation. The
constant is the elipse."*

Let `u_i = D_i / ‖D_i‖` and let `U ∈ ℂ^{N×L}` have the `u_i` as rows.

**Definition 4.1.** `G = U Uᴴ`, Hermitian positive semi-definite, with
eigendecomposition `G = V Λ Vᴴ`. The set `{z : zᴴ G⁻¹ z = 1}` is the
ellipsoid; `√λ_k` are its semi-axes.

**Theorem 4.2 (the invariant).** `tr G = N` for every ensemble.

*Proof.* `G_ii = u_iᴴ u_i = ‖D_i‖² / ‖D_i‖² = 1`, so the trace is the number
of rows. ∎

**Measurement (proof 1).** Trace equals the component count to better than
`10⁻⁹` for independent, collinear, and wildly-different-norm ensembles.

**Remark 4.3 (what this does and does not license).** Theorem 4.2 is an
identity of the normalisation, not a conservation law of the data. What it
licenses is the consequence: since the total is fixed, **only the shape
carries information**. That consequence is the load-bearing one.

**Proposition 4.4 (why this is a collapse).** The rank-increasing recursion
of matched-set differentials — differentials of differentials, "all the way
down" — converges to the eigenbasis of the ensemble's second moment. Solving
the eigenproblem attains that basis in one step.

*Sketch.* Each pass of ordered Gram–Schmidt against the ensemble's leading
direction is one step of orthogonal iteration on `G`; orthogonal iteration
converges to the invariant subspaces of `G`, which `eigh` returns directly. ∎

---

## 5. Reading the spectrum

Three different questions, three different statistics. Conflating them is a
recorded error (§10.4).

### 5.1 How much information

**Design.** *"The total eigenvalue is the total amount of information we
could possible derive from our components."*

By Theorem 4.2, `Σλ_k = N`: **`N` components hold `N` units and no more.**
The share standing in real directions is

```
resolved = Σ_{k significant} (λ_k − bulk)
```

with `bulk` the level of the eigenvalues that are not directions.

### 5.2 How far from a sphere

With `p` the participation ratio of the eigenvalue shares and `n` the
directions still available,

```
p         = 1 / Σ_k (λ_k / Σλ)²
asymmetry = (n − p) / (n − 1) ∈ [0, 1]
```

Zero is a sphere (every direction carries the same); one is a needle (`N`
views of one thing).

**`n` is `N` minus the directions already removed.** A shape measure must be
told the dimensionality it is judging, or it measures its own deflation
(§10.2).

### 5.3 How many directions are real

**Theorem 5.1 (Marchenko–Pastur edge).** For `N` independent unit rows of
length `L`, the spectrum of `G` is supported on `[(1−√c)², (1+√c)²]` with
`c = N/L`, so

```
significant = #{ k : λ_k > (1 + √(N/L))² }
```

**Measurement (proof 3).** On noise ensembles the largest eigenvalue sits at
0.83–0.96 of the edge at the sizes tested, so the edge is conservative there.
The shortfall is the finite-size Tracy–Widom effect and shrinks with `L`; at
larger sizes the exceedance rate settles at a few per cent, so "never admits
noise" holds only at those sizes.

**The hypothesis matters.** Theorem 5.1 requires *independent* rows. By Lemma
2.3 a nested matrix has none, and the edge does not apply to it at all
(§10.1).

### 5.4 `significant` and `p` ask different questions

`significant` counts directions standing **above** the bulk. A perfectly
orthogonal family has none — it *is* the bulk, uniformly — so a sphere is the
correct reading. To ask instead how many independent directions a family
*spans*, use `p`.

**Measurement (proof 9).** Eight echoes spaced at `1/B`: `p` = 7.92 of 8 at
condition 1.20, with asymmetry 0.0002 — a sphere, and `significant` small.

---

## 6. Termination

**Design.** *"This loops until the information floor is reached … Down to the
circular shape of the complex signal."* And: *"The limit is acheived when the
residual stops increasing … which gives you the exact residual."*

### 6.1 The sphere floor

A finite ensemble of random directions is never exactly spherical, so the
threshold is derived.

**Theorem 6.1.** For `N` random unit rows of length `L`,

```
E[asymmetry] ≈ N / (L + N − 1)
```

*Proof.* `Σλ = N` and `Σλ² = tr G² = Σ_{ij}|G_ij|² = N + Σ_{i≠j}|G_ij|²`.
For random unit directions `E|G_ij|² = 1/L`, so
`E[Σλ²] = N + N(N−1)/L`. Then

```
p = (Σλ)²/Σλ² = N² / (N + N(N−1)/L) = NL/(L + N − 1)
```

and substituting into `asymmetry = (N − p)/(N − 1)` gives `N/(L+N−1)`. ∎

(This is a delta-method step — `E[A]/E[B]` for `E[A/B]` — which measurement
shows to be accurate.)

**Measurement (proof 2).** Over `N ∈ {3…32} × L ∈ {64…1024}`, 300 trials
each, measured/predicted stays within **0.94 to 1.05** with no trend. At 20
trials the `N = 3` cell reads 0.797, which is the estimator's own standard
error and not a shortfall.

**Corollary 6.2 (the scatter).** Propagating the second moment,
`sd(asymmetry) ≈ sd(Σ_{i≠j}|G_ij|²)/(N−1)`, giving

```
real rows:     2 √(N/(N−1)) / L
complex rows:  √2 √(N/(N−1)) / L
```

**`√2/L` is the complex large-`N` asymptote.** Measurement (proof 14):
`sd × L` = 2.06 (real) against 1.41 (complex) at `N=8, L=1024`. A real
ensemble's sigma computed with `√2/L` is inflated by about 1.4×, and when `N`
approaches `L` the true scatter collapses much further (§10.3).

Termination: **at the floor when `(asymmetry − floor)/scatter ≤ 3`.**

### 6.2 The residual stops increasing

Each pass attributes energy `g_t` to the directions it found; the attribution
accumulates, `A_t = Σ_{τ≤t} g_τ`. The loop stops when `g_t` falls below a
thousandth of `max_τ g_τ`, and `starting_energy − A` is the **exact
residual**.

**Measurement.** Against planted ground truth (test
`test_the_accumulation_rule_alone_does_not_terminate_correctly`): this rule
**alone** attributes 99.9% against a true structure share of 72–73%, because
a pass can always find something in noise. **It is necessary but not
sufficient and must run beside the sphere test.** With both, the loop
attributes 77–79% — optimistic by about five points of energy, which is
ordinary shrinkage bias.

---

## 7. The step

**Design.** *"Then differentiation using the target curve onto the source
data."*

The target curve is the **whole** ellipse: every direction above the edge,
together. Deflating one at a time part-way rebuilds the recursion the
collapse replaces and does not converge.

**Theorem 7.1 (the weight).** With `λ = bulk + signal`, the MMSE shrinkage
for direction `k` is

```
w_k = (λ_k − bulk)/λ_k = SNR/(1+SNR) = 1/(1 + ρ),   ρ = 1/SNR
```

*Proof.* Immediate rearrangement. ∎

This coincides with the project's correction-gain law `a* = 1/(1+ρ)`, and the
flat half-step is the case `ρ = 1`. **Measurement (proof 4):** identical to
`10⁻¹²` at every eigenvalue.

**Remark 7.2 (an honest qualification).** Theorem 7.1 is a rearrangement of
one MMSE expression, not two independent derivations meeting. And the two `ρ`
are different quantities: this one is read from the *same ensemble the ellipse
was fitted to*, and the correction-gain law's own record states that such a
resampling returns `ρ = 0.00` where the truth is 2.47. It is a
self-calibrating shrinkage estimator *of Wiener form*, and diagonalising the
second moment is the Karhunen–Loève transform in which the Wiener filter is
diagonal — but it is not the same `ρ`.

**Remark 7.3.** `bulk` is estimated as the median of the sub-edge
eigenvalues. Marchenko–Pastur is right-skewed, so that median is biased
**low** and the weight correspondingly **high** — the opposite direction from
"apply at half the believed optimum".

---

## 8. The information accounting

**Design.** *"There are a fixed number of components that exist in a video
signal that represent video and noise randomness."* And: *"once I have
reached the absolute limit of the noise floor, i.e. 40mhz, 8 bits, field
length, then I have excausted the measurable differential in the complex
plane."*

**Proposition 8.1 (the extent).** A signal band-limited to `B` over duration
`T` has `2BT` real degrees of freedom, hence `BT` complex points.

For the capture: `50 MHz × (1/59.94 s) / 2` = **417,084 complex points per
field**. Over the 7 MHz Carson band: 116,783.

**Proposition 8.2 (the precision).** By Shannon, a link of bandwidth `B` and
signal-to-noise `S/N` carries `B log₂(1 + S/N)` bit/s.

**Measurement (proof 22).** Per NTSC field:

| reading | C/N | kbit/field | bits/point |
|---|---|---|---|
| raw envelope spread | 18.55 dB | 722.0 | 1.73 |
| after separating structure | 33.1 dB | 1284.2 | 3.08 |
| the particle-noise floor | 41.1 dB | 1594.5 | 3.82 |

**So the extent is not what runs out.** The run as constructed uses 0.37% of
the plane. What runs out is precision, and VHS carries 2.8–6.5 equivalent
bits against the converter's 8 — which is why the tape binds (proof 20) and
better capture hardware buys nothing.

**Proposition 8.3 (the partition).** Everything is a partition of Prop. 8.1:

| part | what it is | how it is told |
|---|---|---|
| inside the key | the modelled modifications | projection onto the span |
| outside, still structured | mechanisms not yet modelled | stands above the surrogate null |
| outside, unstructured | noise | sits at the null |

**Measurement (proof 19).** Pure noise → 2.6 / 0.0 / 97.4%; a modelled tilt →
61.0 / 0.0 / 39.0%; an **unmodelled** shape → 1.7 / **68.4** / 29.9%. The
middle part is the only thing both unreached and reachable.

---

## 8a. The closing step: the residual over the tape's ideal response

**Design.** *"Using the final unknown residual, do an interpolation modeled
over the ideal response of the VHS tape, i.e. the maximum possible bandpass
of meaningful information ... which leads us to the absolute noise floor,
since our video tape contains less information that what is possible to be
stored on the capture data."*

**Proposition 8a.1 (the depth is capped by demagnetisation, not by the
optimum).** JVC VTG82063 §7.2 gives the optimum recording depth as
`d = λ/4`. The shortest wavelength a coating of depth `d` can hold is
`λ_min = 2π d`, so recording `λ` requires `d ≤ λ/(2π)`. Since
`1/(2π) = 0.159 < 1/4`, **the demagnetisation limit binds everywhere and the
optimum is never reachable.**

*Corollary.* The two have no fixed point: substituting `d = λ/4` into
`λ = 2π d` gives `λ = 1.57 λ`. That divergence is the physical statement, not
a defect in either law.

*The cost is a constant.* Both laws are proportional to the wavelength, so

```
10 log₁₀( (1/4) / (1/2π) ) = 1.96 dB
```

everywhere, and **the noise slope stays at −20 dB/decade** — only its level
moves. The floor at the SP carrier therefore sits at 39.1 dB, not the 41.1 dB
the `λ/4` law implied, which narrows the distance from the measured 33.1 dB
from 8.0 dB to **6.0 dB**.

**Measurement (proof 29).** Depth available 0.237 µm at 3.9 MHz against an
optimum of 0.372 µm; cost 1.96 dB; slope −20.0 dB/decade.

**Proposition 8a.2 (why weighting by the ideal response is not a choice).**
The capture holds five to nine times what the tape has to say (§8), so the
capture's band is strictly wider than the tape's. Any part of the residual
that the tape's own response could not have carried is therefore **not tape
information**, and weighting the remainder by the ideal response discards
what the medium could not have recorded rather than smoothing what it did.

**The partition that closes it.** What remains after that is *inside* the
band, *unmodelled*, and does *not* reconstruct under band-limited
interpolation — so it has nowhere else to be, and is the absolute floor.

**Measurement (proof 29).** With only modelled mechanisms present: 84.4%
in band, 0.9% reconstructable, **83.4% floor**. With a smooth unmodelled
shape added: 85.9% in band, 29.8% reconstructable, **56.1% floor** — the
interpolation recovers what no model reaches, and the floor falls by exactly
what it recovered.

## 8b. The magnetic residual: record level, and where its shape comes from

**Design.** *"I think the phsical models we used are extra dimensions into
the magnetic information, such as record level for example. There should be a
shape that defines that against the curve the tape will exhibit."*

**Proposition 8b.1 (record level is NOT a new frequency shape).** Record
level reaches the response through two physical paths, and both land inside
the family that already collapsed:

| path | coherence with an existing mechanism |
|---|---|
| through the recording depth | **0.9995** with thickness loss |
| through the transition length | **1.0000** with spacing loss |

Adding both to the five-mechanism family moves the effective count from 1.56
to **1.40** — it *lowers* it, because a second copy of a direction already in
the span only worsens the conditioning. As a frequency shape, record level is
another value of the same dimensionless group.

**Proposition 8b.2 (its LEVEL DEPENDENCE is a new axis, and the cap is why).**
Taking the derivative of the response with respect to level and asking how
many independent shapes five levels produce:

| depth law | with the self-demagnetisation cap | without it |
|---|---|---|
| linear in level | **2.90** of 5 | **1.01** |
| saturating, exponential | 2.28 | 1.01 |
| saturating, tanh | 1.93 | — |

**Removing the cap collapses the axis to exactly one direction.** So the whole
information content of the level axis comes from the cap `d ≤ λ/(2π)` binding
at *different frequencies for different levels* — the crossover between
level-limited and cap-limited recording moves along the band as the level
changes, and that movement is the shape.

**And magnetic saturation REDUCES it**, from 2.90 to 1.93–2.28, by
compressing the range of depths the levels explore. That is the opposite of
what one would guess.

**Two errors on the way to this, both caught by a control.** The first
estimate read 4.65 of 5 and was wrong: the depth was written as
`cap × g(level)` with `cap = λ/(2π)`, which makes `x = 2πd/λ = g(level)`
independent of frequency, so the response was a constant, its log-shape was
undefined, and normalising a near-zero vector measured numerical noise. A
linear control that should have read 1.00 read 4.97 instead, which is what
exposed it. Written physically — a depth in metres, capped — the control
behaves and the numbers above stand. **A control that cannot fail is not a
control**, and this one earned its place twice.

## 9. What is knowable

### 9.1 Representable is not knowable

**Design.** *"any possible source of sinusoids … would include pure rf
background noise. So that by definition cannot be knowable."*

**Proposition 9.1.** An orthonormal basis of `k` vectors in `ℝ^L` captures, in
expectation, exactly `k/L` of the energy of an isotropic random vector.

*Proof.* By isotropy the expected squared projection onto each basis vector
is `1/L` of the total. ∎

**Measurement (proof 16).** 8 of 256 spans 1.2% (equal share 3.1%); 64 →
21.1% (25.0%); 200 → 77.1% (78.1%); **256 → 100.0%**.

**Corollary 9.2.** A complete basis represents noise perfectly and predicts
none of it. Spanning is not knowing — and this is exactly how a construction
whose entries spanned the whole space came to report noise as structure
(§10.1).

### 9.2 The curve is a guess; the key is not

The ellipse is *fitted*, so its directions must be earned on evidence not
used to fit them. The key is *specified* — standards fix the shapes, the
tolerances, and the **order** of application — so its synthetic side needs no
estimation. **Only the sizes on a particular tape are guessed.**

Two instruments enforce this: `guess_credibility` (split-half on the
ensemble; a real departure reads 0.589, one planted in a single bank reads
0.005) and `surrogate_null` (§10.1).

### 9.3 Which inner product

**Proposition 9.3.** Under `⟨u,v⟩ = uᴴv`, a pure-amplitude signature `r` and a
pure-phase signature `jr` satisfy `|⟨r, jr⟩| = ‖r‖²` — they are the *same*
direction.

Correct when the complex value is one measurement. **Wrong** when each
component is a real parameter's signature, where a gain and a delay are two
mechanisms. Then the space is `[Re; Im] ∈ ℝ^{2L}`, in which they are
orthogonal. **Measurement (proof 7):** rank 1 against rank 2.

---

## 10. Failure modes, as theorems about the construction

Each of these produced a wrong published number. They are stated as results
because each is reproducible on demand (proofs 12–15).

### 10.1 The nested matrix defeats every analytic statistic

**Theorem 10.1.** On the nested matrix of Definition 2.2 with all pairs
formed:

1. `rank ≤ N`, with equality generic — so `rank` reports the component count;
2. the nonzero eigenvalues average `M/N` with `M = N(N+1)/2`, which exceeds
   the Marchenko–Pastur edge for all sizes of interest — so `significant =
   rank` identically;
3. `bulk = median(sub-edge eigenvalues) = 0`, because the `M − N` structural
   zeros dominate — so `resolved = Σλ = tr G = N`, i.e. **100% by
   construction**;
4. once `significant = rank`, `removed = M`, `effective` clips to 1, and
   `asymmetry` returns the literal `0.0`.

**Measurement (proof 12).** Structureless noise at the published sizes
returns rank 13, 12 directions, **99.1% resolved, +37.9σ** against the
reported 92.9% and +39.3σ. The run was reproduced by noise.

**Theorem 10.2 (the statistic is blind).** On a nested matrix the asymmetry
does not respond to a shared departure at all.

*Proof.* A shared departure `c` present in every `d_i` cancels exactly in
every `D_ij = d_i − d_j`. It therefore enters only the `N` diagonals, of
`N(N+1)/2` entries. And since every row is unit-normalised, scaling the
diagonals does not move their directions. Hence `G` is unchanged. ∎

**Measurement (proof 15).** Diagonal energy 124 → 419 → 2185 as the departure
grows; off-diagonal energy **unchanged at 249.6**; asymmetry 0.8862 at ×4
against pure noise's 0.8862, `z = +0.2` in both.

**Corollary 10.3.** The remedy is not a better null on this matrix. It is to
fit the ellipse over **transfers** (§3) rather than differences.

`surrogate_null` supplies the only valid null for a constructed ensemble:
generate the same construction on structureless input and compare.

### 10.2 A shape measure must be told its dimensionality

Deflating a direction drawn from the ensemble's own span leaves an exact zero
eigenvalue. Counting it reports the deflation's bookkeeping as structure. Not
fixable by dropping zeros — a genuine needle has `N−1` of them and that *is*
maximal asymmetry. The caller passes `removed`.

### 10.3 The scatter, and what "saturated" does not mean

Corollary 6.2 gives the real-ensemble scatter as `≈2/L`, and when `N → L` the
asymmetry is pinned near its ceiling and cannot fluctuate, so the true
scatter collapses by 3× to 245×. The earlier reading that the amplitude and
field axes were "saturated, not empty" does not follow: against the correct
scatter those ensembles clear three sigma.

### 10.4 Two more, recorded

- **Grid-blind projection.** Inner products across two abscissae are memory
  layout, not coherence (proof 5).
- **Phase silently discarded.** `np.asarray(complex, dtype=float64)` keeps the
  real part. Not a precision loss: asymmetry 0.248 → 0.876 (proof 6).

---

## 11. What is Ethan's, and what is measurement

**His, as design:** that dimensions are Wiener transforms of complex pairs;
that the ellipse is the differentiation between dimensions; that the constant
is the ellipse; that the curve is only ever guessed; that the limit is where
the residual stops increasing; that the total eigenvalue is the information;
that termination is the circular shape of the complex signal; that
randomness is itself a residual wherever it has a shape; that the key is all
possible modifications to the video signal; that a fixed number of components
exist; and that the final floor — pure RF background — is unknowable by
definition and lies below the tape's.

**Derived here:** Theorem 4.2 and its remark; Lemma 3.2; Theorem 6.1 and
Corollary 6.2; Theorem 7.1 with Remarks 7.2–7.3; Propositions 8.1–8.3;
Proposition 9.1 and Corollary 9.2; Proposition 9.3; Theorems 10.1–10.2.

**Established by measurement,** with the proof that regenerates it: all 38
checks in `docs/PROOFS.md`.

**Corrected by measurement:** four claims that did not survive — the nested
matrix's statistics (§10.1), the flat halting count, the "saturated" reading
of two axes, and the vertical-interval result, which was reading active
picture two lines past the interval.
