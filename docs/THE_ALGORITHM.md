# The complex multi-dimensional elliptical curve transform

Ethan's algorithm, stated once, precisely, in the order it runs. He defined
it; this is the definition written down with the mathematics beside it and
the implementing function named. Findings and measurements live elsewhere —
`ELLIPTICAL_COLLAPSE.md` for the numbers, `COMPONENT_MAPPINGS.md` and the
`*_COLLAPSE.md` documents for what it says about each domain.

Notation: `N` components, each a vector of length `L` on a shared grid.
Everything is complex.

---

## 0. Objects

A **component** is a measured quantity with three things attached:

| | |
|---|---|
| **synthetic** `s_i` | what the specification or the physics says it should be |
| **measured** `m_i` | what the chain actually produced |
| **place in the graph** | its parents, children and siblings |

An **axis** is the variable a component is a function of. Four exist:
amplitude, frequency, time, and the per-field dimension.

A **period** is the interval over which a component repeats: line, track,
field, drum revolution.

---

## 0a. A dimension is not an axis

> The "dimensions" are the Weiner transform of each complex pair (i.e. luma
> frequency vs. luma amplitude).

An **axis** is one measured channel. A **dimension** is manufactured by
holding TWO channels together on one abscissa and taking the Wiener transfer
between them:

    T(f)      = S_yx(f) / S_xx(f)                    the transfer, complex
    gamma2(f) = |S_yx|^2 / (S_xx S_yy)               how believable it is
    se(f)     = sqrt( (S_yy/S_xx)(1 - gamma2) / n )

`|T|`, `arg T` and the frequency it lives at are the *"amplitude, phase, and
frequency of this differential pair"*. Three channels give three pairs, hence
the three sets of three.

`pair_dimension.pair_transfer`, `pair_dimension.dimensions`.

**The grid problem dissolves because the repository already solved it.**
`residual_channels` places amplitude, frequency, carrier and time on the same
4fsc grid of the final time base correction, with identical shape, so an
index means the same instant in all of them.

**One worked example is already in the tree.**
`head_switch.measured_transfer` returns a complex `g` in "Hz of demodulated
frequency per neper of residual log-amplitude" — luma frequency per luma
amplitude, complex, per band.

**Four things this gets right that a first attempt did not**, each found by
adversarial review and each a way the measurement lies rather than a matter
of taste:

1. **The ellipse is fitted over the transfers, NOT over their differences.**
   Differencing dimensions imports the form of step 1 without its content —
   there the subtraction is expected-relation minus actual-relation and both
   sides are the same kind of thing, whereas a bare difference of two
   measurements cancels exactly the shared mechanism the ellipse exists to
   find. Three channels also give only three differences, and they sum to
   zero, so the ensemble would arrive rank-deficient by one.
2. **Coherence must be debiased.** With `n` segments an uncorrelated pair
   returns `1/n`, and exactly 1 at `n = 1`. Measured: raw coherence 0.5029
   at two segments, 0.0156 at sixty-four, matching `1/n` in both cases.
3. **Never multiply the pair by `W`.** It carries units of y per x, so
   applying it to both halves is dimensionally incoherent, and for real
   halves it is one real kernel acting on each and cannot separate them.
   Carry `gamma(f) Z(f)` instead.
4. **Correlated places shorten the effective length.** `sphere_floor` rests
   on `E|G_ij|^2 = 1/L`, which assumes independent places; the count that
   belongs there is the participation ratio of the weights.

Validated on a known one-pole transfer: **2.7 per cent in gain and 1.58
degrees in phase** over the coherent band.

## 1. Differentiate

> The differentiation is the differential of our synthetic components to the
> actual measured componnet, this has nested differentials in it as well
> according to our graph, so this expands out to a multi dimensional matrix
> that represents our known components.

**Diagonal** — each component against its own specification:

    d_i = s_i − m_i

**Off-diagonal** — each related pair, expected relationship against actual:

    D_ij = (s_i − s_j) − (m_i − m_j) = d_i − d_j

Formed **only** where the graph relates the pair: parents, children, and
siblings sharing a parent. A difference between unrelated components is not a
relationship anyone claimed.

`component_differentials(synthetic, measured, axis, relations)`
`graph_relations(declared)` reads the relations from `pipeline/stages.toml`.

The matrix is rank-deficient by construction, since `D_ij = d_i − d_j`. Its
deficiency must be measured once and carried, not charged to anything later.

---

## 2. Match

> as many matched components are differentiated together … all the way down
> for all matched sets of components

Two components are **matched** when they share an axis **and** their periods
are commensurate. Matching is on the axis and the period, never on the name.
Incommensurate periods are not differenced; the difference would be an
aliasing artefact rather than a component.

`_grouped_directions` groups by grid; only components sharing one are
projected together.

---

## 3. Fit the ellipse

> The constant is the elipse. All I need is to fit these residuals to an
> elipse of how ever many dimensions I have, and I have fully collapsed the
> differentiation.

Unit-normalise each entry, `u_i = D_i / ‖D_i‖`, stack them as the rows of
`U`, and form the quadratic form whose level set is the ellipsoid:

    G = U · Uᴴ            (Hermitian, N × N)
    G = V Λ Vᴴ            (eigendecomposition)

`Λ` are the squared semi-axes and `V` the principal directions.

**The invariant.** Every row is a unit vector, so

    trace(G) = N

for any ensemble whatever. The total is fixed; only the **shape** carries
information. That is the sense in which the ellipse is the constant.

**Which inner product.** `G = U·Uᴴ` treats the complex value as one
measurement, correct for a measured RF residual where amplitude and phase are
two parts of one quantity. When each component is instead the signature of a
**real** physical parameter, stack real and imaginary parts as a real vector
of length `2L` first — otherwise a gain and a delay of the same shape are one
direction rather than two.

`ellipsoid(residuals, axis, removed=, real_parameters=)`

---

## 4. Read the shape

Three quantities come out of `Λ`, answering three different questions.

**How much information is here.** Every component contributes one unit:

    information = N
    resolved    = Σ (λ_k − bulk) over the significant k

> the total eigenvalue is the total amount of information we could possible
> derive from our components

**How far from a sphere.** With `p` the participation ratio of the
eigenvalue shares and `n` the directions still available:

    p         = 1 / Σ (λ_k / Σλ)²
    asymmetry = (n − p) / (n − 1)              ∈ [0, 1]

Zero is a sphere — every direction carries the same. One is a needle — `N`
views of one thing. `n` is `N` minus the directions already removed, because
a shape measure must be told the dimensionality it is judging or it measures
its own deflation.

**How many directions are real.** Under pure noise the spectrum reaches the
Marchenko–Pastur upper edge and no further:

    edge        = (1 + √(N/L))²
    significant = #{ k : λ_k > edge }

Note the two are different questions. `significant` counts directions
standing **above** the bulk; an orthogonal family has none, because it *is*
the bulk, uniformly. To ask instead how many independent directions a family
spans, use the participation ratio.

---

## 5. Is it circular yet?

> This loops until the information floor is reached … Down to the circular
> shape of the complex signal.

A finite ensemble of random directions is never exactly spherical, so the
threshold is derived rather than chosen. From `trace(G) = N`,
`trace(G²) = N + Σ_{i≠j}|G_ij|²` and `E|G_ij|² = 1/L`:

    floor   = N / (L + N − 1)
    scatter = √2 / L     complex ensemble
            = 2 / L      real ensemble
    margin  = (asymmetry − floor) / scatter

**The scatter depends on the ensemble's kind**, and shipping one value for
both understated it by √2 on every real ensemble. `G_ij` is the inner
product of two random unit directions: real, that is ≈ N(0, 1/L), so
`|G_ij|²` is `(1/L)χ²₁` with variance `2/L²`; complex, the real and
imaginary parts carry half the variance each, so `|G_ij|²` is
`(1/L)(χ²₂/2)` with variance `1/L²`. The asymmetry is a function of
`Σ_{i≠j}|G_ij|²`, so its scatter is **√2 wider when the ensemble is real**.

Measured over 400 trials per cell, scatter × L: real 1.81–2.03, complex
1.20–1.43, ratio 1.37–1.61 against the predicted √2.

Understating the scatter **overstates the margin**, so the loop believes
itself further from the floor than it is and keeps going — it runs past its
own floor, fitting noise. That is the dangerous direction, and it applied
to every real ensemble, including `real_parameters=True`, which is the form
this document prescribes for real physical parameters (§3, "Which inner
product"). Corrected 2026-09-06; `ellipsoid` now reads the kind off the
stacked matrix rather than from the caller's flag, because real residuals
passed with the flag off are a real ensemble too.

**At the floor when `margin ≤ 3`.** More components raise the floor while its
scatter does not move, so the margin becomes better determined — which is why
adding components tells you more exactly when you have finished.

`sphere_floor(count, length, real=False)`

---

## 6. Differentiate the target curve onto the source

> Then differentiation using the target curve onto the source data.

The **target curve** is the whole ellipse, not one axis of it: every
direction above the noise edge, together.

    T = V[:, :significant]ᴴ · U           the directions in signal space
    orthonormalise T
    w_k = (λ_k − bulk) / λ_k              the weight for direction k
    D_i ← D_i − Σ_k w_k ⟨t_k, D_i⟩ t_k

`bulk` is the median of the eigenvalues at or below the edge.

**The weight is the correction-gain law.** With `λ = bulk + signal`,

    (λ − bulk)/λ = SNR/(1+SNR) = 1/(1 + ρ)

identical for every eigenvalue. A flat step is its special case at `ρ = 1`.
Diagonalising the second moment is the Karhunen–Loève transform and the
Wiener filter is diagonal in that basis, so this step is Wiener applied in
the basis step 3 found — self-calibrating, because the noise level is read
from the ensemble's own bulk.

**Take every direction at once.** Deflating one at a time, part-way, rebuilds
the recursion this collapse replaces, and it does not converge.

---

## 7. Repeat, and stop

Return to step 3 with the deflated entries. Terminate on the first of:

| | |
|---|---|
| circular | `margin ≤ 3` — the information floor |
| nothing stands | no eigenvalue above the edge |
| refused | the pass did not lower the asymmetry — roll it back |

`differentiate_to_floor(residuals, axis, amount=None, maximum_passes=)`

**The pass count is two, flat**, independent of how many dimensions there
are: one to remove the ellipse, one to confirm the circle. What tracks the
dimensions supplied is the number of directions recovered.

---

## 8. Run it on every axis

> Remember that all measurements and taking the limit happen in all measured
> dimensions simultaneously

Steps 1–7 run per axis: amplitude, frequency, time, and the per-field
dimension — the last being the transpose, each place a component and its
vector running over fields. `axes_present(residuals)` enumerates them.

---

## 8a. The fold on all dimensions: the tesseract, and the traversal back out

Added 2026-09-05 from Ethan's directives, verbatim in the order given:

> Oh I think the tesseract is asymmetric, and you never reach the full
> residual, since a hypercube has a finite number of edges.

> Build out this n dimensional tesseract that folds in onto itself, it is a
> graph that is connected to all of its dimensional siblings and parents,
> and all other relationships, the differentials are determined by the
> color and the luma together for the video part, and the sync pulses, and
> eq pulses, etc.

> The folding happens on all dimensions, previously we were only connecting
> it in sequence three ways, not the full dense graph.

> but across stages we know the path through this graph a depth we can
> trace ... Eventually it collapses down to one real signal which we can
> subtract to remove the residual

> Implementing a progressive multi-dimensional Hypercomplex Hilbert
> transform over a graph topology for iterative residual extraction and
> full related inverse signal deconvolution ... Each adjacent dimension of
> measurements get another dimension added to its hilbert transform. I run
> the signal through, it is hilbert transformed, related to the spec.
> Finally at the end, we take the inverse. Follow the tree back up and we
> come out with the subtracted out residual. ... nested tensor expansion
> using multi-axis Fourier slices

**Vertices.** Every two-state measurement axis (head A/B, falling/rising
edge, first/second half of the fields, one tape/another, luma/chroma) is
an axis of a hypercube; a vertex is one combination of states and carries
the complex log of the measurement made there with its noise variance.
`tesseract.Cube`.

**The fold.** Folding along an axis pairs each vertex with its sibling
across that axis and returns the parent (their mean, one axis fewer) and
the differential (half their difference). Folding on every axis in every
order is the Walsh-Hadamard transform of the cube - the Fourier transform
of Z_2^n - and gives 2^n - 1 contrasts, one per non-empty subset of the
axes, orthogonal and complete: the cube rebuilds exactly from them
(`tesseract.fold`, `walsh`, `reconstruct`). The sequential machinery of
section 6 took the order-one contrasts and one chain through the higher
orders; the fold on all dimensions takes all of them at once, which is the
dense graph: every pair of vertices is connected, and their difference is
the sum of the contrasts on the axes they differ in (`dense_graph`).

**The hypercomplex signal is the same object.** The n-dimensional
hypercomplex analytic signal has 2^n components, the signal and its partial
Hilbert transforms along every subset of axes; on a two-state axis the
partial Hilbert transform is the contrast. So the traversal is: expand by
one axis at a time (each doubles the component count), take the Bode
relation along frequency to split each contrast into its minimum-phase and
excess-phase parts, relate to the specification by subtracting the ideal's
components, keep the contrasts that stand above their own noise, and fold
back up the tree to one real departure per vertex whose `exp(-departure)`
is the real kernel subtracted from the capture (`hypercomplex.expand`,
`causality`, `relate_to_spec`, `collapse_back`, `deconvolve`;
`tesseract.one_real_signal`).

**The path has a depth.** The folds commute, but the chain does not: each
axis is a property of one stage of the chain and a contrast on several axes
is removed at the depth of the deepest stage it touches, shallowest stage
first (`tesseract.trace`, `AXIS_STAGE`, positions from
`interference.full_chain`).

**The finite number of edges.** A contrast is identified when its power
stands above its noise; below that it stays in the residual. Every
contrast's noise is fixed by the cube's vertex count, so the residual after
the fold is the sum of the contrasts the cube is too small to identify,
lowered only by more vertices, never by more folding of the same ones. When
the top-order contrast itself stands above noise the structure has more
dimensions than the cube has axes (`unreached`, `asymmetry`).

**Measured.** Head x polarity x half x tape on the sync exports: all 15
contrasts identified, the four-way included (z 23 to 324); sides unequal by
5.7 to 55 times; every axis's departure mostly excess phase once the
contrasts are placed back on the 4 f_sc grid before the cepstrum - the
polarity contrast a +48 to +128 ns delay, the tape contrast -126 to +78 ns,
0.4 to 1.1 rad of all-pass after the delay, and a minimum-phase share at or
below zero, the signature of a symmetric (zero-phase) shape - so the
dimension the frequency key lacks is time. The chroma burst, read per line
against the phase the decoder imposed, gives a per-head contrast of 0.6 per
cent and 0.06 degrees reproduced on two decodes from different seek points,
of the opposite sign to the luma's in the same band: a channel-specific
per-head gain, not a shared spacing.

**The time axis, folded by the bits of the field index.** A per-field
measurement over 2^k consecutive fields is a cube whose axes are the field
index's bits - bit 0 the head, bit 1 the drum revolution, each higher bit a
time scale twice the last - and the fold on all of them is the sequency
spectrum over field time (`tesseract.from_field_series`). On 256 fields of countdown
(4.27 s, decoded at the recorded 40 MSps flags) every scale from the head
(z 794) and the drum revolution (z 155) to two seconds (z 1020) carries
structure, flat at 40 to 48 times the noise from a quarter second upward:
the response drifts across the transport's whole band. The frequency key
could not see this; this is the dimension it lacked. On 1024 fields of home (17.08 s) the reel band
becomes a line: 33 times the median at 0.293 Hz inside the reels' own
0.118-0.442 Hz. The drum cannot be read from a per-field series at all -
it turns once per two fields, which is exactly that series' Nyquist, so it
aliases onto the head alternation. (A first run on home at the wrong
sample rate was withdrawn.) The residual-to-floor test (`residual_floor`) says
unrecoverable with structure on every tape and head: 24 to 39 dB above the
standard-error floor, not white, reproducing on the held-out half with an
agreement of 0.77 to 1.00.

## What terminates the whole thing

Three floors, in ascending order of how far they let you go:

| floor | what it is |
|---|---|
| the sphere | no direction stands above randomness of this size and length |
| the tape | the medium's own carrier-to-noise |
| the capture | the converter's word length and rate — *"capture card profile → complete limits of the sample rate"* |

No component may claim a residual below the capture's floor, because below it
there was no information to have measured.

---

## The curve is a guess; the key is not

Two things in this method have opposite epistemic status, and keeping them
apart resolves what otherwise looks like a contradiction.

> You only "guess" the curve, you don't know it.

> ... the complex convolusion over n dimensions along with their known
> shapes, which we have modeled using the sync pulse, vits, rf, colro
> carrier, genlock, etc. Those are all known and all have an order and
> strict compliance to them.

**The curve is fitted, so it is a guess** and must earn its directions on
evidence it has not seen — `guess_credibility`, and for a constructed
ensemble `surrogate_null`.

**The key is specified, so it is not.** The components are standardised
objects whose shapes, sequence and tolerances are published, which is why
their synthetic side needs no estimation at all:

| what the standard fixes | example |
|---|---|
| **shape** | IEC 60774-3 table 3 gives the sub pre-emphasis at six frequencies and four input levels |
| **tolerance** | the same table, +-0.30 dB at low frequency to +-0.70 dB at 3 and 5 MHz |
| **amplitude to 0.1 IRE** | SMPTE EG 27 table 2, the 75/7.5/75/7.5 bars, with each bar's subcarrier phase |
| **timing** | SMPTE 170M: burst 40 +- 1 IRE, 9 +- 1 cycles, starting 19 cycles from the horizontal reference |
| **the ORDER** | SMPTE RP 86 section 2.1's normative chain: burst-amplitude modifier, then pre-emphasis, then one FM modulator |

That last row is Ethan's "order": the standards state the SEQUENCE the
modifications are applied in, not merely their shapes. A chain whose order is
published can be inverted in the right order, which is what makes the
traversal's stage ordering a fact rather than a modelling choice.

So the honest statement of the method's epistemics: **the shapes and their
order are known; only their sizes on this particular tape are guessed.**
That is a far smaller thing to have to validate, and it is why a key built
from standards is exact where an ellipse fitted to data is not.

## Representable is not knowable

> We get a really really good filter within the complex 2d plane given a
> number of dimensions encoded in the complex plane, i.e. any possible source
> of sinusoids. Which would include pure rf background noise. So that by
> definition cannot be knowable. That is the final noise floor. That is
> certainly below the noise floor of my vhs recording.

Both halves check out, and the first half is the reason the second matters.

**A basis of enough sinusoids represents anything, noise included, and
captures exactly its equal share and no more.** Measured on pure noise over
256 places:

| basis size | share of the noise it spans | an equal share would be |
|---|---|---|
| 8 | 1.2% | 3.1% |
| 64 | 21.1% | 25.0% |
| 200 | 77.1% | 78.1% |
| 250 | 97.8% | 97.7% |
| 256 | **100.0%** | 100.0% |

A complete basis represents noise perfectly and predicts none of it.
Spanning is not knowing, and **this is precisely how a construction whose
entries spanned the whole space came to report noise as structure** - the
withdrawal in `ELLIPTICAL_COLLAPSE.md` section 6c is this principle arriving
as a bug. A filter over "any possible source of sinusoids" will fit the
background exactly, which is why the surrogate null is not optional.

**And the floor that is unknowable by definition sits far below the one that
operates.** Over the 7 MHz Carson band:

| floor | level | above the fundamental |
|---|---|---|
| the tape's own noise - the operative one | −27.5 dBm | **77.9 dB** |
| the 8-bit capture, in band | −55.7 dBm | 49.8 dB |
| thermal kTB at 293 K | −105.5 dBm | 0.0 dB |

So the final noise floor is real, it is where Ethan places it, and it is
**78 dB below** the floor that actually stops this work. Nothing in the
restoration is limited by it. What limits the restoration is the tape, and
the tape stands 28 dB above the converter as well - which is why every
capacity comparison in this arc lands the same way from whichever side it is
computed.

## The two populations, and which one ends it

Directions above the edge are **knowable**. The eigenvalue bulk is **truly
random** and cannot be promoted to a direction however long it is measured —
only averaged, as one over the root of the count, and averaging never adds a
direction.

Randomness is nonetheless itself a residual wherever the disturbance has a
shape on some axis: a beat has a frequency even when its phase is random.
What survives every axis is one mechanism — the finite number of particles in
the volume the head reads.

The algorithm does not fail to describe that. **It identifies it, and stops.**

---

## The implementing functions, in order

| step | function |
|---|---|
| 1 | `component_differentials`, `graph_relations` |
| 2 | `_grouped_directions`, `_grid_key` |
| 3 | `ellipsoid`, `_stack_real` |
| 4 | `ellipsoid` → `information`, `resolved`, `asymmetry`, `significant` |
| 5 | `sphere_floor`, `ellipsoid` → `sigma`, `at_sphere` |
| 6–7 | `differentiate_to_floor` |
| 8 | `axes_present` |
| floors | `capture_profile.profile_of`, `binding_limit` |

All in `vhsdecode/models/`. The runnable tool is
`tools/ringing_measure/elliptical_collapse.py`.
