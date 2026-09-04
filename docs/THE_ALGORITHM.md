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
    scatter = √2 / L
    margin  = (asymmetry − floor) / scatter

**At the floor when `margin ≤ 3`.** More components raise the floor while its
scatter does not move, so the margin becomes better determined — which is why
adding components tells you more exactly when you have finished.

`sphere_floor(count, length)`

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
