# The frequency derivation

*Ethan's algorithm, written out once so it stops being reconstructed from
fragments. Recorded 3 September 2026, in his own framing, with the guard on
each step and the measurement that justifies it.*

---

## The premise

A measured response is a sum of frequency components. **Which components
are present is knowable exactly. Their magnitudes are not** — those have to
come from the residual.

That asymmetry is the whole design. The discrete cosine transform gives the
frequency support: on a finite segment whose endpoints differ it is a
complete orthogonal basis, and unlike the Fourier transform it spends no
coefficient on a boundary discontinuity that is not in the signal. What it
does not give is how much of each component the channel actually applied.
That is measured, by correcting and looking at what is left.

And one property governs everything that follows:

> Since each correction will affect all the other frequency responses, it is
> critical that this residual is measured and corrected **as a whole**.

Correcting one component changes the residual every other component is
measured against. So the amplitudes are solved **together**, in one system,
never one at a time. This is the single most important sentence in the
method and it is what distinguishes it from the greedy peeling the code
does today.

---

## The algorithm

### 1 — Fold

Accumulate the measurement over many lines and many fields into a mean
shape, with its per-lag variance.

**Alignment comes first, before anything else touches the data.** The DCT
has no shift theorem: a sub-sample timing error does not appear as a phase,
it appears as a *change of shape*, and the derivation would then be fitting
components to its own misalignment. Measured on a channel-shaped pulse:

| shift | change in DCT coefficients | change in DFT magnitude |
|---|---|---|
| 0.5 sample | 2.64% | 0.00% |
| 2.0 samples | 10.49% | 0.00% |

Each line is therefore shifted onto a common origin by a frequency-domain
phase ramp before it enters the average.

### 2 — Difference

    residual = measured − reference

The reference is what the signal should be: piecewise flat at the measured
levels, stepping at each measured transition with **that edge's own**
measured rise width. It must contain the decoder's own band limit, or the
fit reads the edge's finite rise as an artifact.

**A difference, never a ratio.** Spectral division blows up wherever the
denominator is small, and that is precisely the mechanism that manufactured
phantom features in this project before. The differential is a subtraction.

### 3 — Identify the frequencies

Take the DCT of the aligned residual.

This gives the frequency support exactly, and its energy compaction gives
the **rank** — how many components are actually present, rather than how
many the window could hold. Measured on the accumulated sync pulse: 34
coefficients of 120 hold 99.9% of the energy, with a spread of 1.3 across
26 independent cases. That is what one response has to carry.

The DCT is chosen over the DFT here for three reasons, all of which matter
on a short segment: it is real, so there is no phase to unwrap; its even
extension puts no discontinuity at the boundary; and for a smooth,
strongly-correlated segment it is close to the Karhunen–Loève transform, so
it packs the signal into the fewest coefficients of any fixed basis.

### 4 — Solve the amplitudes, jointly

The frequencies are now known. The magnitudes are not.

Solve **all** component amplitudes together against the residual, by
weighted least squares on that known support. Not one at a time, not by
deflation, not greedily — in one system, because each component's
correction moves the residual the others are measured against.

Two components close enough in frequency to overlap within the window's
resolution will be mis-split by any sequential method: the first one fitted
takes energy belonging to the second, and the second is then fitted to a
residual that has been distorted by that error. A joint solve has no such
ordering, and it is the reason this step is stated the way it is.

### 5 — Derive the phase

> It will identify the exact frequencies which, with the amplitude, will
> determine their phase.

For a minimum-phase channel, gain determines phase — the Bode gain–phase
relation. Frequency support plus solved amplitude therefore yields the
phase without measuring it, by real-cepstrum reconstruction.

What that route cannot give is **excess phase**: pure delay, and any
non-minimum-phase content. That is the one quantity which must come from a
transform that has a phase at all, and so it comes from the DFT's argument
— measured phase minus minimum phase is the excess.

### 6 — Correct

Apply the solved response. Take the new residual.

### 7 — Iterate

Re-identify and re-solve on the new residual.

The support is allowed to grow: components emerge from underneath larger
ones once those are removed, and a component that was invisible in the
first pass is a real component, not an artifact of the second. What is *not*
allowed is to keep an amplitude from an earlier pass — every pass re-solves
the whole set, for the reason given in step 4.

### 8 — Stop at the limit

Not when a budget runs out. Not when a candidate list empties. **When the
held-out residual is structureless and at the noise floor.**

Three readings decide it, and the first is decisive: whether the residual
reproduces on lines the fit never saw; whether its spectrum is flat; and
whether it is uncorrelated line to line. A residual that still has stable
structure means there is another component. A residual that is structureless
means the derivation is complete for that recording, and no amount of
further correction can remove what is left, because random noise cannot be
estimated.

A pass that takes the residual further from its limit is not kept.

---

## The two blocks

> The luma and colour-under originate from the RF. Then we perform the
> channel identification there correctly. We then apply the VCR de-emphasis
> controls **and** the picture correction in one block after demodulation.
> The de-emphasis controls will influence the picture correction, so these
> steps need to feed into each other, and each separate residual component
> will be stored and derived out the entire stage.

**Block 1 — RF.** Both the luma and the colour-under originate here, before
the paths separate, so the channel identification is done here and done
once. Everything before the demodulator is linear and time-invariant on the
RF, which is what allows a single response to stand for the whole of it.

**Block 2 — after demodulation.** The VCR de-emphasis and the picture
correction are one block, solved together, because de-emphasis changes what
the picture correction sees. Each component's residual is stored and derived
across the whole stage rather than per stage.

The demodulator is the boundary between them. It is not linear, so nothing
on one side composes with anything on the other, and a response identified
in one block cannot be carried into the other.

---

## Why this is not the method that failed here

Two measured refutations stand in this codebase against deriving amplitudes
from the frequency domain, and they must be read carefully, because they do
not apply to the method above:

> Identification and validation only — never amplitude fitting: a truncated
> window's spectral **division** manufactures phantom features that survive
> averaging.

> Seeding the candidate list from spectral **peaks** was tried and measured
> to certify burst-skirt and FM-remnant structure … with runaway numerator
> pairs that drove the split-half cross-validation negative — the model
> memorised accumulated noise.

Both are refutations of things this algorithm does not do. Step 2 takes a
difference, not a ratio, so there is no denominator to blow up. Step 3 takes
a complete orthogonal basis, not the peaks of a spectrum, so there is no
selection to bias.

The guard those failures argue for is kept regardless: **every amplitude is
scored on held-out lines**, and split-half cross-validation is what decides
whether a pass is kept. That is the instrument that caught the earlier
failure, and it is the instrument this method is judged by.

---

## What this replaces

The picture stage today identifies components by matrix pencil and then
removes them one at a time, deflating the residual after each:
`ring_stacked -= peel_prediction`. It stops when a slot budget is spent or
when no further candidate certifies.

Three changes follow from the algorithm above:

1. The amplitude solve becomes **joint**, over the whole support at once.
2. The stopping rule becomes the **exhaustion test** — held-out residual
   structureless and at the floor — rather than a budget.
3. The frequency support comes from the **DCT**, with the matrix pencil kept
   as an independent cross-check on the frequencies it finds rather than as
   the sole identifier.

The folding, the sub-sample alignment, the spec-derived measurement windows
and the per-lag variance are all already correct and are reused unchanged.

---

## One question left open, to be answered by measurement

Whether the separately-identified channel baseline is still needed at all:

> I don't know if we need this, if we are doing the DCT and residual
> measurement to know the exact magnitude of each frequency identified. I
> think the DCT and the frequency response residuals need to take to the
> limit together to best fit both components.

This is not assumed either way. The loop is run with the identifier's
response as the reference and again without it, and whichever reaches the
lower held-out floor is the answer. If the DCT and the residual converge on
their own, the identifier is redundant, and the measurement will say so.
