# Elliptical curve estimation over the measured components

Ethan's algorithm, and what measuring it established. Written down because
the findings are significant and because the reasoning behind them is the
part that gets lost on a restart.

`docs/RESIDUAL_LIMIT_DESIGN.md` is the parent design and remains the
authority; this document covers the collapse specifically — what it is, what
runs it, and the numbers.


## 1. The algorithm, in Ethan's words

Assembled from the messages that specified it, in order:

> Complex differential over the period of as many matched components are
> differentiated together in a matrice of their component's differentials,
> etc. all the way down for all matched sets of components. The same
> dimension of components extend the same sets of component in all matching
> dimensions, an matrix that is expanding by a dimension each time it is run
> through this algorithm.

> This forms a 3 dimensional elipse of differentiation between thes
> dimensions which is calculus and can be simplified.

> The shapes of the dimensions are asymetric, and we can extrapolate how
> symetric they are based on their fit to our 3 dimensional eliptical plane
> which represents the final result. **The constant is the elipse.** All I
> need is to fit these residuals to an elipse of how ever many dimensions I
> have, and I have fully collapsed the differentiation.

> elliptical curve estimation using multi dimensional complex components

> Then differentiation using the target curve onto the source data. This
> loops until the information floor is reached (total possible information
> on the source signal) 4d eliptical curve I believe. **Down to the circular
> shape of the complex signal.**

> The differentiation is the differential of our synthetic components to the
> actual measured componnet, this has nested differentials in it as well
> according to our graph, so this expands out to a multi dimensional matrix
> that represents our known components.

> I think **the total eigenvalue is the total amount of information we could
> possible derive from our components.** I.e. this is multi component radio
> tuning.

> since radio is where we got our vhs signal, that is the limit

In one paragraph: form the differential of each synthetic component against
its measured one; nest the differentials the graph relates; fit an ellipsoid
to that matrix; take everything standing above the noise as the target curve
and differentiate it onto the source; repeat until what remains is a circle.


## 2. Why it is a collapse and not an acceleration

The rank-increasing recursion — differentials of differentials, all the way
down — converges on the eigenbasis of the ensemble's second moment. Solving
that eigenproblem reaches the same place in **one step**: no passes, no chain
order, no half-steps. That is what "I have fully collapsed the
differentiation" means, and it is correct.

**The constant is the ellipse, exactly.** Every component enters as a unit
direction, so `trace(D · Dᴴ)` is the component count whatever the components
do. Measured on four complex components:

| ensemble | rank | trace | asymmetry | semi-axes |
|---|---|---|---|---|
| four independent | 4 of 4 | 4.0000000000 | 0.029 | 1.118 0.974 0.966 0.932 |
| four views of ONE departure | 1 of 4 | 4.0000000000 | 1.000 | 2.000 0.000 0.000 0.000 |
| shared plus private | 4 of 4 | 4.0000000000 | 0.248 | 1.350 0.906 0.834 0.813 |

Conserved to ten decimal places in all three, so only the *shape* carries
information. A sphere means the components are independent; a needle means
they are several views of one thing.


## 3. The three closed forms

None of these is a fitted constant, which is what keeps the stopping rule
free of tuning.

**The sphere floor — when is a shape circular?** A finite ensemble of random
directions is never exactly spherical. With `trace(G) = N`,
`trace(G²) = N + Σ|G_ij|²` and `E|G_ij|² = 1/L`:

    asymmetry of pure noise = N / (L + N − 1)
    its scatter             = √2 / L

Checked over N ∈ {3…32} and L ∈ {64…1024}, real and complex: measured over
predicted sits at 0.88–1.18 with no trend, and the scatter fit returns 1.408
against √2 = 1.414.

**The noise edge — which directions are real?** Under pure noise the Gram's
spectrum reaches the Marchenko–Pastur upper edge and no further:

    edge = (1 + √(N/L))²

Measured on noise ensembles the largest eigenvalue sits at 0.83–0.96 of the
edge and approaches it as L grows, so the edge is conservative: it never
admits noise as a direction.

**The weight — how much of a direction to remove?** A direction carrying
signal stands at `λ = bulk + signal`, so the share worth removing is

    w = (λ − bulk) / λ = SNR/(1+SNR) = 1 / (1 + ρ)

which is this project's own correction-gain law `a* = 1/(1+ρ)`, identical for
every eigenvalue, arrived at from the opposite direction. The flat half-step
is its special case at SNR = 1.

### 3.1 "Is this the Wiener transform for complex convolution?"

Close enough that it is the right name. Diagonalising the ensemble's second
moment is the Karhunen–Loève transform, and **the Wiener filter is diagonal
in the KLT basis** — the classical result. Fitting the ellipse finds the
basis; the per-direction weight is Wiener applied in it. Complex throughout,
which is what RF is.

The one qualification: a textbook Wiener filter needs the signal and noise
spectra known in advance, whereas here the noise level is read from the
ensemble's own eigenvalue bulk. It is a self-calibrating shrinkage estimator
of Wiener form — the same discipline as the rest of this arc, where the floor
is measured beside the signal in the same chain at the same moment.

It is measurably the right step. On twelve components at 512 points:

| planted | Wiener per direction | flat half-step |
|---|---|---|
| 1 | 2.0 passes, 1.0 recovered | 3.7 passes, 2.7 recovered |
| 2 | 2.0 passes, 2.0 recovered | 4.0 passes, 5.8 recovered |
| 4 | 2.0 passes, 4.0 recovered | 3.8 passes, 8.3 recovered |

A flat step leaves half of each direction standing, so the loop finds the
same direction again and reports two to eight directions where one to four
exist.


## 4. The halting count

Planting a known number of departures and running to the circle:

| components | planted | passes | recovered | reached the circle |
|---|---|---|---|---|
| 6 | 0 | 1.0 | 0.0 | 100% |
| 6 | 1–2 | 2.0 | 1.0, 2.0 | 100% |
| 6 | 4 | 1.0 | 2.0 | **0%** |
| 10 | 1–4 | 2.0 | 1.0–4.0 | 100% |
| 10 | 5 | 2.9 | 7.6 | 62% |
| 16 | 1–5 | 2.0 | 1.0–5.0 | 100% |

**The pass count does not grow with the dimensions — it is two, flat.** One
pass to remove the whole ellipse, one to confirm the circle, because every
direction above the noise edge comes out together. That is the collapse doing
its work. The quantity that tracks the dimensions supplied is the number of
directions *recovered*, and that equals the number planted exactly.

**Roughly `N − 2` dimensions are resolvable from `N` components**: six cannot
resolve four departures at all, ten resolve five 62% of the time, sixteen
resolve five every time. This is the mechanism behind *"you can increase the
constant, how many components, to know exactly when you have reached the
end"* — more components is literally more budget.


## 5. The information budget

> the total eigenvalue is the total amount of information we could possible
> derive from our components

Exactly so. Every component contributes one unit to the trace whatever it
carries, so **N components hold N units and no more**. What the fit adds is
how much of that budget stands in real directions:

| ensemble | components | total | resolved | share |
|---|---|---|---|---|
| six independent | 6 | 6.00 | 0.000 | 0.0% |
| six views of ONE departure | 6 | 6.00 | 6.000 | 100.0% |
| six, shared plus private | 6 | 6.00 | 2.440 | 40.7% |
| twelve, the same shared part | 12 | 12.00 | 5.166 | 43.0% |

Doubling the components roughly doubles what is resolved while the share
holds.


## 6. On real data

`tools/ringing_measure/elliptical_collapse.py`, on the 75-bars NTSC SP
capture, 27 fields:

```
fields harvested 27 (13 on one head, 14 on the other)
within-field envelope spread 0.697 dB per sample

CAPTURE PROFILE, read from the capture
  word length            8 bit (step 256 in the container)
  codes present          117, occupying 45.7% of full scale
  sample rate            50.0 MHz
  quantisation floor     49.9 dB full band, budget 400 Mbit/s

THE NESTED DIFFERENTIAL MATRIX
  head A: 13 fields over 119 shared bins -> 91 matrix entries
  head B: 14 fields over 120 shared bins -> 105 matrix entries
  both:   27 fields over 119 shared bins -> 378 matrix entries

THE ELLIPSE
  head A: rank 13, directions above the noise edge 11
          budget 91 units, 84.537 resolved (92.9%)
          asymmetry 0.9024 vs a sphere floor of 0.4354  ->  +39.3 sigma
  both:   rank 27, directions above the noise edge 14
          budget 378 units, 317.334 resolved (84.0%)

THE LOOP, TO THE FLOOR
  head A: 0.2678 -> 0.0046   circular; 11 directions in 2 passes
  head B: 0.1899 -> 0.0246   circular; 11 directions in 2 passes
  both:   0.7099 -> 0.0813   circular; 14 directions in 2 passes
```

The matrix's numerical rank equals the field count exactly (13, 14, 27), the
loop reaches the circle in two passes on all three, and the structure it
finds is real at 18–39 sigma above the sphere floor.


## 6a. The name, and the run in every dimension

Ethan named the method: **the complex multi-dimensional elliptical curve
transform**. That is apt and it is now the name used here. Each word earns
its place — *complex* because the components are RF and carry phase,
*multi-dimensional* because the ellipse is fitted over however many
components there are, *elliptical curve* because the fitted quadratic form is
the object, and *transform* because the eigenbasis is what the method
produces.

Run in every measured dimension at once, as R2 requires, on 27 real fields:

| axis | ensemble | entries | places | rank | directions | asymmetry to floor | reached |
|---|---|---|---|---|---|---|---|
| frequency | head A | 91 | 119 | 13 | 11 | 0.268 -> 0.005 | yes |
| frequency | head B | 105 | 120 | 14 | 11 | 0.190 -> 0.025 | yes |
| frequency | both | 378 | 119 | 27 | 14 | 0.710 -> 0.081 | yes |
| time | head A | 91 | 264 | 13 | **13** | 0.185 -> **0.000** | yes |
| time | head B | 105 | 262 | 14 | **14** | 0.224 -> **0.000** | yes |
| time | both | 378 | 258 | 27 | 19 | 0.893 -> 0.032 | yes |
| amplitude | head A | 91 | **9** | 9 | 2 | 0.832 -> 0.832 | saturated |
| amplitude | both | 378 | **9** | 9 | 1 | 0.924 -> 0.924 | saturated |
| field | head A | 7140 | **13** | 13 | 2 | 0.726 -> 0.726 | saturated |
| field | both | 7140 | **27** | 27 | 3 | 0.857 -> 0.219 | yes |

**Twelve of twelve ensembles reached the circle, recovering 86 directions in
total.** Two findings stand out and a third is a diagnostic rather than a
result.

**The time axis is completely structured.** Head A resolves 13 directions
from 13 fields and head B 14 from 14 — 100 per cent of the information budget
— at 114 to 118 sigma above the sphere floor, and the asymmetry falls to
exactly zero. Every field's time-base residual is a direction of its own,
with nothing left over. The time base is the most informative axis this
measurement has, by a wide margin over the frequency response's 84 to 93 per
cent.

**The amplitude and field axes are SATURATED, not empty.** Their sphere
floors read 0.83 to 0.998, which is what the floor formula `N/(L+N−1)` gives
when the component count far exceeds the number of places: 91 entries over
9 sweep bins, or 7140 entries over 13 fields. At that ratio a random ensemble
is already almost maximally asymmetric and nothing can stand above it — the
`N − 2` rule from section 4, seen from the other side. **The fix is more
places, not more fields**: a finer sweep binning for the amplitude axis, and
for the per-field dimension either more fields or fewer bins entering as
components. Reporting "0 directions" there without this reading would be
wrong.

## 6b. The part that can only be averaged out

Ethan: *"There are still however unknown components that canno be measured,
but only averaged out. Over time the differentiation will approach zero, or
the limit the information can store. Those are the truly random components,
radio noise."*

The transform already separates these two populations, and it does so by
construction rather than by a choice. **The directions above the
Marchenko-Pastur edge are the knowable part; the eigenvalue BULK below it is
the truly random part.** A component in the bulk cannot be promoted to a
direction however long it is measured — there is no shape there to find —
and the only thing that reduces it is averaging.

**Averaging never adds a direction.** Averaging `k` independent observations
into each component and refitting, with three departures planted:

| observations averaged | directions found | resolved | random share | 1/k would give |
|---|---|---|---|---|
| 1 | 3 | 4.921 | 59.00% | 59.00% |
| 2 | 3 | 6.881 | 42.66% | 29.50% |
| 4 | 3 | 8.635 | 28.04% | 14.75% |
| 8 | 3 | 10.036 | 16.37% | 7.37% |
| 16 | 3 | 10.910 | 9.09% | 3.69% |
| 32 | 3 | 11.421 | 4.83% | 1.84% |
| 64 | 3 | 11.707 | **2.44%** | 0.92% |

The direction count is **three at every depth** — averaging cleans the
directions that exist, it does not create new ones. The random share falls
monotonically toward zero, and it falls roughly as `1/k` with a shortfall of
about two and a half times at the deepest pooling, because the bulk level is
itself estimated from a finite number of sub-edge eigenvalues and so has its
own floor.

**What that share is on the real decode**, per axis, over 27 fields:

| axis | budget | resolved | structured | **truly random** |
|---|---|---|---|---|
| time / head A | 91 | 91.0 | 100.0% | **0.0%** |
| frequency / head A | 91 | 84.5 | 92.9% | **7.1%** |
| time / both heads | 378 | 346.5 | 91.7% | 8.3% |
| frequency / both heads | 378 | 317.3 | 84.0% | 16.0% |
| amplitude / head A | 91 | 75.4 | 82.8% | 17.2% |
| field / head A | 7140 | 3904.6 | 54.7% | **45.3%** |

Two readings worth carrying. **The time base has no random content at this
resolution at all** — every field's time-base residual is a direction, which
is why that axis resolves 100 per cent of its budget. And **the per-field
dimension is nearly half radio noise**, which is the honest statement of what
"the non-reducible residual tracked field by field" turns out to be: about
45 per cent of it is the channel, and no model will ever describe it.

This is also where the two limits in section 7 meet. The random part is the
tape's own carrier-to-noise, 21.6 dB, and averaging drives it toward the
floor the capture can store rather than toward zero — so the differentiation
approaches, in Ethan's phrase, *"the limit the information can store"*, and
that limit is the radio channel's rather than the converter's.

### 6b.1 Infinite averaging is impossible; the useful targets are finite

Ethan: *"The random components here are potentially infinite so you'd need to
average infinitely which we can't do on a computer."*

True at the extreme, and it is worth being precise about where the extreme
starts, because the targets that matter turn out to be cheap. The random part
falls as one over the root of the field count, so the material each target
needs is a computation rather than a guess:

| target | fields per head | seconds of tape | of a two-hour capture |
|---|---|---|---|
| the tape's own noise floor | 45 | 1.5 | 0.02% |
| half the tape floor | 178 | 6.0 | 0.08% |
| the 8-bit capture's floor | 32 500 | 1 084 | 15.1% |
| one tenth of the capture's floor | 3 250 000 | 108 442 | **1 506%** |

So **1.5 seconds of tape reaches the tape's own noise floor and eighteen
minutes reaches the converter's**, both well inside a two-hour capture. It is
the row after that which is unreachable: one further decade costs fifteen
captures of the same tape, which is Ethan's infinity arriving on schedule.
The practical statement is that the transform can be driven to the capture's
own floor and no further, and that is exactly the boundary section 7
identifies anyway.

**But infinity is not what stops it — stationarity is.** Averaging is only
valid across a span over which the quantity being measured does not change,
and this arc has measurements on both sides of that:

| span of tape | fields per head | random part reaches | against the capture floor |
|---|---|---|---|
| 1 s | 30 | 0.00988 dB | 32.9x |
| 10 s | 300 | 0.00312 dB | 10.4x |
| 60 s | 1 798 | 0.00128 dB | 4.3x |
| 540 s | 16 184 | 0.00043 dB | 1.4x |

The head differential does **not** drift — first half against second half of a
decode correlates at r = +0.902 and +0.958 — so it may be averaged over as
long a span as the material allows. The tape position **does** vary: the
position sweep found one frame carrying thirty-nine times its neighbours'
dropouts. Averaging across that blurs a real change into a smaller number and
reports it as noise removed.

So the honest limit is not "you cannot average forever". It is that **each
quantity has its own span over which averaging is legitimate, and that span
is itself a measurement.** A quantity that is static may be averaged to the
capture's floor in eighteen minutes; one that moves must be averaged only
within the span over which it holds still, and its floor is set by how many
fields fit in that span rather than by how long the capture is.

### 6b.2 The fixed-dimension argument, and what it costs

Ethan: *"Since the universe a universe size amount of cosmic radio
information, radio noise is effectly random and probably has a constant above
the total amount of digital information captured in there. You can't possibly
derive it all, since our datasize is at a fixed dimension, in our case 2
dimentions and over time."*

This is the correct closing statement and it can be given a number. The
capture is a **fixed** window — two spatial dimensions, line by sample, and
time — on a process with no such bound.

| | |
|---|---|
| the capture's fixed budget | 400.0 Mbit/s, 360 GB over two hours |
| what the converter carries in band | 124.1 Mbit/s |
| what the tape actually says in band | 57.4 Mbit/s |
| **of what the converter carries in band, noise** | **53.7%** |
| **of every bit the capture stores, noise or out of band** | **85.7%** |

Nearly six bits in every seven that this capture holds are not recoverable
signal. And the noise's degrees of freedom are one per sample —
**360 000 000 000** of them over two hours — each a number held once, drawn
from an instant that has passed and cannot be sampled again.

That makes the noise not merely unknown but unknowable in principle from this
capture. It is incompressible, because it is random; it is inextensible,
because the realisation is over; and it occupies its share of a budget that
is fixed by the converter and cannot be enlarged after the fact. Averaging
reduces its effect on an *estimate* as one over the root of the count, but
those three hundred and sixty billion values stay exactly where they are.

Which is the same conclusion the transform reaches from the other end. The
eigenvalue bulk is precisely this population; the Marchenko-Pastur edge is
precisely the line below which nothing can be promoted to a direction; and
the sphere is precisely the state in which nothing but this population
remains. The algorithm does not fail to describe the noise — **it identifies
it, and stops.**

## 6c. WITHDRAWN: the real-data numbers of section 6 are reproduced by noise

An independent adversarial review, reproduced and confirmed here, established
that **the run reported in section 6 is reproduced number for number by
structureless noise through the same construction.**

| ensemble | reported | pure noise, same construction |
|---|---|---|
| frequency / head A | rank 13, 11 directions, 92.9% resolved, +39.3 sigma | rank 13, 12 directions, **99.1%**, **+37.9 sigma** |
| time / head A | rank 13, 13 directions, **100.0%**, +118 sigma | rank 13, 12 directions, **98.9%**, **+116.3 sigma** |
| field / head A | rank 13, 54.7% | 52.2 +- 5.2% |

**The cause is the construction, and it can be named line by line.**
`component_differentials` emits `N(N+1)/2` entries spanning only `N`
dimensions, because `D_ij = d_i - d_j` is a linear combination of the
diagonals. Three consequences follow mechanically:

- **the rank is forced.** `rank <= N` always, with equality generic, so
  "the numerical rank equals the field count exactly (13, 14, 27)" is
  arithmetic rather than evidence;
- **every nonzero eigenvalue clears the Marchenko-Pastur edge**, because the
  nonzero ones average `M/N` = 7.0 against an edge of 2.52. `significant ==
  rank` is likewise forced, and the MP test has no discriminating power on a
  matrix whose rows are dependent;
- **the exact 0.000 asymmetry is a literal constant.** Once `significant ==
  rank`, `removed` equals the entry count, `effective` clips to 1, and the
  expression returns the `else 0.0` branch. The **100 per cent resolved** is
  forced the same way: the structural zeros make the median bulk exactly
  zero, so `resolved` sums every nonzero eigenvalue and equals the trace.

So the following are withdrawn: every rank, direction count, resolved
fraction and sigma taken on a nested matrix; "12 of 12 ensembles reaching the
circle"; "86 directions recovered"; and the reading in 6a and 6b that the
time axis is completely structured with no random content. **The difference
between the time axis's 100 per cent and the frequency axis's 92.9 was the
number of bins in the grid, not the tape.**

**What survives, measured against a construction-aware null.** Building the
null by generating the same construction on structureless input rather than
deriving it analytically, the pass-zero asymmetry is a genuine statistic and
there is real structure on every axis:

| ensemble | observed | null mean +- sd | z |
|---|---|---|---|
| frequency / head A | 0.268 | 0.1420 +- 0.0118 | **+10.7** |
| frequency / both | 0.710 | 0.2019 +- 0.0103 | **+49.3** |
| time / head A | 0.185 | 0.1013 +- 0.0067 | **+12.6** |
| time / both | 0.893 | 0.1186 +- 0.0060 | **+128.7** |
| amplitude / head A | 0.832 | 0.4084 +- 0.0440 | **+9.6** |

The method is finding something on every axis. It was the quantification
that was broken, not the detection.

`surrogate_null(residuals, axis)` now generates this null, and
`report_ellipse` passes `removed=` so the intrinsic deficiency is no longer
charged to the shape. **No statistic on a nested matrix may be reported
without it.**

Three further defects the same review established, each verified here:

- the scatter `sqrt(2)/L` is the COMPLEX large-`N` asymptote; a real ensemble
  gives about `2/L`, and when `N` approaches `L` the true scatter collapses
  by 3 to 245 times, so section 6a's "saturated, not empty" reading does not
  follow - against the correct scatter the amplitude and field axes clear
  three sigma;
- the planting helper in the tests gives every departure above the second the
  same linear ramp, so at most **three** independent directions exist however
  many are asked for. "Recovered equals planted" was checked against a ground
  truth that does not exist above three;
- on the real decode every Wiener weight is exactly 1.0, because the median
  of the structural zeros is 0, so the mechanism section 3.1 credits was
  never running there.

## 7. What actually binds — two answers, and neither is the algorithm

### 7.1 The radio, not the converter

> since radio is where we got our vhs signal, that is the limit

Confirmed. Over the band Carson's rule gives for the format —
2 × (1.0 MHz deviation + 3.0 MHz baseband) = 8.0 MHz:

| link | carrier-to-noise | Shannon capacity |
|---|---|---|
| the tape (radio) | 21.6 dB, from the measured 0.697 dB envelope spread | 57.4 Mbit/s |
| the capture card | 46.7 dB, from 8 bits at 50 MSps in that band | 124.1 Mbit/s |

**The converter holds 2.2× the tape's capacity and spends 67 Mbit/s on
nothing.** Better capture hardware buys nothing here; the tape's own
carrier-to-noise is the wall. The same verdict appears in the per-bin floors:
a tape floor of 0.0081 dB against an 8-bit floor of 0.00030 dB.

This confirms Ethan's premise that *"our fields contain information that the
bitdepth and bandwidth response of the capture device exceed"* — everything
the tape carries is present in the capture.

### 7.2 Identifiability, which is the real ceiling

The question was whether the algorithm can reach the medium's absolute SNR
"given how many components contributed to noise — i.e. how many possible
things could go wrong with the tape". It can reach the floor. It cannot say
what caused it, and the reason is not the algorithm.

**Nine mechanisms, six witnesses** (`vhsdecode/models/tape_model.py`):

| witness | mechanisms it sees |
|---|---|
| envelope_tilt | coating thickness, coercivity, surface roughness and debris, binder degradation |
| envelope_level | remanence |
| envelope_events | oxide shed and dropouts |
| azimuth | mistracking |
| time_base | stretch and deformation |
| noise_floor | print-through |

Seven of the nine are already recorded as confounded. Ethan's proposal was
that physics breaks the confound — *"we can even predict things based on the
magnetic profiles of the tape, track width, roll off"*. **Tested, and it does
not.** Perturbing each mechanism in `head_model.log_response` and comparing
the predicted signatures:

|  | spacing | gap | thickness | azimuth | contour | gain |
|---|---|---|---|---|---|---|
| spacing | 1.000 | 0.962 | 0.994 | 0.968 | 0.000 | 0.894 |
| gap | 0.962 | 1.000 | 0.926 | **1.000** | 0.000 | 0.761 |
| thickness | 0.994 | 0.926 | 1.000 | 0.935 | 0.000 | 0.931 |
| azimuth | 0.968 | **1.000** | 0.935 | 1.000 | 0.000 | 0.773 |
| contour | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.002 |
| gain | 0.894 | 0.761 | 0.931 | 0.773 | 0.002 | 1.000 |

Gap length and azimuth error are **exactly** collinear — both are sinc losses
in wavelength. Spacing and coating thickness sit at 0.994. Only the contour
effect is orthogonal, because it is an oscillation rather than a decay.

Singular values of the six signatures: `2.159, 1.000, 0.565, 0.133, 0.0078,
0.0002`, condition number 1.06 × 10⁴. One direction explains 77.7% of the
six mechanisms' span, two explain 94.4%, three explain 99.7%.

> **The effective number of distinguishable mechanisms is 1.58 of 6.**

So the ceiling is not the component count and not the converter. It is that
the mechanisms produce nearly the same shape, because every one of these
losses is a function of spacing over wavelength and so is nearly the same
exponential decay in frequency. **The total departure can be measured
precisely; which mechanism caused it cannot be recovered from a frequency
response alone.**

That is the honest form of the limit Ethan has been reaching for. Not that
the floor is unreachable — the loop reaches it in two passes — but that
beyond a point further description does not reduce the residual, because the
descriptions are not distinguishable given the evidence. Separating those
mechanisms needs a witness that is not a frequency response: a second tape, a
second deck, a different speed, or the contour direction, which is the one
that stands apart.


### 7.3 The tape mechanics are on a different axis, and they DO separate

Ethan: *"I still have all the tape mechanics to go through, including reel
tension issues etc."* This is worth doing, and the reason is that it does not
run into the ceiling above. The magnetic mechanisms collapse to 1.58 of 6
because they are all functions of spacing over wavelength and so are nearly
the same decay in frequency. The mechanical ones act on **time**, and there
they are separated by their RATES rather than their shapes.

`vhsdecode/models/transport_model.py` already predicts them from the format
and the geometry:

| part | rate | acts on | | |
|---|---|---|---|---|
| supply reel | 0.2801 Hz | tension | before capstan | a band, 0.118–0.442 Hz |
| impedance roller | 0.8166 Hz | tension | before | |
| entry guide | 1.7693 Hz | tension | before | |
| exit guide | 1.7693 Hz | tension | before | |
| pinch roller | 1.0616 Hz | speed | after capstan | |
| capstan | 4.8253 Hz | speed | before | |
| head drum | 29.9700 Hz | both | before | **certain** — fixed by the format |
| take-up reel | 0.2801 Hz | tension | after capstan | a band, as above |

The reels are the tension mechanism Ethan names, and the model already has
the important detail: their rate is a **band, not a line**, because the pack
radius grows as tape winds across, so the rate drifts through the tape. That
drift is itself an identifying signature.

**How much tape each needs.** Two rates separate once the observation is long
enough to resolve their difference, `1/T < |f₁ − f₂|`:

| observed | resolution | separable pairs |
|---|---|---|
| 0.23 s | 4.274 Hz | 9 of 28 |
| 1.0 s | 1.000 Hz | 17 of 28 |
| 5.0 s | 0.200 Hz | **26 of 28** |
| 120 s | 0.008 Hz | 26 of 28 |

**Five seconds gets 26 of the 28 pairs, and it saturates there.** The two that
never separate are degenerate by construction — the entry and exit guides
have the same diameter and therefore the same rate, and the two reels share a
nominal band — so those need the `after_capstan` and `acts_on` distinctions
the model already carries, not more tape.

The practical consequence is sharp: **every decode this arc has examined is
about 0.22 seconds**, the first row, where no mechanical rate is resolved at
all — the slowest has a 3.6 second period. The source captures are two hours
long. The mechanics work needs a longer decode window, not a longer capture,
and about five seconds of it.

## 7.4 The propagation ratio IS the final residual, and it has one dimension

Ethan asked whether the propagation ratio of radio itself is the final
residual, reducible to exact dimensions. External radio propagation is not —
it is not present at all, by 74 dB (section 7.5). But there is a propagation
ratio that governs everything measured here, and reducing it to exact
dimensions explains the identifiability ceiling completely.

Every loss mechanism in the head model is a function of one dimensionless
group: a length over the recorded wavelength.

| mechanism | form |
|---|---|
| spacing | `exp(−2π d/λ)` |
| gap | `sinc(g/λ)` |
| thickness | `(1 − exp(−2π t/λ)) / (2π t/λ)` |
| azimuth | `sinc(w·tanθ / λ)` |
| contour | `cos(2π ℓ/λ)·e^(−0.15 ℓ/λ)` |

**Five mechanisms, five different lengths, one dimensionless group.** Fitting
the ellipse to the five *shapes* as functions of `x = L/λ` gives an effective
count of **1.90 distinct shapes of 5** — the same answer as the 1.58 of 6
measured on the responses themselves, arrived at from dimensional analysis
rather than from data.

So the ceiling has a cause, and it is not statistical. As far as a frequency
response is concerned these are not five independent physical effects; they
are five values of the same variable. That is the exact-dimensions answer.

## 7.5 Where external radio noise actually sits

| noise source | level | above thermal |
|---|---|---|
| the tape, measured C/N 21.6 dB | −30.6 dBm | 74.3 dB |
| the 8-bit capture in band, 46.7 dB | −55.7 dBm | 49.2 dB |
| thermal kTB at 293 K over 8 MHz | −104.9 dBm | 0.0 dB |

(Absolute levels assume a 1 Vpp full scale, the usual figure for an RF
capture front end; every ratio is independent of it.)

Galactic noise dominates thermal below about 100 MHz, but only for a system
with an aperture to the sky, and this chain has none — a millimetre-scale
magnetic head inside a shielded drum feeding shielded coax. **Ethan's
expectation is right: external radio is silent here, by 74 dB.** What this
arc calls radio noise is the tape's own particulate and modulation noise — a
radio-channel *problem*, not radio-borne interference.

## 7.6 How much noise is unexpected, and what capacity we have

The raw envelope spread counts every departure as noise. The transform
separates the part that correlates across components from the part that does
not, which is exactly Ethan's criterion — *"random noise has no coorelation
at all with the radio signals being measured"*. That separation changes the
answer substantially:

| | |
|---|---|
| raw spread, all of it counted as noise | 21.6 dB C/N |
| of that, correlated across components | 92.9% — structure |
| uncorrelated | 7.1% — truly random |
| **the true random carrier-to-noise** | **33.1 dB** |
| the tape's own magnetics predict | 40.8 dB |
| **unexplained excess** | **7.7 dB** (was 19.2 dB before separating) |

The remaining 7.7 dB is within reach of the prediction's own assumptions —
recording depth, particle volume and packing fraction are all assumed — and
of modulation noise, the coating's own variation scaling with the signal,
which a simple particle count omits.

**The capacity, recomputed on the truly random part:**

| | Mbit/s |
|---|---|
| the converter's raw budget | 400.0 |
| in band, what the converter can carry | 124.1 |
| in band, what particle noise alone would allow | 108.4 |
| **in band, the tape on the truly random part** | **87.9** |
| in band, the tape on the raw spread (pessimistic) | 57.5 |

**The converter holds 1.41x the tape, not 2.2x** — correcting the structure
the transform found closes most of the gap that section 7.1 measured before
the separation was available. The tape still binds, but by a far narrower
margin than the raw figure suggested, and 78% of every captured bit remains
noise or out of band. That last figure is the fixed-dimension cost of section
6b.2, and nothing recovers it.

## 7.7 Randomness is itself a residual, and only one kind survives

Ethan: *"randomness is a residual here that we can effectively differentiate
over since we can model different types of radio interference, maybe that's
all the RF stage boils down to using all of our measured components."*

Tested, and it holds strongly. The transform calls a component random when it
does not correlate across the ensemble — a statement about one axis. A
disturbance can be uncorrelated field to field and still be perfectly
structured, because a beat has a frequency even when its phase is random.
Five disturbance types, every one given a random phase per field so that none
of them correlates trivially:

| disturbance | directions found | asymmetry vs floor | |
|---|---|---|---|
| **particle noise** (broadband, white) | **0** | 0.0301 vs 0.0304 | **−0.1 sigma** |
| beat / co-channel (a line) | 1 | 1.0000 vs 0.0304 | +351 sigma |
| head clog (a slow tilt) | 1 | 1.0000 vs 0.0304 | +351 sigma |
| dropout (impulsive, sparse) | 4 | 0.3978 vs 0.0304 | +133 sigma |
| modulation noise (sidebands) | 4 | 0.2773 vs 0.0304 | +89 sigma |

**Four of the five are not random to the transform at all.** Only particle
noise sits at the sphere floor, and it sits there exactly — within a tenth of
a sigma of where randomness of that size and length is expected to be.

So Ethan's reading is the right one. What a single measurement lumps together
as "noise" is mostly modellable interference that the transform separates the
moment it is given components on an axis where the types differ, and the
irreducible remainder is one specific mechanism: the finite number of
particles in the volume the head reads. That is the sense in which the RF
stage does boil down to the measured components — everything except particle
noise is a direction waiting for the right axis.

It also sets the next measurement precisely. The real decode left 7.1 per
cent of the frequency axis uncorrelated across fields; the table above says
that share should shrink further if the interference types are registered as
components in their own right, and should stop shrinking at whatever part is
genuinely particulate. Section 7.6's 7.7 dB of unexplained excess is the
budget available for that.

## 7.8 The curve is a guess, and the guess is now tested

Ethan: *"You only 'guess' the curve, you don't know it."*

That is a constraint on the implementation, and the code did not honour it:
the ellipse was fitted and judged on the same ensemble, which is the exact
trap the out-of-sample rule exists to prevent. `guess_credibility` now splits
the ensemble into two banks on alternating members, fits each, and asks each
bank's directions how much of the OTHER bank they explain.

| ensemble | held in | held out | credibility | carries |
|---|---|---|---|---|
| a real shared departure | 0.387 | 0.228 | **0.589** | 58.3 |
| a departure planted in ONE BANK only | 0.825 | 0.004 | **0.005** | 1.6 |

Two orders of magnitude between a direction that describes the path and one
that describes the fields it was measured from.

**The admission bar is derived, not chosen.** A direction that explains
nothing it has not seen still carries its equal share of the other bank's
energy, and an equal share of `L` directions is `1/L`. So `carries =
held_out x L` is the share against that null, and a direction earns admission
at `carries > 1`. The real departure returns 58 on a 256-point grid; the
single-bank departure returns 1.6, sitting essentially on the null.

## 7.9 "The limit is achieved when the residual stops increasing"

Ethan's terminating condition, and it is better than the sphere test in one
important way: **it needs no threshold at all.** Each pass attributes some of
the ensemble's energy to the directions it found, that attribution
accumulates, and when a pass adds nothing the accumulation has converged and
its value is the exact residual.

Implemented, and `differentiate_to_floor` now returns `accumulation`,
`attributed`, `attributed_fraction` and `exact_residual`. Building it exposed
a defect in the loop: the refusal test demanded a strict improvement in
asymmetry, so a deliberately conservative step - which makes real but small
progress - was thrown away, and the accumulation rule was never reached. It
now refuses only a pass that makes the shape WORSE by more than the scatter a
random ensemble of that size already shows.

**Tested against ground truth**, which nothing in this arc had done before:
plant a known structure, so the true residual is known exactly.

| configuration | attributed | truth | error on the residual |
|---|---|---|---|
| the accumulation rule ALONE, Wiener step | 99.9% | 72–73% | **−99.6%** |
| the shipping loop, Wiener step | 77–79% | 72–73% | −17% to −22% |
| the shipping loop, step 0.2 | 58–61% | 72–73% | +42% to +54% |

Reproduced on three independent seeds.

**Two findings, and the first corrects Ethan's rule rather than confirming
it.** The accumulation rule *alone* does not terminate correctly: with the
circular test suppressed it grinds on until 99.9 per cent of the energy has
been attributed, because a pass can always find something in noise. It is
necessary but not sufficient, and must run beside the circular test rather
than instead of it. With both active the loop stops in the right place.

**Second, the loop is optimistic by a measured amount.** It attributes 77–79
per cent where the truth is 72–73, so the exact residual comes out about 20
per cent low. The cause is ordinary shrinkage bias: a direction fitted on N
components absorbs its share of their noise along with the structure. The
Wiener step is much the closer of the two settings - a fixed conservative
step under-attributes by more than the Wiener step over-attributes - and this
is the first calibration of the method against a known answer, so the bias is
now a number rather than an unknown.

## 7.10 Is the key complete? Measured on the decode

`span_completeness` projects the per-field differential onto the span of the
modelled modifications. It uses the DIAGONALS only, one per field, so it is
not touched by the nested-matrix defect of section 6c.

Measured on 27 fields of the 75-bars capture, with the modelled signatures
evaluated on the decoder's own carrier axis:

| | |
|---|---|
| modelled basis | 12 directions over 119 places |
| inside the span | **45.4%** |
| what a basis that size captures from anything | 10.1% |
| excess above that null | **+35.3%** |
| **outside the key** | **54.6%** |

So the key covers a little under half of what the tape did, well clear of the
null, and **just over half lies outside it** — which is the honest upper
bound on what no amount of further fitting recovers with the present set.

**One correction to this number.** A first run built the modelled basis on an
assumed 1-7 MHz grid, where the decoder's own carrier axis is
2.543-5.257 MHz, 380 bins of 7.1 kHz - a factor of 2.2 wrong in span, so
every signature was evaluated at the wrong frequencies. That run read 44.3%.
The tool now takes the axis from `carrier_frequency_bins`, and the corrected
figure is the 45.4% above. The two agreeing so closely is luck, not
robustness.

## 8. Three defects found on the way, and the rules they leave

Latent, because each only bites once there is more than one component, more
than one grid, or a complex value — that is, exactly when the algorithm
starts working.

1. **The projection was grid-blind.** `_directions` truncated every vector to
   the shortest and took inner products across them, so a 238-bin frequency
   response and a 262-line time residual returned a "coherence" that was an
   artefact of memory layout. Components are now grouped by grid.
2. **Phase was silently discarded.** Both orthogonalisers cast with
   `dtype=np.float64`, which on a complex array keeps the real part and drops
   the imaginary one, defeating `COMPLEX_COMPONENTS` entirely. Not a
   precision loss: stripping phase moves the same ensemble's asymmetry from
   0.248 to 0.876 — a different answer to a different question.
3. **The field axis was never orthogonalised.** Both loops iterated `AXES`,
   which omits `FIELD_AXIS`, so a per-field dimension could be measured and
   accumulated and still contribute nothing.

The repair is **bitwise identical on the live ensemble** — two same-length
real components on the frequency axis — and a test asserts that against a
replica of the pre-repair arithmetic.

Two more, from building the loop:

4. **Deflating a direction drawn from the ensemble's own span leaves an exact
   zero eigenvalue.** Counting it as an unequal share reports the deflation's
   own bookkeeping as structure, and the loop oscillated instead of
   converging. The fix is not "drop the zeros" — a genuine needle also has
   `N−1` zeros and that *is* maximal asymmetry. **A shape measure must be told
   the dimensionality it is judging, or it measures its own deflation.**
5. **The nested matrix is rank-deficient by construction** — "a vs b" is the
   difference of "a" and "b", so seven entries span four dimensions. Charging
   that to the deflation made the loop refuse its own first pass. The
   intrinsic deficiency is now measured once, at pass zero, separately from
   what is removed.


## 9. What runs it

| file | what it holds |
|---|---|
| `vhsdecode/models/information_extrapolation.py` | `ellipsoid`, `differentiate_to_floor`, `component_differentials`, `graph_relations`, `sphere_floor`, `axes_present` |
| `vhsdecode/models/capture_profile.py` | the fourth link: word length from the code lattice, quantisation floor, `binding_limit` |
| `tools/ringing_measure/elliptical_collapse.py` | the runnable tool |
| `tests/unit/test_information_extrapolation.py` | the closed forms, the three preconditions, the loop, the matrix |

To run it:

```
python3 -m tools.ringing_measure.elliptical_collapse CAPTURE OUTPUT \
    [--json report.json] [-- decode flags ...]
```

Everything after `--` goes to the decoder unchanged; with nothing there the
arc's standard flags are used. The tool only reports — it writes nothing
beyond what the decoder itself writes, and the JSON if asked.
