# The collapse applied to television propagation

`docs/ELLIPTICAL_COLLAPSE.md` established the algorithm and, on VHS tape
magnetics, established its limit: the loop reaches the information floor in
two passes, but the six physical loss mechanisms proved only **1.58
effectively distinguishable of 6**, because every one of them is a function
of spacing over wavelength and so is nearly the same exponential decay in
frequency. The binding limit was identifiability, not the algorithm and not
the converter.

This document asks the same question of a different origin: the broadcast
path from transmitter to receiver, and the other ways a recording can have
acquired its defects before it ever reached the tape. The central question
is whether propagation is better or worse in this respect.

The answer is **better, by a factor of 2.4 on a like-for-like count, and
without limit on multipath**, which is a different kind of mechanism
entirely. The reason is not that the propagation chain is simpler. It is
that a delay is not a decay.


## 1. The chain, component by component

The signal path from the studio to a recorder's input, in order, with what
a decoder can measure of each. "Axis" is this arc's own three, plus the
per-field dimension: **A** amplitude, **F** frequency, **T** time.

System M figures throughout are ITU-R BT.470-6 Table 3 unless another
source is named. The item numbers refer to that table.

| # | component | the physics | what a decoder can measure | axis | the synthetic side |
|---|---|---|---|---|---|
| 1 | studio source and genlock | the programme as transmitted, on the studio's own time base | sync width and spacing, blanking and sync levels, burst amplitude and phase | A, T | SMPTE 170M / ITU-R BT.1700 levels and timings; subcarrier at 455/2 × line rate = 3.579545 MHz |
| 2 | transmitter power amplifier | AM-to-AM compression and AM-to-PM conversion near saturation; differential gain and differential phase | the level dependence of gain and of subcarrier phase, read at several picture levels | **A** | the standard's differential gain and phase limits; the ideal is zero |
| 3 | group-delay precorrection | deliberate predistortion so the average receiver's IF delay is cancelled | the residual delay after the receiver, as a phase against frequency | F, T | **170 ns at 3.58 MHz, nominal** — 47 CFR 73.687(a)(5) |
| 4 | vestigial-sideband filtering | the lower sideband is suppressed beyond the vestigial width; the slope must be Nyquist-symmetric so the two sidebands sum to unity | the response below the vestigial width, where a mistuned carrier tilts the sum | F | channel 6 MHz (item 1); main sideband 4.2 MHz (item 4); vestigial sideband 0.75 MHz (item 5); minimum attenuation 20 dB at −1.25 MHz and 42 dB at −3.58 MHz (item 6) |
| 5 | antenna and feeder | mismatch ripple, and feeder loss rising as the square root of frequency — a straight tilt in decibels across a 6 MHz channel | a smooth amplitude tilt across the band | A, F | flat; the tilt is the departure |
| 6 | **multipath** | reflected paths arriving early or late; each is a delayed scaled replica, so **an echo at delay τ is a ripple of period 1/τ in the frequency response** | the ripple's period, depth and phase — equivalently the delay, amplitude and polarity of each echo | **F, T** | no echo; and where a ghost-cancelling reference is present the standard supplies the transmitted waveform exactly (ATSC A/49, line 19; ITU-R BT.1124) |
| 7 | tropospheric propagation | slow multipath fading, ducting; the channel's own drift over seconds to minutes | the per-field variation of everything above | **field** | stationarity; the drift is the departure |
| 8 | co-channel and adjacent-channel interference | an unwanted vision carrier beating with the wanted one, at the offset the allotment fixes | discrete spectral **lines** at the offset, not a response shape | F | the offset frequency the allotment prescribes; protection ratios in ITU-R BT.655 |
| 9 | receiver front-end selectivity | image and adjacent-channel rejection; a roll-off at the top of the main sideband | amplitude roll-off and its minimum-phase delay near 4.2 MHz | A, F | flat to the main sideband edge |
| 10 | IF SAW filter | the passband's group delay, flat in the middle and rising at both edges | phase against frequency, cubic if the delay is parabolic | F, T | the delay the transmitter's precorrection was specified against (row 3) |
| 11 | intercarrier sound trap | a notch at the sound carrier whose skirt reaches the top of the video band | amplitude dip and rising delay approaching 4.2 MHz | A, F | sound carrier at **+4.5 MHz** (item 2); the video band ends at 4.2 MHz, so only the skirt is in view |
| 12 | video detector | **synchronous** detection is exactly linear; **envelope** detection folds the vestigial signal's quadrature component in as a second-order term | the **quadratic** dependence of the error on modulation depth — see §3.5 | **A** | zero; the quadratic term is the whole departure |
| 13 | AGC | gain following the sync tip; its attack and release, and its interaction with fading and with impulsive noise | overall gain against time, per line and per field | A, T, field | constant gain; sync tip at 100% of peak carrier (item 8) |
| 14 | receiver noise | thermal noise at the front end | the floor beside the signal, in the same chain at the same moment | A | kT plus the receiver noise figure — see §8 |

Two structural facts follow from this table and matter more than any row
in it.

**Negative modulation puts the sync tip at peak carrier.** System M is C3F
negative (item 7) with the synchronizing level at 100% of peak carrier and
blanking at 72.5 to 77.5% (item 8). So the sync pulse is the strongest
thing in the signal, the AGC references it, and it is the best-conditioned
probe in the whole chain — exactly as it is for the tape, and for the same
reason. This arc's sync-only constraint carries over unchanged.

**The transmission demodulator is LINEAR where the tape's is not.** This is
the single most consequential difference and it inverts one of this
project's standing laws. `docs/ELLIPTICAL_COLLAPSE.md` and the ringing arc
established that no video-domain filter can invert a tape-path defect,
because the defect is LTI on the RF ahead of a **nonlinear FM
demodulator**. A vestigial-sideband AM signal detected synchronously is
recovered by a **linear** operation: the in-phase output's spectrum is
`A(f)·[R(f) + R(−f)]/2`, and the Nyquist property the vestigial slope
exists to provide makes `R(f) + R(−f) = 1`, so the output is the modulating
signal exactly, halved. An LTI defect anywhere in the transmission chain is
therefore still LTI in the demodulated video, and **a video-domain filter
is the correct instrument for it.** The one exception is row 12, envelope
detection, whose quadrature term is genuinely nonlinear — and §3.5 measures
how much.


## 2. The method, validated before it is used

The test is the one `docs/ELLIPTICAL_COLLAPSE.md` §7.2 ran on the tape.
Build each mechanism's predicted signature over the measurement axis by
perturbing its own physical parameter; unit-normalise the rows; take the
pairwise coherence matrix and the singular values; report the participation
ratio of the squared singular values as the effective number of
distinguishable mechanisms — the same measure `ellipsoid` computes as
`participation`.

Run first on the tape, to confirm the method reproduces the published
result. Perturbing each mechanism in `head_model.log_response` about the
model's own `TYPICAL` point, on the format's own mechanics for
(VHS, NTSC, SP), over 0.5 to 8.0 MHz:

```
singular values: 2.160  1.000  0.5643  0.1314  0.008726  5.365e-05
share of span:   77.7%  16.7%    5.3%    0.3%      0.0%      0.0%
EFFECTIVE DISTINGUISHABLE: 1.58 of 6
```

Against the document's published `2.159, 1.000, 0.565, 0.133, 0.0078,
0.0002` and its 1.58. The method is the same method.

### 2.1 One correction the propagation case forced

The tape's six signatures are real log magnitudes, so the inner product
never had to be chosen. Propagation's are complex, and the choice moves the
answer by more than a third.

Each mechanism's departure is a **real** physical quantity — a delay in
seconds, an echo amplitude, a tilt in decibels per megahertz — multiplying
a fixed complex signature. The coefficient the evidence must resolve is
real, so the signatures live in a real vector space of twice the length,
with real and imaginary parts stacked. Under the Hermitian product instead,
a purely real signature and a purely imaginary one of the same shape are
the **same** direction, because they differ by the unit scalar `j`. That is
the right accounting for a correction carried as a complex gain per
frequency, and the wrong one for asking whether two mechanisms can be told
apart: an amplitude tilt and a phase cubic are distinguishable precisely
because one is real and the other imaginary.

Measured, on the six named propagation mechanisms:

| convention | effective of 6 |
|---|---|
| real parameters, complex signatures (correct here) | **3.79** |
| Hermitian, complex signatures | 2.20 |
| log magnitude only | 3.29 |
| phase only | 1.78 |

The tape baseline is **1.58 under both conventions**, identically, because
its signatures are real. So the comparison below is fair: the convention
that helps propagation does nothing at all for the tape.

This is the same family of error `docs/ELLIPTICAL_COLLAPSE.md` §8 lists —
a bookkeeping choice that only bites once the components are complex, and
that changes the answer to a different question rather than losing
precision.


## 3. Are the propagation mechanisms distinguishable?

The measurement axis is the demodulated video band, 0 to 4.2 MHz — System
M's nominal width of main sideband (item 4), which is what a decoder can
actually see. Each mechanism is expressed as a departure of the received
radio-frequency response about the vision carrier, then carried down to the
video band the way a synchronous detector carries it, so the vestigial
weighting is applied once, in the place the standard puts it.

### 3.1 Six named mechanisms, against the tape's six

| | multipath | vsb slope | saw delay | sound trap | feeder | frontend |
|---|---|---|---|---|---|---|
| multipath | 1.000 | 0.017 | 0.021 | 0.057 | 0.060 | 0.052 |
| vsb slope | 0.017 | 1.000 | 0.000 | 0.098 | 0.076 | 0.048 |
| saw delay | 0.021 | 0.000 | 1.000 | **0.764** | 0.000 | **0.678** |
| sound trap | 0.057 | 0.098 | **0.764** | 1.000 | 0.466 | 0.367 |
| feeder | 0.060 | 0.076 | 0.000 | 0.466 | 1.000 | **0.574** |
| frontend | 0.052 | 0.048 | **0.678** | 0.367 | **0.574** | 1.000 |

```
singular values: 1.491  1.272  0.9932  0.9881  0.379  0.2319
share of span:   37.0%  27.0%   16.4%   16.3%   2.4%   0.9%
cumulative:      37.0%  64.0%   80.4%   96.7%  99.1% 100.0%
condition number: 6.43
```

> **The effective number of distinguishable propagation mechanisms is
> 3.79 of 6, against 1.58 of 6 for tape magnetics.**

Two comparisons make the difference concrete.

**The condition number falls from 1.06 × 10⁴ to 6.4** — three and a half
orders of magnitude. The tape's mechanism matrix is numerically singular;
propagation's is not.

**No pair is collinear.** The tape's worst pair, gap length against azimuth
error, sits at **1.000** — they are *exactly* the same sinc loss in
wavelength and no measurement of a frequency response can ever separate
them. Propagation's worst pair, the SAW group delay against the sound
trap, sits at 0.764, which is high but invertible; and it is high for an
honest reason, since both are the receiver's IF strip and the trap's skirt
*is* part of the passband's delay.

The confounded set is exactly the receiver's own filtering — SAW, sound
trap, front end, and the feeder tilt they sit behind. The vestigial slope
stands apart because it is confined below 0.75 MHz by construction, and
multipath stands apart for the reason §3.2 is about.

### 3.2 Multipath is a different kind of mechanism

The expectation to be tested was that multipath should be far more
separable than tape magnetics, because an echo at delay τ is a ripple of
period 1/τ in the frequency response — different delays giving genuinely
different shapes — whereas the tape's losses are all monotone decays.

**Confirmed, and more strongly than the framing suggests.** Multipath is
not one mechanism that happens to be well conditioned. It is a mechanism
whose signature space has a dimension of its own, equal to the
time-bandwidth product of the observation.

Eight echoes, spaced by a multiple of the delay resolution `1/B`:

| spacing | effective of 8 | worst coherence |
|---|---|---|
| 0.25 / B | 4.53 | 0.658 |
| 0.50 / B | **7.92** | 0.101 |
| 1.00 / B | **7.92** | 0.099 |
| 2.00 / B | **7.92** | 0.099 |
| 4.00 / B | **7.92** | 0.099 |

Eight echoes separated by half the delay resolution or more are **7.92
effectively distinguishable of 8 — 99%**, and the number does not improve
with wider spacing because it is already at the ceiling. Below half the
resolution it degrades, exactly as it must. The result is scale-free: in
units of `1/B` it is identical for every bandwidth, which is why the table
has no bandwidth column.

The delays this matters over are fixed by the standard. ATSC A/49, the
FCC's line-19 ghost-cancelling reference for NTSC, specifies detection of
ghosts from **−3 µs to +45 µs** using a chirp spanning **0 to 4.2 MHz**. So:

| | value |
|---|---|
| observation bandwidth B | 4.2 MHz (item 4) |
| delay resolution 1/B | **238 ns** |
| ghost window | 48 µs (A/49) |
| **time-bandwidth product** | **202** |

**The multipath subspace has about 202 near-orthogonal dimensions**, and
the evidence supports essentially all of them. Where the tape's six
mechanisms crowd into 1.58 directions, propagation's dominant mechanism
spreads over two hundred.

The reason is structural and worth stating plainly. Every tape loss —
spacing, gap, thickness, azimuth — is `exp(−k·f)` or `sinc(k·f)` for a
different `k`, and all of them are monotone decays that differ only in
rate; a family of monotone decays over a finite band is nearly
one-dimensional. An echo family is `exp(−2πjfτ)`, a **Fourier basis** over
the delay parameter, and a Fourier basis is orthogonal by construction. The
tape's mechanisms are parameterised along the axis that makes them alike;
multipath's along the axis that makes them independent.

### 3.3 The whole set

Five channel filters plus five echoes spread across the ghost window:

```
singular values: 1.510 1.285 1.024 1.004 0.998 0.993 0.974 0.955 0.361 0.195
cumulative:      22.8% 39.3% 49.8% 59.9% 69.8% 79.7% 89.2% 98.3% 99.6% 100%
EFFECTIVE DISTINGUISHABLE: 7.27 of 10  (73%)
```

Adding echoes *raises* the share, from 63% at six mechanisms to 73% at ten,
because each new echo adds a direction of its own while the filters
continue to share theirs. That is the opposite of the tape, where more
mechanisms bought less.

Side by side:

| ensemble | count | effective | share |
|---|---|---|---|
| tape magnetics (`head_model`, 0.5–8 MHz) | 6 | **1.58** | 26% |
| propagation, five channel filters only | 5 | 2.95 | 59% |
| propagation, six named mechanisms | 6 | **3.79** | 63% |
| propagation, five filters + five echoes | 10 | 7.27 | 73% |
| propagation, eight echoes at 1/B spacing | 8 | **7.92** | 99% |

Note the third row against the first. Propagation's *filter-only* subset —
its closest analogue to the tape's problem, a set of smooth responses over
one band — still reaches 59% where the tape reaches 26%. Even without
multipath, propagation is better conditioned, because its filters differ in
**kind** (one is pure delay, one is a notch skirt, one is a tilt, one is
confined below 0.75 MHz) where the tape's differ only in rate.

### 3.4 What the repository's own ellipsoid says

Running `ie.ellipsoid` and `ie.differentiate_to_floor` over the same
ensembles, on the frequency axis:

| ensemble | components | rank | budget | resolved | asymmetry vs floor | loop |
|---|---|---|---|---|---|---|
| six named | 6 | 6 | 6.000 | 3.574 (59.6%) | 0.7597 vs 0.0116, **+271σ** | 5 passes, 6 directions, circular |
| filters + echoes | 10 | 10 | 10.000 | 2.903 (29.0%) | 0.5819 vs 0.0192, **+204σ** | 6 passes, 9 directions, circular |

The matrix is **full rank** in both cases — every mechanism contributes a
direction the evidence supports, which is already a sharper result than the
tape's, where the six signatures span 1.58 — and the loop reaches the circle
in both. The trace is conserved to the component count exactly, as the
invariant requires.

Three readings need qualifying rather than quoting.

**The pass count is 5 and 6, not the flat 2 the tape gave.** On the tape
every direction above the noise edge came out together in one pass. Here it
does not, and the reason is the next reading.

**Only one direction stands above the Marchenko–Pastur edge.** With six to
ten components over 512 bins the edge sits at 1.23 to 1.30, and the second
eigenvalue is near 1.0 — below it. The edge is deliberately conservative,
never admitting noise as a direction, and with this few components against
this long a grid it admits only the leading one, so the loop has to work
down to the circle a direction at a time. This is a property of the
*ensemble's size*, not of propagation: `docs/ELLIPTICAL_COLLAPSE.md` §4's
`N − 2` rule says six components resolve very little, and the real data
there used 91 to 378 matrix entries. A real run would form the nested
differential matrix over many fields and would not be in this regime.

**`resolved` and `largest coherence` are computed on the Hermitian Gram**,
which §2.1 showed is the wrong convention for this question — the reported
0.986 largest coherence is the Hermitian one, against 0.764 under the real
convention. Read both as lower bounds on how well the ensemble separates.

### 3.5 The one nonlinear step, and why it lands on a different axis

Envelope detection is the exception to §1's linearity result, and the
question is how big the exception is. Measured on a System M vestigial
channel with a periodic probe carrying no sync edge, so that nothing is
truncated by the 4.2 MHz band limit and the only nonlinearity present is
the detector's own:

**The level law** — a 1 MHz probe on a 40% pedestal:

| depth, IRE p-p | quadrature rms, IRE | envelope error rms, IRE | error / depth² | 2nd harmonic |
|---|---|---|---|---|
| 5 | 1.77 | 0.0173 | 0.000691 | −40.2 dB |
| 10 | 3.54 | 0.0691 | 0.000691 | −34.2 dB |
| 20 | 7.07 | 0.2762 | 0.000691 | −28.2 dB |
| 40 | 14.14 | 1.1048 | 0.000690 | −22.4 dB |
| 80 | 28.28 | 4.4141 | 0.000690 | −17.1 dB |

The fourth column is flat to three figures: **the error is exactly
quadratic in the modulation depth.** That is a detector nonlinearity, and
no linear frequency response can produce it. It therefore belongs on the
**amplitude axis** and does not enter the frequency-axis ensemble of §3.1
at all — which is itself the finding, since it means quadrature distortion
cannot be identified from a frequency response and must be measured by
sweeping level.

**The pedestal law** — a 40 IRE probe at each picture level:

| pedestal, % carrier | IRE | envelope error rms, IRE |
|---|---|---|
| 12.5 (peak white) | 100 | 3.474 |
| 25.0 | 80 | 1.767 |
| 40.0 | 56 | 1.105 |
| 60.0 | 24 | 0.737 |
| 75.0 (blanking) | 0 | 0.589 |

Worst toward peak white by a factor of 5.9, because the term is `Q²/2I` and
the denominator is the carrier itself, which negative modulation makes
smallest exactly there.

**Where in frequency** — a 40 IRE probe swept:

| probe | envelope error rms, IRE |
|---|---|
| 0.051 MHz | 0.013 |
| 0.148 MHz | 0.106 |
| 0.301 MHz | 0.390 |
| 0.500 MHz | 0.834 |
| 0.750 MHz | 1.105 |
| 1.5 MHz and above | 1.105 (a plateau) |

Near zero at DC, where both sidebands are present in equal measure and the
modulation is effectively double-sideband; rising through the vestigial
transition; and a **plateau** above the vestigial width of 0.75 MHz (item
5), where the modulation is fully single-sideband and the quadrature term
saturates. That shape — zero at DC, saturating above 0.75 MHz — is
distinctive, and §5 uses it.


## 4. The component set and the graph for an off-air source

### 4.1 A fourth stage

`information_extrapolation` declares three stages: `RF_PLAYBACK`,
`RF_RECORDING`, `PICTURE`, ordered by the chain, because a machine can only
be inverted once everything that acted after it has been. An off-air source
needs one more, upstream of `RF_RECORDING`:

```
TRANSMISSION -> RF_RECORDING -> (tape) -> RF_PLAYBACK -> PICTURE
```

inverted in the reverse order, so `TRANSMISSION` is inverted **last**, on a
decode from which both machines have already been removed. That is the same
discipline as the existing `RF_RECORDING`, and for the same reason: what it
then sees is the transmission path's own contribution and not the chain
convolved.

Its correction is legitimate **in the video domain**, by §1's linearity
result. This is not an exception to the standing law that forbids
video-domain inverses; it is that law applied correctly. The law's premise
is a nonlinear demodulator between the defect and the video, and the
transmission chain's demodulator is linear.

### 4.2 The components and their parents

| component | parents | axes | criterion |
|---|---|---|---|
| `transmitter response` | — | A, F | the VSB mask (items 4, 5, 6) and the 170 ns precorrection at 3.58 MHz |
| `transmitter nonlinearity` | transmitter response | **A** | differential gain and differential phase against picture level |
| `multipath` | transmitter response | F, T | echo delay, amplitude and polarity; the GCR where present |
| `tropospheric drift` | multipath | field | the per-field variation of the multipath solution |
| `antenna and feeder` | multipath | A, F | a tilt in decibels across the channel |
| `receiver front end` | antenna and feeder | A, F | roll-off at the main sideband edge |
| `IF group delay` | receiver front end | F, T | the delay the precorrection was specified against |
| `sound trap` | receiver front end | A, F | a notch at +4.5 MHz (item 2), skirt only |
| `vestigial slope` | receiver front end | F | the carrier against the −6 dB point; confined below 0.75 MHz |
| `detector quadrature` | vestigial slope, sound trap, IF group delay | **A** | the quadratic level law of §3.5 |
| `AGC` | detector quadrature | A, T, field | gain against time, referenced to the sync tip |

`graph_relations` relates a node to its parents, its children and its
siblings, and nothing else. The structure above has a property worth
noticing: **the siblings are exactly the confounded set.** SAW delay, sound
trap, vestigial slope and front-end roll-off all share `receiver front end`
as a parent, and §3.1's coherence matrix shows those are precisely the
pairs standing at 0.37 to 0.76. The graph, declared from the physics, and
the coherence, computed from the signatures, agree about where the
nesting is informative — which is what makes `component_differentials`
worth forming here rather than differencing every pair.

### 4.3 What the two chains share

| quantity | the propagation chain | the tape chain | separable? |
|---|---|---|---|
| sync pulses | the transmitted pulse, to BT.1700 timings | the same pulse, re-recorded | **no** — one pulse carries both |
| a frequency response | rows 3–11 of §1 | head, transport, electronics | **no** — the total is the product |
| a time base | genlocked, stable | wow and flutter, per transport rate | **yes** — §5.5 |
| an echo | multipath | RF reflections, decoder ringing | **not directly** — §5 |
| a noise floor | thermal, at the front end | the medium's carrier-to-noise | **yes** — §8, different sizes |
| a level dependence | envelope detector quadrature only | the FM path throughout | **yes** — §5.1 |

The frequency-response row is the hard one and it does not soften: from a
single recording, only the **product** of the transmission response and the
tape response is observable, and no amount of modelling recovers the
factors from the product alone. Everything in §5 is about breaking that.


## 5. What breaks the tape-versus-transmission tie

A ghost recorded onto tape is indistinguishable from a ghost created by the
tape unless something breaks the tie. Eight things do, in decreasing order
of how decisive they are.

### 5.1 Level dependence — the decisive one

A propagation echo is linear in the video domain: its mix does not depend
on the picture level. A tape-path artefact is LTI on the RF ahead of a
**nonlinear** FM demodulator, so its video-domain appearance *is* level
dependent — the instantaneous carrier sweeps with picture level and the
path's response at that carrier changes with it. This is not a conjecture
about the tape: `vhsdecode/addons/RINGING_ANALOG_CHAIN.md:65-97` documents
FM lower-sideband truncation as the measured level-dependent ring
mechanism, with the truncation edge at 1.3–1.7 MHz and the sideband
truncation excited at the tip and blanking carriers.

So: **an echo whose mix is the same on a sync transition and on a white
transition arrived before the modulator; one that changes is the RF path.**
The repository already measures this — the ringing module carries level
strata and per-landing scales, and `--luma_eq` exists because this arc
ruled that the level-dependent scale belongs there.

The one propagation mechanism that is *also* level dependent is the
envelope detector's quadrature term, and §3.5 gives its distinguishing
signature: exactly quadratic in depth, near zero at DC, saturating to a
plateau above 0.75 MHz. The tape's level dependence is concentrated near
the de-emphasis corner and the truncation edge instead. The two level laws
are separable on their frequency shape.

### 5.2 Head parity

A playback-path or tape defect is a property of the head that laid the
field down or reads it, and the heads alternate field to field. A
transmission defect knows nothing about a drum and is identical on both.
The repository already insists on this split for a different reason —
`head_dc_components` records that pooling across heads reports a
head-to-head difference as though it were a relationship — and supplies the
machinery: `per_head_average`, `per_head_differential`,
`head_response_difference`.

**A response that is the same on both heads, to within their measured
difference, is not the tape.**

### 5.3 Absolute frequency landmarks

The two chains have frequency plans that do not overlap, and a defect
pinned to one chain's landmark cannot belong to the other.

| landmark | frequency | belongs to |
|---|---|---|
| sound trap notch | +4.5 MHz (item 2) | transmission only |
| vestigial slope | below 0.75 MHz (item 5) | transmission only |
| colour subcarrier | 3.579545 MHz | shared, but the recorder heterodynes it |
| VHS colour-under carrier | 0.6294 MHz | tape only |
| VHS luma FM carrier | 3.400 to 4.400 MHz | tape only |
| drum rate | 29.97 Hz | tape only |

VHS SP recovers about 3.0 MHz of luma baseband, so **4.5 MHz is not
reachable through the tape at all** — anything at that frequency in an
off-air recording is transmission-side by construction, or it is not there.
Conversely, structure at 629 kHz or at the drum rate cannot have been
broadcast.

### 5.4 Delay sign and delay size

ATSC A/49's window runs from **−3 µs to +45 µs**. Two consequences:

**Long delays are transmission.** 45 µs is 70% of a System M line. The
tape's RF path and the decoder's own filters produce aftermath measured in
single microseconds; the ringing arc's kernels are truncated far below
this. A confirmed echo at tens of microseconds did not come from the deck.

**Pre-echoes are transmission — and this contradicts a standing rule.**
`vhsdecode/addons/RINGING_RULES.md:10-14` states, as a hard constraint,
"Strictly causal. All ringing originates in one direction, from physics. No
pre-echo, no two-sided kernels (a 'reflection before the edge' is the
tell)." That rule is **correct for the tape and playback path** and must
stay. It is **wrong for a transmission-stage component**, where a leading
ghost is a real, standardised phenomenon that the ghost-cancelling
reference is explicitly designed to detect. If a `TRANSMISSION` stage is
ever added, Rule 2 needs a stage qualifier, and the fact that a candidate
echo is non-causal becomes *evidence that it is transmission* rather than
grounds to reject it.

### 5.5 Time-base coupling

The time-base correction re-times the picture to the recorded sync. An echo
that was in the signal before recording sits at a fixed delay relative to
the recorded content *including the sync edges*, so after time-base
correction its delay in output samples is constant. An echo created in the
playback path sits at a fixed delay in **playback** sample time, so after
correction its delay moves with the wow factor.

**Correlate the fitted echo lag against the wow factor.** The residual
channel is already exported: `RESIDUAL_CHANNELS["time"]` is
`"wowfactors - 1"`. A lag that tracks it is playback-side; one that does
not arrived earlier.

### 5.6 Mechanical rates

`transport_model` predicts where every rotating part's disturbance lands —
drum 29.97 Hz, capstan 4.83 Hz, pinch roller 1.06 Hz, guides 1.77 Hz, reels
in a 0.118–0.442 Hz band. A tape defect is modulated at one of those rates.
Tropospheric fading is not: its time constants are seconds to minutes and
bear no relation to a transport's geometry, and multipath from a fixed
installation is static.

`docs/ELLIPTICAL_COLLAPSE.md` §7.3 established that five seconds of decode
resolves 26 of the 28 rate pairs. The same five seconds separates "modulated
at a transport rate" from "not", which is the discriminator here.

### 5.7 Polarity

The ringing arc's polarity-owned poles rest on the fall ringing against the
sync-tip carrier and the rise against the blanking carrier — two different
carriers, hence two different responses, hence an asymmetry that only an RF
path can produce. A linear video-domain echo scales with its driving edge
and is polarity-symmetric.

**An asymmetry between the two edges beyond what their level difference
explains is the RF path.**

### 5.8 The ghost-cancelling reference, where it survives

This is the cleanest discriminator of all and it is worth checking for on
any off-air recording made after the GCR was deployed.

ATSC A/49 (and ITU-R BT.1124 systems A, B, C) place a **known transmitted
waveform** on line 19 of the vertical interval — a chirp over 0 to 4.2 MHz
at a 30 IRE level, phase-inverted once per frame so that frame differencing
removes the picture. It is the only place in the signal where the synthetic
side is exactly specified rather than inferred.

The received line 19 carries the transmission response **and** the tape
response, so on its own it does not separate them. But the sync pulse
carries the same tape response, and the two are related in the graph. Their
**nested differential** — `component_differentials`' off-diagonal entry,
`d_GCR − d_sync` — cancels the shared tape path and leaves the transmission
response alone. That is exactly the construction the collapse is built on,
applied to the one pair whose shared parent is the thing to be removed.

Its processing gain is substantial: the chirp's time-bandwidth product over
a 52 µs active line is 218, or **23.4 dB on a single line**, rising to 38.2
dB over 30 frames and 52.9 dB over 900. That is what makes an echo tens of
decibels below the direct signal measurable at all.

The practical caveats are real: the GCR was carried by a minority of
stations and only from the mid-1990s, and the vertical interval is the
first thing a bad tape or a sloppy dub loses. Where it is absent, §5.1
through §5.7 are what there is.


## 6. What already exists in this repository

Nothing here needs rebuilding. The inventory, with the caveat that the
repository's model of the chain is tape-and-deck-centric and the broadcast
front end is absent entirely.

**Already built and directly reusable**

| file:line | what it does |
|---|---|
| `vhsdecode/addons/ringing_cancellation.py:1533` | `_ghost_deviation` — the delayed-edge signature; its docstring already names "multipath / impedance mismatch" |
| `vhsdecode/addons/ringing_cancellation.py:3003-3009` | one design column per consensus-locked echo lag plus one shared ghost column; the least-squares coefficient **is** the ghost mix |
| `vhsdecode/addons/ringing_cancellation.py:3670-3694` | ghost and echo tracking, an EMA of the fitted coefficient clipped to ±0.5 |
| `vhsdecode/addons/ringing_cancellation.py:3787-4186` | echo lag discovery, modal clustering, cross-edge coverage and confirmation — the machinery §3.2's 202 dimensions would be searched with |
| `vhsdecode/addons/ringing_cancellation.py:504` | `fractional_shift` — causally exact sub-sample delay, the delay operator all echo work uses |
| `vhsdecode/addons/ringing_cancellation.py:5227-5240` | the source-class kernels, already attributed to "the off-air broadcast chain" — the only existing term of transmission provenance |
| `vhsdecode/addons/RINGING_INSTRUCTIONS.md:8-15` | the specification already lists ghosting as pre-recorder, cause "multi-path television reception" |
| `vhsdecode/models/information_extrapolation.py:1881/1923/1975` | `find_interference`, `remove_interference` (coherent subtraction, not a notch), `notch_interference` — §1 row 8, co-channel, already covered |
| `vhsdecode/models/information_extrapolation.py:2524/2757/2689/2656/2488` | `ellipsoid`, `differentiate_to_floor`, `component_differentials`, `graph_relations`, `sphere_floor` |
| `vhsdecode/models/information_extrapolation.py:2116/2142/2163` | `per_head_average`, `per_head_differential`, `head_response_difference` — §5.2's discriminator |
| `vhsdecode/models/information_extrapolation.py:1006` | `vertical_interval_components` — the nearest thing to a VITS instrument, and where a GCR component would sit |
| `vhsdecode/models/capture_profile.py:136/141/146` | `carson_bandwidth`, `channel_capacity`, `binding_limit` — §8 uses these unchanged |
| `vhsdecode/baseband_eq.py:639/656` | `equalizer` and `correct` — a measured **video-domain** inverse, which §1 shows is the right form for a transmission-stage correction |
| `vhsdecode/models/transport_model.py:80/113/137` | `rotation_rates`, `observable`, `find_lines` — §5.6's discriminator |
| `vhsdecode/models/filter_model.py:472` | `identifiability` — the same question asked of the de-emphasis parameters |
| `lddecode/core.py:513-672` | the IEC 60856/60857 group-delay pre-distortion curves — the existing precedent for a **standard-prescribed group delay as the synthetic side**, which is exactly what §1 row 3 needs |
| `tools/ringing_measure/elliptical_collapse.py:369/388` | `report_ellipse`, `report_loop` — the runnable driver |

**Absent entirely** — searched across the whole repository, zero hits
outside `.git` and `build`: `vestigial`, `VSB`, `SAW filter`, `IF
amplifier`, `intercarrier`, `sound trap`, `quadrature distortion`,
`synchronous detector`, `co-channel`, `GCR`, `ghost cancelling reference`,
`VIRS`, `2T pulse`. VITS exists only as LaserDisc SNR instrumentation
(`lddecode/core.py:103-109`, `:3285`, `:3891-3897`); multiburst only as a
test-pattern filename (`tools/ringing_measure/record_playback_transfer.py:58`).
Group delay is well developed throughout, but always as a tape, head or
filter property — never as a transmission-path property.

**One conflict to resolve before any of this ships**: `RINGING_RULES.md`
Rule 2's prohibition on pre-echo, per §5.4.


## 7. Other source types

**Betamax.** Structurally identical to VHS: colour-under at 0.6884 MHz,
luma FM from a 3.600 MHz sync tip to 4.800 MHz at 100 IRE, about 3.0 MHz of
luma baseband, so a delay resolution of 333 ns. Every component of §1
appears or vanishes exactly as it does for VHS, since what determines that
is the *source* rather than the recorder. Identifiability of the
transmission chain is unchanged in kind and marginally worse in degree,
because a narrower baseband means a coarser delay resolution. SuperBeta
lifts the tip to 4.750 MHz and the baseband to about 3.5 MHz — 286 ns — and
correspondingly improves it.

**Video8 and Hi8.** Video8 sits at a 4.200 MHz tip with 0.857 MHz per
100 IRE of deviation, close to VHS in baseband; Hi8 at 5.700 MHz with 1.429
MHz of deviation and about 5.0 MHz of baseband, giving a 200 ns delay
resolution — better than the broadcast channel's own 238 ns, so an Hi8
recording of an off-air source can resolve delays more finely than the
transmission that produced them. Both formats add their own record-side
nonlinear emphasis (`vhsdecode/nonlinear_filter.py:214-311` carries the
tables), which *adds* a level-dependent tape-side mechanism and therefore
makes §5.1's discriminator harder to read, not easier — the level
dependence must first be attributed to the format's own emphasis before any
remainder can be called a defect.

**U-matic.** Colour-under at 0.6884 MHz, tip at 3.800 MHz, 1.143 MHz per
100 IRE. Low band gives about 3.0 MHz of baseband and SP about 3.5 MHz.
U-matic's distinguishing relevance is provenance rather than physics: it was
the broadcast news and industrial format, so its tapes are far more likely
to be off-air or line-feed recordings than a domestic VHS, and far more
likely to carry a usable vertical interval with test signals in it. That
makes §5.8 much more often available.

**LaserDisc.** The most favourable of the recorded formats and a special
case. It is not a colour-under format: the composite video is carried
whole on one FM carrier, which `lddecode/core.py:75-77` places at 8.100 MHz
for 0 IRE with 1.7 MHz per 140 IRE, so the carrier runs **7.61 MHz at sync
tip to 9.31 MHz at 100 IRE** — nearly twice VHS's deviation. There is no
drum-rate head switching to alias with, and about 5.0 MHz of baseband gives
a 200 ns delay resolution. Crucially, `lddecode` already implements the **IEC 60856 and
60857 group-delay pre-distortion targets** (`lddecode/core.py:513-672`) and
already reads **VITS** (`:103-109`, `:3891-3897`), so two of §1's rows have
a synthetic side and an instrument in the tree already. The mastering path
was studio-quality, so most transmission mechanisms **vanish**: a LaserDisc
carries a transmission chain's defects only if the programme was mastered
from an off-air or network-feed source, and when it does, the disc's own
contribution is small enough that the transmission chain is what dominates
the residual. Identifiability improves on every count.

**A direct baseband studio feed (CVBS).** `cvbsdecode` and
`vhsdecode/format_defs/cvbs.py` already handle this. Here almost the entire
tape chain **vanishes** — no head, no transport, no colour-under
heterodyne, no FM demodulator and therefore **no nonlinearity between the
defect and the video at all**. What remains is §1 rows 1 to 14 and nothing
else, so the tie of §5 never has to be broken: whatever is measured is the
transmission chain's, by construction. This is the configuration in which
the propagation model should be developed and validated, and only then
carried onto tape. It is also the only configuration in which the full
4.2 MHz main sideband survives to the measurement, giving the 238 ns
resolution and the full 202-dimensional multipath subspace.

The delay resolution each source affords, which is the quantity that sets
how much of the multipath subspace is reachable:

| source | luma baseband | delay resolution 1/B | dimensions over 48 µs |
|---|---|---|---|
| CVBS direct / System M off air | 4.20 MHz | 238 ns | 202 |
| Type C 1-inch | 5.50 MHz | 182 ns | 264 |
| LaserDisc NTSC, S-VHS, Hi8 | 5.00 MHz | 200 ns | 240 |
| U-matic SP, SuperBeta | 3.50 MHz | 286 ns | 168 |
| VHS SP, Betamax, Video8, U-matic low band | 3.00 MHz | 333 ns | 144 |

Baseband figures are the formats' stated luma bandwidths; carrier and
deviation figures are read from the repository's own tables in
`vhsdecode/format_defs/`.


## 8. The capture requirement

Written in the style of `capture_profile.binding_limit`, and computed with
its functions.

### 8.1 The channel's own capacity

The FCC's system-independent planning factors (Appendix A of FCC 97-115)
state a thermal noise power of **−106.2 dBm**. Against kT at −174 dBm/Hz
that implies a bandwidth of 6.03 MHz, which is System M's nominal channel
of 6 MHz (item 1) — so the planning factors and the channel definition
agree, and the carrier-to-noise below is the channel's own rather than an
assumption. Received power is the field strength plus the dipole factor
plus antenna gain minus feeder loss, all from the same table; the contour
field strengths are those of 47 CFR 73.683, of which the 64 dBu UHF Grade B
figure is stated in the rule directly.

| band | contour | field, dBµV/m | received, dBm | noise, dBm | C/N, dB | capacity |
|---|---|---|---|---|---|---|
| low VHF (2–6) | Grade A | 68.0 | −40.8 | −96.2 | 55.4 | 110.4 Mbit/s |
| low VHF (2–6) | Grade B | 47.0 | −61.8 | −96.2 | 34.4 | 68.6 Mbit/s |
| high VHF (7–13) | Grade A | 71.0 | −45.8 | −96.2 | 50.4 | 100.5 Mbit/s |
| high VHF (7–13) | Grade B | 56.0 | −60.8 | −96.2 | 35.4 | 70.6 Mbit/s |
| UHF (14–69) | Grade A | 74.0 | −50.8 | −99.2 | 48.4 | 96.5 Mbit/s |
| UHF (14–69) | Grade B | 64.0 | −60.8 | −99.2 | 38.4 | 76.5 Mbit/s |

### 8.2 Which link binds

| link | band | C/N | capacity |
|---|---|---|---|
| the broadcast channel, Grade B to Grade A | 6.0 MHz | 34.4–55.4 dB | **68.6 to 110.4 Mbit/s** |
| the tape (VHS SP, Carson band) | 8.0 MHz | 21.6 dB | **57.4 Mbit/s** |
| the capture card (8 bit, 50 MSps) | 8.0 MHz | 46.7 dB | 124.1 Mbit/s |

**For an off-air recording on VHS, the tape still binds.** Even the worst
broadcast contour delivers 68.6 Mbit/s against the tape's 57.4 — a factor
of 1.19 — so the recorder throws away about a sixth of what arrived, and
everything the transmission chain did is present in the capture but
attenuated by the medium that stored it. The chain of binding limits is
therefore

```
capture card  >  broadcast channel  >  tape
124.1 Mbit/s     68.6-110.4           57.4
```

and the tape is the wall, exactly as `docs/ELLIPTICAL_COLLAPSE.md` §7.1
found for a camera-original recording. Better capture hardware buys nothing
for an off-air VHS tape either.

**For a direct capture of the channel — off air, at IF, or a baseband feed
— the propagation link binds**, and the converter must clear it. That is
the case §8.3 sizes.

### 8.3 What the converter needs

A vestigial-sideband signal peaks at the sync tip, since the modulation is
negative and the synchronizing level is 100% of peak carrier (item 8), and
the converter must not clip there. The usable rms therefore sits below full
scale by the crest factor: peak envelope 1.0, mean envelope at blanking
0.75, RF rms that over root two — **5.5 dB**.

In-band signal-to-noise after the crest factor, for a 6 MHz channel:

| sample rate | oversampling gain | 8 bit | 10 bit | 12 bit | 14 bit |
|---|---|---|---|---|---|
| 12 MSps | 0.00 dB | 44.4 | 56.5 | 68.5 | 80.5 |
| 20 MSps | 2.22 dB | 46.6 | 58.7 | 70.7 | 82.7 |
| 40 MSps | 5.23 dB | 49.6 | 61.7 | 73.7 | 85.8 |
| 64 MSps | 7.27 dB | 51.7 | 63.7 | 75.8 | 87.8 |

| to clear | at 20 MSps | at 40 MSps |
|---|---|---|
| UHF Grade B (C/N 38.4 dB) | 6.63 → **7 bits** | 6.13 → **7 bits** |
| UHF Grade A (C/N 48.4 dB) | 8.29 → **9 bits** | 7.79 → **8 bits** |

**The bandwidth requirement.** Twice the 6 MHz channel is the floor for a
direct RF or IF capture, so 12 MSps is the minimum and 20 MSps is the first
comfortable rate. For a baseband composite feed the requirement is only
twice the 4.2 MHz main sideband, 8.4 MHz, with filter margin.

**The verdict.** The same 8-bit converter this arc already uses clears
every Grade B contour with margin, and comes within a decibel of Grade A at
40 MSps. **Ten bits at 20 MSps clears everything with 10 dB to spare and is
the sensible specification.** As with the tape, the channel binds and the
converter does not: 14 bits would spend 34 dB on nothing.

**The dynamic range that actually matters** is not the converter's but the
echo measurement's. Resolving a ghost 40 dB below the direct signal needs a
measurement floor below that, and the processing gain that provides it comes
from the reference waveform, not the converter — 23.4 dB on a single GCR
line (§5.8), 38.2 dB over 30 frames, 52.9 dB over 900. Averaging frames is
what buys echo sensitivity; more bits is not.


## 9. Summary

| question | answer |
|---|---|
| effective distinguishable mechanisms, propagation | **3.79 of 6** |
| the same, tape magnetics | **1.58 of 6** |
| condition number, propagation / tape | 6.4 / 1.06 × 10⁴ |
| worst confounded pair, propagation / tape | 0.764 / **1.000** (exactly collinear) |
| multipath alone, at 1/B delay spacing | **7.92 of 8 (99%)** |
| the multipath subspace's dimension | **≈ 202** (4.2 MHz × 48 µs) |
| what breaks the tape/transmission tie | level dependence, head parity, frequency landmarks, delay sign and size, wow-factor coupling, transport rates, edge polarity, and the GCR's nested differential |
| what binds an off-air VHS capture | the **tape**, at 57.4 Mbit/s against the channel's 68.6–110.4 |
| what binds a direct channel capture | the **channel**; 10 bits at 20 MSps clears every contour |

The honest form of the result: on the tape, the total departure could be
measured precisely but which mechanism caused it could not be recovered,
because the descriptions were not distinguishable given the evidence. On
the propagation chain that ceiling largely lifts. Its filters are still
partly confounded with one another — 2.95 of 5, and the receiver's IF strip
is where the crowding is — but they are confounded, not collinear, and the
mechanism that dominates real off-air impairment is not confounded at all.
Multipath's signatures form a Fourier basis over delay, and a Fourier basis
is orthogonal by construction. That is the whole of the difference: the
tape's mechanisms are parameterised along the axis that makes them alike,
and propagation's dominant one along the axis that makes them independent.

The remaining limit is not identifiability. It is **provenance** — knowing
which chain a measured departure belongs to — and §5 is the answer to that,
with §5.1's level dependence the decisive test and §5.8's ghost-cancelling
reference the clean one where the recording still carries it.
