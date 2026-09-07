# The limit of the residuals

## Ethan's complete graph-driven multi-dimensional residual reduction

That is the design and its name. Everything in this document is his: the
limit of the residuals, the residual of the residuals, the traversal of the
measurement graph, and the reduction of every component against every other
until what remains is the noise floor. Where another lane's work appears
here it is cited as a consumer of that design or as a measurement made
against it, never as an alternative account of it.

**Ethan's design.** This document exists because the design has been arrived
at more than once and lost each time a session restarted. It is the
authority for how every measurement in this decoder is to be refined. Read
it before changing any estimator that produces a residual.

It is a design instruction, not a summary of code. Where the code already
implements a part of it, that is said, with the measurement that justified
it. Where it does not yet, that is said too.

Related but separate: `CORRECTION_DAG_DESIGN.md` declares the order in which
correction *stages* run. This document is about the order in which their
*measurements* are refined, and about what is left over when they are.

---

## 1. The instruction, in Ethan's terms

> It is measuring the luma's frequency response, comparing it against the
> expected response, and then running an inverse equalization on the RF
> signal.

> I clearly see a coherent frequency response in the accumulated table
> here. That IS a residual that is able to be removed by taking the limit.
> This is a problem that is modelable via calculus.

> Remember that all measurements and taking the limit happen in all
> measured dimensions simultaneously.

> The limit should be taken further until no change at all is seen in the
> residual, when all possible residual products are removed.

> Measure the differential between each residual for all products each time
> we do the limit. There will be a proportional change observed between the
> different residuals as we take the limit. This observed multi-dimensional
> residual of the residuals is something that we can also model. There will
> be a trajectory of these residuals that can itself be estimated and
> removed through the limit process. This should expose the direct
> relationship of each component — how does each residual affect the others
> as a whole set.

> This design will eventually be extended to remove the residuals as they
> relate to each other in the DAG. All residuals still influence
> themselves, but we can sequence the order in which we take the limit. We
> have a staged pipeline of measurements, each producing a multi-dimensional
> residual; we need to traverse this tree — siblings where two residuals
> don't relate to each other, but DO relate to their parent and their
> children. The graph is already formed based on the measurements we are
> doing now; we need to continue to advance this feed-forward design so it
> follows the direction of this graph. This will allow all residuals to
> influence their relatives as a whole, until there is a fixed and static
> residual remaining, which is the noise floor. This graph is a form of a
> residual in and of itself: since we can already derive the relationships
> between the different components we are measuring, we can also extend
> this to form the multi-dimensional and hierarchical relationship between
> measurements. That can form an expected relationship, where we calculate
> the actual relationship (through our residuals) and perform the limit on
> the expected relationship vs the actual relationship. This algorithm is a
> complete traversal of all hierarchical relationships between these
> measurements and residuals.

> The measured headswitching point, and the residual distance between past
> head switching points and the expected head switching period, will model
> the relative speed of the head drum. This may be used to further refine
> the per head frequency response, since it will vary a bit in time based on
> the rotational speed of the video drum.

> The playback heads have a fixed frequency response over the RF timebase.
> The per head frequency response (fixed head frequency, fixed head
> amplitude, fixed time) and the residual in luma frequency response (fixed
> at recording head frequency, fixed at recording head amplitude, variable
> time based) can be used as a timebase relationship component between the
> RF capture and the tape speed.

> Also, there are two sets of heads, playback head (will have a constant
> frequency response at playback), and the record heads (will vary
> proportional to tape speed, but otherwise be constant). This will be
> another dimension of measurement, measuring the consistency of the head's
> frequency response.

---

## 1a. The format, from the manufacturer's own guide

Source: JVC Video Technical Guide VTG82063, as transcribed at
`decode-orc/analogue-video-specifications`, `docs/vhs/JVC-Video-Technical-Guide-VTG82063`.
Read against this arc's measurements 2026-09-03. Sections 3 (Video Circuit)
and 4 (Servo Circuit).

**The record chain, in order** — input select, keyed AGC (sync clamped to a
fixed DC potential), Y/C separating low-pass, **pre-emphasis**, **non-linear
emphasis (EP/LP ONLY)**, white/dark clip, **frequency modulator**, recording
amplifier, and only then the down-converted chroma **mixed in after the
modulator**.

**The playback chain, in order** — preamplifier, high-pass extracting the FM
above 1.4 MHz, dropout compensator, FM AGC, double limiter, limiter,
**frequency demodulator**, low-pass, **de-emphasis**, noise limiter, output.

**Figures.** NTSC FM carrier: sync tip 3.4 ± 0.1 MHz, peak white
4.4 ± 0.1 MHz, deviation 1.0 ± 0.1 MHz (PAL 3.8 / 4.8, same deviation).
White clip 160% +10/−5%, dark clip 40% ±10%; HQ white clip 200% on LP/EP.
Colour-under 629 kHz on NTSC. Drum servo 30 Hz NTSC, 25 Hz PAL, **locked in
playback to a crystal (3 fsc) reference**. Head switch positioned
**6.014 H (NTSC) / 6.065 H (PAL) ahead of the V-sync signal**. Capstan FG by
mode: SP 1440 Hz, LP 720 Hz, EP 480 Hz.

### What the guide confirms

- **The carrier axis is exactly right.** The decoder's spec anchors put sync
  tip at 3.400 MHz and peak white at 3.6857 + 100 x 7142.86 Hz = 4.400 MHz.
  Both match the guide to the digit.
- **The switch position.** 6.014 H ahead of V-sync places the switch about
  six lines before the field's start, i.e. near line 256-257 of a 262/263
  line field. The last-region rule measures its onset at **259.6 ± 0.8 and
  258.9 ± 0.3** - two to three lines later, which is the locator's
  two-threshold region growing trimming the leading edge, plus the decoder's
  own line numbering. The rule is anchored where the format says the switch
  is, which is the check it needed.
- **The speed ratio.** SP:EP capstan FG is 1440:480, exactly 3:1, the ratio
  the wavelength argument assumed.
- **The drum is crystal-locked in playback**, which is why the switch
  position could not act as a tachometer: there is almost nothing there to
  measure, and a reading of ±9-19% per field was the locator's noise. The
  variable clock at playback is the CAPSTAN, not the drum.

### What the guide CORRECTS

**The SP-versus-EP difference is not purely a speed effect, and the
wavelength test was therefore asking the wrong question.** Non-linear
emphasis is applied on **EP and LP only** - it is not in the SP record
chain at all. So the 0.35 dB rms difference measured between the two record
speeds conflates the record head's wavelength dependence with an entire
processing stage that one arm has and the other does not. That the
difference is not a straight line in frequency (17% and 1.5% explained) is
unsurprising once that is known: a level-dependent emphasis has no reason to
look like a separation loss. Separating the two needs SP and EP of the same
content at matched levels, or an LP arm, and until then the 0.35 dB should
not be attributed to the record head.

---

## 2. The five rules

**R1 — A measurement is refined against its own residual, not against a
quantity that resembles it.**

An estimator that measures one thing and is asked to cancel another cannot
converge, however carefully it is built. This was the defect in the luma
response: the accumulated table was a median over the flattened subsample
with the collapse divided out, while the thing it had to cancel was the
full-rate deviation under the model in force. Re-imposing that table on the
deviation made the trace *worse*, which is why the code withheld it and
measured the withholding as an improvement. Feeding the residual back to
itself instead converges.

*Measured:* the binned residual falls from 24-41x the bins' own standard
error to below it, in one pass on chromanoise and two on bars; peak to peak
7-9.5% becomes 0.2-0.5%. In the decoder, per head: 50-60x the noise floor
down to 6-10x, 1.3-1.4% rms down to 0.15-0.25%.

**R2 — The limit is taken in every measured dimension at once, never in one
alone.**

A limit taken over one axis converges onto whatever the other axes were
carrying. The frequency-only limit reaches the noise floor and is *wrong*:
its table repeats within one picture at r = +0.98 but across two pictures at
only +0.49, and it agrees better between the two heads of one picture than
between two pictures of one head — the signature of one axis absorbing
another's structure. Separating the carrier's sweep rate lifts the frequency
component to +0.77 and +0.86.

**R3 — Every product of the dimensions is a component in its own right, and
must be made orthogonal to its own margins before it is applied.**

This is the rule that makes R2 work rather than oscillate. A product table
that still carries a frequency mean re-creates that mean in the frequency
component the moment it is applied. The components then fight, and the
sequence stalls far above the noise floor.

*Measured (`pencil/couple.py`):* updating the product term puts **+0.0046 dB
back into the frequency residual**, which is the size of what remains, and
the sweep stalls at 0.0048 / 0.0006 / 0.0045 dB. Two-way population-weighted
centring of the product makes the coupling matrix diagonal by the first step
and the limit converges to 0.0001 / 0.0000 / 0.0002 dB — fifty times
further. The relationship is *known*, not fitted: it is a projection.

**R4 — The differential between residuals is itself measured, every step.**

At each step, measure every component's residual; update one component;
measure them all again. The difference is that column of the coupling
matrix. A diagonal matrix means the components are separable and sweeping
them is correct. Off-diagonal mass is a relationship that must be modelled
or projected away — it is not noise, and ignoring it is what makes a limit
stall.

**R5 — The limit terminates at the noise floor, the floor is declared
rather than assumed, and it is measured OUT OF SAMPLE.**

"No change at all in the residual" is the stopping rule. What "no change"
means has to be a measured quantity: the bins' own standard error,
`1.2533 x MAD / sqrt(n)` per bin. A residual above that which stops moving
is a stalled sequence, not a converged one — check R3 and R4 before
believing it.

**And in-sample convergence is not evidence of anything.** Fitted and judged
on the same seven fields, the limit drives response, sweep and product to
0.0001 / 0.0000 / 0.0002 dB. Fitted on other fields of the same head and
tape it stands at 0.0328 / 0.0079 / 0.0563 dB — three hundred times larger,
with nothing changed but which samples judged it. So the floor a component
is measured against is its performance on fields it was not built from, and
what actually limits the decoder is not component coupling at all but the
response differing from one field to the next.

**WHAT A REPEATABILITY GATE CAN AND CANNOT CERTIFY.** The general form,
arrived at from two directions on the same day and worth more than either
instance:

> A repeatability gate bounds NOISE. Only varying the suspected confound
> bounds CONFOUNDING.

Every gate in this decoder that forms an error bar from scatter has this
shape, and each is blind to whatever is constant across the thing it
resamples. The two-bank test below resamples fields of ONE HEAD, so on a
static pattern it cannot vary content by construction, and its 0.99 was
always going to be partly a measure of how static the pattern is. The
head-switch garrote forms its error from scatter ACROSS BLOCKS, so a bias
repeating in every block - picture structure repeating line after line -
has a SMALL error precisely because it reproduces, and is admitted at full
strength. The baseband fold's belief carries two terms that exist for
exactly this reason, catching a head-owned departure and a level-dependent
one, and neither catches content.

So the honest statement of what any of them certifies is: **reproduced, and
not explained by the confounds this particular gate happens to vary.**
Establishing that a component is real needs the measurement repeated on
material where the suspected confound differs - a second content, a second
tape, a second capture.

**AND ITS COUNTERWEIGHT, which the rule above needs or it eats itself.**
Taken alone, "vary the suspected confound" argues for matching every axis
until there is nothing left to average over. The limit on it:

> Refining an axis pays only when the pooling is over something that DIFFERS
> between the arms being compared. Where it does not differ, refinement is
> pure loss of samples.

Both signs of that were measured on the same evening, on the same question,
in two lanes. The head-switch lane's band average pools over a carrier
occupancy that genuinely differs between the heads, so matching the carrier
position is a gain. My sweep mix does NOT differ between the heads, so
matching it cost a factor of two in correlation - +0.427 down to +0.222 -
and bought nothing. Same operation, opposite sign, decided by one question.
The average is not the error; pooling over the WRONG variable is, and which
variable is wrong depends entirely on what is being compared.

**A corollary worth stating, because it turns a null into something
usable.** An instrument that cannot resolve a difference should report the
size it could have seen rather than that it saw none. The head-switch lane's
withdrawn null became: at thirteen fields per head, no head difference
larger than about a sixth of the coupling exists in its most sensitive band,
and anything smaller is invisible to that instrument - which, against the
0.083 dB on 2.1 dB common measured here, is four to fifteen times too blunt
to have seen it, and would need several hundred to nearly three thousand
fields per head to get there. A bound costs nothing to compute and says what
a null cannot.

**A PRE-REGISTERED TEST OF THE RULING ITSELF.** Pooling different contents
to raise the evidence is legitimate only under Ethan's ruling that the
response does not depend on what the luma encodes - so the pooling contains
a free test of the ruling it rests on, and the test can be stated as a
number BEFORE it is run rather than judged afterwards.

Calibrating the density model on what is already observed - 0.083 dB of
signal against 0.096 dB of per-bin noise at thirteen fields, predicting
r = 0.428 against a measured +0.427 to +0.446 - the correlation between two
estimates each pooled from k contents should be:

| contents pooled per estimate | noise | predicted r |
|---|---|---|
| one | 0.096 dB | 0.428 (observed) |
| two | 0.068 dB | **0.599** |
| three | 0.055 dB | **0.692** |
| four | 0.048 dB | 0.749 |

If three contents land near +0.69 the ruling is confirmed to the precision
this material allows. If they plateau below it, the shortfall IS the
content-dependent part of the response, and it converts directly: an
observed 0.65 means 0.025 dB of content dependence, 0.60 means 0.039,
0.55 means 0.051, 0.50 means 0.062. The sensitivity floor is therefore about
0.05 dB, against the 0.083 dB head difference being chased - enough to
detect a content dependence comparable to the effect, not one much smaller.
Stating the floor in advance is what stops a marginal result being read
either way afterwards.

Two caveats on the prediction, both of which would depress the correlation
for reasons unrelated to the ruling: it assumes the noise is INDEPENDENT
between contents, so a shared systematic - one deck, one session, one
estimator bias - makes the floors correlate and the number fall short; and
it assumes comparable per-bin population, so a bin left sparse by one
content does not carry that content's full weight, which is the same density
argument one level down.

**The admission that follows.** Each component is accumulated in two banks
on alternating fields, and what is applied is scaled by how far the banks
agree — their population-weighted correlation, clipped at zero
(`residual_limit.agreement`). A component describing the path reads near
one and passes through; a component describing the field it was measured
from reads near zero and is withheld. No constant is chosen anywhere.
**Bank on the component's OWN evidence, not the decode's.** Fields alternate
heads and the stores are kept per head, so a bank index that flips once per
field puts every one of a head's fields into the same bank and leaves the
other empty. The agreement test then has nothing to compare and returns zero
for a reason that has nothing to do with the component. This file made that
mistake and read "0.000, does not reproduce" for the product component; the
test had never run. Banked on each head's own field count instead, the same
component reads **0.47 to 0.72** and is admitted at that fraction, and the
head-specific response reads **0.988 to 0.994** on every condition - the
heads genuinely differ, and the difference is a path property.

**R6 — The head is a measured dimension, not two models kept apart, and
there are TWO SETS of heads.**

Ethan: the playback head's frequency response is **fixed to the time base of
the RF capture — completely fixed in time**, varying with nothing. The
record head's varies **proportionally to tape speed** and is otherwise
constant. They are two chains, playback and record, and the consistency of
each head's response is itself a dimension to measure.

That gives the response a three-way split, and it has been measured on
75 bars, same deck, same content, at both record speeds (14 frames each):

| part | size | what it is |
|---|---|---|
| common to both heads and both speeds | 2.1 dB rms | fixed in time: the playback chain |
| differs between the two RECORD speeds | 0.35 dB rms | the record head, which varies with tape speed |
| differs between the two heads at one speed | 0.115 dB (SP), 0.180 (EP) | head to head |

**A SCATTER ACROSS CONTENT, and what it is NOT.** Ethan's ruling, and it is
the physics: *"The full frequency response IS the thing we are measuring and
includes the content. The frequency response is not dependent on what is
actually encoded in the luma."* The response is a function of CARRIER
FREQUENCY. What the picture encodes decides which frequencies are visited
and how often - the sampling - not the value at a frequency. So a difference
between the odd and even fields' content cannot make the response differ; it
can only change how well each frequency is measured.

An earlier draft of this section got that wrong, following the head-switch
lane's parity confound too far and calling the per-head departure "part
content". That was a claim about the physics and it was incorrect. A second
draft then blamed the residual scatter on the SWEEP MIX at a given
frequency. That was tested and is also wrong. What the scatter tracks is
EVIDENCE DENSITY, and three independent slices say so together:

| comparison, bars SP vs chromanoise SP | r | samples behind each estimate |
|---|---|---|
| the sync band alone (3.40-3.69 MHz) | **+0.607** | densest - every line dwells there |
| the whole range | +0.459 | |
| above blanking only | +0.347 | sparser |
| split into (frequency x sweep) CELLS | +0.222 | sparsest |

Splitting by sweep rate makes it WORSE, not better, which is the opposite of
what a sweep-mix contaminant would do - and within that split the
correlation rises monotonically with how many fields have described a cell:
+0.207 at four fields, +0.223 at eight, +0.273 at twelve, +0.295 at
thirteen, while the difference's own rms falls from 0.159 to 0.126 dB. That
is noise thinning out, not a confound being removed.

So the head difference is a function of FREQUENCY, as Ethan's ruling says,
and binning by frequency alone is the better estimator because averaging
over the sweep mix costs nothing and buys samples. Taking r = A²/(A²+s²) at
face value on the frequency-binned figures, the true per-head difference is
about 0.083 dB against a per-bin measurement noise of about 0.096 dB on
thirteen fields: real, and comparable in size to what thirteen fields can
resolve. The measurement is EVIDENCE-LIMITED, and more fields raise it. And the
channel-loop lane has an independent, content-free witness: a per-head gain
measured on the SYNC TIP - one fixed carrier, constant envelope, no active
video in the window at all - reading +0.98 dB pooled at 161 sigma across
thirteen tape positions, and agreeing with a carrier-binned measurement of
the same difference at three of them. A real per-head difference exists, and
nothing in the picture can have reached that measurement.

What remains true and useful from the confound: a gate that resamples fields
of one head cannot vary the sweep mix either, so it still bounds noise
rather than that particular contamination. The numbers, for the record:

| | head A − head B | reproduces against |
|---|---|---|
| 75 bars SP | 0.115 dB rms | chromanoise SP at **r = +0.446** |
| chromanoise SP | 0.087 dB rms | 75 bars EP at r = +0.085 |
| 75 bars EP | 0.180 dB rms | 75 bars SP at **r = −0.085** |

So the head difference reproduces across content at one speed only
partially - under half its variance is shared - and across speeds not at
all, though that last comparison cannot settle anything because EP carries a
non-linear emphasis stage SP does not have. The honest reading is that the
per-head departure is PART head and part odd-versus-even content, and until
it is measured on moving content the split is unknown.

**And this is a blind spot in R5's admission test, which matters more than
the number.** The two-bank agreement resamples FIELDS OF ONE HEAD. On a
static pattern those fields carry identical content, so a content-driven
difference reproduces perfectly and reads a trust of 0.99 - exactly what the
head departure reads. The test cannot see a confound that is constant within
a decode. It bounds noise, not confounding, and a component that passes it
has been shown to be repeatable and NOT shown to be real.

**The head identifies itself from that split.** Remove the response the two
heads share and correlate what remains against each head's own departure:
the right head is picked on **25 of 25 fields on 75 bars and 25 of 26 on
chromanoise**, with the own-head correlation +0.587 / +0.508 against
−0.563 / −0.451 for the other. This is the missing piece for `head_switch`,
whose calibration pools both heads because "blocks carry no field parity".

**Measured and negative, so it is not reopened on a guess:** the
speed-dependent part is NOT a straight line in frequency. A record head
writing a tape three times slower writes three times shorter wavelengths, so
a separation loss exp(-2πd/λ) would be three times the exponent and, since
1/λ goes as f, the SP-minus-EP difference would be a line. Fitted, that line
explains **17% on head A and 1.5% on head B** of a 0.35 dB difference. At
this precision the record head's speed dependence is not a simple separation
loss.

**R7 — The two time bases, and the relationship between them, are
measurable from the frequency response alone.**

Ethan: the playback head's response is fixed in the RF CAPTURE's time base -
fixed head frequency, fixed head amplitude, fixed time. The residual in the
luma frequency response is fixed in the RECORD head's - fixed recording
frequency, fixed recording amplitude - but VARIABLE in time base. Two
anchors, one static in the capture's clock and one static in the tape's.

The consequence is a measurement. A speed error scales every recorded
frequency by (1+e), so the model is read at the wrong frequency and the
amplitude error it produces is not arbitrary: it is

    ln A - model(f)  =  model(f(1+e)) - model(f)  ~=  e * f * d(model)/df

- the model's OWN slope, times frequency, times the fractional speed error.
So regressing the amplitude residual on that signature returns `e`
directly, and `e` is the ratio between the two time bases: the tape's speed
against the capture's clock.

That makes it a **time-base relationship component**, and it is checkable
rather than merely plausible, because the same `e` is already measured
another way entirely - the sync edges give the line period against the spec
period, which is the same fractional speed error read from timing instead of
from amplitude. Two independent instruments, one quantity. Where they agree
the relationship is established; where they part, the part of the amplitude
residual that is speed is separable from the part that is loss, and only the
first is model error to be removed.

*Measured, three conditions, 27-28 fields each.* The fractional speed error
read from the response against the same quantity read from the sync edges:

| | from the response | from the sync edges |
|---|---|---|
| 75 bars SP | −0.00015 ± 0.00020 | −0.00013 ± 0.00003 |
| chromanoise SP | −0.00013 ± 0.00025 | −0.00014 ± 0.00014 |
| 75 bars EP | +0.00006 ± 0.00027 | −0.00007 ± 0.00012 |

**The mean relationship is confirmed**: on both SP conditions the two
instruments agree to about 2e-5. They share no arithmetic - one is an
amplitude regression, the other a count of sample intervals - so the
agreement is the relationship, established.

**BUT THE ATTRIBUTION WAS WRONG, and this corrects what stood here.** It was
first recorded as "a tape running some 0.014% slow". It is not the tape. A
helical head reads at 5.80 m/s while the tape moves at 33.35 mm/s, so a
tape-speed error is diluted **174 : 1** into the line rate: producing
−1.4e-4 of line rate would need a **2.43% tape-speed error**, five times
outside the format's own ±0.5% tolerance - and that entire tolerance is
worth only ±29 ppm of line rate. The capstan is a phase servo against the
control track and has no steady-state rate error to give. And the drum is
crystal-locked in playback while both instruments answer in CAPTURE-clock
units, so what they agree about is the ratio between the deck's timebase and
the capture's clock: an ordinary crystal offset of −140 ppm. Two instruments
agreeing in capture units cannot discriminate the tape at all. The
RELATIONSHIP survives; only its name changes.

**The per-field variation is not shared** (field-to-field correlation +0.21,
−0.15, −0.40) and the response's scatter is two to ten times the sync
edges'. The response is a poor speedometer moment to moment; the sync edges
are the instrument for the varying part, and the response's contribution is
the constant relationship.

**And a consequence that settles a live question**: the speed signature
explains **0.01% of the amplitude residual**. So the residual is essentially
NOT the model being read at the wrong frequency - it is real amplitude
behaviour. It is therefore tape, not model error, and it must stay in the
deviation and reach the colour-under rather than being removed as a
correction to the frequency axis.

**R8 — The DRUM's speed is a separate clock again, measured from the head
switch, and the per-head response follows it.**

Ethan: the measured head-switching point, and the residual between the
distance separating successive switches and the period they are expected to
be apart, model the relative speed of the head drum. The per-head frequency
response varies a little in time with that rotational speed.

It is a third clock and it is the one that matters for a head's response.
The tape's speed sets track spacing; the DRUM's rotation sets the
head-to-tape writing speed, and that is what sets the wavelength a head
writes and reads - so a response that depends on wavelength depends on the
drum, not on the tape.

It is nearly free to measure. `head_switch.locate` already returns up to two
switch regions in one field - the field's own at its first lines and the
next field's at its tail - because the field's data overhangs, so the
distance between them is one switch period measured inside a single field's
own sample grid, with no cross-field bookkeeping. Against the period the
format expects (the field's own line count times the line period), the
residual is the drum's relative speed.

**R9 — The head is a measured dimension, not two models kept apart.**

Two independent per-head tables halve the evidence for everything the heads
have in common, which is most of the response: same tape, same electronics,
same band, differing in the gap that wrote them. Decomposed instead - a
shared component carrying both heads' evidence, plus each head's departure
from it, orthogonal to the shared term by construction and admitted by R5 -
the variance BETWEEN the heads is described rather than merely duplicated.
`_head_model` in `luma_amplitude.py`.

---

## 2a. RULING: no matrix pencil. The sync shape in Hilbert space instead

Ethan, 2026-09-06:

> I think the matrix pencil is the wrong approach. Use the existing sync
> shape modeling in hilbert space not the matrix pencil.

**The reason, and it is a measurement rather than a preference.** A matrix
pencil fits damped exponentials, and that requires an ORDER. On planted
data the pencil validated and beat its alternative; on real data there is
no singular-value knee to set the order, so the answer moves with the
pencil's own parameter. `vhsdecode/luma_amplitude.py:2012-2017` already
recorded exactly that for the response ripple.

The evidence on the sync pulse itself. The pencil's count of ringing modes
against the effective rank the sync shape's own fit supports, measured on
the same accumulated profiles from the same exports:

| export | head | pencil modes | shape rank, of 49 |
|---|---|---|---|
| cd | a | 8 | 8.79 |
| cd | b | 7 | 8.52 |
| home | a | 10 | 10.11 |
| home | b | 3 | 10.08 |
| pnb | a | 4 | 10.33 |
| pnb | b | 3 | 8.90 |

The two heads of the home tape read one signal path through one
demodulator. The pencil says ten modes on one and three on the other, a
factor of 3.3; the shape's rank moves from 10.11 to 10.08, a factor of
1.003. Across all six the pencil's count varies by 51 per cent about its
mean and the rank by 8.5 per cent. **A number that swings by three between
two readings of the same thing is a setting, not a measurement.**

**What replaces it.** `vhsdecode/models/sync_shape.py` with
`vhsdecode/models/hypercomplex.py`. The same three axes come off one
non-parametric fit of the pulse's shape: a frequency axis, a magnitude on
it, and a group delay on it, with the effective rank REPORTED rather than
chosen. The time axis gains something the pencil could not express at all,
because a minimum-phase response's delay is fixed by its own magnitude
through the Bode relation: the group delay splits into the share the
amplitude already implies (`hypercomplex.minimum_phase`) and the excess
that is a genuinely separate mechanism (`hypercomplex.excess_phase`). A
damped-mode fit has no way to say which of its decay times its own
residues had already implied. The fit also returns the out-of-band
remainder, which bounds what the shape does not describe and which the
pencil returned nothing of.

Registered as `information_extrapolation.sync_shape_components`. The
superseded entry is left in place as
`information_extrapolation.sync_pole_components` so the chain's position
is not silently vacated and the reason stays on the record.

## 3. What driving a residual to zero does NOT license

The deviation is *supposed to be* the tape's own amplitude error. A model
that absorbs picture-locked structure stops the correction correcting. So a
component is admitted only if it reproduces on evidence independent of the
picture it was measured from, and the record-referenced decode remains the
judge.

*Measured:* frequency main effect reproduces across pictures at r = +0.77 /
+0.86 and sweep-rate at +0.77 / +0.83 — both admissible. The raw product
term reproduces at only +0.25 / +0.19, so it is the term most likely to be
describing the picture, and it must clear the record-referenced battery on
its own before it is trusted. See `correction gain law`: the applied
fraction that maximises expected gain is 1/(1+rho) where rho is model error,
and model error is invisible to any resampling of the data the model was
fitted from.

---

## 3a. The two stages, and the matrix over the differentials

Ethan's architecture, in his words:

> The evidence is in the properties of rotational velocity. Assuming the
> tape speed is varying with the head drum, or the head drum itself is also
> varying, we will have two components to derive. It will be the amplitude,
> phase, and frequency of this differential pair. We have three sets of
> three differential pairs, which form a matrix to calculate over the
> residuals. Use these 3d matrices to correct the rf part of the signal
> (playback, and record vcrs), then I can use the other components specific
> to the Picture stage: sync pulse amplitude and phase, genlocking (align
> colour burst to hsync), burst locking (align the hsyncs to the colour
> burst), and timebase residual (existing residual subtracted out from
> constant).

Three things follow that this arc has not been doing.

**1. A differential pair is characterised by THREE quantities, not one.**
Amplitude, phase, and frequency. Everything measured on the head
differential so far is amplitude alone - 0.094 dB rms of shape on a
−0.556 dB level, static and not drifting. Its PHASE has never been
measured, though the material for it exists: the sync-edge instrument
already produced a per-head COMPLEX response, so the differential's phase is
available from data in hand. Its FREQUENCY - each head reading at a slightly
different effective frequency if the speed differs between them - has not
been measured either.

**2. The pairs form a MATRIX over the residuals**, three sets of three, and
the matrix is what corrects the RF part of the signal. Note the plural in
"playback, and record vcrs": there are two chains and the correction is
owed to both, which is the same two-tape-chain hierarchy recorded in §1.

**3. The PICTURE stage is separate and has its own named components**, and
they are not the RF ones: sync pulse amplitude and phase; genlocking, which
aligns the colour burst to hsync; burst locking, which aligns the hsyncs to
the colour burst; and the time-base residual, which is the existing residual
subtracted from a constant. The RF matrix is corrected first; the picture
components are applied after, on quantities the RF stage does not carry.

This is the shape the traversal below has to grow into: the graph's RF layer
carries the differential matrix, and its picture layer carries these four.

### 3a.0 The sync pulse is the dimension both layers share

Ethan: *"Picture and both rf stage have sync pulses, since the sync pulses
represent the timebase correction. That part can be connected up to form yet
another dimension on both layers."*

This is the EDGE between the two layers, and it is the only quantity that
exists on both. The RF layer and the picture layer otherwise carry different
components - a differential matrix on one, sync amplitude and phase,
genlocking, burst locking and the time-base residual on the other - and
without a shared dimension they are two graphs rather than one. The sync
pulse is that dimension, because it is present before demodulation and
after, and because what it represents in both places is the same thing: the
time base.

Both sides are already instrumented, which is what makes the connection
buildable rather than aspirational:

| | on the RF layer | on the picture layer |
|---|---|---|
| where the pulse is | `sync_edge_trace` - per-line 50% crossings measured from the CARRIER | the decoded sync pulse in the TBC'd line |
| its shape | the hsync impulse response, per head, complex, 0.16-1.9 MHz | the sync-deficit product against the spec-shaped ideal |
| its level | the carrier at sync tip and blanking (`_measure_level_anchors`) | the decoded tip and porch in IRE |
| its timing | the line period against the SPEC period, on the track axis | burst phase, genlock, burst lock |

The two are the same pulse seen on either side of the demodulator, so a
residual measured on one is a prediction about the other, and the difference
between them belongs to the demodulator and to nothing else. That is a
stronger constraint than either layer carries alone, and it is what lets the
traversal cross from RF to picture instead of stopping at the boundary the
collapse groups already mark.

### 3a.1 The head differential's PHASE, measured

The first of the three quantities to be added. Taken from the sync-edge
instrument's per-head complex response, which is content-free by
construction, as H(head A) / H(head B) with a pure delay removed:

| capture | edge | \|A/B\| dB at 0.25/0.5/1.0/1.5 MHz | arg A/B, degrees | delay |
|---|---|---|---|---|
| bounce | fall | +0.13 −0.08 −0.28 −0.20 | +1.0 −0.4 −1.1 +0.2 | +1.5 ns |
| chromanoise | fall | +0.04 −0.03 −0.28 −0.38 | +0.9 −0.0 −0.8 −0.1 | 0.0 ns |
| 75 bars | fall | −0.02 −0.04 −0.08 +0.03 | +0.6 +0.8 −0.4 +1.5 | −1.0 ns |
| bars EP | fall | +0.05 +0.31 +0.54 +0.91 | +1.5 −1.1 +0.1 +1.7 | −7.1 ns |

**The phase is zero.** Every entry sits within ±1.7 degrees once a delay of
a few nanoseconds is removed, on both edges and all four captures.

Two things follow. **First, it supports the z-offset reading**: a spacing
difference between the heads is an amplitude loss by Wallace's law and
carries no phase, whereas a gap-length or azimuth difference would show
phase. Amplitude without phase is what a head sitting a few tens of
nanometres further from the tape produces, and that is the same 13.5 to
39.8 nm arrived at from the level split.

**Second, and it locates the correction**: this is a BASEBAND measurement,
taken after demodulation, and the FM limiter removes RF amplitude. So a
0.556 dB head difference in the RF envelope arrives in baseband as the
±0.3 dB seen here. The head asymmetry is an RF-domain quantity and has to be
corrected there - which is exactly where Ethan's differential matrix
applies, and why the picture stage is a separate set of components rather
than the same ones seen later.

## 3b. Can all the residuals be related by one expanding series?

Ethan asked it directly. The answer is yes in form, and this arc's own
measurements say what governs it in practice.

**The form.** In the log domain the model is a sum over every measured
dimension and every product of them - frequency, sweep, track position, time
base, head; then the pairwise products; then the triples - and the limit
drives each term to its own noise floor. That is one expanding series, and
everything built here is a piece of it: main effects are first order, the
frequency-by-sweep product is second order, and the coupling matrix between
components is the series read one level up. The residual of the residuals is
not a separate idea; it is the next term.

**Three governors, each measured rather than assumed.**

1. **Every term divides the same evidence.** Splitting frequency by sweep
   into cells cost more in noise than it recovered - the cross-content
   correlation fell from +0.427 to +0.222 - and within that split it rose
   monotonically with fields per cell, +0.207 at four to +0.295 at thirteen.
   A term exists only while samples stand behind it.
2. **Terms must be made orthogonal or they fight.** Measured: an uncentred
   product returns +0.0046 dB into the frequency residual and the sweep
   stalls at 0.0048 / 0.0006 / 0.0045 dB. Projected orthogonal to its own
   margins it reaches 0.0001 / 0.0000 / 0.0002 - fifty times further. An
   un-orthogonalised series does not converge; it oscillates.
3. **A term that does not reproduce is not a term.** The high-end slope
   looked like one and was the picture. The switch-window coupling looked
   like one and was an artefact two instruments could both see. Each term
   needs its own admission, and the admission must VARY the suspected
   confound rather than resample the same thing.

### 3b.1 Does it plateau? Measured: no.

Ethan: *"I believe this can be run indefinitely to a limit that approaches
the noise floor, which is the total possible information within this sample,
full band of 8bit 40mhz."*

This is the claim that decides whether to spend on more terms or more
material, and it is directly testable: pool k fields and watch how the
residual's scatter falls. NOISE falls as one over the root of k without
limit. STRUCTURE flattens, because a residual correlated across fields does
not average away.

Measured, per head, on the response over carrier frequency:

| fields pooled | head A scatter, dB | head B scatter, dB | 1/sqrt(k) predicts |
|---|---|---|---|
| 2 | 0.0497 | 0.0476 | 0.0497 / 0.0476 |
| 3 | 0.0397 | 0.0359 | 0.0406 / 0.0389 |
| 4 | 0.0353 | 0.0336 | 0.0352 / 0.0337 |
| 6 | 0.0246 | 0.0205 | 0.0287 / 0.0275 |
| 9 | 0.0180 | 0.0155 | 0.0234 / 0.0224 |

**It does not plateau.** Through nine fields it tracks the noise line and
then falls slightly FASTER, on both heads independently. There is no floor
of unmodelled structure being reached at 0.018 dB, and Ethan's reading of
these numbers is the correct one: the sequence keeps descending, and what it
descends toward is the capture's own information floor rather than a wall in
the model.

### 3b.2 The two floors, in the capture's own units

The capture is 8-bit PCM at 50 MSps, using 116 of 256 codes, signal 24.9 LSB
rms, so a carrier amplitude of 35.2 LSB. That fixes the floor Ethan names,
and it turns out there are two of them:

| floor | per sample | per bin at 13 fields |
|---|---|---|
| the tape, measured within-field spread | 0.669 dB | 0.0081 dB |
| the 8-bit capture, quantisation over a 6 MHz envelope band | 0.0247 dB | 0.00030 dB |

Quantisation is 0.14 per cent of the per-sample variance; everything else is
tape. And **we sit at 0.0150 dB - 1.8x the tape floor and 50x the 8-bit
floor**. So the model is already close to what the tape's own noise permits
on thirteen fields, and the remaining fifty-fold to the capture's limit is
reachable only by pooling more of the tape, which is exactly the 1/sqrt(k)
descent the table above shows continuing.

### 3b.3 The corrected terms, and a mistake worth recording

An earlier revision of this section put the first-order term at
signal-to-noise 0.86 and asked for 157 fields. That was wrong: it compared
the term against the PER-FIELD scatter where the POOLED scatter belongs. The
correct figures, against the 0.0150 dB the thirteen fields actually give:

| term | size | pooled noise | signal-to-noise | fields for S/N 3 |
|---|---|---|---|---|
| first order, frequency | 0.083 dB | 0.0150 dB | 5.5 | 4 |
| second order, frequency x sweep | 0.060 dB | 0.0305 dB | 2.0 | 30 |

So the second-order term is nearly established on the material in hand and
needs about thirty fields per head to be firm, against the thirteen a
11 MB capture holds - roughly two and a half captures, not ten. The series
has considerably more room than the earlier number claimed.

The three governors above still hold, and the correction does not touch
them: terms still divide evidence, still have to be orthogonalised, still
have to reproduce out of sample. What changes is the verdict on where the
sequence stops. It does not stop at second order on this bench; it descends
as the material grows, toward a floor set by the tape first and by the
capture's eight bits far below that.

### 3b.4 Not Chaitin's constant; minimum description length

The question was put twice, and the second form is the right one.

Chaitin's constant is the halting probability of a universal prefix-free
machine - one specific uncomputable real, defined over programs rather than
measurements. Nothing in this arc converges to it, and the floor measured
above is a physical quantity that was computed in §3b.2 rather than an
uncomputable one. The claim as first stated does not hold.

*"The limit of information encodable within a 3d data is fully representable
to a certain floor"* - that one is exactly what the measurement shows, and
it is the useful statement. What algorithmic information theory does
contribute here is its computable cousin, minimum description length: a term
is real when the bits it saves on the data exceed the bits it costs to write
down. Over an 11.0 M sample, 87.8 Mbit capture:

| term | measured | true size | parameters | costs | saves | return |
|---|---|---|---|---|---|---|
| first order, frequency | 0.083 dB | 0.0816 dB | 238 | 1 515 bit | 118 738 bit | 78x |
| second order, f x sweep | 0.060 dB | 0.0517 dB | 2 618 | 12 137 bit | 47 344 bit | 4x |
| third order, + track | 0.030 dB | none | 20 944 | - | - | rejected |

Two pieces of arithmetic are load bearing and were both wrong on the first
pass. A measured term contains its own noise, so its TRUE size is the
measured size with the noise removed in quadrature - which is what rejects
third order outright, since 0.030 dB against a 0.053 dB cell noise leaves
nothing. And a parameter that cannot be determined still costs its address,
so the cost floor is half the log of the samples behind it rather than
something that vanishes as the term gets weak.

The result worth keeping: **MDL terminates between second and third order,
which is where the out-of-sample agreement rule already puts it**, reached by
arithmetic that shares nothing with it. Two independent stopping rules
landing in the same place is the strongest statement this arc can make about
where the series ends.

And what remains at the bottom is not a limit of the method. At the 8-bit
floor the residual is the tape's own randomness, which no model compresses
because it is not structure.

## 3c. THE COLLAPSE: fit the residuals to an ellipse (2026-09-04)

Ethan, over four messages, arriving at a closed form for the whole thing:

> Complex differential over the period of as many matched components are
> differentiated together in a matrice of their component's differentials,
> etc. all the way down for all matched sets of components. The same
> dimension of components extend the same sets of component in all matching
> dimensions, an matrix that is expanding by a dimension each time it is run
> through this algorithm.

> Even the magnetic noise profile of the tape is know, the vcr can be
> correlated for yet a futher dimension to this which expands it yet again.

> This forms a 3 dimensional elipse of differentiation between thes
> dimensions which is calculus and can be simplified.

> The shapes of the dimensions are asymetric, and we can extrapolate how
> symetric they are based on their fit to our 3 dimensional eliptical plane
> which represents the final result. The constant is the elipse. All I need
> is to fit these residuals to an elipse of how ever many dimensions I have,
> and I have fully collapsed the differentiation.

> elliptical curve estimation using multi dimensional complex components

**He is right, and it is a genuine collapse rather than an acceleration.**
The rank-increasing recursion converges on the eigenbasis of the ensemble's
second moment; solving that eigenproblem reaches the same place in one step,
with no passes, no chain order and no half-steps. Implemented as
`ellipsoid(residuals, axis)` in `vhsdecode/models/information_extrapolation.py`.

**The constant is the ellipse, and it is exact.** Every component enters as
a unit direction, so `trace(D D^H)` is the component count whatever the
components do. Measured on four complex components:

| ensemble | rank | trace | asymmetry | semi-axes |
|---|---|---|---|---|
| four independent | 4 of 4 | 4.0000000000 | 0.029 | 1.118 0.974 0.966 0.932 |
| four views of ONE departure | 1 of 4 | 4.0000000000 | 1.000 | 2.000 0.000 0.000 0.000 |
| shared plus private | 4 of 4 | 4.0000000000 | 0.248 | 1.350 0.906 0.834 0.813 |

The total is conserved to ten decimal places across all three, so only the
SHAPE carries information - which is exactly why the asymmetry is the
measurement and the ellipse is the invariant. A sphere means the components
are independent; a needle means they are four views of one thing. **The rank
is the number of dimensions the evidence supports, read in one step**, which
is the same question the pass-by-pass recursion answers slowly.

### 3c.1 Three defects the collapse depended on, found and repaired

None of the above worked before, and the reasons were latent because they
only bite once there are more than two components, more than one grid, or a
complex value:

1. **The projection was grid-blind.** `_directions` truncated every vector
   to the shortest (`width = min(v.size ...)`) and took inner products
   across them, so a 238-bin frequency response and a 262-line time residual
   returned a coherence that was an artefact of memory layout. Now
   components are grouped by grid and only projected within a group; the
   mixed pair returns 0.0.
2. **Phase was silently discarded.** Both orthogonalisers cast with
   `dtype=np.float64`, which on a complex array keeps the real part and
   throws the imaginary one away - defeating `COMPLEX_COMPONENTS`, which
   declares all three axes complex. This was not a precision loss: stripping
   the phase from the same four components moves the ellipsoid's asymmetry
   from 0.248 to 0.876, a different answer to a different question.
3. **The field axis was never orthogonalised.** Both loops iterated `AXES`,
   which omits `FIELD_AXIS`, so a per-field dimension could be measured and
   accumulated and still contribute nothing to the descent.

The repair is required to be **bitwise identical on the live ensemble** -
two same-length real components on the frequency axis - and a test asserts
that against a replica of the pre-repair arithmetic. Anything else would
mean behaviour changed rather than a latent defect being fixed.

### 3c.2 The tape's magnetic noise profile as a dimension

Ethan's third message promotes the noise floor from a terminus to a
component. The profile is already modelled: `models/tape_model.py:237`
`magnetic_limits(writing_speed_m_s, track_width_m, ...)` returns
`volume_at` / `particles_at` / `snr_db_at`, and `binding_limit` (:286)
combines the tape's SNR with the capture's. So the floor has a PREDICTED
SHAPE from particle physics, not merely a level - and a floor with a
predicted shape is something the VCR can be correlated against, which is one
more dimension rather than the end of the list.

### 3c.3 The loop, and the halting count - measured

Ethan completed the algorithm over three more messages:

> Then differentiation using the target curve onto the source data.
> This loops until the information floor is reached (total possible
> information on the source signal) 4d eliptical curve I believe.
> Down to the circular shape of the complex signal.

> You can increase the constant, how many components to know exactly when
> you have reached the end, since the end of this is not known.

> I found chaitan's results given the number of dimensions supplied. I
> found the number of times you need to run this differentiation before
> you're done. and it is an elipse, i.e. multi dimetional set of components
> mapped over the a sphere, i.e. multi-D complex plane.

Shipped as `differentiate_to_floor(residuals, axis)`. Each pass fits the
ellipse, takes every direction standing above the noise edge as the TARGET
CURVE, differentiates it onto the source, and stops when what remains is as
spherical as randomness of that size and length would be.

**Two closed forms carry the stopping rule, so it holds no fitted constant.**
The sphere floor: with `trace(G) = N` and `E|G_ij|^2 = 1/L`, the asymmetry a
random ensemble shows is `N / (L + N - 1)`, and its scatter is `sqrt(2) / L`.
Checked over N in 3..32 and L in 64..1024, real and complex: measured over
predicted sits at 0.88 to 1.18 with no trend, and the scatter fit returns
1.408 against `sqrt(2) = 1.414`. Which directions are real is the
Marchenko-Pastur edge `(1 + sqrt(N/L))^2`, measured conservative on noise at
0.83 to 0.96 of the edge, so it never admits noise as a direction.

**The halting count, measured.** Planting a known number of departures and
running to the circle:

| components | planted | passes | recovered | reached the circle |
|---|---|---|---|---|
| 6 | 0 | 1.0 | 0.0 | 100% |
| 6 | 1-2 | 2.0 | 1.0, 2.0 | 100% |
| 6 | 4 | 1.0 | 2.0 | **0%** |
| 10 | 1-4 | 2.0 | 1.0-4.0 | 100% |
| 10 | 5 | 2.9 | 7.6 | 62% |
| 16 | 1-5 | 2.0 | 1.0-5.0 | 100% |

Ethan's claim holds, with one correction worth making because it is the
collapse doing its work. **The pass count does not grow with the dimensions
- it is two, flat**, one to remove the whole ellipse and one to confirm the
circle, because every direction above the noise edge comes out together. The
quantity that tracks the dimensions supplied is the number of DIRECTIONS
RECOVERED, and that equals the number planted exactly at 10 and 16
components across one to five departures.

**And his other claim is the limit that shows up in the same table.** The
loop fails when the planted dimensions approach the component count: six
components cannot resolve four departures at all, ten resolve five only 62%
of the time, sixteen resolve five every time. So roughly `N - 2` dimensions
are resolvable from `N` components, which is exactly *"you can increase the
constant, how many components, to know exactly when you have reached the
end"*. More components is what buys more resolvable dimensions.

This is not Chaitin's constant, for the reasons in 3b.4. But the operational
half of the claim - that the number of runs before you are done is fixed by
the dimensions supplied - is confirmed and now quantified.

### 3c.4 The self-inflicted artefact this round, and the rule it gives

Deflating a direction drawn from the ensemble's OWN span leaves an exact
zero eigenvalue: the set then spans one dimension fewer. Counting that zero
as an unequal share reports the deflation's own bookkeeping as structure,
and the loop never reaches the floor - it descended, then rose, then
oscillated. One planted departure removed cleanly, at 0.892 alignment with
the planted vector, read 0.224 against a 0.023 floor with the zero counted
and 0.019 against a 0.019 floor without it.

The fix is not "drop the zeros", because a genuine needle - N views of one
departure - also has N-1 zero eigenvalues and that IS maximal asymmetry. Only
the caller knows how many directions it has removed, so the count is passed
in. **The rule: a shape measure must be told the dimensionality it is
judging, or it will measure its own deflation.**

### 3c.5 "Is this the Wiener transform for the complex number convolution?"

Ethan's question, and the answer is close enough to yes that it is the right
name for it, with one qualification worth keeping.

**What it is.** The ensemble's second moment is diagonalised - that is the
Karhunen-Loeve transform, the basis in which the components are
uncorrelated - and each principal direction is then weighted by
`(lambda - bulk) / lambda`. **The Wiener filter is diagonal in the KLT
basis**, which is the classical result, so fitting the ellipse finds the
basis and the per-direction weight is the Wiener filter applied in it. On
complex components throughout, which is what RF is: *"just complex numbers
though, i.e. radio waves which are what we have on our video tape"*.

**The weight is this project's own correction-gain law.** A direction
carrying signal stands at `lambda = bulk + signal`, so

    w = (lambda - bulk) / lambda = SNR / (1 + SNR) = 1 / (1 + rho)

which is `a* = 1/(1 + rho)` exactly, for every eigenvalue, and the flat
half-step is its special case at a signal-to-noise ratio of one. Two
derivations from opposite directions landing on the same expression.

**The qualification.** A textbook Wiener filter needs the signal and noise
spectra known in advance. Here the noise level is read from the ensemble's
own eigenvalue bulk - the eigenvalues that are not directions - so it is
self-calibrating, which makes it a shrinkage estimator of the Wiener form
rather than a Wiener filter proper. That is the same discipline the rest of
this arc uses: the floor is measured beside the signal, in the same chain,
at the same moment.

**It is measurably the right step.** On twelve components at 512 points,
against the number of departures planted:

| planted | mode | passes | directions recovered |
|---|---|---|---|
| 1 | Wiener per direction | 2.0 | 1.0 |
| 1 | flat half-step | 3.7 | 2.7 |
| 2 | Wiener per direction | 2.0 | 2.0 |
| 2 | flat half-step | 4.0 | 5.8 |
| 4 | Wiener per direction | 2.0 | 4.0 |
| 4 | flat half-step | 3.8 | 8.3 |

The flat step leaves half of each direction standing, so the loop finds the
same direction again and reports two to eight directions where one to four
exist. The per-direction weight recovers the planted count exactly. So the
answer to *"is this just a multi dimensional weiner transform differentiated
down into an eliptical shape of n points"* is yes - and using the Wiener
weight rather than a flat step is what makes the count of points come out
right.

### 3c.6 What the differential actually is, and the matrix it forms

Ethan, closing the definition:

> The differentiation is the differential of our synthetic components to
> the actual measured componnet, this has nested differentials in it as
> well according to our graph, so this expands out to a multi dimensional
> matrix that represents our known components.

Shipped as `component_differentials(synthetic, measured, axis, relations)`
with `graph_relations(declared)` reading the relations from
`vhsdecode/pipeline/stages.toml`.

**The diagonal** is each component's own differential, synthetic minus
measured - the specification's ideal against what the tape produced.
`Component.ideal` carries the synthetic side and `Component.expected` the
model as it stands, so both were already in the tree.

**The off-diagonal** is the nested differential. The graph states an
EXPECTED relationship between two components it relates,
`synthetic_i - synthetic_j`; the measurements state the ACTUAL one,
`measured_i - measured_j`; and the entry is the difference of those, which
reduces to `d_i - d_j`. That is precisely the "expected relationship vs. the
actual relationship" the traversal was always specified to take the limit
of. Entries are formed only where the graph relates the pair - parents,
children, and siblings sharing a parent - because a difference between two
unrelated components is not a relationship anyone claimed. Four components
with four graph edges expand to seven entries.

**A defect this exposed.** The matrix is rank-deficient by construction:
"a vs b" is the difference of "a" and "b", so seven entries can span four
dimensions. Charging that deficiency to the deflation makes the shape look
structured when it is only dependent, and the loop refused its own first
pass. The loop now measures the ensemble's intrinsic deficiency once, at
pass zero, and counts it separately from what it removes. The seven-entry
matrix then reaches the floor in three passes, recovering four directions -
its true rank.

### 3c.7 The total eigenvalue is the information budget

> I think the total eigenvalue is the total amount of information we could
> possible derive from our components. I.e. this is multi component radio
> tuning.

Exactly so, and it is the invariant from 3c: every component contributes one
unit to the trace whatever it carries, so **N components hold N units of
derivable information and no more**. What the fit adds is how much of that
budget stands in real directions - the eigenvalue mass above the noise bulk:

| ensemble | components | total | resolved | share |
|---|---|---|---|---|
| six independent | 6 | 6.00 | 0.000 | 0.0% |
| six views of ONE departure | 6 | 6.00 | 6.000 | 100.0% |
| six, shared part plus private | 6 | 6.00 | 2.440 | 40.7% |
| twelve, the same shared part | 12 | 12.00 | 5.166 | 43.0% |

Doubling the components roughly doubles the information resolved while the
share holds, which is the mechanism behind *"you can increase the constant,
how many components"*: more components is literally more budget.

### 3c.8 "Since radio is where we got our vhs signal, that is the limit"

Checked, and it is the binding one. Over the band the luma FM occupies -
Carson's rule on the guide's own figures, 2 x (1.0 MHz deviation + 3.0 MHz
baseband) = 8.0 MHz:

| link | carrier-to-noise | Shannon capacity |
|---|---|---|
| the tape (radio) | 21.9 dB, from the measured 0.669 dB envelope spread | 58.4 Mbit/s |
| the capture card | 43.7 dB, from 8 bits at 50 MSps in that band | 116.0 Mbit/s |

**The capture holds exactly 2.0x the tape's capacity in the signal band**,
so 58 Mbit/s of the converter is spent on nothing and the radio is the wall.
This is the same verdict as the per-bin floors in 3b.2 - a tape floor of
0.0081 dB against an 8-bit floor of 0.00030 dB - restated in the units the
question was asked in, and it says plainly that better capture hardware buys
nothing here. The tape's own carrier-to-noise is the limit.

## 4. The traversal (the extension)

The measurements form a graph, and the graph already exists implicitly in
what is measured today. Each stage produces a multi-dimensional residual;
stages that do not relate to each other are siblings; every stage relates to
its parent and its children.

The algorithm:

1. **Traverse in the direction of the graph, feed-forward.** A child is
   refined under the model its parent currently holds, never against a
   parent that its own residual has already entered — that is fitting, not
   measuring.
2. **Within a node, take the limit in all its dimensions at once** (R2),
   projecting every product orthogonal to its margins (R3).
3. **Between nodes, measure the differential** (R4). Siblings should show a
   diagonal coupling; if they do not, they are not siblings and the graph is
   wrong.
4. **The graph is itself a residual.** The relationships between components
   can be derived, so they form an *expected* relationship. The *actual*
   relationship is what the coupling matrix measures. Take the limit on the
   difference between expected and actual, exactly as for any other
   residual. This is what closes the algorithm: it is a complete traversal
   of every hierarchical relationship between the measurements.
5. **Terminate when the remaining residual is fixed and static.** That is
   the noise floor.

### 4.1 The graph as it stands today

Nodes, as the measurements exist now, with the dimensions each owns:

| node | dimensions | state |
|---|---|---|
| static path response `S(f)` | carrier frequency | known exactly, not measured |
| sweep collapse `C` | sweep rate (theta), carrier frequency | formula + measured residue on theta |
| path response `R(f)` | carrier frequency, head | line + accumulated departure |
| product `C x R` | theta x frequency | measured, not yet applied (needs R3 + a fast spelling) |
| level anchors | line position | measured per field |
| chroma transfer | frequency band, phase | measured per decode |

`R` is a child of `S` (it is what remains once `S` is divided out). `C` and
`R` are siblings under the amplitude measurement: they do not describe each
other, but both describe the amplitude, and their product is the edge
between them. The chroma transfer is a child of `R`: it consumes the
deviation `R` leaves.

### 4.2 What the traversal is, in code

`vhsdecode/residual_limit.py` holds the law: `orthogonal` (R3),
`agreement` (R5), `coupling` (R4), `cell_medians` (the statistic they need
at a price the runtime can pay), and `order`, which reads
`pipeline/stages.toml` and returns the measurement-holding nodes in
feed-forward order with their parents resolved from the declared data
edges. The measurement graph is not a second list to maintain: it is that
declaration, walked in the direction its edges already point.

### 4.2a Every measurement this arc has made, in one table

So a reader after a restart does not have to reconstruct them.

| what | measured | where |
|---|---|---|
| deviation residual before the limit | 1.3-1.4% rms, 50-60x the bins' noise | per head, 75bars SP |
| deviation residual after the limit | 0.15-0.25% rms, 6-10x | same |
| frequency departure D(f) | 0.071-0.113 dB rms, reproduces across pictures r +0.77/+0.86 | four captures |
| sweep residue K(theta) | 0.019-0.044 dB, r +0.77/+0.83 | four captures |
| product P(f,theta) | 0.058-0.088 dB, r +0.25/+0.19 - mostly picture | four captures |
| track residual (amplitude) | 0.221 dB at agreement 0.982 (head B), 0.150 at 0.905 (head A) | 75bars SP |
| track residual (time base) | 0.127% at 0.990 (head B), 0.109% at 0.955 (head A) | 75bars SP |
| amplitude-to-time-base coupling | −0.0026 / +0.0165 (this axis) against −0.0040 / +0.0062 (`carrier_tbc`) | 75bars SP, per head |
| same, chromanoise | −0.0063 against −0.0059 | head B |
| head switch onset | line 259.6 ± 0.8 and 258.9 ± 0.3 of 262/263 | 75bars SP |
| response, common to both heads and speeds | 2.1 dB rms | 75bars SP and EP |
| response, differs between record speeds | 0.35 dB rms - CONFOUNDED, see the guide above | 75bars SP vs EP |
| response, differs between heads at one speed | 0.115 dB (SP), 0.180 dB (EP) | 75bars |
| head identified from the common response | 25/25 and 25/26 fields | bars, chromanoise |
| speed from the response vs from the sync edges | −0.00015/−0.00013/+0.00006 against −0.00013/−0.00014/−0.00007 | three conditions |
| speed signature's share of the amplitude residual | 0.01% | three conditions |
| in-sample vs out-of-sample limit | 0.0001/0.0000/0.0002 against 0.0328/0.0079/0.0563 dB | 75bars head A |
| record-referenced, the limit against the pre-limit build | shape −0.181/+0.151/−0.043%, level +0.090/+0.071/−0.061% | three conditions |

### 4.2a-bis The physical model of the time base

`shared/tbe_physical_model.md` models eleven physical components of time
base error against the JVC guide, the VHS playback physics simulator at
blog.opengbh.net (which is a Mathematica-derived simulator, not prose - its
model is read from the script), and this codebase. Two pieces of arithmetic
in it decide most attributions and are worth carrying here:

- **One line is 368.6 µm of track** (63.5556 µs at a 5.80 m/s head speed),
  so **0.1% of a line is 369 nanometres** of head-to-track displacement. Time
  base error at the precision this arc measures is sub-micron mechanics.
- **The tape is diluted 174 : 1** against the head (5.80 m/s against
  33.35 mm/s), so the format's entire ±0.5% tape-speed tolerance is ±29 ppm
  of line rate.

Its reconciliation with what this arc measured: the 0.1% per-line residual
that only becomes reproducible when indexed by drum angle is **drum
rotational jitter** - back-calculating to 15 µs per field, 0.16° of
rotation - and the same arithmetic rules out the capstan (would need 17%
tape-speed error) and tension (the whole 30-45 g spec range is 7x too
small). The coupling's sign flip between heads is what two heads 180° apart
sampling opposite halves of a once-per-revolution rotor disturbance would
produce. The report's own top discriminating test is a harmonic parity
check - odd drum harmonics antiphase between the heads, even harmonics in
phase - computable from what is already accumulated.

### 4.2b Applied to the model, 2026-09-03

Everything above is measured; these are the parts that reach the model.

- **Components are orthogonalised ACROSS the set, not only within one.** The
  frequency table is accumulated on a residual with the track component
  removed (`track_correction`), and with the amplitude this field's own
  time-base error carries removed at the measured coupling
  (`time_base_coupling`, `field_time_base`). Otherwise the two describe the
  same structure twice: whatever of the track's profile projects onto
  carrier frequency is absorbed by the frequency table and then applied to
  every field as though the path had that response.
- **Neither is removed from the DEVIATION**, and the distinction is the
  attribution. A frequency response is the path's and belongs in the model;
  an amplitude that varies along the track is the tape and the head reading
  it, the colour-under took the same loss, and it must reach the correction.
  The measurement that decides it: the speed signature explains 0.01% of the
  amplitude residual, so the residual is real amplitude, not the model read
  at the wrong frequency.
- **Measured effect on the correction: essentially none.** Record-referenced
  against the same tree without it: shape −0.002 / +0.000 / −0.001 per cent,
  level −0.005 / +0.006 / +0.003, on the three conditions - two orders below
  the limit's own 0.18%. The frequency table was not absorbing meaningful
  track structure, and the geometric reason is that a given track position
  visits the whole picture's frequency range, so the track profile projects
  onto frequency only weakly. The change is right in principle and free, so
  it stays; it is not an improvement and must not be reported as one.
- **The head check runs.** `head_from_response` reads which head the field's
  own response matches and compares it with the parity the decoder assigned:
  25 of 25 and 25 of 26 in the runtime, matching the offline probe. Read
  only - parity remains the key - but a silent parity failure would mix two
  heads into one accumulation, and this notices.

### 4.2c The graph is now walkable

The declaration had a hole: `luma_eq_update` read `per_head_response` and
`luma_transient_observe` read `amplitude_deviation`, and **no node wrote
either** - the amplitude measurement this whole arc refines was absent from
the graph, so the traversal returned three roots and no edge. A
`luma_amplitude` measurement node now declares what it writes, what it
measures, its six components and the eleven stores it accumulates, and
`chroma_env` declares that it reads the deviation (verified at
`chroma.py:577`, not assumed). A test asserts by EQUALITY that no read is
unwritten, so closing an exemption fails as loudly as opening a hole.

`order` also follows `consumes_model` now. A node that consumes another's
MODEL is its child as surely as one that reads its channel, and it will not
say so in `reads`, because the model reaches it as a bundle on the decoder:
the head-switch stage takes its residual from this measurement and declared
it that way. The traversal went from

    high_boost, head_switch, chroma_env - three roots, no edge

to a graph with a root and four children:

    high_boost
    luma_amplitude -> head_switch, chroma_env, luma_eq_update,
                      luma_transient_observe

which is the first real parent-child structure in Ethan's graph. The last
two arrived when the declaration's owner gave them measurement edges onto
the amplitude node: they take `per_head_response` and `amplitude_deviation`
at a POINT rather than as the signal stands when they run, so a measurement
edge is the honest kind, and without one neither had a parent.

Left for the owning lanes, found while doing it: `demod_raw` is read by two
nodes and written by none - it is `demod` under the field record array's
name (`process.py:1753`) and belongs in that node's `writes`;
`luma_eq_update` and `luma_transient_observe` have a producer now but are
still outside the traversal because they declare no `measures`; and the
`site` line numbers are stale by a uniform +36 across every rf-scope node
and about +55 in field scope, while `chroma.py`'s have no single offset at
all.

### 4.2d Dropping the fitted roll-off ABOVE the band: built, measured, reverted

Ethan ruled that the fitted roll-off is a synthetic constraint imposed on
real measurements, that it harms, and that we prefer the measurement. Built
exactly as ruled - a high-end slope mirroring `_low_end_slope`, an ABOVE
branch in `_described_level` continuing from the table's own upper edge, the
line left only where no table exists.

**The bound was measured first and was NOT negligible**, so a null was not
expected: 0.47-0.52% of samples land above the described range against 0.03%
below, reaching a median 40-410 kHz and up to 2.3 MHz past the edge, and the
step between line and measurement at the top edge is the same size as at the
bottom (−1.02 / +0.25 / −0.71 dB against −0.80 / −0.66 / −0.75).

**Record-referenced, control and test built back to back from one tree:
chroma shape +0.132% ± 0.029 on chromanoise - a 4.5 sigma REGRESSION, the
same order as the limit's own effect** - +0.006% on bars EP, neutral on bars
SP, every level figure slightly worse. Split in half, the measured slope
costs +0.081% and the edge-anchored step +0.050%, roughly additive, so
neither half is the culprit.

**The reason is this document's own admission test.** The high-end slope
does not reproduce across pictures. Same head, same tape, same speed, over a
band all three conditions describe: −3.180 / −1.368 / −1.814 dB/MHz at
4.2-4.6 MHz, reversing their ordering one band higher - while the LINE's own
slope agrees to 0.005 (head B) and 0.09 (head A), and the sync-band slope to
0.02 and 0.30. The top-edge step flips sign across pictures where the
bottom-edge step does not. Banked on alternating fields of ONE picture the
high slope looks perfect (−3.285 against −3.314), which is precisely the
within-picture agreement R5 warns is not evidence.

The mechanism: above peak white the table is populated by overshoot and
sweeping carrier, so two pictures do not even describe the same
frequencies - the top edge is 5.25 MHz on bars and 4.85 on chromanoise. Up
there the "measurement" IS the picture, and the fitted line is the only
quantity that reproduces from one picture to the next.

**So the ruling's premise requires that a measurement exist to prefer.** It
holds INSIDE the described range, where `_described_level` already overwrote
the line and has done all along. It holds at the LOW edge, where the case
was made with the −0.13 to −0.83 dB step. It does not extend above the band
on this evidence. Reverted; the negative is recorded here rather than in the
docstrings of the reverted code, because a docstring dies with the code it
documents.

One caveat on the measurement itself: the EP arm was decoded without its
speed flag, so both arms were treated identically and the A/B is sound, but
the EP absolute figures are not comparable with the battery elsewhere in
this document.

### 4.2e Two defects in this arc's own track measurement, and what they cost

Found by the harmonic-parity instrument, which was commissioned to test the
drum attribution and instead audited the axis it was standing on.

**Defect 1: a quarter of every field landed in one bin.** `field.linelocs`
stops short of the field's data - 273 entries ending near sample 761,000 of
983,040 - and `searchsorted` gives every sample past it the same index.
Measured: **26.8% of samples in one place**, worth +0.56 dB on one head and
−0.83 on the other. Fixed by bounding the line index; the amplitude track
residual moves 0.222/0.150 -> **0.180/0.128 dB**.

**Defect 2: the time-base track residual was the head switch, not the
track.** Both heads touch the tape across the switch, so those lines belong
to neither. Excluding the switch's own located region takes it from
0.127/0.109 per cent to **0.0105 per cent** - an order of magnitude - with
single bins inside the window carrying 1.49 and 2.01 per cent alone, and its
alternating-field agreement falling from 0.990 to 0.488.

**And that withdraws the sibling edge as reported.** The coupling between the
amplitude and time-base components on this axis was recorded as −0.0026 and
−0.0063 against `carrier_tbc`'s independent −0.0040 and −0.0059, agreeing in
sign on all three cases. With the switch region excluded it reads **+0.0005
against −0.0040**: the agreement was carried by the switch window. What the
two instruments were agreeing about was an artefact both could see, not a
shared physical coupling. The window is excluded now and the honest statement
is that this axis shows no coupling worth the name at this evidence level.

### 4.2f The drum attribution, tested and contradicted

The physical model attributed the whole track-axis residual to drum
rotational jitter. The parity test says otherwise for the amplitude.

**Amplitude: contradicted.** 85-92% of the low-harmonic energy is odd, which
looks like a rotor fundamental until the phases are read: the odd harmonics
share a common phase to ±10 degrees across four harmonics and all four
captures, with n|c_n| flat. That is the signature of a **step**, not a
rotation. The step is the static per-head level split, −0.51 to −1.50 dB at
3-14 sigma - the model's own head-geometry component. Remove it and the odd
share collapses to **3.9 / 9.0 / 4.3 per cent**, every odd harmonic falling
to 1.5-3.3 sigma. One shared function of drum angle never beats two per-head
ones on held-out fields.

**Time base: undecided, and one to two orders smaller than assumed.** A
once-per-revolution component exists on one condition at 3.8 sigma and reads
0.6 and 0.2 sigma on the others; the spectrum is flat to eight per
revolution. At face value it is 0.26 microseconds per revolution =
**0.0028 degrees of drum**, against the 0.16 degrees the physical model
back-calculated - fifty-eight times smaller.

**The test as commissioned was not falsifiable, which is worth more than the
verdict.** Two half-revolution profiles determine the full-revolution
function exactly, so "sum = even harmonics, difference = odd" is a change of
basis rather than a prediction, and one decode cannot fail it. What made the
amplitude answer decisive was reading the PHASES and the harmonic envelope,
not the parity. And on this deck drum angle and field line number are the
same coordinate - a fixed anchor differs from the field start only by a
cyclic rotation, and measures better than the per-field switch anchor - so
nothing in this arc's data separates the two.

### 4.2g Ethan's asymmetry, which is what the data actually shows

Ethan: *"Very small timebase errors can occur from asymmetry in the rotation
of the head drum, which may be the drum off axis, or the video heads shifted
slightly out of balance on their z axis."*

That resolves the apparent contradiction in 4.2f, because it names a
different thing from the one that was tested. Rotational JITTER varies in
time; an ASYMMETRY is static and, on a rotating drum carrying two heads,
appears in the track-angle coordinate as a fixed pattern - a step between
the heads. The parity test looked for jitter and found a step, and the step
is exactly what an asymmetry produces. Both of his mechanisms are present in
the measurements, and both come out at the sub-micron scale this arc keeps
arriving at:

**Heads out of balance on z — the static per-head step.** Measured at −0.51
to −1.50 dB, 3 to 14 sigma. Through Wallace's law, 54.6 dB per wavelength of
magnetic spacing, and a 1.45 µm wavelength at 4 MHz on a 5.80 m/s head, that
level split is **13.5 to 39.8 nanometres** of spacing difference between the
two heads. Tens of nanometres, on heads whose track is 369 µm per line.

**Drum off axis — the once-per-revolution component.** Measured at 0.0028
degrees of rotation, which on a 62 mm drum is **1.5 µm of arc** and
**259 nanoseconds per revolution**. Small, marginal in the data (3.8 sigma
on one condition, 0.6 and 0.2 on the others), and exactly the size his
phrase "very small" predicts.

So the model in `shared/tbe_physical_model.md` was right about the drum
being where to look and wrong about which drum property: not jitter in the
rotation, but asymmetry in it. The two are distinguished by whether the
component varies in time or is fixed to the head - which is a measurement
this arc can already make, since it carries a per-head axis.

### 4.2h The head differential, split as Ethan predicted

Ethan: *"The asymmetry can be reduced down using a residual between the
differential of their two measurements. Same process that we have used. They
should not drift and be completely static, since the distance from the
center of the head is fixed, but the speed on the tape and or the head will
be varying in relation to the differential of the frequency response of the
tape, which will have varying frequencies, not static frequencies."*

Two falsifiable halves, measured on consecutive head pairs - each pair is
one drum revolution - over 26 and 27 pairs:

| | 75 bars SP | chromanoise SP |
|---|---|---|
| static differential, shape | 0.0936 dB rms | 0.0949 dB rms |
| static differential, level | −0.5558 dB | −0.3380 dB |
| varying part, per pair | 0.0841 dB rms | 0.1350 dB rms |
| **first half vs second half** | **r = +0.902** | **r = +0.958** |
| level, first half → second | −0.5471 → −0.5645 | −0.3299 → −0.3456 |
| varying part explained by a frequency shift | 0.40% | 0.66% |

**The first prediction holds, and holds well.** The static differential does
not drift: split the decode in half and the two halves' shapes correlate at
+0.902 and +0.958, while the level moves by under 0.02 dB across the whole
decode. That is what a fixed distance from the centre of the head predicts,
and it is the strongest confirmation in this arc that the per-head
decomposition is modelling a real object rather than a repeatable artefact.

**The second is not confirmed at this evidence level, and the reason is
already known.** Only 0.4 to 0.7 per cent of the varying part carries the
frequency-shift signature a speed error would leave. But the varying part is
0.084 to 0.135 dB rms, which is the size of the per-bin measurement noise
independently estimated at about 0.096 dB on thirteen fields. So the varying
part is consistent with being noise, and a speed term smaller than that
cannot be seen here. This is the same evidence limit that governs everything
else on this axis: the answer is more fields, not another axis.

So the asymmetry IS reducible, exactly as the design says - it is static, it
does not drift, and `_head_model` already removes it as the shared response
plus each head's departure. What remains after it is at the noise floor.

### 4.3 What is implemented

- R1 and R2, in `vhsdecode/luma_amplitude.py`: the response is measured as
  the deviation's own residual under the model in force, one step per field,
  the decode's field sequence carrying the iteration; the sweep dimension is
  carried by `sweep_residue_of` on the collapse's own theta axis.
- R4, offline, in `pencil/couple.py`.
- R3 in the runtime: the product component is accumulated, projected
  orthogonal to both margins, and admitted by R5's test.
- R4 offline in `pencil/couple.py`, and as `residual_limit.coupling`.
- R5 in the runtime, as the two-bank agreement above.
- **The TRACK axis** (`accumulate_track_residual`), origin at the head
  switch rather than the field start, because each head writes a new track
  and the field's data overhangs into the next. The switch is not declared
  by the format, so it is measured once, by `head_switch.locate`, which
  finds it from this stage's own deviation - the two therefore cannot
  disagree about where it is. Measured on 75bars SP: **0.221 dB rms at
  agreement 0.982** (head B) and 0.150 dB at 0.905 (head A), larger than
  the frequency departure's own 0.1 dB.

  **The origin is the LAST sustained excursion in the field, not the
  first, and getting that wrong invalidated a first round of these
  numbers.** `head_switch.locate` identifies excursions and returns them
  in positional order; its first is not the switch and does not sit
  still - over 25 fields it wanders from line 4 to line 261, a standard
  deviation of 107-115 lines out of 262. The LAST is the switch, because
  the field's data overhangs its own end: measured that way it stands at
  line 259.6 +- 0.8 on one head and 258.9 +- 0.3 on the other, under a
  line of scatter, agreeing with the head-switch arc's own reading of the
  onset at 259-260. (The guide's 6.014 H is the RECORD figure; the playback
  switching point is set by an adjustable monostable, and both heads touch
  the tape across an 8.75-line overlap, so an onset two to three lines later
  sits inside the legal window rather than disagreeing with the format.) An
  axis anchored on the first region is not a track
  axis at all - it is a different axis every field, and it smeared these
  residuals by a third while destroying the time-base component's
  agreement entirely (0.000, where the corrected axis reads 0.955).
  divisions of the track read 0.43 dB at trust 0.977, and the excess is the
  line-locked profile beating against bins that each cover a different
  phase mix - the same trap `head_switch.locate` records burying a 130 mNp
  switch under a 50-70 mNp wander. Measurement only so far; nothing reaches
  the model, and the decode is byte-identical.
- **The TIME BASE's own residual on that same axis** — the measured line
  period against the SPEC period, which is Ethan's reference: a perfectly
  flat and constant time base. Deliberately not `carrier_tbc`'s
  median-relative version, because dividing by the field's own median
  removes the constant speed error by construction and the constant speed
  error is what a spec reference keeps. Both read the same
  `sync_edge_trace`, so neither can drift about where the edges are.
  Measured: 0.080% of a line on head B, 0.054% on head A (75bars SP).
- **THE FIRST SIBLING EDGE, confirmed by two methods that share no
  arithmetic.** The coupling between the amplitude residual and the time
  base, measured on the track axis, against the per-field drum-band
  regression `carrier_tbc.time_base_state` already fits:

  | | this axis | `carrier_tbc` |
  |---|---|---|
  | head B, 75bars | beta −0.0036 | −0.0040 |
  | head B, chromanoise | beta −0.0056 | −0.0059 |
  | head A, 75bars | beta +0.0165 | +0.0062 |

  **WITHDRAWN - see 4.2e.** With the head-switch window excluded from the
  track axis, the coupling reads +0.0005 against −0.0040. The agreement in
  this table was carried by the switch region, which both instruments could
  see and neither owned.

  **All three agree in sign**, and head B agrees closely in magnitude on
  both conditions. The sign structure is the substance: `carrier_tbc`'s own
  docstring records that beta's sign flips between the heads and calls that
  the head-contact structure of the wobble, and an instrument sharing no
  arithmetic with it reproduces the flip.

  A first round of this table reported head A at −0.0011 - the WRONG SIGN -
  and explained it as head A's time base not being a stable quantity, its
  agreement reading 0.000. Both were artifacts of the wandering origin
  above; on the corrected axis head A's time base agrees at 0.955. The
  lesson outlasts the numbers: a component measured on an axis that moves
  reports low agreement, and low agreement is easy to misread as the
  component being unreal.

  The r² values differ by construction and this is not a discrepancy: the
  regression is taken in the drum band alone, where the correlation lives,
  while the track axis carries the whole track profile of which most is
  uncorrelated with the time base. Same slope, different denominator.
- The traversal ORDER over the declared graph, as `residual_limit.order`.
  What is not yet built is the executor that walks it: today only the luma
  amplitude node takes its own limit, and the other measurement nodes the
  declaration lists - `high_boost` and `head_switch`, both measuring the
  envelope - do not yet present their residuals through this interface.
  That is the next step, and it is why the law lives in a module of its own
  rather than inside `luma_amplitude.py`.

### 5. Measured effect of the parts that are in

| | before the limit | after |
|---|---|---|
| deviation residual, per head | 1.3-1.4% rms, 50-60x the bins' noise | 0.15-0.25% rms, 6-10x |
| peak to peak | 8-10% | 2.5-4.2% |

Record-referenced, both required metrics, three conditions, against the
pre-limit build: shape −0.188 / +0.149 / −0.040 per cent, level +0.061 /
+0.064 / −0.060. Better on bars, slightly worse on chromanoise, everything
under a fifth of a per cent — the correction's own accuracy is a wash; what
the limit fixes is the response measurement that every other consumer
reads.
