# Aligning two radio-frequency captures of one tape pass

Ethan's proposal, verbatim (2026-09-06):

> Additionally, I can do the same with the hifi tracks where I can use the
> residual carriers in the video track to synchronize a separate hifi rf
> capture with the video rf capture.

And what the pair is eventually for:

> if we detect hifi, that needs to be matched the same way we are for all
> the other components and subtracted out. Eventually I have another
> project that I will be combining into this process that decodes the hifi.
> In a later session we will merge these two projects to use both RF
> captures as a complex pair video and hifi.

This document holds three things: the route he named, restated closed so
that nobody re-opens it by accident; the methods that do work, each with a
predicted precision and its assumptions; and the one capture that would
prove the best of them.

**There is no audio radio-frequency capture on this machine.** Nothing here
is a demonstration of two captures being aligned. Everything about the
audio side is a prediction conditional on it behaving as the video side
measurably does, and is labelled as one.

The code is `vhsdecode/models/capture_alignment.py` with
`tests/unit/test_capture_alignment.py`. It is an INSTRUMENT, not a node:
its product needs a capture that does not exist here, and its video half
would duplicate the `head_switch_pair` node rather than add a stage.

---

## 0. The answer, in short

**The head switch is the method.** Not because dropouts fail — they do not,
and §4 measures them working — but because the two answer *different
questions*, and only one of them is available often enough to carry the
alignment.

| | method 1: the drum | method 2: the tape |
| --- | --- | --- |
| what it measures | the **sum** of the capture skew and the head lead, *s + L* | the tape-position lead *L* **alone** |
| marks per second | **59.94** — one per field | 0.65 to 2.65 usable dropouts |
| status | video side **measured**; the deck's topology **verified from the schematic**; the audio side predicted | **measured**, video head against video head |
| the drum's own wow | costs it **nothing** — it folds to Nyquist and becomes a per-head constant | not applicable |
| per mark | **365 ns** measured on the deck's own tape (instrument floor 100–250 ns); 3.6–8.2 µs on the two consumer tapes, where the switch lands on vertical sync | **1.2–1.7 µs** (7–10 µm of tape), floor 27–64 ns |
| over 30 s | **12.2 ns**, or **8.6 ns** with the switch captured as a logic edge | ≈0.3–0.6 µs in 10 s |
| the locator | a **carrier phase step** in the AFM RF (§3.6), or better, a **5 V logic edge** at `CN261` pin 3 and solder pad JL345 | an envelope edge, whose steepest part is 260–820 ns |

They are complementary, not competing. Method 1 determines the alignment
*modulo one drum revolution*, and SMPTE 32M table 5 permits the audio head
to sit anywhere in exactly one drum revolution (§2.2), so the head switch
exhausts precisely what it can determine and no more. Method 2 resolves what
is left, because a defect is at a *place* and not at a time. Together they
give two equations in the two unknowns, **and the closure is its own test**:
with a shared converter clock *s* must come out small and constant, and *L*
must land inside the window the standard permits (§4.5).

**Three things make method 1 stronger than it looks.** Everything the two
captures share with zero lag — the drum's mean rate, the servo's placement
of the switch against the picture, the tape speed — cancels *exactly* in the
difference. The drum's own once-per-revolution wow sits exactly at the
Nyquist of once-a-field sampling and folds into a constant difference
between the two heads rather than into scatter. And what is left cancels by
a computable factor `|2 sin(π f · lead)|`, which is **zero when the lead is a
whole drum revolution and 2 when it is half of one** — a property of the deck
that must be measured and reported, never assumed.

**And on this deck the largest improvement costs nothing.** Both head
switches are logic outputs of one microcontroller — `RF SWP` at IC160 pin 18
and `AF SWP` at pin 19, adjacent pins, one drum tachometer, one 16 MHz
crystal (§3.4, read from the schematic) — and both are brought out to
accessible points. Their offset can be read on **two oscilloscope channels
with the deck merely in playback**: no tape of interest, no synchronised
converters, no radio frequency (§3.5). That is the first thing to do.

**The route Ethan named is closed**, at a bound weaker than previously
written and on fewer records than previously claimed: −40 dB relative to the
luma band on channel 1, −35 dB on channel 2 (§1.3). The single thing that
could still re-open it is worth **+27 dB** and needs no new capture:
time-base correct the record before the transform (§1.4).

**And the alignment is not a convenience.** It is what lets video-against-
audio become the sixth axis of the fold, which widens the observed frequency
span from a ratio of 1.29 to 3.38 — **4.73 times as many decades** for
separating the exponential losses that this arc has measured to be collinear
over the video band alone (§6). **10 ns is the figure to aim at**, because
that is where the contrast's phase error drops below a quarter of a radian
and the 1.16–1.66 rad rms of excess phase
(`algorithmic_information.minimum_phase_is_free`) becomes separable across
the axis (§6.1).

---

## 1. The route Ethan named, and why it is closed

### 1.1 The correction that came first

The original write-up (`docs/HIFI_CAPTURE_PATH.md`, 2026-09-05) reported no
audio carrier at the video tap over **five records**, with a bound of −45 to
−50 dB relative to the luma band. Ethan corrected the subjects on
2026-09-06:

> These sample may not have hifi audio. The zaroff samples should have hifi
> carriers, home and cd will not.

`countdown` and `home` have **no Hi-Fi audio on them at all**. Their rows
searched for a signal that was never recorded, and four of the ten
(record, channel) pairs in that table are those two records. Their −50 dB
and −45 dB figures are **withdrawn**: an absent signal found absent is not
a bound on leakage. They are kept visible with their reason in
`capture_alignment.WITHDRAWN` and in a correction block at the head of
`hifi_carriers.py`, because this arc's rule is that a withdrawn claim stays
visible rather than being edited away.

The valid subjects are the zaroff test-pattern captures. That tape was
recorded on a Sony SLV-778HF — the HF suffix denotes a Hi-Fi machine — and a
Hi-Fi deck lays its carriers down whenever it records, because frequency
modulation of silence is the carrier at its centre. The capture readme
records only a composite video connection (*"TSG composite video out → Sony
VCR L1 composite video in"*), so silence at the audio input is the likely
case, not absence.

**The record-tap control is unaffected and is the strongest part of the
original measurement.** The audio carriers go to separate heads on the drum
(SMPTE 32M clause 5.1), so they can never appear at the video
record-current tap. That is what makes the comparison a proper negative
control rather than a coincidence, and it is what disposes of the apparent
detection: the unplanted hump statistic is 4 to 8.6 on the zaroff records,
which looks like a signal, and the record tap shows the *same* excess in
the *same* bins with the playback *lower*.

### 1.2 The re-measurement, widened from one programme to thirty-two

The original sampled **one** programme (`ntc7composite`) of the thirty-two
zaroff playback captures. Correcting the subjects meant re-measuring, and
the wider sweep — all 32 playback captures and all 32 record-current
captures, 128 (file, channel) pairs — is in `capture_alignment.WIDE_SWEEP`.
The published SP row was reproduced to four figures first (−0.5327 /
+0.1208 line, +7.5867 / +4.1088 hump) and the run refused otherwise. It
found three things.

**The narrow-line statistic is not safe.** It divides a peak by the scatter
of its own neighbourhood, so a *flat* neighbourhood manufactures
significance out of an ordinary peak. Ten of 64 playback pairs cross 3σ and
none of 64 record-tap control pairs does — yet the control shows *larger*
bumps, up to 27.1 dB over its own floor, against the firing pairs' 11.2 dB
median. Only its rougher surround stops it firing. Averaging the sixteen
captures of each (tap, speed) group, the AFM-free record tap peaks *higher*
than the playback tap at channel 1 (+15.70 dB against +9.57 dB). **No
candidate survives the control.**

**A named confound is misattributed.** The original names the colour-under's
second harmonic at 1.258741 MHz as the confound 41 kHz below channel 1. The
y-only recordings test it, since they carry no chrominance: removing
18.93 dB of colour-under fundamental removed **0.14 dB** at that frequency,
where a square-law product would have lost about 38 dB. What sits there is
the luma lower-sideband continuum and the tape's own noise. Excluding it by
name buys nothing.

**The operative statistic reverses, and the bound gets weaker.** The
original argued that the zaroff recordings had no audio connected, so the
carrier is unmodulated and the narrow line is the bound to quote. That does
not hold. SMPTE 32M clause 5.6 puts a **2:1 logarithmic compressor** in the
record path, whose gain at silence is maximal, so whatever noise sits on an
open audio input is amplified before it modulates the carrier. Planted at
−45 dB on the mandatory subject:

| planted carrier at −45 dB | line σ rise | detected? |
| --- | --- | --- |
| unmodulated | +3.033 | **yes** |
| 500 Hz deviation | +2.744 | no |
| 1.58 kHz deviation | +1.859 | no |
| reference 400 Hz / 50 kHz (clause 5.9) | +0.123 | no |

500 Hz is what 2:1 log compression makes of an input 84 dB down. The hump
statistic is by contrast nearly indifferent to deviation across the whole
range (rise 2.556 to 2.697), because it integrates ±70 kHz regardless of how
the power is distributed inside it.

### 1.3 The bound, stated plainly

> **On the only records that could carry a Hi-Fi carrier — the zaroff
> test-pattern captures, recorded on a Hi-Fi deck — no audio carrier is
> attributable to the video tap at either 1.3 MHz or 1.7 MHz. The bound is
> the hump bound, which is the conservative one for the compander reason
> above: −40 dB relative to the luma band on channel 1 for the
> chroma-bearing SP record, −45 dB for the y-only SP record, and −35 dB on
> channel 2 at both speeds. Channel 2 is 10 dB weaker than channel 1
> throughout. The route Ethan named — synchronising by residual carriers in
> the video track — is closed at that bound, and the repository's tightest
> supported figure moves from the withdrawn −50 dB to −45 dB.**

### 1.4 What would lower it — `capture_alignment.bound_lowering`

Stated as arithmetic so the question is settled rather than left ajar.

| what | worth | condition |
| --- | --- | --- |
| **Time-base correct the record before the transform** | **+27.0 dB** | none — it costs no new capture |
| A longer capture, *after* the time base is corrected | +18.0 dB for 0.48 s → 30 s | exactly nothing before it |
| A chrominance-free recording | ~0 dB at the floor; +5 dB on the planted hump bound | measured, and less than expected — see 1.2 |
| More converter bits | 6 dB a bit | **only if** the floor at the carrier is the converter's and not the tape's, which is not settled here |
| A capture at `CN341` pin 3 | — | not a way of lowering this bound at all; it is the signal itself |

The first line is the whole of it and deserves its reasoning. The search is
a transform over the whole record, and a carrier is a line in it only if its
frequency holds still. It does not: the tape's time-base error drags every
recorded frequency by ε·f, so a 1.3 MHz carrier with the measured drum-rate
wow of about 4 parts in ten thousand is smeared over 1040 Hz while the
transform's bin is 2.08 Hz — a factor of 499, or 27.0 dB of the signal
thrown away.

If the carrier reaches the video tap **magnetically** — the video head
reading the buried layer of the track it is on — then it is dragged by the
*same* time base as the video, which this arc already measures. Correcting
the record's time base concentrates the whole carrier into one bin. If the
coupling were **electrical** instead, the carrier would carry the audio
head's time base and the same correction would smear it further. **So the
treatment is a discriminator as well as a gain**, and it is the one
experiment left that could still change the answer.

---

## 2. What is actually shared between the two captures

The audio and video signals are never on one wire. SMPTE 32M clause 5.1:

> The FM audio signal shall be recorded into the depth of the magnetic
> layer of the video tape. … **Separate FM audio heads shall be mounted on
> the scanning drum for this purpose.**

So an audio capture is a *second* capture, and it needs aligning. Three
things are shared, in the order they are worth using: the **drum**, the
**tape**, and — if the two converters run from one clock — the **capture
clock**.

### 2.1 Two alignments, and confusing them is the first trap

| | question | needed for |
| --- | --- | --- |
| **temporal** | which audio sample was taken at the same **instant**? | a shared time base; the complex pair |
| **tape-position** | which audio sample read the same **point of tape**? | lip-sync; attributing a defect to a place |

They differ by a mechanical constant, because the two head assemblies sit at
different angles on one drum and touch different tape at any one instant.
Clause 5.3 fixes its sign: *"The FM audio signal shall be recorded prior to
the video signal which is recorded at the same location on the tape."*

### 2.2 The central result: the standard refuses to fix the constant

SMPTE 32M table 5, "Recording time sequence":

| mode | audio recorded earlier than the associated video by |
| --- | --- |
| SP | 0 to 2 fields |
| LP | 1/3 to 2‑1/3 fields |
| EP | 1‑1/3 to 3‑1/3 fields |

**Every one of those windows is two field periods wide, and two field
periods is exactly one revolution of the drum**, because the drum carries
two video heads and turns once per frame. The permitted range is the whole
circle: 360°. The standard therefore determines *nothing* about the offset.

That is not a defeat. A mechanical angle between two head assemblies bolted
to one drum does not change with time, temperature or tape, so **one
measurement serves for every capture that deck ever makes at that speed**.
What it means is that the measurement cannot be skipped, and that a deck
with separate SP and EP video heads needs it once per speed.

That last point is itself derivable. The three windows differ only in their
centres — 1, 1‑1/3 and 2‑1/3 fields — and the audio heads do not move
between speeds, so the difference can only come from the *video* heads being
elsewhere. At 180° of drum rotation per field, the LP and EP pairs both come
out **60° from the SP pair modulo 180°**, which is one extra head pair
mounted at 60° seen from its two ends. (`head_pair_placement`; a derivation,
labelled one. It assumes table 5's windows are centred on the actual
placements rather than being three independently drawn bands — which their
equal widths support and the standard nowhere states.)

---

## 3. Method 1 — the drum, seen twice

Both captures are of one rotating drum. The head-switch event in each is the
same rotation seen twice, so fitting a straight line through (event number,
event time) in each capture and differencing the two phases gives the
temporal alignment. `switch_phase` and `align_by_switches`.

### 3.1 Why the difference beats either capture alone

Three properties make this the strong method, and all three are structural
rather than lucky.

**The drum's own wow costs nothing.** A head-switch sequence is one
measurement per field, so it samples the transport at the field rate — and
the drum turns at *exactly half* the field rate. The drum's
once-per-revolution disturbance therefore sits exactly at the Nyquist
frequency of that sampling and folds into a **constant difference between
the two heads**, not into scatter. A fit that keeps the two heads apart
absorbs it, and it cancels again between two captures that sample the same
drum the same way. (`field_sampling_alias`.)

**Everything shared with zero lag cancels exactly.** The drum's mean rate,
the servo's placement of the switch against the picture (clause 3.6 permits
5 to 8 lines ahead of vertical sync — a 191 µs window, so it too is a deck
constant and not a format one), the tape speed: all common mode.

**What is left cancels partially, by a computable factor.** The audio and
video switches happen at different drum angles and therefore at instants
`lead` apart, so the difference samples any residual disturbance twice with
that delay. For a disturbance at frequency *f* the surviving factor is
`|2 sin(π f · lead)|` — zero when the lead is a whole number of the
disturbance's periods, and **2 when it is half of one**. Since the lead is a
property of the deck and cannot be chosen, this is a thing to measure and
report, not to assume. (`jitter_survival`.)

### 3.2 The precision, and how it is derived

`offset_precision` has two terms and only they:

```
σ_offset  =  sqrt( (σ_video² + σ_audio²) / N )      the locators
           ⊕  |2 sin(π f · lead)| · A               the transport
```

where σ are the per-event locator scatters, N the number of events, and A
the timing amplitude of a residual disturbance. A fractional speed error
ε₀ at frequency *f* displaces the time base by ε₀/(2π f) — the time base is
the integral of the speed error — so a slow disturbance costs far more
displacement than a fast one of the same fractional size
(`timing_from_speed_error`).

The **first term falls as 1/√N and the second does not**, so past the
crossing point more events buy nothing and only a shorter lead or a quieter
transport does. `events_for_precision` inverts the first term.

For a **drift** between two independent clocks, fit the per-event offset
against time; the slope is the fractional clock difference, with

```
σ_rate  =  σ · sqrt(12 / N) / T ,   N = f_field · T   ⟹   σ_rate ∝ T^(−3/2)
```

That steep exponent is the argument for a *long* capture rather than a
careful one: doubling the length buys 2.83×, where doubling the per-event
precision buys 2. If the two converters share a clock this is a **test**
rather than a measurement — a slope consistent with zero is the evidence
that the sharing held.

#### The per-event scatter, measured on the video side

The switch is located as the maximum-likelihood change point of the log
envelope, after the channel response is divided out (in 64 quantile bins of
the demodulated frequency, separately either side of the switch) and after
the **field-median profile on a vertical-sync-locked grid** is subtracted, so
that everything repeating field after field cancels and what is left
alternates with the head.

| capture | fields | switch, lines ahead of vsync | **residual rms** |
| --- | --- | --- | --- |
| `zaroff-75bars` SP (the deck's own tape) | 28 | **+6.334 H** | **365 ns** |
| `home` | 363 | −1.082 H | 3642 ns |
| `countdown` | 336 | +0.00 H | 8150 ns |

The instrument's own floor, calibrated by planting a gain step of known size
in the raw RF away from any real switch and running the same estimator on
it, is **100–250 ns** on a strong edge — so 365 ns is within a factor of 1.5
to 3 of the instrument, and is conservative, since a planted pure step lacks
the real switch's own ~1 µs transient.

**The ten- and twenty-fold difference is not the tapes' fault, it is where
the switch lands.** SMPTE 32M 3.6 permits 5 to 8 lines ahead of vertical
sync, and on the deck's own tape the switch sits six lines clear. On both
consumer tapes it lands *on or just after the vertical sync leading edge*,
where the broad pulses dominate the envelope and there is nothing to see a
step against. A paired capture would be made on the bench with the deck's
own tape, so **365 ns is the relevant figure**.

#### The premise, confirmed

On `zaroff` the switch minus the recorded vertical sync scatters **159.5 ns
rms** — the drum and the tape move together to within the instrument's floor.
On `home` the **switch is steadier in absolute time than the vertical sync
is** (3642 ns against 5202 ns). That is the expected physics and it is the
method's whole premise: the playback drum is servo-locked to the deck's own
reference while the tape flutters, so a **drum-referenced alignment is better
founded than a tape-referenced one**.

#### The transport term has no measured value, because none was resolved

The switch residual is **white to the measurement's noise floor**: running
means over 3, 9, 27 and 81 fields sit at 0.93 to 1.27 times what a white
series would give, and only 61 % of the power is below 10 Hz where white
would give 33 %. Upper bounds on real drum wander: **below 435 ns** on
`zaroff` over 0.48 s, 1–2 µs on `home` over 0.15–1.35 s — and those are at
the noise. So the second term of the formula above carries no measured value,
the scatter averages down as 1/√N with no floor in sight, and **a single
measured drum-phase offset stays valid across a capture**.

The tape is a different story — `home`'s vertical sync scatters 5202 ns with
excursions to 15 152 ns and 99.2 % of its power below 10 Hz, peaking at
1.9–4.8 Hz; `countdown`'s at 0.17–1.3 Hz. Which is exactly why the alignment
must be expressed as *drum phase* and not against anything tape-locked.

Pooling was **measured rather than assumed**, by taking the scatter of
estimates from disjoint groups of consecutive fields: `countdown` falls
8150 → 5016 → 3305 → 1834 → 643 → 315 ns from 1 to 64 fields, *beating* the
1/√N projection, because the single-field figure is dominated by outright
estimator failures that averaging suppresses. **Half a second to one second
of video RF gives sub-microsecond drum phase on every capture tested,
including the worst.**

#### What that predicts for the pair

With σ_video = 365 ns measured and σ_audio **predicted** equal to it (there
is no audio capture — §3.7):

| capture length | events | audio locator like the video one | audio locator a logic edge (§3.5) | rate bound |
| --- | --- | --- | --- | --- |
| 0.48 s | 28 | 98 ns | 69 ns | 0.69 ppm |
| 1 s | 59 | 67 ns | 48 ns | 0.23 ppm |
| **30 s** | **1798** | **12.2 ns** | **8.6 ns** | **0.0014 ppm** |

**Thirty seconds of the deck's own tape reaches the 10 ns aim of §6.1** — at
12.2 ns the contrast's phase error at peak white is 0.34 rad, a sixth of an
output sample at 4 f_sc, and 0.18 % of the drum-rate time-base displacement.

#### One marker was tried and rejected

The drum's **line in the RF envelope** — the obvious first idea, and it does
not work. It is found easily enough (1429 to 7044 times the local background
at 0.48 s, 84 044 to 383 691 times at 30 s) and its analytic phase error is
41–92 µs at 0.48 s and 6–13 µs at 30 s. But its **split-half reproducibility
is 60–300 µs**, four to twenty times worse, because a small error in the
fitted frequency becomes phase drift. The reason is structural: the line is
mostly the head-A-against-head-B *level difference* — 3.2 to 11.3 % of the
mean — which is a two-state **square wave**, so fitting its fundamental as a
sinusoid is systematically compromised. Recorded so it is not proposed again.

#### And the record mark is better located and is the wrong marker

The colour-under phase reversal (`head_switch_pair`) locates *better* than
the playback switch — **296 ns** against 365 on `zaroff` with none of 28
fields rejected against six, and three times better on `countdown` (2602 ns
against 8150). It is still the wrong marker for method 1, because it is
**frozen on the tape by the recording machine**: it reaches an audio capture
of the same pass only *through the tape*, not through the shared drum. It
belongs to method 2's family.

Two traps in it, recorded because either one swamps the result: the mark
carries an exact **half-line alternation** (0.4991 H peak-to-peak, fitted),
which must be modelled or the residual is 15 832 ns instead of 296; and
`countdown` needs a **four-field** term rather than a two-field one, because
the recorder's mark moves with the colour frame (19 517 → 2602 ns).

**And one estimator failure mode worth carrying.** A first version that
searched for the largest step anywhere in the sixteen lines before vertical
sync **locked onto the vertical interval itself** on both consumer tapes,
returning a switch-minus-vsync residual of exactly zero — a perfect-looking
answer that was measuring nothing. Subtracting the field-median profile on a
vsync-locked grid is what removes it, and it is load-bearing rather than a
refinement.

### 3.3 The ambiguity, stated exactly

A run of switch events cannot say which cycle it is in, so the offset comes
back wrapped into ± half a cycle. The cycle is:

* **one drum revolution (33.367 ms)** when the two heads of a pair can be
  told apart — in the video capture by field parity and by the step in the
  blanking RF level, in an audio capture by the two heads' opposed azimuths
  (clause 5.4, ±30° 30′);
* **one field (16.683 ms)** when they cannot.

And the permitted head placement is *exactly one drum revolution wide*. So
the head-switch method exhausts precisely what it can determine, and the
residual ambiguity is exactly the whole permitted range. Two captures
started by one piece of software are trivially within half a revolution of
each other, so in practice the alignment is complete — but the ambiguity is
real and something must resolve it if the start offset is unknown. That
something is method 2.

### 3.4 The deck's switching topology — read from the schematic, 2026-09-07

Method 1's central assumption is that the video and audio head switches are
two views of *one* rotation rather than two independent events. On this deck
that is no longer an assumption. The Sony SLV-777HF/778HF/788HF schematic
(board MA-327, rendered at 700–4200 dpi with the vector geometry traced
segment by segment, so a net's terminal *count* is known rather than its
path merely followed by eye) says:

| | as printed |
| --- | --- |
| controller | **IC160 `M37777M7A235GP-C`**, "SERVO/SYSTEM CONTROL", sheet 3/8 |
| video switch | **pin 18 `RF SWP`**, 5 Vpp 30 Hz square (waveform ⑰) → `CN261` pin 3 (JL270) and IC260 pin 17 via R270 22 k. Three terminals, no other branch |
| audio switch | **pin 19 `AF SWP`**, waveform ⑱ *"(HiFi STEREO MODEL)"*, **sharing one photograph with ⑰** → IC340 pin 1 `HSW` (direct, no series part) and IC360 pin 40 `AF SW P̄`. Three terminals |
| tachometer | **one**. `CN101` from M901 DRUM MOTOR: pin 3 `D PG` → R177 → IC160 pin 86 (30 Hz), pin 4 `D FG` → R178 → IC160 pin 87 (360 Hz) |
| does any drum signal reach the AFM section? | **no** — sheet 5/8's complete cross-sheet inventory holds no PG, no FG, no tachometer line |
| controller time base | X160 **16 MHz** (pins 38/39), X161 **32.768 kHz** (pins 41/42), both confirmed running |

Direction was established rather than assumed: the inter-sheet terminal
`RS7` carries *opposite* chevrons on the sheets that send and receive it —
leaving 3/8 at IC160, entering both 1/8 and 5/8 — and the sheet-3/8 net has
exactly three terminals with a junction dot. `AF ENV` runs the other way,
into IC160 pin 9, which is the master-and-slave pattern this implies.

**The AFM path really does switch heads.** IC340 (LA7256) carries two
identical amplifier chains on nodes printed `Ch1` and `Ch2` meeting at a
changeover switch, and the control arrow into that switch traces inside the
outline to pin 1 `HSW`. So an AFM capture carries a once-a-field head switch,
and it is driven by **`AF SWP`, not `RF SWP`** — the two are separate
controller outputs, so the audio switch is not the video one reused.

**So both switches are timed by one controller, from one tachometer, against
one crystal.** That their offset is therefore a firmware constant is an
*inference from the topology* and is labelled one; what is verified is that
nothing else in the drawing could set it.

Two cautions carried rather than assumed away. IC360 pin 40 is printed
`AF SW P̄` **with an overbar** while IC340 pin 1 is plain `HSW`, so the AFM
processor appears to expect the inverse sense and a capture must fix its own
polarity convention by observation. And which internal `LOGIC` port of IC260
pin 17 the video switch lands on **could not be read**, even at 4200 dpi; it
is reported as not legible rather than guessed.

### 3.5 A measurement that needs no tape pass and no radio frequency

Both switches are brought out as logic signals: `RF SWP` at **`CN261` pin 3**
(a connector pin, ground on pin 4, on the same low-impedance node as IC260's
feed — R270's 22 k is in the *branch*, not in series with the connector), and
`AF SWP` at the **solder pad JL345** (sheet 1/8; also JL375 on sheet 5/8).
There is no `AF SWP` pin on `CN341`.

So the switch offset can be read **directly on two oscilloscope channels with
the deck merely in playback** — no tape of interest, no synchronised
converters, no RF at all. `switch_offset_on_the_bench`.

It gives the *switch* offset, not the capture skew (no capture is involved)
and not the tape-position lead. Its real value is twofold: a locator on a
five-volt logic edge instead of on an amplitude discontinuity in a noisy
carrier, which is the **single largest improvement available to method 1**;
and an independent check that a paired capture must reproduce — a
disagreement is the evidence that one capture's envelope switch locator is
biased, which is exactly the failure that would otherwise go unnoticed.

### 3.6 The audio switch is a **phase step**, and the guide says so

One deck's schematic is one deck. The format's own manufacturer guide makes
it general — JVC Video Technical Guide VTG82063 §7.3.2:

> The playback signals from the rotary audio heads are sent through the
> rotary transformers, then preamplified approximately 60 dB and supplied to
> the channel switcher. **This uses the drum flipflop signal** to produce
> continuous signals.

and, on what the switch looks like in the signal:

> As indicated in Fig. 7-3-3, **the instantaneous phase of the carrier
> changes at the head switching point.** This results in a frequency
> deviation change that appears as pulse type noise in the demodulated
> signal.

> In the case of the video signal, head switching is performed outside the
> picture region. However, **since blanking is absent from the audio
> signal**, processing is required in order to remove the noise.

Three consequences, and they all favour the method.

**The audio switch is drum-derived on every deck built to the guide**, not
merely on the one whose schematic was read (§3.4).

**The marker is a phase step, not an amplitude edge — so it is *better* than
the video side's, not worse.** The video switch is found from a step in a
noisy *envelope*, and an envelope is a magnitude: it has thrown its phase
away. Two heads reading the same track cannot hand over in phase — nothing
constrains their relative gap positions to a fraction of a 4.5 µm
wavelength — so the carrier's phase jumps at the changeover, and that is a
discontinuity in the **analytic signal**. Its scale is the carrier cycle:
769 ns at channel 1, **588 ns at channel 2**, divided by the amplitude
signal-to-noise ratio in the usual way, and floored at the capture's own
sample period (25 ns at 40 MSps). At 1.3 MHz sampled at 40 MSps there are 31
samples per cycle to work with. `afm_switch_marker` — which **refuses to
quote a figure** without a measured signal-to-noise ratio, because there is
no audio capture here to measure one from.

**It cannot be hidden.** Decks conceal the switch in the *demodulated* audio
by holding the previous sample, but a capture at `CN341` pin 3 is taken at
the head amplifier's output, ahead of that processing. There is no interval
in an audio track for the switch to hide in — which is exactly the property
a timing marker wants, and the opposite of the video case, where the switch
is deliberately placed in blanking.

**Both normative sources decline to fix the offset.** The guide states the
audio head's azimuth (±30°, against SMPTE's ±30′ 30″), its depth, its
wavelengths and its switching source — and says *nothing* about its angular
position relative to the video heads. So §2.2's conclusion is about the
format, not about one document.

### 3.7 What remains assumed

* that the audio-side switch locator's scatter is comparable to the video
  side's. **Unverified — no audio capture exists.** Capturing `AF SWP` as a
  logic signal would make this moot for the offset, though not for a
  capture that only has the AFM RF.
* **that the two audio heads can be told apart in the capture** — which sets
  whether the ambiguity of §3.3 is one drum revolution or only one field.
  The obvious handle is level, and it may not survive: VTG82063 §7.3.2 puts
  an **FM AGC** in the playback chain *"to compensate for level fluctuations
  due to variations in head to tape contact"*, and IC340 (LA7256) carries
  its own AGC, so a tap at `CN341` pin 3 may already be levelled. If head
  identity is not recoverable the modulus halves to one field (16.68 ms) and
  the ambiguity **doubles**. `align_by_switches` takes the cycle as an
  argument and guesses nothing; capturing `AF SWP`, whose two half-cycles
  are the two heads by construction, settles it outright.

---

## 4. Method 2 — the tape, crossed twice

Both heads cross the same physical defects, so a dropout is a mark at a
fixed **tape position** common to both. This method is **measured** here,
video head against video head, and it works.

### 4.1 The geometry that makes it a discriminator

A defect fixed to the *tape* does not recur one field later at the field
period: the tape has crept while the drum turned, so the next head must
travel further along its scan to reach the same oxide.

```
L₁ = T_field · ( 1 + v_tape / (v_writing · cos θ) )
sin θ = pitch / (v_tape · T_field)
```

The pitch scales with the tape speed — the heads sweep the same path
whatever the tape does — so **θ is a constant of the drum**, 5.9836° derived,
against table 2's 5° 58′ 9.9″ = 5.9694°; the 0.014° difference is the
nominal 58 µm pitch's rounding. A defect fixed to the *deck* or to the
*picture* recurs at exactly `T_field` instead.

| mode | tape-locked lag − field period | in lines |
| --- | --- | --- |
| SP | **96.45 µs** | 1.518 |
| LP | 48.21 µs | 0.759 |
| EP | 32.16 µs | 0.506 |

`T_field` must be **measured per capture**, not taken from the format: the
discriminator is 96 µs on a 16.68 ms period, so a 6-parts-per-thousand error
in the period would swallow it.

### 4.2 What was measured

On the raw RF of four captures, with the shipped detector's own state
machine reproduced on a 2–6 MHz band-pass envelope and the field mean
replaced by a running median over one field period.

**The coincidence is overwhelming and it is at the right lag.**

| capture | coincidences at L₁ | uniform-random null (500 draws) | z | recurrence |
| --- | --- | --- | --- | --- |
| `home` | 159 | 0.20 ± 0.46 (max 3) | **345** | 35.96 ± 2.93 % |
| `countdown` | 39 | 0.02 ± 0.13 (max 1) | **311** | 33.64 ± 5.53 % |

At the **field period** instead — the deck-locked control — the count is
zero on every capture (−1.3 to −0.2 σ). The fine histogram is a single clean
peak at +92 to +100 µs with **nothing at 0 µs**. The EP capture is the
independent confirmation: its predicted offset is three times smaller
because the tape crawls three times slower, and the peak lands at +32.5 µs
against a prediction of 32.16. **The tape-speed ratio is written into the
answer.**

**One shared mark locates the tape to 1.2–1.7 µs (7–10 µm of tape).** By
cross-correlating the two heads' full-rate envelope traces of the same
defect: median correlation 0.88–0.94, robust σ 1.70 µs (`countdown`) and
1.73 µs (`home`), falling to 0.71 µs for short events. The Cramér–Rao bound
on the envelope is 27–64 ns, forty to sixty times below what is achieved, so
**the 1.2–1.7 µs is physical** — a different cut through the defect, plus the
transport's own time base — and not measurement noise. There is a per-head
systematic of −0.876 µs (`countdown`) and −0.913 µs (`home`) which is common
to both head orders and agrees between two unrelated tapes to 0.04 µs, so it
is a fixed deck constant and calibratable.

**Yield 0.65/s (`countdown`) and 2.65/s (`home`).** Pooling gives ≈1.5 µs/√N,
so ten seconds of capture reaches **0.3–0.6 µs**.

### 4.3 The trap, which the discriminator removes for free

On `home`, **88.2 %** of what the shipped detector calls a dropout is a
once-per-field event at a fixed field phase — spread 9.7 µs, median 5.17 µs
long — which recurs at exactly `T_field` and is not a tape defect at all.
(`ep75` 17.7 %, `countdown` 5.1 %, `sp75` 0 %.) A method that took those for
tape marks would be aligning to the deck. The `T_field` versus
`T_field + 96.5 µs` test removes them automatically.

### 4.4 The handicap, and the counter-effect

A dropout is head-to-tape separation, and separation costs `54.6 d/λ` dB
(Wallace). The two channels record at very different wavelengths — 4.4615 and
3.4118 µm for the audio carriers against 1.7059 and 1.3182 µm for the
luminance carrier — so **the same lump of dirt is 2.0 to 3.4 times weaker in
the audio channel** (2.6× for channel 1 against the sync tip; 2.92× against
a 3.8 MHz mid-band carrier). A defect costing the video 20 dB costs audio
channel 1 only 7.6 dB.

That is not an inconvenience to apologise for: **it is the mechanism that
makes Hi-Fi audio survive dropouts that wipe the picture**, and it is a hard
limit on this method — a defect must clear the audio capture's own noise
after being divided by that ratio.

Against it, the AFM track is 10–29 µm wide (table 5) against the video's
58 µm, so the same defect covers a larger fraction of it. **The net effect
is not measured** and cannot be on this material.

The measured defect shape says what there is to work with: recurrence at one
track 30–36 %, at two tracks 2.7–5.4 %, at three and beyond consistent with
zero. These are **point defects on the scale of the track pitch, not creases
spanning many tracks**, elongated along the direction of head travel
(60–119 µm median along-track, 273–448 µm at the 90th percentile). Recurrence
rises steeply with severity — 67–75 % for events over 100 µs against 13–18 %
for those under 10 µs — so **selecting on depth or length turns a one-in-three
mark into a two-in-three one**.

### 4.5 Two unknowns, two equations

The head-switch comparison measures the **sum**: an audio switch seen early
is equally well explained by a converter started late (the skew *s*) or by a
head mounted ahead (the tape-position lead *L*). A defect measured in both
captures measures *L* alone, because a defect is at a place and not at a
time.

```
switch offset  =  s + L
dropout lead   =      L
        ⟹  s  =  switch offset − dropout lead
```

**And the test comes free with the arithmetic.** If the two converters share
a clock, *s* is a small constant and nothing more: an *s* that comes out
large, or that changes between captures, says the sharing failed. And *L*
must land inside the window table 5 permits, which is an independent check
that costs nothing. A lead outside it means the coincidence was matched to
the wrong defect or the wrong drum revolution.
(`separate_skew_from_lead`.)

---

## 5. Method 3 — the linear control track

Being investigated by a parallel lane (`models/edge_tracks.py`,
`docs/EDGE_TRACKS.md`) for whether the *video* head can see it. Two things
from this side are worth recording.

SMPTE 32M clause 3.4 **does** specify the linear head's position — *"The
distance X on the tape pattern, from the end of the 180° scan of a video
head to the audio and control track head position, shall be 79.244 mm"* —
which is exactly the geometry the standard declines to fix for the rotary FM
audio head. The silence in table 5 is therefore deliberate, not an omission.

But an audio RF capture at `CN341` pin 3 sees the **AFM head amplifier**, not
the linear head, so the control track is not on it. This route can only ever
align a *video* capture to a *linear-audio* capture, which is a different
pair from the one Ethan wants. It is not pursued here.

---

## 6. What the pair is for: the sixth axis of the fold

Ethan intends to merge a Hi-Fi decoder and treat the two captures as a
complex pair. In this arc's framework that is a literal statement, not a
metaphor. `tesseract` folds a cube on binary measurement axes — head,
polarity, tape, half, channel — and **which tap a measurement came from is
exactly such an axis**. Adding it takes the cube from 32 vertices to 64 and
makes the contrast between the two taps a measurement in its own right.

| | |
| --- | --- |
| **common** (the fold's parent) | the tape and its coating, the defects in it, the drum, the drum's wow, the capstan, tape tension, the transport's whole time base |
| **different** (the fold's child) | the head and its gap, the azimuth (±30° 30′ against ±6°, so 36.5° apart in SP), the depth read, the band, the playback amplifier |

**Why that is worth having, in numbers.** This arc has measured that its
magnetic mechanisms collapse — they are all monotone decays over a narrow
band and so nearly collinear — and that *only fractional bandwidth moves
that*. The video band alone spans a frequency ratio of **1.29** from sync tip
to peak white. With the audio carriers the observed span runs 1.3 to
4.4 MHz, a ratio of **3.38** — **4.73 times as many decades** for telling one
exponential decay from another. And the depth sampling goes from two
clusters to four: 0.210 and 0.272 µm for the luminance, 0.543 and 0.710 for
the audio, 1.466 for the colour under. Two clusters cannot split three
losses; four can be asked to.

The axis's stage assignment is a **labelled assumption**, as
`tesseract.AXIS_STAGE` requires: it is proposed to first matter at the
head-to-tape interface, because that is where clause 5.1 puts the
separation. It is *not* proposed at the capture, because by the capture the
two have been different signals for the whole chain.

### 6.1 What each precision would let the fold resolve

A fold may only pair two measurements of the same moment. A residual
misalignment of σ enters the contrast as a phase error `2π f σ` at every
frequency, and **any component whose own phase is smaller than that is
unrecoverable**. That is the whole translation from a timing figure to a
physical one. (`resolution_from_alignment`.)

| alignment σ | phase at 4.4 MHz | output samples at 4 f_sc | fraction of a line | fraction of the drum-rate time-base displacement |
| --- | --- | --- | --- | --- |
| 1 µs | 27.6 rad | 14.3 | 1.6 % | 15 % |
| **100 ns** | 2.76 rad | 1.43 | 0.16 % | 1.5 % |
| **10 ns** | 0.276 rad | 0.14 | 0.016 % | 0.15 % |
| 1 ns | 0.028 rad | 0.014 | 0.0016 % | 0.015 % |

Reading the table: at **1 µs** the pair can be folded per *field* and per
*line*, and used as a time-base probe to about 15 % of the error it probes —
enough to confirm the video-derived time base but not to improve it. At
**100 ns** the contrast's phase is still ±2.8 rad at the top of the luma
band, so amplitude contrasts are usable and phase ones are not. At **10 ns**
the phase error drops to a quarter of a radian and the excess-phase
term — the 1.16 to 1.66 rad rms that
`algorithmic_information.minimum_phase_is_free` measures as the largest
unexplained component, 65 to 84 per cent of the phase — becomes separable
across the axis. **10 ns is
therefore the figure the design should aim at**, and it is what makes the
sixth axis worth adding rather than merely possible.

---

## 7. If the carriers were ever found, they are a component

Ethan: *"if we detect hifi, that needs to be matched the same way we are for
all the other components and subtracted out."* The door is built before the
detection rather than after it, and it is the same door every other
component uses. No bespoke path.

1. **Signature** — `capture_alignment.afm_carrier_signature`, which delegates
   to `interference.beat`, because a Hi-Fi carrier at the video tap is
   exactly what that entry describes: a line in the spectrum at a known
   frequency with a known origin. It already returns the subtractable form
   `1 + a·shape`.
2. **Position** — `capture_alignment.COMPONENT_ORDER` declares *"afm carrier
   crosstalk"* at **23**, composed into the chain by
   `interference.full_chain`. It is *added at the head's output*, alongside
   the medium's own particle noise, and shares that position because two
   additive terms at one node commute. It is **not** in the capture band at
   30: that would be electrical coupling between the two amplifiers, and the
   record-tap control bounds that path, since the audio record drive is
   present during a record-tap capture and left no excess.
3. **Admission** — `interference.admission` then the held-out judge in
   `holdout`, which decides on out-of-sample evidence and not on whether the
   residual fell.
4. **Subtraction** — only if admitted, and at the gain the correction-gain
   law gives rather than at the full fitted amount.

**On present evidence the entry is refused at step 3 for want of a
detection.** The width is not guessed either: it defaults to the
reference-modulated reach of clause 5.9, the only width the standard states.
A *silent* carrier's linewidth is a property of the **transport** — ε·f, some
520 Hz at 1.3 MHz — and `silent_carrier_width_hz` computes it, with the
compander caveat that it is a floor and not a value.

---

## 8. The capture that would prove it

**Two simultaneous radio-frequency captures of one tape pass, on a shared
sample clock.**

| | |
| --- | --- |
| channel A | `CN261` pin 2 `PB RF` — the video playback tap, where every existing capture in this repository was taken |
| channel B | `CN341` pin 3 `FM PB` — the AFM playback tap, the audio equivalent of pin 2 on its own test connector |
| clock | one clock to both converters. The project's own answer is two CX cards on an external clock (the clockgen mod) |
| duration | **30 s**, because the rate bound falls as T^(−3/2) — 200× better than the half-second captures here — and because it is long enough to contain dropouts large enough to clear the audio channel's reduced sensitivity |
| if further channels are free | the two head-switch square waves themselves — `CN261` pin 3 `RF SWP` (ground on pin 4) and `AF SWP` at solder pad **JL345**. They turn the switch locator from an inference off a noisy RF envelope into a five-volt logic edge |

**Do this first, because it is free:** read `RF SWP` against `AF SWP` on two
oscilloscope channels with the deck merely in playback. That measures the
switch offset directly, needs no capture at all, and is the check the paired
capture must reproduce (§3.5).

**What it settles:** the deck's audio-to-video head offset, once and for
ever; the capture skew, and whether the shared clock held; whether a dropout
is common to the audio and video heads; the AFM RF level against the luma's,
which is the bridge the crosstalk envelope has always been missing; and
whether the sixth axis of the fold can be filled at all.

**What it does not settle:** whether the audio carriers leak into the video
track. That is already answered — section 1 — and this capture would only
make the answer's reference level measurable.

---

## 9. What is refused, and why

`capture_alignment.refusals` carries these in code, each with what would
lift it.

| refused | because | lifted by |
| --- | --- | --- |
| the alignment itself, end to end | no audio RF capture exists on this machine | the paired capture of section 8 |
| the audio head's tape-position lead *L* for this deck | table 5 permits a whole drum revolution, and measuring it needs both captures | the same pair, through `separate_skew_from_lead` |
| the locator noise of an *audio* head-switch event | no audio capture. The **video** side is measured — 365 ns an event on the deck's own tape (§3.2) — and the audio side is predicted equal to it, which is what every paired figure rests on. **No longer refused:** whether there is an audio head switch to locate at all — the schematic shows IC340's two head amplifiers meeting at a changeover switch driven by `AF SWP` (§3.4), and the guide says it appears as a carrier phase step (§3.6) | one audio capture of any length over three fields; or, for the *offset* rather than the locator noise, an oscilloscope on `CN261` pin 3 against JL345, which needs no capture (§3.5) |
| whether a dropout is common to the *audio* and video heads | it is measured between the two *video* heads, which is necessary and not sufficient — the audio case adds a different azimuth, depth and band | the paired capture, looking for the same defect at the lead the head switch predicts |
| the residual skew between two clock-synchronised cards | no two-card capture exists here | one signal split to both cards and cross-correlated |
| whether the carrier bound is converter- or tape-limited | the "6 dB a bit" line of section 1.4 holds only under that condition | `capture_chain_noise` at the carrier frequencies, at two bit depths |
