# The vertical interval, mapped onto the elliptical collapse

`docs/ELLIPTICAL_COLLAPSE.md` fits an ellipse to `synthetic − measured`
differentials. Everything the method can say depends on where the synthetic
side comes from, and everywhere else in this arc it comes from a curve
fitted to the same data it is then judged on — which is what the out-of-
sample rule (R5) exists to contain.

**The vertical interval is the one place where that problem does not
arise.** A vertical interval test signal is specified: its bar amplitude,
its pulse half-amplitude durations, its multiburst frequencies, its
staircase levels are all printed in a standard with tolerances. A vertical
interval *data* signal is stronger still: its run-in and framing code are a
fixed bit pattern, so the transmitted waveform is known exactly, sample for
sample. Neither is fitted. The differential is against the standard.

This document maps both populations onto the method, measures how many
independent dimensions each actually provides, and says plainly which
survive a tape recording.

Three companion documents carry material this one depends on and does not
repeat. `docs/SPECIFICATION_INVENTORY.md` §3 holds the element-by-element
numbers for every insertion test signal and names their sources;
`docs/BT1700_COLLAPSE.md` covers the composite signal specification;
`docs/BROADCAST_VTR_COLLAPSE.md` covers U-matic. Where a number here is
also there, it is cited rather than restated.


## 0. The four findings, first

1. **An NTC-7 composite line provides about 4.6 independent dimensions
   against the 10 characteristics ITU-T J.63 Table IV names for it.** It is
   the *worst* of the insertion test signals measured, not the best — its
   elements are several views of the same low-frequency luminance channel.
2. **The multiburst is the efficient signal.** The line carrying it — the
   NTC-7 *Combination*, J.63 Annex II §3 — resolves 6.58 of 22 readings,
   and a bare multiburst line resolves 6.17 of 16. The reason is the same
   one that makes echoes separable and tape losses not: the packets are
   parameterised along the axis that makes them different.
3. **Of the VBI data signals, EIA-608, VITC, WSS-625 and CGMS-A survive a
   VHS recording intact; World System Teletext and VPS do not; the ghost-
   cancelling reference survives in part and is by a wide margin the
   strongest synthetic component in existence for this arc** — ATSC A/49
   publishes all 910 of its samples at 4f_sc, which is the decoder's own
   output grid.
4. **`tools/ringing_measure/elliptical_collapse.py`'s `vertical_interval()`
   is measuring active picture.** Its line window runs two rows past the end
   of the interval, and the "perfect needle" recorded for `ntc7composite` is
   the first two lines of a full-field test pattern. Reproduced, and the
   cause established, in §7.


## 1. What the vertical interval contains, and in what order

Three populations, and they are not interchangeable.

| population | what fixes it | where it sits |
|---|---|---|
| the interval's **own geometry** — equalizing pulses, serrations, the vertical sync block | the format's own timing specification | lines 1–9 (525), 1–7½ (625) |
| **insertion test signals** — bar, pulse, staircase, multiburst | ITU-T J.63; national practice | 525: lines 17 and 280; 625: lines 17, 18, 330, 331 |
| **data signals** — teletext, captions, timecode, WSS, GCR | each carrier's own standard | 525: 10–21; 625: 6–23 and 318–335 |

The first is already used by this arc — `vertical_interval_components()` in
`vhsdecode/models/information_extrapolation.py` reads the vertical sync
block as the only sub-16 kHz probe the signal contains, and
`docs/RESIDUAL_LIMIT_DESIGN.md` records the corner it sets. The other two
are what this document adds.

### 1.1 Line assignments, corrected

Three premises in common circulation are wrong and are corrected here
because they change which lines a tool should look at.

**330 and 331 are LINE NUMBERS, not signal designations.** They are the
field-2 counterparts of lines 17 and 18 in the 625-line system: field 2
begins at line 313, and 17 + 313 = 330. Confirmed three ways — ITU-T J.63
Annex I §1 ("*lines 17 (330) and 18 (331)*"), ITU-R BT.472-3 §4 footnote 6
("*lines 16, 17, 18, 19, 20, 21 and 329, 330, 331, 332, 333 and 334 are
reserved*"), and EBU Tech. 3209 §1. The 525-line equivalent is J.63 Annex
II §1: line 17 of both fields, "*lines 17 and 280 if numbered
consecutively*".

**ITU-R BT.1700 contains nothing on test signals or the vertical interval.**
It is *Characteristics of composite video signals for conventional analogue
television systems*: Part A defers NTSC entirely to SMPTE 170M-2004 (which
it reproduces in full as Annex 2), Part B tabulates PAL, Part C SECAM. The
waveforms live in **ITU-T J.63** (the normative element specification),
**ITU-R BT.473** (its ITU-R twin), and **ITU-R BT.1439-1** (element
definitions and measurement methods). BT.1700's only role here is being
cited by BT.1439 §3.4.1.4 as the source of the nominal cut-off frequency
that sets *T*.

**ITU-T J.61 does not define insertion test signals either** — it sets
transmission performance objectives. The element designations are letters,
not numbers: B₁ (2T pulse), B₂ (luminance bar), C₁ (reference bar), C₂
(multiburst), D₁/D₂ (staircase, unmodulated and modulated), E (reference
subcarrier), F (composite 20T/12.5T pulse), G/G₁/G₂ (chrominance bar).

**The FCC defines no "FCC composite" waveform.** 47 CFR §73.682(a)(21)
*permits* test signals on lines 17–20 and bounds their modulation
excursions; §73.699 is a set of engineering charts with Figures 13–15
reserved and Figure 16 identified as VIRS. The waveforms circulating as
"FCC composite" and "FCC multiburst" — including the 0.5 / 1.25 / 2.0 / 3.0
/ 3.58 / **4.1** MHz packet set — have no authoritative source and are
marked **UNVERIFIED** throughout this document. J.63's 525-line multiburst
is 0.5 / 1.0 / 2.0 / 3.0 / 3.58 / **4.2** MHz (Annex II, Table III).

Two FCC provisions do matter and are firm:

| provision | text | line |
|---|---|---|
| §73.682(a)(21)(iv) | after 30 June 1994 line 19 may carry **only** the ghost-cancelling reference of OET Bulletin 68 | 19 |
| §73.682(a)(21)(iv) | VIRS, formerly line 19, may move to **any of lines 10 through 16** | 10–16 |
| §73.682(a)(22) | closed captioning | 21 |
| §73.682(a)(23)(i) | NABTS teletext on **lines 10–18 and 20**, ≤70 IRE on 10–12, ≤80 IRE on 13–18 and 20 | 10–20 |


## 2. The test signals, exactly

`docs/SPECIFICATION_INVENTORY.md` §3.1–3.4 carries the full element tables
from ITU-T J.63 Annex I (625) and Annex II (525), with amplitudes,
positions and tolerances. What follows is the part this document needs and
the part that document does not have: the definition of *T*, the
element→characteristic map, and one correction.

### 2.1 The value of T

ITU-R BT.1439-1 Annex 4 §2 defines it: **T = 1/(2·F_c)**, where F_c is the
nominal bandwidth of the channel under test.

| system | F_c | T | 2T HAD | wide pulse |
|---|---|---|---|---|
| 525-line | 4 MHz | **125 ns** | **250 ± 10 ns** | 12.5T, **1.57 ± 0.05 µs** |
| 625-line, 5 MHz | 5 MHz | **100 ns** | **200 ± 10 ns** | 20T, **2 ± 0.06 µs** |
| 625-line, 6 MHz | 6 MHz | 83 ns | 167 ns | — |
| 625-line, OIRT variant | — | 80 ns | **160 ns** | — |

*Sources: J.63 Annex II §2.2 and Annex I §2.2/§4.2; BT.1439-1 Annex 1 Fig. 5
Note 1; EBU Tech. 3209 §7.2.3(b) tightens the 625 tolerance to ±6 ns.*

625-line countries with 5.5 or 6 MHz systems still use T = 100 ns, and
BT.472-3 §2.6 footnote 2 states why: "*for international routine
measurements, it is suggested that the test signals be based on a single
reference frequency which could be 5 MHz*".

One inference, labelled as such: 525's T = 125 ns implies F_c = 4.0 MHz,
whereas BT.1701-1 Table 1 item 4 gives the M/N nominal main sideband as
4.2 MHz, which would give 119 ns. **The 125 ns figure is a rounded
convention; no document states the reconciliation.**

### 2.2 What each element measures — J.63 Annex II Table IV

This is the authoritative element→characteristic map for 525-line, and it
is what "the quantities conventionally read from the line" means in §4.

| characteristic | element | line |
|---|---|---|
| insertion gain | B₂ | 17 / f1 |
| amplitude/frequency response | B₂ (or C₁) **and** C₂ | 17 / f1 **and** f2 |
| line-time waveform distortion | B₂ | 17 / f1 |
| short-time distortion, step response | B₂ | 17 / f1 |
| short-time distortion, pulse response | B₁ | 17 / f1 |
| chrominance-luminance **gain** inequality | B₂ and F | 17 / f1 |
| chrominance-luminance **delay** inequality | F | 17 / f1 |
| line-time luminance non-linearity | D₁ (D₂ where intermodulation is small) | 17 / f1 |
| chrominance non-linearity | G | 17 / f2 |
| differential gain | D₂ | 17 / f1 |
| differential phase | D₂ | 17 / f1 |
| chrominance-luminance intermodulation | G | 17 / f2 |

**Ten of the twelve are read from line 17 field 1 — the NTC-7 composite
line.** That is the count §4 measures against.

Mapped onto this arc's three axes, the map is more instructive than the
list. A multiburst packet measures |H| at its own frequency; a 2T pulse
measures the top of the luminance band and its group delay together; a
modulated pulse measures |H| and arg H at the subcarrier *relative to* the
low band; a modulated staircase measures how both vary with luminance
level, which is the **amplitude** axis and is nonlinear; the bar measures
level and the low-frequency roll-off within a line. Two elements on
different axes are trivially independent — which is the structural reason a
test line beats the tape's six magnetic mechanisms, all of which live on
the frequency axis and are nearly the same decay there (`ELLIPTICAL_COLLAPSE`
§7.2, 1.58 of 6).

### 2.3 One correction to the machine-readable source

`resources/definitions/vits/ntsc/ntc7-composite.yaml` gives the 12.5T
chrominance component `component_amplitude_ire: 50.0`, so ±50 IRE, 100 IRE
peak-to-peak. That is the construction which makes the pulse's peak equal
the bar's, as J.63 requires ("*within ±0.5 IRE of B₂*"), and it makes the
base line sit at blanking. `docs/SPECIFICATION_INVENTORY.md` §3.3
transcribes it as "±20 IRE (40 p-p)"; the YAML's own 50 is the one used
here. The 50/50 luminance/chrominance split is itself **UNVERIFIED against
a primary document** — J.63 specifies only the 100 IRE total.


## 3. The synthetic side, rendered and checked against itself

`scratchpad/vbi/vits_render.py` renders a line directly from the
machine-readable VITS definition — every amplitude, window and rise time
from the specification, nothing from any measurement — and reads back from
it exactly the quantities the specification's own primitives name. Reading
the rendered line back is the first check any synthetic side must pass.

| reading | J.63 / VITS says | rendered line reads |
|---|---|---|
| blanking level | 0 IRE | **0.0000** |
| sync amplitude | −40 IRE | **−39.998** |
| sync width, 50% points | 4.7 µs | **4.706** |
| burst | 40 ± 1 IRE p-p | **39.71** |
| B₂ white bar | 100 ± 0.5 IRE | **100.0000** |
| B₁ 2T pulse amplitude | within ±0.5 IRE of B₂ | **99.911** |
| B₁ 2T half-amplitude duration | 250 ± 10 ns | **250.1 ns** |
| F 12.5T amplitude | within ±0.5 IRE of B₂ | **99.720** |
| D₂ staircase peak | 90 ± 1 IRE | **90.014** |
| staircase terminus | 90 IRE | **90.0000** |
| G three-level chrominance (Combination) | 20 / 40 / 80 IRE p-p | **19.83 / 40.02 / 80.65** |
| C₂ multiburst frequencies (Combination) | 0.5 / 1.0 / 2.0 / 3.0 / 3.58 / 4.2 MHz | as specified |

**Four readings do not land on the specification, and every one of them is
the reading method's bias rather than the signal's.** They are listed
because a gauge built on this must know where its own floor is:

| reading | J.63 bounds the inherent value at | the ideal line reads | what it is |
|---|---|---|---|
| F 12.5T base line | ≤ 0.5 IRE | **2.76 IRE** | the luminance and chrominance envelope estimators disagree at the pulse's skirts |
| D₂ luminance non-linearity | risers within 0.5% of each other | **0.31 IRE** | the step windows include part of the 2T-shaped risers |
| D₂ differential gain | ≤ 0.25% | **1.55%** | the chrominance envelope is estimated across step edges |
| D₂ differential phase | ≤ 0.2° | **0.553°** | the same |
| C₂ multiburst packet amplitude | ±25 IRE (VITS) | **25.5–27.3 IRE** | a ±350 kHz measurement band on a sine-squared-shaped packet |

Each is a constant offset in the reading, so it cancels in a differential —
which is the only way any of them is used. **The floors above are the
resolution of a reading built this way, and nothing measured through them
below about 3 IRE, 1.5% or half a degree is a finding.**

**Two structural facts about the schema had to be got right, and getting
either wrong destroys the element that matters most.** The primitives carry
a `channel: y` / `channel: c` field and a `combine: replace` / `add` rule,
and they only make sense together: the D₂ staircase is a luminance
primitive that *replaces*, laid over a chrominance packet that was *added*
across the same window. Rendered into one array the staircase erases the
subcarrier and the modulated staircase carries no chrominance at all —
which is the whole element. Rendering onto separate luma and chroma planes
and summing at the end is required, and it is what the `channel` field is
for.


## 4. How many independent dimensions a test line actually provides

### 4.1 The measurement

The same question `ELLIPTICAL_COLLAPSE` §7.2 asks of the tape's magnetics,
and `vhsdecode/models/interference.py` asks of the modelled interference,
asked of a test line. The method is identical in kind to §7.2's — perturb,
measure, compare signatures:

1. render the line exactly as the specification states it;
2. perturb the channel along a basis of departures — 20 amplitude modes and
   20 phase modes over the video band, plus three level-dependent modes
   (differential gain, differential phase, luminance non-linearity);
3. read every quantity the line's own primitives name, and record how each
   moved. That row of the Jacobian is the reading's **signature**;
4. fit the ellipse over the signatures with
   `ellipsoid(..., real_parameters=True)` — real, because each reading is
   the response to a real physical parameter and a gain is a different
   mechanism from a delay of the same magnitude profile.

Two readings that respond identically to every departure are one
measurement wearing two names. The reported figure is the participation
ratio `1/Σ share²` over the singular values, which is the same statistic
that gives 1.58 of 6 and 6.89 of 12 elsewhere in this arc.

### 4.2 The result

Eleven insertion test signals, rendered from the collection's VITS
definitions and measured on the same basis:

| signal | system | readings | **effective** | condition | 90% span | 99% span |
|---|---|---|---|---|---|---|
| FCC composite **(UNVERIFIED)** | NTSC | 25 | **7.15** | 602 | 9 | 15 |
| **NTC-7 Combination** (J.63 II §3, line 280) | NTSC | 22 | **6.58** | 327 | 8 | 13 |
| FCC multiburst **(UNVERIFIED)** | NTSC | 16 | **6.17** | 364 | 8 | 10 |
| ITU multiburst (J.63 I §3, line 18) | PAL | 17 | **5.68** | 1747 | 9 | 11 |
| **NTC-7 Composite** (J.63 II §2, line 17) | NTSC | 25 | **4.64** | 866 | 7 | 13 |
| UK national ITS-1 (line 19) | PAL | 25 | 4.46 | 6178 | 7 | 12 |
| UK ITS-2 (line 20) | PAL | 11 | 4.22 | 60 | 4 | 6 |
| ITU combination (line 331) | PAL | 15 | 3.77 | 1054 | 4 | 7 |
| ITU composite (line 330) | PAL | 22 | 3.54 | 1.8e4 | 5 | 8 |
| VITS 17 (line 17) | PAL | 27 | 3.38 | 3.1e5 | 5 | 8 |
| VIRS | NTSC | 13 | 3.02 | 487 | 4 | 7 |

Every one of these stands 11 to 18 sigma above its own sphere floor, so the
structure the ellipse finds is real and not the finite-ensemble scatter of
`sphere_floor`.

### 4.3 What it says

**The NTC-7 composite line provides about 4.6 independent dimensions
against the 10 characteristics J.63 Table IV names for it.** Seven
directions carry 90% of the span and thirteen carry 99%, out of 25 readings
its own primitives support. The conventional reading is roughly twice as
many quantities as the line can independently determine.

**And the NTC-7 composite is the worst of the composite family, not the
best.** Its efficiency — effective dimensions per reading — is 0.186, the
lowest in the table. The reason is exactly the reason the tape's magnetics
collapse to 1.58 of 6: the bar, the 2T pulse, the 12.5T pulse's luminance
half and the staircase levels are all reading the same low-frequency
luminance channel, from four angles that are not as different as their
names suggest. A gain change moves all four; the four readings are then
four views of one departure.

**The multiburst is the efficient element**, and it is efficient for the
reason `interference.py`'s header states about echoes: it is parameterised
along the axis that makes its parts different. Each packet interrogates
|H| at its own frequency and at no other, so six packets are close to six
independent numbers.

| signal | effective | readings | **per reading** |
|---|---|---|---|
| FCC multiburst | 6.17 | 16 | **0.386** |
| UK ITS-2 | 4.22 | 11 | 0.384 |
| ITU multiburst | 5.68 | 17 | 0.334 |
| NTC-7 Combination | 6.58 | 22 | 0.299 |
| FCC composite | 7.15 | 25 | 0.286 |
| NTC-7 Composite | 4.64 | 25 | **0.186** |
| *(for comparison)* modelled interference | 6.89 | 12 | 0.574 |
| *(for comparison)* tape magnetics | 1.58 | 6 | 0.263 |

**Which single line to prefer.** Among signals backed by a standard, the
**NTC-7 Combination — ITU-T J.63 Annex II §3, line 17 field 2 (line 280)**
— gives the most independent components per line at 6.58, and it is the
line that carries the multiburst. The FCC composite scores higher at 7.15
but is not a standard's waveform (§1.1), so it should not be preferred on
that number.

The practical consequence for a decode: **the two lines are a pair.** J.63
puts the composite on field 1 and the combination on field 2 of the same
line number precisely because neither alone is sufficient — Table IV's
"amplitude/frequency response" is the one characteristic that needs both.


## 5. The VBI data signals

A data signal is a stronger component than a test signal, because a known
bit pattern recovered through the whole chain is an impulse response, not a
handful of readings. Every one of these carries a fixed run-in or framing
sequence identical in every field it appears on.

| signal | line (525 / 625) | rate | modulation | the part that is FIXED |
|---|---|---|---|---|
| **EIA-608 / CTA-608-E** | 21 / 284 | 32·f_H = **503 496.5 bit/s** | NRZ, 2T-bar shaped, 240 ns rise | 7 cycles of a 0.5034965 MHz sine + start bits `0 0 1` |
| **WST teletext (System B)** | — / 6–22, 318–335 | 444·f_H = **6 937 500 bit/s** | NRZ, skew-symmetric about 0.5 R_b | 16-bit `1010…` run-in + framing `11100100` |
| **NABTS (System C)** | 10–18, 20 / — | 364·f_H = **5 727 272 bit/s** | raised cosine 55–100% + 4.2 MHz LPF | 16-bit run-in + framing `11100111` |
| **VITC** | 10–20 / 6–22 | 115·f_H = **1 809 441 bit/s** | NRZ, 200 ± 50 ns rise | nine `1,0` sync pairs at bits 0,10,…,80 |
| **WSS 625** | — / 23, first half | 5 MHz clock, **833.33 bit/s** info | bi-phase-L, 200 ± 10 ns elements | 29-element run-in `0x1F1C71C7` + 24-element start code `0x1E3C1F` |
| **WSS 525** | 22 / 285 | 4f_sc clock, 7 SC per bit | **B6–B23 are a tone AT f_sc**, phase-keyed | B1–B5, B24 reference bits |
| **VPS** | — / 16 | 160·f_H = **2 500 000 bit/s** | bi-phase, 200 ns elements | run-in `0xAAAA` + start code `0x8A99` |
| **CGMS-A** | 20 / 283 | f_sc/8 = **447 443 bit/s** | NRZ, 70 IRE / 0 IRE | a reference pulse of exactly 2 bit periods |
| **GCR (BT.1124 System C)** | **19** / 318 | — | a **linear-FM chirp** | **the entire 910-sample waveform** |

*Sources: CTA-608-E §5.2 Table 2; ETSI EN 300 706 §6.1–6.3 and 7.1; ITU-R
BT.653-3 Tables 1a/1b; EIA/CEA-516; SMPTE ST 12-1-2008 §10.4–10.8; ITU-R
BT.1119-2 Annexes 1 and 2; ETS/EN 300 231; IEC 61880; ATSC A/49 §2 and
Annex A; ITU-R BT.1124-3.*

Four corrections worth carrying, because each changes which line to look at
or what to expect there:

- **The US ghost-cancelling reference is BT.1124 System *C*, not B.**
  System A is Japan (a windowed sinc integral, not a chirp), System B is
  Korea (a ternary sequence). There is no System D or E.
- **It is a linear-FM chirp, not a Bessel chirp** — quadratic spectral
  phase. "Bessel" is patent terminology and appears in neither A/49 nor
  BT.1124.
- **WSS on 525 lines is on lines 22 and 285, not line 20.** Line 20 is
  CGMS-A, a different standard (IEC 61880).
- **WSS-625 occupies only the first half of line 23**, once per *frame*,
  and ETSI TR 101 233 §5.1.2 notes that line 23 "*does not fall within the
  strict definition of the VBI*" — the rest of it is active picture.

### 5.1 What each measures that the others do not

| signal | what it uniquely gives |
|---|---|
| **GCR** | the **whole complex response**, bin by bin, over a 48 µs echo window — the only VBI component that determines H(f) rather than a few functionals of it |
| **EIA-608** | a pure **503.5 kHz tone** of specified amplitude, seven cycles, at a specified position — a single-frequency amplitude and phase reference below every other probe, and the only one that repeats on **every field** |
| **VITC** | nine sync transitions spread evenly across the line, all with a **200 ± 50 ns specified edge** — a comb of step responses at nine known positions, which is a *time-base* witness the others are not |
| **WSS 625** | a bi-phase run-in that is a **square wave at exactly 833.33 kHz** with a minimum run length of three elements — a clean low-frequency square-wave response, and its start code deliberately violates the biphase grouping, giving one uniquely locatable instant |
| **WST / NABTS** | the highest-rate transitions in the interval, so the **top of the luminance band**, where nothing else in the VBI reaches |
| **CGMS-A** | a reference pulse of exactly two bit periods with **no run-in**, i.e. a bare isolated step at a known level |

### 5.2 The GCR is the strongest synthetic component this arc has

ATSC A/49 Annex A publishes **all 910 samples of one line at 4f_sc
(14.31818 MHz)**, normalised to unit peak-to-peak, with sample 0 at the
50% point of the sync leading edge. That is the decoder's own output grid,
sample for sample, with no interpolation between the standard and the TBC.

| property | A/49 states | measured from A/49's own table |
|---|---|---|
| start of the GCR (1% of peak) | 12.0 µs | 12.013 µs |
| duration (1% points) | 35.5 µs | 35.479 µs |
| first peak | 16.7 µs | 16.692 µs |
| chirp dispersion b | 110.0 | 110.11, phase-fit residual **0.0099° RMS** |
| spectrum | "flat to 4.1 MHz" | 0.07 dB ripple, 0.2–3.9 MHz |
| roll-off | "very low beyond 4.3 MHz" | −3 dB at 4.123, −40 dB at 4.401 MHz |
| pedestal | 30 IRE, 9.5–58.5 µs, 4T transitions | — |
| chirp swing | −10 to +70 IRE | — |
| echo window | −3 to +45 µs | — |
| field sequence | A B A B B A B A | + − + − − + − + |

Two errata found in the standards themselves and worth recording, because
either will silently produce a wrong reference:

- **A/49 prints `e^{+j·sign(ω)·b·ω²}` in its closed form.** That sign gives
  a time-reversed down-chirp which does not match the standard's own
  published table: reconstruction with `+j` correlates at r = 0.075, with
  `−j` at **r = 0.993**. Use the minus sign.
- **BT.1124-3 Table 10 prints A and b with positive exponents**; Report
  ITU-R BT.2018 prints them negative, and only negative is physical
  (bΩ² = 391.6 rad matches the normalised value exactly).

The eight-field sequence is `+ − + − − + − +`, and the second group of four
is the negation of the first — which is what cancels four-field-periodic
content, that is, NTSC subcarrier crosstalk. It is eight fields and not
four for exactly that reason.

**A note for the owner of `docs/PROPAGATION_COLLAPSE.md`**, whose §5.8
reasons about this signal and which is not edited here: three numbers there
disagree with the measurement above. The chirp spans 0 to 4.30 MHz, not
4.2; it is 35.5 µs long, not 52, so the time-bandwidth product is
**22.65 dB** rather than 23.4 and the derived multi-frame figures inherit
the offset; and it is not "phase-inverted once per frame" but follows the
eight-field pattern above. One further caveat that document does not
mention: the GCR's raw autocorrelation has a peak-to-sidelobe ratio of only
about 19–21 dB, so plain matched filtering cannot resolve a −40 dB echo —
an inverse filter or sidelobe weighting is required.


## 6. What survives a tape recording

Two constraints, and both must be checked.

### 6.1 The band

A colour-under format band-limits its luminance and carries chrominance
apart. The figures used here are this project's standing ones: VHS luma to
about **3 MHz** (the same figure `interference.sound_trap` relies on when it
notes the 4.5 MHz sound trap is unreachable through a colour-under
recording), chrominance about ±400 kHz about the subcarrier; U-matic low
band about **3.5 MHz** and ±500 kHz.

**The test signals, measured through that chain.** The ideal line rendered
from the specification and read back through a modelled colour-under
recording:

| reading | specified | through VHS | change |
|---|---|---|---|
| multiburst packet, 0.5 MHz | 100% | 100% | — |
| multiburst packet, 1.0 MHz | 100% | 100% | — |
| multiburst packet, 2.0 MHz | 100% | 100% | — |
| multiburst packet, **3.0 MHz** | 100% | **54.7%** | **−45.3%** |
| multiburst packet, **3.58 MHz** | 100% | 100% | **routed through the chroma path** |
| multiburst packet, **4.2 MHz** | 100% | **3.5%** | **−96.5%** |
| 2T pulse half-amplitude duration | 250 ns | 255 ns | +1.8% |
| bar amplitude, staircase amplitude, sync | — | within 0.2% | — |

Three things follow, and the third is the one that catches people out.

- **A VHS luma channel measures the multiburst up to 2 MHz cleanly, halves
  the 3.0 MHz packet and destroys the 4.2 MHz one.** Three of six packets
  survive intact; U-matic keeps four.
- **The 2T pulse still responds after the recording, but it is no longer
  measuring the source.** Its half-amplitude duration keeps most of its
  sensitivity because most of that sensitivity is to the *low* end of the
  band, which the recorder passes — so the reading moves, and what it now
  reports is the recorder's own filter. A 2T reading off a tape decode is a
  measurement of the tape, and it is only a measurement of the recording
  when differenced against a direct feed.
- **The 3.58 MHz multiburst packet is not a luminance measurement on a
  colour-under recording at all.** It lands inside the chrominance
  passband and is recorded through the chroma path. It comes back, and it
  comes back measuring a different channel. Anyone reading a flat 3.58 MHz
  packet off a VHS decode as evidence of luminance bandwidth is reading the
  chroma channel.

A caution on the model: the band edges above are brick walls, and real
record-side luma/chroma splitting filters roll off. The 3.0 MHz packet's
45% loss and the exact routing of the 3.58 MHz one are therefore
qualitative; that they happen is not.

**The data signals, by occupied bandwidth:**

| signal | occupied bandwidth | fits under 3 MHz? |
|---|---|---|
| CGMS-A 525 | 447 kHz | **yes, easily** |
| EIA-608 | **99.997%** of energy below 3 MHz; 94% below 503.5 kHz | **yes** |
| WSS 625 | first null 2/T = **1.667 MHz** (BT.1119's own figure) | **yes** |
| VITC | 200 ns edge, first null 1.81 MHz | **yes, ~1 MHz to spare** |
| AMOL I / II | 1 and 2 Mbit/s (patent-sourced) | yes |
| NABTS / WST-525 | Nyquist **2.864 MHz** | **marginal** — only the roll-off skirt is clipped |
| GCR | **72.6%** of energy below 3 MHz, −1.39 dB | **partial, and usable** |
| VPS | first null 2/T = **5.0 MHz**, main lobe peak 1.85 MHz | **no** — the top 40% of the main lobe goes |
| WST-625 | Nyquist **3.469 MHz** | **no, structurally** |
| WSS 525 data bits | B6–B23 are a tone **at f_sc** | not through the luma path — through the **chroma** path |

**World System Teletext cannot be recovered from a VHS luma channel, and
the reason is structural rather than a matter of degree.** Its Nyquist
frequency is 3.469 MHz, *above* the cut-off, so a 3 MHz channel cuts inside
the Nyquist band. No equalizer restores that: the information is not
attenuated, it is gone. The 1976 specification demands at least 5.0 MHz of
video bandwidth. The 525-line systems, whose Nyquist is 2.864 MHz, sit just
inside and survive.

The GCR's partial survival is worth stating in its own terms. Losing the
top of the band costs delay resolution, not the measurement: the echo
resolution goes from 0.233 µs to 0.333 µs, and **382 real numbers** of the
channel response remain determined against 550 direct — the count of
frequency bins the reference still excites within 20 dB of its peak, at the
15.734 kHz spacing one line gives.

### 6.2 The sync-only constraint

The standing rule (`vhsdecode/addons/RINGING_RULES.md`, and the memory
`sync-only-constraint`) is that a correction derives from the sync pulse and
the reserved intervals, never from active picture; active-area gauges are
offline validation only.

| signal | line | inside the reserved interval? |
|---|---|---|
| insertion test signals, 525 | 17, 280 | **yes** |
| insertion test signals, 625 | 17, 18, 330, 331 | **yes** |
| EIA-608 | 21, 284 | **yes** |
| VITC | 10–20 / 6–22 | **yes** |
| CGMS-A | 20, 283 | **yes** |
| GCR | 19, 282 / 318 | **yes** |
| VPS | 16 | **yes** |
| WST / NABTS | 6–22 / 318–335 | **yes** |
| **WSS 625** | 23, first half | **partly** — TR 101 233 §5.1.2 states line 23 "does not fall within the strict definition of the VBI"; the second half is picture |
| **WSS 525** | 22, 285 | **yes**, but line 22 is the first line many decoders treat as active |
| **AMOL** | 20 *or* 22 field 2 — contradictory | **unresolved**; if line 22, it is active picture |

Everything except the last three is unambiguously inside the boundary and
may be used for correction. WSS and AMOL need the line checked against the
decoder's own active-area start before either is used, and a gauge reading
WSS-625 must window the first half of the line only.

### 6.3 The plain answer

| | survives VHS | survives U-matic | needs a direct baseband feed |
|---|---|---|---|
| **test signals** | bar, staircase, 2T pulse *(as a tape measurement)*, 12.5T/20T, multiburst to 2 MHz | the same plus the 3.0 MHz packet | the 4.2 MHz multiburst packet; the 2T pulse *as a measurement of the source*; the 625 multiburst's 4.0 / 4.8 / 5.8 MHz packets |
| **data signals** | EIA-608, VITC, WSS-625, CGMS-A, NABTS/WST-525 (marginal), GCR (72.6% of its energy) | the same, with more of the GCR | WST-625, VPS |

The 625-line multiburst deserves a line of its own: its packets are 0.5,
1.0, 2.0, **4.0, 4.8 and 5.8 MHz** (J.63 Annex I Table I). Only the first
three are below a VHS luma cut-off. **A PAL VHS decode recovers half of a
625 multiburst and none of its top half**, which is a sharper loss than the
525 case and worth knowing before anyone builds a gauge on it.


## 7. `vertical_interval()` in `elliptical_collapse.py`

The function is right about the two things that matter most and wrong about
where to look. Both are established here by measurement, not by reading.

### 7.1 What it gets right

Its docstring states the case for the vertical interval correctly and for
the correct two reasons: the synthetic side is known rather than fitted, so
R5's problem does not arise; and the interval is not active picture, so
using it does not cross the sync-only constraint. It reads the TBC and its
JSON rather than assuming a geometry, it scales to IRE from the decode's own
`black16bIre`/`white16bIre`, it pools every field, and it reports a blank
interval as a finding rather than a failure. All of that is sound and should
be kept.

### 7.2 What it misses

**The line window reaches into active picture, and that is where its result
comes from.** The window is `first, last = 9, min(22, height - 1)`, so rows
9 through 21 inclusive. Measured on this decoder's field layout:

| TBC row | 75-bars decode | `ntc7composite` decode | what it is |
|---|---|---|---|
| 0–7 | 44–58 IRE span, mean to −27 | 44–45 IRE span, mean to −27 | vertical sync block and equalizing pulses |
| 8 | 24–28 | 24 | the transition out of it |
| **9–19** | **8.9 IRE span** | **3.8–4.4 IRE span** | **blanking — the interval proper** |
| **20, 21** | **101, 100 IRE** | **101, 99 IRE** | **active picture** |
| 60, 120, 180, 240, 260 | ~100 | ~99 | active picture, identical to rows 20–21 |

Run against the current tree, the function reports on the 75-bars capture —
the capture `ELLIPTICAL_COLLAPSE` §6 records as blank —

```
2 line(s) carrying a component: line 20 at 101 IRE, line 21 at 100 IRE
```

and the same two rows on `base_off`, an ordinary programme decode of a home
recording. On a fresh decode of `zaroff-ntc7composite-NTSC-SP` it reports
the recorded result exactly:

```
2 line(s) carrying a component: line 20 at 101 IRE, line 21 at 99 IRE
  entries 32, rank 32, directions 1, asymmetry 1.0000 vs floor 0.0407 (+512.1 sigma)
```

**The needle at 1.0000 is the test pattern, not the channel.** The
`ntc7composite` capture's rows 9–19 are blank at 3.8–4.4 IRE and rows 20
through 260 are *identical* — the Tektronix TSG-130A emits its patterns
**full field**, on every active line, so what the function found is the
first two lines of the picture. Every field carries the same waveform, so
32 components that are 32 views of one thing give an asymmetry of exactly
one. That is the correct answer to the question the function actually
asked, and the wrong question.

Three consequences follow:

1. the documented "the 75-bars capture is blank from line 9 to 19" no longer
   holds against the current window;
2. the recorded vertical-interval result for `ntc7composite` is a picture
   measurement;
3. on real programme material the function feeds **two lines of programme
   content** into the ellipse, which is precisely what the sync-only
   constraint forbids.

The boundary should come from the decode rather than a constant. This
decoder's fields put blanking on rows 9–19 and the first active line on row
20 on every NTSC decode examined; the honest window is the blank run itself,
found by the same departure test rather than assumed.

**The horizontal window is a fraction, not the decode's own geometry.**
`int(0.12 * width)` is sample 109 on a 910-sample NTSC field, and the JSON
gives `colourBurstEnd = 110` — so the window opens one sample *inside* the
colour burst. At the other end `int(0.95 * width)` is 864 against
`activeVideoEnd = 894`, discarding the last 30 samples of the line. The
burst leak costs about 0.24 IRE of the measured span; the 30 discarded
samples are where line-time distortion is largest.

**The guard is a hard-coded constant where the docstring promises a measured
one.** The docstring says a line counts as carrying a component "when it
departs from blanking by more than the interval's own scatter allows"; the
code compares against `guard_ire = 12.0`. The interval's own scatter is
available and differs by material:

| decode | blanking span | scatter | 12 IRE guard stands at |
|---|---|---|---|
| 75-bars, EP | 8.90 IRE | 0.26 | **11.8 σ** |
| home recording, SP | 3.93 IRE | 0.17 | **46.3 σ** |

A fixed guard therefore means a completely different confidence on
different material, and on a worse EP tape it can be crossed by blanking
alone. The arc's own rule against hard-coded constants applies.

**The synthetic side is blanking, so the differential is the signal.** The
tool builds `{"amplitude": -row}` — the standard's synthetic side for a
reserved line taken as blanking. That is right for detecting *whether* a
line carries something and wrong for measuring *what the channel did to
it*: the differential is then dominated by the inserted component itself,
about 100 IRE, rather than by the channel's effect on it, a few IRE. The
whole value of a test signal is that the synthetic side can be the
specified waveform. With blanking as the synthetic side the method is
measuring the presence of a signal at 512 sigma and learning nothing about
the path.

**The chrominance is in a different file.** The function reads only
`<stem>.tbc`. On a colour-under decode that is the luminance channel;
chrominance is in `<stem>_chroma.tbc` on its own scale. Every element that
carries subcarrier — the 12.5T pulse, the modulated staircase, the
three-level chrominance packet, the burst itself — is invisible to it. On
the `ntc7composite` decode the luma channel's burst reads 0.7 IRE
peak-to-peak against a specified 40. A composite test line does not exist
as one waveform anywhere in the decoder's output and has to be reassembled
from both files.

**Smaller items.** The axis label is `"amplitude"`, but a row over sample
index is a time-axis quantity in this arc's vocabulary — harmless, since
`ellipsoid` groups by grid, but it makes the per-axis tables read wrongly.
The line range is 525-specific: for 625 the test signals are on lines 17,
18, 330 and 331 and teletext runs to line 335. Head parity is not used, and
every other per-field measurement in this arc is separated by head. Dropouts
recorded in the JSON are not consulted, so a dropout-corrupted VBI line
enters the ensemble as a component.

### 7.3 The measurement must interpolate, not sample

One number bears on any gauge built on this material. The decoder's output
grid is 4f_sc — 910 samples on a 63.5556 µs line — and a 2T pulse is 250 ns
wide. Read naively from that grid, by taking the largest sample and
interpolating the half-amplitude crossings linearly:

| how the 2T pulse is read | HAD | error | amplitude |
|---|---|---|---|
| the specification (J.63 Annex II §2.2) | 250.00 ns | — | 100 IRE |
| **4f_sc grid, sample peak + linear crossing** | **259.52 ns** | **+3.81%** | **95.26** |
| 4f_sc data, band-limited interpolation ×2 | 249.73 ns | −0.11% | 99.94 |
| 4f_sc data, band-limited interpolation ×4 | 249.82 ns | −0.07% | 99.95 |

**The information is on the grid; the naive reading loses it.** 4f_sc
Nyquist is 7.16 MHz and the pulse is band-limited well below that, so a
single band-limited interpolation recovers the half-amplitude duration to
0.1% and the amplitude to 0.06%. A gauge that reads the sample peak reports
a pulse 3.8% wide and 4.7% short, which is several times the ±10 ns
tolerance the standard sets, purely from arithmetic.


## 8. The test material in the tree

`/testdata/test_patterns/vhs/` holds record-tap and playback-tap RF captures
of sixteen patterns in SP and EP, made from a **Tektronix TSG-130A
Multiformat Signal Generator** through a Sony SLV-778HF (`readme.txt`
records the tap points and process). `/output/test_patterns/cvbs/` holds
direct CVBS captures of the same generator's patterns.

**These are full-field patterns, not insertion test signals.** The
TSG-130A's manual Table 3-4 defines each as a field-wide pattern; only its
*Matrix* mode segments a field by line, and only Option 04 emits anything in
the vertical interval at all (VIR on line 19). Verified on a decode: rows
9–19 of `zaroff-ntc7composite` are blank and rows 20 through 260 carry the
identical NTC-7 composite line. **So the test material gives an exactly
specified synthetic side in the ACTIVE PICTURE, which under the sync-only
constraint makes it offline calibration material — the best there is — and
not a runtime correction source.**

### 8.1 What each capture is, and what it specifies

From the TSG-130A instruction manual, Table 3-4 (*NTSC/YC test signal
definitions*) and Table 3-2 (*general test signal characteristics*):

| capture | signal | the specification |
|---|---|---|
| `zaroff-ntc7composite` | NTC-7 Composite | 100 IRE bar with **125 ns rise**, 2T and 12.5T modulated pulse, **90 IRE 5-step staircase modulated with 40 IRE subcarrier** |
| `zaroff-multiburst` | Multiburst | white reference bar **500 mV p-p (70 IRE)**; packets **428.6 mV p-p (60 IRE)**, equal width; pedestal **285.7 mV (40 IRE)**; **0.5, 1.0, 2.0, 3.0, 3.58, 4.2 MHz**; packet rise **140 ns at 0.5 MHz, 400 ns elsewhere**, sin²-shaped |
| `zaroff-pulseandbar` | Pulse & bar window | **2T pulse HAD 250 ns**, white bar 100 IRE, window lines 72–202 |
| `zaroff-modulatedramp` | Modulated ramp | luminance **714.29 mV (100 IRE)**, chrominance **285.7 mV p-p (40 IRE)** |
| `zaroff-ramp` | 0–100% ramp | **714.29 mV (100 IRE)** |
| `zaroff-chromaresponse` | Chrominance frequency response | five packets at **3.08, 3.33, 3.58, 3.83, 4.08 MHz** |
| `zaroff-chromanoise` | Chrominance noise | pedestal **357.14 mV**, chrominance **714.29 mV p-p**, phase **103.5° (red)** |
| `zaroff-sweep` | Line sweep | **500 kHz to 5.0 MHz**, amplitude **714.29 mV p-p** |
| `zaroff-75bars` | 75% colour bars | 75% bars with a **100% flag** and **7.5% setup** |
| `zaroff-bounce1`, `-bounce2` | Bounce | 0 or 100 IRE flat field, **1 s high, 1 s low** |
| `*-y-only` variants | the same with chrominance suppressed | — |

Every capture also carries the generator's own sync and burst, and those
are specified too — which makes them a calibration reference for the parts
of the chain the sync-only constraint *does* permit:

| parameter | TSG-130A specification |
|---|---|
| sync amplitude | 285.7 mV ± 2% |
| horizontal sync duration, 50% points | **4.7 µs ± 50 ns** |
| vertical serration duration | 4.7 µs ± 50 ns |
| equalizing pulse duration | **2.3 µs ± 50 ns** |
| front porch | 1.5 ± 0.1 µs |
| line blanking interval | 10.9 ± 0.2 µs |
| burst amplitude | 285.7 mV p-p ± 2% |
| burst delay from sync | **5.308 µs ± 35 ns** (19 subcarrier cycles) |
| burst duration | 2.51 µs ± 0.1 (9 cycles) |
| breezeway | 600 ns ± 50 ns |
| SC/H phase | 0° ± 5° |
| sync rise time | **140 ns ± 20 ns** |
| luminance rise time | 250 ns ± 25 ns |
| chrominance rise time | 400 ns ± 40 ns |
| composite frequency response | flat to 4.2 MHz ± 2% |
| line tilt / field tilt | < 0.5% each |
| 2T pulse ringing, K2T | < 0.6% |
| differential gain / phase | 0.3% / 0.3° maximum, typically 0.1% / 0.1° |
| oscillator | 14.31818 MHz ± 28 Hz |

**One naming discrepancy to be aware of.** The VHS captures are named
`ntc7composite` while the direct CVBS capture of the same generator is named
`NTC7-combination` (`/output/test_patterns/cvbs/readme.txt`). The TSG-130A
generates both as separate patterns and only the composite appears in the
VHS set. Anything comparing the tape decode against the direct feed must
check it is comparing the same pattern.

### 8.2 A decode checked against the specification rather than against itself

This is the thing the material makes possible and that the arc has not yet
done with it. `zaroff-ntc7composite-NTSC-SP` decoded with the arc's flags,
pooled over 16 fields and 200 picture lines, resampled onto the
specification's own time axis by its sync edge, and read with the same
reader that reads the rendered synthetic line:

| reading | J.63 / VITS | decoded | departure |
|---|---|---|---|
| blanking level | 0 IRE | +1.90 | +1.90 IRE |
| sync tip | −40.00 IRE | −33.0 | **+7.0 IRE (17% shallow)** |
| sync width at the pulse's own 50% | 4.700 µs *(gen. ±50 ns)* | 4.656 | −44 ns — **inside the generator's tolerance** |
| sync rise, 10–90% | 140 ± 20 ns | **294 ns** | **2.1× slower** |
| B₂ white bar level | 100.00 IRE | 88.48 | **−11.5%** |
| B₂ bar tilt over 18 µs | ≤ 0.5% | 3.82 IRE | **7.6× the limit** |
| B₁ 2T pulse amplitude | ~100 IRE | 51.19 | **−48.5%** |
| D₂ staircase peak | 90.00 IRE | 80.40 | −10.7% |
| D₂ luminance non-linearity | ≤ 0.5% of a riser | 0.61 IRE | +0.27 IRE above the reader's bias |
| staircase terminus level | 90.00 IRE | 78.87 | −12.4% |
| head A vs head B, on all of the above | — | within 0.2 IRE | — |

Chrominance, from `<stem>_chroma.tbc`:

| reading | specified | decoded |
|---|---|---|
| burst | 40 IRE p-p | 0.72 |
| 12.5T chrominance | 100 IRE p-p | 1.39 |
| chrominance over the staircase | 40 IRE p-p | 0.44 |

Read carefully, because two different things are in that table.

**The chrominance figures are a scale question, not a loss.** The chroma
TBC is written on its own convention, and nothing in this measurement
establishes what a decoded IRE means there. What the numbers *do* establish
is the ratio between elements, and there the specification is reproduced:
12.5T chrominance to burst is specified at 100:40 = 2.50 and reads 1.39/0.72
= 1.93; staircase chrominance to burst is specified at 1.00 and reads 0.61.
Those are real departures once the scale is pinned, and pinning it is a
one-line calibration this material now makes possible.

**The luminance figures are measurements.** Three stand out. The **2T pulse
has lost about 46% of its amplitude** — the raw reading is −48.5%, and §7.3
says a naive 4f_sc reading is itself 4.7% low, so the honest figure after
that correction is 53.7 IRE against 100, a loss of 46.3%. That is the
luminance bandwidth of the recording, stated in the one number the standard
is built to give. The **bar
tilt is 7.6 times the generator's own limit**, so the line-time response is
the tape's, not the source's. And the **sync edge comes back 2.1 times
slower** than the generator emits it, 294 ns against 140 ± 20 ns — a direct
measurement of the same edge `sync_step_response.py` fits, now against a
specified transition rather than an assumed ideal.

**The sync WIDTH, by contrast, is a caution, and a sharp one.** Read at a
fixed −20 IRE from blanking it comes back 4.589 µs, 111 ns narrow against a
generator specified at 4.700 ± 0.050 µs — apparently well outside tolerance
and apparently a chain effect. Read at the pulse's *own* half-amplitude it
is 4.656 µs, **44 ns short, inside the generator's tolerance, and no
finding at all**. The difference is entirely the threshold: the recovered
tip is only 33 IRE deep, so a fixed −20 IRE level sits at 58% depth rather
than 50%, and on a 294 ns edge that reads the width narrow by 67 ns — more
than the whole tolerance. **A level threshold fixed in absolute terms is not
a width measurement once the amplitude has changed**, which is exactly the
condition a tape decode is always in. `sync_width_correction()` in
`information_extrapolation.py` declines to correct an absolute width on the
grounds that it may be the source's sync generator; this material shows the
prior question is whether the width was measured at all.

**Independently corroborated by the decoder's own instrument.**
`tools/ringing_measure/hsync_model_report.py` was run on the same decode
(the arc's standing per-decode rule; 16/16 fields healthy, 0 resets,
field-to-field stability +0.9999), and its own measurement layer reports:

| quantity | hsync model report | this document |
|---|---|---|
| sync pulse width, 50%–50% | **4.645 µs** measured, 4.700 spec | 4.656 µs at the pulse's own half-amplitude |
| sync edge transition, 10–90% | **0.349 µs** (shared across both edges) | 0.294 µs, rise alone |
| sync tip level | **−32.44 IRE** | −33.0 IRE |
| front porch / back porch | +0.13 / +3.00 IRE | blanking +1.90 IRE |

Two entirely separate code paths, on the same decode, agree that the sync
width is near 4.65 µs and not 4.59 — which settles the threshold question of
the previous paragraph. They also agree that the sync transition is 2.1 to
2.5 times the generator's specified 140 ns; the remaining spread between
them is the shared-edge-width convention the hsync report uses, a known
caveat of that instrument and not a disagreement about the signal.

The systematic ~11% level deficit shared by the bar, the staircase and the
terminus is a single gain term, and it is the kind of thing that only
becomes visible against a specified reference — measured against the
decode's own white level it is invisible by construction.


## 9. What to do with this

In rough order of value.

1. **Give `vertical_interval()` the specified waveform as its synthetic
   side.** The renderer exists (`scratchpad/vbi/vits_render.py`) and the
   machine-readable definitions exist. The differential then measures the
   channel instead of detecting the signal, which is the entire point of a
   test line.
2. **Fix the window.** Find the blank run from the decode rather than
   assuming rows 9–21; take the horizontal window from `activeVideoStart`
   and `activeVideoEnd`; derive the guard from the interval's own scatter as
   the docstring already promises.
3. **Read both TBC files.** A composite test line does not exist in either
   one alone.
4. **Interpolate before reading a pulse, and take every threshold from the
   pulse's own amplitude.** §7.3: one band-limited interpolation turns a
   3.8% half-amplitude-duration error into 0.1%. §8.2: a threshold fixed in
   absolute IRE, applied to a sync pulse whose depth the tape has changed,
   invented a 111 ns width departure that is really 44 ns and inside the
   source's own tolerance. Both defects manufacture a finding out of
   arithmetic, and both are one line to fix.
5. **Use the record/playback pair.** The material has both taps for every
   pattern, and the direct CVBS feed as well. Differencing the tape decode
   against the specification gives the whole chain; differencing it against
   the direct feed separates the recording from the generator. Section 8.2
   is the first half of that and the second half is one more decode.
6. **Treat the GCR as the target component.** It is the only VBI signal
   whose entire waveform is published on the decoder's own grid, it survives
   a VHS recording with 382 of 550 response numbers intact, and it sits on
   line 19, unambiguously inside the reserved interval. No capture in this
   tree is known to carry one — a broadcast recording from the US after
   1994 would.
7. **Look at `/output/test_patterns/575tsgWvbi_…_resample20msps.flac`.** Its
   name says TSG *with VBI*, and it is the only capture in the tree that may
   carry genuine insertion signals rather than full-field patterns. It is
   625-line and 20 Msps and did not lock with the flags tried here; it is
   worth the decode.


## 10. What runs it

Nothing in the repository was modified for this document. The working
material is under
`/tmp/claude-1000/-workspaces-vhs-decode/61dd5ae6-c67f-4bd1-bbaa-8c55cd55ffc0/scratchpad/vbi/`:

| file | what it does |
|---|---|
| `vits_render.py` | renders any of the eleven VITS definitions from its YAML, reads back the quantities its own primitives name, and measures the ellipse — §3, §4 |
| `test_signal_identifiability.py` | the same measurement written directly from the TSG-130A and J.63 numbers, and the 4f_sc grid measurement of §7.3 |
| `against_spec.py` | a decoded line against the specification, element by element — §8.2 |
| `gcr_a49_910.csv` | the 910 published GCR samples at 4f_sc, from ATSC A/49 Annex A |
| `ntsc_*.yaml`, `pal_*.yaml` | the eleven VITS definitions, fetched from the collection |
| `tsg130a.txt` | the TSG-130A manual as text |

**Sources.** ITU-T J.63 (06/90); ITU-R BT.1439-1, BT.1700, BT.1701-1,
BT.471-1, BT.472-3, BT.653-3, BT.1119-2, BT.1124-3; SMPTE 170M-2004 and
ST 12-1-2008; EBU Tech. 3209 and Tech. 3097; ETSI EN 300 706, ETS 300 231,
TR 101 233; ATSC A/49; CTA-608-E; EIA/CEA-516; IEC 61880; 47 CFR §73.682
and §73.699; Tektronix TSG130A Instruction Manual. Numbers reaching this
document through vendor datasheets or a project's own construction rather
than a standard are marked **UNVERIFIED** where they appear.
