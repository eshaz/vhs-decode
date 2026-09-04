# The elliptical collapse applied to ITU-R BT.1700

`docs/ELLIPTICAL_COLLAPSE.md` describes the estimator and what measuring it
on a tape established. This document runs the same estimator on a
specification instead of on a signal: Recommendation ITU-R BT.1700 (2005),
*Characteristics of composite video signals for conventional analogue
television systems*, together with SMPTE 170M-2004, which BT.1700 carries as
its own Annex 2 and which supplies the whole of its NTSC part.

The question the tape run answered was how many distinguishable mechanisms
six magnetic witnesses support, and the answer was 1.58 of 6. The question
here is the same one asked of a document: BT.1700 tabulates several dozen
timing quantities, and a decoder measures all of them, but the tables state
relations that tie them together. How many independent dimensions does the
specification actually have?

Everything below is computed by
`.../scratchpad/bt1700/bt1700_collapse.py`, which imports
`vhsdecode.models.information_extrapolation` and
`vhsdecode.models.capture_profile` unmodified and reads its numbers from
`bt1700_spec.py`, a transcription in which every value carries the table and
row it came from. Neither file is in the repository tree; nothing under
`vhsdecode/` was changed.


## 0. What BT.1700 actually covers, and what it does not

The brief asked about systems M, B, D, G, H, I, K, L and N. **BT.1700 does
not define those.** It is a *production and programme-interchange*
Recommendation and it defines exactly four composite formats, plus one
national variant:

| part | format | line count | source of the numbers |
|---|---|---|---|
| A | NTSC | 525 / 59.94 | deferred entirely to SMPTE 170M-2004, reproduced as BT.1700 Annex 2 |
| B | 525 PAL | 525 / 59.94 | Tables 1, 2, 3 |
| B | 625 PAL | 625 / 50 | Tables 1, 2, 3 |
| B | 625 PAL (Argentina) | 625 / 50 | the braced `{ }` values inside the 625 PAL columns |
| C | 625 SECAM | 625 / 50 | Tables 4, 5, 6 |

The letter systems are emission standards. BT.1700's own *considering* c)
points at Recommendation ITU-R BT.1701 for the radiated signal and Report
ITU-R BT.2043 for which country uses what; the letter-by-letter tables live
in ITU-R BT.470-6. BT.1700 says of 625 PAL that "although there may be
different emission standards using the 625 PAL system, there is only one
studio/production format", which is precisely the point: at baseband the
letters do not exist. Question 4 is therefore answered on the five variants
BT.1700 does define, and the answer turns out to explain why the letters are
an emission distinction rather than a baseband one.

Three further things BT.1700 does not state, recorded rather than filled in:

The **total line-blanking interval `a`** is absent from the NTSC part.
SMPTE 170M table 2 gives the front porch (1.5 µs) and the datum-to-blanking-
end interval (9.20 µs) and never their sum. So is the **field-synchronizing
pulse width `q`**: 170M table 3 gives the serration width and the three-line
block length but no width for the broad pulse.

**The back porch, the breezeway and the burst-to-blanking guard are not
tabulated anywhere in BT.1700**, in any part, for any system. All three are
differences of tabulated intervals. That matters for the rank result below,
because they are quantities a decoder measures directly and the
specification supplies no independent information about any of them.

**No sampling requirement of any kind.** BT.1700 specifies an analogue
interface; SMPTE 170M clause 15 fixes 75 Ω impedances and 140 IRE = 1 V and
says nothing about rate or word length. Section 5 below therefore derives
the capture requirement from the tolerances rather than reading it off.


## 1. The components

A component here is a quantity the standard fixes and a decoder measures.
The synthetic side is the tabulated value; the measured side is what the
decoder recovers from the waveform; the differential is their difference.

### 1.1 Line timing

Symbols are BT.1700's own (Figure 1 for PAL, Figure 10 for SECAM); the NTSC
column is SMPTE 170M's wording mapped onto them.

| sym | quantity | what a decoder measures | M/NTSC | 525 PAL | 625 PAL | 625 SECAM |
|---|---|---|---|---|---|---|
| H | line period | sync-fall to sync-fall spacing | 63.556 µs (derived) | 63.555 µs | 64 µs | 64 µs |
| a | line-blanking interval | blanking start to blanking end | *not stated* | 10.5–11.0 µs | 12 +0/−0.3 µs | 12 +0/−0.3 µs |
| b | datum to blanking end | sync 50% fall to end of blanking | 9.20 +0.20/−0.10 µs | 9.2 +0.2/−0.1 µs | 10.5 µs, no tol | 10.5 µs, no tol |
| c | front porch | blanking start to sync 50% fall | 1.5 ± 0.1 µs | 1.5 ± 0.1 µs | 1.2 +0.32/−0.0 µs † | 1.5 +0.3/−0.0 µs |
| d | sync pulse width | 50%-to-50% across the pulse | 4.70 ± 0.10 µs | 4.7 ± 0.1 µs | 4.7 ± 0.2 µs | 4.7 ± 0.2 µs |
| e | line-blanking edge | 10–90% on the blanking transition | 140 ± 20 ns | 140 ± 20 ns | 300 ± 100 ns | 300 ± 10 ns † |
| f | line-sync edge | 10–90% on the sync transition | 140 ± 20 ns | 140 ± 20 ns † | 200 ± 100 ns † | 200 ± 10 ns † |
| g | datum to burst start | first qualifying burst zero crossing | 19 cycles = 5.3078 µs | 5.3 ± 0.1 µs | 5.6 ± 0.1 µs | 5.6 ± 0.02 µs † (symbol *i*) |
| h | burst duration | first to last qualifying crossing | 9 ± 1 cycles | 2.52 ± 0.28 µs | 2.25 ± 0.23 µs | no burst |
| — | back porch | b − d | *not tabulated* | *not tabulated* | *not tabulated* | *not tabulated* |
| — | breezeway | g − d | *not tabulated* | *not tabulated* | *not tabulated* | *not tabulated* |

† flagged in §1.4 as a place where the document contradicts itself.

### 1.2 Field timing

| sym | quantity | M/NTSC | 525 PAL | 625 PAL | 625 SECAM |
|---|---|---|---|---|---|
| v | field period | 16.6833 ms | 16.6833 ms | 20 ms | 20 ms |
| j | field-blanking interval | 20 lines + 1.5 µs ± 0.1 | 20H + 1.5 µs (1272.62 µs) | 25H + a | 25H + a |
| J | field-blanking edge | 140 ± 20 ns | 140 ± 20 ns | 140 ± 20 ns † | 300 ± 100 ns |
| K | blanking to first eq pulse | 1.50 ± 0.10 µs | 1.5 ± 0.1 µs | 3 ± 2 µs | 3 ± 2 µs |
| l, m, n | equalizing / sync / equalizing sequences | 3H each | 3H each | 2.5H each | 2.5H each |
| p | equalizing pulse width | 2.30 ± 0.10 µs | 2.3 ± 0.1 µs | 2.35 ± 0.1 µs | 2.35 ± 0.1 µs |
| q | field-sync pulse width | *not stated* | 27.1 µs, no tol | 27.3 ± 0.1 µs | 27.3 µs, no tol |
| r | serration interval | 4.70 ± 0.10 µs | 4.7 ± 0.1 µs | 4.7 ± 0.1 µs | 4.7 ± 0.2 µs |
| s | sync/eq pulse edge | 140 ± 20 ns | 140 ± 20 ns | 200 ± 100 ns | 200 ± 100 ns |

### 1.3 Frequencies and levels

| quantity | M/NTSC | 525 PAL | 625 PAL | 625 PAL (AR) | 625 SECAM |
|---|---|---|---|---|---|
| line frequency f<sub>H</sub> | 15 734.2657 Hz (derived) | 15 734.26 Hz ± 0.0003% | 15 625 Hz ± 0.00002% | as 625 PAL | 15 625 Hz ± 0.016 Hz |
| subcarrier f<sub>sc</sub> | 5 MHz × 63/88 = 3 579 545.45 Hz ± 10 Hz | 3 575 611.49 Hz ± 5 Hz | 4 433 618.75 Hz ± 1 Hz | 3 582 056.25 Hz ± 5 Hz | f<sub>OR</sub> 4 406 250 ± 2000, f<sub>OB</sub> 4 250 000 ± 2000 Hz |
| f<sub>sc</sub> / f<sub>H</sub> | 455/2 | 909/4 | 1135/4 + 1/625 | 917/4 + 1/625 | 282 and 272 |
| modulation | suppressed-carrier QAM | QAM | QAM | QAM | frequency modulation |
| white level | 100 ± 1 IRE | 700 mV | 700 mV | 700 mV | 700 mV |
| sync level | −40 ± 1 IRE | −286 mV | −300 mV | −300 mV | −300 mV |
| set-up | 7.5 ± 1 IRE | 0–70 mV | 0 | 0 | 0–49 mV |
| burst amplitude | 40 ± 1 IRE | **316–317 mV** | 300 ± 30 mV | 300 ± 30 mV | subcarrier 23 ± 2.5% of luminance |
| composite p-p | 140 IRE, 171 IRE with chroma | 1330 mV | 1330 mV | 1330 mV | 1161 ± 17.5 mV |
| burst half-envelope variation | ≤ 0.5 IRE | *not stated* | *not stated* | *not stated* | — |
| SC/H phase | 0 ± 10° | *not stated* | *not stated* | *not stated* | — |

The 525 PAL burst window of 316–317 mV — a **one-millivolt band on a
1330 mV signal** — is the tightest amplitude requirement anywhere in the
Recommendation, and it drives the 525 PAL bit-depth answer in §5.

### 1.4 Where the document contradicts itself

Running the graph over the printed nominals surfaced five places where
BT.1700 states the same physical quantity twice at two different numbers.
These are reported, not repaired, and §4 and §5 are run both ways.

**625 PAL front porch.** Table 2 gives `a = 12 +0/−0.3 µs`, `b = 10.5 µs`
and `c = 1.2 +0.32/−0.0 µs`. The geometry of Figure 1 is `a = b + c`, and at
the printed nominals it is short by **300 ns**. The ranges do overlap — `a`
at its lower limit of 11.7 µs equals `b + c` at `c`'s nominal — so the table
is satisfiable, but only at a corner. Part C Table 5, describing the
identical 625-line sync geometry, gives `c = 1.5 +0.3/−0.0 µs`, which closes
`a = b + c` exactly at both nominals. The `1.2` appears to be a
transcription of `1.5`.

**Symbol `f`.** The Table 2 continuation prints `{140 ± 20 ns} 200 ± 100 ns`
across a merged cell, so which column owns which value cannot be recovered
from the text. Read here as 140 ± 20 ns for 525 PAL and 200 ± 100 ns for
625 PAL, each consistent with the same system's `e` and `s`.

**625 PAL field-blanking edge.** Table 3 gives `J = 140 ± 20 ns` across both
PAL columns while Table 2 gives the 625 PAL line-blanking edge `e` as
300 ± 100 ns. The same blanking transition is specified twice at rates
differing by 2.14×; the closure residual is **+160 ns**, outside both the
printed precision and the stated tolerances.

**SECAM edge tolerances.** Table 5 gives `e = 300 ± 10 ns` and
`f = 200 ± 10 ns`. Table 6 gives the same two edges, as `J` and `s`, at
±100 ns. Table 5's figure is an order of magnitude tighter for the same
edge.

**SECAM chrominance blanking.** Table 5 gives `i = 5.6 ± 0.02 µs` where the
corresponding PAL quantity `g` is 5.6 ± 0.1 µs. A ±20 ns tolerance on a
5.6 µs interval is the tightest timing requirement in the document.

### 1.5 What does close

Everything else does, and two of the closures are worth naming because they
are the sort of thing that is usually assumed rather than checked.

`q + r = H/2` — the broad field-synchronizing pulse plus the serration that
follows it filling one half-line — closes **exactly** on 625 PAL, Argentina
and SECAM, and to −22.5 ns on 525 PAL, which is inside the 50 ns the
printing of `27.1` permits. SMPTE 170M does not state `q` at all; the
relation predicts 27.078 µs for NTSC.

`p = d/2` — the equalizing pulse as half the line-sync pulse — is **not
stated anywhere in BT.1700**, so it was tested rather than assumed. It
closes exactly on all three 625-line variants (2.35 = 4.7/2) and misses by
**+50 ns on both 525-line variants**, where 2.30 is printed against a
predicted 2.35. That 50 ns is inside the printing precision of both numbers,
so the honest verdict is that the relation holds for 625 and is not
contradicted for 525, but the 525 tables do not state it and do not quite
imply it either.

The `fsc = k·f<sub>H</sub>` relation closes exactly for every system except
525 PAL, where it is out by **−0.905 Hz**. That is entirely the table's own
rounding of f<sub>H</sub> to 15 734.26 Hz: the exact colour line frequency is
15 734.2657 Hz, and 227.25 × 15 734.2657 reproduces the printed subcarrier.
The observation worth keeping is that **BT.1700 specifies the 525 PAL
subcarrier to ±5 Hz while printing the line frequency it is locked to at a
precision worth ±1.14 Hz of subcarrier** — the tolerance is tighter than the
document's own arithmetic.


## 2. The dependency graph

The relations are classified by provenance, because the rank result depends
on which are admitted. `TABLE` means the relation is written in a BT.1700
table or clause. `FIGURE` means it is the geometry of Figure 1, Figure 5 or
Figure 10, which the Recommendation prints and labels. `TESTED` means it is
not in BT.1700 at all and its residual is measured rather than assumed.

| relation | parents | removes | provenance |
|---|---|---|---|
| `H = 1/f_H` | f<sub>H</sub> | H | table |
| `f_V = 2 f_H / lines` | f<sub>H</sub> | f<sub>V</sub> | table |
| `v = lines / 2 f_H` | f<sub>H</sub> | v | table |
| `f_sc = k f_H` | f<sub>H</sub> | f<sub>sc</sub> | table |
| `h = n_cycles / f_sc` | f<sub>sc</sub> | h | table |
| `g = 19 / f_sc` (NTSC only) | f<sub>sc</sub> | g | table |
| `j = 25H + a` (625) / `20H + K` (525) | H, a or K | j | table |
| `l = m = n = 2.5H` (625) / `3H` (525) | H | l, m, n | table |
| `a = b + c` | b, c | a | figure |
| `back porch = b − d` | b, d | back porch | figure |
| `breezeway = g − d` | g, d | breezeway | figure |
| `guard = b − g − h` | b, g, h | guard | figure |
| `q + r = H/2` | H, r | q | figure |
| `s = f` | f | s | figure |
| `J = e` | e | J | figure |
| `p = d/2` | d | p | **tested — not in BT.1700** |

Parents, children and siblings are resolved by
`information_extrapolation.graph_relations`, called on a declaration in the
shape `residual_limit.order` consumes: one node per quantity, each writing
its own name and reading its parents. The relation map that comes back is
what `component_differentials` uses to decide which pairs to difference, and
it is the reason the nested matrix has 88 entries for 26 quantities rather
than the 351 an all-pairs matrix would have. Two quantities with no path
between them are not differenced, because a difference between them is not a
relationship anyone claimed.


## 3. The effective rank of BT.1700's timing specification

### 3.1 How it is measured

The ensemble is every departure that still conforms. Each quantity's
deviation is drawn inside its own stated tolerance and the whole vector is
then conditioned on every admitted relation, so what remains is exactly the
set of ways a conforming signal can differ from the nominal. The synthetic
value is the nominal, the measured value is the nominal plus the drawn
departure, and the differential is their difference — which is what
`component_differentials` takes, with the graph's relation map, to build the
nested matrix `ellipsoid` is fitted to. 4096 draws.

Two practical notes on the estimator, both of which cost a wrong answer
before they were fixed. The conditioning has to be done in the scaled space:
the quantities are seconds and hertz together, so the raw constraint matrix
has singular values spanning fifteen decades and a pseudo-inverse of it
silently drops every time-domain constraint, reporting more dimensions than
exist. And the null eigenvectors of the Gram are an arbitrary rotation
inside the null space, so reading the degeneracy off one of them names
nothing; what names it is the ensemble's own coherence, reported in §3.3.

### 3.2 The rank

| system | quantities | relations | rank(C) | independent dimensions | ellipsoid rank | participation |
|---|---|---|---|---|---|---|
| M/NTSC | 24 | 15 | 15 | **9** | 9 | 4.50 |
| 525 PAL | 26 | 16 | 16 | **10** | 10 | 5.78 |
| 625 PAL | 26 | 16 | 16 | **10** | 10 | 5.59 |
| 625 PAL (Argentina) | 26 | 16 | 16 | **10** | 10 | 5.62 |
| 625 SECAM | 24 | 14 | 14 | **10** | 10 | 5.90 |

Admitting only the relations written in the tables, and none of the printed
figures' geometry, the counts are 14 of 24 for NTSC and 16–17 of 26 for the
others. Admitting `p = d/2` as well takes 625 PAL to 9 of 26.

So: **BT.1700's 625 PAL timing tables list 26 measurable quantities and have
ten independent dimensions.** The ellipsoid's numerical rank equals the
constraint count's prediction exactly in every case, which is the check that
the ensemble is what it is supposed to be.

The `trace = N` invariant holds to ten decimal places, as it does on tape
data: 26.0000000000 over the quantities, 88.0000000000 over the nested
matrix. The shape carries everything. The semi-axes for 625 PAL are

```
3.0012  2.0313  1.5408  1.4384  1.4122  1.3933  1.1847  1.0488  0.9976  0.9947
```

followed by sixteen exact zeros. The asymmetry of the nested matrix is
0.9445 against a sphere floor of 0.021038, which is 2675 σ — the
specification is not a random collection of numbers, and the estimator says
so with the same statistic it uses on a tape.

The participation ratio is the direct analogue of the tape run's "1.58
distinguishable mechanisms of 6". For 625 PAL it is **5.59 of 26**, or 21.5%
of the quantities the tables list; the tape figure was 26%. The two numbers
are measuring different things — there, several physical mechanisms
producing nearly the same frequency response; here, several tabulated
symbols that are algebraically the same number — but the estimator does not
care which, and the reading is the same: most of what is written down is not
independent evidence.

### 3.3 Which quantities are degenerate with which

Read off the ensemble's coherence, groups whose unit directions coincide
across every conforming departure:

| group | what it is |
|---|---|
| f<sub>H</sub> = f<sub>sc</sub> = f<sub>V</sub> = H = v = h = l = m = n | **the frequency spine — nine symbols, one dimension** |
| a = j | the field-blanking interval is the line-blanking interval plus 25H |
| e = J | the line-blanking and field-blanking edges are the same edge |
| f = s | the line-sync and sync/equalizing edges are the same edge |
| q = r | the broad pulse and its serration fill a half-line between them |

The frequency spine is the large result. Nine of the twenty-six quantities
BT.1700 tabulates for 625 PAL — the line frequency, the subcarrier
frequency, the field frequency, the line period, the field period, the burst
duration and the three vertical-interval sequence lengths — are **one
measurement**. Fix the line frequency and every one of them follows, because
the subcarrier is locked to the line rate by an exact rational, the field
rate is the line rate over half the line count, the burst is a whole number
of subcarrier cycles, and the sequences are whole or half multiples of the
line period. A decoder that measures all nine has measured one thing nine
times and gained root-nine, not nine independent constraints.

One near-degeneracy stands just below the threshold: the datum-to-blanking-
end interval `b` and the burst-to-blanking guard sit at |cos| = 0.908,
because the guard is `b − g − h` and `b` dominates its variance.

Adding a decoder's own measurement error — 10% of each quantity's tolerance,
which is what gives the Marchenko-Pastur edge something to separate, since
the specification alone has no noise floor — the loop on the nested matrix
runs to the circle in three passes, 0.3465 → 0.2462 → 0.0000 against a floor
of 0.000244, ending at −0.7 σ. Between three and eight directions stand
above the MP edge depending on the system, resolving 7% to 27% of the
information budget.


## 4. Which system variants are distinguishable from measurement alone

Each pair is compared quantity by quantity, in units of the combined stated
tolerance: `z = |x_A − x_B| / sqrt(t_A² + t_B²)`. A pair that reaches no
`z > 1` on any timing quantity cannot be told apart by a conforming
measurement of that quantity.

### 4.1 Timing alone, as BT.1700 prints it

| pair | max z | what separates them |
|---|---|---|
| M/NTSC vs 525 PAL | **0.09** | **nothing** |
| 625 PAL vs 625 PAL (Argentina) | **0.88** | **nothing** |
| 625 PAL vs 625 SECAM | 1.57 | field-blanking edge J (1.6); front porch c (1.4) |
| 625 PAL (AR) vs 625 SECAM | 2.68 | line-sync edge f (2.7); field-blanking edge J (1.6) |
| any 525 vs any 625 | 2400 | field-blanking interval j; line frequency f<sub>H</sub>; and four more |

The ellipse over the five variants on thirteen tolerance-scaled timing
quantities has trace 5.0000000000, numerical rank 3 of 5, semi-axes
`2.2361 0.00085 0.00049 0.00004 0.0`, and a participation ratio of
**1.000**. One dimension carries everything, and that dimension is the line
count. In timing terms BT.1700 defines **two** formats, not five.

### 4.2 The two that separate the ones that can be separated

The three quantities that do separate 625 PAL from 625 SECAM — `J`, `c` and
`f` — are all on the list of places where the document contradicts itself
(§1.4). Reconciling each duplicated specification by taking the value the
*other* BT.1700 table gives for the same physical edge (625 PAL `c` = 1.5
+0.3/−0.0 from Table 5; 625 PAL `J` = 300 ± 100 ns from Table 2's `e`; SECAM
`e`, `f` = ±100 ns from Table 6's `J`, `s`; SECAM `i` = 5.6 ± 0.1 µs from
Table 2's `g`) produces this:

> **On the consistent reading, 625 PAL, 625 PAL (Argentina) and 625 SECAM
> have identical timing vectors.** The centred matrix is exactly zero and
> the ellipse is empty — not one dimension below the noise, but none. Every
> pairwise `max z` is 0.00.

The apparent separability of the 625-line variants is therefore an artefact
of the specification's internal inconsistency, not a property of the
signals.

### 4.3 What actually separates them

Adding the subcarrier to the vector:

| pair | max z | separator |
|---|---|---|
| M/NTSC vs 525 PAL | 352 | subcarrier: 3 579 545 Hz ± 10 vs 3 575 611 Hz ± 5, a 3934 Hz gap |
| 625 PAL vs 625 PAL (AR) | 167 005 | subcarrier: 4 433 619 Hz ± 1 vs 3 582 056 Hz ± 5 |
| 625 PAL vs 625 SECAM | 13.7 | subcarrier: 4 433 619 Hz ± 1 vs f<sub>OR</sub> 4 406 250 Hz ± 2000, a 27 kHz gap |
| 625 PAL (AR) vs 625 SECAM | 412 | subcarrier |

**The chrominance subcarrier is the only quantity that separates any pair of
same-line-count variants**, and the participation ratio over all five rises
only from 1.000 to 1.017 when it is included, because the line-count
direction still dominates. The measurement that actually identifies a
variant is not a timing measurement at all; it is the burst frequency, or in
SECAM's case the presence of an FM subcarrier where a burst should be.

This is also why the letter systems are absent from BT.1700 and why their
absence is correct. Everything that distinguishes B from G from I from L is
in the modulation and the RF channel — vision-to-sound spacing, vestigial
sideband width, sound carrier, video bandwidth — and none of it survives
into the baseband composite signal this Recommendation specifies. Two
625-line PAL signals from a B/PAL and an I/PAL transmitter are the same
signal at this interface.

### 4.4 The corollary for a decoder

A decoder cannot determine its system from the sync waveform. It can
determine the **line count** from the sync waveform with enormous margin
(z ≈ 2400), and everything else it must get from the colour subcarrier: the
burst frequency separates NTSC from 525 PAL and PAL from N/PAL, and the
absence of a burst-shaped back-porch reference in favour of an FM subcarrier
separates SECAM. A system-detection routine written against sync geometry
alone is measuring a quantity that carries no information about the answer.


## 5. The capture requirement

BT.1700 states no digitization requirement, so the question is turned round:
at what sample rate and word length does the *specification's* tolerance,
rather than the converter, become what limits the measurement?

### 5.1 The two rate floors

The first floor is the signal's own band. For NTSC and PAL the modulation
BT.1700 names is suppressed-carrier quadrature amplitude modulation, which
is double-sideband, so the occupied band is the subcarrier plus the
colour-difference stop frequency the standard itself asserts — 3.6 MHz at
≥ 20 dB for NTSC (SMPTE 170M 7.2) and 525 PAL, 4.0 MHz for 625 PAL, 3.6 MHz
for Argentina. For SECAM the subcarrier is frequency modulated, so
**Carson's rule applies to the standard's own figures**, and
`capture_profile.carson_bandwidth` is used exactly as it is on a VHS luma
carrier: 2 × (506 kHz maximum deviation + 1.3 MHz colour-difference band) =
3.612 MHz about f<sub>OR</sub>.

The second floor is the sync edge. SMPTE 170M 13.1 states that "raised
cosine shaping is preferred", so the edge shape is specified rather than
assumed and everything about it follows in closed form. For
`s(t) = ½(1 − cos πt/T)` the 10–90% time is 0.59033 T, the slope at the 50%
point is π/2T per unit amplitude, and the derivative is a half-cosine whose
transform first vanishes at 1.5/T — so an edge of 10–90% time `t_r` occupies
0.8855/`t_r`. No rule-of-thumb constant enters.

| system | occupied band top | edge band | **Nyquist floor** | 4f<sub>sc</sub> | 13.5 MHz |
|---|---|---|---|---|---|
| M/NTSC | 7.180 MHz | 6.325 MHz (140 ns) | **14.359 MHz** | 14.318 — below | below |
| 525 PAL | 7.176 MHz | 6.325 MHz (140 ns) | **14.351 MHz** | 14.302 — below | below |
| 625 PAL | 8.434 MHz | 4.428 MHz (200 ns) | **16.867 MHz** | 17.734 — clears | below |
| 625 PAL (AR) | 7.182 MHz | 6.325 MHz (140 ns) | **14.364 MHz** | 14.328 — below | below |
| 625 SECAM | 6.212 MHz | 4.428 MHz (200 ns) | **12.425 MHz** | 17.625 — clears | clears |

Two things fall out. For the 525-line systems it is the **sync edge, not the
subcarrier**, that sets the bandwidth: a 140 ns raised-cosine transition
reaches 6.325 MHz, well past a 3.58 MHz subcarrier, and the composite band
only just overtakes it. And **4f<sub>sc</sub> sampling falls short of the
floor for every 525-line variant** — 14.318 against 14.359 MHz, a 0.3%
shortfall — because the double-sideband chrominance the standard's own
20 dB point implies reaches 7.18 MHz while 4f<sub>sc</sub> Nyquist reaches
7.16 MHz. It is a marginal miss, and it is the standard's colour-difference
template rather than the subcarrier that causes it.

### 5.2 The word length

A uniform quantiser's rms noise is `step/sqrt(12)`
(`capture_profile.quantization_floor`). Requiring it below the tightest
amplitude tolerance the standard states, over the full scale the converter
must span:

| system | full scale | tightest level tolerance | bits |
|---|---|---|---|
| M/NTSC | 1221.4 mV (171 IRE) | 3.571 mV (burst half-envelope ≤ 0.5 IRE) | 6.63 → **7 bit** |
| 525 PAL | 1330 mV | **0.500 mV** (burst 316–317 mV) | 9.58 → **10 bit** |
| 625 PAL | 1330 mV | 30 mV (burst 300 ± 30 mV) | 3.68 → **4 bit** |
| 625 PAL (AR) | 1330 mV | 30 mV | 3.68 → **4 bit** |
| 625 SECAM | 1161 mV | 17.5 mV (composite ± 17.5 mV; subcarrier ± 2.5%) | 4.26 → **5 bit** |

The spread is the specification's, not the signal's. 625 PAL states no
tolerance at all on white level, sync level or composite amplitude, so its
tightest amplitude requirement is the ±30 mV on the burst and four bits
satisfies it. 525 PAL's one-millivolt burst window asks for ten. NTSC's
0.5 IRE half-envelope limit asks for seven. **On amplitude, BT.1700 is a
loose standard**: an ordinary 8-bit capture already sits below the stated
tolerance of every variant except 525 PAL.

### 5.3 The word length the timing needs

Timing and word length couple through the edge. A 50% crossing fitted over
the `M = f_s · t_r` samples on the transition lands within
`σ_t = (step/sqrt12) / (slope · sqrt(M))`, with the slope taken from the
raised-cosine form at the sync amplitude the standard fixes. At each
system's Nyquist floor rate:

| system | tightest timing tolerance | σ_t at 8 bit | margin | bits at which the tolerance binds |
|---|---|---|---|---|
| M/NTSC | 7.8 ns (SC/H ± 10° at f<sub>sc</sub>) | 0.513 ns | 15.1× | 4.08 → **5 bit** (5.67 at 3σ) |
| 525 PAL | 20 ns (edges e, J, s) | 0.559 ns | 35.8× | 2.84 → **3 bit** (4.42 at 3σ) |
| 625 PAL | 20 ns (edge J as printed) | 0.587 ns | 34.1× | 2.91 → **3 bit** (4.49 at 3σ) |
| 625 PAL (AR) | 20 ns (edge f as braced) | 0.532 ns | 37.6× | 2.77 → **3 bit** (4.35 at 3σ) |
| 625 SECAM | 10 ns (edges e, f as printed) | 0.597 ns | 16.7× | 3.93 → **4 bit** (5.52 at 3σ) |

On the consistent reading of §4.2 the 625-line timing tolerances relax to
±100 ns and the required word length falls further.

### 5.4 The answer

| system | sample rate | bit depth | what sets each |
|---|---|---|---|
| M/NTSC | **≥ 14.36 MHz** | **7 bit** | rate by the 140 ns sync edge and the 3.6 MHz chrominance template; depth by the 0.5 IRE burst half-envelope limit |
| 525 PAL | **≥ 14.35 MHz** | **10 bit** | rate as NTSC; depth by the 316–317 mV burst window |
| 625 PAL | **≥ 16.87 MHz** | **4 bit** | rate by the 4.4336 MHz subcarrier plus a 4.0 MHz double sideband; depth by the ± 30 mV burst |
| 625 PAL (AR) | **≥ 14.36 MHz** | **4 bit** | rate by the 140 ns sync edge; depth by the ± 30 mV burst |
| 625 SECAM | **≥ 12.43 MHz** | **5 bit** | rate by Carson on the FM subcarrier; depth by the ± 17.5 mV composite tolerance |

In every case **the rate binds and the depth does not**. Past about
14–17 MHz and 8 bits, further capture quality buys nothing that BT.1700
asks for: at 8 bit the edge-timing precision is 15 to 38 times finer than
the tightest timing tolerance in the standard, and the quantiser noise is
already below the tightest amplitude tolerance for four of the five
variants.

That is the same verdict, arrived at from the other end, that
`docs/ELLIPTICAL_COLLAPSE.md` §7.1 records for tape: the converter holds
2.2× the tape's capacity and spends 67 Mbit/s on nothing. Here the converter
holds far more than the *specification* asks for. Neither the medium nor the
standard is limited by the capture. What limits both is identifiability —
there, nine magnetic mechanisms collapsing to 1.58 distinguishable ones;
here, twenty-six tabulated timing quantities collapsing to ten independent
dimensions and five variants collapsing to one.


## 6. Summary of the findings

**Effective rank.** BT.1700's timing specification lists 24–26 measurable
quantities per system and has **9 to 10 independent dimensions** (NTSC 9 of
24; 525 PAL, 625 PAL and Argentina 10 of 26; SECAM 10 of 24). The
participation ratio, the direct analogue of the tape run's "1.58 of 6", is
5.59 of 26 for 625 PAL. Nine symbols — f<sub>H</sub>, f<sub>sc</sub>,
f<sub>V</sub>, H, v, the burst duration and the three vertical-interval
sequence lengths — are **one** measurement.

**Indistinguishable variants.** From timing alone, M/NTSC and 525 PAL are
indistinguishable (max z 0.09), and 625 PAL and Argentina 625 PAL are
indistinguishable (0.88). 625 PAL and 625 SECAM appear to separate at 1.57 σ
but only on quantities BT.1700 specifies twice at different values; on the
reading that reconciles them, all three 625-line variants have **identical**
timing vectors. The participation ratio over all five variants is 1.000: the
line count is the only timing dimension. The chrominance subcarrier is the
only quantity that separates any same-line-count pair.

**Capture requirement.** 14.36 MHz / 7 bit for NTSC, 14.35 MHz / 10 bit for
525 PAL, 16.87 MHz / 4 bit for 625 PAL, 14.36 MHz / 4 bit for Argentina,
12.43 MHz / 5 bit for SECAM. The rate binds in every case; the depth is
already met by an ordinary 8-bit converter for every variant except 525 PAL,
whose 316–317 mV burst window is the tightest amplitude requirement in the
document. 4f<sub>sc</sub> sampling clears the floor for the 625-line
variants and falls 0.3% short for the 525-line ones.

**Five internal contradictions** were found on the way, all of them by the
closure test rather than by reading: the 625 PAL front porch (300 ns short
of `a = b + c`), the merged `f` rise-time cell, the 625 PAL field-blanking
edge (160 ns, specified twice at 140 ns and 300 ns), the SECAM edge
tolerances (±10 ns against ±100 ns for the same edges), and the SECAM
chrominance blanking tolerance (±20 ns against PAL's ±100 ns). Three of the
five are exactly the quantities that appeared to separate the 625-line
variants, which is why the separation does not survive reconciliation.
