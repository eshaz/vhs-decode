# The elliptical collapse on broadcast videotape recorders

`docs/ELLIPTICAL_COLLAPSE.md` is the parent document and remains the
authority for the algorithm itself. This one asks a narrower question: what
happens when the same apparatus is pointed at a BROADCAST format instead of
a domestic one, with U-matic as the subject and the other broadcast VTR
families sketched after it.

Two results from the VHS work are the things worth testing, because both
were ceilings rather than achievements:

* the tape's six magnetic loss mechanisms proved only **1.58 effectively
  distinguishable of 6**, because every one of them is a function of
  spacing over wavelength and so is nearly the same monotone decay in
  frequency;
* the capture holds **2.2x the tape's Shannon capacity** in the signal band
  (tape 21.6 dB C/N over Carson's 8.0 MHz), so the radio channel binds and
  not the converter.

The broadcast formats were the obvious place to look for relief. U-matic
records at nearly twice VHS's writing speed onto a track half again as
wide, with a carrier 1.4 MHz higher and a deviation 60% larger. If any of
that were going to break the collinearity, it would show here.

**It does not.** The short answer to all three questions is at §7.


## 1. Where the numbers came from, and where they did not

The format specification collection at
`https://github.com/decode-orc/analogue-video-specifications` **holds no
U-matic document and no broadcast VTR document of any kind.** Its `mkdocs.yml`
navigation, fetched in full, lists exactly five subject areas — television
and video formats (ITU-R BT.470-6, BT.601-5, BT.624-4, BT.1700, EBU Tech.
3213-E and 3280-E, SMPTE 170M/244M/272M), video metadata, teletext,
LaserDisc, Compact Disc — and one videotape format, VHS, carrying SMPTE
32M-2004 and the JVC Video Technical Guide VTG82063. There is nothing for
U-matic, Betacam, M/MII, Type C or quadruplex. That is worth recording
plainly: the VHS work could lean on a manufacturer's own technical guide
section by section, and this work cannot.

So the U-matic constants below come from three places, and each row says
which:

| source | what it supplies |
|---|---|
| **this repository** | every frequency the decoder already uses — carriers, deviation, colour-under, de-emphasis, band-pass limits. `vhsdecode/format_defs/umatic.py`, cited by line. |
| **published format references** | the mechanics the repository does not carry: tape speed, drum rate, writing speed, head count. Wikipedia and its HandWiki mirror, agreeing verbatim. |
| **derived here** | track pitch, from the format's own tape speed, drum rate and track angle. Marked *derived*. |

Anything not obtainable from those is marked **ASSUMED** and the effect of
the assumption is stated. There are two such quantities and both turn out
not to matter; see §3.2.


## 2. U-matic's component set, beside VHS's

### 2.1 The mechanics

| quantity | VHS NTSC SP | U-matic NTSC | U-matic PAL | source |
|---|---|---|---|---|
| tape width | 1/2 in nominal | 19 mm (3/4 in) | 19 mm | not carried by this repository; U-matic: Wikipedia |
| linear tape speed | 33.35 mm/s | 95.25 mm/s | 95.25 mm/s | `head_model.py:60`; U-matic 3.75 in/s (Wikipedia), converted |
| drum diameter | 62 mm | 110 mm | 110 mm | `head_model.py:58`; U-matic **corroborated**, see below |
| drum rotation | 30.0 Hz | 29.97 Hz (1800 rpm) | 25 Hz (1500 rpm) | `head_model.py:59`; Wikipedia |
| video heads | 2 | 2 | 2 | Wikipedia |
| scanning | non-segmented, 1 field per sweep | non-segmented | non-segmented | Wikipedia ("each head records one complete field") |
| writing speed, **stated** | 5.80 m/s | 10.26 m/s | 8.54 m/s | `head_model.py:65`; Wikipedia |
| drum surface speed, pi·D·f | 5.843 m/s | 10.367 m/s | 8.639 m/s | computed |
| `head_model.writing_speed` fallback, pi·D·f + tape speed | 5.877 m/s (+1.32%) | 10.463 m/s (+1.97%) | 8.735 m/s (+2.28%) | computed |
| track angle | 5° 58′ 09.9″ | 4° 58′ 06″ | 4° 58′ 06″ | `head_model.py:68`; U-matic: PALsite format page, via search index |
| track pitch, *derived* | — | 137.6 µm | 165.1 µm | computed, see §2.3 |
| video track width | 58 µm | ~85 µm **ASSUMED** | ~95 µm **ASSUMED** | `format_defs/vhs.py:302`; U-matic derived, see §2.3 |
| track separation | azimuth recording, ±6°, no guard band | **guard band**, 0.070 mm low band / 0.040 mm hi band | as NTSC | `head_model.py:72`; U-matic: PALsite format page, via search index |

**U-matic is a guard-band format, not an azimuth format, and that costs the
arc a component.** The two are alternatives: azimuth recording exists to let
adjacent tracks abut without crosstalk, and a format that leaves a physical
guard band does not need it. U-matic's guard band is documented; no source
found states an azimuth alternation angle for it, where the VHS ±6° is
carried in this repository. The consequence is stated in `head_model.py:26-32`
in reverse: azimuth is called "THE term that separates two VHS heads",
*because* the format writes adjacent tracks at opposite azimuth so a
systematic azimuth error cannot affect both heads alike. On U-matic the two
heads are nominally at the SAME azimuth, so a systematic azimuth error
affects both alike and cancels in the head difference. `head_difference_log`
and `fit_head_difference` still work, but the mechanism they were built to
catch is no longer differential, and what remains in a U-matic head
difference is gain, spacing, gap and contour — the first two of which lie in
the carrier law's null space (`CARRIER_LAW_NULL_SPACE`) and must be fitted
on the RF envelope. **On U-matic the head difference is a weaker instrument
than on VHS**, which is the opposite of what the format's superiority would
suggest.

The drum diameter is the one U-matic mechanical figure worth defending,
because everything downstream of it depends on it. It is corroborated
rather than asserted: 110 mm at 30 Hz gives a drum surface speed of
10.367 m/s against the stated writing speed of 10.26 m/s, and at 25 Hz
gives 8.639 against 8.54 — both about 1.1% high. Put through
`head_model.writing_speed`'s own fallback, which adds the linear tape speed
to the drum's surface speed, they become 10.463 and 8.735, or 2.0% and 2.3%
high. The same fallback on VHS gives 5.877 against a stated 5.80, the 1.3%
the repository already documents at `head_model.py:61-66`. Same sign, same
order of magnitude, on a completely different transport. That discrepancy
is not an error in either figure: the head
sweeps helically while the tape moves, so the velocity ALONG THE TRACK is
the vector sum at the track angle, and it is always a little less than the
drum's surface speed. Finding the same 1.0-1.3% offset on a completely
different transport is a consistency check on the whole set. **The stated
figure wins**, exactly as `head_model.writing_speed` already decides.

### 2.2 The signal

Every one of these is already in this repository. This is the part where
U-matic is better served than its mechanics are.

| quantity | VHS NTSC SP | U-matic low band | U-matic hi band (BVU) | U-matic SP | source |
|---|---|---|---|---|---|
| sync-tip carrier | 3.4 MHz | 3.8 MHz | 4.8 MHz | 5.6 MHz | VHS `capture_profile.py:47`; U-matic from `umatic.py:58,174,185` with `hz_ire` |
| peak-white carrier | 4.4 MHz | 5.4 MHz | 6.4 MHz | 7.2 MHz | as above |
| deviation (sync to white) | 1.0 MHz | 1.6 MHz | 1.6 MHz | 1.6 MHz | `umatic.py:59` gives `hz_ire = 1600000/140` |
| Hz per IRE | 7 143 | 11 428.6 | 11 428.6 | 11 428.6 | `umatic.py:59,173,184` |
| ire0 | — | 4 257 143 Hz | 5 257 143 Hz | 6 057 143 Hz | `umatic.py:58,244`; `:174`; `:185` |
| colour-under, NTSC | 629 370.6 Hz | 688 374.1 Hz | — | — | `vhs.py:305`; `umatic.py:205` |
| colour-under, PAL | 626 953 Hz | 685 546.9 Hz | 923 828.1 Hz | 923 828.1 Hz | `vhs.py:154`; `umatic.py:18`; `umatic.py:1` |
| de-emphasis tau | 1.30 µs | 600 ns | 800 ns | 800 ns | `vhs.py:320`; `umatic.py:27,88,140` |
| decoder RF band-pass | 0.5–6.5 MHz | 1.4–6.5 MHz (NTSC) | 2.5–10.0 MHz | 1.4–10.0 MHz | `vhs.py:282-283`; `umatic.py:194-195,69-70,121-122` |
| post-demod LPF | 6.6 MHz | 4.0 MHz (NTSC) | 4.2 MHz (PAL) | 4.2 MHz | `vhs.py:296`; `umatic.py:201,76,128` |
| luma baseband, *derived* | 3.0 MHz | 3.0 MHz | 3.4 MHz | 4.125 MHz | see below |

The carriers cross-check against the outside world exactly: the repository's
`ire0` and `hz_ire` for low band give 3.8 MHz at sync tip and 5.4 MHz at
peak white, which is what Wikipedia states, and the NTSC low-band
colour-under evaluates to 688 374.1 Hz against a published 688.373 kHz.

**The luma baseband is derived, not sourced.** Neither the repository nor
any reference found states U-matic's luma bandwidth in megahertz; what is
published is horizontal resolution — 240 TV lines low band, 330 lines SP.
At the usual 80 TVL per megahertz for a 525-line system those are 3.0 MHz
and 4.125 MHz, and the hi-band figure of 3.4 MHz is interpolated. This
matters only in §4, where it sets Carson's band; the sensitivity is stated
there and it is small.

**One defect found on the way.** `vhsdecode/format_defs/umatic.py:1` reads

```python
UMATIC_HI_COLOR_UNDER_CC = 923828  # (625 * 25) * (351 / 8)
```

The comment is the LOW-BAND formula, copied. `(625 * 25) * (351 / 8)` is
685 546.875, which is the low-band value on line 18, not 923 828. The
hi-band value is `(625 * 25) * (473 / 8)` = 923 828.125. The constant is
right and only the comment is wrong, so nothing decodes incorrectly, but a
reader deriving the format from the comment would get the wrong carrier.
Flagged, not fixed — this document does not touch code.

### 2.3 What the repository does NOT carry

`head_model.FORMAT_MECHANICS` holds two entries, `("VHS","NTSC","SP")` and
`("VHS","PAL","SP")`, and nothing else, so `head_model.mechanics_for` returns
`None` for U-matic and every head-model fit in this arc is unavailable on a
U-matic decode. `format_defs/umatic.py` sets no `video_track_width`, which
is the one mechanical figure the VHS path takes from the decoder's own
parameters rather than from the mechanics table.

The track pitch can be derived rather than looked up, from quantities that
are all sourced. Two tracks are laid per drum revolution, so tracks arrive
at the field rate, and the tape advances by (linear speed)/(field rate)
between them. The pitch measured perpendicular to the track is that
advance times the sine of the track angle:

    NTSC:  95.25 mm/s / 59.94 Hz = 1.5891 mm,  x sin(4°58'06") = 137.6 µm
    PAL:   95.25 mm/s / 50.00 Hz = 1.9050 mm,  x sin(4°58'06") = 165.1 µm

The VHS figure the repository carries, 58 µm at SP, is the recorded track
width and not the pitch, and the two differ by the guard band. A guard band
of 0.070 mm for low band and 0.040 mm for hi band is attributed to U-matic
by the PALsite format reference; subtracting the low-band figure from the
PAL pitch gives about 95 µm of recorded track, and from the NTSC pitch
about 68 µm. The primary page (`umatic.palsite.com/format.html`) was
returning HTTP 520 throughout this work and could not be read directly, and
its width figure as reproduced by search indexes — "0.83 mm" — is
inconsistent with the pitch derived above by a factor of five, so it is
almost certainly a transcription of 0.083 mm. **The track width is
therefore ASSUMED at 85 µm (NTSC) and 95 µm (PAL), and §3.2 shows the
identifiability answer is completely insensitive to it** — identical to
three decimal places across 58 µm to 137 µm. Nothing in this document rests
on the assumption.


## 3. Does identifiability improve? No — it is very slightly worse

### 3.1 The measurement, and the VHS baseline reproduced first

The test is the one that produced 1.58 of 6: perturb each mechanism in
`head_model.log_response`, take the resulting change in log magnitude as
that mechanism's signature over the format's own RF band, normalise each
signature to a unit direction, and read the participation ratio of the
Gram's eigenvalues — which is exactly what
`information_extrapolation.ellipsoid` returns as `participation`.

Before running U-matic the VHS row was reproduced, because a like-for-like
comparison is worthless if the two are measured differently. The band used
is the decoder's own RF band-pass for that format, 0.5–6.5 MHz from
`format_defs/vhs.py:282-283`; the perturbation moves each of spacing, gap
and thickness 10% off `head_model.TYPICAL` and switches azimuth (0.1°),
contour (depth 0.1) and gain (0.5 dB) on from zero.

| | published | reproduced here |
|---|---|---|
| spacing vs gap | 0.962 | 0.963 |
| spacing vs thickness | 0.994 | 0.998 |
| gap vs azimuth | **1.000** | **1.000** |
| gain vs spacing | 0.894 | 0.896 |
| contour vs everything | 0.000–0.002 | 0.001–0.009 |
| singular values | 2.159, 1.000, 0.565, 0.133, 0.0078, 0.0002 | 2.163, 1.001, 0.552, 0.120, 0.0056, 0.000024 |
| one direction explains | 77.7% | 78.0% |
| **effective mechanisms** | **1.58 of 6** | **1.57 of 6** |

Every coherence agrees to within 0.005 and the effective count to 0.01, so
the procedure is the published one and the U-matic rows below are
comparable to it.

### 3.2 U-matic

| format | writing speed | RF band | one direction | **effective of 6** |
|---|---|---|---|---|
| VHS NTSC SP | 5.80 m/s | 0.5–6.5 MHz | 78.0% | **1.57** |
| U-matic NTSC low band | 10.26 m/s | 1.4–6.5 MHz | 79.5% | **1.51** |
| U-matic PAL low band | 8.54 m/s | 1.4–7.0 MHz | 79.3% | **1.52** |
| U-matic PAL hi band | 8.54 m/s | 2.5–10.0 MHz | 79.6% | **1.51** |
| U-matic PAL SP | 8.54 m/s | 1.4–10.0 MHz | 78.5% | **1.55** |

The azimuth mechanism is kept in the U-matic ensemble even though the format
does not alternate azimuth between tracks (§2.1). That is correct and not an
oversight: azimuth loss is a single-head loss — a head whose gap is not
square to the track loses short wavelengths on its OWN track — and it is
present whether or not the neighbour is written at the opposite angle. What
the guard band removes is azimuth's usefulness as a head DIFFERENCE, not its
presence as a loss.

**Worse, marginally, on every variant.** The coherence structure is
unchanged in kind and slightly tightened in degree: gap and azimuth remain
exactly collinear at 1.000, spacing and thickness rise from 0.998 to 1.000,
gain against spacing rises from 0.896 to 0.937. Contour remains the one
orthogonal direction, at 0.001–0.010, because it is an oscillation and not
a decay.

### 3.3 Which variable actually moves the number, and it is none of the mechanics

Holding the perturbation fixed and moving one thing at a time:

| what changes | effective of 6 |
|---|---|
| VHS, as published | 1.566 |
| VHS speed, VHS band, U-matic **track width** (85 µm) | 1.567 |
| U-matic speed, U-matic width, **VHS band** | 1.561 |
| VHS speed, VHS width, **U-matic band** | 1.520 |
| U-matic, all three | 1.514 |

**Track width is inert.** Across 58, 85, 95, 105 and 137 µm the answer is
1.514 to three decimals, and only at an absurd 830 µm does it move at all
(1.545). The reason is structural: track width enters `log_response` in
exactly one place, the azimuth term's `track_width_m * tan(azimuth_error)`,
and there only as a product with the azimuth angle. **A wider track is
algebraically degenerate with a smaller azimuth error.** It changes what
azimuth value fits, never what shape azimuth makes, so it cannot separate
azimuth from anything. This is also why the assumption of §2.3 costs
nothing.

**Writing speed is nearly inert, and what motion there is runs the wrong
way.** Sweeping it over the whole plausible range of magnetic recording,
with everything else held:

| writing speed | effective of 6 |
|---|---|
| 4.85 m/s (VHS PAL) | 1.523 |
| 5.80 m/s (VHS NTSC) | 1.520 |
| 8.54 m/s (U-matic PAL) | 1.515 |
| 10.26 m/s (U-matic NTSC) | 1.514 |
| 21.4 m/s (Type C class) | 1.512 |
| 41.3 m/s (quadruplex class) | 1.511 |

(All rows on the U-matic band and track width, so only the speed moves; the
VHS rows therefore read 1.520 rather than the 1.566 of §3.1, which is the
band difference and not the speed.)

An eightfold increase in writing speed buys −0.8%. The reason is the same
one that made the mechanisms collinear in the first place, seen from the
other side. Every loss in the stack is a function of a length over a
wavelength, and the wavelength is `v/f`, so **every mechanism is a function
of `d·f/v` and raising `v` is a similarity transformation on the frequency
axis.** It rescales all six signatures by the same factor at once. The
shape relationships between them are almost preserved by construction, and
what little does change is a second-order effect of the band's endpoints
sliding to a different part of each curve — which, since a lower `d·f/v`
sits deeper in the small-argument regime where a sinc is indistinguishable
from a parabola and a Wallace exponential from a straight line, makes them
slightly MORE alike, not less.

**The only thing that moves the number is fractional bandwidth** — how many
octaves of the loss curves the observation covers:

| band | ratio | effective of 6 |
|---|---|---|
| 3.0–4.0 MHz | 1.3 | 1.395 |
| 2.5–5.0 MHz | 2.0 | 1.433 |
| 2.0–6.0 MHz | 3.0 | 1.477 |
| 1.4–6.5 MHz (U-matic NTSC) | 4.6 | 1.514 |
| 0.5–6.5 MHz (VHS NTSC) | 13.0 | 1.561 |
| 0.2–12.0 MHz | 60 | 1.589 |

That is the whole of U-matic's deficit. Its carrier sits higher, so the
decoder's band-pass starts at 1.4 MHz where VHS's starts at 0.5 MHz, and
the observation covers 4.6 octave-equivalents where VHS covers 13. Put the
two formats on the SAME band and they agree to within 0.006 (1.520 against
1.514): the mechanics contribute nothing and the band contributes
everything.

The ceiling is structural rather than instrumental. Pushed to bands no
magnetic recording could ever occupy, the effective count reaches 1.59 at a
100:1 band and 2.39 at 10⁴:1, and then FALLS again as the higher decades
put every mechanism into a common deep null. **Six loss mechanisms measured
through a frequency response are worth somewhere between one and a half and
two and a half independent numbers, and no format changes that.**

### 3.4 The verdict

> **U-matic's effective number of distinguishable mechanisms is 1.51 of 6
> on low band and 1.55 on SP, against VHS's 1.58 (1.57 as reproduced here).
> It is the same, to the accuracy the question deserves, and marginally
> worse.**

The physical question — whether a higher writing speed and a wider track
separate the loss mechanisms — is answered no, and the reason is that
neither of those is the relevant variable. They remain functions of spacing
over wavelength; writing speed only rescales the wavelength axis and track
width only trades against azimuth. **Identifiability, not the algorithm,
was the VHS ceiling, and it is the same ceiling here.** Separating these
mechanisms still needs a witness that is not a frequency response: a second
tape, a second deck, a different tape speed, or the contour direction,
which remains the one that stands apart.

There is one genuinely actionable consequence, and it is about the decoder
rather than the format. Since fractional bandwidth is the only lever, the
head-model fit should be run over the **widest band the RF actually
occupies**, not the band-pass the demodulator wants. On U-matic the luma
band-pass discards everything below 1.4 MHz, and the colour-under sits at
688 kHz in the same RF path, already measured by this arc's chroma
envelope. Extending the fit down to it:

| band the fit sees | effective of 6 |
|---|---|
| the luma band-pass, 1.4–6.5 MHz | 1.514 |
| down to the colour-under, 0.688–6.5 MHz | 1.551 |
| VHS's own band, 0.5–6.5 MHz | 1.561 |
| VHS as published | 1.566 |

That one change recovers 0.037 of the 0.052 gap to VHS — about seven tenths
of it — at no cost in hardware and no new measurement, only in where the
existing envelope is read.


## 4. Which link binds for U-matic

### 4.1 The band and the converter

Carson's rule over U-matic's own figures, with `capture_profile.carson_bandwidth`:

| format | deviation | luma baseband | Carson band |
|---|---|---|---|
| VHS NTSC SP | 1.0 MHz | 3.0 MHz | 8.00 MHz |
| U-matic low band | 1.6 MHz | 3.0 MHz *(derived)* | **9.20 MHz** |
| U-matic hi band | 1.6 MHz | 3.4 MHz *(derived)* | **10.00 MHz** |
| U-matic SP | 1.6 MHz | 4.125 MHz *(derived)* | **11.45 MHz** |

The capture profile is the one the parent document's real run used: 8 bit,
50 MSps, 117 of 256 codes occupied. Its `signal_rms_codes` is not printed
there, so it was recovered by inverting the 46.7 dB the document reports —
24.96 codes — and the whole VHS row then reproduces exactly (tape 21.56 dB
against 21.6, capture 46.70 dB against 46.7, capacities 57.4 and
124.1 Mbit/s, ratio 2.16x against 2.2x). The U-matic rows use that same
profile unchanged, so only the band differs.

| format | rate | capture C/N in band | capture capacity |
|---|---|---|---|
| VHS NTSC SP | 40 MSps | 45.73 dB | 121.5 Mbit/s |
| VHS NTSC SP | 50 MSps | 46.70 dB | 124.1 Mbit/s |
| U-matic low band | 40 MSps | 45.12 dB | 137.9 Mbit/s |
| U-matic low band | 50 MSps | 46.09 dB | 140.9 Mbit/s |
| U-matic hi band | 40 MSps | 44.76 dB | 148.7 Mbit/s |
| U-matic hi band | 50 MSps | 45.73 dB | 151.9 Mbit/s |
| U-matic SP | 40 MSps | 44.17 dB | 168.0 Mbit/s |
| U-matic SP | 50 MSps | 45.14 dB | 171.7 Mbit/s |

The capture's C/N falls as the band widens, because quantisation noise is
flat to Nyquist and a wider band admits more of it; its CAPACITY rises
anyway, because the band grows faster than the C/N falls.

### 4.2 What U-matic's tape C/N should be

There is no U-matic capture in this arc, so the tape side is a
**projection**, not a measurement, and it is computed with the repository's
own model rather than asserted. `tape_model.magnetic_limits` gives the
particle-noise limit: a head reads a volume of one wavelength by the track
width by the recording depth, the signal adds coherently and the particle
noise as a square root, so the C/N goes as `10·log10(lambda·W·delta)`. At
each format's own carrier, with the recording depth held at
`head_model.TYPICAL["thickness_m"]` = 0.20 µm. The carrier column is the
midpoint of each format's sync-tip-to-peak-white span, which is where the
carrier sits on average as the picture sweeps it:

| format | writing speed | track width | carrier | particle C/N | vs VHS |
|---|---|---|---|---|---|
| VHS NTSC SP | 5.80 m/s | 58 µm | 3.9 MHz | 38.39 dB | — |
| U-matic low band NTSC | 10.26 m/s | 85 µm | 4.6 MHz | 41.81 dB | **+3.42 dB** |
| U-matic hi band PAL | 8.54 m/s | 95 µm | 5.6 MHz | 40.64 dB | +2.25 dB |
| U-matic SP PAL | 8.54 m/s | 95 µm | 6.4 MHz | 40.06 dB | +1.67 dB |

The absolute figures are the medium's own theoretical ceiling and sit well
above the 21.6 dB actually measured on VHS, because the measured envelope
spread carries head, transport and preamp noise too. What transfers is the
DIFFERENCE, and only under the assumption that the non-particle
contributions are comparable between the two machines. Applying it:

    U-matic low band   21.6 + 3.4  =  ~25.0 dB   PROJECTED
    U-matic hi band    21.6 + 2.3  =  ~23.8 dB   PROJECTED
    U-matic SP         21.6 + 1.7  =  ~23.2 dB   PROJECTED

Note that the higher-band variants project WORSE, not better, despite being
the better formats. That is not a contradiction: they buy their picture
quality by moving the carrier up, and moving the carrier up shortens the
wavelength, which shrinks the volume the head reads. The improvement is
spent on deviation and bandwidth rather than banked as C/N.

### 4.3 The crossover, which is the useful number

The crossover does not depend on the projection at all. The converter takes
over exactly when the tape's C/N in the band exceeds the capture's, and the
capture's is a property of the capture and the band alone:

| format | 40 MSps, 8 bit | 50 MSps, 8 bit |
|---|---|---|
| VHS NTSC SP | 45.7 dB | 46.7 dB |
| **U-matic low band** | **45.1 dB** | **46.1 dB** |
| U-matic hi band | 44.8 dB | 45.7 dB |
| U-matic SP | 44.2 dB | 45.1 dB |

> **An 8-bit capture at 40 or 50 MSps stops being adequate for U-matic once
> the tape's carrier-to-noise exceeds about 44–46 dB. The projected tape
> C/N is 23–25 dB. The margin is about 20 dB, and the tape binds.**

The capacity ratios, at the projected tape C/N:

| format | tape capacity | capture capacity (50 MSps) | ratio |
|---|---|---|---|
| VHS NTSC SP | 57.4 Mbit/s | 124.1 Mbit/s | 2.16x |
| U-matic low band | 76.4 Mbit/s | 140.9 Mbit/s | 1.84x |
| U-matic hi band | 79.1 Mbit/s | 151.9 Mbit/s | 1.92x |
| U-matic SP | 88.3 Mbit/s | 171.7 Mbit/s | 1.94x |

The headroom narrows from 2.2x to about 1.9x, which is the honest size of
the effect: U-matic is a better tape and asks a wider band, and both eat
into the converter's surplus. Neither comes close to consuming it.

**So the practically useful answer is that broadcast formats do NOT need
better capture hardware than VHS does.** The 20 dB of margin would survive
a format with fifty times VHS's tape C/N, and no analogue videotape has
that. To lose the margin at 8 bits the tape would have to reach 45 dB RF
carrier-to-noise, which is above the theoretical particle-noise ceiling of
the medium itself (§4.2 puts that at 38–42 dB). Going to 10 bits raises the
converter to 58 dB in the same band and 12 bits to 70 dB, so the surplus can
be widened arbitrarily and there is no format for which it needs to be.

The sensitivity to the derived luma baseband is small and does not threaten
this: moving the low-band figure from 3.0 MHz to 4.0 MHz widens Carson's
band from 9.20 to 11.20 MHz and moves the crossover from 46.1 to 45.2 dB at
50 MSps — under a decibel, against a 20 dB margin.


## 5. The transport

`transport_model.rotation_rates` needs the linear tape speed and the field
rate, and nothing else format-specific: the drum's rate comes from
`FIELDS_PER_DRUM_REVOLUTION = 2`, which is correct for U-matic unchanged,
since it too lays one field per head sweep with two heads on the drum. The
part diameters in `TRANSPORT` are declared as VHS-typical and are marked
`certain: False` for everything but the drum, so two tables are given: one
running the repository's own diameters at U-matic's tape speed, and one with
U-matic-plausible diameters, **ASSUMED and labelled as such.**

The VHS table was reproduced first, and matches the parent document exactly
(0.2801, 0.8166, 1.7693, 4.8253, 1.0616, 29.9700 Hz; 9 of 28 pairs at
0.23 s, 17 at 1.0 s, 26 at 5.0 s and saturating there).

**U-matic, repository diameters at 95.25 mm/s:**

| part | rate | acts on | | |
|---|---|---|---|---|
| supply reel | 0.8001 Hz | tension | before capstan | band 0.337–1.263 Hz |
| impedance roller | 2.3322 Hz | tension | before | |
| entry guide | 5.0532 Hz | tension | before | |
| exit guide | 5.0532 Hz | tension | before | |
| pinch roller | 3.0319 Hz | speed | after capstan | |
| capstan | 13.7814 Hz | speed | before | |
| head drum | 29.9700 Hz (NTSC) / 25.0000 Hz (PAL) | both | before | **certain** |
| take-up reel | 0.8001 Hz | tension | after capstan | band as above |

**How much tape must be observed:**

| observed | resolution | VHS | U-matic, repo diameters | U-matic, plausible diameters |
|---|---|---|---|---|
| 0.23 s | 4.348 Hz | 9 of 28 | 13 of 28 | 7 of 28 |
| 0.50 s | 2.000 Hz | 13 of 28 | 23 of 28 | 16 of 28 |
| 1.0 s | 1.000 Hz | 17 of 28 | 25 of 28 | 22 of 28 |
| **2.0 s** | 0.500 Hz | 25 of 28 | **26 of 28** | **27 of 28** |
| 5.0 s | 0.200 Hz | **26 of 28** | 26 of 28 | 27 of 28 |
| 120 s | 0.008 Hz | 26 of 28 | 26 of 28 | 27 of 28 |

> **U-matic saturates at two seconds where VHS needs five.**

The reason is that U-matic runs the tape 2.86 times faster (95.25 mm/s
against 33.35 mm/s), so every rate that comes from a diameter is
proportionately higher and the GAPS between them widen by the same factor,
while the drum's rate stays where the field rate puts it. The smallest
non-zero gap goes from 0.2450 Hz on VHS, which needs 4.08 s to resolve, to
0.6997 Hz, which needs 1.43 s. The slowest rate — the reels, which set how
long one cycle takes at all — goes from 0.2801 Hz (3.57 s per revolution)
to 0.8001 Hz (1.25 s). Both bounds tighten together, and 2 seconds clears
both.

Larger U-matic parts pull this partly back. With plausible broadcast-transport
diameters (30 mm impedance roller, 10 and 12 mm guides, 8 mm capstan, 20 mm
pinch roller, reels 26–60 mm — **all assumed**) the rates fall to 0.418,
1.011, 3.032, 2.527, 3.790, 1.516 Hz and the slowest cycle stretches to
2.39 s; saturation is still reached at 2 seconds, and at 27 of 28 rather
than 26, because giving the entry and exit guides different diameters
breaks one of the two degeneracies the VHS table carries. The remaining
degenerate pair is the two reels, which share a nominal band by
construction and are told apart by `after_capstan` and `acts_on` rather
than by rate — exactly as the parent document says, and not something more
tape would fix.

A second-order caution: raising the tape speed pushes the capstan to
13.78 Hz on the repository's 2.2 mm capstan, which is close enough to the
25 Hz PAL drum rate that a harmonic search would want care, and a real
U-matic capstan is much larger than 2.2 mm. This is precisely why the
plausible-diameter table is given beside it; **neither is a specification,
and a real transport's rates should be read from the machine's service
manual before a search is told where to look.**


## 6. The rest of the broadcast VTR family

Sketched by what each does to the component set, since the format constants
for these are not in the specification collection and would have to be
sourced individually. The recurring question is whether the algorithm needs
**re-parameterising** — new numbers in existing tables — or **extending** —
new structure. The dividing lines turn out to be three: segmentation,
head count, and whether the chroma is heterodyned at all.

**Betacam and Betacam SP.** The closest to a pure re-parameterisation, and
therefore the right one to do next after U-matic. Non-segmented, one field
per head sweep, so `FIELDS_PER_DRUM_REVOLUTION = 2` and the
whole per-head structure — `head_dc_components`, the per-head envelope
averaging, `head_response_difference` — carries over untouched. Two things
change. First, **there is no colour-under**: Betacam records Y on one track
and compressed time-division-multiplexed chrominance on a second, parallel
track written by its own head, so the drum carries a luma head set and a
separate chroma head set rather than one set doing both. That deletes the
entire heterodyne family —
`heterodyne_timing_scale`, "chroma heterodyne scale", "colour-under
heterodyne frequency", "color-under footprint under the luma" — and adds
one that does not exist in any domestic format: a **channel-to-channel
differential between the Y head and the C head**, which are different heads
on the same drum reading different tracks. That is a new component, not a
re-parameterised one, and it is a good one: it is a second head pair whose
difference is measurable against the same time base, which is exactly the
"second witness that is not a frequency response" §3.4 says the magnetic
mechanisms need. Second, the CTDM compression means the chroma time base is
a different scale from the luma's on the same line, so the time axis
acquires a per-channel scale factor the current model has no place for.

**M and MII.** Structurally Betacam's answer on VHS-width tape: also
component recording with separate Y and C tracks, also non-segmented, also
two heads. The same deletions and the same new head-pair differential
apply. MII's higher writing speed and different track pitch are
re-parameterisation only, and by §3.3 they will not move the
identifiability figure. The one genuinely different component is that these
formats' Y and C channels have DIFFERENT de-emphasis and different
deviation, so `picture_components`' "emphasis tuning" becomes two
independent components rather than one.

**1-inch Type C.** Non-segmented, and the interesting one for that reason:
a single video head writes a whole field in one sweep of a very large drum
at a writing speed several times U-matic's. Everything in the per-head
apparatus **collapses to one head**, which is a simplification with a sting
in it. `head_dc_components` explicitly exists because the two VHS heads are
independent and must never be pooled; with one head there is nothing to
pool, so that component vanishes and with it the head difference —
`head_difference_log`, `predicted_video_effect`, `fit_head_difference` —
which is where the VHS work put azimuth. **The head difference is the arc's
cleanest measurement and Type C does not have one.** Against that, Type C
records a whole field per sweep with no head switch inside the picture,
which deletes the `head switch` component and its DC step entirely, and
gives a field-length continuous observation where VHS gives two half-field
ones. Type C also carries a separate sync head writing the field-blanking
interval the video head misses, which is a second channel with its own
geometry — again new structure rather than new numbers.

**2-inch quadruplex.** The format that genuinely breaks the model, and in
two independent ways. First, **four heads rather than two.** The
head-differential component is written throughout this arc as a PAIR: a
boolean head parity, `Dict[bool, np.ndarray]` in `per_head_differential`
and `head_response_difference`, a per-head regression with two groups, an
"opposite azimuth" argument that presumes two alternating states. Four
heads make that a set, and the honest generalisation is not four
independent constants but the **ellipsoid over a four-element ensemble** —
which is, pleasingly, the algorithm this document is about, applied one
level down. Six pairwise differences among four heads span three
dimensions, so the head ensemble is itself rank-deficient by construction
in exactly the way §8 point 5 of the parent document describes for the
nested matrix. Second, **quadruplex is SEGMENTED and transverse**: the head
wheel sweeps across the tape's width rather than along it, each sweep
writing only about sixteen lines, so a field is assembled from
sixteen-odd segments written by four heads in rotation. The arithmetic is
worth stating because it is what breaks the code: a head wheel turning at
240 Hz and carrying four heads produces 960 sweeps per second, which over
59.94 fields per second is 16.0 sweeps per field and about 16.4 lines per
sweep. Every assumption
that one head sweep equals one field fails. `FIELDS_PER_DRUM_REVOLUTION`
is meaningless; the per-field axis (`FIELD_AXIS`, `field_differential`,
`per_head_average`) has to become a per-SEGMENT axis with the field
reconstructed above it; the head-switch component moves from twice per
frame to sixteen times per field and stops being an edge artefact and
starts being the dominant periodic structure; and banding — the segment-rate
visibility of head-to-head mismatch — becomes a first-class component that
has no analogue in any two-head format. The transport model changes too:
the head wheel's rate is no longer half the field rate but a mechanical rate
of its own, so `drum_rate_hz` stops being derivable from the field rate and
must be declared.

**Type B (Bosch BCN).** Segmented like quadruplex but helical like Type C:
two heads on a small drum, each sweep writing about 52 lines, a field
assembled from segments. It is the useful intermediate case for testing a
segmentation extension, because it keeps the two-head structure the code
already has and changes only the sweep-to-field relationship. If the
per-segment axis is built for Type B it will serve quadruplex with the head
axis widened.

### Where the algorithm re-parameterises, and where it must be extended

| change | re-parameterise | extend |
|---|---|---|
| different writing speed, track width, tape speed | yes — one `FORMAT_MECHANICS` entry | |
| different carriers, deviation, de-emphasis | yes — `format_defs` | |
| different drum diameter and part diameters | yes — `TRANSPORT` per format | |
| separate Y and C tracks (Betacam, M/MII) | | **yes** — delete the heterodyne family, add a Y-head/C-head differential and a per-channel time scale |
| one video head (Type C) | | **yes** — the head-difference components have no referent; a separate sync-head channel appears |
| segmented scanning (Type B, quad) | | **yes** — the per-field axis becomes per-segment; head switch becomes periodic structure; banding becomes a component |
| four heads (quad) | | **yes** — the boolean head parity becomes an ensemble; the head differential becomes an ellipsoid over four |
| head wheel rate independent of field rate (quad) | | **yes** — `drum_rate_hz` must be declared, not derived |

Nothing in the collapse itself needs changing for any of them. `ellipsoid`,
`sphere_floor`, `component_differentials` and `differentiate_to_floor` are
indifferent to what the components are; the closed forms are functions of
the component count and the vector length only. What needs extending is the
COMPONENT MODEL — which physical parts exist, how many of each, and on what
axis they are indexed.


## 7. The three answers

1. **Identifiability does not improve.** U-matic returns **1.51 of 6
   effectively distinguishable mechanisms on low band and 1.55 on SP**,
   against VHS's 1.58 (reproduced here as 1.57 under an identical
   procedure). Marginally worse, and the same to the accuracy the question
   deserves. Writing speed is a similarity transformation on the frequency
   axis and cannot separate mechanisms that are all functions of `d·f/v`;
   track width is algebraically degenerate with azimuth error and is inert
   from 58 to 137 µm. The only lever is fractional bandwidth, and U-matic's
   higher carrier gives the decoder a narrower one.

2. **The tape binds, with about 20 dB to spare.** U-matic's Carson band is
   9.20 MHz at low band and 11.45 MHz at SP. An 8-bit capture reaches
   46.1 dB C/N in the low-band case at 50 MSps and 45.1 dB at 40 MSps, so
   **the converter becomes the binding limit only once the tape's own
   carrier-to-noise exceeds about 44–46 dB** — which is above the
   particle-noise ceiling of the medium itself. The projected U-matic tape
   C/N is 23–25 dB. The capture holds 1.8–1.9x the tape's capacity, down
   from VHS's 2.2x but nowhere near consumed. **Broadcast formats do not
   need better capture hardware than VHS does.**

3. **The algorithm extends, rather than re-parameterises, at three places:**
   component recording with separate Y and C head channels (Betacam, M,
   MII) deletes the heterodyne family and adds a head-to-head channel
   differential; single-head non-segmented scanning (Type C) removes the
   head difference the arc's cleanest measurements depend on; and segmented
   four-head scanning (quadruplex, and Type B for two heads) turns the
   boolean head parity into an ensemble and the per-field axis into a
   per-segment one. The collapse itself — the ellipsoid, the
   Marchenko–Pastur edge, the sphere floor, the Wiener weight — needs no
   change at all for any of them.

And one incidental finding worth acting on within VHS as well as U-matic:
since fractional bandwidth is the only variable that moves the effective
mechanism count, **the head-model fit should be run over the widest band
the RF occupies rather than over the demodulator's band-pass.** On U-matic,
extending it down from the 1.4 MHz luma band-pass to the 688 kHz
colour-under carrier moves the effective count from 1.514 to 1.551 and
recovers about seven tenths of its deficit against VHS.


## 8. What runs it

Read-only scripts, outside the repository, under
`/tmp/.../scratchpad/umatic/`:

| script | what it does |
|---|---|
| `identifiability.py` | the perturbation, the coherence matrix and the effective count, per format, through `head_model.log_response` and `information_extrapolation.ellipsoid` |
| `sweep.py` | the one-variable-at-a-time separation of speed, width and band |
| `binding.py` | Carson's band, the capture profile recovery, `tape_model.magnetic_limits` and the crossover |
| `transport.py` | `transport_model.rotation_rates` and the pair-separation table |

No repository file was modified. The one defect found (`umatic.py:1`'s
comment) is reported in §2.2 and left in place.
