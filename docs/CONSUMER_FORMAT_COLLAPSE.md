# The consumer helical formats, through the elliptical collapse

Betamax, Video8, Hi8 and the VHS variants — VHS-C and S-VHS — measured on
the identifiability test of `docs/ELLIPTICAL_COLLAPSE.md` §7.2, with each
format's own mechanics in `head_model.log_response`.

`docs/ELLIPTICAL_COLLAPSE.md` is the parent and remains the authority for
the method; `docs/BROADCAST_VTR_COLLAPSE.md` asked the same question of
U-matic and is the row this document is measured against. Nothing here
modifies code — every number is read out of the repository, quoted from a
source, or computed by the repository's own functions.

**The established results this is measured against, not re-derived.** VHS
tape magnetics 1.58 effectively distinguishable of 6, condition 1.06 × 10⁴,
gap-vs-azimuth coherence exactly 1.000, because every loss is a function of
one dimensionless group. U-matic 1.51–1.55 of 6, with writing speed and
track width inert and only fractional bandwidth moving the number.
Propagation 3.79 of 6, condition 6.4. The modelled interference set on the
VHS luma band 6.89 of 12, condition 10.1.


## 1. Summary

| | |
|---|---|
| effective mechanisms, all thirteen consumer variants, head built for the format | **1.565 to 1.578 of 6** |
| the whole spread accounted for by fractional bandwidth | **R² = 0.998, largest residual 0.0005** |
| best azimuth geometry among the consumer formats | **PAL 8 mm, `w/v` = 11.03 µs against VHS's 10.00 — 10%** |
| the only large gain in the head-difference instrument | **the raised carrier: S-VHS 2.9x, Hi8 PAL 3.6x** |
| which link binds, every format, at 8 bits and 40 or 50 MSps | **the tape**, by 2.0x to 2.2x in capacity |
| repository support for these formats in the model | **absent**: `FORMAT_MECHANICS` holds VHS SP only |

Three findings are worth carrying out of this document.

**The consumer formats are one format as far as this measurement is
concerned.** Thirteen format-and-system combinations return effective
counts between 1.565 and 1.578 once each is given a head built for its own
wavelengths, and the 0.013 of spread is a pure function of how many octaves
the observation covers. Betamax's larger drum and faster writing speed,
8 mm's smaller tape and slower one, and S-VHS's and Hi8's raised carriers
change nothing about *which* loss mechanism can be told from which.

**Azimuth is not the lever it looks like.** The angle and the track width
enter `log_response` only as a product, so a larger angle buys exactly what
a wider track buys and no more; and the azimuth loss is an *even* function
of the error, so the ± alternation that the formats are built around
produces two heads with *identical* magnitude responses from a single
misalignment. What the consumer formats have actually done with their
different angles is hold the colour-under crosstalk suppression roughly
constant — 28 to 40 dB across the band for every one of them.

**What does move the instrument is the carrier, and Hi8 and S-VHS are the
controlled experiment that shows it.** Same transport, same heads, same
speed, higher carrier: the azimuth term's sinc argument at the top of the
band rises by 1.5 to 1.6x and the video-domain head difference by 2.8 to
2.9x, with no change in geometry at all — and identifiability moves by 0.7%,
exactly the amount the band ratio predicts.


## 2. The formats and their constants

### 2.1 What the repository encodes

Read by calling the repository's own `formats.get_format_params` — no
figure in this table is typed in. The FM triple is
`ire0 + vsync_ire·hz_ire` (sync tip), `ire0` (blanking) and
`ire0 + 100·hz_ire` (peak white).

| format | system | sync | blank | white | deviation | colour-under | RF band-pass | Y LPF | track width |
|---|---|---|---|---|---|---|---|---|---|
| VHS SP | NTSC | 3.400 | 3.686 | 4.400 | 1.000 | 629.371 kHz | 0.50–6.50 | 6.60 | 58.0 µm |
| VHS SP | PAL | 3.800 | 4.100 | 4.800 | 1.000 | 626.953 kHz | 1.30–5.78 | 3.40 | 49.0 µm |
| VHS LP | NTSC | 3.400 | 3.686 | 4.400 | 1.000 | 629.371 kHz | 0.50–6.50 | 6.60 | 29.0 µm |
| VHS EP | NTSC | 3.400 | 3.686 | 4.400 | 1.000 | 629.371 kHz | 0.50–6.40 | 6.60 | 19.3 µm |
| S-VHS | NTSC | 5.400 | 5.857 | 7.000 | 1.600 | 629.371 kHz | 2.00–8.98 | 7.50 | 58.0 µm |
| S-VHS | PAL | 5.400 | 5.880 | 7.000 | 1.600 | 626.953 kHz | 2.00–8.98 | 7.50 | 49.0 µm |
| Betamax | NTSC | 3.600 | 3.943 | 4.800 | 1.200 | 688.374 kHz | 1.60–5.38 | 3.00 | — |
| Betamax | PAL | 3.800 | 4.220 | 5.200 | 1.400 | 687.500 kHz | 1.90–6.10 | 3.50 | — |
| Betamax Hi-Fi | NTSC | 3.800 | 4.200 | 5.200 | 1.400 | 688.374 kHz | 2.00–5.88 | 3.00 | — |
| SuperBeta | NTSC | 4.750 | 5.150 | 6.150 | 1.400 | 688.374 kHz | 2.00–6.88 | 4.00 | — |
| Video8 | NTSC | 4.200 | 4.543 | 5.400 | 1.200 | 743.444 kHz | 0.02–7.00 | 3.50 | — |
| Video8 | PAL | 4.200 | 4.560 | 5.400 | 1.200 | 732.422 kHz | 0.02–7.00 | 3.50 | — |
| Hi8 | NTSC | 5.700 | 6.271 | 7.700 | 2.000 | 743.444 kHz | 1.12–18.21 | 5.00 | — |
| Hi8 | PAL | 5.700 | 6.300 | 7.700 | 2.000 | 732.422 kHz | 1.12–18.21 | 5.00 | — |

(MHz unless stated. `Y LPF` is `video_lpf_freq`, the decoder's post-demod
low-pass, which is not the format's luma bandwidth and is not used as one
below.)

Where each figure lives, and what the file actually encodes:

| file:line | what it encodes |
|---|---|
| `vhsdecode/format_defs/vhs.py:12-21` | the VHS de-emphasis shelf, computed from IEC 774-1 (1994) p.67 — a 1.3 µs time constant and a 4:1 divider |
| `vhsdecode/format_defs/vhs.py:152` | PAL VHS recorded track width per speed, `[49.0, 24.5, 24.5]` µm, cited to JVC VTG82063 |
| `vhsdecode/format_defs/vhs.py:299-302` | NTSC VHS recorded track width per speed, `[58.0, 29.0, 19.3]` µm, cited to JVC VTG82063 §1 |
| `vhsdecode/format_defs/vhs.py:154`, `:305` | colour-under: 40 f_H + 1953 Hz (PAL), 40 f_H (NTSC) |
| `vhsdecode/format_defs/vhs.py:312` | `luma_carrier` = 455·f_H/2 — set for NTSC VHS only, and **read nowhere in the tree** |
| `vhsdecode/format_defs/vhs.py:46-47`, `:253-256`, `:393-396` | S-VHS: the "5.4-7.0 ± 0.1 mhz" and "white clip at 210%" comments, and the 1.6 MHz deviation as `hz_ire`/`ire0` for both systems |
| `vhsdecode/format_defs/betamax.py:26-32` | PAL Betamax colour-under as the mean of the two track frequencies, 685 546.88 and 689 453.12 Hz, cited to *Television* (UK) June 1980 |
| `vhsdecode/format_defs/betamax.py:132` | NTSC Betamax colour-under, 43.75 f_H |
| `vhsdecode/format_defs/betamax.py:85-88`, `:267-270`, `:293-296`, `:319-322` | the four Betamax FM triples (PAL, NTSC, NTSC Hi-Fi, SuperBeta) |
| `vhsdecode/format_defs/betamax.py:44`, `:144` | `deemph_tau` 1.30 µs; **no `deemph_q`**, so `filter_model.assumed_parameters` supplies 0.5 |
| `vhsdecode/format_defs/video8.py:4-13` | the 8 mm shared block: chroma band-pass 180 kHz–1.15 MHz, FM audio carriers at 1.5 and 1.7 MHz |
| `vhsdecode/format_defs/video8.py:21-22`, `:93-98` | the FM triples as comments — "sync 4.2 mhz / peak white 5.4 mhz" and "sync 5.7 / peak white 7.7", with "white clip 220% - 10.1mhz?" and "dark clip 90%" |
| `vhsdecode/format_defs/video8.py:169`, `:176` | colour-under: 46.875 f_H (PAL), 47.25 f_H (NTSC) |
| `vhsdecode/format_defs/video8.py:142` | Hi8 `deemph_tau` 0.47 µs, marked "From spec" |
| `vhsdecode/format_defs/video8.py:106-108` | Hi8 `deemph_mid` 760 kHz, marked "Not correct, needs to be fixed properly" |
| `vhsdecode/format_defs/video8.py:50-51` | Video8 RF band-pass low edge 20 kHz — a placeholder, not a band edge (§3.2) |
| `vhsdecode/format_defs/video8.py:123-124` | Hi8 RF band-pass high edge 18.21 MHz — likewise |

**No `video_track_width` exists outside `vhs.py`.** Betamax, Video8, Hi8,
U-matic, Type C, EIAJ, VCR, Video2000 and VHD all omit it, so
`chroma.py:591` falls back to `REFERENCE_TRACK_WIDTH = 58.0`
(`chroma.py:49`) — VHS SP's width — for every one of them. **VHS-C is not a
tape format in the decoder at all**: the supported set at
`vhsdecode/main.py:38-60` has no entry, and a VHS-C tape decodes as VHS,
which is right for the signal and wrong for the recording transport (§7).

Three figures in `betamax.py` do not agree with the published ones. The
file's own comments say so — "Needs to be verified", "NOT CORRECT!!",
"eyeball tweaking" — and this document only records the discrepancy:

| entry | repository | published | source |
|---|---|---|---|
| Betamax NTSC | 3.6 / 4.8 MHz, 1.2 dev | 3.6 / 4.8, 1.2 | agrees — [Sencore TT189] |
| Betamax PAL | 3.8 / 5.2 MHz, 1.4 dev | 3.8 / 5.2, 1.4 | agrees — [*Television* Jul 1984] |
| Betamax Hi-Fi NTSC | 3.8 / 5.2 MHz, 1.4 dev | **4.0 / 5.2, 1.2** (the standard triple shifted +400 kHz) | [Wikipedia: Betamax] |
| SuperBeta NTSC | 4.75 / 6.15 MHz, 1.4 dev | **4.4 / 5.6, 1.2**; 6.0 MHz belongs to Super Hi-Band | [Sencore TT189] |

The in-code comment at `betamax.py:90` and `:272`, "Beta black level 3.8,
tip 5.2", has its two labels swapped: the *Television* July 1984
spectrum-analyser photograph from Sony (UK) is captioned "Sync tip 3.8MHz /
Peak white 5.2MHz". The numbers the code computes from it are right; only
the comment is wrong.

### 2.2 The mechanics

SMPTE 32M (title block 32M-1998, filed as 32M-2004) and the JVC Video
Technical Guide VTG82063 are the primary sources for the VHS family, both
transcribed in
[decode-orc/analogue-video-specifications](https://github.com/decode-orc/analogue-video-specifications)
under `docs/vhs/`. SMPTE 32M is 525-line only; PAL figures come from JVC.
**That repository has no Betamax or 8 mm material** — its tape coverage is
VHS alone — so those columns come from Sony patents, service manuals and
the contemporary trade press, and every figure carries its grade.

| | VHS SP | VHS-C | S-VHS | Betamax βII | Video8 SP | Hi8 SP |
|---|---|---|---|---|---|---|
| tape width | 12.65 mm ±0.01 [S §6.1.2] | 12.65 mm [S §6.1.2] | 12.65 mm [S §7.4.1] | 12.7 mm (½ in) [T1] | 8 mm [2°] | 8 mm [2°] |
| tape speed NTSC | 33.35 mm/s ±0.5% [S §3.1.3] | 33.35 mm/s [J §3.3.10] | 33.35 mm/s [S §7.4.1] | 20.0 mm/s [W] | 14.345 mm/s [P] | 14.345 mm/s [P] |
| tape speed PAL | 23.39 mm/s [J tbl 1-1-1] | 23.39 mm/s [J] | 23.39 mm/s [J] | 18.73 mm/s [PS] | 20.051 mm/s [P] | 20.051 mm/s [P] |
| drum diameter | 62.00 mm ±0.01 [S §3.1.5] | **41 mm** [J §3.3.10(1)] | 62.00 mm [S §3.1.5] | 74.487 mm [PS] | 40 mm [P] | 40 mm [P] |
| drum rate NTSC | 29.97 Hz | **45 Hz (2700 rpm)** [J §3.3.10(1)] | 29.97 Hz | 30 Hz (1800 rpm) [W] | 29.97 Hz [P] | 29.97 Hz [P] |
| drum rate PAL | 25 Hz | **37.5 Hz (2250 rpm)** [J] | 25 Hz | 25 Hz (1500 rpm) [PS] | 25 Hz [P] | 25 Hz [P] |
| heads / wrap | 2 / 180° | **4 / 270°** [J §3.3.10(1)] | 2 / 180° | 2 / 186° [PS] | 2 / 221° = 180 video + 36 PCM + 5 [P] | as Video8 |
| writing speed NTSC | 5.80 m/s [S §3.1.4] | 5.80 m/s (interchange) [J] | 5.80 m/s [S §3.1.4] | 6.9 m/s [W] (6.993 [D]) | 3.75 m/s [P] | 3.75 m/s [P] |
| writing speed PAL | 4.85 m/s [J tbl 1-1-1] | 4.85 m/s [J] | 4.85 m/s [J] | 5.832 m/s [PS] | 3.12 m/s [P] | 3.12 m/s [P] |
| track width NTSC | 0.058 mm [S tbl 2] | 0.058 mm [J] | 0.058 mm [S §7.2] | 30 µm, βI 60 µm [G]; βIII 19.5 µm [2°] | 20.5 µm (LP 10.2) [P] | 20.5 µm [P] |
| track width PAL | 0.049 mm [J] | 0.049 mm [J] | 0.049 mm [J] | 32.8 µm [T1] | 34.4 µm (LP 17.2) [P] | 34.4 µm [P] |
| track pitch | = width [S tbl 2] | = width | = width | = width [G, T1] | = width [P] | = width [P] |
| guard band | **none** | none | none | **none** [T1, G, PS] | **none** [P] | none [P] |
| azimuth | **+6° and −6°** [S tbl 2], 6° ±10′ [J] | ±6° [J] | ±6° [S §7.2] | **±7°** [T1, T2, PS] | **±10°** [P] | ±10° [P] |
| track angle | 5°58′9.9″ [J tbl 1-1-1] | as VHS | as VHS | 5°00′ static, 5°58′ moving [PS] | 4.885° lead still angle [P] | as Video8 |
| head gap | — | — | — | **0.4 µm** [T1] | — | — |
| sync / white | 3.4 / 4.4 MHz ±0.1 [S §3.9.1.1.4] | as VHS | **5.4 / 7.0 ±0.1** [S tbl 6] | 3.6 / 4.8 [Sc]; PAL 3.8 / 5.2 [T5] | 4.2 / 5.4 [SM] | **5.7 / 7.7** [SM] |
| deviation | 1.0 MHz ±0.1 [S] | 1.0 MHz | **1.6 MHz ±0.1** [S tbl 6] | 1.2 NTSC / 1.4 PAL | 1.2 MHz [SM] | **2.0 MHz** [SM] |
| colour-under | 629.371 kHz [S §3.9.2.1.4] | 629.371 kHz | **unchanged** [S §7.5.1.2] | 688 kHz NTSC [Sc]; PAL 685.547/689.453 [T4] | 743.444 kHz [2°] | 743.444 kHz |
| luma baseband | ~3.0 MHz [C] | ~3.0 MHz | **5.0 MHz** [J §1.1.19.1] | ~3.2 MHz [2°] | ~3.0 MHz [A] | ~5.0 MHz [A] |
| resolution | 240 lines NTSC [J §1.1.1] | 240 lines | **>400 lines** [J §1.1.1] | 240–260 lines [2°] | 240–250 lines [2°] | **>400 lines** [SM] |
| coercivity | 50 × 10³ A/m [S §3.2.1.4] | 50 × 10³ A/m | **70 × 10³ A/m** [S §7.3.3] | not found | metal particle, no figure [SM] | MP or ME, no figure [SM] |
| white clip | 160% [S §3.9.1.1.3] | 160% | **210%** [S §7.5.1.1.4] | not found | 220% → 10.1 MHz [`video8.py:96`, P] | as Video8 |

Key: **[S]** SMPTE 32M · **[J]** JVC VTG82063 · **[P]** Sony patents
US 6,674,961 B1 (FIG. 6/7 spec tables), US 4,907,102, US 5,488,482,
US 6,038,098 · **[SM]** Sony CCD-V800E PAL Hi8 service manual · **[PS]**
Betamax PALsite format page · **[T1]/[T2]/[T4]/[T5]** *Television* (UK)
Aug 1983 / Sep 1983 / Jun 1980 / Jul 1984 · **[G]** Goodman, *Maintaining
& Repairing Videocassette Recorders*, 1983 · **[Sc]** Sencore Tech Tip
TT189 · **[W]** English Wikipedia · **[2°]** other secondary · **[C]**
`capture_profile.py:47-49` · **[D]** derived · **[A]** assumed.

Two corrections to figures that circulate widely. **VHS-C's camcorder drum
turns at 2700 rpm in NTSC, not 2250** — 2250 rpm is the PAL number — and
each of its four heads writes a *whole* field, not a partial one: at 45 rps
a 270° sweep is 0.75 revolution, which is exactly 1/60 s. JVC states the
drum as "41 mm"; the value that makes πDn match the full-size drum exactly
is 62 × 30/45 = 41.333 mm, and no source states it — it is arithmetic, and
an independent service source describes the drum only as "two-thirds the
size of the normal VHS drum". And **Betamax's NTSC writing speed of
6.9 m/s does not close arithmetically**: π × 74.487 mm × 29.97 s⁻¹ minus the
tape's 20 mm/s is 6.993 m/s, and the same calculation reproduces the PAL
figure to four digits (5.831 against the published 5.832), so 6.9 is very
likely a truncation. It changes nothing below — the whole 1.3% moves the
azimuth leverage by 1.3% and the effective count by 0.0002.

### 2.3 A cross-check on the track widths, and the one identity used

Betamax's and 8 mm's track widths came from secondary sources, so they were
checked against an identity that needs no track angle. For a guard-bandless
format whose tracks are contiguous,

    track pitch = H · (linear tape speed) / (writing speed)

where `H` is the across-tape height of the video band: the tape advances
`v_tape / f_field` per field and the pitch is that times `sin α`; the track
is `v_head / f_field` long and `H` is that times `sin α`; the field rate and
the angle both cancel.

Calibrated once per format, it reproduces every published width:

| | published width | back-solved H |
|---|---|---|
| VHS NTSC SP / LP / EP | 58.0 / 29.0 / 19.3 µm | 10.09 / 10.09 / 10.07 mm |
| VHS PAL SP / LP | 49.0 / 24.5 µm | 10.16 / 10.16 mm |
| Betamax NTSC βI / βII / βIII | 60 / 30 / 19.5 µm | 10.35 / 10.35 / 10.12 mm |
| Betamax PAL βII | 32.8 µm | 10.21 mm |
| Video8 NTSC SP / LP | 20.5 / 10.2 µm | 5.36 / 5.34 mm |
| Video8 PAL SP / LP | 34.4 / 17.2 µm | 5.35 / 5.35 mm |

Each format's `H` is constant across its own speeds and systems to better
than 1%, and Betamax's back-solved 10.2–10.35 mm sits against the
independently published "video recording zone 10.6 mm" of the *Television*
August 1983 article. **The three published width sets are mutually
consistent**, which is the check they were put to; nothing below is
computed from the identity.

Two width figures disagree between sources and the disagreement is inside
the noise of everything that follows: Betamax NTSC βII is 30 µm as a track
pitch in Goodman and 29 µm as a track width in the Japanese Wikipedia
article, and Betamax PAL is 32.8 µm in the *Television* table and 32 µm in
the PALsite page. The 3% between them moves the azimuth leverage of §4.3 by
3% and the effective count of §3.3 by 0.0001.


## 3. Identifiability per format

### 3.1 The method, and the VHS baseline reproduced first

Perturb each mechanism in `head_model.log_response`; take the change in log
magnitude as that mechanism's signature over the format's own band;
unit-normalise the rows; report the participation ratio of the Gram's
eigenvalues, which is what `information_extrapolation.ellipsoid` returns as
`participation`. Spacing, gap and thickness move 10% off
`head_model.TYPICAL`; azimuth (0.1°), contour (depth 0.1) and gain (0.5 dB)
switch on from zero. 512 bins, linear in frequency.

Reproduced on VHS NTSC SP over 0.5–6.5 MHz, against the parent document's
published row and the U-matic document's independent reproduction of it:

| | published | U-matic doc | here |
|---|---|---|---|
| spacing vs gap | 0.962 | 0.963 | 0.963 |
| spacing vs thickness | 0.994 | 0.998 | 0.998 |
| gap vs azimuth | **1.000** | **1.000** | **1.000** |
| gain vs spacing | 0.894 | 0.896 | 0.896 |
| contour vs everything | 0.000–0.002 | 0.001–0.009 | 0.000–0.003 |
| singular values | 2.159 1.000 0.565 0.133 0.0078 0.0002 | 2.163 1.001 0.552 0.120 0.0056 0.000024 | 2.163 1.000 0.554 0.120 0.0056 0.000024 |
| one direction explains | 77.7% | 78.0% | 78.0% |
| **effective of 6** | **1.58** | **1.57** | **1.567** |

The condition number is the one figure that does not reproduce — 8.9 × 10⁴
here against the published 1.06 × 10⁴, because the sixth singular value
differs by an order of magnitude. It is not a grid effect: the effective
count moves by 0.0002 between 64 and 2048 bins while the condition moves by
5%, and the U-matic document's independent run lands on the same
2.4 × 10⁻⁵. **The effective count is the robust statistic and the condition
number is not.** The count also holds at 1.565–1.568 as the perturbation
size is swept from 1% to 20%.

### 3.2 The observation band, and why the decoder's band-pass cannot be it

`docs/BROADCAST_VTR_COLLAPSE.md` used each format's own
`video_bpf_low`/`video_bpf_high`. That works for VHS and U-matic and fails
here: Video8 declares a low edge of 20 kHz (`video8.py:50`) and Hi8 a high
edge of 18.21 MHz (`video8.py:123-124`), and neither is a band edge — they
are "do not filter here" placeholders, with the real shaping done by
`video_hpf_extra` and `video_lpf_extra`. Taken literally Video8's
fractional bandwidth would be 350 and the comparison would be worthless.

The band used throughout is therefore **spec-derived and identical in
construction for every format**: from the format's own colour-under
carrier, the lowest component the tape carries, to the peak-white carrier
plus the luma baseband, the first upper sideband. For VHS NTSC SP that is
0.629 to 7.4 MHz, a ratio of 11.76 against the decoder band-pass's 13.00,
and the effective count differs between them by 0.0001.

### 3.3 The result, with each format's own mechanics

Each format's own writing speed and track width from §2.2 in
`log_response`, `head_model.TYPICAL` otherwise:

| format | band (MHz) | ratio | **effective of 6** | condition | worst coherent pair |
|---|---|---|---|---|---|
| VHS NTSC SP | 0.629–7.40 | 11.76 | **1.566** | 4.8e4 | gap/azimuth 0.9995 |
| VHS PAL SP | 0.627–7.80 | 12.44 | **1.576** | 1.5e4 | gap/azimuth 0.9984 |
| VHS NTSC EP | 0.629–7.40 | 11.76 | **1.566** | 5.3e4 | gap/azimuth 0.9995 |
| VHS-C NTSC SP | 0.629–7.40 | 11.76 | **1.566** | 4.8e4 | gap/azimuth 0.9995 |
| S-VHS NTSC | 0.629–12.00 | 19.07 | **1.608** | 3.7e3 | spacing/thickness 0.9941 |
| S-VHS PAL | 0.627–12.00 | 19.14 | **1.657** | 1.4e3 | spacing/thickness 0.9913 |
| Betamax NTSC βII | 0.688–8.00 | 11.62 | **1.564** | 8.4e4 | gap/azimuth 0.9997 |
| Betamax PAL βII | 0.688–8.40 | 12.22 | **1.571** | 2.8e4 | gap/azimuth 0.9991 |
| SuperBeta NTSC | 0.688–9.35 | 13.58 | **1.572** | 3.7e4 | gap/azimuth 0.9993 |
| Video8 NTSC | 0.743–8.40 | 11.30 | **1.608** | 3.1e3 | spacing/thickness 0.9930 |
| Video8 PAL | 0.732–8.40 | 11.47 | **1.747** | 8.9e2 | spacing/thickness 0.9897 |
| Hi8 NTSC | 0.743–12.70 | 17.08 | **2.244** | 3.2e2 | spacing/thickness 0.9836 |
| Hi8 PAL | 0.732–12.70 | 17.34 | **2.254** | 1.9e2 | spacing/thickness 0.9771 |

**The 8 mm rows are an artefact of the head, not a property of the format,
and must not be read as a finding.** `head_model.TYPICAL` holds a 0.30 µm
gap — a VHS-class construction figure, described at `head_model.py:115-125`
as "typical construction figures for a head of this class". The gap term's
sinc null sits at `λ = 1.11 g`, that is at `f = v / 1.11 g`, which for VHS's
5.80 m/s is 17.4 MHz and comfortably outside its band, and for 8 mm's
3.75 m/s is 11.3 MHz — *inside* Hi8's. A head whose gap is that long
relative to the wavelength cannot record the format's own top frequency at
all, so nobody built one for an 8 mm machine. Those two rows measure a VHS
head bolted onto an 8 mm transport. (Betamax escapes it: its published
0.4 µm gap at 6.9 m/s puts the null at 15.5 MHz, twice its band top.)

### 3.4 The corrected comparison: a head built for the format

Scaling every length in the model — spacing, gap, thickness, contour — with
the format's own shortest recorded wavelength `λ_min = v / f_top`, so
`length / λ_min` is held at its VHS value:

| format | ratio | scale on TYPICAL | **effective of 6** |
|---|---|---|---|
| Video8 NTSC | 11.30 | 0.570 | **1.5651** |
| Video8 PAL | 11.47 | 0.474 | **1.5658** |
| Betamax NTSC βII | 11.62 | 1.101 | **1.5659** |
| VHS NTSC EP | 11.76 | 1.000 | **1.5662** |
| VHS NTSC SP | 11.76 | 1.000 | **1.5664** |
| VHS-C NTSC SP | 11.76 | 1.000 | **1.5664** |
| Betamax PAL βII | 12.22 | 0.886 | **1.5674** |
| VHS PAL SP | 12.44 | 0.793 | **1.5681** |
| SuperBeta NTSC | 13.58 | 0.942 | **1.5703** |
| Hi8 NTSC | 17.08 | 0.377 | **1.5757** |
| Hi8 PAL | 17.34 | 0.313 | **1.5767** |
| S-VHS NTSC | 19.07 | 0.617 | **1.5783** |
| S-VHS PAL | 19.14 | 0.516 | **1.5784** |

Sorted by band ratio, and monotone in it. The coherence structure is
unchanged in kind across the whole table: gap and azimuth 0.9995–0.9997,
spacing and thickness 0.998, contour orthogonal at 0.000–0.001, gain
against spacing 0.887–0.899.

### 3.5 Is it all fractional bandwidth? Yes, to four decimal places

Fitting the effective count against the log of the fractional bandwidth
across all thirteen rows:

| ensemble | fit | R² | largest residual |
|---|---|---|---|
| head built for the format | `eff = 0.0581 log₁₀(ratio) + 1.5042` | **0.998** | **0.0005** |
| VHS-class head throughout | `eff = 1.465 log₁₀(ratio) + 0.040` | 0.279 | 0.400 |

**Fractional bandwidth explains the entire variation** once the head is the
format's own: the largest departure from the line is five ten-thousandths,
against a spread of 0.013 across the table. The second row is §3.3's
artefact refusing to fit, which is the right behaviour for rows that are
measuring an assumption rather than a format.

The two remaining mechanical variables were swept directly, holding the
band at VHS NTSC's 0.5–6.5 MHz, and behave exactly as
`docs/BROADCAST_VTR_COLLAPSE.md` §3.3 found for U-matic:

| track width | 19.3 | 24.5 | 29.0 | 32.8 | 34.4 | 49.0 | 58.0 | 85.0 | 137 µm |
|---|---|---|---|---|---|---|---|---|---|
| effective of 6 | 1.5663 | 1.5663 | 1.5663 | 1.5663 | 1.5663 | 1.5664 | 1.5665 | 1.5667 | 1.5674 |

A sevenfold range of track width moves the answer by 0.07%, because width
enters `log_response` in exactly one place, as the product
`track_width_m · tan(azimuth_error)`, and is therefore algebraically
degenerate with the angle.

| writing speed | 3.12 | 3.75 | 4.85 | 5.80 | 6.90 | 10.26 m/s |
|---|---|---|---|---|---|---|
| effective of 6 | 1.600 | 1.582 | 1.571 | 1.566 | 1.564 | 1.561 |

Speed moves it by 2.5% over the whole consumer range — and *downwards* with
increasing speed, the same wrong-way, second-order effect the U-matic
document measured, for the same reason: raising `v` is a similarity
transformation on the frequency axis, so it rescales all six signatures at
once and preserves the relationships between them.

### 3.6 The interference set, for comparison

`interference.distinguishable` — the twelve modelled types, with
`real_parameters=True` as `interference.py:33-40` requires — on each
format's own band:

| band | effective of 12 | condition |
|---|---|---|
| VHS NTSC SP, 0.5–6.5 MHz (the published row's band) | 6.90 | 6.9 |
| VHS NTSC SP, 0.629–7.4 MHz | 6.84 | 9.1 |
| VHS PAL SP | 6.63 | 10.9 |
| S-VHS NTSC | 6.82 | 9.6 |

Against the established 6.89 of 12. **The interference set's count is a
property of the modelled set and of the echo spacing relative to the band,
not of the format** — it moves by 4% across formats where the magnetics
move by 1%. That is the propagation document's point seen from the consumer
side: an echo family is a Fourier basis over delay and stays
well-conditioned whatever the tape underneath it is doing.


## 4. Azimuth, which is the interesting axis

### 4.1 The angle and the width are one variable, and it is a product

`head_model.log_response:255-259` computes the azimuth term as

    across  = track_width_m · tan(azimuth_error_degrees)
    loss    = sinc(across / λ),   λ = v / f

so the angle and the width never appear apart. A format cannot buy
identifiability with a larger azimuth angle in any way a wider track would
not equally buy, and the two are exactly exchangeable. The natural format
constant is therefore the **time** `w / v`, which multiplied by `tan(Δθ)`
and by a frequency gives the sinc argument.

Sweeping the product on the VHS band, everything else held:

| azimuth error | 0.01° | 0.1° | 10′ | 0.5° | 1.0° | 2° | 5° | 20° |
|---|---|---|---|---|---|---|---|---|
| sinc argument at band top | 0.011 | 0.113 | 0.189 | 0.567 | 1.135 | 2.270 | 5.687 | 23.66 |
| effective of 6 | 1.5663 | 1.5665 | 1.5668 | 1.5722 | **1.6396** | 1.6090 | 1.5928 | 1.5816 |
| azimuth vs gap coherence | 0.9997 | 0.9997 | 0.9998 | 1.0000 | **0.9431** | 0.8654 | 0.8213 | 0.8164 |

**A larger product improves identifiability, and only once the sinc's first
null lands inside the band.** The optimum is at an argument of about 1.1 —
the null just inside the top — and it is worth 0.073 of effective count,
about 5%. Below that the sinc sits in its small-argument regime, where
`1 − (πx)²/6` is a parabola indistinguishable from the leading term of a
Wallace exponential, which is exactly why gap and azimuth read 1.000.

Reaching an argument of 1 on VHS needs an azimuth error of about 1°, which
is **six times the format's entire ±10′ tolerance** (`head_model.py:69`;
JVC gives the azimuth as 6° ± 10′). No consumer format comes close: the
largest sinc argument in the whole table below is 0.41.

### 4.2 The sinc is even, so a symmetric azimuth error is not a head difference

`head_model`'s docstring says azimuth is "THE term that separates two VHS
heads: the format records adjacent tracks at OPPOSITE azimuth, so a
systematic azimuth error cannot affect both heads alike". The first half is
the motivation; the second needs a qualification, and the model itself
supplies it.

`sinc` is even, so `log_response` returns **bit-identical** output for
`azimuth_error_degrees = +0.5` and `−0.5` — checked, maximum difference
0.0. The ± alternation produces exactly that pair from a single physical
misalignment: a drum whose head pair sits at `+θ+δ` and `−θ−δ` reads tracks
written at `+θ` and `−θ` with errors `+δ` and `−δ`. **A symmetric azimuth
error therefore cancels out of the head difference in magnitude.** What
survives into the head difference is the two heads' *independent*
construction errors — real, since each head is bonded separately, but not
what the ± alternation is for.

This does not weaken azimuth as a per-head loss, and it does not touch
mistracking, which `tape_model.py:126-139` correctly identifies as the one
variation that moves the two heads in opposite directions — because there
the neighbour's opposite azimuth enters the *crosstalk*, not the wanted
signal.

### 4.3 The azimuth instrument, format by format

A common 10 arc-minute error applied to every format, so what is compared
is geometry and not manufacture. `x at top` is the sinc argument at the top
of the band; `RF peak` is the largest head difference `head_difference_log`
produces across the band; `video RMS` is the same difference through
`predicted_video_effect` at the format's own blanking carrier, which is the
quantity `fit_head_difference` actually works on.

| format | w/v (µs) | x at top | RF peak (dB) | video RMS (nepers) | **vs VHS** |
|---|---|---|---|---|---|
| Hi8 PAL | **11.026** | 0.4073 | 2.515 | 2.08e−2 | **3.61** |
| S-VHS PAL | 10.103 | 0.3527 | 1.856 | 1.71e−2 | **2.96** |
| S-VHS NTSC | 10.000 | 0.3491 | 1.817 | 1.67e−2 | **2.89** |
| Video8 PAL | **11.026** | 0.2694 | 1.063 | 7.16e−3 | **1.24** |
| VHS PAL SP | 10.103 | 0.2292 | 0.764 | 5.93e−3 | 1.03 |
| **VHS NTSC SP** | **10.000** | **0.2153** | **0.672** | **5.77e−3** | **1.00** |
| VHS-C NTSC SP | 10.000 | 0.2153 | 0.672 | 5.77e−3 | 1.00 |
| Betamax NTSC βI | 8.696 | 0.2024 | 0.593 | 4.95e−3 | 0.86 |
| Hi8 NTSC | 5.467 | 0.2020 | 0.591 | 4.77e−3 | 0.83 |
| Betamax PAL βII | 5.624 | 0.1374 | 0.272 | 2.05e−3 | 0.35 |
| Video8 NTSC | 5.467 | 0.1336 | 0.256 | 1.70e−3 | 0.29 |
| SuperBeta NTSC | 4.348 | 0.1183 | 0.201 | 1.22e−3 | 0.21 |
| Betamax NTSC βII | 4.348 | 0.1012 | 0.147 | 1.22e−3 | 0.21 |
| VHS NTSC EP | 3.328 | 0.0716 | 0.073 | 6.25e−4 | 0.11 |
| Betamax NTSC βIII | 2.826 | 0.0658 | 0.062 | 5.13e−4 | 0.09 |

**No consumer format's azimuth-times-width geometry gives a materially
better head-difference instrument than VHS's.** The best geometry in the
table is PAL 8 mm's, at `w/v` = 11.03 µs against VHS NTSC's 10.00 — ten per
cent, which at Video8's own band top comes to 25% in the sinc argument and
1.24x in the video domain. That is real and it is not material: VHS's own
speed range moves the same quantity by 3.0x from SP to EP and its own two
systems differ by 1.03x, so a ten per cent difference between formats sits
inside the variation a single format already contains.

Everything else is worse than VHS. Betamax is a fifth to a third, because
its higher writing speed shortens the wavelength faster than its narrower
track shortens the path across it — even Beta I, whose 60 µm track is wider
than VHS's 58, comes to 0.86x. NTSC 8 mm is 0.29x, its PAL sibling four
times better than it purely on tape speed.

The three rows that clearly beat VHS beat it by 2.9 to 3.6x, and all three
are high-band variants of transports already in the table. **Azimuth
geometry does not move this quantity. The carrier does.**

### 4.4 What the nominal angle is actually for

The nominal ±θ never enters the wanted signal — a head is nominally square
to its own track. It enters the *neighbour's*, which is written 2θ away, so
crosstalk is suppressed by `sinc(w·tan 2θ / λ)`:

| format | 2θ | w·tan2θ/v (µs) | argument at colour-under | mean suppression across band |
|---|---|---|---|---|
| Video8 PAL / Hi8 PAL | 20° | 4.013 | 2.94 | 39.9 / 42.9 dB |
| VHS PAL SP | 12° | 2.147 | 1.35 | 33.5 dB |
| VHS NTSC SP, VHS-C | 12° | 2.126 | 1.34 | 33.1 dB |
| S-VHS NTSC / PAL | 12° | 2.126 / 2.147 | 1.34 / 1.35 | 37.0 / 36.8 dB |
| Video8 NTSC / Hi8 NTSC | 20° | 1.990 | 1.48 | 33.7 / 36.9 dB |
| Betamax PAL βII | 14° | 1.402 | 0.96 | 30.6 dB |
| Betamax NTSC βII | 14° | 1.084 | 0.75 | 28.0 dB |
| VHS NTSC EP | 12° | 0.707 | 0.45 | 23.8 dB |

**The formats have chosen their azimuth angles to hold the colour-under
crosstalk roughly constant, not to make the luma band more informative.**
Video8's ±10° against VHS's ±6° is compensation for a track a third as wide
read at two-thirds the speed; NTSC Video8 and NTSC VHS land within ten per
cent of each other on the sinc argument at their own chroma carriers (1.48
against 1.34) and within 0.6 dB on the mean suppression. Betamax's ±7°
lands a little lower, at 0.75–0.96 and 28–31 dB. So the three different
angles in the brief are not three different design choices about the luma;
they are one design choice about the chroma, expressed in three geometries.

The single largest azimuth effect in this whole document is not between
formats at all. It is **VHS EP**, whose 19.3 µm track drops the crosstalk
suppression to 23.8 dB and the azimuth instrument to 0.11x — the same
mechanism `chroma.py:34-48` already compensates for with its track-width
trust weight, and the reason that weight "needs the correction held back so
much harder" at the slow speeds.


## 5. Hi8 and S-VHS: the controlled experiment

Same tape, same heads, same transport, same speed; a higher carrier and a
wider deviation. That isolates bandwidth from every mechanical variable at
once, which is what §3.5 otherwise had to get by scaling assumption.

| | VHS NTSC SP → S-VHS NTSC | Video8 NTSC → Hi8 NTSC |
|---|---|---|
| sync / white | 3.4 / 4.4 → **5.4 / 7.0** MHz | 4.2 / 5.4 → **5.7 / 7.7** MHz |
| deviation | 1.0 → **1.6** MHz | 1.2 → **2.0** MHz |
| luma baseband | 3.0 → **5.0** MHz | 3.0 → **5.0** MHz |
| colour-under | 629.371 kHz, **unchanged** | 743.444 kHz, **unchanged** |
| writing speed, track width, azimuth | **unchanged** | **unchanged** |
| tape coercivity | 50 → **70** × 10³ A/m | metal particle → MP or evaporated |
| band ratio | 11.76 → **19.07** | 11.30 → **17.08** |
| effective of 6 (head built for the format) | 1.5664 → **1.5783** (+0.8%) | 1.5651 → **1.5757** (+0.7%) |
| azimuth argument at band top | 0.215 → **0.349** (1.62x) | 0.134 → **0.202** (1.51x) |
| video-domain head difference | 5.77e−3 → **1.67e−2** (2.89x) | 1.70e−3 → **4.77e−3** (2.81x) |
| `carson_bandwidth` | 8.00 → **13.20** MHz | 8.40 → **14.00** MHz |
| capture C/N in band, 8 bit 50 MSps | 46.70 → **44.52** dB | 46.48 → **44.27** dB |
| capture capacity | 124.1 → **195.2** Mbit/s | 129.7 → **205.9** Mbit/s |
| tape capacity at 0.697 dB spread | 57.4 → **94.7** Mbit/s | 60.2 → **100.4** Mbit/s |
| **converter over tape** | 2.16x → **2.06x** | 2.15x → **2.05x** |

Three things this settles.

**Raising the carrier buys almost nothing in identifiability** — 0.7 to
0.8% of effective count, and every bit of it is the band ratio moving,
which §3.5 already predicts from the ratio alone. Both rows sit on the
fitted line to within 0.0005.

**Raising the carrier buys a great deal in the head-difference
instrument** — 1.5 to 1.6x at RF and 2.8 to 2.9x in the video domain, for
no geometric change whatever. That is the largest single lever in this
document, and it is available on any S-VHS or Hi8 material already held.

**Raising the carrier moves `binding_limit` toward the tape.** The
converter's own in-band C/N *falls* by 2.2 dB, because quantisation is flat
to Nyquist and a wider band admits more of it, while its capacity rises
because the band grows faster than the C/N falls; the tape's capacity rises
strictly in proportion to the band. So the converter's margin narrows from
2.16x to 2.06x. The high-band formats are where better capture hardware
would come closest to being worth something, and it is still not close.


## 6. Which link binds, per format

`capture_profile.carson_bandwidth` on each format's own deviation and
baseband, and `capture_profile.binding_limit` with the capture profile the
parent document's real run used — 8 bit, 117 of 256 codes,
`signal_rms_codes` 24.96, recovered by inverting the 46.7 dB that document
reports. The VHS row reproduces it exactly (tape 21.56 dB, capture
46.70 dB, 57.4 and 124.1 Mbit/s, ratio 2.16x).

| format | deviation | luma baseband | **Carson band** | capture C/N 40 MSps | capture C/N 50 MSps | capture capacity 50 | tape capacity | binds |
|---|---|---|---|---|---|---|---|---|
| VHS all speeds, VHS-C | 1.0 | 3.0 | **8.00 MHz** | 45.73 dB | 46.70 dB | 124.1 | 57.4 | tape |
| Video8 | 1.2 | 3.0 | **8.40 MHz** | 45.51 | 46.48 | 129.7 | 60.2 | tape |
| Betamax NTSC | 1.2 | 3.2 | **8.80 MHz** | 45.31 | 46.28 | 135.3 | 63.1 | tape |
| Betamax PAL, SuperBeta | 1.4 | 3.2 | **9.20 MHz** | 45.12 | 46.09 | 140.9 | 66.0 | tape |
| S-VHS | 1.6 | 5.0 | **13.20 MHz** | 43.55 | 44.52 | 195.2 | 94.7 | tape |
| Hi8 | 2.0 | 5.0 | **14.00 MHz** | 43.30 | 44.27 | 205.9 | 100.4 | tape |

(Capacities in Mbit/s. The tape column is Shannon over the same band at the
0.697 dB envelope spread measured on the real VHS decode, carried across as
the only tape-noise figure this arc has measured. A real S-VHS or Hi8 tape
has the higher coercivity of §5 and would do better, which only widens the
margin.)

**The tape binds on every consumer format.** The two capacities are Shannon
over the same band, so they cross exactly where the two carrier-to-noise
ratios are equal — which makes the crossover the capture's own in-band C/N:

| format | the converter takes over above | in envelope spread |
|---|---|---|
| VHS, VHS-C | 45.73 dB at 40 MSps / **46.70 dB** at 50 MSps | 0.0401 dB |
| Video8 | 45.51 / **46.48** dB | 0.0411 dB |
| Betamax NTSC | 45.31 / **46.28** dB | 0.0420 dB |
| Betamax PAL, SuperBeta | 45.12 / **46.09** dB | 0.0430 dB |
| S-VHS | 43.55 / **44.52** dB | 0.0515 dB |
| Hi8 | 43.30 / **44.27** dB | 0.0530 dB |

Against a measured 21.6 dB and a measured 0.697 dB spread. **The tape would
have to be twenty-three to twenty-five decibels quieter — a within-field
envelope spread thirteen to seventeen times smaller — before eight bits at
forty or fifty megasamples became the limiting element**, and the high-band
formats need it least, because their wider Carson band lowers the
converter's own in-band C/N by more than two decibels.


## 7. What this repository would need

None of this is a defect in the decoder: `head_model.mechanics_for` is
called only from `tests/unit/test_head_model.py`, so the models are offline
apparatus and the decode path is untouched by items 1, 2 and 4 to 9. Item 3
is the exception and does reach a decode.

**1. `head_model.FORMAT_MECHANICS` holds VHS SP and nothing else.**
`head_model.py:53-84` has exactly `("VHS","NTSC","SP")` and
`("VHS","PAL","SP")`, so `mechanics_for` (`:188-197`) returns `None` for
Betamax, Video8, Hi8, S-VHS and every VHS speed but SP, and every
head-model fit in this arc is unavailable on them. Entries are needed for
at least `(BETAMAX, NTSC|PAL, BII)`, `(VIDEO8, NTSC|PAL, SP)`,
`(HI8, NTSC|PAL, SP)`, `(SVHS, NTSC|PAL, SP)` and the VHS LP and EP speeds,
each carrying `drum_diameter_m`, `drum_revolutions_per_second`,
`linear_tape_speed_m_s`, `writing_speed_m_s`, `track_width_m`,
`azimuth_degrees` and `azimuth_tolerance_degrees`. §2.2 has every figure.
Betamax's NTSC writing speed should go in with the 6.9-against-6.993 note
of §2.2 attached, exactly as the VHS entry already records that its
5.80 m/s is stated rather than derived.

**2. The key's tape-speed field is a string where the decoder's is an
index.** `mechanics_for` keys on `"SP"` while `formats.TAPE_SPEEDS`
(`vhsdecode/formats.py:32`) maps `sp/lp/ep/slp/vp` to `0..3`, and every
format definition indexes on the integer. Any entry added has to agree with
one of the two conventions.

**3. `video_track_width` exists only for VHS, and the fallback reaches a
decode.** `vhs.py:152` and `:302` are the only two occurrences in the tree,
so `chroma.py:591` falls back to `REFERENCE_TRACK_WIDTH = 58.0`
(`chroma.py:49`) for Betamax, Video8, Hi8 and the rest — VHS SP's width
used as Video8 NTSC's, where the published figure is 20.5 µm. It feeds the
chroma envelope correction's trust weight `w/(w + 18.29)`
(`chroma.py:34-48`) directly: at 58 µm that weight is 0.760 and at 20.5 µm
it is 0.528, so on an NTSC Video8 decode the correction is currently
applied at 1.44 times the confidence the format's own geometry supports.
PAL Video8 at 34.4 µm gives 0.653, still 1.16 times over. Adding
`video_track_width` to `video8.py` and `betamax.py` — 20.5/34.4 µm for
8 mm SP and 30/32.8 µm for Betamax βII, per §2.2 — is the one change on
this list that would alter output.

**4. `transport_model` is VHS's transport.** `transport_model.py:44-69`
hard-codes a 62 mm drum, and `:73-77` sets `FIELDS_PER_DRUM_REVOLUTION = 2`
with `drum_rate_hz = field_rate / 2`. Betamax needs 74.487 mm and 8 mm
needs 40 mm, both of which keep two fields per revolution; **VHS-C does
not** — four heads at 45 rps against 60 fields/s is 4/3 fields per
revolution, and the drum band sits at 45 Hz rather than 29.97 Hz. The
capstan, guide and reel diameters are already declared typical rather than
certain, which is the right treatment; the drum is declared
`"certain": True` on the strength of the format fixing it, and for VHS-C
that is only true once the format is known to be VHS-C.

**5. The same assumption is in the runtime, in prose.**
`carrier_tbc.py:534-544` states that "a two-head helical drum lays one
field per head pass, so it revolves once per FRAME and the flutter
fundamental's period is `frame_lines` lines", and derives its smoothing
window as `frame_lines/(2π)` from exactly that (`carrier_tbc.py:114-117`,
`:543`). `head_switch.py:404-409` reasons the same way to place the switch
line. On a VHS-C camcorder recording the drum turns 1.5 times per frame, so
both the flutter fundamental and the switch cadence are 1.5x what the
comment assumes. Playback is through a full-size 62 mm two-head drum, so
the *playback* side is right; it is the *recording* machine's transport
signature that lands in the wrong place.

**6. `capture_profile.binding_limit` defaults to VHS's FM figures.**
`capture_profile.py:48-49` defines `NTSC_VHS_SP_DEVIATION_HZ = 1.0e6` and
`NTSC_VHS_SP_BASEBAND_HZ = 3.0e6`, and `:148-149` makes them the default
arguments. A caller that omits them for Hi8 gets an 8.00 MHz Carson band
where 14.00 is right — 43% wrong in the band and therefore in both
capacities — with nothing to signal it. The constants are correctly named
after their format; the defaulting is what is unsafe.

**7. `interference.signatures` defaults are VHS's too.**
`interference.py:239-241`: `writing_speed_m_s` 5.8, `track_width_m` 58e−6,
`recording_depth_m` 0.35e−6. Only the particle-noise entry uses them and
only for its level, so nothing in §3.6 turns on it, but the same
observation as (6) applies.

**8. `tape_model`'s magnetics are VHS's gamma-ferric oxide.**
`tape_model.py:233-234` sets `PARTICLE_VOLUME_M3 = 1e-21` and
`PACKING_FRACTION = 0.4`, and `:70-71` gives coercivity as "600 oersted
class" from JVC. SMPTE 32M gives **50 × 10³ A/m (≈628 Oe) for basic VHS and
70 × 10³ A/m (≈880 Oe) for S-VHS** (§3.2.1.4, §7.3.3) — and the JVC guide's
own S-VHS cassette clause repeats its VHS wording verbatim, which is a
copy-paste in the guide rather than a real equality. 8 mm is metal-particle
or metal-evaporated, with a different particle volume and packing again;
no coercivity figure for it was found. `magnetic_limits` and the 40.8 dB
particle-noise prediction of `docs/ELLIPTICAL_COLLAPSE.md` §7.6 are
VHS-specific as they stand.

**9. `head_model.TYPICAL` is a VHS-class head.** `head_model.py:115-125`:
gap 0.30 µm, thickness 0.20 µm, spacing 0.05 µm, contour length 200 µm.
§3.3 shows what happens when they are carried onto an 8 mm transport
unchanged — the gap null enters the band and the effective count reads 2.24
where the format's own head gives 1.58. Either per-format `TYPICAL` entries
or the `λ_min` scaling of §3.4 is needed before any cross-format comparison
is meaningful. One real per-format figure exists to seed it: Betamax's
published head gap of 0.4 µm.

**10. Smaller things.** `betamax.py` defines no `deemph_q`, so
`filter_model.assumed_parameters` (`filter_model.py:49-57`) supplies 0.5 by
default rather than a format figure. `video8.py:106-108` carries Hi8's
`deemph_mid` with an explicit "Not correct, needs to be fixed properly".
`video8.py:50` and `:123-124` carry band-pass edges that are placeholders
rather than band edges, which §3.2 had to work around. `vhs.py:312` sets
`luma_carrier` for NTSC VHS alone and nothing in the tree reads it. The two
Betamax FM triples of §2.1 disagree with the published carriers. And
**VHS-C is not in the supported format set** (`vhsdecode/main.py:38-60`),
so there is nowhere to hang a VHS-C transport even once one exists.


## 8. Verdict

> **Every consumer helical format returns 1.565 to 1.578 effectively
> distinguishable mechanisms of 6, and the whole 0.013 of spread is
> fractional bandwidth — R² = 0.998, largest residual 0.0005. Betamax's
> larger drum and faster writing speed, 8 mm's smaller tape and slower one,
> VHS-C's four-head 41 mm drum, and S-VHS's and Hi8's raised carriers all
> leave the answer exactly where VHS's 1.58 left it.**

> **No consumer format's azimuth geometry gives a materially better
> head-difference instrument than VHS's. PAL 8 mm is the best of them, at
> `w/v` = 11.03 µs against VHS's 10.00 — ten per cent, which is less than
> the spread within VHS's own speed range. Betamax is a fifth to a third of
> VHS, NTSC 8 mm under a third. The only large gain, 2.9 to 3.6x, comes
> from S-VHS's and Hi8's higher carriers, which is bandwidth again and not
> geometry.**

The U-matic document's conclusion therefore holds across the whole consumer
family, and for a second reason. It found writing speed and track width
inert because both are similarity transformations that leave the
dimensionless group `d/λ` alone. The consumer formats add the direct
demonstration: hold `d/λ` fixed by building each head for its own format,
and thirteen different transports collapse onto a single curve in
fractional bandwidth to four decimal places. **The identifiability ceiling
of `docs/ELLIPTICAL_COLLAPSE.md` §7.4 is a property of magnetic recording,
not of VHS.** Separating spacing from gap from azimuth still needs a
witness that is not a frequency response, and no consumer format provides
one.

What the consumer formats *do* provide, which U-matic did not, is a pair of
controlled experiments on the one axis that moves: S-VHS and Hi8 raise the
carrier on an unchanged transport, and they triple the head-difference
instrument while leaving identifiability where it was. If this arc is ever
run on a second format, those are the two worth the effort.


## 9. What produced this

Read-only on the repository throughout. The scripts live in the `consumer/`
directory of this session's scratchpad and each calls the repository's own
functions rather than reimplementing them:

| script | what it does |
|---|---|
| `repo_constants.py` | calls `formats.get_format_params` for every format and prints the FM triple, colour-under, bands and track width |
| `identifiability.py` | the §3.1 perturbation through `head_model.log_response` and `information_extrapolation.ellipsoid` |
| `scaled.py` | the same with every length scaled to the format's `λ_min` (§3.4) |
| `azimuth.py` | the sinc argument, the self-track loss and the crosstalk suppression (§4) |
| `instrument.py` | `head_difference_log` and `predicted_video_effect` as the head-difference instrument (§4.3) |
| `binding.py` | `capture_profile.binding_limit` on the parent document's capture profile (§6) |
| `final.py` | the mechanics table and every table above |

### Sources outside the repository

- [decode-orc/analogue-video-specifications](https://github.com/decode-orc/analogue-video-specifications) — [SMPTE 32M-1998/2004](https://github.com/decode-orc/analogue-video-specifications/blob/main/docs/vhs/SMPTE-32M-2004/markdown.md) (§3 basic type H, §4 LP/EP, §6 compact, §7 high-performance = S-VHS, Annex A HQ, Annex B adaptor; 525-line only) and the [JVC Video Technical Guide VTG82063](https://github.com/decode-orc/analogue-video-specifications/tree/main/docs/vhs/JVC-Video-Technical-Guide-VTG82063) (§1.1.19 S-VHS luma, §1.2 S-VHS technology, §3.3.10 compact VHS, table 1-1-1). Primary.
- Sony patents [US 6,674,961 B1](https://patents.google.com/patent/US6674961B1/en) (FIG. 6/7, the 8 mm specification tables), US 4,907,102, US 5,488,482, US 6,038,098 — 8 mm mechanics, no guard band, ±10° azimuth.
- [Sony CCD-V800E service manual](https://ia800609.us.archive.org/11/items/manual_CCDV800E_SM_SONY/CCDV800E_SM_SONY_text.pdf) — Video8 and Hi8 FM carriers, PAL colour-under, AFM audio.
- *Television* (UK), [August 1983](https://worldradiohistory.com/UK/Practical/Television/80s/Television-Servicing-UK-1983-08.pdf) (Trundle, Betamax part 1: track width, azimuth, head gap, recording zone), [September 1983](https://worldradiohistory.com/UK/Practical/Television/80s/Television-Servicing-UK-1983-09.pdf) (part 2), [June 1980](https://worldradiohistory.com/UK/Practical/Television/80s/Television-Servicing-UK-1980-06.pdf) (Beeching, the two PAL Betamax colour-under carriers — the reference `betamax.py:26-28` already cites), [July 1984](https://worldradiohistory.com/UK/Practical/Television/80s/Television-Servicing-UK-1984-07.pdf) (the Sony spectrum photograph labelling sync tip 3.8 MHz).
- [Sencore Tech Tip TT189](https://docs.ampnuts.ru/eevblog.docs/Sencore/Sencore_Tech_Tips/TT189p%20-%20Comparison%20Of%20VCR%20Formats.pdf) — the VCR format comparison table, Betamax and SuperBeta carriers.
- Goodman, *Maintaining & Repairing Videocassette Recorders* (1983), [pp. 16–18](https://www.worldradiohistory.com/BOOKSHELF-ARH/Technology/Technology-General/Maintaining-&-Repairing-Videocassete-Recorders-Goodman-1983.pdf) — Betamax NTSC track pitches and the negative guard band.
- [Betamax PALsite format page](https://www.palsite.com/format.html) (via the Wayback Machine; the live site returns HTTP 522) — drum diameter, PAL writing speed, track angles.
- [Wikipedia: Betamax](https://en.wikipedia.org/wiki/Betamax), [Japanese Wikipedia: ベータマックス](https://ja.wikipedia.org/wiki/%E3%83%99%E3%83%BC%E3%82%BF%E3%83%9E%E3%83%83%E3%82%AF%E3%82%B9), [VHS-C](https://en.wikipedia.org/wiki/VHS-C), [S-VHS](https://en.wikipedia.org/wiki/S-VHS) — secondary, used only where nothing better was found and labelled as such.
