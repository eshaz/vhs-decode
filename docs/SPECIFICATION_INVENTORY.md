# The analogue video specifications collection, surveyed for measurement

A survey of <https://github.com/decode-orc/analogue-video-specifications>
against what this repository measures. The question throughout is the one
`docs/ELLIPTICAL_COLLAPSE.md` asks: **what does a document fix exactly, so
that it can serve as the SYNTHETIC side of a measured component?**

The collection was cloned and read in full. Its `mkdocs.yml` `nav` is the
authoritative index and every document named there exists on disk. Paths
below are relative to the repository root; each is fetchable at
`https://raw.githubusercontent.com/decode-orc/analogue-video-specifications/main/<path>`.

Three areas are covered by other agents and are noted here only for
completeness: ITU-R BT.1700, the broadcast VTR and U-matic material, and
television propagation.


## 1. What is in the collection

Forty-three Markdown documents in six branches — `efm`, `laserdisc`,
`teletext`, `vhs`, `video_formats`, `video_metadata` — plus **eleven YAML test-signal
definitions under `resources/definitions/vits/`** that are the machine-readable
source for the VITS pages. The YAML is the most valuable single artefact in
the collection for this project and is treated separately in section 3.

### 1.1 VHS — the two documents that matter most here

| path | document | what it fixes |
|---|---|---|
| `docs/vhs/SMPTE-32M-2004/markdown.md` | SMPTE 32M-2004, *Video Recording — 1/2-in Type H — Cassette, Tape and Records* | The whole VHS format as a normative standard: tape speed, writing speed, drum diameter, track geometry, azimuth, coercivity, tape tension, head-switch position, FM carrier and deviation, clipping levels, chroma downconversion, and the **level-dependent sub-preemphasis response tables** for LP/EP and for the high-performance (S-VHS) system |
| `docs/vhs/JVC-Video-Technical-Guide-VTG82063/section-1.md` | JVC Video Technical Guide VTG82063 §1, *VHS Basics Technology* | The same ground for PAL, SECAM, M-PAL, N-PAL and S-VHS, which SMPTE 32M does not cover; plus cassette tape coercivity, thickness and reel dimensions |
| `.../section-2.md` | §2, Mechanism Description | Deck mechanics qualitatively: loading systems, reel/capstan/drum motors, tension arms. **No dimensions** |
| `.../section-3.md` | §3, Video Circuit | The luma and chroma processing chains, AGC, dropout compensation, non-linear emphasis, comb filtering. Mostly block diagrams; few numbers |
| `.../section-4.md` | §4, Servo Circuit | Drum and capstan servo: reference rates, FG pulse rates, phase comparison |
| `.../section-5.md` | §5, Microprocessor Systems | System control. Nothing measurable |
| `.../section-6.md` | §6, Tuner Circuit | NTSC/PAL/SECAM IF chains, IF carrier frequencies, NICAM, VPS |
| `.../section-7.md` | §7, Audio Circuit | VHS Hi-Fi depth multiplex. Carries the **recorded wavelength, recording depth and head gap** figures |
| `.../section-8.md` | §8, Appendix | Glossary and abbreviations |

### 1.2 Television and video formats

| path | document | what it fixes |
|---|---|---|
| `docs/video_formats/BT-1700-E/BT-1700-E.md` | ITU-R BT.1700 | *Covered by another agent* |
| `docs/video_formats/BT-470-6-1998/BT-470-6-1998.md` | ITU-R BT.470-6, *Conventional television systems* | Line counts, field rates, timing, bandwidths and levels for every 525- and 625-line system, NTSC/PAL/SECAM variants |
| `docs/video_formats/BT-624-4-1990/BT-624-4-1990.md` | CCIR Report 624-4 | The same ground as BT.470 in report form, with the historical per-country tabulation |
| `docs/video_formats/BT-601-5-1995/BT-601-5-1995.md` | ITU-R BT.601-5 | Digital component studio sampling: 13.5 MHz luma, 6.75 MHz chroma, 8/10-bit levels, 4:3 and 16:9 |
| `docs/video_formats/SMPTE-170M-2004/SMPTE-170M-2004.md` | SMPTE 170M-2004, *Composite Analog Video Signal — NTSC for Studio Applications* | The normative NTSC studio signal: sync geometry, levels, subcarrier, colorimetry, interface impedance |
| `docs/video_formats/SMPTE-244M-2003/SMPTE-244M-2003.md` | SMPTE 244M-2003 | 4f_sc composite sampling: sampling phase relative to subcarrier, and the **quantisation scale in 8- and 10-bit** |
| `docs/video_formats/SMPTE-272M-1994/SMPTE-272M-1994.md` | SMPTE 272M-1994 | AES/EBU audio embedded in ancillary data space. Not relevant here |
| `docs/video_formats/EBU-Tech-3213-E/EBU-Tech-3213-E.md` | EBU Tech. 3213-E | Studio monitor phosphor chromaticity and tolerances |
| `docs/video_formats/EBU-Tech-3280-E/EBU-Tech-3280-E.md` | EBU Tech. 3280-E | Parallel and serial interfaces for 625-line digital PAL |

### 1.3 Video metadata — including the test-signal standards

| path | document | what it fixes |
|---|---|---|
| `docs/video_metadata/VITS/index.md` | Vertical Interval Test Signals overview | Points at the two definition pages and the YAML source |
| `docs/video_metadata/VITS/NTSC-VITS.md` | NTSC VITS definitions | Five NTSC insertion test signals, element by element — see section 3 |
| `docs/video_metadata/VITS/PAL-VITS.md` | PAL VITS definitions | Six PAL insertion test signals, element by element — see section 3 |
| `docs/video_metadata/T-REC-J-63-1990/T-REC-J-63-1990.md` | ITU-T J.63 (1990) | **The normative source** for insertion test signals. Annex I is 625-line (lines 17, 18, 330, 331); Annex II is 525-line (line 17 of each field). Gives element amplitudes, pulse half-amplitude durations and the characteristic timing instants, plus the list of measurements each signal supports |
| `docs/video_metadata/EBU-Tech-3209/EBU-Tech-3209.md` | EBU Tech. 3209 | Performance requirements for the equipment that *generates and inserts* these signals: §7.2 specifies each element (luminance bar, staircase, 2T pulse, 20T composite pulse, chrominance bar, three-level chrominance bar, chrominance reference, black-level stability, multiburst) with its tolerance |
| `docs/video_metadata/R-REC-BT-1439-1-2006/R-REC-BT-1439-1-2006.md` | ITU-R BT.1439-1 | The measurement vocabulary and method: waveform terminology, insertion gain, the four noise classes, non-linear distortion (luminance non-linearity, differential gain, differential phase, chrominance–luminance intermodulation), and linear distortion split by **time scale** — long-time, field-time, line-time and short-time waveform distortion — plus chrominance–luminance gain and delay inequality |
| `docs/video_metadata/IEC-60461-2010-Time-and-control-code/IEC-60461-2010-Time-and-control-code.md` | IEC 60461:2010 | LTC and VITC structure, flag bits, binary groups |
| `docs/video_metadata/ANSI-CTA-608-E-S-2019/ANSI-CTA-608-E-S-2019.md` | ANSI/CTA-608-E S-2019 | Line 21 closed captioning: waveform, data rate, packet formats, XDS |

### 1.4 Teletext and videotex

Six documents, all data-layer rather than signal-layer, but each fixes a
**data-line waveform** — bit rate, amplitude, and the VBI lines it occupies —
which is a measurable quantity on any tape carrying teletext.

| path | document |
|---|---|
| `docs/teletext/BT-653-3-1998/BT-653-3-1998.md` | ITU-R BT.653-3, *Teletext systems* — tabulates systems A, B, C and D for both 625/50 and 525/60, with bit rate, data amplitude and positioning for each |
| `docs/teletext/BBC-Broadcast-Teletext-1976/BBC-Broadcast-Teletext-1976.md` | Broadcast Teletext Specification 1976 (BBC/IBA/BREMA) — the original UK Ceefax/Oracle spec |
| `docs/teletext/ETSI-EN-300-706-2003/ETSI-EN-300-706-2003.md` | ETSI EN 300 706 V1.2.1, Enhanced Teletext (World System Teletext, System B) |
| `docs/teletext/CEA-516_S-2013/CEA-516_S-2013.md` | CEA-516 S-2013, NABTS — the 525-line North American system, 5.727272 Mbit/s NRZ |
| `docs/teletext/FIPS-121/FIPS-121.md` | FIPS PUB 121 / ANSI X3.110-1983, North American PLPS presentation syntax |
| `docs/teletext/ITU-T_T-101-1988-11/ITU-T_T-101-1988-11.md` | ITU-T T.101 (11/1988), international videotex interworking |

BT.653-3 is the one worth reading first, because it tabulates all four systems
side by side with the physical-layer numbers a decoder can be measured against:

| | System A | System B | System C | System D |
|---|---|---|---|---|
| bit rate, 625/50 | 6.203125 Mbit/s ± 0.005% | 6.9375 Mbit/s ± 25 × 10⁻⁶ | 5.734375 Mbit/s (367 × line frequency) | 5.6427875 Mbit/s (14/11 × f_sc) |
| bit rate, 525/60 | — | 5.727272 Mbit/s ± 25 × 10⁻⁶ | 5.727272 Mbit/s (364 × line frequency) | 5.727272 Mbit/s ± 3 × 10⁻⁶ (364 f_H; 8/5 f_sc) |
| data amplitude, 625/50 | D/S = 0 (± 3%); A/S = 7/3 (+0, −10%) positive modulation, 14/9 (−0, +6%) negative | black level ± 2%; 66% (± 6%) of black-to-white excursion | 0 IRE / 70 IRE negative modulation, 100 IRE positive | 0 ± 2.5 IRE / 70 ± 2.5 IRE |
| data amplitude, 525/60 | — | black level ± 2%; 70% (± 6%) of black-to-white | as above | as above |
| data shaping | sine squared | skew-symmetrical about half the bit rate | typically raised cosine, 100% roll-off, then a video low-pass | 100% cosine roll-off (525/60: controlled cosine, roll-off factor 0.6, cut-off 0.5 × bit rate) |

The document notes (footnotes 2 and 4) that data positioning, amplitude,
shaping and even the System C bit rate "may be altered to suit particular
transmission requirements", so these are nominal rather than hard.

### 1.5 LaserDisc and Laservision

| path | document | what it fixes |
|---|---|---|
| `docs/laserdisc/IEC-60857-1986-Laservision-NTSC/IEC-60857-1986-Laservision-NTSC.md` | IEC 60857:1986, Laservision NTSC | Disc mechanics and optics, and **the whole RF signal**: main carrier frequency and deviation, audio subcarrier frequencies, group-delay pre-distortion with tolerances, VBI line assignments |
| `docs/laserdisc/IEC-60856-1986-Laservision-PAL/IEC-60856-1986-Laservision-PAL.md` | IEC 60856:1986, Laservision PAL | The same for 625/50, plus the PAL pilot burst |
| `docs/laserdisc/IEC-6085{6,7}-...-Amendment-1/...md` | Amendments 1 | Push-pull radial differential tracking signal |
| `docs/laserdisc/IEC-6085{6,7}-...-Amendment-2/...md` | Amendments 2 | Revised mechanical/optical requirements; digital audio provisions |
| `docs/laserdisc/US4485483-.../us4485483.md` | US 4,485,483 | FM stereo with companded difference signal. Background only |

### 1.6 Compact Disc

| path | document | what it fixes |
|---|---|---|
| `docs/efm/ECMA-130/ECMA-130.md` | ECMA-130 | CD-ROM physical and optical characteristics, sector structure, EDC/ECC, CIRC, EFM channel coding |
| `docs/efm/IEC-60908-1999/IEC-60908-1999.md` | IEC 60908:1999 | CD-DA: disc parameters, HF and tracking signals, EFM, CIRC, subcode |

Both are relevant to the sibling EFM work rather than to tape measurement.


## 2. What this project could use and does not

### 2.1 What the repository currently encodes

`vhsdecode/format_defs/` carries RF and system parameters for eighteen tape
formats: VHS, VHSHQ, SVHS, SVHS_ET, UMATIC, UMATIC_HI, BETAMAX,
BETAMAX_HIFI, SUPERBETA, VIDEO8, HI8, EIAJ, QUADRUPLEX, VCR, VCR_LP, TYPEC,
TYPEB, VIDEO2000 — plus CVBS and LaserDisc through separate tools.

`vhsdecode/models/head_model.py` carries the physical mechanics, and its
`FORMAT_MECHANICS` table has **exactly two entries**:

| key | present |
|---|---|
| `("VHS", "NTSC", "SP")` | yes |
| `("VHS", "PAL", "SP")` | yes |
| everything else | **absent** |

So sixteen of the eighteen supported formats, and every tape speed other
than SP, have no mechanics at all. `mechanics_for()` returns `None` for them
and the head model, the tape model and the transport model cannot run.

### 2.2 The gaps the collection closes

**a. VHS LP and EP mechanics, and PAL LP.** SMPTE 32M table 3 and JVC table
1-1-1 give the complete track configuration for every speed. These are drop-in
`FORMAT_MECHANICS` entries:

| quantity | NTSC LP | NTSC EP | PAL LP | source |
|---|---|---|---|---|
| tape speed | 16.67 mm/s ± 0.5% | 11.12 mm/s ± 0.5% | 11.70 mm/s ± 0.5% | SMPTE 32M 4.1; JVC 1-1-1 |
| writing speed | — | 5.83 m/s | 4.86 m/s | JVC 1-1-1 |
| video track pitch/width | 0.029 mm | 0.019 mm | 0.024 mm | SMPTE 32M table 3; JVC 1-1-1 |
| video track angle (running) | 5° 57′ 8.5″ | 5° 56′ 8.1″ | 5° 56′ 58.8″ | SMPTE 32M table 3 (θ); JVC 1-1-1 gives 5° 56′ 48.1″ for NTSC EP |
| track angle (tape stationary) | 5° 56′ 7.4″ | 5° 56′ 7.4″ | 5° 56′ 7.4″ | both |
| audio/control head position X | 79.251 mm | 79.253 mm | 79.248 mm | SMPTE 32M table 3; JVC 1-1-1 |
| track shift error | 0.010 mm | 0.010 mm | — | SMPTE 32M table 3 |
| track bend, peak-to-peak | 0.010 mm or less | 0.010 mm or less | — | SMPTE 32M table 3 |

Note the one disagreement: SMPTE 32M gives the NTSC EP running track angle
as 5° 56′ 8.1″ and JVC gives 5° 56′ 48.1″. Forty arc-seconds apart. Neither
is obviously an OCR artefact and the discrepancy should be recorded rather
than silently resolved.

Note also that the repository's `video_track_width` for NTSC EP is **19.3 µm**
where both documents say 0.019 mm = 19 µm, and for PAL LP is **24.5 µm** where
JVC says 0.024 mm (table 1-1-1) or "approx. 0.025 mm" (table 1-1-2). These are
small but they propagate into every azimuth-loss fit, because the azimuth term
is `sinc(w·tanθ / λ)` and `w` is the track width.

**b. The track angle and stationary track angle are missing from
`FORMAT_MECHANICS` as a pair.** The table currently carries only
`track_angle_degrees` (the running angle). The *stationary* angle
θ₀ = 5° 56′ 7.4″ is the same for every VHS speed and system, and the
difference θ − θ₀ is exactly the tape's own contribution to the helix. That
difference is a witness for tape speed error that the model does not
currently have.

**c. The head-switch position is specified and the repository measures it.**
SMPTE 32M 3.6: *"The switching position between the two heads during playback
shall lie between 5 and 8 horizontal lines ahead of the leading edge of the
vertical sync signal."* `vhsdecode/luma_amplitude.py` records the switch at
line 259.6 ± 0.8 and 258.9 ± 0.3 from its own reading, and comments describe
this as "the head-switch arc's own reading of the onset". It is a specified
quantity with a stated tolerance band, and can be cited rather than measured
blind — and the measurement then becomes a *check against spec* instead of a
free parameter.

**d. Coercivity, with the two grades separated.**
`vhsdecode/models/tape_model.py` line 69 carries `600 Oe × 1000/(4π) = 47.7 kA/m`
as a comment. Both documents confirm it and one adds a second grade the
repository does not have:

| grade | coercivity | source |
|---|---|---|
| standard type H | approximately 50 × 10³ A/m | SMPTE 32M 3.2.1.4 |
| high-performance type H | approximately 70 × 10³ A/m | SMPTE 32M 7.3.3 |
| VHS blank cassette, NTSC and PAL/SECAM | 600 oersted class (nominal) | JVC 1.1.25, 1.1.26 |
| S-VHS cassette | 600 oersted class (nominal) | JVC 1.1.28 |

The 40% coercivity step between standard and high-performance tape is a real,
specified difference that the tape model currently cannot express.

**e. Recording depth — currently assumed, and the assumption is 17% high.**
`vhsdecode/models/interference.py` line 241 reads
`depth = mechanics.get("recording_depth_m", 0.35e-6)`. JVC §7.2.1 states the
video signal *"is recorded to a depth of about 0.3 µm on the magnetic coating
of the tape by the 0.3 µ gap of the video head"*. This matters twice over:

- `magnetic_limits()` computes particle count as volume × packing / particle
  volume, linear in depth — a 17% depth error is 0.7 dB of predicted tape SNR,
  and section 7.6 of `ELLIPTICAL_COLLAPSE.md` is currently carrying a 7.7 dB
  unexplained excess against exactly this prediction.
- The self-demagnetisation limit is `speed / (2π·depth)`: 0.35 µm gives
  2.64 MHz, 0.30 µm gives 3.08 MHz. A 17% shift in a stated frequency ceiling.

The same sentence independently confirms `head_model.TYPICAL["gap_m"] = 0.30e-6`,
which was a guess and is now a citation.

**f. Tape tension, with a number.** SMPTE 32M 3.7: *"normally from 0.30 N to
0.45 N at the entrance of the drum"*, measured with a spring scale for a full
supply reel. JVC 1-1-1 item 21 gives the same as *30–45 g*, at the tape
beginning and at the drum entrance. `transport_model.py` predicts tension
mechanisms but carries no tension value at all.

**g. Reel geometry, which pins the reel rate band.** `transport_model.py`
declares the supply and take-up reels with `radius_range_m: (0.012, 0.045)`
and marks them `certain: False`. Both documents fix the ends of that range:

| quantity | value | source |
|---|---|---|
| reel outside diameter | 89 ± 0.2 mm | JVC 1.1.25 §3 |
| hub diameter, over 60 min | 26 ± 0.15 mm | JVC 1.1.25 §3; SMPTE 32M table 1 |
| hub diameter, 60 min or under | 62 ± 0.2 mm (70 ± 0.2 mm if ≤ 30 min) | JVC 1.1.25 §3; SMPTE 32M table 1 |
| E-value | more than 1.5 mm | JVC 1.1.25 §3 |

So the pack radius runs from 13 mm to 44.5 mm for a T-120, which is very
close to the assumed range but is now *sourced*, and the T-60/T-30 case has a
31 mm hub radius that the current range excludes entirely.

**h. Tape thickness and length, per cassette type.** SMPTE 32M table 1 gives
tape thickness 15.6 ± 0.5 µm for T-160 and 19.0 +1/−2 µm for T-120 through
T-30, with lengths 327/246/185/125/64 m +3/−0. JVC gives the same as a
formula: `L = [2.2t + 2]` metres for NTSC, `L = [1.42t + 2]` for PAL/SECAM,
`t` in minutes. **The two-hour captures this arc works from are T-120s**, so
the pack radius as a function of tape position is directly computable — and
`docs/ELLIPTICAL_COLLAPSE.md` §7.3 names exactly that drift as an identifying
signature it cannot currently predict.

**i. The drum FG rate, which is a transport harmonic the model omits.**
JVC §4 gives DRUM FG at **240 Hz (NTSC) / 200 Hz (PAL/SECAM)** against a drum
rotation of 30 Hz / 25 Hz — eight FG poles per revolution. `transport_model.py`
predicts the drum at 29.97 Hz and its harmonics generically; the 8× FG rate is
a specific line to look for, and any servo ripple lands there.

**j. Tape width fluctuation.** 12.65 ± 0.01 mm average with *"width fluctuation
shall not exceed 6 µm"* (SMPTE 32M 3.2.1.2, JVC 1.1.25). The repository has no
width-fluctuation term, and 6 µm on a 58 µm track is a 10% track-width
modulation — directly in the azimuth-loss argument.

**k. The level-dependent sub-preemphasis, specified as a table with
tolerances.** This is the item most closely matched to what this project
already measures, and it is discussed in section 3.5.

**l. Two sources the repository already cites are now in hand.**
`vhsdecode/format_defs/vhs.py` cites "JVC video technical guide 3.1.4" for the
SECAM method-1 quarter-frequency countdown, and JVC §1.1.10 in this collection
confirms every number it uses: SECAM subcarriers 282 f_H = 4.40625 MHz (R−Y)
and 272 f_H = 4.25000 MHz (B−Y) with f_H = 15.625 kHz, a colour FM carrier
covering 3.9–4.75 MHz (an 850 kHz range), counted down by four to
0.98–1.19 MHz, with the modulation index also divided by four so the sideband
level falls about 10 dB relative to the carrier. The guide adds the anti-bell
filter centre frequency, 4.286 MHz, which the repository does not carry. The
1.5 H horizontal-correlation offset between adjacent tracks (JVC §1.1.3 and
§1.1.10) is likewise now citable rather than folklore.

### 2.3 Gaps the collection does **not** close

Stated plainly, because a clear list of absences is worth as much as the
inventory:

- **No mechanics for any non-VHS format.** Betamax, Video8, Hi8, EIAJ,
  Philips VCR, Video2000, Type B, Type C, Quadruplex and VHD have no document
  in the collection at all. U-matic is covered by another agent's source, not
  by this one. So the sixteen missing `FORMAT_MECHANICS` entries stay missing
  for fifteen of the sixteen formats.
- **No S-VHS track geometry as a separate entry.** SMPTE 32M's
  "high-performance type H" (clause 7) explicitly reuses table 2 and table 3,
  so S-VHS has the same geometry as VHS; only the tape grade, the FM carrier
  set and the sub-preemphasis differ. That is a real answer, but it means
  there is nothing new to add beyond the carrier and the tape.
- **No head parameters at all.** Gap length is given once, in passing, in a
  Hi-Fi section. There is nothing on head core dimensions, contour-effect
  length, protrusion, wear, permeability or inductance. Every parameter in
  `head_model.CHANNEL_PARAMETERS` — azimuth error, gain, height, protrusion —
  remains unsourced, as does `contour_length_m`.
- **No remanence, retentivity, squareness or coating thickness figure**
  anywhere in the collection. `tape_model.TAPE_VARIATIONS` lists remanence and
  coating thickness as mechanisms; neither has a specified value here.
  `PARTICLE_VOLUME_M3 = 1.0e-21` and `PACKING_FRACTION = 0.4` stay assumed.
- **No transport dimensions.** Capstan, pinch roller, impedance roller and
  guide diameters are not given by either VHS document. JVC §2 describes the
  mechanisms at length and dimensions none of them. Six of the eight entries
  in `transport_model.TRANSPORT` stay `certain: False` with typical values.
- **No preemphasis time constants for VHS video.** SMPTE 32M 3.9.1.1.2 says
  only *"as shown in figure 9"*, and the figure is an image. The FM high-pass
  filter is given as `T = C × Rb = 1.3 µs ± 0.05 µs` and `X = Rb/Ra = 4 ± 0.3`
  (SMPTE 32M figure 10) — but the *main preemphasis* network is a picture. The
  repository's `deemph_*` parameters therefore cannot be sourced from here.
- **No RF playback level, head output voltage, or channel noise figure.**
  Record current is specified only as *"the optimum value"* — that which
  maximises playback output. There is no absolute level anywhere.
- **No capture or digitisation guidance for tape RF.** See section 5.


## 3. Test signals and the vertical interval

This is the highest-value material in the collection, because a test signal is
a component whose synthetic side is exactly known — which is precisely the
input `docs/ELLIPTICAL_COLLAPSE.md` is built to consume.

**And this project already records these patterns.**
`tools/ringing_measure/record_playback_transfer.py` has

```python
DEFAULT_PATTERNS = ("ntc7composite-y-only", "multiburst-y-only",
                    "pulseandbar-y-only", "ramp-y-only", "sweep-y-only")
```

and uses them purely as broadband spectral excitation — a Welch spectrum over
fifteen bands. The exact element amplitudes, frequencies and pulse widths are
never used, because until now they were not written down anywhere in this
repository. **They are written down in this collection, to three decimal
places in microseconds.**

### 3.1 ITU-T J.63 — the normative element specification

`docs/video_metadata/T-REC-J-63-1990/T-REC-J-63-1990.md`. This is the document
the VITS pages cite and the one BT.1439-1 defers to for element definitions.
Its waveform drawings are images, but **the numeric specification is complete
in the body text** — every amplitude, position and half-amplitude duration with
its tolerance. It is the best single source of test-signal numbers in the
collection.

**625-line (Annex I), lines 17, 18, 330, 331.** The line is divided into
**32 equal time periods**, which must not differ from each other by more than
**± 40 ns**. Characteristic instants are referenced to the mid-amplitude point
of the leading edge of sync. Actual instants may not differ from nominal by
more than **250 ns** for luminance or **500 ns** for chrominance (the 20T
composite pulse excepted). Subcarrier **4 433 618.75 Hz ± 10 Hz**, locked at
**60 ± 5°** from the positive (B−Y) axis, harmonics at least **40 dB** down.

| line | element | position | amplitude | shape |
|---|---|---|---|---|
| 17 | B2 luminance bar | transitions 6H/32 and 11H/32, duration 5H/32 | **0.700 ± 0.007 V** | overshoot/undershoot ≤ 0.5%, tilt ≤ 0.5% |
| 17 | B1 2T sin² pulse | peak at 13H/32 | within ± 1% of B2 | **half-amplitude duration 200 ± 10 ns** (OIRT variant 160 ns) |
| 17 | F composite 20T pulse | peak 16H/32, base 15H/32–17H/32 | within ± 1% of B2 | **half-amplitude duration 2 ± 0.06 µs**; base-line perturbation ≤ 0.5% |
| 17, 330 | D1 five-riser staircase | transitions at **20, 22, 24, 26, 28 and 31 H/32** (last is the fall) | p-p within ± 1% of B2; each riser **1/5 of B2 = 0.140 V** | largest − smallest riser < 0.5% of the largest; risers shaped by a **Thomson filter whose transfer-function modulus first zeroes at 4.43 MHz** |
| 18 | luminance pedestal | 6H/32 to 31H/32 | within ± 1% of half B2 (**0.350 V**) | |
| 18 | C1 reference bar | transitions 6H/32, 8H/32, 10H/32 | 1st section **4/5 B2 = 0.560 V**, 2nd **1/5 B2 = 0.140 V**, each ± 1% | |
| 18 | C2 sine-wave bursts | see below | p-p within ± 1% of C1 p-p (**0.420 V**) | d.c. component of each burst ≤ 0.5% of C1; harmonics ≥ 40 dB down |
| 330 | D2 chrominance on D1 | 15H/32 to 30H/32 (subcarrier may stop at 28H/32) | **0.280 V ± 2% p-p** | envelope rise/fall ≈ 1 µs; **inherent differential gain ≤ 0.5%, differential phase ≤ 0.2°** |
| 331 | luminance pedestal | 6H/32 to 31H/32 | **0.350 V** ± 1% | |
| 331 | G1 chrominance bar | transitions 7H/32 and 14H/32 | p-p within ± 1% of B2 (**0.700 V**) | envelope rise/fall ≈ 1 µs; inherent chroma-luma cross-talk ≤ 0.5% of pedestal; phase within 2° of line 330 |
| 331 | G2 three-level chrominance | transitions **7, 9, 11 and 14 H/32** | **0.140 V**, **0.420 V**, **0.700 V** (1/5, 3/5, 1 × B2), each ± 1% | inherent phase/amplitude distortion ≤ 0.5° |
| 331 | E reference subcarrier | transitions 17H/32 and 30H/32 | p-p within ± 1% of **3/5 B2 = 0.420 V** | envelope rise/fall ≈ 1 µs |

**625-line multiburst (J.63 Table I), line 18** — each burst starts at zero
phase with the maximum number of complete cycles, and the gaps between bursts
are **not shorter than 0.4 µs nor longer than 2.0 µs**:

| burst | start | frequency |
|---|---|---|
| 1 | 12H/32 | 0.5 MHz |
| 2 | 15H/32 | 1.0 MHz (former OIRT: 1.5 MHz) |
| 3 | 18H/32 | 2.0 MHz (former OIRT: 2.8 MHz) |
| 4 | 21H/32 | 4.0 MHz |
| 5 | 24H/32 | 4.8 MHz |
| 6 | 27H/32 | 5.8 MHz |

**525-line (Annex II), line 17 of each field.** The line is divided into
**128 equal parts**. Instants are referenced to the half-amplitude point of the
leading edge of the luminance bar B2 (field 1) or the reference bar C1
(field 2), designated O_HR; **O_HR shall not exceed 24H/128 ± 125 ns relative
to O_H**. Systematic offset tolerance **± 150 ns** luminance and **± 300 ns**
chrominance, random error **± 25 ns**. Subcarrier **3 579 545 Hz** (M/NTSC) and
**3 575 611.49 Hz** (M/PAL), both ± 10 Hz. 100 IRE = blanking to white.

| field | element | position | amplitude | shape |
|---|---|---|---|---|
| 1 | B2 luminance bar | 0H/128 (O_HR) to 36H/128 | **100 ± 0.5 IRE** | **rise/fall 125 ± 5 ns** (integrated sin²); overshoot ≤ 1%, tilt ≤ 0.5% |
| 1 | B1 2T sin² pulse | peak at 44H/128 | within ± 0.5 IRE of B2 | **half-amplitude duration 250 ± 10 ns** |
| 1 | F modulated 12.5T pulse | peak at 51H/128 | within ± 0.5 IRE of B2 | **half-amplitude duration 1.57 ± 0.05 µs**; inherent chroma-luma delay inequality ≤ 5 ns; subcarrier harmonics ≥ 40 dB down |
| 1 | D1 / D2 five-riser staircase | transitions at **68, 74, 80, 86, 92 and 100 H/128** | p-p **100 ± 1 IRE (D1)** and **90 ± 1 IRE (D2)**; risers **20 IRE** / **18 IRE** | rise/fall shaped by a **2T sin² filter, nominal 250 ns** |
| 1 | D2 superimposed chrominance | transitions 60H/128 and 98H/128 | **envelope p-p 40 ± 0.4 IRE** | **envelope rise/fall 400 ± 25 ns**; inherent differential gain ≤ 0.25%, differential phase ≤ 0.2°; **phase 0 ± 1° relative to the programme burst** |
| 2 | C1 reference bar | 0H/128 (O_HR) to 8H/128 | within ± 0.5 IRE of B2 | rise/fall **125 ± 5 ns** |
| 2 | luminance pedestal | 8H/128 to 100H/128 | **50 IRE** ± 1% | |
| 2 | C2 multiburst on the pedestal | see below | **p-p 50 ± 0.5 IRE** | d.c. component ≤ 0.25 IRE; **envelope rise > 300 ns**, approximately integrated sin² |
| 2 | G three-level chrominance | transitions **68, 76, 84 and 96 H/128** | **20 ± 0.2**, **40 ± 0.4**, **80 ± 0.4 IRE** | envelope rise/fall **400 ± 25 ns**; inherent chroma-luma intermodulation ≤ 0.25 IRE; M/NTSC lags the programme burst by **90 ± 1°**, M/PAL locked to it |

**525-line multiburst (J.63 Table III), line 17 field 2**, same 0.4–2.0 µs
gap rule:

| burst | start | frequency |
|---|---|---|
| 1 | 12H/128 | 0.5 MHz |
| 2 | 24H/128 | 1.0 MHz |
| 3 | 32H/128 | 2.0 MHz |
| 4 | 40H/128 | 3.0 MHz |
| 5 | 48H/128 | 3.58 MHz |
| 6 | 56H/128 | 4.2 MHz |

Note where this **disagrees with the VITS pages** in section 3.3: J.63's
525-line multiburst runs 0.5 / 1.0 / 2.0 / 3.0 / 3.58 / 4.2 MHz, which is what
the collection's NTC-7 *Combination* carries, whereas its *FCC Multiburst*
uses 0.5 / 1.25 / 2.0 / 3.0 / 3.58 / 4.1 MHz — the FCC signal is a different
signal, not a variant reading of the same one.

### 3.2 The machine-readable source

`resources/definitions/vits/` holds eleven YAML files plus an index. Each
primitive carries a type (`colour_bar`, `sin_squared_pulse`, `composite_pulse`,
`burst`, `staircase`), a window in microseconds from the sync leading edge, a
channel (`y` or `c`), a combine rule (`replace` or `add`), an amplitude, a rise
time, and for chroma a `freq_mhz`, a `phase_deg` and a `subcarrier_lock_multiple`.
Global fields set `y_rise_time_us: 0.20` and `c_rise_time_us: 0.40`.

The amplitude convention is stated explicitly and matters: `amplitude_ire` and
`component_amplitude_ire` are the **carrier peak, one-sided** — a burst of
amplitude *N* swings ±*N* around its dc offset and is 2*N* peak-to-peak.

| file | signal |
|---|---|
| `resources/definitions/vits/ntsc/ntc7-composite.yaml` | NTC-7 Composite, line 17 field 1 |
| `resources/definitions/vits/ntsc/ntc7-combination.yaml` | NTC-7 Combination, line 280 field 2 |
| `resources/definitions/vits/ntsc/fcc-multiburst.yaml` | FCC Multiburst, line 18 field 1 |
| `resources/definitions/vits/ntsc/fcc-composite.yaml` | FCC Composite, line 281 field 2 |
| `resources/definitions/vits/ntsc/virs.yaml` | VIRS, line assignment-dependent |
| `resources/definitions/vits/pal/vits17.yaml` | VITS 17, line 17 field 1 |
| `resources/definitions/vits/pal/itu-multiburst.yaml` | ITU multiburst, line 18 field 1 |
| `resources/definitions/vits/pal/uk-national.yaml` | UK ITS-1, line 19 field 1 |
| `resources/definitions/vits/pal/vits20.yaml` | UK ITS-2, line 20 field 1 |
| `resources/definitions/vits/pal/itu-composite.yaml` | ITU composite, line 330 field 2 |
| `resources/definitions/vits/pal/itu-combination.yaml` | ITU combination, line 331 field 2 |

Normative sources are named per signal: ITU-T J.63 Annex I §2–5 and EBU
Tech. 3209 §7.2 for the PAL set, EIA RS-498 / SMPTE RP 168 / FCC Rules Part 73
for the NTSC set. Two PAL signals — UK ITS-1 on line 19 and ITS-2 on line 20 —
are flagged as **national specifications not defined in J.63, EBU 3209 or
BT.1439**, so their numbers carry less authority than the rest.

### 3.3 NTSC — the exact numbers

Reference: *100 IRE = 714.3 mV, 0 IRE = blanking*. All times are from the
leading half-amplitude edge of horizontal sync.

**NTC-7 Composite — line 17, field 1**

| element | type | amplitude | frequency | timing |
|---|---|---|---|---|
| 100 IRE white reference bar | colour bar | 100 IRE | — | 12.000–30.000 µs, rise 0.250 µs |
| 2T luminance pulse | sine-squared | 100 IRE | — | window 33.750–34.250 µs, centre 34.0 µs, **half-amplitude duration 0.25 µs**, rise 0.200 µs |
| 12.5T luminance component | sine-squared | 50 IRE | — | window 35.400–38.600 µs, centre 37.0 µs, **half-amplitude duration 1.6 µs** |
| 12.5T chrominance component | composite pulse | dc 0 IRE, **±50 IRE (100 p-p)** | 3.58 MHz, φ 0°, locked to burst | same window, rise 0.400 µs |
| chrominance reference burst | burst | dc 0 IRE, ±20 IRE (40 p-p) | 3.58 MHz, φ 180° | 42.000–60.000 µs |
| 5-step luminance staircase | staircase | peak 90 IRE | — | 46.000–60.000 µs, 5 equal steps of 2.80 µs |
| reference level bar | colour bar | 90 IRE | — | 60.000–62.000 µs |

Staircase levels: 18.0 / 36.0 / 54.0 / 72.0 / 90.0 IRE at 46.00–48.80 /
48.80–51.60 / 51.60–54.40 / 54.40–57.20 / 57.20–60.00 µs.

**NTC-7 Combination — line 280, field 2**

Grey background 50 IRE spanning 12.000–62.000 µs, with a +50 IRE boost over
12.000–16.000 µs (apparent 100 IRE).

| burst | amplitude | frequency | window |
|---|---|---|---|
| 1 | ±25 IRE (50 p-p) | 0.5 MHz | 18.000–23.000 µs |
| 2 | ±25 IRE | 1.0 MHz | 24.000–27.000 µs |
| 3 | ±25 IRE | 2.0 MHz | 28.000–31.000 µs |
| 4 | ±25 IRE | 3.0 MHz | 32.000–35.000 µs |
| 5 | ±25 IRE | 3.58 MHz | 36.000–39.000 µs |
| 6 | ±25 IRE | 4.2 MHz | 40.000–43.000 µs |

Then a three-level chrominance packet, all at 3.58 MHz, φ 90°, rise 1.000 µs:
zone 1 ±10 IRE over 46.000–50.000 µs (total excursion 40–60 IRE); zone 2
±20 IRE over 50.000–54.000 µs (30–70 IRE); zone 3 ±40 IRE over 54.000–60.000 µs
(10–90 IRE). All bursts rise in 0.200 µs.

**FCC Multiburst — line 18, field 1**

Grey pedestal 40 IRE over 9.200–62.000 µs, with a +60 IRE white reference
boost over 9.200–15.700 µs (apparent 100 IRE). Six packets, all ±30 IRE
(60 p-p), rise 0.200 µs:

| frequency | window |
|---|---|
| 0.5 MHz | 18.200–26.700 µs |
| 1.25 MHz | 28.200–34.200 µs |
| 2.0 MHz | 35.200–40.200 µs |
| 3.0 MHz | 41.200–46.200 µs |
| 3.58 MHz | 47.200–52.200 µs |
| 4.1 MHz | 53.200–58.200 µs |

**FCC Composite — line 281, field 2**

| element | type | amplitude | frequency | timing |
|---|---|---|---|---|
| chrominance reference burst zone | burst | dc 0 IRE, ±20 IRE | 3.58 MHz, φ 180° | 9.500–28.000 µs, rise 0.400 µs |
| 5-step luminance staircase | staircase | peak 80 IRE | — | 13.000–28.000 µs, 5 steps of 3.00 µs |
| staircase terminus reference | colour bar | 80 IRE | — | 28.000–30.000 µs |
| 2T luminance pulse | sine-squared | 100 IRE | — | window 35.250–35.750 µs, centre 35.5 µs, **half-amplitude duration 0.25 µs** |
| 12.5T luminance component | sine-squared | 50 IRE | — | window 37.900–41.100 µs, centre 39.5 µs, **half-amplitude duration 1.6 µs** |
| 12.5T chrominance component | composite pulse | dc 0 IRE, ±50 IRE component | 3.58 MHz, φ 180° | same window, rise 0.400 µs |
| 100 IRE white reference bar | colour bar | 100 IRE | — | 43.900–62.000 µs |

Staircase levels: 16.0 / 32.0 / 48.0 / 64.0 / 80.0 IRE at 13–16 / 16–19 /
19–22 / 22–25 / 25–28 µs.

**VIRS — line assignment-dependent (line 19/282 on LaserDisc, per IEC 60857 9.1.3)**

| element | amplitude | frequency | timing |
|---|---|---|---|
| 68 IRE reference bar (zone 1) | 68 IRE | — | 9.150–35.500 µs |
| chrominance reference burst | dc 0 IRE, ±22 IRE (44 p-p) | 3.58 MHz, φ 180° | 10.100–34.500 µs, rise 1.000 µs |
| 46 IRE reference bar (zone 2) | 46 IRE | — | 35.500–48.700 µs |
| post-VIRS blanking | 0 IRE | — | 48.700–62.000 µs |

Composite excursion during the burst overlap: 46–90 IRE.

### 3.4 PAL — the exact numbers

Reference: 700 mV = 100% white, 0 mV = blanking. PAL subcarrier 4.434 MHz
throughout (4.433619 MHz exactly, per JVC table 1-1-2). Chroma phase φ 60.66°
on the UK/ITU signals, φ 90° on the 20T pulse.

**VITS 17 — line 17, field 1** (J.63 Annex I §2; EBU 3209 §7.2)

| element | type | amplitude | timing |
|---|---|---|---|
| 100% white reference bar | colour bar | 700 mV | 12.0–22.0 µs, rise 0.200 µs |
| 2T luminance pulse | sine-squared | 700 mV | window 25.8–26.2 µs, centre 26.0 µs, **half-amplitude duration 0.200 µs** (T ≈ 100 ns in 625-line) |
| 20T luminance component | sine-squared | 350 mV | window 30.0–34.0 µs, centre 32.0 µs, **half-amplitude duration 2.0 µs** |
| 20T chrominance component | composite pulse | dc 0 mV, 350 mV component | 4.434 MHz, φ 90°, rise 0.400 µs, same window |
| 5-step luminance staircase | staircase | top 700 mV | 40.0–62.0 µs, rise 0.235 µs per riser |

Staircase: 140 / 280 / 420 / 560 / 700 mV (20/40/60/80/100%) at 40–44 / 44–48 /
48–52 / 52–56 / **56–62** µs. The last tread is 6 µs rather than 4 µs because
the risers follow J.63's characteristic instants 20H/32, 22H/32, 24H/32,
26H/32, 28H/32 and 31H/32 (fall), not equal spacing.

**ITU multiburst — line 18, field 1** (J.63 Annex I §3; EBU 3209 §7.2.9)

Grey pedestal 350 mV over 12.0–62.0 µs; positive amplitude reference bar
+210 mV over 12.0–16.0 µs; negative reference bar −210 mV over 16.0–20.0 µs
(both rise 0.350 µs). Six packets, all 210 mV envelope, rise 0.200 µs:

| frequency | window | phase |
|---|---|---|
| 0.5 MHz | 24.0–28.0 µs | 0° |
| 1.0 MHz | 30.0–35.0 µs | 0° |
| 2.0 MHz | 36.0–41.0 µs | 0° |
| 4.0 MHz | 42.0–47.0 µs | 0° |
| 4.8 MHz | 48.0–53.0 µs | 144° |
| 5.8 MHz | 54.0–60.0 µs | −144° |

The document is explicit that J.63 and EBU 3209 fix the frequencies and timing
but **do not mandate the 144°/−144° phases** — those are this repository's
implementation choice. Treat the phase of the top two packets as unspecified.

**UK ITS-1 — line 19, field 1** (national spec, not in J.63/3209/BT.1439)

100% white bar 700 mV over 12.0–22.0 µs; 2T pulse 700 mV centred 26.0 µs,
half-duration 0.200 µs; 20T chrominance reference pulse dc 0 mV at 4.434 MHz
φ 90° centred 30.0 µs; 20T luminance reference 350 mV centred 30.0 µs,
**half-duration 1.0 µs**; full-line chrominance reference burst dc 0 mV /
Y 70 mV over 34.0–60.0 µs at 4.434 MHz φ 60.66°; 5-step staircase 140/280/420/
560/700 mV in equal 4.0 µs steps over 40.0–60.0 µs; end-of-line reference bar
700 mV over 60.0–62.0 µs.

**UK ITS-2 — line 20, field 1** (national spec)

Grey pedestal 350 mV over 12.0–32.0 µs; 100%-level chrominance burst
dc 0 mV / Y 350 mV over 14.0–28.0 µs; 43%-level burst dc 0 mV / Y 150 mV over
34.0–62.0 µs. Both at 4.434 MHz, φ 60.66°.

**ITU composite — line 330, field 2** (J.63 Annex I §4; BT.628/BT.473)

100% white bar 700 mV over 12.0–22.0 µs; 2T pulse 700 mV centred 26.0 µs,
half-duration 0.200 µs; chrominance phase reference burst dc 0 mV / Y 140 mV
over 30.0–60.0 µs at 4.434 MHz φ 60.66°; 5-step staircase over 40.0–62.0 µs
with the same J.63 instants and levels as VITS 17.

**ITU combination — line 331, field 2** (J.63 Annex I §5; EBU 3209 §7.2.6–7.2.7)

Grey pedestal 350 mV over 12.0–62.0 µs. Three-step chrominance staircase, all
4.434 MHz φ 60.66°, dc offset 0: Y 70 mV (20%) over 14.0–18.0 µs; Y 210 mV
(60%) over 18.0–22.0 µs; Y 350 mV (100%) over 22.0–28.0 µs. Then a sustained
60% reference burst, Y 210 mV, over 34.0–60.0 µs.

### 3.5 The other test signal: VHS's own specified sub-preemphasis

Not a vertical-interval signal, but the same kind of object and arguably more
useful here, because it is **a frequency response specified as a function of
input level, with per-point tolerances** — which is exactly the two-dimensional
component `ELLIPTICAL_COLLAPSE.md` §6a reports as *saturated for want of
places* on the amplitude axis. Six frequencies × four levels is 24 specified
points; the current amplitude axis has 9 sweep bins.

**SMPTE 32M table 4 — basic system, LP mode** (0 dB reference at 10 kHz; input
level 0 dB = sync tip to peak white):

| input level | 50 kHz | 200 kHz | 500 kHz | 1 MHz | 2 MHz | 3 MHz |
|---|---|---|---|---|---|---|
| 0 dB | 0 ± 0.3 | 0 ± 0.4 | 0.1 ± 0.4 | 0.7 ± 0.4 | 1.2 ± 0.5 | 1.5 ± 0.6 |
| −10 dB | 0 ± 0.3 | 0.1 ± 0.4 | 0.8 ± 0.4 | 2.2 ± 0.5 | 3.3 ± 0.7 | 4.0 ± 1.0 |
| −14 dB | 0 ± 0.3 | 0.2 ± 0.4 | 2.0 ± 0.5 | 3.8 ± 0.8 | 5.3 ± 1.0 | 6.0 ± 1.2 |
| −20 dB | 0 ± 0.3 | 0.4 ± 0.4 | 2.3 ± 0.5 | 5.0 ± 0.8 | 6.8 ± 1.0 | 7.5 ± 1.2 |

**SMPTE 32M table 4 — basic system, EP mode:**

| input level | 50 kHz | 200 kHz | 500 kHz | 1 MHz | 2 MHz | 3 MHz |
|---|---|---|---|---|---|---|
| 0 dB | 0 ± 0.4 | 0.4 ± 1.5 | 1.0 ± 1.5 | 2.0 ± 1.5 | 2.0 ± 2.0 | 1.0 ± 2.0 |
| −10 dB | 0 ± 0.4 | 1.2 ± 1.5 | 2.5 ± 1.5 | 4.5 ± 1.5 | 4.5 ± 2.0 | 3.0 ± 2.0 |
| −14 dB | 0 ± 0.4 | 1.7 ± 1.5 | 4.5 ± 1.5 | 5.5 ± 1.5 | 5.5 ± 2.0 | 3.5 ± 2.0 |
| −20 dB | 0 ± 0.4 | 3.0 ± 1.5 | 6.0 ± 1.5 | 7.0 ± 1.5 | 6.5 ± 2.0 | 4.5 ± 2.0 |

**SMPTE 32M table 7 — high-performance (S-VHS) system, SP mode:**

| input level | 200 kHz | 500 kHz | 1 MHz | 2 MHz | 3 MHz | 5 MHz |
|---|---|---|---|---|---|---|
| 0 dB | −1.73 ± 0.30 | −1.60 ± 0.30 | −1.04 ± 0.30 | −0.37 ± 0.50 | −0.07 ± 0.50 | −0.06 ± 0.50 |
| −10 dB | −1.30 ± 0.30 | −0.73 ± 0.50 | 0.69 ± 0.50 | 1.75 ± 0.50 | 2.10 ± 0.50 | 2.02 ± 0.50 |
| −20 dB | −0.65 ± 0.50 | 1.09 ± 0.50 | 2.86 ± 0.50 | 4.16 ± 0.50 | 4.60 ± 0.50 | 4.43 ± 0.60 |
| −30 dB | −0.49 ± 0.50 | 2.35 ± 0.50 | 5.30 ± 0.60 | 7.14 ± 0.60 | 7.64 ± 0.70 | 7.34 ± 0.70 |

**SMPTE 32M table 7 — high-performance system, EP mode:**

| input level | 200 kHz | 500 kHz | 1 MHz | 2 MHz | 3 MHz | 5 MHz |
|---|---|---|---|---|---|---|
| 0 dB | −1.86 ± 0.30 | −0.76 ± 0.30 | 0.81 ± 0.30 | 1.60 ± 0.50 | 1.72 ± 0.50 | 1.54 ± 0.50 |
| −10 dB | −1.46 ± 0.30 | 0.11 ± 0.50 | 2.50 ± 0.50 | 3.73 ± 0.50 | 3.89 ± 0.50 | 3.59 ± 0.50 |
| −20 dB | −0.82 ± 0.50 | 1.93 ± 0.50 | 4.70 ± 0.50 | 6.14 ± 0.50 | 6.36 ± 0.50 | 5.97 ± 0.60 |
| −30 dB | −0.63 ± 0.50 | 3.21 ± 0.50 | 7.14 ± 0.60 | 9.15 ± 0.60 | 9.43 ± 0.70 | 8.89 ± 0.70 |

The measurement method is stated with the tables: peak-to-peak sine amplitude
in against peak-to-peak out, 0 dB reference at 10 kHz, input 0 dB defined as a
100% peak-white signal measured sync tip to peak white, compared in the active
part of the line; sync may be included.

`vhsdecode/format_defs/vhs.py` sets `use_sub_deemphasis = [False, True, True,
True][tape_speed]` — off at SP, on at LP and EP — which matches SMPTE 32M
exactly (clause 3.9 has main preemphasis only; clause 4.3.1.1.1 adds
subpreemphasis for LP and EP). But the *shape* comes from tuned constants
(`nonlinear_scaling_1`, `nonlinear_exp_scaling`, `nonlinear_logistic_*`), and
the S-VHS path carries a frank comment that the sub-emphasis *"isn't properly
implemented yet"*. These four tables are the specification those filters are
approximating, and they come with tolerances, which makes them a pass/fail
criterion rather than a target.

### 3.6 LaserDisc's specified group delay — a phase-axis synthetic component

Worth separating out, because almost nothing in the collection specifies
*phase*. IEC 60857 9.1.7 (NTSC) requires the video group delay to be
pre-distorted, to equalise the playback low-pass filter:

| frequency | group delay | frequency | group delay |
|---|---|---|---|
| 0.5 MHz | 0 ns (reference) | 3.58 MHz | −80 ± 15 ns |
| 2 MHz | −15 ± 15 ns | 4 MHz | −135 ± 30 ns |
| 3 MHz | −45 ± 15 ns | 4.2 MHz | −200 ± 50 ns |

IEC 60856 9.1.7 (PAL) gives the 625-line equivalent:

| frequency | group delay | frequency | group delay |
|---|---|---|---|
| 0.5 MHz | 0 ns (reference) | 4.0 MHz | −85 ± 20 ns |
| 2.0 MHz | −10 ± 15 ns | 4.43 MHz | −135 ± 20 ns |
| 3.0 MHz | −35 ± 15 ns | 4.8 MHz | −200 ± 50 ns |


### 3.7 Colour bars — the synthetic side of this arc's own test capture

`tools/ringing_measure/elliptical_collapse.py` is run on *"the 75-bars NTSC SP
capture"*. **SMPTE 170M-2004 Annex A.6 specifies exactly what is on it**, to
four decimal places in IRE, including the subcarrier phase of every bar. This
is the single closest match in the collection to a component this project is
already measuring.

**Table A.3 — 75/7.5/75/7.5 colour bars, to 10⁻⁴ IRE** (the arc's capture):

| bar | luminance (IRE) | chroma level (IRE) | min chroma excursion (IRE) | max chroma excursion (IRE) | phase (degrees) |
|---|---|---|---|---|---|
| white | 76.8750 | 0.0000 | | | |
| yellow | 68.9663 | 62.0675 | 37.9325 | 100.0000 | 167.0812 |
| cyan | 56.1319 | 87.7363 | 12.2638 | 100.0000 | 283.4558 |
| green | 48.2231 | 81.9253 | 7.2605 | 89.1858 | 240.7098 |
| magenta | 36.1519 | 81.9253 | −4.8108 | 77.1145 | 60.7098 |
| red | 28.2431 | 87.7363 | −15.6250 | 72.1113 | 103.4558 |
| blue | 15.4088 | 62.0675 | −15.6250 | 46.4425 | 347.0812 |
| black | 7.5000 | 0.0000 | | | |

**Table A.1 — 100/7.5/100/7.5 colour bars, to 10⁻⁴ IRE:**

| bar | luminance (IRE) | chroma level (IRE) | min chroma excursion (IRE) | max chroma excursion (IRE) | phase (degrees) |
|---|---|---|---|---|---|
| white | 100.0000 | 0.0000 | | | |
| yellow | 89.4550 | 82.7567 | 48.0767 | 130.8333 | 167.0812 |
| cyan | 72.3425 | 116.9817 | 13.8517 | 130.8333 | 283.4558 |
| green | 61.7972 | 109.2338 | 7.1806 | 116.4144 | 240.7098 |
| magenta | 45.7025 | 109.2338 | −8.9144 | 100.3194 | 60.7098 |
| red | 35.1575 | 116.9817 | −23.3333 | 93.6483 | 103.4558 |
| blue | 18.0450 | 82.7567 | −23.3333 | 59.4233 | 347.0812 |
| black | 7.5000 | 0.0000 | | | |

Tables A.2 and A.4 repeat these rounded to 0.1 IRE. Note that the **phases are
identical between the 100% and 75% sets** — saturation scales the amplitude,
not the angle — so the phase column is a saturation-independent reference. The
100% set's chroma excursions reach 130.8 IRE and −23.3 IRE, well outside the
0–100 range, which is worth knowing before treating an over-excursion as a
decoder defect.

Two consequences for this arc:

- The luminance column is eight exactly-known levels spanning 7.5 to 76.9 IRE
  on one line. `ELLIPTICAL_COLLAPSE.md` §6a reports the **amplitude axis as
  saturated for want of places** — nine sweep bins against ninety-one entries.
  Eight specified bar levels, each with a specified chroma level and phase, is
  a second amplitude axis whose synthetic side needs no fitting at all.
- The phase column bears directly on the burst-to-bar phase lock. Every bar's
  subcarrier phase is fixed relative to burst to four decimal places, so a
  measured bar phase is a *residual* against a known number rather than a
  relative measurement.

### 3.8 The LaserDisc vertical interval, line by line

IEC 60857 clause 9.1 assigns the NTSC vertical interval explicitly, which is
worth recording because `lddecode/core.py` currently carries *guessed* VITS
white locations — `LD_VITS_whitelocs` lists lines 20, 13 and 11 as "typical
VITS locations (first most common)" and `LD_VITS_code_slices` lines 16, 17
and 10. The standard fixes them:

| lines | content | clause |
|---|---|---|
| 10–18 and 273–281 | reserved for address or data signals (CCIR Rec. 314-4) | 9.1.5 |
| 19 and 282 | VIRS, per FCC Rec. 73-699 and CCIR Rec. 314-4; absent in monochrome | 9.1.3 |
| 20 | composite test signal, per NTC Report No. 7 or CCIR Rec. 473-2 | 9.1.4 |
| 21 and first half of 284 | handicapped caption data, per PBS Report E7709 | 9.1.5 |
| 283 | combination test signal, per NTC Report No. 7 or CCIR Rec. 473-2 | 9.1.4 |
| unspecified lines | video content at blanking level, reserved | 9.1.5 |

So the "composite test signal on line 20" is the NTC-7 Composite whose exact
element table is in section 3.3, and the "combination test signal on line 283"
is NTC-7 Combination — both already defined numerically in this collection.
The standard also caps the signal: maximum luminance not exceeding 110 IRE
and maximum chroma saturation not exceeding 100% (9.1.6), and requires the
EIA recommended burst to be present in monochrome as well as colour, with
colour burst also during the vertical interval (9.1.2).


### 3.9 EBU Tech. 3209 — the tolerance on every element

`docs/video_metadata/EBU-Tech-3209/EBU-Tech-3209.md` §7.2 specifies what the
*generator* must produce, measured at the main video output terminated in
**75 Ω ± 0.1%**. Where J.63 says what the signal is, this says how far it may
be wrong — which is what turns a measured departure into a verdict:

| element | parameter | tolerance |
|---|---|---|
| luminance bar | amplitude | **0.7 V ± 1%**; duration 10 µs; rise/fall ≈ 200 ns; tilt over the 10 µs < 0.5% |
| staircase | top tread | within ± 1% of the luminance bar; 5 risers; Thomson-filter shaping, first zero at 4.43 MHz |
| | line-time non-linearity | largest − smallest riser **< 0.5%** of the largest |
| | superimposed subcarrier | **4 433 618.75 Hz ± 10 Hz**, phase **60 ± 5°** to the B−Y axis referred to the burst, **0.28 V ± 1% p-p**, envelope rise/fall ≈ 1 µs |
| | inherent differential gain / phase | **≤ 0.5%** / **≤ 0.2°** |
| 2T pulse | amplitude / half-amplitude duration | within ± 1% of the bar / **200 ± 6 ns** |
| 20T composite pulse | amplitude / half-amplitude duration | within ± 1% of the bar / **2 ± 0.06 µs** |
| | base-line perturbation | **< 0.5%** of pulse amplitude |
| | **subcarrier leak** | **< 3.5 mV p-p** on insertion lines; harmonics ≤ −40 dB |
| chrominance bar | p-p / pedestal | within ± 1% of the bar / **0.35 V ± 1%**; chroma-luma intermodulation ≤ 0.5% of pedestal |
| three-level chrominance | transitions / sections | **7, 9, 11, 14 H/32** / **0.14 V, 0.42 V, 0.7 V**, each within 1% |
| chrominance reference | amplitude | **0.42 V ± 1% p-p** on a 0.35 V ± 1% pedestal |
| multiburst | burst frequencies | **0.5, 1.0, 2.0, 4.0, 4.8, 5.8 MHz, each ± 1%** |
| | burst p-p / d.c. component | within 1% of C1 (**0.42 V**) / ≤ 0.5% of the reference bar |
| | gaps between bursts | **400 ns to 2 µs** |
| black level | stability over 0–45 °C | change **≤ 5 mV** |

Phase stability between adjacent lines ≤ 2°, and between the chrominance
elements of line 331 ≤ 0.5°. The equipment must be able to insert into
**lines 16–22 and 329–335**.

Two further parts are useful as *procedure* rather than as constants.
**Appendix 2** defines sub-standard-input stress tests: white noise 30 Hz to
6 MHz; 50 Hz hum swept about the field frequency; low-frequency tilt through a
series capacitor with time constant 2CR_T, R_T = 75 Ω; **drop-out simulated by
a 2 µs negative pulse of 0.6 V** swept through the field; flashing by a 50 µs
pulse; noise conversion measured through complementary filters at 7.5 kHz.
**Appendix 3** reports what a compliant unit actually achieves — normal
operation from +7 dB to −21 dB input level and down to a **+23 dB luminance
signal-to-noise ratio**, with HF-to-LF noise conversion 16 dB down. For a
project whose input is a degraded tape rather than a studio feed, that is the
nearest thing in the collection to a specification of graceful degradation.

### 3.10 ITU-R BT.1439-1 — the measurement methods and their filters

`docs/video_metadata/R-REC-BT-1439-1-2006/R-REC-BT-1439-1-2006.md`. Its 34
test-signal drawings and all its filter schematics are images, so the element
geometry has to come from J.63. What *is* extractable is the method, the
vocabulary and the filter component values — and the vocabulary is worth
adopting, because it splits linear distortion by **time scale** in exactly the
way this arc's axes do:

| term | signal | what is measured |
|---|---|---|
| long-time waveform distortion | — | peak overshoot as % of nominal luminance amplitude, decay time, and **slope at onset in %/s** |
| field-time waveform distortion | field-frequency square wave (element A) | **the first and last 250 µs (≈ 4 lines) are neglected** |
| line-time waveform distortion | bar B3 (625) / B2 or B3 (525) | **the first and last 1 µs are neglected**; base-line measured between the level **400 ns (625) or 500 ns (525)** after the half-amplitude point of the bar trailing edge and the level half a bar duration later; band-limited by the Thomson filter |
| short-time waveform distortion | 2T pulse B1 with bar | pulse-to-bar ratio and time-weighted lobe amplitudes, expressed as **K-rating** |
| chroma-luma delay inequality | composite pulse F | in ns, **positive when chrominance lags luminance** |
| differential gain | D2 | peak-to-peak DG = 100·(A_max − A_min)/A₀, referred to the subcarrier amplitude on the blanking-level tread |
| differential phase | D2 | peak-to-peak DP = Φ_max − Φ_min |
| luminance non-linearity | D1 through a sin²-approximating network | (largest − smallest riser) / largest, in % |
| sync steady-state non-linearity | — | deviation of sync amplitude from **3/7 of the luminance bar for 625-line, 4/10 for 525-line** |

**Noise measurement.** R.m.s. instruments with an integrating time of about
**1 s**; band-limited by the low-pass filter of §1 and the high-pass of §2; if
the circuit is carrying a signal, band-limit with a **first-order 200 kHz
high-pass with a 20 dB per decade slope** before weighting. Low-frequency
noise is the peak-to-peak amplitude after band-limiting to **500 Hz – 10 kHz**.
Periodic noise covers 1 kHz up to the band limit.

**The unified weighting network** is given as component formulae rather than a
picture: L = Z₀τ, C = τ/Z₀, R₁ = aZ₀, R₂ = Z₀/a with **τ = 245 ns and a = 4.5**,
so A_∞ → 20·log(1 + a) = **14.8 dB**. Its **noise weighting factors in a 5 MHz
band are 7.4 dB for flat noise and 12.2 dB for triangular noise** — directly
applicable to a demodulated FM channel, whose noise is triangular.

**Filter component values** (the parts of Annex 2 that survived as text):

- 5 MHz low-pass noise filter: C1 100, C2 545, C3 390, C4 428, C5 563, C6 463,
  C7 259 pF; L1 2.88, L2 1.54, L3 1.72 µH; notches at 9.408, 5.506 and
  6.145 MHz. Capacitors ± 2%; **inductor Q at 5 MHz between 80 and 125**.
- 10 kHz combined high-pass/low-pass: C1 139 000, C2 196 000, C3 335 000,
  C4 81 200 pF (± 5%); L1 0.757, L2 3.12, L3 1.83, L4 1.29 mH (± 2%); Q ≥ 100.
- **Thomson filter**, first zero of the transfer function at **f_ow = 3.3 MHz**:
  C1 147.7, C2 4044, C3 141.6, C4 1057, C5 310.5 pF; L1 2.948, L2 0.5752 µH.

**K-rating (Annex 4)**, which is the standard way of scoring exactly the kind
of ringing this arc measures:

- 2T pulse response mask: **± 4K at ± 200 ns, ± 2K at ± 400 ns, ± K at
  ± 800 ns and beyond**; the illustrated masks correspond to K₍2T₎ = 3%.
- **K₍P/B₎ = ¼·|B/P − 1| × 100%.**
- Mask half-amplitude durations: **2T = 200 ns for f_c = 5 MHz, 167 ns for
  f_c = 6 MHz (625-line); 250 ns (525-line)**.
- Acceptance-test form: C_r = ½B₍r−1₎ + B_r + ½B₍r+1₎;
  **K1 ≥ ⅛·|r·C_r/C₀|** for 2 ≤ |r| ≤ 8 and K1 ≥ |C_r/C₀| for |r| ≥ 8;
  **K2 = ¼·|(1/C₀)ΣB_r − 1|**; **K3 = ⅙·|ΣB_r − 1|**;
  **K4 = (1/20)·(Σ|B_r| − 1)**.

Also worth carrying: **subcarrier phase lock of the test elements**, referenced
to the positive (B−Y) axis (BT.1439 Table 1) — D2 at **60 ± 5°** for PAL and
**180 ± 1°** for M/PAL and NTSC; F at 60 ± 5° / 180 ± 1° / not defined; G at
60 ± 5° / 180 ± 1° / **90 ± 1°** for NTSC. And the 625-line **T value: 83 ns
for a 6 MHz video bandwidth, 100 ns for 5 MHz** — which is what makes "2T"
200 ns in a 5 MHz system.


## 4. Composite sync geometry — what the sync-only constraint measures against

This project's standing rule is that the correction derives from the sync pulse
and nothing else. The sync pulse is therefore the most load-bearing synthetic
component the arc has, and the collection specifies it completely, with
tolerances, in two independent documents.

### 4.1 NTSC — SMPTE 170M-2004

**Levels (table 1).** White **100 ± 1 IRE**; black/setup **7.5 ± 1 IRE**;
blanking **0.0 IRE, reference**; burst amplitude **40 ± 1 IRE p-p**; sync
**−40 ± 1 IRE**. Composite amplitude without chroma **140 IRE p-p**; maximum
with chroma **171 IRE p-p**. **140 IRE nominally equals 1 V**, and in DC-coupled
systems blanking is nominally 0 V.

**Horizontal timing (table 2).**

| specification | measurement point | value | tolerance |
|---|---|---|---|
| total line period (derived) | | **63.556 µs** | |
| horizontal blanking rise time | 10–90% | **140 ns** | ± 20 |
| sync rise time | 10–90% | **140 ns** | ± 20 |
| burst envelope rise time | 10–90% | **300 ns** | +200 / −100 |
| horizontal blanking start → horizontal reference point | 50% | **1.5 µs** | ± 0.1 |
| horizontal sync width | 50% | **4.70 µs** | ± 0.10 |
| horizontal reference point → burst start | 50% | **19 cycles** | defined by SC/H |
| SC/H phase | | **0°** | ± 10 |
| horizontal reference point → horizontal blanking end | 50% | **9.20 µs** | +0.20 / −0.10 |
| burst | 50% | **9 cycles** | ± 1 |

Burst start and end are defined by the zero crossing preceding the first, and
following the last, half cycle that is **≥ 50% of burst amplitude**. **The
level of each half envelope of the burst shall not vary by more than 0.5 IRE.**
No burst during the nine-line vertical blanking interval. Sync and blanking
edges "should be skew symmetric", and **raised cosine shaping is preferred** —
which is a specified *edge shape*, and this arc measures edge shapes.

**Vertical timing (table 3).** Field period **16.6833 ms**; frame period
**33.3667 ms**; vertical blanking start **1.50 ± 0.10 µs** before the first
equalising pulse (50%); vertical blanking **20 lines plus 1.5 µs** =
**1272.62 µs**; pre-equalising **3 lines**, pulse width **2.30 ± 0.10 µs**;
vertical sync **3 lines**, serration width **4.70 ± 0.10 µs**; post-equalising
**3 lines**, pulse width **2.30 ± 0.10 µs**. **All pulse rise and fall times,
unless otherwise specified, are 140 ± 20 ns (10–90%), and all pulse widths are
measured at 50%.**

**Frequencies.** f_sc = 5 MHz × 63/88 = **3.579545… MHz ± 10 Hz**, with a
recommended **drift below 1/10 Hz per second and jitter below 1 ns p-p over one
horizontal line**; f_H = (2/455)·f_sc = **15 734.265… Hz**, **227.5 subcarrier
cycles per line**; f_V = (2/525)·f_H = **59.94005994… Hz**. Time coincidence
between nominally coincident signals: **± 25 ns**.

**Field identification.** **Field I is the field in which the first zero
crossing of burst on line 10 is positive going**; field III is negative going.
Colour frame A = fields I and II, colour frame B = fields III and IV.

### 4.2 The 625-line systems, and the rest — ITU-R BT.470-6 and CCIR 624-4

`docs/video_formats/BT-470-6-1998/BT-470-6-1998.md` and
`docs/video_formats/BT-624-4-1990/BT-624-4-1990.md` are near-duplicates; where
they disagree numerically the 624-4 transcription is generally the cleaner.
All durations are measured between half-amplitude points.

**Levels (BT.470-6 table 1).** Blanking 0, peak white 100 for all systems.
Sync level **−40** for system M, **−43** for the 625-line group. Setup
**7.5 ± 2.5** for M and N, **0** for B/G/H, 0 or 0–7 for I and D/K, **+5** for
L. Peak level including chroma: 120 (M), 133 (PAL), 115 (SECAM), 124 (L).
Nominal video bandwidth 4.2 MHz (M, N), 5 MHz (B, G, H), 5.5 MHz (I), 6 MHz
(D, K, L). **Peak white = 100 corresponds to 1.0 V across a matched 75 Ω
termination.** On the alternative 625-line scale, sync = 0, blanking = 30,
peak white = 100.

**Line sync (BT.470-6 table 1-1).**

| | system M (M/NTSC) | system N (N/PAL) | 625-line group (B, B1, D, D1, G, H, I, K, K1, L) |
|---|---|---|---|
| nominal line period H | 63.492 µs (63.5555) | 64 µs | **64 µs** |
| line-blanking interval | 10.2–11.4 µs (10.9 ± 0.2) | 10.24–11.52 (12 ± 0.3) | **12 (+0.0 / −0.3) µs** |
| front porch | 1.27–2.54 (1.27–2.22) | 1.28–2.56 (1.5 ± 0.3) | **1.5 (+0.3 / −0.0) µs** |
| sync pulse | 4.19–5.71 (4.7 ± 0.1) | 4.22–5.76 (4.7 ± 0.2) | **4.7 ± 0.2 µs** |
| blanking edge build-up, 10–90% | ≤ 0.64 / ≤ 0.48 µs | ≤ 0.64 (0.3 ± 0.1) | **0.3 ± 0.1 µs** |
| sync edge build-up, 10–90% | ≤ 0.25 µs | ≤ 0.25 (0.2 ± 0.1) | **0.2 ± 0.1 µs**; system I **0.25 ± 0.05 µs** |

The asymmetric line-blanking and front-porch tolerances are the preferred
values, chosen to protect Teletext System B. CCIR 624-4 adds a value BT.470-6
omits: **system I front porch 1.65 ± 0.1 µs**, and system I "O_H to back edge
of line blanking" 10.4 µs. France and the former OIRT countries carry an
instantaneous line-period tolerance of **± 0.032 µs**.

**Field sync (BT.470-6 table 1-2).** 625-line group: field period **20 ms**;
field-blanking interval **25H + a**; front edge of field blanking to the first
equalising pulse **3 ± 2 µs**; equalising / sync / equalising sequences of
**2.5H each**; **equalising pulse width 2.35 ± 0.1 µs**; **broad (field-sync)
pulse width 27.3 µs** nominal, system I **27.3 ± 0.1 µs**; interval between
field-sync pulses **4.7 ± 0.2 µs** (system I ± 0.1); build-up **0.2 ± 0.1 µs**
(system I 0.25 ± 0.05). System M: field period 16.667 ms (16.6833);
field blanking (19–21)H + a; equalising pulse 2.3 ± 0.1 µs; broad pulse
27.1 µs; interval 4.7 ± 0.1 µs. Japan: field blanking 0.07v (+0.012v / −0),
and the equalising pulse has **0.45 to 0.5 times the area** of a line-sync
pulse.

**Line-frequency stability.** f_H tolerances: M **15 734.264 ± 0.0003%**; B, D1,
G, D, K, K1, L **15 625 ± 0.02%**; **I 15 625 ± 0.00002%**; N ± 0.15%. Maximum
rate of variation for monochrome 0.05 %/s (0.15 for M); at least **2 × 10⁻⁷**
stability is needed for full precision-offset benefit, and in the UK and Japan
the maximum rate of change is **0.1 Hz/s**.

**Burst.** Start after O_H: M/NTSC **4.71–5.71 µs (5.3 nominal)**, at least
**0.38 µs** after the trailing edge of line sync; M/PAL **5.8 ± 0.1 µs**;
625-line PAL **5.6 ± 0.1 µs**. Duration: M/NTSC **2.23–3.11 µs (9 ± 1 cycles)**;
M/PAL **2.52 ± 0.28 µs (9 ± 1)**; 625-line PAL **2.25 ± 0.23 µs (10 ± 1)**.
Amplitude p-p: NTSC **4/10 of blanking-to-peak-white ± 10%**; PAL **3/7 of
blanking-to-peak-white ± 10%** (± 3% for systems D and I). Phase: NTSC 180°
relative to (B−Y); PAL ± 135° relative to U. Burst blanking: 625-line PAL
**9 lines — 311–319, 623–6, 310–318, 622–5**; M/PAL 11 lines. In monochrome
transmission, residual subcarrier-frequency signal in line blanking must be
**≥ 35 dB** below burst p-p.

**Subcarriers, to full stated precision.** M/NTSC **3 579 545 ± 10 Hz**
(455/2 · f_H); M/PAL **3 575 611.49 ± 10 Hz** (909/4 · f_H — see the defect
note); 625-line PAL **4 433 618.75 ± 5 Hz** (1135/4 + 1/625 · f_H), system I
**± 1 Hz** with a maximum rate of change of 0.1 Hz/s; N/PAL Argentina
**3 582 056.25 ± 5 Hz**; SECAM **f_OR = 4 406 250 ± 2000 Hz (282 f_H)** and
**f_OB = 4 250 000 ± 2000 Hz (272 f_H)**.

**Colorimetry.** NTSC 1953 primaries R 0.67/0.33, G 0.21/0.71, B 0.14/0.08 with
Illuminant C white **x 0.310, y 0.316**; 625-line PAL/SECAM R 0.64/0.33,
G 0.29/0.60, B 0.15/0.06 with D65 **x 0.313, y 0.329**. SMPTE 170M-2004 gives
the modern SMPTE C set: **R 0.630/0.340, G 0.310/0.595, B 0.155/0.070** with
D65 **x 0.3127, y 0.3290**. EBU Tech. 3213-E fixes the same 625-line primaries
and adds the tolerance quadrilaterals in both CIE 1931 and CIE 1960
coordinates, with a skin-tone acceptance test requiring
**√((Δu)² + (Δv)²) ≤ 0.003** in CIE 1960 units.

**Chroma-luma timing.** BT.470-6 requires **luminance/chrominance
time-coincidence error < 0.05 µs**. SECAM: HF pre-emphasis peak-to-peak
**2M₀ = 23 ± 2.5%** of the luminance amplitude with **f₀ = 4286 kHz ± 20 kHz**;
LF pre-correction **f₁ = 85 kHz**; deviations Δf_OR 280 ± 9 kHz and
Δf_OB 230 ± 7 kHz at nominal, with maxima +350/−506 kHz and +506/−350 kHz.

**Group-delay pre-correction (BT.470-6 figure 3a).** The transmitter
pre-distortion the receiver is expected to undo — a specified phase curve, and
the only one besides LaserDisc's in the collection:

| frequency | curve A | curve B |
|---|---|---|
| 0.25 MHz | — | +5 ± 0 ns |
| 1.00 MHz | +30 ± 50 ns | +53 ± 40 ns |
| 2.00 MHz | +60 ± 50 ns | +90 ± 40 ns |
| 3.00 MHz | +60 ± 50 ns | +75 ± 40 ns |
| 3.75 MHz | 0 ± 50 ns | 0 ± 40 ns |
| 4.43 MHz | −170 ± 35 ns | −170 ± 40 ns |
| 4.80 MHz | −260 ± 75 ns | −400 ± 90 ns |

At subcarrier, M/NTSC and M/PAL take **−170 ns nominal**. Numerous national
variants are tabulated.


## 5. Tape, head and transport physics

Consolidated from SMPTE 32M-2004 and JVC VTG82063 §1, §4 and §7. Every value
below is quoted as the document states it.

### 5.1 Format mechanics, all speeds and systems

| quantity | NTSC SP | NTSC LP | NTSC EP | PAL/SECAM SP | PAL/SECAM LP |
|---|---|---|---|---|---|
| tape width | 12.65 ± 0.01 mm | " | " | " | " |
| tape speed | 33.35 ± 0.5% mm/s | 16.67 ± 0.5% mm/s | 11.12 ± 0.5% mm/s | 23.39 ± 0.5% mm/s | 11.70 ± 0.5% mm/s |
| drum diameter | 62.00 ± 0.01 mm | " | " | " | " |
| writing speed | 5.80 m/s | — | 5.83 m/s | 4.85 m/s | 4.86 m/s |
| video track pitch | 0.058 mm | 0.029 mm | 0.019 mm | 0.049 mm | 0.024 mm |
| video track width | 0.058 mm | 0.029 mm | 0.019 mm | 0.049 mm | 0.024 mm |
| video recording area width | 10.60 mm | " | " | " | " |
| video effective width | 10.07 mm | " | " | " | " |
| video track centre from reference edge | 6.2 mm | 6.2 mm | 6.195 mm | 6.2 mm | 6.195 mm |
| head azimuth | +6° and −6°, ± 10′ | " | " | " | " |
| track angle, running | 5° 58′ 9.9″ | 5° 57′ 8.5″ | 5° 56′ 8.1″ ᵃ | 5° 57′ 50.3″ | 5° 56′ 58.8″ |
| track angle, stationary | 5° 56′ 7.4″ | " | " | " | " |
| audio/control head position X | 79.244 mm | 79.251 mm | 79.253 mm | 79.244 mm | 79.248 mm |
| control track width | 0.75 ± 0.10 mm | " | " | " | " |
| audio track width, mono | 1.0 ± 0.1 mm | " | " | " | " |
| audio track width, stereo (each) | 0.35 ± 0.05 mm | " | " | " | " |
| audio track reference line | 11.65 ± 0.05 mm | " | " | " | " |
| audio-to-audio guard band | 0.3 mm | " | " | " | " |
| track shift error | 0.014 mm | 0.010 mm | 0.010 mm | — | — |
| track bend, peak-to-peak | 0.014 mm | ≤ 0.010 mm | ≤ 0.010 mm | — | — |
| head-switch position | 5–8 H ahead of V-sync leading edge | " | " | " | " |
| tape back tension at drum entrance | 0.30–0.45 N (30–45 g) | " | " | " | " |
| FM audio track width (JVC 1-1-1) | min. 0.020 mm | — | — | 0.016–0.049 mm | — |

ᵃ SMPTE 32M table 3; JVC table 1-1-1 gives 5° 56′ 48.1″ for the same cell.

SMPTE 32M table 5 states the FM audio track width differently and by mode
rather than by system: SP, minimum of A₁ and A₂ 0.010 mm and maximum 0.029 mm;
LP, minimum of B 0.012 mm; EP, minimum of B 0.016 mm. The same table fixes the
**recording time sequence** — the FM audio is laid down 0 to 2 fields before
the video in SP, 1/3 to 2⅓ fields in LP, and 1⅓ to 3⅓ fields in EP — and the
azimuth relationship, audio opposite to video in SP and LP but the *same*
direction in EP.

Measurement conditions: SMPTE 32M 3.1.1–3.1.2 requires no transverse or
longitudinal tension, 20 °C ± 1 °C, 50 ± 2% RH. JVC's note under table 1-1-1
gives 20 °C ± 2 °C at 65 ± 5% RH, relaxable to 5–35 °C and 40–80% where not
essential to the judgement. **The two documents disagree on the reference
humidity** — 50% against 65% — which is worth carrying, since tape dimension
is humidity sensitive.

### 5.2 Magnetic media

| quantity | value | source |
|---|---|---|
| tape type | high-resolution video tape, e.g. cobalt iron-oxide | SMPTE 32M 3.2.1.3, 7.3.2 |
| coercivity, standard | approximately 50 × 10³ A/m | SMPTE 32M 3.2.1.4 |
| coercivity, high-performance | approximately 70 × 10³ A/m | SMPTE 32M 7.3.3 |
| coercivity, JVC statement | 600 oersted class (nominal), VHS and S-VHS | JVC 1.1.25, 1.1.26, 1.1.28 |
| tape thickness | 15.6 ± 0.5 µm (T-160); 19.0 +1/−2 µm (T-120…T-30) | SMPTE 32M table 1; JVC 1.1.25 |
| tape width fluctuation | not exceeding 6 µm | SMPTE 32M 3.2.1.2; JVC 1.1.25 |
| recorded wavelength, video FM | ≈1.3 µm at 4.4 MHz, ≈1.7 µm at 3.4 MHz (λ = V/F, V = 5.8 m/s) | JVC 7.2.1 |
| recording depth, video | about 0.3 µm | JVC 7.2.1 |
| video head gap | 0.3 µm | JVC 7.2.1 |
| recorded wavelength, Hi-Fi audio | ≈4 µm (ch 1, 1.3 MHz), ≈3.4 µm (ch 2, 1.7 MHz) | JVC 7.2.1 |
| FM audio azimuth | +30° 30′ and −30° 30′ | SMPTE 32M 5.4; JVC gives ±30° |
| leader/trailer thickness | 40 +5/−25 µm | SMPTE 32M 3.2.2.2; JVC 1.1.25 |
| leader/trailer transparency | > 50% over 800–950 nm | SMPTE 32M 3.2.2.3 |
| splice pull force | at least 30 N (JVC: more than 3 kg) | SMPTE 32M 3.2.2.4; JVC 1.1.25 |
| splice gap | less than 0.07 mm | SMPTE 32M 3.2.2.4 |

**Coercivity units.** 600 Oe converts to 47.7 kA/m, so JVC's "600 oersted
class" and SMPTE's "approximately 50 × 10³ A/m" are the same statement.
70 × 10³ A/m is 880 Oe.

### 5.3 Cassette and reel geometry

| quantity | value | source |
|---|---|---|
| reel outside diameter | 89 ± 0.2 mm | JVC 1.1.25 §3 |
| hub diameter, > 60 min | 26 ± 0.15 mm | JVC; SMPTE 32M table 1 |
| hub diameter, ≤ 60 min | 62 ± 0.2 mm (or 70 ± 0.2 mm if ≤ 30 min) | JVC; SMPTE 32M table 1 |
| E-value | more than 1.5 mm | JVC 1.1.25 §3 |
| tape length, NTSC | L = ⌈2.2t + 2⌉ m, +3/−0, t in minutes | JVC 1.1.25 §1 |
| tape length, PAL/SECAM | L = ⌈1.42t + 2⌉ m, +3/−0 | JVC 1.1.26 §1 |
| T-120 | 246 +3/−0 m, 120 min, hub 26 mm | SMPTE 32M table 1 |
| datum plane flatness | less than 0.2 mm across four planes | SMPTE 32M figure 2 note 5 |
| reel base height from datum | 2.0 +0.8/−0.5 mm | SMPTE 32M figure 4 note 2 |

### 5.4 Transport and servo rates

| quantity | NTSC | PAL/SECAM | source |
|---|---|---|---|
| drum rotation | 30 Hz | 25 Hz | JVC §4 |
| drum FG | 240 Hz | 200 Hz | JVC §4 (so 8 FG poles per revolution) |
| capstan FG, counted down to | 30 Hz | 25 Hz | JVC §4 |
| control pulse rate | 30 Hz | 25 Hz | JVC 1.1.21 |
| servo reference crystal | 3 f_sc = 10.74 MHz | 3 f_sc = 13.34 MHz | JVC §4 |
| control signal rise time | less than 200 µs | — | SMPTE 32M 3.11.3 |
| control pulse phase | positive edge coincident with start of video track 1 scan; drum-side pole north when positive | | SMPTE 32M 3.11.1–3.11.2; JVC 1.1.21 |

### 5.5 Signal chain constants, for completeness

| quantity | VHS NTSC / M-PAL / N-PAL | VHS PAL/SECAM | S-VHS (high-performance type H) |
|---|---|---|---|
| FM carrier, sync tip | 3.4 ± 0.1 MHz | 3.8 ± 0.1 MHz | 5.4 ± 0.1 MHz |
| FM carrier, peak white | 4.4 ± 0.1 MHz | 4.8 ± 0.1 MHz | 7.0 ± 0.1 MHz |
| deviation | 1.0 ± 0.1 MHz | 1.0 ± 0.1 MHz | 1.6 ± 0.1 MHz |
| white clip (from sync tip, 100% = sync to white) | 160% nom, 155% min, 200% max | " | 210% nom, 189% min, 231% max |
| dark clip | 40% nom ᵇ, −50% min, −30% max | " | −70% nom, −77% min, −63% max |
| luma baseband | ≈30 Hz to 3.0 MHz | " | ≈30 Hz to 5.0 MHz |
| chroma downconverted carrier | 629.371 kHz (40 f_H) | 626.953 kHz ᶜ | as VHS |
| chroma separation BPF | centre 3.58 MHz, −3 dB at 3.08 and 4.08 MHz | | |
| burst amplitude doubler | +6.0 ± 0.5 dB (SP only; not doubled in LP) | | |
| chroma phase rotation | track 1 +90°/line, track 2 −90°/line, completed before burst | | |
| chroma record level | played-back chroma 7–10 dB below saturation | | |
| luma separation filter | ≥ 40 dB attenuation at subcarrier | | |
| FM carrier interleave, LP/EP | ch 1 is f_H/2 above ch 2, f_H = 15.734 kHz | | |
| FM high-pass filter | T = C·Rb = 1.3 ± 0.05 µs; X = Rb/Ra = 4 ± 0.3 | | |

ᵇ SMPTE 32M 3.9.1.1.3 prints the dark clip nominal as "40%" between a −50%
minimum and a −30% maximum, which cannot be right; it is almost certainly
−40%. Recorded here as printed, and flagged.

ᶜ **Transcription defect.** JVC table 1-1-2 as transcribed gives the PAL
down-converted subcarrier as **625.953 kHz**. The correct value is
40.125 × 15625 = 626953.125 Hz, which is what `vhsdecode/format_defs/vhs.py`
line 154 computes (`((625 * 25) * 40) + 1953`) and what line 450 states in a
comment. The transcribed digit is wrong by 1 kHz. The N-PAL entry in the same
table reads 626.953 kHz, which suggests a column shift. Do not use the
collection's PAL figure.

For the high-performance system, SMPTE 32M 7.5.1.2.2 adds two more:
residual luma in the down-converted chroma must be attenuated more than 20 dB
in the vicinity of 1.2 MHz, and the chroma record level is set so that the
spurious component at f_y − 2f_c is 20–25 dB below the output at f_y, with
f_y = 6.5 MHz.

VHS Hi-Fi (SMPTE 32M clause 5): centre carriers 1.3 MHz ± 10 kHz (ch 1) and
1.7 MHz ± 10 kHz (ch 2) for NTSC — JVC gives 1.4 MHz and 1.8 MHz for
PAL/SECAM; maximum deviation ±150 kHz, reference deviation ±50 kHz at 400 Hz;
preemphasis t₁ = 56 ± 11 µs, t₂ = 20 ± 4 µs; noise reduction 2:1 logarithmic
compression, peak detection, attack 3–10 ms, recovery 70 ± 14 ms.

Longitudinal audio: reference level 100 nWb/m; deemphasis time constants
120 µs and 3180 µs (SP), 170 µs and 3180 µs (LP/EP).

LaserDisc (IEC 60857 NTSC / IEC 60856 PAL): main carrier at blanking
8.1 MHz ± 50 kHz (NTSC) and 7.1 MHz ± 50 kHz at 30% blanking (PAL); deviation
sync-to-white 1.7 MHz ± 35 kHz (NTSC) and 800 ± 20 kHz blanking-to-white
(PAL); audio subcarriers 146.25 f_H = 2 301 136 Hz and 178.75 f_H =
2 812 499 Hz (NTSC), each ±100 kHz for 100% modulation and peak deviation
below ±150 kHz, level −26 ± 1 dB relative to the unmodulated main carrier;
PAL pilot burst 240 f_H = 3.75 MHz superimposed on the sync level at 6/7 of
blanking-to-peak-white ± 10%, 13.5 ± 1 periods (3.6 µs nominal). The PAL audio
subcarriers are much lower: **43.75 f_H = 683 593.75 Hz** (left) and
**68.25 f_H = 1 066 406.25 Hz** (right).

LaserDisc also carries the collection's only **specified time-base error**, and
it is specified as a function of radius, which is the disc analogue of this
arc's tape-position axis:

| | LV-NTSC (IEC 60857) | LV-PAL (IEC 60856) |
|---|---|---|
| rotation | 1 revolution per TV frame (CAV) | same |
| CLV linear velocity | 10.1 to 11.4 m/s | 8.4 to 9.5 m/s |
| track pitch | min 1.4 µm, max 2 µm | same |
| time-base error, CAV | **10 µs p-p at radius 55 mm** (30 Hz roll-off, 12 dB/octave), **4 µs p-p at 145 mm** | **12 µs p-p at 55 mm** (25 Hz), **4.6 µs p-p at 145 mm** |
| shift between adjacent tracks | **± 25 ns** | ± 25 ns |
| optics | λ 6328 Å, NA 0.40 ± 0.01, reflectivity ≥ 70%, birefringence ≤ 20° | same |

The PAL disc also specifies its own multiburst frequencies —
**0.5, 1.3, 2.3, 4.2, 4.8, 5.8 MHz ± 2%** — which differ from the J.63 set,
and requires **lines 22 and 335 to be blanked before optical recording so that
disc noise can be measured**: a deliberately reserved noise-measurement window,
which is precisely the estimation hygiene this arc practises.


## 6. Capture, digitisation and RF measurement

This is the collection's thinnest area for this project's purposes, and the
absence should be stated plainly.

**What is there.**

- **SMPTE 244M-2003** specifies 4f_sc composite sampling for NTSC. It is
  directly comparable to what `vhsdecode/models/capture_profile.py` reads back
  out of a capture — a code lattice, an occupied fraction of full scale, and a
  quantisation floor:

  | quantity | value |
  |---|---|
  | sampling frequency | **4 f_sc = 14.31818 MHz** |
  | sampling phase | on the **I and Q axes, +123° and +33°**; SC/H in the digital domain shall be zero |
  | samples per line | **910**, of which **768 are digital active video** and **142 digital horizontal blanking** |
  | sync leading edge | the half-amplitude point of the falling edge falls **between samples 784 and 785** |
  | coding | uniformly quantised PCM at **10 bits**; 8-bit carried in the eight MSBs with the two LSBs zeroed |
  | white level | **C8h** (8-bit) / **320h** (10-bit) |
  | blanking level | **3Ch** / **0F0h** |
  | sync tip level | **04h** / **010h** |
  | permitted codes | **01h–FEh** (254 of 256) / **004–3FB** (1016 of 1024) |

  Its tables 4 and 5 print the complete word-by-word hexadecimal values of the
  equalising-pulse interval, the vertical serrations and the whole 142-word
  horizontal blanking interval including both burst phases — a fully specified
  synthetic sync pulse at sample resolution.

- **EBU Tech. 3280-E** does the same for 4f_sc PAL, and adds the one thing
  244M does not: an explicit **millivolt-to-code-value mapping**. Sampling
  **4 f_sc = 17.734475 MHz**, non-orthogonal and frame repetitive,
  **1135 + 4/625 samples per line**, **948 samples of digital active line**
  and 187 of digital blanking. Blanking **40h**, white **D3h**, sync **01h**;
  **903.3 mV = FE.C is the maximum value, 700.0 mV = D3.0 is peak white,
  0.0 mV = 40.0 is blanking, −300.0 mV = 01.0 is sync tip**, giving a stated
  **sync headroom of 0.14 dB** and **picture headroom of 0.23 dB**. Sampling
  phase is zero SC/H with samples at 45°, 135°, 225° and 315° relative to the
  +U axis; the structure advances **0.361 ns per line, four samples per frame**.

- **ITU-R BT.601-5** fixes component digital sampling: **13.5 MHz luma,
  6.75 MHz chroma**, 8 or optionally 10 bits, with **220 quantisation levels
  for luminance, black at 16 and peak white at 235**, **225 levels for each
  colour-difference signal with zero at 128**, and **levels 0 and 255 reserved
  for synchronisation**. 858 total samples per line at 525/60 and 864 at
  625/50, 720 samples of digital active line in both.
- **SMPTE 170M-2004** clause 15 fixes the analogue interface: unbalanced
  coaxial, source 75 Ω resistive, terminating 75 Ω resistive, cable 75 Ω
  nominal, preferred connector a 75 Ω BNC mating non-destructively with the
  50 Ω IEC 60169-8 type.
- **ITU-R BT.1439-1** Part 1 §2 defines nominal impedance and return loss as
  measurable equipment characteristics, and Part 2 gives the measurement
  methods for the parameters it defines.
- **EBU Tech. 3209** Appendix 2 defines *sub-standard input* stress tests for
  insertion equipment — signal level, noise, hum, tilt, drop-outs, flashing,
  noise conversion — which is the nearest thing in the collection to a
  degraded-source measurement procedure.
- **IEC 60461:2010** fixes the VITC waveform, which is a *specified data
  waveform inside the vertical interval on tape*: bit period
  **T_e = 1/(115 f_H) ± 2%**, **rise and fall 200 ± 50 ns (10–90%)**,
  amplitude distortion ≤ 5% of the code p-p, logical 1 at **70–90 IRE** and
  logical 0 at **0–10 IRE** for 525/59.94 (**500–600 mV** and **0–25 mV** for
  625/50), first bit no earlier than **10.0 µs** after the sync leading edge
  for 525 and **11.2 µs** for 625. Line placement: 525/59.94 not earlier than
  line 10(273) nor later than 20(283), preferred **14(277)** and **16(279)**;
  625/50 not earlier than 6(319) nor later than 22(335), preferred **19(332)**
  and **21(334)**. LTC by contrast is 80 bits at **80 × frame rate**, biphase
  mark, **rise/fall 40 ± 10 µs**, amplitude 1–2 V p-p preferred.

- **ANSI/CTA-608-E** fixes the line 21 waveform, which is on many VHS tapes:
  data rate exactly **32 × f_H**, so the bit period is
  **D = 1/(f_H × 32) = 1.986 µs**, adjusted to the instantaneous f_H of line 21;
  clock run-in is **7.0 cycles of a 0.5034965 MHz sine wave symmetrical about
  25 IRE**; **data bit high 50 IRE (encoder ± 2, decoder ± 12)** and **low
  0 IRE**; **rise/fall 0.240 µs nominal, 0.288 µs encoder maximum, 0.480 µs
  decoder limit**, specified as a 2T bar measured 10–90%; sync to clock run-in
  **10.500 ± 0.250 µs**; **no setup is allowed on line 21**; source-generated
  switching transients no greater than **± 2 IRE**.

**What is not there, and this project needs.**

- Nothing on **RF capture of tape**. No sampling rate guidance for an FM
  carrier, no bit-depth recommendation, no anti-alias requirement, no
  front-end level or impedance for a head-amplifier tap. The 50 MHz / 8-bit
  regime this arc works in has no reference in the collection.
- Nothing on **absolute RF levels**. Record current is specified only as
  "the optimum value"; playback level only as "approximately equal and
  maximum". So `ELLIPTICAL_COLLAPSE.md` §7.5's note that absolute levels
  assume a 1 Vpp full scale stays an assumption.
- No **carrier-to-noise or SNR specification** for VHS. Neither document
  states a required S/N, so the measured 21.6 dB envelope C/N and the 33.1 dB
  true-random figure have nothing to be compared against.
- No **test-equipment procedure** for a tape deck. SMPTE 32M specifies what
  the tape must carry, not how to measure a deck reproducing it. EBU Tech. 3209
  Appendix 2 is the closest analogue and it tests an inserter, not a VTR.
- **The filter templates are pictures.** BT.601-5's insertion-loss, passband
  ripple and passband group-delay templates for the 13.5 / 6.75 / 18 / 9 MHz
  filters are images with no extractable numbers, as are all of BT.1439-1's
  filter response curves. Only BT.1439-1's component-value tables survive as
  text. So the collection specifies *which* filter to use and *what parts it is
  made of*, but not, in most cases, the response mask it must meet.


## 7. Summary of what to fetch, in priority order

1. `docs/vhs/SMPTE-32M-2004/markdown.md` — the whole VHS format normatively,
   including the four sub-preemphasis tables and the track geometry for every
   speed.
2. `resources/definitions/vits/**` — eleven machine-readable test signals,
   the exact synthetic side for a component-fitting method.
2a. `docs/video_formats/SMPTE-170M-2004/SMPTE-170M-2004.md` Annex A.6 — the
   75% and 100% colour bar tables, which are the synthetic side of the capture
   `elliptical_collapse.py` is already run on.
3. `docs/vhs/JVC-Video-Technical-Guide-VTG82063/section-1.md` — everything
   SMPTE 32M omits: PAL, SECAM, M-PAL, N-PAL, S-VHS, and the cassette.
4. `docs/video_metadata/T-REC-J-63-1990/T-REC-J-63-1990.md` — the normative
   source behind the VITS definitions, complete in text: every element
   amplitude, position in H/32 or H/128, and half-amplitude duration with its
   tolerance.
5. `docs/video_formats/SMPTE-170M-2004/SMPTE-170M-2004.md` clauses 11–13 — the
   NTSC sync pulse specified to ± 0.10 µs in width and ± 20 ns in edge rate,
   which is what the sync-only constraint measures against.
6. `docs/video_metadata/EBU-Tech-3209/EBU-Tech-3209.md` — per-element
   tolerances for the same signals, and the degraded-input test procedures.
7. `docs/video_metadata/R-REC-BT-1439-1-2006/R-REC-BT-1439-1-2006.md` — the
   measurement vocabulary, linear distortion split by time scale, the K-rating
   formulae, and the weighting network's 7.4 / 12.2 dB noise factors.
8. `docs/video_formats/BT-624-4-1990/BT-624-4-1990.md` (in preference to
   BT.470-6, which is the same content with two OCR errors) — line and field
   sync geometry for every 625-line system, subcarriers to full precision, and
   the transmitter group-delay pre-correction curves.
9. `docs/vhs/JVC-Video-Technical-Guide-VTG82063/section-7.md` — one paragraph
   in §7.2.1 carrying the recording depth and head gap.
10. `docs/laserdisc/IEC-60857-1986-Laservision-NTSC/IEC-60857-1986-Laservision-NTSC.md`
   and its PAL twin — specified group delay, and a time-base error specified
   as a function of radius.
11. `docs/video_formats/SMPTE-244M-2003/SMPTE-244M-2003.md` and
   `docs/video_formats/EBU-Tech-3280-E/EBU-Tech-3280-E.md` — the quantisation
   scales tied to video levels, and in 3280's case to millivolts.


## 8. Defects found in the transcriptions

Recorded so they are not propagated:

1. **JVC table 1-1-2, PAL down-converted colour subcarrier: 625.953 kHz.**
   Should be 626.953 kHz (40.125 f_H). Off by 1 kHz.
2. **SMPTE 32M 3.9.1.1.3, dark clipping nominal: "40%"** between a −50%
   minimum and a −30% maximum. Almost certainly −40%.
3. **NTSC EP running track angle disagrees between the two documents**:
   5° 56′ 8.1″ (SMPTE 32M table 3) against 5° 56′ 48.1″ (JVC table 1-1-1).
4. **Reference humidity disagrees**: 50 ± 2% (SMPTE 32M 3.1.2) against
   65 ± 5% (JVC table 1-1-1 note).
5. **S-VHS tape coercivity disagrees, and JVC's figure looks copied.**
   SMPTE 32M 7.3.3 gives 70 × 10³ A/m (≈880 Oe) for high-performance type H,
   whereas JVC 1.1.28 gives S-VHS tape as "600 oersted class (nominal)" —
   identical to its standard-VHS entry — and adds "optimum recording current
   shall not differ from the standard tape", which is implausible for a
   higher-coercivity formulation. Prefer SMPTE 32M here; treat JVC 1.1.28's
   coercivity line as a carried-over paragraph.
6. **BT.470-6 gives the M/PAL subcarrier as 3 579 611.49 Hz.** It should be
   **3 575 611.49 Hz**, which is what CCIR 624-4 and ITU-T J.63 both print, and
   what JVC table 1-1-2 gives as 3.575611 MHz. Use 624-4 or J.63.
7. **BT.470-6 gives the SECAM D_R′ field-identification amplitude as
   500 +40/−40 mV**; CCIR 624-4 gives **540 +40/−50 mV**. Prefer 624-4.
8. **EBU Tech. 3280-E's excluded-code list mis-renders 00.C_h as 00.0_h and
   FF.C_h as FF.0_h**, contradicted by the level table in the same document.
9. **SMPTE 244M-2003 table 5 labels a sync-leading-edge row "876"** where the
   sequence 783, 784, 785, **786** requires 786; word 876 also appears
   correctly elsewhere in the same table.
10. **J.63 §2.4.2 reads "inherent differential-gain distortion ≤ 0.2°"** where
    differential *phase* is plainly meant.
11. **SMPTE 170M-2004** heads §11.3 "Field frequency (horizontal)" and prints
    Table A.3 as "Table 3". Numbers are otherwise clean.
12. Several documents are largely figures. SMPTE 32M's preemphasis
   characteristic (figure 9), its FM high-pass response (figure 10) and its
   vertical-emphasis curves (figures A.1, A.2) are images with no extractable
   numbers; only the RC constants printed beside figure 10 survive. The same
   is true of the JVC guide's preemphasis, clip-level and spectrum figures.
   **Anything specified only "as shown in figure N" is not available from this
   collection.**
