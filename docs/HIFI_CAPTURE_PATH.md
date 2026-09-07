# Reaching the Hi-Fi carriers

---

> ## CORRECTION, 2026-09-07 — read this before section 2
>
> **Two of the five records section 2 reports on carry no Hi-Fi audio.**
> Ethan, 2026-09-06: *"These sample may not have hifi audio. The zaroff
> samples should have hifi carriers, home and cd will not."*
>
> So the `countdown` and `home` rows searched for a signal that was never
> recorded, and **their −50 dB and −45 dB bounds are withdrawn** — an absent
> signal found absent is not a bound on leakage. Four of the ten
> (record, channel) pairs in the results table are those two records, and the
> verdict line *"no AFM carrier reaches the video tap in any capture in this
> repository"* overstates what the valid subjects support.
>
> **What survives, and it is weaker than what was withdrawn.** The zaroff
> records are the only valid subjects. The **record-tap control is
> unaffected and is the strongest part of the original measurement**, because
> the audio carriers go to separate heads (clause 5.1) and can never appear
> at the video record-current tap.
>
> Re-measured over **all 32 zaroff playback captures and all 32 record
> captures** — 128 (file, channel) pairs against the one programme sampled
> below, with the published SP row reproduced to four figures first — three
> things changed:
>
> 1. **The narrow-line statistic is not safe.** It divides a peak by the
>    scatter of its own neighbourhood, so a flat neighbourhood manufactures
>    significance. Ten of 64 playback pairs cross 3σ and none of 64
>    record-tap control pairs does — yet the control shows *larger* bumps
>    (up to 27.1 dB over its floor against the firing pairs' 11.2 dB median)
>    and only its rougher surround stops it firing. Ensembled, the AFM-free
>    record tap peaks *higher* than the playback tap at channel 1
>    (+15.70 dB against +9.57 dB). **No candidate survives the control.**
> 2. **The colour-under second-harmonic confound named in section 2 is
>    misattributed.** Tested on the chrominance-free y-only recordings:
>    removing 18.93 dB of colour-under fundamental removed **0.14 dB** at
>    1.258741 MHz, where a square-law product would have lost about 38 dB.
>    What sits there is the luma lower-sideband continuum and tape noise.
>    Excluding it by name buys nothing.
> 3. **The operative statistic reverses.** Section 2 argues that no audio was
>    connected, so the carrier is unmodulated and the narrow line is the
>    bound to quote. Clause **5.6**'s 2:1 logarithmic compressor is at
>    maximum gain on silence, so noise on an open input is amplified before
>    it modulates the carrier. Planted at −45 dB, an unmodulated carrier is
>    detected (line rise +3.033) and one with 500 Hz of deviation is not
>    (+2.744) — and 500 Hz is what that compressor makes of an input 84 dB
>    down. The hump statistic is nearly indifferent to deviation
>    (rise 2.556 to 2.697 over the whole range) and is the conservative one.
>
> **THE RESTATED BOUND.** On the only records that could carry a carrier, no
> audio carrier is attributable to the video tap at either frequency. The
> figure to quote is the **hump** bound: **−40 dB relative to the luma band
> on channel 1** for the chroma-bearing SP record, **−45 dB** for the y-only
> SP record, and **−35 dB on channel 2** at both speeds — channel 2 is 10 dB
> weaker than channel 1 throughout. The repository's tightest supported
> figure moves from the withdrawn −50 dB to **−45 dB**.
>
> The numbers below are left exactly as measured rather than edited, because
> this arc's rule is that a withdrawn claim stays visible with its reason.
> The restatement lives in `capture_alignment.carrier_route_bound`, the
> withdrawal in `capture_alignment.WITHDRAWN`, the wider sweep in
> `capture_alignment.WIDE_SWEEP`, and what would lower the bound (chiefly
> **+27 dB for time-base correcting the record before the transform**) in
> `capture_alignment.bound_lowering` and **docs/CAPTURE_ALIGNMENT.md** §1.4.

---

Ethan's question, verbatim: *"Capture question unresolved: cxadc exposes raw
mode only on the video ADC. Scope whether a wideband tap ahead of the
video/audio split gets both, or whether the second ADC's raw path can be
routed at register level. Service manual has the divider ratios and split
topology."*

There are three separable questions and they have three different answers.

1. **Is there a tap on the deck ahead of the video/audio split?** No, and
   not on any deck built to the standard. Verified from the schematic.
2. **Do the AFM carriers reach the video tap anyway, by crosstalk?** No, to
   a measured bound. Verified against five captures with a control.
3. **Can one cxadc card carry both?** No. The raw path is the video ADC
   only, `audsel` is not what its name suggests, and the project's own
   answer is two clock-synchronised cards. The register-level question about
   the CX2388x's *audio* ADC could not be settled here and is marked
   unverified.

Every claim below is labelled **verified** (read from a source named on the
line) or **unverified** (stated, with what would settle it).

---

## 1. The split is at the heads

**Verified — SMPTE 32M-2004 clause 5.1**, fetched from
`decode-orc/analogue-video-specifications`, `docs/vhs/SMPTE-32M-2004`:

> "The FM audio signal shall be recorded into the depth of the magnetic
> layer of the video tape. It shall be recorded at a specified number of TV
> fields earlier in time than the associated video signal and in the video
> track area. **Separate FM audio heads shall be mounted on the scanning
> drum for this purpose.** The video signal shall be subsequently recorded
> on the surface of the same area of the video tape. This system is known as
> depth multiplex recording. The FM audio and video head gaps shall be of
> different azimuth angles to minimize crosstalk between their respective
> recordings."

So the separation is not a design choice a manufacturer makes downstream of
a common amplifier — it is *at the heads*, mandated by the format. There is
no point in the signal chain where a single wire carries both, because the
two signals are never on the same wire.

**Verified — Sony SLV-777HF/778HF/788HF service manual schematic**
(`/testdata/test_patterns/vhs/Sony Slv777Hf 778Hf 788Hf Schematic.pdf`,
11 PDF pages; printed page numbers in brackets). The deck that made every
test-pattern capture in this repository, read at 350–600 dpi on 2026-09-05:

| Item | Where | What |
| --- | --- | --- |
| `CN260`, 13 pins | PDF 1 [4-5/4-6/4-7], MA-327 (1/8) | The drum's rotary-transformer windings. Pins 6–13 are the four **video** heads (SP CH2 S/F, EP CH1 S/F, SP CH1 S/F, EP CH2 S/F); pins 1–3 are `AUDIO CH2`, `AUDIO REC`, `AUDIO CH1`; pin 4 `FE` (flying erase); pin 5 `GND(SW)`. |
| `IC260` Hitachi HA118195NT | same sheet | `VIDEO REC/PB AMP`. The **only** destination of CN260 pins 6–13. |
| `IC340` Sanyo LA7256 | same sheet | `AFM REC/PB AMP`, with its own Ch1/Ch2 head amplifiers, SP/EP switching, record-current amplifier and AGC. The **only** destination of CN260 pins 1–3. |
| `CN261`, 4 pins | same sheet | The **video** test connector, `FOR CHECK`: 1 `REC CURR`, 2 `PB RF`, 3 `RF SWP`, 4 `GND`. |
| `CN341`, 3 pins | same sheet | The **AFM** test connector, separate: 1 `HF ADJ`, 2 `GND`, 3 `FM PB`. |

Two connectors, two amplifiers, two sets of windings, two test points. The
zaroff captures were taken at `CN261` pins 1 and 2 into a Rigol DS1202Z-E at
50 MS/s, 8 bit, with no buffer or amplifier
(`/testdata/test_patterns/vhs/readme.txt`), which is the **video** side.

**Conclusion (verified):** there is no wideband tap ahead of the split,
because there is no "ahead of the split". `CN341` pin 3 `FM PB` is the AFM
equivalent of `CN261` pin 2, and a second capture channel there is what a
Hi-Fi timing probe on this deck would require.

---

## 2. The carriers are not on the video tap — measured

The split being at the heads still leaves crosstalk: a video head reading
the buried audio layer through the azimuth difference, or coupling in the
rotary transformer or between the two amplifiers. That is a measurement,
and it was made.

### What was measured

`vhsdecode/models/hifi_carriers.py`, `detection_bound` — for each record, at
1.3 MHz and 1.7 MHz (SMPTE 32M 5.8):

* the **narrow-line** statistic: the tallest bin inside the ±10 kHz centre
  tolerance, against the tallest a noise neighbourhood of that many bins
  would show, in units of the neighbourhood's scatter. This is the
  statistic for an *unmodulated* carrier, which is what a deck lays down
  with no audio input;
* the **hump** statistic: the mean density over ±70 kHz (the reference
  deviation of 5.9 plus the top of the audio band) over the same floor,
  divided by its standard error. This is the statistic for a *modulated*
  carrier;
* a **calibration**: unmodulated and reference-modulated carriers are
  planted on the record itself at a ladder of levels relative to that
  record's own luma-band power, and the *bound* is the weakest planted level
  at which the statistic rises 3σ above what the unplanted record shows.
  Planting on top of whatever confounds are already present is what stops
  structured neighbours being mistaken for sensitivity.

The floor is a **median** over ±180 kHz excluding the hump band, which
rejects the line-harmonic picket fence. The colour-under's second harmonic
is excluded by name (below).

### The confounds, named before the result

| Confound | Where it lands at 1.3 MHz | Where it lands at 1.7 MHz |
| --- | --- | --- |
| Line harmonics, 15.734 kHz apart (SMPTE 32M 4.3.1.2) | 83rd at 1.305944 MHz, **inside** the ±10 kHz window | 108th at 1.699301 MHz, 699 Hz away, **inside** the window |
| Colour-under harmonics of 629.371 kHz (3.9.2.1.4) | 2nd at 1.258741 MHz, 41.26 kHz below: outside the window, **inside** the ±70 kHz hump band | 3rd at 1.888 MHz, 188 kHz away: outside even the ±180 kHz reach |
| Luma lower sideband (band 2.4–5.4 MHz, from 3.9.1.1.4) | a continuum, not a line: it enters the floor | same |

A picket falls inside the tolerance window at *both* nominal carriers, so
the narrow-line statistic is read together with the peak's distance from the
picket grid (`line_harmonic_offset_hz`). The colour-under second harmonic is
removed from the hump band and the floor by name
(`carrier_presence(..., exclude=...)`).

### The records

| Record | Tap | Rate | Window |
| --- | --- | --- | --- |
| `zaroff-ntc7composite-NTSC-SP-…-rf-pb` | CN261 pin 2, playback | 50 MSps | whole 0.480 s |
| `zaroff-ntc7composite-NTSC-EP-…-rf-pb` | CN261 pin 2, playback | 50 MSps | whole 0.480 s |
| `zaroff-ntc7composite-NTSC-SP-…-rf-rec` | CN261 pin 1, **record current — the control** | 50 MSps | whole 0.480 s |
| `/testdata/countdown.flac` | video tap, 2 h tape | 40 MSps | 0.5 s from 16.68 s |
| `/testdata/home.flac` | video tap, 2 h tape | 40 MSps | 0.5 s from 16.68 s |

The zaroff captures are the SLV-778HF's own recordings of a Tektronix
TSG-130A with **no audio connected**, so any carrier the deck laid down
would be unmodulated and the narrow-line bound is the operative one. The recording deck and the
audio content of the two 2-hour tapes are **not recorded anywhere in this
repository**, so a modulated carrier must be allowed for and the hump bound
is theirs.

Sample rates: the zaroff files' FLAC headers read 50 000 Hz and 24 000 512
frames — 0.480 s at 50 MSps, matching the readme's "horizontal timebase to
10 ms for a sample rate of 50 MS/s". `countdown.flac` and `home.flac` read
40 000 Hz: the *number* is right and the unit is wrong by a factor of 1000,
so they are 40 MSps, consistent with the recorded decode flags `-f 40`.
`home.flac` carries no seek table (`libsndfile psf_fseek` fails past frame
0), so its window is reached by reading and discarding.

### The result

Unplanted statistics, and the bound in dB relative to that record's own
luma-band power. `DETECTION_SIGMA` is 3.

| Record | Ch | line σ | hump σ | peak − carrier | peak − picket | floor (dB re luma) | line bound | hump bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zaroff SP playback | 1 | −0.53 | 7.59 | +812 Hz | −5132 Hz | −43.6 | **−45 dB** | −40 dB |
| zaroff SP playback | 2 | +0.12 | 4.11 | −5511 Hz | −4812 Hz | −42.7 | −35 dB | −35 dB |
| zaroff EP playback | 1 | +0.66 | 8.59 | −1477 Hz | −7421 Hz | −36.0 | −40 dB | −40 dB |
| zaroff EP playback | 2 | +1.02 | 7.39 | +2118 Hz | +2817 Hz | −36.3 | −35 dB | −35 dB |
| zaroff SP **record** | 1 | −0.55 | 7.79 | +812 Hz | −5132 Hz | −44.5 | −30 dB | −35 dB |
| zaroff SP **record** | 2 | +0.41 | 4.76 | −5511 Hz | −4812 Hz | −43.1 | −20 dB | −30 dB |
| countdown | 1 | −0.02 | 4.33 | +1880 Hz | −4064 Hz | −36.1 | **−50 dB** | −45 dB |
| countdown | 2 | −2.64 | −7.00 | −1392 Hz | −692 Hz | −33.6 | **−50 dB** | −45 dB |
| home | 1 | −2.62 | −2.90 | +1880 Hz | −4064 Hz | −38.9 | **−50 dB** | −45 dB |
| home | 2 | −1.73 | +1.08 | −9937 Hz | +6497 Hz | −34.1 | −45 dB | −45 dB |

**No narrow line anywhere.** `line_sigma` runs −2.64 to +1.02 across all ten
(record, channel) pairs, against a threshold of 3. On the deck's own
recordings, where an unmodulated carrier is exactly what would be there, a
carrier 45 dB below the luma band would have been seen at channel 1 and was
not.

**The hump excess is not a carrier — the control says so.** The unplanted
`hump_sigma` is 4 to 8.6 on the zaroff records, which looks like a
detection. It is not. A ±70 kHz band inside a ±180 kHz reach sits above the
surround's median whenever the spectrum has curvature across that span,
which the video RF does. The control settles it: the **record-current**
capture (CN261 pin 1) cannot carry AFM at all — the AFM record signal goes
to IC340 and the audio windings, never to this pin — and it shows

* the *same* hump excess: 7.79 and 4.76 against the playback's 7.59 and
  4.11, the playback **lower** by 0.20 and 0.65;
* the *same* strongest bins: 1 300 811.8 Hz and 1 694 488.5 Hz in both.

There is therefore no playback-only component at either carrier. On
`countdown` and `home` the hump statistic is 4.33 / −7.00 and −2.90 / +1.08:
no consistent excess either.

**Verdict (verified): no AFM carrier reaches the video tap in any capture in
this repository, to a bound of −45 dB (unmodulated, channel 1, the deck's own
SP recording) and −45 dB (reference-modulated, both channels, the 2-hour
tapes) relative to the luma-band power.**

### The crosstalk expectation, and why the comparison is only half-verified

`azimuth_crosstalk_db` computes the two mechanisms the depth multiplex
relies on, in `head_model`'s own forms: the azimuth loss
`sinc(w tan θ / λ)` across the audio track width (SMPTE 32M table 5:
0.010–0.029 mm in SP) at the relative azimuth (30° 30′ from 5.4, plus VHS's
own ±6°, opposite directions in SP so 36.5°), and the spacing loss
`exp(−2πd/λ)` through the video layer recorded over the audio. In SP at
1.3 MHz that is −15.5 to −28.6 dB of azimuth loss and −3.7 dB of spacing
loss: **19 to 32 dB in all**; at 1.7 MHz, 27 to 33 dB; in EP, 19 to 29 dB.

That envelope is **relative to what a head at the audio azimuth reads**, and
the bounds above are relative to the **luma-band power at the video head**.
Bridging the two needs the AFM RF level at the audio head against the luma
RF level at the video head. **Unverified: no capture of `CN341` pin 3
`FM PB` exists in this repository, so that ratio is unmeasured.** The two
numbers are therefore stated side by side and not subtracted. What would
settle it is one capture at `CN341` pin 3, taken the same way as the zaroff
ones.

---

## 3. What cxadc exposes

**Verified by reading the driver source.** The sources are *not installed on
this machine* — this is a devcontainer with no `/dev/cxadc*`, no `modinfo`
and no `dkms`. `cxadc.c` (1475 lines, version 0.5) and its `README.md` were
fetched from `happycube/cxadc-linux3` and a second fetch is byte-identical
to the first, so what is quoted here is upstream as of 2026-09-05.

### The raw path is the video ADC, and there is one of it

* `README.md` line 6: *"The new driver configures the CX2388x to capture in
  its raw output mode in 8-bit or 16-bit unsigned samples **from the video
  input ports**, allowing these cards to be used as a low-cost 28-54mhz
  10bit ADC for SDR and similar applications."*
* `cxadc.c`, `cxadc_char_open()`: the source is selected with
  `cx_write(MO_INPUT_FORMAT, (ctd->vmux<<14)|(1<<13)|0x01|0x10|0x10000)`.
  `vmux` is masked to `&3` — one of four **video** mux inputs into the one
  video ADC. Selecting a different `vmux` changes which pin the single ADC
  looks at; it does not give a second stream.
* The capture is `cx_write(MO_CAPTURE_CTRL, …)` with the raw bits, and the
  DMA is the **VBI channel**: `CHN24_CMDS_BASE 0x180100`, `MO_DMA24_CNT1`,
  `MO_VBI_GPCNT`, a 64 MB ring (`VBI_DMA_BUFF_SIZE`). One RISC program, one
  ring, one character device `/dev/cxadcN` per card.
* The sample rate comes from `MO_SCONV_REG` and `MO_PLL_REG`, computed from
  the card's crystal (`crystal`, default 28 636 363; `tenxfsc` selects
  28.6 / 35.8 / 40 MSps on a stock card, or an arbitrary rate on a
  crystal-modified one).

### `audsel` is not an audio ADC path

The parameter's name invites the guess that it selects a second converter.
It does not. `cxadc.c` lines 1134–1143 and 1388–1397:

```c
if (ctd->audsel != -1) {
        /*
         * Pixelview PlayTVPro Ultracard specific
         * select which output is redirected to audio output jack
         * GPIO bit 3 is to enable 4052 , bit 0-1 4052's AB
         */
        cx_write(MO_GP3_IO, 1<<25); /* use as 24 bit GPIO/GPOE */
        cx_write(MO_GP1_IO, 0x0b);
        cx_write(MO_GP0_IO, ctd->audsel&3);
}
```

It drives three GPIO registers to switch a **4052 analogue multiplexer on
the card**, routing an analogue signal to the card's audio **output jack**.
`README.md`: *"Some TV cards (e.g. the PixelView PlayTV Pro Ultra) have an
external multiplexer attached to the CX2388x's GPIO pins to select an audio
channel."* Nothing it selects enters the DMA stream. It is analogue routing
to a socket, not a capture path.

### The register-level question about the chip's audio ADC

**Verified:** cxadc writes no audio-side register at all. Every `cx_write`
in the driver is a video front-end, VBI-DMA, GPIO, I²C or interrupt-mask
register; there is no audio DMA channel, no audio front-end configuration,
and no second RISC program.

**Unverified:** whether the CX2388x's *audio* ADC could be pointed at a DMA
channel and streamed raw at all. Settling that needs the CX2388x datasheet
(the audio front-end and audio-DMA register map), which is **not on this
machine** and was not consulted. Two further things would have to hold even
if the register path existed, and neither is verified here: that the audio
ADC's sample rate can be raised to something useful for a 1.3–1.7 MHz
carrier — the audio front end is built for baseband audio, so its converter
and anti-alias filtering are unlikely to reach 3.4 MSps, let alone the
≥ 4 MSps that ±180 kHz of reach around 1.7 MHz wants — and that its input
mux can be fed from an RF tap rather than the card's audio input.

Given the answer in the next section is available, cheap and already in use,
this line of enquiry looks like the wrong one to spend effort on; but it is
open, not closed.

### What the project actually does: two clock-synchronised cards

**Verified** from the cxadc wiki's *Modifications* page (fetched to the same
scratchpad):

> "**Clockgen Mod — External Clock** … Software Defined Crystal Rate;
> Improved Signal to Noise; **Hardware Synchronised Multi-Card Capture**;
> Hardware Synchronised Linear/Baseband Audio Capture. The most advanced mod
> available today, using an external crystal timing source we can have drift
> free capture with 2-4 CX Cards and audio feeds. **This allows duel channel
> video formats such as Betacam to be captured, or Video FM + HiFi FM +
> Linear for many other formats like VHS or Betamax.**"

The parts named are a Raspberry Pi Pico as controller, an Adafruit Si5351A
generating the shared clock with up to five outputs, and a PCM1802 for
linear/reference audio, with an SMA clock input on a jig board. So the
established answer to "how do I capture video FM and Hi-Fi FM together" is:
**two CX cards on one clock** — card 1 on `CN261` pin 2, card 2 on `CN341`
pin 3 — and the shared clock is what makes the two streams comparable sample
by sample, which is exactly what a cross-check between a video-derived and a
Hi-Fi-derived time base needs.

**Unverified:** the residual timing offset between two clock-synchronised
cards. A common clock removes drift but not a fixed skew, and a time-base
cross-check tolerates a constant offset (`cross_check` reports it as
`intercept`) while a *drifting* one would show up as a spurious component.
This has not been measured here.

---

## 4. What this means for the Hi-Fi timing probe

* The probe is **not reachable from any capture in this repository.** Every
  capture is the video tap; the AFM carriers are not on it, to −45 dB.
  `time_base_error` and `time_base_error_pair` are exercised only on
  synthesis until a `CN341` capture exists.
* The premise that makes the probe attractive — *"carriers derive from the
  same crystal as video at different divider ratios, so the ratio is fixed
  by design"* — **does not hold on this deck**, and the format does not
  require it. SMPTE 32M clause 5 specifies the two centre frequencies and
  their ±10 kHz tolerance and says nothing about how they are generated; the
  words crystal, oscillator and divider do not appear in it. On the
  SLV-777HF/778HF/788HF the carriers come from two **RC-timed VCOs** inside
  IC360 (Philips TDA9615H/N1-557, 44 pins, `AFM AUDIO PROCESS`, PDF page 5
  [4-17/4-18], MA-327 (5/8)): C399 and C398 3900 pF at the two VCO pins,
  R385/R384 56 kΩ, R372/R370 4.7 kΩ on pins 23–33, R371 39 kΩ ±0.5 % at
  pin 28 `I ref` with pin 27 to ground, and I²C configuration on pins 41/42.
  **No crystal and no clock line crosses IC360's boundary** — verified by
  reading the whole sheet at 350 and 600 dpi; the only signals entering or
  leaving are `FM REC`/`FM PB` to the head amplifier (page 1/8),
  `AF ENV` / `AF SWP` / `I2C DATA 1` / `I2C CLOCK 1` to the servo and system
  controller (page 3/8), the baseband audio on pins 1–22, 12 V on pin 35 and
  `VSS D` on pin 43. The video processor IC201 has its own separate
  3.579545 MHz crystal X202 (PDF 2 [4-8/4-9/4-10], pin 56 waveform), and the
  servo controller IC160 its own 16 MHz and 32.768 kHz crystals (PDF 3
  [4-11/4-12/4-13], pins 38 and 41).
* What survives, and is what the probe is actually for: an RC oscillator
  drifts thermally over minutes, so everything at the transport's rates —
  the drum at 29.97 Hz, the capstan near 4.8 Hz, their harmonics — is
  **mechanical**, while the mean departure from nominal is **electronic**
  (the recording machine's oscillator setting). `time_base_error` splits the
  two and reports the mean separately as `electronic_offset`, rather than
  reading it as a speed error. The *ratio* of the two carriers is a
  per-recording constant to be measured, not 17:13 by construction, and
  `time_base_error_pair` returns it as `measured_ratio` beside
  `nominal_ratio`.
* The third wavelength cluster the directive wants for loss separation is
  real and unaffected by any of this, because it is a property of the
  format's numbers rather than of the capture: at the 5.80 m/s writing speed
  the AFM carriers record at **4.4615 µm and 3.4118 µm**, read from depths
  λ/2π of **0.710 µm and 0.543 µm**, between the colour-under's 9.216 µm
  (1.466 µm depth) and the luma's 1.706/1.318 µm (0.272/0.210 µm).

---

## 5. What would settle the open items

| Open item | What settles it |
| --- | --- |
| The AFM RF level at the audio head against the luma RF at the video head | One capture at `CN341` pin 3 `FM PB`, taken as the zaroff ones were (readme's procedure, 50 MS/s, 8 bit, no buffer). This also turns the crosstalk envelope into a testable prediction. |
| Whether the CX2388x audio ADC has a raw DMA path | The CX2388x datasheet's audio front-end and audio-DMA register map. Not on this machine. |
| Residual skew between two clock-synchronised CX cards | A simultaneous capture of one signal split to both cards, cross-correlated. |
| Whether an AFM carrier is present but *modulated past the ±70 kHz hump band* on the 2-hour tapes | The hump bound quoted here is for a carrier at the **reference** deviation of 5.9 (±50 kHz at 400 Hz). A carrier driven towards the ±150 kHz maximum spreads wider and the hump statistic is less sensitive to it, so those bounds are conservative for loud passages. The narrow-line bound covers quiet ones. A longer window containing a known-silent interval would tighten this. |
