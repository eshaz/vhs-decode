# The VHS luma analog chain — stage inventory and measured-signature mapping

This is the physical grounding document for the analog channel model in
`hsync_artifact_model.py` (Ethan's directive, Aug 27 2026: model the measurements as the
output of an analog filter chain, then reverse that chain's ringing effects). Researched from
the patent/spec literature plus this repository's own decoder chain; measured numbers come
from the three reference captures (cd, home, pnb — NTSC VHS SP).

## 0. Scope: what is actually inside the measured channel response

The tapes are decoded from **RF captures**, so the measured response
H(f) = (record deck's entire luma record chain) × (tape/head channel) × (capture deck's head
+ rotary transformer + preamp up to the RF tap) × (vhs-decode's software chain), divided by
what the software already inverts. The playback deck's baseband stages — post-demod low-pass,
de-emphasis, detail enhancer, noise canceller, combs — are **bypassed by RF capture** and can
only appear on tapes that were dubbed through a VCR's baseband path at some point.

What vhs-decode already inverts or imposes (NTSC VHS SP defaults):

- **Main linear de-emphasis — inverted.** IEC 774-1: RC = 1.3 µs with a 4:1 divider → shelf
  gain 5 (`format_defs/vhs.py`). Equivalent record pre-emphasis: zero 122 kHz, pole 612 kHz,
  +14 dB HF shelf.
- **Non-linear (dynamic) de-emphasis — NOT inverted at SP** unless `--non_linear_deemphasis`
  is passed; the 820 kHz corner in the source is flagged as a guess.
- **Software RF filters imposed zero-phase** (`process.py`): Butterworth BPF 0.5–6.5 MHz
  order 8; LPF 6.0 MHz order 25; **HPF 1.2 MHz order 20** (`video_hpf_extra`); RF peaking
  biquad 3.9 MHz gain 4 (emulating deck playback peaking).
- **Post-demod**: supergauss 6.6 MHz — effectively wide open; the deck-style ~3 MHz output
  LPF is not emulated; demod is Hilbert instantaneous-frequency (no hardware demod
  artifacts).
- Already documented in this branch (`process.py` ~line 1042): a real recorder (SLV-778HF)
  measured **1.45–1.50 µs** actual emphasis time constant against the 1.30 µs spec value the
  software inverts.

## 1. Record side (baked into every copy of the tape)

| Stage | Character | Spec-fixed? | In-band ring? |
|---|---|---|---|
| Input luma LPF ~3.0–3.4 MHz | LC ladder, delay-equalized, max attenuation at f_sc (elliptic-like); Philips US4803549 | Placement universal, ripple model-specific | 2.5–3 MHz, short |
| Main pre-emphasis 1.3 µs / 4:1 | First-order shelf, zero 122 kHz pole 612 kHz, +14 dB | Spec (tolerance 1.25–1.35 µs; real decks drift to 1.45–1.5) | No ring; ×5 HF boost makes every edge FM-wideband |
| Non-linear (dynamic) emphasis | 1st-order HPF → diode compressor → adder; ≤ ~7 dB extra boost for SMALL amplitudes (Matsushita EP0015499, Hitachi US4451746) | Format-wide since ~2nd-generation decks; corner unpublished | No ring; level-dependent HF shelf, NOT inverted at SP |
| White/dark clip 5.3 / 2.9 MHz | Clips emphasized overshoots (EP0015499); the sync edge's undershoot reaches the 2.9 MHz dark clip → sync edges are nonlinearly waveshaped on record | Spec | Nonlinear edge shaping |
| FM modulator | Sync tip **3.4 MHz**, blanking **3.69 MHz**, white **4.4 MHz** (confirmed = repo's ire0/hz_ire) | Spec | — |
| **Luma-FM high-pass ~1.5 MHz** | Clears the color-under band (chroma at 629 kHz occupied to ~1.13 MHz); classic figure 1.5 MHz (Samsung US5245439), model corners ~1.3–1.7 | Universal mechanism, corner machine-specific | **Primary ring suspect — see §3** |
| Record amp / FM record EQ | Constant-current, mild HF boost, lower-sideband suppression shaping (US5124850) | Machine | Low-Q |
| **Record-side detail emphasis (HQ-era decks)** | Delay-line aperture corrector: H(f) = (1+k) − k·cos(2πfT), symmetric taps [−k/2, 1+k, −k/2] at [−T, 0, +T], T ≈ 200–400 ns → peak at 1/(2T) ≈ 1.25–2.5 MHz. **Linear-phase → symmetric pre/post-shoot: the pre-shoot lands BEFORE the edge**, which no causal stage can produce — its unique fingerprint. NOT rational; kept as an explicit cosine factor outside the biquad cascade (US5400149) | Machine (HQ-era decks) | ≤2 cycles, symmetric. **Ethan sees this signature in the home sample.** Correction gated by the VHSHQ format type; (k, T) measured from the pre-edge echo; inverse needs bulk advance (non-causal by design, per Ethan) |

## 2. Tape/head channel and capture-side electronics (in the RF capture path)

- **Tape-head transfer**: differentiation + gap/spacing/thickness losses → broad low-Q hump
  over the 1–7 MHz FM band; the asymmetric tilt across the sidebands feeds the quasi-SSB
  mechanism of §3.
- **Playback head + rotary transformer + preamp LC resonance**: deliberately trimmed near the
  white carrier (4.4 MHz), damping resistor sets Q ≈ 2–5, group delay explicitly non-flat
  (Hitachi US4510530); machine-specific, trimmer-adjusted, ages. In demodulated terms it
  contributes ~0.7–1.5 MHz shaping whose frequency FALLS as luma level rises — the clean
  discriminator against the truncation mechanism (which rises with level).
- **Depth-multiplexed Hi-Fi audio carriers** at 1.3 / 1.7 MHz beat against the luma carrier
  to 1.7–2.4 MHz (Thomson US5089916 "edge noise 0.5–2 MHz"); only on Hi-Fi-recorded tapes,
  and not edge-phase-locked, so edge-locked averaging suppresses them.
- **vhs-decode's own order-20 HPF at 1.2 MHz** sits directly under the deck's truncation
  edge and reflects about the carrier to 2.2–2.5 MHz; part of the measured edge shaping is
  therefore potentially self-inflicted (see §5, experiment a).

## 3. FM lower-sideband truncation — the level-dependent ring mechanism, with numbers

VHS luma FM is **quasi-single-sideband**: the ~1.5 MHz luma-FM HPF removes lower sidebands.
Removing one sideband converts FM into mixed AM/PM; after limiting and demodulation this
appears as overshoot + ringing attached to each edge, at a frequency set by the spacing
between the instantaneous carrier and the channel edge. Carrier position = luma level, so
**the ring frequency is level-dependent and attaches to the level at the transient's START**
(Ethan's transient-start model, now with arithmetic). Pre-emphasis (×5 at HF) drives the
modulation index high at edges, so second-order (J2) sidebands matter exactly at transients:
a missing J2 line at f_c − 2f beats with J1(f_c − f) to place an error component at f itself.

With truncation edge f_e ≈ 1.5 MHz (machine corners 1.3–1.7):

| Transient starts at | Carrier f_c | 1st-order ring f_c − f_e | 2nd-order (J2) ring (f_c − f_e)/2 |
|---|---|---|---|
| Sync tip | 3.4 MHz | **1.7–1.9 MHz** | **0.95–1.05 MHz** |
| Blanking | 3.69 MHz | **2.0–2.3 MHz** | **1.0–1.2 MHz** |
| Mid-grey | ~3.9 MHz | 2.2–2.6 MHz | 1.1–1.3 MHz |
| White | 4.4 MHz | 2.7–3.1 MHz | 1.35–1.55 MHz |

## 4. Mapping: measured signature → ranked candidates

- **~0.95–1.05 MHz dominant visible ring, long decay (Q ≈ 12)** at sync tip and picture
  edges → (1) J2 sideband truncation at tip/blanking carriers (0.95–1.2 MHz — near-exact;
  spec-fixed mechanism, frequency RISES with start level); (2) underdamped head-resonance
  (4.4 − 3.4 = 1.0 MHz; machine-specific, frequency FALLS with level); (3) record detail
  corrector (≤2 cycles — cannot give 4).
- **Back-porch ring after the sync RISE — cd 1.73 MHz Q 11.5, home 2.34 MHz Q 10.4
  (matrix-pencil, this session)** → first-order truncation excited at the rise's start
  carrier; cd matches a ~1.65–1.7 MHz corner from the tip carrier, home a higher/sharper
  corner or blanking-referenced excitation. Machine-specific corners explain the tape-to-tape
  spread.
- **2.0–2.3 MHz lobe on all tapes** → first-order truncation onset for the blanking carrier
  (3.69 − 1.5 = 2.19), plus 2× the J2 mechanism (co-varies with the 1.0 MHz ring); partially
  the software's own 1.2 MHz HPF reflected about the carrier (2.2–2.5).
- **home's fall-edge |H| falling to ~0.3 floor at 2.0–2.3 MHz** → same physics, sharper
  corner: above f_c − f_e the lower sideband is fully gone and recovered amplitude
  asymptotes near the SSB −6 dB floor. Modeled by a low-pass-notch (finite-floor) section,
  which the unified biquad vocabulary covers.
- **Symmetric pre/post-shoot around edges in home (Ethan's eye)** → the HQ record-side
  aperture corrector (§1 last row): pre-shoot before the edge is its unique linear-phase
  fingerprint. Correction is format-gated (VHSHQ); (k, T) measured from the pre-edge echo.
- **First-order lag/smear (pole-zero pair straddling ~0.1 MHz, unity at DC)** → main
  de-emphasis time-constant mismatch (1.45–1.5 µs real vs 1.3 µs inverted) — near-certain;
  already documented in this branch. Level-independent. The model's lag stage IS this stage.
- **cd's +42 % peak at 1.4 MHz / 0.56× notch at 1.8 MHz (fall edge)** → capture/record FM
  equalization error (head resonance offset) is the level-linear candidate; note the fall
  window truncates the certified Q≈11 rings, so part of the 1.8 "notch" is a
  window-truncation phantom (see the measurement-layer notes in the module).

## 5. Discriminating experiments for later rounds

a. **Control decode with `video_hpf_extra` softened** (e.g. corner 0.9 MHz or lower order):
   if the 2.0–2.3 MHz lobe/asymptote moves, part of the ring is self-inflicted by the
   software chain and should be fixed at the source, not corrected after the fact.
b. **Ring frequency vs transient-start level** (active-fall strata already accumulate this):
   truncation predicts the ring frequency RISES with the start level; head resonance predicts
   it FALLS. The strata measured this session (≈2.1 MHz at 14–44 IRE drops → ≈0.9–1.0 MHz at
   78 IRE drops) are the groundwork for the FM-sideband (level-dependent) arc.

## Sources

Patents: EP0015499B2 (Matsushita NL emphasis, clips, deviation), US4451746 (Hitachi NL
emphasis), US5245439 (luma-FM HPF 1.5 MHz), US4510530 (head LC resonance, damping-R sets Q),
US5124850 (FM equalization), US4803549 (delay-equalized luma LPF), US5089916 (Thomson edge
noise / Hi-Fi carriers), US4979046 (SSB + limiter behavior), US5400149 (record detail
emphasis), US5355227 (post-demod LPF group delay). Local: `format_defs/vhs.py`,
`compute_video_filters.py`, `process.py`, `addons/FMdeemph.py`. Uncertainty: the 1.5 MHz
corner is patent-typical, not an IEC-quoted value; NL-emphasis corner unpublished; head
resonance Q engineering-typical.
