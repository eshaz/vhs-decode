# The linear audio and control tracks, as seen by the video head

> On VHS there are other tracks contained on the tape. They may overlap
> wtih the video track to some degree. These are the linear audio and
> control tracks. I believe I can use the geometry of the video heads, and
> helical scan to extract out data from these potentially overlapping
> tracks, these may be useful for synchronizing audio to video.
>
> — Ethan, 2026-09-06 (`docs/CHANNEL_LOOP_PROPOSAL.md` §10g)

**Verdict: a bounded negative, and a specified answer to the question
underneath it.** The video head's sweep does not reach either edge track —
SMPTE 32M's own table leaves a 150 µm guard at both ends — and no fringe
across that guard was found, to −85.5 dBc at the tape edge, 18.3 dB below
the most generous level the geometry allows. The one thing that looked like
a detection is 27 dB too strong for the position it sits at. But
audio-to-video synchronisation, which was the purpose, does not need the
fringe: SMPTE 32M clauses 3.4 and 3.5 fix the audio head's displacement at
79.244 mm, which is 2.376132 s of tape, exactly.

Code: `vhsdecode/models/edge_tracks.py`, tests
`tests/unit/test_edge_tracks.py`, instrument
`tools/ringing_measure/edge_tracks_measure.py`. It is registered as an
INSTRUMENT in `tools/ringing_measure/stage_inventory.py` rather than wired
as a pipeline node, because a negative has nothing to correct.

---

## 1. The geometry, and where every number comes from

The normative source is `decode-orc/analogue-video-specifications`, fetched
by raw URL (it is not in this tree): `docs/vhs/SMPTE-32M-2004/markdown.md`
and `docs/vhs/JVC-Video-Technical-Guide-VTG82063/section-1.md`. The
repository folder says 2004; the document's own title block says *SMPTE
32M-1998, Revision of ANSI/SMPTE 32M-1993*, so it is cited as 32M below,
matching `vhsdecode/models/vhs_specification.py`.

**`vhs_specification` did not carry any of this.** It records the head's
parameters and the signal's; the tape's cross-section had never been asked
for. Every dimension below is therefore new to this tree, and each is in
`edge_tracks.TRACK_GEOMETRY` with its clause and its quotation.

| Symbol | Quantity | Value | Source |
| --- | --- | --- | --- |
| A | Tape width | 12.65 ± 0.01 mm | 32M table 2, 3.2.1.2 |
| B | Video recording area width | 10.60 mm (nominal) | 32M table 2 |
| W | Video recording effective area | 10.07 mm (nominal) | 32M table 2 |
| C | Control track width | 0.75 ± 0.10 mm | 32M table 2 |
| F | Audio track reference line | 11.65 ± 0.05 mm | 32M table 2 |
| R | Audio track width, monophonic | 1.0 ± 0.1 mm | 32M table 2 |
| L | Video track centre from the reference edge | 6.20 mm | 32M table 2 |
| θ | Video track angle, tape running | 5° 58′ 9.9″ | 32M table 2 |
| α | Video head azimuth | ±6°, **± 10′** | 32M table 2; the tolerance is JVC table 1-1-1 item 18 |
| X | Audio and control head position | 79.244 mm | 32M 3.4, table 2 |
| — | Control signal rate | one pulse per frame, **above saturation level** | JVC 1.1.21; 32M states no control waveform at all |
| — | Video track centre, EP | 6.195 mm; θ = 5° 56′ 8.1″ | 32M table 3 |
| — | Writing speed, EP | 5.83 m/s | JVC table 1-1-1 item 4; **32M states none** |
| — | Writing speed, LP | **not stated anywhere** | refused rather than guessed |

**The layout closes, and that is the check that fixes which edge is which.**
The free span between the two edge tracks runs 0.75 to 11.65 mm and its
midpoint is 6.20 mm — L, exactly. Put the audio at the reference edge
instead and the midpoint would be 6.45 mm. So the standard's own arithmetic
says the control track is at the reference edge and the audio track at the
far edge; figure 7 need not be read.

    control track      0.000 .. 0.750 mm
    guard              0.750 .. 0.900 mm      150 µm
    head's sweep (B)   0.900 .. 11.500 mm
      one field (W)    1.165 .. 11.235 mm
    guard             11.500 .. 11.650 mm     150 µm
    audio track       11.650 .. 12.650 mm

**There is no overlap.** The premise of the idea is false at nominal
dimensions, and the tolerances the standard states cannot close it: the
control track's ±0.10 mm and the audio reference line's ±0.05 mm, stacked
against the guard, leave 50 µm. The guard itself is a **derived** number —
32M never prints it — which is why it is recorded as such in
`TRACK_GEOMETRY["video_guard_to_edge_track_m"]`.

Two further consequences fall straight out of the table. W at the track
angle is a track 96.8 mm long, which is 5.80 m/s for one 16.683 ms field —
so the 180° scan and the effective area are the same statement, and B − W =
0.53 mm is the **head-switch overlap**, 13.82 lines of transverse travel.
And W over a field's 262.5 lines gives the transverse rate: **38.362 µm per
line**.

---

## 2. The prediction, written before the measurement

**Speed ratio.** The head's velocity relative to the tape is the writing
speed along the track, at θ to the tape's length, so the longitudinal
component is 5.80 cos(5.96942°) = 5.7686 m/s against 33.35 mm/s: a ratio of
**172.970** — not 174, which is what dropping the cosine gives.

**Frequencies.** The control track's one pulse per frame returns at
29.97003 × 172.970 = **5183.9 Hz**. Linear audio at *f* on tape returns at
172.970 *f*. At EP the same control signal is written at 11.12 mm/s and
returns at **15628.5 Hz** — the one control that moves the target while
leaving every other rate in the deck where it is.

**Fringing.** Outside the medium the potential of a magnetisation varying as
e^{ikx} obeys (∂²/∂y² + ∂²/∂z² − k²)φ = 0, whose Green's function K₀(kr)
decays as e^{−kr}. The decay length is λ/2π in every direction, sideways
included, and it is the only length in the problem. Along the head's path
the control period is 1.11881 mm, so the decay length is **178.1 µm = 4.64
lines = 1.871 dB per line**, and at the 150 µm guard the factor is
e^{−0.842} = 0.431, **−7.3 dB**. `fringing_factor` returns this as an upper
bound, not a value: it drops K₀'s 1/√(kr) prefactor, and the absolute
coupling would need the head's sensitivity function, which needs the gap
length, which 32M does not state.

**What the guard admits.** Only on-tape frequencies below **245.8 Hz** clear
the guard at better than −60 dB; a 1 kHz tone is 244.1 dB down. So the
linear audio's *rumble* is the only part of it the video head could see, and
it would arrive spread from zero to 42.5 kHz.

**Azimuth is not the limit.** With a 58 µm track width and tan 6° = 0.10510
the azimuth scale is 6.096 µm, so the first sinc null is at a reproduced
951.5 kHz — an on-tape 5.50 kHz. At the control track's frequency the
azimuth loss is 0.99995. The premise named azimuth as the discriminator; the
geometry says lateral separation is, by many orders of magnitude, and
azimuth never gets a chance to matter.

**Level.** The head is a flux differentiator, so 5183.9 Hz is 57.0 dB below
the 3.6857 MHz blanking carrier from dΦ/dt alone. With the guard's 7.3 dB
the **generous upper bound is −64.4 dBc**, and it is generous: it credits
the control track with the coupling of an on-track read and charges nothing
for the deck's band-pass around the FM carrier, nor for the rotary
transformer's own low-frequency corner — neither of which 32M specifies,
`vhs_specification.SMPTE_32M["playback_response"]` being *"ABSENT ENTIRELY.
Every video response clause is record-side."*

**Where in the field.** The head switch is 5 to 8 lines ahead of the V-sync
leading edge (32M 3.6). Measured on `zaroff-75bars-NTSC-SP` playback, 27
fields, from the FM envelope normalised to lines −30..−25: the envelope dips
from line **−6.3**, bottoms at −5.29 at 88.3 % of reference, and is back by
**−3.5** — inside the specified window. So the incoming head is nearest the
control track just after line −3.5, falling 1.871 dB a line thereafter, and
the leaving head is nearest the audio track just before line −6.3.

**Three signatures, and the estimator uses all three.** Position (the
fringing exponential anchored at the switch), frequency (5183.9 Hz), and
**field parity** — the tape advances 33.35 mm/s ÷ 59.9401 Hz = 0.556389 mm
per field against a control period of 1.112778 mm, *exactly half*, so a
control-track fringe must reverse sign every field while anything locked to
the drum does not.

---

## 3. The measurement, and its controls

`tools/ringing_measure/edge_tracks_measure.py`. Levels are the fringe's
**peak amplitude at the tape edge** relative to the luma carrier's peak, so
they are directly comparable with the −64.4 dBc bound.

| Capture | Fields | Role |
| --- | --- | --- |
| 16 × `zaroff-*-NTSC-SP-…-rf-pb` | 442 | the deck's own tape, recorded and played on one machine |
| 16 × `zaroff-*-NTSC-SP-…-rf-rec` | 439 | **the control**: the drive going *to* the head, which can hold nothing read off the tape |
| 16 × `zaroff-*-NTSC-EP-…-rf-pb` | 442 | **the speed control**: the line must move to 15628.5 Hz |
| `countdown.flac`, `home.flac` | 90 × 2 positions each | consumer recordings from unknown machines, with real audio and control tracks |

### 3.1 The position scan settled it

At 5183.9 Hz, four-line probe, field-alternating, in dBc:

| line from V-sync | zaroff SP playback | zaroff SP record tap | countdown |
| --- | --- | --- | --- |
| −30 | −91.4 | −89.0 | −52.1 |
| −22 | −62.9 | −94.9 | **−38.1** |
| −18 | **−59.3** | −104.0 | −40.7 |
| −14 | −64.5 | −96.7 | −48.0 |
| −10 | −72.2 | −90.8 | −56.4 |
| −6 (the switch) | −39.3 | −66.8 | −65.0 |
| −2 (the tape edge) | **−85.3** | −77.8 | −57.3 |
| field median | −79.0 | −90.5 | −54.8 |

A control-track fringe **must be largest at the tape edge** and fall inward
at 1.871 dB a line. What is there is the opposite. On the deck's own tape
the alternating energy near 5183.9 Hz is at its *weakest* at the edge and
peaks eighteen lines earlier, in the middle of the picture; on the consumer
tapes the same hump is 20 dB stronger and peaks at line −22.

**The refutation is quantitative, not aesthetic.** By line −18 the leaving
head is 0.449 mm past the edge it will lift off at, even on the most
favourable anchoring, where a fringe is 21.9 dB below its own edge value.
The zaroff hump therefore implies −37.4 dBc at the edge, 27 dB above the
generous bound; the countdown hump at line −22 implies −8.7 dBc, 56 dB
above it. Neither can be an edge track. (`distance_to_track_m`,
`test_the_mid_picture_hump_cannot_be_an_edge_track`.)

### 3.2 The bound

The closest line to the tape edge that the head switch does not occupy is
line −2, where the entering head is 207.5 µm from the control track — 2.8 dB
of extra guard beyond the 150 µm. The prediction there is −67.2 dBc. The
measurement is **−85.3 dBc with a two-sigma bound of −85.5 dBc**: 18.3 dB
below the most generous level the geometry allows, and consistent with
nothing.

For the linear audio the bound is stronger still and needs no measurement to
state: at 150 µm the guard admits nothing above 245.8 Hz on tape, and the
frequency sweep from 1 to 30 kHz — which is on-tape 5.8 to 173 Hz, the whole
of the admitted band — is flat at −78 dBc on the deck's own tape with no
feature anywhere.

### 3.3 The controls

- **Record tap.** Flat at −80 dBc from 3.4 to 7 kHz, every ratio at or below
  1.44; its position scan is flat everywhere except the head switch. So the
  mid-picture hump is read off the tape and is not in the drive — a real
  playback-side effect, and not an edge track.
- **Mid-field.** Four millimetres from any edge track, where the fringing
  factor is 10^−11.6, the filter still read within 8 dB of the sweep end.
  That is how the tap's own low-frequency noise was caught impersonating a
  fringe.
- **Parity.** The non-alternating average of the same window differs by
  about 15 dB, which places the energy in the per-head branch rather than in
  anything locked to the field alone.
- **Tape speed — INCONCLUSIVE, and reported as such.** Over 442 EP fields
  nothing appears at the EP prediction of 15628.5 Hz (ratio 0.93), but the
  EP captures' own floor there is −54.2 dBc, *above* the level an SP-sized
  fringe would reach. Those captures cannot test their own prediction. This
  was to have been the decisive control and it is not one.

### 3.4 One estimator defect, recorded because it changed an answer

The first templates were unit-normalised and not orthogonalised against a
constant. A one-sided exponential of time constant 295 µs still passes zero
frequency at a tenth of its peak gain, so the filter read whatever offset
sat under the window: it returned −46.6 dBc at *every* frequency from 1 to
30 kHz and at *every* line from the switch to line +24, to a tenth of a
decibel — and an apparent −48.3 dBc "detection" at the predicted frequency
was very nearly written down. `matched_reference` now subtracts the
template's mean, which costs 0.03 % of the profile's norm.

The unit norm was the second half of the same mistake. A projection onto a
unit-norm template is not a level: the profile's length goes as 1/f, so the
scaling understates every reading by the ratio of window to profile — 13.8
dB at SP — and tilts a frequency sweep by 3 dB an octave. `Projection` now
returns the peak amplitude, 2|projection|/‖p‖², against the carrier's peak.

Its residual limit is stated rather than discovered later: the profile holds
only **1.53 cycles** at the control frequency, so the projection's
negative-frequency image does not average away and the reading depends on
the fringe's phase against the V-sync edge. Measured over twelve planted
phases at 60 fields: 5.40 dB spread, 1.90 dB rms, at most 2.91 dB from the
planted level. Every bound above carries that ±3 dB.

---

## 4. What the negative does not cover

**The closest approach is behind the head switch.** The two heads are both
on the tape for 0.53 mm — 13.82 lines — and the electronic switch sits
somewhere inside that, so the lines where a fringe would be strongest show
the *other* head. Worse, the lines immediately after the switch, where the
entering head is nearest the control track, are the vertical interval, whose
own structure is the largest field-alternating thing in the record. The
bound above is over the lines that are visible and quiet.

**The capture that would settle it: one head's preamp output taken ahead of
the switching amplifier.** That is the only remaining way to see the head
during the overlap, and it is the one capture that could change this answer.
A second useful capture would be a tape whose linear audio carries a known
loud low-frequency tone — 60 Hz would return at 10.4 kHz — recorded and
played on the same deck.

**An open observation, handed on rather than solved.** The mid-picture hump
is real: field-alternating, near 5 kHz, peaking 16 to 22 lines before
V-sync, present on all three tapes at −59 dBc (the deck's own) to −38 dBc
(countdown), absent from the record tap. It is not an edge track — §3.1 —
but nothing here identifies it. It belongs with the different-deck record
mark that `vhsdecode/models/head_switch_pair.py` already measures (5 to 10
lines of separation on both tapes this deck did not record) and with the
modulation-noise lane.

---

## 5. Can it synchronise audio to video?

**Not by the route proposed** — there is no control-track pulse in the video
head's own record to time. Had there been one, its precision would have been
set by the fringing's own smear: one decay length of tape, 178.1 µm, which
the head crosses in 30.9 µs, so a single field would locate the frame to
about that and a second's 59.94 fields to 4.0 µs, a fortieth of a line. None
of it is available.

**But the standard gives the answer directly, and it is better.** SMPTE 32M
3.5: *"Program audio or other information which is time coincident with
video information recorded at a point S₀ of the video 2 track shall be
recorded on either audio track at a distance X downstream from that point"*,
and 3.4 fixes X at 79.244 mm. Divided by the tape speed:

    79.244 mm ÷ 33.35 mm/s = 2.376132 s = 71.213 frames at NTSC SP

A machine that records and replays with the same geometry cancels this,
which is why nobody notices it. It does **not** cancel when the linear audio
is captured by anything other than that machine's own audio head at the same
time — a separate audio pass, a different deck, a tape scanned for its audio
track — and then this is the number that puts the two back together, with no
external clock and nothing measured. The head switch itself is servo-locked
to the control track, so the switch's position in the video record already
carries the frame reference the fringe was wanted for.

**Its precision is the tape speed's**, and the standard states that too:
±0.5 % on 33.35 mm/s (32M 3.1.3), so **±11.9 ms, ±0.356 frames**. X itself
carries no tolerance in table 2. For comparison, the field-to-field scatter
of the V-sync leading edge against a uniform grid on `zaroff-75bars-NTSC-SP`
playback is 23.5 samples at 50 MSps — 0.470 µs — and 0.4 samples on the
record tap, so the transport's own timing is four orders of magnitude finer
than the speed tolerance that dominates the answer. Tightening the
synchronisation past ±12 ms therefore means measuring the tape speed of the
recording machine, not measuring the tape better.
