# Running the model-based stages

Everything under `vhsdecode/models/` used to be offline: two runtime importers
in the whole tree, and no way to see any of it in a picture. Two of those
models are now stages in the decode, and this is how to run them.

There is **no new command line flag**. The stages live in the pipeline
declaration, `vhsdecode/pipeline/stages.toml`, and the graph selector that was
already there turns them on and off by name.


## The command

Decode once without them and once with them, and compare:

```sh
# before
vhs-decode /testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \
    -n -f 50 -l 14 -t 4 --dctp --lti_gain 0 --ire0_adjust --chroma_env_gain 0.5 \
    --overwrite /output/bars_before

# after
vhs-decode /testdata/test_patterns/vhs/playback/zaroff-75bars-NTSC-SP-SLV-778HF-50msps-rf-pb.flac \
    -n -f 50 -l 14 -t 4 --dctp --lti_gain 0 --ire0_adjust --chroma_env_gain 0.5 \
    --stages +chroma_head_switch,+colour_free_luma \
    --overwrite /output/bars_after
```

`-f 50` for the test patterns; **`-f 40` for `/testdata/home.flac` and
`/testdata/countdown.flac`**. Getting that wrong has silently invalidated
measurements in this arc before.

`--stages` takes a comma-separated list of `+name` / `-name` and validates
every name against the declaration, so a typo fails at startup with the valid
list rather than producing a decode nobody can account for. A leading `-` is
handled, so `--stages -cti` means what it looks like.

To see what a given selection resolves to, add `--debug_plot pipeline_graph`;
it writes `<output>.pipeline.mmd` and, alone among the plots, does not
serialise the decode.


## The two stages, one sentence each

| name | what it does |
| --- | --- |
| `chroma_head_switch` | Compensates the colour-under phase event at the head switch — the hue shift in the band across the bottom of every field — on the four or five lines it occupies. |
| `colour_free_luma` | Subtracts the up-heterodyned residual chroma from the luma, phase-locked and masked to the active area, leaving a colour-free luma. |

Both stand on a third model, `colour_lock`, which is a measurement rather than
a stage: the decoded chroma and the colour-under replica the demodulated luma
carries are the same physical vector in two frames, their difference is a
constant that repeats every four lines, and the departure from that constant
is what the head-switch detector reads. It is taken once a field and shared.

**`colour_free_luma` supersedes `--luma_beat`.** Both subtract the same
colour-under coupling from the same picture, so running the two would take it
out twice and the second fit would be made against a picture the first had
already emptied. The declaration says so, and the decoder refuses the pair at
startup rather than completing and being wrong:

```
ValueError: colour_free_luma supersedes luma_beat and both are enabled - they
would apply the same correction twice. Turn one off: --stages -colour_free_luma
or --stages -luma_beat.
```


## What to look at

**The log.** Each stage says once per decode what it found and what it did:

```
model stages: colour lock 0.961 (needs 0.398), head switch at line 258, 19.4 sigma
model stages: colour_free_luma transfer |h| 0.0144, 5.26% of the colour-under band removed (held out), lock 0.961
model stages: chroma_head_switch correcting lines [258, 259, 261, 262] by
    ['-19.2', '-4.7', '9.6', '-2.0'] degrees at gain 1.00, fields agreeing ['1.00', ...]
```

The lock figure is the one to read first. It is the mean resultant length of
the four-line offset — one for a perfectly fixed offset, zero for none — and
the number beside it is what a random phase would reach at four sigma. Above
it, the tape carries a colour-under the luma replicates and both stages are
meaningful; below it, they decline and say so.

**The picture, bottom band.** The head-switch correction reaches only the last
four or five lines of each field. In the chroma plane that is where a hue
shift sits; on `75 % bars` the correction there peaks at 26.5 IRE and is zero
everywhere else in the field.

**The picture, everywhere.** `colour_free_luma` changes the luma across the
whole active area by a few tenths of an IRE. It is not a visible edit; it is
the removal of energy at 629 kHz that is not the picture.


## Measured, on decodes taken here

Thirteen frames of each capture, decoded twice with nothing changed but
`--stages`. Two gauges, and neither is the quantity either estimator was
fitted with:

* **phase continuity** — each chroma line at the field foot against the mean
  of the quiet picture lines sharing its four-line residue, mean absolute
  departure in degrees, pooled by field parity. The head-switch estimator
  never sees this.
* **colour-under residue** — the luma's own colour-under band over the active
  area, in IRE rms.

| capture | phase continuity, first fields | phase continuity, second fields | luma colour-under residue |
| --- | --- | --- | --- |
| 75 % bars SP | 21.15 → **16.36** deg (−22.6 %) | 32.54 → **29.88** deg (−8.2 %) | 1.2408 → **1.2079** IRE (−2.65 %) |
| pulse and bar SP | 104.09 → **78.71** deg (−24.4 %) | 102.50 → **69.13** deg (−32.6 %) | 1.6865 → **1.6776** IRE (−0.53 %) |
| chroma noise SP | 3.55 → **2.94** deg (−17.3 %) | 3.83 → **2.71** deg (−29.3 %) | 0.9927 → **0.9238** IRE (−6.94 %) |
| home recording | unchanged | unchanged | 2.4534 → **2.4399** IRE (−0.55 %) |
| countdown | unchanged | unchanged | unchanged |

Every row improves and none regresses. How much the pictures moved in total:

| capture | luma changed by | chroma changed by |
| --- | --- | --- |
| 75 % bars SP | 0.2273 IRE rms, 0.75 IRE peak | 0.4609 IRE rms, 26.49 IRE peak |
| pulse and bar SP | 0.0966 IRE rms, 0.89 IRE peak | 0.4770 IRE rms, 34.17 IRE peak |
| chroma noise SP | 0.3258 IRE rms, 0.75 IRE peak | 0.3099 IRE rms, 10.41 IRE peak |
| home recording | 0.2699 IRE rms, 1.64 IRE peak | unchanged |
| countdown | unchanged | unchanged |

The share of the colour-under band `colour_free_luma` removed, as the runtime
measured it on held-out lines — fitted on even picture lines, judged on odd
ones — beside the figure the offline model recorded for the same capture:

| capture | runtime, held out | model's own figure |
| --- | --- | --- |
| 75 % bars SP | 5.26 % | 5.38 % |
| chroma noise SP | 18.52 % | 20.00 % |
| pulse and bar SP | 0.41 % | not measured offline |

That is the offline measurement reproduced by the decoder, on the same
captures, through a different path.


## What declines, and why that is the point

`countdown.flac` comes out **byte for byte identical** with the stages on. The
colour lock reads 0.149, 0.206, 0.153 on its first fields, a median of 0.151
against a floor of 0.398, so the decoder refuses the whole decode and says so:

```
model stages: declined for this decode - the colour lock's median over 4 fields
is 0.151 against the 0.398 a random phase would reach at 4.0 sigma, so the luma
carries no usable colour-under replica on this tape
```

`home.flac` sits astride the floor — per-field resultants from 0.18 to 0.41 —
and is the marginal case. Its median over the first colour frame clears the
floor, so the decode is accepted; the chroma table never reaches the agreement
between fields that it needs, so the chroma is left untouched, and the luma is
corrected by 0.55 % of its colour-under residue.

The verdict is taken ONCE, over one complete colour frame, and latches both
ways. Whether the luma carries a usable colour-under replica is a property of
the tape, not of one field, and gating field by field let this recording
flicker: ten of its twenty-eight fields fell below the floor individually, so
the correction came and went, which steps the picture where it lands. After
the decode is accepted a weak field is not refused — the fit is already
self-limiting where the evidence is thin, which is what the model demonstrates
on its y-only control, 0.07 % removed with no gate and no burst test.

The floor is not tuned. The resultant of `n` random unit phasors exceeds `r`
with probability `exp(-n r²)`; requiring that to be no larger than the
two-sided Gaussian tail at the head-switch detector's own four sigma gives
`r_min = sqrt(9.67 / n)`, which on fifty lines per four-line residue is 0.44.
Against the populations the model recorded, that admits every tape carrying
colour (0.90 to 0.99) and refuses the y-only control (0.07 to 0.20).


## Turning things off

`--stages` is the only control, and it works on everything in the declaration,
not only on the new stages:

```sh
--stages -cti          # retire the chroma transient improvement
--stages -luma_beat    # turn the shipped colour-under beat cancellation off
--stages -ringing.chroma_gate    # one component of one stage
```

`--stages -cti` is the whole of the directive asking the transform-based
chroma stage to supersede the transient improvement, which is why no second
control was added for it. Measured cost of removing it, on line-to-line chroma
phase scatter inside a flat colour over eight fields of a 75 % bars decode:
1.00 to 1.01 degrees, one per cent. The colour filtering named in the same
directive is a **different** answer and is not retired — removed, the same
gauge goes 1.00 to 2.23 degrees, and the acceptance test before it goes is a
superseding stage reaching 1.00 or below with the comb off.


## The safety property

With no `--stages` argument the decode is unchanged. Verified by decoding
`zaroff-75bars-NTSC-SP` before and after the whole of this work with the same
command and comparing the output byte for byte:

```
4a950e064ab4ac5675246da47bb68038  bars_before.tbc
4a950e064ab4ac5675246da47bb68038  bars_after_no_stages.tbc
f962ed7a979def5d29a2dc538f55b586  bars_before_chroma.tbc
f962ed7a979def5d29a2dc538f55b586  bars_after_no_stages_chroma.tbc
```

Two things hold that property up. The declaration seeds an option only where
its gate says `declared_default = true`, so a node documenting what some other
fallback comes to cannot override it; and the seeding never touches a key that
is already present, and `main.py` writes every flag-backed key
unconditionally.


## Every dimension carries

Ethan: *"All of the things that we are modeling all need to be analyzed in
hilbert space so all dimensions carry through the entire graph."* Both
channels are handled as analytic envelopes from end to end here, and nothing
in these two stages reduces one to a magnitude at a boundary: the measurement
hands on the complex chroma envelope, the complex luma replica, the complex
per-line product AND its angle; the lock offset is a table of complex numbers;
the transfer is one complex number a field; and the pooled head-switch table
is used on BOTH its axes — its angle is the phase applied and its magnitude is
the agreement between fields that decides whether to apply it at all.

Three deliberate reductions remain, and they are reductions on evidence rather
than by omission:

* `colour_lock.lock_offset` normalises each line product to unit modulus. On a
  matched pair of captures the amplitude at the colour-under carrier is the
  same to four parts in a thousand while the LOCK collapses from 0.98 to 0.13,
  so amplitude carries no information about this quantity and weighting by it
  would let a saturated field outvote a pale one.
* `chroma_head_switch.accumulate` sums unit phasors, for the same reason.
* `colour_free_luma.colour_free` takes the real part at the very end, which is
  the re-modulation of the estimate to a real waveform and not a loss of rank.

`chroma_stage_supersession.line_phase_scatter` reduces to a scalar in degrees
at its own boundary, but it is a gauge rather than a stage and nothing
downstream reads it.


## Where the code is

| what | where |
| --- | --- |
| the declaration | `vhsdecode/pipeline/stages.toml` |
| the selector, defaults and supersession rules | `vhsdecode/pipeline_graph.py` |
| the runtime glue | `vhsdecode/model_stages.py` |
| the call sites | `vhsdecode/chroma.py`, in `process_chroma` and in `FIELD_CHROMA_STAGES` |
| the models themselves | `vhsdecode/models/colour_lock.py`, `chroma_head_switch.py`, `colour_free_luma.py` |
