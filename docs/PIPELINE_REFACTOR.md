# Refactoring the decode pipeline onto the hypercomplex model

Ethan, 2026-09-06:

> I want you to refactor the entire decode pipeline to use the hypercomplex
> modeling. Anything that does not use that should be retired.

## 1. Where the pipeline stands today, measured

`vhsdecode/pipeline/stages.toml` declares **33 nodes**. Each was checked
for whether its entry point's module reaches the hypercomplex substrate,
the component model, or any complex arithmetic at all:

| criterion | nodes |
|---|---|
| import `vhsdecode.models.hypercomplex` | **0** |
| import anything from `vhsdecode.models` | **0** |
| use complex arithmetic of any kind | 9 |
| declare no callable entry point at all | 24 |

So the answer to the instruction's premise is that **none of the pipeline
uses the hypercomplex model today**. The whole of `vhsdecode/models/` has
exactly two runtime importers in the tree, `vhsdecode/debug_plot.py:667`
and `vhsdecode/field.py:1201`, and neither is a correction.

That is the size of the work, and it is why the first deliverable is a
runnable decode rather than a rewrite: nothing can be judged until the
model-based stages reach a picture.

## 2. The three groups, and why retirement is not deletion

A stage earns its place by one of two things: it is the model, or it beats
the model. Everything else goes. But **a stage is retired only when its
replacement has been measured to do at least as well**, and this arc has
already been shown why. On a burst-referred colour gauge the shipped
decode reads 1.00 degree; removing the chroma transient improvement moves
it to 1.01, a one per cent cost, so that stage can go today. Removing the
comb filter moves it to 2.23, a 123 per cent cost. The comb is doing real
work, and retiring it before its successor reaches 1.00 without it would
be a regression sold as a principle.

### Group A - the signal path. Convert, never retire

`rf_video`, `analytic`, `envelope`, `demod`, `deemphasis`, `video05`,
`resample`, `hz_to_output`.

These are not corrections, they are the chain. Two of them are already the
hypercomplex substrate under another name: `analytic` forms the analytic
signal and `envelope` takes its modulus, which is precisely
`hypercomplex.hilbert` followed by a magnitude. The conversion is to route
them through the model so the phase they compute is carried forward
instead of discarded at the modulus, and so `demod` uses the exact
spectral derivative that `hypercomplex.demodulate` supplies. That
derivative is not a refinement: a central difference in the same place
reads the carrier 239 kHz low and lands 769 times the output floor.

`deemphasis` stays because it IS the model - the pre-emphasis is
specified, and its inverse is not a heuristic.

### Group B - corrections with a model counterpart. Convert

`head_switch`, `baseband_eq`, `carrier_tbc`, `luma_amplitude`,
`luma_eq_update`, `chroma_env`, `decode_chroma`, `burst_phase`,
`chroma_under`, `luma_beat`, `ringing`, `residual_channels`.

Each of these measures something the component model already describes,
and each currently does so with its own private estimator. The conversion
is to have them read the component key and the held-out judge rather than
fit in isolation. `ringing` is already being replaced by the tesseract
graph, and `luma_beat` is superseded by `colour_free_luma`, which
subtracts the same coupling in the form the directive asks for - the two
must not both run, or the coupling is subtracted twice.

### Group C - retire. Heuristics with no model behind them

`notch`, `high_boost`, `rf_eq`, `video_eq`, `spike_replace`, `nldeemp`,
`subdeemp`, `luma_transient`, `luma_transient_observe`, `chroma_trap`,
`cti`, `y_comb`.

These are shaping filters and repair heuristics chosen by ear. They have
no component in the key, no error bar, and nothing that could refuse them.
Nine of the twelve already default to off or zero, so retiring them costs
nothing on a default decode; the exceptions are `video_eq`,
`luma_transient` and `cti`, which are on by default, and `y_comb` and
`chroma_trap`, which the colour path leans on.

**The order of retirement is fixed by the measurement, not by the list.**
`cti` first, at a measured cost of one per cent. `luma_transient` and
`video_eq` next, each against a gauge before it goes. `chroma_trap` and
`y_comb` last, and only once the transform-based colour stage reaches the
comb's own 1.00 degree without it.

## 2a. Retire the one-off options too

Ethan, 2026-09-06, after the first version of this plan:

> Let's retire all the one-off options and have evertying we have done so
> far live in the graph and have a consistently modeling structure. All of
> the things that we are modeling all need to be analyzed in hilbert space
> so all dimensions carry through the entire graph

`vhsdecode/main.py` carries **90 options**. Many are one-off knobs for a
single correction, and at the time he said this two of them were
`--color_free_luma` and `--colour_free_luma`, two spellings of the same
thing. That is the shape of the problem: each correction arrived with its
own switch, so the switches now outnumber the stages and no two agree on
how to be turned off.

**The rule.** A correction is a node in `stages.toml` and its on and off
control is the graph selector that already exists, `--stages -name` and
`--stages +name`. No correction gets a flag of its own. Where the existing
selector cannot express something, that one selector is extended rather
than a second mechanism added beside it.

**The node contract, and this is the half that is not bookkeeping.** Each
node hands on a COMPLEX quantity wherever a phase exists. A node that
reduces to a magnitude at its own boundary forces the next node to
re-derive a phase that was already known, and this arc has repeatedly
found that such a boundary silently halves a measurement's rank - it is
what made one burst instrument rank two of four, and it is why
`hypercomplex.complex_form` exists at all. "All dimensions carry through
the entire graph" is that contract stated once for every node.

**What retiring an option is not.** It is not deleting a capability. A
knob that is doing real work becomes a node default, measured; a knob that
is doing nothing goes. The same gauge discipline as section 2 applies:
nothing leaves until the number that would get worse has been named and
measured both ways.

## 3. What has to exist before any of this is judged

1. **The model stages reachable from a decode.** A flag per correction, an
   umbrella `--model_stages`, every one defaulting off, and a
   bit-identical decode when none is passed.
2. **A per-field residual report and plot**, so the difference each model
   makes is visible rather than argued.
3. **A gauge per retirement.** No stage leaves until the number that would
   get worse has been named and measured both ways.

Items 1 and 2 are in hand. Item 3 is the discipline that makes the rest of
this document safe to execute.

## 4. Performance: where the time actually goes (2026-09-07)

Ethan: *"Let's sweep through and performance optimize. I believe the
hypercomplex stages and the folding can be consolidated through some
mathmetical simplification."*

**The fold is not where the time goes, and that is worth stating first.**
`tesseract.walsh` on four axes and 2049 bins costs **1.02 ms**. It is
already the fast Walsh-Hadamard transform, folding the cube onto itself
once per axis; writing it as a bare in-place butterfly instead of a
dictionary of cubes costs 0.41 ms, so the whole available saving there is
six tenths of a millisecond. The intuition that the fold is expensive is
understandable and the measurement does not support it.

**Where it does go.** Turning one stage on at a time against a run with
none, over four fields:

| stage | cost |
|---|---|
| sync_shape | +9.6 s |
| burst_sync_lock | +9.4 s |
| head_switch_pair | +9.3 s |
| capture_filter | +5.4 s |
| band_delay | +5.0 s |
| the other thirty-one | −0.8 to +0.2 s each |

Five stages are nearly all of it, and what they have in common is that
each takes its own full-field transforms of the same samples.

### 4a. The consolidation: one set of transforms a field

`band_delay` and `head_switch_pair` both called
`band_delay.luma_frequency` and `band_delay.chroma_envelope` on the same
field. `band_delay.analytics` now computes that pair once and both take
it. Measured on one real field of 667334 samples at 40 MSps, back to
back: **2.021 s each computing its own against 1.076 s sharing, a 47 per
cent saving with the answer identical** - it is the same arrays handed
over, not an approximation of them. Of the 1.076 s remaining, 0.963 is the
two transforms themselves, so almost the whole cost of that pair is the
transforms and sharing halves it.

It shares only where the array is the same one. `head_switch_pair`
prepends the previous field's tail so its record starts where the last one
ended, and handing it another array's transforms would be an error rather
than a saving, so it shares only when that concatenation is a no-op.

### 4b. The mathematical identity: one convolution, not two passes

A forward-backward finite-impulse filter convolves the record with the
kernel and with the kernel's reverse. Convolution is associative, so the
pair is ONE convolution with the kernel's own autocorrelation - a
zero-phase kernel of twice the length less one.
`hypercomplex.zero_phase` performs that single pass and reproduces
`filtfilt`'s own odd edge extension so the boundaries agree too.

Measured on a field with a 255-tap band-pass: **the worst disagreement
anywhere is 1.332e-15 on a signal of rms 0.3581, which is machine
precision, and it runs 3.9 times faster - 11.9 ms against 46.6.** A
field's raw radio frequency passes five such filters before any
measurement is taken, so the identity removes about three quarters of that
cost without changing a returned number.

### 4c. The one that was left on the table, and why

A field at 40 MSps is 667334 samples, which factors as 2 x 333667 with
**333667 prime**, so every transform at that length falls back to
Bluestein's algorithm. Padding to the next fast length, 668250, makes the
analytic signal **6.7 times faster** and the real transform **19.6 times**.

It was not taken. The transform is circular either way, so padding is not
wrong in principle, but measured against the unpadded answer a zero pad
costs 0.30 of the output floor in the interior and **1815 floors at the
very edges** - and the edges of a field are where the head switch and the
vertical interval live, which is where this arc reads its most delicate
events. Every pad mode was tried; the best, a circular wrap, still costs
4.8 floors at the worst point. Sharing one exact transform is the honest
saving and padding is a trade this material cannot afford.


## 5. The single transform in the decode (2026-09-07)

> I think I can do much less now than all the stages. I can make a singla
> picture stage that transforms luma chroma all up to the composite
> functions. There doesn't need to be sequencing, just all the dimensions
> execute at once in a single transform.

> Build it and fully incorporate into the decode pipeline. Remember my
> written instructions and all the existing methods we have derived here,
> and build a clean decode stage update. Once it is incorporated, we will
> analyze each possible node our hypercube for redundant corrections that
> exist in the old code. Then we should remove that old code and check
> that everything still matches, or looks better. Continue until all the
> nodes are traversed. The goal of the final result is the best possible
> quality YC video.

**What the audit in section 1 did not say, and the build had to find.**
There is no executor. `stages.toml` validates, seeds defaults and draws a
graph; the order the decode runs is hand-written in three places - the
radio-frequency worker threads in `process.py`, the field side in
`field.py` (where `lock_to_burst` runs on the decode thread for field N+1
while the main thread downscales field N), and the chroma table in
`chroma.py`. Thirty-six model-stage nodes are seeded on by
`declared_default = true` while the comments beside their call sites still
said off. `picture_stage` applied nothing. The shared radio-frequency
analytics never shared, because every caller took its own float copy and
the cache was keyed on the copy's identity. And one decode cannot reach
two of the eight declared axes: head is field parity, so the picture's
field axis is the head axis under another name, and a capture is taken at
one tap. Those two are known constants, not null space, and the runtime
cubes carry three live axes each.

**Two nodes replace the three sequences at the default.** `picture_transform`
is called once inside `chroma.decode_chroma`, outside its `write_chroma`
branch, where the luma after `hz_to_output`, the up-converted chroma and
the tape's own colour-under exist together; with no chroma it runs
luma-only. `rf_transform` is called once in `downscale` where the
radio-frequency measurement group stood, and publishes an immutable
snapshot the workers read: one head-common table multiplied at the
equaliser site after the envelope is frozen, one table per head chosen by
a head schedule from the block's absolute sample, and a one-shot
redemodulation that repays the blocks demodulated before the latch. Every
absorbed node's measurement fills a vertex of one cube; one fold serves
them all; the realisations are composed into one float64 copy and written
back once. Both legacy paths stay selectable with `--stages
-picture_transform` and `--stages -rf_transform` until the traversal below
retires them.

**Two things are measured and deliberately not realised in the build.**
The ringing subtraction stays where it runs, before `hz_to_output` and one
field at a time, because its tests pin that and the latch rule contradicts
it; the transform measures the polarity contrast and the two meet at the
polarity node of the traversal. And a static per-head chroma gain and
phase already exist under other names - the chroma automatic gain's parity
banks and the burst-locked up-conversion - so the (colour, head) contrast
is measured and reported rather than applied a second time.

**The null space, three ways.** Section 10k of `docs/CHANNEL_LOOP_PROPOSAL.md`
records his words. SUBSTITUTE never computes a null contrast. REMOVE folds
everything and subtracts each contrast by the amount it reproduces across
two banks of fields, the R5 agreement gate. The capture's HYPERCUBE - the
sample rate fixes where data can exist, the format fixes where the signal
may - is measured once where the data must not exist and subtracted as the
constant it is. The three floors are reported together, and the applied
treatment is decided by the held-out remainder on the three test decodes
and recorded both ways.

### 5a. The traversal: every node against the old code

The procedure, per node: name the old corrections that model the node's
mechanism; decode control and test back to back from one tree, the old
correction on with the transform's component excluded against the old
correction off with the component on; judge on the gauges - the per-field
sync-edge remainder from `tools/ringing_measure/modelled_residuals.py`
(the cd sample's sync pulse flattens when the source correction is right,
in his words), the chroma leakage ratio, the vectorscope's axis shape and
the per-line against per-field burst constancy, the porch noise, the byte
difference of the Y and C outputs, frames per second interleaved, and the
ringing battery where the polarity node is touched; remove the old code
only when the transform's reading is at least as good on every gauge, and
record the measurement both ways here.

| node | old corrections that model it | note |
| --- | --- | --- |
| picture (polarity,) | `ringing_tesseract`, `luma_transient`, `--nld`, `--sd` | candidate 1: the per-field ringing cube against the transform's latched contrast; `luma_transient` exists only to supply the polarity axis the luma equaliser lacks; the nonlinear de-emphases stay where the format sets them |
| picture (colour, head) | the chroma automatic gain's parity banks (candidate 2), the burst-locked up-conversion's per-head phase (candidate 3) | field parity is the head; the banks are this node |
| picture () | `source_correction`, `precursor`, `sync_depth`, `standard_levels`, `level_from_frequency`; `baseband_eq` (polarity-common, heads pooled); `video_eq`; `--ire0_adjust`, `--level_adjust`; `deemphasis` | `deemphasis` stays, it is the model; six level anchors collapse to the level channel |
| picture (colour,) | `colour_free_luma`, `luma_beat`, `chroma_trap`, `y_comb`, the output subcarrier notch; the burst tilt; `cti` | one remover survives; `cti` first at its measured one per cent; the chroma comb is not retired until the transform reaches its 1.00 degree |
| picture (head,) | `head_switch` (read only), `chroma_head_switch`, `--dctp` | the switch event is the boundary between the head vertices in time |
| rf () | `RFVideo`, `notch`, `high_boost`, `rf_eq` / `channel_eq`, `capture_filter` | `high_boost` has no model; `capture_filter` is a constant of the capture |
| rf (band,) | `chroma_env_gain` / `chroma_env_phase`, `tape_bias` | the luma envelope measures the parent and the chroma receives it: the fold's own statement |
| rf (head,) | `luma_eq` per head, `head_differential`, `head_model` | the schedule makes the per-head radio-frequency differential applicable at all |
| rf (band, head) | `luma_beat`'s dead radio-frequency footprint | dead code, removed |
| the unreached axes | `tap_transfer` (tap); field parity (field is head) | known constants of a decode, measured across captures offline |
| the time dimension | `carrier_tbc`, `--dbh`, `resample`, the wow spline | the delay term of each contrast; the line-by-line time base stays as the resample node |
| the demodulator | `unwrap_hilbert` against `hypercomplex.demodulate` | one step, gauged against the output floor |

The order is by measured cost: `cti`; the polarity node; the (colour, head)
pair; the (colour,) removers; the level anchors; the combs last; the
demodulator when its gauge is built. A two-pass decode that repays the
fields written before the latch, through the redo mechanism, is a
traversal item. The measurements for each step are appended below this
line as they are taken.

### 5b. The first measurement of the build: the RF stage on a real decode

Source: the RF adapter's smoke decode of the SP bars capture, twelve frames
at 50 MSps with every filler on by declaration, its per-field readings in
`scratchpad/rf_probe/decode/fields.jsonl` and the decoder's own log beside
it (2026-09-07). Every number below is read from those files.

The stage latched at the eighth field under the substitute treatment at the
correction-gain law's half, and the decoder's log carries the latch line for
both transforms on the same field. The ninth call was the one-shot
redemodulation re-creating field eight at the same tape position; it was
answered from the memo in a millisecond with no second count and no version
change. Fields one to eight cost 0.70 to 0.90 s each with the fillers
active; after the latch an inactive field costs 0.012 s and the stride
fields (sixteen and onward by doubling) cost 0.69 s.

| reading, luma face of the response channel | before | after the exact inverse | after the half |
| --- | --- | --- | --- |
| field 8, in sample, nepers rms | 0.154 | 0.0067 | 0.081 |
| field 16, held out of the frozen latch | 0.145 | 0.0026 | - |

The held-out field's response is explained to 99.97 per cent by a
correction frozen eight fields earlier, which is the constants rule
measured on the transform itself: the response is a constant over the span.
The chroma face reads 0.120 to 0.0012 in sample and 0.121 to 0.0005 held
out; the band delay 196 ns to 6 ns in sample and 200 ns to 1.6 ns held out.

**The three floors, on the response channel, in the power a contrast bin
carries:** the instrument 2.0e-7, the two banks' disagreement 4.2e-7, and
the capture's hypercube 2.4e-10. The in-band floor stands about eight
hundred times above the file's own constant. That is the case the design
anticipated: the field-to-field variation of the luma band's response is
structure, not noise, and it is named in the reading rather than discarded.
The null pool is identically zero on this channel because every null
contrast of the radio-frequency declaration contains polarity and both
bands' response faces are duplicated on it; the level channel, where the
polarity contrast is the deviation's error, is where that comparison has
content.

Two facts of the wiring were verified on the decoder rather than assumed:
a block's absolute start is `blocknum * blocksize` on the same axis as
`readloc + blockcut + i` (lddecode core.py:1279-1283, 1363-1365, 1446,
4054-4056), and `rf.bytes_per_field` does not exist on the RF decoder (it
lives on the demodulation cache, in samples), so the period is taken from
the format's own field rate and refined by the median of the located
switches, which settled at 666048.5 samples against the format's 667333.7.
