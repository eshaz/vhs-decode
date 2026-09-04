# Multidimensional information extrapolation — Ethan's component model, formulated as a function

## Context

Ethan rejected the channel-identifier plan and specified what he wants
instead: **a function that models the components he has described**. The
restoration is three-dimensional — AMPLITUDE × FREQUENCY × TIME. Every
component of the system is measured on its own criterion along those three
axes; on each axis it has a RESOLUTION (its measurement grid and bounds — the
information boundary) and a RESIDUAL (its departure from the ideal). The
restoration applies the anti-residuals iteratively until no residual is left
(the known low-pass response has been matched), component by component, the
noise floor being the LAST component; then the 2-D differential of what
remains is found. The RF capture is done first ("pure RF frequency and phase
cutoff extrapolation"); the luma follows with the components already
identified today. His words are the Appendix; the function below is their
formulation. Scope and ownership stand as answered (this lane builds it).

**Two structural additions from his second rejection.** (1) The first plan —
the channel identifier — is kept and FEEDS this function as the BASELINE:
the channel identifier's frequency response is what the received response
is differentiated against on the frequency axis. (2) The TIME axis has its
own derivation: timing is corrected by the hsync and chroma-burst lock
elsewhere in the code, and the downscaling — resampling on the distance
between sync pulses — removes the residual RF PHASE component; the
derivation process is applied to the time-base correction itself, helped by
the wow-factor AMPLITUDE difference, because the time-base residual is what
keeps phase transferring through the residual. "THIS was the missing part."

## Step 0 — the note in the repo

On approval, first action: write `docs/CHANNEL_LOOP_PROPOSAL.md` = the
Appendix verbatim (his statements, in order, one-line context labels only).

## The function

### Data model (`vhsdecode/information_extrapolation.py`, NEW)

```python
Axis = Literal["amplitude", "frequency", "time"]

@dataclass(frozen=True)
class Resolution:            # the information boundary on one axis
    unit: str                # what one step is
    minimum: float
    maximum: float
    total: int | None        # count of steps where the axis is discrete

@dataclass
class Residual:              # expected minus received, along one axis,
    axis: Axis               # as a function over the other two axes
    value: np.ndarray | None # None = not yet measured / unknown
    over: tuple[Axis, ...]   # which axes it varies over
    source: str              # how it is measured (the criterion)

@dataclass
class Component:
    name: str
    ideal: Callable          # the expected characteristic (spec-derived)
    resolution: dict[Axis, Resolution]
    residual: dict[Axis, Residual]
    criterion: str           # the unique measurement that identifies it
    frozen: bool = False     # "no longer differentiate over that part"
```

### The first component, exactly as specified

```python
RF_CAPTURE = Component(
    name="pure RF frequency and phase cutoff extrapolation",
    ideal=flat_to_nyquist,                       # ideal: unity to Nyquist
    resolution={
        "amplitude": Resolution(unit="8-bit step (hard-coded 8; an input "
                                     "flag will supply the capture depth)",
                                minimum=0,       # DC
                                maximum=255,     # the initial RF 8 bits
                                total=256),
        "frequency": Resolution(unit="Hz (Nyquist theory: fs/N per bin)",
                                minimum=0,       # DC
                                maximum=20e6,    # Nyquist at 40 MSps
                                total=None),     # with a 13.3 MHz roll-off
                                                 # applied at the capture stage
        "time":      Resolution(unit="sample",
                                minimum=0, maximum=None,
                                total=N_samples),# the number of amplitude
                                                 # measurements
    },
    residual={
        "amplitude": Residual("amplitude", value=None,
            over=("frequency", "time"),
            source="the RF capture's own frequency response (NOT KNOWN "
                   "now - needs the final ENCODED tape signal fed back), "
                   "and the response over time between ideal Nyquist and "
                   "the hardware roll-off filter"),
        "frequency": Residual("frequency", value=None,
            over=("time",),
            source="the differential, over time, between that response "
                   "and the ideal-Nyquist-minus-hardware-roll-off"),
        "time":      Residual("time", value=None, over=(),
            source="the final noise-floor residual - or none, stop here"),
    },
    criterion="ideal Nyquist vs the 13.3 MHz capture roll-off",
)
# effective information budget = 8 bits x 40 MHz sample rate
```

### The last component — the noise floor

```python
NOISE_FLOOR = Component(
    name="noise floor of the block the FM signals were gathered in",
    ideal=lambda: 0.0,
    criterion="integrate the sync pulse shape to infinity: the accumulated "
              "sync converges to its deterministic shape; what does not "
              "converge is the noise. Amplitude = its level relative to "
              "the CORRECTED sync pulse; residual = that remainder.",
    ...)   # amplitude only: distributed over a statistical curve, no phase
```
Implementation of "integrate to infinity" = the existing accumulation
(`ringing_cancellation` slow interval mean and cross-line variance per lag;
`sync_step_response.accumulate`), N → all lines: the mean is the shape, the
variance of the mean → the floor. It is subtracted from the block, and it
gates every other residual by amplitude (the Wiener term); no phase is ever
derived from it.

### The baseline — Part A, the channel identifier, as a component input

```python
CHANNEL_BASELINE = Component(
    name="channel identifier (Part A): the baseline frequency response",
    ideal=None,   # it IS the reference: what the received response is
                  # differentiated against on the frequency axis
    criterion="pilot identification: the sync edges (per polarity, at the "
              "tip and blanking carriers) for phase, the constant envelope "
              "for magnitude, the decoder's own filters compensated "
              "analytically - the first plan's offline loop, exported as "
              "channel_response.npz",
    ...)
```
On the frequency axis every later component's residual is
`received_response − CHANNEL_BASELINE` (in the log domain), never
`received − flat`: the identifier supplies the expected response, the
function measures departures from it.

### The time axis — the time-base residual, THE MISSING PART

```python
TIME_BASE = Component(
    name="time base: wow and flutter",
    ideal=spec_line_timing,                   # hsync + chroma-burst lock
    resolution={"time": Resolution(unit="sample", minimum=0,
                                   maximum=None, total=N_samples)},
    residual={"time": Residual("time", value=None, over=("time",),
        source="the time-base residual (wow/flutter), witnessed by the "
               "wow-factor AMPLITUDE difference: the envelope's drum-rate "
               "wobble that correlates with head-to-tape speed (the AM "
               "residual d; the wow panel already measures 0.0369% rms) - "
               "derived and fed to the carrier TBC as its flutter prior")},
    criterion="timing before anything spectral: correct the time base, "
              "then the downscale (resampling on the sync-to-sync "
              "distance) removes the residual RF PHASE component - which "
              "is what was keeping phase transferring through the residual",
)
```
Timing itself is the decoder's (hsync + burst lock:
`lddecode/core.py refine_linelocs_hsync :2521`, `refine_linelocs_burst
:3430`), refined by `vhsdecode/carrier_tbc.py refine_linelocs`, swapped
into `FieldShared.downscale` at `field.py:1215-1218` (one spline yields
timing and level; band-limited to the flutter band). The derivation adds
the wow-amplitude witness as that refinement's prior — the open "flutter
prior" item of the carrier-TBC round — so the time-base residual is driven
to its floor BEFORE the frequency-axis differentiation runs on each pass.

### The process

```python
def multidimensional_information_extrapolation(signal, components, gauges):
    anti_residuals = []                          # applied during demodulation
    for component in components:                 # RF capture first, luma after
        while True:
            measured = component.measure(signal, gauges)   # its own criterion,
            residual = measured - component.ideal          # on all three axes,
            if at_floor(residual, held_out=True):          # held-out lines
                break                            # e.g. the LPF's response is
            anti = anti_residual(residual)       # matched; the residual IS
            signal = feed_forward(signal, anti)  # the kernel; through the
            anti_residuals.append(anti)          # real chain, all together
        component.frozen = True                  # no longer differentiated
    noise = NOISE_FLOOR.measure(signal, gauges)  # integrate sync to infinity
    signal = subtract(signal, noise)             # the last component
    differential_2d = fft2(residual_at_floor)    # frequency x time
    return demodulate(signal, channel_response=response_of(anti_residuals)), \
           differential_2d
```
- Stage 1: run over the RF (`components=[RF_CAPTURE]`) → the exact incoming
  residual curves for the luma = a set of anti-residuals applied to the
  signal DURING demodulation; demodulate using the residual frequency
  response measured from the final residual.
- Stage 2: apply the same function to the luma with the components already
  identified (sync-step response per polarity, constant-envelope magnitude,
  the emphasis remainder, the dark clip, the source class, head switch,
  dropouts) — each instantiated as a `Component` with its own criterion.
- Order within every pass (his rule: timing before anything spectral):
  `TIME_BASE` first — derive the time-base residual from the wow-amplitude
  witness, refine the linelocs, downscale (the residual RF phase goes with
  it) — then the frequency-axis differentiation of each component against
  `CHANNEL_BASELINE`, then the amplitude-axis noise floor last.
- Rules carried by `at_floor` / `feed_forward` (settled today): feed forward
  all components together; converge TO the floor, never through it; step
  size errs low (the half-amount law); hold-out lines (fit even, measure
  odd); exhaustion = no stable structure in the held-out residual; a
  residual on the sync edge that differs from one on a white transition is
  the level-dependence node, added, not cycled; the noise model is the
  constraint set, never the residual itself; a converged component is
  frozen; any linear residue after the RF converges goes back to the RF
  loop, never into a baseband stage.

## Implementation mapping

0. **Part A — the channel identifier (the first plan), the baseline.**
   `tools/ringing_measure/channel_identify.py` (NEW): the offline pilot
   identification loop — noise PSD first (RF quiet band + blanking),
   magnitude from the constant envelope (`luma_amplitude.rf_path_response`
   and `luma_path_equalizer(rf, curve, -1.0, dense)` for S·L, or the shared
   `dense_response`/`static` exports), phase from the sync edges
   (`sync_step_response.step_response` / `pulse_response` per polarity at
   the tip and blanking carriers; minimum phase from the widest honest
   magnitude via `minimum_phase_from_magnitude`), the decoder's own filters
   compensated analytically from `rf.Filters`, per-head pooling with
   `baseband_eq._pool`, the residual as the kernel each pass, held-out
   lines, the exhaustion test — exported as `channel_response.npz`
   (`freqs_hz`, complex `log_H`, `belief`, per-head arrays, `noise_psd`,
   `passes`, `exhaustion`, `site="rf_pre_demod_post_envelope"`,
   `freq_hz`, `blocklen`, `system/tape_format/tape_speed`, a static-path
   fingerprint). This export is `CHANNEL_BASELINE`.
0b. **The time-base node.** The wow-amplitude witness = the drum-rate
   wobble of the envelope residual (`head_switch`'s `d`; the existing wow
   panel), correlated with the flutter-band timing trace; fed to
   `carrier_tbc.refine_linelocs` as its prior (the round's open
   "flutter-prior" item). Files `carrier_tbc.py` / `head_switch.py` are the
   head-switch lane's: this node is specified here and built with them
   (tell-before-touch, or handed to them with the contract) — never edited
   unilaterally.
1. **`vhsdecode/information_extrapolation.py` (NEW):** the data model, `RF_CAPTURE`,
   `NOISE_FLOOR`, the luma component instances, `multidimensional_information_extrapolation`. Pure
   numpy; the 8-bit depth hard-coded with a `TODO --rf_bits` flag.
2. **`tools/ringing_measure/multidimensional_information_extrapolation.py` (NEW):** the offline runner:
   caches the RF blocks of a field subset, measures each component's
   residuals with the existing gauges (`sync_step_response.step_response`
   / `pulse_response` / `accumulate`; the active-transition gauges;
   the RF-domain quiet-band and blanking noise PSD measured today), feeds
   forward via a subset decode per pass, exports the anti-residual set as
   `<prefix>_channel_response.npz` (contract from the design pass:
   `freqs_hz`, complex `log_H`, `belief`, per-head arrays, `noise_psd`,
   `passes`, `exhaustion`, `site="rf_pre_demod_post_envelope"`, `freq_hz`,
   `blocklen`, `system/tape_format/tape_speed`, a static-path fingerprint).
   Every pass decode is followed by the hsync artifact model plot.
3. **`vhsdecode/channel_eq.py` (NEW, runtime):** loads the export once at
   init (after `_computevideofilters_b`, `process.py:922`), interpolates
   onto the full `blocklen` fft grid, delay-normalizes at the blanking
   carrier (`rf.iretohz(0, spec=True)`; otherwise the sync edges move and
   linelocs re-land), builds the conjugate-symmetric table, stores
   `rf.Filters["ChannelEQ"]`; never mutated (the `luma_eq` reproducibility
   law, `luma_amplitude.py:2020-2035`). Never imports the offline gauges
   (matplotlib/geometry side effects).
4. **`vhsdecode/process.py`** — three small hunks: `:704-709`
   `self._channel_response`; Options NAME/VALUE lists in lockstep
   (`:818-825` / `:875-881`) with `"channel_eq"`; the site `:1575-1593`:
   `rf_eq = ChannelEQ if present else LumaPathEQ`, multiply after the
   envelope (the 17%-chroma rule), Hilbert rebuilt `if boosted or rf_eq is
   not None` — with `ChannelEQ` absent every operation is textually
   today's, so byte-identical.
5. **`vhsdecode/main.py`:** `--channel_response <npz>` and `--channel_eq
   [amount]` (`nargs="?"`, `default=0`, `const=1.0`) beside `:487-509`;
   `rf_options` at `:846-850`.
6. **Supersession:** `luma_eq` forced 0 with a notice when the stage is
   active; `baseband_eq` unchanged (inert without a file); ringing stage:
   `_ringing_state["channel_eq_active"]` set by the loader, read in
   `_gate_parameters` to mute the RF-derived synthesized kernels (new
   sidecar key `synth_blend_under_channel_eq`, default false); witnessed
   and source-class kernels stay live and re-certify on the corrected chain.
7. **Reuse by import only:** `luma_amplitude.rf_path_response`,
   `luma_path_equalizer(rf, curve, -1.0, dense)` (the forward S·L
   magnitude), `sweep_collapse_tables(rf).ceiling` (the physical bound);
   `baseband_eq._pool/_smooth/_log_transfer`; the `head_switch._estimate`
   hygiene ported (~40 lines, attributed) with the noise PSD as the ridge.

## Coordination

Before the two shared files are touched: message b5 (head_switch /
baseband_eq), e4 (luma_amplitude; request a public `measured_response(rf,
head)` accessor so no private name is load-bearing), 46 (writeup), 23
(chroma — the same anti-residuals apply to the chroma path next). Quiet
window on `process.py`/`main.py`; owners' modules imported, never edited.

## Verification

- No-harm: flags absent → `cmp` exact against `/output/decodes/noop_off3`;
  a UNITY export → also exact (`(a+bj)*(1+0j)` is exact; a free second
  check).
- `tests/unit/test_information_extrapolation.py` + `test_channel_eq.py`: the data
  model round-trips; `RF_CAPTURE` carries the specified bounds; unity
  table `array_equal`; conjugate symmetry; loader refuses a mismatched
  system. `tests/run_unit.sh` stays green (41 collected today).
- The loop's own outputs on pnb/cd/home: residual per pass per axis per
  component; convergence-together (sync + white) table; exhaustion verdict
  with any stable lines found; the noise-floor component's amplitude
  relative to the corrected sync pulse; the 2-D differential figure.
- Per-decode hsync plots; the per-head battery; the four-class
  decomposition; Ethan's eye on the A/B.

## Assumption flagged for approval

"The block of data that I gathered these two FM signals in" is read as the
RF block carrying the luma FM carrier and the color-under FM; if the two
signals meant are the sync fall and rise, the noise-floor component's
block is the sync interval instead — same function, different block.

---

## Appendix — Ethan's proposal, verbatim, in the order he stated it

*(The content of `docs/CHANNEL_LOOP_PROPOSAL.md`. Each quotation is exact;
the italic line above each is only the context in which it was said.)*

*On what the two-dimensional model measures — the I and Q halves and the
noise floor:*
> luma carrier amplitude and frequency response (set of FFTs) I of 2DFFT
> hsync shape that persists AFTER the luma is frequency corrected (set of FFTs Q of 2DFFT
> noise floor of the tape is a smooth known roll off (probably some magnetic property based on heads, etc.)
> amplitude of the noise floor of the recording residual of the expected noise floor of the tape.
> phase of the noise floor as it relates to the noise floor of the tape.

*On the kernel — the missing part:*
> I need the derivative of the syne edge, apply this to the time dimension

> That was the missing part

> Wiener-derived kernel from the sync edge

*On the phase — differentiating across the whole pulse:*
> I think we need to differentiate across the sync pulse, which will give us the exact phase response of the roll off

*On what the method is (asked, then confirmed):*
> Is this just an FIR applied on the sync pulse?

> To measure it?

> Yeah I think so to measure just the frequency response of the area we have to measure in the sync pulse.

*On the RF site:*
> Oh I think my ringing correction method I tried this way would work now that I am doing it at the rf before any resampling or filtering

*On the chain of references and the loop run the other way:*
> I think we need to differentiate across the sync pulse, which will give us the exact phase response of the roll off over time. This is the trace of amplitude to expected response (flat), to expecte roll off (causal), and finally to a similar 2d fft fir combo to derive the rolloff's slope (is this going back around the loop of information the other way?

*On iterating the chain:*
> Look around this chain of similarities until all noise is removed and you have perfect measurements in the derived areas based on their surrounding connections.

> I think this set of measurments is cycled around a few times until all residual is removed, and approaches zero, or the noise floor.

*The cycling rules:*
> Right — with "or the noise floor" being the operative clause. The residual should converge *to* the floor, not through it. A few practical things about the cycling itself:
>
> **It converges fast.** With well-conditioned constraints like these, two or three passes usually get you within the noise; if you're still seeing meaningful movement on pass five, that's a sign of a weakly determined node (usually the spacing-from-ratio step, which is sensitive when both bands' losses are small) or of a model error the graph is chasing.
>
> **Order matters for the first pass.** Start with the nodes that don't depend on the others: the noise PSD from blanking lines, then TBE from sync and burst (pure timing, no filter involved). Correct timing before anything spectral, because misaligned edges fake an HF roll-off and you'd deconvolve a ghost. Then the sync-edge deconvolution, then the pair ratio for spacing, then emphasis last since it's the residual that survives everything else.
>
> **Hold out lines to tell "at the floor" from "below the floor."** Fit on even lines, measure the residual on odd lines. While the held-out residual is still dropping, keep cycling. When it flattens and the fitted residual keeps dropping, you've started fitting noise — stop one pass earlier. This is the one test that can distinguish a real improvement from the graph absorbing its own noise, and it costs nothing.
>
> **Watch for the residual that doesn't go down.** If the held-out residual plateaus above the noise model's prediction, the graph is missing an edge. Compare the residual on the sync edge to the residual on a white-level transition; if they differ, that's the level-dependence edge — and it's a node you'd add to the graph rather than something more cycling will find.

*On the function this becomes, and the separation-filter mismatch:*
> If this works, I think we will have a function that does this chain of correlated corrections multiple times on the raw rf. The demodulated luma is the luma after the chroma notch filter is applied. There is going to be a frequency response that is summed if there's a missmatch between those filters. Essentially, this process is needed for the luma, for all linear ringing components.

*On moving the correction to the raw RF:*
> Moving the correction to the raw RF is the strongest version of this, and it plausibly explains the level-dependence problem rather than working around it.
>
> **Why RF is where it's actually linear.** The head/tape/preamp chain and every filter before the demodulator are LTI on the RF. But the FM demodulator is not linear, so a fixed ringing in the RF band shows up in baseband as distortion that depends on where the carrier is sitting — 3.4 MHz at sync tip, toward 4.4 MHz at white. A single baseband FIR derived from the sync edge can only be right at sync level; the same RF ringing produces a different baseband signature at white. Correct before the demodulator and one filter should fix all levels at once. That gives you a clean convergence test the baseband version never had: when the RF correction is right, the baseband residual on sync edges and on white transitions goes to the noise floor *together*. If they don't converge together, whatever's left isn't linear.
>
> **On the notch/separation filter mismatch.** Yes — the band-pass that pulls the FM out from the chroma-under has a transition band that sits in the FM's lower sideband, and its magnitude and phase ripple sum straight into H_RF. Two things follow. Measure and correct at the same point in the chain, immediately before the demodulator, so that filter is inside the path you characterize. And since the filter is yours, don't estimate it — you know its response exactly, so either compensate it analytically or make it linear-phase FIR so it contributes no group delay at all. Every known filter you can take out of the graph analytically lowers the noise on everything you still have to estimate.
>
> **What the RF references look like.** The known properties change form at RF. The sync edge is no longer an amplitude step; it's a frequency step of the carrier. The constant-envelope property becomes your amplitude reference: any envelope variation on the FM is |H_RF| at the instantaneous carrier frequency, and picture content sweeps the carrier across the band for you. Phase at RF is harder to read directly — the practical route is to parameterize H_RF (magnitude from envelope, phase as a fitted model), apply the inverse, demodulate, and drive the parameters by minimizing the baseband residual across levels. The demodulated sync edge and white edges become the objective, the RF filter the variable.
>
> **What stays outside "all linear ringing."** The record-side nonlinear emphasis is level-dependent by design, and no linear filter at any point in the chain undoes it — after the RF correction converges, that's the ringing-like component that will remain, and it needs a level-dependent de-emphasis in baseband. Head switching, dropouts, and FM threshold noise are also outside the linear model. Everything else — heads, tape, preamp, separation filters, demod input filtering, de-emphasis, notch — composes into one H_RF and is covered.
>
> **Practical cost.** The RF is at the cxadc sample rate, so cycling the full chain and re-demodulating per pass is expensive. Fit the filter parameters on a subset of lines with a fast residual measurement, then apply the final correction once to the whole capture. Numba will handle it, but you don't want to demodulate a full tape five times to converge a dozen parameters.

> Correction, RF is at known sample rate, not the cxadc one.

*The principle, whole:*
> I am using these components on my VHS tape: the expected characteristics of the signal (luma, chroma, tbc), vs. the characteristics of the received signal (luma, chroma, tbc), and differentiating that to remove all possible frequencies encoded in the signal that do not match my characteristics. This will go as far as to correct the signal all the way to the source characteristics of the signal. This is a continuous refinement of all those different components, derived together multiple times.

*The loop's operating definition:*
> I've already defined it in my session, feed forward the components before doing the measurement with them again. They should all feed forward together until the residual is zero.

*Freeze and exclude:*
> Once I have the RF part of the tape taken care of, I need to no longer differentiate over that part

*The review request and the theory:*
> Review this and check if it matches what I have in the code. I think we can simplify this significantly by just doing this process on it's own as the filtering stage. This will remove all issues to the extent that the tape and rf bandwidth have information.
>
> Yes, and it sits at the intersection of a few well-established fields. Information theory is the one that sets the *limits*; the *method* comes from estimation theory and inverse problems.
>
> **The method — what you're actually doing:**
>
> - **Pilot-based channel estimation / system identification.** Sync and burst are training sequences: known transmitted signals used to identify the channel. Every digital modem and every OFDM receiver (Wi-Fi, LTE) does exactly this — measure the channel from known symbols, invert it for the unknown ones. The analog video engineers built the pilots into the signal for other reasons, but mathematically they're the same object.
> - **Wiener filtering / deconvolution.** Norbert Wiener, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series* (1949). The regularized inverse you're already using is his; the noise-PSD-as-regularizer idea is the core of it.
> - **Projection onto convex sets (POCS).** Youla & Webb, "Image Restoration by the Method of Convex Projections" (1982). This is the closest match to what you've been describing: you have a set of known constraints (bandwidth, causality, known reference shapes, level range), each defines a set the true signal must lie in, and you iteratively project onto each set in turn until the estimate is consistent with all of them. It converges to the intersection. Your "feed forward all components and re-measure until residual is zero" is a POCS iteration.
> - **Factor graphs and belief propagation.** Kschischang, Frey & Loeliger, "Factor Graphs and the Sum-Product Algorithm" (2001). The formal version of the constraint graph I laid out earlier — nodes for unknowns, edges for known relationships, messages passed around until consistent. Also the EM algorithm (Dempster, Laird & Rubin, 1977) for the alternating-estimation form.
>
> **The limits — where information theory comes in:**
>
> - **Shannon's channel capacity** says how much information survives a noisy, band-limited channel. Anything the tape/head path pushed below the noise floor, or put through a null in |H|, is not in the recording. No estimator recovers it; a prior fills the gap with a plausible guess.
> - **The data processing inequality**: no processing of the received signal can increase the information it carries about the source. Your loop makes the *estimate* better; it cannot make the *data* contain more.
> - **The Cramér–Rao bound** (estimation theory, not strictly information theory, but it's built on Fisher information) gives the minimum variance any unbiased estimator can achieve from N noisy measurements. That's the "√N over independent lines" floor. Your loop can reach it; nothing reaches below it.
>
> So to answer directly: the principle is estimation theory applied to an inverse problem, with a Bayesian/constraint-satisfaction structure. Information theory is what tells you that "completely restore" means "to the Cramér–Rao floor on the parameters and the Shannon limit on the content" — which is complete in every sense that matters, but is a floor, not zero.
>
> If you want one reading list entry that covers the whole thing in one place, Kay's *Fundamentals of Statistical Signal Processing, Vol. 1: Estimation Theory* is the standard text — Wiener, Cramér–Rao, and iterative estimators all in one framework.

*The operational identity:*
> Shannon's channel capacity = luma frequency response; The data processing inequality - residual rolls of to zero; the residual is the kernel to apply.

*The information boundaries:*
> The Cramér–Rao bound, 8 bit amplitude steps in my capture device; The Cramér–Rao bound, input sample rate, these are the information boundaries that exist in the RF channel. The other information boundaries are defined for each spec, i.e. rf capture, magnetic saturation, tape loss, television signal type bandwidth limits, and then try to derive the rest to get the noise profile over time.

> This is the remainder

> They are another component to this set of measurements

> Essentially every component of the system is identified, since every component is known, and the residual will be the decay rate of those components, i.e. their FIR

*The stopping rule, the exhaustion test, and the circularity resolved:*
> Agreed — that's the right stopping rule, and it's the standard one for model-based estimation: fit every component you can identify, and what's left defines the effective noise floor for that recording. The unmodeled sources are what make the floor a floor rather than zero.
>
> One refinement that follows from your own framing. "Knowable" isn't a fixed list; it's whatever the residual still has *structure* in. RF interference is a good example: a stray carrier or switching-supply harmonic isn't noise, it's a narrowband line in the residual spectrum with a stable frequency. That's identifiable — and removable with a notch — even though it isn't a property of the tape or the VCR. Same for hum, head-switch transients, and anything periodic. So the loop's exhaustion test is: look at the held-out residual; if it has stable spectral or temporal structure, there's another component to add; if it's structureless (white or smoothly shaped, uncorrelated line to line), you've reached the floor for that tape. That's a testable criterion rather than a judgment call, and it's the same test as before — fit the residual, see if the fit is stable.
>
> On circularity: it wasn't about whether all components are known. It was specifically about using the measured residual as its own constraint set, which returns the residual unchanged regardless of how many components are modeled. Using the noise model as the set, which is where we landed, resolves it. With that, the process is well-posed: each pass either identifies a new component from residual structure or confirms the residual is structureless, and it terminates when the second happens.

> It is done once all the possible and knowable components are exhausted. that's when the process is done. It would only be circular if you knew every component, which we don't like rf interference and other external and un-modeled interference sources.

*The directive:*
> Let's implement that solution.

*The pattern — no frequency profile needed:*
> Do you see the pattern in this? I don't think I need to know the frequency profile at all, I just iterate until the residual is gone

*The noise model is amplitude-only:*
> I think the noise can only represent amplitude though, since we don't have the phase of the noise

> The noise only represents the amplitude loss at this point, since we have no way to continue deriving a residual with phase, since the noise is distributed over some statistical curve

*The specification of the function (given on rejecting the first plan):*
> I want you to formulate a function that models the examples I have given for the different components to model.
>
> I am doing a 3D signal restoration using the various known components of tape, and then find the 2d differential at the end when I have reached the noise floor.
> Each component has a unique criteria that is used to measure it, listed below:
>
> Pure RF frequency and phase cutoff extrapolation
>    * Amplitude resolution
>      * Unit: (unit is steps in 8 bits, needs an input flag to get this info, for now, just hard code to 8 bits)
>      * Min: 0 (DC)
>      * Max: 255 for the initial RF 8 bits
>    * Frequency resolution -> derive from nyquist therory
>      * Min: 0 (DC)
>      * Max 20MHz (nyquist), but I will have a 13.3 roll off applied at the capture stage
>    * Time resolution ->
>      * Total: Number of amplitude measurements
>
>    * Amplitude residual -> frequency response of the rf capture itself (not known at this time, we'd have to feed back the final ENCODED tape signal), and frequency response of the
>      * frequency response over time netweem the difference between ideal nyquist the frequency response of the hardware roll off filter.
>    * Frequency residual ->
>      * differential between frequency response over time netweem the difference between ideal nyquist the frequency response of the hardware roll off filter.
>    * Time resolution ->
>      * The final noise floor residual possibly, or none if we want to stop here, this should be good enough
>
> Process
>
> 1. Apply the 3d signal restoration iteratively until there is no residual left (i.e. once we have matched the low pass filter's response)
>    * Iterate on the RF using this method which will give you the exact incoming residual curves for the luma
>    * I think we are creating a set of anti-residuals to apply to the signal during demodulation.
>
> Demodulate using the residual frequency response of the channel measured from the final residual.
>
> 2. Apply the technique above to the luma using the components I have already identified.
>
> Comments on the plan:
> [Re: "exactly as written"] I need to include subtracting out the noise floor from my block of data that I gathered these two fm signals in. This noise floor represents the last component, amplitude is the level compared to the corrected sync pulse, and the residual. I can integrate to infinity on the sync pulse shape to model it exactly

> Also, amplitude resolution is only 8 bits, so our effective rsolution is 8 bits x 40 MHz sample rate

*The name of the method:*
> multidimensional information extrapolation using the process we described

*On the channel identifier as the baseline, the time axis, and the time-base
residual — given on rejecting the second plan:*
> I also want the other plan and it needs to feed into this one as the channel identifier, the channel identifier is the baseline to measure against when performing the differentiation between the frequency response. The time part is corrected by the hsync and chroma burst lock in other parts of the code. The downscaling removes the residual RF phase component, since it's based on the distance between the sync pulses. Additionally, do the dirivation process on the time base correction, since that can also be helped by the wow factor amplitude difference. That is the time base residual actually, what keeps phase transferring through the residual. THIS was the missing part

*Relayed, not said to this lane directly — Ethan's ruling on the sync
depth, quoted as the head-switch lane relayed it:*
> The sync depths should be adjusted so they are flat. Instead of reducing the ringing components, this might also mean adding back in the energy lost.
