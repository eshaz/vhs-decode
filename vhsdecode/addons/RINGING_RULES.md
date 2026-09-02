# Ringing correction — standing rules (consolidated from all feedback)


Every rule below is a standing directive from Ethan, consolidated so none
gets missed. Check each round against this list before decoding.

## Hard constraints (never break, never trade away)

1. **Sync-only calibration.** The correction derives from HORIZONTAL sync
   pulses only — no equalizing pulses, no vsync/serration pulses.
2. **Strictly causal.** All ringing originates in one direction, from
   physics. No pre-echo, no two-sided kernels (a "reflection before the
   edge" is the tell). If something looks non-causal, the phase is late —
   fix the phase, don't relax causality. A bulk timing advance is
   allowed; response before the driving edge is not.
3. **No blur.** Blurring the picture is never the solution. No detail
   loss: broad-band attenuation, widened targets, or softened transients
   are all rejected forms ("the solution is not to blur the entire
   picture"; "I don't want to lose any detail"). Ring removal must come
   from phase correction and narrow measured-resonance damping, not from
   band attenuation.
4. **Temporally stable correction.** The channel's ringing is a fixed
   property of the deck: measurements are accumulated and averaged over
   the whole decode (a 2-D stack of pulses), and the applied correction
   mayt must not change field to field. Never remove the temporal
   averaging/accumulation machinery in a refactor.
5. **Noise-aware averaging.** Averaging must account for how noise
   perturbs both amplitude and phase of what is being estimated
   (measured stack variance, significance against the fit covariance,
   noise-vs-N bookkeeping). Never model noise as signal.
6. **Level neutrality.** The correction must never shift levels or sync
   amplitude (sum-zero oscillatory corrections / DC-unity kernels /
   amplitude-tilt invariants). Partial gating must not create local
   level shifts.

## Design principles

7. **Holistic solutions only.** No patches, no stacked guards, no small
   fixes on symptoms. When something fails, find the mechanism and
   rebuild the principle; guards that encode one tape's numbers are
   rejected.
8. **One generic process.** The goal is a single process correcting all
   ringing, ghosting, and group-delay distortion on any tape. No
   per-tape configuration; every constant is either measured from the
   signal or comes from broadcast spec (e.g. sync width 4.7 µs, sync
   depth 40 IRE — the sample rate is derived from the sync width, never
   hard-coded; PAL must remain possible).
11. **Keep the sub-sample alignment machinery** (temporal sub-sample
    alignment fixes are load-bearing; do not remove in refactors).

## Process & validation rules

13. **Decode and show every round.** Nothing counts until it is decoded
    and viewable ("If there's something to test, I need to see the
    result"). Validate on ALL tapes (cd, home, pnb) every round — each
    sample exercises a different aspect.
14. **The eye is the acceptance authority — absolutely.** Instruments
    must be able to see what the eye sees before their readings are
    trusted: integrate the way the eye does (down vertical edges, over
    time on static content), and judge by the tails (p90/max), not
    medians. When the user reports a visible artifact and instruments
    read clean, the instruments are 100% wrong and must be extended
    FIRST, before any further design work (proven three times).
14a. **Run the FULL battery every round and report the full matrix.**
    Never a select few measurements that fit the current hypothesis:
    every tape, every gauge, every band (including LOW frequency —
    ghosting lives below the ring band), every region (back porch,
    post-bp active, borders, content edges), ABSOLUTE levels as the
    primary quantity (the eye sees absolute ring, not deltas vs
    uncorrected — "no added ring" is not success while the disease
    remains visible), plus sharpness, pulse_trace, vsync.
18. **Tooling:** one readable file per tool, readable variable names,
    no configuration via environment variables or knobs buried in big
    JSON. Snapshots of every round under
    `/output/claude_validation/versions/` + NOTES.txt; git is
    read-only for the assistant (the user commits).

## Current approach context (Aug 2026)

The correction direction is the cross-spectral (Wiener-class) path:
per-line cross-spectra are shift-invariant and see the full channel
response where aligned averages see only the locked fraction. The next
realization must satisfy rules 2 and 3 simultaneously: causal, and
sharp — full measured phase correction, damping only narrow measured
resonances, never broad bands. The parametric machinery is reserved
for what LTI cannot represent (polarity asymmetry, level dependence),
subject to all rules above.

## Addendum (Aug 11 2026, from Ethan)

19. **Reference-level boundary.** The front porch and sync tip are
    reference levels: the porch is flat 0 IRE and then the standard
    0 → −40 IRE falling edge into the tip — no other assumptions.
    Structure on them is disease, driven by the preceding active
    content, and measuring that reaction (with the preceding content
    as the drive/stratification variable) is legitimate calibration.
    Active-area pixels themselves remain validation-only as picture
    observables. (This sanctions the porch-fall instrument class:
    stratify line-end falls by their content level, observe in
    blanking.)

20. **No hard-coded constants.** Every quantity in the shipped module
    must be data driven (measured from the tape or the decoded
    signal) or derived from the supplied specifications / system
    specifications (the decoder's format parameters, the video
    standard's timing and levels). Structural engineering numbers
    that remain (resolution gates, loop conditioning) carry their
    derivation or rationale in a comment at the definition.

21. **Video and engineering terminology.** Code, comments, and
    reports use standard video/broadcast and signal-engineering
    terms: front/back porch, horizontal sync, blanking interval,
    active picture, luminance transient, FM deviation, carrier,
    pre-emphasis/de-emphasis, overshoot/undershoot, settling,
    template, servo. Project-internal slang (e.g. "disease",
    "dose") is retired.

## Addendum (Aug 12 2026, measured — see RINGING_HARMONIC_ARC.md §14)


23. **Never measure a sync-region response on a corrected decode.**
    The completion template is subtracted over lags 2–48 after the sync
    crossing — exactly where such a response is read. Measured
    footprint: 1.04 IRE mean / 1.90 max at the sync tip and 3.74 IRE at
    the sync fall, against 0.23 IRE on the front porch. That is 2–3×
    larger than the effect it was masquerading as; on uncorrected
    decodes the apparent effect collapsed 5.08 → 1.16. Use
    `--inverse_eq -1` references for any sync-region measurement.

