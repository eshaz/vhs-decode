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


## Addendum (Sep 2 2026, measured — the identification loop)

24. **Check a decode's PRODUCTS, never its exit status.** The decoder
    can fail and still exit zero: its json writer dies in
    `lddecode.utils._consume` with `'NoneType' object has no attribute
    'items'`, the run reports success, and a downstream tool reads a
    truncated or absent output. Any automated loop must require the
    `.tbc`, the `.tbc.json` and whatever sidecars it asked for to exist
    (and the field count to match `numberOfSequentialFields`) before it
    believes a decode. Two decodes must never share an output prefix
    either: the second trips over the first's half-written files, which
    is how this was found.

25. **The tbc line starts AT the sync fall.** A front-porch window
    written as an offset from the line start lands in the previous
    line's active picture. Read the porch as the last samples BEFORE
    the line boundary and the sync tip just after it, and derive both
    from the format's own measurement plan rather than by eye. Booked
    because it inflated a measured sync-to-burst wander by 5×
    (1.7 samples against a true 0.33).

26. **The demodulated baseband sees the RF magnitude relative to the
    CARRIER'S own gain.** For an applied change `e` of the RF
    log-response, a landing at carrier `fc` moves by
    `d ln|R|(f) = [e(fc+f) + e(fc−f)]/2 − e(fc)` in magnitude and
    `[e(fc+f) − e(fc−f)]/2` in phase. Attenuating the carrier itself
    therefore BOOSTS every baseband frequency, and two landings
    0.29 MHz apart (sync tip and blanking) share bins — one is the
    other's sideband. Any RF-domain fit must solve for the response
    jointly across landings; a bin-by-bin deposit cannot converge.

27. **The RF equalizer never reaches the color-under.** For
    color-under formats `demodblock` takes the chroma from the RAW
    block (`chroma_source = data`), not from the equalized spectrum, so
    an RF-site correction changes the luma only. A chroma correction is
    a separate application at a separate site, and a timing correction
    carried across the heterodyne must be scaled by
    `f_subcarrier / f_color_under` (5.6875 on NTSC VHS), because a
    delay is a phase of 2πfτ while heterodyning only translates.

28. **Average a prediction the way the measurement averages it.** A
    field-wide estimate is made while the carrier sweeps with the
    picture, so a prediction evaluated at one carrier is not comparable
    to it. Weighting by the carrier's measured occupancy moved a
    cross-instrument comparison from 1.9σ to 0.1σ without changing
    either measurement. Where the two instruments cover different
    ranges, report the coverage fraction with every number and claim
    nothing from a band the covered subsample is biased in.

29. **The weaker instrument sets the comparison.** Quoting a small
    formal error against a measurement whose signal-to-noise is near
    one, and calling the gap a systematic, is an error of statistics
    rather than of physics. State how far each candidate sits from the
    weaker measurement in ITS sigma.

30. **Average signed, not in magnitude, when the measurement fits one
    coefficient.** A regression estimating one complex coefficient per
    band recovers the signed mean. Averaging the magnitude instead
    agrees wherever the sign is constant and differs by a factor of
    four or more wherever it is not (measured: 0.618 against 0.136 in
    one band, 1% agreement in the other six). And a band whose value is
    a cancellation — the quantity crossing zero inside it — is not a
    stable number to compare against: narrow the band first.

31. **A band's response is the ratio of MEAN POWERS, then the log.**
    Averaging per-bin decibels weights the quiet bins as heavily as the
    loud ones and biases the answer wherever the spectrum has
    structure. Measured cost of getting this wrong: a sideband
    imbalance inflated from 0.3 to 0.57 nepers, and a 2.3-2.9 dB
    "rise" manufactured out of nothing and then explained with physics
    that was not there. The corrected measurement was both smaller and
    a different shape.

32. **Content dependence in a channel ratio is evidence, not noise.**
    For a linear time-invariant channel the output spectrum is the
    response squared times the input's, at every frequency, whatever
    signal carries it. So if the same ratio differs between signals,
    either the path is not time-invariant or the two measurements
    differ by more than the channel. Report the between-signal spread
    beside every such number, and treat a large one as a finding to
    chase rather than an error bar to widen.

33. **Apply rule 31 at every level of aggregation.** Computing each
    band correctly as a power ratio and then averaging those bands as
    decibels reintroduces exactly the bias the first step removed. Both
    lanes made this error at the second level within hours of agreeing
    the first, and it produced a 1.5 dB disagreement on one signal and
    a false claim that a channel ratio was content-independent to a
    tenth of a decibel. Aggregate powers, then take one logarithm at
    the end.

34. **The general form of 31 and 33, and their boundary.** Average a
    quantity in the domain where the thing being estimated is linear:
    POWER for a power ratio estimated from noisy spectra, LOG for a
    multiplicative transfer being inverted. Where the two differ,
    prefer the one that errs in the safe direction for what the number
    will be used for. Rules 31 and 33 are the two commonest failures of
    this principle, not exceptions to it. The baseband fold smooths the
    logarithm of its transfer over the instrument's resolution
    deliberately and correctly: its quantity is a transfer inverted
    multiplicatively rather than a power ratio, its bins are
    well-determined rather than noise-limited, and its average runs
    over adjacent resolution cells rather than across a band - and the
    geometric mean sits below the arithmetic one, which is the
    direction an equalizer should err. A reader armed with rule 31
    would be right to challenge that code and wrong to change it.

35. **A residual below the gauge's own error is a warning, not an
    achievement.** A per-line correction can drive its residual under
    the measurement noise by absorbing that noise into the geometry it
    corrects, where it stops being a number and becomes real jitter in
    the picture. The residual cannot detect this; the direct test is
    the CORRECTION's own content above the band the physical mechanism
    can produce. Measured here: an unregulated time-base correction
    reached a chi-square of 0.21 while its above-flutter-band content
    grew every pass to parity with the in-band part, and band-limiting
    it to the drum window left a genuine improvement of five percent
    where the raw residual had claimed fifty-three.

36. **A comparison is only a comparison when both sides are computed
    over the same set.** Four errors in one evening had this shape and
    every one looked like physics: an evaluation band that moved
    between passes, a mean taken over decibels, a mean taken over
    sub-bands, and an admitted bin population that GREW as the
    correction improved because bins began passing the quality gates
    the correction itself had improved. The last is the subtlest,
    because the growth is automatic and invisible: the newly admitted
    bins are the band edges, where the residual is worst, so a metric
    can rise while every bin improves and a fit can be driven by
    exactly the bins it should trust least. Freeze the population on
    the first pass - for the fit AND for the metric - and judge
    everything later on the set that was judged at the start.

37. **A share must carry its definition with it.** A fraction always
    looks comparable even when its numerator and denominator are
    different objects, so two lanes can quote "the common share" and
    mean three different quantities measured at different sites on
    different products. Two of them agreeing is then arithmetic
    coincidence, which happened twice in one evening before it was
    caught. State what is over what, and where it was measured, every
    time a share is quoted. Absolute measurements do not need this;
    ratios always do.

38. **`is_first_field` IS the head identity - use it.** Ethan's rule: it
    matches the video head parity and stays consistent for the entire
    period of a decode. It can swap only on a significant recording
    change, such as EP to SP, and the response to that is to REBUILD THE
    MODEL, not to relabel the fields.

    An earlier version of this rule said the opposite, on the strength of
    a colour-under "head signature" that turned out to carry no absolute
    information at all. Measured per field, the colour-under's per-line
    rotation reads `+-+-+-` starting with `+` on the FIRST DECODED FIELD
    of every decode, whatever that field's parity is - so it is aliased
    with the decode's own field index and identifies nothing. The one
    position of thirteen whose per-head gain came out reversed was also
    the only decode that began on `is_first_field=True`; that, and not the
    tape, is the whole explanation.

    The lesson that survives is about the instrument, not the label:
    before trusting a quantity as an identity, check that it is not simply
    counting from wherever the measurement started. A signature that
    always reads the same on the first sample is an index, not a
    signature.

38a. **A bias measured at several positions of ONE tape is that tape's
    repeatability, not the decoder's bias.** The per-head sync-tip gain
    read +0.647, +0.658 and +0.635 IRE at three positions of the home
    tape - agreement to +/-0.012 that looked like proof of a fixed
    window-placement error. Folded across three tapes it reverses: home
    climbs, pnb and bars FALL, and the same window reads above the late
    porch on two of the three. Three positions of one tape is one sample.
    Repeat on another tape before the word "systematic" is used.

39. **The carrier law's null space has two members, not one.** The
    demodulated magnitude follows [e(fc+f)+e(fc-f)]/2 - e(fc), which
    annihilates any constant AND any linear function of frequency. A flat
    gain is a constant; Wallace's spacing loss is -2*pi*d*f/v, linear in
    frequency exactly. Measured through the model, a 1.5 dB gain
    difference reaches demodulated video as 0.0e+00 nepers and a 0.40 um
    spacing difference as 1.8e-15, while gap, thickness and azimuth come
    through at 0.15 to 0.32. A parameter left free in a null direction is
    not merely ill-determined, it is unconstrained and takes whatever the
    noise asks for - so both are refused by `fit_head_difference` and
    belong to the RF envelope, where a level is a level and a slope is a
    slope.

40. **Projecting out the mean IS fitting a constant.** Mean-centring both
    sides of a fit is algebraically identical to giving the model a free
    constant and discarding its value. So a quantity that mean-centring
    removes was never "absorbed" by the other parameters - it was computed
    and thrown away, and naming it leaves every other fitted parameter
    bit-identical (verified to 1e-12). The fix for a discarded quantity is
    to report it, not to re-attribute what it supposedly displaced.

41. **A drive proxy can be anti-correlated with what it proxies for.**
    Ranking captures by the standard deviation of their combed chroma
    power - the obvious stand-in for "how hard is this material driving
    the coupling" - orders them BACKWARDS against the measured
    resolvability: ramp has the highest value (6.91) and the worst null
    ratio (1.3x), chromanoise the lowest (1.58) and the best (14.5x). The
    statistic is normalised by the mean power, so on material carrying
    little chroma it measures the relative size of the NOISE rather than
    the signal. The lesson generalises past this one statistic: when a
    proxy is a ratio, ask what its denominator does on the cases where the
    numerator is small. Where a direct test exists - here the null ratio
    itself - use it and do not proxy at all.

42. **A confident-looking output with no resolvability channel is the
    dangerous failure.** An estimator that returns a plausible shape from
    material that cannot support one, with nothing in its output saying
    so, is worse than one that declines: every consumer downstream reads
    it as a measurement. Where a matched null can be built - destroy the
    pairing, leave both spectra untouched - report the ratio alongside the
    result even when nothing gates on it, so the difference between "this
    is the answer" and "this is what the pipeline returns from noise" is
    visible to whoever has to decide.

43. **Reproducibility bounds noise, not confounding.** An admission rule
    that scales a component by how far two independently accumulated banks
    agree is measuring repeatability, and repeatability answers "is this
    noise?" - it does not by itself answer "is this the thing I named?".
    Where a candidate confound is shared by both banks, their agreement
    cannot separate it from the quantity, and a second, independent
    condition is what does.

    RECORDED WITH ITS OWN LIMIT, because the first application of this rule
    was wrong. It was raised against the per-head luma response, on the
    argument that odd and even fields might carry different content. They
    do not confound it: the head is identified by FIELD PARITY, a property
    of the recording rather than of what is on it, and both heads
    accumulate over the same tape and the same content - so a
    content-driven component appears in both accumulations and CANCELS in
    their difference, exactly so on static material where the fields carry
    identical picture. The independent check agrees: measured on the sync
    tip, where no picture reaches at all, the per-head gain reads +0.98 dB
    at 161 sigma. The general rule stands; that instance of it does not.
44. **Never take the asymptote from inside the window.** Fitting a decay by
    subtracting the trailing median of its own measurement window assumes
    the tail has settled inside that window. On the back porch it has not -
    the round opened with the finding that blanking is never reached before
    active video starts - so the "settled level" is still up the curve, and
    every time constant fitted against it comes back SHORT.

    Measured, on planted truth in a 59-sample window: a tau of 9 samples,
    which does settle inside it, recovered to 0.5 per cent; a tau of 33,
    which does not, came back 29 per cent low. The bias is invisible on the
    short constants and total on the long ones, which is why it survives a
    first look - the estimator appears to work.

    FIT THE OFFSET JOINTLY. For a fixed time constant the model
    A exp(-t/tau) + C is linear in A and C, so a scan over tau with least
    squares at each step is exact, needs no optimizer and no logarithm, and
    recovers all four planted cases to about one per cent. And read a
    RAILED scan as a rail, never as a measurement: a shape with no time
    constant runs the scan to its bound, and the bound is not an answer.

45. **A decay fitted over a window that contains its own driving edge
    measures the EDGE.** The back porch's relaxation is driven by the sync
    rise. `window_plan.back_porch` begins one sample after the rise
    midpoint, so a joint fit over it must span the step, and it answers
    with a time constant long enough to cover one: 2.94 us. The same fold,
    read from after the rise's 10-90 transition and settle, gives 1.25 us,
    and the fit rms falls by a factor of 2.8 - the excess was the edge.

    THE 2.94 THEN MANUFACTURED A SECOND FALSE RESULT, which is how the
    error nearly shipped. At that time constant it matched a section the
    stage already fits at pole 0.976450, against the measurement's
    0.976526 - agreement to four decimals, which read as proof of a double
    correction. It was two fits of the same edge. The real decode A/B says
    the opposite: the shipped correction removes one per cent of the tail
    on head A and seven on head B.

    Derive a fit window from the geometry, exclude the transition that
    drives what is being fitted, and treat a striking agreement between
    two fits as a reason to check they are not both fitting a third thing.

46. **The installed console script is not the working tree.** `vhs-decode`
    resolves to `/opt/venv/.../site-packages/vhsdecode`, a snapshot that
    can be arbitrarily old - the copy present during this round has ZERO
    occurrences of `inverse_eq`, so it predates the entire ringing arc and
    silently decoded without any of it.

    A decode run that way tests neither your change nor anyone else's. Run
    the tree (`PYTHONPATH=/workspaces/vhs-decode`, calling `main()` - the
    module has no `__main__` guard, so `python -m vhsdecode.main` exits
    zero having done nothing), or reinstall - but reinstalling changes a
    venv the other lanes are also running from, so prefer the path.

47. **Isolate your change from the other lanes before claiming no-harm.**
    With several sessions holding uncommitted work in the same tree, a
    baseline decode from yesterday differs for their reasons as well as
    yours, and `cmp` against it proves nothing either way. Neutralize your
    own entry point instead - monkeypatch it to a no-op in a runner - and
    compare two decodes of the SAME tree that differ only in that. It is
    the only comparison that isolates one lane.

    THE ASSERTION IS THE LOAD-BEARING PART, not the strip (23's finding,
    and it had already bitten this runner). A gate that removes a feature
    by patching or stripping it passes VACUOUSLY the moment its target
    stops matching: nothing is removed, the two arms become identical, and
    the gate reports exactly the byte-identity it exists to look for. This
    lane's runner patched `measure_porch_relaxation` while the applied
    path had moved to `accumulate_porch_relaxation` - it would have
    certified an unisolated build. So the runner must REFUSE when a target
    is absent rather than silently patch nothing, and 23's strip-based gate
    is right to assert a count of what it removed and stop when it no
    longer understands the tree. Detecting that you no longer understand
    the tree is a better outcome than a pass.

48. **Measure a correction on the signal it will LAND on, not on the one
    upstream of it.** The back-porch relaxation was fitted on the raw
    accumulated fold and subtracted from the ringing-corrected field. The
    raw tail is -2.41 IRE on head A and -2.48 on head B; what SURVIVES the
    ringing correction is -1.67 and -0.58. Subtracting the raw figure
    flattened head A by 60 per cent and made head B 21 per cent WORSE with
    the sign reversed.

    The heads diverge because the stage's own fitted sections differ
    between them, so the amount already removed differs - which means the
    error is invisible on whichever head the upstream stage happens to
    touch least, and looks like a working correction there. Both polarities
    improving TOGETHER is the test that catches it, and it is the user's
    own criterion for a correction applied before the demodulator.

    Measure after every stage that precedes you, accumulate across fields,
    and take the measurement BEFORE your own subtraction so the loop is
    feed-forward and never stimulates its own kernel.

49. **A gate that never reaches its stage is indistinguishable from a
    working no-harm gate.** `--stages +ringing.relaxation` parsed,
    validated against the declaration, and was silently ignored: the
    selection lives on the Options namedtuple while `process_field`
    receives only its own `shared_state` dict and no rf reference. The
    decode came back byte-identical with the flag ON - which is exactly
    what a correct default-off gate looks like from the outside.

    So the no-harm check is only half the pair. ALWAYS run the flag-ON arm
    too and require it to DIFFER; a component that changes nothing when
    enabled has not been shown to be safe, it has been shown to be absent.
    Seed a selection into the stage's own state the way `channel_eq` seeds
    `channel_eq_active` (channel_eq.py:219).
