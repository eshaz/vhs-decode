# Multidimensional information extrapolation — the proposal

Ethan's proposal, in his own words, in the order he stated it (2026-09-02).
Each quotation is exact; the italic line above each is only the context in
which it was said. Nothing is paraphrased.

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

*On carrying the residuals through the downscale:*
> We need to make sure we downscale the amplitude, time and the amplitude, I think keep the existing residuals, but downscaled to 4fsc, each channel is a residual that we will downscale based on the final TBC.

*On the chroma channel:*
> I think the correction will happen on the chroma channel as well using the component I've identified there.

*On the per-field dimension:*
> Let's track the non-reducable residual between fields, this will be our per field dimension we can use to differentiate on as well

*On the 3-D differential, the three complex components, and the staggered application:*
> Can I do the 3d differential between the amplitude, frequency,and time where possible. I think I'd need 3 complex components to iterate on. Since I have the luma and chroma, there shoud be a part in this process where I have the complex component on all three. Time which is measured using the hsync and the color burst phase. I think I can sub in the higher level components as I iterate down and remove the residuals for each known component. Essentially this staggers their application.

*On the time base correction as a derivable component:*
> The time base correction is the sync resampler. so that now becomes a dirivable component in this information.

> Since we can get the residual from it each time.

*On the model as the exact RF correction, the alignment parameters, and the demodulated burst residual:*
> Another way to think about this is the model that we are building is the exact amplitude and phase correction we need to apply to the RF. Since we model in 3d, sync pulse alignemnt and shape, and burst and it's shape, are the alignement parameters. Add demodulated burst residual as another residual to use in this measurement.

*On the family the method belongs to:*
> Ths might be something like MIT's rf capture work. or radio holography

> gabor's holography applied to RF and our VHS video signal components and physical properties

*On the clipped sync tip - a missing edge - and the color-under under the luma:*
> The sync tip is perfectly flat. It is possible the VCR clipped it low at record time. I think this was a missing edge case, where the recording VCR clipped the input signal. Now we can use measure the frequency response of the sync area, and possibly exclude the sync tip if the bottom part is flat.

> I think when modeling the luma channel, I need to consider that the color under signal is under it. Perhaps the same process, but filter out (frequency, amplitude, and time) of the color under from the luma channel, just like we currently do on the chroma from the luma during tbc and frequency and phase correction.

*On the chroma correction's scale and where it is applied - the heterodyne ratio:*
> The correction that is happens after the record side noise is measured will affect the color channel differently, since the color is down heterodyned from the source color carrier to the color under. This means that the chroma's correction will need to be scaled depending on the ratio of the system's compositie color carrier frequency and the system's color under carrier frequency. Essentially, the correction on the color under will need to be done after it is up heterodyned once we have fully modeled and cancelled the frequency response error of the signal after the color was down heterodyned.

*On what the loop actually produces - the modelled signal, and the limit:*
> The key point is that all three components influence each other. Each time the residual is calculated, we feed forward this residual calculation back into the expected signal. Each time the residuals are calculated they are cross checked and added back into their respective residuals. The residual difference is accumulated in the expected signals. I think the modeled signal is what the result is?

> I think we are iterating to take the limit of the residual to find the accumlated shape of the expected side. The iteration is done when the limit is reached, i.e. residual approaches 0

*Correcting the band-limit reasoning, and on the time base as part of the loop:*
> Your determination: "A tape's speed cannot change from one line to the next, so all of that is measurement noise being written into the geometry." is false. It can, if the time base was not fully corrected. The time base correction, i.e. the current downscale function needs to take also perform the luma eq while the time base correction is happening. I think the TBC needs to happen on each residual and feed forward in all 3 dimensions.

*On the clipped sync tip, the rising edge, and using all dimensions at once:*
> Yes, focus on the rising edge only when running the step after the TBC is done. the sync tip is clipped when the signal entered the recording VCR and correctly resolves to flat. The expected sync pulse shape when this clipping is encountered should account for this. Likely we can keep it flat, and focus on correcting the shape of the back porch. Talk to the ringing cancellation session for that. I am ok with the clipped sync tip changing shape if it results in a more accurate back porch residual.
> The key is to use all dimensions and components simultaneously while running the correction and measuring the residual. We can take the limit through iteration to find the EXACT shape of amplitude, frequency, and time that needs to be applied for FULL correction. Also consider multi dimensional DCT and DFT, for modeling the exact amplitude, frequencies, and time base correction of the residual.

*On the up-heterodyned colour burst as its own set of components:*
> Now use the phase of the color carrier to determine the frequency correction. I need to add this as a new set of amplitude, frequency, and time. It's already used for time because of the color burst shape, but I want to model the expected of color burst and find the frequency, amplitude, and time residual using this same method using the up heterodyned color burst. The frequency and amplitude of the heterodyne will also be a component to perform my logic on to.

*On the heterodyne drifting, the second stage, and splitting the chain in two:*
> I think it's possible for the color under heterodyne itself to have drifted, especially in the home tape, so there might be some drift and variation there. I think we should add the extra component measuring the burst: "On the up-converted burst that gives a group delay of 0.14 microseconds and a response of about 0.25 nepers out to 895 kilohertz, with the burst written 3 percent narrower than specified. That last is itself a channel measurement."

> Onto the next part, where we again measure the ideal shape of the sync pulse, except after the currently working RF correction. I need to run the algorithm again on the demodulated and emphasis corrected video. This will allow us to resolve the signal distortion that is observed between the source of the video signal, and the recording VCR. Siuce this applies before the color was separated from the luma, there may be correlation there to align the up-converted chroma with chroma residual in the luma signal. Use this possible residual as a way to continue to correct the chroma and luma channels together. Specifically, on the sync pulse, we need to account for the possibiliy that the sync tip was clipped, if it was clipped, that may distort a bit as we correct the luma to better model it's shape from the origin signal to the VCR. The modeling at this stage should not rely on the shape of the bottom of the sync tip.

> I think the residuals that we accumulate need to propogate entirely through the chain, so that when one set of residuals is refined, it is differentiated amongst the entire ensemble of residuals and their corrections. I think this can be split into two stages though, RF stage (before deemphasis, represents playback distortion only), and the signal to VCR stage, which represents the unique distortion pattern obtained from the source signal to the recording VCR.

*On the bars sample's known heterodyne drift, the comb filter, and cross-hatch:*
> Excellent finding. The bars sample is known to have a drift on the heterodyne, and this is causing a wobble in the amplitude of the chroma. Measure for this and fix it. additionally, this will help you create the difference of the amount of native cross chroma carrier correction to apply. Since the amplitude variation is locked to the heterodyne, this causes an imbalance in the recording VCR's comb filter, which was likely due to a response curve in the frequency response of the tape (measureale using the sync pulse).

> this explains cross-hatch entirely. The color under coler leakage that was phased locked to the sync and color carrier pulse, is the amount of frequency response to change.

*On what the beat's phase error IS:*
> It is a phase varing beat, the phase error is the amount of frequency correction to run

*On the two parts of the difference - line-locked and wobbling:*
> There is probably an absolute line locked difference, but there is also a wobbling difference, this is the frequency response of the source's video frequency tuning

*On implementing it, and on the equalization pulses as the long-drift and low-frequency reference:*
> Excellent implement that, additionally maybe the equalization bars are used for really long frequency drift locking.I do have those pulses at the top of the field. There may be head switching distortion. I think extrapolate the low frequency response of the equalization bars, and apply it as an interpolated constant that does not vary with time.

*On the burst-to-sync phase relationship as the exact phase lock, at the vertical-sync step:*
> What if I am at the step where I am using the vertical sync to get the long equalization? I think the color burst to sync pulse phase relationship determines the phase lock exactly. Add that as an additional component to use

*On whether the model collapses to a closed form or must stay iterative - Ethan's question and his own answer:*
> Does this model collapse into a simple form, or does it need to be done iteratively to invert the number of stages that caused interference.

> It splits: part of it collapses exactly, part of it doesn't, and the split is along the linear/nonlinear line rather than the number of stages.
>
> **What collapses.** Every LTI stage in the chain - head, tape loss, preamp, separation filters, demod input filtering, de-emphasis, notch - composes by multiplication in the frequency domain. Ten stages or one, the product is a single H(f), and its inverse is a single filter. You never invert stage by stage; you never need to know how many stages there were. That's the whole reason the sync-edge measurement works: it sees the composite directly. So for the linear part, the answer is yes, it collapses to one complex FIR (or one pole-zero set), and once identified it's a single application, not an iteration.
>
> **Why you still iterate.** Not to peel back stages, but because the estimation problem is jointly nonlinear in the *parameters*: timing depends on the channel estimate, the channel estimate depends on alignment, the regularizer depends on the noise model, the noise model depends on the spacing, spacing depends on both bands' measured loss. That's a coupled system with no closed form, so you alternate. But it converges in a few passes and the *result* is a fixed set of parameters - after which correcting a new tape frame is a single pass, no iteration.
>
> **What genuinely doesn't collapse.** Stages that are nonlinear or time-varying don't commute and don't compose into H(f):
>
> - The FM demodulator sits in the middle of the chain, so everything before it must be corrected *before* demodulation and everything after it separately. That's the one place the ordering is forced - two filters, not one, with a fixed non-invertible operation between them. This is exactly why moving to RF mattered.
> - Nonlinear emphasis and saturation are level-dependent; their inverse has to be applied at the right point in the order, and it doesn't merge with the linear filters on either side.
> - Tape loss varies per line, so H_RF is really H(f, spacing) - one parametric family, applied with a per-line parameter. Still a single filter form, just indexed.
>
> So the practical shape of the finished thing: two composite linear filters (one RF, one baseband), a level-dependent nonlinear correction between them, and a small parameter vector updated per line or per field. Iteration is confined to the identification phase; the correction itself runs in one pass.

*Naming the algorithm the loop is:*
> successively fitting and removing one component at a time from the residual is matching pursuit, and in the damped-sinusoid case it's essentially the greedy version of Prony

*That the pursuit is not blind - the order is known circuitry, not a search:*
> We just have it further extrapolated out since we know the order in which the different frequencies were added, which is the complete set of differences.

> Essentially we have traversed this graph by knowing which nodes to connect ahead of time.

> Since we know the sequence that when componet had influence over the other, since we are modeling the exact components of this VCR circuitry

> Actually we are modeling multiple stages, each has their own different set of frequency responses and variables

*Confirming the framing to build on:*
> I think this is the current approach we are following. Continue work on this

*On the DC offset as the total response shift, and which stage owns it:*
> We are modeling multiple stages, each has their own different set of frequency responses and variables, the order is known, and I think the DC offset difference btween the expected and actual is the total shift in frequency response of the rf capture device

> Or the thing immediately before it, i.e. the heads

*On what the DC offset actually models - per video head, and independent:*
> It is modeling the DC response of each video, head; we already split video heads out by groups, and their differences do not coorelate

*Challenging the switch away from the symmetric form, and its proper scope:*
> Let's double check that the symetrical orthoganal approach like we had was actually worse. It's greedy, but it cannot extrapolate unknown types of interference, it is expected only to filter interference within the strict frequency bands, frequency response, amplitude response, and time variation of the modeled signals (chroma and luma video data)

*On averaging the measurements over time, with the field's phase alignment:*
> We should apply the averaged 4d to get these measurements but averaged over time. So we should bring back as the field's phase alignment. We need to run all of these parameters over time

*On averaging WITHIN each head to match the per-head response:*
> Now use the average of each head's signal to run the same frequency, amplitude, and time differential to match per head response rather than average it.

*On the head difference being a constant, and where it lives:*
> Is this useing the difference between the two heads frequency response as a constant? Essentially that response will never change since it lives on the playback VCR and the recording VCR

*On fitting the head difference to a physical model of a VHS video head:*
> No, actually fit it agasint the a model of a VHS video head, and begin to research all of video player's head construction and head measurement. We will be tuning for that.

> When fitting the model use the same frequency, amplitude, and time model as described many times here before.

*On tape decode profiles, and pooling across VCRs (noted for later):*
> Multiple tapes on the same VCR will be used as separate sets of video head frequency response we have then we might be able to tune for the VCR (not that that's important, since we're already tuned; we'd just be tuning against variations in those constants. I believe we should add a to do in here to be able to import tape decode profiles, where each frequency response is saved, and able to be imported so the decode can happen again in a separate pass.

> Also, we can pool multiple models of VCR if we have others who have samples from tapes recorded on that VCR, Just make a note on this one.

**TO DO (from the above, not yet built):**
1. *Tape decode profiles.* Save each tape's fitted frequency responses - the
   per-head constants, the channel response, the vertical-interval
   low-frequency constant, the heterodyne split - as an importable profile,
   so a decode can be re-run in a separate pass without re-measuring. One
   readable file per tape, per the standing configuration preference.
2. *Tuning for the VCR.* Several tapes recorded on ONE VCR give several
   independent sets of head response. Their COMMON part is that VCR's own
   signature and their spread is the variation to tune against. Not urgent -
   the chain is already tuned - but it is what the profiles above enable.
3. *Pooling across contributors.* If other people hold tapes recorded on the
   same VCR model, their profiles pool into a model for that VCR. Noted
   only; it needs the profile format above to exist first.

*On tracking the constants over a whole tape to find warm-up after a recording switch:*
> Also, we can apply the same principle to the rate of change over the period of a tape to determine warm up time after a recording switches given the VCR. This may not correlate to the actual content though, since recordings start and stop after various states, data is overwritten some times.

**Added to the to-do list as item 4.** *Warm-up and drift over a whole tape.*
The same constants, tracked as a RATE OF CHANGE along the length of a tape,
should show the recording VCR settling after a recording starts. Two things
it needs that are not in hand: a long decode (the present subsets are twenty
fields, about a third of a second, where a warm-up measured in minutes cannot
appear at all), and a recording-boundary detector, since Ethan's caveat is
that recordings start and stop from various states and tape is overwritten -
so a boundary in the CONSTANTS need not line up with a boundary in the
content, and the constants are the more reliable witness of the two.

*On the RF stage's constants NOT being constant - camcorder battery and temperature:*
> For the RF stage, yes they should [vary over the period of a tape]. This will allow for controls in variations like battery and temperature of the recording device. For example with home, There is considerable variation since it was a camcorder.

This overrides the "constant" framing for the RF STAGE specifically. The
per-head loss is static given the (recording, playback) pair, but the RF
stage's own parameters are expected to DRIFT along a tape, because the
recording device's battery voltage and temperature drift while it records.
The home tape was recorded on a camcorder and should therefore show
considerably more variation than a tape recorded on a mains-powered deck.
That variation is not noise to be averaged out - it is the control variable.

*On inferring the recording device's state from the CONTENT, and profiling it:*
> Maybe I could make the temperature of the device something that I can use given the contents, since I can identify the contents, if the vcr was hot / cold, battery low, plugged in, perhaps profiles that coorespond to AI analyzed video happening in the content, that change the response and noise profile of the device.

**Flagged against the standing sync-only constraint.** The hard rule on this
arc is that the correction derives from the SYNC PULSE only and never from
visible-area content; active-area gauges are offline validation. Letting a
content-derived device profile set the correction would break it, and worse,
it is circular: the content is what is being recovered, so a correction
steered by content can imprint the content onto its own result.

The direction that keeps the rule is the REVERSE one, and it yields the same
profiles:
  * derive the device state FROM the sync-derived parameters - the envelope
    gain, the FM deviation, the fitted head response, all measured in the
    blanking interval - which is exactly what the drift tracking above does;
  * then use the content only to VALIDATE that inference offline (a snowy
    outdoor scene agreeing with a cold-device profile is a check, not an
    input), and as human-readable metadata on the profile;
  * a profile so derived may then be applied, because every number in it came
    from sync.
This gives the profiles Ethan wants, keeps the correction sync-only, and turns
the content from an input into an independent test - which is worth more,
since a covariate that never entered the fit can actually confirm it.

*On scenario selection and the VCR's aging profile, with real recording dates:*
> Depending on what the user selects, the model selects a combination of scenarios when the tape was recorded. This can be measured over the time the tape was recorded, i.e. the aging profile of the vcr, which we can provide since I have the exact dates and time spans the VCR was active.

**This is the right kind of covariate, unlike the content-derived one above.**
Calendar dates and the VCR's active spans are EXTERNAL metadata: they are not
derived from the signal being recovered, so indexing an aging profile by them
is not circular and does not touch the sync-only constraint. It also supplies
what a single decode cannot - a span of years rather than a third of a second,
which is the only way a genuine aging trend can be separated from a random
walk. Ordering the tapes by recording date turns the per-tape constants into a
time series of the VCR's own condition, and the head model gives that series
physical units: an effective spacing in microns growing as heads and tape wear.

*On deriving the interval between recordings from the same measurements:*
> Use the same process to derive how many years apart these recording were taken, we can already find this out by looking at video content

This is a clean test design, because the content supplies ground truth that
NEVER enters the fit. The estimate comes from sync alone (the fitted effective
spacing); the content only checks it afterwards. A covariate that was not used
in fitting is the only kind that can actually confirm a model.

*The termination rule, stated definitively:*
> The derivation is complete and perfect once a random noise begins to appear in the residual, and no amount of correction can remove it.

> We cannot estimate random noise, that is phisically not possible, this process ends there.

*And the loop back - the per-sample floor is not where the process ends:*
> However, we can use the 3d residual over this set of samples as a loop back and continue to derive until that residual is zero

*On the eq pulses - flat part only, and the transient differentiated with hsync:*
> Only look at the flat part of the eq pulses

> The important port of the eq pulses is the transient response that we have already measured from the hsync pulse. These two need to be differentialed together.

*On genlocking - the two locks differentiated together:*
> Now also differentiate the color burst lock and the hsync lock to assert genlocking. These need to be differentialed together

*What the genlock remainder IS:*
> That's it, the rf and video path needs to be genlocked

> The difference between those two is missmatch in the parameters of the VCR and then also the individual heads

So the remainder after differentiating the two locks is not noise to be
discarded: it decomposes into the VCR's own parameter mismatch (common to
both heads) and the individual heads' differences (alternating with head
parity). The RF path and the video path must be genlocked to each other, and
what separates them is exactly those two terms.

*Keeping it a curve, and naming the inputs to the extrapolation:*
> Let's keep it a curve fit, but I think that residual means something, let's allow the per vcr differential component be the inputs to this extrapolation. Frequency -> length of tape, Time tape taken -> user specified, and scene variation. Probaly can be measured per field.

**How the third input is handled, against the sync-only constraint.** Tape
position and recording date are external: neither is derived from the signal
being recovered, so both may drive the extrapolation freely. Scene variation is
not - it comes from the content. It therefore enters as a NUISANCE regressor:
its effect is measured and REMOVED, so that what remains is the per-VCR
component with the content's influence taken out. That uses the content to
CLEAN the sync-derived measurement rather than to drive it, which keeps the
constraint intact and makes the per-VCR estimate better rather than
circular.

*Total energy into the tape as the environment's differential component:*
> Oh total energy going into the tape might be good to know and be the differential component for the VCR's environment, i.e. tempurature

*The mechanical half of the VCR model:*
> Use the mechanical components of the VCR to estimated as residuals knowing their exact specficiations, how they interact and in what sequence they interact in inside the VCR. I think the RF profile and the mechanical profile determine the VCR's total model

*The chain's true shape - the VCR stage runs twice:*
> The RF (before it goes into the VHS VCR) is corrected using the known components of the vcr which are mechanical and electronic (one recording vcr and one playback vcr). Then the next step corrects the picture using video spec standards. These are differentiated using the electronic variables (tuning knobs for enmphasis, etc)

> So we need to run the VCR stage twice I think

The signal passes through TWO machines, so the VCR stage runs twice, and the
ORDER is forced by the chain rather than chosen: the playback machine acted
LAST on the signal, so it is inverted FIRST, and the recording machine is
inverted after, on a decode the playback correction has already cleaned. Only
then does the picture stage run, against the video specification, with the
electronic tuning variables - the emphasis knobs - as what differentiates it.

*Modelling the tape's own variations as parameters of the playback path:*
> I have another person who helped me distover this, and they are modeling the parameters of a VHS playback device, this contains all the possible variations in tape that can occur. Model these and apply them to the VCR path

(The Discord link could not be opened from here - it needs authentication.
The parameter list from that work should be matched against
`vhsdecode/tape_model.py` when it is available.)

The tape is the THIRD element: the two machines are modelled, but the medium
between them was not. It is written by one machine and read by the other, so
its parameters belong to neither and must be carried separately - and the
coating thickness, which had been sitting in the head model, is a property of
the tape rather than of any head.

*The other model's parameter list (https://blog.opengbh.net/vhs_playback.html):*

A VHS playback physics simulator. Its parameters, against what this arc had:

  ALREADY MODELLED
    tracking knob (%)             = tape_model `track_offset_m` (mistracking)
    head switch offset (%)        = the head-switch lane's stage
    azimuth error (deg) PER CHANNEL = head_model `azimuth_error_degrees`, and
                                    their making it per-channel independently
                                    confirms the per-head structure this arc
                                    reached from the data
    gain (dB) per channel         = the per-head DC/level component
    tape speed multiplier         = the format params' tape speed
    head switch pulse at 30 Hz    = transport_model's drum rate, 29.97 Hz

  NEW, AND ADDED
    protrusion (r, um) per channel  how far the head stands out of the drum -
                                    the PHYSICAL ORIGIN of the spacing that
                                    Wallace's law sees. The arc had been
                                    fitting an effective spacing with no
                                    mechanism behind it; this is the mechanism.
    height (z, um) per channel      the head's vertical position, which sets
                                    where on the track it reads
    parabolic bow error (um)        tape curvature. A head sweeps one field, so
                                    a bow shows as a PARABOLIC variation from
                                    the top of a field to the bottom - directly
                                    testable, and tested below.
    guide relax                     guide compliance and friction, acting on
                                    tension and so on the envelope
    supply / take-up offset         reel positions, setting the tape path
    X-value (um)                    head lateral fine positioning
    A/C head tilt, azimuth, height  the audio/control head - it reads the CTL
                                    track, which is what SETS tracking, so it
                                    sits upstream of the tracking parameter

*The limit, the bound, and the stopping rule:*
> Removing the small residual again on the whole process and mapping that change as a differential is the algorithm that I would run the limit over. this be explained by the actual limit of the whole chunk of rf data, the sample set is the limit's noise floor. differentiate between the expected 8 bit, 40mhz data's expected nosie floor and this measured noise floor to find something? the baseline is the ambient noise of the capture, which would require a separate measurement that I don't have, so this step would have to be bounded since it controls how many times I need to refine all the contains components.

> Ok, now I understand, I need to use the slope of this final residual to assert when it reaches a limit, within the margin of error I get from the rf capture chain

*The ORDER of the limits - the tape binds before the capture does:*
> I think before the limits of the RF capture need to be removed, we need to remove the limits of the magenetic properties of the tape, This is a purly a frequency and phase question, and likely less so on time

*The out-of-band data as the noise differential:*
> The luma and chroma are band limited, so the data outside this bandlimited area is noise, that can use as the differential

*Subtracting the capture's own noise out of the residual:*
> Lets remove that noise entirely from the residual. I think this part just gets subtracted out

*Two head switches, and what the tape cannot carry:*
> It is possible there is a record head switching location (no pulse) and a playback head swiching pulse

> DC is not recorded on the tape

Two switches, at two different places and of two different kinds. The RECORD
switch is fixed to the TAPE - it is wherever the recording machine's drum
happened to be - and it leaves no pulse, only a discontinuity in what was laid
down. The PLAYBACK switch is fixed to the PLAYBACK machine's drum phase and
does produce a pulse, in its own electronics. They coincide only if both
machines phase their drums against sync identically.

And DC is not recorded: a head responds to the rate of change of flux, so zero
frequency produces zero output. That is why the format is FM in the first
place. It also means the record switch CANNOT present as a DC step on the tape
- there is no such thing there - so looking for one is looking for the wrong
shape. (The video's own DC survives, because it is carried as a carrier
FREQUENCY and the demodulator returns it; so the low-frequency droop measured
from the vertical interval is an electronics effect, which is where it was
attributed.)

*The servo-locked switch position as a differential for the TBC:*
> I think we can use the peak detection to determine where the head switching servo lock switched. Servo lock constraint applies to the playback head, and I think this is another differential to add to the thc, there willbe some servo drift but not much.

The constraint belongs to the PLAYBACK head, which is what makes it useful: a
servo holds it, so its position is nearly fixed and its small drift is a clean
differential. It cannot be found by differencing picture lines - tried, and the
located points scattered over 16 microseconds because on real content the
line-to-line difference is the PICTURE. The head-switch lane's own `locate()`
is the validated finder, and it needs a live field, so it is called from the
residual-channel export by import.

*Folding the sync-pulse matrix pencil into the video path:*
> I think the sync pulse matric pencil that measures the sync pulse frequency, amplitude, and time measurement, needs to be folded into the video path. I believe we missed this part.

He is right that it was missed, and right about why it belongs: a matrix
pencil returns DAMPED SINUSOIDS, and each mode carries a frequency, a residue
amplitude and a decay rate - which is exactly this arc's three axes, in
parametric form, measured on the sync pulse. The existing
`sync_step_response.polarity_poles` returns (frequency, decay, |pole|,
is_ring): frequency and time are present, but the AMPLITUDE axis is not -
|pole| is the decay's magnitude, not how much of the residual the mode
accounts for. The residue is solved for here to complete the triple, and the
result is a PICTURE-stage component, because the sync pulse it is measured on
is demodulated video.

*Correcting the width VARIATION, never the absolute width:*
> I think I want to correct variation in the pulse width, it should be the same, but not the absolute width, since that will be miss-calibration in the sync generator. Having no variation in width that coorelates with the burst locked tbc, means that sync pulse's width variation is caused by the input signal only. However that still may need time base correcting, specitically if the input signal was not burst locked.

Measured, and the criterion answers cleanly: the per-line width variation does
NOT follow the burst-locked time base (six correlations across three tape
positions, |r| <= 0.041, none significant even before the effective-sample
correction). So the variation arrived with the input signal.

The ABSOLUTE width is left alone: it is the source's sync generator, and
"correcting" it would be imposing a specification on a machine that was simply
calibrated differently. What is corrected is the departure of each line from the
population's own median, which is the part that cannot be a fixed calibration.

*The transition slope as a fitted parameter:*
> I think it is sync transition slope now, I'm looking at the plot and the ideal slope vs. calibtrated slope is yet another parameter to fit here with the differential.

Measured against the specification's 0.140 us transition, and it splits by
polarity in a way that matters:

  fall  slope 0.363-0.373 us = 2.6-2.7x the ideal, and NO ringing after it
  rise  slope 0.271-0.283 us = 1.94-2.02x the ideal, then 0.33-0.40 us of ringing

The fall is the SLOWER edge yet the clean one; the rise is twice the ideal
slope and rings. Both hold to a few percent across the whole tape, so they are
parameters of the chain and not of the position.

MEASURE THE SLOPE AT THE FIRST 90% CROSSING, the settling at the LAST, and
never conflate them: taking the last crossing as the slope reads the rise as
0.66 us instead of 0.28, because it has swallowed the ringing. And the FALL
needs the previous line joined on - the output line BEGINS at the sync fall, so
the top of that edge lies in the line before, and searching within the line
finds no downward crossing at all.

*Ignoring the fall slope where the tip was clipped:*
> This makes sense because I think the fall was clipped at some point and is not reliable. If the clipped hsync was detected at any point (asymetrically shaped hsync pulse), ignore the fall slope and center the width aligment based on the rise's slope.

The rule is implemented. On THIS tape it does not fire, and the disagreement is
worth keeping rather than smoothing over: the established noise-based detector
(`tip_flatness`, the tip's own noise against the porch's) reads 1.17 to 1.39,
where a clipped-flat tip would read below 0.75 - the tip is NOISIER than the
porch, not flatter. So the 33% edge asymmetry needs another explanation, and
there is a physical one that costs nothing to assume: the fall lands on the TIP
carrier and the rise on the BLANKING carrier, so the demodulator's response
differs between them and some asymmetry is expected with no clipping at all.
Both criteria are therefore kept, and the fall is dropped only when one fires.

*Removing the identified RF interference:*
> Let's make sure we remove that interference component based on the RF residual for it we identified earlier

Nine lines sit INSIDE the VHS band and carry 2.24% of its power. Their
frequencies identify them: 5.9999 MHz at 29.6 dB over background is a 6.000 MHz
crystal (the 12.0001 MHz line found earlier is its second harmonic), and
2.0967 / 3.1451 / 4.1934 / 6.2903 MHz are exact harmonics (x2, x3, x4, x6) of
one ~1.0484 MHz fundamental. Both families are capture-side hardware.

They are removed COHERENTLY, not notched. A notch removes the signal that
shares the bin; a coherent subtraction removes only what is actually periodic
at that frequency and leaves the rest. Line-rate harmonics are excluded from
the search - those are the signal's own structure, as the earlier 78.7 kHz
finding showed.

*Fitting the non-linear parameters with the linear ones, and differentiating all of them:*
> Let's make sure that the non-linear and deemphasis parameters are fitted now in congruence with the current video based filter. I think there are non linearities described by the parameters of the deemphasis and any other non-linear filtering parameters. These may change over time so a full triple differential is needed for all the parameters.

The non-linear stage (`nonlinear_filter.sub_deemphasis_inner`) scales the
high-frequency part by 1 - amplitude^exponential_scale, so its parameters are
INTRINSICALLY amplitude-dependent - the triple differential is the natural
description of them rather than an extra imposed on them. Its parameters, in
the decoder's own names: exponential_scale, linear_scale_1, linear_scale_2,
logistic_mid, logistic_rate, static_factor, and the high-pass corner.

Fitted together with the linear shelf, the first question is not what the
values are but WHICH ARE IDENTIFIABLE. Two parameters whose sensitivities have
the same shape across amplitude and frequency cannot be separated by any
amount of data, and a fit will apportion them arbitrarily - which is what the
railed shelf fit was already showing.

*The width deficit as a de-emphasis miscalibration, and the ORDER of the stages:*
> Sync width may indicate an incorrect callibration of the deemphasis parameters. Let's use a the method to modle the difference between the parameters between the derives parameters. Maure sure this happens in the correct order of the design of the deephasis filter stage, since these are separate non-linear paths. We can model it against the spec, fit a frequency response measurement similar to howe we did the pulse equalization, then correct using the method.

The order, read from process.py rather than assumed:
  1. MAIN de-emphasis - the linear shelf, folded into FVideo in the frequency
     domain, so it acts FIRST.
  2. `nldeemp` - the high-frequency part taken, HARD CLIPPED to limits, and
     subtracted. Non-linear by clipping.
  3. `subdeemp` - the high-frequency part taken, scaled by a power of its own
     instantaneous amplitude, and subtracted. Non-linear by the power law.
Two SEPARATE non-linear paths, as Ethan says. Both take their high-frequency
part from the SAME pre-existing FFT and subtract sequentially, so they do not
cascade through one another's output - which matters for modelling them.

*Fitting the chain one stage at a time, in order:*
> I tihnk we can create the fit one at a time in order. Let's add that in. Possibly a something similar to what the color under was doing with the discrete frequency transform to model that response curve. I am open to options on this.

Implemented as ORDERED COORDINATE DESCENT down the chain. The reasoning is the
one this arc already established for the ordered traversal: the order is known
circuitry, so fitting down it sidesteps the degeneracies instead of fighting
them - at each step only ONE stage's parameters are free, and the stages before
it are already determined.

Three options were considered:
  (a) ORDERED, one stage at a time, swept to convergence - CHOSEN. Each stage
      is fitted against the response the earlier ones leave, which is exactly
      the structure of the chain. Degenerate parameters within a stage are
      fitted as their identifiable COMBINATION rather than separately, since
      `linear_scale_1` and `linear_scale_2` provably cannot be split.
  (b) A single joint fit of all ten. Rejected: two directions carry no
      information at all, so a joint fit apportions them by where it started.
  (c) Response-curve only, with no staging. Rejected on its own, but its
      MEASUREMENT is what (a) fits against - the discrete transform of the
      pulse, the same instrument the colour-under work used.

*Correcting the up-heterodyned colour carrier from the burst:*
> Use the color burst matching that we did with the luma to derive the noise in amplitude and frequency of the color carrier after up heterodyning. I think I need to use this residual to correct the up-heterodyned color carrier.

Measured on the up-heterodyned chroma, per line in the specified burst window:
amplitude noise 1.6-3.5% of the burst, phase noise about one degree,
frequency noise ~16 Hz rms. THE DECIDING NUMBER is the amplitude's lag-one
across lines, +0.57 to +0.80: the noise is SMOOTH, not white, so a per-line
estimate genuinely carries to the lines beside it. Were it white the burst
estimate would be noise and correcting with it would inject that noise into
the picture rather than remove anything.

*Chroma framing and the rotator's starting phase:*
> It's possible the color framing is incorrect, and we need to shift the color itself if it is 180 out of phase. Each alternating chroma line is different for each line. I think this determines the chroma phase polarity of the chroma phase order. Color under shifts the phase, and this reversal may not be locked. Let's roll in this measurement to the starting phase rotation of the chroma phase rotator.

Measured: the reversal IS locked. Removing the standard's half-cycle-per-line
alternation leaves a per-field constant that sits at either -32 or +148 degrees
- exactly 180 apart - and the sequence of those is `+ - - +` REPEATING. Period
4, identical at all three tape positions across 72 minutes, keyed to the field
index mod 4, and NOT head-locked. That is the NTSC four-field colour sequence
behaving exactly as the standard says.

Mod 180 the constant is stable to about 0.2 degrees with coherence 1.000, which
is what makes it usable as the rotator's starting phase.

WHAT THIS CANNOT SETTLE: whether the ABSOLUTE assignment is right - which field
of the four the decoder calls the first. The pattern being locked says the
sequence is intact; it cannot say the sequence starts where the standard says
it starts, and that is precisely the error that would leave the colour 180 out.
Settling it needs an external reference: a known colour in the picture, or the
standard's own definition tying colour framing to the field sequence.

*Using the burst-to-hsync lock to settle the framing, and feeding it back to the TBC:*
> I think we can use the color burst to hsync phase lock for this. The color carrier will have a relationship to the hsync phase. Use that standard to determine the phase alignment.

> Make sure these changes flow back into the time base correction aspect.

The standard does fix it: 227.5 cycles per line over 262.5 lines is 59718.75
cycles per field, a fractional 0.75, so the burst-to-sync phase advances 270
degrees per field and closes after four - which is WHY the sequence is four
fields long. Measured on the decoded output, though, it takes only TWO values
(-36 and +143.5 degrees, coherence 1.000 at twice the angle) where the standard
implies four. The first burst line is 9 on every field, so this is not a
folding artefact of the measurement.

The decoder's burst lock has already collapsed it: the lock nulls the
CONTINUOUS phase and only the discrete 180-degree alternation survives to the
output. So the absolute framing cannot be recovered downstream of the lock, and
the measurement has to move upstream of it - which is exactly why it belongs in
the TIME BASE stage rather than being read off the picture afterwards.

*The colour-under drift is largely TIME BASE - the components are coupled:*
> It is possible some of this drift represents time base error, make sure all components are in the differentiation

Correct, and by a wide margin. Measured on the RAW RF before any correction the
colour under sits +5752 +- 312 Hz above nominal - 0.91% high, the tape running
fast. After the time base it is +0.87 to +3.56 Hz. THE TIME BASE REMOVES 99.94%
OF IT.

So colour-under FREQUENCY and TIME BASE are not independent components: a
fractional timing error maps directly onto a carrier frequency error
(df/f = dt/t), and either one fitted alone will absorb the other. They must be
differentiated jointly, with the timing taken first because the time base is
measured from sync, which the colour under does not touch.

Two invalid comparisons made on the way, both recorded so they are not repeated:
`time_per_line` describes what the INPUT deviated by and the time base has
already removed it, so comparing it against a measurement made on the corrected
OUTPUT grid is meaningless (it produced 37 and 166 Hz of "unexplained" shift).
And the luma carrier cannot serve as a timing reference at all, because its
frequency encodes the picture level - its 1.3 to 27 kHz spread between fields
is content, not timing.

*Mutual luma/colour-under crosstalk removal, at the playback stage:*
> Make sure the color under residual is removed from the luma and vice versa. This should happen at the playback stage, since the error here is caused by inadiquate filtering on the front end.

Placed at RF_PLAYBACK, which is right: the separation is a front-end filter and
its inadequacy is a playback defect, not something the recording did.

MEASURING IT ON THE RF DOES NOT WORK, and the reason is worth recording. Every
band inside the capture's anti-alias passband reads about +14 dB over the
capture floor - INCLUDING 7-9 MHz, where VHS puts no signal at all. That is the
capture's in-band analogue noise, flat across 0-13 MHz, and the FM signal's mean
PSD per bin is comparable to it. The bands are not resolvable that way, by mean
or by median (the median is worse still: an FM carrier concentrates its energy
at its instantaneous frequency, so the median bin in the luma band is between
the excursions).

What IS measured, in the demodulated channels: the separation filter, recovered
from the chroma/luma RATIO where the shared picture content cancels - it peaks
at 634-653 kHz and three tape positions agree at r = +0.93 to +0.96. But the
crosstalk COUPLING itself is not established: fitting the colour under into the
luma at 629 kHz gives coherence 0.0145. So the removal is implemented WITH A
GUARD and declines at that coherence, because subtracting an unestablished
coupling injects the estimate own noise instead of removing anything.

*Cross colour happened at RECORD time, which changes what it is:*
> The cross color would have been added as the chroma was decoded and heterodyned down. There are chroma decoding errors that happened at record time.

This re-stages the component and explains the measurement. The recording VCR
takes a COMPOSITE input, separates chroma from luma with its own decoder, and
heterodynes the chroma down to the colour under. Cross colour is that decoder's
error, baked into the tape - a RECORDING-stage defect, not a playback one.

And it explains why correlating the two recorded channels found so little
(coherence 0.0005-0.041 overall): THE ENERGY MOVED. What sits in the chroma as
cross colour is exactly what is MISSING from the luma, because the record-time
separation took it out of one and put it into the other. A correlation between
two channels cannot recover something that was moved between them - it can only
see something present in both.

So the criterion changes. It is not "correlate the luma against the chroma" but
"model the recording machine's separation filter", because that filter decides
what moved. Its signature is the NOTCH it left in the luma at the subcarrier -
the same kind of feature the equalizing-pulse response found at the colour under
for the playback side.

---

## Correcting the head model (2026-09-03)

*The directive:*
> Yes let's correct the head model.

*Asked how far the per-head gain should go, Ethan chose:* **also route it
into the RF correction.**

### The correction to the previous round's claim

The claim that the fitted 0.45-0.49 um "effective spacing" was ABSORBING a
gain difference is wrong on the mechanism. Mean-centring is algebraically
identical to fitting a free constant and discarding it, so spacing never
absorbed the gain: `fit` computed it and threw it away. Verified, the
fitted spacing is identical to 1e-12 with and without a named `gain_db`.
And 0.45-0.49 um is the per-field ABSOLUTE spacing of one head, from
`perfield_headfit.py`, with `gap_m` railed at its lower bound - a different
quantity from the head difference. Two measurements were conflated.

What stands untouched: the head difference IS a flat gain. It is still
0.15-0.22 nepers at 629 kHz, where Wallace loss would have decayed to 0.01.

### Why the flat gain cannot be routed into the RF correction

Each checked against the code or measured:

1. The demodulator is blind to it. `unwrap_hilbert` takes the angle of the
   analytic signal and `angle(k*z) = angle(z)`. A +0.5 dB and a +3.0 dB
   flat gain each move the demodulated luma by ~3.3 Hz out of a 1 MHz
   deviation, and the figure does not grow with the gain: round-off.
2. The envelope never sees the equalizer. `env` is taken at
   `process.py:1565` and the equalizer applies at `:1595`, deliberately -
   the 17%-chroma rule. A gain in `ChannelEQ` reaches neither path.
3. The chroma consumer already removes it per head. The luma amplitude
   model is accumulated per head at `luma_amplitude.py:1665`,
   `line_key = bool(field.isFirstField)`, commented "the two heads
   genuinely differ". Correcting upstream would double-count.
4. Dropout detection already normalises it away: `doc.py:83` takes its
   threshold from `np.mean(env)` over the field, and one field is one head.

So the flat per-head gain has no live consumer in the signal path. Building
the routing as specified would ship something provably inert.

### What routing the head model to the RF side does mean

Once the flat part is excluded, what is left is the head difference's
SHAPE, and that is not inert - `channel_eq` applies one response to both
heads. The blocker is concrete and structural: `demodblock` receives only
FFT data, runs in `demodcache`'s worker pool, and fields are located AFTER
demodulation. Head identity cannot exist at the equalizer's site by
construction. Per-head RF equalization therefore needs a two-pass decode:
demodulate to find the fields and their heads, then re-equalize per head
and re-demodulate. That is the same cycling the offline runner already
does, and it is the honest cost of the feature.

### The sync-only measurement, and what it found

The gain measured before came from the envelope binned by carrier across
the whole line, which reads active picture. The constraint-compliant
witness is the SYNC TIP: one fixed carrier, constant envelope by
construction. The window comes from the spec-derived measurement plan, and
the carrier found inside it is 3475 kHz with 2 kHz spread across the whole
sweep, which is the sync tip - so the window is where it is meant to be.

  - the per-head gain is real: -1.16 dB pooled over 13 positions, 191 sigma,
    28-88 sigma at each position on its own;
  - it agrees with the independent carrier-binned measurement (sync tip
    -1.36 / -0.88 / -0.0005 dB against FM-band 1.52 / 1.13 / 0.08 dB at
    the same three positions), which is two instruments agreeing where only
    one of them is allowed to drive a correction;
  - IT IS NOT CONSTANT. It decays monotonically from -2.40 dB at the start
    of the tape to -0.0005 dB at 72 minutes. That contradicts "the fixed
    per head video loss is the final constant that is static given the
    VCR" - on this tape the head difference is a function of where you are
    on it.
  - the mechanism is NOT resolved. The overall level rises 4.8 dB over the
    same span, and the two explanations - tape position and contact quality
    - each survive the other's partial correlation at nearly equal strength
    (-0.878 and -0.872) on 13 points. Reported as an effect, not a cause.

### The trap this round caught

`is_first_field` is a field parity, not a head identity, and which parity
lands on which head depends on where a decode started. Across a sweep of
separate decodes it flipped at exactly one position, reversing that
position's sign and dropping the trend from t 9.94 to t 2.20. The
colour-under's +/-90 degrees per line, opposite by head, is the real head
signature and reads -90.0/+90.0 to within 0.3 degrees everywhere. Booked
as standing rule 38; rules 39 and 40 record the carrier law's two-member
null space and the mean-centring identity.

---

## Does the collapsed model work better? (2026-09-03)

*Ethan:*
> Let's double check that this collapsed model works better: template from
> averaged sync pulses, matched filter against each line, edge position
> series, plot against tape position.

*and:*
> And lets closely analyse using DCT for this

### The answer: yes, and the collapse holds

Template built per head on half a position's fields, scored on the other
half, 13 positions of the home tape, 876 lines per case.

**The matched filter beats the crossing in 26 of 26 position-head cases.**
Edge-position noise, as the residual about the line-to-line smooth:

  naive 50% crossing    0.1693 samples
  least-squares fitted  0.1243 samples
  matched filter        0.0165 samples

**And it is an estimator, not a smoother.** Residual-about-the-smooth can
be lowered by anything sluggish, so the claim is settled on a referee that
smoothing cannot game: the pulse has two edges, so each line's timing is
measured from the FALL half and the RISE half separately - disjoint
samples, no shared information - and their disagreement is pure estimator
noise. The matched filter wins that 26 of 26 as well, at 0.0370 samples
against the crossing's 0.4104. On the well-behaved positions the honest
factor is **5.5 to 6x**; the median ratio is larger only because the
fitted crossing fails outright at four positions (up to 5.7 samples).

### The DCT: the model's rank, and what one template cannot carry

The DCT is the right basis here because the pulse is a finite segment whose
endpoints differ - the DFT's periodic extension would spend coefficients on
a boundary discontinuity that is not in the signal - and DCT-II is close to
the Karhunen-Loeve transform for a segment this smooth, so it packs the
pulse into the fewest coefficients of any fixed basis.

**The template needs 34 of 120 coefficients to hold 99.9% of its energy**
(spread 1.3 across all 26 cases). That is the collapsed model's RANK: the
pulse has about 34 degrees of freedom, and that is what one H(f) has to
carry.

**After alignment, the residual is structureless.** Each line is aligned to
the template before its DCT is taken - the DCT has NO shift theorem, so an
unaligned line reads as a change of shape - and then the question is not
whether the residual has a spectrum but whether it is CONSISTENT. Asked per
case with the error bar corrected for the effective sample size, **no
coefficient stands at 4 sigma in all 26 cases**; the most frequent appears
in 3 of 26. So one template carries the pulse, and the collapse holds.

### Two of my own tests that failed and are withdrawn

- Comparing low-k coefficient power against a high-k median flagged 39 of
  40 coefficients as "structured". That test flags any lowpass signal, and
  the residual of a lowpass channel is lowpass. Replaced by the
  consistency test above.
- Pooling all 22776 lines found k=23 significant at 4 sigma. Asked per
  case it appears in 3 of 26. Pooling across cases manufactured it.
- The burst was to be the independent referee and turned out inert: both
  estimators correlate about zero with it (-0.059 and -0.077) while
  swinging between -0.62 and +0.42 position to position. That indicts the
  reference, not the estimators, and the script's printed verdict that the
  matched filter "is a smoother" fired on a rule comparing two numbers both
  consistent with zero. Withdrawn; the split-half referee replaced it.

### The porch anchor window, and a negative result (2026-09-03)

The two hard-coded windows in `field.py` - NTSC (74, 124), PAL (96, 160) -
are replaced by a derivation from the format's own timing, guarded by the
same settling time the ringing measurement plan uses
(`TRANSITION_SETTLE_BANDWIDTHS` of the luma low-pass, 0.455 us on NTSC).
That yields NTSC (112, 128) and PAL (140, 170).

**The unambiguous half of the fix is the SYNC TIP, not the porch.** The tip
window was `(4, ire0_backporch[0] - 4)` = (4, 70), and the sync rise is at
output sample 68 - so the tip level was measured through the rising edge.
The two windows were only ever coupled by the old layout; they are now
independent and each guarded against its own transitions.

**The porch half does NOT reproduce the improvement it was chosen for.**
The luma lane measured, on four captures from one deck, that the
burst-end-to-active-start window moves with content at +0.010 to +0.052
IRE/IRE while windows stopping a settling guard short are 5 to 50 times
cleaner. Repeating that regression on the home tape - 10512 lines across
three positions, effective sample size 976, each window's per-line median
regressed on the preceding line's active level:

  OLD 78-120        -0.0212 IRE/IRE   9.4 sigma   1.162 IRE swing
  NEW 112-128       -0.0221          11.0         1.212
  lddecode 112-135  -0.0227           9.3         1.242
  late only 120-130 -0.0208          12.4         1.138

All four are the same to within the spread, and the new window is
marginally WORSE than the old one. So the ranking is a property of that
deck and does not carry to this tape, exactly as its author scoped it.

Two things follow. The window change stands on the sync-tip fix, on
agreeing with lddecode's own convention, and on being spec-derived instead
of two hand-set tuples - not on content robustness, which it does not buy
here. And the porch on this tape is content-dependent by about 1.2 IRE
NO MATTER WHERE IT IS MEASURED, which is a finding in its own right: it
points at the settling tail itself rather than at the choice of window,
and is the thing the low-frequency correction has to fix.

Measured effect of the change on a decode at pos_5000 with the same flags:
every level shifted DOWN by a constant 0.570 IRE - the late porch, the mid
porch, the early porch and the sync tip all by 0.570 - so the anchor moved
and nothing was distorted, which is what an anchor change should do.

### The CTI width, measured (2026-09-03)

*Ethan:*
> I think the chroma transient improvement needs to consider the frequecy
> response we have measured from the color under. I think we can refine the
> response by looking at the residual chroma in the luma, and better shape
> the transient improvement. Also I think the CTI needs to happen after all
> of these correction steps.

**The ordering** is done: CTI now runs last, after every stage that reads
the chroma. It was running inside `process_chroma`, so `luma_beat` scaled
its correction by a cosmetically sharpened chroma. The chroma lane measured
what that cost on test patterns and it is small there - 75.5% -> 75.3% of
the beat removed on bars, 86.3% -> 86.4% on chromanoise - because the
saturation is smoothed over the shortest interval the colour-under band can
resolve, which averages most of CTI out. Right on principle regardless, and
their own caveat stands: content with many sharp saturated transitions is
not the test-pattern case.

**The shaping needed a different instrument than the one proposed.** Two
candidates were examined and both are the wrong quantity:

- `luma_beat.path_shaping()` is the RF path's sideband imbalance about the
  luma carrier, O(f)/O(f_cu), on an axis of RF OFFSET. Not a chroma
  response at all.
- `_measure_transfer_response` in `chroma.py` is a Wiener weight in
  frequency, on an axis of baseband MODULATION RATE, describing how much of
  the luma envelope's noise is shared with the colour-under as the loss
  gets faster. Its level is discarded by design.

Neither predicts a rise time. Worse, on the home tape the transfer **fails
its own null control**: rotating the segment index destroys the coupling
while leaving Sxx and Syy bit-identical, and the null reproduces the stored
shape band for band (51.6/67.1 samples against the real 57.5/58.7). Peak
coherence is 8e-4 against a null coherence of 1-9e-4. On this material the
roll-off is not resolved at all. That is a finding beyond CTI: the
LEVEL-setting use of the same shape in `apply_chroma_envelope_gain` rests
on the same unresolved measurement here. It does not refute the roll-off
measured earlier on the test patterns; it says this tape cannot see it.

**The instrument that works is the colour burst.** It is a gated
subcarrier at a fixed place on every line, so its envelope IS the chroma
path's step response - no edge detection, no dependence on content.
Measured pre-CTI: 16.41 samples on head A, 16.47 on head B, and -3 dB at
298 kHz, -6 dB at 425 kHz on both heads to 0.2%. Of that, the decoder's own
`FVideoBurst` band-pass accounts for 11.16 samples and everything upstream
for 12.0, so deriving the width from the decoder's filters alone would
understate it by 25%.

  CTI input 10-90                     T = 16.77 samples
  1%-99% span of a smooth edge        1.815 x T          (exact for Gaussian)
  radius must straddle it             R = 0.907 x T = 15.2
  quantised to a whole cycle          R = 16 = 4 cycles
  the old fixed default               R = 8  = 2 cycles

Swept on real content through the decoder's own operator, the fast tail of
the transition distribution has an interior minimum at exactly 16, and the
operator never overshoots at any radius - so the width was short by a
factor of two. Past 16 the median content transition collapses (-20% at 16,
-34% at 20), which is posterisation of detail the tape really carries, and
puts the ceiling where the derivation puts it.

**Shipped:** `--cti_width auto` (the new default) measures the burst rise
per field and sizes the sweep to it; a number still overrides. The noise
gate is decoupled from the width - it was `noise * sqrt(cti_width)`, so a
width change silently moved the gate too and was never a pure width change.

**Verified on a decode**, same position, same everything but the width:
the decoded chroma burst goes 9.23 -> 5.98 samples, against 6.12 predicted
offline for radius 16. Two independent instruments agreeing to 2.3%.

**One bug worth recording** because the fixture hid it: `uphet` is exactly
`outlinecount * outlinelen` long and CTI starts a line into it, so the rows
available are what remain AFTER the offset. Taking `outlinecount` rows from
there overruns by exactly the offset, and the measurement declined on every
real field - while a planted-truth fixture built one line too long passed
happily. The decode is what caught it; the unit test now sizes the buffer
exactly as the decoder does.

#### Correction: the transfer instrument is sound; home is the material

The chroma lane replayed the null control on their own captures and it
passes decisively, so the finding above must be scoped to the MATERIAL and
not read as an indictment of the instrument:

  chromanoise   real coherence max 0.307 against a null of 0.021   14x
  75bars SP                        0.133                  0.029     4.6x
  home.flac                        0.0008                 0.0009    1x

Home's peak coherence is a factor of 130 below chromanoise's. That is not
a weaker measurement of the same thing, it is a different regime: home does
not drive the mechanism. Chromanoise carries broadband chroma modulation,
bars carries saturated flats with few transitions and scores lower, and
ordinary content carries less still.

This was also already known in part. `_measure_transfer_response` carries a
comment recording that "75bars EP and home.flac never once cleared three
sigma in three bands", and the per-band significance gate was REMOVED
deliberately, because dropping the shape entirely cost more than keeping an
imperfect one and because selecting bands on the size of their own estimate
biases the level upward.

What the null adds is a different question from the one that gate asked.
The inverse-variance argument for removing it - a band that resolves
nothing contributes almost nothing - holds while SOME band resolves. When
none does, a weighted combination of unresolved bands is still a
confident-looking shape, and `_project_level` reads the applied LEVEL off
it. On home that is the case.

Landed: `field.chroma_envelope_coherence`, the measurement's own peak
coherence, recorded on every field. It selects no bands, gates nothing and
changes no result - it makes the difference between the two regimes
visible, so whether to act on it is a decision on evidence rather than an
invisible default. The obvious candidate action, a per-head resolvability
test that separates "this head has no shape but the other does" from
"neither has one" - which the current cross-head borrow cannot tell apart -
belongs to the chroma envelope arc's owner, with both lanes' numbers.

Verified: the CTI reorder and the measured width pass the chroma lane's own
matrix, with the Y-only control reading 0.0% at both gain settings (no
injection into material with no chroma) and beat removal unchanged at
75.3% on bars and 86.4% on chromanoise against 75.5/86.3 before.

#### The boundary, on six captures: it is pictures, not one tape

The chroma lane nulled six captures with the same probe. Ranked by how far
the real pairing stands clear of the matched null:

  chromanoise      real max 0.3070   null 0.0212   14.5x
  75bars SP                 0.1329        0.0287    4.6x
  ntc7 composite            0.0549        0.0235    2.3x
  pulse+bar                 0.0451        0.0275    1.6x
  multiburst                0.0388        0.0216    1.8x
  ramp                      0.0336        0.0254    1.3x
  home.flac                 0.0008        0.0009    1.0x

The null floor holds at 0.021-0.029 across all six, which is the check
that the control itself behaves.

THIS OVERTURNS THE SCOPING AGREED ABOVE. "The material, not the
instrument" stands - the instrument resolves decisively when the coupling
is driven. But the material is not a peculiarity of one home tape: the
boundary falls between ntc7 composite and 75bars, and THREE OF SIX
captures sit at 1.3-1.8x, effectively at the null. The ones that resolve
are chroma test signals - chromanoise is broadband chroma modulation,
75bars is saturated flats - and the ones that fail are the ones that look
like ordinary pictures. Ordinary content is therefore more likely to sit
below the boundary than above it, and `_project_level` reads the applied
LEVEL off the shape in exactly that case.

That makes the per-field `chroma_envelope_coherence` channel materially
more useful than it appeared when it landed: the two regimes are not an
edge case, they are the common case and the test case.

Neither lane is gating anything on it. That is the chroma envelope arc's
design and it would change behaviour on the primary tape, so it is a
decision to be taken on this evidence rather than taken silently. What the
evidence now supports, which two points at opposite extremes did not, is
that the decision matters for most real material.

---

## The frequency derivation, and the burst as its first subject (2026-09-03)

Ethan's algorithm is now written out once, in `docs/FREQUENCY_DERIVATION.md`,
because he said it was getting lost and is the most important part of the
process. Eight steps: fold (aligned first), difference against the
reference, identify the frequency support by DCT, solve ALL amplitudes
JOINTLY, derive phase from gain, correct, iterate, and stop when the
held-out residual is structureless and at the floor.

The governing sentence, his: *"Since each correction will affect all the
other frequency responses, it is critical that this residual is measured
and corrected as a whole."* That is what rules out the greedy
one-at-a-time peel the picture stage does today.

### Applied to the chroma transient improvement

Ethan: *"I need to shape the slope of the improvement based on the ENTIRE
burst envelope ... Expected is the spec derived shape of the burst, and
actual is the measured shape of the burst."*

Measured on the home tape with the sharpener disabled (`--cti_mix 0`), 16
fields, the burst envelope folded over every line and compared against an
RS-170A gated subcarrier:

**Alignment is load-bearing and I nearly skipped it.** The measured burst
is delayed relative to the spec window, and a delay is not a roll-off.
Aligning first dropped the residual from 0.2328 to 0.1504 rms - a third of
what looked like response was position.

**Ratio versus difference, demonstrated rather than argued.** At 430 kHz
the ratio reads -12.55 dB and the difference reads -1.76 against an
expected magnitude of 2.32. The ratio is large only because its denominator
is small, which is exactly the spectral division this project has already
refuted twice. Reported as a difference, and only where the expected has
energy to compare: 9 bins of 50 carry evidence, and the other 41 are the
ones a ratio would have reported as a deep roll-off on no evidence at all.

The response that survives that guard, as a difference:

  143 kHz  +0.75    573 kHz  -5.13    1002 kHz  -3.18
  286 kHz  -1.76    716 kHz  -2.97    1289 kHz  -1.27
  430 kHz  -1.76    859 kHz  -1.67    1432 kHz  -1.62

So there IS a shaped roll-off, strongest around 573 kHz, and it is not the
single number the width currently comes from.

**One correction this measurement still needs before it shapes anything.**
The expected must be the spec burst AS SEEN THROUGH THE DECODER'S OWN
CHROMA BANDPASS, not the raw spec burst. `FVideoBurst` contributes 11.16
samples of the measured 16.4 sample rise on its own, so most of what is
measured above is our own filter rather than the tape's. Inverting it would
undo the separation filter that exists to keep the chroma apart - the same
error `rf_path_response` already guards against on the luma side, where the
comment records that leaving the decoder's notch out of the reference
"books its dip into the measured tape response - from where the equalizer
would faithfully re-boost the interference the notch exists to remove."

### The luma_noise panel's two labels were inverted, and it mattered

`measure_amplitude_deviation` passes `dense=None` deliberately, so the
deviation divides by the LINE'S response only. Algebraically that makes
`after_line = after * exp(departure)`: the trace labelled "after the fitted
roll-off" is the line-corrected deviation with the accumulated table's
departure put BACK on, and the trace labelled "after the full model" is the
fitted roll-off's own output.

The labels were therefore the wrong way round, and read off them the
smoother trace looked like the full model beating the line - which argues
for dropping the line. **The opposite is true.** The roll-off is the only
thing being applied, the smoother trace is its result, and the code's own
comment already records the measurement behind that choice: withholding the
accumulated table is worth -0.016/-0.037/-0.041 per cent against keeping
it. Directive 1's answer is that the fitted roll-off is not merely still
needed, it is the whole of what is in force.

Labels corrected, with the derivation written at the site.

### Both heads on the response panel

The panel drew only the field's own head, while the wow panel three lines
below already drew both with a settled convention - fixed colours, the
other head faint, this field's solid, the same colours as
`plot_luma_averaging` "so a head keeps its identity across the two plots".
The response panel now follows that convention exactly, with the field's
head named on it, fed by the existing per-head accessor
`luma_amplitude.measured_response(rf, head)`.

### The sharpener's slope, shaped by the burst (2026-09-03, shipped)

Ethan: *"I need to shape the slope of the improvement based on the entire
burst envelope ... Expected is the spec derived shape of the burst, and
actual is the measured shape of the burst."*

Implemented as the derivation applied to the burst. The operator had ONE
radius and a fixed geometric decay of 0.25 across four passes, so its
emphasis had a shape nothing had measured. Each pass now has its own
radius - a small filter bank, distinct radii because two passes at the same
radius are one pass with twice the weight and make the solve's basis
rank-deficient - and the weights are solved JOINTLY against the measured
roll-off, which is the derivation's governing rule.

**The decoder's own band-pass is divided out, not corrected for.**
`FVideoBurst` is a fourth-order band-pass from 60 kHz to 1.2 MHz about a
629 kHz carrier, so it passes envelope components only to about 571 kHz -
and the roll-off measured on the burst is strongest at 573 kHz. Nearly all
of it is ours. Inverting it would undo the separation filter that exists to
keep the chroma apart, which is the error `rf_path_response` already guards
against on the luma side.

The measured effect of getting that right, on a decode at pos_5000:

  CTI off                        burst 10-90 = 15.10 samples
  one radius + fixed decay        5.98
  shaped bank, solved weights    10.57

The decoder's own filter alone limits the burst to about 11.16 samples. The
shaped sharpener lands at 10.57 - it sharpens up to our own filter's floor
and stops. **The previous build's 5.98 was over-sharpening**: it was
derived from the raw 16.8 sample rise, which includes the decoder's 11.16,
so it was undoing the decoder's own bandpass.

**A bug the planted test caught.** The measurement window reaches back into
the breezeway so the rise can be seen, which puts the middle third on the
rising EDGE - and the flat-top normalisation was taking its scale from a
fraction of the array rather than from the specification. A planted burst
read 12.8 at its own flat top. The flat span now comes from the spec window
with the gate's transition guarded off each end.

### The picture stage corrected rows it never measured (2026-09-03, fixed)

Ethan: *"I believe the ringing correction is correctly being applied in the
RF stages, but it is not being applied correctly in the picture stage as
well."*

Confirmed and fixed. `measure_field_lines` calibrates on
`[first_measurable_line, last_measurable_line)`, and `build_geometry`
excludes the field's FINAL row deliberately - it spans the field boundary
and carries the next field's half-line equalizing pulse. That row entered
no accumulator, template, stratum, variance or fit weight. But
`process_field` corrected `field_lines_ire.reshape(-1)`: every sample of
the buffer, including that row, the vertical interval, and everything above
the first measurable line. The calibration population and the application
population were different sets.

The filter is still RUN over the whole buffer - it is causal with memory
across line boundaries, and restricting its input would change its state at
every row it did reach - and only the write-back is restricted. Applying to
sync, blanking AND picture alike within the measured rows is deliberate and
unchanged: a correction confined to blanking hides its picture behaviour
from every sync gauge.

Measured, ringing on against off, mean absolute change per row:

  the measured rows (20-261)   0.0143 IRE
  the FINAL row (262)          0.0008 IRE
  the vertical interval (0-5)  0.0004-0.0008 IRE

An eighteen-fold reduction on the row the model was never fitted to.

### Stages and components, on and off by name (2026-09-03, shipped)

Ethan: *"Let's change up the flags so I can enable and disable each stage
that is being corrected, and then optionally within each stage enable and
disable certain components"*, and *"Just do an on-off and remove the
amount. The amount will be derived using the residual removal process."*

`--stages` takes a comma-separated list of `+name` / `-name`, where a name
is a stage or `stage.component`:

    --stages +head_switch,-cti,-ringing.ghost

Names are validated against `vhsdecode/pipeline/stages.toml` at parse time,
so a typo fails at startup with the valid list rather than silently
disabling a correction and producing a decode nobody can account for.
Thirteen components are declared across five stages, each with a note
saying what it is.

**Mapped onto the existing gates, not added beside them.** A stage turned
off has its OWN option set to the value its own predicate reads as off, so
every existing `if` in the decoder keeps working and there is still exactly
one gate per stage. A second, parallel gating mechanism would only be a new
way for the two to disagree. Verified: `--stages -cti` gives a burst rise of
15.07 samples against `--cti_mix 0`'s 15.10 - the same decode.

Two honest limits, reported rather than papered over: a stage that always
runs says so when named, and a stage gated on a measurement FILE cannot be
switched on by naming it, because naming it cannot conjure the file.

One argv detail worth recording: a selection may legitimately begin with
`-`, which argparse reads as another option. The same problem was solved
for `--ire0_adjust` by rewriting argv before parsing, and `--stages`
follows that precedent - where the following token parses as a selection it
is attached with `=`, and where it does not, a genuine missing value still
reports as missing.

### The joint solve, the stopping rule, and why both are inert here

**The amplitudes are now solved as a whole.** The peel still IDENTIFIES the
components one at a time - which is right, since a component only becomes
visible once the larger ones are out of the way - but each peel also FITTED
its amplitudes against a residual the previous peels had already deflated.
Every amplitude therefore carried the earlier ones' errors. Once the
support is known, all amplitudes are re-solved together against the
undeflated target. The support comes from the peel; the amplitudes do not.

Planted truth, in `tests/unit/test_joint_amplitudes.py`: two resonances a
quarter of a resolution cell apart, column coherence 0.79. The joint solve
recovers both exactly; the greedy one mis-splits them, and its answer
depends on the order the components were found in - a property of the
method rather than of the signal.

**The peel now stops at the floor**, not at a slot budget. It used to end
only when the budget was spent or no further candidate certified, neither
of which is a statement about the residual.

**Both are inert on the home tape, and the reason is worth recording**
rather than leaving the null ambiguous. Accumulating the model exactly as
the decoder does:

  head A   2 sections fitted - 1 ring, 1 smear
  head B   1 section  fitted - 0 ring, 1 smear

With fewer than two ring components there is nothing for a joint solve to
re-split, so the decode is byte-identical (0.00000 mean absolute IRE). The
mechanism is proved on planted truth; this material does not exercise it.

### The two blocks, and the measurement that justifies them

Ethan: *"We then apply the VCR deemphasis controls AND the picture
correction in one block after demodulation. The deemphasis controls will
influence the picture correction, so these steps need to feed into each
other."*

Declared in `stages.toml` as `[[block]]` - a block being the unit of
SOLVING, which is not the scope a stage executes in nor the group it
composes with. `rf_identification` holds the pre-demodulator path;
`picture_solve` holds de-emphasis, the non-linear paths, the ringing model
and the transient stage.

**And the claim is measured, not assumed.** Fitting the picture
correction's model on two decodes that differ ONLY in the de-emphasis:

                        head A                          head B
  default      2 sections: ring 0.9765+0.1246j,   1 section: smear 0.9585
               smear 0.9700
  --nld on     1 section:  smear 0.9714           0 sections

Turning on the non-linear de-emphasis removes the ring component from head
A entirely, removes head B's model entirely, and shifts head A's smear pole
from 0.9700 to 0.9714. The de-emphasis setting decides what the picture
correction has left to fit, so the two cannot be solved independently -
which is exactly Ethan's reasoning, now with a number against it.

That is the case for the block. What remains is to make them actually feed
each other, which is a restructure of two stages that currently live in
different scopes on different threads.

### The per-head response is NOT content (2026-09-03, corrected)

A confound was raised against the per-head luma response - that odd and
even fields might carry different content and so make the per-head
difference partly a content difference - and I accepted it and labelled the
plot accordingly. Ethan: *"No, you are incorrect, the content is irrelevant
to the head identification."* He is right and the label is withdrawn.

The head is identified by FIELD PARITY, which is a property of the
recording and not of what is on it - no content enters the identification
at all. And for the RESPONSE, the argument runs the other way from how it
was recorded: both heads accumulate over the same tape and the same
content, so a content-driven component appears in BOTH accumulations and
cancels in their difference. On static material, where the alternating
fields carry identical picture, it cancels exactly.

The independent measurement agrees. On the sync tip - one fixed carrier, a
constant envelope, and no active video anywhere in the window - the
per-head gain reads +0.98 dB pooled at 161 sigma, 28 to 88 sigma at each of
thirteen positions. Nothing in the picture reaches that window, so the
difference it measures cannot be content, and it is the same per-head
difference the response describes.

Standing rule 43 keeps the general statement, which is sound - agreement
between two banks bounds noise and not confounding - and records this
instance as the wrong application of it.
### The derivation converges on the sync pulse (2026-09-03)

`docs/FREQUENCY_DERIVATION.md` run on the picture correction's own
measurement, offline, on a real decode. This is the result the plan said
would decide whether it ships.

**It reaches the noise floor, on held-out data.** Fitting on the even
samples of the accumulated sync interval and judging on the odd:

  head A, floor 0.9218 IRE          head B, floor 1.0001 IRE
  pass  fitted   HELD-OUT           pass  fitted   HELD-OUT
   1    2.6472   2.7455              1    2.5768   2.6711
   2    1.8403   1.8236              2    1.7869   1.7711
   3    1.4550   1.4681              3    1.4187   1.4315
   4    1.2682   1.2848              4    1.2253   1.2447
   5    1.0812   1.1150              5    1.0023   1.0472
   6    0.8631   0.8805  AT FLOOR    6    0.8468   0.8887  AT FLOOR

From 11.17 IRE to 0.88 in six passes, six components each. The held-out
column never parts from the fitted one - which is the whole test. A model
memorising its own noise shows the fitted residual still falling while the
held-out flattens or rises; this falls in lockstep and stops at the floor.

**Two errors of my own on the way, both caught by the instrument.**

The first pass subtracted a zero-mean reconstruction from a residual whose
mean was 3.37 IRE, so the level was never removed and the residual
plateaued at exactly that mean. The second was worse and more interesting:
identifying the support with a DCT and then subtracting the DCT
reconstruction of the residual is not a correction at all. On an
orthonormal basis that merely DISCARDS the coefficients left out - it
re-expresses the residual rather than modelling anything, and it cannot
converge to a floor because there is no model to be right or wrong about.

The DCT NAMES THE FREQUENCIES. The model is built from them: a damped
resonance at each, in quadrature so its phase is free, with its decay
scanned and its amplitude solved jointly with all the others. That is the
difference between a change of basis and a channel model, and it is the
difference between a residual that plateaus and one that reaches the floor.

**What it means for the stage.** The plan's condition was "instrument
first, ship if it converges". It converges. The picture stage today
identifies by matrix pencil and stops on a slot budget; this identifies by
DCT and stops on the residual, and reaches a floor the stage does not
currently reach.
