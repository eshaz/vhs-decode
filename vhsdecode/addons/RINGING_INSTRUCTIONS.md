## Artifacts to correct
These are the visual artifacts that are being targeted for correction.

### Artifact description
* Common attributes
  * All artifacts here are causal and caused by changes in the luma signal

* Ghosting (likely happens before ringing and smearing)
  * Definition
    * Similar visual effect as ringing; however, there is no oscillation
    * This is a mirror of the signal after it originally occurs
    * This may be an attribute of the source signal, therefore ringing and smearing may follow a ghost
  * Possible Causes:
    * Multi-path television reception
    * Impedance missmatches causing signal reflections

* Ringing
  * Definition:
    * One or more sets of resonant oscillations that occur after a transient
      * Multiple ringing components may exist, each with their own unique characteristics.
    * The initial transient will have the greatest amplitude
    * The following oscillations (ringing) will follow the initial transient and will always decrease in amplitude
    * The following oscillations will eventually approach zero
    * The amplitude of the oscillations is determined by the amplitude of the initial transient
    * The frequency of these oscillations is likely fixed
  * Visual Appearance:
    * Looks like an echo sounds and is more visible after larger transitions
  * Possible Causes:
    * Resonating low filter
    * Impedance missmatch
  * Quantity:
    * There can be zero or more separate ringing artifacts
  * Length:
    * Ringing should be a relatively high frequency
    * Ringing components should only ever be as long as the sync tip
* Smearing
  * Definition:
    * Decay that immediately follows a transient
      * Multiple smear components may exist, each with their own unique characteristics.
    * The magnitude of this artifact does not oscillate
    * The energy of the causing change in luma is spread out over time across the signal
  * Visual Appearance:
    * A smearing effect that trails the luma signal
    * This is kind of like reverberation in the audio domain
  * Possible Causes:
    * Energy dispirsion after low pass filtering
  * Quantity:
    * There can be zero or one smearing artifact
  * Length:
    * Smear may start with an over or undershoot, and then decay to 0
    * Smear should only be observable within the length of the sync tip. In extreme cases, the smear might extend slightly beyond the length of the sync tip into the rest of the sync region.

### Artifact sequencing 
* The artifacts are applied additively to the signal as it moved from the original source to the RF capture device
* The television signal path, VCR signal path, and RF capture path are all possible places where artifacts can be introduced

1. Original clean signal
2. Ghosting
   * Likely happens before the signal enters the recording VCR
3. Smearing and Ringing
   a. Introduced at VCR record time
   b. Introduced at VCR playback time
   c. Introduced by RF capture chain
      * To a lesser extant than the the VCR artifacts


## Measuring
The entire horizontal sync area will be used while measuring for the artifacts.

### Measurement Points
The horizontal sync area is used as a known reference for measuring.
The horizontal sync area is comprised of the below list of measurement points listed in sequence.
Only horizontal sync lines will be used for measurement, i.e. all lines after the end of the vsync (supplied as an system parameter)

1. Active Area
   * Definition:
     * This is the active video. It is the picture content.
   * Location
     * Between each horizontal sync region
   * Measurement Value
     * The active area contains video data of different and unknown amplitudes.
     * The area just before the Active Falling Edge determines the starting amplitude going into front porch.
2. Active Falling Edge
   * Definition:
     * The falling edge where the active area ends and the front porch starts
     * There may be exceptions where this is not a falling edge, and is just flat.
     * It will almost never be a rising edge, but it's not impossible.
   * Location:
     * The location of the end of the active area is variable, but there will always be some space for the front porch
   * Measurement Value:
     * Determines initial amplitude going into the sync region
     * Possible to determine amplitude dependent ringing and smear
3. Front Porch
   * Definition:
     * Flat region at 0 IRE that is at the end of the video line
   * Location:
     * Between active area and sync tip
     * The start of the front porch is variable depending on the end of the active area
     * The end of the front porch is always at the Sync Falling Edge
   * Length:
     * Supplied as a system parameter in samples in decimal form.
   * Measurement Value:
     * Known flat region to use as baseline for measuring artifacts
4. Sync Falling Edge
   * Definition:
     * Transition from the front porch to the sync tip
     * 0 IRE to -40 IRE
   * Length:
     * Supplied as a system parameter in samples in decimal form.
     * Transition length is known the spec
   * Location:
     * The middle of this falling edge is the reference for the beginning of the video line
   * Measurement Value:
     * Determines the deviation between the expected transition slope and actual transition slope
     * Falling reference pulse that is used to determine the magnitude of the artifacts present in the sync tip
5. Sync Tip
   * Definition:
     * Flat area at -40 IRE between the falling and rising sync pulses
   * Length:
     * Supplied as a system parameter in samples in decimal form.
     * Transition length is known the spec
   * Measurement Value:
     * Reference point for measuring the artifacts caused from of the Sync Falling Edge
     * Note that this may be clipped since it is the lowest value of the video signal
6. Sync Rising Edge
   * Definition:
     * Transition from the sync tip to the back porch
     * -40 IRE to 0 IRE
   * Length:
     * Same as Sync Falling Edge
   * Measurement Value:
     * Determines the deviation between the expected transition slope and actual transition slope
     * Rising reference pulse that is used to determine the magnitude of the artifacts present in the back porch
7. Back Porch
   * Definition:
     * Flat area at 0 IRE between the falling and rising sync pulses
   * Length:
     * Supplied as a system parameter in samples in decimal form.
     * Transition length is known the spec
   * Measurement Value:
     * Reference point for measuring the artifacts caused from of the Sync Rising Edge
     * Note that this this may contain a color burst or remnants of a filtered color burst


### Gathering measurements
  * The luma information is causal. Artifacts will only ever happen from left to right.
  * Artifacts do not add energy, and are only triggered by whatever happened before them.
  * The entire sync area is one part, so as the measurment points progress through time, each prior state will be caried over to the next
  * Any artifacts that exist in the measurement area that are not correlated with the known input data will be excluded from the measurement


## Modeling the artifacts
* Using the different measurement locations above, a model will be created that describes the different artifacts that were detected
* Ringing:
  * Modeled as a resonance that is controlled by the below variables
    * Frequency Response
      * Difference in amount of ringing based on frequency of causing transition
    * Oscillation Frequency
      * Speed of the oscillations
    * Decay
      * How long it takes for the oscillations to settle
      * This may be linear or logarithmic
* Smear:
  * Modeled as a reverberation
    * Frequency Response
      * Difference in amount of smear based on frequency of the initial transition
    * Decay
      * How long it takes for the smear to settle
      * This may be linear or logarithmic
* Sequencing:
  * Use the artifact sequencing rules to determine in which order to create these models
  * The artifacts being corrected are causal and each section that occurs influences the next one
  * Modeling must represent changes that occur causally only


## Correcting the video
* Correction must only be causal
* Correction must be aware of the stateful nature of the different artifacts that were identified in the measurement process
* Correction inverts / reverses the modeled artifacts considering their type and sequence
* Correction is reactive to it's input only
* When correcting ringing and smearing, the correction should move any extra energy that is spread over these artifacts back into the initial transition that caused this artifact.
  * Smear: All smeared energy should be shifted back into the slope of the transition
  * Ringing: All ringing oscillations should be removed and their energy should contribute to the slope of the transition
* Positive indicators:
  * Flat front porch, sync tip, back porch
  * front and back porch normalize to the same level
  * Sharp and symetric sync transitions
* Negative indicators:
  * Correcting ringing causes the input transition energy to smooth out.
    * This means that the original slope of the transition was modified by the correction.
    * The instruments will show a flatter region following the transition, but this is not correct, since it is smoothing the actual energy of the transition.
  * Correction is not correlated to an input transition.
  * Correction adds additional ringing or smear at different components.