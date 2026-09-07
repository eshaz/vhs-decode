# The modelled residuals, field by field

Ethan asked for a report on all the modelled residuals - what is being modelled, and what the difference is for each field. This is that report, and every number in it was produced by `tools/ringing_measure/modelled_residuals.py` on the decodes named below rather than quoted from an earlier run.

Regenerate it, and the figures with it, with

```
PYTHONPATH=/workspaces/vhs-decode python3 \
    tools/ringing_measure/modelled_residuals.py
```

and add decode prefixes as arguments to run it on new decodes. The tool's settings live in one readable file beside it, `tools/ringing_measure/modelled_residuals.toml`.

## What is being measured, and why it has a field axis at all

Every result the component key has been judged against until now came from the pooled `*_sync_step_response.npz` exports, and those have already averaged the field axis away - into one response per head, plus a first-half and a second-half estimate. Two halves are not a field axis.

A single field carries about 254 horizontal sync pulses, and the sync pulse is the only deterministic step the signal contains: its depth, its width and its transition time are specified, so the measured edge divided by the specified edge is the channel's complex response as that one field witnessed it. This tool measures that response for each field separately and fits the whole component key to each, so what follows is a curve over the decode rather than a single number for it.

The price is precision. Pooled over one field's own lines the instrument reaches a standard error of about 0.019 in the response magnitude, some fifteen times coarser than the pooled export's, so a larger part of one field's departure is the instrument's own noise than of the pooled export's - noise no component can explain. Where a figure below is meant to be read against the 27 to 32 dB table recorded in `modelable_subspace`, it is the modelable-subspace column and not the weighted least-squares one: the two estimators differ, and the section on each decode says how.

The instrument was checked against the certified pooled export rather than assumed to agree with it. Pooling this tool's per-field intervals over one head and taking the same step response reproduces the export's window exactly (lags 26 to 105, 79 samples, 0.1812 MHz information spacing), its fitted crossing to 0.007 samples (45.010 against 45.017) and its step to 0.02 IRE (-37.455 against -37.438). The two responses agree to 0.08 dB and 0.6 degrees below 1 MHz and part company above 2 MHz - 1.1 to 2.1 dB, 8 to 17 degrees - where the sync edge's own spectrum is weakest and the two instruments' dropout hygiene differs. That disagreement is the bound on what any figure here claims at the top of the band.

A measurement per field is not a licence for a correction per field, and the two must not be confused. Rule 4 of `vhsdecode/addons/RINGING_RULES.md` stands: the channel's ringing is a fixed property of the deck, the measurement is accumulated over the whole decode and the APPLIED correction must not change from field to field. What the field axis is for is diagnosis - seeing where the model stops describing the tape, and how much of what is left moves - not per-field fitting.

## One correction to the shared nuisance set, and what it was worth

The measured departure is defined as the response with the terms that are not shapes removed - a gain belongs to the levels estimator and a delay to the time base, and charging either to the model would flatter it. `residual_floor.nuisance` offers two such terms, a constant real part and a linear phase taken about the BAND'S CENTRE, and calls the second a delay. It is not one. A delay of tau writes in the log domain as `-2 pi f tau`, which on a centred grid is `-2 pi (f - f0) tau` minus the constant `2 pi f0 tau`; the ramp is in that span and the constant is not, so a pure delay leaves a frequency-independent phase behind that no entry of the key can express and every fit charges to the residual. A second term earns its place alongside it for a different reason: the phase is unwrapped from the first bin's principal value, so the whole curve is defined only up to a constant, and without a free constant phase the answer depends on which branch the unwrap happened to take.

This tool therefore fits three: a level, a phase reference and a delay through zero frequency. What the shared two-term set costs was measured in this run rather than assumed, by fitting the same key both ways on every field:

| decode | explained, three terms | explained, the shared two | remainder, three terms | remainder, the shared two |
| --- | --- | --- | --- | --- |
| countdown tape, correction off | 81.0 % | 72.6 % | 20.7 | 24.7 |
| home recording, correction off | 91.2 % | 46.8 % | 16.4 | 47.9 |
| pluge/needle-pulse bars, correction off | 84.4 % | 67.0 % | 16.5 | 25.4 |

Every figure in the rest of this report uses the three-term set. The shared one is left untouched: other lanes read it, and a change to it is Ethan's to make. It is recorded here because the difference is not small - it is the difference between a model that is judged to explain two thirds of the departure and one that explains five sixths of it.

## The figures

- **countdown tape, correction off**
  - `/output/modelled_residuals_ss_cd_off_fields.png`
  - `/output/modelled_residuals_ss_cd_off_components.png`
  - `/output/modelled_residuals_ss_cd_off_frequency.png`
- **home recording, correction off**
  - `/output/modelled_residuals_ss_home_off_fields.png`
  - `/output/modelled_residuals_ss_home_off_components.png`
  - `/output/modelled_residuals_ss_home_off_frequency.png`
- **pluge/needle-pulse bars, correction off**
  - `/output/modelled_residuals_ss_pnb_off_fields.png`
  - `/output/modelled_residuals_ss_pnb_off_components.png`
  - `/output/modelled_residuals_ss_pnb_off_frequency.png`
- **all decodes**
  - `/output/modelled_residuals_summary.png`

## countdown tape, correction off

`ss_cd_off`: 200 fields, 200 of them carrying a usable sync edge, 254 admitted lines per field, the fall view over 0.203 to 3.499 MHz. 944 frequency bins carry 19 independent points - the window is 79 samples long and its information spacing is 0.1812 MHz, so the bins are 50 times finer than the information and are counted as the cells they belong to.

### The totals, per field

| quantity | median over fields (10th to 90th percentile) |
| --- | --- |
| measured departure, amplitude | 2.33 (2.27 to 2.39) dB rms |
| remainder after every component, amplitude | 0.739 (0.638 to 0.91) dB rms |
| measured departure, phase | 10.3 (8.72 to 12) degrees rms |
| remainder after every component, phase | 6.07 (5.19 to 8.65) degrees rms |
| departure, in the instrument's own standard errors | 46.7 (34 to 61.2) |
| remainder, in the same units, in sample | 20.7 (13.5 to 30.4) |
| remainder, in the same units, held out | 22.7 (14.4 to 32.1) |
| share of the departure explained, weighted least squares | 81 (70.9 to 86.5) per cent |
| share explained, modelable-subspace projection | 45.9 (26.7 to 71.2) per cent |
| rank the data supports, of 14 | 10 (10 to 10) |
| remainder above the medium's particulate floor | 15.2 (13.9 to 17) dB |
| the same, in the convention recorded in `modelable_subspace` | 33.9 (31.2 to 35.2) dB |

The two share rows are two different estimators and are meant to be read as such. The weighted least-squares figure is the one the rest of this report uses: the key fitted with each bin weighted by its own precision, behind the three nuisance terms. The modelable-subspace figure is `modelable_subspace.exhausted`, which removes only the mean, weights every bin equally and projects through the singular value decomposition at a rank cut set by the measurement's own relative error. It is the lower of the two because it lets the band's noisiest bins carry the same weight as its best, and because it does not fit the delay; it is reported because it is the form the arc's recorded result was taken in.

The two floor rows differ by about twenty decibels and both are given because neither is wrong. A resolution cell of this recording holds 4534 oxide particles whose orientations are independent, so its magnetisation fluctuates by one over the root of that count in RELATIVE amplitude - 0.1290 dB of log amplitude, which is the same quantity the residual above is measured in. That is the first row. The second is `modelable_subspace.against_the_floor`, which divides the DEPARTURE's own energy by the particle count rather than the signal's; it is the convention behind the 27 to 32 dB table recorded in that module, and it reads higher because a departure is a small perturbation of a signal and not the signal. The conclusion is the same either way: the remainder is far above anything the medium's own physics imposes.

### Each component: what it is, what it explains, what survives

`in the chain` is the share of the departure the component removes when it is applied after its predecessors in the chain's own declared order. `only this one` is what it explains that no other entry in the key can reach - the gap between the two columns is the key's collinearity, not a measurement error. `held out` refits that same rung on the OTHER fields of the same head, freezes every coefficient and applies it unchanged here, so a component that has learned its own noise does not survive it. `survives held out` is a sign test: under the null that a component carries nothing out of sample its descent is as likely negative as positive, so the count of fields it lowers is binomial with a half, and the decision is that count against a coin at the five per cent level. The test is distribution free because the descents across fields are neither independent nor Gaussian.

| component | what it is | in the chain, % | only this one, % | held out, % | fields lowered | survives held out |
| --- | --- | --- | --- | --- | --- | --- |
| vestigial sideband | the broadcast channel's asymmetric sideband shaping, before the recorder ever saw the signal | 46.600 | 0.232 | +46.502 | 200 of 200 (p = 6.2e-61) | yes |
| group delay | a delay that varies with frequency, so the band's parts arrive at different times | 8.360 | 1.770 | +8.061 | 188 of 200 (p = 4.1e-42) | yes |
| sound trap | the notch that removes the intercarrier sound at 4.5 MHz | 15.444 | 0.781 | +15.283 | 200 of 200 (p = 6.2e-61) | yes |
| echo at 0.30 us | a reflection arriving after the direct path | 0.152 | 1.572 | +0.005 | 105 of 200 (p = 0.26) | no |
| echo at 0.61 us | a reflection arriving after the direct path | 0.542 | 2.082 | +0.511 | 153 of 200 (p = 1.4e-14) | yes |
| echo at 1.21 us | a reflection arriving after the direct path | 0.700 | 1.279 | +0.695 | 200 of 200 (p = 6.2e-61) | yes |
| echo at 2.43 us | a reflection arriving after the direct path | 0.267 | 0.083 | +0.273 | 184 of 200 (p = 1.2e-37) | yes |
| sub emphasis level dependence | the record-side emphasis behaving differently at different input levels | 2.683 | 0.668 | +2.811 | 184 of 200 (p = 1.2e-37) | yes |
| record level dependence | the tape's own transfer changing with how hard it was recorded | 0.057 | 0.066 | +0.000 | 105 of 200 (p = 0.26) | no |
| head contact tilt | the head's contact varying across the track, tilting the response | 0.893 | 1.672 | +0.790 | 143 of 200 (p = 5.1e-10) | yes |
| dropout | a loss of contact, with a shape even where its position is random | 0.242 | 0.066 | +0.249 | 177 of 200 (p = 6.2e-31) | yes |
| modulation noise | noise proportional to the recorded signal rather than added to it | 0.631 | 0.188 | +0.567 | 194 of 200 (p = 5.3e-50) | yes |
| particle noise | the finite number of oxide particles in the volume the head reads | 1.785 | 1.785 | +1.891 | 163 of 200 (p = 2.3e-20) | yes |
| beat / co-channel | a beat at a definite frequency, even when its phase is random | 0.276 | 0.276 | +0.282 | 199 of 200 (p = 1.3e-58) | yes |

### The fields that differ most

The decode is not one population. Searching the total explained share and every component's share separately, the largest single shift on the field axis falls at field 53, in the share explained by dropout: the 53 fields before it have a median of 0.427 per cent and the 147 after it 0.204 per cent. That step is 1.6 times the field-to-field scatter of the same quantity, at p = 1.6e-08. Nothing in the key has a time axis, so a shift like this can only appear in the residual - which is exactly why the field axis had to be measured rather than averaged away.

The comparison is taken WITHIN a head, and that is not a formality. Fields alternate between the two video heads on a helical drum, the two heads' departures do not correlate, and their instruments do not even have the same noise: the standard errors below differ between the heads, so a remainder quoted in standard errors is not comparable across them. Ranking every field of a decode together would report a head difference as a field difference.

| head | fields | the instrument's own standard error in the response magnitude | remainder, standard errors | remainder, dB | explained, % | field of the largest remainder | of the smallest | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 100 | 0.0188 | 24.8 (21.4 to 33.3) | 0.732 (0.617 to 0.865) | 81.5 (72 to 86.9) | 14 at 41.0 | 178 at 17.3 | 2.36x |
| b | 100 | 0.0320 | 15.2 (12.9 to 20.2) | 0.743 (0.666 to 0.922) | 80.4 (70.1 to 86.4) | 151 at 22.7 | 31 at 10.9 | 2.09x |

## home recording, correction off

`ss_home_off`: 200 fields, 200 of them carrying a usable sync edge, 254 admitted lines per field, the fall view over 0.203 to 3.499 MHz. 944 frequency bins carry 19 independent points - the window is 79 samples long and its information spacing is 0.1812 MHz, so the bins are 50 times finer than the information and are counted as the cells they belong to.

### The totals, per field

| quantity | median over fields (10th to 90th percentile) |
| --- | --- |
| measured departure, amplitude | 2.05 (2 to 2.12) dB rms |
| remainder after every component, amplitude | 0.552 (0.498 to 0.648) dB rms |
| measured departure, phase | 21.7 (18.4 to 25.1) degrees rms |
| remainder after every component, phase | 6.67 (5.71 to 7.54) degrees rms |
| departure, in the instrument's own standard errors | 55.4 (46.7 to 76.9) |
| remainder, in the same units, in sample | 16.4 (12.7 to 25.3) |
| remainder, in the same units, held out | 18.7 (13.1 to 29.2) |
| share of the departure explained, weighted least squares | 91.2 (88.3 to 93.9) per cent |
| share explained, modelable-subspace projection | 63.1 (52.2 to 65.3) per cent |
| rank the data supports, of 14 | 10 (10 to 10) |
| remainder above the medium's particulate floor | 12.6 (11.7 to 14) dB |
| the same, in the convention recorded in `modelable_subspace` | 32.2 (32 to 33.4) dB |

The two share rows are two different estimators and are meant to be read as such. The weighted least-squares figure is the one the rest of this report uses: the key fitted with each bin weighted by its own precision, behind the three nuisance terms. The modelable-subspace figure is `modelable_subspace.exhausted`, which removes only the mean, weights every bin equally and projects through the singular value decomposition at a rank cut set by the measurement's own relative error. It is the lower of the two because it lets the band's noisiest bins carry the same weight as its best, and because it does not fit the delay; it is reported because it is the form the arc's recorded result was taken in.

The two floor rows differ by about twenty decibels and both are given because neither is wrong. A resolution cell of this recording holds 4534 oxide particles whose orientations are independent, so its magnetisation fluctuates by one over the root of that count in RELATIVE amplitude - 0.1290 dB of log amplitude, which is the same quantity the residual above is measured in. That is the first row. The second is `modelable_subspace.against_the_floor`, which divides the DEPARTURE's own energy by the particle count rather than the signal's; it is the convention behind the 27 to 32 dB table recorded in that module, and it reads higher because a departure is a small perturbation of a signal and not the signal. The conclusion is the same either way: the remainder is far above anything the medium's own physics imposes.

### Each component: what it is, what it explains, what survives

`in the chain` is the share of the departure the component removes when it is applied after its predecessors in the chain's own declared order. `only this one` is what it explains that no other entry in the key can reach - the gap between the two columns is the key's collinearity, not a measurement error. `held out` refits that same rung on the OTHER fields of the same head, freezes every coefficient and applies it unchanged here, so a component that has learned its own noise does not survive it. `survives held out` is a sign test: under the null that a component carries nothing out of sample its descent is as likely negative as positive, so the count of fields it lowers is binomial with a half, and the decision is that count against a coin at the five per cent level. The test is distribution free because the descents across fields are neither independent nor Gaussian.

| component | what it is | in the chain, % | only this one, % | held out, % | fields lowered | survives held out |
| --- | --- | --- | --- | --- | --- | --- |
| vestigial sideband | the broadcast channel's asymmetric sideband shaping, before the recorder ever saw the signal | 12.806 | 1.025 | +12.792 | 200 of 200 (p = 6.2e-61) | yes |
| group delay | a delay that varies with frequency, so the band's parts arrive at different times | 54.617 | 4.878 | +53.500 | 189 of 200 (p = 2.6e-43) | yes |
| sound trap | the notch that removes the intercarrier sound at 4.5 MHz | 7.957 | 2.522 | +7.928 | 200 of 200 (p = 6.2e-61) | yes |
| echo at 0.30 us | a reflection arriving after the direct path | 2.060 | 0.729 | +2.068 | 186 of 200 (p = 7.9e-40) | yes |
| echo at 0.61 us | a reflection arriving after the direct path | 1.290 | 0.208 | +1.240 | 197 of 200 (p = 8.3e-55) | yes |
| echo at 1.21 us | a reflection arriving after the direct path | 0.089 | 0.606 | +0.084 | 187 of 200 (p = 5.9e-41) | yes |
| echo at 2.43 us | a reflection arriving after the direct path | 0.111 | 0.025 | +0.109 | 177 of 200 (p = 6.2e-31) | yes |
| sub emphasis level dependence | the record-side emphasis behaving differently at different input levels | 0.713 | 0.077 | +0.654 | 186 of 200 (p = 7.9e-40) | yes |
| record level dependence | the tape's own transfer changing with how hard it was recorded | 0.018 | 0.476 | -0.003 | 81 of 200 (p = 1) | no |
| head contact tilt | the head's contact varying across the track, tilting the response | 10.049 | 10.189 | +9.942 | 192 of 200 (p = 3.6e-47) | yes |
| dropout | a loss of contact, with a shape even where its position is random | 0.261 | 0.242 | +0.253 | 185 of 200 (p = 9.9e-39) | yes |
| modulation noise | noise proportional to the recorded signal rather than added to it | 0.209 | 0.209 | +0.202 | 197 of 200 (p = 8.3e-55) | yes |
| particle noise | the finite number of oxide particles in the volume the head reads | 0.015 | 0.017 | -0.013 | 74 of 200 (p = 1) | no |
| beat / co-channel | a beat at a definite frequency, even when its phase is random | 0.368 | 0.368 | +0.376 | 186 of 200 (p = 7.9e-40) | yes |

### The fields that differ most

No shift on the field axis is worth calling a step. The best split found, at field 13 in the share explained by dropout, separates 0.446 per cent from 0.258 - only 2.34 times the field-to-field scatter of that quantity, at p = 0.021. With two hundred fields a step of a twentieth of a point clears any significance test and means nothing, so the size against the scatter is what decides. What varies between fields here varies without a step in it.

The comparison is taken WITHIN a head, and that is not a formality. Fields alternate between the two video heads on a helical drum, the two heads' departures do not correlate, and their instruments do not even have the same noise: the standard errors below differ between the heads, so a remainder quoted in standard errors is not comparable across them. Ranking every field of a decode together would report a head difference as a field difference.

| head | fields | the instrument's own standard error in the response magnitude | remainder, standard errors | remainder, dB | explained, % | field of the largest remainder | of the smallest | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 100 | 0.0263 | 20.2 (17.4 to 27) | 0.568 (0.541 to 0.654) | 89.7 (87.2 to 91.1) | 152 at 38.0 | 80 at 13.3 | 2.86x |
| b | 100 | 0.0342 | 13.6 (12.1 to 15.5) | 0.525 (0.476 to 0.646) | 92.8 (91.3 to 94.2) | 57 at 23.3 | 125 at 10.9 | 2.13x |

## pluge/needle-pulse bars, correction off

`ss_pnb_off`: 26 fields, 26 of them carrying a usable sync edge, 254 admitted lines per field, the fall view over 0.203 to 3.499 MHz. 944 frequency bins carry 19 independent points - the window is 79 samples long and its information spacing is 0.1812 MHz, so the bins are 50 times finer than the information and are counted as the cells they belong to.

### The totals, per field

| quantity | median over fields (10th to 90th percentile) |
| --- | --- |
| measured departure, amplitude | 1.98 (1.95 to 2.02) dB rms |
| remainder after every component, amplitude | 0.651 (0.598 to 0.67) dB rms |
| measured departure, phase | 5.66 (5.16 to 5.99) degrees rms |
| remainder after every component, phase | 3.65 (3.33 to 4.25) degrees rms |
| departure, in the instrument's own standard errors | 41.8 (30.5 to 55.8) |
| remainder, in the same units, in sample | 16.5 (12.3 to 22.1) |
| remainder, in the same units, held out | 16.6 (12.4 to 22.5) |
| share of the departure explained, weighted least squares | 84.4 (81.6 to 86.5) per cent |
| share explained, modelable-subspace projection | 45.6 (43.2 to 49.4) per cent |
| rank the data supports, of 14 | 10 (10 to 10) |
| remainder above the medium's particulate floor | 14.1 (13.3 to 14.3) dB |
| the same, in the convention recorded in `modelable_subspace` | 33.9 (33.6 to 34.1) dB |

The two share rows are two different estimators and are meant to be read as such. The weighted least-squares figure is the one the rest of this report uses: the key fitted with each bin weighted by its own precision, behind the three nuisance terms. The modelable-subspace figure is `modelable_subspace.exhausted`, which removes only the mean, weights every bin equally and projects through the singular value decomposition at a rank cut set by the measurement's own relative error. It is the lower of the two because it lets the band's noisiest bins carry the same weight as its best, and because it does not fit the delay; it is reported because it is the form the arc's recorded result was taken in.

The two floor rows differ by about twenty decibels and both are given because neither is wrong. A resolution cell of this recording holds 4534 oxide particles whose orientations are independent, so its magnetisation fluctuates by one over the root of that count in RELATIVE amplitude - 0.1290 dB of log amplitude, which is the same quantity the residual above is measured in. That is the first row. The second is `modelable_subspace.against_the_floor`, which divides the DEPARTURE's own energy by the particle count rather than the signal's; it is the convention behind the 27 to 32 dB table recorded in that module, and it reads higher because a departure is a small perturbation of a signal and not the signal. The conclusion is the same either way: the remainder is far above anything the medium's own physics imposes.

### Each component: what it is, what it explains, what survives

`in the chain` is the share of the departure the component removes when it is applied after its predecessors in the chain's own declared order. `only this one` is what it explains that no other entry in the key can reach - the gap between the two columns is the key's collinearity, not a measurement error. `held out` refits that same rung on the OTHER fields of the same head, freezes every coefficient and applies it unchanged here, so a component that has learned its own noise does not survive it. `survives held out` is a sign test: under the null that a component carries nothing out of sample its descent is as likely negative as positive, so the count of fields it lowers is binomial with a half, and the decision is that count against a coin at the five per cent level. The test is distribution free because the descents across fields are neither independent nor Gaussian.

| component | what it is | in the chain, % | only this one, % | held out, % | fields lowered | survives held out |
| --- | --- | --- | --- | --- | --- | --- |
| vestigial sideband | the broadcast channel's asymmetric sideband shaping, before the recorder ever saw the signal | 41.136 | 2.758 | +41.087 | 26 of 26 (p = 1.5e-08) | yes |
| group delay | a delay that varies with frequency, so the band's parts arrive at different times | 5.443 | 5.359 | +5.414 | 25 of 26 (p = 4e-07) | yes |
| sound trap | the notch that removes the intercarrier sound at 4.5 MHz | 22.021 | 1.966 | +21.917 | 26 of 26 (p = 1.5e-08) | yes |
| echo at 0.30 us | a reflection arriving after the direct path | 3.860 | 2.556 | +3.934 | 26 of 26 (p = 1.5e-08) | yes |
| echo at 0.61 us | a reflection arriving after the direct path | 2.904 | 0.086 | +2.879 | 26 of 26 (p = 1.5e-08) | yes |
| echo at 1.21 us | a reflection arriving after the direct path | 0.106 | 1.259 | +0.101 | 21 of 26 (p = 0.0012) | yes |
| echo at 2.43 us | a reflection arriving after the direct path | 0.014 | 0.019 | +0.002 | 14 of 26 (p = 0.42) | no |
| sub emphasis level dependence | the record-side emphasis behaving differently at different input levels | 0.292 | 4.956 | +0.291 | 26 of 26 (p = 1.5e-08) | yes |
| record level dependence | the tape's own transfer changing with how hard it was recorded | 1.101 | 2.046 | +1.092 | 26 of 26 (p = 1.5e-08) | yes |
| head contact tilt | the head's contact varying across the track, tilting the response | 1.162 | 0.011 | +1.160 | 26 of 26 (p = 1.5e-08) | yes |
| dropout | a loss of contact, with a shape even where its position is random | 0.418 | 0.019 | +0.405 | 26 of 26 (p = 1.5e-08) | yes |
| modulation noise | noise proportional to the recorded signal rather than added to it | 2.213 | 0.722 | +2.202 | 26 of 26 (p = 1.5e-08) | yes |
| particle noise | the finite number of oxide particles in the volume the head reads | 3.489 | 3.499 | +3.498 | 26 of 26 (p = 1.5e-08) | yes |
| beat / co-channel | a beat at a definite frequency, even when its phase is random | 0.032 | 0.032 | +0.030 | 25 of 26 (p = 4e-07) | yes |

### The fields that differ most

The decode is not one population. Searching the total explained share and every component's share separately, the largest single shift on the field axis falls at field 8, in the share explained by dropout: the 8 fields before it have a median of 0.481 per cent and the 18 after it 0.282 per cent. That step is 1.9 times the field-to-field scatter of the same quantity, at p = 0.0013. Nothing in the key has a time axis, so a shift like this can only appear in the residual - which is exactly why the field axis had to be measured rather than averaged away.

The comparison is taken WITHIN a head, and that is not a formality. Fields alternate between the two video heads on a helical drum, the two heads' departures do not correlate, and their instruments do not even have the same noise: the standard errors below differ between the heads, so a remainder quoted in standard errors is not comparable across them. Ranking every field of a decode together would report a head difference as a field difference.

| head | fields | the instrument's own standard error in the response magnitude | remainder, standard errors | remainder, dB | explained, % | field of the largest remainder | of the smallest | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 13 | 0.0200 | 20.8 (19.8 to 22.5) | 0.62 (0.589 to 0.65) | 85.6 (83.4 to 87.1) | 18 at 23.8 | 12 at 19.4 | 1.23x |
| b | 13 | 0.0356 | 12.5 (12.1 to 13.3) | 0.658 (0.652 to 0.674) | 83.5 (81.2 to 84.8) | 11 at 13.5 | 13 at 11.9 | 1.14x |

## What all of it comes to

- **countdown tape, correction off**: the 14 modelled components together explain 81.0 per cent of the median field's departure, leaving 20.7 standard errors in sample and 22.7 held out, and that remainder sits 15.2 dB above the floor the medium's own particle count fixes (33.9 dB in the convention recorded in `modelable_subspace`).
- **home recording, correction off**: the 14 modelled components together explain 91.2 per cent of the median field's departure, leaving 16.4 standard errors in sample and 18.7 held out, and that remainder sits 12.6 dB above the floor the medium's own particle count fixes (32.2 dB in the convention recorded in `modelable_subspace`).
- **pluge/needle-pulse bars, correction off**: the 14 modelled components together explain 84.4 per cent of the median field's departure, leaving 16.5 standard errors in sample and 16.6 held out, and that remainder sits 14.1 dB above the floor the medium's own particle count fixes (33.9 dB in the convention recorded in `modelable_subspace`).

Three things follow from those numbers, and they are the reason this report has a field axis.

**The remainder is not noise.** It stands well above the instrument's own standard error and far above the medium's particulate floor, and it is a REMAINDER rather than a measurement: the same components, frozen on other fields, reproduce most of what they removed here. Structure remains, and the modelling is not near the limit the physics fixes.

**The key is collinear, and the two share columns say by how much.** A component's share in the chain is often several times what it alone explains. That is the identifiability ceiling this arc has already measured directly - six magnetic mechanisms collapsing to 1.58 distinguishable directions - showing up here as a difference between two columns of the same table.

**The field axis carries real variation.** The per-field remainder moves by a large factor across a decode, and the movement is not the same on the two heads. A key with no time axis cannot follow it, which is exactly what the pooled held-out work predicted when it found the half-split difference exceeding what the within-pool error allows.

---

Generated by `tools/ringing_measure/modelled_residuals.py`. The component key is `vhsdecode/models/interference.signatures`, its chain order `interference.full_chain`, the projection and the medium's floor `vhsdecode/models/modelable_subspace`, and the weighted fit, the log domain and the resolution cells `vhsdecode/models/residual_floor`.
