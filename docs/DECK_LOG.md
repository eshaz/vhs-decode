# The deck log

Two of Ethan's directives, verbatim, are the reason this file exists:

> Log head swaps and maintenance dates. Gap geometry resets at a swap while
> preamp and servo don't - that's a natural experiment for separating head
> terms from deck terms.

> GPS/NTP-disciplined timestamps on captures. Over a long acquisition,
> capture-chain drift becomes a hidden variable correlated with everything;
> timestamps let you regress it out.

The log is one readable TOML file per bench, edited by hand, with nothing in
the shell environment (Ethan's standing rule for tooling). The template is
`tools/ringing_measure/deck_log.toml`; the code that reads it is
`vhsdecode/models/deck_log.py`; the tests are `tests/unit/test_deck_log.py`.

    PYTHONPATH=/workspaces/vhs-decode python3 -m vhsdecode.models.deck_log tools/ringing_measure/deck_log.toml

prints the validation report, every dated event with what it resets, the
spans over which a wear rate may be calibrated, and the natural experiments
the captures already form.

## 1. Why a log is a measurement instrument

The identifiability work established that the magnetic mechanisms of one
head reading one tape collapse to 1.58 distinguishable of six on a frequency
response alone: gap and azimuth are collinear at coherence 1.000, spacing
and thickness at 0.994. `tape_model.confounded()` lists the pairs a single
measurement cannot split, and no amount of better fitting splits them,
because the confound is in the physics of the measurement. What does split
them is a CHANGE that moves one member of a pair and not the other, and the
only such changes available are the ones a deck's life provides. A head swap
resets every head-owned parameter and leaves the preamp, the servo and the
tape untouched. A cleaning resets the debris share of the spacing and not
the wear. A belt or pinch-roller change moves the transport's rates and
nothing magnetic. Each is a natural experiment, and each is usable only if
it was written down with a date.

Time is the same kind of instrument. A capture chain drifts - its crystal
ages and warms, the deck's servo reference too - slowly enough to be a
constant within one short decode and a slow term over a long acquisition,
correlated with everything else that is slow. A timestamp per capture
disciplined to GPS or NTP makes that term a regressor on a known axis. A
file's modification time is a clock nobody checked, marking the end of the
write rather than the start of the capture, and the log records it as
exactly that.

## 2. The schema

```toml
schema = 1

[bench]                     # free text: name, notes

[clock]                     # the drift regressor's bound: LABELLED ASSUMPTIONS
aging_ppm_per_year = 5.0
thermal_ppm_per_kelvin = 0.5
temperature_excursion_kelvin = 10.0
status = "typical AT-cut crystal specification figures; not measured on any device in this log"

[[capture_devices]]         # id, kind, make, model, crystal_hz, bit_depth
[capture_devices.settings]  # a free table: for a CX card the cxadc
                            # parameters exactly as /sys/class/cxadc/... lists
                            # them (sixdb, tenbit, audsel, center_offset,
                            # latency, crystal, vmux, tenxfsc, level); for a
                            # scope its timebase, coupling, probe, connection

[[decks]]                   # id, make, model, serial, reference_crystals_hz,
                            # rf_taps, notes
[[decks.head_swaps]]        # date (required), heads, part, notes
[[decks.maintenance]]       # date (required), kind (required), notes
                            # kind: head cleaning | transport service |
                            #       electronics repair | alignment

[[tapes]]                   # id, stock, recorded / recorded_from / recorded_to,
                            # recorded_on (a deck id), source, notes

[[captures]]                # file, path, deck (required), tape, source, speed,
                            # tap (record | playback), device (a capture_devices
                            # id), sample_rate_hz (required), bit_depth
                            # (required), timestamp (required),
                            # timestamp_source (required: what clock gave it),
                            # timestamp_discipline (required: gps | ntp | none)
```

Dates are TOML dates (`2025-07-12`) or offset date-times
(`2025-07-13T04:02:57Z`); a bare date is taken as midnight UTC.

### Validation

`deck_log.validate` separates what refuses the log from what it merely
reports. Errors, which `load(strict=True)` refuses on:

- a capture without a `deck`, or naming a deck that is not in the log - it
  cannot enter any natural experiment;
- a capture without a `timestamp`, a `timestamp_source`, or a
  `timestamp_discipline` in `gps | ntp | none` - it cannot enter the drift
  regressor;
- a capture without `sample_rate_hz` or `bit_depth`;
- a capture naming a `device` that is not in `[[capture_devices]]`;
- a head swap or maintenance entry without a date, or a maintenance entry
  whose `kind` is not one of the four the reset table knows;
- a `[clock]` table missing any of its four keys, because a bound nobody
  stated is not a bound;
- a wrong `schema`.

Warnings, which name what is unrecorded and are meant to stay visible: a
deck whose make is unknown or whose serial is missing, an undisciplined
timestamp (with its source repeated), a capture without a tape, speed or
device, a tape id not in `[[tapes]]`.

## 3. What each event resets

The physical statement is one table, `deck_log.EVENT_RESETS`, with the
reasoning beside each entry. The head-owned terms are
`head_model.CHANNEL_PARAMETERS` (azimuth, gain, height, protrusion) plus the
gap, the head's efficiency, and the head's own and the debris shares of the
Wallace spacing; the tape-owned terms are `tape_model.variations()`; the
deck terms are the preamp gain and equalization, the record current, the
level control, the de-emphasis, the servo reference, the capstan, pinch
roller and idler rates, and the guide alignment.

| event | resets | leaves alone |
|---|---|---|
| head swap | every head term: azimuth, gain, height, protrusion, gap, efficiency, head spacing, debris | the preamp, the servo, the transport, the tape |
| head cleaning | the debris share of the spacing only | wear (protrusion), the tape's own roughness |
| transport service | capstan, pinch-roller and idler rates | everything magnetic and electronic |
| electronics repair | preamp gain and equalization, level control, de-emphasis, servo reference | the heads and the transport |
| alignment | the guide alignment (tracking offset) | everything else |

## 4. Natural experiments

`deck_log.natural_experiments(log)` returns every pair of captures on one
deck that straddles a dated event, once per event, ranked:

- rank 0, the SAME TAPE replayed before and after - every tape term is held
  exactly;
- rank 1, the same SOURCE signal on a different tape - the picture is held,
  the tape is not;
- rank 2, different material.

Each entry names what the event `moved` (its resets), what the pair `holds`
(the tape terms or the picture, plus every deck term the event did not
touch), and `separates`: the confounded pairs of which the event moved
exactly one member. Those pairs come from `tape_model.confounded()` - each
carrying `instruments_separate`, the verdict of `tape_model.separable` on
whether the project's instruments could already tell the two apart - and
from `deck_log.CHAIN_CONFOUNDS`, the head-against-tape and head-against-deck
pairs any frequency-domain fit apportions arbitrarily: the head's and the
tape's shares of one Wallace spacing, the debris share against both, the
head's efficiency and gain against the preamp's gain, the protrusion against
the head spacing, and the gap-against-azimuth collinearity measured at
coherence 1.000. A head swap splits the head's spacing share from the tape's
surface and the head's gain from the preamp's; it does NOT split gap from
azimuth, because both live on the head and both move, and the entry lists
that pair under `still_confounded`.

The wear clock is tied in through `head_model.aging_rate` and
`head_model.interval_years`. A swap resets the wear the rate is measured
from, so `deck_log.aging_spans(log, deck)` gives the date spans between
swaps within which a rate may be calibrated, `deck_log.dating_valid(log, a,
b)` refuses `interval_years` for two captures with a swap between them, and
every swap-spanning experiment carries `dating_valid = false`.

## 5. The drift regressor

`deck_log.drift_regressor(log, series_by_capture, nominal=None, order=1)`
takes, per capture named in the log, a series against seconds from that
capture's start, and:

1. builds ONE absolute time axis from the captures' timestamps
   (`time_axis`). Within a capture the sample clock is the axis and is
   always usable. Across captures the axis is only as good as the
   timestamps: if any capture in the set is undisciplined the joint axis is
   refused, the result says why, and each capture is fitted alone under
   `per_capture`;
2. fits the slow term, a polynomial of `order` in time (a straight line by
   default - aging is linear over any span a bench sees, and a thermal
   excursion adds curvature only if the temperature moved) and returns the
   slope with an error that respects the residual's serial correlation
   (`head_model.effective_sample_size`, on the residual, never the raw
   series), the fitted term and the residual;
3. with a `nominal` (the series' nominal value, so that the drift is in
   parts per million), checks the total drift over the span against the
   crystal bound from the log's `[clock]` table: aging in ppm per year times
   the span, plus the thermal coefficient in ppm per kelvin times the stated
   temperature excursion. A drift inside the bound is "consistent with
   capture-chain drift"; one beyond it "exceeds what a crystal can do: not
   the capture chain".

`deck_log.regress_out(target, regressor)` then removes the fitted slow
term's shape from any other series on the same axis by least squares and
reports the share it explained.

The bound's figures are assumptions and are labelled as such in the file:
typical AT-cut crystal specification values, 5 ppm per year of aging and
0.5 ppm per kelvin over an assumed 10 K excursion. They live in the log,
not in the module, so a bench that has measured its own reference replaces
them and says so in `status`. The capture card's crystal is 28 636 363 Hz -
the `crystal` parameter recorded in `/testdata/test_patterns/cvbs/readme.txt`
- and the deck's schematic parts list carries 3.579545 MHz, 16 MHz and
32.768 kHz crystals (read with pdftotext, 2026-09-05); which of them
references the servo has not been traced.

## 6. What the template holds, and what it does not

Populated from what is KNOWN:

- the deck: Sony SLV-778HF, from the capture file names; its RF taps (CN261
  pin 1 record, pin 2 playback), the disabled Adaptive Picture Control and
  the TSG-130A source from the vhs readme; its crystals from the schematic;
- the capture device for the test-pattern captures: the Rigol DS1202Z-E
  with the settings the readme gives (10 ms/div for 50 MS/s, channel 1, AC
  coupling, 1X, a Dupont-to-BNC lead with no buffer or termination, 2822
  bytes of settings stripped, 8-bit); and the stock white CX card with its
  cxadc parameters from the cvbs readme, so a CX capture of a deck can be
  entered the same way;
- eight captures: the 75% bars and bounce patterns at SP (playback and
  record taps) and EP (playback), from the file names (deck, speed, 50 MSps)
  and the readme (8-bit); and the two two-hour captures, home.flac and
  countdown.flac, 40 MSps 8-bit from their archive names, under a deck
  entry whose make is `unknown` because no record says which deck played
  them;
- three tapes: the test-pattern tape recorded 2025-07-12 on the deck; the
  home tape whose recording period, 1994-01 to 1994-10, is in its archive
  name; and the off-air tape, whose inserted test signals the
  content-independence check measured on lines 15-20 of every field.

Not populated, because it is not known, and left visible as validation
warnings: the deck's serial number; every tape's stock; the capture device
and the capture date of the two-hour captures; and every timestamp's
discipline, which is `none` throughout - the only timestamps that exist are
file modification times (for the two-hour captures, of the archive copy,
not the capture). No head swap and no maintenance has been recorded, so
`natural_experiments` returns nothing yet; the moment one is entered with
its date, every capture before and after it on that deck pairs up.

Two experiments the existing material could form: replaying the
test-pattern tape after any future head swap is a rank-0 pair with the tape
held exactly; and the SP-against-EP pairs of 2025-07-12 would be a
head-pair change with the preamp and servo held IF the SLV-778HF plays EP
on separate heads, which the schematic text does not settle and the log
therefore does not claim.
