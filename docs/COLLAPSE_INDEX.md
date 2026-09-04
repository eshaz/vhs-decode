# The elliptical collapse: what is where

Ethan's complex multi-dimensional elliptical curve transform — the algorithm,
the mathematics, the evidence, and what each domain it was mapped onto said
back. Start here.

---

## Read in this order

| document | what it is |
|---|---|
| **[THE_ALGORITHM.md](THE_ALGORITHM.md)** | The algorithm alone, in eight numbered steps. His words as the definition, the mathematics beside them, the implementing function named. No findings in it. |
| **[MATHEMATICS.md](MATHEMATICS.md)** | Every closed form derived from its assumptions, each claim marked as design, derivation, or measurement, with the proof number that regenerates it. §10 is the failure modes stated as theorems. |
| **[PROOFS.md](PROOFS.md)** | The output of the proof suite: 38 checks over 22 proofs, regenerating every number in the documents. |
| **[ELLIPTICAL_COLLAPSE.md](ELLIPTICAL_COLLAPSE.md)** | The measured record, in the order it was measured, including §6c — the withdrawal. |
| **[COMPONENT_MAPPINGS.md](COMPONENT_MAPPINGS.md)** | What the domains say to each other: the separability law, what moves identifiability, and the measurement traps. |

Then, per domain, each written by an independent mapping:

| document | subject |
|---|---|
| [BT1700_COLLAPSE.md](BT1700_COLLAPSE.md) | ITU-R BT.1700, the analogue television standards and their timing |
| [PROPAGATION_COLLAPSE.md](PROPAGATION_COLLAPSE.md) | Television propagation — transmitter, multipath, receiver |
| [BROADCAST_VTR_COLLAPSE.md](BROADCAST_VTR_COLLAPSE.md) | U-matic and the broadcast VTR family |
| [CONSUMER_FORMAT_COLLAPSE.md](CONSUMER_FORMAT_COLLAPSE.md) | Betamax, Video8, Hi8, S-VHS, VHS-C |
| [VERTICAL_INTERVAL_COLLAPSE.md](VERTICAL_INTERVAL_COLLAPSE.md) | Test and data signals in the vertical interval |
| [SPECIFICATION_INVENTORY.md](SPECIFICATION_INVENTORY.md) | What the specification collection holds, and what it does not |

`RESIDUAL_LIMIT_DESIGN.md` is the parent design this grew out of and remains
the authority on the residual limit itself.

---

## The code

| file | what it holds |
|---|---|
| `vhsdecode/models/information_extrapolation.py` | `ellipsoid`, `differentiate_to_floor`, `component_differentials`, `graph_relations`, `sphere_floor`, `surrogate_null`, `guess_credibility`, `axes_present` |
| `vhsdecode/models/pair_dimension.py` | `pair_transfer`, `dimensions`, `coherent_pair`, `effective_length` — a dimension is the Wiener transfer between two co-registered channels |
| `vhsdecode/models/interference.py` | the modelled modifications as complex signatures, `span_completeness`, `split_residual`, `distinguishable` |
| `vhsdecode/models/capture_profile.py` | the fourth link: word length from the code lattice, the quantisation floor, `binding_limit` |
| `vhsdecode/models/head_model.py`, `tape_model.py`, `transport_model.py`, `filter_model.py` | the physical models the signatures are built from |

Nothing in `vhsdecode/models/` is imported by the decoder. That is deliberate:
it is the offline model, and only what earns its way through the shipping gate
reaches the runtime.

---

## Running it

```
# the algorithm on a capture, every dimension, with the key's completeness
python3 -m tools.ringing_measure.elliptical_collapse CAPTURE OUT \
    [--json report.json] [-- decode flags ...]

# every claim in the documents, regenerated
python3 -m tools.ringing_measure.proofs
python3 -m tools.ringing_measure.proofs --list
python3 -m tools.ringing_measure.proofs --report docs/PROOFS.md
```

The proof suite reads no capture and runs in seconds. It is also a test
(`tests/unit/test_proofs.py`), so the documents and the code fail together
rather than drift apart.

---

## The evidence

| where | what |
|---|---|
| `tools/ringing_measure/proofs.py` | the maintained proofs — 22 of them, 38 checks |
| `tools/ringing_measure/adversary/` | the independent review that broke four claims, kept as a record with its own README |
| `tests/unit/test_information_extrapolation.py` | 70-odd cases including the ones that assert the defects still reproduce |
| `tests/unit/test_interference.py`, `test_pair_dimension.py`, `test_capture_profile.py` | the modelled key, the pair transfer, the capture profile |

---

## The four claims that did not survive

Kept prominent, because a retraction is worth more than a result.

1. **The nested matrix's statistics.** Pure noise reproduced the real-data run
   number for number — 99.1% resolved at +37.9σ against a reported 92.9% and
   +39.3σ. The construction emits `N(N+1)/2` entries spanning `N` dimensions,
   so the rank was forced, every eigenvalue cleared the noise edge, and the
   asymmetry returned a literal constant. `MATHEMATICS.md` §10.1.
2. **The flat halting count.** The test's planting helper gives every
   departure above the second the same ramp, so at most three independent
   directions ever existed. "Recovered equals planted" was checked against a
   ground truth that does not exist above three.
3. **"Saturated, not empty."** The scatter `√2/L` is the complex large-`N`
   asymptote; a real ensemble scatters at about `2/L`, and when `N` approaches
   `L` the true scatter collapses further. Against the correct scatter the
   amplitude and field axes clear three sigma.
4. **The vertical-interval result.** That function read two lines past the
   interval into active picture, so the "+499σ perfect needle" was the first
   two picture lines of a full-field pattern — and on a programme recording it
   would have fed content into the ellipse, which the sync-only constraint
   forbids.

What survives all four: against a construction-aware null there is real
structure on every axis, from +9.6σ on amplitude to +128.7σ on the time base.
The detection was never the problem; the quantification was.
