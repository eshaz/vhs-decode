# The adversarial review that broke four published claims

These are the scripts of an independent review whose brief was to REFUTE, not
to confirm — to reproduce each claim in `docs/ELLIPTICAL_COLLAPSE.md` from the
code and the data, and to look actively for the construction that would make
each number an artefact.

They are kept because four of the claims did not survive, and a record of a
retraction is worth more than a record of a result. Each script is
self-contained and prints its evidence.

| script | what it established |
|---|---|
| `c1_trace.py` | `trace(G) = N` is an identity of row normalisation, confirmed to 12 decimals for rank-1, wildly-scaled and collinear ensembles — a check that division works, not a discovery |
| `c2_sphere_floor.py` | the floor `N/(L+N−1)` holds to 0.94–1.05 over `N ∈ {3…128} × L ∈ {8…1024}`, tighter than the document claimed — **but the scatter `√2/L` is the complex large-`N` asymptote**, and a real ensemble scatters at about `2/L` |
| `c2b_saturated.py` | when `N` approaches `L` the true scatter collapses by 3× to 245×, so the "saturated, not empty" reading of the amplitude and field axes does not follow — against the correct scatter they clear three sigma |
| `c3b_mp.py` | the Marchenko–Pastur edge value is right and row normalisation does not move it, but "never admits noise" is false: the false-direction rate is 2–11% and *rises* with size |
| `c4_halting.py`, `c4b_diag.py`, `c4c_planting.py` | the halting count is not flat in the dimensions; and the test's planting helper gives every departure above the second the same ramp, so **at most three independent directions exist whatever is asked for** — "recovered equals planted" was checked against a ground truth that does not exist above three |
| `c5_wiener.py` | the Wiener/gain-law identity is a rearrangement of one MMSE expression; `bulk` is biased low so the weight is biased high; and **on the real decode every weight is exactly 1.0**, so the mechanism was inert there |
| `c6_identifiability.py`, `c6b_typical.py` | 1.58 of 6 reproduces exactly — but only after supplying an azimuth value the model sets to zero, and the "exactly collinear" gap/azimuth pair separates at 1° |
| `c7b_noise_baseline.py`, `c7c_correct_null.py` | **the headline finding: pure noise reproduces the real-data run number for number**, and the correct construction-aware null shows real structure on every axis at +3.9σ to +128.7σ |
| `c8_capacity.py` | the envelope-to-C/N conversion is 3.01 dB generous on the tape side only, and Carson's deviation should be the peak, not the peak-to-peak swing |
| `c9_where_it_works.py` | six independent vectors correctly report 0.0% resolved; adding their fifteen pairwise differences — which add no information — makes the same function report 95.7% |
| `c10_tuner_site.py` | the over-the-air sample against the home control on the two witnesses a clip at the **tuner** would leave on the sync tip: **neither separates them**, and the one-sided departure points the other way (+0.19/+0.09 IRE over the air against +0.51/+0.63 at home). Both are confounded — the instrument's ideal is amplitude-fitted, which absorbs the depth a clip removes, and the porch carries burst the tip does not. What *is* attested is the premise: cd's recorded sync is 4.608 µs against home's 4.676 and the test tapes' 4.650 |
| `c37_construction.py` | the seven-line surrogate null that any statistic on a nested matrix needs |

Everything they established is carried in `docs/MATHEMATICS.md` §10 and
`docs/ELLIPTICAL_COLLAPSE.md` §6c, and the defects that must keep reproducing
are proofs 12–15 of `tools/ringing_measure/proofs.py`.

These scripts were written against the tree as it stood on 2026-09-04 and are
kept as a record rather than maintained; `proofs.py` is the maintained form.
