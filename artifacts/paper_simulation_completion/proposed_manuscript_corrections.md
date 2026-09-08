# Proposed manuscript corrections

**Status: no corrections proposed.**

This file is generated as a required deliverable of the paper-evidence
final-closure pass regardless of outcome. As of `paper_claim_evidence_matrix_v3.csv`
/ `paper_simulation_completion_verification_v3.json`, all 35 claim-level rows
carry `final_evidence_class = QUALIFIED_SOURCE_RESULT_VERIFIED`, and the
strict structured-field verifier (`verify_paper_simulation_completion_v3.py`)
reports `classification = PAPER_SIMULATION_EVIDENCE_COMPLETE` with zero
structural failures and zero unresolved manuscript-critical rows. Terminal
condition A (every retained claim source-qualified) is met, so no claim
requires narrowing, softening, or removal.

## What changed to reach this state

Two rows were held at a downgraded/partial status earlier in this same pass
and were subsequently fully resolved by locating better source data, not by
relaxing the evidentiary bar:

1. **Fig4-peak-narrow-attenuation** ("the analytical peak shifts with rate,
   whereas FEM/CZM retains only muted shoulders"). The original 100K-spaced
   FEM-vs-analytic comparison grid only happened to resolve the peak class's
   narrow intermediate-temperature local maximum at the 1x rate by chance of
   grid alignment, leaving the "shifts with rate" component unconfirmed at
   the other three rates. Locating `analytical_predictions_by_rate.csv` (a
   fine 5K-resolution, rate-resolved analytic file, now bundled as
   `fig4_analytical_predictions_by_rate_fine_grid.csv`) allowed a genuine,
   monotonic confirmation of the peak-location shift (860K -> 910K -> 970K
   -> 1035K as rate increases 0.1x -> 1x -> 10x -> 100x), and a
   zero-local-maxima check on FEM's own curve at every rate confirmed the
   "muted shoulders" half of the claim at all four rates, not just one.

2. **Fig5D** (spatial fields showing plastic shielding maintains a broad
   cyclic deformation state without localizing into a connected crack). This
   session directly viewed both `fem_fields_final.png` field snapshots
   (no-shield and shielded, same seed/stress) rather than relying on a
   paraphrased description, and confirmed the qualitative and quantitative
   contrast (accumulated plastic strain and dislocation density both
   dramatically larger and more spatially diffuse in the shielded case,
   consistent with "broader deformation... without localization").

## If a genuine gap re-emerges in a future pass

Should a future review or re-verification (e.g. after this branch's external
dependencies change, or new manuscript text is drafted) find that any row's
`final_evidence_class` regresses from `QUALIFIED_SOURCE_RESULT_VERIFIED`,
this file is the correct place to record the specific proposed replacement
manuscript text -- narrowing a claim to what the surviving evidence supports,
or removing it -- rather than silently omitting the affected figure/number.
No such regression exists at the time of this commit.
