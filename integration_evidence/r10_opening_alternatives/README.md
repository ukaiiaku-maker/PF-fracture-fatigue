# R10 opening alternatives evidence

Additive descendant of source commit `a74c44c5e4f2166cc687f30faedbadecf86c1d8c` on branch
`codex/r10-athermal-mixed-czm-comparison`.

The legacy `D_COHESIVE_ARRHENIUS_EXP_PLASTICITY` implementation and its results
remain unchanged. New runs identify `D_MECHANICAL_MIXED_CZM_EXP_PLASTICITY` and
use `cohesive_athermal` or `cohesive_mixed` with `smooth_exponential` or
`bilinear` shape. The legacy cooperative completion clock is not called by any
new arm.

## Executed hierarchy

1. `analytical_matched_8192.csv.gz`: 8,192 matched tuples and four arms.
2. `state_informed_48_executed.csv`: fresh retained-engine state exposure for
   all 48 allocated complete materials.
3. `native_12_case_dispositions.csv`: 12 materials, four arms, three loads.
4. `developed_followup_histories.json.gz`: uncapped one-micrometre-cell
   follow-up for `R10D_COH_27231`; advancing responses are single-excursion
   bursts and are censored from developed-rate fitting.
5. `selected_2d_qualification.json`: nonlinear same-material 2-D cohesive and
   EXP_floor-state transfer at 300, 750, and 1200 K.

`results_summary.json` contains file hashes and compact dispositions;
`RESULTS.md` is the human-readable decision record.
