# Paper Evidence Provenance Closure — Handoff

**Classification:** `PAPER_SIMULATION_EVIDENCE_COMPLETE_WITH_EXPLICIT_SOURCE_LIMITATIONS`
**Structural checks:** PASS (see `paper_simulation_completion_verification.json`)
**Fully resolved (zero limitations):** NO — 6 of 15 claim-level rows remain unresolved, disclosed explicitly below.

This branch (`codex/v10.2.30-paper-evidence-provenance-closure`) is a child
of `codex/v10.2.30-paper-simulation-completion` (commit `89054bd`), created
in response to a review that correctly identified real semantic errors in
the v1 claim matrix (manuscript-text-only evidence labelled as "qualified")
and asked for genuine source-provenance forensics rather than inference.

## Headline finding

Manuscript Figures 2, 3, and 4 were traced to their **exact** source data:

```
/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM/
```

This is a plain, non-git-tracked directory — `git rev-parse HEAD` fails
there — genuinely distinct from the git-tracked `Arrhenius_FEM_CZM_MPZ_*`
repositories, which host a **separate** θ=0° parity investigation
(`agent/final-pf-fem-fracture-comparison`, `FINAL_PF_FEM_FRACTURE_
COMPARISON_V2.md`) reporting much larger PF-vs-FEM discrepancies at θ=0°.
That separate investigation does **not** contradict the manuscript: the
manuscript's own comparison is explicitly at θ=45°, and the θ=45°
production directory was independently located and verified.

Every one of the following manuscript numbers was independently
**recomputed from raw source CSVs** (not read from a cache) and found to
reproduce the manuscript **exactly**:

- Fig. 2 RMS deviations from V1 at 1× rate: ceramic 0.23, weakT 0.23,
  peak 1.18, DBTT 1.33 MPa√m — recomputed from
  `runs/PF_vs_CZM_first_passage_with_analytic_publication/first_passage_comparison_with_analytic.csv`.
- Fig. 3 R-curve statistics (all 12 K0/Kss/ΔK_R mean±std values, all 4
  classes) — recomputed from
  `runs/four_class_exp_floor_CZM_500K_5rep_1000um_theta45/Rcurve_analysis/class_Rcurve_metric_summary_complete_only.csv`.
- Sec 2.15 fitted saturation parameters (K_ss, characteristic extension) —
  recomputed from `.../Rcurve_analysis/class_mean_Rcurve_fits.csv`.
- Fig. 4 rate-sweep (0.1×/1×/10×/100×) coverage and θ=45° fingerprint —
  confirmed from `runs/four_class_exp_floor_CZM_rates_no_branch_500um_theta45/`.

## What remains unresolved (disclosed, not hidden)

1. **Figures 1, 5, 6, 7, and the SI synthetic-identifiability study** are
   `MANUSCRIPT_RESULT_NOT_SOURCE_TRACED` — their exact source data/scripts
   were not located in the time available. This does not mean the work
   doesn't exist; it means this session did not find it.
2. **6 named campaign branches** (two-scale virtual C(T), A_NATIVE+PT
   panel, R-ratio/ΔK, analytical overlay, physical-slope-transfer,
   inverse-fatigue-barrier-design) pass `py_compile` and `git diff --check`
   but their own strict verifiers require a `--root` pointing at gitignored
   raw result directories not found on this filesystem. None were found to
   be directly, conclusively cited by the current manuscript; none were
   pushed to `origin`.
3. **4 of the 7 temperature-fatigue numerical exclusions** are not
   individually identified by (class, T, Kmax) — only the 3 Peak-class
   exclusions are, via the campaign's own committed progress log. The raw
   36-row output directory is gitignored and was not found on this
   filesystem; re-running the campaign to recover this detail was judged
   not justified (see `temperature_fatigue_exclusion_review.json`).

## Files in this package

- `paper_claim_evidence_matrix_v2.{csv,json}` — the corrected, claim-level
  matrix (15 rows: 8 `QUALIFIED_SOURCE_RESULT_VERIFIED`, 6
  `MANUSCRIPT_RESULT_NOT_SOURCE_TRACED`, 1 `NOT_REQUIRED_FOR_CURRENT_PAPER`).
- `temperature_fatigue_exclusion_review.{csv,json}` — per-exclusion-group
  review; 0 replacements authorized.
- `campaign_branch_verification_registry.{csv,json}` — the 6-branch
  lightweight verification results.
- `paper_simulation_completion_verification.json` — the strict verifier's
  output (`scripts/verify_paper_simulation_completion.py`).
- `paper_completion_contract_v2.json` — this pass's scope/findings summary.
- `file_hashes.json` — sha256 of every file in this directory.

## How to reproduce

```
cd .../v10230-paper-evidence-provenance-closure
<pinned interpreter> scripts/build_paper_claim_evidence_matrix_v2.py
<pinned interpreter> scripts/build_temperature_fatigue_exclusion_table.py
<pinned interpreter> scripts/build_campaign_branch_verification_registry.py
<pinned interpreter> scripts/verify_paper_simulation_completion.py
<pinned interpreter> scripts/finalize_provenance_closure.py
```

The strict verifier reads `/Volumes/Data/Data/Nanopillar_calculation/
Arrhenius_FEM_CZM/` directly (outside the git repository) to independently
recompute the Fig. 2/3 numbers. If that directory is unavailable in a
future environment, the affected checks **fail** rather than silently
skip — this is deliberate, per the review's instruction not to treat an
unreproducible number as qualified.
