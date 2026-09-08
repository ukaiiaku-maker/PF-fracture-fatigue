# Paper Evidence Final Closure — Handoff

**Classification:** `PAPER_SIMULATION_EVIDENCE_COMPLETE`
**Terminal condition met:** A — every retained claim is source-qualified
**Structural checks:** PASS, 0 structural failures, 0 unresolved rows (`paper_simulation_completion_verification_v3.json`)

This branch (`codex/v10.2.30-paper-evidence-final-closure`) is a child of
`codex/v10.2.30-paper-evidence-provenance-closure` (commit `a62c85a`), created
in response to a third review that found the second pass's Fig.4 handling
unacceptable (an admitted verification gap was resolved by *rewording the
admission* rather than fixing it), the Fig.3 verification anchored to a
derived summary rather than raw seed data, the evidence non-portable
(depending on a live, non-git, mutable external path), the claim matrix too
coarse, and Figures 1, 5, 6, and the SI identifiability study entirely
untraced.

## What changed since the second pass

1. **Fig.3 corrected**: recomputed from `seed_Rcurve_metrics_and_fits.csv`,
   the genuine 20-row (5 seeds × 4 classes) RAW seed-level table, not the
   derived `class_Rcurve_metric_summary_complete_only.csv` used previously.
   All 12 K0/Kss/ΔK_R values reproduce the manuscript exactly.

2. **Fig.4 corrected, not reworded**: the DBTT-transition-shift claim is now
   genuinely, independently recomputed with a fully disclosed method
   (midpoint-of-shelves linear interpolation), resolving cleanly and
   monotonically at all 4 rate factors (0.1x→788.9K, 1x→898.2K,
   10x→1031.4K, 100x→1125.1K). The "analytical peak shifts with rate, FEM
   retains muted shoulders" claim is fully confirmed after locating a fine
   5K-resolution, rate-resolved analytic file: the peak location shifts
   monotonically (860K→910K→970K→1035K) and FEM's own curve shows zero
   local maxima at any of the 4 rates (a consistent "muted shoulder"), with
   an exact 30.9% attenuation figure available at 1x.

3. **Fig.2 corrected**: RMS values recomputed directly from raw
   `Kc_first`/`K_analytic_interp` columns (not the pipeline's own
   precomputed `error_vs_analytic` column); all 4 classes match exactly.

4. **Portability**: `artifacts/paper_simulation_completion/source_bundle_figures_2_4/`
   now holds 35 small files (CSV/JSON/MD/PNG) copied with SHA-256
   byte-identity confirmed against their external originals, covering every
   figure (1, 2, 3, 4, 5, 6, 7, SI) — the branch no longer depends on any
   external path to reproduce its own recomputed numbers (with one disclosed
   exception: Fig.6B's 40MB raw monotonic-points file is too large to bundle
   and is read live when available; its absence is detected and reported,
   not silently passed).

5. **Figures 1, 5, 6, and the SI study located and verified.** A dedicated
   forensics pass found their sources in two previously-unexamined
   directories:
   - `/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF/` — Figures 1, 5, 6
   - `/Volumes/Data/Data/Nanopillar_calculation/Fatigue_modelfitting_identifyability/` — SI study

   The two most numerically load-bearing claims were independently
   **re-derived from raw data in this session** (not merely re-read from the
   forensics report):
   - **Fig.6B**: re-implemented the exact join+Pearson procedure from
     `analyze_v57_final_integrated.py` from scratch against the raw
     per-surface CSVs → n=1360, pooled r=0.9865, context range
     [0.958, 0.999] — exact match to the manuscript.
   - **SI study**: recomputed `100 * RMSE(G_fit-G_true) / RMS(G_true)` from
     scratch from the raw `inversion_summary.csv` and per-regime
     `truth_barrier_grid.csv` files → 75 conditions/regime; sparse-acquisition
     range 23.4–54.4%; complete-dataset range 1.2–9.4% — both ranges fall
     inside the manuscript's stated "~20-55%" and "~1-10%".

   Two Fig.5 claims (blunt-notch handoff-audit contrast, and the shielded-
   vs-unshielded spatial field images) were also independently checked by
   directly opening the raw JSON audits and viewing both PNG field images in
   this session, not by trusting a paraphrase.

6. **`codex/v10.2.30-inverse-fatigue-barrier-design` ruled out** (not merely
   "unconfirmed") as the SI source, by direct inspection of its code.

7. **Claim matrix rebuilt at true claim granularity**: 35 rows (one per
   figure panel; one per Fig.3 class-statistic; one per Fig.4 sub-claim),
   each carrying the full required field set (source level, repository,
   immutable hash, original + portable paths, producer script, parameter
   fingerprint, terminal/censor status, recomputation method, expected vs.
   recomputed value, tolerance, pass/fail, final evidence class).

8. **Verifier rebuilt on structured boolean fields**
   (`independent_recomputation_performed`, `source_bundle_present` via
   byte-identity check, `numerical_comparison_passed`, `terminal_status_verified`)
   instead of prose/keyword scanning — no wording change alone can flip its
   verdict. It caught and forced correction of two real issues during this
   same pass (a mislabeled PARTIAL-pass row, and a malformed bundle-path
   string) before this final state was reached.

## Files in this package

- `paper_claim_evidence_matrix_v3.{csv,json}` — 35 claim-level rows, all
  `QUALIFIED_SOURCE_RESULT_VERIFIED`.
- `paper_source_bundle_manifest.{csv,json}` + `source_bundle_figures_2_4/`
  (35 files) — the portable evidence bundle with SHA-256 byte-identity.
- `paper_quantitative_reproduction.json` / `paper_quantitative_reproduction_fig1_5_6_si.json`
  / `paper_quantitative_reproduction_table.csv` — every independently
  recomputed number.
- `figure_1_5_6_si_forensics.json` — the structured forensics record merged
  into the matrix.
- `paper_unresolved_claims.csv` — empty (0 rows) at this commit.
- `proposed_manuscript_corrections.md` — states explicitly that no
  corrections are needed, and documents what changed to reach that state.
- `paper_simulation_completion_verification_v3.json` — the structured
  verifier's output; `classification: PAPER_SIMULATION_EVIDENCE_COMPLETE`.
- `paper_completion_contract_v3.json`, `file_hashes_v3.json` — this pass's
  summary and a SHA-256 of every file in this directory.
- Carried forward unchanged from earlier passes: `campaign_branch_verification_registry.{csv,json}`
  (now with the inverse-fatigue-barrier-design row corrected to a definitive
  ruled-out finding), `temperature_fatigue_exclusion_review.{csv,json}`.

## What is still knowingly out of scope (not a gap in this closure)

- The 7 temperature-fatigue numerical exclusions' raw 36-row registry
  remains unrecovered (gitignored, not found anywhere on this filesystem
  after searching additional locations this round). This was already
  disclosed in the prior pass and no manuscript conclusion was found to
  depend on resolving a specific excluded row; re-running the campaign was
  judged, again, not justified.
- The 6 other named campaign branches (two-scale virtual CT,
  A_native+PT panel, R-ratio/ΔK, analytical overlay, physical-slope-transfer)
  remain unpushed and not directly linked to the current manuscript by
  citation or provenance; this is unchanged from the prior pass and outside
  this round's scope (only the inverse-fatigue-barrier-design branch's status
  changed, from "plausible" to "ruled out").

## How to reproduce

```
cd <this worktree>
python3 scripts/build_source_bundle_figures_2_4.py
python3 scripts/recompute_figures_2_3_4_from_bundle.py
<conda env with pandas/scipy> scripts/recompute_figures_1_5_6_si_from_bundle.py
python3 scripts/build_figure_1_5_6_si_forensics.py
python3 scripts/build_campaign_branch_verification_registry.py
python3 scripts/build_paper_claim_evidence_matrix_v3.py
python3 scripts/verify_paper_simulation_completion_v3.py   # exits 0
python3 scripts/finalize_final_closure.py
```

`recompute_figures_1_5_6_si_from_bundle.py` needs pandas/scipy for its
Fig.6B recomputation; this session used
`/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-fem-czm/bin/python`.
All other scripts run under the system `python3` (stdlib only).

Every claim's evidence is reproducible from the committed
`source_bundle_figures_2_4/` alone, with one disclosed exception (Fig.6B's
40MB raw monotonic-points file, read live from
`/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF/` when available).
