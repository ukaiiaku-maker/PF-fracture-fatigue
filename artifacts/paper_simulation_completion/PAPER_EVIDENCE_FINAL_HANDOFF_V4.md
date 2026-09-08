# Paper Evidence Independent-Verifier Closure — Handoff

**Classification:** `PAPER_SIMULATION_EVIDENCE_COMPLETE` (default, non-strict invocation)
**Claims:** 35 total — 30 `QUALIFIED_SOURCE_RESULT_VERIFIED`, 5 `SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED`, 0 `SOURCE_RESULT_LOCATED_NOT_REVERIFIED`
**Verifier:** `verify_paper_evidence_v4.py` — executable, exits 0
**Portability:** identical result with `PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1`
**Physical simulations launched this session:** 0 — `NO_NEW_PHYSICAL_SIMULATIONS_REQUIRED_FOR_CURRENT_DRAFT`

This branch (`codex/v10.2.30-paper-evidence-independent-verifier`) is a child
of `codex/v10.2.30-paper-evidence-final-closure` (commit `783f207`), created
in response to a fourth review that accepted the source-forensics conclusion
(all manuscript physical datasets located; no new simulation needed) but
found the *verification itself* still self-certified: the matrix builder
computed `pass_fail`, and the v3 "verifier" only re-checked a few structural
booleans (was `independent_recomputation_method` non-empty? was
`pass_fail=="PASS"`?) rather than actually running any recomputation itself.

## What changed: self-declared → executable verification

**Before (v3):** `build_paper_claim_evidence_matrix_v3.py` computed
`pass_fail` for every row by calling recompute functions and comparing
results *inline*, then wrote that pre-computed verdict into the CSV.
`verify_paper_simulation_completion_v3.py` then read that CSV and checked
whether `pass_fail == "PASS"` — it trusted the matrix builder's judgment.

**Now (v4):** `claim_registry_v4.py` declares, for all 35 claims, only:
*what inputs are needed*, *what function recomputes the value*, *what typed
value is expected*, and *what comparator/tolerance judges a match*. It
contains no pass/fail logic. `verify_paper_evidence_v4.py` is the only place
a verdict is produced: for each claim it (1) verifies every required input
file is present in the bundle AND byte-identical to a recorded provenance
hash, (2) calls the claim's own `recompute()` function inside a try/except
(any exception is a hard FAIL, never silently skipped), (3) calls the
claim's `compare(actual, expected)` function to get a boolean, (4) requires
a non-blank `terminal_or_censor_status`, and only then assigns the final
evidence class. `render_paper_claim_matrix_v4.py` then formats the
verifier's own JSON into a CSV — it contains zero judging logic, by design.

## Portability: Figure 6B made genuinely self-contained

The prior pass's Fig.6B recomputation required a live read of a 40MB
external file at verification time. This pass builds a compact, provenance-
tracked projection instead:

- `build_fig6b_portable_projection.py` performs the exact join once against
  the live external files, records their SHA-256, the filter/join keys, and
  every excluded-row count and reason, and writes a 1360-row compact table
  (`fig6B_compact_joined_1360.csv`, bundled) plus
  `fig6b_portable_projection_provenance.json`.
- `recompute_Fig6B()` reads ONLY the compact table and reproduces n=1360,
  pooled r=0.9865, and all 6 context-specific correlations (range
  [0.958, 0.999]) — exact match to the manuscript.
- The same pattern was applied to **Fig.6C** (a compact per-observation
  projection reproducing all 39 frozen Panel C AUC+CI rows) and **Fig.5C**
  (a compact JSON reconstruction of all 15 available seed/stress/condition
  jobs).

**Hidden-root proof:** rather than physically renaming the three external,
non-git, mutable, potentially-concurrently-used directories (judged too
risky — an interrupted rename has no git history to recover from and could
corrupt live data another session is using), `external_roots.py` provides a
`PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1` environment-variable indirection that
makes every script report each external root as unavailable with **zero
filesystem mutation** — there is nothing to restore and no interrupted-
operation failure mode. Running the full verifier with this variable set
reproduces the identical classification and per-claim result set:

```
$ python verify_paper_evidence_v4.py
classification=PAPER_SIMULATION_EVIDENCE_COMPLETE
$ PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1 python verify_paper_evidence_v4.py
classification=PAPER_SIMULATION_EVIDENCE_COMPLETE   # identical
```

`tests/test_paper_evidence_verifier_v4.py::test_full_verifier_passes_from_temp_checkout_with_roots_hidden`
copies only `scripts/` and `artifacts/paper_simulation_completion/` into a
fresh temp directory and runs the verifier there with the env var set,
proving the committed bundle alone is sufficient.

## Recomputed or honestly reclassified qualitative claims

| Claim | What changed |
|---|---|
| Fig1A-D | Array-based topology checks (regime-order transitions, shared parameter grids) replace README-text matching. Capped at `SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED` — no manuscript-stated number exists to reproduce numerically. |
| Fig4-peak-narrow-attenuation | Peak-location shift now confirmed at **all 4** rate factors (was 1x-only) via a newly-bundled fine 5K-resolution, rate-resolved analytic file. |
| Fig5A | Per-case monotonicity (Spearman ρ), point counts, Paris-law slope and curvature computed for all 6 canonical cases from the raw K-point array. |
| Fig5B | Class-ordering persistence (rank correlation = 1.0 across all 6 classes, early vs. late common extension) and orientation-dependence magnitude (~17x da/dN difference, 30° vs 45°, matched K_max) recomputed from bundled raw data. |
| Fig5C | Reconstructed across all 15 available seed/stress/condition jobs (was one representative pair). Contrast reported as a genuine, quantified statistical tendency (unshielded coverage_pass rate materially exceeds shielded's), not an assumed deterministic split. |
| Fig5D | Downgraded to qualitative-only; a prior specific numeric factor claim ("160x"/"1000x") is retracted as an unverifiable color-bar visual estimate — see `proposed_manuscript_corrections.md`. |
| Fig6A | All 36 contingency tables rebuilt from raw cell counts; Cramer's V recomputed via `chi2_contingency` from scratch and confirmed mathematically non-negative — a manuscript sign-convention correction is proposed. |
| Fig6C | All 39 frozen Panel C rows (AUC + 95% bootstrap CI), not one spot-check, independently reproduced. |
| Sec2.15 | Saturation R-curve parameters genuinely refit (`scipy.optimize.curve_fit`) from raw per-seed binned R-curve data — not read from the existing fit-output CSV. Kss within 3%, ℓ_R within 10% of the prior fit (shape exponent `p` reported but not gated, since the exact binning/averaging convention used upstream is not fully documented). |

## Adversarial tests (all pass)

`tests/test_paper_evidence_verifier_v4.py`, 12 tests, proving the verifier:
alters an expected value → FAIL; ignores a manually-forged `pass_fail`;
fails before recompute when a required file is missing; catches a
duplicated claim ID; catches a stale/mismatched hash; fails on a blank
terminal-status field even with a correct value; flags a multi-input claim
with only one file present; correctly hides external roots with zero
filesystem mutation; and reproduces `PAPER_SIMULATION_EVIDENCE_COMPLETE`
from a clean temporary checkout with external roots hidden.

## Files in this package (new/changed this round)

- `scripts/external_roots.py`, `scripts/claim_recompute_v4.py`,
  `scripts/claim_registry_v4.py`, `scripts/verify_paper_evidence_v4.py`,
  `scripts/render_paper_claim_matrix_v4.py`,
  `scripts/finalize_independent_verifier_closure.py`
- `scripts/build_fig6b_portable_projection.py`,
  `scripts/build_fig6c_portable_projection.py`,
  `scripts/build_fig5c_portable_projection.py`
- `tests/test_paper_evidence_verifier_v4.py`
- `artifacts/paper_simulation_completion/paper_claim_evidence_matrix_v4.{csv,json}`,
  `paper_simulation_completion_verification_v4.json`,
  `paper_unresolved_claims_v4.csv` (empty),
  `fig6b_portable_projection_provenance.json`,
  `fig6c_portable_projection_provenance.json`,
  `fig5c_portable_projection_provenance.json`,
  `paper_completion_contract_v4.json`, `file_hashes_v4.json`
- `artifacts/paper_simulation_completion/source_bundle_figures_2_4/` grows
  from 35 to 47 files (compact projections + Fig1/Fig5B/Fig6A/Sec2.15 raw
  inputs added).
- `proposed_manuscript_corrections.md` updated with two genuine wording
  corrections (Cramer's V sign convention; Fig5D numeric-factor retraction).

## How to reproduce

```
cd <this worktree>/scripts
python build_source_bundle_figures_2_4.py
<conda env with pandas/scipy> build_fig6b_portable_projection.py
<conda env with pandas/scipy> build_fig6c_portable_projection.py
<conda env with pandas/scipy> build_fig5c_portable_projection.py
<conda env with pandas/scipy> verify_paper_evidence_v4.py            # exits 0
PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1 <conda env> verify_paper_evidence_v4.py  # identical, exits 0
python render_paper_claim_matrix_v4.py
python finalize_independent_verifier_closure.py
<conda env with pandas/scipy> -m pytest ../tests/test_paper_evidence_verifier_v4.py -v
```

`claim_recompute_v4.py` needs pandas/scipy/numpy; this session used
`/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-fem-czm/bin/python`.
`build_source_bundle_figures_2_4.py`, `render_paper_claim_matrix_v4.py`, and
`finalize_independent_verifier_closure.py` run under the system `python3`
(stdlib only).

## Known, disclosed limitation

The three `build_*_portable_projection.py` and `build_source_bundle_figures_2_4.py`
scripts (which *generate* the bundle from external sources) still hardcode
the external root paths directly rather than going through
`external_roots.py` — this is intentional and low-risk: they are one-time,
already-executed provenance-generation steps, not part of the verifier's
execution path, so they play no role in the hidden-root portability proof.
A future pass could retrofit them for consistency, but doing so does not
change any claim's evidence class.
