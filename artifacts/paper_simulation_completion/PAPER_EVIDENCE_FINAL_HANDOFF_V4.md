# Paper Evidence Independent-Verifier Closure — Handoff

**Classification (as of the round-5 correction commit):** `PAPER_SIMULATION_EVIDENCE_COMPLETE_WITH_6_STRUCTURAL_ONLY_CLAIMS` (default, non-strict invocation; the label names the exact count, computed dynamically by the verifier from its own run, never hardcoded)
**Claims:** 35 total — 29 `QUALIFIED_SOURCE_RESULT_VERIFIED`, 6 `SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED`, 0 `SOURCE_RESULT_LOCATED_NOT_REVERIFIED` (reported separately here on purpose — do not summarize as "35 qualified")
**Verifier:** `verify_paper_evidence_v4.py` — executable, exits 0
**Portability:** identical result with `PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1`
**Physical simulations launched this session:** 0 — `NO_NEW_PHYSICAL_SIMULATIONS_REQUIRED_FOR_CURRENT_DRAFT`

> **Round-5 correction note:** a fifth review accepted this branch's physical-
> simulation conclusion and executable-verifier architecture, but found the
> comparators too permissive (e.g. Fig.6A only checked a table count and
> nonnegativity, not any actual magnitude) and the claim-ID inventory
> self-referential (`EXPECTED_CLAIM_IDS` was derived from `CLAIMS` itself,
> so it could not detect a claim silently deleted from `CLAIMS`). Both are
> fixed in this same branch/commit — see **Round-5 corrections** below. One
> claim (Fig5B) was additionally downgraded from `QUALIFIED_SOURCE_RESULT_VERIFIED`
> to `SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED` as part of tightening
> its comparator (its "path deflection" sub-claim has no bundled geometric
> data), which is why the structural-only count is 6, not the round-4 count
> of 5.

## Round-5 corrections

1. **Independent frozen claim inventory.** `expected_paper_claim_ids_v4.json`
   is now a hand-authored, hash-stamped, 35-ID contract, loaded independently
   by the verifier and checked against the live registry in both directions.
   It is never generated from `claim_registry_v4.CLAIMS`.
2. **Tightened comparators** for `Fig2-peak-narrow-topology` (now requires a
   PF/sharp-front peak, FEM below the analytic/PF peak, and ~30.9%
   attenuation, not just "a peak exists somewhere"), `Fig4-coverage-and-theta`
   (theta is now parsed from the config and gated, not hardcoded as a
   string), `Fig4-DBTT-transition-shift` (adds an independent max-gradient
   transition-temperature definition alongside the disclosed shelf-midpoint
   proxy), `Fig5A` (gates on steep_cleavage having the max Paris slope and
   plastic_shielded the min, not just "six monotonic curves"), `Fig5B` (checks
   all 5 common extension points, not 2; gates the ~17x orientation ratio
   with a declared tolerance; downgraded to STRUCTURAL since path deflection
   itself is not geometrically verified), `Fig5C` (adds a censor-aware
   stress-life reconstruction: does median cycles-to-connection decrease from
   700 to 900 MPa, not just a shielding pass-rate contrast), `Fig5D` (adds a
   distinct `visual_inspection_fig5d.json` record with image hashes, checked
   for existence/consistency rather than treating file-presence as morphology
   verification), `Fig6A` (freezes and checks all 36 Cramer's V values within
   1e-4, plus the four manuscript-described magnitude groups), `Fig6B`
   (checks the pooled correlation and all 6 context correlations explicitly,
   not just n and the min/max range), `Sec2.15` (the 3%/10% tolerances are
   now applied by the registry's comparator against raw error measurements,
   not read from a precomputed boolean), and `SI-identifiability` (narrow
   +/-2-point tolerance around the actual reconstructed extrema, not a broad
   containing interval).
3. **Bundle file count reconciled**: 44 files physically present in
   `source_bundle_figures_2_4/` = 39 manifest-governed + 5 compact-
   projection-governed (see the reconciliation table below; a round-4 draft
   of this document incorrectly said "47").
4. **11 new adversarial tests** (23 total, up from 12): a real whitespace-
   only terminal-status test (the round-4 version only asserted a Python
   truthiness fact without calling `evaluate_claim`), plus tests proving
   each tightened comparator rejects the specific corrupted result the fifth
   review named (Fig6B pooled-r corruption, Fig6A flattened-but-nonnegative
   values, Fig5A monotonic-but-undistinguished curves, Fig5C reversed
   stress-life ordering, Fig2 peak-without-attenuation), and four tests
   proving the independent frozen inventory catches a deleted, renamed,
   added, or duplicated claim.
5. **Dynamic terminal classification**: `PAPER_SIMULATION_EVIDENCE_COMPLETE_WITH_<N>_STRUCTURAL_ONLY_CLAIMS`
   (N computed live) replaces a bare `PAPER_SIMULATION_EVIDENCE_COMPLETE`
   whenever any claim is held at the honest `SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED`
   ceiling, so the label itself cannot be misread as "every claim fully
   quantitatively verified."

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

## Adversarial tests (23 total, all pass)

`tests/test_paper_evidence_verifier_v4.py`. The original 12 (round 4) prove
the verifier: alters an expected value → FAIL; ignores a manually-forged
`pass_fail`; fails before recompute when a required file is missing;
catches a duplicated claim ID; catches a stale/mismatched hash; fails on a
blank terminal-status field even with a correct value; flags a multi-input
claim with only one file present; correctly hides external roots with zero
filesystem mutation; and reproduces the terminal classification from a
clean temporary checkout with external roots hidden.

Round 5 adds 11 more (fixing one that was nominal-only, per the review): a
*real* whitespace-only terminal-status test that calls `evaluate_claim`
directly (the round-4 version only asserted `bool("   ".strip()) is False`
without exercising the verifier); five tests proving each tightened
comparator rejects its specific named corrupted result (Fig6B pooled-r
corruption with n/range untouched, Fig6A values flattened but still
nonnegative, Fig5A monotonic-but-undistinguished curves, Fig5C reversed
stress-life ordering, Fig2 peak-without-attenuation); and five tests proving
the independent frozen inventory (`expected_paper_claim_ids_v4.json`) is
genuinely external to the registry and catches a deleted, renamed, added,
or duplicated claim, plus confirms the real registry matches it exactly.

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
- `artifacts/paper_simulation_completion/source_bundle_figures_2_4/` — see
  **Bundle file count reconciliation** below for the exact, non-conflicting
  counts (a round-4 draft of this document stated "grows from 35 to 47
  files", which did not match the machine-readable manifest and has been
  corrected).
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

## Bundle file count reconciliation

Three counts describe `source_bundle_figures_2_4/`, and they are consistent
(the round-4 handoff draft's "47" figure was simply wrong and is corrected
here):

| Count | Meaning | Value |
|---|---|---|
| Manifest-governed | Files copied verbatim from an external original by `build_source_bundle_figures_2_4.py`, tracked with a byte-identity SHA-256 in `paper_source_bundle_manifest.json` | 39 |
| Compact-projection-governed | Derived/joined files produced in-repo by the three `build_fig*_portable_projection.py` scripts, tracked with their own SHA-256 in `fig6b_portable_projection_provenance.json` / `fig6c_portable_projection_provenance.json` (3 tables) / `fig5c_portable_projection_provenance.json` | 5 |
| **Total physically present** | `find source_bundle_figures_2_4 -maxdepth 1 -type f \| wc -l` | **44** |

`39 + 5 = 44`, matching the physical file count exactly — there is no
unexplained gap. `file_hashes_v4.json` covers the entire
`artifacts/paper_simulation_completion/` directory (not just the bundle
subdirectory), so its file count is larger still and is not directly
comparable to the 44 above; see that file's own `n_files` field for its
scope.

## Known, disclosed limitation

The three `build_*_portable_projection.py` and `build_source_bundle_figures_2_4.py`
scripts (which *generate* the bundle from external sources) still hardcode
the external root paths directly rather than going through
`external_roots.py` — this is intentional and low-risk: they are one-time,
already-executed provenance-generation steps, not part of the verifier's
execution path, so they play no role in the hidden-root portability proof.
A future pass could retrofit them for consistency, but doing so does not
change any claim's evidence class.
