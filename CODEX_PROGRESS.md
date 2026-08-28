# Codex Progress

## 2026-08-26 reversible v10.2.30 integration and refined fatigue campaign

- Worktree: `/private/tmp/v10230-reversible-energy-integration`
- Branch: `codex/v10.2.30-reversible-energy-integration`
- Current integration HEAD before this report commit: `80f89bb83bc583ee37d09db6ad1b35f97b1a1f13`
- Environment: `arrhenius-sharp-front-v10-codex`
- Protected process PID 4847 remains stopped and untouched. PID 49115 disappeared independently; Codex did not signal it.

### Solver qualification

- The production v10.2.30 PF/sharp-front solver now carries phase-resolved signed mobile transport, physical source-linked return, separate escape/wake/return ledgers, and bounded net blunting. Cleavage and emission remain opening-only.
- Frozen-state mechanics selected the mechanically derived local signed retained/GND internal stress; nonlocal cleavage shielding remains separate.
- Forward virgin and non-virgin parity, positive-R zero-return behavior, negative-R true reversal, return invariants, one-event energy/geometry atomicity, multi-event evolution, and explicit/accelerated parity pass.
- Qualification artifact: `runs/reversible_solver_qualification_with_parity.json`.
- Accelerated-versus-explicit maximum relative errors: active state `3.81e-10`, diagnostics `1.136e-4`, hazard `2.842e-4`; the accelerator fails closed to exact evolution when validation rejects a projection.
- Canonical monotonic/R-curve parity is exact for Peak, DBTT, weak-T, and ceramic-like: 464 kinetic records and 26 numerical CSV comparisons. Artifact: `runs/four_class_monotonic_parity_8076d5c/four_class_monotonic_parity.json`.

### Refined Paris results (300 K, R=0.1, 1000 Hz)

- Canonical Peak: `m=100.5373`; DBTT: `m=133.5021`; ceramic-like: `m=71.6980`. Weak-T produced unstable transition points and no admissible stable fit. Regenerated canonical bundle: `runs/four_class_paris_refined`.
- Nine pre-existing/provenance-selected parameterizations were tested: four canonical rows, historical A-D, and atlas benchmark `V914_ATLAS_000206671`.
- Historical candidate A (`v914_endurance_knee_0462`) is the successful low-slope basin.
  - n=80, seed 1720, four-point confirmation: `m=4.19955`, `R2=0.7250`, developed rate span `22.80`.
  - n=128, seed 1720: `m=4.60490`, `R2=0.87937`, rate span `21.67`.
  - n=128, independent seed 1001723: `m=4.55976`, `R2=0.88185`, rate span `21.05`.
  - All six n=128 production points reached approximately 102 um with stable developed growth.
- Historical B is excluded from Paris fitting: DeltaK 16.2 censored with no event; DeltaK 21.6 produced one event and only 2.28 um by the 1e6-cycle censor.
- Historical C measured `m=49.7297` (too steep). Historical D did not yield an admissible developed window. Atlas benchmark 206671 measured `m=11.5690` (too steep), so its old analytical ranking did not survive refined-solver evaluation.
- The measured A basin satisfies the automatic-search objective; no broad 20-30-candidate retraining round is scientifically required at this stage.

### Authoritative analysis outputs

- A n=80 confirmation: `runs/refined_candidate_search/analysis_A_confirmed`
- A n=128 seed 1720: `runs/refined_candidate_search/analysis_A_n128`
- A n=128 seed 1001723: `runs/refined_candidate_search/analysis_A_n128_seed1001723`
- All-data, plot-only consolidated atlas (39 cases, no cross-resolution/seed pooled fit): `runs/refined_candidate_search/all_measured_refined_fatigue`
- The consolidated PNG is `all_refined_da_dN_vs_deltaK.png`; its exact source table is `all_refined_fatigue_cases.csv`.

### Restart status

- Fresh uninterrupted trajectories are authoritative. No restored trajectory was admitted to this refined campaign.
- Serialization support does not constitute production restart qualification; restart remains operationally unsupported for these scientific results.

## 2026-08-27 joint fracture-fatigue mechanism-guided atlas

- Branch: `codex/v10.2.30-joint-fracture-fatigue-archetype-atlas`; starting HEAD: `43b5ec9a259b636c5a950eec1f6fffbf3ef4b65c`.
- Added an analysis-only F0/F1/F2/F2B monotonic first-passage hierarchy using the current bounded EXP-floor, exact cooperative gamma renewal, separated opening/cleavage/emission stresses, pre-event transient blunting, and signed mobile/retained moment balances. Production physics is unchanged.
- Inventoried 187 archived monotonic and fatigue records across reduced, 1-D, PF, FEM-derived, and current fatigue evidence. Current persistent-site and legacy v9.11 finite-source/stored-energy lineages remain explicitly separated.
- The four named DOCX sources were not found after recursive project/home/CloudStorage/attachment and Spotlight searches. This is fail-closed in `fracture_source_manifest.json`; no equation or historical bound was reconstructed from selected rows.
- Recovered the original 26-coordinate v9.13 Sobol policy and produced 131,072 initial plus 16,384 adaptive analytical rows (147,456 total). Asymptotic/no-first-passage rows are rejected from archetype promotion and retained in the rejection audit.
- The analytical result supplies useful response ranking and mechanism decomposition, but the current transient state closure lacks complete cross-fidelity coverage and remains insufficient for prospective physical promotion. F2B does not resolve the dominant residual. Temperature-dependent fatigue is labeled `ANALYTICAL_EXTRAPOLATION_UNVALIDATED`.
- Six Pareto-distinct analytical candidates are frozen. The physical controller is terminal with zero new runs: the F2 quantitative gate did not justify new 1-D, PF, FEM/CZM, or stochastic trajectories. Exact-row A_NATIVE/PT03/PT08 archival controls are reused without resume or retuning.
- Authoritative bundle: `runs/joint_fracture_fatigue_archetype_atlas_v1/`; 48 required machine artifacts and all 23 required figures are present.
- Focused new tests: 22 passed. Combined relevant analytical/fatigue tests: 96 passed. The broad `test_v10_2_30*.py` run passed 341 tests and has four unresolved fixture-dependent failures because historical qualification run roots are absent from this isolated worktree; no tolerance, safety check, or test was changed to suppress them.

## 2026-08-28 corrective fracture-fatigue response-atlas qualification

- Continued on `codex/v10.2.30-joint-fracture-fatigue-archetype-atlas` from `0f9bd97d872fe351b872525983f86641bf1aadc3`. This pass changes analysis/diagnostics/tests only; the qualified production solver hash remains `c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b`.
- Recomputed the analytical fatigue slice by 128-phase waveform quadrature and recorded, for every one of 147,456 atlas rows at all four loads, `x_c`, `Lambda_c tau_c`, normalized `da/dN`, and phase fractions near the cooperative ceiling, barrier floor, stress cap, and effectively zero activity.
- Identified 19,004 cooperative-renewal-ceiling-dominated rows; 122,645 rows fail at least one fail-closed asymptotic/monotonic eligibility gate. The exact identity `da/dN / [ell_bar/(f tau_c)] = <Lambda_c tau_c>` is enforced by tests and the verifier.
- Invalidated all six V1 `JFFA_*` selections and removed them from the current candidate and Pareto registries. The corrected joint archetype candidate count is zero. Five non-asymptotic rows are retained only as `UNVALIDATED_RESPONSE_*` exemplars, and every response cluster remains `MIXED_OR_UNRESOLVED` (bootstrap stability 0.667).
- Split validation into matched local 1-D constitutive-kernel tests, PF front-local/mixed-transfer tests, and FEM applied-load transfer tests. On the common 21-row finite F0/F1/F2 population, matched-1D median absolute relative errors are 0.066%, 0.066%, and 0.102%; the FEM F2 median remains 36.9%, with no transfer map fitted or nominal-K equivalence assumed.
- Replaced misleading Sobol/Morris filenames with explicitly labeled binned-variance and dimensionless standardized partial-dependence screens. Trends are stratified by cooperative regime, source activity, mechanism, barrier-floor regime, and monotonic response class; measured outcomes are three broad, two conditional, six counterexample, and one not-testable trend.
- The three-part classification is now `LOCAL_MONOTONIC_KERNEL_VALIDATED_FOR_MATCHED_1D`, `CURRENT_TRANSIENT_STATE_CLOSURE_PARTIAL`, and `CROSS_FIDELITY_MECHANICAL_TRANSFER_UNRESOLVED`. The tested F2B closure was not promoted, but the broader two-compartment model class is explicitly not rejected.
- The four named DOCX sources and the assessment's sandbox archive link are not mounted locally. The source manifest remains fail-closed (`document_ingestion_complete=false`); the equation-lineage report is expanded to a complete code-auditable derivation without claiming document ingestion.
- Authoritative corrected bundle: `runs/joint_fracture_fatigue_archetype_atlas_v2/`; 57 required machine artifacts and 23 PNG/PDF figure pairs are present. No physical, PF, FEM/CZM, stochastic, or production calculation was launched or resumed; active workers are zero.
- Focused tests: 25 passed. Expanded relevant fracture/fatigue suite: 143 passed. Broad `test_v10_2_30*.py`: 344 passed with the same four absent-historical-fixture failures as V1. `py_compile`, `git diff --check`, and the V2 fail-closed verifier pass; no tolerance or safety check was weakened.
