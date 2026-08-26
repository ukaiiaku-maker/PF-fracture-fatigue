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
