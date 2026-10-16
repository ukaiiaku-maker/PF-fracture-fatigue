# Fracture-framework acceptance consolidation — 16 October 2026

## Source identity and disposition

The integrated calculation and disabled-campaign source pair is PF `1c7841ea2253d5a6a8b1a016c482c92fb77a7dfd` and FEM `ee634968deae2b280cc179d330e35575956f4000`. A later PF documentation-only closeout commit may contain this report; it does not change that calculation-source identity. Production and manuscript work remain disabled. No remote write or scientific-data deletion was performed.

The portable source qualification passes: PF reports 1004 passed with 16 explicitly external-evidence tests deselected; FEM reports 754 passed plus 16 passing subtests. The exact commands and timings are in `FINAL_SOURCE_QUALIFICATION_20261016.json`.

## Controller findings and repair

The confirmed defect was an ownership mismatch in the moving-tip bilinear route. `MovingTipBilinearController.advance(front_id=...)` selected one active trial for acceptance but called constitutive helpers that iterated the entire cohesive network. Its rejection path restored only the selected elements and did not own callback mechanical, plastic or time state. A post-update equilibrium exception could therefore leave a partial state, and the returned equilibrium was not checked against an explicit constitutive splitting residual.

FEM `ee634968` repairs the actual public/native chain:

`bilinear_cohesive_request_v1.build_moving_tip_controller` → `MovingTipBilinearController.begin_trial_segment` → `KineticTrialAdaptiveCZMBackend.begin_trial_segment` → native `fem.assemble_mechanics` → front-scoped constitutive update → returned native equilibrium → accepted commit.

The update domain is now exactly `ActiveTrialSegment.element_indices`. Each interval snapshots the full cohesive network, every active-owner record and log, controller counters, and callback-owned state. A damage-limit rejection, constitutive splitting rejection, post-update equilibrium failure, or commit-publication exception restores that whole interval automatically while retaining the active event. Explicit `rollback_active_trial` remains a separate whole-event cancellation that restores the pre-insertion topology and bulk-history baseline.

Autonomous bilinear damage evolves from opening history without a hazard gate. Its damage residual is evaluated at the returned equilibrium. Activated bilinear state carries a persistent event mark and action, accepted physical time, localized consumed time, and unused interval through active restart. The qualification uses an explicitly constructed unit mark and constructed cohesive support; these are not a production stochastic or material calibration.

The original EXP-floor route was not replaced. Its existing protection remains in `kinetic_progressive_2d_v1002` → `KineticEventLifecycle` → `KineticCohesiveStepper.advance` → `KineticTrialAdaptiveCZMBackend`, including mechanics snapshot/restore, full rollback callback, persistent event ownership, and unused-time carry. Nine focused stepper/lifecycle tests pass and are linked as preservation evidence.

## Native R05 evidence

The combined public qualification is `fem-czm-current-pf-companion/integration_evidence/r05_controller_transaction_20261016/qualification_report.json`; both child reports pass.

- Activated child: native topology insertion and identity state transfer; first accepted traction `1.409186313e9 Pa`; persistent action `1.444595378e-29` over `1e-8 s`; cohesive work `2.482257581e-6 J/m`; exact active restart; localized terminal event returns the full unused `1e-8 s`; automatic post-update failure restoration and explicit whole-event rollback both pass.
- Autonomous child: first accepted damage increment `0.0633422176`; cohesive work changes from `2.736688983e-6` to `2.563341034e-6 J/m`; restart continuation commits with pre-commit work `3.849256464e-5 J/m`; automatic post-update failure restoration and explicit whole-event rollback both pass.
- Both returned-equilibrium constitutive residuals are zero in these constructed native qualifications, and the assembled accommodation norm is `1.414213562e-5`.

## Disabled campaign and remaining native executions

The v3 campaign now has 699 disabled rows and 663 source-bound native commands. All 694 prior rows are unchanged in scope. Five selective P8 rows add actual executable request fields for plane stress, generalized plane strain with prescribed `eps_zz`, generalized plane strain with prescribed `sigma_zz`, pure mode II, and mixed I–II. This is a selective qualification set, not a Cartesian production expansion.

The accepted mixed/constraint EXP-floor topology event (R07) and recursive second-generation branch-owner transition (R09) were not launched. On the final admission check the host had about 50 MiB immediately free, about 36 GiB represented in compressed pages, and load averages above 13. These executions remain resource-blocked individually; the independent implementation and qualification work above continued.

R10 contact and downstream child-continuation evidence remains linked to its current local-incremental producers in `integration_evidence/r10_current_local_incremental/consolidated_qualification.json`; no successful R10 calculation was rerun or replaced by historical collective-law evidence.

## Taylor law and historical-result boundary

The Taylor amplitude revision remains an authorized, real prospective parameter change. The records preserve separate parent and revised identities and distinguish raw thermal amplitude, the stress-dependent barrier, and the numerical safeguard. The 0.10, 0.25 and 0.50 eV margins are constructed readiness/sensitivity choices, not material calibrations or production defaults. Independent shared-stress admissibility remains an unresolved physical selection.

The retained 96-case Peak/DBTT figures and the reported 156%, 164%, and 32–35 MPa√m differences remain producer-bound historical-law/floor-dependent results. They are not predictions of the revised Taylor amplitudes, corrected completion mapping, fitted bilinear laws, or unresolved shared-stress choice. Their exact producer identities and frozen status remain in `LAW_TO_RESULT_MAP_V3.json`; no trajectory was rerun or relabeled as current-law output.

## Evidence boundary

`FULL_EVIDENCE_CURRENT_SCIENCE_AUDIT.json` verifies the current-science subset, but the full mandatory archive profile remains `INCOMPLETE`. The two missing archive objects are:

1. `/Users/shen/Fracture_Backups/fracture-completion-20261010-d1d15bf/final_release_20261011/final-fracture-evidence-20261011.tar` (expected SHA-256 `f0e57d81f8cd94dfe10a2fc0b0c458ce9cfdf5d16063aa1e578200dbc36ef63c`).
2. `/Users/shen/Fracture_Backups/common-framework-20261012/common-framework-supplemental-evidence-20261012.tar` (expected SHA-256 `0195e5e2c04e65b550fac9a96c0791aba5ca218f7d738ec0a3f037e3b08fae1e`).

Their absence is not converted into a passing full-release claim. Physical cohesive activation area/stiffness/work calibration, the production DBTT/shared-stress choice, R07 accepted-event completion, and R09 recursive-owner completion also remain explicit rather than being filled by fallback laws or constructed defaults.
