# Fracture-framework acceptance consolidation — 16 October 2026

## Source identity and disposition

The integrated calculation and disabled-campaign implementation is PF `2dd1372fb45a4b4dd07d77ec5e85ee1bdab25f69`; its exact solved native FEM dependency is `7fde6894fe44038f9dc785da87967056a7d12406`. A later PF documentation-only closeout commit may contain this report; it does not change that calculation-source identity. Production and manuscript work remain disabled. No remote write or scientific-data deletion was performed.

The fixed portable source qualification passes: PF reports 1008 passed with 16 explicitly external-evidence tests deselected; FEM reports 759 passed plus 16 passing subtests at `7fde6894`. The exact commands and timings are in `FINAL_SOURCE_QUALIFICATION_20261016.json`.

## Controller findings and repair

The confirmed defect was an ownership mismatch in the moving-tip bilinear route. `MovingTipBilinearController.advance(front_id=...)` selected one active trial for acceptance but called constitutive helpers that iterated the entire cohesive network. Its rejection path restored only the selected elements and did not own callback mechanical, plastic or time state. A post-update equilibrium exception could therefore leave a partial state, and the returned equilibrium was not checked against an explicit constitutive splitting residual.

FEM `ee634968` repairs the transactional public/native chain, while `12971d37` exposes the specification's exact public names `bilinear_barrier_activated` and `bilinear_czm_classical` as compatibility-preserving mappings to those established internal modes:

`bilinear_cohesive_request_v1.build_moving_tip_controller` → `MovingTipBilinearController.begin_trial_segment` → `KineticTrialAdaptiveCZMBackend.begin_trial_segment` → native `fem.assemble_mechanics` → front-scoped constitutive update → returned native equilibrium → accepted commit.

The update domain is now exactly `ActiveTrialSegment.element_indices`. Each interval snapshots the full cohesive network, every active-owner record and log, controller counters, and callback-owned state. A damage-limit rejection, constitutive splitting rejection, post-update equilibrium failure, or commit-publication exception restores that whole interval automatically while retaining the active event. Explicit `rollback_active_trial` remains a separate whole-event cancellation that restores the pre-insertion topology and bulk-history baseline.

Autonomous bilinear damage evolves from opening history without a hazard gate. Its damage residual is evaluated at the returned equilibrium. Activated bilinear state carries a persistent event mark and action, accepted physical time, localized consumed time, and unused interval through active restart. The qualification uses an explicitly constructed unit mark and constructed cohesive support; these are not a production stochastic or material calibration.

The original EXP-floor route was not replaced. Its existing protection remains in `kinetic_progressive_2d_v1002` → `KineticEventLifecycle` → `KineticCohesiveStepper.advance` → `KineticTrialAdaptiveCZMBackend`, including mechanics snapshot/restore, full rollback callback, persistent event ownership, and unused-time carry. Nine focused stepper/lifecycle tests pass and are linked as preservation evidence.

## Native R05 evidence

The public generic route now drives `solved_moving_tip_bilinear_v2`, a compact equivalent of the original moving-tip specimen rather than the fixed-kinematic component qualifier. It creates the notched mesh, inserts the adaptive cohesive trial segment, performs sparse free-DOF equilibrium solves, calls native `plasticity.update_plasticity`, transfers state, loads/unloads/reloads, accounts for cohesive/external/plastic work, restores the whole attempted state on injected failure, commits after returned equilibrium, and reloads the active checkpoint exactly.

- Activated evidence: `integration_evidence/r05_solved_native_moving_tip_20261016/activated_v2/solved_moving_tip_bilinear_qualification.json`. It uses accepted physical time and persistent event ownership; the maximum restart errors in displacement, plastic strain and density are all zero. Measured compact-run peak RSS is `0.07498 GiB`, retained output before the report is `33,815 bytes`, and wall time is `0.1265 s`.
- Autonomous evidence: `integration_evidence/r05_solved_native_moving_tip_20261016/classical_v2/solved_moving_tip_bilinear_qualification.json`. It evolves from opening history with requested hazard duration and clock increments exactly zero; the same three restart errors are zero. Measured compact-run peak RSS is `0.07124 GiB`, retained output is `34,116 bytes`, and wall time is `0.1008 s`.
- Both show nonzero native plastic accommodation, adaptive topology insertion, irreversible unload/reload, post-event equilibrium, nonnegative cohesive work, terminal event commit and automatic whole-state rollback. The inputs remain explicitly constructed qualification support and do not claim production calibration.

## Disabled campaign and remaining native executions

The v3 campaign has 699 disabled rows and 699 source-bound native commands. Five selective P8 rows carry actual executable request fields for plane stress, generalized plane strain with prescribed `eps_zz`, generalized plane strain with prescribed `sigma_zz`, pure mode II, and mixed I–II. The 36 fitted-bilinear P9 rows use the exact public model IDs and dispatch through the generic companion to the solved native moving-tip driver with explicit constructed qualification inputs. This is an executable qualification binding, not a production material calibration; all production authorization flags remain false.

The accepted mixed/constraint EXP-floor topology event (R07) and recursive second-generation branch-owner transition (R09) were not launched. `RESOURCE_ADMISSION_20261016.json` separates measurements from estimates: four unrelated workers held about 8.01 GiB RSS in total, swap had 777.31 MiB free, immediately free pages represented about 86.56 MiB, and the Data volume had 30 GiB free. The 6 GiB R07 and 24 GiB R09 execution peaks and the 12 GiB reserve are engineering estimates, not measured peaks. Disk-only admission passes R07 but not R09; current RAM pressure admits neither. The existing disk-capacity preflight is not represented as a RAM check.

Both resumptions remain prepared exactly. R07 was dry-run validated on the preserved ancestor and remains present at FEM `7fde6894`; it executes only when `--execute` is supplied:

```bash
cd /Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion
python scripts/resume_r07_exp_floor_event.py --out <new-output> --steps 10000 --execute
```

R09's exact four-front command remains in `integration_evidence/r09_native_branch_a8620304/qualification_report.json` under `native_evidence.next_native_command`, bound to accepted V12 step 2184 and its retained checkpoint hash.

R10 contact and downstream child-continuation evidence remains linked to its current local-incremental producers in `integration_evidence/r10_current_local_incremental/consolidated_qualification.json`; no successful R10 calculation was rerun or replaced by historical collective-law evidence.

## Taylor law and historical-result boundary

The Taylor amplitude revision remains an authorized, real prospective parameter change. The records preserve separate parent and revised identities and distinguish raw thermal amplitude, the stress-dependent barrier, and the numerical safeguard. The 0.10, 0.25 and 0.50 eV margins are constructed readiness/sensitivity choices, not material calibrations or production defaults. Independent shared-stress admissibility remains an unresolved physical selection.

The retained 96-case Peak/DBTT figures and the reported 156%, 164%, and 32–35 MPa√m differences remain producer-bound historical-law/floor-dependent results. They are not predictions of the revised Taylor amplitudes, corrected completion mapping, fitted bilinear laws, or unresolved shared-stress choice. Their exact producer identities and frozen status remain in `LAW_TO_RESULT_MAP_V3.json`; no trajectory was rerun or relabeled as current-law output.

## Evidence boundary

`FULL_EVIDENCE_CURRENT_SCIENCE_AUDIT.json` verifies the current-science subset, but the full mandatory archive profile remains `INCOMPLETE`. The two missing archive objects are:

1. `/Users/shen/Fracture_Backups/fracture-completion-20261010-d1d15bf/final_release_20261011/final-fracture-evidence-20261011.tar` (expected SHA-256 `f0e57d81f8cd94dfe10a2fc0b0c458ce9cfdf5d16063aa1e578200dbc36ef63c`).
2. `/Users/shen/Fracture_Backups/common-framework-20261012/common-framework-supplemental-evidence-20261012.tar` (expected SHA-256 `0195e5e2c04e65b550fac9a96c0791aba5ca218f7d738ec0a3f037e3b08fae1e`).

Their absence is not converted into a passing full-release claim. Physical cohesive activation area/stiffness/work calibration, the production DBTT/shared-stress choice, R07 accepted-event completion, and R09 recursive-owner completion also remain explicit rather than being filled by fallback laws or constructed defaults.

## Cross-record acceptance

The existing live goal record now contains FG-01 through FG-18 and keeps code readiness, numerical qualification, scientific coverage, conclusion status, evidence references and author acceptance separate. The capability matrix has been brought forward from its stale specimen-only bilinear and pre-repair cavity classifications. The v3 builder, acceptance audit, status/block interface and legacy packaged campaign audit agree on their own scopes; the legacy-v2 handoff generator is isolated so it cannot overwrite the current records.
