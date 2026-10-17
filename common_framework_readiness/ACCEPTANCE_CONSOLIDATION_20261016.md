# Fracture-framework acceptance consolidation — 16 October 2026

## Source identity and disposition

The disabled-campaign implementation is PF `1ba0e500a563e839a9ec3fc16264488b80e87755`; its qualified native FEM dependency is calculation commit `917d1a4491ec0abf2f054bac5fe5d1422e1ee781` with evidence commit `f8eba83`. These advance, rather than reset, the earlier PF/FEM checkpoints. A later PF documentation-only closeout commit may contain this report; it does not change either calculation-source identity. Production and manuscript work remain disabled. No remote write or scientific-data deletion was performed.

The current portable PF source qualification remains 1008 passed with 16 explicitly external-evidence tests deselected; the post-binding PF campaign/execution subset passes 31 tests. The post-repair FEM tree reports 758 passed, 2 skipped and 16 passing subtests, including the new real-public-interface regression. The exact commands and timings are in `FINAL_SOURCE_QUALIFICATION_20261016.json`.

## Controller findings and repair

The confirmed defect was an ownership mismatch in the moving-tip bilinear route. `MovingTipBilinearController.advance(front_id=...)` selected one active trial for acceptance but called constitutive helpers that iterated the entire cohesive network. Its rejection path restored only the selected elements and did not own callback mechanical, plastic or time state. A post-update equilibrium exception could therefore leave a partial state, and the returned equilibrium was not checked against an explicit constitutive splitting residual.

FEM `ee634968` repairs the transactional public/native chain, while `12971d37` exposes the specification's exact public names `bilinear_barrier_activated` and `bilinear_czm_classical` as compatibility-preserving mappings to those established internal modes:

`bilinear_cohesive_request_v1.build_moving_tip_controller` → `MovingTipBilinearController.begin_trial_segment` → `KineticTrialAdaptiveCZMBackend.begin_trial_segment` → native `fem.assemble_mechanics` → front-scoped constitutive update → returned native equilibrium → accepted commit.

The update domain is now exactly `ActiveTrialSegment.element_indices`. Each interval snapshots the full cohesive network, every active-owner record and log, controller counters, and callback-owned state. A damage-limit rejection, constitutive splitting rejection, post-update equilibrium failure, or commit-publication exception restores that whole interval automatically while retaining the active event. Explicit `rollback_active_trial` remains a separate whole-event cancellation that restores the pre-insertion topology and bulk-history baseline.

Autonomous bilinear damage evolves from opening history without a hazard gate. Its damage residual is evaluated at the returned equilibrium. Activated bilinear state carries a persistent event mark and action, accepted physical time, localized consumed time, and unused interval through active restart. The qualification uses an explicitly constructed unit mark and constructed cohesive support; these are not a production stochastic or material calibration.

The original EXP-floor route was not replaced. Its existing protection remains in `kinetic_progressive_2d_v1002` → `KineticEventLifecycle` → `KineticCohesiveStepper.advance` → `KineticTrialAdaptiveCZMBackend`, including mechanics snapshot/restore, full rollback callback, persistent event ownership, and unused-time carry. Nine focused stepper/lifecycle tests pass and are linked as preservation evidence.

## Native R05 evidence

The disconnected-constructor repair is preserved, and FEM `917d1a4` completes the remaining constitutive connection. The actual backend is a per-instance extension of the current stochastic-length/atomic-corridor backend; it inserts an active trial interface before equilibrium and supplies that cohesive network to native `assemble_mechanics`. The interface residual and tangent therefore affect the solved displacement, traction, J/K state and subsequent constitutive update. Autonomous damage is an irreversible opening-history continuation with bounded damage correctors and no hazard gate. Activated damage consumes one accepted physical-time clock driven by the assembled interface traction while the inherited EXP-floor opening clock remains suppressed. A topology commit is published once, after accepted equilibrium, and is not reinserted by the outer geometry owner.

Public runtime acceptance is now discriminating: provider construction and advance counters are insufficient. The selected request passes only when the provider is also bound to cohesive assembly and owns an actual trial interface (or restores that interface through the public checkpoint path). The active interface, provider state, front state, mesh, bulk fields and mechanics rollback baseline are serialized through the original specimen owner.

The compact source-bound record is `fem-czm-current-pf-companion@f8eba83:integration_evidence/r05_actual_solved_interface_20261016/qualification_report.json`, produced by calculation source `917d1a4`:

- Classical direct: eight accepted source/time updates, 27 mechanical solves, and one 1 µm native event. The event step required 13 equilibrium solves, had maximum damage subincrement `0.1277899931 <= 0.20`, zero final constitutive residual, and changed interface work from `2.951726786e-6 J/m` to zero on commit.
- Classical active restart: step 7 stores nonzero damage `0.05237946396`; a fresh process restores the active interface and commits at step 8. Direct and restarted terminal histories are byte-identical (SHA-256 `e36eda97e590b390e88e8dbc155e94ee8acbc3c488d19e6b34cfe8f666988008`).
- Activated direct: 47 accepted source/time updates, 94 mechanical solves, and one native event. The final interval is localized from `3.818760853e-11 s` requested to `1.855529613e-11 s` consumed at assembled interface traction `7.142501651 GPa`; maximum clock/damage subincrement is `0.04890875044`, and work changes from `1.275383246e-6 J/m` to zero on commit.
- The source-bound public replay passes without mocking native main, front, barrier, source or mechanics. The complete FEM suite passes with the progressive/fatigue source-transform chain preserved.

The older `r05_solved_native_moving_tip_20261016/{activated_v2,classical_v2}` records remain historical component evidence only. Their transaction and serialization checks unaffected by the known defects may be retained, but their accommodation and work values no longer qualify R05 because that driver advanced plasticity by one second per equilibrium callback and constructed the wrong Peierls/Taylor barriers. They are not promoted by relabeling. The replacement inputs remain explicitly constructed qualification support and do not claim production calibration.

## Disabled campaign and remaining native executions

The v3 campaign has 699 disabled rows and 699 source-bound native commands. Five selective P8 rows carry actual executable request fields for plane stress, generalized plane strain with prescribed `eps_zz`, generalized plane strain with prescribed `sigma_zz`, pure mode II, and mixed I–II. The 36 fitted-bilinear P9 rows use the exact public model IDs and dispatch through the generic companion to the actual atomic-path-corridor driver and current front provider. Their binding explicitly records replacement of the inherited EXP-floor clock and preservation of the persistent source owner. This is an executable qualification binding, not a production material calibration; all production authorization flags remain false.

The accepted mixed/constraint EXP-floor topology event (R07) and recursive second-generation branch-owner transition (R09) were not launched. The latest measurements show about 50.34 MiB immediately free, 873.31 MiB swap free, 28 GiB free on the Data volume, and five unrelated processes at about 1.36–2.65 GiB RSS each. The 6 GiB R07 and 24 GiB R09 execution peaks and the 12 GiB reserve remain engineering estimates, not measured peaks. Disk-only admission passes R07 but not R09; current RAM pressure admits neither.

Both resumptions remain prepared exactly. R07 was dry-run validated on the preserved ancestor and remains present on the current FEM line; it executes only when `--execute` is supplied:

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
