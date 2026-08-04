# Codex Progress

- Repository: `/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30`
- Branch / baseline HEAD: `codex/v10.2.30-fatigue-da-dN` / `d0cdfd3d4078b8caed93fe5a1452e150a496736b`
- Environment: `arrhenius-sharp-front-v10-codex`
- Milestone: restart-safe four-class qualification supervisor implemented and smoke-qualified
- Commands run: supervisor/event tests; shell syntax checks; interrupted/resumed two-worker smoke; live monitor; SIGTERM stop smoke; four-class smoke analysis
- Results: 15 tests passed; maximum active workers observed=2; 12 interrupted cases resumed from step 2 to step 4 with restart_count=1; analyzer accepted all 12 smoke outputs; graceful stop preserved restartable checkpoints
- Commits created: `e692591` checkpoint synchronization; `5687b4d` da/dN provenance and campaign analyzer
- Watchdog / storage: progress-based 300 s default; SIGTERM then 30 s grace before SIGKILL; minimum free space 10 GiB; current free space approximately 21 GiB
- Matrix: peak/dbtt/weakT/ceramic at fractions 0.55, 0.75, 0.95; seeds 1720/1001723/2001726/3001729; maximum two jobs
- Qualification commands: run `FAMILY_JSON=/absolute/family.json bash scripts/run_v10_2_30_four_class_qualification_supervisor.sh`; monitor `bash scripts/monitor_v10_2_30_four_class_qualification.sh`; stop `bash scripts/stop_v10_2_30_four_class_qualification.sh`
- Blocker: production post-event restart remains fail-closed because the 2-D driver does not yet restore matching outer crack geometry; smoke workers prove supervisor resume selection but full qualification must not claim checkpoint resume support yet
- Next action: commit wrapper milestone; do not launch the full 12-case qualification until outer-geometry checkpoint restore is implemented and validated

## 2026-08-04 restart-production equivalence

- Branch / simulation HEAD: `codex/v10.2.30-fatigue-da-dN` / `24b63a5bfd86a8ea249d457750b14b8c19488973`.
- First event-2 divergence: checkpoint restore recomputed `mesh.hbar_tip` around the moved crack tip (`3.0827858824767497e-6 m`) instead of retaining the initial refinement anchor (`1.384579911336366e-6 m`); classification A, selector input differed before scoring.
- Repair: combined checkpoint schema v2 stores authoritative `hbar`, `hbar_tip`, and tip-reference centers; restore validates their exact reconstruction and preserves connectivity dtype. Direction selection consumes no RNG and its fixed alpha-grid ordering is deterministic.
- Focused replay: `runs/v10_2_30_event2_replay_{control,restored}_24b63a5`; event-2 proposal `2.2973400956248734 um`; forensic comparator exit 0 with identical inputs, candidates, scores, winner, direction, endpoint, and commit.
- Long qualification: `runs/v10_2_30_restart_equivalence_24b63a5`; uninterrupted control and supervisor-resumed trajectory both reached 5 events, `658699.784846286` cycles, and `26.37911007410791 um` projected extension.
- Production verifier: exit 0; event-2, terminal outer state, terminal kinetic state, and every checkpoint array identical. NaN pairs and explicitly nonphysical wall-time/cache-residual telemetry are canonicalized; physical fields remain fail-closed.
- Focused gate: 33 tests passed. The 12-case qualification was not launched.
