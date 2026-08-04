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
