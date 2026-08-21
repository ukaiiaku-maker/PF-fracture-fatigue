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

## 2026-08-20 v7 irreversible-baseline control

- Starting branch / HEAD: `codex/v9.14-v7-vhcf-tangent-guard` /
  `c914b535682d501cf6cc578a1cc389412739a117` (clean).
- Added `V7-IRREV`, which disables only returned-source-slip cancellation while
  retaining physical mobile return and every other current v7 constitutive and
  numerical channel. Exact-cycle and rate-separated VHCF launchers expose
  `--return-control REV|IRREV`; the model ID is part of checkpoint identity.
- Positive-R null: REV and IRREV result payloads are exactly equal for 12 exact
  cycles at R=0.1, Kmax=30; the reduced R=0 check is also exact. R=-0.5 return
  is negligible. R=-0.95 produces the expected small controlled difference.
- Historical canonical slope `m=71.45` is a mixed accelerated result: three
  completed points used periodic/frozen-state projection, one used DMD-only,
  and only the highest-load point was explicit in that driver. Later explicit
  controls establish finite life but not a monotone exact Paris region.
- Regression diagnosis: old runs used calibrated transferred-2-D common physics
  SHA `59ebf68...`; the current v7 audit used legacy-1-D common physics SHA
  `b5d7e54...`. At old exact DeltaK=14.6084, current v7-IRREV with the old input
  has H/cycle about 25.45 and a 1 um tip, while the current input produces no
  event through 1186 exact/DMD cycles and H=3.876e-9 against threshold 0.3754.
- Decision: stop the paired Paris/R/frequency campaign under the user's explicit
  regression stop condition. Selecting the authoritative common physics is an
  unresolved physics-contract choice because the historical file also disables
  mobile transport, while the current file changes source and shielding maps.
- Validation: all 93 `tests/test_v914_*.py` tests pass; Python syntax checks and
  `git diff --check` pass.
- Full provenance, comparison tables, and artifact paths:
  `V914_V7_RETURN_CONTROL_AUDIT.md`.

## 2026-08-21 fresh R=+0.1 restart decision

- Decisive virgin-state control: `CAL-TRANS-IRREV`,
  `DeltaK=15.24352267650291 MPa sqrt(m)`, `R=0.1`, `f=1000 Hz`, seed 1720,
  n=128, qualified v5 rate-separated DMD settings.
- Fresh IRREV and fresh REV are physically and numerically identical: both
  reach the `1e12`-cycle censor with 5 events, `16.036529362652427 um`
  extension, hazard action `0.31648762472927244` against threshold
  `2.797055987813155`, maximum accepted stride `274877906944`, and 951 map
  evaluations. Event records, anchor history, final gross activity, and final
  reversibility records are exactly equal.
- Decision A: the earlier `restart_count=1` IRREV growth result is invalid for
  scientific use. Restart/resume is not production-qualified and is treated as
  operationally unsupported. Restart debugging is deferred; the campaign uses
  fresh uninterrupted trajectories.
- Machine-readable decision:
  `artifacts/v914_calibrated_v7_transport_extension/fresh_irrev_rev_restart_decision.json`.
- Campaign continuation began with a fresh adaptive midpoint at
  `DeltaK=16.1962435 MPa sqrt(m)`.

## 2026-08-21 negative-R production blocker

- Three fresh `R=-0.5`, 1000 Hz REV bracket trajectories reached the 100 um
  target: DeltaK 25.405871, 26.464500, and 27.258384 MPa sqrt(m) completed at
  105455, 136944, and 60088 cycles, respectively, each with 15 events and
  102.284 um terminal extension.
- All three fresh `R=-0.95`, 1000 Hz REV bracket trajectories failed closed in
  the same place: an independently resolved exact-cycle validation of a
  speculative DMD midpoint projection encountered a sharp transport sign-change
  transient at the maximum adaptive refinement depth. The last accepted live
  checkpoints remain diagnostic-only; none will be resumed or credited.
- Numerical repair: exact accepted-state/training failures remain fatal, while
  the typed refinement failure from a speculative midpoint/endpoint projection
  now rejects and contracts that DMD block. This changes numerical accelerator
  control flow and diagnostics only; constitutive and fracture physics are
  unchanged.
- Regression and focused v9.14 gate: 110 tests passed. Python syntax and
  `git diff --check` pass. Corrected R=-0.95 production must be rerun fresh.
- The fresh R=+0.1 adaptive midpoint ended `work_budget_exhausted` at 41,203,461
  cycles, 5 events, and 16.0365 um; it is not a completed campaign point.
