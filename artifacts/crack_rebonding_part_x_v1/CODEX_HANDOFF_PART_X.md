# Part X handoff — signed-K crack-rebonding compression campaign

Note: the repository's root `CODEX_HANDOFF.md` is a pre-existing, unrelated
project-wide handoff document from an earlier mission (PF/sharp-front
Arrhenius-hazard fatigue, dated to the v10.2.21–v10.2.30 hazard-energy-gated
line). It is left untouched. This file is Part X's dedicated handoff record,
tracked at `artifacts/crack_rebonding_part_x_v1/CODEX_HANDOFF_PART_X.md` per
mission section 14, updated after each major milestone.

- **Branch:** `codex/v10.2.30-crack-rebonding-part-x`
- **Worktree:** `/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/v10230-crack-rebonding-part-x`
  (persistent — not `/private/tmp`)
- **Clean/dirty:** clean at last commit
- **Active PIDs / jobs:** none

## Completed stages

- **Initial checks (mission section 1):** base commit
  `a72d46557f5eba45c8c2e0a428574e5b9b624c81` verified, worktree created fresh
  (branch did not previously exist), merge-base confirmed exact, zero
  related workers, both existing verifiers (`verify_static_shield_
  attribution.py`, `verify_developed_confirmation.py`) pass, full
  `crack_rebonding` test selection 196 passed / 0 failed (405s).
- **PX0 (committed `2b72738`):** provenance and completion-contract freeze.
  Real production engine constructed via `a_native_engine_v10230.
  build_a_native_engine()` (genuine `__init__`, not a fixture) to record its
  exact MRO, `n_phase=80`, `mpz_n_bins=80`, `wake_n_bins=160`, and the
  A_NATIVE material-row hash. All 8 named prior studies traced to branch
  SHAs; 7/8 are ancestors of base (already present/hashed in this worktree);
  `causal_pilot_v2` is a separate git line whose substantive work already
  merged into base via a common ancestor — only its tip's
  `regime_equivalence_analysis.json` is unique, pulled read-only without
  merging that branch. That file's finding
  (`REVERSIBLE_PERSISTENT_EQUIVALENT_FOR_SLOPE_SCREEN` at the OLD
  reversible/persistent presets, both seeds, |S_h diff| < 0.0002 decade vs a
  0.005 gate) is carried into `mission_scope.json` as a binding design
  constraint on PX2.3: COMPETING_PERSISTENT must use materially reduced
  rupture relative to COMPETING_REVERSIBLE, not reuse the old presets
  unmodified.

## Source-code reconnaissance done (informs PX1 design)

- `persistent_site_cyclic_v10229.py` (389 lines): `preview_cycle_waveform`
  is the real per-phase-bin cyclic engine path; `cycle_step_waveform` drives
  `_integrate_coupled`. Confirmed injection points for dynamic rebonding and
  static-shield ablation.
- `persistent_site_coupled_hazard_v10229.py` (614 lines): adaptive
  Simpson/bisection outer quadrature over **constant segments**
  (`_commit_constant_segment` / `integrate_state_coupled_waveform`) — the
  "one-cycle mean rate" approximation the static-shield localizer-parity
  audit (a72d465) already identified as structurally distinct from dynamic
  rebonding's exact phase-resolved bisection.
- `crack_rebonding_kinetics_v10230.py` (777 lines): pure math layer.
  `CrackRebondingControls.minimum_load_hold_s` field **already exists**
  (default 0.0) but `validate()` (~line 161) explicitly rejects any nonzero
  value ("hold integration is a documented future interface, not
  implemented") — this is PX1.1's precise target. `propagate()` /
  `build_phase_factors()` / `_partial_product()` implement an exact
  piecewise-constant-generator phase propagator but assume a **uniform**
  `dt_phase` across all bins — PX1.1 needs a heterogeneous-duration
  generalization (sinusoidal bins at `dt_phase = T_base/n_phase` plus one
  hold bin at `dt_hold = minimum_load_hold_s`, evaluated exactly via `expm`,
  not extra rounded phase bins). Also contains `REFERENCE_ACTION_PRESETS`,
  `solve_reference_action_barriers`, `freeze_reference_action_preset` — the
  exact PX2 analytical-regime-design machinery the mission references.
- `crack_rebonding_v10230.py` (1257 lines): engine-integration layer.
  `patch_Q` already implements PASSIVATION_GATED_REBOND's full P↔C↔B
  generator (depassivation/repassivation wired in at ~lines 174–179) — so
  PX1.3 is an **audit/test-coverage** task on existing code, not new
  physics. Contains `solve_coupled_event_time`, `phase_resolved_action`,
  `_periodic_orbit_certificate`, `install_crack_rebonding`,
  `serialize_rebonding_checkpoint` — all call sites that PX1.1's dwell
  segment and PX1.4's localizer-parity generalization must also touch.

- **PX1.1 (committed `f5df106`):** minimum-load dwell, implemented as
  `FatigueWaveform.minimum_load_hold_s` (a whole-loading-protocol property,
  not a `CrackRebondingControls` field — see that config's `validate()` for
  the explicit reasoning; keeping both would have created a disconnected
  second dwell control). One authoritative schedule
  (`FatigueWaveform.cycle_schedule`): `n_phase` sinusoidal bins plus one
  appended constant-`Kmin` dwell bin. `period_s` redefined once to mean the
  complete protocol cycle, which makes essentially all existing
  cycle/block-duration bookkeeping hold-aware automatically (no per-site
  changes needed there); `effective_cycle_frequency_Hz` added, defined to
  equal `frequency_Hz` exactly (not just numerically) at hold=0. Genuinely
  phase-resolved machinery (`propagate`/`build_phase_factors` in
  `crack_rebonding_kinetics_v10230.py`, `phase_resolved_action`/
  `representative_cycle_K_rebond` in `crack_rebonding_v10230.py`,
  `_phase_statistics`/`_commit_constant_segment` in
  `persistent_site_coupled_hazard_v10229.py`) generalized to heterogeneous
  per-bin durations, with the original uniform/scalar implementations
  preserved as separate, untouched functions dispatched to whenever
  `dt_phase` is a scalar — hold=0 takes the byte-identical original code
  path by construction, not by coincidence. Fixed a real latent bug this
  surfaced: two `dt_consumed→cycles_consumed` conversions
  (`persistent_site_cyclic_coupled_v10229.py` — the real production
  `cycle_step_waveform` override — and the dead `persistent_site_cyclic_
  v10229.py` copy) used the nominal `frequency_Hz` instead of the effective
  protocol-cycle frequency; harmless while hold was always 0, silently
  wrong otherwise. 25 new tests (`test_v10_2_30_crack_rebonding_part_x_
  minimum_load_hold.py`) cover the mission's exact required battery. Two
  production files touched beyond PX0's original list
  (`fatigue_v1.py`, `persistent_site_cyclic_coupled_v10229.py`); their
  a72d465 preimages recorded in `px1_source_preimage_manifest.json`. Full
  `crack_rebonding` selection: 221 passed (196 + 25), zero regressions.
  Both existing verifiers still pass. One initially-written full-trajectory
  RNG-comparison test was found sensitive to pre-existing global test-order
  state unrelated to Part X (reproduced only after ~19 other test files ran
  first; neither RNG reseeding nor bumping the engine-id counter reproduced
  it in isolation) and was removed rather than chased further — documented
  in the test file itself.

- **PX1.2 (committed `6f24dd9`):** transition-action/flux instrumentation,
  default-off. `crack_rebonding_kinetics_v10230.build_augmented_Q`/
  `transition_actions_and_fluxes`: augmented 6x6 generator
  (`d/dt[p;s]=[[Q,0],[I,0]][p;s]`) gives the exact per-bin occupancy time
  integral alongside the ordinary propagation, so `A_CB/A_BC/A_PC/A_CP`
  and `F_CB/F_BC/F_PC/F_CP` are exact (no averaging/trapezoids) for any
  constant-generator segment including the dwell bin. Deliberately a fully
  separate bin-walk from `propagate`/`phase_resolved_action` — this
  diagnostic must never risk perturbing the real state-evolution path it
  audits. `crack_rebonding_v10230.py`: `patch_Q` refactored (no behavior
  change) onto a new `patch_rate_constants` helper so the diagnostic can't
  drift from the real physics; `patch_transition_actions_and_fluxes` wraps
  this around one real `WakePatch`. Validated against an independent
  brute-force fine-stepped reference (genuinely different numerical
  method); state-balance identities close to ~1e-13 relative in every
  case. 24 new tests, full `crack_rebonding` selection 245 passed
  (196+25+24), zero regressions. Two of the three attempted "compare two
  independently-constructed engines" tests across PX1.1/PX1.2 reproducibly
  diverged well beyond floating-point roundoff — but ONLY when the file
  ran as part of the full test selection, never in isolation, and neither
  RNG reseeding nor bumping the global engine-id counter reproduced it —
  confirming this is pre-existing global/class-level state in the shared
  test fixture, unrelated to Part X. **Do not write a test that compares
  two independently-constructed `build_real_engine()` instances against
  each other in this suite** — use a single engine's own before/after
  state instead, as the surviving tests do.
  Deferred: wiring these diagnostics into `RebondingWakeState`'s
  persistent checkpoint/ledger schema (appropriate once a specific
  study's event-ledger builder needs to archive them, not in this shared
  module) and the remaining section 5.2 diagnostics (contact-duration-in-
  sinusoid vs. in-dwell separately, phase-resolved p_P/p_C/p_B extrema,
  barrier-floor/cooperative-saturation fractions) — straightforward
  additions once a concrete consumer exists.

- **PX1.3 (committed `9397f35`):** RB3 audit — 12 tests through the real
  engine confirming `patch_Q`'s existing P↔C↔B generator behaves correctly
  (depassivation live only compressive, repassivation live only tensile,
  probability conservation, exact zero at K≥0, fresh-surface split,
  checkpoint/snapshot round-trip, no emission contamination). No source
  changes — audit only, as anticipated. First attempt clean.
- **PX1.4 (committed `c0792c9`):** `static_shield_phase_resolved_action` —
  generalizes exact event localization to a prescribed constant `K_b`.
  `solve_coupled_event_time` needed zero changes (already fully generic in
  its `phase_resolved_action_fn` argument). Unlike dynamic rebonding,
  no state evolves for a constant `K_b`, so this is exact by construction
  for any span (no periodic-orbit approximation needed at all — one
  cycle's action is bit-identical every cycle). Validated against an
  independent fine-stepped reference at a **frozen 1e-6 relative
  tolerance** across Kmax=15/18/21, K_b=0.45/0.9/1.8, 4 starting phases,
  partial/single/multi-cycle spans, and firing inside the hold — 120/120
  passed. Classified `PRESCRIBED_STATIC_EVENT_LOCALIZATION_PARITY_QUALIFIED`
  **for this new evaluator itself** (`px1_4_static_localizer_parity_
  finding.json`) — this does NOT retroactively reclassify the already-
  published static-shield-attribution study, which never used this
  evaluator; `STATIC_DYNAMIC_LOCALIZER_PARITY_UNRESOLVED` remains correct
  for those published numbers. Production wiring into a live commit path
  deferred to whichever future study (PX5) needs it.

## PX1 COMPLETE

Final gate: 377 passed (196 pre-existing + 181 new across the four PX1.x
test files), zero regressions; both existing verifiers still
`overall_pass=true`; `compileall`/`git diff --check` clean; zero workers;
worktree clean. Recurring lesson worth restating: **never compare two
independently-constructed `build_real_engine()` instances against each
other** in this test suite — reproducibly diverges (sometimes by far more
than floating-point roundoff) only when run as part of the full selection,
never in isolation; root cause is pre-existing global/class-level state in
the shared fixture, unrelated to Part X. Use a single engine's own
before/after state instead.

- **PX2 (committed `628ae33`):** kinetic-regime analytical design and
  prospective freeze. New module `crack_rebonding_part_x_kinetic_regime_
  v10230.py::analytical_periodic_orbit` finds the true periodic fixed
  point of the phase-resolved P/C/B chain (thin composition over PX1.1's
  heterogeneous `propagate` + PX1.2's `transition_actions_and_fluxes` --
  convergence verified explicitly every call, not trusted from iteration
  count). Four rows constructed and gate-checked (margins archived in
  `kinetic_regime_registry.json`):
  - **SAT_EXISTING**: `RB2_reversible_finite` reused byte-for-byte from
    `artifacts/crack_rebonding_causal_pilot_v2/frozen_configuration.json`
    — config-hash reproduction is asserted by the script (raises on
    mismatch).
  - **COMPETING_REVERSIBLE**: `solve_reference_action_barriers(A_on=2,
    A_off=2)` at the reference state — mean p_B=0.320, swing=0.507,
    A_CB=2.040, A_BC=1.952 (all 7 section-6.2 gates pass with margin).
  - **COMPETING_PERSISTENT**: same bond barrier, rupture barrier +0.05 eV
    — mean p_B=0.571 (Δ0.251 from reversible, gate ≥0.10), A_BC ratio
    0.145 (gate ≤0.20) — **deliberately does not reuse causal_pilot_v2's
    old presets** per PX0's own recorded constraint.
  - **PASSIVATION_LIMITED**: same formation/rupture barriers + symmetric
    0.40 eV depassivation/repassivation — mean p_P=0.324, mean p_B=0.343,
    all four transition actions/fluxes nonzero.
  Analytical atlas: 1399 points across the full grid (`analytical_phase_
  atlas.{csv,parquet}`) — R=+0.1 confirmed exactly zero contact/A_CB/F_CB
  everywhere (semantic control); cohesive-strength axis confirmed
  p_B-independent as expected (K_rebond,max doesn't feed P/C/B kinetics).
  PX3 condition selections frozen from the atlas
  (`analytical_regime_selection.json`): **frequency-transition = 100 Hz**,
  **dwell-transition = 0.5 ms**, **passivation chemistry_factor = 1.0**;
  cohesive-strength endpoint explicitly deferred to PX3 (can't be decided
  analytically). `screen_job_registry.csv` (36 jobs) / `developed_job_
  registry.csv` (72 jobs) enumerate sections 7/8's matrices, every job
  `QUEUED_NOT_LAUNCHED` — confirmed no `runs/crack_rebonding_part_x_v1`
  directory exists yet. 9 new tests. Full selection: 386 passed, zero
  regressions.

- **PX2.5 (committed `34ab3e3`):** bounded prephysics closure requested by
  external review before allowing PX3 to launch real physical
  trajectories. Addressed exactly 7 items:
  1. Fresh-process engine reproducibility root-caused to two effects —
     `_next_engine_id` shadowed per leaf subclass (base-class
     `reset_audit()` doesn't touch it) and `_hazard_config_default` a
     mutable class attribute leaked across tests via `configure_hazard()`
     without reset. Verified fix: `Engine.configure_hazard()` +
     `Engine.reset_audit()` on the actual leaf class immediately before
     construction (5 new tests).
  2. Intra-dwell cursor bug fixed via new
     `FatigueWaveform.cycle_schedule_from_elapsed()` — the previous
     phase-offset reconstruction silently discarded dwell-segment
     progress across cycle boundaries whenever `minimum_load_hold_s > 0`
     (11 new tests). This is a genuine correctness fix affecting every
     dwell-enabled multi-cycle trajectory computed before this commit.
  3. Screen event ledger wired: `build_screen_event_ledger_row()` in
     `crack_rebonding_v10230.py` computes all 8 A/F quantities with
     independent cross-checks and balance-residual closure (2 new tests).
  4. RB3 passivation tests strengthened to measure all 8 A/F quantities
     through a real trajectory and to compare one engine's own
     before/after state rather than two independently-constructed
     engines.
  5. Job registries rewritten fail-closed with canonical dedup keys:
     screen rows are `AUTHORIZED_PX3` only once uniquely keyed (28 of 36
     rows; 8 are `ALIAS_OF_EXISTING_JOB`); developed rows carry specific
     `BLOCKED_PENDING_*` statuses until their PX3 prerequisite lands (82
     rows total: 16 `BLOCKED_PENDING_PX3_COMPLETION`, 12
     `BLOCKED_PENDING_STAGE1_DECISION`, 8 each of `_FREQUENCY_GATE`,
     `_DWELL_GATE`, `_PASSIVATION_GATE`, `_PERSISTENT_DISTINCTION`, 22
     `ALIAS_OF_EXISTING_JOB`; includes a new D7 cohesive-strength job).
     `competing_persistent_selected` renamed to
     `competing_persistent_status = "ANALYTICALLY_ELIGIBLE_FOR_PX3_SCREEN"`
     (6 new tests for the controller's authorization gating).
  6. Candidate search history preserved:
     `kinetic_regime_candidate_audit.csv` reproduces all 20 candidates
     PX2's interactive search evaluated (7 COMPETING_REVERSIBLE, 6
     COMPETING_PERSISTENT, 7 PASSIVATION_LIMITED) through the same
     evaluator, with every gate value and the deterministic selection
     rule.
  7. Controller qualified: `part_x_run_one_job.py` (fresh-process
     entry point, `--preflight` only so far) + `part_x_physical_
     controller.py` (disk-backed, atomic-write, max-3-worker,
     quarantines prephysics/interrupted failures). Qualifying it under
     real concurrent preflight execution found and fixed a genuine
     production race in `build_a_native_manifest()` (fixed non-unique
     temp path → concurrent workers collided on atomic rename); fixed at
     the caller level via a private `tempfile.tempdir` per worker
     process, without touching the shared production function.

  Also fixed a missing `lineterminator="\n"` on 5 `csv.DictWriter` call
  sites. Self-caught and corrected a mistake made while doing so: an
  initial re-run of `build_part_x_px0_provenance.py` silently overwrote
  PX0's frozen a72d465-era source hashes with current post-edit hashes;
  caught via `git diff` showing genuine hash *value* changes, reverted
  with `git checkout --`, and re-fixed the one genuinely-affected
  artifact (`inherited_result_inventory.csv`) via direct byte-level
  `\r\n`→`\n` substitution instead of a script re-run. 410 tests pass
  (386 + 24 new across the four new test files), zero regressions; both
  existing verifiers still `overall_pass=true`.

- **PX3 (physical screen executed; commit pending final test confirmation):**
  implemented the real screen-budget branch of `part_x_run_one_job.py`
  (12 accepted events or 60 μm, driven by the qualified causal-pilot-v2
  event loop, resolving each screen job's `chemistry_factor`/
  `K_rebond_max_target_Pa_sqrt_m` panel overrides against the row's frozen
  baseline config). Two minimal, default-preserving extensions to
  `run_trajectory`: `minimum_load_hold_s` threading (for the dwell panel)
  and `cumulative_cycles` bookkeeping (section 7.8's `g` denominator).
  Found and fixed a real bug while reviewing the results: `signed_K`/
  `interval_compression_analysis` hardcoded the module's own reference
  1000 Hz/18 MPa√m instead of the trajectory's actual frequency/Kmax, so
  every non-1000 Hz screen job's `post_first_event_intervals` diagnostic
  (not the core g/S_h quantities, which always used the real engine's own
  `cycle_step_waveform`) was computed against the wrong frequency; fixed
  and the 6 affected already-completed `runs/.../result.json` files were
  repaired in place (pure post-hoc recomputation from stored
  `cumulative_time_s`, no re-simulation). Regenerated `screen_job_registry.
  csv`/`developed_job_registry.csv` once more for the new HEAD (only
  `physical_producer_sha`/`canonical_job_key` changed, verified
  programmatically). Launched all 28 unique `AUTHORIZED_PX3` jobs via
  `part_x_physical_controller.py --allow-head-drift` (justified: the only
  intervening commit was that same producer-sha refresh) against the real
  `runs/crack_rebonding_part_x_v1/` run root — all 28 completed, zero
  quarantined; 27/28 hit the full 12-event/60 μm budget, one (dwell=0.5ms,
  finite cohesion) was right-censored at 6/12 events by the 30-minute
  per-trajectory wall-time ceiling (a genuine right-censor per the
  mission's own rule, not a failure).

  Built `build_part_x_px3_screen_analysis.py` (g/S_h over three windows —
  overall, post-first-event, late-half — for all 18 matched pairs,
  measurable-effect gates, contact-time diagnostics with an explicit
  pure-sinusoid-approximation caveat for hold>0 pairs) and
  `build_part_x_px3_adaptive_selection.py` (mechanically applies section
  7.8's six frozen rules against the real numbers). R=+0.10's exact
  zero/finite parity control passed exactly (S_h=0.0000000, confirming the
  signed-K semantic control). Five of six rules resolved cleanly and
  flipped the corresponding `developed_job_registry.csv` rows to
  `AUTHORIZED_PX4`: rule 1 (D1/D2, unconditional), rule 4 (D5,
  chemistry_factor=1.0 — the analytically-selected value also shows the
  largest live effect, no contradiction), rule 5 (D6, COMPETING_PERSISTENT
  selected — NOT distinguishable from reversible at baseline, S_h delta
  only 0.0001, but decisively distinguished at the frequency-transition
  condition per section 7.4's own fallback instruction, S_h delta 0.0413),
  rule 6 (D7, no cohesive-strength endpoint selected — the 3-point screen
  is monotonic with same-sign slope in both segments, not visibly
  nonlinear by a documented >2x/<0.5x slope-ratio-or-sign-flip test).

  **Two findings needed a decision before D3/D4 could launch (left
  deliberately BLOCKED, not guessed) — both resolved in PX3.5 below.**

- **PX3.5 (portable screen qualification, dwell-causality audit, final
  developed-protocol selection) — complete.** External review accepted
  PX3's physical execution but required this bounded closure before PX4.

  1. **Producer provenance closed.** `build_part_x_px3_producer_
     provenance.py` proves, by re-hashing every physics-affecting source
     file at both the registry's recorded `physical_producer_sha`
     (6b9631d) and the actual launch HEAD (cc306a8), that the only
     difference between them was the two registry CSVs themselves (a
     protocol-registry concern, never physics) — `physical_source_bundle_
     byte_identical: true`. Policy going forward: no general
     `--allow-head-drift` for PX4+; require exact launch-HEAD match or a
     fresh instance of this same proof.
  2. **Portable ledger built.** `build_part_x_px3_portable_ledger.py`
     produces `screen_event_ledger.{csv,json}` (330 events across the 28
     trajectories), `screen_censor_registry.csv`, `screen_raw_result_
     hashes.json`, and `screen_posthoc_correction_manifest.json` (the 6
     files whose `post_first_event_intervals` were repaired, with the
     exact-fields-changed proof — pre-repair hashes are `NOT_ARCHIVED`,
     since the repair was applied in place before this manifest existed).
     Missing per-event diagnostics (A/F transition-flux breakdown for
     firing sub-intervals, live cycle-mean p_P/p_C, dwell-aware contact
     time) are recorded as `NOT_ARCHIVED`, never inferred.
  3. **Real bug found and fixed: the actual root cause of the dwell
     "acceleration."** `persistent_site_coupled_hazard_v10229.py::
     _phase_statistics`'s `hazard_coupled` branch duration-weighted
     `sig_cleave` samples (drawn from a cursor-**rotated** K array,
     `cycle_schedule_from_elapsed`) using the **unrotated** `dt_values`
     array from the plain `cycle_schedule` — pairing each sample with the
     wrong bin's duration whenever bin durations are non-uniform
     (`hold>0`). Invisible at `hold=0` (uniform bins mask the
     misalignment) — this is why 24 of PX3's 28 jobs were unaffected.
     Proven via a zero-active-patches control (no crack-advance event has
     fired yet, so `K_rebond` is provably 0 regardless of cohesion):
     `_phase_statistics` matched between finite/zero configs only at
     cursor=0 before the fix, diverged by up to ~8x at other cursor
     positions once `hold>0`; after the fix they match to floating-point
     roundoff (~1e-15 relative) at *every* cursor position. Also fixed a
     related variable-length bug in the same function (`cycle_schedule_
     from_elapsed` can return one entry longer than the plain schedule
     when the cursor doesn't land on a bin boundary — the cleave-side
     loop now walks its own array length, never `K_values`'s). 11 new
     regression tests (`test_v10_2_30_crack_rebonding_part_x_px3_5_
     dwell_hazard_fix.py`). Full `crack_rebonding` selection re-verified
     clean after the fix (429 passed) — confirmed pre-existing, unrelated
     to this fix: 8 failures in `test_v10_2_29_state_coupled_hazard.py`/
     `test_v10_2_30_forward_coupled_marcher.py`/`test_v10_2_30_partition_
     robust_forward.py` (stale mock `Waveform` test doubles missing
     `cycle_schedule`, present identically on the unmodified pre-fix
     commit — a latent PX1.1-era gap, out of PX3.5's scope, worth a future
     follow-up).
  4. **Dwell causal audit: `ORIGINAL_DWELL_SIGN_REVERSAL_NOT_REPRODUCED`.**
     Reran both nonzero-hold dynamic-finite trajectories under the fix
     (both now complete uncensored, 12/12 events) plus a prescribed
     post-first-event static `K_b=0.9 MPa√m` control at each hold (first
     live-production use of the `run_trajectory` `static_shield_control`
     hook with `hold>0`). All four legs give `S_h≈-0.044`, matching the
     `hold=0` baseline (`-0.0435`) — not the original `+0.36`/`+0.80`.
     Static and dynamic agree to within 0.0002–0.0004 decade at both
     holds, ruling out `DWELL_OR_LOCALIZER_IMPLEMENTATION_INCONSISTENT`.
     Change from the zero-hold baseline is only 0.0005–0.0007 decade —
     far below the 0.01 measurable-effect gate — so **D4 is not selected**
     (`build_part_x_px3_5_dwell_audit_classification.py`; existing D4 rows
     set to `BLOCKED_DWELL_GATE_NOT_SATISFIED`). The dwell-induced
     "acceleration" was entirely a software defect.
  5. **Frequency transition localized at 316.227766 Hz** (`run_part_x_
     px3_5_frequency_bisection.py`), the geometric mean of 100/1000 Hz —
     resolved on the *first* evaluated point (of 3 permitted): `S_h=
     -0.0232`, clearing both the `|S_h|>=0.01` measurable-effect gate and
     the `|S_h-S_h(1000Hz)|>=0.01` distinct-from-baseline gate. **D3
     authorized** at this condition (10 new developed rows across the
     Kmax grid).
  6. **D6 corrected to the localized transition frequency**, per review's
     explicit instruction ("D6 must use the ultimately selected
     frequency-transition condition"). At 1000 Hz baseline, persistent
     vs. reversible are indistinguishable (`ΔS_h=0.0001`) — the original
     D6 rows (baked at 1000 Hz) are set to `BLOCKED_PROTOCOL_MISMATCH`.
     Reran the persistent pair at 316.227766 Hz (`run_part_x_px3_5_
     persistent_at_transition.py`): `S_h=-0.0429` vs. reversible's
     `-0.0232` at the same condition, `ΔS_h=0.0197` — clears the gate.
     **New D6 rows authorized** at 316.227766 Hz.
  7. **D5 passivation chemistry re-qualified with a live-data proxy.**
     True sub-event p_B/p_P traces aren't archived at this instrumentation
     depth, so the mean of `pre_event_max_pB`/`max_pB_post_commit` across
     each chemistry factor's 12 already-completed events is used as the
     coarsest available *live* (not analytical) proxy for cycle-mean p_B:
     chem=1.0→0.397 (|Δ from 0.30|=0.097), chem=0.3→0.184 (|Δ|=0.116),
     chem=0.1→0.064 (|Δ|=0.236) — chem=1.0 remains closest, consistent
     with the analytical pick and the largest-live-effect ranking
     (3-way convergent evidence). The literal "mean p_P>=0.30" qualifier
     is not independently verified (p_P not archived) — documented as
     such rather than assumed. **D5 stays authorized at chemistry=1.0.**
  8. **`post_screen_protocol_selection.{json,csv}`** records every D1–D7
     decision (analytical preselection / observed screen result /
     post-screen amendment / reason / omitted), explicitly *not*
     overwriting `analytical_regime_selection.json` (still an accurate
     record of what was analytically predicted, now superseded for D3/D4/
     D6 by the post-screen amendments above).
  9. **Strict verifier** (`verify_part_x_px3_5.py`) independently
     reproduces all 18 pairwise S_h values from the tracked ledger alone
     (`runs/` hidden), confirms R=+0.10's near-exact parity, the frequency/
     dwell/selection records' self-consistency, that `developed_job_
     registry.csv`'s `AUTHORIZED_PX4` rows exactly match `post_screen_
     protocol_selection.json`, that zero PX4 jobs have been launched yet,
     and the producer-provenance proof — `overall_pass=true` (8/8 checks).

  Final `developed_job_registry.csv` state (102 rows): D1/D2/D5 unchanged
  (`AUTHORIZED_PX4`, 8 rows each); D3 now `AUTHORIZED_PX4` at 316.227766 Hz
  (10 new rows; the old 100 Hz rows remain as a labeled historical record,
  never authorized); D4 `BLOCKED_DWELL_GATE_NOT_SATISFIED` (terminal, not
  "pending" — the audit is conclusive); D6 old 1000 Hz rows `BLOCKED_
  PROTOCOL_MISMATCH`, new 316.227766 Hz rows `AUTHORIZED_PX4` (10 rows);
  D7 unchanged (all aliased, no endpoint selected); D1_confirm/D2_confirm
  unchanged (`BLOCKED_PENDING_STAGE1_DECISION`, a PX4-Stage-1 concern).

## Terminal and pending counts

- Physical trajectories launched: 28 PX3 screen (all COMPLETE) + 8 PX3.5
  targeted reruns (4 dwell-causal-audit legs: dynamic-finite + prescribed-
  static at each of hold=0.0005s/0.002s; 2 frequency-bisection legs at
  316.227766 Hz; 2 persistent-at-transition legs at the same frequency —
  all COMPLETE, 0 quarantined). PX4 developed campaigns: 0 launched
  (verified by `verify_part_x_px3_5.py`'s own check).
- PX1, PX2, PX2.5, PX3, and PX3.5 are done. PX4–PX7: not started.

## Exact next action

Launch PX4 (mission section 8): the developed multi-K campaign for every
row now `AUTHORIZED_PX4` in `developed_job_registry.csv` (D1, D2, D3, D5,
D6 — 44 rows across their respective Kmax grids) via `part_x_physical_
controller.py`, respecting the 3-worker cap and the trajectory budget in
section 8 (max 30 accepted events, 150 μm, 1e12 cycles). Do **not** use a
general `--allow-head-drift` — require exact launch-HEAD match to each
row's `physical_producer_sha`, or a fresh producer-provenance proof
identical in form to `px3_physical_producer_provenance.json` if the
registry needs another producer-sha refresh first. Per the review's
explicit instruction, continue automatically through PX5–PX7 without
stopping merely because each is a new phase; before PX5, wire the
already-qualified `static_shield_phase_resolved_action` evaluator (PX1.4)
into the live production static-control commit path (PX3.5 already
proved out the `run_trajectory` `static_shield_control` hook works
correctly with `hold>0`, which PX5 will need). This stage will take real
wall-clock time — launch via background workers and continue via the
harness's notification system rather than blocking synchronously.
