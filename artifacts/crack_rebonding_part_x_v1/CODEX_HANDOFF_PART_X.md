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

## Terminal and pending counts

- Physical trajectories launched: 0 (no physical result directory exists
  yet, per mission section 2's gate — PX2 analytical work must land first).
- PX1.1, PX1.2 done. PX1.3–PX7: not started.

## Exact next action

PX1.3 (RB3 passivation audit): `patch_Q`'s P↔C↔B generator already exists
and is wired for `PASSIVATION_GATED_REBOND` (depassivation/repassivation
via `depassivation_rate`/`repassivation_rate`, ~lines 174–179) — this is
audit/test-coverage on existing code, not new physics. Add tests through
the real A_NATIVE production engine demonstrating live, nonzero,
correctly-signed P→C depassivation, C→P repassivation, C→B formation, B→C
rupture; exact probability conservation/nonnegativity; compression-gated
depassivation; exact zero contact formation at K>=0; repassivation at its
qualified phase/stress condition; fresh-surface P/C initialization;
rollback/checkpoint restoration of all three states; no direct
energy-gate/emission contamination. Then PX1.4: generalize the exact
phase-resolved event-time evaluator (`solve_coupled_event_time`/
`phase_resolved_action`) around a common `K_b` shielding provider —
dynamic rebonding, prescribed-static, and a zero provider that must still
bypass to the original baseline exactly — to attempt resolving
`STATIC_DYNAMIC_LOCALIZER_PARITY_UNRESOLVED` per mission section 5.4's
frozen tolerance protocol. PX1 completion (section 5.5) then requires one
more full run of: all new focused tests, full `crack_rebonding` selection,
both existing verifiers, `py_compile`/`compileall`, `git diff --check` —
before PX2 (kinetic-regime analytical design) may begin.
