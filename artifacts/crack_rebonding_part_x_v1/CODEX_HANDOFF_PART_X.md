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

## Terminal and pending counts

- Physical trajectories launched: 0 (no physical result directory exists
  yet — PX3 is the first stage authorized to create one).
- PX1 and PX2 done. PX3–PX7: not started.

## Exact next action

PX3 (bounded single-K mechanism screen, mission section 7): **this is the
first PX stage that launches real physical trajectories.** Before
launching anything, implement the persistent disk-backed controller
mission section 14 requires (max 3 concurrent workers, atomic registry
updates, unique result paths under `runs/crack_rebonding_part_x_v1/`,
expected-HEAD and config-hash enforcement, no-resume enforcement,
prephysics-failure and interrupted-physics quarantine). Common screen
settings: A_NATIVE, 300K, Kmax=18 MPa√m, seed=1720, mpz_n_bins=80,
n_phase=80, 12 accepted events or 60 μm (whichever first), **explicit
phase-resolved integration only** (`V10230_FATIGUE_INTEGRATOR_MODE=
explicit` — DMD/Poincaré/projective acceleration forbidden), virgin paths,
no resume. Launch `screen_job_registry.csv`'s 36 queued jobs (18
finite/zero matched pairs) respecting the worker cap; for each pair
compute `g`/`S_h` per section 7.8 and apply the frozen selection rules
already recorded in `analytical_regime_selection.json` (frequency=100Hz,
dwell=0.5ms, chemistry=1.0 already selected; cohesive-strength endpoint
selection is PX3's own job, decided from the actual 3-point screen once
it exists). This stage will take real wall-clock time for physical
simulation — launch via background workers and continue via the harness's
notification system rather than blocking synchronously.
