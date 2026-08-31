# Optional Crack-Rebonding Ablation — Final Decision (Software-Integration Pass)

## Scope

This pass implemented, tested, and verified the complete **software** deliverable
for an optional, default-off crack-face rebonding module in the qualified
v10.2.30 fatigue solver: Gates S0 through S7 of the approved, twice-reviewed
plan (`docs/v10_2_30_crack_rebonding_approved_plan.md`, SHA-256
`8d56ce3dc5d25cd90fb6fee22d42daefb59e980cfedb520a25ec9c2c681654b9`). The
mission's Part X physical campaign (developed Paris-slope curves,
R/frequency/passivation sweeps, figures) was **explicitly out of scope** for
this pass and was not attempted.

## What was built

- `arrhenius_fracture/crack_rebonding_kinetics_v10230.py`: pure kinetics —
  config, log-domain floor-bounded Arrhenius rates, the exact 3×3
  conservative Markov generator/update, a generic exact phase-aware
  propagator (arbitrary starting phase, fractional/multi-cycle durations, no
  rounding), the two-state analytical bonded-fraction map, and the
  reference-action parameter generator.
- `arrhenius_fracture/crack_rebonding_v10230.py`: engine integration — the
  segment-ledger wake state with transactional snapshot/restore, the
  representative-cycle `K_rebond(phase)` construction, the two-stage
  block-size limiter, and the rebonding-coupled event-time root-finder
  (bisection on the phase-resolved action, reusing the existing
  `_integrate_coupled` machinery only as a pure closed-form predictor/final
  bookkeeping call, never re-invoked statefully with different trial rates).
- Five minimal, guarded production-file edits (`kinetic_tip_cell.py`,
  `persistent_site_cyclic_energy_gated_v10230.py`,
  `reduced_shared_state_v1023.py`,
  `sharp_front_v10_2_30_energy_gated_fatigue.py`,
  `persistent_site_high_cycle_checkpoint_v10230.py`), all gated on
  `getattr(self, "_rebonding_state", None)`.
- 146 new tests, a self-contained software verifier
  (`scripts/verify_v10_2_30_crack_rebonding.py`, exits 0), and two
  documentation artifacts (equation lineage, config schema).

## Review history

Two rounds of independent technical review corrected the design before and
during implementation:
- **Round 1** (12 points): rejected a non-physical `round(cycles)` kinetics
  update, a coupling order letting cleavage sample stale rebonding state,
  an incorrect `m_h=1` recovery claim, a disabled-path parity mechanism that
  edited a hot method's source unnecessarily, a double-installation risk,
  and an unaudited "hazard-only" assumption.
- **Round 2** (9 points): rejected a block-midpoint-scalar `K_rebond`
  approximation, an incorrect midpoint-formula for odd cycle counts, a
  design that would correct the wake-state commit but not the event *time*
  itself, a first-derivative-only block limiter, and a feedback-mode scope
  contradiction.

All corrections are implemented, not merely acknowledged — see
`docs/v10_2_30_crack_rebonding_equation_lineage.md` for the equation-by-
equation trace, including the discovered phase-provenance simplification
and the confirmed `normalized_progress_rate` closed form the event-time
root-finder is built on.

## Verification results

- 146/146 new tests pass.
- Broader existing-test regression sweep: 71 failed / 445 passed both
  **before and after** the production-file edits (confirmed via `git
  stash` comparison) — zero regressions. The 71 pre-existing failures are
  unrelated missing-`runs/`-artifact and external-donor-directory issues in
  this worktree, not caused by this work.
- Disabled-path parity confirmed on real `build_shared_engine`-constructed
  engines: bit-identical `cycle_step_waveform` output across independent
  runs, no rebonding attributes ever allocated, checkpoint key set
  unchanged.
- RB1 (`CONTACT_PROXY_ONLY`) confirmed physically identical to RB0 at both
  positive and negative R on a real engine.
- RB2 confirmed to produce a genuinely nonzero, physically correct
  (shielding, not strengthening) cleavage-rate change when a bonded patch
  exists, while emission stays exactly unchanged — this required manually
  seeding a patch, since `build_shared_engine`'s bare engine has no
  transactional event-commit layer through which a patch could form
  organically (see "Known limitations" below).
- The acceleration fail-closed gate raises before any monkeypatch is
  applied when rebonding is enabled with `model_level != REBOND_OFF`,
  verified against the real `main()` entry point via subprocess.
- `git diff --check` clean; all touched files compile.
- `scripts/verify_v10_2_30_crack_rebonding.py` exits 0.

## Real defects found and fixed during this pass

1. **`write_shared_result` enum-serialization break**: adding a nested
   dataclass with enum fields to `SharedReducedConfig` broke an existing
   `json.dumps` call with no `default=` handler. Fixed.
2. **Event-time root-finder instability**: an initial fixed-point-on-λ
   iteration scheme diverged (confirmed by a failing synthetic test with a
   super-linear time-dependent rate) before being replaced with stable
   bisection on the monotone phase-resolved action.
3. **Dead code in `install_crack_rebonding`**: redundant `NotImplementedError`
   checks after `cfg.validate()` (which already raises for the same
   condition, before any engine mutation) were unreachable and removed.
4. **Test-coverage gap caught while writing the verifier**: an early
   "emission unchanged when rebonding active" check compared two runs where
   `K_rebond` was trivially zero in both (empty wake, no organic event path
   on the bare engine), proving nothing. Fixed by manually seeding a bonded
   patch to force a genuinely nonzero effect.

## Known limitations, honestly scoped

- **`phase_resolved_action` is `O(n_phase × n_cycles)`**, not the `O(log n)`
  the exact propagator achieves for wake-state trajectories alone. A
  production block spanning billions of cycles with no event firing would
  need a representative-cycle bulk approximation before being called on
  such an interval. Documented in the equation-lineage doc and the module
  docstring; not implemented in this pass.
- **The deeper transactional event-commit root-finder wiring
  (`_commit_rebonding_event`) has not been exercised via a live,
  multi-event production CLI run** in this session — that requires an
  external kernel-family fixture file and a real multi-hour run, both out
  of scope given the concurrent compute load from other work on this
  machine during this session. It is validated by: (a) direct source
  confirmation of the exact `normalized_progress_rate` formula it is built
  on, (b) isolated unit tests of `solve_coupled_event_time`/
  `phase_resolved_action` against closed-form synthetic scenarios, and (c)
  confirmation that the "corrected" production engine subclass's method
  overrides both call `super()`, so the edited base-class logic is reached.
  This is a genuine gap between "unit- and integration-tested" and
  "exercised end-to-end on a live multi-cycle run" that should be closed
  before or during the Part X physical campaign, not asserted away.
- **Part X (the physical campaign) has not been run.** No claim is made
  about whether rebonding alters, steepens, flattens, or arrests the Paris
  response.

## Terminal classification

```
REBONDING_SOFTWARE_INTEGRATION_QUALIFIED
PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED
```

## Branch/commit state

- Branch: `codex/v10.2.30-optional-crack-rebonding`
- Worktree: `/private/tmp/v10230-crack-rebonding-ablation` (clean, all work
  committed)
- Base commit: `b7bd38b97da551aaef1b43d6d28c9ea44a06655c`
- Solver-hash provenance: `MISSION_SOLVER_HASH_UNREPRODUCED_BUT_BASE_COMMIT_VERIFIED`
  (`docs/v10_2_30_solver_hash_provenance.md`)

## Note on the protected temperature-fatigue worktree

`/private/tmp/v10230-reversible-energy-integration` was never touched by
this work (confirmed throughout: only read-only `ps`/`git -C` checks were
ever performed against it). By the end of this session its campaign
processes had stopped running (naturally, or by external action — not by
anything in this session) and its `.git` worktree pointer file is now
missing, though the directory and its contents remain intact and the main
repository's worktree metadata still references it. This was observed, not
caused, and no corrective action was taken on that worktree per the
mission's explicit protection requirement.
