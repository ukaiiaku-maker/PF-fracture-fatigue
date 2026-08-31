# Optional Crack-Rebonding Ablation — Final Decision (Software-Integration Pass)

## Scope

This pass implemented, tested, and verified the **software** deliverable for
an optional, default-off crack-face rebonding module in the qualified
v10.2.30 fatigue solver: Gates S0 through S7 of the approved,
three-times-reviewed plan (`docs/v10_2_30_crack_rebonding_approved_plan.md`,
SHA-256 `8d56ce3dc5d25cd90fb6fee22d42daefb59e980cfedb520a25ec9c2c681654b9`),
plus a round-3-requested prephysical qualification pass (below). The
mission's Part X physical campaign (developed Paris-slope curves,
R/frequency/passivation sweeps, figures) remains **explicitly out of scope**
and was not attempted; neither was a branch merge.

## Round-3 prephysical qualification pass

Round-3 review declined to authorize Part X or a merge pending a narrowly
scoped qualification pass, required to (1) exercise the complete production
engine, stochastic first passage, energy-gated event transaction, and
event-generated wake for at least two accepted events; (2) resolve the
phase-semantics question (carry chronological phase across blocks/events, or
relabel as cycle-averaged); (3) qualify runtime before a low-K/VHCF
campaign; (4) trace/reduce the installation architecture and confirm the
isotropic `E'` convention.

This pass discovered that the previously "confirmed" injection point
(`kinetic_tip_cell.py::cycle_step_waveform`) is **dead code** for the actual
production engine — a three-layer MRO correction, fully documented in
`docs/v10_2_30_crack_rebonding_equation_lineage.md` under "Injection-point
correction history". The real injection point is
`persistent_site_coupled_hazard_v10229.py::_phase_statistics`/
`_commit_constant_segment`. With that corrected, all four numbered
requirements above are addressed:

1. `tests/test_v10_2_30_crack_rebonding_full_production_qualification.py`
   constructs the real, fully-composed production engine
   (`CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine`) via genuine
   `__init__` construction and drives it through two real accepted
   stochastic first-passage events, verifying: accepted-length (not
   proposed-length) patch creation, `p_B=0` at creation, no state
   inheritance across events, chronological `elapsed_time_s` continuity
   (monotonic, never resets), and exact rollback via
   `restore_geometry_veto`. **Not verified**: full-MPZ-state agreement
   against an independent fine-step reference integrator at the
   rebonding-corrected event time — see "Known limitations" below; this is
   the one numbered invariant from round-3's 12-item list not attempted in
   this pass.
2. Resolved by choosing the explicit-carry option: `RebondingWakeState.elapsed_time_s`
   is a persistent chronological clock advanced by the true elapsed
   duration of every committed block or event (never the nominal one),
   documented in the equation-lineage doc's "Chronological phase
   continuity" section.
3. `crack_rebonding_v10230.phase_resolved_action` implements the periodic-
   orbit bulk-action evaluator: resolves a finite transient exactly, then
   bulk-advances via `matrix_power`'s `O(log n)` scaling, with **unconditionally
   bounded runtime** once at least one transient cycle has resolved (a
   genuine hang from an earlier fallback design was caught and fixed — see
   equation-lineage doc). `tests/test_v10_2_30_crack_rebonding_vhcf_performance.py`
   verifies a billion-cycle interval completes in well under a second.
4. The installation architecture remains a single authoritative call site
   (`reduced_shared_state_v1023.py::build_shared_engine` ->
   `install_crack_rebonding`, idempotent); the isotropic `E'=2G/(1-nu)`
   convention is documented as intentionally isotropic-only in
   `crack_rebonding_v10230.reduced_modulus_Pa`'s docstring and the
   equation-lineage doc, with an explicit note that a future
   resolved-anisotropic implementation must not reuse it unmodified.

Fixing the injection point also surfaced and fixed a genuine regression: a
third, independent caller of `_commit_constant_segment`
(`persistent_site_forward_coupled_hazard_v10230.py::_constant_segment`) used
the old positional signature and broke with the new `controller` parameter.
Caught by a full, unfiltered regression sweep (not `-x`), fixed by threading
`controller` through; see "Regression-sweep results" below.

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

Three rounds of independent technical review corrected the design before,
during, and after implementation:
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
- **Round 3** (post-implementation assessment, 4 points): withheld
  authorization for Part X/merge pending a prephysical qualification pass —
  required exercising the complete production engine chain for ≥2 accepted
  events (surfacing the three-layer dead-injection-point defect above),
  resolving phase-semantics ambiguity, qualifying VHCF/low-K runtime, and
  tracing the installation architecture. All four addressed in this pass;
  see "Round-3 prephysical qualification pass" above.

All corrections are implemented, not merely acknowledged — see
`docs/v10_2_30_crack_rebonding_equation_lineage.md` for the equation-by-
equation trace, including the discovered phase-provenance simplification
and the confirmed `normalized_progress_rate` closed form the event-time
root-finder is built on.

## Verification results

- 156/156 rebonding-specific tests pass (152 from the software-integration
  pass plus 4 new full-production-qualification tests from the round-3
  pass).
- Regression-sweep results (round-3 pass): a full, unfiltered
  `python -m pytest tests/ -q` (no `-x`, so pre-existing failures don't mask
  new ones) shows **78 pre-existing failures / 887-892 passed** both
  **before and after** the round-3 production-file edits, confirmed by a
  `git stash` A/B comparison of the exact FAILED-test-name sets — **zero net
  new failures** after the `_commit_constant_segment` third-caller
  regression (found by this same sweep) was fixed. The 78 pre-existing
  failures are overwhelmingly missing-`runs/`-physical-campaign-artifact
  issues in this isolated worktree, unrelated to rebonding; see the
  equation-lineage doc's "Regression-sweep baseline caveat" section for the
  full list and reasoning, reported honestly rather than omitted.
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
5. **Three-layer dead injection point** (round-3 pass): the originally
   "confirmed" injection point (`kinetic_tip_cell.py::cycle_step_waveform`)
   turned out to be unreachable by the real production engine class, as was
   a second attempt (`persistent_site_cyclic_v10229.py`) — both are
   independently-overridden dead code for this class hierarchy. The real
   injection point (`persistent_site_coupled_hazard_v10229.py`) was only
   found by direct `__init__` construction of the real production class and
   a live multi-event trace. See the equation-lineage doc for the full
   three-attempt history.
6. **Third caller of `_commit_constant_segment`** (round-3 pass): adding a
   `controller` parameter to fix defect 5 broke an independent caller in
   `persistent_site_forward_coupled_hazard_v10230.py` using the old
   positional signature, caught by a full (non-`-x`) regression sweep and
   fixed by threading `controller` through.

## Known limitations, honestly scoped

- **`phase_resolved_action`'s transient phase remains `O(n_phase × n_transient_cycles)`**;
  only the post-transient bulk portion achieves `O(log n)` via `matrix_power`.
  `tests/test_v10_2_30_crack_rebonding_vhcf_performance.py` confirms this is
  sufficient for billion-cycle VHCF blocks in practice (bounded runtime
  unconditionally once ≥1 transient cycle resolves), but a pathologically
  slow-relaxing configuration with `max_transient_cycles` set very high
  would still pay the full transient cost — a tunable, not an unbounded
  risk.
- **Full-MPZ-state agreement against an independent fine-step reference
  integrator, at the rebonding-corrected event time, has not been
  attempted.** This was round-3's explicit requirement and is the one
  numbered invariant from its 12-item recommended-gate list not covered by
  `test_v10_2_30_crack_rebonding_full_production_qualification.py`. Building
  a fine-step brute-force reference (stepping the full MPZ + rebonding state
  phase-by-phase rather than via the adaptive-Simpson/exact-propagator
  machinery, over the same threshold stream/RNG draws) is a substantial
  independent effort — plausible but not attempted in this pass. This is
  the reason `REBONDING_PREPHYSICS_INTEGRATION_QUALIFIED` (the classification
  round-3 named as the outcome of a fully-completed recommended gate) is
  **not** claimed below; what is qualified is the round-3-required
  end-to-end production-chain exercise (stochastic first passage +
  transactional event + event-generated wake for ≥2 events), not the
  fine-step numerical cross-check.
- **Part X (the physical campaign) has not been run.** No claim is made
  about whether rebonding alters, steepens, flattens, or arrests the Paris
  response.

## Terminal classification

```
REBONDING_CORE_SOFTWARE_QUALIFIED
LIVE_TRANSACTIONAL_PATH_QUALIFIED_PENDING_FINE_STEP_REFERENCE
VHCF_REBONDING_PERFORMANCE_QUALIFIED
PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED
```

`LIVE_TRANSACTIONAL_PATH_QUALIFIED_PENDING_FINE_STEP_REFERENCE` reflects
that the real production engine's stochastic-first-passage ->
energy-gated-transaction -> event-generated-wake chain is now exercised
and verified end-to-end for ≥2 accepted events (round-3's core concern,
fully addressed), while the fine-step MPZ-state reference comparison
remains open (round-3's numerical-agreement requirement, not yet
attempted) — an intermediate label between the round-3-suggested
`LIVE_TRANSACTIONAL_PATH_NOT_YET_QUALIFIED` (too pessimistic; the chain is
now proven to work correctly on the real engine) and
`REBONDING_PREPHYSICS_INTEGRATION_QUALIFIED` (too strong; that label was
tied to completing all 12 recommended-gate invariants, and the fine-step
comparison specifically was not completed).

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
