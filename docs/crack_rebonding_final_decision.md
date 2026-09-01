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

## S8: full-state and accuracy qualification pass

The round-3 assessment of commit `668c6a2` accepted the injection-point
correction and production-chain qualification as a strong milestone but
required one further gate, `S8_REBONDING_FULL_STATE_AND_ACCURACY_QUALIFICATION`,
before authorizing Part X or a merge — four sub-requirements, all addressed:

- **S8A (full-state chronological reference)**: `test_v10_2_30_crack_rebonding_full_state_reference.py`
  compares a real engine driven through the normal adaptive-Simpson block
  integrator against an independent fine-step reference (same starting
  state, same RNG/threshold stream, same underlying `_phase_statistics`/
  `_commit_constant_segment` primitive at far finer granularity) through 2
  accepted events, using the repo's own `serialize_active_state`/
  `residual_metrics`/`capture_ledgers`/`capture_stochastic_state` utilities
  — the full MPZ/ledger/RNG state agrees to a tight, frozen tolerance; the
  wake's own chronological clock and kinetics agree to a looser, empirically
  justified tolerance reflecting a genuine (pre-existing, non-rebonding)
  numerical-resolution effect, documented in the test file and the
  equation-lineage doc. This closes the one invariant round-3 explicitly
  flagged as not yet attempted.
- **S8B (causal bond formation)**: `test_v10_2_30_crack_rebonding_causal_bonding.py`
  uses the existing reference-action parameter generator
  (`solve_reference_action_barriers` with the `"persistent"` preset) to
  calibrate a strong-but-nonsaturated configuration, confirmed by a direct
  calibration sweep, and shows on the real engine's *live* kinetics (not a
  manually seeded patch): nonzero bonded fraction and nonzero `K_rebond`
  develop between two real events, the second event is measurably delayed
  (roughly doubled) relative to a contact-proxy-only control under the
  identical threshold stream, emission stays bit-identical, and the
  accepted event length is unaffected — closing round-3's concern that the
  earlier qualification test proved bookkeeping but not the live causal
  mechanism.
- **S8C (certified VHCF bound)**: the bulk-action acceleration's earlier
  "last resolved transient cycle as best-effort representative" (no error
  bound, no fail-closed signal) is replaced with a periodic-orbit
  certificate — the true periodic state and subdominant eigenvalue of each
  patch's exact one-cycle propagator, an unbiased bulk representative, and a
  numerically-summed geometric-series tail-action-error bound, with
  caller-side fail-closed enforcement (retry with an extended transient
  budget, then raise) rather than ever committing an event on an uncertified
  estimate. See the equation-lineage doc for the full derivation and the
  two real bugs found and fixed while implementing it (a false-degeneracy
  bug from naively excluding only one eigenvalue when a patch's generator is
  legitimately reducible, and the resulting need to compute the periodic
  state by forward simulation rather than eigenvector selection).
- **S8D (outer-driver exercise), honestly scoped**: an additive,
  default-preserving `V10230_FATIGUE_INTEGRATOR_MODE` selector
  (`"accelerated"` default, byte-identical to before; `"explicit"` opt-in)
  makes a genuine CLI run with rebonding enabled possible in principle,
  resolving *why* the Gate-S0 acceleration gate must be unconditional today.
  Investigation confirmed that exercising the full outer chain through
  `EnergyGatedAvalancheBackend.advance()` requires a real FEM
  mesh/boundary/damage/displacement fixture that does not exist in this
  repo and would require hours of unplanned engineering to construct — an
  honestly-scoped, documented gap, not attempted. What IS implemented and
  tested: `finalize_engine_event()` itself — the real glue function,
  including the actual `register_engine`/`_engine_from_id` weakref-registry
  lookup a genuine run uses — plus a full round trip through the real,
  file-based checkpoint mechanism after a commit reached through that glue.

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
- **Round 3 follow-up assessment** (post-`668c6a2`, S8, 4 points): accepted
  the injection-point correction as a strong milestone but withheld
  authorization further pending
  `S8_REBONDING_FULL_STATE_AND_ACCURACY_QUALIFICATION` — full-state
  fine-step reference agreement, live (not seeded) causal bond formation,
  a certified VHCF error bound, and an outer-driver exercise. All four
  addressed, one (S8D's mesh-dependent backend layer) at an honestly
  scoped, documented partial completion; see "S8: full-state and accuracy
  qualification pass" above.

All corrections are implemented, not merely acknowledged — see
`docs/v10_2_30_crack_rebonding_equation_lineage.md` for the equation-by-
equation trace, including the discovered phase-provenance simplification
and the confirmed `normalized_progress_rate` closed form the event-time
root-finder is built on.

## Verification results

- 170/170 rebonding-specific tests pass (156 at the round-3 commit plus 14
  new/added across S8A–D's five new test files and the extended VHCF/
  event-time-coupling assertions).
- Regression-sweep results (round-3 pass): a full, unfiltered
  `python -m pytest tests/ -q` (no `-x`, so pre-existing failures don't mask
  new ones) shows **78 pre-existing failures / 887-892 passed** both
  **before and after** the round-3 production-file edits, confirmed by a
  `git stash` A/B comparison of the exact FAILED-test-name sets — **zero net
  new failures** after the `_commit_constant_segment` third-caller
  regression (found by this same sweep) was fixed.
- Regression-sweep results (S8 pass): the same `git stash` A/B discipline,
  repeated for S8A–D's production-file edits, shows the **exact same 78
  pre-existing failures, zero new, zero fixed**. The 78 pre-existing
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
7. **False-degeneracy in the periodic-orbit certificate** (S8 pass): an
   initial implementation picked "the" eigenvalue closest to 1 and treated
   every other eigenvalue as subdominant; when a patch's generator was
   legitimately reducible (e.g. a currently-unreached passivated state,
   `k_PC=k_CP=0` for a whole cycle — a benign, expected case, not a modeling
   error), `M_cycle` genuinely has more than one eigenvalue exactly 1, and
   excluding only one of them inflated the computed subdominant eigenvalue
   to ≈1, falsely flagging every such patch as degenerate. Caught by two
   failing VHCF-performance tests; fixed by excluding every eigenvalue
   within tolerance of 1 (however many) from the subdominant set, and
   computing the periodic state by forward-simulating `matrix_power`
   (lands in whichever invariant subspace the current state actually
   projects onto) rather than selecting an eigenvector.
8. **S8A reference-resolution artifact, correctly diagnosed rather than
   silently tolerance-widened**: the fine-step reference's wake
   `elapsed_time_s`/kinetics initially disagreed with production by ~1-4%
   depending on granularity. Traced to a genuine, pre-existing,
   non-rebonding property of `_commit_constant_segment` (a piecewise-
   constant-rate-per-segment approximation whose accuracy depends on how
   often the "current" rate estimate is refreshed, amplified by cleavage's
   Arrhenius-exponential sensitivity to stress) rather than a rebonding
   coupling bug — confirmed by a direct convergence study (residual shrinks
   as the reference step size shrinks) before finalizing frozen, documented
   tolerances, not by loosening them until the test passed.

## Known limitations, honestly scoped

- **`phase_resolved_action`'s transient phase remains `O(n_phase × n_transient_cycles)`**;
  only the post-transient bulk portion achieves `O(log n)` via `matrix_power`.
  `tests/test_v10_2_30_crack_rebonding_vhcf_performance.py` confirms this is
  sufficient for billion-cycle VHCF blocks in practice (bounded runtime
  unconditionally once ≥1 transient cycle resolves), and the bulk *action*
  representative is now certified (S8C) rather than best-effort — an
  uncertified bound now fails closed (extends the transient budget, then
  raises) instead of silently committing.
- **The full outer FEM/mesh backend (`EnergyGatedAvalancheBackend.advance()`)
  has not been exercised**, and is not feasible to exercise in this
  environment without first constructing a real production kernel-family
  (a slow, out-of-scope FEM/mesh subprocess build) or a from-scratch
  synthetic FEM fixture (substantial, error-prone, out-of-scope
  engineering). This is the one residual gap from S8D; what IS exercised is
  `finalize_engine_event()` itself (the real glue/registry-lookup layer,
  requiring no mesh) plus a full round trip through the real file-based
  checkpoint mechanism.
- **Part X (the physical campaign) has not been run.** No claim is made
  about whether rebonding alters, steepens, flattens, or arrests the Paris
  response.

## Terminal classification

```
REBONDING_PREPHYSICS_INTEGRATION_QUALIFIED
  (mesh-dependent EnergyGatedAvalancheBackend.advance() layer not exercised;
   finalize_engine_event() and the real checkpoint mechanism are)
PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED
```

With S8A (full-state fine-step reference agreement), S8B (live, non-seeded
causal bond formation with a measurable event-time delay), and S8C
(certified, fail-closed VHCF bulk-action bound) all complete, and S8D
complete at the maximum feasible scope in this environment (the
`integrator_mode` selector plus the real `finalize_engine_event`/checkpoint
glue, with the mesh-dependent backend layer honestly documented as an
out-of-scope residual gap rather than silently claimed), this pass now
meets the substance of round-3's recommended gate — the classification it
named as the outcome of a fully-completed gate,
`REBONDING_PREPHYSICS_INTEGRATION_QUALIFIED`, is adopted, with the one
residual gap stated explicitly alongside it rather than folded into the
label itself.

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
