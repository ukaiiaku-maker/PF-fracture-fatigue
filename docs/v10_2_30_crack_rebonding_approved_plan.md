# Optional Crack-Rebonding Ablation Module (v10.2.30) — Revised

## Context

The qualified v10.2.30 fatigue solver has no way to represent contact-conditioned crack-face
rebonding ("cold welding"): during compressive/contact portions of a fatigue cycle, freshly
created crack surfaces can locally bond, and those bonds partially survive reopening, adding
a history-dependent cohesive shielding contribution that competes with the opening-driven
cleavage hazard. This adds that as a new, optional, **default-off** mechanism to explore
whether contact-conditioned bond formation measurably alters developed da/dN — and if so
whether the effect is a rate offset, onset shift, local Paris-slope change, or high-K
flattening/arrest — without touching existing qualified mechanics when the feature is off.

Provenance/safety already handled:
- Base commit `b7bd38b97da551aaef1b43d6d28c9ea44a06655c` confirmed clean and exactly what the
  currently-running, separate temperature-fatigue campaign (`/private/tmp/v10230-reversible-energy-integration`,
  3 live workers + a controller) is anchored to via `--expected-head`.
- New worktree `/private/tmp/v10230-crack-rebonding-ablation`, branch
  `codex/v10.2.30-optional-crack-rebonding`, clean, isolated.

**This plan has been through two rounds of technical review; both rounds' corrections are
folded in below.** Round 1 covered: a non-physical `round(cycles)` update to the healing
kinetics; a coupling order letting cleavage sample stale rebonding state for a whole
multi-cycle block; an incorrect `m_h=1` recovery claim; a disabled-path parity mechanism that
edited a hot production method's source even though it degenerated to a no-op; a double
top-level installation risk; and an unaudited "cleavage-hazard-only" assumption. Round 2 went
deeper on the numerics: even with an exact `dt_used`-based commit, (a) a *single*
block-midpoint scalar `K_rebond` applied to the whole phase array loses the essential
compression→bond→reopen→rupture correlation within a cycle, (b) the block-midpoint formula as
first written was simply wrong for odd cycle counts, (c) advancing the wake to the correct
elapsed time does not by itself correct an event *time* that was estimated using the wrong
rebonding history, (d) a first-derivative block-size limiter cannot see a P→C→B activation
about to switch on from a currently-passivated patch, and (e) the plan had an internal
contradiction between implementing `COMMON_POSITIVE_LOCAL_K_REDUCTION` (which touches emission)
and simultaneously claiming emission is unconditionally unchanged. All of these are resolved
below, several confirmed against the actual source (not merely argued): I traced the real
`cycle_step_waveform` override chain (`CampaignCalibratedTipEngine → SeparatedSourceKineticTipEngine
→ ContinuumSourceKineticTipEngine → KineticMovingTipFrontEngine`, `campaign_calibrated_tip.py:305`,
`continuum_source_tip.py:376`, `separated_source_tip.py:112`, all confirmed to be
diagnostic-only pass-throughs via `super()`), confirmed the actual `sig`/`lam_e_site`/`lam_c_phase`
computation lives only in `kinetic_tip_cell.py:419-511`, confirmed the energy gate's
`continuum_gate_diagnostics` (`hazard_energy_event_gate_v10230.py:280-326`) calls
`engine.sigma_tip(K)` directly at line 294, and confirmed `commit_energy_gated_event` already
carries an exact elapsed-time field (`kinetic_dt_consumed_s`, line 300-308) that the wake commit
must key off instead of the nominal candidate block size.

**Scope of this pass**: the mission's Part X physical campaign (dozens of solver runs,
R/frequency/passivation sweeps, ~15 figures, ~25 artifacts) remains **out of scope** — it needs
many hours of runtime that would contend with the live temperature-fatigue campaign's CPU. This
plan covers the complete software deliverable, organized as a gated sequence (S0–S7), ending in
an explicit `REBONDING_SOFTWARE_INTEGRATION_QUALIFIED` / `PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED`
classification rather than any claim about the physical effect.

## Key design corrections, resolved

**Phase-resolved coupling via one generic exact propagator** (replaces a rejected single
block-midpoint-scalar `K_rebond` and an incorrect `n//2, η/2` formula):
- Kinetics are **linear and state-independent given a frozen mechanical state**: within one
  cycle-block, `s_j`, `r_eff`, and the K-waveform shape (`Kmax, R, f`) are all constant
  (confirmed: `sig` is computed once per block at `kinetic_tip_cell.py:431` and reused for the
  whole block today, which is the existing tau-leap approximation this design does not change).
  So `Q_k` (the generator at phase-grid index `k`) is a fixed function of phase alone,
  independent of absolute time and of `p`.
- **One generic operator does all propagation**, eliminating the separate (and, per review,
  incorrect) midpoint-formula special case: given a starting phase-grid index `k0` and an exact
  duration `dt`, `p(t+dt) = M(k0, dt) @ p(t)`. Implementation: convert `dt` to phase-step units
  `steps = dt / dt_phase = m·N_phase + r + f` (`m` = whole extra cycles, `r` = whole extra
  phase-steps within a cycle, `0 ≤ f < 1` = leftover fraction of one phase-step). Build the
  cyclic-order one-cycle product starting at `k0` — `M_cycle(k0) = M_{k0-1} · … · M_{k0+1} · M_{k0}`
  (indices mod `N_phase`, cyclic reordering, **not** assumed matrix-order-invariant since these
  are non-commuting 3×3 matrices — each block's `M_cycle` is rebuilt from the `k0`-rotated
  factor sequence) — then `M(k0, dt) = expm(Q_{k'} · f · dt_phase) · M_r(k0) · matrix_power(M_cycle(k0), m)`,
  where `M_r(k0)` is the partial product of the next `r` per-phase factors from `k0`, and `k'`
  is the phase index reached after those `r` steps. `matrix_power` uses
  `numpy.linalg.matrix_power`'s binary exponentiation, `O(log m)`, feasible for
  billion-cycle VHCF blocks. This is **exact for the piecewise-constant-rate phase
  discretization** (not for the continuous sinusoid — the same discretization the rest of the
  solver already uses), and is unconditionally correct for any starting phase, any duration,
  fractional or many-cycle, with no separate odd/even or midpoint special-casing.
- **Representative-cycle reconstruction for cleavage** (the actual physics requirement —
  compression→formation and reopening→rupture must stay correlated within a cycle, not be
  smeared into one scalar): for a block spanning temporal midpoint `t_mid = dt_block/2`, compute
  `p_repr_start = M(k0, t_mid − T_cycle/2) @ p_start` (state one half-cycle before the block's
  temporal midpoint), then run **one full Strang-split cycle** from there: for each phase step
  `k`, `p_{k+1/2} = expm(Q_k·dt_phase/2) @ p_k`, `K_rebond(k+1/2) = f(p_{k+1/2})` (via
  `_rebuild_coupling`'s `H_b`/`K_rebond` formula applied to this single-phase-step state),
  `p_{k+1} = expm(Q_k·dt_phase/2) @ p_{k+1/2}`. This produces a **length-`N_phase` array**
  `K_rebond_phase[k]`, one value per phase point, used directly to build `sig_cleave[k]`
  (replacing the single scalar) — preserving the within-cycle correlation between compression,
  formation, reopening, and rupture that a block-scalar destroys.
- **Convergence tests are mandatory, not optional**: phase-resolution convergence
  (`N_phase, 2·N_phase, 4·N_phase`), block-partition invariance (splitting a candidate block in
  two and composing `M(k0,·)` calls must reproduce the single-call result to floating
  tolerance), and explicit regression cases: a 1.2-cycle block (true midpoint at 0.6 cycles),
  odd- and even-integer-cycle blocks, and arbitrary nonzero starting phase with wraparound —
  all in `tests/test_v10_2_30_crack_rebonding_periodic_propagation.py`.

**Event time itself must be coupled to the rebonding history, not just the committed state**
(the most important numerical point — advancing the wake to the correct elapsed time does not
fix an event time that was estimated from the wrong rebonding trajectory):
- The existing first-passage mechanism (`StochasticHazardDiagnosticTipEngine._integrate_coupled`,
  `stochastic_hazard_tip.py` — **Gate S4 investigation task**: trace exactly how it estimates
  the elapsed time to threshold crossing within a candidate `dt_block`, since the plan must
  layer a correction onto this existing computation, not replace it, to preserve exact RB0
  parity) evidently treats the per-block hazard rate as the single representative value
  `lambda_avg = mu_c · frequency_Hz` (computed once per block, `kinetic_tip_cell.py:459`) and
  solves for the crossing time under that locally-homogeneous-rate assumption. With rebonding
  enabled, `mu_c` (and hence the implied crossing time) itself depends on the rebonding
  trajectory over the very interval being solved for — a circular dependency requiring
  iteration, not a one-shot correction.
- **Rebonding-coupled event-time root-finder**, engaged only when a candidate block's
  unmodified first-passage estimate indicates firing and rebonding is enabled: starting from the
  existing (uncoupled) estimate `dt_guess` as the initial predictor, (1) reconstruct the
  phase-resolved `K_rebond(phase)` trajectory over `[0, dt_guess]` from `p_start` using the
  exact propagator `M(k0,·)` plus per-phase Strang stepping (as above, but integrated
  cumulatively from block-start rather than centered at a temporal midpoint, since this is now
  solving for the true elapsed interval itself); (2) accumulate the actual cleavage action
  `ΔB_c(dt_guess) = Σ_k λ_c(K_k, K_rebond(phase_k)) · dt_phase` along that trajectory; (3) test
  closure `|B_start + ΔB_c(dt_guess) − B_threshold| ≤ ε_B`; (4) if not converged, update
  `dt_guess` via a bounded secant/bisection step using the local slope
  `λ_c(current phase, current K_rebond)` and repeat (capped iteration count, e.g. 10 — the
  rebonding correction is a modest perturbation on the bare hazard, so this should converge
  quickly in practice, verified empirically in Gate S4's tests). The converged `dt_used`
  **replaces** the value `commit_energy_gated_event` would otherwise read from
  `kinetic_dt_consumed_s`; the same iteration's final trajectory directly supplies the exact
  phase-resolved wake state at `dt_used` needed for the transaction (Gate S4), so no separate
  re-derivation is needed.
- Disabled-path parity is preserved because this root-finder is only ever invoked when
  `rebonding.enabled`; RB0 always uses the existing, untouched `dt_used`/threshold-crossing
  computation verbatim.

**Block-size limiter: preliminary estimate plus mandatory exact verification, unit-correct**
(a first-derivative-only limiter cannot see a P→C→B activation with `dp_B/dt(0)=0` from a fully
passivated patch about to become active):
- **Stage 1 (cheap preliminary candidate)**, **unit-corrected**: the local linearized rate
  `dp/dt = Q_start @ p_start` gives a *time* bound in seconds (`eps_pB / |dp_B/dt|`), which must
  be converted to a *cycle count* before joining `cycle_step_waveform`'s existing `limits` list
  (`kinetic_tip_cell.py:439-447`, already in units of cycles — `target_dB/mu_c` etc.):
  `N_lim = (eps_pB / max(|dp_B/dt|, floor)) / waveform.period_s`, same conversion for `eps_pC`
  and for the `K_rebond` bound (via the chain rule through `dH_b/dp_B`). This produces the
  initial candidate `cycles`, exactly as the existing adaptive block-size mechanism already
  works for its other constraints — no new loop needed for this stage.
- **Stage 2 (mandatory exact verification, fail-closed)**: once a candidate `cycles` is
  selected, propagate the wake **exactly** for that candidate duration via `M(k0, dt_candidate)`,
  compute the *actual* `Δp_B, Δp_C, ΔK_rebond`, and a cleavage-action consistency check
  (comparing the action implied by the single representative-rate approximation against the
  action integrated along the actual phase-resolved trajectory). If any exceeds its configured
  tolerance, **bisect the candidate `cycles` and repeat** (bounded iteration, e.g. 8 halvings)
  before accepting the block. This directly answers the concern that a currently-dormant
  `dp_B/dt(0)=0` patch that would activate partway through the candidate block is caught by the
  *exact* trial propagation in Stage 2 even though Stage 1's linearization missed it. New config
  fields: `rebonding_block_max_dpB`, `rebonding_block_max_dpC`,
  `rebonding_block_max_dK_rebond_frac`, plus `rebonding_block_action_consistency_tol`.

**`m_h=1` recovery.** Mirrors `lambda_cleave`'s own existing convention exactly
(`unified_front.py:109-114`, which already special-cases `m ≤ 1` to skip the gamma expression):
```python
if m_h > 1.0 + 1.0e-12:
    k_CB = gammainc(m_h, min(lambda_raw * tau_h, 1.0e12)) / tau_h
else:
    k_CB = lambda_raw
```
The unit test asserts exact recovery at `m_h=1` under this definition (not the unqualified
gamma expression, which only approaches `lambda_raw` in the small-`x` limit), and separately
tests the small-`x` limiting behavior for `m_h > 1`.

**Bit-identical disabled path, single hazard-only injection point, no double install** —
resolved together by one design decision, confirmed against the actual override chain read
directly from source:
- **Zero source diff** to `unified_front.py` (`sigma_tip`, `K_shield`) and to
  `separated_source_tip.py` (`sigma_opening_tip`). Neither method is edited, ever, in either
  the disabled or enabled path.
- The **only** injection point is inside `kinetic_tip_cell.py::cycle_step_waveform` (base
  implementation, `kinetic_tip_cell.py:419-511`), confirmed to be the sole place `sig`
  (line 431), `lam_e_site`/`mu_emit` (lines 432-434, **emission**), and `lam_c_phase`/`mu_c`
  (lines 435-438, **cleavage**) are actually computed — every override in the real MRO chain
  (`campaign_calibrated_tip.py:305`, `continuum_source_tip.py:376`, `separated_source_tip.py:112`,
  and the `HazardEnergyGatedPersistentSiteCyclicTipEngine`/audited mixins) is a confirmed
  diagnostic-only pass-through via `super().cycle_step_waveform(...)`.
- The edit: **leave `sig` (line 431) completely untouched** — it continues to drive
  `lam_e_site`/`mu_emit`/`avg_sig`/`stress_override` exactly as today, so emission is
  unconditionally unaffected. Add a **second, separate array** computed only when a rebonding
  state is present:
  ```python
  rebonding_state = getattr(self, "_rebonding_state", None)
  if rebonding_state is not None and rebonding_state.cfg.enabled:
      sig_cleave = crack_rebonding_v10230.cleavage_stress_with_rebond(self, Kvals, K_rebond_phase)
  else:
      sig_cleave = sig
  lam_c_phase = np.array([self.lambda_cleave(float(s), T_K)[0] for s in sig_cleave])
  ```
  (replacing the existing `lam_c_phase = np.array([self.lambda_cleave(float(s), T_K)[0] for s in sig])`
  at line 435-437, and now using the phase-resolved `K_rebond_phase[k]` array, not a scalar).
  `cleavage_stress_with_rebond` is a **free function** (not a bound method override) that
  replicates `sigma_tip`'s exact arithmetic with the extra per-phase `K_rebond` subtraction,
  called only from this one guarded call site.
- **Disabled-path parity is unconditional and structural**: when `_rebonding_state` is `None`
  (the `getattr` default), `sig_cleave = sig` — literally the same array object, zero extra
  floating-point operations, zero extra allocation beyond the `getattr` call itself. No edit to
  any hot method's source; the only change to `kinetic_tip_cell.py` is the `if/else` guarded by
  a `getattr` that returns `None` unless something explicitly installed `_rebonding_state`.
- **Energy gate is hazard-only by design, confirmed by source, not by convention**: because
  `sigma_tip` itself is never touched, and `continuum_gate_diagnostics`
  (`hazard_energy_event_gate_v10230.py:294`, `stress = ... engine.sigma_tip(K)`) is the only
  place the energy-admissible event length is derived, the energy gate is provably unaffected
  by K_rebond in the primary mode. Rebonding can only delay/hasten **first passage**, never
  change the **length** of an already-triggered event. This mode is labeled
  `HAZARD_ONLY_REBOND_SHIELD` and this classification is written into
  `crack_rebonding_equation_lineage.md` with the file:line evidence above, not asserted.
- **Single authoritative install path**: only `reduced_shared_state_v1023.build_shared_engine()`
  calls `install_crack_rebonding`. The top-level CLI wrapper
  (`sharp_front_v10_2_30_energy_gated_fatigue.py`) only loads `rebonding_cfg` from the
  environment and threads it into `SharedReducedConfig.rebonding` (passed through to
  `build_shared_engine`); it does **not** independently call the installer. This removes the
  double-install risk structurally rather than only defensively.
- **Defense in depth**: `install_crack_rebonding` is still made idempotent — hashes the frozen
  config (`sha256(json.dumps(asdict(cfg), sort_keys=True))`) and, if `engine._rebonding_state`
  already exists, no-ops on identical hash or raises `RuntimeError` on a differing hash. A
  regression test calls the installer twice deliberately (same config → no-op verified by
  identity of the wake-state object; different config → raises).
- **Checkpoint stays structurally unchanged when disabled**: the `crack_rebonding` key is added
  to the `kinetic` payload dict **only when `engine._rebonding_state is not None`** (an `if`
  around the `dict.update`, not a key present with value `None`) — old checkpoints and disabled
  new checkpoints have byte-identical key sets to today.

**Barrier floor + log-domain evaluation.** Every Arrhenius rate function computes
`ln(lambda) = ln(nu) - G_eff/(kB_eV*T)` with
`G_eff = max(G_floor_eV, G_0 - sigma*V/e)` (new config fields `bond_barrier_floor_eV`,
`rupture_barrier_floor_eV`, `depassivation_barrier_floor_eV`, each defaulting to `0.0` and
validated `≥ 0`), then exponentiates only once at the end (`lambda = exp(ln_lambda)`, with a
saturating clip on `ln_lambda` before `exp` to avoid overflow). Each rate function returns a
small diagnostics tuple `(rate, barrier_floor_fraction, log_argument)` so barrier-floor fraction
and `lambda*tau_h`/`k_CB*tau_h` saturation are archivable, without changing the primary scalar
return type used by callers.

**No callables in the frozen/hashed config.** `resolved_gap_oracle: Callable | None` is removed
from `CrackRebondingControls` entirely. Replaced with three stable, JSON-safe, hashable fields:
`contact_oracle_id: str = ""`, `contact_oracle_version: str = ""`,
`contact_oracle_source_hash: str = ""`. A separate, non-frozen, non-hashed
`ResolvedContactOracleBinding` (a plain object holding the actual callable) can be injected at
runtime and is validated against those three identifiers, but is rejected outright in this pass
regardless of whether it's supplied (see next point).

**Unwired controls rejected outright in this pass**, enforced in `.validate()`:
- `minimum_load_hold_s` must be exactly `0.0` (any nonzero value raises `ValueError` — hold
  integration is a documented future interface, not implemented).
- `stochastic_healing_enabled` must be `False` (any `True` raises).
- `contact_model` must be `SIGNED_K_COMPRESSION_PROXY`; `RESOLVED_GAP_TRACTION` **always**
  raises in this pass, even if oracle identifiers are supplied — the oracle contract has no
  implementation yet, so the field existing must not imply the path is usable.

**Pre-existing patches vs. the newly created patch have different histories; commit exactly
once.** The `dt_used`/event-time correction advances **only the patches that existed before the
event**, through the interval up to the true firing instant `t_e`. The transaction order is:
(a) advance every pre-existing active patch from block-start to `t_e⁻` using the converged
trajectory from the event-time root-finder; (b) accept the energy-gated crack increment
(unaffected by rebonding — confirmed hazard-only above); (c) translate the now-advanced
pre-existing patches' `s_j` by the accepted length; (d) create the new patch **fresh** —
`p_B=0, p_C=fresh_surface_clean_fraction, p_P=1−fresh_surface_clean_fraction`, age `0` — it
**never inherits** any pre-event bonded state, since it did not exist during the elapsed
interval; (e) the new patch begins evolving only from its creation point onward, through
whatever remains of the physical cycle. A hard invariant is enforced and tested: for a block
where an event fires, the block's *nominal* `p_end` (what the representative-cycle machinery
would have produced had no event occurred) is **never committed** — only the event-time state,
committed once. For a block where no event fires, the full nominal `p_end` is committed once,
at the natural end of `cycle_step_waveform`. There is exactly one commit per block, and its
content depends on whether an event fired.

**Phase provenance.** `_energy_gate_pending` (the existing transactional-event dict at
`persistent_site_cyclic_energy_gated_v10230.py:171-181`) gains a
`"rebonding_block_start_phase_index"` field (the `k0` used to build this block's propagators)
alongside the existing `"rebonding_state_before"` snapshot — a bare cycle-count scalar is
insufficient provenance for same-cycle and multiple-event ordering, since the *next* event or
block within the same physical cycle must resume propagation from the correct phase-grid index,
not from phase zero.

**Feedback-mode scope, narrowed to remove an internal contradiction**: implementing
`COMMON_POSITIVE_LOCAL_K_REDUCTION` (which touches `sigma_opening_tip`) while simultaneously
claiming emission is unconditionally unchanged cannot both be true. Resolved by narrowing scope:
**only `HAZARD_ONLY_REBOND_SHIELD` is implemented in this pass.**
`COMMON_POSITIVE_LOCAL_K_REDUCTION` and `HAZARD_AND_ENERGY_GATE_COUPLED` are both documented,
selectable-in-config enum values that **raise `NotImplementedError` if selected** — real, named,
future interfaces, not silently-half-built code paths. This lets the coupling test state a
single, unqualified claim ("emission unchanged" — full stop) and removes an entire ablation's
worth of implementation and test surface from this pass, appropriately narrowing scope given
how much numerical machinery the phase-resolved kinetics already require.

**Initial precrack wake vs. fresh-surface state, separated.** `initial_wake_mode` is renamed
`initial_precrack_wake_mode` (governs only whether the pre-existing crack starts with an active
wake, and if so in what state) and is fully independent of `fresh_surface_clean_fraction`
(renamed from `fresh_clean_fraction`; governs the P/C/B split assigned to **every** newly
created patch from an accepted event, unconditionally). The default `NO_INITIAL_ACTIVE_WAKE`
precrack setting has no effect whatsoever on how new event-created patches are seeded —
`commit_event` always uses `p_C = fresh_surface_clean_fraction,
p_P = 1 - fresh_surface_clean_fraction, p_B = 0` for new patches regardless of the precrack mode.

**Solver-hash discrepancy resolved, not footnoted.** Gate S0 includes an explicit investigation
task producing a definitive provenance record rather than an unresolved aside: hash every
production physics file (`unified_front.py`, `kinetic_tip_cell.py`, `separated_source_tip.py`,
`continuum_source_tip.py`, `campaign_calibrated_tip.py`,
`persistent_site_cyclic_energy_gated_v10230.py`, and the CLI entry
`sharp_front_v10_2_30_fixed_deltaK.py`) individually at the base commit, check whether any
concatenation/normalization (line endings, whitespace, a multi-file manifest hash) reproduces
the mission's stated SHA-256, and record a definitive classification —
`MISSION_SOLVER_HASH_STALE_BUT_BASE_COMMIT_VERIFIED` or the specific alternative found — into
`crack_rebonding_equation_lineage.md` and the verifier's JSON output, never left as prose.

## Implementation, as a gated sequence

### Gate S0 — Provenance and disabled architecture
- Resolve the solver-hash discrepancy and write the provenance record.
- Implement `CrackRebondingControls` + `.validate()` + env loader in the new pure module
  (below), including all rejection rules above.
- Re-confirm at implementation time (in case the worktree differs from this exploration) the
  exact `cycle_step_waveform` override chain and the energy gate's `sigma_tip` call site.
- **Investigate `StochasticHazardDiagnosticTipEngine._integrate_coupled`
  (`stochastic_hazard_tip.py`)** to locate exactly where/how the existing first-passage
  crossing time is estimated within a candidate block — required before Gate S4 can correctly
  layer the rebonding-coupled root-finder on top of it without disturbing RB0.
- No behavior change yet: this gate only lands config + provenance + confirmed hook points.

### Gate S1 — Pure kinetics
New module `arrhenius_fracture/crack_rebonding_kinetics_v10230.py` (pure, engine-independent,
fully unit-testable):
- Enums: `RebondModelLevel`, `ContactModel`,
  `FeedbackMode` (`HAZARD_ONLY_REBOND_SHIELD` implemented; `COMMON_POSITIVE_LOCAL_K_REDUCTION`
  and `HAZARD_AND_ENERGY_GATE_COUPLED` both raise `NotImplementedError` if selected),
  `InitialPrecrackWakeMode`.
- `CrackRebondingControls` (frozen dataclass) with the corrected field list above
  (callable-free, `fresh_surface_clean_fraction`/`initial_precrack_wake_mode` separated,
  barrier-floor fields, block-limiter tolerance fields).
- Log-domain, floor-bounded rate functions: `bond_formation_rate`, `bond_rupture_rate`,
  `depassivation_rate`, `repassivation_rate`, each returning `(rate, diagnostics_dict)`.
  Cooperative gating uses the corrected `m_h=1` convention.
- `build_Q(k_CB, k_BC, k_PC, k_CP)`, `advance_markov(p, Q, dt)` (`scipy.linalg.expm`, used to
  build the per-phase-step `M_k` factors).
- **The generic exact propagator**: `build_phase_factors(Q_phase_list, dt_phase) -> list[np.ndarray]`
  (the `M_k` list), `propagate(p, phase_factors, k0, dt) -> np.ndarray` (implements
  `M(k0, dt) @ p` exactly as specified above, using `numpy.linalg.matrix_power` for the
  whole-cycle-repeat portion), `strang_cycle_trajectory(p_start, phase_factors) -> np.ndarray`
  (returns the full length-`N_phase` per-phase-midpoint state array for one representative
  cycle, per the Strang-split construction above).
- `two_state_fixed_point`, `two_state_iterate` (RB2 analytical map) — used as the unit-test
  oracle for `propagate`/`strang_cycle_trajectory` under simple piecewise-constant scenarios.
- `solve_reference_action_barriers`, `REFERENCE_ACTION_PRESETS`, `freeze_reference_action_preset`
  (callable-free, cleanly hashable).

### Gate S2 — Exact periodic propagation
- Implement and test `propagate`/`strang_cycle_trajectory` in isolation against: (a) explicit
  repeated per-phase-step stepping for small durations; (b) positivity and conservation
  (`Σp = 1`) preserved under `matrix_power` for large cycle counts (e.g. `10**9`); (c)
  block-partition invariance; (d) the named regression cases — 1.2-cycle block (true midpoint
  at 0.6 cycles), odd- and even-integer-cycle blocks, arbitrary nonzero starting phase with
  wraparound.

### Gate S3 — Coupled phase integration
New module `arrhenius_fracture/crack_rebonding_v10230.py` (engine-integration layer):
- `WakePatch`, `RebondingWakeState` (`snapshot()`/`restore()` for transactional rollback), all
  advancement driven by Gate S1/S2's exact `propagate`/`strang_cycle_trajectory` primitives —
  no rounding, no block-scalar shortcuts.
- `RebondingWakeState.representative_cycle(engine, waveform, phase, dt_phase, k0, cycles, T_K)`:
  per active patch, builds the per-phase `Q_k` list from the frozen block-start mechanical
  state, computes `p_repr_start` (temporal-midpoint-centered, per the design above) and runs
  `strang_cycle_trajectory` to produce the length-`N_phase` `K_rebond_phase[k]` array used for
  `sig_cleave` — this is a **trial/provisional** computation only, used to bias the block's
  cleavage-hazard estimate; it does not itself commit any wake-state change.
- `RebondingWakeState.stage1_block_limits(p_start, Q_start, ...) -> list[float]`: cheap
  linearized, unit-corrected (seconds→cycles) candidate limits for `cycle_step_waveform`'s
  `limits` list.
- `RebondingWakeState.stage2_verify_block(p_start, phase_factors, k0, dt_candidate, ...) -> bool`:
  exact trial propagation + actual `Δp_B, Δp_C, ΔK_rebond` + action-consistency check; the
  caller bisects `cycles` and retries on failure (bounded iterations).
- `_rebuild_coupling`: `w(s)`, `H_b`, `K_rebond_max`, `K_rebond` from a given P/C/B state.
- `cleavage_stress_with_rebond(engine, Kvals, K_rebond_phase)`: the free function used at the
  single confirmed injection point in `kinetic_tip_cell.py`.
- Edit to `kinetic_tip_cell.py::cycle_step_waveform`:
  1. After `dt_phase` (line 430): if a rebonding state exists, run Stage 1 (cheap limits) and
     fold the results into the `limits` list construction (lines 439-447) *before* `cycles` is
     finalized at line 452.
  2. Once a candidate `cycles` passes Stage 2 verification (bisecting as needed), compute
     `representative_cycle(...)` to get `K_rebond_phase`, and replace line 435-437's
     `lam_c_phase` computation with the guarded `sig_cleave` branch (now phase-resolved, not
     scalar).
  3. **Do not commit any wake-state change here.** Whether this block's *provisional* state
     becomes the *committed* state depends on whether an event fires within it — see Gate S4's
     two-branch commit design. `cycle_step_waveform` only produces the provisional
     `K_rebond_phase` array used for this block's hazard evaluation; finalization happens
     downstream.
- Convergence tests: compare `K_rebond`/bonded-fraction trajectories under
  `N_phase, 2·N_phase, 4·N_phase` phase resolution and under forced block subdivision, assert
  agreement to a documented tolerance; explicit fail-closed test that a candidate block failing
  Stage 2 verification is bisected and re-checked, not silently accepted.

### Gate S4 — Transactional wake and rebonding-coupled event time
Two-branch finalization, replacing the earlier plan's single block-scalar commit:
- **No event fires in the block**: the block's full nominal duration was genuinely consumed as
  proposed, so finalize by committing `propagate(p_start, phase_factors, k0, dt_block)` (the
  exact full-block end state) as the new `_rebonding_state`, at the natural end of
  `cycle_step_waveform`. This is committed **once**.
- **An event fires**: the block's provisional `K_rebond_phase`/Stage-1/Stage-2 machinery was
  only ever used to *evaluate* the hazard that led to firing; it is **never committed**. Instead,
  the rebonding-coupled event-time root-finder (described above) runs — seeded by the existing
  uncoupled `dt_guess` from `StochasticHazardDiagnosticTipEngine`, iterating a phase-resolved
  trajectory and action-closure check until `dt_used` converges — and its **final iterate's
  trajectory** is what gets committed: pre-existing active patches advance from block-start to
  `t_e⁻` using that exact converged trajectory (never the nominal block state), then are
  translated by the accepted length, then the new patch is created fresh (never inheriting
  pre-event bonded state — see above). This is committed **once**, replacing whatever
  `kinetic_dt_consumed_s` would otherwise have supplied.
- A hard invariant, tested explicitly: exactly one of these two branches executes per block that
  advances rebonding state; the "no-event nominal `p_end`" and "event-time converged state" are
  never both committed, and neither is skipped.
- `commit_event(engine, accepted_length_m, event_index, converged_event_trajectory)` hooked into
  `persistent_site_cyclic_energy_gated_v10230.py::commit_energy_gated_event`, immediately after
  `advance = self.mpz.advance(length)` (line 262) — uses the committed `length`, never
  `pending["proposal_m"]`, and the root-finder's converged trajectory, never the raw
  `kinetic_dt_consumed_s` value alone (that value seeds the root-finder's predictor but the
  rebonding-coupled result supersedes it for wake-state purposes).
- Snapshot into `_energy_gate_pending["rebonding_state_before"]` plus
  `_energy_gate_pending["rebonding_block_start_phase_index"]` (phase provenance) at event-fire
  time; restore both in `restore_geometry_veto` (line 378-400) immediately before
  `self._energy_gate_pending = None`.
- Retirement beyond `wake_length_m` into the archival ledger; never touches crack geometry.
- Tests: 1 patch/accepted event, 0/rejected, truncated-length patch, multi-event ordering (each
  event individually inserts/translates before the next, using the correct resumed phase
  provenance — not batched, not restarting from phase zero), rollback restores full state
  (including phase provenance), retired+active length balance equals total created surface,
  checkpoint round trip (only when a state exists), root-finder convergence within the capped
  iteration budget across the named regression scenarios, event-time substantially earlier than
  the candidate block, two events in one physical cycle with the second correctly seeing the
  first's newly created patch and correct phase provenance, action-closure tolerance satisfied
  at the converged `dt_used`.

### Gate S5 — Feedback mode and energy-gate audit
- Qualify `HAZARD_ONLY_REBOND_SHIELD` as the only implemented mode, backed by the
  source-confirmed classification above (written into the lineage doc with file:line evidence).
- `COMMON_POSITIVE_LOCAL_K_REDUCTION` and `HAZARD_AND_ENERGY_GATE_COUPLED`: enum values exist
  and are documented as future interfaces; selecting either raises `NotImplementedError` at
  config-resolution time, before any engine construction.
- Explicit test: `continuum_gate_diagnostics`'s stress input is bit-identical with rebonding
  enabled vs. disabled, direct evidence that the energy gate is unaffected (not merely an
  architectural argument).

### Gate S6 — Disabled and zero-effect parity
- Config zero-effect limits (`chemistry_factor=0`, `restored_work_of_separation_J_m2=0`,
  `rebond_K_geometry_factor=0`, `bond_attempt_frequency_s=0`) ⇒ near-machine parity.
- **Disabled-mode parity is exact (`==`), not tolerance-based**: with no `_rebonding_state`
  installed, `sig_cleave is sig` (same array), so cleavage/emission/energy gate/RNG/geometry are
  bit-identical by construction, not merely numerically close.
- Checkpoint structural-equality test: disabled-run checkpoints must have identical JSON key
  sets to a pre-rebonding-module baseline checkpoint — not just equal values, equal *shape*.
- Canonicalized byte-for-byte audit-JSON comparison (exclude timestamps/paths/PIDs/wall-clock;
  compare everything else exactly) for one explicit disabled smoke run.

### Gate S7 — Software verifier
`scripts/verify_v10_2_30_crack_rebonding.py`, structured like the repo's existing qualification
verifiers. Static/pure-function tier (config defaults, rejection rules, Markov
conservation/positivity, exact-propagation agreement with brute force including the named
regression cases, `m_h=1` convention, contact-semantics labeling, solver-hash provenance
classification) plus a small-fixture-run tier (disabled parity, RB1 zero-effect at positive R,
patch-on-rejected-event absence, accepted-vs-proposed length, rollback restoration including
phase provenance, multi-event ordering, K_shield non-contamination, no double-counting,
emission unchanged, energy gate unaffected, no topological healing/negative advance, no
accelerated block admitted while enabled, no contact-proxy mislabeling, no Paris law inserted,
no double-install, event-time root-finder convergence, exactly-once-commit invariant). **The
verifier's terminal classification is always** `REBONDING_SOFTWARE_INTEGRATION_QUALIFIED`
**plus** `PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED` — it must never claim the module alters,
steepens, flattens, or arrests fatigue behavior, since no developed physical trajectory has
been run in this pass.

## Acceleration fail-closed gate (confirmed by source)

`sharp_front_v10_2_30_energy_gated_fatigue.py:181-183` is the only call site in the repo that
ever installs the DMD/Poincaré/projective accelerated engine
(`_coupled_commit.integrate_state_coupled_waveform = _high_cycle.integrate_state_coupled_waveform`,
restored at line 223). Immediately before this line, raise `RuntimeError` if rebonding is
enabled with `model_level != REBOND_OFF`, keeping the explicit phase-resolved integrator
installed for the entire run. Test: enabled rebonding + non-REBOND_OFF raises before
acceleration install.

## Documentation artifacts

- `docs/v10_2_30_crack_rebonding_equation_lineage.md`: every equation mapped to its implementing
  function and file:line, including the corrected `m_h=1` convention, the generic exact
  propagator and representative-cycle construction (replacing the rejected `round(cycles)` and
  the rejected block-midpoint-scalar approach), the rebonding-coupled event-time root-finder,
  the `HAZARD_ONLY_REBOND_SHIELD` classification with its supporting file:line evidence, and the
  resolved solver-hash provenance classification.
- `docs/v10_2_30_crack_rebonding_config_schema.json`: JSON Schema for the corrected
  `CrackRebondingControls` (callable-free, with the hard-rejected-in-this-pass fields documented
  as such rather than silently absent).

## Test files

| File | Covers |
|---|---|
| `tests/test_v10_2_30_crack_rebonding_config.py` | `.validate()` incl. all rejection rules, env loader |
| `tests/test_v10_2_30_crack_rebonding_periodic_propagation.py` | exact `propagate`/`strang_cycle_trajectory` vs. brute force; positivity/conservation at large cycle counts; block-partition invariance; 1.2-cycle block, odd/even integer blocks, arbitrary starting phase + wraparound |
| `tests/test_v10_2_30_crack_rebonding_markov.py` | conservation, positivity, corrected `m_h=1` convention, monotonicity, barrier-floor/log-domain diagnostics |
| `tests/test_v10_2_30_crack_rebonding_analytical_map.py` | two-state fixed point/iterate oracle; reference-action solver presets; hash freeze (callable-free) |
| `tests/test_v10_2_30_crack_rebonding_coupling_convergence.py` | phase-resolution convergence; block-partition invariance for the representative-cycle construction; Stage-2 fail-closed bisection, incl. a P→C→B activation with `dp_B/dt(0)=0` |
| `tests/test_v10_2_30_crack_rebonding_event_time_coupling.py` | root-finder convergence within the iteration cap; action-closure tolerance at converged `dt_used`; event substantially earlier than the candidate block; agreement with brute-force phase-by-phase integration |
| `tests/test_v10_2_30_crack_rebonding_wake_transaction.py` | patch creation/rejection/truncation/ordering/rollback (incl. phase provenance)/retirement balance/checkpoint round-trip; fresh patch never inherits pre-event bonding; exactly-once-commit invariant (event vs. no-event branch); two events in one physical cycle, second sees first's patch and correct phase provenance |
| `tests/test_v10_2_30_crack_rebonding_disabled_parity.py` | exact `==` parity incl. structural checkpoint-key-set equality |
| `tests/test_v10_2_30_crack_rebonding_contact_only_parity.py` | RB1 ≡ RB0 physically; nonzero diagnostics at negative R, zero at positive R |
| `tests/test_v10_2_30_crack_rebonding_zero_effect_limits.py` | four zero-effect configs ⇒ near-machine parity |
| `tests/test_v10_2_30_crack_rebonding_coupling.py` | `K_rebond` bounds; `K_shield` unchanged; no double-counting; emission unchanged (unconditional); energy gate unaffected (direct assertion against `continuum_gate_diagnostics`, bit-identical stress input) |
| `tests/test_v10_2_30_crack_rebonding_feedback_mode_scope.py` | `HAZARD_ONLY_REBOND_SHIELD` qualified; the other two modes raise `NotImplementedError` at config resolution |
| `tests/test_v10_2_30_crack_rebonding_install_idempotency.py` | single-authoritative-path install; double-install same-config no-op; different-config raises |
| `tests/test_v10_2_30_crack_rebonding_acceleration_gate.py` | enabled + non-REBOND_OFF raises before acceleration install |

## Verification (end-to-end)

1. `python -m pytest tests/test_v10_2_30_crack_rebonding_*.py -v` — full new suite green.
2. Regression run on touched production files:
   `python -m pytest tests/test_v10_2_30_partition_equivalence.py
   tests/test_v10_2_30_transactional_engine.py tests/test_v10_2_30_fixed_deltaK_entry.py -v`.
3. `python scripts/verify_v10_2_30_crack_rebonding.py --output runs/optional_crack_rebonding_ablation_v1/crack_rebonding_verification.json`
   exits 0, and its JSON explicitly carries `REBONDING_SOFTWARE_INTEGRATION_QUALIFIED` /
   `PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED` plus the resolved solver-hash provenance
   classification.
4. One explicit disabled-mode smoke run of the real CLI, canonicalized byte-for-byte audit-JSON
   diff against a pre-module baseline run.
5. `git status` clean in the new worktree; re-confirm
   `/private/tmp/v10230-reversible-energy-integration`'s controller/workers are still untouched
   and running.
6. Report back that Part X (the physical campaign) is deferred, and ask how the user wants to
   scope/schedule it given concurrent compute load.
