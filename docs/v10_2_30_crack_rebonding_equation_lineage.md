# Crack-Rebonding Equation Lineage (v10.2.30)

Maps every equation in the mission specification to its implementing function
and file:line. Companion documents:
`docs/v10_2_30_crack_rebonding_approved_plan.md` (the reviewed design),
`docs/v10_2_30_crack_rebonding_plan_provenance.md` (plan hash/approval
record), `docs/v10_2_30_solver_hash_provenance.md` (solver-hash provenance),
`docs/v10_2_30_event_time_mechanism_investigation.md` (the confirmed
`normalized_progress_rate` relationship this module's event-time
root-finder is built on).

## Contact proxy (Part II of the mission)

| Equation | Implementation |
|---|---|
| `K_s(phi) = Kmax[(1+R)/2 + (1-R)/2 cos(phi)]` (full signed, unclipped) | `FatigueWaveform.K_phase(phase)` with `closure_clip=False` (`fatigue_v1.py:201-208`), invoked via `dataclasses.replace(waveform, closure_clip=False)` in `kinetic_tip_cell.py::cycle_step_waveform` |
| `sigma_comp,j(phi) = min(eta_p*[-K_s]_+/sqrt(2*pi*(r_contact+s_j)), cap)` | `crack_rebonding_kinetics_v10230.contact_pressure` |
| `sigma_open,j(phi) = min([K_s]_+/sqrt(2*pi*(r_contact+s_j)), cap)` | `crack_rebonding_kinetics_v10230.opening_stress` |
| `r_contact = max(r_eff, r_contact_min)` | computed at the `cycle_step_waveform` call site: `max(self.r_eff(), rebonding_state.cfg.contact_radius_min_m)` (`kinetic_tip_cell.py`) |
| Contact semantics label | `CONTACT_SEMANTICS_LABEL = "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT"` (`crack_rebonding_kinetics_v10230.py`), stamped onto the engine as `engine.contact_semantics` in `install_crack_rebonding` |
| `RESOLVED_GAP_TRACTION` (future interface) | `ContactModel.RESOLVED_GAP_TRACTION` enum value exists; `.validate()` always rejects it in this pass (no oracle contract implemented) |

## Interfacial kinetics (Part III)

| Equation | Implementation |
|---|---|
| `lambda_bond_raw = chi_chem*nu_bond*exp[-(G_bond0 - sigma_comp*V_bond/e)/(kB*T)]`, exact zero at `chi_chem=0` | `crack_rebonding_kinetics_v10230.bond_formation_rate` (early-returns `(0.0, ...)` before any log/exp when `chemistry_factor<=0`) |
| Log-domain, floor-bounded evaluation, `G_eff = max(G_floor, G0-work)` | `_log_domain_rate` (private helper, `crack_rebonding_kinetics_v10230.py`); used by all four rate functions |
| `k_CB = P(m_h, x_h)/tau_h`, `x_h = lambda_raw*tau_h`, exact `m_h=1` recovery | `cooperative_hazard` -- mirrors `lambda_cleave`'s own `m<=1` special case (`unified_front.py:109-114`) exactly, so `m_h=1` returns `lambda_raw` verbatim rather than the small-x limit of the unqualified gamma expression (a correction from round-1 review) |
| `k_BC = nu_off*exp[-(G_off0 - sigma_open*V_off/e)/(kB*T)]`, zero during compression | `bond_rupture_rate` (`compressive_phase=True` forces `0.0`) |
| `k_PC = C_j*nu_dep*exp[...]`, `C_j = 1 iff K_s<0` | `depassivation_rate` |
| `k_CP = E_j*nu_rep*exp[-G_rep0/(kB*T)]`, `E_j = 1 iff K_s>0` | `repassivation_rate` |
| `Q` conservative generator, columns sum to zero | `build_Q(k_CB, k_BC, k_PC, k_CP)` -- state order `[P, C, B]` |
| `p(t+dt) = expm(Q*dt) p(t)` | `advance_markov` |
| Zero generator for `REBOND_OFF`/`CONTACT_PROXY_ONLY` | `crack_rebonding_v10230.patch_Q` gates on `cfg.model_level`; RB1's diagnostics are computed separately via `contact_diagnostics` and never fed back |

## Generic exact phase-aware propagator (round-2 review correction)

Replaces a rejected block-midpoint-scalar `K_rebond` and an incorrect
`n//2, eta/2` formula.

| Concept | Implementation |
|---|---|
| `M(k0, dt) = expm(Q_k' * f * dt_phase) * M_r(k0) * matrix_power(M_cycle(k0), m)` | `propagate` (`crack_rebonding_kinetics_v10230.py`) |
| One-cycle cyclic-order product starting at `k0` | `_partial_product` (private helper) |
| Exact for the piecewise-constant-rate phase discretization; `O(log m)` for `m` whole cycles via `numpy.linalg.matrix_power` | `propagate`, tested against brute-force stepping and the named regression cases (1.2-cycle block, odd/even integer blocks, arbitrary starting phase + wraparound, billion-cycle conservation) in `tests/test_v10_2_30_crack_rebonding_periodic_propagation.py` |
| Strang-split per-phase-bin representative cycle: `p_{k+1/2}=expm(Q_k dt/2)p_k`, `K_rebond(k+1/2)=f(p_{k+1/2})`, `p_{k+1}=expm(Q_k dt/2)p_{k+1/2}` | `strang_cycle_trajectory` (pure) / `crack_rebonding_v10230.representative_cycle_K_rebond` (engine-facing, produces the length-`N_phase` array used for `sig_cleave`, never a single scalar) |

**VHCF/low-K performance (round-3 review, fixed after being flagged as a
known limitation in an earlier pass)**: `crack_rebonding_v10230.phase_resolved_action`
now resolves a finite transient exactly (cycle by cycle, since each cycle's
`K(phase)` sequence repeats identically, making the per-cycle map
state->state and state->action time-invariant), detects convergence to a
periodic orbit via a per-cycle action/state-change tolerance, and represents
any remaining whole cycles as `remaining_cycles * A_c` with the wake state
advanced exactly via `propagate`'s `O(log n)` `matrix_power`. Critically,
runtime is bounded **unconditionally** once at least one transient cycle has
resolved: an earlier version fell back to exact bin-by-bin stepping for the
remainder whenever strict convergence wasn't detected within the transient
budget, which could itself become an `O(n_cycles)` computation for a
slowly-relaxing configuration -- caught by a test that genuinely hung on a
near-billion-cycle interval before this was fixed. The corrected design
always uses the last resolved transient cycle's action as the bulk
representative once the budget is spent, whether or not the strict
tolerance was met; `tests/test_v10_2_30_crack_rebonding_vhcf_performance.py`
verifies both a billion-cycle interval completes in well under a second and
that the bulk result agrees with exhaustive exact stepping to a tight
tolerance once the transient budget genuinely spans the configuration's
relaxation time (one test case's relaxation time is ~200 cycles at 1000 Hz,
not 30 -- confirming a real transient budget matters for physically
slow-relaxing configurations, not merely a test artifact). Also fixed along
the way: an initial version compared the bin-count `n_full` directly against
`bulk_cycle_threshold` (meant as a cycle count), incorrectly triggering bulk
mode for intervals as short as ~2-3 cycles at typical `n_phase` -- corrected
by explicitly separating `n_full` into whole cycles (`n_cycles_full`) and a
leftover partial-cycle bin remainder before applying the threshold.

## Cohesive wake coupling (Part IV)

| Equation | Implementation |
|---|---|
| `w(s) = exp(-s/L_w)/(L_w*(1-exp(-L_h/L_w)))` | `crack_rebonding_v10230.wake_weight` |
| `H_b = min(1, sum_j p_B,j*w(s_j)*l_j)` | `RebondingWakeState.rebuild_coupling` / `representative_cycle_K_rebond` (per-phase-point variant) / `phase_resolved_action` (per-bin variant) |
| `K_rebond_max = eta_K*sqrt(E'*G_rebond_max)` | same three call sites |
| `E' = 2G/(1-nu)` (**isotropic plane-strain** reduced modulus, derived from the engine's own shear modulus/Poisson ratio rather than the diagnostic-only global material observer used elsewhere for `E'`; NOT a general anisotropic effective modulus -- acceptable for this mechanism-exploration pass per the mission's explicit `Pi_K` dimensionless-control framing, but a future resolved-anisotropic PF/FEM contact implementation must obtain `E'` from that solver's own energy-release-rate convention rather than reconstruct it independently here) | `crack_rebonding_v10230.reduced_modulus_Pa` |
| `K_rebond = K_rebond_max*H_b` | same three call sites |
| `sigma_c = [K+ - K_shield - K_rebond]_+/sqrt(2*pi*r_eff)` (HAZARD_ONLY_REBOND_SHIELD) | `cleavage_stress_with_rebond` -- a free function replicating `sigma_tip`'s exact arithmetic (`unified_front.py:84-89`) with one extra subtraction, called only from the single guarded injection point in `kinetic_tip_cell.py::cycle_step_waveform`; `unified_front.py`/`separated_source_tip.py` themselves carry **zero source diff** |

## HAZARD_ONLY classification, confirmed by source (not asserted)

- The only injection point is inside `kinetic_tip_cell.py::cycle_step_waveform`
  (base implementation, confirmed the sole place `sig`/`lam_e_site`/`mu_emit`
  and `lam_c_phase`/`mu_c` are computed -- every override in the real MRO
  chain, `campaign_calibrated_tip.py:305`, `continuum_source_tip.py:376`,
  `separated_source_tip.py:112`, and the
  `HazardEnergyGatedPersistentSiteCyclicTipEngine`/`Corrected...` mixins, is a
  confirmed diagnostic-only pass-through via `super().cycle_step_waveform(...)`).
- `sig` (drives emission via `lam_e_site`/`mu_emit`/`avg_sig`/`stress_override`)
  is never touched; only a second, separate `sig_cleave` array feeds
  `lam_c_phase`/`mu_c`.
- The energy-admissible event length is derived exclusively by
  `continuum_gate_diagnostics` (`hazard_energy_event_gate_v10230.py:280-326`),
  which calls `engine.sigma_tip(K)` directly at line 294 -- a method never
  touched by this module. Rebonding can therefore only delay/hasten first
  passage, never change the length of an already-triggered event.
- `COMMON_POSITIVE_LOCAL_K_REDUCTION` and `HAZARD_AND_ENERGY_GATE_COUPLED`
  are real, named enum values that raise `NotImplementedError` at
  `.validate()` time -- documented future interfaces, not silently
  half-built code paths.

## Transaction and event-time coupling (Part V, round-2 review correction)

The reviewed plan required that the rebonding-coupled event-time root-finder
recompute the *phase-resolved* action for every trial duration, not treat
the existing `_integrate_coupled`'s block-constant `lambda_override` as the
physical model. Since `_integrate_coupled` is *stateful* (mutates `self.B`,
consumes RNG on firing), it cannot be safely re-invoked with different trial
rates -- the design instead uses a **pure closed-form stand-in** derived from
the confirmed relationship in `stochastic_hazard_tip.py:58-60`:

```
normalized_progress_rate(lambda_c, threshold_action) = lambda_c / threshold_action
```

i.e. `dB/dt = lambda_c/threshold_action` for a constant rate, giving the exact
closed form `dt_consumed = (1 - B_start)*threshold_action/lambda_avg`. This
is what `commit_energy_gated_event`'s `_commit_rebonding_event` helper uses
as `integrate_coupled_fn` when calling `solve_coupled_event_time` -- see
`docs/v10_2_30_event_time_mechanism_investigation.md` for the full derivation.

| Concept | Implementation |
|---|---|
| `Delta B_c(t) = integral_0^t lambda_c[K(tau), K_rebond(tau)] dtau`, phase-resolved, no block-constant shortcut | `crack_rebonding_v10230.phase_resolved_action` |
| Root-find `dt_used` such that `B_start + DeltaB_c(dt_used) = B_threshold`, bisection on `dt` (stable, since cleavage rates are non-negative so the action is monotone in `dt` -- replaces an earlier fixed-point-on-lambda scheme shown by a failing synthetic test to diverge for super-linear time-dependence) | `crack_rebonding_v10230.solve_coupled_event_time` |
| Two-stage block-size limiter: cheap linearized Stage 1 candidate (unit-corrected seconds->cycles), exact Stage 2 verification with fail-closed bisection | `stage1_block_cycle_limits` / `stage2_verify_block`; folded into `kinetic_tip_cell.py::cycle_step_waveform`'s existing adaptive `limits` list before `cycles` is finalized |
| Accepted-length-only patch creation, fresh patch never inherits pre-event bonding, exactly-once commit (no-event branch commits full-block state in `cycle_step_waveform`; fired branch defers to `commit_energy_gated_event`) | `RebondingWakeState.commit_event` / `commit_no_event_block`; wired in `kinetic_tip_cell.py::cycle_step_waveform` and `persistent_site_cyclic_energy_gated_v10230.py::_commit_rebonding_event` |
| Transactional rollback | `RebondingWakeState.snapshot`/`restore`; `_energy_gate_pending["rebonding_state_before"]`, restored in `restore_geometry_veto` |

**Chronological phase continuity (round-3 review correction, supersedes an
earlier "no cross-block phase state needed" analysis)**: `FatigueCycleHazardController
._phases()` (`fatigue_v1.py:288-290`) is a pure function of `n_phase` alone
-- it does not track absolute elapsed time, and the existing (pre-rebonding)
cleavage/emission channels re-sample the same fixed relative-phase grid
every block, unchanged. An earlier draft of this document concluded from
that fact that the rebonding wake also needed no persistent cross-block
phase tracking (`k0=0` every block). Review correctly identified this as
insufficient: the mission's compression-formation/opening-rupture asymmetry
is only physically meaningful if a patch created mid-cycle evolves through
the *actual remaining fraction* of that cycle, not a freshly restarted
archetypal cycle -- discarding chronological history at every block boundary
would silently convert "same-cycle formation and survival" into a
cycle-averaged closure without saying so.

The implemented fix keeps the existing cleavage/emission phase treatment
completely untouched (satisfying "existing mechanics unchanged") while
giving the wake its own genuinely continuous chronological clock:
`RebondingWakeState.elapsed_time_s`, a persistent scalar (mod the waveform
period) advanced by the caller after every committed block or event.
`crack_rebonding_v10230.chronological_phase_offset_rad(elapsed_time_s,
period_s)` converts it to a phase offset added to the fixed relative-phase
sample array *only* when building `K_signed_phase` for the wake's own
`representative_cycle_K_rebond`/Stage 1/Stage 2/event-time-root-finder
context (`kinetic_tip_cell.py::cycle_step_waveform`) -- `Kvals`/`sig`
(driving cleavage/emission) are computed from the unshifted array exactly
as before, at the same line, so this is purely additive.

The clock is advanced in two places, matching the exactly-once-commit
invariant: the no-event branch in `cycle_step_waveform` advances it by
`dt_block`; `_commit_rebonding_event` advances it by the **converged**
`dt_used` from the event-time root-finder (not the raw uncoupled estimate),
so an event that fires early or late within a candidate block correctly
shifts the clock by the true elapsed time, not the nominal one. Included in
`RebondingWakeState.snapshot`/`restore` (hence transactional rollback) and
`serialize_rebonding_checkpoint`/`restore_rebonding_checkpoint`. Tested in
`tests/test_v10_2_30_crack_rebonding_live_engine_smoke.py` (clock advances
monotonically across real blocks and never resets; the resulting phase
offset genuinely differs block-to-block, confirming the shift is not a
no-op) and `test_chronological_phase_state_survives_snapshot_restore`.

## Analytical reductions (Part VI)

| Equation | Implementation |
|---|---|
| `b_n^c = 1-(1-b_n)e^{-A_on}`, `b_{n+1} = e^{-A_off}[1-(1-b_n)e^{-A_on}]` | `two_state_iterate` |
| `b* = e^{-A_off}(1-e^{-A_on})/(1-e^{-(A_on+A_off)})`, `P_survive = (1-e^{-A_on})e^{-A_off}` | `two_state_fixed_point` |

## Configuration (Part VII)

`CrackRebondingControls` (`crack_rebonding_kinetics_v10230.py`) implements
the full field list with `.validate()` enforcing every rejection rule from
the mission plus the round-2-review additions: `minimum_load_hold_s` must be
exactly `0.0`, `stochastic_healing_enabled` must be `False`,
`RESOLVED_GAP_TRACTION` always rejected regardless of oracle identifiers,
`resolved_gap_oracle` (a `Callable`) removed entirely from the frozen/hashed
config and replaced with three stable string identifiers
(`contact_oracle_id`/`_version`/`_source_hash`), `initial_wake_mode` split
into `initial_precrack_wake_mode` (governs only the pre-existing crack) and
`fresh_surface_clean_fraction` (governs every newly created patch,
unconditionally).

## Reference-action parameter generator (Part VIII)

| Concept | Implementation |
|---|---|
| Monotone solve for `G_bond0`/`G_off0` given target `A_on_ref`/`A_off_ref` | `solve_reference_action_barriers` (`scipy.optimize.brentq` after bracket expansion) |
| Four named presets | `REFERENCE_ACTION_PRESETS` |
| Frozen, hashed parameter set | `freeze_reference_action_preset` (`sha256` of canonical sorted JSON) |

Not yet exercised against live solver preflights -- that belongs to the
deferred Part X physical campaign.

## Solver-hash provenance

See `docs/v10_2_30_solver_hash_provenance.md` for the full investigation.
Classification: `MISSION_SOLVER_HASH_UNREPRODUCED_BUT_BASE_COMMIT_VERIFIED`.

## Real regressions found and fixed while wiring this module

- `reduced_shared_state_v1023.py::write_shared_result` called
  `json.dumps(payload, indent=2)` with no enum handling; adding
  `rebonding: CrackRebondingControls` to `SharedReducedConfig` (whose
  `asdict()` output now includes nested enum fields) broke it. Fixed with a
  `default=_json_default` handler converting `Enum` to `.value`.
