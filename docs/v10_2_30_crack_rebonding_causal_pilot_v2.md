# v10.2.30 Crack-Rebonding Causal Pilot V2 (Corrected)

## 1. Why v2 exists

The v1 pilot (`runs/crack_rebonding_causal_pilot_v1/`, worktree
`/private/tmp/v10230-crack-rebonding-causal-pilot` at commit `dce8cc0`) is
preserved as a diagnostic artifact, not a physical result. It reported a
2.3% timing shift as evidence of contact-conditioned crack rebonding, but
four defects mean that number cannot be trusted as a causal, physical
effect:

1. **Bond formation was not exactly gated on contact.** Under
   `SIGNED_K_COMPRESSION_PROXY`, contact should be `K_signed < 0` and only
   then. The v1 kinetics computed `sigma_comp_Pa` correctly as zero for
   `K_signed >= 0` (`contact_pressure`'s `max(-K, 0)`), but
   `bond_formation_rate`'s underlying Arrhenius rate still had a nonzero
   *unassisted-thermal* floor (`nu * exp(-G0/kT)` at zero work) that was
   never itself gated to zero outside contact. The result was a tiny but
   nonzero background formation rate at every phase of the cycle, not just
   during compression.
2. **The R=-0.95, Kmax=18 MPa*sqrt(m), f=1000 Hz reference protocol does
   not sample compression.** The real production engine's native cleavage
   hazard at this Kmax typically fires the next event within microseconds
   to a few percent of one 1 ms waveform period (confirmed empirically —
   see Section 4 below), so an event-created wake patch essentially never
   experiences a complete negative-K excursion before the next event.
3. **RB1 (CONTACT_PROXY_ONLY) did not take the same code path as RB0.**
   `patch_Q` already returned the exact zero generator for RB1 (so its
   Markov state genuinely never moves), but the *routing* to the
   rebonding-coupled bisection event-time root-finder
   (`solve_coupled_event_time`/`phase_resolved_action`) was gated on
   `cfg.enabled` alone in three places, not on whether `model_level` is
   kinetically active. RB1 (`enabled=True`, inert `model_level`) was
   therefore routed through the same numerically-approximate coupled path
   as RB2/RB3 even though there was no physics to couple, producing a
   spurious ~0.02%-to-0.9%-compounding timing drift relative to RB0 that
   v1 absorbed with a post-hoc 2% tolerance instead of fixing.
4. **RB2 vs RB1 conflated multiple effects.** Comparing RB2 directly against
   RB1 mixes genuine cohesive shielding with any residual Markov-propagation
   or event-time-localization differences. A zero-cohesion RB2 twin
   (`K_rebond_max=0`, otherwise identical) is required to isolate the
   cohesive causal effect specifically.

v1's status line:

```
PILOT_EXECUTION_AND_ARTIFACT_INTEGRITY_VERIFIED
CONTACT_CONDITIONING_NOT_IMPLEMENTED
REFERENCE_CONDITION_DID_NOT_SAMPLE_COMPRESSION
RB0_RB1_NUMERICAL_PARITY_NOT_STRICT
COHESIVE_CAUSAL_EFFECT_NOT_ISOLATED
PHYSICAL_REBONDING_EFFECT_NOT_QUALIFIED
```

## 2. V2-A: A_NATIVE provenance recovery

The historical `runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv`
(the intended source of the pilot's material row) does not survive: its
entire run-directory tree (`/private/tmp/v10230-reversible-energy-integration/
runs/A_native_plus_8PT_fatigue_v1/`) was found to be 1371 *empty* directories
with zero files anywhere in the subtree (consistent with macOS's periodic
`/tmp` file-cleanup sweeping files whose mtimes were 6+ days old while
leaving the directory skeleton behind), and the file was never git-tracked.

Recovery chain (implemented in `scripts/build_v2_A_native_provenance.py`,
output `artifacts/crack_rebonding_causal_pilot_v2/A_native_provenance.json`):

1. The qualified Candidate A source row (`candidate_id =
   v914_endurance_knee_0462`) survives, byte-identical, in the **immutable,
   git-tracked** v9.14 knee-search source registry:
   `Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search/runtime_inputs/
   v914/endurance_knee_global_300K_1024.csv`
   (sha256 `d05958c568610fab1fbb8795c6815f05ea6f36ef6cb1336dfecafa3ffc297ad1`,
   independently re-hashed and matching the hash recorded in a downstream
   audit file — see below).
2. That 45-field row is compared, field by field, against the same
   candidate's row in the 73-field MPZ-schema registry
   `PF-fracture-fatigue_codex_v10_2_30/arrhenius_fracture/data/materials/
   v10_2_31_endurance_knee_ABCD_registry.csv` (git-tracked in the main
   v10.2.x repo). Every one of the 45 common fields is byte-identical
   **except** `material_class` (`"endurance_knee"` -> `"DBTT"`, an explicit,
   disclosed relabel performed by that registry's own builder script,
   `scripts/build_v10_2_31_endurance_knee_registry.py`, not a physics
   change).
3. The 28 additional MPZ-solver-configuration fields present only in the
   73-field registry (bin count, forest-floor density, mobile-shield
   fraction, recovery-rate/legacy-feature toggles, etc — never any
   cleavage/emission/peierls/taylor barrier field, never `rho_source0_m2`
   or `c_blunt`) are explicitly recorded as disclosed
   `shared_spatial_constants` in that registry's own `.audit.json`, applied
   uniformly across a 4-candidate spatial-transfer study rather than
   silently invented. `n_bins_recommended` in this disclosed set is `80`,
   independently matching the mission's own required `mpz_n_bins=80`.
4. A_NATIVE is then built from this row by the audited
   `scripts/build_v10_2_30_A_native_plus_8PT_registry.py`'s own documented
   per-field algorithm (`composite = dict(A)`; the "A_NATIVE" variant has
   `vector=None`, so its PT-substitution loop never executes; only 5
   identity/descriptive fields are overwritten:
   `option_key`/`candidate_id` -> `"A_NATIVE"`, plus `role`,
   `mechanism_summary`, `validation_status`). The original builder's own
   `--A-registry`/`--common-physics` CLI inputs cannot be replayed today
   (both targets are gone and were never git-tracked), so this algorithm is
   reproduced directly in Python against the recovered row rather than by
   invoking the builder's CLI.

**Classification:**
`A_NATIVE_REGISTRY_DETERMINISTICALLY_RECONSTRUCTED_FROM_QUALIFIED_INPUTS`
— not a rediscovery of the original archived registry file. This is Tier B
of the mission's provenance hierarchy (Tier A quality for every
physics-critical field, since those are hash-verified against an immutable
untouched source; the MPZ-solver-configuration overlay is Tier B, since it
is disclosed but not independently proven identical to what the original,
now-gone launch registry held for this specific candidate).

The qualified production HEAD constant embedded in the 8PT builder
(`94871be15702e7fb85116b92af62c1226c61be42`) is confirmed to be a real
commit and an ancestor of this worktree's HEAD in both the main repo and
this worktree's own history.

### Manifest construction

`parameter_registry_v9111.py::select_option()` cannot be used for A_NATIVE:
it unconditionally calls `_validate_current_spatial_contract()`, which
hard-requires `Tref_K == 481.33` (A_NATIVE's row has `Tref_K = 300.0`).
Separately, `material_manifest.py::MaterialManifest.from_csv()` never reads
a `Tref_K` column at all — both `ExpFloorBarrier` objects it builds get the
module constant `TREF_K = 481.33` unconditionally — so the row's `Tref_K`
value is inert to the actual physics either way; the `select_option` gate
is a legacy campaign-eligibility check that simply doesn't apply here, not
a physics constraint we need to satisfy.

`arrhenius_fracture/a_native_engine_v10230.py::build_a_native_engine()`
therefore builds a `parameter_registry_v9111.SelectedResponseOption`
directly from the recovered row (bypassing only that inapplicable check and
the Stage-3 canonical-candidate fingerprint check, which also doesn't apply
to A_NATIVE), then reuses `write_compatibility_manifest()` /
`MaterialManifest.from_csv()` verbatim — byte-identical to the real
production conversion for every candidate that *is* routable through
`select_option`. Everything else (kernel family, persistent-site config,
generic solver mechanics constants `G_Pa`/`poisson`/`b_m`, numerical
substep controls) mirrors `tests/_crack_rebonding_engine_fixture.py::
build_real_engine`'s already-qualified recipe, since those are campaign-wide
solver conventions, not candidate-specific data.

Confirmed empirically: the real A_NATIVE engine at the reference protocol
(T=300K, R=-0.95, Kmax=18 MPa*sqrt(m), f=1000 Hz) fires its first event
after essentially **one full cycle** (`kinetic_dt_consumed_s ~ 0.999 ms`),
unlike the unrelated DBTT fallback candidate the v1 pilot's test fixture
used, which fires within microseconds. This is a materially better starting
point for compression-sampling than the fixture ever offered, and must be
re-checked with the actual RB1 preflight before any frequency-protocol
question is escalated (mission Section 9).

## 3. V2-B: exact contact gating and strict RB0/RB1 parity

### Contact gate

`crack_rebonding_v10230.py::patch_Q()` and
`crack_rebonding_kinetics_v10230.py::_integrate_A_on()` (the analytical
reference-action inversion) now both gate bond formation's cooperative
hazard exactly on `compressive_phase` (`K_signed < 0`):

```python
if diag["compressive_phase"]:
    k_CB = cooperative_hazard(lam_bond_raw, cfg.healing_cooperative_order, cfg.healing_correlation_time_s)
else:
    k_CB = 0.0
```

`K_signed > 0` and `K_signed == 0` now give `k_CB == 0.0` exactly, not a
floored/underflowed residual. `depassivation_rate()` already gated its own
`compressive_phase` correctly (no fix needed); the new `patch_Q` gate
applies the same rule to RB3's mechanical depassivation for free since both
flow through the same function.

New helper `crack_rebonding_v10230.rebonding_kinetics_active(cfg)`
centralizes the "does this `model_level` actually couple kinetics into the
generator" predicate (`enabled and model_level in {CLEAN_REVERSIBLE_REBOND,
PASSIVATION_GATED_REBOND}`), replacing three previously-inconsistent
`cfg.enabled`-only checks.

### Strict RB0/RB1 parity

Three call sites previously gated routing on `cfg.enabled` alone, so RB1
(`enabled=True`, inert `model_level`) was routed through machinery meant
only for kinetically-active model levels:

- `persistent_site_coupled_hazard_v10229.py::_phase_statistics` (built a
  phase-shifted, chronologically-offset signed-K array and used
  `cleavage_stress_with_rebond` for the block's average hazard even when
  `K_rebond` was provably always zero — a *second*, previously undiagnosed
  source of RB0/RB1 divergence beyond the one v1 found, since the
  phase-shifted quadrature samples different points on the same periodic
  curve than the baseline's fixed, unshifted phase array).
- `persistent_site_coupled_hazard_v10229.py::_commit_constant_segment`
  (stashed a rebonding block context and invoked `commit_no_event_block`'s
  exact propagation for every non-firing segment).
- `persistent_site_cyclic_energy_gated_v10230.py::commit_energy_gated_event`
  (routed every committed event through `_commit_rebonding_event`'s
  bisection root-finder — this is the mechanism v1 diagnosed and papered
  over with the 2% tolerance).

All three now key off `rebonding_kinetics_active(cfg)` instead. For RB1
specifically, `commit_energy_gated_event` now takes a dedicated branch that
still creates/translates the wake ledger (so RB1's own contact diagnostics
remain populated) via the plain `rebonding_state.commit_event(...,
pre_event_states=None, ...)` path, without ever constructing
`phase_resolved_action_fn` or calling `solve_coupled_event_time` — i.e.
RB1 now takes the literal same numerical path as `cfg is None` for every
quantity that matters (event time, action, state, event length, RNG
threshold stream), not a 2%-tolerance-away approximation of it.

The pilot module's `_PHYSICALLY_IDENTICAL_REL_TOL` (2% on
`waiting_time_s_this_event`) is no longer needed and must be replaced with
an exact-parity check in the v2 pilot's own causal-analysis rewrite (not
yet done — pending Section 8/10 work).

### Regression status

Full `tests/` suite (996 tests): 918 passed, 77 pre-existing failures, 1
skipped — every failure traced to `FileNotFoundError` on
`arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv`
(a different, unrelated missing-artifact issue, same "generated-file never
git-tracked" pattern as the A_NATIVE registry loss, confirmed unrelated to
any file this pilot touched). All 181 `crack_rebonding`-selected tests pass,
including two pre-existing unit tests
(`test_v10_2_30_crack_rebonding_coupling_convergence.py::
test_p_to_c_to_b_activation_not_missed_by_linear_stage1_estimate` and
`test_v10_2_30_crack_rebonding_causal_bonding.py::
test_live_kinetics_generate_nonzero_bonding_that_delays_the_next_event`)
that were updated because their specific timing parameters no longer
demonstrate the mechanism they were written for now that the (buggy)
background formation leak is gone — see inline comments at each edit for
the exact reasoning; both changes are minimal (a non-whole-cycle
`dt_candidate` in the first case, a documented unit-test-only frequency in
the second) and preserve each test's original intent.

## 4. V2-C: protocol/compression-sampling preflight

`scripts/run_v2_rb1_compression_preflight.py` drives the real A_NATIVE
engine through 8 accepted events via RB1 (CONTACT_PROXY_ONLY -- contact
diagnostics only, `patch_Q` exactly zero, so this cannot perturb event
timing) at the mission's reference protocol (T=300K, R=-0.95, Kmax=18
MPa*sqrt(m), f=1000 Hz, mpz_n_bins=80, seed=1720), then independently (no
engine-internal clock changes needed -- `FatigueWaveform.K_phase` is a pure
function of continuous phase `2*pi*f*t` with no resets, confirmed in
`fatigue_v1.py`) evaluates the signed K(t) waveform each event-created
patch actually sees between its creation and the next event, using only the
physically-summed `kinetic_dt_consumed_s` cumulative time each block
already reports.

**Result** (`artifacts/crack_rebonding_causal_pilot_v2/
preflight_protocol_selection.json`): of 7 post-first-event intervals,
**2 contain a complete negative-K (compressive) excursion** (event
0->1: 5.76 cycles elapsed, 2.95 ms of negative dwell; event 2->3: 6.31
cycles elapsed, 2.95 ms of negative dwell) -- exactly meeting the mission's
"at least two" bar. Inter-event spacing under RB1 ranges from 0.06 to 62
cycles (highly variable, stochastic), unlike v1's DBTT test-fixture
candidate, whose native hazard fired every event within a small fraction of
one cycle. **This refutes v1's documented defect #2 as a property of the
real A_NATIVE material/protocol combination** -- it was an artifact of the
v1 pilot's actual use of the (undocumented, out-of-scope) DBTT test-fixture
candidate rather than the true A_NATIVE row (see Section 2: the v1 pilot's
`build_engine` was `tests/_crack_rebonding_engine_fixture.py::
build_real_engine`, which unconditionally loads
`load_manifest(candidate_id="DBTT_A0003837")`, never A_NATIVE, and uses
`mpz_n_bins=8`, not 80).

**Status: `REFERENCE_PROTOCOL_SAMPLES_COMPRESSION`.** No frequency
escalation (mission Section 9's option A/B choice) is needed -- the
reference protocol is frozen as-is: T=300K, R=-0.95, Kmax=18 MPa*sqrt(m),
f=1000 Hz, mpz_n_bins=80, seed=1720.

## 5. V2-D: corrected 8-case pilot, zero-cohesion controls, decision

### Configurations

`arrhenius_fracture/crack_rebonding_causal_pilot_v2_v10230.py` builds six
distinct `CrackRebondingControls` (RB1, RB2-reversible-zero,
RB2-reversible-finite, RB2-persistent-zero, RB2-persistent-finite; RB0 is
`cfg=None`). The finite-cohesion reversible/persistent configs are
calibrated via `solve_reference_action_barriers` exactly as v1 did
(target dimensionless actions `A_on=1.0/A_off=1.0` and `A_on=10/A_off=0.1`
at the reference condition), with `K_rebond_max` target `Pi_K * Kmax =
0.05 * 18 MPa*sqrt(m) = 900 kPa*sqrt(m)`. Each zero-cohesion twin is
`dataclasses.replace(finite_cfg, restored_work_of_separation_J_m2=0.0)` --
identical in every other field by construction, not independently tuned.

The R=0.1 residual-bonding check (hard gate 2) is now a **closed-form
guarantee** rather than a calibration search: since `_integrate_A_on` gates
formation exactly on `K_signed < 0` and `R_POSITIVE=0.1` keeps
`Kmin = 0.1*Kmax > 0` at every phase, the predicted single-patch action at
R=0.1 is *exactly* 0.0 for any barrier choice -- confirmed programmatically
(`r_positive_action_ratios = {"reversible": 0.0, "persistent": 0.0}`) before
any trajectory runs, and empirically in gate 2 below.

### Run

`scripts/run_v10_2_30_crack_rebonding_causal_pilot_v2.py`, real A_NATIVE
engine, seed=1720, T=300K, f=1000Hz, mpz_n_bins=80, n_phase=80. All eight
trajectories (C0, C1, C2R, C3R, C2P, C3P, C4, C5) completed **uncensored**
with **7 accepted events each** (`runs/crack_rebonding_causal_pilot_v2/`,
gitignored; tracked summary/decision artifacts under
`artifacts/crack_rebonding_causal_pilot_v2/`).

### Hard gates (`scripts/analyze_v10_2_30_crack_rebonding_causal_pilot_v2.py`)

| Gate | Result |
|---|---|
| 1. C0/C1 event-time-and-length parity | **PASS** -- 0 mismatches across all 7 matched events (`rel_tol=1e-9` on both `accepted_length_m` and `waiting_time_s_this_event`; no 2% tolerance used). This checks those two fields specifically, not an unqualified full-state claim -- MPZ populations, emission ledgers, etc. were not separately diffed between C0 and C1. |
| 2. C5 exactly zero bond formation at R=0.1 | **PASS** -- `max_pB_post_commit`/`max_K_rebond` are exactly `0.0` at every one of C5's 7 events (C4 control likewise) |
| 3. Dynamically formed bonds from an event-created patch | **PASS** -- C3R and C3P both show genuine, dynamically-generated nonzero bonding (C2R/C2P, the zero-cohesion twins, correctly show `K_rebond` staying exactly 0 despite `p_B` evolving) |
| 4. Finite-cohesion delay in >=2 compression-containing intervals | **PASS** -- 2 unique post-first-event intervals (`event_0_to_1`, `event_2_to_3`) contain a complete negative-K excursion, each evaluated once under the reversible preset and once under the persistent preset (4 rows total, not 4 independent intervals), and every one of the 4 rows shows a finite-cohesion delay |
| 5. Only certified periodic-orbit/explicit actions admitted | **PASS** -- every `bulk_action_qualified` flag true across all 8 trajectories x 7 events |
| 6. Contact semantics label preserved | **PASS** -- `SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT` |
| 7. No physical Paris-slope inference | **PASS** -- single-Kmax pilot only, explicitly noted |

All 7 hard gates pass (`overall_gate_pass: true`).

### Causal quantities and the expansion threshold

`Delta t = t_finite_cohesion - t_zero_cohesion` at matched event indices
between each zero/finite twin pair (`interval_causal_analysis.csv`, 12
rows -- 6 unique post-first-event intervals x {reversible, persistent}
preset). The effect is small but **real, correctly signed, and remarkably
consistent**: every interval (compression-containing or not) shows
`t_finite > t_zero` by `log10(t_finite/t_zero)` in a narrow band of
**0.028 to 0.044 decades** across all 12 rows. The **maximum** ratio among
the 4 rows drawn from the 2 unique compression-containing intervals is
**0.0420 decades** -- below the mission's frozen expansion threshold of
**0.05 decades**.

```
classification: REBONDING_KINETICALLY_ACTIVE_BUT_MACROSCOPICALLY_SMALL
overall_gate_pass: true
expansion_threshold_exceeded: false
max_log10_ratio_abs_decade_in_compression_containing_intervals: 0.0420
multi_K_paris_slope_campaign_authorized: false
```

Per mission Section 12: bonds form, the causal mechanism is real and
correctly isolated from Markov-propagation/event-time-localization
artifacts (unlike v1's RB2-minus-RB1 comparison), but the cohesive
waiting-time effect at this Kmax/Pi_K combination does not clear the
prospectively-frozen macroscopic-significance bar. **Stop here -- no
multi-K matrix, no Part X.**

### Evidence hardening (analysis-only, no new physics)

The originally tracked artifacts (`trajectory_summary.json`,
`interval_causal_analysis.csv`) were insufficient to independently
reproduce the hard gates without the gitignored
`runs/crack_rebonding_causal_pilot_v2/trajectories.json`, and did not carry
several diagnostics the mission requested. Fixed, with no change to any
physical result:

1. **Portable event ledger.** `scripts/build_v2_event_ledger.py` extracts,
   from the raw `trajectories.json`, a tracked
   `artifacts/crack_rebonding_causal_pilot_v2/event_ledger.json` (+ a flat
   `.csv` export) containing every per-event field the gates need
   (`accepted_length_m`, `waiting_time_s_this_event`, `max_pB_post_commit`,
   `max_K_rebond_post_commit_Pa_sqrt_m`, `all_bulk_action_qualified`,
   `post_first_event_intervals`, etc.) for all 8 trajectories x 7 events =
   56 rows. `analyze_v10_2_30_crack_rebonding_causal_pilot_v2.py` and
   `verify_v10_2_30_crack_rebonding_causal_pilot_v2.py` were rewritten to
   read ONLY this tracked ledger -- the verifier no longer opens
   `runs/.../trajectories.json` at all, and keeps working even if that
   /private/tmp directory is gone.
2. **New per-event diagnostics**, added via non-invasive instrumentation
   of `crack_rebonding_v10230.py::phase_resolved_action` (three new
   diagnostics-only accumulators -- `max_K_rebond_Pa_sqrt_m`,
   `action_weighted_K_rebond_Pa_sqrt_m`, `action` -- that are never fed
   back into `total_action`/`p_by_patch`/`idx`, so they cannot change any
   decision the function's callers make) plus a few additional live-state
   reads already available in `run_trajectory` at commit time:
   - `pre_event_p_B` / `pre_event_K_rebond_Pa_sqrt_m` (already-existing
     live state, read immediately before commit).
   - `max_phase_resolved_K_rebond_Pa_sqrt_m` and
     `action_weighted_K_rebond_Pa_sqrt_m` (from the new
     `phase_resolved_action` diagnostics).
   - `cleavage_action` (the converged action from the same bisection call
     `dt_used`/`patch_states` are taken from -- not an arbitrary trial).
   - `finite_minus_zero_cleavage_action` (computed in
     `analyze.py::_matched_delay_rows` from the two trajectories' own
     `cleavage_action` at the matched event).
   - `hazard_threshold_action`, `hazard_event_index`, `engine_id` (RNG/
     threshold provenance).
   - `mpz_state` (`mpz_mobile_count`, `mpz_retained_count`,
     `mpz_emitted_total`, `mpz_escaped_total`, `mpz_recovered_total`,
     `r_eff`, `sigma_tip`, `mpz_total_K_shield_Pa_sqrt_m`), read directly
     off the real production engine's own `cycle_step_waveform` result.
   - `barrier_floor_saturation_diagnostics` (bond-barrier floor fraction,
     floored/saturated flags, cooperative-order regime), computed
     analytically from the frozen config at the reference protocol's
     deepest compression point -- a pure function of `(cfg, K_min,
     r_contact_m)`, needing no replay. `healing_cooperative_order=1.0` for
     both presets, so the cooperative-hazard regime is
     `single_hit_elementary` throughout (no gamma-incomplete saturation
     branch is exercised by this pilot).

   These required re-running the same 8 trajectories once more under the
   identical seed=1720/frozen configuration to populate the richer
   instrumentation. **Verified as a pure re-observation, not new physics**:
   every event's `accepted_length_m`, `waiting_time_s_this_event`,
   `max_pB_post_commit`, and `max_K_rebond_post_commit_Pa_sqrt_m` compares
   byte-identical against the original run's `trajectories.json` (0
   mismatches across all 56 events).
3. **Interesting finding surfaced by this pass**: `action_weighted_K_rebond`
   and `max_phase_resolved_K_rebond` sit at essentially the full
   `K_rebond_max` ceiling (900 kPa*sqrt(m)) in nearly every interval once
   any meaningful bonding has occurred, rather than tracking `p_B`
   continuously. This is a real, pre-existing property of the already-
   frozen wake geometry (`wake_weight_length_m=5e-7 m` is far smaller than
   a single event's accepted patch length, ~5e-6 m, so `H_b = p_B * w *
   length_m` saturates its `min(1.0, ...)` cap at a fairly small `p_B`) --
   not a defect introduced by this hardening pass, and not something this
   analysis-only pass changes.
4. **Wording corrections**: gate 1 is now explicitly labeled
   "event-time-and-length parity", and gate 4's rows are explicitly
   annotated with `interval_group_id` so it is clear the 4 compression-
   containing rows are 2 unique physical intervals x 2 kinetics presets,
   not 4 independent samples.

### Verification

`scripts/verify_v10_2_30_crack_rebonding_causal_pilot_v2.py`: depends ONLY
on tracked artifacts (no `--run-root`, no read of any gitignored
`runs/...` file). Independently rebuilds the bare A_NATIVE engine and
reproduces `frozen_configuration_sha256` from scratch, confirms the tracked
`event_ledger.json` points at that same hash, re-derives all gate-relevant
predicates directly from the ledger (bypassing the saved
`causal_decision.json` entirely), confirms
`mpz_n_bins=80`/`n_phase=80`/`seed=1720`/the reference protocol values, and
confirms DMD/Poincare acceleration, passivation, and topological healing
are all disabled. **`overall_pass: true`**, zero failed checks.

## 6. Second-seed replication (seed=1001723)

Bounded replication of ONLY the zero/finite-cohesion comparison
(`scripts/run_v2_second_seed_replication.py`): RB2 reversible zero/finite
and RB2 persistent zero/finite, reloaded verbatim from the committed
`frozen_configuration.json` (config hashes checked against that file
before running -- refuses to proceed on any mismatch), same A_NATIVE row,
Kmax=18 MPa*sqrt(m), T=300K, R=-0.95, f=1000 Hz, mpz_n_bins=80, n_phase=80,
cohesive scale, kinetics, event target, and numerical controls as the
original pilot. Only the hazard RNG seed changed: 1720 -> **1001723**. C0,
C1, C4, C5 were not re-run (out of scope for this replication). The frozen
0.05-decade threshold was not touched.

All 4 trajectories completed uncensored, 7 accepted events each. Two
different post-first-event intervals contain a complete compressive
excursion under this seed's own stochastic event timing --
`event_0_to_1` and `event_4_to_5` (not the same pair as seed 1720's
`event_0_to_1`/`event_2_to_3`, as expected for an independent threshold
draw) -- so gate 4's ">= 2 unique compression-containing intervals" bar is
independently met again.

**The result replicates the first seed's finding closely**: every one of
the 12 interval rows (6 unique intervals x 2 presets) shows `t_finite >
t_zero`, in an even tighter band of **0.0415 to 0.0437 decades** (seed
1720: 0.028-0.044). The maximum ratio among the 4 compression-containing
rows is **0.0436 decades** -- still below the 0.05-decade threshold.
Cumulative-waiting-time ratios (summed over all 7 events) agree closely:
**0.0421 decades** (reversible), **0.0422 decades** (persistent).

```
seed=1001723
n_unique_compression_containing_intervals: 2
max_log10_ratio_abs_decade_in_compression_containing_intervals: 0.0436
expansion_threshold_exceeded: false
cumulative: reversible=0.0421 decades, persistent=0.0422 decades
```

This is a second, independent, correctly-signed confirmation of a small,
real, consistently sub-threshold cohesive effect -- not a fluke of the
first seed's particular threshold draws. The terminal classification is
unchanged. Result recorded at `artifacts/crack_rebonding_causal_pilot_v2/
second_seed_replication.json`; raw run at
`runs/crack_rebonding_causal_pilot_v2_seed1001723/` (gitignored).

No further seeds, no multi-K Paris-slope matrix, and no threshold
adjustment were made or are proposed as a result of this replication.

## 7. Terminal classification

**`REBONDING_KINETICALLY_ACTIVE_BUT_MACROSCOPICALLY_SMALL`**
(mission completion-contract outcome C).

Remain unauthorized (per mission Section 14), and not attempted: full
multi-K Paris-slope campaign, passivation/repassivation sweep, frequency
sweep beyond the already-frozen reference protocol, DMD/Poincare
acceleration with rebonding, energy-gate coupling, topological crack
retreat, mesh-resolved contact claims, branch merge into the authoritative
production line.
