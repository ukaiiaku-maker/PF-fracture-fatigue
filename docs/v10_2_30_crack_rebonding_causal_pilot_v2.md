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

Pending.
