# V5 bounded hybrid directional-drive reuse report

## Scope and decision

This checkpoint replaces only the V5 downstream child's single-contour load
adapter.  The child now uses a candidate-owned hybrid drive: a valid nested
local J plateau, otherwise an isolated exact fixed-void virtual-extension
marginal G.  The result is supplied to the unchanged production cleavage-rate
adapter.  The FrontEngine radius law, barriers, material constants, renewal,
thresholds, and RNG are unchanged.

The scalar sent through the FrontEngine interface is
`sqrt(Eprime * max(G_kinetic, 0))`.  It is recorded exclusively as
`CANDIDATE_ENERGY_EQUIVALENT_NOT_WILLIAMS_ABSOLUTE_K`; this work makes no claim
to a continuum-qualified absolute Williams KI or KII.

## Read-only source audit

No source branch was merged or cherry-picked.  No V13 branching probability,
mark, material result, or campaign evidence was imported.

| Read-only source | Reviewed reusable surface | Reuse in this checkpoint |
| --- | --- | --- |
| `0238aae096aa29e79829d3c562383c38f2290ad6` | `LiveTopologyRequest`, `request_contour_definitions`, `evaluate_exact_topology`, `topology_fingerprint` in `live_topology_kernel_v11.py` | Existing nested-contour construction, geometric invalidity, adjacent-contour 15% plateau, and exact topology fingerprint. |
| `b3d0add6cbb0605adaa3e04006fe987961ad6452` | `DynamicExactTopologyProviderV12`; `candidate_tip_owners`, `observations_from_provider_by_front_candidate`, `require_same_tip_coupling`, `require_observation_state_contract`, `marginal_trial_origin`; `accepted_fem_state_identity`, `stress_field_identity`, `AcceptedTrialKey`, `ExactAcceptedTrialCacheV12`, `_front_observations`, `evaluate_candidate_marginal_kinetic_v12`, `production_directional_rate_adapter_v12` | Explicit front/candidate ownership, accepted/stress/topology identities, isolated exact marginal trials, ephemeral cache keys, and delegation to the unchanged production rate adapter. |
| `ca4abfe47765fcdaf0d266bfc0558ecd82d0c64e` | Accepted/trial isolation in `primary_race_production_v13.py` and private reservation ownership in `marked_topology_trial_v13.py` | Audit pattern only: a mechanics observation cannot reserve or duplicate the canonical first-passage clock. |

The V11 file at the first source SHA is byte-identical to the version inherited
by the starting PR head for the audited provider surface.  The new module
records these three source SHAs in every provider result.

## Exact call graph

Before this repair, downstream continuation followed:

```text
downstream_front_transaction(continuation=True)
  -> sharp_front_load_provider
     -> measure_directional_front_loads
        -> assemble_mechanics
        -> compute_J_integral(one contour, explicit branch_id)
  -> _complete_next_clock(front_load_rows=...)
     -> directional_sharp_front_rates
        -> directional K / sqrt(gamma_rel)
        -> sharp_front_constitutive_response
           -> FrontEngine.r_eff / sigma_tip / lambda_cleave / lambda_emit
```

The first missing edge was the edge from the explicit child load provider to
the branching solver's nested-contour validity and exact marginal fallback.

After this repair:

```text
downstream_front_transaction(continuation=True)
  -> sharp_front_load_provider(explicit branch_id)
     -> hybrid_directional_drive_provider(explicit candidates_by_tip)
        -> evaluate_exact_topology
           -> candidate-owned nested local-J contours
           -> cavity/crack/junction/specimen/support validity
           -> inherited adjacent-contour 15% plateau
        -> if invalid: _exact_marginal_trial
           -> isolated accepted state
           -> conform ephemeral endpoint
           -> apply_v12_production_trial_geometry(owning arm only)
           -> exact V12 support rebuild
           -> equilibrate_fixed_load_with_production_fem
           -> (Pi_base - Pi_trial) / delta_a
           -> discard exact trial cache entry
  -> _complete_next_clock(front_load_rows=...)
     -> directional_sharp_front_rates
        -> preview_production_cleavage_rate(G_kinetic)
           -> sqrt(Eprime*G_kinetic) / sqrt(gamma_rel), exactly once
           -> unchanged FrontEngine sigma_tip / cleavage rate
        -> unchanged FrontEngine emission/barrier/clock preview
  -> exactly one canonical first passage
  -> exactly one accepted topology transaction
```

## Ownership and isolation contracts

- The request must enumerate every active tip; every emitted row retains the
  exact `(tip_id, candidate_id)` pair.
- Scalar drive, accepted-state ID, stress-field-state ID, and topology
  fingerprint travel together.  A row detached from its explicit branch is
  rejected before rate evaluation.
- The optional cavity/free-surface inventory is empty for no-void states.  For
  void states it contains the actual one-owner mesh-boundary edges and ordered
  polygon, and therefore participates in the topology fingerprint.
- Every nested contour records cavity polygon/boundary, other crack/wake,
  junction, specimen boundary, finite-element support, and all invalid reasons.
  A cavity-intersecting contour can never control kinetics.
- A marginal trial extends only the owning branch/candidate.  Its void state,
  competition, process-engine state, thresholds, RNG, and accepted state are
  hashed before and after observation.  An uncertified trial returns
  `DIRECTIONAL_DRIVE_UNAVAILABLE`, zero kinetic drive/rate, and no clock change.
- Trial-cache identity includes accepted state, stress field, topology,
  candidate, front, process owner, delta-a, and local mesh level.  Every entry
  is destroyed immediately after observation.
- Local and marginal drives are never blended.  A valid local plateau remains
  authoritative even when a marginal is computed for the overlap check.
- `R_void` is absent from the FrontEngine and rate-adapter interfaces.  Void
  effects enter the child drive only through the actual FEM geometry and its
  re-equilibrated potential energy.

## Bounded qualification definition

The runtime marginal route is prospectively governed by the following fixed
contract.  It requires at least two positive delta-a values, at least two local
mesh levels, and the complete Cartesian product.  Every row must be a
`CERTIFIED_EXACT_FIXED_VOID_MARGINAL` observation with identical candidate,
front, process-owner, accepted-state, stress-field-state, topology, void, and
source identities.  The two finest mesh levels must agree within 10% at each
delta-a, and the two smallest delta-a values must agree within 10% on the
finest mesh.  A sign change is accepted only when every signed G lies within a
numerical-zero floor derived from the existing `1e-8` relative and `1e-12 J/m`
absolute topology-transaction energy tolerances.  Only after all gates pass may
the smallest delta-a at the finest mesh level control kinetics.  Otherwise the
provider returns an unavailable drive with zero G, zero energy-equivalent K,
and zero rate without changing accepted or process state.

```text
MARGINAL_G_MESH_RELATIVE_LIMIT = 0.10
MARGINAL_G_DELTA_A_RELATIVE_LIMIT = 0.10
MARGINAL_ENERGY_RELATIVE_ACCURACY = 1.0e-8
MARGINAL_ENERGY_ABSOLUTE_ACCURACY_J_PER_M = 1.0e-12
MARGINAL_DELTA_A_VALUES_M = (3.0e-5, 2.0e-5)
MARGINAL_LOCAL_MESH_LEVELS = (2, 3, 4)
```

The original dedicated clean-worker matrix contained exactly seven tests:

1. no-void root provider parity;
2. two-tip candidate/observation ownership;
3. far-void child with a valid local-J plateau;
4. overlap with both local J and exact marginal G;
5. near-void cavity-contour rejection, two delta-a values, two mesh levels,
   cache isolation, and checkpoint replay;
6. a reflected candidate-direction pair;
7. one retained real child continuation with one first passage and one topology
   action.

No static matrix, natural-seed ensemble, complete void trajectory, full A/B
campaign, V13 branching ensemble, fatigue calculation, or calibration belongs
to this checkpoint.

That original exact-head clean-worker attestation is complete:

- implementation SHA: `2faba01ed68d0e2474c88ecf98d2c75b0ce05eb9`;
- workflow run: `34567120033`;
- result: 7 passed / 0 failed / 0 errors / 0 skipped;
- artifact ID: `10186464166`;
- artifact digest:
  `sha256:fead9b7e97144cf3dc111a3eb23634942f3ebbfed6b458bac8f8db26124a1d2a`.

## Final runtime hardening

The hardening implementation is
`b58997bdb18cf4e9a32c251c073115d8b405bb27`, a normal descendant of the
prospective contract commit `8f55a6f`.  It makes the fixed runtime marginal
family 30 and 20 micrometres across local refinement levels 2, 3, and 4.  All
six signed rows remain in diagnostics; only levels 3 and 4 control the mesh
gate, and only 20 micrometres at level 4 becomes authoritative after every
gate passes.

The provider now receives an explicit front-to-process-owner map.  The owner
must exist in the accepted process registry and match both the branch process
region and any checkpoint owner map.  Trial cache and output identities retain
both front and owner, including the shared-owner/multiple-front case.

Only `MarginalDriveNotCertified` is converted into an unavailable drive.
Recognized endpoint-domain and V12-support failures are translated at their
narrow call boundaries.  Other runtime errors, value errors, malformed data,
identity inconsistencies, mutation detections, and cache failures propagate.

The new bounded matrix contains the prior seven provider/continuation tests and
eight hardening tests covering converged, mesh-nonconverged,
delta-a-nonconverged, incomplete/uncertified, and sign-inconsistent families;
distinct and shared process owners; fail-before-solve owner rejection; and the
expected/unexpected exception taxonomy.  The retained real child continuation
appears exactly once.  Local bounded verification passed all 14 non-continuation
nodes together and the one retained continuation separately.  The clean-worker
record will be appended after the dedicated exact-head workflow completes.

## Terminal classifications

The implementation classifications, pending the final dedicated exact-head
clean-worker attestation, are:

```text
DIRECTIONAL_CRACK_DRIVE_PROVIDER = PASS_HYBRID_LOCAL_J_AND_CONVERGED_EXACT_MARGINAL_G
DIRECTIONAL_K_KINETIC_INTERPRETATION = ENERGY_EQUIVALENT_CANDIDATE_LOAD
VOID_INTERACTION_IN_DIRECTIONAL_DRIVE = PASS_EXACT_FEM
PRODUCTION_RUNTIME_MARGINAL_CONVERGENCE_GUARD = PASS
PROCESS_OWNER_IDENTITY = PASS_CANONICAL_PRODUCTION_OWNER
EXPECTED_NONCERTIFICATION_EXCEPTION_TAXONOMY = PASS
UNEXPECTED_PROGRAMMING_EXCEPTIONS_PROPAGATE = PASS
EXISTING_R_TIP_LAW_REUSED_BY_DOWNSTREAM_CHILD = PASS
DIRECT_R_VOID_SUBSTITUTION = ABSENT
CLASSICAL_ABSOLUTE_WILLIAMS_KI_KII = NOT_CLAIMED
CONFORMING_TIP_PATCH = NOT_REQUIRED_FOR_KINETIC_EVENT_SELECTION
```

A conforming-tip patch remains a possible separately scoped reporting route for
classical absolute KI/KII.  It is not a kinetic-event-selection prerequisite.
