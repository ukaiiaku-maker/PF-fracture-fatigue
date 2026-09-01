# PF current-source branching corrected V5.2.1 decision

Permanent interpretation boundary: `CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS`

This is an analysis-only record. It supersedes only the morphology interpretation in the immutable V5.2 decision; it does not modify the V5.2 raw results, terminal failure, mechanics, parameters, or provenance.

## Superseding scientific decision

```text
pair_terminal_result:
CORRECTED_THETA40_REPLAY_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE

atomic_topology_transaction_capability:
DEMONSTRATED

corrected_process_state_branch_birth:
DEMONSTRATED

corrected_morphology_capability:
CORRECTED_CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED_BEFORE_ENVELOPE_STOP

bounded_300um_pair_completion:
NOT_COMPLETED_SIGNED_KERNEL_ENVELOPE

final_independent_tip_mechanics:
UNQUALIFIED_ONE_OF_TWO_DAUGHTER_LOCAL_J_CONTOURS_VALID_AT_STOP

cluster_handoff_status:
UNRESOLVED_HANDOFF_NOT_REQUIRED_AT_STOP

predictive_branching_physics_validated:
false
```

The terminal result and morphology result answer different questions. The former records that neither bounded trajectory completed the 300 micrometre stopping target. The latter records what the corrected source demonstrated before the independent input-coverage stop. Three hundred micrometres is a stopping target and hard-negative ceiling, not a twelfth morphology-success gate.

## Eleven authoritative morphology criteria

1. **directional_first_passage — PASS.** `{"completion_times_s":[2410.800011135129,2410.800011226723],"correlation_time_difference_s":9.159430192084983e-08,"pending_event_ids":["cleave:cleavage:(010):694b159e5c3c21b95300#event:0000000000000014","cleave:cleavage:(100):77278fedde405b16bcd0#event:0000000000000010"],"step":369}`
2. **positive_signed_directional_J — PASS.** `{"local_J_valid":[true,true],"signed_directional_J_J_per_m2":[23327.3760714061,25791.839235003165]}`
3. **energy_acceptance — PASS.** `{"net_energy_margin_J_per_m":0.14740394023225453,"released_energy_J_per_m":0.14741141338225816,"total_cost_J_per_m":7.473150003639843e-06}`
4. **valid_daughter_topology — PASS.** `{"accepted_two_arm_birth_count":1,"daughter_ids":["b0fa22bb892937f8","b7d5efe0822562b9"]}`
5. **nonstub_propagation — PASS.** `{"daughter_event_counts":{"b0fa22bb892937f8":6,"b7d5efe0822562b9":57},"daughter_growth_um":{"b0fa22bb892937f8":30.0,"b7d5efe0822562b9":285.0}}`
6. **cluster_bookkeeping — PASS.** `{"cluster_bookkeeping_valid":true,"final_cluster_unresolved":true}`
7. **conditional_handoff — PASS.** `{"handoff_length_um":50.0,"handoff_required_at_stop":false,"independent_local_J_flags":[false,true],"short_arm_below_handoff_length":true,"short_arm_growth_um":30.0}`
8. **no_front_cap_binding — PASS.** `{"maximum_fronts":2,"no_front_cap_veto_recorded":true}`
9. **no_immediate_retirement_or_reversal — PASS.** `{"all_committed_segments_forward_in_x":true,"daughter_statuses_at_stop":{"b0fa22bb892937f8":"active","b7d5efe0822562b9":"active"},"no_bridge_or_reconnection_veto_recorded":true}`
10. **state_hazard_rng_wake_geometry_ledger_closure — PASS.** `{"hazard_event_ordinal_closure":{"control_max1":{"cleave:cleavage:(010):694b159e5c3c21b95300":true,"cleave:cleavage:(100):77278fedde405b16bcd0":true},"enabled_max2":{"cleave:cleavage:(010):694b159e5c3c21b95300":true,"cleave:cleavage:(100):77278fedde405b16bcd0":true}},"hazard_rng_state_serialized":true,"one_process_update_per_interval":{"control_max1":true,"enabled_max2":true},"population_and_signed_ledgers_pass":true,"renewal_and_geometry_closure_pass":true}`
11. **prebranch_matched_pair_identity — PASS.** `{"action_rows_each":[23,23],"directional_rows_each":[736,736],"identical_through_step":368}`

All eleven criteria pass. In addition, scalar K and tensor data share the same physical tip in all 1467 audited accepted steps, including all 438 postbirth steps. The corrected process-state branch birth is therefore demonstrated, not merely reproduced provisionally.

## Coverage-planning diagnosis

The old planner reproduced

`300/cos(40 deg) + 18.378720764999 = 410.0009075646826 micrometres`,

then selected a 415 micrometre measured endpoint. That bound assumed one fixed +40 degree path and the inherited threshold-scaled maximum event reward. The corrected topology instead owns fixed 5 micrometre accepted arm geometry and permits +40 and -50 degree candidates.

The control accepted 84 events: 44 at +40 degrees and 40 at -50 degrees. It accumulated 420 micrometres shared process advance while reaching only 297.087299 micrometres projected extension. Either next candidate would cross 300 micrometres, but selection required evaluating the 420 micrometre pre-event state, five micrometres beyond the family endpoint.

For a single front, conservative coverage must use the smallest positive projection in the actual candidate inventory. For unresolved multiple fronts it must separately track shared process extension, total network geometry, and maximum forward reach.

At the exact enabled terminal topology, the long arm can take one more event while remaining below target and the short arm can take 63. Therefore the latest possible pre-event shared-extension query is 740 micrometres. A 745 micrometre measured endpoint adds one fixed topology quantum. This is a bound only for this frozen two-front topology with no remaining branch capacity; it is not universal.

## Disposition

The corrected current-source branching morphology capability is complete for scientific purposes. The 300 micrometre pair remains formally incomplete. No PF or FEM calculation was run for V5.2.1. Extending the measured family is new deterministic FEM qualification and remains authorization-gated.

Provenance: predecessor result `ff6bc653298b13f6da12c6b5722684b9f115afde`; execution source `e2aff736afe0e1d2d1b600c25743de317a71c7ba` / tree `14a953037a44a25288c41d317a0ae0c7f36d22bc`; raw tree `cec0e2523bd16ce18b541c1eb7cdf65ee26ba553b5ecfb657da617ed2321565e`.
