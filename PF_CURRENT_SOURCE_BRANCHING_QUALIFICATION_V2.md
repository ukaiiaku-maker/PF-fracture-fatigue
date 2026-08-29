# PF current-source branching qualification V2

`CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS`

## Two-axis decision

- Morphology capability: **CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED**.
- Final independent-tip mechanics: **UNQUALIFIED_FINAL_LOCAL_CONTOURS**.
- Cluster handoff: **NOT_TRIGGERED_UNRESOLVED_CLUSTER** (conditional production gate; not an unconditional morphology gate).
- Predictive branching physics validated: **false**.

The immutable V1 decision remains `NO_CURRENT_SOURCE_BRANCH_BIRTH_IN_BOUNDED_TEST` at SHA-256 `d98865ea32710f73e70186a884ea4aeb29a5779ac533668c3724bccb3c162941`. V2 does not reuse that phrase as its headline because committed branch births occurred.

The authoritative handoff required reliable positive secondary directional J at birth, non-stub daughter growth, topology/state/RNG/geometry closure, and independent handoff only when its production guard fired. It did not separately define final local-probe reliability as an unconditional morphology-capability gate. The θ40 transaction met those morphology requirements: its birth probes were reliable and positive, one daughter grew to 295 µm, maximum forward reach reached 302.236 µm, the length/topology ledgers closed, pre-birth enabled/control histories were neutral, no prohibited bridge/reconnection/backward-growth/cap event occurred, and the run terminated normally.

## Local contours and long reload

The final θ40 nested contours are geometrically clean and have adequate finite-element support, but they lack a 15% numerical plateau. The exact classification is `absence_of_15_percent_numerical_plateau`. The last archived state where both active-tip probes were reliable is step 2147 at 275.424750 µm. All nested rows and pairwise differences are in `pf_branching_local_contour_convergence_audit.csv`.

Across every archived post-birth span, the shared-state-update counter increases exactly once per accepted physical interval. Two active fronts contribute directional observations to the shared unresolved-cluster competition, but only the maximum drive controls the single shared update callback. No duplicated two-front update was found. There is no process-state checkpoint strictly inside the 271.595 µm long-wait archive interval, so the internal timing of the large state change is unresolved and is not inferred.

## θ45 frozen visibility diagnosis

The last accepted checkpoint was replayed only through deterministic mesh/trial preparation; physical time, accepted state, stochastic thresholds, and RNG state were unchanged. The zero-visibility proposal is class A: its exact causal support lies entirely in already committed P0 wake material. Nested refinement inherits that damage and cannot manufacture new stiffness contrast. The implementation now emits the reason-specific fail-closed veto `candidate_segment_already_in_committed_wake_material`; it does not relax the causal-visibility gate and does not add a visibility mark. No production replay is scientifically permitted from this A classification.

## Asymmetric unresolved cluster

θ40 remained one cluster because the present source uses an all-arm handoff: the 295 µm arm passes length/separation, but the 30 µm arm does not, and final independent local J is not qualified. The source contains neither independent long-arm resolution while a sibling remains junction-owned nor a physical short-arm retirement rule. No empirical stagnation time, length, or J cutoff was invented; this remains a future model limitation.

## Resumability

Continuation is implemented fail-closed as a fresh-output fork. It restores the complete accepted state and RNG/threshold state, starts at the next accepted step, requires a strictly larger target, rebinds the live mechanics cache to the destination, and fingerprints the immutable source tree before and after execution. Unit and checkpoint-contract tests pass, but production replay qualification remains false because the class-A diagnosis does not authorize a heavy replay. A deterministic 1000 µm command plan is recorded as the next-step handoff; it was not launched in this audit.

## Validation

The branching-focused suite passed 93 tests with one skip. The full repository suite passed 767 tests with one skip and only the same seven legacy failures as the parent baseline; there were no new failures. `compileall` and `git diff --check` passed. Two consecutive regenerations produced byte-identical V2 outputs. Fresh read-only tree fingerprints for θ40 (`a28d5c312f1747658efa1ab5ec87aa85170da1180832714cadc3bce5bb9890b5`) and θ45 (`d08a706d12cf16d43390a59a0aeaa1113cda828c0f88de084f6e472048827384`) exactly match the immutable V1 provenance.
