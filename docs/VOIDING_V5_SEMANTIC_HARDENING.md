# V5 semantic-hardening checkpoint

Status: `PHASE_A_REQUALIFIED`; broad finalization may begin only after the
dedicated exact-head workflow reaches a successful terminal result.

This checkpoint follows, and does not rewrite, exact green checkpoint
`2b39339d7738cc3d251146b69f10ce1a9e0e17c4`.

The active-front ledger now distinguishes connected cavity span from traversed
cavity span. Connection stops at the near boundary and arrests the incoming
root; both graph and V12 support therefore report zero active tips in the
connected state. Downstream child creation atomically makes `void-front-1` the
sole graph and support tip. Checkpoint/restart and injected rollback tests cover
both ownership states.

The solver-backed causal comparison uses one frozen incident element and records
its node, element, operator identity, and unit recovery weight in every row.

`cavity_free_surface_certificate` derives the exact boundary node and edge IDs
and certifies one degree-two closed traction-free cycle. The separate
`crack_void_connection_certificate` derives every crack-segment/open-disk
distance, exact support-triangle/open-disk overlap, sampled former-ligament
coverage, the intersected boundary edge, and the actual connected components
of the branch/cavity incidence graph. Adversarial tests cover wrong identity,
stale support, a surviving crack segment through the cavity, a broken cycle,
and a triangle that overlaps the cavity while its centroid remains outside.

Single-arm V5 transactions filter proposal selection to one-arm proposals.
Unselected simultaneous events remain pending at the original completion time
for reconsideration on rebuilt geometry.

Interior oblique ray/cavity-edge intersections are inserted as explicit
boundary nodes inside the isolated trial before graph realization and support
rebuilding. The true lateral-offset test holds one crack path fixed while the
cavity moves by plus/minus 10 micrometres; it checks mirrored intersections,
reaction/compliance symmetry, even tensor components, the reflection-odd shear
increment, certification, acceptance, and exact rollback. The oblique test
also derives physical chord/travel and projected span/advance independently;
it does not substitute `2*R` for a non-diametral chord.

Initial seed area is part of finite defect inventory and is debited atomically.
New causality, inventory, and combined-topology evidence predicates resolve raw
source rows and recompute their scientific decisions.

Repository-wide CI is not green. The machine-readable comparison at
`artifacts/voiding_v5_semantic_hardening/general_ci_inheritance.json` records the
same seven failure identities at the exact base, retained V5 checkpoint, and
semantic-hardening head `b786f7c9c5d2175803d50b88d4215853468d4157`:
`V5_INTRODUCED_NEW_GENERAL_CI_FAILURES=NO_OBSERVED` and
`GENERAL_REPOSITORY_CI=FAIL_INHERITED_BASELINE`.

The requalification additionally separates geometric eligibility, kinetic
activity, and first-passage completion. A connected cavity with an all-zero
downstream rate set is accepted as `CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE`,
retains every admissible source-native candidate and its untouched clock/RNG
provenance, and has neither child nor active tip. Partition, restart, reload,
and rollback tests preserve this state exactly.

Physical external roots and inactive cavity terminals now use a strict
per-arc/per-endpoint boundary-relative V12 certificate. The audit records exact
incidence, component and cavity identity, independently closed cavity-cycle
status, incident tangent, all tube/boundary intersections, interior two-sided
seed counts, intact-path and node-star results, and the precise classification.
Only incident traction-free-boundary clearance is conditional; all other V12
mechanical-separation predicates and tolerances remain frozen.
