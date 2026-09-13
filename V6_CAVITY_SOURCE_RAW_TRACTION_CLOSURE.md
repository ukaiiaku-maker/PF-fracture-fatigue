# V6 equilibrium-observable and raw-traction closure

Status: **DBTT source readiness remains blocked**.

V6 preserved the V1–V5 results and froze the final N128 local family before
running its two new levels. The accepted FEM residual now supplies direct top
and bottom reactions, applied opening, compliance, full and free residuals,
external work, recoverable energy, and the plastic-strain-aware production
energy identity. Missing or zero reaction ledgers fail closed and cannot become
a scientific zero or a finite sentinel compliance.

The retained raw adjacent-element traction sequence and the final V6 values are:

| level | first-strip radial subdivisions | raw traction | global minimum quality |
|---|---:|---:|---:|
| A | 2 | 0.14906976345973477 | retained V5 |
| B | 4 | 0.09144336014017264 | retained V5 |
| C | 8 | 0.06715193304848284 | retained V5 |
| D | 16 | 0.05795457514046478 | 0.12451519968640087 |
| E | 32 | 0.055309963627059575 | 0.047548651614431246 |

The conditional N256 value `0.038672813574278736` remains a low-quality V5
diagnostic and is not a qualification level.

D-to-E source tensor, reaction, compliance, recoverable energy, and ligament
release all converge within their retained limits. Both levels pass eta-n,
eta-t, weak cavity-boundary residual, fit rank, fit conditioning, material and
state identity, and physical equilibrium-observable checks. The V6 decision is
blocked by exactly three prospective predicates:

- `RAW_ADJACENT_ELEMENT_TRACTION_E`: `0.055309963627059575 > 0.05`;
- `MESH_QUALITY`: E gives `0.047548651614431246 < 0.05`;
- `FIXED_GEOMETRY_IDENTITY`: the complete post-connection cavity-boundary
  fingerprint changes at E, while the physical source window stays fixed.

Therefore:

```text
V5_SOURCE_CONFORMING_GEOMETRY = PASS
V5_SHAPE_REGULAR_LOCAL_PATCH = PASS
V5_SOURCE_TENSOR_CONVERGENCE = PASS
V5_WEAK_TRACTION_FREE_BOUNDARY = PASS
V5_RAW_ADJACENT_ELEMENT_TRACTION = FAIL_0.0671519_GT_0.05
V5_REACTION_COMPLIANCE_OBSERVATION = INVALID_DEFAULT_OR_STALE_LEDGER
DBTT_SOURCE_READINESS = BLOCKED
FINITE_ACTIVATION_ZONE_REQUIRED = NOT_ESTABLISHED

DBTT_SOURCE_READINESS = BLOCKED_WITH_EXACT_V6_FAILURE_CLASS
ORACLE_STATES_ACCEPTED = 0_OF_18
PAIRED_TRAJECTORIES_RUN = 0_OF_12
FATIGUE_IMPLEMENTATION = NOT_STARTED_BY_CONTRACT
NEXT_BOUNDED_STEP = EQUILIBRATED_BOUNDARY_STRESS_RECONSTRUCTION
```

V6 adds no recovery level, activation-zone observable, mechanics-map fit,
trajectory, fatigue model, or change to the established `r_tip` law. The
source radius remains distinct from `R_void`.

The dedicated clean worker at
`774db759df503b2eef4580041f51586376b0125b` passed exactly 34/34 bounded
tests in run `34777671961`, job `103778647069`. Artifact `10323943431` has
digest
`sha256:034be555d5d4f3f65cf16202d23a1612520612392bc6ab34dcd7086d5c84f748`.
Automatically triggered non-V6 runs were cancelled as
`CANCELLED_SCOPE_CONTROL_NOT_A_TEST_FAILURE`.
