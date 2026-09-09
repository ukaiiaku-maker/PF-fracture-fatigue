# V5 final physics closure prospective criteria V1

These criteria are frozen after retained-head diagnostics and before any new
controlled-history or full-campaign execution. They do not modify material
parameters, crystallographic eligibility, topology predicates, or numerical
tolerances.

## Static numerical families

The retained 33-family V1 criteria remain the final versioned static criteria.
The diagnostic table must report requested/realized geometry, all three mesh
levels, mesh quality, reaction/compliance/energy, raw cavity traction, fixed
physical crack-tip and cavity-arc tensor sequences, derivatives, every failed
predicate, and the first failed predicate. No V2 relaxation is authorized by
the retained evidence: lack of convergence or inadequate quality is classified
`STATIC_NUMERICAL_CLOSURE_BLOCKED`, not converted to PASS. The current physical
limits remain frozen.

## Controlled-history V2 terminal classifications

The eight retained failures receive these prospective classifications:

| Case | Required terminal classification |
| --- | --- |
| `delayed_downstream` | `DOWNSTREAM_FRONT_ACTIVE_AFTER_DORMANT_INTERVAL_AND_RELOAD` |
| `diffusion_limited` | `STABLE_SUBGRID_VOID_WITH_VACANCY_TRANSPORT_MINIMUM_ALL_INTERVALS` |
| `downstream_zero_drive` | `CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE` |
| `fixed_mesh_oblique` | `CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE` |
| `local_remesh_refinement` | `DOWNSTREAM_FRONT_ACTIVE_AFTER_QUALIFIED_LOCAL_REFINEMENT` |
| `long_ligament` | `DOWNSTREAM_FRONT_ACTIVE` |
| `negative_offset` | `CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE` |
| `short_ligament` | `DOWNSTREAM_FRONT_ACTIVE` |

Short and long cases vary actual ligament geometry while retaining the fixed
crack; their retained real first passages make `DOWNSTREAM_FRONT_ACTIVE` the
prospective expectation rather than the V1 name-derived `CONNECTED_VOID`
expectation. The fixed oblique case may remain connected and dormant but must
not be represented as an active front. Delayed activation must preserve the
same threshold, hazard, RNG, candidate/source identity through the dormant
interval and fire only after a qualifying tensile reload. Every qualified-source
requirement remains mandatory.

## Owned child radius

The causal audit must use 0.75/1.00/1.25 `r_tip` peers and a reciprocal
`R_void` peer at fixed source state. If the accepted rate and barrier equations
contain no `r_tip` edge, the required classification is
`MISSING_ACCEPTED_R_TIP_CONSTITUTIVE_LINK`; no empirical factor may be added.
The reciprocal peer must show that `R_void` is not silently substituted.

## Natural partitioning and disabled neutrality

Natural partitions compare complete accepted-state fingerprints and midpoint
restart, without deleting fields. A kinetic subdivision cannot increment a
physical geometry generation or create an extra remesh.

Disabled neutrality uses `FUTURE_CAUSAL_STATE_FINGERPRINT_V1` and the V2
one-crossing protocol. Historical raw byte identity remains failed. Exact causal
mechanics, physical reactions/energies, hazards, thresholds, RNG, event,
remesh, topology, and direct/restart results are required.
