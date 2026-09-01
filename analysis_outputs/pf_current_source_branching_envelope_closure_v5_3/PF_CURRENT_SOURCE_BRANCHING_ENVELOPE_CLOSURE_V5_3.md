# PF current-source branching envelope closure V5.3

## Decision

The authorized deterministic theta-40 signed-kernel family extension is
qualified through 745 micrometres of cumulative shared-process advance.

This closes the input-coverage defect that stopped the corrected V5.2 pair at
its first 420 micrometre query. It does not change the V5.2.1 scientific
classification:

```text
CORRECTED_CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED_BEFORE_ENVELOPE_STOP
```

The permanent interpretation boundary remains:

```text
CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS
```

No PF run, stochastic evolution, hazard integration, source emission, or
moving-process-zone evolution was launched in this extension task.

## Qualified family

The measured extension stations are:

```text
0, 200, 400, 415, 420, 425, 600, 745 micrometres
```

The 0, 200, 400, and 415 micrometre snapshot and load-invariance payloads were
copied byte-for-byte from the qualified historical family. They were not
recomputed. The 420, 425, 600, and 745 micrometre states were generated with the
same direct prescribed-geometry FEM provider, mechanical configuration,
elasticity, measurement mesh policy, process-zone grid, station normalization,
and signed candidate-independent construction.

```text
old family SHA-256:
b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847

new family SHA-256:
423bc3232326b8ccc3ffcca0aa6b5363c67bad2c64debee49925bf1e6413e8cb

new family physics fingerprint:
e0bc48eb8c3f5526877a21f8551e500046a614df8081693f340084fb73400302

mechanical-configuration fingerprint:
adb7754436a66542a38c17d671bc62639939d85075168a5db721b93b791e87d0

committed family-producer code:
33210af0ef01deb5b08905ea4f49ac5085c7e19d
```

The canonical durable root is:

```text
/Volumes/Data/Data/Nanopillar_calculation/
PF-fracture-fatigue_theta40_signed_kernel_append_only_v5_3_20260901T120000Z_final
```

## Mechanical qualification

Direct load invariance and positive/negative multi-amplitude linearity passed
at every new station. The maximum relative load variation and within-load
spread were:

| Station (micrometres) | Relative load variation | Within-load spread |
|---:|---:|---:|
| 420 | 6.989e-12 | 4.486e-11 |
| 425 | 2.547e-12 | 8.481e-11 |
| 600 | 6.799e-12 | 4.372e-10 |
| 745 | 7.660e-12 | 1.638e-10 |

The final family was independently regenerated from the same committed code.
The family bytes, family SHA-256, physics fingerprint, all four new state-array
payloads, and all four load-1 response tables were identical.

## Exact append-only continuity

Appending states to the legacy inverse-distance atlas would normally change its
global extension-coordinate scale and therefore perturb interpolation below the
old endpoint. The V5.3 family consequently carries a fail-closed exact-prefix
policy: every query in the closed 0--415 micrometre interval is delegated to a
resolver constructed only from the byte-identical four-state legacy prefix with
the unchanged interpolation metadata.

Exact array identity was confirmed at every 5 micrometre physical topology
quantum and every 2.5 micrometre midpoint. The maximum absolute difference was
zero. Continuum identity follows from the delegation construction, rather than
from sampled numerical agreement alone.

The exact-prefix resolver support is part of producer commit `33210af`. An
older execution source that does not understand this opt-in policy must not be
used for a completion replay, because it would silently change interpolation in
the old domain. No completion replay was authorized or launched here.

## Coverage and scope

For the exact terminal two-front topology, the latest possible pre-event shared
process query is 740 micrometres. The 745 micrometre endpoint supplies one fixed
5 micrometre topology quantum as a guard. There is no clipping, endpoint hold,
or extrapolation.

This bound is not universal. The family coordinate is cumulative shared-process
advance along the prescribed single-front measurement geometry. The extension
does not newly validate full branched shielding geometry or independent-tip
mechanics.

## Build disposition

Two fresh scratch roots stopped before qualification: the first exposed the
already-known NumPy-2 two-component cross-product incompatibility; the second
exposed a capture-manifest assembly assumption. Neither root was resumed or
promoted. The established scalar 2-D orientation adapter and manifest-owned
state metadata were then used, and the canonical and reproducibility builds
were both generated from fresh roots.

The compact records beside this report bind the full durable artifact, preserved
payloads, direct validation, exact old-domain identity, terminal-topology
coverage, reproducibility, and the portable family itself.
