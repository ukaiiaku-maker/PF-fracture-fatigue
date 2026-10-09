# PF and full FEM/CZM relationship

The spatial PF and full FEM/CZM solvers remain separate repositories and mechanics backends. They share a configuration/observation contract; their source trees are not merged.

## Tested pair

| Role | Repository | Commit / branch | Configuration |
|---|---|---|---|
| Spatial PF integration | `ukaiiaku-maker/PF-fracture-fatigue` | source commit `baedb069c912db69032eea8b339cb01b4bf97278` on this branch, based on `9e884fb0b0845da621d2612bdf1042e481b8df49` with recorded donors | typed option receipt selects the concrete PF provider |
| Full FEM/CZM comparison | `ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ` | `293491157063484bb10df6adc479f847a6a08ba6`, `v10.0.5.18.4.0-theta0-pf-parity` | theta=0 full anisotropic FEM, adaptive CZM, native cohesive/J/work fields |

The corresponding machine-readable record is `PF_FEM_TESTED_PAIR.json`. No FEM source change was required, so no companion FEM branch was created.

## What is shared

- units, material/kernel identities, temperature and loading histories;
- source geography and lifecycle labels rather than one overloaded source flag;
- chronological observations with front, loading segment, stage and validity;
- event transaction semantics and restart identities;
- comparison coordinates such as crack extension, K, J, G, work, and native state summaries.

## What remains native

- PF retains its signed mobile/retained/wake arrays, persistent-site clocks, and PF/sharp-front topology providers.
- FEM/CZM retains its anisotropic mesh, cohesive history, conservative remesh transfer, bulk constitutive fields, and native contour/work observations.
- Array shapes are not forced to match. Cross-backend plots compare named physical observables and explicitly matched stages.

The reduced `FEM-CZM-fatigue-fracture` package is not substituted for the full FEM/CZM pair.
