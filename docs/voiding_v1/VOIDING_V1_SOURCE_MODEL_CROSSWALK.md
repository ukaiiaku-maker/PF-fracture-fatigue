# Voiding V1 source/model crosswalk

All four inputs in `voiding_v1_source_input_manifest.json` were read before production edits. Their numerical values are not material data for this branch.

| Source concept | Classification | V1 disposition |
|---|---|---|
| Site-resolved Arrhenius birth intensity and exact tensor work | ADOPTED_DIRECTLY | `birth_intensity` and `tensorial_activation_work`; hydrostatic, opening, and signed shear contributions remain separate. |
| Multi-hit incomplete-gamma/Poisson completion | ADOPTED_DIRECTLY | Site-local accepted-history clock with rollback/checkpoint state. |
| First-passage survival threshold and RNG | ADOPTED_DIRECTLY | Typed `FirstPassageState`; RNG state and threshold are serialized. |
| Candidate-site availability and weight | ADOPTED_DIRECTLY | Weight multiplies birth only. Existing objects control every later transition. |
| Reversible embryo stabilization versus healing | ADOPTED_DIRECTLY | Competing site-local first passages; healing creates no geometry. |
| Stable-cavity growth | ADOPTED_DIRECTLY | Diffusion and plastic accommodation use the harmonic series limit when both are required. |
| Defect/vacancy supply | ADAPTED_TO_EXPLICIT_2D_GEOMETRY | Replaceable site-local closure in V1; no claim of a resolved reaction-diffusion field. |
| Spherical area, volume, and Laplace pressure | ADAPTED_TO_EXPLICIT_2D_GEOMETRY | Replaced by plane-strain through-thickness area, perimeter, and cylindrical `gamma/R`; 3-D equivalents are diagnostic only. |
| Existing-cavity mechanics | ADAPTED_TO_EXPLICIT_2D_GEOMETRY | True element-free, traction-free internal boundary; no empirical amplification. |
| Population coalescence | DEFERRED | Multi-void population coalescence is outside single-void V1. |
| Grain-boundary, particle, and triple-junction activation | DEFERRED | Schema compatibility only because the qualified geometry is single crystal. |
| Spatial vacancy reaction-diffusion | DEFERRED | Provider interface reserved; not claimed in V1. |
| Prototype global damage and load-bearing-area loss | REJECTED_AS_INCOMPATIBLE | Never used for a resolved hole. |
| Gurson/GTN and diffuse phase-field options in the review | REJECTED_AS_INCOMPATIBLE | They do not enter production. |
| Scalar topology/coalescence variable | REJECTED_AS_INCOMPATIBLE | Connectivity is explicit geometry/component lineage. |
| Generic Mori–Zwanzig/Prony modes | REJECTED_AS_INCOMPATIBLE | No production state or rate uses them. |
| Prototype numerical rate caps and parameter values | DIAGNOSTIC_ONLY_NOT_PRODUCTION_PHYSICS | Not transferred to the canonical registry or claimed as calibration. |

Birth, stabilization, healing, growth, representation promotion, ligament fracture, connection, and downstream nucleation are distinct events. Crack–void ligament failure reuses the existing cleavage barrier; it is not a new mechanism.
