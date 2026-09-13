# V5 unified model lineage

The restoration branch starts at retained V5 commit `619b66b0f538742482e7385ddd3d0171cb3edd8b` and transplants the qualified current-source multi-tip core from `ca4abfe47765fcdaf0d266bfc0558ecd82d0c64e` by exact source content. Historical branches remain read-only and were not merged.

The ancestry audit identifies `7e71b8f27b0682060fd161e7e5e8fe0d3587e8ac` as the canonical single-tip base and `0238aae096aa29e79829d3c562383c38f2290ad6` as the qualified multi-tip head and the V12/V13 merge base. This decision came from commit ancestry and merge-base inspection.

## Route decision

`TRANSPLANT_VOID_EXTENSION_ONTO_QUALIFIED_MULTITIP_CORE`

The retained V5 source differed materially in the front, FEM/source, process-owner, directional-competition, checkpoint, and topology paths. The restored line therefore retains V13 core behavior and adds the cavity inventory, void state, V12 separating-support transactions, hybrid provider, and downstream void-front lifecycle as explicit extensions. The fixed-load equilibrium routine is the qualified V13 implementation.

## Component classification

| Role | Path | Classification |
|---|---|---|
| sharp-front constructor and renewal | `arrhenius_fracture/sharp_front.py` | `CORE_ADAPTER_ONLY` |
| material manifest | `arrhenius_fracture/material_manifest.py` | `CORE_IDENTICAL` |
| active/wake MPZ state | `arrhenius_fracture/unified_mpz.py` | `CORE_IDENTICAL` |
| hazard energy gate | `arrhenius_fracture/hazard_energy_event_gate_v10230.py` | `CORE_IDENTICAL` |
| directional hazard/RNG competition | `arrhenius_fracture/directional_competition_v11.py` | `CORE_IDENTICAL` |
| multi-front runtime ownership | `arrhenius_fracture/general_multifront_v12.py` | `BRANCHING_EXTENSION` |
| stateful multi-front production | `arrhenius_fracture/stateful_multifront_production_v12.py` | `BRANCHING_EXTENSION` |
| topology energy transaction | `arrhenius_fracture/topology_transaction_v11.py` | `CORE_ADAPTER_ONLY` |
| live directional provider | `arrhenius_fracture/live_topology_kernel_v11.py` | `CORE_ADAPTER_ONLY` |
| accepted checkpoint | `arrhenius_fracture/checkpoint_v11.py` | `CORE_ADAPTER_ONLY` |
| zero-active-tip mesh adapter | `arrhenius_fracture/mesh.py` | `CORE_ADAPTER_ONLY` |
| hybrid void directional drive | `arrhenius_fracture/hybrid_directional_drive_v5.py` | `VOIDING_EXTENSION` |
| fixed-arc cavity source recovery V2 | `arrhenius_fracture/cavity_source_recovery_v2.py` | `VOIDING_EXTENSION` |
| fixed-physical-arc cavity source recovery V3 | `arrhenius_fracture/cavity_source_recovery_v3.py` | `VOIDING_EXTENSION` |
| source-conforming cavity geometry V4 | `arrhenius_fracture/cavity_source_conforming_geometry_v4.py` | `VOIDING_EXTENSION` |
| shape-regular cavity source mesh V5 | `arrhenius_fracture/cavity_source_shape_regular_mesh_v5.py` | `VOIDING_EXTENSION` |
| void state and kinetics | `arrhenius_fracture/voiding_v5.py` | `VOIDING_EXTENSION` |
| void production driver | `arrhenius_fracture/voiding_production_v5.py` | `VOIDING_EXTENSION` |
| unified material bundle/factory | `arrhenius_fracture/unified_fracture_material_v5.py` | `CORE_ADAPTER_ONLY` |

No component remains classified as `UNINTENDED_CORE_DIVERGENCE`.

The qualified-source inventory is exactly **238/244 exact blob matches**, with
**six reviewed adapter files**. The machine-readable JSON contains the same
counts and the complete six-file list.

## Material and state ownership

`UnifiedFractureMaterialBundle` owns separate fracture, void-kinetics, elastic, site-population, and loading row identities. The exact Peak, DBTT, weak-T, and ceramic-like fracture rows are immutable. One canonical factory constructs the root, branch, and downstream unified MPZ engines. Accepted FEM state and checkpoints retain the core, bundle, elasticity, plasticity, FrontConfig, cleavage, emission, and process-geometry fingerprints across refinement and restart.

The retained tip-radius law is `r_tip = r0 + c_blunt*b*local_weighted_accumulated_slip`. The cavity radius is independent, and `r_tip != R_void`.

## Bounded decision

All thirteen terminal identity and nested-limit gates are encoded as `PASS` in the machine-readable ledger. Validation is limited to the unified lineage tests plus the retained provider and child-continuation tests. No broad static, natural-seed, closure, branching, fatigue, or calibration campaign was run.

The Peak negative energy-gate sentinel (`REJECTED_NEGATIVE_ENERGY_MARGIN`) and
the DBTT positive continuation sentinel (`PASS_POSITIVE_CONTINUATION`) remain
separate results.

`CAVITY_SOURCE_RECOVERY_V1_INCIDENT_CST_MAX_PRINCIPAL = FAIL_NONCONVERGENT` is
retained. Its replacement is the prospectively frozen
`CAVITY_FIXED_ARC_PATCH_RECOVERY_V2` operator. V2 remains retained evidence.

The frozen production source still applies `eta_n <= 0.03` and
`eta_t <= 0.025` as mandatory V2 readiness gates. The final DBTT V2 row passes
the tangential-resolution gate and fails the normal-resolution gate; together
with the failed tensor-change predicate, its complete failure classification
is `TANGENTIAL_STRESS_CONVERGENCE + NORMAL_DIRECTION_RESOLUTION`.

The active V3 source operator is prospectively frozen as
`CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3`. It uses one identical physical
arc window at every mesh level: tangential half-width and outward normal depth
are each `0.5*min(R_void,L_pz)`. Its quadratic curvilinear WLS fit enforces
`sigma_nn(s,0)=sigma_nt(s,0)=0`; its bounded manufactured, Kirsch, rotation,
edge-order, and reflection tests pass before the central DBTT evaluation.

The single central DBTT V3 evaluation fails before recovery as
`FAIL_POLYGON_VERTEX_VS_NOMINAL_CIRCLE`: the exact owned polygon source coordinate is
`0.2661214806971317 um` outside the nominal circular radius, beyond the frozen
V3 geometry-identity tolerance. V3 tensor convergence is `NOT_RUN`; this is not
a failed V3 stress-recovery result.

The prospective `CAVITY_SOURCE_CONFORMING_GEOMETRY_V4` contract rotates the
circumscribed polygon to center a facet on the source ray, splits that facet at
the exact nominal-circle point, and certifies a degree-two collinear node with
one-owner incident boundary edges. Both aligned near and far points satisfy the
contract. Cavity area and kinetics continue to use the nominal circle; the
actual polygon is only the FEM boundary.

The bounded 32/64/128 central DBTT V4 evaluation passes geometry registration,
successive tensor convergence, traction, physical-window identity, and patch
conditioning. It fails closed on mandatory normal and tangential resolution
and global mesh quality at the final level. Its exact failure classification is
`NORMAL_DIRECTION_RESOLUTION + TANGENTIAL_DIRECTION_RESOLUTION + MESH_QUALITY`.
The V4 source tensor changes by `0.006818275586047867` from N64 to N128, so the
point-source formulation remains viable. Its zero boundary traction is the
traction-free constraint of the recovered boundary-limit tensor, not an
independent raw adjacent-element traction measurement. The geometry-only V4
audit locates the quality collapse in the crack-support/cavity interaction.

The prospective `CAVITY_SOURCE_SHAPE_REGULAR_LOCAL_PATCH_V5` contract fixes a
512-edge discrete source boundary, a matched radial source column, and
deterministic structured connectivity. The required N64/N128 angular family
and fixed-polygon A/B/C local family pass tensor, reaction, compliance, energy,
resolution, quality, conditioning, source-window identity, and accepted-state
immutability checks. The independent raw adjacent-element traction remains
`0.06715193304848284` at N128/C, above the frozen `0.05` limit. The exact V5
failure class is `RAW_ADJACENT_ELEMENT_TRACTION`. The conditional N256 angular
diagnostic reaches lower raw traction but has minimum quality
`0.00814983630892272`, below the frozen `0.05` gate, and cannot qualify the
family. The oracle remains `0/18`, and point-source mesh development stops.
No finite activation-zone observable was derived.

The one-job V5 clean worker at
`{v5_attestation['bounded_worker']['workflow_head'] if v5_attestation else 'NOT_YET_ATTESTED'}`
passed exactly
`{v5_attestation['bounded_worker']['tests_passed'] if v5_attestation else 0}` bounded tests in run
`{v5_attestation['bounded_worker']['run_id'] if v5_attestation else 'NOT_YET_ATTESTED'}`.
Artifact `{v5_attestation['bounded_worker']['artifact_id'] if v5_attestation else 'NOT_YET_ATTESTED'}`
has digest
`{v5_attestation['bounded_worker']['artifact_digest'] if v5_attestation else 'NOT_YET_ATTESTED'}`.
The automatically triggered broad repository run was cancelled as
`CANCELLED_SCOPE_CONTROL_NOT_A_TEST_FAILURE`.

The one-job V4 clean worker at `b12e21b5c7485c490bf0cb5139feb2d736844a16`
passed all 36 bounded tests in run `34723836462`. Artifact `10307550298` has
digest `sha256:70666a9df0c0b43e2b358059cc8960fb88d01db28bf5165c929c5323456199f3`.
Automatically triggered broad repository runs were cancelled as
`CANCELLED_SCOPE_CONTROL_NOT_A_TEST_FAILURE`.

No paired temperature trajectory or fatigue run belongs to this checkpoint.
