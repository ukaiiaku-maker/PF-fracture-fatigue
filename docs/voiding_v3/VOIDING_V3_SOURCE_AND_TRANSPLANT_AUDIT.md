# V12 voiding V3 source and transplant audit

Production base: `8d4f524ccc1c7993c91c1e48c5eb133a7281317c`.
Historical source: PR #57 at
`c71bd96f1e028c49b3596a4c5c3437159d7d5107`; PR #57 remains unchanged.

The four original inputs are identified by the preserved PR #57 manifest:

| Input | SHA-256 |
|---|---|
| `notes(20260829-010935).docx` | `9c67a1d459b019dfa67d9fb0d06de933ba85665c21cb3e3dd19195fb350274ea` |
| `Arrhenius-Based Multiphysics Model for Void Nucleation and Failure in Creep Plasticity(2).docx` | `063075414e852bc5ebee55764277e8290fc63bcc22dad5fe36d851bcf33eba29` |
| `StatefulVoiding_ConstantLoadCreep_MZ_v5(3).m` | `41a89a601355ed0fba2350bd1df5a75fc68a8b86ed3618b0e02266b232fef8bb` |
| `StatefulVoiding_ArrheniusHazard_MZ_v2(3).m` | `77a8d3d71cf3486096025aab175c630a39afdf15c003907822d428a3a48c0733` |

Function-level disposition from `arrhenius_fracture/voiding_v2.py`:

- rewrite the typed site, embryo, and cavity records into the backend-neutral V3 contract;
- transplant the localized first-passage stepping and positive series-limited
  growth semantics with production transaction ownership replacing
  `TransactionAdapter`;
- transplant the body-fitted `build_explicit_hole_mesh`,
  `fill_explicit_hole_mesh`, and static FEM benchmark concepts;
- retain disk/triangle and boundary-component utilities only as geometry tests;
- reject every causal V11 mask, centroid-band, residual-stiffness crack
  comparison, and native-J/absolute-K-only decision;
- reject registry-only rollback and every generated PR #57 PASS/FAIL artifact.

No old scientific decision is transplanted. V3 evidence is regenerated from
the qualified V12 production base.
