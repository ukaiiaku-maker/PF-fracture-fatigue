# V12 mechanically separating sharp-wake qualification

This draft branch is based on V11 production commit `2b5e5351add0bf0db67f2cda35a1480c3e7efc91`
and defines the separate model identity `sharp_wake_mechanically_separating_v12`.
V12 is not installed in the production transaction path and PR #57 is unchanged.

## Attested geometry baseline

The hardened geometry evidence commit `020b5af` (generated from exact
implementation commit `094eb7a4559114f5a359951c236aa4892ac786dc`) reports
`MECHANICALLY_SEPARATING_WAKE_GEOMETRY_QUALIFIED = PASS`. Scoped Python 3.12
workflow run `33446100847` tested the preceding `1b426ef` baseline and passed 88 scoped tests,
two deterministic evidence regenerations, `git diff --check`, and the clean-worker check.

The Phase 0 integrity patch replaces literal scientific PASS values with
evidence-derived Booleans, adds bidirectional graph/support component incidence,
detects nonadjacent-arc support short circuits, calculates legal junction
overlap geometrically, adds Y/T defective controls and exact physical
coalescence, and adds an explicit no-mechanical-novelty event control. Its local
focused suite passes 66 tests with the paired V11 causal-wake suite; its exact implementation source is recorded in
the regenerated qualification provenance.

## Primal mechanics screen

The analysis-only straight-crack screen uses the conforming node-splitting
construction sourced from PR #57 commit `8ad7f42`; it does not import or
transplant the voiding solver. For each `h_tip = 25, 12.5, 6.25 um`, the intact,
V11 P0, V12 separating, and conforming-slit representations derive from one
parent geometry and connectivity. The authoritative straight matrix now also
contains the 3.125 um level with identical columns, so "finest" is no longer
split between a main matrix and a prescreen. V12 is evaluated at `kappa = 1e-4, 1e-6,
1e-8`. Centered energy and compliance release rates use the common physical
increment `delta_a = 25 um` on the two finest meshes.

The thresholds in `scripts/qualify_v12_primal_mechanics.py` were frozen before
the authoritative result: 5% at the finest mesh for global observables,
outside-support stress, and centered G; 1% for the energy/compliance G identity;
0.1% for low-kappa reaction spread and killed energy at `kappa=1e-6`; and
`1e-10` for equilibrium and energy/reaction identities. The 30 and 45 degree
matrices may run only after the straight gate passes.

The subsequent hardening pass found that post-solve stress recovery had used a
non-interleaved element DOF order even though the CST B matrix requires
`x0,y0,x1,y1,x2,y2`. Stiffness, displacement, reaction, energy, COD, and G were
unaffected; the earlier tensor, traction, and mirror numbers were invalid and
are superseded. Correct recovery plus symmetry-plane exterior pins qualifies
Mode-I mirror symmetry. Support-aware COD sampling and the bounded 3.125 um
level qualifies matched COD without relaxing its 5% tolerance (2.786%
h-scaled and 2.618% fixed-distance error). The three-delta G plateau remains
qualified.

Permanent affine patch tests now recover arbitrary constant strain and stress
on translated and rotated triangles, reconcile analytical, element, and
quadratic-form energy, and prove that the superseded blocked DOF ordering is
detected.

The aggregate primal gate remains OPEN. Every fixed-region tensor error
decreases, but the 3.125 um face-adjacent strip error is 17.825%, above the
frozen 5% limit. A deterministic bounded matched 1.5625 um crack/tip refinement
reduces that metric to 9.033% and gives a 2.117% production tensor-probe error.
Because this was below the predeclared 10% stop line and remained convergent,
one final 0.78125 um level was run. It closes at 4.564% face-strip error and
1.199% production-probe error, so `V12_P0_LOCAL_TENSOR_FIDELITY = PASS` and
`V12_LOCAL_TENSOR_FIELDS_QUALIFIED = PASS` under the unchanged 5% limit.

The joint corridor matrix retains fixed `kappa = 1e-6, 1e-8` rows and coupled
`kappa(h)=kappa0(h/h0)^p` policies for `p=1,2`, with `h0=25 um` and
`kappa0=1e-6`. Under `p=2`, recovered traction, discrete normal transfer, and
killed energy all fall by about an order of magnitude and are far below the
declared mechanics budget at 3.125 um. The discrete shear series is already at
numerical-cancellation scale but is not strictly monotone, so the predeclared
all-metric monotonic gate remains FAIL rather than being changed after seeing
the result.

## Prospective V3 physical transmission criterion

The historical result is retained as
`V12_SOFT_CORRIDOR_TRANSMISSION_V2_FROZEN =
FAIL_STRICT_SIGNED_COMPONENT_MONOTONICITY`; V3 does not relabel it.

Before regenerating V3, the physical gate is frozen on the existing `p=2`
policy and mesh levels. It requires decreasing `kappa/h`, recovered soft-side
traction magnitude `T_sigma`, discrete normal-transfer magnitude `T_n`, vector
resultant `T_resultant`, and killed-energy fraction `E_soft`. Both finest
`T_sigma`, `T_resultant`, and `E_soft` values must be below `1e-3`; `T_t` must
be below `1e-3` at every level; straight Mode-I symmetry, the 5% primary
budget, `1e12` conditioning limit, and equilibrium residual limit must remain
qualified. Signed shear is published but is not required to be strictly
monotone when its magnitude is numerical zero. The `p=1` sequence remains a
nonqualifying diagnostic control.

Recovered soft-side traction and discrete element-internal-force balance are
independent diagnostics. Intact-side traction and the elementwise traction
jump are retained diagnostically and are not required to vanish.

The angle result is classified only as rotation covariance. The rotated graph
endpoints are asserted to be actual mesh vertices and the off-grid screen
override is not used. Absolute K and all production work remain outside scope.

## Geometry evidence requirements

The runner derives every scientific geometry gate from its rows:

- uniform fixed-domain refinement requires decreasing physical width and area,
  convergent signed tip footprint, bounded `width/h`, and bounded `area/(L h)`;
- the fixed crack-local patch must remain invariant under independent far-field
  remeshing by physical-coordinate support fingerprint;
- event classification requires production-valid acceptance, alignment-remesh,
  and no-mechanical-novelty controls, correct classifications, a stiffness
  fingerprint change for every accepted event, and sequential equivalence;
- every graph component and node-connected support component must have a
  one-to-one incidence outside declared junction/coalescence neighborhoods;
- nonadjacent certification arcs in one graph component may not become
  node-connected through their support;
- kink, Y, T, mixed-role, merged-terminal, and exact-coalescence neighborhoods
  use geometric overlap and annular sector certificates; deliberately defective
  kink, Y, and T supports must be rejected.

## Reproduction

```bash
python scripts/qualify_v12_mechanically_separating_wake.py
python scripts/qualify_v12_primal_mechanics.py
python -m pytest -q \
  tests/test_v12_primal_crack_mechanics.py \
  tests/test_v12_mechanically_separating_wake.py \
  tests/test_v11_causal_sharp_wake.py \
  tests/test_crack_network_v11.py \
  tests/test_v11_live_topology_multitip.py \
  tests/test_topology_transaction_v11.py
```

## Gate ledger

Authoritative geometry states are generated in
`artifacts/v12_mechanically_separating_wake/qualification.json`; they are not
manually asserted here. At the attested baseline all construction, partition,
local-objectivity, component-topology, coalescence, junction-sector,
active-tip/event-resolution, and overall geometry gates are PASS.

| Later gate | State |
|---|---|
| `PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED` | NOT_RUN |
| `V12_PRIMAL_GLOBAL_RESPONSE_SCREEN` | PASS |
| `V12_CENTERED_G_SINGLE_INCREMENT_SCREEN` | PASS |
| `V12_ROTATION_COVARIANCE_SCREEN` | PASS |
| `V12_MATCHED_COD_QUALIFIED` | PASS at unified 3.125 um level |
| `V12_INTERFACE_TRACTION_DIAGNOSTIC` | RETAINED_NOT_AGGREGATE |
| `V12_LOCAL_TENSOR_FIELDS_QUALIFIED` | PASS at bounded 0.78125 um refinement |
| `V12_P0_LOCAL_TENSOR_FIDELITY` | PASS |
| `V12_G_PERTURBATION_CONVERGENCE` | PASS |
| `V12_STRAIGHT_MODE_I_SYMMETRY_QUALIFIED` | PASS |
| `V12_SOFT_CORRIDOR_TRANSMISSION_V2_FROZEN` | FAIL_STRICT_SIGNED_COMPONENT_MONOTONICITY |
| `V12_SOFT_CORRIDOR_TRANSMISSION_V3_PHYSICAL` | PASS |
| `V12_SOFT_CORRIDOR_TRANSMISSION_QUALIFIED` | PASS under V3 physical criterion |
| `V12_STRAIGHT_MODE_I_PRIMAL_MECHANICS_QUALIFIED` | PASS |
| `V12_PRIMAL_CLEAN_WORKER_REPRODUCIBLE` | NOT_RUN |
| `MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED` | PASS_STRAIGHT_MODE_I_SCOPE |
| `MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED` | NOT_RUN |
| `V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED` | OPEN |

Geometry and the three preliminary mechanics screens do not authorize production wiring,
absolute-K evaluation, or voiding. Absolute KI remains NOT_RUN and the overall
production prerequisite remains OPEN.

## Prospective absolute-K criterion

Before any V12 absolute-K result is generated, Stage I freezes criterion
`v12_absolute_k_qualification_v1` in
`arrhenius_fracture.absolute_k_criterion_v12`.  It uses the existing 5% mechanics
budget, a 5% Mode-I symmetry bound, three geometry-only physical contours
`(240,260)`, `(250,270)`, and `(260,280)` micrometres, and matched mesh levels
12.5, 6.25, and 3.125 micrometres.  Those contours satisfy `r_inner/h_tip >= 8`,
`(r_outer-r_inner)/r_inner <= 0.1`, and at least 15 micrometres of root,
exterior, and patch clearance without reference to extracted K values.

The fixed Williams stress basis contains K-I, K-II, and constant sigma-xx
T-stress.  Its primary annulus is 75--175 micrometres; two predeclared radius
variants measure sensitivity.  Missing checks fail closed.  A standard-integral
failure may only receive the exact qualified-unavailable classification when
the conforming control, primal mechanics, V3 transmission, global energy, and
Williams gates pass.  Production may then continue only if the event-law audit
proves absolute K is not consumed (or a separately qualified conforming tip
source exists).

### Absolute-K result

The SHA-bound v1 matrix classifies the conforming interaction-integral control,
geometry admissibility, global energy/compliance identity, and K-I mesh
convergence as PASS.  The fixed-`kappa=1e-8` and joint-`p=2` V12 contour spreads
are far inside 5%, but the prospectively fixed Williams stress fit differs from
`sqrt(Eprime*G_energy)` by more than 5%.  Consequently the complete standard
absolute-K gate fails closed even though its interaction-integral subdiagnostic
is stable:

`V12_STANDARD_INTERACTION_INTEGRAL_ABSOLUTE_K = NOT_QUALIFIED`

`MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED =
CONFORMING_TIP_PATCH_REQUIRED`

The matched conforming representation provides converged interaction-integral
and energy extraction, and the production dependency audit finds no absolute-K
consumer in event selection.  Thus `STAGE_II_PERMITTED = PASS`; unqualified V12
K must remain unavailable to production.  The initially generated report that
incorrectly aggregated contour stability alone into PASS is retained as
`superseded_classifier_failure.json` and is not scientific evidence.
