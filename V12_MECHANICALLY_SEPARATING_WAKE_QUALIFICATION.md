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
parent geometry and connectivity. V12 is evaluated at `kappa = 1e-4, 1e-6,
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
pre-screen qualify matched COD without relaxing its 5% tolerance (2.786%
h-scaled and 2.618% fixed-distance error). The three-delta G plateau remains
qualified.

The aggregate primal gate remains OPEN. Although all fixed-region tensor errors
decrease, the 3.125 um face-adjacent strip error is 17.825%, above the frozen 5%
limit, and no qualified joint h-kappa soft-corridor transmission limit has yet
been demonstrated. Those failed matrices are retained.

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
| `V12_MATCHED_COD_QUALIFIED` | PASS at 3.125 um pre-screen |
| `V12_INTERFACE_TRACTION_QUALIFIED` | FAIL, matrix retained |
| `V12_LOCAL_TENSOR_FIELDS_QUALIFIED` | FAIL, matrix retained |
| `V12_G_PERTURBATION_CONVERGENCE` | PASS |
| `V12_STRAIGHT_MODE_I_SYMMETRY_QUALIFIED` | PASS |
| `V12_SOFT_CORRIDOR_TRANSMISSION_QUALIFIED` | FAIL, matrix retained |
| `V12_PRIMAL_CLEAN_WORKER_REPRODUCIBLE` | NOT_RUN |
| `MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED` | OPEN |
| `MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED` | NOT_RUN |
| `V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED` | OPEN |

Geometry and the three preliminary mechanics screens do not authorize production wiring,
absolute-K evaluation, or voiding. Absolute KI remains NOT_RUN and the overall
production prerequisite remains OPEN.
