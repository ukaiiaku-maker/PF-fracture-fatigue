# V7 piecewise-equilibrated traction diagnosis

Status: **blocked with exact traction-verification failure**.

The V1--V6 physics, accepted equilibrium observations, source tensor, material
identity, 512-edge cavity geometry, thresholds, RNG state, and D/E mesh family
are unchanged.  The raw CST result remains a non-authoritative diagnostic.
No 18-state oracle, production birth event, OneD fit, paired trajectory, or
fatigue run was started.

## Arc-resolved diagnosis

The complete raw CST norms reproduce the retained values to roundoff:

| level | complete raw norm | fixed-source-window norm | junction norm |
|---|---:|---:|---:|
| D | 0.0580471402 | 0.1112883683 | 0.0068465519 |
| E | 0.0553974176 | 0.1071823959 | 0.0063630856 |

The fixed downstream source window contains 74 of 512 edges but contributes
about 53.1% (D) and 54.1% (E) of the complete squared L2 traction.  The crack--
cavity junction contributes only about 0.20% and 0.19%.  The ten largest edges
contribute only about 12.1% (D) and 11.2% (E), and their adjacent-element
qualities are well above 0.05.  The failure is therefore distributed over the
smooth downstream source arc; it is not a junction singularity, duplicate-edge
error, omitted-edge error, or a few poor mesh-transition elements.

Every exported row records the actual polygon tangent and outward normal,
nominal-circle normal, arc coordinate, edge length, unique adjacent element and
quality, normal/shear traction, L2/resultant/moment contributions, plastic
strain, density, and distance from both source and connection.  Direct tests
cover edge order and orientation, polygon versus nominal normals, source-window
membership, length weighting, normalization, duplicate edges, and one-owner
identity.

## Independent V2 verifier

`PIECEWISE_EQUILIBRATED_BOUNDARY_STRESS_V2` uses an affine symmetric stress in
each element of the fixed physical patch.  Local divergence is zero by
parameterization.  A constrained least-squares projection fits the accepted
CST stresses while matching the two traction components at both endpoints of
every internal patch edge.  Patch resultants and moments are independently
integrated after reconstruction.  Cavity traction is never imposed.

The prospective manufactured qualification passes constant and affine
self-equilibrated fields, exact local equilibrium, interface continuity,
resultant/moment balance, rigid rotation, reversed edge orientation/order,
Kirsch, reflection, and two shape-regular refinement levels.  The central
results are:

| level | piecewise traction | fit residual | max interface jump |
|---|---:|---:|---:|
| D | 0.1376379 | 0.04391 | 6.81e-13 |
| E | 0.1333688 | 0.04388 | 2.61e-12 |

The D-to-E absolute change is 0.00427 and the fractional change is about 3.20%,
so convergence passes.  Rank/conditioning, local equilibrium, interface
continuity, resultant/moment balance, weak FEM residual, source-tensor
convergence, reactions/compliance/energy, mesh quality, and accepted-state
immutability pass.  Boundary traction alone remains above the frozen 0.05
limit.

Therefore:

```text
TRACTION_FREE_BOUNDARY_VERIFICATION = FAIL
RAW_CST_TRACTION = RETAINED_DIAGNOSTIC_FAIL
DBTT_SOURCE_READINESS = BLOCKED_WITH_EXACT_FAILURE
```

This result does not alter fracture or void physics and does not itself select
a finite activation-zone observable.  It closes numerical point-source
qualification at the requested stop and isolates the next possible inquiry to
field transfer, local constitutive state, or a separately derived physical
activation-zone work observable.
