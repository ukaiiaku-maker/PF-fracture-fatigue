# R1 restoration result

Status: **PARTIAL_RESTORATION — do not label `COMPLETE_RESTORATION`.**

## Outcome

R1 is implemented on `codex/r1-reduced-spatial-physics-restoration`, descended
from the latest fetched `origin/codex/shared-fracture-fatigue-options-v1`
(`995402f2e30988c032cc966fb46350d82ff12957`).  That head contains both pinned
tested ancestors: numerical source `b927dac18c58f2fb896fe3520fb4c23ba950ec7e`
and evidence source `e167bbe520019dddf1383fd2bd60a7a770d6832e`.
Calculation changes through `b8adedc82503a2ff6698d35bf3d03af671ea9f24`
add the R1 identities, live geometry/source/void ownership audit, a native
full-order finite-activity V5 fixture, explicit positive-R V5 load history and
phase-exact restart, a public typed cyclic-V5 preset, and a configurable reduced
replay integration budget. No material row, barrier, Paris exponent, toughness,
or mechanical multiplier was fitted.

The default is `RESTORED_REDUCED_REFERENCE`. `FULL_SPATIAL_REFERENCE` and
`SEAM_DIAGNOSTIC_V4_10` remain separate runnable identities. Importing the R1
registry does not install the V4.10 seam mechanics or mix its circular-cut
assumptions with the persistent-source radius closure.

## Fresh R1 executions

- The 1200 K shared-state replay contains two accepted 5 um advances, retained
  signed state, reload, and a second event. Replaying the recorded full-spatial
  schedule through the reduced state passes: maximum array relative error
  `2.7035439210412937e-14`, maximum array absolute error
  `1.4551915228366852e-11`, maximum scalar absolute error
  `2.223532646894455e-07`; all event flags match.
- The public cyclic V5 execution at `b8adedc8` completed the accepted structural
  history `1.0,0.2,1.0` in both matched cavity-on/off arms. It is source-attested,
  uses the native full spatial FEM/event scheduler, and has an exact native
  checkpoint round trip. It is a no-event neutrality/cyclic-state check, not a
  cavity-growth claim.
- A native full-order V5 functional fixture exercises plastic vacancy production,
  multi-hit birth, stabilization, conservative vacancy debit, subgrid growth,
  promotion, body-fitted remesh, resolved growth, full spatial FEM, and exact
  restart. Radius grows from 1 to 3 um and the vacancy conservation residual is
  zero. It starts from a declared 1% vacancy field, uses no rate multiplier, and
  is explicitly `production_eligible_material_evidence=false` and
  `natural_nucleation_claim=false`.
- A direct interrupted/uninterrupted positive-R V5 comparison is terminal-state
  exact in both arms after the same `1.0,0.2,1.0` history.
- The inherited qualified long spatial history remains callable and preserved:
  1002.9028827375072 um projected extension, 201 events, 2,085 accepted rows,
  stage-aware K/J observations, and the separate pinned FEM/CZM comparator.

## What is not restored to the requested completion threshold

The full-order V5 equilibrium currently accepts positive opening only. A real
negative applied displacement attempt reached the native equilibrium boundary;
R1 now rejects nonpositive V5 load factors before execution rather than
mislabeling internal recovery as applied reversal. Therefore a V5 applied-
reversal cavity campaign is not qualified.

The combined branching-plus-cavity executor remains open; existing branch and
cavity paths are independently callable, but R1 does not claim their joint
calibration. The finite-activity cavity trajectory is a functional initial-
condition fixture, not natural nucleation or crack-void coalescence material
evidence. The fresh reduced replay is an exact same-contract state replay of a
recorded 2-D schedule, not independent shape validation: the full spatial
sharp-front model uses an analytical local kinetic radius rather than a rounded
FEM notch. The fixed-range-plus-branch composition also remains unmapped.

The production negative-R run is recorded separately in the compact evidence
and its terminal status governs whether it can count beyond a nonzero-state
reversal check. These limitations prevent the full-restoration label.

## Preservation and provenance

No donor worktree, historical result, mesh, checkpoint, or calibration data was
deleted or reset. The unrelated dirty V4.10 worktree was inspected read-only and
left unchanged. Generated R1 runs remain under ignored `runs/`; compact,
reviewable evidence is committed under `integration_evidence/r1_restoration/`.
The generated four-class runtime registry is installed from committed source at
execution time and remains ignored by design.

## Next search

Do not resume a barrier search on the seam diagnostic. First close, in order:
(1) a supported applied-reversal structural equilibrium for the full spatial V5
path or a clearly separate reversible reduced cavity surrogate; (2) a native
material-row finite-activity cavity/crack interaction; (3) common event ordering
for branch birth and cavity connection; and (4) a same-load independent shape
comparison. Then use `RESTORED_REDUCED_REFERENCE` for screening and confirm
selected rows with `FULL_SPATIAL_REFERENCE`, with no slope target or refit.
