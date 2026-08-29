# Existing source and mesh audit

## Frozen lineage

- Selected base branch: `codex/canonical-pf-full-trajectory-and-mechanism-audit`
- Selected base commit: `7e71b8f27b0682060fd161e7e5e8fe0d3587e8ac`
- Qualified ancestor and merge base: `9e884fb0b0845da621d2612bdf1042e481b8df49`
- New branch/worktree: `codex/sharp-front-stateful-voiding-v1`, `/private/tmp/pf-sharp-front-stateful-voiding-v1`

The descendant changes three physical package files: `sharp_front.py`, `sharp_front_v10_1_7_5.py`, and `anisotropic_emission_v10174.py`. They retain process-zone state and add sparse default-off observer/tensor forwarding. The other 498 changed paths are tests, runners, canonical analyses, compact results, and figures. The source worktree had an unrelated untracked archive and was not modified.

## Ownership map

`mesh.py` owns Delaunay creation, connectivity, CST gradients, and rebuilding. `fem.py` owns plane-strain CST assembly, natural zero-traction boundaries, reactions, residuals, and energy. `sharp_front.py` owns the finite-width stiffness-killed wake, active-front state, analytical tip radius, event candidates, tensor observers, and accepted-event lifecycle. Persistent-site modules own first-passage thresholds and RNG. The v10.2.30 checkpoint modules demonstrate exact serialization/restoration of active fatigue-side state.

The existing Delaunay/CST solver directly supports a hole by retaining a closed polygonal ring and removing all interior triangles. An internal boundary not listed in Dirichlet or Neumann data has the correct natural zero traction. Node duplication is not required for a closed hole. `rebuild_tri_mesh` validates and recomputes mechanics data after connectivity changes. No new dependency or solver replacement is required.

V1 isolates hole construction behind `make_explicit_circular_hole_mesh`. Every topology-changing operation is copy-on-trial: registry, geometry lineage, clocks, thresholds, and RNG are restored on remesh, equilibrium, or late-veto failure.
