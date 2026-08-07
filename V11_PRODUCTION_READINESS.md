# V11 production readiness

Status: **blocked; do not launch the four-class campaign**.

## Adaptive nested-refinement follow-up

Topology-preserving conforming edge subdivision is now implemented.  The
refined P1 space contains the parent P1 space; old nodes are retained, midpoint
values use exact parent shape functions, element plastic/dislocation histories
and P0 stiffness degradation are inherited by every child, and boundary edge
classes are inherited.  The preserved step-481 Stage-A energy comparison is
bitwise exact (`247.71266262810803 J/m` before and after prolongation), replacing
the rejected global-remesh discontinuity.

A clean seed-3621 trajectory was rerun from initiation with proactive union
refinement before sibling A1/A2/A12 trials.  Its 40 µm gate reached eight
committed births, five active tips, four coalescences, mesh generation 43, 2,498
nodes, and 4,882 elements.  Stage-A relative error stayed below 2.28e-16.  An
actual duplicated restart from the refined checkpoint reproduced the exact
mesh/topology fingerprints, directional state, process/junction state, energy
ledger, RNG state, trial log, and next 45 µm checkpoint.

The adaptive long continuation then reached an accepted 75 µm root-to-tip
checkpoint (step 332, generation 80, 3,745 nodes, 7,368 elements) and failed
closed before the next energy evaluation.  One active tip
(`b5f2bd5610a01132`) has local hbar about 1.0 µm, but both candidate 5 µm
segments change zero stiffness elements.  The preceding sharp-wake advances
have already killed the complete prospective corridor through the inherited
element-support halo.  Exact nested parent-to-child damage inheritance cannot
un-kill it; further subdivision therefore cannot make either trial mechanically
distinct.

This is a new crack-representation blocker, not a mesh-spacing or physical
energy veto.  Repair would require changing how sharp-wake support is committed
so it cannot create ahead-of-tip killed material (or introducing a separately
qualified discontinuous crack representation).  That is outside adaptive
refinement and must not be hidden by repartitioning inherited damage, reducing
event rewards, or changing energy tolerances.

The canonical weakT, 700 K, 45 degree, seed-3621 continuation was resumed from
the accepted 100 µm legacy checkpoint using the exact-topology live FEM
provider.  Correct graph accounting showed that checkpoint contains 100 µm of
total new network length but only 40 µm of maximum root-to-tip path extension.

The continuation reached an accepted checkpoint at step 481 with:

- 60 µm maximum root-to-tip extension;
- 125 µm total new crack-network length;
- 42.4264068712 µm maximum forward projection;
- 7 committed branch births, 6 active tips, and 2 coalescences.

Long-growth qualification then exposed a multi-tip mesh-validity blocker.  The
static mesh remains refined around the original tip instead of following every
active tip.  Measured local active-tip element sizes range from 5.262 to
11.396 µm while the physical advance increment is 5 µm.  Exact-topology trials
at step 481 consequently report energy release at floating-point zero
(-2.84e-14 J/m) for positive finite directional J, and every admissible action
is rolled back with `insufficient_whole_topology_energy_release`.  Pending
events remain preserved and adaptive steps shrink, so continuing would be an
asymptotic numerical stall rather than qualified crack growth.

Production requires deterministic multi-tip-following remeshing with
conservative transfer of displacement, damage, plastic strain, dislocation
density, tip-local state, and junction/wake reservoirs.  Reinitializing or
partitioning these histories is not acceptable.  The 1000 µm qualification and
mid-run restart comparison remain incomplete until that capability is
implemented and requalified.

A deterministic global multi-center Delaunay rebuild was evaluated as a repair
and rejected.  It reduced all six active-tip local element sizes to
1.11--1.40 µm and preserved the area integrals of plastic strain and
dislocation density exactly, but fixed-load re-equilibration changed accepted
stored energy from 247.7126626281 to 225.9155861982 J/m (−8.80%).  That violates
the accepted-state parity requirement.  The remaining implementation need is
topology-preserving local refinement, or an equivalently qualified variational
state transfer, with energy/reaction parity at the remesh boundary.

Evidence is under
`runs/v11_canonical_45deg_700K_seed3621_1000um_long_growth_v1`, including the
latest atomic checkpoint, action/energy ledgers, failure summary, full
traceback, and inspectable failure snapshot.
