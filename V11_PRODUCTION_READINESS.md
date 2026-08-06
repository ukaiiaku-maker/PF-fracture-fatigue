# V11 production readiness

Status: **blocked; do not launch the four-class campaign**.

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

Evidence is under
`runs/v11_canonical_45deg_700K_seed3621_1000um_long_growth_v1`, including the
latest atomic checkpoint, action/energy ledgers, failure summary, full
traceback, and inspectable failure snapshot.
