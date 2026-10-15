# FEM observer and specimen-restart completion

Date: 2026-10-14

## Result

The R08 elastic observer discrepancy is resolved without changing the mesh,
elastic tensor, crack representation, J implementation, kinetic laws, or any
consumed drive.  The generic current-PF companion now has a real native
specimen load-and-continue path.  The original historical benchmark and the
invalid refinement receipt remain unchanged.

## What the two original quantities measured

The native domain integral and the reported fixed-grip derivative were not the
same virtual operation.

- Native J integrates the accepted anisotropic elastic field over a plateau
  material-coordinate virtual extension.  The three physical outer radii are
  180, 240, and 300 micrometres.  The crack direction is +x, boundary nodes are
  outside the varied interior support, and the full stored-energy stress is
  used.
- The old independent derivative advances a finite-height nodal killed-notch
  band and divides the total stored-energy change by the furthest damaged-node
  x coordinate.  On the unstructured reachable-envelope mesh, that projected
  length is not proportional to either added nodal-damage area or stiffness-loss
  area.  It is a useful finite-band growth diagnostic, but not J's local
  coordinate variation.

At the fine level this intentionally different comparison is therefore retained
as 1.9372500338123593 / 2.3540644513243905 = 0.8229384003154491.  It is not
scaled or relabelled.

## Matched operation

The new independent observer solves the same accepted body after centered
plus/minus material-coordinate perturbations using the identical plateau weight
as each J contour.  Topology, nodal damage, boundary nodes, load, constitutive
tensor, and internal state are held fixed; total stored energy is redifferenced
after equilibrium.  The coordinate perturbation is 50 nm and is stable against
100 and 25 nm samples to the regression tolerance.

| h/Lpz | outer virtual G, J/m2 | outer native J, J/m2 | J/G |
|---:|---:|---:|---:|
| 0.245 | 1.9381985019543968 | 1.9549015251592499 | 1.0086178083349102 |
| 0.180 | 1.9525004344182404 | 1.9582919952030504 | 1.0029662276549176 |
| 0.120 | 1.9499277439764817 | 1.9648572263318205 | 1.0076564285017522 |

The maximum discrepancy is 0.861781%, below the predeclared 2% criterion.
Medium-to-fine change is 0.131938% for matched virtual G and 0.417244% for
native median J.  Individual contour values and active-element counts are in
`evidence/fem_baseline_reconciliation_v2.json`.

### Consequence map

The correction is observer-only.  Native J remains the quantity converted to
K and consumed by the hazard/path-selection code.  The projected-band G and
matched virtual G are diagnostic outputs and do not feed a rate, admission
gate, or topology proposal.  Thus no historical trajectory was renumbered and
no kinetic rerun is implied by the observer repair.

## Metadata owner repair

The retained four-event run ended after its valid atomic transitions because
the terminal verifier could no longer see the live refinement-provider record.
The current companion now recovers only the provider's persisted
`compact_corridor_mesh_v91852.json` record and verifies all of:

- exact requested physical radius;
- preserved physical-refinement metadata;
- detected production physical provider;
- active path-aware refinement;
- the exact global-forward reachable-envelope policy.

A provider or support mismatch remains rejected.  The generic entry installs
this verified fallback transactionally and restores the prior hook afterward.

## Native specimen restart

`fem-czm-specimen-checkpoint/1` adds only the specimen envelope around existing
native state owners.  It stores mesh/connectivity, displacement, nodal damage,
bulk plastic/density and spatial PZ fields, accepted loading/controller state,
front graph and path state, topology backend bookkeeping, observation history,
and provider attestation.  Kinetic state is captured/restored through the
most-derived native engine `_capture_state/_restore_state`; signed MPZ state is
reconstructed through its native `from_state_dict`; cohesive separation,
damage, contact/history, and numerical subsegments use
`CohesiveNetwork.state_dict/from_state_dict`.  The stochastic geometry
adapter's pending and realized event queues are explicitly owned and restored.

Checkpoint publication is atomic and binds a SHA-256 digest of the no-pickle
NPZ array payload.  Runtime engine/backend class identities, physics arguments,
temperature, and persisted physical-refinement provider are fail-closed.

### Executed current-law transition

Using seed 8666, the selected 0.10 eV Taylor-margin Peak row, the active-only
kernel-family compatibility input, and the dedicated `arrhenius-fem-czm`
environment with `triangle==20250106`:

- accepted step 121 checkpoint: threshold 0.8573135604369198, hazard action
  0.2735864819783538, B = 0.3191206748659425, no crack event;
- fresh-process resume: two physical stochastic events were accepted at steps
  140 and 142;
- the two events produced six numerical cohesive subsegments, but only two
  stochastic event marks and two MPZ translations;
- accepted step 145: 2029 nodes, 3904 bulk triangles, six cohesive elements,
  tip x = 0.0005062364217294542 m, MPZ cumulative advance
  7.394281137649096e-06 m, next threshold 0.5250238935573996, and B =
  0.47997815545598504;
- a second fresh-process load from the active post-event checkpoint completed
  accepted step 146 and post-event equilibrium.

The attempted crossing in the PF/Codex interpreter is retained as negative
environment evidence: all 600 corridor candidates failed because that
interpreter lacks the mandatory `triangle` dependency, and the solver restored
the renewal origin and stopped.  This is not a mechanics or restart failure.

The uninterrupted and fresh-process resumed `steps_1000K.csv`, stochastic
geometry record, atomic-corridor audit, and scientific summary are byte-for-byte
identical through step 145; no timestamp, path, or process-local normalization
was required.  The comparison, hashes, and active post-event continuation are
recorded in `SPECIMEN_RESTART_QUALIFICATION.json`.

The executed FEM source tree is committed as
`784c57f2b48172c6bd5b22f6892faff623cbb1f3`.  The focused observer,
attestation, restart, stochastic-ownership, historical-reference, and physical
refinement group passes 41 tests.  The complete FEM repository passes 713 tests
plus 16 subtests.  After advancing the source-bound external binding, the
complete PF/shared repository passes 960 tests with 15 intentionally
deselected tests.

## Opening-only execution and common PF/FEM comparison

The generic FEM adapter now maps `accommodation_mode=opening_only` to the
native `--disable-tip-accommodation` control.  The control is enforced by the
most-derived persistent stochastic owner, not merely by zeroing a base
prefactor.  A real two-step spatial run retained mechanics and cleavage-clock
evolution while emission, mobile, retained, escaped, and emitted-work channels
were exactly zero at both steps.

An independent three-level PF mechanics sequence and the qualified FEM
path-envelope sequence were then compared at the same 0.2 micrometre total
grip opening.  PF medium-to-fine reaction and native K changes are 0.02014%
and 0.00113%; FEM reaction and native-J changes are 0.22764% and 0.41724%.
The fine PF/FEM reaction and global stored-energy ratio is 1.118727.  This is
a converged representation-dependent offset, not a remaining mesh-convergence
failure and not authority to scale either backend.  PF native domain-J/K,
FEM native full-energy J, and FEM matched material-coordinate G remain
separately identified in `PF_FEM_COMMON_MECHANICS_COMPARISON.json`.

## Scope remaining

This closes the generic tip-only active-accommodation specimen restart gap,
the opening-only state-owner gap, the R08 matched-operation observer identity,
and the scoped independently refined common-mechanics comparison.  It does not select a physical
activation area/interface stiffness for the bilinear law, add plane-stress or
generalized transverse cracked-body execution, connect generic full-field
source scope, validate spontaneous recursive branching, or complete the local
cavity/combined branch-cavity research items.  Those remain separate R04-R10
statuses and were not converted into claims by this work.

Production and manuscript modification remain disabled.
