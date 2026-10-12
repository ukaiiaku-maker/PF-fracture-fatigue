# R5 dynamic voiding and state-correspondence result

## Lineage and status

- Worktree: `PF-fracture-fatigue_worktrees/r1-reduced-spatial-physics-restoration`
- Branch: `codex/r5-dynamic-void-state-correspondence`
- R4 parent: `2f689a98f1dae7734de5f8b9bec02ee596738aa1`
- Recovered void donor: `47a5a4bae048e5a8b1c6178efb6406325005f7ef`
- R4 source was preserved before edits in `preflight/r4_source_2f689a98.tar.gz`.

This is an additive R5 continuation. It does not replace the R3 topology
regressions, the R4 fixed-cavity functional test, or either fatigue strategy.

## Results

### Outer source-coupling convergence

`no_event_partition.json` reproduces the audited constant-K discrepancy:
one 8.4 s call gives `N_em=4360.146965128` and two 4.2 s calls give
`N_em=3418.293054428`. The current native fracture-action step limiter is
inactive (one native substep per outer call). The error is caused by the outer
nonlinear source/backstress, post-half-step rate, transport, and moving-window
refresh. It is not a fracture-root error. Refinement is monotone through 128
outer slices (`N_em=1913.809435690`); the final doubling changes population by
0.748%. Fracture action is reported on its absolute tiny scale rather than as a
large relative “fracture effect.”

### Signed spatial mechanics and complete-input replay

`current_law_spatial_signed/` is a fresh 10/20/30/20/30 micrometre execution.
The spatial engine no longer inherits the reduced K-only positive-sign
placeholder. Its actual resolved inputs include one positive and one negative
component in every row; the initial pair is approximately +0.316/-0.504 GPa.
The correction materially changes the positive-loading result (stage-3 active
population is about 148 rather than R4's 398).

`mechanical_correspondence.json` replays K, duration, orientation/path
convention, drive factors, and resolved signed shear. The K-only and
complete-input arms are complete-state forks of one stochastic owner. No
population or event time is imported. The complete input closes the input
operator mismatch but does not close the later state mismatch: stages 3-5
retain roughly -56%, -79%, and -79% population errors. This residual is
classified as continuous spatial remap, source-geometry/event continuation,
and re-equilibration versus the reduced outer advance—not stress agreement.

The signed consumer probe exercises the actual spatial class with
`[-2e8,+3e8] Pa`: the pair remains signed at the consumer and produces a
negative transport velocity. The positive reference remains positive.

### Dynamic recovered void owner on the current fracture backend

`dynamic_void_run/dynamic_void_r5.json` is one accepted-state execution. It
contains the current fracture engine, recovered vacancy fields and ledgers,
vacancy and direct birth clocks, losing embryo clocks, cavity/source state,
mechanics provider/cache, and loading cursor.

- A: void-off load/reload is bitwise identical to direct current-backend
  execution for the same consumed mechanics.
- B: native cavity-free incubation advances both birth mechanisms without
  forcing a birth. Vacancy action grows from zero and direct action also grows.
- C: the declared recovered near-third-hit fixture becomes a stable ~42 nm
  subgrid cavity, undergoes native source/emission growth, receives one native
  5 micrometre current-backend fracture reward at the corrected spatial
  high-load level, refreshes mechanics, and continues afterward.
- D: the same recovered scheduler exposure and partition are repeated with
  cavity mechanical feedback disabled while clocks and cavity ownership remain
  active.

The provider is re-queried after every mechanically consequential packet. Cache
identity includes material, branch, opening/load, crack extension, plastic
history, and site phase/radius. The R4 scalar fixed-void ratio is not used.
Its frozen functional test remains separately named and supported only at its
original geometry.

Every fracture preview owns a context-local geometry descriptor queue.
Rejected whole-system previews leave accepted fracture, RNG, void/vacancy,
clock, provider, geometry, and loading-cursor state unchanged.

### Fatigue convergence

`fatigue_convergence.json` forks one complete marked owner. Phase counts
12/24/48 alter the midpoint drive discretization, while native action limits
0.02/0.01/0.005 alter internal state resolution separately. Every arm includes
a native event and subsequent reload. Rows distinguish physical fractional
cycle time, continuous micro-advance, and checkpoint/reward advance.

The R4 conventions remain explicit: 23 is the number of completed cycles,
while 23.9178286745 is physical elapsed cycles. The 24-bin midpoint path and
mean-cycle strategy remain distinct and are not called quantitatively
interchangeable.

## Scope limits

- The active cavity is a recovered admissible fixture, not a claim of natural
  nucleation under the current material path.
- The recovered aligned mechanics closure is bounded to its qualified domain;
  this is not a universal moving-void surrogate.
- Complete mechanical input replay does not reproduce continuous full-spatial
  remapping or identical post-event source geometry.
- The outer source-coupling refinement is convergent but 128 slices are not an
  asymptotic zero-error proof.
- Full compressive-contact and branch-plus-cavity compositions remain future
  obligations; they were not prerequisites for this positive-load R5 closure.

## Verification

`focused_suite.xml` records 19 focused R3/R4/R5 tests passing. After the final
queue-compatibility addition, `final_full_suite.xml` records 810 tests passing
in 687.52 s on the complete repository. Commands are recorded in `commands.txt`
and primary artifact hashes in `PACKAGE_SHA256.txt`.
