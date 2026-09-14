# Corrected thermodynamic joint-search protocol

This protocol was frozen before generation of any v2 candidate. The immutable
parent is commit `97916f6f42ba063d5d6fe4a8c4590bd35047b197`. The v1 artifacts remain
unchanged and serve as the positive/sign-ambiguous entropy and intrinsic-opening
baseline.

## Thermodynamic convention

The conventional activation entropy is
`activation_entropy_over_kB = Delta S^‡/k_B = -(1/k_B) partial_T Delta G^‡`.
Every v2 record also stores
`barrier_temperature_derivative_over_kB = -activation_entropy_over_kB`.
The primary emission prior is `Delta S_emit^‡/k_B in [-50,-30]`; the complete
structured set is `[-60,-50,-45,-40,-35,-30,-25,-20,-10,0,10,20,30,40,50,60]`.
Positive values are sign controls. Entropy enters only the direct free-energy
surface. Attempt frequencies remain fixed and no entropy prefactor is added.

The reference surface is fixed at 300 K. Entropy and bounded activation heat
capacity therefore leave every paired v1/v2 300 K free-energy surface exactly
equal. Opening entropy remains symmetric over `[-60,60] k_B`; activation heat
capacity remains in `[-15,15] k_B` with zero as the minimum-complexity case.

## Deterministic banks and gates

The structured bank contains 3,072 rows. The paired primary bank contains
131,072 rows using scrambled Sobol seeds 260911 and 260912. The conditional
paired heat-capacity bank contains 32,768 rows using seed 260913. Every pair
inherits all nonentropy coordinates and has an explicit legacy ID.

Both opening and emission must have finite, strictly positive free energies on
the frozen 300--1200 K and 0--30 GPa audit domain. Analytic/finite-difference
entropy and activation volume, the Maxwell identity, and exact 300 K recovery
must pass. Numerical floor and ceiling path occupancy must each be below 0.80;
an isolated constitutive ceiling contact is diagnostic and is not rejection.
The complete path may not lose rate discrimination. Collapsed controls remain
recorded and cannot be promoted.

The P25/P40 300 K fatigue gates are unchanged: RMS log-rate error <=0.05 decade,
maximum error <=0.10 decade, absolute global-slope change <=0.15, maximum
adjacent-local-slope change <=0.30, positive monotonic rate, and no new renewal
ceiling. Nonreference diagnostics are frozen at 600 and 1200 K and are not fit
to a target.

## Fidelity and promotion

F0 is an intrinsic-opening prescreen only. F1A is the existing continuous
emission/reduced-blunting diagnostic and cannot promote a system class. F1B is
the actual production persistent-source process engine with signed mobile and
retained populations, Peierls transport, Taylor release, retention and physical
return, source multiplicity, backstress, active/wake/total shielding, and radius
evolution. F2R uses the same production reduced-model execution path with fresh
state or parity-qualified restoration and common random numbers. No discrete
emission clock or cleavage/emission race is permitted.

The original 168-condition comparison is preserved as 24 deterministically
selected rows times temperatures `[300,450,600,750,900,1050,1200]` at
`Kdot=0.005 MPa sqrt(m)/s` and `Xi=ln(2)`. Separate F1B loading-rate anchors use
`Kdot=[0.0005,0.005,0.05]` at `[450,750,1050] K`. Missing states remain missing.
F1B promotion requires at least 80% accessible core conditions and every
class-defining anchor. At most six F1B finalists enter F2R; selection is
deterministic by admissibility, coupled topology, sign-control coverage,
minimum complexity, and numerical margin.

The numerical topology gates remain: ceramic endpoint ratio <=0.70; weak-T
span <=1.25; DBTT increase ratio >=1.50 over a contiguous >=100 K interval with
the expected loading-rate shift; Peak-T prominence >=15% with valid branches
and a positive-to-negative derivative reversal. These gates apply only to F1B
and F2R curves and require the causal state conditions in the JSON protocol.

## Execution boundary

The analytical bank uses one worker. F1B/F2R use at most two reduced-model
workers system-wide. Unknown runtime model IDs, failed initialization/restore
parity, state-conservation failure, stale state identity, generic renewal
substitution, or missing ledgers fail closed. F2R stops after at most six rows
and 78 preregistered core/rate-anchor conditions, or earlier if no F1B row is
promotable. Two-dimensional PF, FEM/CZM, multifront, branching, and 1000-um
continuations are outside authorization.

