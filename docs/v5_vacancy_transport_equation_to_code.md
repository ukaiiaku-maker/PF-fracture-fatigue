# V5 vacancy transport: equation-to-code map

This note freezes the production equation and inventory meanings before the
positivity repair.  The active implementation is
`arrhenius_fracture/vacancy_transport_phase_field_v2.py`.

## Transport equation

The cell-owned unknown is the atomic vacancy fraction `c`.  On each connected
solid component the implemented flux is

`j = -D grad(c) + D Omega_f/(k_B T) c(1-c) grad(sigma_m)`.

`D` uses the row-owned migration prefactor and energy, `Omega_f` is the vacancy
formation volume, and tensile-positive mean stress lowers the chemical
potential through `mu = E_f - T S_f + k_B T log(c/(1-c)) - Omega_f sigma_m`.
The production spatial operator is the mixed RT0/P0 element-mass method in
`_rt0_system`: one oriented integrated normal-flux degree of freedom is shared
by the two owners of every intact interior face.  Crack, cavity, grip, symmetry,
and external boundaries come only from the accepted `TransportSurfaceRegistry`.

Backward Euler advances

`A (c-c0)/dt + div(j) = A s_p (1-c) - A lambda_climb (c-c_eq)
                         - A sum_i lambda_i max(c-c_eq,0)`.

The plastic source `s_p` contains the baseline and the material-row interaction
partition.  Climb contains the dislocation-density sink and, when configured,
the independent bulk-sink spacing.  Site capture uses normalized compact
supports and debits the field while crediting the same site's reserved embryo
inventory.  Perfect-sink concentrations use the zero-stress thermodynamic
equilibrium unless a diagnostic manufactured boundary value is supplied.

## Units and ledgers

Cell integration first produces vacancy-fraction area (`m^2` in the 2-D unit
thickness model).  Multiplication by
`vacancy_formation_volume/atomic_volume` converts it to the owned inventory-area
ledger.  Plastic production is a credit.  Climb annihilation, crack/external
surface absorption, and embryo healing are debits; climb emission is a credit.
Active-cavity surface absorption is simultaneously removed from the field and
credited to active-cavity inventory.  Site capture is simultaneously removed
from the field and credited to reserved embryo inventory.  These transfers are
counted once in `solve_transport_interval`, and the resulting state is the sole
input to capture, growth, hazard, and mechanics coupling.

## Numerical repair

An admissibility-preserving Picard line search was tested first.  On the
retained failure it approaches the lower boundary while the undamped centered
RT0/P0 fixed point remains negative, establishing that nonlinear damping alone
is insufficient.

The production repair therefore keeps the RT0 element-mass solve as the
high-order reference and applies a conservative correction only when that
reference has a negative cell.  Among the registered integrated face fluxes it
finds the minimum Euclidean change subject to positive backward-Euler cell
balances.  Concentration is then *derived* from `C c + B q = r`; it is not
clipped or globally renormalized.  An interior face remains one unknown with
opposite signs in its two owners, and the correction cannot cross a blocked or
absent topology edge.  The correction is identically inactive for an
admissible high-order state.  Its magnitude and the remaining high-order
constitutive residual are reported explicitly.

The unchanged dilute upper limit is not included in this numerical projection:
a converged upper crossing is still rejected as a physical-domain limit.  No
deleted cell, altered source, or altered boundary classification is permitted.
