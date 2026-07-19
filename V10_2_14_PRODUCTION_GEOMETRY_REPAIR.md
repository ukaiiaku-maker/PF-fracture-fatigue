# v10.2.14 production-geometry repair

This branch supersedes the blocked v10.2.13 ribbon-placement attempts.

## Constitutive scope

- The active signed FEM operator is measured from a surface-terminated slip ribbon.
- The ribbon source remains at the physical crack tip/free surface.
- Eigenstrain is removed only from elements whose residual stiffness
  `g=(1-d_e)^2+kappa` is below the configured threshold.
- Partially damaged load-bearing transition elements are retained.
- Persistent wake shielding is disabled because the current scalar wake bins do
  not preserve two-dimensional signed line positions after crack advance or
  deflection.
- The same active-only kernel family is used by monotonic fracture and fatigue.

## Validation

The production-geometry regression test rebuilds the exact E000 graded mesh and
notch stamp used by the 700 K capture:

- 1667 nodes
- 3212 triangular elements
- tip-local mean size approximately 2.815211e-7 m
- element mean-damage counts 2314 at 0, 64 at 1/3, 76 at 2/3, and 758 at 1
- both reduced BCC slip traces for a 45 degree crystal orientation

Production parameterization remains blocked until the real E000 frozen-geometry
load-invariance evaluation passes and the subsequent convergence/replay gates are
completed.
