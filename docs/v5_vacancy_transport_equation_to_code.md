# V5 vacancy transport: executed equation and correction-active scope

The production implementation is
`arrhenius_fracture/vacancy_transport_phase_field_v2.py`. The centered mixed
RT0/P0 method is the named high-order reference. When its P0 concentration is
negative, production instead accepts the constrained face-flux formulation
described below. It is therefore a modified discrete method in an
active-correction interval, not a line search that recovers the centered fixed
point.

## Unknowns, units, and signs

Each solid triangle owns one atomic vacancy fraction `c` (dimensionless). Each
registered non-impermeable face owns one oriented integrated normal flux `q`
(vacancy-fraction area per time, `m^2/s`, in the 2-D unit-thickness model). An
intact interior face has one shared `q`: its first owner receives `+q` in `B*q`
and its second owner receives `-q`. Positive flux on a one-owner boundary leaves
the solid.

The continuum flux represented by RT0 is

`j = -D grad(c) + D Omega_f/(k_B T) c(1-c) grad(sigma_m)`.

Here `j` has units `m/s`, `D` is `m^2/s`, and `Omega_f` is `m^3`. Stress is
tensile-positive. The sign follows
`mu = E_f - T S_f + k_B T log(c/(1-c)) - Omega_f sigma_m`, so tensile stress
lowers vacancy chemical potential. `q` is the line integral of `j.n` over a
transport face.

For triangle area `A`, backward Euler executes

`A(c-c0)/dt + div(q) = A*s0*(1-c) - A*lambda_climb*(c-c_eq)`

`                         - A*sum_i lambda_i*w_i*max(c-c_eq,0)`.

`s0` and bulk `lambda` values have units `1/s`. Normalized site support `w_i`
has units `1/m^2`, while `lambda_i` (`capture_rate_area_per_s`) has units
`m^2/s`. Every integrated cell term therefore has units `m^2/s`. Plastic
production is positive. Positive climb/capture removes vacancy above
equilibrium; negative climb exchange is recorded as emission.

At a fixed Picard/active-set iterate the cell block is

`C*c + B*q = r`,

where `C = A/dt + A*(s0 + lambda_climb + lambda_capture_active)` is positive,
and `r = A*c0/dt + A*(s0 + lambda_climb*c_eq +
lambda_capture_active*c_eq)`. The centered constitutive block is
`M*q - B.T*c = f_drift+f_Dirichlet`; `M` uses exact degree-two triangle
quadrature of the RT0 element mass matrix divided by `D`.

## Boundary and topology ownership

- Symmetry and contact/grip faces are absent from the flux unknowns and are
  exactly zero flux.
- Intact interior faces own one shared flux with equal-and-opposite cell signs.
- Crack, external free-surface, active-cavity, and merged-wake faces are
  one-owner Dirichlet sink faces. Their concentration is the zero-stress
  thermodynamic equilibrium unless an exactly covering manufactured reservoir
  map is supplied by a diagnostic test.
- The exact-cut mesh duplicates accepted crack flanks. There is no shared face
  and no transport across an opened slit; intact material ahead remains
  connected.
- Active-cavity absorption is one-way at acceptance. A negative value beyond
  conservation roundoff is rejected. An external reservoir can exchange with
  either sign, retained in the external-surface ledger.

## Conservative positive face-flux formulation

An admissible centered candidate is accepted unchanged. Otherwise the code
scales all integrated fluxes by

`q_scale = max(norm_inf(q_H), norm_inf(r), tiny)`

and solves

`min_x 0.5*||x-x_H||_2^2`, subject to
`B*x <= (r-C*c_floor)/q_scale`,

where `x=q/q_scale`. Concentration is reconstructed as
`c=(r-B*q)/diag(C)`. It is never clipped or globally renormalized. The dilute
upper limit (`c <= 0.01`) is outside the projection and remains fail-closed.

Scaled ADMM uses `rho=1`, at most 20,000 iterations, and relative primal/dual
tolerance `5e-15`. Iteration-count exit is not acceptance. The code also tests
inequality violation, stationarity, dual feasibility, and complementarity,
rejecting unless every scaled KKT residual is below
`max(4*tol*scale,128*eps*scale)`. The earlier `5e-13` stop was too loose: an
active refinement probe could terminate with a reconstructed value slightly
below its requested floor. The tighter stop and KKT gate repair that specific
defect.

The metric is Euclidean in registered **integrated** face DOFs. A common change
of units multiplies `q_H`, `r`, and `q_scale` together, leaving the dimensionless
problem and minimizer unchanged. The metric is not claimed as an abstract
mesh-independent RT0 norm. Its refinement behavior is bounded empirically
below: correction support collapses, correction vanishes on the finest mesh,
and field, reservoir flux, capture, and birth action converge. Qualification is
limited to this family; materially different anisotropic meshes must repeat the
check. No observed consistency defect justifies changing the metric here.

## Nonlinear and ledger consistency

Correction occurs **inside** each Picard iteration. Drift mobility uses the
current `c*(1-c)`. Plastic saturation, climb equilibrium, and site-capture
active sets enter `C,r`; the corrected candidate becomes the next iterate.
Convergence requires both the concentration residual and exact agreement of
assembled and candidate capture active sets, preventing stale coefficients.

After convergence, boundary fluxes, RT0 vectors, plastic production, climb
exchange, site capture, reserved inventory, birth action, cavity absorption,
and global inventory are recomputed from the same accepted `c,q`.

Three residual classes have different meanings:

1. `local_element_conservation_residual_inf` and
   `accepted_cell_balance_block_residual_inf` measure cell balance and must be
   at roundoff.
2. `centered_constitutive_residual_inf` applies the centered RT0 constitutive
   equation to corrected flux. It is intentionally nonzero when correction is
   active and measures the modification.
3. Per-correction ADMM primal/dual and KKT residuals, together with the Picard
   residual and active-set match, test corrected feasibility and optimality.

## Spatial checks and scope

The original Neumann test uses `T=1800 K`, `L=4 um`, `dt=1e-3 s`, zero boundary
flux, and `c0=1e-5+2e-6 cos(pi*x/L)`. Its backward-Euler analytic amplitude is
`2e-6/(1+dt*D*(pi/L)^2)`. Error is the dimensional area-weighted L2 norm
`sqrt(sum A*(c-c_exact)^2)`. The 16/64/256-element errors are
`5.4694556e-15`, `1.6710501e-15`, and `2.9969943e-16 m`; correction is inactive
at all levels. This qualifies only the centered/reference path.

The supplemental active check uses the same full equations on fixed graded,
affine-sheared 16x8, 32x16, and 64x32 meshes (256, 1,024, and 4,096 triangles).
It advances a smooth compact vacancy packet at `1800 K` for `1e-4 s`, with a
zero-concentration left reservoir, impermeable remaining boundaries, and one
native localized capture site. The finest full-order solve is an internal
numerical reference, not reduced or paper data. Correction is active on the
first two levels and inactive on the finest. From 256 to 1,024 cells:

- fixed-bin normalized concentration error falls `0.169619 -> 0.039290`;
- captured-area and birth-action error falls `2.074% -> 0.334%`;
- reservoir-flux error falls `49.02% -> 6.89%`;
- local correction/affected-flux ratio falls `0.5521 -> 0.1469`; and
- integrated absolute concentration correction falls
  `4.7925e-21 -> 5.2618e-23 m^2`.

This qualifies the tested correction-active transport/site-delivery regime. It
does not turn the manufactured reservoir fixture into physical cavity evidence
or claim unconditional stability on every future mesh.
