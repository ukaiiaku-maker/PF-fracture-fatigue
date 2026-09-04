# V5 single-void model specification

Status: `DIAGNOSTIC_SINGLE_VOID_REFERENCE`, `NOT_MATERIAL_CALIBRATED`, and
`NOT_CANONICAL_MATERIAL_REGISTRY`.

This specification describes the implemented two-dimensional plane-strain,
single-void capability. It is a software and diagnostic mechanics model, not a
calibrated tungsten material model and not experimental validation.

## State and eligibility

A `VoidSite` is eligible only in `AVAILABLE_SITE`. Its unit normal `n` is stored
as `normal_xy`; its tangent is `t=(-n_y,n_x)`. A site owns three integrated
hazard clocks, its multi-hit count, candidate weight, and RNG state. One stable
site may own at most one cavity; duplicate creation fails closed.

For the local symmetric Cauchy tensor `sigma` (Pa),

- `sigma_h = tr(sigma)/2` (Pa),
- `sigma_n = n.sigma.n` (Pa),
- `tau_nt = t.sigma.n` (Pa, signed),
- `sigma_1 = max_eigenvalue(sigma)` (Pa), and
- `sigma_vm = sqrt(3 dev(sigma):dev(sigma)/2)` (Pa).

There is no scalar resolved-void damage variable.

## Activated clocks

The birth work is

`W_b = A_act(c_h sigma_h + c_n sigma_n + c_s tau_nt)` (J).

The birth rate is

`lambda_b = nu exp[-max(Q_b-W_b,0)/(k_B T)]` (s^-1).

Only `lambda_b` is multiplied by dimensionless `candidate_weight` during clock
integration. Stabilization, healing, and growth never inherit that weight.
Birth completes after `required_birth_hits` threshold crossings. After every
nonterminal hit, exactly one new unit-exponential threshold is drawn from the
state-owned RNG; the RNG advances exactly once.

Stabilization uses `lambda_s = nu exp[-max(Q_s-A_act sigma_1,0)/(k_B T)]`.
Healing uses `lambda_h = nu exp[-max(Q_h+A_act sigma_1,0)/(k_B T)]`; tensile
opening therefore favors stabilization and suppresses healing. Equal crossing
times resolve deterministically to healing. A healed embryo creates no geometry
and no inventory.

## Growth and shrinkage

Surface reaction, vacancy transport, and plastic accommodation each use the
same Arrhenius form with barriers `Q_surface`, `Q_diff`, and `Q_plast`. Their
positive series-limited rate is

`lambda_g = 1/(1/lambda_surface + 1/lambda_diff + 1/lambda_plast)`.

If a required positive channel is unavailable, positive growth is zero. The
radial increment is `dR = l_g lambda_g dt` for positive signed chemical-
potential drive. Zero drive gives zero growth. Negative drive invokes only the
explicit shrinkage branch and cannot remove more than half the current radius
in one update.

Plane-strain inventory is area per unit thickness:

`A_void = pi R_void^2` (m^2), `dA = pi(R_new^2-R_old^2)` (m^2).

The inventory ledger is updated by exactly `dA`. It is not a spherical volume.

## Promotion and topology

A stable subgrid cavity promotes when `R_void >= promotion_radius_m`. Promotion
builds the explicit body-fitted free boundary, transfers state, rebuilds exact
V12 support, equilibrates the production FEM system, and atomically commits.

Ligament rupture is an existing-cleavage-law event. The selected candidate,
direction, first cavity-boundary intersection, stochastic threshold, barrier,
positive hazard-derived dissipation, graph edit, support rebuild, equilibrium,
and `CONNECTED_VOID` transition belong to one transaction.

Downstream nucleation evaluates the production cleavage law from a tensor on
the explicit cavity boundary. No child branch exists in accepted state before
threshold crossing. Child creation and the first sharp segment commit together.
Zero cleavage drive creates no branch. The void-birth barrier does not enter
either cleavage event.

The analytical crack-tip renewal radius `r_tip` belongs to a sharp-front process
state and is initialized only after a sharp segment exists. The geometric cavity
radius `R_void` belongs to `Cavity2D`. **`r_tip != R_void`**, and V5 defines no
combined effective radius.

## Length accounting

V5 records separate cumulative SI-metre ledgers for fractured ligament,
ordinary newly fractured crack, pre-existing void free span, active-front
coordinate advance, their projected counterparts, and connected free-surface
extent. Front-coordinate traversal of pre-existing void space is never counted
as newly fractured material. Cavity area and defect inventory remain separate
SI-square-metre ledgers.

## Parameter registry

| Symbol | Code field | SI units | Range | Role | Source | Calibrated |
|---|---|---:|---|---|---|---|
| `N_hit` | `required_birth_hits` | 1 | integer >=1 | multi-hit completion | diagnostic | no |
| `nu` | `attempt_frequency_s` | s^-1 | >0 | all Arrhenius rates | material-dependent | no |
| `Q_b` | `birth_barrier_J` | J | >=0 | birth rate | material-dependent | no |
| `Q_s` | `stabilization_barrier_J` | J | >=0 | stabilization | material-dependent | no |
| `Q_h` | `healing_barrier_J` | J | >=0 | healing | material-dependent | no |
| `A_act` | `activation_area_m2` | m^2 | >=0 | stress work | material-dependent | no |
| `c_h` | `hydrostatic_work_coefficient` | 1 | signed | birth work | diagnostic | no |
| `c_n` | `normal_opening_work_coefficient` | 1 | signed | birth work | diagnostic | no |
| `c_s` | `signed_shear_work_coefficient` | 1 | signed | birth work | diagnostic | no |
| `Q_surface` | `surface_reaction_barrier_J` | J | >=0 | surface rate | material-dependent | no |
| `Q_diff` | `vacancy_transport_barrier_J` | J | >=0 | diffusion rate | material-dependent | no |
| `Q_plast` | `plastic_accommodation_barrier_J` | J | >=0 | accommodation rate | material-dependent | no |
| `l_g` | `radial_growth_scale_m` | m | >0 | radial growth | diagnostic | no |
| `R_promote` | `promotion_radius_m` | m | >0 | promotion | numerical/model | no |
| `w_site` | `candidate_weight` | 1 | [0,1] | birth only | diagnostic | no |
| `sigma` | solver tensor | Pa | finite symmetric | all local work | solver-derived | n/a |
| `Theta` | hazard threshold | 1 | >0 | first passage | RNG-derived | n/a |

## Explicit exclusions

V5 contains no AT1, AT2, GTN, diffuse void order parameter, scalar resolved-
void damage law, global area-loss law, generic MZ memory, or Prony memory. It
does not modify the four-class canonical material registry. Multiple voids,
three-dimensional geometry, fatigue calibration, material calibration, and
experimental validation are deferred.
