# Native Taylor–Peierls equation-to-consumer map

Status: current qualification baseline, 13 October 2026. This map describes the
constructed and restarted runtime, not the historical one-second reproduction.

## Source identity

- Authoritative Taylor donor: repository
  `PF-fracture-fatigue_worktrees/v3-fatigue-informed-joint-barrier-search`, commit
  `f75194852107e4cf39b156bca1bb7789cf02d76f`, file
  `arrhenius_fracture/taylor_release_v4_1_v10230.py`, SHA-256
  `cb55ab11fc1cd320cf7d477397ec135883b1cf06a015da5881a5b8e61d25b096`.
- Companion donor: the same commit,
  `arrhenius_fracture/mechanism_specific_feedback_v10230.py`, SHA-256
  `db24c84021a8be684c23e452ba4eebf259f96c527b924f4379df55affa98c35c`.
- The donor's finite `taylor_phi_max` expression is not active here. The current
  consumer uses the intended uncapped successor while retaining the later signed,
  persistent-source, spatial, restart, and F4 repairs.
- Current source files are `arrhenius_fracture/taylor_release.py`,
  `arrhenius_fracture/unified_mpz.py`, and
  `arrhenius_fracture/coupled_transport_exchange.py`. Runtime receipts store file
  and callable-body hashes before and after fresh-process restart.

## Equations and consumers

| Quantity | Executed equation and units | Active consumer after construction and restart |
|---|---|---|
| Peierls/Taylor barriers | `DeltaG(stress,T)` is the selected EXP-floor surface in eV. `G0(T)`, `sigc(T)`, the nonlinear stress attenuation, and the selected floor are evaluated without an added rate or velocity cap. | `MaterialManifest.peierls/taylor.as_surface(...).values_eV/rate`, called by `UnifiedMPZState._transport_rates`. |
| Signed elementary rates | `lambda_+ = nu_+ exp(-DeltaG(max(tau,0),T)/(k_B T))`; `lambda_- = nu_- exp(-DeltaG(max(-tau,0),T)/(k_B T))`; `lambda_net=lambda_+-lambda_-`, in s^-1. Zero bias cancels and stress reversal changes the sign. | `taylor_release.signed_arrhenius_components`, called independently for Peierls and Taylor channels by `_transport_rates`. |
| Stress projections | `tau_P=f_P sigma_local`; `tau_T=f_T sigma_local phi`, in Pa, where the manifest/config fractions are applied once. | `UnifiedMPZState._transport_rates`; its output is consumed directly by `_coupled_persistent_source_transport`. |
| Peierls event displacement | `ell_P=f_jump/(2 sqrt(rho_f))`, in m. It is a forest-spacing event length, not one Burgers vector, a cell width, a crack velocity, or a fitted escape length. | `_transport_rates` field `jump`; independent of mesh spacing. |
| Peierls glide velocity | `v_P=ell_P(lambda_P,+-lambda_P,-)`, in m s^-1. | `_transport_rates` field `velocity`; the full per-bin field is passed to `joint_transport_exchange`. The scalar velocity in the trace is diagnostic averaging only. |
| Taylor cooperative order | `m_T=1+s_m sqrt(rho_f/rho_c)`, dimensionless, using the selected manifest scale and density. No order clamp is applied. | `taylor_release.density_completion_order`, called by `_transport_rates`. |
| Taylor completion | `k_T=max(lambda_T,+-lambda_T,-,0)/m_T`, in s^-1. There is no correlation window, implicit second, gamma map, or single-rate cap. | `taylor_release.completion_rate`; runtime selection is `mean_net_rate_over_density_order`, `window_s=null`, `hard_cap=null`. |
| Geometric encounter | `k_enc=eta |v_P| sqrt(rho_f)`, in s^-1. | `_transport_rates`; passed with `k_T` and `v_P(x)` into the coupled generator. |
| Spatial flux | The first-order finite-volume face transfer rate is `(v_P/dx) mobile`; units are population s^-1. The physical velocity and jump length do not depend on `dx`. | `joint_transport_exchange`; complete per-system/bin face flux is exported in the maximum and terminal traces. |
| Coupled field update | `dx/dt=Q(x,state)x+s(state)` with mobile/retained exchange, spatial transport, escape/recovery, continuous persistent source, and diagnostic integrals in one fixed-coefficient subproblem. The nonlinear outer iteration updates stress, density, source/backstress, multiplicity, and coefficients. | `_coupled_persistent_source_transport` calls `joint_transport_exchange`; `persistent_binding_receipt` verifies these bindings before and after every qualification step and after reconstruction from checkpoint. |

## Numerical composition

For one frozen nonlinear iterate, `F(x)=P x+g`, with `P=exp(Qh)` and the exact
continuous-source integral `g`. Binary affine composition evaluates repeated
bounded substeps of that same autonomous generator. It changes multiplication
order only; it does not alter rates or remove state. A conserved-total similarity
coordinate conditions the mobile/retained block. The production path never
projects populations to local equilibrium. A BDF integration of the same full
augmented generator is retained only as an independent qualification oracle.

The compact evidence is in
`integration_evidence/native_taylor_peierls_spatial_e5982762`. The maximum and
terminal traces contain local stress, temperature, density, event length,
forward/reverse barriers and prefactors, forward/reverse/net rates, velocity,
face flux, completion order, residuals, and diagnostic averaging.
