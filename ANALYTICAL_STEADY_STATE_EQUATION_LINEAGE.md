# v10.2.30 analytical steady-state equation lineage

This document records the code lineage used by the analysis-only A0/A1/A2
hierarchy. Current source is authoritative; older manuscript equations are
included only when their constitutive assumptions remain active.

## Common waveform and activated rates

The cyclic load is

\[
K(\phi)=K_{\max}[(1+R)/2+(1-R)\cos\phi/2],
\]

with negative values clipped to zero. Source: `fatigue_v1.py:190-215`.
Units are Pa√m and seconds. The analysis uses midpoint quadrature with 4096
phases; it does not replace the integral by the peak rate.

Every active barrier uses

\[
G=G_f+(G_0-G_f)\exp[-a(\sigma/\sigma_c)^n],\qquad
\lambda=\nu_0\exp[-G/(k_BT)].
\]

Source: `material_manifest.py:32-64`. Transport surfaces inherit the emission
stress scale and floor semantics while replacing height, entropy, shape, and
attempt frequency (`material_manifest.py:67-90`). Energies are eV, stresses Pa,
temperature K, and rates s⁻¹.

## A0: opening only

The reference stress and cooperative opening rate are

\[
\sigma_c=\min\{[K-K_{shield}]_+/(2\pi r)^{1/2},\sigma_{cap}\},
\quad
\Lambda_c=P(m,\lambda_{raw}\tau)/\tau.
\]

Sources: `unified_front.py:78-89` and `unified_front.py:106-115`. A0 sets
`r=r0` and `K_shield=0`. The expected rate is

\[
g_0=\bar\ell(2\pi f)^{-1}\int_0^{2\pi}\Lambda_c\,d\phi.
\]

The prospective mean event length is 5 µm. The stochastic factor is explicitly
bounded and mean preserving (`stochastic_avalanche_tip.py:50-73`), and the
physical checkpoint multiplies that factor (`stochastic_avalanche_tip.py:163-175`).
The post-passage energy gate can truncate individual events; no numerical event
sizes or da/dN values are used to alter the prospective mean.

## A1: persistent emission and blunting moment

Production source multiplicity is

\[
M_s=\rho_{site,0}c_{arc}r_{eff}w_{eff}.
\]

Source: `persistent_site_source_v10221.py:190-223`. Emission is evaluated after
anisotropic drive and local Taylor backstress and is integrated by the
backstress-limited implicit activation law
(`persistent_site_source_v10221.py:244-328`). Sites remain persistent:
`available_sites` is restored to capacity and neither inventory depletion nor
source refresh is active (`persistent_site_source_v10221.py:350-357`).

The active blunting moment is net emitted minus physically returned source slip,
exponentially weighted over the current blunting length:

\[
q=\sum_{s,j}(N^{emit}_{s,j}-N^{return}_{s,j})e^{-x_j/L_q},
\qquad r_{eff}=r_0+c_{blunt}bq.
\]

Source: `persistent_site_reversible_transport_v10230.py:49-77`. A1 reduces the
spatial moving-frame loss to its exact first moment, `g q/Lq`, and solves

\[
q=\mu_{emit}(q)L_q/g(q)
\]

by a bracketed algebraic root. This is the principal steady-state assumption;
the production solver instead retains the full within-renewal spatial transient.

## A2: Peierls/Taylor population moments

Production evaluates signed Peierls velocity, forest encounter, mobile-retained
exchange, Taylor release, boundary escape, and physical surface return in
`persistent_site_reversible_transport_v10230.py:265-377`. The analysis takes
phase averages of those same rates and solves

\[
0=R_e-(k_{enc}+k_{esc}+k_{adv})N_m+k_TN_r,
\]

\[
0=k_{enc}N_m-(k_T+k_{r,rec}+k_{adv})N_r.
\]

Here `k_adv=fg/L_MPZ`; populations are counts and every k is s⁻¹. The closed
2×2 solution is used without fitting. In the stationary signed mean, the two
symmetry-paired retained channels cancel in the linear K projection, so A2 uses
zero mean shielding. Archived nonzero shielding and the PT08 negative-R return
fraction are validation residuals, not inferred coefficients.

## Explicitly rejected historical equations

The supplement's finite source-capacity and crack-advance source-refresh fixed
point is not used. It belongs to an older architecture and conflicts with the
qualified persistent-site implementation. Stored-energy lowering of cleavage,
mobile cleavage shielding, scalar saturation, arbitrary recovery, and the old
harmonic PT evolution rate are likewise inactive. This exclusion is recorded in
`analytical_input_manifest.json` before numerical da/dN is loaded.

## Domain and qualification

Predictions are admitted only for Kmax = 12–24.3 MPa√m and R in {-0.95, 0.1,
0.5}. Only archived trajectories that passed the production developed-growth
gate enter aggregate errors. Explicit/accelerated parity paths remain inventory
duplicates, and evolving constant-load trajectories remain transient validation
cases.
