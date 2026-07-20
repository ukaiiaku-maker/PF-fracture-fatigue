# v10.2.15 Stage 3 final 2-D source-lifecycle audit

## Conclusion

The final accepted 2-D implementation does not hold a literal source-site count
fixed and does not create a spatial inventory of discrete source positions.
`source_sites_per_system` is the promoted reference continuum hazard budget
`S0` for each crystallographic channel. The active budget `S(t)` evolves during
the calculation.

The Stage 3 signed-kernel overlay preserves that evolving source-budget law. It
adds Burgers sign and a mechanically derived activation-to-line-content
conversion; it does not replace the source evolution with a static population.

The 10--100 b source-position range in the mechanics-normalization artifact is
therefore audit metadata for a different geometric interpretation and must not
be used to accept or reject the promoted continuum budget `S0`.

## Executed class and wrapper chain

The Stage 3 entry executes the protected v10.1.7.5/v10.1.7.4/v10.1.7.3 path,
which is rooted in `sharp_front_v10_1_5.py`.

`sharp_front_v10_1_5.py` replaces the parser's nominal `continuum` source class
with `CampaignCalibratedTipEngine`. Its persistent audit explicitly identifies:

- `source_sites_per_system` as the promoted initial continuum tip budget;
- no source recovery while the crack is stationary;
- source recovery only through crack advance over the promoted refresh length.

The v10.1.7.4 anisotropic layer then replaces the scalar emission update with a
two-channel tensor-driven update while retaining the campaign source budget.
The v10.2.5 signed layer replaces the unsigned population fields with positive
and negative Burgers species and evaluates the measured signed shielding
operator.

## Reference budget and active source state

For each system, initialization sets

```
S0 = manifest.source_sites_per_system
S(0) = S0
```

`S0` is a dimensionless promoted hazard multiplicity/budget. It is not a count
obtained by packing line sources at 10--100 b spacing.

At fixed geometry the anisotropic emission law is

```
sigma_emit,s = max(f_s sigma_open - sigma_back,s, 0)
lambda_s = emission_rate_per_site(sigma_emit,s, T)
p_s = 1 - exp(-lambda_s dt)
dA_s = S_s p_s
S_s <- S_s - dA_s
```

Thus the available source activity evolves continuously through the Arrhenius
hazard. It is not held at `S0`.

## Coupling to the field ahead of the tip

The local Taylor back stress is computed from the evolving mobile-plus-retained
population near and ahead of the crack tip:

```
rho_tip,s <- weighted mobile_s + retained_s field
sigma_back,s <- G b sqrt(rho_tip,s) / resolved_fraction
```

Emitted population is inserted into the source-region bins and then evolves by
the Peierls--Taylor transport, trapping/release, recovery, and moving-frame
operators. These fields feed back into subsequent emission through the local
back stress.

This is the interaction-driven 2-D state evolution. The prescribed registry
quantity is only the reference budget `S0` and the other promoted kinetic
parameters; the instantaneous source activity and distributed MPZ state are
computed during the run.

## Crack-advance renewal

The accepted campaign law has no stationary temporal source recycling. When the
crack advances by `da`, fresh geometry restores part of the depleted budget:

```
f_refresh = 1 - exp(-da / L_refresh)
S <- S + (S0 - S) f_refresh
```

The mobile, retained, and accumulated-slip fields are translated in the moving
frame. In the signed implementation, positive and negative populations are
translated separately and the crossed population is transferred to the wake.
Stage 3 disables wake shielding because no measured signed 2-D wake operator is
available, while retaining the wake state for transport/bookkeeping.

## Signed-overlay equivalence

The v10.2.5/v10.2.14 signed path preserves the accepted v10.1.7.4 source law:

- same tensor-derived channel factors inside `sigma_emit`;
- same campaign local-density/Taylor-backstress calculation;
- same exact bounded depletion probability `1-exp(-lambda dt)`;
- same promoted `S0` and crack-advance refresh length;
- same Peierls--Taylor active-zone transport ordering;
- no post-hazard directional weighting;
- no stationary temporal source recycling.

The signed additions are:

- the sign of each activation comes from the resolved shear direction;
- positive and negative mobile/retained species are stored separately;
- each accepted activation is converted to signed line content using the
  mechanics-derived conversion;
- shielding is the signed line-content field contracted with the measured FEM
  kernel.

## Correct Stage 3 validation policy

Stage 3 must validate:

- finite, nonnegative promoted `S0` values from the exact registry rows;
- exact preservation of those registry values;
- evolving depletion and crack-advance refresh;
- mechanically derived activation-to-line-content conversion;
- runtime interpolation of the physical FEM kernel onto the selected 80- or
  200-bin MPZ grid;
- zero wake shielding and zero mobile shielding fraction.

Stage 3 must not compare `S0` against the geometric 10--100 b possible-position
interval. That interval is not the state variable used by the final 2-D source
law.

## Separate runtime-preflight issue

A one-step 2-D initialization may legitimately produce no cleavage event. The
legacy avalanche-summary adapter previously converted an empty event list into
an exception after the solver had completed normally. v10.2.15 treats that as a
valid zero-event/right-censored result while retaining strict checks for every
non-empty event list.
