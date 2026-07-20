# v10.2.15 Stage 3 final 2-D source-lifecycle audit

## Conclusion

Stage 3 now executes the final accepted 2-D model directly through
`arrhenius_fracture.sharp_front_v10_1_7_5`. The only model inputs changed by the
Stage 3 wrapper are:

- the selected exact v9.11.1 material-manifest row;
- the option-specific MPZ length and number of bins.

The Stage 3 execution path does not install the signed-atlas engine, does not
replace the source lifecycle, and does not reinterpret the promoted source
parameter through a geometric source-packing model.

## Source-state semantics in the final 2-D model

The final 2-D implementation does not hold the instantaneous source activity
fixed and does not create a spatial inventory of discrete source positions.
`source_sites_per_system` supplies the promoted reference continuum hazard
budget `S0` for each crystallographic channel. The available activity `S(t)`
evolves during the calculation.

For each system, initialization sets

```
S0 = manifest.source_sites_per_system
S(0) = S0
```

At fixed geometry, the tensor-resolved emission law is

```
sigma_emit,s = max(f_s sigma_open - sigma_back,s, 0)
lambda_s = emission_rate_per_site(sigma_emit,s, T)
p_s = 1 - exp(-lambda_s dt)
dA_s = S_s p_s
S_s <- S_s - dA_s
```

Thus `S0` is prescribed by the promoted parameterization, but the active source
state is not held at that value. It is depleted by the evolving Arrhenius
emission process.

## Coupling to interactions ahead of the tip

The local Taylor back stress is calculated from the mobile-plus-retained
population near and ahead of the crack tip:

```
rho_tip,s <- weighted mobile_s + retained_s field
sigma_back,s <- G b sqrt(rho_tip,s) / resolved_fraction
```

Emitted population is inserted into the source-region bins and evolves through
the existing Peierls--Taylor transport, trapping/release, recovery, and
moving-frame operators. The resulting field feeds back into subsequent
emission through the local back stress.

This is the interaction-driven 2-D evolution described by the project owner:
the registry supplies the reference kinetic parameters, while the available
source activity and the distributed MPZ state are computed during the run.

## Crack-advance renewal

The final accepted model has no stationary temporal source recycling. When the
crack advances by `da`, fresh geometry restores part of the depleted budget:

```
f_refresh = 1 - exp(-da / L_refresh)
S <- S + (S0 - S) f_refresh
```

The active mobile, retained, and accumulated-slip fields are translated through
the existing moving-frame implementation. Stage 3 retains the requested
no-wake-shielding and zero-mobile-shield-fraction configuration without
replacing the underlying 2-D model.

## Verified unchanged production modules

Relative to the pre-Stage-3 base branch, the following production files retain
their original Git blob hashes and are not modified by Stage 3:

- `arrhenius_fracture/sharp_front_v10_1_7_5.py`;
- `arrhenius_fracture/anisotropic_emission_v10174.py`;
- `arrhenius_fracture/campaign_calibrated_tip.py`.

The wrapper records `parameter_overlay_only=true` and explicitly records that
the tip engine, source lifecycle, transport operator, shielding law, and
geometry backend were not replaced.

## Zero-event output handling

A calculation can legitimately complete with no cleavage event. The legacy
avalanche-summary adapter previously converted an empty geometry-event list into
a postprocessing exception even after the 2-D solver completed normally.
v10.2.15 records zero events explicitly so such cases can be classified as
right-censored. This changes output handling only; it does not alter the 2-D
physics or event evolution.
