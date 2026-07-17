# v10.1.4 separated tip-stress channels

This patch separates three stresses that were previously conflated in the
moving-tip continuum source path.

## Opening stress

The FEM/J-derived opening stress is

```text
sigma_opening = K_J / sqrt(2*pi*r_tip)
```

subject only to the existing lattice stress cap. It is not reduced by the local
Taylor back stress or by the scalar K-shielding projection. This is the stress
reported as `sigma_tip` in the console and legacy step CSV.

## Cleavage stress

The cleavage clock retains the existing elastic shielding channel:

```text
sigma_cleave = max(K_J - K_shield, 0) / sqrt(2*pi*r_tip)
```

This is recorded separately as `sigma_cleave_eff_Pa`. The patch does not remove
plastic shielding from the cleavage calculation.

## Emission stress

Peierls--Taylor plastic evolution receives `sigma_opening`. The continuum tip
source then subtracts only its local Taylor back stress:

```text
sigma_emit = max(sigma_opening - sigma_back_local, 0)
```

The local back stress therefore affects only dislocation emission. It does not
modify the FEM opening stress or the cleavage stress directly.

## Diagnostics

Kinetic audit records include:

- `sigma_tip_Pa` and `sigma_opening_tip_Pa`;
- `sigma_cleave_eff_Pa`;
- `sigma_emission_backstress_Pa`;
- `sigma_emission_effective_Pa`;
- `stress_channels_separated`.

The v10.1.4 source gate fails if these fields are absent, non-finite, or if the
reported tip stress differs from the opening stress.
