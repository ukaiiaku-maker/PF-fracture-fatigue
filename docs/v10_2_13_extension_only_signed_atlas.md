# v10.2.13 frozen-geometry signed shielding atlas

## Why v10.2.12 was revised

The 300, 700, and 1200 K discovery trajectories showed that local opening and
cumulative crack-path extension are strongly correlated.  They do not provide
an independently populated Cartesian opening/extension state grid.  For the
sharp-front atlas backend, the fixed-crack FEM is linear elastic and the
analytical `r_eff` does not alter the FEM geometry.  At a fixed geometry and
fixed internal state, a unit signed-slip influence coefficient should therefore
be independent of external displacement.

v10.2.13 treats opening as a validation coordinate.  The production kernel is
allowed to depend on cumulative crack-path extension only after an explicit
frozen-geometry load-invariance test passes.

## State semantics

The authoritative production coordinate is:

```text
cumulative_crack_path_extension_m
```

The legacy serialized field `crack_extension_m` is retained as a compatibility
alias and has the same cumulative path-length meaning.  It is not projected
x-extension.  Discovery traces also record:

```text
projected_crack_extension_m
projected_x_extension_m
opening_strength_fraction
observed_analytical_r_eff_over_r0
```

Opening and analytical radius remain diagnostics and are not used for kernel
interpolation.

## Production mesh gate

Trajectory discovery may use a coarse mesh.  Snapshot capture fails unless:

```text
FrontConfig.L_pz / mesh.hbar_tip >= MIN_ELEMENTS_PER_PZ
```

The default is `MIN_ELEMENTS_PER_PZ=3`.  The reduced MPZ domain length is not a
physical FEM process-zone resolution and is never used by this gate.

## Workflow

1. Run discovery and select representative cumulative path extensions.
2. Capture those extensions on a production-quality mesh.
3. For each frozen snapshot, evaluate at least three external load scales,
   including 1.0, using positive and negative perturbations at two or more
   magnitudes.
4. Require signed/amplitude linearity and less than the reviewed load-invariance
   tolerance, initially 5 percent.
5. Use only the `load_scale=1` measured-station response from each passing load
   sweep to build the extension-only atlas.
6. Derive activation-to-line conversion and source-capacity bounds from the
   process-zone/source geometry; do not fit them to toughness or fatigue.
7. Validate spatial projection, contour, mesh, ribbon width, exact replay, and
   the common monotonic/fatigue entry.
8. Complete independent review before setting production authorization.

## Example extension-only state table

```csv
state_id,temperature_K,cumulative_crack_path_extension_m,extension_tolerance_m,interaction_ell_m
E000,700,0.0,2.5e-6,2.0e-6
E200,700,2.0e-4,2.5e-6,2.0e-6
E500,700,5.0e-4,2.5e-6,2.0e-6
E800,700,8.0e-4,2.5e-6,2.0e-6
```

The interaction-integral length and extension tolerances must be reviewed for
the actual mesh and contour-convergence study; the values above are a table
format example, not an automatic physical authorization.

## Shared production implementation

Both monotonic and fatigue loading use:

```bash
python -m arrhenius_fracture.sharp_front_v10_2_13 \
  --signed-kernel-family path/to/v10_2_13_atlas.json \
  ...
```

Adding `--fatigue-cycles` changes the loading history only.  The signed
population state, source normalization, shielding family, transport, recovery,
and constitutive engine are identical.

No shielding cap, fitted attenuation coefficient, unsigned fallback, or
opening interpolation fallback is permitted.
