# v10.2.12 real signed 2-D shielding atlas

This workflow generates the physical shielding closure required by the shared signed-Burgers fracture/fatigue engine. It does not fit a shielding attenuation factor and does not introduce a `K_shield` cap.

## Physical interpretation

The shared state stores positive and negative dislocation-line content separately. Unsigned content controls density-dependent transport, recovery, Taylor back stress, and blunting. Signed content controls shielding and antishielding through the measured interaction operator.

The PF/FEM crack geometry remains a sharp stiffness-kill front. The engine variable `r_eff` is an analytical local-tip stress/blunting state; it is **not** a finite-radius FEM crack geometry. It may be used to select and audit captured trajectories, but it is not a physical interpolation axis of the 2-D shielding kernel.

For compatibility with the existing three-coordinate family schema, every built kernel state stores `r_eff_over_r0 = 1`. The production resolver replaces the runtime analytical radius by this constant before interpolation while retaining the actual runtime value in diagnostics. The physical atlas grid is therefore:

```text
opening_strength_fraction × crack_extension_m
```

The reduced MPZ grid is usually finer than the continuum FEM mesh. Direct FEM perturbations are evaluated only at distinct, mesh-resolved spatial stations. The full active/wake MPZ-grid operator is an audited piecewise-linear projection from those measured stations. Projected sub-element values are never labeled as direct FEM measurements.

## 1. Create a state request table

Required columns:

```text
state_id,temperature_K,r_eff_over_r0,opening_strength_fraction,crack_extension_m,r_tolerance,opening_tolerance,extension_tolerance_m,interaction_ell_m
```

`r_eff_over_r0` and `r_tolerance` select an observed trajectory state only. After physical responses are measured, the builder collapses that non-geometric coordinate to the constant compatibility value. The retained kernel states must form a complete Cartesian grid in opening fraction and crack extension. Do not keep two snapshots with the same opening/extension coordinates merely because their analytical `r_eff` values differ.

## 2. Capture accepted physical FEM states

Capture mode observes the production 2-D sharp-front FEM and tensor-probe path. It disables the inherited unsigned shielding operator during mechanics-data collection because that operator is the closure being replaced. It does not replace the FEM equilibrium solver, crack geometry, transport law, source kinetics, or local 30 GPa strength limit.

```bash
MODE=capture \
STATE_TABLE=/absolute/path/to/v10_2_12_states.csv \
ATLAS_OUTROOT=runs/v10_2_12_state_snapshots_v1 \
RUN_OUT=runs/v10_2_12_capture_mechanics_v1 \
MATERIAL_MANIFEST=/absolute/path/to/mechanics_control_manifest.csv \
TEMPERATURES="300 700 1200" \
THETA=45 \
EXTRA_ARGS="--steps 1200 --tip-h-fine 7e-7 --L-pz 1e-6" \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

The runner uses `arrhenius_fracture.sharp_front_v10_2_12_capture`. Each snapshot contains the mesh, connectivity, accepted displacement, plastic-strain and density fields, damage field, stiffness tensor, boundary sets, imposed displacement, local crack direction, signed slip-channel directions/normals, active/wake MPZ coordinates, material properties, and complete engine configuration.

Capture fails closed if a requested state is not reached. `--allow-incomplete-atlas-capture` is diagnostic only. Cohesive-network snapshots are rejected until cohesive internal state has an explicit serializer.

## 3. Measure signed responses at FEM-resolved stations

For each captured state:

```bash
MODE=evaluate \
SNAPSHOT=runs/v10_2_12_state_snapshots_v1/S000 \
OUT=runs/v10_2_12_responses_v1/S000.csv \
MAGNITUDES="0.25 0.50" \
MINIMUM_STATION_SPACING_M=2.0e-7 \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

If `MINIMUM_STATION_SPACING_M` is omitted, the evaluator uses the larger of twice the ribbon width and twice the tip-local FEM spacing. The first and last MPZ positions are always included. Production authorization requires at least three measured stations per state/region/system curve so leave-one-out spatial interpolation can be tested.

The ribbon plastic shear is fixed by

```text
gamma = signed_line_content * b / ribbon_width
```

so the imposed displacement discontinuity equals `signed_line_content * b`. Positive and negative Burgers signs use the exact serialized slip-direction convention. Active ribbons terminate in intact material ahead of the tip. Wake ribbons originate in intact crack-face-adjacent material rather than inside stiffness-killed crack elements. A ribbon with more than 5% damage-weighted represented area is rejected.

The base and perturbed equilibria use fixed crack geometry and fixed external displacement. Signed mode-I and mode-II responses use the v10.2.9 analytic-gradient, C1-Hermite interaction integral. Both Burgers signs and at least two perturbation magnitudes are mandatory. Every measured file must carry that exact reviewed interaction-integral schema.

## 4. Build the mechanics-derived normalization

The normalization command reads the captured `snapshot.json` directly:

```bash
MODE=normalization \
ENGINE_CONFIG=runs/v10_2_12_state_snapshots_v1/S000/snapshot.json \
OUT=runs/v10_2_12_source_normalization_v1.json \
MINIMUM_SPACING_B=10 \
MAXIMUM_SPACING_B=100 \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

One accepted activation maps to

```text
activation_to_line_content = kinetic_packet_length / b
```

Source-capacity bounds are obtained from the process-zone source length divided by a reviewed admissible source-spacing interval. The plane-strain convention is one through-thickness line per in-plane source position; no arbitrary specimen-thickness multiplier is applied. Historical source counts are retained only as diagnostics and may lie outside the new bounds.

## 5. Build an unauthorized review artifact

```bash
MODE=build-review \
RESPONSES="runs/v10_2_12_responses_v1/S000.csv runs/v10_2_12_responses_v1/S001.csv" \
NORMALIZATION=runs/v10_2_12_source_normalization_v1.json \
SPATIAL_CV_TOL=0.10 \
OUT=runs/v10_2_12_real_signed_atlas_review_v1.json \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

The builder validates the directly measured station rows:

- positive/negative antisymmetry after division by signed line content;
- multi-amplitude linearity;
- one exact interaction-integral schema;
- intact-material ribbon placement;
- endpoint coverage of each active/wake spatial grid;
- at least three measured stations per curve;
- leave-one-out error for the piecewise-linear spatial projection;
- one unique physical state per opening/extension coordinate.

It then projects the measured coefficients onto the complete MPZ grid, sets the compatibility radius coordinate to one, and applies the complete opening/extension grid and boundary-stationarity gates. The output remains unauthorized.

Independent review must also cover contour-radius stability, mesh/ribbon convergence, source normalization, exact reduced/2-D replay, and monotonic-fracture plus cyclic-fatigue validation using the same artifact.

## 6. Authorize only after independent review

Complete `docs/v10_2_12_independent_real_signed_atlas_review_template.json`, then run:

```bash
MODE=authorize \
RESPONSES="runs/v10_2_12_responses_v1/S000.csv runs/v10_2_12_responses_v1/S001.csv" \
NORMALIZATION=runs/v10_2_12_source_normalization_v1.json \
SPATIAL_CV_TOL=0.10 \
INDEPENDENT_REVIEW=/absolute/path/to/completed_v10_2_12_review.json \
OUT=kernels/v10_2_12_real_signed_2d_atlas_authorized.json \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

Authorization fails if any physical, numerical, spatial-projection, normalization, replay, or shared-loading-path gate is incomplete.

## 7. Use the same artifact for monotonic fracture and fatigue

Monotonic example:

```bash
python -m arrhenius_fracture.sharp_front_v10_2_12 \
  --signed-kernel-family kernels/v10_2_12_real_signed_2d_atlas_authorized.json \
  --mode 2d \
  --material-manifest /absolute/path/to/material.csv \
  --temperatures 300 700 1200 \
  --out runs/v10_2_12_monotonic_v1 \
  --crystal-aniso --crystal-theta-deg 45
```

Fatigue uses the same entry point, engine class, family loader, fixed-point solution, signed populations, source normalization, and atlas:

```bash
python -m arrhenius_fracture.sharp_front_v10_2_12 \
  --signed-kernel-family kernels/v10_2_12_real_signed_2d_atlas_authorized.json \
  --mode 2d --fatigue-cycles \
  --material-manifest /absolute/path/to/material.csv \
  --temperatures 700 \
  --out runs/v10_2_12_fatigue_v1 \
  --crystal-aniso --crystal-theta-deg 45
```

Setting `PARAMETER_CAMPAIGN=1` blocks either loading path unless the artifact explicitly authorizes production parameterization.

## Required convergence matrix

At minimum, repeat representative states with:

- two FEM mesh resolutions;
- two slip-ribbon widths resolved by both meshes;
- two or more interaction-integral contour choices;
- positive and negative perturbations;
- at least two perturbation magnitudes;
- at least three spatial stations per active/wake/system curve;
- low/intermediate/high opening states;
- at least two crack extensions;
- observed analytical `r_eff/r0` values reported as diagnostics, not treated as an interpolation dimension.

A fixed opening/extension kernel is acceptable only if measured state variation remains below the reviewed tolerance. Otherwise retain the interpolated opening/extension family. No response may be clipped or renormalized to force either result.
