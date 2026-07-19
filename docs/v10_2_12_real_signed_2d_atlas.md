# v10.2.12 real signed 2-D shielding atlas

This workflow generates the physical closure required by the shared signed-Burgers fracture/fatigue engine. It does not fit a shielding attenuation factor and does not introduce a `K_shield` cap.

## Physical interpretation

The shared state stores positive and negative dislocation-line content separately. Unsigned content controls density-dependent transport, recovery, Taylor back stress, and blunting. Signed content controls shielding and antishielding through the measured interaction operator.

The PF/FEM crack geometry remains a sharp stiffness-kill front. The engine variable `r_eff` is an analytical local-tip stress/blunting state; it is not a finite-radius FEM crack geometry. Captured states may be indexed by `r_eff/r0` to test whether the 2-D influence operator is insensitive to that analytical state, but the resulting artifact must not be described as a finite-radius geometric kernel.

## 1. Create a state request table

Required columns:

```text
state_id,temperature_K,r_eff_over_r0,opening_strength_fraction,crack_extension_m,r_tolerance,opening_tolerance,extension_tolerance_m,interaction_ell_m
```

The table should cover the intended production envelope. A complete Cartesian state grid is required by the v10.2.9 family builder. Because one monotonic trajectory may not pass every requested combination, collect states in several versioned runs and combine their response tables later.

## 2. Capture accepted physical FEM states

Capture mode observes the production 2-D sharp-front FEM and tensor-probe path. It disables the inherited unsigned shielding operator during mechanics-data collection, because that operator is the closure being replaced. It does not alter the FEM equilibrium, crack geometry, transport, source kinetics, or local 30 GPa strength limit.

```bash
MODE=capture \
STATE_TABLE=/absolute/path/to/v10_2_12_states.csv \
ATLAS_OUTROOT=runs/v10_2_12_state_snapshots_v1 \
RUN_OUT=runs/v10_2_12_capture_mechanics_v1 \
MATERIAL_MANIFEST=/absolute/path/to/mechanics_control_manifest.csv \
TEMPERATURES="300 700 1200" \
THETA=45 \
EXTRA_ARGS="--steps 1200 --target-extension-um 50 --tip-h-fine 7e-7 --L-pz 1e-6" \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

Each captured state contains the mesh, connectivity, accepted displacement, plastic-strain and density fields, damage field, stiffness tensor, boundary sets, imposed displacement, local crack direction, slip-channel directions/normals, active/wake bin coordinates, material properties, and full shared-engine configuration.

Capture fails closed if a requested state is not reached. `--allow-incomplete-atlas-capture` is available only for diagnostic runs. Cohesive-network snapshots are not yet serialized; the collector therefore rejects a non-null cohesive network rather than silently dropping its state.

## 3. Evaluate positive and negative signed-slip perturbations

For every state, system, active/wake bin, Burgers sign, and perturbation magnitude:

```bash
MODE=evaluate \
SNAPSHOT=runs/v10_2_12_state_snapshots_v1/S000 \
OUT=runs/v10_2_12_responses_v1/S000.csv \
MAGNITUDES="0.25 0.50" \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

The ribbon plastic shear is fixed by

```text
gamma = signed_line_content * b / ribbon_width
```

so the imposed displacement discontinuity equals `signed_line_content * b`. The base and perturbed equilibria use fixed crack geometry and fixed external displacement. Signed mode-I and mode-II responses are extracted with the v10.2.9 analytic-gradient, C1-Hermite interaction integral.

At least two magnitudes and both Burgers signs are mandatory. The family builder rejects sign or amplitude dependence beyond the requested tolerance.

## 4. Build the mechanics-derived normalization

Use the complete engine configuration stored in a snapshot:

```bash
MODE=normalization \
ENGINE_CONFIG=runs/v10_2_12_state_snapshots_v1/S000/snapshot_engine_config.json \
OUT=runs/v10_2_12_source_normalization_v1.json \
MINIMUM_SPACING_B=10 \
MAXIMUM_SPACING_B=100 \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

If the snapshot directory does not contain a standalone engine JSON, extract the `engine_config` object from `snapshot.json` without changing its contents.

One accepted activation maps to

```text
activation_to_line_content = kinetic_packet_length / b
```

Source-capacity bounds are obtained from the process-zone source length divided by a reviewed admissible source-spacing interval. The plane-strain convention is one through-thickness line per in-plane source position; no arbitrary specimen-thickness multiplier is applied. Historical source counts are retained only as diagnostics and may lie outside the new bounds.

## 5. Build a review artifact

```bash
MODE=build-review \
RESPONSES="runs/v10_2_12_responses_v1/S000.csv runs/v10_2_12_responses_v1/S001.csv" \
NORMALIZATION=runs/v10_2_12_source_normalization_v1.json \
OUT=runs/v10_2_12_real_signed_atlas_review_v1.json \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

The output remains unauthorized. Review:

- positive/negative antisymmetry;
- multi-amplitude linearity;
- contour-radius stability;
- mesh and ribbon-width convergence;
- complete state-envelope coverage and boundary stationarity;
- source normalization and source-capacity geometry;
- exact reduced/2-D replay;
- monotonic-fracture and cyclic-fatigue validation using the same artifact.

## 6. Authorize only after independent review

Copy and complete `docs/v10_2_12_independent_real_signed_atlas_review_template.json`, then run:

```bash
MODE=authorize \
RESPONSES="runs/v10_2_12_responses_v1/S000.csv runs/v10_2_12_responses_v1/S001.csv" \
NORMALIZATION=runs/v10_2_12_source_normalization_v1.json \
INDEPENDENT_REVIEW=/absolute/path/to/completed_v10_2_12_review.json \
OUT=kernels/v10_2_12_real_signed_2d_atlas_authorized.json \
bash scripts/run_v10_2_12_real_signed_atlas.sh
```

Authorization fails if any physical, numerical, normalization, replay, or shared-loading-path review gate is incomplete.

## Required convergence matrix

At minimum, repeat representative states with:

- two FEM mesh resolutions;
- two slip-ribbon widths that remain resolved by both meshes;
- two or more interaction-integral contour choices;
- positive and negative perturbations;
- at least two perturbation magnitudes;
- low/intermediate/high opening states;
- at least two crack extensions;
- states spanning the attained analytical `r_eff/r0` range.

A fixed kernel is acceptable only if the measured variation remains below the reviewed tolerance. Otherwise retain the interpolated state-resolved family. No response may be clipped or renormalized to force that result.
