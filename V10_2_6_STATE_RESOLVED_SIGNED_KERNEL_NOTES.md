# v10.2.6 state-resolved signed-kernel generation

## Status

This branch implements the mechanical and software prerequisites identified after
v10.2.5. It does **not** contain a fabricated material kernel and it does not
restart the parameter campaign.

Production parameterization remains blocked until a real equilibrated 2-D state
atlas, signed unit-slip responses, and source-normalization artifact have passed
all gates below.

## Immediate corrections

### Signed interaction integral

`arrhenius_fracture.interaction_integral_v1026` extracts signed mode-I and mode-II
stress-intensity factors using a domain interaction integral and unit Williams
auxiliary fields. The implementation is plane-strain isotropic and fails closed
when a supplied stiffness matrix is not numerically isotropic. The tungsten
baseline with Zener ratio one is admissible; deliberately anisotropic branch-test
materials require a separate anisotropic interaction integral.

### Mechanically normalized perturbation

`arrhenius_fracture.unit_slip_perturbation_v1026` applies a finite-width signed
slip ribbon beginning at the crack/free-surface side and terminating at the target
spatial bin. The normalization is

```text
delta_N_line = gamma * ribbon_width / b
```

so a requested signed line content determines the plastic shear directly. This
avoids computing a Nye tensor from the existing symmetric `ep_gp` field, which
does not contain the complete plastic-distortion tensor.

The perturbation evaluator keeps crack geometry and external displacement fixed,
re-equilibrates the FEM, and records signed base/perturbed `K_I` and `K_II`.

### Multi-amplitude and sign validation

`scripts/build_v10_2_6_state_resolved_kernel_family.py` requires, for every
state/region/system/bin:

- positive and negative Burgers perturbations;
- at least two nonzero perturbation magnitudes for each sign;
- agreement of the normalized influence coefficient across signs;
- agreement across perturbation magnitudes.

A default 3% linearity tolerance is applied. This can be tightened after mesh and
perturbation convergence are established.

### State dependence

The family axes are

```text
r_eff_over_r0
opening_strength_fraction = sigma_local / sigma_cap
crack_extension_m
```

The builder requires at least two tip-radius states, three opening states, and two
crack-extension states by default. It calculates the maximum significant kernel
variation relative to a selected reference state.

If mode-I variation is at or below the fixed-kernel tolerance, the artifact may
use the reference kernel within the validated envelope. Otherwise it uses
inverse-distance interpolation. Extrapolation is forbidden.

### Shared monotonic/fatigue engine

Both loading paths use

```bash
python -m arrhenius_fracture.sharp_front_v10_2_6 \
  --signed-kernel-family path/to/family.json \
  ...
```

Adding `--fatigue-cycles` changes only the loading path. Every monotonic and
fatigue phase-point stress evaluation resolves the same kernel family through
`StateResolvedSignedBurgersTipEngine`.

### Parameter-campaign guard

The family builder sets

```json
"production_parameterization_allowed": false
```

unless `--authorize-production-parameterization` is supplied after independent
review. Even an authorized artifact must pass the default state-coverage
requirements. The v10.2.6 entry point refuses `PARAMETER_CAMPAIGN=1` when the
artifact is not authorized.

## Required response table

The family builder consumes a CSV with

```text
state_id
r_eff_over_r0
opening_strength_fraction
crack_extension_m
region
system
bin
x_m
burgers_sign
delta_signed_line_content
K_I_base_Pa_sqrt_m
K_I_perturbed_Pa_sqrt_m
K_II_base_Pa_sqrt_m
K_II_perturbed_Pa_sqrt_m
```

For each perturbation,

```text
H_I  = (K_I_base  - K_I_perturbed)  / delta_signed_line_content
H_II = (K_II_base - K_II_perturbed) / delta_signed_line_content
```

## Required normalization artifact

The normalization JSON must identify one mechanics-derived route:

- `2d_unit_slip_to_line_content`;
- `plastic_distortion_burgers_integral`;
- `process_zone_geometry_and_line_spacing`;
- `front_thickness_source_geometry`.

It must include

```json
{
  "normalization_source": "2d_unit_slip_to_line_content",
  "activation_to_line_content_by_system": [0.0, 0.0],
  "source_capacity_bounds_per_system": [[0.0, 0.0], [0.0, 0.0]],
  "fitted_to_toughness_or_fatigue": false
}
```

The zeros above are placeholders describing the schema only; they are rejected by
the loader and must be replaced by derived positive values.

## Build command

```bash
python scripts/build_v10_2_6_state_resolved_kernel_family.py \
  --responses path/to/state_resolved_signed_responses.csv \
  --normalization path/to/source_normalization.json \
  --out kernels/v10_2_6_state_resolved_signed_kernel_family.json
```

Do not add `--authorize-production-parameterization` during initial generation.
First inspect the linearity checks, state coverage, fixed-kernel assessment, and
normalization derivation.

## Validation sequence

1. Pass the complete regression suite.
2. Validate signed `K_I/K_II` against analytic Williams fields and contour changes.
3. Perform mesh and ribbon-width convergence.
4. Generate both signs at two or more magnitudes for every state/bin/channel.
5. Inspect amplitude linearity and Burgers antisymmetry.
6. Inspect the state-dependence atlas.
7. Validate one monotonic 2-D trace and exact replay.
8. Validate one cyclic trace through the same engine and family.
9. Only then authorize a new physically bounded parameter campaign.
