# v10.2.10 physical mechanics artifact workflow

v10.2.10 separates four operations that must not be conflated:

1. define the mechanical state envelope;
2. collect signed interaction-integral and tensor-probe responses from the physical FEM;
3. build review-only kernel and drive families;
4. authorize those artifacts only after an independent mechanics review.

The workflow never creates synthetic response values and never turns on
`production_parameterization_allowed` during planning or review builds.

## State request table

Create a CSV with these columns:

```text
state_id,r_eff_over_r0,opening_strength_fraction,crack_extension_m,engine_template
```

The production coverage gate requires at least two distinct effective-radius
ratios, three opening-strength fractions, and two crack extensions. Each engine
template must be a complete JSON object containing `front_config`, `mpz_config`,
`tip_config`, and `anisotropic_config`.

## Write the physical collection plan

```bash
MODE=plan \
STATE_TABLE=/absolute/path/to/mechanical_states.csv \
OUTROOT=runs/v10_2_10_physical_artifact_plan_v1 \
ACTIVE_BINS=200 \
WAKE_BINS=200 \
N_SYSTEMS=2 \
PERTURBATION_MAGNITUDES="0.25 0.50" \
bash scripts/run_v10_2_10_physical_artifact_workflow.sh
```

The output contains:

- `mechanical_state_jobs.csv`, with every state, slip system, active/wake bin,
  Burgers sign, and perturbation magnitude required by the physical FEM run;
- a header-only signed interaction-integral response table;
- a header-only tensor-probe response table;
- `physical_artifact_plan.json`, which records that no physical responses have
  yet been generated.

Header-only response templates are intentional. They prevent a planning run from
being mistaken for physical data.

## Preflight measured response tables

```bash
MODE=preflight \
SIGNED_RESPONSES=/absolute/path/to/signed_interaction_integral_responses.csv \
TENSOR_RESPONSES=/absolute/path/to/tensor_probe_responses.csv \
NORMALIZATION=/absolute/path/to/source_normalization.json \
OUT=runs/v10_2_10_raw_preflight_v1.json \
bash scripts/run_v10_2_10_physical_artifact_workflow.sh
```

The state sets and state coordinates must match exactly between the two response
tables. The normalization must be mechanically derived and must not be fitted to
toughness or fatigue output.

## Build review-only artifacts

```bash
MODE=build-review \
SIGNED_RESPONSES=/absolute/path/to/signed_interaction_integral_responses.csv \
TENSOR_RESPONSES=/absolute/path/to/tensor_probe_responses.csv \
NORMALIZATION=/absolute/path/to/source_normalization.json \
OUTROOT=runs/v10_2_10_mechanics_review_build_v1 \
bash scripts/run_v10_2_10_physical_artifact_workflow.sh
```

Both output families remain unauthorized. Review the builder diagnostics,
linearity checks, state variation, contour stability, mesh/ribbon convergence,
source normalization, tensor repeatability, and replay results.

## Independent review record

Authorization requires a JSON record with schema
`v10.2.10_independent_mechanics_review`, reviewer and UTC timestamp fields, and
all eight review checks set to `true`. A template is supplied at
`docs/v10_2_10_independent_review_template.json`.

## Build authorized artifacts

```bash
MODE=authorize \
SIGNED_RESPONSES=/absolute/path/to/signed_interaction_integral_responses.csv \
TENSOR_RESPONSES=/absolute/path/to/tensor_probe_responses.csv \
NORMALIZATION=/absolute/path/to/source_normalization.json \
INDEPENDENT_REVIEW=/absolute/path/to/completed_independent_review.json \
OUTROOT=runs/v10_2_10_mechanics_authorized_v1 \
bash scripts/run_v10_2_10_physical_artifact_workflow.sh
```

Authorization is rejected if any review check is absent or false.

## Parameterization readiness

```bash
MODE=readiness \
KERNEL_FAMILY=/absolute/path/to/state_resolved_signed_kernel_family_authorized.json \
DRIVE_FAMILY=/absolute/path/to/state_resolved_signed_drive_family_authorized.json \
ENGINE_TEMPLATE=/absolute/path/to/v10_2_3_2d_engine_config.json \
OUT=runs/v10_2_10_parameterization_readiness_v1.json \
bash scripts/run_v10_2_10_physical_artifact_workflow.sh
```

This command exits nonzero unless both families are authorized and contain the
same mechanical state IDs.

## Corrected staged parameterization

After readiness passes, use
`scripts/run_v10_2_10_staged_parameterization.sh`. It retains the v10.2.9
quality-diversity policy but hard-reserves every stage passer when the passing
population is smaller than the promotion budget. Historical DBTT anchor lineage
protection is active only during analytical DBTT promotion.
