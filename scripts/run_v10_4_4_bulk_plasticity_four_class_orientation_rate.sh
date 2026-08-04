#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || {
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
}

LOADING_RATE_FACTOR=${LOADING_RATE_FACTOR:-1}
DU_M=${DU_M:-2e-7}
BASE_DT_S=${BASE_DT_S:-8.4}
PLASTIC_FLOW_WINDOW_STEPS=${PLASTIC_FLOW_WINDOW_STEPS:-32}
PLASTIC_FLOW_MIN_STEP=${PLASTIC_FLOW_MIN_STEP:-32}
PLASTIC_FLOW_MAX_DA_FRACTION=${PLASTIC_FLOW_MAX_DA_FRACTION:-0.1}
PLASTIC_FLOW_MIN_PLASTIC_FRACTION=${PLASTIC_FLOW_MIN_PLASTIC_FRACTION:-0.90}
PLASTIC_FLOW_MIN_CUMULATIVE_FRACTION=${PLASTIC_FLOW_MIN_CUMULATIVE_FRACTION:-0.90}
PLASTIC_FLOW_MAX_ELASTIC_FRACTION=${PLASTIC_FLOW_MAX_ELASTIC_FRACTION:-0.05}
PLASTIC_FLOW_MAX_TANGENT_FRACTION=${PLASTIC_FLOW_MAX_TANGENT_FRACTION:-0.05}

RATE_FIELDS=$("$PYTHON_BIN" -m arrhenius_fracture.loading_rate_v10228 \
  --factor "$LOADING_RATE_FACTOR" \
  --dU-m "$DU_M" \
  --base-dt-s "$BASE_DT_S" \
  --format tsv)

IFS=$'\t' read -r LOADING_RATE_FACTOR DU_M BASE_DT_S DT_S \
  NOMINAL_OPENING_RATE_M_PER_S RATE_TAG <<< "$RATE_FIELDS"

export LOADING_RATE_FACTOR DU_M BASE_DT_S DT_S
export NOMINAL_OPENING_RATE_M_PER_S RATE_TAG
export PLASTIC_FLOW_WINDOW_STEPS PLASTIC_FLOW_MIN_STEP
export PLASTIC_FLOW_MAX_DA_FRACTION
export PLASTIC_FLOW_MIN_PLASTIC_FRACTION
export PLASTIC_FLOW_MIN_CUMULATIVE_FRACTION
export PLASTIC_FLOW_MAX_ELASTIC_FRACTION
export PLASTIC_FLOW_MAX_TANGENT_FRACTION

BASE="$ROOT/scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh"
GENERATED_BASE=$(mktemp "$ROOT/scripts/.v10_4_9_bulk_plasticity_orientation.XXXXXX")
GENERATED="${GENERATED_BASE}.sh"
mv "$GENERATED_BASE" "$GENERATED"
trap 'rm -f "$GENERATED"' EXIT

"$PYTHON_BIN" scripts/build_v10_4_4_bulk_plasticity_orientation_launcher.py \
  --source "$BASE" \
  --output "$GENERATED"
chmod +x "$GENERATED"
bash -n "$GENERATED"

printf 'Launcher revision: %s\n' 'v10.4.9_exec_namespace_contract'
printf 'Model entry: %s\n' \
  'arrhenius_fracture.sharp_front_v10_4_8_numerical_failure_audited'
printf 'Generated patcher file context: supplied explicitly\n'
printf 'Bulk plasticity: full_field; tip plasticity retained\n'
printf 'Fracture law: unchanged Arrhenius first passage plus event-energy gate\n'
printf 'Valid case terminals: target fracture extension OR physically plasticity dominated\n'
printf 'Plasticity terminal window: %s nominal increments\n' \
  "$PLASTIC_FLOW_WINDOW_STEPS"
printf 'Severe-substep physical terminal: 128-step J/force plateau plus cumulative plastic fraction >= %s\n' \
  "$PLASTIC_FLOW_MIN_CUMULATIVE_FRACTION"
printf 'Severe-substep nonplastic plateau: fail fast as NUMERICAL_STAGNATION (exit 4)\n'
printf 'Minimum-trial-fraction fixed-point exhaustion: fail fast as NUMERICAL_FIXED_POINT_FAILURE (exit 5)\n'
printf 'Nonzero solver exits: exit_code.txt plus RUN_FAILED retained before scheduler continuation\n'
printf 'Positive cumulative Wp alone: insufficient\n'
printf 'Tiny-window energy fractions: diagnostic only for severe-substep physical terminal\n'
printf 'Projected future cleavage action: diagnostic only\n'
printf 'Loading-rate factor: %s\n' "$LOADING_RATE_FACTOR"
printf 'Nominal dU: %s m\n' "$DU_M"
printf 'Nominal dt: %s s\n' "$DT_S"
printf 'Opening rate: %s m/s\n' "$NOMINAL_OPENING_RATE_M_PER_S"
printf 'Rate tag: %s\n' "$RATE_TAG"

bash "$GENERATED"
