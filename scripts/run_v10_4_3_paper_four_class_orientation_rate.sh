#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

: "${OUTROOT:?OUTROOT must identify the materialized v10.4.3 campaign root}"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  command -v conda >/dev/null 2>&1 || {
    echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
    exit 2
  }
  exec conda run --no-capture-output -n "$CONDA_ENV" \
    env OUTROOT="$OUTROOT" \
    LOADING_RATE_FACTOR="${LOADING_RATE_FACTOR:-1}" \
    DU_M="${DU_M:-2e-7}" \
    BASE_DT_S="${BASE_DT_S:-8.4}" \
    MAX_JOBS="${MAX_JOBS:-2}" \
    CASE_FILTER_OPTION="${CASE_FILTER_OPTION:-}" \
    CASE_FILTER_TEMPERATURE="${CASE_FILTER_TEMPERATURE:-}" \
    PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
    PRINT_ONLY="${PRINT_ONLY:-0}" \
    bash "$0"
fi

LOADING_RATE_FACTOR=${LOADING_RATE_FACTOR:-1}
DU_M=${DU_M:-2e-7}
BASE_DT_S=${BASE_DT_S:-8.4}

RATE_FIELDS=$("$PYTHON_BIN" -m arrhenius_fracture.loading_rate_v10228 \
  --factor "$LOADING_RATE_FACTOR" \
  --dU-m "$DU_M" \
  --base-dt-s "$BASE_DT_S" \
  --format tsv)

IFS=$'\t' read -r LOADING_RATE_FACTOR DU_M BASE_DT_S DT_S \
  NOMINAL_OPENING_RATE_M_PER_S RATE_TAG <<< "$RATE_FIELDS"

export LOADING_RATE_FACTOR DU_M BASE_DT_S DT_S
export NOMINAL_OPENING_RATE_M_PER_S RATE_TAG
export OUTROOT
export CASE_FILTER_OPTION=${CASE_FILTER_OPTION:-}
export CASE_FILTER_TEMPERATURE=${CASE_FILTER_TEMPERATURE:-}
export MAX_JOBS=${MAX_JOBS:-2}

BASE="$ROOT/scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh"
GENERATED_BASE=$(mktemp "$ROOT/scripts/.v10_4_3_plastic_dominance.XXXXXX")
GENERATED="${GENERATED_BASE}.sh"
mv "$GENERATED_BASE" "$GENERATED"
trap 'rm -f "$GENERATED"' EXIT

"$PYTHON_BIN" scripts/build_v10_4_3_plastic_dominance_launcher.py \
  --source "$BASE" \
  --output "$GENERATED"
chmod +x "$GENERATED"
bash -n "$GENERATED"

grep -q 'sharp_front_v10_4_3_plastic_dominance_audited' "$GENERATED"
grep -q 'SKIP_REUSED_VERIFIED' "$GENERATED"
grep -q 'CASE_FILTER_OPTION' "$GENERATED"
grep -q -- '--plastic-flow-min-plastic-fraction 0.50' "$GENERATED"
grep -q -- '--plastic-flow-energy-balance-tolerance 0.01' "$GENERATED"

printf 'Model entry: %s\n' \
  'arrhenius_fracture.sharp_front_v10_4_3_plastic_dominance_audited'
printf 'Outcome competition: sharp-fracture first passage vs plastic-dominance censor\n'
printf 'Hazard-energy gate: active; no absolute athermal Gc\n'
printf 'Directional J: positive raw signed J; J_eff=max(J_signed,0)\n'
printf 'Bulk plasticity: full-field detailed-balance Peierls/Taylor kinetics\n'
printf 'Stagger time: every nonlinear iterate reintegrates one dt from step-start state\n'
printf 'Plastic work: final constitutive iterate; diagnostic only\n'
printf 'Plastic-dominance threshold: median Phi_p >= 0.50\n'
printf 'Energy-balance relative tolerance: 0.01\n'
printf 'Terminal meaning: model-limit censor; not simulated ductile fracture\n'
printf 'Output root: %s\n' "$OUTROOT"
printf 'Execution filter option: %s\n' "${CASE_FILTER_OPTION:-<all>}"
printf 'Execution filter temperature: %s\n' "${CASE_FILTER_TEMPERATURE:-<all>}"
printf 'Loading-rate factor: %s\n' "$LOADING_RATE_FACTOR"
printf 'Nominal dU: %s m\n' "$DU_M"
printf 'Nominal dt: %s s\n' "$DT_S"
printf 'Opening rate: %s m/s\n' "$NOMINAL_OPENING_RATE_M_PER_S"
printf 'Rate tag: %s\n' "$RATE_TAG"

if [[ "${PRINT_ONLY:-0}" == 1 ]]; then
  cat "$GENERATED"
  exit 0
fi

bash "$GENERATED"

if [[ "${PREFLIGHT_ONLY:-0}" != 1 \
  && -z "${CASE_FILTER_OPTION:-}" \
  && -z "${CASE_FILTER_TEMPERATURE:-}" ]]; then
  "$PYTHON_BIN" scripts/plot_v10_4_2_fracture_plastic_temperature.py \
    --outroot "$OUTROOT"
else
  echo "Filtered/preflight execution: skipping full-matrix temperature plot"
fi
