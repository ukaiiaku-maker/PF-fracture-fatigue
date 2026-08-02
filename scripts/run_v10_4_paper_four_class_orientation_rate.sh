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

RATE_FIELDS=$("$PYTHON_BIN" -m arrhenius_fracture.loading_rate_v10228 \
  --factor "$LOADING_RATE_FACTOR" \
  --dU-m "$DU_M" \
  --base-dt-s "$BASE_DT_S" \
  --format tsv)

IFS=$'\t' read -r LOADING_RATE_FACTOR DU_M BASE_DT_S DT_S \
  NOMINAL_OPENING_RATE_M_PER_S RATE_TAG <<< "$RATE_FIELDS"

export LOADING_RATE_FACTOR DU_M BASE_DT_S DT_S
export NOMINAL_OPENING_RATE_M_PER_S RATE_TAG

BASE="$ROOT/scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh"
GENERATED_BASE=$(mktemp "$ROOT/scripts/.v10_4_2_bulk_terminal_rate_orientation.XXXXXX")
GENERATED="${GENERATED_BASE}.sh"
mv "$GENERATED_BASE" "$GENERATED"
trap 'rm -f "$GENERATED"' EXIT

"$PYTHON_BIN" scripts/build_v10_4_2_reuse_aware_launcher.py \
  --source "$BASE" \
  --output "$GENERATED"
chmod +x "$GENERATED"
bash -n "$GENERATED"

printf 'Model entry: %s\n' \
  'arrhenius_fracture.sharp_front_v10_4_2_plastic_flow_audited'
printf 'Hazard-energy gate: active; no absolute athermal Gc\n'
printf 'Directional J: positive raw signed J is forward work; J_eff=max(J_signed,0)\n'
printf 'First-nonzero directional-J sign latch: disabled\n'
printf 'Bulk plasticity: full-field emission-derived Peierls/Taylor multihit\n'
printf 'Bulk net slip: detailed balance; exact zero at zero stress\n'
printf 'Plastic-flow terminal: enabled; 2000 accepted-step persistence window\n'
printf 'Plastic dissipation measure: diagnostic only; excluded from fracture J and hazard\n'
printf 'Contour shielding: multi-contour elastic configurational-J diagnostic only\n'
printf 'Audited inherited cases: verify v10.4.2 reuse audit before native command checks\n'
printf 'Bulk source population: homogeneous persistent background\n'
printf 'Direct tip-to-bulk density transfer: disabled\n'
printf 'Loading-rate factor: %s\n' "$LOADING_RATE_FACTOR"
printf 'Nominal dU: %s m\n' "$DU_M"
printf 'Nominal dt: %s s\n' "$DT_S"
printf 'Opening rate: %s m/s\n' "$NOMINAL_OPENING_RATE_M_PER_S"
printf 'Rate tag: %s\n' "$RATE_TAG"

bash "$GENERATED"

if [[ "${PREFLIGHT_ONLY:-0}" != 1 ]]; then
  : "${OUTROOT:?OUTROOT must be set for v10.4.2 production plotting}"
  "$PYTHON_BIN" scripts/plot_v10_4_2_fracture_plastic_temperature.py \
    --outroot "$OUTROOT"
fi
