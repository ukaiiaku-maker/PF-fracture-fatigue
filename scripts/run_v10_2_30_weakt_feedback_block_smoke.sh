#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

CAP_CYCLES=${CAP_CYCLES:-1e4}
DELTA_K_MPA_SQRT_M=${DELTA_K_MPA_SQRT_M:-6.9866145600638339}
HAZARD_SEED=${HAZARD_SEED:-2001726}
PARAMETER_OPTION=${PARAMETER_OPTION:-v913_paper_weakT01_0129902_persistent_sites}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json}
TAG=$(printf '%g' "$CAP_CYCLES" | tr '+.' '__')
OUTROOT=${OUTROOT:-$ROOT/runs/v10_2_30_weakt_feedback_block_smoke_${TAG}_$(date +%Y%m%d_%H%M%S)}

[[ -s "$FAMILY_JSON" ]] || {
  echo "ERROR: missing FAMILY_JSON=$FAMILY_JSON" >&2
  exit 2
}
rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

PATCH_DIR="$ROOT/scripts/v10230_active_state_runtime"
export PYTHONPATH="$PATCH_DIR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PARAMETER_CAMPAIGN=1
export V10230_ACTIVE_STATE_BLOCK_CONTROL=1
export V10230_FEEDBACK_STATE_BLOCK_CONTROL=1
export V10230_VHCF_RELATIVE_CYCLE_TOL=${V10230_VHCF_RELATIVE_CYCLE_TOL:-1e-4}
export V10229_COUPLED_HAZARD_ABS_DB_TOL=0
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_HAZARD_SEED="$HAZARD_SEED"
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
export CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1
export KERNEL_STRICT_FAMILY_OVERRIDE=1
export SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON"
export PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
export V10230_ENERGY_GATE_ENABLED=1
export V10230_ENERGY_GATE_TRIAL_FRACTION=${V10230_ENERGY_GATE_TRIAL_FRACTION:-0.10}
export V10230_ENERGY_GATE_BISECTION_ITERATIONS=${V10230_ENERGY_GATE_BISECTION_ITERATIONS:-24}
export V10230_ENERGY_GATE_RELATIVE_TOL=${V10230_ENERGY_GATE_RELATIVE_TOL:-1e-8}
export V10230_ENERGY_GATE_ABSOLUTE_TOL_J_PER_M=${V10230_ENERGY_GATE_ABSOLUTE_TOL_J_PER_M:-1e-12}

"$PYTHON_BIN" - <<'PY'
from arrhenius_fracture import feedback_state_block_control_v10230 as feedback
feedback.install_feedback_state_block_control()
audit = feedback.audit_payload()
assert audit["installed"] is True
assert audit["raw_population_counts_are_block_limiters"] is False
print("v10.2.30 feedback-state block smoke verified")
print("  raw mobile/retained counts are not block limiters")
print("  low-hazard absolute dB bypass is disabled")
PY

START=$(date +%s)
set +e
"$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_30_fixed_deltaK \
  --signed-kernel-family "$FAMILY_JSON" \
  --mode 2d --temperatures 300 \
  --nx 36 --ny 72 --dt 8.4 --n-stagger 2 \
  --tip-h-fine 1e-6 --tip-ratio 1.20 \
  --da-phys 5e-6 --target-crack-extension-um 25 \
  --front-state-model moving_pz \
  --tip-source-model continuum \
  --tip-kinetics-mode moving_velocity \
  --bulk-plasticity-mode tip_only \
  --directional-j-mode root_signed \
  --tip-plasticity --active-shielding --signed-active-shielding \
  --mobile-shield-fraction 0 --no-wake-shielding \
  --crystal-aniso --crystal-compete --crystal-theta-deg 30 \
  --crystal-material w --j-decomposition cluster \
  --max-fronts 1 --crack-backend sharp_wake --dU 2e-7 \
  --fatigue-cycles --fatigue-hold-load --R 0.1 --frequency-Hz 1000 \
  --cycle-block-mode hazard_limited --min-block-cycles 1e-6 \
  --target-dB 0.10 \
  --target-dN-store 0.10 --target-dN-emit 0.10 \
  --target-dN-mobile 0.10 --target-dN-escape 0.10 \
  --target-dN-peierls 0.10 --target-dN-taylor 0.10 \
  --n-phase 48 --max-da-per-block-um 5 \
  --adaptive-events --adaptive-event-target 0.2 \
  --print-every 1 --save-snapshots 0 --no-plots \
  --parameter-option "$PARAMETER_OPTION" \
  --target-deltaK-MPa-sqrt-m "$DELTA_K_MPA_SQRT_M" \
  --steps 1 --cycles-max "$CAP_CYCLES" \
  --block-cycles "$CAP_CYCLES" --max-block-cycles "$CAP_CYCLES" \
  --out "$OUTROOT" \
  2>&1 | tee "$OUTROOT/run.log"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)

printf '%s\n' "$RC" > "$OUTROOT/exit_code.txt"
printf '%s\n' "$((END - START))" > "$OUTROOT/wall_seconds.txt"
printf '%s\n' "$CAP_CYCLES" > "$OUTROOT/requested_cap_cycles.txt"

echo "exit_code=$RC"
echo "wall_seconds=$((END - START))"
echo "requested_cap_cycles=$CAP_CYCLES"
echo "output=$OUTROOT"
exit "$RC"
